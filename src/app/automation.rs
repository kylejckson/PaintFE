#[cfg(not(target_arch = "wasm32"))]
mod automation_state {
    use std::sync::{Arc, Mutex, mpsc};
    pub struct Pending {
        pub document: uuid::Uuid,
        pub layer: uuid::Uuid,
        pub revision: u64,
        pub action: crate::components::history::HistoryAction,
        pub receiver: mpsc::Receiver<Result<Vec<u8>, String>>,
        pub cancel: Arc<std::sync::atomic::AtomicBool>,
    }
    #[derive(Default)]
    pub struct Session {
        pub token: String,
        pub path: Option<std::path::PathBuf>,
        pub pending: Option<Pending>,
        pub job: String,
        #[cfg(not(target_os = "windows"))]
        pub scanned: f64,
        #[cfg(target_os = "windows")]
        pub pipe: Option<crate::services::automation::pipe::Server>,
    }
    impl Session {
        pub fn start(&mut self, path: std::path::PathBuf) -> Result<(), String> {
            let token = uuid::Uuid::new_v4().to_string();
            #[cfg(target_os = "windows")]
            let pipe = crate::services::automation::pipe::Server::start()?;
            #[cfg(target_os = "windows")]
            let discovery = serde_json::json!({"token":token,"pipe":pipe.endpoint});
            #[cfg(not(target_os = "windows"))]
            let discovery = serde_json::json!({"token":token});
            crate::services::plugins::atomic_write(
                &path.join("session.json"), discovery.to_string().as_bytes(),
            )?;
            self.path = Some(path);
            self.token = token;
            #[cfg(target_os = "windows")]
            { self.pipe = Some(pipe); }
            Ok(())
        }
    }
    impl Drop for Session {
        fn drop(&mut self) {
            if let Some(job) = &self.pending {
                job.cancel.store(true, std::sync::atomic::Ordering::Relaxed);
            }
            if let Some(path) = &self.path {
                let file = path.join("session.json");
                if crate::services::plugins::read_limited(&file, 4096)
                    .ok()
                    .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
                    .is_some_and(|v| v["token"].as_str() == Some(self.token.as_str()))
                {
                    let _ = std::fs::remove_file(file);
                }
            }
        }
    }
    pub fn state(ctx: &egui::Context) -> Arc<Mutex<Session>> {
        let id = egui::Id::new("automation_session");
        if let Some(s) = ctx.data(|d| d.get_temp::<Arc<Mutex<Session>>>(id)) {
            return s;
        }
        let s = Arc::new(Mutex::new(Session::default()));
        ctx.data_mut(|d| d.insert_temp(id, s.clone()));
        s
    }
}
#[cfg(not(target_arch = "wasm32"))]
impl PaintFEApp {
    fn update_automation(&mut self, ctx: &egui::Context) {
        use crate::services::automation as a;
        use std::sync::atomic::Ordering;
        let arc = automation_state::state(ctx);
        let Ok(mut s) = arc.lock() else { return };
        if !self.settings.automation.enabled {
            if let Some(job) = s.pending.take() {
                job.cancel.store(true, Ordering::Relaxed);
            }
            if let Some(path) = s.path.take() {
                let file = path.join("session.json");
                if crate::services::plugins::read_limited(&file, 4096)
                    .ok()
                    .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
                    .is_some_and(|v| v["token"].as_str() == Some(s.token.as_str()))
                {
                    let _ = std::fs::remove_file(file);
                }
            }
            s.token.clear();
            #[cfg(target_os = "windows")]
            { s.pipe = None; }
            return;
        }
        if s.path.is_none() {
            match a::prepare().and_then(|path| s.start(path)) {
                Ok(()) => {
                    a::CANCEL.store(false, Ordering::Relaxed);
                    a::set_status(crate::t!("automation.waiting"));
                }
                Err(e) => {
                    a::set_status(e);
                    self.settings.automation.enabled = false;
                    self.settings.save();
                    return;
                }
            }
        }
        if (a::CANCEL.swap(false, Ordering::Relaxed) || !self.settings.automation.editing)
            && let Some(job) = s.pending.as_ref()
        {
            job.cancel.store(true, Ordering::Relaxed);
        }
        let result = if self.document_modal_open() {
            None
        } else {
            s.pending
                .as_ref()
                .and_then(|job| match job.receiver.try_recv() {
                    Ok(value) => Some(value),
                    Err(std::sync::mpsc::TryRecvError::Empty) => None,
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                        Some(Err("Worker stopped before returning a result".into()))
                    }
                })
        };
        if let Some(result) = result {
            let job = s.pending.take().unwrap();
            s.job = match result.and_then(|pixels| {
                if job.cancel.load(Ordering::Relaxed) || !self.settings.automation.editing {
                    return Err("Cancelled".into());
                }
                let p = self
                    .projects
                    .iter_mut()
                    .find(|p| p.id == job.document && p.edit_revision == job.revision)
                    .ok_or("Document changed; result discarded")?;
                let st = &mut p.canvas_state;
                let i = st
                    .layers
                    .iter()
                    .position(|l| l.runtime_id == job.layer)
                    .ok_or("Layer unavailable")?;
                let old = st.layers[i]
                    .pixels
                    .extract_region_rgba(0, 0, st.width, st.height);
                if pixels.len() != old.len() {
                    return Err("Invalid automation output dimensions".into());
                }
                if pixels != old {
                    if job.action == HistoryAction::Script {
                        let mut command = SingleLayerSnapshotCommand::new_for_layer(
                            job.action.label("Agent script"),
                            st,
                            i,
                        );
                        st.layers[i].pixels =
                            TiledImage::from_raw_rgba(st.width, st.height, &pixels);
                        command.set_after(st);
                        p.history.push(Box::new(command));
                        st.mark_dirty(None);
                    } else {
                        use crate::components::history::{BrushCommand, PixelPatch};
                        let mut bounds = (st.width, st.height, 0u32, 0u32);
                        for (index, (a, b)) in
                            old.chunks_exact(4).zip(pixels.chunks_exact(4)).enumerate()
                        {
                            if a != b {
                                let x = index as u32 % st.width;
                                let y = index as u32 / st.width;
                                bounds.0 = bounds.0.min(x);
                                bounds.1 = bounds.1.min(y);
                                bounds.2 = bounds.2.max(x + 1);
                                bounds.3 = bounds.3.max(y + 1);
                            }
                        }
                        let rect = egui::Rect::from_min_max(
                            egui::pos2(bounds.0 as f32, bounds.1 as f32),
                            egui::pos2(bounds.2 as f32, bounds.3 as f32),
                        );
                        let before = PixelPatch::capture(st, i, rect);
                        let mut after = before.clone();
                        for y in 0..after.height {
                            for x in 0..after.width {
                                let src = (((bounds.1 + y) * st.width + bounds.0 + x) * 4) as usize;
                                after.pixels[(y * after.width + x) as usize] =
                                    image::Rgba(pixels[src..src + 4].try_into().unwrap());
                            }
                        }
                        after.apply(st);
                        p.history.push(Box::new(BrushCommand::new(
                            job.action.label("Agent drawing"),
                            before,
                            after,
                        )));
                    }
                    p.mark_dirty();
                }
                Ok(())
            }) {
                Ok(()) => "Applied".into(),
                Err(e) => e,
            };
        }
        #[cfg(target_os = "windows")]
        {
            let reap_ms = if s.pending.is_some() { 20 } else { 50 };
            ctx.request_repaint_after(std::time::Duration::from_millis(reap_ms));
            if let Some(request) = s.pipe.as_ref().and_then(|pipe| pipe.incoming.try_recv().ok()) {
                let response = self.automation_envelope(ctx, &mut s, &request.envelope);
                let _ = request.reply.try_send(Self::automation_reply(response));
                ctx.request_repaint();
            }
        }
        #[cfg(not(target_os = "windows"))]
        {
        ctx.request_repaint_after(std::time::Duration::from_millis(100));
        let now = ctx.input(|i| i.time);
        if now - s.scanned < 0.09 {
            return;
        }
        s.scanned = now;
        let path = s.path.clone().unwrap();
        let Ok(entries) = std::fs::read_dir(&path) else {
            return;
        };
        for entry in entries.flatten().take(128) {
            let file = entry.path();
            let extension = file.extension().and_then(|s| s.to_str());
            let stem = file.file_stem().and_then(|s| s.to_str()).unwrap_or("");
            if extension == Some("response") && uuid::Uuid::parse_str(stem).is_ok() {
                if entry
                    .metadata()
                    .ok()
                    .and_then(|m| m.modified().ok())
                    .and_then(|t| t.elapsed().ok())
                    .is_some_and(|elapsed| elapsed.as_secs() > 30)
                {
                    let _ = std::fs::remove_file(&file);
                }
                continue;
            }
            if extension != Some("request") || entry.file_type().is_ok_and(|kind| kind.is_symlink())
            {
                continue;
            }
            if uuid::Uuid::parse_str(stem).is_err() {
                continue;
            }
            let result = crate::services::plugins::read_limited(&file, 512 * 1024)
                .and_then(|b| {
                    serde_json::from_slice::<serde_json::Value>(&b).map_err(|e| e.to_string())
                })
                .and_then(|envelope| self.automation_envelope(ctx, &mut s, &envelope));
            let response = Self::automation_reply(result);
            let _ = crate::services::plugins::atomic_write(
                &path.join(format!("{stem}.response")),
                response.to_string().as_bytes(),
            );
            let _ = std::fs::remove_file(file);
            break;
        }
        }
    }

    fn automation_envelope(
        &mut self, ctx: &egui::Context, s: &mut automation_state::Session,
        envelope: &serde_json::Value,
    ) -> Result<serde_json::Value, String> {
        let expires = envelope["expires"].as_u64().unwrap_or(0);
        let time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs();
        if envelope["token"].as_str() != Some(s.token.as_str())
            || expires < time || expires > time + 30
        {
            return Err("Expired or invalid automation session".into());
        }
        crate::services::automation::note_request();
        self.automation_call(ctx, &envelope["params"], s)
    }

    fn automation_reply(result: Result<serde_json::Value, String>) -> serde_json::Value {
        match result {
            Ok(value) => value,
            Err(error) => serde_json::json!({"isError":true,"content":[{"type":"text","text":error}]}),
        }
    }

    fn automation_call(
        &mut self,
        _ctx: &egui::Context,
        params: &serde_json::Value,
        session: &mut automation_state::Session,
    ) -> Result<serde_json::Value, String> {
        use serde_json::json;
        let name = params["name"].as_str().ok_or("Missing tool name")?;
        let args = &params["arguments"];
        let text = |v: serde_json::Value| json!({"content":[{"type":"text","text":v.to_string()}],"isError":false});
        match name {
            "documents"=>return Ok(text(json!(self.projects.iter().map(|p|json!({"document":p.id.to_string(),"name":p.name,"revision":p.edit_revision,"width":p.canvas_state.width,"height":p.canvas_state.height,"layers":p.canvas_state.layers.iter().map(|l|json!({"layer":l.runtime_id.to_string(),"name":l.name,"visible":l.visible})).collect::<Vec<_>>(),"active_layer":p.canvas_state.layers[p.canvas_state.active_layer_index].runtime_id.to_string()})).collect::<Vec<_>>()))),
            "job_status"=>return Ok(text(json!({"busy":session.pending.is_some(),"status":session.job}))),
            "cancel"=>{if let Some(job)=&session.pending{job.cancel.store(true,std::sync::atomic::Ordering::Relaxed);}return Ok(text(json!({"cancel_requested":true})));},
            _=>{}
        }
        let document = args["document"]
            .as_str()
            .ok_or("Missing document identity")?;
        let idx = self
            .projects
            .iter()
            .position(|p| p.id.to_string() == document)
            .ok_or("Document unavailable")?;
        if name == "preview" {
            use base64::Engine;
            let st = &self.projects[idx].canvas_state;
            let scale = st.width.max(st.height).div_ceil(768).max(1);
            let (small, _offset) = st.composite_committed_downscaled(
                egui::Rect::from_min_max(
                    egui::Pos2::ZERO,
                    egui::pos2(st.width as f32, st.height as f32),
                ),
                scale,
            );
            let size = small.size;
            let bytes: Vec<u8> = small
                .pixels
                .iter()
                .flat_map(|p| p.to_srgba_unmultiplied())
                .collect();
            let mut data = std::io::Cursor::new(Vec::new());
            image::RgbaImage::from_raw(size[0] as u32, size[1] as u32, bytes)
                .ok_or("Invalid preview")?
                .write_to(&mut data, image::ImageFormat::Png)
                .map_err(|e| e.to_string())?;
            return Ok(
                json!({"content":[{"type":"image","mimeType":"image/png","data":base64::engine::general_purpose::STANDARD.encode(data.into_inner())}]}),
            );
        }
        if !self.settings.automation.editing {
            return Err("Editing permission is disabled".into());
        }
        if self.document_modal_open()
            || self.paste_overlay.is_some()
            || self.tools_panel.is_stroke_active()
        {
            return Err("Finish the current operation first".into());
        }
        if session.pending.is_some() {
            return Err("An automation operation is already running".into());
        }
        let revision = args["revision"].as_u64().ok_or("Missing revision")?;
        let layer = args["layer"].as_str().ok_or("Missing layer identity")?;
        let p = &mut self.projects[idx];
        let st = &mut p.canvas_state;
        let target = st
            .layers
            .iter()
            .position(|l| l.runtime_id.to_string() == layer)
            .ok_or("Layer unavailable")?;
        if p.edit_revision != revision {
            return Err("Document revision changed; inspect it again".into());
        }
        if (name != "activate_layer" && target != st.active_layer_index) || st.edit_layer_mask {
            return Err("Target must be the active raster layer, not a mask".into());
        }
        if st.layers[target].deep_pixels.is_some()
            || !matches!(
                st.layers[target].content,
                crate::canvas::LayerContent::Raster
            )
        {
            return Err("Choose an 8-bit raster layer".into());
        }
        if u64::from(st.width) * u64::from(st.height) > 16 * 1024 * 1024 {
            return Err("Automation raster limit is 16 million pixels".into());
        }
        match name {
            "activate_layer" => {
                st.active_layer_index = target;
                return Ok(text(
                    json!({"layer":st.layers[target].runtime_id.to_string(),"revision":p.edit_revision}),
                ));
            }
            "layer_opacity" => {
                use crate::components::history::{LayerOpCommand, LayerOperation};
                let opacity = args["opacity"]
                    .as_f64()
                    .filter(|n| n.is_finite() && (0.0..=1.0).contains(n))
                    .ok_or("Invalid opacity")? as f32;
                let old = st.layers[target].opacity;
                if old != opacity {
                    st.layers[target].opacity = opacity;
                    p.history
                        .push(Box::new(LayerOpCommand::new(LayerOperation::Opacity {
                            index: target,
                            old_opacity: old,
                            new_opacity: opacity,
                        })));
                    st.mark_dirty(None);
                    p.mark_dirty();
                }
                return Ok(text(json!({"revision":p.edit_revision})));
            }
            "undo" | "redo" => {
                let changed = if name == "undo" {
                    p.history.undo(st)
                } else {
                    p.history.redo(st)
                };
                if changed.is_some() {
                    p.mark_dirty();
                }
                return Ok(text(json!({"revision":p.edit_revision})));
            }
            "new_layer" => {
                use crate::components::history::{LayerOpCommand, LayerOperation};
                let name = args["name"]
                    .as_str()
                    .filter(|n| {
                        !n.trim().is_empty()
                            && n.chars().count() <= 80
                            && !n.chars().any(char::is_control)
                    })
                    .ok_or("Invalid layer name")?
                    .to_string();
                let index = target + 1;
                st.layers.insert(
                    index,
                    Layer::new(name.clone(), st.width, st.height, image::Rgba([0; 4])),
                );
                st.active_layer_index = index;
                p.history
                    .push(Box::new(LayerOpCommand::new(LayerOperation::Add {
                        index,
                        name,
                        width: st.width,
                        height: st.height,
                        folder_id: None,
                    })));
                st.mark_dirty(None);
                let id = st.layers[index].runtime_id.to_string();
                p.mark_dirty();
                return Ok(text(json!({"revision":p.edit_revision,"layer":id})));
            }
            "select_rectangle" | "deselect" => {
                let before = st.selection_mask.clone();
                let before_all = st.selection_all;
                if name == "deselect" {
                    st.clear_selection();
                } else {
                    let number = |key: &str| {
                        args[key]
                            .as_u64()
                            .and_then(|n| u32::try_from(n).ok())
                            .ok_or("Invalid rectangle")
                    };
                    let x = number("x")?;
                    let y = number("y")?;
                    let w = number("width")?;
                    let h = number("height")?;
                    if w == 0
                        || h == 0
                        || x.checked_add(w).is_none_or(|n| n > st.width)
                        || y.checked_add(h).is_none_or(|n| n > st.height)
                    {
                        return Err("Rectangle outside canvas".into());
                    }
                    st.apply_selection_shape(
                        &crate::canvas::SelectionShape::Rectangle {
                            min_x: x,
                            min_y: y,
                            max_x: x + w - 1,
                            max_y: y + h - 1,
                        },
                        crate::canvas::SelectionMode::Replace,
                    );
                }
                p.history.push(Box::new(SelectionCommand::new_states(
                    HistoryAction::Deselect.label("Agent selection"),
                    before,
                    before_all,
                    st.selection_mask.clone(),
                    st.selection_all,
                )));
                p.mark_dirty();
                return Ok(text(json!({"revision":p.edit_revision})));
            }
            "stroke" | "script" | "shape" => {}
            _ => return Err("Unknown tool".into()),
        }
        let source = args["source"].as_str().unwrap_or("").to_string();
        if name == "script"
            && (!self.settings.automation.scripts || source.is_empty() || source.len() > 256 * 1024)
        {
            return Err("Scripts are disabled or source is invalid".into());
        }
        let mut shape_args = args.clone();
        if name == "shape" {
            let field = |key: &str| {
                args[key]
                    .as_f64()
                    .filter(|v| v.is_finite())
                    .ok_or("Invalid shape geometry")
            };
            let (x, y, w, h) = (field("x")?, field("y")?, field("width")?, field("height")?);
            if w <= 0. || h <= 0. {
                return Err("Shape dimensions must be positive".into());
            }
            let points = match args["shape"].as_str() {
                Some("rectangle") => vec![
                    json!([x, y]),
                    json!([x + w, y]),
                    json!([x + w, y + h]),
                    json!([x, y + h]),
                    json!([x, y]),
                ],
                Some("ellipse") => (0..=128)
                    .map(|i| {
                        let a = i as f64 / 128. * std::f64::consts::TAU;
                        json!([
                            x + w * 0.5 + a.cos() * w * 0.5,
                            y + h * 0.5 + a.sin() * h * 0.5
                        ])
                    })
                    .collect(),
                _ => return Err("Unknown shape".into()),
            };
            shape_args["points"] = json!(points);
        }
        let args = &shape_args;
        let mut points = Vec::new();
        let size = args["size"].as_f64().unwrap_or(1.) as f32;
        let mut colour = [0u8; 4];
        if name == "stroke" || name == "shape" {
            let raw = args["points"]
                .as_array()
                .filter(|a| !a.is_empty() && a.len() <= 4096)
                .ok_or("Invalid stroke points")?;
            let colours = args["colour"]
                .as_array()
                .filter(|a| a.len() == 4)
                .ok_or("Invalid RGBA colour")?;
            for (i, v) in colours.iter().enumerate() {
                colour[i] = v
                    .as_u64()
                    .and_then(|n| u8::try_from(n).ok())
                    .ok_or("Invalid colour")?;
            }
            if !size.is_finite() || !(1.0..=512.).contains(&size) {
                return Err("Invalid brush size".into());
            }
            let mut distance = 0.;
            for item in raw {
                let a = item
                    .as_array()
                    .filter(|a| (2..=3).contains(&a.len()))
                    .ok_or("Invalid stroke point")?;
                let x = a[0].as_f64().ok_or("Invalid x")? as f32;
                let y = a[1].as_f64().ok_or("Invalid y")? as f32;
                let pressure = a
                    .get(2)
                    .map(|v| v.as_f64().unwrap_or(f64::NAN) as f32)
                    .unwrap_or(1.);
                if !x.is_finite()
                    || !y.is_finite()
                    || !pressure.is_finite()
                    || x < 0.
                    || y < 0.
                    || x >= st.width as f32
                    || y >= st.height as f32
                    || !(0.0..=1.0).contains(&pressure)
                {
                    return Err("Stroke point outside canvas or invalid pressure".into());
                }
                if let Some(prev) = points.last() {
                    let prev: &[f32; 3] = prev;
                    distance += (x - prev[0]).hypot(y - prev[1]);
                }
                points.push([x, y, pressure]);
            }
            if distance * size > 4_000_000. {
                return Err("Stroke exceeds work limit; split it into shorter strokes".into());
            }
        }
        let pixels = st.layers[target]
            .pixels
            .extract_region_rgba(0, 0, st.width, st.height);
        let mask = st.selection_mask.clone();
        let (width, height) = (st.width, st.height);
        let pencil = args["pencil"].as_bool().unwrap_or(false);
        let erase = args["erase"].as_bool().unwrap_or(false);
        let cancel = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let worker_cancel = cancel.clone();
        let (tx, rx) = std::sync::mpsc::channel();
        let script = name == "script";
        std::thread::spawn(move || {
            let result = if script {
                let effect = crate::services::plugins::Effect {
                    id: "automation".into(),
                    name: "Agent script".into(),
                    category: "Automation".into(),
                    script: "automation.rhai".into(),
                    description: String::new(),
                    parameters: Vec::new(),
                    histogram: false,
                };
                crate::services::plugins::worker::execute(
                    crate::services::plugins::worker::Request {
                        source,
                        width,
                        height,
                        values: Default::default(),
                        seed: 0,
                        colours: None,
                        effect,
                        has_mask: mask.is_some(),
                    },
                    pixels,
                    mask.map(image::GrayImage::into_raw),
                    &worker_cancel,
                )
                .map(|o| o.pixels)
            } else {
                crate::ui::panels::tools::ToolsPanel::render_automation_stroke(
                    width,
                    height,
                    &pixels,
                    mask,
                    &points,
                    colour,
                    size,
                    pencil,
                    erase,
                    &worker_cancel,
                )
            };
            let _ = tx.send(result);
        });
        session.pending = Some(automation_state::Pending {
            document: p.id,
            layer: st.layers[target].runtime_id,
            revision,
            action: if script {
                HistoryAction::Script
            } else if name == "shape" {
                HistoryAction::Shapes
            } else if pencil {
                HistoryAction::Pencil
            } else if erase {
                HistoryAction::Eraser
            } else {
                HistoryAction::Brush
            },
            receiver: rx,
            cancel,
        });
        session.job = "Processing".into();
        Ok(text(json!({"queued":true,"check":"job_status"})))
    }
}
