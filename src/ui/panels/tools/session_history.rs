impl MeshWarpToolState {
    pub fn reset_session(&mut self) {
        self.preview_cancel
            .store(true, std::sync::atomic::Ordering::Relaxed);
        let (grid_cols, grid_rows, edit_reference) =
            (self.grid_cols, self.grid_rows, self.edit_reference);
        *self = Self {
            grid_cols,
            grid_rows,
            edit_reference,
            ..Self::default()
        };
    }

    pub fn layout(&self) -> MeshLayout {
        MeshLayout {
            original: self.original_points.clone(),
            points: self.points.clone(),
        }
    }

    pub fn finish_drag(&mut self) {
        if let Some(before) = self.drag_before.take()
            && before != self.layout()
        {
            if self.undo_layouts.len() >= 128 {
                self.undo_layouts.remove(0);
            }
            self.undo_layouts.push(before);
            self.redo_layouts.clear();
        }
    }

    pub fn reference_move_valid(&self, index: usize, point: [f32; 2]) -> bool {
        let mut points = self.original_points.clone();
        if index >= points.len() || !point.iter().all(|p| p.is_finite()) {
            return false;
        }
        points[index] = point;
        let stride = self.grid_cols + 1;
        for row in 0..self.grid_rows {
            for col in 0..self.grid_cols {
                let i = row * stride + col;
                let corners = [
                    points[i],
                    points[i + 1],
                    points[i + stride + 1],
                    points[i + stride],
                ];
                for k in 0..4 {
                    let a = corners[k];
                    let b = corners[(k + 1) % 4];
                    let c = corners[(k + 2) % 4];
                    if (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]) <= 0.01 {
                        return false;
                    }
                }
            }
        }
        true
    }
}

impl ToolsPanel {
    pub(crate) fn needs_deformation_update(&self, canvas: &CanvasState) -> bool {
        let mesh = &self.mesh_warp_state;
        let liquify = &self.liquify_state;
        mesh.commit_pending
            || mesh.cancel_pending
            || liquify.commit_pending
            || liquify.cancel_pending
            || liquify.reset_pending
            || liquify.protect_selection_pending
            || liquify.clear_protection_pending
            || (liquify.is_active && self.active_tool != Tool::Liquify)
            || (mesh.is_active
                && (self.active_tool != Tool::MeshWarp
                    || mesh.preview_receiver.is_some()
                    || mesh.preview_selection_generation != canvas.selection_overlay_generation
                    || if mesh.points == mesh.original_points {
                        mesh.preview_layout.is_some()
                    } else {
                        mesh.preview_layout.as_ref().is_none_or(|layout| {
                            layout.points != mesh.points || layout.original != mesh.original_points
                        })
                    }))
    }

    pub(crate) fn cancel_pending_deformation_commit(&mut self) {
        self.mesh_warp_state.commit_pending = false;
        self.mesh_warp_state.commit_pending_frame = 0;
        self.liquify_state.reset_pending = false;
        self.liquify_state.commit_pending = false;
        self.liquify_state.commit_pending_frame = 0;
    }

    pub(crate) fn reset_mesh_deformation(&mut self) {
        let state = &mut self.mesh_warp_state;
        state.commit_pending = false;
        state.commit_pending_frame = 0;
        state.drag_before = Some(state.layout());
        state.points = state.original_points.clone();
        state.finish_drag();
        state.dragging_index = None;
    }

    pub(crate) fn clear_deformation_sessions(&mut self) {
        self.mesh_warp_state.reset_session();
        self.liquify_state.reset_session();
    }

    pub(crate) fn draw_liquify_protection(
        &mut self,
        ui: &egui::Ui,
        painter: &egui::Painter,
        image_rect: egui::Rect,
    ) {
        let state = &mut self.liquify_state;
        let Some(mask) = &state.protection else {
            return;
        };
        if mask.width() == 0 || mask.height() == 0 {
            return;
        }
        if state.protection_dirty || state.protection_texture.is_none() {
            let factor = mask.width().max(mask.height()).div_ceil(512).max(1);
            let w = mask.width().div_ceil(factor);
            let h = mask.height().div_ceil(factor);
            let mut pixels = Vec::with_capacity(w as usize * h as usize);
            for y in 0..h {
                for x in 0..w {
                    let alpha = mask
                        .get_pixel(
                            (x * factor).min(mask.width() - 1),
                            (y * factor).min(mask.height() - 1),
                        )
                        .0[0];
                    pixels.push(egui::Color32::from_rgba_unmultiplied(
                        70,
                        110,
                        230,
                        alpha / 3,
                    ));
                }
            }
            let image = egui::ColorImage::new([w as usize, h as usize], pixels);
            if let Some(texture) = state.protection_texture.as_mut() {
                texture.set(image, egui::TextureOptions::LINEAR);
            } else {
                state.protection_texture = Some(ui.ctx().load_texture(
                    "liquify_protection",
                    image,
                    egui::TextureOptions::LINEAR,
                ));
            }
            state.protection_dirty = false;
        }
        if let Some(texture) = &state.protection_texture {
            painter.image(
                texture.id(),
                image_rect,
                egui::Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1., 1.)),
                egui::Color32::WHITE,
            );
        }
    }

    fn advance_mesh_preview(&mut self, ctx: &egui::Context, canvas: &mut CanvasState) {
        let state = &mut self.mesh_warp_state;
        if state.preview_selection_generation != canvas.selection_overlay_generation {
            state
                .preview_cancel
                .store(true, std::sync::atomic::Ordering::Relaxed);
            state.preview_cancel = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
            state.preview_receiver = None;
            state.preview_layout = None;
            state.preview_epoch = state.preview_epoch.wrapping_add(1);
            state.preview_selection_generation = canvas.selection_overlay_generation;
            state.selection_snapshot = canvas.selection_mask.clone().map(std::sync::Arc::new);
        }
        let layout = state.layout();
        let mut downscale = 1u32;
        while canvas.width.max(canvas.height).div_ceil(downscale) > 512 {
            downscale *= 2;
        }
        let received = state
            .preview_receiver
            .as_ref()
            .map(|rx| rx.lock().map(|rx| rx.try_recv()));
        let disconnected = matches!(
            received,
            Some(Ok(Err(std::sync::mpsc::TryRecvError::Disconnected))) | Some(Err(_))
        );
        let result = received.and_then(|result| result.ok()?.ok());
        if disconnected {
            state.preview_receiver = None;
        }
        if let Some(result) = result {
            state.preview_receiver = None;
            if result.epoch == state.preview_epoch {
                state.preview_source = Some(result.source);
                if result.layout == layout && state.is_active {
                    let (w, h) = result.pixels.dimensions();
                    canvas.preview_layer =
                        Some(TiledImage::from_raw_rgba(w, h, result.pixels.as_raw()));
                    canvas.preview_blend_mode = BlendMode::Normal;
                    canvas.preview_force_composite = true;
                    canvas.preview_replaces_layer = true;
                    canvas.preview_is_eraser = false;
                    canvas.preview_downscale = result.downscale;
                    canvas.preview_stroke_bounds = Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(canvas.width as f32, canvas.height as f32),
                    ));
                    canvas.preview_texture_cache = None;
                    canvas.mark_render_dirty(None);
                    state.preview_layout = Some(layout.clone());
                }
            }
        }
        if state.points == state.original_points {
            if state.preview_layout.take().is_some() {
                canvas.clear_preview_state();
                canvas.mark_render_dirty(None);
            }
            return;
        }
        if state.preview_receiver.is_some() {
            ctx.request_repaint_after(std::time::Duration::from_millis(16));
            return;
        }
        if state.preview_layout.as_ref() == Some(&layout) {
            return;
        }
        let Some(source) = state.source_snapshot.clone() else {
            return;
        };
        let cached = state.preview_source.clone();
        let cols = state.grid_cols;
        let rows = state.grid_rows;
        let epoch = state.preview_epoch;
        let cancel = state.preview_cancel.clone();
        let selection = state.selection_snapshot.clone();
        let (tx, rx) = std::sync::mpsc::channel();
        state.preview_receiver = Some(std::sync::Arc::new(std::sync::Mutex::new(rx)));
        let ctx = ctx.clone();
        crate::par_compat::spawn(move || {
            use std::sync::atomic::Ordering;
            if cancel.load(Ordering::Relaxed) {
                return;
            }
            let small = cached.unwrap_or_else(|| {
                std::sync::Arc::new(image::imageops::resize(
                    &*source,
                    source.width().div_ceil(downscale),
                    source.height().div_ceil(downscale),
                    image::imageops::FilterType::Triangle,
                ))
            });
            let sx = small.width() as f32 / source.width() as f32;
            let sy = small.height() as f32 / source.height() as f32;
            let scale = |points: &[[f32; 2]]| {
                points
                    .iter()
                    .map(|p| [p[0] * sx, p[1] * sy])
                    .collect::<Vec<_>>()
            };
            if cancel.load(Ordering::Relaxed) {
                return;
            }
            let mut pixels = crate::ops::transform::warp_reference_mesh(
                &small,
                &scale(&layout.original),
                &scale(&layout.points),
                cols,
                rows,
            );
            if let Some(mask) = selection.filter(|mask| mask.width() > 0 && mask.height() > 0) {
                for (x, y, pixel) in pixels.enumerate_pixels_mut() {
                    let mx = (x * downscale).min(mask.width().saturating_sub(1));
                    let my = (y * downscale).min(mask.height().saturating_sub(1));
                    if mask.get_pixel(mx, my).0[0] == 0 {
                        *pixel = *small.get_pixel(x, y);
                    }
                }
            }
            if !cancel.load(Ordering::Relaxed) {
                let _ = tx.send(MeshPreview {
                    layout,
                    epoch,
                    downscale,
                    source: small,
                    pixels,
                });
                ctx.request_repaint();
            }
        });
    }

    pub(crate) fn can_undo_adjustment(&self) -> bool {
        (self.active_tool == Tool::MeshWarp && !self.mesh_warp_state.undo_layouts.is_empty())
            || (self.active_tool == Tool::Liquify && self.liquify_state.history.can_undo())
    }

    pub(crate) fn can_redo_adjustment(&self) -> bool {
        (self.active_tool == Tool::MeshWarp && !self.mesh_warp_state.redo_layouts.is_empty())
            || (self.active_tool == Tool::Liquify && self.liquify_state.history.can_redo())
    }

    pub(crate) fn undo_adjustment(&mut self, canvas: &mut CanvasState) -> bool {
        if self.active_tool == Tool::Liquify {
            let state = &mut self.liquify_state;
            let changed = state
                .displacement
                .as_mut()
                .is_some_and(|field| state.history.undo(field));
            if changed {
                state.dirty_rect = None;
                self.render_liquify_preview(canvas, None);
            }
            return changed;
        }
        if !self.can_undo_adjustment() {
            return false;
        }
        let state = &mut self.mesh_warp_state;
        let previous = state
            .undo_layouts
            .pop()
            .expect("checked adjustment history");
        state.redo_layouts.push(state.layout());
        state.original_points = previous.original;
        state.points = previous.points;
        state.preview_layout = None;
        canvas.clear_preview_state();
        canvas.mark_render_dirty(None);
        true
    }

    pub(crate) fn redo_adjustment(&mut self, canvas: &mut CanvasState) -> bool {
        if self.active_tool == Tool::Liquify {
            let state = &mut self.liquify_state;
            let changed = state
                .displacement
                .as_mut()
                .is_some_and(|field| state.history.redo(field));
            if changed {
                state.dirty_rect = None;
                self.render_liquify_preview(canvas, None);
            }
            return changed;
        }
        if !self.can_redo_adjustment() {
            return false;
        }
        let state = &mut self.mesh_warp_state;
        let next = state
            .redo_layouts
            .pop()
            .expect("checked adjustment history");
        state.undo_layouts.push(state.layout());
        state.original_points = next.original;
        state.points = next.points;
        state.preview_layout = None;
        canvas.clear_preview_state();
        canvas.mark_render_dirty(None);
        true
    }
}
impl LiquifyToolState {
    pub fn sample_frame_motion(
        &mut self,
        motion: &[(f32, f32)],
        time: f64,
        frame_dt: f64,
        released: bool,
    ) -> Vec<([f32; 2], [f32; 2])> {
        let mut path: Vec<_> = motion
            .iter()
            .filter(|point| point.0.is_finite() && point.1.is_finite())
            .map(|point| [point.0, point.1])
            .collect();
        path.dedup();
        if path.is_empty() {
            return Vec::new();
        }
        let frame_dt = if frame_dt.is_finite() {
            frame_dt.clamp(1. / 240., 0.25)
        } else {
            1. / 60.
        };
        let initial = self.last_frame_time.is_none();
        let start_time = self.last_frame_time.unwrap_or({
            if path.len() == 1 {
                time
            } else {
                time - frame_dt
            }
        });
        let mut samples = Vec::new();
        if initial {
            self.last_pos = Some(path[0]);
            samples.extend(self.sample_motion(path[0], start_time));
            path.remove(0);
        } else if path.len() > 1 && Some(path[0]) == self.last_pos {
            path.remove(0);
        }
        let count = path.len();
        for (index, point) in path.into_iter().enumerate() {
            let sample_time = start_time + (time - start_time) * (index + 1) as f64 / count as f64;
            samples.extend(self.sample_motion(point, sample_time));
            self.last_pos = Some(point);
        }
        if released
            && let (Some(point), Some(last)) = (self.last_pos, self.sample_position)
            && (point[0] - last[0]).abs().max((point[1] - last[1]).abs()) > 0.0001
        {
            samples.push((point, [point[0] - last[0], point[1] - last[1]]));
            self.sample_position = Some(point);
        }
        samples
    }

    pub fn reset_session(&mut self) {
        let (strength, diameter, mode) = (self.strength, self.diameter, self.mode);
        *self = Self {
            strength,
            diameter,
            mode,
            ..Self::default()
        };
    }

    /// Resample a piecewise-linear pointer path at 60 Hz, independently of UI FPS.
    pub fn sample_motion(&mut self, point: [f32; 2], time: f64) -> Vec<([f32; 2], [f32; 2])> {
        const STEP: f64 = 1.0 / 60.0;
        let Some(previous_time) = self.last_frame_time else {
            self.last_frame_time = Some(time);
            self.next_sample_time = time + STEP;
            self.sample_position = Some(point);
            return vec![(point, [0., 0.])];
        };
        let previous = self.last_pos.unwrap_or(point);
        let elapsed = time - previous_time;
        if !(0.0..=0.25).contains(&elapsed) {
            self.last_frame_time = None;
            return self.sample_motion(point, time);
        }
        let mut samples = Vec::new();
        while self.next_sample_time <= time + 1e-9 && samples.len() < 16 {
            let t =
                ((self.next_sample_time - previous_time) / elapsed.max(1e-9)).clamp(0., 1.) as f32;
            let p = [
                previous[0] + (point[0] - previous[0]) * t,
                previous[1] + (point[1] - previous[1]) * t,
            ];
            let last = self.sample_position.unwrap_or(previous);
            samples.push((p, [p[0] - last[0], p[1] - last[1]]));
            self.sample_position = Some(p);
            self.next_sample_time += STEP;
        }
        self.last_frame_time = Some(time);
        samples
    }
}
