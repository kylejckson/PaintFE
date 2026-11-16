//! Plugin properties compose the existing script editor and shared parameter controls.
use super::plugin_browser::{self as browser, tr};
use crate::services::plugins::{self as p, *};
use egui::Context;
use std::{
    collections::BTreeMap,
    io::Write,
    sync::{Arc, Mutex},
};
struct Draft {
    open: bool,
    manifest: Manifest,
    resources: BTreeMap<String, Vec<u8>>,
    selected: usize,
    message: String,
    tab: usize,
    modified: bool,
    art_generation: u64,
    was_open: bool,
}
impl Draft {
    fn remove_effect(&mut self, code: &mut String) {
        if self.manifest.effects.len() <= 1 {
            return;
        }
        let removed = self.manifest.effects.remove(self.selected);
        if !self
            .manifest
            .effects
            .iter()
            .any(|effect| effect.script == removed.script)
        {
            self.resources.remove(&removed.script);
        }
        self.selected = self.selected.min(self.manifest.effects.len() - 1);
        *code = String::from_utf8_lossy(
            self.resources
                .get(&self.manifest.effects[self.selected].script)
                .unwrap(),
        )
        .into_owned();
    }
}

type State = Arc<Mutex<Option<Draft>>>;
fn state(ctx: &Context) -> State {
    let id = egui::Id::new("plugin_author");
    if let Some(s) = ctx.data(|d| d.get_temp::<State>(id)) {
        return s;
    }
    let s = Arc::new(Mutex::new(None));
    ctx.data_mut(|d| d.insert_temp(id, s.clone()));
    s
}
pub fn editing(ctx: &Context) -> bool {
    state(ctx).lock().is_ok_and(|s| s.is_some())
}
pub fn unbind(ctx: &Context) {
    if let Ok(mut s) = state(ctx).lock() {
        *s = None;
    }
}
pub fn create(ctx: &Context, source: &str) {
    let effect = Effect {
        id: "effect".into(),
        name: tr("my_effect"),
        category: tr("custom"),
        script: "effects/effect.rhai".into(),
        description: String::new(),
        parameters: Vec::new(),
        histogram: false,
    };
    let mut resources = BTreeMap::new();
    resources.insert(effect.script.clone(), source.as_bytes().to_vec());
    resources.insert(
        "LICENSE".into(),
        b"Personal local plugin. Choose a distribution licence before submitting.\n".to_vec(),
    );
    let manifest = Manifest {
        kind: crate::services::plugins::PackKind::Effects,
        appearance: String::new(),
        schema: 1,
        api: 1,
        id: format!(
            "local.effect-{}",
            &uuid::Uuid::new_v4().simple().to_string()[..8]
        ),
        version: "1.0.0".into(),
        name: tr("my_plugin"),
        author: tr("local_author"),
        description: tr("my_description"),
        license: "Personal".into(),
        tags: vec!["Custom".into()],
        effects: vec![effect],
        files: Vec::new(),
        icon: String::new(),
        menu_light: String::new(),
        menu_dark: String::new(),
        screenshot: String::new(),
    };
    if let Ok(mut s) = state(ctx).lock() {
        *s = Some(Draft {
            open: true,
            manifest,
            resources,
            selected: 0,
            message: String::new(),
            tab: 0,
            modified: true,
            art_generation: 0,
            was_open: false,
        });
    }
}
pub fn edit(ctx: &Context, pack: &Installed, code: &mut String) -> Result<(), String> {
    if pack.manifest.kind != crate::services::plugins::PackKind::Effects {
        return Err("Appearance packs are edited through theme and icon settings.".into());
    }
    let mut resources = BTreeMap::new();
    for f in &pack.manifest.files {
        let b = read_limited(&pack.directory.join(&f.path), 4 * 1024 * 1024)?;
        if hash(&b) != f.sha256 {
            return Err("Plugin resource has changed".into());
        }
        resources.insert(f.path.clone(), b);
    }
    let mut manifest = pack.manifest.clone();
    manifest.id = format!(
        "local.{}-{}",
        manifest.id.chars().take(42).collect::<String>(),
        &uuid::Uuid::new_v4().simple().to_string()[..8]
    );
    manifest.version = "1.0.0".into();
    manifest.name = format!("{} ({})", manifest.name, tr("local_copy"));
    *code = String::from_utf8(
        resources
            .get(&manifest.effects[0].script)
            .cloned()
            .unwrap_or_default(),
    )
    .map_err(|e| e.to_string())?;
    if let Ok(mut s) = state(ctx).lock() {
        *s = Some(Draft {
            open: true,
            manifest,
            resources,
            selected: 0,
            message: String::new(),
            tab: 0,
            modified: true,
            art_generation: 0,
            was_open: false,
        });
    }
    Ok(())
}
fn package(
    mut manifest: Manifest,
    resources: &BTreeMap<String, Vec<u8>>,
) -> Result<Vec<u8>, String> {
    manifest.files = resources
        .iter()
        .map(|(path, b)| PackFile {
            path: path.clone(),
            size: b.len() as u64,
            sha256: hash(b),
        })
        .collect();
    manifest.validate()?;
    let envelope = Envelope {
        payload: serde_json::to_string(&manifest).map_err(|e| e.to_string())?,
        signature: String::new(),
    };
    let mut out = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
    let options = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated);
    out.start_file("manifest.json", options)
        .map_err(|e| e.to_string())?;
    out.write_all(&serde_json::to_vec(&envelope).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    for (path, b) in resources {
        out.start_file(path, options).map_err(|e| e.to_string())?;
        out.write_all(b).map_err(|e| e.to_string())?;
    }
    Ok(out.finish().map_err(|e| e.to_string())?.into_inner())
}
fn export_bytes(name: &str, bytes: &[u8]) -> Result<(), String> {
    if let Some(path) = crate::services::file_dialogs::dialog(
        crate::services::file_dialogs::Purpose::Resource,
        None,
    )
    .set_file_name(format!("{name}.pfeplugin"))
    .add_filter("PaintFE plugin", &["pfeplugin"])
    .save_file()
    {
        atomic_write(&path, bytes)?;
    }
    Ok(())
}
pub fn export_installed(pack: &Installed) -> Result<(), String> {
    let mut bytes = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
    let opts = zip::write::SimpleFileOptions::default();
    for path in
        std::iter::once("manifest.json").chain(pack.manifest.files.iter().map(|f| f.path.as_str()))
    {
        let b = read_limited(&pack.directory.join(path), MAX_CATALOGUE)?;
        bytes.start_file(path, opts).map_err(|e| e.to_string())?;
        bytes.write_all(&b).map_err(|e| e.to_string())?;
    }
    export_bytes(
        &pack.manifest.id,
        &bytes.finish().map_err(|e| e.to_string())?.into_inner(),
    )
}
fn save(d: &mut Draft) -> Result<(), String> {
    let prefs = Preferences::load(&p::root());
    if prefs
        .packs
        .get(&d.manifest.id)
        .is_some_and(|p| p.current == d.manifest.version)
    {
        let v = version_key(&d.manifest.version);
        d.manifest.version = format!("{}.{}.{}", v[0], v[1], v[2] + 1);
    }
    let bytes = package(d.manifest.clone(), &d.resources)?;
    let m = install(&bytes, &p::root(), true, None)?;
    d.manifest = m;
    d.modified = false;
    Ok(())
}
pub fn request_save(ctx: &Context, code: &str, copy: bool) {
    if let Ok(mut slot) = state(ctx).lock()
        && let Some(d) = slot.as_mut()
    {
        d.resources.insert(
            d.manifest.effects[d.selected].script.clone(),
            code.as_bytes().to_vec(),
        );
        if copy {
            d.modified = true;
            d.manifest.id = format!(
                "local.effect-{}",
                &uuid::Uuid::new_v4().simple().to_string()[..8]
            );
            d.manifest.version = "1.0.0".into();
            d.open = true;
        } else {
            d.message = match save(d) {
                Ok(()) => tr("saved"),
                Err(e) => {
                    d.open = true;
                    e
                }
            };
        }
    }
    browser::reload(ctx);
}
pub fn properties(ctx: &Context) {
    if let Ok(mut slot) = state(ctx).lock()
        && let Some(d) = slot.as_mut()
    {
        if !d.open {
            d.was_open = false;
        }
        d.open = true;
    }
}
pub fn is_open(ctx: &Context) -> bool {
    ctx.data(|d| d.get_temp::<State>(egui::Id::new("plugin_author")))
        .is_some_and(|state| {
            state
                .lock()
                .is_ok_and(|s| s.as_ref().is_some_and(|d| d.open))
        })
}
pub fn frame(ctx: &Context, code: &mut String) {
    let arc = state(ctx);
    let Ok(mut slot) = arc.lock() else { return };
    let Some(d) = slot.as_mut() else { return };
    let path = d.manifest.effects[d.selected].script.clone();
    if d.resources
        .get(&path)
        .is_none_or(|bytes| bytes.as_slice() != code.as_bytes())
    {
        d.modified = true;
        d.resources.insert(path, code.as_bytes().to_vec());
    }
    if !d.open {
        d.was_open = false;
        return;
    }
    let just_opened = !d.was_open;
    d.was_open = true;
    let mut open = true;
    let mut close = false;
    let bounds = ctx.content_rect();
    let shown = crate::ui::polish::window(ctx, tr("properties"))
        .id(egui::Id::new("plugin_properties"))
        .title_bar(false)
        .collapsible(false)
        .open(&mut open)
        .heading_drag(
            ctx,
            egui::Id::new("plugin_author_heading"),
            bounds.center() - egui::vec2(390., 300.),
        )
        .default_size([780., 600.])
        .min_size([440., 360.])
        .max_size([bounds.width() - 32., bounds.height() - 64.])
        .constrain_to(bounds.shrink(12.))
        .show(ctx, |ui| {
            let colours = crate::ui::dialogs::core::DialogColors::from_ctx(ctx);
            if just_opened {
                ctx.data_mut(|data| {
                    data.insert_temp(egui::Id::new("dialog_footer_keys"), (false, false))
                });
            }
            let finish_keys = crate::ui::dialogs::core::dialog_footer_keys(ctx);
            let editing_text = ctx.text_edit_focused();
            close |=
                crate::ui::dialogs::core::paint_dialog_header(ui, &colours, "", &tr("properties"));
            ui.add_space(8.);
            ui.weak(tr("properties_hint"));
            ui.horizontal(|ui| {
                for (index, key) in ["overview", "effects_controls", "artwork_sharing"]
                    .iter()
                    .enumerate()
                {
                    ui.selectable_value(&mut d.tab, index, tr(key));
                }
            });
            ui.separator();
            let before = serde_json::to_string(&d.manifest).unwrap_or_default();
            egui::ScrollArea::vertical()
                .id_salt(("plugin_properties_content", d.tab))
                .max_height((bounds.height() - 260.).clamp(140., 430.))
                .show(ui, |ui| {
                    ui.set_min_height(260_f32.min((bounds.height() - 260.).max(100.)));
                    crate::ui::controls::without_wheel_adjustment(ui, |ui| match d.tab {
                        0 => overview(ui, d),
                        1 => effects(ui, d, code),
                        _ => artwork(ui, d),
                    });
                });
            if before != serde_json::to_string(&d.manifest).unwrap_or_default() {
                d.modified = true;
            }
            ui.separator();
            ui.weak(if d.message.is_empty() {
                tr(if d.modified { "draft_unsaved" } else { "saved" })
            } else {
                d.message.clone()
            });
            ui.horizontal(|ui| {
                if ui.button(tr("edit_code")).clicked() {
                    close = true;
                }
                ui.hyperlink_to(
                    tr("author_guide"),
                    format!("{REPOSITORY}/blob/main/docs/getting-started.md"),
                );
            });
            let (save_requested, cancel) = crate::ui::dialogs::core::dialog_footer_action(
                ui,
                &colours,
                &tr("save_local"),
                d.modified,
            );
            close |=
                cancel && !crate::ui::numeric::editor_handled_finish_key(ctx, egui::Key::Escape);
            if save_requested
                && !(finish_keys.0 && editing_text)
                && !crate::ui::numeric::editor_handled_finish_key(ctx, egui::Key::Enter)
            {
                d.message = match save(d) {
                    Ok(()) => tr("saved"),
                    Err(error) => error,
                };
                browser::reload(ctx);
            }
        });
    #[cfg(test)]
    if let Some(shown) = shown {
        ctx.data_mut(|d| d.insert_temp(egui::Id::new("plugin_author_bounds"), shown.response.rect));
    }
    #[cfg(not(test))]
    let _ = shown;
    d.open = open && !close;
}
fn overview(ui: &mut egui::Ui, d: &mut Draft) {
    ui.strong(tr("overview"));
    ui.add_space(8.);
    egui::Grid::new("plugin_identity")
        .num_columns(2)
        .spacing([16., 10.])
        .show(ui, |ui| {
            for (key, value) in [
                ("name", &mut d.manifest.name),
                ("author", &mut d.manifest.author),
            ] {
                ui.label(tr(key));
                ui.add(
                    egui::TextEdit::singleline(value)
                        .char_limit(80)
                        .desired_width((ui.available_width() - 30.).max(180.)),
                );
                ui.end_row();
            }
        });
    ui.add_space(12.);
    ui.label(tr("description"));
    ui.add(
        egui::TextEdit::multiline(&mut d.manifest.description)
            .desired_width(ui.available_width())
            .desired_rows(3)
            .char_limit(4096),
    );
    ui.add_space(10.);
    let mut tags = d.manifest.tags.join(", ");
    ui.label(tr("tags"));
    if ui.text_edit_singleline(&mut tags).changed() {
        d.manifest.tags = tags
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .take(12)
            .map(str::to_owned)
            .collect();
    }
    ui.add_space(12.);
    ui.collapsing(tr("advanced"), |ui| {
        for (key, value) in [
            ("version", &mut d.manifest.version),
            ("license", &mut d.manifest.license),
            ("pack_id", &mut d.manifest.id),
        ] {
            ui.horizontal(|ui| {
                ui.label(tr(key));
                ui.add(egui::TextEdit::singleline(value).char_limit(100));
            });
        }
    });
}
fn effects(ui: &mut egui::Ui, d: &mut Draft, code: &mut String) {
    if ui.available_width() > 650. {
        let width = ui.available_width();
        ui.horizontal_top(|ui| {
            ui.allocate_ui_with_layout(
                egui::vec2(175., 200.),
                egui::Layout::top_down(egui::Align::Min),
                |ui| {
                    ui.set_min_width(175.);
                    ui.strong(tr("effects"));
                    let mut next = d.selected;
                    for (index, effect) in d.manifest.effects.iter().enumerate() {
                        ui.selectable_value(&mut next, index, &effect.name);
                    }
                    if next != d.selected {
                        d.selected = next;
                        *code = String::from_utf8_lossy(
                            d.resources.get(&d.manifest.effects[next].script).unwrap(),
                        )
                        .into_owned();
                    }
                    effect_actions(ui, d, code);
                },
            );
            ui.separator();
            ui.vertical(|ui| {
                ui.set_width((width - 200.).max(160.));
                effect_fields(ui, d);
            });
        });
    } else {
        let mut next = d.selected;
        egui::ComboBox::from_id_salt("author_effect")
            .selected_text(&d.manifest.effects[d.selected].name)
            .show_ui(ui, |ui| {
                for (index, effect) in d.manifest.effects.iter().enumerate() {
                    ui.selectable_value(&mut next, index, &effect.name);
                }
            });
        if next != d.selected {
            d.selected = next;
            *code =
                String::from_utf8_lossy(d.resources.get(&d.manifest.effects[next].script).unwrap())
                    .into_owned();
        }
        ui.horizontal_wrapped(|ui| effect_actions(ui, d, code));
        ui.separator();
        effect_fields(ui, d);
    }
}
fn effect_actions(ui: &mut egui::Ui, d: &mut Draft, code: &mut String) {
    if ui
        .add_enabled(
            d.manifest.effects.len() < 16,
            egui::Button::new(tr("add_effect")),
        )
        .clicked()
    {
        let next = (1..=1024)
            .find(|n| {
                !d.manifest
                    .effects
                    .iter()
                    .any(|e| e.id == format!("effect_{n}"))
            })
            .unwrap();
        let id = format!("effect_{next}");
        let script = format!("effects/{id}.rhai");
        d.resources
            .insert(script.clone(), b"apply_invert();\n".to_vec());
        d.manifest.effects.push(Effect {
            id,
            name: tr("my_effect"),
            category: tr("custom"),
            script,
            description: String::new(),
            parameters: vec![],
            histogram: false,
        });
        d.selected = d.manifest.effects.len() - 1;
        *code = "apply_invert();\n".into();
    }
    if ui
        .add_enabled(
            d.manifest.effects.len() > 1,
            egui::Button::new(tr("remove_effect")),
        )
        .clicked()
    {
        d.remove_effect(code);
    }
}
fn effect_fields(ui: &mut egui::Ui, d: &mut Draft) {
    let e = &mut d.manifest.effects[d.selected];
    ui.horizontal(|ui| {
        ui.label(tr("name"));
        ui.text_edit_singleline(&mut e.name);
    });
    ui.horizontal(|ui| {
        ui.label(tr("category"));
        ui.text_edit_singleline(&mut e.category);
    });
    ui.label(tr("description"));
    ui.text_edit_singleline(&mut e.description);
    ui.checkbox(&mut e.histogram, tr("histogram"));
    ui.strong(tr("parameters"));
    let mut remove = None;
    for (index, p) in e.parameters.iter_mut().enumerate() {
        ui.push_id(p.editor_id, |ui| {
            egui::Frame::group(ui.style())
                .inner_margin(10.)
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.label(tr("name"));
                        ui.text_edit_singleline(&mut p.label);
                    });
                    let previous = p.kind.clone();
                    egui::ComboBox::from_id_salt("kind")
                        .selected_text(tr(&p.kind))
                        .show_ui(ui, |ui| {
                            for kind in ["number", "integer", "boolean", "text", "choice", "colour"]
                            {
                                ui.selectable_value(&mut p.kind, kind.to_string(), tr(kind));
                            }
                        });
                    if p.kind != previous {
                        p.default = match p.kind.as_str() {
                            "boolean" => false.into(),
                            "text" => "".into(),
                            "choice" => {
                                p.choices = vec!["Option".into()];
                                "Option".into()
                            }
                            "colour" => serde_json::json!([0, 0, 0, 255]),
                            "integer" => {
                                p.min = p.min.floor();
                                p.max = p.max.ceil().max(p.min);
                                p.step = 1.;
                                serde_json::json!(p.min as i64)
                            }
                            _ => serde_json::json!(p.min),
                        };
                    }
                    if matches!(p.kind.as_str(), "number" | "integer") {
                        for (key, v) in [
                            ("minimum", &mut p.min),
                            ("maximum", &mut p.max),
                            ("step", &mut p.step),
                        ] {
                            ui.horizontal(|ui| {
                                ui.label(tr(key));
                                ui.add(crate::ui::controls::Numeric::new(v).id_salt(key));
                            });
                        }
                    }
                    if p.kind == "choice" {
                        let mut choices = p.choices.join(", ");
                        ui.label(tr("choices"));
                        if ui.text_edit_singleline(&mut choices).changed() {
                            p.choices = choices
                                .split(',')
                                .map(str::trim)
                                .filter(|s| !s.is_empty())
                                .map(str::to_owned)
                                .collect();
                            if !p.validate_value(&p.default) {
                                p.default = p.choices.first().cloned().unwrap_or_default().into();
                            }
                        }
                    }
                    ui.label(tr("default"));
                    let mut value = p.default.clone();
                    if p.min.is_finite()
                        && p.max.is_finite()
                        && p.step.is_finite()
                        && p.step > 0.
                        && p.min <= p.max
                    {
                        super::plugin_effect::parameter(ui, p, &mut value);
                    }
                    p.default = value;
                    ui.collapsing(tr("advanced"), |ui| {
                        ui.label(tr("parameter_id"));
                        ui.text_edit_singleline(&mut p.id);
                        ui.label(tr("group"));
                        ui.text_edit_singleline(&mut p.group);
                        ui.label(tr("help"));
                        ui.text_edit_singleline(&mut p.help);
                    });
                    if ui.button(tr("remove_parameter")).clicked() {
                        remove = Some(index);
                    }
                });
        });
    }
    if let Some(i) = remove {
        e.parameters.remove(i);
    }
    if e.parameters.len() < 24 && ui.button(tr("add_parameter")).clicked() {
        let n = (1..=1024)
            .find(|n| !e.parameters.iter().any(|p| p.id == format!("amount_{n}")))
            .unwrap();
        e.parameters.push(Parameter {
            editor_id: uuid::Uuid::new_v4(),
            id: format!("amount_{n}"),
            label: format!("{} {n}", tr("amount")),
            kind: "number".into(),
            group: String::new(),
            default: serde_json::json!(0.5),
            help: String::new(),
            min: 0.,
            max: 1.,
            step: 0.01,
            choices: vec![],
        });
    }
}
fn artwork(ui: &mut egui::Ui, d: &mut Draft) {
    ui.strong(tr("artwork_sharing"));
    ui.add_space(8.);
    for key in ["icon", "screenshot", "menu_light", "menu_dark"] {
        ui.push_id(key, |ui| {
            egui::Frame::group(ui.style())
                .inner_margin(10.)
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        let field = match key {
                            "icon" => &mut d.manifest.icon,
                            "screenshot" => &mut d.manifest.screenshot,
                            "menu_light" => &mut d.manifest.menu_light,
                            _ => &mut d.manifest.menu_dark,
                        };
                        if let Some(bytes) = d.resources.get(field) {
                            let id = egui::Id::new((
                                "plugin_art_preview",
                                &d.manifest.id,
                                key,
                                d.art_generation,
                            ));
                            let cached = ui
                                .ctx()
                                .data(|data| data.get_temp::<egui::TextureHandle>(id));
                            let texture = cached.or_else(|| {
                                image::load_from_memory(bytes).ok().map(|image| {
                                    let image = p::thumbnail(&image.to_rgba8(), 128, 72);
                                    let texture = ui.ctx().load_texture(
                                        format!("{id:?}"),
                                        egui::ColorImage::from_rgba_unmultiplied(
                                            [image.width() as usize, image.height() as usize],
                                            image.as_raw(),
                                        ),
                                        egui::TextureOptions::LINEAR,
                                    );
                                    ui.ctx()
                                        .data_mut(|data| data.insert_temp(id, texture.clone()));
                                    texture
                                })
                            });
                            if let Some(texture) = texture {
                                ui.add(egui::Image::new(&texture).max_size(egui::vec2(128., 72.)));
                            }
                        }
                        ui.vertical(|ui| {
                            ui.strong(tr(key));
                            if ui.button(tr("replace_art")).clicked()
                                && let Some(path) = crate::services::file_dialogs::dialog(
                                    crate::services::file_dialogs::Purpose::Resource,
                                    None,
                                )
                                .add_filter("PNG", &["png"])
                                .pick_file()
                            {
                                match read_limited(&path, 4 * 1024 * 1024)
                                    .and_then(|bytes| validate_image(&bytes).map(|_| bytes))
                                {
                                    Ok(bytes) => {
                                        let name = format!("art/{key}.png");
                                        *field = name.clone();
                                        d.resources.insert(name, bytes);
                                        d.modified = true;
                                        d.art_generation = d.art_generation.wrapping_add(1);
                                    }
                                    Err(error) => d.message = error,
                                }
                            }
                        });
                    });
                });
        });
    }
    ui.add_space(12.);
    ui.horizontal_wrapped(|ui| {
        if ui.button(tr("export")).clicked() {
            d.message = match package(d.manifest.clone(), &d.resources)
                .and_then(|bytes| export_bytes(&d.manifest.id, &bytes))
            {
                Ok(()) => tr("exported"),
                Err(error) => error,
            };
        }
        ui.hyperlink_to(
            tr("submit"),
            format!("{REPOSITORY}/issues/new?template=plugin-submission.yml"),
        );
    });
}

pub fn preview(ctx: &Context, code: &str) -> Result<(String, Effect, String), String> {
    let arc = state(ctx);
    let mut slot = arc.lock().map_err(|_| "Plugin editor is busy")?;
    let d = slot.as_mut().ok_or("No plugin is being edited")?;
    let e = d.manifest.effects[d.selected].clone();
    e.validate_values(&e.defaults())?;
    d.open = false;
    Ok((d.manifest.id.clone(), e, code.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn author_package_roundtrip_and_copy_keep_signed_original() {
        crate::i18n::init();
        let ctx = Context::default();
        create(&ctx, "apply_invert();");
        assert!(editing(&ctx));
        let state = state(&ctx);
        let mut guard = state.lock().unwrap();
        let draft = guard.as_mut().unwrap();
        let bytes = package(draft.manifest.clone(), &draft.resources).unwrap();
        let at = std::env::temp_dir().join(format!("paintfe-author-{}", uuid::Uuid::new_v4()));
        let manifest = install(&bytes, &at, true, None).unwrap();
        let packs = installed(&at, &Preferences::load(&at));
        assert_eq!(packs.len(), 1);
        let original = manifest.id;
        drop(guard);
        let mut code = String::new();
        edit(&ctx, &packs[0], &mut code).unwrap();
        assert_eq!(code, "apply_invert();");
        let guard = state.lock().unwrap();
        assert_ne!(guard.as_ref().unwrap().manifest.id, original);
        drop(guard);
        let (id, _, source) = preview(&ctx, "apply_grayscale();").unwrap();
        assert_ne!(id, original);
        assert_eq!(source, "apply_grayscale();");
        unbind(&ctx);
        assert!(!editing(&ctx));
        std::fs::remove_dir_all(at).unwrap();
    }
    #[test]
    fn author_control_identity_survives_manifest_roundtrip_and_renaming() {
        let manifest: Envelope = serde_json::from_slice(&zip_manifest()).unwrap();
        let mut manifest: Manifest = serde_json::from_str(&manifest.payload).unwrap();
        let p = &mut manifest.effects[0].parameters[0];
        let identity = p.editor_id;
        p.id = "renamed".into();
        p.label = "Different name".into();
        assert_eq!(identity, p.editor_id);
        let json = serde_json::to_string(p).unwrap();
        assert!(!json.contains("editor_id"));
        let restored: Parameter = serde_json::from_str(&json).unwrap();
        assert_ne!(restored.editor_id, identity);
    }
    #[test]
    fn removing_an_effect_keeps_a_script_shared_by_another_effect() {
        crate::i18n::init();
        let ctx = Context::default();
        create(&ctx, "apply_invert();");
        let state = state(&ctx);
        let mut guard = state.lock().unwrap();
        let draft = guard.as_mut().unwrap();
        let mut second = draft.manifest.effects[0].clone();
        second.id = "other".into();
        draft.manifest.effects.push(second);
        let mut code = String::new();
        draft.remove_effect(&mut code);
        assert_eq!(draft.manifest.effects.len(), 1);
        assert_eq!(code, "apply_invert();");
        assert!(
            draft
                .resources
                .contains_key(&draft.manifest.effects[0].script)
        );
    }
    #[test]
    fn properties_tabs_render_in_both_themes_and_preserve_drafts_when_closed() {
        crate::i18n::init();
        for dark in [false, true] {
            for (width, height, zoom) in [(1100., 850., 1.), (620., 600., 1.), (1000., 850., 1.5)] {
                let ctx = Context::default();
                let mut fonts = egui::FontDefinitions::default();
                fonts.families.insert(
                    egui::FontFamily::Name("WidgetTitle".into()),
                    fonts.families[&egui::FontFamily::Proportional].clone(),
                );
                ctx.set_fonts(fonts);
                ctx.set_visuals(if dark {
                    egui::Visuals::dark()
                } else {
                    egui::Visuals::light()
                });
                ctx.set_zoom_factor(zoom);
                create(&ctx, "apply_invert();");
                let mut code = "apply_invert();".to_string();
                let shared = state(&ctx);
                for tab in 0..3 {
                    shared.lock().unwrap().as_mut().unwrap().tab = tab;
                    for frame_nr in 0..3 {
                        let output = ctx.run_ui(
                            egui::RawInput {
                                screen_rect: Some(egui::Rect::from_min_size(
                                    egui::Pos2::ZERO,
                                    egui::vec2(width / zoom, height / zoom),
                                )),
                                time: Some((tab * 3 + frame_nr) as f64),
                                ..Default::default()
                            },
                            |ui| frame(ui.ctx(), &mut code),
                        );
                        if frame_nr == 2 {
                            assert!(!output.shapes.is_empty());
                            assert!(ctx.memory(|m| m.top_modal_layer()).is_some());
                            let rect = ctx
                                .data(|d| {
                                    d.get_temp::<egui::Rect>(egui::Id::new("plugin_author_bounds"))
                                })
                                .unwrap();
                            assert!(
                                ctx.content_rect().expand(1.).contains_rect(rect),
                                "Window escaped viewport: {rect:?}, viewport {:?}",
                                ctx.content_rect()
                            );
                            assert!(
                                rect.height() <= ctx.content_rect().height() + 1.,
                                "Properties exceeded viewport: {rect:?}"
                            );
                        }
                    }
                    assert!(is_open(&ctx));
                }
                let id = shared.lock().unwrap().as_ref().unwrap().manifest.id.clone();
                let _ = preview(&ctx, &code).unwrap();
                assert!(!is_open(&ctx));
                assert!(editing(&ctx));
                assert_eq!(shared.lock().unwrap().as_ref().unwrap().manifest.id, id);
            }
        }
    }
    #[test]
    fn save_as_marks_a_previously_saved_draft_as_unsaved() {
        crate::i18n::init();
        let ctx = Context::default();
        create(&ctx, "apply_invert();");
        let shared = state(&ctx);
        shared.lock().unwrap().as_mut().unwrap().modified = false;
        let old = shared.lock().unwrap().as_ref().unwrap().manifest.id.clone();
        request_save(&ctx, "apply_invert();", true);
        let guard = shared.lock().unwrap();
        let draft = guard.as_ref().unwrap();
        assert!(draft.modified);
        assert_ne!(draft.manifest.id, old);
    }
    #[test]
    fn opening_properties_with_enter_does_not_save_a_draft() {
        crate::i18n::init();
        let ctx = Context::default();
        let mut fonts = egui::FontDefinitions::default();
        fonts.families.insert(
            egui::FontFamily::Name("WidgetTitle".into()),
            fonts.families[&egui::FontFamily::Proportional].clone(),
        );
        ctx.set_fonts(fonts);
        create(&ctx, "apply_invert();");
        let mut code = "apply_invert();".to_string();
        let _ = ctx.run_ui(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(1000., 800.),
                )),
                events: vec![egui::Event::Key {
                    key: egui::Key::Enter,
                    physical_key: None,
                    pressed: true,
                    repeat: false,
                    modifiers: egui::Modifiers::NONE,
                }],
                ..Default::default()
            },
            |ui| frame(ui.ctx(), &mut code),
        );
        let shared = state(&ctx);
        let guard = shared.lock().unwrap();
        let draft = guard.as_ref().unwrap();
        assert!(draft.modified);
        assert!(draft.manifest.files.is_empty());
        assert!(draft.open);
    }
    #[test]
    fn keyboard_focus_on_save_can_save_without_treating_the_button_as_text_input() {
        crate::i18n::init();
        crate::i18n::set_language("en");
        let ctx = Context::default();
        let mut fonts = egui::FontDefinitions::default();
        fonts.families.insert(
            egui::FontFamily::Name("WidgetTitle".into()),
            fonts.families[&egui::FontFamily::Proportional].clone(),
        );
        ctx.set_fonts(fonts);
        create(&ctx, "apply_invert();");
        let mut code = "apply_invert();".to_string();
        let render = |events, code: &mut String| {
            ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(1000., 800.),
                    )),
                    events,
                    ..Default::default()
                },
                |ui| frame(ui.ctx(), code),
            )
        };
        for _ in 0..3 {
            render(vec![], &mut code);
        }
        let key = |key, pressed| egui::Event::Key {
            key,
            physical_key: None,
            pressed,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        };
        let mut focused = false;
        for _ in 0..40 {
            let output = render(
                vec![key(egui::Key::Tab, true), key(egui::Key::Tab, false)],
                &mut code,
            );
            if output.platform_output.events.iter().any(|event|matches!(event,egui::output::OutputEvent::FocusGained(info) if info.label.as_deref().is_some_and(|label|label.trim()==tr("save_local")))) {focused=true;break;}
        }
        assert!(focused, "Save must be reachable by keyboard");
        assert!(!ctx.text_edit_focused());
        render(
            vec![key(egui::Key::Enter, true), key(egui::Key::Enter, false)],
            &mut code,
        );
        assert!(
            !state(&ctx).lock().unwrap().as_ref().unwrap().modified,
            "Enter on the focused Save button must save the draft"
        );
    }
    fn zip_manifest() -> Vec<u8> {
        use std::io::Read;
        let bytes = include_bytes!(
            "../../../tests/fixtures/plugin-paintfe.colour-workshop-1.0.0.pfeplugin"
        );
        let mut zip = zip::ZipArchive::new(std::io::Cursor::new(bytes)).unwrap();
        let mut result = Vec::new();
        zip.by_name("manifest.json")
            .unwrap()
            .read_to_end(&mut result)
            .unwrap();
        result
    }
}
