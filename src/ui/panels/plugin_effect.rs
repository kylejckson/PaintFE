//! Host-owned dialogs. A plugin result cannot alter a document before Apply.
use super::plugin_browser::tr;
use crate::services::plugins::{self as p, *};
use egui::{Context, Ui};
use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    time::{Duration, Instant},
};
#[derive(Clone)]
pub struct Target {
    pub document: uuid::Uuid,
    pub layer: uuid::Uuid,
    pub revision: u64,
    pub width: u32,
    pub height: u32,
    pub pixels: Arc<Vec<u8>>,
    pub mask: Option<Vec<u8>>,
    pub pack: String,
    pub effect: Effect,
    pub source: String,
}
pub struct Applied {
    pub target: Target,
    pub pixels: Vec<u8>,
    pub values: Values,
}
struct Work {
    cancel: Arc<AtomicBool>,
    rx: mpsc::Receiver<Result<p::worker::Output, String>>,
    generation: u64,
}
impl Drop for Work {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}
struct Session {
    target: Target,
    values: Values,
    work: Option<Work>,
    result: Option<Vec<u8>>,
    texture: Option<egui::TextureHandle>,
    original: Option<egui::TextureHandle>,
    generation: u64,
    dirty: bool,
    changed: Instant,
    live: bool,
    error: String,
    show_original: bool,
    apply: bool,
    histogram: [u32; 64],
}
type State = Arc<Mutex<Option<Session>>>;
fn state(ctx: &Context) -> State {
    let id = egui::Id::new("plugin_effect_session");
    if let Some(s) = ctx.data(|d| d.get_temp::<State>(id)) {
        return s;
    }
    let s = Arc::new(Mutex::new(None));
    ctx.data_mut(|d| d.insert_temp(id, s.clone()));
    s
}
pub fn is_open(ctx: &Context) -> bool {
    ctx.data(|d| d.get_temp::<bool>(egui::Id::new("plugin_effect_open")))
        .unwrap_or(false)
}
pub fn open(ctx: &Context, target: Target) {
    let prefs = Preferences::load(&p::root());
    let values = prefs
        .packs
        .get(&target.pack)
        .and_then(|p| p.values.get(&target.effect.id))
        .filter(|v| target.effect.validate_values(v).is_ok())
        .cloned()
        .unwrap_or_else(|| target.effect.defaults());
    if let Ok(mut slot) = state(ctx).lock() {
        *slot = Some(Session {
            target,
            values,
            work: None,
            result: None,
            texture: None,
            original: None,
            generation: 0,
            dirty: true,
            changed: Instant::now() - Duration::from_secs(1),
            live: true,
            error: String::new(),
            show_original: false,
            apply: false,
            histogram: [0; 64],
        });
    }
    ctx.data_mut(|d| d.insert_temp(egui::Id::new("plugin_effect_open"), true));
}
impl Session {
    fn start(&mut self, ctx: &Context) {
        if self.work.is_some() {
            return;
        }
        self.dirty = false;
        self.result = None;
        self.error.clear();
        let target = self.target.clone();
        let values = self.values.clone();
        let generation = self.generation;
        let cancel = Arc::new(AtomicBool::new(false));
        let stop = cancel.clone();
        let (tx, rx) = mpsc::channel();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let r = p::worker::execute(
                p::worker::Request {
                    source: target.source,
                    width: target.width,
                    height: target.height,
                    values,
                    seed: 37477,
                    effect: target.effect,
                    has_mask: target.mask.is_some(),
                },
                target.pixels.as_ref().clone(),
                target.mask,
                &stop,
            );
            let _ = tx.send(r);
            ctx.request_repaint();
        });
        self.work = Some(Work {
            cancel,
            rx,
            generation,
        });
    }
    fn changed(&mut self) {
        self.generation = self.generation.wrapping_add(1);
        self.dirty = true;
        self.apply = false;
        self.result = None;
        self.changed = Instant::now();
        if let Some(w) = &self.work {
            w.cancel.store(true, Ordering::Relaxed);
        }
    }
}
pub fn parameter(ui: &mut Ui, p: &Parameter, value: &mut serde_json::Value) -> bool {
    let before = value.clone();
    ui.push_id(&p.id, |ui| {
        match p.kind.as_str() {
            "number" => {
                let mut v = value.as_f64().unwrap_or(p.min);
                ui.add(
                    crate::ui::controls::Slider::new(&mut v, p.min..=p.max)
                        .id_salt(&p.id)
                        .text(&p.label)
                        .step_by(p.step)
                        .reset_to(p.default.as_f64().unwrap_or(p.min)),
                );
                *value = serde_json::json!(v);
            }
            "integer" => {
                let mut v = value
                    .as_i64()
                    .or_else(|| value.as_f64().map(|v| v as i64))
                    .unwrap_or(p.min as i64);
                ui.add(
                    crate::ui::controls::Slider::new(&mut v, p.min as i64..=p.max as i64)
                        .id_salt(&p.id)
                        .text(&p.label)
                        .step_by(p.step)
                        .reset_to(p.default.as_f64().unwrap_or(p.min) as i64),
                );
                *value = serde_json::json!(v);
            }
            "boolean" => {
                let mut v = value.as_bool().unwrap_or(false);
                ui.checkbox(&mut v, &p.label);
                *value = v.into();
            }
            "text" => {
                ui.label(&p.label);
                let mut v = value.as_str().unwrap_or_default().to_owned();
                ui.add(egui::TextEdit::singleline(&mut v).char_limit(2048));
                *value = v.into();
            }
            "choice" => {
                ui.label(&p.label);
                let mut v = value.as_str().unwrap_or_default().to_owned();
                egui::ComboBox::from_id_salt(&p.id)
                    .selected_text(&v)
                    .show_ui(ui, |ui| {
                        for c in &p.choices {
                            ui.selectable_value(&mut v, c.clone(), c);
                        }
                    });
                *value = v.into();
            }
            "colour" => {
                let a = value
                    .as_array()
                    .filter(|a| a.len() == 4)
                    .map(|a| {
                        a.iter()
                            .map(|v| v.as_u64().unwrap_or(0) as u8)
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_else(|| vec![0, 0, 0, 255]);
                let mut c = egui::Color32::from_rgba_unmultiplied(a[0], a[1], a[2], a[3]);
                let edited = ui
                    .horizontal(|ui| {
                        ui.label(&p.label);
                        crate::ui::controls::color_edit_button_srgba(
                            ui,
                            &mut c,
                            egui::color_picker::Alpha::BlendOrAdditive,
                        )
                        .changed()
                    })
                    .inner;
                if edited {
                    *value = serde_json::json!(c.to_srgba_unmultiplied());
                }
            }
            _ => {}
        }
        if !p.help.is_empty() {
            ui.weak(&p.help);
        }
        ui.add_space(5.);
    });
    *value != before
}
fn texture(ctx: &Context, id: &str, w: u32, h: u32, pixels: &[u8]) -> egui::TextureHandle {
    let img = image::RgbaImage::from_raw(w, h, pixels.to_vec()).unwrap();
    let img = p::thumbnail(&img, 720, 480);
    ctx.load_texture(
        id,
        egui::ColorImage::from_rgba_unmultiplied(
            [img.width() as usize, img.height() as usize],
            img.as_raw(),
        ),
        egui::TextureOptions::LINEAR,
    )
}
pub fn frame(ctx: &Context) -> Option<Applied> {
    let arc = state(ctx);
    let Ok(mut slot) = arc.lock() else {
        return None;
    };
    let s = slot.as_mut()?;
    if let Some(work) = &s.work {
        let message = work.rx.try_recv();
        if let Ok(r) = message {
            let generation = work.generation;
            s.work = None;
            if generation == s.generation {
                match r {
                    Ok(output) => {
                        s.texture = Some(texture(
                            ctx,
                            "plugin-preview",
                            s.target.width,
                            s.target.height,
                            &output.pixels,
                        ));
                        s.histogram = [0; 64];
                        for px in output
                            .pixels
                            .chunks_exact(4)
                            .step_by((output.pixels.len() / 4 / 65536).max(1))
                        {
                            s.histogram
                                [((px[0] as usize + px[1] as usize + px[2] as usize) / 3) / 4] += 1;
                        }
                        s.result = Some(output.pixels);
                    }
                    Err(e) => {
                        s.error = e;
                        s.apply = false;
                    }
                }
            }
        } else if matches!(message, Err(mpsc::TryRecvError::Disconnected)) {
            s.work = None;
            s.error = tr("task_failed");
            s.apply = false;
        } else {
            ctx.request_repaint_after(Duration::from_millis(40));
        }
    }
    if s.dirty
        && (s.live || s.apply)
        && s.work.is_none()
        && s.changed.elapsed() >= Duration::from_millis(200)
    {
        s.start(ctx);
    }
    if s.dirty {
        ctx.request_repaint_after(Duration::from_millis(100));
    }
    if s.original.is_none() {
        s.original = Some(texture(
            ctx,
            "plugin-original",
            s.target.width,
            s.target.height,
            &s.target.pixels,
        ));
    }
    let popup_open = egui::Popup::is_any_open(ctx);
    let finish_keys = ctx.input(|i| {
        (
            !popup_open && i.key_pressed(egui::Key::Enter),
            !popup_open && i.key_pressed(egui::Key::Escape),
        )
    });
    ctx.data_mut(|d| d.insert_temp(egui::Id::new("dialog_footer_keys"), finish_keys));
    let mut cancel = false;
    let bounds = ctx.content_rect();
    egui::Modal::new(egui::Id::new("plugin_effect_dialog")).show(ctx, |ui| {
        ui.set_width((bounds.width() - 64.).clamp(300., 900.));
        let colours = crate::ui::dialogs::core::DialogColors::from_ctx(ctx);
        if crate::ui::dialogs::core::paint_dialog_header(ui, &colours, "", &s.target.effect.name) {
            cancel = true;
        }
        ui.label(&s.target.effect.description);
        egui::ScrollArea::vertical()
            .id_salt("plugin_effect_content")
            .max_height((bounds.height() - 240.).clamp(120., 520.))
            .show(ui, |ui| {
                if ui.available_width() > 680. {
                    ui.columns(2, |columns| {
                        preview(&mut columns[0], s);
                        controls(&mut columns[1], s);
                    });
                } else {
                    preview(ui, s);
                    controls(ui, s);
                }
            });
        ui.separator();
        ui.horizontal(|ui| {
            if s.work.is_some() {
                ui.spinner();
                ui.label(tr("processing"));
            }
            if !s.error.is_empty() {
                ui.colored_label(ui.visuals().error_fg_color, &s.error);
            }
        });
        ui.horizontal(|ui| {
            if ui.checkbox(&mut s.live, tr("live_preview")).changed() && s.live {
                s.dirty = true;
            }
            if ui.button(tr("preview")).clicked() {
                s.dirty = true;
                s.changed = Instant::now() - Duration::from_secs(1);
                s.start(ctx);
            }
            if ui.button(tr("reset")).clicked() {
                s.values = s.target.effect.defaults();
                s.changed();
            }
        });
        let (apply, dismiss) = crate::ui::dialogs::core::dialog_footer_action(
            ui,
            &colours,
            &tr("apply"),
            s.error.is_empty() || s.dirty,
        );
        cancel |= dismiss && !crate::ui::numeric::editor_handled_finish_key(ctx, egui::Key::Escape);
        if apply && !crate::ui::numeric::editor_handled_finish_key(ctx, egui::Key::Enter) {
            s.apply = true;
            if s.dirty {
                s.changed = Instant::now() - Duration::from_secs(1);
            }
        }
    });
    if cancel {
        *slot = None;
        ctx.data_mut(|d| d.insert_temp(egui::Id::new("plugin_effect_open"), false));
        return None;
    }
    if s.apply
        && !s.dirty
        && s.work.is_none()
        && let Some(pixels) = s.result.take()
    {
        let applied = Applied {
            target: s.target.clone(),
            pixels,
            values: s.values.clone(),
        };
        *slot = None;
        ctx.data_mut(|d| d.insert_temp(egui::Id::new("plugin_effect_open"), false));
        return Some(applied);
    }
    None
}
fn controls(ui: &mut Ui, s: &mut Session) {
    let identity = (s.target.pack.clone(), s.target.effect.id.clone());
    ui.push_id(identity, |ui| controls_body(ui, s));
}
fn controls_body(ui: &mut Ui, s: &mut Session) {
    let before = s.values.clone();
    let mut group = String::new();
    for p in &s.target.effect.parameters {
        if !p.group.is_empty() && p.group != group {
            ui.add_space(8.);
            ui.strong(&p.group);
            group = p.group.clone();
        }
        let v = s
            .values
            .entry(p.id.clone())
            .or_insert_with(|| p.default.clone());
        parameter(ui, p, v);
    }
    if before != s.values {
        s.changed();
    }
}
fn preview(ui: &mut Ui, s: &mut Session) {
    ui.checkbox(&mut s.show_original, tr("show_original"));
    let t = if s.show_original {
        s.original.as_ref()
    } else {
        s.texture.as_ref().or(s.original.as_ref())
    };
    if let Some(t) = t {
        ui.add(
            egui::Image::new(t)
                .max_size(egui::vec2(ui.available_width(), 340.))
                .maintain_aspect_ratio(true),
        );
    }
    if s.target.effect.histogram {
        let (rect, _) =
            ui.allocate_exact_size(egui::vec2(ui.available_width(), 55.), egui::Sense::hover());
        let max = s.histogram.iter().copied().max().unwrap_or(1).max(1) as f32;
        for (i, v) in s.histogram.iter().enumerate() {
            let w = rect.width() / 64.;
            ui.painter().rect_filled(
                egui::Rect::from_min_max(
                    egui::pos2(
                        rect.left() + i as f32 * w,
                        rect.bottom() - *v as f32 / max * rect.height(),
                    ),
                    egui::pos2(rect.left() + (i + 1) as f32 * w, rect.bottom()),
                ),
                0.,
                ui.visuals().selection.stroke.color,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn every_parameter_kind_uses_production_controls_without_changing_defaults() {
        crate::i18n::init();
        crate::i18n::set_language("en");
        let parameters = vec![
            ("number", serde_json::json!(0.5), vec![]),
            ("integer", serde_json::json!(2), vec![]),
            ("boolean", serde_json::json!(true), vec![]),
            ("text", serde_json::json!("hello"), vec![]),
            (
                "choice",
                serde_json::json!("A"),
                vec!["A".into(), "B".into()],
            ),
            ("colour", serde_json::json!([12, 80, 180, 127]), vec![]),
        ];
        for dark in [false, true] {
            for zoom in [1., 1.5] {
                let ctx = Context::default();
                ctx.set_visuals(if dark {
                    egui::Visuals::dark()
                } else {
                    egui::Visuals::light()
                });
                ctx.set_zoom_factor(zoom);
                for (kind, default, choices) in &parameters {
                    let p = Parameter {
                        editor_id: uuid::Uuid::new_v4(),
                        id: format!("p_{kind}"),
                        label: kind.to_string(),
                        kind: kind.to_string(),
                        group: String::new(),
                        default: default.clone(),
                        help: String::new(),
                        min: 0.,
                        max: 10.,
                        step: 0.1,
                        choices: choices.clone(),
                    };
                    let mut value = default.clone();
                    for _ in 0..3 {
                        let _ = ctx.run_ui(Default::default(), |ui| {
                            parameter(ui, &p, &mut value);
                        });
                    }
                    assert_eq!(&value, default, "{kind}");
                }
            }
        }
    }
    #[test]
    fn obsolete_generations_do_not_have_an_applicable_result() {
        let ctx = Context::default();
        let target = Target {
            document: uuid::Uuid::new_v4(),
            layer: uuid::Uuid::new_v4(),
            revision: 0,
            width: 1,
            height: 1,
            pixels: Arc::new(vec![0; 4]),
            mask: None,
            pack: "local.test".into(),
            effect: Effect {
                id: "test".into(),
                name: "Test".into(),
                category: "Custom".into(),
                script: "effect.rhai".into(),
                description: String::new(),
                parameters: vec![],
                histogram: false,
            },
            source: "apply_invert();".into(),
        };
        open(&ctx, target);
        let shared = state(&ctx);
        let mut slot = shared.lock().unwrap();
        let s = slot.as_mut().unwrap();
        s.result = Some(vec![255; 4]);
        s.apply = true;
        s.changed();
        assert!(s.result.is_none());
        assert!(!s.apply);
        assert!(s.dirty);
        *slot = None;
    }
}
