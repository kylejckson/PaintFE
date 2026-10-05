//! Shared interaction and motion policy. Document changes are never animated.
use egui::{Color32, Context, Id, Response, Stroke, Ui};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum MotionMode {
    Off,
    #[default]
    Subtle,
    Expressive,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MotionKind {
    Hover,
    Selection,
    Expansion,
    Overlay,
    Rows,
    Confirmation,
    Snapping,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PolishSettings {
    pub mode: MotionMode,
    pub speed: f32,
    pub respect_reduced_motion: bool,
    pub hover_motion: bool,
    pub selection_motion: bool,
    pub expansion_motion: bool,
    pub overlay_motion: bool,
    pub row_motion: bool,
    pub confirmation_motion: bool,
    pub snap_motion: bool,
    pub spring_strength: f32,
    pub focus_width: f32,
    pub border_width: f32,
    pub spacing_scale: f32,
    pub text_scale: f32,
    pub icon_gap: f32,
    pub numeric_wheel: bool,
    pub fine_adjustment: f32,
}
impl Default for PolishSettings {
    fn default() -> Self {
        Self {
            mode: MotionMode::Subtle,
            speed: 1.0,
            respect_reduced_motion: true,
            hover_motion: true,
            selection_motion: true,
            expansion_motion: true,
            overlay_motion: true,
            row_motion: true,
            confirmation_motion: true,
            snap_motion: true,
            spring_strength: 0.08,
            focus_width: 1.5,
            border_width: 1.0,
            spacing_scale: 1.0,
            text_scale: 1.0,
            icon_gap: 4.0,
            numeric_wheel: true,
            fine_adjustment: 0.1,
        }
    }
}
impl PolishSettings {
    pub fn sanitize(&mut self) {
        fn bounded(v: &mut f32, default: f32, min: f32, max: f32) {
            *v = if v.is_finite() {
                v.clamp(min, max)
            } else {
                default
            };
        }
        bounded(&mut self.speed, 1.0, 0.5, 2.0);
        bounded(&mut self.spring_strength, 0.08, 0.0, 0.15);
        bounded(&mut self.focus_width, 1.5, 1.0, 3.0);
        bounded(&mut self.border_width, 1.0, 0.5, 2.0);
        bounded(&mut self.spacing_scale, 1.0, 0.8, 1.5);
        bounded(&mut self.text_scale, 1.0, 0.85, 1.35);
        bounded(&mut self.icon_gap, 4.0, 2.0, 8.0);
        bounded(&mut self.fine_adjustment, 0.1, 0.01, 0.5);
    }
    pub fn duration(&self, kind: MotionKind, reduced: bool) -> f32 {
        if self.mode == MotionMode::Off || (self.respect_reduced_motion && reduced) {
            return 0.0;
        }
        let (enabled, duration) = match kind {
            MotionKind::Hover => (self.hover_motion, 0.075),
            MotionKind::Selection => (self.selection_motion, 0.12),
            MotionKind::Expansion => (self.expansion_motion, 0.16),
            MotionKind::Overlay => (self.overlay_motion, 0.12),
            MotionKind::Rows => (self.row_motion, 0.15),
            MotionKind::Confirmation => (self.confirmation_motion, 0.10),
            MotionKind::Snapping => (self.snap_motion, 0.10),
        };
        if enabled { duration / self.speed } else { 0.0 }
    }
}

pub fn settings(ctx: &Context) -> PolishSettings {
    ctx.data(|d| d.get_temp(Id::new("paintfe_polish")))
        .unwrap_or_default()
}
pub fn reduced_motion(ctx: &Context) -> bool {
    ctx.data(|d| d.get_temp(Id::new("paintfe_reduced_motion")))
        .unwrap_or(false)
}

pub fn remember_popup_input(ctx: &Context, was_open: bool) {
    let blocked = was_open || egui::Popup::is_any_open(ctx);
    ctx.data_mut(|d| d.insert_temp(Id::new("paintfe_popup_input_blocked"), blocked));
}
pub fn popup_blocks_input(ctx: &Context) -> bool {
    egui::Popup::is_any_open(ctx)
        || ctx
            .data(|d| d.get_temp::<bool>(Id::new("paintfe_popup_input_blocked")))
            .unwrap_or(false)
}
pub fn duration(ctx: &Context, kind: MotionKind) -> f32 {
    settings(ctx).duration(kind, reduced_motion(ctx))
}

pub fn window<'a>(ctx: &Context, title: impl Into<egui::WidgetText>) -> PolishedWindow<'a> {
    let enabled = duration(ctx, MotionKind::Overlay) > 0.0;
    PolishedWindow(
        egui::Window::new(title).fade_in(enabled).fade_out(enabled),
        false,
    )
}

pub struct PolishedWindow<'a>(egui::Window<'a>, bool);
impl<'a> PolishedWindow<'a> {
    pub fn open(mut self, open: &'a mut bool) -> Self {
        self.0 = self.0.open(open);
        self
    }
    pub fn id(mut self, id: Id) -> Self {
        self.0 = self.0.id(id);
        self
    }
    pub fn frame(mut self, frame: egui::Frame) -> Self {
        self.0 = self.0.frame(frame);
        self
    }
    pub fn anchor(mut self, align: egui::Align2, offset: impl Into<egui::Vec2>) -> Self {
        self.0 = self.0.anchor(align, offset);
        self
    }
    pub fn pivot(mut self, pivot: egui::Align2) -> Self {
        self.0 = self.0.pivot(pivot);
        self
    }
    pub fn auto_sized(mut self) -> Self {
        self.0 = self.0.auto_sized();
        self
    }
    pub fn resizable(mut self, value: impl Into<egui::emath::Vec2b>) -> Self {
        self.0 = self.0.resizable(value);
        self
    }
    pub fn title_bar(mut self, value: bool) -> Self {
        self.0 = self.0.title_bar(value);
        self
    }
    pub fn movable(mut self, value: bool) -> Self {
        self.0 = self.0.movable(value);
        self
    }
    pub fn collapsible(mut self, value: bool) -> Self {
        self.0 = self.0.collapsible(value);
        self
    }
    pub fn constrain(mut self, value: bool) -> Self {
        self.0 = self.0.constrain(value);
        self
    }
    pub fn vscroll(mut self, value: bool) -> Self {
        self.0 = self.0.vscroll(value);
        self
    }
    pub fn hscroll(mut self, value: bool) -> Self {
        self.0 = self.0.hscroll(value);
        self
    }
    pub fn fade_in(mut self, value: bool) -> Self {
        self.0 = self.0.fade_in(value);
        self
    }
    pub fn fade_out(mut self, value: bool) -> Self {
        self.0 = self.0.fade_out(value);
        self
    }
    pub fn default_width(mut self, value: f32) -> Self {
        self.0 = self.0.default_width(value);
        self
    }
    pub fn default_height(mut self, value: f32) -> Self {
        self.0 = self.0.default_height(value);
        self
    }
    pub fn min_width(mut self, value: f32) -> Self {
        self.0 = self.0.min_width(value);
        self
    }
    pub fn min_height(mut self, value: f32) -> Self {
        self.0 = self.0.min_height(value);
        self
    }
    pub fn max_width(mut self, value: f32) -> Self {
        self.0 = self.0.max_width(value);
        self
    }
    pub fn max_height(mut self, value: f32) -> Self {
        self.0 = self.0.max_height(value);
        self
    }
    pub fn default_size(mut self, value: impl Into<egui::Vec2>) -> Self {
        self.0 = self.0.default_size(value);
        self
    }
    pub fn fixed_size(mut self, value: impl Into<egui::Vec2>) -> Self {
        self.0 = self.0.fixed_size(value);
        self.1 = true;
        self
    }
    pub fn min_size(mut self, value: impl Into<egui::Vec2>) -> Self {
        self.0 = self.0.min_size(value);
        self
    }
    pub fn max_size(mut self, value: impl Into<egui::Vec2>) -> Self {
        self.0 = self.0.max_size(value);
        self
    }
    pub fn default_pos(mut self, value: impl Into<egui::Pos2>) -> Self {
        self.0 = self.0.default_pos(value);
        self
    }
    pub fn fixed_pos(mut self, value: impl Into<egui::Pos2>) -> Self {
        self.0 = self.0.fixed_pos(value);
        self
    }
    pub fn current_pos(mut self, value: impl Into<egui::Pos2>) -> Self {
        self.0 = self.0.current_pos(value);
        self
    }
    pub fn show<R>(
        self,
        ctx: &Context,
        content: impl FnOnce(&mut Ui) -> R,
    ) -> Option<egui::InnerResponse<Option<R>>> {
        let fixed_size = self.1;
        self.0.show(ctx, |ui| {
            // egui 0.35's non-resizable Window measures only content at the end
            // of layout. Fill its requested body so fixed outer sizes survive.
            if fixed_size {
                ui.set_min_size(ui.available_size());
            }
            // Window fades use the global Overlay timing. Body disclosure widgets
            // get Expansion timing without changing other windows or their hit areas.
            ui.style_mut().animation_time = duration(ctx, MotionKind::Expansion);
            content(ui)
        })
    }
}

pub struct Collapse(egui::CollapsingHeader);
impl Collapse {
    pub fn new(title: impl Into<egui::WidgetText>) -> Self {
        Self(egui::CollapsingHeader::new(title))
    }
    pub fn default_open(mut self, open: bool) -> Self {
        self.0 = self.0.default_open(open);
        self
    }
    pub fn id_salt(mut self, id: impl std::hash::Hash + std::fmt::Debug) -> Self {
        self.0 = self.0.id_salt(id);
        self
    }
    pub fn open(mut self, open: Option<bool>) -> Self {
        self.0 = self.0.open(open);
        self
    }
    pub fn show<R>(
        self,
        ui: &mut Ui,
        content: impl FnOnce(&mut Ui) -> R,
    ) -> egui::collapsing_header::CollapsingResponse<R> {
        let seconds = duration(ui.ctx(), MotionKind::Expansion);
        ui.scope(|ui| {
            ui.style_mut().animation_time = seconds;
            self.0.show(ui, content)
        })
        .inner
    }
}

pub fn panel_hover(ctx: &Context, id: Id) -> f32 {
    let hovered = ctx
        .data(|d| d.get_temp::<bool>(id.with("last_hover")))
        .unwrap_or(false);
    hover(ctx, id, hovered)
}
pub fn remember_hover(ctx: &Context, id: Id, hovered: bool) {
    let changed = ctx.data_mut(|d| {
        let old = d.get_temp::<bool>(id.with("last_hover")).unwrap_or(false);
        d.insert_temp(id.with("last_hover"), hovered);
        old != hovered
    });
    if changed {
        ctx.request_repaint();
    }
}

#[derive(Clone, Copy)]
struct Transition {
    value: f32,
    from: f32,
    target: f32,
    start: f64,
}

pub fn animate(ctx: &Context, id: Id, target: f32, kind: MotionKind) -> f32 {
    let state_id = id.with("paintfe_transition");
    let previous = ctx.data(|d| d.get_temp::<Transition>(state_id));
    if previous.is_some_and(|s| s.target == target && s.value == target) {
        return target;
    }
    let config = settings(ctx);
    let duration = config.duration(kind, reduced_motion(ctx));
    let now = ctx.input(|i| i.time);
    let mut state = previous.unwrap_or(Transition {
        value: target,
        from: target,
        target,
        start: now,
    });
    if duration == 0.0 {
        state = Transition {
            value: target,
            from: target,
            target,
            start: now,
        };
    } else {
        if state.target != target {
            state.from = state.value;
            state.target = target;
            state.start = now;
        }
        let t = ((now - state.start) as f32 / duration).clamp(0.0, 1.0);
        let mut eased = 1.0 - (1.0 - t).powi(3);
        if config.mode == MotionMode::Expressive
            && matches!(
                kind,
                MotionKind::Selection | MotionKind::Expansion | MotionKind::Snapping
            )
        {
            // A short, damped settle; bounded strength, no queued transitions.
            eased += config.spring_strength * (std::f32::consts::PI * t).sin() * t * (1.0 - t);
        }
        state.value = if t >= 1.0 {
            target
        } else {
            state.from + (target - state.from) * eased
        };
        if t < 1.0 && state.from != target {
            ctx.request_repaint();
        }
    }
    ctx.data_mut(|d| d.insert_temp(state_id, state));
    state.value
}

pub fn reset(ctx: &Context, id: Id) {
    ctx.data_mut(|d| d.remove::<Transition>(id.with("paintfe_transition")));
}

pub fn hover(ctx: &Context, id: Id, active: bool) -> f32 {
    animate(ctx, id, if active { 1.0 } else { 0.0 }, MotionKind::Hover)
}

pub fn focus(ui: &Ui, response: &Response, rounding: impl Into<egui::CornerRadius> + Copy) {
    if ui.is_enabled() && response.has_focus() {
        ui.painter().rect_stroke(
            response.rect.shrink(0.75),
            rounding,
            Stroke::new(
                settings(ui.ctx()).focus_width,
                ui.visuals().selection.stroke.color,
            ),
            egui::StrokeKind::Inside,
        );
    }
}

pub fn control(
    ui: &Ui,
    response: &Response,
    selected: bool,
    rounding: impl Into<egui::CornerRadius> + Copy,
) {
    let visuals = ui.visuals();
    let t = hover(
        ui.ctx(),
        response.id.with("hover"),
        response.hovered() && ui.is_enabled(),
    );
    let fill = if !ui.is_enabled() {
        visuals.widgets.noninteractive.weak_bg_fill
    } else if response.is_pointer_button_down_on() {
        visuals.widgets.active.bg_fill
    } else if selected {
        visuals.selection.bg_fill
    } else {
        visuals.widgets.hovered.bg_fill.gamma_multiply(t)
    };
    if fill.a() > 0 {
        ui.painter().rect_filled(response.rect, rounding, fill);
    }
    if selected || response.is_pointer_button_down_on() {
        ui.painter().rect_stroke(
            response.rect.shrink(0.5),
            rounding,
            Stroke::new(
                settings(ui.ctx()).border_width,
                visuals.selection.stroke.color,
            ),
            egui::StrokeKind::Inside,
        );
    }
    focus(ui, response, rounding);
}

pub fn icon_color(ui: &Ui) -> Color32 {
    if ui.is_enabled() {
        ui.visuals().text_color()
    } else {
        ui.visuals().weak_text_color()
    }
}

/// Called once per frame. OS queries run infrequently, never during painting.
pub fn configure(ctx: &Context, config: &PolishSettings) {
    let _scope = super::perf::Scope::new(1);
    let mut config = config.clone();
    config.sanitize();
    let changed = ctx
        .data(|d| d.get_temp::<PolishSettings>(Id::new("paintfe_polish")))
        .as_ref()
        != Some(&config);
    if changed {
        ctx.data_mut(|d| d.insert_temp(Id::new("paintfe_polish"), config.clone()));
    }
    let now = ctx.input(|i| i.time);
    let last = ctx
        .data(|d| d.get_temp::<f64>(Id::new("paintfe_motion_os_check")))
        .unwrap_or(-100.0);
    if config.respect_reduced_motion && now - last > 5.0 && !ctx.input(|i| i.pointer.any_down()) {
        ctx.data_mut(|d| d.insert_temp(Id::new("paintfe_motion_os_check"), now));
        update_system_motion(ctx);
    }
    let seconds = config.duration(MotionKind::Overlay, reduced_motion(ctx));
    if changed && config.mode == MotionMode::Off {
        ctx.clear_animations();
    }
    if ctx.global_style().animation_time != seconds {
        ctx.global_style_mut(|s| s.animation_time = seconds);
    }
}

#[cfg(target_os = "windows")]
fn update_system_motion(ctx: &Context) {
    let _scope = super::perf::Scope::new(2);
    let mut enabled: winapi::shared::minwindef::BOOL = 1;
    let ok = unsafe {
        winapi::um::winuser::SystemParametersInfoW(
            winapi::um::winuser::SPI_GETCLIENTAREAANIMATION,
            0,
            &mut enabled as *mut _ as *mut _,
            0,
        )
    };
    if ok != 0 {
        ctx.data_mut(|d| d.insert_temp(Id::new("paintfe_reduced_motion"), enabled == 0));
    }
}

#[cfg(target_os = "linux")]
fn update_system_motion(ctx: &Context) {
    // KDE's animation factor is local configuration. GNOME's setting is read
    // on a worker so invoking gsettings can never stall the UI thread.
    let ctx = ctx.clone();
    let busy = ctx.data_mut(|d| {
        let id = Id::new("paintfe_motion_os_busy");
        let busy = d.get_temp::<bool>(id).unwrap_or(false);
        if !busy {
            d.insert_temp(id, true);
        }
        busy
    });
    if busy {
        return;
    }
    std::thread::spawn(move || {
        let home = std::env::var_os("HOME").map(std::path::PathBuf::from);
        let kde = home
            .and_then(|p| std::fs::read_to_string(p.join(".config/kdeglobals")).ok())
            .is_some_and(|s| {
                s.lines().any(|line| {
                    line.trim()
                        .strip_prefix("AnimationDurationFactor=")
                        .and_then(|v| v.parse::<f32>().ok())
                        .is_some_and(|v| v == 0.0)
                })
            });
        let mut gnome = false;
        if let Ok(mut child) = std::process::Command::new("gsettings")
            .args(["get", "org.gnome.desktop.interface", "enable-animations"])
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .spawn()
        {
            let start = std::time::Instant::now();
            loop {
                if child.try_wait().ok().flatten().is_some() {
                    gnome = child.wait_with_output().ok().is_some_and(|o| {
                        o.status.success() && String::from_utf8_lossy(&o.stdout).trim() == "false"
                    });
                    break;
                }
                if start.elapsed() > std::time::Duration::from_millis(300) {
                    let _ = child.kill();
                    let _ = child.wait();
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
        }
        ctx.data_mut(|d| {
            d.insert_temp(Id::new("paintfe_reduced_motion"), kde || gnome);
            d.insert_temp(Id::new("paintfe_motion_os_busy"), false);
        });
        ctx.request_repaint();
    });
}
#[cfg(not(any(target_os = "windows", target_os = "linux")))]
fn update_system_motion(_ctx: &Context) {}

pub fn preferences(ui: &mut Ui, config: &mut PolishSettings) -> bool {
    let before = config.clone();
    ui.horizontal(|ui| {
        ui.label("Motion");
        for (mode, label) in [
            (MotionMode::Off, "Off"),
            (MotionMode::Subtle, "Subtle"),
            (MotionMode::Expressive, "Expressive"),
        ] {
            ui.selectable_value(&mut config.mode, mode, label);
        }
    });
    ui.checkbox(
        &mut config.respect_reduced_motion,
        "Respect system reduced motion",
    );
    ui.add(
        egui::Slider::new(&mut config.speed, 0.5..=2.0)
            .text("Motion speed")
            .suffix("×"),
    );
    Collapse::new("Motion and interaction details").show(ui, |ui| {
        for (value, label) in [
            (&mut config.hover_motion, "Hover and press"),
            (&mut config.selection_motion, "Tool and tab selection"),
            (&mut config.expansion_motion, "Expand and collapse"),
            (&mut config.overlay_motion, "Menus and dialogs"),
            (&mut config.row_motion, "History and layer rows"),
            (&mut config.confirmation_motion, "Copy confirmation"),
            (&mut config.snap_motion, "Panel settling"),
        ] {
            ui.checkbox(value, label);
        }
        ui.add(egui::Slider::new(&mut config.spring_strength, 0.0..=0.15).text("Spring strength"));
        ui.checkbox(
            &mut config.numeric_wheel,
            "Adjust numeric controls with the wheel",
        );
        ui.add(
            egui::Slider::new(&mut config.fine_adjustment, 0.01..=0.5)
                .text("Shift fine adjustment")
                .suffix("×"),
        );
        ui.label("Shift adjusts precisely; Alt bypasses panel snapping.");
    });
    Collapse::new("Shared spacing and typography").show(ui, |ui| {
        ui.add(
            egui::Slider::new(&mut config.spacing_scale, 0.8..=1.5)
                .text("Spacing")
                .suffix("×"),
        );
        ui.add(
            egui::Slider::new(&mut config.text_scale, 0.85..=1.35)
                .text("Text size")
                .suffix("×"),
        );
        ui.add(
            egui::Slider::new(&mut config.icon_gap, 2.0..=8.0)
                .text("Icon spacing")
                .suffix(" px"),
        );
        ui.add(
            egui::Slider::new(&mut config.border_width, 0.5..=2.0)
                .text("Control borders")
                .suffix(" px"),
        );
        ui.add(
            egui::Slider::new(&mut config.focus_width, 1.0..=3.0)
                .text("Keyboard focus outline")
                .suffix(" px"),
        );
    });
    if ui.button("Reset motion and interaction").clicked() {
        *config = PolishSettings::default();
    }
    config.sanitize();
    *config != before
}

#[cfg(test)]
mod polish_tests {
    use super::*;
    #[test]
    fn closing_popup_consumes_click_frame_then_returns_canvas_input() {
        let ctx = Context::default();
        let _ = ctx.run_ui(egui::RawInput::default(), |_| {
            egui::Popup::open_id(&ctx, Id::new("menu"));
            let was_open = egui::Popup::is_any_open(&ctx);
            egui::Popup::close_all(&ctx);
            remember_popup_input(&ctx, was_open);
            assert!(popup_blocks_input(&ctx));
        });
        let _ = ctx.run_ui(egui::RawInput::default(), |_| {
            remember_popup_input(&ctx, egui::Popup::is_any_open(&ctx));
            assert!(!popup_blocks_input(&ctx));
        });
    }
    fn frame(ctx: &Context, time: f64, target: f32) -> f32 {
        let mut result = 0.0;
        let _ = ctx.run_ui(
            egui::RawInput {
                time: Some(time),
                ..Default::default()
            },
            |_| {
                result = animate(ctx, Id::new("test"), target, MotionKind::Selection);
            },
        );
        result
    }
    #[test]
    fn retargets_without_jumps_and_settles_exactly() {
        let ctx = Context::default();
        assert_eq!(frame(&ctx, 0.0, 0.0), 0.0);
        assert_eq!(frame(&ctx, 0.01, 10.0), 0.0);
        let halfway = frame(&ctx, 0.07, 10.0);
        assert!(halfway > 0.0 && halfway < 10.0);
        assert_eq!(frame(&ctx, 0.08, -10.0), halfway);
        assert_eq!(frame(&ctx, 0.3, -10.0), -10.0);
        assert_eq!(frame(&ctx, 2.0, -10.0), -10.0);
    }
    #[test]
    fn off_reduced_and_categories_are_independent() {
        let mut config = PolishSettings {
            expansion_motion: false,
            ..Default::default()
        };
        assert_eq!(config.duration(MotionKind::Expansion, false), 0.0);
        assert!(config.duration(MotionKind::Overlay, false) > 0.0);
        assert_eq!(config.duration(MotionKind::Hover, true), 0.0);
        config.respect_reduced_motion = false;
        assert!(config.duration(MotionKind::Hover, true) > 0.0);
        config.mode = MotionMode::Off;
        assert_eq!(config.duration(MotionKind::Rows, false), 0.0);
        let ctx = Context::default();
        ctx.data_mut(|d| d.insert_temp(Id::new("paintfe_polish"), config));
        assert_eq!(frame(&ctx, 0.0, 0.0), 0.0);
        assert_eq!(frame(&ctx, 0.01, 100.0), 100.0);
    }
    #[test]
    fn old_partial_preferences_default_and_invalid_numbers_clamp() {
        let mut config: PolishSettings = serde_json::from_str(r#"{"mode":"Off"}"#).unwrap();
        assert_eq!(config.text_scale, 1.0);
        config.speed = f32::NAN;
        config.focus_width = 500.0;
        config.fine_adjustment = -1.0;
        config.sanitize();
        assert_eq!(config.speed, 1.0);
        assert_eq!(config.focus_width, 3.0);
        assert_eq!(config.fine_adjustment, 0.01);
    }
    #[test]
    fn restored_window_uses_outer_size_and_body_expansion_policy() {
        let ctx = Context::default();
        let config = PolishSettings {
            expansion_motion: false,
            ..Default::default()
        };
        ctx.data_mut(|d| d.insert_temp(Id::new("paintfe_polish"), config));
        let requested = egui::vec2(300.0, 200.0);
        for n in 0..3 {
            let _ = ctx.run_ui(
                egui::RawInput {
                    time: Some(n as f64),
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(1000.0, 800.0),
                    )),
                    ..Default::default()
                },
                |_| {
                    let result = window(&ctx, "restore test")
                        .title_bar(false)
                        .fixed_size(requested)
                        .default_pos(egui::pos2(100.0, 100.0))
                        .show(&ctx, |ui| {
                            assert_eq!(ui.style().animation_time, 0.0);
                            ui.label("Body");
                        })
                        .unwrap();
                    // egui's first sizing pass measures content before its Resize state exists.
                    if n > 0 {
                        assert!(
                            (result.response.rect.width() - requested.x).abs() < 1.0,
                            "frame {n}: {:?}",
                            result.response.rect.size()
                        );
                        assert!(
                            (result.response.rect.height() - requested.y).abs() < 1.0,
                            "frame {n}: {:?}",
                            result.response.rect.size()
                        );
                    }
                },
            );
        }
    }
    #[test]
    fn theme_switch_keeps_motion_and_density() {
        let mut theme = crate::theme::Theme::default();
        theme.polish.mode = MotionMode::Off;
        theme.toggle();
        assert_eq!(theme.polish.mode, MotionMode::Off);
        let next = theme.with_accent(theme.preset, theme.accent_colors);
        assert_eq!(next.polish, theme.polish);
    }
}
