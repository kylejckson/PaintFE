//! Numeric editing with a shared wheel/fine-adjustment policy.
use egui::{Response, Ui, Widget, emath::Numeric as Number};
use std::ops::RangeInclusive;
mod expression;
pub(crate) use expression::evaluate_dimension_expression;
pub(crate) type Formatter<'a> = Box<dyn Fn(f64, RangeInclusive<usize>) -> String + 'a>;
pub(crate) type Parser<'a> = Box<dyn Fn(&str) -> Option<f64> + 'a>;

pub fn step(ui: &Ui, value: f32) -> f32 {
    if ui.input(|i| i.modifiers.shift) {
        value * super::polish::settings(ui.ctx()).fine_adjustment
    } else {
        value
    }
}

pub fn slider_input<T: Number>(
    ui: &mut Ui,
    response: &Response,
    value: &mut T,
    range: RangeInclusive<T>,
    increment: f32,
) -> bool {
    slider_policy(ui, response, value, range, increment as f64)
}

fn slider_policy<T: Number>(
    ui: &mut Ui,
    response: &Response,
    value: &mut T,
    range: RangeInclusive<T>,
    increment: f64,
) -> bool {
    if !ui.is_enabled() {
        return false;
    }
    let before = value.to_f64();
    let mut next = before;
    let increment = step(ui, increment as f32) as f64;
    let increment = if T::INTEGRAL {
        increment.max(1.0)
    } else {
        increment
    };
    if response.clicked() {
        response.request_focus();
    }
    if response.has_focus() {
        ui.memory_mut(|m| {
            m.set_focus_lock_filter(
                response.id,
                egui::EventFilter {
                    horizontal_arrows: true,
                    vertical_arrows: true,
                    ..Default::default()
                },
            )
        });
        ui.input_mut(|i| {
            for (key, sign) in [
                (egui::Key::ArrowLeft, -1.0),
                (egui::Key::ArrowDown, -1.0),
                (egui::Key::ArrowRight, 1.0),
                (egui::Key::ArrowUp, 1.0),
            ] {
                if i.consume_key(egui::Modifiers::NONE, key)
                    || i.consume_key(egui::Modifiers::SHIFT, key)
                {
                    next += sign * increment;
                }
            }
            if i.consume_key(egui::Modifiers::NONE, egui::Key::Home) {
                next = range.start().to_f64();
            }
            if i.consume_key(egui::Modifiers::NONE, egui::Key::End) {
                next = range.end().to_f64();
            }
        });
    }
    if response.hovered()
        && super::controls::wheel_adjustment_allowed(ui)
        && super::polish::settings(ui.ctx()).numeric_wheel
        && !ui.input(|i| i.pointer.any_down())
    {
        let scroll: f32 = ui.input_mut(|i| {
            let amount = i
                .events
                .iter()
                .filter_map(|event| match event {
                    egui::Event::MouseWheel { delta, .. } => Some(delta.y.signum()),
                    _ => None,
                })
                .sum();
            if amount != 0.0 {
                i.events
                    .retain(|event| !matches!(event, egui::Event::MouseWheel { .. }));
                i.smooth_scroll_delta.y = 0.0;
            }
            amount
        });
        next += scroll as f64 * increment;
    }
    // Do not clamp an untouched external model value (SliderClamping::Edits).
    if next != before {
        *value = T::from_f64(next.clamp(range.start().to_f64(), range.end().to_f64()));
    }
    super::polish::focus(ui, response, ui.visuals().widgets.inactive.corner_radius);
    value.to_f64() != before
}

pub struct Numeric<'a, T: Number> {
    value: &'a mut T,
    speed: f64,
    range: RangeInclusive<f64>,
    suffix: String,
    prefix: String,
    decimals: Option<usize>,
    min_decimals: usize,
    clamp: bool,
    live: bool,
    expressions: bool,
    id_salt: egui::Id,
    buttons: Option<f64>,
    reset: Option<T>,
    formatter: Option<Formatter<'a>>,
    parser: Option<Parser<'a>>,
    focus_on_open: bool,
}
impl<'a, T: Number> Numeric<'a, T> {
    #[track_caller]
    pub fn new(value: &'a mut T) -> Self {
        let source = std::panic::Location::caller();
        Self {
            id_salt: egui::Id::new((source.file(), source.line(), source.column())),
            buttons: None,
            reset: None,
            formatter: None,
            parser: None,
            focus_on_open: false,
            value,
            speed: 1.0,
            range: f64::NEG_INFINITY..=f64::INFINITY,
            suffix: String::new(),
            prefix: String::new(),
            decimals: None,
            min_decimals: 0,
            clamp: true,
            live: true,
            expressions: false,
        }
    }
    /// Override the call-site ID for repeated controls; use a stable model key.
    pub fn id_salt(mut self, salt: impl egui::AsId) -> Self {
        self.id_salt = egui::Id::new(salt);
        self
    }
    /// Auto-focus and select the existing value when a dialog opens.
    pub fn focus_on_open(mut self, focus: bool) -> Self {
        self.focus_on_open = focus;
        self
    }
    pub fn steppers(mut self, step: f64) -> Self {
        self.buttons = Some(step.abs());
        self
    }
    pub fn reset_to(mut self, default: T) -> Self {
        self.reset = Some(default);
        self.buttons.get_or_insert(self.speed.abs());
        self
    }
    pub fn custom_formatter(
        mut self,
        formatter: impl Fn(f64, RangeInclusive<usize>) -> String + 'a,
    ) -> Self {
        self.formatter = Some(Box::new(formatter));
        self
    }
    pub fn custom_parser(mut self, parser: impl Fn(&str) -> Option<f64> + 'a) -> Self {
        self.parser = Some(Box::new(parser));
        self
    }
    /// Accept arithmetic while retaining the same drag, focus and wheel behaviour.
    pub fn expressions(mut self) -> Self {
        self.expressions = true;
        self
    }
    pub fn speed(mut self, speed: impl Into<f64>) -> Self {
        self.speed = speed.into();
        self
    }
    pub fn range<N: Number>(mut self, range: RangeInclusive<N>) -> Self {
        self.range = range.start().to_f64()..=range.end().to_f64();
        self
    }
    pub fn suffix(mut self, suffix: impl Into<String>) -> Self {
        self.suffix = suffix.into();
        self
    }
    pub fn prefix(mut self, prefix: impl Into<String>) -> Self {
        self.prefix = prefix.into();
        self
    }
    pub fn max_decimals(mut self, decimals: usize) -> Self {
        self.decimals = Some(decimals);
        self
    }
    pub fn min_decimals(mut self, decimals: usize) -> Self {
        self.min_decimals = decimals;
        self
    }
    pub fn fixed_decimals(mut self, decimals: usize) -> Self {
        self.min_decimals = decimals;
        self.decimals = Some(decimals);
        self
    }
    pub fn clamp_existing_to_range(mut self, clamp: bool) -> Self {
        self.clamp = clamp;
        self
    }
    pub fn update_while_editing(mut self, live: bool) -> Self {
        self.live = live;
        self
    }
}
impl<T: Number> Widget for Numeric<'_, T> {
    fn ui(mut self, ui: &mut Ui) -> Response {
        ui.scope_builder(
            egui::UiBuilder::new().id(ui.make_persistent_id(("control", self.id_salt))),
            |ui| {
                if let Some(increment) = self.buttons {
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 2.0;
                        let mut changed = false;
                        let amount = if T::INTEGRAL {
                            1.0_f64.max(step(ui, increment as f32) as f64)
                        } else {
                            step(ui, increment as f32) as f64
                        };
                        if crate::ui::controls::stepper(ui, false).clicked() {
                            *self.value = T::from_f64(
                                (self.value.to_f64() - amount)
                                    .clamp(*self.range.start(), *self.range.end()),
                            );
                            changed = true;
                        }
                        let default = self.reset;
                        let mut response = self.show(ui);
                        // `show` borrows the value only for its own lifetime.
                        if crate::ui::controls::stepper(ui, true).clicked() {
                            *self.value = T::from_f64(
                                (self.value.to_f64() + amount)
                                    .clamp(*self.range.start(), *self.range.end()),
                            );
                            changed = true;
                        }
                        if let Some(default) = default
                            && crate::ui::controls::reset(
                                ui,
                                self.value.to_f64() != default.to_f64(),
                            )
                            .clicked()
                        {
                            *self.value = default;
                            changed = true;
                        }
                        if changed {
                            response.mark_changed();
                        }
                        response
                    })
                    .inner
                } else {
                    self.show(ui)
                }
            },
        )
        .inner
    }
}

impl<T: Number> Numeric<'_, T> {
    fn show(&mut self, ui: &mut Ui) -> Response {
        let settings = super::polish::settings(ui.ctx());
        let id = ui.next_auto_id();
        let frame = ui.ctx().cumulative_frame_nr();
        ui.data_mut(|d| {
            let last = d.get_temp::<u64>(id.with("numeric_last_frame"));
            if last.is_some_and(|last| frame > last + 1) {
                // A closed popup/collapsed section cannot report lost_focus.
                // Reopening must start a new edit transaction.
                d.remove::<f64>(id.with("numeric_original"));
                d.remove::<bool>(id.with("numeric_select_all"));
            }
            d.insert_temp(id.with("numeric_last_frame"), frame);
        });
        let shift = ui.input(|i| i.modifiers.shift);
        // egui already multiplies drag speed by 0.1 while Shift is held.
        let speed = if shift && ui.ctx().is_being_dragged(id) {
            self.speed * settings.fine_adjustment as f64 / 0.1
        } else {
            self.speed
        };
        let before = self.value.to_f64();
        // Preserve the native Ctrl+A fallback previously implemented only by
        // New Image. Every focused numeric editor now receives the same policy.
        #[cfg(target_os = "windows")]
        if ui.memory(|m| m.has_focus(id)) {
            let probe = crate::windows_key_probe::snapshot();
            let down = probe.ctrl_down && crate::windows_key_probe::is_vk_down(0x41);
            let previous =
                ui.data_mut(|d| d.get_temp::<bool>(id.with("ctrl_a_down")).unwrap_or(false));
            if down && !previous && !ui.input(|i| i.key_pressed(egui::Key::A)) {
                ui.input_mut(|i| {
                    i.events.push(egui::Event::Key {
                        key: egui::Key::A,
                        physical_key: None,
                        pressed: true,
                        repeat: false,
                        modifiers: egui::Modifiers {
                            ctrl: true,
                            command: true,
                            ..Default::default()
                        },
                    })
                });
            }
            ui.data_mut(|d| d.insert_temp(id.with("ctrl_a_down"), down));
        }
        let value_id = id.with("numeric_rendered_value");
        let externally_changed = ui
            .ctx()
            .data(|d| d.get_temp::<f64>(value_id))
            .is_some_and(|previous| previous != before);
        if externally_changed && ui.memory(|m| m.has_focus(id)) {
            // A linked field (for example resize Width) changed this value.
            // Discard the old edit buffer before it can overwrite the new
            // Height on blur, and return to the current numeric display.
            if !self.focus_on_open {
                ui.memory_mut(|m| m.surrender_focus(id));
            }
            ui.data_mut(|d| d.remove::<String>(id));
        }
        let mut widget = egui::DragValue::new(&mut *self.value)
            .speed(speed)
            .range(self.range.clone())
            .suffix(self.suffix.clone())
            .prefix(self.prefix.clone())
            .min_decimals(self.min_decimals)
            .clamp_existing_to_range(self.clamp)
            .update_while_editing(self.live);
        if self.expressions {
            widget =
                widget.custom_parser(|text| evaluate_dimension_expression(text).map(f64::from));
        }
        if let Some(formatter) = self.formatter.take() {
            widget = widget.custom_formatter(formatter);
        }
        if let Some(parser) = self.parser.take() {
            widget = widget.custom_parser(parser);
        }
        if let Some(decimals) = self.decimals {
            widget = widget.max_decimals(decimals);
        }
        let mut response = ui.add(widget);
        if self.focus_on_open {
            // Request after registration: windows may still be in their sizing
            // pass before the widget has been registered as a focus candidate.
            response.request_focus();
            ui.data_mut(|d| {
                d.remove::<String>(id);
                d.remove::<f64>(id.with("numeric_original"));
                d.insert_temp(id.with("numeric_select_all"), true);
            });
            ui.ctx().request_repaint();
        }
        if response.has_focus()
            && ui.data(|d| {
                d.get_temp::<bool>(id.with("numeric_select_all"))
                    .unwrap_or(false)
            })
            && let Some(text) = ui.data(|d| d.get_temp::<String>(id))
        {
            let mut state = egui::TextEdit::load_state(ui.ctx(), id).unwrap_or_default();
            state
                .cursor
                .set_char_range(Some(egui::text::CCursorRange::two(
                    egui::text::CCursor::default(),
                    egui::text::CCursor::new(text.chars().count()),
                )));
            state.store(ui.ctx(), id);
            ui.data_mut(|d| d.remove::<bool>(id.with("numeric_select_all")));
            ui.ctx().request_repaint();
        }
        #[cfg(test)]
        ui.ctx()
            .data_mut(|d| d.insert_temp(egui::Id::new("control_test_editor"), response.clone()));
        if !response.has_focus() {
            response = response.on_hover_cursor(egui::CursorIcon::ResizeHorizontal);
        }
        if ui.is_enabled()
            && settings.numeric_wheel
            && response.hovered()
            && !response.has_focus()
            && super::controls::wheel_adjustment_allowed(ui)
            && !ui.input(|i| i.pointer.any_down())
        {
            let scroll = ui.input_mut(|i| {
                let scroll: f32 = i
                    .events
                    .iter()
                    .filter_map(|event| {
                        if let egui::Event::MouseWheel { delta, .. } = event {
                            Some(delta.y.signum())
                        } else {
                            None
                        }
                    })
                    .sum();
                if scroll != 0.0 {
                    i.events
                        .retain(|event| !matches!(event, egui::Event::MouseWheel { .. }));
                    i.smooth_scroll_delta.y = 0.0;
                }
                scroll
            });
            if scroll != 0.0 {
                let factor = if shift {
                    settings.fine_adjustment as f64
                } else {
                    1.0
                };
                let amount = if T::INTEGRAL {
                    (self.speed * factor).abs().max(1.0)
                } else {
                    self.speed * factor
                };
                let next = (self.value.to_f64() + scroll as f64 * amount)
                    .clamp(*self.range.start(), *self.range.end());
                *self.value = T::from_f64(next);
                response.mark_changed();
            }
        }
        // Escape restores the value captured on entering text editing.
        let edit_id = id.with("numeric_original");
        if response.has_focus() || response.gained_focus() {
            // Focus may be requested before the frame (keyboard navigation,
            // accessibility or opening a dialog), without gained_focus here.
            // Capture once per edit session, never again after a live text edit.
            ui.ctx().data_mut(|d| {
                if d.get_temp::<f64>(edit_id).is_none() {
                    d.insert_temp(edit_id, before);
                }
            });
        }
        if ui.input(|i| i.key_pressed(egui::Key::Escape))
            && (response.has_focus() || response.lost_focus())
        {
            if let Some(original) = ui.ctx().data_mut(|d| d.remove_temp::<f64>(edit_id)) {
                *self.value = T::from_f64(original);
                response.mark_changed();
            }
            response.surrender_focus();
        } else if response.lost_focus() {
            ui.ctx().data_mut(|d| {
                d.remove_temp::<f64>(edit_id);
                d.remove::<bool>(id.with("numeric_select_all"));
            });
        }
        ui.ctx()
            .data_mut(|d| d.insert_temp(value_id, self.value.to_f64()));
        super::polish::focus(
            ui,
            &response,
            ui.visuals().widgets.inactive.corner_radius.nw as f32,
        );
        response
    }
}

#[cfg(test)]
mod polish_tests {
    use super::*;
    #[test]
    fn issue_185_shared_field_drags_and_accepts_expressions_with_tab_and_escape() {
        fn frame(
            ctx: &egui::Context,
            value: &mut f32,
            events: Vec<egui::Event>,
        ) -> (egui::FullOutput, egui::Response) {
            let mut response = None;
            let output = ctx.run_ui(
                egui::RawInput {
                    events,
                    ..Default::default()
                },
                |ui| {
                    response = Some(
                        ui.add(
                            Numeric::new(value)
                                .id_salt("test_value")
                                .expressions()
                                .range(1.0..=20000.0)
                                .fixed_decimals(0)
                                .update_while_editing(false),
                        ),
                    );
                },
            );
            (output, response.unwrap())
        }
        let ctx = egui::Context::default();
        let mut value = 100.0_f32;
        let (_, response) = frame(&ctx, &mut value, vec![]);
        let center = response.rect.center();
        let pointer = |pos, pressed| egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::NONE,
        };
        let key = |key| egui::Event::Key {
            key,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        };
        let (output, _) = frame(&ctx, &mut value, vec![egui::Event::PointerMoved(center)]);
        assert_eq!(
            output.platform_output.cursor_icon,
            egui::CursorIcon::ResizeHorizontal
        );
        frame(&ctx, &mut value, vec![pointer(center, true)]);
        frame(
            &ctx,
            &mut value,
            vec![egui::Event::PointerMoved(center + egui::vec2(32.0, 0.0))],
        );
        frame(
            &ctx,
            &mut value,
            vec![egui::Event::PointerMoved(center + egui::vec2(48.0, 0.0))],
        );
        frame(
            &ctx,
            &mut value,
            vec![pointer(center + egui::vec2(48.0, 0.0), false)],
        );
        assert!(
            value > 100.0 && value < 200.0,
            "numeric drag was not incremental: {value}"
        );
        ctx.memory_mut(|m| m.request_focus(response.id));
        frame(&ctx, &mut value, vec![]);
        frame(
            &ctx,
            &mut value,
            vec![egui::Event::Key {
                key: egui::Key::A,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: egui::Modifiers {
                    ctrl: true,
                    command: true,
                    ..Default::default()
                },
            }],
        );
        frame(&ctx, &mut value, vec![egui::Event::Text("800/2".into())]);
        frame(&ctx, &mut value, vec![key(egui::Key::Tab)]);
        assert_eq!(value, 400.0);
        ctx.memory_mut(|m| m.request_focus(response.id));
        frame(&ctx, &mut value, vec![]);
        frame(
            &ctx,
            &mut value,
            vec![egui::Event::Key {
                key: egui::Key::A,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: egui::Modifiers {
                    ctrl: true,
                    command: true,
                    ..Default::default()
                },
            }],
        );
        frame(&ctx, &mut value, vec![egui::Event::Text("99".into())]);
        frame(&ctx, &mut value, vec![key(egui::Key::Escape)]);
        assert_eq!(value, 400.0);
    }

    fn wheel(delta: f32) -> egui::Event {
        egui::Event::MouseWheel {
            unit: egui::MouseWheelUnit::Line,
            delta: egui::vec2(0.0, delta),
            modifiers: egui::Modifiers::NONE,
            phase: egui::TouchPhase::Move,
        }
    }
    #[test]
    fn wheel_once_fine_clamped_and_disabled() {
        let ctx = egui::Context::default();
        let mut value = 5.0f32;
        let mut rect = egui::Rect::NOTHING;
        let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
            rect = ui
                .add(
                    Numeric::new(&mut value)
                        .id_salt("test_value")
                        .range(0.0..=10.0),
                )
                .rect;
        });
        for (shift, enabled, delta, expected) in [
            (false, true, 1.0, 6.0),
            (true, true, -1.0, 5.9),
            (false, false, 1.0, 5.9),
        ] {
            let _ = ctx.run_ui(
                egui::RawInput {
                    modifiers: egui::Modifiers {
                        shift,
                        ..Default::default()
                    },
                    events: vec![egui::Event::PointerMoved(rect.center()), wheel(delta)],
                    ..Default::default()
                },
                |ui| {
                    ui.add_enabled(
                        enabled,
                        Numeric::new(&mut value)
                            .id_salt("test_value")
                            .range(0.0..=10.0),
                    );
                },
            );
            assert!((value - expected).abs() < 0.001, "{value} vs {expected}");
        }
        value = 10.0;
        let _ = ctx.run_ui(
            egui::RawInput {
                events: vec![wheel(1.0)],
                ..Default::default()
            },
            |ui| {
                ui.add(
                    Numeric::new(&mut value)
                        .id_salt("test_value")
                        .range(0.0..=10.0),
                );
            },
        );
        assert_eq!(value, 10.0);
    }
    #[test]
    fn typing_escape_restores_and_enter_commits() {
        let ctx = egui::Context::default();
        let mut value = 5.0f32;
        let mut rect = egui::Rect::NOTHING;
        let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
            rect = ui
                .add(
                    Numeric::new(&mut value)
                        .id_salt("test_value")
                        .range(0.0..=10.0),
                )
                .rect;
        });
        let click = |pressed| egui::Event::PointerButton {
            pos: rect.center(),
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::NONE,
        };
        let _ = ctx.run_ui(
            egui::RawInput {
                events: vec![
                    egui::Event::PointerMoved(rect.center()),
                    click(true),
                    click(false),
                ],
                ..Default::default()
            },
            |ui| {
                ui.add(
                    Numeric::new(&mut value)
                        .id_salt("test_value")
                        .range(0.0..=10.0),
                );
            },
        );
        let _ = ctx.run_ui(
            egui::RawInput {
                events: vec![egui::Event::Text("8".into())],
                ..Default::default()
            },
            |ui| {
                ui.add(
                    Numeric::new(&mut value)
                        .id_salt("test_value")
                        .range(0.0..=10.0),
                );
            },
        );
        assert_eq!(value, 8.0);
        let escape = egui::Event::Key {
            key: egui::Key::Escape,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        };
        let _ = ctx.run_ui(
            egui::RawInput {
                events: vec![escape],
                ..Default::default()
            },
            |ui| {
                ui.add(
                    Numeric::new(&mut value)
                        .id_salt("test_value")
                        .range(0.0..=10.0),
                );
            },
        );
        assert_eq!(value, 5.0);
        let _ = ctx.run_ui(
            egui::RawInput {
                events: vec![click(true), click(false)],
                ..Default::default()
            },
            |ui| {
                ui.add(
                    Numeric::new(&mut value)
                        .id_salt("test_value")
                        .range(0.0..=10.0),
                );
            },
        );
        let _ = ctx.run_ui(
            egui::RawInput {
                events: vec![egui::Event::Text("7".into())],
                ..Default::default()
            },
            |ui| {
                ui.add(
                    Numeric::new(&mut value)
                        .id_salt("test_value")
                        .range(0.0..=10.0),
                );
            },
        );
        let enter = egui::Event::Key {
            key: egui::Key::Enter,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        };
        let _ = ctx.run_ui(
            egui::RawInput {
                events: vec![enter],
                ..Default::default()
            },
            |ui| {
                ui.add(
                    Numeric::new(&mut value)
                        .id_salt("test_value")
                        .range(0.0..=10.0),
                );
            },
        );
        assert_eq!(value, 7.0);
    }
    #[test]
    fn custom_slider_keyboard_fine_home_end() {
        let ctx = egui::Context::default();
        let mut value = 5.0_f32;
        let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
            let response = ui
                .allocate_exact_size(egui::vec2(100.0, 24.0), egui::Sense::click_and_drag())
                .1;
            response.request_focus();
        });
        for (key, shift, expected) in [
            (egui::Key::ArrowRight, true, 5.1),
            (egui::Key::End, false, 10.0),
            (egui::Key::Home, false, 0.0),
        ] {
            let modifiers = egui::Modifiers {
                shift,
                ..Default::default()
            };
            let _ = ctx.run_ui(
                egui::RawInput {
                    modifiers,
                    events: vec![egui::Event::Key {
                        key,
                        physical_key: None,
                        pressed: true,
                        repeat: false,
                        modifiers,
                    }],
                    ..Default::default()
                },
                |ui| {
                    let response = ui
                        .allocate_exact_size(egui::vec2(100.0, 24.0), egui::Sense::click_and_drag())
                        .1;
                    slider_input(ui, &response, &mut value, 0.0..=10.0, 1.0);
                },
            );
            assert!((value - expected).abs() < 0.001, "{value} vs {expected}");
        }
    }
}

/// Named presets configure the same editor rather than implementing new controls.
impl<'a> Numeric<'a, f32> {
    #[track_caller]
    pub fn dimension(value: &'a mut f32) -> Self {
        Self::new(value)
            .expressions()
            .range(1.0..=20000.0)
            .fixed_decimals(0)
            .update_while_editing(false)
            .suffix(" px")
            .steppers(1.0)
    }
    #[track_caller]
    pub fn percentage(value: &'a mut f32) -> Self {
        Self::new(value)
            .range(0.0..=100.0)
            .suffix("%")
            .speed(1.0)
            .max_decimals(1)
    }
    #[track_caller]
    pub fn angle(value: &'a mut f32) -> Self {
        Self::new(value)
            .range(-180.0..=180.0)
            .suffix("°")
            .speed(1.0)
            .max_decimals(1)
    }
}
