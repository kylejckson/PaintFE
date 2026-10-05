//! Numeric editing with a shared wheel/fine-adjustment policy.
use egui::{Response, Ui, Widget, emath::Numeric as Number};
use std::ops::RangeInclusive;

pub fn step(ui: &Ui, value: f32) -> f32 {
    if ui.input(|i| i.modifiers.shift) {
        value * super::polish::settings(ui.ctx()).fine_adjustment
    } else {
        value
    }
}

pub fn slider_input(
    ui: &mut Ui,
    response: &Response,
    value: &mut f32,
    range: RangeInclusive<f32>,
    increment: f32,
) -> bool {
    if !ui.is_enabled() {
        return false;
    }
    let before = *value;
    let increment = step(ui, increment);
    if response.has_focus() {
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
                    *value += sign * increment;
                }
            }
            if i.consume_key(egui::Modifiers::NONE, egui::Key::Home) {
                *value = *range.start();
            }
            if i.consume_key(egui::Modifiers::NONE, egui::Key::End) {
                *value = *range.end();
            }
        });
    }
    if response.hovered()
        && super::polish::settings(ui.ctx()).numeric_wheel
        && !ui.input(|i| i.pointer.any_down())
    {
        let scroll: f32 = ui.input_mut(|i| {
            let amount = i
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
            if amount != 0.0 {
                i.events
                    .retain(|event| !matches!(event, egui::Event::MouseWheel { .. }));
                i.smooth_scroll_delta.y = 0.0;
            }
            amount
        });
        *value += scroll * increment;
    }
    *value = value.clamp(*range.start(), *range.end());
    super::polish::focus(ui, response, ui.visuals().widgets.inactive.corner_radius);
    *value != before
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
}
impl<'a, T: Number> Numeric<'a, T> {
    pub fn new(value: &'a mut T) -> Self {
        Self {
            value,
            speed: 1.0,
            range: f64::NEG_INFINITY..=f64::INFINITY,
            suffix: String::new(),
            prefix: String::new(),
            decimals: None,
            min_decimals: 0,
            clamp: true,
            live: true,
        }
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
    fn ui(self, ui: &mut Ui) -> Response {
        let settings = super::polish::settings(ui.ctx());
        let id = ui.next_auto_id();
        let shift = ui.input(|i| i.modifiers.shift);
        // egui already multiplies drag speed by 0.1 while Shift is held.
        let speed = if shift && ui.ctx().is_being_dragged(id) {
            self.speed * settings.fine_adjustment as f64 / 0.1
        } else {
            self.speed
        };
        let before = self.value.to_f64();
        let mut widget = egui::DragValue::new(&mut *self.value)
            .speed(speed)
            .range(self.range.clone())
            .suffix(self.suffix)
            .prefix(self.prefix)
            .min_decimals(self.min_decimals)
            .clamp_existing_to_range(self.clamp)
            .update_while_editing(self.live);
        if let Some(decimals) = self.decimals {
            widget = widget.max_decimals(decimals);
        }
        let mut response = ui.add(widget);
        if ui.is_enabled()
            && settings.numeric_wheel
            && response.hovered()
            && !response.has_focus()
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
        if response.gained_focus() {
            ui.ctx().data_mut(|d| d.insert_temp(edit_id, before));
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
            ui.ctx().data_mut(|d| d.remove_temp::<f64>(edit_id));
        }
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
            rect = ui.add(Numeric::new(&mut value).range(0.0..=10.0)).rect;
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
                    ui.add_enabled(enabled, Numeric::new(&mut value).range(0.0..=10.0));
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
                ui.add(Numeric::new(&mut value).range(0.0..=10.0));
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
            rect = ui.add(Numeric::new(&mut value).range(0.0..=10.0)).rect;
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
                ui.add(Numeric::new(&mut value).range(0.0..=10.0));
            },
        );
        let _ = ctx.run_ui(
            egui::RawInput {
                events: vec![egui::Event::Text("8".into())],
                ..Default::default()
            },
            |ui| {
                ui.add(Numeric::new(&mut value).range(0.0..=10.0));
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
                ui.add(Numeric::new(&mut value).range(0.0..=10.0));
            },
        );
        assert_eq!(value, 5.0);
        let _ = ctx.run_ui(
            egui::RawInput {
                events: vec![click(true), click(false)],
                ..Default::default()
            },
            |ui| {
                ui.add(Numeric::new(&mut value).range(0.0..=10.0));
            },
        );
        let _ = ctx.run_ui(
            egui::RawInput {
                events: vec![egui::Event::Text("7".into())],
                ..Default::default()
            },
            |ui| {
                ui.add(Numeric::new(&mut value).range(0.0..=10.0));
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
                ui.add(Numeric::new(&mut value).range(0.0..=10.0));
            },
        );
        assert_eq!(value, 7.0);
    }
    #[test]
    fn custom_slider_keyboard_fine_home_end() {
        let ctx = egui::Context::default();
        let mut value = 5.0;
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
