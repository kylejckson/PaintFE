use egui::{Response, Ui, Widget, WidgetText, emath::Numeric as Number};
use std::ops::RangeInclusive;

use super::Numeric;
use crate::ui::numeric::{Formatter, Parser};

/// Native settings appearance, with PaintFE's shared editor instead of egui's
/// hidden second implementation. Specialised tracks use `numeric::slider_input`.
pub struct Slider<'a, T: Number> {
    value: &'a mut T,
    range: RangeInclusive<T>,
    id: egui::Id,
    step: Option<f64>,
    decimals: Option<usize>,
    min_decimals: usize,
    suffix: String,
    text: WidgetText,
    show_value: bool,
    logarithmic: bool,
    clamping: egui::SliderClamping,
    formatter: Option<Formatter<'a>>,
    parser: Option<Parser<'a>>,
    buttons: bool,
    default: Option<T>,
    reset_button: bool,
    reset_action: Option<Box<dyn FnMut() + 'a>>,
}

impl<'a, T: Number> Slider<'a, T> {
    #[track_caller]
    pub fn new(value: &'a mut T, range: RangeInclusive<T>) -> Self {
        let source = std::panic::Location::caller();
        Self {
            value,
            range,
            id: egui::Id::new((source.file(), source.line(), source.column())),
            step: None,
            decimals: None,
            min_decimals: 0,
            suffix: String::new(),
            text: WidgetText::default(),
            show_value: true,
            logarithmic: false,
            clamping: egui::SliderClamping::Always,
            formatter: None,
            parser: None,
            buttons: false,
            default: None,
            reset_button: false,
            reset_action: None,
        }
    }
    pub fn id_salt(mut self, id: impl egui::AsId) -> Self {
        self.id = egui::Id::new(id);
        self
    }
    pub fn step_by(mut self, step: f64) -> Self {
        self.step = Some(step.abs());
        self
    }
    pub fn max_decimals(mut self, places: usize) -> Self {
        self.decimals = Some(places);
        self
    }
    pub fn fixed_decimals(mut self, places: usize) -> Self {
        self.decimals = Some(places);
        self.min_decimals = places;
        self
    }
    pub fn suffix(mut self, suffix: impl ToString) -> Self {
        self.suffix = suffix.to_string();
        self
    }
    pub fn text(mut self, text: impl Into<WidgetText>) -> Self {
        self.text = text.into();
        self
    }
    pub fn show_value(mut self, show: bool) -> Self {
        self.show_value = show;
        self
    }
    pub fn logarithmic(mut self, enabled: bool) -> Self {
        self.logarithmic = enabled;
        self
    }
    pub fn clamping(mut self, clamping: egui::SliderClamping) -> Self {
        self.clamping = clamping;
        self
    }
    pub fn steppers(mut self, enabled: bool) -> Self {
        self.buttons = enabled;
        self
    }
    pub fn reset_to(mut self, default: T) -> Self {
        self.default = Some(default);
        self.reset_button = true;
        self
    }
    /// Add the track gesture while retaining a dialog's existing header reset.
    pub fn double_click_reset_to(mut self, default: T) -> Self {
        self.default = Some(default);
        self
    }
    /// Optional model overrides reset to inheritance rather than a numeric value.
    pub fn on_double_click_reset(mut self, action: impl FnMut() + 'a) -> Self {
        self.reset_action = Some(Box::new(action));
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
}

impl<T: Number> Widget for Slider<'_, T> {
    fn ui(mut self, ui: &mut Ui) -> Response {
        ui.scope_builder(
            egui::UiBuilder::new().id(ui.make_persistent_id(("control", self.id))),
            |ui| {
                ui.horizontal(|ui| {
                    let before = self.value.to_f64();
                    let low = self.range.start().to_f64();
                    let high = self.range.end().to_f64();
                    let increment = self.step.unwrap_or_else(|| {
                        if T::INTEGRAL {
                            1.0
                        } else {
                            ((high - low).abs() / 100.0).max(0.001)
                        }
                    });
                    let adjusted = if T::INTEGRAL {
                        1.0_f64.max(crate::ui::numeric::step(ui, increment as f32) as f64)
                    } else {
                        crate::ui::numeric::step(ui, increment as f32) as f64
                    };
                    // Native pointer/accessibility behaviour is retained. Route
                    // keyboard stepping through one policy, avoiding double edits.
                    let track_id_key = ui.make_persistent_id("native_track");
                    let focused = ui
                        .ctx()
                        .data(|d| d.get_temp::<egui::Id>(track_id_key))
                        .is_some_and(|id| ui.memory(|m| m.has_focus(id)));
                    let mut keys = Vec::new();
                    if focused {
                        ui.input_mut(|i| {
                            i.events.retain(|event| {
                                let stepping = matches!(
                                    event,
                                    egui::Event::Key {
                                        key: egui::Key::ArrowLeft
                                            | egui::Key::ArrowRight
                                            | egui::Key::ArrowUp
                                            | egui::Key::ArrowDown
                                            | egui::Key::Home
                                            | egui::Key::End,
                                        ..
                                    }
                                );
                                if stepping {
                                    keys.push(event.clone());
                                }
                                !stepping
                            })
                        });
                    }
                    if self.clamping == egui::SliderClamping::Always {
                        *self.value = T::from_f64(self.value.to_f64().clamp(low, high));
                    }
                    let mut native = egui::Slider::new(&mut *self.value, self.range.clone())
                        .show_value(false)
                        .logarithmic(self.logarithmic)
                        .clamping(if self.clamping == egui::SliderClamping::Always {
                            egui::SliderClamping::Edits
                        } else {
                            self.clamping
                        });
                    if self.step.is_some() || T::INTEGRAL {
                        native = native.step_by(adjusted);
                    }
                    let mut track = ui.add(native);
                    if self.default.is_some() || self.reset_action.is_some() {
                        track = track.interact(egui::Sense::click());
                    }
                    ui.ctx().data_mut(|d| d.insert_temp(track_id_key, track.id));
                    ui.input_mut(|i| i.events.extend(keys));
                    #[cfg(test)]
                    ui.ctx().data_mut(|d| {
                        d.insert_temp(egui::Id::new("control_test_track"), track.clone())
                    });
                    crate::ui::numeric::slider_input(
                        ui,
                        &track,
                        self.value,
                        self.range.clone(),
                        increment as f32,
                    );
                    if self.default.is_some() || self.reset_action.is_some() {
                        track = track.on_hover_text(crate::t!("ui.double_click_reset_hint"));
                        if ui.is_enabled()
                            && super::super::polish::settings(ui.ctx()).double_click_slider_reset
                            && track.double_clicked()
                        {
                            if let Some(action) = &mut self.reset_action {
                                action();
                            } else if let Some(default) = self.default {
                                *self.value = default;
                            }
                            track.mark_changed();
                        }
                    }
                    if self.show_value {
                        let mut editor = Numeric::new(&mut *self.value)
                            .id_salt("value")
                            .speed(increment)
                            .suffix(self.suffix)
                            .min_decimals(self.min_decimals);
                        editor = match self.clamping {
                            egui::SliderClamping::Never => editor,
                            egui::SliderClamping::Edits => editor
                                .range(self.range.clone())
                                .clamp_existing_to_range(false),
                            egui::SliderClamping::Always => editor.range(self.range.clone()),
                        };
                        if let Some(decimals) = self.decimals {
                            editor = editor.max_decimals(decimals);
                        }
                        if self.buttons {
                            editor = editor.steppers(increment);
                        }
                        if let Some(default) = self.default.filter(|_| self.reset_button) {
                            editor = editor.reset_to(default);
                        }
                        if let Some(formatter) = self.formatter.take() {
                            editor = editor.custom_formatter(formatter);
                        }
                        if let Some(parser) = self.parser.take() {
                            editor = editor.custom_parser(parser);
                        }
                        let response = ui.add(editor);
                        track = if response.has_focus()
                            || response.gained_focus()
                            || response.lost_focus()
                        {
                            response.union(track)
                        } else {
                            track.union(response)
                        };
                    }
                    if !self.text.is_empty() {
                        let label = ui.label(self.text);
                        track = track.union(label);
                    }
                    if self.value.to_f64() != before {
                        track.mark_changed();
                    }
                    track
                })
                .inner
            },
        )
        .inner
    }
}
