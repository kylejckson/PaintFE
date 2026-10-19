//! Approved numeric controls. See docs/ui-controls.md before adding a control.
mod color;
mod gradient;
mod interaction;
mod row;
mod slider;

pub use super::numeric::Numeric;
pub use color::{color_edit_button_rgb, color_edit_button_srgb, color_edit_button_srgba};
pub use gradient::GradientSlider;
pub(crate) use interaction::drag_motion;

#[cfg(test)]
mod tests;
pub use row::ValueRow;
pub use slider::Slider;

const NO_WHEEL_ADJUSTMENT: &str = "paintfe_no_numeric_wheel";

/// Keep the wheel available to a scrolling container without changing the
/// user's global numeric-wheel preference or the controls' other interactions.
pub(crate) fn without_wheel_adjustment<R>(
    ui: &mut egui::Ui,
    contents: impl FnOnce(&mut egui::Ui) -> R,
) -> egui::InnerResponse<R> {
    ui.scope_builder(
        egui::UiBuilder::new()
            .ui_stack_info(egui::UiStackInfo::default().with_tag(NO_WHEEL_ADJUSTMENT)),
        contents,
    )
}

pub(crate) fn wheel_adjustment_allowed(ui: &egui::Ui) -> bool {
    !ui.stack()
        .iter()
        .any(|s| s.tags().contains(NO_WHEEL_ADJUSTMENT))
}

/// One step-button appearance for dialogs, panels and preferences.
pub fn stepper(ui: &mut egui::Ui, increase: bool) -> egui::Response {
    let background = crate::theme::Theme::stepper_button_bg_for(ui);
    let response = ui
        .scope(|ui| {
            ui.visuals_mut().widgets.inactive.bg_fill = background;
            ui.visuals_mut().widgets.inactive.weak_bg_fill = background;
            // A spinner has one keyboard stop: its editor. Arrow keys and
            // accessibility increment/decrement actions provide the same steps.
            ui.add(
                egui::Button::new(if increase { "+" } else { "−" })
                    .small()
                    .sense(egui::Sense::CLICK),
            )
            .on_hover_text(if increase {
                crate::t!("ui.increase")
            } else {
                crate::t!("ui.decrease")
            })
        })
        .inner;
    #[cfg(test)]
    ui.ctx().data_mut(|d| {
        d.insert_temp(
            egui::Id::new(if increase {
                "control_test_plus"
            } else {
                "control_test_minus"
            }),
            response.clone(),
        )
    });
    response
}

pub fn reset(ui: &mut egui::Ui, enabled: bool) -> egui::Response {
    let response = ui
        .add_enabled(enabled, egui::Button::new("↺").small())
        .on_hover_text(crate::t!("ui.reset_to_default"));
    #[cfg(test)]
    ui.ctx()
        .data_mut(|d| d.insert_temp(egui::Id::new("control_test_reset"), response.clone()));
    response
}
