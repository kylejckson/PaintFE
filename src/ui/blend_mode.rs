use crate::canvas::BlendMode;

/// Shared translated labels and explanations for every blend selector.
pub(crate) fn picker(
    ui: &mut egui::Ui,
    id: impl std::hash::Hash + std::fmt::Debug,
    selected: &mut BlendMode,
    width: f32,
) -> egui::Response {
    let before = *selected;
    let mut response = egui::ComboBox::from_id_salt(id)
        .selected_text(selected.display_name())
        .width(width)
        .show_ui(ui, |ui| {
            for &mode in BlendMode::all() {
                let response = ui
                    .selectable_label(mode == *selected, mode.display_name())
                    .on_hover_text(mode.help_text());
                #[cfg(test)]
                ui.ctx().data_mut(|data| {
                    data.insert_temp(egui::Id::new(("blend_choice", mode.to_u8())), response.rect);
                });
                if response.clicked() {
                    *selected = mode;
                }
            }
        })
        .response
        .on_hover_text(selected.help_text());
    if before != *selected {
        response.mark_changed();
    }
    response
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn shared_blend_picker_uses_translated_help_and_changes_the_selected_mode() {
        crate::i18n::init();
        for dark in [false, true] {
            let ctx = egui::Context::default();
            ctx.set_visuals(if dark {
                egui::Visuals::dark()
            } else {
                egui::Visuals::light()
            });
            let mut selected = BlendMode::Normal;
            let mut time = 0.;
            let mut draw = |events| {
                time += 0.1;
                let mut response = None;
                let output = ctx.run_ui(
                    egui::RawInput {
                        time: Some(time),
                        events,
                        screen_rect: Some(egui::Rect::from_min_size(
                            egui::Pos2::ZERO,
                            egui::vec2(800., 700.),
                        )),
                        ..Default::default()
                    },
                    |ui| {
                        response = Some(picker(ui, "test_blend", &mut selected, 120.));
                    },
                );
                (response.unwrap(), output)
            };
            let initial = draw(vec![]).0;
            let pointer = |pos, pressed| egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed,
                modifiers: egui::Modifiers::NONE,
            };
            let pos = initial.rect.center();
            draw(vec![egui::Event::PointerMoved(pos), pointer(pos, true)]);
            draw(vec![pointer(pos, false)]);
            draw(vec![]);
            let rect = ctx
                .data(|data| {
                    data.get_temp::<egui::Rect>(egui::Id::new((
                        "blend_choice",
                        BlendMode::HardLight.to_u8(),
                    )))
                })
                .unwrap();
            let point = rect.center();
            draw(vec![egui::Event::PointerMoved(point), pointer(point, true)]);
            let response = draw(vec![pointer(point, false)]).0;
            assert!(response.changed());
            assert_eq!(selected, BlendMode::HardLight);
            for mode in BlendMode::all() {
                let help = mode.help_text();
                assert!(help.starts_with(&mode.display_name()));
                assert!(!help.contains("blend_help."), "{mode:?}");
                assert!(help.lines().count() >= 2);
            }
        }
    }
}
