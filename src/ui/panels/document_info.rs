//! Compact read-only document inspector using the shared dialog card style.
pub(crate) struct Info<'a> {
    pub name: &'a str,
    pub width: u32,
    pub height: u32,
    pub layers: usize,
    pub path: Option<&'a str>,
}
pub(crate) fn contents(ui: &mut egui::Ui, assets: &crate::assets::Assets, info: Info<'_>) -> bool {
    use crate::ops::dialogs::{DialogColors, dialog_card_header, section_label};
    let mut close = dialog_card_header(
        ui,
        assets,
        crate::assets::Icon::MenuFileInfo,
        &crate::t!("ui.document_info"),
    );
    let colors = DialogColors::from_ctx(ui.ctx());
    ui.add_space(6.0);
    ui.add(
        egui::Label::new(egui::RichText::new(info.name).strong())
            .selectable(true)
            .wrap(),
    );
    ui.add_space(8.0);
    section_label(ui, &colors, &crate::t!("info.document"));
    egui::Grid::new("document_info_fields")
        .spacing(egui::vec2(24.0, 8.0))
        .show(ui, |ui| {
            ui.weak(crate::t!("info.dimensions"));
            ui.label(crate::i18n::format_translation(
                "ui.0_1_pixels",
                &[
                    ("0", info.width.to_string()),
                    ("1", info.height.to_string()),
                ],
            ));
            ui.end_row();
            ui.weak(crate::t!("info.layers"));
            ui.label(info.layers.to_string());
            ui.end_row();
        });
    ui.add_space(10.0);
    ui.separator();
    section_label(ui, &colors, &crate::t!("info.source"));
    if let Some(path) = info.path {
        let response = ui.add(egui::Label::new(path).selectable(true).wrap());
        #[cfg(test)]
        ui.ctx()
            .data_mut(|d| d.insert_temp(egui::Id::new("info_test_path"), response.rect));
        #[cfg(not(test))]
        let _ = response;
    } else {
        ui.weak(crate::t!("info.unsaved"));
    }
    ui.add_space(10.0);
    ui.horizontal(|ui| {
        let copy = ui.add_enabled(
            info.path.is_some(),
            egui::Button::new(crate::t!("info.copy_path")),
        );
        #[cfg(test)]
        ui.ctx().data_mut(|d| {
            d.insert_temp(egui::Id::new("info_test_copy"), copy.rect);
            d.insert_temp(egui::Id::new("info_test_copy_enabled"), copy.enabled());
        });
        if copy.clicked() {
            ui.ctx().copy_text(info.path.unwrap_or_default().to_owned());
        }
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let response = ui.button(crate::t!("ui.close"));
            #[cfg(test)]
            ui.ctx()
                .data_mut(|d| d.insert_temp(egui::Id::new("info_test_close"), response.rect));
            close |= response.clicked();
        });
    });
    close
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn inspector_wraps_paths_copies_full_text_and_closes_at_large_text_sizes() {
        crate::i18n::init();
        for dark in [false, true] {
            for scale in [1.0, 1.5] {
                let ctx = egui::Context::default();
                ctx.set_visuals(if dark {
                    egui::Visuals::dark()
                } else {
                    egui::Visuals::light()
                });
                let mut style = (*ctx.global_style()).clone();
                for font in style.text_styles.values_mut() {
                    font.size *= scale;
                }
                ctx.set_global_style(style);
                let assets = crate::assets::Assets::new();
                let path = "D:/Pictures/a long folder name/another long folder name/sample-transparent.avif";
                let closed = std::cell::Cell::new(false);
                let frame = |events: Vec<egui::Event>, saved: bool| {
                    ctx.run_ui(
                        egui::RawInput {
                            screen_rect: Some(egui::Rect::from_min_size(
                                egui::Pos2::ZERO,
                                egui::vec2(900.0, 800.0),
                            )),
                            events,
                            ..Default::default()
                        },
                        |_| {
                            egui::Window::new("info_test")
                                .title_bar(false)
                                .fixed_pos(egui::pos2(30.0, 30.0))
                                .show(&ctx, |ui| {
                                    ui.set_width(360.0);
                                    closed.set(contents(
                                        ui,
                                        &assets,
                                        Info {
                                            name: "sample-transparent.avif",
                                            width: 29,
                                            height: 100,
                                            layers: 2,
                                            path: saved.then_some(path),
                                        },
                                    ));
                                });
                        },
                    )
                };
                for _ in 0..3 {
                    frame(vec![], true);
                }
                let rect = ctx
                    .data(|d| d.get_temp::<egui::Rect>(egui::Id::new("info_test_path")))
                    .unwrap();
                assert!(rect.height() > 20.0 * scale, "path must wrap");
                for (key, is_copy) in [("info_test_copy", true), ("info_test_close", false)] {
                    let pos = ctx
                        .data(|d| d.get_temp::<egui::Rect>(egui::Id::new(key)))
                        .unwrap()
                        .center();
                    let event = |pressed| {
                        vec![
                            egui::Event::PointerMoved(pos),
                            egui::Event::PointerButton {
                                pos,
                                button: egui::PointerButton::Primary,
                                pressed,
                                modifiers: egui::Modifiers::NONE,
                            },
                        ]
                    };
                    frame(event(true), true);
                    let output = frame(event(false), true);
                    if is_copy {
                        assert!(output.platform_output.commands.iter().any(|command|matches!(command,egui::OutputCommand::CopyText(text) if text==path)));
                    } else {
                        assert!(closed.get());
                    }
                }
                frame(vec![], false);
                assert!(
                    !ctx.data(|d| d.get_temp::<bool>(egui::Id::new("info_test_copy_enabled")))
                        .unwrap()
                );
            }
        }
    }
}
