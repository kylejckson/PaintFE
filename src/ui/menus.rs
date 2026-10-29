//! Shared menu dismissal and row geometry.
pub fn config() -> egui::containers::menu::MenuConfig {
    egui::containers::menu::MenuConfig::new()
        .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
        .style(|style: &mut egui::Style| {
            egui::containers::menu::menu_style(style);
            style.spacing.item_spacing.y = 0.0;
        })
}

pub fn context_menu(
    response: &egui::Response,
    contents: impl FnOnce(&mut egui::Ui),
) -> Option<egui::InnerResponse<()>> {
    egui::Popup::context_menu(response)
        .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
        .show(|ui| {
            ui.spacing_mut().item_spacing.y = 0.0;
            contents(ui);
        })
}

pub fn row_size(ui: &egui::Ui, width: f32, height: f32) -> egui::Vec2 {
    // A popup's sizing pass starts with a generous guessed area, not its content width.
    let width = if ui.is_sizing_pass() {
        width
    } else {
        width.max(ui.available_width())
    };
    egui::vec2(width, height.max(26.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{brushes::Assets, icons::Icon};

    #[derive(Default)]
    struct Menu {
        button: Option<egui::Rect>,
        disabled: Option<egui::Rect>,
        action: Option<egui::Rect>,
        fired: usize,
        visible: bool,
    }
    fn frame(ctx: &egui::Context, menu: &mut Menu, events: Vec<egui::Event>) {
        menu.visible = false;
        let assets = Assets::new();
        let _ = ctx.run_ui(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(900.0, 700.0),
                )),
                events,
                ..Default::default()
            },
            |ui| {
                egui::MenuBar::new().config(config()).ui(ui, |ui| {
                    menu.button = Some(
                        ui.menu_button("File", |ui| {
                            menu.visible = true;
                            ui.set_min_width(220.0);
                            menu.disabled = Some(
                                assets
                                    .menu_item_enabled(ui, Icon::Brush, "Unavailable", false)
                                    .rect,
                            );
                            let response = assets.menu_item(ui, Icon::Brush, "Run action");
                            menu.action = Some(response.rect);
                            if response.clicked() {
                                menu.fired += 1;
                                ui.close();
                            }
                        })
                        .response
                        .rect,
                    );
                });
            },
        );
    }
    fn click(ctx: &egui::Context, menu: &mut Menu, pos: egui::Pos2) {
        frame(
            ctx,
            menu,
            vec![
                egui::Event::PointerMoved(pos),
                egui::Event::PointerButton {
                    pos,
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::NONE,
                },
            ],
        );
        frame(
            ctx,
            menu,
            vec![egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed: false,
                modifiers: egui::Modifiers::NONE,
            }],
        );
        frame(ctx, menu, Vec::new());
    }
    #[test]
    fn disabled_entries_and_frame_margin_keep_menu_open_full_rows_activate_and_outside_closes() {
        for dark in [false, true] {
            let ctx = egui::Context::default();
            ctx.set_visuals(if dark {
                egui::Visuals::dark()
            } else {
                egui::Visuals::light()
            });
            let mut menu = Menu::default();
            frame(&ctx, &mut menu, Vec::new());
            let pos = menu.button.unwrap().center();
            click(&ctx, &mut menu, pos);
            assert!(menu.visible);
            let disabled = menu.disabled.unwrap();
            let action = menu.action.unwrap();
            assert!(disabled.height() >= 26.0);
            assert!(
                (disabled.bottom() - action.top()).abs() < 0.1,
                "unresponsive gap between rows"
            );
            assert!(action.width() >= 220.0);
            assert!(
                action.width() < 300.0,
                "menu expanded to the default popup area: {action:?}"
            );
            click(&ctx, &mut menu, disabled.center());
            assert!(menu.visible, "disabled item closed menu");
            click(
                &ctx,
                &mut menu,
                disabled.left_center() - egui::vec2(1.0, 0.0),
            );
            assert!(menu.visible, "inside frame margin closed menu");
            click(
                &ctx,
                &mut menu,
                action.right_center() - egui::vec2(2.0, 0.0),
            );
            assert_eq!(menu.fired, 1, "right side of full row did not activate");
            assert!(!menu.visible);
            let pos = menu.button.unwrap().center();
            click(&ctx, &mut menu, pos);
            click(&ctx, &mut menu, egui::pos2(800.0, 600.0));
            assert!(!menu.visible, "outside click did not dismiss menu");
        }
    }
}
