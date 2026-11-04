pub const DOCUMENTATION_URL: &str = "https://paintfe.com/docs";

const SECTIONS: [(&str, &str); 6] = [
    ("help.start.open", "help.start.open_body"),
    ("help.start.navigate", "help.start.navigate_body"),
    ("help.start.edit", "help.start.edit_body"),
    ("help.start.transform", "help.start.transform_body"),
    ("help.start.undo", "help.start.undo_body"),
    ("help.start.save", "help.start.save_body"),
];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HelpAction {
    AiSetup,
    Preferences,
    Keybindings,
}

pub struct HelpResponse {
    pub rect: egui::Rect,
    pub action: Option<HelpAction>,
}

pub fn open(ctx: &egui::Context) {
    ctx.data_mut(|data| data.insert_temp(egui::Id::new("quick_start_open"), true));
}

pub fn is_open(ctx: &egui::Context) -> bool {
    ctx.data(|data| data.get_temp::<bool>(egui::Id::new("quick_start_open")))
        .unwrap_or(false)
}

pub fn show(ctx: &egui::Context) -> Option<HelpResponse> {
    let id = egui::Id::new("quick_start_open");
    let mut open = ctx.data(|data| data.get_temp::<bool>(id)).unwrap_or(false);
    if !open {
        return None;
    }
    let bounds = ctx.content_rect();
    let width = (bounds.width() - 32.).clamp(120., 540.);
    let mut action = None;
    let shown = egui::Window::new(crate::t!("help.quick_start"))
        .id(egui::Id::new("quick_start_window"))
        .open(&mut open)
        .default_width(width)
        .default_height((bounds.height() - 160.).clamp(180., 520.))
        .max_width(width)
        .default_pos(egui::pos2(
            bounds.center().x - width * 0.5,
            bounds.top() + 40.,
        ))
        .constrain_to(bounds)
        .resizable(true)
        .show(ctx, |ui| {
            egui::ScrollArea::vertical()
                .max_height((bounds.height() - 120.).clamp(120., 520.))
                .show(ui, |ui| {
                    ui.label(crate::t!("help.start.intro"));
                    ui.add_space(8.);
                    for (title, body) in SECTIONS {
                        ui.strong(crate::i18n::translate(title));
                        ui.add(egui::Label::new(crate::i18n::translate(body)).wrap());
                        ui.add_space(10.);
                    }
                    ui.separator();
                    ui.strong(crate::t!("help.start.explore"));
                    ui.horizontal_wrapped(|ui| {
                        for (label, target) in [
                            ("help.ai_setup", HelpAction::AiSetup),
                            ("settings.title", HelpAction::Preferences),
                            ("settings.tab.keybinds", HelpAction::Keybindings),
                        ] {
                            ui.push_id(target, |ui| {
                                if ui.button(crate::i18n::translate(label)).clicked() {
                                    action = Some(target);
                                }
                            });
                        }
                    });
                    ui.hyperlink_to(crate::t!("help.documentation"), DOCUMENTATION_URL);
                });
        });
    if action.is_some() || ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
        open = false;
    }
    ctx.data_mut(|data| data.insert_temp(id, open));
    shown.map(|window| HelpResponse {
        rect: window.response.rect,
        action,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quick_start_fits_small_and_large_windows_and_escape_closes_it() {
        crate::i18n::init();
        for size in [egui::vec2(1280., 752.), egui::vec2(480., 420.)] {
            for dark in [false, true] {
                for scale in [1., 1.35] {
                    let ctx = egui::Context::default();
                    ctx.set_visuals(if dark {
                        egui::Visuals::dark()
                    } else {
                        egui::Visuals::light()
                    });
                    ctx.all_styles_mut(|style| {
                        for font in style.text_styles.values_mut() {
                            font.size *= scale;
                        }
                    });
                    let bounds = egui::Rect::from_min_size(egui::Pos2::ZERO, size);
                    open(&ctx);
                    let mut rect = None;
                    for _ in 0..3 {
                        let _ = ctx.run_ui(
                            egui::RawInput {
                                screen_rect: Some(bounds),
                                ..Default::default()
                            },
                            |_| {
                                rect = show(&ctx).map(|shown| shown.rect);
                            },
                        );
                    }
                    assert!(
                        bounds.expand(1.).contains_rect(rect.unwrap()),
                        "{size:?}, dark={dark}, scale={scale}: {rect:?}"
                    );
                    let _ = ctx.run_ui(
                        egui::RawInput {
                            screen_rect: Some(bounds),
                            events: vec![egui::Event::Key {
                                key: egui::Key::Escape,
                                physical_key: None,
                                pressed: true,
                                repeat: false,
                                modifiers: egui::Modifiers::NONE,
                            }],
                            ..Default::default()
                        },
                        |_| {
                            show(&ctx);
                        },
                    );
                    assert!(!is_open(&ctx));
                }
            }
        }
    }
}
