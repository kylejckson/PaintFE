//! Searchable personal labels. Drafts apply together and never change action IDs.
use egui::{Context, Id};
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};

#[derive(Default)]
struct Editor {
    language: String,
    entries: Vec<crate::i18n::LabelEntry>,
    query: String,
    selected: Option<String>,
    modified_only: bool,
    dirty: bool,
    json: String,
    message: Option<String>,
    confirm_discard: bool,
    reveal_detail: bool,
}
type Handle = Arc<Mutex<Editor>>;
fn id() -> Id {
    Id::new("personal_label_editor")
}

pub fn is_open(ctx: &Context) -> bool {
    ctx.data(|d| d.get_temp::<Handle>(id())).is_some()
}

pub fn open(ctx: &Context) {
    let language = crate::i18n::current_language();
    let editor = Editor {
        entries: crate::i18n::label_entries(&language),
        language,
        ..Default::default()
    };
    ctx.data_mut(|d| d.insert_temp(id(), Arc::new(Mutex::new(editor))));
}

fn values(entries: &[crate::i18n::LabelEntry]) -> BTreeMap<String, String> {
    entries
        .iter()
        .filter_map(|e| e.custom.as_ref().map(|s| (e.key.clone(), s.clone())))
        .collect()
}

fn results(ui: &mut egui::Ui, editor: &mut Editor, filtered: &[usize], height: f32, stacked: bool) {
    let row_height = (ui.text_style_height(&egui::TextStyle::Button) * 2.0 + 12.0).max(44.0);
    egui::ScrollArea::vertical()
        .id_salt("label_results")
        .max_height(height)
        .show_rows(ui, row_height, filtered.len(), |ui, rows| {
            for row in rows {
                let entry = &editor.entries[filtered[row]];
                ui.push_id(&entry.key, |ui| {
                    let selected = editor.selected.as_ref() == Some(&entry.key);
                    let (rect, response) = ui.allocate_exact_size(
                        egui::vec2(ui.available_width(), row_height),
                        egui::Sense::click(),
                    );
                    if selected || response.hovered() {
                        ui.painter().rect_filled(
                            rect.shrink(1.0),
                            4.0,
                            if selected {
                                ui.visuals().selection.bg_fill
                            } else {
                                ui.visuals().widgets.hovered.bg_fill
                            },
                        );
                    }
                    let current = entry.custom.as_ref().unwrap_or(&entry.original);
                    let foreground = if selected {
                        ui.visuals().selection.stroke.color
                    } else {
                        ui.visuals().text_color()
                    };
                    let mut row_ui = ui.new_child(
                        egui::UiBuilder::new()
                            .max_rect(rect.shrink2(egui::vec2(8.0, 3.0)))
                            .layout(egui::Layout::top_down(egui::Align::Min)),
                    );
                    row_ui.add(
                        egui::Label::new(egui::RichText::new(current).color(foreground).strong())
                            .truncate()
                            .selectable(false),
                    );
                    row_ui.add(
                        egui::Label::new(
                            egui::RichText::new(&entry.key)
                                .small()
                                .color(foreground.gamma_multiply(0.7)),
                        )
                        .truncate()
                        .selectable(false),
                    );
                    if entry.custom.is_some() {
                        ui.painter().circle_filled(
                            rect.right_center() - egui::vec2(8.0, 0.0),
                            2.5,
                            foreground,
                        );
                    }
                    response.clone().on_hover_text(&entry.key);
                    #[cfg(test)]
                    ui.ctx().data_mut(|d| {
                        d.insert_temp(Id::new(("label_test_row", &entry.key)), response.rect)
                    });
                    if response.clicked() {
                        editor.selected = Some(entry.key.clone());
                        editor.reveal_detail = stacked;
                    }
                });
            }
        });
    if filtered.is_empty() {
        ui.weak(crate::t!("labels.no_results"));
    }
}

fn detail(ui: &mut egui::Ui, editor: &mut Editor) {
    use crate::ops::dialogs::{DialogColors, section_label};
    let colors = DialogColors::from_ctx(ui.ctx());
    if let Some(entry) = editor
        .entries
        .iter_mut()
        .find(|e| Some(&e.key) == editor.selected.as_ref())
    {
        section_label(ui, &colors, &crate::t!("labels.original"));
        ui.add(egui::Label::new(&entry.original).selectable(true).wrap());
        ui.add_space(10.0);
        section_label(ui, &colors, &crate::t!("labels.your_wording"));
        let mut replacement = entry
            .custom
            .clone()
            .unwrap_or_else(|| entry.original.clone());
        let response = ui.add(
            egui::TextEdit::multiline(&mut replacement)
                .id_salt(("label_value", &entry.key))
                .desired_rows(3)
                .desired_width(f32::INFINITY),
        );
        #[cfg(test)]
        ui.ctx().data_mut(|d| {
            d.insert_temp(Id::new("label_test_value"), response.rect);
            d.insert_temp(Id::new("label_test_value_id"), response.id);
            d.insert_temp(Id::new("label_test_value_clip"), ui.clip_rect());
        });
        if editor.reveal_detail {
            ui.scroll_to_rect_animation(
                response.rect,
                Some(egui::Align::Center),
                egui::style::ScrollAnimation::none(),
            );
            editor.reveal_detail = false;
        }
        if response.changed() {
            entry.custom = (replacement != entry.original).then_some(replacement);
            editor.dirty = true;
        }
        if let Some(custom) = &entry.custom
            && let Err(error) = crate::i18n::validate_label(&entry.original, custom)
        {
            ui.colored_label(ui.visuals().error_fg_color, error);
        }
        ui.add_space(6.0);
        if ui
            .add_enabled(
                entry.custom.is_some(),
                egui::Button::new(crate::t!("labels.reset_one")),
            )
            .clicked()
        {
            entry.custom = None;
            editor.dirty = true;
        }
        ui.add_space(10.0);
        ui.weak(&entry.key);
    } else {
        ui.weak(crate::t!("labels.select_label"));
    }
}

/// Returns true while the modal owns canvas/menu input.
pub fn show(ctx: &Context, assets: &crate::assets::Assets) -> bool {
    let Some(handle) = ctx.data(|d| d.get_temp::<Handle>(id())) else {
        return false;
    };
    let mut editor = handle.lock().unwrap_or_else(|e| e.into_inner());
    let editor = &mut *editor;
    let mut close_requested = false;
    let mut close = false;
    let theme = ctx
        .data(|d| d.get_temp::<crate::theme::Theme>(Id::new("floating_widget_theme")))
        .unwrap_or_else(|| {
            if ctx.global_style().visuals.dark_mode {
                crate::theme::Theme::dark()
            } else {
                crate::theme::Theme::light()
            }
        });
    // Bound the entire dialog, including its footer, to the app viewport.
    let viewport = ctx.content_rect();
    let maximum = (viewport.size() - egui::vec2(48.0, 48.0)).max(egui::vec2(200.0, 240.0));
    let initial = egui::vec2(760.0, 520.0).min(maximum);
    crate::ui::polish::window(ctx, "personal_label_editor_window")
        .id(id())
        .title_bar(false)
        .collapsible(false)
        .resizable(true)
        .default_size(initial)
        .min_size(egui::vec2(320.0, 360.0).min(maximum))
        .max_size(maximum)
        .heading_drag(
            ctx,
            id().with("heading_position"),
            viewport.center() - initial * 0.5,
        )
        .frame(theme.dialog_frame())
        .show(ctx, |ui| {
            ui.memory_mut(|memory| memory.set_modal_layer(ui.layer_id()));
            close_requested |= crate::ops::dialogs::dialog_card_header(
                ui,
                assets,
                crate::assets::Icon::Settings,
                &crate::t!("labels.title"),
            );
            ui.weak(format!(
                "{} ({})",
                crate::t!("labels.description"),
                editor.language
            ));
            ui.add_space(8.0);
            let query_response = ui.add(
                egui::TextEdit::singleline(&mut editor.query)
                    .id_salt("label_query")
                    .hint_text(crate::t!("labels.search"))
                    .desired_width(f32::INFINITY),
            );
            #[cfg(test)]
            ctx.data_mut(|d| d.insert_temp(Id::new("label_test_query"), query_response.rect));
            #[cfg(not(test))]
            let _ = query_response;
            ui.add_space(4.0);
            let query = editor.query.to_lowercase();
            ui.horizontal(|ui| {
                if ui
                    .selectable_label(!editor.modified_only, crate::t!("labels.all"))
                    .clicked()
                {
                    editor.modified_only = false;
                }
                if ui
                    .selectable_label(editor.modified_only, crate::t!("labels.customized"))
                    .clicked()
                {
                    editor.modified_only = true;
                }
            });
            let mut filtered: Vec<_> = editor
                .entries
                .iter()
                .enumerate()
                .filter(|(_, e)| {
                    (!editor.modified_only || e.custom.is_some())
                        && (query.is_empty()
                            || e.key.to_lowercase().contains(&query)
                            || e.original.to_lowercase().contains(&query)
                            || e.custom
                                .as_ref()
                                .is_some_and(|s| s.to_lowercase().contains(&query)))
                })
                .map(|(i, _)| i)
                .collect();
            if !query.is_empty() {
                filtered.sort_by_key(|&i| {
                    let e = &editor.entries[i];
                    (
                        !(e.key.to_lowercase() == query
                            || e.original.to_lowercase() == query
                            || e.custom.as_ref().is_some_and(|s| s.to_lowercase() == query)),
                        e.key.clone(),
                    )
                });
            }
            ui.weak(crate::i18n::format_translation(
                "labels.results",
                &[("0", filtered.len().to_string())],
            ));
            ui.add_space(6.0);
            let line = ui.text_style_height(&egui::TextStyle::Body);
            let footer_height = line * (if editor.confirm_discard { 5.0 } else { 2.0 })
                + if editor.message.is_some() {
                    line * 2.0
                } else {
                    0.0
                }
                + 24.0;
            let body_limit = (ui.available_height() - footer_height).max(40.0);
            egui::ScrollArea::vertical()
                .id_salt("label_editor_body")
                .max_height(body_limit)
                .min_scrolled_height(0.0)
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    let height = (body_limit - 72.0).clamp(80.0, 320.0);
                    if ui.available_width() >= 620.0 {
                        ui.columns(2, |columns| {
                            columns[0].push_id("label_list_pane", |ui| {
                                results(ui, editor, &filtered, height, false)
                            });
                            columns[1].push_id("label_detail_pane", |ui| {
                                ui.set_min_height(height);
                                detail(ui, editor);
                            });
                        });
                    } else {
                        results(ui, editor, &filtered, height.min(150.0), true);
                        ui.separator();
                        detail(ui, editor);
                    }
                    ui.add_space(8.0);
                    ui.collapsing(crate::t!("labels.share"), |ui| {
                        ui.label(crate::t!("labels.share_help"));
                        ui.add(
                            egui::TextEdit::multiline(&mut editor.json)
                                .id_salt("labels_json")
                                .desired_rows(3)
                                .desired_width(f32::INFINITY),
                        );
                        ui.horizontal(|ui| {
                            if ui.button(crate::t!("labels.export")).clicked() {
                                editor.json =
                                    serde_json::to_string_pretty(&values(&editor.entries))
                                        .unwrap_or_default();
                                ctx.copy_text(editor.json.clone());
                            }
                            if ui.button(crate::t!("labels.import")).clicked() {
                                match crate::i18n::parse_label_overrides(
                                    &editor.language,
                                    &editor.json,
                                ) {
                                    Ok(values) => {
                                        for entry in &mut editor.entries {
                                            if let Some(value) = values.get(&entry.key) {
                                                entry.custom = Some(value.clone());
                                            }
                                        }
                                        editor.dirty = true;
                                        editor.message = None;
                                    }
                                    Err(error) => editor.message = Some(error),
                                }
                            }
                        });
                    });
                    ui.collapsing(crate::t!("labels.reset_options"), |ui| {
                        if ui.button(crate::t!("labels.reset_all")).clicked() {
                            for entry in &mut editor.entries {
                                entry.custom = None;
                            }
                            editor.dirty = true;
                        }
                    });
                });
            ui.separator();
            if let Some(message) = &editor.message {
                ui.label(message);
            }
            let footer = ui.horizontal(|ui| {
                ui.weak(if editor.dirty {
                    crate::t!("labels.unsaved")
                } else {
                    crate::t!("labels.up_to_date")
                });
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui
                        .add_enabled(
                            editor.dirty,
                            egui::Button::new(crate::t!("labels.apply"))
                                .fill(ui.visuals().selection.bg_fill),
                        )
                        .clicked()
                    {
                        match crate::i18n::save_label_overrides(
                            &editor.language,
                            values(&editor.entries),
                        ) {
                            Ok(()) => {
                                editor.entries = crate::i18n::label_entries(&editor.language);
                                editor.dirty = false;
                                editor.message = Some(crate::t!("labels.saved"));
                                ctx.request_repaint();
                            }
                            Err(error) => editor.message = Some(error),
                        }
                    }
                    close_requested |= ui.button(crate::t!("ui.close")).clicked();
                });
            });
            #[cfg(test)]
            ctx.data_mut(|d| d.insert_temp(Id::new("label_test_footer"), footer.response.rect));
            #[cfg(not(test))]
            let _ = footer;
            if editor.confirm_discard {
                ui.label(crate::t!("labels.discard_question"));
                ui.horizontal(|ui| {
                    if ui.button(crate::t!("labels.discard")).clicked() {
                        close = true;
                    }
                    if ui.button(crate::t!("ui.cancel")).clicked() {
                        editor.confirm_discard = false;
                    }
                });
            }
        });
    close_requested |= ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::Escape));
    if close_requested {
        if editor.dirty {
            editor.confirm_discard = true;
        } else {
            close = true;
        }
    }
    if close {
        ctx.data_mut(|d| d.remove::<Handle>(id()));
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    fn frame(ctx: &Context, events: Vec<egui::Event>) {
        let _ = ctx.run_ui(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    ctx.data(|d| d.get_temp::<egui::Vec2>(Id::new("label_test_screen_size")))
                        .unwrap_or(egui::vec2(1100.0, 900.0)),
                )),
                events,
                ..Default::default()
            },
            |_| {
                show(ctx, &crate::assets::Assets::new());
            },
        );
    }
    fn click(ctx: &Context, id: Id) {
        let position = ctx.data(|d| d.get_temp::<egui::Rect>(id)).unwrap().center();
        for pressed in [true, false] {
            frame(
                ctx,
                vec![
                    egui::Event::PointerMoved(position),
                    egui::Event::PointerButton {
                        pos: position,
                        button: egui::PointerButton::Primary,
                        pressed,
                        modifiers: egui::Modifiers::NONE,
                    },
                ],
            );
        }
    }
    #[test]
    fn searchable_editor_stages_real_text_input_in_both_themes_and_large_text() {
        crate::i18n::init();
        for dark in [false, true] {
            for scale in [1.0, 1.5] {
                for width in [1100.0, 560.0] {
                    let ctx = Context::default();
                    ctx.data_mut(|d| {
                        d.insert_temp(
                            Id::new("label_test_screen_size"),
                            egui::vec2(width, if width < 600.0 { 700.0 } else { 900.0 }),
                        )
                    });
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
                    open(&ctx);
                    for _ in 0..3 {
                        frame(&ctx, vec![]);
                    }
                    let footer = ctx
                        .data(|d| d.get_temp::<egui::Rect>(Id::new("label_test_footer")))
                        .unwrap();
                    assert!(
                        ctx.content_rect().contains_rect(footer),
                        "footer must fit the viewport at {width}, scale {scale}: {footer:?}"
                    );
                    click(&ctx, Id::new("label_test_query"));
                    frame(&ctx, vec![egui::Event::Text("menu.view".into())]);
                    frame(&ctx, vec![]);
                    click(&ctx, Id::new(("label_test_row", "menu.view")));
                    frame(&ctx, vec![]);
                    for _ in 0..3 {
                        frame(&ctx, vec![]);
                    }
                    let rect = ctx
                        .data(|d| d.get_temp::<egui::Rect>(Id::new("label_test_value")))
                        .unwrap();
                    let clip = ctx
                        .data(|d| d.get_temp::<egui::Rect>(Id::new("label_test_value_clip")))
                        .unwrap();
                    assert!(
                        clip.contains(rect.center()),
                        "editing field must be visible at {width}, scale {scale}"
                    );
                    click(&ctx, Id::new("label_test_value"));
                    let expected = ctx
                        .data(|d| d.get_temp::<Id>(Id::new("label_test_value_id")))
                        .unwrap();
                    assert!(
                        ctx.memory(|m| m.has_focus(expected)),
                        "editing field must take focus at {width}, scale {scale}"
                    );
                    frame(
                        &ctx,
                        vec![
                            egui::Event::Key {
                                key: egui::Key::A,
                                physical_key: None,
                                pressed: true,
                                repeat: false,
                                modifiers: egui::Modifiers::COMMAND,
                            },
                            egui::Event::Text("Panels".into()),
                        ],
                    );
                    let handle = ctx.data(|d| d.get_temp::<Handle>(id())).unwrap();
                    let editor = handle.lock().unwrap();
                    assert_eq!(editor.query, "menu.view");
                    assert_eq!(editor.selected.as_deref(), Some("menu.view"));
                    assert_eq!(
                        editor
                            .entries
                            .iter()
                            .find(|e| e.key == "menu.view")
                            .unwrap()
                            .custom
                            .as_deref(),
                        Some("Panels"),
                        "dark={dark} scale={scale} width={width}"
                    );
                    assert!(editor.dirty);
                    // Changes are staged, so no test writes to the user's profile.
                    assert_eq!(crate::i18n::translate("menu.view"), "View");
                }
            }
        }
    }
}
