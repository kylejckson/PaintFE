//! Preferences uses the same themed rows/buttons as the other settings pages.
use crate::services::optional_packs::{self as packs, Pack};
use egui::{Context, FontDefinitions, FontFamily, Id, Ui};
use std::sync::{Arc, Mutex, mpsc};

#[derive(Clone)]
struct State {
    status: [String; 4],
    installed: [bool; 4],
    job: Option<Arc<Mutex<mpsc::Receiver<Result<(Pack, Option<Vec<u8>>), String>>>>>,
    prompt: Option<Pack>,
    pending_language: Option<String>,
    pending_auto: bool,
    message: Option<String>,
    skipped: Vec<String>,
}
impl Default for State {
    fn default() -> Self {
        Self {
            status: std::array::from_fn(|_| crate::t!("packs.not_installed")),
            installed: [false; 4],
            job: None,
            prompt: None,
            pending_language: None,
            pending_auto: false,
            message: None,
            skipped: Vec::new(),
        }
    }
}
fn state(ctx: &Context) -> State {
    ctx.data(|d| d.get_temp::<State>(Id::new("optional_pack_ui")))
        .unwrap_or_default()
}
fn save(ctx: &Context, state: State) {
    ctx.data_mut(|d| d.insert_temp(Id::new("optional_pack_ui"), state));
}
fn index(pack: Pack) -> usize {
    match pack {
        Pack::Cjk => 0,
        Pack::Jxl => 1,
        Pack::Avif => 2,
        Pack::Heic => 3,
    }
}
fn available(ctx: &Context, language: &str) -> bool {
    !packs::needs_cjk(language)
        || ctx
            .data(|d| d.get_temp::<Vec<String>>(Id::new("cjk_available_languages")))
            .unwrap_or_default()
            .iter()
            .any(|l| l == language)
}
/// Keep regional system glyphs ahead of the generic pack, and both behind the
/// primary UI and Latin/Cyrillic families. Also works with minimal test fonts.
fn add_fallback(definitions: &mut FontDefinitions, name: &str) {
    for family in [
        FontFamily::Proportional,
        FontFamily::Monospace,
        FontFamily::Name("WidgetTitle".into()),
    ] {
        let fonts = definitions.families.entry(family).or_default();
        let position = fonts
            .iter()
            .enumerate()
            .filter(|(_, font)| font.as_str() == "noto_sans" || font.starts_with("system_cjk"))
            .map(|(i, _)| i + 1)
            .max()
            .unwrap_or(fonts.len());
        fonts.insert(position, name.into());
    }
}
fn apply_font(ctx: &Context, font: Option<Vec<u8>>) {
    let Some(mut definitions) =
        ctx.data(|d| d.get_temp::<FontDefinitions>(Id::new("optional_font_base")))
    else {
        return;
    };
    let mut languages = ctx
        .data(|d| d.get_temp::<Vec<String>>(Id::new("system_cjk_languages")))
        .unwrap_or_default();
    if let Some(bytes) = font {
        definitions.font_data.insert(
            "optional_cjk".into(),
            egui::FontData::from_owned(bytes).into(),
        );
        add_fallback(&mut definitions, "optional_cjk");
        languages = vec!["ja".into(), "ko".into(), "zh-CN".into(), "zh-TW".into()];
    }
    ctx.data_mut(|d| d.insert_temp(Id::new("cjk_available_languages"), languages));
    ctx.set_fonts(definitions);
    ctx.data_mut(|d| {
        let id = Id::new("font_layout_revision");
        let revision = d.get_temp::<u64>(id).unwrap_or(0);
        d.insert_temp(id, revision.wrapping_add(1));
    });
}
/// Base definitions include a small language-picker subset, never the full CJK font.
pub fn initialize_fonts(
    ctx: &Context,
    definitions: FontDefinitions,
    system_languages: Vec<String>,
    auto_language: bool,
) {
    ctx.data_mut(|d| {
        d.insert_temp(Id::new("optional_font_base"), definitions);
        d.insert_temp(Id::new("system_cjk_languages"), system_languages);
    });
    let mut state = State::default();
    let font = packs::installed(Pack::Cjk)
        .ok()
        .and_then(|p| std::fs::read(p).ok());
    state.installed[0] = font.is_some();
    if state.installed[0] {
        state.status[0] = crate::t!("packs.installed");
    }
    state.installed[1] = packs::installed(Pack::Jxl).is_ok();
    if state.installed[1] {
        state.status[1] = crate::t!("packs.installed");
    }
    state.installed[2] = packs::installed(Pack::Avif).is_ok();
    if state.installed[2] {
        state.status[2] = crate::t!("packs.installed");
    }
    for pack in Pack::ALL {
        if !state.installed[index(pack)] && packs::is_present(pack) {
            state.status[index(pack)] = crate::t!("packs.invalid");
        }
    }
    apply_font(ctx, font);
    let language = crate::i18n::current_language();
    if !available(ctx, &language) {
        state.prompt = Some(Pack::Cjk);
        state.pending_language = Some(language);
        state.pending_auto = auto_language;
        crate::i18n::set_language("en");
    }
    save(ctx, state);
}
pub fn select_language(ctx: &Context, language: &str, auto_language: bool) -> bool {
    if available(ctx, language) {
        return true;
    }
    if let Some((name, bytes)) = crate::app::discover_system_cjk_font(language) {
        let mut base = ctx
            .data(|d| d.get_temp::<FontDefinitions>(Id::new("optional_font_base")))
            .unwrap_or_default();
        let font_name = format!("{name}_{language}");
        base.font_data
            .insert(font_name.clone(), egui::FontData::from_owned(bytes).into());
        add_fallback(&mut base, &font_name);
        let mut languages = ctx
            .data(|d| d.get_temp::<Vec<String>>(Id::new("system_cjk_languages")))
            .unwrap_or_default();
        languages.push(language.into());
        ctx.data_mut(|d| {
            d.insert_temp(Id::new("optional_font_base"), base);
            d.insert_temp(Id::new("system_cjk_languages"), languages);
        });
        apply_font(ctx, None);
        return true;
    }
    let mut state = state(ctx);
    state.prompt = Some(Pack::Cjk);
    state.pending_language = Some(language.into());
    state.pending_auto = auto_language;
    state.message = None;
    save(ctx, state);
    false
}
fn start(
    ctx: &Context,
    state: &mut State,
    pack: Pack,
    path: Option<std::path::PathBuf>,
    remove: bool,
) {
    let (tx, rx) = mpsc::channel();
    state.job = Some(Arc::new(Mutex::new(rx)));
    state.message = None;
    let ctx = ctx.clone();
    std::thread::spawn(move || {
        let result = (|| {
            if remove {
                packs::remove(pack)?;
            } else if let Some(path) = path {
                packs::install(&path, pack)?;
            } else {
                packs::download(pack)?;
            }
            let font = if pack == Pack::Cjk && !remove {
                Some(std::fs::read(packs::installed(pack)?).map_err(|e| e.to_string())?)
            } else {
                None
            };
            Ok((pack, font))
        })();
        let _ = tx.send(result);
        ctx.request_repaint();
    });
}
fn import(ctx: &Context, state: &mut State, pack: Pack) {
    // The native picker is invoked only by the user's Import button.
    if let Some(path) = crate::services::file_dialogs::resource_dialog()
        .add_filter("PaintFE feature pack", &["pfepack"])
        .pick_file()
    {
        start(ctx, state, pack, Some(path), false);
    }
}
pub fn show_settings(ui: &mut Ui) {
    let ctx = ui.ctx().clone();
    let mut state = state(&ctx);
    super::settings_window::SettingsWindow::section_header(ui, &crate::t!("packs.title"));
    ui.label(crate::t!("packs.offline"));
    ui.add_space(8.0);
    ui.weak(crate::t!("packs.download_warning"));
    for pack in Pack::ALL {
        ui.push_id(pack.id(), |ui| {
            ui.add_space(12.0);
            ui.separator();
            ui.strong(pack.title());
            ui.label(crate::t!(match pack {
                Pack::Cjk => "packs.cjk_description",
                Pack::Jxl => "packs.jxl_description",
                Pack::Avif => "packs.avif_description",
                Pack::Heic => "packs.heic_description",
            }));
            ui.label(&state.status[index(pack)]);
            if let Some(size) = match pack {
                Pack::Cjk => option_env!("PAINTFE_CJK_PACK_SIZE"),
                Pack::Jxl => option_env!("PAINTFE_JXL_PACK_SIZE"),
                Pack::Avif => option_env!("PAINTFE_AVIF_PACK_SIZE"),
                Pack::Heic => option_env!("PAINTFE_HEIC_PACK_SIZE"),
            } && let Ok(size) = size.parse::<u64>()
            {
                ui.weak(format!("{:.2} MiB", size as f64 / 1048576.0));
            }
            ui.add_enabled_ui(state.job.is_none(), |ui| {
                ui.horizontal_wrapped(|ui| {
                    if ui.button(crate::t!("packs.download")).clicked() {
                        state.prompt = Some(pack);
                        state.message = None;
                    }
                    if ui.button(crate::t!("packs.import")).clicked() {
                        import(&ctx, &mut state, pack);
                    }
                    if ui
                        .add_enabled(
                            packs::is_present(pack),
                            egui::Button::new(crate::t!("packs.remove")),
                        )
                        .clicked()
                    {
                        start(&ctx, &mut state, pack, None, true);
                    }
                });
            });
        });
    }
    if state.job.is_some() {
        ui.horizontal(|ui| {
            ui.spinner();
            ui.label(crate::t!("packs.working"));
        });
    }
    if let Some(message) = &state.message {
        ui.label(message);
    }
    super::settings_window::SettingsWindow::section_header(ui, &crate::t!("heic.system_title"));
    ui.label(crate::t!("heic.system_description"));
    ui.label(crate::io::heic::setup_message());
    let caps = crate::io::heic::display_capabilities();
    ui.label(format!(
        "{}: {}   {}: {}",
        crate::t!("heic.import"),
        crate::t!(if caps.decode {
            "heic.available"
        } else {
            "heic.unavailable"
        }),
        crate::t!("heic.export"),
        crate::t!(if caps.encode {
            "heic.available"
        } else {
            "heic.unavailable"
        })
    ));
    if !caps.detail.is_empty() {
        ui.weak(&caps.detail);
    }
    save(&ctx, state);
}
pub fn show_heic_library_settings(
    ui: &mut Ui,
    settings: &mut crate::config::settings::AppSettings,
) {
    #[cfg(target_os = "windows")]
    {
        ui.add_space(8.0);
        ui.strong(crate::t!("heic.custom_library"));
        ui.label(crate::t!("heic.custom_hint"));
        ui.collapsing(crate::t!("heic.setup_instructions"), |ui| {
            ui.label(crate::t!("heic.manual_setup"));
            ui.hyperlink_to(
                crate::t!("heic.setup_instructions"),
                "https://github.com/kylejckson/PaintFE/blob/main/docs/heic-libraries.md",
            );
            ui.hyperlink_to("libheif", "https://github.com/strukturag/libheif");
        });
        if !settings.heic_library_path.is_empty() {
            ui.label(
                std::path::Path::new(&settings.heic_library_path)
                    .display()
                    .to_string(),
            );
        }
        ui.horizontal_wrapped(|ui| {
            if ui.button(crate::t!("heic.choose_library")).clicked()
                && let Some(path) = crate::services::file_dialogs::resource_dialog()
                    .add_filter("libheif", &["dll"])
                    .pick_file()
            {
                settings.heic_library_path = path.to_string_lossy().into_owned();
                crate::io::heic::set_library_path(&settings.heic_library_path);
                settings.save();
            }
            if ui.button(crate::t!("heic.use_system")).clicked() {
                settings.heic_library_path.clear();
                crate::io::heic::set_library_path("");
                settings.save();
            }
            let id = Id::new("heic_library_test_running");
            let running = ui.ctx().data(|d| d.get_temp::<bool>(id)).unwrap_or(false);
            if ui
                .add_enabled(!running, egui::Button::new(crate::t!("heic.test_library")))
                .clicked()
            {
                let ctx = ui.ctx().clone();
                ctx.data_mut(|d| d.insert_temp(id, true));
                std::thread::spawn(move || {
                    crate::io::heic::forget_capabilities();
                    crate::io::heic::capabilities();
                    ctx.data_mut(|d| d.insert_temp(id, false));
                    ctx.request_repaint();
                });
            }
            if running {
                ui.spinner();
            }
        });
    }
    #[cfg(not(target_os = "windows"))]
    let _ = (ui, settings);
}

pub fn frame(ctx: &Context, settings: &mut crate::config::settings::AppSettings) -> bool {
    let mut state = state(ctx);
    let result = state
        .job
        .as_ref()
        .and_then(|rx| match rx.lock().ok()?.try_recv() {
            Ok(result) => Some(result),
            Err(mpsc::TryRecvError::Disconnected) => Some(Err(
                "Optional-pack worker stopped unexpectedly. Please retry.".into(),
            )),
            Err(mpsc::TryRecvError::Empty) => None,
        });
    if let Some(result) = result {
        state.job = None;
        match result {
            Ok((pack, font)) => {
                state.installed[index(pack)] = packs::is_present(pack);
                state.status[index(pack)] = crate::t!(if state.installed[index(pack)] {
                    "packs.installed"
                } else {
                    "packs.not_installed"
                });
                state.message = Some(crate::t!(if state.installed[index(pack)] {
                    "packs.ready"
                } else {
                    "packs.removed"
                }));
                if pack == Pack::Cjk {
                    apply_font(ctx, font);
                    if state.installed[0] {
                        if let Some(language) = state.pending_language.take() {
                            settings.language = if state.pending_auto {
                                String::new()
                            } else {
                                language.clone()
                            };
                            crate::i18n::set_language(&language);
                            settings.save();
                        }
                    } else {
                        let language = crate::i18n::current_language();
                        if !available(ctx, &language) {
                            crate::i18n::set_language("en");
                        }
                    }
                }
                state.prompt = None;
            }
            Err(error) => state.message = Some(error),
        }
    }
    if state.job.is_some() {
        ctx.request_repaint_after(std::time::Duration::from_millis(100));
    }
    if packs::take_jxl_request() && !state.installed[1] {
        state.prompt = Some(Pack::Jxl);
    }
    if packs::take_avif_request() && !state.installed[2] {
        state.prompt = Some(Pack::Avif);
    }
    let open = state.prompt.is_some();
    ctx.data_mut(|d| d.insert_temp(Id::new("optional_pack_modal_open"), open));
    if let Some(pack) = state.prompt {
        egui::Modal::new(Id::new("optional_pack_confirmation")).show(ctx, |ui| {
            ui.set_max_width(460.0);
            ui.set_min_width(300.0);
            ui.heading(pack.title());
            ui.label(crate::t!("packs.download_warning"));
            ui.label(crate::t!(if pack == Pack::Cjk {
                "packs.cjk_size"
            } else {
                if pack == Pack::Heic {
                    "packs.heic_size"
                } else if pack == Pack::Avif {
                    "packs.avif_size"
                } else {
                    "packs.jxl_size"
                }
            }));
            if let Some(size) = match pack {
                Pack::Cjk => option_env!("PAINTFE_CJK_PACK_SIZE"),
                Pack::Jxl => option_env!("PAINTFE_JXL_PACK_SIZE"),
                Pack::Avif => option_env!("PAINTFE_AVIF_PACK_SIZE"),
                Pack::Heic => option_env!("PAINTFE_HEIC_PACK_SIZE"),
            } && let Ok(size) = size.parse::<u64>()
            {
                ui.label(format!("{:.2} MiB", size as f64 / 1048576.0));
            }
            ui.weak(crate::t!("packs.github"));
            if let Some(message) = &state.message {
                ui.colored_label(ui.visuals().error_fg_color, message);
            }
            ui.add_enabled_ui(state.job.is_none(), |ui| {
                ui.horizontal_wrapped(|ui| {
                    if ui.button(crate::t!("packs.download")).clicked() {
                        start(ctx, &mut state, pack, None, false);
                    }
                    if ui.button(crate::t!("packs.import")).clicked() {
                        import(ctx, &mut state, pack);
                    }
                    if ui.button(crate::t!("packs.later")).clicked() {
                        state.prompt = None;
                        if let Some(l) = state.pending_language.take() {
                            state.skipped.push(l);
                        }
                    }
                });
            });
            if state.job.is_some() {
                ui.spinner();
            }
        });
    }
    if open
        && state.job.is_none()
        && ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::Escape))
    {
        state.prompt = None;
        state.pending_language = None;
    }
    save(ctx, state);
    open
}
#[cfg(test)]
mod tests {
    use super::*;
    fn text_center(shapes: &[egui::epaint::ClippedShape], needle: &str) -> egui::Pos2 {
        fn find(shape: &egui::Shape, needle: &str) -> Option<egui::Pos2> {
            match shape {
                egui::Shape::Text(text) if text.galley.job.text == needle => {
                    Some(text.pos + text.galley.size() * 0.5)
                }
                egui::Shape::Vec(shapes) => shapes.iter().find_map(|shape| find(shape, needle)),
                _ => None,
            }
        }
        shapes
            .iter()
            .find_map(|shape| find(&shape.shape, needle))
            .expect("button label missing")
    }
    #[test]
    fn settings_fit_light_dark_and_large_text_without_horizontal_overflow() {
        for dark in [false, true] {
            for scale in [1.0, 1.5] {
                let ctx = Context::default();
                ctx.set_visuals(if dark {
                    egui::Visuals::dark()
                } else {
                    egui::Visuals::light()
                });
                ctx.style_mut_of(
                    if dark {
                        egui::Theme::Dark
                    } else {
                        egui::Theme::Light
                    },
                    |style| {
                        for font in style.text_styles.values_mut() {
                            font.size *= scale;
                        }
                    },
                );
                let input = egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(480.0, 1000.0),
                    )),
                    ..Default::default()
                };
                let output = ctx.run_ui(input, |ui| {
                    ui.set_width(440.0);
                    show_settings(ui);
                    show_heic_library_settings(
                        ui,
                        &mut crate::config::settings::AppSettings::default(),
                    );
                    assert!(ui.min_rect().width() <= 442.0, "{:?}", ui.min_rect());
                });
                text_center(
                    &output.shapes,
                    &crate::t!("heic.system_title").to_uppercase(),
                );
                assert!(state(&ctx).job.is_none());
                assert!(state(&ctx).prompt.is_none());
            }
        }
    }
    #[test]
    fn download_button_opens_confirmation_without_starting_network() {
        let ctx = Context::default();
        let input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(480.0, 1000.0),
            )),
            ..Default::default()
        };
        let output = ctx.run_ui(input.clone(), |ui| {
            ui.set_width(440.0);
            show_settings(ui);
        });
        let point = text_center(&output.shapes, &crate::t!("packs.download"));
        for pressed in [true, false] {
            let mut input = input.clone();
            input.events = vec![
                egui::Event::PointerMoved(point),
                egui::Event::PointerButton {
                    pos: point,
                    button: egui::PointerButton::Primary,
                    pressed,
                    modifiers: Default::default(),
                },
            ];
            let _ = ctx.run_ui(input, |ui| {
                ui.set_width(440.0);
                show_settings(ui);
            });
        }
        assert_eq!(state(&ctx).prompt, Some(Pack::Cjk));
        assert!(state(&ctx).job.is_none());
    }
    #[test]
    fn disconnected_worker_is_reported_instead_of_spinning_forever() {
        let ctx = Context::default();
        let (tx, rx) = mpsc::channel();
        drop(tx);
        let initial = State {
            job: Some(Arc::new(Mutex::new(rx))),
            ..Default::default()
        };
        save(&ctx, initial);
        let mut settings = crate::config::settings::AppSettings::default();
        let _ = ctx.run_ui(egui::RawInput::default(), |_| {
            frame(&ctx, &mut settings);
        });
        assert!(state(&ctx).job.is_none());
        assert!(state(&ctx).message.is_some());
    }
    #[test]
    fn later_and_escape_do_not_change_saved_language_or_start_download() {
        let ctx = Context::default();
        save(
            &ctx,
            State {
                prompt: Some(Pack::Cjk),
                pending_language: Some("ko".into()),
                ..Default::default()
            },
        );
        let mut settings = crate::config::settings::AppSettings {
            language: "fr".into(),
            ..Default::default()
        };
        let input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(900.0, 700.0),
            )),
            ..Default::default()
        };
        for _ in 0..2 {
            let _ = ctx.run_ui(input.clone(), |_| {
                frame(&ctx, &mut settings);
            });
        }
        let output = ctx.run_ui(input.clone(), |_| {
            frame(&ctx, &mut settings);
        });
        let point = text_center(&output.shapes, &crate::t!("packs.later"));
        for pressed in [true, false] {
            let mut input = input.clone();
            input.events = vec![
                egui::Event::PointerMoved(point),
                egui::Event::PointerButton {
                    pos: point,
                    button: egui::PointerButton::Primary,
                    pressed,
                    modifiers: Default::default(),
                },
            ];
            let _ = ctx.run_ui(input, |_| {
                frame(&ctx, &mut settings);
            });
        }
        assert!(state(&ctx).prompt.is_none());
        assert!(state(&ctx).job.is_none());
        assert_eq!(settings.language, "fr");
        save(
            &ctx,
            State {
                prompt: Some(Pack::Jxl),
                ..Default::default()
            },
        );
        let mut escape = input;
        escape.events = vec![egui::Event::Key {
            key: egui::Key::Escape,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: Default::default(),
        }];
        let _ = ctx.run_ui(escape, |_| {
            frame(&ctx, &mut settings);
        });
        assert!(state(&ctx).prompt.is_none());
    }
}

#[cfg(test)]
mod font_order_tests {
    use super::*;
    #[test]
    fn optional_font_preserves_primary_and_regional_system_font_priority() {
        let mut fonts = FontDefinitions::default();
        let expected = vec![
            "dm_sans".to_string(),
            "noto_sans".into(),
            "system_cjk".into(),
            "fallback".into(),
        ];
        for family in [
            FontFamily::Proportional,
            FontFamily::Monospace,
            FontFamily::Name("WidgetTitle".into()),
        ] {
            fonts.families.insert(family, expected.clone());
        }
        add_fallback(&mut fonts, "optional_cjk");
        assert_eq!(
            fonts.families[&FontFamily::Proportional],
            vec![
                "dm_sans",
                "noto_sans",
                "system_cjk",
                "optional_cjk",
                "fallback"
            ]
        );
        let mut minimal = FontDefinitions::default();
        add_fallback(&mut minimal, "optional_cjk");
        assert!(
            minimal.families[&FontFamily::Name("WidgetTitle".into())]
                .contains(&"optional_cjk".to_string())
        );
    }
}

#[cfg(test)]
mod language_prompt_tests {
    use super::*;
    #[test]
    fn missing_cjk_prompts_before_switching_and_retains_auto_detection() {
        if packs::is_present(Pack::Cjk) {
            eprintln!(
                "Use an isolated PAINTFE_TEST_STORAGE_DIR without CJK for the missing-font startup case."
            );
            return;
        }
        crate::i18n::init();
        let previous = crate::i18n::current_language();
        crate::i18n::set_language("ko");
        let ctx = Context::default();
        initialize_fonts(&ctx, FontDefinitions::default(), Vec::new(), true);
        let result = state(&ctx);
        assert_eq!(result.prompt, Some(Pack::Cjk));
        assert_eq!(result.pending_language.as_deref(), Some("ko"));
        assert!(result.pending_auto);
        assert!(result.job.is_none());
        assert_eq!(crate::i18n::current_language(), "en");
        crate::i18n::set_language(&previous);
    }
    #[test]
    fn suitable_system_font_avoids_download_prompt_and_keeps_requested_language() {
        crate::i18n::init();
        let previous = crate::i18n::current_language();
        crate::i18n::set_language("ja");
        let ctx = Context::default();
        initialize_fonts(&ctx, FontDefinitions::default(), vec!["ja".into()], true);
        assert!(state(&ctx).prompt.is_none());
        assert!(state(&ctx).job.is_none());
        assert_eq!(crate::i18n::current_language(), "ja");
        crate::i18n::set_language(&previous);
    }
}

#[cfg(test)]
mod issue_195_activation_tests {
    use super::*;
    #[test]
    fn font_changes_invalidate_layout_measurements_without_restart() {
        let ctx = Context::default();
        initialize_fonts(&ctx, FontDefinitions::default(), Vec::new(), false);
        let before = ctx
            .data(|d| d.get_temp::<u64>(Id::new("font_layout_revision")))
            .unwrap();
        apply_font(
            &ctx,
            Some(include_bytes!("../../../assets/fonts/NotoSansArabic.ttf").to_vec()),
        );
        assert!(
            ctx.data(|d| d.get_temp::<u64>(Id::new("font_layout_revision")))
                .unwrap()
                > before
        );
    }
    #[test]
    fn opening_catalogue_always_includes_jxl_with_no_pack_dependency() {
        assert!(crate::io::IMAGE_EXTENSIONS.contains(&"jxl"));
        assert!(crate::io::document_extensions().contains(&"jxl"));
        let mut unique = crate::io::document_extensions();
        let length = unique.len();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), length);
    }
}
