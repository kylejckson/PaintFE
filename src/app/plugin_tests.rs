use super::history_tests::{app, context};
use super::*;
use crate::services::plugins::Effect;
use crate::ui::panels::plugin_effect::{Applied, Target};

#[test]
#[ignore = "requires PAINTFE_TEST_THEME_COLLECTION pointing to the local collection packs"]
fn local_theme_collection_installs_previews_keeps_restores_and_removes() {
    use crate::services::plugins as packs;
    let source = PathBuf::from(
        std::env::var("PAINTFE_TEST_THEME_COLLECTION").expect("Collection pack directory"),
    );
    let at =
        std::env::temp_dir().join(format!("paintfe-theme-collection-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&at).unwrap();
    let mut paths: Vec<_> = std::fs::read_dir(source)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "pfeplugin"))
        .collect();
    paths.sort();
    assert_eq!(paths.len(), 11);
    let mut application = app();
    let ctx = context();
    let original_pixels = result(&application).target.pixels;
    let permissions = serde_json::to_value(&application.settings.automation).unwrap();
    for path in paths {
        let bytes = std::fs::read(path).unwrap();
        let manifest = packs::install(&bytes, &at, true, None).unwrap();
        let mut preferences = packs::Preferences::default();
        preferences
            .packs
            .entry(manifest.id.clone())
            .or_default()
            .current = manifest.version.clone();
        let installed = packs::installed(&at, &preferences)
            .into_iter()
            .find(|pack| pack.manifest.id == manifest.id)
            .unwrap();
        let before = application.settings.clone();
        application
            .preview_appearance_pack(&ctx, &installed)
            .unwrap();
        for _ in 0..3 {
            let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
                application.appearance_preview_dialog(ui.ctx())
            });
        }
        let _ = ctx.run_ui(
            egui::RawInput {
                events: vec![egui::Event::Key {
                    key: egui::Key::Escape,
                    physical_key: None,
                    pressed: true,
                    repeat: false,
                    modifiers: egui::Modifiers::NONE,
                }],
                ..Default::default()
            },
            |ui| application.appearance_preview_dialog(ui.ctx()),
        );
        assert_eq!(
            application.settings.export_theme_to_string(),
            before.export_theme_to_string(),
            "Restore {}",
            manifest.id
        );
        assert_eq!(application.settings.icon_pack_path, before.icon_pack_path);
        application
            .preview_appearance_pack(&ctx, &installed)
            .unwrap();
        for _ in 0..3 {
            let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
                application.appearance_preview_dialog(ui.ctx())
            });
        }
        let point = ctx
            .data(|data| data.get_temp::<egui::Rect>(egui::Id::new("appearance_keep_rect")))
            .unwrap()
            .center();
        for pressed in [true, false] {
            let _ = ctx.run_ui(
                egui::RawInput {
                    events: vec![
                        egui::Event::PointerMoved(point),
                        egui::Event::PointerButton {
                            pos: point,
                            button: egui::PointerButton::Primary,
                            pressed,
                            modifiers: egui::Modifiers::NONE,
                        },
                    ],
                    ..Default::default()
                },
                |ui| application.appearance_preview_dialog(ui.ctx()),
            );
        }
        assert!(ctx.data(|data| data.get_temp::<AppearancePreview>(egui::Id::new("appearance_preview"))).is_none(), "Keep {}", manifest.id);
        AppSettings::flush_pending_saves().unwrap();
        let restored = AppSettings::load();
        assert_eq!(
            restored.export_theme_to_string(),
            application.settings.export_theme_to_string()
        );
        assert_eq!(restored.icon_pack_path, application.settings.icon_pack_path);
        assert_eq!(
            serde_json::to_value(&application.settings.automation).unwrap(),
            permissions
        );
        assert_eq!(*result(&application).target.pixels, *original_pixels);
        packs::uninstall(&manifest.id, &at).unwrap();
    }
}

fn result(app: &PaintFEApp) -> Applied {
    let p = &app.projects[0];
    let state = &p.canvas_state;
    let pixels = state.layers[0]
        .pixels
        .extract_region_rgba(0, 0, state.width, state.height);
    let target = Target {
        document: p.id,
        layer: state.layers[0].runtime_id,
        revision: p.edit_revision,
        width: state.width,
        height: state.height,
        pixels: std::sync::Arc::new(pixels.clone()),
        mask: None,
        pack: "local.test".into(),
        effect: Effect {
            id: "invert".into(),
            name: "Plugin invert".into(),
            category: "Custom".into(),
            script: "effect.rhai".into(),
            description: String::new(),
            parameters: vec![],
            histogram: false,
        },
        source: "apply_invert();".into(),
        colours: crate::ops::scripting::DEFAULT_SCRIPT_COLOURS,
    };
    let mut output = pixels;
    for p in output.chunks_exact_mut(4) {
        p[0] = 255 - p[0];
    }
    Applied {
        target,
        pixels: output,
        values: Default::default(),
    }
}
#[test]
fn plugin_apply_is_one_exact_undo_redo_step() {
    let mut a = app();
    let r = result(&a);
    let before = r.target.pixels.as_ref().clone();
    let after = r.pixels.clone();
    a.apply_plugin_result(r).unwrap();
    assert_eq!(a.projects[0].history.undo_count(), 1);
    assert_eq!(
        a.projects[0].canvas_state.layers[0]
            .pixels
            .extract_region_rgba(0, 0, 32, 24),
        after
    );
    a.perform_undo();
    assert_eq!(
        a.projects[0].canvas_state.layers[0]
            .pixels
            .extract_region_rgba(0, 0, 32, 24),
        before
    );
    a.perform_redo();
    assert_eq!(
        a.projects[0].canvas_state.layers[0]
            .pixels
            .extract_region_rgba(0, 0, 32, 24),
        after
    );
}
#[test]
fn plugin_result_never_follows_the_current_document() {
    let mut a = app();
    let r = result(&a);
    let after = r.pixels.clone();
    a.projects.push(Project::new_untitled(2, 32, 24));
    a.active_project_index = 1;
    let other = a.projects[1].canvas_state.layers[0].pixels.to_rgba_image();
    a.apply_plugin_result(r).unwrap();
    assert_eq!(
        a.projects[1].canvas_state.layers[0].pixels.to_rgba_image(),
        other
    );
    assert_eq!(
        a.projects[0].canvas_state.layers[0]
            .pixels
            .extract_region_rgba(0, 0, 32, 24),
        after
    );
}
#[test]
fn stale_plugin_revision_and_invalid_buffers_do_not_edit() {
    let mut a = app();
    let r = result(&a);
    let before = r.target.pixels.clone();
    a.projects[0].mark_dirty();
    assert!(a.apply_plugin_result(r).is_err());
    let mut r = result(&a);
    r.pixels.pop();
    assert!(a.apply_plugin_result(r).is_err());
    assert_eq!(a.projects[0].history.undo_count(), 0);
    assert_eq!(
        a.projects[0].canvas_state.layers[0]
            .pixels
            .extract_region_rgba(0, 0, 32, 24),
        *before
    );
}
#[test]
fn plugin_result_uses_layer_identity_after_reordering() {
    let mut a = app();
    let r = result(&a);
    let after = r.pixels.clone();
    a.projects[0].canvas_state.layers.insert(
        0,
        crate::canvas::Layer::new("Other".into(), 32, 24, image::Rgba([1, 2, 3, 255])),
    );
    a.apply_plugin_result(r).unwrap();
    assert_eq!(
        a.projects[0].canvas_state.layers[0]
            .pixels
            .get_pixel(0, 0)
            .0,
        [1, 2, 3, 255]
    );
    assert_eq!(
        a.projects[0].canvas_state.layers[1]
            .pixels
            .extract_region_rgba(0, 0, 32, 24),
        after
    );
}
#[test]
fn unchanged_plugin_result_does_not_dirty_or_add_history() {
    let mut a = app();
    let mut r = result(&a);
    r.pixels = r.target.pixels.as_ref().clone();
    let revision = a.projects[0].edit_revision;
    a.apply_plugin_result(r).unwrap();
    assert_eq!(a.projects[0].history.undo_count(), 0);
    assert_eq!(a.projects[0].edit_revision, revision);
}
#[test]
fn plugin_run_rejects_a_mask_target() {
    let mut a = app();
    let e = result(&a).target.effect;
    a.projects[0].canvas_state.edit_layer_mask = true;
    assert!(
        a.start_plugin_effect(&context(), "local.test".into(), e, "apply_invert();".into())
            .is_err()
    );
}

#[test]
fn open_plugin_browser_blocks_document_input_and_releases_it_on_close() {
    let mut a = app();
    let ctx = context();
    crate::ui::panels::plugin_browser::open(&ctx);
    let _ = ctx.run_ui(egui::RawInput::default(), |ui| a.update_plugins(ui.ctx()));
    assert!(a.plugin_modal_open);
    assert!(a.document_modal_open());
    crate::ui::panels::plugin_browser::state(&ctx)
        .lock()
        .unwrap()
        .open = false;
    let _ = ctx.run_ui(egui::RawInput::default(), |ui| a.update_plugins(ui.ctx()));
    assert!(!a.plugin_modal_open);
}

#[test]
fn properties_capture_pointer_input_without_painting_the_canvas() {
    use crate::components::tools::Tool;
    let mut app = app();
    let ctx = context();
    crate::ui::panels::plugin_author::create(&ctx, "apply_invert();");
    let before = app.projects[0].canvas_state.layers[0]
        .pixels
        .to_rgba_image();
    let revision = app.projects[0].edit_revision;
    let render = |app: &mut PaintFEApp, events| {
        let _ = ctx.run_ui(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(1000., 700.),
                )),
                events,
                ..Default::default()
            },
            |ui| {
                app.reset_ui_cursor_blocking_rects();
                app.update_plugins(&ctx);
                app.show_runtime_canvas_tail(&ctx, ui, false);
            },
        );
    };
    for _ in 0..3 {
        render(&mut app, vec![]);
    }
    let point = app.canvas.last_image_rect.unwrap().center();
    for tool in [Tool::Brush, Tool::Pencil, Tool::Line, Tool::Fill] {
        app.tools_panel.change_tool(tool);
        render(
            &mut app,
            vec![
                egui::Event::PointerMoved(point),
                egui::Event::PointerButton {
                    pos: point,
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::NONE,
                },
            ],
        );
        render(
            &mut app,
            vec![egui::Event::PointerMoved(point + egui::vec2(12., 6.))],
        );
        render(
            &mut app,
            vec![egui::Event::PointerButton {
                pos: point + egui::vec2(12., 6.),
                button: egui::PointerButton::Primary,
                pressed: false,
                modifiers: egui::Modifiers::NONE,
            }],
        );
        assert!(app.document_modal_open());
        assert_eq!(
            app.projects[0].canvas_state.layers[0]
                .pixels
                .to_rgba_image(),
            before
        );
        assert_eq!(app.projects[0].history.undo_count(), 0);
        assert_eq!(app.projects[0].edit_revision, revision);
    }
    crate::ui::panels::plugin_author::preview(&ctx, "apply_invert();").unwrap();
    render(&mut app, vec![]);
    assert!(!app.document_modal_open());
}

#[test]
fn appearance_confirmation_owns_its_closing_frame() {
    let mut a = app();
    let ctx = context();
    ctx.data_mut(|d| {
        d.insert_temp(
            egui::Id::new("appearance_preview"),
            AppearancePreview {
                settings: a.settings.clone(),
                icons: a.settings.icon_pack_path.clone(),
            },
        )
    });
    let render = |a: &mut PaintFEApp, events| {
        let _ = ctx.run_ui(
            egui::RawInput {
                events,
                ..Default::default()
            },
            |ui| a.update_plugins(ui.ctx()),
        );
    };
    for _ in 0..3 {
        render(&mut a, vec![]);
    }
    let pos = ctx
        .data(|d| d.get_temp::<egui::Rect>(egui::Id::new("appearance_keep_rect")))
        .unwrap()
        .center();
    for pressed in [true, false] {
        render(
            &mut a,
            vec![
                egui::Event::PointerMoved(pos),
                egui::Event::PointerButton {
                    pos,
                    button: egui::PointerButton::Primary,
                    pressed,
                    modifiers: egui::Modifiers::NONE,
                },
            ],
        );
        assert!(
            a.plugin_modal_open,
            "The closing click must remain owned by the dialog"
        );
    }
    assert!(
        ctx.data(|d| d.get_temp::<AppearancePreview>(egui::Id::new("appearance_preview")))
            .is_none()
    );
    render(&mut a, vec![]);
    assert!(
        !a.plugin_modal_open,
        "Input must be released on the next frame"
    );
}

#[test]
fn escape_restores_appearance_and_keeps_the_closing_frame_blocked() {
    let mut a = app();
    let ctx = context();
    let original = a.settings.clone();
    ctx.data_mut(|d| {
        d.insert_temp(
            egui::Id::new("appearance_preview"),
            AppearancePreview {
                settings: original.clone(),
                icons: original.icon_pack_path.clone(),
            },
        )
    });
    a.settings.glass.enabled = true;
    for _ in 0..3 {
        let _ = ctx.run_ui(egui::RawInput::default(), |ui| a.update_plugins(ui.ctx()));
    }
    let _ = ctx.run_ui(
        egui::RawInput {
            events: vec![egui::Event::Key {
                key: egui::Key::Escape,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: egui::Modifiers::NONE,
            }],
            ..Default::default()
        },
        |ui| a.update_plugins(ui.ctx()),
    );
    assert_eq!(a.settings.glass, original.glass);
    assert!(a.plugin_modal_open);
    assert!(
        ctx.data(|d| d.get_temp::<AppearancePreview>(egui::Id::new("appearance_preview")))
            .is_none()
    );
    let _ = ctx.run_ui(egui::RawInput::default(), |ui| a.update_plugins(ui.ctx()));
    assert!(!a.plugin_modal_open);
}
