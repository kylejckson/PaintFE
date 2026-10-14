use super::*;

#[test]
fn preview_burst_keeps_only_latest_request_and_original_document() {
    let mut app = app();
    let ctx = egui::Context::default();
    let original = app.projects[0].canvas_state.layers[0].pixels.clone();
    let flat = original.to_rgba_image();
    let (started_tx, started_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    app.spawn_preview_job(
        0.,
        "blocked".into(),
        0,
        original.clone(),
        flat.clone(),
        move |img| {
            started_tx.send(()).unwrap();
            release_rx
                .recv_timeout(std::time::Duration::from_secs(10))
                .unwrap();
            img.clone()
        },
    );
    started_rx
        .recv_timeout(std::time::Duration::from_secs(10))
        .unwrap();
    let executed = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    for value in 1..=100u8 {
        let executed = executed.clone();
        app.spawn_preview_job(
            1.,
            "latest".into(),
            0,
            original.clone(),
            flat.clone(),
            move |img| {
                executed.lock().unwrap().push(value);
                image::RgbaImage::from_pixel(
                    img.width(),
                    img.height(),
                    image::Rgba([value, 0, 0, 255]),
                )
            },
        );
        assert_eq!(app.pending_filter_jobs, 1);
        assert!(app.pending_preview.is_some());
    }
    app.projects.push(Project::new_untitled(2, 16, 16));
    app.active_project_index = 1;
    let other = app.projects[1].canvas_state.layers[0]
        .pixels
        .to_rgba_image();
    release_tx.send(()).unwrap();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while app.pending_filter_jobs > 0 {
        assert!(std::time::Instant::now() < deadline);
        app.update_runtime_lifecycle_async(&ctx);
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    assert_eq!(*executed.lock().unwrap(), [100]);
    assert!(!app.preview_job_running);
    assert!(app.pending_preview.is_none());
    assert_eq!(
        app.projects[0].canvas_state.layers[0]
            .pixels
            .to_rgba_image()
            .get_pixel(0, 0)[0],
        100
    );
    assert_eq!(
        app.projects[1].canvas_state.layers[0]
            .pixels
            .to_rgba_image(),
        other
    );
}

#[test]
fn cancel_preview_drops_queued_work_and_preserves_pixels() {
    let mut app = app();
    let ctx = egui::Context::default();
    let original = app.projects[0].canvas_state.layers[0].pixels.clone();
    let flat = original.to_rgba_image();
    let (started_tx, started_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    app.spawn_preview_job(
        0.,
        "blocked".into(),
        0,
        original.clone(),
        flat.clone(),
        move |img| {
            started_tx.send(()).unwrap();
            release_rx
                .recv_timeout(std::time::Duration::from_secs(10))
                .unwrap();
            img.clone()
        },
    );
    started_rx
        .recv_timeout(std::time::Duration::from_secs(10))
        .unwrap();
    app.spawn_preview_job(1., "must not run".into(), 0, original, flat.clone(), |_| {
        panic!("cancelled queue ran")
    });
    app.cancel_preview_jobs();
    release_tx.send(()).unwrap();
    let result = app
        .filter_receiver
        .recv_timeout(std::time::Duration::from_secs(10))
        .unwrap();
    app.filter_sender.send(result).unwrap();
    app.update_runtime_lifecycle_async(&ctx);
    assert_eq!(app.pending_filter_jobs, 0);
    assert!(!app.preview_job_running);
    assert!(app.pending_preview.is_none());
    assert_eq!(
        app.projects[0].canvas_state.layers[0]
            .pixels
            .to_rgba_image(),
        flat
    );
}

fn app() -> PaintFEApp {
    // Match main's startup so localized errors interpolate filenames in tests.
    crate::i18n::init();
    let cc = eframe::CreationContext::_new_kittest(egui::Context::default());
    let mut app = PaintFEApp::new(&cc, Vec::new(), mpsc::channel().1);
    app.projects = vec![Project::new_untitled(1, 16, 16)];
    app.active_project_index = 0;
    app
}

fn selection_key(app: &mut PaintFEApp, ctx: &egui::Context, key: egui::Key) {
    let _ = ctx.run_ui(
        egui::RawInput {
            events: vec![egui::Event::Key {
                key,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: egui::Modifiers::NONE,
            }],
            ..Default::default()
        },
        |ui| {
            app.update_runtime_input(ui.ctx());
        },
    );
}

#[test]
fn selection_pixel_shortcuts_can_swap_backspace_and_delete_and_undo() {
    use crate::config::keybindings::{BindableAction, KeyCombo};
    let mut app = app();
    let ctx = egui::Context::default();
    app.projects[0].canvas_state.selection_all = true;
    app.settings.keybindings.bindings.insert(
        BindableAction::ClearSelectedPixels,
        KeyCombo::key(egui::Key::Backspace),
    );
    app.settings.keybindings.bindings.insert(
        BindableAction::FillSelectedPixels,
        KeyCombo::key(egui::Key::Delete),
    );
    let original = app.projects[0].canvas_state.layers[0]
        .pixels
        .to_rgba_image();
    selection_key(&mut app, &ctx, egui::Key::Backspace);
    assert!(
        app.projects[0].canvas_state.layers[0]
            .pixels
            .to_rgba_image()
            .pixels()
            .all(|p| p[3] == 0)
    );
    assert_eq!(app.projects[0].history.undo_count(), 1);
    assert!(app.projects[0].canvas_state.has_selection());
    let project = &mut app.projects[0];
    project.history.undo(&mut project.canvas_state);
    assert_eq!(
        project.canvas_state.layers[0].pixels.to_rgba_image(),
        original
    );
    selection_key(&mut app, &ctx, egui::Key::Delete);
    let primary = app.colors_panel.get_primary_color();
    assert!(
        app.projects[0].canvas_state.layers[0]
            .pixels
            .to_rgba_image()
            .pixels()
            .all(|p| p.0 == [primary.r(), primary.g(), primary.b(), primary.a()])
    );
    assert_eq!(app.projects[0].history.undo_count(), 1);
}

#[test]
fn selection_pixel_shortcuts_respect_modal_text_and_rebinding_focus() {
    for mode in 0..3 {
        let mut app = app();
        let ctx = egui::Context::default();
        app.projects[0].canvas_state.selection_all = true;
        let original = app.projects[0].canvas_state.layers[0]
            .pixels
            .to_rgba_image();
        match mode {
            0 => app.settings_window.open = true,
            1 => app.settings_window.rebinding_action = Some(BindableAction::ClearSelectedPixels),
            _ => {
                let mut text = String::from("editing");
                let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
                    ui.text_edit_singleline(&mut text).request_focus();
                });
                assert!(ctx.text_edit_focused());
            }
        }
        selection_key(&mut app, &ctx, egui::Key::Delete);
        assert_eq!(app.projects[0].history.undo_count(), 0);
        assert_eq!(
            app.projects[0].canvas_state.layers[0]
                .pixels
                .to_rgba_image(),
            original
        );
    }
}

#[test]
fn selection_pixel_shortcuts_keep_layer_delete_and_paste_cancel_separate() {
    let mut app = app();
    let ctx = egui::Context::default();
    app.projects[0].canvas_state.selection_all = true;
    let original = app.projects[0].canvas_state.layers[0]
        .pixels
        .to_rgba_image();
    app.paste_overlay = Some(crate::ops::clipboard::PasteOverlay::new(
        image::RgbaImage::from_pixel(2, 2, image::Rgba([255, 0, 0, 255])),
        16,
        16,
    ));
    selection_key(&mut app, &ctx, egui::Key::Backspace);
    assert!(app.paste_overlay.is_none());
    assert_eq!(app.projects[0].history.undo_count(), 0);
    assert_eq!(
        app.projects[0].canvas_state.layers[0]
            .pixels
            .to_rgba_image(),
        original
    );
    app.is_pointer_over_layers_panel = true;
    selection_key(&mut app, &ctx, egui::Key::Delete);
    // Delete over Layers performs a layer operation, not a pixel operation.
    assert!(app.projects[0].canvas_state.layers.is_empty());
    assert_eq!(app.projects[0].history.undo_count(), 1);
}

#[test]
fn colors_resize_keeps_anchor_and_stays_stable_across_frames() {
    for snapping in [false, true] {
        let ctx = egui::Context::default();
        let cc = eframe::CreationContext::_new_kittest(ctx.clone());
        let mut app = PaintFEApp::new(&cc, Vec::new(), mpsc::channel().1);
        app.window_visibility.colors = true;
        app.settings.persist_colors_panel_size = Some((220., 310.));
        app.colors_panel_pos = Some((100., 150.));
        let mut previous = None;
        for frame in 0..12 {
            let mut events = Vec::new();
            if frame == 4 {
                let (position, size): (egui::Pos2, egui::Vec2) = previous.unwrap();
                let from = position + size - egui::vec2(5., 5.);
                let to = from - egui::vec2(55., 65.);
                events = vec![
                    egui::Event::PointerMoved(from),
                    egui::Event::PointerButton {
                        pos: from,
                        button: egui::PointerButton::Primary,
                        pressed: true,
                        modifiers: Default::default(),
                    },
                    egui::Event::PointerMoved(to),
                    egui::Event::PointerButton {
                        pos: to,
                        button: egui::PointerButton::Primary,
                        pressed: false,
                        modifiers: Default::default(),
                    },
                ];
            }
            let _ = ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(1000., 800.),
                    )),
                    time: Some(frame as f64),
                    events,
                    ..Default::default()
                },
                |_| {
                    crate::ui::workspace::begin_frame(
                        &ctx,
                        &crate::ui::workspace::WorkspaceSettings {
                            snapping,
                            ..Default::default()
                        },
                        &[("Colors", true)],
                    );
                    app.show_floating_colors_panel(&ctx, false);
                },
            );
            let (x, y) = app.colors_panel_pos.unwrap();
            let (w, h) = app.settings.persist_colors_panel_size.unwrap();
            assert!(
                (x - 100.).abs() < 1. && (y - 150.).abs() < 1.,
                "frame {frame}: {x},{y}"
            );
            if frame >= 6 {
                assert!(w < 200. && h < 280., "frame {frame}: {w} x {h}");
                let (_, old) = previous.unwrap();
                assert!(
                    (old.x - w).abs() < 1. && (old.y - h).abs() < 1.,
                    "size drift at frame {frame}: {old:?} -> {w},{h}"
                );
            }
            previous = Some((egui::pos2(x, y), egui::vec2(w, h)));
        }
    }
}

#[test]
fn duplicate_document_has_independent_pixels_identity_path_and_history() {
    let mut app = app();
    app.projects[0].path = Some(PathBuf::from("original.png"));
    let id = app.projects[0].id;
    app.duplicate_project(id);
    assert_eq!(app.projects.len(), 2);
    assert_ne!(app.projects[1].id, id);
    assert!(app.projects[1].path.is_none());
    assert!(!app.projects[1].file_handler.has_current_path());
    assert!(app.projects[1].is_dirty);
    assert!(!app.projects[1].history.can_undo());
    app.projects[1].canvas_state.layers[0]
        .pixels
        .put_pixel(0, 0, image::Rgba([10, 20, 30, 255]));
    assert_ne!(
        app.projects[0].canvas_state.layers[0]
            .pixels
            .get_pixel(0, 0),
        app.projects[1].canvas_state.layers[0]
            .pixels
            .get_pixel(0, 0)
    );
}

#[test]
fn save_as_prefers_document_directory_and_remembered_format_for_all_files() {
    let mut app = app();
    app.settings.last_export_format = "jpg".into();
    app.settings.last_export_directory = "export-directory".into();
    app.open_save_as_for_project(0);
    assert_eq!(app.save_file_dialog.format(), SaveFormat::Jpeg);
    assert_eq!(
        app.save_file_dialog.target_directory,
        Some(PathBuf::from("export-directory"))
    );
    app.projects[0].path = Some(PathBuf::from("document-directory/original.pfe"));
    app.open_save_as_for_project(0);
    assert_eq!(app.save_file_dialog.format(), SaveFormat::Jpeg);
    assert_eq!(
        app.save_file_dialog.target_directory,
        Some(PathBuf::from("document-directory"))
    );
    app.projects[0].path = Some(PathBuf::from("document-directory/photo.png"));
    app.settings.last_export_format = "webp".into();
    app.open_save_as_for_project(0);
    assert_eq!(app.save_file_dialog.format(), SaveFormat::Webp);
}
#[test]
fn dropped_native_image_retains_path_and_save_as_defaults() {
    let mut app = app();
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target/drop-tests")
        .join(uuid::Uuid::new_v4().to_string());
    std::fs::create_dir_all(&root).unwrap();
    let source = root.join("пример image.png");
    image::RgbaImage::from_pixel(8, 6, image::Rgba([12, 34, 56, 255]))
        .save(&source)
        .unwrap();
    app.queue_import_or_open(source.clone(), 1.0);
    let preflight = app
        .io_receiver
        .recv_timeout(std::time::Duration::from_secs(10))
        .unwrap();
    app.io_sender.send(preflight).unwrap();
    let ctx = egui::Context::default();
    let _ = ctx.run_ui(egui::RawInput::default(), |_| {
        app.update_runtime_lifecycle_async(&ctx);
    });
    let item = app.pending_import_queue.remove(0);
    assert_eq!(item.source_path.as_ref(), Some(&source));
    app.apply_import_choice(&item, 0, 2.0);
    assert_eq!(app.io_ops_start_time, Some(2.0));
    let result = app
        .io_receiver
        .recv_timeout(std::time::Duration::from_secs(10))
        .unwrap();
    app.io_sender.send(result).unwrap();
    let ctx = egui::Context::default();
    let _ = ctx.run_ui(egui::RawInput::default(), |_| {
        app.update_runtime_lifecycle_async(&ctx);
    });
    let index = app.active_project_index;
    assert_eq!(app.projects[index].path.as_ref(), Some(&source));
    assert_eq!(
        app.projects[index].file_handler.current_path.as_ref(),
        Some(&source)
    );
    assert!(!app.projects[index].is_dirty);
    assert_eq!(app.projects[index].canvas_state.width, 8);
    app.settings.last_export_format = "webp".into();
    app.open_save_as_for_project(index);
    assert_eq!(app.save_file_dialog.target_directory.as_ref(), Some(&root));
    assert_eq!(app.save_file_dialog.format(), SaveFormat::Webp);
    let count = app.projects.len();
    app.apply_import_choice(&item, 0, 3.0);
    assert_eq!(app.projects.len(), count);
    assert!(app.pending_open_paths.is_empty());
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn dropped_layer_cancel_and_pathless_bytes_preserve_document_identity() {
    let mut app = app();
    let original = PathBuf::from("original/document.pfe");
    app.projects[0].path = Some(original.clone());
    let mut encoded = std::io::Cursor::new(Vec::new());
    image::RgbaImage::from_pixel(4, 4, image::Rgba([12, 34, 56, 255]))
        .write_to(&mut encoded, image::ImageFormat::Png)
        .unwrap();
    let item = PendingImport {
        bytes: std::sync::Arc::new(encoded.into_inner()),
        name: "Dropped Image.png".into(),
        source_path: Some(PathBuf::from("different/source.png")),
        width: 4,
        height: 4,
    };
    let layers = app.projects[0].canvas_state.layers.len();
    app.apply_import_choice(&item, 2, 0.0);
    assert_eq!(app.projects[0].canvas_state.layers.len(), layers);
    app.apply_import_choice(&item, 1, 0.0);
    assert_eq!(app.projects[0].canvas_state.layers.len(), layers + 1);
    assert_eq!(app.projects[0].path.as_ref(), Some(&original));
    app.apply_import_choice(
        &PendingImport {
            source_path: None,
            ..item
        },
        0,
        0.0,
    );
    assert!(app.projects[app.active_project_index].path.is_none());
    assert_eq!(app.projects[app.active_project_index].canvas_state.width, 4);
}

#[test]
fn failed_ai_job_preserves_pixels_history_and_clean_state() {
    let mut app = app();
    let ctx = egui::Context::default();
    let original = app.projects[0].canvas_state.layers[0].pixels.clone();
    let flat = original.to_rgba_image();
    let before = app.projects[0].history.undo_count();
    app.projects[0].is_dirty = false;
    app.background_removal_pending = true;
    app.spawn_fallible_filter_job(
        0.,
        "Remove Background".into(),
        0,
        original,
        flat.clone(),
        |_| Err("native inference failed".into()),
    );
    let result = app
        .filter_receiver
        .recv_timeout(std::time::Duration::from_secs(10))
        .unwrap();
    assert!(result.error.is_some());
    app.filter_sender.send(result).unwrap();
    app.update_runtime_lifecycle_async(&ctx);
    assert_eq!(
        app.projects[0].canvas_state.layers[0]
            .pixels
            .to_rgba_image(),
        flat
    );
    assert_eq!(app.projects[0].history.undo_count(), before);
    assert!(!app.projects[0].is_dirty);
    assert!(!app.background_removal_pending);
    assert_eq!(app.pending_filter_jobs, 0);
    assert!(app.filter_error.is_some());
}
#[test]
fn closing_a_tab_does_not_redirect_its_pending_filter_result() {
    let mut app = app();
    let ctx = egui::Context::default();
    let original = app.projects[0].canvas_state.layers[0].pixels.clone();
    app.projects.push(Project::new_untitled(2, 16, 16));
    let other = app.projects[1].canvas_state.layers[0]
        .pixels
        .to_rgba_image();
    app.spawn_filter_job(
        0.,
        "Remove Background".into(),
        0,
        original.clone(),
        original.to_rgba_image(),
        |input| {
            image::RgbaImage::from_pixel(
                input.width(),
                input.height(),
                image::Rgba([255, 0, 0, 255]),
            )
        },
    );
    let result = app
        .filter_receiver
        .recv_timeout(std::time::Duration::from_secs(10))
        .unwrap();
    app.close_project(0);
    assert_eq!(app.projects.len(), 1);
    app.filter_sender.send(result).unwrap();
    app.update_runtime_lifecycle_async(&ctx);
    assert_eq!(
        app.projects[0].canvas_state.layers[0]
            .pixels
            .to_rgba_image(),
        other
    );
    assert_eq!(app.projects[0].history.undo_count(), 0);
    assert!(!app.projects[0].is_dirty);
}
#[test]
fn keyboard_undo_removes_exactly_one_history_entry_per_press() {
    let mut app = app();
    let ctx = egui::Context::default();
    for index in 0..10 {
        app.projects[0]
            .history
            .push(Box::new(crate::components::history::MarkerCommand::new(
                format!("{index}"),
            )));
    }
    for remaining in (0..10).rev() {
        crate::windows_key_probe::observe_windows_message(0x0101, 0x5a);
        let _ = ctx.run_ui(
            egui::RawInput {
                modifiers: egui::Modifiers::CTRL,
                events: vec![egui::Event::Key {
                    key: egui::Key::Z,
                    physical_key: None,
                    pressed: false,
                    repeat: false,
                    modifiers: egui::Modifiers::CTRL,
                }],
                ..Default::default()
            },
            |ui| {
                app.update_runtime_input(ui.ctx());
            },
        );
        crate::windows_key_probe::observe_windows_message(0x0100, 0x5a);
        let _ = ctx.run_ui(
            egui::RawInput {
                modifiers: egui::Modifiers::CTRL,
                events: vec![egui::Event::Key {
                    key: egui::Key::Z,
                    physical_key: None,
                    pressed: true,
                    repeat: false,
                    modifiers: egui::Modifiers::CTRL,
                }],
                ..Default::default()
            },
            |ui| {
                app.update_runtime_input(ui.ctx());
            },
        );
        assert_eq!(app.projects[0].history.undo_count(), remaining);
    }
    crate::windows_key_probe::observe_windows_message(0x0101, 0x5a);
}
#[test]
fn keyboard_undo_works_with_tool_button_focus_but_preserves_text_field_undo() {
    for text_field in [false, true] {
        let mut app = app();
        let ctx = egui::Context::default();
        app.canvas.canvas_widget_id = Some(egui::Id::new("regression_canvas"));
        app.projects[0]
            .history
            .push(Box::new(crate::components::history::MarkerCommand::new(
                "stroke",
            )));
        let mut text = String::from("editable field");
        let mut focused_id = None;
        let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
            let response = if text_field {
                ui.text_edit_singleline(&mut text)
            } else {
                ui.button("Brush")
            };
            response.request_focus();
            focused_id = Some(response.id);
        });
        assert_eq!(ctx.memory(|memory| memory.focused()), focused_id);
        assert!(ctx.egui_wants_keyboard_input());
        assert_eq!(ctx.text_edit_focused(), text_field);
        let _ = ctx.run_ui(
            egui::RawInput {
                modifiers: egui::Modifiers::CTRL,
                events: vec![egui::Event::Key {
                    key: egui::Key::Z,
                    physical_key: None,
                    pressed: true,
                    repeat: false,
                    modifiers: egui::Modifiers::CTRL,
                }],
                ..Default::default()
            },
            |ui| {
                app.update_runtime_input(ui.ctx());
                if text_field {
                    ui.text_edit_singleline(&mut text);
                } else {
                    let _ = ui.button("Brush");
                }
            },
        );
        assert_eq!(
            app.projects[0].history.undo_count(),
            usize::from(text_field)
        );
    }
}

#[test]
fn tab_close_preserves_dirty_projects_for_save_discard_cancel() {
    let mut app = app();
    app.projects[0].is_dirty = true;
    app.close_project(0);
    assert_eq!(app.pending_close_index, Some(0));
    assert_eq!(app.projects.len(), 1);
    app.pending_close_index = None;
    assert_eq!(app.projects.len(), 1); // Cancel.
    app.projects[0].is_dirty = false;
    app.close_project(0);
    assert!(app.projects.is_empty());
}

#[test]
fn invalid_image_imports_report_errors_without_changing_documents() {
    let mut app = app();
    let before = app.projects[0].canvas_state.layers[0]
        .pixels
        .to_rgba_image();
    app.open_image_from_bytes(b"not an image", Some("bad.png".into()));
    let item = PendingImport {
        bytes: std::sync::Arc::new(b"not an image".to_vec()),
        name: "layer.jpg".into(),
        source_path: None,
        width: 0,
        height: 0,
    };
    app.apply_import_choice(&item, 1, 0.0);
    let error = app.filter_error.as_deref().unwrap();
    assert!(error.contains("bad.png"), "{error}");
    assert!(error.contains("layer.jpg"), "{error}");
    assert_eq!(app.projects.len(), 1);
    assert_eq!(app.projects[0].canvas_state.layers.len(), 1);
    assert_eq!(
        app.projects[0].canvas_state.layers[0]
            .pixels
            .to_rgba_image(),
        before
    );
}

#[test]
fn renamed_script_fails_preflight_before_import_prompt() {
    let mut app = app();
    let source = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join(format!("invalid-drop-{}.jpg", uuid::Uuid::new_v4()));
    std::fs::write(&source, b"Write-Host 'not an image'").unwrap();
    app.queue_import_or_open(source.clone(), 0.0);
    let result = app
        .io_receiver
        .recv_timeout(std::time::Duration::from_secs(10))
        .unwrap();
    assert!(matches!(result, IoResult::LoadFailed { .. }));
    app.io_sender.send(result).unwrap();
    let ctx = egui::Context::default();
    let _ = ctx.run_ui(egui::RawInput::default(), |_| {
        app.update_runtime_lifecycle_async(&ctx);
    });
    assert!(app.pending_import_queue.is_empty());
    assert!(
        app.filter_error
            .as_deref()
            .unwrap()
            .contains(source.file_name().unwrap().to_str().unwrap())
    );
    assert_eq!(app.projects.len(), 1);
    assert_eq!(app.pending_io_ops, 0);
    std::fs::remove_file(source).unwrap();
}
