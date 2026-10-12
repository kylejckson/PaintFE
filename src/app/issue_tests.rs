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
    let cc = eframe::CreationContext::_new_kittest(egui::Context::default());
    let mut app = PaintFEApp::new(&cc, Vec::new(), mpsc::channel().1);
    app.projects = vec![Project::new_untitled(1, 16, 16)];
    app.active_project_index = 0;
    app
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
fn save_as_prefers_document_path_and_restores_export_preference_for_new_files() {
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
    assert_eq!(app.save_file_dialog.format(), SaveFormat::Pfe);
    assert_eq!(
        app.save_file_dialog.target_directory,
        Some(PathBuf::from("document-directory"))
    );
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
