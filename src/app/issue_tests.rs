use super::*;
fn app() -> PaintFEApp {
    let cc = eframe::CreationContext::_new_kittest(egui::Context::default());
    let mut app = PaintFEApp::new(&cc, Vec::new(), mpsc::channel().1);
    app.projects = vec![Project::new_untitled(1, 16, 16)];
    app.active_project_index = 0;
    app
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
