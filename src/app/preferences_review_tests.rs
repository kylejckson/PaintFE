use super::history_tests::{app, context, tool_frame};
use super::*;
use crate::components::tools::Tool;

#[test]
fn workspace_restore_and_reset_update_the_palette_without_unhiding_tools() {
    use crate::ui::workspace::{Command, PanelPlacement, WorkspaceLayout};
    let mut app = app();
    let ctx = context();
    app.settings.tools_palette.hidden = vec!["brush".into()];
    app.settings.tools_palette.mode = "row".into();
    app.settings.tools_palette.columns = 23;
    let _ = ctx.run_ui(
        egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(1200., 900.),
            )),
            ..Default::default()
        },
        |ui| {
            crate::ui::workspace::request(ui.ctx(), Command::Reset);
            app.apply_workspace_command(ui.ctx());
            assert_eq!(app.settings.tools_palette.columns, 3);
            assert_eq!(app.settings.tools_palette.mode, "auto");
            for (size, narrow) in [([62., 850.], true), ([800., 80.], false)] {
                crate::ui::workspace::request(
                    ui.ctx(),
                    Command::Apply(WorkspaceLayout {
                        version: 1,
                        name: "Palette test".into(),
                        viewport: [1200., 900.],
                        panels: vec![PanelPlacement {
                            panel: "Tools".into(),
                            position: [12., 12.],
                            size,
                            visible: true,
                            right_anchored: false,
                            expanded: None,
                            section_mask: None,
                        }],
                    }),
                );
                app.apply_workspace_command(ui.ctx());
                if narrow {
                    assert_eq!(app.settings.tools_palette.columns, 1);
                } else {
                    assert!(app.settings.tools_palette.columns >= 20);
                }
                assert_eq!(app.settings.tools_palette.hidden, ["brush"]);
            }
        },
    );
}

fn key(key: egui::Key, modifiers: egui::Modifiers) -> egui::RawInput {
    egui::RawInput {
        modifiers,
        events: vec![egui::Event::Key {
            key,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers,
        }],
        ..Default::default()
    }
}

#[test]
fn paste_placement_respects_source_cursor_visibility_and_large_copies() {
    let canvas = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(100., 100.));
    let visible = egui::Rect::from_min_max(egui::pos2(50., 50.), egui::pos2(100., 100.));
    let source = Some(egui::pos2(20., 20.));
    let cursor = Some(egui::pos2(80., 80.));
    let choose = |source, cursor, image, prefer_source, keep_visible| {
        PaintFEApp::paste_placement(
            image,
            canvas,
            source,
            cursor,
            Some(visible),
            prefer_source,
            keep_visible,
        )
    };
    let small = egui::vec2(10., 10.);
    assert_eq!(
        choose(source, cursor, small, true, false),
        egui::pos2(20., 20.)
    );
    assert_eq!(
        choose(source, cursor, small, true, true),
        egui::pos2(55., 55.)
    );
    assert_eq!(
        choose(source, cursor, small, false, true),
        egui::pos2(80., 80.)
    );
    assert_eq!(choose(None, None, small, true, true), visible.center());
    assert_eq!(
        choose(Some(egui::pos2(200., 200.)), cursor, small, true, true),
        egui::pos2(95., 95.)
    );
    assert_eq!(
        choose(None, cursor, egui::vec2(150., 150.), true, true),
        visible.center()
    );
    assert_eq!(
        choose(source, cursor, canvas.size(), false, true),
        canvas.center()
    );
}

#[test]
fn actual_paste_dispatch_uses_preferences_for_internal_and_external_images() {
    for prefer_source in [false, true] {
        for internal in [false, true] {
            let mut app = app();
            app.settings.paste_at_source_position = prefer_source;
            app.settings.paste_keep_visible = true;
            app.queue_paste_image(
                RgbaImage::from_pixel(4, 4, image::Rgba([1, 2, 3, 255])),
                Some((26., 18.)),
                Some(egui::pos2(6., 6.)),
                internal,
                false,
                None,
            );
            assert_eq!(
                app.paste_overlay.as_ref().unwrap().center,
                if prefer_source && internal {
                    egui::pos2(6., 6.)
                } else {
                    egui::pos2(26., 18.)
                }
            );
        }
    }
}

#[test]
fn enter_deselection_is_opt_in_and_undoable_for_idle_selection_tools() {
    assert!(!AppSettings::default().enter_deselect);
    for tool in [
        Tool::Brush,
        Tool::RectangleSelect,
        Tool::EllipseSelect,
        Tool::Lasso,
        Tool::MoveSelection,
    ] {
        for enabled in [false, true] {
            let mut app = app();
            let ctx = context();
            app.settings.enter_deselect = enabled;
            app.tools_panel.change_tool(tool);
            app.projects[0].canvas_state.selection_all = true;
            let _ = ctx.run_ui(key(egui::Key::Enter, egui::Modifiers::NONE), |ui| {
                app.update_runtime_input(ui.ctx());
            });
            tool_frame(&mut app, &ctx, (8., 8.), Vec::new());
            assert_eq!(
                app.projects[0].canvas_state.has_selection(),
                !enabled,
                "{tool:?} enabled={enabled}"
            );
            assert_eq!(app.projects[0].history.undo_count(), usize::from(enabled));
            if enabled {
                let project = &mut app.projects[0];
                assert_eq!(
                    project.history.presentation_history()[0].action,
                    HistoryAction::Deselect
                );
                project.history.undo(&mut project.canvas_state);
                assert!(project.canvas_state.selection_all);
            }
        }
    }
}

#[test]
fn enter_deselection_does_not_override_dialogs_or_active_tools() {
    for kind in ["dialog", "text", "shape", "mesh", "selection", "paste"] {
        let mut app = app();
        let ctx = context();
        app.settings.enter_deselect = true;
        app.projects[0].canvas_state.selection_all = true;
        match kind {
            "dialog" => app.new_file_dialog.open_dialog(),
            "text" => app.tools_panel.text_state.is_editing = true,
            "shape" => app.tools_panel.shapes_state.is_drawing = true,
            "mesh" => app.tools_panel.mesh_warp_state.is_active = true,
            "selection" => app.tools_panel.selection_state.dragging = true,
            "paste" => app.paste_overlay = Some(PasteOverlay::new(RgbaImage::new(4, 4), 32, 24)),
            _ => unreachable!(),
        }
        let _ = ctx.run_ui(key(egui::Key::Enter, egui::Modifiers::NONE), |ui| {
            app.update_runtime_input(ui.ctx());
        });
        assert!(app.projects[0].canvas_state.has_selection(), "{kind}");
        assert!(
            app.projects[0]
                .history
                .presentation_history()
                .iter()
                .all(|entry| entry.action != HistoryAction::Deselect)
        );
        if kind == "paste" {
            assert!(app.paste_overlay.is_none(), "Enter must still commit paste");
        }
    }
}

#[test]
fn traditional_clipboard_shortcuts_are_consumed_once_and_can_be_disabled() {
    use crate::config::keybindings::{BindableAction, KeyBindings};
    let bindings = KeyBindings::default();
    for (action, key_code, modifiers) in [
        (
            BindableAction::Copy,
            egui::Key::Insert,
            egui::Modifiers::CTRL,
        ),
        (
            BindableAction::Paste,
            egui::Key::Insert,
            egui::Modifiers::SHIFT,
        ),
        (
            BindableAction::Cut,
            egui::Key::Delete,
            egui::Modifiers::SHIFT,
        ),
    ] {
        for enabled in [false, true] {
            let ctx = context();
            let _ = ctx.run_ui(key(key_code, modifiers), |ui| {
                assert_eq!(
                    bindings.is_pressed_with_legacy(ui.ctx(), action, enabled),
                    enabled
                );
                assert!(!bindings.is_pressed_with_legacy(ui.ctx(), action, enabled));
            });
        }
    }
}

#[test]
fn unsaved_dialog_arrow_navigation_activates_discard_instead_of_default_save() {
    let mut app = app();
    let ctx = context();
    app.projects.push(Project::new_untitled(2, 32, 24));
    let remaining_id = app.projects[1].id;
    app.projects[0].mark_dirty();
    app.pending_close_index = Some(0);
    for input in [
        egui::RawInput::default(),
        key(egui::Key::ArrowRight, egui::Modifiers::NONE),
        key(egui::Key::Enter, egui::Modifiers::NONE),
    ] {
        let _ = ctx.run_ui(input, |ui| {
            app.handle_runtime_modal_flow(ui.ctx());
        });
    }
    assert!(app.pending_close_index.is_none());
    assert_eq!(app.projects.len(), 1);
    assert_eq!(app.projects[0].id, remaining_id);
    assert!(!app.save_file_dialog.open);
}

fn import_dialog_fixture(kind: usize) -> PaintFEApp {
    let mut app = app();
    let size = if kind == 0 { (4, 4) } else { (40, 30) };
    let image = RgbaImage::from_pixel(size.0, size.1, image::Rgba([20, 100, 230, 255]));
    if kind == 2 {
        app.queue_paste_image(image, None, None, false, false, None);
    } else {
        let mut bytes = std::io::Cursor::new(Vec::new());
        image.write_to(&mut bytes, image::ImageFormat::Png).unwrap();
        let item = PendingImport {
            bytes: std::sync::Arc::new(bytes.into_inner()),
            name: "keyboard-fixture.png".into(),
            source_path: None,
            width: size.0,
            height: size.1,
        };
        if kind == 0 {
            app.pending_import_queue.push(item);
        } else {
            app.pending_oversized_import = Some(item);
        }
    }
    app
}

fn import_keyboard_frame(app: &mut PaintFEApp, ctx: &egui::Context, input: egui::RawInput) {
    let _ = ctx.run_ui(input, |ui| {
        app.handle_runtime_modal_flow(ui.ctx());
    });
}

#[test]
fn import_and_paste_dialog_navigation_uses_the_focused_action() {
    for kind in 0..3 {
        for navigation in [egui::Key::ArrowDown, egui::Key::Tab] {
            let mut app = import_dialog_fixture(kind);
            let ctx = context();
            import_keyboard_frame(&mut app, &ctx, Default::default());
            import_keyboard_frame(&mut app, &ctx, key(navigation, egui::Modifiers::NONE));
            import_keyboard_frame(&mut app, &ctx, key(egui::Key::Enter, egui::Modifiers::NONE));
            assert!(
                !ctx.input(|i| i.key_pressed(egui::Key::Enter)),
                "Dialog activation must not also commit the new paste overlay"
            );
            assert_eq!(
                app.projects.len(),
                1,
                "kind {kind}, {navigation:?} must not open a new document"
            );
            assert_eq!(
                (
                    app.projects[0].canvas_state.width,
                    app.projects[0].canvas_state.height
                ),
                (32, 24),
                "Keep Canvas must not expand"
            );
            if kind == 2 {
                assert!(app.paste_overlay.is_some());
            } else {
                assert_eq!(app.projects[0].canvas_state.layers.len(), 2);
            }
            assert!(app.pending_import_queue.is_empty());
            assert!(app.pending_oversized_import.is_none());
            assert!(app.pending_paste_request.is_none());
        }
    }
}

#[test]
fn import_and_paste_dialogs_support_reverse_navigation_space_and_escape() {
    for kind in 0..3 {
        for cancel_with_escape in [false, true] {
            let mut app = import_dialog_fixture(kind);
            let ctx = context();
            import_keyboard_frame(&mut app, &ctx, Default::default());
            if cancel_with_escape {
                import_keyboard_frame(
                    &mut app,
                    &ctx,
                    key(egui::Key::Escape, egui::Modifiers::NONE),
                );
            } else {
                for (key_code, modifiers) in [
                    (egui::Key::Tab, egui::Modifiers::NONE),
                    (egui::Key::ArrowDown, egui::Modifiers::NONE),
                    (egui::Key::Tab, egui::Modifiers::SHIFT),
                    (egui::Key::ArrowUp, egui::Modifiers::NONE),
                    (egui::Key::ArrowDown, egui::Modifiers::NONE),
                    (egui::Key::ArrowDown, egui::Modifiers::NONE),
                    (egui::Key::Space, egui::Modifiers::NONE),
                ] {
                    import_keyboard_frame(&mut app, &ctx, key(key_code, modifiers));
                }
            }
            assert!(app.pending_import_queue.is_empty());
            assert!(app.pending_oversized_import.is_none());
            assert!(app.pending_paste_request.is_none());
            assert!(app.paste_overlay.is_none());
            assert_eq!(app.projects.len(), 1);
            assert_eq!(app.projects[0].canvas_state.layers.len(), 1);
        }
    }
}

#[test]
fn import_batch_checkbox_keeps_keyboard_focus_and_enter_does_not_import() {
    let mut app = import_dialog_fixture(0);
    app.pending_import_queue
        .push(app.pending_import_queue[0].clone());
    let ctx = context();
    import_keyboard_frame(&mut app, &ctx, Default::default());
    for _ in 0..3 {
        import_keyboard_frame(&mut app, &ctx, key(egui::Key::Tab, egui::Modifiers::NONE));
    }
    import_keyboard_frame(&mut app, &ctx, key(egui::Key::Space, egui::Modifiers::NONE));
    assert!(app.import_apply_to_all);
    import_keyboard_frame(&mut app, &ctx, key(egui::Key::Enter, egui::Modifiers::NONE));
    assert_eq!(app.pending_import_queue.len(), 2);
    assert_eq!(app.projects.len(), 1);
}

#[test]
fn unsaved_dialog_tab_and_shift_tab_reach_discard_and_cancel() {
    for (modifiers, discard) in [
        (egui::Modifiers::NONE, true),
        (egui::Modifiers::SHIFT, false),
    ] {
        let mut app = app();
        let ctx = context();
        app.projects.push(Project::new_untitled(2, 32, 24));
        app.projects[0].mark_dirty();
        app.pending_close_index = Some(0);
        for input in [
            egui::RawInput::default(),
            key(egui::Key::Tab, modifiers),
            key(egui::Key::Enter, egui::Modifiers::NONE),
        ] {
            let _ = ctx.run_ui(input, |ui| {
                app.handle_runtime_modal_flow(ui.ctx());
            });
        }
        assert!(app.pending_close_index.is_none());
        assert_eq!(app.projects.len(), if discard { 1 } else { 2 });
        assert!(!app.save_file_dialog.open);
    }
}

#[test]
fn unsaved_dialogs_can_be_dragged_by_their_heading() {
    for exiting in [false, true] {
        let mut app = app();
        let ctx = context();
        app.projects[0].mark_dirty();
        app.pending_exit = exiting;
        app.pending_close_index = (!exiting).then_some(0);
        let id = egui::Id::new(if exiting {
            "Exit PaintFE"
        } else {
            "Unsaved Changes"
        });
        let mut frame = |events| {
            let _ = ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(1200., 900.),
                    )),
                    events,
                    ..Default::default()
                },
                |ui| {
                    app.handle_runtime_modal_flow(ui.ctx());
                },
            );
        };
        for _ in 0..3 {
            frame(vec![]);
        }
        let before = egui::AreaState::load(&ctx, id).unwrap().rect();
        let start = before.min + egui::vec2(100., 26.);
        frame(vec![egui::Event::PointerMoved(start)]);
        frame(vec![egui::Event::PointerButton {
            pos: start,
            button: egui::PointerButton::Primary,
            pressed: true,
            modifiers: egui::Modifiers::NONE,
        }]);
        let end = start + egui::vec2(100., -70.);
        frame(vec![egui::Event::PointerMoved(end)]);
        frame(vec![egui::Event::PointerButton {
            pos: end,
            button: egui::PointerButton::Primary,
            pressed: false,
            modifiers: egui::Modifiers::NONE,
        }]);
        frame(vec![]);
        let after = egui::AreaState::load(&ctx, id).unwrap().rect();
        assert!(
            (after.min - before.min - egui::vec2(100., -70.)).length() < 2.,
            "{exiting}: {before:?} -> {after:?}"
        );
    }
}

#[test]
fn issue_193_text_editing_owns_delete_and_remapped_fill_keys() {
    use crate::config::keybindings::{BindableAction, KeyCombo};
    for (pressed, custom) in [
        (egui::Key::Backspace, false),
        (egui::Key::Delete, false),
        (egui::Key::F9, true),
    ] {
        let mut app = app();
        let ctx = context();
        app.tools_panel.change_tool(Tool::Text);
        app.tools_panel.text_state.is_editing = true;
        app.tools_panel.text_state.text = "Editing".into();
        app.projects[0].canvas_state.selection_all = true;
        let before = app.projects[0].canvas_state.layers[0]
            .pixels
            .to_rgba_image();
        let history = app.projects[0].history.undo_count();
        if custom {
            app.settings.keybindings.bindings.insert(
                BindableAction::FillSelectedPixels,
                KeyCombo {
                    key: Some(pressed),
                    text_char: None,
                    ctrl: false,
                    shift: false,
                    alt: false,
                },
            );
        }
        let _ = ctx.run_ui(key(pressed, egui::Modifiers::NONE), |ui| {
            app.update_runtime_input(ui.ctx());
            assert!(
                ui.input(|i| i.key_pressed(pressed)),
                "text lost {pressed:?}"
            );
        });
        assert_eq!(
            app.projects[0].canvas_state.layers[0]
                .pixels
                .to_rgba_image(),
            before
        );
        assert_eq!(app.projects[0].history.undo_count(), history);
    }
}

#[test]
fn issue_188_navigation_preserves_transform_and_painting_commits_it_once() {
    let mut app = app();
    let ctx = context();
    app.tools_panel.change_tool(Tool::MovePixels);
    app.paste_overlay = Some(PasteOverlay::new(
        RgbaImage::from_pixel(4, 4, image::Rgba([255, 0, 0, 255])),
        32,
        24,
    ));
    for tool in [Tool::MovePixels, Tool::Pan, Tool::MovePixels, Tool::Pan] {
        app.tools_panel.change_tool(tool);
        let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
            app.update_runtime_input(ui.ctx());
        });
        assert!(app.paste_overlay.is_some());
    }
    let count = app.projects[0].history.undo_count();
    app.tools_panel.change_tool(Tool::Brush);
    for _ in 0..2 {
        let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
            app.update_runtime_input(ui.ctx());
        });
    }
    assert!(app.paste_overlay.is_none());
    assert_eq!(app.projects[0].history.undo_count(), count + 1);
}

#[test]
fn issue_189_repeated_snapshot_restore_reuploads_actual_cropped_pixels() {
    use crate::components::history::{CanvasSnapshot, SnapshotCommand};
    let mut app = app();
    let ctx = context();
    let state = &mut app.projects[0].canvas_state;
    let before = CanvasSnapshot::capture(state);
    crate::ops::transform::resize_canvas(state, 32, 12, (0, 0), image::Rgba([0, 0, 0, 0]));
    let expected = state.layers[0].pixels.to_rgba_image();
    let after = CanvasSnapshot::capture(state);
    app.projects[0]
        .history
        .push(Box::new(SnapshotCommand::from_snapshots(
            HistoryAction::ResizeCanvas.label("Resize Canvas"),
            before,
            after,
        )));
    for _ in 0..3 {
        {
            let project = &mut app.projects[0];
            project.history.undo(&mut project.canvas_state);
        }
        let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
            app.canvas.show(ui, &mut app.projects[0].canvas_state);
        });
        {
            let project = &mut app.projects[0];
            project.history.redo(&mut project.canvas_state);
        }
        assert!(app.projects[0].canvas_state.gpu_layers_invalid);
        let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
            app.canvas.show(ui, &mut app.projects[0].canvas_state);
        });
        let gpu = app.canvas.gpu_renderer_for_sync_readback().unwrap();
        let rendered = gpu
            .composite(32, 12, &[(0, 1.0, true, 0)])
            .expect("GPU render");
        assert_eq!(rendered, expected.as_raw().as_slice());
    }
}

#[test]
fn issue_191_selection_dimensions_include_single_pixels_and_survive_release() {
    let mut state = CanvasState::new(32, 24);
    assert_eq!(state.selection_dimensions(), None);
    let mut mask = image::GrayImage::new(32, 24);
    mask.put_pixel(7, 9, image::Luma([255]));
    state.selection_mask = Some(mask);
    assert_eq!(state.selection_dimensions(), Some((1, 1)));
    state
        .selection_mask
        .as_mut()
        .unwrap()
        .put_pixel(12, 14, image::Luma([128]));
    assert_eq!(state.selection_dimensions(), Some((6, 6)));
    state.selection_all = true;
    assert_eq!(state.selection_dimensions(), Some((32, 24)));
}

#[test]
fn issue_193_text_paste_typing_and_backspace_reach_the_editor() {
    let mut app = app();
    let ctx = context();
    app.tools_panel.change_tool(Tool::Text);
    tool_frame(&mut app, &ctx, (8.0, 8.0), Vec::new());
    app.tools_panel.text_state.is_editing = true;
    app.tools_panel.text_state.origin = Some([4.0, 4.0]);
    app.projects[0].canvas_state.selection_all = true;
    let before = app.projects[0].canvas_state.layers[0]
        .pixels
        .to_rgba_image();
    tool_frame(
        &mut app,
        &ctx,
        (8.0, 8.0),
        vec![egui::Event::Paste("HELLO".into())],
    );
    assert_eq!(app.tools_panel.text_state.text, "HELLO");
    tool_frame(
        &mut app,
        &ctx,
        (8.0, 8.0),
        vec![egui::Event::Text("!".into())],
    );
    assert_eq!(app.tools_panel.text_state.text, "HELLO!");
    tool_frame(
        &mut app,
        &ctx,
        (8.0, 8.0),
        key(egui::Key::Backspace, egui::Modifiers::NONE).events,
    );
    assert_eq!(app.tools_panel.text_state.text, "HELLO");
    assert_eq!(
        app.projects[0].canvas_state.layers[0]
            .pixels
            .to_rgba_image(),
        before
    );
}
