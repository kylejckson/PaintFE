use super::*;
use crate::components::dialogs::SaveAction;

fn app() -> PaintFEApp {
    crate::i18n::init();
    let cc = eframe::CreationContext::_new_kittest(egui::Context::default());
    let mut app = PaintFEApp::new(&cc, Vec::new(), mpsc::channel().1);
    app.projects = vec![Project::new_untitled(1, 4, 4)];
    app.active_project_index = 0;
    app
}

fn action(path: PathBuf, format: SaveFormat) -> SaveAction {
    SaveAction {
        path,
        format,
        quality: 90,
        webp_lossless: true,
        tiff_compression: TiffCompression::None,
        animated: false,
        animation_fps: 10.0,
        gif_colors: 256,
        gif_dither: true,
    }
}

fn layered(app: &mut PaintFEApp) {
    let layer = Layer::new("Second".into(), 4, 4, image::Rgba([200, 50, 20, 255]));
    app.projects[0].canvas_state.layers.push(layer);
    app.projects[0].mark_dirty();
}

fn directory() -> PathBuf {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target/save-safety-tests")
        .join(uuid::Uuid::new_v4().to_string());
    std::fs::create_dir_all(&path).unwrap();
    path
}

fn wait(app: &mut PaintFEApp, ctx: &egui::Context) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    while !app.save_jobs.is_empty() {
        assert!(std::time::Instant::now() < deadline, "Save timed out");
        app.update_runtime_lifecycle_async(ctx);
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
}

#[test]
fn quick_save_warns_before_touching_original_and_cancel_preserves_it() {
    let mut app = app();
    layered(&mut app);
    let path = directory().join("source.png");
    std::fs::write(&path, b"original").unwrap();
    app.projects[0].file_handler.current_path = Some(path.clone());
    app.handle_save(0.0);
    assert!(app.save_warning.is_some());
    assert!(app.save_jobs.is_empty());
    assert_eq!(std::fs::read(&path).unwrap(), b"original");
    app.resolve_save_warning(2, 0.0);
    assert!(app.projects[0].is_dirty);
    assert!(app.save_warning.is_none());
    assert_eq!(std::fs::read(path).unwrap(), b"original");
}

#[test]
fn save_as_uses_the_same_loss_check() {
    let mut app = app();
    layered(&mut app);
    let path = directory().join("export.png");
    let request = PendingSave {
        project_id: app.projects[0].id,
        action: action(path.clone(), SaveFormat::Png),
        update_path: true,
    };
    app.request_save(request, 0.0);
    assert!(app.save_warning.is_some());
    assert!(!path.exists());
}

#[test]
fn project_is_recommended_for_layered_save_as_and_warning_choice() {
    let mut app = app();
    layered(&mut app);
    app.open_save_as_for_project(0);
    assert_eq!(app.save_file_dialog.format(), SaveFormat::Pfe);
    app.save_file_dialog.open = false;
    app.request_save(
        PendingSave {
            project_id: app.projects[0].id,
            action: action(directory().join("image.png"), SaveFormat::Png),
            update_path: true,
        },
        0.0,
    );
    app.resolve_save_warning(0, 0.0);
    assert!(app.save_file_dialog.open);
    assert_eq!(app.save_file_dialog.format(), SaveFormat::Pfe);
    assert!(app.save_jobs.is_empty());
}

#[test]
fn pfe_save_keeps_layers_and_clears_dirty_only_after_completion() {
    let mut app = app();
    layered(&mut app);
    app.projects[0].canvas_state.layers[1].ensure_mask();
    app.projects[0].canvas_state.layers[1]
        .mask
        .as_mut()
        .unwrap()
        .put_pixel(0, 0, image::Rgba([0, 0, 0, 128]));
    let mut text = Layer::new_text("Editable text".into(), 4, 4);
    if let crate::canvas::LayerContent::Text(data) = &mut text.content {
        data.blocks[0].runs[0].text = "Still editable".into();
        data.mark_dirty();
    }
    app.projects[0].canvas_state.layers.push(text);
    let ctx = egui::Context::default();
    let path = directory().join("layers.pfe");
    app.request_save(
        PendingSave {
            project_id: app.projects[0].id,
            action: action(path.clone(), SaveFormat::Pfe),
            update_path: true,
        },
        0.0,
    );
    assert!(app.projects[0].is_dirty);
    assert!(app.save_warning.is_none());
    wait(&mut app, &ctx);
    assert!(!app.projects[0].is_dirty);
    let reopened = crate::io::load_pfe(&path).unwrap();
    assert_eq!(reopened.layers.len(), 3);
    assert_eq!(
        reopened.layers[1].mask.as_ref().unwrap().get_pixel(0, 0)[3],
        128
    );
    let crate::canvas::LayerContent::Text(text) = &reopened.layers[2].content else {
        panic!("Text was flattened");
    };
    assert_eq!(text.flat_text(), "Still editable");
    assert_eq!(
        app.projects[0].file_handler.current_path.as_ref(),
        Some(&path)
    );
}

#[test]
fn flattening_writes_image_but_keeps_editable_document_dirty_and_intact() {
    let mut app = app();
    layered(&mut app);
    let ctx = egui::Context::default();
    let path = directory().join("flat.png");
    app.request_save(
        PendingSave {
            project_id: app.projects[0].id,
            action: action(path.clone(), SaveFormat::Png),
            update_path: true,
        },
        0.0,
    );
    app.resolve_save_warning(1, 0.0);
    wait(&mut app, &ctx);
    assert!(image::open(path).is_ok());
    assert_eq!(app.projects[0].canvas_state.layers.len(), 2);
    assert!(app.projects[0].is_dirty);
}

#[test]
fn image_export_does_not_replace_existing_pfe_save_target() {
    let mut app = app();
    layered(&mut app);
    let ctx = egui::Context::default();
    let directory = directory();
    let project = directory.join("project.pfe");
    app.projects[0].file_handler.current_path = Some(project.clone());
    app.projects[0].file_handler.last_format = SaveFormat::Pfe;
    app.projects[0].path = Some(project.clone());
    app.request_save(
        PendingSave {
            project_id: app.projects[0].id,
            action: action(directory.join("flat.png"), SaveFormat::Png),
            update_path: true,
        },
        0.0,
    );
    app.resolve_save_warning(1, 0.0);
    wait(&mut app, &ctx);
    assert_eq!(
        app.projects[0].file_handler.current_path.as_ref(),
        Some(&project)
    );
    assert_eq!(app.projects[0].file_handler.last_format, SaveFormat::Pfe);
    assert!(app.projects[0].is_dirty);
}

#[test]
fn newer_edits_stay_dirty_after_successful_background_save() {
    let mut app = app();
    app.projects[0].mark_dirty();
    let ctx = egui::Context::default();
    app.request_save(
        PendingSave {
            project_id: app.projects[0].id,
            action: action(directory().join("saved.pfe"), SaveFormat::Pfe),
            update_path: true,
        },
        0.0,
    );
    app.projects[0].mark_dirty();
    wait(&mut app, &ctx);
    assert!(app.projects[0].is_dirty);
}

#[test]
fn background_save_tracks_document_identity_across_tab_reordering() {
    let mut app = app();
    app.projects[0].mark_dirty();
    let id = app.projects[0].id;
    let path = directory().join("correct.pfe");
    let ctx = egui::Context::default();
    app.request_save(
        PendingSave {
            project_id: id,
            action: action(path.clone(), SaveFormat::Pfe),
            update_path: true,
        },
        0.0,
    );
    let mut other = Project::new_untitled(2, 4, 4);
    other.mark_dirty();
    app.projects.insert(0, other);
    wait(&mut app, &ctx);
    assert!(app.projects[0].is_dirty);
    assert!(app.projects[0].path.is_none());
    assert_eq!(app.projects[1].id, id);
    assert_eq!(app.projects[1].path.as_ref(), Some(&path));
    assert!(!app.projects[1].is_dirty);
}

#[test]
fn failed_save_keeps_document_open_dirty_and_exit_cancelled() {
    let mut app = app();
    app.projects[0].mark_dirty();
    let ctx = egui::Context::default();
    app.projects[0].file_handler.current_path = Some(directory().join("missing/target.pfe"));
    app.projects[0].file_handler.last_format = SaveFormat::Pfe;
    app.begin_save_sequence(vec![app.projects[0].id], true, false);
    app.advance_save_sequence(&ctx);
    wait(&mut app, &ctx);
    assert!(app.save_sequence.is_none());
    assert!(!app.force_exit);
    assert!(app.filter_error.is_some());
    assert!(app.projects[0].is_dirty);
}

#[test]
fn close_waits_for_save_then_removes_only_the_saved_document() {
    let mut app = app();
    app.projects[0].mark_dirty();
    let ctx = egui::Context::default();
    app.projects[0].file_handler.current_path = Some(directory().join("close.pfe"));
    app.projects[0].file_handler.last_format = SaveFormat::Pfe;
    app.begin_save_sequence(vec![app.projects[0].id], false, true);
    app.advance_save_sequence(&ctx);
    assert_eq!(app.projects.len(), 1);
    assert!(!app.save_jobs.is_empty());
    wait(&mut app, &ctx);
    assert!(app.projects.is_empty());
}

#[test]
fn exit_waits_for_every_write_and_checks_all_documents() {
    let mut app = app();
    let ctx = egui::Context::default();
    let root = directory();
    app.projects.push(Project::new_untitled(2, 4, 4));
    for (i, p) in app.projects.iter_mut().enumerate() {
        p.mark_dirty();
        p.file_handler.current_path = Some(root.join(format!("{i}.pfe")));
        p.file_handler.last_format = SaveFormat::Pfe;
    }
    app.begin_save_sequence(app.projects.iter().map(|p| p.id).collect(), true, false);
    for _ in 0..2 {
        app.advance_save_sequence(&ctx);
        assert!(!app.force_exit);
        wait(&mut app, &ctx);
    }
    assert!(!app.force_exit);
    app.advance_save_sequence(&ctx);
    assert!(app.force_exit);
    assert!(root.join("0.pfe").exists());
    assert!(root.join("1.pfe").exists());
}

#[test]
fn exit_flattening_requires_confirmation_and_success_before_closing() {
    let mut app = app();
    layered(&mut app);
    let ctx = egui::Context::default();
    app.projects[0].file_handler.current_path = Some(directory().join("flat.png"));
    app.begin_save_sequence(vec![app.projects[0].id], true, false);
    app.advance_save_sequence(&ctx);
    assert!(app.save_warning.is_some());
    assert!(!app.force_exit);
    assert!(app.save_jobs.is_empty());
    app.resolve_save_warning(1, 0.0);
    assert!(!app.force_exit);
    wait(&mut app, &ctx);
    assert!(app.projects[0].is_dirty);
    app.advance_save_sequence(&ctx);
    assert!(app.force_exit);
}

#[test]
fn save_all_cancellation_does_not_save_later_documents() {
    let mut app = app();
    layered(&mut app);
    let ctx = egui::Context::default();
    app.projects[0].file_handler.current_path = Some(directory().join("first.png"));
    let mut second = Project::new_untitled(2, 4, 4);
    second.mark_dirty();
    app.projects.push(second);
    app.handle_save_all(0.0);
    app.advance_save_sequence(&ctx);
    assert!(app.save_warning.is_some());
    app.resolve_save_warning(2, 0.0);
    app.advance_save_sequence(&ctx);
    assert!(app.save_sequence.is_none());
    assert!(!app.save_file_dialog.open);
}

#[test]
fn same_document_cannot_start_two_overlapping_writes() {
    let mut app = app();
    app.projects[0].mark_dirty();
    let ctx = egui::Context::default();
    app.projects[0].file_handler.current_path = Some(directory().join("same.png"));
    app.handle_save(0.0);
    app.handle_save(0.0);
    assert_eq!(app.save_jobs.len(), 1);
    assert_eq!(app.pending_io_ops, 1);
    wait(&mut app, &ctx);
}

#[test]
fn single_layer_masks_and_opacity_are_editable_information_too() {
    let mut app = app();
    let save = action(PathBuf::from("image.png"), SaveFormat::Png);
    assert!(app.projects[0].save_losses(&save).is_empty());
    app.projects[0].canvas_state.layers[0].ensure_mask();
    assert!(!app.projects[0].save_losses(&save).is_empty());
    app.projects[0].canvas_state.layers[0].mask = None;
    app.projects[0].canvas_state.layers[0].opacity = 0.5;
    assert!(!app.projects[0].save_losses(&save).is_empty());
}

#[test]
fn jpeg_alpha_and_animation_losses_are_detected() {
    let mut app = app();
    app.projects[0].canvas_state.layers[0].pixels = TiledImage::new(4, 4);
    app.projects[0].canvas_state.mark_dirty(None);
    assert!(
        !app.projects[0]
            .save_losses(&action(PathBuf::from("alpha.jpg"), SaveFormat::Jpeg))
            .is_empty()
    );
    app.projects[0].was_animated = true;
    assert!(
        !app.projects[0]
            .save_losses(&action(PathBuf::from("still.png"), SaveFormat::Png))
            .is_empty()
    );
}

#[test]
fn atomic_write_failure_preserves_destination_and_removes_temporary_file() {
    let root = directory();
    let destination = root.join("destination.pfe");
    std::fs::create_dir(&destination).unwrap();
    std::fs::write(destination.join("original"), b"keep").unwrap();
    let app = app();
    let result = write_save_atomically(
        &action(destination.clone(), SaveFormat::Pfe),
        SavePayload::Project(crate::io::build_pfe(&app.projects[0].canvas_state)),
    );
    assert!(result.is_err());
    assert_eq!(
        std::fs::read(destination.join("original")).unwrap(),
        b"keep"
    );
    assert_eq!(std::fs::read_dir(root).unwrap().count(), 1);
}

#[test]
fn failed_encoding_does_not_truncate_an_existing_image() {
    let path = directory().join("original.png");
    std::fs::write(&path, b"original bytes").unwrap();
    let result = write_save_atomically(
        &action(path.clone(), SaveFormat::Bmp),
        SavePayload::Frames(vec![], vec![]),
    );
    assert!(result.is_err());
    assert_eq!(std::fs::read(path).unwrap(), b"original bytes");
}

#[test]
fn single_editable_text_layer_requires_project_save() {
    let mut app = app();
    app.projects[0].canvas_state.layers[0] = Layer::new_text("Text".into(), 4, 4);
    assert!(
        !app.projects[0]
            .save_losses(&action(PathBuf::from("text.png"), SaveFormat::Png))
            .is_empty()
    );
    assert!(
        app.projects[0]
            .save_losses(&action(PathBuf::from("text.pfe"), SaveFormat::Pfe))
            .is_empty()
    );
}

#[test]
fn high_depth_warns_for_eight_bit_export_but_not_sixteen_bit_png() {
    let mut app = app();
    let state = &mut app.projects[0].canvas_state;
    state.layers[0].deep_pixels = Some(crate::experimental::DeepRgbaBuffer::U16(vec![
        32000;
        4 * 4 * 4
    ]));
    state.layers[0].pixel_format = crate::canvas::PixelFormat::RgbaU16;
    assert!(
        app.projects[0]
            .save_losses(&action(PathBuf::from("deep.png"), SaveFormat::Png))
            .is_empty()
    );
    assert!(
        !app.projects[0]
            .save_losses(&action(PathBuf::from("deep.webp"), SaveFormat::Webp))
            .is_empty()
    );
}

#[test]
fn warning_escape_cancels_and_enter_defaults_to_project_in_both_themes() {
    for mode in [
        crate::theme::ThemeMode::Light,
        crate::theme::ThemeMode::Dark,
    ] {
        let mut app = app();
        layered(&mut app);
        let ctx = egui::Context::default();
        app.theme.mode = mode;
        app.theme.apply(&ctx);
        app.request_save(
            PendingSave {
                project_id: app.projects[0].id,
                action: action(directory().join("image.png"), SaveFormat::Png),
                update_path: true,
            },
            0.0,
        );
        let frame = |app: &mut PaintFEApp, key: Option<egui::Key>| {
            let events = key
                .map(|key| {
                    vec![egui::Event::Key {
                        key,
                        physical_key: None,
                        pressed: true,
                        repeat: false,
                        modifiers: egui::Modifiers::NONE,
                    }]
                })
                .unwrap_or_default();
            let _ = ctx.run_ui(
                egui::RawInput {
                    events,
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(1000.0, 800.0),
                    )),
                    ..Default::default()
                },
                |_| app.show_save_warning(&ctx),
            );
        };
        frame(&mut app, None);
        frame(&mut app, Some(egui::Key::Escape));
        assert!(app.save_warning.is_none());
        assert!(app.save_jobs.is_empty());
        app.request_save(
            PendingSave {
                project_id: app.projects[0].id,
                action: action(directory().join("image.png"), SaveFormat::Png),
                update_path: true,
            },
            0.0,
        );
        frame(&mut app, None);
        frame(&mut app, Some(egui::Key::Enter));
        assert!(app.save_file_dialog.open);
        assert_eq!(app.save_file_dialog.format(), SaveFormat::Pfe);
        assert!(app.save_jobs.is_empty());
    }
}

#[test]
fn close_and_exit_are_cancelled_if_document_changes_during_the_write() {
    for exit in [false, true] {
        let mut app = app();
        app.projects[0].mark_dirty();
        let ctx = egui::Context::default();
        app.projects[0].file_handler.current_path = Some(directory().join("before-edit.pfe"));
        app.projects[0].file_handler.last_format = SaveFormat::Pfe;
        app.begin_save_sequence(vec![app.projects[0].id], exit, !exit);
        app.advance_save_sequence(&ctx);
        app.projects[0].mark_dirty();
        wait(&mut app, &ctx);
        assert_eq!(app.projects.len(), 1);
        assert!(app.projects[0].is_dirty);
        assert!(!app.force_exit);
        assert!(app.save_sequence.is_none());
        assert!(app.filter_error.is_some());
    }
}

#[test]
fn warning_tab_and_enter_can_explicitly_choose_image_export() {
    let mut app = app();
    layered(&mut app);
    let ctx = egui::Context::default();
    let path = directory().join("keyboard.png");
    app.request_save(
        PendingSave {
            project_id: app.projects[0].id,
            action: action(path.clone(), SaveFormat::Png),
            update_path: true,
        },
        0.0,
    );
    for key in [
        None,
        None,
        Some(egui::Key::Tab),
        None,
        Some(egui::Key::Enter),
    ] {
        let events = key
            .map(|key| {
                vec![egui::Event::Key {
                    key,
                    physical_key: None,
                    pressed: true,
                    repeat: false,
                    modifiers: egui::Modifiers::NONE,
                }]
            })
            .unwrap_or_default();
        let _ = ctx.run_ui(
            egui::RawInput {
                events,
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(1000.0, 800.0),
                )),
                ..Default::default()
            },
            |_| app.show_save_warning(&ctx),
        );
    }
    assert!(app.save_warning.is_none());
    assert!(!app.save_file_dialog.open);
    assert_eq!(app.save_jobs.len(), 1);
    wait(&mut app, &ctx);
    assert!(path.exists());
}

#[test]
fn native_close_waits_for_inflight_write_even_when_exit_confirmation_is_disabled() {
    let mut app = app();
    let ctx = egui::Context::default();
    app.settings.confirm_on_exit = false;
    app.projects[0].mark_dirty();
    let id = app.projects[0].id;
    app.save_jobs.insert(
        id,
        SaveJob {
            request: PendingSave {
                project_id: id,
                action: action(PathBuf::from("pending.png"), SaveFormat::Png),
                update_path: false,
            },
            revision: app.projects[0].save_revision(),
            editable_loss: false,
        },
    );
    let mut input = egui::RawInput::default();
    input
        .viewports
        .get_mut(&egui::ViewportId::ROOT)
        .unwrap()
        .events
        .push(egui::ViewportEvent::Close);
    let output = ctx.run_ui(input, |_| app.update_runtime_lifecycle_async(&ctx));
    assert!(
        output.viewport_output[&egui::ViewportId::ROOT]
            .commands
            .iter()
            .any(|cmd| matches!(cmd, egui::ViewportCommand::CancelClose))
    );
    assert!(!app.force_exit);
    assert!(app.pending_exit);
}
