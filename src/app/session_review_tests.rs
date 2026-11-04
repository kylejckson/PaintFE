use super::history_tests::{app, context, tool_frame};
use super::*;
use crate::components::tools::Tool;

fn committed_edit(app: &mut PaintFEApp) {
    let project = &mut app.projects[0];
    let mut command = SingleLayerSnapshotCommand::new_for_layer(
        HistoryAction::Brush.label("Brush"),
        &project.canvas_state,
        0,
    );
    project.canvas_state.layers[0]
        .pixels
        .put_pixel(2, 2, image::Rgba([200, 10, 20, 255]));
    command.set_after(&project.canvas_state);
    project.history.push(Box::new(command));
}

#[test]
fn ready_mesh_does_not_intercept_committed_undo_after_reinitialisation() {
    let mut app = app();
    let ctx = context();
    let before = app.projects[0].canvas_state.layers[0]
        .pixels
        .to_rgba_image();
    committed_edit(&mut app);
    app.tools_panel.active_tool = Tool::MeshWarp;
    tool_frame(&mut app, &ctx, (10., 10.), vec![]);
    assert!(!app.tools_panel.has_active_tool_preview());
    app.perform_undo();
    assert_eq!(
        before,
        app.projects[0].canvas_state.layers[0]
            .pixels
            .to_rgba_image()
    );
    tool_frame(&mut app, &ctx, (10., 10.), vec![]);
    app.perform_redo();
    assert_eq!(
        app.projects[0].canvas_state.layers[0]
            .pixels
            .get_pixel(2, 2)
            .0,
        [200, 10, 20, 255]
    );
}

#[test]
fn dirty_mesh_is_discarded_and_committed_edit_undone_in_one_action() {
    let mut app = app();
    let ctx = context();
    let before = app.projects[0].canvas_state.layers[0]
        .pixels
        .to_rgba_image();
    committed_edit(&mut app);
    app.tools_panel.active_tool = Tool::MeshWarp;
    tool_frame(&mut app, &ctx, (10., 10.), vec![]);
    app.tools_panel.mesh_warp_state.points[6][0] += 2.;
    assert!(app.tools_panel.has_active_tool_preview());
    app.perform_undo();
    assert_eq!(
        before,
        app.projects[0].canvas_state.layers[0]
            .pixels
            .to_rgba_image()
    );
}

#[test]
fn closed_resource_dialog_releases_the_app_modal_state() {
    for shape in [false, true] {
        let mut app = app();
        let ctx = context();
        app.active_dialog = if shape {
            ActiveDialog::AddShape(crate::ui::dialogs::core::AddShapeDialog::new(&[]))
        } else {
            ActiveDialog::AddBrushTip(crate::ui::dialogs::core::AddBrushTipDialog::new(&[]))
        };
        let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
            app.process_active_dialog(ui.ctx())
        });
        assert!(matches!(app.active_dialog, ActiveDialog::None));
    }
}

#[test]
fn four_pixel_hard_brush_has_a_stable_four_pixel_footprint() {
    let mut outputs = Vec::new();
    for position in [(8.1, 8.1), (8.9, 8.9)] {
        let mut tools = crate::components::tools::ToolsPanel::default();
        tools.active_tool = Tool::Brush;
        tools.properties = Default::default();
        tools.properties.size = 4.;
        tools.properties.hardness = 1.;
        tools.properties.anti_aliased = false;
        tools.rebuild_brush_lut();

        let mut pixels = TiledImage::from_rgba_image(&image::RgbaImage::new(16, 16));
        tools.draw_circle_no_dirty(
            &mut pixels,
            16,
            16,
            position,
            false,
            false,
            [1., 0., 0., 1.],
            [0., 0., 0., 1.],
            None,
        );
        let image = pixels.to_rgba_image();
        let painted: Vec<_> = image
            .enumerate_pixels()
            .filter(|(_, _, p)| p.0[3] > 0)
            .map(|(x, y, _)| (x, y))
            .collect();
        assert_eq!(painted.len(), 12);
        assert_eq!(painted.iter().map(|p| p.0).min(), Some(6));
        assert_eq!(painted.iter().map(|p| p.0).max(), Some(9));
        outputs.push(image);
    }
    assert_eq!(outputs[0], outputs[1]);
}

#[test]
fn optional_adjustment_undo_does_not_consume_committed_history() {
    let mut app = app();
    let ctx = context();
    committed_edit(&mut app);
    app.settings.undo_tool_adjustments = true;
    app.tools_panel.active_tool = Tool::MeshWarp;
    tool_frame(&mut app, &ctx, (10., 10.), vec![]);
    let mesh = &mut app.tools_panel.mesh_warp_state;
    let original = mesh.layout();
    mesh.drag_before = Some(original.clone());
    mesh.points[6][0] += 2.;
    mesh.finish_drag();
    let modified = app.tools_panel.mesh_warp_state.layout();
    app.perform_undo();
    assert_eq!(app.tools_panel.mesh_warp_state.layout(), original);
    assert_eq!(app.projects[0].history.undo_count(), 1);
    app.perform_redo();
    assert_eq!(app.tools_panel.mesh_warp_state.layout(), modified);
}

#[test]
fn switching_documents_cancels_deformation_and_its_pending_commit() {
    let mut app = app();
    let ctx = context();
    app.restore_active_project_view();
    app.tools_panel.active_tool = Tool::MeshWarp;
    tool_frame(&mut app, &ctx, (10., 10.), vec![]);
    app.tools_panel.mesh_warp_state.points[6][0] += 2.;
    app.tools_panel.mesh_warp_state.commit_pending = true;
    app.projects.push(Project::new_untitled(2, 32, 24));
    app.active_project_index = 1;
    app.restore_active_project_view();
    assert!(!app.tools_panel.mesh_warp_state.is_active);
    assert!(!app.tools_panel.mesh_warp_state.commit_pending);
    assert!(app.projects[1].canvas_state.preview_layer.is_none());
    assert_eq!(app.projects[1].history.undo_count(), 0);
}

#[test]
fn shortcut_undo_and_redo_reach_committed_history_in_one_press() {
    let mut app = app();
    let ctx = context();
    let before = app.projects[0].canvas_state.layers[0]
        .pixels
        .to_rgba_image();
    committed_edit(&mut app);
    app.tools_panel.active_tool = Tool::MeshWarp;
    tool_frame(&mut app, &ctx, (10., 10.), vec![]);
    app.tools_panel.mesh_warp_state.points[6][0] += 2.;
    for (key, expected) in [
        (egui::Key::Z, before),
        (egui::Key::Y, {
            let mut image = app.projects[0].canvas_state.layers[0]
                .pixels
                .to_rgba_image();
            image.put_pixel(2, 2, image::Rgba([200, 10, 20, 255]));
            image
        }),
    ] {
        let modifiers = egui::Modifiers::CTRL;
        let _ = ctx.run_ui(
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
            },
            |ui| {
                app.update_runtime_input(ui.ctx());
            },
        );
        assert_eq!(
            app.projects[0].canvas_state.layers[0]
                .pixels
                .to_rgba_image(),
            expected
        );
        let _ = ctx.run_ui(
            egui::RawInput {
                events: vec![egui::Event::Key {
                    key,
                    physical_key: None,
                    pressed: false,
                    repeat: false,
                    modifiers,
                }],
                ..Default::default()
            },
            |_| {},
        );
    }
}

#[test]
fn tool_bar_height_is_stable_across_every_tool() {
    let mut app = app();
    let ctx = context();
    for dark in [false, true] {
        let theme = if dark {
            crate::theme::Theme::dark()
        } else {
            crate::theme::Theme::light()
        };
        ctx.set_visuals(if dark {
            egui::Visuals::dark()
        } else {
            egui::Visuals::light()
        });
        let mut heights = Vec::new();
        for &tool in Tool::ALL {
            app.tools_panel.active_tool = tool;
            let _ = ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(2400., 900.),
                    )),
                    ..Default::default()
                },
                |ui| {
                    let response = ui.horizontal(|ui| {
                        app.tools_panel.show_context_bar(
                            ui,
                            &app.assets,
                            egui::Color32::BLACK,
                            egui::Color32::WHITE,
                            &theme,
                        );
                    });
                    heights.push((tool, response.response.rect.height()));
                },
            );
        }
        let min = heights
            .iter()
            .map(|(_, h)| *h)
            .fold(f32::INFINITY, f32::min);
        let max = heights.iter().map(|(_, h)| *h).fold(0., f32::max);
        assert!(max - min <= 1., "bar heights: {heights:?}");
    }
}

fn brush_pointer(point: (f32, f32), down: bool) -> egui::Event {
    egui::Event::PointerButton {
        pos: egui::pos2(100. + point.0 * 4., 100. + point.1 * 4.),
        button: egui::PointerButton::Primary,
        pressed: down,
        modifiers: egui::Modifiers::NONE,
    }
}

#[test]
fn fast_mesh_drags_capture_the_press_anchor_and_keep_the_release_position() {
    for edit_reference in [false, true] {
        for release_in_same_frame in [false, true] {
            let mut app = app();
            let ctx = context();
            app.tools_panel.active_tool = Tool::MeshWarp;
            app.tools_panel.mesh_warp_state.edit_reference = edit_reference;
            tool_frame(&mut app, &ctx, (16., 12.), vec![]);
            let before = app.projects[0].canvas_state.layers[0]
                .pixels
                .to_rgba_image();
            let target = (19., 14.);
            let mut events = vec![
                brush_pointer((16., 12.), true),
                egui::Event::PointerMoved(egui::pos2(176., 156.)),
            ];
            if release_in_same_frame {
                events.push(brush_pointer(target, false));
            }
            tool_frame(&mut app, &ctx, target, events);
            if !release_in_same_frame {
                tool_frame(&mut app, &ctx, target, vec![brush_pointer(target, false)]);
            }
            let state = &app.tools_panel.mesh_warp_state;
            assert_eq!(state.points[12], [target.0, target.1]);
            assert_eq!(state.dragging_index, None);
            assert_eq!(state.undo_layouts.len(), 1);
            if edit_reference {
                assert_eq!(state.original_points[12], [target.0, target.1]);
            } else {
                assert_eq!(state.original_points[12], [16., 12.]);
            }
            assert_eq!(
                app.projects[0].canvas_state.layers[0]
                    .pixels
                    .to_rgba_image(),
                before
            );
        }
    }
}

#[test]
fn fast_liquify_stroke_changes_the_preview_and_finishes_local_history() {
    let mut app = app();
    let ctx = context();
    app.tools_panel.active_tool = Tool::Liquify;
    app.tools_panel.record_tool_adjustments = true;
    app.tools_panel.liquify_state.diameter = 24.;
    let before = app.projects[0].canvas_state.layers[0]
        .pixels
        .to_rgba_image();
    tool_frame(
        &mut app,
        &ctx,
        (20., 12.),
        vec![
            brush_pointer((10., 12.), true),
            egui::Event::PointerMoved(egui::pos2(180., 148.)),
            brush_pointer((20., 12.), false),
        ],
    );
    let state = &app.tools_panel.liquify_state;
    assert!(
        state
            .displacement
            .as_ref()
            .unwrap()
            .data
            .iter()
            .any(|value| *value != 0.)
    );
    assert!(state.history.can_undo());
    assert!(state.last_pos.is_none());
    assert_ne!(
        app.projects[0]
            .canvas_state
            .preview_layer
            .as_ref()
            .unwrap()
            .to_rgba_image(),
        before
    );
    assert_eq!(
        app.projects[0].canvas_state.layers[0]
            .pixels
            .to_rgba_image(),
        before
    );
}

#[test]
fn enter_applies_mesh_and_liquify_without_invalidating_the_source_and_undo_round_trips() {
    for tool in [Tool::MeshWarp, Tool::Liquify] {
        let mut app = app();
        let ctx = context();
        app.tools_panel.active_tool = tool;
        let before = app.projects[0].canvas_state.layers[0]
            .pixels
            .to_rgba_image();
        if tool == Tool::MeshWarp {
            tool_frame(&mut app, &ctx, (16., 12.), vec![]);
            app.tools_panel.mesh_warp_state.points[12][0] += 2.;
        } else {
            app.tools_panel.liquify_state.mode = crate::components::tools::LiquifyMode::Expand;
            app.tools_panel.liquify_state.diameter = 24.;
            tool_frame(
                &mut app,
                &ctx,
                (16., 12.),
                vec![brush_pointer((16., 12.), true)],
            );
            tool_frame(
                &mut app,
                &ctx,
                (16., 12.),
                vec![brush_pointer((16., 12.), false)],
            );
        }
        let generation = app.projects[0].canvas_state.layers[0].gpu_generation;
        let _ = ctx.run_ui(
            egui::RawInput {
                events: vec![egui::Event::Key {
                    key: egui::Key::Enter,
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
        assert_eq!(
            app.projects[0].canvas_state.layers[0].gpu_generation, generation,
            "{tool:?} request changed its own source generation"
        );
        for _ in 0..4 {
            tool_frame(&mut app, &ctx, (16., 12.), vec![]);
        }
        let after = app.projects[0].canvas_state.layers[0]
            .pixels
            .to_rgba_image();
        assert_ne!(after, before, "{tool:?} Enter discarded the edit");
        assert_eq!(app.projects[0].history.undo_count(), 1, "{tool:?}");
        app.perform_undo();
        assert_eq!(
            app.projects[0].canvas_state.layers[0]
                .pixels
                .to_rgba_image(),
            before,
            "{tool:?} Undo"
        );
        app.perform_redo();
        assert_eq!(
            app.projects[0].canvas_state.layers[0]
                .pixels
                .to_rgba_image(),
            after,
            "{tool:?} Redo"
        );
    }
}

fn blocked_deformation_frame(app: &mut PaintFEApp, ctx: &egui::Context) {
    let _ = ctx.run_ui(
        egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(900., 700.),
            )),
            events: vec![egui::Event::PointerMoved(egui::pos2(10., 10.))],
            ..Default::default()
        },
        |ui| {
            app.canvas.show_with_state(
                ui,
                &mut app.projects[0].canvas_state,
                Some(&mut app.tools_panel),
                [0., 0., 0., 1.],
                [1., 1., 1., 1.],
                egui::Color32::GRAY,
                None,
                None,
                false,
                &app.settings,
                0,
                0,
                egui::Color32::BLUE,
                None,
                None,
                "",
                true,
                true,
                false,
            );
        },
    );
}

#[test]
fn mesh_reset_and_tool_switch_finish_while_the_pointer_is_over_ui() {
    let mut app = app();
    let ctx = context();
    app.tools_panel.active_tool = Tool::MeshWarp;
    tool_frame(&mut app, &ctx, (16., 12.), vec![]);
    let before = app.projects[0].canvas_state.layers[0]
        .pixels
        .to_rgba_image();
    app.tools_panel.mesh_warp_state.points[12][0] += 2.;
    app.tools_panel.mesh_warp_state.preview_layout = Some(app.tools_panel.mesh_warp_state.layout());
    app.projects[0].canvas_state.preview_layer = Some(TiledImage::from_rgba_image(
        &image::RgbaImage::from_pixel(32, 24, image::Rgba([0, 255, 0, 255])),
    ));
    app.projects[0].canvas_state.preview_replaces_layer = true;
    app.tools_panel.reset_mesh_deformation();
    assert!(
        app.tools_panel
            .needs_deformation_update(&app.projects[0].canvas_state)
    );
    blocked_deformation_frame(&mut app, &ctx);
    assert!(app.projects[0].canvas_state.preview_layer.is_none());
    app.tools_panel.active_tool = Tool::Liquify;
    for _ in 0..4 {
        blocked_deformation_frame(&mut app, &ctx);
    }
    assert!(!app.tools_panel.mesh_warp_state.is_active);
    assert!(!app.tools_panel.mesh_warp_state.commit_pending);
    assert_eq!(
        app.projects[0].canvas_state.layers[0]
            .pixels
            .to_rgba_image(),
        before
    );
    let rendered = app
        .canvas
        .gpu_renderer_for_sync_readback()
        .unwrap()
        .composite(32, 24, &[(0, 1., true, 0)])
        .unwrap();
    assert_eq!(rendered, before.as_raw().as_slice());
}

#[test]
fn liquify_protection_actions_finish_while_the_pointer_is_over_ui() {
    let mut app = app();
    let ctx = context();
    app.tools_panel.active_tool = Tool::Liquify;
    app.projects[0].canvas_state.selection_mask =
        Some(image::GrayImage::from_pixel(32, 24, image::Luma([255])));
    app.tools_panel.liquify_state.protect_selection_pending = true;
    blocked_deformation_frame(&mut app, &ctx);
    assert!(!app.tools_panel.liquify_state.protect_selection_pending);
    assert_eq!(
        app.tools_panel
            .liquify_state
            .protection
            .as_ref()
            .unwrap()
            .get_pixel(10, 10)
            .0[0],
        255
    );
    app.tools_panel.liquify_state.clear_protection_pending = true;
    blocked_deformation_frame(&mut app, &ctx);
    assert!(app.tools_panel.liquify_state.protection.is_none());
}

#[test]
fn holding_and_releasing_a_stationary_brush_does_not_add_a_second_deposit() {
    for mode in [
        crate::components::tools::BrushMode::Normal,
        crate::components::tools::BrushMode::Uniform,
        crate::components::tools::BrushMode::BuildUp,
    ] {
        let mut app = app();
        let ctx = context();
        app.tools_panel.active_tool = Tool::Brush;
        app.tools_panel.properties.size = 8.;
        app.tools_panel.properties.opacity = 0.5;
        app.tools_panel.properties.brush_mode = mode;
        app.tools_panel.properties.pressure_size = false;
        app.tools_panel.properties.pressure_opacity = false;
        let pos = (10., 10.);
        tool_frame(&mut app, &ctx, pos, vec![brush_pointer(pos, true)]);
        let rect = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(32., 24.));
        let preview = app.projects[0].canvas_state.composite_partial(rect).0;
        tool_frame(&mut app, &ctx, pos, vec![]);
        assert_eq!(
            app.projects[0].canvas_state.composite_partial(rect).0,
            preview,
            "{mode:?} hold"
        );
        tool_frame(&mut app, &ctx, pos, vec![brush_pointer(pos, false)]);
        let committed = app.projects[0]
            .canvas_state
            .composite_committed_downscaled(rect, 1)
            .0;
        assert_eq!(committed, preview, "{mode:?} release");
        assert_eq!(app.projects[0].history.undo_count(), 1);
    }
}

#[test]
fn liquify_protection_preserves_pixels_while_the_surrounding_image_changes() {
    let mut app = app();
    let ctx = context();
    app.tools_panel.active_tool = Tool::Liquify;
    app.tools_panel.liquify_state.mode = crate::components::tools::LiquifyMode::Expand;
    app.tools_panel.liquify_state.diameter = 24.;
    let before = app.projects[0].canvas_state.layers[0]
        .pixels
        .to_rgba_image();
    app.projects[0].canvas_state.selection_mask =
        Some(image::GrayImage::from_fn(32, 24, |x, y| {
            image::Luma([if x >= 12 && y >= 10 { 255 } else { 0 }])
        }));
    app.tools_panel.liquify_state.protect_selection_pending = true;
    tool_frame(&mut app, &ctx, (16., 12.), vec![]);
    app.projects[0].canvas_state.selection_mask = None;
    app.projects[0].canvas_state.invalidate_selection_overlay();
    tool_frame(
        &mut app,
        &ctx,
        (16., 12.),
        vec![brush_pointer((16., 12.), true)],
    );
    tool_frame(
        &mut app,
        &ctx,
        (16., 12.),
        vec![brush_pointer((16., 12.), false)],
    );
    assert!(app.tools_panel.liquify_state.is_active);
    assert!(
        app.tools_panel
            .liquify_state
            .displacement
            .as_ref()
            .unwrap()
            .data
            .iter()
            .any(|v| *v != 0.)
    );
    app.tools_panel
        .commit_active_tool_preview(&mut app.projects[0].canvas_state, [1.; 4]);
    app.commit_pending_tool_history();
    let after = app.projects[0].canvas_state.layers[0]
        .pixels
        .to_rgba_image();
    assert_ne!(after, before);
    for y in 10..24 {
        for x in 12..32 {
            assert_eq!(after.get_pixel(x, y), before.get_pixel(x, y));
        }
    }
    assert_eq!(app.projects[0].history.undo_count(), 1);
}

#[test]
fn mesh_cancel_button_does_not_cancel_every_later_frame() {
    let mut app = app();
    let ctx = context();
    app.tools_panel.active_tool = Tool::MeshWarp;
    tool_frame(&mut app, &ctx, (10., 10.), vec![]);
    app.tools_panel.mesh_warp_state.cancel_pending = true;
    tool_frame(&mut app, &ctx, (10., 10.), vec![]);
    assert!(!app.tools_panel.mesh_warp_state.cancel_pending);
    app.tools_panel.mesh_warp_state.points[6][0] += 2.;
    tool_frame(&mut app, &ctx, (10., 10.), vec![]);
    assert!(app.tools_panel.has_active_tool_preview());
}

#[test]
fn liquify_protection_is_recreated_after_canvas_dimensions_change() {
    let mut app = app();
    let ctx = context();
    app.tools_panel.active_tool = Tool::Liquify;
    app.tools_panel.liquify_state.protection =
        Some(image::GrayImage::from_pixel(8, 8, image::Luma([255])));
    app.projects[0].canvas_state.selection_mask =
        Some(image::GrayImage::from_pixel(32, 24, image::Luma([255])));
    app.tools_panel.liquify_state.protect_selection_pending = true;
    tool_frame(&mut app, &ctx, (10., 10.), vec![]);
    assert_eq!(
        app.tools_panel
            .liquify_state
            .protection
            .as_ref()
            .unwrap()
            .dimensions(),
        (32, 24)
    );
}

#[test]
fn default_paste_undo_discards_unapplied_transforms_and_undoes_one_committed_edit() {
    let mut app = app();
    app.settings.undo_tool_adjustments = false;
    let ctx = context();
    let before = app.projects[0].canvas_state.layers[0]
        .pixels
        .to_rgba_image();
    committed_edit(&mut app);
    let mut overlay = PasteOverlay::new(
        RgbaImage::from_pixel(4, 4, image::Rgba([255, 0, 0, 255])),
        32,
        24,
    );
    let initial = overlay.transform();
    overlay.center.x += 2.;
    overlay.pending_transform_checkpoint = Some((initial, overlay.transform()));
    app.paste_overlay = Some(overlay);
    let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
        app.show_runtime_canvas_tail(&ctx, ui, false)
    });
    assert_eq!(app.projects[0].history.undo_count(), 1);
    app.perform_undo();
    assert!(app.paste_overlay.is_none());
    assert_eq!(
        app.projects[0].canvas_state.layers[0]
            .pixels
            .to_rgba_image(),
        before
    );
}

#[test]
fn local_adjustment_undo_and_redo_cancel_a_deferred_apply_request() {
    let mut app = app();
    let ctx = context();
    app.settings.undo_tool_adjustments = true;
    app.tools_panel.active_tool = Tool::MeshWarp;
    tool_frame(&mut app, &ctx, (10., 10.), vec![]);
    let mesh = &mut app.tools_panel.mesh_warp_state;
    mesh.drag_before = Some(mesh.layout());
    mesh.points[6][0] += 2.;
    mesh.finish_drag();
    mesh.commit_pending = true;
    app.perform_undo();
    assert!(!app.tools_panel.mesh_warp_state.commit_pending);
    app.tools_panel.mesh_warp_state.commit_pending = true;
    app.perform_redo();
    assert!(!app.tools_panel.mesh_warp_state.commit_pending);
    assert_eq!(app.projects[0].history.undo_count(), 0);
    assert!(app.tools_panel.has_active_tool_preview());
}

#[test]
fn liquify_reset_clears_deformation_retains_protection_and_is_locally_undoable() {
    let mut app = app();
    let ctx = context();
    app.settings.undo_tool_adjustments = true;
    app.tools_panel.record_tool_adjustments = true;
    app.tools_panel.active_tool = Tool::Liquify;
    app.tools_panel.liquify_state.mode = crate::components::tools::LiquifyMode::Expand;
    app.tools_panel.liquify_state.diameter = 24.;
    let point = (16., 12.);
    tool_frame(&mut app, &ctx, point, vec![brush_pointer(point, true)]);
    tool_frame(&mut app, &ctx, point, vec![brush_pointer(point, false)]);
    let before = app
        .tools_panel
        .liquify_state
        .displacement
        .as_ref()
        .unwrap()
        .data
        .clone();
    assert!(before.iter().any(|v| *v != 0.));
    app.tools_panel.liquify_state.protection = Some(image::GrayImage::new(32, 24));
    app.tools_panel.liquify_state.reset_pending = true;
    tool_frame(&mut app, &ctx, point, vec![]);
    assert!(
        app.tools_panel
            .liquify_state
            .displacement
            .as_ref()
            .unwrap()
            .data
            .iter()
            .all(|v| *v == 0.)
    );
    assert!(app.tools_panel.liquify_state.protection.is_some());
    app.perform_undo();
    assert_eq!(
        app.tools_panel
            .liquify_state
            .displacement
            .as_ref()
            .unwrap()
            .data,
        before
    );
    assert_eq!(app.projects[0].history.undo_count(), 0);
}
