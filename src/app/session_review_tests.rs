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
                            &app.settings,
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

#[test]
fn docked_shelf_keeps_the_actual_canvas_fixed_through_overflow_and_tool_switches() {
    let mut app = app();
    let ctx = context();
    app.settings.tool_shelf.docked = true;
    for dark in [false, true] {
        for scale in [1., 1.35] {
            app.theme = if dark {
                crate::theme::Theme::dark()
            } else {
                crate::theme::Theme::light()
            };
            app.theme.polish.text_scale = scale;
            app.theme.apply(&ctx);
            for width in [640., 1050., 2400.] {
                app.tools_panel.active_tool = Tool::Brush;
                for _ in 0..2 {
                    let _ = ctx.run_ui(
                        egui::RawInput {
                            screen_rect: Some(egui::Rect::from_min_size(
                                egui::Pos2::ZERO,
                                egui::vec2(width, 900.),
                            )),
                            ..Default::default()
                        },
                        |ui| app.show_runtime_canvas_tail(&ctx, ui, false),
                    );
                }
                let expected = app.canvas.last_canvas_rect.unwrap();
                for tool in [Tool::Fill, Tool::Brush, Tool::Fill, Tool::Brush] {
                    app.tools_panel.active_tool = tool;
                    for _ in 0..2 {
                        let _ = ctx.run_ui(
                            egui::RawInput {
                                screen_rect: Some(egui::Rect::from_min_size(
                                    egui::Pos2::ZERO,
                                    egui::vec2(width, 900.),
                                )),
                                ..Default::default()
                            },
                            |ui| app.show_runtime_canvas_tail(&ctx, ui, false),
                        );
                        assert_eq!(
                            app.canvas.last_canvas_rect,
                            Some(expected),
                            "{tool:?}, width {width}, dark {dark}, scale {scale}"
                        );
                    }
                }
            }
        }
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
fn actual_canvas_frames_keep_stationary_release_identical_to_the_held_stroke() {
    check_paint_release(false);
}

#[test]
fn actual_canvas_frames_keep_moving_soft_brush_release_identical_and_undoable() {
    check_paint_release(true);
}

fn check_paint_release(moving: bool) {
    for blend in [
        crate::canvas::BlendMode::Normal,
        crate::canvas::BlendMode::HardLight,
    ] {
        for mode in [
            crate::components::tools::BrushMode::Normal,
            crate::components::tools::BrushMode::Uniform,
            crate::components::tools::BrushMode::BuildUp,
        ] {
            let mut app = app();
            let ctx = context();
            app.tools_panel.active_tool = Tool::Brush;
            app.tools_panel.properties.size = 16.;
            app.tools_panel.properties.hardness = 0.5;
            app.tools_panel.properties.opacity = 0.5;
            app.tools_panel.properties.flow = 0.3;
            app.tools_panel.properties.brush_mode = mode;
            app.tools_panel.properties.blending_mode = blend;
            app.tools_panel.properties.pressure_size = false;
            app.tools_panel.properties.pressure_opacity = false;
            let frame = |app: &mut PaintFEApp, events| {
                ctx.run_ui(
                    egui::RawInput {
                        events,
                        screen_rect: Some(egui::Rect::from_min_size(
                            egui::Pos2::ZERO,
                            egui::vec2(900., 700.),
                        )),
                        ..Default::default()
                    },
                    |ui| app.show_runtime_canvas_tail(&ctx, ui, false),
                )
            };
            for _ in 0..3 {
                let _ = frame(&mut app, vec![]);
            }
            assert_eq!(app.tools_panel.properties.brush_mode, mode);
            assert_eq!(app.tools_panel.properties.blending_mode, blend);
            let point = app.canvas.last_image_rect.unwrap().center();
            let button = |pressed| egui::Event::PointerButton {
                pos: point,
                button: egui::PointerButton::Primary,
                pressed,
                modifiers: egui::Modifiers::NONE,
            };
            let _ = frame(
                &mut app,
                vec![egui::Event::PointerMoved(point), button(true)],
            );
            let before = app.projects[0].canvas_state.layers[0]
                .pixels
                .to_rgba_image();
            let release = if moving {
                point + egui::vec2(5.3, 1.1)
            } else {
                point
            };
            if moving {
                let _ = frame(&mut app, vec![egui::Event::PointerMoved(release)]);
            }
            let bounds = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(32., 24.));
            let expected = app.projects[0].canvas_state.composite_partial(bounds).0;
            for _ in 0..3 {
                let _ = frame(&mut app, vec![]);
            }
            assert_eq!(
                app.projects[0]
                    .canvas_state
                    .composite_partial(bounds)
                    .0
                    .pixels,
                expected.pixels,
                "{blend:?} {mode:?} hold"
            );
            let _ = frame(
                &mut app,
                vec![egui::Event::PointerButton {
                    pos: release,
                    button: egui::PointerButton::Primary,
                    pressed: false,
                    modifiers: egui::Modifiers::NONE,
                }],
            );
            let state = &app.projects[0].canvas_state;
            assert_eq!(
                state.composite_committed_downscaled(bounds, 1).0.pixels,
                expected.pixels,
                "{blend:?} {mode:?} release"
            );
            assert!(state.preview_clear_pending);
            assert!(state.preview_texture_replaces_composite);
            assert_eq!(app.projects[0].history.undo_count(), 1);
            for _ in 0..30 {
                let _ = frame(&mut app, vec![]);
                if !app.projects[0].canvas_state.preview_clear_pending {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
            assert!(!app.projects[0].canvas_state.preview_clear_pending);
            assert!(app.projects[0].canvas_state.preview_layer.is_none());
            assert_eq!(
                app.projects[0]
                    .canvas_state
                    .composite_partial(bounds)
                    .0
                    .pixels,
                expected.pixels
            );
            app.perform_undo();
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
fn docked_appearance_is_used_by_the_actual_frame_painter() {
    let mut app = app();
    let ctx = context();
    app.theme = crate::theme::Theme::light();
    app.theme.apply(&ctx);
    app.theme.tool_shelf_bg = egui::Color32::from_rgba_unmultiplied(177, 83, 29, 128);
    app.settings.tool_shelf.docked = true;
    app.settings.tool_shelf.opacity = Some(0.5);
    app.settings.tool_shelf.top_separator = Some([17, 41, 233]);
    app.settings.tool_shelf.separator_opacity = 0.25;
    app.settings.tool_shelf_rounding = Some(12.);
    let mut output = None;
    for tick in 0..3 {
        output = Some(ctx.run_ui(
            egui::RawInput {
                time: Some(tick as f64),
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(900., 700.),
                )),
                ..Default::default()
            },
            |ui| app.show_runtime_canvas_tail(&ctx, ui, false),
        ));
    }
    let output = output.unwrap();
    let fill = app.theme.tool_shelf_bg.gamma_multiply(0.5);
    assert!(output.shapes.iter().any(|shape|matches!(&shape.shape,egui::epaint::Shape::Rect(rect) if rect.fill==fill && rect.corner_radius.nw==12 && rect.rect.width()>800.)));
    let separator = egui::Color32::from_rgb(17, 41, 233).gamma_multiply(0.25);
    assert!(output.shapes.iter().any(|shape|matches!(&shape.shape,egui::epaint::Shape::LineSegment{stroke,..} if stroke.color==separator)));
}

#[test]
fn pixel_grid_is_painted_once_above_live_brush_previews_at_pixel_zoom() {
    fn flatten<'a>(shape: &'a egui::Shape, result: &mut Vec<&'a egui::Shape>) {
        if let egui::Shape::Vec(children) = shape {
            for child in children {
                flatten(child, result);
            }
        } else {
            result.push(shape);
        }
    }
    for blend in [
        crate::canvas::BlendMode::Normal,
        crate::canvas::BlendMode::HardLight,
        crate::canvas::BlendMode::Multiply,
        crate::canvas::BlendMode::Overwrite,
    ] {
        for zoom in [8., 16.] {
            for transparent in [false, true] {
                let mut app = app();
                let ctx = context();
                app.settings.pixel_grid_opacity = 1.;
                app.settings.pixel_grid_dashed = false;
                let grid = egui::Color32::from_rgb(17, 211, 109);
                app.settings.pixel_grid_outline_color = grid;
                app.tools_panel.active_tool = Tool::Brush;
                app.tools_panel.properties.size = 5.;
                app.tools_panel.properties.blending_mode = blend;
                app.tools_panel.properties.opacity = 0.4;
                app.tools_panel.properties.pressure_size = false;
                app.tools_panel.properties.pressure_opacity = false;
                if transparent {
                    app.projects[0].canvas_state.layers[0].pixels = TiledImage::new(32, 24);
                    app.projects[0].canvas_state.mark_dirty(None);
                }
                app.projects[0].canvas_state.show_pixel_grid = true;
                app.projects[0].canvas_state.selection_mask =
                    Some(image::GrayImage::from_pixel(32, 24, image::Luma([255])));
                let frame = |app: &mut PaintFEApp, events| {
                    ctx.run_ui(
                        egui::RawInput {
                            events,
                            screen_rect: Some(egui::Rect::from_min_size(
                                egui::Pos2::ZERO,
                                egui::vec2(900., 700.),
                            )),
                            ..Default::default()
                        },
                        |ui| app.show_runtime_canvas_tail(&ctx, ui, false),
                    )
                };
                for _ in 0..3 {
                    let _ = frame(&mut app, vec![]);
                }
                app.canvas.zoom = zoom;
                let _ = frame(&mut app, vec![]);
                let baseline = frame(&mut app, vec![]);
                let is_grid = |shape: &egui::Shape, ctx: &egui::Context| {
                    if let egui::Shape::Mesh(mesh) = shape {
                        ctx.tex_manager()
                            .read()
                            .meta(mesh.texture_id)
                            .is_some_and(|meta| {
                                matches!(
                                    meta.name.as_str(),
                                    "pixel_grid_vertical" | "pixel_grid_horizontal"
                                )
                            })
                    } else {
                        false
                    }
                };
                let grid_count = |output: &egui::FullOutput| {
                    let mut shapes = Vec::new();
                    for shape in &output.shapes {
                        flatten(&shape.shape, &mut shapes);
                    }
                    shapes.iter().filter(|shape| is_grid(shape, &ctx)).count()
                };
                let count = grid_count(&baseline);
                assert_eq!(count, 2, "Both grid axes must be visible at {zoom}x");
                let point = app.canvas.last_image_rect.unwrap().center();
                let button = |pressed| egui::Event::PointerButton {
                    pos: point,
                    button: egui::PointerButton::Primary,
                    pressed,
                    modifiers: egui::Modifiers::NONE,
                };
                for events in [
                    vec![egui::Event::PointerMoved(point), button(true)],
                    vec![egui::Event::PointerMoved(point + egui::vec2(zoom, zoom))],
                    vec![],
                    vec![button(false)],
                ] {
                    let output = frame(&mut app, events);
                    assert_eq!(
                        grid_count(&output),
                        count,
                        "Grid must not be duplicated or removed: {blend:?} {zoom}"
                    );
                    let state = &app.projects[0].canvas_state;
                    let mut ids: Vec<_> = state
                        .preview_tile_textures
                        .values()
                        .map(|texture| texture.id())
                        .collect();
                    ids.extend(
                        state
                            .preview_texture_cache
                            .as_ref()
                            .map(|texture| texture.id()),
                    );
                    assert!(!ids.is_empty(), "Exercise an actual held/committed preview");
                    let mut shapes = Vec::new();
                    for shape in &output.shapes {
                        flatten(&shape.shape, &mut shapes);
                    }
                    let preview = shapes.iter().rposition(|shape| matches!(shape,egui::Shape::Mesh(mesh) if ids.contains(&mesh.texture_id))).unwrap();
                    let first_grid = shapes
                        .iter()
                        .position(|shape| is_grid(shape, &ctx))
                        .unwrap();
                    assert!(
                        preview < first_grid,
                        "Preview covers the grid: {blend:?} {zoom}"
                    );
                }
            }
        }
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

#[test]
fn compact_brush_controls_preserve_values_and_toolbar_height() {
    fn collect_text(shape: &egui::Shape, labels: &mut Vec<String>) {
        match shape {
            egui::Shape::Text(text) => labels.push(text.galley.text().to_owned()),
            egui::Shape::Vec(shapes) => {
                for shape in shapes {
                    collect_text(shape, labels);
                }
            }
            _ => {}
        }
    }
    let mut app = app();
    app.tools_panel.active_tool = Tool::Brush;
    app.tools_panel.properties.opacity = 0.4;
    app.tools_panel.properties.flow = 0.2;
    let ctx = context();
    let mut heights = Vec::new();
    for show in [false, true, false] {
        app.settings.show_advanced_brush_controls = show;
        let mut height = 0.;
        let output = ctx.run_ui(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(1600., 600.),
                )),
                ..Default::default()
            },
            |ui| {
                height = ui
                    .horizontal(|ui| {
                        app.tools_panel.show_context_bar(
                            ui,
                            &app.assets,
                            egui::Color32::BLACK,
                            egui::Color32::WHITE,
                            &app.theme,
                            &app.settings,
                        );
                    })
                    .response
                    .rect
                    .height();
            },
        );
        let mut labels = Vec::new();
        for shape in output.shapes {
            collect_text(&shape.shape, &mut labels);
        }
        for key in ["Opacity", "Flow"] {
            assert_eq!(labels.contains(&crate::i18n::translate_english(key)), show);
        }
        assert_eq!(app.tools_panel.properties.opacity, 0.4);
        assert_eq!(app.tools_panel.properties.flow, 0.2);
        heights.push(height);
    }
    assert!(
        heights
            .iter()
            .all(|height| (height - heights[0]).abs() < 0.1)
    );
}

#[test]
fn generated_paintbrush_stroke_keeps_normal_undo_and_redo() {
    for tip in crate::brushes::PaintTip::ALL {
        let mut app = app();
        let ctx = context();
        app.tools_panel.active_tool = Tool::Brush;
        app.tools_panel.properties.size = 10.;
        app.tools_panel.select_brush_tip(
            crate::components::tools::BrushTip::Image(tip.name().into()),
            &app.assets,
        );
        app.tools_panel.properties.blending_mode = crate::canvas::BlendMode::HardLight;
        app.tools_panel.properties.opacity = 0.4;
        app.tools_panel.properties.flow = 0.35;
        app.tools_panel.properties.hardness = 1.;
        let before = app.projects[0].canvas_state.layers[0]
            .pixels
            .to_rgba_image();
        for _ in 0..3 {
            tool_frame(&mut app, &ctx, (10., 10.), vec![]);
        }
        let rect = app.canvas.last_image_rect.unwrap();
        let first = egui::pos2(
            rect.min.x + 8. * app.canvas.zoom,
            rect.min.y + 8. * app.canvas.zoom,
        );
        let second = egui::pos2(
            rect.min.x + 22. * app.canvas.zoom,
            rect.min.y + 16. * app.canvas.zoom,
        );
        let button = |position, pressed| egui::Event::PointerButton {
            pos: position,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::NONE,
        };
        tool_frame(
            &mut app,
            &ctx,
            (8., 8.),
            vec![egui::Event::PointerMoved(first), button(first, true)],
        );
        let corner = egui::pos2(
            rect.min.x + 22. * app.canvas.zoom,
            rect.min.y + 8. * app.canvas.zoom,
        );
        tool_frame(
            &mut app,
            &ctx,
            (22., 8.),
            vec![egui::Event::PointerMoved(corner)],
        );
        tool_frame(
            &mut app,
            &ctx,
            (22., 16.),
            vec![egui::Event::PointerMoved(second)],
        );
        let full = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(32., 24.));
        let preview = app.projects[0].canvas_state.composite_partial(full).0;
        tool_frame(&mut app, &ctx, (22., 16.), vec![button(second, false)]);
        let committed = app.projects[0].canvas_state.composite_partial(full).0;
        assert!(
            preview.pixels.iter().zip(&committed.pixels).all(|(a, b)| a
                .to_array()
                .iter()
                .zip(b.to_array())
                .all(|(a, b)| (*a as i16 - b as i16).abs() <= 1)),
            "{} preview handover",
            tip.name()
        );
        let after = app.projects[0].canvas_state.layers[0]
            .pixels
            .to_rgba_image();
        assert_ne!(before, after, "{} must paint", tip.name());
        app.perform_undo();
        assert_eq!(
            before,
            app.projects[0].canvas_state.layers[0]
                .pixels
                .to_rgba_image()
        );
        app.perform_redo();
        assert_eq!(
            after,
            app.projects[0].canvas_state.layers[0]
                .pixels
                .to_rgba_image()
        );
    }
}

#[test]
fn held_paint_tool_switch_finishes_once_and_undo_redo_restore_exact_pixels() {
    for original in [Tool::Brush, Tool::Pencil, Tool::Eraser] {
        for next in [Tool::Fill, Tool::Eraser, Tool::Brush] {
            let mut app = app();
            let ctx = context();
            app.tools_panel.change_tool(original);
            app.tools_panel.properties.size = 7.;
            app.tools_panel.properties.pressure_size = false;
            app.tools_panel.properties.pressure_opacity = false;
            let before = app.projects[0].canvas_state.layers[0]
                .pixels
                .to_rgba_image();
            let pos = (10., 10.);
            tool_frame(&mut app, &ctx, pos, vec![brush_pointer(pos, true)]);
            app.tools_panel.change_tool(next);
            assert_eq!(app.tools_panel.active_tool, original);
            tool_frame(&mut app, &ctx, pos, vec![]);
            assert_eq!(app.projects[0].history.undo_count(), 0);
            tool_frame(&mut app, &ctx, pos, vec![brush_pointer(pos, false)]);
            assert_eq!(app.tools_panel.active_tool, next);
            assert_eq!(app.projects[0].history.undo_count(), 1);
            let after = app.projects[0].canvas_state.layers[0]
                .pixels
                .to_rgba_image();
            assert_ne!(before, after, "{original:?}");
            app.perform_undo();
            assert_eq!(
                app.projects[0].canvas_state.layers[0]
                    .pixels
                    .to_rgba_image(),
                before
            );
            app.perform_redo();
            assert_eq!(
                app.projects[0].canvas_state.layers[0]
                    .pixels
                    .to_rgba_image(),
                after
            );
        }
    }
}

#[test]
fn undo_during_held_paint_undoes_that_stroke_without_restarting_until_release() {
    for tool in [Tool::Brush, Tool::Pencil, Tool::Eraser] {
        let mut app = app();
        let ctx = context();
        app.tools_panel.change_tool(tool);
        app.tools_panel.properties.size = 6.;
        app.tools_panel.properties.pressure_size = false;
        let before = app.projects[0].canvas_state.layers[0]
            .pixels
            .to_rgba_image();
        let pos = (12., 10.);
        tool_frame(&mut app, &ctx, pos, vec![brush_pointer(pos, true)]);
        app.perform_undo();
        assert_eq!(
            app.projects[0].canvas_state.layers[0]
                .pixels
                .to_rgba_image(),
            before
        );
        assert_eq!(app.projects[0].history.undo_count(), 0);
        tool_frame(&mut app, &ctx, pos, vec![]);
        tool_frame(&mut app, &ctx, pos, vec![brush_pointer(pos, false)]);
        assert_eq!(
            app.projects[0].canvas_state.layers[0]
                .pixels
                .to_rgba_image(),
            before
        );
        app.perform_redo();
        assert_ne!(
            app.projects[0].canvas_state.layers[0]
                .pixels
                .to_rgba_image(),
            before
        );
    }
}

#[test]
fn painting_tool_settings_are_independent_and_survive_restart() {
    let mut tools = crate::ui::panels::tools::ToolsPanel::default();
    tools.properties.size = 31.;
    tools.properties.flow = 0.42;
    tools.properties.opacity = 0.63;
    tools.change_tool(Tool::Pencil);
    tools.properties.size = 2.;
    tools.change_tool(Tool::Eraser);
    tools.properties.size = 77.;
    tools.properties.flow = 0.8;
    tools.change_tool(Tool::Brush);
    assert_eq!(tools.properties.size, 31.);
    assert_eq!(tools.properties.flow, 0.42);
    assert_eq!(tools.properties.opacity, 0.63);
    let mut settings = crate::config::settings::AppSettings::default();
    tools.store_brush_browser(&mut settings);
    settings.painting_tool_settings =
        serde_json::from_str(&serde_json::to_string(&settings.painting_tool_settings).unwrap())
            .unwrap();
    let mut restored = crate::ui::panels::tools::ToolsPanel::default();
    restored.restore_brush_browser(&settings);
    assert_eq!(restored.properties.size, 31.);
    restored.change_tool(Tool::Pencil);
    assert_eq!(restored.properties.size, 2.);
    restored.change_tool(Tool::Eraser);
    assert_eq!(restored.properties.size, 77.);
    assert_eq!(restored.properties.flow, 0.8);
    restored.change_tool(Tool::Brush);
    assert_eq!(restored.properties.flow, 0.42);
}

#[test]
fn command_catalogue_has_unique_bindable_localised_entries_and_safe_document_gates() {
    use crate::config::{
        commands::{COMMANDS, MenuCommand},
        keybindings::{BindableAction, KeyBindings, KeyCombo},
    };
    let mut app = app();
    let ctx = context();
    let mut seen = std::collections::HashSet::new();
    for spec in COMMANDS {
        assert!(seen.insert(spec.action), "duplicate {:?}", spec.action);
        assert!(
            BindableAction::all().contains(&spec.action),
            "missing {:?}",
            spec.action
        );
        assert_ne!(
            crate::i18n::translate(spec.label_key),
            spec.label_key,
            "{}",
            spec.label_key
        );
        assert_ne!(
            crate::i18n::translate(spec.category_key),
            spec.category_key,
            "{}",
            spec.category_key
        );
    }
    let bindings = KeyBindings::default();
    for (command, key) in [
        (MenuCommand::CanvasRotate90Cw, egui::Key::H),
        (MenuCommand::CanvasRotate90Ccw, egui::Key::G),
    ] {
        assert_eq!(
            bindings.get(BindableAction::Menu(command)),
            Some(&KeyCombo::ctrl_key(key))
        );
    }
    assert_eq!(
        bindings.get(BindableAction::Menu(MenuCommand::CanvasCropToSelection)),
        Some(&KeyCombo::ctrl_shift_key(egui::Key::X))
    );
    app.projects.clear();
    for spec in COMMANDS.iter().filter(|spec| spec.needs_document) {
        assert!(
            !app.menu_command_enabled(&ctx, spec.action),
            "{:?}",
            spec.action
        );
    }
    assert!(app.menu_command_enabled(&ctx, BindableAction::Menu(MenuCommand::EditPreferences)));
}

#[test]
fn menu_commands_rotate_and_history_restore_exact_pixels() {
    use crate::config::{commands::MenuCommand, keybindings::BindableAction};
    for command in [
        MenuCommand::CanvasRotate90Cw,
        MenuCommand::CanvasRotate90Ccw,
    ] {
        let mut app = app();
        let ctx = context();
        let before = app.projects[0].canvas_state.layers[0]
            .pixels
            .to_rgba_image();
        app.execute_menu_command(&ctx, BindableAction::Menu(command));
        assert_eq!(
            (
                app.projects[0].canvas_state.width,
                app.projects[0].canvas_state.height
            ),
            (24, 32)
        );
        let expected = if command == MenuCommand::CanvasRotate90Cw {
            image::imageops::rotate90(&before)
        } else {
            image::imageops::rotate270(&before)
        };
        assert_eq!(
            app.projects[0].canvas_state.layers[0]
                .pixels
                .to_rgba_image(),
            expected
        );
        app.perform_undo();
        assert_eq!(
            app.projects[0].canvas_state.layers[0]
                .pixels
                .to_rgba_image(),
            before
        );
    }
}

#[test]
fn selection_during_held_paint_commits_before_clipboard_reads_and_waits_for_release() {
    let mut app = app();
    let ctx = context();
    app.tools_panel.properties.pressure_size = false;
    let before = app.projects[0].canvas_state.layers[0]
        .pixels
        .to_rgba_image();
    let pos = (12., 10.);
    tool_frame(&mut app, &ctx, pos, vec![brush_pointer(pos, true)]);
    app.select_all_canvas();
    assert_eq!(app.projects[0].history.undo_count(), 2);
    let after = app.projects[0].canvas_state.layers[0]
        .pixels
        .to_rgba_image();
    assert_ne!(after, before);
    assert!(app.projects[0].canvas_state.has_selection());
    tool_frame(&mut app, &ctx, pos, vec![]);
    tool_frame(&mut app, &ctx, pos, vec![brush_pointer(pos, false)]);
    assert_eq!(app.projects[0].history.undo_count(), 2);
    assert_eq!(
        app.projects[0].canvas_state.layers[0]
            .pixels
            .to_rgba_image(),
        after
    );
}

#[test]
fn pencil_context_bar_never_exposes_ignored_controls_in_either_theme() {
    fn text(shape: &egui::Shape, labels: &mut Vec<String>) {
        match shape {
            egui::Shape::Text(t) => labels.push(t.galley.text().into()),
            egui::Shape::Vec(children) => {
                for child in children {
                    text(child, labels)
                }
            }
            _ => {}
        }
    }
    for dark in [false, true] {
        for scale in [1., 1.4] {
            let mut app = app();
            let ctx = context();
            app.theme = if dark {
                crate::theme::Theme::dark()
            } else {
                crate::theme::Theme::light()
            };
            app.theme.polish.text_scale = scale;
            app.theme.apply(&ctx);
            app.settings.show_advanced_brush_controls = true;
            app.settings.tablet_pressure_enabled = true;
            app.tools_panel.change_tool(Tool::Pencil);
            let output = ctx.run_ui(egui::RawInput::default(), |ui| {
                ui.horizontal(|ui| {
                    app.tools_panel.show_context_bar(
                        ui,
                        &app.assets,
                        egui::Color32::BLACK,
                        egui::Color32::WHITE,
                        &app.theme,
                        &app.settings,
                    )
                });
            });
            let mut labels = Vec::new();
            for shape in output.shapes {
                text(&shape.shape, &mut labels)
            }
            for key in ["Opacity", "Flow", "AA", "Hardness:", "Size:"] {
                assert!(
                    !labels.contains(&crate::i18n::translate_english(key)),
                    "{key}: {labels:?}"
                );
            }
            assert!(labels.contains(&crate::i18n::translate("ctx.blend")));
        }
    }
}

#[test]
fn new_command_shortcuts_respect_canvas_text_editing_and_use_the_same_execution_path() {
    for editing in [false, true] {
        let mut app = app();
        let ctx = context();
        app.tools_panel.active_tool = if editing { Tool::Text } else { Tool::Brush };
        app.tools_panel.text_state.is_editing = editing;
        let _ = ctx.run_ui(
            egui::RawInput {
                modifiers: egui::Modifiers::CTRL,
                events: vec![egui::Event::Key {
                    key: egui::Key::H,
                    physical_key: None,
                    pressed: true,
                    repeat: false,
                    modifiers: egui::Modifiers::CTRL,
                }],
                ..Default::default()
            },
            |_| {
                app.update_runtime_input(&ctx);
            },
        );
        assert_eq!(
            (
                app.projects[0].canvas_state.width,
                app.projects[0].canvas_state.height
            ),
            if editing { (32, 24) } else { (24, 32) }
        );
    }
}

#[test]
#[ignore = "manual issue review screenshots and rendered opacity comparison"]
fn render_issue_review_screenshots() {
    let gpu = crate::gpu::context::GpuContext::new("").expect("GPU for screenshot review");
    let output_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target/issue-pass-230-233/screenshots");
    for dark in [false, true] {
        let mut samples = Vec::new();
        for opacity in [0., 0.25, 0.5, 1.] {
            let mut app = app();
            let ctx = context();
            app.theme = if dark {
                crate::theme::Theme::dark()
            } else {
                crate::theme::Theme::light()
            };
            app.theme.apply(&ctx);
            app.settings.tool_shelf.docked = true;
            app.settings.tool_shelf.opacity = Some(opacity);
            app.tools_panel.change_tool(Tool::Pencil);
            let mut delta = egui::TexturesDelta::default();
            let mut output = None;
            for _ in 0..3 {
                let mut frame = ctx.run_ui(
                    egui::RawInput {
                        screen_rect: Some(egui::Rect::from_min_size(
                            egui::Pos2::ZERO,
                            egui::vec2(1000., 700.),
                        )),
                        ..Default::default()
                    },
                    |ui| app.show_runtime_canvas_tail(&ctx, ui, false),
                );
                delta.append(std::mem::take(&mut frame.textures_delta));
                output = Some(frame);
            }
            let mut output = output.unwrap();
            output.textures_delta = delta;
            let path = output_dir.join(format!(
                "docked-{}-{}.png",
                if dark { "dark" } else { "light" },
                (opacity * 100.) as u32
            ));
            crate::ui::panels::tools::brush_picker_tests::save_ui_capture(
                &gpu,
                &ctx,
                output,
                [1000, 700],
                &path,
            );
            let image = image::open(&path).unwrap().to_rgba8();
            samples.push(*image.get_pixel(850, 25));
        }
        assert_ne!(
            samples[0], samples[3],
            "docked opacity must change rendered pixels"
        );
        assert_ne!(
            samples[1], samples[2],
            "partial opacity must change rendered pixels"
        );
    }
}

#[test]
fn held_paint_tracks_originating_layer_through_reorder_and_deletion() {
    for delete in [false, true] {
        let mut app = app();
        let ctx = context();
        let extra =
            crate::canvas::Layer::new("Other".into(), 32, 24, image::Rgba([12, 34, 56, 255]));
        let untouched = extra.pixels.to_rgba_image();
        app.projects[0].canvas_state.layers.push(extra);
        app.tools_panel.properties.pressure_size = false;
        let original = app.projects[0].canvas_state.layers[0]
            .pixels
            .to_rgba_image();
        let pos = (12., 10.);
        tool_frame(&mut app, &ctx, pos, vec![brush_pointer(pos, true)]);
        if delete {
            app.projects[0].canvas_state.layers.remove(0);
        } else {
            app.projects[0].canvas_state.layers.swap(0, 1);
        }
        app.projects[0].canvas_state.active_layer_index = 0;
        tool_frame(&mut app, &ctx, pos, vec![]);
        tool_frame(&mut app, &ctx, pos, vec![brush_pointer(pos, false)]);
        assert_eq!(
            app.projects[0].canvas_state.layers[0]
                .pixels
                .to_rgba_image(),
            untouched
        );
        assert_eq!(app.projects[0].history.undo_count(), usize::from(!delete));
        if !delete {
            assert_ne!(
                app.projects[0].canvas_state.layers[1]
                    .pixels
                    .to_rgba_image(),
                original
            );
            app.perform_undo();
            assert_eq!(
                app.projects[0].canvas_state.layers[1]
                    .pixels
                    .to_rgba_image(),
                original
            );
        }
    }
}

#[test]
fn actual_fill_shortcut_during_held_brush_defers_switch_until_release() {
    let mut app = app();
    let ctx = context();
    app.tools_panel.properties.pressure_size = false;
    let pos = (12., 10.);
    tool_frame(&mut app, &ctx, pos, vec![brush_pointer(pos, true)]);
    let id = egui::Id::new("issue231_canvas_focus");
    app.canvas.canvas_widget_id = Some(id);
    let _ = ctx.run_ui(
        egui::RawInput {
            events: vec![egui::Event::Key {
                key: egui::Key::F,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: egui::Modifiers::NONE,
            }],
            ..Default::default()
        },
        |_| {
            ctx.memory_mut(|m| m.request_focus(id));
            app.update_runtime_input(&ctx);
        },
    );
    assert_eq!(app.tools_panel.active_tool, Tool::Brush);
    tool_frame(&mut app, &ctx, pos, vec![brush_pointer(pos, false)]);
    assert_eq!(app.tools_panel.active_tool, Tool::Fill);
    assert_eq!(app.projects[0].history.undo_count(), 1);
}

#[test]
fn release_over_blocking_ui_does_not_swallow_the_next_canvas_gesture() {
    let mut app = app();
    let ctx = context();
    let pos = (12., 10.);
    app.tools_panel.properties.pressure_size = false;
    tool_frame(&mut app, &ctx, pos, vec![brush_pointer(pos, true)]);
    app.finish_pending_paint_stroke();
    assert_eq!(app.projects[0].history.undo_count(), 1);
    // Modal UI receives the release, so the canvas tool does not run in this frame.
    let _ = ctx.run_ui(
        egui::RawInput {
            events: vec![brush_pointer(pos, false)],
            ..Default::default()
        },
        |_| {},
    );
    let next = (20., 15.);
    tool_frame(&mut app, &ctx, next, vec![brush_pointer(next, true)]);
    tool_frame(&mut app, &ctx, next, vec![brush_pointer(next, false)]);
    assert_eq!(app.projects[0].history.undo_count(), 2);
}

#[test]
fn another_button_cannot_restart_paint_while_the_original_contact_is_held() {
    for secondary in [false, true] {
        let mut app = app();
        let ctx = context();
        let pos = (12., 10.);
        app.tools_panel.properties.pressure_size = false;
        let button = |which, down| egui::Event::PointerButton {
            pos: egui::pos2(148., 140.),
            button: which,
            pressed: down,
            modifiers: egui::Modifiers::NONE,
        };
        let original = if secondary {
            egui::PointerButton::Secondary
        } else {
            egui::PointerButton::Primary
        };
        let other = if secondary {
            egui::PointerButton::Primary
        } else {
            egui::PointerButton::Secondary
        };
        tool_frame(&mut app, &ctx, pos, vec![button(original, true)]);
        app.finish_pending_paint_stroke();
        let committed = app.projects[0].canvas_state.layers[0]
            .pixels
            .to_rgba_image();
        tool_frame(&mut app, &ctx, pos, vec![button(other, true)]);
        tool_frame(&mut app, &ctx, pos, vec![button(other, false)]);
        tool_frame(&mut app, &ctx, pos, vec![button(original, false)]);
        assert_eq!(app.projects[0].history.undo_count(), 1);
        assert_eq!(
            app.projects[0].canvas_state.layers[0]
                .pixels
                .to_rgba_image(),
            committed
        );
    }
}

#[test]
fn crop_keeps_view_anchor_through_undo_and_redo_and_crops_layer_mask() {
    let mut app = app();
    let state = &mut app.projects[0].canvas_state;
    let mut selection = image::GrayImage::new(32, 24);
    for y in 3..9 {
        for x in 4..12 {
            selection.put_pixel(x, y, image::Luma([255]));
        }
    }
    state.selection_mask = Some(selection);
    state.layers[0].ensure_mask();
    state.layers[0]
        .mask
        .as_mut()
        .unwrap()
        .put_pixel(4, 3, image::Rgba([0, 0, 0, 255]));
    let pan = egui::vec2(20., 10.);
    app.canvas.set_view_state(2., pan);
    app.crop_to_selection_in_place();
    let cropped_pan = pan + egui::vec2(-16., -12.);
    assert_eq!(app.canvas.view_state(), (2., cropped_pan));
    let state = &app.projects[0].canvas_state;
    assert_eq!((state.width, state.height), (8, 6));
    let mask = state.layers[0].mask.as_ref().unwrap();
    assert_eq!((mask.width(), mask.height()), (8, 6));
    assert_eq!(mask.get_pixel(0, 0).0, [0, 0, 0, 255]);
    app.perform_undo();
    assert_eq!(app.canvas.view_state(), (2., pan));
    assert_eq!(app.projects[0].canvas_state.width, 32);
    app.perform_redo();
    assert_eq!(app.canvas.view_state(), (2., cropped_pan));
    assert_eq!(app.projects[0].canvas_state.width, 8);
}

#[test]
fn rotate_selection_command_can_cancel_reject_clipping_and_apply_once() {
    use crate::config::{commands::MenuCommand, keybindings::BindableAction};
    let mut app = app();
    let ctx = context();
    let state = &mut app.projects[0].canvas_state;
    let mut selection = image::GrayImage::new(32, 24);
    for y in 7..12 {
        for x in 8..15 {
            selection.put_pixel(x, y, image::Luma([255]));
        }
    }
    state.selection_mask = Some(selection.clone());
    let before = state.layers[0].pixels.to_rgba_image();
    let action = BindableAction::Menu(MenuCommand::CanvasRotateSelection);
    assert!(app.execute_menu_command(&ctx, action));
    {
        let overlay = app.paste_overlay.as_mut().unwrap();
        assert!(overlay.selection_rotation);
        overlay.rotation = 0.6;
        overlay.center = egui::Pos2::ZERO;
    }
    app.commit_paste_overlay(true);
    assert!(app.paste_overlay.is_some());
    assert_eq!(app.projects[0].history.undo_count(), 0);
    app.cancel_paste_overlay();
    assert_eq!(
        app.projects[0].canvas_state.selection_mask.as_ref(),
        Some(&selection)
    );
    assert_eq!(
        app.projects[0].canvas_state.layers[0]
            .pixels
            .to_rgba_image(),
        before
    );

    assert!(app.execute_menu_command(&ctx, action));
    app.paste_overlay.as_mut().unwrap().rotation = 0.6;
    app.finish_pending_line_edit(); // Same preflight used by Save and Save As.
    assert!(app.paste_overlay.is_none());
    assert_eq!(app.projects[0].history.undo_count(), 1);
    assert_ne!(
        app.projects[0].canvas_state.layers[0]
            .pixels
            .to_rgba_image(),
        before
    );
    app.perform_undo();
    assert_eq!(
        app.projects[0].canvas_state.selection_mask.as_ref(),
        Some(&selection)
    );
    assert_eq!(
        app.projects[0].canvas_state.layers[0]
            .pixels
            .to_rgba_image(),
        before
    );
}

#[test]
fn straighten_commits_with_enter_on_the_runtime_input_path() {
    let mut app = app();
    let ctx = context();
    app.start_straighten();
    app.straighten_session.as_mut().unwrap().angle_degrees = 15.;
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
    assert!(app.straighten_session.is_none());
    assert_eq!(
        app.projects[0].history.presentation_history()[0].action,
        HistoryAction::Straighten
    );
}

#[test]
fn eraser_resize_drag_does_not_paint_or_change_brush_size() {
    let mut app = app();
    let ctx = context();
    app.tools_panel.change_tool(Tool::Brush);
    app.tools_panel.properties.size = 19.;
    app.tools_panel.change_tool(Tool::Eraser);
    app.tools_panel.properties.size = 10.;
    let before = app.projects[0].canvas_state.layers[0]
        .pixels
        .to_rgba_image();
    let shift = egui::Modifiers::SHIFT;
    let frame = |app: &mut PaintFEApp, events| {
        let _ = ctx.run_ui(
            egui::RawInput {
                events,
                modifiers: shift,
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(900., 700.),
                )),
                ..Default::default()
            },
            |ui| app.show_runtime_canvas_tail(&ctx, ui, false),
        );
    };
    for _ in 0..3 {
        frame(&mut app, vec![]);
    }
    let start = app.canvas.last_image_rect.unwrap().center();
    let end = start + egui::vec2(12., 0.);
    let button = |pos, pressed| egui::Event::PointerButton {
        pos,
        button: egui::PointerButton::Primary,
        pressed,
        modifiers: shift,
    };
    frame(
        &mut app,
        vec![egui::Event::PointerMoved(start), button(start, true)],
    );
    frame(&mut app, vec![egui::Event::PointerMoved(end)]);
    frame(&mut app, vec![button(end, false)]);
    assert!(app.tools_panel.properties.size > 10.);
    assert_eq!(app.projects[0].history.undo_count(), 0);
    assert_eq!(
        app.projects[0].canvas_state.layers[0]
            .pixels
            .to_rgba_image(),
        before
    );
    app.tools_panel.change_tool(Tool::Brush);
    assert_eq!(app.tools_panel.properties.size, 19.);
}

#[test]
fn temporary_picker_samples_without_editing_and_does_not_paint_when_ctrl_is_released_first() {
    for (tool, pending) in [
        (Tool::Brush, false),
        (Tool::Line, false),
        (Tool::Line, true),
    ] {
        for secondary in [false, true] {
            let mut app = app();
            let ctx = context();
            app.tools_panel.active_tool = tool;
            let before = app.projects[0].canvas_state.layers[0]
                .pixels
                .to_rgba_image();
            let frame = |app: &mut PaintFEApp, modifiers, events| {
                let _ = ctx.run_ui(
                    egui::RawInput {
                        modifiers,
                        events,
                        screen_rect: Some(egui::Rect::from_min_size(
                            egui::Pos2::ZERO,
                            egui::vec2(900., 700.),
                        )),
                        ..Default::default()
                    },
                    |ui| {
                        app.update_runtime_input(ui.ctx());
                        app.show_runtime_canvas_tail(&ctx, ui, false);
                    },
                );
            };
            for _ in 0..3 {
                frame(&mut app, egui::Modifiers::NONE, vec![]);
            }
            let rect = app.canvas.last_image_rect.unwrap();
            let point = rect.min + egui::vec2(4.5, 5.5) * app.canvas.view_state().0;
            if pending {
                let endpoint = point + egui::vec2(12., 6.) * app.canvas.view_state().0;
                frame(
                    &mut app,
                    egui::Modifiers::NONE,
                    vec![
                        egui::Event::PointerMoved(point),
                        screen_pointer(point, true),
                    ],
                );
                frame(
                    &mut app,
                    egui::Modifiers::NONE,
                    vec![egui::Event::PointerMoved(endpoint)],
                );
                frame(
                    &mut app,
                    egui::Modifiers::NONE,
                    vec![screen_pointer(endpoint, false)],
                );
                assert_eq!(
                    app.tools_panel.line_state.line_tool.stage,
                    crate::components::tools::LineStage::Editing
                );
            }
            let button = if secondary {
                egui::PointerButton::Secondary
            } else {
                egui::PointerButton::Primary
            };
            let control = if secondary {
                egui::Key::ControlRight
            } else {
                egui::Key::ControlLeft
            };
            frame(
                &mut app,
                egui::Modifiers::CTRL,
                vec![egui::Event::Key {
                    key: control,
                    physical_key: Some(control),
                    pressed: true,
                    repeat: false,
                    modifiers: egui::Modifiers::CTRL,
                }],
            );
            assert_eq!(app.tools_panel.active_tool, Tool::ColorPicker);
            frame(
                &mut app,
                egui::Modifiers::CTRL,
                vec![
                    egui::Event::PointerMoved(point),
                    egui::Event::PointerButton {
                        pos: point,
                        button,
                        pressed: true,
                        modifiers: egui::Modifiers::CTRL,
                    },
                ],
            );
            assert_eq!(app.tools_panel.active_tool, Tool::ColorPicker);
            let (colour, picked_secondary) = app
                .tools_panel
                .last_picked_color
                .take()
                .expect("sampled colour");
            assert_eq!(colour.to_srgba_unmultiplied(), before.get_pixel(4, 5).0);
            assert_eq!(picked_secondary, secondary);
            frame(
                &mut app,
                egui::Modifiers::NONE,
                vec![
                    egui::Event::Key {
                        key: control,
                        physical_key: Some(control),
                        pressed: false,
                        repeat: false,
                        modifiers: egui::Modifiers::NONE,
                    },
                    egui::Event::PointerMoved(point + egui::vec2(2., 0.)),
                ],
            );
            assert_eq!(app.tools_panel.active_tool, tool);
            frame(
                &mut app,
                egui::Modifiers::NONE,
                vec![egui::Event::PointerButton {
                    pos: point,
                    button,
                    pressed: false,
                    modifiers: egui::Modifiers::NONE,
                }],
            );
            assert_eq!(
                app.projects[0].canvas_state.layers[0]
                    .pixels
                    .to_rgba_image(),
                before
            );
            assert_eq!(app.projects[0].history.undo_count(), 0);
        }
    }
}

#[test]
fn eraser_shortcut_toggles_once_honours_rebinding_and_leaves_text_alone() {
    use crate::config::keybindings::{BindableAction, KeyCombo};
    for editing in [false, true] {
        let mut app = app();
        let ctx = context();
        app.tools_panel.change_tool(Tool::Pencil);
        let focus_id = egui::Id::new("eraser_shortcut_canvas");
        let frame = |app: &mut PaintFEApp, key, repeat, pressed| {
            let _ = ctx.run_ui(
                egui::RawInput {
                    events: vec![egui::Event::Key {
                        key,
                        physical_key: None,
                        pressed,
                        repeat,
                        modifiers: egui::Modifiers::NONE,
                    }],
                    ..Default::default()
                },
                |ui| {
                    let response = ui.interact(ui.max_rect(), focus_id, egui::Sense::click());
                    response.request_focus();
                    app.canvas.canvas_widget_id = Some(focus_id);
                    if editing {
                        let mut text = String::new();
                        ui.add(egui::TextEdit::singleline(&mut text).id_salt("eraser_test_text"))
                            .request_focus();
                    }
                    app.update_runtime_input(ui.ctx());
                },
            );
        };
        frame(&mut app, egui::Key::E, false, true);
        if editing {
            assert_eq!(app.tools_panel.active_tool, Tool::Pencil);
            continue;
        }
        assert_eq!(app.tools_panel.active_tool, Tool::Eraser);
        frame(&mut app, egui::Key::E, true, true);
        assert_eq!(app.tools_panel.active_tool, Tool::Eraser);
        frame(&mut app, egui::Key::E, false, false);
        frame(&mut app, egui::Key::E, false, true);
        assert_eq!(app.tools_panel.active_tool, Tool::Pencil);
        app.settings
            .keybindings
            .set(BindableAction::ToolEraser, KeyCombo::key(egui::Key::Q));
        frame(&mut app, egui::Key::E, false, false);
        frame(&mut app, egui::Key::E, false, true);
        assert_eq!(app.tools_panel.active_tool, Tool::Pencil);
        frame(&mut app, egui::Key::Q, false, true);
        assert_eq!(app.tools_panel.active_tool, Tool::Eraser);
    }
}

#[test]
fn eraser_toggle_and_temporary_picker_preserve_an_in_progress_stroke() {
    use crate::config::keybindings::KeyCombo;
    for toggle_count in [1, 2] {
        let mut app = app();
        let ctx = context();
        app.tools_panel.properties.size = 6.;
        app.tools_panel.properties.pressure_size = false;
        let before = app.projects[0].canvas_state.layers[0]
            .pixels
            .to_rgba_image();
        let position = (12., 10.);
        tool_frame(
            &mut app,
            &ctx,
            position,
            vec![brush_pointer(position, true)],
        );
        assert!(app.tools_panel.is_stroke_active());
        let ctrl = KeyCombo::modifiers_only(true, false, false);
        let _ = ctx.run_ui(
            egui::RawInput {
                modifiers: egui::Modifiers::CTRL,
                ..Default::default()
            },
            |_| {
                app.tools_panel
                    .update_temporary_colour_picker(&ctx, Some(&ctrl), None, true);
            },
        );
        assert_eq!(app.tools_panel.active_tool, Tool::Brush);
        for _ in 0..toggle_count {
            app.tools_panel.toggle_eraser();
        }
        assert_eq!(app.tools_panel.active_tool, Tool::Brush);
        tool_frame(
            &mut app,
            &ctx,
            position,
            vec![brush_pointer(position, false)],
        );
        assert_eq!(
            app.tools_panel.active_tool,
            if toggle_count == 1 {
                Tool::Eraser
            } else {
                Tool::Brush
            }
        );
        assert_eq!(app.projects[0].history.undo_count(), 1);
        assert_ne!(
            app.projects[0].canvas_state.layers[0]
                .pixels
                .to_rgba_image(),
            before
        );
        app.perform_undo();
        assert_eq!(
            app.projects[0].canvas_state.layers[0]
                .pixels
                .to_rgba_image(),
            before
        );
    }
}

#[test]
#[ignore = "manual palette and layer thumbnail screenshot review"]
fn render_palette_and_thumbnail_review() {
    let gpu = crate::gpu::context::GpuContext::new("").expect("screenshot GPU");
    for dark in [false, true] {
        for (width, height) in [(320, 200), (160, 320), (120, 120)] {
            let mut app = app();
            let ctx = context();
            app.theme = if dark {
                crate::theme::Theme::dark()
            } else {
                crate::theme::Theme::light()
            };
            app.theme.apply(&ctx);
            app.assets.init(&ctx);
            app.assets.update_theme(&ctx, dark);
            app.window_visibility.palette = true;
            app.window_visibility.layers = true;
            app.palette_panel_pos = Some((30., 40.));
            app.palette_startup_target_pos = None;
            app.palette_reposition_settle_frames = 0;
            app.palette_panel_size = Some((360., 180.));
            app.layers_panel_right_offset = Some((200., 40.));
            app.layers_panel_size = Some((350., 260.));
            app.projects[0].canvas_state = CanvasState::new(width, height);
            let state = &mut app.projects[0].canvas_state;
            let image = RgbaImage::from_fn(width, height, |x, y| {
                image::Rgba([
                    30,
                    115,
                    195,
                    if x > width / 2 && y > height / 2 {
                        0
                    } else {
                        255
                    },
                ])
            });
            state.layers[0].pixels = TiledImage::from_rgba_image(&image);
            state.layers[0].name = "Transparent cut-out".into();
            state.layers.push(crate::canvas::Layer::new(
                "Opaque image".into(),
                width,
                height,
                image::Rgba([225, 110, 45, 255]),
            ));
            let mut delta = egui::TexturesDelta::default();
            let mut output = None;
            for tick in 0..4 {
                let mut frame = ctx.run_ui(
                    egui::RawInput {
                        screen_rect: Some(egui::Rect::from_min_size(
                            egui::Pos2::ZERO,
                            egui::vec2(900., 400.),
                        )),
                        time: Some(tick as f64 * 0.1),
                        ..Default::default()
                    },
                    |ui| {
                        ui.painter()
                            .rect_filled(ui.max_rect(), 0., app.theme.window_bg);
                        app.layers_panel_right_offset = Some((400., 40.));
                        app.show_floating_palette_panel(&ctx, false);
                        app.show_floating_layers_panel(&ctx, false);
                    },
                );
                delta.append(std::mem::take(&mut frame.textures_delta));
                output = Some(frame);
            }
            let mut output = output.unwrap();
            output.textures_delta = delta;
            let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!(
                "target/visual-refinement/screenshots/panels-{}-{}x{}.png",
                if dark { "dark" } else { "light" },
                width,
                height,
            ));
            crate::ui::panels::tools::brush_picker_tests::save_ui_capture(
                &gpu,
                &ctx,
                output,
                [900, 400],
                &path,
            );
        }
    }
}

#[test]
fn selected_quarter_turns_lift_once_preserve_background_and_commit_one_history_entry() {
    use crate::config::{commands::MenuCommand, keybindings::BindableAction};
    for (w, h) in [(7, 5), (8, 5), (7, 6), (8, 6)] {
        let mut app = app();
        let ctx = context();
        let mut selection = image::GrayImage::new(32, 24);
        for y in 8..8 + h {
            for x in 10..10 + w {
                selection.put_pixel(x, y, image::Luma([255]));
            }
        }
        app.projects[0].canvas_state.selection_mask = Some(selection.clone());
        let before = app.projects[0].canvas_state.layers[0]
            .pixels
            .to_rgba_image();
        let action = BindableAction::Menu(MenuCommand::CanvasRotate90Cw);
        for turn in 1..=4 {
            assert!(app.execute_menu_command(&ctx, action));
            let overlay = app.paste_overlay.as_ref().unwrap();
            assert_eq!(overlay.source.width(), w);
            assert_eq!(overlay.source.height(), h);
            assert!(overlay.rotation.abs() <= std::f32::consts::PI);
            assert_eq!(app.projects[0].history.undo_count(), 0);
            assert!(!app.menu_command_enabled(&ctx, BindableAction::ColorInvertColors));
            let source = app.projects[0].canvas_state.layers[0]
                .pixels
                .to_rgba_image();
            for y in 0..24 {
                for x in 0..32 {
                    if x < 10 || x >= 10 + w || y < 8 || y >= 8 + h {
                        assert_eq!(
                            source.get_pixel(x, y),
                            before.get_pixel(x, y),
                            "turn {turn}"
                        );
                    }
                }
            }
        }
        app.commit_paste_overlay(true);
        assert!(app.paste_overlay.is_none());
        assert_eq!(app.projects[0].history.undo_count(), 1);
        assert_eq!(
            app.projects[0].canvas_state.layers[0]
                .pixels
                .to_rgba_image(),
            before
        );
        assert_eq!(
            app.projects[0].canvas_state.selection_mask.as_ref(),
            Some(&selection)
        );
        app.perform_undo();
        assert_eq!(
            app.projects[0].canvas_state.layers[0]
                .pixels
                .to_rgba_image(),
            before
        );
        app.perform_redo();
        assert_eq!(
            app.projects[0].canvas_state.layers[0]
                .pixels
                .to_rgba_image(),
            before
        );
        assert!(app.projects[0].canvas_state.has_selection());
    }
}

#[test]
fn docked_shelf_keeps_real_image_and_grid_underlay_without_input_or_navigation_changes() {
    fn meshes<'a>(shape: &'a egui::Shape, result: &mut Vec<&'a egui::epaint::Mesh>) {
        match shape {
            egui::Shape::Mesh(mesh) => result.push(mesh),
            egui::Shape::Vec(children) => {
                for child in children {
                    meshes(child, result);
                }
            }
            _ => {}
        }
    }
    for dark in [false, true] {
        let mut app = app();
        let ctx = context();
        app.theme = if dark {
            crate::theme::Theme::dark()
        } else {
            crate::theme::Theme::light()
        };
        app.theme.apply(&ctx);
        app.settings.tool_shelf.docked = true;
        app.projects[0].canvas_state.show_pixel_grid = true;
        let frame = |app: &mut PaintFEApp, events| {
            ctx.run_ui(
                egui::RawInput {
                    events,
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(900., 700.),
                    )),
                    ..Default::default()
                },
                |ui| app.show_runtime_canvas_tail(&ctx, ui, false),
            )
        };
        for _ in 0..3 {
            let _ = frame(&mut app, vec![]);
        }
        let navigation = app.canvas.last_canvas_rect.unwrap();
        app.canvas
            .set_view_state(8., egui::vec2(0., 4. - navigation.center().y + 96.));
        let mut image_rect = None;
        for opacity in [0_f32, 0.5, 1.] {
            app.settings.tool_shelf.opacity = Some(opacity);
            for tool in [Tool::Brush, Tool::Fill, Tool::Line] {
                app.tools_panel.change_tool(tool);
                let output = frame(&mut app, vec![]);
                assert_eq!(app.canvas.last_canvas_rect.unwrap(), navigation);
                if let Some(rect) = image_rect {
                    assert_eq!(app.canvas.last_image_rect.unwrap(), rect);
                } else {
                    image_rect = app.canvas.last_image_rect;
                }
                let texture = app.projects[0]
                    .canvas_state
                    .composite_cache
                    .as_ref()
                    .unwrap()
                    .id();
                let mut actual_image = false;
                let mut actual_grid = false;
                for shape in &output.shapes {
                    if shape.clip_rect.top() >= navigation.top() {
                        continue;
                    }
                    let mut found = Vec::new();
                    meshes(&shape.shape, &mut found);
                    for mesh in found {
                        if mesh.vertices.iter().any(|v| v.pos.y < navigation.top()) {
                            actual_image |= mesh.texture_id == texture;
                            actual_grid |= ctx
                                .tex_manager()
                                .read()
                                .meta(mesh.texture_id)
                                .is_some_and(|m| m.name.starts_with("pixel_grid_"));
                        }
                    }
                }
                assert!(actual_image, "image was clipped at the docked shelf");
                assert!(actual_grid, "pixel grid was clipped at the docked shelf");
            }
        }
        app.tools_panel.change_tool(Tool::Brush);
        let before = app.projects[0].canvas_state.layers[0]
            .pixels
            .to_rgba_image();
        let x = app.canvas.last_image_rect.unwrap().center().x;
        let y = navigation.top() - 12.;
        let _ = frame(
            &mut app,
            vec![
                egui::Event::PointerMoved(egui::pos2(x, y)),
                screen_pointer(egui::pos2(x, y), true),
            ],
        );
        let _ = frame(&mut app, vec![screen_pointer(egui::pos2(x, y), false)]);
        assert_eq!(
            app.projects[0].canvas_state.layers[0]
                .pixels
                .to_rgba_image(),
            before
        );
        assert_eq!(app.projects[0].history.undo_count(), 0);
    }
}

fn screen_pointer(point: egui::Pos2, down: bool) -> egui::Event {
    egui::Event::PointerButton {
        pos: point,
        button: egui::PointerButton::Primary,
        pressed: down,
        modifiers: egui::Modifiers::NONE,
    }
}

#[test]
fn fast_line_gestures_keep_press_anchor_release_position_and_distinct_curve_handles() {
    for one_frame in [false, true] {
        let mut app = app();
        let ctx = context();
        app.tools_panel.active_tool = Tool::Line;
        let start = (4., 5.);
        let end = (24., 17.);
        let moved = egui::Event::PointerMoved(egui::pos2(100. + end.0 * 4., 100. + end.1 * 4.));
        let mut events = vec![brush_pointer(start, true), moved];
        if one_frame {
            events.push(brush_pointer(end, false));
        }
        tool_frame(&mut app, &ctx, end, events);
        if !one_frame {
            tool_frame(&mut app, &ctx, end, vec![brush_pointer(end, false)]);
        }
        let line = &app.tools_panel.line_state.line_tool;
        assert_eq!(line.stage, crate::components::tools::LineStage::Editing);
        assert_eq!(line.control_points[0], egui::pos2(start.0, start.1));
        assert_eq!(line.control_points[3], egui::pos2(end.0, end.1));
        assert_ne!(line.control_points[1], line.control_points[2]);
        let old = line.control_points[1];
        let target = old + egui::vec2(0., 3.);
        tool_frame(
            &mut app,
            &ctx,
            (target.x, target.y),
            vec![
                brush_pointer((old.x, old.y), true),
                egui::Event::PointerMoved(egui::pos2(100. + target.x * 4., 100. + target.y * 4.)),
                brush_pointer((target.x, target.y), false),
            ],
        );
        assert!(
            (app.tools_panel.line_state.line_tool.control_points[1] - target).length() < 0.0001
        );
        assert_eq!(app.projects[0].history.undo_count(), 0);
    }
}

#[test]
fn numeric_edit_finish_keys_do_not_also_apply_or_cancel_an_editable_line() {
    for key in [egui::Key::Enter, egui::Key::Escape] {
        let mut app = app();
        let ctx = context();
        app.tools_panel.active_tool = Tool::Line;
        tool_frame(
            &mut app,
            &ctx,
            (24., 17.),
            vec![
                brush_pointer((4., 5.), true),
                brush_pointer((24., 17.), false),
            ],
        );
        assert_eq!(
            app.tools_panel.line_state.line_tool.stage,
            crate::components::tools::LineStage::Editing
        );
        let mut value = 5_f32;
        for (index, events) in [
            vec![],
            vec![],
            vec![
                egui::Event::Key {
                    key: egui::Key::A,
                    physical_key: None,
                    pressed: true,
                    repeat: false,
                    modifiers: egui::Modifiers {
                        ctrl: true,
                        command: true,
                        ..Default::default()
                    },
                },
                egui::Event::Text("7".into()),
            ],
            vec![egui::Event::Key {
                key,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: egui::Modifiers::NONE,
            }],
        ]
        .into_iter()
        .enumerate()
        {
            let _ = ctx.run_ui(
                egui::RawInput {
                    events,
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(900., 700.),
                    )),
                    ..Default::default()
                },
                |ui| {
                    app.update_runtime_input(ui.ctx());
                    ui.add(
                        crate::ui::controls::Numeric::new(&mut value)
                            .id_salt("line_finish_key")
                            .focus_on_open(index == 0),
                    );
                    app.show_runtime_canvas_tail(&ctx, ui, false);
                },
            );
        }
        assert_eq!(value, if key == egui::Key::Enter { 7. } else { 5. });
        assert_eq!(
            app.tools_panel.line_state.line_tool.stage,
            crate::components::tools::LineStage::Editing
        );
        assert_eq!(app.projects[0].history.undo_count(), 0);
        let _ = ctx.run_ui(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(900., 700.),
                )),
                ..Default::default()
            },
            |ui| {
                app.update_runtime_input(ui.ctx());
                app.show_runtime_canvas_tail(&ctx, ui, false);
            },
        );
        assert_eq!(
            app.tools_panel.line_state.line_tool.stage,
            crate::components::tools::LineStage::Editing,
            "field key replayed on the following frame"
        );
        app.tools_panel.injected_enter_pressed = key == egui::Key::Enter;
        app.tools_panel.injected_escape_pressed = key == egui::Key::Escape;
        let _ = ctx.run_ui(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(900., 700.),
                )),
                ..Default::default()
            },
            |ui| app.show_runtime_canvas_tail(&ctx, ui, false),
        );
        app.commit_pending_tool_history();
        assert_eq!(
            app.tools_panel.line_state.line_tool.stage,
            crate::components::tools::LineStage::Idle
        );
        assert_eq!(
            app.projects[0].history.undo_count(),
            usize::from(key == egui::Key::Enter)
        );
    }
}

#[test]
fn selection_rotation_shortcuts_reject_text_and_mask_targets_before_cutting_pixels() {
    use crate::config::{commands::MenuCommand, keybindings::BindableAction};
    for text in [false, true] {
        let mut app = app();
        let ctx = context();
        if text {
            let project = &mut app.projects[0];
            crate::ops::canvas_ops::add_text_layer(&mut project.canvas_state, &mut project.history);
        } else {
            app.projects[0].canvas_state.layers[0].ensure_mask();
            app.projects[0].canvas_state.edit_layer_mask = true;
        }
        app.projects[0].canvas_state.selection_mask =
            Some(image::GrayImage::from_pixel(32, 24, image::Luma([255])));
        let before = app.projects[0]
            .canvas_state
            .layers
            .iter()
            .map(|l| l.pixels.to_rgba_image())
            .collect::<Vec<_>>();
        let count = app.projects[0].history.undo_count();
        for command in [
            MenuCommand::CanvasRotate90Cw,
            MenuCommand::CanvasRotate90Ccw,
            MenuCommand::CanvasRotate180,
        ] {
            assert!(!app.execute_menu_command(&ctx, BindableAction::Menu(command)));
            assert!(app.paste_overlay.is_none());
            assert_eq!(app.projects[0].history.undo_count(), count);
            assert_eq!(
                app.projects[0]
                    .canvas_state
                    .layers
                    .iter()
                    .map(|l| l.pixels.to_rgba_image())
                    .collect::<Vec<_>>(),
                before
            );
        }
    }
}
