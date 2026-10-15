use super::*;
use crate::ui::dialogs::*;
use image::{GrayImage, Luma, Rgba};

fn app() -> PaintFEApp {
    crate::i18n::init();
    let cc = eframe::CreationContext::_new_kittest(egui::Context::default());
    let mut app = PaintFEApp::new(&cc, Vec::new(), mpsc::channel().1);
    app.projects = vec![Project::new_untitled(1, 32, 24)];
    app.active_project_index = 0;
    app
}

fn context() -> egui::Context {
    let ctx = egui::Context::default();
    let mut fonts = egui::FontDefinitions::default();
    fonts.families.insert(
        egui::FontFamily::Name("WidgetTitle".into()),
        fonts.families[&egui::FontFamily::Proportional].clone(),
    );
    ctx.set_fonts(fonts);
    ctx
}

fn pattern() -> RgbaImage {
    RgbaImage::from_fn(32, 24, |x, y| {
        Rgba([
            (x * 7) as u8,
            (y * 9) as u8,
            ((x * 11 + y * 17) % 255) as u8,
            255,
        ])
    })
}

#[test]
fn auto_levels_ignores_sparse_outliers_and_preserves_alpha_and_selection() {
    let mut state = CanvasState::new(1000, 1);
    let image = RgbaImage::from_fn(1000, 1, |x, _| {
        let value = match x {
            0 => 0,
            1 => 255,
            _ => 80 + (x % 81) as u8,
        };
        Rgba([value, value, value, if x == 999 { 0 } else { 255 }])
    });
    state.layers[0].pixels = TiledImage::from_rgba_image(&image);
    state.selection_mask = Some(GrayImage::from_fn(1000, 1, |x, _| {
        Luma([if x < 800 { 255 } else { 0 }])
    }));
    crate::ops::adjustments::auto_levels(&mut state, 0);
    let result = state.layers[0].pixels.to_rgba_image();
    assert_eq!(result.get_pixel(81, 0)[0], 0);
    assert_eq!(result.get_pixel(80, 0)[0], 255);
    for x in 0..1000 {
        assert_eq!(result.get_pixel(x, 0)[3], image.get_pixel(x, 0)[3]);
        if x >= 800 {
            assert_eq!(result.get_pixel(x, 0), image.get_pixel(x, 0));
        }
    }
    // Small images keep an exact min/max stretch, and a flat channel is stable.
    let small = RgbaImage::from_raw(
        3,
        1,
        vec![80, 42, 100, 255, 120, 42, 150, 255, 160, 42, 200, 255],
    )
    .unwrap();
    state = CanvasState::new(3, 1);
    state.layers[0].pixels = TiledImage::from_rgba_image(&small);
    crate::ops::adjustments::auto_levels(&mut state, 0);
    assert_eq!(
        state.layers[0].pixels.to_rgba_image().as_raw(),
        &[0, 42, 0, 255, 128, 42, 128, 255, 255, 42, 255, 255]
    );
}

#[test]
fn pixel_drag_directions_produce_distinct_results() {
    let original = pattern();
    let results: Vec<_> = [0., 90., 180., -90.]
        .into_iter()
        .map(|angle| crate::ops::effects::pixel_drag_core(&original, 42, 100., 5, angle, None))
        .collect();
    for (index, result) in results.iter().enumerate() {
        assert_ne!(result, &original);
        for other in &results[..index] {
            assert_ne!(result, other);
        }
    }
}

fn frame(app: &mut PaintFEApp, ctx: &egui::Context, key: Option<egui::Key>) {
    let events = key
        .map(|key| egui::Event::Key {
            key,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        })
        .into_iter()
        .collect();
    let _ = ctx.run_ui(
        egui::RawInput {
            events,
            ..Default::default()
        },
        |ui| {
            app.process_active_dialog(ui.ctx());
        },
    );
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while app.pending_filter_jobs > 0 {
        assert!(std::time::Instant::now() < deadline, "preview timed out");
        app.update_runtime_lifecycle_async(ctx);
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
}

fn pixels(app: &PaintFEApp) -> RgbaImage {
    app.projects[0].canvas_state.layers[0]
        .pixels
        .to_rgba_image()
}

macro_rules! selected_dialog_test {
    ($name:ident, $dialog:ident, $variant:ident, $configure:expr) => {
        #[test]
        fn $name() {
            // Exercise real dialog dispatch, asynchronous preview, Apply and Undo.
            for preview in [true, false] {
                let mut app = app();
                let original = pattern();
                let state = &mut app.projects[0].canvas_state;
                state.layers[0].pixels = TiledImage::from_rgba_image(&original);
                state.selection_mask = Some(GrayImage::from_fn(32, 24, |x, _| {
                    Luma([if x < 12 { 255 } else { 0 }])
                }));
                let mut dialog = $dialog::new(state);
                dialog.pending_flat = None;
                dialog.original_flat = Some(original.clone());
                dialog.live_preview = preview;
                ($configure)(&mut dialog);
                app.active_dialog = ActiveDialog::$variant(dialog);
                let ctx = context();
                frame(&mut app, &ctx, None);
                if preview {
                    assert_ne!(
                        pixels(&app),
                        original,
                        "preview must change selected pixels"
                    );
                    for y in 0..24 {
                        for x in 12..32 {
                            assert_eq!(
                                pixels(&app).get_pixel(x, y),
                                original.get_pixel(x, y),
                                "preview leaked outside selection"
                            );
                        }
                    }
                }
                frame(&mut app, &ctx, Some(egui::Key::Enter));
                assert!(matches!(app.active_dialog, ActiveDialog::None));
                let result = pixels(&app);
                assert_ne!(result, original, "Apply must change selected pixels");
                for y in 0..24 {
                    for x in 12..32 {
                        assert_eq!(
                            result.get_pixel(x, y),
                            original.get_pixel(x, y),
                            "Apply leaked outside selection"
                        );
                    }
                }
                let project = &mut app.projects[0];
                project.history.undo(&mut project.canvas_state);
                assert_eq!(pixels(&app), original, "Undo must restore exact bytes");
            }
            let mut app = app();
            let original = pattern();
            app.projects[0].canvas_state.layers[0].pixels = TiledImage::from_rgba_image(&original);
            let mut dialog = $dialog::new(&app.projects[0].canvas_state);
            dialog.pending_flat = None;
            dialog.original_flat = Some(original.clone());
            ($configure)(&mut dialog);
            app.active_dialog = ActiveDialog::$variant(dialog);
            let ctx = context();
            frame(&mut app, &ctx, None);
            frame(&mut app, &ctx, Some(egui::Key::Escape));
            assert_eq!(pixels(&app), original, "Cancel must restore preview");
        }
    };
}

selected_dialog_test!(
    pixel_drag_selection_preview_apply_cancel_undo,
    PixelDragDialog,
    PixelDrag,
    |d: &mut PixelDragDialog| {
        d.amount = 100.;
        d.distance = 5.;
        d.direction = 90.;
    }
);
selected_dialog_test!(
    rgb_displace_selection_preview_apply_cancel_undo,
    RgbDisplaceDialog,
    RgbDisplace,
    |d: &mut RgbDisplaceDialog| {
        d.r_x = 3.;
        d.g_y = 2.;
    }
);
selected_dialog_test!(
    ink_selection_preview_apply_cancel_undo,
    InkDialog,
    Ink,
    |d: &mut InkDialog| {
        d.edge_strength = 200.;
        d.threshold = 0.3;
    }
);
selected_dialog_test!(
    oil_painting_selection_preview_apply_cancel_undo,
    OilPaintingDialog,
    OilPainting,
    |d: &mut OilPaintingDialog| {
        d.radius = 3.;
        d.levels = 5.;
    }
);
selected_dialog_test!(
    color_filter_selection_preview_apply_cancel_undo,
    ColorFilterDialog,
    ColorFilter,
    |d: &mut ColorFilterDialog| {
        d.color = [1., 0., 0.];
        d.intensity = 1.;
    }
);

#[test]
fn canvas_border_preview_and_apply_draw_all_four_edges() {
    let mut app = app();
    let original = pattern();
    app.projects[0].canvas_state.layers[0].pixels = TiledImage::from_rgba_image(&original);
    app.colors_panel.primary_color = egui::Color32::BLACK;
    let mut dialog = CanvasBorderDialog::new(&app.projects[0].canvas_state);
    dialog.pending_flat = None;
    dialog.original_flat = Some(original.clone());
    dialog.width = 3.;
    app.active_dialog = ActiveDialog::CanvasBorder(dialog);
    let ctx = context();
    frame(&mut app, &ctx, None);
    let preview = pixels(&app);
    for y in 0..24 {
        for x in 0..32 {
            let expected = if x < 3 || y < 3 || x >= 29 || y >= 21 {
                Rgba([0, 0, 0, 255])
            } else {
                *original.get_pixel(x, y)
            };
            assert_eq!(*preview.get_pixel(x, y), expected, "border at {x},{y}");
        }
    }
    frame(&mut app, &ctx, Some(egui::Key::Enter));
    assert_eq!(pixels(&app), preview);
    let p = &mut app.projects[0];
    p.history.undo(&mut p.canvas_state);
    assert_eq!(pixels(&app), original);
}

#[test]
fn anchored_panels_preserve_saved_positions_through_startup_dpi_resize() {
    for offset in [300., 1480.] {
        let mut app = app();
        app.layers_panel_right_offset = Some((offset, 150.));
        app.history_panel_right_offset = Some((offset, 400.));
        app.palette_reposition_settle_frames = 8;
        app.window_visibility.layers = true;
        app.window_visibility.history = true;
        let ctx = context();
        ctx.set_pixels_per_point(1.25);
        for width in [800., 1536., 1536.] {
            let _ = ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(width, 864.),
                    )),
                    ..Default::default()
                },
                |ui| {
                    crate::ui::workspace::begin_frame(
                        ui.ctx(),
                        &app.settings.workspace,
                        &[("Layers", true), ("History", true)],
                    );
                    app.show_floating_layers_panel(ui.ctx(), true);
                    app.show_floating_history_panel(ui.ctx(), true);
                },
            );
            assert_eq!(app.layers_panel_right_offset, Some((offset, 150.)));
            assert_eq!(app.history_panel_right_offset, Some((offset, 400.)));
        }
        let layout = crate::ui::workspace::capture(&ctx, "Startup".into()).unwrap();
        for panel in layout.panels {
            assert!(
                (panel.position[0] - (1536. - offset)).abs() < 2.,
                "{} failed to return to its saved horizontal position",
                panel.panel
            );
        }
    }
}

#[test]
fn save_and_adjustment_dialogs_block_document_shortcuts_but_allow_canvas_wheel() {
    for save in [true, false] {
        let mut app = app();
        app.projects[0]
            .history
            .push(Box::new(crate::components::history::MarkerCommand::new(
                "Existing edit",
            )));
        if save {
            app.save_file_dialog.open = true;
        } else {
            app.active_dialog =
                ActiveDialog::Levels(LevelsDialog::new(&app.projects[0].canvas_state));
        }
        let original = pixels(&app);
        let ctx = context();
        // Real headers install egui's modal layer. It reports the dialog's
        // layer even outside its rectangle, so render it before testing zoom.
        let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
            if save {
                app.save_file_dialog.show(ui.ctx());
            } else {
                app.process_active_dialog(ui.ctx());
            }
        });
        for key in [
            egui::Key::N,
            egui::Key::W,
            egui::Key::Z,
            egui::Key::X,
            egui::Key::V,
        ] {
            let _ = ctx.run_ui(
                egui::RawInput {
                    modifiers: egui::Modifiers::CTRL,
                    events: vec![egui::Event::Key {
                        key,
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
            assert_eq!(app.projects.len(), 1);
            assert_eq!(pixels(&app), original);
            assert_eq!(app.projects[0].history.undo_count(), 1);
        }
        app.canvas.last_canvas_rect = Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(800., 600.),
        ));
        let zoom = app.canvas.zoom;
        let _ = ctx.run_ui(
            egui::RawInput {
                events: vec![
                    egui::Event::PointerMoved(egui::pos2(700., 500.)),
                    egui::Event::MouseWheel {
                        phase: egui::TouchPhase::Move,
                        unit: egui::MouseWheelUnit::Line,
                        delta: egui::vec2(0., 1.),
                        modifiers: egui::Modifiers::NONE,
                    },
                ],
                ..Default::default()
            },
            |ui| {
                app.update_runtime_input(ui.ctx());
            },
        );
        assert_ne!(app.canvas.zoom, zoom, "wheel outside modal must still zoom");
    }
}

#[test]
fn completed_file_load_waits_until_the_adjustment_closes() {
    let mut app = app();
    let ctx = context();
    let original_id = app.projects[0].id;
    app.active_dialog = ActiveDialog::Levels(LevelsDialog::new(&app.projects[0].canvas_state));
    let path = std::env::temp_dir().join("paintfe-modal-load-fixture.png");
    app.io_sender
        .send(IoResult::ImageLoaded {
            tiled: TiledImage::from_rgba_image(&pattern()),
            width: 32,
            height: 24,
            path: path.clone(),
            format: SaveFormat::Png,
        })
        .unwrap();
    app.pending_io_ops = 1;
    app.update_runtime_lifecycle_async(&ctx);
    assert_eq!(app.projects[app.active_project_index].id, original_id);
    assert_eq!(app.pending_io_ops, 1);
    app.active_dialog = ActiveDialog::None;
    app.update_runtime_lifecycle_async(&ctx);
    assert_eq!(
        app.projects[app.active_project_index].path.as_ref(),
        Some(&path)
    );
    assert_eq!(app.pending_io_ops, 0);
}

fn escape_frame(app: &mut PaintFEApp, ctx: &egui::Context) {
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
        |ui| {
            app.update_runtime_input(ui.ctx());
        },
    );
}

#[test]
fn escape_deselect_is_undoable_and_preference_can_disable_it() {
    for enabled in [true, false] {
        let mut app = app();
        app.settings.escape_deselect = enabled;
        app.tools_panel
            .change_tool(crate::components::tools::Tool::Brush);
        app.projects[0].canvas_state.selection_all = true;
        let ctx = context();
        escape_frame(&mut app, &ctx);
        assert_eq!(app.projects[0].canvas_state.has_selection(), !enabled);
        assert_eq!(app.projects[0].history.undo_count(), usize::from(enabled));
        if enabled {
            let project = &mut app.projects[0];
            project.history.undo(&mut project.canvas_state);
            assert!(project.canvas_state.has_selection());
        }
        assert!(!app.tools_panel.injected_escape_pressed);
    }
}

#[test]
fn escape_keeps_selection_when_a_tool_operation_or_dialog_has_priority() {
    for kind in ["shape", "text", "mesh", "liquify", "selection", "settings"] {
        let mut app = app();
        app.settings.escape_deselect = true;
        app.projects[0].canvas_state.selection_all = true;
        match kind {
            "shape" => app.tools_panel.shapes_state.is_drawing = true,
            "text" => app.tools_panel.text_state.is_editing = true,
            "mesh" => app.tools_panel.mesh_warp_state.is_active = true,
            "liquify" => app.tools_panel.liquify_state.is_active = true,
            "selection" => app.tools_panel.selection_state.dragging = true,
            "settings" => app.settings_window.open = true,
            _ => unreachable!(),
        }
        escape_frame(&mut app, &context());
        assert!(app.projects[0].canvas_state.has_selection(), "{kind}");
        assert_eq!(app.projects[0].history.undo_count(), 0, "{kind}");
    }
}

#[test]
fn escape_deselect_preserves_partial_mask_when_undone() {
    let mut app = app();
    app.settings.escape_deselect = true;
    let mask = GrayImage::from_fn(32, 24, |x, _| Luma([if x < 12 { 127 } else { 0 }]));
    app.projects[0].canvas_state.selection_mask = Some(mask.clone());
    escape_frame(&mut app, &context());
    assert!(!app.projects[0].canvas_state.has_selection());
    let project = &mut app.projects[0];
    project.history.undo(&mut project.canvas_state);
    assert_eq!(project.canvas_state.selection_mask, Some(mask));
}

#[test]
fn docked_options_reserve_canvas_space_and_floating_options_keep_margins() {
    let mut app = app();
    let ctx = context();
    app.settings.tool_shelf.left = 20.0;
    app.settings.tool_shelf.right = 30.0;
    app.settings.tool_shelf.link_sides = false;
    for docked in [false, true, false] {
        app.settings.tool_shelf.docked = docked;
        for _ in 0..3 {
            app.reset_ui_cursor_blocking_rects();
            let _ = ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(1000.0, 700.0),
                    )),
                    ..Default::default()
                },
                |ui| {
                    let available = ui.available_rect_before_wrap();
                    let ctx = ui.ctx().clone();
                    app.show_runtime_canvas_tail(&ctx, ui, false);
                    let bar = app.ui_cursor_blocking_rects[0].shrink(12.0);
                    let canvas = app.canvas.last_canvas_rect.unwrap();
                    if docked {
                        assert!(
                            canvas.top() >= bar.bottom() - 1.0,
                            "{canvas:?} overlaps {bar:?}"
                        );
                        assert!((bar.left() - available.left()).abs() <= 1.0);
                        assert!((bar.right() - available.right()).abs() <= 1.0);
                    } else {
                        assert!((bar.left() - available.left() - 20.0).abs() <= 1.0);
                        assert!((available.right() - bar.right() - 30.0).abs() <= 1.0);
                        assert!(canvas.top() < bar.top());
                    }
                },
            );
        }
    }
}

#[test]
fn escape_does_not_deselect_behind_popups_or_focused_text_fields() {
    for popup in [true, false] {
        let mut app = app();
        app.settings.escape_deselect = true;
        app.projects[0].canvas_state.selection_all = true;
        let ctx = context();
        let id = egui::Id::new("escape_owner");
        let mut text = String::from("123");
        let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
            if popup {
                egui::Popup::open_id(ui.ctx(), id);
            } else {
                ui.add(egui::TextEdit::singleline(&mut text).id(id))
                    .request_focus();
            }
            app.remember_keyboard_owners(ui.ctx());
        });
        escape_frame(&mut app, &ctx);
        assert!(
            app.projects[0].canvas_state.has_selection(),
            "popup={popup}"
        );
        assert_eq!(app.projects[0].history.undo_count(), 0);
    }
}

#[test]
fn hiding_tools_colors_reclaims_space_without_changing_tool_geometry() {
    let mut app = app();
    let ctx = context();
    let mut heights = Vec::new();
    for show_colors in [true, false] {
        let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
            ui.set_width(90.0);
            app.tools_panel.show_compact(
                ui,
                &app.assets,
                (egui::Color32::BLACK, egui::Color32::WHITE),
                &app.settings.keybindings,
                false,
                show_colors,
            );
            heights.push(ui.min_rect().height());
        });
    }
    assert!(heights[0] - heights[1] >= 50.0, "{heights:?}");
}

#[test]
fn escape_cancels_selection_transform_without_also_deselecting() {
    let mut app = app();
    app.settings.escape_deselect = true;
    let before = GrayImage::from_fn(32, 24, |x, _| Luma([if x < 12 { 255 } else { 0 }]));
    app.move_sel_start_mask = Some(before.clone());
    app.move_sel_dragging = true;
    app.projects[0].canvas_state.selection_mask = Some(GrayImage::from_pixel(32, 24, Luma([255])));
    escape_frame(&mut app, &context());
    assert_eq!(app.projects[0].canvas_state.selection_mask, Some(before));
    assert!(!app.move_sel_dragging);
    assert_eq!(app.projects[0].history.undo_count(), 0);
}

#[test]
fn escape_cancels_line_preview_before_deselecting() {
    let mut app = app();
    app.settings.escape_deselect = true;
    app.tools_panel
        .change_tool(crate::components::tools::Tool::Line);
    app.tools_panel.line_state.line_tool.stage = crate::components::tools::LineStage::Editing;
    app.projects[0].canvas_state.selection_all = true;
    let original = pixels(&app);
    escape_frame(&mut app, &context());
    assert_eq!(
        app.tools_panel.line_state.line_tool.stage,
        crate::components::tools::LineStage::Idle
    );
    assert!(app.projects[0].canvas_state.has_selection());
    assert!(!app.tools_panel.line_state.line_tool.require_mouse_release);
    assert_eq!(pixels(&app), original);
    assert_eq!(app.projects[0].history.undo_count(), 0);
}
