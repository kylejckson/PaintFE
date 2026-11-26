use super::history_tests::{app, context};
use super::*;

fn frame(
    app: &mut PaintFEApp,
    ctx: &egui::Context,
    events: Vec<egui::Event>,
    modifiers: egui::Modifiers,
) {
    let _ = ctx.run_ui(
        egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(1200., 900.),
            )),
            events,
            modifiers,
            ..Default::default()
        },
        |ui| {
            app.update_runtime_input(ctx);
            app.show_runtime_dialogs_menu(ctx, ui);
        },
    );
}
fn click(app: &mut PaintFEApp, ctx: &egui::Context, rect: egui::Rect) {
    for pressed in [true, false] {
        frame(
            app,
            ctx,
            vec![
                egui::Event::PointerMoved(rect.center()),
                egui::Event::PointerButton {
                    pos: rect.center(),
                    button: egui::PointerButton::Primary,
                    pressed,
                    modifiers: egui::Modifiers::NONE,
                },
            ],
            egui::Modifiers::NONE,
        );
    }
}
fn rect(ctx: &egui::Context, key: &str) -> egui::Rect {
    ctx.data(|d| d.get_temp::<egui::Rect>(egui::Id::new(key)))
        .unwrap_or_else(|| panic!("Missing production control: {key}"))
}
#[test]
fn toolbar_menu_and_shortcut_fit_the_same_document_in_both_themes() {
    for theme in [egui::Theme::Light, egui::Theme::Dark] {
        let mut app = app();
        let ctx = context();
        ctx.set_theme(theme);
        app.projects[0].canvas_state.width = 4000;
        app.projects[0].canvas_state.height = 3000;
        app.canvas.last_canvas_rect = Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1000., 600.),
        ));
        for _ in 0..3 {
            frame(&mut app, &ctx, vec![], egui::Modifiers::NONE);
        }
        let fit = rect(&ctx, "toolbar_fit_test_rect");
        click(&mut app, &ctx, fit);
        let expected = app.canvas.view_state();
        assert!((expected.0 - 0.192).abs() < 0.00001);
        assert_eq!(expected.1, egui::Vec2::ZERO);
        app.canvas.set_view_state(2., egui::vec2(200., 100.));
        let menu = rect(&ctx, "view_menu_test_rect");
        click(&mut app, &ctx, menu);
        frame(&mut app, &ctx, vec![], egui::Modifiers::NONE);
        let fit = rect(&ctx, "menu_fit_test_rect");
        click(&mut app, &ctx, fit);
        assert_eq!(app.canvas.view_state(), expected);
        app.canvas.set_view_state(3., egui::vec2(-100., 20.));
        frame(
            &mut app,
            &ctx,
            vec![egui::Event::Key {
                key: egui::Key::Num0,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: egui::Modifiers::CTRL,
            }],
            egui::Modifiers::CTRL,
        );
        assert_eq!(app.canvas.view_state(), expected);
        frame(
            &mut app,
            &ctx,
            vec![egui::Event::Key {
                key: egui::Key::Num1,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: egui::Modifiers::CTRL,
            }],
            egui::Modifiers::CTRL,
        );
        assert_eq!(app.canvas.view_state(), (1., egui::Vec2::ZERO));
    }
}

#[test]
fn inspector_press_over_canvas_stays_owned_by_ui() {
    let mut app = app();
    let ctx = context();
    let id = app.projects[0].id;
    ctx.data_mut(|d| d.insert_temp(egui::Id::new("document_info"), id));
    for _ in 0..3 {
        frame(&mut app, &ctx, vec![], egui::Modifiers::NONE);
    }
    let inspector = *app.ui_cursor_blocking_rects_next.last().unwrap();
    let start = inspector.center();
    let mut captured = false;
    for (at, pressed) in [
        (start, Some(true)),
        (egui::pos2(1180., 880.), None),
        (egui::pos2(1180., 880.), Some(false)),
    ] {
        let mut events = vec![egui::Event::PointerMoved(at)];
        if let Some(pressed) = pressed {
            events.push(egui::Event::PointerButton {
                pos: at,
                button: egui::PointerButton::Primary,
                pressed,
                modifiers: egui::Modifiers::NONE,
            });
        }
        let _ = ctx.run_ui(
            egui::RawInput {
                events,
                ..Default::default()
            },
            |_| {
                captured = app.update_ui_pointer_capture(&ctx);
            },
        );
        if pressed != Some(false) {
            assert!(captured);
        }
    }
    assert!(!captured);
}

#[test]
fn selection_modifiers_each_undo_and_redo_the_mask() {
    use crate::canvas::SelectionShape;
    use crate::components::tools::SelectionModifyOp;
    let mut app = app();
    let ctx = context();
    app.projects[0].canvas_state.apply_selection_shape(
        &SelectionShape::Rectangle {
            min_x: 8,
            min_y: 7,
            max_x: 22,
            max_y: 18,
        },
        crate::canvas::SelectionMode::Replace,
    );
    for operation in [
        SelectionModifyOp::Feather(2.0),
        SelectionModifyOp::Expand(2),
        SelectionModifyOp::Contract(2),
    ] {
        let before = app.projects[0].canvas_state.selection_mask.clone();
        let count = app.projects[0].history.undo_count();
        app.tools_panel.pending_sel_modify = Some(operation);
        let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
            app.show_runtime_canvas_tail(&ctx, ui, false);
        });
        let project = &mut app.projects[0];
        let after = project.canvas_state.selection_mask.clone();
        assert_ne!(before, after);
        assert_eq!(project.history.undo_count(), count + 1);
        project.history.undo(&mut project.canvas_state).unwrap();
        assert_eq!(project.canvas_state.selection_mask, before);
        project.history.redo(&mut project.canvas_state).unwrap();
        assert_eq!(project.canvas_state.selection_mask, after);
    }
}

#[test]
fn pen_contact_keeps_moving_and_ignores_emulated_mouse_and_other_contacts() {
    let mut app = app();
    let tools = &mut app.tools_panel;
    tools.active_tool = crate::components::tools::Tool::Brush;
    let (a, b) = (egui::pos2(2., 3.), egui::pos2(18., 6.));
    let pen = |id, phase, pos, force| egui::Event::Touch {
        device_id: egui::TouchDeviceId(1),
        id: egui::TouchId(id),
        phase,
        pos,
        force,
    };
    let mut samples = Vec::new();
    let first = tools
        .track_paint_contact(
            &[pen(1, egui::TouchPhase::Start, a, Some(0.25))],
            |_| true,
            &mut samples,
        )
        .unwrap();
    assert!(first.started && !first.ended);
    assert_eq!(first.force, Some(0.25));
    let moved = tools
        .track_paint_contact(
            &[
                egui::Event::PointerMoved(egui::pos2(90., 90.)),
                pen(2, egui::TouchPhase::Start, egui::pos2(80., 80.), Some(1.0)),
                pen(1, egui::TouchPhase::Move, b, Some(0.75)),
            ],
            |_| true,
            &mut samples,
        )
        .unwrap();
    assert_eq!(samples, [a, b]);
    assert_eq!(moved.pos, b);
    assert_eq!(moved.force, Some(0.75));
    assert!(!moved.started && !moved.ended);
    assert!(
        tools
            .track_paint_contact(&[], |_| true, &mut samples)
            .is_some()
    );
    assert!(
        tools
            .track_paint_contact(
                &[pen(1, egui::TouchPhase::Cancel, egui::Pos2::ZERO, None)],
                |_| true,
                &mut samples,
            )
            .unwrap()
            .ended
    );
    assert_eq!(samples, [a, b]);
    assert!(
        tools
            .track_paint_contact(&[], |_| true, &mut samples)
            .is_none()
    );
}

#[test]
fn rectangle_drag_bounds_clip_to_inclusive_canvas_pixels() {
    use crate::components::tools::SelectionToolState;
    let mut state = SelectionToolState {
        drag_start: Some(egui::pos2(7.15, 3.2)),
        drag_end: Some(egui::pos2(7.88, 3.7)),
        ..Default::default()
    };
    assert_eq!(state.clipped_bounds(8, 4), Some((7, 3, 7, 3)));
    state.drag_end = Some(egui::pos2(-5.0, -3.0));
    assert_eq!(state.clipped_bounds(8, 4), Some((0, 0, 7, 3)));
    state.drag_start = Some(egui::pos2(-1.0, -1.0));
    assert_eq!(state.clipped_bounds(8, 4), None);
}

fn editor_frame(
    app: &mut PaintFEApp,
    ctx: &egui::Context,
    events: Vec<egui::Event>,
) -> egui::FullOutput {
    ctx.run_ui(
        egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(1200., 900.),
            )),
            events,
            ..Default::default()
        },
        |ui| {
            let modal = app.update_runtime_input(ctx);
            app.show_runtime_dialogs_menu(ctx, ui);
            app.show_runtime_canvas_tail(ctx, ui, modal);
        },
    )
}

fn primary_at(pos: egui::Pos2, pressed: bool) -> egui::Event {
    egui::Event::PointerButton {
        pos,
        button: egui::PointerButton::Primary,
        pressed,
        modifiers: egui::Modifiers::NONE,
    }
}

#[test]
fn lasso_empty_gestures_leave_history_and_subtracting_from_select_all_undoes_cleanly() {
    use crate::canvas::SelectionMode;
    use crate::components::tools::Tool;

    let mut app = app();
    let ctx = context();
    app.tools_panel.active_tool = Tool::Lasso;
    for _ in 0..3 {
        editor_frame(&mut app, &ctx, vec![]);
    }
    app.canvas.set_view_state(8., egui::Vec2::ZERO);
    editor_frame(&mut app, &ctx, vec![]);
    let rect = app.canvas.last_image_rect.unwrap();
    let at = |x, y| rect.min + egui::vec2(x, y) * 8.;
    let button_at = |pos, pressed, button| egui::Event::PointerButton {
        pos,
        button,
        pressed,
        modifiers: egui::Modifiers::NONE,
    };
    let primary = egui::PointerButton::Primary;
    let secondary = egui::PointerButton::Secondary;
    app.projects[0].canvas_state.selection_all = true;
    let baseline = app.projects[0].history.undo_count();
    let start = at(6.05, 6.05);
    let drag = |app: &mut PaintFEApp, button, points: &[egui::Pos2]| {
        editor_frame(
            app,
            &ctx,
            vec![
                egui::Event::PointerMoved(points[0]),
                button_at(points[0], true, button),
            ],
        );
        for &point in &points[1..] {
            editor_frame(app, &ctx, vec![egui::Event::PointerMoved(point)]);
        }
        let end = *points.last().unwrap();
        editor_frame(app, &ctx, vec![button_at(end, false, button)]);
    };
    drag(&mut app, secondary, &[start]);
    for mode in [
        SelectionMode::Add,
        SelectionMode::Subtract,
        SelectionMode::Intersect,
    ] {
        app.tools_panel.lasso_state.mode = mode;
        drag(&mut app, primary, &[start]);
        drag(&mut app, primary, &[start, at(6.40, 6.05), at(6.05, 6.40)]);
    }
    assert!(app.projects[0].canvas_state.selection_all);
    assert_eq!(app.projects[0].history.undo_count(), baseline);

    drag(
        &mut app,
        secondary,
        &[start, at(10., 6.), at(10., 10.), at(6., 10.)],
    );
    let project = &mut app.projects[0];
    assert!(!project.canvas_state.selection_all);
    let mask = project.canvas_state.selection_mask.as_ref().unwrap();
    assert_eq!(mask.get_pixel(7, 7).0[0], 0);
    assert_eq!(mask.get_pixel(3, 3).0[0], 255);
    assert_eq!(project.history.undo_count(), baseline + 1);
    assert!(project.history.undo(&mut project.canvas_state).is_some());
    assert!(project.canvas_state.selection_all);
    assert!(project.history.redo(&mut project.canvas_state).is_some());
    assert!(!project.canvas_state.selection_all);
}

#[test]
fn lasso_draws_the_closing_segment_for_both_mouse_buttons() {
    use crate::components::tools::Tool;

    let mut app = app();
    let ctx = context();
    app.tools_panel.active_tool = Tool::Lasso;
    for _ in 0..3 {
        editor_frame(&mut app, &ctx, vec![]);
    }
    app.canvas.set_view_state(8., egui::Vec2::ZERO);
    editor_frame(&mut app, &ctx, vec![]);
    let top_left = app.canvas.last_image_rect.unwrap().min;
    let start = top_left + egui::vec2(48., 40.);
    let bend = top_left + egui::vec2(88., 88.);
    let end = top_left + egui::vec2(120., 48.);
    for button in [egui::PointerButton::Primary, egui::PointerButton::Secondary] {
        let button_at = |pos, pressed| egui::Event::PointerButton {
            pos,
            button,
            pressed,
            modifiers: egui::Modifiers::NONE,
        };
        editor_frame(
            &mut app,
            &ctx,
            vec![egui::Event::PointerMoved(start), button_at(start, true)],
        );
        editor_frame(&mut app, &ctx, vec![egui::Event::PointerMoved(bend)]);
        let frame = editor_frame(&mut app, &ctx, vec![egui::Event::PointerMoved(end)]);
        assert!(
            frame.shapes.iter().any(|clipped| {
                matches!(&clipped.shape, egui::Shape::LineSegment { points, stroke }
                if stroke.width == 0.75
                    && points[0].distance(end) < 0.5
                    && points[1].distance(start) < 0.5)
            }),
            "missing closing preview for {button:?}"
        );
        editor_frame(&mut app, &ctx, vec![button_at(end, false)]);
    }
}

#[test]
fn rectangle_one_pixel_and_move_selection_edge_keep_exact_bounds() {
    let mut app = app();
    let ctx = context();
    use crate::components::tools::Tool;
    app.tools_panel.active_tool = Tool::RectangleSelect;
    for _ in 0..3 {
        editor_frame(&mut app, &ctx, vec![]);
    }
    app.canvas.set_view_state(8., egui::Vec2::ZERO);
    editor_frame(&mut app, &ctx, vec![]);
    let rect = app.canvas.last_image_rect.unwrap();
    let at = |x, y| rect.min + egui::vec2(x, y) * 8.;
    let press = at(31.15, 23.15);
    let end = at(31.8, 23.85);
    editor_frame(
        &mut app,
        &ctx,
        vec![egui::Event::PointerMoved(press), primary_at(press, true)],
    );
    editor_frame(&mut app, &ctx, vec![egui::Event::PointerMoved(end)]);
    editor_frame(&mut app, &ctx, vec![primary_at(end, false)]);
    assert_eq!(
        app.projects[0].canvas_state.selection_mask_bounds(),
        Some((31, 23, 31, 23))
    );

    let state = &mut app.projects[0].canvas_state;
    state.apply_selection_shape(
        &crate::canvas::SelectionShape::Rectangle {
            min_x: 4,
            min_y: 5,
            max_x: 31,
            max_y: 20,
        },
        crate::canvas::SelectionMode::Replace,
    );
    app.tools_panel.active_tool = Tool::MoveSelection;
    editor_frame(&mut app, &ctx, vec![]);
    let grab = at(31.7, 13.);
    editor_frame(
        &mut app,
        &ctx,
        vec![egui::Event::PointerMoved(grab), primary_at(grab, true)],
    );
    assert_eq!(
        app.projects[0]
            .canvas_state
            .selection_transform_preview_bounds,
        Some((4, 5, 31, 20))
    );
    let resized = at(28.7, 13.);
    editor_frame(&mut app, &ctx, vec![egui::Event::PointerMoved(resized)]);
    assert_eq!(
        app.projects[0]
            .canvas_state
            .selection_transform_preview_bounds,
        Some((4, 5, 28, 20))
    );
    editor_frame(&mut app, &ctx, vec![primary_at(resized, false)]);
    assert_eq!(
        app.projects[0].canvas_state.selection_mask_bounds(),
        Some((4, 5, 28, 20))
    );
}

#[test]
fn tablet_touch_only_moves_reach_the_real_brush_and_commit_once() {
    let mut app = app();
    let ctx = context();
    app.tools_panel.active_tool = crate::components::tools::Tool::Brush;
    app.tools_panel.properties.size = 2.;
    let before = app.projects[0].canvas_state.layers[0]
        .pixels
        .to_rgba_image();
    for _ in 0..3 {
        editor_frame(&mut app, &ctx, vec![]);
    }
    app.canvas.set_view_state(8., egui::Vec2::ZERO);
    editor_frame(&mut app, &ctx, vec![]);
    let rect = app.canvas.last_image_rect.unwrap();
    let at = |x, y| rect.min + egui::vec2(x, y) * 8.;
    let pen = |phase, pos, force| egui::Event::Touch {
        device_id: egui::TouchDeviceId(1),
        id: egui::TouchId(7),
        phase,
        pos,
        force: Some(force),
    };
    let (start, corner, end) = (at(4.5, 4.5), at(18.5, 4.5), at(18.5, 16.5));
    editor_frame(
        &mut app,
        &ctx,
        vec![
            egui::Event::PointerMoved(start),
            primary_at(start, true),
            pen(egui::TouchPhase::Start, start, 0.65),
        ],
    );
    editor_frame(
        &mut app,
        &ctx,
        vec![pen(egui::TouchPhase::Move, corner, 1.)],
    );
    editor_frame(&mut app, &ctx, vec![pen(egui::TouchPhase::Move, end, 1.)]);
    editor_frame(
        &mut app,
        &ctx,
        vec![
            primary_at(start, false),
            pen(egui::TouchPhase::End, end, 0.),
        ],
    );
    let project = &mut app.projects[0];
    let pixels = project.canvas_state.layers[0].pixels.to_rgba_image();
    assert_ne!(pixels.get_pixel(12, 4), before.get_pixel(12, 4));
    assert_ne!(pixels.get_pixel(18, 12), before.get_pixel(18, 12));
    assert_eq!(pixels.get_pixel(11, 10), before.get_pixel(11, 10));
    assert_eq!(project.history.undo_count(), 1);
    project.history.undo(&mut project.canvas_state).unwrap();
    assert_eq!(
        project.canvas_state.layers[0].pixels.to_rgba_image(),
        before
    );
}

#[test]
fn duplicate_layer_dispatch_preserves_mask_and_supports_undo() {
    use crate::config::{
        commands::MenuCommand,
        keybindings::{BindableAction, KeyCombo},
    };
    let mut app = app();
    let ctx = context();
    let action = BindableAction::Menu(MenuCommand::DuplicateLayer);
    let keys = crate::config::keybindings::KeyBindings::default();
    assert_eq!(
        keys.get(action),
        Some(&KeyCombo::ctrl_shift_key(egui::Key::D))
    );
    assert_eq!(
        keys.get(BindableAction::Menu(MenuCommand::EditPasteAsLayer)),
        Some(&KeyCombo::ctrl_shift_key(egui::Key::V))
    );
    let layer = &mut app.projects[0].canvas_state.layers[0];
    layer.ensure_mask();
    layer.opacity = 0.5;
    layer.blend_mode = crate::canvas::BlendMode::Multiply;
    let pixels = layer.pixels.to_rgba_image();
    app.settings.keybindings = keys;
    let modifiers = egui::Modifiers::CTRL | egui::Modifiers::SHIFT;
    frame(
        &mut app,
        &ctx,
        vec![egui::Event::Key {
            key: egui::Key::D,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers,
        }],
        modifiers,
    );
    let project = &mut app.projects[0];
    assert_eq!(project.canvas_state.layers.len(), 2);
    let duplicate = &project.canvas_state.layers[1];
    assert_eq!(duplicate.pixels.to_rgba_image(), pixels);
    assert!(duplicate.mask.is_some());
    assert_eq!(duplicate.opacity, 0.5);
    project.history.undo(&mut project.canvas_state).unwrap();
    assert_eq!(project.canvas_state.layers.len(), 1);
    project.history.redo(&mut project.canvas_state).unwrap();
    assert_eq!(project.canvas_state.layers.len(), 2);
    assert_eq!(
        project.canvas_state.layers[1].blend_mode,
        crate::canvas::BlendMode::Multiply
    );
}

#[test]
fn shifted_crop_does_not_fall_through_to_legacy_cut() {
    let mut app = app();
    let ctx = context();
    let canvas = &mut app.projects[0].canvas_state;
    let source = *canvas.layers[0].pixels.get_pixel(4, 3);
    canvas.apply_selection_shape(
        &crate::canvas::SelectionShape::Rectangle {
            min_x: 4,
            min_y: 3,
            max_x: 14,
            max_y: 13,
        },
        crate::canvas::SelectionMode::Replace,
    );
    let modifiers = egui::Modifiers::CTRL | egui::Modifiers::SHIFT;
    frame(
        &mut app,
        &ctx,
        vec![
            egui::Event::Key {
                key: egui::Key::X,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers,
            },
            egui::Event::Cut,
        ],
        modifiers,
    );
    let project = &app.projects[0];
    assert_eq!(
        (project.canvas_state.width, project.canvas_state.height),
        (11, 11)
    );
    assert_eq!(
        *project.canvas_state.layers[0].pixels.get_pixel(0, 0),
        source
    );
    assert_eq!(project.history.undo_count(), 1);
}

#[test]
fn shifted_paste_creates_one_layer_without_legacy_paste_overlay() {
    let mut app = app();
    let ctx = context();
    let modifiers = egui::Modifiers::CTRL | egui::Modifiers::SHIFT;
    crate::ops::clipboard::test_with_internal_image(
        image::RgbaImage::from_pixel(2, 2, image::Rgba([10, 20, 30, 255])),
        || {
            frame(
                &mut app,
                &ctx,
                vec![
                    egui::Event::Key {
                        key: egui::Key::V,
                        physical_key: None,
                        pressed: true,
                        repeat: false,
                        modifiers,
                    },
                    egui::Event::Paste(String::new()),
                ],
                modifiers,
            );
            frame(
                &mut app,
                &ctx,
                vec![egui::Event::Key {
                    key: egui::Key::V,
                    physical_key: None,
                    pressed: false,
                    repeat: false,
                    modifiers,
                }],
                modifiers,
            );
        },
    );
    assert_eq!(app.projects[0].canvas_state.layers.len(), 2);
    assert_eq!(app.projects[0].history.undo_count(), 1);
    assert!(app.paste_overlay.is_none());
    assert!(app.pending_paste_request.is_none());
}

#[test]
fn rectangle_drag_keeps_the_press_origin_when_motion_and_release_share_a_frame() {
    let mut app = app();
    let ctx = context();
    app.tools_panel.active_tool = crate::components::tools::Tool::RectangleSelect;
    for _ in 0..3 {
        editor_frame(&mut app, &ctx, vec![]);
    }
    app.canvas.set_view_state(8., egui::Vec2::ZERO);
    editor_frame(&mut app, &ctx, vec![]);
    let rect = app.canvas.last_image_rect.unwrap();
    let start = rect.min + egui::vec2(4.5, 3.5) * 8.;
    let end = rect.min + egui::vec2(13.5, 12.5) * 8.;
    editor_frame(
        &mut app,
        &ctx,
        vec![
            egui::Event::PointerMoved(start),
            primary_at(start, true),
            egui::Event::PointerMoved(end),
            primary_at(end, false),
        ],
    );
    assert_eq!(
        app.projects[0].canvas_state.selection_mask_bounds(),
        Some((4, 3, 13, 12))
    );
}
