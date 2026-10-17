use super::*;
use crate::components::tools::Tool;
use crate::ui::dialogs::*;
use image::{GrayImage, Luma, Rgba};

pub(super) fn app() -> PaintFEApp {
    crate::i18n::init();
    let ctx = egui::Context::default();
    let cc = eframe::CreationContext::_new_kittest(ctx);
    let mut app = PaintFEApp::new(&cc, Vec::new(), mpsc::channel().1);
    app.projects = vec![Project::new_untitled(1, 32, 24)];
    app.active_project_index = 0;
    app.projects[0].canvas_state.layers[0].pixels = TiledImage::from_rgba_image(&pattern());
    app
}

pub(super) fn context() -> egui::Context {
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

fn wait_jobs(app: &mut PaintFEApp, ctx: &egui::Context) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
    while app.pending_filter_jobs > 0 {
        assert!(
            std::time::Instant::now() < deadline,
            "background History job timed out"
        );
        app.update_runtime_lifecycle_async(ctx);
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
}

fn dialog_frame(app: &mut PaintFEApp, ctx: &egui::Context, key: Option<egui::Key>) {
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
        |ui| app.process_active_dialog(ui.ctx()),
    );
    wait_jobs(app, ctx);
}

fn assert_history_roundtrip(app: &mut PaintFEApp, expected: HistoryAction) {
    let project = &mut app.projects[0];
    let entries = project.history.presentation_history();
    let latest = entries
        .first()
        .unwrap_or_else(|| panic!("no History entry for {expected:?}"));
    assert_eq!(latest.action, expected);
    assert_eq!(latest.icon, expected.icon());
    let pixels = project.canvas_state.layers[0].pixels.to_rgba_image();
    let mask = project.canvas_state.selection_mask.clone();
    let all = project.canvas_state.selection_all;
    let layer_mask = project.canvas_state.layers[0]
        .mask
        .as_ref()
        .map(TiledImage::to_rgba_image);
    project.history.undo(&mut project.canvas_state);
    project.history.redo(&mut project.canvas_state);
    assert_eq!(project.history.presentation_history()[0].action, expected);
    assert_eq!(
        project.canvas_state.layers[0].pixels.to_rgba_image(),
        pixels
    );
    assert_eq!(project.canvas_state.selection_mask, mask);
    assert_eq!(project.canvas_state.selection_all, all);
    assert_eq!(
        project.canvas_state.layers[0]
            .mask
            .as_ref()
            .map(TiledImage::to_rgba_image),
        layer_mask
    );
    eprintln!("HISTORY_COMMAND\t{expected:?}\t{:?}", latest.icon);
}

#[test]
fn selection_actions_restore_partial_masks_and_semantic_select_all() {
    let mut app = app();
    let original = pattern();
    for all in [false, true] {
        let before =
            (!all).then(|| GrayImage::from_fn(32, 24, |x, y| Luma([((x * 7 + y) % 256) as u8])));
        app.projects[0].canvas_state.selection_mask = before.clone();
        app.projects[0].canvas_state.selection_all = all;
        app.invert_selection_with_history();
        assert_history_roundtrip(&mut app, HistoryAction::InvertSelection);
        app.perform_undo();
        assert_eq!(app.projects[0].canvas_state.selection_mask, before);
        assert_eq!(app.projects[0].canvas_state.selection_all, all);
        app.clear_selection_with_history();
        assert_history_roundtrip(&mut app, HistoryAction::Deselect);
        app.perform_undo();
        assert_eq!(app.projects[0].canvas_state.selection_mask, before);
        assert_eq!(app.projects[0].canvas_state.selection_all, all);
        assert_eq!(
            app.projects[0].canvas_state.layers[0]
                .pixels
                .to_rgba_image(),
            original
        );
    }
}

#[test]
fn color_range_apply_and_cancel_preserve_all_selection_modes() {
    for mode in [
        crate::canvas::SelectionMode::Replace,
        crate::canvas::SelectionMode::Add,
        crate::canvas::SelectionMode::Subtract,
        crate::canvas::SelectionMode::Intersect,
    ] {
        for all in [false, true] {
            for cancel in [false, true] {
                let mut app = app();
                let ctx = context();
                let before = (!all).then(|| GrayImage::from_pixel(32, 24, Luma([100])));
                app.projects[0].canvas_state.selection_mask = before.clone();
                app.projects[0].canvas_state.selection_all = all;
                let mut dlg = ColorRangeDialog::new(&app.projects[0].canvas_state);
                dlg.mode = mode;
                app.preview_color_range(&dlg);
                let preview = app.projects[0].canvas_state.selection_mask.clone();
                app.preview_color_range(&dlg);
                assert_eq!(app.projects[0].canvas_state.selection_mask, preview);
                app.active_dialog = ActiveDialog::ColorRange(dlg);
                dialog_frame(&mut app, &ctx, None);
                dialog_frame(
                    &mut app,
                    &ctx,
                    Some(if cancel {
                        egui::Key::Escape
                    } else {
                        egui::Key::Enter
                    }),
                );
                assert!(app.active_dialog.is_none());
                if cancel {
                    assert_eq!(app.projects[0].history.undo_count(), 0);
                    assert_eq!(app.projects[0].canvas_state.selection_mask, before);
                    assert_eq!(app.projects[0].canvas_state.selection_all, all);
                } else {
                    assert_eq!(app.projects[0].canvas_state.selection_mask, preview);
                    assert_history_roundtrip(&mut app, HistoryAction::ColorRange);
                    app.perform_undo();
                    assert_eq!(app.projects[0].canvas_state.selection_mask, before);
                    assert_eq!(app.projects[0].canvas_state.selection_all, all);
                }
            }
        }
    }
}

#[test]
fn async_history_keeps_action_when_tool_caption_and_document_change() {
    for action in [
        HistoryAction::ColorRemover,
        HistoryAction::ContentAwareBrush,
        HistoryAction::RemoveBackground,
        HistoryAction::ExternalEffect,
    ] {
        let mut app = app();
        let ctx = context();
        let original = app.projects[0].canvas_state.layers[0].pixels.clone();
        app.spawn_filter_job(
            0.,
            action.label("Magic Wand / Clone / 任意"),
            0,
            original.clone(),
            pattern(),
            |img| {
                let mut out = img.clone();
                out.put_pixel(0, 0, Rgba([1, 2, 3, 255]));
                out
            },
        );
        app.tools_panel.active_tool = Tool::Pencil;
        app.projects.push(Project::new_untitled(2, 32, 24));
        app.active_project_index = 1;
        wait_jobs(&mut app, &ctx);
        assert_eq!(app.projects[1].history.undo_count(), 0);
        assert_history_roundtrip(&mut app, action);
    }
}

macro_rules! dialog_history_case {
    ($test:ident,$dialog:ident,$variant:ident,$prepare:expr) => {
        #[test]
        fn $test() {
            for preview in [false, true] {
                let mut app = app();
                let ctx = context();
                let state = &app.projects[0].canvas_state;
                let mut dlg = $dialog::new(state);
                ($prepare)(&mut dlg, preview);
                app.active_dialog = ActiveDialog::$variant(dlg);
                let expected = app.active_dialog.history_action().unwrap();
                dialog_frame(&mut app, &ctx, None);
                dialog_frame(&mut app, &ctx, Some(egui::Key::Enter));
                assert!(app.active_dialog.is_none(), "{expected:?} did not apply");
                assert_history_roundtrip(&mut app, expected);
            }
        }
    };
}

pub(super) fn tool_frame(
    app: &mut PaintFEApp,
    ctx: &egui::Context,
    point: (f32, f32),
    events: Vec<egui::Event>,
) {
    let rect = egui::Rect::from_min_size(egui::pos2(100., 100.), egui::vec2(128., 96.));
    if app.tools_panel.active_tool != Tool::Fill {
        app.canvas.last_canvas_rect = Some(rect);
        app.canvas.last_image_rect = Some(rect);
        app.canvas.zoom = 4.;
    }
    let _ = ctx.run_ui(
        egui::RawInput {
            events,
            ..Default::default()
        },
        |ui| {
            if matches!(
                app.tools_panel.active_tool,
                Tool::MovePixels | Tool::MoveSelection | Tool::Text
            ) {
                app.update_runtime_input(ui.ctx());
            }
            let tools = &mut app.tools_panel;
            let state = &mut app.projects[0].canvas_state;
            if tools.active_tool == Tool::Fill {
                app.canvas.show_with_state(
                    ui,
                    state,
                    Some(tools),
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
                    false,
                    false,
                    false,
                );
                return;
            }
            tools.handle_input(
                ui,
                state,
                Some((point.0 as u32, point.1 as u32)),
                Some(point),
                Some(point),
                Some(point),
                &[],
                ui.painter(),
                rect,
                egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1024., 768.)),
                4.,
                [0., 0., 0., 1.],
                [1., 1., 1., 1.],
                None,
                false,
            );
        },
    );
    app.commit_pending_tool_history();
    app.dispatch_pending_tool_jobs(ctx);
    wait_jobs(app, ctx);
}

fn tool_gesture(tool: Tool) {
    tool_gesture_on(tool, false);
}

fn tool_gesture_on(tool: Tool, mask: bool) {
    let mut app = app();
    let ctx = context();
    app.tools_panel.change_tool(tool);
    app.tools_panel.properties.size = 3.;
    app.tools_panel.rebuild_brush_lut();
    if mask {
        crate::ops::canvas_ops::add_layer_mask_reveal_all(&mut app.projects[0].canvas_state, 0);
        app.projects[0].canvas_state.edit_layer_mask = true;
    }
    if tool == Tool::CloneStamp {
        app.tools_panel.clone_stamp_state.source = Some(egui::pos2(4., 4.));
    }
    if matches!(tool, Tool::MovePixels | Tool::MoveSelection) {
        app.projects[0].canvas_state.selection_mask = Some(GrayImage::from_fn(32, 24, |x, y| {
            Luma([if (6..22).contains(&x) && (5..19).contains(&y) {
                255
            } else {
                0
            }])
        }));
    }
    tool_frame(&mut app, &ctx, (8., 8.), vec![]);
    let image_rect = if tool == Tool::Fill {
        app.canvas.last_image_rect.unwrap()
    } else {
        egui::Rect::from_min_size(egui::pos2(100., 100.), egui::vec2(128., 96.))
    };
    let screen = |p: (f32, f32)| {
        image_rect.min
            + egui::vec2(
                p.0 * image_rect.width() / 32.,
                p.1 * image_rect.height() / 24.,
            )
    };
    let button = |p, pressed| egui::Event::PointerButton {
        pos: screen(p),
        button: egui::PointerButton::Primary,
        pressed,
        modifiers: egui::Modifiers::NONE,
    };
    tool_frame(
        &mut app,
        &ctx,
        (8., 8.),
        vec![
            egui::Event::PointerMoved(screen((8., 8.))),
            button((8., 8.), true),
        ],
    );
    let click_only = matches!(
        tool,
        Tool::ColorRemover | Tool::Fill | Tool::MagicWand | Tool::Text | Tool::ColorPicker
    );
    if !click_only {
        for p in [(18., 8.), (18., 16.), (20., 16.)] {
            tool_frame(
                &mut app,
                &ctx,
                p,
                vec![egui::Event::PointerMoved(screen(p))],
            );
        }
    }
    let end = if click_only { (8., 8.) } else { (20., 16.) };
    tool_frame(&mut app, &ctx, end, vec![button(end, false)]);
    if tool == Tool::Text {
        tool_frame(
            &mut app,
            &ctx,
            (8., 16.),
            vec![egui::Event::Text("A".into())],
        );
    }
    // Let native asynchronous tool previews settle before committing.
    for _ in 0..5 {
        tool_frame(&mut app, &ctx, (8., 16.), vec![]);
        std::thread::sleep(std::time::Duration::from_millis(3));
    }
    if tool == Tool::MagicWand {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while app.tools_panel.magic_wand_state.computing
            || app.projects[0].canvas_state.selection_mask.is_none()
        {
            assert!(
                std::time::Instant::now() < deadline,
                "Magic Wand preview timed out"
            );
            tool_frame(&mut app, &ctx, end, vec![]);
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
    }
    if matches!(
        tool,
        Tool::MagicWand
            | Tool::Line
            | Tool::Gradient
            | Tool::Shapes
            | Tool::Liquify
            | Tool::MeshWarp
            | Tool::PerspectiveCrop
    ) {
        tool_frame(
            &mut app,
            &ctx,
            end,
            vec![egui::Event::Key {
                key: egui::Key::Enter,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: egui::Modifiers::NONE,
            }],
        );
    }
    for _ in 0..5 {
        tool_frame(&mut app, &ctx, (8., 16.), vec![]);
    }
    app.tools_panel
        .commit_active_tool_preview(&mut app.projects[0].canvas_state, [1., 1., 1., 1.]);
    app.commit_pending_tool_history();
    app.dispatch_pending_tool_jobs(&ctx);
    wait_jobs(&mut app, &ctx);
    if app.paste_overlay.is_some() {
        app.commit_paste_overlay(false);
    }
    if let Some(action) = tool.history_action() {
        let expected = if mask {
            match tool {
                Tool::Eraser => HistoryAction::MaskErase,
                Tool::Line => HistoryAction::MaskLine,
                _ => HistoryAction::MaskPaint,
            }
        } else {
            action
        };
        assert_history_roundtrip(&mut app, expected);
    } else {
        assert_eq!(
            app.projects[0].history.undo_count(),
            0,
            "navigation/sampling must not add History"
        );
    }
}

#[test]
fn mask_painting_keeps_the_painting_tool_identity() {
    for tool in [Tool::Brush, Tool::Pencil, Tool::Eraser, Tool::Line] {
        tool_gesture_on(tool, true);
    }
}

macro_rules! tool_history_case {
    ($test:ident,$tool:ident) => {
        #[test]
        fn $test() {
            tool_gesture(Tool::$tool);
        }
    };
}

#[test]
fn paste_transform_checkpoint_uses_move_pixels_icon_and_keeps_overlay_undo() {
    let mut app = app();
    let ctx = context();
    let mut overlay =
        PasteOverlay::new(RgbaImage::from_pixel(4, 4, Rgba([255, 0, 0, 255])), 32, 24);
    let before = overlay.transform();
    overlay.center.x += 2.;
    let after = overlay.transform();
    overlay.pending_transform_checkpoint = Some((before, after));
    app.paste_overlay = Some(overlay);
    let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
        app.show_runtime_canvas_tail(&ctx, ui, false)
    });
    assert_eq!(
        app.projects[0].history.presentation_history()[0].action,
        HistoryAction::MovePaste
    );
    app.perform_undo();
    assert_eq!(app.paste_overlay.as_ref().unwrap().center, before.center);
    let _ = ctx.run_ui(
        egui::RawInput {
            events: vec![egui::Event::Key {
                key: egui::Key::Y,
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
    assert_eq!(app.paste_overlay.as_ref().unwrap().center, after.center);
}

#[test]
fn layer_mask_actions_use_the_actual_layer_menu_dispatcher() {
    use crate::components::layers::LayerAppAction as A;
    for (action, expected) in [
        (A::FlipHorizontal, HistoryAction::FlipLayerH),
        (A::FlipVertical, HistoryAction::FlipLayerV),
        (A::AddLayerMaskRevealAll(0), HistoryAction::AddMask),
        (
            A::AddLayerMaskFromSelection(0),
            HistoryAction::MaskFromSelection,
        ),
        (A::ToggleLayerMask(0), HistoryAction::ToggleMask),
        (A::InvertLayerMask(0), HistoryAction::InvertMask),
        (A::ApplyLayerMask(0), HistoryAction::ApplyMask),
        (A::DeleteLayerMask(0), HistoryAction::DeleteMask),
    ] {
        let mut app = app();
        crate::ops::canvas_ops::add_layer_mask_reveal_all(&mut app.projects[0].canvas_state, 0);
        app.projects[0].canvas_state.selection_mask =
            Some(GrayImage::from_pixel(32, 24, Luma([127])));
        app.handle_layer_app_action(action);
        assert_history_roundtrip(&mut app, expected);
    }
}

dialog_history_case!(
    history_dialog_resizeimage,
    ResizeImageDialog,
    ResizeImage,
    |_d: &mut ResizeImageDialog, _p| {}
);
dialog_history_case!(
    history_dialog_resizecanvas,
    ResizeCanvasDialog,
    ResizeCanvas,
    |_d: &mut ResizeCanvasDialog, _p| {}
);
dialog_history_case!(
    history_dialog_alignlayer,
    AlignLayerDialog,
    AlignLayer,
    |d: &mut AlignLayerDialog, p| {
        d.live_preview = p;
    }
);
dialog_history_case!(
    history_dialog_gaussianblur,
    GaussianBlurDialog,
    GaussianBlur,
    |d: &mut GaussianBlurDialog, p| {
        d.live_preview = p;
    }
);
dialog_history_case!(
    history_dialog_layertransform,
    LayerTransformDialog,
    LayerTransform,
    |d: &mut LayerTransformDialog, p| {
        d.live_preview = p;
    }
);
dialog_history_case!(
    history_dialog_brightnesscontrast,
    BrightnessContrastDialog,
    BrightnessContrast,
    |d: &mut BrightnessContrastDialog, p| {
        d.live_preview = p;
    }
);
dialog_history_case!(
    history_dialog_huesaturation,
    HueSaturationDialog,
    HueSaturation,
    |d: &mut HueSaturationDialog, p| {
        d.live_preview = p;
        d.per_band = true;
    }
);
dialog_history_case!(
    history_dialog_exposure,
    ExposureDialog,
    Exposure,
    |d: &mut ExposureDialog, p| {
        d.live_preview = p;
    }
);
dialog_history_case!(
    history_dialog_highlightsshadows,
    HighlightsShadowsDialog,
    HighlightsShadows,
    |d: &mut HighlightsShadowsDialog, p| {
        d.live_preview = p;
    }
);
dialog_history_case!(
    history_dialog_levels,
    LevelsDialog,
    Levels,
    |d: &mut LevelsDialog, p| {
        d.live_preview = p;
    }
);
dialog_history_case!(
    history_dialog_curves,
    CurvesDialog,
    Curves,
    |d: &mut CurvesDialog, p| {
        d.live_preview = p;
    }
);
dialog_history_case!(
    history_dialog_temperaturetint,
    TemperatureTintDialog,
    TemperatureTint,
    |d: &mut TemperatureTintDialog, p| {
        d.live_preview = p;
    }
);
dialog_history_case!(
    history_dialog_bokehblur,
    BokehBlurDialog,
    BokehBlur,
    |d: &mut BokehBlurDialog, p| {
        d.original_flat = Some(pattern());
        d.pending_flat = None;
        d.live_preview = p;
    }
);
dialog_history_case!(
    history_dialog_motionblur,
    MotionBlurDialog,
    MotionBlur,
    |d: &mut MotionBlurDialog, p| {
        d.original_flat = Some(pattern());
        d.pending_flat = None;
        d.live_preview = p;
    }
);
dialog_history_case!(
    history_dialog_boxblur,
    BoxBlurDialog,
    BoxBlur,
    |d: &mut BoxBlurDialog, p| {
        d.original_flat = Some(pattern());
        d.pending_flat = None;
        d.live_preview = p;
    }
);
dialog_history_case!(
    history_dialog_zoomblur,
    ZoomBlurDialog,
    ZoomBlur,
    |d: &mut ZoomBlurDialog, p| {
        d.original_flat = Some(pattern());
        d.pending_flat = None;
        d.live_preview = p;
    }
);
dialog_history_case!(
    history_dialog_crystallize,
    CrystallizeDialog,
    Crystallize,
    |d: &mut CrystallizeDialog, p| {
        d.original_flat = Some(pattern());
        d.pending_flat = None;
        d.live_preview = p;
    }
);
dialog_history_case!(
    history_dialog_dents,
    DentsDialog,
    Dents,
    |d: &mut DentsDialog, p| {
        d.original_flat = Some(pattern());
        d.pending_flat = None;
        d.live_preview = p;
    }
);
dialog_history_case!(
    history_dialog_pixelate,
    PixelateDialog,
    Pixelate,
    |d: &mut PixelateDialog, p| {
        d.original_flat = Some(pattern());
        d.pending_flat = None;
        d.live_preview = p;
    }
);
dialog_history_case!(
    history_dialog_bulge,
    BulgeDialog,
    Bulge,
    |d: &mut BulgeDialog, p| {
        d.original_flat = Some(pattern());
        d.pending_flat = None;
        d.live_preview = p;
    }
);
dialog_history_case!(
    history_dialog_twist,
    TwistDialog,
    Twist,
    |d: &mut TwistDialog, p| {
        d.original_flat = Some(pattern());
        d.pending_flat = None;
        d.live_preview = p;
    }
);
dialog_history_case!(
    history_dialog_addnoise,
    AddNoiseDialog,
    AddNoise,
    |d: &mut AddNoiseDialog, p| {
        d.original_flat = Some(pattern());
        d.pending_flat = None;
        d.live_preview = p;
    }
);
dialog_history_case!(
    history_dialog_reducenoise,
    ReduceNoiseDialog,
    ReduceNoise,
    |d: &mut ReduceNoiseDialog, p| {
        d.original_flat = Some(pattern());
        d.pending_flat = None;
        d.live_preview = p;
    }
);
dialog_history_case!(
    history_dialog_median,
    MedianDialog,
    Median,
    |d: &mut MedianDialog, p| {
        d.original_flat = Some(pattern());
        d.pending_flat = None;
        d.live_preview = p;
    }
);
dialog_history_case!(
    history_dialog_glow,
    GlowDialog,
    Glow,
    |d: &mut GlowDialog, p| {
        d.original_flat = Some(pattern());
        d.pending_flat = None;
        d.live_preview = p;
    }
);
dialog_history_case!(
    history_dialog_sharpen,
    SharpenDialog,
    Sharpen,
    |d: &mut SharpenDialog, p| {
        d.original_flat = Some(pattern());
        d.pending_flat = None;
        d.live_preview = p;
    }
);
dialog_history_case!(
    history_dialog_vignette,
    VignetteDialog,
    Vignette,
    |d: &mut VignetteDialog, p| {
        d.original_flat = Some(pattern());
        d.pending_flat = None;
        d.live_preview = p;
    }
);
dialog_history_case!(
    history_dialog_halftone,
    HalftoneDialog,
    Halftone,
    |d: &mut HalftoneDialog, p| {
        d.original_flat = Some(pattern());
        d.pending_flat = None;
        d.live_preview = p;
    }
);
dialog_history_case!(
    history_dialog_grid,
    GridDialog,
    Grid,
    |d: &mut GridDialog, p| {
        d.original_flat = Some(pattern());
        d.pending_flat = None;
        d.live_preview = p;
    }
);
dialog_history_case!(
    history_dialog_dropshadow,
    DropShadowDialog,
    DropShadow,
    |d: &mut DropShadowDialog, p| {
        d.original_flat = Some(pattern());
        d.pending_flat = None;
        d.live_preview = p;
    }
);
dialog_history_case!(
    history_dialog_outline,
    OutlineDialog,
    Outline,
    |d: &mut OutlineDialog, p| {
        d.original_flat = Some(pattern());
        d.pending_flat = None;
        d.live_preview = p;
    }
);
dialog_history_case!(
    history_dialog_pixeldrag,
    PixelDragDialog,
    PixelDrag,
    |d: &mut PixelDragDialog, p| {
        d.original_flat = Some(pattern());
        d.pending_flat = None;
        d.live_preview = p;
    }
);
dialog_history_case!(
    history_dialog_rgbdisplace,
    RgbDisplaceDialog,
    RgbDisplace,
    |d: &mut RgbDisplaceDialog, p| {
        d.original_flat = Some(pattern());
        d.pending_flat = None;
        d.live_preview = p;
    }
);
dialog_history_case!(
    history_dialog_ink,
    InkDialog,
    Ink,
    |d: &mut InkDialog, p| {
        d.original_flat = Some(pattern());
        d.pending_flat = None;
        d.live_preview = p;
    }
);
dialog_history_case!(
    history_dialog_oilpainting,
    OilPaintingDialog,
    OilPainting,
    |d: &mut OilPaintingDialog, p| {
        d.original_flat = Some(pattern());
        d.pending_flat = None;
        d.live_preview = p;
    }
);
dialog_history_case!(
    history_dialog_colorfilter,
    ColorFilterDialog,
    ColorFilter,
    |d: &mut ColorFilterDialog, p| {
        d.original_flat = Some(pattern());
        d.pending_flat = None;
        d.live_preview = p;
    }
);
dialog_history_case!(
    history_dialog_colortoalpha,
    ColorToAlphaDialog,
    ColorToAlpha,
    |d: &mut ColorToAlphaDialog, p| {
        d.original_flat = Some(pattern());
        d.pending_flat = None;
        d.live_preview = p;
    }
);
dialog_history_case!(
    history_dialog_recovertransparency,
    RecoverTransparencyDialog,
    RecoverTransparency,
    |d: &mut RecoverTransparencyDialog, p| {
        d.original_flat = Some(pattern());
        d.pending_flat = None;
        d.live_preview = p;
    }
);
dialog_history_case!(
    history_dialog_canvasborder,
    CanvasBorderDialog,
    CanvasBorder,
    |d: &mut CanvasBorderDialog, p| {
        d.original_flat = Some(pattern());
        d.pending_flat = None;
        d.live_preview = p;
    }
);
dialog_history_case!(
    history_dialog_seamlesstexture,
    SeamlessTextureDialog,
    SeamlessTexture,
    |d: &mut SeamlessTextureDialog, p| {
        d.original_flat = Some(pattern());
        d.pending_flat = None;
        d.live_preview = p;
    }
);
dialog_history_case!(
    history_dialog_contours,
    ContoursDialog,
    Contours,
    |d: &mut ContoursDialog, p| {
        d.original_flat = Some(pattern());
        d.pending_flat = None;
        d.live_preview = p;
    }
);
dialog_history_case!(
    history_dialog_threshold,
    ThresholdDialog,
    Threshold,
    |d: &mut ThresholdDialog, p| {
        d.live_preview = p;
    }
);
dialog_history_case!(
    history_dialog_posterize,
    PosterizeDialog,
    Posterize,
    |d: &mut PosterizeDialog, p| {
        d.live_preview = p;
    }
);
dialog_history_case!(
    history_dialog_colorbalance,
    ColorBalanceDialog,
    ColorBalance,
    |d: &mut ColorBalanceDialog, p| {
        d.live_preview = p;
    }
);
dialog_history_case!(
    history_dialog_gradientmap,
    GradientMapDialog,
    GradientMap,
    |d: &mut GradientMapDialog, p| {
        d.live_preview = p;
    }
);
dialog_history_case!(
    history_dialog_blackandwhite,
    BlackAndWhiteDialog,
    BlackAndWhite,
    |d: &mut BlackAndWhiteDialog, p| {
        d.live_preview = p;
    }
);
dialog_history_case!(
    history_dialog_vibrance,
    VibranceDialog,
    Vibrance,
    |d: &mut VibranceDialog, p| {
        d.live_preview = p;
    }
);

tool_history_case!(history_tool_brush, Brush);

tool_history_case!(history_tool_eraser, Eraser);

tool_history_case!(history_tool_pencil, Pencil);

tool_history_case!(history_tool_line, Line);

tool_history_case!(history_tool_rectangleselect, RectangleSelect);

tool_history_case!(history_tool_ellipseselect, EllipseSelect);

tool_history_case!(history_tool_movepixels, MovePixels);

tool_history_case!(history_tool_moveselection, MoveSelection);

tool_history_case!(history_tool_magicwand, MagicWand);

tool_history_case!(history_tool_fill, Fill);

tool_history_case!(history_tool_colorpicker, ColorPicker);

tool_history_case!(history_tool_gradient, Gradient);

tool_history_case!(history_tool_contentawarebrush, ContentAwareBrush);

tool_history_case!(history_tool_liquify, Liquify);

tool_history_case!(history_tool_meshwarp, MeshWarp);

tool_history_case!(history_tool_colorremover, ColorRemover);

tool_history_case!(history_tool_smudge, Smudge);

tool_history_case!(history_tool_clonestamp, CloneStamp);

tool_history_case!(history_tool_text, Text);

tool_history_case!(history_tool_perspectivecrop, PerspectiveCrop);

tool_history_case!(history_tool_lasso, Lasso);

tool_history_case!(history_tool_zoom, Zoom);

tool_history_case!(history_tool_pan, Pan);

tool_history_case!(history_tool_shapes, Shapes);

#[test]
fn immediate_adjustments_and_selection_shortcuts_keep_their_identity() {
    use crate::config::keybindings::{BindableAction, KeyCombo};
    for (binding, action) in [
        (BindableAction::ColorAutoLevels, HistoryAction::AutoLevels),
        (BindableAction::ColorDesaturate, HistoryAction::Desaturate),
        (
            BindableAction::ColorInvertColors,
            HistoryAction::InvertColors,
        ),
        (BindableAction::ColorInvertAlpha, HistoryAction::InvertAlpha),
        (BindableAction::ColorSepiaTone, HistoryAction::Sepia),
        (
            BindableAction::ClearSelectedPixels,
            HistoryAction::DeleteSelection,
        ),
        (
            BindableAction::FillSelectedPixels,
            HistoryAction::FillSelection,
        ),
        (BindableAction::Deselect, HistoryAction::Deselect),
    ] {
        let mut app = app();
        let ctx = context();
        app.projects[0].canvas_state.selection_mask = Some(GrayImage::from_fn(32, 24, |x, y| {
            Luma([if x < 16 && y < 12 { 255 } else { 0 }])
        }));
        app.settings
            .keybindings
            .set(binding, KeyCombo::key(egui::Key::F12));
        let _ = ctx.run_ui(
            egui::RawInput {
                events: vec![egui::Event::Key {
                    key: egui::Key::F12,
                    physical_key: None,
                    pressed: true,
                    repeat: false,
                    modifiers: egui::Modifiers::NONE,
                }],
                ..Default::default()
            },
            |ui| {
                let response = ui.interact(
                    ui.max_rect(),
                    egui::Id::new("history-test-canvas"),
                    egui::Sense::click(),
                );
                app.canvas.canvas_widget_id = Some(response.id);
                response.request_focus();
                app.update_runtime_input(ui.ctx());
            },
        );
        wait_jobs(&mut app, &ctx);
        assert_history_roundtrip(&mut app, action);
    }
}

#[test]
fn straighten_parameterised_caption_keeps_the_straighten_icon() {
    let mut app = app();
    app.start_straighten();
    app.straighten_session.as_mut().unwrap().angle_degrees = 12.5;
    app.commit_straighten();
    assert_history_roundtrip(&mut app, HistoryAction::Straighten);
}

#[test]
fn color_range_apply_without_touching_controls_creates_one_undo_entry() {
    let mut app = app();
    let ctx = context();
    app.active_dialog =
        ActiveDialog::ColorRange(ColorRangeDialog::new(&app.projects[0].canvas_state));
    dialog_frame(&mut app, &ctx, None);
    dialog_frame(&mut app, &ctx, Some(egui::Key::Enter));
    assert!(app.active_dialog.is_none());
    assert_eq!(app.projects[0].history.undo_count(), 1);
    assert_history_roundtrip(&mut app, HistoryAction::ColorRange);
    app.perform_undo();
    assert!(app.projects[0].canvas_state.selection_mask.is_none());
    assert!(!app.projects[0].canvas_state.selection_all);
}
