use egui::{Color32, Context, Event, RawInput, Rect, pos2, vec2};
use paintfe::{
    assets::AppSettings,
    canvas::{Canvas, CanvasState},
    components::tools::{Tool, ToolsPanel},
};
fn raw(events: Vec<Event>) -> RawInput {
    RawInput {
        screen_rect: Some(Rect::from_min_size(pos2(0., 0.), vec2(640., 480.))),
        events,
        ..Default::default()
    }
}
fn input(ui: &egui::Ui, tools: &mut ToolsPanel, state: &mut CanvasState) {
    let rect = Rect::from_min_size(pos2(0., 0.), vec2(640., 480.));
    tools.handle_input(
        ui,
        state,
        Some((20, 20)),
        Some((20., 20.)),
        Some((20., 20.)),
        Some((20., 20.)),
        &[],
        ui.painter(),
        rect,
        rect,
        1.,
        [0., 0., 0., 1.],
        [1., 1., 1., 1.],
        None,
        false,
    );
}
fn visible(state: &CanvasState) -> usize {
    state
        .preview_layer
        .as_ref()
        .unwrap()
        .to_rgba_image()
        .pixels()
        .filter(|p| p[3] > 0)
        .count()
}
#[test]
fn text_preview_uploads_after_second_letter() {
    let ctx = Context::default();
    let mut tools = ToolsPanel::default();
    tools.active_tool = Tool::Text;
    let mut state = CanvasState::new(400, 200);
    let mut canvas = Canvas::new_without_state();
    let _ = ctx.run_ui(raw(vec![]), |ui| input(ui, &mut tools, &mut state));
    tools.text_state.is_editing = true;
    tools.text_state.origin = Some([20., 20.]);
    tools.text_state.font_family = "Arial".into();
    tools.stroke_tracker.uses_preview_layer = true;
    let first = ctx.run_ui(raw(vec![Event::Text("H".into())]), |ui| {
        input(ui, &mut tools, &mut state);
        canvas.show_with_state(
            ui,
            &mut state,
            Some(&mut tools),
            [0., 0., 0., 1.],
            [1., 1., 1., 1.],
            Color32::GRAY,
            None,
            None,
            false,
            &AppSettings::default(),
            0,
            0,
            Color32::BLUE,
            None,
            None,
            "",
            true,
            true,
            false,
        )
    });
    println!(
        "FIRST text={} editing={} preview={} font={}",
        tools.text_state.text,
        tools.text_state.is_editing,
        state.preview_layer.is_some(),
        tools.text_state.loaded_font.is_some()
    );
    let first_pixels = visible(&state);
    let tiles = state.preview_tile_textures.len();
    assert!(tiles > 0, "expected tiled preview path");
    assert!(state.preview_texture_cache.is_none());
    let second = ctx.run_ui(raw(vec![Event::Text("e".into())]), |ui| {
        input(ui, &mut tools, &mut state);
        canvas.show_with_state(
            ui,
            &mut state,
            Some(&mut tools),
            [0., 0., 0., 1.],
            [1., 1., 1., 1.],
            Color32::GRAY,
            None,
            None,
            false,
            &AppSettings::default(),
            0,
            0,
            Color32::BLUE,
            None,
            None,
            "",
            true,
            true,
            false,
        )
    });
    let second_pixels = visible(&state);
    let uploaded = second
        .textures_delta
        .set
        .iter()
        .filter(|(id, _)| state.preview_tile_textures.values().any(|t| t.id() == *id))
        .count();
    println!(
        "OBSERVED text={} first_pixels={first_pixels} second_pixels={second_pixels} tiles={tiles} second_tile_uploads={uploaded} initial_uploads={}",
        tools.text_state.text,
        first.textures_delta.set.len()
    );
    assert_eq!(tools.text_state.text, "He");
    assert!(second_pixels > first_pixels);
    assert!(uploaded > 0, "each edit must upload the tiled preview");
    let deletion = ctx.run_ui(
        raw(vec![Event::Key {
            key: egui::Key::Backspace,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: Default::default(),
        }]),
        |ui| {
            input(ui, &mut tools, &mut state);
            canvas.show_with_state(
                ui,
                &mut state,
                Some(&mut tools),
                [0., 0., 0., 1.],
                [1., 1., 1., 1.],
                Color32::GRAY,
                None,
                None,
                false,
                &AppSettings::default(),
                0,
                0,
                Color32::BLUE,
                None,
                None,
                "",
                true,
                true,
                false,
            );
        },
    );
    assert_eq!(tools.text_state.text, "H");
    assert_eq!(visible(&state), first_pixels);
    assert!(deletion.textures_delta.set.iter().any(|(id, _)| {
        state
            .preview_tile_textures
            .values()
            .any(|texture| texture.id() == *id)
    }));
    tools.text_state.origin = Some([200., 50.]);
    tools.text_state.preview_dirty = true;
    canvas.zoom = 0.5;
    let _ = ctx.run_ui(raw(vec![]), |ui| {
        input(ui, &mut tools, &mut state);
        canvas.show_with_state(
            ui,
            &mut state,
            Some(&mut tools),
            [0., 0., 0., 1.],
            [1., 1., 1., 1.],
            Color32::GRAY,
            None,
            None,
            false,
            &AppSettings::default(),
            0,
            0,
            Color32::BLUE,
            None,
            None,
            "",
            true,
            true,
            false,
        );
    });
    let preview = state.preview_layer.as_ref().unwrap();
    assert!(
        (0..100).all(|x| (0..100).all(|y| preview.get_pixel(x, y)[3] == 0)),
        "moving text must remove old glyphs"
    );
    tools.properties.blending_mode = paintfe::canvas::BlendMode::Multiply;
    tools.text_state.preview_dirty = true;
    canvas.zoom = 2.5;
    let _ = ctx.run_ui(raw(vec![Event::Text("i".into())]), |ui| {
        input(ui, &mut tools, &mut state);
        canvas.show_with_state(
            ui,
            &mut state,
            Some(&mut tools),
            [0., 0., 0., 1.],
            [1., 1., 1., 1.],
            Color32::GRAY,
            None,
            None,
            false,
            &AppSettings::default(),
            0,
            0,
            Color32::BLUE,
            None,
            None,
            "",
            true,
            true,
            false,
        );
    });
    assert_eq!(tools.text_state.text, "Hi");
    assert!(state.preview_force_composite);
}
fn draw(
    ctx: &Context,
    canvas: &mut Canvas,
    state: &mut CanvasState,
    tools: &mut ToolsPanel,
    events: Vec<Event>,
    blocked: bool,
) {
    let settings = AppSettings::default();
    let _ = ctx.run_ui(raw(events), |ui| {
        canvas.show_with_state(
            ui,
            state,
            Some(tools),
            [0., 0., 0., 1.],
            [1., 1., 1., 1.],
            Color32::GRAY,
            None,
            None,
            false,
            &settings,
            0,
            0,
            Color32::BLUE,
            None,
            None,
            "",
            blocked,
            blocked,
            false,
        )
    });
}
fn mask_sum(state: &CanvasState) -> u64 {
    state
        .selection_mask
        .as_ref()
        .map(|m| m.as_raw().iter().map(|v| *v as u64).sum())
        .unwrap_or(0)
}
#[test]
fn magic_wand_refreshes_while_controls_own_pointer() {
    let ctx = Context::default();
    let mut canvas = Canvas::new_without_state();
    let mut state = CanvasState::new(64, 64);
    let mut tools = ToolsPanel::default();
    tools.active_tool = Tool::MagicWand;
    tools.magic_wand_state.tolerance = 0.;
    for y in 0..64 {
        for x in 0..64 {
            let v = if x < 32 { 255 } else { 120 };
            state.layers[0]
                .pixels
                .put_pixel(x, y, image::Rgba([v, v, v, 255]));
        }
    }
    state.mark_dirty(None);
    draw(&ctx, &mut canvas, &mut state, &mut tools, vec![], false);
    let rect = canvas.last_image_rect.unwrap();
    let p = pos2(rect.left() + rect.width() * 0.25, rect.center().y);
    draw(
        &ctx,
        &mut canvas,
        &mut state,
        &mut tools,
        vec![
            Event::PointerMoved(p),
            Event::PointerButton {
                pos: p,
                button: egui::PointerButton::Primary,
                pressed: true,
                modifiers: Default::default(),
            },
        ],
        false,
    );
    draw(
        &ctx,
        &mut canvas,
        &mut state,
        &mut tools,
        vec![Event::PointerButton {
            pos: p,
            button: egui::PointerButton::Primary,
            pressed: false,
            modifiers: Default::default(),
        }],
        false,
    );
    for _ in 0..50 {
        if mask_sum(&state) > 0 && !tools.magic_wand_state.computing {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
        draw(&ctx, &mut canvas, &mut state, &mut tools, vec![], false);
    }
    let before = mask_sum(&state);
    assert!(before > 0);
    tools.magic_wand_state.anti_aliased = !tools.magic_wand_state.anti_aliased;
    draw(&ctx, &mut canvas, &mut state, &mut tools, vec![], true);
    assert_eq!(
        tools.magic_wand_state.last_applied_aa,
        tools.magic_wand_state.anti_aliased
    );
    tools.magic_wand_state.tolerance = 100.;
    for _ in 0..3 {
        draw(&ctx, &mut canvas, &mut state, &mut tools, vec![], true);
    }
    let blocked = mask_sum(&state);
    draw(&ctx, &mut canvas, &mut state, &mut tools, vec![], false);
    let after = mask_sum(&state);
    println!("OBSERVED Magic Wand before={before} over_controls={blocked} back_on_canvas={after}");
    assert!(
        blocked > before,
        "controls must refresh the selection immediately"
    );
    assert_eq!(blocked, after);
}

use egui::{Key, Modifiers};
use paintfe::assets::{BindableAction, KeyBindings};
fn event(pressed: bool) -> Event {
    Event::Key {
        key: Key::Z,
        physical_key: Some(Key::Z),
        pressed,
        repeat: false,
        modifiers: Modifiers::CTRL,
    }
}
fn frame(ctx: &Context, kb: &KeyBindings, events: Vec<Event>, check: bool) -> (bool, usize) {
    let mut result = (false, 0);
    let _ = ctx.run_ui(
        RawInput {
            modifiers: Modifiers::CTRL,
            events,
            ..Default::default()
        },
        |ui| {
            if check {
                result.0 = kb.is_pressed(ui.ctx(), BindableAction::Undo);
            }
            result.1 = ui.input(|i| {
                i.events
                    .iter()
                    .filter(|e| {
                        matches!(
                            e,
                            Event::Key {
                                key: Key::Z,
                                pressed: true,
                                ..
                            }
                        )
                    })
                    .count()
            });
        },
    );
    result
}
#[test]
fn shortcut_consumes_one_press_without_held_retrigger() {
    let ctx = Context::default();
    let kb = KeyBindings::default();
    paintfe::windows_key_probe::observe_windows_message(0x0100, 0x11);
    paintfe::windows_key_probe::observe_windows_message(0x0101, 0x5a);
    paintfe::windows_key_probe::observe_windows_message(0x0100, 0x5a);
    let first = frame(&ctx, &kb, vec![event(true)], true);
    let second = frame(&ctx, &kb, vec![], true);
    let third = frame(&ctx, &kb, vec![], true);
    paintfe::windows_key_probe::observe_windows_message(0x0101, 0x5a);
    paintfe::windows_key_probe::observe_windows_message(0x0101, 0x11);
    println!("OBSERVED one held Ctrl+Z: first={first:?}, second={second:?}, third={third:?}");
    assert!(first.0 && !second.0 && !third.0);
    assert_eq!(first.1, 0, "shortcut events must be consumed");
}
#[test]
#[cfg(target_os = "windows")]
fn shortcut_observes_release_when_shortcuts_are_skipped() {
    let ctx = Context::default();
    let kb = KeyBindings::default();
    paintfe::windows_key_probe::observe_windows_message(0x0100, 0x11);
    paintfe::windows_key_probe::observe_windows_message(0x0101, 0x5a);
    paintfe::windows_key_probe::observe_windows_message(0x0100, 0x5a);
    assert!(frame(&ctx, &kb, vec![event(true)], true).0);
    paintfe::windows_key_probe::observe_windows_message(0x0101, 0x5a);
    frame(&ctx, &kb, vec![event(false)], false);
    paintfe::windows_key_probe::observe_windows_message(0x0100, 0x5a);
    let missed = frame(&ctx, &kb, vec![], true);
    paintfe::windows_key_probe::observe_windows_message(0x0101, 0x5a);
    paintfe::windows_key_probe::observe_windows_message(0x0101, 0x11);
    println!("OBSERVED Windows probe press after unpolled release: {missed:?}");
    assert!(missed.0, "the next Windows press must trigger");
}

#[test]
fn ten_separate_undo_presses_each_trigger_once() {
    let ctx = Context::default();
    let kb = KeyBindings::default();
    paintfe::windows_key_probe::observe_windows_message(0x0100, 0x11);
    for _ in 0..10 {
        paintfe::windows_key_probe::observe_windows_message(0x0101, 0x5a);
        frame(&ctx, &kb, vec![event(false)], true);
        paintfe::windows_key_probe::observe_windows_message(0x0100, 0x5a);
        assert_eq!(frame(&ctx, &kb, vec![event(true)], true), (true, 0));
        assert!(!frame(&ctx, &kb, vec![], true).0);
    }
    paintfe::windows_key_probe::observe_windows_message(0x0101, 0x5a);
    paintfe::windows_key_probe::observe_windows_message(0x0101, 0x11);
}

#[test]
#[cfg(target_os = "windows")]
fn released_native_shortcut_retains_its_press_modifiers() {
    let ctx = Context::default();
    let kb = KeyBindings::default();
    // Establish a baseline so earlier native presses cannot affect this context.
    kb.discard_pending_presses(&ctx);
    paintfe::windows_key_probe::observe_windows_message(0x0100, 0x11);
    paintfe::windows_key_probe::observe_windows_message(0x0100, 0x5a);
    paintfe::windows_key_probe::observe_windows_message(0x0101, 0x5a);
    paintfe::windows_key_probe::observe_windows_message(0x0101, 0x11);
    let mut triggered = false;
    let _ = ctx.run_ui(RawInput::default(), |ui| {
        triggered = kb.is_pressed(ui.ctx(), BindableAction::Undo);
    });
    assert!(triggered, "a tap released before the frame must still undo");
    let _ = ctx.run_ui(RawInput::default(), |ui| {
        assert!(!kb.is_pressed(ui.ctx(), BindableAction::Undo));
    });
    paintfe::windows_key_probe::observe_windows_message(0x0100, 0x5a);
    paintfe::windows_key_probe::observe_windows_message(0x0101, 0x5a);
    let _ = ctx.run_ui(RawInput::default(), |ui| {
        assert!(
            !kb.is_pressed(ui.ctx(), BindableAction::Undo),
            "plain Z must not undo"
        );
    });
}
