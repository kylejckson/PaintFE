use super::*;
use egui::{Event, Key, Modifiers, Response};

fn key(key: Key, modifiers: Modifiers) -> Event {
    Event::Key {
        key,
        physical_key: None,
        pressed: true,
        repeat: false,
        modifiers,
    }
}
fn pointer(pos: egui::Pos2, pressed: bool) -> Event {
    Event::PointerButton {
        pos,
        button: egui::PointerButton::Primary,
        pressed,
        modifiers: Modifiers::NONE,
    }
}

/// The same controls inside a panel, a labelled grid and a popup-style frame.
fn frame(
    ctx: &egui::Context,
    value: &mut f32,
    colour: bool,
    place: usize,
    enabled: bool,
    input: egui::RawInput,
) -> Response {
    let mut result = None;
    let _ = ctx.run_ui(input, |ui| {
        ui.set_max_width(420.0);
        let mut draw = |ui: &mut egui::Ui| {
            if !enabled {
                ui.disable();
            }
            result = Some(if colour {
                ui.add(
                    GradientSlider::new(value, 100.0, |t| {
                        egui::Color32::from_gray((t * 255.0) as u8)
                    })
                    .id_salt("sample")
                    .label("A")
                    .alpha(true)
                    .theme(if ui.visuals().dark_mode {
                        crate::theme::Theme::dark()
                    } else {
                        crate::theme::Theme::light()
                    }),
                )
            } else {
                ui.add(
                    Slider::new(value, 0.0..=100.0)
                        .id_salt("sample")
                        .step_by(1.0)
                        .suffix("%")
                        .max_decimals(0),
                )
            });
        };
        match place {
            0 => draw(ui),
            1 => {
                egui::Grid::new("settings").show(ui, |ui| {
                    ui.label("Opacity");
                    draw(ui);
                    ui.end_row();
                });
            }
            3 => {
                without_wheel_adjustment(ui, draw);
            }
            _ => {
                egui::Frame::popup(ui.style()).show(ui, draw);
            }
        }
    });
    result.unwrap()
}
fn parts(ctx: &egui::Context, name: &str) -> Response {
    ctx.data(|d| d.get_temp::<Response>(egui::Id::new(name)))
        .unwrap()
}
fn input(events: Vec<Event>) -> egui::RawInput {
    egui::RawInput {
        events,
        ..Default::default()
    }
}

#[test]
fn double_click_reset_is_track_only_and_respects_policy_and_enabled_state() {
    for colour in [false, true] {
        for enabled in [false, true] {
            for policy in [false, true] {
                let ctx = egui::Context::default();
                ctx.data_mut(|d| {
                    d.insert_temp(
                        egui::Id::new("paintfe_polish"),
                        crate::ui::polish::PolishSettings {
                            double_click_slider_reset: policy,
                            ..Default::default()
                        },
                    )
                });
                let mut value = 45.0f32;
                let mut tick = 0;
                let mut run = |value: &mut f32, events: Vec<Event>| {
                    tick += 1;
                    let _ = ctx.run_ui(
                        egui::RawInput {
                            time: Some(tick as f64 * 0.04),
                            events,
                            ..Default::default()
                        },
                        |ui| {
                            if !enabled {
                                ui.disable();
                            }
                            if colour {
                                ui.add(
                                    GradientSlider::new(value, 100.0, |_| egui::Color32::RED)
                                        .reset_to(20.0),
                                );
                            } else {
                                ui.add(Slider::new(value, 0.0..=100.0).reset_to(20.0));
                            }
                        },
                    );
                };
                run(&mut value, vec![]);
                run(&mut value, vec![]);
                let pos = parts(&ctx, "control_test_track").rect.center();
                for _ in 0..2 {
                    run(
                        &mut value,
                        vec![Event::PointerMoved(pos), pointer(pos, true)],
                    );
                    run(&mut value, vec![pointer(pos, false)]);
                }
                if enabled && policy {
                    assert_eq!(value, 20.0, "colour={colour}");
                } else if enabled {
                    assert!(value > 40.0);
                } else {
                    assert_eq!(value, 45.0);
                }
            }
        }
    }
}

#[test]
fn double_clicking_number_edits_text_instead_of_resetting_slider() {
    for colour in [false, true] {
        let ctx = egui::Context::default();
        let mut value = 45.0f32;
        let mut tick = 0;
        let mut run = |value: &mut f32, events: Vec<Event>| {
            tick += 1;
            let _ = ctx.run_ui(
                egui::RawInput {
                    time: Some(tick as f64 * 0.04),
                    events,
                    ..Default::default()
                },
                |ui| {
                    if colour {
                        ui.add(
                            GradientSlider::new(value, 100.0, |_| egui::Color32::RED)
                                .reset_to(20.0),
                        );
                    } else {
                        ui.add(Slider::new(value, 0.0..=100.0).reset_to(20.0));
                    }
                },
            );
        };
        run(&mut value, vec![]);
        run(&mut value, vec![]);
        let pos = parts(&ctx, "control_test_editor").rect.center();
        for _ in 0..2 {
            run(
                &mut value,
                vec![Event::PointerMoved(pos), pointer(pos, true)],
            );
            run(&mut value, vec![pointer(pos, false)]);
        }
        assert_eq!(value, 45.0);
    }
}

#[test]
fn double_click_override_reset_returns_to_inherited_value() {
    let ctx = egui::Context::default();
    let mut value = Some(0.8f32);
    let mut tick = 0;
    let mut run = |value: &mut Option<f32>, events: Vec<Event>| {
        tick += 1;
        let _ = ctx.run_ui(
            egui::RawInput {
                time: Some(tick as f64 * 0.04),
                events,
                ..Default::default()
            },
            |ui| {
                let mut inherit = false;
                if let Some(number) = value {
                    ui.add(Slider::new(number, 0.0..=1.0).on_double_click_reset(|| inherit = true));
                }
                if inherit {
                    *value = None;
                }
            },
        );
    };
    run(&mut value, vec![]);
    run(&mut value, vec![]);
    let pos = parts(&ctx, "control_test_track").rect.center();
    for _ in 0..2 {
        run(
            &mut value,
            vec![Event::PointerMoved(pos), pointer(pos, true)],
        );
        run(&mut value, vec![pointer(pos, false)]);
    }
    assert_eq!(value, None);
}

#[test]
fn scrolling_scope_preserves_wheel_and_keyboard_for_both_control_styles() {
    for colour in [false, true] {
        for numeric in [false, true] {
            let ctx = egui::Context::default();
            let mut value = 50.0;
            for _ in 0..3 {
                frame(&ctx, &mut value, colour, 3, true, input(vec![]));
            }
            let part = parts(
                &ctx,
                if numeric {
                    "control_test_editor"
                } else {
                    "control_test_track"
                },
            );
            let wheel = || Event::MouseWheel {
                phase: egui::TouchPhase::Move,
                unit: egui::MouseWheelUnit::Line,
                delta: egui::vec2(0.0, -1.0),
                modifiers: Modifiers::NONE,
            };
            frame(
                &ctx,
                &mut value,
                colour,
                3,
                true,
                input(vec![Event::PointerMoved(part.rect.center()), wheel()]),
            );
            assert_eq!(value, 50.0);
            assert!(ctx.input(|i| {
                i.events
                    .iter()
                    .any(|e| matches!(e, Event::MouseWheel { .. }))
            }));
            assert!(crate::ui::polish::settings(&ctx).numeric_wheel);
            if !numeric {
                ctx.memory_mut(|m| m.request_focus(part.id));
                frame(
                    &ctx,
                    &mut value,
                    colour,
                    3,
                    true,
                    input(vec![key(Key::ArrowRight, Modifiers::NONE)]),
                );
                assert_eq!(value, 51.0, "scroll policy disabled keyboard input");
            }
            // The policy belongs to the parent UI, not a global preference.
            frame(&ctx, &mut value, colour, 0, true, input(vec![]));
            let part = parts(&ctx, "control_test_track");
            let before = value;
            frame(
                &ctx,
                &mut value,
                colour,
                0,
                true,
                input(vec![Event::PointerMoved(part.rect.center()), wheel()]),
            );
            assert_eq!(value, before - 1.0);
        }
    }
}

#[test]
fn standard_and_colour_sliders_share_keyboard_wheel_and_disabled_contract_in_three_places() {
    for dark in [false, true] {
        for colour in [false, true] {
            for place in 0..3 {
                let ctx = egui::Context::default();
                ctx.set_visuals(if dark {
                    egui::Visuals::dark()
                } else {
                    egui::Visuals::light()
                });
                let mut value = 50.0;
                frame(&ctx, &mut value, colour, place, true, input(vec![]));
                let track = parts(&ctx, "control_test_track");
                ctx.memory_mut(|m| m.request_focus(track.id));
                frame(&ctx, &mut value, colour, place, true, input(vec![]));
                let changed = frame(
                    &ctx,
                    &mut value,
                    colour,
                    place,
                    true,
                    input(vec![key(Key::ArrowRight, Modifiers::NONE)]),
                );
                assert_eq!(
                    value, 51.0,
                    "arrow applied twice or lost: colour={colour}, place={place}"
                );
                assert!(changed.changed());
                frame(
                    &ctx,
                    &mut value,
                    colour,
                    place,
                    true,
                    egui::RawInput {
                        modifiers: Modifiers::SHIFT,
                        events: vec![key(Key::ArrowUp, Modifiers::SHIFT)],
                        ..Default::default()
                    },
                );
                assert!(
                    (value - 51.1).abs() < 0.001,
                    "fine step: value={value}, colour={colour}, place={place}"
                );
                frame(
                    &ctx,
                    &mut value,
                    colour,
                    place,
                    true,
                    input(vec![key(Key::End, Modifiers::NONE)]),
                );
                assert_eq!(value, 100.0);
                frame(
                    &ctx,
                    &mut value,
                    colour,
                    place,
                    true,
                    input(vec![key(Key::Home, Modifiers::NONE)]),
                );
                assert_eq!(value, 0.0);
                let wheel = || Event::MouseWheel {
                    phase: egui::TouchPhase::Move,
                    unit: egui::MouseWheelUnit::Line,
                    delta: egui::vec2(0.0, 1.0),
                    modifiers: Modifiers::NONE,
                };
                frame(
                    &ctx,
                    &mut value,
                    colour,
                    place,
                    true,
                    input(vec![Event::PointerMoved(track.rect.center()), wheel()]),
                );
                assert_eq!(value, 1.0);
                assert_eq!(
                    ctx.input(|i| i.smooth_scroll_delta.y),
                    0.0,
                    "wheel also scrolled container"
                );
                frame(
                    &ctx,
                    &mut value,
                    colour,
                    place,
                    false,
                    input(vec![wheel(), key(Key::End, Modifiers::NONE)]),
                );
                assert_eq!(value, 1.0, "disabled control changed");
            }
        }
    }
}

#[test]
fn standard_and_colour_tracks_drag_and_release_outside_in_both_themes() {
    for dark in [false, true] {
        for colour in [false, true] {
            let ctx = egui::Context::default();
            ctx.set_visuals(if dark {
                egui::Visuals::dark()
            } else {
                egui::Visuals::light()
            });
            let mut value = 50.0;
            for _ in 0..2 {
                frame(&ctx, &mut value, colour, 0, true, input(vec![]));
            }
            let track = parts(&ctx, "control_test_track").rect;
            let start = track.center();
            let target = egui::pos2(track.left() + track.width() * 0.75, track.center().y);
            frame(
                &ctx,
                &mut value,
                colour,
                0,
                true,
                input(vec![Event::PointerMoved(start), pointer(start, true)]),
            );
            frame(
                &ctx,
                &mut value,
                colour,
                0,
                true,
                input(vec![Event::PointerMoved(target)]),
            );
            assert!(
                value > 65.0 && value < 85.0,
                "drag failed for colour={colour}: {value}"
            );
            frame(
                &ctx,
                &mut value,
                colour,
                0,
                true,
                input(vec![
                    Event::PointerMoved(track.right_center() + egui::vec2(70.0, 0.0)),
                    pointer(track.right_center() + egui::vec2(70.0, 0.0), false),
                ]),
            );
            frame(&ctx, &mut value, colour, 0, true, input(vec![]));
            let released = value;
            frame(
                &ctx,
                &mut value,
                colour,
                0,
                true,
                input(vec![Event::PointerMoved(start)]),
            );
            assert_eq!(value, released, "capture remained after release");
            assert!((0.0..=100.0).contains(&value));
        }
    }
}

#[test]
fn both_slider_styles_use_the_shared_editor_for_typing_cancel_and_tab_commit() {
    for colour in [false, true] {
        for place in 0..3 {
            let ctx = egui::Context::default();
            let mut value = 50.0;
            frame(&ctx, &mut value, colour, place, true, input(vec![]));
            let editor = parts(&ctx, "control_test_editor");
            ctx.memory_mut(|m| m.request_focus(editor.id));
            frame(&ctx, &mut value, colour, place, true, input(vec![]));
            let select_all = || {
                key(
                    Key::A,
                    Modifiers {
                        ctrl: true,
                        command: true,
                        ..Default::default()
                    },
                )
            };
            frame(
                &ctx,
                &mut value,
                colour,
                place,
                true,
                input(vec![select_all(), Event::Text("72".into())]),
            );
            frame(
                &ctx,
                &mut value,
                colour,
                place,
                true,
                input(vec![key(Key::Escape, Modifiers::NONE)]),
            );
            assert_eq!(
                value, 50.0,
                "Escape failed for colour={colour}, place={place}"
            );
            ctx.memory_mut(|m| m.request_focus(editor.id));
            frame(&ctx, &mut value, colour, place, true, input(vec![]));
            frame(
                &ctx,
                &mut value,
                colour,
                place,
                true,
                input(vec![select_all(), Event::Text("72".into())]),
            );
            frame(
                &ctx,
                &mut value,
                colour,
                place,
                true,
                input(vec![key(Key::Tab, Modifiers::NONE)]),
            );
            assert_eq!(value, 72.0);
        }
    }
}

#[test]
fn explicit_numeric_identity_survives_sibling_insertions_and_reordering() {
    let ctx = egui::Context::default();
    let mut first = 1.0;
    let mut second = 2.0;
    let mut ids = Vec::new();
    for reverse in [false, true] {
        let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
            if reverse {
                ui.label("An unrelated new setting");
            }
            for name in if reverse {
                ["second", "first"]
            } else {
                ["first", "second"]
            } {
                let value = if name == "first" {
                    &mut first
                } else {
                    &mut second
                };
                let response = ui.add(Numeric::new(value).id_salt(name));
                ids.push((name, response.id));
            }
        });
    }
    assert_eq!(ids[0].1, ids[3].1);
    assert_eq!(ids[1].1, ids[2].1);
    assert_ne!(ids[0].1, ids[1].1);
}

#[test]
fn integer_steppers_saturate_and_reset_without_overflow() {
    let ctx = egui::Context::default();
    let mut value = u32::MAX;
    fn draw(ctx: &egui::Context, value: &mut u32, events: Vec<Event>) {
        let _ = ctx.run_ui(input(events), |ui| {
            ui.add(
                Numeric::new(value)
                    .id_salt("integer")
                    .range(0..=u32::MAX)
                    .steppers(1.0)
                    .reset_to(42),
            );
        });
    }
    draw(&ctx, &mut value, vec![]);
    let plus = parts(&ctx, "control_test_plus").rect.center();
    draw(
        &ctx,
        &mut value,
        vec![Event::PointerMoved(plus), pointer(plus, true)],
    );
    draw(&ctx, &mut value, vec![pointer(plus, false)]);
    assert_eq!(value, u32::MAX, "increment overflowed");
    let reset = parts(&ctx, "control_test_reset").rect.center();
    draw(
        &ctx,
        &mut value,
        vec![Event::PointerMoved(reset), pointer(reset, true)],
    );
    draw(&ctx, &mut value, vec![pointer(reset, false)]);
    assert_eq!(value, 42);
    value = 0;
    draw(&ctx, &mut value, vec![]);
    let minus = parts(&ctx, "control_test_minus").rect.center();
    draw(
        &ctx,
        &mut value,
        vec![Event::PointerMoved(minus), pointer(minus, true)],
    );
    draw(&ctx, &mut value, vec![pointer(minus, false)]);
    assert_eq!(value, 0, "decrement underflowed");
}

#[test]
fn rendering_preserves_f64_precision_and_values_between_slider_steps() {
    for explicit_step in [false, true] {
        let ctx = egui::Context::default();
        let mut value = 0.123_456_789_012_345_f64;
        let original = value;
        for _ in 0..3 {
            let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
                let mut slider = Slider::new(&mut value, 0.0..=1.0).max_decimals(3);
                if explicit_step {
                    slider = slider.step_by(0.01);
                }
                let response = ui.add(slider);
                assert!(!response.changed(), "rendering must not edit the model");
            });
        }
        assert_eq!(value, original);
    }
}

#[test]
fn formatted_fraction_uses_shared_typing_and_wheel_can_be_disabled() {
    let ctx = egui::Context::default();
    let mut value = 0.5_f64;
    let draw = |value: &mut f64, events| {
        let _ = ctx.run_ui(input(events), |ui| {
            ui.add(
                Slider::new(value, 0.0..=1.0)
                    .custom_formatter(|value, _| format!("{:.0}%", value * 100.0))
                    .custom_parser(|text| {
                        text.trim_end_matches('%')
                            .parse::<f64>()
                            .ok()
                            .map(|value| value / 100.0)
                    }),
            );
        });
    };
    draw(&mut value, vec![]);
    let editor = parts(&ctx, "control_test_editor");
    ctx.memory_mut(|m| m.request_focus(editor.id));
    draw(&mut value, vec![]);
    draw(
        &mut value,
        vec![
            key(
                Key::A,
                Modifiers {
                    ctrl: true,
                    command: true,
                    ..Default::default()
                },
            ),
            Event::Text("25%".into()),
        ],
    );
    draw(&mut value, vec![key(Key::Tab, Modifiers::NONE)]);
    assert_eq!(value, 0.25);
    ctx.data_mut(|d| {
        d.insert_temp(
            egui::Id::new("paintfe_polish"),
            crate::ui::polish::PolishSettings {
                numeric_wheel: false,
                ..Default::default()
            },
        )
    });
    let track = parts(&ctx, "control_test_track");
    draw(&mut value, vec![Event::PointerMoved(track.rect.center())]);
    draw(
        &mut value,
        vec![Event::MouseWheel {
            unit: egui::MouseWheelUnit::Line,
            delta: egui::vec2(0.0, 1.0),
            modifiers: Modifiers::NONE,
            phase: egui::TouchPhase::Move,
        }],
    );
    assert_eq!(value, 0.25, "disabled wheel preference was ignored");
}

#[test]
fn value_rows_keep_numeric_identity_when_labels_change_language() {
    let ctx = egui::Context::default();
    let mut value = 25.0_f32;
    let mut previous = None;
    for label in ["Opacity", "Непрозрачность", "Opacity"] {
        let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
            let row = ValueRow::new(label)
                .id_salt("opacity")
                .label_width(160.0)
                .show(ui, |ui| {
                    ui.add(Numeric::percentage(&mut value).id_salt("value"))
                });
            assert!(row.label.rect.right() <= row.inner.rect.left());
            if let Some(id) = previous {
                assert_eq!(row.inner.id, id);
            }
            previous = Some(row.inner.id);
        });
    }
}

#[test]
fn reopening_a_hidden_numeric_editor_starts_a_new_escape_transaction() {
    let ctx = egui::Context::default();
    let mut value = 50.0_f32;
    fn draw(
        ctx: &egui::Context,
        value: &mut f32,
        visible: bool,
        events: Vec<Event>,
    ) -> Option<Response> {
        let mut response = None;
        let _ = ctx.run_ui(input(events), |ui| {
            if visible {
                response = Some(ui.add(Numeric::percentage(value).id_salt("transient")));
            }
        });
        response
    }
    let select_all = || {
        key(
            Key::A,
            Modifiers {
                ctrl: true,
                command: true,
                ..Default::default()
            },
        )
    };
    let field = draw(&ctx, &mut value, true, vec![]).unwrap();
    ctx.memory_mut(|m| m.request_focus(field.id));
    draw(&ctx, &mut value, true, vec![]);
    draw(
        &ctx,
        &mut value,
        true,
        vec![select_all(), Event::Text("72".into())],
    );
    assert_eq!(value, 72.0);
    draw(&ctx, &mut value, false, vec![]);
    draw(&ctx, &mut value, true, vec![]);
    ctx.memory_mut(|m| m.request_focus(field.id));
    draw(&ctx, &mut value, true, vec![]);
    draw(
        &ctx,
        &mut value,
        true,
        vec![select_all(), Event::Text("90".into())],
    );
    draw(
        &ctx,
        &mut value,
        true,
        vec![key(Key::Escape, Modifiers::NONE)],
    );
    assert_eq!(
        value, 72.0,
        "Escape restored a stale transaction from the closed popup"
    );
}
