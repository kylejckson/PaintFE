//! Colour popups must not reintroduce egui's private channel editors.
use egui::color_picker::Alpha;
use egui::{
    Color32, Response, Ui,
    ecolor::{Hsva, HsvaGamma},
};

#[track_caller]
pub fn color_edit_button_srgba(ui: &mut Ui, color: &mut Color32, alpha: Alpha) -> Response {
    let mut hsv = Hsva::from(*color);
    let response = color_button(ui, &mut hsv, alpha);
    if response.changed() {
        *color = hsv.into();
    }
    response
}

#[track_caller]
pub fn color_edit_button_srgb(ui: &mut Ui, color: &mut [u8; 3]) -> Response {
    let mut value = Color32::from_rgb(color[0], color[1], color[2]);
    let response = color_edit_button_srgba(ui, &mut value, Alpha::Opaque);
    if response.changed() {
        *color = [value.r(), value.g(), value.b()];
    }
    response
}

#[track_caller]
pub fn color_edit_button_rgb(ui: &mut Ui, color: &mut [f32; 3]) -> Response {
    let mut hsv = Hsva::from(egui::Rgba::from_rgb(color[0], color[1], color[2]));
    let response = color_button(ui, &mut hsv, Alpha::Opaque);
    if response.changed() {
        let rgba = egui::Rgba::from(hsv);
        *color = [rgba.r(), rgba.g(), rgba.b()];
    }
    response
}

#[track_caller]
fn color_button(ui: &mut Ui, hsv: &mut Hsva, alpha: Alpha) -> Response {
    let source = std::panic::Location::caller();
    let id = ui.make_persistent_id((
        "color_control",
        source.file(),
        source.line(),
        source.column(),
    ));
    ui.scope_builder(egui::UiBuilder::new().id(id), |ui| {
        let mut response = ui.allocate_response(ui.spacing().interact_size, egui::Sense::click());
        response.widget_info(|| {
            egui::WidgetInfo::labeled(
                egui::WidgetType::Button,
                ui.is_enabled(),
                crate::t!("keybind_category.color"),
            )
        });
        if ui.is_rect_visible(response.rect) {
            egui::color_picker::show_color_at(
                ui.painter(),
                Color32::from(*hsv),
                response.rect.shrink(1.0),
            );
            ui.painter().rect_stroke(
                response.rect,
                ui.visuals().widgets.inactive.corner_radius,
                ui.style().interact(&response).bg_stroke,
                egui::StrokeKind::Inside,
            );
        }
        let cache_id = id.with("hue_cache");
        let mut gamma = ui
            .ctx()
            .data(|d| d.get_temp::<(Color32, HsvaGamma)>(cache_id))
            .filter(|(previous, _)| *previous == Color32::from(*hsv))
            .map(|(_, gamma)| gamma)
            .unwrap_or_else(|| HsvaGamma::from(*hsv));
        egui::Popup::menu(&response)
            .id(id.with("popup"))
            .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
            .show(|ui| {
                if picker_body(ui, &mut gamma, alpha) {
                    *hsv = gamma.into();
                    response.mark_changed();
                }
            });
        ui.ctx()
            .data_mut(|d| d.insert_temp(cache_id, (Color32::from(*hsv), gamma)));
        response
    })
    .inner
}

pub(crate) fn picker_body(ui: &mut Ui, hsv: &mut HsvaGamma, alpha: Alpha) -> bool {
    ui.set_width(270.0);
    let before = *hsv;
    let theme = ui
        .ctx()
        .data(|d| d.get_temp::<crate::theme::Theme>(egui::Id::new("floating_widget_theme")))
        .unwrap_or_else(|| {
            if ui.visuals().dark_mode {
                crate::theme::Theme::dark()
            } else {
                crate::theme::Theme::light()
            }
        });
    let (rect, response) =
        ui.allocate_exact_size(egui::vec2(270.0, 140.0), egui::Sense::click_and_drag());
    response
        .widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Other, ui.is_enabled(), "HSV"));
    if let (_, Some(pos), _) = super::drag_motion(ui, rect, response.id) {
        hsv.s = ((pos.x - rect.left()) / rect.width()).clamp(0.0, 1.0);
        hsv.v = (1.0 - (pos.y - rect.top()) / rect.height()).clamp(0.0, 1.0);
    }
    if ui.is_rect_visible(rect) {
        let mut mesh = egui::Mesh::default();
        for y in 0..=16 {
            for x in 0..=16 {
                let s = x as f32 / 16.0;
                let v = 1.0 - y as f32 / 16.0;
                mesh.colored_vertex(
                    egui::pos2(
                        rect.left() + s * rect.width(),
                        rect.top() + (1.0 - v) * rect.height(),
                    ),
                    Color32::from(HsvaGamma {
                        s,
                        v,
                        a: 1.0,
                        ..*hsv
                    }),
                );
            }
        }
        for y in 0..16 {
            for x in 0..16 {
                let i = y * 17 + x;
                mesh.add_triangle(i, i + 1, i + 17);
                mesh.add_triangle(i + 1, i + 18, i + 17);
            }
        }
        ui.painter().add(egui::Shape::mesh(mesh));
        let position = egui::pos2(
            rect.left() + hsv.s * rect.width(),
            rect.bottom() - hsv.v * rect.height(),
        );
        ui.painter()
            .circle_stroke(position, 4.0, egui::Stroke::new(1.5, Color32::WHITE));
        ui.painter()
            .circle_stroke(position, 5.5, egui::Stroke::new(1.0, Color32::BLACK));
    }
    for (label, max) in [("H", 360.0), ("S", 100.0), ("V", 100.0)] {
        let current = *hsv;
        let mut value = match label {
            "H" => hsv.h * max,
            "S" => hsv.s * max,
            _ => hsv.v * max,
        };
        let response = ui.add(
            super::GradientSlider::new(&mut value, max, |t| {
                Color32::from(match label {
                    "H" => HsvaGamma {
                        h: t,
                        s: 1.0,
                        v: 1.0,
                        a: 1.0,
                    },
                    "S" => HsvaGamma {
                        s: t,
                        a: 1.0,
                        ..current
                    },
                    _ => HsvaGamma {
                        v: t,
                        a: 1.0,
                        ..current
                    },
                })
            })
            .id_salt(label)
            .label(label)
            .theme(theme.clone()),
        );
        if response.changed() {
            match label {
                "H" => hsv.h = value / max,
                "S" => hsv.s = value / max,
                _ => hsv.v = value / max,
            }
        }
    }
    if alpha == Alpha::BlendOrAdditive {
        let mut additive = hsv.a < 0.0;
        if ui
            .checkbox(&mut additive, crate::t!("blend.additive"))
            .changed()
        {
            hsv.a = if additive { -1.0 } else { 1.0 };
        }
    }
    if alpha != Alpha::Opaque && hsv.a >= 0.0 {
        let current = *hsv;
        let mut value = hsv.a * 255.0;
        if ui
            .add(
                super::GradientSlider::new(&mut value, 255.0, |t| {
                    Color32::from(HsvaGamma { a: t, ..current })
                })
                .id_salt("alpha")
                .reset_to(255.0)
                .label("A")
                .alpha(true)
                .theme(theme.clone()),
            )
            .changed()
        {
            hsv.a = value / 255.0;
        }
    }
    let mode_id = ui.make_persistent_id("linear_channels");
    let mut linear = ui
        .ctx()
        .data(|d| d.get_temp::<bool>(mode_id))
        .unwrap_or(matches!(
            ui.visuals().numeric_color_space,
            egui::style::NumericColorSpace::Linear
        ));
    ui.horizontal(|ui| {
        ui.selectable_value(&mut linear, false, "RGB 0–255");
        ui.selectable_value(&mut linear, true, "RGB 0–1");
    });
    ui.ctx().data_mut(|d| d.insert_temp(mode_id, linear));
    ui.horizontal(|ui| {
        let colour = Color32::from(*hsv);
        let mut channels = colour.to_srgba_unmultiplied();
        let mut floats = Hsva::from(*hsv).to_rgba_unmultiplied();
        let mut edited = false;
        for (index, label) in ["R", "G", "B", "A"].iter().enumerate() {
            if index == 3 && (alpha == Alpha::Opaque || hsv.a < 0.0) {
                continue;
            }
            edited |= if linear {
                ui.add(
                    super::Numeric::new(&mut floats[index])
                        .id_salt(index)
                        .range(0.0..=1.0)
                        .clamp_existing_to_range(false)
                        .prefix(format!("{label} "))
                        .speed(0.01)
                        .max_decimals(3),
                )
                .changed()
            } else {
                ui.add(
                    super::Numeric::new(&mut channels[index])
                        .id_salt(index)
                        .range(0..=255)
                        .prefix(format!("{label} "))
                        .speed(1.0),
                )
                .changed()
            };
        }
        if edited {
            let a = if alpha == Alpha::Opaque || hsv.a < 0.0 {
                255
            } else {
                channels[3]
            };
            let mut next = if linear {
                if alpha == Alpha::Opaque || hsv.a < 0.0 {
                    floats[3] = 1.0;
                }
                HsvaGamma::from(Hsva::from_rgba_unmultiplied(
                    floats[0], floats[1], floats[2], floats[3],
                ))
            } else {
                HsvaGamma::from(Hsva::from(Color32::from_rgba_unmultiplied(
                    channels[0],
                    channels[1],
                    channels[2],
                    a,
                )))
            };
            if hsv.a < 0.0 {
                next.a = -1.0;
            }
            *hsv = next;
        }
    });
    ui.horizontal(|ui| {
        let [r, g, b, a] = Color32::from(*hsv).to_srgba_unmultiplied();
        let hex = format!("#{r:02X}{g:02X}{b:02X}{a:02X}");
        ui.monospace(&hex);
        if ui.small_button(crate::t!("common.copy_hex")).clicked() {
            ui.ctx().copy_text(hex);
        }
    });
    egui::color_picker::show_color(ui, Color32::from(*hsv), egui::vec2(270.0, 22.0));
    *hsv != before
}

#[cfg(test)]
mod tests {
    use super::*;

    fn draw(ctx: &egui::Context, hsv: &mut HsvaGamma, alpha: Alpha, events: Vec<egui::Event>) {
        let _ = ctx.run_ui(
            egui::RawInput {
                events,
                ..Default::default()
            },
            |ui| {
                picker_body(ui, hsv, alpha);
            },
        );
    }

    #[test]
    fn colour_picker_rendering_preserves_alpha_additive_and_linear_values() {
        for dark in [false, true] {
            for linear in [false, true] {
                for alpha in [Alpha::Opaque, Alpha::OnlyBlend, Alpha::BlendOrAdditive] {
                    let ctx = egui::Context::default();
                    let mut visuals = if dark {
                        egui::Visuals::dark()
                    } else {
                        egui::Visuals::light()
                    };
                    visuals.numeric_color_space = if linear {
                        egui::style::NumericColorSpace::Linear
                    } else {
                        egui::style::NumericColorSpace::GammaByte
                    };
                    ctx.set_visuals(visuals);
                    let mut hsv = HsvaGamma {
                        h: 0.63,
                        s: 0.42,
                        v: 1.2,
                        a: if alpha == Alpha::BlendOrAdditive {
                            -1.0
                        } else {
                            0.4
                        },
                    };
                    let original = hsv;
                    for _ in 0..3 {
                        draw(&ctx, &mut hsv, alpha, vec![]);
                    }
                    assert_eq!(
                        hsv, original,
                        "rendering edited colour: dark={dark}, linear={linear}"
                    );
                }
            }
        }
    }

    #[test]
    fn colour_popup_alpha_channel_uses_shared_typing_without_double_premultiplication() {
        let ctx = egui::Context::default();
        let mut hsv = HsvaGamma::from(Hsva::from(Color32::from_rgba_unmultiplied(
            200, 100, 50, 128,
        )));
        draw(&ctx, &mut hsv, Alpha::OnlyBlend, vec![]);
        let editor = ctx
            .data(|d| d.get_temp::<Response>(egui::Id::new("control_test_editor")))
            .unwrap();
        ctx.memory_mut(|m| m.request_focus(editor.id));
        draw(&ctx, &mut hsv, Alpha::OnlyBlend, vec![]);
        let key = |key, modifiers| egui::Event::Key {
            key,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers,
        };
        draw(
            &ctx,
            &mut hsv,
            Alpha::OnlyBlend,
            vec![
                key(
                    egui::Key::A,
                    egui::Modifiers {
                        ctrl: true,
                        command: true,
                        ..Default::default()
                    },
                ),
                egui::Event::Text("64".into()),
            ],
        );
        draw(
            &ctx,
            &mut hsv,
            Alpha::OnlyBlend,
            vec![key(egui::Key::Tab, egui::Modifiers::NONE)],
        );
        let [r, g, b, a] = Color32::from(hsv).to_srgba_unmultiplied();
        assert_eq!(a, 64);
        for (value, expected) in [(r, 200_u8), (g, 100), (b, 50)] {
            assert!(
                value.abs_diff(expected) <= 3,
                "channel darkened after alpha edit: {value} vs {expected}"
            );
        }
    }

    #[test]
    fn closed_colour_buttons_do_not_quantise_linear_or_premultiplied_models() {
        let ctx = egui::Context::default();
        let mut rgb = [1.234_567, 0.123_456, 0.654_321];
        let mut srgba = Color32::from_rgba_unmultiplied(200, 100, 50, 37);
        let original_rgb = rgb;
        let original_srgba = srgba;
        for _ in 0..3 {
            let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
                assert!(!color_edit_button_rgb(ui, &mut rgb).changed());
                assert!(!color_edit_button_srgba(ui, &mut srgba, Alpha::OnlyBlend).changed());
            });
        }
        assert_eq!(rgb, original_rgb);
        assert_eq!(srgba, original_srgba);
    }
}
