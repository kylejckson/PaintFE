use eframe::egui::{self, Color32, Response, Sense, Stroke, Ui, Vec2};

pub(crate) fn theme(ui: &Ui) -> crate::theme::Theme {
    ui.ctx()
        .data(|d| d.get_temp::<crate::theme::Theme>(egui::Id::new("floating_widget_theme")))
        .unwrap_or_default()
}
pub(crate) fn widget_radius(ui: &Ui) -> egui::CornerRadius {
    theme(ui).widget_cr(5)
}
pub(crate) fn field_fill(ui: &Ui) -> Color32 {
    theme(ui).bg2
}

pub(crate) fn icon_button(ui: &mut Ui, label: &str, copy: bool) -> Response {
    let (rect, response) = ui.allocate_exact_size(Vec2::splat(20.0), Sense::click());
    response.widget_info(|| {
        egui::WidgetInfo::labeled(egui::WidgetType::Button, ui.is_enabled(), label)
    });
    crate::ui::polish::control(ui, &response, false, widget_radius(ui));
    let p = ui.painter();
    let c = rect.center();
    let stroke = Stroke::new(1.1, ui.visuals().weak_text_color());
    let now = ui.input(|i| i.time);
    let confirmation_id = response.id.with("copied_until");
    if copy && response.clicked() && crate::ui::polish::settings(ui.ctx()).confirmation_motion {
        ui.ctx()
            .data_mut(|d| d.insert_temp(confirmation_id, now + 0.8));
        ui.ctx()
            .request_repaint_after(std::time::Duration::from_millis(800));
    }
    let confirmed = copy
        && ui
            .ctx()
            .data(|d| d.get_temp::<f64>(confirmation_id))
            .is_some_and(|until| now < until);
    let confirmation = if copy {
        crate::ui::polish::animate(
            ui.ctx(),
            response.id.with("confirmation"),
            if confirmed { 1.0 } else { 0.0 },
            crate::ui::polish::MotionKind::Confirmation,
        )
    } else {
        0.0
    };
    if copy {
        let check_stroke = Stroke::new(stroke.width, stroke.color.gamma_multiply(confirmation));
        let icon_stroke = Stroke::new(
            stroke.width,
            stroke.color.gamma_multiply(1.0 - confirmation),
        );
        if confirmation > 0.0 {
            p.line_segment(
                [c + egui::vec2(-4.0, 0.0), c + egui::vec2(-1.0, 3.0)],
                check_stroke,
            );
            p.line_segment(
                [c + egui::vec2(-1.0, 3.0), c + egui::vec2(5.0, -4.0)],
                check_stroke,
            );
        }
        if confirmation < 1.0 {
            p.rect_stroke(
                egui::Rect::from_min_size(c + egui::vec2(-3.0, -5.0), Vec2::splat(7.0)),
                1,
                icon_stroke,
                egui::StrokeKind::Inside,
            );
            p.rect_stroke(
                egui::Rect::from_min_size(c + egui::vec2(-5.0, -3.0), Vec2::splat(7.0)),
                1,
                icon_stroke,
                egui::StrokeKind::Inside,
            );
        }
    } else {
        for (y, direction) in [(-2.5, 1.0), (2.5, -1.0)] {
            p.line_segment([c + egui::vec2(-5.0, y), c + egui::vec2(5.0, y)], stroke);
            let end = c + egui::vec2(5.0 * direction, y);
            p.line_segment([end, end + egui::vec2(-3.0 * direction, -2.5)], stroke);
            p.line_segment([end, end + egui::vec2(-3.0 * direction, 2.5)], stroke);
        }
    }
    response.on_hover_text(label)
}

pub(crate) fn tabs(ui: &mut Ui, id: &str, labels: &[&str], selected: &mut usize) {
    let width = ui.available_width();
    let (rect, _) = ui.allocate_exact_size(Vec2::new(width, 24.0), Sense::hover());
    ui.painter()
        .rect_filled(rect, theme(ui).tab_rounding as u8, field_fill(ui));
    let cell = width / labels.len() as f32;
    for (i, label) in labels.iter().enumerate() {
        let r = egui::Rect::from_min_size(
            rect.min + Vec2::new(i as f32 * cell, 0.0),
            Vec2::new(cell, 24.0),
        );
        let response = ui.interact(r, ui.id().with((id, i)), Sense::click());
        response.widget_info(|| {
            egui::WidgetInfo::selected(
                egui::WidgetType::SelectableLabel,
                ui.is_enabled(),
                *selected == i,
                *label,
            )
        });
        if response.clicked() {
            *selected = i;
        }
        let t = crate::ui::polish::animate(
            ui.ctx(),
            response.id.with("active"),
            if *selected == i { 1.0 } else { 0.0 },
            crate::ui::polish::MotionKind::Selection,
        );
        crate::ui::polish::control(ui, &response, false, theme(ui).tab_rounding);
        if t > 0.0 {
            let active = theme(ui).accent_faint;
            ui.painter().rect_filled(
                r.shrink(0.5),
                theme(ui).tab_rounding as u8,
                active.gamma_multiply(t),
            );
            ui.painter().rect_stroke(
                r.shrink(0.5),
                theme(ui).tab_rounding as u8,
                Stroke::new(
                    theme(ui).polish.border_width,
                    theme(ui).accent.gamma_multiply(t),
                ),
                egui::StrokeKind::Inside,
            );
        }
        ui.painter().text(
            r.center(),
            egui::Align2::CENTER_CENTER,
            label,
            egui::FontId::proportional(10.0 * crate::ui::polish::settings(ui.ctx()).text_scale),
            ui.visuals().text_color(),
        );
    }
}

pub(crate) fn checker(ui: &Ui, rect: egui::Rect, radius: u8) {
    let style = theme(ui);
    ui.painter().rect_filled(rect, radius, style.bg3);
    let painter = ui.painter().with_clip_rect(rect.intersect(ui.clip_rect()));
    for y in 0..(rect.height() / 4.0).ceil() as usize {
        for x in 0..(rect.width() / 4.0).ceil() as usize {
            if (x + y) % 2 == 0 {
                let cell = egui::Rect::from_min_size(
                    rect.min + Vec2::new(x as f32 * 4.0, y as f32 * 4.0),
                    Vec2::splat(4.0),
                )
                .intersect(rect.shrink(1.0));
                painter.rect_filled(cell, radius.min(1), style.panel_bg);
            }
        }
    }
}

pub(crate) fn swatch(
    ui: &mut Ui,
    id: impl std::hash::Hash + std::fmt::Debug,
    color: Color32,
    size: f32,
    selected: bool,
) -> Response {
    let (rect, _) = ui.allocate_exact_size(Vec2::splat(size), Sense::hover());
    let response = ui.interact(rect, ui.id().with(id), Sense::click());
    response.widget_info(|| {
        egui::WidgetInfo::labeled(
            egui::WidgetType::Button,
            ui.is_enabled(),
            format!("Color {}", hex(color)),
        )
    });
    if ui.is_rect_visible(rect) {
        checker(ui, rect, theme(ui).widget_rounding as u8);
        ui.painter().rect_filled(rect, widget_radius(ui), color);
        let style = theme(ui);
        let hover = crate::ui::polish::hover(
            ui.ctx(),
            response.id.with("hover"),
            response.hovered() && ui.is_enabled(),
        );
        let border = if selected {
            style.accent
        } else {
            crate::theme::Theme::lerp_color(
                ui.visuals().widgets.noninteractive.bg_stroke.color,
                style.accent,
                hover,
            )
        };
        ui.painter().rect_stroke(
            rect,
            widget_radius(ui),
            Stroke::new(
                theme(ui).polish.border_width * if selected { 2.0 } else { 1.0 },
                border,
            ),
            egui::StrokeKind::Inside,
        );
    }
    crate::ui::polish::focus(ui, &response, theme(ui).widget_rounding);
    response.on_hover_text(format!(
        "{} · opacity {}\nClick: selected slot · Right: Secondary",
        hex(color),
        color.a()
    ))
}

pub(crate) fn rgba(r: u8, g: u8, b: u8, a: u8) -> Color32 {
    let p = |c: u8| ((c as u16 * a as u16 + 127) / 255) as u8;
    Color32::from_rgba_premultiplied(p(r), p(g), p(b), a)
}

pub(crate) fn straight(c: Color32) -> [u8; 4] {
    let a = c.a();
    let u = |v: u8| {
        if a == 0 {
            0
        } else {
            ((v as u32 * 255 + a as u32 / 2) / a as u32).min(255) as u8
        }
    };
    [u(c.r()), u(c.g()), u(c.b()), a]
}

pub(crate) fn hex(c: Color32) -> String {
    let [r, g, b, _] = straight(c);
    format!("#{r:02X}{g:02X}{b:02X}")
}

pub(crate) fn parse_hex(text: &str) -> Option<[u8; 3]> {
    let text = text.trim().strip_prefix('#').unwrap_or(text.trim());
    if text.len() != 6 || !text.is_ascii() {
        return None;
    }
    Some([
        u8::from_str_radix(&text[0..2], 16).ok()?,
        u8::from_str_radix(&text[2..4], 16).ok()?,
        u8::from_str_radix(&text[4..6], 16).ok()?,
    ])
}

/// Capture the original press, including gestures whose release is outside the control.
pub(crate) fn drag_motion(
    ui: &Ui,
    rect: egui::Rect,
    id: egui::Id,
) -> (Option<egui::Pos2>, Option<egui::Pos2>, bool) {
    let id = id.with("pointer_capture");
    let mut active = ui
        .ctx()
        .data_mut(|d| d.get_temp::<bool>(id))
        .unwrap_or(false);
    let mut start = None;
    let mut position = None;
    let events = ui.input(|input| input.events.clone());
    for event in &events {
        match event {
            egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed: true,
                ..
            } if ui.is_enabled()
                && rect.intersect(ui.clip_rect()).contains(*pos)
                && ui
                    .ctx()
                    .layer_id_at(*pos)
                    .is_none_or(|layer| layer == ui.layer_id()) =>
            {
                active = true;
                start = Some(*pos);
                position = Some(*pos);
            }
            egui::Event::PointerMoved(pos) if active => position = Some(*pos),
            egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed: false,
                ..
            } if active => {
                position = Some(*pos);
                active = false;
            }
            egui::Event::PointerGone => active = false,
            _ => {}
        }
    }
    ui.ctx().data_mut(|d| d.insert_temp(id, active));
    (start, position, active)
}

/// Thin rounded track, immediate pointer updates, and a directly editable numeric capsule.
pub(crate) fn slider(
    ui: &mut Ui,
    id: &str,
    label: &str,
    value: &mut f32,
    max: f32,
    sample: impl Fn(f32) -> Color32,
    alpha: bool,
) -> bool {
    let mut changed = false;
    ui.push_id(id, |ui| {
        ui.horizontal(|ui| {
            if !label.is_empty() {
                ui.add_sized(
                    [12.0, 22.0],
                    egui::Label::new(egui::RichText::new(label).size(10.0)),
                );
            }
            let width = (ui.available_width() - 47.0).max(40.0);
            let (rect, response) =
                ui.allocate_exact_size(Vec2::new(width, 22.0), Sense::click_and_drag());
            response
                .widget_info(|| egui::WidgetInfo::slider(ui.is_enabled(), *value as f64, label));
            let track = egui::Rect::from_center_size(rect.center(), Vec2::new(width - 4.0, 10.0));
            if let (_, Some(pos), _) = drag_motion(ui, rect, response.id) {
                let next = ((pos.x - track.left()) / track.width()).clamp(0.0, 1.0) * max;
                changed |= next != *value;
                *value = next;
            }
            changed |= crate::ui::numeric::slider_input(ui, &response, value, 0.0..=max, 1.0);
            if ui.is_rect_visible(rect) {
                let style = theme(ui);
                let radius = style.widget_rounding.clamp(0.0, 5.0);
                if alpha {
                    checker(ui, track, radius as u8);
                }
                // Rounded outer silhouette: each vertical slice follows the circular end caps.
                let n = track.width().ceil() as usize;
                for x in 0..n {
                    let px = x as f32 + 0.5;
                    let edge = px.min(track.width() - px).max(0.0);
                    let half = if edge < radius {
                        5.0 - radius + (radius.powi(2) - (radius - edge).powi(2)).sqrt()
                    } else {
                        5.0
                    };
                    let r = egui::Rect::from_center_size(
                        egui::pos2(track.left() + px, track.center().y),
                        Vec2::new(1.1, half * 2.0),
                    );
                    ui.painter().rect_filled(r, 0, sample(px / track.width()));
                }
                ui.painter().rect_stroke(
                    track,
                    radius as u8,
                    Stroke::new(0.5, ui.visuals().widgets.noninteractive.bg_stroke.color),
                    egui::StrokeKind::Inside,
                );
                let center = egui::pos2(
                    track.left() + (*value / max).clamp(0.0, 1.0) * track.width(),
                    track.center().y,
                );
                ui.painter().circle_filled(center, 5.0, style.panel_bg);
                ui.painter()
                    .circle_stroke(center, 5.0, Stroke::new(1.0, style.border_color));
                ui.painter().circle_filled(center, 1.5, style.text_color);
            }
            egui::Frame::new()
                .fill(field_fill(ui))
                .corner_radius(widget_radius(ui))
                .show(ui, |ui| {
                    changed |= ui
                        .add_sized(
                            [37.0, 20.0],
                            crate::ui::numeric::Numeric::new(value)
                                .range(0.0..=max)
                                .speed(1.0)
                                .max_decimals(0),
                        )
                        .changed();
                });
        });
    });
    changed
}

#[cfg(test)]
mod redesign_tests {
    use super::*;

    #[test]
    fn hex_validation_and_alpha_roundtrip() {
        assert_eq!(parse_hex(" #a1B2c3 "), Some([161, 178, 195]));
        for text in ["#12345", "#GG0000", "é12345", "#12345678", ""] {
            assert!(parse_hex(text).is_none());
        }
        for a in [0, 1, 128, 255] {
            let color = rgba(200, 80, 30, a);
            let [r, g, b, a] = straight(color);
            assert_eq!(rgba(r, g, b, a), color);
        }
    }

    #[test]
    fn fast_drag_release_outside_keeps_last_position_and_releases_capture() {
        let ctx = egui::Context::default();
        let origin = egui::pos2(30.0, 30.0);
        let outside = egui::pos2(250.0, 30.0);
        let button = |pos, pressed| egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::NONE,
        };
        let input = egui::RawInput {
            events: vec![
                button(origin, true),
                egui::Event::PointerMoved(outside),
                button(outside, false),
            ],
            ..Default::default()
        };
        let mut result = (None, None, false);
        let _ = ctx.run_ui(input, |ui| {
            result = drag_motion(
                ui,
                egui::Rect::from_min_size(egui::pos2(20.0, 20.0), Vec2::new(100.0, 20.0)),
                egui::Id::new("test_drag"),
            );
        });
        assert_eq!(result, (Some(origin), Some(outside), false));
        let _ = ctx.run_ui(
            egui::RawInput {
                events: vec![egui::Event::PointerMoved(origin)],
                ..Default::default()
            },
            |ui| {
                result = drag_motion(ui, egui::Rect::EVERYTHING, egui::Id::new("test_drag"));
            },
        );
        assert_eq!(result, (None, None, false));
    }
}
