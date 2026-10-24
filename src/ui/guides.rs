//! Shared guide interactions; only completed gestures mutate the document.
use crate::{
    canvas::{
        CanvasState,
        guides::{Axis, GuideCommand, Guides},
    },
    project::Project,
};
use egui::{Context, Id, Rect, Ui};
#[derive(Clone)]
struct Drag {
    project: uuid::Uuid,
    before: Guides,
    id: Option<u64>,
    axis: Axis,
    position: f32,
}
#[derive(Clone)]
struct Edit {
    project: uuid::Uuid,
    before: Guides,
    after: Guides,
}
#[derive(Clone)]
struct NewGuide {
    project: uuid::Uuid,
    axis: Axis,
    position: f32,
    percent: bool,
}
fn project(ctx: &Context) -> Option<uuid::Uuid> {
    ctx.data(|d| d.get_temp(Id::new("guides_project")))
}
pub fn begin(ctx: &Context, id: uuid::Uuid) {
    ctx.data_mut(|d| d.insert_temp(Id::new("guides_project"), id));
}
pub fn gesture_active(ctx: &Context) -> bool {
    ctx.data(|d| d.get_temp::<Drag>(Id::new("guide_drag")))
        .is_some_and(|d| Some(d.project) == project(ctx))
}
pub fn dialog_open(ctx: &Context) -> bool {
    ctx.data(|d| d.get_temp::<NewGuide>(Id::new("new_guide_dialog")))
        .is_some()
}
fn queue(ctx: &Context, before: Guides, after: Guides) {
    if before == after {
        return;
    }
    if let Some(project) = project(ctx) {
        ctx.data_mut(|d| {
            let mut edits = d
                .get_temp::<Vec<Edit>>(Id::new("guide_edits"))
                .unwrap_or_default();
            edits.push(Edit {
                project,
                before,
                after,
            });
            d.insert_temp(Id::new("guide_edits"), edits);
        });
    }
}
pub fn drain(ctx: &Context, p: &mut Project) {
    let edits = ctx
        .data_mut(|d| d.remove_temp::<Vec<Edit>>(Id::new("guide_edits")))
        .unwrap_or_default();
    for e in edits {
        if e.project == p.id {
            p.history.push(Box::new(GuideCommand {
                before: e.before,
                after: e.after,
            }));
            p.mark_dirty();
        }
    }
}
pub fn menu(ui: &mut Ui, p: &mut Project) {
    begin(ui.ctx(), p.id);
    let before = p.canvas_state.guides.clone();
    let g = &mut p.canvas_state.guides;
    ui.checkbox(&mut g.rulers, crate::t!("guides.rulers"));
    ui.checkbox(&mut g.visible, crate::t!("guides.visible"));
    ui.checkbox(&mut g.locked, crate::t!("guides.locked"));
    ui.checkbox(&mut g.snap, crate::t!("guides.snap"));
    ui.separator();
    if ui
        .add_enabled(!g.locked, egui::Button::new(crate::t!("guides.new")))
        .clicked()
    {
        ui.ctx().data_mut(|d| {
            d.insert_temp(
                Id::new("new_guide_dialog"),
                NewGuide {
                    project: p.id,
                    axis: Axis::Horizontal,
                    position: 50.0,
                    percent: true,
                },
            )
        });
        ui.close();
    }
    if ui
        .add_enabled(
            !g.locked && !g.items.is_empty(),
            egui::Button::new(crate::t!("guides.clear")),
        )
        .clicked()
    {
        g.items.clear();
    }
    if before.items != g.items {
        queue(ui.ctx(), before, g.clone());
    }
}
pub fn dialog(ctx: &Context, p: &mut Project) {
    let Some(mut new) = ctx.data(|d| d.get_temp::<NewGuide>(Id::new("new_guide_dialog"))) else {
        return;
    };
    if new.project != p.id {
        ctx.data_mut(|d| d.remove::<NewGuide>(Id::new("new_guide_dialog")));
        return;
    }
    let mut close = false;
    egui::Modal::new(Id::new("guide_position_dialog")).show(ctx, |ui| {
        ui.set_width(320.0);
        ui.heading(crate::t!("guides.new"));
        ui.horizontal(|ui| {
            ui.selectable_value(
                &mut new.axis,
                Axis::Horizontal,
                crate::t!("guides.horizontal"),
            );
            ui.selectable_value(&mut new.axis, Axis::Vertical, crate::t!("guides.vertical"));
        });
        let old_percent = new.percent;
        let extent = match new.axis {
            Axis::Horizontal => p.canvas_state.height,
            Axis::Vertical => p.canvas_state.width,
        } as f32;
        ui.checkbox(&mut new.percent, crate::t!("guides.percent"));
        if old_percent != new.percent {
            new.position = if new.percent {
                new.position / extent * 100.0
            } else {
                new.position / 100.0 * extent
            };
        }
        let limit = if new.percent { 100.0 } else { extent };
        ui.horizontal(|ui| {
            ui.label(crate::t!("guides.position"));
            ui.add(
                crate::ui::controls::Numeric::new(&mut new.position)
                    .id_salt("guide_position")
                    .range(0.0..=limit)
                    .suffix(if new.percent { " %" } else { " px" }),
            );
        });
        ui.weak(crate::t!("guides.save_hint"));
        ui.horizontal(|ui| {
            if ui.button(crate::t!("guides.add")).clicked()
                || ui.input(|i| i.key_pressed(egui::Key::Enter))
            {
                let before = p.canvas_state.guides.clone();
                let position = if new.percent {
                    new.position * extent / 100.0
                } else {
                    new.position
                };
                if p.canvas_state.guides.add(
                    new.axis,
                    position,
                    p.canvas_state.width,
                    p.canvas_state.height,
                ) {
                    p.canvas_state.guides.visible = true;
                    queue(ctx, before, p.canvas_state.guides.clone());
                    close = true;
                }
            }
            if ui.button(crate::t!("guides.cancel")).clicked() {
                close = true;
            }
        });
    });
    close |= ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::Escape));
    ctx.data_mut(|d| {
        if close {
            d.remove::<NewGuide>(Id::new("new_guide_dialog"));
        } else {
            d.insert_temp(Id::new("new_guide_dialog"), new);
        }
    });
}
fn press_origin(ctx: &Context) -> Option<egui::Pos2> {
    ctx.input(|i| {
        i.events
            .iter()
            .find_map(|event| match event {
                egui::Event::PointerButton {
                    pos,
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    ..
                } => Some(*pos),
                _ => None,
            })
            .or(i.pointer.press_origin())
    })
}
pub fn set_appearance(ctx: &Context, appearance: &crate::config::settings::RulerSettings) {
    let mut value = appearance.clone();
    value.normalize();
    ctx.data_mut(|d| d.insert_temp(Id::new("ruler_appearance"), value));
}
fn appearance(ctx: &Context) -> crate::config::settings::RulerSettings {
    ctx.data(|d| d.get_temp(Id::new("ruler_appearance")))
        .unwrap_or_default()
}
fn ruler_size(style: &crate::config::settings::RulerSettings, height: u32) -> egui::Vec2 {
    // Both registered UI families fit within this conservative numeric width.
    egui::vec2(
        (style.vertical_font().size * 0.7 * height.to_string().len() as f32
            + if style.compact_vertical { 8.0 } else { 12.0 })
        .max(if style.compact_vertical { 20.0 } else { 36.0 }),
        (style.font_size + 9.0).max(20.0),
    )
}
fn rulers(image: Rect, view: Rect, size: egui::Vec2) -> (Rect, Rect) {
    // Document-space edges, projected to screen space. Clip rather than pinning
    // rulers to the workspace when an edge is panned or zoomed out of view.
    let clip = |rect: Rect| {
        let visible = rect.intersect(view);
        if visible.is_positive() {
            visible
        } else {
            Rect::NOTHING
        }
    };
    (
        clip(Rect::from_min_max(
            egui::pos2(image.left(), image.top() - size.y),
            image.right_top(),
        )),
        clip(Rect::from_min_max(
            egui::pos2(image.left() - size.x, image.top()),
            image.left_bottom(),
        )),
    )
}
pub fn owns_pointer(
    ctx: &Context,
    state: &CanvasState,
    image: Rect,
    view: Rect,
    zoom: f32,
    move_tool: bool,
) -> bool {
    if ctx
        .data(|d| d.get_temp::<Drag>(Id::new("guide_drag")))
        .is_some_and(|d| Some(d.project) == project(ctx))
    {
        return true;
    }
    let origin = press_origin(ctx);
    let Some(p) = ctx.input(|i| {
        if i.pointer.primary_pressed() {
            origin
        } else {
            i.pointer.hover_pos().or(i.pointer.interact_pos())
        }
    }) else {
        return false;
    };
    let (top, left) = rulers(image, view, ruler_size(&appearance(ctx), state.height));
    if state.guides.rulers && (top.contains(p) || left.contains(p)) {
        return true;
    }
    !state.guides.locked
        && state.guides.visible
        && move_tool
        && view.contains(p)
        && image.contains(p)
        && state.guides.items.iter().any(|g| match g.axis {
            Axis::Horizontal => (p.y - image.top() - g.position * zoom).abs() <= 5.0,
            Axis::Vertical => (p.x - image.left() - g.position * zoom).abs() <= 5.0,
        })
}
pub fn interact(
    ui: &Ui,
    state: &mut CanvasState,
    image: Rect,
    view: Rect,
    zoom: f32,
    move_tool: bool,
    blocked: bool,
) -> bool {
    let ctx = ui.ctx();
    let key = Id::new("guide_drag");
    let mut drag = ctx.data(|d| d.get_temp::<Drag>(key));
    let pid = project(ctx);
    if blocked || dialog_open(ctx) || drag.as_ref().is_some_and(|d| Some(d.project) != pid) {
        ctx.data_mut(|d| d.remove::<Drag>(key));
        return false;
    }
    let pos = ui.input(|i| i.pointer.interact_pos().or(i.pointer.hover_pos()));
    let (top, left) = rulers(image, view, ruler_size(&appearance(ctx), state.height));
    // A fast drag can leave the narrow ruler before the next redraw. Decide
    // ownership from the actual press, not the pointer's latest frame position.
    let start = press_origin(ctx);
    let start_over_ruler =
        state.guides.rulers && start.is_some_and(|p| top.contains(p) || left.contains(p));
    let over_ruler =
        state.guides.rulers && pos.is_some_and(|p| top.contains(p) || left.contains(p));
    if state.guides.locked {
        ctx.data_mut(|d| d.remove::<Drag>(key));
        return over_ruler || (ui.input(|i| i.pointer.primary_pressed()) && start_over_ruler);
    }
    let hit = if state.guides.visible && move_tool {
        (if ui.input(|i| i.pointer.primary_pressed()) {
            start.or(pos)
        } else {
            pos
        })
        .and_then(|p| {
            state
                .guides
                .items
                .iter()
                .filter_map(|g| {
                    let screen = match g.axis {
                        Axis::Horizontal => image.top() + g.position * zoom,
                        Axis::Vertical => image.left() + g.position * zoom,
                    };
                    let delta = match g.axis {
                        Axis::Horizontal => (p.y - screen).abs(),
                        Axis::Vertical => (p.x - screen).abs(),
                    };
                    (delta <= 5.0 && view.contains(p) && image.contains(p)).then_some((g, delta))
                })
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .map(|(g, _)| g.clone())
        })
    } else {
        None
    };
    if drag.is_none()
        && ui.input(|i| i.pointer.primary_pressed())
        && let (Some(p), Some(pid)) = (pos, pid)
    {
        let axis = if start_over_ruler {
            Some(if top.contains(start.unwrap_or(p)) {
                Axis::Horizontal
            } else {
                Axis::Vertical
            })
        } else {
            hit.as_ref().map(|g| g.axis)
        };
        if let Some(axis) = axis {
            drag = Some(Drag {
                project: pid,
                before: state.guides.clone(),
                id: if start_over_ruler {
                    None
                } else {
                    hit.as_ref().map(|g| g.id)
                },
                axis,
                position: 0.0,
            });
        }
    }
    let handled = over_ruler || hit.is_some() || drag.is_some();
    if handled {
        ctx.set_cursor_icon(egui::CursorIcon::Grab);
    }
    if let Some(mut d) = drag {
        if let Some(p) = pos {
            d.position = match d.axis {
                Axis::Horizontal => (p.y - image.top()) / zoom,
                Axis::Vertical => (p.x - image.left()) / zoom,
            };
            if !ui.input(|i| i.modifiers.alt) {
                d.position = d.position.round();
            }
        }
        ctx.set_cursor_icon(egui::CursorIcon::Grabbing);
        if ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::Escape)) {
            ctx.data_mut(|data| data.remove::<Drag>(key));
            return true;
        }
        if ui.input(|i| i.pointer.primary_released()) {
            let mut after = d.before.clone();
            let inside = pos.is_some_and(|p| {
                image.contains(p)
                    && view.contains(p)
                    && !(state.guides.rulers && (top.contains(p) || left.contains(p)))
            });
            if let Some(id) = d.id {
                if inside {
                    if let Some(g) = after.items.iter_mut().find(|g| g.id == id) {
                        g.position = d.position;
                    }
                } else {
                    after.items.retain(|g| g.id != id);
                }
            } else if inside {
                after.add(d.axis, d.position, state.width, state.height);
                after.visible = true;
            }
            queue(ctx, d.before, after.clone());
            state.guides = after;
            ctx.data_mut(|data| data.remove::<Drag>(key));
        } else {
            ctx.data_mut(|data| data.insert_temp(key, d));
        }
    }
    handled
}
pub fn paint(ui: &Ui, state: &CanvasState, image: Rect, view: Rect, zoom: f32) {
    let painter = ui.painter().with_clip_rect(view);
    let visual = ui.visuals();
    let draw = |axis: Axis, position: f32, active: bool| {
        let c = if active {
            visual.selection.stroke.color
        } else {
            egui::Color32::from_rgb(50, 150, 230)
        };
        let segment = match axis {
            Axis::Horizontal => {
                let y = image.top() + position * zoom;
                [
                    egui::pos2(image.left().max(view.left()), y),
                    egui::pos2(image.right().min(view.right()), y),
                ]
            }
            Axis::Vertical => {
                let x = image.left() + position * zoom;
                [
                    egui::pos2(x, image.top().max(view.top())),
                    egui::pos2(x, image.bottom().min(view.bottom())),
                ]
            }
        };
        painter.line_segment(segment, (2.5, egui::Color32::from_black_alpha(110)));
        painter.add(egui::Shape::dashed_line(
            &segment,
            egui::Stroke::new(1.0, c),
            5.0,
            4.0,
        ));
    };
    let drag = ui
        .ctx()
        .data(|d| d.get_temp::<Drag>(Id::new("guide_drag")))
        .filter(|d| Some(d.project) == project(ui.ctx()));
    if state.guides.visible {
        for g in &state.guides.items {
            if drag.as_ref().is_none_or(|d| d.id != Some(g.id)) {
                draw(g.axis, g.position, false);
            }
        }
    }
    if let Some(d) = drag {
        draw(d.axis, d.position, true);
    }
    if !state.guides.rulers {
        return;
    }
    let style = appearance(ui.ctx());
    let size = ruler_size(&style, state.height);
    let (top, left) = rulers(image, view, size);
    let color = |value: Option<[u8; 3]>, fallback: egui::Color32| {
        value
            .map(|[r, g, b]| egui::Color32::from_rgb(r, g, b))
            .unwrap_or(fallback)
            .gamma_multiply(style.opacity)
    };
    let text_color = color(style.text_color, visual.text_color());
    let tick_color = color(style.tick_color, visual.text_color());
    let background =
        color(style.background_color, visual.panel_fill).gamma_multiply(style.background_opacity);
    for rect in [top, left] {
        if rect.is_positive() {
            painter.rect_filled(rect, 0.0, background);
        }
    }
    let rough = (style.font_size * 7.0).max(65.0) / zoom.max(0.001);
    let base = 10.0_f32.powf(rough.log10().floor());
    let step = [1.0, 2.0, 5.0, 10.0]
        .into_iter()
        .map(|m| m * base)
        .find(|s| *s >= rough)
        .unwrap_or(base * 10.0);
    for axis in [Axis::Horizontal, Axis::Vertical] {
        let (start, end, origin) = match axis {
            Axis::Horizontal => (top.left(), top.right(), image.left()),
            Axis::Vertical => (left.top(), left.bottom(), image.top()),
        };
        let ruler = match axis {
            Axis::Horizontal => top,
            Axis::Vertical => left,
        };
        if !ruler.is_positive() {
            continue;
        }
        let ruler_painter = painter.with_clip_rect(ruler);
        if style.minor_ticks {
            let divisions = if (step / base - 5.0).abs() < 0.001 {
                5.0
            } else {
                10.0
            };
            let minor = (step / divisions).max(1.0);
            if minor < step && minor * zoom >= 4.0 {
                let first_minor = ((start - origin) / zoom / minor).ceil();
                let last_minor = ((end - origin) / zoom / minor).floor();
                for i in 0..=((last_minor - first_minor).max(0.0) as usize).min(2000) {
                    let value = (first_minor + i as f32) * minor;
                    let fraction = value / step;
                    if (fraction - fraction.round()).abs() < 0.0001 {
                        continue;
                    }
                    let length = if (fraction - fraction.floor() - 0.5).abs() < 0.0001 {
                        3.5
                    } else {
                        2.5
                    };
                    let screen = origin + value * zoom;
                    let segment = match axis {
                        Axis::Horizontal => [
                            egui::pos2(screen, top.bottom() - length),
                            egui::pos2(screen, top.bottom()),
                        ],
                        Axis::Vertical => [
                            egui::pos2(left.right() - length, screen),
                            egui::pos2(left.right(), screen),
                        ],
                    };
                    ruler_painter.line_segment(segment, (1.0, tick_color.gamma_multiply(0.7)));
                }
            }
        }
        let vertical_font = style.vertical_font();
        let gap = if style.compact_vertical { 6.0 } else { 8.0 };
        let first = ((start - origin) / zoom / step).ceil();
        let last = ((end - origin) / zoom / step).floor();
        for i in 0..=((last - first).max(0.0) as usize).min(200) {
            let value = (first + i as f32) * step;
            let screen = origin + value * zoom;
            let (a, b, text) = match axis {
                Axis::Horizontal => (
                    egui::pos2(screen, top.bottom() - 5.0),
                    egui::pos2(screen, top.bottom()),
                    egui::pos2(screen + 2.0, top.top() + 2.0),
                ),
                Axis::Vertical => (
                    egui::pos2(left.right() - 5.0, screen),
                    egui::pos2(left.right(), screen),
                    egui::pos2(left.right() - gap, {
                        let half = (vertical_font.size * 0.65).min(left.height() * 0.5);
                        screen.clamp(left.top() + half, left.bottom() - half)
                    }),
                ),
            };
            ruler_painter.line_segment([a, b], (1.0, tick_color));
            ruler_painter.text(
                text,
                match axis {
                    Axis::Horizontal => egui::Align2::LEFT_TOP,
                    Axis::Vertical => egui::Align2::RIGHT_CENTER,
                },
                format!("{value:.0}"),
                match axis {
                    Axis::Horizontal => style.font(),
                    Axis::Vertical => vertical_font.clone(),
                },
                text_color,
            );
        }
    }
    let corner = Rect::from_min_max(image.min - size, image.min).intersect(view);
    if corner.is_positive() {
        painter.rect_filled(corner, 0.0, background);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn compact_vertical_ruler_is_narrower_and_intermediate_ticks_are_optional() {
        let ctx = Context::default();
        let mut p = Project::new_untitled(1, 200, 160);
        p.canvas_state.guides.rulers = true;
        let compact = crate::config::settings::RulerSettings::default();
        let full = crate::config::settings::RulerSettings {
            compact_vertical: false,
            ..compact.clone()
        };
        assert!(compact.compact_vertical && compact.minor_ticks);
        assert!(ruler_size(&compact, 600).x < ruler_size(&full, 600).x);
        assert!(compact.vertical_font().size < compact.font().size);
        assert_eq!(full.vertical_font(), full.font());
        let mut counts = Vec::new();
        for enabled in [false, true] {
            let style = crate::config::settings::RulerSettings {
                minor_ticks: enabled,
                ..compact.clone()
            };
            set_appearance(&ctx, &style);
            let output = ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(640.0, 480.0),
                    )),
                    ..Default::default()
                },
                |ui| {
                    paint(
                        ui,
                        &p.canvas_state,
                        Rect::from_min_size(egui::pos2(100.0, 80.0), egui::vec2(400.0, 320.0)),
                        ui.clip_rect(),
                        2.0,
                    )
                },
            );
            counts.push(
                output
                    .shapes
                    .iter()
                    .filter(|s| matches!(s.shape, egui::Shape::LineSegment { .. }))
                    .count(),
            );
        }
        assert!(
            counts[1] > counts[0] * 2,
            "intermediate ticks must reach the painter"
        );
    }
    #[test]
    fn ruler_labels_share_a_right_edge_in_both_fonts_themes_and_sizes() {
        let ctx = Context::default();
        let mut p = Project::new_untitled(1, 200, 160);
        p.canvas_state.guides.rulers = true;
        let image = Rect::from_min_size(egui::pos2(160.0, 100.0), egui::vec2(400.0, 320.0));
        for dark in [false, true] {
            ctx.set_visuals(if dark {
                egui::Visuals::dark()
            } else {
                egui::Visuals::light()
            });
            for monospace in [false, true] {
                for font_size in [9.0, 18.0, 24.0] {
                    let style = crate::config::settings::RulerSettings {
                        monospace,
                        font_size,
                        text_color: Some([40, 150, 220]),
                        ..Default::default()
                    };
                    set_appearance(&ctx, &style);
                    let output = ctx.run_ui(
                        egui::RawInput {
                            screen_rect: Some(Rect::from_min_size(
                                egui::Pos2::ZERO,
                                egui::vec2(640.0, 480.0),
                            )),
                            ..Default::default()
                        },
                        |ui| paint(ui, &p.canvas_state, image, ui.clip_rect(), 2.0),
                    );
                    let labels: Vec<_> = output
                        .shapes
                        .iter()
                        .filter_map(|s| match &s.shape {
                            egui::Shape::Text(t)
                                if t.pos.x < image.left() && t.pos.y >= image.top() =>
                            {
                                Some(t)
                            }
                            _ => None,
                        })
                        .collect();
                    assert!(labels.len() >= 2);
                    for label in labels {
                        assert!(
                            (label.pos.x + label.galley.size().x - (image.left() - 6.0)).abs()
                                < 0.5
                        );
                        assert_eq!(label.fallback_color, egui::Color32::from_rgb(40, 150, 220));
                        assert!(label.pos.y >= image.top());
                    }
                }
            }
        }
        let large = crate::config::settings::RulerSettings {
            font_size: 24.0,
            ..Default::default()
        };
        assert!(ruler_size(&large, 1_000_000).x > ruler_size(&large, 100).x);
    }
    #[test]
    fn rulers_follow_canvas_edges_and_clip_instead_of_pinning_to_viewport() {
        assert!(!Guides::default().rulers, "rulers must be opt-in");
        let view = Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(640.0, 480.0));
        for (origin, size) in [
            (egui::pos2(100.0, 80.0), egui::vec2(400.0, 320.0)),
            (egui::pos2(180.0, 160.0), egui::vec2(200.0, 160.0)),
        ] {
            let image = Rect::from_min_size(origin, size);
            let (top, left) = rulers(image, view, egui::vec2(36.0, 20.0));
            assert_eq!(top.bottom(), image.top());
            assert_eq!(top.x_range(), image.x_range());
            assert_eq!(left.right(), image.left());
            assert_eq!(left.y_range(), image.y_range());
            assert!(!image.contains(top.center()));
            assert!(!image.contains(left.center()));
        }
        let (top, left) = rulers(
            Rect::from_min_max(egui::pos2(-100.0, -80.0), egui::pos2(500.0, 400.0)),
            view,
            egui::vec2(36.0, 20.0),
        );
        assert!(!top.is_positive());
        assert!(!left.is_positive());
        let (top, left) = rulers(
            Rect::from_min_max(egui::pos2(10.0, 10.0), egui::pos2(500.0, 400.0)),
            view,
            egui::vec2(36.0, 20.0),
        );
        assert_eq!(top.top(), view.top());
        assert_eq!(top.bottom(), 10.0);
        assert_eq!(left.left(), view.left());
        assert_eq!(left.right(), 10.0);
    }
    fn frame(
        ctx: &Context,
        p: &mut Project,
        events: Vec<egui::Event>,
        move_tool: bool,
        blocked: bool,
    ) -> bool {
        let mut handled = false;
        let _ = ctx.run_ui(
            egui::RawInput {
                screen_rect: Some(Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(640.0, 480.0),
                )),
                events,
                ..Default::default()
            },
            |ui| {
                begin(ctx, p.id);
                let image = Rect::from_min_size(egui::pos2(100.0, 80.0), egui::vec2(400.0, 320.0));
                handled = interact(
                    ui,
                    &mut p.canvas_state,
                    image,
                    ui.clip_rect(),
                    2.0,
                    move_tool,
                    blocked,
                );
                paint(ui, &p.canvas_state, image, ui.clip_rect(), 2.0);
            },
        );
        drain(ctx, p);
        handled
    }
    fn pointer(x: f32, y: f32, pressed: Option<bool>) -> Vec<egui::Event> {
        let pos = egui::pos2(x, y);
        let mut events = vec![egui::Event::PointerMoved(pos)];
        if let Some(pressed) = pressed {
            events.push(egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed,
                modifiers: Default::default(),
            });
        }
        events
    }
    fn escape() -> egui::Event {
        egui::Event::Key {
            key: egui::Key::Escape,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: Default::default(),
        }
    }
    #[test]
    fn fast_ruler_and_guide_drags_use_press_origin_when_events_share_a_frame() {
        let ctx = Context::default();
        let mut p = Project::new_untitled(1, 200, 160);
        p.canvas_state.guides.rulers = true;
        let mut events = pointer(180.0, 70.0, Some(true));
        events.extend(pointer(180.0, 180.0, None));
        assert!(frame(&ctx, &mut p, events, false, false));
        frame(
            &ctx,
            &mut p,
            pointer(180.0, 180.0, Some(false)),
            false,
            false,
        );
        assert_eq!(p.canvas_state.guides.items.len(), 1);
        assert_eq!(p.canvas_state.guides.items[0].position, 50.0);
        let mut events = pointer(200.0, 180.0, Some(true));
        events.extend(pointer(200.0, 220.0, None));
        assert!(frame(&ctx, &mut p, events, true, false));
        frame(
            &ctx,
            &mut p,
            pointer(200.0, 220.0, Some(false)),
            true,
            false,
        );
        assert_eq!(p.canvas_state.guides.items[0].position, 70.0);
        assert_eq!(p.history.undo_count(), 2);
        let ctx = Context::default();
        let mut p = Project::new_untitled(1, 200, 160);
        p.canvas_state.guides.rulers = true;
        let mut events = pointer(180.0, 70.0, Some(true));
        events.extend(pointer(180.0, 180.0, Some(false)));
        assert!(frame(&ctx, &mut p, events, false, false));
        assert_eq!(p.canvas_state.guides.items.len(), 1);
        assert_eq!(p.history.undo_count(), 1);
    }
    #[test]
    fn ruler_drag_is_one_undo_step_and_never_edits_pixels() {
        for dark in [false, true] {
            let ctx = Context::default();
            ctx.set_visuals(if dark {
                egui::Visuals::dark()
            } else {
                egui::Visuals::light()
            });
            let mut p = Project::new_untitled(1, 200, 160);
            p.canvas_state.guides.rulers = true;
            assert!(frame(
                &ctx,
                &mut p,
                pointer(180.0, 70.0, Some(true)),
                false,
                false
            ));
            frame(&ctx, &mut p, pointer(180.0, 180.0, None), false, false);
            assert!(p.canvas_state.guides.items.is_empty());
            assert_eq!(p.history.undo_count(), 0);
            frame(
                &ctx,
                &mut p,
                pointer(180.0, 180.0, Some(false)),
                false,
                false,
            );
            assert_eq!(p.canvas_state.guides.items[0].position, 50.0);
            assert_eq!(p.history.undo_count(), 1);
            let pixels = p.canvas_state.layers[0].pixels.to_rgba_image();
            p.history.undo(&mut p.canvas_state);
            assert!(p.canvas_state.guides.items.is_empty());
            p.history.redo(&mut p.canvas_state);
            assert_eq!(p.canvas_state.guides.items[0].position, 50.0);
            assert_eq!(p.canvas_state.layers[0].pixels.to_rgba_image(), pixels);
        }
    }
    #[test]
    fn guide_move_delete_lock_cancel_and_document_ownership() {
        let ctx = Context::default();
        let mut p = Project::new_untitled(1, 200, 160);
        p.canvas_state.guides.add(Axis::Horizontal, 50.0, 200, 160);
        assert!(!frame(
            &ctx,
            &mut p,
            pointer(220.0, 180.0, Some(true)),
            false,
            false
        ));
        frame(
            &ctx,
            &mut p,
            pointer(220.0, 180.0, Some(false)),
            false,
            false,
        );
        assert!(frame(
            &ctx,
            &mut p,
            pointer(220.0, 180.0, Some(true)),
            true,
            false
        ));
        frame(
            &ctx,
            &mut p,
            pointer(220.0, 200.0, Some(false)),
            true,
            false,
        );
        assert_eq!(p.canvas_state.guides.items[0].position, 60.0);
        frame(&ctx, &mut p, pointer(220.0, 200.0, Some(true)), true, false);
        frame(&ctx, &mut p, vec![escape()], true, false);
        frame(
            &ctx,
            &mut p,
            pointer(220.0, 220.0, Some(false)),
            true,
            false,
        );
        assert_eq!(p.canvas_state.guides.items[0].position, 60.0);
        p.canvas_state.guides.locked = true;
        assert!(!frame(
            &ctx,
            &mut p,
            pointer(220.0, 200.0, Some(true)),
            true,
            false
        ));
        frame(
            &ctx,
            &mut p,
            pointer(220.0, 200.0, Some(false)),
            true,
            false,
        );
        p.canvas_state.guides.locked = false;
        frame(&ctx, &mut p, pointer(220.0, 200.0, Some(true)), true, false);
        let mut other = Project::new_untitled(2, 200, 160);
        frame(
            &ctx,
            &mut other,
            pointer(220.0, 220.0, Some(false)),
            true,
            false,
        );
        assert!(other.canvas_state.guides.items.is_empty());
        assert_eq!(p.canvas_state.guides.items[0].position, 60.0);
        frame(&ctx, &mut p, pointer(220.0, 200.0, Some(true)), true, false);
        frame(
            &ctx,
            &mut p,
            pointer(220.0, 450.0, Some(false)),
            true,
            false,
        );
        assert!(p.canvas_state.guides.items.is_empty());
        p.history.undo(&mut p.canvas_state);
        assert_eq!(p.canvas_state.guides.items.len(), 1);
    }
    #[test]
    fn exact_position_dialog_uses_shared_numeric_and_retains_project_target() {
        let ctx = Context::default();
        let mut p = Project::new_untitled(1, 200, 160);
        begin(&ctx, p.id);
        ctx.data_mut(|d| {
            d.insert_temp(
                Id::new("new_guide_dialog"),
                NewGuide {
                    project: p.id,
                    axis: Axis::Vertical,
                    position: 37.5,
                    percent: true,
                },
            )
        });
        assert!(dialog_open(&ctx));
        let _ = ctx.run_ui(
            egui::RawInput {
                events: vec![egui::Event::Key {
                    key: egui::Key::Enter,
                    physical_key: None,
                    pressed: true,
                    repeat: false,
                    modifiers: Default::default(),
                }],
                ..Default::default()
            },
            |_| dialog(&ctx, &mut p),
        );
        drain(&ctx, &mut p);
        assert_eq!(p.canvas_state.guides.items[0].position, 75.0);
        assert!(!dialog_open(&ctx));
        ctx.data_mut(|d| {
            d.insert_temp(
                Id::new("new_guide_dialog"),
                NewGuide {
                    project: uuid::Uuid::new_v4(),
                    axis: Axis::Vertical,
                    position: 10.0,
                    percent: false,
                },
            )
        });
        dialog(&ctx, &mut p);
        assert!(!dialog_open(&ctx));
        assert_eq!(p.canvas_state.guides.items.len(), 1);
    }
}
