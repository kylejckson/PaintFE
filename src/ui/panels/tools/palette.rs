//! Deterministic, content-sized tool palettes. No tool changes its reading order.
use super::Tool;
use crate::config::settings::ToolsPaletteSettings;
use egui::{Pos2, Rect, Vec2, pos2, vec2};

pub(crate) const GROUPS: [&[(Tool, &str)]; 4] = [
    &[
        (Tool::Brush, "brush"),
        (Tool::Pencil, "pencil"),
        (Tool::Eraser, "eraser"),
        (Tool::Line, "line"),
        (Tool::Fill, "fill"),
        (Tool::Gradient, "gradient"),
    ],
    &[
        (Tool::RectangleSelect, "rectangle"),
        (Tool::EllipseSelect, "ellipse"),
        (Tool::Lasso, "lasso"),
        (Tool::MagicWand, "wand"),
        (Tool::MovePixels, "move_pixels"),
        (Tool::MoveSelection, "move_selection"),
    ],
    &[
        (Tool::CloneStamp, "clone"),
        (Tool::ContentAwareBrush, "content_aware"),
        (Tool::ColorRemover, "color_remover"),
        (Tool::Liquify, "liquify"),
        (Tool::MeshWarp, "mesh"),
        (Tool::PerspectiveCrop, "perspective"),
    ],
    &[
        (Tool::ColorPicker, "picker"),
        (Tool::Text, "text"),
        (Tool::Zoom, "zoom"),
        (Tool::Pan, "pan"),
    ],
];

#[derive(Clone, Debug)]
pub(crate) struct PaletteLayout {
    pub groups: [Vec<Rect>; 4],
    pub shapes: Rect,
    pub separators: Vec<[Pos2; 2]>,
    pub colors: Option<[Rect; 2]>,
    pub size: Vec2,
    pub columns: usize,
}

pub(crate) fn visible(settings: &ToolsPaletteSettings, id: &str) -> bool {
    !settings.hidden.iter().any(|hidden| hidden == id)
}

pub(crate) fn count(settings: &ToolsPaletteSettings, colors: bool) -> usize {
    GROUPS
        .iter()
        .flat_map(|g| g.iter())
        .filter(|(_, id)| visible(settings, id))
        .count()
        + usize::from(visible(settings, "shapes"))
        + 2 * usize::from(colors)
}

fn slot_capacity(settings: &ToolsPaletteSettings, colors: bool) -> usize {
    count(settings, colors) + usize::from(visible(settings, "shapes"))
}

/// One pass over the fixed catalogue; separators occupy gaps, never empty cells.
pub(crate) fn layout(
    settings: &ToolsPaletteSettings,
    colors: bool,
    columns: usize,
) -> PaletteLayout {
    let columns = columns.clamp(1, slot_capacity(settings, colors).max(1));
    let mut plan = PaletteLayout {
        groups: std::array::from_fn(|_| Vec::new()),
        shapes: Rect::NOTHING,
        separators: Vec::new(),
        colors: None,
        size: Vec2::ZERO,
        columns,
    };
    let mut slot = 0;
    let mut previous_group = None;
    let mut cell = |group: usize, desired_span: usize, plan: &mut PaletteLayout| {
        let span = desired_span.min(columns - slot % columns);
        let x = (slot % columns) as f32 * 32. + if columns == 1 { 8. } else { 0. };
        let y = (slot / columns) as f32 * 32.;
        if previous_group.is_some_and(|previous| previous != group) {
            if slot % columns == 0 {
                plan.separators
                    .push([pos2(0., y - 3.), pos2(columns as f32 * 32. - 6., y - 3.)]);
            } else {
                plan.separators
                    .push([pos2(x - 3., y + 2.), pos2(x - 3., y + 24.)]);
            }
        }
        let rect = Rect::from_min_size(pos2(x, y), vec2(span as f32 * 32. - 6., 26.));
        plan.size = plan.size.max(rect.max.to_vec2());
        previous_group = Some(group);
        slot += span;
        rect
    };
    for (group, tools) in GROUPS.iter().enumerate() {
        for (_, id) in *tools {
            if visible(settings, id) {
                let rect = cell(group, 1, &mut plan);
                plan.groups[group].push(rect);
            }
        }
    }
    if visible(settings, "shapes") {
        plan.shapes = cell(3, 2, &mut plan);
    }
    // Colours are a distinct control group, never orphaned tool-grid cells.
    if colors {
        let width = plan.size.x.max(if slot == 0 { 64. } else { 42. });
        let inline = columns >= 4 && slot > 0 && slot <= columns;
        let origin = if inline {
            let x = width + 19.;
            plan.separators
                .push([pos2(width + 8., 0.), pos2(width + 8., 32.)]);
            pos2(x, 0.)
        } else {
            let y = if slot > 0 { plan.size.y + 19. } else { 0. };
            if slot > 0 {
                plan.separators
                    .push([pos2(0., y - 10.), pos2(width, y - 10.)]);
            }
            pos2(((width - 64.) / 2.).max(0.), y)
        };
        let swatch = Rect::from_min_size(
            origin
                + if width < 58. && !inline {
                    vec2((width - 32.) / 2., 0.)
                } else {
                    Vec2::ZERO
                },
            Vec2::splat(32.),
        );
        let swap = if width < 58. && !inline {
            Rect::from_min_size(
                pos2((width - 24.) / 2., swatch.bottom() + 4.),
                vec2(24., 24.),
            )
        } else {
            let gap = if width < 64. && !inline { 4. } else { 8. };
            Rect::from_min_size(
                swatch.min + vec2(32. + gap, 0.),
                vec2(if width < 64. && !inline { 22. } else { 24. }, 32.),
            )
        };
        plan.size = plan.size.max(swatch.max.to_vec2()).max(swap.max.to_vec2());
        plan.colors = Some([swatch, swap]);
    }
    // The configure link and header remain usable when every tool is hidden.
    plan.size = plan.size.max(vec2(
        if slot == 0 && !colors { 140. } else { 42. },
        if slot == 0 { 24. } else { 26. },
    ));
    plan
}

/// Animate geometry once, then share it between painting, hit testing and the window.
/// Stable tool IDs preserve motion when visibility changes. No background timer runs
/// after settling, and the shared motion policy also respects reduced motion.
pub(crate) fn animated_layout(
    ctx: &egui::Context,
    settings: &ToolsPaletteSettings,
    mut plan: PaletteLayout,
) -> PaletteLayout {
    use crate::ui::polish::{self, MotionKind};
    let animate = |key: &str, axis: &str, target: f32| {
        let id = egui::Id::new(("tools_reflow", key, axis));
        if !settings.animate_resize {
            polish::reset(ctx, id);
        }
        polish::animate(ctx, id, target, MotionKind::Snapping)
    };
    let mut moving = false;
    let mut bounds = Vec2::ZERO;
    let mut cell = |key: &str, rect: &mut Rect| {
        let target = rect.min;
        *rect = Rect::from_min_size(
            pos2(animate(key, "x", target.x), animate(key, "y", target.y)),
            rect.size(),
        );
        moving |= rect.min != target;
        bounds = bounds.max(rect.max.to_vec2());
    };
    for (group, tools) in GROUPS.iter().enumerate() {
        for (rect, (_, key)) in plan.groups[group]
            .iter_mut()
            .zip(tools.iter().filter(|(_, key)| visible(settings, key)))
        {
            cell(key, rect);
        }
    }
    if plan.shapes.is_positive() {
        cell("shapes", &mut plan.shapes);
    }
    if let Some(colors) = &mut plan.colors {
        cell("primary", &mut colors[0]);
        cell("secondary", &mut colors[1]);
    }
    plan.size = vec2(
        animate("body", "width", plan.size.x),
        animate("body", "height", plan.size.y),
    )
    .max(bounds);
    // Category lines belong to settled gaps, not the paths between moving icons.
    if moving {
        plan.separators.clear();
    }
    plan
}

/// Choose the closest fitting footprint; retain the current columns near a boundary.
pub(crate) fn choose(
    settings: &ToolsPaletteSettings,
    colors: bool,
    requested: Vec2,
    max: Vec2,
    current: usize,
    axis: u8,
    chrome: impl Fn(Vec2) -> Vec2,
) -> PaletteLayout {
    let count = slot_capacity(settings, colors).max(1);
    let desired = match settings.mode.as_str() {
        "column" => 1,
        "row" => count,
        _ => settings.columns,
    };
    // A chosen preset must replace the previous footprint, rather than compete
    // against the distance to that footprint (a tall strip otherwise wins).
    if axis == 0 {
        let selected = layout(settings, colors, desired);
        let size = chrome(selected.size);
        if size.x <= max.x && size.y <= max.y {
            return selected;
        }
    }
    let score = |plan: &PaletteLayout| {
        let outer = chrome(plan.size);
        let overflow = (outer - max).max(Vec2::ZERO);
        let delta = outer - requested;
        let distance = match axis {
            1 => delta.x.abs(),
            2 => delta.y.abs(),
            _ => delta.length(),
        };
        let preference = if axis == 0 {
            (plan.columns as f32 - desired as f32).abs() * 80.
        } else {
            0.
        };
        overflow.length() * 10000. + distance + preference
    };
    let mut best = layout(settings, colors, 1);
    for columns in 2..=count {
        let next = layout(settings, colors, columns);
        if score(&next) < score(&best) {
            best = next;
        }
    }
    let previous = layout(settings, colors, current);
    if axis != 0 && score(&previous) <= score(&best) + 8. {
        previous
    } else {
        best
    }
}

pub(crate) fn header_height(_ctx: &egui::Context, _width: f32, _scale: f32) -> f32 {
    26.
}

/// Prefer one coherent header row. Shrink the title slightly for two columns;
/// omit it when the translated title cannot fit beside a full-size close button.
fn header_font(ctx: &egui::Context, width: f32, scale: f32) -> Option<egui::FontId> {
    if width < 58. {
        return None;
    }
    let title = crate::i18n::translate_english("TOOLS");
    for size in [10., 9., 8.] {
        let font = egui::FontId::proportional(size * scale);
        let measured =
            ctx.fonts_mut(|f| f.layout_no_wrap(title.clone(), font.clone(), egui::Color32::WHITE));
        if measured.size().x <= width - 26. {
            return Some(font);
        }
    }
    None
}

/// Header layout uses the same measured height as the outer footprint.
pub(crate) fn header(ui: &mut egui::Ui, theme: &crate::theme::Theme) -> (bool, egui::Response) {
    let width = ui.available_width();
    let height = header_height(ui.ctx(), width, theme.polish.text_scale);
    let font = header_font(ui.ctx(), width, theme.polish.text_scale);
    let compact = font.is_none();
    let (rect, response) = ui.allocate_exact_size(vec2(width, height), egui::Sense::click());
    let close = Rect::from_min_size(
        if compact {
            pos2(rect.center().x - 11., rect.center().y - 11.)
        } else {
            pos2(rect.right() - 22., rect.center().y - 11.)
        },
        Vec2::splat(22.),
    );
    if let Some(font) = font {
        let galley = ui.fonts_mut(|f| {
            f.layout_no_wrap(
                crate::i18n::translate_english("TOOLS"),
                font,
                theme.text_color,
            )
        });
        let origin = pos2(rect.left(), rect.center().y - galley.size().y / 2.);
        ui.painter().galley(origin, galley, theme.text_color);
    }
    let close_response = ui.interact(close, ui.id().with("palette_close"), egui::Sense::click());
    close_response.widget_info(|| {
        egui::WidgetInfo::labeled(
            egui::WidgetType::Button,
            ui.is_enabled(),
            crate::t!("ui.close"),
        )
    });
    crate::signal_widgets::paint_close_cross(ui, close, theme.text_muted);
    let drag = if height == 26. {
        Rect::from_min_max(rect.min, pos2(close.left() - 2., rect.bottom()))
    } else {
        Rect::from_min_max(rect.min, pos2(rect.right(), close.top()))
    };
    let drag_id = egui::Id::new(("floating_header_pointer", "Tools"));
    let mut previous = ui.ctx().data(|d| d.get_temp::<Pos2>(drag_id));
    let mut delta = Vec2::ZERO;
    for event in ui.input(|i| i.events.clone()) {
        match event {
            egui::Event::PointerButton {
                pos,
                pressed: true,
                button: egui::PointerButton::Primary,
                ..
            } if ui.is_enabled()
                && (drag.contains(pos)
                    || (compact && rect.contains(pos) && !close.contains(pos)))
                && ui.ctx().layer_id_at(pos) == Some(ui.layer_id()) =>
            {
                previous = Some(pos)
            }
            egui::Event::PointerMoved(pos) if previous.is_some() => {
                delta += pos - previous.unwrap();
                previous = Some(pos);
            }
            egui::Event::PointerButton {
                pressed: false,
                button: egui::PointerButton::Primary,
                ..
            }
            | egui::Event::PointerGone
            | egui::Event::WindowFocused(false) => previous = None,
            _ => {}
        }
    }
    ui.ctx().data_mut(|d| {
        if let Some(p) = previous {
            d.insert_temp(drag_id, p);
        } else {
            d.remove::<Pos2>(drag_id);
        }
        if delta != Vec2::ZERO {
            d.insert_temp(egui::Id::new(("floating_header_drag", "Tools")), delta);
        }
    });
    if drag.contains(ui.input(|i| i.pointer.hover_pos()).unwrap_or(Pos2::ZERO)) {
        ui.ctx().set_cursor_icon(egui::CursorIcon::Grab);
    }
    (close_response.clicked(), response)
}

#[derive(Clone, Copy)]
struct Drag {
    start: Pos2,
    rect: Rect,
    edges: u8,
}

pub(crate) fn resize(
    ctx: &egui::Context,
    response: &egui::Response,
    settings: &ToolsPaletteSettings,
    colors: bool,
    chrome: impl Fn(Vec2) -> Vec2,
) -> Option<(Rect, usize)> {
    let id = egui::Id::new("tools_palette_resize");
    let mut capture = ctx.data(|d| d.get_temp::<Drag>(id));
    let rect = response.rect;
    let (custom, opacity) = ctx
        .data(|d| {
            d.get_temp::<(Option<egui::Color32>, f32)>(egui::Id::new("resize_grip_appearance"))
        })
        .unwrap_or((None, 0.5));
    let color = custom
        .unwrap_or(ctx.global_style().visuals.weak_text_color())
        .gamma_multiply(opacity);
    for offset in [9., 13.] {
        ctx.layer_painter(response.layer_id).line_segment(
            [rect.max - vec2(offset, 3.), rect.max - vec2(3., offset)],
            egui::Stroke::new(1., color),
        );
    }
    let edges = |p: Pos2| -> u8 {
        if !rect.expand(5.).contains(p) {
            return 0;
        }
        let mut e = 0;
        if (p.x - rect.left()).abs() < 6. {
            e |= 1;
        } else if (p.x - rect.right()).abs() < 6. {
            e |= 2;
        }
        if (p.y - rect.top()).abs() < 5. {
            e |= 4;
        } else if (p.y - rect.bottom()).abs() < 6. {
            e |= 8;
        }
        // Keep the familiar, forgiving corner target.
        if Rect::from_min_max(rect.max - Vec2::splat(18.), rect.max).contains(p) {
            e = 10;
        }
        e
    };
    let hovered = ctx.input(|i| i.pointer.hover_pos()).map_or(0, edges);
    let cursor = match capture.map_or(hovered, |d| d.edges) {
        1 | 2 => Some(egui::CursorIcon::ResizeHorizontal),
        4 | 8 => Some(egui::CursorIcon::ResizeVertical),
        5 | 10 => Some(egui::CursorIcon::ResizeNwSe),
        6 | 9 => Some(egui::CursorIcon::ResizeNeSw),
        _ => None,
    };
    if let Some(cursor) = cursor {
        ctx.set_cursor_icon(cursor);
    }
    let mut result = None;
    for event in ctx.input(|i| i.events.clone()) {
        match event {
            egui::Event::PointerButton {
                pos,
                pressed: true,
                button: egui::PointerButton::Primary,
                ..
            } if response.enabled()
                && edges(pos) != 0
                && ctx.layer_id_at(pos) == Some(response.layer_id) =>
            {
                capture = Some(Drag {
                    start: pos,
                    rect,
                    edges: edges(pos),
                })
            }
            egui::Event::PointerMoved(pos)
            | egui::Event::PointerButton {
                pos,
                pressed: false,
                button: egui::PointerButton::Primary,
                ..
            } if capture.is_some() => {
                let d = capture.unwrap();
                let delta = pos - d.start;
                let size = (d.rect.size()
                    + vec2(
                        if d.edges & 1 != 0 {
                            -delta.x
                        } else if d.edges & 2 != 0 {
                            delta.x
                        } else {
                            0.
                        },
                        if d.edges & 4 != 0 {
                            -delta.y
                        } else if d.edges & 8 != 0 {
                            delta.y
                        } else {
                            0.
                        },
                    ))
                .max(vec2(42., 60.));
                let viewport = ctx.content_rect().shrink(6.);
                let max = vec2(
                    if d.edges & 1 != 0 {
                        d.rect.right() - viewport.left()
                    } else {
                        viewport.right() - d.rect.left()
                    },
                    if d.edges & 4 != 0 {
                        d.rect.bottom() - viewport.top()
                    } else {
                        viewport.bottom() - d.rect.top()
                    },
                );
                let axis = if d.edges & 3 == 0 {
                    2
                } else if d.edges & 12 == 0 {
                    1
                } else {
                    3
                };
                let plan = choose(settings, colors, size, max, settings.columns, axis, &chrome);
                let fitted = chrome(plan.size);
                let origin = pos2(
                    if d.edges & 1 != 0 {
                        d.rect.right() - fitted.x
                    } else {
                        d.rect.left()
                    },
                    if d.edges & 4 != 0 {
                        d.rect.bottom() - fitted.y
                    } else {
                        d.rect.top()
                    },
                );
                result = Some((Rect::from_min_size(origin, fitted), plan.columns));
                if matches!(event, egui::Event::PointerButton { pressed: false, .. }) {
                    capture = None;
                }
            }
            egui::Event::PointerGone | egui::Event::WindowFocused(false) => capture = None,
            _ => {}
        }
    }
    ctx.data_mut(|d| {
        if let Some(c) = capture {
            d.insert_temp(id, c);
        } else {
            d.remove::<Drag>(id);
        }
    });
    if let Some((target, _)) = result {
        ctx.layer_painter(response.layer_id).rect_stroke(
            target,
            5.,
            egui::Stroke::new(1., ctx.global_style().visuals.selection.bg_fill),
            egui::StrokeKind::Inside,
        );
        ctx.request_repaint();
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn shapes_span_two_slots_unless_only_one_fits() {
        for hidden in [
            vec![],
            vec!["pan".into()],
            vec!["brush".into(), "text".into()],
        ] {
            let settings = ToolsPaletteSettings {
                hidden,
                ..Default::default()
            };
            for columns in 1..=26 {
                for colors in [false, true] {
                    let plan = layout(&settings, colors, columns);
                    let before = GROUPS
                        .iter()
                        .flat_map(|group| group.iter())
                        .filter(|(_, id)| visible(&settings, id))
                        .count();
                    let remaining = plan.columns - before % plan.columns;
                    assert_eq!(plan.shapes.width(), if remaining >= 2 { 58. } else { 26. });
                    for rect in plan
                        .groups
                        .iter()
                        .flatten()
                        .chain(plan.colors.iter().flatten())
                    {
                        assert!(!plan.shapes.intersects(*rect));
                    }
                    let row_start = plan.shapes.left() % 32.;
                    assert!(row_start < 0.01 || plan.columns == 1);
                }
            }
        }
        let settings = ToolsPaletteSettings {
            mode: "row".into(),
            ..Default::default()
        };
        let row = choose(
            &settings,
            false,
            vec2(100., 500.),
            vec2(1800., 1000.),
            3,
            0,
            |s| s,
        );
        assert_eq!(row.shapes.width(), 58.);
        assert_eq!(row.size.y, 26.);
        let hidden = ToolsPaletteSettings {
            hidden: vec!["shapes".into()],
            ..Default::default()
        };
        assert!(!layout(&hidden, true, 3).shapes.is_positive());
    }
    #[test]
    fn narrow_header_is_short_and_icons_are_centered() {
        let ctx = egui::Context::default();
        let _ = ctx.run_ui(Default::default(), |ui| {
            let ctx = ui.ctx();
            let settings = ToolsPaletteSettings::default();
            let plan = layout(&settings, true, 1);
            assert_eq!(header_height(ctx, plan.size.x, 1.35), 26.);
            for rect in plan.groups.iter().flatten() {
                assert_eq!(rect.center().x, plan.size.x / 2.);
            }
            for columns in 1..=26 {
                for colors in [false, true] {
                    let plan = layout(&settings, colors, columns);
                    for scale in [0.85, 1., 1.35] {
                        assert_eq!(header_height(ctx, plan.size.x, scale), 26.);
                        if let Some(font) = header_font(ctx, plan.size.x, scale) {
                            let text = ctx.fonts_mut(|f| {
                                f.layout_no_wrap(
                                    crate::i18n::translate_english("TOOLS"),
                                    font,
                                    egui::Color32::WHITE,
                                )
                            });
                            assert!(text.size().x <= plan.size.x - 26.);
                        }
                    }
                }
            }
            assert!(
                header_font(ctx, 58., 1.).is_some(),
                "Two-column English title fits beside close"
            );
        });
    }

    #[test]
    fn colors_remain_a_separate_group_in_every_layout() {
        let settings = ToolsPaletteSettings::default();
        for columns in 1..=26 {
            let plain = layout(&settings, false, columns);
            let colors = layout(&settings, true, columns);
            assert_eq!(plain.groups, colors.groups);
            assert_eq!(plain.shapes, colors.shapes);
            let [swatch, swap] = colors.colors.unwrap();
            assert_eq!(swatch.size(), Vec2::splat(32.));
            assert!(!swatch.intersects(swap));
            if columns < 23 {
                assert!(swatch.top() >= plain.size.y + 19.);
                assert!(
                    colors
                        .separators
                        .iter()
                        .any(|line| { line[0].y == plain.size.y + 9. && line[1].y == line[0].y })
                );
            } else {
                assert_eq!(colors.size.y, 32.);
                assert!(swatch.left() >= plain.size.x + 19.);
            }
        }
    }

    #[test]
    fn reflow_has_intermediate_geometry_settles_and_can_be_disabled() {
        let ctx = egui::Context::default();
        let mut settings = ToolsPaletteSettings::default();
        // Old saved settings opt in without losing any other palette choices.
        assert!(
            serde_json::from_str::<ToolsPaletteSettings>("{}")
                .unwrap()
                .animate_resize
        );
        let sample = |time, columns, settings: &ToolsPaletteSettings| {
            let mut result = None;
            let _ = ctx.run_ui(
                egui::RawInput {
                    time: Some(time),
                    ..Default::default()
                },
                |ui| {
                    let ctx = ui.ctx();
                    result = Some(animated_layout(
                        ctx,
                        settings,
                        layout(settings, true, columns),
                    ));
                },
            );
            result.unwrap()
        };
        sample(0., 6, &settings);
        sample(0.01, 1, &settings);
        let midway = sample(0.05, 1, &settings);
        let target = layout(&settings, true, 1);
        assert_ne!(midway.groups[0][1], target.groups[0][1]);
        assert!(midway.separators.is_empty());
        for rect in midway
            .groups
            .iter()
            .flatten()
            .chain(midway.colors.iter().flatten())
        {
            assert!(Rect::from_min_size(Pos2::ZERO, midway.size).contains_rect(*rect));
        }
        let settled = sample(1., 1, &settings);
        assert_eq!(settled.groups, target.groups);
        assert_eq!(settled.size, target.size);
        settings.animate_resize = false;
        let immediate = sample(1.01, 6, &settings);
        assert_eq!(immediate.groups, layout(&settings, true, 6).groups);
        assert_eq!(immediate.size, layout(&settings, true, 6).size);
    }

    #[test]
    fn presets_replace_the_previous_shape() {
        let mut settings = ToolsPaletteSettings {
            mode: "grid".into(),
            columns: 6,
            ..Default::default()
        };
        let grid = choose(
            &settings,
            true,
            vec2(62., 856.),
            vec2(1800., 1000.),
            1,
            0,
            |s| s + vec2(20., 50.),
        );
        assert_eq!(grid.columns, 6);
        settings.mode = "row".into();
        let row = choose(
            &settings,
            true,
            vec2(62., 856.),
            vec2(1800., 1000.),
            1,
            0,
            |s| s + vec2(20., 50.),
        );
        assert_eq!(row.size.y, 32.);
    }
    #[test]
    fn fast_edge_resizes_preserve_opposite_anchor_and_release_capture() {
        for edges in [1, 2, 4, 8, 5, 6, 9, 10] {
            let ctx = egui::Context::default();
            let rect = Rect::from_min_max(pos2(100., 100.), pos2(300., 300.));
            let start = pos2(
                if edges & 1 != 0 {
                    101.
                } else if edges & 2 != 0 {
                    299.
                } else {
                    200.
                },
                if edges & 4 != 0 {
                    101.
                } else if edges & 8 != 0 {
                    299.
                } else {
                    200.
                },
            );
            let end = start + vec2(70., 90.);
            let raw = |events| egui::RawInput {
                screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(1200., 900.))),
                events,
                ..Default::default()
            };
            let settings = ToolsPaletteSettings::default();
            let mut result = None;
            let mut frame = |input| {
                let _ = ctx.run_ui(input, |ui| {
                    let response = ui.allocate_rect(rect, egui::Sense::hover());
                    result = resize(ui.ctx(), &response, &settings, false, |s| {
                        s + vec2(24., 50.)
                    });
                });
            };
            frame(raw(vec![]));
            frame(raw(vec![
                egui::Event::PointerMoved(start),
                egui::Event::PointerButton {
                    pos: start,
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::NONE,
                },
                egui::Event::PointerMoved(end),
                egui::Event::PointerButton {
                    pos: end,
                    button: egui::PointerButton::Primary,
                    pressed: false,
                    modifiers: egui::Modifiers::NONE,
                },
            ]));
            let (target, _) = result.unwrap_or_else(|| panic!("edges={edges}"));
            if edges & 1 != 0 {
                assert_eq!(target.right(), rect.right());
            } else {
                assert_eq!(target.left(), rect.left());
            }
            if edges & 4 != 0 {
                assert_eq!(target.bottom(), rect.bottom());
            } else {
                assert_eq!(target.top(), rect.top());
            }
            assert!(
                ctx.data(|d| d.get_temp::<Drag>(egui::Id::new("tools_palette_resize")))
                    .is_none()
            );
        }
    }
    #[test]
    fn every_visibility_and_column_arrangement_is_tight_and_nonoverlapping() {
        for hidden in [
            vec![],
            vec!["brush".into(), "shapes".into()],
            GROUPS
                .iter()
                .flat_map(|g| g.iter())
                .map(|(_, id)| id.to_string())
                .chain(["shapes".into()])
                .collect(),
        ] {
            let settings = ToolsPaletteSettings {
                hidden,
                ..Default::default()
            };
            for colors in [false, true] {
                for columns in 1..=26 {
                    let p = layout(&settings, colors, columns);
                    let rects: Vec<_> = p
                        .groups
                        .iter()
                        .flatten()
                        .copied()
                        .chain(p.shapes.is_positive().then_some(p.shapes))
                        .chain(p.colors.into_iter().flatten())
                        .collect();
                    assert_eq!(rects.len(), count(&settings, colors));
                    for (i, r) in rects.iter().enumerate() {
                        assert!(Rect::from_min_size(Pos2::ZERO, p.size).contains_rect(*r));
                        for other in &rects[..i] {
                            assert!(!r.intersects(*other));
                        }
                    }
                    if !rects.is_empty() {
                        assert_eq!(
                            p.size.y,
                            rects.iter().map(|r| r.bottom()).fold(0., f32::max)
                        );
                    }
                }
            }
        }
    }
    #[test]
    fn height_and_width_drags_choose_fitting_layouts_without_scrollbar() {
        let settings = ToolsPaletteSettings::default();
        for w in (60..900).step_by(7) {
            for h in [100., 160., 250., 600.] {
                let p = choose(
                    &settings,
                    false,
                    vec2(w as f32, h),
                    vec2(950., 850.),
                    3,
                    3,
                    |s| s + vec2(24., 50.),
                );
                assert!(p.size.x + 24. <= 950. && p.size.y + 50. <= 850.);
            }
        }
        let narrow = choose(
            &settings,
            false,
            vec2(60., 800.),
            vec2(950., 850.),
            3,
            1,
            |s| s + vec2(24., 50.),
        );
        assert_eq!(narrow.columns, 1);
        let wide = choose(
            &settings,
            false,
            vec2(850., 85.),
            vec2(950., 850.),
            3,
            2,
            |s| s + vec2(24., 50.),
        );
        assert_eq!(wide.size.y, 26.);
    }
}
