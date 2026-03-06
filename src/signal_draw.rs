//! Low-level custom painting helpers for the Signal Grid design language.
//!
//! These functions draw directly onto an egui `Painter` and are used by
//! both `signal_widgets` and ad-hoc UI code throughout the app.

use eframe::egui::{self, Color32, Mesh, Pos2, Rect, Rounding, Stroke};

use crate::theme::Theme;

// ============================================================================
// Gradient divider
// ============================================================================

/// Draw a horizontal gradient-fade divider: transparent → color → transparent.
///
/// Replaces `ui.separator()` with a visually softer line that fades at the edges.
/// Height is 1 logical pixel.
pub fn draw_gradient_divider(painter: &egui::Painter, rect: Rect, color: Color32) {
    // Build a mesh with 6 vertices (left-transparent, left-mid, center-top,
    // center-bottom, right-mid, right-transparent) forming a horizontal strip.
    let y_top = rect.center().y - 0.5;
    let y_bot = rect.center().y + 0.5;
    let x_min = rect.left();
    let x_max = rect.right();
    let x_mid_l = x_min + rect.width() * 0.15;
    let x_mid_r = x_max - rect.width() * 0.15;

    let transparent = Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), 0);

    let mut mesh = Mesh::default();
    // 8 vertices: 4 on top edge, 4 on bottom edge
    // left edge (transparent)
    mesh.colored_vertex(Pos2::new(x_min, y_top), transparent); // 0
    mesh.colored_vertex(Pos2::new(x_min, y_bot), transparent); // 1
    // 15% in (full color)
    mesh.colored_vertex(Pos2::new(x_mid_l, y_top), color); // 2
    mesh.colored_vertex(Pos2::new(x_mid_l, y_bot), color); // 3
    // 85% in (full color)
    mesh.colored_vertex(Pos2::new(x_mid_r, y_top), color); // 4
    mesh.colored_vertex(Pos2::new(x_mid_r, y_bot), color); // 5
    // right edge (transparent)
    mesh.colored_vertex(Pos2::new(x_max, y_top), transparent); // 6
    mesh.colored_vertex(Pos2::new(x_max, y_bot), transparent); // 7

    // 3 quads = 6 triangles
    // Left fade: verts 0,1,2,3
    mesh.add_triangle(0, 1, 2);
    mesh.add_triangle(2, 1, 3);
    // Center solid: verts 2,3,4,5
    mesh.add_triangle(2, 3, 4);
    mesh.add_triangle(4, 3, 5);
    // Right fade: verts 4,5,6,7
    mesh.add_triangle(4, 5, 6);
    mesh.add_triangle(6, 5, 7);

    painter.add(egui::Shape::mesh(mesh));
}

// ============================================================================
// Grid texture
// ============================================================================

/// Draw subtle grid lines on a rect (like the website's `.grid-bg`).
///
/// `cell_size` is in logical pixels (typically 40.0).
/// `line_color` should be very low-alpha (e.g. `rgba(255,255,255,8)` dark mode).
///
/// Only draws lines within the visible `clip_rect` for performance.
pub fn draw_grid_texture(painter: &egui::Painter, rect: Rect, cell_size: f32, line_color: Color32) {
    if cell_size < 2.0 {
        return;
    }
    let clip = painter.clip_rect().intersect(rect);
    if clip.is_negative() {
        return;
    }

    let stroke = Stroke::new(1.0, line_color);

    // Vertical lines
    let x_start = (clip.left() / cell_size).floor() * cell_size;
    let mut x = x_start;
    while x <= clip.right() {
        if x >= clip.left() {
            painter.line_segment(
                [Pos2::new(x, clip.top()), Pos2::new(x, clip.bottom())],
                stroke,
            );
        }
        x += cell_size;
    }

    // Horizontal lines
    let y_start = (clip.top() / cell_size).floor() * cell_size;
    let mut y = y_start;
    while y <= clip.bottom() {
        if y >= clip.top() {
            painter.line_segment(
                [Pos2::new(clip.left(), y), Pos2::new(clip.right(), y)],
                stroke,
            );
        }
        y += cell_size;
    }
}

// ============================================================================
// Glow rect
// ============================================================================

/// Draw a colored glow behind an element (faking CSS `box-shadow` with color).
///
/// `expansion` controls how far outside `rect` the glow extends.
/// `rounding` is the rounding of the element the glow surrounds (glow uses
/// `rounding + expansion` for smooth falloff).
pub fn draw_glow_rect(
    painter: &egui::Painter,
    rect: Rect,
    color: Color32,
    expansion: f32,
    rounding: f32,
) {
    if color.a() == 0 || expansion <= 0.0 {
        return;
    }
    let glow_rect = rect.expand(expansion);
    painter.rect_filled(glow_rect, Rounding::same(rounding + expansion), color);
}

// ============================================================================
// Accent line
// ============================================================================

/// Draw a thin horizontal accent-colored line (full width of `rect`, 1px tall).
///
/// Used below toolbars, menu bars, panel titles for subtle color emphasis.
pub fn draw_accent_line(painter: &egui::Painter, rect: Rect, accent: Color32) {
    let y = rect.bottom();
    painter.line_segment(
        [Pos2::new(rect.left(), y), Pos2::new(rect.right(), y)],
        Stroke::new(1.0, accent),
    );
}

// ============================================================================
// Frosted panel background
// ============================================================================

/// Draw a semi-transparent filled rect with rounding — approximation of
/// frosted glass (no real blur, but the alpha + dark fill looks convincing).
pub fn draw_frosted_panel_bg(
    painter: &egui::Painter,
    rect: Rect,
    rounding: f32,
    bg_color: Color32,
    alpha: u8,
) {
    let fill = Color32::from_rgba_unmultiplied(bg_color.r(), bg_color.g(), bg_color.b(), alpha);
    painter.rect_filled(rect, Rounding::same(rounding), fill);
}

// ============================================================================
// Pill container
// ============================================================================

/// Draw the pill-shaped container background for tab bars.
///
/// `bg2` fill, 1px border, `Rounding::same(10.0)`, 4px padding inside.
pub fn draw_pill_container(painter: &egui::Painter, rect: Rect, theme: &Theme) {
    painter.rect(
        rect,
        Rounding::same(10.0),
        theme.bg2,
        Stroke::new(1.0, theme.border_color),
    );
}
