// =============================================================================
// Visual regression tests — Shape Rendering (SDF‐based)
// =============================================================================
//
// Each test creates a PlacedShape, rasterizes it via `rasterize_shape`, and
// compares the output buffer against a golden PNG.
//
// Run with GENERATE_GOLDEN=1 to create/update golden files:
//   GENERATE_GOLDEN=1 cargo test --test visual_shapes

mod common;

use common::*;
use image::RgbaImage;
use paintfe::ops::shapes::*;

/// Convert the rasterize_shape output into a full-canvas RgbaImage for comparison.
fn rasterize_to_canvas(placed: &PlacedShape, w: u32, h: u32) -> RgbaImage {
    let (buf, buf_w, buf_h, off_x, off_y) = rasterize_shape(placed, w, h);
    let mut canvas = RgbaImage::new(w, h);
    for row in 0..buf_h {
        for col in 0..buf_w {
            let cx = off_x + col as i32;
            let cy = off_y + row as i32;
            if cx >= 0 && cy >= 0 && (cx as u32) < w && (cy as u32) < h {
                let idx = ((row * buf_w + col) * 4) as usize;
                if idx + 3 < buf.len() {
                    let a = buf[idx + 3];
                    if a > 0 {
                        canvas.put_pixel(
                            cx as u32,
                            cy as u32,
                            image::Rgba([buf[idx], buf[idx + 1], buf[idx + 2], a]),
                        );
                    }
                }
            }
        }
    }
    canvas
}

/// Helper: create a standard centered shape on 128×128 canvas.
fn make_shape(kind: ShapeKind, fill: ShapeFillMode) -> PlacedShape {
    PlacedShape {
        cx: 64.0,
        cy: 64.0,
        hw: 40.0,
        hh: 40.0,
        rotation: 0.0,
        kind,
        fill_mode: fill,
        outline_width: 3.0,
        primary_color: [255, 80, 80, 255],   // red outline
        secondary_color: [80, 80, 255, 255], // blue fill
        anti_alias: true,
        corner_radius: 0.0,
        handle_dragging: None,
        drag_offset: [0.0, 0.0],
        drag_anchor: [0.0, 0.0],
        rotate_start_angle: 0.0,
        rotate_start_rotation: 0.0,
    }
}

const W: u32 = 128;
const H: u32 = 128;

// =============================================================================
// Outline shapes
// =============================================================================

macro_rules! shape_outline_test {
    ($name:ident, $kind:expr) => {
        #[test]
        fn $name() {
            let p = make_shape($kind, ShapeFillMode::Outline);
            let result = rasterize_to_canvas(&p, W, H);
            assert_golden("shapes", concat!(stringify!($name)), &result);
        }
    };
}

shape_outline_test!(ellipse_outline, ShapeKind::Ellipse);
shape_outline_test!(rectangle_outline, ShapeKind::Rectangle);
shape_outline_test!(triangle_outline, ShapeKind::Triangle);
shape_outline_test!(pentagon_outline, ShapeKind::Pentagon);
shape_outline_test!(hexagon_outline, ShapeKind::Hexagon);
shape_outline_test!(octagon_outline, ShapeKind::Octagon);
shape_outline_test!(cross_outline, ShapeKind::Cross);
shape_outline_test!(heart_outline, ShapeKind::Heart);
shape_outline_test!(star5_outline, ShapeKind::Star5);

// =============================================================================
// Filled shapes (fill + outline)
// =============================================================================

macro_rules! shape_filled_test {
    ($name:ident, $kind:expr) => {
        #[test]
        fn $name() {
            let p = make_shape($kind, ShapeFillMode::Both);
            let result = rasterize_to_canvas(&p, W, H);
            assert_golden("shapes", concat!(stringify!($name)), &result);
        }
    };
}

shape_filled_test!(ellipse_filled, ShapeKind::Ellipse);
shape_filled_test!(rectangle_filled, ShapeKind::Rectangle);
shape_filled_test!(triangle_filled, ShapeKind::Triangle);
shape_filled_test!(pentagon_filled, ShapeKind::Pentagon);
shape_filled_test!(hexagon_filled, ShapeKind::Hexagon);
shape_filled_test!(heart_filled, ShapeKind::Heart);

// =============================================================================
// Rounded rectangle
// =============================================================================

#[test]
fn rounded_rect_outline() {
    let mut p = make_shape(ShapeKind::RoundedRect, ShapeFillMode::Outline);
    p.corner_radius = 12.0;
    let result = rasterize_to_canvas(&p, W, H);
    assert_golden("shapes", "rounded_rect_outline", &result);
}

#[test]
fn rounded_rect_filled() {
    let mut p = make_shape(ShapeKind::RoundedRect, ShapeFillMode::Both);
    p.corner_radius = 12.0;
    let result = rasterize_to_canvas(&p, W, H);
    assert_golden("shapes", "rounded_rect_filled", &result);
}

// =============================================================================
// Rotation
// =============================================================================

#[test]
fn rectangle_rotated_45() {
    let mut p = make_shape(ShapeKind::Rectangle, ShapeFillMode::Both);
    p.rotation = std::f32::consts::FRAC_PI_4; // 45°
    let result = rasterize_to_canvas(&p, W, H);
    assert_golden("shapes", "rectangle_rotated_45", &result);
}

// =============================================================================
// Fill-only mode
// =============================================================================

#[test]
fn ellipse_fill_only() {
    let p = make_shape(ShapeKind::Ellipse, ShapeFillMode::Filled);
    let result = rasterize_to_canvas(&p, W, H);
    assert_golden("shapes", "ellipse_fill_only", &result);
}

// =============================================================================
// No anti-alias
// =============================================================================

#[test]
fn rectangle_no_aa() {
    let mut p = make_shape(ShapeKind::Rectangle, ShapeFillMode::Both);
    p.anti_alias = false;
    let result = rasterize_to_canvas(&p, W, H);
    assert_golden("shapes", "rectangle_no_aa", &result);
}
