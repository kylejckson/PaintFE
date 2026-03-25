// =============================================================================
// Integration tests — Tool stroke rendering
// =============================================================================
//
// Tests brush, eraser, and line drawing via ToolsPanel's public draw methods.
// Each test renders directly to a TiledImage and compares against golden PNGs.

mod common;

use common::*;
use image::{GrayImage, Rgba, RgbaImage};
use paintfe::canvas::{CanvasState, Layer, TiledImage};
use paintfe::components::tools::ToolsPanel;

/// Helper: create a ToolsPanel, set size/hardness/AA, rebuild LUT, return it.
fn make_brush(size: f32, hardness: f32, anti_aliased: bool) -> ToolsPanel {
    let mut tp = ToolsPanel::default();
    tp.properties.size = size;
    tp.properties.hardness = hardness;
    tp.properties.anti_aliased = anti_aliased;
    tp.rebuild_brush_lut();
    tp
}

/// White color in f32 format.
const WHITE_F32: [f32; 4] = [1.0, 1.0, 1.0, 1.0];
/// Black color in f32 format.
const BLACK_F32: [f32; 4] = [0.0, 0.0, 0.0, 1.0];
/// Red color in f32 format.
const RED_F32: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
/// Semi-transparent blue.
const BLUE_SEMI_F32: [f32; 4] = [0.0, 0.0, 1.0, 0.5];

const W: u32 = 64;
const H: u32 = 64;

/// Create a blank (transparent) TiledImage.
fn blank_tile(w: u32, h: u32) -> TiledImage {
    let img = RgbaImage::from_pixel(w, h, Rgba([0, 0, 0, 0]));
    TiledImage::from_rgba_image(&img)
}

/// Create a white TiledImage (opaque background).
fn white_tile(w: u32, h: u32) -> TiledImage {
    let img = RgbaImage::from_pixel(w, h, Rgba([255, 255, 255, 255]));
    TiledImage::from_rgba_image(&img)
}

/// Extract full RgbaImage from TiledImage.
fn tile_to_image(tile: &TiledImage, w: u32, h: u32) -> RgbaImage {
    let mut img = RgbaImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let p = tile.get_pixel(x, y);
            img.put_pixel(x, y, *p);
        }
    }
    img
}

// =============================================================================
// Brush tool — circle stamps
// =============================================================================

#[test]
fn brush_circle_center() {
    let tp = make_brush(20.0, 1.0, true);
    let mut tile = blank_tile(W, H);
    tp.draw_circle_no_dirty(
        &mut tile,
        W,
        H,
        (32.0, 32.0),
        false,
        false,
        BLACK_F32,
        WHITE_F32,
        None,
    );
    let img = tile_to_image(&tile, W, H);
    assert_golden("tools", "brush_circle_center", &img);
}

#[test]
fn brush_circle_soft() {
    let tp = make_brush(30.0, 0.0, true);
    let mut tile = blank_tile(W, H);
    tp.draw_circle_no_dirty(
        &mut tile,
        W,
        H,
        (32.0, 32.0),
        false,
        false,
        BLACK_F32,
        WHITE_F32,
        None,
    );
    let img = tile_to_image(&tile, W, H);
    assert_golden("tools", "brush_circle_soft", &img);
}

#[test]
fn brush_circle_hard() {
    let tp = make_brush(20.0, 1.0, false);
    let mut tile = blank_tile(W, H);
    tp.draw_circle_no_dirty(
        &mut tile,
        W,
        H,
        (32.0, 32.0),
        false,
        false,
        BLACK_F32,
        WHITE_F32,
        None,
    );
    let img = tile_to_image(&tile, W, H);
    assert_golden("tools", "brush_circle_hard", &img);
}

#[test]
fn brush_circle_tiny() {
    let tp = make_brush(3.0, 1.0, true);
    let mut tile = blank_tile(W, H);
    tp.draw_circle_no_dirty(
        &mut tile,
        W,
        H,
        (32.0, 32.0),
        false,
        false,
        RED_F32,
        WHITE_F32,
        None,
    );
    let img = tile_to_image(&tile, W, H);
    assert_golden("tools", "brush_circle_tiny", &img);
}

#[test]
fn brush_circle_large() {
    let tp = make_brush(60.0, 0.5, true);
    let mut tile = blank_tile(W, H);
    tp.draw_circle_no_dirty(
        &mut tile,
        W,
        H,
        (32.0, 32.0),
        false,
        false,
        BLACK_F32,
        WHITE_F32,
        None,
    );
    let img = tile_to_image(&tile, W, H);
    assert_golden("tools", "brush_circle_large", &img);
}

#[test]
fn brush_semi_transparent() {
    let tp = make_brush(20.0, 1.0, true);
    let mut tile = blank_tile(W, H);
    tp.draw_circle_no_dirty(
        &mut tile,
        W,
        H,
        (32.0, 32.0),
        false,
        false,
        BLUE_SEMI_F32,
        WHITE_F32,
        None,
    );
    let img = tile_to_image(&tile, W, H);
    assert_golden("tools", "brush_semi_transparent", &img);
}

#[test]
fn brush_secondary_color() {
    let tp = make_brush(20.0, 1.0, true);
    let mut tile = blank_tile(W, H);
    tp.draw_circle_no_dirty(
        &mut tile,
        W,
        H,
        (32.0, 32.0),
        false,
        true, // use_secondary
        BLACK_F32,
        RED_F32,
        None,
    );
    let img = tile_to_image(&tile, W, H);
    assert_golden("tools", "brush_secondary_color", &img);
}

// =============================================================================
// Eraser tool
// =============================================================================

#[test]
fn eraser_circle() {
    let tp = make_brush(20.0, 1.0, true);
    let mut tile = white_tile(W, H);
    tp.draw_circle_no_dirty(
        &mut tile,
        W,
        H,
        (32.0, 32.0),
        true, // is_eraser
        false,
        BLACK_F32,
        WHITE_F32,
        None,
    );
    let img = tile_to_image(&tile, W, H);
    assert_golden("tools", "eraser_circle", &img);
}

#[test]
fn eraser_soft() {
    let tp = make_brush(30.0, 0.0, true);
    let mut tile = white_tile(W, H);
    tp.draw_circle_no_dirty(
        &mut tile,
        W,
        H,
        (32.0, 32.0),
        true,
        false,
        BLACK_F32,
        WHITE_F32,
        None,
    );
    let img = tile_to_image(&tile, W, H);
    assert_golden("tools", "eraser_soft", &img);
}

// =============================================================================
// Line drawing
// =============================================================================

#[test]
fn line_horizontal() {
    let mut tp = make_brush(8.0, 1.0, true);
    let mut tile = blank_tile(W, H);
    tp.draw_line_no_dirty(
        &mut tile,
        W,
        H,
        (4.0, 32.0),
        (60.0, 32.0),
        false,
        false,
        BLACK_F32,
        WHITE_F32,
        None,
    );
    let img = tile_to_image(&tile, W, H);
    assert_golden("tools", "line_horizontal", &img);
}

#[test]
fn line_vertical() {
    let mut tp = make_brush(8.0, 1.0, true);
    let mut tile = blank_tile(W, H);
    tp.draw_line_no_dirty(
        &mut tile,
        W,
        H,
        (32.0, 4.0),
        (32.0, 60.0),
        false,
        false,
        BLACK_F32,
        WHITE_F32,
        None,
    );
    let img = tile_to_image(&tile, W, H);
    assert_golden("tools", "line_vertical", &img);
}

#[test]
fn line_diagonal() {
    let mut tp = make_brush(6.0, 0.8, true);
    let mut tile = blank_tile(W, H);
    tp.draw_line_no_dirty(
        &mut tile,
        W,
        H,
        (4.0, 4.0),
        (60.0, 60.0),
        false,
        false,
        BLACK_F32,
        WHITE_F32,
        None,
    );
    let img = tile_to_image(&tile, W, H);
    assert_golden("tools", "line_diagonal", &img);
}

#[test]
fn line_soft_thick() {
    let mut tp = make_brush(16.0, 0.3, true);
    let mut tile = blank_tile(W, H);
    tp.draw_line_no_dirty(
        &mut tile,
        W,
        H,
        (10.0, 50.0),
        (54.0, 10.0),
        false,
        false,
        RED_F32,
        WHITE_F32,
        None,
    );
    let img = tile_to_image(&tile, W, H);
    assert_golden("tools", "line_soft_thick", &img);
}

#[test]
fn line_eraser() {
    let mut tp = make_brush(10.0, 1.0, true);
    let mut tile = white_tile(W, H);
    tp.draw_line_no_dirty(
        &mut tile,
        W,
        H,
        (4.0, 32.0),
        (60.0, 32.0),
        true, // eraser
        false,
        BLACK_F32,
        WHITE_F32,
        None,
    );
    let img = tile_to_image(&tile, W, H);
    assert_golden("tools", "line_eraser", &img);
}

// =============================================================================
// Selection mask interaction
// =============================================================================

#[test]
fn brush_with_selection_mask() {
    let tp = make_brush(40.0, 1.0, true);
    let mut tile = blank_tile(W, H);

    // Create a selection mask: only the left half is selected
    let mut mask = GrayImage::new(W, H);
    for y in 0..H {
        for x in 0..W {
            let v = if x < W / 2 { 255 } else { 0 };
            mask.put_pixel(x, y, image::Luma([v]));
        }
    }

    tp.draw_circle_no_dirty(
        &mut tile,
        W,
        H,
        (32.0, 32.0), // center of canvas — straddles the mask edge
        false,
        false,
        BLACK_F32,
        WHITE_F32,
        Some(&mask),
    );
    let img = tile_to_image(&tile, W, H);
    assert_golden("tools", "brush_with_selection_mask", &img);
}

// =============================================================================
// Multiple stamps (stroke simulation)
// =============================================================================

#[test]
fn stroke_multiple_stamps() {
    let tp = make_brush(10.0, 0.8, true);
    let mut tile = blank_tile(W, H);
    // Simulate a short stroke by stamping along a path
    for i in 0..8 {
        let x = 8.0 + i as f32 * 7.0;
        let y = 32.0;
        tp.draw_circle_no_dirty(
            &mut tile,
            W,
            H,
            (x, y),
            false,
            false,
            BLACK_F32,
            WHITE_F32,
            None,
        );
    }
    let img = tile_to_image(&tile, W, H);
    assert_golden("tools", "stroke_multiple_stamps", &img);
}

// =============================================================================
// Edge cases
// =============================================================================

#[test]
fn brush_at_origin() {
    let tp = make_brush(10.0, 1.0, true);
    let mut tile = blank_tile(W, H);
    tp.draw_circle_no_dirty(
        &mut tile,
        W,
        H,
        (0.0, 0.0),
        false,
        false,
        BLACK_F32,
        WHITE_F32,
        None,
    );
    let img = tile_to_image(&tile, W, H);
    assert_golden("tools", "brush_at_origin", &img);
}

#[test]
fn brush_at_corner() {
    let tp = make_brush(20.0, 1.0, true);
    let mut tile = blank_tile(W, H);
    tp.draw_circle_no_dirty(
        &mut tile,
        W,
        H,
        (63.0, 63.0),
        false,
        false,
        BLACK_F32,
        WHITE_F32,
        None,
    );
    let img = tile_to_image(&tile, W, H);
    assert_golden("tools", "brush_at_corner", &img);
}

#[test]
fn line_zero_length() {
    // A zero-length line should produce just a single circle stamp
    let mut tp = make_brush(12.0, 1.0, true);
    let mut tile = blank_tile(W, H);
    tp.draw_line_no_dirty(
        &mut tile,
        W,
        H,
        (32.0, 32.0),
        (32.0, 32.0),
        false,
        false,
        BLACK_F32,
        WHITE_F32,
        None,
    );
    let img = tile_to_image(&tile, W, H);
    assert_golden("tools", "line_zero_length", &img);
}

// =============================================================================
// Blend modes via brush
// =============================================================================

#[test]
fn brush_dodge_mode() {
    use paintfe::components::tools::BrushMode;

    let mut tp = make_brush(24.0, 1.0, true);
    tp.properties.brush_mode = BrushMode::Dodge;
    // Dodge mode works on existing pixels, so start with a colored tile
    let mut tile = TiledImage::from_rgba_image(&create_test_gradient(W, H));
    tp.draw_circle_no_dirty(
        &mut tile,
        W,
        H,
        (32.0, 32.0),
        false,
        false,
        BLACK_F32,
        WHITE_F32,
        None,
    );
    let img = tile_to_image(&tile, W, H);
    assert_golden("tools", "brush_dodge_mode", &img);
}

#[test]
fn brush_burn_mode() {
    use paintfe::components::tools::BrushMode;

    let mut tp = make_brush(24.0, 1.0, true);
    tp.properties.brush_mode = BrushMode::Burn;
    let mut tile = TiledImage::from_rgba_image(&create_test_gradient(W, H));
    tp.draw_circle_no_dirty(
        &mut tile,
        W,
        H,
        (32.0, 32.0),
        false,
        false,
        BLACK_F32,
        WHITE_F32,
        None,
    );
    let img = tile_to_image(&tile, W, H);
    assert_golden("tools", "brush_burn_mode", &img);
}

// =============================================================================
// Pencil tool (aliased hard brush)
// =============================================================================

#[test]
fn pencil_circle() {
    // Pencil = hard brush with no anti-aliasing
    let tp = make_brush(12.0, 1.0, false);
    let mut tile = blank_tile(W, H);
    tp.draw_circle_no_dirty(
        &mut tile,
        W,
        H,
        (32.0, 32.0),
        false,
        false,
        BLACK_F32,
        WHITE_F32,
        None,
    );
    let img = tile_to_image(&tile, W, H);
    // Should be aliased — all alpha values are either 0 or 255
    let mut all_binary = true;
    for p in img.pixels() {
        if p[3] != 0 && p[3] != 255 {
            all_binary = false;
            break;
        }
    }
    assert!(
        all_binary,
        "pencil (no AA) should produce only 0 or 255 alpha"
    );
    assert_golden("tools", "pencil_circle", &img);
}

#[test]
fn pencil_line() {
    let mut tp = make_brush(4.0, 1.0, false);
    let mut tile = blank_tile(W, H);
    tp.draw_line_no_dirty(
        &mut tile,
        W,
        H,
        (4.0, 4.0),
        (60.0, 60.0),
        false,
        false,
        RED_F32,
        WHITE_F32,
        None,
    );
    let img = tile_to_image(&tile, W, H);
    assert_golden("tools", "pencil_line", &img);
}

// =============================================================================
// Color picker (read pixel from the canvas)
// =============================================================================

#[test]
fn color_picker_reads_layer_pixel() {
    let state = CanvasState::new(32, 32);
    // Background is white
    let px = state.layers[0].pixels.get_pixel(16, 16);
    assert_eq!(*px, Rgba([255, 255, 255, 255]));
}

#[test]
fn color_picker_reads_composited() {
    let mut state = CanvasState::new(32, 32);
    let mut red_layer = Layer::new("Red".into(), 32, 32, Rgba([0, 0, 0, 0]));
    red_layer.pixels =
        TiledImage::from_rgba_image(&RgbaImage::from_pixel(32, 32, Rgba([255, 0, 0, 255])));
    state.layers.push(red_layer);

    let comp = state.composite();
    let px = *comp.get_pixel(16, 16);
    assert_eq!(px, Rgba([255, 0, 0, 255]), "composite shows red on top");
}

#[test]
fn color_picker_reads_painted_pixel() {
    let tp = make_brush(10.0, 1.0, true);
    let mut tile = blank_tile(W, H);
    tp.draw_circle_no_dirty(
        &mut tile,
        W,
        H,
        (32.0, 32.0),
        false,
        false,
        RED_F32,
        WHITE_F32,
        None,
    );
    // Center of the brush stamp should be red
    let center = tile.get_pixel(32, 32);
    assert_eq!(center[0], 255, "red channel at brush center");
    assert_eq!(center[1], 0, "green channel at brush center");
    assert_eq!(center[3], 255, "alpha at brush center");
}
