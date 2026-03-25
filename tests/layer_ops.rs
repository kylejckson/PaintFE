// =============================================================================
// Integration tests — Layer operations
// =============================================================================
//
// Tests layer add/delete/duplicate/flatten/reorder/visibility/opacity on
// CanvasState and canvas_ops functions.

mod common;

#[allow(unused_imports)]
use common::*;
use image::{Rgba, RgbaImage};
use paintfe::canvas::{CanvasState, Layer, TiledImage};
use paintfe::components::history::HistoryManager;
use paintfe::ops::canvas_ops;
use paintfe::ops::transform::flatten_image;

fn history() -> HistoryManager {
    HistoryManager::new(100)
}

// =============================================================================
// Basic layer management
// =============================================================================

#[test]
fn new_canvas_has_one_layer() {
    let state = CanvasState::new(64, 64);
    assert_eq!(state.layers.len(), 1);
    assert_eq!(state.layers[0].name, "Background");
    assert!(state.layers[0].visible);
    assert_eq!(state.layers[0].opacity, 1.0);
}

#[test]
fn add_layer_creates_transparent_layer() {
    let mut state = CanvasState::new(64, 64);
    let mut hist = history();
    canvas_ops::add_layer(&mut state, &mut hist);

    assert_eq!(state.layers.len(), 2);
    // New layer should be transparent (alpha = 0 everywhere)
    let px = state.layers[1].pixels.get_pixel(0, 0);
    assert_eq!(px[3], 0, "new layer should be transparent");
}

#[test]
fn add_text_layer() {
    let mut state = CanvasState::new(64, 64);
    let mut hist = history();
    canvas_ops::add_text_layer(&mut state, &mut hist);

    assert_eq!(state.layers.len(), 2);
    assert!(state.layers[1].is_text_layer());
}

#[test]
fn delete_layer_removes_it() {
    let mut state = CanvasState::new(64, 64);
    let mut hist = history();
    canvas_ops::add_layer(&mut state, &mut hist);
    assert_eq!(state.layers.len(), 2);

    state.active_layer_index = 1;
    canvas_ops::delete_layer(&mut state, &mut hist);
    assert_eq!(state.layers.len(), 1);
}

#[test]
fn delete_last_layer_denied() {
    let mut state = CanvasState::new(64, 64);
    let mut hist = history();
    // Should not delete the only layer
    canvas_ops::delete_layer(&mut state, &mut hist);
    assert_eq!(state.layers.len(), 1, "cannot delete the only layer");
}

#[test]
fn duplicate_layer_copies_pixels() {
    let mut state = CanvasState::new(32, 32);
    // Paint something on layer 0
    state.layers[0]
        .pixels
        .put_pixel(10, 10, Rgba([255, 0, 0, 255]));

    let mut hist = history();
    state.active_layer_index = 0;
    canvas_ops::duplicate_layer(&mut state, &mut hist);

    assert_eq!(state.layers.len(), 2);
    let dup_px = state.layers[1].pixels.get_pixel(10, 10);
    assert_eq!(
        *dup_px,
        Rgba([255, 0, 0, 255]),
        "duplicated layer should have same pixels"
    );
}

// =============================================================================
// Layer visibility and composite
// =============================================================================

#[test]
fn hidden_layer_not_composited() {
    let mut state = CanvasState::new(32, 32);
    // Add a red layer
    let mut layer = Layer::new("Red".into(), 32, 32, Rgba([0, 0, 0, 0]));
    let red_img = RgbaImage::from_pixel(32, 32, Rgba([255, 0, 0, 255]));
    layer.pixels = TiledImage::from_rgba_image(&red_img);
    layer.visible = false;
    state.layers.push(layer);

    let comp = state.composite();
    // Should be white (background only) since red layer is hidden
    assert_eq!(*comp.get_pixel(16, 16), Rgba([255, 255, 255, 255]));
}

#[test]
fn layer_opacity_affects_composite() {
    let mut state = CanvasState::new(32, 32);
    let mut layer = Layer::new("Black50".into(), 32, 32, Rgba([0, 0, 0, 0]));
    let black_img = RgbaImage::from_pixel(32, 32, Rgba([0, 0, 0, 255]));
    layer.pixels = TiledImage::from_rgba_image(&black_img);
    layer.opacity = 0.5;
    state.layers.push(layer);

    let comp = state.composite();
    let px = *comp.get_pixel(16, 16);
    // White bg + 50% opaque black = ~128 gray
    assert!(
        (px[0] as i32 - 128).unsigned_abs() <= 2,
        "expected ~128 gray, got {}",
        px[0]
    );
}

// =============================================================================
// Layer reordering
// =============================================================================

#[test]
fn layer_reorder_changes_composite() {
    let mut state = CanvasState::new(32, 32);

    // Layer 1: full red
    let mut red = Layer::new("Red".into(), 32, 32, Rgba([0, 0, 0, 0]));
    red.pixels =
        TiledImage::from_rgba_image(&RgbaImage::from_pixel(32, 32, Rgba([255, 0, 0, 255])));
    state.layers.push(red);

    // Layer 2: full blue
    let mut blue = Layer::new("Blue".into(), 32, 32, Rgba([0, 0, 0, 0]));
    blue.pixels =
        TiledImage::from_rgba_image(&RgbaImage::from_pixel(32, 32, Rgba([0, 0, 255, 255])));
    state.layers.push(blue);

    // Top layer (blue) should dominate
    let comp1 = state.composite();
    assert_eq!(comp1.get_pixel(16, 16)[2], 255, "blue on top");

    // Swap layers 1 and 2
    state.layers.swap(1, 2);

    // Now red is on top
    let comp2 = state.composite();
    assert_eq!(comp2.get_pixel(16, 16)[0], 255, "red on top after swap");
}

// =============================================================================
// Flatten
// =============================================================================

#[test]
fn flatten_multiple_layers() {
    let mut state = CanvasState::new(32, 32);

    let mut red = Layer::new("Red".into(), 32, 32, Rgba([0, 0, 0, 0]));
    red.pixels =
        TiledImage::from_rgba_image(&RgbaImage::from_pixel(32, 32, Rgba([255, 0, 0, 128])));
    state.layers.push(red);

    let before = state.composite();
    flatten_image(&mut state);

    assert_eq!(state.layers.len(), 1, "flatten should produce one layer");
    let after = state.composite();
    assert_eq!(before, after, "composite should be unchanged after flatten");
}

#[test]
fn flatten_preserves_hidden_layer_exclusion() {
    let mut state = CanvasState::new(32, 32);

    let mut green = Layer::new("Green".into(), 32, 32, Rgba([0, 0, 0, 0]));
    green.pixels =
        TiledImage::from_rgba_image(&RgbaImage::from_pixel(32, 32, Rgba([0, 255, 0, 255])));
    green.visible = false;
    state.layers.push(green);

    let before = state.composite();
    flatten_image(&mut state);

    assert_eq!(state.layers.len(), 1);
    let after = state.composite();
    // Green was hidden, so flatten should not include it
    assert_eq!(*after.get_pixel(16, 16), Rgba([255, 255, 255, 255]));
    assert_eq!(before, after);
}

// =============================================================================
// Active layer index safety
// =============================================================================

#[test]
fn active_layer_index_clamped_after_delete() {
    let mut state = CanvasState::new(32, 32);
    let mut hist = history();
    canvas_ops::add_layer(&mut state, &mut hist);
    canvas_ops::add_layer(&mut state, &mut hist);
    assert_eq!(state.layers.len(), 3);

    state.active_layer_index = 2;
    canvas_ops::delete_layer(&mut state, &mut hist);

    assert!(
        state.active_layer_index < state.layers.len(),
        "active index should be valid after delete"
    );
}
