// =============================================================================
// Visual regression tests — Canvas Transforms
// =============================================================================
//
// Tests for flip, rotate, resize, and flatten operations.
//
// Run with GENERATE_GOLDEN=1 to create/update golden files:
//   GENERATE_GOLDEN=1 cargo test --test visual_transforms

mod common;

use common::*;
use image::{Rgba, RgbaImage};
use paintfe::ops::transform::*;

/// Asymmetric 64×48 test image (non-square so flips/rotates are distinguishable).
fn test_image() -> RgbaImage {
    create_test_gradient(64, 48)
}

// =============================================================================
// Flips (canvas-wide)
// =============================================================================

#[test]
fn flip_canvas_h() {
    let img = test_image();
    let mut state = canvas_from_image(&img);
    flip_canvas_horizontal(&mut state);
    let result = extract_layer(&state, 0);
    assert_golden("transforms", "flip_canvas_h", &result);
}

#[test]
fn flip_canvas_v() {
    let img = test_image();
    let mut state = canvas_from_image(&img);
    flip_canvas_vertical(&mut state);
    let result = extract_layer(&state, 0);
    assert_golden("transforms", "flip_canvas_v", &result);
}

#[test]
fn flip_canvas_h_roundtrip() {
    let img = test_image();
    let mut state = canvas_from_image(&img);
    flip_canvas_horizontal(&mut state);
    flip_canvas_horizontal(&mut state);
    let result = extract_layer(&state, 0);
    assert_eq!(img, result, "flip h × 2 should be identity");
}

#[test]
fn flip_canvas_v_roundtrip() {
    let img = test_image();
    let mut state = canvas_from_image(&img);
    flip_canvas_vertical(&mut state);
    flip_canvas_vertical(&mut state);
    let result = extract_layer(&state, 0);
    assert_eq!(img, result, "flip v × 2 should be identity");
}

// =============================================================================
// Rotations (canvas-wide)
// =============================================================================

#[test]
fn rotate_90cw() {
    let img = test_image();
    let mut state = canvas_from_image(&img);
    rotate_canvas_90cw(&mut state);
    let result = extract_layer(&state, 0);
    assert_eq!(result.width(), 48, "90cw: width should be old height");
    assert_eq!(result.height(), 64, "90cw: height should be old width");
    assert_golden("transforms", "rotate_90cw", &result);
}

#[test]
fn rotate_90ccw() {
    let img = test_image();
    let mut state = canvas_from_image(&img);
    rotate_canvas_90ccw(&mut state);
    let result = extract_layer(&state, 0);
    assert_eq!(result.width(), 48, "90ccw: width should be old height");
    assert_eq!(result.height(), 64, "90ccw: height should be old width");
    assert_golden("transforms", "rotate_90ccw", &result);
}

#[test]
fn rotate_180() {
    let img = test_image();
    let mut state = canvas_from_image(&img);
    rotate_canvas_180(&mut state);
    let result = extract_layer(&state, 0);
    assert_golden("transforms", "rotate_180", &result);
}

#[test]
fn rotate_90cw_x4_identity() {
    let img = test_image();
    let mut state = canvas_from_image(&img);
    for _ in 0..4 {
        rotate_canvas_90cw(&mut state);
    }
    let result = extract_layer(&state, 0);
    assert_eq!(img, result, "4 × 90cw should be identity");
}

#[test]
fn rotate_180_roundtrip() {
    let img = test_image();
    let mut state = canvas_from_image(&img);
    rotate_canvas_180(&mut state);
    rotate_canvas_180(&mut state);
    let result = extract_layer(&state, 0);
    assert_eq!(img, result, "180 × 2 should be identity");
}

#[test]
fn rotate_90cw_then_ccw_identity() {
    let img = test_image();
    let mut state = canvas_from_image(&img);
    rotate_canvas_90cw(&mut state);
    rotate_canvas_90ccw(&mut state);
    let result = extract_layer(&state, 0);
    assert_eq!(img, result, "90cw + 90ccw should be identity");
}

// =============================================================================
// Resize
// =============================================================================

#[test]
fn resize_2x_nearest() {
    let img = test_image();
    let mut state = canvas_from_image(&img);
    resize_image(&mut state, 128, 96, Interpolation::Nearest);
    let result = extract_layer(&state, 0);
    assert_eq!(result.width(), 128);
    assert_eq!(result.height(), 96);
    assert_golden("transforms", "resize_2x_nearest", &result);
}

#[test]
fn resize_half_bilinear() {
    let img = test_image();
    let mut state = canvas_from_image(&img);
    resize_image(&mut state, 32, 24, Interpolation::Bilinear);
    let result = extract_layer(&state, 0);
    assert_eq!(result.width(), 32);
    assert_eq!(result.height(), 24);
    assert_golden("transforms", "resize_half_bilinear", &result);
}

#[test]
fn resize_half_lanczos() {
    let img = test_image();
    let mut state = canvas_from_image(&img);
    resize_image(&mut state, 32, 24, Interpolation::Lanczos3);
    let result = extract_layer(&state, 0);
    assert_eq!(result.width(), 32);
    assert_eq!(result.height(), 24);
    assert_golden("transforms", "resize_half_lanczos", &result);
}

// =============================================================================
// Resize Canvas (anchor‐based)
// =============================================================================

#[test]
fn resize_canvas_center() {
    let img = test_image();
    let mut state = canvas_from_image(&img);
    // Grow by 16px each side, centered, transparent fill
    resize_canvas(&mut state, 96, 80, (1, 1), Rgba([0, 0, 0, 0]));
    let result = extract_layer(&state, 0);
    assert_eq!(result.width(), 96);
    assert_eq!(result.height(), 80);
    assert_golden("transforms", "resize_canvas_center", &result);
}

#[test]
fn resize_canvas_topleft() {
    let img = test_image();
    let mut state = canvas_from_image(&img);
    // Grow canvas anchored to top-left, red fill
    resize_canvas(&mut state, 80, 64, (0, 0), Rgba([255, 0, 0, 255]));
    let result = extract_layer(&state, 0);
    assert_eq!(result.width(), 80);
    assert_eq!(result.height(), 64);
    assert_golden("transforms", "resize_canvas_topleft", &result);
}

// =============================================================================
// Single-layer flips
// =============================================================================

#[test]
fn flip_layer_h() {
    let img = test_image();
    let mut state = canvas_from_image(&img);
    flip_layer_horizontal(&mut state, 0);
    let result = extract_layer(&state, 0);
    assert_golden("transforms", "flip_layer_h", &result);
}

#[test]
fn flip_layer_v() {
    let img = test_image();
    let mut state = canvas_from_image(&img);
    flip_layer_vertical(&mut state, 0);
    let result = extract_layer(&state, 0);
    assert_golden("transforms", "flip_layer_v", &result);
}

// =============================================================================
// Flatten
// =============================================================================

#[test]
fn flatten_single_layer() {
    let img = test_image();
    let mut state = canvas_from_image(&img);
    flatten_image(&mut state);
    assert_eq!(state.layers.len(), 1, "flatten should produce 1 layer");
    let result = extract_layer(&state, 0);
    assert_golden("transforms", "flatten_single", &result);
}

// =============================================================================
// Affine transform
// =============================================================================

#[test]
fn affine_rotate_45() {
    let img = test_image();
    let mut state = canvas_from_image(&img);
    affine_transform_layer(
        &mut state,
        0,
        45.0_f32.to_radians(),
        0.0,
        0.0,
        1.0,
        (0.0, 0.0),
    );
    let result = extract_layer(&state, 0);
    assert_golden("transforms", "affine_rotate_45", &result);
}

#[test]
fn affine_identity() {
    let img = test_image();
    let mut state = canvas_from_image(&img);
    affine_transform_layer(&mut state, 0, 0.0, 0.0, 0.0, 1.0, (0.0, 0.0));
    let result = extract_layer(&state, 0);
    // Identity affine should produce pixel-identical result (bilinear sampling may have rounding)
    let diff = compare_images(&result, &img, 1);
    assert!(
        diff.matches,
        "identity affine should match within tolerance 1: max channel diff = {}",
        diff.max_channel_diff
    );
}
