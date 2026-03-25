// =============================================================================
// Integration tests — Transform / Warp / Displacement pure functions
// =============================================================================
//
// Tests the CPU-side pure functions used by Liquify, Mesh Warp, Perspective Crop
// (Strategy 1 from TEST_PLAN.md §12B).

mod common;

use common::*;
use image::{Rgba, RgbaImage};
use paintfe::canvas::CanvasState;
use paintfe::ops::transform::{
    self, DisplacementField, catmull_rom_curve_point, catmull_rom_surface, catmull_rom_weights,
    generate_displacement_from_mesh, generate_displacement_from_mesh_fast, warp_displacement_full,
    warp_mesh_catmull_rom,
};

// =============================================================================
// Helpers
// =============================================================================

/// Create a 32×32 gradient image (red channel increases left-to-right,
/// green increases top-to-bottom) for warp visibility.
fn gradient_32() -> RgbaImage {
    let mut img = RgbaImage::new(32, 32);
    for y in 0..32 {
        for x in 0..32 {
            img.put_pixel(x, y, Rgba([x as u8 * 8, y as u8 * 8, 128, 255]));
        }
    }
    img
}

/// Build a uniform NxN mesh grid covering a WxH image.
fn uniform_grid(cols: usize, rows: usize, w: f32, h: f32) -> Vec<[f32; 2]> {
    let mut pts = Vec::with_capacity((rows + 1) * (cols + 1));
    for r in 0..=rows {
        for c in 0..=cols {
            pts.push([c as f32 / cols as f32 * w, r as f32 / rows as f32 * h]);
        }
    }
    pts
}

// =============================================================================
// Catmull-Rom math
// =============================================================================

#[test]
fn catmull_rom_weights_at_zero() {
    let w = catmull_rom_weights(0.0);
    // At t=0: w = [0, 1, 0, 0] (passes through P_i)
    assert!((w[0]).abs() < 1e-6, "w0 = {}", w[0]);
    assert!((w[1] - 1.0).abs() < 1e-6, "w1 = {}", w[1]);
    assert!((w[2]).abs() < 1e-6, "w2 = {}", w[2]);
    assert!((w[3]).abs() < 1e-6, "w3 = {}", w[3]);
}

#[test]
fn catmull_rom_weights_at_one() {
    let w = catmull_rom_weights(1.0);
    // At t=1: w = [0, 0, 1, 0] (passes through P_{i+1})
    assert!((w[0]).abs() < 1e-6, "w0 = {}", w[0]);
    assert!((w[1]).abs() < 1e-6, "w1 = {}", w[1]);
    assert!((w[2] - 1.0).abs() < 1e-6, "w2 = {}", w[2]);
    assert!((w[3]).abs() < 1e-6, "w3 = {}", w[3]);
}

#[test]
fn catmull_rom_weights_partition_of_unity() {
    // For any t, weights must sum to 1.0
    for i in 0..=10 {
        let t = i as f32 / 10.0;
        let w = catmull_rom_weights(t);
        let sum: f32 = w.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "weights at t={} sum to {} (not 1.0)",
            t,
            sum
        );
    }
}

#[test]
fn catmull_rom_surface_identity_grid() {
    // A uniform 2×2 grid covering 32×32 should map each (u,v) ≈ to itself.
    let grid = uniform_grid(2, 2, 32.0, 32.0);
    // Sample center
    let p = catmull_rom_surface(&grid, 2, 2, 1.0, 1.0);
    assert!((p[0] - 16.0).abs() < 0.5, "x={} expected 16", p[0]);
    assert!((p[1] - 16.0).abs() < 0.5, "y={} expected 16", p[1]);
    // Sample origin
    let p0 = catmull_rom_surface(&grid, 2, 2, 0.0, 0.0);
    assert!((p0[0]).abs() < 0.5, "x={} expected 0", p0[0]);
    assert!((p0[1]).abs() < 0.5, "y={} expected 0", p0[1]);
}

#[test]
fn catmull_rom_curve_point_endpoints() {
    let pts = vec![[0.0, 0.0], [10.0, 5.0], [20.0, 0.0], [30.0, 5.0]];
    let p0 = catmull_rom_curve_point(&pts, 0.0);
    assert!((p0[0] - 0.0).abs() < 0.01 && (p0[1] - 0.0).abs() < 0.01);
    // At t = n-1 = 3.0, should be at last point
    let pn = catmull_rom_curve_point(&pts, 3.0);
    assert!((pn[0] - 30.0).abs() < 0.01 && (pn[1] - 5.0).abs() < 0.01);
}

#[test]
fn catmull_rom_curve_point_midpoint() {
    // Straight line: points along y=0 at x = 0, 10, 20, 30
    let pts = vec![[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [30.0, 0.0]];
    // At t=1.5 (midpoint of segment 1..2), should be at x=15
    let p = catmull_rom_curve_point(&pts, 1.5);
    assert!((p[0] - 15.0).abs() < 0.01, "x={} expected 15", p[0]);
    assert!((p[1]).abs() < 0.01, "y={} expected 0", p[1]);
}

// =============================================================================
// Displacement field
// =============================================================================

#[test]
fn displacement_identity_preserves_image() {
    let src = gradient_32();
    let field = DisplacementField::new(32, 32); // zero displacement
    let result = warp_displacement_full(&src, &field);
    // Identity displacement should produce identical image
    for y in 0..32 {
        for x in 0..32 {
            assert_eq!(
                result.get_pixel(x, y),
                src.get_pixel(x, y),
                "pixel ({},{}) changed under identity displacement",
                x,
                y
            );
        }
    }
}

#[test]
fn displacement_translate_shifts_pixels() {
    let src = gradient_32();
    let mut field = DisplacementField::new(32, 32);
    // Displacement convention: output(x,y) = src(x - dx, y - dy)
    // To have output(10,16) come from src(5,16), set dx = 5 for all pixels
    for y in 0..32 {
        for x in 0..32 {
            field.add(x, y, 5.0, 0.0);
        }
    }
    let result = warp_displacement_full(&src, &field);
    // Pixel at (10, 16) in result should come from (5, 16) in source
    let rp = result.get_pixel(10, 16);
    let sp = src.get_pixel(5, 16);
    assert_eq!(rp, sp, "shifted pixel mismatch");
}

#[test]
fn displacement_field_golden() {
    let src = gradient_32();
    // Create a radial push from center
    let mut field = DisplacementField::new(32, 32);
    field.apply_push(16.0, 16.0, 3.0, 0.0, 10.0, 0.8);
    let result = warp_displacement_full(&src, &field);
    assert_golden("transform", "displacement_radial_push", &result);
}

// =============================================================================
// Mesh warp
// =============================================================================

#[test]
fn mesh_warp_identity() {
    let src = gradient_32();
    let grid = uniform_grid(2, 2, 32.0, 32.0);
    let result = warp_mesh_catmull_rom(&src, &grid, &grid, 2, 2, 32, 32);
    // Identity mesh: deformed == original — should produce identical output.
    // Allow small interpolation tolerance.
    let mut max_diff: u8 = 0;
    for y in 0..32 {
        for x in 0..32 {
            let rp = result.get_pixel(x, y);
            let sp = src.get_pixel(x, y);
            for c in 0..4 {
                let d = (rp[c] as i16 - sp[c] as i16).unsigned_abs() as u8;
                max_diff = max_diff.max(d);
            }
        }
    }
    assert!(
        max_diff <= 2,
        "identity mesh warp has max diff {} (expected ≤ 2)",
        max_diff
    );
}

#[test]
fn mesh_warp_deformed_golden() {
    let src = gradient_32();
    let original = uniform_grid(2, 2, 32.0, 32.0);
    let mut deformed = original.clone();
    // Move center point (1,1) to (20, 20) — distorts the image
    deformed[4] = [20.0, 20.0]; // row 1, col 1 in 3×3 = index 4
    let result = warp_mesh_catmull_rom(&src, &original, &deformed, 2, 2, 32, 32);
    assert_golden("transform", "mesh_warp_deformed", &result);
}

#[test]
fn generate_displacement_from_mesh_identity_is_zero() {
    let grid = uniform_grid(2, 2, 32.0, 32.0);
    let field = generate_displacement_from_mesh(&grid, &grid, 2, 2, 32, 32);
    // Identity mesh should produce near-zero displacement
    let max_disp = field.data.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    assert!(
        max_disp < 1.0,
        "identity mesh displacement max = {} (expected < 1.0)",
        max_disp
    );
}

#[test]
fn generate_displacement_from_mesh_fast_matches_full() {
    let original = uniform_grid(2, 2, 32.0, 32.0);
    let mut deformed = original.clone();
    deformed[4] = [20.0, 20.0]; // center point shifted

    let field_full = generate_displacement_from_mesh(&original, &deformed, 2, 2, 32, 32);
    let mut fast_data = vec![0.0f32; 32 * 32 * 2];
    generate_displacement_from_mesh_fast(&deformed, 2, 2, 32, 32, &mut fast_data);

    // The "fast" version only uses deformed points and computes displacement as
    // (deformed_surface - identity_surface), while the "full" version computes
    // (deformed_surface - original_surface). With identical original == uniform,
    // both should match closely.
    let mut max_diff = 0.0f32;
    for (a, b) in field_full.data.iter().zip(fast_data.iter()) {
        max_diff = max_diff.max((a - b).abs());
    }
    assert!(
        max_diff < 2.0,
        "fast vs full displacement max diff = {} (expected < 2.0)",
        max_diff
    );
}

// =============================================================================
// Affine transform
// =============================================================================

#[test]
fn affine_identity_preserves_pixels() {
    let mut state = CanvasState::new(32, 32);
    // Paint a red dot
    state.layers[0]
        .pixels
        .put_pixel(16, 16, Rgba([255, 0, 0, 255]));
    let before = state.composite();

    // Identity affine: 0 rotation, scale 1, no offset
    transform::affine_transform_layer(&mut state, 0, 0.0, 0.0, 0.0, 1.0, (0.0, 0.0));
    let after = state.composite();

    for y in 0..32 {
        for x in 0..32 {
            assert_eq!(
                before.get_pixel(x, y),
                after.get_pixel(x, y),
                "pixel ({},{}) changed under identity affine",
                x,
                y
            );
        }
    }
}

#[test]
fn affine_rotate_90_golden() {
    let src = create_test_gradient(32, 32);
    let mut state = canvas_from_image(&src);
    transform::affine_transform_layer(
        &mut state,
        0,
        std::f32::consts::FRAC_PI_2, // 90° Z rotation
        0.0,
        0.0,
        1.0,
        (0.0, 0.0),
    );
    let result = state.composite();
    assert_golden("transform", "affine_rotate_90", &result);
}

#[test]
fn affine_scale_half_golden() {
    let src = create_test_gradient(32, 32);
    let mut state = canvas_from_image(&src);
    transform::affine_transform_layer(&mut state, 0, 0.0, 0.0, 0.0, 0.5, (0.0, 0.0));
    let result = state.composite();
    assert_golden("transform", "affine_scale_half", &result);
}

// =============================================================================
// Golden tests for displacement warp
// =============================================================================

#[test]
fn warp_displacement_full_golden() {
    let src = gradient_32();
    let mut field = DisplacementField::new(32, 32);
    // Create a swirl-like displacement
    for y in 0..32 {
        for x in 0..32 {
            let dx = x as f32 - 16.0;
            let dy = y as f32 - 16.0;
            let r = (dx * dx + dy * dy).sqrt().max(0.001);
            let strength = (1.0 - r / 16.0).max(0.0);
            field.add(x, y, -dy * strength * 0.5, dx * strength * 0.5);
        }
    }
    let result = warp_displacement_full(&src, &field);
    assert_golden("transform", "displacement_swirl", &result);
}
