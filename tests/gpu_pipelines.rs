// =============================================================================
// Integration tests — GPU pipeline (headless software fallback)
// =============================================================================
//
// Strategy 3C from TEST_PLAN.md §12D: Tests GPU compute pipelines using wgpu's
// software fallback adapter (no real GPU needed). Each test tries to create a
// GpuRenderer; if the software adapter is unavailable (rare), the test is skipped.
//
// Cover: Gaussian blur, gradient, displacement warp (liquify), mesh warp,
//        flood fill distance map, compositor.

mod common;

use image::{Rgba, RgbaImage};
use paintfe::canvas::{CanvasState, Layer};
use paintfe::gpu::GradientGpuParams;
use paintfe::gpu::renderer::GpuRenderer;
use paintfe::ops::transform::DisplacementField;

// =============================================================================
// Helper: try to create a headless GPU renderer (software fallback)
// =============================================================================

/// Attempt to create a GpuRenderer with the software fallback adapter.
/// Returns None if no adapter is available (test will be skipped).
fn try_gpu() -> Option<GpuRenderer> {
    GpuRenderer::try_new("")
}

macro_rules! require_gpu {
    ($gpu:ident) => {
        #[allow(unused_mut)]
        let Some(mut $gpu) = try_gpu() else {
            eprintln!(
                "SKIP: no GPU adapter available (software fallback not supported on this OS)"
            );
            return;
        };
    };
}

/// Create a 64×64 gradient image for GPU pipeline tests.
fn gpu_test_image() -> RgbaImage {
    let mut img = RgbaImage::new(64, 64);
    for y in 0..64 {
        for x in 0..64 {
            img.put_pixel(x, y, Rgba([x as u8 * 4, y as u8 * 4, 128, 255]));
        }
    }
    img
}

// =============================================================================
// GPU Blur pipeline
// =============================================================================

#[test]
fn gpu_blur_produces_output() {
    require_gpu!(gpu);
    let img = gpu_test_image();
    let (w, h) = (img.width(), img.height());
    let flat: Vec<u8> = img.into_raw();

    // blur_image: upload → blur → readback (convenience method)
    let result = gpu.blur_pipeline.blur_image(&gpu.ctx, &flat, w, h, 3.0);

    // Result should differ from input (blur changes pixels)
    assert_ne!(result, flat, "blurred output should differ from input");

    // Center pixel should still be somewhat close to original (not wildly different)
    let ci = ((32 * w + 32) * 4) as usize;
    let diff = (result[ci] as i16 - flat[ci] as i16).unsigned_abs();
    assert!(diff < 30, "blur diff at center = {} (expected < 30)", diff);
}

#[test]
fn gpu_blur_identity_sigma_zero() {
    require_gpu!(gpu);
    let img = gpu_test_image();
    let (w, h) = (img.width(), img.height());
    let flat: Vec<u8> = img.into_raw();

    // Sigma 0 should produce identity (or near-identity)
    let result = gpu.blur_pipeline.blur_image(&gpu.ctx, &flat, w, h, 0.0);

    // Should be very close to original
    let max_diff: u8 = flat
        .iter()
        .zip(result.iter())
        .map(|(a, b)| (*a as i16 - *b as i16).unsigned_abs() as u8)
        .max()
        .unwrap_or(0);
    assert!(
        max_diff <= 1,
        "sigma=0 blur max diff = {} (expected ≤ 1)",
        max_diff
    );
}

// =============================================================================
// GPU Gradient pipeline
// =============================================================================

#[test]
fn gpu_gradient_linear() {
    require_gpu!(gpu);
    let w = 64u32;
    let h = 64u32;

    // Build a simple 2-stop LUT: black → white
    let mut lut = [0u8; 256 * 4];
    for i in 0..256 {
        let v = i as u8;
        lut[i * 4] = v;
        lut[i * 4 + 1] = v;
        lut[i * 4 + 2] = v;
        lut[i * 4 + 3] = 255;
    }

    let mut buf = vec![0u8; (w * h * 4) as usize];

    let params = GradientGpuParams {
        start_x: 0.0,
        start_y: 0.5,
        end_x: 1.0,
        end_y: 0.5,
        width: w,
        height: h,
        shape: 0,  // linear
        repeat: 0, // clamp
        is_eraser: 0,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };

    gpu.gradient_pipeline
        .generate_into(&gpu.ctx, &params, &lut, &mut buf);

    // The gradient should vary across the width. Check that rightmost pixel
    // is significantly brighter than leftmost pixel.
    let left_r = buf[0]; // pixel (0,0) R
    let right_r = buf[(63 * 4) as usize]; // pixel (63,0) R
    assert!(
        right_r > left_r.saturating_add(100),
        "gradient should increase left→right: left_r={} right_r={}",
        left_r,
        right_r
    );
}

// =============================================================================
// GPU Liquify (displacement warp) pipeline
// =============================================================================

#[test]
fn gpu_liquify_identity_displacement() {
    require_gpu!(gpu);
    let img = gpu_test_image();
    let (w, h) = (img.width(), img.height());
    let flat_rgba: Vec<u8> = img.into_raw();

    // Zero displacement = identity
    let displacement = vec![0.0f32; (w * h * 2) as usize];

    gpu.liquify_pipeline.invalidate_source();
    let mut result = vec![0u8; (w * h * 4) as usize];
    gpu.liquify_pipeline
        .warp_into(&gpu.ctx, &flat_rgba, &displacement, w, h, &mut result);

    // Should be very close to original
    let max_diff: u8 = flat_rgba
        .iter()
        .zip(result.iter())
        .map(|(a, b)| (*a as i16 - *b as i16).unsigned_abs() as u8)
        .max()
        .unwrap_or(0);
    assert!(
        max_diff <= 2,
        "identity liquify max diff = {} (expected ≤ 2)",
        max_diff
    );
}

#[test]
fn gpu_liquify_push_changes_pixels() {
    require_gpu!(gpu);
    let img = gpu_test_image();
    let (w, h) = (img.width(), img.height());
    let flat_rgba: Vec<u8> = img.into_raw();

    // Create a displacement that pushes pixels rightward from center
    let mut field = DisplacementField::new(w, h);
    field.apply_push(32.0, 32.0, 5.0, 0.0, 15.0, 0.8);

    gpu.liquify_pipeline.invalidate_source();
    let mut result = vec![0u8; (w * h * 4) as usize];
    gpu.liquify_pipeline
        .warp_into(&gpu.ctx, &flat_rgba, &field.data, w, h, &mut result);

    // Center area should be different from the original
    let ci = ((32 * w + 32) * 4) as usize;
    let diff = (result[ci] as i16 - flat_rgba[ci] as i16).unsigned_abs();
    assert!(diff > 0, "liquify push should change center pixels");
}

// =============================================================================
// GPU Mesh Warp Displacement pipeline
// =============================================================================

#[test]
fn gpu_mesh_warp_identity() {
    require_gpu!(gpu);
    let w = 64u32;
    let h = 64u32;
    let cols = 2u32;
    let rows = 2u32;

    // Uniform grid as &[[f32; 2]]
    let mut grid_pts: Vec<[f32; 2]> = Vec::new();
    for r in 0..=(rows) {
        for c in 0..=(cols) {
            grid_pts.push([
                c as f32 / cols as f32 * w as f32,
                r as f32 / rows as f32 * h as f32,
            ]);
        }
    }

    let mut displacement = vec![0.0f32; (w * h * 2) as usize];
    gpu.mesh_warp_disp_pipeline.generate_displacement(
        &gpu.ctx,
        &grid_pts,
        cols,
        rows,
        w,
        h,
        &mut displacement,
    );

    // Identity grid → near-zero displacement (GPU compute may have small numerical error)
    let max_disp = displacement.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    assert!(
        max_disp < 4.0,
        "identity mesh warp GPU displacement max = {} (expected < 4.0)",
        max_disp
    );
}

// =============================================================================
// GPU Flood Fill pipeline
// =============================================================================

#[test]
fn gpu_flood_fill_uniform_image() {
    require_gpu!(gpu);

    // Create a uniform white 32×32 image
    let w = 32u32;
    let h = 32u32;
    let flat = vec![255u8; (w * h * 4) as usize];
    let input_key = flat.as_ptr() as usize;
    let target_color = [255u8, 255, 255, 255]; // seed color = white

    let mut distances = vec![255u8; (w * h) as usize];
    let success = gpu.flood_fill_pipeline.compute_flood_distances(
        &gpu.ctx,
        &flat,
        input_key,
        target_color,
        16, // seed x
        16, // seed y
        w,
        h,
        &mut distances,
    );
    assert!(success, "GPU flood fill should succeed");

    // On a uniform image, all distances from seed should be 0 (same color)
    let max_dist = *distances.iter().max().unwrap_or(&0);
    assert_eq!(max_dist, 0, "uniform image flood fill max dist should be 0");
}

#[test]
fn gpu_flood_fill_two_colors() {
    require_gpu!(gpu);

    // Left half white, right half black — seed in white region
    let w = 32u32;
    let h = 32u32;
    let mut flat = vec![0u8; (w * h * 4) as usize];
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 4) as usize;
            let v = if x < 16 { 255 } else { 0 };
            flat[i] = v;
            flat[i + 1] = v;
            flat[i + 2] = v;
            flat[i + 3] = 255;
        }
    }

    let input_key = flat.as_ptr() as usize;
    let target_color = [255u8, 255, 255, 255]; // seed pixel is white

    let mut distances = vec![255u8; (w * h) as usize];
    let success = gpu.flood_fill_pipeline.compute_flood_distances(
        &gpu.ctx,
        &flat,
        input_key,
        target_color,
        8, // seed in white region
        16,
        w,
        h,
        &mut distances,
    );
    assert!(success);

    // White region pixels should have distance 0
    let d_white = distances[(16 * w + 8) as usize];
    assert_eq!(d_white, 0, "white region distance should be 0");

    // Black region should have high distance from white seed
    let d_black = distances[(16 * w + 24) as usize];
    assert!(
        d_black > 200,
        "black region distance should be >200, got {}",
        d_black
    );
}

// =============================================================================
// GPU Compositor: composite layers
// =============================================================================

#[test]
fn gpu_composite_single_layer() {
    require_gpu!(gpu);

    let w = 32u32;
    let h = 32u32;
    let mut state = CanvasState::new(w, h);
    // Paint a red pixel on layer 0
    state.layers[0]
        .pixels
        .put_pixel(10, 10, Rgba([255, 0, 0, 255]));

    // Extract layer data, upload to GPU
    let layer_data = state.layers[0].pixels.extract_region_rgba(0, 0, w, h);
    gpu.ensure_layer_texture(0, w, h, &layer_data, 1);

    // Composite with blend mode Normal (0)
    let layer_info = vec![(0usize, 1.0f32, true, 0u8)];
    let result = gpu.composite(w, h, &layer_info);
    assert!(result.is_some(), "GPU composite should succeed");

    let pixels = result.unwrap();
    // Check the red pixel at (10, 10)
    let idx = ((10 * w + 10) * 4) as usize;
    assert!(
        pixels[idx] > 240,
        "red channel at (10,10) = {} (expected > 240)",
        pixels[idx]
    );
    assert!(
        pixels[idx + 1] < 15,
        "green channel at (10,10) = {} (expected < 15)",
        pixels[idx + 1]
    );
    assert!(
        pixels[idx + 2] < 15,
        "blue channel at (10,10) = {} (expected < 15)",
        pixels[idx + 2]
    );
    assert!(
        pixels[idx + 3] > 240,
        "alpha channel at (10,10) = {} (expected > 240)",
        pixels[idx + 3]
    );
}

#[test]
fn gpu_composite_two_layers() {
    require_gpu!(gpu);

    let w = 32u32;
    let h = 32u32;
    let mut state = CanvasState::new(w, h);
    state
        .layers
        .push(Layer::new("Layer 1".into(), w, h, Rgba([0, 0, 0, 0])));

    // Layer 0: solid red at (10, 10)
    state.layers[0]
        .pixels
        .put_pixel(10, 10, Rgba([255, 0, 0, 255]));
    // Layer 1: solid blue at (10, 10) (on top)
    state.layers[1]
        .pixels
        .put_pixel(10, 10, Rgba([0, 0, 255, 255]));

    // Upload both layers
    let data0 = state.layers[0].pixels.extract_region_rgba(0, 0, w, h);
    gpu.ensure_layer_texture(0, w, h, &data0, 1);
    let data1 = state.layers[1].pixels.extract_region_rgba(0, 0, w, h);
    gpu.ensure_layer_texture(1, w, h, &data1, 1);

    // Composite: layer 0 bottom, layer 1 top (Normal blend)
    let layer_info = vec![(0usize, 1.0f32, true, 0u8), (1usize, 1.0f32, true, 0u8)];
    let result = gpu.composite(w, h, &layer_info);
    assert!(result.is_some());

    let pixels = result.unwrap();
    let idx = ((10 * w + 10) * 4) as usize;
    // Blue layer on top should dominate
    assert!(
        pixels[idx] < 15,
        "red = {} (expected < 15, blue on top)",
        pixels[idx]
    );
    assert!(
        pixels[idx + 2] > 240,
        "blue = {} (expected > 240)",
        pixels[idx + 2]
    );
}
