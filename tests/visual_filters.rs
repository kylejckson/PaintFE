// =============================================================================
// Visual regression tests — Filters & Effects
// =============================================================================
//
// Each test applies a _core function to a standardised 64×64 gradient image
// and compares the result against a committed golden PNG.
//
// Run with GENERATE_GOLDEN=1 to create/update golden files:
//   GENERATE_GOLDEN=1 cargo test --test visual_filters
//
// On mismatch, actual + diff images are saved to tests/output/.

mod common;

use common::*;
use image::RgbaImage;
use paintfe::ops::effects::*;
use paintfe::ops::filters::parallel_gaussian_blur_pub;

/// Standard 64×64 test image used by all filter/effect visual tests.
fn test_image() -> RgbaImage {
    create_test_gradient(64, 64)
}

// =============================================================================
// Blur filters
// =============================================================================

#[test]
fn gaussian_blur_s2() {
    let img = test_image();
    let result = parallel_gaussian_blur_pub(&img, 2.0);
    assert_golden("filters", "gaussian_blur_s2", &result);
}

#[test]
fn gaussian_blur_s5() {
    let img = test_image();
    let result = parallel_gaussian_blur_pub(&img, 5.0);
    assert_golden("filters", "gaussian_blur_s5", &result);
}

#[test]
fn bokeh_blur_r5() {
    let img = test_image();
    let result = bokeh_blur_core(&img, 5.0, None);
    assert_golden("filters", "bokeh_blur_r5", &result);
}

#[test]
fn motion_blur_45_10() {
    let img = test_image();
    let result = motion_blur_core(&img, 45.0, 10.0, None);
    assert_golden("filters", "motion_blur_45_10", &result);
}

#[test]
fn box_blur_r3() {
    let img = test_image();
    let result = box_blur_core(&img, 3.0, None);
    assert_golden("filters", "box_blur_r3", &result);
}

#[test]
fn zoom_blur() {
    let img = test_image();
    let result = zoom_blur_core(&img, 0.5, 0.5, 0.3, 8, [0.0, 0.0, 0.0, 0.0], 0.0, None);
    assert_golden("filters", "zoom_blur", &result);
}

// =============================================================================
// Distortion effects
// =============================================================================

#[test]
fn crystallize_s16() {
    let img = test_image();
    let result = crystallize_core(&img, 16.0, 42, None);
    assert_golden("filters", "crystallize_s16", &result);
}

#[test]
fn dents() {
    let img = test_image();
    let result = dents_core(&img, 20.0, 10.0, 42, 2, 0.5, false, false, None);
    assert_golden("filters", "dents", &result);
}

#[test]
fn pixelate_8() {
    let img = test_image();
    let result = pixelate_core(&img, 8, None);
    assert_golden("filters", "pixelate_8", &result);
}

#[test]
fn bulge_05() {
    let img = test_image();
    let result = bulge_core(&img, 0.5, None);
    assert_golden("filters", "bulge_05", &result);
}

#[test]
fn twist_45() {
    let img = test_image();
    let result = twist_core(&img, 45.0, None);
    assert_golden("filters", "twist_45", &result);
}

// =============================================================================
// Noise effects
// =============================================================================

#[test]
fn add_noise_uniform() {
    let img = test_image();
    let result = add_noise_core(&img, 30.0, NoiseType::Uniform, false, 42, 1.0, 1, None);
    assert_golden("filters", "add_noise_uniform", &result);
}

#[test]
fn add_noise_gaussian() {
    let img = test_image();
    let result = add_noise_core(&img, 30.0, NoiseType::Gaussian, true, 42, 1.0, 1, None);
    assert_golden("filters", "add_noise_gaussian_mono", &result);
}

#[test]
fn add_noise_perlin() {
    let img = test_image();
    let result = add_noise_core(&img, 50.0, NoiseType::Perlin, false, 42, 5.0, 3, None);
    assert_golden("filters", "add_noise_perlin", &result);
}

#[test]
fn reduce_noise() {
    let img = test_image();
    let result = reduce_noise_core(&img, 0.5, 2, None);
    assert_golden("filters", "reduce_noise", &result);
}

#[test]
fn median_r2() {
    let img = test_image();
    let result = median_core(&img, 2, None);
    assert_golden("filters", "median_r2", &result);
}

// =============================================================================
// Stylize effects
// =============================================================================

#[test]
fn glow_r3_i05() {
    let img = test_image();
    let result = glow_core(&img, 3.0, 0.5, None);
    assert_golden("filters", "glow_r3_i05", &result);
}

#[test]
fn sharpen_a1_r1() {
    let img = test_image();
    let result = sharpen_core(&img, 1.0, 1.0, None);
    assert_golden("filters", "sharpen_a1_r1", &result);
}

#[test]
fn vignette_08_05() {
    let img = test_image();
    let result = vignette_core(&img, 0.8, 0.5, None);
    assert_golden("filters", "vignette_08_05", &result);
}

#[test]
fn halftone_circle() {
    let img = test_image();
    let result = halftone_core(&img, 4.0, 45.0, HalftoneShape::Circle, None);
    assert_golden("filters", "halftone_circle", &result);
}

// =============================================================================
// Render / Generate effects
// =============================================================================

#[test]
fn grid_lines_16() {
    let img = test_image();
    let result = grid_core(&img, 16, 16, 1, [0, 0, 0, 255], GridStyle::Lines, 1.0, None);
    assert_golden("filters", "grid_lines_16", &result);
}

#[test]
fn drop_shadow() {
    let img = create_solid(64, 64, [0, 0, 0, 0]); // transparent background
    // Put a white square in the center
    let mut img = img;
    for y in 16..48 {
        for x in 16..48 {
            img.put_pixel(x, y, image::Rgba([255, 255, 255, 255]));
        }
    }
    let result = shadow_core(&img, 5, 5, 3.0, [0, 0, 0, 255], 0.8, None);
    assert_golden("filters", "drop_shadow", &result);
}

#[test]
fn outline_outside() {
    let mut img = create_solid(64, 64, [0, 0, 0, 0]);
    for y in 16..48 {
        for x in 16..48 {
            img.put_pixel(x, y, image::Rgba([255, 0, 0, 255]));
        }
    }
    let result = outline_core(&img, 2, [0, 0, 255, 255], OutlineMode::Outside, None);
    assert_golden("filters", "outline_outside", &result);
}

#[test]
fn contours() {
    let img = test_image();
    let result = contours_core(&img, 10.0, 5.0, 1.0, [0, 0, 0, 255], 42, 2, 0.5, None);
    assert_golden("filters", "contours", &result);
}

// =============================================================================
// Glitch effects
// =============================================================================

#[test]
fn pixel_drag() {
    let img = test_image();
    let result = pixel_drag_core(&img, 42, 50.0, 20, 0.0, None);
    assert_golden("filters", "pixel_drag", &result);
}

#[test]
fn rgb_displace() {
    let img = test_image();
    let result = rgb_displace_core(&img, (5, 0), (0, 0), (-5, 0), None);
    assert_golden("filters", "rgb_displace", &result);
}

// =============================================================================
// Artistic effects
// =============================================================================

#[test]
fn ink() {
    let img = test_image();
    let result = ink_core(&img, 1.0, 0.5, None);
    assert_golden("filters", "ink", &result);
}

#[test]
fn oil_painting() {
    let img = test_image();
    let result = oil_painting_core(&img, 3, 20, None);
    assert_golden("filters", "oil_painting", &result);
}

#[test]
fn color_filter_multiply() {
    let img = test_image();
    let result = color_filter_core(
        &img,
        [255, 128, 0, 255],
        0.5,
        ColorFilterMode::Multiply,
        None,
    );
    assert_golden("filters", "color_filter_multiply", &result);
}

// =============================================================================
// Identity / no-op tests (Tier 1 — should be pixel-exact with input)
// =============================================================================

#[test]
fn gaussian_blur_identity() {
    let img = test_image();
    let result = parallel_gaussian_blur_pub(&img, 0.0);
    assert_eq!(img, result, "blur with sigma=0 should be identity");
}

#[test]
fn pixelate_identity() {
    let img = test_image();
    let result = pixelate_core(&img, 1, None);
    let diff = compare_images(&result, &img, 5);
    assert!(
        diff.matches,
        "pixelate(1) should match within tolerance 5: max diff = {}",
        diff.max_channel_diff
    );
}

#[test]
fn sharpen_identity() {
    let img = test_image();
    let result = sharpen_core(&img, 0.0, 1.0, None);
    assert_eq!(img, result, "sharpen with amount=0 should be identity");
}

#[test]
fn bulge_identity() {
    let img = test_image();
    let result = bulge_core(&img, 0.0, None);
    assert_eq!(img, result, "bulge with amount=0 should be identity");
}

#[test]
fn twist_identity() {
    let img = test_image();
    let result = twist_core(&img, 0.0, None);
    assert_eq!(img, result, "twist with angle=0 should be identity");
}

#[test]
fn vignette_identity() {
    let img = test_image();
    let result = vignette_core(&img, 0.0, 0.5, None);
    assert_eq!(img, result, "vignette with amount=0 should be identity");
}

#[test]
fn color_filter_identity() {
    let img = test_image();
    let result = color_filter_core(
        &img,
        [255, 255, 255, 255],
        0.0,
        ColorFilterMode::Multiply,
        None,
    );
    assert_eq!(
        img, result,
        "color filter with intensity=0 should be identity"
    );
}
