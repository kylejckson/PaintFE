// =============================================================================
// Visual regression tests — Color Adjustments
// =============================================================================
//
// Adjustment functions operate on CanvasState. We create a minimal CanvasState
// from a standard 64×64 test image, apply the adjustment, then compare the
// resulting layer against a golden PNG.
//
// Run with GENERATE_GOLDEN=1 to create/update golden files:
//   GENERATE_GOLDEN=1 cargo test --test visual_adjustments

mod common;

use common::*;
use image::RgbaImage;
use paintfe::canvas::CanvasState;
use paintfe::ops::adjustments::*;
use paintfe::ops::filters::desaturate_layer;

/// Standard 64×64 test image.
fn test_image() -> RgbaImage {
    create_test_gradient(64, 64)
}

/// Helper: apply an adjustment via `_from_flat` pattern and return the result.
fn apply_and_extract<F>(img: &RgbaImage, apply_fn: F) -> RgbaImage
where
    F: FnOnce(&mut CanvasState, &RgbaImage),
{
    let mut state = canvas_from_image(img);
    apply_fn(&mut state, img);
    extract_layer(&state, 0)
}

/// Helper for adjustments that don't use _from_flat (operate directly on state).
fn apply_direct_and_extract<F>(img: &RgbaImage, apply_fn: F) -> RgbaImage
where
    F: FnOnce(&mut CanvasState),
{
    let mut state = canvas_from_image(img);
    apply_fn(&mut state);
    extract_layer(&state, 0)
}

// =============================================================================
// Instant adjustments (no parameters)
// =============================================================================

#[test]
fn invert_colors() {
    let img = test_image();
    let result = apply_direct_and_extract(&img, |s| {
        paintfe::ops::adjustments::invert_colors(s, 0);
    });
    assert_golden("adjustments", "invert_colors", &result);
}

#[test]
fn invert_colors_roundtrip() {
    let img = test_image();
    let mut state = canvas_from_image(&img);
    paintfe::ops::adjustments::invert_colors(&mut state, 0);
    paintfe::ops::adjustments::invert_colors(&mut state, 0);
    let result = extract_layer(&state, 0);
    assert_eq!(img, result, "invert × 2 should be identity");
}

#[test]
fn invert_alpha() {
    let img = test_image();
    let result = apply_direct_and_extract(&img, |s| {
        paintfe::ops::adjustments::invert_alpha(s, 0);
    });
    assert_golden("adjustments", "invert_alpha", &result);
}

#[test]
fn invert_alpha_visual() {
    // Separate visual test — roundtrip is not guaranteed identity since
    // alpha=0 pixels may lose color information in premultiplied storage
    let img = test_image();
    let result = apply_direct_and_extract(&img, |s| {
        paintfe::ops::adjustments::invert_alpha(s, 0);
    });
    assert_golden("adjustments", "invert_alpha_double", &result);
}

#[test]
fn sepia() {
    let img = test_image();
    let result = apply_direct_and_extract(&img, |s| {
        paintfe::ops::adjustments::sepia(s, 0);
    });
    assert_golden("adjustments", "sepia", &result);
}

#[test]
fn auto_levels() {
    let img = test_image();
    let result = apply_direct_and_extract(&img, |s| {
        paintfe::ops::adjustments::auto_levels(s, 0);
    });
    assert_golden("adjustments", "auto_levels", &result);
}

#[test]
fn desaturate() {
    let img = test_image();
    let result = apply_direct_and_extract(&img, |s| {
        desaturate_layer(s, 0);
    });
    assert_golden("adjustments", "desaturate", &result);
}

// =============================================================================
// Parameterized adjustments (via _from_flat)
// =============================================================================

#[test]
fn brightness_contrast() {
    let img = test_image();
    let result = apply_and_extract(&img, |s, flat| {
        brightness_contrast_from_flat(s, 0, 30.0, 20.0, flat);
    });
    assert_golden("adjustments", "brightness_30_contrast_20", &result);
}

#[test]
fn brightness_contrast_identity() {
    let img = test_image();
    let result = apply_and_extract(&img, |s, flat| {
        brightness_contrast_from_flat(s, 0, 0.0, 0.0, flat);
    });
    assert_eq!(img, result, "brightness=0, contrast=0 should be identity");
}

#[test]
fn hsl_adjustment() {
    let img = test_image();
    let result = apply_and_extract(&img, |s, flat| {
        hue_saturation_lightness_from_flat(s, 0, 30.0, -20.0, 10.0, flat);
    });
    assert_golden("adjustments", "hsl_h30_s-20_l10", &result);
}

#[test]
fn hsl_identity() {
    let img = test_image();
    let result = apply_and_extract(&img, |s, flat| {
        hue_saturation_lightness_from_flat(s, 0, 0.0, 0.0, 0.0, flat);
    });
    assert_eq!(img, result, "hsl(0,0,0) should be identity");
}

#[test]
fn exposure() {
    let img = test_image();
    let result = apply_and_extract(&img, |s, flat| {
        exposure_from_flat(s, 0, 1.0, flat);
    });
    assert_golden("adjustments", "exposure_1ev", &result);
}

#[test]
fn exposure_identity() {
    let img = test_image();
    let result = apply_and_extract(&img, |s, flat| {
        exposure_from_flat(s, 0, 0.0, flat);
    });
    assert_eq!(img, result, "exposure=0 should be identity");
}

#[test]
fn highlights_shadows() {
    let img = test_image();
    let result = apply_and_extract(&img, |s, flat| {
        highlights_shadows_from_flat(s, 0, 30.0, -20.0, flat);
    });
    assert_golden("adjustments", "highlights_shadows", &result);
}

#[test]
fn highlights_shadows_identity() {
    let img = test_image();
    let result = apply_and_extract(&img, |s, flat| {
        highlights_shadows_from_flat(s, 0, 0.0, 0.0, flat);
    });
    assert_eq!(img, result, "highlights=0, shadows=0 should be identity");
}

#[test]
fn levels() {
    let img = test_image();
    let result = apply_and_extract(&img, |s, flat| {
        levels_from_flat(s, 0, 20.0, 235.0, 1.2, 0.0, 255.0, flat);
    });
    assert_golden("adjustments", "levels", &result);
}

#[test]
fn levels_identity() {
    let img = test_image();
    let result = apply_and_extract(&img, |s, flat| {
        levels_from_flat(s, 0, 0.0, 255.0, 1.0, 0.0, 255.0, flat);
    });
    assert_eq!(img, result, "levels defaults should be identity");
}

#[test]
fn temperature_tint() {
    let img = test_image();
    let result = apply_and_extract(&img, |s, flat| {
        temperature_tint_from_flat(s, 0, 30.0, 10.0, flat);
    });
    assert_golden("adjustments", "temperature_tint", &result);
}

#[test]
fn temperature_tint_identity() {
    let img = test_image();
    let result = apply_and_extract(&img, |s, flat| {
        temperature_tint_from_flat(s, 0, 0.0, 0.0, flat);
    });
    assert_eq!(img, result, "temp=0, tint=0 should be identity");
}

#[test]
fn curves_identity() {
    let img = test_image();
    // No channels enabled — should be identity
    let empty: &[(f32, f32)] = &[];
    let channels: [(&[(f32, f32)], bool); 5] = [
        (empty, false), // RGB master disabled
        (empty, false), // R
        (empty, false), // G
        (empty, false), // B
        (empty, false), // A
    ];
    let result = apply_and_extract(&img, |s, flat| {
        curves_from_flat_multi(s, 0, &channels, flat);
    });
    assert_eq!(
        img, result,
        "all curves channels disabled should be identity"
    );
}

#[test]
fn threshold() {
    let img = test_image();
    let result = apply_and_extract(&img, |s, flat| {
        threshold_from_flat(s, 0, 128.0, flat);
    });
    assert_golden("adjustments", "threshold_128", &result);
}

#[test]
fn posterize() {
    let img = test_image();
    let result = apply_and_extract(&img, |s, flat| {
        posterize_from_flat(s, 0, 4, flat);
    });
    assert_golden("adjustments", "posterize_4", &result);
}

#[test]
fn color_balance() {
    let img = test_image();
    let result = apply_and_extract(&img, |s, flat| {
        color_balance_from_flat(
            s,
            0,
            [10.0, 0.0, -10.0], // shadows: warm
            [0.0, 0.0, 0.0],    // midtones: neutral
            [-10.0, 0.0, 10.0], // highlights: cool
            flat,
        );
    });
    assert_golden("adjustments", "color_balance", &result);
}

#[test]
fn color_balance_identity() {
    let img = test_image();
    let result = apply_and_extract(&img, |s, flat| {
        color_balance_from_flat(
            s,
            0,
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            flat,
        );
    });
    assert_eq!(img, result, "color balance all zeros should be identity");
}

#[test]
fn gradient_map() {
    let img = test_image();
    // Simple warm gradient map: black → dark red → orange → yellow → white
    let mut lut = [[0u8; 4]; 256];
    for i in 0..256 {
        let t = i as f32 / 255.0;
        lut[i] = [
            (t * 255.0) as u8,
            (t * t * 200.0) as u8,
            (t * t * t * 150.0) as u8,
            255,
        ];
    }
    let result = apply_and_extract(&img, |s, flat| {
        gradient_map_from_flat(s, 0, &lut, flat);
    });
    assert_golden("adjustments", "gradient_map", &result);
}

#[test]
fn black_and_white() {
    let img = create_color_bands(64, 64);
    let result = apply_and_extract(&img, |s, flat| {
        black_and_white_from_flat(s, 0, 0.3, 0.59, 0.11, flat);
    });
    assert_golden("adjustments", "black_and_white", &result);
}

#[test]
fn vibrance() {
    let img = test_image();
    let result = apply_and_extract(&img, |s, flat| {
        vibrance_from_flat(s, 0, 50.0, flat);
    });
    assert_golden("adjustments", "vibrance_50", &result);
}

#[test]
fn vibrance_identity() {
    let img = test_image();
    let result = apply_and_extract(&img, |s, flat| {
        vibrance_from_flat(s, 0, 0.0, flat);
    });
    assert_eq!(img, result, "vibrance=0 should be identity");
}
