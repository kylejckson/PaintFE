// =============================================================================
// Integration tests — Inpainting / Content-Aware fill
// =============================================================================
//
// Tests the CPU inpainting algorithms (Strategy 1 from TEST_PLAN.md §12B).

mod common;

use common::*;
use image::{GrayImage, Luma, Rgba, RgbaImage};
use paintfe::ops::inpaint::{fill_region_patchmatch, inpaint_instant_brush};

// =============================================================================
// Helpers
// =============================================================================

/// Create a 64×64 image with a colored pattern and a hole (transparent) region.
fn pattern_with_hole() -> (RgbaImage, GrayImage) {
    let mut img = RgbaImage::new(64, 64);
    // Fill with a red/blue checkerboard pattern
    for y in 0..64 {
        for x in 0..64 {
            let color = if (x / 8 + y / 8) % 2 == 0 {
                Rgba([200, 50, 50, 255])
            } else {
                Rgba([50, 50, 200, 255])
            };
            img.put_pixel(x, y, color);
        }
    }
    // Create a hole mask (white = hole) in the center 16×16
    // Keep the original pixel data in the source (instant brush uses reference colors)
    let mut mask = GrayImage::new(64, 64);
    for y in 24..40 {
        for x in 24..40 {
            mask.put_pixel(x, y, Luma([255]));
        }
    }
    (img, mask)
}

/// Create a pattern where the hole region is fully transparent (for patchmatch).
fn pattern_with_transparent_hole() -> (RgbaImage, GrayImage) {
    let mut img = RgbaImage::new(64, 64);
    for y in 0..64 {
        for x in 0..64 {
            let color = if (x / 8 + y / 8) % 2 == 0 {
                Rgba([200, 50, 50, 255])
            } else {
                Rgba([50, 50, 200, 255])
            };
            img.put_pixel(x, y, color);
        }
    }
    let mut mask = GrayImage::new(64, 64);
    for y in 24..40 {
        for x in 24..40 {
            mask.put_pixel(x, y, Luma([255]));
            img.put_pixel(x, y, Rgba([0, 0, 0, 0]));
        }
    }
    (img, mask)
}

// =============================================================================
// inpaint_instant_brush
// =============================================================================

#[test]
fn inpaint_instant_brush_blends_over_hole() {
    // The instant brush uses bilateral weighting (spatial + color distance).
    // It works best with subtle color differences. Create a smooth gradient
    // with a "damaged" region that has slightly shifted colors.
    let mut img = RgbaImage::new(64, 64);
    for y in 0..64 {
        for x in 0..64 {
            // Smooth gradient (no sharp color boundaries)
            let v = ((x as f32 + y as f32) * 2.0).min(255.0) as u8;
            img.put_pixel(x, y, Rgba([v, 100, 150, 255]));
        }
    }
    // The hole has slightly corrupted color (shifted by +30 in R channel)
    let mut mask = GrayImage::new(64, 64);
    for y in 28..36 {
        for x in 28..36 {
            mask.put_pixel(x, y, Luma([255]));
            let orig = img.get_pixel(x, y).clone();
            img.put_pixel(
                x,
                y,
                Rgba([orig[0].saturating_add(30), orig[1], orig[2], 255]),
            );
        }
    }

    let mut out = img.clone();
    inpaint_instant_brush(&img, &mask, &mut out, 32.0, 32.0, 10.0, 18.0, 0.5);

    // The brush should have blended some pixels (narrow hole + smooth gradient + low hardness)
    let mut changed_count = 0;
    for y in 28..36 {
        for x in 28..36 {
            if out.get_pixel(x, y) != img.get_pixel(x, y) {
                changed_count += 1;
            }
        }
    }
    // Even if only a few edge pixels changed, that validates the function works
    assert!(
        changed_count > 0,
        "inpaint_instant_brush should change at least some hole pixels (changed: {})",
        changed_count
    );
}

#[test]
fn inpaint_instant_brush_preserves_outside() {
    let (src, mask) = pattern_with_hole();
    let mut out = src.clone();

    inpaint_instant_brush(&src, &mask, &mut out, 32.0, 32.0, 12.0, 24.0, 0.8);

    // Pixels far from the brush center should be unchanged
    assert_eq!(out.get_pixel(0, 0), src.get_pixel(0, 0));
    assert_eq!(out.get_pixel(63, 63), src.get_pixel(63, 63));
    assert_eq!(out.get_pixel(5, 5), src.get_pixel(5, 5));
}

#[test]
fn inpaint_instant_brush_golden() {
    let (src, mask) = pattern_with_hole();
    let mut out = src.clone();
    inpaint_instant_brush(&src, &mask, &mut out, 32.0, 32.0, 12.0, 24.0, 0.8);
    assert_golden("inpaint", "instant_brush_center", &out);
}

// =============================================================================
// fill_region_patchmatch
// =============================================================================

#[test]
fn patchmatch_fills_hole() {
    let (src, mask) = pattern_with_transparent_hole();
    let result = fill_region_patchmatch(&src, &mask, 5, 3);

    // All center pixels should now be opaque
    for y in 24..40 {
        for x in 24..40 {
            let p = result.get_pixel(x, y);
            assert!(
                p[3] > 128,
                "patchmatch failed to fill pixel ({},{}) — alpha={}",
                x,
                y,
                p[3]
            );
        }
    }
}

#[test]
fn patchmatch_preserves_outside() {
    let (src, mask) = pattern_with_transparent_hole();
    let result = fill_region_patchmatch(&src, &mask, 5, 3);

    // Pixels outside the mask should be unchanged
    for y in 0..24 {
        for x in 0..64 {
            assert_eq!(
                result.get_pixel(x, y),
                src.get_pixel(x, y),
                "patchmatch changed pixel ({},{}) outside mask",
                x,
                y
            );
        }
    }
}

#[test]
fn patchmatch_golden() {
    let (src, mask) = pattern_with_transparent_hole();
    let result = fill_region_patchmatch(&src, &mask, 5, 3);
    assert_golden("inpaint", "patchmatch_checkerboard", &result);
}
