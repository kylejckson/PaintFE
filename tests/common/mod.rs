// =============================================================================
// PaintFE Test Helpers — shared utilities for all test tiers
// =============================================================================

#![allow(dead_code)]

use image::{Rgba, RgbaImage};
use std::path::{Path, PathBuf};

// =============================================================================
// Paths
// =============================================================================

/// Root of the golden reference images.
pub fn golden_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("golden")
}

/// Root of the test output (failure artefacts). Git-ignored.
pub fn output_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("output")
}

/// Root of committed test fixture files.
pub fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
}

// =============================================================================
// Image comparison
// =============================================================================

/// Result of comparing two images pixel-by-pixel.
#[derive(Debug)]
pub struct CompareResult {
    pub matches: bool,
    pub total_pixels: u64,
    pub mismatched_pixels: u64,
    pub max_channel_diff: u8,
    pub mean_channel_diff: f64,
    pub mismatch_percentage: f64,
    pub dimensions_match: bool,
    pub actual_size: (u32, u32),
    pub expected_size: (u32, u32),
}

/// Compare two images pixel-by-pixel. `tolerance` is the maximum allowed
/// per-channel difference (0 = pixel-exact).
pub fn compare_images(actual: &RgbaImage, expected: &RgbaImage, tolerance: u8) -> CompareResult {
    let actual_size = (actual.width(), actual.height());
    let expected_size = (expected.width(), expected.height());
    let dimensions_match = actual_size == expected_size;

    if !dimensions_match {
        return CompareResult {
            matches: false,
            total_pixels: 0,
            mismatched_pixels: 0,
            max_channel_diff: 255,
            mean_channel_diff: 255.0,
            mismatch_percentage: 100.0,
            dimensions_match,
            actual_size,
            expected_size,
        };
    }

    let total_pixels = (actual.width() as u64) * (actual.height() as u64);
    let mut mismatched_pixels = 0u64;
    let mut max_channel_diff: u8 = 0;
    let mut sum_diff: f64 = 0.0;

    for (a, e) in actual.pixels().zip(expected.pixels()) {
        let dr = (a[0] as i16 - e[0] as i16).unsigned_abs() as u8;
        let dg = (a[1] as i16 - e[1] as i16).unsigned_abs() as u8;
        let db = (a[2] as i16 - e[2] as i16).unsigned_abs() as u8;
        let da = (a[3] as i16 - e[3] as i16).unsigned_abs() as u8;
        let pixel_max = dr.max(dg).max(db).max(da);
        if pixel_max > tolerance {
            mismatched_pixels += 1;
            sum_diff += pixel_max as f64;
        }
        if pixel_max > max_channel_diff {
            max_channel_diff = pixel_max;
        }
    }

    let mismatch_percentage = if total_pixels > 0 {
        (mismatched_pixels as f64 / total_pixels as f64) * 100.0
    } else {
        0.0
    };
    let mean_channel_diff = if mismatched_pixels > 0 {
        sum_diff / mismatched_pixels as f64
    } else {
        0.0
    };

    CompareResult {
        matches: mismatched_pixels == 0,
        total_pixels,
        mismatched_pixels,
        max_channel_diff,
        mean_channel_diff,
        mismatch_percentage,
        dimensions_match,
        actual_size,
        expected_size,
    }
}

/// Generate a visual diff image. Green = match, red tint = mismatch (brighter = larger diff).
pub fn generate_diff_image(actual: &RgbaImage, expected: &RgbaImage) -> RgbaImage {
    let w = actual.width().max(expected.width());
    let h = actual.height().max(expected.height());
    let mut diff = RgbaImage::new(w, h);

    for y in 0..h {
        for x in 0..w {
            let in_actual = x < actual.width() && y < actual.height();
            let in_expected = x < expected.width() && y < expected.height();

            let pixel = if in_actual && in_expected {
                let a = actual.get_pixel(x, y);
                let e = expected.get_pixel(x, y);
                let dr = (a[0] as i16 - e[0] as i16).unsigned_abs() as u8;
                let dg = (a[1] as i16 - e[1] as i16).unsigned_abs() as u8;
                let db = (a[2] as i16 - e[2] as i16).unsigned_abs() as u8;
                let da = (a[3] as i16 - e[3] as i16).unsigned_abs() as u8;
                let max_d = dr.max(dg).max(db).max(da);
                if max_d == 0 {
                    Rgba([0, 128, 0, 255]) // green = match
                } else {
                    // Red, brightness proportional to difference
                    let intensity = ((max_d as f32 / 255.0).sqrt() * 255.0) as u8;
                    Rgba([
                        255,
                        255_u8.saturating_sub(intensity),
                        255_u8.saturating_sub(intensity),
                        255,
                    ])
                }
            } else if in_actual {
                Rgba([255, 128, 0, 255]) // orange = in actual only (larger)
            } else {
                Rgba([0, 128, 255, 255]) // blue = in expected only
            };
            diff.put_pixel(x, y, pixel);
        }
    }
    diff
}

/// Save failure artefacts (actual image + diff) to `tests/output/`.
pub fn save_failure_artifacts(test_name: &str, actual: &RgbaImage, expected: &RgbaImage) {
    let out = output_dir();
    std::fs::create_dir_all(&out).ok();
    let actual_path = out.join(format!("{}_actual.png", test_name));
    let diff_path = out.join(format!("{}_diff.png", test_name));
    actual.save(&actual_path).ok();
    let diff = generate_diff_image(actual, expected);
    diff.save(&diff_path).ok();
}

// =============================================================================
// Golden file management
// =============================================================================

/// Returns true when the `GENERATE_GOLDEN` env var is set (any non-empty value).
pub fn should_generate_golden() -> bool {
    std::env::var("GENERATE_GOLDEN").is_ok_and(|v| !v.is_empty())
}

/// Golden-file tolerance from env, or default 0 (pixel-exact).
pub fn golden_tolerance() -> u8 {
    std::env::var("GOLDEN_TOLERANCE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0)
}

/// Load a golden reference image. Returns `None` if the file doesn't exist.
pub fn load_golden(category: &str, name: &str) -> Option<RgbaImage> {
    let path = golden_dir().join(category).join(format!("{}.png", name));
    image::open(&path).ok().map(|img| img.to_rgba8())
}

/// Save a golden reference image, creating directories as needed.
pub fn save_golden(category: &str, name: &str, img: &RgbaImage) {
    let dir = golden_dir().join(category);
    std::fs::create_dir_all(&dir).expect("failed to create golden dir");
    let path = dir.join(format!("{}.png", name));
    img.save(&path).expect("failed to save golden image");
}

// =============================================================================
// Golden-file assertion helper
// =============================================================================

/// Compare `actual` against the golden file `{category}/{name}.png`.
///
/// - If `GENERATE_GOLDEN=1`, writes `actual` as the new golden and passes.
/// - Otherwise loads the existing golden and asserts pixel-exact match (or
///   within `GOLDEN_TOLERANCE`). On failure, saves artefacts to `tests/output/`.
pub fn assert_golden(category: &str, name: &str, actual: &RgbaImage) {
    if should_generate_golden() {
        save_golden(category, name, actual);
        eprintln!(
            "[golden] wrote {}/{}.png ({}×{})",
            category,
            name,
            actual.width(),
            actual.height()
        );
        return;
    }

    let expected = load_golden(category, name).unwrap_or_else(|| {
        panic!(
            "Golden file not found: {}/{}.png — run with GENERATE_GOLDEN=1 to create it",
            category, name,
        );
    });

    let tolerance = golden_tolerance();
    let result = compare_images(actual, &expected, tolerance);

    if !result.matches {
        save_failure_artifacts(&format!("{}_{}", category, name), actual, &expected);
        panic!(
            "FAILED: {}/{}\n\
             \x20 Dimensions: {}×{} vs {}×{} ({})\n\
             \x20 Mismatched pixels: {} / {} ({:.2}%)\n\
             \x20 Max channel diff: {}\n\
             \x20 Mean channel diff: {:.1}\n\
             \x20 Tolerance: {}\n\
             \x20 Artefacts saved to tests/output/",
            category,
            name,
            result.actual_size.0,
            result.actual_size.1,
            result.expected_size.0,
            result.expected_size.1,
            if result.dimensions_match {
                "match"
            } else {
                "MISMATCH"
            },
            result.mismatched_pixels,
            result.total_pixels,
            result.mismatch_percentage,
            result.max_channel_diff,
            result.mean_channel_diff,
            tolerance,
        );
    }
}

// =============================================================================
// Test image generators (deterministic, no randomness)
// =============================================================================

/// Create a `w×h` image with a horizontal red→green gradient and vertical blue gradient.
/// Fully opaque. Pixel at (x, y) = (r, g, b, 255) where:
///   r = x * 255 / (w-1), g = 255 - r, b = y * 255 / (h-1)
pub fn create_test_gradient(w: u32, h: u32) -> RgbaImage {
    let mut img = RgbaImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let r = if w > 1 {
                (x * 255 / (w - 1)) as u8
            } else {
                128
            };
            let g = 255 - r;
            let b = if h > 1 {
                (y * 255 / (h - 1)) as u8
            } else {
                128
            };
            img.put_pixel(x, y, Rgba([r, g, b, 255]));
        }
    }
    img
}

/// Create a `w×h` checkerboard (8-pixel cells). Fully opaque.
/// Cell (0,0) = white, cell (1,0) = black, alternating.
pub fn create_test_checkerboard(w: u32, h: u32) -> RgbaImage {
    let mut img = RgbaImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let cell_x = x / 8;
            let cell_y = y / 8;
            let white = (cell_x + cell_y) % 2 == 0;
            let v = if white { 255u8 } else { 0u8 };
            img.put_pixel(x, y, Rgba([v, v, v, 255]));
        }
    }
    img
}

/// Create a `w×h` solid-colour image.
pub fn create_solid(w: u32, h: u32, color: [u8; 4]) -> RgbaImage {
    RgbaImage::from_pixel(w, h, Rgba(color))
}

/// Create a `w×h` fully transparent image.
pub fn create_transparent(w: u32, h: u32) -> RgbaImage {
    RgbaImage::new(w, h) // default is [0,0,0,0]
}

/// Create a test image with distinct colour bands for testing colour-space operations.
/// 8 vertical bands: red, green, blue, cyan, magenta, yellow, white, black.
pub fn create_color_bands(w: u32, h: u32) -> RgbaImage {
    let mut img = RgbaImage::new(w, h);
    let colors: [[u8; 4]; 8] = [
        [255, 0, 0, 255],     // red
        [0, 255, 0, 255],     // green
        [0, 0, 255, 255],     // blue
        [0, 255, 255, 255],   // cyan
        [255, 0, 255, 255],   // magenta
        [255, 255, 0, 255],   // yellow
        [255, 255, 255, 255], // white
        [0, 0, 0, 255],       // black
    ];
    for y in 0..h {
        for x in 0..w {
            let band = ((x as usize) * 8 / w as usize).min(7);
            img.put_pixel(x, y, Rgba(colors[band]));
        }
    }
    img
}

/// Create a simple `CanvasState` with one layer containing the given image.
pub fn canvas_from_image(img: &RgbaImage) -> paintfe::canvas::CanvasState {
    let w = img.width();
    let h = img.height();
    let mut state = paintfe::canvas::CanvasState::new(w, h);
    state.layers[0].pixels = paintfe::canvas::TiledImage::from_rgba_image(img);
    state
}

/// Extract the active layer as a flat `RgbaImage`.
pub fn extract_layer(state: &paintfe::canvas::CanvasState, layer: usize) -> RgbaImage {
    let raw = state.layers[layer]
        .pixels
        .extract_region_rgba(0, 0, state.width, state.height);
    RgbaImage::from_raw(state.width, state.height, raw).expect("extract_layer: invalid dimensions")
}
