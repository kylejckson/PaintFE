// =============================================================================
// Integration tests — File I/O roundtrips
// =============================================================================
//
// Tests save → load → compare for all supported image formats and PFE project
// files. Each test writes to a temp directory and verifies the loaded result
// matches the original within a format-appropriate tolerance.

mod common;

use common::*;
use image::{Rgba, RgbaImage};
use paintfe::canvas::{CanvasState, Layer, TiledImage};
use paintfe::components::dialogs::{SaveFormat, TiffCompression};
use paintfe::io::{encode_and_write, load_image_sync, load_pfe, save_pfe};
use std::path::PathBuf;

/// Temp directory for this test run, auto-cleaned.
fn temp_dir() -> PathBuf {
    let dir = std::env::temp_dir().join("paintfe_io_tests");
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Standard 64×64 test image with full color range.
fn test_image() -> RgbaImage {
    create_test_gradient(64, 64)
}

/// Helper: save an image, load it back, return the loaded composite.
fn roundtrip_format(img: &RgbaImage, name: &str, format: SaveFormat, quality: u8, tolerance: u8) {
    let path = temp_dir().join(name);
    encode_and_write(img, &path, format, quality, TiffCompression::None).unwrap();

    let loaded_state = load_image_sync(&path).unwrap();
    let loaded = loaded_state.composite();

    let diff = compare_images(&loaded, img, tolerance);
    assert!(
        diff.matches,
        "{}: roundtrip failed — max channel diff = {}, mismatched = {}/{} ({:.2}%)",
        name,
        diff.max_channel_diff,
        diff.mismatched_pixels,
        diff.total_pixels,
        diff.mismatch_percentage,
    );

    // Clean up
    let _ = std::fs::remove_file(&path);
}

// =============================================================================
// Lossless formats (pixel-exact roundtrip)
// =============================================================================

#[test]
fn roundtrip_png() {
    roundtrip_format(&test_image(), "rt.png", SaveFormat::Png, 95, 0);
}

#[test]
fn roundtrip_bmp() {
    roundtrip_format(&test_image(), "rt.bmp", SaveFormat::Bmp, 95, 0);
}

#[test]
fn roundtrip_tga() {
    roundtrip_format(&test_image(), "rt.tga", SaveFormat::Tga, 95, 0);
}

#[test]
fn roundtrip_tiff() {
    roundtrip_format(&test_image(), "rt.tiff", SaveFormat::Tiff, 95, 0);
}

// =============================================================================
// Lossy formats (tolerance needed)
// =============================================================================

#[test]
fn roundtrip_jpeg() {
    // JPEG is lossy — allow up to 10 per-channel difference at q95
    roundtrip_format(&test_image(), "rt.jpg", SaveFormat::Jpeg, 95, 10);
}

#[test]
fn roundtrip_webp() {
    // WebP lossy at q95
    roundtrip_format(&test_image(), "rt.webp", SaveFormat::Webp, 95, 10);
}

// =============================================================================
// GIF (quantized to 256 colors)
// =============================================================================

#[test]
fn roundtrip_gif() {
    // GIF quantizes to 256 palette — allow wide tolerance
    roundtrip_format(&test_image(), "rt.gif", SaveFormat::Gif, 95, 55);
}

// =============================================================================
// PFE project file (layer-preserving roundtrip)
// =============================================================================

#[test]
fn roundtrip_pfe_single_layer() {
    let img = test_image();
    let state = canvas_from_image(&img);
    let path = temp_dir().join("rt_single.pfe");

    save_pfe(&state, &path).unwrap();
    let loaded = load_pfe(&path).unwrap();

    assert_eq!(loaded.layers.len(), 1);
    assert_eq!(loaded.width, state.width);
    assert_eq!(loaded.height, state.height);

    let original_px = extract_layer(&state, 0);
    let loaded_px = extract_layer(&loaded, 0);
    assert_eq!(
        original_px, loaded_px,
        "PFE single layer should be pixel-exact"
    );

    let _ = std::fs::remove_file(&path);
}

#[test]
fn roundtrip_pfe_multi_layer() {
    let w = 64;
    let h = 64;

    let mut state = CanvasState::new(w, h);
    // Layer 0: white background (already created)

    // Layer 1: red semi-transparent overlay
    let mut red_img = RgbaImage::new(w, h);
    for p in red_img.pixels_mut() {
        *p = Rgba([255, 0, 0, 128]);
    }
    let mut layer1 = Layer::new("Red".into(), w, h, Rgba([0, 0, 0, 0]));
    layer1.pixels = TiledImage::from_rgba_image(&red_img);
    layer1.opacity = 0.75;
    state.layers.push(layer1);

    // Layer 2: gradient
    let grad = create_test_gradient(w, h);
    let mut layer2 = Layer::new("Gradient".into(), w, h, Rgba([0, 0, 0, 0]));
    layer2.pixels = TiledImage::from_rgba_image(&grad);
    state.layers.push(layer2);

    let path = temp_dir().join("rt_multi.pfe");
    save_pfe(&state, &path).unwrap();
    let loaded = load_pfe(&path).unwrap();

    assert_eq!(loaded.layers.len(), 3, "should preserve all 3 layers");
    assert_eq!(loaded.layers[1].opacity, 0.75);
    assert_eq!(loaded.layers[1].name, "Red");
    assert_eq!(loaded.layers[2].name, "Gradient");

    // Composite should match
    let original_comp = state.composite();
    let loaded_comp = loaded.composite();
    assert_eq!(
        original_comp, loaded_comp,
        "PFE composite should be pixel-exact"
    );

    let _ = std::fs::remove_file(&path);
}

#[test]
fn roundtrip_pfe_blend_modes() {
    use paintfe::canvas::BlendMode;

    let w = 32;
    let h = 32;
    let mut state = CanvasState::new(w, h);

    let mut overlay = Layer::new("Multiply".into(), w, h, Rgba([0, 0, 0, 0]));
    overlay.blend_mode = BlendMode::Multiply;
    let img = create_test_gradient(w, h);
    overlay.pixels = TiledImage::from_rgba_image(&img);
    state.layers.push(overlay);

    let path = temp_dir().join("rt_blend.pfe");
    save_pfe(&state, &path).unwrap();
    let loaded = load_pfe(&path).unwrap();

    assert_eq!(loaded.layers[1].blend_mode, BlendMode::Multiply);

    let _ = std::fs::remove_file(&path);
}

// =============================================================================
// load_image_sync dispatch
// =============================================================================

#[test]
fn load_png_via_load_image_sync() {
    let img = test_image();
    let path = temp_dir().join("load_test.png");
    encode_and_write(&img, &path, SaveFormat::Png, 95, TiffCompression::None).unwrap();

    let state = load_image_sync(&path).unwrap();
    assert_eq!(state.width, 64);
    assert_eq!(state.height, 64);
    assert_eq!(state.layers.len(), 1);

    let _ = std::fs::remove_file(&path);
}

#[test]
fn load_pfe_via_load_image_sync() {
    let state = canvas_from_image(&test_image());
    let path = temp_dir().join("load_test.pfe");
    save_pfe(&state, &path).unwrap();

    let loaded = load_image_sync(&path).unwrap();
    assert_eq!(loaded.width, 64);
    assert_eq!(loaded.height, 64);

    let _ = std::fs::remove_file(&path);
}
