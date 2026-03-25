// =============================================================================
// Integration tests — Text layer rasterization
// =============================================================================
//
// Tests TextLayerData construction, rasterization to TiledImage, multi-block,
// multi-run, and text effects. Font-dependent — requires Arial on the system.

mod common;

use paintfe::canvas::{CanvasState, Layer, LayerContent};
use paintfe::ops::text_layer::{
    ParagraphStyle, TextBlock, TextLayerData, TextRun, TextStyle, TextWarp,
};
use std::collections::HashMap;

/// Helper: rasterize a TextLayerData and return the TiledImage.
fn rasterize(td: &mut TextLayerData, w: u32, h: u32) -> paintfe::canvas::TiledImage {
    let mut coverage_buf = Vec::new();
    let mut glyph_cache = HashMap::new();
    td.rasterize(w, h, &mut coverage_buf, &mut glyph_cache)
}

/// Check that at least one pixel in the TiledImage has alpha > 0.
fn has_visible_pixels(tile: &paintfe::canvas::TiledImage, w: u32, h: u32) -> bool {
    for y in 0..h {
        for x in 0..w {
            if tile.get_pixel(x, y)[3] > 0 {
                return true;
            }
        }
    }
    false
}

/// Count pixels with alpha > 0.
fn count_visible(tile: &paintfe::canvas::TiledImage, w: u32, h: u32) -> u32 {
    let mut count = 0u32;
    for y in 0..h {
        for x in 0..w {
            if tile.get_pixel(x, y)[3] > 0 {
                count += 1;
            }
        }
    }
    count
}

// =============================================================================
// Basic rasterization
// =============================================================================

#[test]
fn empty_text_produces_no_pixels() {
    let mut td = TextLayerData::default();
    // Default has one block with empty text
    td.mark_dirty();
    let pixels = rasterize(&mut td, 200, 200);
    assert!(!has_visible_pixels(&pixels, 200, 200));
}

#[test]
fn simple_text_produces_pixels() {
    let mut td = TextLayerData::default();
    td.blocks[0].runs[0].text = "Hello".into();
    td.mark_dirty();
    let pixels = rasterize(&mut td, 400, 200);
    assert!(
        has_visible_pixels(&pixels, 400, 200),
        "rasterizing 'Hello' should produce visible pixels"
    );
}

#[test]
fn text_color_appears_in_output() {
    let mut td = TextLayerData::default();
    td.blocks[0].runs[0].text = "Red".into();
    td.blocks[0].runs[0].style.color = [255, 0, 0, 255];
    td.blocks[0].runs[0].style.font_size = 72.0;
    td.mark_dirty();

    let pixels = rasterize(&mut td, 400, 200);
    // Find a pixel that has red in it
    let mut found_red = false;
    for y in 0..200 {
        for x in 0..400 {
            let p = pixels.get_pixel(x, y);
            if p[0] > 200 && p[1] < 50 && p[3] > 128 {
                found_red = true;
                break;
            }
        }
        if found_red {
            break;
        }
    }
    assert!(found_red, "should find red-colored text pixels");
}

// =============================================================================
// Font size
// =============================================================================

#[test]
fn larger_font_produces_more_pixels() {
    let mut td_small = TextLayerData::default();
    td_small.blocks[0].runs[0].text = "X".into();
    td_small.blocks[0].runs[0].style.font_size = 20.0;
    td_small.mark_dirty();
    let small = rasterize(&mut td_small, 400, 200);
    let small_count = count_visible(&small, 400, 200);

    let mut td_large = TextLayerData::default();
    td_large.blocks[0].runs[0].text = "X".into();
    td_large.blocks[0].runs[0].style.font_size = 80.0;
    td_large.mark_dirty();
    let large = rasterize(&mut td_large, 400, 200);
    let large_count = count_visible(&large, 400, 200);

    assert!(
        large_count > small_count * 2,
        "80pt 'X' ({large_count} px) should have far more pixels than 20pt ({small_count} px)"
    );
}

// =============================================================================
// Multi-block
// =============================================================================

#[test]
fn multi_block_both_rasterized() {
    let mut td = TextLayerData::default();
    td.blocks[0].position = [10.0, 10.0];
    td.blocks[0].runs[0].text = "Top".into();

    td.blocks.push(TextBlock {
        id: 2,
        position: [10.0, 120.0],
        rotation: 0.0,
        runs: vec![TextRun {
            text: "Bottom".into(),
            style: TextStyle::default(),
        }],
        paragraph: ParagraphStyle::default(),
        max_width: None,
        max_height: None,
        warp: TextWarp::None,
        glyph_overrides: Vec::new(),
        cached_raster: None,
    });
    td.mark_dirty();

    let pixels = rasterize(&mut td, 400, 200);

    // Check top section has pixels
    let mut top_has = false;
    for y in 0..80 {
        for x in 0..400 {
            if pixels.get_pixel(x, y)[3] > 0 {
                top_has = true;
                break;
            }
        }
        if top_has {
            break;
        }
    }

    // Check bottom section has pixels
    let mut bottom_has = false;
    for y in 100..200 {
        for x in 0..400 {
            if pixels.get_pixel(x, y)[3] > 0 {
                bottom_has = true;
                break;
            }
        }
        if bottom_has {
            break;
        }
    }

    assert!(top_has, "top block should produce pixels");
    assert!(bottom_has, "bottom block should produce pixels");
}

// =============================================================================
// Multi-run (rich text)
// =============================================================================

#[test]
fn multi_run_block_rasterizes() {
    let mut td = TextLayerData::default();
    td.blocks[0].runs = vec![
        TextRun {
            text: "Bold".into(),
            style: TextStyle {
                font_weight: 700,
                ..TextStyle::default()
            },
        },
        TextRun {
            text: " Normal".into(),
            style: TextStyle::default(),
        },
    ];
    td.mark_dirty();

    let pixels = rasterize(&mut td, 500, 200);
    assert!(
        has_visible_pixels(&pixels, 500, 200),
        "multi-run text should render"
    );
}

// =============================================================================
// Dirty tracking / caching
// =============================================================================

#[test]
fn needs_rasterize_after_mark_dirty() {
    let mut td = TextLayerData::default();
    td.blocks[0].runs[0].text = "Test".into();
    td.mark_dirty();
    assert!(td.needs_rasterize());

    // Rasterize, then simulate what ensure_text_layers_rasterized does by
    // updating raster_generation to match cache_generation.
    let _pixels = rasterize(&mut td, 200, 200);
    td.raster_generation = td.cache_generation;
    assert!(!td.needs_rasterize(), "should be clean after rasterize");

    td.mark_dirty();
    assert!(td.needs_rasterize(), "should need rasterize after dirty");
}

// =============================================================================
// Text layer in CanvasState
// =============================================================================

#[test]
fn text_layer_in_canvas() {
    let mut state = CanvasState::new(400, 200);

    let mut layer = Layer::new_text("Text".into(), 400, 200);
    if let LayerContent::Text(ref mut td) = layer.content {
        td.blocks[0].runs[0].text = "Canvas Text".into();
        td.blocks[0].runs[0].style.color = [0, 0, 0, 255]; // Black on white bg
        td.mark_dirty();
    }

    // Manually rasterize before compositing
    {
        let mut coverage_buf = Vec::new();
        let mut glyph_cache = HashMap::new();
        if let LayerContent::Text(ref mut td) = layer.content {
            layer.pixels = td.rasterize(400, 200, &mut coverage_buf, &mut glyph_cache);
        }
    }

    state.layers.push(layer);

    let comp = state.composite();
    // Should not be pure white anymore — text was rendered
    let mut has_dark = false;
    for y in 0..200 {
        for x in 0..400 {
            let p = comp.get_pixel(x, y);
            if p[0] < 200 {
                has_dark = true;
                break;
            }
        }
        if has_dark {
            break;
        }
    }
    assert!(
        has_dark,
        "composited canvas should show black text on white background"
    );
}
