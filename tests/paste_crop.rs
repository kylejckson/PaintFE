// Tests for the paste overlay content crop (flat crop handles) and the
// "Select Layer Content Bounds" helper.

use image::{Rgba, RgbaImage};
use paintfe::canvas::{CanvasState, TiledImage};
use paintfe::ops::adjustments::select_layer_content_bounds;
use paintfe::ops::clipboard::PasteOverlay;

fn solid_image(w: u32, h: u32, px: Rgba<u8>) -> RgbaImage {
    RgbaImage::from_pixel(w, h, px)
}

#[test]
fn paste_overlay_starts_uncropped() {
    let overlay = PasteOverlay::new(solid_image(8, 4, Rgba([1, 2, 3, 255])), 64, 64);
    assert_eq!(overlay.crop_rect, (0, 0, 8, 4));
    assert_eq!(overlay.content_size(), (8, 4));
    assert!(overlay.allow_crop);
}

#[test]
fn set_crop_rect_clamps_to_source_and_keeps_min_size() {
    let mut overlay = PasteOverlay::new(solid_image(8, 4, Rgba([1, 2, 3, 255])), 64, 64);
    overlay.set_crop_rect((2, 1, 6, 3));
    assert_eq!(overlay.crop_rect, (2, 1, 6, 3));
    assert_eq!(overlay.content_size(), (4, 2));

    // Out-of-bounds crop is clamped to the source.
    overlay.set_crop_rect((0, 0, 100, 100));
    assert_eq!(overlay.crop_rect, (0, 0, 8, 4));

    // A degenerate crop keeps at least one content pixel.
    overlay.set_crop_rect((8, 4, 8, 4));
    assert_eq!(overlay.crop_rect, (7, 3, 8, 4));
    assert_eq!(overlay.content_size(), (1, 1));
}

#[test]
fn crop_is_part_of_transform_undo() {
    let mut overlay = PasteOverlay::new(solid_image(8, 4, Rgba([1, 2, 3, 255])), 64, 64);
    let before = overlay.transform();
    overlay.set_crop_rect((1, 1, 5, 3));
    let after = overlay.transform();
    assert_ne!(before, after);

    // Undo → full source, redo → cropped (crop round-trips like the transform).
    overlay.set_transform(before);
    assert_eq!(overlay.crop_rect, (0, 0, 8, 4));
    overlay.set_transform(after);
    assert_eq!(overlay.crop_rect, (1, 1, 5, 3));
}

#[test]
fn reset_crop_restores_full_source() {
    let mut overlay = PasteOverlay::new(solid_image(8, 4, Rgba([1, 2, 3, 255])), 64, 64);
    overlay.set_crop_rect((1, 1, 5, 3));
    overlay.reset_crop();
    assert_eq!(overlay.crop_rect, (0, 0, 8, 4));
    assert_eq!(overlay.content_size(), (8, 4));
}

#[test]
fn select_layer_content_bounds_matches_alpha_bbox() {
    let mut state = CanvasState::new(16, 16);
    state.layers[0].pixels = TiledImage::new(16, 16);
    // Opaque 4x2 block at (5, 7)..(8, 8).
    for y in 7..9 {
        for x in 5..9 {
            state.layers[0]
                .pixels
                .put_pixel(x, y, Rgba([255, 0, 0, 255]));
        }
    }
    assert!(select_layer_content_bounds(&mut state));
    let mask = state.selection_mask.as_ref().expect("selection created");
    assert!(mask.get_pixel(5, 7)[0] > 0);
    assert!(mask.get_pixel(8, 8)[0] > 0);
    assert_eq!(mask.get_pixel(4, 7)[0], 0, "no selection left of the bbox");
    assert_eq!(mask.get_pixel(9, 8)[0], 0, "no selection right of the bbox");
    assert_eq!(mask.get_pixel(5, 6)[0], 0, "no selection above the bbox");
    assert_eq!(mask.get_pixel(5, 9)[0], 0, "no selection below the bbox");
}

#[test]
fn select_layer_content_bounds_ignores_transparent_layer() {
    let mut state = CanvasState::new(8, 8);
    state.layers[0].pixels = TiledImage::new(8, 8);
    assert!(!select_layer_content_bounds(&mut state));
    assert!(state.selection_mask.is_none());
}
