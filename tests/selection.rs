// =============================================================================
// Integration tests — Selection masks
// =============================================================================
//
// Tests selection shape generation, boolean operations (add/subtract/intersect),
// translate, fill, delete, and clear on CanvasState.

mod common;

use image::Rgba;
use paintfe::canvas::{CanvasState, SelectionMode, SelectionShape};

#[test]
fn selection_modifiers_match_a_disk_and_preserve_original_soft_coverage() {
    use paintfe::ops::adjustments::{contract_selection, expand_selection, feather_selection};

    let (width, height) = (23, 21);
    let source = image::GrayImage::from_fn(width, height, |x, y| {
        let alpha = match (x, y) {
            (0..=7, 0..=10) | (5..=14, 9..=17) => 255,
            (17, 11) => 99,
            _ => 0,
        };
        image::Luma([alpha])
    });
    for radius in [1i32, 3, 6] {
        let mut expanded = CanvasState::new(width, height);
        let mut contracted = CanvasState::new(width, height);
        let mut feathered = CanvasState::new(width, height);
        for state in [&mut expanded, &mut contracted, &mut feathered] {
            state.selection_mask = Some(source.clone());
        }
        expand_selection(&mut expanded, radius);
        contract_selection(&mut contracted, radius);
        feather_selection(&mut feathered, radius as f32);
        for (x, y, pixel) in source.enumerate_pixels() {
            let mut closest = None;
            let mut erode = false;
            for dy in -radius..=radius {
                for dx in -radius..=radius {
                    let distance = dx * dx + dy * dy;
                    if distance > radius * radius {
                        continue;
                    }
                    let (sx, sy) = (x as i32 + dx, y as i32 + dy);
                    let alpha = if sx >= 0 && sy >= 0 && sx < width as i32 && sy < height as i32 {
                        source.get_pixel(sx as u32, sy as u32).0[0]
                    } else {
                        0
                    };
                    if alpha == 0 {
                        erode = true;
                    } else if closest.is_none_or(|(best, _)| distance < best) {
                        closest = Some((distance, alpha));
                    }
                }
            }
            let original = pixel.0[0];
            let expected_expand = if original == 0 && closest.is_some() {
                255
            } else {
                original
            };
            let expected_contract = if erode { 0 } else { original };
            let expected_feather = if original != 0 {
                original
            } else {
                closest.map_or(0, |(distance, alpha)| {
                    let weight = 1.0 - (distance as f32).sqrt() / (radius + 1) as f32;
                    ((f32::from(alpha) * weight).round() as u8).max(1)
                })
            };
            assert_eq!(
                expanded.selection_mask.as_ref().unwrap().get_pixel(x, y).0[0],
                expected_expand
            );
            assert_eq!(
                contracted
                    .selection_mask
                    .as_ref()
                    .unwrap()
                    .get_pixel(x, y)
                    .0[0],
                expected_contract
            );
            assert_eq!(
                feathered.selection_mask.as_ref().unwrap().get_pixel(x, y).0[0],
                expected_feather
            );
        }
    }
}

#[test]
fn selection_modifiers_cross_tile_seams_and_contract_select_all_from_each_edge() {
    use paintfe::ops::adjustments::{contract_selection, expand_selection, feather_selection};

    let mut state = CanvasState::new(1100, 560);
    state.selection_all = true;
    contract_selection(&mut state, 200);
    assert_eq!(state.selection_mask_bounds(), Some((200, 200, 899, 359)));
    assert!(!state.selection_all);
    contract_selection(&mut state, 10);
    assert_eq!(state.selection_mask_bounds(), Some((210, 210, 889, 349)));
    let contracted = state.selection_mask.clone();
    expand_selection(&mut state, 60);
    let expanded = state.selection_mask.clone().unwrap();
    state.selection_mask = contracted;
    feather_selection(&mut state, 60.0);
    let feathered = state.selection_mask.unwrap();
    assert!(
        expanded
            .pixels()
            .zip(feathered.pixels())
            .all(|(a, b)| (a.0[0] > 0) == (b.0[0] > 0))
    );
    for x in 150u32..950 {
        for y in 150u32..210 {
            let dx = x.abs_diff(x.clamp(210u32, 889));
            assert_eq!(
                expanded.get_pixel(x, y).0[0] > 0,
                dx * dx + (210 - y).pow(2) <= 60 * 60
            );
        }
    }
}

// =============================================================================
// Rectangle selection
// =============================================================================

#[test]
fn rect_selection_replace() {
    let mut state = CanvasState::new(100, 100);
    let shape = SelectionShape::Rectangle {
        min_x: 10,
        min_y: 10,
        max_x: 50,
        max_y: 50,
    };
    state.apply_selection_shape(&shape, SelectionMode::Replace);

    assert!(state.has_selection());
    let mask = state.selection_mask.as_ref().unwrap();
    assert_eq!(mask.get_pixel(20, 20).0[0], 255, "inside rect");
    assert_eq!(mask.get_pixel(0, 0).0[0], 0, "outside rect");
    assert_eq!(mask.get_pixel(10, 10).0[0], 255, "on edge (inclusive)");
    assert_eq!(mask.get_pixel(51, 20).0[0], 0, "just outside right edge");
}

#[test]
fn ellipse_selection_replace() {
    let mut state = CanvasState::new(100, 100);
    let shape = SelectionShape::Ellipse {
        cx: 50.0,
        cy: 50.0,
        rx: 20.0,
        ry: 20.0,
    };
    state.apply_selection_shape(&shape, SelectionMode::Replace);

    assert!(state.has_selection());
    let mask = state.selection_mask.as_ref().unwrap();
    assert_eq!(mask.get_pixel(50, 50).0[0], 255, "center of ellipse");
    assert_eq!(mask.get_pixel(0, 0).0[0], 0, "far corner");
}

// =============================================================================
// Boolean operations
// =============================================================================

#[test]
fn selection_add() {
    let mut state = CanvasState::new(100, 100);

    // First rectangle
    let r1 = SelectionShape::Rectangle {
        min_x: 0,
        min_y: 0,
        max_x: 30,
        max_y: 30,
    };
    state.apply_selection_shape(&r1, SelectionMode::Replace);

    // Add second rectangle
    let r2 = SelectionShape::Rectangle {
        min_x: 70,
        min_y: 70,
        max_x: 99,
        max_y: 99,
    };
    state.apply_selection_shape(&r2, SelectionMode::Add);

    let mask = state.selection_mask.as_ref().unwrap();
    assert_eq!(mask.get_pixel(15, 15).0[0], 255, "first rect selected");
    assert_eq!(mask.get_pixel(85, 85).0[0], 255, "second rect selected");
    assert_eq!(mask.get_pixel(50, 50).0[0], 0, "gap between rects");
}

#[test]
fn selection_subtract() {
    let mut state = CanvasState::new(100, 100);

    // Full rect
    let full = SelectionShape::Rectangle {
        min_x: 0,
        min_y: 0,
        max_x: 99,
        max_y: 99,
    };
    state.apply_selection_shape(&full, SelectionMode::Replace);

    // Subtract center
    let center = SelectionShape::Rectangle {
        min_x: 30,
        min_y: 30,
        max_x: 70,
        max_y: 70,
    };
    state.apply_selection_shape(&center, SelectionMode::Subtract);

    let mask = state.selection_mask.as_ref().unwrap();
    assert_eq!(mask.get_pixel(50, 50).0[0], 0, "center subtracted");
    assert_eq!(mask.get_pixel(10, 10).0[0], 255, "corner remains");
}

#[test]
fn selection_intersect() {
    let mut state = CanvasState::new(100, 100);

    let r1 = SelectionShape::Rectangle {
        min_x: 0,
        min_y: 0,
        max_x: 60,
        max_y: 60,
    };
    state.apply_selection_shape(&r1, SelectionMode::Replace);

    let r2 = SelectionShape::Rectangle {
        min_x: 40,
        min_y: 40,
        max_x: 99,
        max_y: 99,
    };
    state.apply_selection_shape(&r2, SelectionMode::Intersect);

    let mask = state.selection_mask.as_ref().unwrap();
    assert_eq!(mask.get_pixel(50, 50).0[0], 255, "overlap region");
    assert_eq!(mask.get_pixel(10, 10).0[0], 0, "only in first rect");
    assert_eq!(mask.get_pixel(90, 90).0[0], 0, "only in second rect");
}

// =============================================================================
// Clear selection
// =============================================================================

#[test]
fn clear_selection_removes_mask() {
    let mut state = CanvasState::new(100, 100);
    let shape = SelectionShape::Rectangle {
        min_x: 10,
        min_y: 10,
        max_x: 50,
        max_y: 50,
    };
    state.apply_selection_shape(&shape, SelectionMode::Replace);
    assert!(state.has_selection());

    state.clear_selection();
    assert!(!state.has_selection());
}

// =============================================================================
// Translate selection
// =============================================================================

#[test]
fn translate_selection_moves_mask() {
    let mut state = CanvasState::new(100, 100);
    let shape = SelectionShape::Rectangle {
        min_x: 10,
        min_y: 10,
        max_x: 30,
        max_y: 30,
    };
    state.apply_selection_shape(&shape, SelectionMode::Replace);

    state.translate_selection(20, 20);

    let mask = state.selection_mask.as_ref().unwrap();
    // Original position should be clear, new position selected
    assert_eq!(mask.get_pixel(15, 15).0[0], 0, "old position cleared");
    assert_eq!(mask.get_pixel(35, 35).0[0], 255, "new position selected");
}

#[test]
fn translate_selection_clips_at_edge() {
    let mut state = CanvasState::new(100, 100);
    let shape = SelectionShape::Rectangle {
        min_x: 80,
        min_y: 80,
        max_x: 99,
        max_y: 99,
    };
    state.apply_selection_shape(&shape, SelectionMode::Replace);

    // Move right/down so part goes off-canvas
    state.translate_selection(10, 10);

    let mask = state.selection_mask.as_ref().unwrap();
    assert_eq!(mask.get_pixel(95, 95).0[0], 255, "in-bounds portion");
    // Original position should be clear
    assert_eq!(mask.get_pixel(85, 85).0[0], 0, "old position");
}

// =============================================================================
// Fill & delete selected pixels
// =============================================================================

#[test]
fn fill_selected_pixels_fills_area() {
    let mut state = CanvasState::new(100, 100);
    let shape = SelectionShape::Rectangle {
        min_x: 10,
        min_y: 10,
        max_x: 50,
        max_y: 50,
    };
    state.apply_selection_shape(&shape, SelectionMode::Replace);

    // Background is white — fill selection with red
    state.fill_selected_pixels(Rgba([255, 0, 0, 255]));

    let px_inside = *state.layers[0].pixels.get_pixel(20, 20);
    let px_outside = *state.layers[0].pixels.get_pixel(0, 0);
    assert_eq!(px_inside, Rgba([255, 0, 0, 255]), "filled area is red");
    assert_eq!(px_outside, Rgba([255, 255, 255, 255]), "outside unchanged");
}

#[test]
fn delete_selected_pixels_makes_transparent() {
    let mut state = CanvasState::new(100, 100);
    let shape = SelectionShape::Rectangle {
        min_x: 10,
        min_y: 10,
        max_x: 50,
        max_y: 50,
    };
    state.apply_selection_shape(&shape, SelectionMode::Replace);

    state.delete_selected_pixels();

    let px_inside = *state.layers[0].pixels.get_pixel(20, 20);
    let px_outside = *state.layers[0].pixels.get_pixel(0, 0);
    assert_eq!(px_inside[3], 0, "deleted area is transparent");
    assert_eq!(px_outside[3], 255, "outside unchanged");
}

// =============================================================================
// SelectionShape::contains
// =============================================================================

#[test]
fn selection_shape_contains_rect() {
    let shape = SelectionShape::Rectangle {
        min_x: 10,
        min_y: 20,
        max_x: 30,
        max_y: 40,
    };
    assert_eq!(shape.contains(15, 25), 255);
    assert_eq!(shape.contains(0, 0), 0);
    assert_eq!(shape.contains(10, 20), 255, "edges inclusive");
    assert_eq!(shape.contains(30, 40), 255, "max edges inclusive");
    assert_eq!(shape.contains(31, 25), 0, "past max_x");
}

#[test]
fn selection_shape_contains_ellipse() {
    let shape = SelectionShape::Ellipse {
        cx: 50.0,
        cy: 50.0,
        rx: 10.0,
        ry: 10.0,
    };
    assert_eq!(shape.contains(50, 50), 255, "center");
    assert_eq!(shape.contains(0, 0), 0, "far away");
}
