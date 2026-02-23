// ============================================================================
// CANVAS-LEVEL OPERATIONS — add / delete / duplicate layers
// ============================================================================

use image::Rgba;
use crate::canvas::{CanvasState, Layer};
use crate::components::history::{HistoryManager, LayerOpCommand, LayerOperation};

/// Use the top layer's luminance (brightness) as an alpha mask for the layer below,
/// then remove the top layer.
///
/// For each pixel the effective mask value is `lerp(255, luminance, alpha/255)`:
///   - Transparent mask pixel  → treated as white → bottom alpha unchanged
///   - Opaque white mask pixel → bottom alpha unchanged
///   - Opaque black mask pixel → bottom alpha set to 0 (fully transparent)
///   - Semi-transparent / grey → proportional blend toward the painted luminance
///
/// This means only painted (opaque) dark areas erase; unpainted (transparent)
/// areas leave the layer below fully intact.
///
/// This function does NOT push undo history itself; callers should wrap it in
/// `do_snapshot_op` to get a full-canvas undo snapshot.
pub fn merge_down_as_mask(state: &mut CanvasState, layer_idx: usize) {
    if layer_idx == 0 || layer_idx >= state.layers.len() {
        return;
    }

    let width = state.width;
    let height = state.height;

    // Collect the effective mask value for each pixel.
    //
    // Transparent pixels on the mask layer are treated as WHITE (no erase),
    // because the user only painted where they wanted to erase — unpainted
    // (transparent) areas should leave the layer below untouched.
    //
    // Formula: lerp(255, luminance, alpha/255)
    //   alpha=0   (transparent) → 255 (white, full preservation)
    //   alpha=255 (opaque)      → luminance of the painted color
    //   in between              → proportional blend toward the painted value
    let mask_luma: Vec<u8> = {
        let mask_layer = &state.layers[layer_idx];
        (0..height).flat_map(|y| {
            (0..width).map(move |x| {
                let p = *mask_layer.pixels.get_pixel(x, y);
                let r = p[0] as f32;
                let g = p[1] as f32;
                let b = p[2] as f32;
                let a = p[3] as f32 / 255.0;
                // Rec.601 perceptual luminance of the painted colour
                let luma = 0.299 * r + 0.587 * g + 0.114 * b;
                // Transparent pixels → white (255); opaque pixels → their luminance
                (255.0 * (1.0 - a) + luma * a + 0.5) as u8
            })
        }).collect()
    };

    // Apply the luminance mask to the bottom layer's alpha channel.
    {
        let bottom = &mut state.layers[layer_idx - 1];
        for y in 0..height {
            for x in 0..width {
                let i = (y * width + x) as usize;
                let luma = mask_luma[i];
                let mut px = *bottom.pixels.get_pixel(x, y);
                px[3] = ((px[3] as u32 * luma as u32) / 255) as u8;
                bottom.pixels.put_pixel(x, y, px);
            }
        }
    }

    // Remove the mask layer and adjust the active layer index.
    state.layers.remove(layer_idx);
    if state.active_layer_index >= layer_idx && state.active_layer_index > 0 {
        state.active_layer_index -= 1;
    }

    state.mark_dirty(None);
}

/// Add a new transparent layer above the active layer.
pub fn add_layer(state: &mut CanvasState, history: &mut HistoryManager) {
    let idx = (state.active_layer_index + 1).min(state.layers.len());
    let name = format!("Layer {}", state.layers.len() + 1);
    let layer = Layer::new(name.clone(), state.width, state.height, Rgba([0, 0, 0, 0]));
    state.layers.insert(idx, layer);
    state.active_layer_index = idx;

    history.push(Box::new(LayerOpCommand::new(LayerOperation::Add {
        index: idx,
        name,
        width: state.width,
        height: state.height,
    })));

    state.mark_dirty(None);
}

/// Delete the active layer (must keep at least one layer).
pub fn delete_layer(state: &mut CanvasState, history: &mut HistoryManager) {
    if state.layers.len() <= 1 { return; }
    let idx = state.active_layer_index;
    let removed = state.layers.remove(idx);

    history.push(Box::new(LayerOpCommand::new(LayerOperation::Delete {
        index: idx,
        pixels: removed.pixels,
        name: removed.name,
        visible: removed.visible,
        opacity: removed.opacity,
    })));

    if state.active_layer_index >= state.layers.len() {
        state.active_layer_index = state.layers.len() - 1;
    }
    state.mark_dirty(None);
}

/// Duplicate the active layer.
pub fn duplicate_layer(state: &mut CanvasState, history: &mut HistoryManager) {
    let idx = state.active_layer_index;
    if idx >= state.layers.len() { return; }

    let src = &state.layers[idx];
    let mut dup = Layer::new(
        format!("{} Copy", src.name),
        src.pixels.width(),
        src.pixels.height(),
        Rgba([0, 0, 0, 0]),
    );
    dup.pixels = src.pixels.clone();
    dup.visible = src.visible;
    dup.opacity = src.opacity;
    dup.blend_mode = src.blend_mode;

    let new_idx = idx + 1;
    let dup_pixels = dup.pixels.clone();
    let dup_name = dup.name.clone();
    let dup_visible = dup.visible;
    let dup_opacity = dup.opacity;

    state.layers.insert(new_idx, dup);
    state.active_layer_index = new_idx;

    history.push(Box::new(LayerOpCommand::new(LayerOperation::Duplicate {
        source_index: idx,
        new_index: new_idx,
        pixels: dup_pixels,
        name: dup_name,
        visible: dup_visible,
        opacity: dup_opacity,
    })));

    state.mark_dirty(None);
}
