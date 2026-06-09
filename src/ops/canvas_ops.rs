// ============================================================================
// CANVAS-LEVEL OPERATIONS — add / delete / duplicate layers
// ============================================================================

use crate::canvas::{CanvasState, Layer, LayerContent};
use crate::components::history::{HistoryManager, LayerOpCommand, LayerOperation};
use image::Rgba;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageChannel {
    Red,
    Green,
    Blue,
    Alpha,
    Luminance,
}

impl ImageChannel {
    fn sample(self, p: Rgba<u8>) -> u8 {
        match self {
            ImageChannel::Red => p[0],
            ImageChannel::Green => p[1],
            ImageChannel::Blue => p[2],
            ImageChannel::Alpha => p[3],
            ImageChannel::Luminance => {
                (0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32).round() as u8
            }
        }
    }
}

pub fn extract_channel_to_layer(state: &mut CanvasState, layer_idx: usize, channel: ImageChannel) {
    if layer_idx >= state.layers.len() {
        return;
    }
    let mut out = Layer::new(
        format!("{:?} Channel", channel),
        state.width,
        state.height,
        Rgba([0, 0, 0, 0]),
    );
    for y in 0..state.height {
        for x in 0..state.width {
            let v = channel.sample(*state.layers[layer_idx].pixels.get_pixel(x, y));
            out.pixels.put_pixel(x, y, Rgba([v, v, v, 255]));
        }
    }
    let insert_idx = layer_idx + 1;
    state.layers.insert(insert_idx, out);
    state.active_layer_index = insert_idx;
    state.mark_dirty(None);
}

pub fn replace_channel_from_layer(
    state: &mut CanvasState,
    target_idx: usize,
    source_idx: usize,
    target_channel: ImageChannel,
    source_channel: ImageChannel,
) {
    if target_idx >= state.layers.len() || source_idx >= state.layers.len() {
        return;
    }
    let source = state.layers[source_idx].pixels.clone();
    let target = &mut state.layers[target_idx];
    for y in 0..state.height {
        for x in 0..state.width {
            let v = source_channel.sample(*source.get_pixel(x, y));
            let mut p = *target.pixels.get_pixel(x, y);
            match target_channel {
                ImageChannel::Red => p[0] = v,
                ImageChannel::Green => p[1] = v,
                ImageChannel::Blue => p[2] = v,
                ImageChannel::Alpha | ImageChannel::Luminance => p[3] = v,
            }
            target.pixels.put_pixel(x, y, p);
        }
    }
    target.gpu_generation = target.gpu_generation.wrapping_add(1);
    state.mark_dirty(None);
}

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

    // Auto-rasterize text layers before merge (pixels must be up-to-date)
    for idx in [layer_idx, layer_idx - 1] {
        if state.layers[idx].is_text_layer() {
            state.ensure_all_text_layers_rasterized();
            state.layers[idx].content = LayerContent::Raster;
        }
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
        (0..height)
            .flat_map(|y| {
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
            })
            .collect()
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

pub fn add_layer_mask_reveal_all(state: &mut CanvasState, layer_idx: usize) {
    let Some(layer) = state.layers.get_mut(layer_idx) else {
        return;
    };
    if layer.has_live_mask() {
        layer.mask_enabled = true;
        return;
    }

    layer.ensure_mask();
    let Some(mask) = layer.mask.as_mut() else {
        return;
    };
    for y in 0..mask.height() {
        for x in 0..mask.width() {
            mask.put_pixel(x, y, Rgba([0, 0, 0, 0]));
        }
    }
    layer.mask_enabled = true;
    state.mark_dirty(None);
}

pub fn add_layer_mask_from_selection(state: &mut CanvasState, layer_idx: usize) {
    let Some(layer) = state.layers.get_mut(layer_idx) else {
        return;
    };
    if layer.has_live_mask() {
        layer.mask_enabled = true;
        return;
    }

    layer.ensure_mask();
    let Some(mask) = layer.mask.as_mut() else {
        return;
    };
    if let Some(sel) = &state.selection_mask {
        let w = mask.width().min(sel.width());
        let h = mask.height().min(sel.height());
        for y in 0..h {
            for x in 0..w {
                // Selection=255 should reveal fully, so mask conceal is inverted.
                let reveal = sel.get_pixel(x, y)[0];
                let conceal = 255u8.saturating_sub(reveal);
                mask.put_pixel(x, y, Rgba([0, 0, 0, conceal]));
            }
        }
    } else {
        // No selection: default to reveal-all mask.
        for y in 0..mask.height() {
            for x in 0..mask.width() {
                mask.put_pixel(x, y, Rgba([0, 0, 0, 0]));
            }
        }
    }

    layer.mask_enabled = true;
    state.mark_dirty(None);
}

pub fn toggle_layer_mask(state: &mut CanvasState, layer_idx: usize) {
    let Some(layer) = state.layers.get_mut(layer_idx) else {
        return;
    };
    if layer.has_live_mask() {
        layer.mask_enabled = !layer.mask_enabled;
        state.mark_dirty(None);
    }
}

pub fn invert_layer_mask(state: &mut CanvasState, layer_idx: usize) {
    let Some(layer) = state.layers.get_mut(layer_idx) else {
        return;
    };
    let Some(mask) = layer.mask.as_mut() else {
        return;
    };

    for y in 0..mask.height() {
        for x in 0..mask.width() {
            let mut p = *mask.get_pixel(x, y);
            p[3] = 255 - p[3];
            mask.put_pixel(x, y, p);
        }
    }
    layer.mask_enabled = true;
    state.mark_dirty(None);
}

pub fn apply_layer_mask(state: &mut CanvasState, layer_idx: usize) {
    let is_active = state.active_layer_index == layer_idx;
    let Some(layer) = state.layers.get_mut(layer_idx) else {
        return;
    };
    let Some(mask) = layer.mask.clone() else {
        return;
    };

    let w = layer.pixels.width().min(mask.width());
    let h = layer.pixels.height().min(mask.height());
    for y in 0..h {
        for x in 0..w {
            let conceal = mask.get_pixel(x, y)[3];
            if conceal == 0 {
                continue;
            }
            let mut px = *layer.pixels.get_pixel(x, y);
            px[3] = ((px[3] as u32 * (255 - conceal as u32)) / 255) as u8;
            layer.pixels.put_pixel(x, y, px);
        }
    }
    layer.mask = None;
    layer.mask_enabled = true;
    if is_active {
        state.edit_layer_mask = false;
    }
    state.mark_dirty(None);
}

pub fn delete_layer_mask(state: &mut CanvasState, layer_idx: usize) {
    let is_active = state.active_layer_index == layer_idx;
    let Some(layer) = state.layers.get_mut(layer_idx) else {
        return;
    };
    if layer.mask.take().is_some() {
        layer.mask_enabled = true;
        if is_active {
            state.edit_layer_mask = false;
        }
        state.mark_dirty(None);
    }
}

/// Add a new transparent layer above the active layer.
pub fn add_layer(state: &mut CanvasState, history: &mut HistoryManager) {
    let idx = (state.active_layer_index + 1).min(state.layers.len());
    let name = format!("Layer {}", state.layers.len() + 1);
    let mut layer = Layer::new(name.clone(), state.width, state.height, Rgba([0, 0, 0, 0]));
    layer.folder_id = state.layers[state.active_layer_index].folder_id;
    state.layers.insert(idx, layer);
    state.active_layer_index = idx;

    history.push(Box::new(LayerOpCommand::new(LayerOperation::Add {
        index: idx,
        name,
        width: state.width,
        height: state.height,
        folder_id: state.layers[idx].folder_id,
    })));

    state.mark_dirty(None);
}

/// Add a new editable text layer above the active layer.
pub fn add_text_layer(state: &mut CanvasState, history: &mut HistoryManager) {
    let idx = (state.active_layer_index + 1).min(state.layers.len());
    let name = format!("Text Layer {}", state.layers.len() + 1);
    let mut layer = Layer::new_text(name.clone(), state.width, state.height);
    layer.folder_id = state.layers[state.active_layer_index].folder_id;
    state.layers.insert(idx, layer);
    state.active_layer_index = idx;

    history.push(Box::new(LayerOpCommand::new(LayerOperation::Add {
        index: idx,
        name,
        width: state.width,
        height: state.height,
        folder_id: state.layers[idx].folder_id,
    })));

    state.mark_dirty(None);
}

/// Delete the active layer (must keep at least one layer).
pub fn delete_layer(state: &mut CanvasState, history: &mut HistoryManager) {
    if state.layers.len() <= 1 {
        return;
    }
    let idx = state.active_layer_index;
    let removed = state.layers.remove(idx);

    history.push(Box::new(LayerOpCommand::new(LayerOperation::Delete {
        index: idx,
        pixels: removed.pixels,
        mask: removed.mask,
        mask_enabled: removed.mask_enabled,
        name: removed.name,
        visible: removed.visible,
        folder_id: removed.folder_id,
        opacity: removed.opacity,
        content: removed.content,
        pixel_format: removed.pixel_format,
        hdr_metadata: removed.hdr_metadata,
        source_metadata: removed.source_metadata,
        deep_pixels: removed.deep_pixels,
    })));

    // Clear active text layer if it was the deleted layer
    if state.text_editing_layer == Some(idx) {
        state.text_editing_layer = None;
    } else if let Some(text_idx) = state.text_editing_layer {
        // Adjust text layer index if layer removed before it
        if idx < text_idx {
            state.text_editing_layer = Some(text_idx - 1);
        }
    }

    if state.active_layer_index >= state.layers.len() {
        state.active_layer_index = state.layers.len() - 1;
    }
    state.mark_dirty(None);
}

/// Duplicate the active layer.
pub fn duplicate_layer(state: &mut CanvasState, history: &mut HistoryManager) {
    let idx = state.active_layer_index;
    if idx >= state.layers.len() {
        return;
    }

    let src = &state.layers[idx];
    let mut dup = Layer::new(
        format!("{} Copy", src.name),
        src.pixels.width(),
        src.pixels.height(),
        Rgba([0, 0, 0, 0]),
    );
    dup.pixels = src.pixels.clone();
    dup.visible = src.visible;
    dup.folder_id = src.folder_id;
    dup.opacity = src.opacity;
    dup.blend_mode = src.blend_mode;
    dup.content = src.content.clone();
    dup.mask = src.mask.clone();
    dup.mask_enabled = src.mask_enabled;
    dup.pixel_format = src.pixel_format;
    dup.hdr_metadata = src.hdr_metadata.clone();
    dup.source_metadata = src.source_metadata.clone();
    dup.deep_pixels = src.deep_pixels.clone();

    let new_idx = idx + 1;
    let dup_pixels = dup.pixels.clone();
    let dup_mask = dup.mask.clone();
    let dup_mask_enabled = dup.mask_enabled;
    let dup_name = dup.name.clone();
    let dup_visible = dup.visible;
    let dup_folder_id = dup.folder_id;
    let dup_opacity = dup.opacity;
    let dup_content = dup.content.clone();
    let dup_pixel_format = dup.pixel_format;
    let dup_hdr_metadata = dup.hdr_metadata.clone();
    let dup_source_metadata = dup.source_metadata.clone();
    let dup_deep_pixels = dup.deep_pixels.clone();

    state.layers.insert(new_idx, dup);
    state.active_layer_index = new_idx;

    history.push(Box::new(LayerOpCommand::new(LayerOperation::Duplicate {
        source_index: idx,
        new_index: new_idx,
        pixels: dup_pixels,
        mask: dup_mask,
        mask_enabled: dup_mask_enabled,
        name: dup_name,
        visible: dup_visible,
        folder_id: dup_folder_id,
        opacity: dup_opacity,
        content: dup_content,
        pixel_format: dup_pixel_format,
        hdr_metadata: dup_hdr_metadata,
        source_metadata: dup_source_metadata,
        deep_pixels: dup_deep_pixels,
    })));

    state.mark_dirty(None);
}
