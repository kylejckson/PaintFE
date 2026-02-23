// ============================================================================
// IMAGE FILTERS — Gaussian blur, desaturation, etc.
// ============================================================================

use image::{RgbaImage, imageops};
use rayon::prelude::*;
use crate::canvas::{TiledImage, CanvasState};

/// Apply a Gaussian blur to the active layer.
/// `sigma` controls the blur radius / strength.
/// If a selection mask exists, only selected pixels are blurred.
///
/// When `gpu` is `Some`, the blur is executed on the GPU (compute shader)
/// for dramatically faster processing on large images.
pub fn gaussian_blur_layer(state: &mut CanvasState, layer_idx: usize, sigma: f32) {
    if layer_idx >= state.layers.len() { return; }
    let layer = &mut state.layers[layer_idx];
    let flat = layer.pixels.to_rgba_image();
    let result = blur_with_selection(&flat, sigma, state.selection_mask.as_ref());
    let layer = &mut state.layers[layer_idx];
    layer.pixels = TiledImage::from_rgba_image(&result);
    state.mark_dirty(None);
}

/// GPU-accelerated Gaussian blur.  Falls through to CPU if GPU is unavailable.
pub fn gaussian_blur_layer_gpu(
    state: &mut CanvasState,
    layer_idx: usize,
    sigma: f32,
    gpu: &crate::gpu::GpuRenderer,
) {
    if layer_idx >= state.layers.len() { return; }

    let layer = &mut state.layers[layer_idx];
    let flat = layer.pixels.to_rgba_image();
    let (w, h) = (flat.width(), flat.height());

    // GPU blur (no selection support yet — use CPU fallback for selections).
    if state.selection_mask.is_some() {
        let result = blur_with_selection(&flat, sigma, state.selection_mask.as_ref());
        let layer = &mut state.layers[layer_idx];
        layer.pixels = TiledImage::from_rgba_image(&result);
    } else {
        let result_data = gpu.blur_rgba(flat.as_raw(), w, h, sigma);
        let result = RgbaImage::from_raw(w, h, result_data).unwrap();
        let layer = &mut state.layers[layer_idx];
        layer.pixels = TiledImage::from_rgba_image(&result);
    }
    state.mark_dirty(None);
}

/// Fast blur for live preview: takes a pre-flattened image to avoid
/// re-flattening every frame.  Writes the result directly to the layer.
pub fn gaussian_blur_layer_from_flat(
    state: &mut CanvasState,
    layer_idx: usize,
    sigma: f32,
    original_flat: &RgbaImage,
) {
    if layer_idx >= state.layers.len() { return; }
    let result = blur_with_selection(original_flat, sigma, state.selection_mask.as_ref());
    let layer = &mut state.layers[layer_idx];
    layer.pixels = TiledImage::from_rgba_image(&result);
    state.mark_dirty(None);
}

/// GPU-accelerated live preview blur.
pub fn gaussian_blur_layer_from_flat_gpu(
    state: &mut CanvasState,
    layer_idx: usize,
    sigma: f32,
    original_flat: &RgbaImage,
    gpu: &crate::gpu::GpuRenderer,
) {
    if layer_idx >= state.layers.len() { return; }

    if state.selection_mask.is_some() {
        // Fall back to CPU for selection-masked blurs.
        let result = blur_with_selection(original_flat, sigma, state.selection_mask.as_ref());
        let layer = &mut state.layers[layer_idx];
        layer.pixels = TiledImage::from_rgba_image(&result);
    } else {
        let (w, h) = (original_flat.width(), original_flat.height());
        let result_data = gpu.blur_rgba(original_flat.as_raw(), w, h, sigma);
        let result = RgbaImage::from_raw(w, h, result_data).unwrap();
        let layer = &mut state.layers[layer_idx];
        layer.pixels = TiledImage::from_rgba_image(&result);
    }
    state.mark_dirty(None);
}

/// Public alias for blur_with_selection for async pipeline use.
pub fn blur_with_selection_pub(
    flat: &RgbaImage,
    sigma: f32,
    mask: Option<&image::GrayImage>,
) -> RgbaImage {
    blur_with_selection(flat, sigma, mask)
}

/// Core blur logic: when a selection mask exists, only blur the bounding-box
/// region of the selection, then composite back.  This is dramatically faster
/// for small selections on large canvases.
fn blur_with_selection(
    flat: &RgbaImage,
    sigma: f32,
    mask: Option<&image::GrayImage>,
) -> RgbaImage {
    if let Some(mask) = mask {
        let (mw, mh) = (mask.width(), mask.height());
        let mask_raw = mask.as_raw();
        // Find bounding box of selected pixels.
        let mut min_x = mw;
        let mut min_y = mh;
        let mut max_x = 0u32;
        let mut max_y = 0u32;
        for y in 0..mh {
            let row_off = y as usize * mw as usize;
            for x in 0..mw {
                if mask_raw[row_off + x as usize] > 0 {
                    min_x = min_x.min(x);
                    min_y = min_y.min(y);
                    max_x = max_x.max(x);
                    max_y = max_y.max(y);
                }
            }
        }
        if min_x > max_x || min_y > max_y {
            return flat.clone(); // Nothing selected
        }

        // Expand bbox by ceil(3*sigma) so the blur kernel has room to read
        // from surrounding pixels (avoids dark edge artifacts).
        let pad = (sigma * 3.0).ceil() as u32;
        let crop_x = min_x.saturating_sub(pad);
        let crop_y = min_y.saturating_sub(pad);
        let crop_x2 = (max_x + 1 + pad).min(flat.width());
        let crop_y2 = (max_y + 1 + pad).min(flat.height());
        let crop_w = crop_x2 - crop_x;
        let crop_h = crop_y2 - crop_y;

        // Extract sub-image and blur only that region.
        let sub = imageops::crop_imm(flat, crop_x, crop_y, crop_w, crop_h).to_image();
        let blurred_sub = parallel_gaussian_blur(&sub, sigma);

        // Composite: start from original, replace only selected pixels.
        let mut out = flat.clone();
        let out_stride = flat.width() as usize * 4;
        let out_raw = out.as_mut();
        let blur_raw = blurred_sub.as_raw();
        let blur_stride = crop_w as usize * 4;

        for y in min_y..=max_y {
            let mask_row = y as usize * mw as usize;
            let out_row = y as usize * out_stride;
            for x in min_x..=max_x {
                if x < mw && y < mh && mask_raw[mask_row + x as usize] > 0 {
                    let local_x = x - crop_x;
                    let local_y = y - crop_y;
                    if local_x < crop_w && local_y < crop_h {
                        let src_off = local_y as usize * blur_stride + local_x as usize * 4;
                        let dst_off = out_row + x as usize * 4;
                        out_raw[dst_off..dst_off + 4].copy_from_slice(&blur_raw[src_off..src_off + 4]);
                    }
                }
            }
        }
        out
    } else {
        // No selection — blur entire image.
        parallel_gaussian_blur(flat, sigma)
    }
}

// ---------------------------------------------------------------------------
//  Parallel separable Gaussian blur (rayon)
// ---------------------------------------------------------------------------

/// Build a 1-D Gaussian kernel truncated at ceil(3*sigma).
fn build_gaussian_kernel(sigma: f32) -> Vec<f32> {
    let radius = (sigma * 3.0).ceil() as usize;
    if radius == 0 {
        return vec![1.0];
    }
    let len = radius * 2 + 1;
    let mut kernel = vec![0.0f32; len];
    let s2 = 2.0 * sigma * sigma;
    let mut sum = 0.0f32;
    for i in 0..len {
        let x = i as f32 - radius as f32;
        let v = (-x * x / s2).exp();
        kernel[i] = v;
        sum += v;
    }
    let inv = 1.0 / sum;
    for v in &mut kernel { *v *= inv; }
    kernel
}

/// Rayon-parallelized separable Gaussian blur operating on raw f32 buffers.
/// Public alias for use by other effect modules.
pub fn parallel_gaussian_blur_pub(src: &RgbaImage, sigma: f32) -> RgbaImage {
    parallel_gaussian_blur(src, sigma)
}

fn parallel_gaussian_blur(src: &RgbaImage, sigma: f32) -> RgbaImage {
    let w = src.width() as usize;
    let h = src.height() as usize;
    if w == 0 || h == 0 { return src.clone(); }

    let kernel = build_gaussian_kernel(sigma);
    let radius = kernel.len() / 2;
    let src_raw = src.as_raw();

    // Convert to f32 buffer (4 channels interleaved).
    let pixel_count = w * h * 4;
    let mut buf_in: Vec<f32> = Vec::with_capacity(pixel_count);
    for &b in src_raw.iter() {
        buf_in.push(b as f32);
    }

    // --- Horizontal pass (parallel by row) ---
    let mut buf_h = vec![0.0f32; pixel_count];
    buf_h.par_chunks_mut(w * 4).enumerate().for_each(|(y, row_out)| {
        let row_in_start = y * w * 4;
        for x in 0..w {
            let mut r = 0.0f32;
            let mut g = 0.0f32;
            let mut b = 0.0f32;
            let mut a = 0.0f32;
            for (ki, &kv) in kernel.iter().enumerate() {
                let sx = (x as isize + ki as isize - radius as isize)
                    .max(0)
                    .min(w as isize - 1) as usize;
                let idx = row_in_start + sx * 4;
                r += buf_in[idx]     * kv;
                g += buf_in[idx + 1] * kv;
                b += buf_in[idx + 2] * kv;
                a += buf_in[idx + 3] * kv;
            }
            let out_idx = x * 4;
            row_out[out_idx]     = r;
            row_out[out_idx + 1] = g;
            row_out[out_idx + 2] = b;
            row_out[out_idx + 3] = a;
        }
    });

    // --- Vertical pass (parallel by row) ---
    let mut buf_v = vec![0.0f32; pixel_count];
    buf_v.par_chunks_mut(w * 4).enumerate().for_each(|(y, row_out)| {
        for x in 0..w {
            let mut r = 0.0f32;
            let mut g = 0.0f32;
            let mut b = 0.0f32;
            let mut a = 0.0f32;
            for (ki, &kv) in kernel.iter().enumerate() {
                let sy = (y as isize + ki as isize - radius as isize)
                    .max(0)
                    .min(h as isize - 1) as usize;
                let idx = sy * w * 4 + x * 4;
                r += buf_h[idx]     * kv;
                g += buf_h[idx + 1] * kv;
                b += buf_h[idx + 2] * kv;
                a += buf_h[idx + 3] * kv;
            }
            let out_idx = x * 4;
            row_out[out_idx]     = r;
            row_out[out_idx + 1] = g;
            row_out[out_idx + 2] = b;
            row_out[out_idx + 3] = a;
        }
    });

    // Convert back to u8.
    let dst_raw: Vec<u8> = buf_v.iter().map(|&v| v.round().clamp(0.0, 255.0) as u8).collect();
    RgbaImage::from_raw(w as u32, h as u32, dst_raw).unwrap()
}

/// Convert the active layer to greyscale (luminance-based desaturation).
/// Uses the BT.709 luminance weights: 0.2126 R + 0.7152 G + 0.0722 B.
/// If a selection mask exists, only selected pixels are desaturated.
pub fn desaturate_layer(state: &mut CanvasState, layer_idx: usize) {
    if layer_idx >= state.layers.len() { return; }
    let layer = &mut state.layers[layer_idx];
    let w = layer.pixels.width() as usize;
    let h = layer.pixels.height() as usize;

    let flat = layer.pixels.to_rgba_image();
    let src_raw = flat.as_raw();
    let mut dst_raw = vec![0u8; w * h * 4];
    let stride = w * 4;

    let mask_raw = state.selection_mask.as_ref().map(|m| m.as_raw().as_slice());
    let mask_w = state.selection_mask.as_ref().map_or(0, |m| m.width() as usize);
    let mask_h = state.selection_mask.as_ref().map_or(0, |m| m.height() as usize);

    // Parallel by row.
    dst_raw.par_chunks_mut(stride).enumerate().for_each(|(y, row_out)| {
        let row_in = &src_raw[y * stride..(y + 1) * stride];
        for x in 0..w {
            let pi = x * 4;
            // Check mask
            if let Some(mr) = mask_raw {
                if x < mask_w && y < mask_h {
                    if mr[y * mask_w + x] == 0 {
                        row_out[pi..pi + 4].copy_from_slice(&row_in[pi..pi + 4]);
                        continue;
                    }
                }
            }
            let r = row_in[pi] as f32;
            let g = row_in[pi + 1] as f32;
            let b = row_in[pi + 2] as f32;
            let lum = (0.2126 * r + 0.7152 * g + 0.0722 * b).round().clamp(0.0, 255.0) as u8;
            row_out[pi]     = lum;
            row_out[pi + 1] = lum;
            row_out[pi + 2] = lum;
            row_out[pi + 3] = row_in[pi + 3];
        }
    });

    let out = RgbaImage::from_raw(w as u32, h as u32, dst_raw).unwrap();
    let layer = &mut state.layers[layer_idx];
    layer.pixels = TiledImage::from_rgba_image(&out);
    state.mark_dirty(None);
}
