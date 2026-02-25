// ============================================================================
// EFFECTS ENGINE â€” GPU-ready, rayon-parallelized image effects
// ============================================================================
//
// All effects are selection-aware: if a selection mask exists, only selected
// pixels are modified. Every effect has a `_from_flat` variant for live preview.
//
// Effects are grouped into categories:
//   - Blur: Bokeh, Motion, Box
//   - Distort: Crystallize, Dents, Pixelate, Bulge, Twist
//   - Noise: Add Noise, Reduce Noise, Median
//   - Stylize: Glow, Sharpen, Vignette, Halftone
//   - Render: Grid, Drop Shadow, Outline
//   - Glitch: Pixel Drag, RGB Displace
//   - Artistic: Ink, Oil Painting, Color Filter
// ============================================================================

use crate::canvas::{CanvasState, TiledImage};
use image::{GrayImage, Rgba, RgbaImage};
use rayon::prelude::*;

// ============================================================================
// SHARED HELPERS
// ============================================================================

/// Apply a spatial effect (reads neighbours) with selection masking.
/// `processor` receives (source image, x, y) and returns the output pixel.
fn apply_spatial_effect<F>(flat: &RgbaImage, mask: Option<&GrayImage>, processor: F) -> RgbaImage
where
    F: Fn(&RgbaImage, u32, u32) -> Rgba<u8> + Sync,
{
    let w = flat.width() as usize;
    let h = flat.height() as usize;
    if w == 0 || h == 0 {
        return flat.clone();
    }

    let src_raw = flat.as_raw();
    let mut dst_raw = vec![0u8; w * h * 4];
    let stride = w * 4;

    let mask_raw = mask.map(|m| m.as_raw().as_slice());
    let mask_w = mask.map_or(0, |m| m.width() as usize);
    let mask_h = mask.map_or(0, |m| m.height() as usize);

    dst_raw
        .par_chunks_mut(stride)
        .enumerate()
        .for_each(|(y, row_out)| {
            let row_in = &src_raw[y * stride..(y + 1) * stride];
            for x in 0..w {
                let pi = x * 4;
                if let Some(mr) = mask_raw
                    && x < mask_w
                    && y < mask_h
                    && mr[y * mask_w + x] == 0
                {
                    row_out[pi..pi + 4].copy_from_slice(&row_in[pi..pi + 4]);
                    continue;
                }
                let px = processor(flat, x as u32, y as u32);
                row_out[pi] = px[0];
                row_out[pi + 1] = px[1];
                row_out[pi + 2] = px[2];
                row_out[pi + 3] = px[3];
            }
        });

    RgbaImage::from_raw(w as u32, h as u32, dst_raw).unwrap()
}

/// Per-pixel transform with selection masking (like adjustments helper).
fn apply_per_pixel<F>(flat: &RgbaImage, mask: Option<&GrayImage>, transform: F) -> RgbaImage
where
    F: Fn(u32, u32, f32, f32, f32, f32) -> (f32, f32, f32, f32) + Sync,
{
    let w = flat.width() as usize;
    let h = flat.height() as usize;
    if w == 0 || h == 0 {
        return flat.clone();
    }

    let src_raw = flat.as_raw();
    let mut dst_raw = vec![0u8; w * h * 4];
    let stride = w * 4;
    let mask_raw = mask.map(|m| m.as_raw().as_slice());
    let mask_w = mask.map_or(0, |m| m.width() as usize);
    let mask_h = mask.map_or(0, |m| m.height() as usize);

    dst_raw
        .par_chunks_mut(stride)
        .enumerate()
        .for_each(|(y, row_out)| {
            let row_in = &src_raw[y * stride..(y + 1) * stride];
            for x in 0..w {
                let pi = x * 4;
                if let Some(mr) = mask_raw
                    && x < mask_w
                    && y < mask_h
                    && mr[y * mask_w + x] == 0
                {
                    row_out[pi..pi + 4].copy_from_slice(&row_in[pi..pi + 4]);
                    continue;
                }
                let r = row_in[pi] as f32;
                let g = row_in[pi + 1] as f32;
                let b = row_in[pi + 2] as f32;
                let a = row_in[pi + 3] as f32;
                let (nr, ng, nb, na) = transform(x as u32, y as u32, r, g, b, a);
                row_out[pi] = nr.round().clamp(0.0, 255.0) as u8;
                row_out[pi + 1] = ng.round().clamp(0.0, 255.0) as u8;
                row_out[pi + 2] = nb.round().clamp(0.0, 255.0) as u8;
                row_out[pi + 3] = na.round().clamp(0.0, 255.0) as u8;
            }
        });

    RgbaImage::from_raw(w as u32, h as u32, dst_raw).unwrap()
}

/// Write effect result back to a layer.
fn commit_to_layer(state: &mut CanvasState, layer_idx: usize, result: &RgbaImage) {
    if layer_idx >= state.layers.len() {
        return;
    }
    state.layers[layer_idx].pixels = TiledImage::from_rgba_image(result);
    state.mark_dirty(None);
}

/// Clamp-sample a pixel from an image (mirror-clamp at edges).
#[inline]
fn sample_clamped(img: &RgbaImage, x: i32, y: i32) -> [f32; 4] {
    let cx = x.clamp(0, img.width() as i32 - 1) as u32;
    let cy = y.clamp(0, img.height() as i32 - 1) as u32;
    let p = img.get_pixel(cx, cy);
    [p[0] as f32, p[1] as f32, p[2] as f32, p[3] as f32]
}

/// Bilinear-sample at fractional coordinates.
#[inline]
fn sample_bilinear(img: &RgbaImage, fx: f32, fy: f32) -> [f32; 4] {
    let _w = img.width() as i32;
    let _h = img.height() as i32;
    let x0 = fx.floor() as i32;
    let y0 = fy.floor() as i32;
    let x1 = x0 + 1;
    let y1 = y0 + 1;
    let dx = fx - x0 as f32;
    let dy = fy - y0 as f32;

    let p00 = sample_clamped(img, x0, y0);
    let p10 = sample_clamped(img, x1, y0);
    let p01 = sample_clamped(img, x0, y1);
    let p11 = sample_clamped(img, x1, y1);

    let mut out = [0.0f32; 4];
    for c in 0..4 {
        out[c] = p00[c] * (1.0 - dx) * (1.0 - dy)
            + p10[c] * dx * (1.0 - dy)
            + p01[c] * (1.0 - dx) * dy
            + p11[c] * dx * dy;
    }
    out
}

/// Simple hash for deterministic noise.
#[inline]
fn hash_u32(mut x: u32) -> u32 {
    x = x.wrapping_mul(0x9E3779B9);
    x ^= x >> 16;
    x = x.wrapping_mul(0x85EBCA6B);
    x ^= x >> 13;
    x = x.wrapping_mul(0xC2B2AE35);
    x ^= x >> 16;
    x
}

/// Hash to f32 in [0, 1).
#[inline]
fn hash_f32(x: u32, y: u32, seed: u32) -> f32 {
    let h = hash_u32(
        x.wrapping_mul(374761393)
            .wrapping_add(y.wrapping_mul(668265263))
            .wrapping_add(seed),
    );
    (h & 0x00FFFFFF) as f32 / 16777216.0
}

// ============================================================================
// BLUR EFFECTS
// ============================================================================

// --- Bokeh Blur (disc-shaped kernel) ---

pub fn bokeh_blur(state: &mut CanvasState, layer_idx: usize, radius: f32) {
    if layer_idx >= state.layers.len() {
        return;
    }
    let flat = state.layers[layer_idx].pixels.to_rgba_image();
    let result = bokeh_blur_core(&flat, radius, state.selection_mask.as_ref());
    commit_to_layer(state, layer_idx, &result);
}

pub fn bokeh_blur_from_flat(
    state: &mut CanvasState,
    layer_idx: usize,
    radius: f32,
    original_flat: &RgbaImage,
) {
    let result = bokeh_blur_core(original_flat, radius, state.selection_mask.as_ref());
    commit_to_layer(state, layer_idx, &result);
}

pub fn bokeh_blur_core(flat: &RgbaImage, radius: f32, mask: Option<&GrayImage>) -> RgbaImage {
    if radius < 0.5 {
        return flat.clone();
    }
    let w = flat.width() as usize;
    let h = flat.height() as usize;
    if w == 0 || h == 0 {
        return flat.clone();
    }

    // Build disc kernel (coordinates + weights).
    let r = radius.ceil() as i32;
    let r2 = radius * radius;
    let mut offsets: Vec<(i32, i32, f32)> = Vec::new();
    for dy in -r..=r {
        for dx in -r..=r {
            let d2 = (dx * dx + dy * dy) as f32;
            if d2 <= r2 {
                // Bokeh uses equal weight (disc average).
                offsets.push((dx, dy, 1.0));
            }
        }
    }
    let inv_count = 1.0 / offsets.len() as f32;

    let src_raw = flat.as_raw();
    let mut dst_raw = vec![0u8; w * h * 4];
    let stride = w * 4;
    let mask_raw = mask.map(|m| m.as_raw().as_slice());
    let mask_w = mask.map_or(0, |m| m.width() as usize);
    let mask_h = mask.map_or(0, |m| m.height() as usize);

    dst_raw
        .par_chunks_mut(stride)
        .enumerate()
        .for_each(|(y, row_out)| {
            for x in 0..w {
                let pi = x * 4;
                if let Some(mr) = mask_raw
                    && x < mask_w
                    && y < mask_h
                    && mr[y * mask_w + x] == 0
                {
                    let src_off = y * stride + pi;
                    row_out[pi..pi + 4].copy_from_slice(&src_raw[src_off..src_off + 4]);
                    continue;
                }
                let mut r_sum = 0.0f32;
                let mut g_sum = 0.0f32;
                let mut b_sum = 0.0f32;
                let mut a_sum = 0.0f32;
                for &(dx, dy, _wt) in &offsets {
                    let sx = (x as i32 + dx).clamp(0, w as i32 - 1) as usize;
                    let sy = (y as i32 + dy).clamp(0, h as i32 - 1) as usize;
                    let si = sy * stride + sx * 4;
                    r_sum += src_raw[si] as f32;
                    g_sum += src_raw[si + 1] as f32;
                    b_sum += src_raw[si + 2] as f32;
                    a_sum += src_raw[si + 3] as f32;
                }
                row_out[pi] = (r_sum * inv_count).round().clamp(0.0, 255.0) as u8;
                row_out[pi + 1] = (g_sum * inv_count).round().clamp(0.0, 255.0) as u8;
                row_out[pi + 2] = (b_sum * inv_count).round().clamp(0.0, 255.0) as u8;
                row_out[pi + 3] = (a_sum * inv_count).round().clamp(0.0, 255.0) as u8;
            }
        });

    RgbaImage::from_raw(w as u32, h as u32, dst_raw).unwrap()
}

// --- Motion Blur (directional) ---

pub fn motion_blur(state: &mut CanvasState, layer_idx: usize, angle_deg: f32, distance: f32) {
    if layer_idx >= state.layers.len() {
        return;
    }
    let flat = state.layers[layer_idx].pixels.to_rgba_image();
    let result = motion_blur_core(&flat, angle_deg, distance, state.selection_mask.as_ref());
    commit_to_layer(state, layer_idx, &result);
}

pub fn motion_blur_from_flat(
    state: &mut CanvasState,
    layer_idx: usize,
    angle_deg: f32,
    distance: f32,
    original_flat: &RgbaImage,
) {
    let result = motion_blur_core(
        original_flat,
        angle_deg,
        distance,
        state.selection_mask.as_ref(),
    );
    commit_to_layer(state, layer_idx, &result);
}

pub fn motion_blur_core(
    flat: &RgbaImage,
    angle_deg: f32,
    distance: f32,
    mask: Option<&GrayImage>,
) -> RgbaImage {
    if distance < 1.0 {
        return flat.clone();
    }
    let w = flat.width() as usize;
    let h = flat.height() as usize;
    if w == 0 || h == 0 {
        return flat.clone();
    }

    let angle = angle_deg.to_radians();
    let steps = distance.ceil() as i32;
    let dx = angle.cos();
    let dy = angle.sin();
    let inv_steps = 1.0 / (steps * 2 + 1) as f32;

    let src_raw = flat.as_raw();
    let mut dst_raw = vec![0u8; w * h * 4];
    let stride = w * 4;
    let mask_raw = mask.map(|m| m.as_raw().as_slice());
    let mask_w = mask.map_or(0, |m| m.width() as usize);
    let mask_h = mask.map_or(0, |m| m.height() as usize);

    dst_raw
        .par_chunks_mut(stride)
        .enumerate()
        .for_each(|(y, row_out)| {
            for x in 0..w {
                let pi = x * 4;
                if let Some(mr) = mask_raw
                    && x < mask_w
                    && y < mask_h
                    && mr[y * mask_w + x] == 0
                {
                    let src_off = y * stride + pi;
                    row_out[pi..pi + 4].copy_from_slice(&src_raw[src_off..src_off + 4]);
                    continue;
                }
                let mut r_sum = 0.0f32;
                let mut g_sum = 0.0f32;
                let mut b_sum = 0.0f32;
                let mut a_sum = 0.0f32;
                for i in -steps..=steps {
                    let sx = (x as f32 + i as f32 * dx).round() as i32;
                    let sy = (y as f32 + i as f32 * dy).round() as i32;
                    let sx = sx.clamp(0, w as i32 - 1) as usize;
                    let sy = sy.clamp(0, h as i32 - 1) as usize;
                    let si = sy * stride + sx * 4;
                    r_sum += src_raw[si] as f32;
                    g_sum += src_raw[si + 1] as f32;
                    b_sum += src_raw[si + 2] as f32;
                    a_sum += src_raw[si + 3] as f32;
                }
                row_out[pi] = (r_sum * inv_steps).round().clamp(0.0, 255.0) as u8;
                row_out[pi + 1] = (g_sum * inv_steps).round().clamp(0.0, 255.0) as u8;
                row_out[pi + 2] = (b_sum * inv_steps).round().clamp(0.0, 255.0) as u8;
                row_out[pi + 3] = (a_sum * inv_steps).round().clamp(0.0, 255.0) as u8;
            }
        });

    RgbaImage::from_raw(w as u32, h as u32, dst_raw).unwrap()
}

// --- Box Blur (square kernel, separable for speed) ---

pub fn box_blur(state: &mut CanvasState, layer_idx: usize, radius: f32) {
    if layer_idx >= state.layers.len() {
        return;
    }
    let flat = state.layers[layer_idx].pixels.to_rgba_image();
    let result = box_blur_core(&flat, radius, state.selection_mask.as_ref());
    commit_to_layer(state, layer_idx, &result);
}

pub fn box_blur_from_flat(
    state: &mut CanvasState,
    layer_idx: usize,
    radius: f32,
    original_flat: &RgbaImage,
) {
    let result = box_blur_core(original_flat, radius, state.selection_mask.as_ref());
    commit_to_layer(state, layer_idx, &result);
}

pub fn box_blur_core(flat: &RgbaImage, radius: f32, mask: Option<&GrayImage>) -> RgbaImage {
    if radius < 0.5 {
        return flat.clone();
    }
    let w = flat.width() as usize;
    let h = flat.height() as usize;
    if w == 0 || h == 0 {
        return flat.clone();
    }

    let r = radius.ceil() as usize;
    let kernel_size = r * 2 + 1;
    let inv_k = 1.0 / (kernel_size as f32);
    let src_raw = flat.as_raw();

    // Separable: horizontal pass
    let mut h_buf = vec![0.0f32; w * h * 4];
    h_buf
        .par_chunks_mut(w * 4)
        .enumerate()
        .for_each(|(y, row_out)| {
            for x in 0..w {
                let mut sums = [0.0f32; 4];
                for k in 0..kernel_size {
                    let sx = (x as i32 + k as i32 - r as i32).clamp(0, w as i32 - 1) as usize;
                    let si = y * w * 4 + sx * 4;
                    for c in 0..4 {
                        sums[c] += src_raw[si + c] as f32;
                    }
                }
                let oi = x * 4;
                for c in 0..4 {
                    row_out[oi + c] = sums[c] * inv_k;
                }
            }
        });

    // Vertical pass
    let mut v_buf = vec![0.0f32; w * h * 4];
    v_buf
        .par_chunks_mut(w * 4)
        .enumerate()
        .for_each(|(y, row_out)| {
            for x in 0..w {
                let mut sums = [0.0f32; 4];
                for k in 0..kernel_size {
                    let sy = (y as i32 + k as i32 - r as i32).clamp(0, h as i32 - 1) as usize;
                    let si = sy * w * 4 + x * 4;
                    for c in 0..4 {
                        sums[c] += h_buf[si + c];
                    }
                }
                let oi = x * 4;
                for c in 0..4 {
                    row_out[oi + c] = sums[c] * inv_k;
                }
            }
        });

    // Apply mask
    let mask_raw = mask.map(|m| m.as_raw().as_slice());
    let mask_w = mask.map_or(0, |m| m.width() as usize);
    let mask_h = mask.map_or(0, |m| m.height() as usize);
    let mut dst_raw = vec![0u8; w * h * 4];
    dst_raw
        .par_chunks_mut(w * 4)
        .enumerate()
        .for_each(|(y, row_out)| {
            for x in 0..w {
                let pi = x * 4;
                if let Some(mr) = mask_raw
                    && x < mask_w
                    && y < mask_h
                    && mr[y * mask_w + x] == 0
                {
                    let si = y * w * 4 + pi;
                    row_out[pi..pi + 4].copy_from_slice(&src_raw[si..si + 4]);
                    continue;
                }
                let vi = y * w * 4 + pi;
                for c in 0..4 {
                    row_out[pi + c] = v_buf[vi + c].round().clamp(0.0, 255.0) as u8;
                }
            }
        });

    RgbaImage::from_raw(w as u32, h as u32, dst_raw).unwrap()
}

// --- Zoom Blur (radial speed-zoom effect) ---

pub fn zoom_blur_core(
    flat: &RgbaImage,
    center_x: f32,        // 0.0–1.0 normalized horizontal position of the zoom origin
    center_y: f32,        // 0.0–1.0 normalized vertical position of the zoom origin
    strength: f32,        // 0.0–1.0: fraction of distance to sample back toward center
    samples: u32,         // quality: 8 (fast) / 16 (normal) / 32 (high)
    tint_color: [f32; 4], // RGBA 0–1 tint applied near the zoom origin (if tint_strength > 0)
    tint_strength: f32,   // 0.0 = no tint, 1.0 = full tint at dead-center
    mask: Option<&GrayImage>,
) -> RgbaImage {
    if strength < 0.001 {
        return flat.clone();
    }
    let w = flat.width() as usize;
    let h = flat.height() as usize;
    if w == 0 || h == 0 {
        return flat.clone();
    }

    let cx = center_x * w as f32;
    let cy = center_y * h as f32;
    let s = strength.clamp(0.0, 0.99);
    let n = samples.max(2) as usize;
    let inv_n = 1.0 / n as f32;

    // Max distance from center to any corner — used to normalise tint falloff.
    let max_dist = [
        (cx, cy),
        (w as f32 - cx, cy),
        (cx, h as f32 - cy),
        (w as f32 - cx, h as f32 - cy),
    ]
    .iter()
    .map(|(dx, dy)| (dx * dx + dy * dy).sqrt())
    .fold(0.0f32, f32::max)
    .max(1.0);

    let src_raw = flat.as_raw();
    let stride = w * 4;
    let mask_raw = mask.map(|m| m.as_raw().as_slice());
    let mask_w = mask.map_or(0, |m| m.width() as usize);
    let mask_h = mask.map_or(0, |m| m.height() as usize);

    let mut dst_raw = vec![0u8; w * h * 4];
    dst_raw
        .par_chunks_mut(stride)
        .enumerate()
        .for_each(|(y, row_out)| {
            for x in 0..w {
                let pi = x * 4;
                if let Some(mr) = mask_raw
                    && x < mask_w
                    && y < mask_h
                    && mr[y * mask_w + x] == 0
                {
                    let src_off = y * stride + pi;
                    row_out[pi..pi + 4].copy_from_slice(&src_raw[src_off..src_off + 4]);
                    continue;
                }
                let px = x as f32;
                let py = y as f32;
                let dx = px - cx;
                let dy = py - cy;

                // Sample from the pixel position back toward the zoom center.
                // i=0 → pixel position (t=1.0), i=n-1 → closest to center (t=1-s).
                let mut r_sum = 0.0f32;
                let mut g_sum = 0.0f32;
                let mut b_sum = 0.0f32;
                let mut a_sum = 0.0f32;
                for i in 0..n {
                    let t = 1.0 - s * (i as f32 / (n - 1) as f32);
                    let sx = (cx + dx * t).round() as i32;
                    let sy = (cy + dy * t).round() as i32;
                    let sx = sx.clamp(0, w as i32 - 1) as usize;
                    let sy = sy.clamp(0, h as i32 - 1) as usize;
                    let si = sy * stride + sx * 4;
                    r_sum += src_raw[si] as f32;
                    g_sum += src_raw[si + 1] as f32;
                    b_sum += src_raw[si + 2] as f32;
                    a_sum += src_raw[si + 3] as f32;
                }
                let mut r = r_sum * inv_n;
                let mut g = g_sum * inv_n;
                let mut b = b_sum * inv_n;
                let mut a = a_sum * inv_n;

                // Optional radial tint — strongest at the zoom origin, fading to zero at corners.
                if tint_strength > 0.001 {
                    let dist = (dx * dx + dy * dy).sqrt();
                    let t = (1.0 - dist / max_dist).max(0.0) * tint_strength;
                    r = r + (tint_color[0] * 255.0 - r) * t;
                    g = g + (tint_color[1] * 255.0 - g) * t;
                    b = b + (tint_color[2] * 255.0 - b) * t;
                    a = a + (tint_color[3] * 255.0 - a) * t;
                }

                row_out[pi] = r.round().clamp(0.0, 255.0) as u8;
                row_out[pi + 1] = g.round().clamp(0.0, 255.0) as u8;
                row_out[pi + 2] = b.round().clamp(0.0, 255.0) as u8;
                row_out[pi + 3] = a.round().clamp(0.0, 255.0) as u8;
            }
        });

    RgbaImage::from_raw(w as u32, h as u32, dst_raw).unwrap()
}

// ============================================================================
// DISTORTION EFFECTS
// ============================================================================

// --- Crystallize (Voronoi polygon effect) ---

pub fn crystallize(state: &mut CanvasState, layer_idx: usize, cell_size: f32, seed: u32) {
    if layer_idx >= state.layers.len() {
        return;
    }
    let flat = state.layers[layer_idx].pixels.to_rgba_image();
    let result = crystallize_core(&flat, cell_size, seed, state.selection_mask.as_ref());
    commit_to_layer(state, layer_idx, &result);
}

pub fn crystallize_from_flat(
    state: &mut CanvasState,
    layer_idx: usize,
    cell_size: f32,
    seed: u32,
    original_flat: &RgbaImage,
) {
    let result = crystallize_core(
        original_flat,
        cell_size,
        seed,
        state.selection_mask.as_ref(),
    );
    commit_to_layer(state, layer_idx, &result);
}

pub fn crystallize_core(
    flat: &RgbaImage,
    cell_size: f32,
    seed: u32,
    mask: Option<&GrayImage>,
) -> RgbaImage {
    let cs = cell_size.max(2.0);
    let w = flat.width();
    let h = flat.height();
    if w == 0 || h == 0 {
        return flat.clone();
    }

    let src_raw = flat.as_raw();
    let stride = w as usize * 4;
    let mut dst_raw = src_raw.clone();
    let mask_raw = mask.map(|m| m.as_raw().as_slice());
    let mask_w = mask.map_or(0, |m| m.width() as usize);
    let mask_h = mask.map_or(0, |m| m.height() as usize);

    // Grid dimensions for Voronoi cell lookup
    let cells_x = ((w as f32 / cs).ceil() as i32).max(1);
    let cells_y = ((h as f32 / cs).ceil() as i32).max(1);

    // Generate one random seed point per grid cell (jittered grid)
    let num_cells = (cells_x * cells_y) as usize;
    let mut seed_points: Vec<(f32, f32)> = Vec::with_capacity(num_cells);
    for cy in 0..cells_y {
        for cx in 0..cells_x {
            let base_x = cx as f32 * cs;
            let base_y = cy as f32 * cs;
            let jx = hash_f32(cx as u32, cy as u32, seed);
            let jy = hash_f32(cx as u32, cy as u32, seed.wrapping_add(77));
            seed_points.push((base_x + jx * cs, base_y + jy * cs));
        }
    }

    // Precompute average color for each Voronoi cell by assigning every pixel
    // to its nearest seed point, then averaging.
    let mut sums: Vec<[f64; 4]> = vec![[0.0; 4]; num_cells];
    let mut counts: Vec<u32> = vec![0; num_cells];

    for y in 0..h {
        for x in 0..w {
            // Find which grid cell this pixel is near and search 3Ã—3 neighbourhood
            let gcx = (x as f32 / cs) as i32;
            let gcy = (y as f32 / cs) as i32;
            let px = x as f32 + 0.5;
            let py = y as f32 + 0.5;

            let mut best_dist = f32::MAX;
            let mut best_idx = 0usize;

            for dy in -1..=1 {
                for dx in -1..=1 {
                    let nx = gcx + dx;
                    let ny = gcy + dy;
                    if nx < 0 || ny < 0 || nx >= cells_x || ny >= cells_y {
                        continue;
                    }
                    let idx = (ny * cells_x + nx) as usize;
                    let (sx, sy) = seed_points[idx];
                    let d = (px - sx) * (px - sx) + (py - sy) * (py - sy);
                    if d < best_dist {
                        best_dist = d;
                        best_idx = idx;
                    }
                }
            }

            let si = (y as usize * stride) + (x as usize * 4);
            sums[best_idx][0] += src_raw[si] as f64;
            sums[best_idx][1] += src_raw[si + 1] as f64;
            sums[best_idx][2] += src_raw[si + 2] as f64;
            sums[best_idx][3] += src_raw[si + 3] as f64;
            counts[best_idx] += 1;
        }
    }

    // Compute averages
    let mut averages: Vec<[u8; 4]> = vec![[0; 4]; num_cells];
    for i in 0..num_cells {
        if counts[i] > 0 {
            let inv = 1.0 / counts[i] as f64;
            averages[i] = [
                (sums[i][0] * inv).round().clamp(0.0, 255.0) as u8,
                (sums[i][1] * inv).round().clamp(0.0, 255.0) as u8,
                (sums[i][2] * inv).round().clamp(0.0, 255.0) as u8,
                (sums[i][3] * inv).round().clamp(0.0, 255.0) as u8,
            ];
        }
    }

    // Assign pixels to Voronoi cells (parallel by row)
    let seed_pts = &seed_points;
    let avgs = &averages;
    dst_raw
        .par_chunks_mut(stride)
        .enumerate()
        .for_each(|(y, row_out)| {
            for x in 0..w as usize {
                if let Some(mr) = mask_raw
                    && x < mask_w
                    && y < mask_h
                    && mr[y * mask_w + x] == 0
                {
                    continue;
                }

                let gcx = (x as f32 / cs) as i32;
                let gcy = (y as f32 / cs) as i32;
                let px = x as f32 + 0.5;
                let py = y as f32 + 0.5;

                let mut best_dist = f32::MAX;
                let mut best_idx = 0usize;

                for dy in -1..=1 {
                    for dx in -1..=1 {
                        let nx = gcx + dx;
                        let ny = gcy + dy;
                        if nx < 0 || ny < 0 || nx >= cells_x || ny >= cells_y {
                            continue;
                        }
                        let idx = (ny * cells_x + nx) as usize;
                        let (sx, sy) = seed_pts[idx];
                        let d = (px - sx) * (px - sx) + (py - sy) * (py - sy);
                        if d < best_dist {
                            best_dist = d;
                            best_idx = idx;
                        }
                    }
                }

                let pi = x * 4;
                row_out[pi] = avgs[best_idx][0];
                row_out[pi + 1] = avgs[best_idx][1];
                row_out[pi + 2] = avgs[best_idx][2];
                row_out[pi + 3] = avgs[best_idx][3];
            }
        });

    RgbaImage::from_raw(w, h, dst_raw).unwrap()
}

// --- Dents (turbulence-based distortion) ---

pub fn dents(
    state: &mut CanvasState,
    layer_idx: usize,
    scale: f32,
    amount: f32,
    seed: u32,
    octaves: u32,
    roughness: f32,
    pinch: bool,
    wrap: bool,
) {
    if layer_idx >= state.layers.len() {
        return;
    }
    let flat = state.layers[layer_idx].pixels.to_rgba_image();
    let result = dents_core(
        &flat,
        scale,
        amount,
        seed,
        octaves,
        roughness,
        pinch,
        wrap,
        state.selection_mask.as_ref(),
    );
    commit_to_layer(state, layer_idx, &result);
}

pub fn dents_from_flat(
    state: &mut CanvasState,
    layer_idx: usize,
    scale: f32,
    amount: f32,
    seed: u32,
    octaves: u32,
    roughness: f32,
    pinch: bool,
    wrap: bool,
    original_flat: &RgbaImage,
) {
    let result = dents_core(
        original_flat,
        scale,
        amount,
        seed,
        octaves,
        roughness,
        pinch,
        wrap,
        state.selection_mask.as_ref(),
    );
    commit_to_layer(state, layer_idx, &result);
}

/// Multi-octave turbulence noise for displacement
fn turbulence_2d(x: f32, y: f32, seed: u32, octaves: u32, roughness: f32) -> f32 {
    let mut total = 0.0f32;
    let mut amplitude = 1.0f32;
    let mut frequency = 1.0f32;
    let mut max_amplitude = 0.0f32;
    for i in 0..octaves {
        let s = seed.wrapping_add(i * 1000);
        total += perlin_noise_2d(x * frequency, y * frequency, s) * amplitude;
        max_amplitude += amplitude;
        amplitude *= roughness;
        frequency *= 2.0;
    }
    if max_amplitude > 0.0 {
        total / max_amplitude
    } else {
        0.0
    }
}

pub fn dents_core(
    flat: &RgbaImage,
    scale: f32,
    amount: f32,
    seed: u32,
    octaves: u32,
    roughness: f32,
    pinch: bool,
    wrap: bool,
    mask: Option<&GrayImage>,
) -> RgbaImage {
    let w = flat.width();
    let h = flat.height();
    if w == 0 || h == 0 {
        return flat.clone();
    }

    let oct = octaves.clamp(1, 8);
    let inv_scale = 1.0 / scale.max(0.5);

    apply_per_pixel(flat, mask, |x, y, _r, _g, _b, _a| {
        let nx_raw = turbulence_2d(
            x as f32 * inv_scale,
            y as f32 * inv_scale,
            seed,
            oct,
            roughness,
        ) * 2.0
            - 1.0;
        let ny_raw = turbulence_2d(
            x as f32 * inv_scale,
            y as f32 * inv_scale,
            seed.wrapping_add(9999),
            oct,
            roughness,
        ) * 2.0
            - 1.0;

        let (nx, ny) = if pinch {
            // Pinch mode: displacement toward center
            let cx = w as f32 * 0.5;
            let cy = h as f32 * 0.5;
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            let dist = (dx * dx + dy * dy).sqrt().max(1.0);
            let factor = (1.0 - dist / (cx.max(cy))) * 0.5;
            (nx_raw + dx / dist * factor, ny_raw + dy / dist * factor)
        } else {
            (nx_raw, ny_raw)
        };

        let mut src_x = x as f32 + nx * amount * scale;
        let mut src_y = y as f32 + ny * amount * scale;

        if wrap {
            src_x = src_x.rem_euclid(w as f32);
            src_y = src_y.rem_euclid(h as f32);
        }

        let p = sample_bilinear(flat, src_x, src_y);
        (p[0], p[1], p[2], p[3])
    })
}

// --- Pixelate ---

pub fn pixelate(state: &mut CanvasState, layer_idx: usize, block_size: u32) {
    if layer_idx >= state.layers.len() {
        return;
    }
    let flat = state.layers[layer_idx].pixels.to_rgba_image();
    let result = pixelate_core(&flat, block_size, state.selection_mask.as_ref());
    commit_to_layer(state, layer_idx, &result);
}

pub fn pixelate_from_flat(
    state: &mut CanvasState,
    layer_idx: usize,
    block_size: u32,
    original_flat: &RgbaImage,
) {
    let result = pixelate_core(original_flat, block_size, state.selection_mask.as_ref());
    commit_to_layer(state, layer_idx, &result);
}

pub fn pixelate_core(flat: &RgbaImage, block_size: u32, mask: Option<&GrayImage>) -> RgbaImage {
    let bs = block_size.max(2);
    // Same as crystallize but with nearest-neighbour (center pixel) instead of average.
    let w = flat.width();
    let h = flat.height();
    if w == 0 || h == 0 {
        return flat.clone();
    }

    let src_raw = flat.as_raw();
    let stride = w as usize * 4;
    let mut dst_raw = src_raw.clone();
    let mask_raw = mask.map(|m| m.as_raw().as_slice());
    let mask_w = mask.map_or(0, |m| m.width() as usize);
    let mask_h = mask.map_or(0, |m| m.height() as usize);

    dst_raw
        .par_chunks_mut(stride)
        .enumerate()
        .for_each(|(y, row_out)| {
            for x in 0..w as usize {
                if let Some(mr) = mask_raw
                    && x < mask_w
                    && y < mask_h
                    && mr[y * mask_w + x] == 0
                {
                    continue;
                }
                // Sample from the center of the block.
                let bx = (x as u32 / bs) * bs + bs / 2;
                let by = (y as u32 / bs) * bs + bs / 2;
                let sx = bx.min(w - 1) as usize;
                let sy = by.min(h - 1) as usize;
                let si = sy * stride + sx * 4;
                let pi = x * 4;
                row_out[pi..pi + 4].copy_from_slice(&src_raw[si..si + 4]);
            }
        });

    RgbaImage::from_raw(w, h, dst_raw).unwrap()
}

// --- Bulge ---

pub fn bulge(state: &mut CanvasState, layer_idx: usize, amount: f32) {
    if layer_idx >= state.layers.len() {
        return;
    }
    let flat = state.layers[layer_idx].pixels.to_rgba_image();
    let result = bulge_core(&flat, amount, state.selection_mask.as_ref());
    commit_to_layer(state, layer_idx, &result);
}

pub fn bulge_from_flat(
    state: &mut CanvasState,
    layer_idx: usize,
    amount: f32,
    original_flat: &RgbaImage,
) {
    let result = bulge_core(original_flat, amount, state.selection_mask.as_ref());
    commit_to_layer(state, layer_idx, &result);
}

pub fn bulge_core(flat: &RgbaImage, amount: f32, mask: Option<&GrayImage>) -> RgbaImage {
    let w = flat.width() as f32;
    let h = flat.height() as f32;
    let cx = w / 2.0;
    let cy = h / 2.0;
    let max_r = cx.min(cy);

    apply_per_pixel(flat, mask, |x, y, _r, _g, _b, _a| {
        let dx = x as f32 - cx;
        let dy = y as f32 - cy;
        let dist = (dx * dx + dy * dy).sqrt();
        let norm = (dist / max_r).min(1.0);

        if norm >= 1.0 {
            let p = sample_clamped(flat, x as i32, y as i32);
            return (p[0], p[1], p[2], p[3]);
        }

        // Spherical distortion
        let power = (1.0 - norm).powf(amount.abs());
        let factor = if amount > 0.0 {
            // Bulge out
            norm * (1.0 - power) + power
        } else {
            // Pinch in
            power
        };
        let src_x = cx + dx * factor;
        let src_y = cy + dy * factor;
        let p = sample_bilinear(flat, src_x, src_y);
        (p[0], p[1], p[2], p[3])
    })
}

// --- Twist ---

pub fn twist(state: &mut CanvasState, layer_idx: usize, angle_deg: f32) {
    if layer_idx >= state.layers.len() {
        return;
    }
    let flat = state.layers[layer_idx].pixels.to_rgba_image();
    let result = twist_core(&flat, angle_deg, state.selection_mask.as_ref());
    commit_to_layer(state, layer_idx, &result);
}

pub fn twist_from_flat(
    state: &mut CanvasState,
    layer_idx: usize,
    angle_deg: f32,
    original_flat: &RgbaImage,
) {
    let result = twist_core(original_flat, angle_deg, state.selection_mask.as_ref());
    commit_to_layer(state, layer_idx, &result);
}

pub fn twist_core(flat: &RgbaImage, angle_deg: f32, mask: Option<&GrayImage>) -> RgbaImage {
    let w = flat.width() as f32;
    let h = flat.height() as f32;
    let cx = w / 2.0;
    let cy = h / 2.0;
    let max_r = (cx * cx + cy * cy).sqrt();
    let twist_amount = angle_deg.to_radians();

    apply_per_pixel(flat, mask, |x, y, _r, _g, _b, _a| {
        let dx = x as f32 - cx;
        let dy = y as f32 - cy;
        let dist = (dx * dx + dy * dy).sqrt();
        let norm = dist / max_r;

        let rotation = twist_amount * (1.0 - norm);
        let cos_r = rotation.cos();
        let sin_r = rotation.sin();
        let src_x = cx + dx * cos_r - dy * sin_r;
        let src_y = cy + dx * sin_r + dy * cos_r;
        let p = sample_bilinear(flat, src_x, src_y);
        (p[0], p[1], p[2], p[3])
    })
}

// ============================================================================
// NOISE EFFECTS
// ============================================================================

// --- Add Noise ---

#[derive(Clone, Copy, PartialEq)]
pub enum NoiseType {
    Uniform,
    Gaussian,
    Perlin,
}

pub fn add_noise(
    state: &mut CanvasState,
    layer_idx: usize,
    amount: f32,
    noise_type: NoiseType,
    monochrome: bool,
    seed: u32,
    scale: f32,
    octaves: u32,
) {
    if layer_idx >= state.layers.len() {
        return;
    }
    let flat = state.layers[layer_idx].pixels.to_rgba_image();
    let result = add_noise_core(
        &flat,
        amount,
        noise_type,
        monochrome,
        seed,
        scale,
        octaves,
        state.selection_mask.as_ref(),
    );
    commit_to_layer(state, layer_idx, &result);
}

pub fn add_noise_from_flat(
    state: &mut CanvasState,
    layer_idx: usize,
    amount: f32,
    noise_type: NoiseType,
    monochrome: bool,
    seed: u32,
    scale: f32,
    octaves: u32,
    original_flat: &RgbaImage,
) {
    let result = add_noise_core(
        original_flat,
        amount,
        noise_type,
        monochrome,
        seed,
        scale,
        octaves,
        state.selection_mask.as_ref(),
    );
    commit_to_layer(state, layer_idx, &result);
}

/// Simple 2D Perlin-style value noise.
fn perlin_noise_2d(x: f32, y: f32, seed: u32) -> f32 {
    let xi = x.floor() as i32;
    let yi = y.floor() as i32;
    let xf = x - xi as f32;
    let yf = y - yi as f32;

    let fade = |t: f32| t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
    let u = fade(xf);
    let v = fade(yf);

    let n00 = hash_f32(xi as u32, yi as u32, seed);
    let n10 = hash_f32((xi + 1) as u32, yi as u32, seed);
    let n01 = hash_f32(xi as u32, (yi + 1) as u32, seed);
    let n11 = hash_f32((xi + 1) as u32, (yi + 1) as u32, seed);

    let nx0 = n00 + u * (n10 - n00);
    let nx1 = n01 + u * (n11 - n01);
    nx0 + v * (nx1 - nx0)
}

pub fn add_noise_core(
    flat: &RgbaImage,
    amount: f32,
    noise_type: NoiseType,
    monochrome: bool,
    seed: u32,
    scale: f32,
    octaves: u32,
    mask: Option<&GrayImage>,
) -> RgbaImage {
    let inv_scale = 1.0 / scale.max(0.1);
    let oct = octaves.clamp(1, 8);

    apply_per_pixel(flat, mask, |x, y, r, g, b, a| {
        let sx = x as f32 * inv_scale;
        let sy = y as f32 * inv_scale;

        let noise_val = match noise_type {
            NoiseType::Uniform => {
                // For uniform, scale controls block size: quantize coordinates
                let qx = (x as f32 * inv_scale).floor() as u32;
                let qy = (y as f32 * inv_scale).floor() as u32;
                hash_f32(qx, qy, seed) * 2.0 - 1.0
            }
            NoiseType::Gaussian => {
                let qx = (x as f32 * inv_scale).floor() as u32;
                let qy = (y as f32 * inv_scale).floor() as u32;
                let u1 = hash_f32(qx, qy, seed).max(0.0001);
                let u2 = hash_f32(qx, qy, seed.wrapping_add(7));
                (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos() * 0.33
            }
            NoiseType::Perlin => turbulence_2d(sx, sy, seed, oct, 0.5) * 2.0 - 1.0,
        };

        let strength = amount * 255.0 / 100.0;
        if monochrome {
            let n = noise_val * strength;
            (r + n, g + n, b + n, a)
        } else {
            let nr = match noise_type {
                NoiseType::Perlin => (turbulence_2d(sx, sy, seed, oct, 0.5) * 2.0 - 1.0) * strength,
                _ => {
                    let qx = (x as f32 * inv_scale).floor() as u32;
                    let qy = (y as f32 * inv_scale).floor() as u32;
                    (hash_f32(qx, qy, seed) * 2.0 - 1.0) * strength
                }
            };
            let ng = match noise_type {
                NoiseType::Perlin => {
                    (turbulence_2d(sx, sy, seed.wrapping_add(1), oct, 0.5) * 2.0 - 1.0) * strength
                }
                _ => {
                    let qx = (x as f32 * inv_scale).floor() as u32;
                    let qy = (y as f32 * inv_scale).floor() as u32;
                    (hash_f32(qx, qy, seed.wrapping_add(1)) * 2.0 - 1.0) * strength
                }
            };
            let nb = match noise_type {
                NoiseType::Perlin => {
                    (turbulence_2d(sx, sy, seed.wrapping_add(2), oct, 0.5) * 2.0 - 1.0) * strength
                }
                _ => {
                    let qx = (x as f32 * inv_scale).floor() as u32;
                    let qy = (y as f32 * inv_scale).floor() as u32;
                    (hash_f32(qx, qy, seed.wrapping_add(2)) * 2.0 - 1.0) * strength
                }
            };
            (r + nr, g + ng, b + nb, a)
        }
    })
}

// --- Reduce Noise (bilateral-like filter) ---

pub fn reduce_noise(state: &mut CanvasState, layer_idx: usize, strength: f32, radius: u32) {
    if layer_idx >= state.layers.len() {
        return;
    }
    let flat = state.layers[layer_idx].pixels.to_rgba_image();
    let result = reduce_noise_core(&flat, strength, radius, state.selection_mask.as_ref());
    commit_to_layer(state, layer_idx, &result);
}

pub fn reduce_noise_from_flat(
    state: &mut CanvasState,
    layer_idx: usize,
    strength: f32,
    radius: u32,
    original_flat: &RgbaImage,
) {
    let result = reduce_noise_core(
        original_flat,
        strength,
        radius,
        state.selection_mask.as_ref(),
    );
    commit_to_layer(state, layer_idx, &result);
}

pub fn reduce_noise_core(
    flat: &RgbaImage,
    strength: f32,
    radius: u32,
    mask: Option<&GrayImage>,
) -> RgbaImage {
    let w = flat.width() as usize;
    let h = flat.height() as usize;
    if w == 0 || h == 0 {
        return flat.clone();
    }

    let r = radius.max(1) as i32;
    let sigma_s = r as f32;
    let sigma_r = strength * 2.55; // 0-100 â†’ 0-255

    let src_raw = flat.as_raw();
    let stride = w * 4;
    let mut dst_raw = vec![0u8; w * h * 4];
    let mask_raw = mask.map(|m| m.as_raw().as_slice());
    let mask_w = mask.map_or(0, |m| m.width() as usize);
    let mask_h = mask.map_or(0, |m| m.height() as usize);

    dst_raw
        .par_chunks_mut(stride)
        .enumerate()
        .for_each(|(y, row_out)| {
            for x in 0..w {
                let pi = x * 4;
                if let Some(mr) = mask_raw
                    && x < mask_w
                    && y < mask_h
                    && mr[y * mask_w + x] == 0
                {
                    let si = y * stride + pi;
                    row_out[pi..pi + 4].copy_from_slice(&src_raw[si..si + 4]);
                    continue;
                }

                let center_si = y * stride + pi;
                let cr = src_raw[center_si] as f32;
                let cg = src_raw[center_si + 1] as f32;
                let cb = src_raw[center_si + 2] as f32;

                let mut sum_r = 0.0f32;
                let mut sum_g = 0.0f32;
                let mut sum_b = 0.0f32;
                let mut sum_a = 0.0f32;
                let mut weight_sum = 0.0f32;

                for dy in -r..=r {
                    let sy = (y as i32 + dy).clamp(0, h as i32 - 1) as usize;
                    for dx in -r..=r {
                        let sx = (x as i32 + dx).clamp(0, w as i32 - 1) as usize;
                        let si = sy * stride + sx * 4;
                        let pr = src_raw[si] as f32;
                        let pg = src_raw[si + 1] as f32;
                        let pb = src_raw[si + 2] as f32;
                        let pa = src_raw[si + 3] as f32;

                        let spatial = (dx * dx + dy * dy) as f32 / (2.0 * sigma_s * sigma_s);
                        let diff_r = cr - pr;
                        let diff_g = cg - pg;
                        let diff_b = cb - pb;
                        let range = (diff_r * diff_r + diff_g * diff_g + diff_b * diff_b)
                            / (2.0 * sigma_r * sigma_r + 0.001);
                        let weight = (-spatial - range).exp();

                        sum_r += pr * weight;
                        sum_g += pg * weight;
                        sum_b += pb * weight;
                        sum_a += pa * weight;
                        weight_sum += weight;
                    }
                }

                if weight_sum > 0.0 {
                    let inv = 1.0 / weight_sum;
                    row_out[pi] = (sum_r * inv).round().clamp(0.0, 255.0) as u8;
                    row_out[pi + 1] = (sum_g * inv).round().clamp(0.0, 255.0) as u8;
                    row_out[pi + 2] = (sum_b * inv).round().clamp(0.0, 255.0) as u8;
                    row_out[pi + 3] = (sum_a * inv).round().clamp(0.0, 255.0) as u8;
                } else {
                    row_out[pi..pi + 4].copy_from_slice(&src_raw[center_si..center_si + 4]);
                }
            }
        });

    RgbaImage::from_raw(w as u32, h as u32, dst_raw).unwrap()
}

// --- Median filter ---

pub fn median_filter(state: &mut CanvasState, layer_idx: usize, radius: u32) {
    if layer_idx >= state.layers.len() {
        return;
    }
    let flat = state.layers[layer_idx].pixels.to_rgba_image();
    let result = median_core(&flat, radius, state.selection_mask.as_ref());
    commit_to_layer(state, layer_idx, &result);
}

pub fn median_filter_from_flat(
    state: &mut CanvasState,
    layer_idx: usize,
    radius: u32,
    original_flat: &RgbaImage,
) {
    let result = median_core(original_flat, radius, state.selection_mask.as_ref());
    commit_to_layer(state, layer_idx, &result);
}

/// GPU-accelerated median filter.  Falls back to CPU for radius > 7 or when
/// a selection mask is present.
pub fn median_filter_gpu(
    state: &mut CanvasState,
    layer_idx: usize,
    radius: u32,
    gpu: &crate::gpu::GpuRenderer,
) {
    if layer_idx >= state.layers.len() {
        return;
    }
    if state.selection_mask.is_some() || radius > 7 {
        median_filter(state, layer_idx, radius);
        return;
    }
    let flat = state.layers[layer_idx].pixels.to_rgba_image();
    let (w, h) = (flat.width(), flat.height());
    if let Some(result_data) = gpu.median_rgba(flat.as_raw(), w, h, radius) {
        let result = image::RgbaImage::from_raw(w, h, result_data).unwrap();
        commit_to_layer(state, layer_idx, &result);
    } else {
        median_filter(state, layer_idx, radius);
    }
}

pub fn median_filter_from_flat_gpu(
    state: &mut CanvasState,
    layer_idx: usize,
    radius: u32,
    original_flat: &RgbaImage,
    gpu: &crate::gpu::GpuRenderer,
) {
    if state.selection_mask.is_some() || radius > 7 {
        median_filter_from_flat(state, layer_idx, radius, original_flat);
        return;
    }
    let (w, h) = (original_flat.width(), original_flat.height());
    if let Some(result_data) = gpu.median_rgba(original_flat.as_raw(), w, h, radius) {
        let result = image::RgbaImage::from_raw(w, h, result_data).unwrap();
        commit_to_layer(state, layer_idx, &result);
    } else {
        median_filter_from_flat(state, layer_idx, radius, original_flat);
    }
}

pub fn median_core(flat: &RgbaImage, radius: u32, mask: Option<&GrayImage>) -> RgbaImage {
    let w = flat.width() as usize;
    let h = flat.height() as usize;
    if w == 0 || h == 0 {
        return flat.clone();
    }

    let r = radius.max(1) as i32;
    let src_raw = flat.as_raw();
    let stride = w * 4;
    let mut dst_raw = vec![0u8; w * h * 4];
    let mask_raw = mask.map(|m| m.as_raw().as_slice());
    let mask_w = mask.map_or(0, |m| m.width() as usize);
    let mask_h = mask.map_or(0, |m| m.height() as usize);

    dst_raw
        .par_chunks_mut(stride)
        .enumerate()
        .for_each(|(y, row_out)| {
            let mut channels: [Vec<u8>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
            for x in 0..w {
                let pi = x * 4;
                if let Some(mr) = mask_raw
                    && x < mask_w
                    && y < mask_h
                    && mr[y * mask_w + x] == 0
                {
                    let si = y * stride + pi;
                    row_out[pi..pi + 4].copy_from_slice(&src_raw[si..si + 4]);
                    continue;
                }

                for c in &mut channels {
                    c.clear();
                }
                for dy in -r..=r {
                    let sy = (y as i32 + dy).clamp(0, h as i32 - 1) as usize;
                    for dx in -r..=r {
                        let sx = (x as i32 + dx).clamp(0, w as i32 - 1) as usize;
                        let si = sy * stride + sx * 4;
                        for c in 0..4 {
                            channels[c].push(src_raw[si + c]);
                        }
                    }
                }
                for c in 0..4 {
                    channels[c].sort_unstable();
                    row_out[pi + c] = channels[c][channels[c].len() / 2];
                }
            }
        });

    RgbaImage::from_raw(w as u32, h as u32, dst_raw).unwrap()
}

// ============================================================================
// STYLIZE EFFECTS
// ============================================================================

// --- Glow ---

pub fn glow(state: &mut CanvasState, layer_idx: usize, radius: f32, intensity: f32) {
    if layer_idx >= state.layers.len() {
        return;
    }
    let flat = state.layers[layer_idx].pixels.to_rgba_image();
    let result = glow_core(&flat, radius, intensity, state.selection_mask.as_ref());
    commit_to_layer(state, layer_idx, &result);
}

pub fn glow_from_flat(
    state: &mut CanvasState,
    layer_idx: usize,
    radius: f32,
    intensity: f32,
    original_flat: &RgbaImage,
) {
    let result = glow_core(
        original_flat,
        radius,
        intensity,
        state.selection_mask.as_ref(),
    );
    commit_to_layer(state, layer_idx, &result);
}

pub fn glow_core(
    flat: &RgbaImage,
    radius: f32,
    intensity: f32,
    mask: Option<&GrayImage>,
) -> RgbaImage {
    // Glow = original + blurred * intensity (screen blend)
    let blurred = crate::ops::filters::parallel_gaussian_blur_pub(flat, radius);
    let w = flat.width() as usize;
    let h = flat.height() as usize;
    let src_raw = flat.as_raw();
    let blur_raw = blurred.as_raw();
    let stride = w * 4;
    let mut dst_raw = vec![0u8; w * h * 4];
    let mask_raw = mask.map(|m| m.as_raw().as_slice());
    let mask_w = mask.map_or(0, |m| m.width() as usize);
    let mask_h = mask.map_or(0, |m| m.height() as usize);

    dst_raw
        .par_chunks_mut(stride)
        .enumerate()
        .for_each(|(y, row_out)| {
            for x in 0..w {
                let pi = x * 4;
                if let Some(mr) = mask_raw
                    && x < mask_w
                    && y < mask_h
                    && mr[y * mask_w + x] == 0
                {
                    let si = y * stride + pi;
                    row_out[pi..pi + 4].copy_from_slice(&src_raw[si..si + 4]);
                    continue;
                }
                let si = y * stride + pi;
                for c in 0..3 {
                    let s = src_raw[si + c] as f32 / 255.0;
                    let b = blur_raw[si + c] as f32 / 255.0;
                    // Screen blend: 1 - (1 - s) * (1 - b * intensity)
                    let result = 1.0 - (1.0 - s) * (1.0 - b * intensity);
                    row_out[pi + c] = (result * 255.0).round().clamp(0.0, 255.0) as u8;
                }
                row_out[pi + 3] = src_raw[si + 3]; // preserve alpha
            }
        });

    RgbaImage::from_raw(w as u32, h as u32, dst_raw).unwrap()
}

// --- Sharpen (unsharp mask) ---

pub fn sharpen(state: &mut CanvasState, layer_idx: usize, amount: f32, radius: f32) {
    if layer_idx >= state.layers.len() {
        return;
    }
    let flat = state.layers[layer_idx].pixels.to_rgba_image();
    let result = sharpen_core(&flat, amount, radius, state.selection_mask.as_ref());
    commit_to_layer(state, layer_idx, &result);
}

pub fn sharpen_from_flat(
    state: &mut CanvasState,
    layer_idx: usize,
    amount: f32,
    radius: f32,
    original_flat: &RgbaImage,
) {
    let result = sharpen_core(original_flat, amount, radius, state.selection_mask.as_ref());
    commit_to_layer(state, layer_idx, &result);
}

pub fn sharpen_core(
    flat: &RgbaImage,
    amount: f32,
    radius: f32,
    mask: Option<&GrayImage>,
) -> RgbaImage {
    // Unsharp mask: result = original + amount * (original - blurred)
    let blurred = crate::ops::filters::parallel_gaussian_blur_pub(flat, radius);
    let w = flat.width() as usize;
    let h = flat.height() as usize;
    let src_raw = flat.as_raw();
    let blur_raw = blurred.as_raw();
    let stride = w * 4;
    let mut dst_raw = vec![0u8; w * h * 4];
    let mask_raw = mask.map(|m| m.as_raw().as_slice());
    let mask_w = mask.map_or(0, |m| m.width() as usize);
    let mask_h = mask.map_or(0, |m| m.height() as usize);

    dst_raw
        .par_chunks_mut(stride)
        .enumerate()
        .for_each(|(y, row_out)| {
            for x in 0..w {
                let pi = x * 4;
                if let Some(mr) = mask_raw
                    && x < mask_w
                    && y < mask_h
                    && mr[y * mask_w + x] == 0
                {
                    let si = y * stride + pi;
                    row_out[pi..pi + 4].copy_from_slice(&src_raw[si..si + 4]);
                    continue;
                }
                let si = y * stride + pi;
                for c in 0..3 {
                    let s = src_raw[si + c] as f32;
                    let b = blur_raw[si + c] as f32;
                    let v = s + amount * (s - b);
                    row_out[pi + c] = v.round().clamp(0.0, 255.0) as u8;
                }
                row_out[pi + 3] = src_raw[si + 3];
            }
        });

    RgbaImage::from_raw(w as u32, h as u32, dst_raw).unwrap()
}

// --- Vignette ---

pub fn vignette(state: &mut CanvasState, layer_idx: usize, amount: f32, softness: f32) {
    if layer_idx >= state.layers.len() {
        return;
    }
    let flat = state.layers[layer_idx].pixels.to_rgba_image();
    let result = vignette_core(&flat, amount, softness, state.selection_mask.as_ref());
    commit_to_layer(state, layer_idx, &result);
}

pub fn vignette_from_flat(
    state: &mut CanvasState,
    layer_idx: usize,
    amount: f32,
    softness: f32,
    original_flat: &RgbaImage,
) {
    let result = vignette_core(
        original_flat,
        amount,
        softness,
        state.selection_mask.as_ref(),
    );
    commit_to_layer(state, layer_idx, &result);
}

pub fn vignette_core(
    flat: &RgbaImage,
    amount: f32,
    softness: f32,
    mask: Option<&GrayImage>,
) -> RgbaImage {
    let w = flat.width() as f32;
    let h = flat.height() as f32;
    let cx = w / 2.0;
    let cy = h / 2.0;
    let max_dist = (cx * cx + cy * cy).sqrt();
    let soft = softness.max(0.01);

    apply_per_pixel(flat, mask, |x, y, r, g, b, a| {
        let dx = x as f32 - cx;
        let dy = y as f32 - cy;
        let dist = (dx * dx + dy * dy).sqrt() / max_dist;
        let vignette_factor = 1.0 - (amount * ((dist / soft).min(1.0)).powf(2.0));
        let vf = vignette_factor.clamp(0.0, 1.0);
        (r * vf, g * vf, b * vf, a)
    })
}

// --- Halftone ---

#[derive(Clone, Copy, PartialEq)]
pub enum HalftoneShape {
    Circle,
    Square,
    Diamond,
    Line,
}

pub fn halftone(
    state: &mut CanvasState,
    layer_idx: usize,
    dot_size: f32,
    angle_deg: f32,
    shape: HalftoneShape,
) {
    if layer_idx >= state.layers.len() {
        return;
    }
    let flat = state.layers[layer_idx].pixels.to_rgba_image();
    let result = halftone_core(
        &flat,
        dot_size,
        angle_deg,
        shape,
        state.selection_mask.as_ref(),
    );
    commit_to_layer(state, layer_idx, &result);
}

pub fn halftone_from_flat(
    state: &mut CanvasState,
    layer_idx: usize,
    dot_size: f32,
    angle_deg: f32,
    shape: HalftoneShape,
    original_flat: &RgbaImage,
) {
    let result = halftone_core(
        original_flat,
        dot_size,
        angle_deg,
        shape,
        state.selection_mask.as_ref(),
    );
    commit_to_layer(state, layer_idx, &result);
}

pub fn halftone_core(
    flat: &RgbaImage,
    dot_size: f32,
    angle_deg: f32,
    shape: HalftoneShape,
    mask: Option<&GrayImage>,
) -> RgbaImage {
    let ds = dot_size.max(2.0);
    let angle = angle_deg.to_radians();
    let cos_a = angle.cos();
    let sin_a = angle.sin();

    apply_per_pixel(flat, mask, |x, y, r, g, b, a| {
        let lum = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0;

        // Rotate coordinate space
        let fx = x as f32 * cos_a + y as f32 * sin_a;
        let fy = -(x as f32) * sin_a + y as f32 * cos_a;

        // Position within the halftone cell
        let cell_x = (fx / ds).fract().abs();
        let cell_y = (fy / ds).fract().abs();
        let cx = cell_x - 0.5;
        let cy = cell_y - 0.5;

        let threshold = match shape {
            HalftoneShape::Circle => (cx * cx + cy * cy).sqrt() * 2.0,
            HalftoneShape::Square => cx.abs().max(cy.abs()) * 2.0,
            HalftoneShape::Diamond => cx.abs() + cy.abs(),
            HalftoneShape::Line => cy.abs() * 2.0,
        };

        let val = if threshold < lum { 255.0 } else { 0.0 };
        (val, val, val, a)
    })
}

// ============================================================================
// RENDER EFFECTS
// ============================================================================

// --- Grid ---

#[derive(Clone, Copy, PartialEq)]
pub enum GridStyle {
    Lines,
    Checkerboard,
}

pub fn render_grid(
    state: &mut CanvasState,
    layer_idx: usize,
    cell_w: u32,
    cell_h: u32,
    line_width: u32,
    color: [u8; 4],
    style: GridStyle,
    opacity: f32,
) {
    if layer_idx >= state.layers.len() {
        return;
    }
    let flat = state.layers[layer_idx].pixels.to_rgba_image();
    let result = grid_core(
        &flat,
        cell_w,
        cell_h,
        line_width,
        color,
        style,
        opacity,
        state.selection_mask.as_ref(),
    );
    commit_to_layer(state, layer_idx, &result);
}

pub fn render_grid_from_flat(
    state: &mut CanvasState,
    layer_idx: usize,
    cell_w: u32,
    cell_h: u32,
    line_width: u32,
    color: [u8; 4],
    style: GridStyle,
    opacity: f32,
    original_flat: &RgbaImage,
) {
    let result = grid_core(
        original_flat,
        cell_w,
        cell_h,
        line_width,
        color,
        style,
        opacity,
        state.selection_mask.as_ref(),
    );
    commit_to_layer(state, layer_idx, &result);
}

pub fn grid_core(
    flat: &RgbaImage,
    cell_w: u32,
    cell_h: u32,
    line_width: u32,
    color: [u8; 4],
    style: GridStyle,
    opacity: f32,
    mask: Option<&GrayImage>,
) -> RgbaImage {
    let cw = cell_w.max(2);
    let ch = cell_h.max(2);
    let lw = line_width.max(1);

    apply_per_pixel(flat, mask, |x, y, r, g, b, a| {
        let draw = match style {
            GridStyle::Lines => (x % cw) < lw || (y % ch) < lw,
            GridStyle::Checkerboard => {
                let cell_x = x / cw;
                let cell_y = y / ch;
                (cell_x + cell_y).is_multiple_of(2)
            }
        };

        if draw {
            let t = opacity;
            let gr = color[0] as f32;
            let gg = color[1] as f32;
            let gb = color[2] as f32;
            let ga = color[3] as f32;
            (
                r * (1.0 - t) + gr * t,
                g * (1.0 - t) + gg * t,
                b * (1.0 - t) + gb * t,
                a * (1.0 - t) + ga * t,
            )
        } else {
            (r, g, b, a)
        }
    })
}

// --- Drop Shadow ---

pub fn drop_shadow(
    state: &mut CanvasState,
    layer_idx: usize,
    offset_x: i32,
    offset_y: i32,
    blur_radius: f32,
    color: [u8; 4],
    opacity: f32,
) {
    if layer_idx >= state.layers.len() {
        return;
    }
    let flat = state.layers[layer_idx].pixels.to_rgba_image();
    let result = shadow_core(
        &flat,
        offset_x,
        offset_y,
        blur_radius,
        color,
        opacity,
        state.selection_mask.as_ref(),
    );
    commit_to_layer(state, layer_idx, &result);
}

pub fn drop_shadow_from_flat(
    state: &mut CanvasState,
    layer_idx: usize,
    offset_x: i32,
    offset_y: i32,
    blur_radius: f32,
    color: [u8; 4],
    opacity: f32,
    original_flat: &RgbaImage,
) {
    let result = shadow_core(
        original_flat,
        offset_x,
        offset_y,
        blur_radius,
        color,
        opacity,
        state.selection_mask.as_ref(),
    );
    commit_to_layer(state, layer_idx, &result);
}

pub fn shadow_core(
    flat: &RgbaImage,
    offset_x: i32,
    offset_y: i32,
    blur_radius: f32,
    color: [u8; 4],
    opacity: f32,
    mask: Option<&GrayImage>,
) -> RgbaImage {
    let w = flat.width();
    let h = flat.height();

    // 1. Create alpha mask from source (offset).
    let mut shadow_alpha = vec![0u8; (w * h) as usize];
    let src_raw = flat.as_raw();
    let stride = w as usize * 4;

    for y in 0..h as i32 {
        for x in 0..w as i32 {
            let sx = x - offset_x;
            let sy = y - offset_y;
            if sx >= 0 && sx < w as i32 && sy >= 0 && sy < h as i32 {
                let si = sy as usize * stride + sx as usize * 4;
                shadow_alpha[y as usize * w as usize + x as usize] = src_raw[si + 3];
            }
        }
    }

    // 2. Blur the alpha mask.
    let alpha_img = GrayImage::from_raw(w, h, shadow_alpha).unwrap();
    let alpha_rgba = RgbaImage::from_fn(w, h, |x, y| {
        let a = alpha_img.get_pixel(x, y)[0];
        Rgba([a, a, a, a])
    });
    let blurred_alpha_rgba = if blur_radius > 0.5 {
        crate::ops::filters::parallel_gaussian_blur_pub(&alpha_rgba, blur_radius)
    } else {
        alpha_rgba
    };

    // 3. Composite: shadow underneath, original on top.
    let mask_raw_sel = mask.map(|m| m.as_raw().as_slice());
    let mask_w = mask.map_or(0, |m| m.width() as usize);
    let mask_h = mask.map_or(0, |m| m.height() as usize);
    let blur_raw = blurred_alpha_rgba.as_raw();
    let mut dst_raw = vec![0u8; (w * h * 4) as usize];

    dst_raw
        .par_chunks_mut(stride)
        .enumerate()
        .for_each(|(y, row_out)| {
            for x in 0..w as usize {
                let pi = x * 4;
                if let Some(mr) = mask_raw_sel
                    && x < mask_w
                    && y < mask_h
                    && mr[y * mask_w + x] == 0
                {
                    let si = y * stride + pi;
                    row_out[pi..pi + 4].copy_from_slice(&src_raw[si..si + 4]);
                    continue;
                }
                let si = y * stride + pi;
                let shadow_a = (blur_raw[y * stride + pi] as f32 / 255.0) * opacity;
                let src_a = src_raw[si + 3] as f32 / 255.0;

                // Shadow first, then source on top (premultiplied-style compositing).
                for c in 0..3 {
                    let shadow_c = color[c] as f32 * shadow_a;
                    let src_c = src_raw[si + c] as f32 * src_a;
                    let out_c = src_c + shadow_c * (1.0 - src_a);
                    row_out[pi + c] = out_c.round().clamp(0.0, 255.0) as u8;
                }
                let out_a = src_a + shadow_a * (1.0 - src_a);
                row_out[pi + 3] = (out_a * 255.0).round().clamp(0.0, 255.0) as u8;
            }
        });

    RgbaImage::from_raw(w, h, dst_raw).unwrap()
}

// --- Outline ---

#[derive(Clone, Copy, PartialEq)]
pub enum OutlineMode {
    Outside,
    Inside,
    Center,
}

pub fn outline(
    state: &mut CanvasState,
    layer_idx: usize,
    width: u32,
    color: [u8; 4],
    mode: OutlineMode,
) {
    if layer_idx >= state.layers.len() {
        return;
    }
    let flat = state.layers[layer_idx].pixels.to_rgba_image();
    let result = outline_core(&flat, width, color, mode, state.selection_mask.as_ref());
    commit_to_layer(state, layer_idx, &result);
}

pub fn outline_from_flat(
    state: &mut CanvasState,
    layer_idx: usize,
    width: u32,
    color: [u8; 4],
    mode: OutlineMode,
    original_flat: &RgbaImage,
) {
    let result = outline_core(
        original_flat,
        width,
        color,
        mode,
        state.selection_mask.as_ref(),
    );
    commit_to_layer(state, layer_idx, &result);
}

pub fn outline_core(
    flat: &RgbaImage,
    width: u32,
    color: [u8; 4],
    mode: OutlineMode,
    mask: Option<&GrayImage>,
) -> RgbaImage {
    let w = flat.width() as usize;
    let h = flat.height() as usize;
    if w == 0 || h == 0 {
        return flat.clone();
    }

    let ow = width.max(1) as i32;
    let src_raw = flat.as_raw();
    let stride = w * 4;

    // Build dilated and eroded alpha masks for edge detection.
    let alpha: Vec<u8> = (0..w * h).map(|i| src_raw[i * 4 + 3]).collect();

    let dilated: Vec<u8> = (0..w * h)
        .into_par_iter()
        .map(|idx| {
            let x = idx % w;
            let y = idx / w;
            let mut max_a = 0u8;
            for dy in -ow..=ow {
                for dx in -ow..=ow {
                    if dx * dx + dy * dy <= ow * ow {
                        let sx = (x as i32 + dx).clamp(0, w as i32 - 1) as usize;
                        let sy = (y as i32 + dy).clamp(0, h as i32 - 1) as usize;
                        max_a = max_a.max(alpha[sy * w + sx]);
                    }
                }
            }
            max_a
        })
        .collect();

    let eroded: Vec<u8> = (0..w * h)
        .into_par_iter()
        .map(|idx| {
            let x = idx % w;
            let y = idx / w;
            let mut min_a = 255u8;
            for dy in -ow..=ow {
                for dx in -ow..=ow {
                    if dx * dx + dy * dy <= ow * ow {
                        let sx = (x as i32 + dx).clamp(0, w as i32 - 1) as usize;
                        let sy = (y as i32 + dy).clamp(0, h as i32 - 1) as usize;
                        min_a = min_a.min(alpha[sy * w + sx]);
                    }
                }
            }
            min_a
        })
        .collect();

    let mask_raw = mask.map(|m| m.as_raw().as_slice());
    let mask_w = mask.map_or(0, |m| m.width() as usize);
    let mask_h = mask.map_or(0, |m| m.height() as usize);
    let mut dst_raw = src_raw.to_vec();

    dst_raw
        .par_chunks_mut(stride)
        .enumerate()
        .for_each(|(y, row_out)| {
            for x in 0..w {
                let pi = x * 4;
                if let Some(mr) = mask_raw
                    && x < mask_w
                    && y < mask_h
                    && mr[y * mask_w + x] == 0
                {
                    continue;
                }

                let idx = y * w + x;
                let is_edge = match mode {
                    OutlineMode::Outside => dilated[idx] > 0 && alpha[idx] == 0,
                    OutlineMode::Inside => alpha[idx] > 0 && eroded[idx] == 0,
                    OutlineMode::Center => dilated[idx] > 0 && eroded[idx] == 0,
                };

                if is_edge {
                    row_out[pi] = color[0];
                    row_out[pi + 1] = color[1];
                    row_out[pi + 2] = color[2];
                    row_out[pi + 3] = color[3];
                }
            }
        });

    RgbaImage::from_raw(w as u32, h as u32, dst_raw).unwrap()
}

// ============================================================================
// GLITCH EFFECTS
// ============================================================================

// --- Pixel Drag ---

pub fn pixel_drag(
    state: &mut CanvasState,
    layer_idx: usize,
    seed: u32,
    amount: f32,
    distance: u32,
    direction: f32,
) {
    if layer_idx >= state.layers.len() {
        return;
    }
    let flat = state.layers[layer_idx].pixels.to_rgba_image();
    let result = pixel_drag_core(
        &flat,
        seed,
        amount,
        distance,
        direction,
        state.selection_mask.as_ref(),
    );
    commit_to_layer(state, layer_idx, &result);
}

pub fn pixel_drag_from_flat(
    state: &mut CanvasState,
    layer_idx: usize,
    seed: u32,
    amount: f32,
    distance: u32,
    direction: f32,
    original_flat: &RgbaImage,
) {
    let result = pixel_drag_core(
        original_flat,
        seed,
        amount,
        distance,
        direction,
        state.selection_mask.as_ref(),
    );
    commit_to_layer(state, layer_idx, &result);
}

pub fn pixel_drag_core(
    flat: &RgbaImage,
    seed: u32,
    amount: f32,
    distance: u32,
    direction: f32,
    mask: Option<&GrayImage>,
) -> RgbaImage {
    let w = flat.width() as usize;
    let h = flat.height() as usize;
    if w == 0 || h == 0 {
        return flat.clone();
    }

    let src_raw = flat.as_raw();
    let stride = w * 4;
    let mut dst_raw = src_raw.to_vec();
    let mask_raw = mask.map(|m| m.as_raw().as_slice());
    let mask_w = mask.map_or(0, |m| m.width() as usize);
    let mask_h = mask.map_or(0, |m| m.height() as usize);
    let dir_rad = direction.to_radians();
    let dx_dir = dir_rad.cos();
    let dy_dir = dir_rad.sin();
    let dist = distance.max(1) as f32;

    // Generate drag bands per row
    dst_raw
        .par_chunks_mut(stride)
        .enumerate()
        .for_each(|(y, row_out)| {
            let row_hash = hash_f32(y as u32, 0, seed);
            if row_hash > amount / 100.0 {
                return; // This row is not affected.
            }
            let drag_dist = (hash_f32(y as u32, 1, seed) * dist) as i32;

            for x in 0..w {
                let pi = x * 4;
                if let Some(mr) = mask_raw
                    && x < mask_w
                    && y < mask_h
                    && mr[y * mask_w + x] == 0
                {
                    continue;
                }
                let sx = (x as f32 - drag_dist as f32 * dx_dir).round() as i32;
                let sy = (y as f32 - drag_dist as f32 * dy_dir).round() as i32;
                let sx = sx.clamp(0, w as i32 - 1) as usize;
                let sy = sy.clamp(0, h as i32 - 1) as usize;
                let si = sy * stride + sx * 4;
                row_out[pi..pi + 4].copy_from_slice(&src_raw[si..si + 4]);
            }
        });

    RgbaImage::from_raw(w as u32, h as u32, dst_raw).unwrap()
}

// --- RGB Displace ---

pub fn rgb_displace(
    state: &mut CanvasState,
    layer_idx: usize,
    r_offset: (i32, i32),
    g_offset: (i32, i32),
    b_offset: (i32, i32),
) {
    if layer_idx >= state.layers.len() {
        return;
    }
    let flat = state.layers[layer_idx].pixels.to_rgba_image();
    let result = rgb_displace_core(
        &flat,
        r_offset,
        g_offset,
        b_offset,
        state.selection_mask.as_ref(),
    );
    commit_to_layer(state, layer_idx, &result);
}

pub fn rgb_displace_from_flat(
    state: &mut CanvasState,
    layer_idx: usize,
    r_offset: (i32, i32),
    g_offset: (i32, i32),
    b_offset: (i32, i32),
    original_flat: &RgbaImage,
) {
    let result = rgb_displace_core(
        original_flat,
        r_offset,
        g_offset,
        b_offset,
        state.selection_mask.as_ref(),
    );
    commit_to_layer(state, layer_idx, &result);
}

pub fn rgb_displace_core(
    flat: &RgbaImage,
    r_off: (i32, i32),
    g_off: (i32, i32),
    b_off: (i32, i32),
    mask: Option<&GrayImage>,
) -> RgbaImage {
    let w = flat.width() as usize;
    let h = flat.height() as usize;
    if w == 0 || h == 0 {
        return flat.clone();
    }

    let src_raw = flat.as_raw();
    let stride = w * 4;
    let mut dst_raw = vec![0u8; w * h * 4];
    let mask_raw = mask.map(|m| m.as_raw().as_slice());
    let mask_w = mask.map_or(0, |m| m.width() as usize);
    let mask_h = mask.map_or(0, |m| m.height() as usize);

    dst_raw
        .par_chunks_mut(stride)
        .enumerate()
        .for_each(|(y, row_out)| {
            for x in 0..w {
                let pi = x * 4;
                if let Some(mr) = mask_raw
                    && x < mask_w
                    && y < mask_h
                    && mr[y * mask_w + x] == 0
                {
                    let si = y * stride + pi;
                    row_out[pi..pi + 4].copy_from_slice(&src_raw[si..si + 4]);
                    continue;
                }

                // Sample each channel from offset position.
                let rx = (x as i32 + r_off.0).clamp(0, w as i32 - 1) as usize;
                let ry = (y as i32 + r_off.1).clamp(0, h as i32 - 1) as usize;
                let gx = (x as i32 + g_off.0).clamp(0, w as i32 - 1) as usize;
                let gy = (y as i32 + g_off.1).clamp(0, h as i32 - 1) as usize;
                let bx = (x as i32 + b_off.0).clamp(0, w as i32 - 1) as usize;
                let by = (y as i32 + b_off.1).clamp(0, h as i32 - 1) as usize;

                row_out[pi] = src_raw[ry * stride + rx * 4];
                row_out[pi + 1] = src_raw[gy * stride + gx * 4 + 1];
                row_out[pi + 2] = src_raw[by * stride + bx * 4 + 2];
                // Alpha from center pixel.
                let si = y * stride + pi;
                row_out[pi + 3] = src_raw[si + 3];
            }
        });

    RgbaImage::from_raw(w as u32, h as u32, dst_raw).unwrap()
}

// ============================================================================
// ARTISTIC EFFECTS
// ============================================================================

// --- Ink ---

pub fn ink(state: &mut CanvasState, layer_idx: usize, edge_strength: f32, threshold: f32) {
    if layer_idx >= state.layers.len() {
        return;
    }
    let flat = state.layers[layer_idx].pixels.to_rgba_image();
    let result = ink_core(
        &flat,
        edge_strength,
        threshold,
        state.selection_mask.as_ref(),
    );
    commit_to_layer(state, layer_idx, &result);
}

pub fn ink_from_flat(
    state: &mut CanvasState,
    layer_idx: usize,
    edge_strength: f32,
    threshold: f32,
    original_flat: &RgbaImage,
) {
    let result = ink_core(
        original_flat,
        edge_strength,
        threshold,
        state.selection_mask.as_ref(),
    );
    commit_to_layer(state, layer_idx, &result);
}

pub fn ink_core(
    flat: &RgbaImage,
    edge_strength: f32,
    threshold: f32,
    mask: Option<&GrayImage>,
) -> RgbaImage {
    // Sobel edge detection â†’ thresholded black/white ink effect.
    let w = flat.width() as usize;
    let h = flat.height() as usize;
    if w == 0 || h == 0 {
        return flat.clone();
    }

    let src_raw = flat.as_raw();
    let stride = w * 4;
    let mut dst_raw = vec![0u8; w * h * 4];
    let mask_raw = mask.map(|m| m.as_raw().as_slice());
    let mask_w = mask.map_or(0, |m| m.width() as usize);
    let mask_h = mask.map_or(0, |m| m.height() as usize);

    dst_raw
        .par_chunks_mut(stride)
        .enumerate()
        .for_each(|(y, row_out)| {
            for x in 0..w {
                let pi = x * 4;
                if let Some(mr) = mask_raw
                    && x < mask_w
                    && y < mask_h
                    && mr[y * mask_w + x] == 0
                {
                    let si = y * stride + pi;
                    row_out[pi..pi + 4].copy_from_slice(&src_raw[si..si + 4]);
                    continue;
                }

                // Compute luminance of surrounding pixels for Sobel.
                let lum = |px: i32, py: i32| -> f32 {
                    let cx = px.clamp(0, w as i32 - 1) as usize;
                    let cy = py.clamp(0, h as i32 - 1) as usize;
                    let si = cy * stride + cx * 4;
                    0.2126 * src_raw[si] as f32
                        + 0.7152 * src_raw[si + 1] as f32
                        + 0.0722 * src_raw[si + 2] as f32
                };

                let ix = x as i32;
                let iy = y as i32;
                let gx = -lum(ix - 1, iy - 1) - 2.0 * lum(ix - 1, iy) - lum(ix - 1, iy + 1)
                    + lum(ix + 1, iy - 1)
                    + 2.0 * lum(ix + 1, iy)
                    + lum(ix + 1, iy + 1);
                let gy = -lum(ix - 1, iy - 1) - 2.0 * lum(ix, iy - 1) - lum(ix + 1, iy - 1)
                    + lum(ix - 1, iy + 1)
                    + 2.0 * lum(ix, iy + 1)
                    + lum(ix + 1, iy + 1);
                let edge = (gx * gx + gy * gy).sqrt() * edge_strength / 100.0;
                let val = if edge > threshold { 0u8 } else { 255u8 };

                let si = y * stride + pi;
                row_out[pi] = val;
                row_out[pi + 1] = val;
                row_out[pi + 2] = val;
                row_out[pi + 3] = src_raw[si + 3];
            }
        });

    RgbaImage::from_raw(w as u32, h as u32, dst_raw).unwrap()
}

// --- Oil Painting ---

pub fn oil_painting(state: &mut CanvasState, layer_idx: usize, radius: u32, levels: u32) {
    if layer_idx >= state.layers.len() {
        return;
    }
    let flat = state.layers[layer_idx].pixels.to_rgba_image();
    let result = oil_painting_core(&flat, radius, levels, state.selection_mask.as_ref());
    commit_to_layer(state, layer_idx, &result);
}

pub fn oil_painting_from_flat(
    state: &mut CanvasState,
    layer_idx: usize,
    radius: u32,
    levels: u32,
    original_flat: &RgbaImage,
) {
    let result = oil_painting_core(original_flat, radius, levels, state.selection_mask.as_ref());
    commit_to_layer(state, layer_idx, &result);
}

pub fn oil_painting_core(
    flat: &RgbaImage,
    radius: u32,
    levels: u32,
    mask: Option<&GrayImage>,
) -> RgbaImage {
    let w = flat.width() as usize;
    let h = flat.height() as usize;
    if w == 0 || h == 0 {
        return flat.clone();
    }

    let r = radius.clamp(1, 10) as i32;
    let num_levels = levels.clamp(2, 64) as usize;
    let src_raw = flat.as_raw();
    let stride = w * 4;
    let mut dst_raw = vec![0u8; w * h * 4];
    let mask_raw = mask.map(|m| m.as_raw().as_slice());
    let mask_w = mask.map_or(0, |m| m.width() as usize);
    let mask_h = mask.map_or(0, |m| m.height() as usize);

    dst_raw
        .par_chunks_mut(stride)
        .enumerate()
        .for_each(|(y, row_out)| {
            let mut intensity_count = vec![0u32; num_levels];
            let mut sum_r = vec![0u32; num_levels];
            let mut sum_g = vec![0u32; num_levels];
            let mut sum_b = vec![0u32; num_levels];

            for x in 0..w {
                let pi = x * 4;
                if let Some(mr) = mask_raw
                    && x < mask_w
                    && y < mask_h
                    && mr[y * mask_w + x] == 0
                {
                    let si = y * stride + pi;
                    row_out[pi..pi + 4].copy_from_slice(&src_raw[si..si + 4]);
                    continue;
                }

                // Reset bins.
                for i in 0..num_levels {
                    intensity_count[i] = 0;
                    sum_r[i] = 0;
                    sum_g[i] = 0;
                    sum_b[i] = 0;
                }

                for dy in -r..=r {
                    let sy = (y as i32 + dy).clamp(0, h as i32 - 1) as usize;
                    for dx in -r..=r {
                        let sx = (x as i32 + dx).clamp(0, w as i32 - 1) as usize;
                        let si = sy * stride + sx * 4;
                        let pr = src_raw[si] as u32;
                        let pg = src_raw[si + 1] as u32;
                        let pb = src_raw[si + 2] as u32;
                        let intensity = ((pr + pg + pb) / 3 * num_levels as u32 / 256) as usize;
                        let intensity = intensity.min(num_levels - 1);
                        intensity_count[intensity] += 1;
                        sum_r[intensity] += pr;
                        sum_g[intensity] += pg;
                        sum_b[intensity] += pb;
                    }
                }

                // Find the most common intensity bin.
                let mut max_count = 0u32;
                let mut max_idx = 0usize;
                for (i, &count) in intensity_count.iter().enumerate() {
                    if count > max_count {
                        max_count = count;
                        max_idx = i;
                    }
                }

                if max_count > 0 {
                    row_out[pi] = (sum_r[max_idx] / max_count) as u8;
                    row_out[pi + 1] = (sum_g[max_idx] / max_count) as u8;
                    row_out[pi + 2] = (sum_b[max_idx] / max_count) as u8;
                }
                let si = y * stride + pi;
                row_out[pi + 3] = src_raw[si + 3];
            }
        });

    RgbaImage::from_raw(w as u32, h as u32, dst_raw).unwrap()
}

// --- Color Filter ---

#[derive(Clone, Copy, PartialEq)]
pub enum ColorFilterMode {
    Multiply,
    Screen,
    Overlay,
    SoftLight,
}

pub fn color_filter(
    state: &mut CanvasState,
    layer_idx: usize,
    filter_color: [u8; 4],
    intensity: f32,
    mode: ColorFilterMode,
) {
    if layer_idx >= state.layers.len() {
        return;
    }
    let flat = state.layers[layer_idx].pixels.to_rgba_image();
    let result = color_filter_core(
        &flat,
        filter_color,
        intensity,
        mode,
        state.selection_mask.as_ref(),
    );
    commit_to_layer(state, layer_idx, &result);
}

pub fn color_filter_from_flat(
    state: &mut CanvasState,
    layer_idx: usize,
    filter_color: [u8; 4],
    intensity: f32,
    mode: ColorFilterMode,
    original_flat: &RgbaImage,
) {
    let result = color_filter_core(
        original_flat,
        filter_color,
        intensity,
        mode,
        state.selection_mask.as_ref(),
    );
    commit_to_layer(state, layer_idx, &result);
}

pub fn color_filter_core(
    flat: &RgbaImage,
    filter_color: [u8; 4],
    intensity: f32,
    mode: ColorFilterMode,
    mask: Option<&GrayImage>,
) -> RgbaImage {
    let fc = [
        filter_color[0] as f32 / 255.0,
        filter_color[1] as f32 / 255.0,
        filter_color[2] as f32 / 255.0,
    ];

    apply_per_pixel(flat, mask, |_x, _y, r, g, b, a| {
        let rs = r / 255.0;
        let gs = g / 255.0;
        let bs = b / 255.0;
        let blend_fn = |s: f32, f: f32| -> f32 {
            match mode {
                ColorFilterMode::Multiply => s * f,
                ColorFilterMode::Screen => 1.0 - (1.0 - s) * (1.0 - f),
                ColorFilterMode::Overlay => {
                    if s < 0.5 {
                        2.0 * s * f
                    } else {
                        1.0 - 2.0 * (1.0 - s) * (1.0 - f)
                    }
                }
                ColorFilterMode::SoftLight => {
                    if f < 0.5 {
                        s - (1.0 - 2.0 * f) * s * (1.0 - s)
                    } else {
                        s + (2.0 * f - 1.0) * (s.sqrt() - s)
                    }
                }
            }
        };

        let nr = (rs * (1.0 - intensity) + blend_fn(rs, fc[0]) * intensity) * 255.0;
        let ng = (gs * (1.0 - intensity) + blend_fn(gs, fc[1]) * intensity) * 255.0;
        let nb = (bs * (1.0 - intensity) + blend_fn(bs, fc[2]) * intensity) * 255.0;
        (nr, ng, nb, a)
    })
}

// ============================================================================
// RENDER â€” Contours (topographic map lines)
// ============================================================================

pub fn contours(
    state: &mut CanvasState,
    layer_idx: usize,
    scale: f32,
    frequency: f32,
    line_width: f32,
    line_color: [u8; 4],
    seed: u32,
    octaves: u32,
    blend: f32,
) {
    if layer_idx >= state.layers.len() {
        return;
    }
    let flat = state.layers[layer_idx].pixels.to_rgba_image();
    let result = contours_core(
        &flat,
        scale,
        frequency,
        line_width,
        line_color,
        seed,
        octaves,
        blend,
        state.selection_mask.as_ref(),
    );
    commit_to_layer(state, layer_idx, &result);
}

pub fn contours_from_flat(
    state: &mut CanvasState,
    layer_idx: usize,
    scale: f32,
    frequency: f32,
    line_width: f32,
    line_color: [u8; 4],
    seed: u32,
    octaves: u32,
    blend: f32,
    original_flat: &RgbaImage,
) {
    let result = contours_core(
        original_flat,
        scale,
        frequency,
        line_width,
        line_color,
        seed,
        octaves,
        blend,
        state.selection_mask.as_ref(),
    );
    commit_to_layer(state, layer_idx, &result);
}

pub fn contours_core(
    flat: &RgbaImage,
    scale: f32,
    frequency: f32,
    line_width: f32,
    line_color: [u8; 4],
    seed: u32,
    octaves: u32,
    blend: f32,
    mask: Option<&GrayImage>,
) -> RgbaImage {
    let w = flat.width() as usize;
    let h = flat.height() as usize;
    if w == 0 || h == 0 {
        return flat.clone();
    }

    let inv_scale = 1.0 / scale.max(0.5);
    let oct = octaves.clamp(1, 8);
    let half_lw = (line_width * 0.5).max(0.3);
    let lr = line_color[0] as f32;
    let lg = line_color[1] as f32;
    let lb = line_color[2] as f32;
    let la = line_color[3] as f32 / 255.0;

    // Frequency controls how many contour levels in the 0..1 noise range
    let freq = frequency.max(0.5);

    apply_per_pixel(flat, mask, |x, y, r, g, b, a| {
        // Sample noise field
        let sx = x as f32 * inv_scale;
        let sy = y as f32 * inv_scale;
        let noise_val = turbulence_2d(sx, sy, seed, oct, 0.5);

        // Map to contour: distance to nearest contour level
        let level = noise_val * freq;
        let dist_to_contour = (level - level.round()).abs() / freq;

        // Anti-aliased contour line (smooth step)
        let edge = half_lw * inv_scale * 0.5;
        let line_alpha = if dist_to_contour < edge {
            1.0
        } else if dist_to_contour < edge * 2.0 {
            1.0 - (dist_to_contour - edge) / edge
        } else {
            0.0
        };

        let alpha = line_alpha * la * blend;

        // Blend contour line color over original pixel
        let nr = r * (1.0 - alpha) + lr * alpha;
        let ng = g * (1.0 - alpha) + lg * alpha;
        let nb = b * (1.0 - alpha) + lb * alpha;
        (nr, ng, nb, a)
    })
}
