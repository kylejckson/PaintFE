// ============================================================================
// ADJUSTMENT OPERATIONS — pixel-level image adjustments (selection + layer aware)
// ============================================================================
//
// All operations work on the active layer only.
// If a selection mask exists, only selected pixels are modified.
// Operations are parallelized via rayon for multi-core performance.
// ============================================================================

use image::RgbaImage;
use rayon::prelude::*;
use crate::canvas::{CanvasState, TiledImage};

// ============================================================================
// HELPER: selection-aware per-pixel transform
// ============================================================================

/// Apply a per-pixel transform function to the active layer.
/// If a selection mask exists, only selected pixels are affected.
/// `transform` receives (r, g, b, a) as f32 and returns (r, g, b, a) as f32.
fn apply_pixel_transform<F>(
    state: &mut CanvasState,
    layer_idx: usize,
    transform: F,
)
where
    F: Fn(f32, f32, f32, f32) -> (f32, f32, f32, f32) + Sync,
{
    if layer_idx >= state.layers.len() { return; }
    let layer = &mut state.layers[layer_idx];
    let w = layer.pixels.width() as usize;
    let h = layer.pixels.height() as usize;
    if w == 0 || h == 0 { return; }

    let flat = layer.pixels.to_rgba_image();
    let src_raw = flat.as_raw();
    let mut dst_raw = vec![0u8; w * h * 4];
    let stride = w * 4;

    let mask_raw = state.selection_mask.as_ref().map(|m| m.as_raw().as_slice());
    let mask_w = state.selection_mask.as_ref().map_or(0, |m| m.width() as usize);
    let mask_h = state.selection_mask.as_ref().map_or(0, |m| m.height() as usize);

    dst_raw.par_chunks_mut(stride).enumerate().for_each(|(y, row_out)| {
        let row_in = &src_raw[y * stride..(y + 1) * stride];
        for x in 0..w {
            let pi = x * 4;
            // Check selection mask
            if let Some(mr) = mask_raw {
                if x < mask_w && y < mask_h && mr[y * mask_w + x] == 0 {
                    row_out[pi..pi + 4].copy_from_slice(&row_in[pi..pi + 4]);
                    continue;
                }
            }
            let r = row_in[pi] as f32;
            let g = row_in[pi + 1] as f32;
            let b = row_in[pi + 2] as f32;
            let a = row_in[pi + 3] as f32;
            let (nr, ng, nb, na) = transform(r, g, b, a);
            row_out[pi]     = nr.round().clamp(0.0, 255.0) as u8;
            row_out[pi + 1] = ng.round().clamp(0.0, 255.0) as u8;
            row_out[pi + 2] = nb.round().clamp(0.0, 255.0) as u8;
            row_out[pi + 3] = na.round().clamp(0.0, 255.0) as u8;
        }
    });

    let out = RgbaImage::from_raw(w as u32, h as u32, dst_raw).unwrap();
    let layer = &mut state.layers[layer_idx];
    layer.pixels = TiledImage::from_rgba_image(&out);
    state.mark_dirty(None);
}

/// Same as apply_pixel_transform but operates from a pre-flattened original
/// (for live preview without repeated to_rgba_image() calls).
pub fn apply_pixel_transform_from_flat<F>(
    state: &mut CanvasState,
    layer_idx: usize,
    original_flat: &RgbaImage,
    transform: F,
)
where
    F: Fn(f32, f32, f32, f32) -> (f32, f32, f32, f32) + Sync,
{
    if layer_idx >= state.layers.len() { return; }
    let w = original_flat.width() as usize;
    let h = original_flat.height() as usize;
    if w == 0 || h == 0 { return; }

    let src_raw = original_flat.as_raw();
    let mut dst_raw = vec![0u8; w * h * 4];
    let stride = w * 4;

    let mask_raw = state.selection_mask.as_ref().map(|m| m.as_raw().as_slice());
    let mask_w = state.selection_mask.as_ref().map_or(0, |m| m.width() as usize);
    let mask_h = state.selection_mask.as_ref().map_or(0, |m| m.height() as usize);

    dst_raw.par_chunks_mut(stride).enumerate().for_each(|(y, row_out)| {
        let row_in = &src_raw[y * stride..(y + 1) * stride];
        for x in 0..w {
            let pi = x * 4;
            if let Some(mr) = mask_raw {
                if x < mask_w && y < mask_h && mr[y * mask_w + x] == 0 {
                    row_out[pi..pi + 4].copy_from_slice(&row_in[pi..pi + 4]);
                    continue;
                }
            }
            let r = row_in[pi] as f32;
            let g = row_in[pi + 1] as f32;
            let b = row_in[pi + 2] as f32;
            let a = row_in[pi + 3] as f32;
            let (nr, ng, nb, na) = transform(r, g, b, a);
            row_out[pi]     = nr.round().clamp(0.0, 255.0) as u8;
            row_out[pi + 1] = ng.round().clamp(0.0, 255.0) as u8;
            row_out[pi + 2] = nb.round().clamp(0.0, 255.0) as u8;
            row_out[pi + 3] = na.round().clamp(0.0, 255.0) as u8;
        }
    });

    let out = RgbaImage::from_raw(w as u32, h as u32, dst_raw).unwrap();
    let layer = &mut state.layers[layer_idx];
    layer.pixels = TiledImage::from_rgba_image(&out);
    state.mark_dirty(None);
}

// ============================================================================
// INSTANT OPERATIONS (no dialog needed)
// ============================================================================

/// Invert all color channels (R, G, B). Alpha is preserved.
pub fn invert_colors(state: &mut CanvasState, layer_idx: usize) {
    apply_pixel_transform(state, layer_idx, |r, g, b, a| {
        (255.0 - r, 255.0 - g, 255.0 - b, a)
    });
}

/// Invert the alpha channel. RGB is preserved.
pub fn invert_alpha(state: &mut CanvasState, layer_idx: usize) {
    apply_pixel_transform(state, layer_idx, |r, g, b, a| {
        (r, g, b, 255.0 - a)
    });
}

/// Apply a sepia tone effect.
pub fn sepia(state: &mut CanvasState, layer_idx: usize) {
    apply_pixel_transform(state, layer_idx, |r, g, b, a| {
        let sr = 0.393 * r + 0.769 * g + 0.189 * b;
        let sg = 0.349 * r + 0.686 * g + 0.168 * b;
        let sb = 0.272 * r + 0.534 * g + 0.131 * b;
        (sr.min(255.0), sg.min(255.0), sb.min(255.0), a)
    });
}

/// Auto Levels: stretches the tonal range of each channel independently
/// so min → 0 and max → 255, improving contrast.
pub fn auto_levels(state: &mut CanvasState, layer_idx: usize) {
    if layer_idx >= state.layers.len() { return; }
    let layer = &state.layers[layer_idx];
    let flat = layer.pixels.to_rgba_image();
    let src_raw = flat.as_raw();
    let w = flat.width() as usize;
    let h = flat.height() as usize;
    if w == 0 || h == 0 { return; }

    let mask_raw = state.selection_mask.as_ref().map(|m| m.as_raw().as_slice());
    let mask_w = state.selection_mask.as_ref().map_or(0, |m| m.width() as usize);
    let mask_h = state.selection_mask.as_ref().map_or(0, |m| m.height() as usize);

    // Compute per-channel min/max (only for selected pixels)
    let mut min_r = 255u8; let mut max_r = 0u8;
    let mut min_g = 255u8; let mut max_g = 0u8;
    let mut min_b = 255u8; let mut max_b = 0u8;

    for y in 0..h {
        for x in 0..w {
            if let Some(mr) = mask_raw {
                if x < mask_w && y < mask_h && mr[y * mask_w + x] == 0 {
                    continue;
                }
            }
            let pi = (y * w + x) * 4;
            let a = src_raw[pi + 3];
            if a == 0 { continue; } // skip fully transparent
            min_r = min_r.min(src_raw[pi]);     max_r = max_r.max(src_raw[pi]);
            min_g = min_g.min(src_raw[pi + 1]); max_g = max_g.max(src_raw[pi + 1]);
            min_b = min_b.min(src_raw[pi + 2]); max_b = max_b.max(src_raw[pi + 2]);
        }
    }

    // Build lookup tables
    let lut_r = build_stretch_lut(min_r, max_r);
    let lut_g = build_stretch_lut(min_g, max_g);
    let lut_b = build_stretch_lut(min_b, max_b);

    // Apply via the flat transform
    let mut dst_raw = vec![0u8; w * h * 4];
    let stride = w * 4;

    dst_raw.par_chunks_mut(stride).enumerate().for_each(|(y, row_out)| {
        let row_in = &src_raw[y * stride..(y + 1) * stride];
        for x in 0..w {
            let pi = x * 4;
            if let Some(mr) = mask_raw {
                if x < mask_w && y < mask_h && mr[y * mask_w + x] == 0 {
                    row_out[pi..pi + 4].copy_from_slice(&row_in[pi..pi + 4]);
                    continue;
                }
            }
            row_out[pi]     = lut_r[row_in[pi] as usize];
            row_out[pi + 1] = lut_g[row_in[pi + 1] as usize];
            row_out[pi + 2] = lut_b[row_in[pi + 2] as usize];
            row_out[pi + 3] = row_in[pi + 3];
        }
    });

    let out = RgbaImage::from_raw(w as u32, h as u32, dst_raw).unwrap();
    let layer = &mut state.layers[layer_idx];
    layer.pixels = TiledImage::from_rgba_image(&out);
    state.mark_dirty(None);
}

fn build_stretch_lut(min: u8, max: u8) -> [u8; 256] {
    let mut lut = [0u8; 256];
    if max <= min {
        // No range to stretch — identity
        for i in 0..256 { lut[i] = i as u8; }
    } else {
        let range = (max - min) as f32;
        for i in 0..256 {
            let v = if (i as u8) <= min { 0.0 }
                    else if (i as u8) >= max { 255.0 }
                    else { (i as f32 - min as f32) / range * 255.0 };
            lut[i] = v.round().clamp(0.0, 255.0) as u8;
        }
    }
    lut
}

// ============================================================================
// PARAMETERIZED OPERATIONS (used by dialogs with live preview)
// ============================================================================

/// Brightness/Contrast adjustment.
/// `brightness`: -100..100 (additive offset)
/// `contrast`: -100..100 (multiplier around midpoint)
pub fn brightness_contrast(
    state: &mut CanvasState,
    layer_idx: usize,
    brightness: f32,
    contrast: f32,
) {
    let factor = (259.0 * (contrast + 255.0)) / (255.0 * (259.0 - contrast));
    apply_pixel_transform(state, layer_idx, move |r, g, b, a| {
        let nr = factor * (r + brightness - 128.0) + 128.0;
        let ng = factor * (g + brightness - 128.0) + 128.0;
        let nb = factor * (b + brightness - 128.0) + 128.0;
        (nr, ng, nb, a)
    });
}

pub fn brightness_contrast_from_flat(
    state: &mut CanvasState,
    layer_idx: usize,
    brightness: f32,
    contrast: f32,
    original_flat: &RgbaImage,
) {
    let factor = (259.0 * (contrast + 255.0)) / (255.0 * (259.0 - contrast));
    apply_pixel_transform_from_flat(state, layer_idx, original_flat, move |r, g, b, a| {
        let nr = factor * (r + brightness - 128.0) + 128.0;
        let ng = factor * (g + brightness - 128.0) + 128.0;
        let nb = factor * (b + brightness - 128.0) + 128.0;
        (nr, ng, nb, a)
    });
}

/// Hue/Saturation/Lightness adjustment.
/// `hue_shift`: -180..180 degrees
/// `saturation`: -100..100 (0 = no change)
/// `lightness`: -100..100 (0 = no change)
pub fn hue_saturation_lightness(
    state: &mut CanvasState,
    layer_idx: usize,
    hue_shift: f32,
    saturation: f32,
    lightness: f32,
) {
    let sat_factor = 1.0 + saturation / 100.0;
    let light_offset = lightness * 255.0 / 100.0;
    apply_pixel_transform(state, layer_idx, move |r, g, b, a| {
        let (h, s, l) = rgb_to_hsl(r / 255.0, g / 255.0, b / 255.0);
        let nh = (h + hue_shift / 360.0).fract();
        let nh = if nh < 0.0 { nh + 1.0 } else { nh };
        let ns = (s * sat_factor).clamp(0.0, 1.0);
        let (nr, ng, nb) = hsl_to_rgb(nh, ns, l);
        (nr * 255.0 + light_offset, ng * 255.0 + light_offset, nb * 255.0 + light_offset, a)
    });
}

pub fn hue_saturation_lightness_from_flat(
    state: &mut CanvasState,
    layer_idx: usize,
    hue_shift: f32,
    saturation: f32,
    lightness: f32,
    original_flat: &RgbaImage,
) {
    let sat_factor = 1.0 + saturation / 100.0;
    let light_offset = lightness * 255.0 / 100.0;
    apply_pixel_transform_from_flat(state, layer_idx, original_flat, move |r, g, b, a| {
        let (h, s, l) = rgb_to_hsl(r / 255.0, g / 255.0, b / 255.0);
        let nh = (h + hue_shift / 360.0).fract();
        let nh = if nh < 0.0 { nh + 1.0 } else { nh };
        let ns = (s * sat_factor).clamp(0.0, 1.0);
        let (nr, ng, nb) = hsl_to_rgb(nh, ns, l);
        (nr * 255.0 + light_offset, ng * 255.0 + light_offset, nb * 255.0 + light_offset, a)
    });
}

/// Exposure adjustment (simulated).
/// `exposure`: -5.0..5.0 (EV stops, where 0 = no change)
/// Applies a simple gain: pixel * 2^exposure
pub fn exposure_adjust(
    state: &mut CanvasState,
    layer_idx: usize,
    exposure: f32,
) {
    let gain = 2.0f32.powf(exposure);
    apply_pixel_transform(state, layer_idx, move |r, g, b, a| {
        (r * gain, g * gain, b * gain, a)
    });
}

pub fn exposure_from_flat(
    state: &mut CanvasState,
    layer_idx: usize,
    exposure: f32,
    original_flat: &RgbaImage,
) {
    let gain = 2.0f32.powf(exposure);
    apply_pixel_transform_from_flat(state, layer_idx, original_flat, move |r, g, b, a| {
        (r * gain, g * gain, b * gain, a)
    });
}

/// Highlights/Shadows adjustment.
/// `shadows`: -100..100 (positive = brighten shadows)
/// `highlights`: -100..100 (positive = brighten highlights)
pub fn highlights_shadows(
    state: &mut CanvasState,
    layer_idx: usize,
    shadows: f32,
    highlights: f32,
) {
    let shadow_amt = shadows / 100.0;
    let highlight_amt = highlights / 100.0;
    apply_pixel_transform(state, layer_idx, move |r, g, b, a| {
        apply_hs_pixel(r, g, b, a, shadow_amt, highlight_amt)
    });
}

pub fn highlights_shadows_from_flat(
    state: &mut CanvasState,
    layer_idx: usize,
    shadows: f32,
    highlights: f32,
    original_flat: &RgbaImage,
) {
    let shadow_amt = shadows / 100.0;
    let highlight_amt = highlights / 100.0;
    apply_pixel_transform_from_flat(state, layer_idx, original_flat, move |r, g, b, a| {
        apply_hs_pixel(r, g, b, a, shadow_amt, highlight_amt)
    });
}

fn apply_hs_pixel(r: f32, g: f32, b: f32, a: f32, shadow_amt: f32, highlight_amt: f32) -> (f32, f32, f32, f32) {
    let lum = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0; // 0..1
    // Shadow weight: strong for dark pixels, falls off for bright
    let sw = (1.0 - lum).powi(2);
    // Highlight weight: strong for bright pixels
    let hw = lum.powi(2);
    let adjustment = sw * shadow_amt * 128.0 + hw * highlight_amt * 128.0;
    (r + adjustment, g + adjustment, b + adjustment, a)
}

/// Levels adjustment.
/// `input_black`: 0..255 (input shadow point)
/// `input_white`: 0..255 (input highlight point)
/// `gamma`: 0.1..10.0 (midtone adjustment, 1.0 = linear)
/// `output_black`: 0..255
/// `output_white`: 0..255
pub fn levels_adjust(
    state: &mut CanvasState,
    layer_idx: usize,
    input_black: f32,
    input_white: f32,
    gamma: f32,
    output_black: f32,
    output_white: f32,
) {
    let lut = build_levels_lut(input_black, input_white, gamma, output_black, output_white);
    apply_pixel_transform(state, layer_idx, move |r, g, b, a| {
        (lut[r as usize] as f32, lut[g as usize] as f32, lut[b as usize] as f32, a)
    });
}

pub fn levels_from_flat(
    state: &mut CanvasState,
    layer_idx: usize,
    input_black: f32,
    input_white: f32,
    gamma: f32,
    output_black: f32,
    output_white: f32,
    original_flat: &RgbaImage,
) {
    let lut = build_levels_lut(input_black, input_white, gamma, output_black, output_white);
    apply_pixel_transform_from_flat(state, layer_idx, original_flat, move |r, g, b, a| {
        (lut[r as usize] as f32, lut[g as usize] as f32, lut[b as usize] as f32, a)
    });
}

fn build_levels_lut(in_black: f32, in_white: f32, gamma: f32, out_black: f32, out_white: f32) -> [u8; 256] {
    let mut lut = [0u8; 256];
    let in_range = (in_white - in_black).max(1.0);
    let out_range = out_white - out_black;
    let inv_gamma = 1.0 / gamma.max(0.01);

    for i in 0..256 {
        let v = i as f32;
        // Remap input range
        let normalized = ((v - in_black) / in_range).clamp(0.0, 1.0);
        // Apply gamma
        let gamma_corrected = normalized.powf(inv_gamma);
        // Map to output range
        let output = out_black + gamma_corrected * out_range;
        lut[i] = output.round().clamp(0.0, 255.0) as u8;
    }
    lut
}

/// Levels adjustment with per-channel R/G/B support on top of a master adjustment.
/// Each channel tuple: (in_black, in_white, gamma, out_black, out_white).
/// Master LUT is applied first to all R/G/B channels, then per-channel LUTs are applied on top.
pub fn levels_from_flat_per_channel(
    state: &mut CanvasState,
    layer_idx: usize,
    master: (f32, f32, f32, f32, f32),
    r_ch: (f32, f32, f32, f32, f32),
    g_ch: (f32, f32, f32, f32, f32),
    b_ch: (f32, f32, f32, f32, f32),
    original_flat: &RgbaImage,
) {
    let lut_m = build_levels_lut(master.0, master.1, master.2, master.3, master.4);
    let lut_r = build_levels_lut(r_ch.0, r_ch.1, r_ch.2, r_ch.3, r_ch.4);
    let lut_g = build_levels_lut(g_ch.0, g_ch.1, g_ch.2, g_ch.3, g_ch.4);
    let lut_b = build_levels_lut(b_ch.0, b_ch.1, b_ch.2, b_ch.3, b_ch.4);
    apply_pixel_transform_from_flat(state, layer_idx, original_flat, move |r, g, b, a| {
        let r2 = lut_r[lut_m[r as usize] as usize] as f32;
        let g2 = lut_g[lut_m[g as usize] as usize] as f32;
        let b2 = lut_b[lut_m[b as usize] as usize] as f32;
        (r2, g2, b2, a)
    });
}

/// Temperature/Tint adjustment.
/// `temperature`: -100..100 (negative = cooler/blue, positive = warmer/yellow)
/// `tint`: -100..100 (negative = green, positive = magenta)
pub fn temperature_tint(
    state: &mut CanvasState,
    layer_idx: usize,
    temperature: f32,
    tint: f32,
) {
    let temp_shift = temperature * 1.5;
    let tint_shift = tint * 1.0;
    apply_pixel_transform(state, layer_idx, move |r, g, b, a| {
        let nr = r + temp_shift;       // warm adds red
        let ng = g - tint_shift * 0.5; // tint: green vs magenta
        let nb = b - temp_shift;       // warm removes blue
        (nr, ng, nb, a)
    });
}

pub fn temperature_tint_from_flat(
    state: &mut CanvasState,
    layer_idx: usize,
    temperature: f32,
    tint: f32,
    original_flat: &RgbaImage,
) {
    let temp_shift = temperature * 1.5;
    let tint_shift = tint * 1.0;
    apply_pixel_transform_from_flat(state, layer_idx, original_flat, move |r, g, b, a| {
        let nr = r + temp_shift;
        let ng = g - tint_shift * 0.5;
        let nb = b - temp_shift;
        (nr, ng, nb, a)
    });
}

/// Multi-channel curves adjustment.
/// `channel_points`: slice of 5 entries: [RGB, R, G, B, A].
/// Each entry is `(points, enabled)`.  The RGB LUT is applied to all three
/// channels, then per-channel LUTs are layered on top.
pub fn curves_adjust_multi(
    state: &mut CanvasState,
    layer_idx: usize,
    channel_points: &[(&[(f32, f32)], bool); 5],
) {
    let luts = build_multi_channel_luts(channel_points);
    apply_pixel_transform(state, layer_idx, move |r, g, b, a| {
        (luts[0][r as usize] as f32, luts[1][g as usize] as f32, luts[2][b as usize] as f32, luts[3][a as usize] as f32)
    });
}

pub fn curves_from_flat_multi(
    state: &mut CanvasState,
    layer_idx: usize,
    channel_points: &[(&[(f32, f32)], bool); 5],
    original_flat: &RgbaImage,
) {
    let luts = build_multi_channel_luts(channel_points);
    apply_pixel_transform_from_flat(state, layer_idx, original_flat, move |r, g, b, a| {
        (luts[0][r as usize] as f32, luts[1][g as usize] as f32, luts[2][b as usize] as f32, luts[3][a as usize] as f32)
    });
}

/// Build composite LUTs for R, G, B, A from [RGB, R, G, B, A] channel data.
/// The RGB curve is applied first, then per-channel curves are composed on top.
fn build_multi_channel_luts(channel_points: &[(&[(f32, f32)], bool); 5]) -> [[u8; 256]; 4] {
    // Identity LUTs
    let mut identity = [0u8; 256];
    for i in 0..256 { identity[i] = i as u8; }

    // RGB master curve (index 0)
    let rgb_lut = if channel_points[0].1 { build_curves_lut(channel_points[0].0) } else { identity };
    // Per-channel curves
    let r_lut = if channel_points[1].1 { build_curves_lut(channel_points[1].0) } else { identity };
    let g_lut = if channel_points[2].1 { build_curves_lut(channel_points[2].0) } else { identity };
    let b_lut = if channel_points[3].1 { build_curves_lut(channel_points[3].0) } else { identity };
    let a_lut = if channel_points[4].1 { build_curves_lut(channel_points[4].0) } else { identity };

    // Compose: per-channel applied after RGB master
    let mut final_r = [0u8; 256];
    let mut final_g = [0u8; 256];
    let mut final_b = [0u8; 256];
    let mut final_a = [0u8; 256];
    for i in 0..256 {
        final_r[i] = r_lut[rgb_lut[i] as usize];
        final_g[i] = g_lut[rgb_lut[i] as usize];
        final_b[i] = b_lut[rgb_lut[i] as usize];
        final_a[i] = a_lut[i]; // Alpha is independent of RGB master
    }
    [final_r, final_g, final_b, final_a]
}

/// Public wrapper for dialog visualization.
pub fn build_curves_lut_pub(points: &[(f32, f32)]) -> [u8; 256] {
    build_curves_lut(points)
}

/// Build a 256-entry lookup table from curve control points using
/// monotone cubic (Fritsch-Carlson) interpolation.
fn build_curves_lut(points: &[(f32, f32)]) -> [u8; 256] {
    let mut lut = [0u8; 256];
    if points.len() < 2 {
        // Identity
        for i in 0..256 { lut[i] = i as u8; }
        return lut;
    }

    let n = points.len();
    // Compute slopes
    let mut delta: Vec<f32> = Vec::with_capacity(n - 1);
    for i in 0..n - 1 {
        let dx = points[i + 1].0 - points[i].0;
        let dy = points[i + 1].1 - points[i].1;
        delta.push(if dx.abs() < 1e-6 { 0.0 } else { dy / dx });
    }

    // Compute tangents (Fritsch-Carlson monotone)
    let mut m = vec![0.0f32; n];
    m[0] = delta[0];
    m[n - 1] = delta[n - 2];
    for i in 1..n - 1 {
        if delta[i - 1] * delta[i] <= 0.0 {
            m[i] = 0.0;
        } else {
            m[i] = (delta[i - 1] + delta[i]) / 2.0;
        }
    }

    // Ensure monotonicity
    for i in 0..n - 1 {
        if delta[i].abs() < 1e-6 {
            m[i] = 0.0;
            m[i + 1] = 0.0;
        } else {
            let alpha = m[i] / delta[i];
            let beta = m[i + 1] / delta[i];
            let s = alpha * alpha + beta * beta;
            if s > 9.0 {
                let tau = 3.0 / s.sqrt();
                m[i] = tau * alpha * delta[i];
                m[i + 1] = tau * beta * delta[i];
            }
        }
    }

    // Evaluate spline for each input value
    for i in 0..256 {
        let x = i as f32;
        // Find segment
        let seg = {
            let mut s = 0;
            for j in 0..n - 1 {
                if x >= points[j].0 { s = j; }
            }
            s
        };

        if x <= points[0].0 {
            lut[i] = points[0].1.round().clamp(0.0, 255.0) as u8;
        } else if x >= points[n - 1].0 {
            lut[i] = points[n - 1].1.round().clamp(0.0, 255.0) as u8;
        } else {
            let x0 = points[seg].0;
            let x1 = points[seg + 1].0;
            let y0 = points[seg].1;
            let y1 = points[seg + 1].1;
            let h = x1 - x0;
            if h.abs() < 1e-6 {
                lut[i] = y0.round().clamp(0.0, 255.0) as u8;
            } else {
                let t = (x - x0) / h;
                let t2 = t * t;
                let t3 = t2 * t;
                // Hermite basis
                let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
                let h10 = t3 - 2.0 * t2 + t;
                let h01 = -2.0 * t3 + 3.0 * t2;
                let h11 = t3 - t2;
                let val = h00 * y0 + h10 * h * m[seg] + h01 * y1 + h11 * h * m[seg + 1];
                lut[i] = val.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    lut
}

// ============================================================================
// CROP TO SELECTION (Image-level operation affecting all layers)
// ============================================================================

/// Crop the canvas to the bounding rectangle of the selection mask.
/// Affects all layers. Clears the selection after cropping.
pub fn crop_to_selection(state: &mut CanvasState) {
    let mask = match state.selection_mask.as_ref() {
        Some(m) => m,
        None => return,
    };

    let (mw, mh) = (mask.width(), mask.height());
    let mask_raw = mask.as_raw();

    // Find bounding box of selected pixels
    let mut min_x = mw;
    let mut min_y = mh;
    let mut max_x = 0u32;
    let mut max_y = 0u32;
    for y in 0..mh {
        let row = y as usize * mw as usize;
        for x in 0..mw {
            if mask_raw[row + x as usize] > 0 {
                min_x = min_x.min(x);
                min_y = min_y.min(y);
                max_x = max_x.max(x);
                max_y = max_y.max(y);
            }
        }
    }

    if min_x > max_x || min_y > max_y { return; } // empty selection

    let new_w = max_x - min_x + 1;
    let new_h = max_y - min_y + 1;
    if new_w == 0 || new_h == 0 { return; }

    // Crop each layer
    for layer in &mut state.layers {
        let flat = layer.pixels.to_rgba_image();
        let cropped = image::imageops::crop_imm(&flat, min_x, min_y, new_w, new_h).to_image();
        layer.pixels = TiledImage::from_rgba_image(&cropped);
    }

    state.width = new_w;
    state.height = new_h;
    state.selection_mask = None;
    state.invalidate_selection_overlay();
    state.composite_cache = None;
    state.mark_dirty(None);
}

// ============================================================================
// LAYER OPERATIONS
// ============================================================================

/// Move the active layer to the top of the stack.
pub fn move_layer_to_top(state: &mut CanvasState) {
    let idx = state.active_layer_index;
    let last = state.layers.len() - 1;
    if idx >= last { return; }
    let layer = state.layers.remove(idx);
    state.layers.push(layer);
    state.active_layer_index = last;
    state.mark_dirty(None);
}

/// Move the active layer to the bottom of the stack.
pub fn move_layer_to_bottom(state: &mut CanvasState) {
    let idx = state.active_layer_index;
    if idx == 0 { return; }
    let layer = state.layers.remove(idx);
    state.layers.insert(0, layer);
    state.active_layer_index = 0;
    state.mark_dirty(None);
}

/// Move the active layer up one position.
pub fn move_layer_up(state: &mut CanvasState) {
    let idx = state.active_layer_index;
    if idx >= state.layers.len() - 1 { return; }
    state.layers.swap(idx, idx + 1);
    state.active_layer_index = idx + 1;
    state.mark_dirty(None);
}

/// Move the active layer down one position.
pub fn move_layer_down(state: &mut CanvasState) {
    let idx = state.active_layer_index;
    if idx == 0 { return; }
    state.layers.swap(idx, idx - 1);
    state.active_layer_index = idx - 1;
    state.mark_dirty(None);
}

/// Import an image file as a new layer.
/// The image is placed at top-left and cropped/padded to canvas size.
pub fn import_layer_from_image(state: &mut CanvasState, img: &RgbaImage, name: &str) {
    let cw = state.width;
    let ch = state.height;
    let mut layer = crate::canvas::Layer::new(
        name.to_string(), cw, ch, image::Rgba([0, 0, 0, 0]),
    );

    // Copy pixels from source (cropped to canvas size, centered)
    let iw = img.width();
    let ih = img.height();
    // Center the imported image on the canvas
    let ox = (cw as i32 - iw as i32) / 2;
    let oy = (ch as i32 - ih as i32) / 2;

    for sy in 0..ih {
        let dy = oy + sy as i32;
        if dy < 0 || dy >= ch as i32 { continue; }
        for sx in 0..iw {
            let dx = ox + sx as i32;
            if dx < 0 || dx >= cw as i32 { continue; }
            let pixel = *img.get_pixel(sx, sy);
            if pixel[3] > 0 {
                layer.pixels.put_pixel(dx as u32, dy as u32, pixel);
            }
        }
    }

    let new_idx = state.active_layer_index + 1;
    state.layers.insert(new_idx, layer);
    state.active_layer_index = new_idx;
    state.mark_dirty(None);
}

// ============================================================================
// HISTOGRAM COMPUTATION (for Levels dialog)
// ============================================================================

/// Compute per-channel histograms (R, G, B, Luminance).
/// Each histogram has 256 bins with counts.
/// Only counts pixels that are selected (if mask exists).
pub fn compute_histogram(state: &CanvasState, layer_idx: usize) -> ([u32; 256], [u32; 256], [u32; 256], [u32; 256]) {
    let mut hist_r = [0u32; 256];
    let mut hist_g = [0u32; 256];
    let mut hist_b = [0u32; 256];
    let mut hist_l = [0u32; 256];

    if layer_idx >= state.layers.len() { return (hist_r, hist_g, hist_b, hist_l); }
    let layer = &state.layers[layer_idx];
    let flat = layer.pixels.to_rgba_image();
    let src_raw = flat.as_raw();
    let w = flat.width() as usize;
    let h = flat.height() as usize;

    let mask_raw = state.selection_mask.as_ref().map(|m| m.as_raw().as_slice());
    let mask_w = state.selection_mask.as_ref().map_or(0, |m| m.width() as usize);
    let mask_h = state.selection_mask.as_ref().map_or(0, |m| m.height() as usize);

    for y in 0..h {
        for x in 0..w {
            if let Some(mr) = mask_raw {
                if x < mask_w && y < mask_h && mr[y * mask_w + x] == 0 { continue; }
            }
            let pi = (y * w + x) * 4;
            let r = src_raw[pi];
            let g = src_raw[pi + 1];
            let b = src_raw[pi + 2];
            let a = src_raw[pi + 3];
            if a == 0 { continue; }
            let lum = (0.2126 * r as f32 + 0.7152 * g as f32 + 0.0722 * b as f32).round() as usize;
            hist_r[r as usize] += 1;
            hist_g[g as usize] += 1;
            hist_b[b as usize] += 1;
            hist_l[lum.min(255)] += 1;
        }
    }

    (hist_r, hist_g, hist_b, hist_l)
}

// ============================================================================
// COLOR SPACE HELPERS
// ============================================================================

/// RGB (0..1) → HSL (H: 0..1, S: 0..1, L: 0..1)
pub fn rgb_to_hsl(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let l = (max + min) / 2.0;

    if (max - min).abs() < 1e-6 {
        return (0.0, 0.0, l);
    }

    let d = max - min;
    let s = if l > 0.5 { d / (2.0 - max - min) } else { d / (max + min) };

    let h = if (max - r).abs() < 1e-6 {
        let mut h = (g - b) / d;
        if h < 0.0 { h += 6.0; }
        h / 6.0
    } else if (max - g).abs() < 1e-6 {
        ((b - r) / d + 2.0) / 6.0
    } else {
        ((r - g) / d + 4.0) / 6.0
    };

    (h, s, l)
}

/// HSL (H: 0..1, S: 0..1, L: 0..1) → RGB (0..1)
pub fn hsl_to_rgb(h: f32, s: f32, l: f32) -> (f32, f32, f32) {
    if s.abs() < 1e-6 {
        return (l, l, l);
    }

    let q = if l < 0.5 { l * (1.0 + s) } else { l + s - l * s };
    let p = 2.0 * l - q;

    let r = hue_to_rgb(p, q, h + 1.0 / 3.0);
    let g = hue_to_rgb(p, q, h);
    let b = hue_to_rgb(p, q, h - 1.0 / 3.0);

    (r, g, b)
}

pub fn hue_to_rgb(p: f32, q: f32, mut t: f32) -> f32 {
    if t < 0.0 { t += 1.0; }
    if t > 1.0 { t -= 1.0; }
    if t < 1.0 / 6.0 { return p + (q - p) * 6.0 * t; }
    if t < 1.0 / 2.0 { return q; }
    if t < 2.0 / 3.0 { return p + (q - p) * (2.0 / 3.0 - t) * 6.0; }
    p
}

// ============================================================================
// GPU-ACCELERATED VARIANTS
// ============================================================================

/// GPU-accelerated brightness/contrast.  Falls back to CPU if selection exists.
pub fn brightness_contrast_gpu(
    state: &mut CanvasState,
    layer_idx: usize,
    brightness: f32,
    contrast: f32,
    gpu: &crate::gpu::GpuRenderer,
) {
    if layer_idx >= state.layers.len() { return; }
    if state.selection_mask.is_some() {
        // CPU fallback for selection-masked adjustment
        brightness_contrast(state, layer_idx, brightness, contrast);
        return;
    }
    let layer = &state.layers[layer_idx];
    let flat = layer.pixels.to_rgba_image();
    let (w, h) = (flat.width(), flat.height());
    let result_data = gpu.brightness_contrast_rgba(flat.as_raw(), w, h, brightness, contrast);
    let result = RgbaImage::from_raw(w, h, result_data).unwrap();
    let layer = &mut state.layers[layer_idx];
    layer.pixels = TiledImage::from_rgba_image(&result);
    state.mark_dirty(None);
}

pub fn brightness_contrast_from_flat_gpu(
    state: &mut CanvasState,
    layer_idx: usize,
    brightness: f32,
    contrast: f32,
    original_flat: &RgbaImage,
    gpu: &crate::gpu::GpuRenderer,
) {
    if layer_idx >= state.layers.len() { return; }
    if state.selection_mask.is_some() {
        brightness_contrast_from_flat(state, layer_idx, brightness, contrast, original_flat);
        return;
    }
    let (w, h) = (original_flat.width(), original_flat.height());
    let result_data = gpu.brightness_contrast_rgba(original_flat.as_raw(), w, h, brightness, contrast);
    let result = RgbaImage::from_raw(w, h, result_data).unwrap();
    let layer = &mut state.layers[layer_idx];
    layer.pixels = TiledImage::from_rgba_image(&result);
    state.mark_dirty(None);
}

/// GPU-accelerated hue/saturation/lightness.
pub fn hue_saturation_lightness_gpu(
    state: &mut CanvasState,
    layer_idx: usize,
    hue_shift: f32,
    saturation: f32,
    lightness: f32,
    gpu: &crate::gpu::GpuRenderer,
) {
    if layer_idx >= state.layers.len() { return; }
    if state.selection_mask.is_some() {
        hue_saturation_lightness(state, layer_idx, hue_shift, saturation, lightness);
        return;
    }
    let layer = &state.layers[layer_idx];
    let flat = layer.pixels.to_rgba_image();
    let (w, h) = (flat.width(), flat.height());
    let result_data = gpu.hsl_rgba(flat.as_raw(), w, h, hue_shift, saturation, lightness);
    let result = RgbaImage::from_raw(w, h, result_data).unwrap();
    let layer = &mut state.layers[layer_idx];
    layer.pixels = TiledImage::from_rgba_image(&result);
    state.mark_dirty(None);
}

pub fn hue_saturation_lightness_from_flat_gpu(
    state: &mut CanvasState,
    layer_idx: usize,
    hue_shift: f32,
    saturation: f32,
    lightness: f32,
    original_flat: &RgbaImage,
    gpu: &crate::gpu::GpuRenderer,
) {
    if layer_idx >= state.layers.len() { return; }
    if state.selection_mask.is_some() {
        hue_saturation_lightness_from_flat(state, layer_idx, hue_shift, saturation, lightness, original_flat);
        return;
    }
    let (w, h) = (original_flat.width(), original_flat.height());
    let result_data = gpu.hsl_rgba(original_flat.as_raw(), w, h, hue_shift, saturation, lightness);
    let result = RgbaImage::from_raw(w, h, result_data).unwrap();
    let layer = &mut state.layers[layer_idx];
    layer.pixels = TiledImage::from_rgba_image(&result);
    state.mark_dirty(None);
}

/// GPU-accelerated color inversion.
pub fn invert_colors_gpu(
    state: &mut CanvasState,
    layer_idx: usize,
    gpu: &crate::gpu::GpuRenderer,
) {
    if layer_idx >= state.layers.len() { return; }
    if state.selection_mask.is_some() {
        invert_colors(state, layer_idx);
        return;
    }
    let layer = &state.layers[layer_idx];
    let flat = layer.pixels.to_rgba_image();
    let (w, h) = (flat.width(), flat.height());
    let result_data = gpu.invert_rgba(flat.as_raw(), w, h);
    let result = RgbaImage::from_raw(w, h, result_data).unwrap();
    let layer = &mut state.layers[layer_idx];
    layer.pixels = TiledImage::from_rgba_image(&result);
    state.mark_dirty(None);
}

// ============================================================================
// THRESHOLD
// ============================================================================

/// Converts the layer to black-and-white by threshold on luminance.
/// `level`: 0..255 — pixels brighter than this become white, others black.
pub fn threshold(state: &mut CanvasState, layer_idx: usize, level: f32) {
    apply_pixel_transform(state, layer_idx, move |r, g, b, a| {
        let lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        let v = if lum >= level { 255.0 } else { 0.0 };
        (v, v, v, a)
    });
}

pub fn threshold_from_flat(
    state: &mut CanvasState,
    layer_idx: usize,
    level: f32,
    original_flat: &RgbaImage,
) {
    apply_pixel_transform_from_flat(state, layer_idx, original_flat, move |r, g, b, a| {
        let lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        let v = if lum >= level { 255.0 } else { 0.0 };
        (v, v, v, a)
    });
}

// ============================================================================
// POSTERIZE
// ============================================================================

/// Reduces the number of tonal levels per channel.
/// `levels`: 2..=16
pub fn posterize(state: &mut CanvasState, layer_idx: usize, levels: u32) {
    let factor = levels.max(2) as f32;
    apply_pixel_transform(state, layer_idx, move |r, g, b, a| {
        let p = |v: f32| (v / 255.0 * (factor - 1.0)).round() / (factor - 1.0) * 255.0;
        (p(r), p(g), p(b), a)
    });
}

pub fn posterize_from_flat(
    state: &mut CanvasState,
    layer_idx: usize,
    levels: u32,
    original_flat: &RgbaImage,
) {
    let factor = levels.max(2) as f32;
    apply_pixel_transform_from_flat(state, layer_idx, original_flat, move |r, g, b, a| {
        let p = |v: f32| (v / 255.0 * (factor - 1.0)).round() / (factor - 1.0) * 255.0;
        (p(r), p(g), p(b), a)
    });
}

// ============================================================================
// COLOR BALANCE
// ============================================================================

/// Adjusts color balance in shadows, midtones, and highlights separately.
/// Each component is `[r, g, b]` shift in -100..100.
pub fn color_balance(
    state: &mut CanvasState,
    layer_idx: usize,
    shadows: [f32; 3],
    midtones: [f32; 3],
    highlights: [f32; 3],
) {
    apply_pixel_transform(state, layer_idx, move |r, g, b, a| {
        color_balance_pixel(r, g, b, a, shadows, midtones, highlights)
    });
}

pub fn color_balance_from_flat(
    state: &mut CanvasState,
    layer_idx: usize,
    shadows: [f32; 3],
    midtones: [f32; 3],
    highlights: [f32; 3],
    original_flat: &RgbaImage,
) {
    apply_pixel_transform_from_flat(state, layer_idx, original_flat, move |r, g, b, a| {
        color_balance_pixel(r, g, b, a, shadows, midtones, highlights)
    });
}

#[inline]
fn color_balance_pixel(r: f32, g: f32, b: f32, a: f32, shadows: [f32; 3], midtones: [f32; 3], highlights: [f32; 3]) -> (f32, f32, f32, f32) {
    let lum = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0;
    let sw = (1.0 - lum * 2.0).max(0.0).powi(2);
    let hw = (lum * 2.0 - 1.0).max(0.0).powi(2);
    let mw = (1.0 - sw - hw).max(0.0);
    let adj_r = sw * shadows[0] + mw * midtones[0] + hw * highlights[0];
    let adj_g = sw * shadows[1] + mw * midtones[1] + hw * highlights[1];
    let adj_b = sw * shadows[2] + mw * midtones[2] + hw * highlights[2];
    (r + adj_r * 1.28, g + adj_g * 1.28, b + adj_b * 1.28, a)
}

// ============================================================================
// GRADIENT MAP
// ============================================================================

/// Maps the luminance of each pixel to a color from a 256-entry RGBA LUT.
pub fn gradient_map(state: &mut CanvasState, layer_idx: usize, lut: &[[u8; 4]; 256]) {
    let lut_copy = *lut;
    apply_pixel_transform(state, layer_idx, move |r, g, b, a| {
        let lum = ((0.2126 * r + 0.7152 * g + 0.0722 * b) as usize).min(255);
        let [lr, lg, lb, _] = lut_copy[lum];
        (lr as f32, lg as f32, lb as f32, a)
    });
}

pub fn gradient_map_from_flat(
    state: &mut CanvasState,
    layer_idx: usize,
    lut: &[[u8; 4]; 256],
    original_flat: &RgbaImage,
) {
    let lut_copy = *lut;
    apply_pixel_transform_from_flat(state, layer_idx, original_flat, move |r, g, b, a| {
        let lum = ((0.2126 * r + 0.7152 * g + 0.0722 * b) as usize).min(255);
        let [lr, lg, lb, _] = lut_copy[lum];
        (lr as f32, lg as f32, lb as f32, a)
    });
}

// ============================================================================
// BLACK AND WHITE (CHANNEL MIX)
// ============================================================================

/// Converts to grayscale using weighted RGB channels.
/// Weights are percentages (100 = neutral), values 0..200.
pub fn black_and_white(
    state: &mut CanvasState,
    layer_idx: usize,
    r_weight: f32,
    g_weight: f32,
    b_weight: f32,
) {
    apply_pixel_transform(state, layer_idx, move |r, g, b, a| {
        let v = (r * r_weight + g * g_weight + b * b_weight) / 100.0;
        let v = v.clamp(0.0, 255.0);
        (v, v, v, a)
    });
}

pub fn black_and_white_from_flat(
    state: &mut CanvasState,
    layer_idx: usize,
    r_weight: f32,
    g_weight: f32,
    b_weight: f32,
    original_flat: &RgbaImage,
) {
    apply_pixel_transform_from_flat(state, layer_idx, original_flat, move |r, g, b, a| {
        let v = (r * r_weight + g * g_weight + b * b_weight) / 100.0;
        let v = v.clamp(0.0, 255.0);
        (v, v, v, a)
    });
}

// ============================================================================
// VIBRANCE
// ============================================================================

/// Boosts saturation selectively — less-saturated colors are boosted more.
/// `amount`: -100..100 (positive = boost, negative = reduce)
pub fn vibrance(state: &mut CanvasState, layer_idx: usize, amount: f32) {
    let v = amount / 100.0;
    apply_pixel_transform(state, layer_idx, move |r, g, b, a| {
        vibrance_pixel(r, g, b, a, v)
    });
}

pub fn vibrance_from_flat(
    state: &mut CanvasState,
    layer_idx: usize,
    amount: f32,
    original_flat: &RgbaImage,
) {
    let v = amount / 100.0;
    apply_pixel_transform_from_flat(state, layer_idx, original_flat, move |r, g, b, a| {
        vibrance_pixel(r, g, b, a, v)
    });
}

#[inline]
fn vibrance_pixel(r: f32, g: f32, b: f32, a: f32, v: f32) -> (f32, f32, f32, f32) {
    let rn = r / 255.0;
    let gn = g / 255.0;
    let bn = b / 255.0;
    let (h, s, l) = rgb_to_hsl(rn, gn, bn);
    let boost = if v >= 0.0 {
        v * (1.0 - s).powi(2)
    } else {
        v * s.powi(2)
    };
    let ns = (s + boost).clamp(0.0, 1.0);
    let (nr, ng, nb) = hsl_to_rgb(h, ns, l);
    (nr * 255.0, ng * 255.0, nb * 255.0, a)
}

// ============================================================================
// SELECTION MODIFICATION OPS
// ============================================================================

/// Feathers (blurs) the selection mask by `radius` pixels using a box blur approximation.
pub fn feather_selection(state: &mut CanvasState, radius: f32) {
    let mask = match state.selection_mask.take() {
        Some(m) => m,
        None => return,
    };
    let (w, h) = (mask.width(), mask.height());
    let passes = ((radius / 2.0) as u32).max(1);
    let r = (radius as usize).max(1);
    let mut data = mask.into_raw();

    for _ in 0..passes {
        // Horizontal pass
        let mut tmp = data.clone();
        for y in 0..h as usize {
            let row_start = y * w as usize;
            for x in 0..w as usize {
                let x0 = x.saturating_sub(r);
                let x1 = (x + r).min(w as usize - 1);
                let count = x1 - x0 + 1;
                let mut sum = 0u32;
                for xi in x0..=x1 { sum += data[row_start + xi] as u32; }
                tmp[row_start + x] = (sum / count as u32) as u8;
            }
        }
        // Vertical pass
        let mut out = tmp.clone();
        for x in 0..w as usize {
            for y in 0..h as usize {
                let y0 = y.saturating_sub(r);
                let y1 = (y + r).min(h as usize - 1);
                let count = y1 - y0 + 1;
                let mut sum = 0u32;
                for yi in y0..=y1 { sum += tmp[yi * w as usize + x] as u32; }
                out[y * w as usize + x] = (sum / count as u32) as u8;
            }
        }
        data = out;
    }

    state.selection_mask = image::GrayImage::from_raw(w, h, data);
    state.invalidate_selection_overlay();
    state.mark_dirty(None);
}

/// Expands (dilates) the selection mask by `radius` pixels.
pub fn expand_selection(state: &mut CanvasState, radius: i32) {
    let mask = match state.selection_mask.take() {
        Some(m) => m,
        None => return,
    };
    let (w, h) = (mask.width(), mask.height());
    let data = mask.into_raw();
    let r = radius.max(0) as usize;
    let mut out = data.clone();

    for y in 0..h as usize {
        for x in 0..w as usize {
            if data[y * w as usize + x] > 127 { continue; } // already selected
            let x0 = x.saturating_sub(r);
            let x1 = (x + r).min(w as usize - 1);
            let y0 = y.saturating_sub(r);
            let y1 = (y + r).min(h as usize - 1);
            let mut found = false;
            'outer: for yy in y0..=y1 {
                for xx in x0..=x1 {
                    let dx = xx as i32 - x as i32;
                    let dy = yy as i32 - y as i32;
                    if dx*dx + dy*dy <= (r as i32) * (r as i32) {
                        if data[yy * w as usize + xx] > 127 { found = true; break 'outer; }
                    }
                }
            }
            if found { out[y * w as usize + x] = 255; }
        }
    }

    state.selection_mask = image::GrayImage::from_raw(w, h, out);
    state.invalidate_selection_overlay();
    state.mark_dirty(None);
}

/// Contracts (erodes) the selection mask by `radius` pixels.
pub fn contract_selection(state: &mut CanvasState, radius: i32) {
    let mask = match state.selection_mask.take() {
        Some(m) => m,
        None => return,
    };
    let (w, h) = (mask.width(), mask.height());
    let data = mask.into_raw();
    let r = radius.max(0) as usize;
    let mut out = data.clone();

    for y in 0..h as usize {
        for x in 0..w as usize {
            if data[y * w as usize + x] == 0 { continue; } // not selected
            let x0 = x.saturating_sub(r);
            let x1 = (x + r).min(w as usize - 1);
            let y0 = y.saturating_sub(r);
            let y1 = (y + r).min(h as usize - 1);
            let mut all_selected = true;
            'outer2: for yy in y0..=y1 {
                for xx in x0..=x1 {
                    let dx = xx as i32 - x as i32;
                    let dy = yy as i32 - y as i32;
                    if dx*dx + dy*dy <= (r as i32) * (r as i32) {
                        if data[yy * w as usize + xx] == 0 { all_selected = false; break 'outer2; }
                    }
                }
            }
            if !all_selected { out[y * w as usize + x] = 0; }
        }
    }

    state.selection_mask = image::GrayImage::from_raw(w, h, out);
    state.invalidate_selection_overlay();
    state.mark_dirty(None);
}

// ============================================================================
// PER-BAND HUE/SATURATION
// ============================================================================

/// Adjustment values for a single hue band.
#[derive(Clone, Copy, Debug)]
pub struct HueBandAdjust {
    pub hue: f32,        // -180..180 degrees
    pub saturation: f32, // -100..100
    pub lightness: f32,  // -100..100
}

impl Default for HueBandAdjust {
    fn default() -> Self { Self { hue: 0.0, saturation: 0.0, lightness: 0.0 } }
}

/// Hue centers for the 6 color bands: Reds, Yellows, Greens, Cyans, Blues, Magentas
const BAND_CENTERS: [f32; 6] = [0.0, 60.0, 120.0, 180.0, 240.0, 300.0];

/// Blend weight of a pixel's hue into a given band.
/// Full weight within ±30°, smooth falloff to 0 at ±45°.
fn band_weight(pixel_hue_deg: f32, band_center_deg: f32) -> f32 {
    let mut dist = (pixel_hue_deg - band_center_deg).abs() % 360.0;
    if dist > 180.0 { dist = 360.0 - dist; }
    if dist <= 30.0 { 1.0 }
    else if dist < 45.0 { 1.0 - (dist - 30.0) / 15.0 }
    else { 0.0 }
}

/// Hue/Sat/Lightness with per-band control (6 bands: R/Y/G/C/B/M) plus global adjustments.
pub fn hue_saturation_per_band_from_flat(
    state: &mut CanvasState,
    layer_idx: usize,
    global_hue: f32,
    global_sat: f32,
    global_light: f32,
    bands: &[HueBandAdjust; 6],
    original_flat: &RgbaImage,
) {
    let g_sat = 1.0 + global_sat / 100.0;
    let g_light = global_light * 255.0 / 100.0;
    let bands = *bands;
    apply_pixel_transform_from_flat(state, layer_idx, original_flat, move |r, g, b, a| {
        let (h, s, l) = rgb_to_hsl(r / 255.0, g / 255.0, b / 255.0);
        let h_deg = h * 360.0;

        // Accumulate weighted band contributions on top of global
        let mut extra_hue = global_hue;
        let mut extra_sat_factor = g_sat;
        let mut extra_light = g_light;
        for i in 0..6 {
            let w = band_weight(h_deg, BAND_CENTERS[i]);
            if w > 0.0 {
                extra_hue += bands[i].hue * w;
                extra_sat_factor += bands[i].saturation / 100.0 * w;
                extra_light += bands[i].lightness * 255.0 / 100.0 * w;
            }
        }

        let nh = ((h + extra_hue / 360.0) % 1.0 + 1.0) % 1.0;
        let ns = (s * extra_sat_factor).clamp(0.0, 1.0);
        let (nr, ng, nb) = hsl_to_rgb(nh, ns, l);
        (
            nr * 255.0 + extra_light,
            ng * 255.0 + extra_light,
            nb * 255.0 + extra_light,
            a,
        )
    });
}
// ============================================================================
// COLOR RANGE SELECTION — select pixels by HSL hue/saturation proximity
// ============================================================================

/// Select pixels on the active layer whose hue (degrees) is within `hue_tolerance`
/// of `hue_center` and whose saturation fraction is ≥ `sat_min`.
/// `fuzziness` (0.0–1.0) controls how gradual/soft the selection edge is.
/// The resulting mask is applied using `mode`, which determines how it is merged
/// with any existing selection.
pub fn select_color_range(
    state: &mut CanvasState,
    hue_center_deg: f32,
    hue_tolerance_deg: f32,
    sat_min: f32,
    fuzziness: f32,
    mode: crate::canvas::SelectionMode,
) {
    use image::{GrayImage, Luma};

    let idx = state.active_layer_index;
    if idx >= state.layers.len() { return; }

    let w = state.width;
    let h = state.height;

    // Build the new selection mask from the active layer
    let layer = &state.layers[idx];
    let mut new_mask = GrayImage::new(w, h);
    let hue_center = hue_center_deg / 360.0;
    let hue_tol = (hue_tolerance_deg / 360.0).max(0.001);
    let fuzz = fuzziness.clamp(0.001, 1.0);

    for y in 0..h {
        for x in 0..w {
            let px = layer.pixels.get_pixel(x, y);
            // Skip fully transparent pixels
            if px[3] == 0 { continue; }
            let r = px[0] as f32 / 255.0;
            let g = px[1] as f32 / 255.0;
            let b = px[2] as f32 / 255.0;
            let (h_frac, s, _l) = rgb_to_hsl(r, g, b);
            // Saturation gate
            if s < sat_min { continue; }
            // Angular distance on the hue wheel
            let mut diff = (h_frac - hue_center).abs();
            if diff > 0.5 { diff = 1.0 - diff; }
            // Within tolerance?
            if diff > hue_tol { continue; }
            // Soft edge via fuzziness
            let weight = 1.0 - (diff / hue_tol).powf(1.0 / fuzz.max(0.01));
            let alpha = (weight * 255.0).clamp(0.0, 255.0) as u8;
            new_mask.put_pixel(x, y, Luma([alpha]));
        }
    }

    // Merge with existing selection using SelectionMode
    use crate::canvas::SelectionMode;
    let final_mask = match mode {
        SelectionMode::Replace => new_mask,
        SelectionMode::Add => {
            let mut base = state.selection_mask.clone().unwrap_or_else(|| GrayImage::new(w, h));
            for y in 0..h {
                for x in 0..w {
                    let a = new_mask.get_pixel(x, y).0[0];
                    let b = base.get_pixel(x, y).0[0];
                    base.put_pixel(x, y, Luma([a.max(b)]));
                }
            }
            base
        }
        SelectionMode::Subtract => {
            let mut base = state.selection_mask.clone().unwrap_or_else(|| GrayImage::new(w, h));
            for y in 0..h {
                for x in 0..w {
                    let a = new_mask.get_pixel(x, y).0[0];
                    let b = base.get_pixel(x, y).0[0];
                    base.put_pixel(x, y, Luma([b.saturating_sub(a)]));
                }
            }
            base
        }
        SelectionMode::Intersect => {
            let base = state.selection_mask.clone().unwrap_or_else(|| GrayImage::new(w, h));
            let mut intersect = GrayImage::new(w, h);
            for y in 0..h {
                for x in 0..w {
                    let a = new_mask.get_pixel(x, y).0[0];
                    let b = base.get_pixel(x, y).0[0];
                    intersect.put_pixel(x, y, Luma([((a as u16 * b as u16 / 255) as u8)]));
                }
            }
            intersect
        }
    };

    state.selection_mask = Some(final_mask);
    state.invalidate_selection_overlay();
    state.mark_dirty(None);
}