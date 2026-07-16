// ============================================================================
// EFFECTS ENGINE -- GPU-ready, rayon-parallelized image effects
// ============================================================================

use crate::canvas::{CanvasState, TiledImage};
use crate::par_compat::*;
use image::{GrayImage, Rgba, RgbaImage};

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

fn commit_to_layer(state: &mut CanvasState, layer_idx: usize, result: &RgbaImage) {
    if layer_idx >= state.layers.len() {
        return;
    }
    state.layers[layer_idx].pixels = TiledImage::from_rgba_image(result);
    state.mark_dirty(None);
}

#[inline]
fn sample_clamped(img: &RgbaImage, x: i32, y: i32) -> [f32; 4] {
    let cx = x.clamp(0, img.width() as i32 - 1) as u32;
    let cy = y.clamp(0, img.height() as i32 - 1) as u32;
    let p = img.get_pixel(cx, cy);
    [p[0] as f32, p[1] as f32, p[2] as f32, p[3] as f32]
}

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

#[inline]
fn hash_f32(x: u32, y: u32, seed: u32) -> f32 {
    let h = hash_u32(
        x.wrapping_mul(374761393)
            .wrapping_add(y.wrapping_mul(668265263))
            .wrapping_add(seed),
    );
    (h & 0x00FFFFFF) as f32 / 16777216.0
}

include!("effects/blur.rs");
include!("effects/distort.rs");
include!("effects/noise.rs");
include!("effects/stylize.rs");
include!("effects/render.rs");
include!("effects/glitch.rs");
include!("effects/artistic.rs");
include!("effects/contours.rs");
