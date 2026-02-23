// ============================================================================
// TRANSFORM OPERATIONS — flip, rotate, affine for images and layers
// ============================================================================

use image::{RgbaImage, Rgba, imageops};
use rayon::prelude::*;
use crate::canvas::{TiledImage, CanvasState, Layer};

/// Interpolation method for resize operations.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum Interpolation {
    Nearest,
    #[default]
    Bilinear,
    Bicubic,
    Lanczos3,
}

impl Interpolation {
    pub fn label(&self) -> String {
        match self {
            Interpolation::Nearest  => t!("interpolation.nearest"),
            Interpolation::Bilinear => t!("interpolation.bilinear"),
            Interpolation::Bicubic  => t!("interpolation.bicubic"),
            Interpolation::Lanczos3 => t!("interpolation.lanczos3"),
        }
    }

    pub fn all() -> &'static [Interpolation] {
        &[
            Interpolation::Nearest,
            Interpolation::Bilinear,
            Interpolation::Bicubic,
            Interpolation::Lanczos3,
        ]
    }

    pub fn to_filter(&self) -> imageops::FilterType {
        match self {
            Interpolation::Nearest  => imageops::FilterType::Nearest,
            Interpolation::Bilinear => imageops::FilterType::Triangle,
            Interpolation::Bicubic  => imageops::FilterType::CatmullRom,
            Interpolation::Lanczos3 => imageops::FilterType::Lanczos3,
        }
    }
}

// ---------------------------------------------------------------------------
//  Whole-canvas transforms (affect ALL layers)
// ---------------------------------------------------------------------------

/// Flip the entire canvas horizontally (mirror left↔right).
pub fn flip_canvas_horizontal(state: &mut CanvasState) {
    state.layers.par_iter_mut().for_each(|layer| {
        layer.pixels.flip_horizontal_chunked();
    });
    state.mark_dirty(None);
}

/// Flip the entire canvas vertically (mirror top↔bottom).
pub fn flip_canvas_vertical(state: &mut CanvasState) {
    state.layers.par_iter_mut().for_each(|layer| {
        layer.pixels.flip_vertical_chunked();
    });
    state.mark_dirty(None);
}

/// Rotate the entire canvas 90° clockwise (swaps W↔H).
pub fn rotate_canvas_90cw(state: &mut CanvasState) {
    let new_pixels: Vec<_> = state.layers.par_iter()
        .map(|layer| layer.pixels.rotate_90cw_chunked())
        .collect();
    for (layer, new_px) in state.layers.iter_mut().zip(new_pixels) {
        layer.pixels = new_px;
    }
    std::mem::swap(&mut state.width, &mut state.height);
    state.composite_cache = None;
    state.clear_preview_state();
    state.mark_dirty(None);
}

/// Rotate the entire canvas 90° counter-clockwise (swaps W↔H).
pub fn rotate_canvas_90ccw(state: &mut CanvasState) {
    let new_pixels: Vec<_> = state.layers.par_iter()
        .map(|layer| layer.pixels.rotate_90ccw_chunked())
        .collect();
    for (layer, new_px) in state.layers.iter_mut().zip(new_pixels) {
        layer.pixels = new_px;
    }
    std::mem::swap(&mut state.width, &mut state.height);
    state.composite_cache = None;
    state.clear_preview_state();
    state.mark_dirty(None);
}

/// Rotate the entire canvas 180°.
pub fn rotate_canvas_180(state: &mut CanvasState) {
    state.layers.par_iter_mut().for_each(|layer| {
        layer.pixels.rotate_180_chunked();
    });
    state.mark_dirty(None);
}

/// Resize the entire image (all layers) to new dimensions with given interpolation.
pub fn resize_image(state: &mut CanvasState, new_w: u32, new_h: u32, interp: Interpolation) {
    let filter = interp.to_filter();
    for layer in &mut state.layers {
        let flat = layer.pixels.to_rgba_image();
        let resized = imageops::resize(&flat, new_w, new_h, filter);
        layer.pixels = TiledImage::from_rgba_image(&resized);
    }
    state.width = new_w;
    state.height = new_h;
    state.composite_cache = None;
    state.clear_preview_state();
    state.mark_dirty(None);
}

/// Resize the canvas (change dimensions), placing the old content at an anchor position.
/// `anchor` is (ax, ay) each in {0, 1, 2} mapping to start/center/end.
/// `fill` is the colour used to fill new empty space.
pub fn resize_canvas(
    state: &mut CanvasState,
    new_w: u32,
    new_h: u32,
    anchor: (u32, u32),
    fill: Rgba<u8>,
) {
    let old_w = state.width;
    let old_h = state.height;

    // The pixel offset of the old image within the new canvas
    let offset_x: i32 = match anchor.0 {
        0 => 0,
        1 => ((new_w as i32) - (old_w as i32)) / 2,
        _ => (new_w as i32) - (old_w as i32),
    };
    let offset_y: i32 = match anchor.1 {
        0 => 0,
        1 => ((new_h as i32) - (old_h as i32)) / 2,
        _ => (new_h as i32) - (old_h as i32),
    };

    for layer in &mut state.layers {
        let old_flat = layer.pixels.to_rgba_image();
        let mut new_img = RgbaImage::from_pixel(new_w, new_h, fill);

        for y in 0..old_h {
            for x in 0..old_w {
                let nx = x as i32 + offset_x;
                let ny = y as i32 + offset_y;
                if nx >= 0 && ny >= 0 && (nx as u32) < new_w && (ny as u32) < new_h {
                    new_img.put_pixel(nx as u32, ny as u32, *old_flat.get_pixel(x, y));
                }
            }
        }
        layer.pixels = TiledImage::from_rgba_image(&new_img);
    }
    state.width = new_w;
    state.height = new_h;
    state.composite_cache = None;
    state.clear_preview_state();
    state.mark_dirty(None);
}

/// Flatten all visible layers into a single "Background" layer.
pub fn flatten_image(state: &mut CanvasState) {
    let composite = state.composite();
    state.layers.clear();
    let mut bg = Layer::new(
        "Background".to_string(),
        state.width,
        state.height,
        Rgba([0, 0, 0, 0]),
    );
    bg.pixels = TiledImage::from_rgba_image(&composite);
    state.layers.push(bg);
    state.active_layer_index = 0;
    state.composite_cache = None;
    state.clear_preview_state();
    state.mark_dirty(None);
}

// ---------------------------------------------------------------------------
//  Single-layer transforms
// ---------------------------------------------------------------------------

/// Flip a single layer horizontally.
pub fn flip_layer_horizontal(state: &mut CanvasState, layer_idx: usize) {
    if let Some(layer) = state.layers.get_mut(layer_idx) {
        layer.pixels.flip_horizontal_chunked();
    }
    state.mark_dirty(None);
}

/// Flip a single layer vertically.
pub fn flip_layer_vertical(state: &mut CanvasState, layer_idx: usize) {
    if let Some(layer) = state.layers.get_mut(layer_idx) {
        layer.pixels.flip_vertical_chunked();
    }
    state.mark_dirty(None);
}

/// Apply an affine transform to a single layer.
/// `rotation_z`: 2D rotation in degrees, `rotation_x`/`rotation_y`: perspective tilt,
/// `scale`: scale factor (1.0 = 100%), `offset`: (dx, dy) pixel offset.
pub fn affine_transform_layer(
    state: &mut CanvasState,
    layer_idx: usize,
    rotation_z: f32,
    rotation_x: f32,
    rotation_y: f32,
    scale: f32,
    offset: (f32, f32),
) {
    if layer_idx >= state.layers.len() { return; }
    let layer = &mut state.layers[layer_idx];
    let w = state.width;
    let h = state.height;

    let flat = layer.pixels.to_rgba_image();
    let result = apply_affine(&flat, w, h, rotation_z, rotation_x, rotation_y, scale, offset);
    layer.pixels = TiledImage::from_rgba_image(&result);
    state.mark_dirty(None);
}

/// Fast-path affine transform for live preview: works from a pre-flattened
/// `RgbaImage` so we skip the clone→flatten round-trip on every slider tick.
pub fn affine_transform_layer_from_flat(
    state: &mut CanvasState,
    layer_idx: usize,
    rotation_z: f32,
    rotation_x: f32,
    rotation_y: f32,
    scale: f32,
    offset: (f32, f32),
    original_flat: &RgbaImage,
) {
    if layer_idx >= state.layers.len() { return; }
    let w = state.width;
    let h = state.height;
    let result = apply_affine(original_flat, w, h, rotation_z, rotation_x, rotation_y, scale, offset);
    let layer = &mut state.layers[layer_idx];
    layer.pixels = TiledImage::from_rgba_image(&result);
    state.mark_dirty(None);
}

// ---------------------------------------------------------------------------
//  TiledImage helpers (legacy, kept for reference)
// ---------------------------------------------------------------------------

/// Apply a 2D affine + 3D perspective transform to an RgbaImage,
/// using bilinear sampling against a transparent background.
///
/// * `rotation_z` — normal 2D rotation (degrees)
/// * `rotation_x` / `rotation_y` — perspective tilt around X/Y axes (degrees)
/// * `scale` — uniform scale factor
/// * `offset` — (dx, dy) pixel translation
fn apply_affine(
    src: &RgbaImage,
    canvas_w: u32,
    canvas_h: u32,
    rotation_z: f32,
    rotation_x: f32,
    rotation_y: f32,
    scale: f32,
    offset: (f32, f32),
) -> RgbaImage {
    let mut dst = RgbaImage::new(canvas_w, canvas_h);
    let cx = canvas_w as f32 * 0.5;
    let cy = canvas_h as f32 * 0.5;
    let inv_scale = if scale.abs() > 1e-6 { 1.0 / scale } else { 1.0 };

    // Focal length for perspective projection (proportional to image size).
    let focal = (canvas_w.max(canvas_h) as f32) * 1.5;

    // 3D rotation matrix R = Rz * Ry * Rx  (first two columns, since z_input = 0).
    let (sz, cz) = rotation_z.to_radians().sin_cos();
    let (sxr, cxr) = rotation_x.to_radians().sin_cos();
    let (syr, cyr) = rotation_y.to_radians().sin_cos();

    let r00 = cz * cyr;
    let r01 = cz * syr * sxr - sz * cxr;
    let r10 = sz * cyr;
    let r11 = sz * syr * sxr + cz * cxr;
    let r20 = -syr;
    let r21 = cyr * sxr;

    let h = [
        [focal * r00, focal * r01, 0.0f32],
        [focal * r10, focal * r11, 0.0f32],
        [r20,         r21,         focal],
    ];
    let hi = invert_3x3(h);

    let (h00, h01, h02) = (hi[0][0], hi[0][1], hi[0][2]);
    let (h10, h11, h12) = (hi[1][0], hi[1][1], hi[1][2]);
    let (h20, h21, h22) = (hi[2][0], hi[2][1], hi[2][2]);

    let src_w = src.width() as i32;
    let src_h = src.height() as i32;
    let src_stride = src_w as usize * 4;
    let src_raw = src.as_raw();

    let row_bytes = canvas_w as usize * 4;
    let dst_raw = dst.as_mut();

    // Process rows in parallel using rayon.
    dst_raw.par_chunks_mut(row_bytes).enumerate().for_each(|(dy, row)| {
        let v = (dy as f32 - cy - offset.1) * inv_scale;
        let base_sx = h01 * v + h02;
        let base_sy = h11 * v + h12;
        let base_sw = h21 * v + h22;

        for dx in 0..canvas_w as usize {
            let u = (dx as f32 - cx - offset.0) * inv_scale;

            let w = h20 * u + base_sw;
            if w.abs() < 1e-8 { continue; }
            let inv_w = 1.0 / w;
            let src_x = (h00 * u + base_sx) * inv_w + cx;
            let src_y = (h10 * u + base_sy) * inv_w + cy;

            let x0 = src_x.floor() as i32;
            let y0 = src_y.floor() as i32;

            if x0 < -1 || y0 < -1 || x0 >= src_w || y0 >= src_h { continue; }

            let fx = src_x - x0 as f32;
            let fy = src_y - y0 as f32;

            let sample = |sx: i32, sy: i32| -> [f32; 4] {
                if sx < 0 || sy < 0 || sx >= src_w || sy >= src_h {
                    [0.0; 4]
                } else {
                    let idx = sy as usize * src_stride + sx as usize * 4;
                    [
                        src_raw[idx]     as f32,
                        src_raw[idx + 1] as f32,
                        src_raw[idx + 2] as f32,
                        src_raw[idx + 3] as f32,
                    ]
                }
            };

            let tl = sample(x0, y0);
            let tr = sample(x0 + 1, y0);
            let bl = sample(x0, y0 + 1);
            let br = sample(x0 + 1, y0 + 1);

            let px = dx * 4;
            for c in 0..4 {
                let top = tl[c] + (tr[c] - tl[c]) * fx;
                let bot = bl[c] + (br[c] - bl[c]) * fx;
                row[px + c] = (top + (bot - top) * fy).round().clamp(0.0, 255.0) as u8;
            }
        }
    });
    dst
}

/// Invert a 3×3 matrix. Returns identity on singular input.
fn invert_3x3(m: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let (a, b, c) = (m[0][0], m[0][1], m[0][2]);
    let (d, e, f) = (m[1][0], m[1][1], m[1][2]);
    let (g, h, i) = (m[2][0], m[2][1], m[2][2]);

    let det = a * (e * i - f * h)
            - b * (d * i - f * g)
            + c * (d * h - e * g);
    if det.abs() < 1e-12 {
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    }
    let inv = 1.0 / det;
    [
        [(e*i - f*h) * inv, (c*h - b*i) * inv, (b*f - c*e) * inv],
        [(f*g - d*i) * inv, (a*i - c*g) * inv, (c*d - a*f) * inv],
        [(d*h - e*g) * inv, (b*g - a*h) * inv, (a*e - b*d) * inv],
    ]
}

/// Bilinear interpolation sampling from an RgbaImage.
fn bilinear_sample(img: &RgbaImage, x: f32, y: f32) -> Rgba<u8> {
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;

    let sample = |sx: i32, sy: i32| -> [f32; 4] {
        if sx < 0 || sy < 0 || sx >= img.width() as i32 || sy >= img.height() as i32 {
            [0.0; 4]
        } else {
            let p = img.get_pixel(sx as u32, sy as u32);
            [p[0] as f32, p[1] as f32, p[2] as f32, p[3] as f32]
        }
    };

    let tl = sample(x0, y0);
    let tr = sample(x0 + 1, y0);
    let bl = sample(x0, y0 + 1);
    let br = sample(x0 + 1, y0 + 1);

    let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
    let mut out = [0u8; 4];
    for c in 0..4 {
        let top = lerp(tl[c], tr[c], fx);
        let bot = lerp(bl[c], br[c], fx);
        out[c] = lerp(top, bot, fy).round().clamp(0.0, 255.0) as u8;
    }
    Rgba(out)
}

// ============================================================================
// DISPLACEMENT WARP — for Liquify tool
// ============================================================================

/// A displacement field storing (dx, dy) offsets per pixel.
#[derive(Clone, Debug)]
pub struct DisplacementField {
    pub width: u32,
    pub height: u32,
    /// Flat array of (dx, dy) pairs.  Length = width * height * 2.
    pub data: Vec<f32>,
}

impl DisplacementField {
    /// Create a zero-displacement field.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            data: vec![0.0; (width as usize) * (height as usize) * 2],
        }
    }

    #[inline]
    pub fn get(&self, x: u32, y: u32) -> (f32, f32) {
        let idx = (y as usize * self.width as usize + x as usize) * 2;
        (self.data[idx], self.data[idx + 1])
    }

    #[inline]
    pub fn add(&mut self, x: u32, y: u32, dx: f32, dy: f32) {
        let idx = (y as usize * self.width as usize + x as usize) * 2;
        self.data[idx] += dx;
        self.data[idx + 1] += dy;
    }

    /// Apply a push-mode displacement within the brush radius.
    /// `center_x, center_y`: brush center in canvas coords.
    /// `delta_x, delta_y`: displacement direction.
    /// `radius`: brush radius.
    /// `strength`: 0.0–1.0.
    /// Returns the bounding box of affected pixels: (x0, y0, x1, y1).
    pub fn apply_push(
        &mut self,
        center_x: f32,
        center_y: f32,
        delta_x: f32,
        delta_y: f32,
        radius: f32,
        strength: f32,
    ) -> (i32, i32, i32, i32) {
        let r = radius.max(1.0);
        let sigma = r / 3.0;
        let sigma_sq_2 = 2.0 * sigma * sigma;

        let x0 = ((center_x - r).floor() as i32).max(0);
        let y0 = ((center_y - r).floor() as i32).max(0);
        let x1 = ((center_x + r).ceil() as i32).min(self.width as i32);
        let y1 = ((center_y + r).ceil() as i32).min(self.height as i32);

        for py in y0..y1 {
            for px in x0..x1 {
                let dx = px as f32 - center_x;
                let dy = py as f32 - center_y;
                let dist_sq = dx * dx + dy * dy;
                if dist_sq > r * r {
                    continue;
                }
                let weight = (-dist_sq / sigma_sq_2).exp() * strength;
                self.add(px as u32, py as u32, delta_x * weight, delta_y * weight);
            }
        }

        (x0, y0, x1, y1)
    }

    /// Apply expand (bloat) mode — push outward from center.
    ///
    /// Uses a zero-at-centre radial profile `t*(1-t)^2` (peaks at t=1/3 of
    /// radius, zero at both centre and edge) to prevent the "donut" artefact
    /// that Gaussian weighting causes when strokes accumulate: with Gaussian
    /// the centre received maximum displacement and eventually sampled from
    /// the far ring, hollowing out.  This profile guarantees zero force exactly
    /// at the brush centre so repeated strokes keep expanding cleanly.
    pub fn apply_expand(
        &mut self,
        center_x: f32,
        center_y: f32,
        radius: f32,
        strength: f32,
    ) -> (i32, i32, i32, i32) {
        let r = radius.max(1.0);

        let x0 = ((center_x - r).floor() as i32).max(0);
        let y0 = ((center_y - r).floor() as i32).max(0);
        let x1 = ((center_x + r).ceil() as i32).min(self.width as i32);
        let y1 = ((center_y + r).ceil() as i32).min(self.height as i32);

        for py in y0..y1 {
            for px in x0..x1 {
                let dx = px as f32 - center_x;
                let dy = py as f32 - center_y;
                let dist_sq = dx * dx + dy * dy;
                if dist_sq > r * r { continue; }
                let dist = dist_sq.sqrt().max(0.001);
                // Smooth centre-peaked falloff: max force at centre, zero at edge.
                // (1-t)^2 with t=dist/r gives 1.0 at centre, 0.0 at edge, no discontinuity.
                let t = dist / r;
                let weight = (1.0 - t) * (1.0 - t) * strength * 3.0;
                self.add(px as u32, py as u32, dx / dist * weight, dy / dist * weight);
            }
        }

        (x0, y0, x1, y1)
    }

    /// Apply contract (pinch) mode — pull inward toward center.
    pub fn apply_contract(
        &mut self,
        center_x: f32,
        center_y: f32,
        radius: f32,
        strength: f32,
    ) -> (i32, i32, i32, i32) {
        let r = radius.max(1.0);
        let sigma = r / 3.0;
        let sigma_sq_2 = 2.0 * sigma * sigma;

        let x0 = ((center_x - r).floor() as i32).max(0);
        let y0 = ((center_y - r).floor() as i32).max(0);
        let x1 = ((center_x + r).ceil() as i32).min(self.width as i32);
        let y1 = ((center_y + r).ceil() as i32).min(self.height as i32);

        for py in y0..y1 {
            for px in x0..x1 {
                let dx = px as f32 - center_x;
                let dy = py as f32 - center_y;
                let dist_sq = dx * dx + dy * dy;
                if dist_sq > r * r {
                    continue;
                }
                let dist = dist_sq.sqrt().max(0.001);
                let weight = (-dist_sq / sigma_sq_2).exp() * strength;
                self.add(px as u32, py as u32, -dx / dist * weight * 2.0, -dy / dist * weight * 2.0);
            }
        }

        (x0, y0, x1, y1)
    }

    /// Apply clockwise twirl around brush center.
    pub fn apply_twirl(
        &mut self,
        center_x: f32,
        center_y: f32,
        radius: f32,
        strength: f32,
        clockwise: bool,
    ) -> (i32, i32, i32, i32) {
        let r = radius.max(1.0);
        let sigma = r / 3.0;
        let sigma_sq_2 = 2.0 * sigma * sigma;
        let dir = if clockwise { 1.0 } else { -1.0 };

        let x0 = ((center_x - r).floor() as i32).max(0);
        let y0 = ((center_y - r).floor() as i32).max(0);
        let x1 = ((center_x + r).ceil() as i32).min(self.width as i32);
        let y1 = ((center_y + r).ceil() as i32).min(self.height as i32);

        for py in y0..y1 {
            for px in x0..x1 {
                let dx = px as f32 - center_x;
                let dy = py as f32 - center_y;
                let dist_sq = dx * dx + dy * dy;
                if dist_sq > r * r {
                    continue;
                }
                let weight = (-dist_sq / sigma_sq_2).exp() * strength * dir;
                // Perpendicular direction for twirl
                self.add(px as u32, py as u32, -dy * weight * 0.1, dx * weight * 0.1);
            }
        }

        (x0, y0, x1, y1)
    }
}

/// Warp a source image using a displacement field.
/// For each destination pixel, samples `src` at `(x - dx, y - dy)`.
/// Only warps pixels within the `dirty_rect` region, leaving others as `prev`.
pub fn warp_displacement_region(
    src: &RgbaImage,
    displacement: &DisplacementField,
    prev: &[u8],
    dirty_rect: (i32, i32, i32, i32),
    out_w: u32,
    out_h: u32,
) -> Vec<u8> {
    let (dx0, dy0, dx1, dy1) = dirty_rect;
    let dx0 = dx0.max(0) as u32;
    let dy0 = dy0.max(0) as u32;
    let dx1 = (dx1 as u32).min(out_w);
    let dy1 = (dy1 as u32).min(out_h);

    let mut output = prev.to_vec();
    let row_bytes = out_w as usize * 4;

    let src_w = src.width() as i32;
    let src_h = src.height() as i32;
    let src_raw = src.as_raw();
    let src_stride = src_w as usize * 4;

    // Process dirty region rows in parallel
    let dirty_rows: Vec<(u32, Vec<u8>)> = (dy0..dy1)
        .into_par_iter()
        .map(|y| {
            let mut row = vec![0u8; (dx1 - dx0) as usize * 4];
            for x in dx0..dx1 {
                let (ddx, ddy) = displacement.get(x, y);
                let sx = x as f32 - ddx;
                let sy = y as f32 - ddy;

                let x0 = sx.floor() as i32;
                let y0 = sy.floor() as i32;

                if x0 < -1 || y0 < -1 || x0 >= src_w || y0 >= src_h {
                    continue;
                }

                let fx = sx - x0 as f32;
                let fy = sy - y0 as f32;

                let sample = |sx: i32, sy: i32| -> [f32; 4] {
                    if sx < 0 || sy < 0 || sx >= src_w || sy >= src_h {
                        [0.0; 4]
                    } else {
                        let idx = sy as usize * src_stride + sx as usize * 4;
                        [
                            src_raw[idx] as f32,
                            src_raw[idx + 1] as f32,
                            src_raw[idx + 2] as f32,
                            src_raw[idx + 3] as f32,
                        ]
                    }
                };

                let tl = sample(x0, y0);
                let tr = sample(x0 + 1, y0);
                let bl = sample(x0, y0 + 1);
                let br = sample(x0 + 1, y0 + 1);

                let col = (x - dx0) as usize * 4;
                for c in 0..4 {
                    let top = tl[c] + (tr[c] - tl[c]) * fx;
                    let bot = bl[c] + (br[c] - bl[c]) * fx;
                    row[col + c] = (top + (bot - top) * fy).round().clamp(0.0, 255.0) as u8;
                }
            }
            (y, row)
        })
        .collect();

    for (y, row) in dirty_rows {
        let out_start = y as usize * row_bytes + dx0 as usize * 4;
        let len = row.len();
        output[out_start..out_start + len].copy_from_slice(&row);
    }

    output
}

/// Warp an entire source image using a displacement field (full resolution).
pub fn warp_displacement_full(
    src: &RgbaImage,
    displacement: &DisplacementField,
) -> RgbaImage {
    let w = displacement.width;
    let h = displacement.height;
    let mut dst = RgbaImage::new(w, h);
    let row_bytes = w as usize * 4;
    let src_w = src.width() as i32;
    let src_h = src.height() as i32;
    let src_raw = src.as_raw();
    let src_stride = src_w as usize * 4;

    dst.as_mut().par_chunks_mut(row_bytes).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let (ddx, ddy) = displacement.get(x, y as u32);
            let sx = x as f32 - ddx;
            let sy = y as f32 - ddy;

            let x0 = sx.floor() as i32;
            let y0 = sy.floor() as i32;

            if x0 < -1 || y0 < -1 || x0 >= src_w || y0 >= src_h {
                continue;
            }

            let fx = sx - x0 as f32;
            let fy = sy - y0 as f32;

            let sample = |sx: i32, sy: i32| -> [f32; 4] {
                if sx < 0 || sy < 0 || sx >= src_w || sy >= src_h {
                    [0.0; 4]
                } else {
                    let idx = sy as usize * src_stride + sx as usize * 4;
                    [
                        src_raw[idx] as f32,
                        src_raw[idx + 1] as f32,
                        src_raw[idx + 2] as f32,
                        src_raw[idx + 3] as f32,
                    ]
                }
            };

            let tl = sample(x0, y0);
            let tr = sample(x0 + 1, y0);
            let bl = sample(x0, y0 + 1);
            let br = sample(x0 + 1, y0 + 1);

            let px = x as usize * 4;
            for c in 0..4 {
                let top = tl[c] + (tr[c] - tl[c]) * fx;
                let bot = bl[c] + (br[c] - bl[c]) * fx;
                row[px + c] = (top + (bot - top) * fy).round().clamp(0.0, 255.0) as u8;
            }
        }
    });
    dst
}

// ============================================================================
// MESH WARP — bilinear cell interpolation for mesh warp tool
// ============================================================================

/// Warp a source image using a control point mesh.
///
/// `original_points` and `deformed_points` are `(cols+1) * (rows+1)` arrays.
/// For each output pixel, find which cell it belongs to in the DEFORMED grid,
/// compute local (u,v), then bilinear-interpolate the ORIGINAL positions to
/// find the source coordinate to sample from.
pub fn warp_mesh(
    src: &RgbaImage,
    original_points: &[[f32; 2]],
    deformed_points: &[[f32; 2]],
    grid_cols: usize,
    grid_rows: usize,
    out_w: u32,
    out_h: u32,
) -> RgbaImage {
    let mut dst = RgbaImage::new(out_w, out_h);
    let row_bytes = out_w as usize * 4;
    let src_w = src.width() as i32;
    let src_h = src.height() as i32;
    let src_raw = src.as_raw();
    let src_stride = src_w as usize * 4;

    let pts_per_row = grid_cols + 1;

    dst.as_mut().par_chunks_mut(row_bytes).enumerate().for_each(|(y, row)| {
        let py = y as f32 + 0.5;
        for x in 0..out_w as usize {
            let px = x as f32 + 0.5;

            // Find which cell this pixel belongs to in the DEFORMED grid
            let mut best_cell = None;
            let mut best_u = 0.0f32;
            let mut best_v = 0.0f32;

            for cr in 0..grid_rows {
                for cc in 0..grid_cols {
                    let i00 = cr * pts_per_row + cc;
                    let i10 = cr * pts_per_row + cc + 1;
                    let i01 = (cr + 1) * pts_per_row + cc;
                    let i11 = (cr + 1) * pts_per_row + cc + 1;

                    // Use DEFORMED grid for cell containment test
                    let p00 = deformed_points[i00];
                    let p10 = deformed_points[i10];
                    let p01 = deformed_points[i01];
                    let p11 = deformed_points[i11];

                    // Quick AABB test
                    let min_x = p00[0].min(p10[0]).min(p01[0]).min(p11[0]);
                    let max_x = p00[0].max(p10[0]).max(p01[0]).max(p11[0]);
                    let min_y = p00[1].min(p10[1]).min(p01[1]).min(p11[1]);
                    let max_y = p00[1].max(p10[1]).max(p01[1]).max(p11[1]);

                    if px < min_x - 1.0 || px > max_x + 1.0 || py < min_y - 1.0 || py > max_y + 1.0 {
                        continue;
                    }

                    // Compute (u, v) in this deformed cell using inverse bilinear
                    if let Some((u, v)) = inverse_bilinear(px, py, p00, p10, p01, p11) {
                        if u >= -0.001 && u <= 1.001 && v >= -0.001 && v <= 1.001 {
                            best_cell = Some((cc, cr));
                            best_u = u.clamp(0.0, 1.0);
                            best_v = v.clamp(0.0, 1.0);
                            break;
                        }
                    }
                }
                if best_cell.is_some() {
                    break;
                }
            }

            if let Some((cc, cr)) = best_cell {
                // Map back to ORIGINAL grid to find source sample position
                let i00 = cr * pts_per_row + cc;
                let i10 = cr * pts_per_row + cc + 1;
                let i01 = (cr + 1) * pts_per_row + cc;
                let i11 = (cr + 1) * pts_per_row + cc + 1;

                let o00 = original_points[i00];
                let o10 = original_points[i10];
                let o01 = original_points[i01];
                let o11 = original_points[i11];

                let u = best_u;
                let v = best_v;

                let src_x = (1.0 - u) * (1.0 - v) * o00[0]
                           + u * (1.0 - v) * o10[0]
                           + (1.0 - u) * v * o01[0]
                           + u * v * o11[0];
                let src_y = (1.0 - u) * (1.0 - v) * o00[1]
                           + u * (1.0 - v) * o10[1]
                           + (1.0 - u) * v * o01[1]
                           + u * v * o11[1];

                // Bilinear sample from source
                let x0 = src_x.floor() as i32;
                let y0 = src_y.floor() as i32;

                if x0 < -1 || y0 < -1 || x0 >= src_w || y0 >= src_h {
                    continue;
                }

                let fx = src_x - x0 as f32;
                let fy = src_y - y0 as f32;

                let sample = |sx: i32, sy: i32| -> [f32; 4] {
                    if sx < 0 || sy < 0 || sx >= src_w || sy >= src_h {
                        [0.0; 4]
                    } else {
                        let idx = sy as usize * src_stride + sx as usize * 4;
                        [
                            src_raw[idx] as f32,
                            src_raw[idx + 1] as f32,
                            src_raw[idx + 2] as f32,
                            src_raw[idx + 3] as f32,
                        ]
                    }
                };

                let tl = sample(x0, y0);
                let tr = sample(x0 + 1, y0);
                let bl = sample(x0, y0 + 1);
                let br = sample(x0 + 1, y0 + 1);

                let col = x * 4;
                for c in 0..4 {
                    let top = tl[c] + (tr[c] - tl[c]) * fx;
                    let bot = bl[c] + (br[c] - bl[c]) * fx;
                    row[col + c] = (top + (bot - top) * fy).round().clamp(0.0, 255.0) as u8;
                }
            }
        }
    });
    dst
}

/// Inverse bilinear interpolation: given a point (px, py) and four corners
/// of a bilinear quad, find (u, v) such that bilinear(u,v) ≈ (px, py).
fn inverse_bilinear(
    px: f32, py: f32,
    p00: [f32; 2], p10: [f32; 2], p01: [f32; 2], p11: [f32; 2],
) -> Option<(f32, f32)> {
    // Use iterative Newton's method (2-3 iterations suffice for smooth quads)
    let mut u = 0.5f32;
    let mut v = 0.5f32;

    for _ in 0..6 {
        // Evaluate bilinear at (u, v)
        let qx = (1.0 - u) * (1.0 - v) * p00[0]
                + u * (1.0 - v) * p10[0]
                + (1.0 - u) * v * p01[0]
                + u * v * p11[0];
        let qy = (1.0 - u) * (1.0 - v) * p00[1]
                + u * (1.0 - v) * p10[1]
                + (1.0 - u) * v * p01[1]
                + u * v * p11[1];

        let ex = px - qx;
        let ey = py - qy;

        if ex.abs() < 0.01 && ey.abs() < 0.01 {
            return Some((u, v));
        }

        // Jacobian
        let dxdu = -(1.0 - v) * p00[0] + (1.0 - v) * p10[0] - v * p01[0] + v * p11[0];
        let dxdv = -(1.0 - u) * p00[0] - u * p10[0] + (1.0 - u) * p01[0] + u * p11[0];
        let dydu = -(1.0 - v) * p00[1] + (1.0 - v) * p10[1] - v * p01[1] + v * p11[1];
        let dydv = -(1.0 - u) * p00[1] - u * p10[1] + (1.0 - u) * p01[1] + u * p11[1];

        let det = dxdu * dydv - dxdv * dydu;
        if det.abs() < 1e-8 {
            return None;
        }

        let inv_det = 1.0 / det;
        let du = (ex * dydv - ey * dxdv) * inv_det;
        let dv = (ey * dxdu - ex * dydu) * inv_det;

        u += du;
        v += dv;
    }

    Some((u, v))
}

// ============================================================================
// CATMULL-ROM MESH WARP — smooth bicubic interpolation for mesh warp tool
// ============================================================================

/// Catmull-Rom basis functions (cardinal spline, tau = 0.5).
/// Returns weights for P_{i-1}, P_i, P_{i+1}, P_{i+2} given parameter t in [0,1].
#[inline]
fn catmull_rom_weights(t: f32) -> [f32; 4] {
    let t2 = t * t;
    let t3 = t2 * t;
    [
        -0.5 * t3 + t2 - 0.5 * t,          // w_{i-1}
         1.5 * t3 - 2.5 * t2 + 1.0,         // w_i
        -1.5 * t3 + 2.0 * t2 + 0.5 * t,     // w_{i+1}
         0.5 * t3 - 0.5 * t2,               // w_{i+2}
    ]
}

/// Evaluate a 1D Catmull-Rom spline at parameter `t` (0..1 within segment `i`..`i+1`).
/// `vals` is the full row/column of control point coordinates.
/// `i` is the segment index. Boundary indices are clamped.
#[inline]
fn catmull_rom_eval_1d(vals: &[[f32; 2]], i: usize, t: f32, count: usize) -> [f32; 2] {
    let w = catmull_rom_weights(t);
    let i0 = if i == 0 { 0 } else { i - 1 };
    let i1 = i;
    let i2 = (i + 1).min(count - 1);
    let i3 = (i + 2).min(count - 1);
    [
        w[0] * vals[i0][0] + w[1] * vals[i1][0] + w[2] * vals[i2][0] + w[3] * vals[i3][0],
        w[0] * vals[i0][1] + w[1] * vals[i1][1] + w[2] * vals[i2][1] + w[3] * vals[i3][1],
    ]
}

/// Evaluate the bicubic Catmull-Rom surface at global parametric (u, v)
/// where u spans [0, cols] and v spans [0, rows].
/// `points` is row-major (rows+1) × (cols+1).
#[inline]
fn catmull_rom_surface(
    points: &[[f32; 2]],
    cols: usize,
    rows: usize,
    u_global: f32,
    v_global: f32,
) -> [f32; 2] {
    let pts_per_row = cols + 1;
    let num_rows = rows + 1;

    // Determine cell and local parameters
    let col_f = u_global.clamp(0.0, cols as f32 - 0.0001);
    let row_f = v_global.clamp(0.0, rows as f32 - 0.0001);
    let ci = (col_f as usize).min(cols - 1);
    let ri = (row_f as usize).min(rows - 1);
    let u_local = col_f - ci as f32;
    let v_local = row_f - ri as f32;

    // Compute v-weights
    let wv = catmull_rom_weights(v_local);
    let rv0 = if ri == 0 { 0 } else { ri - 1 };
    let rv1 = ri;
    let rv2 = (ri + 1).min(num_rows - 1);
    let rv3 = (ri + 2).min(num_rows - 1);

    // For each of the 4 v-rows, evaluate Catmull-Rom in u-direction
    let row_indices = [rv0, rv1, rv2, rv3];
    let mut row_vals = [[0.0f32; 2]; 4];
    let wu = catmull_rom_weights(u_local);
    let cu0 = if ci == 0 { 0 } else { ci - 1 };
    let cu1 = ci;
    let cu2 = (ci + 1).min(pts_per_row - 1);
    let cu3 = (ci + 2).min(pts_per_row - 1);

    for (j, &rv) in row_indices.iter().enumerate() {
        let base = rv * pts_per_row;
        let p0 = points[base + cu0];
        let p1 = points[base + cu1];
        let p2 = points[base + cu2];
        let p3 = points[base + cu3];
        row_vals[j] = [
            wu[0] * p0[0] + wu[1] * p1[0] + wu[2] * p2[0] + wu[3] * p3[0],
            wu[0] * p0[1] + wu[1] * p1[1] + wu[2] * p2[1] + wu[3] * p3[1],
        ];
    }

    // Blend 4 row values in v-direction
    [
        wv[0] * row_vals[0][0] + wv[1] * row_vals[1][0] + wv[2] * row_vals[2][0] + wv[3] * row_vals[3][0],
        wv[0] * row_vals[0][1] + wv[1] * row_vals[1][1] + wv[2] * row_vals[2][1] + wv[3] * row_vals[3][1],
    ]
}

/// Evaluate a 1D Catmull-Rom curve along a row of control points at parameter t_global ∈ [0, n].
/// Used for drawing smooth overlay curves.
pub fn catmull_rom_curve_point(
    points: &[[f32; 2]],
    t_global: f32,
) -> [f32; 2] {
    let n = points.len();
    if n == 0 { return [0.0, 0.0]; }
    if n == 1 { return points[0]; }
    let max_t = (n - 1) as f32 - 0.0001;
    let t = t_global.clamp(0.0, max_t);
    let i = (t as usize).min(n - 2);
    let local_t = t - i as f32;
    catmull_rom_eval_1d(points, i, local_t, n)
}

/// Generate a displacement field from mesh warp control points using Catmull-Rom
/// bicubic interpolation. For each output pixel, evaluates the spline surface
/// to find where the deformed grid maps that position, then stores the offset.
///
/// The displacement convention matches Liquify: `src(x,y) = output(x - dx, y - dy)`.
pub fn generate_displacement_from_mesh(
    original_points: &[[f32; 2]],
    deformed_points: &[[f32; 2]],
    grid_cols: usize,
    grid_rows: usize,
    out_w: u32,
    out_h: u32,
) -> DisplacementField {
    let mut field = DisplacementField::new(out_w, out_h);
    let row_floats = out_w as usize * 2;

    field.data.par_chunks_mut(row_floats).enumerate().for_each(|(y, row)| {
        for x in 0..out_w as usize {
            // Global parametric coords for this pixel
            let u_global = (x as f32 + 0.5) / out_w as f32 * grid_cols as f32;
            let v_global = (y as f32 + 0.5) / out_h as f32 * grid_rows as f32;

            // Evaluate spline on both original and deformed grids
            let orig = catmull_rom_surface(original_points, grid_cols, grid_rows, u_global, v_global);
            let def = catmull_rom_surface(deformed_points, grid_cols, grid_rows, u_global, v_global);

            // Displacement = deformed - original position
            let col = x * 2;
            row[col]     = def[0] - orig[0];
            row[col + 1] = def[1] - orig[1];
        }
    });

    field
}

/// Fast displacement generation — skips evaluating the original grid since a
/// uniform lattice's Catmull-Rom surface is the identity mapping. Only evaluates
/// the deformed grid, halving per-pixel work.
///
/// Writes directly into `out_data` (must be pre-sized to `w * h * 2` floats).
pub fn generate_displacement_from_mesh_fast(
    deformed_points: &[[f32; 2]],
    grid_cols: usize,
    grid_rows: usize,
    out_w: u32,
    out_h: u32,
    out_data: &mut [f32],
) {
    let row_floats = out_w as usize * 2;

    out_data.par_chunks_mut(row_floats).enumerate().for_each(|(y, row)| {
        for x in 0..out_w as usize {
            let u_global = (x as f32 + 0.5) / out_w as f32 * grid_cols as f32;
            let v_global = (y as f32 + 0.5) / out_h as f32 * grid_rows as f32;

            // Only evaluate deformed grid — original is identity
            let def = catmull_rom_surface(deformed_points, grid_cols, grid_rows, u_global, v_global);

            let col = x * 2;
            row[col]     = def[0] - (x as f32 + 0.5);
            row[col + 1] = def[1] - (y as f32 + 0.5);
        }
    });
}

/// Warp a source image using Catmull-Rom mesh + displacement field (full resolution).
/// Used for final commit.
pub fn warp_mesh_catmull_rom(
    src: &RgbaImage,
    original_points: &[[f32; 2]],
    deformed_points: &[[f32; 2]],
    grid_cols: usize,
    grid_rows: usize,
    out_w: u32,
    out_h: u32,
) -> RgbaImage {
    let displacement = generate_displacement_from_mesh(
        original_points, deformed_points, grid_cols, grid_rows, out_w, out_h,
    );
    warp_displacement_full(src, &displacement)
}
