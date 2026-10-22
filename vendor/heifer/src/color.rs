//! YCbCr to RGB conversion (ITU-T H.273 matrix coefficients).

use heifer_hevc_dec::recon::Frame;

use crate::image::Image;

/// How to interpret YCbCr samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColorParams {
    /// `matrix_coefficients` (H.273): 0 = identity (GBR), 1 = BT.709, 5/6 = BT.601, 9 = BT.2020.
    pub matrix: u16,
    /// Whether samples use the full range (otherwise "video" range, e.g. 16..235 for 8-bit).
    pub full_range: bool,
}

impl ColorParams {
    /// Colour description of an HEVC stream's VUI. When absent, HEVC specifies limited range,
    /// and the matrix is unspecified (treated as BT.601).
    pub fn from_frame(frame: &Frame) -> Self {
        Self {
            matrix: u16::from(frame.matrix_coeffs),
            full_range: frame.full_range,
        }
    }
}

/// `(Kr, Kb)` for a matrix coefficients code point.
fn kr_kb(matrix: u16) -> (f64, f64) {
    match matrix {
        1 => (0.2126, 0.0722),
        4 => (0.30, 0.11),
        7 => (0.212, 0.087),
        9 | 10 => (0.2627, 0.0593),
        _ => (0.299, 0.114), // 5, 6 and unspecified: BT.601
    }
}

/// Converts a decoded frame to an RGBA image (alpha opaque). Chroma is upsampled by
/// nearest neighbour.
pub fn frame_to_rgba(frame: &Frame, params: ColorParams) -> Image {
    let (w, h) = (frame.widths[0], frame.heights[0]);
    let mut data = vec![0; w as usize * h as usize * 4];
    let mut rows: Vec<&mut [u16]> = data.chunks_exact_mut(w as usize * 4).collect();
    frame_to_rgba_into(frame, params, &mut rows, w as usize, h as usize);
    Image {
        width: w,
        height: h,
        bit_depth: frame.bit_depth[0].max(frame.bit_depth[1]),
        has_alpha: false,
        data,
    }
}

/// Like [`frame_to_rgba`], but writes the top-left `width`×`height` pixels of `frame` (clamped
/// to its size) into the rows `out` (4 samples per pixel, each row at least `width` pixels), so
/// that grid tiles can be converted in place in the final image.
pub fn frame_to_rgba_into(
    frame: &Frame,
    params: ColorParams,
    out: &mut [&mut [u16]],
    width: usize,
    height: usize,
) {
    let (w, h) = (frame.widths[0] as usize, frame.heights[0] as usize);
    let (width, height) = (width.min(w), height.min(h));
    let bd_y = frame.bit_depth[0];
    let bd_c = frame.bit_depth[1];
    let out_depth = bd_y.max(bd_c);
    let max_out = f64::from((1u32 << out_depth) - 1);
    let alpha = (1u32 << out_depth) as u16 - 1;
    let rows = out.iter_mut().take(height).enumerate();
    let luma_row = |y: usize| &frame.planes[0][y * w..y * w + width];

    if frame.widths[1] == 0 {
        // Monochrome.
        let scale = max_out / f64::from((1u32 << bd_y) - 1);
        for (y, out_row) in rows {
            for (px, &v) in out_row[..width * 4].chunks_exact_mut(4).zip(luma_row(y)) {
                let g = (f64::from(v) * scale).round() as u16;
                px.copy_from_slice(&[g, g, g, alpha]);
            }
        }
        return;
    }

    let (cw, chh) = (frame.widths[1], frame.heights[1]);
    let (sx, sy) = (frame.widths[0] / cw.max(1), frame.heights[0] / chh.max(1));
    let (kr, kb) = kr_kb(params.matrix);
    let kg = 1.0 - kr - kb;

    // Normalize luma to 0..1 and chroma to -0.5..0.5.
    let (y_off, y_scale, c_scale) = {
        let (fy, fc) = (f64::from(1u32 << (bd_y - 8)), f64::from(1u32 << (bd_c - 8)));
        if params.full_range {
            (
                0.0,
                f64::from((1u32 << bd_y) - 1),
                f64::from((1u32 << bd_c) - 1),
            )
        } else {
            (16.0 * fy, 219.0 * fy, 224.0 * fc)
        }
    };
    let c_mid = f64::from(1u32 << (bd_c - 1));
    let cwu = cw as usize;
    let xcs: Vec<usize> = (0..width as u32)
        .map(|x| (x / sx).min(cw - 1) as usize)
        .collect();
    let chroma_rows = |y: usize| {
        let yc = (y as u32 / sy).min(chh - 1) as usize;
        (
            yc,
            &frame.planes[1][yc * cwu..(yc + 1) * cwu],
            &frame.planes[2][yc * cwu..(yc + 1) * cwu],
        )
    };

    if params.matrix != 0 {
        // Fixed point with FRAC fractional bits: per-sample terms come from tables, then each
        // pixel is three additions. Chroma terms are computed once per chroma sample.
        const FRAC: u32 = 14;
        let scale = max_out * f64::from(1u32 << FRAC);
        let fixed = |bits: u8, f: &dyn Fn(f64) -> f64| -> Vec<i32> {
            (0..1u32 << bits)
                .map(|v| (f(f64::from(v)) * scale).round() as i32)
                .collect()
        };
        let cn = |v: f64| (v - c_mid) / c_scale;
        let y_lut = fixed(bd_y, &|v| (v - y_off) / y_scale);
        let r_lut = fixed(bd_c, &|v| 2.0 * (1.0 - kr) * cn(v));
        let b_lut = fixed(bd_c, &|v| 2.0 * (1.0 - kb) * cn(v));
        // g = (y - kr r - kb b) / kg = y - (kr (r - y) + kb (b - y)) / kg
        let gr_lut = fixed(bd_c, &|v| kr * 2.0 * (1.0 - kr) * cn(v) / kg);
        let gb_lut = fixed(bd_c, &|v| kb * 2.0 * (1.0 - kb) * cn(v) / kg);
        let at = |t: &[i32], v: u16| t[usize::from(v).min(t.len() - 1)];
        let max = (1i32 << out_depth) - 1;
        let to_out = |v: i32| ((v + (1 << (FRAC - 1))) >> FRAC).clamp(0, max) as u16;
        let (mut rc, mut gc, mut bc) = (vec![0; cwu], vec![0; cwu], vec![0; cwu]);
        let mut current_chroma_row = usize::MAX;
        for (y, out_row) in rows {
            let (yc, cb_row, cr_row) = chroma_rows(y);
            if yc != current_chroma_row {
                current_chroma_row = yc;
                for (i, (&cb, &cr)) in cb_row.iter().zip(cr_row).enumerate() {
                    rc[i] = at(&r_lut, cr);
                    bc[i] = at(&b_lut, cb);
                    gc[i] = at(&gr_lut, cr) + at(&gb_lut, cb);
                }
            }
            let pixels = out_row[..width * 4]
                .chunks_exact_mut(4)
                .zip(luma_row(y))
                .zip(&xcs);
            for ((px, &yv), &xc) in pixels {
                let yf = at(&y_lut, yv);
                px.copy_from_slice(&[
                    to_out(yf + rc[xc]),
                    to_out(yf - gc[xc]),
                    to_out(yf + bc[xc]),
                    alpha,
                ]);
            }
        }
        return;
    }

    // Identity matrix: planes are G, B, R.
    let s = max_out / f64::from((1u32 << bd_y) - 1);
    let to_out = |v: f64| (v * s).round().clamp(0.0, max_out) as u16;
    for (y, out_row) in rows {
        let (_, cb_row, cr_row) = chroma_rows(y);
        let pixels = out_row[..width * 4]
            .chunks_exact_mut(4)
            .zip(luma_row(y))
            .zip(&xcs);
        for ((px, &g), &xc) in pixels {
            let (b, r) = (f64::from(cb_row[xc]), f64::from(cr_row[xc]));
            px.copy_from_slice(&[to_out(r), to_out(f64::from(g)), to_out(b), alpha]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(y: u16, cb: u16, cr: u16) -> Frame {
        Frame {
            widths: [2, 1, 1],
            heights: [2, 1, 1],
            planes: [vec![y; 4], vec![cb], vec![cr]],
            bit_depth: [8, 8],
            chroma_format: 1,
            full_range: false,
            matrix_coeffs: 2,
        }
    }

    #[test]
    fn grey_and_primaries() {
        let full = ColorParams {
            matrix: 6,
            full_range: true,
        };
        let img = frame_to_rgba(&frame(128, 128, 128), full);
        assert_eq!(&img.data[..4], &[128, 128, 128, 255]);
        // Limited range black and white.
        let limited = ColorParams {
            matrix: 1,
            full_range: false,
        };
        assert_eq!(
            &frame_to_rgba(&frame(16, 128, 128), limited).data[..3],
            &[0, 0, 0]
        );
        assert_eq!(
            &frame_to_rgba(&frame(235, 128, 128), limited).data[..3],
            &[255, 255, 255]
        );
        // Pure red in BT.601 full range: Y=76, Cb=85, Cr=255.
        let red = frame_to_rgba(&frame(76, 85, 255), full);
        assert!(
            red.data[0] >= 253 && red.data[1] <= 2 && red.data[2] <= 2,
            "{:?}",
            &red.data[..3]
        );
    }
}
