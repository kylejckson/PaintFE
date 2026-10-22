//! Decoded RGBA images and the geometric transforms of HEIF (crop, rotation, mirror).

/// A decoded image: interleaved RGBA samples at `bit_depth` bits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Image {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Bits per sample (8 for most photos, 10 for HDR).
    pub bit_depth: u8,
    /// Whether the alpha channel carries information (otherwise it is fully opaque).
    pub has_alpha: bool,
    /// RGBA samples, row-major, 4 values per pixel.
    pub data: Vec<u16>,
}

impl Image {
    /// Creates an image filled with one RGBA colour.
    pub fn filled(width: u32, height: u32, bit_depth: u8, rgba: [u16; 4]) -> Self {
        let n = width as usize * height as usize;
        // An all-zero image is allocated zeroed by the system, which costs nothing upfront.
        let data = if rgba == [0; 4] {
            vec![0; n * 4]
        } else {
            rgba.repeat(n)
        };
        Self {
            width,
            height,
            bit_depth,
            has_alpha: false,
            data,
        }
    }

    /// Maximum sample value.
    pub fn max_value(&self) -> u16 {
        ((1u32 << self.bit_depth) - 1) as u16
    }

    /// Converts to 8-bit RGBA, rounding higher bit depths.
    pub fn to_rgba8(&self) -> Vec<u8> {
        if self.bit_depth == 8 {
            return self.data.iter().map(|&v| v as u8).collect();
        }
        let lut = self.to_8bit_table();
        self.data.iter().map(|&v| lut[usize::from(v)]).collect()
    }

    /// Converts to 8-bit RGB, dropping alpha.
    pub fn to_rgb8(&self) -> Vec<u8> {
        let lut = self.to_8bit_table();
        let mut out = Vec::with_capacity(self.data.len() / 4 * 3);
        for p in self.data.chunks_exact(4) {
            out.extend([
                lut[usize::from(p[0])],
                lut[usize::from(p[1])],
                lut[usize::from(p[2])],
            ]);
        }
        out
    }

    /// 8-bit value of every possible sample, rounded.
    fn to_8bit_table(&self) -> Vec<u8> {
        let max = u32::from(self.max_value());
        (0..=max)
            .map(|v| ((v * 255 + max / 2) / max) as u8)
            .collect()
    }

    fn pixel(&self, x: u32, y: u32) -> [u16; 4] {
        let i = (y as usize * self.width as usize + x as usize) * 4;
        [
            self.data[i],
            self.data[i + 1],
            self.data[i + 2],
            self.data[i + 3],
        ]
    }

    /// Builds a `width`×`height` image whose pixel (x, y) is the pixel `src(x, y)` of `self`.
    /// Works in 32×32 tiles so that rotations stay cache-friendly.
    fn remap(&self, width: u32, height: u32, src: impl Fn(u32, u32) -> (u32, u32)) -> Self {
        const TILE: u32 = 32;
        let mut data = vec![0u16; width as usize * height as usize * 4];
        for ty in (0..height).step_by(TILE as usize) {
            for tx in (0..width).step_by(TILE as usize) {
                for y in ty..(ty + TILE).min(height) {
                    for x in tx..(tx + TILE).min(width) {
                        let (sx, sy) = src(x, y);
                        let i = (sy as usize * self.width as usize + sx as usize) * 4;
                        let o = (y as usize * width as usize + x as usize) * 4;
                        data[o..o + 4].copy_from_slice(&self.data[i..i + 4]);
                    }
                }
            }
        }
        Self {
            width,
            height,
            data,
            ..*self
        }
    }

    /// Crops to the rectangle at (`x`, `y`) of size `w`×`h` (clamped to the image).
    pub fn crop(&self, x: u32, y: u32, w: u32, h: u32) -> Self {
        let x = x.min(self.width);
        let y = y.min(self.height);
        let w = w.min(self.width - x);
        let h = h.min(self.height - y);
        self.remap(w, h, |px, py| (px + x, py + y))
    }

    /// Rotates anti-clockwise by `quarters` × 90° (`irot`).
    pub fn rotate_ccw(&self, quarters: u8) -> Self {
        let (w, h) = (self.width, self.height);
        match quarters % 4 {
            1 => self.remap(h, w, |x, y| (w - 1 - y, x)),
            2 => self.remap(w, h, |x, y| (w - 1 - x, h - 1 - y)),
            3 => self.remap(h, w, |x, y| (y, h - 1 - x)),
            _ => self.clone(),
        }
    }

    /// Mirrors the image (`imir`): `top_bottom` flips around the horizontal axis, otherwise
    /// around the vertical axis (left-right).
    pub fn mirror(&self, top_bottom: bool) -> Self {
        let (w, h) = (self.width, self.height);
        if top_bottom {
            self.remap(w, h, |x, y| (x, h - 1 - y))
        } else {
            self.remap(w, h, |x, y| (w - 1 - x, y))
        }
    }

    /// Draws `src` at (`x0`, `y0`) (may be negative or overflow), blending with its alpha.
    pub fn draw(&mut self, src: &Image, x0: i64, y0: i64) {
        let max = u32::from(src.max_value());
        let dst_max = u32::from(self.max_value());
        let same_depth = src.bit_depth == self.bit_depth;
        // Converts a sample from the source bit depth to the canvas bit depth.
        let shift_to = move |v: u16| -> u16 {
            if same_depth {
                v
            } else {
                ((u32::from(v) * dst_max + max / 2) / max) as u16
            }
        };
        for sy in 0..src.height {
            let y = y0 + i64::from(sy);
            if y < 0 || y >= i64::from(self.height) {
                continue;
            }
            for sx in 0..src.width {
                let x = x0 + i64::from(sx);
                if x < 0 || x >= i64::from(self.width) {
                    continue;
                }
                let s = src.pixel(sx, sy);
                let i = (y as usize * self.width as usize + x as usize) * 4;
                let a = u32::from(s[3]);
                for (d, &sc) in self.data[i..i + 3].iter_mut().zip(&s[..3]) {
                    let (fg, bg) = (u32::from(shift_to(sc)), u32::from(*d));
                    *d = ((fg * a + bg * (max - a) + max / 2) / max) as u16;
                }
                if src.has_alpha {
                    let ab = u32::from(self.data[i + 3]);
                    let out_a = a + ab * (max - a) / max;
                    self.data[i + 3] = shift_to(out_a.min(max) as u16);
                    self.has_alpha = true;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn numbered(w: u32, h: u32) -> Image {
        let data = (0..w * h).flat_map(|i| [i as u16, 0, 0, 255]).collect();
        Image {
            width: w,
            height: h,
            bit_depth: 8,
            has_alpha: false,
            data,
        }
    }

    fn reds(img: &Image) -> Vec<u16> {
        img.data.chunks(4).map(|p| p[0]).collect()
    }

    #[test]
    fn rotations_and_mirrors() {
        // 3×2 image:  0 1 2
        //             3 4 5
        let img = numbered(3, 2);
        assert_eq!(reds(&img.rotate_ccw(1)), vec![2, 5, 1, 4, 0, 3]);
        assert_eq!(reds(&img.rotate_ccw(2)), vec![5, 4, 3, 2, 1, 0]);
        assert_eq!(reds(&img.rotate_ccw(3)), vec![3, 0, 4, 1, 5, 2]);
        assert_eq!(reds(&img.mirror(false)), vec![2, 1, 0, 5, 4, 3]);
        assert_eq!(reds(&img.mirror(true)), vec![3, 4, 5, 0, 1, 2]);
        assert_eq!(reds(&img.crop(1, 1, 5, 5)), vec![4, 5]);
    }

    #[test]
    fn draw_with_alpha() {
        let mut canvas = Image::filled(2, 1, 8, [0, 0, 0, 255]);
        let mut src = Image::filled(1, 1, 8, [255, 255, 255, 128]);
        src.has_alpha = true;
        canvas.draw(&src, 1, 0);
        assert_eq!(&canvas.data[4..7], &[128, 128, 128]);
        assert_eq!(&canvas.data[0..3], &[0, 0, 0]);
        canvas.draw(&src, -5, 0); // fully outside: no effect, no panic
    }
}
