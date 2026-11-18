use super::OnnxError;
use image::{ImageBuffer, Luma, RgbaImage, imageops};

pub(super) type Mask = ImageBuffer<Luma<f32>, Vec<f32>>;
pub(super) const MAX_PIXELS: u64 = 16 * 1024 * 1024;
const MAX_TILES: usize = 6;
const MAX_CANDIDATES: usize = 1024;

fn origins(length: u32, tile: u32) -> Vec<u32> {
    let end = length - tile;
    let step = (tile * 3 / 4).max(1);
    // Bound work on extreme aspect ratios before ranking the tile grid.
    let mut starts: Vec<_> = (0..end)
        .step_by(step as usize)
        .take(MAX_CANDIDATES)
        .collect();
    starts.push(end);
    starts
}

fn gate(alpha: f32, threshold: f32) -> f32 {
    let distance = if alpha < threshold {
        (threshold - alpha) / threshold
    } else {
        (alpha - threshold) / (1.0 - threshold)
    };
    (1.0 - distance * 1.5).max(0.0)
}

/// The first pass sets subject context. Crops can only refine areas near
/// uncertain alpha, and taper out before a neighbouring crop or the base.
pub(super) fn refine(
    image: &RgbaImage,
    base: &Mask,
    model_size: (u32, u32),
    threshold: f32,
    mut infer: impl FnMut(&RgbaImage) -> Result<Mask, OnnxError>,
) -> Result<Option<Mask>, OnnxError> {
    let (width, height) = image.dimensions();
    if width <= model_size.0 && height <= model_size.1 {
        return Ok(None);
    }
    let tw = width.min(model_size.0.max(128));
    let th = height.min(model_size.1.max(128));
    let xs = origins(width, tw);
    let ys = origins(height, th);
    if xs.len() * ys.len() > MAX_CANDIDATES {
        return Ok(None);
    }
    let threshold = threshold.clamp(0.001, 0.999);
    let mut global = imageops::resize(base, width, height, imageops::FilterType::Triangle);
    let mut candidates = Vec::new();
    for y in ys {
        for &x in &xs {
            let mut score = 0.0;
            for iy in (0..th).step_by(8) {
                for ix in (0..tw).step_by(8) {
                    score += gate(global.get_pixel(x + ix, y + iy)[0], threshold);
                }
            }
            if score > 0.0 {
                candidates.push((score, x, y));
            }
        }
    }
    candidates.sort_by(|a, b| b.0.total_cmp(&a.0));
    candidates.truncate(MAX_TILES);
    if candidates.is_empty() {
        return Ok(None);
    }
    let area = global.as_raw().len();
    let mut sum = vec![0.0f32; area];
    let mut weights = vec![0.0f32; area];
    let fade_x = (tw / 8).max(1) as f32;
    let fade_y = (th / 8).max(1) as f32;
    for (_, x, y) in candidates {
        let patch = imageops::crop_imm(image, x, y, tw, th).to_image();
        let local = imageops::resize(&infer(&patch)?, tw, th, imageops::FilterType::Triangle);
        for iy in 0..th {
            let top = if y == 0 { 1.0 } else { iy as f32 / fade_y };
            let bottom = if y + th == height {
                1.0
            } else {
                (th - 1 - iy) as f32 / fade_y
            };
            let wy = top.min(bottom).min(1.0);
            for ix in 0..tw {
                let left = if x == 0 { 1.0 } else { ix as f32 / fade_x };
                let right = if x + tw == width {
                    1.0
                } else {
                    (tw - 1 - ix) as f32 / fade_x
                };
                let weight = wy * left.min(right).min(1.0);
                let index = (y + iy) as usize * width as usize + (x + ix) as usize;
                sum[index] += weight * local.get_pixel(ix, iy)[0];
                weights[index] += weight;
            }
        }
    }
    for ((value, sum), weight) in global.as_mut().iter_mut().zip(sum).zip(weights) {
        if weight > 0.0 {
            // Normalise overlaps but still crossfade to the base if a tile ends.
            *value += (sum / weight - *value) * gate(*value, threshold) * weight.min(1.0);
        }
    }
    Ok(Some(global))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uncertain_edges_can_change_but_solid_regions_and_bounds_are_preserved() {
        let image = RgbaImage::new(260, 160);
        let mask = Mask::from_fn(26, 16, |x, _| {
            Luma([match x {
                0..=7 => 0.0,
                18.. => 1.0,
                _ => 0.5,
            }])
        });
        let mut calls = 0;
        let out = refine(&image, &mask, (128, 128), 0.5, |crop| {
            calls += 1;
            assert_eq!(crop.dimensions(), (128, 128));
            Ok(Mask::from_pixel(128, 128, Luma([0.8])))
        })
        .unwrap()
        .unwrap();
        assert!((1..=MAX_TILES).contains(&calls));
        assert_eq!(out.dimensions(), image.dimensions());
        assert_eq!(out.get_pixel(0, 50)[0], 0.0);
        assert_eq!(out.get_pixel(259, 50)[0], 1.0);
        assert!(out.get_pixel(130, 70)[0] > 0.7);
        assert!(out.as_raw().iter().all(|v| (0.0..=1.0).contains(v)));
    }

    #[test]
    fn small_images_and_certain_masks_do_not_run_crops() {
        for (width, height) in [(80, 80), (400, 300)] {
            let image = RgbaImage::new(width, height);
            let mask = Mask::from_pixel(8, 8, Luma([0.0]));
            assert!(
                refine(&image, &mask, (128, 128), 0.5, |_| {
                    panic!("No refinement needed")
                })
                .unwrap()
                .is_none()
            );
        }
    }
}
