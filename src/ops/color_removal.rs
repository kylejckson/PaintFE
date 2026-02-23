use image::{Rgba, RgbaImage};
use rayon::prelude::*;
use std::collections::VecDeque;

/// Smart Contiguous Eraser — three-step color removal:
///
/// 1. **Flood fill** (BFS) from click point using strict tolerance → binary mask.
/// 2. **Mask dilation** — expand the mask by `smoothness` pixels (iterative 1px rings).
/// 3. **Color-to-Alpha** — for every pixel in the dilated mask, compute alpha from
///    max-channel distance to seed, recover true RGB via inverse un-premultiply,
///    and smoothly fade at the dilation fringe.
///
/// Returns `Vec<(x, y, [R, G, B, A])>` — full RGBA replacement values.
pub fn compute_color_removal(
    pixels: &RgbaImage,
    start_x: u32,
    start_y: u32,
    tolerance: f32,
    smoothness: u32,
    contiguous: bool,
    selection_mask: Option<&image::GrayImage>,
) -> Vec<(u32, u32, [u8; 4])> {
    let w = pixels.width();
    let h = pixels.height();
    if start_x >= w || start_y >= h {
        return Vec::new();
    }

    let seed = pixels.get_pixel(start_x, start_y);
    // Skip if the clicked pixel is fully transparent
    if seed[3] == 0 {
        return Vec::new();
    }
    let seed_rgb = [seed[0] as f32, seed[1] as f32, seed[2] as f32];
    let tol_sq = (tolerance * 2.55) * (tolerance * 2.55); // 0-100 → 0-255 range, squared

    let pixel_count = (w * h) as usize;

    // ========================================================================
    // Step 1: Build core mask via BFS flood fill (contiguous) or global match
    // ========================================================================
    let mut core_mask = vec![false; pixel_count];

    if contiguous {
        let mut queue = VecDeque::with_capacity(1024);
        let start_idx = (start_y * w + start_x) as usize;
        core_mask[start_idx] = true;
        queue.push_back((start_x, start_y));

        while let Some((px, py)) = queue.pop_front() {
            let neighbors = [
                (px.wrapping_sub(1), py),
                (px + 1, py),
                (px, py.wrapping_sub(1)),
                (px, py + 1),
            ];
            for (nx, ny) in neighbors {
                if nx >= w || ny >= h {
                    continue;
                }
                let idx = (ny * w + nx) as usize;
                if core_mask[idx] {
                    continue;
                }
                if let Some(mask) = selection_mask {
                    if mask.get_pixel(nx, ny).0[0] == 0 {
                        continue;
                    }
                }
                let p = pixels.get_pixel(nx, ny);
                if p[3] == 0 {
                    // Already transparent — include and keep expanding
                    core_mask[idx] = true;
                    queue.push_back((nx, ny));
                    continue;
                }
                let dist_sq = color_dist_sq(p, &seed_rgb);
                if dist_sq <= tol_sq {
                    core_mask[idx] = true;
                    queue.push_back((nx, ny));
                }
            }
        }
    } else {
        // Global match — all pixels matching the seed color
        core_mask.par_iter_mut().enumerate().for_each(|(idx, m)| {
            let x = (idx % w as usize) as u32;
            let y = (idx / w as usize) as u32;
            if let Some(mask) = selection_mask {
                if mask.get_pixel(x, y).0[0] == 0 {
                    return;
                }
            }
            let p = pixels.get_pixel(x, y);
            if p[3] == 0 {
                return;
            }
            if color_dist_sq(p, &seed_rgb) <= tol_sq {
                *m = true;
            }
        });
    }

    // ========================================================================
    // Step 2: Dilate mask by `smoothness` pixels (iterative 1-pixel rings)
    // ========================================================================
    // `ring` stores the distance (in dilation iterations) from the core edge.
    // 0 = core pixel, 1..=smoothness = dilated fringe.
    // u32::MAX = not in mask at all.
    let mut distance: Vec<u32> = core_mask
        .iter()
        .map(|&m| if m { 0 } else { u32::MAX })
        .collect();

    if smoothness > 0 {
        // Seed the BFS frontier from edges of core mask
        let mut frontier: VecDeque<(u32, u32)> = VecDeque::new();
        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) as usize;
                if !core_mask[idx] {
                    continue;
                }
                // If any 4-neighbor is NOT in core, this is an edge pixel
                let neighbors = [
                    (x.wrapping_sub(1), y),
                    (x + 1, y),
                    (x, y.wrapping_sub(1)),
                    (x, y + 1),
                ];
                for (nx, ny) in neighbors {
                    if nx >= w || ny >= h {
                        continue;
                    }
                    let nidx = (ny * w + nx) as usize;
                    if !core_mask[nidx] && distance[nidx] == u32::MAX {
                        // Check selection mask
                        if let Some(mask) = selection_mask {
                            if mask.get_pixel(nx, ny).0[0] == 0 {
                                continue;
                            }
                        }
                        distance[nidx] = 1;
                        frontier.push_back((nx, ny));
                    }
                }
            }
        }

        // Continue BFS dilation for remaining rings
        while let Some((px, py)) = frontier.pop_front() {
            let cur_dist = distance[(py * w + px) as usize];
            if cur_dist >= smoothness {
                continue;
            }
            let neighbors = [
                (px.wrapping_sub(1), py),
                (px + 1, py),
                (px, py.wrapping_sub(1)),
                (px, py + 1),
            ];
            for (nx, ny) in neighbors {
                if nx >= w || ny >= h {
                    continue;
                }
                let nidx = (ny * w + nx) as usize;
                if distance[nidx] != u32::MAX {
                    continue;
                }
                if let Some(mask) = selection_mask {
                    if mask.get_pixel(nx, ny).0[0] == 0 {
                        continue;
                    }
                }
                distance[nidx] = cur_dist + 1;
                frontier.push_back((nx, ny));
            }
        }
    }

    // ========================================================================
    // Step 3: Color-to-Alpha with RGB recovery
    // ========================================================================
    // For each pixel in the dilated mask (distance != u32::MAX):
    //  - Compute `removal_alpha` from max-channel distance to seed color
    //  - For fringe pixels (distance > 0), fade by (smoothness - dist + 1) / (smoothness + 1)
    //  - Recover true RGB from the "color-to-alpha" inverse formula
    let mut results: Vec<(u32, u32, [u8; 4])> = Vec::new();

    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) as usize;
            let dist = distance[idx];
            if dist == u32::MAX {
                continue;
            }

            let p = pixels.get_pixel(x, y);
            let orig_a = p[3];
            if orig_a == 0 {
                continue; // already transparent, nothing to do
            }

            // --- Color-to-Alpha ---
            // Max-channel distance (like GIMP's color-to-alpha)
            let r = p[0] as f32;
            let g = p[1] as f32;
            let b = p[2] as f32;

            let dr = (r - seed_rgb[0]).abs() / 255.0;
            let dg = (g - seed_rgb[1]).abs() / 255.0;
            let db = (b - seed_rgb[2]).abs() / 255.0;
            let max_d = dr.max(dg).max(db); // 0..1

            // `max_d` is the new alpha contribution from the color removal.
            // If max_d == 0, pixel is exactly the seed color → fully transparent.
            // If max_d == 1, pixel is maximally different → keep fully opaque.
            let mut removal = 1.0 - max_d; // how much to remove (1 = full removal)

            // For fringe pixels, fade the removal strength linearly
            if dist > 0 && smoothness > 0 {
                let fade = 1.0 - (dist as f32 / (smoothness as f32 + 1.0));
                removal *= fade;
            }

            removal = removal.clamp(0.0, 1.0);
            if removal < 0.004 {
                continue; // negligible change (< 1/255)
            }

            // New alpha = original alpha * (1 - removal)
            let new_a_f = (orig_a as f32 / 255.0) * (1.0 - removal);
            let new_a = (new_a_f * 255.0).round().clamp(0.0, 255.0) as u8;

            if new_a == 0 {
                // Fully removed
                results.push((x, y, [0, 0, 0, 0]));
                continue;
            }

            // RGB recovery: invert the premultiplication
            // new_color = (original - seed * removal) / new_alpha_ratio
            // where new_alpha_ratio = new_a / orig_a
            let kept = 1.0 - removal;
            let recover = |orig: f32, seed_ch: f32| -> u8 {
                // orig = seed_ch * removal + result * kept
                // result = (orig - seed_ch * removal) / kept
                if kept < 0.001 {
                    return orig as u8;
                }
                let val = (orig - seed_ch * removal) / kept;
                val.round().clamp(0.0, 255.0) as u8
            };

            let new_r = recover(r, seed_rgb[0]);
            let new_g = recover(g, seed_rgb[1]);
            let new_b = recover(b, seed_rgb[2]);

            results.push((x, y, [new_r, new_g, new_b, new_a]));
        }
    }

    results
}

/// Apply color removal results (full RGBA) to a flat image.
pub fn apply_color_removal(pixels: &mut RgbaImage, changes: &[(u32, u32, [u8; 4])]) {
    for &(x, y, rgba) in changes {
        *pixels.get_pixel_mut(x, y) = Rgba(rgba);
    }
}

/// Squared Euclidean distance in RGB space.
#[inline]
fn color_dist_sq(pixel: &Rgba<u8>, seed_rgb: &[f32; 3]) -> f32 {
    let dr = pixel[0] as f32 - seed_rgb[0];
    let dg = pixel[1] as f32 - seed_rgb[1];
    let db = pixel[2] as f32 - seed_rgb[2];
    dr * dr + dg * dg + db * db
}
