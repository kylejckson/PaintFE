// ============================================================================
// Inpainting / Content-Aware Fill algorithms
// ============================================================================

use image::{GrayImage, Rgba, RgbaImage};

// -- Quality levels -----------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ContentAwareQuality {
    /// Interactive — runs on each frame while painting. Weighted ring sampling.
    Instant,
    /// Async — PatchMatch-lite, 3 iterations, 5×5 patch (~0.5–2 s).
    Balanced,
    /// Async — PatchMatch, 6 iterations, 7×7 patch (~3–10 s).
    HighQuality,
}

impl ContentAwareQuality {
    pub fn label(&self) -> &'static str {
        match self {
            ContentAwareQuality::Instant => "Instant",
            ContentAwareQuality::Balanced => "Balanced",
            ContentAwareQuality::HighQuality => "High Quality",
        }
    }
    pub fn all() -> &'static [ContentAwareQuality] {
        &[
            ContentAwareQuality::Instant,
            ContentAwareQuality::Balanced,
            ContentAwareQuality::HighQuality,
        ]
    }
    /// Iterations to use for PatchMatch. 0 for Instant.
    pub fn patchmatch_iters(&self) -> usize {
        match self {
            ContentAwareQuality::Instant => 0,
            ContentAwareQuality::Balanced => 3,
            ContentAwareQuality::HighQuality => 6,
        }
    }
    /// Returns true for modes that should schedule an async PatchMatch job.
    pub fn is_async(&self) -> bool {
        !matches!(self, ContentAwareQuality::Instant)
    }
}

// -- Request type (passed to spawn_filter_job) -----------------------------

/// Encapsulates a deferred PatchMatch job produced on mouse-release.
#[derive(Clone, Debug)]
pub struct InpaintRequest {
    /// Original layer before the stroke (no brush marks).
    pub original_flat: RgbaImage,
    /// Binary mask: >0 = pixels to fill (union of all rasterized brush positions).
    pub hole_mask: GrayImage,
    /// Effective patch side length (odd, 3–11).
    pub patch_size: u32,
    /// PatchMatch iterations.
    pub iterations: usize,
    /// Layer index to replace.
    pub layer_idx: usize,
}

// -- Instant fill (per-frame, runs in-place on preview layer) --------------

/// Improved ring-sampling inpaint used for the interactive Instant pass.
///
/// For each pixel that lies inside `hole_mask`, samples `num_candidates`
/// points distributed across a spiral in the annulus [`inner_r`, `outer_r`].
/// Each candidate is weighted by colour similarity to the pixel's reference
/// value (from the unmodified source). Candidates that are themselves hole
/// pixels are skipped.
///
/// The result is written into `out` in-place.
pub fn inpaint_instant_brush(
    src: &RgbaImage, // source layer (unmodified original or preview snapshot)
    hole_mask: &GrayImage,
    out: &mut RgbaImage,
    cx: f32,
    cy: f32,
    brush_radius: f32,
    sample_radius: f32,
    hardness: f32,
) {
    let (w, h) = src.dimensions();
    let r = brush_radius.max(1.0);
    let inner_r = sample_radius * 0.25;
    let outer_r = sample_radius;
    let num_candidates: usize = 32;
    let sigma_color_sq = 50.0_f32 * 50.0_f32;

    let min_x = (cx - r).max(0.0) as u32;
    let max_x = ((cx + r).ceil() as u32).min(w.saturating_sub(1));
    let min_y = (cy - r).max(0.0) as u32;
    let max_y = ((cy + r).ceil() as u32).min(h.saturating_sub(1));

    for y in min_y..=max_y {
        for x in min_x..=max_x {
            // Only touch painted-over pixels
            if let Some(m) = hole_mask.get_pixel_checked(x, y) {
                if m.0[0] == 0 {
                    continue;
                }
            } else {
                continue;
            }

            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            let dist = (dx * dx + dy * dy).sqrt();
            if dist > r {
                continue;
            }

            // Geometric alpha (hardness-aware smoothstep)
            let t = (dist / r).clamp(0.0, 1.0);
            let hard_t = (hardness * 0.9 + 0.1).clamp(0.0, 1.0);
            let geom_alpha = if t < hard_t {
                1.0
            } else {
                let s = (t - hard_t) / (1.0 - hard_t + 1e-6);
                1.0 - s * s * (3.0 - 2.0 * s)
            };
            if geom_alpha < 0.01 {
                continue;
            }

            // Reference colour from the source (not from preview — avoids feedback)
            let ref_p = src.get_pixel(x, y);
            let ref_r = ref_p.0[0] as f32;
            let ref_g = ref_p.0[1] as f32;
            let ref_b = ref_p.0[2] as f32;

            let mut sum_r = 0.0_f32;
            let mut sum_g = 0.0_f32;
            let mut sum_b = 0.0_f32;
            let mut sum_a = 0.0_f32;
            let mut weight_total = 0.0_f32;

            for i in 0..num_candidates {
                // Uniform spiral: angle advances uniformly, radius spread linearly
                let angle = i as f32 * (std::f32::consts::TAU / num_candidates as f32);
                let rr =
                    inner_r + (outer_r - inner_r) * (i as f32 / (num_candidates - 1).max(1) as f32);
                let sx = (x as f32 + angle.cos() * rr).round() as i32;
                let sy = (y as f32 + angle.sin() * rr).round() as i32;
                if sx < 0 || sx >= w as i32 || sy < 0 || sy >= h as i32 {
                    continue;
                }
                let (ux, uy) = (sx as u32, sy as u32);

                // Skip if candidate is inside the hole
                if hole_mask.get_pixel(ux, uy).0[0] > 0 {
                    continue;
                }

                let sp = src.get_pixel(ux, uy);
                let dr = sp.0[0] as f32 - ref_r;
                let dg = sp.0[1] as f32 - ref_g;
                let db = sp.0[2] as f32 - ref_b;
                let w_color = (-(dr * dr + dg * dg + db * db) / sigma_color_sq).exp();
                sum_r += sp.0[0] as f32 * w_color;
                sum_g += sp.0[1] as f32 * w_color;
                sum_b += sp.0[2] as f32 * w_color;
                sum_a += sp.0[3] as f32 * w_color;
                weight_total += w_color;
            }

            if weight_total < 1e-6 {
                continue;
            }

            let filled_r = (sum_r / weight_total).clamp(0.0, 255.0) as u8;
            let filled_g = (sum_g / weight_total).clamp(0.0, 255.0) as u8;
            let filled_b = (sum_b / weight_total).clamp(0.0, 255.0) as u8;
            let _filled_a = (sum_a / weight_total).clamp(0.0, 255.0) as u8;

            // Blend with geom_alpha into the output
            let existing = out.get_pixel(x, y);
            let ea = existing.0[3] as f32 / 255.0;
            if geom_alpha >= ea {
                *out.get_pixel_mut(x, y) = Rgba([
                    lerp_u8(existing.0[0], filled_r, geom_alpha),
                    lerp_u8(existing.0[1], filled_g, geom_alpha),
                    lerp_u8(existing.0[2], filled_b, geom_alpha),
                    (geom_alpha * 255.0) as u8,
                ]);
            }
        }
    }
}

#[inline(always)]
fn lerp_u8(a: u8, b: u8, t: f32) -> u8 {
    (a as f32 + (b as f32 - a as f32) * t).clamp(0.0, 255.0) as u8
}

// -- PatchMatch inpainting ---------------------------------------------
//
// Core correctness principle: SSD must IGNORE hole pixels in the query patch.
// The old code compared hole content → found "more blemishes" to copy.
// Fixed: only border/context pixels participate in the comparison.
//
// Algorithm: onion-peeling + PatchMatch
//   Each outer pass fills the current boundary layer (hole pixels adjacent to
//   non-hole pixels) by finding the best-matching source patch via PatchMatch.
//   Filled pixels become source candidates for the next pass, so texture
//   propagates naturally from the outside inward.

/// Returns true if (x,y) is in the hole AND has at least one non-hole
/// direct (4-connected) neighbour.
#[inline]
fn is_boundary_hole(mask: &GrayImage, x: u32, y: u32) -> bool {
    if mask.get_pixel(x, y).0[0] == 0 {
        return false;
    }
    let (w, h) = mask.dimensions();
    for (dx, dy) in [(-1i32, 0), (1, 0), (0, -1i32), (0, 1)] {
        let nx = x as i32 + dx;
        let ny = y as i32 + dy;
        if nx >= 0
            && ny >= 0
            && nx < w as i32
            && ny < h as i32
            && mask.get_pixel(nx as u32, ny as u32).0[0] == 0
        {
            return true;
        }
    }
    false
}

/// Masked SSD: compares patches around `(ax,ay)` in `img` and `(bx,by)` in `img`.
/// Only counts pixels where BOTH positions are non-hole.
/// Returns f32::MAX if fewer than `min_valid` valid pixels are found.
#[inline]
fn patch_ssd_masked(
    img: &RgbaImage,
    mask: &GrayImage,
    ax: i32,
    ay: i32,
    bx: i32,
    by: i32,
    half: i32,
    min_valid: usize,
) -> f32 {
    let (w, h) = (img.width() as i32, img.height() as i32);
    let mut ssd = 0.0f32;
    let mut count = 0usize;
    for dy in -half..=half {
        for dx in -half..=half {
            let apx = ax + dx;
            let apy = ay + dy;
            let bpx = bx + dx;
            let bpy = by + dy;
            if apx < 0 || apy < 0 || apx >= w || apy >= h {
                continue;
            }
            if bpx < 0 || bpy < 0 || bpx >= w || bpy >= h {
                continue;
            }
            // Skip if the query pixel is in the hole (don't compare hole content)
            if mask.get_pixel(apx as u32, apy as u32).0[0] > 0 {
                continue;
            }
            // Skip if the candidate pixel is in the hole
            if mask.get_pixel(bpx as u32, bpy as u32).0[0] > 0 {
                continue;
            }
            let pa = img.get_pixel(apx as u32, apy as u32);
            let pb = img.get_pixel(bpx as u32, bpy as u32);
            for c in 0..3usize {
                let d = pa.0[c] as f32 - pb.0[c] as f32;
                ssd += d * d;
            }
            count += 1;
        }
    }
    if count < min_valid {
        f32::MAX
    } else {
        ssd / count as f32
    }
}

/// Run one PatchMatch pass over `pixels` (a set of hole pixels to refine).
/// Reads/writes NNF arrays in-place. Uses `img` as the image to sample from.
fn patchmatch_pass(
    img: &RgbaImage,
    mask: &GrayImage,
    pixels: &[(u32, u32)],
    nnf_ox: &mut [i32],
    nnf_oy: &mut [i32],
    nnf_ssd: &mut [f32],
    half: i32,
    min_valid: usize,
    max_radius: f32,
    iter: usize,
) {
    let (w, h) = (img.width() as i32, img.height() as i32);
    let forward = iter.is_multiple_of(2);

    let iter_order: Box<dyn Iterator<Item = &(u32, u32)>> = if forward {
        Box::new(pixels.iter())
    } else {
        Box::new(pixels.iter().rev())
    };

    for &(hx, hy) in iter_order {
        let idx = (hy * img.width() + hx) as usize;
        let mut best_ox = nnf_ox[idx];
        let mut best_oy = nnf_oy[idx];
        let mut best_ssd = nnf_ssd[idx];

        // Propagation: try offsets from spatial neighbours
        let neighbours: &[(i32, i32)] = if forward {
            &[(-1, 0), (0, -1)]
        } else {
            &[(1, 0), (0, 1)]
        };
        for &(ndx, ndy) in neighbours {
            let nx = hx as i32 + ndx;
            let ny = hy as i32 + ndy;
            if nx < 0 || ny < 0 || nx >= w || ny >= h {
                continue;
            }
            let ni = (ny as u32 * img.width() + nx as u32) as usize;
            if nnf_ssd[ni] == f32::MAX {
                continue;
            }
            let cx = hx as i32 + nnf_ox[ni];
            let cy = hy as i32 + nnf_oy[ni];
            if cx < 0 || cy < 0 || cx >= w || cy >= h {
                continue;
            }
            if mask.get_pixel(cx as u32, cy as u32).0[0] > 0 {
                continue;
            }
            let ssd = patch_ssd_masked(img, mask, hx as i32, hy as i32, cx, cy, half, min_valid);
            if ssd < best_ssd {
                best_ssd = ssd;
                best_ox = cx - hx as i32;
                best_oy = cy - hy as i32;
            }
        }

        // Random search with LCG PRNG — halving radius each step
        let mut rng = (hx as u64)
            .wrapping_mul(6364136223846793005)
            .wrapping_add((hy as u64).wrapping_mul(982451653))
            .wrapping_add(iter as u64 * 1234567891);
        let mut search_r = max_radius;
        while search_r >= 1.0 {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let ra = (rng >> 33) as f32 / (u32::MAX as f32);
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let rb = (rng >> 33) as f32 / (u32::MAX as f32);
            let cx = (hx as f32 + best_ox as f32 + (ra * 2.0 - 1.0) * search_r).round() as i32;
            let cy = (hy as f32 + best_oy as f32 + (rb * 2.0 - 1.0) * search_r).round() as i32;
            if cx >= 0
                && cy >= 0
                && cx < w
                && cy < h
                && mask.get_pixel(cx as u32, cy as u32).0[0] == 0
            {
                let ssd =
                    patch_ssd_masked(img, mask, hx as i32, hy as i32, cx, cy, half, min_valid);
                if ssd < best_ssd {
                    best_ssd = ssd;
                    best_ox = cx - hx as i32;
                    best_oy = cy - hy as i32;
                }
            }
            search_r *= 0.5;
        }

        nnf_ox[idx] = best_ox;
        nnf_oy[idx] = best_oy;
        nnf_ssd[idx] = best_ssd;
    }
}

/// Exemplar-based inpainting with onion-peeling + PatchMatch refinement.
///
/// Key correctness properties:
/// - SSD ignores hole pixels → matches surrounding context, not blemish content
/// - Fills boundary layer first, updates mask, repeats → texture grows inward coherently
/// - Each peeling pass uses previously-filled pixels as valid source candidates
pub fn fill_region_patchmatch(
    src: &RgbaImage,
    hole_mask: &GrayImage,
    patch_size: u32,
    iterations: usize,
) -> RgbaImage {
    let (w, h) = src.dimensions();
    let ps = patch_size.max(3) as i32;
    let half = ps / 2;
    let min_valid = ((half as usize * 2 + 1).pow(2)).max(4) / 4;
    let max_radius = w.max(h) as f32;
    let total = (w * h) as usize;

    let mut out = src.clone();
    let mut live_mask = hole_mask.clone();
    let mut nnf_ox = vec![0i32; total];
    let mut nnf_oy = vec![0i32; total];
    let mut nnf_ssd = vec![f32::MAX; total];

    // Collect initial source pixels for seeding the NNF
    let mut source_pixels: Vec<(u32, u32)> = (0..h)
        .flat_map(|y| (0..w).map(move |x| (x, y)))
        .filter(|&(x, y)| hole_mask.get_pixel(x, y).0[0] == 0)
        .collect();

    if source_pixels.is_empty() {
        return out;
    }

    // Onion-peeling loop: each pass fills one "layer" of the hole boundary
    let max_peeling_passes = (w.max(h) as usize + 1) * 2;
    for _peel in 0..max_peeling_passes {
        // Collect boundary hole pixels for this pass
        let boundary: Vec<(u32, u32)> = (0..h)
            .flat_map(|y| (0..w).map(move |x| (x, y)))
            .filter(|&(x, y)| is_boundary_hole(&live_mask, x, y))
            .collect();

        if boundary.is_empty() {
            break;
        }

        let src_count = source_pixels.len();

        // --- Random Init for boundary pixels ---
        for &(hx, hy) in &boundary {
            let seed = ((hx as usize).wrapping_mul(7919))
                .wrapping_add((hy as usize).wrapping_mul(6271))
                % src_count;
            let (sx, sy) = source_pixels[seed];
            let ssd = patch_ssd_masked(
                &out, &live_mask, hx as i32, hy as i32, sx as i32, sy as i32, half, min_valid,
            );
            let idx = (hy * w + hx) as usize;
            nnf_ox[idx] = sx as i32 - hx as i32;
            nnf_oy[idx] = sy as i32 - hy as i32;
            nnf_ssd[idx] = ssd;

            // Also try a few more random seeds to warm-start better
            let mut rng = (hx as u64)
                .wrapping_mul(1234567891)
                .wrapping_add(hy as u64 * 987654321);
            for _ in 0..4 {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                let si = (rng >> 33) as usize % src_count;
                let (tx, ty) = source_pixels[si];
                let s2 = patch_ssd_masked(
                    &out, &live_mask, hx as i32, hy as i32, tx as i32, ty as i32, half, min_valid,
                );
                if s2 < nnf_ssd[idx] {
                    nnf_ox[idx] = tx as i32 - hx as i32;
                    nnf_oy[idx] = ty as i32 - hy as i32;
                    nnf_ssd[idx] = s2;
                }
            }
        }

        // --- PatchMatch iterations ---
        let pm_iters = if iterations <= 3 { 2 } else { 4 };
        for iter in 0..pm_iters {
            patchmatch_pass(
                &out,
                &live_mask,
                &boundary,
                &mut nnf_ox,
                &mut nnf_oy,
                &mut nnf_ssd,
                half,
                min_valid,
                max_radius,
                iter,
            );
        }

        // --- Fill boundary pixels & update mask ---
        // Collect fills first to avoid borrow conflict (out is both source and dest)
        let fills: Vec<(u32, u32, image::Rgba<u8>)> = boundary
            .iter()
            .filter_map(|&(hx, hy)| {
                let idx = (hy * w + hx) as usize;
                if nnf_ssd[idx] == f32::MAX {
                    return None;
                }
                let sx = hx as i32 + nnf_ox[idx];
                let sy = hy as i32 + nnf_oy[idx];
                if sx < 0 || sy < 0 || sx >= w as i32 || sy >= h as i32 {
                    return None;
                }
                if live_mask.get_pixel(sx as u32, sy as u32).0[0] > 0 {
                    return None;
                }
                Some((hx, hy, *out.get_pixel(sx as u32, sy as u32)))
            })
            .collect();

        for (x, y, pixel) in fills {
            out.put_pixel(x, y, pixel);
        }

        for (x, y) in &boundary {
            live_mask.put_pixel(*x, *y, image::Luma([0u8]));
            source_pixels.push((*x, *y));
        }
    }

    out
}
