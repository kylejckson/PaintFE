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
    bulge_core_at(flat, amount, (0.5, 0.5), mask)
}

pub fn bulge_core_at(
    flat: &RgbaImage,
    amount: f32,
    origin: (f32, f32),
    mask: Option<&GrayImage>,
) -> RgbaImage {
    let w = flat.width() as f32;
    let h = flat.height() as f32;
    let cx = origin.0.clamp(0.0, 1.0) * (w - 1.0).max(0.0);
    let cy = origin.1.clamp(0.0, 1.0) * (h - 1.0).max(0.0);
    let max_r = cx.max(w - cx).max(cy.max(h - cy)).max(1.0);
    let strength = amount.abs().max(0.0001);

    apply_per_pixel(flat, mask, |x, y, _r, _g, _b, _a| {
        let dx = x as f32 - cx;
        let dy = y as f32 - cy;
        let dist = (dx * dx + dy * dy).sqrt();
        let norm = (dist / max_r).min(1.0);

        if norm >= 1.0 {
            let p = sample_clamped(flat, x as i32, y as i32);
            return (p[0], p[1], p[2], p[3]);
        }

        let falloff = 1.0 - norm;
        let factor = if amount > 0.0 {
            1.0 - falloff * strength * 0.5
        } else if amount < 0.0 {
            1.0 + falloff * strength * 0.5
        } else {
            1.0
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
    twist_core_at(flat, angle_deg, (0.5, 0.5), mask)
}

pub fn twist_core_at(
    flat: &RgbaImage,
    angle_deg: f32,
    origin: (f32, f32),
    mask: Option<&GrayImage>,
) -> RgbaImage {
    let w = flat.width() as f32;
    let h = flat.height() as f32;
    let cx = origin.0.clamp(0.0, 1.0) * (w - 1.0).max(0.0);
    let cy = origin.1.clamp(0.0, 1.0) * (h - 1.0).max(0.0);
    let max_r = ((cx.max(w - cx)).powi(2) + (cy.max(h - cy)).powi(2))
        .sqrt()
        .max(1.0);
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

