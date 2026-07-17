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

    // Build one horizontal span per kernel row. Maintaining a sliding sum for
    // each span reduces a disc convolution from O(radius² * pixels) to
    // O(radius * pixels), while preserving the exact equal-weight disc kernel.
    let r = radius.ceil() as i32;
    let r2 = radius * radius;
    let mut spans: Vec<(i32, i32)> = Vec::new();
    let mut sample_count = 0usize;
    for dy in -r..=r {
        let remaining = r2 - (dy * dy) as f32;
        if remaining >= 0.0 {
            let span = remaining.sqrt().floor() as i32;
            spans.push((dy, span));
            sample_count += (span * 2 + 1) as usize;
        }
    }
    let inv_count = 1.0 / sample_count as f32;

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
            let mut row_sums = vec![[0u32; 4]; spans.len()];
            for (span_index, &(dy, span)) in spans.iter().enumerate() {
                let sy = (y as i32 + dy).clamp(0, h as i32 - 1) as usize;
                for dx in -span..=span {
                    let sx = dx.clamp(0, w as i32 - 1) as usize;
                    let si = sy * stride + sx * 4;
                    for c in 0..4 {
                        row_sums[span_index][c] += src_raw[si + c] as u32;
                    }
                }
            }
            for x in 0..w {
                let pi = x * 4;
                let masked_out = if let Some(mr) = mask_raw
                    && x < mask_w
                    && y < mask_h
                    && mr[y * mask_w + x] == 0
                {
                    let src_off = y * stride + pi;
                    row_out[pi..pi + 4].copy_from_slice(&src_raw[src_off..src_off + 4]);
                    true
                } else {
                    false
                };
                if !masked_out {
                    let totals = row_sums.iter().fold([0u64; 4], |mut total, sum| {
                        for c in 0..4 {
                            total[c] += sum[c] as u64;
                        }
                        total
                    });
                    for c in 0..4 {
                        row_out[pi + c] = (totals[c] as f32 * inv_count)
                            .round()
                            .clamp(0.0, 255.0) as u8;
                    }
                }
                if x + 1 < w {
                    for (span_index, &(dy, span)) in spans.iter().enumerate() {
                        let sy = (y as i32 + dy).clamp(0, h as i32 - 1) as usize;
                        let remove_x = (x as i32 - span).clamp(0, w as i32 - 1) as usize;
                        let add_x = (x as i32 + span + 1).clamp(0, w as i32 - 1) as usize;
                        let remove_i = sy * stride + remove_x * 4;
                        let add_i = sy * stride + add_x * 4;
                        for c in 0..4 {
                            row_sums[span_index][c] = row_sums[span_index][c]
                                - src_raw[remove_i + c] as u32
                                + src_raw[add_i + c] as u32;
                        }
                    }
                }
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
    let divisor = kernel_size as u32;
    let src_raw = flat.as_raw();

    // Sliding-window separable blur: O(width * height), independent of radius.
    // Keep the intermediate as bytes to halve peak memory versus two f32 frames.
    let mut h_buf = vec![0u8; w * h * 4];
    h_buf
        .par_chunks_mut(w * 4)
        .enumerate()
        .for_each(|(y, row_out)| {
            let row = &src_raw[y * w * 4..(y + 1) * w * 4];
            let mut sums = [0u32; 4];
            for k in 0..kernel_size {
                let sx = (k as i32 - r as i32).clamp(0, w as i32 - 1) as usize;
                for c in 0..4 {
                    sums[c] += row[sx * 4 + c] as u32;
                }
            }
            for x in 0..w {
                let oi = x * 4;
                for c in 0..4 {
                    row_out[oi + c] = ((sums[c] + divisor / 2) / divisor) as u8;
                }
                if x + 1 < w {
                    let remove_x = (x as i32 - r as i32).clamp(0, w as i32 - 1) as usize;
                    let add_x = (x as i32 + r as i32 + 1).clamp(0, w as i32 - 1) as usize;
                    for c in 0..4 {
                        sums[c] = sums[c] - row[remove_x * 4 + c] as u32
                            + row[add_x * 4 + c] as u32;
                    }
                }
            }
        });

    // Vertical sliding pass, applying the selection mask directly into output.
    let mask_raw = mask.map(|m| m.as_raw().as_slice());
    let mask_w = mask.map_or(0, |m| m.width() as usize);
    let mask_h = mask.map_or(0, |m| m.height() as usize);
    let mut dst_raw = vec![0u8; w * h * 4];
    for x in 0..w {
        let mut sums = [0u32; 4];
        for k in 0..kernel_size {
            let sy = (k as i32 - r as i32).clamp(0, h as i32 - 1) as usize;
            let si = (sy * w + x) * 4;
            for c in 0..4 {
                sums[c] += h_buf[si + c] as u32;
            }
        }
        for y in 0..h {
            let oi = (y * w + x) * 4;
            if mask_raw.is_some_and(|mr| {
                x < mask_w && y < mask_h && mr[y * mask_w + x] == 0
            }) {
                dst_raw[oi..oi + 4].copy_from_slice(&src_raw[oi..oi + 4]);
            } else {
                for c in 0..4 {
                    dst_raw[oi + c] = ((sums[c] + divisor / 2) / divisor) as u8;
                }
            }
            if y + 1 < h {
                let remove_y = (y as i32 - r as i32).clamp(0, h as i32 - 1) as usize;
                let add_y = (y as i32 + r as i32 + 1).clamp(0, h as i32 - 1) as usize;
                let remove_i = (remove_y * w + x) * 4;
                let add_i = (add_y * w + x) * 4;
                for c in 0..4 {
                    sums[c] = sums[c] - h_buf[remove_i + c] as u32
                        + h_buf[add_i + c] as u32;
                }
            }
        }
    }

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

