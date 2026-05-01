pub fn render_grid(
    state: &mut CanvasState,
    layer_idx: usize,
    cell_w: u32,
    cell_h: u32,
    line_width: u32,
    color: [u8; 4],
    style: GridStyle,
    opacity: f32,
) {
    if layer_idx >= state.layers.len() {
        return;
    }
    let flat = state.layers[layer_idx].pixels.to_rgba_image();
    let result = grid_core(
        &flat,
        cell_w,
        cell_h,
        line_width,
        color,
        style,
        opacity,
        state.selection_mask.as_ref(),
    );
    commit_to_layer(state, layer_idx, &result);
}

pub fn render_grid_from_flat(
    state: &mut CanvasState,
    layer_idx: usize,
    cell_w: u32,
    cell_h: u32,
    line_width: u32,
    color: [u8; 4],
    style: GridStyle,
    opacity: f32,
    original_flat: &RgbaImage,
) {
    let result = grid_core(
        original_flat,
        cell_w,
        cell_h,
        line_width,
        color,
        style,
        opacity,
        state.selection_mask.as_ref(),
    );
    commit_to_layer(state, layer_idx, &result);
}

pub fn grid_core(
    flat: &RgbaImage,
    cell_w: u32,
    cell_h: u32,
    line_width: u32,
    color: [u8; 4],
    style: GridStyle,
    opacity: f32,
    mask: Option<&GrayImage>,
) -> RgbaImage {
    let cw = cell_w.max(2);
    let ch = cell_h.max(2);
    let lw = line_width.max(1);

    apply_per_pixel(flat, mask, |x, y, r, g, b, a| {
        let draw = match style {
            GridStyle::Lines => (x % cw) < lw || (y % ch) < lw,
            GridStyle::Checkerboard => {
                let cell_x = x / cw;
                let cell_y = y / ch;
                (cell_x + cell_y).is_multiple_of(2)
            }
        };

        if draw {
            let t = opacity;
            let gr = color[0] as f32;
            let gg = color[1] as f32;
            let gb = color[2] as f32;
            let ga = color[3] as f32;
            (
                r * (1.0 - t) + gr * t,
                g * (1.0 - t) + gg * t,
                b * (1.0 - t) + gb * t,
                a * (1.0 - t) + ga * t,
            )
        } else {
            (r, g, b, a)
        }
    })
}

pub fn canvas_border(state: &mut CanvasState, layer_idx: usize, width: u32, color: [u8; 4]) {
    if layer_idx >= state.layers.len() {
        return;
    }
    let flat = state.layers[layer_idx].pixels.to_rgba_image();
    let result = canvas_border_core(&flat, width, color, state.selection_mask.as_ref());
    commit_to_layer(state, layer_idx, &result);
}

pub fn canvas_border_from_flat(
    state: &mut CanvasState,
    layer_idx: usize,
    width: u32,
    color: [u8; 4],
    original_flat: &RgbaImage,
) {
    let result = canvas_border_core(original_flat, width, color, state.selection_mask.as_ref());
    commit_to_layer(state, layer_idx, &result);
}

pub fn canvas_border_core(
    flat: &RgbaImage,
    width: u32,
    color: [u8; 4],
    mask: Option<&GrayImage>,
) -> RgbaImage {
    let w = flat.width();
    let h = flat.height();
    if w == 0 || h == 0 {
        return flat.clone();
    }

    let border_w = width.max(1).min(w.min(h));
    let src_raw = flat.as_raw();
    let mut dst_raw = src_raw.clone();

    let mask_raw = mask.map(|m| m.as_raw().as_slice());
    let mask_w = mask.map_or(0, |m| m.width() as usize);
    let mask_h = mask.map_or(0, |m| m.height() as usize);
    let stride = w as usize * 4;

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

                let x_u = x as u32;
                let y_u = y as u32;
                let is_border =
                    x_u < border_w || y_u < border_w || x_u >= w - border_w || y_u >= h - border_w;
                if !is_border {
                    continue;
                }

                let pi = x * 4;
                row_out[pi] = color[0];
                row_out[pi + 1] = color[1];
                row_out[pi + 2] = color[2];
                row_out[pi + 3] = color[3];
            }
        });

    RgbaImage::from_raw(w, h, dst_raw).unwrap()
}

// --- Drop Shadow ---

pub fn drop_shadow(
    state: &mut CanvasState,
    layer_idx: usize,
    offset_x: i32,
    offset_y: i32,
    blur_radius: f32,
    widen_radius: bool,
    color: [u8; 4],
    opacity: f32,
) {
    if layer_idx >= state.layers.len() {
        return;
    }
    let flat = state.layers[layer_idx].pixels.to_rgba_image();
    let result = shadow_core(
        &flat,
        offset_x,
        offset_y,
        blur_radius,
        widen_radius,
        color,
        opacity,
        state.selection_mask.as_ref(),
    );
    commit_to_layer(state, layer_idx, &result);
}

pub fn drop_shadow_from_flat(
    state: &mut CanvasState,
    layer_idx: usize,
    offset_x: i32,
    offset_y: i32,
    blur_radius: f32,
    widen_radius: bool,
    color: [u8; 4],
    opacity: f32,
    original_flat: &RgbaImage,
) {
    let result = shadow_core(
        original_flat,
        offset_x,
        offset_y,
        blur_radius,
        widen_radius,
        color,
        opacity,
        state.selection_mask.as_ref(),
    );
    commit_to_layer(state, layer_idx, &result);
}

pub fn shadow_core(
    flat: &RgbaImage,
    offset_x: i32,
    offset_y: i32,
    blur_radius: f32,
    widen_radius: bool,
    color: [u8; 4],
    opacity: f32,
    mask: Option<&GrayImage>,
) -> RgbaImage {
    let w = flat.width();
    let h = flat.height();

    // 1. Create alpha mask from source (offset).
    let mut shadow_alpha = vec![0u8; (w * h) as usize];
    let src_raw = flat.as_raw();
    let stride = w as usize * 4;

    for y in 0..h as i32 {
        for x in 0..w as i32 {
            let sx = x - offset_x;
            let sy = y - offset_y;
            if sx >= 0 && sx < w as i32 && sy >= 0 && sy < h as i32 {
                let si = sy as usize * stride + sx as usize * 4;
                shadow_alpha[y as usize * w as usize + x as usize] = src_raw[si + 3];
            }
        }
    }

    // 2. Optional widening/spread pass before blur.
    if widen_radius {
        let spread = blur_radius.max(1.0).round() as i32;
        if spread > 0 {
            let wu = w as usize;
            let hu = h as usize;
            let r = spread as usize;
            let src = shadow_alpha;
            let mut horizontal = vec![0u8; src.len()];
            horizontal
                .par_chunks_mut(wu)
                .enumerate()
                .for_each(|(y, row)| {
                    let base = y * wu;
                    for (x, cell) in row.iter_mut().enumerate().take(wu) {
                        let x0 = x.saturating_sub(r);
                        let x1 = (x + r).min(wu - 1);
                        let mut max_a = 0u8;
                        for sx in x0..=x1 {
                            max_a = max_a.max(src[base + sx]);
                        }
                        *cell = max_a;
                    }
                });
            shadow_alpha = vec![0u8; horizontal.len()];
            shadow_alpha
                .par_chunks_mut(wu)
                .enumerate()
                .for_each(|(y, row)| {
                    let y0 = y.saturating_sub(r);
                    let y1 = (y + r).min(hu - 1);
                    for x in 0..wu {
                        let mut max_a = 0u8;
                        for sy in y0..=y1 {
                            max_a = max_a.max(horizontal[sy * wu + x]);
                        }
                        row[x] = max_a;
                    }
                });
        }
    }

    // 3. Blur the alpha mask.
    let alpha_img = GrayImage::from_raw(w, h, shadow_alpha).unwrap();
    let alpha_rgba = RgbaImage::from_fn(w, h, |x, y| {
        let a = alpha_img.get_pixel(x, y)[0];
        Rgba([a, a, a, a])
    });
    let blurred_alpha_rgba = if blur_radius > 0.5 {
        crate::ops::filters::parallel_gaussian_blur_pub(&alpha_rgba, blur_radius)
    } else {
        alpha_rgba
    };

    // 4. Composite: shadow underneath, original on top.
    let mask_raw_sel = mask.map(|m| m.as_raw().as_slice());
    let mask_w = mask.map_or(0, |m| m.width() as usize);
    let mask_h = mask.map_or(0, |m| m.height() as usize);
    let blur_raw = blurred_alpha_rgba.as_raw();
    let mut dst_raw = vec![0u8; (w * h * 4) as usize];

    dst_raw
        .par_chunks_mut(stride)
        .enumerate()
        .for_each(|(y, row_out)| {
            for x in 0..w as usize {
                let pi = x * 4;
                if let Some(mr) = mask_raw_sel
                    && x < mask_w
                    && y < mask_h
                    && mr[y * mask_w + x] == 0
                {
                    let si = y * stride + pi;
                    row_out[pi..pi + 4].copy_from_slice(&src_raw[si..si + 4]);
                    continue;
                }
                let si = y * stride + pi;
                let shadow_a = (blur_raw[y * stride + pi] as f32 / 255.0)
                    * opacity
                    * (color[3] as f32 / 255.0);
                let src_a = src_raw[si + 3] as f32 / 255.0;

                // Shadow first, then source on top, written back as straight RGBA.
                let out_a = src_a + shadow_a * (1.0 - src_a);
                for c in 0..3 {
                    let shadow_c = color[c] as f32 / 255.0;
                    let src_c = src_raw[si + c] as f32 / 255.0;
                    let out_c = if out_a > 0.0 {
                        (src_c * src_a + shadow_c * shadow_a * (1.0 - src_a)) / out_a
                    } else {
                        0.0
                    };
                    let out_c = out_c * 255.0;
                    row_out[pi + c] = out_c.round().clamp(0.0, 255.0) as u8;
                }
                row_out[pi + 3] = (out_a * 255.0).round().clamp(0.0, 255.0) as u8;
            }
        });

    RgbaImage::from_raw(w, h, dst_raw).unwrap()
}

// --- Outline ---

#[derive(Clone, Copy, PartialEq)]
pub enum OutlineMode {
    Outside,
    Inside,
    Center,
}

pub fn outline(
    state: &mut CanvasState,
    layer_idx: usize,
    width: u32,
    color: [u8; 4],
    mode: OutlineMode,
    anti_alias: bool,
) {
    if layer_idx >= state.layers.len() {
        return;
    }
    let flat = state.layers[layer_idx].pixels.to_rgba_image();
    let result = outline_core(
        &flat,
        width,
        color,
        mode,
        anti_alias,
        state.selection_mask.as_ref(),
    );
    commit_to_layer(state, layer_idx, &result);
}

pub fn outline_from_flat(
    state: &mut CanvasState,
    layer_idx: usize,
    width: u32,
    color: [u8; 4],
    mode: OutlineMode,
    anti_alias: bool,
    original_flat: &RgbaImage,
) {
    let result = outline_core(
        original_flat,
        width,
        color,
        mode,
        anti_alias,
        state.selection_mask.as_ref(),
    );
    commit_to_layer(state, layer_idx, &result);
}

pub fn outline_core(
    flat: &RgbaImage,
    width: u32,
    color: [u8; 4],
    mode: OutlineMode,
    anti_alias: bool,
    mask: Option<&GrayImage>,
) -> RgbaImage {
    let w = flat.width() as usize;
    let h = flat.height() as usize;
    if w == 0 || h == 0 {
        return flat.clone();
    }

    let radius = width.max(1) as f32;
    let search_radius = radius.ceil() as i32 + 1;
    let src_raw = flat.as_raw();
    let stride = w * 4;
    let alpha: Vec<u8> = (0..w * h).map(|i| src_raw[i * 4 + 3]).collect();
    let mut alpha_min_x = w;
    let mut alpha_min_y = h;
    let mut alpha_max_x = 0usize;
    let mut alpha_max_y = 0usize;
    for (idx, a) in alpha.iter().enumerate() {
        if *a > 0 {
            let x = idx % w;
            let y = idx / w;
            alpha_min_x = alpha_min_x.min(x);
            alpha_min_y = alpha_min_y.min(y);
            alpha_max_x = alpha_max_x.max(x);
            alpha_max_y = alpha_max_y.max(y);
        }
    }
    if alpha_min_x == w {
        return flat.clone();
    }
    let proc_min_x = alpha_min_x.saturating_sub(search_radius as usize + 1);
    let proc_min_y = alpha_min_y.saturating_sub(search_radius as usize + 1);
    let proc_max_x = (alpha_max_x + search_radius as usize + 1).min(w - 1);
    let proc_max_y = (alpha_max_y + search_radius as usize + 1).min(h - 1);

    let mask_raw = mask.map(|m| m.as_raw().as_slice());
    let mask_w = mask.map_or(0, |m| m.width() as usize);
    let mask_h = mask.map_or(0, |m| m.height() as usize);
    let mut dst_raw = src_raw.to_vec();

    let shell_coverage = |distance: f32| {
        if anti_alias {
            let t = ((radius + 0.5 - distance) / 1.0).clamp(0.0, 1.0);
            t * t * (3.0 - 2.0 * t)
        } else if distance <= radius {
            1.0
        } else {
            0.0
        }
    };

    let nearest_distance = |x: usize, y: usize, want_filled: bool| -> Option<f32> {
        let mut best_sq: Option<i32> = None;
        for dy in -search_radius..=search_radius {
            for dx in -search_radius..=search_radius {
                let dist_sq = dx * dx + dy * dy;
                let current_best = best_sq.unwrap_or(i32::MAX);
                if dist_sq > current_best {
                    continue;
                }

                let sx = x as i32 + dx;
                let sy = y as i32 + dy;
                if sx < 0 || sy < 0 || sx >= w as i32 || sy >= h as i32 {
                    continue;
                }

                let sample_alpha = alpha[sy as usize * w + sx as usize];
                let matches = if want_filled {
                    sample_alpha > 0
                } else {
                    sample_alpha == 0
                };
                if matches {
                    best_sq = Some(dist_sq);
                }
            }
        }
        best_sq.map(|dist_sq| (dist_sq as f32).sqrt())
    };

    dst_raw
        .par_chunks_mut(stride)
        .enumerate()
        .for_each(|(y, row_out)| {
            for x in 0..w {
                let pi = x * 4;
                if x < proc_min_x || x > proc_max_x || y < proc_min_y || y > proc_max_y {
                    continue;
                }
                if let Some(mr) = mask_raw
                    && x < mask_w
                    && y < mask_h
                    && mr[y * mask_w + x] == 0
                {
                    continue;
                }

                let idx = y * w + x;
                let src_a = alpha[idx] as f32 / 255.0;
                let outside_cov = nearest_distance(x, y, true)
                    .map(|distance| shell_coverage((distance - 1.0).max(0.0)))
                    .unwrap_or(0.0)
                    * (1.0 - src_a);
                let inside_cov = nearest_distance(x, y, false)
                    .map(shell_coverage)
                    .unwrap_or(0.0)
                    * src_a;

                let (under_cov, over_cov) = match mode {
                    OutlineMode::Outside => (outside_cov, 0.0),
                    OutlineMode::Inside => (0.0, inside_cov),
                    OutlineMode::Center => (outside_cov, inside_cov),
                };

                let outline_a_under = (color[3] as f32 / 255.0) * under_cov;
                let outline_a_over = (color[3] as f32 / 255.0) * over_cov;

                let mut comp_r = row_out[pi] as f32 / 255.0;
                let mut comp_g = row_out[pi + 1] as f32 / 255.0;
                let mut comp_b = row_out[pi + 2] as f32 / 255.0;
                let mut comp_a = row_out[pi + 3] as f32 / 255.0;

                if outline_a_under > 0.0 {
                    let out_a = comp_a + outline_a_under * (1.0 - comp_a);
                    if out_a > 0.0 {
                        comp_r = (comp_r * comp_a
                            + (color[0] as f32 / 255.0) * outline_a_under * (1.0 - comp_a))
                            / out_a;
                        comp_g = (comp_g * comp_a
                            + (color[1] as f32 / 255.0) * outline_a_under * (1.0 - comp_a))
                            / out_a;
                        comp_b = (comp_b * comp_a
                            + (color[2] as f32 / 255.0) * outline_a_under * (1.0 - comp_a))
                            / out_a;
                    }
                    comp_a = out_a;
                }

                if outline_a_over > 0.0 {
                    let out_a = outline_a_over + comp_a * (1.0 - outline_a_over);
                    if out_a > 0.0 {
                        comp_r = ((color[0] as f32 / 255.0) * outline_a_over
                            + comp_r * comp_a * (1.0 - outline_a_over))
                            / out_a;
                        comp_g = ((color[1] as f32 / 255.0) * outline_a_over
                            + comp_g * comp_a * (1.0 - outline_a_over))
                            / out_a;
                        comp_b = ((color[2] as f32 / 255.0) * outline_a_over
                            + comp_b * comp_a * (1.0 - outline_a_over))
                            / out_a;
                    }
                    comp_a = out_a;
                }

                row_out[pi] = (comp_r.clamp(0.0, 1.0) * 255.0).round() as u8;
                row_out[pi + 1] = (comp_g.clamp(0.0, 1.0) * 255.0).round() as u8;
                row_out[pi + 2] = (comp_b.clamp(0.0, 1.0) * 255.0).round() as u8;
                row_out[pi + 3] = (comp_a.clamp(0.0, 1.0) * 255.0).round() as u8;
            }
        });

    RgbaImage::from_raw(w as u32, h as u32, dst_raw).unwrap()
}

// ============================================================================
// GLITCH EFFECTS
// ============================================================================

// --- Pixel Drag ---

