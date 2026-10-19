fn apply_text_effects(text_rgba: &[u8], w: u32, h: u32, effects: &TextEffects) -> Vec<u8> {
    let count = (w as usize) * (h as usize);
    let coverage = extract_coverage_mask(text_rgba, w, h);

    // Start with a transparent output buffer
    let mut output = vec![0u8; count * 4];

    // 1. Shadow (behind everything)
    if let Some(ref shadow) = effects.shadow {
        render_shadow(&coverage, w, h, shadow, &mut output);
    }

    // 2. Outline (behind filled text, in front of shadow)
    if let Some(ref outline) = effects.outline
        && (outline.position == OutlinePosition::Outside
            || outline.position == OutlinePosition::Center)
    {
        render_outline(&coverage, w, h, outline, &mut output);
    }

    // 3. Text fill (flat color, gradient, or texture)
    if let Some(ref gradient) = effects.gradient_fill {
        render_gradient_fill(&coverage, w, h, gradient, &mut output);
    } else if let Some(ref tex) = effects.texture_fill {
        render_texture_fill(text_rgba, &coverage, w, h, tex, &mut output);
    } else {
        // Normal text fill — composite text onto output
        composite_over(text_rgba, &mut output, count);
    }

    // 4. Inside outline (on top of text fill)
    if let Some(ref outline) = effects.outline
        && outline.position == OutlinePosition::Inside
    {
        render_outline_inside(&coverage, w, h, outline, &mut output);
    }

    // 5. Inner shadow (on top of everything, clipped to text shape)
    if let Some(ref inner) = effects.inner_shadow {
        render_inner_shadow(&coverage, w, h, inner, &mut output);
    }

    output
}

/// Composite src over dst (premultiplied-aware alpha blending on straight-alpha buffers).
fn composite_over(src: &[u8], dst: &mut [u8], pixel_count: usize) {
    for i in 0..pixel_count {
        let si = i * 4;
        let sa = src[si + 3] as u32;
        if sa == 0 {
            continue;
        }
        if sa == 255 {
            dst[si..si + 4].copy_from_slice(&src[si..si + 4]);
            continue;
        }
        let da = dst[si + 3] as u32;
        let inv_sa = 255 - sa;
        let out_a = sa + (da * inv_sa) / 255;
        if out_a == 0 {
            continue;
        }
        for c in 0..3 {
            let sc = src[si + c] as u32;
            let dc = dst[si + c] as u32;
            dst[si + c] = ((sc * sa + dc * da * inv_sa / 255) / out_a).min(255) as u8;
        }
        dst[si + 3] = out_a.min(255) as u8;
    }
}

// ---------------------------------------------------------------------------
// Outline Effect
// ---------------------------------------------------------------------------

/// Compute a distance field from a coverage mask, then render the outline.
/// For Outside: dilated mask minus original = outline ring.
/// For Center: half-dilated in both directions.
fn render_outline(coverage: &[f32], w: u32, h: u32, outline: &OutlineEffect, output: &mut [u8]) {
    let radius = match outline.position {
        OutlinePosition::Outside => outline.width,
        OutlinePosition::Center => outline.width * 0.5,
        OutlinePosition::Inside => return, // handled separately
    };
    if radius <= 0.0 {
        return;
    }

    // Dilate the coverage mask by `radius` using a distance-field approach.
    let dilated = dilate_mask(coverage, w, h, radius);
    let count = (w as usize) * (h as usize);
    let [or, og, ob, oa] = outline.color;

    // Outline = dilated - original coverage (smooth edges)
    for i in 0..count {
        let outline_alpha = (dilated[i] - coverage[i]).clamp(0.0, 1.0) * (oa as f32 / 255.0);
        if outline_alpha < 1.0 / 255.0 {
            continue;
        }
        let sa = (outline_alpha * 255.0).round() as u32;
        let si = i * 4;
        let da = output[si + 3] as u32;
        let inv_sa = 255 - sa;
        let out_a = sa + (da * inv_sa) / 255;
        if out_a == 0 {
            continue;
        }
        for (c, &sc) in [or, og, ob].iter().enumerate() {
            let dc = output[si + c] as u32;
            output[si + c] = ((sc as u32 * sa + dc * da * inv_sa / 255) / out_a).min(255) as u8;
        }
        output[si + 3] = out_a.min(255) as u8;
    }
}

/// Render an inside outline: erode the mask, then the outline is original minus eroded.
fn render_outline_inside(
    coverage: &[f32],
    w: u32,
    h: u32,
    outline: &OutlineEffect,
    output: &mut [u8],
) {
    let radius = match outline.position {
        OutlinePosition::Inside => outline.width,
        OutlinePosition::Center => outline.width * 0.5,
        _ => return,
    };
    if radius <= 0.0 {
        return;
    }

    // Erode the mask: dilate the inverted mask, then invert back
    let count = (w as usize) * (h as usize);
    let inverted: Vec<f32> = coverage.iter().map(|&c| 1.0 - c).collect();
    let dilated_inv = dilate_mask(&inverted, w, h, radius);
    let eroded: Vec<f32> = dilated_inv.iter().map(|&d| (1.0 - d).max(0.0)).collect();

    let [or, og, ob, oa] = outline.color;

    // Inside outline = original - eroded, clipped to text shape
    for i in 0..count {
        let outline_alpha = (coverage[i] - eroded[i]).clamp(0.0, 1.0) * (oa as f32 / 255.0);
        if outline_alpha < 1.0 / 255.0 {
            continue;
        }
        let sa = (outline_alpha * 255.0).round() as u32;
        let si = i * 4;
        let da = output[si + 3] as u32;
        let inv_sa = 255 - sa;
        let out_a = sa + (da * inv_sa) / 255;
        if out_a == 0 {
            continue;
        }
        for (c, &sc) in [or, og, ob].iter().enumerate() {
            let dc = output[si + c] as u32;
            output[si + c] = ((sc as u32 * sa + dc * da * inv_sa / 255) / out_a).min(255) as u8;
        }
        output[si + 3] = out_a.min(255) as u8;
    }
}

/// Dilate a coverage mask by the given radius (in pixels).
/// Uses a fast box-filter approximation (3-pass box blur on the binary mask)
/// which gives a good approximation to a circular dilation for moderate radii.
fn dilate_mask(mask: &[f32], w: u32, h: u32, radius: f32) -> Vec<f32> {
    let ww = w as usize;
    let hh = h as usize;
    let count = ww * hh;

    if radius <= 0.0 {
        return mask.to_vec();
    }

    let int_radius = radius.ceil() as usize;
    if int_radius == 0 {
        return mask.to_vec();
    }

    let r_sq = radius * radius;

    // Euclidean-distance circular dilation: for each pixel, find the maximum
    // coverage value within a circular kernel of the given radius. This
    // preserves the original anti-aliased coverage values, producing smooth
    // outlines without binary thresholding or rectangular artifacts.
    let mut out = vec![0.0f32; count];
    out.par_chunks_mut(ww).enumerate().for_each(|(y, row_out)| {
        let y_start = y.saturating_sub(int_radius);
        let y_end = (y + int_radius + 1).min(hh);
        for (x, out_val) in row_out.iter_mut().enumerate() {
            let x_start = x.saturating_sub(int_radius);
            let x_end = (x + int_radius + 1).min(ww);
            let mut max_val = 0.0f32;
            for sy in y_start..y_end {
                let dy = sy as f32 - y as f32;
                let dy_sq = dy * dy;
                if dy_sq > r_sq {
                    continue;
                }
                let row_off = sy * ww;
                for sx in x_start..x_end {
                    let dx = sx as f32 - x as f32;
                    if dx * dx + dy_sq <= r_sq {
                        max_val = max_val.max(mask[row_off + sx]);
                    }
                }
            }
            *out_val = max_val;
        }
    });

    out
}

// ---------------------------------------------------------------------------
// Shadow Effect
// ---------------------------------------------------------------------------

/// Render a drop shadow: offset the coverage mask, optionally spread (dilate),
/// then apply Gaussian blur, tinted with the shadow color.
fn render_shadow(coverage: &[f32], w: u32, h: u32, shadow: &ShadowEffect, output: &mut [u8]) {
    let ww = w as usize;
    let hh = h as usize;
    let count = ww * hh;

    // Create an offset mask
    let mut offset_mask = vec![0.0f32; count];
    let dx = shadow.offset_x.round() as i32;
    let dy = shadow.offset_y.round() as i32;

    for y in 0..hh {
        let sy = y as i32 - dy;
        if sy < 0 || sy >= hh as i32 {
            continue;
        }
        for x in 0..ww {
            let sx = x as i32 - dx;
            if sx < 0 || sx >= ww as i32 {
                continue;
            }
            offset_mask[y * ww + x] = coverage[sy as usize * ww + sx as usize];
        }
    }

    // Spread (dilate) if needed
    if shadow.spread > 0.5 {
        offset_mask = dilate_mask(&offset_mask, w, h, shadow.spread);
    }

    // Apply Gaussian blur using image crate
    if shadow.blur_radius > 0.5 {
        // Convert mask to RGBA for the blur function
        let [sr, sg, sb, sa] = shadow.color;
        let mut shadow_rgba = vec![0u8; count * 4];
        for (i, &mask_val) in offset_mask.iter().enumerate().take(count) {
            let alpha = (mask_val * sa as f32).round().clamp(0.0, 255.0) as u8;
            shadow_rgba[i * 4] = sr;
            shadow_rgba[i * 4 + 1] = sg;
            shadow_rgba[i * 4 + 2] = sb;
            shadow_rgba[i * 4 + 3] = alpha;
        }
        let shadow_img = RgbaImage::from_raw(w, h, shadow_rgba).unwrap();
        let blurred =
            crate::ops::filters::parallel_gaussian_blur_pub(&shadow_img, shadow.blur_radius);
        let blurred_raw = blurred.as_raw();

        // Composite blurred shadow onto output
        composite_over(blurred_raw, output, count);
    } else {
        // No blur — just composite the solid shadow color with offset mask alpha
        let [sr, sg, sb, sa] = shadow.color;
        for (i, &mask_val) in offset_mask.iter().enumerate().take(count) {
            let alpha = (mask_val * sa as f32).round().clamp(0.0, 255.0) as u32;
            if alpha == 0 {
                continue;
            }
            let si = i * 4;
            let da = output[si + 3] as u32;
            let inv_sa = 255 - alpha;
            let out_a = alpha + (da * inv_sa) / 255;
            if out_a == 0 {
                continue;
            }
            for (c, &sc) in [sr, sg, sb].iter().enumerate() {
                let dc = output[si + c] as u32;
                output[si + c] =
                    ((sc as u32 * alpha + dc * da * inv_sa / 255) / out_a).min(255) as u8;
            }
            output[si + 3] = out_a.min(255) as u8;
        }
    }
}

// ---------------------------------------------------------------------------
// Inner Shadow Effect
// ---------------------------------------------------------------------------

/// Render an inner shadow: invert mask → offset → blur → clip to original mask.
fn render_inner_shadow(
    coverage: &[f32],
    w: u32,
    h: u32,
    inner: &InnerShadowEffect,
    output: &mut [u8],
) {
    let ww = w as usize;
    let hh = h as usize;
    let count = ww * hh;

    // Invert the mask, then offset
    let mut inv_offset = vec![0.0f32; count];
    let dx = inner.offset_x.round() as i32;
    let dy = inner.offset_y.round() as i32;

    for y in 0..hh {
        let sy = y as i32 - dy;
        if sy < 0 || sy >= hh as i32 {
            // Outside original → inverted mask is 1.0
            inv_offset[y * ww..y * ww + ww].fill(1.0);
            continue;
        }
        for x in 0..ww {
            let sx = x as i32 - dx;
            if sx < 0 || sx >= ww as i32 {
                inv_offset[y * ww + x] = 1.0;
            } else {
                inv_offset[y * ww + x] = 1.0 - coverage[sy as usize * ww + sx as usize];
            }
        }
    }

    // Apply Gaussian blur
    let [ir, ig, ib, ia] = inner.color;
    if inner.blur_radius > 0.5 {
        let mut shadow_rgba = vec![0u8; count * 4];
        for i in 0..count {
            let alpha = (inv_offset[i] * ia as f32).round().clamp(0.0, 255.0) as u8;
            shadow_rgba[i * 4] = ir;
            shadow_rgba[i * 4 + 1] = ig;
            shadow_rgba[i * 4 + 2] = ib;
            shadow_rgba[i * 4 + 3] = alpha;
        }
        let shadow_img = RgbaImage::from_raw(w, h, shadow_rgba).unwrap();
        let blurred =
            crate::ops::filters::parallel_gaussian_blur_pub(&shadow_img, inner.blur_radius);
        let blurred_raw = blurred.as_raw();

        // Clip to original text shape and composite onto output
        for (i, &clip_alpha) in coverage.iter().enumerate().take(count) {
            if clip_alpha < 1.0 / 255.0 {
                continue;
            }
            let bi = i * 4;
            let ba = blurred_raw[bi + 3] as f32 * clip_alpha;
            let sa = ba.round() as u32;
            if sa == 0 {
                continue;
            }
            let si = i * 4;
            let da = output[si + 3] as u32;
            let inv_sa = 255u32.saturating_sub(sa);
            let out_a = sa + (da * inv_sa) / 255;
            if out_a == 0 {
                continue;
            }
            for (c, &sc) in [blurred_raw[bi], blurred_raw[bi + 1], blurred_raw[bi + 2]]
                .iter()
                .enumerate()
            {
                let dc = output[si + c] as u32;
                output[si + c] = ((sc as u32 * sa + dc * da * inv_sa / 255) / out_a).min(255) as u8;
            }
            output[si + 3] = out_a.min(255) as u8;
        }
    } else {
        // No blur
        for i in 0..count {
            let clip_alpha = coverage[i];
            if clip_alpha < 1.0 / 255.0 {
                continue;
            }
            let alpha = (inv_offset[i] * ia as f32 * clip_alpha)
                .round()
                .clamp(0.0, 255.0) as u32;
            if alpha == 0 {
                continue;
            }
            let si = i * 4;
            let da = output[si + 3] as u32;
            let inv_sa = 255u32.saturating_sub(alpha);
            let out_a = alpha + (da * inv_sa) / 255;
            if out_a == 0 {
                continue;
            }
            for (c, &sc) in [ir, ig, ib].iter().enumerate() {
                let dc = output[si + c] as u32;
                output[si + c] =
                    ((sc as u32 * alpha + dc * da * inv_sa / 255) / out_a).min(255) as u8;
            }
            output[si + 3] = out_a.min(255) as u8;
        }
    }
}

// ---------------------------------------------------------------------------
// Gradient Fill Effect
// ---------------------------------------------------------------------------

fn render_gradient_fill(
    coverage: &[f32],
    w: u32,
    h: u32,
    gradient: &GradientFillEffect,
    output: &mut [u8],
) {
    let ww = w as usize;
    let hh = h as usize;
    let count = ww * hh;
    let angle = gradient.angle_degrees.to_radians();
    let dir_x = angle.cos();
    let dir_y = angle.sin();
    let scale = gradient.scale.max(1.0);
    let off_x = gradient.offset[0];
    let off_y = gradient.offset[1];
    let [sr, sg, sb, sa] = gradient.start_color;
    let [er, eg, eb, ea] = gradient.end_color;

    let mut filled = vec![0u8; count * 4];
    filled
        .par_chunks_mut(ww * 4)
        .enumerate()
        .for_each(|(y, row)| {
            for x in 0..ww {
                let cov = coverage[y * ww + x];
                if cov < 1.0 / 255.0 {
                    continue;
                }
                let proj = ((x as f32 - off_x) * dir_x + (y as f32 - off_y) * dir_y) / scale;
                let t = if gradient.repeat {
                    proj.rem_euclid(1.0)
                } else {
                    proj.clamp(0.0, 1.0)
                };
                let inv_t = 1.0 - t;
                let idx = x * 4;
                row[idx] = (sr as f32 * inv_t + er as f32 * t)
                    .round()
                    .clamp(0.0, 255.0) as u8;
                row[idx + 1] = (sg as f32 * inv_t + eg as f32 * t)
                    .round()
                    .clamp(0.0, 255.0) as u8;
                row[idx + 2] = (sb as f32 * inv_t + eb as f32 * t)
                    .round()
                    .clamp(0.0, 255.0) as u8;
                let grad_alpha = sa as f32 * inv_t + ea as f32 * t;
                row[idx + 3] = (grad_alpha * cov).round().clamp(0.0, 255.0) as u8;
            }
        });

    composite_over(&filled, output, count);
}

// ---------------------------------------------------------------------------
// Texture Fill Effect (Batch 6)
// ---------------------------------------------------------------------------

/// Fill text with a tiled texture pattern.
fn render_texture_fill(
    text_rgba: &[u8],
    coverage: &[f32],
    w: u32,
    h: u32,
    tex: &TextureFillEffect,
    output: &mut [u8],
) {
    // Decode texture image from embedded data
    if tex.texture_data.is_empty() || tex.texture_width == 0 || tex.texture_height == 0 {
        // No valid texture — fall back to normal text fill
        composite_over(text_rgba, output, (w as usize) * (h as usize));
        return;
    }

    // Try to decode the texture
    let tex_img = match crate::io::decode_image_bytes(&tex.texture_data) {
        Ok(img) => img.to_rgba8(),
        Err(_) => {
            // Decode failed — fall back to normal text fill
            composite_over(text_rgba, output, (w as usize) * (h as usize));
            return;
        }
    };

    let tw = tex_img.width() as usize;
    let th = tex_img.height() as usize;
    let tex_raw = tex_img.as_raw();
    let ww = w as usize;
    let hh = h as usize;
    let count = ww * hh;
    let scale = tex.scale.max(0.01);
    let off_x = tex.offset[0];
    let off_y = tex.offset[1];

    // Build a textured fill: for each pixel with text coverage, sample the tiled texture
    let mut textured = vec![0u8; count * 4];
    textured
        .par_chunks_mut(ww * 4)
        .enumerate()
        .for_each(|(y, row)| {
            for x in 0..ww {
                let cov = coverage[y * ww + x];
                if cov < 1.0 / 255.0 {
                    continue;
                }
                // Sample tiled texture at scaled+offset coordinates
                let tx_f = ((x as f32 - off_x) / scale) % tw as f32;
                let ty_f = ((y as f32 - off_y) / scale) % th as f32;
                let tx = ((tx_f + tw as f32) as usize) % tw;
                let ty = ((ty_f + th as f32) as usize) % th;
                let ti = (ty * tw + tx) * 4;
                let alpha = (cov * 255.0).round().clamp(0.0, 255.0) as u8;
                let pi = x * 4;
                row[pi] = tex_raw[ti];
                row[pi + 1] = tex_raw[ti + 1];
                row[pi + 2] = tex_raw[ti + 2];
                row[pi + 3] = alpha.min(tex_raw[ti + 3]);
            }
        });

    composite_over(&textured, output, count);
}
