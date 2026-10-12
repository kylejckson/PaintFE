pub fn contours(
    state: &mut CanvasState,
    layer_idx: usize,
    scale: f32,
    frequency: f32,
    line_width: f32,
    line_color: [u8; 4],
    seed: u32,
    octaves: u32,
    blend: f32,
) {
    if layer_idx >= state.layers.len() {
        return;
    }
    let flat = state.layers[layer_idx].pixels.to_rgba_image();
    let result = contours_core(
        &flat,
        scale,
        frequency,
        line_width,
        line_color,
        seed,
        octaves,
        blend,
        state.selection_mask.as_ref(),
    );
    commit_to_layer(state, layer_idx, &result);
}

pub fn contours_from_flat(
    state: &mut CanvasState,
    layer_idx: usize,
    scale: f32,
    frequency: f32,
    line_width: f32,
    line_color: [u8; 4],
    seed: u32,
    octaves: u32,
    blend: f32,
    original_flat: &RgbaImage,
) {
    let result = contours_core(
        original_flat,
        scale,
        frequency,
        line_width,
        line_color,
        seed,
        octaves,
        blend,
        state.selection_mask.as_ref(),
    );
    commit_to_layer(state, layer_idx, &result);
}

pub fn contours_core(
    flat: &RgbaImage,
    scale: f32,
    frequency: f32,
    line_width: f32,
    line_color: [u8; 4],
    seed: u32,
    octaves: u32,
    blend: f32,
    mask: Option<&GrayImage>,
) -> RgbaImage {
    contours_core_cancellable(
        flat, scale, frequency, line_width, line_color, seed, octaves, blend, mask, None,
    )
}

pub fn contours_core_cancellable(
    flat: &RgbaImage,
    scale: f32,
    frequency: f32,
    line_width: f32,
    line_color: [u8; 4],
    seed: u32,
    octaves: u32,
    blend: f32,
    mask: Option<&GrayImage>,
    cancel: Option<&std::sync::atomic::AtomicBool>,
) -> RgbaImage {
    let w = flat.width() as usize;
    let h = flat.height() as usize;
    if w == 0 || h == 0 {
        return flat.clone();
    }

    let inv_scale = 1.0 / scale.max(0.5);
    let oct = octaves.clamp(1, 8);
    let half_lw = (line_width * 0.5).max(0.3);
    let lr = line_color[0] as f32;
    let lg = line_color[1] as f32;
    let lb = line_color[2] as f32;
    let la = line_color[3] as f32 / 255.0;

    // Frequency controls how many contour levels in the 0..1 noise range
    let freq = frequency.max(0.5);

    apply_per_pixel_cancellable(flat, mask, cancel, |x, y, r, g, b, a| {
        // Sample noise field
        let sx = x as f32 * inv_scale;
        let sy = y as f32 * inv_scale;
        let noise_val = turbulence_2d(sx, sy, seed, oct, 0.5);

        // Map to contour: distance to nearest contour level
        let level = noise_val * freq;
        let dist_to_contour = (level - level.round()).abs() / freq;

        // Anti-aliased contour line (smooth step)
        let edge = half_lw * inv_scale * 0.5;
        let line_alpha = if dist_to_contour < edge {
            1.0
        } else if dist_to_contour < edge * 2.0 {
            1.0 - (dist_to_contour - edge) / edge
        } else {
            0.0
        };

        let alpha = line_alpha * la * blend;

        // Blend contour line color over original pixel
        let nr = r * (1.0 - alpha) + lr * alpha;
        let ng = g * (1.0 - alpha) + lg * alpha;
        let nb = b * (1.0 - alpha) + lb * alpha;
        (nr, ng, nb, a)
    })
}
