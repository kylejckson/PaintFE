use crate::par_compat::*;
use image::{GrayImage, Rgba, RgbaImage};
use std::collections::{HashMap, VecDeque};

#[derive(Clone, Copy, Debug)]
pub struct ColorToAlphaSettings {
    pub target: [u8; 3],
    pub tolerance: f32,
    pub softness: f32,
    pub strength: f32,
    pub spill_suppression: f32,
    pub alpha_floor: f32,
    pub alpha_ceiling: f32,
    pub protect_luminance: f32,
}

impl Default for ColorToAlphaSettings {
    fn default() -> Self {
        Self {
            target: [255, 0, 0],
            tolerance: 18.0,
            softness: 35.0,
            strength: 1.0,
            spill_suppression: 0.35,
            alpha_floor: 0.0,
            alpha_ceiling: 1.0,
            protect_luminance: 0.15,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InteriorRecoveryMode {
    Off,
    SmallIslands,
    NearExterior,
    AllMatching,
}

#[derive(Clone, Debug)]
pub struct RecoverTransparencySettings {
    pub background: [u8; 3],
    pub auto_sample_edges: bool,
    pub sample_depth: u32,
    pub noise_tolerance: f32,
    pub edge_width: u32,
    pub eight_connected: bool,
    pub preserve_hard_pixels: bool,
    pub transparent_snap: f32,
    pub opaque_snap: f32,
    pub foreground_influence: f32,
    pub interior_mode: InteriorRecoveryMode,
    pub island_max_size: u32,
    pub island_tolerance: f32,
    pub island_max_depth: u32,
    pub bridge_gaps: u32,
    pub remove_seeds: Vec<(u32, u32)>,
    pub protect_seeds: Vec<(u32, u32)>,
}

impl Default for RecoverTransparencySettings {
    fn default() -> Self {
        Self {
            background: [255, 0, 255],
            auto_sample_edges: true,
            sample_depth: 2,
            noise_tolerance: 4.0,
            edge_width: 2,
            eight_connected: false,
            preserve_hard_pixels: true,
            transparent_snap: 0.025,
            opaque_snap: 0.97,
            foreground_influence: 0.8,
            interior_mode: InteriorRecoveryMode::Off,
            island_max_size: 24,
            island_tolerance: 0.65,
            island_max_depth: 12,
            bridge_gaps: 0,
            remove_seeds: Vec::new(),
            protect_seeds: Vec::new(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecoverTransparencyPreview {
    Result,
    Alpha,
    ReconstructionError,
}

#[derive(Clone, Copy)]
struct BackgroundModel {
    color: [f32; 3],
    tolerance: [f32; 3],
}

/// Recover a sprite that was flattened over a solid or slightly noisy backing.
/// Only the backing-connected component and a narrow inward edge band are changed;
/// opaque interior pixels are preserved exactly.
pub fn recover_transparency_core(
    img: &RgbaImage,
    settings: &RecoverTransparencySettings,
    mask: Option<&GrayImage>,
) -> RgbaImage {
    recover_transparency_impl(img, settings, mask).0
}

pub fn recover_transparency_preview_core(
    img: &RgbaImage,
    settings: &RecoverTransparencySettings,
    mask: Option<&GrayImage>,
    preview: RecoverTransparencyPreview,
) -> RgbaImage {
    let (result, background) = recover_transparency_impl(img, settings, mask);
    match preview {
        RecoverTransparencyPreview::Result => result,
        RecoverTransparencyPreview::Alpha => {
            let mut out = result.clone();
            for p in out.pixels_mut() {
                let a = p[3];
                *p = Rgba([a, a, a, 255]);
            }
            out
        }
        RecoverTransparencyPreview::ReconstructionError => {
            let mut out = RgbaImage::new(img.width(), img.height());
            for (x, y, p) in out.enumerate_pixels_mut() {
                let src = img.get_pixel(x, y);
                let recovered = result.get_pixel(x, y);
                let a = recovered[3] as f32 / 255.0;
                let mut max_error = 0.0_f32;
                for c in 0..3 {
                    let recomposed = recovered[c] as f32 * a + background[c] * (1.0 - a);
                    max_error = max_error.max((recomposed - src[c] as f32).abs());
                }
                let error = (max_error * 8.0).round().clamp(0.0, 255.0) as u8;
                *p = Rgba([error, 0, 0, 255]);
            }
            out
        }
    }
}

fn recover_transparency_impl(
    img: &RgbaImage,
    settings: &RecoverTransparencySettings,
    mask: Option<&GrayImage>,
) -> (RgbaImage, [f32; 3]) {
    let w = img.width();
    let h = img.height();
    if w == 0 || h == 0 {
        return (img.clone(), settings.background.map(|v| v as f32));
    }

    let model = background_model(img, settings);
    let count = (w * h) as usize;
    let mut background = vec![false; count];
    let mut queue = VecDeque::new();

    let seed = |x: u32, y: u32, background: &mut [bool], queue: &mut VecDeque<(u32, u32)>| {
        let idx = (y * w + x) as usize;
        if !background[idx] && matches_background(img.get_pixel(x, y), &model) {
            background[idx] = true;
            queue.push_back((x, y));
        }
    };
    for x in 0..w {
        seed(x, 0, &mut background, &mut queue);
        if h > 1 {
            seed(x, h - 1, &mut background, &mut queue);
        }
    }
    for y in 1..h.saturating_sub(1) {
        seed(0, y, &mut background, &mut queue);
        if w > 1 {
            seed(w - 1, y, &mut background, &mut queue);
        }
    }

    while let Some((x, y)) = queue.pop_front() {
        for (nx, ny) in neighbors(x, y, w, h, settings.eight_connected) {
            let idx = (ny * w + nx) as usize;
            if !background[idx] && matches_background(img.get_pixel(nx, ny), &model) {
                background[idx] = true;
                queue.push_back((nx, ny));
            }
        }
    }

    recover_interior_background(img, settings, &model, &mut background);

    let mut distance = vec![u32::MAX; count];
    let mut frontier = VecDeque::new();
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) as usize;
            if background[idx] {
                distance[idx] = 0;
                frontier.push_back((x, y));
            }
        }
    }
    while let Some((x, y)) = frontier.pop_front() {
        let d = distance[(y * w + x) as usize];
        if d >= settings.edge_width {
            continue;
        }
        for (nx, ny) in neighbors(x, y, w, h, settings.eight_connected) {
            let idx = (ny * w + nx) as usize;
            if distance[idx] == u32::MAX {
                distance[idx] = d + 1;
                frontier.push_back((nx, ny));
            }
        }
    }

    let mut out = img.clone();
    for y in 0..h {
        for x in 0..w {
            if !mask_allows(mask, x, y) {
                continue;
            }
            let idx = (y * w + x) as usize;
            if background[idx] {
                *out.get_pixel_mut(x, y) = Rgba([0, 0, 0, 0]);
                continue;
            }
            let d = distance[idx];
            if d == u32::MAX || d == 0 || d > settings.edge_width {
                continue;
            }

            let src = img.get_pixel(x, y);
            let mut alpha = minimum_feasible_alpha(src, model.color);
            if let Some(local_foreground) = estimate_local_foreground(
                img,
                &background,
                &distance,
                x,
                y,
                settings.edge_width.max(2) + 1,
                model.color,
                alpha,
            ) {
                let projected = projected_alpha(src, model.color, local_foreground);
                let refined = projected.max(alpha);
                alpha += (refined - alpha) * settings.foreground_influence.clamp(0.0, 1.0);
            }

            if alpha <= settings.transparent_snap.clamp(0.0, 1.0) {
                *out.get_pixel_mut(x, y) = Rgba([0, 0, 0, 0]);
                continue;
            }
            if settings.preserve_hard_pixels && alpha >= settings.opaque_snap.clamp(0.0, 1.0) {
                continue;
            }

            let alpha = alpha.clamp(1.0 / 255.0, 1.0);
            let mut rgba = [0_u8; 4];
            for c in 0..3 {
                let foreground = (src[c] as f32 - model.color[c] * (1.0 - alpha)) / alpha;
                rgba[c] = foreground.round().clamp(0.0, 255.0) as u8;
            }
            rgba[3] = ((src[3] as f32 / 255.0) * alpha * 255.0)
                .round()
                .clamp(0.0, 255.0) as u8;
            *out.get_pixel_mut(x, y) = Rgba(rgba);
        }
    }

    (out, model.color)
}

fn recover_interior_background(
    img: &RgbaImage,
    settings: &RecoverTransparencySettings,
    model: &BackgroundModel,
    background: &mut [bool],
) {
    let w = img.width();
    let h = img.height();
    let count = (w * h) as usize;
    let island_model = BackgroundModel {
        color: model.color,
        tolerance: model
            .tolerance
            .map(|v| (v * settings.island_tolerance.clamp(0.1, 2.0)).max(0.5)),
    };
    let candidate: Vec<bool> = img
        .pixels()
        .map(|p| matches_background(p, &island_model))
        .collect();
    let mut protected = vec![false; count];
    for &(x, y) in &settings.protect_seeds {
        for idx in component_from_seed(x, y, w, h, &candidate, settings.eight_connected) {
            protected[idx] = true;
        }
    }

    for &(x, y) in &settings.remove_seeds {
        for idx in component_from_seed(x, y, w, h, &candidate, settings.eight_connected) {
            if !protected[idx] {
                background[idx] = true;
            }
        }
    }
    if settings.interior_mode == InteriorRecoveryMode::Off {
        return;
    }

    let exterior_distance = distance_from_mask(w, h, background, settings.eight_connected);
    let mut visited = background.to_vec();
    for start in 0..count {
        if visited[start] || protected[start] || !candidate[start] {
            continue;
        }
        let sx = start as u32 % w;
        let sy = start as u32 / w;
        let mut queue = VecDeque::from([(sx, sy)]);
        let mut component = Vec::new();
        visited[start] = true;
        while let Some((x, y)) = queue.pop_front() {
            let idx = (y * w + x) as usize;
            component.push(idx);
            for (nx, ny) in neighbors(x, y, w, h, settings.eight_connected) {
                let nidx = (ny * w + nx) as usize;
                if candidate[nidx] && !visited[nidx] {
                    visited[nidx] = true;
                    queue.push_back((nx, ny));
                }
            }
        }
        if component.iter().any(|&idx| protected[idx]) {
            continue;
        }
        let min_depth = component
            .iter()
            .map(|&idx| exterior_distance[idx])
            .min()
            .unwrap_or(u32::MAX);
        let size_ok = component.len() as u32 <= settings.island_max_size.max(1);
        let bridged =
            settings.bridge_gaps > 0 && min_depth <= settings.bridge_gaps.saturating_add(1);
        let remove = match settings.interior_mode {
            InteriorRecoveryMode::Off => false,
            InteriorRecoveryMode::SmallIslands => size_ok || bridged,
            InteriorRecoveryMode::NearExterior => {
                size_ok && min_depth <= settings.island_max_depth.max(1)
            }
            InteriorRecoveryMode::AllMatching => true,
        };
        if remove {
            for idx in component {
                background[idx] = true;
            }
        }
    }
}

fn component_from_seed(
    x: u32,
    y: u32,
    w: u32,
    h: u32,
    candidate: &[bool],
    eight_connected: bool,
) -> Vec<usize> {
    if x >= w || y >= h {
        return Vec::new();
    }
    let start = (y * w + x) as usize;
    if !candidate[start] {
        return Vec::new();
    }
    let mut visited = vec![false; candidate.len()];
    let mut queue = VecDeque::from([(x, y)]);
    let mut component = Vec::new();
    visited[start] = true;
    while let Some((cx, cy)) = queue.pop_front() {
        let idx = (cy * w + cx) as usize;
        component.push(idx);
        for (nx, ny) in neighbors(cx, cy, w, h, eight_connected) {
            let nidx = (ny * w + nx) as usize;
            if candidate[nidx] && !visited[nidx] {
                visited[nidx] = true;
                queue.push_back((nx, ny));
            }
        }
    }
    component
}

fn distance_from_mask(w: u32, h: u32, source: &[bool], eight_connected: bool) -> Vec<u32> {
    let mut distance = vec![u32::MAX; source.len()];
    let mut queue = VecDeque::new();
    for (idx, &set) in source.iter().enumerate() {
        if set {
            distance[idx] = 0;
            queue.push_back((idx as u32 % w, idx as u32 / w));
        }
    }
    while let Some((x, y)) = queue.pop_front() {
        let next = distance[(y * w + x) as usize].saturating_add(1);
        for (nx, ny) in neighbors(x, y, w, h, eight_connected) {
            let idx = (ny * w + nx) as usize;
            if distance[idx] == u32::MAX {
                distance[idx] = next;
                queue.push_back((nx, ny));
            }
        }
    }
    distance
}

fn background_model(img: &RgbaImage, settings: &RecoverTransparencySettings) -> BackgroundModel {
    let fallback = settings.background.map(|v| v as f32);
    if !settings.auto_sample_edges {
        return BackgroundModel {
            color: fallback,
            tolerance: [settings.noise_tolerance.max(0.5); 3],
        };
    }

    let w = img.width();
    let h = img.height();
    let depth = settings.sample_depth.max(1).min(w.min(h).div_ceil(2));
    let mut bins: HashMap<[u8; 3], usize> = HashMap::new();
    let mut samples = Vec::new();
    for y in 0..h {
        for x in 0..w {
            if x >= depth && y >= depth && x < w - depth && y < h - depth {
                continue;
            }
            let p = img.get_pixel(x, y);
            if p[3] == 0 {
                continue;
            }
            let key = [p[0] / 16, p[1] / 16, p[2] / 16];
            *bins.entry(key).or_default() += 1;
            samples.push([p[0], p[1], p[2]]);
        }
    }
    let Some((&dominant, _)) = bins.iter().max_by_key(|(_, count)| *count) else {
        return BackgroundModel {
            color: fallback,
            tolerance: [settings.noise_tolerance.max(0.5); 3],
        };
    };
    let mut cluster: Vec<[u8; 3]> = samples
        .into_iter()
        .filter(|p| {
            (0..3).all(|c| {
                let bin = p[c] / 16;
                bin.abs_diff(dominant[c]) <= 1
            })
        })
        .collect();
    if cluster.is_empty() {
        return BackgroundModel {
            color: fallback,
            tolerance: [settings.noise_tolerance.max(0.5); 3],
        };
    }

    let mut color = [0.0; 3];
    for c in 0..3 {
        cluster.sort_unstable_by_key(|p| p[c]);
        color[c] = cluster[cluster.len() / 2][c] as f32;
    }
    let mut tolerance = [0.0; 3];
    for c in 0..3 {
        let mut deviations: Vec<f32> = cluster
            .iter()
            .map(|p| (p[c] as f32 - color[c]).abs())
            .collect();
        deviations.sort_by(|a, b| a.total_cmp(b));
        let mad = deviations[deviations.len() / 2];
        tolerance[c] = (settings.noise_tolerance + mad * 3.0).clamp(0.5, 64.0);
    }
    BackgroundModel { color, tolerance }
}

#[inline]
fn matches_background(pixel: &Rgba<u8>, model: &BackgroundModel) -> bool {
    pixel[3] > 0 && (0..3).all(|c| (pixel[c] as f32 - model.color[c]).abs() <= model.tolerance[c])
}

fn neighbors(x: u32, y: u32, w: u32, h: u32, eight_connected: bool) -> Vec<(u32, u32)> {
    let mut result = Vec::with_capacity(if eight_connected { 8 } else { 4 });
    for dy in -1_i32..=1 {
        for dx in -1_i32..=1 {
            if dx == 0 && dy == 0 || (!eight_connected && dx != 0 && dy != 0) {
                continue;
            }
            let nx = x as i32 + dx;
            let ny = y as i32 + dy;
            if nx >= 0 && ny >= 0 && nx < w as i32 && ny < h as i32 {
                result.push((nx as u32, ny as u32));
            }
        }
    }
    result
}

#[inline]
fn mask_allows(mask: Option<&GrayImage>, x: u32, y: u32) -> bool {
    mask.is_none_or(|m| x < m.width() && y < m.height() && m.get_pixel(x, y)[0] > 0)
}

fn minimum_feasible_alpha(pixel: &Rgba<u8>, background: [f32; 3]) -> f32 {
    let mut alpha = 0.0_f32;
    for c in 0..3 {
        let value = pixel[c] as f32;
        let backing = background[c];
        let required = if value > backing && backing < 255.0 {
            (value - backing) / (255.0 - backing)
        } else if value < backing && backing > 0.0 {
            (backing - value) / backing
        } else {
            0.0
        };
        alpha = alpha.max(required);
    }
    alpha.clamp(0.0, 1.0)
}

fn projected_alpha(pixel: &Rgba<u8>, background: [f32; 3], foreground: [f32; 3]) -> f32 {
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for c in 0..3 {
        let direction = foreground[c] - background[c];
        numerator += (pixel[c] as f32 - background[c]) * direction;
        denominator += direction * direction;
    }
    if denominator < 1.0 {
        0.0
    } else {
        (numerator / denominator).clamp(0.0, 1.0)
    }
}

fn estimate_local_foreground(
    img: &RgbaImage,
    background: &[bool],
    distance: &[u32],
    x: u32,
    y: u32,
    radius: u32,
    backing: [f32; 3],
    current_alpha: f32,
) -> Option<[f32; 3]> {
    let w = img.width();
    let h = img.height();
    let x0 = x.saturating_sub(radius);
    let y0 = y.saturating_sub(radius);
    let x1 = (x + radius).min(w - 1);
    let y1 = (y + radius).min(h - 1);
    let mut sum = [0.0; 3];
    let mut weight_sum = 0.0;
    for ny in y0..=y1 {
        for nx in x0..=x1 {
            let idx = (ny * w + nx) as usize;
            if background[idx] || (nx == x && ny == y) {
                continue;
            }
            let p = img.get_pixel(nx, ny);
            let candidate_alpha = minimum_feasible_alpha(p, backing);
            let current_distance = distance[(y * w + x) as usize];
            let deeper = distance[idx] == u32::MAX || distance[idx] > current_distance;
            if !deeper && candidate_alpha < 0.85 && candidate_alpha < current_alpha + 0.05 {
                continue;
            }
            let dx = nx.abs_diff(x) as f32;
            let dy = ny.abs_diff(y) as f32;
            let confidence = candidate_alpha.max(0.05).powi(4);
            let depth_weight = if deeper { 2.0 } else { 1.0 };
            let weight = confidence * depth_weight / (1.0 + dx + dy);
            for c in 0..3 {
                sum[c] += p[c] as f32 * weight;
            }
            weight_sum += weight;
        }
    }
    (weight_sum > 0.0).then(|| sum.map(|v| v / weight_sum))
}

pub fn color_to_alpha_core(
    img: &RgbaImage,
    settings: &ColorToAlphaSettings,
    mask: Option<&GrayImage>,
) -> RgbaImage {
    let w = img.width() as usize;
    let h = img.height() as usize;
    if w == 0 || h == 0 {
        return img.clone();
    }

    let src = img.as_raw();
    let mut dst = src.clone();
    let stride = w * 4;
    let target = [
        settings.target[0] as f32,
        settings.target[1] as f32,
        settings.target[2] as f32,
    ];
    let tolerance = (settings.tolerance / 255.0).clamp(0.0, 1.0);
    let softness = (settings.softness / 255.0).max(0.001);
    let strength = settings.strength.clamp(0.0, 1.0);
    let spill = settings.spill_suppression.clamp(0.0, 1.0);
    let alpha_floor = settings.alpha_floor.clamp(0.0, 1.0);
    let alpha_ceiling = settings.alpha_ceiling.clamp(alpha_floor, 1.0);
    let protect_luma = settings.protect_luminance.clamp(0.0, 1.0);
    let target_luma = luma(target[0], target[1], target[2]);

    let mask_raw = mask.map(|m| m.as_raw().as_slice());
    let mask_w = mask.map_or(0, |m| m.width() as usize);
    let mask_h = mask.map_or(0, |m| m.height() as usize);

    dst.par_chunks_mut(stride).enumerate().for_each(|(y, row)| {
        let src_row = &src[y * stride..(y + 1) * stride];
        for x in 0..w {
            if let Some(mr) = mask_raw
                && (x >= mask_w || y >= mask_h || mr[y * mask_w + x] == 0)
            {
                continue;
            }

            let pi = x * 4;
            let orig_a = src_row[pi + 3];
            if orig_a == 0 {
                continue;
            }

            let r = src_row[pi] as f32;
            let g = src_row[pi + 1] as f32;
            let b = src_row[pi + 2] as f32;
            let max_d = ((r - target[0]).abs() / 255.0)
                .max((g - target[1]).abs() / 255.0)
                .max((b - target[2]).abs() / 255.0);

            let mut contribution = 1.0 - ((max_d - tolerance) / softness).clamp(0.0, 1.0);
            if protect_luma > 0.0 {
                let luma_delta = ((luma(r, g, b) - target_luma).abs() / 255.0).clamp(0.0, 1.0);
                let protection = (luma_delta * protect_luma).clamp(0.0, 1.0);
                contribution *= 1.0 - protection;
            }

            let removal = (contribution * strength).clamp(0.0, 1.0);
            if removal <= 0.0 {
                continue;
            }

            let new_a_f =
                ((orig_a as f32 / 255.0) * (1.0 - removal)).clamp(alpha_floor, alpha_ceiling);
            let kept = if orig_a == 0 {
                0.0
            } else {
                (new_a_f / (orig_a as f32 / 255.0)).clamp(0.0, 1.0)
            };
            let new_a = (new_a_f * 255.0).round().clamp(0.0, 255.0) as u8;
            row[pi + 3] = new_a;

            if new_a == 0 || kept < 0.001 {
                row[pi] = 0;
                row[pi + 1] = 0;
                row[pi + 2] = 0;
                continue;
            }

            let recover = |orig: f32, target_ch: f32| -> f32 {
                ((orig - target_ch * removal) / kept).clamp(0.0, 255.0)
            };
            let mut nr = recover(r, target[0]);
            let mut ng = recover(g, target[1]);
            let mut nb = recover(b, target[2]);

            if spill > 0.0 {
                let spill_amount = spill * contribution * (1.0 - kept);
                nr = suppress_channel_spill(nr, target[0], spill_amount);
                ng = suppress_channel_spill(ng, target[1], spill_amount);
                nb = suppress_channel_spill(nb, target[2], spill_amount);
            }

            row[pi] = nr.round() as u8;
            row[pi + 1] = ng.round() as u8;
            row[pi + 2] = nb.round() as u8;
        }
    });

    RgbaImage::from_raw(w as u32, h as u32, dst).unwrap()
}

#[inline]
fn luma(r: f32, g: f32, b: f32) -> f32 {
    r * 0.2126 + g * 0.7152 + b * 0.0722
}

#[inline]
fn suppress_channel_spill(value: f32, target: f32, amount: f32) -> f32 {
    if target <= 0.0 {
        value
    } else {
        value * (1.0 - amount.clamp(0.0, 1.0))
    }
}

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
    if let Some(mask) = selection_mask
        && (start_x >= mask.width()
            || start_y >= mask.height()
            || mask.get_pixel(start_x, start_y).0[0] == 0)
    {
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
                if let Some(mask) = selection_mask
                    && mask.get_pixel(nx, ny).0[0] == 0
                {
                    continue;
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
            if let Some(mask) = selection_mask
                && mask.get_pixel(x, y).0[0] == 0
            {
                return;
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
                        if let Some(mask) = selection_mask
                            && mask.get_pixel(nx, ny).0[0] == 0
                        {
                            continue;
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
                if let Some(mask) = selection_mask
                    && mask.get_pixel(nx, ny).0[0] == 0
                {
                    continue;
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

#[cfg(test)]
mod tests {
    use super::*;
    use image::{GrayImage, Luma};

    #[test]
    fn recover_transparency_removes_noisy_connected_backing() {
        let mut img = RgbaImage::from_pixel(7, 7, Rgba([250, 3, 251, 255]));
        img.put_pixel(0, 2, Rgba([249, 4, 250, 255]));
        img.put_pixel(6, 4, Rgba([252, 2, 249, 255]));
        for y in 2..=4 {
            for x in 2..=4 {
                img.put_pixel(x, y, Rgba([30, 150, 45, 255]));
            }
        }
        let out = recover_transparency_core(&img, &RecoverTransparencySettings::default(), None);
        assert_eq!(out.get_pixel(0, 2)[3], 0);
        assert_eq!(out.get_pixel(6, 4)[3], 0);
        assert_eq!(out.get_pixel(3, 3).0, [30, 150, 45, 255]);
    }

    #[test]
    fn recover_transparency_recovers_antialiased_orange_edge() {
        let backing = [255_u8, 0, 255];
        let foreground = [235_u8, 105, 18];
        let mut img = RgbaImage::from_pixel(7, 7, Rgba([255, 0, 255, 255]));
        for y in 2..=4 {
            for x in 2..=4 {
                img.put_pixel(
                    x,
                    y,
                    Rgba([foreground[0], foreground[1], foreground[2], 255]),
                );
            }
        }
        let alpha = 0.5_f32;
        let mixed: [u8; 3] = std::array::from_fn(|c| {
            (foreground[c] as f32 * alpha + backing[c] as f32 * (1.0 - alpha)).round() as u8
        });
        img.put_pixel(1, 3, Rgba([mixed[0], mixed[1], mixed[2], 255]));

        let out = recover_transparency_core(&img, &RecoverTransparencySettings::default(), None);
        let edge = out.get_pixel(1, 3);
        assert!((edge[3] as i16 - 128).abs() <= 3);
        for c in 0..3 {
            assert!((edge[c] as i16 - foreground[c] as i16).abs() <= 4);
        }
    }

    #[test]
    fn recover_transparency_keeps_enclosed_backing_color() {
        let mut img = RgbaImage::from_pixel(7, 7, Rgba([255, 0, 255, 255]));
        for y in 1..=5 {
            for x in 1..=5 {
                img.put_pixel(x, y, Rgba([20, 150, 40, 255]));
            }
        }
        img.put_pixel(3, 3, Rgba([255, 0, 255, 255]));
        let out = recover_transparency_core(&img, &RecoverTransparencySettings::default(), None);
        assert_eq!(out.get_pixel(3, 3).0, [255, 0, 255, 255]);
    }

    #[test]
    fn recover_transparency_can_remove_small_enclosed_island() {
        let mut img = RgbaImage::from_pixel(7, 7, Rgba([255, 0, 255, 255]));
        for y in 1..=5 {
            for x in 1..=5 {
                img.put_pixel(x, y, Rgba([20, 150, 40, 255]));
            }
        }
        img.put_pixel(3, 3, Rgba([254, 1, 255, 255]));
        let settings = RecoverTransparencySettings {
            interior_mode: InteriorRecoveryMode::SmallIslands,
            ..RecoverTransparencySettings::default()
        };
        let out = recover_transparency_core(&img, &settings, None);
        assert_eq!(out.get_pixel(3, 3)[3], 0);
    }

    #[test]
    fn recover_transparency_protect_seed_wins_over_island_recovery() {
        let mut img = RgbaImage::from_pixel(7, 7, Rgba([255, 0, 255, 255]));
        for y in 1..=5 {
            for x in 1..=5 {
                img.put_pixel(x, y, Rgba([20, 150, 40, 255]));
            }
        }
        img.put_pixel(3, 3, Rgba([255, 0, 255, 255]));
        let settings = RecoverTransparencySettings {
            interior_mode: InteriorRecoveryMode::AllMatching,
            protect_seeds: vec![(3, 3)],
            ..RecoverTransparencySettings::default()
        };
        let out = recover_transparency_core(&img, &settings, None);
        assert_eq!(out.get_pixel(3, 3).0, [255, 0, 255, 255]);
    }

    #[test]
    fn color_to_alpha_makes_exact_target_transparent() {
        let img = RgbaImage::from_pixel(1, 1, Rgba([255, 0, 0, 255]));
        let out = color_to_alpha_core(&img, &ColorToAlphaSettings::default(), None);
        assert_eq!(out.get_pixel(0, 0).0, [0, 0, 0, 0]);
    }

    #[test]
    fn color_to_alpha_keeps_distant_color_unchanged() {
        let img = RgbaImage::from_pixel(1, 1, Rgba([0, 180, 40, 255]));
        let out = color_to_alpha_core(&img, &ColorToAlphaSettings::default(), None);
        assert_eq!(out.get_pixel(0, 0).0, [0, 180, 40, 255]);
    }

    #[test]
    fn color_to_alpha_partially_removes_mixed_target_color() {
        let img = RgbaImage::from_pixel(1, 1, Rgba([220, 35, 0, 255]));
        let out = color_to_alpha_core(&img, &ColorToAlphaSettings::default(), None);
        let p = out.get_pixel(0, 0);
        assert!(p[3] > 0 && p[3] < 255);
        assert!(p[1] >= 35);
    }

    #[test]
    fn color_to_alpha_respects_selection_mask() {
        let mut img = RgbaImage::from_pixel(2, 1, Rgba([255, 0, 0, 255]));
        *img.get_pixel_mut(1, 0) = Rgba([255, 0, 0, 255]);
        let mut mask = GrayImage::from_pixel(2, 1, Luma([0]));
        *mask.get_pixel_mut(0, 0) = Luma([255]);

        let out = color_to_alpha_core(&img, &ColorToAlphaSettings::default(), Some(&mask));
        assert_eq!(out.get_pixel(0, 0).0, [0, 0, 0, 0]);
        assert_eq!(out.get_pixel(1, 0).0, [255, 0, 0, 255]);
    }

    #[test]
    fn color_to_alpha_preserves_existing_alpha_ratio() {
        let img = RgbaImage::from_pixel(1, 1, Rgba([255, 0, 0, 128]));
        let settings = ColorToAlphaSettings {
            strength: 0.5,
            ..Default::default()
        };
        let out = color_to_alpha_core(&img, &settings, None);
        let p = out.get_pixel(0, 0);
        assert!(p[3] > 0 && p[3] < 128);
    }
}
