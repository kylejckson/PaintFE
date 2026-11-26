use crate::canvas::CanvasState;
use image::GrayImage;

const TILE: usize = 512;
const MAX_RADIUS: usize = 200;
const NO_SEED: i32 = -1;

#[derive(Clone, Copy)]
enum Modifier {
    Expand,
    Contract,
    Feather,
}

pub fn expand_selection(state: &mut CanvasState, radius: i32) {
    modify(state, radius, Modifier::Expand);
}

pub fn contract_selection(state: &mut CanvasState, radius: i32) {
    modify(state, radius, Modifier::Contract);
}

pub fn feather_selection(state: &mut CanvasState, radius: f32) {
    modify(state, radius.round() as i32, Modifier::Feather);
}

fn modify(state: &mut CanvasState, radius: i32, modifier: Modifier) {
    let radius = (radius.max(0) as usize).min(MAX_RADIUS);
    let contract = matches!(modifier, Modifier::Contract);
    if radius == 0 || (state.selection_all && !contract) {
        return;
    }
    let Some((x0, y0, x1, y1)) = state.selection_mask_bounds() else {
        return;
    };
    state.materialize_full_selection();
    let Some(source) = state.selection_mask.as_ref() else {
        return;
    };
    let (width, height) = (source.width() as usize, source.height() as usize);
    let padding = if contract { 0 } else { radius };
    let xs = (x0 as usize).saturating_sub(padding)..(x1 as usize + 1 + padding).min(width);
    let ys = (y0 as usize).saturating_sub(padding)..(y1 as usize + 1 + padding).min(height);
    let mut output: GrayImage = source.clone();

    for y in ys.clone().step_by(TILE) {
        for x in xs.clone().step_by(TILE) {
            modify_tile(
                source,
                &mut output,
                x..(x + TILE).min(xs.end),
                y..(y + TILE).min(ys.end),
                radius,
                modifier,
            );
        }
    }
    if *source != output {
        state.selection_mask = Some(output);
        state.invalidate_selection_overlay();
    }
}

fn modify_tile(
    source: &GrayImage,
    output: &mut GrayImage,
    xs: std::ops::Range<usize>,
    ys: std::ops::Range<usize>,
    radius: usize,
    modifier: Modifier,
) {
    let (width, height) = (source.width() as usize, source.height() as usize);
    let left = xs.start.saturating_sub(radius);
    let top = ys.start.saturating_sub(radius);
    let columns = (xs.end + radius).min(width) - left;
    let rows = (ys.end + radius).min(height) - top;
    let contract = matches!(modifier, Modifier::Contract);
    let mut nearest_x = vec![NO_SEED; columns * rows];
    let radius_squared = (radius * radius) as i32;

    // A radius-sized halo makes each tile exact within the requested distance.
    // Two horizontal sweeps find each row's nearest seed in linear time.
    for (y, nearest) in nearest_x.chunks_exact_mut(columns).enumerate() {
        let start = (top + y) * width + left;
        let row = &source.as_raw()[start..start + columns];
        let mut seed_x = NO_SEED;
        for (x, (&alpha, nearest)) in row.iter().zip(nearest.iter_mut()).enumerate() {
            if (alpha == 0) == contract {
                seed_x = x as i32;
            }
            *nearest = seed_x;
        }
        seed_x = NO_SEED;
        for (x, (&alpha, nearest)) in row.iter().zip(nearest.iter_mut()).enumerate().rev() {
            let x = x as i32;
            if (alpha == 0) == contract {
                seed_x = x;
            }
            if seed_x != NO_SEED && (*nearest == NO_SEED || seed_x - x < x - *nearest) {
                *nearest = seed_x;
            }
        }
    }

    // Each row contributes a parabola (Felzenszwalb/Huttenlocher distance transform).
    // The envelope stores only the rows that are closest for at least one output y.
    let mut envelope: Vec<(i32, i32, i32)> = Vec::with_capacity(rows);
    for x in xs {
        let local_x = (x - left) as i32;
        envelope.clear();
        for y in 0..rows {
            let seed = nearest_x[y * columns + local_x as usize];
            let distance_x = (seed - local_x).pow(2);
            if seed == NO_SEED || distance_x > radius_squared {
                continue;
            }
            let y = y as i32;
            let mut start = i32::MIN;
            while let Some(&(prev_y, prev_cost, prev_start)) = envelope.last() {
                // First integer coordinate where the new row is strictly closer.
                start = (distance_x + y * y - prev_cost - prev_y * prev_y)
                    .div_euclid(2 * (y - prev_y))
                    + 1;
                if start > prev_start {
                    break;
                }
                envelope.pop();
            }
            envelope.push((y, distance_x, start));
        }

        let mut best = 0;
        for y in ys.clone() {
            let index = y * width + x;
            let original = source.as_raw()[index];
            if (original == 0) == contract {
                continue;
            }
            // Off-canvas pixels are unselected and only matter for contraction.
            if contract && (x + 1).min(width - x).min(y + 1).min(height - y) <= radius {
                output.as_mut()[index] = 0;
                continue;
            }
            let local_y = (y - top) as i32;
            while envelope.get(best + 1).is_some_and(|site| site.2 <= local_y) {
                best += 1;
            }
            let Some(&(seed_y, cost, _)) = envelope.get(best) else {
                continue;
            };
            let distance_squared = cost + (seed_y - local_y).pow(2);
            if distance_squared > radius_squared {
                continue;
            }
            output.as_mut()[index] = match modifier {
                Modifier::Expand => 255,
                Modifier::Contract => 0,
                Modifier::Feather => {
                    let seed_x = nearest_x[seed_y as usize * columns + local_x as usize] as usize;
                    let alpha = source.as_raw()[(top + seed_y as usize) * width + left + seed_x];
                    let weight = 1.0 - (distance_squared as f32).sqrt() / (radius + 1) as f32;
                    ((f32::from(alpha) * weight).round() as u8).max(1)
                }
            };
        }
    }
}
