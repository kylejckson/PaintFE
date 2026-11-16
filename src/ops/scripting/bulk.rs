//! Bounded native raster building blocks for scripts, sharing the host sandbox and cancellation.
use super::*;
use crate::par_compat::*;
type Error = Box<EvalAltResult>;
fn bounded(v: f64, min: f64, max: f64) -> Result<f64, Error> {
    if v.is_finite() && (min..=max).contains(&v) {
        Ok(v)
    } else {
        Err("Raster parameter outside its supported range".into())
    }
}
fn rgb(a: Array) -> Result<[f64; 3], Error> {
    if a.len() != 4 {
        return Err("Expected an RGBA colour".into());
    }
    let mut out = [0.; 3];
    for (i, v) in a.iter().enumerate() {
        let n = v.as_int().map_err(|_| "Expected integer RGBA channels")?;
        if !(0..=255).contains(&n) {
            return Err("Invalid colour channel".into());
        }
        if i < 3 {
            out[i] = n as f64;
        }
    }
    Ok(out)
}
fn map(
    ctx: &SharedContext,
    op: impl Fn(u32, u32, [u8; 4], &[u8], u32, u32) -> [u8; 4] + Send + Sync,
) -> Result<(), Error> {
    let (w, h, source, mask, cancel) = {
        let l = ctx.lock().unwrap_or_else(|e| e.into_inner());
        (
            l.width,
            l.height,
            l.pixels.clone(),
            l.mask.clone(),
            l.cancelled.clone(),
        )
    };
    if w == 0
        || h == 0
        || (w as usize)
            .checked_mul(h as usize)
            .and_then(|n| n.checked_mul(4))
            != Some(source.len())
    {
        return Err("Invalid raster dimensions".into());
    }
    if mask.as_ref().is_some_and(|m| m.len() != source.len() / 4) {
        return Err("Invalid selection dimensions".into());
    }
    let mut output = source.clone();
    let row = |y: usize, bytes: &mut [u8]| {
        if cancel.load(Ordering::Relaxed) {
            return;
        }
        for (x, p) in bytes.chunks_exact_mut(4).enumerate() {
            let offset = (y * w as usize + x) * 4;
            let before: [u8; 4] = source[offset..offset + 4].try_into().unwrap();
            let weight = mask
                .as_ref()
                .and_then(|m| m.get(y * w as usize + x))
                .copied()
                .unwrap_or(255);
            if weight == 0 {
                continue;
            }
            let after = op(x as u32, y as u32, before, &source, w, h);
            for c in 0..4 {
                p[c] = ((u32::from(before[c]) * u32::from(255 - weight)
                    + u32::from(after[c]) * u32::from(weight)
                    + 127)
                    / 255) as u8;
            }
        }
    };
    if source.len() > 16384 {
        output
            .par_chunks_mut(w as usize * 4)
            .enumerate()
            .for_each(|(y, b)| row(y, b));
    } else {
        for (y, b) in output.chunks_mut(w as usize * 4).enumerate() {
            row(y, b);
        }
    }
    if cancel.load(Ordering::Relaxed) {
        return Err("Script cancelled by user".into());
    }
    ctx.lock().unwrap_or_else(|e| e.into_inner()).pixels = output;
    Ok(())
}
pub(super) fn register(engine: &mut Engine, ctx: SharedContext) {
    let c = ctx.clone();
    engine.register_fn(
        "map_luminance",
        move |shadows: Array,
              highlights: Array,
              contrast: f64,
              strength: f64|
              -> Result<(), Error> {
            let a = rgb(shadows)?;
            let b = rgb(highlights)?;
            let contrast = bounded(contrast, 0., 8.)?;
            let strength = bounded(strength, 0., 1.)?;
            map(&c, move |_, _, p, _, _, _| {
                let l = (((f64::from(p[0]) * 0.2126
                    + f64::from(p[1]) * 0.7152
                    + f64::from(p[2]) * 0.0722)
                    / 255.
                    - 0.5)
                    * contrast
                    + 0.5)
                    .clamp(0., 1.);
                let mut out = p;
                for i in 0..3 {
                    out[i] = (f64::from(p[i]) * (1. - strength)
                        + (a[i] * (1. - l) + b[i] * l) * strength)
                        .round()
                        .clamp(0., 255.) as u8;
                }
                out
            })
        },
    );
    let c = ctx.clone();
    engine.register_fn(
        "apply_seeded_grain",
        move |amount: f64, size: i64, seed: i64, monochrome: bool| -> Result<(), Error> {
            let amount = bounded(amount, 0., 1.)?;
            if !(1..=256).contains(&size) || !(0..=1_000_000).contains(&seed) {
                return Err("Invalid grain size or seed".into());
            }
            map(&c, move |x, y, p, _, _, _| {
                let hx = (i64::from(x) / size) * 73856093;
                let hy = (i64::from(y) / size) * 19349663;
                let hs = seed * 83492791;
                let hash = (hx ^ hy ^ hs) & 2147483647;
                let noise = |div: i64| (((hash / div) % 1024) as f64 / 1023. - 0.5) * amount * 255.;
                let n = noise(1);
                let mut out = p;
                for (i, d) in [1, 1024, 1048576].into_iter().enumerate() {
                    out[i] = (f64::from(p[i]) + if monochrome { n } else { noise(d) })
                        .round()
                        .clamp(0., 255.) as u8;
                }
                out
            })
        },
    );
    let c = ctx.clone();
    engine.register_fn(
        "offset_channel",
        move |channel: i64, dx: i64, dy: i64, strength: f64| -> Result<(), Error> {
            let strength = bounded(strength, 0., 1.)?;
            if !(0..=2).contains(&channel)
                || !(-10000..=10000).contains(&dx)
                || !(-10000..=10000).contains(&dy)
            {
                return Err("Invalid channel offset".into());
            }
            map(&c, move |x, y, p, source, w, h| {
                let sx = (i64::from(x) - dx).clamp(0, i64::from(w) - 1) as usize;
                let sy = (i64::from(y) - dy).clamp(0, i64::from(h) - 1) as usize;
                let mut out = p;
                let i = channel as usize;
                out[i] = (f64::from(p[i]) * (1. - strength)
                    + f64::from(source[(sy * w as usize + sx) * 4 + i]) * strength)
                    .round() as u8;
                out
            })
        },
    );
    engine.register_fn(
        "apply_vignette_at",
        move |cx: f64, cy: f64, radius: f64, softness: f64, amount: f64| -> Result<(), Error> {
            let cx = bounded(cx, 0., 1.)?;
            let cy = bounded(cy, 0., 1.)?;
            let radius = bounded(radius, 0.01, 4.)?;
            let softness = bounded(softness, 0.01, 1.)?;
            let amount = bounded(amount, 0., 1.)?;
            map(&ctx, move |x, y, p, _, w, h| {
                let dx = (f64::from(x) + 0.5) / f64::from(w) - cx;
                let dy = (f64::from(y) + 0.5) / f64::from(h) - cy;
                let t =
                    ((dx.hypot(dy) - radius * (1. - softness)) / (radius * softness)).clamp(0., 1.);
                let f = 1. - amount * t * t * (3. - 2. * t);
                let mut out = p;
                for i in 0..3 {
                    out[i] = (f64::from(p[i]) * f).round() as u8;
                }
                out
            })
        },
    );
}
