//! Bounded native raster building blocks for scripts, sharing the host sandbox and cancellation.
use super::*;
use crate::canvas::{BlendMode, CanvasState};
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

type Raster = (u32, u32, Arc<Vec<u8>>);

fn rgba(values: Array) -> Result<[u8; 4], Error> {
    if values.len() != 4 {
        return Err("Expected four RGBA channels".into());
    }
    let mut out = [0; 4];
    for (channel, value) in out.iter_mut().zip(values) {
        *channel = value
            .as_int()
            .ok()
            .and_then(|n| u8::try_from(n).ok())
            .ok_or("Invalid RGBA channel")?;
    }
    Ok(out)
}

fn channels(pixel: [u8; 4]) -> Array {
    pixel
        .into_iter()
        .map(|v| Dynamic::from(i64::from(v)))
        .collect()
}

fn mode(name: &str) -> Result<BlendMode, Error> {
    BlendMode::all()
        .iter()
        .copied()
        .find(|m| m.name().eq_ignore_ascii_case(name))
        .ok_or_else(|| "Unknown PaintFE blend mode".into())
}

fn bilinear(source: &[u8], w: u32, h: u32, x: f64, y: f64) -> [u8; 4] {
    if w == 0 || h == 0 {
        return [0; 4];
    }
    let x = x.clamp(0., f64::from(w - 1));
    let y = y.clamp(0., f64::from(h - 1));
    let x0 = x.floor() as u32;
    let y0 = y.floor() as u32;
    let (fx, fy) = (x.fract(), y.fract());
    let mut out = [0.; 4];
    for (px, wx) in [(x0, 1. - fx), ((x0 + 1).min(w - 1), fx)] {
        for (py, wy) in [(y0, 1. - fy), ((y0 + 1).min(h - 1), fy)] {
            let index = (py as usize * w as usize + px as usize) * 4;
            let Some(pixel) = source.get(index..index + 4) else {
                continue;
            };
            let weighted_alpha = wx * wy * f64::from(pixel[3]);
            out[3] += weighted_alpha;
            for c in 0..3 {
                out[c] += weighted_alpha * f64::from(pixel[c]);
            }
        }
    }
    if out[3].round() == 0. {
        return [0; 4];
    }
    [
        (out[0] / out[3]).round() as u8,
        (out[1] / out[3]).round() as u8,
        (out[2] / out[3]).round() as u8,
        out[3].round() as u8,
    ]
}

pub(super) fn register_image_tools(engine: &mut Engine, ctx: SharedContext, colours: [[u8; 4]; 2]) {
    use image::Rgba;
    engine.register_fn("primary_colour", move || channels(colours[0]));
    engine.register_fn("secondary_colour", move || channels(colours[1]));
    engine.register_fn(
        "blend_rgba",
        |bottom: Array, top: Array, name: ImmutableString, opacity: f64| -> Result<Array, Error> {
            Ok(channels(
                CanvasState::blend_pixel_static(
                    Rgba(rgba(bottom)?),
                    Rgba(rgba(top)?),
                    mode(&name)?,
                    bounded(opacity, 0., 1.)? as f32,
                )
                .0,
            ))
        },
    );
    engine.register_fn(
        "gradient_lut",
        |start: Array, end: Array, count: i64| -> Result<Array, Error> {
            if !(2..=256).contains(&count) {
                return Err("Gradient table requires 2 to 256 entries".into());
            }
            let (start, end) = (rgba(start)?, rgba(end)?);
            Ok((0..count)
                .map(|i| {
                    let t = i as f64 / (count - 1) as f64;
                    let pixel = std::array::from_fn(|c| {
                        (f64::from(start[c]) * (1. - t) + f64::from(end[c]) * t).round() as u8
                    });
                    Dynamic::from_array(channels(pixel))
                })
                .collect())
        },
    );

    let c = ctx.clone();
    engine.register_fn(
        "fill_rect",
        move |left: i64, top: i64, w: i64, h: i64, colour: Array| -> Result<(), Error> {
            if w <= 0 || h <= 0 {
                return Err("Rectangle dimensions must be positive".into());
            }
            let colour = Rgba(rgba(colour)?);
            let (right, bottom) = (left.saturating_add(w), top.saturating_add(h));
            map(&c, move |x, y, p, _, _, _| {
                if (left..right).contains(&i64::from(x)) && (top..bottom).contains(&i64::from(y)) {
                    CanvasState::blend_pixel_static(Rgba(p), colour, BlendMode::Normal, 1.).0
                } else {
                    p
                }
            })
        },
    );
    let c = ctx.clone();
    engine.register_fn(
        "fill_circle",
        move |cx: f64, cy: f64, radius: f64, colour: Array| -> Result<(), Error> {
            let (cx, cy, r) = (
                bounded(cx, -65536., 65536.)?,
                bounded(cy, -65536., 65536.)?,
                bounded(radius, 0., 65536.)?,
            );
            let colour = Rgba(rgba(colour)?);
            map(&c, move |x, y, p, _, _, _| {
                let distance = (f64::from(x) - cx).hypot(f64::from(y) - cy);
                let coverage = (r + 0.5 - distance).clamp(0., 1.) as f32;
                if coverage == 0. {
                    p
                } else {
                    CanvasState::blend_pixel_static(Rgba(p), colour, BlendMode::Normal, coverage).0
                }
            })
        },
    );
    let c = ctx.clone();
    engine.register_fn(
        "draw_line",
        move |x0: f64, y0: f64, x1: f64, y1: f64, width: f64, colour: Array| -> Result<(), Error> {
            let (x0, y0, x1, y1) = (
                bounded(x0, -65536., 65536.)?,
                bounded(y0, -65536., 65536.)?,
                bounded(x1, -65536., 65536.)?,
                bounded(y1, -65536., 65536.)?,
            );
            let r = bounded(width, 0.01, 65536.)? * 0.5;
            let colour = Rgba(rgba(colour)?);
            let (dx, dy) = (x1 - x0, y1 - y0);
            let length_sq = dx * dx + dy * dy;
            map(&c, move |x, y, p, _, _, _| {
                let (x, y) = (f64::from(x) - x0, f64::from(y) - y0);
                let t = if length_sq == 0. {
                    0.
                } else {
                    ((x * dx + y * dy) / length_sq).clamp(0., 1.)
                };
                let coverage = (r + 0.5 - (x - t * dx).hypot(y - t * dy)).clamp(0., 1.) as f32;
                if coverage == 0. {
                    p
                } else {
                    CanvasState::blend_pixel_static(Rgba(p), colour, BlendMode::Normal, coverage).0
                }
            })
        },
    );
    let c = ctx.clone();
    engine.register_fn(
        "sample_pixel",
        move |x: f64, y: f64| -> Result<Array, Error> {
            let (x, y) = (bounded(x, -1e9, 1e9)?, bounded(y, -1e9, 1e9)?);
            let image = c.lock().unwrap_or_else(|e| e.into_inner());
            Ok(channels(bilinear(
                &image.pixels,
                image.width,
                image.height,
                x,
                y,
            )))
        },
    );

    // Store opaque handles in the engine, with a combined limit across every live copy.
    let snapshots = Arc::new(Mutex::new(Vec::<Option<Raster>>::new()));
    let (store, c) = (snapshots.clone(), ctx.clone());
    engine.register_fn("snapshot_pixels", move || -> Result<i64, Error> {
        let image = c.lock().unwrap_or_else(|e| e.into_inner());
        let mut store = store.lock().unwrap_or_else(|e| e.into_inner());
        let live = store.iter().flatten();
        let bytes = live
            .clone()
            .map(|(_, _, pixels)| pixels.len())
            .sum::<usize>();
        if live.count() == 4
            || store.len() >= 1024
            || bytes.saturating_add(image.pixels.len()) > 64 * 1024 * 1024
        {
            return Err("Script raster snapshot budget exceeded".into());
        }
        store.push(Some((
            image.width,
            image.height,
            Arc::new(image.pixels.clone()),
        )));
        Ok(store.len() as i64 - 1)
    });
    let store = snapshots.clone();
    engine.register_fn(
        "release_snapshot",
        move |handle: i64| -> Result<(), Error> {
            store
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .get_mut(handle as usize)
                .and_then(Option::take)
                .ok_or("Invalid raster snapshot")?;
            Ok(())
        },
    );
    let store = snapshots.clone();
    engine.register_fn(
        "sample_snapshot",
        move |handle: i64, x: f64, y: f64| -> Result<Array, Error> {
            let (x, y) = (bounded(x, -1e9, 1e9)?, bounded(y, -1e9, 1e9)?);
            let (w, h, pixels) = snapshot(&store, handle)?;
            Ok(channels(bilinear(&pixels, w, h, x, y)))
        },
    );
    engine.register_fn(
        "blend_snapshot",
        move |handle: i64,
              dx: f64,
              dy: f64,
              name: ImmutableString,
              opacity: f64|
              -> Result<(), Error> {
            let (w, h, pixels) = snapshot(&snapshots, handle)?;
            let (dx, dy) = (bounded(dx, -1e9, 1e9)?, bounded(dy, -1e9, 1e9)?);
            let (mode, opacity) = (mode(&name)?, bounded(opacity, 0., 1.)? as f32);
            map(&ctx, move |x, y, p, _, _, _| {
                let (x, y) = (f64::from(x) - dx, f64::from(y) - dy);
                if x < 0. || y < 0. || x >= f64::from(w) || y >= f64::from(h) {
                    return p;
                }
                let top = Rgba(bilinear(&pixels, w, h, x, y));
                CanvasState::blend_pixel_static(Rgba(p), top, mode, opacity).0
            })
        },
    );
}

fn snapshot(store: &Mutex<Vec<Option<Raster>>>, handle: i64) -> Result<Raster, Error> {
    store
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(handle as usize)
        .and_then(Clone::clone)
        .ok_or_else(|| "Invalid raster snapshot".into())
}

#[cfg(test)]
mod image_tool_tests {
    use super::*;

    #[test]
    fn subpixel_sampling_ignores_rgb_of_transparent_pixels() {
        assert_eq!(
            bilinear(&[10, 20, 30, 255, 255, 255, 255, 0], 2, 1, 0.5, 0.),
            [10, 20, 30, 128]
        );
    }

    #[test]
    fn native_raster_tools_run_inside_the_sandbox() {
        let run = |script| execute_script_sync(script, [0, 0, 0, 255].repeat(16), 4, 4, None);
        let rgb = run(r#"
            let table = gradient_lut(primary_colour(), secondary_colour(), 3);
            let c = blend_rgba(table[0], table[1], "normal", 1.0);
            draw_line(0.0, 0.0, 3.0, 0.0, 1.0, c);
            let snapshot = snapshot_pixels();
            fill_circle(3.0, 3.0, 0.5, sample_snapshot(snapshot, 1.5, 0.0));
            blend_snapshot(snapshot, 0.0, 1.0, "screen", 1.0);
            release_snapshot(snapshot);
        "#)
        .unwrap()
        .0;
        assert_eq!(&rgb[4..8], &[128, 128, 128, 255]);
        assert!(rgb[60] > 0);
        assert!(run("snapshot_pixels(); snapshot_pixels(); snapshot_pixels(); snapshot_pixels(); snapshot_pixels();").is_err());
        assert!(run("gradient_lut(primary_colour(), secondary_colour(), 1000000);").is_err());
        assert!(
            run("let s = snapshot_pixels(); release_snapshot(s); sample_snapshot(s, 0.0, 0.0);")
                .is_err()
        );
    }
}
