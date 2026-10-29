//! Isolated codec experiment; never linked into PaintFE.
use std::{error::Error, time::Instant};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

fn timed<T>(runs: usize, mut f: impl FnMut() -> Result<T>) -> Result<T> {
    let mut times = Vec::new();
    let mut last = None;
    for _ in 0..runs.clamp(1, 20) {
        let start = Instant::now();
        last = Some(f()?);
        times.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    let first = times[0];
    times.sort_by(f64::total_cmp);
    println!(
        "first_ms={first:.3} median_ms={:.3} runs={}",
        times[times.len() / 2],
        times.len()
    );
    Ok(last.unwrap())
}

fn output(path: &str, bytes: &[u8]) -> Result<()> {
    std::fs::write(path, bytes)?;
    let hash = bytes.iter().fold(0xcbf29ce484222325_u64, |h, b| {
        (h ^ u64::from(*b)).wrapping_mul(0x100000001b3)
    });
    println!("output_bytes={} fnv1a={hash:016x}", bytes.len());
    Ok(())
}

fn run() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    match args.get(1).map(String::as_str).unwrap_or("baseline") {
        "baseline" => println!(
            "avif_encode={} avif_decode={}",
            cfg!(feature = "avif-encode"),
            cfg!(feature = "avif-decode")
        ),
        #[cfg(feature = "avif-encode")]
        "avif-encode" => {
            use image::ImageEncoder;
            if args.len() < 6 {
                return Err(
                    "avif-encode INPUT.rgba WIDTH HEIGHT OUTPUT.avif [SPEED] [QUALITY] [RUNS]"
                        .into(),
                );
            }
            let width: u32 = args[3].parse()?;
            let height: u32 = args[4].parse()?;
            if u64::from(width) * u64::from(height) > 32_000_000 {
                return Err("pixel limit".into());
            }
            let rgba = std::fs::read(&args[2])?;
            if rgba.len() != width as usize * height as usize * 4 {
                return Err("RGBA length mismatch".into());
            }
            let speed = args.get(6).map(|s| s.parse()).transpose()?.unwrap_or(8);
            let quality = args.get(7).map(|s| s.parse()).transpose()?.unwrap_or(80);
            let bytes = timed(
                args.get(8).map(|s| s.parse()).transpose()?.unwrap_or(3),
                || {
                    let mut bytes = Vec::new();
                    image::codecs::avif::AvifEncoder::new_with_speed_quality(
                        &mut bytes, speed, quality,
                    )
                    .with_num_threads(Some(4))
                    .write_image(
                        &rgba,
                        width,
                        height,
                        image::ExtendedColorType::Rgba8,
                    )?;
                    Ok(bytes)
                },
            )?;
            output(&args[5], &bytes)?;
        }
        #[cfg(feature = "avif-decode")]
        "avif-decode" => {
            if args.len() < 4 {
                return Err("avif-decode INPUT OUTPUT.rgba [RUNS]".into());
            }
            let bytes = std::fs::read(&args[2])?;
            let image = timed(
                args.get(4).map(|s| s.parse()).transpose()?.unwrap_or(5),
                || {
                    Ok(
                        image::load_from_memory_with_format(&bytes, image::ImageFormat::Avif)?
                            .to_rgba8(),
                    )
                },
            )?;
            println!("width={} height={}", image.width(), image.height());
            output(&args[3], image.as_raw())?;
        }
        _ => return Err("unknown command or missing codec feature".into()),
    }
    Ok(())
}

fn main() {
    match std::panic::catch_unwind(run) {
        Ok(Ok(())) => {}
        Ok(Err(error)) => {
            eprintln!("ERROR: {error}");
            std::process::exit(1);
        }
        Err(_) => {
            eprintln!("ERROR: codec panicked");
            std::process::exit(2);
        }
    }
}
