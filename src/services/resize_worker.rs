//! Disposable same-executable resize jobs. Pixel transfer is file-backed and bounded.
use crate::canvas::{CHUNK_SIZE, TiledImage};
use crate::ops::transform::{ResizeImageOptions, ResizeMethod};
use image::{GrayImage, RgbaImage};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

static ACTIVE: AtomicUsize = AtomicUsize::new(0);
static JOBS: AtomicUsize = AtomicUsize::new(0);
struct Running;
impl Drop for Running {
    fn drop(&mut self) {
        JOBS.fetch_sub(1, Ordering::Relaxed);
    }
}
pub fn drain_cancelled() {
    let deadline = Instant::now() + Duration::from_secs(2);
    while JOBS.load(Ordering::Relaxed) > 0 && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(5));
    }
}
pub fn active_workers() -> usize {
    ACTIVE.load(Ordering::Relaxed)
}
fn cancelled(cancel: &AtomicBool) -> Result<(), String> {
    if cancel.load(Ordering::Relaxed) {
        Err("Cancelled".into())
    } else {
        Ok(())
    }
}
fn io(error: std::io::Error) -> String {
    format!("Resize worker: {error}")
}
struct Job(PathBuf);
impl Drop for Job {
    fn drop(&mut self) {
        for name in ["input", "output", "error", "working"] {
            let _ = fs::remove_file(self.0.join(name));
        }
        let _ = fs::remove_dir(&self.0);
    }
}
impl Job {
    fn new() -> Result<Self, String> {
        let root = std::env::temp_dir();
        #[cfg(debug_assertions)]
        let root = std::env::var_os("PAINTFE_TEST_STORAGE_DIR")
            .map(PathBuf::from)
            .map(|p| p.join("resize-jobs"))
            .unwrap_or(root);
        fs::create_dir_all(&root).map_err(io)?;
        let path = root.join(format!("paintfe-resize-{}", uuid::Uuid::new_v4()));
        let builder = fs::DirBuilder::new();
        #[cfg(unix)]
        let builder = {
            use std::os::unix::fs::DirBuilderExt;
            let mut builder = builder;
            builder.mode(0o700);
            builder
        };
        builder.create(&path).map_err(io)?;
        Ok(Self(path))
    }
}
fn worker_executable() -> Result<PathBuf, String> {
    let path = std::env::current_exe().map_err(io)?;
    if path
        .parent()
        .and_then(Path::file_name)
        .is_some_and(|p| p == "deps")
    {
        return Ok(path
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join(if cfg!(windows) {
                "PaintFE.exe"
            } else {
                "PaintFE"
            }));
    }
    Ok(path)
}
fn write_pixels(file: &mut File, image: &TiledImage, cancel: &AtomicBool) -> Result<(), String> {
    let mut rows = Vec::new();
    for y in (0..image.height).step_by(CHUNK_SIZE as usize) {
        cancelled(cancel)?;
        image.extract_region_rgba_fast(
            0,
            y,
            image.width,
            CHUNK_SIZE.min(image.height - y),
            &mut rows,
        );
        file.write_all(&rows).map_err(io)?;
    }
    Ok(())
}
fn read_pixels(
    file: &mut File,
    width: u32,
    height: u32,
    cancel: &AtomicBool,
) -> Result<TiledImage, String> {
    let mut image = TiledImage::new(width, height);
    for y in (0..height).step_by(CHUNK_SIZE as usize) {
        cancelled(cancel)?;
        let rows = CHUNK_SIZE.min(height - y);
        let mut bytes = vec![0; width as usize * rows as usize * 4];
        file.read_exact(&mut bytes).map_err(io)?;
        for x in (0..width).step_by(CHUNK_SIZE as usize) {
            cancelled(cancel)?;
            let columns = CHUNK_SIZE.min(width - x);
            let mut chunk = RgbaImage::new(CHUNK_SIZE, CHUNK_SIZE);
            for row in 0..rows as usize {
                let start = (row * width as usize + x as usize) * 4;
                let dest = row * CHUNK_SIZE as usize * 4;
                chunk.as_mut()[dest..dest + columns as usize * 4]
                    .copy_from_slice(&bytes[start..start + columns as usize * 4]);
            }
            if chunk.as_raw().chunks_exact(4).any(|p| p[3] != 0) {
                image.set_chunk(x / CHUNK_SIZE, y / CHUNK_SIZE, chunk);
            }
        }
    }
    Ok(image)
}

pub fn resize_layers(
    layers: Vec<TiledImage>,
    width: u32,
    height: u32,
    options: ResizeImageOptions,
    mask: Option<GrayImage>,
    cancel: &AtomicBool,
) -> Result<Vec<TiledImage>, String> {
    crate::io::validate_open_dimensions(width, height)?;
    JOBS.fetch_add(1, Ordering::Relaxed);
    let _running = Running;
    let mut result = Vec::with_capacity(layers.len());
    for layer in layers {
        cancelled(cancel)?;
        let job = Job::new()?;
        let options = serde_json::to_vec(&options).map_err(|e| e.to_string())?;
        let use_mask = mask
            .as_ref()
            .filter(|m| m.dimensions() == (layer.width, layer.height));
        {
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(job.0.join("input"))
                .map_err(io)?;
            file.write_all(b"PFERES01").map_err(io)?;
            for value in [
                layer.width,
                layer.height,
                width,
                height,
                options.len() as u32,
                u32::from(use_mask.is_some()),
            ] {
                file.write_all(&value.to_le_bytes()).map_err(io)?;
            }
            file.write_all(&options).map_err(io)?;
            write_pixels(&mut file, &layer, cancel)?;
            if let Some(mask) = use_mask {
                for row in mask.as_raw().chunks(layer.width as usize) {
                    cancelled(cancel)?;
                    file.write_all(row).map_err(io)?;
                }
            }
        }
        cancelled(cancel)?;
        let mut child = crate::services::optional_packs::hidden_command(&worker_executable()?)
            .arg("--paintfe-resize-worker")
            .arg(&job.0)
            .env("RAYON_NUM_THREADS", "2")
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .map_err(io)?;
        let started = Instant::now();
        let mut working = false;
        let status = loop {
            if !working && job.0.join("working").is_file() {
                ACTIVE.fetch_add(1, Ordering::Relaxed);
                working = true;
            }
            if cancel.load(Ordering::Relaxed) || started.elapsed() > Duration::from_secs(600) {
                let _ = child.kill();
                let _ = child.wait();
                break Err(if cancel.load(Ordering::Relaxed) {
                    "Cancelled".to_string()
                } else {
                    "Image resizing exceeded the time limit".to_string()
                });
            }
            match child.try_wait() {
                Ok(Some(status)) => break Ok(status),
                Ok(None) => std::thread::sleep(Duration::from_millis(10)),
                Err(error) => {
                    let _ = child.kill();
                    let _ = child.wait();
                    break Err(io(error));
                }
            }
        };
        if working {
            ACTIVE.fetch_sub(1, Ordering::Relaxed);
        }
        let status = status?;
        if !status.success() {
            let mut message = String::new();
            if let Ok(file) = File::open(job.0.join("error")) {
                let _ = file.take(8192).read_to_string(&mut message);
            }
            return Err(if message.is_empty() {
                "Image resizing failed. The document was not changed.".into()
            } else {
                message
            });
        }
        cancelled(cancel)?;
        let mut file = File::open(job.0.join("output")).map_err(io)?;
        if file.metadata().map_err(io)?.len() != u64::from(width) * u64::from(height) * 4 {
            return Err("Invalid resize output length".into());
        }
        result.push(read_pixels(&mut file, width, height, cancel)?);
    }
    cancelled(cancel)?;
    Ok(result)
}

fn process(directory: &Path) -> Result<(), String> {
    let mut input = File::open(directory.join("input")).map_err(io)?;
    let mut header = [0; 32];
    input.read_exact(&mut header).map_err(io)?;
    if &header[..8] != b"PFERES01" {
        return Err("Invalid resize protocol".into());
    }
    let number = |offset| u32::from_le_bytes(header[offset..offset + 4].try_into().unwrap());
    let (sw, sh, w, h, len, mask) = (
        number(8),
        number(12),
        number(16),
        number(20),
        number(24),
        number(28),
    );
    crate::io::validate_open_dimensions(sw, sh)?;
    crate::io::validate_open_dimensions(w, h)?;
    if len > 4096
        || mask > 1
        || input.metadata().map_err(io)?.len()
            != 32 + u64::from(len) + u64::from(sw) * u64::from(sh) * (4 + u64::from(mask))
    {
        return Err("Invalid resize input length".into());
    }
    let mut options = vec![0; len as usize];
    input.read_exact(&mut options).map_err(io)?;
    let options: ResizeImageOptions =
        serde_json::from_slice(&options).map_err(|e| e.to_string())?;
    let mut pixels = vec![0; sw as usize * sh as usize * 4];
    input.read_exact(&mut pixels).map_err(io)?;
    let source = RgbaImage::from_raw(sw, sh, pixels).ok_or("Invalid resize pixels")?;
    let mask = if mask == 1 {
        let mut bits = vec![0; sw as usize * sh as usize];
        input.read_exact(&mut bits).map_err(io)?;
        Some(GrayImage::from_raw(sw, sh, bits).ok_or("Invalid resize mask")?)
    } else {
        None
    };
    File::create(directory.join("working")).map_err(io)?;
    let output = match options.method {
        ResizeMethod::Standard => {
            image::imageops::resize(&source, w, h, options.interpolation.to_filter())
        }
        ResizeMethod::PixelArtRetarget => crate::ops::transform::pixel_art_retarget(
            &source,
            w,
            h,
            &options.retarget,
            mask.as_ref(),
        ),
    };
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(directory.join("output"))
        .map_err(io)?;
    file.write_all(output.as_raw()).map_err(io)
}
pub fn run_worker() -> i32 {
    let Some(directory) = std::env::args_os().nth(2).map(PathBuf::from) else {
        return 2;
    };
    // Parent owns the pipe. Its closure also stops the worker if the editor exits.
    std::thread::spawn(|| {
        let mut byte = [0];
        let _ = std::io::stdin().read(&mut byte);
        std::process::exit(2);
    });
    let result = crate::io::heic::apply_worker_limits().and_then(|()| process(&directory));
    match result {
        Ok(()) => 0,
        Err(error) => {
            let _ = fs::write(directory.join("error"), error);
            1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn malformed_resize_requests_fail_before_allocation_and_cleanup() {
        for header in [
            [0u8; 32],
            {
                let mut header = [0; 32];
                header[..8].copy_from_slice(b"PFERES01");
                for (offset, value) in [(8, 1u32), (12, 1), (16, 20000), (20, 20000)] {
                    header[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
                }
                header
            },
            {
                let mut header = [0; 32];
                header[..8].copy_from_slice(b"PFERES01");
                for offset in [8, 12, 16, 20] {
                    header[offset..offset + 4].copy_from_slice(&1u32.to_le_bytes());
                }
                header[24..28].copy_from_slice(&u32::MAX.to_le_bytes());
                header
            },
        ] {
            let job = Job::new().unwrap();
            let root = job.0.clone();
            fs::write(root.join("input"), header).unwrap();
            assert!(process(&root).is_err());
            assert!(!root.join("output").exists());
            drop(job);
            assert!(!root.exists());
        }
    }
}
