//! Bounded, disposable processing. A worker never writes a document or performs network I/O.
use super::*;
use serde::{Deserialize, Serialize};
use std::{
    fs,
    io::Read,
    path::{Path, PathBuf},
    sync::atomic::{AtomicBool, Ordering},
    time::{Duration, Instant},
};
const MAX_PIXELS: usize = 64 * 1024 * 1024;
static SHUTDOWN: AtomicBool = AtomicBool::new(false);
static ACTIVE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
struct Active;
impl Drop for Active {
    fn drop(&mut self) {
        ACTIVE.fetch_sub(1, Ordering::Relaxed);
    }
}
pub fn shutdown() {
    SHUTDOWN.store(true, Ordering::Relaxed);
    let until = Instant::now() + Duration::from_secs(2);
    while ACTIVE.load(Ordering::Relaxed) > 0 && Instant::now() < until {
        std::thread::sleep(Duration::from_millis(10));
    }
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub source: String,
    pub width: u32,
    pub height: u32,
    pub values: Values,
    pub seed: u64,
    #[serde(default)]
    pub colours: Option<[[u8; 4]; 2]>,
    pub effect: Effect,
    pub has_mask: bool,
}
pub struct Output {
    pub pixels: Vec<u8>,
    pub console: Vec<String>,
}
struct Job(PathBuf);
impl Drop for Job {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn length(w: u32, h: u32) -> Result<usize, String> {
    let n = (w as usize)
        .checked_mul(h as usize)
        .and_then(|n| n.checked_mul(4))
        .ok_or("Invalid plugin image dimensions")?;
    if w == 0 || h == 0 || n > MAX_PIXELS {
        return Err("This image exceeds the plugin processing limit (16 million pixels).".into());
    }
    Ok(n)
}
pub fn execute(
    request: Request,
    pixels: Vec<u8>,
    mask: Option<Vec<u8>>,
    cancel: &AtomicBool,
) -> Result<Output, String> {
    if SHUTDOWN.load(Ordering::Relaxed) {
        return Err("Cancelled".into());
    }
    ACTIVE.fetch_add(1, Ordering::Relaxed);
    let _active = Active;
    let n = length(request.width, request.height)?;
    if pixels.len() != n
        || mask.as_ref().is_some_and(|m| m.len() != n / 4)
        || request.has_mask != mask.is_some()
    {
        return Err("Invalid plugin input".into());
    }
    request.effect.validate_values(&request.values)?;
    if request.source.len() > 256 * 1024 {
        return Err("Plugin source is too large".into());
    }
    let path = std::env::temp_dir().join(format!("paintfe-plugin-job-{}", uuid::Uuid::new_v4()));
    #[allow(unused_mut)]
    let mut builder = fs::DirBuilder::new();
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    builder.create(&path).map_err(|e| e.to_string())?;
    let job = Job(path);
    fs::write(
        job.0.join("request.json"),
        serde_json::to_vec(&request).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    fs::write(job.0.join("input"), pixels).map_err(|e| e.to_string())?;
    if let Some(m) = mask {
        fs::write(job.0.join("mask"), m).map_err(|e| e.to_string())?;
    }
    let mut exe = std::env::current_exe().map_err(|e| e.to_string())?;
    if exe
        .parent()
        .and_then(Path::file_name)
        .is_some_and(|p| p == "deps")
    {
        exe = exe
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join(if cfg!(windows) {
                "PaintFE.exe"
            } else {
                "PaintFE"
            });
    }
    let mut cmd = crate::services::optional_packs::process::hidden_command(&exe);
    cmd.arg("--paintfe-plugin-worker")
        .arg(&job.0)
        .env("RAYON_NUM_THREADS", "2")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null());
    let mut child = cmd
        .spawn()
        .map_err(|e| format!("Could not start plugin worker: {e}"))?;
    let start = Instant::now();
    loop {
        if cancel.load(Ordering::Relaxed)
            || SHUTDOWN.load(Ordering::Relaxed)
            || start.elapsed() > Duration::from_secs(45)
        {
            let _ = child.kill();
            let _ = child.wait();
            return Err(if cancel.load(Ordering::Relaxed) {
                "Cancelled".into()
            } else {
                "Plugin exceeded the 45 second time limit. The document was not changed.".into()
            });
        }
        match child.try_wait() {
            Ok(Some(s)) => {
                if !s.success() {
                    let message = read_limited(&job.0.join("error"), 8192)
                        .ok()
                        .map(|b| String::from_utf8_lossy(&b).into_owned())
                        .unwrap_or_else(|| {
                            "Plugin worker stopped. The document was not changed.".into()
                        });
                    return Err(message);
                }
                break;
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(15)),
            Err(e) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(e.to_string());
            }
        }
    }
    if cancel.load(Ordering::Relaxed) {
        return Err("Cancelled".into());
    }
    let pixels = read_limited(&job.0.join("output"), n)?;
    if pixels.len() != n {
        return Err("Invalid plugin output".into());
    }
    let console = read_limited(&job.0.join("console.json"), 128 * 1024)
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or_default();
    Ok(Output { pixels, console })
}
fn process(at: &Path) -> Result<(), String> {
    let r: Request =
        serde_json::from_slice(&read_limited(&at.join("request.json"), 2 * 1024 * 1024)?)
            .map_err(|e| e.to_string())?;
    let n = length(r.width, r.height)?;
    r.effect.validate_values(&r.values)?;
    let input = read_limited(&at.join("input"), n)?;
    if input.len() != n {
        return Err("Invalid input length".into());
    }
    let original = input.clone();
    let mask = if r.has_mask {
        let m = read_limited(&at.join("mask"), n / 4)?;
        if m.len() != n / 4 {
            return Err("Invalid selection mask".into());
        }
        Some(m)
    } else {
        None
    };
    let seed = r.effect.parameters.iter().find(|p| p.kind == "seed")
        .and_then(|p| r.values.get(&p.id))
        .and_then(serde_json::Value::as_u64)
        .map(|seed| seed + 1)
        .unwrap_or(r.seed);
    let (mut output, w, h, console, ops) = crate::ops::scripting::execute_script_with_colours(
        &r.source,
        input,
        r.width,
        r.height,
        mask.clone(),
        &r.values,
        Some(seed),
        r.colours.unwrap_or(crate::ops::scripting::DEFAULT_SCRIPT_COLOURS),
    )
    .map_err(|e| e.friendly_message())?;
    if w != r.width || h != r.height || !ops.is_empty() || output.len() != n {
        return Err("Plugin effects cannot change canvas dimensions. Use a normal script for canvas transforms.".into());
    }
    // Selection isolation is enforced by the host even if a script ignores it.
    if let Some(mask) = mask {
        for (i, (dst, src)) in output
            .chunks_exact_mut(4)
            .zip(original.chunks_exact(4))
            .enumerate()
        {
            if mask[i] == 0 {
                dst.copy_from_slice(src);
            }
        }
    }
    fs::write(at.join("output"), output).map_err(|e| e.to_string())?;
    fs::write(
        at.join("console.json"),
        serde_json::to_vec(&console).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())
}
pub fn run_worker() -> i32 {
    let Some(path) = std::env::args_os().nth(2).map(PathBuf::from) else {
        return 2;
    };
    std::thread::spawn(|| {
        let mut b = [0];
        let _ = std::io::stdin().read(&mut b);
        std::process::exit(2);
    });
    match crate::io::heic::apply_worker_limits().and_then(|()| process(&path)) {
        Ok(()) => 0,
        Err(e) => {
            let _ = fs::write(path.join("error"), e);
            1
        }
    }
}
