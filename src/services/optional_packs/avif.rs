//! Bounded offline Image codec helper IPC, with cancellation and no shell execution.
use super::*;
use std::{
    process::Stdio,
    sync::atomic::AtomicU64,
    time::{Duration, Instant},
};
static GENERATION: AtomicU64 = AtomicU64::new(0);
static ACTIVE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static CODEC_LOCK: Mutex<()> = Mutex::new(());
const MAX_BYTES: usize = 512 * 1024 * 1024;

pub fn cancel() {
    GENERATION.fetch_add(1, Ordering::Relaxed);
}
pub fn active() -> bool {
    ACTIVE.load(Ordering::Relaxed) > 0
}
struct Active(bool);
impl Drop for Active {
    fn drop(&mut self) {
        if self.0 {
            ACTIVE.fetch_sub(1, Ordering::Relaxed);
        }
    }
}

pub fn exchange(header: [u8; 40], data: &[u8], tracked: bool) -> Result<Vec<u8>, String> {
    exchange_for(Pack::Avif, header, data, tracked)
}

pub(crate) fn exchange_for(
    pack: Pack,
    header: [u8; 40],
    data: &[u8],
    tracked: bool,
) -> Result<Vec<u8>, String> {
    let generation = GENERATION.load(Ordering::Relaxed);
    if tracked {
        ACTIVE.fetch_add(1, Ordering::Relaxed);
    }
    let _active = Active(tracked);
    let _lock = CODEC_LOCK.lock().map_err(|e| e.to_string())?;
    if generation != GENERATION.load(Ordering::Relaxed) {
        return Err(crate::t!("codec.cancelled"));
    }
    let helper = installed(pack).map_err(|_| {
        if pack == Pack::Heic {
            request_heic();
            crate::t!("packs.heic_missing")
        } else {
            request_avif();
            crate::t!("packs.avif_missing")
        }
    })?;
    let mut child = process::hidden_command(&helper)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| e.to_string())?;
    let overflow = AtomicBool::new(false);
    std::thread::scope(|scope| {
        let mut stdin = child.stdin.take().unwrap();
        let mut stdout = child.stdout.take().unwrap();
        let mut stderr = child.stderr.take().unwrap();
        let writer = scope.spawn(move || -> std::io::Result<()> {
            stdin.write_all(&header)?;
            stdin.write_all(data)
        });
        let overflow_ref = &overflow;
        let reader = scope.spawn(move || -> std::io::Result<Vec<u8>> {
            let mut bytes = Vec::new();
            let mut buffer = [0; 64 * 1024];
            loop {
                let count = stdout.read(&mut buffer)?;
                if count == 0 {
                    break;
                }
                if bytes.len() + count > MAX_BYTES + 32 {
                    overflow_ref.store(true, Ordering::Relaxed);
                    return Err(std::io::Error::other("Image codec output exceeds limit"));
                }
                bytes.extend_from_slice(&buffer[..count]);
            }
            Ok(bytes)
        });
        let errors = scope.spawn(move || -> std::io::Result<Vec<u8>> {
            let mut captured = Vec::new();
            let mut buffer = [0; 4096];
            loop {
                let count = stderr.read(&mut buffer)?;
                if count == 0 {
                    break;
                }
                let keep = count.min(8192usize.saturating_sub(captured.len()));
                captured.extend_from_slice(&buffer[..keep]);
            }
            Ok(captured)
        });
        let start = Instant::now();
        let result = loop {
            if generation != GENERATION.load(Ordering::Relaxed)
                || overflow.load(Ordering::Relaxed)
                || start.elapsed() > Duration::from_secs(120)
            {
                let _ = child.kill();
                let _ = child.wait();
                break Err(if generation != GENERATION.load(Ordering::Relaxed) {
                    crate::t!("codec.cancelled")
                } else {
                    "Image codec helper exceeded time or output limits".into()
                });
            }
            match child.try_wait() {
                Ok(Some(status)) => break Ok(status),
                Ok(None) => std::thread::sleep(Duration::from_millis(10)),
                Err(error) => {
                    let _ = child.kill();
                    let _ = child.wait();
                    break Err(error.to_string());
                }
            }
        };
        let written = writer.join().map_err(|_| "Image codec writer failed")?;
        let bytes = reader.join().map_err(|_| "Image codec reader failed")?;
        let errors = errors
            .join()
            .map_err(|_| "Image codec error reader failed")?
            .map_err(|e| e.to_string())?;
        let status = result?;
        if !status.success() {
            let diagnostic = String::from_utf8_lossy(&errors);
            return Err(if diagnostic.trim().is_empty() {
                format!("Image codec helper exited with {status}")
            } else {
                format!("AVIF: {}", diagnostic.trim())
            });
        }
        written.map_err(|e| e.to_string())?;
        bytes.map_err(|e| e.to_string())
    })
}
