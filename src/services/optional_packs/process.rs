use super::*;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};
pub(crate) fn hidden_command(path: &Path) -> Command {
    let command = Command::new(path);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        let mut command = command;
        command.creation_flags(0x08000000);
        command
    }
    #[cfg(not(windows))]
    command
}
pub(super) fn curl_path() -> PathBuf {
    #[cfg(windows)]
    {
        PathBuf::from(std::env::var_os("SystemRoot").unwrap_or_else(|| "C:\\Windows".into()))
            .join("System32/curl.exe")
    }
    #[cfg(not(windows))]
    {
        PathBuf::from("/usr/bin/curl")
    }
}
pub fn encode_jxl(
    width: u32,
    height: u32,
    bytes: &[u8],
    bits: u32,
    quality: u8,
    options: crate::io::jxl::JxlOptions,
) -> Result<Vec<u8>, String> {
    crate::io::validate_open_dimensions(width, height)?;
    let expected = u64::from(width) * u64::from(height) * 4 * u64::from(bits / 8);
    if !matches!(bits, 8 | 16) || expected != bytes.len() as u64 || expected > 1024 * 1024 * 1024 {
        return Err("Invalid or oversized JPEG XL input".into());
    }
    let _guard = LOCK.lock().map_err(|e| e.to_string())?;
    let helper = installed(Pack::Jxl).map_err(|_| crate::t!("packs.jxl_missing"))?;
    let mut child = hidden_command(&helper)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Could not start JPEG XL export: {e}"))?;
    let mut header = [0u8; 32];
    header[..8].copy_from_slice(b"PFEJXL01");
    for (offset, value) in [(8, width), (12, height), (16, bits)] {
        header[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }
    header[20] = quality;
    header[21] = options.effort;
    header[22] = u8::from(options.lossless);
    header[24..].copy_from_slice(&(bytes.len() as u64).to_le_bytes());
    // Drain pipes concurrently to avoid blocking on either a full pipe or early helper failure.
    std::thread::scope(|scope| {
        let mut input = child.stdin.take().unwrap();
        let mut output = child.stdout.take().unwrap();
        let mut errors = child.stderr.take().unwrap();
        let writer = scope.spawn(move || -> std::io::Result<()> {
            input.write_all(&header)?;
            #[cfg(target_endian = "big")]
            if bits == 16 {
                for sample in bytes.chunks_exact(2) {
                    input
                        .write_all(&u16::from_ne_bytes(sample.try_into().unwrap()).to_le_bytes())?;
                }
                return Ok(());
            }
            input.write_all(bytes)
        });
        let reader = scope.spawn(move || -> std::io::Result<Vec<u8>> {
            let mut bytes = Vec::new();
            output
                .by_ref()
                .take(1024 * 1024 * 1024 + 1)
                .read_to_end(&mut bytes)?;
            Ok(bytes)
        });
        let stderr = scope.spawn(move || {
            let mut bytes = Vec::new();
            let _ = errors.by_ref().take(8192).read_to_end(&mut bytes);
            bytes
        });
        let started = Instant::now();
        let result = loop {
            match child.try_wait() {
                Ok(Some(status)) => break Ok(status),
                Ok(None) if started.elapsed() < Duration::from_secs(300) => {
                    std::thread::sleep(Duration::from_millis(10))
                }
                Ok(None) => {
                    let _ = child.kill();
                    let _ = child.wait();
                    break Err("JPEG XL export timed out".to_string());
                }
                Err(e) => {
                    let _ = child.kill();
                    let _ = child.wait();
                    break Err(e.to_string());
                }
            }
        };
        let written = writer.join().map_err(|_| "JPEG XL input worker failed")?;
        let output = reader
            .join()
            .map_err(|_| "JPEG XL output worker failed")?
            .map_err(|e| e.to_string())?;
        let errors = stderr.join().map_err(|_| "JPEG XL error worker failed")?;
        if !result?.success() {
            return Err(format!(
                "JPEG XL export failed: {}",
                String::from_utf8_lossy(&errors)
            ));
        }
        written.map_err(|e| e.to_string())?;
        if !output.starts_with(b"\0\0\0\x0cJXL \r\n\x87\n") || output.len() > 1024 * 1024 * 1024 {
            return Err("Invalid JPEG XL helper output".into());
        }
        Ok(output)
    })
}
