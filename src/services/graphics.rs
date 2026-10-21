//! Prefer a known working native backend without removing automatic recovery.
use std::path::PathBuf;

const RECOVERY: &str = "PAINTFE_GRAPHICS_RECOVERY_PARENT";
fn cache_path() -> PathBuf {
    super::storage::initialize()
        .data_dir()
        .join("graphics_backend.cfg")
}

fn select(cache: &str, explicit: bool, recovery: bool, windows: bool) -> Option<wgpu::Backends> {
    if explicit || recovery || !windows {
        return None;
    }
    Some(match cache.trim() {
        "dx12" => wgpu::Backends::DX12,
        "gl" => wgpu::Backends::GL,
        _ => wgpu::Backends::VULKAN,
    })
}

pub fn preferred_backends() -> Option<wgpu::Backends> {
    let explicit = std::env::var_os("WGPU_BACKEND").is_some();
    let cache = if cfg!(target_os = "windows") && !explicit {
        std::fs::read_to_string(cache_path()).unwrap_or_default()
    } else {
        String::new()
    };
    select(
        &cache,
        explicit,
        std::env::var_os(RECOVERY).is_some(),
        cfg!(target_os = "windows"),
    )
}

pub fn record_working_backend(backend: wgpu::Backend) {
    let name = match backend {
        wgpu::Backend::Vulkan => "vulkan",
        wgpu::Backend::Dx12 => "dx12",
        wgpu::Backend::Gl => "gl",
        _ => return,
    };
    let marker = match name {
        "vulkan" => "backend_vulkan",
        "dx12" => "backend_dx12",
        _ => "backend_gl",
    };
    drop(crate::ui::perf::StartupSpan::new(marker));
    if cfg!(target_os = "windows") && std::env::var_os("WGPU_BACKEND").is_none() {
        crate::config::persistence::request(cache_path(), name.into());
    }
}

/// A replacement process is needed: winit cannot recreate its event loop.
/// Called only before an app/document has been constructed, at most once.
pub fn restart_with_automatic_backend() -> std::io::Result<()> {
    let mut command = std::process::Command::new(std::env::current_exe()?);
    command
        .args(std::env::args_os().skip(1))
        .env(RECOVERY, std::process::id().to_string());
    if let Some(path) = std::env::var_os("PAINTFE_UI_PROFILE") {
        command.env(
            "PAINTFE_UI_PROFILE",
            PathBuf::from(path).with_extension("recovery.jsonl"),
        );
    }
    command.spawn()?;
    Ok(())
}

/// Let the failed parent's single-instance pipe disappear before normal IPC.
pub fn wait_for_recovery_parent() -> Result<(), String> {
    #[cfg(target_os = "windows")]
    if let Some(pid) = std::env::var(RECOVERY)
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
    {
        #[link(name = "kernel32")]
        unsafe extern "system" {
            fn OpenProcess(access: u32, inherit: i32, pid: u32) -> isize;
            fn WaitForSingleObject(handle: isize, timeout: u32) -> u32;
            fn CloseHandle(handle: isize) -> i32;
            fn GetLastError() -> u32;
        }
        // SYNCHRONIZE grants waiting only, not control of the parent process.
        let handle = unsafe { OpenProcess(0x0010_0000, 0, pid) };
        if handle == 0 {
            let error = unsafe { GetLastError() };
            if error != 87 {
                return Err(format!("Cannot wait for graphics recovery parent: {error}"));
            }
        } else {
            let result = unsafe { WaitForSingleObject(handle, 10_000) };
            unsafe {
                CloseHandle(handle);
            }
            if result != 0 {
                return Err("Graphics recovery parent did not exit".into());
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn backend_policy_preserves_overrides_recovery_and_other_platforms() {
        assert_eq!(select("", false, false, true), Some(wgpu::Backends::VULKAN));
        assert_eq!(
            select("dx12\n", false, false, true),
            Some(wgpu::Backends::DX12)
        );
        assert_eq!(
            select("invalid", false, false, true),
            Some(wgpu::Backends::VULKAN)
        );
        for (explicit, recovery, windows) in [
            (true, false, true),
            (false, true, true),
            (false, false, false),
        ] {
            assert_eq!(select("vulkan", explicit, recovery, windows), None);
        }
    }
}
