//! Prefer a known working native backend without removing automatic recovery.
use std::path::PathBuf;

static STARTUP_SELECTED: std::sync::Mutex<String> = std::sync::Mutex::new(String::new());
static RUNNING_ADAPTER: std::sync::Mutex<Option<wgpu::AdapterInfo>> = std::sync::Mutex::new(None);
static AVAILABLE_ADAPTERS: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

#[cfg(not(target_arch = "wasm32"))]
fn adapter_keys(infos: impl IntoIterator<Item = wgpu::AdapterInfo>) -> Vec<String> {
    let mut keys = vec!["Auto".to_owned()];
    for info in infos {
        if matches!(
            info.device_type,
            wgpu::DeviceType::Cpu | wgpu::DeviceType::Other
        ) {
            continue;
        }
        let key = adapter_key(&info);
        if !keys.contains(&key) {
            keys.push(key);
        }
    }
    keys
}

pub fn record_render_state(state: &eframe::egui_wgpu::RenderState) {
    record_adapter(&state.adapter.get_info());
    #[cfg(not(target_arch = "wasm32"))]
    {
        *AVAILABLE_ADAPTERS.lock().unwrap_or_else(|e| e.into_inner()) = adapter_keys(
            state
                .available_adapters
                .iter()
                .chain(std::iter::once(&state.adapter))
                .map(|a| a.get_info()),
        );
    }
}

/// Reuse startup's inventory: creating another graphics instance from the
/// Preferences page can disturb a live device on some Windows drivers.
pub fn available_adapter_keys() -> Vec<String> {
    let keys = AVAILABLE_ADAPTERS.lock().unwrap_or_else(|e| e.into_inner());
    if keys.is_empty() {
        vec!["Auto".to_owned()]
    } else {
        keys.clone()
    }
}

pub fn adapter_key(info: &wgpu::AdapterInfo) -> String {
    format!("gpu:{:x}:{:x}:{}", info.vendor, info.device, info.name)
}

pub fn adapter_label(key: &str) -> &str {
    if key.starts_with("gpu:") {
        key.splitn(4, ':').nth(3).unwrap_or(key)
    } else {
        key
    }
}

pub fn adapter_matches(info: &wgpu::AdapterInfo, selected: &str) -> bool {
    selected == adapter_key(info) || selected == info.name
}

pub fn record_adapter(info: &wgpu::AdapterInfo) {
    *RUNNING_ADAPTER.lock().unwrap_or_else(|e| e.into_inner()) = Some(info.clone());
}

pub fn running_adapter() -> String {
    RUNNING_ADAPTER
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .map(|a| format!("{} ({:?})", a.name, a.backend))
        .unwrap_or_else(|| crate::t!("graphics.not_initialized"))
}

pub fn selection_notice(selected: &str) -> Option<String> {
    let startup = STARTUP_SELECTED.lock().unwrap_or_else(|e| e.into_inner());
    if !startup.is_empty() && selected != startup.as_str() {
        return Some(
            crate::t!("graphics.selection_pending").replace("{0}", adapter_label(selected)),
        );
    }
    let current = RUNNING_ADAPTER.lock().unwrap_or_else(|e| e.into_inner());
    if selected == "Auto"
        || current
            .as_ref()
            .is_some_and(|a| adapter_matches(a, selected))
    {
        return None;
    }
    Some(crate::t!("graphics.selection_fallback").replace("{0}", adapter_label(selected)))
}

fn choose_adapter(
    infos: &[wgpu::AdapterInfo],
    compatible: &[bool],
    selected: &str,
) -> Option<usize> {
    let candidates = || {
        infos
            .iter()
            .enumerate()
            .filter(|(i, _)| compatible.get(*i) == Some(&true))
    };
    candidates()
        .find(|(_, a)| adapter_matches(a, selected))
        .map(|(i, _)| i)
        .or_else(|| {
            candidates()
                .max_by_key(|(_, a)| match a.device_type {
                    wgpu::DeviceType::DiscreteGpu => 4,
                    wgpu::DeviceType::IntegratedGpu => 3,
                    wgpu::DeviceType::VirtualGpu => 2,
                    _ => 1,
                })
                .map(|(i, _)| i)
        })
}

/// Filter by presentation support before choosing a requested physical device.
/// Older preferences containing only a name remain valid.
pub fn configure_adapter(config: &mut eframe::egui_wgpu::WgpuConfiguration, selected: &str) {
    *STARTUP_SELECTED.lock().unwrap_or_else(|e| e.into_inner()) = selected.to_owned();
    if selected == "Auto" || std::env::var_os(RECOVERY).is_some() {
        return;
    }
    if let eframe::egui_wgpu::WgpuSetup::CreateNew(setup) = &mut config.wgpu_setup {
        let selected = selected.to_owned();
        setup.native_adapter_selector = Some(std::sync::Arc::new(move |adapters, surface| {
            let infos: Vec<_> = adapters.iter().map(|a| a.get_info()).collect();
            let compatible: Vec<_> = adapters
                .iter()
                .map(|a| surface.is_none_or(|s| a.is_surface_supported(s)))
                .collect();
            choose_adapter(&infos, &compatible, &selected)
                .map(|i| adapters[i].clone())
                .ok_or_else(|| "No compatible graphics adapter".into())
        }));
    }
}

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
    fn adapter(name: &str, vendor: u32, kind: wgpu::DeviceType) -> wgpu::AdapterInfo {
        wgpu::AdapterInfo {
            name: name.into(),
            vendor,
            device: 1,
            device_type: kind,
            device_pci_bus_id: String::new(),
            driver: String::new(),
            driver_info: String::new(),
            backend: wgpu::Backend::Vulkan,
            subgroup_min_size: 4,
            subgroup_max_size: 32,
            transient_saves_memory: false,
        }
    }
    #[test]
    fn explicit_physical_choice_legacy_names_and_fallbacks() {
        let infos = [
            adapter("Same name", 1, wgpu::DeviceType::DiscreteGpu),
            adapter("Same name", 2, wgpu::DeviceType::IntegratedGpu),
        ];
        assert_eq!(
            choose_adapter(&infos, &[true, true], &adapter_key(&infos[1])),
            Some(1)
        );
        assert_eq!(adapter_label(&adapter_key(&infos[1])), "Same name");
        assert_eq!(choose_adapter(&infos, &[true, true], "Same name"), Some(0));
        assert_eq!(
            choose_adapter(&infos, &[true, false], &adapter_key(&infos[1])),
            Some(0)
        );
        assert_eq!(choose_adapter(&infos, &[false, false], "Missing"), None);
        let mut config = eframe::egui_wgpu::WgpuConfiguration::default();
        configure_adapter(&mut config, "Auto");
        if let eframe::egui_wgpu::WgpuSetup::CreateNew(setup) = &config.wgpu_setup {
            assert!(setup.native_adapter_selector.is_none());
        }
        configure_adapter(&mut config, &adapter_key(&infos[1]));
        if let eframe::egui_wgpu::WgpuSetup::CreateNew(setup) = &config.wgpu_setup {
            assert!(setup.native_adapter_selector.is_some());
        }
    }

    #[test]
    fn startup_inventory_deduplicates_backends_without_merging_physical_choices() {
        let a = adapter("Same name", 1, wgpu::DeviceType::DiscreteGpu);
        let mut duplicate = a.clone();
        duplicate.backend = wgpu::Backend::Dx12;
        let b = adapter("Same name", 2, wgpu::DeviceType::IntegratedGpu);
        let cpu = adapter("Software", 3, wgpu::DeviceType::Cpu);
        assert_eq!(
            adapter_keys([a.clone(), duplicate, b.clone(), cpu]),
            vec!["Auto".to_owned(), adapter_key(&a), adapter_key(&b)]
        );
    }
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
