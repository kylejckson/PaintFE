// GUI-subsystem binary: no console window is ever allocated by Windows.
// • GUI mode: nothing extra needed — no console to free.
// • CLI mode (--input/-i flag present): AttachConsole(ATTACH_PARENT_PROCESS) attaches to
//   the launching terminal, then we reopen CONOUT$/CONIN$ so Rust's println!/eprintln!
//   route through the correct handles (necessary when SUBSYSTEM:WINDOWS is set).
#![windows_subsystem = "windows"]

use eframe::egui;
use paintfe::app::PaintFEApp;
use paintfe::assets::AppSettings;
use paintfe::cli;
use paintfe::i18n;
use paintfe::ipc;
use paintfe::logger;

/// Install a Windows SEH unhandled exception filter that catches assertion
/// failures and access violations from third-party injected DLLs (e.g. overlay
/// hooks like AlphaRing/WTSAPI32.dll) so they don't crash PaintFE.
///
/// Returns `true` if the exception was handled (process should continue),
/// `false` if it should be passed to the default handler.
#[cfg(target_os = "windows")]
unsafe extern "system" fn seh_exception_filter(
    exception_info: *mut winapi::um::winnt::EXCEPTION_POINTERS,
) -> i32 {
    use winapi::um::minwinbase::{EXCEPTION_ACCESS_VIOLATION, EXCEPTION_INT_DIVIDE_BY_ZERO};

    // EXCEPTION_ASSERTION_FAILURE is not in winapi; it's 0x42424242 (MSVC __debugbreak)
    const EXCEPTION_ASSERTION_FAILURE: u32 = 0x4242_4242;

    let record = unsafe { &*(*exception_info).ExceptionRecord };
    let exception_code = record.ExceptionCode;

    // Catch assertion failures, access violations, and divide-by-zero from
    // injected DLLs. These are almost always from third-party overlays/hooks.
    let is_injected_dll_exception = exception_code == EXCEPTION_ASSERTION_FAILURE
        || exception_code == EXCEPTION_ACCESS_VIOLATION
        || exception_code == EXCEPTION_INT_DIVIDE_BY_ZERO;

    if is_injected_dll_exception {
        // Log the exception for diagnostics
        let exc_addr = record.ExceptionAddress as usize;
        eprintln!(
            "[PaintFE] Caught SEH exception 0x{:08X} at 0x{:X} from injected DLL — ignoring",
            exception_code, exc_addr
        );
        paintfe::logger::write(
            "WARN",
            &format!(
                "Caught SEH exception 0x{:08X} at 0x{:X} from injected DLL — ignoring",
                exception_code, exc_addr
            ),
        );

        // EXCEPTION_CONTINUE_EXECUTION = 0 — try to continue past the fault
        // This works for assertion failures (int 3 / __debugbreak) by skipping
        // the instruction. For access violations, we skip the faulting instruction.
        0 // EXCEPTION_CONTINUE_EXECUTION
    } else {
        // EXCEPTION_CONTINUE_SEARCH = 1 — pass to next handler
        1
    }
}

/// Wrap the main function body with Windows SEH protection on Windows.
#[cfg(target_os = "windows")]
fn run_with_seh_protection<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    use winapi::um::errhandlingapi::SetUnhandledExceptionFilter;

    // Install our SEH filter before running the app
    let prev_filter = unsafe { SetUnhandledExceptionFilter(Some(seh_exception_filter)) };

    let result = f();

    // Restore the previous filter (good practice, though process is exiting)
    unsafe { SetUnhandledExceptionFilter(prev_filter) };

    result
}

#[cfg(target_os = "windows")]
fn sanitize_window_position(pos: (f32, f32)) -> Option<[f32; 2]> {
    unsafe extern "system" {
        fn GetSystemMetrics(nIndex: i32) -> i32;
    }
    const SM_XVIRTUALSCREEN: i32 = 76;
    const SM_YVIRTUALSCREEN: i32 = 77;
    const SM_CXVIRTUALSCREEN: i32 = 78;
    const SM_CYVIRTUALSCREEN: i32 = 79;

    let vx = unsafe { GetSystemMetrics(SM_XVIRTUALSCREEN) } as f32;
    let vy = unsafe { GetSystemMetrics(SM_YVIRTUALSCREEN) } as f32;
    let vw = unsafe { GetSystemMetrics(SM_CXVIRTUALSCREEN) } as f32;
    let vh = unsafe { GetSystemMetrics(SM_CYVIRTUALSCREEN) } as f32;

    if vw <= 0.0 || vh <= 0.0 {
        return Some([pos.0, pos.1]);
    }

    let margin = 16.0;
    let min_x = vx;
    let min_y = vy;
    let max_x = vx + vw - margin;
    let max_y = vy + vh - margin;
    Some([pos.0.clamp(min_x, max_x), pos.1.clamp(min_y, max_y)])
}

#[cfg(not(target_os = "windows"))]
fn sanitize_window_position(pos: (f32, f32)) -> Option<[f32; 2]> {
    Some([pos.0, pos.1])
}

/// Inner main function — contains all actual logic.
fn main_inner() -> Result<(), eframe::Error> {
    let startup_profile = paintfe::ui::perf::StartupSpan::new("process_to_event_loop");
    paintfe::io::jxl::register();
    // -- Windows console management ------------------------------------
    // The binary is SUBSYSTEM:WINDOWS so Windows never allocates a console.
    // In CLI mode preserve inherited redirection and connect missing handles
    // to the parent terminal for Rust's Unicode console output.
    #[cfg(target_os = "windows")]
    if cli::CliArgs::is_cli_mode() {
        unsafe extern "system" {
            fn AttachConsole(dwProcessId: u32) -> i32;
            fn SetStdHandle(nStdHandle: u32, hHandle: isize) -> i32;
            fn GetStdHandle(nStdHandle: u32) -> isize;
            fn GetFileType(hFile: isize) -> u32;
            fn CloseHandle(hObject: isize) -> i32;
            fn CreateFileW(
                lpFileName: *const u16,
                dwDesiredAccess: u32,
                dwShareMode: u32,
                lpSecurityAttributes: *const std::ffi::c_void,
                dwCreationDisposition: u32,
                dwFlagsAndAttributes: u32,
                hTemplateFile: isize,
            ) -> isize;
        }
        const ATTACH_PARENT_PROCESS: u32 = 0xFFFF_FFFF;
        const GENERIC_READ: u32 = 0x8000_0000;
        const GENERIC_WRITE: u32 = 0x4000_0000;
        const FILE_SHARE_READ_WRITE: u32 = 0x0000_0003;
        const OPEN_EXISTING: u32 = 3;
        const STD_INPUT_HANDLE: u32 = 0xFFFF_FFF6_u32; // -10
        const STD_OUTPUT_HANDLE: u32 = 0xFFFF_FFF5_u32; // -11
        const STD_ERROR_HANDLE: u32 = 0xFFFF_FFF4_u32; // -12
        const INVALID_HANDLE_VALUE: isize = -1;
        unsafe {
            // AttachConsole can replace inherited standard handles. Retain valid
            // ones first, including pipes, redirected files and NUL, per stream.
            let inherited = [STD_INPUT_HANDLE, STD_OUTPUT_HANDLE, STD_ERROR_HANDLE].map(|kind| {
                let handle = GetStdHandle(kind);
                (
                    kind,
                    handle,
                    handle != 0 && handle != INVALID_HANDLE_VALUE && GetFileType(handle) != 0,
                )
            });
            AttachConsole(ATTACH_PARENT_PROCESS);
            let conout: Vec<u16> = "CONOUT$\0".encode_utf16().collect();
            let conin: Vec<u16> = "CONIN$\0".encode_utf16().collect();
            for (kind, handle, valid) in inherited {
                if valid {
                    SetStdHandle(kind, handle);
                    continue;
                }
                // Rust uses GetConsoleMode to select Unicode console output.
                // It needs GENERIC_READ even on the output buffer; write-only
                // access makes UTF-8 bytes fall through to the legacy codepage.
                let name = if kind == STD_INPUT_HANDLE {
                    &conin
                } else {
                    &conout
                };
                let replacement = CreateFileW(
                    name.as_ptr(),
                    GENERIC_READ | GENERIC_WRITE,
                    FILE_SHARE_READ_WRITE,
                    std::ptr::null(),
                    OPEN_EXISTING,
                    0,
                    0,
                );
                if replacement != INVALID_HANDLE_VALUE && SetStdHandle(kind, replacement) == 0 {
                    CloseHandle(replacement);
                }
            }
        }
    }

    // -- CLI / headless mode ---------------------------------------------
    if cli::CliArgs::is_cli_mode() {
        use clap::Parser;
        // Initialize i18n so any deeply-nested t!() calls resolve cleanly
        i18n::init();
        let args = cli::CliArgs::parse();
        let code = cli::run(args);
        let _ = AppSettings::flush_pending_saves();
        std::process::exit(if code == std::process::ExitCode::SUCCESS {
            0
        } else {
            1
        });
    }

    // -- GUI mode -----------------------------------------------------

    #[cfg(target_os = "linux")]
    {
        // Backend policy:
        // - default on Wayland sessions: prefer wayland backend
        // - default elsewhere: let winit auto-select backend
        // - optional override: PAINTFE_FORCE_WAYLAND=1 forces Wayland
        // - optional override: PAINTFE_FORCE_X11=1 forces X11
        // This keeps native DnD behavior on Wayland while preserving explicit
        // fallbacks for setups that work better with X11.
        let wayland_session = std::env::var_os("WAYLAND_DISPLAY").is_some()
            || std::env::var("XDG_SESSION_TYPE")
                .map(|v| v.eq_ignore_ascii_case("wayland"))
                .unwrap_or(false);
        let has_x11 = std::env::var_os("DISPLAY").is_some();
        let force_wayland = std::env::var("PAINTFE_FORCE_WAYLAND")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let force_x11 = std::env::var("PAINTFE_FORCE_X11")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);

        if std::env::var_os("WINIT_UNIX_BACKEND").is_none() {
            if force_wayland {
                unsafe { std::env::set_var("WINIT_UNIX_BACKEND", "wayland") };
            } else if force_x11 && has_x11 {
                unsafe { std::env::set_var("WINIT_UNIX_BACKEND", "x11") };
            } else if wayland_session {
                unsafe { std::env::set_var("WINIT_UNIX_BACKEND", "wayland") };
            }
        }
    }

    // Suppress GLib/GIO/DBus debug noise that appears in some Wayland/Fedora setups.
    // Only set these if the user hasn't overridden them.
    #[cfg(target_os = "linux")]
    {
        if std::env::var_os("G_MESSAGES_DEBUG").is_none() {
            unsafe { std::env::set_var("G_MESSAGES_DEBUG", "") };
        }
        if std::env::var_os("G_DEBUG").is_none() {
            unsafe { std::env::set_var("G_DEBUG", "") };
        }
    }

    let profile = paintfe::ui::perf::StartupSpan::new("storage");
    paintfe::services::storage::initialize();
    drop(profile);

    // Initialize session log (overwrites previous session log)
    logger::init();

    // Initialize the internationalization system
    let profile = paintfe::ui::perf::StartupSpan::new("i18n");
    i18n::init();
    drop(profile);

    // Collect positional file arguments (e.g. `paintfe.exe photo.png` from
    // right-click → "Open with" on Windows, or drag-onto-exe).
    let profile = paintfe::ui::perf::StartupSpan::new("ipc");
    if let Err(error) = paintfe::services::graphics::wait_for_recovery_parent() {
        logger::write("ERROR", &error);
        return Ok(());
    }
    let startup_files = ipc::collect_startup_files();

    // Single-instance: if another PaintFE GUI is already running, send the
    // file paths to it via named pipe and exit this process.
    if ipc::try_send_to_existing(&startup_files) {
        ipc::focus_existing_window();
        std::process::exit(0);
    }

    // We are the first instance — start the IPC listener so future
    // invocations can send their file paths to us.
    let ipc_receiver = ipc::start_listener();
    drop(profile);

    // Load application icon (window title bar, taskbar, Alt+Tab)
    let profile = paintfe::ui::perf::StartupSpan::new("window_icon_and_settings");
    let icon = load_app_icon();

    let startup_settings = AppSettings::load();
    drop(profile);

    // Define the native window options
    let preferred_backend = paintfe::services::graphics::preferred_backends();
    let options = eframe::NativeOptions {
        viewport: {
            let mut vp = egui::ViewportBuilder::default()
                .with_inner_size([
                    startup_settings.persist_window_width.max(640.0),
                    startup_settings.persist_window_height.max(480.0),
                ])
                .with_title("PaintFE")
                .with_app_id("io.github.paintfe.PaintFE");
            if let Some((x, y)) = startup_settings.persist_window_pos
                && let Some(safe_pos) = sanitize_window_position((x, y))
            {
                vp = vp.with_position(safe_pos);
            }
            if let Some(icon_data) = icon {
                vp = vp.with_icon(std::sync::Arc::new(icon_data));
            }
            vp
        },
        event_loop_builder: Some(Box::new(configure_event_loop)),
        wgpu_options: {
            let mut wgpu_options = eframe::egui_wgpu::WgpuConfiguration::default();
            if let Some(backends) = preferred_backend
                && let eframe::egui_wgpu::WgpuSetup::CreateNew(setup) = &mut wgpu_options.wgpu_setup
            {
                setup.instance_descriptor.backends = backends;
                // Opt-in fault injection exercises recovery without a broken driver.
                if paintfe::ui::perf::requested_scenario().is_some()
                    && std::env::var_os("PAINTFE_PROFILE_FAIL_PREFERRED").is_some()
                {
                    setup.native_adapter_selector = Some(std::sync::Arc::new(|_, _| {
                        Err("Profiling: simulated preferred backend failure".into())
                    }));
                }
            }
            if startup_settings.low_latency_present {
                // AutoNoVsync picks the first supported mode out of Immediate →
                // Mailbox → Fifo (graceful fallback; plain Mailbox would crash on
                // unsupported systems). Combined with a single queued frame this
                // removes 1-3 frames of input latency versus the AutoVsync (Fifo)
                // + 2-frame-queue defaults.
                wgpu_options.surface.present_mode = wgpu::PresentMode::AutoNoVsync;
                wgpu_options.surface.desired_maximum_frame_latency = Some(1);
            }
            wgpu_options
        },
        ..Default::default()
    };

    // Run the application
    drop(startup_profile);
    let renderer_profile = paintfe::ui::perf::StartupSpan::new("event_loop_to_app_creation");
    let app_created = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let creation_flag = app_created.clone();
    let result = eframe::run_native(
        "PaintFE",
        options,
        Box::new(move |cc| {
            drop(renderer_profile);
            creation_flag.store(true, std::sync::atomic::Ordering::Release);
            if let Some(state) = &cc.wgpu_render_state {
                paintfe::services::graphics::record_working_backend(
                    state.adapter.get_info().backend,
                );
            }
            #[cfg(target_os = "windows")]
            set_windows_app_icons(cc);
            Ok(Box::new(PaintFEApp::new(cc, startup_files, ipc_receiver)))
        }),
    );
    if result.is_err()
        && preferred_backend.is_some()
        && !app_created.load(std::sync::atomic::Ordering::Acquire)
    {
        logger::write(
            "WARN",
            "Preferred graphics backend failed; restarting once with automatic selection",
        );
        if paintfe::services::graphics::restart_with_automatic_backend().is_ok() {
            return Ok(());
        }
    }
    result
}

/// Entry point — on Windows, wraps main_inner with SEH protection to catch
/// assertion failures from third-party injected DLLs (e.g. AlphaRing overlay).
fn main() -> Result<(), eframe::Error> {
    #[cfg(target_os = "windows")]
    {
        run_with_seh_protection(main_inner)
    }
    #[cfg(not(target_os = "windows"))]
    {
        main_inner()
    }
}

/// Configure the winit event loop to suppress Windows error dings.
///
/// Two classes of messages must be suppressed:
///
/// 1. WM_CHAR control characters (< 0x20) — generated by TranslateMessage for
///    Ctrl+letter combos (e.g. Ctrl+C → WM_CHAR(0x03)).  They are never used
///    by egui (is_printable_char returns false for ascii-control), and if they
///    reach DefWindowProc they can trigger MessageBeep(0).
///
/// 2. WM_SYSCHAR — generated by TranslateMessage whenever a key is pressed
///    while Alt is held.  winit 0.28 unconditionally calls DefWindowProcW for
///    every WM_SYSCHAR it receives.  Because this app has no Win32 native menu
///    bar (menus are drawn by egui), DefWindowProc finds no matching menu
///    mnemonic for any Alt+letter and plays MessageBeep(0) — the Windows error
///    sound.  Suppressing WM_SYSCHAR here prevents the beep while Alt+F4 /
///    Alt+Space still work (those are handled through WM_SYSKEYDOWN →
///    DefWindowProc → WM_SYSCOMMAND, not through WM_SYSCHAR).
///
/// Important: do NOT manually dispatch WM_KEYDOWN/WM_SYSKEYDOWN from this hook.
/// That interferes with winit's normal message routing and can cause Windows to
/// play the default error sound even during ordinary typing in text fields.
#[cfg(target_os = "windows")]
fn configure_event_loop(builder: &mut eframe::EventLoopBuilder<eframe::UserEvent>) {
    use winit::platform::windows::EventLoopBuilderExtWindows;

    builder.with_msg_hook(|msg_ptr| {
        #[repr(C)]
        #[allow(non_snake_case, clippy::upper_case_acronyms)]
        struct MSG {
            hwnd: isize,
            message: u32,
            wParam: usize,
            lParam: isize,
            time: u32,
            pt_x: i32,
            pt_y: i32,
        }

        const WM_CHAR: u32 = 0x0102;
        const WM_SYSCHAR: u32 = 0x0106;
        let msg = unsafe { &*(msg_ptr as *const MSG) };

        // Preserve native key/text ordering before winit defers control-key
        // presses to the WM_CHAR messages suppressed below.
        paintfe::windows_key_probe::observe_native_message(msg.message, msg.wParam, msg.lParam);

        // Key events handle navigation; control-character text must never reach egui.
        if msg.message == WM_CHAR && msg.wParam < 0x20 {
            return true;
        }

        // Without a native menu, WM_SYSCHAR can trigger MessageBeep(0).
        if msg.message == WM_SYSCHAR {
            return true;
        }

        false
    });
}

#[cfg(not(target_os = "windows"))]
fn configure_event_loop(_builder: &mut eframe::EventLoopBuilder<eframe::UserEvent>) {
    // No-op on non-Windows platforms
}

/// Decode the embedded PNG icon into raw RGBA for the egui viewport.
fn load_app_icon() -> Option<egui::viewport::IconData> {
    let png_bytes = include_bytes!("../assets/icons/app_icon_128.png");
    let img = image::load_from_memory(png_bytes).ok()?.into_rgba8();
    let (w, h) = img.dimensions();
    Some(egui::viewport::IconData {
        rgba: img.into_raw(),
        width: w,
        height: h,
    })
}

/// Select authored small and large ICO frames independently of winit's PNG icon.
#[cfg(target_os = "windows")]
fn set_windows_app_icons(cc: &eframe::CreationContext<'_>) {
    use winapi::um::winuser::{
        IMAGE_ICON, LR_SHARED, LoadImageW, MAKEINTRESOURCEW, SendMessageW, WM_SETICON,
    };
    use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};

    let Ok(handle) = cc.window_handle() else {
        return;
    };
    let RawWindowHandle::Win32(window) = handle.as_raw() else {
        return;
    };
    let Some(instance) = window.hinstance else {
        return;
    };
    // Resource 1 is embedded by winresource. Shared resource handles live for
    // the process lifetime and must not be destroyed by the application.
    for (slot, size) in [(0, 32), (1, 128)] {
        unsafe {
            let icon = LoadImageW(
                instance.get() as _,
                MAKEINTRESOURCEW(1),
                IMAGE_ICON,
                size,
                size,
                LR_SHARED,
            );
            if !icon.is_null() {
                SendMessageW(window.hwnd.get() as _, WM_SETICON, slot, icon as isize);
            }
        }
    }
}

// dummy comment
