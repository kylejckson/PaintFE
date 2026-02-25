// GUI-subsystem binary: no console window is ever allocated by Windows.
// • GUI mode: nothing extra needed — no console to free.
// • CLI mode (--input/-i flag present): AttachConsole(ATTACH_PARENT_PROCESS) attaches to
//   the launching terminal, then we reopen CONOUT$/CONIN$ so Rust's println!/eprintln!
//   route through the correct handles (necessary when SUBSYSTEM:WINDOWS is set).
#![windows_subsystem = "windows"]
#![allow(dead_code)] // API surface kept for future features, scripting, and GPU pipelines
#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]
#![allow(clippy::large_enum_variant)]
#![allow(clippy::unnecessary_unwrap)]

#[macro_use]
mod i18n;
mod app;
mod assets;
mod canvas;
mod cli;
mod components;
mod gpu;
mod io;
pub mod logger;
mod ops;
mod project;
mod theme;

use app::PaintFEApp;
use eframe::egui;

fn main() -> Result<(), eframe::Error> {
    // -- Windows console management ------------------------------------
    // The binary is SUBSYSTEM:WINDOWS so Windows never allocates a console.
    // In CLI mode we attach to the parent terminal and reconnect stdio handles
    // so that Rust's println!/eprintln! write to the correct console buffers.
    #[cfg(target_os = "windows")]
    if cli::CliArgs::is_cli_mode() {
        unsafe extern "system" {
            fn AttachConsole(dwProcessId: u32) -> i32;
            fn SetStdHandle(nStdHandle: u32, hHandle: isize) -> i32;
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
            AttachConsole(ATTACH_PARENT_PROCESS);
            // Reopen CONOUT$ / CONIN$ so the process's std handles are valid.
            let conout: Vec<u16> = "CONOUT$\0".encode_utf16().collect();
            let conin: Vec<u16> = "CONIN$\0".encode_utf16().collect();
            let hout = CreateFileW(
                conout.as_ptr(),
                GENERIC_WRITE,
                FILE_SHARE_READ_WRITE,
                std::ptr::null(),
                OPEN_EXISTING,
                0,
                0,
            );
            if hout != INVALID_HANDLE_VALUE {
                SetStdHandle(STD_OUTPUT_HANDLE, hout);
                SetStdHandle(STD_ERROR_HANDLE, hout);
            }
            let hin = CreateFileW(
                conin.as_ptr(),
                GENERIC_READ,
                FILE_SHARE_READ_WRITE,
                std::ptr::null(),
                OPEN_EXISTING,
                0,
                0,
            );
            if hin != INVALID_HANDLE_VALUE {
                SetStdHandle(STD_INPUT_HANDLE, hin);
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
        std::process::exit(if code == std::process::ExitCode::SUCCESS {
            0
        } else {
            1
        });
    }

    // -- GUI mode -----------------------------------------------------

    // Initialize session log (overwrites previous session log)
    logger::init();

    // Initialize the internationalization system
    i18n::init();

    // Load application icon (window title bar, taskbar, Alt+Tab)
    let icon = load_app_icon();

    // Define the native window options
    let options = eframe::NativeOptions {
        viewport: {
            let mut vp = egui::ViewportBuilder::default()
                .with_inner_size([1280.0, 720.0])
                .with_maximized(true)
                .with_title("PaintFE");
            if let Some(icon_data) = icon {
                vp = vp.with_icon(std::sync::Arc::new(icon_data));
            }
            vp
        },
        ..Default::default()
    };

    // Run the application
    eframe::run_native(
        "PaintFE",
        options,
        Box::new(|cc| Box::new(PaintFEApp::new(cc))),
    )
}

/// Decode the embedded PNG icon into raw RGBA for the egui viewport.
fn load_app_icon() -> Option<egui::viewport::IconData> {
    let png_bytes = include_bytes!("../assets/icons/app_icon.png");
    let img = image::load_from_memory(png_bytes).ok()?.into_rgba8();
    let (w, h) = img.dimensions();
    Some(egui::viewport::IconData {
        rgba: img.into_raw(),
        width: w,
        height: h,
    })
}
