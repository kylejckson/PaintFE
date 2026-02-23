// ============================================================================
// PRINT — save composite to temp PNG and open with OS default viewer
// ============================================================================

use std::path::PathBuf;
use image::RgbaImage;

/// "Print" by saving the composite to a temp file and opening it with the OS.
pub fn print_image(composite: &RgbaImage) -> Result<(), String> {
    let dir = std::env::temp_dir();
    let path = dir.join("paintfe_print.png");
    composite
        .save(&path)
        .map_err(|e| format!("Failed to save print image: {}", e))?;

    open_with_os(&path)
}

#[cfg(target_os = "windows")]
fn open_with_os(path: &PathBuf) -> Result<(), String> {
    use std::ffi::OsStr;
    use std::os::windows::ffi::OsStrExt;
    use winapi::um::shellapi::ShellExecuteW;
    use winapi::um::winuser::SW_SHOWNORMAL;

    // Convert strings to Windows wide (UTF-16) null-terminated
    fn to_wide(s: &str) -> Vec<u16> {
        OsStr::new(s).encode_wide().chain(std::iter::once(0)).collect()
    }

    let verb = to_wide("print");
    let file = to_wide(&path.to_string_lossy());

    let result = unsafe {
        ShellExecuteW(
            std::ptr::null_mut(), // hwnd
            verb.as_ptr(),
            file.as_ptr(),
            std::ptr::null(),     // parameters
            std::ptr::null(),     // directory
            SW_SHOWNORMAL,
        )
    };

    // ShellExecuteW returns > 32 on success
    if result as usize > 32 {
        Ok(())
    } else {
        Err(format!("ShellExecuteW print failed (code {})", result as usize))
    }
}

#[cfg(target_os = "macos")]
fn open_with_os(path: &PathBuf) -> Result<(), String> {
    std::process::Command::new("open")
        .arg(path)
        .spawn()
        .map_err(|e| format!("Failed to open image: {}", e))?;
    Ok(())
}

#[cfg(target_os = "linux")]
fn open_with_os(path: &PathBuf) -> Result<(), String> {
    std::process::Command::new("xdg-open")
        .arg(path)
        .spawn()
        .map_err(|e| format!("Failed to open image: {}", e))?;
    Ok(())
}
