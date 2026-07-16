// ============================================================================
// PRINT — save composite to temp PNG and open with OS default viewer
// ============================================================================

use image::RgbaImage;
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;

/// "Print" on web: encode the composite as a PNG data URL, open it in a new
/// tab sized to just the image, and trigger the browser's own print dialog
/// once it has loaded. There's no OS temp dir or default viewer to shell out
/// to like the native path does, so `window.print()` is the closest
/// equivalent — it still goes through the user's real printer picker.
#[cfg(target_arch = "wasm32")]
pub fn print_image(composite: &RgbaImage) -> Result<(), String> {
    use base64::Engine;
    use image::ImageEncoder;

    let mut png_bytes: Vec<u8> = Vec::new();
    image::codecs::png::PngEncoder::new(&mut png_bytes)
        .write_image(
            composite.as_raw(),
            composite.width(),
            composite.height(),
            image::ExtendedColorType::Rgba8,
        )
        .map_err(|e| format!("Failed to encode print image: {}", e))?;
    let b64 = base64::engine::general_purpose::STANDARD.encode(&png_bytes);

    let window = web_sys::window().ok_or("no window")?;
    let print_window = window
        .open_with_url_and_target("about:blank", "_blank")
        .map_err(|_| "Failed to open print window".to_string())?
        .ok_or("Browser blocked the print window (pop-up blocker?)".to_string())?;
    let document = print_window
        .document()
        .ok_or("print window has no document")?;
    document.set_title("PaintFE — Print");
    let body = document.body().ok_or("print window has no body")?;
    let html = format!(
        r#"<style>
            html, body {{ margin: 0; padding: 0; background: #fff; }}
            img {{ display: block; max-width: 100%; height: auto; margin: 0 auto; }}
            @media print {{ img {{ max-width: 100%; }} }}
        </style>
        <img src="data:image/png;base64,{b64}" onload="window.focus(); window.print();" />"#
    );
    body.set_inner_html(&html);
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
pub fn print_image(composite: &RgbaImage) -> Result<(), String> {
    let dir = std::env::temp_dir();
    let path = dir.join("paintfe_print.png");
    composite
        .save(&path)
        .map_err(|e| format!("Failed to save print image: {}", e))?;

    open_with_os(&path)
}

#[cfg(target_os = "windows")]
fn open_with_os(path: &Path) -> Result<(), String> {
    use std::ffi::OsStr;
    use std::os::windows::ffi::OsStrExt;
    use winapi::um::shellapi::ShellExecuteW;
    use winapi::um::winuser::SW_SHOWNORMAL;

    // Convert strings to Windows wide (UTF-16) null-terminated
    fn to_wide(s: &str) -> Vec<u16> {
        OsStr::new(s)
            .encode_wide()
            .chain(std::iter::once(0))
            .collect()
    }

    let verb = to_wide("print");
    let file = to_wide(&path.to_string_lossy());

    let result = unsafe {
        ShellExecuteW(
            std::ptr::null_mut(), // hwnd
            verb.as_ptr(),
            file.as_ptr(),
            std::ptr::null(), // parameters
            std::ptr::null(), // directory
            SW_SHOWNORMAL,
        )
    };

    // ShellExecuteW returns > 32 on success
    if result as usize > 32 {
        Ok(())
    } else {
        Err(format!(
            "ShellExecuteW print failed (code {})",
            result as usize
        ))
    }
}

#[cfg(target_os = "macos")]
fn open_with_os(path: &Path) -> Result<(), String> {
    std::process::Command::new("open")
        .arg(path)
        .spawn()
        .map_err(|e| format!("Failed to open image: {}", e))?;
    Ok(())
}

#[cfg(target_os = "linux")]
fn open_with_os(path: &Path) -> Result<(), String> {
    std::process::Command::new("xdg-open")
        .arg(path)
        .spawn()
        .map_err(|e| format!("Failed to open image: {}", e))?;
    Ok(())
}
