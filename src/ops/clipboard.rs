// ============================================================================
// CLIPBOARD OPERATIONS — cut, copy, paste with manipulatable paste overlay
// ============================================================================

use crate::canvas::{CanvasState, TiledImage};
use crate::log_info;
use crate::ops::transform::Interpolation;
use crate::par_compat::*;
use eframe::egui;
use egui::{Color32, Pos2, Rect, Stroke, Vec2};
use image::{GrayImage, Luma, Rgba, RgbaImage, imageops};
use std::sync::{Arc, Mutex};

#[cfg(target_os = "linux")]
use image::ImageFormat;
#[cfg(target_os = "linux")]
use std::io::{Cursor, Write};
#[cfg(target_os = "linux")]
use std::path::PathBuf;

// ---------------------------------------------------------------------------
//  Internal clipboard (application-level, supports transparency)
// ---------------------------------------------------------------------------

/// In-app clipboard storing an RGBA image with full transparency support.
static APP_CLIPBOARD: Mutex<Option<ClipboardPayload>> = Mutex::new(None);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClipboardImageSource {
    Internal,
    External,
}

#[derive(Clone)]
struct ClipboardPayload {
    image: Arc<RgbaImage>,
    source: ClipboardImageSource,
    origin_center: Option<Pos2>,
    overwrite_transparent_pixels: bool,
    overwrite_mask: Option<GrayImage>,
    system_sequence: Option<u32>,
}

pub struct ClipboardImageForPaste {
    pub image: RgbaImage,
    pub source: ClipboardImageSource,
    pub origin_center: Option<Pos2>,
    pub overwrite_transparent_pixels: bool,
    pub overwrite_mask: Option<GrayImage>,
}

/// Store an image in the app clipboard.
fn set_clipboard_image(
    img: RgbaImage,
    origin_center: Option<Pos2>,
    overwrite_transparent_pixels: bool,
    overwrite_mask: Option<GrayImage>,
) {
    *APP_CLIPBOARD.lock().unwrap_or_else(|e| e.into_inner()) = Some(ClipboardPayload {
        image: Arc::new(img),
        source: ClipboardImageSource::Internal,
        origin_center,
        overwrite_transparent_pixels,
        overwrite_mask,
        system_sequence: clipboard_sequence(),
    });
}

fn copy_image_to_clipboard(
    img: RgbaImage,
    origin_center: Option<Pos2>,
    overwrite_transparent_pixels: bool,
    overwrite_mask: Option<GrayImage>,
) {
    set_clipboard_image(
        img,
        origin_center,
        overwrite_transparent_pixels,
        overwrite_mask,
    );
    #[cfg(not(target_arch = "wasm32"))]
    {
        let image = APP_CLIPBOARD
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            .map(|payload| Arc::clone(&payload.image));
        if let Some(image) = image {
            std::thread::spawn(move || {
                copy_to_system_clipboard(&image);
                let sequence = clipboard_sequence();
                let mut clipboard = APP_CLIPBOARD.lock().unwrap_or_else(|e| e.into_inner());
                if let Some(payload) = clipboard.as_mut()
                    && Arc::ptr_eq(&payload.image, &image)
                {
                    payload.system_sequence = sequence;
                }
            });
        }
    }
    #[cfg(target_arch = "wasm32")]
    if let Some(image) = APP_CLIPBOARD
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .map(|payload| Arc::clone(&payload.image))
    {
        copy_to_system_clipboard(&image);
    }
}

/// Retrieve a clone from the app clipboard.
fn get_clipboard_image() -> Option<RgbaImage> {
    APP_CLIPBOARD
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .map(|p| (*p.image).clone())
}

fn get_clipboard_payload() -> Option<ClipboardPayload> {
    APP_CLIPBOARD
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone()
}

fn images_match(a: &RgbaImage, b: &RgbaImage) -> bool {
    a.dimensions() == b.dimensions() && a.as_raw() == b.as_raw()
}

/// Unified paste source selection.
///
/// Priority:
/// 1) If system clipboard contains an image that byte-matches the internal clipboard image,
///    treat it as an in-app copy/cut and preserve origin metadata.
/// 2) If system clipboard contains a different image, treat it as external.
/// 3) Otherwise fall back to internal clipboard payload.
pub fn get_clipboard_image_for_paste() -> Option<ClipboardImageForPaste> {
    let internal = get_clipboard_payload();

    if let Some(payload) = internal.as_ref()
        && payload.system_sequence.is_some()
        && payload.system_sequence == clipboard_sequence()
    {
        return Some(ClipboardImageForPaste {
            image: (*payload.image).clone(),
            source: payload.source,
            origin_center: payload.origin_center,
            overwrite_transparent_pixels: payload.overwrite_transparent_pixels,
            overwrite_mask: payload.overwrite_mask.clone(),
        });
    }

    let system = get_from_system_clipboard();

    if let Some(system_img) = system {
        if let Some(internal_payload) = &internal
            && images_match(&system_img, &internal_payload.image)
        {
            return Some(ClipboardImageForPaste {
                image: (*internal_payload.image).clone(),
                source: internal_payload.source,
                origin_center: internal_payload.origin_center,
                overwrite_transparent_pixels: internal_payload.overwrite_transparent_pixels,
                overwrite_mask: internal_payload.overwrite_mask.clone(),
            });
        }
        return Some(ClipboardImageForPaste {
            image: system_img,
            source: ClipboardImageSource::External,
            origin_center: None,
            overwrite_transparent_pixels: true,
            overwrite_mask: None,
        });
    }

    internal.map(|payload| ClipboardImageForPaste {
        image: (*payload.image).clone(),
        source: payload.source,
        origin_center: payload.origin_center,
        overwrite_transparent_pixels: payload.overwrite_transparent_pixels,
        overwrite_mask: payload.overwrite_mask,
    })
}

#[cfg(target_os = "windows")]
fn clipboard_sequence() -> Option<u32> {
    Some(unsafe { winapi::um::winuser::GetClipboardSequenceNumber() })
}

#[cfg(not(target_os = "windows"))]
fn clipboard_sequence() -> Option<u32> {
    None
}

pub fn has_clipboard_image() -> bool {
    let has_internal = APP_CLIPBOARD
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .is_some();
    has_internal || get_from_system_clipboard().is_some()
}

/// Public accessor so app.rs can use the internal clipboard as a fallback.
pub fn get_clipboard_image_pub() -> Option<RgbaImage> {
    get_clipboard_image()
}

/// Get the dimensions of whatever image is available on the clipboard
/// (internal app clipboard first, then system clipboard).
/// Returns `Some((width, height))` without retaining the full pixel data.
pub fn get_clipboard_image_dimensions() -> Option<(u32, u32)> {
    // A13: Fast path — lock the mutex and read dimensions without cloning.
    if let Ok(guard) = APP_CLIPBOARD.lock()
        && let Some(ref payload) = *guard
    {
        return Some((payload.image.width(), payload.image.height()));
    }
    // Slow path: check system clipboard.
    if let Some(img) = get_from_system_clipboard() {
        return Some((img.width(), img.height()));
    }
    None
}

/// Get clipboard image dimensions from the in-app clipboard only.
/// Useful for UI paths where system clipboard probing could cause latency.
pub fn get_internal_clipboard_image_dimensions() -> Option<(u32, u32)> {
    if let Ok(guard) = APP_CLIPBOARD.lock()
        && let Some(ref payload) = *guard
    {
        return Some((payload.image.width(), payload.image.height()));
    }
    None
}

// ---------------------------------------------------------------------------
//  System clipboard helpers (OS-level copy/paste via arboard)
// ---------------------------------------------------------------------------

/// Write an RGBA image to the system clipboard via the browser's async
/// Clipboard API. Fire-and-forget: the write happens in the background and
/// failures are swallowed (e.g. unsupported browser, no permission, insecure
/// context) — PaintFE's internal copy/paste still works within the app
/// regardless.
#[cfg(target_arch = "wasm32")]
pub fn copy_to_system_clipboard(img: &RgbaImage) {
    use image::ImageEncoder;

    let mut png_bytes: Vec<u8> = Vec::new();
    if image::codecs::png::PngEncoder::new(&mut png_bytes)
        .write_image(
            img.as_raw(),
            img.width(),
            img.height(),
            image::ExtendedColorType::Rgba8,
        )
        .is_err()
    {
        return;
    }

    wasm_bindgen_futures::spawn_local(async move {
        let _ = web_clipboard::write_png_to_clipboard(png_bytes).await;
    });
}

#[cfg(not(target_arch = "wasm32"))]
pub fn copy_to_system_clipboard(img: &RgbaImage) {
    // arboard wants ImageData { width, height, bytes: Cow<[u8]> } in RGBA order.
    #[cfg(target_os = "linux")]
    {
        let mut copied = false;

        // Prefer wl-copy only when the active window backend is Wayland.
        if use_wayland_clipboard_backend() {
            copied = copy_to_wayland_clipboard(img);
        }

        if !copied && let Ok(mut clip) = arboard::Clipboard::new() {
            let data = arboard::ImageData {
                width: img.width() as usize,
                height: img.height() as usize,
                bytes: std::borrow::Cow::Borrowed(img.as_raw()),
            };
            let _ = clip.set_image(data).is_ok();
        }
    }

    #[cfg(not(target_os = "linux"))]
    {
        if let Ok(mut clip) = arboard::Clipboard::new() {
            let data = arboard::ImageData {
                width: img.width() as usize,
                height: img.height() as usize,
                bytes: std::borrow::Cow::Borrowed(img.as_raw()),
            };
            let _ = clip.set_image(data);
        }
    }
}

/// Try to read an image from the system clipboard. Returns None if nothing available.
/// Handles three cases:
///   1. Raw image data (e.g. Print Screen, copied from another image editor).
///   2. Text on clipboard that happens to be a valid image file path.
///   3. A file copied in Explorer (CF_HDROP file list) — Windows-specific.
///
/// Returns the most recent image captured by the browser's native `paste`
/// event (installed once at startup, see `web_clipboard::install_paste_listener`).
/// Reading the OS clipboard directly (`navigator.clipboard.read()`) requires
/// an async round-trip this function's synchronous signature can't express,
/// so instead we listen for the real Ctrl+V / browser paste event and decode
/// whatever image data it carries ahead of time.
#[cfg(target_arch = "wasm32")]
pub fn get_from_system_clipboard() -> Option<RgbaImage> {
    web_clipboard::take_pasted_image()
}

#[cfg(not(target_arch = "wasm32"))]
pub fn get_from_system_clipboard() -> Option<RgbaImage> {
    #[cfg(target_os = "linux")]
    {
        // On Wayland backend, prefer wl-paste to avoid reading stale X11 clipboard
        // contents when running under XWayland.
        if use_wayland_clipboard_backend()
            && let Some(img) = get_from_wayland_clipboard()
        {
            return Some(img);
        }
    }

    // 1. Try raw image data via arboard.
    if let Ok(mut clip) = arboard::Clipboard::new()
        && let Ok(img_data) = clip.get_image()
        && let Some(img) = RgbaImage::from_raw(
            img_data.width as u32,
            img_data.height as u32,
            img_data.bytes.into_owned(),
        )
    {
        return Some(img);
    }

    #[cfg(target_os = "linux")]
    {
        // Fallback for Wayland backend sessions where wl-paste is unavailable or empty.
        if use_wayland_clipboard_backend()
            && let Some(img) = get_from_wayland_clipboard()
        {
            return Some(img);
        }
    }

    // 2. On Windows, try the CF_HDROP file list that Explorer puts on the
    //    clipboard when the user Ctrl+C's a file.
    #[cfg(target_os = "windows")]
    {
        if let Some(img) = read_image_from_clipboard_file_list() {
            return Some(img);
        }
    }

    // 3. Try plain-text clipboard content as a file path.
    if let Ok(mut clip) = arboard::Clipboard::new()
        && let Ok(text) = clip.get_text()
    {
        #[cfg(target_os = "linux")]
        {
            for path in parse_file_uri_or_path_list(&text) {
                if let Ok(dyn_img) = image::open(&path) {
                    return Some(dyn_img.to_rgba8());
                }
            }
        }

        #[cfg(not(target_os = "linux"))]
        {
            let path = std::path::Path::new(text.trim());
            if path.is_file()
                && let Ok(dyn_img) = image::open(path)
            {
                return Some(dyn_img.to_rgba8());
            }
        }
    }

    None
}

#[cfg(target_os = "linux")]
fn is_wayland_session() -> bool {
    std::env::var_os("WAYLAND_DISPLAY").is_some()
        || std::env::var("XDG_SESSION_TYPE")
            .map(|v| v.eq_ignore_ascii_case("wayland"))
            .unwrap_or(false)
}

#[cfg(target_os = "linux")]
fn use_wayland_clipboard_backend() -> bool {
    if !is_wayland_session() {
        return false;
    }
    !std::env::var("WINIT_UNIX_BACKEND")
        .map(|v| v.eq_ignore_ascii_case("x11"))
        .unwrap_or(false)
}

#[cfg(target_os = "linux")]
fn percent_decode_path_component(input: &str) -> String {
    let bytes = input.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            let h1 = bytes[i + 1];
            let h2 = bytes[i + 2];
            let hex = |c: u8| -> Option<u8> {
                match c {
                    b'0'..=b'9' => Some(c - b'0'),
                    b'a'..=b'f' => Some(c - b'a' + 10),
                    b'A'..=b'F' => Some(c - b'A' + 10),
                    _ => None,
                }
            };
            if let (Some(a), Some(b)) = (hex(h1), hex(h2)) {
                out.push((a << 4) | b);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).to_string()
}

#[cfg(target_os = "linux")]
fn parse_file_uri_or_path_list(text: &str) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    for raw_line in text.split(['\n', '\0']) {
        let line = raw_line.trim().trim_end_matches('\r');
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line.eq_ignore_ascii_case("copy") || line.eq_ignore_ascii_case("cut") {
            continue;
        }

        if let Some(mut rest) = line.strip_prefix("file://") {
            if let Some(after_localhost) = rest.strip_prefix("localhost/") {
                rest = after_localhost;
            } else if let Some(idx) = rest.find('/') {
                rest = &rest[idx + 1..];
            }

            let decoded = percent_decode_path_component(rest);
            let candidate = PathBuf::from(format!("/{}", decoded));
            if candidate.is_file() {
                paths.push(candidate);
            }
            continue;
        }

        let direct = PathBuf::from(line);
        if direct.is_file() {
            paths.push(direct);
        }
    }
    paths
}

#[cfg(target_os = "linux")]
fn copy_to_wayland_clipboard(img: &RgbaImage) -> bool {
    if !is_wayland_session() {
        return false;
    }

    let mut cursor = Cursor::new(Vec::<u8>::new());
    if image::DynamicImage::ImageRgba8(img.clone())
        .write_to(&mut cursor, ImageFormat::Png)
        .is_err()
    {
        return false;
    }

    let bytes = cursor.into_inner();
    let mut child = match std::process::Command::new("wl-copy")
        .arg("--type")
        .arg("image/png")
        .stdin(std::process::Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(_) => return false,
    };

    if let Some(stdin) = child.stdin.as_mut() {
        if stdin.write_all(&bytes).is_err() {
            let _ = child.kill();
            return false;
        }
    } else {
        let _ = child.kill();
        return false;
    }

    child.wait().map(|s| s.success()).unwrap_or(false)
}

#[cfg(target_os = "linux")]
fn get_from_wayland_clipboard() -> Option<RgbaImage> {
    if !is_wayland_session() {
        return None;
    }

    let list = std::process::Command::new("wl-paste")
        .arg("--list-types")
        .output()
        .ok()?;
    if !list.status.success() {
        return None;
    }

    let mime_list = String::from_utf8_lossy(&list.stdout);
    let preferred = [
        "image/png",
        "image/webp",
        "image/jpeg",
        "image/bmp",
        "image/tiff",
    ];
    let chosen = preferred
        .iter()
        .find(|m| mime_list.lines().any(|line| line.trim() == **m))?;

    let output = std::process::Command::new("wl-paste")
        .arg("--no-newline")
        .arg("--type")
        .arg(chosen)
        .output()
        .ok()?;
    if !output.status.success() || output.stdout.is_empty() {
        return None;
    }

    image::load_from_memory(&output.stdout)
        .ok()
        .map(|img| img.to_rgba8())
}

/// On Windows, read the CF_HDROP file list from the clipboard and try to
/// open the first image-format file found.
#[cfg(target_os = "windows")]
fn read_image_from_clipboard_file_list() -> Option<RgbaImage> {
    use std::ptr;
    use winapi::um::shellapi::{DragQueryFileW, HDROP};
    use winapi::um::winuser::{CF_HDROP, CloseClipboard, GetClipboardData, OpenClipboard};

    unsafe {
        if OpenClipboard(ptr::null_mut()) == 0 {
            return None;
        }

        let handle = GetClipboardData(CF_HDROP);
        if handle.is_null() {
            CloseClipboard();
            return None;
        }

        let hdrop = handle as HDROP;
        let count = DragQueryFileW(hdrop, 0xFFFFFFFF, ptr::null_mut(), 0);

        let mut result: Option<RgbaImage> = None;

        for i in 0..count {
            let len = DragQueryFileW(hdrop, i, ptr::null_mut(), 0);
            if len == 0 {
                continue;
            }
            let mut buf: Vec<u16> = vec![0u16; (len + 1) as usize];
            DragQueryFileW(hdrop, i, buf.as_mut_ptr(), len + 1);
            let path_str = String::from_utf16_lossy(&buf[..len as usize]);
            let path = std::path::PathBuf::from(&path_str);

            let ext = path
                .extension()
                .map(|e| e.to_string_lossy().to_lowercase())
                .unwrap_or_default();
            let is_image = matches!(
                ext.as_str(),
                "png" | "jpg" | "jpeg" | "bmp" | "gif" | "webp" | "tiff" | "tga"
            );

            if is_image && let Ok(dyn_img) = image::open(&path) {
                result = Some(dyn_img.to_rgba8());
                break;
            }
        }

        CloseClipboard();
        result
    }
}

// ---------------------------------------------------------------------------
//  Cut / Copy
// ---------------------------------------------------------------------------

/// Copy selected pixels from the active layer into the clipboard.
/// Returns true if anything was copied.
pub fn copy_selection(state: &CanvasState, transparent_cutout: bool) -> bool {
    log_info!(
        "Clipboard: copy selection (transparent_cutout={})",
        transparent_cutout
    );
    let idx = state.active_layer_index;
    if idx >= state.layers.len() {
        return false;
    }

    let layer = &state.layers[idx];
    if state.selection_all {
        let clip = layer.pixels.to_rgba_image();
        copy_image_to_clipboard(
            clip,
            Some(Pos2::new(
                state.width as f32 * 0.5,
                state.height as f32 * 0.5,
            )),
            transparent_cutout,
            None,
        );
        return true;
    }

    let mask = match &state.selection_mask {
        Some(m) => m,
        None => return false,
    };
    let (mw, mh) = (mask.width(), mask.height());
    let mask_raw = mask.as_raw();

    // Find bounding box of selection.
    let mut min_x = mw;
    let mut min_y = mh;
    let mut max_x = 0u32;
    let mut max_y = 0u32;
    for y in 0..mh {
        let row = y as usize * mw as usize;
        for x in 0..mw {
            if mask_raw[row + x as usize] > 0 {
                min_x = min_x.min(x);
                min_y = min_y.min(y);
                max_x = max_x.max(x);
                max_y = max_y.max(y);
            }
        }
    }
    if min_x > max_x {
        return false;
    }

    let w = max_x - min_x + 1;
    let h = max_y - min_y + 1;
    let mut clip = RgbaImage::new(w, h);
    let mut overwrite_mask = GrayImage::new(w, h);

    for y in min_y..=max_y {
        let mask_row = y as usize * mw as usize;
        for x in min_x..=max_x {
            if mask_raw[mask_row + x as usize] > 0 {
                let px = layer.pixels.get_pixel(x, y);
                clip.put_pixel(x - min_x, y - min_y, *px);
                overwrite_mask.put_pixel(x - min_x, y - min_y, Luma([255]));
            } else if !transparent_cutout {
                // Preserve the selection silhouette in the copied image even when
                // transparent overwrite is disabled on paste.
                clip.put_pixel(x - min_x, y - min_y, Rgba([0, 0, 0, 0]));
            }
        }
    }

    let center_x = min_x as f32 + w as f32 / 2.0;
    let center_y = min_y as f32 + h as f32 / 2.0;
    copy_image_to_clipboard(
        clip,
        Some(Pos2::new(center_x, center_y)),
        transparent_cutout,
        Some(overwrite_mask),
    );
    true
}

pub fn copy_overlay(overlay: &PasteOverlay) -> bool {
    log_info!("Clipboard: copy overlay");
    let Some((img, origin_center)) = overlay.rasterize_for_clipboard() else {
        return false;
    };
    copy_image_to_clipboard(
        img,
        Some(origin_center),
        overlay.overwrite_transparent_pixels,
        overlay.overwrite_mask.clone(),
    );
    true
}

/// Cut = copy + delete selected pixels.
pub fn cut_selection(state: &mut CanvasState, transparent_cutout: bool) -> bool {
    log_info!(
        "Clipboard: cut selection (transparent_cutout={})",
        transparent_cutout
    );
    if !copy_selection(state, transparent_cutout) {
        return false;
    }
    state.delete_selected_pixels();
    state.clear_selection(); // Auto-deselect after cut
    state.mark_dirty(None);
    true
}

/// Extract selected pixels from the active layer into a PasteOverlay,
/// positioned at their original location. Blanks the source area.
/// If no selection exists, extracts the entire active layer contents.
/// Returns None only if the layer is empty / fully transparent.
pub fn extract_to_overlay(state: &mut CanvasState) -> Option<PasteOverlay> {
    let idx = state.active_layer_index;
    if idx >= state.layers.len() {
        return None;
    }

    let cw = state.width;
    let ch = state.height;

    if let Some(mask) = &state.selection_mask {
        // -- With selection: extract only masked pixels --
        let (mw, mh) = (mask.width(), mask.height());
        let mask_raw = mask.as_raw();

        // Bounding box of selection.
        let mut min_x = mw;
        let mut min_y = mh;
        let mut max_x = 0u32;
        let mut max_y = 0u32;
        for y in 0..mh {
            let row = y as usize * mw as usize;
            for x in 0..mw {
                if mask_raw[row + x as usize] > 0 {
                    min_x = min_x.min(x);
                    min_y = min_y.min(y);
                    max_x = max_x.max(x);
                    max_y = max_y.max(y);
                }
            }
        }
        if min_x > max_x {
            return None;
        }

        let w = max_x - min_x + 1;
        let h = max_y - min_y + 1;

        let layer = &state.layers[idx];
        let mut clip = RgbaImage::new(w, h);
        let mut overwrite_mask = GrayImage::new(w, h);
        for y in min_y..=max_y {
            let mask_row = y as usize * mw as usize;
            for x in min_x..=max_x {
                if mask_raw[mask_row + x as usize] > 0 {
                    clip.put_pixel(x - min_x, y - min_y, *layer.pixels.get_pixel(x, y));
                    overwrite_mask.put_pixel(x - min_x, y - min_y, Luma([255]));
                }
            }
        }

        // Blank the area on the layer.
        state.delete_selected_pixels();
        state.clear_selection();

        // Create overlay centered on the bounding box.
        let center_x = min_x as f32 + w as f32 / 2.0;
        let center_y = min_y as f32 + h as f32 / 2.0;
        let mut overlay = PasteOverlay::new(clip, cw, ch);
        overlay.center = Pos2::new(center_x, center_y);
        overlay.overwrite_transparent_pixels = false;
        overlay.overwrite_mask = Some(overwrite_mask);
        Some(overlay)
    } else {
        // -- No selection: extract entire active layer --
        let layer = &state.layers[idx];
        let img = layer.pixels.to_rgba_image();
        // Check if there's any content.
        let has_content = img.pixels().any(|p| p[3] > 0);
        if !has_content {
            return None;
        }

        // Blank the layer.
        let blank = TiledImage::new(cw, ch);
        state.layers[idx].pixels = blank;
        state.mark_dirty(None);

        let mut overlay = PasteOverlay::new(img, cw, ch);
        overlay.center = Pos2::new(cw as f32 / 2.0, ch as f32 / 2.0);
        Some(overlay)
    }
}

// ---------------------------------------------------------------------------
//  Paste Overlay — manipulatable floating image
// ---------------------------------------------------------------------------

/// Represents the pasted image floating above the canvas, being positioned /
/// rotated / scaled before committing.
pub struct PasteOverlay {
    /// The original pasted image (un-transformed).
    pub source: RgbaImage,
    /// Position of the center of the image on the canvas (canvas coords).
    pub center: Pos2,
    /// Rotation in radians.
    pub rotation: f32,
    /// Scale factor (1.0 = original size).
    pub scale_x: f32,
    pub scale_y: f32,
    /// Anchor point offset from center (canvas coords, relative to source center).
    pub anchor_offset: Vec2,
    /// Interpolation filter to use when committing.
    pub interpolation: Interpolation,
    /// Anti-aliasing toggle for rotation pass.
    pub anti_aliasing: bool,
    /// When true, transparent source pixels overwrite destination pixels.
    pub overwrite_transparent_pixels: bool,
    /// Optional mask limiting transparent overwrite to the copied selection shape.
    pub overwrite_mask: Option<GrayImage>,

    // --- Interaction state ---
    /// Which handle is being dragged, if any.
    pub active_handle: Option<HandleKind>,
    /// Mouse position at drag start (screen coords).
    pub drag_start_mouse: Option<Pos2>,
    /// State at drag start (for relative computation).
    pub drag_start_center: Pos2,
    pub drag_start_scale_x: f32,
    pub drag_start_scale_y: f32,
    pub drag_start_rotation: f32,
    pub drag_start_anchor: Vec2,
    /// Whether shift is held (lock aspect ratio).
    pub shift_held: bool,
    pub pending_transform_checkpoint: Option<(PasteOverlayTransform, PasteOverlayTransform)>,

    // --- Preview cache (avoids re-rendering every frame) ---
    /// Cached pre-scaled source image.
    cached_scaled: Option<(RgbaImage, u32, u32)>, // (img, scaled_w, scaled_h)
    /// Cached preview TiledImage + the transform state used to produce it.
    cached_preview: Option<(TiledImage, Pos2, f32, f32, f32, Vec2, u32, u32)>,
    // (tiled, center, scale_x, scale_y, rotation, anchor_offset, cw, ch)

    // --- GPU texture cache ---
    /// Cached GPU texture of the source image for GPU-accelerated rendering.
    /// Uploaded once; scale, rotation, and translation are mesh transforms.
    pub gpu_texture: Option<egui::TextureHandle>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum HandleKind {
    Move,
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
    Top,
    Bottom,
    Left,
    Right,
    Rotate,
    Anchor,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PasteOverlayTransform {
    pub center: Pos2,
    pub rotation: f32,
    pub scale_x: f32,
    pub scale_y: f32,
    pub anchor_offset: Vec2,
}

impl PasteOverlay {
    /// Create a new overlay from a clipboard image, centered on the canvas.
    pub fn new(source: RgbaImage, canvas_w: u32, canvas_h: u32) -> Self {
        Self {
            center: Pos2::new(canvas_w as f32 / 2.0, canvas_h as f32 / 2.0),
            scale_x: 1.0,
            scale_y: 1.0,
            rotation: 0.0,
            anchor_offset: Vec2::ZERO,
            interpolation: Interpolation::Bilinear,
            anti_aliasing: true,
            overwrite_transparent_pixels: false,
            overwrite_mask: None,
            active_handle: None,
            drag_start_mouse: None,
            drag_start_center: Pos2::ZERO,
            drag_start_scale_x: 1.0,
            drag_start_scale_y: 1.0,
            drag_start_rotation: 0.0,
            drag_start_anchor: Vec2::ZERO,
            shift_held: false,
            pending_transform_checkpoint: None,
            cached_scaled: None,
            cached_preview: None,
            gpu_texture: None,
            source,
        }
    }

    pub fn transform(&self) -> PasteOverlayTransform {
        PasteOverlayTransform {
            center: self.center,
            rotation: self.rotation,
            scale_x: self.scale_x,
            scale_y: self.scale_y,
            anchor_offset: self.anchor_offset,
        }
    }

    pub fn set_transform(&mut self, t: PasteOverlayTransform) {
        self.center = t.center;
        self.rotation = t.rotation;
        self.scale_x = t.scale_x;
        self.scale_y = t.scale_y;
        self.anchor_offset = t.anchor_offset;
        self.invalidate_cache();
    }

    pub fn transformed_bounds(&self, cw: u32, ch: u32) -> Option<(u32, u32, u32, u32)> {
        if cw == 0 || ch == 0 {
            return None;
        }
        let corners = self.corners_canvas();
        let mut min_x = cw as f32;
        let mut min_y = ch as f32;
        let mut max_x = 0.0f32;
        let mut max_y = 0.0f32;
        for c in &corners {
            min_x = min_x.min(c.x);
            min_y = min_y.min(c.y);
            max_x = max_x.max(c.x);
            max_y = max_y.max(c.y);
        }
        let x0 = min_x.floor().max(0.0).min(cw as f32) as u32;
        let y0 = min_y.floor().max(0.0).min(ch as f32) as u32;
        let x1 = max_x.ceil().max(0.0).min(cw as f32) as u32;
        let y1 = max_y.ceil().max(0.0).min(ch as f32) as u32;
        (x1 > x0 && y1 > y0).then_some((x0, y0, x1, y1))
    }

    pub fn transformed_selection_mask(&self, cw: u32, ch: u32) -> image::GrayImage {
        let mut mask = image::GrayImage::from_pixel(cw, ch, image::Luma([0u8]));
        if cw == 0 || ch == 0 {
            return mask;
        }
        let Some((x0, y0, x1, y1)) = self.transformed_bounds(cw, ch) else {
            return mask;
        };

        for y in y0..y1 {
            for x in x0..x1 {
                mask.put_pixel(x, y, image::Luma([255u8]));
            }
        }
        mask
    }

    pub fn solid_bounds_selection_mask(&self, cw: u32, ch: u32) -> image::GrayImage {
        let mut mask = image::GrayImage::from_pixel(cw, ch, image::Luma([0u8]));
        if let Some((x0, y0, x1, y1)) = self.transformed_bounds(cw, ch) {
            for y in y0..y1 {
                for x in x0..x1 {
                    mask.put_pixel(x, y, image::Luma([255u8]));
                }
            }
        }
        mask
    }

    fn invalidate_cache(&mut self) {
        self.cached_scaled = None;
        self.cached_preview = None;
    }

    /// Start a paste from the app clipboard. Returns None if clipboard is empty.
    pub fn from_clipboard(canvas_w: u32, canvas_h: u32) -> Option<Self> {
        let payload = get_clipboard_image_for_paste()?;
        let mut overlay = Self::new(payload.image, canvas_w, canvas_h);
        overlay.overwrite_transparent_pixels = payload.overwrite_transparent_pixels;
        overlay.overwrite_mask = payload.overwrite_mask;
        Some(overlay)
    }

    /// Start a paste from an external image.
    pub fn from_image(img: RgbaImage, canvas_w: u32, canvas_h: u32) -> Self {
        Self::new(img, canvas_w, canvas_h)
    }

    /// Start a paste centered at a specific canvas position.
    pub fn from_image_at(img: RgbaImage, canvas_w: u32, canvas_h: u32, center: Pos2) -> Self {
        let mut overlay = Self::new(img, canvas_w, canvas_h);
        overlay.center = center;
        overlay
    }

    pub fn rasterize_for_clipboard(&self) -> Option<(RgbaImage, Pos2)> {
        let src_w = self.source.width() as f32;
        let src_h = self.source.height() as f32;
        let scaled_w = (src_w * self.scale_x).round().max(1.0) as u32;
        let scaled_h = (src_h * self.scale_y).round().max(1.0) as u32;
        let scaled = imageops::resize(
            &self.source,
            scaled_w,
            scaled_h,
            self.interpolation.to_filter(),
        );

        let corners = self.corners_canvas();
        let mut min_x = f32::MAX;
        let mut min_y = f32::MAX;
        let mut max_x = f32::MIN;
        let mut max_y = f32::MIN;
        for corner in &corners {
            min_x = min_x.min(corner.x);
            min_y = min_y.min(corner.y);
            max_x = max_x.max(corner.x);
            max_y = max_y.max(corner.y);
        }

        let col_start = min_x.floor() as i32;
        let row_start = min_y.floor() as i32;
        let col_end = max_x.ceil() as i32;
        let row_end = max_y.ceil() as i32;
        if col_end < col_start || row_end < row_start {
            return None;
        }

        let out_w = (col_end - col_start + 1) as u32;
        let out_h = (row_end - row_start + 1) as u32;
        let mut out = RgbaImage::new(out_w, out_h);

        let anchor = self.anchor_canvas();
        let cos = self.rotation.cos();
        let sin = self.rotation.sin();
        let origin_x = self.center.x - scaled_w as f32 / 2.0;
        let origin_y = self.center.y - scaled_h as f32 / 2.0;
        let use_aa = self.anti_aliasing;

        for out_y in 0..out_h {
            let py = row_start as f32 + out_y as f32 + 0.5;
            let ry = py - anchor.y;
            for out_x in 0..out_w {
                let px = col_start as f32 + out_x as f32 + 0.5;
                let rx = px - anchor.x;
                let ur_x = rx * cos + ry * sin + anchor.x;
                let ur_y = -rx * sin + ry * cos + anchor.y;

                let local_x = ur_x - origin_x;
                let local_y = ur_y - origin_y;
                if local_x < -0.5
                    || local_y < -0.5
                    || local_x >= scaled_w as f32 + 0.5
                    || local_y >= scaled_h as f32 + 0.5
                {
                    continue;
                }

                if !use_aa
                    && (local_x < 0.0
                        || local_y < 0.0
                        || local_x >= scaled_w as f32
                        || local_y >= scaled_h as f32)
                {
                    continue;
                }

                let src_px = if use_aa {
                    sample_bilinear(&scaled, local_x - 0.5, local_y - 0.5, scaled_w, scaled_h)
                } else {
                    let ix = (local_x as u32).min(scaled_w - 1);
                    let iy = (local_y as u32).min(scaled_h - 1);
                    *scaled.get_pixel(ix, iy)
                };
                if src_px[3] > 0 {
                    out.put_pixel(out_x, out_y, src_px);
                }
            }
        }

        let has_pixels = out.pixels().any(|px| px[3] > 0);
        if !has_pixels {
            return None;
        }

        let origin_center = Pos2::new(
            col_start as f32 + out_w as f32 / 2.0,
            row_start as f32 + out_h as f32 / 2.0,
        );
        Some((out, origin_center))
    }

    pub fn render_replacement_preview(&self, base_layer: &RgbaImage) -> RgbaImage {
        let mut out = base_layer.clone();
        self.apply_transformed_pixels_to_image(&mut out, self.overwrite_transparent_pixels);
        out
    }

    fn overwrite_mask_allows(
        &self,
        local_x: f32,
        local_y: f32,
        scaled_mask: Option<&GrayImage>,
    ) -> bool {
        let Some(mask) = scaled_mask else {
            return true;
        };
        if local_x < 0.0 || local_y < 0.0 {
            return false;
        }
        let ix = (local_x as u32).min(mask.width().saturating_sub(1));
        let iy = (local_y as u32).min(mask.height().saturating_sub(1));
        mask.get_pixel(ix, iy)[0] > 0
    }

    fn apply_transformed_pixels_to_image(&self, out: &mut RgbaImage, overwrite_transparent: bool) {
        let src_w = self.source.width() as f32;
        let src_h = self.source.height() as f32;
        let scaled_w = (src_w * self.scale_x).round().max(1.0) as u32;
        let scaled_h = (src_h * self.scale_y).round().max(1.0) as u32;
        let scaled = imageops::resize(
            &self.source,
            scaled_w,
            scaled_h,
            self.interpolation.to_filter(),
        );
        let scaled_mask = self
            .overwrite_mask
            .as_ref()
            .map(|mask| imageops::resize(mask, scaled_w, scaled_h, imageops::FilterType::Nearest));

        let cw = out.width();
        let ch = out.height();
        if cw == 0 || ch == 0 {
            return;
        }

        let corners = self.corners_canvas();
        let mut bb_min_x = cw as f32;
        let mut bb_min_y = ch as f32;
        let mut bb_max_x = 0.0f32;
        let mut bb_max_y = 0.0f32;
        for c in &corners {
            bb_min_x = bb_min_x.min(c.x);
            bb_min_y = bb_min_y.min(c.y);
            bb_max_x = bb_max_x.max(c.x);
            bb_max_y = bb_max_y.max(c.y);
        }
        let row_start = (bb_min_y.floor().max(0.0)) as u32;
        let row_end = (bb_max_y.ceil().min(ch as f32 - 1.0)) as u32;
        let col_start = (bb_min_x.floor().max(0.0)) as u32;
        let col_end = (bb_max_x.ceil().min(cw as f32 - 1.0)) as u32;

        let anchor = self.anchor_canvas();
        let cos = self.rotation.cos();
        let sin = self.rotation.sin();
        let origin_x = self.center.x - scaled_w as f32 / 2.0;
        let origin_y = self.center.y - scaled_h as f32 / 2.0;
        let use_aa = self.anti_aliasing;

        for dy in row_start..=row_end {
            let py = dy as f32 + 0.5;
            let ry = py - anchor.y;
            for dx in col_start..=col_end {
                let px = dx as f32 + 0.5;
                let rx = px - anchor.x;
                let ur_x = rx * cos + ry * sin + anchor.x;
                let ur_y = -rx * sin + ry * cos + anchor.y;

                let local_x = ur_x - origin_x;
                let local_y = ur_y - origin_y;

                if local_x < -0.5
                    || local_y < -0.5
                    || local_x >= scaled_w as f32 + 0.5
                    || local_y >= scaled_h as f32 + 0.5
                {
                    continue;
                }

                if !use_aa
                    && (local_x < 0.0
                        || local_y < 0.0
                        || local_x >= scaled_w as f32
                        || local_y >= scaled_h as f32)
                {
                    continue;
                }

                let src_px = if use_aa {
                    sample_bilinear(&scaled, local_x - 0.5, local_y - 0.5, scaled_w, scaled_h)
                } else {
                    let ix = (local_x as u32).min(scaled_w - 1);
                    let iy = (local_y as u32).min(scaled_h - 1);
                    *scaled.get_pixel(ix, iy)
                };

                if overwrite_transparent
                    && self.overwrite_mask_allows(local_x, local_y, scaled_mask.as_ref())
                {
                    out.put_pixel(dx, dy, src_px);
                } else if src_px[3] > 0 {
                    let dst = *out.get_pixel(dx, dy);
                    out.put_pixel(dx, dy, alpha_blend(dst, src_px));
                }
            }
        }
    }

    fn half_size(&self) -> Vec2 {
        Vec2::new(
            self.source.width() as f32 / 2.0,
            self.source.height() as f32 / 2.0,
        )
    }

    /// The scaled half-size.
    fn scaled_half(&self) -> Vec2 {
        Vec2::new(
            self.source.width() as f32 * self.scale_x / 2.0,
            self.source.height() as f32 * self.scale_y / 2.0,
        )
    }

    /// Snap `center` so the image's top-left corner lands on a whole canvas pixel.
    /// This matches the rounding that `commit()` uses when blending pixels, so
    /// the drag preview and the committed result are always pixel-aligned.
    /// Only applied when the overlay has no rotation (rotating introduces
    /// sub-pixel placement by design).
    pub fn snap_center_to_pixel(&mut self) {
        if self.rotation != 0.0 {
            return;
        }
        let sh = self.scaled_half();
        let top_left_x = self.center.x - sh.x;
        let top_left_y = self.center.y - sh.y;
        self.center = Pos2::new(top_left_x.round() + sh.x, top_left_y.round() + sh.y);
    }

    /// Rotation origin in canvas coords.
    fn anchor_canvas(&self) -> Pos2 {
        Pos2::new(
            self.center.x + self.anchor_offset.x,
            self.center.y + self.anchor_offset.y,
        )
    }

    /// Rotate a point around the anchor.
    fn rotate_point(&self, p: Pos2) -> Pos2 {
        let anchor = self.anchor_canvas();
        let dx = p.x - anchor.x;
        let dy = p.y - anchor.y;
        let cos = self.rotation.cos();
        let sin = self.rotation.sin();
        Pos2::new(
            anchor.x + dx * cos - dy * sin,
            anchor.y + dx * sin + dy * cos,
        )
    }

    /// Get the 4 corner positions of the overlay in canvas coords (rotated).
    fn corners_canvas(&self) -> [Pos2; 4] {
        let hs = self.scaled_half();
        let c = self.center;
        let tl = Pos2::new(c.x - hs.x, c.y - hs.y);
        let tr = Pos2::new(c.x + hs.x, c.y - hs.y);
        let bl = Pos2::new(c.x - hs.x, c.y + hs.y);
        let br = Pos2::new(c.x + hs.x, c.y + hs.y);
        [
            self.rotate_point(tl),
            self.rotate_point(tr),
            self.rotate_point(bl),
            self.rotate_point(br),
        ]
    }

    /// Get midpoints of edges in canvas coords (rotated).
    fn edge_midpoints_canvas(&self) -> [Pos2; 4] {
        let hs = self.scaled_half();
        let c = self.center;
        let top = Pos2::new(c.x, c.y - hs.y);
        let bottom = Pos2::new(c.x, c.y + hs.y);
        let left = Pos2::new(c.x - hs.x, c.y);
        let right = Pos2::new(c.x + hs.x, c.y);
        [
            self.rotate_point(top),
            self.rotate_point(bottom),
            self.rotate_point(left),
            self.rotate_point(right),
        ]
    }

    /// Convert canvas position to screen position.
    fn canvas_to_screen(&self, canvas_pos: Pos2, image_rect: Rect, zoom: f32) -> Pos2 {
        Pos2::new(
            image_rect.min.x + canvas_pos.x * zoom,
            image_rect.min.y + canvas_pos.y * zoom,
        )
    }

    /// Convert screen position to canvas position.
    fn screen_to_canvas_f(&self, screen_pos: Pos2, image_rect: Rect, zoom: f32) -> Pos2 {
        Pos2::new(
            (screen_pos.x - image_rect.min.x) / zoom,
            (screen_pos.y - image_rect.min.y) / zoom,
        )
    }

    // -----------------------------------------------------------------------
    //  GPU-accelerated rendering
    // -----------------------------------------------------------------------

    /// Ensure the source image is uploaded once. Interactive transforms only
    /// move the textured quad, so resize/rotate never trigger CPU resampling or
    /// a texture upload.
    pub fn ensure_gpu_texture(&mut self, ctx: &egui::Context) {
        if self.gpu_texture.is_some() {
            return;
        }

        // Convert to egui ColorImage.
        let upload_w = self.source.width();
        let upload_h = self.source.height();
        let raw = self.source.as_raw();
        let pixels: Vec<egui::Color32> = raw
            .par_chunks_exact(4)
            .map(|px| egui::Color32::from_rgba_unmultiplied(px[0], px[1], px[2], px[3]))
            .collect();
        let color_image = egui::ColorImage {
            size: [upload_w as usize, upload_h as usize],
            source_size: egui::Vec2::new(upload_w as f32, upload_h as f32),
            pixels,
        };

        // A11: reuse existing TextureHandle via tex.set() instead of allocating new
        // Derive filter mode from the anti_aliasing setting so nearest-neighbour
        // paste (pixel art) is not blurred during drag.
        let tex_filter = if self.anti_aliasing {
            egui::TextureFilter::Linear
        } else {
            egui::TextureFilter::Nearest
        };
        let tex_options = egui::TextureOptions {
            magnification: tex_filter,
            minification: tex_filter,
            ..Default::default()
        };
        let image_data = egui::ImageData::Color(std::sync::Arc::new(color_image));
        if let Some(ref mut tex) = self.gpu_texture {
            tex.set(image_data, tex_options);
        } else {
            let tex = ctx.load_texture("paste_overlay_gpu", image_data, tex_options);
            self.gpu_texture = Some(tex);
        }
    }

    /// Draw the paste overlay image using GPU-accelerated textured mesh.
    /// The GPU handles rotation and translation — no per-pixel CPU work needed.
    pub fn draw_gpu(&self, painter: &egui::Painter, image_rect: Rect, zoom: f32) {
        let tex = match &self.gpu_texture {
            Some(t) => t,
            None => return,
        };

        let corners = self.corners_canvas();
        // corners: [TL, TR, BL, BR] in canvas coords
        let s_tl = self.canvas_to_screen(corners[0], image_rect, zoom);
        let s_tr = self.canvas_to_screen(corners[1], image_rect, zoom);
        let s_bl = self.canvas_to_screen(corners[2], image_rect, zoom);
        let s_br = self.canvas_to_screen(corners[3], image_rect, zoom);

        // Build a textured quad (two triangles).
        let white = Color32::WHITE;
        let mut mesh = egui::Mesh::with_texture(tex.id());

        // Vertices: TL, TR, BL, BR  with UV corners
        mesh.vertices.push(egui::epaint::Vertex {
            pos: s_tl,
            uv: Pos2::new(0.0, 0.0),
            color: white,
        });
        mesh.vertices.push(egui::epaint::Vertex {
            pos: s_tr,
            uv: Pos2::new(1.0, 0.0),
            color: white,
        });
        mesh.vertices.push(egui::epaint::Vertex {
            pos: s_bl,
            uv: Pos2::new(0.0, 1.0),
            color: white,
        });
        mesh.vertices.push(egui::epaint::Vertex {
            pos: s_br,
            uv: Pos2::new(1.0, 1.0),
            color: white,
        });

        // Two triangles: TL-TR-BL and TR-BR-BL
        mesh.indices.extend_from_slice(&[0, 1, 2, 1, 3, 2]);

        painter.add(egui::Shape::mesh(mesh));
    }

    // -----------------------------------------------------------------------
    //  Rendering
    // -----------------------------------------------------------------------

    /// Draw the paste overlay on the canvas painter, including handles.
    pub fn draw(
        &self,
        painter: &egui::Painter,
        image_rect: Rect,
        zoom: f32,
        _is_dark: bool,
        accent: Color32,
    ) {
        let [ar, ag, ab, _] = accent.to_array();
        let accent_glow = Color32::from_rgba_unmultiplied(ar, ag, ab, 50);
        let accent_semi = Color32::from_rgba_unmultiplied(ar, ag, ab, 180);
        let accent_fill = Color32::from_rgba_unmultiplied(ar, ag, ab, 220);
        let handle_border = Color32::WHITE;

        let corners = self.corners_canvas();
        let screen_corners: Vec<Pos2> = corners
            .iter()
            .map(|c| self.canvas_to_screen(*c, image_rect, zoom))
            .collect();

        // Draw edges: TL→TR, TR→BR, BR→BL, BL→TL
        let edge_order = [(0, 1), (1, 3), (3, 2), (2, 0)];

        // Outer glow (wide, transparent accent).
        for &(a, b) in &edge_order {
            painter.line_segment(
                [screen_corners[a], screen_corners[b]],
                Stroke::new(5.0, accent_glow),
            );
        }
        // Mid glow.
        for &(a, b) in &edge_order {
            painter.line_segment(
                [screen_corners[a], screen_corners[b]],
                Stroke::new(3.0, accent_semi),
            );
        }
        // Crisp inner edge.
        for &(a, b) in &edge_order {
            painter.line_segment(
                [screen_corners[a], screen_corners[b]],
                Stroke::new(1.0, accent),
            );
        }

        // --- Corner handles (rounded accent squares with white border + shadow) ---
        let handle_size = 5.0;
        for &sc in &screen_corners {
            let r = Rect::from_center_size(sc, Vec2::splat(handle_size * 2.0));
            // Shadow.
            let shadow_r = r.translate(Vec2::new(1.0, 1.0));
            painter.rect_filled(shadow_r, 3.0, Color32::from_black_alpha(60));
            // Fill.
            painter.rect_filled(r, 3.0, accent_fill);
            painter.rect_stroke(
                r,
                3.0,
                Stroke::new(1.5, handle_border),
                egui::StrokeKind::Middle,
            );
        }

        // --- Edge midpoint handles (smaller rounded rectangles) ---
        let midpoints = self.edge_midpoints_canvas();
        let screen_mids: Vec<Pos2> = midpoints
            .iter()
            .map(|c| self.canvas_to_screen(*c, image_rect, zoom))
            .collect();
        let mid_handle_size = 4.0;
        for &sm in &screen_mids {
            let r = Rect::from_center_size(sm, Vec2::splat(mid_handle_size * 2.0));
            let shadow_r = r.translate(Vec2::new(1.0, 1.0));
            painter.rect_filled(shadow_r, 2.0, Color32::from_black_alpha(50));
            painter.rect_filled(r, 2.0, accent_fill);
            painter.rect_stroke(
                r,
                2.0,
                Stroke::new(1.0, handle_border),
                egui::StrokeKind::Middle,
            );
        }

        // --- Rotation handle: accent stem + glowing circle ---
        let top_mid_screen = screen_mids[0];
        let rotate_distance = 30.0;
        let center_screen = self.canvas_to_screen(self.center, image_rect, zoom);
        let dir = if top_mid_screen.distance(center_screen) > 0.1 {
            let d = Pos2::new(
                top_mid_screen.x - center_screen.x,
                top_mid_screen.y - center_screen.y,
            );
            let len = (d.x * d.x + d.y * d.y).sqrt();
            Vec2::new(d.x / len, d.y / len)
        } else {
            Vec2::new(0.0, -1.0)
        };
        let rotate_pos = Pos2::new(
            top_mid_screen.x + dir.x * rotate_distance,
            top_mid_screen.y + dir.y * rotate_distance,
        );
        // Stem line (glow + solid).
        painter.line_segment([top_mid_screen, rotate_pos], Stroke::new(3.0, accent_glow));
        painter.line_segment([top_mid_screen, rotate_pos], Stroke::new(1.0, accent_semi));
        // Outer glow circle.
        painter.circle_filled(rotate_pos, 8.0, accent_glow);
        // Filled circle.
        painter.circle_filled(rotate_pos, 6.0, accent_fill);
        painter.circle_stroke(rotate_pos, 6.0, Stroke::new(1.5, handle_border));
        // Rotation icon.
        painter.text(
            rotate_pos,
            egui::Align2::CENTER_CENTER,
            "↻",
            egui::FontId::proportional(9.0),
            Color32::WHITE,
        );

        // --- Anchor point (accent crosshair with glowing center) ---
        let anchor_screen = self.canvas_to_screen(self.anchor_canvas(), image_rect, zoom);
        let ch = 9.0;
        // Glow cross.
        painter.line_segment(
            [
                Pos2::new(anchor_screen.x - ch, anchor_screen.y),
                Pos2::new(anchor_screen.x + ch, anchor_screen.y),
            ],
            Stroke::new(3.0, accent_glow),
        );
        painter.line_segment(
            [
                Pos2::new(anchor_screen.x, anchor_screen.y - ch),
                Pos2::new(anchor_screen.x, anchor_screen.y + ch),
            ],
            Stroke::new(3.0, accent_glow),
        );
        // Solid cross.
        painter.line_segment(
            [
                Pos2::new(anchor_screen.x - ch, anchor_screen.y),
                Pos2::new(anchor_screen.x + ch, anchor_screen.y),
            ],
            Stroke::new(1.5, accent),
        );
        painter.line_segment(
            [
                Pos2::new(anchor_screen.x, anchor_screen.y - ch),
                Pos2::new(anchor_screen.x, anchor_screen.y + ch),
            ],
            Stroke::new(1.5, accent),
        );
        // Center dot with glow.
        painter.circle_filled(anchor_screen, 5.0, accent_glow);
        painter.circle_filled(anchor_screen, 3.5, accent_fill);
        painter.circle_stroke(anchor_screen, 3.5, Stroke::new(1.0, Color32::WHITE));
    }

    // -----------------------------------------------------------------------
    //  Hit testing
    // -----------------------------------------------------------------------

    /// Determine which handle (if any) is under the given screen position.
    pub fn hit_test(&self, screen_pos: Pos2, image_rect: Rect, zoom: f32) -> Option<HandleKind> {
        let grab_radius = 10.0;

        // Rotation handle.
        let midpoints = self.edge_midpoints_canvas();
        let top_mid_screen = self.canvas_to_screen(midpoints[0], image_rect, zoom);
        let center_screen = self.canvas_to_screen(self.center, image_rect, zoom);
        let dir = if top_mid_screen.distance(center_screen) > 0.1 {
            let d = Pos2::new(
                top_mid_screen.x - center_screen.x,
                top_mid_screen.y - center_screen.y,
            );
            let len = (d.x * d.x + d.y * d.y).sqrt();
            Vec2::new(d.x / len, d.y / len)
        } else {
            Vec2::new(0.0, -1.0)
        };
        let rotate_pos = Pos2::new(
            top_mid_screen.x + dir.x * 30.0,
            top_mid_screen.y + dir.y * 30.0,
        );
        if screen_pos.distance(rotate_pos) < grab_radius {
            return Some(HandleKind::Rotate);
        }

        // Anchor.
        let anchor_screen = self.canvas_to_screen(self.anchor_canvas(), image_rect, zoom);
        if screen_pos.distance(anchor_screen) < grab_radius {
            return Some(HandleKind::Anchor);
        }

        // Corner handles.
        let corners = self.corners_canvas();
        let screen_corners: Vec<Pos2> = corners
            .iter()
            .map(|c| self.canvas_to_screen(*c, image_rect, zoom))
            .collect();
        let corner_handles = [
            (0, HandleKind::TopLeft),
            (1, HandleKind::TopRight),
            (2, HandleKind::BottomLeft),
            (3, HandleKind::BottomRight),
        ];
        for (idx, kind) in &corner_handles {
            if screen_pos.distance(screen_corners[*idx]) < grab_radius {
                return Some(*kind);
            }
        }

        // Edge midpoint handles.
        let screen_mids: Vec<Pos2> = midpoints
            .iter()
            .map(|c| self.canvas_to_screen(*c, image_rect, zoom))
            .collect();
        let mid_handles = [
            (0, HandleKind::Top),
            (1, HandleKind::Bottom),
            (2, HandleKind::Left),
            (3, HandleKind::Right),
        ];
        for (idx, kind) in &mid_handles {
            if screen_pos.distance(screen_mids[*idx]) < grab_radius {
                return Some(*kind);
            }
        }

        // Inside the bounding quad → move.
        if self.point_in_rotated_rect(screen_pos, image_rect, zoom) {
            return Some(HandleKind::Move);
        }

        None
    }

    /// Cursor for the currently active or hovered transform handle.
    pub fn cursor_icon(
        &self,
        screen_pos: Pos2,
        image_rect: Rect,
        zoom: f32,
    ) -> Option<egui::CursorIcon> {
        let handle = self
            .active_handle
            .or_else(|| self.hit_test(screen_pos, image_rect, zoom))?;
        use egui::CursorIcon;
        Some(match handle {
            HandleKind::Move | HandleKind::Anchor if self.active_handle.is_some() => {
                CursorIcon::Grabbing
            }
            HandleKind::Move | HandleKind::Anchor => CursorIcon::Grab,
            HandleKind::Top | HandleKind::Bottom => CursorIcon::ResizeVertical,
            HandleKind::Left | HandleKind::Right => CursorIcon::ResizeHorizontal,
            HandleKind::TopLeft | HandleKind::BottomRight => CursorIcon::ResizeNwSe,
            HandleKind::TopRight | HandleKind::BottomLeft => CursorIcon::ResizeNeSw,
            HandleKind::Rotate => CursorIcon::Alias,
        })
    }

    /// Test if a screen point is inside the rotated bounding rectangle.
    fn point_in_rotated_rect(&self, screen_pos: Pos2, image_rect: Rect, zoom: f32) -> bool {
        // Un-rotate the point around the center, then test axis-aligned rect.
        let center_screen = self.canvas_to_screen(self.center, image_rect, zoom);
        let dx = screen_pos.x - center_screen.x;
        let dy = screen_pos.y - center_screen.y;
        let cos = (-self.rotation).cos();
        let sin = (-self.rotation).sin();
        let ux = dx * cos - dy * sin;
        let uy = dx * sin + dy * cos;

        let hs = self.scaled_half();
        let hw = hs.x * zoom;
        let hh = hs.y * zoom;
        ux.abs() <= hw && uy.abs() <= hh
    }

    // -----------------------------------------------------------------------
    //  Interaction (called each frame from canvas rendering)
    // -----------------------------------------------------------------------

    /// Process mouse interaction. Returns true if the overlay consumed the event.
    pub fn handle_input(
        &mut self,
        ui: &egui::Ui,
        image_rect: Rect,
        workspace_rect: Rect,
        zoom: f32,
        input_blocked: bool,
    ) -> bool {
        let mouse_pos = match ui.input(|i| {
            i.pointer.interact_pos().or_else(|| {
                i.events.iter().rev().find_map(|e| match e {
                    egui::Event::PointerButton {
                        pressed: true, pos, ..
                    } => Some(*pos),
                    egui::Event::Touch {
                        phase: egui::TouchPhase::Start | egui::TouchPhase::Move,
                        pos,
                        ..
                    } => Some(*pos),
                    _ => None,
                })
            })
        }) {
            Some(p) => p,
            None => return false,
        };
        let primary_down = ui.input(|i| i.pointer.primary_down());
        let primary_pressed = ui.input(|i| i.pointer.primary_pressed());
        let primary_released = ui.input(|i| i.pointer.any_released());
        self.shift_held = ui.input(|i| i.modifiers.shift);

        // New grabs are allowed anywhere in the unobstructed workspace, even
        // outside the document. An active grab keeps pointer capture to release.
        if self.active_handle.is_none() && (input_blocked || !workspace_rect.contains(mouse_pos)) {
            return false;
        }

        // Start drag.
        if primary_pressed && let Some(handle) = self.hit_test(mouse_pos, image_rect, zoom) {
            self.active_handle = Some(handle);
            self.drag_start_mouse = Some(mouse_pos);
            self.drag_start_center = self.center;
            self.drag_start_scale_x = self.scale_x;
            self.drag_start_scale_y = self.scale_y;
            self.drag_start_rotation = self.rotation;
            self.drag_start_anchor = self.anchor_offset;
            return true;
        }

        // Continue drag.
        if primary_down
            && let (Some(handle), Some(start)) = (self.active_handle, self.drag_start_mouse)
        {
            let delta_screen = Vec2::new(mouse_pos.x - start.x, mouse_pos.y - start.y);
            let delta_canvas = Vec2::new(delta_screen.x / zoom, delta_screen.y / zoom);

            match handle {
                HandleKind::Move => {
                    self.center = Pos2::new(
                        self.drag_start_center.x + delta_canvas.x,
                        self.drag_start_center.y + delta_canvas.y,
                    );
                    self.snap_center_to_pixel();
                }
                HandleKind::Anchor => {
                    self.anchor_offset = Vec2::new(
                        self.drag_start_anchor.x + delta_canvas.x,
                        self.drag_start_anchor.y + delta_canvas.y,
                    );
                }
                HandleKind::Rotate => {
                    let anchor_screen = self.canvas_to_screen(
                        Pos2::new(
                            self.drag_start_center.x + self.anchor_offset.x,
                            self.drag_start_center.y + self.anchor_offset.y,
                        ),
                        image_rect,
                        zoom,
                    );
                    let start_angle = (start.y - anchor_screen.y).atan2(start.x - anchor_screen.x);
                    let current_angle =
                        (mouse_pos.y - anchor_screen.y).atan2(mouse_pos.x - anchor_screen.x);
                    let mut new_rot = self.drag_start_rotation + (current_angle - start_angle);
                    // Snap to 45° increments when shift held.
                    if self.shift_held {
                        let snap = std::f32::consts::FRAC_PI_4; // 45°
                        new_rot = (new_rot / snap).round() * snap;
                    }
                    self.rotation = new_rot;
                }
                _ => {
                    // Resize handles.
                    self.handle_resize(handle, delta_canvas);
                }
            }
            return true;
        }

        // End drag.
        if primary_released && self.active_handle.is_some() {
            let before = PasteOverlayTransform {
                center: self.drag_start_center,
                rotation: self.drag_start_rotation,
                scale_x: self.drag_start_scale_x,
                scale_y: self.drag_start_scale_y,
                anchor_offset: self.drag_start_anchor,
            };
            let after = self.transform();
            if before != after {
                self.pending_transform_checkpoint = Some((before, after));
            }
            self.active_handle = None;
            self.drag_start_mouse = None;
            return true;
        }

        false
    }

    /// Handle resize from edge/corner dragging.
    fn handle_resize(&mut self, handle: HandleKind, delta_canvas: Vec2) {
        let src_w = self.source.width() as f32;
        let src_h = self.source.height() as f32;
        if src_w < 1.0 || src_h < 1.0 {
            return;
        }

        // Un-rotate delta to local axes.
        let cos = (-self.rotation).cos();
        let sin = (-self.rotation).sin();
        let local_dx = delta_canvas.x * cos - delta_canvas.y * sin;
        let local_dy = delta_canvas.x * sin + delta_canvas.y * cos;

        let start_sx = self.drag_start_scale_x;
        let start_sy = self.drag_start_scale_y;
        let start_w = src_w * start_sx;
        let start_h = src_h * start_sy;

        let (mut new_sx, mut new_sy, _offset_x, _offset_y) = match handle {
            HandleKind::Right => {
                let new_w = (start_w + local_dx).max(4.0);
                let sx = new_w / src_w;
                (sx, start_sy, local_dx / 2.0, 0.0)
            }
            HandleKind::Left => {
                let new_w = (start_w - local_dx).max(4.0);
                let sx = new_w / src_w;
                (sx, start_sy, local_dx / 2.0, 0.0)
            }
            HandleKind::Bottom => {
                let new_h = (start_h + local_dy).max(4.0);
                let sy = new_h / src_h;
                (start_sx, sy, 0.0, local_dy / 2.0)
            }
            HandleKind::Top => {
                let new_h = (start_h - local_dy).max(4.0);
                let sy = new_h / src_h;
                (start_sx, sy, 0.0, local_dy / 2.0)
            }
            HandleKind::TopLeft => {
                let new_w = (start_w - local_dx).max(4.0);
                let new_h = (start_h - local_dy).max(4.0);
                (new_w / src_w, new_h / src_h, local_dx / 2.0, local_dy / 2.0)
            }
            HandleKind::TopRight => {
                let new_w = (start_w + local_dx).max(4.0);
                let new_h = (start_h - local_dy).max(4.0);
                (new_w / src_w, new_h / src_h, local_dx / 2.0, local_dy / 2.0)
            }
            HandleKind::BottomLeft => {
                let new_w = (start_w - local_dx).max(4.0);
                let new_h = (start_h + local_dy).max(4.0);
                (new_w / src_w, new_h / src_h, local_dx / 2.0, local_dy / 2.0)
            }
            HandleKind::BottomRight => {
                let new_w = (start_w + local_dx).max(4.0);
                let new_h = (start_h + local_dy).max(4.0);
                (new_w / src_w, new_h / src_h, local_dx / 2.0, local_dy / 2.0)
            }
            _ => return,
        };

        // Lock aspect ratio when shift is held.
        if self.shift_held {
            let aspect = start_sx / start_sy;
            // Use the larger scale change.
            let avg = (new_sx / start_sx).max(new_sy / start_sy);
            new_sx = start_sx * avg;
            new_sy = new_sx / aspect;
        }

        self.scale_x = new_sx;
        self.scale_y = new_sy;

        // Move center so the opposite edge/corner stays fixed, including Shift-locked scaling.
        let cos_r = self.rotation.cos();
        let sin_r = self.rotation.sin();
        let anchor_local = match handle {
            HandleKind::Right => Vec2::new(-start_w * 0.5, 0.0),
            HandleKind::Left => Vec2::new(start_w * 0.5, 0.0),
            HandleKind::Bottom => Vec2::new(0.0, -start_h * 0.5),
            HandleKind::Top => Vec2::new(0.0, start_h * 0.5),
            HandleKind::TopLeft => Vec2::new(start_w * 0.5, start_h * 0.5),
            HandleKind::TopRight => Vec2::new(-start_w * 0.5, start_h * 0.5),
            HandleKind::BottomLeft => Vec2::new(start_w * 0.5, -start_h * 0.5),
            HandleKind::BottomRight => Vec2::new(-start_w * 0.5, -start_h * 0.5),
            _ => Vec2::ZERO,
        };
        let fixed_anchor = Pos2::new(
            self.drag_start_center.x + anchor_local.x * cos_r - anchor_local.y * sin_r,
            self.drag_start_center.y + anchor_local.x * sin_r + anchor_local.y * cos_r,
        );
        let new_w = src_w * new_sx;
        let new_h = src_h * new_sy;
        let new_anchor_local = match handle {
            HandleKind::Right => Vec2::new(-new_w * 0.5, 0.0),
            HandleKind::Left => Vec2::new(new_w * 0.5, 0.0),
            HandleKind::Bottom => Vec2::new(0.0, -new_h * 0.5),
            HandleKind::Top => Vec2::new(0.0, new_h * 0.5),
            HandleKind::TopLeft => Vec2::new(new_w * 0.5, new_h * 0.5),
            HandleKind::TopRight => Vec2::new(-new_w * 0.5, new_h * 0.5),
            HandleKind::BottomLeft => Vec2::new(new_w * 0.5, -new_h * 0.5),
            HandleKind::BottomRight => Vec2::new(-new_w * 0.5, -new_h * 0.5),
            _ => Vec2::ZERO,
        };
        self.center = Pos2::new(
            fixed_anchor.x - (new_anchor_local.x * cos_r - new_anchor_local.y * sin_r),
            fixed_anchor.y - (new_anchor_local.x * sin_r + new_anchor_local.y * cos_r),
        );
    }

    // -----------------------------------------------------------------------
    //  Context Menu
    // -----------------------------------------------------------------------

    /// Show a context menu for the paste overlay. Returns true if Commit or
    /// Cancel was clicked.
    pub fn context_menu(&mut self, ui: &mut egui::Ui) -> Option<bool> {
        let mut result = None;
        ui.menu_button("Paste Options", |ui| {
            // Interpolation selector
            ui.label("Filter:");
            for interp in Interpolation::all() {
                if ui
                    .selectable_label(self.interpolation == *interp, interp.label())
                    .clicked()
                {
                    self.interpolation = *interp;
                }
            }
            ui.separator();

            // Reset transforms
            if ui.button("Reset Position").clicked() {
                self.rotation = 0.0;
                self.scale_x = 1.0;
                self.scale_y = 1.0;
                self.anchor_offset = Vec2::ZERO;
                ui.close();
            }
            if ui.button("Center Anchor").clicked() {
                self.anchor_offset = Vec2::ZERO;
                ui.close();
            }

            ui.separator();
            if ui.button("✓ Commit (Enter)").clicked() {
                result = Some(true);
                ui.close();
            }
            if ui.button("✗ Cancel (Esc)").clicked() {
                result = Some(false);
                ui.close();
            }
        });
        result
    }

    // -----------------------------------------------------------------------
    //  Commit — rasterize the transformed overlay onto the active layer
    // -----------------------------------------------------------------------

    /// Commit the paste overlay to the active layer of the canvas.
    pub fn commit(self, state: &mut CanvasState) {
        let idx = state.active_layer_index;
        if idx >= state.layers.len() {
            return;
        }

        let cw = state.width;
        let ch = state.height;
        let src_w = self.source.width() as f32;
        let src_h = self.source.height() as f32;

        let filter = self.interpolation.to_filter();

        // Scale the source image and mask first.
        let scaled_w = (src_w * self.scale_x).round().max(1.0) as u32;
        let scaled_h = (src_h * self.scale_y).round().max(1.0) as u32;
        let scaled = imageops::resize(&self.source, scaled_w, scaled_h, filter);
        let scaled_mask = self
            .overwrite_mask
            .as_ref()
            .map(|mask| imageops::resize(mask, scaled_w, scaled_h, imageops::FilterType::Nearest));

        // Compute tight bounding box of the rotated paste to limit iteration.
        let corners = self.corners_canvas();
        let mut bb_min_x = cw as f32;
        let mut bb_min_y = ch as f32;
        let mut bb_max_x = 0.0f32;
        let mut bb_max_y = 0.0f32;
        for c in &corners {
            bb_min_x = bb_min_x.min(c.x);
            bb_min_y = bb_min_y.min(c.y);
            bb_max_x = bb_max_x.max(c.x);
            bb_max_y = bb_max_y.max(c.y);
        }
        let row_start = (bb_min_y.floor().max(0.0)) as u32;
        let row_end = (bb_max_y.ceil().min(ch as f32 - 1.0)) as u32;
        let col_start = (bb_min_x.floor().max(0.0)) as u32;
        let col_end = (bb_max_x.ceil().min(cw as f32 - 1.0)) as u32;

        let anchor = self.anchor_canvas();
        let cos = self.rotation.cos();
        let sin = self.rotation.sin();
        let origin_x = self.center.x - scaled_w as f32 / 2.0;
        let origin_y = self.center.y - scaled_h as f32 / 2.0;
        let use_aa = self.anti_aliasing;
        let overwrite_transparent = self.overwrite_transparent_pixels;

        // Flatten layer for fast reads, compute source pixels in parallel,
        // then apply patches back to the tiled layer.
        let layer = &mut state.layers[idx];
        let flat = layer.pixels.to_rgba_image();

        let rows: Vec<u32> = (row_start..=row_end).collect();
        let patches: Vec<(u32, u32, Rgba<u8>)> = rows
            .par_iter()
            .flat_map(|&dy| {
                let mut row_patches = Vec::new();
                let py = dy as f32 + 0.5;
                let ry = py - anchor.y;
                for dx in col_start..=col_end {
                    let px = dx as f32 + 0.5;
                    let rx = px - anchor.x;
                    let ur_x = rx * cos + ry * sin + anchor.x;
                    let ur_y = -rx * sin + ry * cos + anchor.y;

                    let local_x = ur_x - origin_x;
                    let local_y = ur_y - origin_y;

                    if local_x < -0.5
                        || local_y < -0.5
                        || local_x >= scaled_w as f32 + 0.5
                        || local_y >= scaled_h as f32 + 0.5
                    {
                        continue;
                    }

                    // For nearest-neighbour mode enforce tight pixel-grid bounds to
                    // eliminate the edge-bleed that the ±0.5 AA window can cause.
                    if !use_aa
                        && (local_x < 0.0
                            || local_y < 0.0
                            || local_x >= scaled_w as f32
                            || local_y >= scaled_h as f32)
                    {
                        continue;
                    }

                    let src_px = if use_aa {
                        sample_bilinear(&scaled, local_x - 0.5, local_y - 0.5, scaled_w, scaled_h)
                    } else {
                        let ix = (local_x as u32).min(scaled_w - 1);
                        let iy = (local_y as u32).min(scaled_h - 1);
                        *scaled.get_pixel(ix, iy)
                    };
                    if overwrite_transparent
                        && self.overwrite_mask_allows(local_x, local_y, scaled_mask.as_ref())
                    {
                        row_patches.push((dx, dy, src_px));
                    } else if src_px[3] > 0 {
                        let dst = *flat.get_pixel(dx, dy);
                        let blended = alpha_blend(dst, src_px);
                        row_patches.push((dx, dy, blended));
                    }
                }
                row_patches
            })
            .collect();

        for (dx, dy, px) in patches {
            layer.pixels.put_pixel(dx, dy, px);
        }

        state.mark_dirty(None);
    }

    /// Check whether the cached preview is still valid for the current transform.
    pub fn preview_cache_valid(&self, canvas_w: u32, canvas_h: u32) -> bool {
        match &self.cached_preview {
            Some((_, c, sx, sy, rot, ao, cw, ch)) => {
                *cw == canvas_w
                    && *ch == canvas_h
                    && (c.x - self.center.x).abs() < 0.001
                    && (c.y - self.center.y).abs() < 0.001
                    && (*sx - self.scale_x).abs() < 0.0001
                    && (*sy - self.scale_y).abs() < 0.0001
                    && (*rot - self.rotation).abs() < 0.0001
                    && (ao.x - self.anchor_offset.x).abs() < 0.001
                    && (ao.y - self.anchor_offset.y).abs() < 0.001
            }
            None => false,
        }
    }

    /// Render a preview of the transformed paste into a temporary image
    /// for compositing on the canvas. Uses caching — only re-renders when
    /// the transform actually changes.
    pub fn render_preview(&mut self, canvas_w: u32, canvas_h: u32) -> &TiledImage {
        if self.preview_cache_valid(canvas_w, canvas_h) {
            return &self.cached_preview.as_ref().unwrap().0;
        }

        let src_w = self.source.width() as f32;
        let src_h = self.source.height() as f32;
        let scaled_w = (src_w * self.scale_x).round().max(1.0) as u32;
        let scaled_h = (src_h * self.scale_y).round().max(1.0) as u32;

        // Re-use cached scaled image if dimensions haven't changed.
        let need_rescale = match &self.cached_scaled {
            Some((_, cw, ch)) => *cw != scaled_w || *ch != scaled_h,
            None => true,
        };
        if need_rescale {
            let scaled = imageops::resize(
                &self.source,
                scaled_w,
                scaled_h,
                imageops::FilterType::Nearest,
            );
            self.cached_scaled = Some((scaled, scaled_w, scaled_h));
        }
        let scaled = &self.cached_scaled.as_ref().unwrap().0;

        // -- FAST PATH: translation-only (no rotation) --
        // Write the scaled image directly into a TiledImage at an offset,
        // skipping the full canvas-sized buffer and per-pixel rasterization.
        let is_translation_only = self.rotation.abs() < 0.0001
            && self.anchor_offset.x.abs() < 0.001
            && self.anchor_offset.y.abs() < 0.001;

        let tiled = if is_translation_only {
            let origin_x = (self.center.x - scaled_w as f32 / 2.0).round() as i32;
            let origin_y = (self.center.y - scaled_h as f32 / 2.0).round() as i32;

            let mut tiled = TiledImage::new(canvas_w, canvas_h);
            // Determine the visible region.
            let dst_x0 = origin_x.max(0) as u32;
            let dst_y0 = origin_y.max(0) as u32;
            let dst_x1 = ((origin_x + scaled_w as i32) as u32).min(canvas_w);
            let dst_y1 = ((origin_y + scaled_h as i32) as u32).min(canvas_h);
            let src_x0 = (dst_x0 as i32 - origin_x) as u32;
            let src_y0 = (dst_y0 as i32 - origin_y) as u32;

            for dy in dst_y0..dst_y1 {
                let sy = src_y0 + (dy - dst_y0);
                for dx in dst_x0..dst_x1 {
                    let sx = src_x0 + (dx - dst_x0);
                    let px = *scaled.get_pixel(sx, sy);
                    if px[3] > 0 {
                        tiled.put_pixel(dx, dy, px);
                    }
                }
            }
            tiled
        } else {
            // -- GENERAL PATH: rotation + translation --
            let anchor = self.anchor_canvas();
            let cos = self.rotation.cos();
            let sin = self.rotation.sin();
            let origin_x = self.center.x - scaled_w as f32 / 2.0;
            let origin_y = self.center.y - scaled_h as f32 / 2.0;

            // Bounding box.
            let corners = self.corners_canvas();
            let mut bb_min_x = canvas_w as f32;
            let mut bb_min_y = canvas_h as f32;
            let mut bb_max_x = 0.0f32;
            let mut bb_max_y = 0.0f32;
            for c in &corners {
                bb_min_x = bb_min_x.min(c.x);
                bb_min_y = bb_min_y.min(c.y);
                bb_max_x = bb_max_x.max(c.x);
                bb_max_y = bb_max_y.max(c.y);
            }
            let px_min_x = (bb_min_x.floor().max(0.0)) as u32;
            let px_min_y = (bb_min_y.floor().max(0.0)) as u32;
            let px_max_x = (bb_max_x.ceil().min(canvas_w as f32 - 1.0)) as u32;
            let px_max_y = (bb_max_y.ceil().min(canvas_h as f32 - 1.0)) as u32;

            // Render rows in parallel directly into a flat pixel buffer.
            let full_w = canvas_w as usize;
            let rows: Vec<u32> = (px_min_y..=px_max_y).collect();
            let row_strips: Vec<(u32, Vec<u8>)> = rows
                .par_iter()
                .map(|&dy| {
                    let mut strip = Vec::new();
                    let py = dy as f32 + 0.5;
                    let ry = py - anchor.y;
                    for dx in px_min_x..=px_max_x {
                        let px = dx as f32 + 0.5;
                        let rx = px - anchor.x;
                        let ur_x = rx * cos + ry * sin + anchor.x;
                        let ur_y = -rx * sin + ry * cos + anchor.y;

                        let local_x = ur_x - origin_x;
                        let local_y = ur_y - origin_y;

                        if local_x < 0.0
                            || local_y < 0.0
                            || local_x >= scaled_w as f32
                            || local_y >= scaled_h as f32
                        {
                            continue;
                        }

                        let src_px = *scaled.get_pixel(
                            (local_x as u32).min(scaled_w - 1),
                            (local_y as u32).min(scaled_h - 1),
                        );
                        if src_px[3] > 0 {
                            strip.push(dx as u8);
                            strip.push((dx >> 8) as u8);
                            strip.extend_from_slice(&src_px.0);
                        }
                    }
                    (dy, strip)
                })
                .collect();

            // Assemble into a flat RGBA buffer.
            let mut buf = vec![0u8; canvas_w as usize * canvas_h as usize * 4];
            for (dy, strip) in &row_strips {
                let mut i = 0;
                while i + 5 < strip.len() {
                    let dx = strip[i] as u32 | ((strip[i + 1] as u32) << 8);
                    let off = (*dy as usize * full_w + dx as usize) * 4;
                    buf[off] = strip[i + 2];
                    buf[off + 1] = strip[i + 3];
                    buf[off + 2] = strip[i + 4];
                    buf[off + 3] = strip[i + 5];
                    i += 6;
                }
            }

            let preview = RgbaImage::from_raw(canvas_w, canvas_h, buf).unwrap();
            TiledImage::from_rgba_image(&preview)
        };
        self.cached_preview = Some((
            tiled,
            self.center,
            self.scale_x,
            self.scale_y,
            self.rotation,
            self.anchor_offset,
            canvas_w,
            canvas_h,
        ));
        &self.cached_preview.as_ref().unwrap().0
    }
}

/// Bilinear interpolation sample from an RgbaImage at fractional coords.
/// Uses clamp-to-edge for out-of-bounds samples (matches GPU ClampToEdge
/// behavior) to prevent darkened borders from blending with transparent black.
#[inline]
fn sample_bilinear(img: &RgbaImage, x: f32, y: f32, w: u32, h: u32) -> Rgba<u8> {
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;

    let sample = |sx: i32, sy: i32| -> [f32; 4] {
        let cx = sx.clamp(0, w as i32 - 1) as u32;
        let cy = sy.clamp(0, h as i32 - 1) as u32;
        let p = img.get_pixel(cx, cy).0;
        [p[0] as f32, p[1] as f32, p[2] as f32, p[3] as f32]
    };

    let p00 = sample(x0, y0);
    let p10 = sample(x0 + 1, y0);
    let p01 = sample(x0, y0 + 1);
    let p11 = sample(x0 + 1, y0 + 1);

    let inv_fx = 1.0 - fx;
    let inv_fy = 1.0 - fy;
    let w00 = inv_fx * inv_fy;
    let w10 = fx * inv_fy;
    let w01 = inv_fx * fy;
    let w11 = fx * fy;

    Rgba([
        (p00[0] * w00 + p10[0] * w10 + p01[0] * w01 + p11[0] * w11)
            .round()
            .clamp(0.0, 255.0) as u8,
        (p00[1] * w00 + p10[1] * w10 + p01[1] * w01 + p11[1] * w11)
            .round()
            .clamp(0.0, 255.0) as u8,
        (p00[2] * w00 + p10[2] * w10 + p01[2] * w01 + p11[2] * w11)
            .round()
            .clamp(0.0, 255.0) as u8,
        (p00[3] * w00 + p10[3] * w10 + p01[3] * w01 + p11[3] * w11)
            .round()
            .clamp(0.0, 255.0) as u8,
    ])
}

/// Simple alpha-composite: src over dst.
fn alpha_blend(dst: Rgba<u8>, src: Rgba<u8>) -> Rgba<u8> {
    if src[3] == 0 {
        return dst;
    }
    if src[3] == 255 || dst[3] == 0 {
        return src;
    }
    let sa = src[3] as f32 / 255.0;
    let da = dst[3] as f32 / 255.0;
    let out_a = sa + da * (1.0 - sa);
    if out_a < 0.001 {
        return Rgba([0, 0, 0, 0]);
    }
    let inv = 1.0 / out_a;
    Rgba([
        ((src[0] as f32 * sa + dst[0] as f32 * da * (1.0 - sa)) * inv)
            .round()
            .clamp(0.0, 255.0) as u8,
        ((src[1] as f32 * sa + dst[1] as f32 * da * (1.0 - sa)) * inv)
            .round()
            .clamp(0.0, 255.0) as u8,
        ((src[2] as f32 * sa + dst[2] as f32 * da * (1.0 - sa)) * inv)
            .round()
            .clamp(0.0, 255.0) as u8,
        (out_a * 255.0).round().clamp(0.0, 255.0) as u8,
    ])
}

// ---------------------------------------------------------------------------
//  Web system clipboard (browser Clipboard API / native `paste` event)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "wasm32")]
mod web_clipboard {
    use super::RgbaImage;
    use std::cell::RefCell;
    use wasm_bindgen::JsCast;
    use wasm_bindgen::prelude::*;

    thread_local! {
        static LAST_PASTED: RefCell<Option<RgbaImage>> = const { RefCell::new(None) };
    }

    /// Take whatever image the last browser `paste` event decoded, if any.
    pub fn take_pasted_image() -> Option<RgbaImage> {
        LAST_PASTED.with(|p| p.borrow_mut().clone())
    }

    /// Install a single global `paste` listener that decodes image data from
    /// the OS clipboard (Ctrl+V / the browser's native paste event) into
    /// `LAST_PASTED`, where `get_from_system_clipboard` picks it up. There is
    /// no synchronous way to read `navigator.clipboard` on demand from egui's
    /// input handling, so this listens for the real paste event instead —
    /// call once at startup.
    pub fn install_paste_listener(ctx: egui::Context) {
        let result = (|| -> Result<(), JsValue> {
            let window = web_sys::window().ok_or("no window")?;
            let document = window.document().ok_or("no document")?;

            let closure = Closure::<dyn FnMut(_)>::new(move |event: web_sys::ClipboardEvent| {
                let Some(data) = event.clipboard_data() else {
                    return;
                };
                let items = data.items();
                for i in 0..items.length() {
                    let Some(item) = items.get(i) else { continue };
                    if !item.type_().starts_with("image/") {
                        continue;
                    }
                    let Some(file) = item.get_as_file().ok().flatten() else {
                        continue;
                    };
                    let ctx = ctx.clone();
                    wasm_bindgen_futures::spawn_local(async move {
                        if let Ok(buf) =
                            wasm_bindgen_futures::JsFuture::from(file.array_buffer()).await
                        {
                            let array = js_sys::Uint8Array::new(&buf);
                            let bytes = array.to_vec();
                            if let Ok(img) = image::load_from_memory(&bytes) {
                                LAST_PASTED.with(|p| *p.borrow_mut() = Some(img.to_rgba8()));
                                ctx.request_repaint();
                            }
                        }
                    });
                    break;
                }
            });

            document.add_event_listener_with_callback("paste", closure.as_ref().unchecked_ref())?;
            // Leak intentionally: one global listener for the app's lifetime.
            closure.forget();
            Ok(())
        })();

        if let Err(e) = result {
            web_sys::console::error_2(
                &"PaintFE: failed to install clipboard paste listener:".into(),
                &e,
            );
        }
    }

    /// Write PNG bytes to the OS clipboard via the async Clipboard API.
    /// Requires a secure context (https/localhost) and browser support
    /// (Chromium, Safari; Firefox does not support image `ClipboardItem`s as
    /// of writing) — returns `Err` on any failure so the caller can decide
    /// whether to surface it, though callers currently treat this as
    /// best-effort and ignore the result.
    pub async fn write_png_to_clipboard(png_bytes: Vec<u8>) -> Result<(), JsValue> {
        let window = web_sys::window().ok_or("no window")?;
        let clipboard = window.navigator().clipboard();

        let array = js_sys::Uint8Array::from(png_bytes.as_slice());
        let parts = js_sys::Array::new();
        parts.push(&array);
        let blob_opts = web_sys::BlobPropertyBag::new();
        blob_opts.set_type("image/png");
        let blob = web_sys::Blob::new_with_u8_array_sequence_and_options(&parts, &blob_opts)?;

        let record = js_sys::Object::new();
        js_sys::Reflect::set(&record, &JsValue::from_str("image/png"), &blob)?;
        let item = web_sys::ClipboardItem::new_with_record_from_str_to_blob_promise(&record)?;
        let items = js_sys::Array::new();
        items.push(&item);

        wasm_bindgen_futures::JsFuture::from(clipboard.write(&items)).await?;
        Ok(())
    }
}

#[cfg(target_arch = "wasm32")]
pub use web_clipboard::install_paste_listener;
