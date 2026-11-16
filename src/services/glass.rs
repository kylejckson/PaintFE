//! Native backdrop adapters. Unsupported effects leave the editor usable and opaque-capable.
use crate::ui::glass::GlassSettings;
use std::sync::{Arc, Mutex, Weak};
use winit::window::Window;
static WINDOW: Mutex<Option<Weak<Window>>> = Mutex::new(None);
static LAST: Mutex<Option<(bool, bool)>> = Mutex::new(None);
static STATUS: Mutex<String> = Mutex::new(String::new());
pub fn register(window: &Arc<Window>) {
    *WINDOW.lock().unwrap() = Some(Arc::downgrade(window));
    *LAST.lock().unwrap() = None;
}
pub fn status() -> String {
    STATUS.lock().unwrap().clone()
}
pub fn configure(s: &GlassSettings) {
    let blur = s.desktop_blur;
    #[cfg(target_os = "windows")]
    let blur = blur && !crate::windows_key_probe::is_live_resize();
    let key = (s.enabled && s.desktop, blur);
    let mut last = LAST.lock().unwrap();
    if *last == Some(key) {
        return;
    }
    let window = WINDOW.lock().unwrap().as_ref().and_then(Weak::upgrade);
    let Some(window) = window else { return };
    if last.is_none_or(|old| old.0 != key.0) {
        window.set_transparent(key.0);
    }
    let result = native(&window, key.0, key.0 && key.1);
    *STATUS.lock().unwrap() = match result {
        Ok(()) => crate::t!("glass.native_available"),
        Err(e) => e,
    };
    *last = Some(key);
}
#[cfg(target_os = "windows")]
fn native(window: &Window, desktop: bool, blur: bool) -> Result<(), String> {
    use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};
    let handle = window.window_handle().map_err(|e| e.to_string())?;
    let RawWindowHandle::Win32(h) = handle.as_raw() else {
        return Err(crate::t!("glass.native_unavailable"));
    };
    let hwnd = h.hwnd.get();
    static SYSTEM_BACKDROP: std::sync::atomic::AtomicBool =
        std::sync::atomic::AtomicBool::new(false);
    #[link(name = "dwmapi")]
    unsafe extern "system" {
        fn DwmSetWindowAttribute(hwnd: isize, attribute: u32, value: *const u32, size: u32) -> i32;
    }
    // Windows 11 22H2 and newer no longer reliably render the legacy blur
    // accent. Prefer its documented desktop Acrylic backdrop instead.
    if blur {
        windows_alpha(hwnd, desktop)?;
        let material = 3u32; // DWMSBT_TRANSIENTWINDOW
        if unsafe { DwmSetWindowAttribute(hwnd, 38, &material, 4) } >= 0 {
            SYSTEM_BACKDROP.store(true, std::sync::atomic::Ordering::Relaxed);
            return Ok(());
        }
    } else if SYSTEM_BACKDROP.swap(false, std::sync::atomic::Ordering::Relaxed) {
        let material = 1u32; // DWMSBT_NONE
        let cleared = unsafe { DwmSetWindowAttribute(hwnd, 38, &material, 4) };
        windows_alpha(hwnd, desktop)?;
        return if cleared >= 0 {
            Ok(())
        } else {
            Err(crate::t!("glass.native_unavailable"))
        };
    }
    static ACCENT_ACTIVE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    let clear_accent = ACCENT_ACTIVE.load(std::sync::atomic::Ordering::Relaxed);
    if !blur && !clear_accent {
        return windows_alpha(hwnd, desktop);
    }
    #[repr(C)]
    struct Accent {
        state: i32,
        flags: u32,
        color: u32,
        animation: u32,
    }
    #[repr(C)]
    struct Data {
        attribute: i32,
        data: *mut std::ffi::c_void,
        size: usize,
    }
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetModuleHandleW(name: *const u16) -> isize;
        fn GetProcAddress(module: isize, name: *const std::ffi::c_char) -> *const std::ffi::c_void;
    }
    // Prepare zero-alpha backing before setting the accent. Reapplying the DWM
    // alpha region afterwards can replace the accent's desktop backdrop.
    // Older Windows uses the legacy Acrylic policy. Its zero-alpha tint is
    // invalid, so keep the minimum nonzero alpha. Radius is OS-managed.
    let mut accent = Accent {
        state: if blur { 4 } else { 0 },
        flags: 0,
        color: 0x01000000,
        animation: 0,
    };
    let mut data = Data {
        attribute: 19,
        data: (&mut accent as *mut Accent).cast(),
        size: std::mem::size_of::<Accent>(),
    };
    let library: Vec<u16> = "user32.dll\0".encode_utf16().collect();
    let entry = unsafe {
        GetProcAddress(
            GetModuleHandleW(library.as_ptr()),
            c"SetWindowCompositionAttribute".as_ptr(),
        )
    };
    if entry.is_null() {
        return Err(crate::t!("glass.native_unavailable"));
    }
    let apply: unsafe extern "system" fn(isize, *mut Data) -> i32 =
        unsafe { std::mem::transmute(entry) };
    let ok = unsafe { apply(hwnd, &mut data) };
    if ok == 0 {
        Err(crate::t!("glass.native_unavailable"))
    } else {
        ACCENT_ACTIVE.store(blur, std::sync::atomic::Ordering::Relaxed);
        if blur {
            Ok(())
        } else {
            windows_alpha(hwnd, desktop)
        }
    }
}

/// Clear the HWND redirection surface as well as the GPU backbuffer. A null
/// class brush leaves its alpha undefined, even when Vulkan advertises alpha.
#[cfg(target_os = "windows")]
fn windows_alpha(hwnd: isize, enabled: bool) -> Result<(), String> {
    if hwnd == 0 {
        return Err(crate::t!("glass.native_unavailable"));
    }
    use winapi::shared::windef::{HWND, RECT};
    use winapi::um::dwmapi::{
        DWM_BB_BLURREGION, DWM_BB_ENABLE, DWM_BLURBEHIND, DwmEnableBlurBehindWindow,
    };
    use winapi::um::winuser::{
        GCLP_HBRBACKGROUND, GetClientRect, GetDC, ReleaseDC, SetClassLongPtrW,
    };
    #[link(name = "gdi32")]
    unsafe extern "system" {
        fn GetStockObject(index: i32) -> isize;
        fn PatBlt(dc: isize, x: i32, y: i32, width: i32, height: i32, op: u32) -> i32;
        fn CreateRectRgn(left: i32, top: i32, right: i32, bottom: i32) -> isize;
        fn DeleteObject(object: isize) -> i32;
    }
    let hwnd = hwnd as HWND;
    unsafe {
        if enabled {
            // BLACK_BRUSH writes zero alpha. It is a stock object, never delete it.
            SetClassLongPtrW(hwnd, GCLP_HBRBACKGROUND, GetStockObject(4));
            let dc = GetDC(hwnd);
            if dc.is_null() {
                return Err(crate::t!("glass.native_unavailable"));
            }
            let mut rect: RECT = std::mem::zeroed();
            let bounds = GetClientRect(hwnd, &mut rect);
            let cleared = bounds != 0
                && PatBlt(
                    dc as isize,
                    0,
                    0,
                    rect.right - rect.left,
                    rect.bottom - rect.top,
                    0x00000042,
                ) != 0;
            ReleaseDC(hwnd, dc);
            if !cleared {
                return Err(crate::t!("glass.native_unavailable"));
            }
        }
        let region = if enabled {
            CreateRectRgn(0, 0, -1, -1)
        } else {
            0
        };
        if enabled && region == 0 {
            return Err(crate::t!("glass.native_unavailable"));
        }
        let bb = DWM_BLURBEHIND {
            dwFlags: DWM_BB_ENABLE | DWM_BB_BLURREGION,
            fEnable: i32::from(enabled),
            hRgnBlur: region as _,
            fTransitionOnMaximized: 0,
        };
        let result = DwmEnableBlurBehindWindow(hwnd, &bb);
        if region != 0 {
            DeleteObject(region);
        }
        if result < 0 {
            Err(crate::t!("glass.native_unavailable"))
        } else {
            Ok(())
        }
    }
}
#[cfg(target_os = "macos")]
fn native(window: &Window, _desktop: bool, blur: bool) -> Result<(), String> {
    use std::ffi::{c_char, c_void};
    use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};
    #[link(name = "objc")]
    unsafe extern "C" {
        fn objc_getClass(name: *const c_char) -> *mut c_void;
        fn sel_registerName(name: *const c_char) -> *mut c_void;
        fn objc_msgSend();
        #[cfg(target_arch = "x86_64")]
        fn objc_msgSend_stret();
    }
    #[link(name = "AppKit", kind = "framework")]
    unsafe extern "C" {}
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct Point {
        x: f64,
        y: f64,
    }
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct Size {
        w: f64,
        h: f64,
    }
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct Bounds {
        origin: Point,
        size: Size,
    }
    static VIEW: Mutex<usize> = Mutex::new(0);
    unsafe fn selector(s: &'static [u8]) -> *mut c_void {
        unsafe { sel_registerName(s.as_ptr().cast()) }
    }
    let handle = window.window_handle().map_err(|e| e.to_string())?;
    let RawWindowHandle::AppKit(h) = handle.as_raw() else {
        return Err(crate::t!("glass.native_unavailable"));
    };
    unsafe {
        let msg: unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void =
            std::mem::transmute(objc_msgSend as *const ());
        let mut view = VIEW.lock().unwrap();
        if !blur {
            if *view != 0 {
                msg(*view as _, selector(b"removeFromSuperview\0"));
                msg(*view as _, selector(b"release\0"));
                *view = 0;
            }
            return Ok(());
        }
        if *view != 0 {
            return Ok(());
        }
        let parent = msg(h.ns_view.as_ptr(), selector(b"superview\0"));
        if parent.is_null() {
            return Err(crate::t!("glass.native_unavailable"));
        }
        let class = objc_getClass(c"NSVisualEffectView".as_ptr());
        if class.is_null() {
            return Err(crate::t!("glass.native_unavailable"));
        }
        let alloc = msg(class, selector(b"alloc\0"));
        #[cfg(not(target_arch = "x86_64"))]
        let bounds: unsafe extern "C" fn(*mut c_void, *mut c_void) -> Bounds =
            std::mem::transmute(objc_msgSend as *const ());
        #[cfg(not(target_arch = "x86_64"))]
        let rect = bounds(h.ns_view.as_ptr(), selector(b"frame\0"));
        #[cfg(target_arch = "x86_64")]
        let rect = {
            let mut rect = std::mem::MaybeUninit::<Bounds>::uninit();
            let bounds: unsafe extern "C" fn(*mut Bounds, *mut c_void, *mut c_void) =
                std::mem::transmute(objc_msgSend_stret as *const ());
            bounds(rect.as_mut_ptr(), h.ns_view.as_ptr(), selector(b"frame\0"));
            rect.assume_init()
        };
        let init: unsafe extern "C" fn(*mut c_void, *mut c_void, Bounds) -> *mut c_void =
            std::mem::transmute(objc_msgSend as *const ());
        let effect = init(alloc, selector(b"initWithFrame:\0"), rect);
        let set: unsafe extern "C" fn(*mut c_void, *mut c_void, isize) =
            std::mem::transmute(objc_msgSend as *const ());
        set(effect, selector(b"setBlendingMode:\0"), 0);
        set(effect, selector(b"setMaterial:\0"), 13);
        set(effect, selector(b"setState:\0"), 1);
        set(effect, selector(b"setAutoresizingMask:\0"), 18);
        let add: unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void, isize, *mut c_void) =
            std::mem::transmute(objc_msgSend as *const ());
        add(
            parent,
            selector(b"addSubview:positioned:relativeTo:\0"),
            effect,
            -1,
            h.ns_view.as_ptr(),
        );
        *view = effect as usize;
    }
    Ok(())
}
#[cfg(not(any(target_os = "windows", target_os = "macos")))]
fn native(_window: &Window, _desktop: bool, _blur: bool) -> Result<(), String> {
    Err(crate::t!("glass.compositor"))
}
