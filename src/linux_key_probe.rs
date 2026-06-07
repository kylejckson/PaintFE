#[cfg(target_os = "linux")]
mod imp {
    use std::sync::{Mutex, OnceLock};
    use x11_dl::keysym;
    use x11_dl::xlib;

    #[derive(Clone, Copy, Debug, Default)]
    pub struct KeyProbeSnapshot {
        pub ctrl_down: bool,
        pub c_down: bool,
        pub x_down: bool,
        pub v_down: bool,
    }

    struct ProbeState {
        xlib: xlib::Xlib,
        display: *mut xlib::Display,
        ctrl_l: u8,
        ctrl_r: u8,
        c: u8,
        x: u8,
        v: u8,
    }

    // Accessed under a Mutex only.
    unsafe impl Send for ProbeState {}

    static PROBE: OnceLock<Mutex<Option<ProbeState>>> = OnceLock::new();

    fn probe_mutex() -> &'static Mutex<Option<ProbeState>> {
        PROBE.get_or_init(|| Mutex::new(None))
    }

    fn ensure_probe_locked(slot: &mut Option<ProbeState>) {
        if slot.is_some() {
            return;
        }

        let Ok(xlib) = xlib::Xlib::open() else {
            return;
        };

        // Skip X11 connection attempt if DISPLAY is not set — avoids printing
        // "No protocol specified" / "Cannot open display" errors on pure Wayland.
        if std::env::var_os("DISPLAY").is_none() {
            return;
        }

        // SAFETY: null display name means use DISPLAY env var.
        let display = unsafe { (xlib.XOpenDisplay)(std::ptr::null()) };
        if display.is_null() {
            return;
        }

        // SAFETY: display is valid while kept open.
        let ctrl_l = unsafe { (xlib.XKeysymToKeycode)(display, keysym::XK_Control_L as _) };
        // SAFETY: display is valid while kept open.
        let ctrl_r = unsafe { (xlib.XKeysymToKeycode)(display, keysym::XK_Control_R as _) };
        // SAFETY: display is valid while kept open.
        let c = unsafe { (xlib.XKeysymToKeycode)(display, keysym::XK_c as _) };
        // SAFETY: display is valid while kept open.
        let x = unsafe { (xlib.XKeysymToKeycode)(display, keysym::XK_x as _) };
        // SAFETY: display is valid while kept open.
        let v = unsafe { (xlib.XKeysymToKeycode)(display, keysym::XK_v as _) };

        *slot = Some(ProbeState {
            xlib,
            display,
            ctrl_l,
            ctrl_r,
            c,
            x,
            v,
        });
    }

    fn key_down(mask: &[i8; 32], keycode: u8) -> bool {
        if keycode == 0 {
            return false;
        }
        let idx = (keycode / 8) as usize;
        let bit = 1u8 << (keycode % 8);
        (mask[idx] as u8 & bit) != 0
    }

    pub fn snapshot() -> KeyProbeSnapshot {
        // Only meaningful when running X11 backend.
        if !std::env::var("WINIT_UNIX_BACKEND")
            .map(|v| v.eq_ignore_ascii_case("x11"))
            .unwrap_or(false)
        {
            return KeyProbeSnapshot::default();
        }

        let mutex = probe_mutex();
        let Ok(mut guard) = mutex.lock() else {
            return KeyProbeSnapshot::default();
        };

        ensure_probe_locked(&mut guard);
        let Some(state) = guard.as_ref() else {
            return KeyProbeSnapshot::default();
        };

        let mut keys: [i8; 32] = [0; 32];
        // SAFETY: display is valid for the lifetime of the probe state, and
        // keys points to a writable 32-byte buffer as required by XQueryKeymap.
        unsafe {
            (state.xlib.XQueryKeymap)(state.display, keys.as_mut_ptr() as *mut _);
        }

        let ctrl_down = key_down(&keys, state.ctrl_l) || key_down(&keys, state.ctrl_r);
        KeyProbeSnapshot {
            ctrl_down,
            c_down: key_down(&keys, state.c),
            x_down: key_down(&keys, state.x),
            v_down: key_down(&keys, state.v),
        }
    }
}

#[cfg(not(target_os = "linux"))]
mod imp {
    #[derive(Clone, Copy, Debug, Default)]
    pub struct KeyProbeSnapshot {
        pub ctrl_down: bool,
        pub c_down: bool,
        pub x_down: bool,
        pub v_down: bool,
    }

    pub fn snapshot() -> KeyProbeSnapshot {
        KeyProbeSnapshot::default()
    }
}

pub use imp::{KeyProbeSnapshot, snapshot};
