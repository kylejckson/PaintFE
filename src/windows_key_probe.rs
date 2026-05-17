#[cfg(target_os = "windows")]
mod imp {
    use std::sync::LazyLock;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

    const WM_KEYDOWN: u32 = 0x0100;
    const WM_KEYUP: u32 = 0x0101;
    const WM_SYSKEYDOWN: u32 = 0x0104;
    const WM_SYSKEYUP: u32 = 0x0105;
    const WM_SIZING: u32 = 0x0214;
    const WM_ENTERSIZEMOVE: u32 = 0x0231;
    const WM_EXITSIZEMOVE: u32 = 0x0232;

    const VK_CONTROL: usize = 0x11;
    const VK_LCONTROL: usize = 0xA2;
    const VK_RCONTROL: usize = 0xA3;
    const VK_C: usize = 0x43;
    const VK_X: usize = 0x58;
    const VK_V: usize = 0x56;
    const VK_RETURN: usize = 0x0D;
    const VK_ESCAPE: usize = 0x1B;

    static CTRL_DOWN: AtomicBool = AtomicBool::new(false);
    static C_DOWN: AtomicBool = AtomicBool::new(false);
    static X_DOWN: AtomicBool = AtomicBool::new(false);
    static V_DOWN: AtomicBool = AtomicBool::new(false);
    static ENTER_DOWN: AtomicBool = AtomicBool::new(false);
    static ESCAPE_DOWN: AtomicBool = AtomicBool::new(false);
    static SIZE_MOVE_ACTIVE: AtomicBool = AtomicBool::new(false);
    static LIVE_RESIZE_ACTIVE: AtomicBool = AtomicBool::new(false);

    static C_PRESS_COUNT: AtomicU64 = AtomicU64::new(0);
    static X_PRESS_COUNT: AtomicU64 = AtomicU64::new(0);
    static V_PRESS_COUNT: AtomicU64 = AtomicU64::new(0);
    static ENTER_PRESS_COUNT: AtomicU64 = AtomicU64::new(0);
    static ESCAPE_PRESS_COUNT: AtomicU64 = AtomicU64::new(0);

    // Generic VK tracking (0..=255) so fallback shortcut detection can cover
    // all Ctrl+letter combinations, not only C/X/V.
    static VK_DOWN: LazyLock<Vec<AtomicBool>> =
        LazyLock::new(|| (0..=255).map(|_| AtomicBool::new(false)).collect());

    #[derive(Clone, Copy, Debug, Default)]
    pub struct KeyProbeSnapshot {
        pub ctrl_down: bool,
        pub c_down: bool,
        pub x_down: bool,
        pub v_down: bool,
        pub enter_down: bool,
        pub escape_down: bool,
        pub c_press_count: u64,
        pub x_press_count: u64,
        pub v_press_count: u64,
        pub enter_press_count: u64,
        pub escape_press_count: u64,
    }

    pub fn observe_windows_message(message: u32, wparam: usize) {
        match message {
            WM_KEYDOWN | WM_SYSKEYDOWN => {
                if wparam <= 255 {
                    VK_DOWN[wparam].store(true, Ordering::Relaxed);
                }
                match wparam {
                    VK_CONTROL | VK_LCONTROL | VK_RCONTROL => {
                        CTRL_DOWN.store(true, Ordering::Relaxed);
                    }
                    VK_C if !C_DOWN.swap(true, Ordering::Relaxed) => {
                        C_PRESS_COUNT.fetch_add(1, Ordering::Relaxed);
                    }
                    VK_X if !X_DOWN.swap(true, Ordering::Relaxed) => {
                        X_PRESS_COUNT.fetch_add(1, Ordering::Relaxed);
                    }
                    VK_V if !V_DOWN.swap(true, Ordering::Relaxed) => {
                        V_PRESS_COUNT.fetch_add(1, Ordering::Relaxed);
                    }
                    VK_RETURN if !ENTER_DOWN.swap(true, Ordering::Relaxed) => {
                        ENTER_PRESS_COUNT.fetch_add(1, Ordering::Relaxed);
                    }
                    VK_ESCAPE if !ESCAPE_DOWN.swap(true, Ordering::Relaxed) => {
                        ESCAPE_PRESS_COUNT.fetch_add(1, Ordering::Relaxed);
                    }
                    _ => {}
                }
            }
            WM_KEYUP | WM_SYSKEYUP => {
                if wparam <= 255 {
                    VK_DOWN[wparam].store(false, Ordering::Relaxed);
                }
                match wparam {
                    VK_CONTROL | VK_LCONTROL | VK_RCONTROL => {
                        CTRL_DOWN.store(false, Ordering::Relaxed);
                    }
                    VK_C => {
                        C_DOWN.store(false, Ordering::Relaxed);
                    }
                    VK_X => {
                        X_DOWN.store(false, Ordering::Relaxed);
                    }
                    VK_V => {
                        V_DOWN.store(false, Ordering::Relaxed);
                    }
                    VK_RETURN => {
                        ENTER_DOWN.store(false, Ordering::Relaxed);
                    }
                    VK_ESCAPE => {
                        ESCAPE_DOWN.store(false, Ordering::Relaxed);
                    }
                    _ => {}
                }
            }
            WM_ENTERSIZEMOVE => {
                SIZE_MOVE_ACTIVE.store(true, Ordering::Relaxed);
                LIVE_RESIZE_ACTIVE.store(false, Ordering::Relaxed);
            }
            WM_SIZING => {
                if SIZE_MOVE_ACTIVE.load(Ordering::Relaxed) {
                    LIVE_RESIZE_ACTIVE.store(true, Ordering::Relaxed);
                }
            }
            WM_EXITSIZEMOVE => {
                SIZE_MOVE_ACTIVE.store(false, Ordering::Relaxed);
                LIVE_RESIZE_ACTIVE.store(false, Ordering::Relaxed);
            }
            _ => {}
        }
    }

    pub fn is_vk_down(vk: usize) -> bool {
        if vk <= 255 {
            return VK_DOWN[vk].load(Ordering::Relaxed);
        }
        false
    }

    pub fn snapshot() -> KeyProbeSnapshot {
        KeyProbeSnapshot {
            ctrl_down: CTRL_DOWN.load(Ordering::Relaxed),
            c_down: C_DOWN.load(Ordering::Relaxed),
            x_down: X_DOWN.load(Ordering::Relaxed),
            v_down: V_DOWN.load(Ordering::Relaxed),
            enter_down: ENTER_DOWN.load(Ordering::Relaxed),
            escape_down: ESCAPE_DOWN.load(Ordering::Relaxed),
            c_press_count: C_PRESS_COUNT.load(Ordering::Relaxed),
            x_press_count: X_PRESS_COUNT.load(Ordering::Relaxed),
            v_press_count: V_PRESS_COUNT.load(Ordering::Relaxed),
            enter_press_count: ENTER_PRESS_COUNT.load(Ordering::Relaxed),
            escape_press_count: ESCAPE_PRESS_COUNT.load(Ordering::Relaxed),
        }
    }

    pub fn is_live_resize() -> bool {
        LIVE_RESIZE_ACTIVE.load(Ordering::Relaxed)
    }
}

#[cfg(not(target_os = "windows"))]
mod imp {
    #[derive(Clone, Copy, Debug, Default)]
    pub struct KeyProbeSnapshot {
        pub ctrl_down: bool,
        pub c_down: bool,
        pub x_down: bool,
        pub v_down: bool,
        pub enter_down: bool,
        pub escape_down: bool,
        pub c_press_count: u64,
        pub x_press_count: u64,
        pub v_press_count: u64,
        pub enter_press_count: u64,
        pub escape_press_count: u64,
    }

    pub fn observe_windows_message(_message: u32, _wparam: usize) {}

    pub fn is_vk_down(_vk: usize) -> bool {
        false
    }

    pub fn snapshot() -> KeyProbeSnapshot {
        KeyProbeSnapshot::default()
    }

    pub fn is_live_resize() -> bool {
        false
    }
}

pub use imp::{KeyProbeSnapshot, is_live_resize, is_vk_down, observe_windows_message, snapshot};
