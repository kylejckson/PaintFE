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
    const VK_MENU: usize = 0x12;
    const VK_LMENU: usize = 0xA4;
    const VK_RMENU: usize = 0xA5;
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

    // Store the press serial and its modifiers together. A quick tap may be
    // released before egui processes a frame, so current key state is insufficient.
    static VK_PRESS_STATE: LazyLock<Vec<AtomicU64>> =
        LazyLock::new(|| (0..=255).map(|_| AtomicU64::new(0)).collect());

    static INPUT_EVENTS: std::sync::Mutex<Vec<egui::Event>> = std::sync::Mutex::new(Vec::new());
    static PENDING_SURROGATE: std::sync::Mutex<Option<u16>> = std::sync::Mutex::new(None);

    /// Preserve presses whose WM_CHAR completion is consumed by our hook.
    /// Never dispatch a native window message twice.
    pub fn observe_native_message(message: u32, wparam: usize, lparam: isize) {
        use winapi::um::winuser::GetKeyState;
        if message == 0x0102 && (0x20..=0xFFFF).contains(&wparam) && wparam != 0x7F {
            let unit = wparam as u16;
            let mut pending = PENDING_SURROGATE.lock().unwrap_or_else(|e| e.into_inner());
            if (0xD800..=0xDBFF).contains(&unit) {
                *pending = Some(unit);
            } else {
                let units = pending
                    .take()
                    .map_or_else(|| vec![unit], |first| vec![first, unit]);
                let text = String::from_utf16_lossy(&units);
                INPUT_EVENTS
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .push(egui::Event::Text(text));
            }
        }
        if matches!(message, WM_KEYDOWN | WM_KEYUP | WM_SYSKEYDOWN | WM_SYSKEYUP) {
            let down = |vk| unsafe { (GetKeyState(vk) as u16 & 0x8000) != 0 };
            for vk in [0x10, 0xA0, 0xA1, 0x11, 0xA2, 0xA3, 0x12, 0xA4, 0xA5] {
                VK_DOWN[vk].store(down(vk as i32), Ordering::Relaxed);
            }
            CTRL_DOWN.store(down(0x11), Ordering::Relaxed);
            let modifiers = egui::Modifiers {
                ctrl: down(0x11),
                shift: down(0x10),
                alt: down(0x12),
                command: down(0x11),
                mac_cmd: false,
            };
            let key = match wparam {
                0x08 => Some(egui::Key::Backspace),
                0x09 => Some(egui::Key::Tab),
                0x0D => Some(egui::Key::Enter),
                0x1B => Some(egui::Key::Escape),
                0x20 => Some(egui::Key::Space),
                0x21 => Some(egui::Key::PageUp),
                0x22 => Some(egui::Key::PageDown),
                0x23 => Some(egui::Key::End),
                0x24 => Some(egui::Key::Home),
                0x25 => Some(egui::Key::ArrowLeft),
                0x26 => Some(egui::Key::ArrowUp),
                0x27 => Some(egui::Key::ArrowRight),
                0x28 => Some(egui::Key::ArrowDown),
                0x2D => Some(egui::Key::Insert),
                0x2E => Some(egui::Key::Delete),
                0x30..=0x39 => egui::Key::from_name(&char::from(wparam as u8).to_string()),
                0x41..=0x5A => egui::Key::from_name(&char::from(wparam as u8).to_string()),
                _ => None,
            };
            // Capture matching releases even when Ctrl has already been released.
            if let Some(key) = key {
                INPUT_EVENTS
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .push(egui::Event::Key {
                        key,
                        physical_key: Some(key),
                        pressed: matches!(message, WM_KEYDOWN | WM_SYSKEYDOWN),
                        repeat: lparam & (1 << 30) != 0
                            && matches!(message, WM_KEYDOWN | WM_SYSKEYDOWN),
                        modifiers,
                    });
            }
        }
        observe_windows_message(message, wparam);
    }

    pub fn bridge_raw_input(input: &mut egui::RawInput) {
        let events = std::mem::take(&mut *INPUT_EVENTS.lock().unwrap_or_else(|e| e.into_inner()));
        if !input.focused {
            observe_windows_message(0x0008, 0);
            *PENDING_SURROGATE.lock().unwrap_or_else(|e| e.into_inner()) = None;
            return;
        }
        merge_native_events(input, events);
    }

    fn merge_native_events(input: &mut egui::RawInput, events: Vec<egui::Event>) {
        input
            .events
            .retain(|event| !matches!(event, egui::Event::Key { key, .. } if bridged_key(*key)));
        // Match individual text events so IME commits that do not use WM_CHAR
        // remain intact. Text and editing keys must retain their native order.
        for native in &events {
            if let egui::Event::Text(text) = native
                && let Some(index) = input
                    .events
                    .iter()
                    .position(|event| matches!(event, egui::Event::Text(other) if other == text))
            {
                input.events.remove(index);
            }
        }
        for event in &events {
            if let egui::Event::Key {
                key,
                pressed: true,
                repeat: false,
                modifiers,
                ..
            } = event
                && modifiers.ctrl
                && !modifiers.alt
            {
                match key {
                    egui::Key::C
                        if !input.events.iter().any(|e| matches!(e, egui::Event::Copy)) =>
                    {
                        input.events.push(egui::Event::Copy)
                    }
                    egui::Key::X if !input.events.iter().any(|e| matches!(e, egui::Event::Cut)) => {
                        input.events.push(egui::Event::Cut)
                    }
                    egui::Key::V
                        if !input
                            .events
                            .iter()
                            .any(|e| matches!(e, egui::Event::Paste(_))) =>
                    {
                        if let Ok(mut clipboard) = arboard::Clipboard::new()
                            && let Ok(text) = clipboard.get_text()
                        {
                            input.events.push(egui::Event::Paste(text));
                        }
                    }
                    _ => {}
                }
            }
        }
        input.events.extend(events);
    }

    fn bridged_key(key: egui::Key) -> bool {
        matches!(
            key,
            egui::Key::Backspace
                | egui::Key::Tab
                | egui::Key::Enter
                | egui::Key::Escape
                | egui::Key::Space
                | egui::Key::PageUp
                | egui::Key::PageDown
                | egui::Key::End
                | egui::Key::Home
                | egui::Key::ArrowLeft
                | egui::Key::ArrowUp
                | egui::Key::ArrowRight
                | egui::Key::ArrowDown
                | egui::Key::Insert
                | egui::Key::Delete
        ) || (key.name().len() == 1 && key.name().as_bytes()[0].is_ascii_alphanumeric())
    }

    #[cfg(test)]
    mod bridge_tests {
        use super::*;

        fn key(key: egui::Key, pressed: bool) -> egui::Event {
            egui::Event::Key {
                key,
                physical_key: Some(key),
                pressed,
                repeat: false,
                modifiers: egui::Modifiers::CTRL,
            }
        }

        #[test]
        fn backend_duplicates_are_replaced_and_quick_taps_keep_order() {
            let native = vec![
                key(egui::Key::Z, true),
                key(egui::Key::Z, false),
                key(egui::Key::Z, true),
                key(egui::Key::Z, false),
            ];
            let mut input = egui::RawInput {
                events: vec![key(egui::Key::Z, true), key(egui::Key::Z, false)],
                ..Default::default()
            };
            merge_native_events(&mut input, native.clone());
            assert_eq!(input.events, native);
        }

        #[test]
        fn text_and_backspace_keep_order_without_losing_ime_commits() {
            let native = vec![
                key(egui::Key::Backspace, true),
                egui::Event::Text("a".into()),
                key(egui::Key::Backspace, false),
            ];
            let mut input = egui::RawInput {
                events: vec![
                    egui::Event::Text("a".into()),
                    egui::Event::Text("日本語".into()),
                ],
                ..Default::default()
            };
            merge_native_events(&mut input, native.clone());
            assert_eq!(input.events[0], egui::Event::Text("日本語".into()));
            assert_eq!(&input.events[1..], native.as_slice());
        }
    }

    pub fn vk_press_count(vk: usize) -> u64 {
        vk_press_state(vk).0
    }

    pub fn vk_press_state(vk: usize) -> (u64, bool, bool, bool) {
        let state = VK_PRESS_STATE
            .get(vk)
            .map_or(0, |state| state.load(Ordering::Relaxed));
        (state >> 3, state & 1 != 0, state & 2 != 0, state & 4 != 0)
    }

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
                if wparam <= 255 && !VK_DOWN[wparam].swap(true, Ordering::Relaxed) {
                    let any_down = |keys: [usize; 3]| {
                        keys.into_iter()
                            .any(|vk| VK_DOWN[vk].load(Ordering::Relaxed))
                    };
                    let modifiers = u64::from(any_down([VK_CONTROL, VK_LCONTROL, VK_RCONTROL]))
                        | (u64::from(any_down([0x10, 0xA0, 0xA1])) << 1)
                        | (u64::from(any_down([VK_MENU, VK_LMENU, VK_RMENU])) << 2);
                    let state = &VK_PRESS_STATE[wparam];
                    let previous = state.load(Ordering::Relaxed);
                    state.store(
                        (previous.wrapping_add(8) & !7) | modifiers,
                        Ordering::Relaxed,
                    );
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
            0x0008 => {
                // Focus loss may prevent winit receiving the matching releases.
                for down in VK_DOWN.iter() {
                    down.store(false, Ordering::Relaxed);
                }
                for down in [
                    &CTRL_DOWN,
                    &C_DOWN,
                    &X_DOWN,
                    &V_DOWN,
                    &ENTER_DOWN,
                    &ESCAPE_DOWN,
                ] {
                    down.store(false, Ordering::Relaxed);
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

    /// Poll physical Ctrl state directly. This remains reliable when a pointer
    /// capture prevents normal modifier/key messages from reaching egui.
    pub fn ctrl_down_realtime() -> bool {
        use winapi::um::winuser::GetAsyncKeyState;
        unsafe {
            (GetAsyncKeyState(VK_CONTROL as i32) as u16 & 0x8000) != 0
                || (GetAsyncKeyState(VK_LCONTROL as i32) as u16 & 0x8000) != 0
                || (GetAsyncKeyState(VK_RCONTROL as i32) as u16 & 0x8000) != 0
        }
    }

    pub fn alt_down_realtime() -> bool {
        use winapi::um::winuser::GetAsyncKeyState;
        unsafe {
            (GetAsyncKeyState(VK_MENU as i32) as u16 & 0x8000) != 0
                || (GetAsyncKeyState(VK_LMENU as i32) as u16 & 0x8000) != 0
                || (GetAsyncKeyState(VK_RMENU as i32) as u16 & 0x8000) != 0
        }
    }

    pub fn enter_down_realtime() -> bool {
        use winapi::um::winuser::GetAsyncKeyState;
        unsafe { (GetAsyncKeyState(VK_RETURN as i32) as u16 & 0x8000) != 0 }
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

    pub fn vk_press_count(_vk: usize) -> u64 {
        0
    }

    pub fn vk_press_state(_vk: usize) -> (u64, bool, bool, bool) {
        (0, false, false, false)
    }

    pub fn is_vk_down(_vk: usize) -> bool {
        false
    }

    pub fn snapshot() -> KeyProbeSnapshot {
        KeyProbeSnapshot::default()
    }

    pub fn is_live_resize() -> bool {
        false
    }

    pub fn ctrl_down_realtime() -> bool {
        false
    }

    pub fn alt_down_realtime() -> bool {
        false
    }

    pub fn enter_down_realtime() -> bool {
        false
    }
}

pub use imp::{
    KeyProbeSnapshot, alt_down_realtime, ctrl_down_realtime, enter_down_realtime, is_live_resize,
    is_vk_down, observe_windows_message, snapshot, vk_press_count, vk_press_state,
};

#[cfg(target_os = "windows")]
pub use imp::{bridge_raw_input, observe_native_message};
