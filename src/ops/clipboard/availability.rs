//! Menu availability must never decode clipboard pixels on the UI thread.
#[cfg(not(target_arch = "wasm32"))]
use std::sync::{Arc, Mutex, OnceLock};
#[cfg(not(target_arch = "wasm32"))]
use std::time::{Duration, Instant};

#[cfg(not(target_arch = "wasm32"))]
#[derive(Default)]
struct Probe {
    sequence: Option<u32>,
    checked: Option<Instant>,
    available: bool,
    running: bool,
    generation: u64,
}

#[cfg(not(target_arch = "wasm32"))]
impl Probe {
    fn begin(&mut self, sequence: Option<u32>, now: Instant) -> Option<u64> {
        if self.sequence != sequence {
            self.sequence = sequence;
            self.checked = None;
            self.available = false;
            self.generation += 1;
        }
        let fresh = self.checked.is_some_and(|at| {
            sequence.is_some() || now.duration_since(at) < Duration::from_secs(2)
        });
        if self.running || fresh {
            return None;
        }
        self.running = true;
        Some(self.generation)
    }

    fn finish(&mut self, generation: u64, available: bool) {
        self.running = false;
        if self.generation == generation {
            self.available = available;
            self.checked = Some(Instant::now());
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub(super) fn has_system_image(ctx: &egui::Context) -> bool {
    // Native format metadata is cheap and does not open or copy clipboard data.
    #[cfg(target_os = "windows")]
    {
        use winapi::um::winuser::{CF_BITMAP, CF_DIB, CF_DIBV5, IsClipboardFormatAvailable};
        if unsafe {
            IsClipboardFormatAvailable(CF_DIB) != 0
                || IsClipboardFormatAvailable(CF_DIBV5) != 0
                || IsClipboardFormatAvailable(CF_BITMAP) != 0
        } {
            return true;
        }
    }
    // Keep file-list/text/Wayland compatibility, but probe once in the background.
    static CACHE: OnceLock<Arc<Mutex<Probe>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Arc::new(Mutex::new(Probe::default())));
    let sequence = super::clipboard_sequence();
    let mut state = cache.lock().unwrap_or_else(|e| e.into_inner());
    let generation = state.begin(sequence, Instant::now());
    let available = state.available;
    drop(state);
    if let Some(generation) = generation {
        let cache = Arc::clone(cache);
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let available = super::get_from_system_clipboard().is_some();
            let mut state = cache.lock().unwrap_or_else(|e| e.into_inner());
            // Discard a probe if the OS clipboard changed while it was reading.
            if super::clipboard_sequence() != sequence {
                state.generation += 1;
            }
            state.finish(generation, available);
            drop(state);
            ctx.request_repaint();
        });
    }
    #[cfg(not(target_os = "windows"))]
    ctx.request_repaint_after(Duration::from_secs(2));
    available
}

#[cfg(target_arch = "wasm32")]
pub(super) fn has_system_image(_: &egui::Context) -> bool {
    super::web_clipboard::has_pasted_image()
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    #[test]
    fn clipboard_probe_deduplicates_and_caches() {
        let mut state = Probe::default();
        let now = Instant::now();
        let generation = state.begin(Some(1), now).unwrap();
        assert_eq!(state.begin(Some(1), now), None);
        state.finish(generation, true);
        assert!(state.available);
        assert_eq!(state.begin(Some(1), now + Duration::from_secs(60)), None);
        assert!(state.begin(Some(2), now).is_some());
        assert!(!state.available);
    }

    #[test]
    fn clipboard_probe_discards_stale_result() {
        let mut state = Probe::default();
        let now = Instant::now();
        let generation = state.begin(Some(1), now).unwrap();
        assert_eq!(state.begin(Some(2), now), None);
        state.finish(generation, true);
        assert!(!state.available);
        assert!(state.begin(Some(2), now).is_some());
    }

    #[test]
    fn clipboard_probe_refreshes_without_native_sequence() {
        let mut state = Probe::default();
        let now = Instant::now();
        let generation = state.begin(None, now).unwrap();
        state.finish(generation, false);
        assert_eq!(state.begin(None, Instant::now()), None);
        assert!(
            state
                .begin(None, Instant::now() + Duration::from_secs(3))
                .is_some()
        );
    }
}
