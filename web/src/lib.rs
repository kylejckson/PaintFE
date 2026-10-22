// PaintFE Web entry point — instantiates the real desktop `PaintFEApp`
// (from the `paintfe` lib crate) inside `eframe::WebRunner`.
//
// Kept as a separate crate (rather than adding wasm entry code to the native
// `PaintFE` binary) so the native `src/main.rs` stays untouched: cargo only
// builds a path dependency's `[lib]` target, never its `[[bin]]`, so this
// crate never pulls in main.rs's native-only startup code.
use paintfe::app::PaintFEApp;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub struct WebHandle {
    runner: eframe::WebRunner,
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl WebHandle {
    #[allow(clippy::new_without_default)]
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        console_error_panic_hook::set_once();
        Self {
            runner: eframe::WebRunner::new(),
        }
    }

    #[wasm_bindgen]
    pub async fn start(&self, canvas: web_sys::HtmlCanvasElement) -> Result<(), wasm_bindgen::JsValue> {
        self.runner
            .start(
                canvas,
                eframe::WebOptions::default(),
                Box::new(|cc| {
                    paintfe::i18n::init();
                    paintfe::i18n::load_label_overrides();
                    paintfe::io::register_codecs();
                    paintfe::web_bridge::set_egui_context(cc.egui_ctx.clone());
                    // No startup files (no CLI args in a browser) and an
                    // inert IPC receiver (single-instance IPC is a native-only
                    // concept — the non-Windows fallback channel never
                    // receives anything, which is exactly what we want here).
                    let ipc_receiver = paintfe::ipc::start_listener();
                    Ok(Box::new(PaintFEApp::new(cc, Vec::new(), ipc_receiver)) as Box<dyn eframe::App>)
                }),
            )
            .await
    }
}
