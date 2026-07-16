// Browser file-open bridge (wasm32 only).
//
// There is no synchronous native file dialog in the browser: `<input
// type=file>` is the only picker, and reading the selected files is async
// (`File::array_buffer()` returns a Promise). This module opens that picker,
// reads each selected file in the background, and queues the resulting
// `(name, bytes)` pairs — the app drains the queue once per frame and feeds
// them into whichever byte-oriented path is waiting for that purpose.
//
// Pickers are tagged (`"open"`, `"texture_fill"`, `"brush_tip"`, `"font"`,
// ...) so several independent one-shot pickers can be in flight without
// clobbering each other's results.
use std::cell::RefCell;
use std::collections::HashMap;
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;

thread_local! {
    static PENDING: RefCell<HashMap<&'static str, Vec<(String, Vec<u8>)>>> =
        RefCell::new(HashMap::new());
    static EGUI_CTX: RefCell<Option<egui::Context>> = const { RefCell::new(None) };
}

/// Store the egui context so the async file-read callback (which runs
/// outside egui's normal input-driven update cycle) can wake the render
/// loop once a file finishes loading, instead of waiting for the next
/// mouse/keyboard event.
pub fn set_egui_context(ctx: egui::Context) {
    EGUI_CTX.with(|c| *c.borrow_mut() = Some(ctx));
}

fn wake() {
    EGUI_CTX.with(|c| {
        if let Some(ctx) = c.borrow().as_ref() {
            ctx.request_repaint();
        }
    });
}

/// Open the browser's native file picker for a tagged purpose. `accept` is a
/// standard `<input accept>` filter string (e.g. `"image/*"` or
/// `".ttf,.otf,.woff,.woff2"`). Selected files are read in the background;
/// poll their bytes with [`drain_pending`] using the same tag.
pub fn open_picker(tag: &'static str, accept: &str, multiple: bool) {
    let result = (|| -> Result<(), JsValue> {
        let window = web_sys::window().ok_or("no window")?;
        let document = window.document().ok_or("no document")?;
        let input = document
            .create_element("input")?
            .dyn_into::<web_sys::HtmlInputElement>()?;
        input.set_type("file");
        input.set_multiple(multiple);
        input.set_accept(accept);
        input.style().set_property("display", "none").ok();

        let onchange = Closure::<dyn FnMut(_)>::new(move |event: web_sys::Event| {
            let Some(target) = event.target() else { return };
            let Ok(input) = target.dyn_into::<web_sys::HtmlInputElement>() else {
                return;
            };
            let Some(files) = input.files() else { return };
            for i in 0..files.length() {
                let Some(file) = files.get(i) else { continue };
                let name = file.name();
                wasm_bindgen_futures::spawn_local(async move {
                    if let Ok(buf) = wasm_bindgen_futures::JsFuture::from(file.array_buffer()).await
                    {
                        let array = js_sys::Uint8Array::new(&buf);
                        let bytes = array.to_vec();
                        PENDING
                            .with(|p| p.borrow_mut().entry(tag).or_default().push((name, bytes)));
                        wake();
                    }
                });
            }
            if let Some(parent) = input.parent_node() {
                let _ = parent.remove_child(&input);
            }
        });

        document.body().ok_or("no body")?.append_child(&input)?;
        input.add_event_listener_with_callback("change", onchange.as_ref().unchecked_ref())?;
        // Leak intentionally: this is a one-shot UI callback tied to a single
        // <input> element's lifetime, not a recurring listener.
        onchange.forget();
        input.click();
        Ok(())
    })();

    if let Err(e) = result {
        web_sys::console::error_2(&"PaintFE: failed to open file picker:".into(), &e);
    }
}

/// Take all files that finished reading for `tag` since the last call.
/// Intended to be polled once per frame from wherever requested the picker.
pub fn drain_pending(tag: &'static str) -> Vec<(String, Vec<u8>)> {
    PENDING.with(|p| p.borrow_mut().remove(tag).unwrap_or_default())
}

const OPEN_TAG: &str = "open";

/// Open the browser's native file picker for importing images/.pfe projects.
pub fn trigger_open_picker() {
    open_picker(
        OPEN_TAG,
        ".pfe,.png,.jpg,.jpeg,.webp,.bmp,.tga,.gif,.ico,.tiff,.tif",
        true,
    );
}

/// Take all project/image files that have finished reading since the last call.
pub fn drain_pending_opens() -> Vec<(String, Vec<u8>)> {
    drain_pending(OPEN_TAG)
}
