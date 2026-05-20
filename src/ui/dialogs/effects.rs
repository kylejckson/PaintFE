// ============================================================================
// EFFECT DIALOGS — Modal dialog UIs for all 24 effects
// ============================================================================
//
// Each dialog follows the standard pattern:
//   - struct with effect params + original_pixels + original_flat + layer_idx + live_preview
//   - new(&CanvasState) constructor
//   - show(&mut self, ctx) -> DialogResult<params>
//   - Accent header, Grid params, preview controls, OK/Cancel footer
// ============================================================================

use eframe::egui;
use egui::Color32;
use image::RgbaImage;

use crate::canvas::{CanvasState, TiledImage};
use crate::ops::effects::{ColorFilterMode, GridStyle, HalftoneShape, NoiseType, OutlineMode};
use crate::ui::dialogs::core::{
    DialogColors, DialogResult, accent_separator, contrast_text_color, dialog_footer,
    dialog_footer_with_reset, dialog_slider, numeric_field_with_buttons, paint_dialog_header,
    paint_dialog_header_with_texture, preview_controls, section_label,
};

// Create common effect dialog fields
macro_rules! effect_dialog_base {
    ($name:ident { $($field:ident : $ty:ty = $default:expr),* $(,)? }) => {
        #[allow(dead_code)]
        pub struct $name {
            pub original_pixels: Option<TiledImage>,
            pub original_flat: Option<RgbaImage>,
            /// Unused legacy fields kept for API compatibility.
            pub preview_flat_small: Option<RgbaImage>,
            pub preview_scale: f32,
            pub layer_idx: usize,
            pub live_preview: bool,
            /// Set true when a slider value changes; cleared when the effect is applied.
            pub needs_apply: bool,
            /// True on any frame where a slider is actively being dragged.
            pub dragging: bool,
            /// Fully-computed effect result at preview scale, used for progressive reveal.
            pub processed_preview: Option<RgbaImage>,
            /// Next row to reveal in the progressive top-down display.
            pub progressive_row: u32,
            /// Background flat-extraction job: populated by rayon, polled each frame.
            pub pending_flat: Option<std::sync::Arc<std::sync::Mutex<Option<RgbaImage>>>>,
            $(pub $field: $ty,)*
        }

        impl $name {
            pub fn new(state: &CanvasState) -> Self {
                let idx = state.active_layer_index;
                // Clone original pixels (fast — COW Arc clones only).
                let original_pixels = state.layers.get(idx).map(|l| l.pixels.clone());
                // Defer the expensive to_rgba_image() to a rayon thread so the dialog
                // opens on the very next frame.  poll_flat() will resolve it.
                let pending_flat = if original_pixels.is_some() {
                    let arc: std::sync::Arc<std::sync::Mutex<Option<RgbaImage>>> =
                        std::sync::Arc::new(std::sync::Mutex::new(None));
                    let arc_clone = arc.clone();
                    let tiled = original_pixels.clone().unwrap();
                    rayon::spawn(move || {
                        let flat = tiled.to_rgba_image();
                        if let Ok(mut guard) = arc_clone.lock() {
                            *guard = Some(flat);
                        }
                    });
                    Some(arc)
                } else {
                    None
                };
                Self {
                    original_pixels,
                    original_flat: None, // populated by poll_flat()
                    preview_flat_small: None,
                    preview_scale: 1.0,
                    layer_idx: idx,
                    live_preview: true,
                    needs_apply: false,
                    dragging: false,
                    processed_preview: None,
                    progressive_row: 0,
                    pending_flat,
                    $($field: $default,)*
                }
            }

            /// Call each frame while the dialog is open.  Returns `true` the first
            /// time `original_flat` becomes available (i.e. the background job just
            /// finished).  Callers should use the `true` return to trigger an initial
            /// preview render.
            pub fn poll_flat(&mut self) -> bool {
                if self.original_flat.is_some() {
                    return false; // already resolved
                }
                // Extract the flat image without holding a borrow on self.pending_flat
                // (so we can immediately assign self.original_flat / self.pending_flat).
                let maybe_flat = self.pending_flat.as_ref().and_then(|arc| {
                    arc.try_lock().ok().and_then(|mut guard| guard.take())
                });
                if let Some(flat) = maybe_flat {
                    self.original_flat = Some(flat);
                    self.pending_flat = None;
                    return true;
                }
                false
            }
        }
    };
}

/// Track a slider for deferred preview: the slider value updates instantly
/// (no lag) but the effect is only applied when the slider is *released*.
/// Returns `true` when the slider was just released after being dragged.
pub fn track_slider(response: &egui::Response, dragging: &mut bool) -> bool {
    if response.dragged() || response.changed() {
        *dragging = true;
    }
    if response.drag_stopped() {
        *dragging = false;
        return true; // "apply now"
    }
    false
}

// ============================================================================
// BLUR DIALOGS
// ============================================================================

effect_dialog_base!(BokehBlurDialog {
    radius: f32 = 0.0,
    advanced_blur: bool = false
});

mod blur {
    use super::*;
    include!("effects/blur.rs");
}
pub use blur::*;

mod distort {
    use super::*;
    include!("effects/distort.rs");
}
pub use distort::*;

mod noise {
    use super::*;
    include!("effects/noise.rs");
}
pub use noise::*;

mod stylize {
    use super::*;
    include!("effects/stylize.rs");
}
pub use stylize::*;

mod render {
    use super::*;
    include!("effects/render.rs");
}
pub use render::*;

mod artistic {
    use super::*;
    include!("effects/artistic.rs");
}
pub use artistic::*;

mod ai {
    use super::*;
    include!("effects/ai.rs");
}
pub use ai::*;
