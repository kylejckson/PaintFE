pub(crate) mod palette;
include!("state.rs");
include!("behavior.rs");
include!("session_history.rs");

include!("liquify_history.rs");

include!("brush_geometry.rs");
include!("paint_session.rs");
include!("brush_dynamics.rs");
include!("bristle_raster.rs");
include!("brush_picker.rs");

#[cfg(test)]
mod session_tests;

#[cfg(not(target_arch = "wasm32"))]
include!("automation.rs");
