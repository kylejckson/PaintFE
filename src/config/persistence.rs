//! Native settings writes run off the UI thread; browser storage stays local.
#[cfg(not(target_arch = "wasm32"))]
mod native;
#[cfg(target_arch = "wasm32")]
mod web;
#[cfg(not(target_arch = "wasm32"))]
pub use native::drain;
#[cfg(not(target_arch = "wasm32"))]
pub(crate) use native::{begin, error, flush, request};
#[cfg(target_arch = "wasm32")]
pub use web::drain;
#[cfg(target_arch = "wasm32")]
pub(crate) use web::{begin, error, flush, request};
