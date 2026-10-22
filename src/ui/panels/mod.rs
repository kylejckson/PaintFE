pub(crate) mod document_info;
pub mod label_editor;
mod settings_sections;
pub mod settings_window;
pub mod tools;

#[cfg(not(target_arch = "wasm32"))]
pub mod optional_features;
