pub(crate) mod document_info;
pub mod label_editor;
mod settings_sections;
pub mod settings_window;
mod tablet_settings;
pub mod tools;

#[cfg(not(target_arch = "wasm32"))]
pub mod optional_features;

#[cfg(not(target_arch = "wasm32"))]
pub mod plugin_browser;

#[cfg(not(target_arch = "wasm32"))]
pub mod plugin_effect;

#[cfg(not(target_arch = "wasm32"))]
pub mod plugin_author;
