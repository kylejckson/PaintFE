// PaintFE library crate — re-exports modules for integration tests.
// The binary entry point remains in main.rs.
#![allow(dead_code)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]
#![allow(clippy::large_enum_variant)]
#![allow(clippy::unnecessary_unwrap)]
#![allow(clippy::new_without_default)]
#![allow(private_interfaces)]

#[macro_use]
pub mod i18n;
pub mod app;
pub mod assets;
pub mod canvas;
#[cfg(not(target_arch = "wasm32"))]
pub mod cli;
pub mod components;
pub mod config;
pub mod document;
pub mod experimental;
pub mod gpu;
pub mod io;
pub mod ipc;
pub mod linux_key_probe;
pub mod logger;
pub mod ops;
#[cfg(not(target_arch = "wasm32"))]
pub mod paintdotnet_plugins;
pub mod par_compat;
#[cfg(not(target_arch = "wasm32"))]
pub mod pdn;
pub mod project;
pub mod render;
pub mod services;
pub mod signal_draw;
pub mod signal_widgets;
pub mod theme;
pub mod time_compat;
pub mod ui;
#[cfg(target_arch = "wasm32")]
pub mod web_bridge;
#[cfg(target_arch = "wasm32")]
pub mod web_fs;
#[cfg(target_arch = "wasm32")]
pub mod web_storage;
pub mod windows_key_probe;
