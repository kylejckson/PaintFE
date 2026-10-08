use crate::assets::{
    AppSettings, Assets, BindableAction, Icon, KeyCombo, PixelGridMode, SettingsWindow,
};
use crate::canvas::{BlendMode, Canvas, CanvasState, Layer, TiledImage};
use crate::components::dialogs::{NewFileDialog, SaveFileDialog, SaveFormat, TiffCompression};
use crate::components::history::{
    CanvasSnapshot, SelectionCommand, SingleLayerSnapshotCommand, SnapshotCommand,
};
use crate::components::*;
use crate::io::FileHandler;
use crate::log_info;
use crate::ops::clipboard::{ClipboardImageSource, PasteOverlay};
use crate::ops::dialogs::{ActiveDialog, DialogResult};
use crate::ops::scripting::{ScriptMessage, apply_canvas_ops};
use crate::project::Project;
use crate::signal_widgets;
use crate::theme::{Theme, WindowVisibility};
use eframe::egui;
use image::RgbaImage;
use std::collections::{HashSet, hash_map::DefaultHasher};
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::sync::mpsc;

include!("app/types.rs");
include!("app/bootstrap.rs");
include!("app/runtime.rs");
include!("app/project_io.rs");
include!("app/ops.rs");
include!("app/panels.rs");

#[cfg(test)]
mod issue_tests;
