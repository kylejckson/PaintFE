use crate::assets::{AppSettings, Assets, Icon};
use crate::canvas::{BlendMode, CanvasState, Layer, LayerContent, TiledImage};
use crate::components::history::{HistoryManager, LayerOpCommand, LayerOperation, SnapshotCommand};
use crate::ops::canvas_ops::ImageChannel;
use crate::ops::dialogs::DialogColors;
use crate::ops::text_layer::{
    EnvelopeWarp, GradientFillEffect, InnerShadowEffect, OutlineEffect, OutlinePosition,
    ShadowEffect, TextEffects, TextWarp, TextureFillEffect,
};
use crate::time_compat::Instant;
use eframe::egui;
use egui::{
    Color32, ColorImage, CursorIcon, Id, Pos2, Rect, Sense, TextureHandle, TextureOptions, Vec2,
};
use image::Rgba;
use std::collections::{HashMap, HashSet};

const THUMBNAIL_SIZE: u32 = 40;
const MAX_RECOMMENDED_LAYERS: usize = 200;

struct ThumbnailCache {
    texture: Option<TextureHandle>,
    /// Canvas dirty_generation at last compute.
    last_generation: u64,
    last_update: Instant,
}

#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub enum LayerSettingsTab {
    #[default]
    General,
    Effects,
    Warp,
}

#[derive(Default)]
pub struct LayerSettingsState {
    pub editing_layer: Option<usize>,
    pub editing_name: String,
    pub editing_opacity: f32,
    pub editing_blend_mode: BlendMode,
    /// Active tab for the settings window.
    pub tab: LayerSettingsTab,
    /// Cloned text effects for editing (applied on "Apply").
    pub text_effects: TextEffects,
    /// Cloned text warp for editing (applied to ALL blocks live).
    pub text_warp: TextWarp,
    /// Receiver for async texture file loading.
    pub texture_load_rx: Option<std::sync::mpsc::Receiver<Vec<u8>>>,
}

/// State for peek functionality
#[derive(Default)]
pub struct PeekState {
    pub is_peeking: bool,
    pub peek_layer_index: Option<usize>,
    pub saved_visibility: Vec<bool>,
    /// True when a layer is permanently soloed (right-click peek).
    pub is_soloed: bool,
    pub solo_layer_index: Option<usize>,
    pub solo_saved_visibility: Vec<bool>,
    /// Set for one frame after a peek ends, so the `secondary_clicked` event
    /// on the same release frame doesn't accidentally toggle solo.
    peek_just_ended: bool,
}

/// State for inline rename
#[derive(Default)]
pub struct RenameState {
    pub renaming_layer: Option<usize>,
    pub rename_text: String,
    pub focus_requested: bool,
}

#[derive(Default)]
pub struct FolderRenameState {
    pub renaming_folder: Option<u64>,
    pub rename_text: String,
    pub focus_requested: bool,
}

/// State for drag-and-drop layer reordering
#[derive(Default)]
struct DragState {
    /// Display index currently being dragged (0 = topmost in UI).
    dragging_display_idx: Option<usize>,
    dragging_folder_id: Option<u64>,
    drag_offset_y: f32,
    origin_display_idx: usize,
    /// Animated visual offsets per display index (elastic slide-out).
    anim_offsets: Vec<f32>,
}

/// Actions that need app-level handling (file dialogs, active dialogs, etc.)
#[derive(Debug, Clone)]
pub enum LayerAppAction {
    ImportFromFile,
    FlipHorizontal,
    FlipVertical,
    RotateScale,
    AlignLayer,
    /// Merge the layer at `layer_idx` down as an alpha mask onto the layer below it.
    MergeDownAsMask(usize),
    AddLayerMaskRevealAll(usize),
    AddLayerMaskFromSelection(usize),
    ToggleLayerMaskEdit(usize),
    ToggleLayerMask(usize),
    InvertLayerMask(usize),
    ApplyLayerMask(usize),
    DeleteLayerMask(usize),
    /// Rasterize the text layer at `layer_idx`, closing the settings dialog.
    RasterizeTextLayer(usize),
}

#[derive(Default)]
pub struct LayersPanel {
    settings_state: LayerSettingsState,
    peek_state: PeekState,
    rename_state: RenameState,
    folder_rename_state: FolderRenameState,
    drag_state: DragState,
    selected_layers: HashSet<usize>,
    selected_folder: Option<u64>,
    thumbnail_cache: HashMap<usize, ThumbnailCache>,
    last_layer_count: usize,
    /// Layer index to remove from GPU texture cache.
    pub pending_gpu_delete: Option<usize>,
    /// Layer index deleted this frame so text/tool state can reindex or cancel.
    pub pending_deleted_layer: Option<usize>,
    /// When set, all GPU layer textures should be invalidated.
    pub pending_gpu_clear: bool,
    pub pending_app_action: Option<LayerAppAction>,
    search_query: String,
}

include!("layers/list.rs");
include!("layers/settings.rs");
include!("layers/operations.rs");

enum LayerAction {
    Select { additive: bool },
    StartRename,
    FinishRename,
    CancelRename,
    ToggleVisibility,
    BeginDrag,
}

/// Actions from context menu
enum ContextAction {
    AddNew,
    AddNewTextLayer,
    AddAdjustment,
    AddFolder,
    MergeDown,
    MergeDownAsMask,
    AddLayerMaskRevealAll,
    AddLayerMaskFromSelection,
    ToggleLayerMaskEdit,
    ToggleLayerMask,
    InvertLayerMask,
    ApplyLayerMask,
    DeleteLayerMask,
    FlattenImage,
    Duplicate,
    Delete,
    OpenSettings,
    MoveToTop,
    MoveUp,
    MoveDown,
    MoveToBottom,
    Rename,
    ImportFromFile,
    FlipHorizontal,
    FlipVertical,
    RotateScale,
    AlignLayer,
    SoloLayer,
    HideAll,
    ShowAll,
    RasterizeTextLayer,
    TextLayerEffects,
    TextLayerWarp,
    MoveToFolder(u64),
    RemoveFromFolder,
    ExtractChannel(ImageChannel),
    ReplaceAlphaFromBelowLuminance,
}

enum FolderAction {
    Select(u64),
    BeginDrag(u64),
    ToggleCollapsed(u64),
    ToggleVisibility(u64),
    StartRename(u64),
    FinishRename,
    CancelRename,
    Delete(u64),
    AddLayer(u64),
    AddTextLayer(u64),
    AddFolderAbove(u64),
    SelectContents(u64),
    SetColor(u64, Option<u8>),
}

#[derive(Clone, Copy)]
enum LayerListRow {
    Folder(u64),
    Layer {
        display_idx: usize,
        layer_idx: usize,
    },
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum FolderDropZone {
    Above,
    Inside,
    Below,
}
