/// Action requested from the paste overlay context menu.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PasteAction {
    Commit,
    CommitAndSelect,
    Cancel,
}

pub struct Canvas {
    pub zoom: f32,
    pan_offset: Vec2,
    last_filter_was_linear: Option<bool>, // Track last filter state to detect changes
    pub last_canvas_rect: Option<Rect>,
    /// Accent color for selection outlines (set from theme).
    pub selection_stroke: Color32,
    /// Faint accent for selection fill overlay (set from theme).
    pub selection_fill: Color32,
    /// Contrasting color for selection dashes (white in dark mode, black in light).
    pub selection_contrast: Color32,
    /// Result from paste overlay context menu.
    pub paste_context_action: Option<PasteAction>,
    /// When true, the paste overlay context menu should auto-open on next frame.
    pub open_paste_menu: bool,
    /// GPU renderer (always initialised — uses software fallback if no hardware).
    pub gpu_renderer: crate::gpu::GpuRenderer,
    /// FPS tracking: recent frame times for averaging
    frame_times: VecDeque<f64>,
    /// Cached FPS value (updated periodically)
    pub fps: f32,
    /// True while the fill tool is recalculating its preview (shown in loading bar).
    pub fill_recalc_active: bool,
    /// Non-user-facing tool warmup/build operation shown in the debug loading bar.
    pub tool_map_build_label: Option<String>,
    /// True while a gradient commit is in progress (shown in loading bar).
    pub gradient_commit_active: bool,
    /// Cached texture for layers above the active layer during paste preview.
    /// Rebuilt each frame while a paste overlay is active and layers above exist.
    paste_layers_above_cache: Option<egui::TextureHandle>,
    /// Cached texture for layers below the active layer during overwrite-mode paste preview.
    paste_layers_below_cache: Option<egui::TextureHandle>,
    /// Cached texture for overwrite-mode paste preview of the active layer.
    paste_overwrite_preview_cache: Option<egui::TextureHandle>,
    /// Transform represented by the exact overwrite preview texture.
    paste_overwrite_transform_cache: Option<crate::ops::clipboard::PasteOverlayTransform>,
    /// Static active-layer texture used for the fast approximate overwrite
    /// preview while a transform gesture is in progress.
    paste_overwrite_base_cache: Option<egui::TextureHandle>,
    /// Canvas generation represented by the static paste-session textures.
    paste_static_cache_generation: u64,
    /// Cached texture for brush tip cursor overlay (reused across frames).
    brush_tip_cursor_tex: Option<egui::TextureHandle>,
    /// Second pass texture (inverted) for visibility on all backgrounds.
    brush_tip_cursor_tex_inv: Option<egui::TextureHandle>,
    /// Cache key for brush tip cursor: (tip_name, mask_size, hardness_pct)
    brush_tip_cursor_key: (String, u32, u32),
    /// Tool icon texture for custom cursor overlay (set from app.rs each frame).
    pub tool_cursor_icon: Option<egui::TextureHandle>,
    /// The egui widget Id of the main canvas area (used to distinguish canvas focus
    /// from text-input focus so single-key tool shortcuts are suppressed while typing).
    pub canvas_widget_id: Option<egui::Id>,
    /// Cached checkerboard texture (screen-space tiled, Nearest filtered).
    /// Rebuilt only when viewport size or brightness changes.
    checkerboard_texture: Option<egui::TextureHandle>,
    /// Brightness value used to build the cached checkerboard texture.
    checkerboard_brightness_cached: f32,
    /// Viewport cell dimensions of the cached checkerboard texture (cols, rows).
    checkerboard_cached_size: (usize, usize),
}

include!("view/core.rs");
include!("view/overlay.rs");
include!("view/helpers.rs");
