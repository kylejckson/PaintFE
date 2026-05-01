pub struct FilterResult {
    /// Index of the project that spawned this job.
    pub project_index: usize,
    /// Index of the layer that was processed.
    pub layer_idx: usize,
    /// The original pixels before the filter (for undo snapshot).
    pub original_pixels: TiledImage,
    /// The processed full-resolution pixels.
    pub result_pixels: TiledImage,
    /// Human-readable name for the undo history entry.
    pub description: String,
    /// Non-zero for live-preview jobs; result is discarded when token != current preview_job_token.
    pub preview_token: u64,
}

/// Result delivered from a background canvas-wide operation (resize image/canvas).
/// Unlike `FilterResult` which targets a single layer, this carries all layers.
pub struct CanvasOpResult {
    pub project_index: usize,
    /// Snapshot of the canvas state before the operation (for undo).
    pub before: crate::components::history::CanvasSnapshot,
    /// The processed layers (one `TiledImage` per layer, in order).
    pub result_layers: Vec<TiledImage>,
    /// New canvas dimensions after the operation.
    pub new_width: u32,
    pub new_height: u32,
    /// Human-readable name for the undo history entry.
    pub description: String,
}

// ============================================================================
// ASYNC IO PIPELINE — background image loading / saving
// ============================================================================

/// Action from the Filter > Custom context menu
enum CustomScriptAction {
    Run(usize),
    Edit(usize),
    Delete(usize),
}

struct PendingPasteRequest {
    image: RgbaImage,
    target_project_id: uuid::Uuid,
    cursor_canvas: Option<(f32, f32)>,
    source_center: Option<egui::Pos2>,
    use_source_center: bool,
    overwrite_transparent_pixels: bool,
    overwrite_mask: Option<image::GrayImage>,
}

/// Result delivered from a background IO thread.
pub enum IoResult {
    /// An image file was decoded and tiled, ready to become a new project.
    ImageLoaded {
        tiled: TiledImage,
        width: u32,
        height: u32,
        path: std::path::PathBuf,
        format: SaveFormat,
    },
    /// Image decoding failed.
    LoadFailed(String),
    /// Image save completed successfully.
    SaveComplete {
        project_index: usize,
        path: std::path::PathBuf,
        format: SaveFormat,
        quality: u8,
        tiff_compression: TiffCompression,
        /// When saving from Save As, also update the project name/path.
        update_project_path: bool,
    },
    /// Image save failed.
    SaveFailed { project_index: usize, error: String },
    /// Animated file opened: first frame loaded as a project with this ID,
    /// remaining frames arrive separately.
    AnimatedLoaded {
        tiled: TiledImage,
        width: u32,
        height: u32,
        path: std::path::PathBuf,
        format: SaveFormat,
        fps: f32,
        frame_count: u32,
    },
    /// Additional animation frames decoded in background — append as layers
    /// to the project matching `path`.
    AnimatedFramesLoaded {
        path: std::path::PathBuf,
        frames: Vec<(TiledImage, String)>, // (pixels, layer_name)
    },
    /// A .pfe project file was loaded in background — carries full multi-layer state.
    PfeLoaded {
        canvas_state: CanvasState,
        path: std::path::PathBuf,
    },
}

pub struct PaintFEApp {
    // Multi-Document State
    projects: Vec<Project>,
    active_project_index: usize,
    untitled_counter: usize,

    // Shared Canvas Renderer
    canvas: Canvas,

    // Shared File Handler (for dialogs)
    file_handler: FileHandler,

    // UI Components
    tools_panel: tools::ToolsPanel,
    layers_panel: layers::LayersPanel,
    colors_panel: colors::ColorsPanel,
    palette_panel: palette::PalettePanel,
    history_panel: history::HistoryPanel,

    // Dialogs
    new_file_dialog: NewFileDialog,
    save_file_dialog: SaveFileDialog,
    settings_window: SettingsWindow,

    // Assets & Settings
    assets: Assets,
    settings: AppSettings,

    // Theme & Floating Windows
    theme: Theme,
    window_visibility: WindowVisibility,

    // Modal dialog system (at most one open at a time)
    active_dialog: ActiveDialog,

    // Paste overlay (floating pasted image being manipulated)
    paste_overlay: Option<PasteOverlay>,
    pending_paste_request: Option<PendingPasteRequest>,
    paste_transform_undo: Vec<crate::ops::clipboard::PasteOverlayTransform>,
    paste_transform_redo: Vec<crate::ops::clipboard::PasteOverlayTransform>,

    // Move-selection drag state (tracks screen-space mouse for translating the mask)
    move_sel_dragging: bool,
    move_sel_last_canvas: Option<(i32, i32)>,
    move_sel_handle: Option<crate::ops::shapes::ShapeHandle>,
    move_sel_start_mask: Option<image::GrayImage>,
    move_sel_start_bounds: Option<(u32, u32, u32, u32)>,
    pending_selection_reassert: Option<image::GrayImage>,

    // Floating panel edge tracking: store offset from screen edge so panels
    // move with window resizes while still being user-draggable.
    layers_panel_right_offset: Option<(f32, f32)>, // (offset_from_right, y)
    history_panel_right_offset: Option<(f32, f32)>, // (offset_from_right, offset_from_bottom)
    colors_panel_left_offset: Option<(f32, f32)>,  // (x, offset_from_bottom)
    palette_panel_pos: Option<(f32, f32)>,         // (x, y)
    tools_panel_pos: Option<(f32, f32)>,           // (x, y) absolute
    last_screen_size: (f32, f32),

    // True while a MovePixels overlay is active (extraction already pushed to history).
    is_move_pixels_active: bool,
    // True when pointer is currently over the floating Layers window.
    is_pointer_over_layers_panel: bool,

    // Async filter pipeline
    filter_sender: mpsc::Sender<FilterResult>,
    filter_receiver: mpsc::Receiver<FilterResult>,
    /// When > 0, a background filter job is in progress; show spinner.
    pending_filter_jobs: usize,
    /// Time when filter operations started (for elapsed time display)
    filter_ops_start_time: Option<f64>,
    /// Human-readable description of the currently running filter operation.
    filter_status_description: String,
    /// Monotonically-increasing token; preview jobs carrying an older token are discarded on receipt.
    preview_job_token: u64,
    /// Cancellation flag for the current preview job; set to true before spawning a new one.
    filter_cancel: std::sync::Arc<std::sync::atomic::AtomicBool>,

    // Async canvas-wide operation pipeline (resize image/canvas)
    canvas_op_sender: mpsc::Sender<CanvasOpResult>,
    canvas_op_receiver: mpsc::Receiver<CanvasOpResult>,

    // Async IO pipeline (background image load / save)
    io_sender: mpsc::Sender<IoResult>,
    io_receiver: mpsc::Receiver<IoResult>,
    /// When > 0, a background IO job is in progress; show spinner.
    pending_io_ops: usize,
    /// Time when IO operations started (for elapsed time display)
    io_ops_start_time: Option<f64>,
    /// Whether ONNX Runtime is available (both DLL and model configured + DLL probed OK)
    onnx_available: bool,
    /// Cached ONNX paths used for last probe (re-probe only when changed)
    onnx_last_probed_paths: (String, String),

    // Script editor
    script_editor: script_editor::ScriptEditorPanel,
    script_right_offset: Option<(f32, f32)>,
    script_sender: mpsc::Sender<ScriptMessage>,
    script_receiver: mpsc::Receiver<ScriptMessage>,
    /// Custom script effects registered in Filter > Custom
    custom_scripts: Vec<script_editor::CustomScriptEffect>,
    /// Backup of layer pixels before script execution (for restore on error)
    script_original_pixels: Option<(usize, usize, TiledImage)>,

    /// Project index pending close confirmation (unsaved-changes dialog)
    pending_close_index: Option<usize>,

    /// True when we are showing the "exit with unsaved projects?" dialog.
    pending_exit: bool,
    /// True after the user confirmed "Exit without Saving" — suppresses the dialog on the
    /// next close_requested so the OS close actually goes through.
    force_exit: bool,

    /// Queue of project indices (untitled/dirty) that still need Save As dialogs
    /// before the app can exit. Populated when user clicks "Save" in the exit dialog.
    /// Each entry is processed sequentially: switch to tab → open Save As → wait for
    /// completion/cancel → advance to next or exit.
    exit_save_queue: Vec<usize>,
    /// True while working through exit_save_queue — the current Save As dialog
    /// is part of the exit flow (not a normal Save As).
    exit_save_active: bool,

    /// Instant of the last auto-save tick (used to measure the interval).
    last_autosave: std::time::Instant,

    /// True only on the very first update() call — used to send a reliable Maximized command.
    first_frame: bool,

    // Single-instance IPC: file paths sent from other PaintFE invocations
    ipc_receiver: mpsc::Receiver<PathBuf>,
    /// File paths to open on the first update() frame (from positional CLI args).
    pending_startup_files: Vec<PathBuf>,
    /// True when startup files have been queued and the initial blank project
    /// should be auto-closed once the first real file finishes loading.
    close_initial_blank: bool,

    prev_ctrl_c_down: bool,
    prev_ctrl_x_down: bool,
    prev_ctrl_v_down: bool,
    prev_enter_down: bool,
    prev_escape_down: bool,
    prev_vk_c_press_count: u64,
    prev_vk_x_press_count: u64,
    prev_vk_v_press_count: u64,
    prev_vk_enter_press_count: u64,
    prev_vk_escape_press_count: u64,
    recent_color_project_id: Option<uuid::Uuid>,
    recent_color_undo_count: usize,
    palette_reposition_settle_frames: u8,
    palette_startup_target_pos: Option<(f32, f32)>,
    last_tool_settings_fingerprint: u64,
    last_window_state_fingerprint: u64,
    last_paste_trigger_time: f64,
}

/// Discover a system CJK font at runtime (for Japanese, Korean, Chinese support).
/// Returns `(font_name, font_bytes)` if found.
fn discover_system_cjk_font() -> Option<(String, Vec<u8>)> {
    // Common CJK font paths to try, ordered by quality/coverage
    let candidates: &[(&str, &[&str])] = &[
        #[cfg(target_os = "windows")]
        (
            "system_cjk",
            &[
                // Noto Sans CJK (if user installed it)
                "C:\\Windows\\Fonts\\NotoSansCJK-Regular.ttc",
                "C:\\Windows\\Fonts\\NotoSansCJKsc-Regular.otf",
                // Microsoft YaHei (best quality CJK on Windows, covers CJK Unified)
                "C:\\Windows\\Fonts\\msyh.ttc",
                "C:\\Windows\\Fonts\\msyh.ttf",
                // Yu Gothic (good JP coverage)
                "C:\\Windows\\Fonts\\YuGothR.ttc",
                "C:\\Windows\\Fonts\\YuGothM.ttc",
                // Malgun Gothic (Korean)
                "C:\\Windows\\Fonts\\malgun.ttf",
                // SimSun (Chinese fallback, older but widely available)
                "C:\\Windows\\Fonts\\simsun.ttc",
                // MS Gothic (JP fallback)
                "C:\\Windows\\Fonts\\msgothic.ttc",
                // Meiryo (JP)
                "C:\\Windows\\Fonts\\meiryo.ttc",
            ],
        ),
        #[cfg(target_os = "linux")]
        (
            "system_cjk",
            &[
                // Noto Sans CJK (most common on Linux)
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/OTF/NotoSansCJK-Regular.ttc",
                // WenQuanYi (common Chinese font on Linux)
                "/usr/share/fonts/wenquanyi/wqy-microhei/wqy-microhei.ttc",
                "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
                // Droid Sans Fallback
                "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
                "/usr/share/fonts/droid/DroidSansFallbackFull.ttf",
            ],
        ),
        #[cfg(target_os = "macos")]
        (
            "system_cjk",
            &[
                // macOS always has Hiragino
                "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
                "/System/Library/Fonts/HiraginoSans-W3.ttc",
                "/Library/Fonts/Arial Unicode.ttf",
            ],
        ),
    ];

    for (name, paths) in candidates {
        for path in *paths {
            if let Ok(data) = std::fs::read(path)
                && data.len() > 100
            {
                // sanity check
                return Some((name.to_string(), data));
            }
        }
    }
    None
}

