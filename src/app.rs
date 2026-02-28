use crate::assets::{AppSettings, Assets, BindableAction, Icon, PixelGridMode, SettingsWindow};
use crate::canvas::{BlendMode, Canvas, CanvasState, Layer, TiledImage};
use crate::components::dialogs::{NewFileDialog, SaveFileDialog, SaveFormat, TiffCompression};
use crate::components::history::{SingleLayerSnapshotCommand, SnapshotCommand};
use crate::components::*;
use crate::io::FileHandler;
use crate::ops::clipboard::PasteOverlay;
use crate::ops::dialogs::{ActiveDialog, DialogResult};
use crate::ops::scripting::{ScriptMessage, apply_canvas_ops};
use crate::project::Project;
use crate::theme::{Theme, WindowVisibility};
use eframe::egui;
use image::RgbaImage;
use std::path::PathBuf;
use std::sync::mpsc;

// ============================================================================
// ASYNC FILTER PIPELINE — background processing with channel completion
// ============================================================================

/// Result delivered from a background filter thread.
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

// ============================================================================
// ASYNC IO PIPELINE — background image loading / saving
// ============================================================================

/// Action from the Filter > Custom context menu
enum CustomScriptAction {
    Run(usize),
    Edit(usize),
    Delete(usize),
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

    // Move-selection drag state (tracks screen-space mouse for translating the mask)
    move_sel_dragging: bool,
    move_sel_last_canvas: Option<(i32, i32)>,

    // Floating panel edge tracking: store offset from screen edge so panels
    // move with window resizes while still being user-draggable.
    layers_panel_right_offset: Option<(f32, f32)>, // (offset_from_right, y)
    history_panel_right_offset: Option<(f32, f32)>, // (offset_from_right, offset_from_bottom)
    colors_panel_left_offset: Option<(f32, f32)>,  // (x, offset_from_bottom)
    tools_panel_pos: Option<(f32, f32)>,           // (x, y) absolute
    last_screen_size: (f32, f32),

    // True while a MovePixels overlay is active (extraction already pushed to history).
    is_move_pixels_active: bool,

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

impl PaintFEApp {
    pub fn new(
        cc: &eframe::CreationContext<'_>,
        startup_files: Vec<PathBuf>,
        ipc_receiver: mpsc::Receiver<PathBuf>,
    ) -> Self {
        // Initialize settings from disk (or defaults if no saved file)
        let settings = AppSettings::load();

        // Apply saved language preference (or auto-detect on first boot)
        if settings.language.is_empty() {
            let detected = crate::i18n::detect_system_language();
            crate::i18n::set_language(&detected);
        } else {
            crate::i18n::set_language(&settings.language);
        }

        // -- Font configuration ------------------------------------------------
        // Primary: Noto Sans (Latin, Cyrillic, Greek, Thai — ~556KB embedded)
        // Fallback: System CJK font (discovered at runtime for JP/KO/ZH support)
        {
            let mut fonts = egui::FontDefinitions::default();

            // Embed Noto Sans Regular as primary UI font
            fonts.font_data.insert(
                "noto_sans".to_owned(),
                egui::FontData::from_static(include_bytes!("../assets/fonts/NotoSans-Regular.ttf")),
            );

            // Build proportional font family: Noto Sans first, then egui defaults
            let proportional = fonts
                .families
                .entry(egui::FontFamily::Proportional)
                .or_default();
            proportional.insert(0, "noto_sans".to_owned());

            // Try to discover a system CJK font at runtime for JP/KO/ZH support
            if let Some((name, data)) = discover_system_cjk_font() {
                fonts
                    .font_data
                    .insert(name.clone(), egui::FontData::from_owned(data));
                // Insert CJK font after Noto Sans but before egui defaults
                let proportional = fonts
                    .families
                    .entry(egui::FontFamily::Proportional)
                    .or_default();
                proportional.insert(1, name);
            }

            cc.egui_ctx.set_fonts(fonts);
        }

        // Initialize theme from settings with accent and apply immediately
        let accent = if settings.theme_preset == crate::theme::ThemePreset::Custom {
            settings.custom_accent
        } else {
            settings.theme_preset.accent_colors()
        };
        let theme = match settings.theme_mode {
            crate::theme::ThemeMode::Dark => Theme::dark_with_accent(settings.theme_preset, accent),
            crate::theme::ThemeMode::Light => {
                Theme::light_with_accent(settings.theme_preset, accent)
            }
        };
        theme.apply(&cc.egui_ctx);

        // Initialize with one default project
        let initial_project = Project::new_untitled(1, 800, 600);

        // Initialize assets
        let mut assets = Assets::new();
        assets.init(&cc.egui_ctx);

        let (filter_sender, filter_receiver) = mpsc::channel();
        let (io_sender, io_receiver) = mpsc::channel();
        let (script_sender, script_receiver) = mpsc::channel();

        // Probe ONNX Runtime availability
        let onnx_available =
            if !settings.onnx_runtime_path.is_empty() && !settings.birefnet_model_path.is_empty() {
                crate::ops::ai::probe_onnx_runtime(&settings.onnx_runtime_path).is_ok()
                    && std::path::Path::new(&settings.birefnet_model_path).exists()
            } else {
                false
            };
        let onnx_last_probed_paths = (
            settings.onnx_runtime_path.clone(),
            settings.birefnet_model_path.clone(),
        );

        let canvas = Canvas::new(&settings.preferred_gpu);

        Self {
            projects: vec![initial_project],
            active_project_index: 0,
            untitled_counter: 1,
            canvas,
            file_handler: FileHandler::new(),
            tools_panel: tools::ToolsPanel::default(),
            layers_panel: layers::LayersPanel::default(),
            colors_panel: colors::ColorsPanel::default(),
            history_panel: history::HistoryPanel::default(),
            new_file_dialog: NewFileDialog::default(),
            save_file_dialog: SaveFileDialog::default(),
            settings_window: SettingsWindow::default(),
            assets,
            settings,
            theme,
            window_visibility: WindowVisibility::new(),
            active_dialog: ActiveDialog::default(),
            paste_overlay: None,
            move_sel_dragging: false,
            move_sel_last_canvas: None,
            layers_panel_right_offset: None,
            history_panel_right_offset: None,
            colors_panel_left_offset: None,
            tools_panel_pos: None,
            last_screen_size: (0.0, 0.0),
            is_move_pixels_active: false,
            filter_sender,
            filter_receiver,
            pending_filter_jobs: 0,
            filter_ops_start_time: None,
            filter_status_description: String::new(),
            preview_job_token: 1,
            io_sender,
            io_receiver,
            pending_io_ops: 0,
            io_ops_start_time: None,
            onnx_available,
            onnx_last_probed_paths,
            script_editor: {
                let mut se = script_editor::ScriptEditorPanel::default();
                se.load_saved_scripts();
                se
            },
            script_right_offset: None,
            script_sender,
            script_receiver,
            custom_scripts: script_editor::load_custom_effects(),
            script_original_pixels: None,
            pending_close_index: None,
            pending_exit: false,
            force_exit: false,
            last_autosave: std::time::Instant::now(),
            first_frame: true,
            ipc_receiver,
            close_initial_blank: !startup_files.is_empty(),
            pending_startup_files: startup_files,
        }
    }

    /// Get a reference to the active project
    fn active_project(&self) -> Option<&Project> {
        self.projects.get(self.active_project_index)
    }

    /// Get a mutable reference to the active project
    fn active_project_mut(&mut self) -> Option<&mut Project> {
        self.projects.get_mut(self.active_project_index)
    }

    /// Create a new untitled project and switch to it
    fn new_project(&mut self, width: u32, height: u32) {
        self.untitled_counter += 1;
        let project = Project::new_untitled(self.untitled_counter, width, height);
        self.projects.push(project);
        self.active_project_index = self.projects.len() - 1;
        self.canvas.reset_zoom();
    }

    /// If the initial blank project should be auto-closed (because we opened a
    /// file from the command line or IPC), close it now. Called once after the
    /// first real file finishes loading.
    fn maybe_close_initial_blank(&mut self) {
        if !self.close_initial_blank {
            return;
        }
        self.close_initial_blank = false;

        // The initial blank project is always at index 0.  Only auto-close it
        // if it's still untitled, unmodified, and there's now at least one
        // other project.
        if self.projects.len() > 1 {
            let is_blank = self.projects[0].path.is_none() && !self.projects[0].is_dirty;
            if is_blank {
                self.projects.remove(0);
                // The newly loaded project was pushed to the end; adjust index.
                self.active_project_index = self.projects.len() - 1;
            }
        }
    }

    /// Close a project by index, with dirty check
    fn close_project(&mut self, index: usize) {
        if index >= self.projects.len() {
            return;
        }

        // If closing the active project and there's a paste overlay, commit it first.
        if index == self.active_project_index && self.paste_overlay.is_some() {
            self.commit_paste_overlay();
        }

        let project = &self.projects[index];
        if project.is_dirty {
            // Defer: show unsaved-changes dialog
            self.pending_close_index = Some(index);
            return;
        }

        self.projects.remove(index);

        // Adjust active index
        if self.projects.is_empty() {
            // Create a new untitled project if all are closed
            self.untitled_counter += 1;
            let project = Project::new_untitled(self.untitled_counter, 800, 600);
            self.projects.push(project);
            self.active_project_index = 0;
        } else if self.active_project_index >= self.projects.len() {
            self.active_project_index = self.projects.len() - 1;
        } else if index < self.active_project_index {
            self.active_project_index -= 1;
        }

        self.canvas.reset_zoom();
    }

    /// Close a project by index unconditionally (no dirty check).
    /// Used after the user has confirmed they want to discard changes.
    fn force_close_project(&mut self, index: usize) {
        if index >= self.projects.len() {
            return;
        }
        if index == self.active_project_index && self.paste_overlay.is_some() {
            self.commit_paste_overlay();
        }
        self.projects.remove(index);
        if self.projects.is_empty() {
            self.untitled_counter += 1;
            let project = Project::new_untitled(self.untitled_counter, 800, 600);
            self.projects.push(project);
            self.active_project_index = 0;
        } else if self.active_project_index >= self.projects.len() {
            self.active_project_index = self.projects.len() - 1;
        } else if index < self.active_project_index {
            self.active_project_index -= 1;
        }
        self.canvas.reset_zoom();
    }

    /// Switch to a different project tab
    fn switch_to_project(&mut self, index: usize) {
        if index < self.projects.len() && index != self.active_project_index {
            // Commit any active paste overlay before switching tabs.
            // The overlay belongs to the current project's canvas — switching
            // without committing would leave it orphaned.
            if self.paste_overlay.is_some() {
                self.commit_paste_overlay();
            }
            // Clear selection on the project we're leaving so it doesn't
            // linger as an unremovable ghost when we come back.
            if let Some(project) = self.projects.get_mut(self.active_project_index) {
                project.canvas_state.clear_selection();
            }

            // Clear move-selection drag state
            self.move_sel_dragging = false;
            self.move_sel_last_canvas = None;
            self.is_move_pixels_active = false;

            // Clear GPU layer textures — different project, different layers.
            self.canvas.gpu_clear_layers();

            self.active_project_index = index;
            // Don't reset zoom - preserve per-project view state if desired
        }
    }
}

impl eframe::App for PaintFEApp {
    fn clear_color(&self, _visuals: &egui::Visuals) -> [f32; 4] {
        let c = self.theme.canvas_bg_bottom;
        [
            c.r() as f32 / 255.0,
            c.g() as f32 / 255.0,
            c.b() as f32 / 255.0,
            1.0,
        ]
    }

    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // --- Dynamic window title: "PaintFE - <project name>[*]" ---
        {
            let title = if let Some(project) = self.projects.get(self.active_project_index) {
                let dirty = if project.is_dirty { "*" } else { "" };
                format!("PaintFE - {}{}", project.name, dirty)
            } else {
                "PaintFE".to_string()
            };
            ctx.send_viewport_cmd(egui::ViewportCommand::Title(title));
        }

        // --- Intercept OS window-close button ---
        // If the user hasn't yet confirmed the exit dialog, cancel the close and show it.
        if ctx.input(|i| i.viewport().close_requested()) {
            if self.force_exit {
                // Already confirmed — let eframe proceed with the close.
            } else if self.settings.confirm_on_exit {
                let dirty_count = self.projects.iter().filter(|p| p.is_dirty).count();
                if dirty_count > 0 {
                    ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
                    self.pending_exit = true;
                }
            }
        }

        // --- Sync icon colors with theme (catches settings window, menu toggle, etc.) ---
        let is_dark = matches!(self.theme.mode, crate::theme::ThemeMode::Dark);
        self.assets.update_theme(ctx, is_dark);

        // --- Sync OS window chrome (title bar) with app theme on Windows/macOS ---
        let system_theme = if is_dark {
            egui::SystemTheme::Dark
        } else {
            egui::SystemTheme::Light
        };
        ctx.send_viewport_cmd(egui::ViewportCommand::SetTheme(system_theme));

        // --- Maximize window on first frame (more reliable than viewport builder hint) ---
        if self.first_frame {
            self.first_frame = false;
            ctx.send_viewport_cmd(egui::ViewportCommand::Maximized(true));

            // Open files passed as positional arguments (e.g. right-click → "Open with PaintFE")
            let files = std::mem::take(&mut self.pending_startup_files);
            let current_time = ctx.input(|i| i.time);
            for path in files {
                self.open_file_by_path(path, current_time);
            }
        }

        // --- Single-instance IPC: open files sent from other PaintFE invocations ---
        {
            let current_time = ctx.input(|i| i.time);
            while let Ok(path) = self.ipc_receiver.try_recv() {
                self.open_file_by_path(path, current_time);
                // Bring our window to the foreground (the sender already tried via
                // FindWindow, but this handles the case where it didn't have permission).
                ctx.send_viewport_cmd(egui::ViewportCommand::Focus);
            }
        }

        // --- Auto-save tick ---
        // Saves every open project as a .autosave.pfe in the platform data dir.
        // Controlled by settings.auto_save_minutes (0 = disabled).
        {
            let interval_secs = (self.settings.auto_save_minutes as u64) * 60;
            if interval_secs > 0 && self.last_autosave.elapsed().as_secs() >= interval_secs {
                self.last_autosave = std::time::Instant::now();
                if let Some(dir) = crate::io::autosave_dir() {
                    let _ = std::fs::create_dir_all(&dir);
                    for project in &self.projects {
                        // Sanitize project name into a safe filename component
                        let safe_name: String = project
                            .name
                            .chars()
                            .map(|c| {
                                if c.is_alphanumeric() || c == '-' || c == '_' {
                                    c
                                } else {
                                    '_'
                                }
                            })
                            .collect();
                        let path = dir.join(format!("{}.autosave.pfe", safe_name));
                        let pfe_data = crate::io::build_pfe_v1(&project.canvas_state);
                        let proj_name = project.name.clone();
                        rayon::spawn(move || match crate::io::write_pfe_v1(&pfe_data, &path) {
                            Ok(()) => {
                                crate::logger::write(
                                    "INFO",
                                    &format!(
                                        "Auto-save OK  \"{}\"  →  {}",
                                        proj_name,
                                        path.display()
                                    ),
                                );
                            }
                            Err(e) => {
                                crate::logger::write(
                                    "ERROR",
                                    &format!("Auto-save FAILED for \"{}\": {}", proj_name, e),
                                );
                            }
                        });
                    }
                }
            }
        }

        // --- Poll async filter results ---
        while let Ok(result) = self.filter_receiver.try_recv() {
            self.pending_filter_jobs = self.pending_filter_jobs.saturating_sub(1);
            if self.pending_filter_jobs == 0 {
                self.filter_ops_start_time = None;
                self.filter_status_description.clear();
            }
            // Discard stale live-preview results (token mismatch = superseded by newer job)
            if result.preview_token != 0 && result.preview_token != self.preview_job_token {
                continue;
            }
            if result.project_index < self.projects.len()
                && let Some(project) = self.projects.get_mut(result.project_index)
            {
                let idx = result.layer_idx;
                if idx < project.canvas_state.layers.len() {
                    // Swap original back for "before" snapshot, then install result
                    project.canvas_state.layers[idx].pixels = result.original_pixels;
                    let mut cmd = SingleLayerSnapshotCommand::new_for_layer(
                        result.description,
                        &project.canvas_state,
                        idx,
                    );
                    project.canvas_state.layers[idx].pixels = result.result_pixels;
                    cmd.set_after(&project.canvas_state);
                    project.history.push(Box::new(cmd));
                    project.canvas_state.mark_dirty(None);
                    project.mark_dirty();
                }
            }
        }
        // Request a repaint while filter jobs are pending so polling stays active
        if self.pending_filter_jobs > 0 {
            ctx.request_repaint();
        }

        // --- Poll async script results ---
        while let Ok(msg) = self.script_receiver.try_recv() {
            match msg {
                ScriptMessage::Completed {
                    project_index,
                    layer_idx,
                    original_pixels,
                    result_pixels,
                    width,
                    height,
                    console_output,
                    elapsed_ms,
                    canvas_ops,
                } => {
                    self.script_editor.is_running = false;
                    self.script_editor.progress = None;
                    // Clear spinner status
                    self.pending_filter_jobs = self.pending_filter_jobs.saturating_sub(1);
                    if self.pending_filter_jobs == 0 {
                        self.filter_ops_start_time = None;
                        self.filter_status_description.clear();
                    }
                    for line in console_output {
                        self.script_editor.add_console_line(
                            line,
                            crate::components::script_editor::ConsoleLineKind::Output,
                        );
                    }
                    self.script_editor.add_console_line(
                        format!("Completed in {}ms", elapsed_ms),
                        crate::components::script_editor::ConsoleLineKind::Info,
                    );
                    // Apply result to canvas with undo — route to the project that spawned the script
                    if project_index < self.projects.len()
                        && let Some(project) = self.projects.get_mut(project_index)
                        && layer_idx < project.canvas_state.layers.len()
                    {
                        // Restore original pixels so snapshot captures the true before-state
                        project.canvas_state.layers[layer_idx].pixels = original_pixels;

                        if canvas_ops.is_empty() {
                            // Layer-only script: lightweight single-layer snapshot
                            let mut cmd = SingleLayerSnapshotCommand::new_for_layer(
                                "Script".to_string(),
                                &project.canvas_state,
                                layer_idx,
                            );
                            let result_tiled =
                                TiledImage::from_raw_rgba(width, height, &result_pixels);
                            project.canvas_state.layers[layer_idx].pixels = result_tiled;
                            cmd.set_after(&project.canvas_state);
                            project.history.push(Box::new(cmd));
                        } else {
                            // Canvas-wide transform: full snapshot (dimensions may change)
                            let mut cmd =
                                SnapshotCommand::new("Script".to_string(), &project.canvas_state);

                            // Apply result to active layer
                            let result_tiled =
                                TiledImage::from_raw_rgba(width, height, &result_pixels);
                            project.canvas_state.layers[layer_idx].pixels = result_tiled;

                            // Replay canvas ops on all other layers via shared helper
                            // (also updates state.width / state.height to final dims)
                            apply_canvas_ops(&mut project.canvas_state, layer_idx, &canvas_ops);

                            project.canvas_state.composite_cache = None;
                            project.canvas_state.clear_preview_state();
                            cmd.set_after(&project.canvas_state);
                            project.history.push(Box::new(cmd));
                        }

                        project.canvas_state.mark_dirty(None);
                        project.mark_dirty();
                    }
                    self.script_original_pixels = None;
                }
                ScriptMessage::Error {
                    error,
                    console_output,
                } => {
                    self.script_editor.is_running = false;
                    self.script_editor.progress = None;
                    // Clear spinner status
                    self.pending_filter_jobs = self.pending_filter_jobs.saturating_sub(1);
                    if self.pending_filter_jobs == 0 {
                        self.filter_ops_start_time = None;
                        self.filter_status_description.clear();
                    }
                    for line in console_output {
                        self.script_editor.add_console_line(
                            line,
                            crate::components::script_editor::ConsoleLineKind::Output,
                        );
                    }
                    // Show error with tips
                    let friendly = error.friendly_message();
                    for err_line in friendly.lines() {
                        self.script_editor.add_console_line(
                            err_line.to_string(),
                            crate::components::script_editor::ConsoleLineKind::Error,
                        );
                    }
                    // Restore original layer pixels if preview had modified them
                    if let Some((proj_idx, layer_idx, original)) =
                        self.script_original_pixels.take()
                        && let Some(project) = self.projects.get_mut(proj_idx)
                        && layer_idx < project.canvas_state.layers.len()
                    {
                        project.canvas_state.layers[layer_idx].pixels = original;
                        project.canvas_state.mark_dirty(None);
                    }
                    // Highlight the error line in the code editor
                    self.script_editor.error_line = error.line;
                    // Auto-expand console so user sees the error
                    self.script_editor.console_expanded = true;
                }
                ScriptMessage::ConsoleOutput(line) => {
                    self.script_editor.add_console_line(
                        line,
                        crate::components::script_editor::ConsoleLineKind::Output,
                    );
                }
                ScriptMessage::Progress(p) => {
                    self.script_editor.progress = Some(p);
                }
                ScriptMessage::Preview {
                    project_index,
                    pixels,
                    width,
                    height,
                } => {
                    if self.script_editor.live_preview
                        && let Some(project) = self.projects.get_mut(project_index)
                    {
                        let layer_idx = project.canvas_state.active_layer_index;
                        if layer_idx < project.canvas_state.layers.len() {
                            let preview_tiled = TiledImage::from_raw_rgba(width, height, &pixels);
                            project.canvas_state.layers[layer_idx].pixels = preview_tiled;
                            project.canvas_state.mark_dirty(None);
                        }
                    }
                }
            }
        }
        if self.script_editor.is_running {
            ctx.request_repaint();
        }

        // --- Poll async IO results (image load / save) ---
        while let Ok(result) = self.io_receiver.try_recv() {
            self.pending_io_ops = self.pending_io_ops.saturating_sub(1);
            if self.pending_io_ops == 0 {
                self.io_ops_start_time = None;
            }
            match result {
                IoResult::ImageLoaded {
                    tiled,
                    width,
                    height,
                    path,
                    format,
                } => {
                    let mut canvas_state = CanvasState::new(width, height);
                    if let Some(layer) = canvas_state.layers.first_mut() {
                        layer.pixels = tiled;
                    }
                    canvas_state.composite_cache = None;
                    canvas_state.mark_dirty(None);

                    let file_handler = FileHandler {
                        current_path: Some(path.clone()),
                        last_format: format,
                        last_quality: 90,
                        last_tiff_compression: TiffCompression::None,
                        last_animated: false,
                        last_animation_fps: 10.0,
                        last_gif_colors: 256,
                        last_gif_dither: true,
                    };
                    let project = Project::from_file(path, canvas_state, file_handler);
                    self.projects.push(project);
                    self.active_project_index = self.projects.len() - 1;
                    self.canvas.reset_zoom();
                    // Clear GPU layer cache for the new project
                    self.canvas.gpu_clear_layers();
                    self.maybe_close_initial_blank();
                }
                IoResult::LoadFailed(msg) => {
                    eprintln!("Failed to open image: {}", msg);
                }
                IoResult::SaveComplete {
                    project_index,
                    path,
                    format,
                    quality,
                    tiff_compression,
                    update_project_path,
                } => {
                    if let Some(project) = self.projects.get_mut(project_index) {
                        project.file_handler.current_path = Some(path.clone());
                        project.file_handler.last_format = format;
                        project.file_handler.last_quality = quality;
                        project.file_handler.last_tiff_compression = tiff_compression;
                        if update_project_path {
                            project.path = Some(path);
                            project.update_name_from_path();
                        }
                        project.mark_clean();
                    }
                }
                IoResult::SaveFailed {
                    project_index: _,
                    error,
                } => {
                    eprintln!("Failed to save: {}", error);
                }
                IoResult::AnimatedLoaded {
                    tiled,
                    width,
                    height,
                    path,
                    format,
                    fps,
                    frame_count: _,
                } => {
                    let mut canvas_state = CanvasState::new(width, height);
                    if let Some(layer) = canvas_state.layers.first_mut() {
                        layer.pixels = tiled;
                        layer.name = "Frame 1".to_string();
                    }
                    canvas_state.composite_cache = None;
                    canvas_state.mark_dirty(None);

                    let file_handler = FileHandler {
                        current_path: Some(path.clone()),
                        last_format: format,
                        last_quality: 90,
                        last_tiff_compression: TiffCompression::None,
                        last_animated: true,
                        last_animation_fps: fps,
                        last_gif_colors: 256,
                        last_gif_dither: true,
                    };
                    let mut project = Project::from_file(path, canvas_state, file_handler);
                    project.was_animated = true;
                    project.animation_fps = fps;
                    self.projects.push(project);
                    self.active_project_index = self.projects.len() - 1;
                    self.canvas.reset_zoom();
                    self.canvas.gpu_clear_layers();
                    self.maybe_close_initial_blank();
                }
                IoResult::AnimatedFramesLoaded { path, frames } => {
                    // Find the project that was created by the AnimatedLoaded handler
                    if let Some(project) = self
                        .projects
                        .iter_mut()
                        .find(|p| p.path.as_deref() == Some(path.as_path()))
                    {
                        for (tiled, name) in frames {
                            let layer = Layer {
                                name,
                                visible: true,
                                opacity: 1.0,
                                blend_mode: BlendMode::Normal,
                                pixels: tiled,
                                lod_cache: None,
                                gpu_generation: 0,
                            };
                            project.canvas_state.layers.push(layer);
                        }
                        project.canvas_state.composite_cache = None;
                        project.canvas_state.mark_dirty(None);
                        self.canvas.gpu_clear_layers();
                    }
                }
                IoResult::PfeLoaded {
                    mut canvas_state,
                    path,
                } => {
                    canvas_state.composite_cache = None;
                    canvas_state.mark_dirty(None);

                    let file_handler = FileHandler {
                        current_path: Some(path.clone()),
                        last_format: SaveFormat::Pfe,
                        last_quality: 90,
                        last_tiff_compression: TiffCompression::None,
                        last_animated: false,
                        last_animation_fps: 10.0,
                        last_gif_colors: 256,
                        last_gif_dither: true,
                    };
                    let project = Project::from_file(path, canvas_state, file_handler);
                    self.projects.push(project);
                    self.active_project_index = self.projects.len() - 1;
                    self.canvas.reset_zoom();
                    self.canvas.gpu_clear_layers();
                    self.maybe_close_initial_blank();
                }
            }
        }
        if self.pending_io_ops > 0 {
            ctx.request_repaint();
        }

        // --- Drag-and-Drop: open dropped image files as new projects ---
        {
            let dropped: Vec<egui::DroppedFile> = ctx.input(|i| i.raw.dropped_files.clone());
            for file in dropped {
                if let Some(path) = file.path {
                    let ext = path
                        .extension()
                        .map(|e| e.to_string_lossy().to_lowercase())
                        .unwrap_or_default();
                    let supported = matches!(
                        ext.as_str(),
                        "png"
                            | "jpg"
                            | "jpeg"
                            | "bmp"
                            | "gif"
                            | "webp"
                            | "tiff"
                            | "tif"
                            | "tga"
                            | "ico"
                            | "pfe"
                    );
                    if supported {
                        self.open_file_by_path(path, ctx.input(|i| i.time));
                    }
                }
            }
        }

        // Determine if a modal dialog is open — block all shortcuts and canvas interaction.
        let modal_open = self.save_file_dialog.open
            || self.new_file_dialog.open
            || !matches!(self.active_dialog, ActiveDialog::None);

        // Handle scroll wheel zoom — only when mouse is over the canvas and NOT over a widget

        let mut should_zoom = false;
        let mut zoom_amount = 0.0;

        // Check if any floating window/widget is under the pointer
        let pointer_over_widget = ctx.is_pointer_over_area();

        if !modal_open {
            ctx.input_mut(|i| {
                if i.scroll_delta.y.abs() > 0.1 {
                    let mouse_over_canvas = i.pointer.hover_pos().is_some_and(|pos| {
                        self.canvas
                            .last_canvas_rect
                            .is_some_and(|rect| rect.contains(pos))
                    });
                    if mouse_over_canvas && !pointer_over_widget {
                        should_zoom = true;
                        zoom_amount = i.scroll_delta.y;
                        i.scroll_delta.y = 0.0;
                    }
                }
            });
        }

        if should_zoom {
            let zoom_factor = 1.0 + zoom_amount * 0.005;
            // Zoom around the mouse cursor so the point under the pointer stays fixed
            let mouse_pos = ctx.input(|i| i.pointer.hover_pos());
            if let (Some(pos), Some(rect)) = (mouse_pos, self.canvas.last_canvas_rect) {
                self.canvas.zoom_around_screen_point(zoom_factor, pos, rect);
            } else {
                self.canvas.apply_zoom(zoom_factor);
            }
        }

        // --- Selection Keyboard Shortcuts ---
        if !modal_open {
            let delete_pressed = ctx.input(|i| i.key_pressed(egui::Key::Delete));
            let backspace_pressed = ctx.input(|i| i.key_pressed(egui::Key::Backspace));

            if delete_pressed {
                // Delete selected pixels (make transparent) on active layer
                let has_sel = self
                    .active_project()
                    .is_some_and(|p| p.canvas_state.has_selection());
                if has_sel {
                    self.do_snapshot_op("Delete Selection", |s| {
                        s.delete_selected_pixels();
                    });
                }
            }

            if backspace_pressed {
                // Fill selected area with primary colour on active layer
                let has_sel = self
                    .active_project()
                    .is_some_and(|p| p.canvas_state.has_selection());
                if has_sel {
                    let pc = self.colors_panel.get_primary_color();
                    let fill = image::Rgba([pc.r(), pc.g(), pc.b(), pc.a()]);
                    self.do_snapshot_op("Fill Selection", |s| {
                        s.fill_selected_pixels(fill);
                    });
                }
            }
        }

        // --- Keyboard Shortcuts (uses editable keybindings system) ---
        //
        // The KeyBindings::is_pressed method uses consume_key internally,
        // so egui will NOT forward consumed keys to text widgets.
        // Clone keybindings to avoid borrow conflicts with &mut self methods.
        if !modal_open {
            use crate::assets::BindableAction;
            let ctrl = ctx.input(|i| i.modifiers.command);
            let kb = self.settings.keybindings.clone();

            // NOTE: Check Ctrl+Shift combos before plain Ctrl combos
            // so that e.g. Ctrl+Shift+S is not consumed as Ctrl+S.

            // Ctrl+Shift+S — Save As
            if kb.is_pressed(ctx, BindableAction::SaveAs) {
                // Trigger Save As dialog (mirrors File > Save As menu logic)
                let save_as_data = if self.active_project_index < self.projects.len() {
                    let project = &self.projects[self.active_project_index];
                    let composite = project.canvas_state.composite();
                    let frame_images: Option<Vec<image::RgbaImage>> =
                        if project.canvas_state.layers.len() > 1 {
                            Some(
                                project
                                    .canvas_state
                                    .layers
                                    .iter()
                                    .map(|l| l.pixels.to_rgba_image())
                                    .collect(),
                            )
                        } else {
                            None
                        };
                    let was_animated = project.was_animated;
                    let animation_fps = project.animation_fps;
                    let path = project.path.clone();
                    Some((composite, frame_images, was_animated, animation_fps, path))
                } else {
                    None
                };
                self.save_file_dialog.reset();
                if let Some((composite, frame_images, was_animated, animation_fps, path)) =
                    save_as_data
                {
                    self.save_file_dialog.set_source_image(&composite);
                    if let Some(frames) = frame_images.as_ref() {
                        self.save_file_dialog.set_source_animated(
                            frames,
                            was_animated,
                            animation_fps,
                        );
                    }
                    if let Some(ref p) = path {
                        self.save_file_dialog.set_from_path(p);
                    }
                }
                self.save_file_dialog.open = true;
            }

            // Ctrl+Shift+F — Flatten All Layers
            if kb.is_pressed(ctx, BindableAction::FlattenLayers) {
                self.do_snapshot_op("Flatten All Layers", |s| {
                    crate::ops::transform::flatten_image(s);
                });
            }

            // Ctrl+R — Resize Image
            if kb.is_pressed(ctx, BindableAction::ResizeImage)
                && let Some(project) = self.active_project()
            {
                self.active_dialog = ActiveDialog::ResizeImage(
                    crate::ops::dialogs::ResizeImageDialog::new(&project.canvas_state),
                );
            }

            // Ctrl+Shift+R — Resize Canvas
            if kb.is_pressed(ctx, BindableAction::ResizeCanvas)
                && let Some(project) = self.active_project()
            {
                self.active_dialog = ActiveDialog::ResizeCanvas(
                    crate::ops::dialogs::ResizeCanvasDialog::new(&project.canvas_state),
                );
            }

            // Ctrl++ — Zoom In
            if kb.is_pressed(ctx, BindableAction::ViewZoomIn) {
                self.canvas.zoom_in();
            }

            // Ctrl+- — Zoom Out
            if kb.is_pressed(ctx, BindableAction::ViewZoomOut) {
                self.canvas.zoom_out();
            }

            // Ctrl+0 — Fit to Window
            if kb.is_pressed(ctx, BindableAction::ViewFitToWindow) {
                self.canvas.reset_zoom();
            }

            // Ctrl+N — New File
            if kb.is_pressed(ctx, BindableAction::NewFile) {
                self.new_file_dialog.load_clipboard_dimensions();
                self.new_file_dialog.open = true;
            }

            // Ctrl+O — Open File
            if kb.is_pressed(ctx, BindableAction::OpenFile) {
                self.handle_open_file(ctx.input(|i| i.time));
            }

            // Ctrl+S — Save
            if kb.is_pressed(ctx, BindableAction::Save) {
                self.handle_save(ctx.input(|i| i.time));
            }

            // Ctrl+Alt+S — Save All
            if kb.is_pressed(ctx, BindableAction::SaveAll) {
                self.handle_save_all(ctx.input(|i| i.time));
            }

            // Ctrl+Z — Undo
            if kb.is_pressed(ctx, BindableAction::Undo) {
                if self.paste_overlay.is_some() {
                    self.cancel_paste_overlay();
                    if let Some(project) = self.active_project_mut() {
                        project.canvas_state.clear_selection();
                    }
                } else if self.tools_panel.has_active_tool_preview() {
                    // Cancel in-progress tool operation instead of undoing
                    if let Some(project) = self.projects.get_mut(self.active_project_index) {
                        self.tools_panel
                            .cancel_active_tool(&mut project.canvas_state);
                    }
                } else if let Some(project) = self.active_project_mut() {
                    project.canvas_state.clear_selection();
                    project.history.undo(&mut project.canvas_state);
                }
            }

            // Ctrl+Y — Redo
            if kb.is_pressed(ctx, BindableAction::Redo)
                && let Some(project) = self.active_project_mut()
            {
                project.history.redo(&mut project.canvas_state);
            }

            // Ctrl+C — Copy
            if kb.is_pressed(ctx, BindableAction::Copy)
                && let Some(project) = self.active_project()
            {
                crate::ops::clipboard::copy_selection(&project.canvas_state);
            }

            // Ctrl+X — Cut
            if kb.is_pressed(ctx, BindableAction::Cut) {
                let has_sel = self
                    .active_project()
                    .is_some_and(|p| p.canvas_state.has_selection());
                if has_sel {
                    self.do_snapshot_op("Cut Selection", |s| {
                        crate::ops::clipboard::cut_selection(s);
                    });
                }
            }

            // Ctrl+V — Paste
            if kb.is_pressed(ctx, BindableAction::Paste) {
                if self.paste_overlay.is_some() {
                    self.commit_paste_overlay();
                }
                // Compute cursor canvas position before mutable borrow
                let cursor_canvas = if self.active_project_index < self.projects.len() {
                    ctx.input(|i| i.pointer.latest_pos())
                        .and_then(|screen_pos| {
                            self.canvas.last_canvas_rect.and_then(|rect| {
                                let state = &self.projects[self.active_project_index].canvas_state;
                                self.canvas
                                    .screen_to_canvas_f32_pub(screen_pos, rect, state)
                            })
                        })
                } else {
                    None
                };
                if let Some(project) = self.active_project_mut() {
                    let (cw, ch) = (project.canvas_state.width, project.canvas_state.height);
                    let img = crate::ops::clipboard::get_from_system_clipboard()
                        .or_else(crate::ops::clipboard::get_clipboard_image_pub);
                    if let Some(src) = img {
                        let overlay = if let Some((cx, cy)) = cursor_canvas {
                            PasteOverlay::from_image_at(src, cw, ch, egui::Pos2::new(cx, cy))
                        } else {
                            PasteOverlay::from_image(src, cw, ch)
                        };
                        project.canvas_state.clear_selection();
                        self.paste_overlay = Some(overlay);
                        self.canvas.open_paste_menu = true;
                    }
                }
            }

            // Enter — Commit paste (not rebindable)
            if ctx.input(|i| i.key_pressed(egui::Key::Enter)) && self.paste_overlay.is_some() {
                self.commit_paste_overlay();
            }
            // Escape — Cancel paste (not rebindable)
            if ctx.input(|i| i.key_pressed(egui::Key::Escape)) && self.paste_overlay.is_some() {
                self.cancel_paste_overlay();
            }

            // Tab — Center active transform on canvas (paste/move-pixels, placed shape, line edit)
            let tab_applies = self.paste_overlay.is_some()
                || (self.tools_panel.active_tool == crate::components::tools::Tool::Shapes
                    && self.tools_panel.shapes_state.placed.is_some())
                || (self.tools_panel.active_tool == crate::components::tools::Tool::Line
                    && self.tools_panel.line_state.line_tool.stage
                        == crate::components::tools::LineStage::Editing);
            if tab_applies && ctx.input(|i| i.key_pressed(egui::Key::Tab)) {
                let (cw, ch) = self
                    .active_project()
                    .map(|p| (p.canvas_state.width as f32, p.canvas_state.height as f32))
                    .unwrap_or((0.0, 0.0));
                if cw > 0.0 {
                    let cx = cw / 2.0;
                    let cy = ch / 2.0;
                    if let Some(ref mut overlay) = self.paste_overlay {
                        overlay.center = egui::Pos2::new(cx, cy);
                    } else if self.tools_panel.active_tool == crate::components::tools::Tool::Shapes
                    {
                        if let Some(ref mut placed) = self.tools_panel.shapes_state.placed {
                            placed.cx = cx;
                            placed.cy = cy;
                        }
                        // Re-rasterize the shape preview at its new position.
                        let primary = self.colors_panel.get_primary_color_f32();
                        let secondary = self.colors_panel.get_secondary_color_f32();
                        let canvas_ptr: Option<*mut crate::canvas::CanvasState> = self
                            .active_project_mut()
                            .map(|p| &mut p.canvas_state as *mut _);
                        if let Some(ptr) = canvas_ptr {
                            // SAFETY: same-frame, single-threaded, no other borrow active.
                            let canvas = unsafe { &mut *ptr };
                            self.tools_panel
                                .render_shape_preview(canvas, primary, secondary);
                        }
                    } else {
                        // Line tool — translate control-point bounding box to canvas center
                        let cps = &mut self.tools_panel.line_state.line_tool.control_points;
                        let min_x = cps.iter().map(|p| p.x).fold(f32::INFINITY, f32::min);
                        let max_x = cps.iter().map(|p| p.x).fold(f32::NEG_INFINITY, f32::max);
                        let min_y = cps.iter().map(|p| p.y).fold(f32::INFINITY, f32::min);
                        let max_y = cps.iter().map(|p| p.y).fold(f32::NEG_INFINITY, f32::max);
                        let dx = cx - (min_x + max_x) / 2.0;
                        let dy = cy - (min_y + max_y) / 2.0;
                        for pt in cps.iter_mut() {
                            pt.x = (pt.x + dx).clamp(0.0, cw - 1.0);
                            pt.y = (pt.y + dy).clamp(0.0, ch - 1.0);
                        }
                        // Re-rasterize the bezier preview at its new position.
                        let canvas_ptr: Option<*mut crate::canvas::CanvasState> = self
                            .active_project_mut()
                            .map(|p| &mut p.canvas_state as *mut _);
                        let cps = self.tools_panel.line_state.line_tool.control_points;
                        let last_bounds = self.tools_panel.line_state.line_tool.last_bounds;
                        let pattern = self.tools_panel.line_state.line_tool.options.pattern;
                        let cap = self.tools_panel.line_state.line_tool.options.cap_style;
                        let color = self.tools_panel.properties.color;
                        let new_bounds = self
                            .tools_panel
                            .get_bezier_bounds(cps, cw as u32, ch as u32);
                        self.tools_panel.stroke_tracker.expand_bounds(new_bounds);
                        self.tools_panel.line_state.line_tool.last_bounds = Some(new_bounds);
                        if let Some(ptr) = canvas_ptr {
                            let canvas = unsafe { &mut *ptr };
                            self.tools_panel.rasterize_bezier(
                                canvas,
                                cps,
                                color,
                                pattern,
                                cap,
                                last_bounds,
                            );
                            let dirty = last_bounds.map_or(new_bounds, |lb| lb.union(new_bounds));
                            canvas.mark_preview_changed_rect(dirty);
                        }
                    }
                    if let Some(project) = self.active_project_mut() {
                        project.canvas_state.mark_dirty(None);
                    }
                    ctx.request_repaint();
                }
            }

            // Ctrl+A — Select All (skip when script editor has text focus)
            let script_editor_open = self.window_visibility.script_editor;
            let text_tool_editing = self.tools_panel.active_tool
                == crate::components::tools::Tool::Text
                && self.tools_panel.text_state.is_editing;
            if !script_editor_open
                && !text_tool_editing
                && kb.is_pressed(ctx, BindableAction::SelectAll)
            {
                let is_selection_tool = matches!(
                    self.tools_panel.active_tool,
                    crate::components::tools::Tool::RectangleSelect
                        | crate::components::tools::Tool::EllipseSelect
                        | crate::components::tools::Tool::MovePixels
                        | crate::components::tools::Tool::MoveSelection
                );
                if is_selection_tool && let Some(project) = self.active_project_mut() {
                    let w = project.canvas_state.width;
                    let h = project.canvas_state.height;
                    let mask = image::GrayImage::from_pixel(w, h, image::Luma([255u8]));
                    project.canvas_state.selection_mask = Some(mask);
                    project.canvas_state.invalidate_selection_overlay();
                    project.canvas_state.mark_dirty(None);
                }
            }

            // Ctrl+D — Deselect
            if kb.is_pressed(ctx, BindableAction::Deselect)
                && let Some(project) = self.active_project_mut()
            {
                project.canvas_state.clear_selection();
                project.canvas_state.mark_dirty(None);
            }

            // Arrow keys — Move paste overlay (not rebindable)
            if self.paste_overlay.is_some() {
                let shift = ctx.input(|i| i.modifiers.shift);
                let arrows = [
                    (egui::Key::ArrowUp, 0.0f32, -1.0f32),
                    (egui::Key::ArrowDown, 0.0, 1.0),
                    (egui::Key::ArrowLeft, -1.0, 0.0),
                    (egui::Key::ArrowRight, 1.0, 0.0),
                ];
                for (key, dx_dir, dy_dir) in &arrows {
                    if ctx.input(|i| i.key_pressed(*key))
                        && let Some(ref mut overlay) = self.paste_overlay
                    {
                        let (step_x, step_y) = if shift {
                            let sw = overlay.source.width() as f32 * overlay.scale_x;
                            let sh = overlay.source.height() as f32 * overlay.scale_y;
                            (sw * dx_dir.abs(), sh * dy_dir.abs())
                        } else if ctrl {
                            (100.0, 100.0)
                        } else {
                            (1.0, 1.0)
                        };
                        overlay.center.x += dx_dir * step_x;
                        overlay.center.y += dy_dir * step_y;
                    }
                }
            }

            // Arrow keys — Move selection mask (MoveSelection tool, no paste overlay)
            if self.paste_overlay.is_none()
                && self.tools_panel.active_tool == crate::components::tools::Tool::MoveSelection
            {
                let shift = ctx.input(|i| i.modifiers.shift);
                let arrows = [
                    (egui::Key::ArrowUp, 0i32, -1i32),
                    (egui::Key::ArrowDown, 0, 1),
                    (egui::Key::ArrowLeft, -1, 0),
                    (egui::Key::ArrowRight, 1, 0),
                ];
                for (key, dx_dir, dy_dir) in &arrows {
                    if ctx.input(|i| i.key_pressed(*key))
                        && let Some(project) = self.active_project_mut()
                    {
                        let (step_x, step_y) = if shift {
                            (10, 10)
                        } else if ctrl {
                            (100, 100)
                        } else {
                            (1, 1)
                        };
                        project
                            .canvas_state
                            .translate_selection(dx_dir * step_x, dy_dir * step_y);
                    }
                }
            }

            // Arrow keys — Nudge line tool endpoints while in Editing stage
            // Shift = line bounding-box dimension in move direction (tiling, mirrors paste overlay),
            // Ctrl = 100px, plain = 1px
            if self.paste_overlay.is_none()
                && self.tools_panel.active_tool == crate::components::tools::Tool::Line
                && self.tools_panel.line_state.line_tool.stage
                    == crate::components::tools::LineStage::Editing
            {
                let shift = ctx.input(|i| i.modifiers.shift);
                let arrows = [
                    (egui::Key::ArrowUp, 0.0f32, -1.0f32),
                    (egui::Key::ArrowDown, 0.0, 1.0),
                    (egui::Key::ArrowLeft, -1.0, 0.0),
                    (egui::Key::ArrowRight, 1.0, 0.0),
                ];
                let any_pressed = arrows
                    .iter()
                    .any(|(k, _, _)| ctx.input(|i| i.key_pressed(*k)));
                if any_pressed {
                    // Obtain a raw canvas_state pointer so we can free the mutable
                    // borrow on `self` before calling tools_panel methods.
                    let canvas_ptr: Option<*mut crate::canvas::CanvasState> = self
                        .active_project_mut()
                        .map(|p| &mut p.canvas_state as *mut _);
                    let (cw, ch) = self
                        .active_project_mut()
                        .map(|p| (p.canvas_state.width as f32, p.canvas_state.height as f32))
                        .unwrap_or((0.0, 0.0));

                    // Pre-compute bounding box for Shift tiling step
                    let cps_for_bounds = self.tools_panel.line_state.line_tool.control_points;
                    let (bbox_w, bbox_h) = if shift && cw > 0.0 {
                        let b = self.tools_panel.get_bezier_bounds(
                            cps_for_bounds,
                            cw as u32,
                            ch as u32,
                        );
                        (b.width().max(1.0), b.height().max(1.0))
                    } else {
                        (1.0, 1.0)
                    };

                    if cw > 0.0 {
                        for (key, dx_dir, dy_dir) in &arrows {
                            if ctx.input(|i| i.key_pressed(*key)) {
                                // Shift: move by bounding-box size in that axis (tiling)
                                // Ctrl: 100px, plain: 1px
                                let step_x = if shift {
                                    bbox_w
                                } else if ctrl {
                                    100.0
                                } else {
                                    1.0
                                };
                                let step_y = if shift {
                                    bbox_h
                                } else if ctrl {
                                    100.0
                                } else {
                                    1.0
                                };
                                let dx = dx_dir * step_x;
                                let dy = dy_dir * step_y;

                                // Translate all control points
                                for pt in self
                                    .tools_panel
                                    .line_state
                                    .line_tool
                                    .control_points
                                    .iter_mut()
                                {
                                    pt.x = (pt.x + dx).clamp(0.0, cw - 1.0);
                                    pt.y = (pt.y + dy).clamp(0.0, ch - 1.0);
                                }

                                let cps = self.tools_panel.line_state.line_tool.control_points;
                                let last_bounds = self.tools_panel.line_state.line_tool.last_bounds;
                                let pattern = self.tools_panel.line_state.line_tool.options.pattern;
                                let cap = self.tools_panel.line_state.line_tool.options.cap_style;
                                let color = self.tools_panel.properties.color;
                                let new_bounds = self
                                    .tools_panel
                                    .get_bezier_bounds(cps, cw as u32, ch as u32);
                                self.tools_panel.stroke_tracker.expand_bounds(new_bounds);
                                self.tools_panel.line_state.line_tool.last_bounds =
                                    Some(new_bounds);

                                // SAFETY: canvas_ptr was obtained from self.active_project_mut() in this same
                                // frame, no other code touches canvas_state between these two points, and we
                                // ensure the tools_panel borrow ends before any further project access.
                                if let Some(ptr) = canvas_ptr {
                                    let canvas = unsafe { &mut *ptr };
                                    self.tools_panel.rasterize_bezier(
                                        canvas,
                                        cps,
                                        color,
                                        pattern,
                                        cap,
                                        last_bounds,
                                    );
                                    let dirty =
                                        last_bounds.map_or(new_bounds, |lb| lb.union(new_bounds));
                                    canvas.mark_preview_changed_rect(dirty);
                                }
                                ctx.request_repaint();
                            }
                        }
                    }
                }
            }

            // ================================================================
            // TOOL SWITCHING SHORTCUTS (rebindable single letter keys)
            // Only active when not typing into Text tool
            // ================================================================
            let text_tool_active = self.tools_panel.active_tool
                == crate::components::tools::Tool::Text
                && self.tools_panel.text_state.is_editing;
            if !text_tool_active && self.paste_overlay.is_none() {
                use crate::components::tools::Tool;
                let tool_actions: &[(BindableAction, Tool)] = &[
                    (BindableAction::ToolBrush, Tool::Brush),
                    (BindableAction::ToolEraser, Tool::Eraser),
                    (BindableAction::ToolPencil, Tool::Pencil),
                    (BindableAction::ToolLine, Tool::Line),
                    (BindableAction::ToolGradient, Tool::Gradient),
                    (BindableAction::ToolFill, Tool::Fill),
                    (BindableAction::ToolMagicWand, Tool::MagicWand),
                    (BindableAction::ToolColorPicker, Tool::ColorPicker),
                    (BindableAction::ToolMovePixels, Tool::MovePixels),
                    (BindableAction::ToolRectSelect, Tool::RectangleSelect),
                    (BindableAction::ToolText, Tool::Text),
                    (BindableAction::ToolZoom, Tool::Zoom),
                    (BindableAction::ToolPan, Tool::Pan),
                    (BindableAction::ToolCloneStamp, Tool::CloneStamp),
                    (BindableAction::ToolShapes, Tool::Shapes),
                    (BindableAction::ToolLasso, Tool::Lasso),
                    (BindableAction::ToolColorRemover, Tool::ColorRemover),
                    (BindableAction::ToolMeshWarp, Tool::MeshWarp),
                ];
                for (action, tool) in tool_actions {
                    if kb.is_pressed(ctx, *action) {
                        self.tools_panel.change_tool(*tool);
                        break;
                    }
                }
            }

            // [ / ] — Decrease / Increase brush size
            if !text_tool_active {
                use crate::components::tools::Tool;
                let brush_tool = matches!(
                    self.tools_panel.active_tool,
                    Tool::Brush
                        | Tool::Eraser
                        | Tool::CloneStamp
                        | Tool::ContentAwareBrush
                        | Tool::Liquify
                );
                if brush_tool {
                    if kb.is_pressed(ctx, BindableAction::BrushSizeDecrease) {
                        self.tools_panel.properties.size =
                            (self.tools_panel.properties.size - 1.0).max(1.0);
                    }
                    if kb.is_pressed(ctx, BindableAction::BrushSizeIncrease) {
                        self.tools_panel.properties.size =
                            (self.tools_panel.properties.size + 1.0).min(500.0);
                    }
                }
            }
        }

        // -- Move Pixels tool: activate paste overlay on first click --
        if !modal_open
            && self.tools_panel.active_tool == crate::components::tools::Tool::MovePixels
            && self.paste_overlay.is_none()
        {
            let primary_pressed = ctx.input(|i| i.pointer.primary_pressed());
            let canvas_rect = self.canvas.last_canvas_rect;
            let over_canvas = ctx.input(|i| {
                i.pointer
                    .hover_pos()
                    .is_some_and(|pos| canvas_rect.is_some_and(|r| r.contains(pos)))
            });
            let over_ui = ctx.is_pointer_over_area();

            if primary_pressed && over_canvas && !over_ui {
                // Extract pixels into overlay and blank the source area.
                // Push extraction snapshot immediately — commit will be a separate entry.
                let mut overlay_out: Option<crate::ops::clipboard::PasteOverlay> = None;
                if let Some(project) = self.active_project_mut() {
                    let mut cmd = crate::components::history::SnapshotCommand::new(
                        "Move Pixels".to_string(),
                        &project.canvas_state,
                    );
                    if let Some(overlay) =
                        crate::ops::clipboard::extract_to_overlay(&mut project.canvas_state)
                    {
                        overlay_out = Some(overlay);
                        cmd.set_after(&project.canvas_state);
                        project.history.push(Box::new(cmd));
                    }
                    project.mark_dirty();
                }
                if let Some(overlay) = overlay_out {
                    self.paste_overlay = Some(overlay);
                    self.is_move_pixels_active = true;
                }
            }
        }

        // -- Move Selection tool: drag to translate the selection mask --
        if !modal_open
            && self.tools_panel.active_tool == crate::components::tools::Tool::MoveSelection
            && self.paste_overlay.is_none()
        {
            let primary_pressed = ctx.input(|i| i.pointer.primary_pressed());
            let primary_down = ctx.input(|i| i.pointer.primary_down());
            let primary_released = ctx.input(|i| i.pointer.primary_released());
            let canvas_rect = self.canvas.last_canvas_rect;
            let over_canvas = ctx.input(|i| {
                i.pointer
                    .hover_pos()
                    .is_some_and(|pos| canvas_rect.is_some_and(|r| r.contains(pos)))
            });
            let over_ui = ctx.is_pointer_over_area();

            // Compute current canvas position from mouse (without borrowing self mutably).
            let cur_canvas_pos: Option<(i32, i32)> =
                ctx.input(|i| i.pointer.hover_pos()).and_then(|pos| {
                    canvas_rect.and_then(|rect| {
                        self.canvas
                            .screen_to_canvas_pub(
                                pos,
                                rect,
                                &self.projects[self.active_project_index].canvas_state,
                            )
                            .map(|(x, y)| (x as i32, y as i32))
                    })
                });

            if primary_pressed
                && over_canvas
                && !over_ui
                && let Some(cp) = cur_canvas_pos
            {
                self.move_sel_dragging = true;
                self.move_sel_last_canvas = Some(cp);
            }

            if primary_down
                && self.move_sel_dragging
                && let Some((cx, cy)) = cur_canvas_pos
                && let Some((lx, ly)) = self.move_sel_last_canvas
            {
                let dx = cx - lx;
                let dy = cy - ly;
                if dx != 0 || dy != 0 {
                    if let Some(project) = self.active_project_mut() {
                        project.canvas_state.translate_selection(dx, dy);
                    }
                    self.move_sel_last_canvas = Some((cx, cy));
                }
            }

            if primary_released {
                self.move_sel_dragging = false;
                self.move_sel_last_canvas = None;
            }
        }

        // --- Process Dialogs ---

        // Unsaved Changes Confirmation
        if let Some(close_idx) = self.pending_close_index {
            let name = self
                .projects
                .get(close_idx)
                .map(|p| p.name.clone())
                .unwrap_or_default();
            let mut do_save = false;
            let mut do_discard = false;
            let mut do_cancel = false;
            egui::Window::new("Unsaved Changes")
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .show(ctx, |ui| {
                    ui.label(format!("\"{}\" has unsaved changes.", name));
                    ui.label("Do you want to save before closing?");
                    ui.add_space(8.0);
                    ui.horizontal(|ui| {
                        if ui.button("Save").clicked() {
                            do_save = true;
                        }
                        if ui.button("Don't Save").clicked() {
                            do_discard = true;
                        }
                        if ui.button("Cancel").clicked() {
                            do_cancel = true;
                        }
                    });
                });
            if do_save {
                // Switch to the target tab and open the Save As dialog.
                self.active_project_index = close_idx;
                self.save_file_dialog.open = true;
                // Clear pending — after saving the user can re-close cleanly.
                self.pending_close_index = None;
            }
            if do_discard {
                self.pending_close_index = None;
                self.force_close_project(close_idx);
            }
            if do_cancel {
                self.pending_close_index = None;
            }
        }

        // --- Exit with unsaved projects dialog ---
        if self.pending_exit {
            let dirty_projects: Vec<String> = self
                .projects
                .iter()
                .filter(|p| p.is_dirty)
                .map(|p| p.name.clone())
                .collect();

            let mut do_exit = false;
            let mut do_cancel = false;

            egui::Window::new("Exit PaintFE")
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .min_width(380.0)
                .show(ctx, |ui| {
                    if dirty_projects.is_empty() {
                        // All projects were saved while the dialog was open — just exit.
                        do_exit = true;
                    }

                    if !dirty_projects.is_empty() {
                        const SHOW_MAX: usize = 3;
                        let overflow = dirty_projects.len().saturating_sub(SHOW_MAX);

                        ui.vertical_centered(|ui| {
                            // Header
                            if dirty_projects.len() == 1 {
                                ui.label(format!("\"{}\" has unsaved changes.", dirty_projects[0]));
                            } else {
                                ui.label(format!(
                                    "{} projects have unsaved changes:",
                                    dirty_projects.len()
                                ));
                                ui.add_space(4.0);
                                for name in dirty_projects.iter().take(SHOW_MAX) {
                                    ui.label(format!("\u{2022}  {}", name));
                                }
                                if overflow > 0 {
                                    ui.label(
                                        egui::RichText::new(format!("...and {} more", overflow))
                                            .weak()
                                            .italics(),
                                    );
                                }
                            }

                            ui.add_space(8.0);
                            ui.label("Save your work first, or your changes will be lost.");
                            ui.add_space(12.0);

                            // Warning button — red tint that adapts to light/dark theme.
                            let is_dark = ui.visuals().dark_mode;
                            let (btn_fill, btn_text) = if is_dark {
                                (
                                    egui::Color32::from_rgb(170, 35, 35),
                                    egui::Color32::from_rgb(255, 220, 220),
                                )
                            } else {
                                (egui::Color32::from_rgb(192, 38, 38), egui::Color32::WHITE)
                            };
                            let exit_btn = egui::Button::new(
                                egui::RichText::new("Exit without Saving")
                                    .color(btn_text)
                                    .strong(),
                            )
                            .fill(btn_fill)
                            .min_size(egui::vec2(160.0, 26.0));

                            if ui.add(exit_btn).clicked() {
                                do_exit = true;
                            }
                            ui.add_space(4.0);
                            if ui.button("Cancel").clicked() {
                                do_cancel = true;
                            }

                            ui.add_space(6.0);
                            ui.label(
                                egui::RichText::new(
                                    "Save from File \u{2192} Save, then quit again.  \
                                     Disable this prompt in Settings \u{2192} General.",
                                )
                                .small()
                                .weak(),
                            );
                        });
                    }
                });

            if do_exit {
                self.pending_exit = false;
                self.force_exit = true;
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            }
            if do_cancel {
                self.pending_exit = false;
            }
        }

        // New File Dialog - creates a new project tab
        if let Some((width, height)) = self.new_file_dialog.show(ctx) {
            self.new_project(width, height);
        }

        // Save File Dialog - saves the active project
        if let Some(action) = self.save_file_dialog.show(ctx) {
            let project_index = self.active_project_index;
            if project_index < self.projects.len() {
                if action.format == SaveFormat::Pfe {
                    // Save as .pfe project file — build data snapshot, serialize in background
                    let project = &mut self.projects[project_index];
                    let pfe_data = crate::io::build_pfe_v1(&project.canvas_state);
                    let path = action.path.clone();

                    let sender = self.io_sender.clone();
                    if self.pending_io_ops == 0 {
                        self.io_ops_start_time = Some(ctx.input(|i| i.time));
                    }
                    self.pending_io_ops += 1;

                    rayon::spawn(move || match crate::io::write_pfe_v1(&pfe_data, &path) {
                        Ok(()) => {
                            let _ = sender.send(IoResult::SaveComplete {
                                project_index,
                                path,
                                format: SaveFormat::Pfe,
                                quality: 100,
                                tiff_compression: TiffCompression::None,
                                update_project_path: true,
                            });
                        }
                        Err(e) => {
                            let _ = sender.send(IoResult::SaveFailed {
                                project_index,
                                error: format!("{}", e),
                            });
                        }
                    });
                } else if action.animated && action.format.supports_animation() {
                    // Animated save — composite each layer as a frame (include hidden layers)
                    let project = &mut self.projects[project_index];
                    let frames: Vec<image::RgbaImage> = project
                        .canvas_state
                        .layers
                        .iter()
                        .map(|l| l.pixels.to_rgba_image())
                        .collect();

                    let path = action.path.clone();
                    let format = action.format;
                    let quality = action.quality;
                    let tiff_compression = action.tiff_compression;
                    let fps = action.animation_fps;
                    let gif_colors = action.gif_colors;
                    let gif_dither = action.gif_dither;

                    // Store animation settings on the project for quick-save
                    project.file_handler.last_animated = true;
                    project.file_handler.last_animation_fps = fps;
                    project.file_handler.last_gif_colors = gif_colors;
                    project.file_handler.last_gif_dither = gif_dither;
                    project.was_animated = true;
                    project.animation_fps = fps;

                    let sender = self.io_sender.clone();
                    if self.pending_io_ops == 0 {
                        self.io_ops_start_time = Some(ctx.input(|i| i.time));
                    }
                    self.pending_io_ops += 1;

                    rayon::spawn(move || {
                        let result = match format {
                            SaveFormat::Gif => crate::io::encode_animated_gif(
                                &frames, fps, gif_colors, gif_dither, &path,
                            ),
                            SaveFormat::Png => crate::io::encode_animated_png(&frames, fps, &path),
                            _ => Err("Format does not support animation".to_string()),
                        };
                        match result {
                            Ok(()) => {
                                let _ = sender.send(IoResult::SaveComplete {
                                    project_index,
                                    path,
                                    format,
                                    quality,
                                    tiff_compression,
                                    update_project_path: true,
                                });
                            }
                            Err(e) => {
                                let _ = sender.send(IoResult::SaveFailed {
                                    project_index,
                                    error: e,
                                });
                            }
                        }
                    });
                } else {
                    // Static image save — encode on background thread
                    let project = &mut self.projects[project_index];
                    let composite = project.canvas_state.composite();
                    let path = action.path.clone();
                    let format = action.format;
                    let quality = action.quality;
                    let tiff_compression = action.tiff_compression;

                    // Clear animation state for non-animated saves
                    project.file_handler.last_animated = false;

                    let sender = self.io_sender.clone();
                    if self.pending_io_ops == 0 {
                        self.io_ops_start_time = Some(ctx.input(|i| i.time));
                    }
                    self.pending_io_ops += 1;

                    rayon::spawn(move || {
                        match crate::io::encode_and_write(
                            &composite,
                            &path,
                            format,
                            quality,
                            tiff_compression,
                        ) {
                            Ok(()) => {
                                let _ = sender.send(IoResult::SaveComplete {
                                    project_index,
                                    path,
                                    format,
                                    quality,
                                    tiff_compression,
                                    update_project_path: true,
                                });
                            }
                            Err(e) => {
                                let _ = sender.send(IoResult::SaveFailed {
                                    project_index,
                                    error: format!("{}", e),
                                });
                            }
                        }
                    });
                }
            }
        }

        // Settings Window
        self.settings_window
            .show(ctx, &mut self.settings, &mut self.theme, &self.assets);
        // Re-probe ONNX availability only when paths change
        let current_paths = (
            self.settings.onnx_runtime_path.clone(),
            self.settings.birefnet_model_path.clone(),
        );
        if current_paths != self.onnx_last_probed_paths {
            self.onnx_last_probed_paths = current_paths;
            self.onnx_available = if !self.settings.onnx_runtime_path.is_empty()
                && !self.settings.birefnet_model_path.is_empty()
            {
                crate::ops::ai::probe_onnx_runtime(&self.settings.onnx_runtime_path).is_ok()
                    && std::path::Path::new(&self.settings.birefnet_model_path).exists()
            } else {
                false
            };
        }

        // --- Process Modal Dialogs ---
        self.process_active_dialog(ctx);

        // --- Top Menu Bar ---
        let menu_kb = self.settings.keybindings.clone();
        egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                ui.menu_button(t!("menu.file"), |ui| {
                    if self
                        .assets
                        .menu_item_shortcut(
                            ui,
                            Icon::MenuFileNew,
                            &t!("menu.file.new"),
                            &menu_kb,
                            BindableAction::NewFile,
                        )
                        .clicked()
                    {
                        self.new_file_dialog.load_clipboard_dimensions();
                        self.new_file_dialog.open = true;
                        ui.close_menu();
                    }
                    if self
                        .assets
                        .menu_item_shortcut(
                            ui,
                            Icon::MenuFileOpen,
                            &t!("menu.file.open"),
                            &menu_kb,
                            BindableAction::OpenFile,
                        )
                        .clicked()
                    {
                        self.handle_open_file(ctx.input(|i| i.time));
                        ui.close_menu();
                    }
                    ui.separator();
                    if self
                        .assets
                        .menu_item_shortcut(
                            ui,
                            Icon::MenuFileSave,
                            &t!("menu.file.save"),
                            &menu_kb,
                            BindableAction::Save,
                        )
                        .clicked()
                    {
                        self.handle_save(ctx.input(|i| i.time));
                        ui.close_menu();
                    }
                    {
                        let any_dirty =
                            self.projects.iter().any(|p| p.is_dirty && p.path.is_some());
                        if self
                            .assets
                            .menu_item_shortcut_enabled(
                                ui,
                                Icon::MenuFileSaveAll,
                                &t!("menu.file.save_all"),
                                any_dirty,
                                &menu_kb,
                                BindableAction::SaveAll,
                            )
                            .clicked()
                        {
                            self.handle_save_all(ctx.input(|i| i.time));
                            ui.close_menu();
                        }
                    }
                    if self
                        .assets
                        .menu_item_shortcut(
                            ui,
                            Icon::MenuFileSaveAs,
                            &t!("menu.file.save_as"),
                            &menu_kb,
                            BindableAction::SaveAs,
                        )
                        .clicked()
                    {
                        // Extract data from project before mutating save_file_dialog
                        let save_as_data = if self.active_project_index < self.projects.len() {
                            let project = &self.projects[self.active_project_index];
                            let composite = project.canvas_state.composite();
                            let frame_images: Option<Vec<image::RgbaImage>> =
                                if project.canvas_state.layers.len() > 1 {
                                    Some(
                                        project
                                            .canvas_state
                                            .layers
                                            .iter()
                                            .map(|l| l.pixels.to_rgba_image())
                                            .collect(),
                                    )
                                } else {
                                    None
                                };
                            let was_animated = project.was_animated;
                            let animation_fps = project.animation_fps;
                            let path = project.path.clone();
                            Some((composite, frame_images, was_animated, animation_fps, path))
                        } else {
                            None
                        };

                        self.save_file_dialog.reset();
                        if let Some((composite, frame_images, was_animated, animation_fps, path)) =
                            save_as_data
                        {
                            self.save_file_dialog.set_source_image(&composite);
                            if let Some(frames) = frame_images.as_ref() {
                                self.save_file_dialog.set_source_animated(
                                    frames,
                                    was_animated,
                                    animation_fps,
                                );
                            }
                            if let Some(ref p) = path {
                                self.save_file_dialog.set_from_path(p);
                            }
                        }
                        self.save_file_dialog.open = true;
                        ui.close_menu();
                    }
                    ui.separator();
                    if self
                        .assets
                        .menu_item(ui, Icon::MenuFilePrint, &t!("menu.file.print"))
                        .clicked()
                    {
                        if let Some(project) = self.active_project() {
                            let composite = project.canvas_state.composite();
                            if let Err(e) = crate::ops::print::print_image(&composite) {
                                eprintln!("Print error: {}", e);
                            }
                        }
                        ui.close_menu();
                    }
                    ui.separator();
                    if self
                        .assets
                        .menu_item(ui, Icon::MenuFileQuit, &t!("menu.file.quit"))
                        .clicked()
                    {
                        ui.close_menu();
                        let dirty_count = self.projects.iter().filter(|p| p.is_dirty).count();
                        if self.settings.confirm_on_exit && dirty_count > 0 {
                            self.pending_exit = true;
                        } else {
                            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                        }
                    }
                });

                ui.menu_button(t!("menu.edit"), |ui| {
                    let can_undo = self.active_project().is_some_and(|p| p.history.can_undo());
                    let can_redo = self.active_project().is_some_and(|p| p.history.can_redo());

                    if self
                        .assets
                        .menu_item_shortcut_enabled(
                            ui,
                            Icon::MenuEditUndo,
                            &t!("menu.edit.undo"),
                            can_undo,
                            &menu_kb,
                            BindableAction::Undo,
                        )
                        .clicked()
                    {
                        if self.paste_overlay.is_some() {
                            self.cancel_paste_overlay();
                            if let Some(project) = self.active_project_mut() {
                                project.canvas_state.clear_selection();
                            }
                        } else if self.tools_panel.has_active_tool_preview() {
                            if let Some(project) = self.projects.get_mut(self.active_project_index)
                            {
                                self.tools_panel
                                    .cancel_active_tool(&mut project.canvas_state);
                            }
                        } else if let Some(project) = self.active_project_mut() {
                            project.canvas_state.clear_selection();
                            project.history.undo(&mut project.canvas_state);
                        }
                        ui.close_menu();
                    }
                    if self
                        .assets
                        .menu_item_shortcut_enabled(
                            ui,
                            Icon::MenuEditRedo,
                            &t!("menu.edit.redo"),
                            can_redo,
                            &menu_kb,
                            BindableAction::Redo,
                        )
                        .clicked()
                    {
                        if let Some(project) = self.active_project_mut() {
                            project.history.redo(&mut project.canvas_state);
                        }
                        ui.close_menu();
                    }

                    ui.separator();

                    let has_sel = self
                        .active_project()
                        .is_some_and(|p| p.canvas_state.has_selection());
                    let has_clip = crate::ops::clipboard::has_clipboard_image();

                    if self
                        .assets
                        .menu_item_shortcut_enabled(
                            ui,
                            Icon::MenuEditCut,
                            &t!("menu.edit.cut"),
                            has_sel,
                            &menu_kb,
                            BindableAction::Cut,
                        )
                        .clicked()
                    {
                        self.do_snapshot_op("Cut Selection", |s| {
                            crate::ops::clipboard::cut_selection(s);
                        });
                        ui.close_menu();
                    }
                    if self
                        .assets
                        .menu_item_shortcut_enabled(
                            ui,
                            Icon::MenuEditCopy,
                            &t!("menu.edit.copy"),
                            has_sel,
                            &menu_kb,
                            BindableAction::Copy,
                        )
                        .clicked()
                    {
                        if let Some(project) = self.active_project() {
                            crate::ops::clipboard::copy_selection(&project.canvas_state);
                        }
                        ui.close_menu();
                    }
                    if self
                        .assets
                        .menu_item_shortcut_enabled(
                            ui,
                            Icon::MenuEditPaste,
                            &t!("menu.edit.paste"),
                            has_clip,
                            &menu_kb,
                            BindableAction::Paste,
                        )
                        .clicked()
                    {
                        // Commit existing paste first.
                        if self.paste_overlay.is_some() {
                            self.commit_paste_overlay();
                        }
                        if let Some(project) = self.active_project_mut() {
                            let (cw, ch) =
                                (project.canvas_state.width, project.canvas_state.height);
                            if let Some(overlay) = PasteOverlay::from_clipboard(cw, ch) {
                                project.canvas_state.clear_selection();
                                self.paste_overlay = Some(overlay);
                                self.canvas.open_paste_menu = true;
                            }
                        }
                        ui.close_menu();
                    }
                    if self
                        .assets
                        .menu_item_enabled(
                            ui,
                            Icon::MenuEditPasteLayer,
                            &t!("menu.edit.paste_as_layer"),
                            has_clip,
                        )
                        .clicked()
                    {
                        // Paste clipboard contents as a new layer
                        let img = crate::ops::clipboard::get_from_system_clipboard()
                            .or_else(crate::ops::clipboard::get_clipboard_image_pub);
                        if let Some(src) = img {
                            self.do_snapshot_op("Paste as New Layer", |s| {
                                crate::ops::adjustments::import_layer_from_image(
                                    s,
                                    &src,
                                    "Pasted Layer",
                                );
                            });
                        }
                        ui.close_menu();
                    }

                    ui.separator();

                    // Selection operations
                    if self
                        .assets
                        .menu_item_shortcut(
                            ui,
                            Icon::MenuEditSelectAll,
                            &t!("menu.edit.select_all"),
                            &menu_kb,
                            BindableAction::SelectAll,
                        )
                        .clicked()
                    {
                        if let Some(project) = self.active_project_mut() {
                            let w = project.canvas_state.width;
                            let h = project.canvas_state.height;
                            let mask = image::GrayImage::from_pixel(w, h, image::Luma([255u8]));
                            project.canvas_state.selection_mask = Some(mask);
                            project.canvas_state.invalidate_selection_overlay();
                            project.canvas_state.mark_dirty(None);
                        }
                        ui.close_menu();
                    }
                    if self
                        .assets
                        .menu_item_shortcut_enabled(
                            ui,
                            Icon::MenuEditDeselect,
                            &t!("menu.edit.deselect"),
                            has_sel,
                            &menu_kb,
                            BindableAction::Deselect,
                        )
                        .clicked()
                    {
                        if let Some(project) = self.active_project_mut() {
                            project.canvas_state.clear_selection();
                            project.canvas_state.mark_dirty(None);
                        }
                        ui.close_menu();
                    }
                    if self
                        .assets
                        .menu_item_enabled(
                            ui,
                            Icon::MenuEditInvertSel,
                            &t!("menu.edit.invert_selection"),
                            has_sel,
                        )
                        .clicked()
                    {
                        if let Some(project) = self.active_project_mut() {
                            let w = project.canvas_state.width;
                            let h = project.canvas_state.height;
                            if let Some(ref mask) = project.canvas_state.selection_mask {
                                let inverted = image::GrayImage::from_fn(w, h, |x, y| {
                                    let v = mask.get_pixel(x, y).0[0];
                                    image::Luma([255u8 - v])
                                });
                                project.canvas_state.selection_mask = Some(inverted);
                            } else {
                                // No selection = select all
                                let mask = image::GrayImage::from_pixel(w, h, image::Luma([255u8]));
                                project.canvas_state.selection_mask = Some(mask);
                            }
                            project.canvas_state.invalidate_selection_overlay();
                            project.canvas_state.mark_dirty(None);
                        }
                        ui.close_menu();
                    }
                    if self
                        .assets
                        .menu_item_enabled(
                            ui,
                            Icon::MenuEditColorRange,
                            "Select Color Range...",
                            true,
                        )
                        .clicked()
                    {
                        if let Some(project) = self.active_project_mut() {
                            let dlg =
                                crate::ops::dialogs::ColorRangeDialog::new(&project.canvas_state);
                            self.active_dialog = crate::ops::dialogs::ActiveDialog::ColorRange(dlg);
                        }
                        ui.close_menu();
                    }

                    ui.separator();

                    if self
                        .assets
                        .menu_item(ui, Icon::MenuEditPreferences, &t!("menu.edit.preferences"))
                        .clicked()
                    {
                        self.settings_window.open = true;
                        ui.close_menu();
                    }
                });

                // ==================== CANVAS MENU (was: Image) ====================
                let no_dialog = !modal_open;
                ui.menu_button(t!("menu.canvas"), |ui| {
                    if self
                        .assets
                        .menu_item_shortcut_below_enabled(
                            ui,
                            Icon::MenuCanvasResize,
                            &t!("menu.canvas.resize_image"),
                            no_dialog,
                            &menu_kb,
                            BindableAction::ResizeImage,
                        )
                        .clicked()
                    {
                        if let Some(project) = self.active_project() {
                            self.active_dialog = ActiveDialog::ResizeImage(
                                crate::ops::dialogs::ResizeImageDialog::new(&project.canvas_state),
                            );
                        }
                        ui.close_menu();
                    }
                    if self
                        .assets
                        .menu_item_shortcut_below_enabled(
                            ui,
                            Icon::MenuCanvasResize,
                            &t!("menu.canvas.resize_canvas"),
                            no_dialog,
                            &menu_kb,
                            BindableAction::ResizeCanvas,
                        )
                        .clicked()
                    {
                        if let Some(project) = self.active_project() {
                            self.active_dialog = ActiveDialog::ResizeCanvas(
                                crate::ops::dialogs::ResizeCanvasDialog::new(&project.canvas_state),
                            );
                        }
                        ui.close_menu();
                    }
                    let has_sel = self
                        .active_project()
                        .is_some_and(|p| p.canvas_state.has_selection());
                    if self
                        .assets
                        .menu_item_enabled(
                            ui,
                            Icon::MenuCanvasCrop,
                            &t!("menu.canvas.crop_to_selection"),
                            has_sel,
                        )
                        .clicked()
                    {
                        self.do_snapshot_op("Crop to Selection", |s| {
                            crate::ops::adjustments::crop_to_selection(s);
                        });
                        ui.close_menu();
                    }
                    ui.separator();
                    ui.menu_button(t!("menu.canvas.flip_canvas"), |ui| {
                        ui.set_min_width(ui.min_rect().width().max(160.0));
                        if self
                            .assets
                            .menu_item(
                                ui,
                                Icon::MenuCanvasFlipH,
                                &t!("menu.canvas.flip_horizontal"),
                            )
                            .clicked()
                        {
                            self.do_snapshot_op("Flip Horizontal", |s| {
                                crate::ops::transform::flip_canvas_horizontal(s);
                            });
                            ui.close_menu();
                        }
                        if self
                            .assets
                            .menu_item(ui, Icon::MenuCanvasFlipV, &t!("menu.canvas.flip_vertical"))
                            .clicked()
                        {
                            self.do_snapshot_op("Flip Vertical", |s| {
                                crate::ops::transform::flip_canvas_vertical(s);
                            });
                            ui.close_menu();
                        }
                    });
                    ui.menu_button(t!("menu.canvas.rotate_canvas"), |ui| {
                        ui.set_min_width(ui.min_rect().width().max(160.0));
                        if self
                            .assets
                            .menu_item(ui, Icon::MenuCanvasRotateCw, &t!("menu.canvas.rotate_90cw"))
                            .clicked()
                        {
                            self.do_snapshot_op("Rotate 90° CW", |s| {
                                crate::ops::transform::rotate_canvas_90cw(s);
                            });
                            ui.close_menu();
                        }
                        if self
                            .assets
                            .menu_item(
                                ui,
                                Icon::MenuCanvasRotateCcw,
                                &t!("menu.canvas.rotate_90ccw"),
                            )
                            .clicked()
                        {
                            self.do_snapshot_op("Rotate 90° CCW", |s| {
                                crate::ops::transform::rotate_canvas_90ccw(s);
                            });
                            ui.close_menu();
                        }
                        if self
                            .assets
                            .menu_item(ui, Icon::MenuCanvasRotate180, &t!("menu.canvas.rotate_180"))
                            .clicked()
                        {
                            self.do_snapshot_op("Rotate 180°", |s| {
                                crate::ops::transform::rotate_canvas_180(s);
                            });
                            ui.close_menu();
                        }
                    });
                    ui.separator();
                    if self
                        .assets
                        .menu_item_shortcut_below(
                            ui,
                            Icon::MenuCanvasFlatten,
                            &t!("menu.canvas.flatten_all_layers"),
                            &menu_kb,
                            BindableAction::FlattenLayers,
                        )
                        .clicked()
                    {
                        self.do_snapshot_op("Flatten All Layers", |s| {
                            crate::ops::transform::flatten_image(s);
                        });
                        ui.close_menu();
                    }
                });

                // (Layers menu removed — layer operations are now in the Layers Panel context menu)

                // ==================== COLOR MENU (was: Adjustments) ====================
                ui.menu_button(t!("menu.color"), |ui| {
                    // --- Instant adjustments (no dialog) ---
                    if self
                        .assets
                        .menu_item(ui, Icon::MenuColorAutoLevels, &t!("menu.color.auto_levels"))
                        .clicked()
                    {
                        self.do_layer_snapshot_op("Auto Levels", |s| {
                            let idx = s.active_layer_index;
                            crate::ops::adjustments::auto_levels(s, idx);
                        });
                        ui.close_menu();
                    }
                    if self
                        .assets
                        .menu_item(ui, Icon::MenuColorDesaturate, &t!("menu.color.desaturate"))
                        .clicked()
                    {
                        self.do_layer_snapshot_op("Desaturate", |s| {
                            let idx = s.active_layer_index;
                            crate::ops::filters::desaturate_layer(s, idx);
                        });
                        ui.close_menu();
                    }
                    if self
                        .assets
                        .menu_item(ui, Icon::MenuColorInvert, &t!("menu.color.invert_colors"))
                        .clicked()
                    {
                        self.do_gpu_snapshot_op("Invert Colors", |s, gpu| {
                            let idx = s.active_layer_index;
                            crate::ops::adjustments::invert_colors_gpu(s, idx, gpu);
                        });
                        ui.close_menu();
                    }
                    if self
                        .assets
                        .menu_item(
                            ui,
                            Icon::MenuColorInvertAlpha,
                            &t!("menu.color.invert_alpha"),
                        )
                        .clicked()
                    {
                        self.do_layer_snapshot_op("Invert Alpha", |s| {
                            let idx = s.active_layer_index;
                            crate::ops::adjustments::invert_alpha(s, idx);
                        });
                        ui.close_menu();
                    }
                    if self
                        .assets
                        .menu_item(ui, Icon::MenuColorSepia, &t!("menu.color.sepia_tone"))
                        .clicked()
                    {
                        self.do_layer_snapshot_op("Sepia Tone", |s| {
                            let idx = s.active_layer_index;
                            crate::ops::adjustments::sepia(s, idx);
                        });
                        ui.close_menu();
                    }
                    ui.separator();
                    // --- Parameterized adjustments (with dialog) ---
                    if self
                        .assets
                        .menu_item_enabled(
                            ui,
                            Icon::MenuColorBrightness,
                            &t!("menu.color.brightness_contrast"),
                            no_dialog,
                        )
                        .clicked()
                    {
                        if let Some(project) = self.active_project() {
                            self.active_dialog = ActiveDialog::BrightnessContrast(
                                crate::ops::dialogs::BrightnessContrastDialog::new(
                                    &project.canvas_state,
                                ),
                            );
                        }
                        ui.close_menu();
                    }
                    if self
                        .assets
                        .menu_item_enabled(
                            ui,
                            Icon::MenuColorCurves,
                            &t!("menu.color.curves"),
                            no_dialog,
                        )
                        .clicked()
                    {
                        if let Some(project) = self.active_project() {
                            self.active_dialog = ActiveDialog::Curves(
                                crate::ops::dialogs::CurvesDialog::new(&project.canvas_state),
                            );
                        }
                        ui.close_menu();
                    }
                    if self
                        .assets
                        .menu_item_enabled(
                            ui,
                            Icon::MenuColorExposure,
                            &t!("menu.color.exposure"),
                            no_dialog,
                        )
                        .clicked()
                    {
                        if let Some(project) = self.active_project() {
                            self.active_dialog = ActiveDialog::Exposure(
                                crate::ops::dialogs::ExposureDialog::new(&project.canvas_state),
                            );
                        }
                        ui.close_menu();
                    }
                    if self
                        .assets
                        .menu_item_enabled(
                            ui,
                            Icon::MenuColorHighlights,
                            &t!("menu.color.highlights_shadows"),
                            no_dialog,
                        )
                        .clicked()
                    {
                        if let Some(project) = self.active_project() {
                            self.active_dialog = ActiveDialog::HighlightsShadows(
                                crate::ops::dialogs::HighlightsShadowsDialog::new(
                                    &project.canvas_state,
                                ),
                            );
                        }
                        ui.close_menu();
                    }
                    if self
                        .assets
                        .menu_item_enabled(
                            ui,
                            Icon::MenuColorHsl,
                            &t!("menu.color.hue_saturation"),
                            no_dialog,
                        )
                        .clicked()
                    {
                        if let Some(project) = self.active_project() {
                            self.active_dialog = ActiveDialog::HueSaturation(
                                crate::ops::dialogs::HueSaturationDialog::new(
                                    &project.canvas_state,
                                ),
                            );
                        }
                        ui.close_menu();
                    }
                    if self
                        .assets
                        .menu_item_enabled(
                            ui,
                            Icon::MenuColorLevels,
                            &t!("menu.color.levels"),
                            no_dialog,
                        )
                        .clicked()
                    {
                        if let Some(project) = self.active_project() {
                            self.active_dialog = ActiveDialog::Levels(
                                crate::ops::dialogs::LevelsDialog::new(&project.canvas_state),
                            );
                        }
                        ui.close_menu();
                    }
                    if self
                        .assets
                        .menu_item_enabled(
                            ui,
                            Icon::MenuColorTemperature,
                            &t!("menu.color.temperature_tint"),
                            no_dialog,
                        )
                        .clicked()
                    {
                        if let Some(project) = self.active_project() {
                            self.active_dialog = ActiveDialog::TemperatureTint(
                                crate::ops::dialogs::TemperatureTintDialog::new(
                                    &project.canvas_state,
                                ),
                            );
                        }
                        ui.close_menu();
                    }
                    ui.separator();
                    // --- Additional color adjustments ---
                    if self
                        .assets
                        .menu_item_enabled(
                            ui,
                            Icon::MenuColorHsl,
                            &t!("menu.color.vibrance"),
                            no_dialog,
                        )
                        .clicked()
                    {
                        if let Some(project) = self.active_project() {
                            self.active_dialog = ActiveDialog::Vibrance(
                                crate::ops::dialogs::VibranceDialog::new(&project.canvas_state),
                            );
                        }
                        ui.close_menu();
                    }
                    if self
                        .assets
                        .menu_item_enabled(
                            ui,
                            Icon::MenuColorInvert,
                            &t!("menu.color.threshold"),
                            no_dialog,
                        )
                        .clicked()
                    {
                        if let Some(project) = self.active_project() {
                            self.active_dialog = ActiveDialog::Threshold(
                                crate::ops::dialogs::ThresholdDialog::new(&project.canvas_state),
                            );
                        }
                        ui.close_menu();
                    }
                    if self
                        .assets
                        .menu_item_enabled(
                            ui,
                            Icon::MenuColorDesaturate,
                            &t!("menu.color.posterize"),
                            no_dialog,
                        )
                        .clicked()
                    {
                        if let Some(project) = self.active_project() {
                            self.active_dialog = ActiveDialog::Posterize(
                                crate::ops::dialogs::PosterizeDialog::new(&project.canvas_state),
                            );
                        }
                        ui.close_menu();
                    }
                    if self
                        .assets
                        .menu_item_enabled(
                            ui,
                            Icon::MenuColorBrightness,
                            &t!("menu.color.color_balance"),
                            no_dialog,
                        )
                        .clicked()
                    {
                        if let Some(project) = self.active_project() {
                            self.active_dialog = ActiveDialog::ColorBalance(
                                crate::ops::dialogs::ColorBalanceDialog::new(&project.canvas_state),
                            );
                        }
                        ui.close_menu();
                    }
                    if self
                        .assets
                        .menu_item_enabled(
                            ui,
                            Icon::MenuColorCurves,
                            &t!("menu.color.gradient_map"),
                            no_dialog,
                        )
                        .clicked()
                    {
                        if let Some(project) = self.active_project() {
                            self.active_dialog = ActiveDialog::GradientMap(
                                crate::ops::dialogs::GradientMapDialog::new(&project.canvas_state),
                            );
                        }
                        ui.close_menu();
                    }
                    if self
                        .assets
                        .menu_item_enabled(
                            ui,
                            Icon::MenuColorDesaturate,
                            &t!("menu.color.black_and_white"),
                            no_dialog,
                        )
                        .clicked()
                    {
                        if let Some(project) = self.active_project() {
                            self.active_dialog = ActiveDialog::BlackAndWhite(
                                crate::ops::dialogs::BlackAndWhiteDialog::new(
                                    &project.canvas_state,
                                ),
                            );
                        }
                        ui.close_menu();
                    }
                });

                // ==================== FILTER MENU (was: Effects) ====================
                ui.menu_button(t!("menu.filter"), |ui| {
                    // -- Blur submenu --
                    ui.menu_button(t!("menu.filter.blur"), |ui| {
                        if self
                            .assets
                            .menu_item_enabled(
                                ui,
                                Icon::MenuFilterGaussian,
                                &t!("menu.filter.blur.gaussian"),
                                no_dialog,
                            )
                            .clicked()
                        {
                            if let Some(project) = self.active_project() {
                                self.active_dialog = ActiveDialog::GaussianBlur(
                                    crate::ops::dialogs::GaussianBlurDialog::new(
                                        &project.canvas_state,
                                    ),
                                );
                            }
                            ui.close_menu();
                        }
                        if self
                            .assets
                            .menu_item_enabled(
                                ui,
                                Icon::MenuFilterBokeh,
                                &t!("menu.filter.blur.bokeh"),
                                no_dialog,
                            )
                            .clicked()
                        {
                            if let Some(project) = self.active_project() {
                                self.active_dialog = ActiveDialog::BokehBlur(
                                    crate::ops::effect_dialogs::BokehBlurDialog::new(
                                        &project.canvas_state,
                                    ),
                                );
                            }
                            ui.close_menu();
                        }
                        if self
                            .assets
                            .menu_item_enabled(
                                ui,
                                Icon::MenuFilterMotionBlur,
                                &t!("menu.filter.blur.motion"),
                                no_dialog,
                            )
                            .clicked()
                        {
                            if let Some(project) = self.active_project() {
                                self.active_dialog = ActiveDialog::MotionBlur(
                                    crate::ops::effect_dialogs::MotionBlurDialog::new(
                                        &project.canvas_state,
                                    ),
                                );
                            }
                            ui.close_menu();
                        }
                        if self
                            .assets
                            .menu_item_enabled(
                                ui,
                                Icon::MenuFilterBoxBlur,
                                &t!("menu.filter.blur.box"),
                                no_dialog,
                            )
                            .clicked()
                        {
                            if let Some(project) = self.active_project() {
                                self.active_dialog = ActiveDialog::BoxBlur(
                                    crate::ops::effect_dialogs::BoxBlurDialog::new(
                                        &project.canvas_state,
                                    ),
                                );
                            }
                            ui.close_menu();
                        }
                        if self
                            .assets
                            .menu_item_enabled(
                                ui,
                                Icon::MenuFilterZoomBlur,
                                &t!("menu.filter.blur.zoom"),
                                no_dialog,
                            )
                            .clicked()
                        {
                            if let Some(project) = self.active_project() {
                                self.active_dialog = ActiveDialog::ZoomBlur(
                                    crate::ops::effect_dialogs::ZoomBlurDialog::new(
                                        &project.canvas_state,
                                    ),
                                );
                            }
                            ui.close_menu();
                        }
                    });

                    // -- Sharpen submenu (was in Stylize + Noise) --
                    ui.menu_button(t!("menu.filter.sharpen"), |ui| {
                        if self
                            .assets
                            .menu_item_enabled(
                                ui,
                                Icon::MenuFilterSharpenItem,
                                &t!("menu.filter.sharpen.sharpen"),
                                no_dialog,
                            )
                            .clicked()
                        {
                            if let Some(project) = self.active_project() {
                                self.active_dialog = ActiveDialog::Sharpen(
                                    crate::ops::effect_dialogs::SharpenDialog::new(
                                        &project.canvas_state,
                                    ),
                                );
                            }
                            ui.close_menu();
                        }
                        if self
                            .assets
                            .menu_item_enabled(
                                ui,
                                Icon::MenuFilterReduceNoise,
                                &t!("menu.filter.sharpen.reduce_noise"),
                                no_dialog,
                            )
                            .clicked()
                        {
                            if let Some(project) = self.active_project() {
                                self.active_dialog = ActiveDialog::ReduceNoise(
                                    crate::ops::effect_dialogs::ReduceNoiseDialog::new(
                                        &project.canvas_state,
                                    ),
                                );
                            }
                            ui.close_menu();
                        }
                    });

                    // -- Distort submenu --
                    ui.menu_button(t!("menu.filter.distort"), |ui| {
                        if self
                            .assets
                            .menu_item_enabled(
                                ui,
                                Icon::MenuFilterCrystallize,
                                &t!("menu.filter.distort.crystallize"),
                                no_dialog,
                            )
                            .clicked()
                        {
                            if let Some(project) = self.active_project() {
                                self.active_dialog = ActiveDialog::Crystallize(
                                    crate::ops::effect_dialogs::CrystallizeDialog::new(
                                        &project.canvas_state,
                                    ),
                                );
                            }
                            ui.close_menu();
                        }
                        if self
                            .assets
                            .menu_item_enabled(
                                ui,
                                Icon::MenuFilterDents,
                                &t!("menu.filter.distort.dents"),
                                no_dialog,
                            )
                            .clicked()
                        {
                            if let Some(project) = self.active_project() {
                                self.active_dialog = ActiveDialog::Dents(
                                    crate::ops::effect_dialogs::DentsDialog::new(
                                        &project.canvas_state,
                                    ),
                                );
                            }
                            ui.close_menu();
                        }
                        if self
                            .assets
                            .menu_item_enabled(
                                ui,
                                Icon::MenuFilterPixelate,
                                &t!("menu.filter.distort.pixelate"),
                                no_dialog,
                            )
                            .clicked()
                        {
                            if let Some(project) = self.active_project() {
                                self.active_dialog = ActiveDialog::Pixelate(
                                    crate::ops::effect_dialogs::PixelateDialog::new(
                                        &project.canvas_state,
                                    ),
                                );
                            }
                            ui.close_menu();
                        }
                        if self
                            .assets
                            .menu_item_enabled(
                                ui,
                                Icon::MenuFilterBulge,
                                &t!("menu.filter.distort.bulge_pinch"),
                                no_dialog,
                            )
                            .clicked()
                        {
                            if let Some(project) = self.active_project() {
                                self.active_dialog = ActiveDialog::Bulge(
                                    crate::ops::effect_dialogs::BulgeDialog::new(
                                        &project.canvas_state,
                                    ),
                                );
                            }
                            ui.close_menu();
                        }
                        if self
                            .assets
                            .menu_item_enabled(
                                ui,
                                Icon::MenuFilterTwist,
                                &t!("menu.filter.distort.twist"),
                                no_dialog,
                            )
                            .clicked()
                        {
                            if let Some(project) = self.active_project() {
                                self.active_dialog = ActiveDialog::Twist(
                                    crate::ops::effect_dialogs::TwistDialog::new(
                                        &project.canvas_state,
                                    ),
                                );
                            }
                            ui.close_menu();
                        }
                    });

                    // -- Noise submenu --
                    ui.menu_button(t!("menu.filter.noise"), |ui| {
                        if self
                            .assets
                            .menu_item_enabled(
                                ui,
                                Icon::MenuFilterAddNoise,
                                &t!("menu.filter.noise.add_noise"),
                                no_dialog,
                            )
                            .clicked()
                        {
                            if let Some(project) = self.active_project() {
                                self.active_dialog = ActiveDialog::AddNoise(
                                    crate::ops::effect_dialogs::AddNoiseDialog::new(
                                        &project.canvas_state,
                                    ),
                                );
                            }
                            ui.close_menu();
                        }
                        if self
                            .assets
                            .menu_item_enabled(
                                ui,
                                Icon::MenuFilterMedian,
                                &t!("menu.filter.noise.median"),
                                no_dialog,
                            )
                            .clicked()
                        {
                            if let Some(project) = self.active_project() {
                                self.active_dialog = ActiveDialog::Median(
                                    crate::ops::effect_dialogs::MedianDialog::new(
                                        &project.canvas_state,
                                    ),
                                );
                            }
                            ui.close_menu();
                        }
                    });

                    // -- Stylize submenu (absorbs old Artistic) --
                    ui.menu_button(t!("menu.filter.stylize"), |ui| {
                        if self
                            .assets
                            .menu_item_enabled(
                                ui,
                                Icon::MenuFilterGlow,
                                &t!("menu.filter.stylize.glow"),
                                no_dialog,
                            )
                            .clicked()
                        {
                            if let Some(project) = self.active_project() {
                                self.active_dialog = ActiveDialog::Glow(
                                    crate::ops::effect_dialogs::GlowDialog::new(
                                        &project.canvas_state,
                                    ),
                                );
                            }
                            ui.close_menu();
                        }
                        if self
                            .assets
                            .menu_item_enabled(
                                ui,
                                Icon::MenuFilterVignette,
                                &t!("menu.filter.stylize.vignette"),
                                no_dialog,
                            )
                            .clicked()
                        {
                            if let Some(project) = self.active_project() {
                                self.active_dialog = ActiveDialog::Vignette(
                                    crate::ops::effect_dialogs::VignetteDialog::new(
                                        &project.canvas_state,
                                    ),
                                );
                            }
                            ui.close_menu();
                        }
                        if self
                            .assets
                            .menu_item_enabled(
                                ui,
                                Icon::MenuFilterHalftone,
                                &t!("menu.filter.stylize.halftone"),
                                no_dialog,
                            )
                            .clicked()
                        {
                            if let Some(project) = self.active_project() {
                                self.active_dialog = ActiveDialog::Halftone(
                                    crate::ops::effect_dialogs::HalftoneDialog::new(
                                        &project.canvas_state,
                                    ),
                                );
                            }
                            ui.close_menu();
                        }
                        if self
                            .assets
                            .menu_item_enabled(
                                ui,
                                Icon::MenuFilterInk,
                                &t!("menu.filter.stylize.ink"),
                                no_dialog,
                            )
                            .clicked()
                        {
                            if let Some(project) = self.active_project() {
                                self.active_dialog =
                                    ActiveDialog::Ink(crate::ops::effect_dialogs::InkDialog::new(
                                        &project.canvas_state,
                                    ));
                            }
                            ui.close_menu();
                        }
                        if self
                            .assets
                            .menu_item_enabled(
                                ui,
                                Icon::MenuFilterOilPainting,
                                &t!("menu.filter.stylize.oil_painting"),
                                no_dialog,
                            )
                            .clicked()
                        {
                            if let Some(project) = self.active_project() {
                                self.active_dialog = ActiveDialog::OilPainting(
                                    crate::ops::effect_dialogs::OilPaintingDialog::new(
                                        &project.canvas_state,
                                    ),
                                );
                            }
                            ui.close_menu();
                        }
                        if self
                            .assets
                            .menu_item_enabled(
                                ui,
                                Icon::MenuFilterColorFilter,
                                &t!("menu.filter.stylize.color_filter"),
                                no_dialog,
                            )
                            .clicked()
                        {
                            if let Some(project) = self.active_project() {
                                self.active_dialog = ActiveDialog::ColorFilter(
                                    crate::ops::effect_dialogs::ColorFilterDialog::new(
                                        &project.canvas_state,
                                    ),
                                );
                            }
                            ui.close_menu();
                        }
                    });

                    // -- Glitch submenu --
                    ui.menu_button(t!("menu.filter.glitch"), |ui| {
                        if self
                            .assets
                            .menu_item_enabled(
                                ui,
                                Icon::MenuFilterPixelDrag,
                                &t!("menu.filter.glitch.pixel_drag"),
                                no_dialog,
                            )
                            .clicked()
                        {
                            if let Some(project) = self.active_project() {
                                self.active_dialog = ActiveDialog::PixelDrag(
                                    crate::ops::effect_dialogs::PixelDragDialog::new(
                                        &project.canvas_state,
                                    ),
                                );
                            }
                            ui.close_menu();
                        }
                        if self
                            .assets
                            .menu_item_enabled(
                                ui,
                                Icon::MenuFilterRgbDisplace,
                                &t!("menu.filter.glitch.rgb_displace"),
                                no_dialog,
                            )
                            .clicked()
                        {
                            if let Some(project) = self.active_project() {
                                self.active_dialog = ActiveDialog::RgbDisplace(
                                    crate::ops::effect_dialogs::RgbDisplaceDialog::new(
                                        &project.canvas_state,
                                    ),
                                );
                            }
                            ui.close_menu();
                        }
                    });

                    // -- AI submenu --
                    ui.separator();
                    let remove_bg_resp = self.assets.menu_item_enabled(
                        ui,
                        Icon::MenuFilterRemoveBg,
                        &t!("menu.filter.remove_background"),
                        no_dialog && self.onnx_available,
                    );
                    if !self.onnx_available {
                        remove_bg_resp.clone().on_disabled_hover_text(
                            "Configure ONNX Runtime and BiRefNet model in Preferences > AI tab",
                        );
                    }
                    if remove_bg_resp.clicked() {
                        self.active_dialog = ActiveDialog::RemoveBackground(
                            crate::ops::effect_dialogs::RemoveBackgroundDialog::new(),
                        );
                        ui.close_menu();
                    }

                    // -- Custom Scripts submenu (only shown if scripts exist) --
                    if !self.custom_scripts.is_empty() {
                        ui.separator();
                        ui.menu_button("Custom", |ui| {
                            let mut action: Option<CustomScriptAction> = None;
                            for (idx, effect) in self.custom_scripts.iter().enumerate() {
                                ui.horizontal(|ui| {
                                    // Run button — effect name, takes remaining space
                                    if ui.button(&effect.name).clicked() {
                                        action = Some(CustomScriptAction::Run(idx));
                                        ui.close_menu();
                                    }
                                    // Push Edit and Delete to the right
                                    ui.with_layout(
                                        egui::Layout::right_to_left(egui::Align::Center),
                                        |ui| {
                                            let del_btn = ui
                                                .small_button(
                                                    egui::RichText::new("Del")
                                                        .color(egui::Color32::from_rgb(180, 80, 80))
                                                        .size(10.0),
                                                )
                                                .on_hover_text("Delete this custom effect");
                                            if del_btn.clicked() {
                                                action = Some(CustomScriptAction::Delete(idx));
                                                ui.close_menu();
                                            }
                                            let edit_btn = ui
                                                .small_button(
                                                    egui::RichText::new("Edit").size(10.0),
                                                )
                                                .on_hover_text("Edit in Script Editor");
                                            if edit_btn.clicked() {
                                                action = Some(CustomScriptAction::Edit(idx));
                                                ui.close_menu();
                                            }
                                        },
                                    );
                                });
                            }
                            if let Some(act) = action {
                                match act {
                                    CustomScriptAction::Run(idx) => {
                                        if let Some(effect) = self.custom_scripts.get(idx) {
                                            let code = effect.code.clone();
                                            let name = effect.name.clone();
                                            self.run_custom_script(code, name);
                                        }
                                    }
                                    CustomScriptAction::Edit(idx) => {
                                        if let Some(effect) = self.custom_scripts.get(idx) {
                                            self.script_editor.code = effect.code.clone();
                                            self.window_visibility.script_editor = true;
                                        }
                                    }
                                    CustomScriptAction::Delete(idx) => {
                                        if let Some(effect) = self.custom_scripts.get(idx) {
                                            script_editor::delete_custom_effect(&effect.name);
                                        }
                                        self.custom_scripts.remove(idx);
                                    }
                                }
                            }
                        });
                    }
                });

                // ==================== GENERATE MENU (was: Effects > Render) ====================
                ui.menu_button(t!("menu.generate"), |ui| {
                    if self
                        .assets
                        .menu_item_enabled(
                            ui,
                            Icon::MenuGenerateGrid,
                            &t!("menu.generate.grid"),
                            no_dialog,
                        )
                        .clicked()
                    {
                        if let Some(project) = self.active_project() {
                            self.active_dialog = ActiveDialog::Grid(
                                crate::ops::effect_dialogs::GridDialog::new(&project.canvas_state),
                            );
                        }
                        ui.close_menu();
                    }
                    if self
                        .assets
                        .menu_item_enabled(
                            ui,
                            Icon::MenuGenerateShadow,
                            &t!("menu.generate.drop_shadow"),
                            no_dialog,
                        )
                        .clicked()
                    {
                        if let Some(project) = self.active_project() {
                            self.active_dialog = ActiveDialog::DropShadow(
                                crate::ops::effect_dialogs::DropShadowDialog::new(
                                    &project.canvas_state,
                                ),
                            );
                        }
                        ui.close_menu();
                    }
                    if self
                        .assets
                        .menu_item_enabled(
                            ui,
                            Icon::MenuGenerateOutline,
                            &t!("menu.generate.outline"),
                            no_dialog,
                        )
                        .clicked()
                    {
                        if let Some(project) = self.active_project() {
                            self.active_dialog = ActiveDialog::Outline(
                                crate::ops::effect_dialogs::OutlineDialog::new(
                                    &project.canvas_state,
                                ),
                            );
                        }
                        ui.close_menu();
                    }
                    if self
                        .assets
                        .menu_item_enabled(
                            ui,
                            Icon::MenuGenerateContours,
                            &t!("menu.generate.contours"),
                            no_dialog,
                        )
                        .clicked()
                    {
                        if let Some(project) = self.active_project() {
                            self.active_dialog = ActiveDialog::Contours(
                                crate::ops::effect_dialogs::ContoursDialog::new(
                                    &project.canvas_state,
                                ),
                            );
                        }
                        ui.close_menu();
                    }
                });

                ui.menu_button(t!("menu.view"), |ui| {
                    // Panel toggles
                    ui.label(egui::RichText::new("Panels").strong().size(11.0));
                    ui.checkbox(
                        &mut self.window_visibility.tools,
                        t!("menu.view.tools_panel"),
                    );
                    ui.checkbox(
                        &mut self.window_visibility.layers,
                        t!("menu.view.layers_panel"),
                    );
                    ui.checkbox(
                        &mut self.window_visibility.history,
                        t!("menu.view.history_panel"),
                    );
                    ui.checkbox(
                        &mut self.window_visibility.colors,
                        t!("menu.view.colors_panel"),
                    );
                    ui.checkbox(
                        &mut self.window_visibility.script_editor,
                        t!("menu.view.script_editor"),
                    );

                    ui.separator();

                    // Pixel grid toggle
                    let show_grid = self
                        .active_project()
                        .map(|p| p.canvas_state.show_pixel_grid)
                        .unwrap_or(false);
                    let mut grid_checked = show_grid;
                    if ui
                        .checkbox(&mut grid_checked, t!("menu.view.toggle_pixel_grid"))
                        .changed()
                        && let Some(project) = self.active_project_mut()
                    {
                        project.canvas_state.show_pixel_grid = grid_checked;
                    }

                    ui.separator();

                    // Zoom controls
                    if self
                        .assets
                        .menu_item_shortcut_below(
                            ui,
                            Icon::MenuViewZoomIn,
                            &t!("menu.view.zoom_in"),
                            &menu_kb,
                            BindableAction::ViewZoomIn,
                        )
                        .clicked()
                    {
                        self.canvas.zoom_in();
                        ui.close_menu();
                    }
                    if self
                        .assets
                        .menu_item_shortcut_below(
                            ui,
                            Icon::MenuViewZoomOut,
                            &t!("menu.view.zoom_out"),
                            &menu_kb,
                            BindableAction::ViewZoomOut,
                        )
                        .clicked()
                    {
                        self.canvas.zoom_out();
                        ui.close_menu();
                    }
                    if self
                        .assets
                        .menu_item_shortcut_below(
                            ui,
                            Icon::MenuViewFitWindow,
                            &t!("menu.view.fit_to_window"),
                            &menu_kb,
                            BindableAction::ViewFitToWindow,
                        )
                        .clicked()
                    {
                        self.canvas.reset_zoom();
                        ui.close_menu();
                    }

                    ui.separator();

                    // Theme submenu
                    ui.menu_button(t!("menu.view.theme"), |ui| {
                        ui.set_min_width(ui.min_rect().width().max(160.0));
                        let is_light = matches!(self.theme.mode, crate::theme::ThemeMode::Light);
                        let is_dark = matches!(self.theme.mode, crate::theme::ThemeMode::Dark);
                        if ui.radio(is_light, t!("menu.view.theme.light")).clicked() {
                            if !is_light {
                                self.theme.toggle();
                                self.theme.apply(ctx);
                                self.settings.theme_mode = self.theme.mode;
                                self.settings.save();
                            }
                            ui.close_menu();
                        }
                        if ui.radio(is_dark, t!("menu.view.theme.dark")).clicked() {
                            if !is_dark {
                                self.theme.toggle();
                                self.theme.apply(ctx);
                                self.settings.theme_mode = self.theme.mode;
                                self.settings.save();
                            }
                            ui.close_menu();
                        }
                    });
                });
            });
        });

        // --- Row 2: Actions + Project Tabs ---
        let toolbar_resp = egui::TopBottomPanel::top("toolbar_tabs")
            .frame(self.theme.toolbar_frame())
            .show(ctx, |ui| {
            let toolbar_height = ui.available_height();
            ui.horizontal(|ui| {
                // === File Actions ===
                if self.assets.small_icon_button(ui, Icon::New).clicked() {
                    self.new_file_dialog.load_clipboard_dimensions();
                    self.new_file_dialog.open = true;
                }
                if self.assets.small_icon_button(ui, Icon::Open).clicked() {
                    self.handle_open_file(ctx.input(|i| i.time));
                }
                let is_dirty = self.active_project().is_some_and(|p| p.is_dirty);
                if self.assets.icon_button_enabled(ui, Icon::Save, is_dirty).clicked() {
                    self.save_file_dialog.open = true;
                }
                
                ui.separator();
                
                // === Undo/Redo ===
                let can_undo = self.active_project().is_some_and(|p| p.history.can_undo());
                let can_redo = self.active_project().is_some_and(|p| p.history.can_redo());
                
                if self.assets.icon_button_enabled(ui, Icon::Undo, can_undo).clicked() {
                    if self.paste_overlay.is_some() {
                        self.cancel_paste_overlay();
                        if let Some(project) = self.active_project_mut() {
                            project.canvas_state.clear_selection();
                        }
                    } else if self.tools_panel.has_active_tool_preview() {
                        if let Some(project) = self.projects.get_mut(self.active_project_index) {
                            self.tools_panel.cancel_active_tool(&mut project.canvas_state);
                        }
                    } else if let Some(project) = self.active_project_mut() {
                        project.canvas_state.clear_selection();
                        project.history.undo(&mut project.canvas_state);
                    }
                }
                if self.assets.icon_button_enabled(ui, Icon::Redo, can_redo).clicked()
                    && let Some(project) = self.active_project_mut() {
                        project.history.redo(&mut project.canvas_state);
                    }
                
                ui.separator();
                
                // === Zoom Controls ===
                if self.assets.small_icon_button(ui, Icon::ZoomIn).clicked() {
                    self.canvas.zoom_in();
                }
                if self.assets.small_icon_button(ui, Icon::ZoomOut).clicked() {
                    self.canvas.zoom_out();
                }
                if self.assets.small_icon_button(ui, Icon::ResetZoom).clicked() {
                    self.canvas.reset_zoom();
                }
                
                ui.separator();
                
                // Pixel grid toggle (respects settings mode)
                let pixel_grid_mode = self.settings.pixel_grid_mode;
                let show_grid = self.active_project().map(|p| p.canvas_state.show_pixel_grid).unwrap_or(false);
                
                match pixel_grid_mode {
                    PixelGridMode::Auto => {
                        // Auto mode: Show toggle for manual override
                        let grid_icon = if show_grid { Icon::GridOn } else { Icon::GridOff };
                        if self.assets.small_icon_button(ui, grid_icon).clicked()
                            && let Some(project) = self.active_project_mut() {
                                project.canvas_state.show_pixel_grid = !project.canvas_state.show_pixel_grid;
                            }
                    }
                    PixelGridMode::AlwaysOn => {
                        if let Some(project) = self.active_project_mut() {
                            project.canvas_state.show_pixel_grid = true;
                        }
                        self.assets.icon_button_enabled(ui, Icon::GridOn, false);
                    }
                    PixelGridMode::AlwaysOff => {
                        if let Some(project) = self.active_project_mut() {
                            project.canvas_state.show_pixel_grid = false;
                        }
                        self.assets.icon_button_enabled(ui, Icon::GridOff, false);
                    }
                }
                
                // Guidelines toggle
                {
                    let show_guides = self.active_project().map(|p| p.canvas_state.show_guidelines).unwrap_or(false);
                    let guide_icon = if show_guides { Icon::GuidesOn } else { Icon::GuidesOff };
                    if self.assets.small_icon_button(ui, guide_icon).clicked()
                        && let Some(project) = self.active_project_mut() {
                            project.canvas_state.show_guidelines = !project.canvas_state.show_guidelines;
                        }
                }
                
                // Mirror mode toggle (cycles: None → H → V → Quarters → None)
                {
                    use crate::canvas::MirrorMode;
                    let mode = self.active_project().map(|p| p.canvas_state.mirror_mode).unwrap_or(MirrorMode::None);
                    let mirror_icon = match mode {
                        MirrorMode::None => Icon::MirrorOff,
                        MirrorMode::Horizontal => Icon::MirrorH,
                        MirrorMode::Vertical => Icon::MirrorV,
                        MirrorMode::Quarters => Icon::MirrorQ,
                    };
                    if self.assets.small_icon_button(ui, mirror_icon).clicked()
                        && let Some(project) = self.active_project_mut() {
                            project.canvas_state.mirror_mode = project.canvas_state.mirror_mode.next();
                        }
                }
                
                ui.separator();
                
                // === Project Tabs Section ===
                // Allocate remaining space for the tab tray
                ui.allocate_ui_with_layout(
                    egui::Vec2::new(ui.available_width(), ui.available_height()),
                    egui::Layout::left_to_right(egui::Align::Center),
                    |ui| {
                        // Create a rect that spans the full toolbar height + extra padding
                        let extra_height = 6.0; // Add extra height to ensure tabs fit
                        let tray_rect = egui::Rect::from_min_max(
                            egui::pos2(ui.cursor().min.x, ui.cursor().min.y - 3.0), // Raise it up 3px
                            egui::pos2(ui.cursor().min.x + ui.available_width(), ui.cursor().min.y + toolbar_height + extra_height),
                        );
                        
                        // Paint darker recessed background (inner shadow effect)
                        let painter = ui.painter();
                        
                        // Main darker background
                        painter.rect_filled(
                            tray_rect,
                            egui::Rounding::same(3.0),
                            egui::Color32::from_black_alpha(25),
                        );
                        
                        // Smooth left-edge gradient for recessed/shadow effect
                        let is_neon = self.settings.neon_mode && self.theme.mode == crate::theme::ThemeMode::Dark;
                        let gradient_width = if is_neon { 160.0 } else { 40.0 }; // 4x stretch for neon
                        let steps = if is_neon { 64 } else { 32 }; // more steps for wider glow
                        for i in 0..steps {
                            let t = i as f32 / steps as f32;
                            let eased = 1.0 - (t * t);
                            let x = tray_rect.min.x + t * gradient_width;
                            let next_x = tray_rect.min.x + (t + 1.0 / steps as f32) * gradient_width;
                            let gradient_rect = egui::Rect::from_min_max(
                                egui::pos2(x, tray_rect.min.y),
                                egui::pos2(next_x, tray_rect.max.y),
                            );
                            let color = if is_neon {
                                let [r, g, b, _] = self.theme.accent.to_array();
                                egui::Color32::from_rgba_unmultiplied(r, g, b, (eased * 22.0) as u8)
                            } else {
                                egui::Color32::from_black_alpha((eased * 30.0) as u8)
                            };
                            painter.rect_filled(gradient_rect, egui::Rounding::ZERO, color);
                        }
                        
                        // Now layout the actual tabs with arrows - use TOP alignment and offset up
                        ui.allocate_ui_with_layout(
                            egui::Vec2::new(ui.available_width(), ui.available_height()),
                            egui::Layout::left_to_right(egui::Align::Min),
                            |ui| {
                                ui.add_space(-3.0); // Negative space to pull content up
                                ui.horizontal(|ui| {
                            let mut tab_to_switch: Option<usize> = None;
                            let mut tab_to_close: Option<usize> = None;
                            
                            ui.add_space(8.0);
                            
                            // Left scroll arrow (conditional - subtle chevron)
                            // For now, always show but make it very subtle
                            // In a more advanced version, check scroll state to hide/disable
                            let left_chevron = egui::RichText::new("‹").size(16.0);
                            let left_arrow = egui::Button::new(left_chevron)
                                .frame(false)
                                .min_size(egui::vec2(20.0, 20.0));
                            if ui.add(left_arrow).on_hover_text("Scroll Left").clicked() {
                                // Visual cue for scrolling
                            }
                            
                            ui.add_space(4.0);
                            
                            // Scrollable area for tabs - takes remaining width
                            let scroll_width = ui.available_width() - 32.0; // Reserve space for right arrow
                            ui.allocate_ui_with_layout(
                                egui::Vec2::new(scroll_width, ui.available_height()),
                                egui::Layout::left_to_right(egui::Align::Center),
                                |ui| {
                                    egui::ScrollArea::horizontal()
                                        .id_source("project_tabs_scroll")
                                        .show(ui, |ui| {
                                                ui.horizontal(|ui| {
                                                for (idx, project) in self.projects.iter().enumerate() {
                                                    let is_active = idx == self.active_project_index;
                                                    
                                                    // Tab styling
                                                    let tab_text = project.display_title();
                                                    let text_color = if is_active {
                                                        egui::Color32::WHITE
                                                    } else {
                                                        match self.theme.mode {
                                                            crate::theme::ThemeMode::Dark => egui::Color32::from_gray(180),
                                                            crate::theme::ThemeMode::Light => egui::Color32::from_gray(50),
                                                        }
                                                    };
                                                    let text = egui::RichText::new(&tab_text).color(text_color);
                                                    let text = if is_active { text.strong() } else { text };
                                                    
                                                    // Tab frame with white for inactive, accent for active
                                                    let fill = if is_active {
                                                        self.theme.accent
                                                    } else {
                                                        match self.theme.mode {
                                                            crate::theme::ThemeMode::Dark => egui::Color32::from_gray(38),
                                                            crate::theme::ThemeMode::Light => egui::Color32::WHITE,
                                                        }
                                                    };
                                                    
                                                    // Get the frame response to paint shadow
                                                    egui::Frame::none()
                                                        .fill(fill)
                                                        .inner_margin(egui::Margin::symmetric(8.0, 2.85))
                                                        .rounding(egui::Rounding::same(4.0))
                                                        .show(ui, |ui| {
                                                            ui.horizontal(|ui| {
                                                                // Tab label (clickable)
                                                                if ui.add(egui::Label::new(text).sense(egui::Sense::click())).clicked() {
                                                                    tab_to_switch = Some(idx);
                                                                }
                                                                
                                                                // Close button with lighter blue for active tab
                                                                let close_btn = if is_active {
                                                                    egui::Button::new(egui::RichText::new("×").color(self.theme.accent_strong))
                                                                        .frame(false)
                                                                } else {
                                                                    egui::Button::new("×").frame(false)
                                                                };
                                                                if ui.add(close_btn).on_hover_text("Close").clicked() {
                                                                    tab_to_close = Some(idx);
                                                                }
                                                            });
                                                        });
                                                    
                                                    ui.add_space(2.0);
                                                }
                                                
                                                // "+" button to create new tab
                                                let plus_fill = match self.theme.mode {
                                                    crate::theme::ThemeMode::Dark => egui::Color32::from_gray(38),
                                                    crate::theme::ThemeMode::Light => egui::Color32::WHITE,
                                                };
                                                let plus_btn = egui::Button::new("+")
                                                    .fill(plus_fill)
                                                    .rounding(egui::Rounding::same(4.0));
                                                if ui.add(plus_btn).on_hover_text("New Tab").clicked() {
                                                    self.new_file_dialog.load_clipboard_dimensions();
                                                    self.new_file_dialog.open = true;
                                                }
                                            });
                                        });
                                },
                            );
                            
                            ui.add_space(4.0);
                            
                            // Right scroll arrow (subtle chevron, vertically centered)
                            let right_chevron = egui::RichText::new("›").size(16.0);
                            let right_arrow = egui::Button::new(right_chevron)
                                .frame(false)
                                .min_size(egui::vec2(20.0, 20.0));
                            if ui.add(right_arrow).on_hover_text("Scroll Right").clicked() {
                                // Visual cue for scrolling
                            }
                            
                            ui.add_space(4.0);
                            
                            // Process tab actions after iteration
                            if let Some(idx) = tab_to_switch {
                                self.switch_to_project(idx);
                            }
                            if let Some(idx) = tab_to_close {
                                self.close_project(idx);
                            }
                        });
                    });
                },
            );
        });
    });

        // Sync primary color from colors panel to tools
        self.tools_panel.properties.color = self.colors_panel.get_primary_color();

        // Sync color picker result back to colors panel
        // If the color picker tool picked a color this frame, update the colors panel
        if let Some(picked_color) = self.tools_panel.last_picked_color.take() {
            self.colors_panel.primary_color = picked_color;
        }

        // Paint a downward shadow from the toolbar onto the context bar
        // Use Middle order so it renders below menus/popups but above the panel content
        {
            let toolbar_rect = toolbar_resp.response.rect;
            let screen_rect = ctx.screen_rect();
            let painter = ctx.layer_painter(egui::LayerId::new(
                egui::Order::Background,
                egui::Id::new("toolbar_shadow"),
            ));
            let shadow_height = 8.0_f32;
            let steps = 8;
            let is_neon =
                self.settings.neon_mode && self.theme.mode == crate::theme::ThemeMode::Dark;
            for i in 0..steps {
                let t = i as f32 / steps as f32;
                let y = toolbar_rect.bottom() + t * shadow_height;
                let next_y = toolbar_rect.bottom() + (t + 1.0 / steps as f32) * shadow_height;
                let rect = egui::Rect::from_min_max(
                    egui::pos2(screen_rect.left(), y),
                    egui::pos2(screen_rect.right(), next_y),
                );
                let color = if is_neon {
                    let [r, g, b, _] = self.theme.accent.to_array();
                    egui::Color32::from_rgba_unmultiplied(r, g, b, ((1.0 - t) * 20.0) as u8)
                } else {
                    egui::Color32::from_black_alpha(((1.0 - t) * 30.0) as u8)
                };
                painter.rect_filled(rect, 0.0, color);
            }
        }

        // --- Row 3: Context Bar (Dynamic Tool Options / Paste Options) ---
        egui::TopBottomPanel::top("context_bar")
            .frame(self.theme.context_bar_frame())
            .exact_height(36.0) // Fixed height — prevents canvas shifting when tool context bar changes size
            .show(ctx, |ui| {
                ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                    if let Some(ref mut overlay) = self.paste_overlay {
                        // --- Paste overlay context bar ---
                        ui.label(egui::RichText::new("Paste").strong());
                        ui.separator();

                        // Filter mode
                        ui.label("Filter:");
                        let current_interp = overlay.interpolation;
                        egui::ComboBox::from_id_source("ctx_paste_filter")
                            .selected_text(current_interp.label())
                            .width(110.0)
                            .show_ui(ui, |ui| {
                                for interp in crate::ops::transform::Interpolation::all() {
                                    if ui
                                        .selectable_label(*interp == current_interp, interp.label())
                                        .clicked()
                                    {
                                        overlay.interpolation = *interp;
                                    }
                                }
                            });

                        ui.separator();

                        // Anti-aliasing toggle
                        ui.checkbox(&mut overlay.anti_aliasing, "Anti-aliasing");

                        ui.separator();

                        // Position info
                        ui.label(format!(
                            "X: {:.0}  Y: {:.0}  W: {:.0}  H: {:.0}  Rot: {:.1}°",
                            overlay.center.x
                                - overlay.source.width() as f32 * overlay.scale_x / 2.0,
                            overlay.center.y
                                - overlay.source.height() as f32 * overlay.scale_y / 2.0,
                            overlay.source.width() as f32 * overlay.scale_x,
                            overlay.source.height() as f32 * overlay.scale_y,
                            overlay.rotation.to_degrees(),
                        ));

                        ui.separator();

                        // Quick actions
                        if ui
                            .button("Reset")
                            .on_hover_text("Reset all transforms")
                            .clicked()
                        {
                            overlay.rotation = 0.0;
                            overlay.scale_x = 1.0;
                            overlay.scale_y = 1.0;
                            overlay.anchor_offset = egui::Vec2::ZERO;
                        }
                    } else {
                        let ctx_primary = self.colors_panel.get_primary_color();
                        let ctx_secondary = self.colors_panel.get_secondary_color();
                        self.tools_panel.show_context_bar(
                            ui,
                            &self.assets,
                            ctx_primary,
                            ctx_secondary,
                        );
                    }
                });
            });

        // Process pending selection modification from context bar
        if let Some(op) = self.tools_panel.pending_sel_modify.take()
            && let Some(project) = self.active_project_mut()
            && project.canvas_state.has_selection()
        {
            use crate::components::tools::SelectionModifyOp;
            match op {
                SelectionModifyOp::Feather(r) => {
                    crate::ops::adjustments::feather_selection(&mut project.canvas_state, r)
                }
                SelectionModifyOp::Expand(r) => {
                    crate::ops::adjustments::expand_selection(&mut project.canvas_state, r)
                }
                SelectionModifyOp::Contract(r) => {
                    crate::ops::adjustments::contract_selection(&mut project.canvas_state, r)
                }
            }
        }

        // --- Full-Screen Canvas (CentralPanel fills remaining space) ---
        let canvas_bg_top = self.theme.canvas_bg_top;
        let canvas_bg_bottom = self.theme.canvas_bg_bottom;

        egui::CentralPanel::default()
            .frame(egui::Frame {
                fill: canvas_bg_bottom,
                ..Default::default()
            })
            .show(ctx, |ui| {
                // Draw subtle gradient background over the solid fill
                let rect = ui.max_rect();
                let painter = ui.painter();

                // Vertical gradient from top to bottom
                let mesh = {
                    let mut mesh = egui::Mesh::default();
                    mesh.colored_vertex(rect.left_top(), canvas_bg_top);
                    mesh.colored_vertex(rect.right_top(), canvas_bg_top);
                    mesh.colored_vertex(rect.left_bottom(), canvas_bg_bottom);
                    mesh.colored_vertex(rect.right_bottom(), canvas_bg_bottom);
                    mesh.add_triangle(0, 1, 2);
                    mesh.add_triangle(1, 2, 3);
                    mesh
                };
                painter.add(egui::Shape::mesh(mesh));

                if let Some(project) = self.projects.get_mut(self.active_project_index) {
                    let primary_color_f32 = self.colors_panel.get_primary_color_f32();
                    let secondary_color_f32 = self.colors_panel.get_secondary_color_f32();
                    // Push theme accent colours into canvas for selection rendering.
                    self.canvas.selection_stroke = self.theme.accent;
                    self.canvas.selection_fill = {
                        let [r, g, b, _] = self.theme.accent.to_array();
                        egui::Color32::from_rgba_unmultiplied(r, g, b, 25)
                    };
                    self.canvas.selection_contrast = match self.theme.mode {
                        crate::theme::ThemeMode::Dark => egui::Color32::BLACK,
                        crate::theme::ThemeMode::Light => egui::Color32::WHITE,
                    };
                    // Set tool icon cursor texture for the canvas overlay.
                    {
                        use crate::assets::Icon;
                        use crate::components::tools::Tool;
                        let icon_for_cursor: Option<Icon> = match self.tools_panel.active_tool {
                            Tool::Pencil => Some(Icon::Pencil),
                            Tool::Fill => Some(Icon::Fill),
                            Tool::ColorPicker => Some(Icon::ColorPicker),
                            Tool::Zoom => Some(Icon::Zoom),
                            Tool::Pan => Some(Icon::Pan),
                            _ => None,
                        };
                        self.canvas.tool_cursor_icon =
                            icon_for_cursor.and_then(|ic| self.assets.get_texture(ic).cloned());
                    }
                    self.canvas.show_with_state(
                        ui,
                        &mut project.canvas_state,
                        Some(&mut self.tools_panel),
                        primary_color_f32,
                        secondary_color_f32,
                        canvas_bg_bottom,
                        self.paste_overlay.as_mut(),
                        modal_open,
                        &self.settings,
                        self.pending_filter_jobs,
                        self.pending_io_ops,
                        self.theme.accent,
                        self.filter_ops_start_time,
                        self.io_ops_start_time,
                        &self.filter_status_description,
                    );

                    // Handle paste overlay context menu results.
                    if let Some(action) = self.canvas.paste_context_action.take() {
                        if action {
                            // Commit — always a fresh snapshot (extraction is already in history).
                            if let Some(overlay) = self.paste_overlay.take() {
                                let desc = if self.is_move_pixels_active {
                                    "Move Pixels"
                                } else {
                                    "Paste"
                                };
                                let mut cmd =
                                    SnapshotCommand::new(desc.to_string(), &project.canvas_state);
                                project.canvas_state.clear_preview_state();
                                overlay.commit(&mut project.canvas_state);
                                cmd.set_after(&project.canvas_state);
                                project.history.push(Box::new(cmd));
                                project.mark_dirty();
                            }
                            self.is_move_pixels_active = false;
                        } else {
                            // Cancel.
                            self.paste_overlay = None;
                            if self.is_move_pixels_active {
                                // MovePixels: undo the extraction entry we already pushed
                                project.history.undo(&mut project.canvas_state);
                                self.is_move_pixels_active = false;
                            }
                            project.canvas_state.clear_preview_state();
                            project.canvas_state.mark_dirty(None);
                        }
                    }
                }
            });

        // --- Floating Panels ---
        // Detect screen size changes ONCE before any panel renders,
        // so all panels see the same change flag.
        let screen_rect = ctx.screen_rect();
        let screen_w = screen_rect.max.x;
        let screen_h = screen_rect.max.y;
        let screen_size_changed = self.last_screen_size.0 > 0.0
            && ((screen_w - self.last_screen_size.0).abs() > 0.5
                || (screen_h - self.last_screen_size.1).abs() > 0.5);

        self.show_floating_tools_panel(ctx, screen_size_changed);
        self.show_floating_layers_panel(ctx, screen_size_changed);
        self.show_floating_history_panel(ctx, screen_size_changed);
        self.show_floating_colors_panel(ctx, screen_size_changed);
        self.show_floating_script_editor(ctx, screen_size_changed);

        // Update last_screen_size AFTER all panels have used the flag
        self.last_screen_size = (screen_w, screen_h);

        // --- Process Stroke Events for Undo/Redo ---
        // Check if a stroke just completed and add it to history
        if let Some(stroke_event) = self.tools_panel.take_stroke_event()
            && let Some(project) = self.active_project_mut()
        {
            // Capture "after" state for the stroke bounds
            // Use same padding as "before" capture (10.0) to ensure bounds match
            let after_patch = history::PixelPatch::capture(
                &project.canvas_state,
                stroke_event.layer_index,
                stroke_event.bounds.expand(10.0),
            );

            // Create the brush command with before/after patches
            if let Some(before_patch) = stroke_event.before_snapshot {
                let command =
                    history::BrushCommand::new(stroke_event.description, before_patch, after_patch);
                project.history.push(Box::new(command));
                project.mark_dirty();
            }
        }

        // --- Process pending history commands from tools (e.g., perspective crop) ---
        let pending_cmds: Vec<_> = self
            .tools_panel
            .pending_history_commands
            .drain(..)
            .collect();
        for cmd in pending_cmds {
            if let Some(project) = self.active_project_mut() {
                project.history.push(cmd);
                project.mark_dirty();
            }
        }

        // --- Async Color Removal ---
        // Check if a color removal was requested and dispatch via spawn_filter_job
        if let Some(req) = self.tools_panel.take_pending_color_removal()
            && let Some(project) = self.projects.get(self.active_project_index)
        {
            let idx = req.layer_idx;
            if idx < project.canvas_state.layers.len() {
                let original_pixels = project.canvas_state.layers[idx].pixels.clone();
                let original_flat = original_pixels.to_rgba_image();
                let current_time = ctx.input(|i| i.time);
                self.spawn_filter_job(
                    current_time,
                    "Color Remover".to_string(),
                    idx,
                    original_pixels,
                    original_flat,
                    move |img| {
                        let changes = crate::ops::color_removal::compute_color_removal(
                            img,
                            req.click_x,
                            req.click_y,
                            req.tolerance,
                            req.smoothness,
                            req.contiguous,
                            req.selection_mask.as_ref(),
                        );
                        let mut result = img.clone();
                        crate::ops::color_removal::apply_color_removal(&mut result, &changes);
                        result
                    },
                );
            }
        }

        // --- Async Content-Aware Inpaint (Balanced / High Quality) ---
        if let Some(req) = self.tools_panel.take_pending_inpaint()
            && let Some(project) = self.projects.get(self.active_project_index)
        {
            let idx = req.layer_idx;
            if idx < project.canvas_state.layers.len() {
                let original_pixels = project.canvas_state.layers[idx].pixels.clone();
                let original_flat = req.original_flat;
                let hole_mask = req.hole_mask;
                let patch_size = req.patch_size;
                let iterations = req.iterations;
                let current_time = ctx.input(|i| i.time);
                self.spawn_filter_job(
                    current_time,
                    "Content-Aware Brush".to_string(),
                    idx,
                    original_pixels,
                    original_flat,
                    move |img| {
                        crate::ops::inpaint::fill_region_patchmatch(
                            img, &hole_mask, patch_size, iterations,
                        )
                    },
                );
            }
        }

        // --- Continuous Repaint While Painting ---
        // Request repaint during active brush/eraser strokes for smooth 60fps.
        // This ensures we don't miss mouse input events and get jittery results.
        if self.tools_panel.is_stroke_active() {
            ctx.request_repaint();
        }
    }
}

impl PaintFEApp {
    /// Handle opening a file - creates a new project tab
    fn handle_open_file(&mut self, current_time: f64) {
        if let Some(path) = self.file_handler.pick_file_path() {
            self.open_file_by_path(path, current_time);
        }
    }

    /// Open a file by path — creates a new project tab.
    /// Used by both "Open…" menu and drag-and-drop.
    fn open_file_by_path(&mut self, path: std::path::PathBuf, current_time: f64) {
        // Check if file is already open in another tab
        if self
            .projects
            .iter()
            .any(|p| p.path.as_ref().is_some_and(|pp| *pp == path))
        {
            // Already open — just switch to that tab
            if let Some(idx) = self
                .projects
                .iter()
                .position(|p| p.path.as_ref().is_some_and(|pp| *pp == path))
            {
                self.active_project_index = idx;
            }
            return;
        }

        let is_pfe = path
            .extension()
            .map(|ext| ext.to_string_lossy().to_lowercase() == "pfe")
            .unwrap_or(false);

        if is_pfe {
            // Open .pfe project file in background (preserves layers)
            let sender = self.io_sender.clone();
            if self.pending_io_ops == 0 {
                self.io_ops_start_time = Some(current_time);
            }
            self.pending_io_ops += 1;
            rayon::spawn(move || match crate::io::load_pfe(&path) {
                Ok(canvas_state) => {
                    let _ = sender.send(IoResult::PfeLoaded { canvas_state, path });
                }
                Err(e) => {
                    let _ = sender.send(IoResult::LoadFailed(format!(
                        "Failed to open project: {}",
                        e
                    )));
                }
            });
        } else {
            // Open as flat image on background thread (decode + tile in parallel)
            // Check for animated GIF/APNG first
            let sender = self.io_sender.clone();
            if self.pending_io_ops == 0 {
                self.io_ops_start_time = Some(current_time);
            }
            self.pending_io_ops += 1;
            rayon::spawn(move || {
                let ext = path
                    .extension()
                    .map(|e| e.to_string_lossy().to_lowercase())
                    .unwrap_or_default();
                let is_potential_animation = ext == "gif" || ext == "png";

                if is_potential_animation {
                    let anim_info = crate::io::detect_animation(&path);
                    if anim_info.is_animated && anim_info.frame_count > 1 {
                        // Animated file: decode all frames
                        let decode_result = match ext.as_str() {
                            "gif" => crate::io::decode_gif_frames(&path),
                            "png" => crate::io::decode_apng_frames(&path),
                            _ => unreachable!(),
                        };

                        match decode_result {
                            Ok(frames) if !frames.is_empty() => {
                                let format = if ext == "gif" {
                                    SaveFormat::Gif
                                } else {
                                    SaveFormat::Png
                                };
                                let width = frames[0].0.width();
                                let height = frames[0].0.height();

                                // Calculate FPS from average delay
                                let avg_delay = anim_info.avg_delay_ms.max(10) as f32;
                                let fps = (1000.0 / avg_delay).clamp(1.0, 60.0);

                                // First frame becomes the base layer
                                let first_tiled = TiledImage::from_rgba_image(&frames[0].0);

                                // Send first frame immediately so the project opens fast
                                let _ = sender.send(IoResult::AnimatedLoaded {
                                    tiled: first_tiled,
                                    width,
                                    height,
                                    path: path.clone(),
                                    format,
                                    fps,
                                    frame_count: frames.len() as u32,
                                });

                                // Remaining frames converted to TiledImages in background
                                if frames.len() > 1 {
                                    // We need to get the project_id back, but since it hasn't been
                                    // created yet, we use a second spawn that waits a frame.
                                    // Instead, we pack all remaining frames into one message.
                                    let remaining: Vec<(TiledImage, String)> = frames[1..]
                                        .iter()
                                        .enumerate()
                                        .map(|(i, (img, _delay))| {
                                            let tiled = TiledImage::from_rgba_image(img);
                                            let name = format!("Frame {}", i + 2);
                                            (tiled, name)
                                        })
                                        .collect();

                                    let _ = sender.send(IoResult::AnimatedFramesLoaded {
                                        path: path.clone(),
                                        frames: remaining,
                                    });
                                }
                            }
                            Ok(_) => {
                                let _ = sender.send(IoResult::LoadFailed(
                                    "Animated file contains no frames".to_string(),
                                ));
                            }
                            Err(e) => {
                                let _ = sender.send(IoResult::LoadFailed(e));
                            }
                        }
                        return;
                    }
                }

                // Non-animated or non-GIF/PNG: standard single-image load
                // Check for RAW camera files first
                let is_raw = crate::io::is_raw_extension(&ext);
                if is_raw {
                    match crate::io::decode_raw_image(&path) {
                        Ok(image) => {
                            let width = image.width();
                            let height = image.height();
                            let tiled = TiledImage::from_rgba_image(&image);
                            // RAW files open as PNG (user must Save As to choose format)
                            let _ = sender.send(IoResult::ImageLoaded {
                                tiled,
                                width,
                                height,
                                path,
                                format: SaveFormat::Png,
                            });
                        }
                        Err(e) => {
                            let _ = sender.send(IoResult::LoadFailed(format!("RAW: {}", e)));
                        }
                    }
                    return;
                }

                match image::open(&path) {
                    Ok(img) => {
                        let image = img.to_rgba8();
                        let width = image.width();
                        let height = image.height();
                        let tiled = TiledImage::from_rgba_image(&image);

                        let format = path
                            .extension()
                            .map(|ext| match ext.to_string_lossy().to_lowercase().as_str() {
                                "jpg" | "jpeg" => SaveFormat::Jpeg,
                                "webp" => SaveFormat::Webp,
                                "bmp" => SaveFormat::Bmp,
                                "tga" => SaveFormat::Tga,
                                "ico" => SaveFormat::Ico,
                                "tiff" | "tif" => SaveFormat::Tiff,
                                "gif" => SaveFormat::Gif,
                                _ => SaveFormat::Png,
                            })
                            .unwrap_or(SaveFormat::Png);

                        let _ = sender.send(IoResult::ImageLoaded {
                            tiled,
                            width,
                            height,
                            path,
                            format,
                        });
                    }
                    Err(e) => {
                        let _ = sender.send(IoResult::LoadFailed(format!("{}", e)));
                    }
                }
            });
        }
    }

    /// Handle save (quick save or open dialog) for the active project.
    /// For image formats, the encoding runs on a background thread so the UI
    /// stays responsive on large canvases.
    fn handle_save(&mut self, current_time: f64) {
        let idx = self.active_project_index;
        if idx >= self.projects.len() {
            return;
        }

        let has_path = self.projects[idx].file_handler.has_current_path();

        if has_path {
            let project = &mut self.projects[idx];
            let is_pfe = project.file_handler.last_format == SaveFormat::Pfe;
            let is_animated = project.file_handler.last_animated
                && project.file_handler.last_format.supports_animation();

            if is_pfe {
                // PFE save — build data snapshot, serialize in background
                if let Some(path) = project.file_handler.current_path.clone() {
                    let pfe_data = crate::io::build_pfe_v1(&project.canvas_state);
                    let sender = self.io_sender.clone();
                    if self.pending_io_ops == 0 {
                        self.io_ops_start_time = Some(current_time);
                    }
                    self.pending_io_ops += 1;
                    rayon::spawn(move || match crate::io::write_pfe_v1(&pfe_data, &path) {
                        Ok(()) => {
                            let _ = sender.send(IoResult::SaveComplete {
                                project_index: idx,
                                path,
                                format: SaveFormat::Pfe,
                                quality: 100,
                                tiff_compression: TiffCompression::None,
                                update_project_path: false,
                            });
                        }
                        Err(e) => {
                            let _ = sender.send(IoResult::SaveFailed {
                                project_index: idx,
                                error: format!("{}", e),
                            });
                        }
                    });
                }
            } else if is_animated {
                // Quick-save animated format (include all layers, even hidden)
                let frames: Vec<image::RgbaImage> = project
                    .canvas_state
                    .layers
                    .iter()
                    .map(|l| l.pixels.to_rgba_image())
                    .collect();
                let path = project.file_handler.current_path.clone().unwrap();
                let format = project.file_handler.last_format;
                let quality = project.file_handler.last_quality;
                let tiff_compression = project.file_handler.last_tiff_compression;
                let fps = project.file_handler.last_animation_fps;
                let gif_colors = project.file_handler.last_gif_colors;
                let gif_dither = project.file_handler.last_gif_dither;
                let sender = self.io_sender.clone();
                if self.pending_io_ops == 0 {
                    self.io_ops_start_time = Some(current_time);
                }
                self.pending_io_ops += 1;
                rayon::spawn(move || {
                    let result = match format {
                        SaveFormat::Gif => crate::io::encode_animated_gif(
                            &frames, fps, gif_colors, gif_dither, &path,
                        ),
                        SaveFormat::Png => crate::io::encode_animated_png(&frames, fps, &path),
                        _ => Err("Format does not support animation".to_string()),
                    };
                    match result {
                        Ok(()) => {
                            let _ = sender.send(IoResult::SaveComplete {
                                project_index: idx,
                                path,
                                format,
                                quality,
                                tiff_compression,
                                update_project_path: false,
                            });
                        }
                        Err(e) => {
                            let _ = sender.send(IoResult::SaveFailed {
                                project_index: idx,
                                error: e,
                            });
                        }
                    }
                });
            } else {
                // Static image save — composite on main thread (usually cached),
                // encode + write on background thread.
                let composite = project.canvas_state.composite();
                let path = project.file_handler.current_path.clone().unwrap();
                let format = project.file_handler.last_format;
                let quality = project.file_handler.last_quality;
                let tiff_compression = project.file_handler.last_tiff_compression;
                let sender = self.io_sender.clone();
                if self.pending_io_ops == 0 {
                    self.io_ops_start_time = Some(current_time);
                }
                self.pending_io_ops += 1;
                rayon::spawn(move || {
                    match crate::io::encode_and_write(
                        &composite,
                        &path,
                        format,
                        quality,
                        tiff_compression,
                    ) {
                        Ok(()) => {
                            let _ = sender.send(IoResult::SaveComplete {
                                project_index: idx,
                                path,
                                format,
                                quality,
                                tiff_compression,
                                update_project_path: false,
                            });
                        }
                        Err(e) => {
                            let _ = sender.send(IoResult::SaveFailed {
                                project_index: idx,
                                error: format!("{}", e),
                            });
                        }
                    }
                });
            }
        } else {
            // No existing path — extract all data we need from project
            let project = &self.projects[idx];
            let composite = project.canvas_state.composite();
            let was_animated = project.was_animated;
            let animation_fps = project.animation_fps;
            let frame_images: Option<Vec<image::RgbaImage>> =
                if project.canvas_state.layers.len() > 1 {
                    Some(
                        project
                            .canvas_state
                            .layers
                            .iter()
                            .map(|l| l.pixels.to_rgba_image())
                            .collect(),
                    )
                } else {
                    None
                };

            self.save_file_dialog.reset();
            self.save_file_dialog.set_source_image(&composite);

            if let Some(frames) = frame_images.as_ref() {
                self.save_file_dialog
                    .set_source_animated(frames, was_animated, animation_fps);
            }

            self.save_file_dialog.open = true;
        }
    }

    /// Save a specific project by index (only if it has a path — skips dialog).
    /// Returns true if a background save was launched.
    fn handle_save_project(&mut self, idx: usize, current_time: f64) -> bool {
        if idx >= self.projects.len() {
            return false;
        }
        if !self.projects[idx].file_handler.has_current_path() {
            return false;
        }
        if !self.projects[idx].is_dirty {
            return false;
        }

        let project = &mut self.projects[idx];
        let is_pfe = project.file_handler.last_format == SaveFormat::Pfe;
        let is_animated = project.file_handler.last_animated
            && project.file_handler.last_format.supports_animation();

        if is_pfe {
            if let Some(path) = project.file_handler.current_path.clone() {
                let pfe_data = crate::io::build_pfe_v1(&project.canvas_state);
                let sender = self.io_sender.clone();
                if self.pending_io_ops == 0 {
                    self.io_ops_start_time = Some(current_time);
                }
                self.pending_io_ops += 1;
                rayon::spawn(move || match crate::io::write_pfe_v1(&pfe_data, &path) {
                    Ok(()) => {
                        let _ = sender.send(IoResult::SaveComplete {
                            project_index: idx,
                            path,
                            format: SaveFormat::Pfe,
                            quality: 100,
                            tiff_compression: TiffCompression::None,
                            update_project_path: false,
                        });
                    }
                    Err(e) => {
                        let _ = sender.send(IoResult::SaveFailed {
                            project_index: idx,
                            error: format!("{}", e),
                        });
                    }
                });
            }
        } else if is_animated {
            let frames: Vec<image::RgbaImage> = project
                .canvas_state
                .layers
                .iter()
                .map(|l| l.pixels.to_rgba_image())
                .collect();
            let path = project.file_handler.current_path.clone().unwrap();
            let format = project.file_handler.last_format;
            let quality = project.file_handler.last_quality;
            let tiff_compression = project.file_handler.last_tiff_compression;
            let fps = project.file_handler.last_animation_fps;
            let gif_colors = project.file_handler.last_gif_colors;
            let gif_dither = project.file_handler.last_gif_dither;
            let sender = self.io_sender.clone();
            if self.pending_io_ops == 0 {
                self.io_ops_start_time = Some(current_time);
            }
            self.pending_io_ops += 1;
            rayon::spawn(move || {
                let result = match format {
                    SaveFormat::Gif => {
                        crate::io::encode_animated_gif(&frames, fps, gif_colors, gif_dither, &path)
                    }
                    SaveFormat::Png => crate::io::encode_animated_png(&frames, fps, &path),
                    _ => Err("Format does not support animation".to_string()),
                };
                match result {
                    Ok(()) => {
                        let _ = sender.send(IoResult::SaveComplete {
                            project_index: idx,
                            path,
                            format,
                            quality,
                            tiff_compression,
                            update_project_path: false,
                        });
                    }
                    Err(e) => {
                        let _ = sender.send(IoResult::SaveFailed {
                            project_index: idx,
                            error: e,
                        });
                    }
                }
            });
        } else {
            let composite = project.canvas_state.composite();
            let path = project.file_handler.current_path.clone().unwrap();
            let format = project.file_handler.last_format;
            let quality = project.file_handler.last_quality;
            let tiff_compression = project.file_handler.last_tiff_compression;
            let sender = self.io_sender.clone();
            if self.pending_io_ops == 0 {
                self.io_ops_start_time = Some(current_time);
            }
            self.pending_io_ops += 1;
            rayon::spawn(move || {
                match crate::io::encode_and_write(
                    &composite,
                    &path,
                    format,
                    quality,
                    tiff_compression,
                ) {
                    Ok(()) => {
                        let _ = sender.send(IoResult::SaveComplete {
                            project_index: idx,
                            path,
                            format,
                            quality,
                            tiff_compression,
                            update_project_path: false,
                        });
                    }
                    Err(e) => {
                        let _ = sender.send(IoResult::SaveFailed {
                            project_index: idx,
                            error: format!("{}", e),
                        });
                    }
                }
            });
        }
        true
    }

    /// Save all dirty projects that already have a file path.
    fn handle_save_all(&mut self, current_time: f64) {
        let count = self.projects.len();
        for i in 0..count {
            self.handle_save_project(i, current_time);
        }
    }
}

// --- Operation Helpers (snapshot undo for menus) ---
impl PaintFEApp {
    /// Spawn a filter job on a background thread.
    ///
    /// `description`: undo entry label (e.g. "Gaussian Blur").
    /// `layer_idx`: which layer to operate on.
    /// `original_pixels`: clone of layer pixels before the filter.
    /// `original_flat`: pre-flattened RGBA for the filter function.
    /// `filter_fn`: closure that takes the flat image and returns the processed image.
    ///
    /// The closure runs on `rayon::spawn`; when done it sends a `FilterResult`
    /// back via the channel.  The main thread polls the channel in `update()`.
    fn spawn_filter_job(
        &mut self,
        current_time: f64,
        description: String,
        layer_idx: usize,
        original_pixels: TiledImage,
        original_flat: image::RgbaImage,
        filter_fn: impl FnOnce(&image::RgbaImage) -> image::RgbaImage + Send + 'static,
    ) {
        self.spawn_filter_job_with_token(
            current_time,
            description,
            layer_idx,
            original_pixels,
            original_flat,
            0,
            filter_fn,
        );
    }

    /// Spawn a live-preview filter job. The token is incremented so any in-flight
    /// job from a previous slider position is automatically discarded on arrival.
    fn spawn_preview_job(
        &mut self,
        current_time: f64,
        description: String,
        layer_idx: usize,
        original_pixels: TiledImage,
        original_flat: image::RgbaImage,
        filter_fn: impl FnOnce(&image::RgbaImage) -> image::RgbaImage + Send + 'static,
    ) {
        self.preview_job_token = self.preview_job_token.wrapping_add(1);
        let token = self.preview_job_token;
        self.spawn_filter_job_with_token(
            current_time,
            description,
            layer_idx,
            original_pixels,
            original_flat,
            token,
            filter_fn,
        );
    }

    fn spawn_filter_job_with_token(
        &mut self,
        current_time: f64,
        description: String,
        layer_idx: usize,
        original_pixels: TiledImage,
        original_flat: image::RgbaImage,
        preview_token: u64,
        filter_fn: impl FnOnce(&image::RgbaImage) -> image::RgbaImage + Send + 'static,
    ) {
        let sender = self.filter_sender.clone();
        let project_index = self.active_project_index;
        if self.pending_filter_jobs == 0 {
            self.filter_ops_start_time = Some(current_time);
        }
        self.filter_status_description = description.clone();
        self.pending_filter_jobs += 1;
        rayon::spawn(move || {
            let result_flat = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                filter_fn(&original_flat)
            }));
            match result_flat {
                Ok(processed) => {
                    let result_tiled = TiledImage::from_rgba_image(&processed);
                    let _ = sender.send(FilterResult {
                        project_index,
                        layer_idx,
                        original_pixels,
                        result_pixels: result_tiled,
                        description,
                        preview_token,
                    });
                }
                Err(panic_info) => {
                    // Panic in filter — revert to original (no-op: don't send)
                    let msg = if let Some(s) = panic_info.downcast_ref::<&str>() {
                        s.to_string()
                    } else if let Some(s) = panic_info.downcast_ref::<String>() {
                        s.to_string()
                    } else {
                        "unknown panic payload".to_string()
                    };
                    eprintln!("Filter '{}' panicked: {}", description, msg);
                    let _ = sender.send(FilterResult {
                        project_index,
                        layer_idx,
                        original_pixels: original_pixels.clone(),
                        result_pixels: original_pixels,
                        description,
                        preview_token,
                    });
                }
            }
        });
    }

    /// Perform a canvas operation with full-snapshot undo.
    /// The closure receives `&mut CanvasState` and should apply the operation.
    fn do_snapshot_op(&mut self, description: &str, op: impl FnOnce(&mut CanvasState)) {
        if let Some(project) = self.active_project_mut() {
            let mut cmd = SnapshotCommand::new(description.to_string(), &project.canvas_state);
            op(&mut project.canvas_state);
            cmd.set_after(&project.canvas_state);
            project.history.push(Box::new(cmd));
            project.mark_dirty();
        }
    }

    /// Commit the active paste overlay.
    /// Extraction is already in history (for MovePixels) — this pushes a separate commit entry.
    fn commit_paste_overlay(&mut self) {
        if let Some(overlay) = self.paste_overlay.take() {
            let desc = if self.is_move_pixels_active {
                "Move Pixels"
            } else {
                "Paste"
            };
            self.do_snapshot_op(desc, |s| {
                s.clear_preview_state();
                overlay.commit(s);
            });
            self.is_move_pixels_active = false;
        }
    }

    /// Cancel the active paste overlay.
    /// If MovePixels is active, undo the extraction entry to restore original pixels.
    fn cancel_paste_overlay(&mut self) {
        self.paste_overlay = None;
        if self.is_move_pixels_active {
            // Undo the extraction snapshot we already pushed
            if let Some(project) = self.active_project_mut() {
                project.history.undo(&mut project.canvas_state);
                project.canvas_state.clear_preview_state();
            }
            self.is_move_pixels_active = false;
        } else if let Some(project) = self.active_project_mut() {
            project.canvas_state.clear_preview_state();
            project.canvas_state.mark_dirty(None);
        }
    }

    /// Same as do_snapshot_op, but only captures the active layer (not all layers).
    /// Much more memory-efficient for single-layer operations.
    fn do_layer_snapshot_op(&mut self, description: &str, op: impl FnOnce(&mut CanvasState)) {
        if let Some(project) = self.active_project_mut() {
            let mut cmd =
                SingleLayerSnapshotCommand::new(description.to_string(), &project.canvas_state);
            op(&mut project.canvas_state);
            cmd.set_after(&project.canvas_state);
            project.history.push(Box::new(cmd));
            project.mark_dirty();
        }
    }

    /// Like `do_layer_snapshot_op`, but splits the borrow so the closure can also
    /// access the GPU renderer for compute-shader operations.
    fn do_gpu_snapshot_op(
        &mut self,
        description: &str,
        op: impl FnOnce(&mut CanvasState, &crate::gpu::GpuRenderer),
    ) {
        let active_idx = self.active_project_index;
        if let Some(project) = self.projects.get_mut(active_idx) {
            let mut cmd =
                SingleLayerSnapshotCommand::new(description.to_string(), &project.canvas_state);
            op(&mut project.canvas_state, &self.canvas.gpu_renderer);
            cmd.set_after(&project.canvas_state);
            project.history.push(Box::new(cmd));
            project.mark_dirty();
        }
    }

    /// Downscale an image for fast low-res live preview.
    /// Returns (downscaled_image, scale_factor).  If the image is already small
    /// enough the original is returned with factor 1.0.
    fn make_preview_image(flat: &RgbaImage, max_edge: u32) -> (RgbaImage, f32) {
        let (w, h) = (flat.width(), flat.height());
        let longest = w.max(h);
        if longest <= max_edge {
            return (flat.clone(), 1.0);
        }
        let scale = max_edge as f32 / longest as f32;
        let nw = ((w as f32 * scale).round() as u32).max(1);
        let nh = ((h as f32 * scale).round() as u32).max(1);
        let small = image::imageops::resize(flat, nw, nh, image::imageops::FilterType::Triangle);
        (small, scale)
    }

    /// Upscale a processed preview back to the original layer dimensions then
    /// write it into the layer for on-screen display.
    fn apply_preview_to_layer(
        state: &mut CanvasState,
        layer_idx: usize,
        preview: &RgbaImage,
        orig_w: u32,
        orig_h: u32,
    ) {
        if layer_idx >= state.layers.len() {
            return;
        }
        let upscaled = if preview.width() == orig_w && preview.height() == orig_h {
            preview.clone()
        } else {
            image::imageops::resize(
                preview,
                orig_w,
                orig_h,
                image::imageops::FilterType::Triangle,
            )
        };
        state.layers[layer_idx].pixels = TiledImage::from_rgba_image(&upscaled);
        state.mark_dirty(None);
    }

    /// Apply a low‑res effect preview: runs the effect on `preview_flat_small`,
    /// then upscales to the original layer dimensions.
    /// `effect_fn` receives the small image and returns the processed result.
    fn apply_lowres_preview<F>(
        state: &mut CanvasState,
        layer_idx: usize,
        preview_small: &RgbaImage,
        orig_flat: Option<&RgbaImage>,
        effect_fn: F,
    ) where
        F: FnOnce(&RgbaImage) -> RgbaImage,
    {
        let result = effect_fn(preview_small);
        let (orig_w, orig_h) = orig_flat
            .map(|f| (f.width(), f.height()))
            .unwrap_or((preview_small.width(), preview_small.height()));
        Self::apply_preview_to_layer(state, layer_idx, &result, orig_w, orig_h);
    }

    /// Compute a low-res preview (run effect on the small image then upscale)
    /// but do NOT write it to the layer.  Returns the upscaled result so the
    /// caller can drive progressive reveal.
    fn compute_lowres_preview<F>(
        preview_small: &RgbaImage,
        orig_flat: Option<&RgbaImage>,
        effect_fn: F,
    ) -> RgbaImage
    where
        F: FnOnce(&RgbaImage) -> RgbaImage,
    {
        let result = effect_fn(preview_small);
        let (orig_w, orig_h) = orig_flat
            .map(|f| (f.width(), f.height()))
            .unwrap_or((preview_small.width(), preview_small.height()));
        if result.width() == orig_w && result.height() == orig_h {
            result
        } else {
            image::imageops::resize(
                &result,
                orig_w,
                orig_h,
                image::imageops::FilterType::Triangle,
            )
        }
    }

    /// Apply an effect at full resolution and commit it to the layer.
    /// Used when the user clicks OK.
    fn apply_fullres_effect<F>(
        state: &mut CanvasState,
        layer_idx: usize,
        original_flat: &RgbaImage,
        effect_fn: F,
    ) where
        F: FnOnce(&RgbaImage) -> RgbaImage,
    {
        if layer_idx >= state.layers.len() {
            return;
        }
        let result = effect_fn(original_flat);
        state.layers[layer_idx].pixels = TiledImage::from_rgba_image(&result);
        state.mark_dirty(None);
    }

    /// Process whichever modal dialog is currently active.
    fn process_active_dialog(&mut self, ctx: &egui::Context) {
        // We need to take ownership temporarily to satisfy the borrow checker,
        // since dialogs need &mut self (via active_project_mut) on completion.
        let mut dialog = std::mem::take(&mut self.active_dialog);

        match &mut dialog {
            ActiveDialog::None => {}

            ActiveDialog::ResizeImage(dlg) => match dlg.show(ctx) {
                DialogResult::Ok((w, h, interp)) => {
                    self.active_dialog = ActiveDialog::None;
                    if let Some(project) = self.active_project_mut() {
                        let mut cmd =
                            SnapshotCommand::new("Resize Image".to_string(), &project.canvas_state);
                        crate::ops::transform::resize_image(
                            &mut project.canvas_state,
                            w,
                            h,
                            interp,
                        );
                        cmd.set_after(&project.canvas_state);
                        project.history.push(Box::new(cmd));
                        project.mark_dirty();
                    }
                    return;
                }
                DialogResult::Cancel => {
                    self.active_dialog = ActiveDialog::None;
                    return;
                }
                _ => {}
            },

            ActiveDialog::ResizeCanvas(dlg) => {
                let secondary = self.colors_panel.get_secondary_color_f32();
                match dlg.show(ctx, secondary) {
                    DialogResult::Ok((w, h, anchor, fill)) => {
                        self.active_dialog = ActiveDialog::None;
                        if let Some(project) = self.active_project_mut() {
                            let mut cmd = SnapshotCommand::new(
                                "Resize Canvas".to_string(),
                                &project.canvas_state,
                            );
                            crate::ops::transform::resize_canvas(
                                &mut project.canvas_state,
                                w,
                                h,
                                anchor,
                                fill,
                            );
                            cmd.set_after(&project.canvas_state);
                            project.history.push(Box::new(cmd));
                            project.mark_dirty();
                        }
                        return;
                    }
                    DialogResult::Cancel => {
                        self.active_dialog = ActiveDialog::None;
                        return;
                    }
                    _ => {}
                }
            }

            ActiveDialog::GaussianBlur(dlg) => {
                match dlg.show(ctx) {
                    DialogResult::Changed => {
                        // Spawn async preview job (token-tracked, interrupts stale previews)
                        let idx = dlg.layer_idx;
                        if let (Some(original), Some(flat)) =
                            (&dlg.original_pixels, &dlg.original_flat)
                        {
                            let sigma = dlg.sigma;
                            let sel_mask = self
                                .active_project()
                                .and_then(|p| p.canvas_state.selection_mask.clone());
                            self.spawn_preview_job(
                                ctx.input(|i| i.time),
                                "Gaussian Blur".to_string(),
                                idx,
                                original.clone(),
                                flat.clone(),
                                move |img| {
                                    crate::ops::filters::blur_with_selection_pub(
                                        img,
                                        sigma,
                                        sel_mask.as_ref(),
                                    )
                                },
                            );
                        }
                    }
                    DialogResult::Ok(_sigma) => {
                        self.preview_job_token = self.preview_job_token.wrapping_add(1);
                        // Spawn full-resolution blur on background thread.
                        let sigma = dlg.sigma;
                        let idx = dlg.layer_idx;
                        self.active_dialog = ActiveDialog::None;
                        if let (Some(original_pixels), Some(original_flat)) =
                            (&dlg.original_pixels, &dlg.original_flat)
                        {
                            // Restore originals so the layer isn't left with preview data
                            if let Some(project) = self.active_project_mut()
                                && idx < project.canvas_state.layers.len()
                            {
                                project.canvas_state.layers[idx].pixels = original_pixels.clone();
                                project.canvas_state.mark_dirty(None);
                            }
                            let orig_clone = original_pixels.clone();
                            let flat_clone = original_flat.clone();
                            let sel_mask = self
                                .active_project()
                                .and_then(|p| p.canvas_state.selection_mask.clone());
                            self.spawn_filter_job(
                                ctx.input(|i| i.time),
                                "Gaussian Blur".to_string(),
                                idx,
                                orig_clone,
                                flat_clone,
                                move |flat| {
                                    crate::ops::filters::blur_with_selection_pub(
                                        flat,
                                        sigma,
                                        sel_mask.as_ref(),
                                    )
                                },
                            );
                        }
                        return;
                    }
                    DialogResult::Cancel => {
                        self.preview_job_token = self.preview_job_token.wrapping_add(1);
                        // Restore original pixels.
                        let idx = dlg.layer_idx;
                        if let Some(original) = &dlg.original_pixels
                            && let Some(project) = self.active_project_mut()
                        {
                            if let Some(layer) = project.canvas_state.layers.get_mut(idx) {
                                layer.pixels = original.clone();
                            }
                            project.canvas_state.mark_dirty(None);
                        }
                        self.active_dialog = ActiveDialog::None;
                        return;
                    }
                    _ => {}
                }
            }

            ActiveDialog::LayerTransform(dlg) => {
                match dlg.show(ctx) {
                    DialogResult::Changed => {
                        // Live preview: use pre-flattened original to skip
                        // the expensive clone → flatten round-trip.
                        let rz = dlg.rotation_z;
                        let rx = dlg.rotation_x;
                        let ry = dlg.rotation_y;
                        let scale = dlg.scale_percent / 100.0;
                        let offset = (dlg.offset_x, dlg.offset_y);
                        let idx = dlg.layer_idx;
                        if let Some(flat) = &dlg.original_flat
                            && let Some(project) = self.active_project_mut()
                        {
                            crate::ops::transform::affine_transform_layer_from_flat(
                                &mut project.canvas_state,
                                idx,
                                rz,
                                rx,
                                ry,
                                scale,
                                offset,
                                flat,
                            );
                        }
                    }
                    DialogResult::Ok((_rot_z, _rot_x, _rot_y, _scale, _offset)) => {
                        // Accept current preview — push undo.
                        self.active_dialog = ActiveDialog::None;
                        if let Some(project) = self.active_project_mut() {
                            let idx = dlg.layer_idx;
                            if let Some(original) = &dlg.original_pixels
                                && idx < project.canvas_state.layers.len()
                            {
                                let transformed = project.canvas_state.layers[idx].pixels.clone();
                                project.canvas_state.layers[idx].pixels = original.clone();
                                let mut cmd = SingleLayerSnapshotCommand::new_for_layer(
                                    "Layer Transform".to_string(),
                                    &project.canvas_state,
                                    idx,
                                );
                                project.canvas_state.layers[idx].pixels = transformed;
                                cmd.set_after(&project.canvas_state);
                                project.history.push(Box::new(cmd));
                            }
                            project.mark_dirty();
                        }
                        return;
                    }
                    DialogResult::Cancel => {
                        // Restore original pixels.
                        let idx = dlg.layer_idx;
                        if let Some(original) = &dlg.original_pixels
                            && let Some(project) = self.active_project_mut()
                        {
                            if let Some(layer) = project.canvas_state.layers.get_mut(idx) {
                                layer.pixels = original.clone();
                            }
                            project.canvas_state.mark_dirty(None);
                        }
                        self.active_dialog = ActiveDialog::None;
                        return;
                    }
                    _ => {}
                }
            }

            // ================================================================
            // BRIGHTNESS / CONTRAST
            // ================================================================
            ActiveDialog::BrightnessContrast(dlg) => match dlg.show(ctx) {
                DialogResult::Changed => {
                    let brightness = dlg.brightness;
                    let contrast = dlg.contrast;
                    let idx = dlg.layer_idx;
                    if let Some(flat) = &dlg.original_flat {
                        let active_idx = self.active_project_index;
                        if let Some(project) = self.projects.get_mut(active_idx) {
                            crate::ops::adjustments::brightness_contrast_from_flat_gpu(
                                &mut project.canvas_state,
                                idx,
                                brightness,
                                contrast,
                                flat,
                                &self.canvas.gpu_renderer,
                            );
                        }
                    }
                }
                DialogResult::Ok(_) => {
                    self.active_dialog = ActiveDialog::None;
                    if let Some(project) = self.active_project_mut() {
                        let idx = dlg.layer_idx;
                        if let Some(original) = &dlg.original_pixels
                            && idx < project.canvas_state.layers.len()
                        {
                            let adjusted = project.canvas_state.layers[idx].pixels.clone();
                            project.canvas_state.layers[idx].pixels = original.clone();
                            let mut cmd = SingleLayerSnapshotCommand::new_for_layer(
                                "Brightness/Contrast".to_string(),
                                &project.canvas_state,
                                idx,
                            );
                            project.canvas_state.layers[idx].pixels = adjusted;
                            cmd.set_after(&project.canvas_state);
                            project.history.push(Box::new(cmd));
                        }
                        project.mark_dirty();
                    }
                    return;
                }
                DialogResult::Cancel => {
                    let idx = dlg.layer_idx;
                    if let Some(original) = &dlg.original_pixels
                        && let Some(project) = self.active_project_mut()
                    {
                        if let Some(layer) = project.canvas_state.layers.get_mut(idx) {
                            layer.pixels = original.clone();
                        }
                        project.canvas_state.mark_dirty(None);
                    }
                    self.active_dialog = ActiveDialog::None;
                    return;
                }
                _ => {}
            },

            // ================================================================
            // HUE / SATURATION
            // ================================================================
            ActiveDialog::HueSaturation(dlg) => match dlg.show(ctx) {
                DialogResult::Changed => {
                    let hue = dlg.hue;
                    let sat = dlg.saturation;
                    let light = dlg.lightness;
                    let idx = dlg.layer_idx;
                    let per_band = dlg.per_band;
                    let bands = dlg.bands;
                    if let Some(flat) = &dlg.original_flat {
                        let active_idx = self.active_project_index;
                        if let Some(project) = self.projects.get_mut(active_idx) {
                            if per_band {
                                crate::ops::adjustments::hue_saturation_per_band_from_flat(
                                    &mut project.canvas_state,
                                    idx,
                                    hue,
                                    sat,
                                    light,
                                    &bands,
                                    flat,
                                );
                            } else {
                                crate::ops::adjustments::hue_saturation_lightness_from_flat_gpu(
                                    &mut project.canvas_state,
                                    idx,
                                    hue,
                                    sat,
                                    light,
                                    flat,
                                    &self.canvas.gpu_renderer,
                                );
                            }
                        }
                    }
                }
                DialogResult::Ok(_) => {
                    self.active_dialog = ActiveDialog::None;
                    if let Some(project) = self.active_project_mut() {
                        let idx = dlg.layer_idx;
                        let label = if dlg.per_band {
                            "Hue/Saturation (Per Band)"
                        } else {
                            "Hue/Saturation"
                        };
                        if let Some(original) = &dlg.original_pixels
                            && idx < project.canvas_state.layers.len()
                        {
                            let adjusted = project.canvas_state.layers[idx].pixels.clone();
                            project.canvas_state.layers[idx].pixels = original.clone();
                            let mut cmd = SingleLayerSnapshotCommand::new_for_layer(
                                label.to_string(),
                                &project.canvas_state,
                                idx,
                            );
                            project.canvas_state.layers[idx].pixels = adjusted;
                            cmd.set_after(&project.canvas_state);
                            project.history.push(Box::new(cmd));
                        }
                        project.mark_dirty();
                    }
                    return;
                }
                DialogResult::Cancel => {
                    let idx = dlg.layer_idx;
                    if let Some(original) = &dlg.original_pixels
                        && let Some(project) = self.active_project_mut()
                    {
                        if let Some(layer) = project.canvas_state.layers.get_mut(idx) {
                            layer.pixels = original.clone();
                        }
                        project.canvas_state.mark_dirty(None);
                    }
                    self.active_dialog = ActiveDialog::None;
                    return;
                }
                _ => {}
            },

            // ================================================================
            // EXPOSURE
            // ================================================================
            ActiveDialog::Exposure(dlg) => match dlg.show(ctx) {
                DialogResult::Changed => {
                    let exposure = dlg.exposure;
                    let idx = dlg.layer_idx;
                    if let Some(flat) = &dlg.original_flat
                        && let Some(project) = self.active_project_mut()
                    {
                        crate::ops::adjustments::exposure_from_flat(
                            &mut project.canvas_state,
                            idx,
                            exposure,
                            flat,
                        );
                    }
                }
                DialogResult::Ok(_) => {
                    self.active_dialog = ActiveDialog::None;
                    if let Some(project) = self.active_project_mut() {
                        let idx = dlg.layer_idx;
                        if let Some(original) = &dlg.original_pixels
                            && idx < project.canvas_state.layers.len()
                        {
                            let adjusted = project.canvas_state.layers[idx].pixels.clone();
                            project.canvas_state.layers[idx].pixels = original.clone();
                            let mut cmd = SingleLayerSnapshotCommand::new_for_layer(
                                "Exposure".to_string(),
                                &project.canvas_state,
                                idx,
                            );
                            project.canvas_state.layers[idx].pixels = adjusted;
                            cmd.set_after(&project.canvas_state);
                            project.history.push(Box::new(cmd));
                        }
                        project.mark_dirty();
                    }
                    return;
                }
                DialogResult::Cancel => {
                    let idx = dlg.layer_idx;
                    if let Some(original) = &dlg.original_pixels
                        && let Some(project) = self.active_project_mut()
                    {
                        if let Some(layer) = project.canvas_state.layers.get_mut(idx) {
                            layer.pixels = original.clone();
                        }
                        project.canvas_state.mark_dirty(None);
                    }
                    self.active_dialog = ActiveDialog::None;
                    return;
                }
                _ => {}
            },

            // ================================================================
            // HIGHLIGHTS / SHADOWS
            // ================================================================
            ActiveDialog::HighlightsShadows(dlg) => match dlg.show(ctx) {
                DialogResult::Changed => {
                    let shadows = dlg.shadows;
                    let highlights = dlg.highlights;
                    let idx = dlg.layer_idx;
                    if let Some(flat) = &dlg.original_flat
                        && let Some(project) = self.active_project_mut()
                    {
                        crate::ops::adjustments::highlights_shadows_from_flat(
                            &mut project.canvas_state,
                            idx,
                            shadows,
                            highlights,
                            flat,
                        );
                    }
                }
                DialogResult::Ok(_) => {
                    self.active_dialog = ActiveDialog::None;
                    if let Some(project) = self.active_project_mut() {
                        let idx = dlg.layer_idx;
                        if let Some(original) = &dlg.original_pixels
                            && idx < project.canvas_state.layers.len()
                        {
                            let adjusted = project.canvas_state.layers[idx].pixels.clone();
                            project.canvas_state.layers[idx].pixels = original.clone();
                            let mut cmd = SingleLayerSnapshotCommand::new_for_layer(
                                "Highlights/Shadows".to_string(),
                                &project.canvas_state,
                                idx,
                            );
                            project.canvas_state.layers[idx].pixels = adjusted;
                            cmd.set_after(&project.canvas_state);
                            project.history.push(Box::new(cmd));
                        }
                        project.mark_dirty();
                    }
                    return;
                }
                DialogResult::Cancel => {
                    let idx = dlg.layer_idx;
                    if let Some(original) = &dlg.original_pixels
                        && let Some(project) = self.active_project_mut()
                    {
                        if let Some(layer) = project.canvas_state.layers.get_mut(idx) {
                            layer.pixels = original.clone();
                        }
                        project.canvas_state.mark_dirty(None);
                    }
                    self.active_dialog = ActiveDialog::None;
                    return;
                }
                _ => {}
            },

            // ================================================================
            // LEVELS
            // ================================================================
            ActiveDialog::Levels(dlg) => match dlg.show(ctx) {
                DialogResult::Changed => {
                    let res = dlg.as_result();
                    let idx = dlg.layer_idx;
                    if let Some(flat) = &dlg.original_flat
                        && let Some(project) = self.active_project_mut()
                    {
                        crate::ops::adjustments::levels_from_flat_per_channel(
                            &mut project.canvas_state,
                            idx,
                            res.master,
                            res.r_ch,
                            res.g_ch,
                            res.b_ch,
                            flat,
                        );
                    }
                }
                DialogResult::Ok(_) => {
                    self.active_dialog = ActiveDialog::None;
                    if let Some(project) = self.active_project_mut() {
                        let idx = dlg.layer_idx;
                        if let Some(original) = &dlg.original_pixels
                            && idx < project.canvas_state.layers.len()
                        {
                            let adjusted = project.canvas_state.layers[idx].pixels.clone();
                            project.canvas_state.layers[idx].pixels = original.clone();
                            let mut cmd = SingleLayerSnapshotCommand::new_for_layer(
                                "Levels".to_string(),
                                &project.canvas_state,
                                idx,
                            );
                            project.canvas_state.layers[idx].pixels = adjusted;
                            cmd.set_after(&project.canvas_state);
                            project.history.push(Box::new(cmd));
                        }
                        project.mark_dirty();
                    }
                    return;
                }
                DialogResult::Cancel => {
                    let idx = dlg.layer_idx;
                    if let Some(original) = &dlg.original_pixels
                        && let Some(project) = self.active_project_mut()
                    {
                        if let Some(layer) = project.canvas_state.layers.get_mut(idx) {
                            layer.pixels = original.clone();
                        }
                        project.canvas_state.mark_dirty(None);
                    }
                    self.active_dialog = ActiveDialog::None;
                    return;
                }
                _ => {}
            },

            // ================================================================
            // CURVES
            // ================================================================
            ActiveDialog::Curves(dlg) => match dlg.show(ctx) {
                DialogResult::Changed => {
                    let ch_data: [(&[(f32, f32)], bool); 5] = [
                        (&dlg.channels[0].points, dlg.channels[0].enabled),
                        (&dlg.channels[1].points, dlg.channels[1].enabled),
                        (&dlg.channels[2].points, dlg.channels[2].enabled),
                        (&dlg.channels[3].points, dlg.channels[3].enabled),
                        (&dlg.channels[4].points, dlg.channels[4].enabled),
                    ];
                    let idx = dlg.layer_idx;
                    if let Some(flat) = &dlg.original_flat
                        && let Some(project) = self.active_project_mut()
                    {
                        crate::ops::adjustments::curves_from_flat_multi(
                            &mut project.canvas_state,
                            idx,
                            &ch_data,
                            flat,
                        );
                    }
                }
                DialogResult::Ok(_) => {
                    self.active_dialog = ActiveDialog::None;
                    if let Some(project) = self.active_project_mut() {
                        let idx = dlg.layer_idx;
                        if let Some(original) = &dlg.original_pixels
                            && idx < project.canvas_state.layers.len()
                        {
                            let adjusted = project.canvas_state.layers[idx].pixels.clone();
                            project.canvas_state.layers[idx].pixels = original.clone();
                            let mut cmd = SingleLayerSnapshotCommand::new_for_layer(
                                "Curves".to_string(),
                                &project.canvas_state,
                                idx,
                            );
                            project.canvas_state.layers[idx].pixels = adjusted;
                            cmd.set_after(&project.canvas_state);
                            project.history.push(Box::new(cmd));
                        }
                        project.mark_dirty();
                    }
                    return;
                }
                DialogResult::Cancel => {
                    let idx = dlg.layer_idx;
                    if let Some(original) = &dlg.original_pixels
                        && let Some(project) = self.active_project_mut()
                    {
                        if let Some(layer) = project.canvas_state.layers.get_mut(idx) {
                            layer.pixels = original.clone();
                        }
                        project.canvas_state.mark_dirty(None);
                    }
                    self.active_dialog = ActiveDialog::None;
                    return;
                }
                _ => {}
            },

            // ================================================================
            // TEMPERATURE / TINT
            // ================================================================
            ActiveDialog::TemperatureTint(dlg) => match dlg.show(ctx) {
                DialogResult::Changed => {
                    let temperature = dlg.temperature;
                    let tint = dlg.tint;
                    let idx = dlg.layer_idx;
                    if let Some(flat) = &dlg.original_flat
                        && let Some(project) = self.active_project_mut()
                    {
                        crate::ops::adjustments::temperature_tint_from_flat(
                            &mut project.canvas_state,
                            idx,
                            temperature,
                            tint,
                            flat,
                        );
                    }
                }
                DialogResult::Ok(_) => {
                    self.active_dialog = ActiveDialog::None;
                    if let Some(project) = self.active_project_mut() {
                        let idx = dlg.layer_idx;
                        if let Some(original) = &dlg.original_pixels
                            && idx < project.canvas_state.layers.len()
                        {
                            let adjusted = project.canvas_state.layers[idx].pixels.clone();
                            project.canvas_state.layers[idx].pixels = original.clone();
                            let mut cmd = SingleLayerSnapshotCommand::new_for_layer(
                                "Temperature/Tint".to_string(),
                                &project.canvas_state,
                                idx,
                            );
                            project.canvas_state.layers[idx].pixels = adjusted;
                            cmd.set_after(&project.canvas_state);
                            project.history.push(Box::new(cmd));
                        }
                        project.mark_dirty();
                    }
                    return;
                }
                DialogResult::Cancel => {
                    let idx = dlg.layer_idx;
                    if let Some(original) = &dlg.original_pixels
                        && let Some(project) = self.active_project_mut()
                    {
                        if let Some(layer) = project.canvas_state.layers.get_mut(idx) {
                            layer.pixels = original.clone();
                        }
                        project.canvas_state.mark_dirty(None);
                    }
                    self.active_dialog = ActiveDialog::None;
                    return;
                }
                _ => {}
            },

            // ================================================================
            // EFFECT DIALOGS — macroified common patterns
            // ================================================================

            // Helper closure-like pattern: all effect dialogs follow the same
            // Changed / Ok / Cancel structure, just with different apply functions.
            ActiveDialog::BokehBlur(dlg) => {
                match dlg.show(ctx) {
                    DialogResult::Changed => {
                        // Spawn async preview job (token-tracked, interrupts stale previews)
                        dlg.poll_flat();
                        let idx = dlg.layer_idx;
                        if let (Some(original), Some(flat)) =
                            (&dlg.original_pixels, &dlg.original_flat)
                        {
                            let radius = dlg.radius;
                            self.spawn_preview_job(
                                ctx.input(|i| i.time),
                                "Bokeh Blur".to_string(),
                                idx,
                                original.clone(),
                                flat.clone(),
                                move |img| crate::ops::effects::bokeh_blur_core(img, radius, None),
                            );
                        }
                    }
                    DialogResult::Ok(_) => {
                        self.preview_job_token = self.preview_job_token.wrapping_add(1);
                        // Apply at full resolution
                        let idx = dlg.layer_idx;
                        if let Some(flat) = &dlg.original_flat {
                            let radius = dlg.radius;
                            if let Some(project) = self.active_project_mut() {
                                Self::apply_fullres_effect(
                                    &mut project.canvas_state,
                                    idx,
                                    flat,
                                    |img| crate::ops::effects::bokeh_blur_core(img, radius, None),
                                );
                            }
                        }
                        self.active_dialog = ActiveDialog::None;
                        if let Some(project) = self.active_project_mut() {
                            let idx = dlg.layer_idx;
                            if let Some(original) = &dlg.original_pixels
                                && idx < project.canvas_state.layers.len()
                            {
                                let adjusted = project.canvas_state.layers[idx].pixels.clone();
                                project.canvas_state.layers[idx].pixels = original.clone();
                                let mut cmd = SingleLayerSnapshotCommand::new_for_layer(
                                    "Bokeh Blur".to_string(),
                                    &project.canvas_state,
                                    idx,
                                );
                                project.canvas_state.layers[idx].pixels = adjusted;
                                cmd.set_after(&project.canvas_state);
                                project.history.push(Box::new(cmd));
                            }
                            project.mark_dirty();
                        }
                        return;
                    }
                    DialogResult::Cancel => {
                        self.preview_job_token = self.preview_job_token.wrapping_add(1);
                        let idx = dlg.layer_idx;
                        if let Some(original) = &dlg.original_pixels
                            && let Some(project) = self.active_project_mut()
                        {
                            if let Some(layer) = project.canvas_state.layers.get_mut(idx) {
                                layer.pixels = original.clone();
                            }
                            project.canvas_state.mark_dirty(None);
                        }
                        self.active_dialog = ActiveDialog::None;
                        return;
                    }
                    _ => {
                        dlg.poll_flat();
                    }
                }
            }

            ActiveDialog::MotionBlur(dlg) => {
                match dlg.show(ctx) {
                    DialogResult::Changed => {
                        // Spawn async preview job (token-tracked, interrupts stale previews)
                        dlg.poll_flat();
                        let idx = dlg.layer_idx;
                        if let (Some(original), Some(flat)) =
                            (&dlg.original_pixels, &dlg.original_flat)
                        {
                            let (angle, distance) = (dlg.angle, dlg.distance);
                            self.spawn_preview_job(
                                ctx.input(|i| i.time),
                                "Motion Blur".to_string(),
                                idx,
                                original.clone(),
                                flat.clone(),
                                move |img| {
                                    crate::ops::effects::motion_blur_core(
                                        img, angle, distance, None,
                                    )
                                },
                            );
                        }
                    }
                    DialogResult::Ok(_) => {
                        self.preview_job_token = self.preview_job_token.wrapping_add(1);
                        let idx = dlg.layer_idx;
                        if let Some(flat) = &dlg.original_flat {
                            let (angle, distance) = (dlg.angle, dlg.distance);
                            if let Some(project) = self.active_project_mut() {
                                Self::apply_fullres_effect(
                                    &mut project.canvas_state,
                                    idx,
                                    flat,
                                    |img| {
                                        crate::ops::effects::motion_blur_core(
                                            img, angle, distance, None,
                                        )
                                    },
                                );
                            }
                        }
                        self.active_dialog = ActiveDialog::None;
                        if let Some(project) = self.active_project_mut() {
                            let idx = dlg.layer_idx;
                            if let Some(original) = &dlg.original_pixels
                                && idx < project.canvas_state.layers.len()
                            {
                                let adjusted = project.canvas_state.layers[idx].pixels.clone();
                                project.canvas_state.layers[idx].pixels = original.clone();
                                let mut cmd = SingleLayerSnapshotCommand::new_for_layer(
                                    "Motion Blur".to_string(),
                                    &project.canvas_state,
                                    idx,
                                );
                                project.canvas_state.layers[idx].pixels = adjusted;
                                cmd.set_after(&project.canvas_state);
                                project.history.push(Box::new(cmd));
                            }
                            project.mark_dirty();
                        }
                        return;
                    }
                    DialogResult::Cancel => {
                        self.preview_job_token = self.preview_job_token.wrapping_add(1);
                        let idx = dlg.layer_idx;
                        if let Some(original) = &dlg.original_pixels
                            && let Some(project) = self.active_project_mut()
                        {
                            if let Some(layer) = project.canvas_state.layers.get_mut(idx) {
                                layer.pixels = original.clone();
                            }
                            project.canvas_state.mark_dirty(None);
                        }
                        self.active_dialog = ActiveDialog::None;
                        return;
                    }
                    _ => {
                        dlg.poll_flat();
                    }
                }
            }

            ActiveDialog::BoxBlur(dlg) => {
                match dlg.show(ctx) {
                    DialogResult::Changed => {
                        // Spawn async preview job (token-tracked, interrupts stale previews)
                        dlg.poll_flat();
                        let idx = dlg.layer_idx;
                        if let (Some(original), Some(flat)) =
                            (&dlg.original_pixels, &dlg.original_flat)
                        {
                            let radius = dlg.radius;
                            self.spawn_preview_job(
                                ctx.input(|i| i.time),
                                "Box Blur".to_string(),
                                idx,
                                original.clone(),
                                flat.clone(),
                                move |img| crate::ops::effects::box_blur_core(img, radius, None),
                            );
                        }
                    }
                    DialogResult::Ok(_) => {
                        self.preview_job_token = self.preview_job_token.wrapping_add(1);
                        let idx = dlg.layer_idx;
                        if let Some(flat) = &dlg.original_flat {
                            let radius = dlg.radius;
                            if let Some(project) = self.active_project_mut() {
                                Self::apply_fullres_effect(
                                    &mut project.canvas_state,
                                    idx,
                                    flat,
                                    |img| crate::ops::effects::box_blur_core(img, radius, None),
                                );
                            }
                        }
                        self.active_dialog = ActiveDialog::None;
                        if let Some(project) = self.active_project_mut() {
                            let idx = dlg.layer_idx;
                            if let Some(original) = &dlg.original_pixels
                                && idx < project.canvas_state.layers.len()
                            {
                                let adjusted = project.canvas_state.layers[idx].pixels.clone();
                                project.canvas_state.layers[idx].pixels = original.clone();
                                let mut cmd = SingleLayerSnapshotCommand::new_for_layer(
                                    "Box Blur".to_string(),
                                    &project.canvas_state,
                                    idx,
                                );
                                project.canvas_state.layers[idx].pixels = adjusted;
                                cmd.set_after(&project.canvas_state);
                                project.history.push(Box::new(cmd));
                            }
                            project.mark_dirty();
                        }
                        return;
                    }
                    DialogResult::Cancel => {
                        self.preview_job_token = self.preview_job_token.wrapping_add(1);
                        let idx = dlg.layer_idx;
                        if let Some(original) = &dlg.original_pixels
                            && let Some(project) = self.active_project_mut()
                        {
                            if let Some(layer) = project.canvas_state.layers.get_mut(idx) {
                                layer.pixels = original.clone();
                            }
                            project.canvas_state.mark_dirty(None);
                        }
                        self.active_dialog = ActiveDialog::None;
                        return;
                    }
                    _ => {
                        dlg.poll_flat();
                    }
                }
            }

            ActiveDialog::ZoomBlur(dlg) => match dlg.show(ctx) {
                DialogResult::Changed => {
                    dlg.poll_flat();
                    let idx = dlg.layer_idx;
                    if let (Some(original), Some(flat)) = (&dlg.original_pixels, &dlg.original_flat)
                    {
                        let (cx, cy, strength, samples, tint, ts) = dlg.current_params();
                        self.spawn_preview_job(
                            ctx.input(|i| i.time),
                            "Zoom Blur".to_string(),
                            idx,
                            original.clone(),
                            flat.clone(),
                            move |img| {
                                crate::ops::effects::zoom_blur_core(
                                    img, cx, cy, strength, samples, tint, ts, None,
                                )
                            },
                        );
                    }
                }
                DialogResult::Ok(_) => {
                    self.preview_job_token = self.preview_job_token.wrapping_add(1);
                    let idx = dlg.layer_idx;
                    if let Some(flat) = &dlg.original_flat {
                        let (cx, cy, strength, samples, tint, ts) = dlg.current_params();
                        if let Some(project) = self.active_project_mut() {
                            Self::apply_fullres_effect(
                                &mut project.canvas_state,
                                idx,
                                flat,
                                |img| {
                                    crate::ops::effects::zoom_blur_core(
                                        img, cx, cy, strength, samples, tint, ts, None,
                                    )
                                },
                            );
                        }
                    }
                    self.active_dialog = ActiveDialog::None;
                    if let Some(project) = self.active_project_mut() {
                        let idx = dlg.layer_idx;
                        if let Some(original) = &dlg.original_pixels
                            && idx < project.canvas_state.layers.len()
                        {
                            let adjusted = project.canvas_state.layers[idx].pixels.clone();
                            project.canvas_state.layers[idx].pixels = original.clone();
                            let mut cmd = SingleLayerSnapshotCommand::new_for_layer(
                                "Zoom Blur".to_string(),
                                &project.canvas_state,
                                idx,
                            );
                            project.canvas_state.layers[idx].pixels = adjusted;
                            cmd.set_after(&project.canvas_state);
                            project.history.push(Box::new(cmd));
                        }
                        project.mark_dirty();
                    }
                    return;
                }
                DialogResult::Cancel => {
                    self.preview_job_token = self.preview_job_token.wrapping_add(1);
                    let idx = dlg.layer_idx;
                    if let Some(original) = &dlg.original_pixels
                        && let Some(project) = self.active_project_mut()
                    {
                        if let Some(layer) = project.canvas_state.layers.get_mut(idx) {
                            layer.pixels = original.clone();
                        }
                        project.canvas_state.mark_dirty(None);
                    }
                    self.active_dialog = ActiveDialog::None;
                    return;
                }
                _ => {
                    dlg.poll_flat();
                }
            },

            ActiveDialog::Crystallize(dlg) => match dlg.show(ctx) {
                DialogResult::Changed => {
                    dlg.first_open = false;
                    let idx = dlg.layer_idx;
                    if let (Some(original), Some(flat)) = (&dlg.original_pixels, &dlg.original_flat)
                    {
                        let (cell_size, seed) = (dlg.cell_size, dlg.seed);
                        self.spawn_preview_job(
                            ctx.input(|i| i.time),
                            "Crystallize".to_string(),
                            idx,
                            original.clone(),
                            flat.clone(),
                            move |img| {
                                crate::ops::effects::crystallize_core(img, cell_size, seed, None)
                            },
                        );
                    }
                }
                DialogResult::Ok(_) => {
                    self.preview_job_token = self.preview_job_token.wrapping_add(1);
                    let idx = dlg.layer_idx;
                    if let Some(flat) = &dlg.original_flat {
                        let (cell_size, seed) = (dlg.cell_size, dlg.seed);
                        if let Some(project) = self.active_project_mut() {
                            Self::apply_fullres_effect(
                                &mut project.canvas_state,
                                idx,
                                flat,
                                |img| {
                                    crate::ops::effects::crystallize_core(
                                        img, cell_size, seed, None,
                                    )
                                },
                            );
                        }
                    }
                    self.active_dialog = ActiveDialog::None;
                    if let Some(project) = self.active_project_mut() {
                        let idx = dlg.layer_idx;
                        if let Some(original) = &dlg.original_pixels
                            && idx < project.canvas_state.layers.len()
                        {
                            let adjusted = project.canvas_state.layers[idx].pixels.clone();
                            project.canvas_state.layers[idx].pixels = original.clone();
                            let mut cmd = SingleLayerSnapshotCommand::new_for_layer(
                                "Crystallize".to_string(),
                                &project.canvas_state,
                                idx,
                            );
                            project.canvas_state.layers[idx].pixels = adjusted;
                            cmd.set_after(&project.canvas_state);
                            project.history.push(Box::new(cmd));
                        }
                        project.mark_dirty();
                    }
                    return;
                }
                DialogResult::Cancel => {
                    self.preview_job_token = self.preview_job_token.wrapping_add(1);
                    let idx = dlg.layer_idx;
                    if let Some(original) = &dlg.original_pixels
                        && let Some(project) = self.active_project_mut()
                    {
                        if let Some(layer) = project.canvas_state.layers.get_mut(idx) {
                            layer.pixels = original.clone();
                        }
                        project.canvas_state.mark_dirty(None);
                    }
                    self.active_dialog = ActiveDialog::None;
                    return;
                }
                _ => {
                    dlg.poll_flat();
                    if dlg.first_open && dlg.live_preview && dlg.original_flat.is_some() {
                        dlg.first_open = false;
                        let idx = dlg.layer_idx;
                        if let (Some(original), Some(flat)) =
                            (&dlg.original_pixels, &dlg.original_flat)
                        {
                            let (cell_size, seed) = (dlg.cell_size, dlg.seed);
                            self.spawn_preview_job(
                                ctx.input(|i| i.time),
                                "Crystallize".to_string(),
                                idx,
                                original.clone(),
                                flat.clone(),
                                move |img| {
                                    crate::ops::effects::crystallize_core(
                                        img, cell_size, seed, None,
                                    )
                                },
                            );
                        }
                    }
                }
            },

            ActiveDialog::Dents(dlg) => match dlg.show(ctx) {
                DialogResult::Changed => {
                    dlg.first_open = false;
                    let idx = dlg.layer_idx;
                    if let (Some(original), Some(flat)) = (&dlg.original_pixels, &dlg.original_flat)
                    {
                        let (scale_p, amount, seed) = (dlg.scale, dlg.amount, dlg.seed);
                        let (octaves, roughness) = (dlg.octaves as u32, dlg.roughness);
                        let (pinch, wrap) = (dlg.pinch, dlg.wrap);
                        self.spawn_preview_job(
                            ctx.input(|i| i.time),
                            "Dents".to_string(),
                            idx,
                            original.clone(),
                            flat.clone(),
                            move |img| {
                                crate::ops::effects::dents_core(
                                    img, scale_p, amount, seed, octaves, roughness, pinch, wrap,
                                    None,
                                )
                            },
                        );
                    }
                }
                DialogResult::Ok(_) => {
                    self.preview_job_token = self.preview_job_token.wrapping_add(1);
                    let idx = dlg.layer_idx;
                    if let Some(flat) = &dlg.original_flat {
                        let (scale_p, amount, seed) = (dlg.scale, dlg.amount, dlg.seed);
                        let (octaves, roughness) = (dlg.octaves as u32, dlg.roughness);
                        let (pinch, wrap) = (dlg.pinch, dlg.wrap);
                        if let Some(project) = self.active_project_mut() {
                            Self::apply_fullres_effect(
                                &mut project.canvas_state,
                                idx,
                                flat,
                                |img| {
                                    crate::ops::effects::dents_core(
                                        img, scale_p, amount, seed, octaves, roughness, pinch,
                                        wrap, None,
                                    )
                                },
                            );
                        }
                    }
                    self.active_dialog = ActiveDialog::None;
                    if let Some(project) = self.active_project_mut() {
                        let idx = dlg.layer_idx;
                        if let Some(original) = &dlg.original_pixels
                            && idx < project.canvas_state.layers.len()
                        {
                            let adjusted = project.canvas_state.layers[idx].pixels.clone();
                            project.canvas_state.layers[idx].pixels = original.clone();
                            let mut cmd = SingleLayerSnapshotCommand::new_for_layer(
                                "Dents".to_string(),
                                &project.canvas_state,
                                idx,
                            );
                            project.canvas_state.layers[idx].pixels = adjusted;
                            cmd.set_after(&project.canvas_state);
                            project.history.push(Box::new(cmd));
                        }
                        project.mark_dirty();
                    }
                    return;
                }
                DialogResult::Cancel => {
                    self.preview_job_token = self.preview_job_token.wrapping_add(1);
                    let idx = dlg.layer_idx;
                    if let Some(original) = &dlg.original_pixels
                        && let Some(project) = self.active_project_mut()
                    {
                        if let Some(layer) = project.canvas_state.layers.get_mut(idx) {
                            layer.pixels = original.clone();
                        }
                        project.canvas_state.mark_dirty(None);
                    }
                    self.active_dialog = ActiveDialog::None;
                    return;
                }
                _ => {
                    dlg.poll_flat();
                    if dlg.first_open && dlg.live_preview && dlg.original_flat.is_some() {
                        dlg.first_open = false;
                        let idx = dlg.layer_idx;
                        if let (Some(original), Some(flat)) =
                            (&dlg.original_pixels, &dlg.original_flat)
                        {
                            let (scale_p, amount, seed) = (dlg.scale, dlg.amount, dlg.seed);
                            let (octaves, roughness) = (dlg.octaves as u32, dlg.roughness);
                            let (pinch, wrap) = (dlg.pinch, dlg.wrap);
                            self.spawn_preview_job(
                                ctx.input(|i| i.time),
                                "Dents".to_string(),
                                idx,
                                original.clone(),
                                flat.clone(),
                                move |img| {
                                    crate::ops::effects::dents_core(
                                        img, scale_p, amount, seed, octaves, roughness, pinch,
                                        wrap, None,
                                    )
                                },
                            );
                        }
                    }
                }
            },

            ActiveDialog::Pixelate(dlg) => match dlg.show(ctx) {
                DialogResult::Changed => {
                    dlg.first_open = false;
                    let idx = dlg.layer_idx;
                    if let (Some(original), Some(flat)) = (&dlg.original_pixels, &dlg.original_flat)
                    {
                        let block_size = dlg.block_size as u32;
                        self.spawn_preview_job(
                            ctx.input(|i| i.time),
                            "Pixelate".to_string(),
                            idx,
                            original.clone(),
                            flat.clone(),
                            move |img| crate::ops::effects::pixelate_core(img, block_size, None),
                        );
                    }
                }
                DialogResult::Ok(_) => {
                    self.preview_job_token = self.preview_job_token.wrapping_add(1);
                    let idx = dlg.layer_idx;
                    if let Some(flat) = &dlg.original_flat {
                        let block_size = dlg.block_size as u32;
                        if let Some(project) = self.active_project_mut() {
                            Self::apply_fullres_effect(
                                &mut project.canvas_state,
                                idx,
                                flat,
                                |img| crate::ops::effects::pixelate_core(img, block_size, None),
                            );
                        }
                    }
                    self.active_dialog = ActiveDialog::None;
                    if let Some(project) = self.active_project_mut() {
                        let idx = dlg.layer_idx;
                        if let Some(original) = &dlg.original_pixels
                            && idx < project.canvas_state.layers.len()
                        {
                            let adjusted = project.canvas_state.layers[idx].pixels.clone();
                            project.canvas_state.layers[idx].pixels = original.clone();
                            let mut cmd = SingleLayerSnapshotCommand::new_for_layer(
                                "Pixelate".to_string(),
                                &project.canvas_state,
                                idx,
                            );
                            project.canvas_state.layers[idx].pixels = adjusted;
                            cmd.set_after(&project.canvas_state);
                            project.history.push(Box::new(cmd));
                        }
                        project.mark_dirty();
                    }
                    return;
                }
                DialogResult::Cancel => {
                    self.preview_job_token = self.preview_job_token.wrapping_add(1);
                    let idx = dlg.layer_idx;
                    if let Some(original) = &dlg.original_pixels
                        && let Some(project) = self.active_project_mut()
                    {
                        if let Some(layer) = project.canvas_state.layers.get_mut(idx) {
                            layer.pixels = original.clone();
                        }
                        project.canvas_state.mark_dirty(None);
                    }
                    self.active_dialog = ActiveDialog::None;
                    return;
                }
                _ => {
                    dlg.poll_flat();
                    if dlg.first_open && dlg.live_preview && dlg.original_flat.is_some() {
                        dlg.first_open = false;
                        let idx = dlg.layer_idx;
                        if let (Some(original), Some(flat)) =
                            (&dlg.original_pixels, &dlg.original_flat)
                        {
                            let block_size = dlg.block_size as u32;
                            self.spawn_preview_job(
                                ctx.input(|i| i.time),
                                "Pixelate".to_string(),
                                idx,
                                original.clone(),
                                flat.clone(),
                                move |img| {
                                    crate::ops::effects::pixelate_core(img, block_size, None)
                                },
                            );
                        }
                    }
                }
            },

            ActiveDialog::Bulge(dlg) => match dlg.show(ctx) {
                DialogResult::Changed => {
                    dlg.first_open = false;
                    let idx = dlg.layer_idx;
                    if let (Some(original), Some(flat)) = (&dlg.original_pixels, &dlg.original_flat)
                    {
                        let amount = dlg.amount;
                        self.spawn_preview_job(
                            ctx.input(|i| i.time),
                            "Bulge".to_string(),
                            idx,
                            original.clone(),
                            flat.clone(),
                            move |img| crate::ops::effects::bulge_core(img, amount, None),
                        );
                    }
                }
                DialogResult::Ok(_) => {
                    self.preview_job_token = self.preview_job_token.wrapping_add(1);
                    let idx = dlg.layer_idx;
                    if let Some(flat) = &dlg.original_flat {
                        let amount = dlg.amount;
                        if let Some(project) = self.active_project_mut() {
                            Self::apply_fullres_effect(
                                &mut project.canvas_state,
                                idx,
                                flat,
                                |img| crate::ops::effects::bulge_core(img, amount, None),
                            );
                        }
                    }
                    self.active_dialog = ActiveDialog::None;
                    if let Some(project) = self.active_project_mut() {
                        let idx = dlg.layer_idx;
                        if let Some(original) = &dlg.original_pixels
                            && idx < project.canvas_state.layers.len()
                        {
                            let adjusted = project.canvas_state.layers[idx].pixels.clone();
                            project.canvas_state.layers[idx].pixels = original.clone();
                            let mut cmd = SingleLayerSnapshotCommand::new_for_layer(
                                "Bulge".to_string(),
                                &project.canvas_state,
                                idx,
                            );
                            project.canvas_state.layers[idx].pixels = adjusted;
                            cmd.set_after(&project.canvas_state);
                            project.history.push(Box::new(cmd));
                        }
                        project.mark_dirty();
                    }
                    return;
                }
                DialogResult::Cancel => {
                    self.preview_job_token = self.preview_job_token.wrapping_add(1);
                    let idx = dlg.layer_idx;
                    if let Some(original) = &dlg.original_pixels
                        && let Some(project) = self.active_project_mut()
                    {
                        if let Some(layer) = project.canvas_state.layers.get_mut(idx) {
                            layer.pixels = original.clone();
                        }
                        project.canvas_state.mark_dirty(None);
                    }
                    self.active_dialog = ActiveDialog::None;
                    return;
                }
                _ => {
                    dlg.poll_flat();
                    if dlg.first_open && dlg.live_preview && dlg.original_flat.is_some() {
                        dlg.first_open = false;
                        let idx = dlg.layer_idx;
                        if let (Some(original), Some(flat)) =
                            (&dlg.original_pixels, &dlg.original_flat)
                        {
                            let amount = dlg.amount;
                            self.spawn_preview_job(
                                ctx.input(|i| i.time),
                                "Bulge".to_string(),
                                idx,
                                original.clone(),
                                flat.clone(),
                                move |img| crate::ops::effects::bulge_core(img, amount, None),
                            );
                        }
                    }
                }
            },

            ActiveDialog::Twist(dlg) => match dlg.show(ctx) {
                DialogResult::Changed => {
                    dlg.first_open = false;
                    let idx = dlg.layer_idx;
                    if let (Some(original), Some(flat)) = (&dlg.original_pixels, &dlg.original_flat)
                    {
                        let angle = dlg.angle;
                        self.spawn_preview_job(
                            ctx.input(|i| i.time),
                            "Twist".to_string(),
                            idx,
                            original.clone(),
                            flat.clone(),
                            move |img| crate::ops::effects::twist_core(img, angle, None),
                        );
                    }
                }
                DialogResult::Ok(_) => {
                    self.preview_job_token = self.preview_job_token.wrapping_add(1);
                    let idx = dlg.layer_idx;
                    if let Some(flat) = &dlg.original_flat {
                        let angle = dlg.angle;
                        if let Some(project) = self.active_project_mut() {
                            Self::apply_fullres_effect(
                                &mut project.canvas_state,
                                idx,
                                flat,
                                |img| crate::ops::effects::twist_core(img, angle, None),
                            );
                        }
                    }
                    self.active_dialog = ActiveDialog::None;
                    if let Some(project) = self.active_project_mut() {
                        let idx = dlg.layer_idx;
                        if let Some(original) = &dlg.original_pixels
                            && idx < project.canvas_state.layers.len()
                        {
                            let adjusted = project.canvas_state.layers[idx].pixels.clone();
                            project.canvas_state.layers[idx].pixels = original.clone();
                            let mut cmd = SingleLayerSnapshotCommand::new_for_layer(
                                "Twist".to_string(),
                                &project.canvas_state,
                                idx,
                            );
                            project.canvas_state.layers[idx].pixels = adjusted;
                            cmd.set_after(&project.canvas_state);
                            project.history.push(Box::new(cmd));
                        }
                        project.mark_dirty();
                    }
                    return;
                }
                DialogResult::Cancel => {
                    self.preview_job_token = self.preview_job_token.wrapping_add(1);
                    let idx = dlg.layer_idx;
                    if let Some(original) = &dlg.original_pixels
                        && let Some(project) = self.active_project_mut()
                    {
                        if let Some(layer) = project.canvas_state.layers.get_mut(idx) {
                            layer.pixels = original.clone();
                        }
                        project.canvas_state.mark_dirty(None);
                    }
                    self.active_dialog = ActiveDialog::None;
                    return;
                }
                _ => {
                    dlg.poll_flat();
                    if dlg.first_open && dlg.live_preview && dlg.original_flat.is_some() {
                        dlg.first_open = false;
                        let idx = dlg.layer_idx;
                        if let (Some(original), Some(flat)) =
                            (&dlg.original_pixels, &dlg.original_flat)
                        {
                            let angle = dlg.angle;
                            self.spawn_preview_job(
                                ctx.input(|i| i.time),
                                "Twist".to_string(),
                                idx,
                                original.clone(),
                                flat.clone(),
                                move |img| crate::ops::effects::twist_core(img, angle, None),
                            );
                        }
                    }
                }
            },

            ActiveDialog::AddNoise(dlg) => match dlg.show(ctx) {
                DialogResult::Changed => {
                    dlg.first_open = false;
                    let idx = dlg.layer_idx;
                    if let (Some(original), Some(flat)) = (&dlg.original_pixels, &dlg.original_flat)
                    {
                        let amount = dlg.amount;
                        let noise_type = dlg.noise_type();
                        let monochrome = dlg.monochrome;
                        let seed = dlg.seed;
                        let noise_scale = dlg.scale;
                        let octaves = dlg.octaves as u32;
                        self.spawn_preview_job(
                            ctx.input(|i| i.time),
                            "Add Noise".to_string(),
                            idx,
                            original.clone(),
                            flat.clone(),
                            move |img| {
                                crate::ops::effects::add_noise_core(
                                    img,
                                    amount,
                                    noise_type,
                                    monochrome,
                                    seed,
                                    noise_scale,
                                    octaves,
                                    None,
                                )
                            },
                        );
                    }
                }
                DialogResult::Ok(_) => {
                    self.preview_job_token = self.preview_job_token.wrapping_add(1);
                    let idx = dlg.layer_idx;
                    if let Some(flat) = &dlg.original_flat {
                        let amount = dlg.amount;
                        let noise_type = dlg.noise_type();
                        let monochrome = dlg.monochrome;
                        let seed = dlg.seed;
                        let noise_scale = dlg.scale;
                        let octaves = dlg.octaves as u32;
                        if let Some(project) = self.active_project_mut() {
                            Self::apply_fullres_effect(
                                &mut project.canvas_state,
                                idx,
                                flat,
                                |img| {
                                    crate::ops::effects::add_noise_core(
                                        img,
                                        amount,
                                        noise_type,
                                        monochrome,
                                        seed,
                                        noise_scale,
                                        octaves,
                                        None,
                                    )
                                },
                            );
                        }
                    }
                    self.active_dialog = ActiveDialog::None;
                    if let Some(project) = self.active_project_mut() {
                        let idx = dlg.layer_idx;
                        if let Some(original) = &dlg.original_pixels
                            && idx < project.canvas_state.layers.len()
                        {
                            let adjusted = project.canvas_state.layers[idx].pixels.clone();
                            project.canvas_state.layers[idx].pixels = original.clone();
                            let mut cmd = SingleLayerSnapshotCommand::new_for_layer(
                                "Add Noise".to_string(),
                                &project.canvas_state,
                                idx,
                            );
                            project.canvas_state.layers[idx].pixels = adjusted;
                            cmd.set_after(&project.canvas_state);
                            project.history.push(Box::new(cmd));
                        }
                        project.mark_dirty();
                    }
                    return;
                }
                DialogResult::Cancel => {
                    self.preview_job_token = self.preview_job_token.wrapping_add(1);
                    let idx = dlg.layer_idx;
                    if let Some(original) = &dlg.original_pixels
                        && let Some(project) = self.active_project_mut()
                    {
                        if let Some(layer) = project.canvas_state.layers.get_mut(idx) {
                            layer.pixels = original.clone();
                        }
                        project.canvas_state.mark_dirty(None);
                    }
                    self.active_dialog = ActiveDialog::None;
                    return;
                }
                _ => {
                    dlg.poll_flat();
                    if dlg.first_open && dlg.live_preview && dlg.original_flat.is_some() {
                        dlg.first_open = false;
                        let idx = dlg.layer_idx;
                        if let (Some(original), Some(flat)) =
                            (&dlg.original_pixels, &dlg.original_flat)
                        {
                            let amount = dlg.amount;
                            let noise_type = dlg.noise_type();
                            let monochrome = dlg.monochrome;
                            let seed = dlg.seed;
                            let noise_scale = dlg.scale;
                            let octaves = dlg.octaves as u32;
                            self.spawn_preview_job(
                                ctx.input(|i| i.time),
                                "Add Noise".to_string(),
                                idx,
                                original.clone(),
                                flat.clone(),
                                move |img| {
                                    crate::ops::effects::add_noise_core(
                                        img,
                                        amount,
                                        noise_type,
                                        monochrome,
                                        seed,
                                        noise_scale,
                                        octaves,
                                        None,
                                    )
                                },
                            );
                        }
                    }
                }
            },

            ActiveDialog::ReduceNoise(dlg) => match dlg.show(ctx) {
                DialogResult::Changed => {
                    dlg.first_open = false;
                    let idx = dlg.layer_idx;
                    if let (Some(original), Some(flat)) = (&dlg.original_pixels, &dlg.original_flat)
                    {
                        let (strength, radius) = (dlg.strength, dlg.radius as u32);
                        self.spawn_preview_job(
                            ctx.input(|i| i.time),
                            "Reduce Noise".to_string(),
                            idx,
                            original.clone(),
                            flat.clone(),
                            move |img| {
                                crate::ops::effects::reduce_noise_core(img, strength, radius, None)
                            },
                        );
                    }
                }
                DialogResult::Ok(_) => {
                    self.preview_job_token = self.preview_job_token.wrapping_add(1);
                    let idx = dlg.layer_idx;
                    if let Some(flat) = &dlg.original_flat {
                        let strength = dlg.strength;
                        let radius = dlg.radius as u32;
                        if let Some(project) = self.active_project_mut() {
                            Self::apply_fullres_effect(
                                &mut project.canvas_state,
                                idx,
                                flat,
                                |img| {
                                    crate::ops::effects::reduce_noise_core(
                                        img, strength, radius, None,
                                    )
                                },
                            );
                        }
                    }
                    self.active_dialog = ActiveDialog::None;
                    if let Some(project) = self.active_project_mut() {
                        let idx = dlg.layer_idx;
                        if let Some(original) = &dlg.original_pixels
                            && idx < project.canvas_state.layers.len()
                        {
                            let adjusted = project.canvas_state.layers[idx].pixels.clone();
                            project.canvas_state.layers[idx].pixels = original.clone();
                            let mut cmd = SingleLayerSnapshotCommand::new_for_layer(
                                "Reduce Noise".to_string(),
                                &project.canvas_state,
                                idx,
                            );
                            project.canvas_state.layers[idx].pixels = adjusted;
                            cmd.set_after(&project.canvas_state);
                            project.history.push(Box::new(cmd));
                        }
                        project.mark_dirty();
                    }
                    return;
                }
                DialogResult::Cancel => {
                    self.preview_job_token = self.preview_job_token.wrapping_add(1);
                    let idx = dlg.layer_idx;
                    if let Some(original) = &dlg.original_pixels
                        && let Some(project) = self.active_project_mut()
                    {
                        if let Some(layer) = project.canvas_state.layers.get_mut(idx) {
                            layer.pixels = original.clone();
                        }
                        project.canvas_state.mark_dirty(None);
                    }
                    self.active_dialog = ActiveDialog::None;
                    return;
                }
                _ => {
                    dlg.poll_flat();
                    if dlg.first_open && dlg.live_preview && dlg.original_flat.is_some() {
                        dlg.first_open = false;
                        let idx = dlg.layer_idx;
                        if let (Some(original), Some(flat)) =
                            (&dlg.original_pixels, &dlg.original_flat)
                        {
                            let (strength, radius) = (dlg.strength, dlg.radius as u32);
                            self.spawn_preview_job(
                                ctx.input(|i| i.time),
                                "Reduce Noise".to_string(),
                                idx,
                                original.clone(),
                                flat.clone(),
                                move |img| {
                                    crate::ops::effects::reduce_noise_core(
                                        img, strength, radius, None,
                                    )
                                },
                            );
                        }
                    }
                }
            },

            ActiveDialog::Median(dlg) => {
                match dlg.show(ctx) {
                    DialogResult::Changed => {
                        // Spawn async preview job (token-tracked, interrupts stale previews)
                        let idx = dlg.layer_idx;
                        if let (Some(original), Some(flat)) =
                            (&dlg.original_pixels, &dlg.original_flat)
                        {
                            let radius = dlg.radius as u32;
                            self.spawn_preview_job(
                                ctx.input(|i| i.time),
                                "Median Filter".to_string(),
                                idx,
                                original.clone(),
                                flat.clone(),
                                move |img| crate::ops::effects::median_core(img, radius, None),
                            );
                        }
                    }
                    DialogResult::Ok(_) => {
                        self.preview_job_token = self.preview_job_token.wrapping_add(1);
                        // Use GPU-accelerated median at full resolution
                        let idx = dlg.layer_idx;
                        let radius = dlg.radius as u32;
                        if let Some(flat) = &dlg.original_flat {
                            let active_idx = self.active_project_index;
                            if let Some(project) = self.projects.get_mut(active_idx) {
                                crate::ops::effects::median_filter_from_flat_gpu(
                                    &mut project.canvas_state,
                                    idx,
                                    radius,
                                    flat,
                                    &self.canvas.gpu_renderer,
                                );
                            }
                        }
                        self.active_dialog = ActiveDialog::None;
                        if let Some(project) = self.active_project_mut() {
                            let idx = dlg.layer_idx;
                            if let Some(original) = &dlg.original_pixels
                                && idx < project.canvas_state.layers.len()
                            {
                                let adjusted = project.canvas_state.layers[idx].pixels.clone();
                                project.canvas_state.layers[idx].pixels = original.clone();
                                let mut cmd = SingleLayerSnapshotCommand::new_for_layer(
                                    "Median Filter".to_string(),
                                    &project.canvas_state,
                                    idx,
                                );
                                project.canvas_state.layers[idx].pixels = adjusted;
                                cmd.set_after(&project.canvas_state);
                                project.history.push(Box::new(cmd));
                            }
                            project.mark_dirty();
                        }
                        return;
                    }
                    DialogResult::Cancel => {
                        self.preview_job_token = self.preview_job_token.wrapping_add(1);
                        let idx = dlg.layer_idx;
                        if let Some(original) = &dlg.original_pixels
                            && let Some(project) = self.active_project_mut()
                        {
                            if let Some(layer) = project.canvas_state.layers.get_mut(idx) {
                                layer.pixels = original.clone();
                            }
                            project.canvas_state.mark_dirty(None);
                        }
                        self.active_dialog = ActiveDialog::None;
                        return;
                    }
                    _ => {}
                }
            }

            ActiveDialog::Glow(dlg) => match dlg.show(ctx) {
                DialogResult::Changed => {
                    dlg.first_open = false;
                    let idx = dlg.layer_idx;
                    if let (Some(original), Some(flat)) = (&dlg.original_pixels, &dlg.original_flat)
                    {
                        let (radius, intensity) = (dlg.radius, dlg.intensity);
                        self.spawn_preview_job(
                            ctx.input(|i| i.time),
                            "Glow".to_string(),
                            idx,
                            original.clone(),
                            flat.clone(),
                            move |img| crate::ops::effects::glow_core(img, radius, intensity, None),
                        );
                    }
                }
                DialogResult::Ok(_) => {
                    self.preview_job_token = self.preview_job_token.wrapping_add(1);
                    let idx = dlg.layer_idx;
                    if let Some(flat) = &dlg.original_flat {
                        let (radius, intensity) = (dlg.radius, dlg.intensity);
                        if let Some(project) = self.active_project_mut() {
                            Self::apply_fullres_effect(
                                &mut project.canvas_state,
                                idx,
                                flat,
                                |img| crate::ops::effects::glow_core(img, radius, intensity, None),
                            );
                        }
                    }
                    self.active_dialog = ActiveDialog::None;
                    if let Some(project) = self.active_project_mut() {
                        let idx = dlg.layer_idx;
                        if let Some(original) = &dlg.original_pixels
                            && idx < project.canvas_state.layers.len()
                        {
                            let adjusted = project.canvas_state.layers[idx].pixels.clone();
                            project.canvas_state.layers[idx].pixels = original.clone();
                            let mut cmd = SingleLayerSnapshotCommand::new_for_layer(
                                "Glow".to_string(),
                                &project.canvas_state,
                                idx,
                            );
                            project.canvas_state.layers[idx].pixels = adjusted;
                            cmd.set_after(&project.canvas_state);
                            project.history.push(Box::new(cmd));
                        }
                        project.mark_dirty();
                    }
                    return;
                }
                DialogResult::Cancel => {
                    self.preview_job_token = self.preview_job_token.wrapping_add(1);
                    let idx = dlg.layer_idx;
                    if let Some(original) = &dlg.original_pixels
                        && let Some(project) = self.active_project_mut()
                    {
                        if let Some(layer) = project.canvas_state.layers.get_mut(idx) {
                            layer.pixels = original.clone();
                        }
                        project.canvas_state.mark_dirty(None);
                    }
                    self.active_dialog = ActiveDialog::None;
                    return;
                }
                _ => {
                    dlg.poll_flat();
                    if dlg.first_open && dlg.live_preview && dlg.original_flat.is_some() {
                        dlg.first_open = false;
                        let idx = dlg.layer_idx;
                        if let (Some(original), Some(flat)) =
                            (&dlg.original_pixels, &dlg.original_flat)
                        {
                            let (radius, intensity) = (dlg.radius, dlg.intensity);
                            self.spawn_preview_job(
                                ctx.input(|i| i.time),
                                "Glow".to_string(),
                                idx,
                                original.clone(),
                                flat.clone(),
                                move |img| {
                                    crate::ops::effects::glow_core(img, radius, intensity, None)
                                },
                            );
                        }
                    }
                }
            },

            ActiveDialog::Sharpen(dlg) => match dlg.show(ctx) {
                DialogResult::Changed => {
                    dlg.first_open = false;
                    let idx = dlg.layer_idx;
                    if let (Some(original), Some(flat)) = (&dlg.original_pixels, &dlg.original_flat)
                    {
                        let (amount, radius) = (dlg.amount, dlg.radius);
                        self.spawn_preview_job(
                            ctx.input(|i| i.time),
                            "Sharpen".to_string(),
                            idx,
                            original.clone(),
                            flat.clone(),
                            move |img| crate::ops::effects::sharpen_core(img, amount, radius, None),
                        );
                    }
                }
                DialogResult::Ok(_) => {
                    self.preview_job_token = self.preview_job_token.wrapping_add(1);
                    let idx = dlg.layer_idx;
                    if let Some(flat) = &dlg.original_flat {
                        let (amount, radius) = (dlg.amount, dlg.radius);
                        if let Some(project) = self.active_project_mut() {
                            Self::apply_fullres_effect(
                                &mut project.canvas_state,
                                idx,
                                flat,
                                |img| crate::ops::effects::sharpen_core(img, amount, radius, None),
                            );
                        }
                    }
                    self.active_dialog = ActiveDialog::None;
                    if let Some(project) = self.active_project_mut() {
                        let idx = dlg.layer_idx;
                        if let Some(original) = &dlg.original_pixels
                            && idx < project.canvas_state.layers.len()
                        {
                            let adjusted = project.canvas_state.layers[idx].pixels.clone();
                            project.canvas_state.layers[idx].pixels = original.clone();
                            let mut cmd = SingleLayerSnapshotCommand::new_for_layer(
                                "Sharpen".to_string(),
                                &project.canvas_state,
                                idx,
                            );
                            project.canvas_state.layers[idx].pixels = adjusted;
                            cmd.set_after(&project.canvas_state);
                            project.history.push(Box::new(cmd));
                        }
                        project.mark_dirty();
                    }
                    return;
                }
                DialogResult::Cancel => {
                    self.preview_job_token = self.preview_job_token.wrapping_add(1);
                    let idx = dlg.layer_idx;
                    if let Some(original) = &dlg.original_pixels
                        && let Some(project) = self.active_project_mut()
                    {
                        if let Some(layer) = project.canvas_state.layers.get_mut(idx) {
                            layer.pixels = original.clone();
                        }
                        project.canvas_state.mark_dirty(None);
                    }
                    self.active_dialog = ActiveDialog::None;
                    return;
                }
                _ => {
                    dlg.poll_flat();
                    if dlg.first_open && dlg.live_preview && dlg.original_flat.is_some() {
                        dlg.first_open = false;
                        let idx = dlg.layer_idx;
                        if let (Some(original), Some(flat)) =
                            (&dlg.original_pixels, &dlg.original_flat)
                        {
                            let (amount, radius) = (dlg.amount, dlg.radius);
                            self.spawn_preview_job(
                                ctx.input(|i| i.time),
                                "Sharpen".to_string(),
                                idx,
                                original.clone(),
                                flat.clone(),
                                move |img| {
                                    crate::ops::effects::sharpen_core(img, amount, radius, None)
                                },
                            );
                        }
                    }
                }
            },

            ActiveDialog::Vignette(dlg) => match dlg.show(ctx) {
                DialogResult::Changed => {
                    dlg.first_open = false;
                    let idx = dlg.layer_idx;
                    if let (Some(original), Some(flat)) = (&dlg.original_pixels, &dlg.original_flat)
                    {
                        let (amount, softness) = (dlg.amount, dlg.softness);
                        self.spawn_preview_job(
                            ctx.input(|i| i.time),
                            "Vignette".to_string(),
                            idx,
                            original.clone(),
                            flat.clone(),
                            move |img| {
                                crate::ops::effects::vignette_core(img, amount, softness, None)
                            },
                        );
                    }
                }
                DialogResult::Ok(_) => {
                    self.preview_job_token = self.preview_job_token.wrapping_add(1);
                    let idx = dlg.layer_idx;
                    if let Some(flat) = &dlg.original_flat {
                        let (amount, softness) = (dlg.amount, dlg.softness);
                        if let Some(project) = self.active_project_mut() {
                            Self::apply_fullres_effect(
                                &mut project.canvas_state,
                                idx,
                                flat,
                                |img| {
                                    crate::ops::effects::vignette_core(img, amount, softness, None)
                                },
                            );
                        }
                    }
                    self.active_dialog = ActiveDialog::None;
                    if let Some(project) = self.active_project_mut() {
                        let idx = dlg.layer_idx;
                        if let Some(original) = &dlg.original_pixels
                            && idx < project.canvas_state.layers.len()
                        {
                            let adjusted = project.canvas_state.layers[idx].pixels.clone();
                            project.canvas_state.layers[idx].pixels = original.clone();
                            let mut cmd = SingleLayerSnapshotCommand::new_for_layer(
                                "Vignette".to_string(),
                                &project.canvas_state,
                                idx,
                            );
                            project.canvas_state.layers[idx].pixels = adjusted;
                            cmd.set_after(&project.canvas_state);
                            project.history.push(Box::new(cmd));
                        }
                        project.mark_dirty();
                    }
                    return;
                }
                DialogResult::Cancel => {
                    self.preview_job_token = self.preview_job_token.wrapping_add(1);
                    let idx = dlg.layer_idx;
                    if let Some(original) = &dlg.original_pixels
                        && let Some(project) = self.active_project_mut()
                    {
                        if let Some(layer) = project.canvas_state.layers.get_mut(idx) {
                            layer.pixels = original.clone();
                        }
                        project.canvas_state.mark_dirty(None);
                    }
                    self.active_dialog = ActiveDialog::None;
                    return;
                }
                _ => {
                    dlg.poll_flat();
                    if dlg.first_open && dlg.live_preview && dlg.original_flat.is_some() {
                        dlg.first_open = false;
                        let idx = dlg.layer_idx;
                        if let (Some(original), Some(flat)) =
                            (&dlg.original_pixels, &dlg.original_flat)
                        {
                            let (amount, softness) = (dlg.amount, dlg.softness);
                            self.spawn_preview_job(
                                ctx.input(|i| i.time),
                                "Vignette".to_string(),
                                idx,
                                original.clone(),
                                flat.clone(),
                                move |img| {
                                    crate::ops::effects::vignette_core(img, amount, softness, None)
                                },
                            );
                        }
                    }
                }
            },

            ActiveDialog::Halftone(dlg) => match dlg.show(ctx) {
                DialogResult::Changed => {
                    dlg.first_open = false;
                    let idx = dlg.layer_idx;
                    if let (Some(original), Some(flat)) = (&dlg.original_pixels, &dlg.original_flat)
                    {
                        let (dot_size, angle) = (dlg.dot_size, dlg.angle);
                        let shape = dlg.halftone_shape();
                        self.spawn_preview_job(
                            ctx.input(|i| i.time),
                            "Halftone".to_string(),
                            idx,
                            original.clone(),
                            flat.clone(),
                            move |img| {
                                crate::ops::effects::halftone_core(
                                    img, dot_size, angle, shape, None,
                                )
                            },
                        );
                    }
                }
                DialogResult::Ok(_) => {
                    self.preview_job_token = self.preview_job_token.wrapping_add(1);
                    let idx = dlg.layer_idx;
                    if let Some(flat) = &dlg.original_flat {
                        let (dot_size, angle) = (dlg.dot_size, dlg.angle);
                        let shape = dlg.halftone_shape();
                        if let Some(project) = self.active_project_mut() {
                            Self::apply_fullres_effect(
                                &mut project.canvas_state,
                                idx,
                                flat,
                                |img| {
                                    crate::ops::effects::halftone_core(
                                        img, dot_size, angle, shape, None,
                                    )
                                },
                            );
                        }
                    }
                    self.active_dialog = ActiveDialog::None;
                    if let Some(project) = self.active_project_mut() {
                        let idx = dlg.layer_idx;
                        if let Some(original) = &dlg.original_pixels
                            && idx < project.canvas_state.layers.len()
                        {
                            let adjusted = project.canvas_state.layers[idx].pixels.clone();
                            project.canvas_state.layers[idx].pixels = original.clone();
                            let mut cmd = SingleLayerSnapshotCommand::new_for_layer(
                                "Halftone".to_string(),
                                &project.canvas_state,
                                idx,
                            );
                            project.canvas_state.layers[idx].pixels = adjusted;
                            cmd.set_after(&project.canvas_state);
                            project.history.push(Box::new(cmd));
                        }
                        project.mark_dirty();
                    }
                    return;
                }
                DialogResult::Cancel => {
                    self.preview_job_token = self.preview_job_token.wrapping_add(1);
                    let idx = dlg.layer_idx;
                    if let Some(original) = &dlg.original_pixels
                        && let Some(project) = self.active_project_mut()
                    {
                        if let Some(layer) = project.canvas_state.layers.get_mut(idx) {
                            layer.pixels = original.clone();
                        }
                        project.canvas_state.mark_dirty(None);
                    }
                    self.active_dialog = ActiveDialog::None;
                    return;
                }
                _ => {
                    dlg.poll_flat();
                    if dlg.first_open && dlg.live_preview && dlg.original_flat.is_some() {
                        dlg.first_open = false;
                        let idx = dlg.layer_idx;
                        if let (Some(original), Some(flat)) =
                            (&dlg.original_pixels, &dlg.original_flat)
                        {
                            let (dot_size, angle) = (dlg.dot_size, dlg.angle);
                            let shape = dlg.halftone_shape();
                            self.spawn_preview_job(
                                ctx.input(|i| i.time),
                                "Halftone".to_string(),
                                idx,
                                original.clone(),
                                flat.clone(),
                                move |img| {
                                    crate::ops::effects::halftone_core(
                                        img, dot_size, angle, shape, None,
                                    )
                                },
                            );
                        }
                    }
                }
            },

            ActiveDialog::Grid(dlg) => match dlg.show(ctx) {
                DialogResult::Changed => {
                    dlg.first_open = false;
                    let idx = dlg.layer_idx;
                    if let (Some(original), Some(flat)) = (&dlg.original_pixels, &dlg.original_flat)
                    {
                        let cw = dlg.cell_w as u32;
                        let ch = dlg.cell_h as u32;
                        let lw = dlg.line_width as u32;
                        let c = [
                            (dlg.color[0] * 255.0) as u8,
                            (dlg.color[1] * 255.0) as u8,
                            (dlg.color[2] * 255.0) as u8,
                            255,
                        ];
                        let style = dlg.grid_style();
                        let opacity = dlg.opacity;
                        self.spawn_preview_job(
                            ctx.input(|i| i.time),
                            "Grid".to_string(),
                            idx,
                            original.clone(),
                            flat.clone(),
                            move |img| {
                                crate::ops::effects::grid_core(
                                    img, cw, ch, lw, c, style, opacity, None,
                                )
                            },
                        );
                    }
                }
                DialogResult::Ok(_) => {
                    self.preview_job_token = self.preview_job_token.wrapping_add(1);
                    let idx = dlg.layer_idx;
                    if let Some(flat) = &dlg.original_flat {
                        let cw = dlg.cell_w as u32;
                        let ch = dlg.cell_h as u32;
                        let lw = dlg.line_width as u32;
                        let c = [
                            (dlg.color[0] * 255.0) as u8,
                            (dlg.color[1] * 255.0) as u8,
                            (dlg.color[2] * 255.0) as u8,
                            255,
                        ];
                        let style = dlg.grid_style();
                        let opacity = dlg.opacity;
                        if let Some(project) = self.active_project_mut() {
                            Self::apply_fullres_effect(
                                &mut project.canvas_state,
                                idx,
                                flat,
                                |img| {
                                    crate::ops::effects::grid_core(
                                        img, cw, ch, lw, c, style, opacity, None,
                                    )
                                },
                            );
                        }
                    }
                    self.active_dialog = ActiveDialog::None;
                    if let Some(project) = self.active_project_mut() {
                        let idx = dlg.layer_idx;
                        if let Some(original) = &dlg.original_pixels
                            && idx < project.canvas_state.layers.len()
                        {
                            let adjusted = project.canvas_state.layers[idx].pixels.clone();
                            project.canvas_state.layers[idx].pixels = original.clone();
                            let mut cmd = SingleLayerSnapshotCommand::new_for_layer(
                                "Grid".to_string(),
                                &project.canvas_state,
                                idx,
                            );
                            project.canvas_state.layers[idx].pixels = adjusted;
                            cmd.set_after(&project.canvas_state);
                            project.history.push(Box::new(cmd));
                        }
                        project.mark_dirty();
                    }
                    return;
                }
                DialogResult::Cancel => {
                    self.preview_job_token = self.preview_job_token.wrapping_add(1);
                    let idx = dlg.layer_idx;
                    if let Some(original) = &dlg.original_pixels
                        && let Some(project) = self.active_project_mut()
                    {
                        if let Some(layer) = project.canvas_state.layers.get_mut(idx) {
                            layer.pixels = original.clone();
                        }
                        project.canvas_state.mark_dirty(None);
                    }
                    self.active_dialog = ActiveDialog::None;
                    return;
                }
                _ => {
                    dlg.poll_flat();
                    if dlg.first_open && dlg.live_preview && dlg.original_flat.is_some() {
                        dlg.first_open = false;
                        let idx = dlg.layer_idx;
                        if let (Some(original), Some(flat)) =
                            (&dlg.original_pixels, &dlg.original_flat)
                        {
                            let cw = dlg.cell_w as u32;
                            let ch = dlg.cell_h as u32;
                            let lw = dlg.line_width as u32;
                            let c = [
                                (dlg.color[0] * 255.0) as u8,
                                (dlg.color[1] * 255.0) as u8,
                                (dlg.color[2] * 255.0) as u8,
                                255,
                            ];
                            let style = dlg.grid_style();
                            let opacity = dlg.opacity;
                            self.spawn_preview_job(
                                ctx.input(|i| i.time),
                                "Grid".to_string(),
                                idx,
                                original.clone(),
                                flat.clone(),
                                move |img| {
                                    crate::ops::effects::grid_core(
                                        img, cw, ch, lw, c, style, opacity, None,
                                    )
                                },
                            );
                        }
                    }
                }
            },

            ActiveDialog::DropShadow(dlg) => match dlg.show(ctx) {
                DialogResult::Changed => {
                    dlg.first_open = false;
                    let idx = dlg.layer_idx;
                    if let (Some(original), Some(flat)) = (&dlg.original_pixels, &dlg.original_flat)
                    {
                        let ox = dlg.offset_x as i32;
                        let oy = dlg.offset_y as i32;
                        let br = dlg.blur_radius;
                        let c = [
                            (dlg.color[0] * 255.0) as u8,
                            (dlg.color[1] * 255.0) as u8,
                            (dlg.color[2] * 255.0) as u8,
                            255,
                        ];
                        let opacity = dlg.opacity;
                        self.spawn_preview_job(
                            ctx.input(|i| i.time),
                            "Drop Shadow".to_string(),
                            idx,
                            original.clone(),
                            flat.clone(),
                            move |img| {
                                crate::ops::effects::shadow_core(img, ox, oy, br, c, opacity, None)
                            },
                        );
                    }
                }
                DialogResult::Ok(_) => {
                    self.preview_job_token = self.preview_job_token.wrapping_add(1);
                    let idx = dlg.layer_idx;
                    if let Some(flat) = &dlg.original_flat {
                        let ox = dlg.offset_x as i32;
                        let oy = dlg.offset_y as i32;
                        let br = dlg.blur_radius;
                        let c = [
                            (dlg.color[0] * 255.0) as u8,
                            (dlg.color[1] * 255.0) as u8,
                            (dlg.color[2] * 255.0) as u8,
                            255,
                        ];
                        let opacity = dlg.opacity;
                        if let Some(project) = self.active_project_mut() {
                            Self::apply_fullres_effect(
                                &mut project.canvas_state,
                                idx,
                                flat,
                                |img| {
                                    crate::ops::effects::shadow_core(
                                        img, ox, oy, br, c, opacity, None,
                                    )
                                },
                            );
                        }
                    }
                    self.active_dialog = ActiveDialog::None;
                    if let Some(project) = self.active_project_mut() {
                        let idx = dlg.layer_idx;
                        if let Some(original) = &dlg.original_pixels
                            && idx < project.canvas_state.layers.len()
                        {
                            let adjusted = project.canvas_state.layers[idx].pixels.clone();
                            project.canvas_state.layers[idx].pixels = original.clone();
                            let mut cmd = SingleLayerSnapshotCommand::new_for_layer(
                                "Drop Shadow".to_string(),
                                &project.canvas_state,
                                idx,
                            );
                            project.canvas_state.layers[idx].pixels = adjusted;
                            cmd.set_after(&project.canvas_state);
                            project.history.push(Box::new(cmd));
                        }
                        project.mark_dirty();
                    }
                    return;
                }
                DialogResult::Cancel => {
                    self.preview_job_token = self.preview_job_token.wrapping_add(1);
                    let idx = dlg.layer_idx;
                    if let Some(original) = &dlg.original_pixels
                        && let Some(project) = self.active_project_mut()
                    {
                        if let Some(layer) = project.canvas_state.layers.get_mut(idx) {
                            layer.pixels = original.clone();
                        }
                        project.canvas_state.mark_dirty(None);
                    }
                    self.active_dialog = ActiveDialog::None;
                    return;
                }
                _ => {
                    dlg.poll_flat();
                    if dlg.first_open && dlg.live_preview && dlg.original_flat.is_some() {
                        dlg.first_open = false;
                        let idx = dlg.layer_idx;
                        if let (Some(original), Some(flat)) =
                            (&dlg.original_pixels, &dlg.original_flat)
                        {
                            let ox = dlg.offset_x as i32;
                            let oy = dlg.offset_y as i32;
                            let br = dlg.blur_radius;
                            let c = [
                                (dlg.color[0] * 255.0) as u8,
                                (dlg.color[1] * 255.0) as u8,
                                (dlg.color[2] * 255.0) as u8,
                                255,
                            ];
                            let opacity = dlg.opacity;
                            self.spawn_preview_job(
                                ctx.input(|i| i.time),
                                "Drop Shadow".to_string(),
                                idx,
                                original.clone(),
                                flat.clone(),
                                move |img| {
                                    crate::ops::effects::shadow_core(
                                        img, ox, oy, br, c, opacity, None,
                                    )
                                },
                            );
                        }
                    }
                }
            },

            ActiveDialog::Outline(dlg) => match dlg.show(ctx) {
                DialogResult::Changed => {
                    dlg.first_open = false;
                    let idx = dlg.layer_idx;
                    if let (Some(original), Some(flat)) = (&dlg.original_pixels, &dlg.original_flat)
                    {
                        let width = dlg.width as u32;
                        let c = [
                            (dlg.color[0] * 255.0) as u8,
                            (dlg.color[1] * 255.0) as u8,
                            (dlg.color[2] * 255.0) as u8,
                            255,
                        ];
                        let mode = dlg.outline_mode();
                        self.spawn_preview_job(
                            ctx.input(|i| i.time),
                            "Outline".to_string(),
                            idx,
                            original.clone(),
                            flat.clone(),
                            move |img| crate::ops::effects::outline_core(img, width, c, mode, None),
                        );
                    }
                }
                DialogResult::Ok(_) => {
                    self.preview_job_token = self.preview_job_token.wrapping_add(1);
                    let idx = dlg.layer_idx;
                    if let Some(flat) = &dlg.original_flat {
                        let width = dlg.width as u32;
                        let c = [
                            (dlg.color[0] * 255.0) as u8,
                            (dlg.color[1] * 255.0) as u8,
                            (dlg.color[2] * 255.0) as u8,
                            255,
                        ];
                        let mode = dlg.outline_mode();
                        if let Some(project) = self.active_project_mut() {
                            Self::apply_fullres_effect(
                                &mut project.canvas_state,
                                idx,
                                flat,
                                |img| crate::ops::effects::outline_core(img, width, c, mode, None),
                            );
                        }
                    }
                    self.active_dialog = ActiveDialog::None;
                    if let Some(project) = self.active_project_mut() {
                        let idx = dlg.layer_idx;
                        if let Some(original) = &dlg.original_pixels
                            && idx < project.canvas_state.layers.len()
                        {
                            let adjusted = project.canvas_state.layers[idx].pixels.clone();
                            project.canvas_state.layers[idx].pixels = original.clone();
                            let mut cmd = SingleLayerSnapshotCommand::new_for_layer(
                                "Outline".to_string(),
                                &project.canvas_state,
                                idx,
                            );
                            project.canvas_state.layers[idx].pixels = adjusted;
                            cmd.set_after(&project.canvas_state);
                            project.history.push(Box::new(cmd));
                        }
                        project.mark_dirty();
                    }
                    return;
                }
                DialogResult::Cancel => {
                    self.preview_job_token = self.preview_job_token.wrapping_add(1);
                    let idx = dlg.layer_idx;
                    if let Some(original) = &dlg.original_pixels
                        && let Some(project) = self.active_project_mut()
                    {
                        if let Some(layer) = project.canvas_state.layers.get_mut(idx) {
                            layer.pixels = original.clone();
                        }
                        project.canvas_state.mark_dirty(None);
                    }
                    self.active_dialog = ActiveDialog::None;
                    return;
                }
                _ => {
                    dlg.poll_flat();
                    if dlg.first_open && dlg.live_preview && dlg.original_flat.is_some() {
                        dlg.first_open = false;
                        let idx = dlg.layer_idx;
                        if let (Some(original), Some(flat)) =
                            (&dlg.original_pixels, &dlg.original_flat)
                        {
                            let width = dlg.width as u32;
                            let c = [
                                (dlg.color[0] * 255.0) as u8,
                                (dlg.color[1] * 255.0) as u8,
                                (dlg.color[2] * 255.0) as u8,
                                255,
                            ];
                            let mode = dlg.outline_mode();
                            self.spawn_preview_job(
                                ctx.input(|i| i.time),
                                "Outline".to_string(),
                                idx,
                                original.clone(),
                                flat.clone(),
                                move |img| {
                                    crate::ops::effects::outline_core(img, width, c, mode, None)
                                },
                            );
                        }
                    }
                }
            },

            ActiveDialog::PixelDrag(dlg) => match dlg.show(ctx) {
                DialogResult::Changed => {
                    dlg.first_open = false;
                    let idx = dlg.layer_idx;
                    if let (Some(original), Some(flat)) = (&dlg.original_pixels, &dlg.original_flat)
                    {
                        let (seed, amount) = (dlg.seed, dlg.amount);
                        let distance = dlg.distance as u32;
                        let direction = dlg.direction;
                        self.spawn_preview_job(
                            ctx.input(|i| i.time),
                            "Pixel Drag".to_string(),
                            idx,
                            original.clone(),
                            flat.clone(),
                            move |img| {
                                crate::ops::effects::pixel_drag_core(
                                    img, seed, amount, distance, direction, None,
                                )
                            },
                        );
                    }
                }
                DialogResult::Ok(_) => {
                    self.preview_job_token = self.preview_job_token.wrapping_add(1);
                    let idx = dlg.layer_idx;
                    if let Some(flat) = &dlg.original_flat {
                        let (seed, amount) = (dlg.seed, dlg.amount);
                        let distance = dlg.distance as u32;
                        let direction = dlg.direction;
                        if let Some(project) = self.active_project_mut() {
                            Self::apply_fullres_effect(
                                &mut project.canvas_state,
                                idx,
                                flat,
                                |img| {
                                    crate::ops::effects::pixel_drag_core(
                                        img, seed, amount, distance, direction, None,
                                    )
                                },
                            );
                        }
                    }
                    self.active_dialog = ActiveDialog::None;
                    if let Some(project) = self.active_project_mut() {
                        let idx = dlg.layer_idx;
                        if let Some(original) = &dlg.original_pixels
                            && idx < project.canvas_state.layers.len()
                        {
                            let adjusted = project.canvas_state.layers[idx].pixels.clone();
                            project.canvas_state.layers[idx].pixels = original.clone();
                            let mut cmd = SingleLayerSnapshotCommand::new_for_layer(
                                "Pixel Drag".to_string(),
                                &project.canvas_state,
                                idx,
                            );
                            project.canvas_state.layers[idx].pixels = adjusted;
                            cmd.set_after(&project.canvas_state);
                            project.history.push(Box::new(cmd));
                        }
                        project.mark_dirty();
                    }
                    return;
                }
                DialogResult::Cancel => {
                    self.preview_job_token = self.preview_job_token.wrapping_add(1);
                    let idx = dlg.layer_idx;
                    if let Some(original) = &dlg.original_pixels
                        && let Some(project) = self.active_project_mut()
                    {
                        if let Some(layer) = project.canvas_state.layers.get_mut(idx) {
                            layer.pixels = original.clone();
                        }
                        project.canvas_state.mark_dirty(None);
                    }
                    self.active_dialog = ActiveDialog::None;
                    return;
                }
                _ => {
                    dlg.poll_flat();
                    if dlg.first_open && dlg.live_preview && dlg.original_flat.is_some() {
                        dlg.first_open = false;
                        let idx = dlg.layer_idx;
                        if let (Some(original), Some(flat)) =
                            (&dlg.original_pixels, &dlg.original_flat)
                        {
                            let (seed, amount) = (dlg.seed, dlg.amount);
                            let distance = dlg.distance as u32;
                            let direction = dlg.direction;
                            self.spawn_preview_job(
                                ctx.input(|i| i.time),
                                "Pixel Drag".to_string(),
                                idx,
                                original.clone(),
                                flat.clone(),
                                move |img| {
                                    crate::ops::effects::pixel_drag_core(
                                        img, seed, amount, distance, direction, None,
                                    )
                                },
                            );
                        }
                    }
                }
            },

            ActiveDialog::RgbDisplace(dlg) => match dlg.show(ctx) {
                DialogResult::Changed => {
                    dlg.first_open = false;
                    let idx = dlg.layer_idx;
                    if let (Some(original), Some(flat)) = (&dlg.original_pixels, &dlg.original_flat)
                    {
                        let r_off = (dlg.r_x as i32, dlg.r_y as i32);
                        let g_off = (dlg.g_x as i32, dlg.g_y as i32);
                        let b_off = (dlg.b_x as i32, dlg.b_y as i32);
                        self.spawn_preview_job(
                            ctx.input(|i| i.time),
                            "RGB Displace".to_string(),
                            idx,
                            original.clone(),
                            flat.clone(),
                            move |img| {
                                crate::ops::effects::rgb_displace_core(
                                    img, r_off, g_off, b_off, None,
                                )
                            },
                        );
                    }
                }
                DialogResult::Ok(_) => {
                    self.preview_job_token = self.preview_job_token.wrapping_add(1);
                    let idx = dlg.layer_idx;
                    if let Some(flat) = &dlg.original_flat {
                        let r_off = (dlg.r_x as i32, dlg.r_y as i32);
                        let g_off = (dlg.g_x as i32, dlg.g_y as i32);
                        let b_off = (dlg.b_x as i32, dlg.b_y as i32);
                        if let Some(project) = self.active_project_mut() {
                            Self::apply_fullres_effect(
                                &mut project.canvas_state,
                                idx,
                                flat,
                                |img| {
                                    crate::ops::effects::rgb_displace_core(
                                        img, r_off, g_off, b_off, None,
                                    )
                                },
                            );
                        }
                    }
                    self.active_dialog = ActiveDialog::None;
                    if let Some(project) = self.active_project_mut() {
                        let idx = dlg.layer_idx;
                        if let Some(original) = &dlg.original_pixels
                            && idx < project.canvas_state.layers.len()
                        {
                            let adjusted = project.canvas_state.layers[idx].pixels.clone();
                            project.canvas_state.layers[idx].pixels = original.clone();
                            let mut cmd = SingleLayerSnapshotCommand::new_for_layer(
                                "RGB Displace".to_string(),
                                &project.canvas_state,
                                idx,
                            );
                            project.canvas_state.layers[idx].pixels = adjusted;
                            cmd.set_after(&project.canvas_state);
                            project.history.push(Box::new(cmd));
                        }
                        project.mark_dirty();
                    }
                    return;
                }
                DialogResult::Cancel => {
                    self.preview_job_token = self.preview_job_token.wrapping_add(1);
                    let idx = dlg.layer_idx;
                    if let Some(original) = &dlg.original_pixels
                        && let Some(project) = self.active_project_mut()
                    {
                        if let Some(layer) = project.canvas_state.layers.get_mut(idx) {
                            layer.pixels = original.clone();
                        }
                        project.canvas_state.mark_dirty(None);
                    }
                    self.active_dialog = ActiveDialog::None;
                    return;
                }
                _ => {
                    dlg.poll_flat();
                    if dlg.first_open && dlg.live_preview && dlg.original_flat.is_some() {
                        dlg.first_open = false;
                        let idx = dlg.layer_idx;
                        if let (Some(original), Some(flat)) =
                            (&dlg.original_pixels, &dlg.original_flat)
                        {
                            let r_off = (dlg.r_x as i32, dlg.r_y as i32);
                            let g_off = (dlg.g_x as i32, dlg.g_y as i32);
                            let b_off = (dlg.b_x as i32, dlg.b_y as i32);
                            self.spawn_preview_job(
                                ctx.input(|i| i.time),
                                "RGB Displace".to_string(),
                                idx,
                                original.clone(),
                                flat.clone(),
                                move |img| {
                                    crate::ops::effects::rgb_displace_core(
                                        img, r_off, g_off, b_off, None,
                                    )
                                },
                            );
                        }
                    }
                }
            },

            ActiveDialog::Ink(dlg) => match dlg.show(ctx) {
                DialogResult::Changed => {
                    dlg.first_open = false;
                    let idx = dlg.layer_idx;
                    if let (Some(original), Some(flat)) = (&dlg.original_pixels, &dlg.original_flat)
                    {
                        let (edge_strength, threshold) = (dlg.edge_strength, dlg.threshold);
                        self.spawn_preview_job(
                            ctx.input(|i| i.time),
                            "Ink".to_string(),
                            idx,
                            original.clone(),
                            flat.clone(),
                            move |img| {
                                crate::ops::effects::ink_core(img, edge_strength, threshold, None)
                            },
                        );
                    }
                }
                DialogResult::Ok(_) => {
                    self.preview_job_token = self.preview_job_token.wrapping_add(1);
                    let idx = dlg.layer_idx;
                    if let Some(flat) = &dlg.original_flat {
                        let (edge_strength, threshold) = (dlg.edge_strength, dlg.threshold);
                        if let Some(project) = self.active_project_mut() {
                            Self::apply_fullres_effect(
                                &mut project.canvas_state,
                                idx,
                                flat,
                                |img| {
                                    crate::ops::effects::ink_core(
                                        img,
                                        edge_strength,
                                        threshold,
                                        None,
                                    )
                                },
                            );
                        }
                    }
                    self.active_dialog = ActiveDialog::None;
                    if let Some(project) = self.active_project_mut() {
                        let idx = dlg.layer_idx;
                        if let Some(original) = &dlg.original_pixels
                            && idx < project.canvas_state.layers.len()
                        {
                            let adjusted = project.canvas_state.layers[idx].pixels.clone();
                            project.canvas_state.layers[idx].pixels = original.clone();
                            let mut cmd = SingleLayerSnapshotCommand::new_for_layer(
                                "Ink".to_string(),
                                &project.canvas_state,
                                idx,
                            );
                            project.canvas_state.layers[idx].pixels = adjusted;
                            cmd.set_after(&project.canvas_state);
                            project.history.push(Box::new(cmd));
                        }
                        project.mark_dirty();
                    }
                    return;
                }
                DialogResult::Cancel => {
                    self.preview_job_token = self.preview_job_token.wrapping_add(1);
                    let idx = dlg.layer_idx;
                    if let Some(original) = &dlg.original_pixels
                        && let Some(project) = self.active_project_mut()
                    {
                        if let Some(layer) = project.canvas_state.layers.get_mut(idx) {
                            layer.pixels = original.clone();
                        }
                        project.canvas_state.mark_dirty(None);
                    }
                    self.active_dialog = ActiveDialog::None;
                    return;
                }
                _ => {
                    dlg.poll_flat();
                    if dlg.first_open && dlg.live_preview && dlg.original_flat.is_some() {
                        dlg.first_open = false;
                        let idx = dlg.layer_idx;
                        if let (Some(original), Some(flat)) =
                            (&dlg.original_pixels, &dlg.original_flat)
                        {
                            let (edge_strength, threshold) = (dlg.edge_strength, dlg.threshold);
                            self.spawn_preview_job(
                                ctx.input(|i| i.time),
                                "Ink".to_string(),
                                idx,
                                original.clone(),
                                flat.clone(),
                                move |img| {
                                    crate::ops::effects::ink_core(
                                        img,
                                        edge_strength,
                                        threshold,
                                        None,
                                    )
                                },
                            );
                        }
                    }
                }
            },

            ActiveDialog::OilPainting(dlg) => match dlg.show(ctx) {
                DialogResult::Changed => {
                    dlg.first_open = false;
                    let idx = dlg.layer_idx;
                    if let (Some(original), Some(flat)) = (&dlg.original_pixels, &dlg.original_flat)
                    {
                        let (radius, levels) = (dlg.radius as u32, dlg.levels as u32);
                        self.spawn_preview_job(
                            ctx.input(|i| i.time),
                            "Oil Painting".to_string(),
                            idx,
                            original.clone(),
                            flat.clone(),
                            move |img| {
                                crate::ops::effects::oil_painting_core(img, radius, levels, None)
                            },
                        );
                    }
                }
                DialogResult::Ok(_) => {
                    self.preview_job_token = self.preview_job_token.wrapping_add(1);
                    let idx = dlg.layer_idx;
                    if let Some(flat) = &dlg.original_flat {
                        let (radius, levels) = (dlg.radius as u32, dlg.levels as u32);
                        if let Some(project) = self.active_project_mut() {
                            Self::apply_fullres_effect(
                                &mut project.canvas_state,
                                idx,
                                flat,
                                |img| {
                                    crate::ops::effects::oil_painting_core(
                                        img, radius, levels, None,
                                    )
                                },
                            );
                        }
                    }
                    self.active_dialog = ActiveDialog::None;
                    if let Some(project) = self.active_project_mut() {
                        let idx = dlg.layer_idx;
                        if let Some(original) = &dlg.original_pixels
                            && idx < project.canvas_state.layers.len()
                        {
                            let adjusted = project.canvas_state.layers[idx].pixels.clone();
                            project.canvas_state.layers[idx].pixels = original.clone();
                            let mut cmd = SingleLayerSnapshotCommand::new_for_layer(
                                "Oil Painting".to_string(),
                                &project.canvas_state,
                                idx,
                            );
                            project.canvas_state.layers[idx].pixels = adjusted;
                            cmd.set_after(&project.canvas_state);
                            project.history.push(Box::new(cmd));
                        }
                        project.mark_dirty();
                    }
                    return;
                }
                DialogResult::Cancel => {
                    self.preview_job_token = self.preview_job_token.wrapping_add(1);
                    let idx = dlg.layer_idx;
                    if let Some(original) = &dlg.original_pixels
                        && let Some(project) = self.active_project_mut()
                    {
                        if let Some(layer) = project.canvas_state.layers.get_mut(idx) {
                            layer.pixels = original.clone();
                        }
                        project.canvas_state.mark_dirty(None);
                    }
                    self.active_dialog = ActiveDialog::None;
                    return;
                }
                _ => {
                    dlg.poll_flat();
                    if dlg.first_open && dlg.live_preview && dlg.original_flat.is_some() {
                        dlg.first_open = false;
                        let idx = dlg.layer_idx;
                        if let (Some(original), Some(flat)) =
                            (&dlg.original_pixels, &dlg.original_flat)
                        {
                            let (radius, levels) = (dlg.radius as u32, dlg.levels as u32);
                            self.spawn_preview_job(
                                ctx.input(|i| i.time),
                                "Oil Painting".to_string(),
                                idx,
                                original.clone(),
                                flat.clone(),
                                move |img| {
                                    crate::ops::effects::oil_painting_core(
                                        img, radius, levels, None,
                                    )
                                },
                            );
                        }
                    }
                }
            },

            ActiveDialog::ColorFilter(dlg) => match dlg.show(ctx) {
                DialogResult::Changed => {
                    dlg.first_open = false;
                    let idx = dlg.layer_idx;
                    if let (Some(original), Some(flat)) = (&dlg.original_pixels, &dlg.original_flat)
                    {
                        let fc = [
                            (dlg.color[0] * 255.0) as u8,
                            (dlg.color[1] * 255.0) as u8,
                            (dlg.color[2] * 255.0) as u8,
                            255,
                        ];
                        let intensity = dlg.intensity;
                        let mode = dlg.filter_mode();
                        self.spawn_preview_job(
                            ctx.input(|i| i.time),
                            "Color Filter".to_string(),
                            idx,
                            original.clone(),
                            flat.clone(),
                            move |img| {
                                crate::ops::effects::color_filter_core(
                                    img, fc, intensity, mode, None,
                                )
                            },
                        );
                    }
                }
                DialogResult::Ok(_) => {
                    self.preview_job_token = self.preview_job_token.wrapping_add(1);
                    let idx = dlg.layer_idx;
                    if let Some(flat) = &dlg.original_flat {
                        let fc = [
                            (dlg.color[0] * 255.0) as u8,
                            (dlg.color[1] * 255.0) as u8,
                            (dlg.color[2] * 255.0) as u8,
                            255,
                        ];
                        let intensity = dlg.intensity;
                        let mode = dlg.filter_mode();
                        if let Some(project) = self.active_project_mut() {
                            Self::apply_fullres_effect(
                                &mut project.canvas_state,
                                idx,
                                flat,
                                |img| {
                                    crate::ops::effects::color_filter_core(
                                        img, fc, intensity, mode, None,
                                    )
                                },
                            );
                        }
                    }
                    self.active_dialog = ActiveDialog::None;
                    if let Some(project) = self.active_project_mut() {
                        let idx = dlg.layer_idx;
                        if let Some(original) = &dlg.original_pixels
                            && idx < project.canvas_state.layers.len()
                        {
                            let adjusted = project.canvas_state.layers[idx].pixels.clone();
                            project.canvas_state.layers[idx].pixels = original.clone();
                            let mut cmd = SingleLayerSnapshotCommand::new_for_layer(
                                "Color Filter".to_string(),
                                &project.canvas_state,
                                idx,
                            );
                            project.canvas_state.layers[idx].pixels = adjusted;
                            cmd.set_after(&project.canvas_state);
                            project.history.push(Box::new(cmd));
                        }
                        project.mark_dirty();
                    }
                    return;
                }
                DialogResult::Cancel => {
                    self.preview_job_token = self.preview_job_token.wrapping_add(1);
                    let idx = dlg.layer_idx;
                    if let Some(original) = &dlg.original_pixels
                        && let Some(project) = self.active_project_mut()
                    {
                        if let Some(layer) = project.canvas_state.layers.get_mut(idx) {
                            layer.pixels = original.clone();
                        }
                        project.canvas_state.mark_dirty(None);
                    }
                    self.active_dialog = ActiveDialog::None;
                    return;
                }
                _ => {
                    dlg.poll_flat();
                    if dlg.first_open && dlg.live_preview && dlg.original_flat.is_some() {
                        dlg.first_open = false;
                        let idx = dlg.layer_idx;
                        if let (Some(original), Some(flat)) =
                            (&dlg.original_pixels, &dlg.original_flat)
                        {
                            let fc = [
                                (dlg.color[0] * 255.0) as u8,
                                (dlg.color[1] * 255.0) as u8,
                                (dlg.color[2] * 255.0) as u8,
                                255,
                            ];
                            let intensity = dlg.intensity;
                            let mode = dlg.filter_mode();
                            self.spawn_preview_job(
                                ctx.input(|i| i.time),
                                "Color Filter".to_string(),
                                idx,
                                original.clone(),
                                flat.clone(),
                                move |img| {
                                    crate::ops::effects::color_filter_core(
                                        img, fc, intensity, mode, None,
                                    )
                                },
                            );
                        }
                    }
                }
            },

            ActiveDialog::Contours(dlg) => match dlg.show(ctx) {
                DialogResult::Changed => {
                    dlg.first_open = false;
                    let idx = dlg.layer_idx;
                    if let (Some(original), Some(flat)) = (&dlg.original_pixels, &dlg.original_flat)
                    {
                        let (contour_scale, frequency, line_width) =
                            (dlg.scale, dlg.frequency, dlg.line_width);
                        let lc = [
                            (dlg.line_color[0] * 255.0) as u8,
                            (dlg.line_color[1] * 255.0) as u8,
                            (dlg.line_color[2] * 255.0) as u8,
                            255u8,
                        ];
                        let (seed, octaves, blend) = (dlg.seed, dlg.octaves as u32, dlg.blend);
                        self.spawn_preview_job(
                            ctx.input(|i| i.time),
                            "Contours".to_string(),
                            idx,
                            original.clone(),
                            flat.clone(),
                            move |img| {
                                crate::ops::effects::contours_core(
                                    img,
                                    contour_scale,
                                    frequency,
                                    line_width,
                                    lc,
                                    seed,
                                    octaves,
                                    blend,
                                    None,
                                )
                            },
                        );
                    }
                }
                DialogResult::Ok(_) => {
                    self.preview_job_token = self.preview_job_token.wrapping_add(1);
                    let idx = dlg.layer_idx;
                    if let Some(flat) = &dlg.original_flat {
                        let (contour_scale, frequency, line_width) =
                            (dlg.scale, dlg.frequency, dlg.line_width);
                        let lc = [
                            (dlg.line_color[0] * 255.0) as u8,
                            (dlg.line_color[1] * 255.0) as u8,
                            (dlg.line_color[2] * 255.0) as u8,
                            255u8,
                        ];
                        let (seed, octaves, blend) = (dlg.seed, dlg.octaves as u32, dlg.blend);
                        if let Some(project) = self.active_project_mut() {
                            Self::apply_fullres_effect(
                                &mut project.canvas_state,
                                idx,
                                flat,
                                |img| {
                                    crate::ops::effects::contours_core(
                                        img,
                                        contour_scale,
                                        frequency,
                                        line_width,
                                        lc,
                                        seed,
                                        octaves,
                                        blend,
                                        None,
                                    )
                                },
                            );
                        }
                    }
                    self.active_dialog = ActiveDialog::None;
                    if let Some(project) = self.active_project_mut() {
                        let idx = dlg.layer_idx;
                        if let Some(original) = &dlg.original_pixels
                            && idx < project.canvas_state.layers.len()
                        {
                            let adjusted = project.canvas_state.layers[idx].pixels.clone();
                            project.canvas_state.layers[idx].pixels = original.clone();
                            let mut cmd = SingleLayerSnapshotCommand::new_for_layer(
                                "Contours".to_string(),
                                &project.canvas_state,
                                idx,
                            );
                            project.canvas_state.layers[idx].pixels = adjusted;
                            cmd.set_after(&project.canvas_state);
                            project.history.push(Box::new(cmd));
                        }
                        project.mark_dirty();
                    }
                    return;
                }
                DialogResult::Cancel => {
                    self.preview_job_token = self.preview_job_token.wrapping_add(1);
                    let idx = dlg.layer_idx;
                    if let Some(original) = &dlg.original_pixels
                        && let Some(project) = self.active_project_mut()
                    {
                        if let Some(layer) = project.canvas_state.layers.get_mut(idx) {
                            layer.pixels = original.clone();
                        }
                        project.canvas_state.mark_dirty(None);
                    }
                    self.active_dialog = ActiveDialog::None;
                    return;
                }
                _ => {
                    dlg.poll_flat();
                    if dlg.first_open && dlg.live_preview && dlg.original_flat.is_some() {
                        dlg.first_open = false;
                        let idx = dlg.layer_idx;
                        if let (Some(original), Some(flat)) =
                            (&dlg.original_pixels, &dlg.original_flat)
                        {
                            let (contour_scale, frequency, line_width) =
                                (dlg.scale, dlg.frequency, dlg.line_width);
                            let lc = [
                                (dlg.line_color[0] * 255.0) as u8,
                                (dlg.line_color[1] * 255.0) as u8,
                                (dlg.line_color[2] * 255.0) as u8,
                                255u8,
                            ];
                            let (seed, octaves, blend) = (dlg.seed, dlg.octaves as u32, dlg.blend);
                            self.spawn_preview_job(
                                ctx.input(|i| i.time),
                                "Contours".to_string(),
                                idx,
                                original.clone(),
                                flat.clone(),
                                move |img| {
                                    crate::ops::effects::contours_core(
                                        img,
                                        contour_scale,
                                        frequency,
                                        line_width,
                                        lc,
                                        seed,
                                        octaves,
                                        blend,
                                        None,
                                    )
                                },
                            );
                        }
                    }
                }
            },

            ActiveDialog::RemoveBackground(dlg) => match dlg.show(ctx) {
                DialogResult::Ok(settings) => {
                    self.active_dialog = ActiveDialog::None;
                    if let Some(project) = self.active_project() {
                        let layer_idx = project.canvas_state.active_layer_index;
                        let original_pixels = project.canvas_state.layers[layer_idx].pixels.clone();
                        let original_flat = original_pixels.to_rgba_image();
                        let dll_path = self.settings.onnx_runtime_path.clone();
                        let model_path = self.settings.birefnet_model_path.clone();

                        self.filter_status_description = t!("status.remove_background");
                        self.spawn_filter_job(
                            ctx.input(|i| i.time),
                            "Remove Background".to_string(),
                            layer_idx,
                            original_pixels,
                            original_flat,
                            move |input_img| match crate::ops::ai::remove_background(
                                &dll_path,
                                &model_path,
                                input_img,
                                &settings,
                            ) {
                                Ok(result) => result,
                                Err(e) => {
                                    eprintln!("Remove Background failed: {}", e);
                                    input_img.clone()
                                }
                            },
                        );
                    }
                    return;
                }
                DialogResult::Cancel => {
                    self.active_dialog = ActiveDialog::None;
                    return;
                }
                _ => {}
            },

            // ================================================================
            // THRESHOLD
            // ================================================================
            ActiveDialog::Threshold(dlg) => match dlg.show(ctx) {
                DialogResult::Changed => {
                    let level = dlg.level;
                    let idx = dlg.layer_idx;
                    if let Some(flat) = &dlg.original_flat
                        && let Some(project) = self.active_project_mut()
                    {
                        crate::ops::adjustments::threshold_from_flat(
                            &mut project.canvas_state,
                            idx,
                            level,
                            flat,
                        );
                    }
                }
                DialogResult::Ok(_) => {
                    self.active_dialog = ActiveDialog::None;
                    if let Some(project) = self.active_project_mut() {
                        let idx = dlg.layer_idx;
                        if let Some(original) = &dlg.original_pixels
                            && idx < project.canvas_state.layers.len()
                        {
                            let adjusted = project.canvas_state.layers[idx].pixels.clone();
                            project.canvas_state.layers[idx].pixels = original.clone();
                            let mut cmd = SingleLayerSnapshotCommand::new_for_layer(
                                "Threshold".to_string(),
                                &project.canvas_state,
                                idx,
                            );
                            project.canvas_state.layers[idx].pixels = adjusted;
                            cmd.set_after(&project.canvas_state);
                            project.history.push(Box::new(cmd));
                        }
                        project.mark_dirty();
                    }
                    return;
                }
                DialogResult::Cancel => {
                    let idx = dlg.layer_idx;
                    if let Some(original) = &dlg.original_pixels
                        && let Some(project) = self.active_project_mut()
                    {
                        if let Some(layer) = project.canvas_state.layers.get_mut(idx) {
                            layer.pixels = original.clone();
                        }
                        project.canvas_state.mark_dirty(None);
                    }
                    self.active_dialog = ActiveDialog::None;
                    return;
                }
                _ => {}
            },

            // ================================================================
            // POSTERIZE
            // ================================================================
            ActiveDialog::Posterize(dlg) => match dlg.show(ctx) {
                DialogResult::Changed => {
                    let levels = dlg.levels;
                    let idx = dlg.layer_idx;
                    if let Some(flat) = &dlg.original_flat
                        && let Some(project) = self.active_project_mut()
                    {
                        crate::ops::adjustments::posterize_from_flat(
                            &mut project.canvas_state,
                            idx,
                            levels,
                            flat,
                        );
                    }
                }
                DialogResult::Ok(_) => {
                    self.active_dialog = ActiveDialog::None;
                    if let Some(project) = self.active_project_mut() {
                        let idx = dlg.layer_idx;
                        if let Some(original) = &dlg.original_pixels
                            && idx < project.canvas_state.layers.len()
                        {
                            let adjusted = project.canvas_state.layers[idx].pixels.clone();
                            project.canvas_state.layers[idx].pixels = original.clone();
                            let mut cmd = SingleLayerSnapshotCommand::new_for_layer(
                                "Posterize".to_string(),
                                &project.canvas_state,
                                idx,
                            );
                            project.canvas_state.layers[idx].pixels = adjusted;
                            cmd.set_after(&project.canvas_state);
                            project.history.push(Box::new(cmd));
                        }
                        project.mark_dirty();
                    }
                    return;
                }
                DialogResult::Cancel => {
                    let idx = dlg.layer_idx;
                    if let Some(original) = &dlg.original_pixels
                        && let Some(project) = self.active_project_mut()
                    {
                        if let Some(layer) = project.canvas_state.layers.get_mut(idx) {
                            layer.pixels = original.clone();
                        }
                        project.canvas_state.mark_dirty(None);
                    }
                    self.active_dialog = ActiveDialog::None;
                    return;
                }
                _ => {}
            },

            // ================================================================
            // COLOR BALANCE
            // ================================================================
            ActiveDialog::ColorBalance(dlg) => match dlg.show(ctx) {
                DialogResult::Changed => {
                    let (shadows, midtones, highlights) =
                        (dlg.shadows, dlg.midtones, dlg.highlights);
                    let idx = dlg.layer_idx;
                    if let Some(flat) = &dlg.original_flat
                        && let Some(project) = self.active_project_mut()
                    {
                        crate::ops::adjustments::color_balance_from_flat(
                            &mut project.canvas_state,
                            idx,
                            shadows,
                            midtones,
                            highlights,
                            flat,
                        );
                    }
                }
                DialogResult::Ok(_) => {
                    self.active_dialog = ActiveDialog::None;
                    if let Some(project) = self.active_project_mut() {
                        let idx = dlg.layer_idx;
                        if let Some(original) = &dlg.original_pixels
                            && idx < project.canvas_state.layers.len()
                        {
                            let adjusted = project.canvas_state.layers[idx].pixels.clone();
                            project.canvas_state.layers[idx].pixels = original.clone();
                            let mut cmd = SingleLayerSnapshotCommand::new_for_layer(
                                "Color Balance".to_string(),
                                &project.canvas_state,
                                idx,
                            );
                            project.canvas_state.layers[idx].pixels = adjusted;
                            cmd.set_after(&project.canvas_state);
                            project.history.push(Box::new(cmd));
                        }
                        project.mark_dirty();
                    }
                    return;
                }
                DialogResult::Cancel => {
                    let idx = dlg.layer_idx;
                    if let Some(original) = &dlg.original_pixels
                        && let Some(project) = self.active_project_mut()
                    {
                        if let Some(layer) = project.canvas_state.layers.get_mut(idx) {
                            layer.pixels = original.clone();
                        }
                        project.canvas_state.mark_dirty(None);
                    }
                    self.active_dialog = ActiveDialog::None;
                    return;
                }
                _ => {}
            },

            // ================================================================
            // GRADIENT MAP
            // ================================================================
            ActiveDialog::GradientMap(dlg) => match dlg.show(ctx) {
                DialogResult::Changed => {
                    let lut = dlg.build_lut();
                    let idx = dlg.layer_idx;
                    if let Some(flat) = &dlg.original_flat
                        && let Some(project) = self.active_project_mut()
                    {
                        crate::ops::adjustments::gradient_map_from_flat(
                            &mut project.canvas_state,
                            idx,
                            &lut,
                            flat,
                        );
                    }
                }
                DialogResult::Ok(lut) => {
                    self.active_dialog = ActiveDialog::None;
                    if let Some(project) = self.active_project_mut() {
                        let idx = dlg.layer_idx;
                        if let Some(original) = &dlg.original_pixels
                            && idx < project.canvas_state.layers.len()
                        {
                            let adjusted = project.canvas_state.layers[idx].pixels.clone();
                            project.canvas_state.layers[idx].pixels = original.clone();
                            let mut cmd = SingleLayerSnapshotCommand::new_for_layer(
                                "Gradient Map".to_string(),
                                &project.canvas_state,
                                idx,
                            );
                            project.canvas_state.layers[idx].pixels = adjusted;
                            cmd.set_after(&project.canvas_state);
                            project.history.push(Box::new(cmd));
                        }
                        project.mark_dirty();
                    }
                    let _ = lut;
                    return;
                }
                DialogResult::Cancel => {
                    let idx = dlg.layer_idx;
                    if let Some(original) = &dlg.original_pixels
                        && let Some(project) = self.active_project_mut()
                    {
                        if let Some(layer) = project.canvas_state.layers.get_mut(idx) {
                            layer.pixels = original.clone();
                        }
                        project.canvas_state.mark_dirty(None);
                    }
                    self.active_dialog = ActiveDialog::None;
                    return;
                }
                _ => {}
            },

            // ================================================================
            // BLACK AND WHITE
            // ================================================================
            ActiveDialog::BlackAndWhite(dlg) => match dlg.show(ctx) {
                DialogResult::Changed => {
                    let (r, g, b) = (dlg.r_weight, dlg.g_weight, dlg.b_weight);
                    let idx = dlg.layer_idx;
                    if let Some(flat) = &dlg.original_flat
                        && let Some(project) = self.active_project_mut()
                    {
                        crate::ops::adjustments::black_and_white_from_flat(
                            &mut project.canvas_state,
                            idx,
                            r,
                            g,
                            b,
                            flat,
                        );
                    }
                }
                DialogResult::Ok(_) => {
                    self.active_dialog = ActiveDialog::None;
                    if let Some(project) = self.active_project_mut() {
                        let idx = dlg.layer_idx;
                        if let Some(original) = &dlg.original_pixels
                            && idx < project.canvas_state.layers.len()
                        {
                            let adjusted = project.canvas_state.layers[idx].pixels.clone();
                            project.canvas_state.layers[idx].pixels = original.clone();
                            let mut cmd = SingleLayerSnapshotCommand::new_for_layer(
                                "Black & White".to_string(),
                                &project.canvas_state,
                                idx,
                            );
                            project.canvas_state.layers[idx].pixels = adjusted;
                            cmd.set_after(&project.canvas_state);
                            project.history.push(Box::new(cmd));
                        }
                        project.mark_dirty();
                    }
                    return;
                }
                DialogResult::Cancel => {
                    let idx = dlg.layer_idx;
                    if let Some(original) = &dlg.original_pixels
                        && let Some(project) = self.active_project_mut()
                    {
                        if let Some(layer) = project.canvas_state.layers.get_mut(idx) {
                            layer.pixels = original.clone();
                        }
                        project.canvas_state.mark_dirty(None);
                    }
                    self.active_dialog = ActiveDialog::None;
                    return;
                }
                _ => {}
            },

            // ================================================================
            // VIBRANCE
            // ================================================================
            ActiveDialog::Vibrance(dlg) => match dlg.show(ctx) {
                DialogResult::Changed => {
                    let amount = dlg.amount;
                    let idx = dlg.layer_idx;
                    if let Some(flat) = &dlg.original_flat
                        && let Some(project) = self.active_project_mut()
                    {
                        crate::ops::adjustments::vibrance_from_flat(
                            &mut project.canvas_state,
                            idx,
                            amount,
                            flat,
                        );
                    }
                }
                DialogResult::Ok(_) => {
                    self.active_dialog = ActiveDialog::None;
                    if let Some(project) = self.active_project_mut() {
                        let idx = dlg.layer_idx;
                        if let Some(original) = &dlg.original_pixels
                            && idx < project.canvas_state.layers.len()
                        {
                            let adjusted = project.canvas_state.layers[idx].pixels.clone();
                            project.canvas_state.layers[idx].pixels = original.clone();
                            let mut cmd = SingleLayerSnapshotCommand::new_for_layer(
                                "Vibrance".to_string(),
                                &project.canvas_state,
                                idx,
                            );
                            project.canvas_state.layers[idx].pixels = adjusted;
                            cmd.set_after(&project.canvas_state);
                            project.history.push(Box::new(cmd));
                        }
                        project.mark_dirty();
                    }
                    return;
                }
                DialogResult::Cancel => {
                    let idx = dlg.layer_idx;
                    if let Some(original) = &dlg.original_pixels
                        && let Some(project) = self.active_project_mut()
                    {
                        if let Some(layer) = project.canvas_state.layers.get_mut(idx) {
                            layer.pixels = original.clone();
                        }
                        project.canvas_state.mark_dirty(None);
                    }
                    self.active_dialog = ActiveDialog::None;
                    return;
                }
                _ => {}
            },
            ActiveDialog::ColorRange(dlg) => {
                match dlg.show(ctx) {
                    DialogResult::Changed => {
                        // Restore original selection then re-run so live preview is correct
                        if let Some(project) = self.active_project_mut() {
                            project.canvas_state.selection_mask = dlg.original_selection.clone();
                            crate::ops::adjustments::select_color_range(
                                &mut project.canvas_state,
                                dlg.hue_center,
                                dlg.hue_tolerance,
                                dlg.sat_min,
                                dlg.fuzziness,
                                dlg.mode,
                            );
                            project.canvas_state.invalidate_selection_overlay();
                            project.canvas_state.mark_dirty(None);
                        }
                    }
                    DialogResult::Ok(_) => {
                        // Selection already applied from last Changed event; just commit
                        self.active_dialog = ActiveDialog::None;
                        return;
                    }
                    DialogResult::Cancel => {
                        // Restore original selection
                        if let Some(project) = self.active_project_mut() {
                            project.canvas_state.selection_mask = dlg.original_selection.clone();
                            project.canvas_state.invalidate_selection_overlay();
                            project.canvas_state.mark_dirty(None);
                        }
                        self.active_dialog = ActiveDialog::None;
                        return;
                    }
                    _ => {}
                }
            }
        }

        // If we reach here the dialog is still open — put it back
        self.active_dialog = dialog;
    }
}

// --- Floating Panel Methods ---
impl PaintFEApp {
    /// Show the floating Tools panel (minimalist vertical strip) - anchored to left edge
    fn show_floating_tools_panel(&mut self, ctx: &egui::Context, screen_size_changed: bool) {
        let mut show = self.window_visibility.tools;
        let mut close_clicked = false;

        let first_show = self.tools_panel_pos.is_none();
        let (pos_x, pos_y) = self.tools_panel_pos.unwrap_or((12.0, 128.0));

        let mut window = egui::Window::new("ToolsStrip")
            .open(&mut show)
            .resizable(false)
            .collapsible(false)
            .fixed_size(egui::vec2(60.0, 150.0))
            .title_bar(false)
            .frame(self.theme.floating_window_frame());

        if first_show || screen_size_changed {
            window = window.current_pos(egui::pos2(pos_x, pos_y));
        }

        let resp = window.show(ctx, |ui| {
            // Header: title left, × right
            let header_rect = ui.available_rect_before_wrap();
            let header_rect =
                egui::Rect::from_min_size(header_rect.min, egui::vec2(header_rect.width(), 18.0));
            let (header_rect, _) = ui.allocate_exact_size(header_rect.size(), egui::Sense::hover());
            ui.painter().text(
                header_rect.left_center(),
                egui::Align2::LEFT_CENTER,
                "Tools",
                egui::FontId::proportional(14.0),
                self.theme.text_color,
            );
            let close_rect = egui::Rect::from_min_size(
                egui::pos2(header_rect.right() + 13.0, header_rect.center().y - 8.0),
                egui::vec2(16.0, 16.0),
            );
            let close_response = ui.allocate_rect(close_rect, egui::Sense::click());
            // Draw button background with border on hover
            let (bg_color, border_stroke) = if close_response.hovered() {
                (
                    ui.visuals().widgets.hovered.bg_fill,
                    egui::Stroke::new(1.0, ui.visuals().widgets.hovered.bg_stroke.color),
                )
            } else {
                (egui::Color32::from_black_alpha(40), egui::Stroke::NONE)
            };
            ui.painter().rect(close_rect, 2.0, bg_color, border_stroke);
            ui.painter().text(
                close_rect.center(),
                egui::Align2::CENTER_CENTER,
                "×",
                egui::FontId::proportional(14.0),
                if close_response.hovered() {
                    ui.visuals().widgets.hovered.fg_stroke.color
                } else {
                    ui.visuals().text_color()
                },
            );
            if close_response.clicked() {
                close_clicked = true;
            }
            ui.add_space(5.0);
            // Make all text in this window slightly smaller
            ui.style_mut().override_text_style = Some(egui::TextStyle::Small);
            let primary = self.colors_panel.get_primary_color();
            let secondary = self.colors_panel.get_secondary_color();

            let action = self.tools_panel.show_compact(
                ui,
                &self.assets,
                primary,
                secondary,
                &self.settings.keybindings,
            );

            match action {
                tools::ToolsPanelAction::OpenColors => {
                    self.window_visibility.colors = !self.window_visibility.colors;
                }
                tools::ToolsPanelAction::SwapColors => {
                    self.colors_panel.swap_colors();
                }
                tools::ToolsPanelAction::None => {}
            }
        });

        if let Some(inner_resp) = resp {
            let win_rect = inner_resp.response.rect;
            self.tools_panel_pos = Some((win_rect.min.x, win_rect.min.y));
        }

        if close_clicked {
            show = false;
        }
        self.window_visibility.tools = show;
    }

    /// Show the floating Layers panel
    fn show_floating_layers_panel(&mut self, ctx: &egui::Context, screen_size_changed: bool) {
        let mut show = self.window_visibility.layers;
        let mut close_clicked = false;

        let screen_rect = ctx.screen_rect();
        let screen_w = screen_rect.max.x;

        let first_show = self.layers_panel_right_offset.is_none();

        // Default: 12px from right edge, 12px below menu bar
        let (right_off, y_pos) = self.layers_panel_right_offset.unwrap_or((264.0, 128.0));
        let pos_x = screen_w - right_off;

        let mut window = egui::Window::new("Layers")
            .open(&mut show)
            .resizable(true)
            .collapsible(false)
            .default_size(egui::vec2(240.0, 200.0))
            .min_width(180.0)
            .min_height(200.0)
            .title_bar(false)
            .frame(self.theme.floating_window_frame());

        // Only force position on first show or when screen size changes
        if first_show || screen_size_changed {
            window = window.current_pos(egui::pos2(pos_x, y_pos));
        }

        let resp = window.show(ctx, |ui| {
            // Custom left-aligned title with close button
            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new("Layers")
                        .size(14.0)
                        .color(self.theme.text_color),
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.small_button("×").clicked() {
                        close_clicked = true;
                    }
                });
            });
            ui.add_space(4.0);
            if let Some(project) = self.projects.get_mut(self.active_project_index) {
                self.layers_panel.show(
                    ui,
                    &mut project.canvas_state,
                    &self.assets,
                    &mut project.history,
                );
            }
            // Drain pending GPU delete from the layers panel.
            if let Some(del_idx) = self.layers_panel.pending_gpu_delete.take() {
                self.canvas.gpu_remove_layer(del_idx);
            }
            if self.layers_panel.pending_gpu_clear {
                self.layers_panel.pending_gpu_clear = false;
                self.canvas.gpu_clear_layers();
            }
            // Handle pending app-level actions from layers context menu
            if let Some(action) = self.layers_panel.pending_app_action.take() {
                match action {
                    crate::components::layers::LayerAppAction::ImportFromFile => {
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter(
                                "Image",
                                &[
                                    "png", "jpg", "jpeg", "bmp", "gif", "webp", "tiff", "tif",
                                    "tga", "ico",
                                ],
                            )
                            .pick_file()
                            && let Ok(img) = image::open(&path)
                        {
                            let rgba = img.to_rgba8();
                            let name = path
                                .file_stem()
                                .map(|s| s.to_string_lossy().to_string())
                                .unwrap_or_else(|| "Imported".to_string());
                            self.do_snapshot_op("Import Layer", |s| {
                                crate::ops::adjustments::import_layer_from_image(s, &rgba, &name);
                            });
                        }
                    }
                    crate::components::layers::LayerAppAction::FlipHorizontal => {
                        self.do_layer_snapshot_op("Flip Layer H", |s| {
                            let idx = s.active_layer_index;
                            crate::ops::transform::flip_layer_horizontal(s, idx);
                        });
                    }
                    crate::components::layers::LayerAppAction::FlipVertical => {
                        self.do_layer_snapshot_op("Flip Layer V", |s| {
                            let idx = s.active_layer_index;
                            crate::ops::transform::flip_layer_vertical(s, idx);
                        });
                    }
                    crate::components::layers::LayerAppAction::RotateScale => {
                        if let Some(project) = self.projects.get(self.active_project_index) {
                            self.active_dialog = ActiveDialog::LayerTransform(
                                crate::ops::dialogs::LayerTransformDialog::new(
                                    &project.canvas_state,
                                ),
                            );
                        }
                    }
                    crate::components::layers::LayerAppAction::MergeDownAsMask(layer_idx) => {
                        self.do_snapshot_op("Merge Down as Mask", |s| {
                            crate::ops::canvas_ops::merge_down_as_mask(s, layer_idx);
                        });
                        self.layers_panel.pending_gpu_clear = true;
                    }
                }
            }
        });

        // Update the stored right-edge offset from the window's actual position
        // so that user drags are remembered and window resizes keep the offset.
        if let Some(inner_resp) = resp {
            let win_rect = inner_resp.response.rect;
            self.layers_panel_right_offset = Some((screen_w - win_rect.min.x, win_rect.min.y));
        }

        if close_clicked {
            show = false;
        }
        self.window_visibility.layers = show;
    }

    /// Show the floating History panel
    fn show_floating_history_panel(&mut self, ctx: &egui::Context, screen_size_changed: bool) {
        let mut show = self.window_visibility.history;
        let mut close_clicked = false;

        let screen_rect = ctx.screen_rect();
        let screen_w = screen_rect.max.x;
        let screen_h = screen_rect.max.y;

        let first_show = self.history_panel_right_offset.is_none();

        // Default: 12px from right edge, 12px from bottom
        let (right_off, bot_off) = self.history_panel_right_offset.unwrap_or((230.0, 242.0));
        let pos_x = screen_w - right_off;
        let pos_y = screen_h - bot_off;

        let mut window = egui::Window::new("History")
            .open(&mut show)
            .resizable(false)
            .collapsible(false)
            .min_width(200.0)
            .default_size(egui::vec2(200.0, 200.0))
            .title_bar(false)
            .frame(self.theme.floating_window_frame());

        if first_show || screen_size_changed {
            window = window.current_pos(egui::pos2(pos_x, pos_y));
        }

        let resp = window.show(ctx, |ui| {
            // Custom left-aligned title with close button
            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new("History")
                        .size(14.0)
                        .color(self.theme.text_color),
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.small_button("×").clicked() {
                        close_clicked = true;
                    }
                });
            });
            ui.add_space(2.0);
            ui.style_mut().override_text_style = Some(egui::TextStyle::Small);
            if let Some(project) = self.projects.get_mut(self.active_project_index) {
                self.history_panel.show_interactive(
                    ui,
                    &mut project.history,
                    &mut project.canvas_state,
                    &self.assets,
                );
            }
        });

        if let Some(inner_resp) = resp {
            let win_rect = inner_resp.response.rect;
            self.history_panel_right_offset =
                Some((screen_w - win_rect.min.x, screen_h - win_rect.min.y));
        }

        if close_clicked {
            show = false;
        }
        self.window_visibility.history = show;
    }

    /// Show the floating Colors panel - anchored below tools
    fn show_floating_colors_panel(&mut self, ctx: &egui::Context, screen_size_changed: bool) {
        let mut show = self.window_visibility.colors;
        let mut close_clicked = false;

        let screen_rect = ctx.screen_rect();
        let screen_h = screen_rect.max.y;

        let first_show = self.colors_panel_left_offset.is_none();

        // Default: 12px from left, 12px from bottom (bot_off = ~360px panel height + 12)
        let (x_off, bot_off) = self.colors_panel_left_offset.unwrap_or((12.0, 372.0));
        let pos_y = screen_h - bot_off;

        // Dynamic size based on compact / expanded state
        let panel_size = if self.colors_panel.is_expanded() {
            egui::vec2(420.0, 340.0)
        } else {
            egui::vec2(210.0, 330.0)
        };

        let mut window = egui::Window::new("Colors")
            .open(&mut show)
            .resizable(false)
            .collapsible(false)
            .fixed_size(panel_size)
            .title_bar(false)
            .frame(self.theme.floating_window_frame());

        if first_show || screen_size_changed {
            window = window.current_pos(egui::pos2(x_off, pos_y));
        }

        let resp = window.show(ctx, |ui| {
            // Custom left-aligned title with close button
            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new("Colors")
                        .size(14.0)
                        .color(self.theme.text_color),
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.small_button("×").clicked() {
                        close_clicked = true;
                    }
                });
            });
            ui.add_space(2.0);
            ui.style_mut().override_text_style = Some(egui::TextStyle::Small);
            self.colors_panel.show(ui, &self.assets);
        });

        if let Some(inner_resp) = resp {
            let win_rect = inner_resp.response.rect;
            self.colors_panel_left_offset = Some((win_rect.min.x, screen_h - win_rect.min.y));
        }

        if close_clicked {
            show = false;
        }
        self.window_visibility.colors = show;
    }

    /// Show the floating Script Editor panel
    fn show_floating_script_editor(&mut self, ctx: &egui::Context, screen_size_changed: bool) {
        let mut show = self.window_visibility.script_editor;
        if !show {
            return;
        }

        let screen_rect = ctx.screen_rect();
        let screen_w = screen_rect.max.x;
        let screen_h = screen_rect.max.y;

        let first_show = self.script_right_offset.is_none();

        // Default: centered-ish position
        let (right_off, top_off) = self
            .script_right_offset
            .unwrap_or((screen_w * 0.3, screen_h * 0.15));
        let pos_x = screen_w - right_off;
        let pos_y = top_off;

        let mut window = egui::Window::new("ScriptEditor")
            .open(&mut show)
            .resizable(true)
            .collapsible(false)
            .min_width(400.0)
            .min_height(350.0)
            .default_size(egui::vec2(520.0, 500.0))
            .title_bar(false)
            .frame(self.theme.floating_window_frame());

        if first_show || screen_size_changed {
            window = window.current_pos(egui::pos2(pos_x, pos_y));
        }

        let theme_copy = self.theme.clone();
        let resp = window.show(ctx, |ui| {
            self.script_editor.show(ui, &theme_copy);

            // Handle run request
            if self.script_editor.run_requested {
                self.run_script();
            }
        });

        // Handle "Add to Filters" request from the editor
        if let Some((name, code)) = self.script_editor.pending_add_effect.take() {
            let effect = script_editor::CustomScriptEffect { name, code };
            script_editor::save_custom_effect(&effect);
            // Avoid duplicates by name
            self.custom_scripts.retain(|e| e.name != effect.name);
            self.custom_scripts.push(effect);
            self.custom_scripts
                .sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));
        }

        if let Some(inner_resp) = resp {
            let win_rect = inner_resp.response.rect;
            self.script_right_offset = Some((screen_w - win_rect.min.x, win_rect.min.y));
        }

        if self.script_editor.close_requested {
            show = false;
        }
        self.window_visibility.script_editor = show;
    }

    /// Execute the current script on the active layer
    fn run_script(&mut self) {
        if self.script_editor.is_running {
            return;
        }

        let project = match self.projects.get(self.active_project_index) {
            Some(p) => p,
            None => {
                self.script_editor.add_console_line(
                    "No active project".to_string(),
                    crate::components::script_editor::ConsoleLineKind::Error,
                );
                return;
            }
        };

        let state = &project.canvas_state;
        let layer_idx = state.active_layer_index;
        if layer_idx >= state.layers.len() {
            self.script_editor.add_console_line(
                "No active layer".to_string(),
                crate::components::script_editor::ConsoleLineKind::Error,
            );
            return;
        }

        let layer = &state.layers[layer_idx];
        let w = state.width;
        let h = state.height;

        // Extract flat pixels from tiled image
        let flat_pixels = layer.pixels.extract_region_rgba(0, 0, w, h);

        // Clone original for undo
        let original_pixels = layer.pixels.clone();

        // Get selection mask if any
        let mask = state.selection_mask.as_ref().map(|m| m.as_raw().clone());

        // Reset cancel flag
        let cancel_flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        self.script_editor.cancel_flag = cancel_flag.clone();

        // Save backup of layer pixels for restore on error
        self.script_original_pixels =
            Some((self.active_project_index, layer_idx, layer.pixels.clone()));

        self.script_editor.is_running = true;
        self.script_editor.progress = None;
        self.script_editor.error_line = None; // Clear previous error highlight
        self.script_editor.add_console_line(
            "Running script...".to_string(),
            crate::components::script_editor::ConsoleLineKind::Info,
        );

        crate::ops::scripting::execute_script(
            self.script_editor.code.clone(),
            self.active_project_index,
            layer_idx,
            original_pixels,
            flat_pixels,
            w,
            h,
            mask,
            cancel_flag,
            self.script_sender.clone(),
        );
    }

    /// Execute a custom script effect from Filter > Custom (no editor needed)
    fn run_custom_script(&mut self, code: String, name: String) {
        if self.script_editor.is_running {
            return;
        }

        let project = match self.projects.get(self.active_project_index) {
            Some(p) => p,
            None => return,
        };

        let state = &project.canvas_state;
        let layer_idx = state.active_layer_index;
        if layer_idx >= state.layers.len() {
            return;
        }

        let layer = &state.layers[layer_idx];
        let w = state.width;
        let h = state.height;

        let flat_pixels = layer.pixels.extract_region_rgba(0, 0, w, h);
        let original_pixels = layer.pixels.clone();
        let mask = state.selection_mask.as_ref().map(|m| m.as_raw().clone());

        let cancel_flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        self.script_editor.cancel_flag = cancel_flag.clone();
        // Save backup of layer pixels for restore on error
        self.script_original_pixels =
            Some((self.active_project_index, layer_idx, layer.pixels.clone()));

        self.script_editor.is_running = true;
        self.script_editor.progress = None;
        self.script_editor.error_line = None;
        self.script_editor.console_output.clear();
        self.script_editor.add_console_line(
            "Running custom effect...".to_string(),
            crate::components::script_editor::ConsoleLineKind::Info,
        );

        // Show spinner in status bar
        self.pending_filter_jobs += 1;
        self.filter_status_description = format!("Script: {}", name);

        crate::ops::scripting::execute_script(
            code,
            self.active_project_index,
            layer_idx,
            original_pixels,
            flat_pixels,
            w,
            h,
            mask,
            cancel_flag,
            self.script_sender.clone(),
        );
    }
}
