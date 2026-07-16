use crate::config::keybindings::KeyBindings;
use crate::theme::{AccentColors, ThemeMode, ThemeOverrides, ThemePreset, UiDensity};
use egui::Color32;
use std::path::PathBuf;

/// Zoom filter mode for canvas rendering
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ZoomFilterMode {
    /// Smooth (LINEAR) - better for zoomed out, may show edge artifacts on transparency
    Linear,
    /// Sharp (NEAREST) - pixel-perfect, no artifacts
    Nearest,
}

impl ZoomFilterMode {
    pub fn name(&self) -> String {
        match self {
            ZoomFilterMode::Linear => t!("zoom_filter_mode.linear"),
            ZoomFilterMode::Nearest => t!("zoom_filter_mode.nearest"),
        }
    }

    pub fn all() -> &'static [ZoomFilterMode] {
        &[ZoomFilterMode::Linear, ZoomFilterMode::Nearest]
    }
}

/// Application settings that persist across sessions
#[derive(Clone, Debug)]
pub struct AppSettings {
    /// Theme mode (Light or Dark)
    pub theme_mode: ThemeMode,
    /// Active accent preset
    pub theme_preset: ThemePreset,
    /// Custom accent colors (used when preset == Custom)
    pub custom_accent: AccentColors,
    /// Preferred GPU adapter name (e.g. \"NVIDIA RTX 3080\")
    pub preferred_gpu: String,
    /// Pixel grid display mode
    pub pixel_grid_mode: PixelGridMode,
    /// Pixel grid outline color (dual-stroke black outline)
    pub pixel_grid_outline_color: Color32,
    /// Pixel grid center color (dual-stroke white center)
    pub pixel_grid_center_color: Color32,
    pub selection_stripe_color: Color32,
    pub selection_stripe_alpha: u8,
    /// Preset colors shown for layer folders.
    pub folder_color_palette: [Color32; 8],
    /// Maximum number of undo steps
    pub max_undo_steps: usize,
    /// Auto-save interval in minutes (0 = disabled)
    pub auto_save_minutes: u32,
    /// Neon glow mode – accent-colored shadows in dark theme
    pub neon_mode: bool,
    /// Zoom filter mode for canvas rendering
    pub zoom_filter_mode: ZoomFilterMode,
    /// Checkerboard brightness multiplier (1.0 = default, 0.5 = darker, 1.5 = lighter)
    pub checkerboard_brightness: f32,

    // AI / ONNX Runtime settings
    /// Path to onnxruntime.dll / libonnxruntime.so
    pub onnx_runtime_path: String,
    /// Path to BiRefNet .onnx model file
    pub birefnet_model_path: String,

    // Experimental Paint.NET legacy plugin compatibility.
    pub paintdotnet_plugins_enabled: bool,

    // Debug panel settings
    pub show_debug_panel: bool,
    pub show_tool_info: bool,
    pub debug_show_canvas_size: bool,
    pub debug_show_zoom: bool,
    pub debug_show_fps: bool,
    pub debug_show_gpu: bool,
    pub debug_show_operations: bool,

    // Keybindings
    pub keybindings: KeyBindings,

    // Localisation
    /// Language code (e.g. "en", "es", "fr"). Empty string = auto-detect system language.
    pub language: String,

    // Startup canvas
    /// Default canvas width for new untitled projects.
    pub default_canvas_width: u32,
    /// Default canvas height for new untitled projects.
    pub default_canvas_height: u32,
    /// Whether to create a blank canvas on startup (false = app starts empty).
    pub create_canvas_on_startup: bool,

    // Behaviour
    /// Show a save-confirmation dialog when the user exits with unsaved projects.
    pub confirm_on_exit: bool,

    // Main window size persistence
    pub persist_window_width: f32,
    pub persist_window_height: f32,
    pub persist_window_pos: Option<(f32, f32)>,

    // Floating widget persistence
    pub persist_tools_visible: bool,
    pub persist_layers_visible: bool,
    pub persist_history_visible: bool,
    pub persist_colors_visible: bool,
    pub persist_palette_visible: bool,
    pub persist_script_editor_visible: bool,
    pub persist_tools_panel_pos: Option<(f32, f32)>,
    pub persist_layers_panel_right_offset: Option<(f32, f32)>,
    pub persist_layers_panel_size: Option<(f32, f32)>,
    pub persist_history_panel_right_offset: Option<(f32, f32)>,
    pub persist_history_panel_size: Option<(f32, f32)>,
    pub persist_colors_panel_left_offset: Option<(f32, f32)>,
    pub persist_palette_panel_pos: Option<(f32, f32)>,
    pub persist_palette_panel_left_offset: Option<(f32, f32)>,
    pub persist_palette_panel_right_offset: Option<(f32, f32)>,
    pub persist_palette_recent_colors: String,
    pub persist_script_right_offset: Option<(f32, f32)>,
    pub persist_colors_panel_expanded: bool,

    // Dialog option persistence
    pub persist_new_file_lock_aspect: bool,
    pub persist_resize_lock_aspect: bool,

    // Tool option persistence
    pub persisted_active_tool: String,
    pub persisted_brush_size: f32,
    pub persisted_brush_hardness: f32,
    pub persisted_brush_flow: f32,
    pub persisted_brush_spacing: f32,
    pub persisted_brush_scatter: f32,
    pub persisted_brush_hue_jitter: f32,
    pub persisted_brush_brightness_jitter: f32,
    pub persisted_brush_anti_aliased: bool,
    pub persisted_pressure_size: bool,
    pub persisted_pressure_opacity: bool,
    pub persisted_pressure_min_size: f32,
    pub persisted_pressure_min_opacity: f32,
    pub persisted_brush_mode: String,
    pub persisted_brush_tip: String,
    /// Custom brush tips persisted across restarts: (name, category, base64_png_data)
    pub custom_brush_tips: Vec<(String, String, String)>,
    /// Custom vector shapes persisted across restarts: (name, category, base64_svg_path_data)
    pub custom_shapes: Vec<(String, String, String)>,
    pub persisted_fill_tolerance: f32,
    pub persisted_fill_anti_aliased: bool,
    pub persisted_fill_global: bool,
    pub persisted_wand_tolerance: f32,
    pub persisted_wand_anti_aliased: bool,
    pub persisted_wand_global: bool,
    pub persisted_color_remover_tolerance: f32,
    pub persisted_color_remover_smoothness: u32,
    pub persisted_color_remover_contiguous: bool,
    pub persisted_smudge_strength: f32,
    pub persisted_shapes_fill_mode: String,
    pub persisted_shapes_anti_alias: bool,
    pub persisted_shapes_corner_radius: f32,
    pub persisted_move_interpolation: String,
    pub persisted_move_anti_aliasing: bool,

    // --- Advanced Customization (Phase 10) ---
    // --- Text tool persistence ---
    /// Persisted font family for the text tool.
    pub persisted_text_font_family: String,

    // --- Clipboard behaviour ---
    /// When true, pasting cuts out transparent pixels (holes) from the destination layer.
    pub clipboard_copy_transparent_cutout: bool,

    // --- Window state ---
    /// Whether the main window was maximized on last exit.
    pub persist_window_maximized: bool,

    // --- Advanced Customization (Phase 10) ---
    /// Master toggle — when false, all overrides are ignored.
    pub advanced_customization: bool,
    /// UI density (Compact / Normal / Spacious).
    pub ui_density: UiDensity,
    /// Show subtle grid texture on canvas background.
    pub canvas_grid_visible: bool,
    /// Grid texture opacity (0.0–1.0).
    pub canvas_grid_opacity: f32,
    /// Glow intensity multiplier (0.0–2.0, default 1.0).
    pub glow_intensity: f32,
    /// Shadow strength multiplier (0.0–2.0, default 1.0).
    pub shadow_strength: f32,
    /// Widget rounding override (px).
    pub widget_rounding: Option<f32>,
    /// Window rounding override (px).
    pub window_rounding: Option<f32>,
    /// Menu rounding override (px).
    pub menu_rounding: Option<f32>,
    /// Tool shelf corner radius override (px).
    pub tool_shelf_rounding: Option<f32>,
    pub tool_button_rounding: Option<f32>,
    /// Badge corner radius override (px).
    pub badge_rounding: Option<f32>,
    /// Tab corner radius override (px).
    pub tab_rounding: Option<f32>,

    // Theme color overrides (None = use preset default)
    pub ov_bg_color: Option<Color32>,
    pub ov_panel_bg: Option<Color32>,
    pub ov_window_bg: Option<Color32>,
    pub ov_bg2: Option<Color32>,
    pub ov_bg3: Option<Color32>,
    pub ov_text_color: Option<Color32>,
    pub ov_text_muted: Option<Color32>,
    pub ov_text_faint: Option<Color32>,
    pub ov_border_color: Option<Color32>,
    pub ov_border_lit: Option<Color32>,
    pub ov_separator_color: Option<Color32>,
    pub ov_button_bg: Option<Color32>,
    pub ov_button_hover: Option<Color32>,
    pub ov_button_active: Option<Color32>,
    pub ov_floating_window_bg: Option<Color32>,
    pub ov_tool_shelf_bg: Option<Color32>,
    pub ov_tool_shelf_border: Option<Color32>,
    pub ov_toolbar_bg: Option<Color32>,
    pub ov_menu_bg: Option<Color32>,
    pub ov_icon_button_bg: Option<Color32>,
    pub ov_icon_button_active: Option<Color32>,
    pub ov_icon_button_disabled: Option<Color32>,
    pub ov_text_input_bg: Option<Color32>,
    pub ov_stepper_button_bg: Option<Color32>,
    pub ov_canvas_bg_top: Option<Color32>,
    pub ov_canvas_bg_bottom: Option<Color32>,
    pub ov_glow_accent: Option<Color32>,
    pub ov_accent3: Option<Color32>,
    pub ov_accent4: Option<Color32>,
}

impl Default for AppSettings {
    fn default() -> Self {
        let preset = ThemePreset::Signal;
        Self {
            // Web has no persisted settings file (no real filesystem), so
            // `default()` runs fresh every session — making this the
            // effective permanent default there. Desktop keeps Light as
            // its long-standing default; web starts in Dark instead.
            #[cfg(target_arch = "wasm32")]
            theme_mode: ThemeMode::Dark,
            #[cfg(not(target_arch = "wasm32"))]
            theme_mode: ThemeMode::Light,
            theme_preset: preset,
            custom_accent: preset.accent_colors(),
            preferred_gpu: "Auto".to_string(),
            pixel_grid_mode: PixelGridMode::Auto,
            pixel_grid_outline_color: Color32::from_black_alpha(90),
            pixel_grid_center_color: Color32::from_white_alpha(100),
            selection_stripe_color: Color32::from_rgba_premultiplied(255, 255, 255, 255),
            selection_stripe_alpha: 22,
            folder_color_palette: AppSettings::default_folder_color_palette(),
            max_undo_steps: 50,
            auto_save_minutes: 0,
            neon_mode: false,
            zoom_filter_mode: ZoomFilterMode::Linear,
            checkerboard_brightness: 1.0,
            onnx_runtime_path: String::new(),
            birefnet_model_path: String::new(),
            paintdotnet_plugins_enabled: false,

            show_debug_panel: true,
            show_tool_info: true,
            debug_show_canvas_size: true,
            debug_show_zoom: true,
            debug_show_fps: false,
            debug_show_gpu: false,
            debug_show_operations: true,

            keybindings: KeyBindings::default(),

            language: String::new(), // empty = auto-detect on first boot

            default_canvas_width: 800,
            default_canvas_height: 600,
            create_canvas_on_startup: true,

            confirm_on_exit: true,

            persist_window_width: 1280.0,
            persist_window_height: 720.0,
            persist_window_pos: None,

            persist_tools_visible: true,
            persist_layers_visible: true,
            persist_history_visible: false,
            persist_colors_visible: false,
            persist_palette_visible: false,
            persist_script_editor_visible: false,
            persist_tools_panel_pos: None,
            persist_layers_panel_right_offset: None,
            persist_layers_panel_size: None,
            persist_history_panel_right_offset: None,
            persist_history_panel_size: None,
            persist_colors_panel_left_offset: None,
            persist_palette_panel_pos: None,
            persist_palette_panel_left_offset: None,
            persist_palette_panel_right_offset: None,
            persist_palette_recent_colors: String::new(),
            persist_script_right_offset: None,
            persist_colors_panel_expanded: false,

            persist_new_file_lock_aspect: true,
            persist_resize_lock_aspect: true,

            persisted_active_tool: "brush".to_string(),
            persisted_brush_size: 10.0,
            persisted_brush_hardness: 0.75,
            persisted_brush_flow: 1.0,
            persisted_brush_spacing: 0.01,
            persisted_brush_scatter: 0.0,
            persisted_brush_hue_jitter: 0.0,
            persisted_brush_brightness_jitter: 0.0,
            persisted_brush_anti_aliased: true,
            persisted_pressure_size: false,
            persisted_pressure_opacity: false,
            persisted_pressure_min_size: 0.1,
            persisted_pressure_min_opacity: 0.1,
            persisted_brush_mode: "normal".to_string(),
            persisted_brush_tip: String::new(),
            custom_brush_tips: Vec::new(),
            custom_shapes: Vec::new(),
            persisted_fill_tolerance: 5.0,
            persisted_fill_anti_aliased: true,
            persisted_fill_global: false,
            persisted_wand_tolerance: 5.0,
            persisted_wand_anti_aliased: true,
            persisted_wand_global: false,
            persisted_color_remover_tolerance: 5.0,
            persisted_color_remover_smoothness: 3,
            persisted_color_remover_contiguous: true,
            persisted_smudge_strength: 0.6,
            persisted_shapes_fill_mode: "filled".to_string(),
            persisted_shapes_anti_alias: true,
            persisted_shapes_corner_radius: 10.0,
            persisted_move_interpolation: "bilinear".to_string(),
            persisted_move_anti_aliasing: true,

            // Advanced Customization defaults
            persisted_text_font_family: String::new(),
            clipboard_copy_transparent_cutout: true,
            persist_window_maximized: false,

            // Advanced Customization defaults
            advanced_customization: false,
            ui_density: UiDensity::Normal,
            canvas_grid_visible: true,
            canvas_grid_opacity: 0.4,
            glow_intensity: 1.0,
            shadow_strength: 1.0,
            widget_rounding: None,
            window_rounding: None,
            menu_rounding: None,
            tool_shelf_rounding: None,
            tool_button_rounding: None,
            badge_rounding: None,
            tab_rounding: None,
            ov_bg_color: None,
            ov_panel_bg: None,
            ov_window_bg: None,
            ov_bg2: None,
            ov_bg3: None,
            ov_text_color: None,
            ov_text_muted: None,
            ov_text_faint: None,
            ov_border_color: None,
            ov_border_lit: None,
            ov_separator_color: None,
            ov_button_bg: None,
            ov_button_hover: None,
            ov_button_active: None,
            ov_floating_window_bg: None,
            ov_tool_shelf_bg: None,
            ov_tool_shelf_border: None,
            ov_toolbar_bg: None,
            ov_menu_bg: None,
            ov_icon_button_bg: None,
            ov_icon_button_active: None,
            ov_icon_button_disabled: None,
            ov_text_input_bg: None,
            ov_stepper_button_bg: None,
            ov_canvas_bg_top: None,
            ov_canvas_bg_bottom: None,
            ov_glow_accent: None,
            ov_accent3: None,
            ov_accent4: None,
        }
    }
}

impl AppSettings {
    pub fn default_folder_color_palette() -> [Color32; 8] {
        [
            Color32::from_rgb(79, 140, 255),
            Color32::from_rgb(32, 181, 126),
            Color32::from_rgb(238, 184, 48),
            Color32::from_rgb(238, 108, 77),
            Color32::from_rgb(169, 120, 255),
            Color32::from_rgb(236, 105, 170),
            Color32::from_rgb(80, 185, 214),
            Color32::from_rgb(142, 151, 166),
        ]
    }

    /// Path to the settings file.
    /// On Linux:   ~/.config/paintfe/paintfe_settings.cfg  (XDG_CONFIG_HOME respected)
    /// On Windows: %APPDATA%\PaintFE\paintfe_settings.cfg
    /// On macOS:   ~/Library/Application Support/PaintFE/paintfe_settings.cfg
    /// Fallback:   same directory as the executable.
    pub(crate) fn settings_path() -> Option<PathBuf> {
        #[cfg(target_os = "linux")]
        {
            let config_dir = std::env::var("XDG_CONFIG_HOME")
                .map(PathBuf::from)
                .unwrap_or_else(|_| {
                    let home = std::env::var("HOME").unwrap_or_else(|_| "~".to_string());
                    PathBuf::from(home).join(".config")
                })
                .join("paintfe");
            let _ = std::fs::create_dir_all(&config_dir);
            Some(config_dir.join("paintfe_settings.cfg"))
        }
        #[cfg(target_os = "windows")]
        {
            // Use %APPDATA% so the settings are stored in the user profile and isolated
            // from other users — avoids the security issue of a world-writable EXE directory.
            let appdata = std::env::var("APPDATA")
                .or_else(|_| std::env::var("USERPROFILE"))
                .unwrap_or_else(|_| {
                    std::env::current_exe()
                        .ok()
                        .and_then(|p| p.parent().map(|d| d.to_string_lossy().into_owned()))
                        .unwrap_or_default()
                });
            let config_dir = PathBuf::from(appdata).join("PaintFE");
            let _ = std::fs::create_dir_all(&config_dir);
            Some(config_dir.join("paintfe_settings.cfg"))
        }
        #[cfg(target_os = "macos")]
        {
            let home = std::env::var("HOME").unwrap_or_else(|_| "~".to_string());
            let config_dir = PathBuf::from(home)
                .join("Library")
                .join("Application Support")
                .join("PaintFE");
            let _ = std::fs::create_dir_all(&config_dir);
            Some(config_dir.join("paintfe_settings.cfg"))
        }
        #[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "macos")))]
        {
            std::env::current_exe()
                .ok()
                .and_then(|p| p.parent().map(|d| d.join("paintfe_settings.cfg")))
        }
    }

    /// Serialize a Color32 as "r,g,b,a"
    fn color_to_str(c: Color32) -> String {
        format!("{},{},{},{}", c.r(), c.g(), c.b(), c.a())
    }

    /// Parse a Color32 from "r,g,b,a"
    fn str_to_color(s: &str) -> Option<Color32> {
        let parts: Vec<&str> = s.split(',').collect();
        if parts.len() == 4 {
            let r = parts[0].trim().parse::<u8>().ok()?;
            let g = parts[1].trim().parse::<u8>().ok()?;
            let b = parts[2].trim().parse::<u8>().ok()?;
            let a = parts[3].trim().parse::<u8>().ok()?;
            Some(Color32::from_rgba_unmultiplied(r, g, b, a))
        } else {
            None
        }
    }

    /// Serialize an optional Color32 as "r,g,b,a" or empty string.
    fn opt_color_to_str(c: Option<Color32>) -> String {
        match c {
            Some(c) => Self::color_to_str(c),
            None => String::new(),
        }
    }

    /// Deserialize an optional Color32 from "r,g,b,a" (empty/invalid → None).
    fn str_to_opt_color(s: &str) -> Option<Color32> {
        if s.is_empty() {
            None
        } else {
            Self::str_to_color(s)
        }
    }

    fn opt_pair_to_str(v: Option<(f32, f32)>) -> String {
        match v {
            Some((a, b)) => format!("{a},{b}"),
            None => String::new(),
        }
    }

    fn str_to_opt_pair(s: &str) -> Option<(f32, f32)> {
        let (a, b) = s.split_once(',')?;
        Some((a.trim().parse().ok()?, b.trim().parse().ok()?))
    }

    /// Build a `ThemeOverrides` from the current settings.
    /// Returns `ThemeOverrides::empty()` when `advanced_customization` is off.
    pub fn build_theme_overrides(&self) -> ThemeOverrides {
        if !self.advanced_customization {
            return ThemeOverrides::empty();
        }
        ThemeOverrides {
            bg_color: self.ov_bg_color,
            panel_bg: self.ov_panel_bg,
            window_bg: self.ov_window_bg,
            bg2: self.ov_bg2,
            bg3: self.ov_bg3,
            text_color: self.ov_text_color,
            text_muted: self.ov_text_muted,
            text_faint: self.ov_text_faint,
            border_color: self.ov_border_color,
            border_lit: self.ov_border_lit,
            separator_color: self.ov_separator_color,
            button_bg: self.ov_button_bg,
            button_hover: self.ov_button_hover,
            button_active: self.ov_button_active,
            floating_window_bg: self.ov_floating_window_bg,
            tool_shelf_bg: self.ov_tool_shelf_bg,
            tool_shelf_border: self.ov_tool_shelf_border,
            toolbar_bg: self.ov_toolbar_bg,
            menu_bg: self.ov_menu_bg,
            icon_button_bg: self.ov_icon_button_bg,
            icon_button_active: self.ov_icon_button_active,
            icon_button_disabled: self.ov_icon_button_disabled,
            text_input_bg: self.ov_text_input_bg,
            stepper_button_bg: self.ov_stepper_button_bg,
            canvas_bg_top: self.ov_canvas_bg_top,
            canvas_bg_bottom: self.ov_canvas_bg_bottom,
            glow_accent: self.ov_glow_accent,
            accent3: self.ov_accent3,
            accent4: self.ov_accent4,
            widget_rounding: self.widget_rounding,
            window_rounding: self.window_rounding,
            menu_rounding: self.menu_rounding,
            tool_shelf_rounding: self.tool_shelf_rounding,
            tool_button_rounding: self.tool_button_rounding,
            badge_rounding: self.badge_rounding,
            tab_rounding: self.tab_rounding,
            glow_intensity: Some(self.glow_intensity),
            shadow_strength: Some(self.shadow_strength),
        }
    }

    /// Export all theme-related settings as a shareable plain-text string.
    pub fn export_theme_to_string(&self) -> String {
        let effective_accent = if self.theme_preset == ThemePreset::Custom {
            self.custom_accent
        } else {
            self.theme_preset.accent_colors()
        };
        let mode_str = match self.theme_mode {
            ThemeMode::Light => "light",
            ThemeMode::Dark => "dark",
        };
        let preset_str = match self.theme_preset {
            ThemePreset::Blue => "blue",
            ThemePreset::Orange => "orange",
            ThemePreset::Purple => "purple",
            ThemePreset::Red => "red",
            ThemePreset::Green => "green",
            ThemePreset::Lime => "lime",
            ThemePreset::Nebula => "nebula",
            ThemePreset::Ember => "ember",
            ThemePreset::Sakura => "sakura",
            ThemePreset::Glacier => "glacier",
            ThemePreset::Midnight => "midnight",
            ThemePreset::Signal => "signal",
            ThemePreset::Custom => "custom",
        };
        let density_str = match self.ui_density {
            UiDensity::Compact => "compact",
            UiDensity::Normal => "normal",
            UiDensity::Spacious => "spacious",
        };
        let mut content = format!(
            "# PaintFE Theme\n\
             theme_mode={mode_str}\n\
             theme_preset={preset_str}\n\
             accent_light_normal={}\n\
             accent_light_faint={}\n\
             accent_light_strong={}\n\
             accent_dark_normal={}\n\
             accent_dark_faint={}\n\
             accent_dark_strong={}\n\
             neon_mode={}\n\
             advanced_customization={}\n\
             ui_density={density_str}\n\
             glow_intensity={}\n\
             shadow_strength={}\n",
            Self::color_to_str(effective_accent.light_normal),
            Self::color_to_str(effective_accent.light_faint),
            Self::color_to_str(effective_accent.light_strong),
            Self::color_to_str(effective_accent.dark_normal),
            Self::color_to_str(effective_accent.dark_faint),
            Self::color_to_str(effective_accent.dark_strong),
            self.neon_mode,
            self.advanced_customization,
            self.glow_intensity,
            self.shadow_strength,
        );
        if let Some(v) = self.widget_rounding {
            content.push_str(&format!("widget_rounding={v}\n"));
        }
        if let Some(v) = self.window_rounding {
            content.push_str(&format!("window_rounding={v}\n"));
        }
        if let Some(v) = self.menu_rounding {
            content.push_str(&format!("menu_rounding={v}\n"));
        }
        if let Some(v) = self.tool_shelf_rounding {
            content.push_str(&format!("tool_shelf_rounding={v}\n"));
        }
        if let Some(v) = self.tool_button_rounding {
            content.push_str(&format!("tool_button_rounding={v}\n"));
        }
        let ov_fields: &[(&str, Option<Color32>)] = &[
            ("ov_bg_color", self.ov_bg_color),
            ("ov_panel_bg", self.ov_panel_bg),
            ("ov_window_bg", self.ov_window_bg),
            ("ov_bg2", self.ov_bg2),
            ("ov_bg3", self.ov_bg3),
            ("ov_text_color", self.ov_text_color),
            ("ov_text_muted", self.ov_text_muted),
            ("ov_text_faint", self.ov_text_faint),
            ("ov_border_color", self.ov_border_color),
            ("ov_border_lit", self.ov_border_lit),
            ("ov_separator_color", self.ov_separator_color),
            ("ov_button_bg", self.ov_button_bg),
            ("ov_button_hover", self.ov_button_hover),
            ("ov_button_active", self.ov_button_active),
            ("ov_floating_window_bg", self.ov_floating_window_bg),
            ("ov_tool_shelf_bg", self.ov_tool_shelf_bg),
            ("ov_tool_shelf_border", self.ov_tool_shelf_border),
            ("ov_toolbar_bg", self.ov_toolbar_bg),
            ("ov_menu_bg", self.ov_menu_bg),
            ("ov_icon_button_bg", self.ov_icon_button_bg),
            ("ov_icon_button_active", self.ov_icon_button_active),
            ("ov_icon_button_disabled", self.ov_icon_button_disabled),
            ("ov_text_input_bg", self.ov_text_input_bg),
            ("ov_stepper_button_bg", self.ov_stepper_button_bg),
            ("ov_canvas_bg_top", self.ov_canvas_bg_top),
            ("ov_canvas_bg_bottom", self.ov_canvas_bg_bottom),
            ("ov_glow_accent", self.ov_glow_accent),
            ("ov_accent3", self.ov_accent3),
            ("ov_accent4", self.ov_accent4),
        ];
        for (key, val) in ov_fields {
            if let Some(c) = val {
                content.push_str(&format!("{}={}\n", key, Self::color_to_str(*c)));
            }
        }
        content
    }

    /// Import theme-related settings from a plain-text string.
    /// Only updates theme fields; non-theme settings are untouched.
    pub fn import_theme_from_string(&mut self, content: &str) {
        let mut map = std::collections::HashMap::new();
        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if let Some((key, val)) = line.split_once('=') {
                map.insert(key.trim().to_string(), val.trim().to_string());
            }
        }
        if let Some(v) = map.get("theme_mode") {
            self.theme_mode = match v.as_str() {
                "light" => ThemeMode::Light,
                "dark" => ThemeMode::Dark,
                _ => self.theme_mode,
            };
        }
        if let Some(v) = map.get("theme_preset") {
            self.theme_preset = match v.as_str() {
                "blue" => ThemePreset::Blue,
                "orange" => ThemePreset::Orange,
                "purple" => ThemePreset::Purple,
                "red" => ThemePreset::Red,
                "green" => ThemePreset::Green,
                "lime" => ThemePreset::Lime,
                "nebula" => ThemePreset::Nebula,
                "ember" => ThemePreset::Ember,
                "sakura" => ThemePreset::Sakura,
                "glacier" => ThemePreset::Glacier,
                "midnight" => ThemePreset::Midnight,
                "signal" => ThemePreset::Signal,
                "custom" => ThemePreset::Custom,
                _ => self.theme_preset,
            };
        }
        if let Some(v) = map.get("accent_light_normal")
            && let Some(c) = Self::str_to_color(v)
        {
            self.custom_accent.light_normal = c;
        }
        if let Some(v) = map.get("accent_light_faint")
            && let Some(c) = Self::str_to_color(v)
        {
            self.custom_accent.light_faint = c;
        }
        if let Some(v) = map.get("accent_light_strong")
            && let Some(c) = Self::str_to_color(v)
        {
            self.custom_accent.light_strong = c;
        }
        if let Some(v) = map.get("accent_dark_normal")
            && let Some(c) = Self::str_to_color(v)
        {
            self.custom_accent.dark_normal = c;
        }
        if let Some(v) = map.get("accent_dark_faint")
            && let Some(c) = Self::str_to_color(v)
        {
            self.custom_accent.dark_faint = c;
        }
        if let Some(v) = map.get("accent_dark_strong")
            && let Some(c) = Self::str_to_color(v)
        {
            self.custom_accent.dark_strong = c;
        }
        if let Some(v) = map.get("neon_mode") {
            self.neon_mode = v == "true";
        }
        if let Some(v) = map.get("advanced_customization") {
            self.advanced_customization = v == "true";
        }
        if let Some(v) = map.get("ui_density") {
            self.ui_density = match v.as_str() {
                "compact" => UiDensity::Compact,
                "normal" => UiDensity::Normal,
                "spacious" => UiDensity::Spacious,
                _ => self.ui_density,
            };
        }
        if let Some(v) = map.get("glow_intensity")
            && let Ok(f) = v.parse::<f32>()
        {
            self.glow_intensity = f;
        }
        if let Some(v) = map.get("shadow_strength")
            && let Ok(f) = v.parse::<f32>()
        {
            self.shadow_strength = f;
        }
        if let Some(v) = map.get("widget_rounding") {
            self.widget_rounding = v.parse::<f32>().ok();
        }
        if let Some(v) = map.get("window_rounding") {
            self.window_rounding = v.parse::<f32>().ok();
        }
        if let Some(v) = map.get("menu_rounding") {
            self.menu_rounding = v.parse::<f32>().ok();
        }
        if let Some(v) = map.get("tool_shelf_rounding") {
            self.tool_shelf_rounding = v.parse::<f32>().ok();
        }
        if let Some(v) = map.get("tool_button_rounding") {
            self.tool_button_rounding = v.parse::<f32>().ok();
        }
        macro_rules! import_ov {
            ($key:expr, $field:ident) => {
                if let Some(v) = map.get($key) {
                    self.$field = Self::str_to_color(v);
                }
            };
        }
        import_ov!("ov_bg_color", ov_bg_color);
        import_ov!("ov_panel_bg", ov_panel_bg);
        import_ov!("ov_window_bg", ov_window_bg);
        import_ov!("ov_bg2", ov_bg2);
        import_ov!("ov_bg3", ov_bg3);
        import_ov!("ov_text_color", ov_text_color);
        import_ov!("ov_text_muted", ov_text_muted);
        import_ov!("ov_text_faint", ov_text_faint);
        import_ov!("ov_border_color", ov_border_color);
        import_ov!("ov_border_lit", ov_border_lit);
        import_ov!("ov_separator_color", ov_separator_color);
        import_ov!("ov_button_bg", ov_button_bg);
        import_ov!("ov_button_hover", ov_button_hover);
        import_ov!("ov_button_active", ov_button_active);
        import_ov!("ov_floating_window_bg", ov_floating_window_bg);
        import_ov!("ov_tool_shelf_bg", ov_tool_shelf_bg);
        import_ov!("ov_tool_shelf_border", ov_tool_shelf_border);
        import_ov!("ov_toolbar_bg", ov_toolbar_bg);
        import_ov!("ov_menu_bg", ov_menu_bg);
        import_ov!("ov_icon_button_bg", ov_icon_button_bg);
        import_ov!("ov_icon_button_active", ov_icon_button_active);
        import_ov!("ov_icon_button_disabled", ov_icon_button_disabled);
        import_ov!("ov_text_input_bg", ov_text_input_bg);
        import_ov!("ov_stepper_button_bg", ov_stepper_button_bg);
        import_ov!("ov_canvas_bg_top", ov_canvas_bg_top);
        import_ov!("ov_canvas_bg_bottom", ov_canvas_bg_bottom);
        import_ov!("ov_glow_accent", ov_glow_accent);
        import_ov!("ov_accent3", ov_accent3);
        import_ov!("ov_accent4", ov_accent4);

        if self.theme_preset != ThemePreset::Custom {
            self.custom_accent = self.theme_preset.accent_colors();
        }
    }

    /// Save settings to disk
    pub fn save(&self) {
        #[cfg(not(target_arch = "wasm32"))]
        let Some(path) = Self::settings_path() else {
            return;
        };
        let effective_accent = if self.theme_preset == ThemePreset::Custom {
            self.custom_accent
        } else {
            self.theme_preset.accent_colors()
        };
        let mode_str = match self.theme_mode {
            ThemeMode::Light => "light",
            ThemeMode::Dark => "dark",
        };
        let preset_str = match self.theme_preset {
            ThemePreset::Blue => "blue",
            ThemePreset::Orange => "orange",
            ThemePreset::Purple => "purple",
            ThemePreset::Red => "red",
            ThemePreset::Green => "green",
            ThemePreset::Lime => "lime",
            ThemePreset::Nebula => "nebula",
            ThemePreset::Ember => "ember",
            ThemePreset::Sakura => "sakura",
            ThemePreset::Glacier => "glacier",
            ThemePreset::Midnight => "midnight",
            ThemePreset::Signal => "signal",
            ThemePreset::Custom => "custom",
        };
        let grid_str = match self.pixel_grid_mode {
            PixelGridMode::Auto => "auto",
            PixelGridMode::AlwaysOn => "on",
            PixelGridMode::AlwaysOff => "off",
        };
        let filter_str = match self.zoom_filter_mode {
            ZoomFilterMode::Linear => "linear",
            ZoomFilterMode::Nearest => "nearest",
        };
        let content = format!(
            "theme_mode={mode_str}\n\
             theme_preset={preset_str}\n\
             preferred_gpu={}\n\
             pixel_grid_mode={grid_str}\n\
             pixel_grid_outline_color={}\n\
             pixel_grid_center_color={}\n\
             selection_stripe_color={}\n\
             selection_stripe_alpha={}\n\
             max_undo_steps={}\n\
             auto_save_minutes={}\n\
             accent_light_normal={}\n\
             accent_light_faint={}\n\
             accent_light_strong={}\n\
             accent_dark_normal={}\n\
             accent_dark_faint={}\n\
             accent_dark_strong={}\n\
             neon_mode={}\n\
             zoom_filter_mode={filter_str}\n\
             checkerboard_brightness={}\n\
             onnx_runtime_path={}\n\
             birefnet_model_path={}\n\
             paintdotnet_plugins_enabled={}\n\
             language={}\n\
             default_canvas_width={}\n\
             default_canvas_height={}\n\
             create_canvas_on_startup={}\n\
             confirm_on_exit={}\n",
            self.preferred_gpu,
            Self::color_to_str(self.pixel_grid_outline_color),
            Self::color_to_str(self.pixel_grid_center_color),
            Self::color_to_str(self.selection_stripe_color),
            self.selection_stripe_alpha,
            self.max_undo_steps,
            self.auto_save_minutes,
            Self::color_to_str(effective_accent.light_normal),
            Self::color_to_str(effective_accent.light_faint),
            Self::color_to_str(effective_accent.light_strong),
            Self::color_to_str(effective_accent.dark_normal),
            Self::color_to_str(effective_accent.dark_faint),
            Self::color_to_str(effective_accent.dark_strong),
            self.neon_mode,
            self.checkerboard_brightness,
            self.onnx_runtime_path,
            self.birefnet_model_path,
            self.paintdotnet_plugins_enabled,
            self.language,
            self.default_canvas_width,
            self.default_canvas_height,
            self.create_canvas_on_startup,
            self.confirm_on_exit,
        );
        // Append keybinding lines
        let mut content = content;
        content.push_str(&format!(
            "persist_window_width={}\n",
            self.persist_window_width
        ));
        content.push_str(&format!(
            "persist_window_height={}\n",
            self.persist_window_height
        ));
        content.push_str(&format!(
            "persist_window_pos={}\n",
            Self::opt_pair_to_str(self.persist_window_pos)
        ));
        content.push_str(&format!(
            "persist_tools_visible={}\n",
            self.persist_tools_visible
        ));
        content.push_str(&format!(
            "persist_layers_visible={}\n",
            self.persist_layers_visible
        ));
        content.push_str(&format!(
            "persist_history_visible={}\n",
            self.persist_history_visible
        ));
        content.push_str(&format!(
            "persist_colors_visible={}\n",
            self.persist_colors_visible
        ));
        content.push_str(&format!(
            "persist_palette_visible={}\n",
            self.persist_palette_visible
        ));
        content.push_str(&format!(
            "persist_script_editor_visible={}\n",
            self.persist_script_editor_visible
        ));
        content.push_str(&format!(
            "persist_tools_panel_pos={}\n",
            Self::opt_pair_to_str(self.persist_tools_panel_pos)
        ));
        content.push_str(&format!(
            "persist_layers_panel_right_offset={}\n",
            Self::opt_pair_to_str(self.persist_layers_panel_right_offset)
        ));
        content.push_str(&format!(
            "persist_layers_panel_size={}\n",
            Self::opt_pair_to_str(self.persist_layers_panel_size)
        ));
        content.push_str(&format!(
            "persist_history_panel_right_offset={}\n",
            Self::opt_pair_to_str(self.persist_history_panel_right_offset)
        ));
        content.push_str(&format!(
            "persist_history_panel_size={}\n",
            Self::opt_pair_to_str(self.persist_history_panel_size)
        ));
        content.push_str(&format!(
            "persist_colors_panel_left_offset={}\n",
            Self::opt_pair_to_str(self.persist_colors_panel_left_offset)
        ));
        content.push_str(&format!(
            "persist_palette_panel_pos={}\n",
            Self::opt_pair_to_str(self.persist_palette_panel_pos)
        ));
        content.push_str(&format!(
            "persist_palette_panel_left_offset={}\n",
            Self::opt_pair_to_str(self.persist_palette_panel_left_offset)
        ));
        content.push_str(&format!(
            "persist_palette_panel_right_offset={}\n",
            Self::opt_pair_to_str(self.persist_palette_panel_right_offset)
        ));
        content.push_str(&format!(
            "persist_palette_recent_colors={}\n",
            self.persist_palette_recent_colors
        ));
        content.push_str(&format!(
            "persist_script_right_offset={}\n",
            Self::opt_pair_to_str(self.persist_script_right_offset)
        ));
        content.push_str(&format!(
            "persist_colors_panel_expanded={}\n",
            self.persist_colors_panel_expanded
        ));
        content.push_str(&format!(
            "persist_new_file_lock_aspect={}\n",
            self.persist_new_file_lock_aspect
        ));
        content.push_str(&format!(
            "persist_resize_lock_aspect={}\n",
            self.persist_resize_lock_aspect
        ));
        content.push_str(&format!(
            "persisted_active_tool={}\n",
            self.persisted_active_tool
        ));
        content.push_str(&format!(
            "persisted_brush_size={}\n",
            self.persisted_brush_size
        ));
        content.push_str(&format!(
            "persisted_brush_hardness={}\n",
            self.persisted_brush_hardness
        ));
        content.push_str(&format!(
            "persisted_brush_flow={}\n",
            self.persisted_brush_flow
        ));
        content.push_str(&format!(
            "persisted_brush_spacing={}\n",
            self.persisted_brush_spacing
        ));
        content.push_str(&format!(
            "persisted_brush_scatter={}\n",
            self.persisted_brush_scatter
        ));
        content.push_str(&format!(
            "persisted_brush_hue_jitter={}\n",
            self.persisted_brush_hue_jitter
        ));
        content.push_str(&format!(
            "persisted_brush_brightness_jitter={}\n",
            self.persisted_brush_brightness_jitter
        ));
        content.push_str(&format!(
            "persisted_brush_anti_aliased={}\n",
            self.persisted_brush_anti_aliased
        ));
        content.push_str(&format!(
            "persisted_pressure_size={}\n",
            self.persisted_pressure_size
        ));
        content.push_str(&format!(
            "persisted_pressure_opacity={}\n",
            self.persisted_pressure_opacity
        ));
        content.push_str(&format!(
            "persisted_pressure_min_size={}\n",
            self.persisted_pressure_min_size
        ));
        content.push_str(&format!(
            "persisted_pressure_min_opacity={}\n",
            self.persisted_pressure_min_opacity
        ));
        content.push_str(&format!(
            "persisted_brush_mode={}\n",
            self.persisted_brush_mode
        ));
        content.push_str(&format!(
            "persisted_brush_tip={}\n",
            self.persisted_brush_tip
        ));
        // Save custom brush tips (name, category, base64 png)
        for (name, cat, b64) in &self.custom_brush_tips {
            content.push_str(&format!("custom_brush_tip={},{},{}\n", name, cat, b64));
        }
        for (name, cat, b64) in &self.custom_shapes {
            content.push_str(&format!("custom_shape={},{},{}\n", name, cat, b64));
        }
        content.push_str(&format!(
            "persisted_fill_tolerance={}\n",
            self.persisted_fill_tolerance
        ));
        content.push_str(&format!(
            "persisted_fill_anti_aliased={}\n",
            self.persisted_fill_anti_aliased
        ));
        content.push_str(&format!(
            "persisted_fill_global={}\n",
            self.persisted_fill_global
        ));
        content.push_str(&format!(
            "persisted_wand_tolerance={}\n",
            self.persisted_wand_tolerance
        ));
        content.push_str(&format!(
            "persisted_wand_anti_aliased={}\n",
            self.persisted_wand_anti_aliased
        ));
        content.push_str(&format!(
            "persisted_wand_global={}\n",
            self.persisted_wand_global
        ));
        content.push_str(&format!(
            "persisted_color_remover_tolerance={}\n",
            self.persisted_color_remover_tolerance
        ));
        content.push_str(&format!(
            "persisted_color_remover_smoothness={}\n",
            self.persisted_color_remover_smoothness
        ));
        content.push_str(&format!(
            "persisted_color_remover_contiguous={}\n",
            self.persisted_color_remover_contiguous
        ));
        content.push_str(&format!(
            "persisted_smudge_strength={}\n",
            self.persisted_smudge_strength
        ));
        content.push_str(&format!(
            "persisted_shapes_fill_mode={}\n",
            self.persisted_shapes_fill_mode
        ));
        content.push_str(&format!(
            "persisted_shapes_anti_alias={}\n",
            self.persisted_shapes_anti_alias
        ));
        content.push_str(&format!(
            "persisted_shapes_corner_radius={}\n",
            self.persisted_shapes_corner_radius
        ));
        content.push_str(&format!(
            "persisted_move_interpolation={}\n",
            self.persisted_move_interpolation
        ));
        content.push_str(&format!(
            "persisted_move_anti_aliasing={}\n",
            self.persisted_move_anti_aliasing
        ));
        content.push_str(&format!(
            "persisted_text_font_family={}\n",
            self.persisted_text_font_family
        ));
        content.push_str(&format!(
            "clipboard_copy_transparent_cutout={}\n",
            self.clipboard_copy_transparent_cutout
        ));
        content.push_str(&format!(
            "persist_window_maximized={}\n",
            self.persist_window_maximized
        ));
        for (idx, color) in self.folder_color_palette.iter().enumerate() {
            content.push_str(&format!(
                "folder_color_{}={}\n",
                idx,
                Self::color_to_str(*color)
            ));
        }
        for line in self.keybindings.to_config_lines() {
            content.push_str(&line);
            content.push('\n');
        }
        // Append advanced customization settings
        let density_str = match self.ui_density {
            UiDensity::Compact => "compact",
            UiDensity::Normal => "normal",
            UiDensity::Spacious => "spacious",
        };
        content.push_str(&format!(
            "advanced_customization={}\n",
            self.advanced_customization
        ));
        content.push_str(&format!("ui_density={density_str}\n"));
        content.push_str(&format!(
            "canvas_grid_visible={}\n",
            self.canvas_grid_visible
        ));
        content.push_str(&format!(
            "canvas_grid_opacity={}\n",
            self.canvas_grid_opacity
        ));
        content.push_str(&format!("glow_intensity={}\n", self.glow_intensity));
        content.push_str(&format!("shadow_strength={}\n", self.shadow_strength));
        if let Some(v) = self.widget_rounding {
            content.push_str(&format!("widget_rounding={v}\n"));
        }
        if let Some(v) = self.window_rounding {
            content.push_str(&format!("window_rounding={v}\n"));
        }
        if let Some(v) = self.menu_rounding {
            content.push_str(&format!("menu_rounding={v}\n"));
        }
        if let Some(v) = self.tool_shelf_rounding {
            content.push_str(&format!("tool_shelf_rounding={v}\n"));
        }
        if let Some(v) = self.tool_button_rounding {
            content.push_str(&format!("tool_button_rounding={v}\n"));
        }
        if let Some(v) = self.badge_rounding {
            content.push_str(&format!("badge_rounding={v}\n"));
        }
        if let Some(v) = self.tab_rounding {
            content.push_str(&format!("tab_rounding={v}\n"));
        }
        // Color overrides — skip lines for None values (saves space)
        let ov_fields: &[(&str, Option<Color32>)] = &[
            ("ov_bg_color", self.ov_bg_color),
            ("ov_panel_bg", self.ov_panel_bg),
            ("ov_window_bg", self.ov_window_bg),
            ("ov_bg2", self.ov_bg2),
            ("ov_bg3", self.ov_bg3),
            ("ov_text_color", self.ov_text_color),
            ("ov_text_muted", self.ov_text_muted),
            ("ov_text_faint", self.ov_text_faint),
            ("ov_border_color", self.ov_border_color),
            ("ov_border_lit", self.ov_border_lit),
            ("ov_separator_color", self.ov_separator_color),
            ("ov_button_bg", self.ov_button_bg),
            ("ov_button_hover", self.ov_button_hover),
            ("ov_button_active", self.ov_button_active),
            ("ov_floating_window_bg", self.ov_floating_window_bg),
            ("ov_tool_shelf_bg", self.ov_tool_shelf_bg),
            ("ov_tool_shelf_border", self.ov_tool_shelf_border),
            ("ov_toolbar_bg", self.ov_toolbar_bg),
            ("ov_menu_bg", self.ov_menu_bg),
            ("ov_icon_button_bg", self.ov_icon_button_bg),
            ("ov_icon_button_active", self.ov_icon_button_active),
            ("ov_icon_button_disabled", self.ov_icon_button_disabled),
            ("ov_text_input_bg", self.ov_text_input_bg),
            ("ov_stepper_button_bg", self.ov_stepper_button_bg),
            ("ov_canvas_bg_top", self.ov_canvas_bg_top),
            ("ov_canvas_bg_bottom", self.ov_canvas_bg_bottom),
            ("ov_glow_accent", self.ov_glow_accent),
            ("ov_accent3", self.ov_accent3),
            ("ov_accent4", self.ov_accent4),
        ];
        for (key, val) in ov_fields {
            if let Some(c) = val {
                content.push_str(&format!("{}={}\n", key, Self::color_to_str(*c)));
            }
        }
        #[cfg(not(target_arch = "wasm32"))]
        if let Err(e) = std::fs::write(&path, &content) {
            eprintln!(
                "[PaintFE] Failed to save settings to {}: {}",
                path.display(),
                e
            );
        }
        #[cfg(target_arch = "wasm32")]
        if let Err(error) = crate::web_storage::save_settings(&content) {
            web_sys::console::warn_1(&format!("PaintFE: {error}").into());
        }
    }

    /// Load settings from disk (returns default if file missing or corrupt)
    pub fn load() -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        let content = {
            let Some(path) = Self::settings_path() else {
                return Self::default();
            };
            let Ok(content) = std::fs::read_to_string(&path) else {
                return Self::default();
            };
            content
        };
        #[cfg(target_arch = "wasm32")]
        let Some(content) = crate::web_storage::load_settings() else {
            return Self::default();
        };

        let mut s = Self::default();
        for line in content.lines() {
            let Some((key, val)) = line.split_once('=') else {
                continue;
            };
            let key = key.trim();
            let val = val.trim();
            match key {
                "theme_mode" => {
                    s.theme_mode = match val {
                        "dark" => ThemeMode::Dark,
                        _ => ThemeMode::Light,
                    };
                }
                "theme_preset" => {
                    s.theme_preset = match val {
                        "blue" => ThemePreset::Blue,
                        "orange" => ThemePreset::Orange,
                        "purple" => ThemePreset::Purple,
                        "red" => ThemePreset::Red,
                        "green" => ThemePreset::Green,
                        "lime" => ThemePreset::Lime,
                        "nebula" => ThemePreset::Nebula,
                        "ember" => ThemePreset::Ember,
                        "sakura" => ThemePreset::Sakura,
                        "glacier" => ThemePreset::Glacier,
                        "midnight" => ThemePreset::Midnight,
                        "signal" => ThemePreset::Signal,
                        "custom" => ThemePreset::Custom,
                        _ => ThemePreset::Blue,
                    };
                }
                "gpu_acceleration" => {}
                "preferred_gpu" => {
                    s.preferred_gpu = val.to_string();
                }
                "pixel_grid_mode" => {
                    s.pixel_grid_mode = match val {
                        "on" => PixelGridMode::AlwaysOn,
                        "off" => PixelGridMode::AlwaysOff,
                        _ => PixelGridMode::Auto,
                    };
                }
                "pixel_grid_outline_color" => {
                    s.pixel_grid_outline_color =
                        Self::str_to_color(val).unwrap_or(Color32::from_black_alpha(90));
                }
                "pixel_grid_center_color" => {
                    s.pixel_grid_center_color =
                        Self::str_to_color(val).unwrap_or(Color32::from_white_alpha(100));
                }
                "selection_stripe_color" => {
                    s.selection_stripe_color = Self::str_to_color(val)
                        .unwrap_or(Color32::from_rgba_premultiplied(255, 255, 255, 255));
                }
                "selection_stripe_alpha" => {
                    s.selection_stripe_alpha = val.parse().unwrap_or(22);
                }
                key if key.starts_with("folder_color_") => {
                    if let Ok(idx) = key.trim_start_matches("folder_color_").parse::<usize>()
                        && idx < s.folder_color_palette.len()
                        && let Some(color) = Self::str_to_color(val)
                    {
                        s.folder_color_palette[idx] = color;
                    }
                }
                "max_undo_steps" => {
                    s.max_undo_steps = val.parse().unwrap_or(50);
                }
                "auto_save_minutes" => {
                    s.auto_save_minutes = val.parse().unwrap_or(0);
                }
                "accent_light_normal" => {
                    if let Some(c) = Self::str_to_color(val) {
                        s.custom_accent.light_normal = c;
                    }
                }
                "accent_light_faint" => {
                    if let Some(c) = Self::str_to_color(val) {
                        s.custom_accent.light_faint = c;
                    }
                }
                "accent_light_strong" => {
                    if let Some(c) = Self::str_to_color(val) {
                        s.custom_accent.light_strong = c;
                    }
                }
                "accent_dark_normal" => {
                    if let Some(c) = Self::str_to_color(val) {
                        s.custom_accent.dark_normal = c;
                    }
                }
                "accent_dark_faint" => {
                    if let Some(c) = Self::str_to_color(val) {
                        s.custom_accent.dark_faint = c;
                    }
                }
                "accent_dark_strong" => {
                    if let Some(c) = Self::str_to_color(val) {
                        s.custom_accent.dark_strong = c;
                    }
                }
                "neon_mode" => {
                    s.neon_mode = val == "true";
                }
                "zoom_filter_mode" => {
                    s.zoom_filter_mode = match val {
                        "nearest" => ZoomFilterMode::Nearest,
                        _ => ZoomFilterMode::Linear,
                    };
                }
                "checkerboard_brightness" => {
                    s.checkerboard_brightness = val.parse().unwrap_or(1.0);
                }
                "onnx_runtime_path" => {
                    s.onnx_runtime_path = val.to_string();
                }
                "birefnet_model_path" => {
                    s.birefnet_model_path = val.to_string();
                }
                "paintdotnet_plugins_enabled" => {
                    s.paintdotnet_plugins_enabled = val == "true";
                }
                "language" => {
                    s.language = val.to_string();
                }
                "confirm_on_exit" => {
                    s.confirm_on_exit = val == "true";
                }
                "persist_window_width" => {
                    s.persist_window_width = val.parse().unwrap_or(1280.0);
                }
                "persist_window_height" => {
                    s.persist_window_height = val.parse().unwrap_or(720.0);
                }
                "persist_window_pos" => {
                    s.persist_window_pos = Self::str_to_opt_pair(val);
                }
                "persist_tools_visible" => {
                    s.persist_tools_visible = val == "true";
                }
                "persist_layers_visible" => {
                    s.persist_layers_visible = val == "true";
                }
                "persist_history_visible" => {
                    s.persist_history_visible = val == "true";
                }
                "persist_colors_visible" => {
                    s.persist_colors_visible = val == "true";
                }
                "persist_palette_visible" => {
                    s.persist_palette_visible = val == "true";
                }
                "persist_script_editor_visible" => {
                    s.persist_script_editor_visible = val == "true";
                }
                "persist_tools_panel_pos" => {
                    s.persist_tools_panel_pos = Self::str_to_opt_pair(val);
                }
                "persist_layers_panel_right_offset" => {
                    s.persist_layers_panel_right_offset = Self::str_to_opt_pair(val);
                }
                "persist_layers_panel_size" => {
                    s.persist_layers_panel_size = Self::str_to_opt_pair(val);
                }
                "persist_history_panel_right_offset" => {
                    s.persist_history_panel_right_offset = Self::str_to_opt_pair(val);
                }
                "persist_history_panel_size" => {
                    s.persist_history_panel_size = Self::str_to_opt_pair(val);
                }
                "persist_colors_panel_left_offset" => {
                    s.persist_colors_panel_left_offset = Self::str_to_opt_pair(val);
                }
                "persist_palette_panel_pos" => {
                    s.persist_palette_panel_pos = Self::str_to_opt_pair(val);
                }
                "persist_palette_panel_left_offset" => {
                    s.persist_palette_panel_left_offset = Self::str_to_opt_pair(val);
                }
                "persist_palette_panel_right_offset" => {
                    s.persist_palette_panel_right_offset = Self::str_to_opt_pair(val);
                }
                "persist_palette_recent_colors" => {
                    s.persist_palette_recent_colors = val.to_string();
                }
                "persist_script_right_offset" => {
                    s.persist_script_right_offset = Self::str_to_opt_pair(val);
                }
                "persist_colors_panel_expanded" => {
                    s.persist_colors_panel_expanded = val == "true";
                }
                "persist_new_file_lock_aspect" => {
                    s.persist_new_file_lock_aspect = val == "true";
                }
                "persist_resize_lock_aspect" => {
                    s.persist_resize_lock_aspect = val == "true";
                }
                "persisted_active_tool" => {
                    s.persisted_active_tool = val.to_string();
                }
                "persisted_brush_size" => {
                    s.persisted_brush_size = val.parse().unwrap_or(10.0);
                }
                "persisted_brush_hardness" => {
                    s.persisted_brush_hardness = val.parse().unwrap_or(0.75);
                }
                "persisted_brush_flow" => {
                    s.persisted_brush_flow = val.parse().unwrap_or(1.0);
                }
                "persisted_brush_spacing" => {
                    s.persisted_brush_spacing = val.parse().unwrap_or(0.01);
                }
                "persisted_brush_scatter" => {
                    s.persisted_brush_scatter = val.parse().unwrap_or(0.0);
                }
                "persisted_brush_hue_jitter" => {
                    s.persisted_brush_hue_jitter = val.parse().unwrap_or(0.0);
                }
                "persisted_brush_brightness_jitter" => {
                    s.persisted_brush_brightness_jitter = val.parse().unwrap_or(0.0);
                }
                "persisted_brush_anti_aliased" => {
                    s.persisted_brush_anti_aliased = val == "true";
                }
                "persisted_pressure_size" => {
                    s.persisted_pressure_size = val == "true";
                }
                "persisted_pressure_opacity" => {
                    s.persisted_pressure_opacity = val == "true";
                }
                "persisted_pressure_min_size" => {
                    s.persisted_pressure_min_size = val.parse().unwrap_or(0.1);
                }
                "persisted_pressure_min_opacity" => {
                    s.persisted_pressure_min_opacity = val.parse().unwrap_or(0.1);
                }
                "persisted_brush_mode" => {
                    s.persisted_brush_mode = val.to_string();
                }
                "persisted_brush_tip" => {
                    s.persisted_brush_tip = val.to_string();
                }
                "custom_brush_tip" => {
                    // Format: name,category,base64data
                    if let Some((rest, b64)) = val.rsplit_once(',')
                        && let Some((name, cat)) = rest.split_once(',')
                    {
                        s.custom_brush_tips.push((
                            name.to_string(),
                            cat.to_string(),
                            b64.to_string(),
                        ));
                    }
                }
                "custom_shape" => {
                    if let Some((rest, b64)) = val.rsplit_once(',')
                        && let Some((name, cat)) = rest.split_once(',')
                    {
                        s.custom_shapes
                            .push((name.to_string(), cat.to_string(), b64.to_string()));
                    }
                }
                "persisted_fill_tolerance" => {
                    s.persisted_fill_tolerance = val.parse().unwrap_or(5.0);
                }
                "persisted_fill_anti_aliased" => {
                    s.persisted_fill_anti_aliased = val == "true";
                }
                "persisted_fill_global" => {
                    s.persisted_fill_global = val == "true";
                }
                "persisted_wand_tolerance" => {
                    s.persisted_wand_tolerance = val.parse().unwrap_or(5.0);
                }
                "persisted_wand_anti_aliased" => {
                    s.persisted_wand_anti_aliased = val == "true";
                }
                "persisted_wand_global" => {
                    s.persisted_wand_global = val == "true";
                }
                "persisted_color_remover_tolerance" => {
                    s.persisted_color_remover_tolerance = val.parse().unwrap_or(5.0);
                }
                "persisted_color_remover_smoothness" => {
                    s.persisted_color_remover_smoothness = val.parse().unwrap_or(3);
                }
                "persisted_color_remover_contiguous" => {
                    s.persisted_color_remover_contiguous = val == "true";
                }
                "persisted_smudge_strength" => {
                    s.persisted_smudge_strength = val.parse().unwrap_or(0.6);
                }
                "persisted_shapes_fill_mode" => {
                    s.persisted_shapes_fill_mode = val.to_string();
                }
                "persisted_shapes_anti_alias" => {
                    s.persisted_shapes_anti_alias = val == "true";
                }
                "persisted_shapes_corner_radius" => {
                    s.persisted_shapes_corner_radius = val.parse().unwrap_or(10.0);
                }
                "persisted_move_interpolation" => {
                    s.persisted_move_interpolation = val.to_string();
                }
                "persisted_move_anti_aliasing" => {
                    s.persisted_move_anti_aliasing = val == "true";
                }
                "persisted_text_font_family" => {
                    s.persisted_text_font_family = val.to_string();
                }
                "clipboard_copy_transparent_cutout" => {
                    s.clipboard_copy_transparent_cutout = val == "true";
                }
                "persist_window_maximized" => {
                    s.persist_window_maximized = val == "true";
                }
                "default_canvas_width" => {
                    s.default_canvas_width = val.parse().unwrap_or(800u32).clamp(1, 65535);
                }
                "default_canvas_height" => {
                    s.default_canvas_height = val.parse().unwrap_or(600u32).clamp(1, 65535);
                }
                "create_canvas_on_startup" => {
                    s.create_canvas_on_startup = val == "true";
                }
                // Advanced customization fields
                "advanced_customization" => {
                    s.advanced_customization = val == "true";
                }
                "ui_density" => {
                    s.ui_density = match val {
                        "compact" => UiDensity::Compact,
                        "spacious" => UiDensity::Spacious,
                        _ => UiDensity::Normal,
                    };
                }
                "canvas_grid_visible" => {
                    s.canvas_grid_visible = val == "true";
                }
                "canvas_grid_opacity" => {
                    s.canvas_grid_opacity = val.parse().unwrap_or(0.4);
                }
                "glow_intensity" => {
                    s.glow_intensity = val.parse().unwrap_or(1.0);
                }
                "shadow_strength" => {
                    s.shadow_strength = val.parse().unwrap_or(1.0);
                }
                "widget_rounding" => {
                    s.widget_rounding = val.parse().ok();
                }
                "window_rounding" => {
                    s.window_rounding = val.parse().ok();
                }
                "menu_rounding" => {
                    s.menu_rounding = val.parse().ok();
                }
                "tool_shelf_rounding" => {
                    s.tool_shelf_rounding = val.parse().ok();
                }
                "tool_button_rounding" => {
                    s.tool_button_rounding = val.parse().ok();
                }
                "badge_rounding" => {
                    s.badge_rounding = val.parse().ok();
                }
                "tab_rounding" => {
                    s.tab_rounding = val.parse().ok();
                }
                // Color overrides
                "ov_bg_color" => {
                    s.ov_bg_color = Self::str_to_opt_color(val);
                }
                "ov_panel_bg" => {
                    s.ov_panel_bg = Self::str_to_opt_color(val);
                }
                "ov_window_bg" => {
                    s.ov_window_bg = Self::str_to_opt_color(val);
                }
                "ov_bg2" => {
                    s.ov_bg2 = Self::str_to_opt_color(val);
                }
                "ov_bg3" => {
                    s.ov_bg3 = Self::str_to_opt_color(val);
                }
                "ov_text_color" => {
                    s.ov_text_color = Self::str_to_opt_color(val);
                }
                "ov_text_muted" => {
                    s.ov_text_muted = Self::str_to_opt_color(val);
                }
                "ov_text_faint" => {
                    s.ov_text_faint = Self::str_to_opt_color(val);
                }
                "ov_border_color" => {
                    s.ov_border_color = Self::str_to_opt_color(val);
                }
                "ov_border_lit" => {
                    s.ov_border_lit = Self::str_to_opt_color(val);
                }
                "ov_separator_color" => {
                    s.ov_separator_color = Self::str_to_opt_color(val);
                }
                "ov_button_bg" => {
                    s.ov_button_bg = Self::str_to_opt_color(val);
                }
                "ov_button_hover" => {
                    s.ov_button_hover = Self::str_to_opt_color(val);
                }
                "ov_button_active" => {
                    s.ov_button_active = Self::str_to_opt_color(val);
                }
                "ov_floating_window_bg" => {
                    s.ov_floating_window_bg = Self::str_to_opt_color(val);
                }
                "ov_tool_shelf_bg" => {
                    s.ov_tool_shelf_bg = Self::str_to_opt_color(val);
                }
                "ov_tool_shelf_border" => {
                    s.ov_tool_shelf_border = Self::str_to_opt_color(val);
                }
                "ov_toolbar_bg" => {
                    s.ov_toolbar_bg = Self::str_to_opt_color(val);
                }
                "ov_menu_bg" => {
                    s.ov_menu_bg = Self::str_to_opt_color(val);
                }
                "ov_icon_button_bg" => {
                    s.ov_icon_button_bg = Self::str_to_opt_color(val);
                }
                "ov_icon_button_active" => {
                    s.ov_icon_button_active = Self::str_to_opt_color(val);
                }
                "ov_icon_button_disabled" => {
                    s.ov_icon_button_disabled = Self::str_to_opt_color(val);
                }
                "ov_text_input_bg" => {
                    s.ov_text_input_bg = Self::str_to_opt_color(val);
                }
                "ov_stepper_button_bg" => {
                    s.ov_stepper_button_bg = Self::str_to_opt_color(val);
                }
                "ov_canvas_bg_top" => {
                    s.ov_canvas_bg_top = Self::str_to_opt_color(val);
                }
                "ov_canvas_bg_bottom" => {
                    s.ov_canvas_bg_bottom = Self::str_to_opt_color(val);
                }
                "ov_glow_accent" => {
                    s.ov_glow_accent = Self::str_to_opt_color(val);
                }
                "ov_accent3" => {
                    s.ov_accent3 = Self::str_to_opt_color(val);
                }
                "ov_accent4" => {
                    s.ov_accent4 = Self::str_to_opt_color(val);
                }
                _ => {
                    // Parse keybinding lines: keybind.ActionName=combo
                    if let Some(action_name) = key.strip_prefix("keybind.") {
                        s.keybindings.load_config_line(action_name, val);
                    }
                }
            }
        }

        // Corruption detection: if language field looks like a number, the
        // settings file was saved with the old misaligned argument order.
        // Reset to defaults to avoid cascading corruption.
        if s.language.parse::<f64>().is_ok() {
            return Self::default();
        }

        // If not custom, override accent with preset colors
        if s.theme_preset != ThemePreset::Custom {
            s.custom_accent = s.theme_preset.accent_colors();
        }

        s
    }
}

/// Pixel grid display modes
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PixelGridMode {
    /// Show grid only when zoomed in enough
    Auto,
    /// Always show grid
    AlwaysOn,
    /// Never show grid
    AlwaysOff,
}

impl PixelGridMode {
    pub fn name(&self) -> String {
        match self {
            PixelGridMode::Auto => t!("pixel_grid_mode.auto"),
            PixelGridMode::AlwaysOn => t!("pixel_grid_mode.always_on"),
            PixelGridMode::AlwaysOff => t!("pixel_grid_mode.always_off"),
        }
    }

    pub fn all() -> &'static [PixelGridMode] {
        &[
            PixelGridMode::Auto,
            PixelGridMode::AlwaysOn,
            PixelGridMode::AlwaysOff,
        ]
    }
}
