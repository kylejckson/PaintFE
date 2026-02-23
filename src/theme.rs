use eframe::egui::{self, Color32, Rounding, Stroke, Visuals};
use eframe::epaint::Shadow;

/// Theme mode for the application
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub enum ThemeMode {
    #[default]
    Dark,
    Light,
}

// ============================================================================
// ACCENT THEME SYSTEM
// ============================================================================

/// The 6 accent color slots for a theme definition
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AccentColors {
    /// Light mode: vibrant accent for buttons, main UI elements
    pub light_normal: Color32,
    /// Light mode: faint/transparent for backgrounds, selection highlights, glows
    pub light_faint: Color32,
    /// Light mode: darker/contrast for underlines, active borders, focus rings
    pub light_strong: Color32,
    /// Dark mode: vibrant accent
    pub dark_normal: Color32,
    /// Dark mode: faint/alpha for backgrounds
    pub dark_faint: Color32,
    /// Dark mode: brighter/contrast for borders and focus
    pub dark_strong: Color32,
}

impl AccentColors {
    pub fn for_mode(&self, mode: ThemeMode) -> (Color32, Color32, Color32) {
        match mode {
            ThemeMode::Light => (self.light_normal, self.light_faint, self.light_strong),
            ThemeMode::Dark => (self.dark_normal, self.dark_faint, self.dark_strong),
        }
    }
}

/// Preset accent themes
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum ThemePreset {
    #[default]
    Blue,
    Orange,
    Purple,
    Red,
    Green,
    Lime,
    Nebula,
    Ember,
    Sakura,
    Glacier,
    Midnight,
    Custom,
}

impl ThemePreset {
    pub fn label(&self) -> String {
        match self {
            ThemePreset::Blue => t!("theme_preset.blue"),
            ThemePreset::Orange => t!("theme_preset.orange"),
            ThemePreset::Purple => t!("theme_preset.purple"),
            ThemePreset::Red => t!("theme_preset.red"),
            ThemePreset::Green => t!("theme_preset.green"),
            ThemePreset::Lime => t!("theme_preset.lime"),
            ThemePreset::Nebula => t!("theme_preset.nebula"),
            ThemePreset::Ember => t!("theme_preset.ember"),
            ThemePreset::Sakura => t!("theme_preset.sakura"),
            ThemePreset::Glacier => t!("theme_preset.glacier"),
            ThemePreset::Midnight => t!("theme_preset.midnight"),
            ThemePreset::Custom => t!("theme_preset.custom"),
        }
    }

    /// All selectable presets (excludes Custom since it is auto-selected)
    pub fn all() -> &'static [ThemePreset] {
        &[
            ThemePreset::Blue,
            ThemePreset::Orange,
            ThemePreset::Purple,
            ThemePreset::Red,
            ThemePreset::Green,
            ThemePreset::Lime,
            ThemePreset::Nebula,
            ThemePreset::Ember,
            ThemePreset::Sakura,
            ThemePreset::Glacier,
            ThemePreset::Midnight,
        ]
    }

    pub fn accent_colors(&self) -> AccentColors {
        match self {
            // Blue (current default, snapshotted from existing theme)
            ThemePreset::Blue => AccentColors {
                light_normal: Color32::from_rgb(56, 123, 234),    // #387BEA
                light_faint:  Color32::from_rgba_unmultiplied(56, 123, 234, 50),
                light_strong: Color32::from_rgb(30, 80, 180),     // #1E50B4
                dark_normal:  Color32::from_rgb(66, 133, 244),    // #4285F4
                dark_faint:   Color32::from_rgba_unmultiplied(66, 133, 244, 60),
                dark_strong:  Color32::from_rgb(120, 175, 255),   // #78AFFF
            },
            // Orange — warm, energetic
            ThemePreset::Orange => AccentColors {
                light_normal: Color32::from_rgb(230, 126, 34),    // #E67E22
                light_faint:  Color32::from_rgba_unmultiplied(230, 126, 34, 50),
                light_strong: Color32::from_rgb(180, 82, 10),     // #B4520A
                dark_normal:  Color32::from_rgb(243, 156, 18),    // #F39C12
                dark_faint:   Color32::from_rgba_unmultiplied(243, 156, 18, 60),
                dark_strong:  Color32::from_rgb(255, 200, 100),   // #FFC864
            },
            // Purple — creative, modern
            ThemePreset::Purple => AccentColors {
                light_normal: Color32::from_rgb(142, 68, 173),    // #8E44AD
                light_faint:  Color32::from_rgba_unmultiplied(142, 68, 173, 50),
                light_strong: Color32::from_rgb(100, 30, 140),    // #641E8C
                dark_normal:  Color32::from_rgb(165, 105, 210),   // #A569D2
                dark_faint:   Color32::from_rgba_unmultiplied(165, 105, 210, 60),
                dark_strong:  Color32::from_rgb(200, 160, 245),   // #C8A0F5
            },
            // Red — bold, action-oriented
            ThemePreset::Red => AccentColors {
                light_normal: Color32::from_rgb(211, 47, 47),     // #D32F2F
                light_faint:  Color32::from_rgba_unmultiplied(211, 47, 47, 50),
                light_strong: Color32::from_rgb(160, 20, 20),     // #A01414
                dark_normal:  Color32::from_rgb(239, 83, 80),     // #EF5350
                dark_faint:   Color32::from_rgba_unmultiplied(239, 83, 80, 60),
                dark_strong:  Color32::from_rgb(255, 150, 150),   // #FF9696
            },
            // Green — natural, calm
            ThemePreset::Green => AccentColors {
                light_normal: Color32::from_rgb(39, 174, 96),     // #27AE60
                light_faint:  Color32::from_rgba_unmultiplied(39, 174, 96, 50),
                light_strong: Color32::from_rgb(20, 120, 60),     // #14783C
                dark_normal:  Color32::from_rgb(46, 204, 113),    // #2ECC71
                dark_faint:   Color32::from_rgba_unmultiplied(46, 204, 113, 60),
                dark_strong:  Color32::from_rgb(120, 235, 170),   // #78EBAA
            },
            // Lime — fresh, vibrant
            ThemePreset::Lime => AccentColors {
                light_normal: Color32::from_rgb(130, 190, 20),    // #82BE14
                light_faint:  Color32::from_rgba_unmultiplied(130, 190, 20, 50),
                light_strong: Color32::from_rgb(85, 135, 5),      // #558705
                dark_normal:  Color32::from_rgb(160, 220, 50),    // #A0DC32
                dark_faint:   Color32::from_rgba_unmultiplied(160, 220, 50, 60),
                dark_strong:  Color32::from_rgb(200, 245, 120),   // #C8F578
            },
            // Nebula — deep violet with electric cyan highlights
            ThemePreset::Nebula => AccentColors {
                light_normal: Color32::from_rgb(124, 58, 237),    // #7C3AED rich violet
                light_faint:  Color32::from_rgba_unmultiplied(124, 58, 237, 45),
                light_strong: Color32::from_rgb(79, 22, 178),     // #4F16B2 deep violet
                dark_normal:  Color32::from_rgb(167, 139, 250),   // #A78BFA soft violet
                dark_faint:   Color32::from_rgba_unmultiplied(139, 92, 246, 60),
                dark_strong:  Color32::from_rgb(216, 180, 254),   // #D8B4FE lavender
            },
            // Ember — molten amber-gold, like live coals
            ThemePreset::Ember => AccentColors {
                light_normal: Color32::from_rgb(180, 83, 9),      // #B45309 deep amber
                light_faint:  Color32::from_rgba_unmultiplied(180, 83, 9, 45),
                light_strong: Color32::from_rgb(120, 50, 5),      // #783205 dark ember
                dark_normal:  Color32::from_rgb(251, 191, 36),    // #FBBF24 golden amber
                dark_faint:   Color32::from_rgba_unmultiplied(251, 191, 36, 55),
                dark_strong:  Color32::from_rgb(253, 224, 120),   // #FDE078 pale gold
            },
            // Sakura — cherry blossom rose pink
            ThemePreset::Sakura => AccentColors {
                light_normal: Color32::from_rgb(225, 29, 72),     // #E11D48 vivid rose
                light_faint:  Color32::from_rgba_unmultiplied(225, 29, 72, 40),
                light_strong: Color32::from_rgb(159, 18, 57),     // #9F1239 deep rose
                dark_normal:  Color32::from_rgb(251, 113, 133),   // #FB7185 soft pink
                dark_faint:   Color32::from_rgba_unmultiplied(251, 113, 133, 55),
                dark_strong:  Color32::from_rgb(253, 164, 175),   // #FDA4AF blush
            },
            // Glacier — arctic teal and ice blue
            ThemePreset::Glacier => AccentColors {
                light_normal: Color32::from_rgb(8, 145, 178),     // #0891B2 deep cyan
                light_faint:  Color32::from_rgba_unmultiplied(8, 145, 178, 45),
                light_strong: Color32::from_rgb(14, 116, 144),    // #0E7490 dark teal
                dark_normal:  Color32::from_rgb(34, 211, 238),    // #22D3EE electric cyan
                dark_faint:   Color32::from_rgba_unmultiplied(34, 211, 238, 55),
                dark_strong:  Color32::from_rgb(103, 232, 249),   // #67E8F9 ice blue
            },
            // Midnight — deep indigo with electric periwinkle highlights
            ThemePreset::Midnight => AccentColors {
                light_normal: Color32::from_rgb(67, 56, 202),     // #4338CA indigo
                light_faint:  Color32::from_rgba_unmultiplied(67, 56, 202, 45),
                light_strong: Color32::from_rgb(49, 46, 129),     // #312E81 deep indigo
                dark_normal:  Color32::from_rgb(129, 140, 248),   // #818CF8 periwinkle
                dark_faint:   Color32::from_rgba_unmultiplied(129, 140, 248, 55),
                dark_strong:  Color32::from_rgb(165, 180, 252),   // #A5B4FC soft periwinkle
            },
            // Custom returns Blue as fallback (actual custom colors stored separately)
            ThemePreset::Custom => ThemePreset::Blue.accent_colors(),
        }
    }
}

/// Application theme (light/dark mode + accent colors).
#[derive(Clone, Debug)]
pub struct Theme {
    pub mode: ThemeMode,
    pub preset: ThemePreset,
    pub accent_colors: AccentColors,
    
    // Base colors
    pub bg_color: Color32,
    pub panel_bg: Color32,
    pub window_bg: Color32,
    pub text_color: Color32,
    pub text_muted: Color32,
    
    // Interactive elements (derived from accent)
    pub accent: Color32,
    pub accent_hover: Color32,
    pub accent_faint: Color32,
    pub accent_strong: Color32,
    pub selection_bg: Color32,
    
    // Borders and strokes
    pub border_color: Color32,
    pub separator_color: Color32,
    
    // Button colors
    pub button_bg: Color32,
    pub button_hover: Color32,
    pub button_active: Color32,
    
    // Panel-specific
    pub floating_window_bg: Color32,
    pub toolbar_bg: Color32,
    pub menu_bg: Color32,
    
    // Canvas background gradient
    pub canvas_bg_top: Color32,
    pub canvas_bg_bottom: Color32,
    
    // Transparency values for glass effect
    pub window_opacity: u8,
    pub panel_opacity: u8,
}

impl Default for Theme {
    fn default() -> Self {
        Self::light_with_accent(ThemePreset::Blue, ThemePreset::Blue.accent_colors())
    }
}

impl Theme {
    pub fn dark_with_accent(preset: ThemePreset, accent_colors: AccentColors) -> Self {
        let (normal, faint, strong) = accent_colors.for_mode(ThemeMode::Dark);
        let hover = Self::lighten(normal, 25);

        Self {
            mode: ThemeMode::Dark,
            preset,
            accent_colors,
            
            // Base colors - neutral dark greys (VS Code style, no blue tint)
            bg_color: Color32::from_gray(20),
            panel_bg: Color32::from_gray(30),
            window_bg: Color32::from_gray(35),
            text_color: Color32::from_gray(240),
            text_muted: Color32::from_gray(170),
            
            // Accent - derived from accent definition
            accent: normal,
            accent_hover: hover,
            accent_faint: faint,
            accent_strong: strong,
            selection_bg: faint,
            
            // Borders
            border_color: Color32::from_gray(55),
            separator_color: Color32::from_gray(48),
            
            // Buttons - darker for better icon contrast in dark mode
            button_bg: Color32::from_gray(28),
            button_hover: Color32::from_gray(40),
            button_active: Color32::from_gray(50),
            
            // Floating elements - subtle transparency
            floating_window_bg: Color32::from_rgba_unmultiplied(35, 35, 35, 245),
            toolbar_bg: Color32::from_gray(22),
            menu_bg: Color32::from_gray(32),
            
            // Canvas background - dark grey (darker than other elements but not black)
            canvas_bg_top: Color32::from_gray(18),
            canvas_bg_bottom: Color32::from_gray(16),
            
            // Glass effect opacity - mostly opaque
            window_opacity: 245,
            panel_opacity: 248,
        }
    }
    
    pub fn light_with_accent(preset: ThemePreset, accent_colors: AccentColors) -> Self {
        let (normal, faint, strong) = accent_colors.for_mode(ThemeMode::Light);
        let hover = Self::lighten(normal, 25);

        Self {
            mode: ThemeMode::Light,
            preset,
            accent_colors,
            
            // Base colors - clean whites and grays
            bg_color: Color32::from_rgb(245, 245, 248),
            panel_bg: Color32::from_rgb(255, 255, 255),
            window_bg: Color32::from_rgb(250, 250, 252),
            text_color: Color32::from_rgb(30, 30, 35),
            text_muted: Color32::from_rgb(100, 100, 110),
            
            // Accent - derived from accent definition
            accent: normal,
            accent_hover: hover,
            accent_faint: faint,
            accent_strong: strong,
            selection_bg: faint,
            
            // Borders
            border_color: Color32::from_rgb(200, 200, 210),
            separator_color: Color32::from_rgb(220, 220, 228),
            
            // Buttons
            button_bg: Color32::from_rgb(235, 235, 240),
            button_hover: Color32::from_rgb(225, 225, 232),
            button_active: Color32::from_rgb(210, 210, 220),
            
            // Floating elements
            floating_window_bg: Color32::from_rgba_unmultiplied(255, 255, 255, 250),
            toolbar_bg: Color32::from_rgb(248, 248, 250),
            menu_bg: Color32::from_rgb(255, 255, 255),
            
            // Canvas background gradient colors
            canvas_bg_top: Color32::from_rgb(235, 235, 240),
            canvas_bg_bottom: Color32::from_rgb(220, 220, 228),
            
            // Glass effect opacity
            window_opacity: 250,
            panel_opacity: 252,
        }
    }

    /// Legacy constructors (use default Blue accent)
    pub fn dark() -> Self {
        Self::dark_with_accent(ThemePreset::Blue, ThemePreset::Blue.accent_colors())
    }

    pub fn light() -> Self {
        Self::light_with_accent(ThemePreset::Blue, ThemePreset::Blue.accent_colors())
    }

    pub fn with_accent(&self, preset: ThemePreset, accent_colors: AccentColors) -> Self {
        match self.mode {
            ThemeMode::Dark => Self::dark_with_accent(preset, accent_colors),
            ThemeMode::Light => Self::light_with_accent(preset, accent_colors),
        }
    }
    
    /// Toggle between light and dark mode (preserving accent)
    pub fn toggle(&mut self) {
        *self = match self.mode {
            ThemeMode::Dark => Self::light_with_accent(self.preset, self.accent_colors),
            ThemeMode::Light => Self::dark_with_accent(self.preset, self.accent_colors),
        };
    }

    /// Lighten a color by adding `amount` to each RGB channel
    fn lighten(c: Color32, amount: u8) -> Color32 {
        Color32::from_rgba_unmultiplied(
            c.r().saturating_add(amount),
            c.g().saturating_add(amount),
            c.b().saturating_add(amount),
            c.a(),
        )
    }
    
    pub fn apply(&self, ctx: &egui::Context) {
        let mut visuals = match self.mode {
            ThemeMode::Dark => Visuals::dark(),
            ThemeMode::Light => Visuals::light(),
        };
        
        // Override with our custom colors
        visuals.panel_fill = self.panel_bg;
        visuals.window_fill = self.window_bg;
        visuals.faint_bg_color = self.button_bg;
        visuals.extreme_bg_color = self.bg_color;
        
        // Widget styling
        visuals.widgets.noninteractive.bg_fill = self.panel_bg;
        visuals.widgets.noninteractive.fg_stroke = Stroke::new(1.0, self.text_muted);
        visuals.widgets.noninteractive.bg_stroke = Stroke::new(1.0, self.border_color);
        visuals.widgets.noninteractive.rounding = Rounding::same(4.0);
        
        visuals.widgets.inactive.bg_fill = self.button_bg;
        visuals.widgets.inactive.fg_stroke = Stroke::new(1.0, self.text_color);
        visuals.widgets.inactive.bg_stroke = Stroke::NONE;
        visuals.widgets.inactive.rounding = Rounding::same(4.0);
        
        // Hover: use accent_normal for border stroke
        visuals.widgets.hovered.bg_fill = self.button_hover;
        visuals.widgets.hovered.fg_stroke = Stroke::new(1.0, self.text_color);
        visuals.widgets.hovered.bg_stroke = Stroke::new(1.0, self.accent);
        visuals.widgets.hovered.rounding = Rounding::same(4.0);
        
        // Active: accent_strong for border
        visuals.widgets.active.bg_fill = self.button_active;
        visuals.widgets.active.fg_stroke = Stroke::new(1.0, self.text_color);
        visuals.widgets.active.bg_stroke = Stroke::new(1.0, self.accent_strong);
        visuals.widgets.active.rounding = Rounding::same(4.0);
        
        // Open menus: accent_strong for border
        visuals.widgets.open.bg_fill = self.button_active;
        visuals.widgets.open.fg_stroke = Stroke::new(1.0, self.text_color);
        visuals.widgets.open.bg_stroke = Stroke::new(1.0, self.accent_strong);
        visuals.widgets.open.rounding = Rounding::same(4.0);
        
        // Selection: accent_faint background, accent_strong stroke
        visuals.selection.bg_fill = self.accent_faint;
        visuals.selection.stroke = Stroke::new(1.0, self.accent_strong);
        
        // Window styling
        visuals.window_rounding = Rounding::same(6.0);
        let shadow_alpha = match self.mode {
            ThemeMode::Dark => 54,  // 50% stronger in dark mode (36 * 1.5)
            ThemeMode::Light => 36,
        };
        visuals.window_shadow = Shadow {
            extrusion: 8.0,
            color: Color32::from_black_alpha(shadow_alpha),
        };
        visuals.window_stroke = Stroke::new(1.0, self.border_color);
        
        // Menu styling
        visuals.menu_rounding = Rounding::same(6.0);
        
        // Popup styling
        let popup_alpha = match self.mode {
            ThemeMode::Dark => 36,  // 50% stronger
            ThemeMode::Light => 24,
        };
        visuals.popup_shadow = Shadow {
            extrusion: 4.0,
            color: Color32::from_black_alpha(popup_alpha),
        };
        
        // Separator
        visuals.widgets.noninteractive.bg_stroke = Stroke::new(1.0, self.separator_color);
        
        // Hyperlinks
        visuals.hyperlink_color = self.accent;
        
        // Warn/error colors (keep defaults but slightly adjust)
        visuals.warn_fg_color = Color32::from_rgb(255, 180, 100);
        visuals.error_fg_color = Color32::from_rgb(255, 100, 100);
        
        ctx.set_visuals(visuals);

        // Update native window title bar to match theme (Windows 10+)
        #[cfg(target_os = "windows")]
        set_native_dark_title_bar(matches!(self.mode, ThemeMode::Dark));
    }
    
    pub fn floating_window_frame(&self) -> egui::Frame {
        let shadow_alpha = match self.mode {
            ThemeMode::Dark => 36,  // 50% stronger
            ThemeMode::Light => 24,
        };
        egui::Frame::none()
            .fill(self.floating_window_bg)
            .rounding(Rounding::same(6.0))
            .stroke(Stroke::new(1.0, self.border_color))
            .shadow(Shadow {
                extrusion: 6.0,
                color: Color32::from_black_alpha(shadow_alpha),
            })
            .inner_margin(egui::Margin::same(8.0))
    }
    
    /// Context bar frame (no shadow).
    pub fn context_bar_frame(&self) -> egui::Frame {
        egui::Frame::none()
            .fill(self.panel_bg)
            .inner_margin(egui::Margin::symmetric(10.0, 10.0))
    }
    
    pub fn toolbar_frame(&self) -> egui::Frame {
        egui::Frame::none()
            .fill(self.toolbar_bg)
            .inner_margin(egui::Margin::symmetric(8.0, 5.5))
    }
    
    pub fn menu_frame(&self) -> egui::Frame {
        egui::Frame::none()
            .fill(self.menu_bg)
            .inner_margin(egui::Margin::symmetric(8.0, 2.0))
    }
    
    pub fn tab_frame(&self) -> egui::Frame {
        egui::Frame::none()
            .fill(self.toolbar_bg)
            .inner_margin(egui::Margin::symmetric(4.0, 2.0))
    }
    
    /// Active tab fill — uses accent_strong for contrast
    pub fn active_tab_fill(&self) -> Color32 {
        self.accent
    }
    
    /// Inactive tab fill
    pub fn inactive_tab_fill(&self) -> Color32 {
        self.button_bg
    }
    
    /// Text color for active tabs
    pub fn active_tab_text(&self) -> Color32 {
        Color32::WHITE
    }
    
    /// Text color for inactive tabs
    pub fn inactive_tab_text(&self) -> Color32 {
        self.text_muted
    }
}

/// Window visibility state for floating panels
#[derive(Clone, Debug, Default)]
pub struct WindowVisibility {
    pub tools: bool,
    pub layers: bool,
    pub history: bool,
    pub colors: bool,
    pub script_editor: bool,
}

impl WindowVisibility {
    pub fn new() -> Self {
        Self {
            tools: true,     // Tools always visible by default
            layers: true,    // Layers visible by default
            history: false,  // History hidden by default
            colors: false,   // Colors hidden by default (toggle from swatch)
            script_editor: false, // Script editor hidden by default
        }
    }
}

// ============================================================================
// NATIVE WINDOW DARK MODE (Windows 10 1809+)
// ============================================================================

/// Set the native window title bar to dark or light mode via DWM.
/// Uses `DWMWA_USE_IMMERSIVE_DARK_MODE` (attribute 20) on the app's HWND.
#[cfg(target_os = "windows")]
fn set_native_dark_title_bar(dark: bool) {
    use std::ffi::OsStr;
    use std::os::windows::ffi::OsStrExt;

    unsafe {
        // Find our window by its title ("PaintFE").
        let title: Vec<u16> = OsStr::new("PaintFE")
            .encode_wide()
            .chain(std::iter::once(0))
            .collect();
        let hwnd = winapi::um::winuser::FindWindowW(std::ptr::null(), title.as_ptr());
        if hwnd.is_null() {
            return;
        }

        // DWMWA_USE_IMMERSIVE_DARK_MODE = 20
        let value: winapi::shared::minwindef::BOOL = if dark { 1 } else { 0 };
        winapi::um::dwmapi::DwmSetWindowAttribute(
            hwnd,
            20, // DWMWA_USE_IMMERSIVE_DARK_MODE
            &value as *const _ as *const _,
            std::mem::size_of::<winapi::shared::minwindef::BOOL>() as u32,
        );
    }
}
