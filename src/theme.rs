use eframe::egui::{self, Color32, Rounding, Stroke, Visuals};
use eframe::epaint::Shadow;

/// Theme mode for the application
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub enum ThemeMode {
    #[default]
    Dark,
    Light,
}

/// UI density level — controls spacing, margins, and row heights.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub enum UiDensity {
    Compact,
    #[default]
    Normal,
    Spacious,
}

impl UiDensity {
    pub fn label(&self) -> &'static str {
        match self {
            UiDensity::Compact => "Compact",
            UiDensity::Normal => "Normal",
            UiDensity::Spacious => "Spacious",
        }
    }

    pub fn all() -> &'static [UiDensity] {
        &[UiDensity::Compact, UiDensity::Normal, UiDensity::Spacious]
    }

    /// Item spacing (vertical distance between widgets)
    pub fn item_spacing(&self) -> f32 {
        match self {
            UiDensity::Compact => 4.0,
            UiDensity::Normal => 6.0,
            UiDensity::Spacious => 8.0,
        }
    }

    /// Panel inner margin
    pub fn margin(&self) -> f32 {
        match self {
            UiDensity::Compact => 6.0,
            UiDensity::Normal => 10.0,
            UiDensity::Spacious => 16.0,
        }
    }
}

/// User-facing overrides for individual theme color/geometry fields.
///
/// Every field is `Option` — `None` means "use the preset default".
/// `Some(value)` pins that field regardless of which preset is active.
#[derive(Clone, Debug, Default)]
pub struct ThemeOverrides {
    // Surface colors
    pub bg_color: Option<Color32>,
    pub panel_bg: Option<Color32>,
    pub window_bg: Option<Color32>,
    pub bg2: Option<Color32>,
    pub bg3: Option<Color32>,

    // Text
    pub text_color: Option<Color32>,
    pub text_muted: Option<Color32>,
    pub text_faint: Option<Color32>,

    // Borders
    pub border_color: Option<Color32>,
    pub border_lit: Option<Color32>,
    pub separator_color: Option<Color32>,

    // Buttons
    pub button_bg: Option<Color32>,
    pub button_hover: Option<Color32>,
    pub button_active: Option<Color32>,

    // Panels & windows
    pub floating_window_bg: Option<Color32>,
    pub toolbar_bg: Option<Color32>,
    pub menu_bg: Option<Color32>,

    // Canvas
    pub canvas_bg_top: Option<Color32>,
    pub canvas_bg_bottom: Option<Color32>,

    // Glow & effects
    pub glow_accent: Option<Color32>,

    // Additional accents
    pub accent3: Option<Color32>,
    pub accent4: Option<Color32>,

    // Geometry
    pub widget_rounding: Option<f32>,
    pub window_rounding: Option<f32>,
    pub menu_rounding: Option<f32>,

    // Atmosphere
    pub glow_intensity: Option<f32>,
    pub shadow_strength: Option<f32>,
}

impl ThemeOverrides {
    /// An empty set of overrides (all `None`).
    pub fn empty() -> Self {
        Self::default()
    }

    /// Returns true if every field is `None`.
    pub fn is_empty(&self) -> bool {
        self.bg_color.is_none()
            && self.panel_bg.is_none()
            && self.window_bg.is_none()
            && self.bg2.is_none()
            && self.bg3.is_none()
            && self.text_color.is_none()
            && self.text_muted.is_none()
            && self.text_faint.is_none()
            && self.border_color.is_none()
            && self.border_lit.is_none()
            && self.separator_color.is_none()
            && self.button_bg.is_none()
            && self.button_hover.is_none()
            && self.button_active.is_none()
            && self.floating_window_bg.is_none()
            && self.toolbar_bg.is_none()
            && self.menu_bg.is_none()
            && self.canvas_bg_top.is_none()
            && self.canvas_bg_bottom.is_none()
            && self.glow_accent.is_none()
            && self.accent3.is_none()
            && self.accent4.is_none()
            && self.widget_rounding.is_none()
            && self.window_rounding.is_none()
            && self.menu_rounding.is_none()
            && self.glow_intensity.is_none()
            && self.shadow_strength.is_none()
    }
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
    Signal,
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
            ThemePreset::Signal => t!("theme_preset.signal"),
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
            ThemePreset::Signal,
        ]
    }

    pub fn accent_colors(&self) -> AccentColors {
        match self {
            // Blue (current default, snapshotted from existing theme)
            ThemePreset::Blue => AccentColors {
                light_normal: Color32::from_rgb(56, 123, 234), // #387BEA
                light_faint: Color32::from_rgba_unmultiplied(56, 123, 234, 50),
                light_strong: Color32::from_rgb(30, 80, 180), // #1E50B4
                dark_normal: Color32::from_rgb(66, 133, 244), // #4285F4
                dark_faint: Color32::from_rgba_unmultiplied(66, 133, 244, 60),
                dark_strong: Color32::from_rgb(120, 175, 255), // #78AFFF
            },
            // Orange — warm, energetic
            ThemePreset::Orange => AccentColors {
                light_normal: Color32::from_rgb(230, 126, 34), // #E67E22
                light_faint: Color32::from_rgba_unmultiplied(230, 126, 34, 50),
                light_strong: Color32::from_rgb(180, 82, 10), // #B4520A
                dark_normal: Color32::from_rgb(243, 156, 18), // #F39C12
                dark_faint: Color32::from_rgba_unmultiplied(243, 156, 18, 60),
                dark_strong: Color32::from_rgb(255, 200, 100), // #FFC864
            },
            // Purple — creative, modern
            ThemePreset::Purple => AccentColors {
                light_normal: Color32::from_rgb(142, 68, 173), // #8E44AD
                light_faint: Color32::from_rgba_unmultiplied(142, 68, 173, 50),
                light_strong: Color32::from_rgb(100, 30, 140), // #641E8C
                dark_normal: Color32::from_rgb(165, 105, 210), // #A569D2
                dark_faint: Color32::from_rgba_unmultiplied(165, 105, 210, 60),
                dark_strong: Color32::from_rgb(200, 160, 245), // #C8A0F5
            },
            // Red — bold, action-oriented
            ThemePreset::Red => AccentColors {
                light_normal: Color32::from_rgb(211, 47, 47), // #D32F2F
                light_faint: Color32::from_rgba_unmultiplied(211, 47, 47, 50),
                light_strong: Color32::from_rgb(160, 20, 20), // #A01414
                dark_normal: Color32::from_rgb(239, 83, 80),  // #EF5350
                dark_faint: Color32::from_rgba_unmultiplied(239, 83, 80, 60),
                dark_strong: Color32::from_rgb(255, 150, 150), // #FF9696
            },
            // Green — natural, calm
            ThemePreset::Green => AccentColors {
                light_normal: Color32::from_rgb(39, 174, 96), // #27AE60
                light_faint: Color32::from_rgba_unmultiplied(39, 174, 96, 50),
                light_strong: Color32::from_rgb(20, 120, 60), // #14783C
                dark_normal: Color32::from_rgb(46, 204, 113), // #2ECC71
                dark_faint: Color32::from_rgba_unmultiplied(46, 204, 113, 60),
                dark_strong: Color32::from_rgb(120, 235, 170), // #78EBAA
            },
            // Lime — fresh, vibrant
            ThemePreset::Lime => AccentColors {
                light_normal: Color32::from_rgb(130, 190, 20), // #82BE14
                light_faint: Color32::from_rgba_unmultiplied(130, 190, 20, 50),
                light_strong: Color32::from_rgb(85, 135, 5), // #558705
                dark_normal: Color32::from_rgb(160, 220, 50), // #A0DC32
                dark_faint: Color32::from_rgba_unmultiplied(160, 220, 50, 60),
                dark_strong: Color32::from_rgb(200, 245, 120), // #C8F578
            },
            // Nebula — deep violet with electric cyan highlights
            ThemePreset::Nebula => AccentColors {
                light_normal: Color32::from_rgb(124, 58, 237), // #7C3AED rich violet
                light_faint: Color32::from_rgba_unmultiplied(124, 58, 237, 45),
                light_strong: Color32::from_rgb(79, 22, 178), // #4F16B2 deep violet
                dark_normal: Color32::from_rgb(167, 139, 250), // #A78BFA soft violet
                dark_faint: Color32::from_rgba_unmultiplied(139, 92, 246, 60),
                dark_strong: Color32::from_rgb(216, 180, 254), // #D8B4FE lavender
            },
            // Ember — molten amber-gold, like live coals
            ThemePreset::Ember => AccentColors {
                light_normal: Color32::from_rgb(180, 83, 9), // #B45309 deep amber
                light_faint: Color32::from_rgba_unmultiplied(180, 83, 9, 45),
                light_strong: Color32::from_rgb(120, 50, 5), // #783205 dark ember
                dark_normal: Color32::from_rgb(251, 191, 36), // #FBBF24 golden amber
                dark_faint: Color32::from_rgba_unmultiplied(251, 191, 36, 55),
                dark_strong: Color32::from_rgb(253, 224, 120), // #FDE078 pale gold
            },
            // Sakura — cherry blossom rose pink
            ThemePreset::Sakura => AccentColors {
                light_normal: Color32::from_rgb(225, 29, 72), // #E11D48 vivid rose
                light_faint: Color32::from_rgba_unmultiplied(225, 29, 72, 40),
                light_strong: Color32::from_rgb(159, 18, 57), // #9F1239 deep rose
                dark_normal: Color32::from_rgb(251, 113, 133), // #FB7185 soft pink
                dark_faint: Color32::from_rgba_unmultiplied(251, 113, 133, 55),
                dark_strong: Color32::from_rgb(253, 164, 175), // #FDA4AF blush
            },
            // Glacier — arctic teal and ice blue
            ThemePreset::Glacier => AccentColors {
                light_normal: Color32::from_rgb(8, 145, 178), // #0891B2 deep cyan
                light_faint: Color32::from_rgba_unmultiplied(8, 145, 178, 45),
                light_strong: Color32::from_rgb(14, 116, 144), // #0E7490 dark teal
                dark_normal: Color32::from_rgb(34, 211, 238),  // #22D3EE electric cyan
                dark_faint: Color32::from_rgba_unmultiplied(34, 211, 238, 55),
                dark_strong: Color32::from_rgb(103, 232, 249), // #67E8F9 ice blue
            },
            // Midnight — deep indigo with electric periwinkle highlights
            ThemePreset::Midnight => AccentColors {
                light_normal: Color32::from_rgb(67, 56, 202), // #4338CA indigo
                light_faint: Color32::from_rgba_unmultiplied(67, 56, 202, 45),
                light_strong: Color32::from_rgb(49, 46, 129), // #312E81 deep indigo
                dark_normal: Color32::from_rgb(129, 140, 248), // #818CF8 periwinkle
                dark_faint: Color32::from_rgba_unmultiplied(129, 140, 248, 55),
                dark_strong: Color32::from_rgb(165, 180, 252), // #A5B4FC soft periwinkle
            },
            // Signal — website-inspired burnt orange + green (Signal Grid design language)
            ThemePreset::Signal => AccentColors {
                light_normal: Color32::from_rgb(232, 89, 12),     // #e8590c burnt orange
                light_faint: Color32::from_rgba_unmultiplied(232, 89, 12, 30),
                light_strong: Color32::from_rgb(200, 60, 5),     // deep orange
                dark_normal: Color32::from_rgb(232, 89, 12),     // #e8590c
                dark_faint: Color32::from_rgba_unmultiplied(232, 89, 12, 40),
                dark_strong: Color32::from_rgb(249, 115, 22),    // #f97316 bright orange
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

    // Base colors — 4-tier depth system (Signal Grid)
    pub bg_color: Color32,
    pub panel_bg: Color32,
    pub window_bg: Color32,
    pub bg2: Color32,      // Elevated surface (controls, insets)
    pub bg3: Color32,      // Highest surface (active controls, headers)
    pub text_color: Color32,
    pub text_muted: Color32,
    pub text_faint: Color32, // Tertiary text (hints, placeholder)

    // Interactive elements (derived from accent)
    pub accent: Color32,
    pub accent_hover: Color32,
    pub accent_faint: Color32,
    pub accent_strong: Color32,
    pub selection_bg: Color32,

    // Additional semantic accents
    pub accent3: Color32,       // Secondary accent (green — success, links)
    pub accent4: Color32,       // Tertiary accent (amber — warning)

    // Borders and strokes
    pub border_color: Color32,
    pub border_lit: Color32,    // Brighter "hover" border
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

    // Glow/atmosphere colors
    pub glow_accent: Color32,   // Accent glow (rgba with low alpha)
    pub glow_accent3: Color32,  // Green glow

    // Transparency values for glass effect
    pub window_opacity: u8,
    pub panel_opacity: u8,

    // Geometry overrides (applied from ThemeOverrides, used in apply())
    pub widget_rounding: f32,
    pub window_rounding: f32,
    pub menu_rounding: f32,
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

            // Base colors — blue-tinted neutrals (Signal Grid dark palette)
            bg_color: Color32::from_rgb(10, 10, 11),       // #0a0a0b deepest
            panel_bg: Color32::from_rgb(17, 17, 20),       // #111114
            window_bg: Color32::from_rgb(23, 23, 28),      // #17171c
            bg2: Color32::from_rgb(23, 23, 28),            // #17171c elevated controls
            bg3: Color32::from_rgb(30, 30, 38),            // #1e1e26 active controls
            text_color: Color32::from_rgb(232, 232, 240),  // #e8e8f0
            text_muted: Color32::from_rgb(122, 122, 144),  // #7a7a90
            text_faint: Color32::from_rgb(61, 61, 85),     // #3d3d55

            // Accent — derived from accent definition
            accent: normal,
            accent_hover: hover,
            accent_faint: faint,
            accent_strong: strong,
            selection_bg: faint,

            // Additional semantic accents
            accent3: Color32::from_rgb(76, 175, 124),      // #4caf7c green
            accent4: Color32::from_rgb(234, 179, 8),       // #eab308 amber

            // Borders
            border_color: Color32::from_rgb(42, 42, 53),   // #2a2a35
            border_lit: Color32::from_rgb(61, 61, 85),     // #3d3d55 hover border
            separator_color: Color32::from_rgb(35, 35, 45),

            // Buttons
            button_bg: Color32::from_rgb(23, 23, 28),      // #17171c (bg2)
            button_hover: Color32::from_rgb(30, 30, 38),   // #1e1e26 (bg3)
            button_active: Color32::from_rgb(42, 42, 53),  // matches border

            // Floating elements — subtle transparency
            floating_window_bg: Color32::from_rgba_unmultiplied(17, 17, 20, 240),
            toolbar_bg: Color32::from_rgb(10, 10, 11),     // #0a0a0b deepest
            menu_bg: Color32::from_rgb(17, 17, 20),        // #111114

            // Canvas background — dark blue-tinted
            canvas_bg_top: Color32::from_rgb(10, 10, 14),
            canvas_bg_bottom: Color32::from_rgb(8, 8, 11),

            // Glow colors
            glow_accent: Color32::from_rgba_unmultiplied(normal.r(), normal.g(), normal.b(), 38),
            glow_accent3: Color32::from_rgba_unmultiplied(76, 175, 124, 38),

            // Glass effect opacity
            window_opacity: 245,
            panel_opacity: 248,

            // Default geometry
            widget_rounding: 6.0,
            window_rounding: 10.0,
            menu_rounding: 8.0,
        }
    }

    pub fn light_with_accent(preset: ThemePreset, accent_colors: AccentColors) -> Self {
        let (normal, faint, strong) = accent_colors.for_mode(ThemeMode::Light);
        let hover = Self::lighten(normal, 25);

        Self {
            mode: ThemeMode::Light,
            preset,
            accent_colors,

            // Base colors — blue-tinted neutrals (Signal Grid light palette)
            bg_color: Color32::from_rgb(244, 244, 247),    // #f4f4f7
            panel_bg: Color32::from_rgb(255, 255, 255),    // white
            window_bg: Color32::from_rgb(234, 234, 240),   // #eaeaf0
            bg2: Color32::from_rgb(234, 234, 240),         // #eaeaf0 elevated controls
            bg3: Color32::from_rgb(224, 224, 234),         // #e0e0ea active controls
            text_color: Color32::from_rgb(24, 24, 42),     // #18182a
            text_muted: Color32::from_rgb(85, 85, 110),    // #55556e
            text_faint: Color32::from_rgb(180, 180, 200),  // #b4b4c8

            // Accent — derived from accent definition
            accent: normal,
            accent_hover: hover,
            accent_faint: faint,
            accent_strong: strong,
            selection_bg: faint,

            // Additional semantic accents
            accent3: Color32::from_rgb(58, 138, 94),       // #3a8a5e green
            accent4: Color32::from_rgb(234, 179, 8),       // #eab308 amber

            // Borders
            border_color: Color32::from_rgb(208, 208, 222),// #d0d0de
            border_lit: Color32::from_rgb(180, 180, 204),  // #b4b4cc hover border
            separator_color: Color32::from_rgb(216, 216, 230),

            // Buttons
            button_bg: Color32::from_rgb(234, 234, 240),   // #eaeaf0 (bg2)
            button_hover: Color32::from_rgb(224, 224, 234), // #e0e0ea (bg3)
            button_active: Color32::from_rgb(208, 208, 222),// matches border

            // Floating elements
            floating_window_bg: Color32::from_rgba_unmultiplied(255, 255, 255, 248),
            toolbar_bg: Color32::from_rgb(244, 244, 247),  // #f4f4f7
            menu_bg: Color32::from_rgb(255, 255, 255),

            // Canvas background — light blue-tinted gradient
            canvas_bg_top: Color32::from_rgb(234, 234, 240),
            canvas_bg_bottom: Color32::from_rgb(220, 220, 232),

            // Glow colors
            glow_accent: Color32::from_rgba_unmultiplied(normal.r(), normal.g(), normal.b(), 30),
            glow_accent3: Color32::from_rgba_unmultiplied(58, 138, 94, 30),

            // Glass effect opacity
            window_opacity: 250,
            panel_opacity: 252,

            // Default geometry
            widget_rounding: 6.0,
            window_rounding: 10.0,
            menu_rounding: 8.0,
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

    /// Apply user overrides on top of the current theme values.
    pub fn apply_overrides(&mut self, ov: &ThemeOverrides) {
        if let Some(c) = ov.bg_color { self.bg_color = c; }
        if let Some(c) = ov.panel_bg { self.panel_bg = c; }
        if let Some(c) = ov.window_bg { self.window_bg = c; }
        if let Some(c) = ov.bg2 { self.bg2 = c; }
        if let Some(c) = ov.bg3 { self.bg3 = c; }
        if let Some(c) = ov.text_color { self.text_color = c; }
        if let Some(c) = ov.text_muted { self.text_muted = c; }
        if let Some(c) = ov.text_faint { self.text_faint = c; }
        if let Some(c) = ov.border_color { self.border_color = c; }
        if let Some(c) = ov.border_lit { self.border_lit = c; }
        if let Some(c) = ov.separator_color { self.separator_color = c; }
        if let Some(c) = ov.button_bg { self.button_bg = c; }
        if let Some(c) = ov.button_hover { self.button_hover = c; }
        if let Some(c) = ov.button_active { self.button_active = c; }
        if let Some(c) = ov.floating_window_bg { self.floating_window_bg = c; }
        if let Some(c) = ov.toolbar_bg { self.toolbar_bg = c; }
        if let Some(c) = ov.menu_bg { self.menu_bg = c; }
        if let Some(c) = ov.canvas_bg_top { self.canvas_bg_top = c; }
        if let Some(c) = ov.canvas_bg_bottom { self.canvas_bg_bottom = c; }
        if let Some(c) = ov.glow_accent { self.glow_accent = c; }
        if let Some(c) = ov.accent3 {
            self.accent3 = c;
            self.glow_accent3 = Color32::from_rgba_unmultiplied(c.r(), c.g(), c.b(), 38);
        }
        if let Some(c) = ov.accent4 { self.accent4 = c; }
        // Geometry overrides
        if let Some(v) = ov.widget_rounding { self.widget_rounding = v; }
        if let Some(v) = ov.window_rounding { self.window_rounding = v; }
        if let Some(v) = ov.menu_rounding { self.menu_rounding = v; }
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

        // Widget styling
        visuals.widgets.noninteractive.bg_fill = self.panel_bg;
        visuals.widgets.noninteractive.fg_stroke = Stroke::new(1.0, self.text_muted);
        visuals.widgets.noninteractive.bg_stroke = Stroke::new(1.0, self.border_color);
        visuals.widgets.noninteractive.rounding = Rounding::same(self.widget_rounding);

        // Inactive widgets: slider rails, checkbox bg, combo-box bg.
        // Must contrast clearly against panel_bg / window_bg (which equal button_bg in our palette).
        let inactive_fill = match self.mode {
            ThemeMode::Dark  => Color32::from_rgb(42, 42, 53),   // #2a2a35 — blue-tinted mid-gray
            ThemeMode::Light => Color32::from_rgb(208, 208, 222),// #d0d0de — noticeably darker than white
        };
        visuals.widgets.inactive.bg_fill = inactive_fill;
        visuals.widgets.inactive.weak_bg_fill = inactive_fill;
        visuals.widgets.inactive.fg_stroke = Stroke::new(1.0, self.text_color);
        visuals.widgets.inactive.bg_stroke = Stroke::NONE;
        visuals.widgets.inactive.rounding = Rounding::same(self.widget_rounding);

        // Hover: use accent_normal for border stroke + subtle expansion for micro-interaction feel
        visuals.widgets.hovered.bg_fill = self.button_hover;
        visuals.widgets.hovered.fg_stroke = Stroke::new(1.0, self.text_color);
        visuals.widgets.hovered.bg_stroke = Stroke::new(1.0, self.accent);
        visuals.widgets.hovered.rounding = Rounding::same(self.widget_rounding);
        visuals.widgets.hovered.expansion = 1.0; // subtle grow on hover (Phase 9)

        // Active: accent_strong for border
        visuals.widgets.active.bg_fill = self.button_active;
        visuals.widgets.active.fg_stroke = Stroke::new(1.0, self.text_color);
        visuals.widgets.active.bg_stroke = Stroke::new(1.0, self.accent_strong);
        visuals.widgets.active.rounding = Rounding::same(self.widget_rounding);

        // Open menus: accent3 (green) text, accent3 alpha-20 bg — matches website nav active style
        let accent3_bg = Color32::from_rgba_unmultiplied(
            self.accent3.r(), self.accent3.g(), self.accent3.b(), 20,
        );
        visuals.widgets.open.bg_fill = accent3_bg;
        visuals.widgets.open.fg_stroke = Stroke::new(1.0, self.accent3);
        visuals.widgets.open.bg_stroke = Stroke::new(1.0, self.accent3);
        visuals.widgets.open.rounding = Rounding::same(self.widget_rounding);

        // Selection: accent_faint background, accent_strong stroke
        visuals.selection.bg_fill = self.accent_faint;
        visuals.selection.stroke = Stroke::new(1.0, self.accent_strong);

        // Window styling
        visuals.window_rounding = Rounding::same(self.window_rounding);
        let shadow_alpha = match self.mode {
            ThemeMode::Dark => 54, // 50% stronger in dark mode (36 * 1.5)
            ThemeMode::Light => 36,
        };
        visuals.window_shadow = Shadow {
            extrusion: 10.0,
            color: Color32::from_black_alpha(shadow_alpha),
        };
        visuals.window_stroke = Stroke::new(1.0, self.border_color);

        // Menu styling
        visuals.menu_rounding = Rounding::same(self.menu_rounding);

        // Popup styling
        let popup_alpha = match self.mode {
            ThemeMode::Dark => 36, // 50% stronger
            ThemeMode::Light => 24,
        };
        visuals.popup_shadow = Shadow {
            extrusion: 4.0,
            color: Color32::from_black_alpha(popup_alpha),
        };

        // Separator
        visuals.widgets.noninteractive.bg_stroke = Stroke::new(1.0, self.separator_color);

        // Scrollbar: track background distinct from panel bg
        match self.mode {
            ThemeMode::Dark => {
                visuals.extreme_bg_color = Color32::from_rgb(8, 8, 10); // darker than panel_bg for visible track groove
            }
            ThemeMode::Light => {
                visuals.extreme_bg_color = Color32::from_rgb(208, 208, 222); // darker than white for visible track
            }
        }

        // Hyperlinks
        visuals.hyperlink_color = self.accent;

        // Warn/error colors (keep defaults but slightly adjust)
        visuals.warn_fg_color = Color32::from_rgb(255, 180, 100);
        visuals.error_fg_color = Color32::from_rgb(255, 100, 100);

        ctx.set_visuals(visuals);

        // Smooth transitions — slightly longer animation time for polished feel
        let mut style = (*ctx.style()).clone();
        style.animation_time = 0.15; // 150ms (default ~83ms)

        // Scrollbar: solid style with foreground_color=true for high-contrast handles.
        // Handle uses fg_stroke (text color) instead of bg_fill (button bg) which was
        // near-invisible against the track in our blue-tinted Signal Grid palette.
        let mut scroll = egui::style::ScrollStyle::solid();
        scroll.foreground_color = true;  // handle = fg_stroke (visible text color)
        scroll.bar_width = 8.0;
        scroll.bar_inner_margin = 2.0;
        style.spacing.scroll = scroll;

        ctx.set_style(style);

        // Update native window title bar to match theme (Windows 10+)
        #[cfg(target_os = "windows")]
        set_native_dark_title_bar(matches!(self.mode, ThemeMode::Dark));
    }

    // ====================================================================
    // PHASE 2: Spacing & Typography Constants
    // ====================================================================

    // Spacing scale (8px base unit, matching Signal Grid)
    pub const SPACE_XS: f32 = 4.0;
    pub const SPACE_SM: f32 = 8.0;
    pub const SPACE_MD: f32 = 12.0;
    pub const SPACE_LG: f32 = 16.0;
    pub const SPACE_XL: f32 = 24.0;
    pub const SPACE_2XL: f32 = 32.0;

    // Font sizes
    pub const FONT_LABEL: f32 = 11.0;  // Badges, metadata, mono labels
    pub const FONT_BODY: f32 = 13.0;   // Normal text
    pub const FONT_HEADING: f32 = 14.0; // Panel titles
    pub const FONT_TITLE: f32 = 16.0;  // Dialog titles

    pub fn floating_window_frame(&self) -> egui::Frame {
        let shadow_alpha = match self.mode {
            ThemeMode::Dark => 50,
            ThemeMode::Light => 32,
        };
        egui::Frame::none()
            .fill(self.panel_bg)
            .rounding(Rounding::same(10.0))
            .stroke(Stroke::new(1.0, self.border_color))
            .shadow(Shadow {
                extrusion: 10.0,
                color: Color32::from_black_alpha(shadow_alpha),
            })
            .inner_margin(egui::Margin::same(10.0))
    }

    /// Floating window frame with animated border — call with hover_t from
    /// `ctx.animate_bool(id, is_hovered)` to smoothly transition border color.
    pub fn floating_window_frame_animated(&self, hover_t: f32) -> egui::Frame {
        let shadow_alpha = match self.mode {
            ThemeMode::Dark => 50,
            ThemeMode::Light => 32,
        };
        let border = Self::lerp_color(self.border_color, self.border_lit, hover_t);
        egui::Frame::none()
            .fill(self.panel_bg)
            .rounding(Rounding::same(10.0))
            .stroke(Stroke::new(1.0, border))
            .shadow(Shadow {
                extrusion: 10.0,
                color: Color32::from_black_alpha(shadow_alpha),
            })
            .inner_margin(egui::Margin::same(10.0))
    }

    /// Linearly interpolate between two colors.
    pub fn lerp_color(a: Color32, b: Color32, t: f32) -> Color32 {
        let t = t.clamp(0.0, 1.0);
        let inv = 1.0 - t;
        Color32::from_rgba_unmultiplied(
            (a.r() as f32 * inv + b.r() as f32 * t) as u8,
            (a.g() as f32 * inv + b.g() as f32 * t) as u8,
            (a.b() as f32 * inv + b.b() as f32 * t) as u8,
            (a.a() as f32 * inv + b.a() as f32 * t) as u8,
        )
    }

    /// Dialog frame — larger rounding, stronger shadow, generous padding.
    pub fn signal_dialog_frame(&self) -> egui::Frame {
        let shadow_alpha = match self.mode {
            ThemeMode::Dark => 60,
            ThemeMode::Light => 36,
        };
        egui::Frame::none()
            .fill(self.panel_bg)
            .rounding(Rounding::same(12.0))
            .stroke(Stroke::new(1.0, self.border_color))
            .shadow(Shadow {
                extrusion: 16.0,
                color: Color32::from_black_alpha(shadow_alpha),
            })
            .inner_margin(egui::Margin::same(16.0))
    }

    /// Context bar frame — deepest bg with bottom border line.
    pub fn context_bar_frame(&self) -> egui::Frame {
        egui::Frame::none()
            .fill(self.bg_color)
            .stroke(Stroke::new(1.0, self.border_color))
            .inner_margin(egui::Margin::symmetric(8.0, 6.0))
    }

    pub fn toolbar_frame(&self) -> egui::Frame {
        egui::Frame::none()
            .fill(self.toolbar_bg)
            .inner_margin(egui::Margin::symmetric(8.0, 5.5))
    }

    pub fn menu_frame(&self) -> egui::Frame {
        // No stroke — the accent bottom line is painted separately in app.rs
        // to avoid a 4-sided border from egui's Frame::stroke
        egui::Frame::none()
            .fill(self.bg_color)
            .inner_margin(egui::Margin::symmetric(8.0, 2.0))
    }

    pub fn tab_frame(&self) -> egui::Frame {
        egui::Frame::none()
            .fill(self.bg2)
            .rounding(Rounding::same(8.0))
            .stroke(Stroke::new(1.0, self.border_color))
            .inner_margin(egui::Margin::symmetric(4.0, 2.0))
    }

    /// Active tab fill — accent-tinted bg3 for contrast
    pub fn active_tab_fill(&self) -> Color32 {
        self.bg3
    }

    /// Inactive tab fill
    pub fn inactive_tab_fill(&self) -> Color32 {
        Color32::TRANSPARENT
    }

    /// Text color for active tabs
    pub fn active_tab_text(&self) -> Color32 {
        self.text_color
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
            tools: true,          // Tools always visible by default
            layers: true,         // Layers visible by default
            history: false,       // History hidden by default
            colors: false,        // Colors hidden by default (toggle from swatch)
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
