use eframe::egui;

#[derive(Clone, Debug, PartialEq)]
pub struct KeyCombo {
    pub ctrl: bool,
    pub shift: bool,
    pub alt: bool,
    /// The egui key (for named keys like A, S, F1, etc.)
    pub key: Option<egui::Key>,
    /// Text character (for keys not in egui::Key, like [ ] etc.)
    pub text_char: Option<String>,
}

impl KeyCombo {
    pub fn modifiers_only(ctrl: bool, shift: bool, alt: bool) -> Self {
        Self {
            ctrl,
            shift,
            alt,
            key: None,
            text_char: None,
        }
    }

    pub fn key(k: egui::Key) -> Self {
        Self {
            ctrl: false,
            shift: false,
            alt: false,
            key: Some(k),
            text_char: None,
        }
    }
    pub fn ctrl_key(k: egui::Key) -> Self {
        Self {
            ctrl: true,
            shift: false,
            alt: false,
            key: Some(k),
            text_char: None,
        }
    }
    pub fn ctrl_shift_key(k: egui::Key) -> Self {
        Self {
            ctrl: true,
            shift: true,
            alt: false,
            key: Some(k),
            text_char: None,
        }
    }
    pub fn text(s: &str) -> Self {
        Self {
            ctrl: false,
            shift: false,
            alt: false,
            key: None,
            text_char: Some(s.to_string()),
        }
    }

    /// Human-readable display string
    pub fn display(&self) -> String {
        let mut parts = Vec::new();
        if self.ctrl {
            if cfg!(target_os = "macos") {
                parts.push("\u{2318}");
            } else {
                parts.push("Ctrl");
            }
        }
        if self.shift {
            parts.push("Shift");
        }
        if self.alt {
            if cfg!(target_os = "macos") {
                parts.push("\u{2325}");
            } else {
                parts.push("Alt");
            }
        }
        if let Some(ref k) = self.key {
            parts.push(key_name(*k));
        } else if let Some(ref t) = self.text_char {
            parts.push(t.as_str());
        }
        if parts.is_empty() {
            "—".to_string()
        } else {
            parts.join("+")
        }
    }

    /// Serialize to config string. Returns "none" for unbound (empty) combos.
    pub fn to_config_string(&self) -> String {
        let mut parts = Vec::new();
        if self.ctrl {
            parts.push("ctrl".to_string());
        }
        if self.shift {
            parts.push("shift".to_string());
        }
        if self.alt {
            parts.push("alt".to_string());
        }
        if let Some(ref k) = self.key {
            parts.push(format!("key:{}", key_name(*k)));
        } else if let Some(ref t) = self.text_char {
            parts.push(format!("text:{}", t));
        }
        if parts.is_empty() {
            "none".to_string()
        } else {
            parts.join("+")
        }
    }

    /// Deserialize from config string. Returns `None` for "none" (unbound).
    pub fn from_config_string(s: &str) -> Option<Self> {
        let s = s.trim();
        if s.is_empty() || s.eq_ignore_ascii_case("none") {
            return None;
        }
        let mut combo = Self {
            ctrl: false,
            shift: false,
            alt: false,
            key: None,
            text_char: None,
        };
        for part in s.split('+') {
            let part = part.trim();
            match part {
                "ctrl" => combo.ctrl = true,
                "shift" => combo.shift = true,
                "alt" => combo.alt = true,
                _ => {
                    if let Some(key_name) = part.strip_prefix("key:") {
                        combo.key = parse_key_name(key_name);
                    } else if let Some(text) = part.strip_prefix("text:") {
                        combo.text_char = Some(text.to_string());
                    }
                }
            }
        }
        if combo.key.is_some()
            || combo.text_char.is_some()
            || combo.ctrl
            || combo.shift
            || combo.alt
        {
            Some(combo)
        } else {
            None
        }
    }
}

/// All bindable actions in the application
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BindableAction {
    // File
    NewFile,
    OpenFile,
    CloseProject,
    Save,
    SaveAll,
    SaveAs,
    // Edit
    Undo,
    Redo,
    Copy,
    Cut,
    Paste,
    SelectAll,
    Deselect,
    FlattenLayers,
    // Canvas
    ResizeImage,
    ResizeCanvas,
    // View
    ViewZoomIn,
    ViewZoomOut,
    ViewFitToWindow,
    // Tools
    ToolBrush,
    ToolEraser,
    ToolPencil,
    ToolLine,
    ToolGradient,
    ToolFill,
    ToolMagicWand,
    ToolColorPicker,
    ToolMovePixels,
    ToolRectSelect,
    ToolText,
    ToolZoom,
    ToolPan,
    ToolCloneStamp,
    ToolShapes,
    ToolLasso,
    ToolColorRemover,
    ToolMeshWarp,
    // Brush
    BrushResizeDragModifier,
    BrushSizeDecrease,
    BrushSizeIncrease,
    // Color — instant (no dialog)
    ColorAutoLevels,
    ColorDesaturate,
    ColorInvertColors,
    ColorInvertAlpha,
    ColorSepiaTone,
    // Color — dialog
    ColorBrightnessContrast,
    ColorCurves,
    ColorExposure,
    ColorHighlightsShadows,
    ColorHueSaturation,
    ColorLevels,
    ColorTemperatureTint,
    ColorVibrance,
    ColorThreshold,
    ColorPosterize,
    ColorBalance,
    ColorGradientMap,
    ColorBlackAndWhite,
    // Filter — blur
    FilterGaussianBlur,
    FilterBokehBlur,
    FilterMotionBlur,
    FilterBoxBlur,
    FilterZoomBlur,
    // Filter — sharpen/noise
    FilterSharpen,
    FilterReduceNoise,
    FilterAddNoise,
    FilterMedian,
    // Filter — distort
    FilterCrystallize,
    FilterDents,
    FilterPixelate,
    FilterBulge,
    FilterTwist,
    // Filter — stylize
    FilterGlow,
    FilterVignette,
    FilterHalftone,
    FilterInk,
    FilterOilPainting,
    FilterColorFilter,
    // Filter — glitch
    FilterPixelDrag,
    FilterRgbDisplace,
    // Filter — AI
    FilterRemoveBackground,
    // Generate
    GenerateGrid,
    GenerateDropShadow,
    GenerateOutline,
    GenerateContours,
}

impl BindableAction {
    /// Human-readable name for display
    pub fn display_name(&self) -> String {
        match self {
            Self::NewFile => t!("keybind.new_file"),
            Self::OpenFile => t!("keybind.open_file"),
            Self::CloseProject => t!("keybind.close_file"),
            Self::Save => t!("keybind.save"),
            Self::SaveAll => t!("keybind.save_all"),
            Self::SaveAs => t!("keybind.save_as"),
            Self::Undo => t!("keybind.undo"),
            Self::Redo => t!("keybind.redo"),
            Self::Copy => t!("keybind.copy"),
            Self::Cut => t!("keybind.cut"),
            Self::Paste => t!("keybind.paste"),
            Self::SelectAll => t!("keybind.select_all"),
            Self::Deselect => t!("keybind.deselect"),
            Self::FlattenLayers => t!("keybind.flatten_layers"),
            Self::ResizeImage => t!("keybind.resize_image"),
            Self::ResizeCanvas => t!("keybind.resize_canvas"),
            Self::ViewZoomIn => t!("keybind.view_zoom_in"),
            Self::ViewZoomOut => t!("keybind.view_zoom_out"),
            Self::ViewFitToWindow => t!("keybind.view_fit_to_window"),
            Self::ToolBrush => t!("keybind.tool_brush"),
            Self::ToolEraser => t!("keybind.tool_eraser"),
            Self::ToolPencil => t!("keybind.tool_pencil"),
            Self::ToolLine => t!("keybind.tool_line"),
            Self::ToolGradient => t!("keybind.tool_gradient"),
            Self::ToolFill => t!("keybind.tool_fill"),
            Self::ToolMagicWand => t!("keybind.tool_magic_wand"),
            Self::ToolColorPicker => t!("keybind.tool_color_picker"),
            Self::ToolMovePixels => t!("keybind.tool_move_pixels"),
            Self::ToolRectSelect => t!("keybind.tool_rect_select"),
            Self::ToolText => t!("keybind.tool_text"),
            Self::ToolZoom => t!("keybind.tool_zoom"),
            Self::ToolPan => t!("keybind.tool_pan"),
            Self::ToolCloneStamp => t!("keybind.tool_clone_stamp"),
            Self::ToolShapes => t!("keybind.tool_shapes"),
            Self::ToolLasso => t!("keybind.tool_lasso"),
            Self::ToolColorRemover => t!("keybind.tool_color_remover"),
            Self::ToolMeshWarp => t!("keybind.tool_mesh_warp"),
            Self::BrushResizeDragModifier => t!("keybind.brush_resize_drag_modifier"),
            Self::BrushSizeDecrease => t!("keybind.brush_size_decrease"),
            Self::BrushSizeIncrease => t!("keybind.brush_size_increase"),
            // Color
            Self::ColorAutoLevels => t!("menu.color.auto_levels"),
            Self::ColorDesaturate => t!("menu.color.desaturate"),
            Self::ColorInvertColors => t!("menu.color.invert_colors"),
            Self::ColorInvertAlpha => t!("menu.color.invert_alpha"),
            Self::ColorSepiaTone => t!("menu.color.sepia_tone"),
            Self::ColorBrightnessContrast => t!("menu.color.brightness_contrast"),
            Self::ColorCurves => t!("menu.color.curves"),
            Self::ColorExposure => t!("menu.color.exposure"),
            Self::ColorHighlightsShadows => t!("menu.color.highlights_shadows"),
            Self::ColorHueSaturation => t!("menu.color.hue_saturation"),
            Self::ColorLevels => t!("menu.color.levels"),
            Self::ColorTemperatureTint => t!("menu.color.temperature_tint"),
            Self::ColorVibrance => t!("menu.color.vibrance"),
            Self::ColorThreshold => t!("menu.color.threshold"),
            Self::ColorPosterize => t!("menu.color.posterize"),
            Self::ColorBalance => t!("menu.color.color_balance"),
            Self::ColorGradientMap => t!("menu.color.gradient_map"),
            Self::ColorBlackAndWhite => t!("menu.color.black_and_white"),
            // Filter
            Self::FilterGaussianBlur => t!("menu.filter.blur.gaussian"),
            Self::FilterBokehBlur => t!("menu.filter.blur.bokeh"),
            Self::FilterMotionBlur => t!("menu.filter.blur.motion"),
            Self::FilterBoxBlur => t!("menu.filter.blur.box"),
            Self::FilterZoomBlur => t!("menu.filter.blur.zoom"),
            Self::FilterSharpen => t!("menu.filter.sharpen.sharpen"),
            Self::FilterReduceNoise => t!("menu.filter.sharpen.reduce_noise"),
            Self::FilterAddNoise => t!("menu.filter.noise.add_noise"),
            Self::FilterMedian => t!("menu.filter.noise.median"),
            Self::FilterCrystallize => t!("menu.filter.distort.crystallize"),
            Self::FilterDents => t!("menu.filter.distort.dents"),
            Self::FilterPixelate => t!("menu.filter.distort.pixelate"),
            Self::FilterBulge => t!("menu.filter.distort.bulge_pinch"),
            Self::FilterTwist => t!("menu.filter.distort.twist"),
            Self::FilterGlow => t!("menu.filter.stylize.glow"),
            Self::FilterVignette => t!("menu.filter.stylize.vignette"),
            Self::FilterHalftone => t!("menu.filter.stylize.halftone"),
            Self::FilterInk => t!("menu.filter.stylize.ink"),
            Self::FilterOilPainting => t!("menu.filter.stylize.oil_painting"),
            Self::FilterColorFilter => t!("menu.filter.stylize.color_filter"),
            Self::FilterPixelDrag => t!("menu.filter.glitch.pixel_drag"),
            Self::FilterRgbDisplace => t!("menu.filter.glitch.rgb_displace"),
            Self::FilterRemoveBackground => t!("menu.filter.remove_background"),
            // Generate
            Self::GenerateGrid => t!("menu.generate.grid"),
            Self::GenerateDropShadow => t!("menu.generate.drop_shadow"),
            Self::GenerateOutline => t!("menu.generate.outline"),
            Self::GenerateContours => t!("menu.generate.contours"),
        }
    }

    /// Category for grouping in UI
    pub fn category(&self) -> String {
        match self {
            Self::NewFile
            | Self::OpenFile
            | Self::CloseProject
            | Self::Save
            | Self::SaveAll
            | Self::SaveAs => {
                t!("keybind_category.file")
            }
            Self::Undo
            | Self::Redo
            | Self::Copy
            | Self::Cut
            | Self::Paste
            | Self::SelectAll
            | Self::Deselect
            | Self::FlattenLayers => t!("keybind_category.edit"),
            Self::ResizeImage | Self::ResizeCanvas => t!("keybind_category.canvas"),
            Self::ViewZoomIn | Self::ViewZoomOut | Self::ViewFitToWindow => {
                t!("keybind_category.view")
            }
            Self::ToolBrush
            | Self::ToolEraser
            | Self::ToolPencil
            | Self::ToolLine
            | Self::ToolGradient
            | Self::ToolFill
            | Self::ToolMagicWand
            | Self::ToolColorPicker
            | Self::ToolMovePixels
            | Self::ToolRectSelect
            | Self::ToolText
            | Self::ToolZoom
            | Self::ToolPan
            | Self::ToolCloneStamp
            | Self::ToolShapes
            | Self::ToolLasso
            | Self::ToolColorRemover
            | Self::ToolMeshWarp => t!("keybind_category.tools"),
            Self::BrushResizeDragModifier | Self::BrushSizeDecrease | Self::BrushSizeIncrease => {
                t!("keybind_category.brush")
            }
            Self::ColorAutoLevels
            | Self::ColorDesaturate
            | Self::ColorInvertColors
            | Self::ColorInvertAlpha
            | Self::ColorSepiaTone
            | Self::ColorBrightnessContrast
            | Self::ColorCurves
            | Self::ColorExposure
            | Self::ColorHighlightsShadows
            | Self::ColorHueSaturation
            | Self::ColorLevels
            | Self::ColorTemperatureTint
            | Self::ColorVibrance
            | Self::ColorThreshold
            | Self::ColorPosterize
            | Self::ColorBalance
            | Self::ColorGradientMap
            | Self::ColorBlackAndWhite => t!("keybind_category.color"),
            Self::FilterGaussianBlur
            | Self::FilterBokehBlur
            | Self::FilterMotionBlur
            | Self::FilterBoxBlur
            | Self::FilterZoomBlur
            | Self::FilterSharpen
            | Self::FilterReduceNoise
            | Self::FilterAddNoise
            | Self::FilterMedian
            | Self::FilterCrystallize
            | Self::FilterDents
            | Self::FilterPixelate
            | Self::FilterBulge
            | Self::FilterTwist
            | Self::FilterGlow
            | Self::FilterVignette
            | Self::FilterHalftone
            | Self::FilterInk
            | Self::FilterOilPainting
            | Self::FilterColorFilter
            | Self::FilterPixelDrag
            | Self::FilterRgbDisplace
            | Self::FilterRemoveBackground => t!("keybind_category.filter"),
            Self::GenerateGrid
            | Self::GenerateDropShadow
            | Self::GenerateOutline
            | Self::GenerateContours => t!("keybind_category.generate"),
        }
    }

    /// All actions in display order
    pub fn all() -> &'static [BindableAction] {
        use BindableAction::*;
        &[
            NewFile,
            OpenFile,
            CloseProject,
            Save,
            SaveAll,
            SaveAs,
            Undo,
            Redo,
            Copy,
            Cut,
            Paste,
            SelectAll,
            Deselect,
            FlattenLayers,
            ResizeImage,
            ResizeCanvas,
            ViewZoomIn,
            ViewZoomOut,
            ViewFitToWindow,
            ToolBrush,
            ToolEraser,
            ToolPencil,
            ToolLine,
            ToolGradient,
            ToolFill,
            ToolMagicWand,
            ToolColorPicker,
            ToolMovePixels,
            ToolRectSelect,
            ToolText,
            ToolZoom,
            ToolPan,
            ToolCloneStamp,
            ToolShapes,
            ToolLasso,
            ToolColorRemover,
            ToolMeshWarp,
            BrushResizeDragModifier,
            BrushSizeDecrease,
            BrushSizeIncrease,
            // Color
            ColorAutoLevels,
            ColorDesaturate,
            ColorInvertColors,
            ColorInvertAlpha,
            ColorSepiaTone,
            ColorBrightnessContrast,
            ColorCurves,
            ColorExposure,
            ColorHighlightsShadows,
            ColorHueSaturation,
            ColorLevels,
            ColorTemperatureTint,
            ColorVibrance,
            ColorThreshold,
            ColorPosterize,
            ColorBalance,
            ColorGradientMap,
            ColorBlackAndWhite,
            // Filter
            FilterGaussianBlur,
            FilterBokehBlur,
            FilterMotionBlur,
            FilterBoxBlur,
            FilterZoomBlur,
            FilterSharpen,
            FilterReduceNoise,
            FilterAddNoise,
            FilterMedian,
            FilterCrystallize,
            FilterDents,
            FilterPixelate,
            FilterBulge,
            FilterTwist,
            FilterGlow,
            FilterVignette,
            FilterHalftone,
            FilterInk,
            FilterOilPainting,
            FilterColorFilter,
            FilterPixelDrag,
            FilterRgbDisplace,
            FilterRemoveBackground,
            // Generate
            GenerateGrid,
            GenerateDropShadow,
            GenerateOutline,
            GenerateContours,
        ]
    }
}

/// Keybinding map — stores all customizable keyboard shortcuts
#[derive(Clone, Debug)]
pub struct KeyBindings {
    pub bindings: std::collections::HashMap<BindableAction, KeyCombo>,
    pub unbound: std::collections::HashSet<BindableAction>,
}

impl Default for KeyBindings {
    fn default() -> Self {
        use BindableAction::*;
        use egui::Key;
        let mut map = std::collections::HashMap::new();
        // File
        map.insert(NewFile, KeyCombo::ctrl_key(Key::N));
        map.insert(OpenFile, KeyCombo::ctrl_key(Key::O));
        map.insert(CloseProject, KeyCombo::ctrl_key(Key::W));
        map.insert(Save, KeyCombo::ctrl_key(Key::S));
        map.insert(
            SaveAll,
            KeyCombo {
                ctrl: true,
                shift: false,
                alt: true,
                key: Some(Key::S),
                text_char: None,
            },
        );
        map.insert(SaveAs, KeyCombo::ctrl_shift_key(Key::S));
        // Edit
        map.insert(Undo, KeyCombo::ctrl_key(Key::Z));
        map.insert(Redo, KeyCombo::ctrl_key(Key::Y));
        map.insert(Copy, KeyCombo::ctrl_key(Key::C));
        map.insert(Cut, KeyCombo::ctrl_key(Key::X));
        map.insert(Paste, KeyCombo::ctrl_key(Key::V));
        map.insert(SelectAll, KeyCombo::ctrl_key(Key::A));
        map.insert(Deselect, KeyCombo::ctrl_key(Key::D));
        map.insert(FlattenLayers, KeyCombo::ctrl_shift_key(Key::F));
        // Canvas
        map.insert(ResizeImage, KeyCombo::ctrl_key(Key::R));
        map.insert(ResizeCanvas, KeyCombo::ctrl_shift_key(Key::R));
        // View
        map.insert(ViewZoomIn, KeyCombo::ctrl_key(Key::Equals));
        map.insert(ViewZoomOut, KeyCombo::ctrl_key(Key::Minus));
        map.insert(ViewFitToWindow, KeyCombo::ctrl_key(Key::Num0));
        // Tools
        map.insert(ToolBrush, KeyCombo::key(Key::B));
        map.insert(ToolEraser, KeyCombo::key(Key::E));
        map.insert(ToolPencil, KeyCombo::key(Key::P));
        map.insert(ToolLine, KeyCombo::key(Key::L));
        map.insert(ToolGradient, KeyCombo::key(Key::G));
        map.insert(ToolFill, KeyCombo::key(Key::F));
        map.insert(ToolMagicWand, KeyCombo::key(Key::W));
        map.insert(ToolColorPicker, KeyCombo::key(Key::I));
        map.insert(ToolMovePixels, KeyCombo::key(Key::M));
        map.insert(ToolRectSelect, KeyCombo::key(Key::S));
        map.insert(ToolText, KeyCombo::key(Key::T));
        map.insert(ToolZoom, KeyCombo::key(Key::Z));
        map.insert(ToolPan, KeyCombo::key(Key::H));
        map.insert(ToolCloneStamp, KeyCombo::key(Key::K));
        map.insert(ToolShapes, KeyCombo::key(Key::U));
        map.insert(ToolLasso, KeyCombo::key(Key::J));
        map.insert(ToolColorRemover, KeyCombo::key(Key::R));
        map.insert(ToolMeshWarp, KeyCombo::key(Key::Q));
        // Brush size
        map.insert(
            BrushResizeDragModifier,
            KeyCombo::modifiers_only(false, true, false),
        );
        map.insert(BrushSizeDecrease, KeyCombo::text("["));
        map.insert(BrushSizeIncrease, KeyCombo::text("]"));

        Self {
            bindings: map,
            unbound: std::collections::HashSet::new(),
        }
    }
}

impl KeyBindings {
    /// Returns the combo for an action, falling back to default if missing.
    pub fn get(&self, action: BindableAction) -> Option<&KeyCombo> {
        self.bindings.get(&action)
    }

    /// Set a binding
    pub fn set(&mut self, action: BindableAction, combo: KeyCombo) {
        if combo.key.is_none() && combo.text_char.is_none() && !combo.ctrl && !combo.shift && !combo.alt
        {
            self.bindings.remove(&action);
            self.unbound.insert(action);
        } else {
            self.unbound.remove(&action);
            self.bindings.insert(action, combo);
        }
    }

    /// Serialize all bindings for config file
    pub fn to_config_lines(&self) -> Vec<String> {
        let mut lines = Vec::new();
        for action in BindableAction::all() {
            if self.unbound.contains(action) {
                lines.push(format!("keybind.{:?}=none", action));
            } else if let Some(combo) = self.bindings.get(action) {
                lines.push(format!("keybind.{:?}={}", action, combo.to_config_string()));
            }
        }
        lines
    }

    /// Load a single keybind line from config.
    /// If combo_str is "none", the binding is removed (unbound).
    pub fn load_config_line(&mut self, action_name: &str, combo_str: &str) {
        let action = match action_name {
            "NewFile" => Some(BindableAction::NewFile),
            "OpenFile" => Some(BindableAction::OpenFile),
            "CloseProject" => Some(BindableAction::CloseProject),
            "Save" => Some(BindableAction::Save),
            "SaveAll" => Some(BindableAction::SaveAll),
            "SaveAs" => Some(BindableAction::SaveAs),
            "Undo" => Some(BindableAction::Undo),
            "Redo" => Some(BindableAction::Redo),
            "Copy" => Some(BindableAction::Copy),
            "Cut" => Some(BindableAction::Cut),
            "Paste" => Some(BindableAction::Paste),
            "SelectAll" => Some(BindableAction::SelectAll),
            "Deselect" => Some(BindableAction::Deselect),
            "FlattenLayers" => Some(BindableAction::FlattenLayers),
            "ResizeImage" => Some(BindableAction::ResizeImage),
            "ResizeCanvas" => Some(BindableAction::ResizeCanvas),
            "ViewZoomIn" => Some(BindableAction::ViewZoomIn),
            "ViewZoomOut" => Some(BindableAction::ViewZoomOut),
            "ViewFitToWindow" => Some(BindableAction::ViewFitToWindow),
            "ToolBrush" => Some(BindableAction::ToolBrush),
            "ToolEraser" => Some(BindableAction::ToolEraser),
            "ToolPencil" => Some(BindableAction::ToolPencil),
            "ToolLine" => Some(BindableAction::ToolLine),
            "ToolGradient" => Some(BindableAction::ToolGradient),
            "ToolFill" => Some(BindableAction::ToolFill),
            "ToolMagicWand" => Some(BindableAction::ToolMagicWand),
            "ToolColorPicker" => Some(BindableAction::ToolColorPicker),
            "ToolMovePixels" => Some(BindableAction::ToolMovePixels),
            "ToolRectSelect" => Some(BindableAction::ToolRectSelect),
            "ToolText" => Some(BindableAction::ToolText),
            "ToolZoom" => Some(BindableAction::ToolZoom),
            "ToolPan" => Some(BindableAction::ToolPan),
            "ToolCloneStamp" => Some(BindableAction::ToolCloneStamp),
            "ToolShapes" => Some(BindableAction::ToolShapes),
            "ToolLasso" => Some(BindableAction::ToolLasso),
            "ToolColorRemover" => Some(BindableAction::ToolColorRemover),
            "ToolMeshWarp" => Some(BindableAction::ToolMeshWarp),
            "BrushResizeDragModifier" => Some(BindableAction::BrushResizeDragModifier),
            "BrushSizeDecrease" => Some(BindableAction::BrushSizeDecrease),
            "BrushSizeIncrease" => Some(BindableAction::BrushSizeIncrease),
            "ColorAutoLevels" => Some(BindableAction::ColorAutoLevels),
            "ColorDesaturate" => Some(BindableAction::ColorDesaturate),
            "ColorInvertColors" => Some(BindableAction::ColorInvertColors),
            "ColorInvertAlpha" => Some(BindableAction::ColorInvertAlpha),
            "ColorSepiaTone" => Some(BindableAction::ColorSepiaTone),
            "ColorBrightnessContrast" => Some(BindableAction::ColorBrightnessContrast),
            "ColorCurves" => Some(BindableAction::ColorCurves),
            "ColorExposure" => Some(BindableAction::ColorExposure),
            "ColorHighlightsShadows" => Some(BindableAction::ColorHighlightsShadows),
            "ColorHueSaturation" => Some(BindableAction::ColorHueSaturation),
            "ColorLevels" => Some(BindableAction::ColorLevels),
            "ColorTemperatureTint" => Some(BindableAction::ColorTemperatureTint),
            "ColorVibrance" => Some(BindableAction::ColorVibrance),
            "ColorThreshold" => Some(BindableAction::ColorThreshold),
            "ColorPosterize" => Some(BindableAction::ColorPosterize),
            "ColorBalance" => Some(BindableAction::ColorBalance),
            "ColorGradientMap" => Some(BindableAction::ColorGradientMap),
            "ColorBlackAndWhite" => Some(BindableAction::ColorBlackAndWhite),
            "FilterGaussianBlur" => Some(BindableAction::FilterGaussianBlur),
            "FilterBokehBlur" => Some(BindableAction::FilterBokehBlur),
            "FilterMotionBlur" => Some(BindableAction::FilterMotionBlur),
            "FilterBoxBlur" => Some(BindableAction::FilterBoxBlur),
            "FilterZoomBlur" => Some(BindableAction::FilterZoomBlur),
            "FilterSharpen" => Some(BindableAction::FilterSharpen),
            "FilterReduceNoise" => Some(BindableAction::FilterReduceNoise),
            "FilterAddNoise" => Some(BindableAction::FilterAddNoise),
            "FilterMedian" => Some(BindableAction::FilterMedian),
            "FilterCrystallize" => Some(BindableAction::FilterCrystallize),
            "FilterDents" => Some(BindableAction::FilterDents),
            "FilterPixelate" => Some(BindableAction::FilterPixelate),
            "FilterBulge" => Some(BindableAction::FilterBulge),
            "FilterTwist" => Some(BindableAction::FilterTwist),
            "FilterGlow" => Some(BindableAction::FilterGlow),
            "FilterVignette" => Some(BindableAction::FilterVignette),
            "FilterHalftone" => Some(BindableAction::FilterHalftone),
            "FilterInk" => Some(BindableAction::FilterInk),
            "FilterOilPainting" => Some(BindableAction::FilterOilPainting),
            "FilterColorFilter" => Some(BindableAction::FilterColorFilter),
            "FilterPixelDrag" => Some(BindableAction::FilterPixelDrag),
            "FilterRgbDisplace" => Some(BindableAction::FilterRgbDisplace),
            "FilterRemoveBackground" => Some(BindableAction::FilterRemoveBackground),
            "GenerateGrid" => Some(BindableAction::GenerateGrid),
            "GenerateDropShadow" => Some(BindableAction::GenerateDropShadow),
            "GenerateOutline" => Some(BindableAction::GenerateOutline),
            "GenerateContours" => Some(BindableAction::GenerateContours),
            _ => None,
        };
        let Some(action) = action else {
            return;
        };
        let trimmed = combo_str.trim();
        if trimmed.eq_ignore_ascii_case("none") || trimmed.is_empty() {
            self.bindings.remove(&action);
            self.unbound.insert(action);
        } else if let Some(combo) = KeyCombo::from_config_string(combo_str) {
            self.unbound.remove(&action);
            self.bindings.insert(action, combo);
        }
    }

    /// Check if a keybinding was triggered this frame
    pub fn is_pressed(&self, ctx: &egui::Context, action: BindableAction) -> bool {
        let Some(combo) = self.bindings.get(&action) else {
            return false;
        };

        if let Some(ref text_char) = combo.text_char {
            // Text-based binding (like [ or ])
            ctx.input_mut(|i| {
                // Check modifier match
                let command_or_ctrl = i.modifiers.command || i.modifiers.ctrl;
                if combo.ctrl != command_or_ctrl {
                    return false;
                }
                if combo.shift != i.modifiers.shift {
                    return false;
                }
                if combo.alt != i.modifiers.alt {
                    return false;
                }
                let mut found = false;
                i.events.retain(|ev| {
                    if !found
                        && let egui::Event::Text(t) = ev
                        && t == text_char
                    {
                        found = true;
                        return false; // consume the event
                    }
                    true
                });
                found
            })
        } else if let Some(key) = combo.key {
            #[cfg(target_os = "windows")]
            {
                if let Some(vk) = egui_key_to_windows_vk(key) {
                    let win_mod_match = ctx.input(|i| {
                        let event_ctrl = i.modifiers.command || i.modifiers.ctrl;
                        combo.ctrl == event_ctrl
                            && combo.shift == i.modifiers.shift
                            && combo.alt == i.modifiers.alt
                    });
                    let is_down_now = win_mod_match && crate::windows_key_probe::is_vk_down(vk);
                    let edge_id = egui::Id::new(format!("kb_win_edge_{:?}", action));
                    let was_down = ctx.data_mut(|d| d.get_temp::<bool>(edge_id).unwrap_or(false));
                    ctx.data_mut(|d| d.insert_temp(edge_id, is_down_now));
                    if is_down_now && !was_down {
                        return true;
                    }
                }
            }

            // Primary path: state-based edge detection. This is resilient when
            // backend event streams are inconsistent but key-down state is valid.
            let is_down_now = ctx.input(|i| {
                let event_ctrl = i.modifiers.command || i.modifiers.ctrl;
                i.key_down(key)
                    && combo.ctrl == event_ctrl
                    && combo.shift == i.modifiers.shift
                    && combo.alt == i.modifiers.alt
            });
            let edge_id = egui::Id::new(format!("kb_edge_{:?}", action));
            let was_down = ctx.data_mut(|d| d.get_temp::<bool>(edge_id).unwrap_or(false));
            ctx.data_mut(|d| d.insert_temp(edge_id, is_down_now));
            if is_down_now && !was_down {
                return true;
            }

            ctx.input_mut(|i| {
                // Prefer event-level matching so we can normalize Ctrl/Command
                // consistently across backends and keyboard layouts.
                let mut found = false;
                i.events.retain(|ev| {
                    if found {
                        return true;
                    }
                    if let egui::Event::Key {
                        key: pressed_key,
                        pressed,
                        modifiers,
                        ..
                    } = ev
                    {
                        let event_ctrl = modifiers.command || modifiers.ctrl;
                        if *pressed
                            && *pressed_key == key
                            && combo.ctrl == event_ctrl
                            && combo.shift == modifiers.shift
                            && combo.alt == modifiers.alt
                        {
                            found = true;
                            return false; // consume the matched key event
                        }
                    }
                    true
                });
                if found {
                    return true;
                }

                // Fallback to egui consume_key for platforms/backends that don't
                // emit the expected key event shape.
                let mods = egui::Modifiers {
                    alt: combo.alt,
                    ctrl: if cfg!(target_os = "macos") {
                        false
                    } else {
                        combo.ctrl
                    },
                    shift: combo.shift,
                    mac_cmd: if cfg!(target_os = "macos") {
                        combo.ctrl
                    } else {
                        false
                    },
                    command: combo.ctrl,
                };
                i.consume_key(mods, key)
                    || (!cfg!(target_os = "macos")
                        && combo.ctrl
                        && i.consume_key(
                            egui::Modifiers {
                                alt: combo.alt,
                                ctrl: true,
                                shift: combo.shift,
                                mac_cmd: false,
                                command: false,
                            },
                            key,
                        ))
            })
        } else {
            false
        }
    }
}

#[cfg(target_os = "windows")]
fn egui_key_to_windows_vk(key: egui::Key) -> Option<usize> {
    Some(match key {
        egui::Key::A => 0x41,
        egui::Key::B => 0x42,
        egui::Key::C => 0x43,
        egui::Key::D => 0x44,
        egui::Key::E => 0x45,
        egui::Key::F => 0x46,
        egui::Key::G => 0x47,
        egui::Key::H => 0x48,
        egui::Key::I => 0x49,
        egui::Key::J => 0x4A,
        egui::Key::K => 0x4B,
        egui::Key::L => 0x4C,
        egui::Key::M => 0x4D,
        egui::Key::N => 0x4E,
        egui::Key::O => 0x4F,
        egui::Key::P => 0x50,
        egui::Key::Q => 0x51,
        egui::Key::R => 0x52,
        egui::Key::S => 0x53,
        egui::Key::T => 0x54,
        egui::Key::U => 0x55,
        egui::Key::V => 0x56,
        egui::Key::W => 0x57,
        egui::Key::X => 0x58,
        egui::Key::Y => 0x59,
        egui::Key::Z => 0x5A,
        egui::Key::Num0 => 0x30,
        egui::Key::Num1 => 0x31,
        egui::Key::Num2 => 0x32,
        egui::Key::Num3 => 0x33,
        egui::Key::Num4 => 0x34,
        egui::Key::Num5 => 0x35,
        egui::Key::Num6 => 0x36,
        egui::Key::Num7 => 0x37,
        egui::Key::Num8 => 0x38,
        egui::Key::Num9 => 0x39,
        egui::Key::Minus => 0xBD,
        egui::Key::Equals => 0xBB,
        egui::Key::Enter => 0x0D,
        egui::Key::Escape => 0x1B,
        _ => return None,
    })
}

/// Convert egui::Key to display name
pub(crate) fn key_name(k: egui::Key) -> &'static str {
    match k {
        egui::Key::ArrowDown => "Down",
        egui::Key::ArrowLeft => "Left",
        egui::Key::ArrowRight => "Right",
        egui::Key::ArrowUp => "Up",
        egui::Key::Escape => "Esc",
        egui::Key::Tab => "Tab",
        egui::Key::Backspace => "Backspace",
        egui::Key::Enter => "Enter",
        egui::Key::Space => "Space",
        egui::Key::Insert => "Insert",
        egui::Key::Delete => "Delete",
        egui::Key::Home => "Home",
        egui::Key::End => "End",
        egui::Key::PageUp => "PageUp",
        egui::Key::PageDown => "PageDown",
        egui::Key::Minus => "-",
        egui::Key::Equals => "+",
        egui::Key::Num0 => "0",
        egui::Key::Num1 => "1",
        egui::Key::Num2 => "2",
        egui::Key::Num3 => "3",
        egui::Key::Num4 => "4",
        egui::Key::Num5 => "5",
        egui::Key::Num6 => "6",
        egui::Key::Num7 => "7",
        egui::Key::Num8 => "8",
        egui::Key::Num9 => "9",
        egui::Key::A => "A",
        egui::Key::B => "B",
        egui::Key::C => "C",
        egui::Key::D => "D",
        egui::Key::E => "E",
        egui::Key::F => "F",
        egui::Key::G => "G",
        egui::Key::H => "H",
        egui::Key::I => "I",
        egui::Key::J => "J",
        egui::Key::K => "K",
        egui::Key::L => "L",
        egui::Key::M => "M",
        egui::Key::N => "N",
        egui::Key::O => "O",
        egui::Key::P => "P",
        egui::Key::Q => "Q",
        egui::Key::R => "R",
        egui::Key::S => "S",
        egui::Key::T => "T",
        egui::Key::U => "U",
        egui::Key::V => "V",
        egui::Key::W => "W",
        egui::Key::X => "X",
        egui::Key::Y => "Y",
        egui::Key::Z => "Z",
        egui::Key::F1 => "F1",
        egui::Key::F2 => "F2",
        egui::Key::F3 => "F3",
        egui::Key::F4 => "F4",
        egui::Key::F5 => "F5",
        egui::Key::F6 => "F6",
        egui::Key::F7 => "F7",
        egui::Key::F8 => "F8",
        egui::Key::F9 => "F9",
        egui::Key::F10 => "F10",
        egui::Key::F11 => "F11",
        egui::Key::F12 => "F12",
        egui::Key::F13 => "F13",
        egui::Key::F14 => "F14",
        egui::Key::F15 => "F15",
        egui::Key::F16 => "F16",
        egui::Key::F17 => "F17",
        egui::Key::F18 => "F18",
        egui::Key::F19 => "F19",
        egui::Key::F20 => "F20",
        _ => "?",
    }
}

/// Parse a key name string back to egui::Key
fn parse_key_name(s: &str) -> Option<egui::Key> {
    match s {
        "Down" => Some(egui::Key::ArrowDown),
        "Left" => Some(egui::Key::ArrowLeft),
        "Right" => Some(egui::Key::ArrowRight),
        "Up" => Some(egui::Key::ArrowUp),
        "Esc" => Some(egui::Key::Escape),
        "Tab" => Some(egui::Key::Tab),
        "Backspace" => Some(egui::Key::Backspace),
        "Enter" => Some(egui::Key::Enter),
        "Space" => Some(egui::Key::Space),
        "Insert" => Some(egui::Key::Insert),
        "Delete" => Some(egui::Key::Delete),
        "Home" => Some(egui::Key::Home),
        "End" => Some(egui::Key::End),
        "PageUp" => Some(egui::Key::PageUp),
        "PageDown" => Some(egui::Key::PageDown),
        "-" => Some(egui::Key::Minus),
        "+" => Some(egui::Key::Equals),
        "0" => Some(egui::Key::Num0),
        "1" => Some(egui::Key::Num1),
        "2" => Some(egui::Key::Num2),
        "3" => Some(egui::Key::Num3),
        "4" => Some(egui::Key::Num4),
        "5" => Some(egui::Key::Num5),
        "6" => Some(egui::Key::Num6),
        "7" => Some(egui::Key::Num7),
        "8" => Some(egui::Key::Num8),
        "9" => Some(egui::Key::Num9),
        "A" => Some(egui::Key::A),
        "B" => Some(egui::Key::B),
        "C" => Some(egui::Key::C),
        "D" => Some(egui::Key::D),
        "E" => Some(egui::Key::E),
        "F" => Some(egui::Key::F),
        "G" => Some(egui::Key::G),
        "H" => Some(egui::Key::H),
        "I" => Some(egui::Key::I),
        "J" => Some(egui::Key::J),
        "K" => Some(egui::Key::K),
        "L" => Some(egui::Key::L),
        "M" => Some(egui::Key::M),
        "N" => Some(egui::Key::N),
        "O" => Some(egui::Key::O),
        "P" => Some(egui::Key::P),
        "Q" => Some(egui::Key::Q),
        "R" => Some(egui::Key::R),
        "S" => Some(egui::Key::S),
        "T" => Some(egui::Key::T),
        "U" => Some(egui::Key::U),
        "V" => Some(egui::Key::V),
        "W" => Some(egui::Key::W),
        "X" => Some(egui::Key::X),
        "Y" => Some(egui::Key::Y),
        "Z" => Some(egui::Key::Z),
        "F1" => Some(egui::Key::F1),
        "F2" => Some(egui::Key::F2),
        "F3" => Some(egui::Key::F3),
        "F4" => Some(egui::Key::F4),
        "F5" => Some(egui::Key::F5),
        "F6" => Some(egui::Key::F6),
        "F7" => Some(egui::Key::F7),
        "F8" => Some(egui::Key::F8),
        "F9" => Some(egui::Key::F9),
        "F10" => Some(egui::Key::F10),
        "F11" => Some(egui::Key::F11),
        "F12" => Some(egui::Key::F12),
        "F13" => Some(egui::Key::F13),
        "F14" => Some(egui::Key::F14),
        "F15" => Some(egui::Key::F15),
        "F16" => Some(egui::Key::F16),
        "F17" => Some(egui::Key::F17),
        "F18" => Some(egui::Key::F18),
        "F19" => Some(egui::Key::F19),
        "F20" => Some(egui::Key::F20),
        _ => None,
    }
}
