use eframe::egui;
use egui::{ColorImage, Color32, Sense, TextureHandle, TextureOptions, Vec2};
use std::collections::HashMap;
use std::path::PathBuf;
use crate::theme::{ThemeMode, ThemePreset, AccentColors};
use crate::ops::shapes::ShapeKind;

/// Icon identifiers for the asset system
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Icon {
    // === Tools ===
    Brush,
    Eraser,
    Pencil,
    Line,
    Fill,
    RectSelect,
    EllipseSelect,
    Lasso,
    MagicWand,
    MovePixels,
    MoveSelection,
    PerspectiveCrop,
    ColorPicker,
    CloneStamp,
    Zoom,
    Pan,
    Gradient,
    Text,
    ContentAwareBrush,
    Liquify,
    MeshWarp,
    ColorRemover,
    Smudge,
    Shapes,

    // === UI (toolbar, panels, misc) ===
    Undo,
    Redo,
    ZoomIn,
    ZoomOut,
    ResetZoom,
    Grid,
    GridOn,
    GridOff,
    GuidesOn,
    GuidesOff,
    MirrorOff,
    MirrorH,
    MirrorV,
    MirrorQ,
    Settings,
    Layers,
    Visible,
    Hidden,
    New,
    Open,
    Save,
    Close,
    NewLayer,
    Delete,
    Duplicate,
    Flatten,
    MergeDown,
    MergeDownAsMask,
    Peek,
    SwapColors,
    CopyHex,
    Commit,
    ResetCancel,
    Expand,
    Collapse,
    Info,
    Search,
    ClearSearch,
    MoveUp,
    MoveDown,
    MoveTop,
    MoveBottom,
    ImportLayer,
    Rename,
    LayerFlipH,
    LayerFlipV,
    LayerRotate,
    LayerProperties,
    LayerAdd,
    LayerDelete,
    LayerDuplicate,
    CurrentMarker,
    ApplyPrimary,
    SoloLayer,
    HideAll,
    ShowAll,

    // === Settings Tabs ===
    SettingsGeneral,
    SettingsInterface,
    SettingsHardware,
    SettingsKeybinds,
    SettingsAI,

    // === Menu: File ===
    MenuFileNew,
    MenuFileOpen,
    MenuFileSave,
    MenuFileSaveAll,
    MenuFileSaveAs,
    MenuFilePrint,
    MenuFileQuit,

    // === Menu: Edit ===
    MenuEditUndo,
    MenuEditRedo,
    MenuEditCut,
    MenuEditCopy,
    MenuEditPaste,
    MenuEditPasteLayer,
    MenuEditSelectAll,
    MenuEditDeselect,
    MenuEditInvertSel,
    MenuEditColorRange,
    MenuEditPreferences,

    // === Menu: Canvas ===
    MenuCanvasResize,
    MenuCanvasCrop,
    MenuCanvasFlipH,
    MenuCanvasFlipV,
    MenuCanvasRotateCw,
    MenuCanvasRotateCcw,
    MenuCanvasRotate180,
    MenuCanvasFlatten,

    // === Menu: Color ===
    MenuColorAutoLevels,
    MenuColorDesaturate,
    MenuColorInvert,
    MenuColorInvertAlpha,
    MenuColorSepia,
    MenuColorBrightness,
    MenuColorCurves,
    MenuColorExposure,
    MenuColorHighlights,
    MenuColorHsl,
    MenuColorLevels,
    MenuColorTemperature,

    // === Menu: Filter ===
    MenuFilterBlur,
    MenuFilterGaussian,
    MenuFilterBokeh,
    MenuFilterMotionBlur,
    MenuFilterBoxBlur,
    MenuFilterZoomBlur,
    MenuFilterSharpen,
    MenuFilterSharpenItem,
    MenuFilterReduceNoise,
    MenuFilterDistort,
    MenuFilterCrystallize,
    MenuFilterDents,
    MenuFilterPixelate,
    MenuFilterBulge,
    MenuFilterTwist,
    MenuFilterNoise,
    MenuFilterAddNoise,
    MenuFilterMedian,
    MenuFilterStylize,
    MenuFilterGlow,
    MenuFilterVignette,
    MenuFilterHalftone,
    MenuFilterInk,
    MenuFilterOilPainting,
    MenuFilterColorFilter,
    MenuFilterGlitch,
    MenuFilterPixelDrag,
    MenuFilterRgbDisplace,
    MenuFilterRemoveBg,

    // === Menu: Generate ===
    MenuGenerateGrid,
    MenuGenerateShadow,
    MenuGenerateOutline,
    MenuGenerateContours,

    // === Menu: View ===
    MenuViewZoomIn,
    MenuViewZoomOut,
    MenuViewFitWindow,
    MenuViewThemeLight,
    MenuViewThemeDark,

    // === UI: Context bar ===
    UiBrushDynamics,
}

impl Icon {
    /// Get the emoji fallback for this icon (used when textures aren't available)
    pub fn emoji(&self) -> &'static str {
        match self {
            // Tools
            Icon::Brush => "[B]",
            Icon::Eraser => "[E]",
            Icon::Pencil => "[P]",
            Icon::Line => "[L]",
            Icon::Fill => "[F]",
            Icon::RectSelect => "[R]",
            Icon::EllipseSelect => "[O]",
            Icon::Lasso => "[L]",
            Icon::MagicWand => "[W]",
            Icon::MovePixels => "[M]",
            Icon::MoveSelection => "[S]",
            Icon::PerspectiveCrop => "[P]",
            Icon::ColorPicker => "[C]",
            Icon::CloneStamp => "[S]",
            Icon::Zoom => "[Z]",
            Icon::Pan => "[H]",
            Icon::Gradient => "[G]",
            Icon::Text => "[T]",
            Icon::ContentAwareBrush => "[A]",
            Icon::Liquify => "[Q]",
            Icon::MeshWarp => "[W]",
            Icon::ColorRemover => "[R]",
            Icon::Smudge => "[Sm]",
            Icon::Shapes => "[Sh]",
            // UI
            Icon::Undo => "<-",
            Icon::Redo => "->",
            Icon::ZoomIn => "[+]",
            Icon::ZoomOut => "[-]",
            Icon::ResetZoom => "1:1",
            Icon::Grid => "[#]",
            Icon::GridOn => "[#]",
            Icon::GridOff => "[ ]",
            Icon::GuidesOn => "[+]",
            Icon::GuidesOff => "[-]",
            Icon::MirrorOff => "[|]",
            Icon::MirrorH => "[|]",
            Icon::MirrorV => "[-]",
            Icon::MirrorQ => "[+]",
            Icon::Settings => "\u{2699}",
            Icon::Layers => "[=]",
            Icon::Visible => "\u{1F441}",
            Icon::Hidden => "\u{25CB}",
            Icon::New => "[N]",
            Icon::Open => "[O]",
            Icon::Save => "[S]",
            Icon::Close => "\u{00D7}",
            Icon::NewLayer => "[+]",
            Icon::Delete => "[X]",
            Icon::Duplicate => "[D]",
            Icon::Flatten => "[F]",
            Icon::MergeDown => "[v]",
            Icon::MergeDownAsMask => "[vm]",
            Icon::Peek => "\u{1F50D}",
            Icon::SettingsGeneral => "[G]",
            Icon::SettingsInterface => "[I]",
            Icon::SettingsHardware => "[H]",
            Icon::SettingsKeybinds => "[K]",
            Icon::SettingsAI => "[A]",
            Icon::SwapColors => "[<>]",
            Icon::CopyHex => "\u{2398}",
            Icon::Commit => "\u{2714}",
            Icon::ResetCancel => "\u{2718}",
            Icon::Expand => "\u{25B8}",
            Icon::Collapse => "\u{25C2}",
            Icon::Info => "\u{2139}",
            Icon::Search => "\u{1F50D}",
            Icon::ClearSearch => "\u{00D7}",
            Icon::MoveUp => "\u{2191}",
            Icon::MoveDown => "\u{2193}",
            Icon::MoveTop => "\u{21E7}",
            Icon::MoveBottom => "\u{21E9}",
            Icon::ImportLayer => "\u{1F4C2}",
            Icon::Rename => "\u{270F}",
            Icon::LayerFlipH => "\u{2194}",
            Icon::LayerFlipV => "\u{2195}",
            Icon::LayerRotate => "\u{1F504}",
            Icon::LayerProperties => "\u{2699}",
            Icon::LayerAdd => "\u{2795}",
            Icon::LayerDelete => "\u{2796}",
            Icon::LayerDuplicate => "\u{1F4CB}",
            Icon::CurrentMarker => "\u{25B6}",
            Icon::ApplyPrimary => "\u{1F3A8}",
            Icon::SoloLayer => "\u{1F4CD}",
            Icon::HideAll => "\u{1F6AB}",
            Icon::ShowAll => "\u{1F441}",
            // Menu: File
            Icon::MenuFileNew => "\u{1F4C4}",
            Icon::MenuFileOpen => "\u{1F4C2}",
            Icon::MenuFileSave => "\u{1F4BE}",
            Icon::MenuFileSaveAll => "\u{1F4BE}",
            Icon::MenuFileSaveAs => "\u{1F4BE}",
            Icon::MenuFilePrint => "\u{1F5A8}",
            Icon::MenuFileQuit => "\u{274C}",
            // Menu: Edit
            Icon::MenuEditUndo => "\u{21A9}",
            Icon::MenuEditRedo => "\u{21AA}",
            Icon::MenuEditCut => "\u{2702}",
            Icon::MenuEditCopy => "\u{1F4CB}",
            Icon::MenuEditPaste => "\u{1F4CC}",
            Icon::MenuEditPasteLayer => "\u{1F4CC}",
            Icon::MenuEditSelectAll => "\u{2B1C}",
            Icon::MenuEditDeselect => "\u{2B1B}",
            Icon::MenuEditInvertSel => "\u{1F504}",
            Icon::MenuEditColorRange => "\u{1F308}",
            Icon::MenuEditPreferences => "\u{2699}",
            // Menu: Canvas
            Icon::MenuCanvasResize => "\u{1F4D0}",
            Icon::MenuCanvasCrop => "\u{2702}",
            Icon::MenuCanvasFlipH => "\u{2194}",
            Icon::MenuCanvasFlipV => "\u{2195}",
            Icon::MenuCanvasRotateCw => "\u{21BB}",
            Icon::MenuCanvasRotateCcw => "\u{21BA}",
            Icon::MenuCanvasRotate180 => "\u{1F504}",
            Icon::MenuCanvasFlatten => "\u{229F}",
            // Menu: Color
            Icon::MenuColorAutoLevels => "\u{1F532}",
            Icon::MenuColorDesaturate => "\u{25D1}",
            Icon::MenuColorInvert => "\u{1F503}",
            Icon::MenuColorInvertAlpha => "\u{1F532}",
            Icon::MenuColorSepia => "\u{1F4DC}",
            Icon::MenuColorBrightness => "\u{2600}",
            Icon::MenuColorCurves => "\u{1F4C8}",
            Icon::MenuColorExposure => "\u{1F4F7}",
            Icon::MenuColorHighlights => "\u{25D0}",
            Icon::MenuColorHsl => "\u{1F3A8}",
            Icon::MenuColorLevels => "\u{1F4CA}",
            Icon::MenuColorTemperature => "\u{1F321}",
            // Menu: Filter
            Icon::MenuFilterBlur => "\u{1F4A7}",
            Icon::MenuFilterGaussian => "\u{1F4A7}",
            Icon::MenuFilterBokeh => "\u{2B55}",
            Icon::MenuFilterMotionBlur => "\u{27A1}",
            Icon::MenuFilterBoxBlur => "\u{25A3}",
            Icon::MenuFilterZoomBlur => "\u{25CE}",
            Icon::MenuFilterSharpen => "\u{1F4CC}",
            Icon::MenuFilterSharpenItem => "\u{1F4CC}",
            Icon::MenuFilterReduceNoise => "\u{1F50A}",
            Icon::MenuFilterDistort => "\u{1F300}",
            Icon::MenuFilterCrystallize => "\u{1F48E}",
            Icon::MenuFilterDents => "\u{1F30A}",
            Icon::MenuFilterPixelate => "\u{1F9E9}",
            Icon::MenuFilterBulge => "\u{1F534}",
            Icon::MenuFilterTwist => "\u{1F300}",
            Icon::MenuFilterNoise => "\u{1F4A5}",
            Icon::MenuFilterAddNoise => "\u{1F4A5}",
            Icon::MenuFilterMedian => "\u{1F4CA}",
            Icon::MenuFilterStylize => "\u{2728}",
            Icon::MenuFilterGlow => "\u{2728}",
            Icon::MenuFilterVignette => "\u{1F311}",
            Icon::MenuFilterHalftone => "\u{25CF}",
            Icon::MenuFilterInk => "\u{1F58B}",
            Icon::MenuFilterOilPainting => "\u{1F3A8}",
            Icon::MenuFilterColorFilter => "\u{1F3AD}",
            Icon::MenuFilterGlitch => "\u{26A1}",
            Icon::MenuFilterPixelDrag => "\u{1F4A2}",
            Icon::MenuFilterRgbDisplace => "\u{1F308}",
            Icon::MenuFilterRemoveBg => "\u{1FA84}",
            // Menu: Generate
            Icon::MenuGenerateGrid => "\u{1F4D0}",
            Icon::MenuGenerateShadow => "\u{1F4A4}",
            Icon::MenuGenerateOutline => "\u{1F58A}",
            Icon::MenuGenerateContours => "\u{1F5FA}",
            // Menu: View
            Icon::MenuViewZoomIn => "\u{1F50D}",
            Icon::MenuViewZoomOut => "\u{1F50D}",
            Icon::MenuViewFitWindow => "\u{229E}",
            Icon::MenuViewThemeLight => "\u{2600}",
            Icon::MenuViewThemeDark => "\u{1F319}",
            // UI: Context bar
            Icon::UiBrushDynamics => "\u{1F3B5}",
        }
    }
    
    /// Get the tooltip description for this icon (base text without keybind)
    pub fn tooltip(&self) -> &'static str {
        match self {
            // Tools — no hardcoded keybind letters
            Icon::Brush => "Brush Tool",
            Icon::Eraser => "Eraser Tool",
            Icon::Pencil => "Pencil Tool",
            Icon::Line => "Line Tool",
            Icon::Fill => "Fill Tool",
            Icon::RectSelect => "Rectangle Select",
            Icon::EllipseSelect => "Ellipse Select",
            Icon::Lasso => "Lasso Select",
            Icon::MagicWand => "Magic Wand",
            Icon::MovePixels => "Move Selected Pixels",
            Icon::MoveSelection => "Move Selection",
            Icon::PerspectiveCrop => "Perspective Crop",
            Icon::ColorPicker => "Color Picker",
            Icon::CloneStamp => "Clone Stamp",
            Icon::Zoom => "Zoom Tool",
            Icon::Pan => "Pan Tool",
            Icon::Gradient => "Gradient Tool",
            Icon::Text => "Text Tool",
            Icon::ContentAwareBrush => "Content-Aware Brush",
            Icon::Liquify => "Liquify Tool",
            Icon::MeshWarp => "Mesh Warp",
            Icon::ColorRemover => "Color Remover",
            Icon::Smudge => "Smudge Tool",
            Icon::Shapes => "Shapes Tool",
            // UI
            Icon::Undo => "Undo",
            Icon::Redo => "Redo",
            Icon::ZoomIn => "Zoom In",
            Icon::ZoomOut => "Zoom Out",
            Icon::ResetZoom => "Fit to Window",
            Icon::Grid => "Toggle Pixel Grid",
            Icon::GridOn => "Pixel Grid On",
            Icon::GridOff => "Pixel Grid Off",
            Icon::GuidesOn => "Guidelines On",
            Icon::GuidesOff => "Guidelines Off",
            Icon::MirrorOff => "Mirror: Off",
            Icon::MirrorH => "Mirror: Horizontal",
            Icon::MirrorV => "Mirror: Vertical",
            Icon::MirrorQ => "Mirror: Quarters",
            Icon::Settings => "Settings",
            Icon::Layers => "Layers",
            Icon::Visible => "Visible",
            Icon::Hidden => "Hidden",
            Icon::New => "New File",
            Icon::Open => "Open File",
            Icon::Save => "Save File",
            Icon::Close => "Close",
            Icon::NewLayer => "New Layer",
            Icon::Delete => "Delete",
            Icon::Duplicate => "Duplicate Layer",
            Icon::Flatten => "Flatten All Layers",
            Icon::MergeDown => "Merge Down",
            Icon::MergeDownAsMask => "Merge Down as Mask",
            Icon::Peek => "Peek Layer",
            Icon::SettingsGeneral => "General",
            Icon::SettingsInterface => "Interface",
            Icon::SettingsHardware => "Hardware",
            Icon::SettingsKeybinds => "Keybinds",
            Icon::SettingsAI => "AI",
            Icon::SwapColors => "Swap Colors",
            Icon::CopyHex => "Copy Hex",
            Icon::Commit => "Commit",
            Icon::ResetCancel => "Reset",
            Icon::Expand => "Expand",
            Icon::Collapse => "Collapse",
            Icon::Info => "Info",
            Icon::Search => "Search",
            Icon::ClearSearch => "Clear Search",
            Icon::MoveUp => "Move Up",
            Icon::MoveDown => "Move Down",
            Icon::MoveTop => "Move to Top",
            Icon::MoveBottom => "Move to Bottom",
            Icon::ImportLayer => "Import Layer from File",
            Icon::Rename => "Rename",
            Icon::LayerFlipH => "Flip Horizontal",
            Icon::LayerFlipV => "Flip Vertical",
            Icon::LayerRotate => "Rotate / Scale",
            Icon::LayerProperties => "Layer Properties",
            Icon::LayerAdd => "Add Layer",
            Icon::LayerDelete => "Delete Layer",
            Icon::LayerDuplicate => "Duplicate Layer",
            Icon::CurrentMarker => "Current",
            Icon::ApplyPrimary => "Apply Primary Color",
            Icon::SoloLayer => "Solo Layer",
            Icon::HideAll => "Hide All Layers",
            Icon::ShowAll => "Show All Layers",
            // UI: Context bar
            Icon::UiBrushDynamics => "Brush Dynamics (Flow, Scatter, Color Jitter)",
            // All menu items — tooltip matches the label
            _ => "",
        }
    }

    /// Map a tool icon to its BindableAction (for dynamic tooltip keybind lookup)
    pub fn bindable_action(&self) -> Option<BindableAction> {
        match self {
            Icon::Brush => Some(BindableAction::ToolBrush),
            Icon::Eraser => Some(BindableAction::ToolEraser),
            Icon::Pencil => Some(BindableAction::ToolPencil),
            Icon::Line => Some(BindableAction::ToolLine),
            Icon::Fill => Some(BindableAction::ToolFill),
            Icon::RectSelect => Some(BindableAction::ToolRectSelect),
            Icon::MagicWand => Some(BindableAction::ToolMagicWand),
            Icon::MovePixels => Some(BindableAction::ToolMovePixels),
            Icon::ColorPicker => Some(BindableAction::ToolColorPicker),
            Icon::CloneStamp => Some(BindableAction::ToolCloneStamp),
            Icon::Zoom => Some(BindableAction::ToolZoom),
            Icon::Pan => Some(BindableAction::ToolPan),
            Icon::Gradient => Some(BindableAction::ToolGradient),
            Icon::Text => Some(BindableAction::ToolText),
            Icon::Lasso => Some(BindableAction::ToolLasso),
            Icon::ColorRemover => Some(BindableAction::ToolColorRemover),
            Icon::MeshWarp => Some(BindableAction::ToolMeshWarp),
            Icon::Shapes => Some(BindableAction::ToolShapes),
            Icon::Undo => Some(BindableAction::Undo),
            Icon::Redo => Some(BindableAction::Redo),
            _ => None,
        }
    }

    /// Get tooltip with the current keybinding appended (e.g. "Brush Tool (B)")
    pub fn tooltip_with_keybind(&self, kb: &KeyBindings) -> String {
        let base = self.tooltip();
        if base.is_empty() {
            return String::new();
        }
        if let Some(action) = self.bindable_action() {
            if let Some(combo) = kb.get(action) {
                return format!("{} ({})", base, combo.display());
            }
        }
        base.to_string()
    }
}

/// Category grouping brush tips in the picker UI
pub struct BrushTipCategory {
    pub name: String,
    pub tips: Vec<String>,
}

/// Loaded brush tip data — normalized alpha mask + icon
#[allow(dead_code)]
pub struct BrushTipData {
    pub name: String,
    pub category: String,
    /// Alpha mask at canonical resolution, row-major, single channel (white=opaque)
    pub mask: Vec<u8>,
    pub mask_size: u32,
    /// Original source pixels for picker icon (RGBA)
    pub icon_pixels: Vec<u8>,
    pub icon_size: [usize; 2],
}

/// Asset manager for loading and caching textures
pub struct Assets {
    textures: HashMap<Icon, TextureHandle>,
    shape_textures: HashMap<ShapeKind, TextureHandle>,
    /// Original (light-mode) RGBA pixel data for each icon
    icon_pixels: HashMap<Icon, Vec<u8>>,
    /// Original (light-mode) RGBA pixel data for each shape icon
    shape_pixels: HashMap<ShapeKind, Vec<u8>>,
    /// Dimensions [width, height] for each icon
    icon_sizes: HashMap<Icon, [usize; 2]>,
    /// Dimensions [width, height] for each shape icon
    shape_sizes: HashMap<ShapeKind, [usize; 2]>,
    /// Whether icons are currently inverted (dark mode)
    icons_inverted: bool,
    icons_loaded: bool,
    /// Brush tip data indexed by name
    brush_tip_data: Vec<BrushTipData>,
    /// Brush tip categories (ordered)
    brush_tip_categories: Vec<BrushTipCategory>,
    /// Brush tip picker icon textures by name
    brush_tip_textures: HashMap<String, TextureHandle>,
    /// Original RGBA pixels for brush tip icons (for theme inversion)
    brush_tip_icon_pixels: HashMap<String, Vec<u8>>,
    /// Icon sizes for brush tip icons
    brush_tip_icon_sizes: HashMap<String, [usize; 2]>,
}

impl Default for Assets {
    fn default() -> Self {
        Self {
            textures: HashMap::new(),
            shape_textures: HashMap::new(),
            icon_pixels: HashMap::new(),
            shape_pixels: HashMap::new(),
            icon_sizes: HashMap::new(),
            shape_sizes: HashMap::new(),
            icons_inverted: false,
            icons_loaded: false,
            brush_tip_data: Vec::new(),
            brush_tip_categories: Vec::new(),
            brush_tip_textures: HashMap::new(),
            brush_tip_icon_pixels: HashMap::new(),
            brush_tip_icon_sizes: HashMap::new(),
        }
    }
}

impl Assets {
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Initialize assets - call once during app startup with the egui context
    pub fn init(&mut self, ctx: &egui::Context) {
        // === Tool icons ===
        self.load_icon(ctx, Icon::Brush, include_bytes!("../assets/icons/tools/brush.png"));
        self.load_icon(ctx, Icon::Eraser, include_bytes!("../assets/icons/tools/eraser.png"));
        self.load_icon(ctx, Icon::Pencil, include_bytes!("../assets/icons/tools/pencil.png"));
        self.load_icon(ctx, Icon::Line, include_bytes!("../assets/icons/tools/line.png"));
        self.load_icon(ctx, Icon::Fill, include_bytes!("../assets/icons/tools/fill.png"));
        self.load_icon(ctx, Icon::RectSelect, include_bytes!("../assets/icons/tools/rect_select.png"));
        self.load_icon(ctx, Icon::EllipseSelect, include_bytes!("../assets/icons/tools/ellipse_select.png"));
        self.load_icon(ctx, Icon::Lasso, include_bytes!("../assets/icons/tools/lasso.png"));
        self.load_icon(ctx, Icon::MagicWand, include_bytes!("../assets/icons/tools/magic_wand.png"));
        self.load_icon(ctx, Icon::MovePixels, include_bytes!("../assets/icons/tools/move_pixels.png"));
        self.load_icon(ctx, Icon::MoveSelection, include_bytes!("../assets/icons/tools/move_selection.png"));
        self.load_icon(ctx, Icon::PerspectiveCrop, include_bytes!("../assets/icons/tools/perspective_crop.png"));
        self.load_icon(ctx, Icon::ColorPicker, include_bytes!("../assets/icons/tools/color_picker.png"));
        self.load_icon(ctx, Icon::CloneStamp, include_bytes!("../assets/icons/tools/clone_stamp.png"));
        self.load_icon(ctx, Icon::Zoom, include_bytes!("../assets/icons/tools/zoom.png"));
        self.load_icon(ctx, Icon::Pan, include_bytes!("../assets/icons/tools/pan.png"));
        self.load_icon(ctx, Icon::Gradient, include_bytes!("../assets/icons/tools/gradient.png"));
        self.load_icon(ctx, Icon::Text, include_bytes!("../assets/icons/tools/text.png"));
        self.load_icon(ctx, Icon::ContentAwareBrush, include_bytes!("../assets/icons/tools/content_aware_brush.png"));
        self.load_icon(ctx, Icon::Liquify, include_bytes!("../assets/icons/tools/liquify.png"));
        self.load_icon(ctx, Icon::MeshWarp, include_bytes!("../assets/icons/tools/mesh_warp.png"));
        self.load_icon(ctx, Icon::ColorRemover, include_bytes!("../assets/icons/tools/color_remover.png"));
        self.load_icon(ctx, Icon::Smudge, include_bytes!("../assets/icons/tools/smudge.png"));
        self.load_icon(ctx, Icon::Shapes, include_bytes!("../assets/icons/tools/shapes.png"));

        // === UI icons ===
        self.load_icon(ctx, Icon::Undo, include_bytes!("../assets/icons/ui/undo.png"));
        self.load_icon(ctx, Icon::Redo, include_bytes!("../assets/icons/ui/redo.png"));
        self.load_icon(ctx, Icon::ZoomIn, include_bytes!("../assets/icons/ui/zoom_in.png"));
        self.load_icon(ctx, Icon::ZoomOut, include_bytes!("../assets/icons/ui/zoom_out.png"));
        self.load_icon(ctx, Icon::ResetZoom, include_bytes!("../assets/icons/ui/reset_zoom.png"));
        self.load_icon(ctx, Icon::Grid, include_bytes!("../assets/icons/ui/grid.png"));
        self.load_icon(ctx, Icon::GridOn, include_bytes!("../assets/icons/ui/grid_on.png"));
        self.load_icon(ctx, Icon::GridOff, include_bytes!("../assets/icons/ui/grid_off.png"));
        self.load_icon(ctx, Icon::GuidesOn, include_bytes!("../assets/icons/ui/guides_on.png"));
        self.load_icon(ctx, Icon::GuidesOff, include_bytes!("../assets/icons/ui/guides_off.png"));
        self.load_icon(ctx, Icon::MirrorOff, include_bytes!("../assets/icons/ui/mirror_off.png"));
        self.load_icon(ctx, Icon::MirrorH, include_bytes!("../assets/icons/ui/mirror_h.png"));
        self.load_icon(ctx, Icon::MirrorV, include_bytes!("../assets/icons/ui/mirror_v.png"));
        self.load_icon(ctx, Icon::MirrorQ, include_bytes!("../assets/icons/ui/mirror_q.png"));
        self.load_icon(ctx, Icon::Settings, include_bytes!("../assets/icons/ui/settings.png"));
        self.load_icon(ctx, Icon::Layers, include_bytes!("../assets/icons/ui/layers.png"));
        self.load_icon(ctx, Icon::Visible, include_bytes!("../assets/icons/ui/visible.png"));
        self.load_icon(ctx, Icon::Hidden, include_bytes!("../assets/icons/ui/hidden.png"));
        self.load_icon(ctx, Icon::New, include_bytes!("../assets/icons/ui/new.png"));
        self.load_icon(ctx, Icon::Open, include_bytes!("../assets/icons/ui/open.png"));
        self.load_icon(ctx, Icon::Save, include_bytes!("../assets/icons/ui/save.png"));
        self.load_icon(ctx, Icon::Close, include_bytes!("../assets/icons/ui/close.png"));
        self.load_icon(ctx, Icon::NewLayer, include_bytes!("../assets/icons/ui/new_layer.png"));
        self.load_icon(ctx, Icon::Delete, include_bytes!("../assets/icons/ui/delete.png"));
        self.load_icon(ctx, Icon::Duplicate, include_bytes!("../assets/icons/ui/duplicate.png"));
        self.load_icon(ctx, Icon::Flatten, include_bytes!("../assets/icons/ui/flatten.png"));
        self.load_icon(ctx, Icon::MergeDown, include_bytes!("../assets/icons/ui/merge_down.png"));
        self.load_icon(ctx, Icon::MergeDownAsMask, include_bytes!("../assets/icons/ui/merge_down_as_mask.png"));
        self.load_icon(ctx, Icon::Peek, include_bytes!("../assets/icons/ui/peek.png"));
        self.load_icon(ctx, Icon::SwapColors, include_bytes!("../assets/icons/ui/swap_colors.png"));
        self.load_icon(ctx, Icon::CopyHex, include_bytes!("../assets/icons/ui/copy_hex.png"));
        self.load_icon(ctx, Icon::Commit, include_bytes!("../assets/icons/ui/commit.png"));
        self.load_icon(ctx, Icon::ResetCancel, include_bytes!("../assets/icons/ui/reset_cancel.png"));
        self.load_icon(ctx, Icon::Expand, include_bytes!("../assets/icons/ui/expand.png"));
        self.load_icon(ctx, Icon::Collapse, include_bytes!("../assets/icons/ui/collapse.png"));
        self.load_icon(ctx, Icon::Info, include_bytes!("../assets/icons/ui/info.png"));
        self.load_icon(ctx, Icon::Search, include_bytes!("../assets/icons/ui/search.png"));
        self.load_icon(ctx, Icon::ClearSearch, include_bytes!("../assets/icons/ui/clear_search.png"));
        self.load_icon(ctx, Icon::MoveUp, include_bytes!("../assets/icons/ui/move_up.png"));
        self.load_icon(ctx, Icon::MoveDown, include_bytes!("../assets/icons/ui/move_down.png"));
        self.load_icon(ctx, Icon::MoveTop, include_bytes!("../assets/icons/ui/move_top.png"));
        self.load_icon(ctx, Icon::MoveBottom, include_bytes!("../assets/icons/ui/move_bottom.png"));
        self.load_icon(ctx, Icon::ImportLayer, include_bytes!("../assets/icons/ui/import_layer.png"));
        self.load_icon(ctx, Icon::Rename, include_bytes!("../assets/icons/ui/rename.png"));
        self.load_icon(ctx, Icon::LayerFlipH, include_bytes!("../assets/icons/ui/layer_flip_h.png"));
        self.load_icon(ctx, Icon::LayerFlipV, include_bytes!("../assets/icons/ui/layer_flip_v.png"));
        self.load_icon(ctx, Icon::LayerRotate, include_bytes!("../assets/icons/ui/layer_rotate.png"));
        self.load_icon(ctx, Icon::LayerProperties, include_bytes!("../assets/icons/ui/layer_properties.png"));
        self.load_icon(ctx, Icon::LayerAdd, include_bytes!("../assets/icons/ui/layer_add.png"));
        self.load_icon(ctx, Icon::LayerDelete, include_bytes!("../assets/icons/ui/layer_delete.png"));
        self.load_icon(ctx, Icon::LayerDuplicate, include_bytes!("../assets/icons/ui/layer_duplicate.png"));
        self.load_icon(ctx, Icon::CurrentMarker, include_bytes!("../assets/icons/ui/current_marker.png"));
        self.load_icon(ctx, Icon::ApplyPrimary, include_bytes!("../assets/icons/ui/apply_primary.png"));
        self.load_icon(ctx, Icon::SoloLayer, include_bytes!("../assets/icons/ui/solo_layer.png"));
        self.load_icon(ctx, Icon::HideAll, include_bytes!("../assets/icons/ui/hide_all.png"));
        self.load_icon(ctx, Icon::ShowAll, include_bytes!("../assets/icons/ui/show_all.png"));

        // === Settings tab icons ===
        self.load_icon(ctx, Icon::SettingsGeneral, include_bytes!("../assets/icons/ui/settings_general.png"));
        self.load_icon(ctx, Icon::SettingsInterface, include_bytes!("../assets/icons/ui/settings_interface.png"));
        self.load_icon(ctx, Icon::SettingsHardware, include_bytes!("../assets/icons/ui/settings_hardware.png"));
        self.load_icon(ctx, Icon::SettingsKeybinds, include_bytes!("../assets/icons/ui/settings_keybinds.png"));
        self.load_icon(ctx, Icon::SettingsAI, include_bytes!("../assets/icons/ui/settings_ai.png"));

        // === Menu: File ===
        self.load_icon(ctx, Icon::MenuFileNew, include_bytes!("../assets/icons/menu/file_new.png"));
        self.load_icon(ctx, Icon::MenuFileOpen, include_bytes!("../assets/icons/menu/file_open.png"));
        self.load_icon(ctx, Icon::MenuFileSave, include_bytes!("../assets/icons/menu/file_save.png"));
        self.load_icon(ctx, Icon::MenuFileSaveAll, include_bytes!("../assets/icons/menu/file_save.png"));
        self.load_icon(ctx, Icon::MenuFileSaveAs, include_bytes!("../assets/icons/menu/file_save_as.png"));
        self.load_icon(ctx, Icon::MenuFilePrint, include_bytes!("../assets/icons/menu/file_print.png"));
        self.load_icon(ctx, Icon::MenuFileQuit, include_bytes!("../assets/icons/menu/file_quit.png"));

        // === Menu: Edit ===
        self.load_icon(ctx, Icon::MenuEditUndo, include_bytes!("../assets/icons/menu/edit_undo.png"));
        self.load_icon(ctx, Icon::MenuEditRedo, include_bytes!("../assets/icons/menu/edit_redo.png"));
        self.load_icon(ctx, Icon::MenuEditCut, include_bytes!("../assets/icons/menu/edit_cut.png"));
        self.load_icon(ctx, Icon::MenuEditCopy, include_bytes!("../assets/icons/menu/edit_copy.png"));
        self.load_icon(ctx, Icon::MenuEditPaste, include_bytes!("../assets/icons/menu/edit_paste.png"));
        self.load_icon(ctx, Icon::MenuEditPasteLayer, include_bytes!("../assets/icons/menu/edit_paste_layer.png"));
        self.load_icon(ctx, Icon::MenuEditSelectAll, include_bytes!("../assets/icons/menu/edit_select_all.png"));
        self.load_icon(ctx, Icon::MenuEditDeselect, include_bytes!("../assets/icons/menu/edit_deselect.png"));
        self.load_icon(ctx, Icon::MenuEditInvertSel, include_bytes!("../assets/icons/menu/edit_invert_sel.png"));
        self.load_icon(ctx, Icon::MenuEditColorRange, include_bytes!("../assets/icons/menu/edit_color_range.png"));
        self.load_icon(ctx, Icon::MenuEditPreferences, include_bytes!("../assets/icons/menu/edit_preferences.png"));

        // === Menu: Canvas ===
        self.load_icon(ctx, Icon::MenuCanvasResize, include_bytes!("../assets/icons/menu/canvas_resize.png"));
        self.load_icon(ctx, Icon::MenuCanvasCrop, include_bytes!("../assets/icons/menu/canvas_crop.png"));
        self.load_icon(ctx, Icon::MenuCanvasFlipH, include_bytes!("../assets/icons/menu/canvas_flip_h.png"));
        self.load_icon(ctx, Icon::MenuCanvasFlipV, include_bytes!("../assets/icons/menu/canvas_flip_v.png"));
        self.load_icon(ctx, Icon::MenuCanvasRotateCw, include_bytes!("../assets/icons/menu/canvas_rotate_cw.png"));
        self.load_icon(ctx, Icon::MenuCanvasRotateCcw, include_bytes!("../assets/icons/menu/canvas_rotate_ccw.png"));
        self.load_icon(ctx, Icon::MenuCanvasRotate180, include_bytes!("../assets/icons/menu/canvas_rotate_180.png"));
        self.load_icon(ctx, Icon::MenuCanvasFlatten, include_bytes!("../assets/icons/menu/canvas_flatten.png"));

        // === Menu: Color ===
        self.load_icon(ctx, Icon::MenuColorAutoLevels, include_bytes!("../assets/icons/menu/color_auto_levels.png"));
        self.load_icon(ctx, Icon::MenuColorDesaturate, include_bytes!("../assets/icons/menu/color_desaturate.png"));
        self.load_icon(ctx, Icon::MenuColorInvert, include_bytes!("../assets/icons/menu/color_invert.png"));
        self.load_icon(ctx, Icon::MenuColorInvertAlpha, include_bytes!("../assets/icons/menu/color_invert_alpha.png"));
        self.load_icon(ctx, Icon::MenuColorSepia, include_bytes!("../assets/icons/menu/color_sepia.png"));
        self.load_icon(ctx, Icon::MenuColorBrightness, include_bytes!("../assets/icons/menu/color_brightness.png"));
        self.load_icon(ctx, Icon::MenuColorCurves, include_bytes!("../assets/icons/menu/color_curves.png"));
        self.load_icon(ctx, Icon::MenuColorExposure, include_bytes!("../assets/icons/menu/color_exposure.png"));
        self.load_icon(ctx, Icon::MenuColorHighlights, include_bytes!("../assets/icons/menu/color_highlights.png"));
        self.load_icon(ctx, Icon::MenuColorHsl, include_bytes!("../assets/icons/menu/color_hsl.png"));
        self.load_icon(ctx, Icon::MenuColorLevels, include_bytes!("../assets/icons/menu/color_levels.png"));
        self.load_icon(ctx, Icon::MenuColorTemperature, include_bytes!("../assets/icons/menu/color_temperature.png"));

        // === Menu: Filter ===
        self.load_icon(ctx, Icon::MenuFilterBlur, include_bytes!("../assets/icons/menu/filter_blur.png"));
        self.load_icon(ctx, Icon::MenuFilterGaussian, include_bytes!("../assets/icons/menu/filter_gaussian.png"));
        self.load_icon(ctx, Icon::MenuFilterBokeh, include_bytes!("../assets/icons/menu/filter_bokeh.png"));
        self.load_icon(ctx, Icon::MenuFilterMotionBlur, include_bytes!("../assets/icons/menu/filter_motion_blur.png"));
        self.load_icon(ctx, Icon::MenuFilterBoxBlur, include_bytes!("../assets/icons/menu/filter_box_blur.png"));
        self.load_icon(ctx, Icon::MenuFilterZoomBlur, include_bytes!("../assets/icons/menu/filter_zoom_blur.png"));
        self.load_icon(ctx, Icon::MenuFilterSharpen, include_bytes!("../assets/icons/menu/filter_sharpen.png"));
        self.load_icon(ctx, Icon::MenuFilterSharpenItem, include_bytes!("../assets/icons/menu/filter_sharpen_item.png"));
        self.load_icon(ctx, Icon::MenuFilterReduceNoise, include_bytes!("../assets/icons/menu/filter_reduce_noise.png"));
        self.load_icon(ctx, Icon::MenuFilterDistort, include_bytes!("../assets/icons/menu/filter_distort.png"));
        self.load_icon(ctx, Icon::MenuFilterCrystallize, include_bytes!("../assets/icons/menu/filter_crystallize.png"));
        self.load_icon(ctx, Icon::MenuFilterDents, include_bytes!("../assets/icons/menu/filter_dents.png"));
        self.load_icon(ctx, Icon::MenuFilterPixelate, include_bytes!("../assets/icons/menu/filter_pixelate.png"));
        self.load_icon(ctx, Icon::MenuFilterBulge, include_bytes!("../assets/icons/menu/filter_bulge.png"));
        self.load_icon(ctx, Icon::MenuFilterTwist, include_bytes!("../assets/icons/menu/filter_twist.png"));
        self.load_icon(ctx, Icon::MenuFilterNoise, include_bytes!("../assets/icons/menu/filter_noise.png"));
        self.load_icon(ctx, Icon::MenuFilterAddNoise, include_bytes!("../assets/icons/menu/filter_add_noise.png"));
        self.load_icon(ctx, Icon::MenuFilterMedian, include_bytes!("../assets/icons/menu/filter_median.png"));
        self.load_icon(ctx, Icon::MenuFilterStylize, include_bytes!("../assets/icons/menu/filter_stylize.png"));
        self.load_icon(ctx, Icon::MenuFilterGlow, include_bytes!("../assets/icons/menu/filter_glow.png"));
        self.load_icon(ctx, Icon::MenuFilterVignette, include_bytes!("../assets/icons/menu/filter_vignette.png"));
        self.load_icon(ctx, Icon::MenuFilterHalftone, include_bytes!("../assets/icons/menu/filter_halftone.png"));
        self.load_icon(ctx, Icon::MenuFilterInk, include_bytes!("../assets/icons/menu/filter_ink.png"));
        self.load_icon(ctx, Icon::MenuFilterOilPainting, include_bytes!("../assets/icons/menu/filter_oil_painting.png"));
        self.load_icon(ctx, Icon::MenuFilterColorFilter, include_bytes!("../assets/icons/menu/filter_color_filter.png"));
        self.load_icon(ctx, Icon::MenuFilterGlitch, include_bytes!("../assets/icons/menu/filter_glitch.png"));
        self.load_icon(ctx, Icon::MenuFilterPixelDrag, include_bytes!("../assets/icons/menu/filter_pixel_drag.png"));
        self.load_icon(ctx, Icon::MenuFilterRgbDisplace, include_bytes!("../assets/icons/menu/filter_rgb_displace.png"));
        self.load_icon(ctx, Icon::MenuFilterRemoveBg, include_bytes!("../assets/icons/menu/filter_remove_bg.png"));

        // === Menu: Generate ===
        self.load_icon(ctx, Icon::MenuGenerateGrid, include_bytes!("../assets/icons/menu/generate_grid.png"));
        self.load_icon(ctx, Icon::MenuGenerateShadow, include_bytes!("../assets/icons/menu/generate_shadow.png"));
        self.load_icon(ctx, Icon::MenuGenerateOutline, include_bytes!("../assets/icons/menu/generate_outline.png"));
        self.load_icon(ctx, Icon::MenuGenerateContours, include_bytes!("../assets/icons/menu/generate_contours.png"));

        // === Menu: View ===
        self.load_icon(ctx, Icon::MenuViewZoomIn, include_bytes!("../assets/icons/menu/view_zoom_in.png"));
        self.load_icon(ctx, Icon::MenuViewZoomOut, include_bytes!("../assets/icons/menu/view_zoom_out.png"));
        self.load_icon(ctx, Icon::MenuViewFitWindow, include_bytes!("../assets/icons/menu/view_fit_window.png"));
        self.load_icon(ctx, Icon::MenuViewThemeLight, include_bytes!("../assets/icons/menu/view_theme_light.png"));
        self.load_icon(ctx, Icon::MenuViewThemeDark, include_bytes!("../assets/icons/menu/view_theme_dark.png"));
        // UI: Context bar
        self.load_icon(ctx, Icon::UiBrushDynamics, include_bytes!("../assets/icons/ui/dynamics.png"));
        
        self.icons_loaded = true;
        
        // Load shape picker icons (from new location)
        self.load_shape_icon(ctx, ShapeKind::Ellipse, include_bytes!("../assets/icons/shapes/ellipse.png"));
        self.load_shape_icon(ctx, ShapeKind::Rectangle, include_bytes!("../assets/icons/shapes/rectangle.png"));
        self.load_shape_icon(ctx, ShapeKind::RoundedRect, include_bytes!("../assets/icons/shapes/rounded_rect.png"));
        self.load_shape_icon(ctx, ShapeKind::Trapezoid, include_bytes!("../assets/icons/shapes/trapezoid.png"));
        self.load_shape_icon(ctx, ShapeKind::Parallelogram, include_bytes!("../assets/icons/shapes/parallelogram.png"));
        self.load_shape_icon(ctx, ShapeKind::Triangle, include_bytes!("../assets/icons/shapes/triangle.png"));
        self.load_shape_icon(ctx, ShapeKind::RightTriangle, include_bytes!("../assets/icons/shapes/right_triangle.png"));
        self.load_shape_icon(ctx, ShapeKind::Pentagon, include_bytes!("../assets/icons/shapes/pentagon.png"));
        self.load_shape_icon(ctx, ShapeKind::Hexagon, include_bytes!("../assets/icons/shapes/hexagon.png"));
        self.load_shape_icon(ctx, ShapeKind::Octagon, include_bytes!("../assets/icons/shapes/octagon.png"));
        self.load_shape_icon(ctx, ShapeKind::Cross, include_bytes!("../assets/icons/shapes/cross.png"));
        self.load_shape_icon(ctx, ShapeKind::Check, include_bytes!("../assets/icons/shapes/check.png"));
        self.load_shape_icon(ctx, ShapeKind::Heart, include_bytes!("../assets/icons/shapes/heart.png"));

        // === Brush tip icons (embedded at compile time via build.rs) ===
        self.load_embedded_brush_tips(ctx);
    }

    /// Load brush tips embedded at compile time by build.rs.
    /// The generated file contains `EMBEDDED_BRUSH_TIPS: &[(&str, &str, &[u8])]`
    /// where each entry is (display_name, category_name, png_bytes).
    fn load_embedded_brush_tips(&mut self, ctx: &egui::Context) {
        let tips: &[(&str, &str, &[u8])] =
            include!(concat!(env!("OUT_DIR"), "/brush_tips_embedded.rs"));

        for &(name, category, png_data) in tips {
            self.load_brush_tip(ctx, name, category, png_data);
        }
    }
    
    /// Load a single icon from PNG bytes (caches original pixel data for theme inversion)
    fn load_icon(&mut self, ctx: &egui::Context, icon: Icon, png_data: &[u8]) {
        match image::load_from_memory(png_data) {
            Ok(img) => {
                let rgba = img.to_rgba8();
                let size = [rgba.width() as usize, rgba.height() as usize];
                let pixels = rgba.into_raw();
                // Cache original pixels and size for theme-based inversion
                self.icon_pixels.insert(icon, pixels.clone());
                self.icon_sizes.insert(icon, size);
                let display_pixels = if self.icons_inverted {
                    Self::invert_rgb(&pixels)
                } else {
                    pixels
                };
                let color_image = ColorImage::from_rgba_unmultiplied(size, &display_pixels);
                let texture = ctx.load_texture(
                    format!("icon_{:?}", icon),
                    color_image,
                    TextureOptions::LINEAR
                );
                self.textures.insert(icon, texture);
            }
            Err(e) => {
                eprintln!("Failed to load icon {:?}: {}", icon, e);
            }
        }
    }
    
    /// Load a shape picker icon from PNG bytes (caches original pixel data for theme inversion)
    fn load_shape_icon(&mut self, ctx: &egui::Context, kind: ShapeKind, png_data: &[u8]) {
        match image::load_from_memory(png_data) {
            Ok(img) => {
                let rgba = img.to_rgba8();
                let size = [rgba.width() as usize, rgba.height() as usize];
                let pixels = rgba.into_raw();
                // Cache original pixels and size for theme-based inversion
                self.shape_pixels.insert(kind, pixels.clone());
                self.shape_sizes.insert(kind, size);
                let display_pixels = if self.icons_inverted {
                    Self::invert_rgb(&pixels)
                } else {
                    pixels
                };
                let color_image = ColorImage::from_rgba_unmultiplied(size, &display_pixels);
                let texture = ctx.load_texture(
                    format!("shape_{:?}", kind),
                    color_image,
                    TextureOptions::LINEAR
                );
                self.shape_textures.insert(kind, texture);
            }
            Err(e) => {
                eprintln!("Failed to load shape icon {:?}: {}", kind, e);
            }
        }
    }

    /// Invert RGB channels for dark-mode display.
    ///
    /// Simple `255 - x` inversion: black → white, white → black.
    /// Fully-transparent pixels are skipped so their RGB noise
    /// doesn't bleed through when composited.
    fn invert_rgb(pixels: &[u8]) -> Vec<u8> {
        let mut out = pixels.to_vec();
        for chunk in out.chunks_exact_mut(4) {
            let a = chunk[3];
            if a == 0 {
                // Fully transparent — leave RGB as-is (invisible anyway).
                continue;
            }
            chunk[0] = 255 - chunk[0];
            chunk[1] = 255 - chunk[1];
            chunk[2] = 255 - chunk[2];
            // alpha unchanged
        }
        out
    }

    /// Update all icon textures for the current theme.
    /// Call this when the theme changes between light and dark mode.
    pub fn update_theme(&mut self, ctx: &egui::Context, dark: bool) {
        if dark == self.icons_inverted {
            return; // already in the correct state
        }
        self.icons_inverted = dark;

        // Re-upload all icon textures
        for (icon, original_pixels) in &self.icon_pixels {
            if let Some(size) = self.icon_sizes.get(icon) {
                let display_pixels = if dark {
                    Self::invert_rgb(original_pixels)
                } else {
                    original_pixels.clone()
                };
                let color_image = ColorImage::from_rgba_unmultiplied(*size, &display_pixels);
                let texture = ctx.load_texture(
                    format!("icon_{:?}", icon),
                    color_image,
                    TextureOptions::LINEAR,
                );
                self.textures.insert(*icon, texture);
            }
        }

        // Re-upload all shape textures
        for (kind, original_pixels) in &self.shape_pixels {
            if let Some(size) = self.shape_sizes.get(kind) {
                let display_pixels = if dark {
                    Self::invert_rgb(original_pixels)
                } else {
                    original_pixels.clone()
                };
                let color_image = ColorImage::from_rgba_unmultiplied(*size, &display_pixels);
                let texture = ctx.load_texture(
                    format!("shape_{:?}", kind),
                    color_image,
                    TextureOptions::LINEAR,
                );
                self.shape_textures.insert(*kind, texture);
            }
        }

        // Re-upload all brush tip icon textures
        for (name, original_pixels) in &self.brush_tip_icon_pixels {
            if let Some(size) = self.brush_tip_icon_sizes.get(name) {
                let display_pixels = if dark {
                    Self::invert_rgb(original_pixels)
                } else {
                    original_pixels.clone()
                };
                let color_image = ColorImage::from_rgba_unmultiplied(*size, &display_pixels);
                let texture = ctx.load_texture(
                    format!("brush_tip_{}", name),
                    color_image,
                    TextureOptions::LINEAR,
                );
                self.brush_tip_textures.insert(name.clone(), texture);
            }
        }
    }
    
    pub fn get_shape_texture(&self, kind: ShapeKind) -> Option<&TextureHandle> {
        self.shape_textures.get(&kind)
    }

    /// Load a brush tip from grayscale PNG bytes.
    /// Extracts the luminance/alpha as a single-channel mask, creates an icon texture.
    fn load_brush_tip(&mut self, ctx: &egui::Context, name: &str, category: &str, png_data: &[u8]) {
        match image::load_from_memory(png_data) {
            Ok(img) => {
                let gray = img.to_luma8();
                let gw = gray.width();
                let gh = gray.height();
                let mask: Vec<u8> = gray.into_raw();

                // Upscale icon to 2× for better visibility in the picker
                let iw = (gw * 2) as usize;
                let ih = (gh * 2) as usize;
                let icon_size = [iw, ih];

                // Create RGBA icon at 2× size: black shape on transparent bg
                let mut icon_pixels = vec![0u8; iw * ih * 4];
                for y in 0..ih {
                    for x in 0..iw {
                        let sx = x / 2;
                        let sy = y / 2;
                        let v = mask[sy * gw as usize + sx];
                        let dst = (y * iw + x) * 4;
                        icon_pixels[dst]     = 0;
                        icon_pixels[dst + 1] = 0;
                        icon_pixels[dst + 2] = 0;
                        icon_pixels[dst + 3] = v;
                    }
                }

                // Cache icon pixels for theme inversion
                self.brush_tip_icon_pixels.insert(name.to_string(), icon_pixels.clone());
                self.brush_tip_icon_sizes.insert(name.to_string(), icon_size);

                let display_pixels = if self.icons_inverted {
                    Self::invert_rgb(&icon_pixels)
                } else {
                    icon_pixels
                };

                let color_image = ColorImage::from_rgba_unmultiplied(icon_size, &display_pixels);
                let texture = ctx.load_texture(
                    format!("brush_tip_{}", name),
                    color_image,
                    TextureOptions::LINEAR,
                );
                self.brush_tip_textures.insert(name.to_string(), texture);

                // Normalize mask to canonical size (square)
                let canonical = gw.max(gh) as usize;
                let mask_size = canonical;
                // Pad to square if needed
                let mask = if gw as usize == canonical && gh as usize == canonical {
                    mask
                } else {
                    let mut padded = vec![0u8; canonical * canonical];
                    for y in 0..gh as usize {
                        for x in 0..gw as usize {
                            padded[y * canonical + x] = mask[y * gw as usize + x];
                        }
                    }
                    padded
                };

                // Add to category
                let cat_idx = self.brush_tip_categories.iter().position(|c| c.name == category);
                if let Some(idx) = cat_idx {
                    self.brush_tip_categories[idx].tips.push(name.to_string());
                } else {
                    self.brush_tip_categories.push(BrushTipCategory {
                        name: category.to_string(),
                        tips: vec![name.to_string()],
                    });
                }

                self.brush_tip_data.push(BrushTipData {
                    name: name.to_string(),
                    category: category.to_string(),
                    mask,
                    mask_size: mask_size as u32,
                    icon_pixels: self.brush_tip_icon_pixels.get(name).unwrap().clone(),
                    icon_size,
                });
            }
            Err(e) => {
                eprintln!("Failed to load brush tip '{}': {}", name, e);
            }
        }
    }

    /// Get the picker icon texture for a brush tip by name
    pub fn get_brush_tip_texture(&self, name: &str) -> Option<&TextureHandle> {
        self.brush_tip_textures.get(name)
    }

    /// Get brush tip mask data by name
    pub fn get_brush_tip_data(&self, name: &str) -> Option<&BrushTipData> {
        self.brush_tip_data.iter().find(|d| d.name == name)
    }

    pub fn brush_tip_categories(&self) -> &[BrushTipCategory] {
        &self.brush_tip_categories
    }
    
    /// Check if a texture is available for the given icon
    pub fn has_texture(&self, icon: Icon) -> bool {
        self.textures.contains_key(&icon)
    }
    
    pub fn get_texture(&self, icon: Icon) -> Option<&TextureHandle> {
        self.textures.get(&icon)
    }
    
    /// Create an icon button that uses texture if available, emoji fallback otherwise
    pub fn icon_button(&self, ui: &mut egui::Ui, icon: Icon, size: Vec2) -> egui::Response {
        let response = if let Some(texture) = self.textures.get(&icon) {
            // Use texture-based button
            let sized_texture = egui::load::SizedTexture::from_handle(texture);
            let img = egui::Image::from_texture(sized_texture).fit_to_exact_size(size);
            ui.add_sized(size, egui::Button::image(img))
        } else {
            // Use text fallback
            ui.add_sized(size, egui::Button::new(icon.emoji()))
        };
        
        response.on_hover_text(icon.tooltip())
    }
    
    /// Create a selectable icon (for tool selection) with custom size
    pub fn icon_selectable_sized(&self, ui: &mut egui::Ui, icon: Icon, selected: bool, size: Vec2) -> bool {
        let response = if let Some(texture) = self.textures.get(&icon) {
            // Use texture-based selectable
            let sized_texture = egui::load::SizedTexture::from_handle(texture);
            let img = egui::Image::from_texture(sized_texture).fit_to_exact_size(size * 0.75);
            let mut button = egui::Button::image(img);
            if selected {
                button = button.fill(ui.visuals().selection.bg_fill);
            }
            ui.add_sized(size, button)
        } else {
            // Use text fallback
            let text = egui::RichText::new(icon.emoji()).size(size.y * 0.5);
            ui.add_sized(size, egui::SelectableLabel::new(selected, text))
        };
        response.on_hover_text(icon.tooltip()).clicked()
    }
    
    /// Create a selectable icon (for tool selection) with default size
    pub fn icon_selectable(&self, ui: &mut egui::Ui, icon: Icon, selected: bool) -> bool {
        self.icon_selectable_sized(ui, icon, selected, Vec2::new(32.0, 32.0))
    }
    
    /// Create a small icon button (for toolbar)
    pub fn small_icon_button(&self, ui: &mut egui::Ui, icon: Icon) -> egui::Response {
        let response = if let Some(texture) = self.textures.get(&icon) {
            let sized_texture = egui::load::SizedTexture::from_handle(texture);
            let img = egui::Image::from_texture(sized_texture).fit_to_exact_size(Vec2::splat(24.0));
            let mut btn = egui::Button::image(img);
            if ui.visuals().dark_mode {
                btn = btn.fill(egui::Color32::from_gray(18));
            }
            ui.add(btn)
        } else {
            ui.button(icon.emoji())
        };
        response.on_hover_text(icon.tooltip())
    }

    /// Create a small icon button with no background frame (transparent).
    pub fn small_icon_button_frameless(&self, ui: &mut egui::Ui, icon: Icon) -> egui::Response {
        let response = if let Some(texture) = self.textures.get(&icon) {
            let sized_texture = egui::load::SizedTexture::from_handle(texture);
            let img = egui::Image::from_texture(sized_texture).fit_to_exact_size(Vec2::splat(24.0));
            ui.add(egui::Button::image(img).frame(false))
        } else {
            ui.add(egui::Button::new(icon.emoji()).frame(false))
        };
        response.on_hover_text(icon.tooltip())
    }
    
    /// Create an enabled/disabled icon button
    pub fn icon_button_enabled(&self, ui: &mut egui::Ui, icon: Icon, enabled: bool) -> egui::Response {
        let response = if let Some(texture) = self.textures.get(&icon) {
            let sized_texture = egui::load::SizedTexture::from_handle(texture);
            let img = egui::Image::from_texture(sized_texture).fit_to_exact_size(Vec2::splat(24.0));
            let mut btn = egui::Button::image(img);
            if ui.visuals().dark_mode {
                btn = btn.fill(egui::Color32::from_gray(18));
            }
            ui.add_enabled(enabled, btn)
        } else {
            ui.add_enabled(enabled, egui::Button::new(icon.emoji()))
        };
        response.on_hover_text(icon.tooltip())
    }

    /// Paint an icon image into a specific rect, returning a click-sense response.
    /// Used for inline icons in layer rows (eye, peek).
    pub fn icon_in_rect(&self, ui: &mut egui::Ui, icon: Icon, rect: egui::Rect, tint: Color32) -> egui::Response {
        if let Some(texture) = self.textures.get(&icon) {
            let sized_texture = egui::load::SizedTexture::from_handle(texture);
            let img = egui::Image::from_texture(sized_texture)
                .fit_to_exact_size(rect.size())
                .tint(tint);
            ui.put(rect, img.sense(Sense::click()))
        } else {
            ui.put(rect, egui::Label::new(
                egui::RichText::new(icon.emoji()).size(rect.height() * 0.7).color(tint)
            ).sense(Sense::click()))
        }
    }

    /// Create a selectable label with an icon image + text.
    /// Used for settings sidebar tabs.
    pub fn icon_selectable_label(&self, ui: &mut egui::Ui, icon: Icon, text: &str, selected: bool) -> bool {
        let icon_size = Vec2::splat(16.0);
        if let Some(texture) = self.textures.get(&icon) {
            let sized_texture = egui::load::SizedTexture::from_handle(texture);
            let _img = egui::Image::from_texture(sized_texture).fit_to_exact_size(icon_size);
            let layout = egui::Layout::left_to_right(egui::Align::Center);
            let response = ui.with_layout(layout, |ui| {
                let r = ui.add(egui::SelectableLabel::new(selected, ""));
                // Paint icon on top of the response rect, left-aligned
                let icon_rect = egui::Rect::from_min_size(
                    r.rect.left_center() - Vec2::new(-4.0, icon_size.y / 2.0),
                    icon_size,
                );
                let tint = if selected {
                    ui.visuals().strong_text_color()
                } else {
                    ui.visuals().text_color()
                };
                ui.painter().image(
                    texture.id(),
                    icon_rect,
                    egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                    tint,
                );
                r
            });
            response.inner.clicked()
        } else {
            ui.selectable_label(selected, format!("{} {}", icon.emoji(), text)).clicked()
        }
    }

    /// Create a menu item button with a small icon + text label
    pub fn menu_item(&self, ui: &mut egui::Ui, icon: Icon, text: &str) -> egui::Response {
        if let Some(texture) = self.textures.get(&icon) {
            let sized_texture = egui::load::SizedTexture::from_handle(texture);
            let img = egui::Image::from_texture(sized_texture).fit_to_exact_size(Vec2::splat(16.0));
            ui.add(egui::Button::image_and_text(img, text))
        } else {
            ui.button(format!("{} {}", icon.emoji(), text))
        }
    }

    /// Create a menu item button with a small icon + text label, with enabled/disabled state
    pub fn menu_item_enabled(&self, ui: &mut egui::Ui, icon: Icon, text: &str, enabled: bool) -> egui::Response {
        if let Some(texture) = self.textures.get(&icon) {
            let sized_texture = egui::load::SizedTexture::from_handle(texture);
            let img = egui::Image::from_texture(sized_texture).fit_to_exact_size(Vec2::splat(16.0));
            ui.add_enabled(enabled, egui::Button::image_and_text(img, text))
        } else {
            ui.add_enabled(enabled, egui::Button::new(format!("{} {}", icon.emoji(), text)))
        }
    }

    /// Create a menu item button with a small icon + text label + right-aligned shortcut hint
    pub fn menu_item_shortcut(&self, ui: &mut egui::Ui, icon: Icon, text: &str, kb: &KeyBindings, action: BindableAction) -> egui::Response {
        let shortcut = kb.get(action).map(|c| c.display()).unwrap_or_default();
        if shortcut.is_empty() {
            return self.menu_item(ui, icon, text);
        }
        self.render_menu_item_with_shortcut(ui, icon, text, &shortcut, true)
    }

    /// Create a menu item button with icon + text + shortcut hint, with enabled/disabled state
    pub fn menu_item_shortcut_enabled(&self, ui: &mut egui::Ui, icon: Icon, text: &str, enabled: bool, kb: &KeyBindings, action: BindableAction) -> egui::Response {
        let shortcut = kb.get(action).map(|c| c.display()).unwrap_or_default();
        if shortcut.is_empty() {
            return self.menu_item_enabled(ui, icon, text, enabled);
        }
        self.render_menu_item_with_shortcut(ui, icon, text, &shortcut, enabled)
    }

    /// Render a menu item with icon + label on left, shortcut text right-aligned
    fn render_menu_item_with_shortcut(&self, ui: &mut egui::Ui, icon: Icon, text: &str, shortcut: &str, enabled: bool) -> egui::Response {
        let icon_size = 16.0_f32;
        let padding = ui.spacing().button_padding;
        let icon_gap = 4.0_f32;

        let text_font = egui::TextStyle::Button.resolve(ui.style());
        let shortcut_font = egui::FontId::proportional(text_font.size * 0.9);

        let label_color = if enabled { ui.visuals().text_color() } else { ui.visuals().weak_text_color() };
        let shortcut_color = ui.visuals().weak_text_color();
        let text_galley = ui.painter().layout_no_wrap(text.to_string(), text_font.clone(), label_color);
        let shortcut_galley = ui.painter().layout_no_wrap(shortcut.to_string(), shortcut_font.clone(), shortcut_color);

        let min_shortcut_gap = 24.0_f32;
        let desired_width = (padding.x + icon_size + icon_gap + text_galley.size().x + min_shortcut_gap + shortcut_galley.size().x + padding.x)
            .max(ui.available_width());
        let row_height = icon_size.max(text_galley.size().y).max(shortcut_galley.size().y) + padding.y * 2.0;

        let sense = if enabled { egui::Sense::click() } else { egui::Sense::hover() };
        let (rect, response) = ui.allocate_exact_size(
            egui::vec2(desired_width, row_height),
            sense,
        );

        if ui.is_rect_visible(rect) {
            // Hover/active background (only for enabled items)
            let visuals = ui.style().interact(&response);
            if enabled && (response.hovered() || response.has_focus()) {
                ui.painter().rect_filled(rect, 0.0, visuals.bg_fill);
            }

            let text_color = if enabled { visuals.text_color() } else { ui.visuals().weak_text_color() };

            // Icon
            let icon_rect = egui::Rect::from_min_size(
                egui::pos2(rect.left() + padding.x, rect.center().y - icon_size / 2.0),
                egui::vec2(icon_size, icon_size),
            );
            if let Some(texture) = self.textures.get(&icon) {
                let tint = if enabled { text_color } else { Color32::from_gray(128) };
                ui.painter().image(
                    texture.id(),
                    icon_rect,
                    egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                    tint,
                );
            }

            // Label text (left-aligned after icon)
            let text_pos = egui::pos2(
                icon_rect.right() + icon_gap,
                rect.center().y - text_galley.size().y / 2.0,
            );
            ui.painter().galley(text_pos, text_galley);

            // Shortcut text (right-aligned)
            let shortcut_pos = egui::pos2(
                rect.right() - padding.x - shortcut_galley.size().x,
                rect.center().y - shortcut_galley.size().y / 2.0,
            );
            ui.painter().galley(shortcut_pos, shortcut_galley);
        }

        response
    }

    /// Create a menu item with icon + text label, shortcut shown below in smaller text
    pub fn menu_item_shortcut_below(&self, ui: &mut egui::Ui, icon: Icon, text: &str, kb: &KeyBindings, action: BindableAction) -> egui::Response {
        let shortcut = kb.get(action).map(|c| c.display()).unwrap_or_default();
        if shortcut.is_empty() {
            return self.menu_item(ui, icon, text);
        }
        self.render_menu_item_with_shortcut_below(ui, icon, text, &shortcut, true)
    }

    /// Create a menu item with icon + text label + shortcut below, with enabled/disabled state
    pub fn menu_item_shortcut_below_enabled(&self, ui: &mut egui::Ui, icon: Icon, text: &str, enabled: bool, kb: &KeyBindings, action: BindableAction) -> egui::Response {
        let shortcut = kb.get(action).map(|c| c.display()).unwrap_or_default();
        if shortcut.is_empty() {
            return self.menu_item_enabled(ui, icon, text, enabled);
        }
        self.render_menu_item_with_shortcut_below(ui, icon, text, &shortcut, enabled)
    }

    /// Render a menu item with icon + label, shortcut text below label in smaller font
    fn render_menu_item_with_shortcut_below(&self, ui: &mut egui::Ui, icon: Icon, text: &str, shortcut: &str, enabled: bool) -> egui::Response {
        let icon_size = 16.0_f32;
        let padding = ui.spacing().button_padding;
        let icon_gap = 4.0_f32;

        let text_font = egui::TextStyle::Button.resolve(ui.style());
        let shortcut_font = egui::FontId::proportional(text_font.size * 0.78);

        let label_color = if enabled { ui.visuals().text_color() } else { ui.visuals().weak_text_color() };
        let shortcut_color = ui.visuals().weak_text_color();
        let text_galley = ui.painter().layout_no_wrap(text.to_string(), text_font.clone(), label_color);
        let shortcut_galley = ui.painter().layout_no_wrap(shortcut.to_string(), shortcut_font.clone(), shortcut_color);

        let text_width = text_galley.size().x.max(shortcut_galley.size().x);
        let desired_width = (padding.x + icon_size + icon_gap + text_width + padding.x)
            .max(ui.available_width());
        let line_gap = 1.0_f32;
        let row_height = (text_galley.size().y + line_gap + shortcut_galley.size().y)
            .max(icon_size) + padding.y * 2.0;

        let sense = if enabled { egui::Sense::click() } else { egui::Sense::hover() };
        let (rect, response) = ui.allocate_exact_size(
            egui::vec2(desired_width, row_height),
            sense,
        );

        if ui.is_rect_visible(rect) {
            let visuals = ui.style().interact(&response);
            if enabled && (response.hovered() || response.has_focus()) {
                ui.painter().rect_filled(rect, 0.0, visuals.bg_fill);
            }

            let text_color = if enabled { visuals.text_color() } else { ui.visuals().weak_text_color() };

            // Icon (vertically centered)
            let icon_rect = egui::Rect::from_min_size(
                egui::pos2(rect.left() + padding.x, rect.center().y - icon_size / 2.0),
                egui::vec2(icon_size, icon_size),
            );
            if let Some(texture) = self.textures.get(&icon) {
                let tint = if enabled { text_color } else { Color32::from_gray(128) };
                ui.painter().image(
                    texture.id(),
                    icon_rect,
                    egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                    tint,
                );
            }

            let text_x = icon_rect.right() + icon_gap;
            let total_text_h = text_galley.size().y + line_gap + shortcut_galley.size().y;
            let text_top = rect.center().y - total_text_h / 2.0;

            // Label text
            ui.painter().galley(egui::pos2(text_x, text_top), text_galley);

            // Shortcut text below
            ui.painter().galley(
                egui::pos2(text_x, text_top + total_text_h - shortcut_galley.size().y),
                shortcut_galley,
            );
        }

        response
    }
}

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

// ═══════════════════════════════════════════════════════════════════════════
// KEYBINDINGS SYSTEM
// ═══════════════════════════════════════════════════════════════════════════

/// A single key combination (modifier flags + optional key + optional text char)
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
    pub fn key(k: egui::Key) -> Self {
        Self { ctrl: false, shift: false, alt: false, key: Some(k), text_char: None }
    }
    pub fn ctrl_key(k: egui::Key) -> Self {
        Self { ctrl: true, shift: false, alt: false, key: Some(k), text_char: None }
    }
    pub fn ctrl_shift_key(k: egui::Key) -> Self {
        Self { ctrl: true, shift: true, alt: false, key: Some(k), text_char: None }
    }
    pub fn text(s: &str) -> Self {
        Self { ctrl: false, shift: false, alt: false, key: None, text_char: Some(s.to_string()) }
    }

    /// Human-readable display string
    pub fn display(&self) -> String {
        let mut parts = Vec::new();
        if self.ctrl { parts.push("Ctrl"); }
        if self.shift { parts.push("Shift"); }
        if self.alt { parts.push("Alt"); }
        if let Some(ref k) = self.key {
            parts.push(key_name(*k));
        } else if let Some(ref t) = self.text_char {
            parts.push(t.as_str());
        }
        parts.join("+")
    }

    /// Serialize to config string
    pub fn to_config_string(&self) -> String {
        let mut parts = Vec::new();
        if self.ctrl { parts.push("ctrl".to_string()); }
        if self.shift { parts.push("shift".to_string()); }
        if self.alt { parts.push("alt".to_string()); }
        if let Some(ref k) = self.key {
            parts.push(format!("key:{}", key_name(*k)));
        } else if let Some(ref t) = self.text_char {
            parts.push(format!("text:{}", t));
        }
        parts.join("+")
    }

    /// Deserialize from config string
    pub fn from_config_string(s: &str) -> Option<Self> {
        let mut combo = Self { ctrl: false, shift: false, alt: false, key: None, text_char: None };
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
        if combo.key.is_some() || combo.text_char.is_some() {
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
    BrushSizeDecrease,
    BrushSizeIncrease,
}

impl BindableAction {
    /// Human-readable name for display
    pub fn display_name(&self) -> String {
        match self {
            Self::NewFile => t!("keybind.new_file"),
            Self::OpenFile => t!("keybind.open_file"),
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
            Self::BrushSizeDecrease => t!("keybind.brush_size_decrease"),
            Self::BrushSizeIncrease => t!("keybind.brush_size_increase"),
        }
    }

    /// Category for grouping in UI
    pub fn category(&self) -> String {
        match self {
            Self::NewFile | Self::OpenFile | Self::Save | Self::SaveAll | Self::SaveAs => t!("keybind_category.file"),
            Self::Undo | Self::Redo | Self::Copy | Self::Cut | Self::Paste
            | Self::SelectAll | Self::Deselect
            | Self::FlattenLayers => t!("keybind_category.edit"),
            Self::ResizeImage | Self::ResizeCanvas => t!("keybind_category.canvas"),
            Self::ViewZoomIn | Self::ViewZoomOut | Self::ViewFitToWindow => t!("keybind_category.view"),
            Self::ToolBrush | Self::ToolEraser | Self::ToolPencil | Self::ToolLine
            | Self::ToolGradient | Self::ToolFill | Self::ToolMagicWand
            | Self::ToolColorPicker | Self::ToolMovePixels | Self::ToolRectSelect
            | Self::ToolText | Self::ToolZoom | Self::ToolPan | Self::ToolCloneStamp
            | Self::ToolShapes | Self::ToolLasso | Self::ToolColorRemover
            | Self::ToolMeshWarp => t!("keybind_category.tools"),
            Self::BrushSizeDecrease | Self::BrushSizeIncrease => t!("keybind_category.brush"),
        }
    }

    /// All actions in display order
    pub fn all() -> &'static [BindableAction] {
        use BindableAction::*;
        &[
            NewFile, OpenFile, Save, SaveAll, SaveAs,
            Undo, Redo, Copy, Cut, Paste, SelectAll, Deselect, FlattenLayers,
            ResizeImage, ResizeCanvas,
            ViewZoomIn, ViewZoomOut, ViewFitToWindow,
            ToolBrush, ToolEraser, ToolPencil, ToolLine, ToolGradient, ToolFill,
            ToolMagicWand, ToolColorPicker, ToolMovePixels, ToolRectSelect,
            ToolText, ToolZoom, ToolPan, ToolCloneStamp, ToolShapes, ToolLasso,
            ToolColorRemover, ToolMeshWarp,
            BrushSizeDecrease, BrushSizeIncrease,
        ]
    }
}

/// Keybinding map — stores all customizable keyboard shortcuts
#[derive(Clone, Debug)]
pub struct KeyBindings {
    pub bindings: std::collections::HashMap<BindableAction, KeyCombo>,
}

impl Default for KeyBindings {
    fn default() -> Self {
        use BindableAction::*;
        use egui::Key;
        let mut map = std::collections::HashMap::new();
        // File
        map.insert(NewFile,           KeyCombo::ctrl_key(Key::N));
        map.insert(OpenFile,          KeyCombo::ctrl_key(Key::O));
        map.insert(Save,              KeyCombo::ctrl_key(Key::S));
        map.insert(SaveAll,           KeyCombo { ctrl: true, shift: false, alt: true, key: Some(Key::S), text_char: None });
        map.insert(SaveAs,            KeyCombo::ctrl_shift_key(Key::S));
        // Edit
        map.insert(Undo,              KeyCombo::ctrl_key(Key::Z));
        map.insert(Redo,              KeyCombo::ctrl_key(Key::Y));
        map.insert(Copy,              KeyCombo::ctrl_key(Key::C));
        map.insert(Cut,               KeyCombo::ctrl_key(Key::X));
        map.insert(Paste,             KeyCombo::ctrl_key(Key::V));
        map.insert(SelectAll,         KeyCombo::ctrl_key(Key::A));
        map.insert(Deselect,          KeyCombo::ctrl_key(Key::D));
        map.insert(FlattenLayers,     KeyCombo::ctrl_shift_key(Key::F));
        // Canvas
        map.insert(ResizeImage,       KeyCombo::ctrl_key(Key::R));
        map.insert(ResizeCanvas,      KeyCombo::ctrl_shift_key(Key::R));
        // View
        map.insert(ViewZoomIn,        KeyCombo::ctrl_key(Key::PlusEquals));
        map.insert(ViewZoomOut,       KeyCombo::ctrl_key(Key::Minus));
        map.insert(ViewFitToWindow,   KeyCombo::ctrl_key(Key::Num0));
        // Tools
        map.insert(ToolBrush,         KeyCombo::key(Key::B));
        map.insert(ToolEraser,        KeyCombo::key(Key::E));
        map.insert(ToolPencil,        KeyCombo::key(Key::P));
        map.insert(ToolLine,          KeyCombo::key(Key::L));
        map.insert(ToolGradient,      KeyCombo::key(Key::G));
        map.insert(ToolFill,          KeyCombo::key(Key::F));
        map.insert(ToolMagicWand,     KeyCombo::key(Key::W));
        map.insert(ToolColorPicker,   KeyCombo::key(Key::I));
        map.insert(ToolMovePixels,    KeyCombo::key(Key::M));
        map.insert(ToolRectSelect,    KeyCombo::key(Key::S));
        map.insert(ToolText,          KeyCombo::key(Key::T));
        map.insert(ToolZoom,          KeyCombo::key(Key::Z));
        map.insert(ToolPan,           KeyCombo::key(Key::H));
        map.insert(ToolCloneStamp,    KeyCombo::key(Key::K));
        map.insert(ToolShapes,        KeyCombo::key(Key::U));
        map.insert(ToolLasso,         KeyCombo::key(Key::J));
        map.insert(ToolColorRemover,  KeyCombo::key(Key::R));
        map.insert(ToolMeshWarp,      KeyCombo::key(Key::Q));
        // Brush size
        map.insert(BrushSizeDecrease, KeyCombo::text("["));
        map.insert(BrushSizeIncrease, KeyCombo::text("]"));

        Self { bindings: map }
    }
}

impl KeyBindings {
    /// Returns the combo for an action, falling back to default if missing.
    pub fn get(&self, action: BindableAction) -> Option<&KeyCombo> {
        self.bindings.get(&action)
    }

    /// Set a binding
    pub fn set(&mut self, action: BindableAction, combo: KeyCombo) {
        self.bindings.insert(action, combo);
    }

    /// Serialize all bindings for config file
    pub fn to_config_lines(&self) -> Vec<String> {
        let mut lines = Vec::new();
        for action in BindableAction::all() {
            if let Some(combo) = self.bindings.get(action) {
                lines.push(format!("keybind.{:?}={}", action, combo.to_config_string()));
            }
        }
        lines
    }

    /// Load a single keybind line from config
    pub fn load_config_line(&mut self, action_name: &str, combo_str: &str) {
        let action = match action_name {
            "NewFile" => Some(BindableAction::NewFile),
            "OpenFile" => Some(BindableAction::OpenFile),
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
            "BrushSizeDecrease" => Some(BindableAction::BrushSizeDecrease),
            "BrushSizeIncrease" => Some(BindableAction::BrushSizeIncrease),
            _ => None,
        };
        if let Some(action) = action {
            if let Some(combo) = KeyCombo::from_config_string(combo_str) {
                self.bindings.insert(action, combo);
            }
        }
    }

    /// Check if a keybinding was triggered this frame
    pub fn is_pressed(&self, ctx: &egui::Context, action: BindableAction) -> bool {
        let Some(combo) = self.bindings.get(&action) else { return false };
        
        if let Some(ref text_char) = combo.text_char {
            // Text-based binding (like [ or ])
            ctx.input(|i| {
                // Check modifier match
                if combo.ctrl != i.modifiers.command { return false; }
                if combo.shift != i.modifiers.shift { return false; }
                if combo.alt != i.modifiers.alt { return false; }
                for ev in &i.events {
                    if let egui::Event::Text(t) = ev {
                        if t == text_char { return true; }
                    }
                }
                false
            })
        } else if let Some(key) = combo.key {
            let mods = egui::Modifiers {
                alt: combo.alt,
                ctrl: combo.ctrl,
                shift: combo.shift,
                mac_cmd: false,
                command: combo.ctrl,
            };
            ctx.input_mut(|i| i.consume_key(mods, key))
        } else {
            false
        }
    }
}

/// Convert egui::Key to display name
fn key_name(k: egui::Key) -> &'static str {
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
        egui::Key::PlusEquals => "+",
        egui::Key::Num0 => "0", egui::Key::Num1 => "1", egui::Key::Num2 => "2",
        egui::Key::Num3 => "3", egui::Key::Num4 => "4", egui::Key::Num5 => "5",
        egui::Key::Num6 => "6", egui::Key::Num7 => "7", egui::Key::Num8 => "8",
        egui::Key::Num9 => "9",
        egui::Key::A => "A", egui::Key::B => "B", egui::Key::C => "C",
        egui::Key::D => "D", egui::Key::E => "E", egui::Key::F => "F",
        egui::Key::G => "G", egui::Key::H => "H", egui::Key::I => "I",
        egui::Key::J => "J", egui::Key::K => "K", egui::Key::L => "L",
        egui::Key::M => "M", egui::Key::N => "N", egui::Key::O => "O",
        egui::Key::P => "P", egui::Key::Q => "Q", egui::Key::R => "R",
        egui::Key::S => "S", egui::Key::T => "T", egui::Key::U => "U",
        egui::Key::V => "V", egui::Key::W => "W", egui::Key::X => "X",
        egui::Key::Y => "Y", egui::Key::Z => "Z",
        egui::Key::F1 => "F1", egui::Key::F2 => "F2", egui::Key::F3 => "F3",
        egui::Key::F4 => "F4", egui::Key::F5 => "F5", egui::Key::F6 => "F6",
        egui::Key::F7 => "F7", egui::Key::F8 => "F8", egui::Key::F9 => "F9",
        egui::Key::F10 => "F10", egui::Key::F11 => "F11", egui::Key::F12 => "F12",
        egui::Key::F13 => "F13", egui::Key::F14 => "F14", egui::Key::F15 => "F15",
        egui::Key::F16 => "F16", egui::Key::F17 => "F17", egui::Key::F18 => "F18",
        egui::Key::F19 => "F19", egui::Key::F20 => "F20",
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
        "+" => Some(egui::Key::PlusEquals),
        "0" => Some(egui::Key::Num0), "1" => Some(egui::Key::Num1), "2" => Some(egui::Key::Num2),
        "3" => Some(egui::Key::Num3), "4" => Some(egui::Key::Num4), "5" => Some(egui::Key::Num5),
        "6" => Some(egui::Key::Num6), "7" => Some(egui::Key::Num7), "8" => Some(egui::Key::Num8),
        "9" => Some(egui::Key::Num9),
        "A" => Some(egui::Key::A), "B" => Some(egui::Key::B), "C" => Some(egui::Key::C),
        "D" => Some(egui::Key::D), "E" => Some(egui::Key::E), "F" => Some(egui::Key::F),
        "G" => Some(egui::Key::G), "H" => Some(egui::Key::H), "I" => Some(egui::Key::I),
        "J" => Some(egui::Key::J), "K" => Some(egui::Key::K), "L" => Some(egui::Key::L),
        "M" => Some(egui::Key::M), "N" => Some(egui::Key::N), "O" => Some(egui::Key::O),
        "P" => Some(egui::Key::P), "Q" => Some(egui::Key::Q), "R" => Some(egui::Key::R),
        "S" => Some(egui::Key::S), "T" => Some(egui::Key::T), "U" => Some(egui::Key::U),
        "V" => Some(egui::Key::V), "W" => Some(egui::Key::W), "X" => Some(egui::Key::X),
        "Y" => Some(egui::Key::Y), "Z" => Some(egui::Key::Z),
        "F1" => Some(egui::Key::F1), "F2" => Some(egui::Key::F2), "F3" => Some(egui::Key::F3),
        "F4" => Some(egui::Key::F4), "F5" => Some(egui::Key::F5), "F6" => Some(egui::Key::F6),
        "F7" => Some(egui::Key::F7), "F8" => Some(egui::Key::F8), "F9" => Some(egui::Key::F9),
        "F10" => Some(egui::Key::F10), "F11" => Some(egui::Key::F11), "F12" => Some(egui::Key::F12),
        "F13" => Some(egui::Key::F13), "F14" => Some(egui::Key::F14), "F15" => Some(egui::Key::F15),
        "F16" => Some(egui::Key::F16), "F17" => Some(egui::Key::F17), "F18" => Some(egui::Key::F18),
        "F19" => Some(egui::Key::F19), "F20" => Some(egui::Key::F20),
        _ => None,
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
    /// Enable GPU acceleration (placeholder)
    pub gpu_acceleration: bool,
    /// Preferred GPU adapter name (e.g. \"NVIDIA RTX 3080\")
    pub preferred_gpu: String,
    /// Pixel grid display mode
    pub pixel_grid_mode: PixelGridMode,
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

    // Debug panel settings
    pub show_debug_panel: bool,
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

    // Behaviour
    /// Show a save-confirmation dialog when the user exits with unsaved projects.
    pub confirm_on_exit: bool,
}

impl Default for AppSettings {
    fn default() -> Self {
        let preset = ThemePreset::Blue;
        Self {
            theme_mode: ThemeMode::Light,
            theme_preset: preset,
            custom_accent: preset.accent_colors(),
            gpu_acceleration: true,
            preferred_gpu: "Auto".to_string(),
            pixel_grid_mode: PixelGridMode::Auto,
            max_undo_steps: 50,
            auto_save_minutes: 0,
            neon_mode: false,
            zoom_filter_mode: ZoomFilterMode::Linear,
            checkerboard_brightness: 1.0,
            onnx_runtime_path: String::new(),
            birefnet_model_path: String::new(),
            
            show_debug_panel: true,
            debug_show_canvas_size: true,
            debug_show_zoom: true,
            debug_show_fps: true,
            debug_show_gpu: false,
            debug_show_operations: true,

            keybindings: KeyBindings::default(),

            language: String::new(), // empty = auto-detect on first boot

            confirm_on_exit: true,
        }
    }
}

impl AppSettings {
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
            return Some(config_dir.join("paintfe_settings.cfg"));
        }
        #[cfg(target_os = "windows")]
        {
            // Use %APPDATA% so the settings are stored in the user profile and isolated
            // from other users — avoids the security issue of a world-writable EXE directory.
            let appdata = std::env::var("APPDATA").or_else(|_| std::env::var("USERPROFILE")).unwrap_or_else(|_| {
                std::env::current_exe()
                    .ok()
                    .and_then(|p| p.parent().map(|d| d.to_string_lossy().into_owned()))
                    .unwrap_or_default()
            });
            let config_dir = PathBuf::from(appdata).join("PaintFE");
            let _ = std::fs::create_dir_all(&config_dir);
            return Some(config_dir.join("paintfe_settings.cfg"));
        }
        #[cfg(target_os = "macos")]
        {
            let home = std::env::var("HOME").unwrap_or_else(|_| "~".to_string());
            let config_dir = PathBuf::from(home)
                .join("Library")
                .join("Application Support")
                .join("PaintFE");
            let _ = std::fs::create_dir_all(&config_dir);
            return Some(config_dir.join("paintfe_settings.cfg"));
        }
        #[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "macos")))]
        {
            std::env::current_exe().ok().and_then(|p| p.parent().map(|d| d.join("paintfe_settings.cfg")))
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

    /// Save settings to disk
    pub fn save(&self) {
        let Some(path) = Self::settings_path() else { return };
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
             gpu_acceleration={}\n\
             preferred_gpu={}\n\
             pixel_grid_mode={grid_str}\n\
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
             language={}\n\
             confirm_on_exit={}\n",
            self.gpu_acceleration,
            self.preferred_gpu,
            self.max_undo_steps,
            self.auto_save_minutes,
            Self::color_to_str(self.custom_accent.light_normal),
            Self::color_to_str(self.custom_accent.light_faint),
            Self::color_to_str(self.custom_accent.light_strong),
            Self::color_to_str(self.custom_accent.dark_normal),
            Self::color_to_str(self.custom_accent.dark_faint),
            Self::color_to_str(self.custom_accent.dark_strong),
            self.neon_mode,
            self.checkerboard_brightness,
            self.onnx_runtime_path,
            self.birefnet_model_path,
            self.language,
            self.confirm_on_exit,
        );
        // Append keybinding lines
        let mut content = content;
        for line in self.keybindings.to_config_lines() {
            content.push_str(&line);
            content.push('\n');
        }
        let _ = std::fs::write(path, content);
    }

    /// Load settings from disk (returns default if file missing or corrupt)
    pub fn load() -> Self {
        let Some(path) = Self::settings_path() else { return Self::default() };
        let Ok(content) = std::fs::read_to_string(&path) else { return Self::default() };

        let mut s = Self::default();
        for line in content.lines() {
            let Some((key, val)) = line.split_once('=') else { continue };
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
                        "custom" => ThemePreset::Custom,
                        _ => ThemePreset::Blue,
                    };
                }
                "gpu_acceleration" => {
                    s.gpu_acceleration = val == "true";
                }
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
                "max_undo_steps" => {
                    s.max_undo_steps = val.parse().unwrap_or(50);
                }
                "auto_save_minutes" => {
                    s.auto_save_minutes = val.parse().unwrap_or(0);
                }
                "accent_light_normal" => {
                    if let Some(c) = Self::str_to_color(val) { s.custom_accent.light_normal = c; }
                }
                "accent_light_faint" => {
                    if let Some(c) = Self::str_to_color(val) { s.custom_accent.light_faint = c; }
                }
                "accent_light_strong" => {
                    if let Some(c) = Self::str_to_color(val) { s.custom_accent.light_strong = c; }
                }
                "accent_dark_normal" => {
                    if let Some(c) = Self::str_to_color(val) { s.custom_accent.dark_normal = c; }
                }
                "accent_dark_faint" => {
                    if let Some(c) = Self::str_to_color(val) { s.custom_accent.dark_faint = c; }
                }
                "accent_dark_strong" => {
                    if let Some(c) = Self::str_to_color(val) { s.custom_accent.dark_strong = c; }
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
                "language" => {
                    s.language = val.to_string();
                }
                "confirm_on_exit" => {
                    s.confirm_on_exit = val == "true";
                }
                _ => {
                    // Parse keybinding lines: keybind.ActionName=combo
                    if let Some(action_name) = key.strip_prefix("keybind.") {
                        s.keybindings.load_config_line(action_name, val);
                    }
                }
            }
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
        &[PixelGridMode::Auto, PixelGridMode::AlwaysOn, PixelGridMode::AlwaysOff]
    }
}

/// Settings window state — split-view with sidebar navigation
pub struct SettingsWindow {
    pub open: bool,
    active_tab: SettingsTab,
    /// Staging copy of accent colors for the "Interface" tab (applied on "Apply")
    staged_accent: AccentColors,
    staged_preset: ThemePreset,
    staged_mode: ThemeMode,
    /// Whether staged values need applying
    dirty: bool,
    /// Cached list of available GPU adapter names
    gpu_adapters: Vec<String>,
    /// Receiver for the background GPU enumeration task (None when not running)
    gpu_adapters_receiver: Option<std::sync::mpsc::Receiver<Vec<String>>>,
    /// Staging copy of ONNX Runtime DLL path (AI tab)
    staged_onnx_path: String,
    /// Staging copy of BiRefNet model path (AI tab)
    staged_model_path: String,
    /// Result of last ONNX Runtime probe (None = not tested yet)
    onnx_probe_result: Option<Result<String, String>>,
    /// Staging copy of keybindings for the Keybinds tab
    staged_keybindings: KeyBindings,
    /// Which action is currently being rebound (waiting for key press)
    rebinding_action: Option<BindableAction>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum SettingsTab {
    General,
    Interface,
    Hardware,
    Keybinds,
    AI,
}

impl Default for SettingsWindow {
    fn default() -> Self {
        let preset = ThemePreset::Blue;
        Self {
            open: false,
            active_tab: SettingsTab::General,
            staged_accent: preset.accent_colors(),
            staged_preset: preset,
            staged_mode: ThemeMode::Light,
            dirty: false,
            gpu_adapters: Vec::new(),
            gpu_adapters_receiver: None,
            staged_onnx_path: String::new(),
            staged_model_path: String::new(),
            onnx_probe_result: None,
            staged_keybindings: KeyBindings::default(),
            rebinding_action: None,
        }
    }
}

impl SettingsWindow {
    /// Sync staged state from current settings (call when opening)
    fn sync_from_settings(&mut self, settings: &AppSettings) {
        self.staged_mode = settings.theme_mode;
        self.staged_preset = settings.theme_preset;
        self.staged_accent = if settings.theme_preset == ThemePreset::Custom {
            settings.custom_accent
        } else {
            settings.theme_preset.accent_colors()
        };
        self.staged_onnx_path = settings.onnx_runtime_path.clone();
        self.staged_model_path = settings.birefnet_model_path.clone();
        self.onnx_probe_result = None;
        self.staged_keybindings = settings.keybindings.clone();
        self.rebinding_action = None;
        self.dirty = false;
    }

    pub fn show(&mut self, ctx: &egui::Context, settings: &mut AppSettings, theme: &mut crate::theme::Theme, assets: &Assets) {
        if !self.open {
            return;
        }

        // Sync on first frame the window is shown
        let id = egui::Id::new("settings_sync_flag");
        let was_open_last_frame = ctx.data_mut(|d| {
            let prev: bool = d.get_temp(id).unwrap_or(false);
            d.insert_temp(id, true);
            prev
        });
        if !was_open_last_frame {
            self.sync_from_settings(settings);
        }
        
        let show = self.open;
        let mut should_close = false;
        
        egui::Window::new("settings_window_internal")
            .title_bar(false)
            .resizable(true)
            .collapsible(false)
            .default_width(680.0)
            .default_height(540.0)
            .min_width(600.0)
            .min_height(400.0)
            .show(ctx, |ui| {
                // ── Custom header strip ─────────────────────────────────────
                {
                    let available_width = ui.available_width();
                    let header_height = 32.0;
                    let v = ctx.style().visuals.clone();
                    let accent = v.selection.stroke.color;
                    let is_dark = v.dark_mode;
                    let accent_faint = if is_dark {
                        Color32::from_rgba_unmultiplied(accent.r(), accent.g(), accent.b(), 35)
                    } else {
                        Color32::from_rgba_unmultiplied(accent.r(), accent.g(), accent.b(), 25)
                    };

                    let (rect, _) = ui.allocate_exact_size(
                        Vec2::new(available_width, header_height),
                        Sense::hover(),
                    );
                    let painter = ui.painter();
                    painter.rect_filled(rect, egui::Rounding::ZERO, accent_faint);
                    painter.rect_filled(
                        egui::Rect::from_min_size(rect.min, Vec2::new(3.0, header_height)),
                        egui::Rounding::ZERO,
                        accent,
                    );
                    painter.text(
                        egui::pos2(rect.min.x + 12.0, rect.center().y),
                        egui::Align2::LEFT_CENTER,
                        format!("\u{2699} {}", t!("settings.title")),
                        egui::FontId::proportional(14.0),
                        accent,
                    );

                    // X close button on the right
                    let btn_size = Vec2::splat(header_height);
                    let btn_rect = egui::Rect::from_min_size(
                        egui::pos2(rect.max.x - btn_size.x, rect.min.y),
                        btn_size,
                    );
                    let btn_response = ui.interact(btn_rect, ui.id().with("hdr_close"), Sense::click());
                    if btn_response.hovered() {
                        painter.rect_filled(btn_rect, egui::Rounding::ZERO,
                            Color32::from_rgba_unmultiplied(accent.r(), accent.g(), accent.b(), 55));
                    }
                    painter.text(
                        btn_rect.center(),
                        egui::Align2::CENTER_CENTER,
                        "×",
                        egui::FontId::proportional(14.0),
                        accent,
                    );
                    if btn_response.clicked() {
                        should_close = true;
                    }
                }
                // Force the window body to use the full allocated height so the
                // sidebar doesn't collapse the window to just ~165 px.
                let available_h = ui.available_height();
                ui.set_min_height(available_h);
                ui.horizontal(|ui| {
                    // -- Left Sidebar --
                    ui.vertical(|ui| {
                        ui.set_min_width(132.0);
                        ui.set_max_width(132.0);
                        ui.set_min_height(available_h);
                        ui.add_space(4.0);

                        let tabs: [(SettingsTab, Icon, String); 5] = [
                            (SettingsTab::General,   Icon::SettingsGeneral,   t!("settings.tab.general")),
                            (SettingsTab::Interface, Icon::SettingsInterface, t!("settings.tab.interface")),
                            (SettingsTab::Hardware,  Icon::SettingsHardware,  t!("settings.tab.hardware")),
                            (SettingsTab::Keybinds,  Icon::SettingsKeybinds,  t!("settings.tab.keybinds")),
                            (SettingsTab::AI,        Icon::SettingsAI,        t!("settings.tab.ai")),
                        ];
                        for (tab, icon, label) in &tabs {
                            let selected = self.active_tab == *tab;
                            // Render icon + text as a selectable row
                            let icon_size = Vec2::splat(16.0);
                            let total_width = ui.available_width();
                            let (rect, response) = ui.allocate_exact_size(
                                Vec2::new(total_width, 30.0),
                                Sense::click(),
                            );
                            if response.clicked() {
                                self.active_tab = *tab;
                            }
                            // Draw selection background
                            if selected {
                                ui.painter().rect_filled(rect, 2.0, ui.visuals().selection.bg_fill);
                            } else if response.hovered() {
                                ui.painter().rect_filled(rect, 2.0, ui.visuals().widgets.hovered.bg_fill);
                            }
                            // Draw icon
                            let icon_rect = egui::Rect::from_min_size(
                                egui::pos2(rect.left() + 6.0, rect.center().y - icon_size.y / 2.0),
                                icon_size,
                            );
                            let text_color = if selected {
                                ui.visuals().strong_text_color()
                            } else {
                                ui.visuals().text_color()
                            };
                            if let Some(texture) = assets.textures.get(icon) {
                                ui.painter().image(
                                    texture.id(),
                                    icon_rect,
                                    egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                                    text_color,
                                );
                            }
                            // Draw label
                            let text_pos = egui::pos2(icon_rect.right() + 6.0, rect.center().y - 6.0);
                            ui.painter().text(
                                text_pos,
                                egui::Align2::LEFT_TOP,
                                label,
                                egui::FontId::proportional(12.0),
                                text_color,
                            );
                        }
                    });

                    ui.separator();

                    // -- Right Content Area --
                    ui.vertical(|ui| {
                        ui.set_min_width(450.0);
                        ui.set_min_height(available_h);
                        // Wrap all tab content in a per-tab ScrollArea.
                        // `set_min_height` ensures short tabs don't shrink the window;
                        // tall tabs scroll instead of growing the window.
                        let scroll_id = format!("settings_tab_scroll_{:?}", self.active_tab);
                        egui::ScrollArea::vertical()
                            .id_source(scroll_id)
                            .auto_shrink([false; 2])
                            .max_height(1000.0)
                            .show(ui, |ui| {
                                ui.set_min_height(1000.0);
                                ui.add_space(4.0);
                                match self.active_tab {
                                    SettingsTab::General => {
                                        self.show_general_tab(ui, settings);
                                    }
                                    SettingsTab::Interface => {
                                        self.show_interface_tab(ui, ctx, settings, theme);
                                    }
                                    SettingsTab::Hardware => {
                                        self.show_hardware_tab(ui, settings);
                                    }
                                    SettingsTab::Keybinds => {
                                        self.show_keybinds_tab(ui, settings);
                                    }
                                    SettingsTab::AI => {
                                        self.show_ai_tab(ui, settings);
                                    }
                                }
                            });
                    });
                });
            });

        self.open = show && !should_close;
        if !self.open {
            // Clear the sync flag when window closes
            ctx.data_mut(|d| d.insert_temp(id, false));
        }
    }

    /// Draws a bold section label followed by a thin separator rule.
    /// Used consistently across all settings tabs.
    fn section_header(ui: &mut egui::Ui, label: &str) {
        ui.add_space(10.0);
        ui.label(egui::RichText::new(label).strong().size(13.0));
        ui.separator();
        ui.add_space(3.0);
    }

    // -- General Tab -------------------------------------------
    fn show_general_tab(&mut self, ui: &mut egui::Ui, settings: &mut AppSettings) {
        // -- Language -------------------------------------------
        Self::section_header(ui, "Language");
        ui.horizontal(|ui| {
            ui.label("Language:");
            let current_code = if settings.language.is_empty() { "auto".to_string() } else { settings.language.clone() };
            let display_text = if current_code == "auto" {
                "Auto (System)".to_string()
            } else {
                crate::i18n::LANGUAGES.iter()
                    .find(|(c, _)| *c == current_code.as_str())
                    .map(|(_, name)| name.to_string())
                    .unwrap_or_else(|| current_code.clone())
            };
            egui::ComboBox::from_id_source("language_select")
                .selected_text(&display_text)
                .show_ui(ui, |ui| {
                    if ui.selectable_label(current_code == "auto", "Auto (System)").clicked() {
                        settings.language = String::new();
                        let detected = crate::i18n::detect_system_language();
                        crate::i18n::set_language(&detected);
                    }
                    for &(code, name) in crate::i18n::LANGUAGES {
                        if ui.selectable_label(current_code.as_str() == code, format!("{} ({})", name, code)).clicked() {
                            settings.language = code.to_string();
                            crate::i18n::set_language(code);
                        }
                    }
                });
        });
        ui.label(egui::RichText::new("Restart recommended after changing language.").small().weak());

        // -- History & Auto-save -------------------------------------------
        Self::section_header(ui, &t!("settings.general.history"));
        egui::Grid::new("general_history_grid")
            .num_columns(2)
            .spacing([16.0, 6.0])
            .min_col_width(160.0)
            .show(ui, |ui| {
                ui.label(t!("settings.general.max_undo_steps"));
                ui.add(egui::DragValue::new(&mut settings.max_undo_steps)
                    .clamp_range(10..=500)
                    .speed(1.0)
                    .suffix(" steps"));
                ui.end_row();

                ui.label("Auto-save interval:");
                {
                    const OPTIONS: &[(u32, &str)] = &[
                        (0,  "Disabled"),
                        (1,  "Every 1 minute"),
                        (3,  "Every 3 minutes"),
                        (5,  "Every 5 minutes"),
                        (10, "Every 10 minutes"),
                        (15, "Every 15 minutes"),
                        (30, "Every 30 minutes"),
                    ];
                    let current_label = OPTIONS.iter()
                        .find(|(v, _)| *v == settings.auto_save_minutes)
                        .map(|(_, l)| *l)
                        .unwrap_or("Custom");
                    egui::ComboBox::from_id_source("auto_save_interval")
                        .selected_text(current_label)
                        .show_ui(ui, |ui| {
                            for &(val, label) in OPTIONS {
                                if ui.selectable_label(settings.auto_save_minutes == val, label).clicked() {
                                    settings.auto_save_minutes = val;
                                }
                            }
                        });
                }
                ui.end_row();
            });
        ui.label(egui::RichText::new(t!("settings.general.max_undo_hint")).small().weak());
        if settings.auto_save_minutes > 0 {
            if let Some(dir) = crate::io::autosave_dir() {
                ui.label(egui::RichText::new(format!("Auto-saves to: {}", dir.display())).small().weak());
            }
        } else {
            ui.label(egui::RichText::new("Auto-save is disabled — enable it to protect against crashes.").small().weak());
        }

        // -- Canvas Display -------------------------------------------
        Self::section_header(ui, &t!("settings.general.display"));
        ui.horizontal(|ui| {
            ui.label(t!("settings.general.pixel_grid"));
            egui::ComboBox::from_id_source("pixel_grid_mode")
                .selected_text(settings.pixel_grid_mode.name())
                .show_ui(ui, |ui| {
                    for &mode in PixelGridMode::all() {
                        if ui.selectable_label(mode == settings.pixel_grid_mode, mode.name()).clicked() {
                            settings.pixel_grid_mode = mode;
                        }
                    }
                });
        });

        // -- Effects -----------------------------------------------------
        Self::section_header(ui, &t!("settings.general.effects"));
        ui.checkbox(&mut settings.neon_mode, t!("settings.general.neon_mode"));
        ui.label(egui::RichText::new(t!("settings.general.neon_hint")).small().weak());

        // -- Behaviour -------------------------------------------------
        Self::section_header(ui, "Behaviour");
        ui.checkbox(&mut settings.confirm_on_exit, "Confirm before exiting with unsaved changes");
        ui.label(egui::RichText::new("Shows a save prompt when quitting with unsaved projects.").small().weak());

        // -- Debug Info ---------------------------------------------------
        Self::section_header(ui, &t!("settings.general.debug_info"));
        ui.checkbox(&mut settings.show_debug_panel, t!("settings.general.show_debug"));
        ui.label(egui::RichText::new(t!("settings.general.debug_hint")).small().weak());
        if settings.show_debug_panel {
            ui.add_space(4.0);
            ui.indent("debug_options", |ui| {
                ui.checkbox(&mut settings.debug_show_canvas_size, t!("settings.general.debug_canvas_size"));
                ui.checkbox(&mut settings.debug_show_zoom, t!("settings.general.debug_zoom"));
                ui.checkbox(&mut settings.debug_show_fps, t!("settings.general.debug_fps"));
                ui.checkbox(&mut settings.debug_show_gpu, t!("settings.general.debug_gpu"));
                ui.checkbox(&mut settings.debug_show_operations, t!("settings.general.debug_operations"));
            });
        }

        // -- About ------------------------------------------------------
        Self::section_header(ui, "About");
        egui::Grid::new("about_grid")
            .num_columns(2)
            .spacing([16.0, 6.0])
            .show(ui, |ui| {
                ui.label(egui::RichText::new("Version").strong());
                ui.label(env!("CARGO_PKG_VERSION"));
                ui.end_row();

                ui.label(egui::RichText::new("Author").strong());
                ui.label("Kyle Jackson");
                ui.end_row();

                ui.label(egui::RichText::new("License").strong());
                ui.label("MIT");
                ui.end_row();

                ui.label(egui::RichText::new("Source").strong());
                ui.hyperlink_to("paintfe.com", "https://paintfe.com");
                ui.end_row();
            });

        // -- Reset ------------------------------------------------------
        ui.add_space(16.0);
        ui.separator();
        ui.add_space(4.0);
        if ui.button(t!("settings.general.reset_defaults")).clicked() {
            *settings = AppSettings::default();
        }
        settings.save();
    }

    // -- Hardware Tab -----------------------------------------------
    fn show_hardware_tab(&mut self, ui: &mut egui::Ui, settings: &mut AppSettings) {
        // -- Acceleration ----------------------------------------------
        Self::section_header(ui, "GPU Acceleration");
        ui.checkbox(&mut settings.gpu_acceleration, "Enable hardware-accelerated rendering");
        ui.label(egui::RichText::new("Disable only if you experience rendering glitches. Takes effect after restart.").small().weak());

        // -- Preferred Adapter --------------------------------------------
        Self::section_header(ui, &t!("settings.hardware.preferred_gpu"));

        // Poll background GPU enumeration result
        if let Some(rx) = &self.gpu_adapters_receiver {
            if let Ok(adapters) = rx.try_recv() {
                self.gpu_adapters = adapters;
                self.gpu_adapters_receiver = None;
            }
        }

        // Kick off background enumeration on first visit (non-blocking)
        if self.gpu_adapters.is_empty() && self.gpu_adapters_receiver.is_none() {
            let (tx, rx) = std::sync::mpsc::channel();
            self.gpu_adapters_receiver = Some(rx);
            std::thread::spawn(move || {
                // Use wgpu's own adapter enumeration — zero child processes spawned,
                // no console flash, and it returns the same GPU names the app actually uses.
                let mut adapters = vec!["Auto".to_string()];
                let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                    backends: wgpu::Backends::all(),
                    ..Default::default()
                });
                for adapter in instance.enumerate_adapters(wgpu::Backends::all()) {
                    let info = adapter.get_info();
                    // Skip software/fallback adapters (CPU-simulated or unknown)
                    if info.device_type != wgpu::DeviceType::Other
                        && info.device_type != wgpu::DeviceType::Cpu {
                        let name = info.name.clone();
                        if !name.is_empty() && !adapters.contains(&name) {
                            adapters.push(name);
                        }
                    }
                }
                let _ = tx.send(adapters);
            });
        }

        egui::Grid::new("hardware_gpu_grid")
            .num_columns(2)
            .spacing([16.0, 4.0])
            .show(ui, |ui| {
                ui.label(t!("settings.hardware.adapter"));
                if self.gpu_adapters_receiver.is_some() {
                    // Still loading — show a spinner label
                    ui.weak("Detecting GPUs...");
                    ui.ctx().request_repaint();
                } else {
                    egui::ComboBox::from_id_source("gpu_adapter_sel")
                        .selected_text(&settings.preferred_gpu)
                        .show_ui(ui, |ui| {
                            for adapter in &self.gpu_adapters {
                                if ui.selectable_label(settings.preferred_gpu == *adapter, adapter).clicked() {
                                    settings.preferred_gpu = adapter.clone();
                                }
                            }
                        });
                }
                ui.end_row();
            });
        ui.label(egui::RichText::new(t!("settings.hardware.restart_warning")).small().weak());

        // -- Status -----------------------------------------------------
        Self::section_header(ui, &t!("settings.hardware.status"));
        let status_text = t!("settings.hardware.running_on").replace("{0}", &settings.preferred_gpu);
        ui.label(egui::RichText::new(status_text).strong());

        // -- Reset ------------------------------------------------------
        ui.add_space(16.0);
        ui.separator();
        ui.add_space(4.0);
        if ui.button(t!("settings.hardware.reset")).clicked() {
            settings.preferred_gpu = "Auto".to_string();
            settings.gpu_acceleration = true;
            self.gpu_adapters.clear();
            self.gpu_adapters_receiver = None;
        }
        settings.save();
    }

    // -- Interface Tab ----------------------------------------------
    fn show_interface_tab(
        &mut self,
        ui: &mut egui::Ui,
        ctx: &egui::Context,
        settings: &mut AppSettings,
        theme: &mut crate::theme::Theme,
    ) {
        // -- Theme ----------------------------------------------------
        Self::section_header(ui, &t!("settings.interface.theme_mode"));
        egui::Grid::new("interface_theme_grid")
            .num_columns(2)
            .spacing([16.0, 6.0])
            .min_col_width(140.0)
            .show(ui, |ui| {
                ui.label(t!("settings.interface.mode"));
                let current_name = match self.staged_mode {
                    ThemeMode::Light => t!("settings.interface.mode.light"),
                    ThemeMode::Dark  => t!("settings.interface.mode.dark"),
                };
                egui::ComboBox::from_id_source("theme_mode_sel")
                    .selected_text(&current_name)
                    .show_ui(ui, |ui| {
                        if ui.selectable_label(self.staged_mode == ThemeMode::Light, t!("settings.interface.mode.light")).clicked() {
                            self.staged_mode = ThemeMode::Light;
                            self.dirty = true;
                        }
                        if ui.selectable_label(self.staged_mode == ThemeMode::Dark, t!("settings.interface.mode.dark")).clicked() {
                            self.staged_mode = ThemeMode::Dark;
                            self.dirty = true;
                        }
                    });
                ui.end_row();

                ui.label(t!("settings.interface.accent_preset"));
                egui::ComboBox::from_id_source("accent_preset")
                    .selected_text(self.staged_preset.label())
                    .show_ui(ui, |ui| {
                        for &preset in ThemePreset::all() {
                            if ui.selectable_label(self.staged_preset == preset, preset.label()).clicked() {
                                self.staged_preset = preset;
                                self.staged_accent = preset.accent_colors();
                                self.dirty = true;
                            }
                        }
                        if self.staged_preset == ThemePreset::Custom {
                            let _ = ui.selectable_label(true, "Custom");
                        }
                    });
                ui.end_row();
            });

        // -- Accent Colors ---------------------------------------------
        Self::section_header(ui, &t!("settings.interface.accent_color"));
        let mode_colors_label = match self.staged_mode {
            ThemeMode::Light => t!("settings.interface.light_mode_colors"),
            ThemeMode::Dark  => t!("settings.interface.dark_mode_colors"),
        };
        ui.label(egui::RichText::new(mode_colors_label).weak());
        ui.add_space(4.0);

        // Show the 3 color slots for the staged mode
        let changed = match self.staged_mode {
            ThemeMode::Light => {
                let mut c = false;
                c |= Self::color_row(ui, &t!("settings.interface.accent_normal"), &mut self.staged_accent.light_normal);
                c |= Self::color_row(ui, &t!("settings.interface.accent_faint"),  &mut self.staged_accent.light_faint);
                c |= Self::color_row(ui, &t!("settings.interface.accent_strong"), &mut self.staged_accent.light_strong);
                c
            }
            ThemeMode::Dark => {
                let mut c = false;
                c |= Self::color_row(ui, &t!("settings.interface.accent_normal"), &mut self.staged_accent.dark_normal);
                c |= Self::color_row(ui, &t!("settings.interface.accent_faint"),  &mut self.staged_accent.dark_faint);
                c |= Self::color_row(ui, &t!("settings.interface.accent_strong"), &mut self.staged_accent.dark_strong);
                c
            }
        };
        if changed {
            self.staged_preset = ThemePreset::Custom;
            self.dirty = true;
        }

        // -- Canvas Rendering -----------------------------------------
        Self::section_header(ui, &t!("settings.interface.canvas_rendering"));
        egui::Grid::new("interface_canvas_grid")
            .num_columns(2)
            .spacing([16.0, 6.0])
            .min_col_width(140.0)
            .show(ui, |ui| {
                ui.label(t!("settings.interface.zoom_filter"));
                egui::ComboBox::from_id_source("zoom_filter_sel")
                    .selected_text(settings.zoom_filter_mode.name())
                    .show_ui(ui, |ui| {
                        for &mode in ZoomFilterMode::all() {
                            if ui.selectable_label(settings.zoom_filter_mode == mode, mode.name()).clicked() {
                                settings.zoom_filter_mode = mode;
                                settings.save();
                            }
                        }
                    });
                ui.end_row();

                ui.label(t!("settings.interface.brightness"));
                if ui.add(egui::Slider::new(&mut settings.checkerboard_brightness, 0.5..=2.0)
                    .step_by(0.1)
                    .text(""))
                    .changed() {
                    settings.save();
                }
                ui.end_row();
            });
        ui.label(egui::RichText::new(t!("settings.interface.zoom_filter_hint")).small().weak());
        ui.label(egui::RichText::new(t!("settings.interface.brightness_hint")).small().weak());

        // -- Apply / Reset -------------------------------------------
        ui.add_space(16.0);
        ui.separator();
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            let apply_btn = ui.add_enabled(
                self.dirty,
                egui::Button::new(format!("  {}  ", t!("settings.interface.apply_changes"))),
            );
            if apply_btn.clicked() {
                self.apply_theme(settings, theme, ctx);
            }
            if ui.button(t!("settings.interface.reset_to_blue")).clicked() {
                self.staged_preset = ThemePreset::Blue;
                self.staged_accent = ThemePreset::Blue.accent_colors();
                self.staged_mode = ThemeMode::Light;
                self.dirty = true;
                self.apply_theme(settings, theme, ctx);
            }
        });
    }

    /// Render a single color picker row, returns true if value changed
    fn color_row(ui: &mut egui::Ui, label: &str, color: &mut egui::Color32) -> bool {
        let mut changed = false;
        ui.horizontal(|ui| {
            ui.label(format!("{}:", label));
            // Convert to [f32; 4] for color_edit_button
            let mut rgba = [
                color.r() as f32 / 255.0,
                color.g() as f32 / 255.0,
                color.b() as f32 / 255.0,
                color.a() as f32 / 255.0,
            ];
            if ui.color_edit_button_rgba_unmultiplied(&mut rgba).changed() {
                *color = egui::Color32::from_rgba_unmultiplied(
                    (rgba[0] * 255.0) as u8,
                    (rgba[1] * 255.0) as u8,
                    (rgba[2] * 255.0) as u8,
                    (rgba[3] * 255.0) as u8,
                );
                changed = true;
            }
        });
        changed
    }

    /// Commit staged theme changes to settings and rebuild the theme
    fn apply_theme(
        &mut self,
        settings: &mut AppSettings,
        theme: &mut crate::theme::Theme,
        ctx: &egui::Context,
    ) {
        settings.theme_mode = self.staged_mode;
        settings.theme_preset = self.staged_preset;
        settings.custom_accent = self.staged_accent;

        *theme = theme.with_accent(self.staged_preset, self.staged_accent);
        // If mode changed, rebuild with correct mode
        if settings.theme_mode != theme.mode {
            *theme = match settings.theme_mode {
                ThemeMode::Light => crate::theme::Theme::light_with_accent(self.staged_preset, self.staged_accent),
                ThemeMode::Dark  => crate::theme::Theme::dark_with_accent(self.staged_preset, self.staged_accent),
            };
        }
        theme.apply(ctx);
        self.dirty = false;
        settings.save();
    }

    // -- AI Tab -------------------------------------------------------
    fn show_ai_tab(&mut self, ui: &mut egui::Ui, settings: &mut AppSettings) {
        // -- ONNX Runtime ----------------------------------------------
        Self::section_header(ui, &t!("settings.ai.onnx_runtime"));

        ui.label(t!("settings.ai.library_path"));
        ui.horizontal(|ui| {
            let field_w = (ui.available_width() - 38.0).max(120.0);
            ui.add(egui::TextEdit::singleline(&mut self.staged_onnx_path)
                .desired_width(field_w)
                .hint_text(t!("settings.ai.onnx_path_placeholder")));
            if ui.button("\u{1F4C2}").clicked() {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("Dynamic Library", &["dll", "so", "dylib"])
                    .pick_file()
                {
                    self.staged_onnx_path = path.display().to_string();
                    self.onnx_probe_result = None;
                }
            }
        });
        ui.label(egui::RichText::new(t!("settings.ai.library_hint")).small().weak());

        // Security warning — remind users to only use the official runtime
        ui.add_space(4.0);
        let warning_color = egui::Color32::from_rgb(180, 120, 0);
        ui.horizontal_wrapped(|ui| {
            ui.label(
                egui::RichText::new(t!("settings.ai.security_warning"))
                    .small()
                    .color(warning_color),
            );
        });

        ui.add_space(8.0);
        ui.horizontal(|ui| {
            if ui.button(t!("settings.ai.test")).clicked() {
                if self.staged_onnx_path.is_empty() {
                    self.onnx_probe_result = Some(Err(t!("settings.ai.path_empty")));
                } else {
                    match crate::ops::ai::probe_onnx_runtime(&self.staged_onnx_path) {
                        Ok(version) => {
                            self.onnx_probe_result = Some(Ok(version));
                        }
                        Err(e) => {
                            self.onnx_probe_result = Some(Err(format!("{}", e)));
                        }
                    }
                }
            }
            match &self.onnx_probe_result {
                Some(Ok(version)) => {
                    ui.label(egui::RichText::new(t!("settings.ai.onnx_loaded").replace("{0}", version))
                        .color(egui::Color32::from_rgb(0, 180, 0)));
                }
                Some(Err(e)) => {
                    ui.label(egui::RichText::new(format!("\u{274C} {}", e))
                        .color(egui::Color32::from_rgb(220, 0, 0)));
                }
                None => {
                    ui.weak(t!("settings.ai.not_tested"));
                }
            }
        });

        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new(t!("settings.ai.download_from")).small().weak());
            ui.hyperlink_to(
                egui::RichText::new("github.com/microsoft/onnxruntime").small(),
                "https://github.com/microsoft/onnxruntime/releases",
            );
        });

        // -- Segmentation Model ----------------------------------------
        Self::section_header(ui, &t!("settings.ai.segmentation_model"));

        ui.label(t!("settings.ai.model_path"));
        ui.horizontal(|ui| {
            let field_w = (ui.available_width() - 38.0).max(120.0);
            ui.add(egui::TextEdit::singleline(&mut self.staged_model_path)
                .desired_width(field_w)
                .hint_text(t!("settings.ai.model_path_placeholder")));
            if ui.button("\u{1F4C2}").clicked() {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("ONNX Model", &["onnx"])
                    .pick_file()
                {
                    self.staged_model_path = path.display().to_string();
                }
            }
        });
        ui.label(egui::RichText::new(t!("settings.ai.model_hint")).small().weak());

        ui.add_space(12.0);
        ui.label(egui::RichText::new(t!("settings.ai.supported_models")).strong());
        ui.add_space(2.0);
        egui::Grid::new("supported_models_grid")
            .num_columns(3)
            .spacing([16.0, 4.0])
            .show(ui, |ui| {
                ui.label(egui::RichText::new(t!("settings.ai.model_header")).strong().small());
                ui.label(egui::RichText::new(t!("settings.ai.input_size_header")).strong().small());
                ui.label(egui::RichText::new(t!("settings.ai.best_for_header")).strong().small());
                ui.end_row();

                ui.label(egui::RichText::new(t!("settings.ai.birefnet")).small());
                ui.label(egui::RichText::new("1024\u{00D7}1024").small().weak());
                ui.label(egui::RichText::new(t!("settings.ai.birefnet_desc")).small().weak());
                ui.end_row();

                ui.label(egui::RichText::new(t!("settings.ai.u2net")).small());
                ui.label(egui::RichText::new("320\u{00D7}320").small().weak());
                ui.label(egui::RichText::new(t!("settings.ai.u2net_desc")).small().weak());
                ui.end_row();

                ui.label(egui::RichText::new(t!("settings.ai.isnet")).small());
                ui.label(egui::RichText::new("1024\u{00D7}1024").small().weak());
                ui.label(egui::RichText::new(t!("settings.ai.isnet_desc")).small().weak());
                ui.end_row();
            });

        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new(t!("settings.ai.download_models")).small().weak());
            ui.hyperlink_to(
                egui::RichText::new("BiRefNet").small(),
                "https://github.com/ZhengPeng7/BiRefNet/releases",
            );
            ui.label(egui::RichText::new("\u{2022}").small().weak());
            ui.hyperlink_to(
                egui::RichText::new("U\u{00B2}-Net").small(),
                "https://github.com/xuebinqin/U-2-Net",
            );
            ui.label(egui::RichText::new("\u{2022}").small().weak());
            ui.hyperlink_to(
                egui::RichText::new("IS-Net").small(),
                "https://github.com/xuebinqin/DIS",
            );
        });

        // -- Apply / Reset -------------------------------------------
        ui.add_space(16.0);
        ui.separator();
        ui.add_space(4.0);

        ui.horizontal(|ui| {
            if ui.button(t!("common.apply")).clicked() {
                settings.onnx_runtime_path = self.staged_onnx_path.clone();
                settings.birefnet_model_path = self.staged_model_path.clone();
                settings.save();
            }
            if ui.button(t!("common.reset")).clicked() {
                self.staged_onnx_path.clear();
                self.staged_model_path.clear();
                self.onnx_probe_result = None;
                settings.onnx_runtime_path.clear();
                settings.birefnet_model_path.clear();
                settings.save();
            }
        });

        ui.add_space(8.0);
        // Show current status
        if !settings.onnx_runtime_path.is_empty() && !settings.birefnet_model_path.is_empty() {
            ui.label(egui::RichText::new(t!("settings.ai.configured")).small()
                .color(egui::Color32::from_rgb(0, 160, 0)));
        } else {
            ui.label(egui::RichText::new(t!("settings.ai.not_configured")).small().weak());
        }
    }

    // -- Keybinds Tab ---------------------------------------------
    fn show_keybinds_tab(&mut self, ui: &mut egui::Ui, settings: &mut AppSettings) {
        Self::section_header(ui, &t!("settings.keybinds.title"));
        ui.weak(t!("settings.keybinds.hint"));
        ui.add_space(8.0);

        // Detect key press for rebinding
        if let Some(rebind_action) = self.rebinding_action {
            // Check for Escape to cancel
            let esc = ui.input(|i| i.key_pressed(egui::Key::Escape));
            if esc {
                self.rebinding_action = None;
            } else {
                // Check for any key press
                let new_combo = ui.input(|i| {
                    let ctrl = i.modifiers.command;
                    let shift = i.modifiers.shift;
                    let alt = i.modifiers.alt;
                    // Check for text events (for bracket keys, etc.)
                    for ev in &i.events {
                        match ev {
                            egui::Event::Text(t) if !t.is_empty() => {
                                // Only use text events for non-letter chars (brackets etc.)
                                let ch = t.chars().next().unwrap_or(' ');
                                if !ch.is_ascii_alphabetic() && !ch.is_ascii_digit() {
                                    return Some(KeyCombo {
                                        ctrl, shift, alt,
                                        key: None,
                                        text_char: Some(t.clone()),
                                    });
                                }
                            }
                            egui::Event::Key { key, pressed: true, .. } => {
                                // Skip pure modifier keys
                                match key {
                                    egui::Key::Escape => {} // handled above
                                    _ => {
                                        return Some(KeyCombo {
                                            ctrl, shift, alt,
                                            key: Some(*key),
                                            text_char: None,
                                        });
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                    None
                });

                if let Some(combo) = new_combo {
                    self.staged_keybindings.set(rebind_action, combo);
                    self.rebinding_action = None;
                }
            }
        }

        // Keybinds table — the outer ScrollArea (in show()) handles scrolling.
        // Cap height so buttons below remain accessible without scrolling far.
        egui::ScrollArea::vertical()
            .id_source("keybinds_inner_scroll")
            .auto_shrink([false; 2])
            .max_height(880.0)
            .show(ui, |ui| {
            let mut current_category = String::new();
            for action in BindableAction::all() {
                let cat = action.category();
                if cat != current_category {
                    if !current_category.is_empty() {
                        ui.add_space(6.0);
                    }
                    ui.label(egui::RichText::new(&cat).strong().size(13.0));
                    ui.separator();
                    current_category = cat;
                }

                ui.horizontal(|ui| {
                    let name = action.display_name();
                    ui.label(egui::RichText::new(name).size(12.0));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let is_rebinding = self.rebinding_action == Some(*action);
                        let btn_text = if is_rebinding {
                            egui::RichText::new(t!("settings.keybinds.press_key")).italics().color(
                                if ui.visuals().dark_mode {
                                    Color32::from_rgb(100, 200, 255)
                                } else {
                                    Color32::from_rgb(0, 100, 200)
                                }
                            )
                        } else {
                            let combo_text = self.staged_keybindings
                                .get(*action)
                                .map(|c| c.display())
                                .unwrap_or_else(|| "—".to_string());
                            egui::RichText::new(combo_text).monospace()
                        };
                        let btn = ui.add(egui::Button::new(btn_text).min_size(egui::vec2(100.0, 20.0)));
                        if btn.clicked() && !is_rebinding {
                            self.rebinding_action = Some(*action);
                        }
                    });
                });
            }
        });

        ui.add_space(12.0);
        ui.separator();
        ui.add_space(4.0);

        ui.horizontal(|ui| {
            if ui.button(t!("common.apply")).clicked() {
                settings.keybindings = self.staged_keybindings.clone();
                settings.save();
            }
            if ui.button(t!("settings.keybinds.reset_defaults")).clicked() {
                self.staged_keybindings = KeyBindings::default();
                self.rebinding_action = None;
            }
        });
    }
}

/// Common brush size presets
pub const BRUSH_SIZE_PRESETS: &[f32] = &[1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0, 96.0, 128.0];
