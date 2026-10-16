//! Stable undo identities shared by command producers, menus and History.
use crate::assets::Icon;

macro_rules! history_actions {
    ($($action:ident => $icon:ident),+ $(,)?) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub enum HistoryAction { $($action),+ }
        impl HistoryAction {
            pub const ALL: &'static [Self] = &[$(Self::$action),+];
            pub const fn icon(self) -> Icon {
                match self { $(Self::$action => Icon::$icon),+ }
            }
        }
    };
}
history_actions! {
    Brush => Brush,
    Eraser => Eraser,
    Pencil => Pencil,
    Line => Line,
    RectangleSelect => RectSelect,
    EllipseSelect => EllipseSelect,
    Lasso => Lasso,
    MagicWand => MagicWand,
    MovePixels => MovePixels,
    MoveSelection => MoveSelection,
    MovePaste => MovePixels,
    Fill => Fill,
    Gradient => Gradient,
    ContentAwareBrush => ContentAwareBrush,
    Liquify => Liquify,
    MeshWarp => MeshWarp,
    ColorRemover => ColorRemover,
    Smudge => Smudge,
    CloneStamp => CloneStamp,
    Text => Text,
    PerspectiveCrop => PerspectiveCrop,
    Shapes => Shapes,
    MaskPaint => Brush,
    MaskErase => Eraser,
    MaskLine => Line,
    Paste => MenuEditPaste,
    PasteLayer => MenuEditPasteLayer,
    Cut => MenuEditCut,
    SelectAll => MenuEditSelectAll,
    Deselect => MenuEditDeselect,
    InvertSelection => MenuEditInvertSel,
    ColorRange => MenuEditColorRange,
    SelectContents => Layers,
    DeleteSelection => Delete,
    FillSelection => Fill,
    AutoLevels => MenuColorAutoLevels,
    Desaturate => MenuColorDesaturate,
    InvertColors => MenuColorInvert,
    InvertAlpha => MenuColorInvertAlpha,
    Sepia => MenuColorSepia,
    BrightnessContrast => MenuColorBrightness,
    HueSaturation => MenuColorHsl,
    Exposure => MenuColorExposure,
    HighlightsShadows => MenuColorHighlights,
    Levels => MenuColorLevels,
    Curves => MenuColorCurves,
    TemperatureTint => MenuColorTemperature,
    Threshold => MenuColorInvert,
    Posterize => MenuColorDesaturate,
    ColorBalance => MenuColorBrightness,
    GradientMap => MenuColorCurves,
    BlackAndWhite => MenuColorDesaturate,
    Vibrance => MenuColorHsl,
    BokehBlur => MenuFilterBokeh,
    MotionBlur => MenuFilterMotionBlur,
    BoxBlur => MenuFilterBoxBlur,
    ZoomBlur => MenuFilterZoomBlur,
    GaussianBlur => MenuFilterGaussian,
    Crystallize => MenuFilterCrystallize,
    Dents => MenuFilterDents,
    Pixelate => MenuFilterPixelate,
    Bulge => MenuFilterBulge,
    Twist => MenuFilterTwist,
    AddNoise => MenuFilterAddNoise,
    ReduceNoise => MenuFilterReduceNoise,
    Median => MenuFilterMedian,
    Glow => MenuFilterGlow,
    Sharpen => MenuFilterSharpenItem,
    Vignette => MenuFilterVignette,
    Halftone => MenuFilterHalftone,
    Ink => MenuFilterInk,
    OilPainting => MenuFilterOilPainting,
    ColorFilter => MenuFilterColorFilter,
    ColorToAlpha => ColorRemover,
    RecoverTransparency => MenuColorRecoverTransparency,
    CanvasBorder => MenuFilterCanvasBorder,
    SeamlessTexture => MenuFilterSeamless,
    PixelDrag => MenuFilterPixelDrag,
    RgbDisplace => MenuFilterRgbDisplace,
    RemoveBackground => MenuFilterRemoveBg,
    Grid => MenuGenerateGrid,
    DropShadow => MenuGenerateShadow,
    Outline => MenuGenerateOutline,
    Contours => MenuGenerateContours,
    ResizeImage => MenuCanvasResize,
    ResizeCanvas => MenuCanvasResize,
    Crop => MenuCanvasCrop,
    Straighten => UiStraighten,
    AlignLayer => MenuCanvasAlign,
    LayerTransform => LayerRotate,
    FlipHorizontal => MenuCanvasFlipH,
    FlipVertical => MenuCanvasFlipV,
    RotateCw => MenuCanvasRotateCw,
    RotateCcw => MenuCanvasRotateCcw,
    Rotate180 => MenuCanvasRotate180,
    Flatten => MenuCanvasFlatten,
    AddLayer => NewLayer,
    AddTextLayer => Rename,
    AddAdjustmentLayer => MenuColorExposure,
    DeleteLayer => Delete,
    DuplicateLayer => Duplicate,
    MoveLayer => MoveUp,
    MoveLayerDown => MoveDown,
    LayerOpacity => LayerProperties,
    HideLayer => Hidden,
    ShowLayer => Visible,
    RenameLayer => Rename,
    ImportLayer => ImportLayer,
    FlipLayerH => LayerFlipH,
    FlipLayerV => LayerFlipV,
    MergeDown => MergeDown,
    MergeMask => MergeDownAsMask,
    AddMask => AddLayerMaskRevealAll,
    MaskFromSelection => AddLayerMaskFromSelection,
    ToggleMask => ToggleLayerMask,
    InvertMask => InvertLayerMask,
    ApplyMask => ApplyLayerMask,
    DeleteMask => DeleteLayerMask,
    RasterizeText => Text,
    AddFolder => Layers,
    MoveFolder => MoveUp,
    MoveFolderDown => MoveDown,
    MoveToFolder => MoveDown,
    RenameFolder => Rename,
    FolderVisibility => Visible,
    HideFolder => Hidden,
    DeleteFolder => Delete,
    FolderColor => ApplyPrimary,
    ExtractRed => MenuColorCurves,
    ExtractGreen => MenuColorHsl,
    ExtractBlue => MenuFilterColorFilter,
    ExtractAlpha => MenuColorInvertAlpha,
    ExtractLuminance => MenuColorLevels,
    ReplaceAlpha => MergeDownAsMask,
    Script => Settings,
    ExternalEffect => SettingsPlugins,
    Custom => Info,
}

/// Titles may contain translated text, user names and parameters; identity may not.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HistoryLabel {
    pub action: HistoryAction,
    pub title: String,
}
impl HistoryAction {
    pub fn label(self, title: impl Into<String>) -> HistoryLabel {
        HistoryLabel {
            action: self,
            title: title.into(),
        }
    }
}
impl Default for HistoryLabel {
    fn default() -> Self {
        HistoryAction::Brush.label("")
    }
}
impl std::fmt::Display for HistoryLabel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.title.fmt(f)
    }
}
