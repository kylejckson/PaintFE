use crate::config::icons::Icon;
use crate::config::keybindings::{BindableAction, KeyBindings};
use crate::ops::shapes::{CustomShapeData, ShapeKind};
use eframe::egui;
use egui::{Color32, ColorImage, Sense, TextureHandle, TextureOptions, Vec2};
use std::collections::HashMap;

pub struct BrushTipCategory {
    pub name: String,
    pub tips: Vec<String>,
}

pub struct CustomShapeCategory {
    pub name: String,
    pub shapes: Vec<String>,
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
#[derive(Default)]
pub struct Assets {
    pub(crate) textures: HashMap<Icon, TextureHandle>,
    shape_textures: HashMap<ShapeKind, TextureHandle>,
    custom_shape_data: Vec<CustomShapeData>,
    custom_shape_categories: Vec<CustomShapeCategory>,
    custom_shape_textures: HashMap<String, TextureHandle>,
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

impl Assets {
    pub fn new() -> Self {
        Self::default()
    }

    /// Initialize assets - call once during app startup with the egui context
    pub fn init(&mut self, ctx: &egui::Context) {
        // === Tool icons ===
        self.load_icon(
            ctx,
            Icon::Brush,
            include_bytes!("../../assets/icons/tools/brush.png"),
        );
        self.load_icon(
            ctx,
            Icon::Eraser,
            include_bytes!("../../assets/icons/tools/eraser.png"),
        );
        self.load_icon(
            ctx,
            Icon::Pencil,
            include_bytes!("../../assets/icons/tools/pencil.png"),
        );
        self.load_icon(
            ctx,
            Icon::Line,
            include_bytes!("../../assets/icons/tools/line.png"),
        );
        self.load_icon(
            ctx,
            Icon::Fill,
            include_bytes!("../../assets/icons/tools/fill.png"),
        );
        self.load_icon(
            ctx,
            Icon::RectSelect,
            include_bytes!("../../assets/icons/tools/rect_select.png"),
        );
        self.load_icon(
            ctx,
            Icon::EllipseSelect,
            include_bytes!("../../assets/icons/tools/ellipse_select.png"),
        );
        self.load_icon(
            ctx,
            Icon::Lasso,
            include_bytes!("../../assets/icons/tools/lasso.png"),
        );
        self.load_icon(
            ctx,
            Icon::MagicWand,
            include_bytes!("../../assets/icons/tools/magic_wand.png"),
        );
        self.load_icon(
            ctx,
            Icon::MovePixels,
            include_bytes!("../../assets/icons/tools/move_pixels.png"),
        );
        self.load_icon(
            ctx,
            Icon::MoveSelection,
            include_bytes!("../../assets/icons/tools/move_selection.png"),
        );
        self.load_icon(
            ctx,
            Icon::PerspectiveCrop,
            include_bytes!("../../assets/icons/tools/perspective_crop.png"),
        );
        self.load_icon(
            ctx,
            Icon::ColorPicker,
            include_bytes!("../../assets/icons/tools/color_picker.png"),
        );
        self.load_icon(
            ctx,
            Icon::CloneStamp,
            include_bytes!("../../assets/icons/tools/clone_stamp.png"),
        );
        self.load_icon(
            ctx,
            Icon::Zoom,
            include_bytes!("../../assets/icons/tools/zoom.png"),
        );
        self.load_icon(
            ctx,
            Icon::Pan,
            include_bytes!("../../assets/icons/tools/pan.png"),
        );
        self.load_icon(
            ctx,
            Icon::Gradient,
            include_bytes!("../../assets/icons/tools/gradient.png"),
        );
        self.load_icon(
            ctx,
            Icon::Text,
            include_bytes!("../../assets/icons/tools/text.png"),
        );
        self.load_icon(
            ctx,
            Icon::ContentAwareBrush,
            include_bytes!("../../assets/icons/tools/content_aware_brush.png"),
        );
        self.load_icon(
            ctx,
            Icon::Liquify,
            include_bytes!("../../assets/icons/tools/liquify.png"),
        );
        self.load_icon(
            ctx,
            Icon::MeshWarp,
            include_bytes!("../../assets/icons/tools/mesh_warp.png"),
        );
        self.load_icon(
            ctx,
            Icon::ColorRemover,
            include_bytes!("../../assets/icons/tools/color_remover.png"),
        );
        self.load_icon(
            ctx,
            Icon::Smudge,
            include_bytes!("../../assets/icons/tools/smudge.png"),
        );
        self.load_icon(
            ctx,
            Icon::Shapes,
            include_bytes!("../../assets/icons/tools/shapes.png"),
        );

        // === UI icons ===
        self.load_icon(
            ctx,
            Icon::Undo,
            include_bytes!("../../assets/icons/ui/undo.png"),
        );
        self.load_icon(
            ctx,
            Icon::Redo,
            include_bytes!("../../assets/icons/ui/redo.png"),
        );
        self.load_icon(
            ctx,
            Icon::ZoomIn,
            include_bytes!("../../assets/icons/ui/zoom_in.png"),
        );
        self.load_icon(
            ctx,
            Icon::ZoomOut,
            include_bytes!("../../assets/icons/ui/zoom_out.png"),
        );
        self.load_icon(
            ctx,
            Icon::ResetZoom,
            include_bytes!("../../assets/icons/ui/reset_zoom.png"),
        );
        self.load_icon(
            ctx,
            Icon::Grid,
            include_bytes!("../../assets/icons/ui/grid.png"),
        );
        self.load_icon(
            ctx,
            Icon::GridOn,
            include_bytes!("../../assets/icons/ui/grid_on.png"),
        );
        self.load_icon(
            ctx,
            Icon::GridOff,
            include_bytes!("../../assets/icons/ui/grid_off.png"),
        );
        self.load_icon(
            ctx,
            Icon::GuidesOn,
            include_bytes!("../../assets/icons/ui/guides_on.png"),
        );
        self.load_icon(
            ctx,
            Icon::GuidesOff,
            include_bytes!("../../assets/icons/ui/guides_off.png"),
        );
        self.load_icon(
            ctx,
            Icon::MirrorOff,
            include_bytes!("../../assets/icons/ui/mirror_off.png"),
        );
        self.load_icon(
            ctx,
            Icon::MirrorH,
            include_bytes!("../../assets/icons/ui/mirror_h.png"),
        );
        self.load_icon(
            ctx,
            Icon::MirrorV,
            include_bytes!("../../assets/icons/ui/mirror_v.png"),
        );
        self.load_icon(
            ctx,
            Icon::MirrorQ,
            include_bytes!("../../assets/icons/ui/mirror_q.png"),
        );
        self.load_icon(
            ctx,
            Icon::WrapPreviewOff,
            include_bytes!("../../assets/icons/ui/wrap_preview_off.png"),
        );
        self.load_icon(
            ctx,
            Icon::WrapPreviewOn,
            include_bytes!("../../assets/icons/ui/wrap_preview_on.png"),
        );
        self.load_icon(
            ctx,
            Icon::Settings,
            include_bytes!("../../assets/icons/ui/settings.png"),
        );
        self.load_icon(
            ctx,
            Icon::Layers,
            include_bytes!("../../assets/icons/ui/layers.png"),
        );
        self.load_icon(
            ctx,
            Icon::Visible,
            include_bytes!("../../assets/icons/ui/visible.png"),
        );
        self.load_icon(
            ctx,
            Icon::Hidden,
            include_bytes!("../../assets/icons/ui/hidden.png"),
        );
        self.load_icon(ctx, Icon::New, include_bytes!("../../assets/icons/ui/new.png"));
        self.load_icon(
            ctx,
            Icon::Open,
            include_bytes!("../../assets/icons/ui/open.png"),
        );
        self.load_icon(
            ctx,
            Icon::Save,
            include_bytes!("../../assets/icons/ui/save.png"),
        );
        self.load_icon(
            ctx,
            Icon::Close,
            include_bytes!("../../assets/icons/ui/close.png"),
        );
        self.load_icon(
            ctx,
            Icon::NewLayer,
            include_bytes!("../../assets/icons/ui/new_layer.png"),
        );
        self.load_icon(
            ctx,
            Icon::Delete,
            include_bytes!("../../assets/icons/ui/delete.png"),
        );
        self.load_icon(
            ctx,
            Icon::Duplicate,
            include_bytes!("../../assets/icons/ui/duplicate.png"),
        );
        self.load_icon(
            ctx,
            Icon::Flatten,
            include_bytes!("../../assets/icons/ui/flatten.png"),
        );
        self.load_icon(
            ctx,
            Icon::MergeDown,
            include_bytes!("../../assets/icons/ui/merge_down.png"),
        );
        self.load_icon(
            ctx,
            Icon::MergeDownAsMask,
            include_bytes!("../../assets/icons/ui/merge_down_as_mask.png"),
        );
        self.load_icon(
            ctx,
            Icon::AddLayerMaskRevealAll,
            include_bytes!("../../assets/icons/ui/new_mask.png"),
        );
        self.load_icon(
            ctx,
            Icon::AddLayerMaskFromSelection,
            include_bytes!("../../assets/icons/ui/new_mask.png"),
        );
        self.load_icon(
            ctx,
            Icon::ToggleLayerMask,
            include_bytes!("../../assets/icons/ui/disable_mask.png"),
        );
        self.load_icon(
            ctx,
            Icon::InvertLayerMask,
            include_bytes!("../../assets/icons/ui/invert_mask.png"),
        );
        self.load_icon(
            ctx,
            Icon::ApplyLayerMask,
            include_bytes!("../../assets/icons/ui/apply_mask.png"),
        );
        self.load_icon(
            ctx,
            Icon::DeleteLayerMask,
            include_bytes!("../../assets/icons/ui/delete_mask.png"),
        );
        self.load_icon(
            ctx,
            Icon::Peek,
            include_bytes!("../../assets/icons/ui/peek.png"),
        );
        self.load_icon(
            ctx,
            Icon::SwapColors,
            include_bytes!("../../assets/icons/ui/swap_colors.png"),
        );
        self.load_icon(
            ctx,
            Icon::CopyHex,
            include_bytes!("../../assets/icons/ui/copy_hex.png"),
        );
        self.load_icon(
            ctx,
            Icon::Commit,
            include_bytes!("../../assets/icons/ui/commit.png"),
        );
        self.load_icon(
            ctx,
            Icon::ResetCancel,
            include_bytes!("../../assets/icons/ui/reset_cancel.png"),
        );
        self.load_icon(
            ctx,
            Icon::DropDown,
            include_bytes!("../../assets/icons/ui/drop_down.png"),
        );
        self.load_icon(
            ctx,
            Icon::ShapeOutline,
            include_bytes!("../../assets/icons/ui/shape_outline.png"),
        );
        self.load_icon(
            ctx,
            Icon::ShapeFilled,
            include_bytes!("../../assets/icons/ui/shape_filled.png"),
        );
        self.load_icon(
            ctx,
            Icon::ShapeBoth,
            include_bytes!("../../assets/icons/ui/shape_both.png"),
        );
        self.load_icon(
            ctx,
            Icon::Expand,
            include_bytes!("../../assets/icons/ui/expand.png"),
        );
        self.load_icon(
            ctx,
            Icon::Collapse,
            include_bytes!("../../assets/icons/ui/collapse.png"),
        );
        self.load_icon(
            ctx,
            Icon::Info,
            include_bytes!("../../assets/icons/ui/info.png"),
        );
        self.load_icon(
            ctx,
            Icon::Search,
            include_bytes!("../../assets/icons/ui/search.png"),
        );
        self.load_icon(
            ctx,
            Icon::ClearSearch,
            include_bytes!("../../assets/icons/ui/clear_search.png"),
        );
        self.load_icon(
            ctx,
            Icon::MoveUp,
            include_bytes!("../../assets/icons/ui/move_up.png"),
        );
        self.load_icon(
            ctx,
            Icon::MoveDown,
            include_bytes!("../../assets/icons/ui/move_down.png"),
        );
        self.load_icon(
            ctx,
            Icon::MoveTop,
            include_bytes!("../../assets/icons/ui/move_top.png"),
        );
        self.load_icon(
            ctx,
            Icon::MoveBottom,
            include_bytes!("../../assets/icons/ui/move_bottom.png"),
        );
        self.load_icon(
            ctx,
            Icon::ImportLayer,
            include_bytes!("../../assets/icons/ui/import_layer.png"),
        );
        self.load_icon(
            ctx,
            Icon::Rename,
            include_bytes!("../../assets/icons/ui/rename.png"),
        );
        self.load_icon(
            ctx,
            Icon::LayerFlipH,
            include_bytes!("../../assets/icons/ui/layer_flip_h.png"),
        );
        self.load_icon(
            ctx,
            Icon::LayerFlipV,
            include_bytes!("../../assets/icons/ui/layer_flip_v.png"),
        );
        self.load_icon(
            ctx,
            Icon::LayerRotate,
            include_bytes!("../../assets/icons/ui/layer_rotate.png"),
        );
        self.load_icon(
            ctx,
            Icon::LayerProperties,
            include_bytes!("../../assets/icons/ui/layer_properties.png"),
        );
        self.load_icon(
            ctx,
            Icon::LayerAdd,
            include_bytes!("../../assets/icons/ui/layer_add.png"),
        );
        self.load_icon(
            ctx,
            Icon::LayerDelete,
            include_bytes!("../../assets/icons/ui/layer_delete.png"),
        );
        self.load_icon(
            ctx,
            Icon::LayerDuplicate,
            include_bytes!("../../assets/icons/ui/layer_duplicate.png"),
        );
        self.load_icon(
            ctx,
            Icon::CurrentMarker,
            include_bytes!("../../assets/icons/ui/current_marker.png"),
        );
        self.load_icon(
            ctx,
            Icon::ApplyPrimary,
            include_bytes!("../../assets/icons/ui/apply_primary.png"),
        );
        self.load_icon(
            ctx,
            Icon::SoloLayer,
            include_bytes!("../../assets/icons/ui/solo_layer.png"),
        );
        self.load_icon(
            ctx,
            Icon::HideAll,
            include_bytes!("../../assets/icons/ui/hide_all.png"),
        );
        self.load_icon(
            ctx,
            Icon::ShowAll,
            include_bytes!("../../assets/icons/ui/show_all.png"),
        );

        // === Settings tab icons ===
        self.load_icon(
            ctx,
            Icon::SettingsGeneral,
            include_bytes!("../../assets/icons/ui/settings_general.png"),
        );
        self.load_icon(
            ctx,
            Icon::SettingsInterface,
            include_bytes!("../../assets/icons/ui/settings_interface.png"),
        );
        self.load_icon(
            ctx,
            Icon::SettingsHardware,
            include_bytes!("../../assets/icons/ui/settings_hardware.png"),
        );
        self.load_icon(
            ctx,
            Icon::SettingsKeybinds,
            include_bytes!("../../assets/icons/ui/settings_keybinds.png"),
        );
        self.load_icon(
            ctx,
            Icon::SettingsAI,
            include_bytes!("../../assets/icons/ui/settings_ai.png"),
        );

        // === Menu: File ===
        self.load_icon(
            ctx,
            Icon::MenuFileNew,
            include_bytes!("../../assets/icons/menu/file_new.png"),
        );
        self.load_icon(
            ctx,
            Icon::MenuFileOpen,
            include_bytes!("../../assets/icons/menu/file_open.png"),
        );
        self.load_icon(
            ctx,
            Icon::MenuFileSave,
            include_bytes!("../../assets/icons/menu/file_save.png"),
        );
        self.load_icon(
            ctx,
            Icon::MenuFileSaveAll,
            include_bytes!("../../assets/icons/menu/file_save.png"),
        );
        self.load_icon(
            ctx,
            Icon::MenuFileSaveAs,
            include_bytes!("../../assets/icons/menu/file_save_as.png"),
        );
        self.load_icon(
            ctx,
            Icon::MenuFilePrint,
            include_bytes!("../../assets/icons/menu/file_print.png"),
        );
        self.load_icon(
            ctx,
            Icon::MenuFileQuit,
            include_bytes!("../../assets/icons/menu/file_quit.png"),
        );

        // === Menu: Edit ===
        self.load_icon(
            ctx,
            Icon::MenuEditUndo,
            include_bytes!("../../assets/icons/menu/edit_undo.png"),
        );
        self.load_icon(
            ctx,
            Icon::MenuEditRedo,
            include_bytes!("../../assets/icons/menu/edit_redo.png"),
        );
        self.load_icon(
            ctx,
            Icon::MenuEditCut,
            include_bytes!("../../assets/icons/menu/edit_cut.png"),
        );
        self.load_icon(
            ctx,
            Icon::MenuEditCopy,
            include_bytes!("../../assets/icons/menu/edit_copy.png"),
        );
        self.load_icon(
            ctx,
            Icon::MenuEditPaste,
            include_bytes!("../../assets/icons/menu/edit_paste.png"),
        );
        self.load_icon(
            ctx,
            Icon::MenuEditPasteLayer,
            include_bytes!("../../assets/icons/menu/edit_paste_layer.png"),
        );
        self.load_icon(
            ctx,
            Icon::MenuEditSelectAll,
            include_bytes!("../../assets/icons/menu/edit_select_all.png"),
        );
        self.load_icon(
            ctx,
            Icon::MenuEditDeselect,
            include_bytes!("../../assets/icons/menu/edit_deselect.png"),
        );
        self.load_icon(
            ctx,
            Icon::MenuEditInvertSel,
            include_bytes!("../../assets/icons/menu/edit_invert_sel.png"),
        );
        self.load_icon(
            ctx,
            Icon::MenuEditColorRange,
            include_bytes!("../../assets/icons/menu/edit_color_range.png"),
        );
        self.load_icon(
            ctx,
            Icon::MenuEditPreferences,
            include_bytes!("../../assets/icons/menu/edit_preferences.png"),
        );

        // === Menu: Canvas ===
        self.load_icon(
            ctx,
            Icon::MenuCanvasResize,
            include_bytes!("../../assets/icons/menu/canvas_resize.png"),
        );
        self.load_icon(
            ctx,
            Icon::MenuCanvasCrop,
            include_bytes!("../../assets/icons/menu/canvas_crop.png"),
        );
        self.load_icon(
            ctx,
            Icon::MenuCanvasFlipH,
            include_bytes!("../../assets/icons/menu/canvas_flip_h.png"),
        );
        self.load_icon(
            ctx,
            Icon::MenuCanvasFlipV,
            include_bytes!("../../assets/icons/menu/canvas_flip_v.png"),
        );
        self.load_icon(
            ctx,
            Icon::MenuCanvasRotateCw,
            include_bytes!("../../assets/icons/menu/canvas_rotate_cw.png"),
        );
        self.load_icon(
            ctx,
            Icon::MenuCanvasRotateCcw,
            include_bytes!("../../assets/icons/menu/canvas_rotate_ccw.png"),
        );
        self.load_icon(
            ctx,
            Icon::MenuCanvasRotate180,
            include_bytes!("../../assets/icons/menu/canvas_rotate_180.png"),
        );
        self.load_icon(
            ctx,
            Icon::MenuCanvasAlign,
            include_bytes!("../../assets/icons/menu/canvas_align.png"),
        );
        self.load_icon(
            ctx,
            Icon::MenuCanvasFlatten,
            include_bytes!("../../assets/icons/menu/canvas_flatten.png"),
        );

        // === Menu: Color ===
        self.load_icon(
            ctx,
            Icon::MenuColorAutoLevels,
            include_bytes!("../../assets/icons/menu/color_auto_levels.png"),
        );
        self.load_icon(
            ctx,
            Icon::MenuColorDesaturate,
            include_bytes!("../../assets/icons/menu/color_desaturate.png"),
        );
        self.load_icon(
            ctx,
            Icon::MenuColorInvert,
            include_bytes!("../../assets/icons/menu/color_invert.png"),
        );
        self.load_icon(
            ctx,
            Icon::MenuColorInvertAlpha,
            include_bytes!("../../assets/icons/menu/color_invert_alpha.png"),
        );
        self.load_icon(
            ctx,
            Icon::MenuColorSepia,
            include_bytes!("../../assets/icons/menu/color_sepia.png"),
        );
        self.load_icon(
            ctx,
            Icon::MenuColorBrightness,
            include_bytes!("../../assets/icons/menu/color_brightness.png"),
        );
        self.load_icon(
            ctx,
            Icon::MenuColorCurves,
            include_bytes!("../../assets/icons/menu/color_curves.png"),
        );
        self.load_icon(
            ctx,
            Icon::MenuColorExposure,
            include_bytes!("../../assets/icons/menu/color_exposure.png"),
        );
        self.load_icon(
            ctx,
            Icon::MenuColorHighlights,
            include_bytes!("../../assets/icons/menu/color_highlights.png"),
        );
        self.load_icon(
            ctx,
            Icon::MenuColorHsl,
            include_bytes!("../../assets/icons/menu/color_hsl.png"),
        );
        self.load_icon(
            ctx,
            Icon::MenuColorLevels,
            include_bytes!("../../assets/icons/menu/color_levels.png"),
        );
        self.load_icon(
            ctx,
            Icon::MenuColorTemperature,
            include_bytes!("../../assets/icons/menu/color_temperature.png"),
        );

        // === Menu: Filter ===
        self.load_icon(
            ctx,
            Icon::MenuFilterBlur,
            include_bytes!("../../assets/icons/menu/filter_blur.png"),
        );
        self.load_icon(
            ctx,
            Icon::MenuFilterGaussian,
            include_bytes!("../../assets/icons/menu/filter_gaussian.png"),
        );
        self.load_icon(
            ctx,
            Icon::MenuFilterBokeh,
            include_bytes!("../../assets/icons/menu/filter_bokeh.png"),
        );
        self.load_icon(
            ctx,
            Icon::MenuFilterMotionBlur,
            include_bytes!("../../assets/icons/menu/filter_motion_blur.png"),
        );
        self.load_icon(
            ctx,
            Icon::MenuFilterBoxBlur,
            include_bytes!("../../assets/icons/menu/filter_box_blur.png"),
        );
        self.load_icon(
            ctx,
            Icon::MenuFilterZoomBlur,
            include_bytes!("../../assets/icons/menu/filter_zoom_blur.png"),
        );
        self.load_icon(
            ctx,
            Icon::MenuFilterSharpen,
            include_bytes!("../../assets/icons/menu/filter_sharpen.png"),
        );
        self.load_icon(
            ctx,
            Icon::MenuFilterSharpenItem,
            include_bytes!("../../assets/icons/menu/filter_sharpen_item.png"),
        );
        self.load_icon(
            ctx,
            Icon::MenuFilterReduceNoise,
            include_bytes!("../../assets/icons/menu/filter_reduce_noise.png"),
        );
        self.load_icon(
            ctx,
            Icon::MenuFilterDistort,
            include_bytes!("../../assets/icons/menu/filter_distort.png"),
        );
        self.load_icon(
            ctx,
            Icon::MenuFilterCrystallize,
            include_bytes!("../../assets/icons/menu/filter_crystallize.png"),
        );
        self.load_icon(
            ctx,
            Icon::MenuFilterDents,
            include_bytes!("../../assets/icons/menu/filter_dents.png"),
        );
        self.load_icon(
            ctx,
            Icon::MenuFilterPixelate,
            include_bytes!("../../assets/icons/menu/filter_pixelate.png"),
        );
        self.load_icon(
            ctx,
            Icon::MenuFilterBulge,
            include_bytes!("../../assets/icons/menu/filter_bulge.png"),
        );
        self.load_icon(
            ctx,
            Icon::MenuFilterTwist,
            include_bytes!("../../assets/icons/menu/filter_twist.png"),
        );
        self.load_icon(
            ctx,
            Icon::MenuFilterNoise,
            include_bytes!("../../assets/icons/menu/filter_noise.png"),
        );
        self.load_icon(
            ctx,
            Icon::MenuFilterAddNoise,
            include_bytes!("../../assets/icons/menu/filter_add_noise.png"),
        );
        self.load_icon(
            ctx,
            Icon::MenuFilterMedian,
            include_bytes!("../../assets/icons/menu/filter_median.png"),
        );
        self.load_icon(
            ctx,
            Icon::MenuFilterStylize,
            include_bytes!("../../assets/icons/menu/filter_stylize.png"),
        );
        self.load_icon(
            ctx,
            Icon::MenuFilterGlow,
            include_bytes!("../../assets/icons/menu/filter_glow.png"),
        );
        self.load_icon(
            ctx,
            Icon::MenuFilterVignette,
            include_bytes!("../../assets/icons/menu/filter_vignette.png"),
        );
        self.load_icon(
            ctx,
            Icon::MenuFilterHalftone,
            include_bytes!("../../assets/icons/menu/filter_halftone.png"),
        );
        self.load_icon(
            ctx,
            Icon::MenuFilterInk,
            include_bytes!("../../assets/icons/menu/filter_ink.png"),
        );
        self.load_icon(
            ctx,
            Icon::MenuFilterOilPainting,
            include_bytes!("../../assets/icons/menu/filter_oil_painting.png"),
        );
        self.load_icon(
            ctx,
            Icon::MenuFilterColorFilter,
            include_bytes!("../../assets/icons/menu/filter_color_filter.png"),
        );
        self.load_icon(
            ctx,
            Icon::MenuFilterCanvasBorder,
            include_bytes!("../../assets/icons/menu/filter_canvas_border.png"),
        );
        self.load_icon(
            ctx,
            Icon::MenuFilterGlitch,
            include_bytes!("../../assets/icons/menu/filter_glitch.png"),
        );
        self.load_icon(
            ctx,
            Icon::MenuFilterPixelDrag,
            include_bytes!("../../assets/icons/menu/filter_pixel_drag.png"),
        );
        self.load_icon(
            ctx,
            Icon::MenuFilterRgbDisplace,
            include_bytes!("../../assets/icons/menu/filter_rgb_displace.png"),
        );
        self.load_icon(
            ctx,
            Icon::MenuFilterRemoveBg,
            include_bytes!("../../assets/icons/menu/filter_remove_bg.png"),
        );

        // === Menu: Generate ===
        self.load_icon(
            ctx,
            Icon::MenuGenerateGrid,
            include_bytes!("../../assets/icons/menu/generate_grid.png"),
        );
        self.load_icon(
            ctx,
            Icon::MenuGenerateShadow,
            include_bytes!("../../assets/icons/menu/generate_shadow.png"),
        );
        self.load_icon(
            ctx,
            Icon::MenuGenerateOutline,
            include_bytes!("../../assets/icons/menu/generate_outline.png"),
        );
        self.load_icon(
            ctx,
            Icon::MenuGenerateContours,
            include_bytes!("../../assets/icons/menu/generate_contours.png"),
        );

        // === Menu: View ===
        self.load_icon(
            ctx,
            Icon::MenuViewZoomIn,
            include_bytes!("../../assets/icons/menu/view_zoom_in.png"),
        );
        self.load_icon(
            ctx,
            Icon::MenuViewZoomOut,
            include_bytes!("../../assets/icons/menu/view_zoom_out.png"),
        );
        self.load_icon(
            ctx,
            Icon::MenuViewFitWindow,
            include_bytes!("../../assets/icons/menu/view_fit_window.png"),
        );
        self.load_icon(
            ctx,
            Icon::MenuViewThemeLight,
            include_bytes!("../../assets/icons/menu/view_theme_light.png"),
        );
        self.load_icon(
            ctx,
            Icon::MenuViewThemeDark,
            include_bytes!("../../assets/icons/menu/view_theme_dark.png"),
        );
        // UI: Context bar
        self.load_icon(
            ctx,
            Icon::UiBrushDynamics,
            include_bytes!("../../assets/icons/ui/dynamics.png"),
        );

        self.icons_loaded = true;

        // Load shape picker icons (from new location)
        self.load_shape_icon(
            ctx,
            ShapeKind::Ellipse,
            include_bytes!("../../assets/icons/shapes/ellipse.png"),
        );
        self.load_shape_icon(
            ctx,
            ShapeKind::Rectangle,
            include_bytes!("../../assets/icons/shapes/rectangle.png"),
        );
        self.load_shape_icon(
            ctx,
            ShapeKind::RoundedRect,
            include_bytes!("../../assets/icons/shapes/rounded_rect.png"),
        );
        self.load_shape_icon(
            ctx,
            ShapeKind::Trapezoid,
            include_bytes!("../../assets/icons/shapes/trapezoid.png"),
        );
        self.load_shape_icon(
            ctx,
            ShapeKind::Parallelogram,
            include_bytes!("../../assets/icons/shapes/parallelogram.png"),
        );
        self.load_shape_icon(
            ctx,
            ShapeKind::Triangle,
            include_bytes!("../../assets/icons/shapes/triangle.png"),
        );
        self.load_shape_icon(
            ctx,
            ShapeKind::RightTriangle,
            include_bytes!("../../assets/icons/shapes/right_triangle.png"),
        );
        self.load_shape_icon(
            ctx,
            ShapeKind::Pentagon,
            include_bytes!("../../assets/icons/shapes/pentagon.png"),
        );
        self.load_shape_icon(
            ctx,
            ShapeKind::Hexagon,
            include_bytes!("../../assets/icons/shapes/hexagon.png"),
        );
        self.load_shape_icon(
            ctx,
            ShapeKind::Octagon,
            include_bytes!("../../assets/icons/shapes/octagon.png"),
        );
        self.load_shape_icon(
            ctx,
            ShapeKind::Cross,
            include_bytes!("../../assets/icons/shapes/cross.png"),
        );
        self.load_shape_icon(
            ctx,
            ShapeKind::Check,
            include_bytes!("../../assets/icons/shapes/check.png"),
        );
        self.load_shape_icon(
            ctx,
            ShapeKind::Heart,
            include_bytes!("../../assets/icons/shapes/heart.png"),
        );

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
                    TextureOptions::LINEAR,
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
                    TextureOptions::LINEAR,
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

        for shape in &self.custom_shape_data {
            let icon_pixels = crate::ops::shapes::render_custom_shape_icon(shape, 48, dark);
            let color_image = ColorImage::from_rgba_unmultiplied([48, 48], &icon_pixels);
            let texture = ctx.load_texture(
                format!("custom_shape_{}", shape.name),
                color_image,
                TextureOptions::LINEAR,
            );
            self.custom_shape_textures.insert(shape.name.clone(), texture);
        }
    }

    pub fn get_shape_texture(&self, kind: ShapeKind) -> Option<&TextureHandle> {
        self.shape_textures.get(&kind)
    }

    pub fn load_custom_shape(&mut self, ctx: &egui::Context, name: &str, category: &str, svg_path_data: &str) -> Result<(), String> {
        let data = crate::ops::shapes::parse_custom_shape(name, category, svg_path_data)?;
        let icon_pixels = crate::ops::shapes::render_custom_shape_icon(&data, 48, self.icons_inverted);
        let color_image = ColorImage::from_rgba_unmultiplied([48, 48], &icon_pixels);
        let texture = ctx.load_texture(format!("custom_shape_{name}"), color_image, TextureOptions::LINEAR);
        self.custom_shape_data.retain(|s| s.name != name);
        self.custom_shape_textures.insert(name.to_string(), texture);
        self.custom_shape_data.push(data);
        for cat in &mut self.custom_shape_categories {
            cat.shapes.retain(|s| s != name);
        }
        if let Some(cat) = self.custom_shape_categories.iter_mut().find(|c| c.name == category) {
            cat.shapes.push(name.to_string());
        } else {
            self.custom_shape_categories.push(CustomShapeCategory {
                name: category.to_string(),
                shapes: vec![name.to_string()],
            });
        }
        self.custom_shape_categories.retain(|c| !c.shapes.is_empty());
        Ok(())
    }

    pub fn remove_custom_shape(&mut self, name: &str) -> bool {
        let before = self.custom_shape_data.len();
        self.custom_shape_data.retain(|s| s.name != name);
        self.custom_shape_textures.remove(name);
        for cat in &mut self.custom_shape_categories {
            cat.shapes.retain(|s| s != name);
        }
        self.custom_shape_categories.retain(|c| !c.shapes.is_empty());
        self.custom_shape_data.len() != before
    }

    pub fn custom_shape_categories(&self) -> &[CustomShapeCategory] {
        &self.custom_shape_categories
    }

    pub fn get_custom_shape_texture(&self, name: &str) -> Option<&TextureHandle> {
        self.custom_shape_textures.get(name)
    }

    pub fn get_custom_shape_data(&self, name: &str) -> Option<&CustomShapeData> {
        self.custom_shape_data.iter().find(|s| s.name == name)
    }

    /// Load a brush tip from grayscale PNG bytes.
    /// Extracts the luminance/alpha as a single-channel mask, creates an icon texture.
    pub fn load_brush_tip(&mut self, ctx: &egui::Context, name: &str, category: &str, png_data: &[u8]) {
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
                        icon_pixels[dst] = 0;
                        icon_pixels[dst + 1] = 0;
                        icon_pixels[dst + 2] = 0;
                        icon_pixels[dst + 3] = v;
                    }
                }

                // Cache icon pixels for theme inversion
                self.brush_tip_icon_pixels
                    .insert(name.to_string(), icon_pixels.clone());
                self.brush_tip_icon_sizes
                    .insert(name.to_string(), icon_size);

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
                let cat_idx = self
                    .brush_tip_categories
                    .iter()
                    .position(|c| c.name == category);
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

    /// Remove a custom brush tip by name. Returns true if found and removed.
    /// Built-in (embedded) tips cannot be removed — only tips loaded at runtime.
    pub fn remove_brush_tip(&mut self, name: &str) -> bool {
        // Remove from data
        let data_idx = self.brush_tip_data.iter().position(|d| d.name == name);
        if let Some(idx) = data_idx {
            let cat = self.brush_tip_data[idx].category.clone();
            self.brush_tip_data.remove(idx);
            // Remove from category
            if let Some(cat_idx) = self.brush_tip_categories.iter().position(|c| c.name == cat) {
                self.brush_tip_categories[cat_idx].tips.retain(|t| t != name);
                if self.brush_tip_categories[cat_idx].tips.is_empty() {
                    self.brush_tip_categories.remove(cat_idx);
                }
            }
            // Remove textures
            self.brush_tip_textures.remove(name);
            self.brush_tip_icon_pixels.remove(name);
            self.brush_tip_icon_sizes.remove(name);
            true
        } else {
            false
        }
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
    pub fn icon_selectable_sized(
        &self,
        ui: &mut egui::Ui,
        icon: Icon,
        selected: bool,
        size: Vec2,
    ) -> bool {
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
            ui.add_sized(size, egui::Button::selectable(selected, text))
        };
        response.on_hover_text(icon.tooltip()).clicked()
    }

    /// Create a selectable icon (for tool selection) with default size
    pub fn icon_selectable(&self, ui: &mut egui::Ui, icon: Icon, selected: bool) -> bool {
        self.icon_selectable_sized(ui, icon, selected, Vec2::new(32.0, 32.0))
    }

    /// Create a small icon button (for toolbar)
    pub fn small_icon_button(&self, ui: &mut egui::Ui, icon: Icon) -> egui::Response {
        let icon_bg = crate::theme::Theme::icon_button_bg_for(ui);
        let icon_active = crate::theme::Theme::icon_button_active_for(ui);
        let icon_disabled = crate::theme::Theme::icon_button_disabled_for(ui);
        let response = if let Some(texture) = self.textures.get(&icon) {
            let sized_texture = egui::load::SizedTexture::from_handle(texture);
            let img = egui::Image::from_texture(sized_texture).fit_to_exact_size(Vec2::splat(24.0));
            let btn = egui::Button::image(img);
            ui.scope(|ui| {
                ui.visuals_mut().widgets.inactive.bg_fill = icon_bg;
                ui.visuals_mut().widgets.inactive.weak_bg_fill = icon_bg;
                ui.visuals_mut().widgets.active.bg_fill = icon_active;
                ui.visuals_mut().widgets.active.weak_bg_fill = icon_active;
                ui.visuals_mut().widgets.noninteractive.bg_fill = icon_disabled;
                ui.add(btn)
            })
            .inner
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

    /// Create an enabled/disabled icon button.
    /// When disabled, the icon is faded instead of changing the button background.
    pub fn icon_button_enabled(
        &self,
        ui: &mut egui::Ui,
        icon: Icon,
        enabled: bool,
    ) -> egui::Response {
        let icon_bg = crate::theme::Theme::icon_button_bg_for(ui);
        let icon_active = crate::theme::Theme::icon_button_active_for(ui);
        let icon_disabled = crate::theme::Theme::icon_button_disabled_for(ui);
        let response = if let Some(texture) = self.textures.get(&icon) {
            let sized_texture = egui::load::SizedTexture::from_handle(texture);
            let tint = if !enabled {
                // Fade the icon to look disabled instead of changing background
                if ui.visuals().dark_mode {
                    egui::Color32::from_white_alpha(60)
                } else {
                    egui::Color32::from_black_alpha(60)
                }
            } else if ui.visuals().dark_mode {
                egui::Color32::WHITE
            } else {
                egui::Color32::from_gray(0)
            };
            let img = egui::Image::from_texture(sized_texture)
                .fit_to_exact_size(Vec2::splat(24.0))
                .tint(tint);
            let btn = egui::Button::image(img);
            ui.scope(|ui| {
                ui.visuals_mut().widgets.inactive.bg_fill = icon_bg;
                ui.visuals_mut().widgets.inactive.weak_bg_fill = icon_bg;
                ui.visuals_mut().widgets.active.bg_fill = icon_active;
                ui.visuals_mut().widgets.active.weak_bg_fill = icon_active;
                ui.visuals_mut().widgets.noninteractive.bg_fill = icon_disabled;
                if enabled {
                    ui.add(btn)
                } else {
                    ui.add_enabled(false, btn)
                }
            })
            .inner
        } else {
            ui.add_enabled(enabled, egui::Button::new(icon.emoji()))
        };
        response.on_hover_text(icon.tooltip())
    }

    /// Paint an icon image into a specific rect, returning a click-sense response.
    /// Used for inline icons in layer rows (eye, peek).
    pub fn icon_in_rect(
        &self,
        ui: &mut egui::Ui,
        icon: Icon,
        rect: egui::Rect,
        tint: Color32,
    ) -> egui::Response {
        if let Some(texture) = self.textures.get(&icon) {
            let sized_texture = egui::load::SizedTexture::from_handle(texture);
            let img = egui::Image::from_texture(sized_texture)
                .fit_to_exact_size(rect.size())
                .tint(tint);
            ui.put(rect, img.sense(Sense::click()))
        } else {
            ui.put(
                rect,
                egui::Label::new(
                    egui::RichText::new(icon.emoji())
                        .size(rect.height() * 0.7)
                        .color(tint),
                )
                .sense(Sense::click()),
            )
        }
    }

    /// Create a selectable label with an icon image + text.
    /// Used for settings sidebar tabs.
    pub fn icon_selectable_label(
        &self,
        ui: &mut egui::Ui,
        icon: Icon,
        text: &str,
        selected: bool,
    ) -> bool {
        let icon_size = Vec2::splat(16.0);
        if let Some(texture) = self.textures.get(&icon) {
            let sized_texture = egui::load::SizedTexture::from_handle(texture);
            let _img = egui::Image::from_texture(sized_texture).fit_to_exact_size(icon_size);
            let layout = egui::Layout::left_to_right(egui::Align::Center);
            let response = ui.with_layout(layout, |ui| {
                let r = ui.add(egui::Button::selectable(selected, ""));
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
            ui.selectable_label(selected, format!("{} {}", icon.emoji(), text))
                .clicked()
        }
    }

    /// Create a menu item button with a small icon + text label
    pub fn menu_item(&self, ui: &mut egui::Ui, icon: Icon, text: &str) -> egui::Response {
        self.render_menu_item_no_shortcut(ui, icon, text, true)
    }

    /// Create a menu item button with a small icon + text label, with enabled/disabled state
    pub fn menu_item_enabled(
        &self,
        ui: &mut egui::Ui,
        icon: Icon,
        text: &str,
        enabled: bool,
    ) -> egui::Response {
        self.render_menu_item_no_shortcut(ui, icon, text, enabled)
    }

    /// Custom-painted menu item without shortcut — matches the accent-bar style of
    /// `render_menu_item_with_shortcut` for visual consistency.
    fn render_menu_item_no_shortcut(
        &self,
        ui: &mut egui::Ui,
        icon: Icon,
        text: &str,
        enabled: bool,
    ) -> egui::Response {
        let icon_size = 16.0_f32;
        let padding = ui.spacing().button_padding;
        let icon_gap = 4.0_f32;
        let accent_bar_width = 2.0_f32;

        let text_font = egui::TextStyle::Button.resolve(ui.style());

        let label_color = if enabled {
            ui.visuals().text_color()
        } else {
            ui.visuals().weak_text_color()
        };
        let text_galley =
            ui.painter()
                .layout_no_wrap(text.to_string(), text_font.clone(), label_color);

        let desired_width = padding.x + icon_size + icon_gap + text_galley.size().x + padding.x;
        let row_height = icon_size.max(text_galley.size().y) + padding.y * 2.0;

        let sense = if enabled {
            egui::Sense::click()
        } else {
            egui::Sense::hover()
        };
        let (rect, response) = ui.allocate_exact_size(egui::vec2(desired_width, row_height), sense);

        if ui.is_rect_visible(rect) {
            let visuals = ui.style().interact(&response);
            if enabled && (response.hovered() || response.has_focus()) {
                ui.painter().rect_filled(rect, 0.0, visuals.bg_fill);
                // Left accent bar (Signal Grid hover indicator)
                let accent = ui.visuals().selection.stroke.color;
                let bar_rect = egui::Rect::from_min_max(
                    rect.left_top(),
                    egui::pos2(rect.left() + accent_bar_width, rect.bottom()),
                );
                ui.painter().rect_filled(bar_rect, 0.0, accent);
            }

            let text_color = if enabled {
                visuals.text_color()
            } else {
                ui.visuals().weak_text_color()
            };

            // Icon
            let icon_rect = egui::Rect::from_min_size(
                egui::pos2(rect.left() + padding.x, rect.center().y - icon_size / 2.0),
                egui::vec2(icon_size, icon_size),
            );
            if let Some(texture) = self.textures.get(&icon) {
                let tint = if enabled {
                    text_color
                } else {
                    Color32::from_gray(128)
                };
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
            ui.painter()
                .galley(text_pos, text_galley, egui::Color32::TRANSPARENT);
        }

        response
    }

    /// Create a menu item button with a small icon + text label + right-aligned shortcut hint
    pub fn menu_item_shortcut(
        &self,
        ui: &mut egui::Ui,
        icon: Icon,
        text: &str,
        kb: &KeyBindings,
        action: BindableAction,
    ) -> egui::Response {
        let shortcut = kb.get(action).map(|c| c.display()).unwrap_or_default();
        if shortcut.is_empty() {
            return self.menu_item(ui, icon, text);
        }
        self.render_menu_item_with_shortcut(ui, icon, text, &shortcut, true)
    }

    /// Create a menu item button with icon + text + shortcut hint, with enabled/disabled state
    pub fn menu_item_shortcut_enabled(
        &self,
        ui: &mut egui::Ui,
        icon: Icon,
        text: &str,
        enabled: bool,
        kb: &KeyBindings,
        action: BindableAction,
    ) -> egui::Response {
        let shortcut = kb.get(action).map(|c| c.display()).unwrap_or_default();
        if shortcut.is_empty() {
            return self.menu_item_enabled(ui, icon, text, enabled);
        }
        self.render_menu_item_with_shortcut(ui, icon, text, &shortcut, enabled)
    }

    /// Render a menu item with icon + label on left, shortcut text right-aligned
    fn render_menu_item_with_shortcut(
        &self,
        ui: &mut egui::Ui,
        icon: Icon,
        text: &str,
        shortcut: &str,
        enabled: bool,
    ) -> egui::Response {
        let icon_size = 16.0_f32;
        let padding = ui.spacing().button_padding;
        let icon_gap = 4.0_f32;
        let accent_bar_width = 2.0_f32;

        let text_font = egui::TextStyle::Button.resolve(ui.style());
        let shortcut_font = egui::FontId::monospace(text_font.size * 0.85);

        let label_color = if enabled {
            ui.visuals().text_color()
        } else {
            ui.visuals().weak_text_color()
        };
        let shortcut_color = ui.visuals().weak_text_color();
        let text_galley =
            ui.painter()
                .layout_no_wrap(text.to_string(), text_font.clone(), label_color);
        let shortcut_galley = ui.painter().layout_no_wrap(
            shortcut.to_string(),
            shortcut_font.clone(),
            shortcut_color,
        );

        let min_shortcut_gap = 14.0_f32;
        // Keep a stable label column so shortcut hints align to one right column
        // across rows (prevents uneven spacing in menus like File).
        let min_label_column_width = 76.0_f32;
        let label_column_width = text_galley.size().x.max(min_label_column_width);
        let desired_width = padding.x
            + icon_size
            + icon_gap
            + label_column_width
            + min_shortcut_gap
            + shortcut_galley.size().x
            + padding.x;
        let row_height = icon_size
            .max(text_galley.size().y)
            .max(shortcut_galley.size().y)
            + padding.y * 2.0;

        let sense = if enabled {
            egui::Sense::click()
        } else {
            egui::Sense::hover()
        };
        let (rect, response) = ui.allocate_exact_size(egui::vec2(desired_width, row_height), sense);

        if ui.is_rect_visible(rect) {
            // Hover/active background (only for enabled items)
            let visuals = ui.style().interact(&response);
            if enabled && (response.hovered() || response.has_focus()) {
                ui.painter().rect_filled(rect, 0.0, visuals.bg_fill);
                // Left accent bar (Signal Grid hover indicator)
                let accent = ui.visuals().selection.stroke.color;
                let bar_rect = egui::Rect::from_min_max(
                    rect.left_top(),
                    egui::pos2(rect.left() + accent_bar_width, rect.bottom()),
                );
                ui.painter().rect_filled(bar_rect, 0.0, accent);
            }

            let text_color = if enabled {
                visuals.text_color()
            } else {
                ui.visuals().weak_text_color()
            };

            // Icon
            let icon_rect = egui::Rect::from_min_size(
                egui::pos2(rect.left() + padding.x, rect.center().y - icon_size / 2.0),
                egui::vec2(icon_size, icon_size),
            );
            if let Some(texture) = self.textures.get(&icon) {
                let tint = if enabled {
                    text_color
                } else {
                    Color32::from_gray(128)
                };
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
            ui.painter()
                .galley(text_pos, text_galley, egui::Color32::TRANSPARENT);

            // Shortcut text (right-aligned)
            let shortcut_pos = egui::pos2(
                rect.right() - padding.x - shortcut_galley.size().x,
                rect.center().y - shortcut_galley.size().y / 2.0,
            );
            ui.painter()
                .galley(shortcut_pos, shortcut_galley, egui::Color32::TRANSPARENT);
        }

        response
    }

    /// Create a menu item with icon + text label, shortcut shown below in smaller text
    pub fn menu_item_shortcut_below(
        &self,
        ui: &mut egui::Ui,
        icon: Icon,
        text: &str,
        kb: &KeyBindings,
        action: BindableAction,
    ) -> egui::Response {
        let shortcut = kb.get(action).map(|c| c.display()).unwrap_or_default();
        if shortcut.is_empty() {
            return self.menu_item(ui, icon, text);
        }
        self.render_menu_item_with_shortcut_below(ui, icon, text, &shortcut, true)
    }

    /// Create a menu item with icon + text label + shortcut below, with enabled/disabled state
    pub fn menu_item_shortcut_below_enabled(
        &self,
        ui: &mut egui::Ui,
        icon: Icon,
        text: &str,
        enabled: bool,
        kb: &KeyBindings,
        action: BindableAction,
    ) -> egui::Response {
        let shortcut = kb.get(action).map(|c| c.display()).unwrap_or_default();
        if shortcut.is_empty() {
            return self.menu_item_enabled(ui, icon, text, enabled);
        }
        self.render_menu_item_with_shortcut_below(ui, icon, text, &shortcut, enabled)
    }

    /// Render a menu item with icon + label, shortcut text below label in smaller font
    fn render_menu_item_with_shortcut_below(
        &self,
        ui: &mut egui::Ui,
        icon: Icon,
        text: &str,
        shortcut: &str,
        enabled: bool,
    ) -> egui::Response {
        let icon_size = 16.0_f32;
        let padding = ui.spacing().button_padding;
        let icon_gap = 4.0_f32;
        let accent_bar_width = 2.0_f32;

        let text_font = egui::TextStyle::Button.resolve(ui.style());
        let shortcut_font = egui::FontId::monospace(text_font.size * 0.78);

        let label_color = if enabled {
            ui.visuals().text_color()
        } else {
            ui.visuals().weak_text_color()
        };
        let shortcut_color = ui.visuals().weak_text_color();
        let text_galley =
            ui.painter()
                .layout_no_wrap(text.to_string(), text_font.clone(), label_color);
        let shortcut_galley = ui.painter().layout_no_wrap(
            shortcut.to_string(),
            shortcut_font.clone(),
            shortcut_color,
        );

        let text_width = text_galley.size().x.max(shortcut_galley.size().x);
        let desired_width = padding.x + icon_size + icon_gap + text_width + padding.x;
        let line_gap = 1.0_f32;
        let row_height = (text_galley.size().y + line_gap + shortcut_galley.size().y)
            .max(icon_size)
            + padding.y * 2.0;

        let sense = if enabled {
            egui::Sense::click()
        } else {
            egui::Sense::hover()
        };
        let (rect, response) = ui.allocate_exact_size(egui::vec2(desired_width, row_height), sense);

        if ui.is_rect_visible(rect) {
            let visuals = ui.style().interact(&response);
            if enabled && (response.hovered() || response.has_focus()) {
                ui.painter().rect_filled(rect, 0.0, visuals.bg_fill);
                // Left accent bar (Signal Grid hover indicator)
                let accent = ui.visuals().selection.stroke.color;
                let bar_rect = egui::Rect::from_min_max(
                    rect.left_top(),
                    egui::pos2(rect.left() + accent_bar_width, rect.bottom()),
                );
                ui.painter().rect_filled(bar_rect, 0.0, accent);
            }

            let text_color = if enabled {
                visuals.text_color()
            } else {
                ui.visuals().weak_text_color()
            };

            // Icon (vertically centered)
            let icon_rect = egui::Rect::from_min_size(
                egui::pos2(rect.left() + padding.x, rect.center().y - icon_size / 2.0),
                egui::vec2(icon_size, icon_size),
            );
            if let Some(texture) = self.textures.get(&icon) {
                let tint = if enabled {
                    text_color
                } else {
                    Color32::from_gray(128)
                };
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
            ui.painter().galley(
                egui::pos2(text_x, text_top),
                text_galley,
                egui::Color32::TRANSPARENT,
            );

            // Shortcut text below
            ui.painter().galley(
                egui::pos2(text_x, text_top + total_text_h - shortcut_galley.size().y),
                shortcut_galley,
                egui::Color32::TRANSPARENT,
            );
        }

        response
    }
}

/// Common brush size presets
pub const BRUSH_SIZE_PRESETS: &[f32] = &[
    1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0, 96.0, 128.0,
];

pub const TEXT_SIZE_PRESETS: &[f32] = &[
    8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 24.0, 28.0, 32.0, 36.0, 48.0, 64.0, 72.0, 96.0,
    128.0, 192.0, 256.0,
];

