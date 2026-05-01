use crate::assets::{Assets, BRUSH_SIZE_PRESETS, Icon, KeyCombo, TEXT_SIZE_PRESETS};
use crate::canvas::{
    BlendMode, CHUNK_SIZE, CanvasState, SelectionMode, SelectionShape, TiledImage,
};
use crate::components::history::{PixelPatch, SelectionCommand};
use eframe::egui;
use egui::{Color32, Pos2, Rect, Vec2};
use image::{GrayImage, Rgba};
use rayon::prelude::*;
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Instant;

#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum Tool {
    #[default]
    Brush,
    Eraser,
    Pencil,
    Line,
    RectangleSelect,
    EllipseSelect,
    MovePixels,
    MoveSelection,
    MagicWand,
    Fill,
    ColorPicker,
    Gradient,
    ContentAwareBrush,
    Liquify,
    MeshWarp,
    ColorRemover,
    Smudge,
    CloneStamp,
    Text,
    PerspectiveCrop,
    Lasso,
    Zoom,
    Pan,
    Shapes,
}

/// Identifies a brush tip — either the built-in procedural circle or a named image tip
#[derive(Clone, Debug, PartialEq, Eq, Hash, Default)]
pub enum BrushTip {
    /// Default circle — uses existing LUT + hardness system
    #[default]
    Circle,
    /// Image-based tip, identified by name (derived from filename)
    Image(String),
}

impl BrushTip {
    pub fn is_circle(&self) -> bool {
        matches!(self, BrushTip::Circle)
    }
    pub fn display_name(&self) -> &str {
        match self {
            BrushTip::Circle => "Circle",
            BrushTip::Image(name) => name.as_str(),
        }
    }
}

/// Painting mode for the Brush tool.
/// Normal: standard alpha-blend paint
/// Dodge: lightens (increases luminosity)
/// Burn: darkens (decreases luminosity)
/// Sponge: desaturates
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BrushMode {
    Normal,
    Dodge,
    Burn,
    Sponge,
}

impl BrushMode {
    pub fn label(&self) -> &'static str {
        match self {
            BrushMode::Normal => "Normal",
            BrushMode::Dodge => "Dodge",
            BrushMode::Burn => "Burn",
            BrushMode::Sponge => "Sponge",
        }
    }
    pub fn all() -> &'static [BrushMode] {
        &[
            BrushMode::Normal,
            BrushMode::Dodge,
            BrushMode::Burn,
            BrushMode::Sponge,
        ]
    }
}

#[derive(Clone, Debug)]
pub struct ToolProperties {
    pub size: f32,
    pub color: Color32,
    pub hardness: f32,
    /// Pen pressure sensitivity: scale brush size by pen pressure.
    pub pressure_size: bool,
    /// Pen pressure sensitivity: scale brush opacity by pen pressure.
    pub pressure_opacity: bool,
    /// Minimum size multiplier at zero pressure (0.0..1.0). Default 0.1.
    pub pressure_min_size: f32,
    /// Minimum opacity multiplier at zero pressure (0.0..1.0). Default 0.1.
    pub pressure_min_opacity: f32,
    pub blending_mode: BlendMode,
    pub anti_aliased: bool,
    pub brush_tip: BrushTip,
    /// Stamp spacing as fraction of brush diameter (0.01–2.0). Only used for image tips.
    pub spacing: f32,
    /// Fixed rotation angle in degrees (0–360). Only for non-circle tips.
    pub tip_rotation: f32,
    /// When true, each stamp uses a random rotation within `tip_rotation_range`.
    pub tip_random_rotation: bool,
    /// Min/max rotation range in degrees for random mode.
    pub tip_rotation_range: (f32, f32),
    /// Flow rate: 0.0..1.0 — scales final brush opacity per stamp. Default 1.0 (full opacity).
    pub flow: f32,
    /// Scatter: 0.0..1.0 — random positional offset as fraction of brush diameter. Default 0.0.
    pub scatter: f32,
    /// Hue jitter: 0.0..1.0 — random hue shift per stamp (0=none, 1=up to ±180°). Default 0.0.
    pub hue_jitter: f32,
    /// Brightness jitter: 0.0..1.0 — random brightness variation per stamp. Default 0.0.
    pub brightness_jitter: f32,
    /// Painting mode: Normal, Dodge, Burn, or Sponge.
    pub brush_mode: BrushMode,
}

impl Default for ToolProperties {
    fn default() -> Self {
        Self {
            size: 10.0,
            color: Color32::BLACK,
            hardness: 0.75,
            pressure_size: false,
            pressure_opacity: false,
            pressure_min_size: 0.1,
            pressure_min_opacity: 0.1,
            blending_mode: BlendMode::Normal,
            anti_aliased: true,
            brush_tip: BrushTip::Circle,
            spacing: 0.01,
            tip_rotation: 0.0,
            tip_random_rotation: false,
            tip_rotation_range: (0.0, 360.0),
            flow: 1.0,
            scatter: 0.0,
            hue_jitter: 0.0,
            brightness_jitter: 0.0,
            brush_mode: BrushMode::Normal,
        }
    }
}

// Bézier Line Tool specific structures
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LineStage {
    Idle,
    Dragging,
    Editing,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CapStyle {
    Round,
    Flat,
}

impl CapStyle {
    pub fn label(&self) -> &'static str {
        match self {
            CapStyle::Round => "Round",
            CapStyle::Flat => "Flat",
        }
    }

    pub fn all() -> &'static [CapStyle] {
        &[CapStyle::Round, CapStyle::Flat]
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LineEndShape {
    None,
    Arrow,
}

impl LineEndShape {
    pub fn label(&self) -> String {
        match self {
            LineEndShape::None => t!("line_end.none"),
            LineEndShape::Arrow => t!("line_end.arrow"),
        }
    }

    pub fn all() -> &'static [LineEndShape] {
        &[LineEndShape::None, LineEndShape::Arrow]
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ArrowSide {
    End,
    Start,
    Both,
}

impl ArrowSide {
    pub fn label(&self) -> String {
        match self {
            ArrowSide::End => t!("arrow_side.end"),
            ArrowSide::Start => t!("arrow_side.start"),
            ArrowSide::Both => t!("arrow_side.both"),
        }
    }

    pub fn all() -> &'static [ArrowSide] {
        &[ArrowSide::End, ArrowSide::Start, ArrowSide::Both]
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LinePattern {
    Solid,
    Dotted,
    Dashed,
}

impl LinePattern {
    pub fn label(&self) -> String {
        match self {
            LinePattern::Solid => t!("line_pattern.solid"),
            LinePattern::Dotted => t!("line_pattern.dotted"),
            LinePattern::Dashed => t!("line_pattern.dashed"),
        }
    }

    pub fn all() -> &'static [LinePattern] {
        &[LinePattern::Solid, LinePattern::Dotted, LinePattern::Dashed]
    }
}

#[derive(Clone, Debug)]
pub struct LineOptions {
    pub pattern: LinePattern,
    pub cap_style: CapStyle,
    pub end_shape: LineEndShape,
    pub arrow_side: ArrowSide,
    pub anti_alias: bool,
}

impl Default for LineOptions {
    fn default() -> Self {
        Self {
            pattern: LinePattern::Solid,
            cap_style: CapStyle::Round,
            end_shape: LineEndShape::None,
            arrow_side: ArrowSide::End,
            anti_alias: true,
        }
    }
}

#[derive(Clone, Debug)]
pub struct LineToolState {
    pub stage: LineStage,
    pub control_points: [Pos2; 4], // [Start, Control1, Control2, End]
    pub options: LineOptions,
    pub dragging_handle: Option<usize>, // Which handle is being dragged (0-3 = endpoints/controls, 4 = pan)
    pub pan_handle_dragging: bool,      // True while the pan (move-whole-line) handle is active
    pub pan_handle_hovering: bool,      // True when cursor hovers the pan handle (for cursor icon)
    pub pan_drag_canvas_start: Option<Pos2>, // Canvas position where pan drag started
    pub last_bounds: Option<Rect>,      // Track where we drew last frame
    pub require_mouse_release: bool, // Prevent starting new line until mouse is released after commit
    pub initial_mouse_pos: Option<Pos2>, // Track initial position to detect drag
    pub last_size: f32,              // Track size for change detection
    pub last_pattern: LinePattern,   // Track pattern for change detection
    pub last_cap_style: CapStyle,    // Track cap style for change detection
    pub last_end_shape: LineEndShape, // Track end shape for change detection
    pub last_arrow_side: ArrowSide,  // Track arrow side for change detection
    pub last_anti_alias: bool,       // Track AA for change detection
}

impl Default for LineToolState {
    fn default() -> Self {
        Self {
            stage: LineStage::Idle,
            control_points: [Pos2::ZERO; 4],
            options: LineOptions::default(),
            dragging_handle: None,
            pan_handle_dragging: false,
            pan_handle_hovering: false,
            pan_drag_canvas_start: None,
            last_bounds: None,
            require_mouse_release: false,
            initial_mouse_pos: None,
            last_size: 10.0,
            last_pattern: LinePattern::Solid,
            last_cap_style: CapStyle::Round,
            last_end_shape: LineEndShape::None,
            last_arrow_side: ArrowSide::End,
            last_anti_alias: true,
        }
    }
}

pub struct ToolState {
    last_pos: Option<(u32, u32)>,
    last_precise_pos: Option<Pos2>, // Float position for sub-pixel spacing tracking
    distance_remainder: f32,        // How far we've moved since the last stamp
    last_brush_pos: Option<(u32, u32)>, // For Shift+Click straight lines
    using_secondary_color: bool,
    /// EMA-smoothed brush position for rounding off angular corners
    /// during fast mouse movement.
    smooth_pos: Option<Pos2>,
    /// Current pen pressure (0.0..1.0). Defaults to 1.0 (no pen / full pressure).
    pub current_pressure: f32,
    brush_resize_drag_origin: Option<Pos2>,
    brush_resize_drag_start_size: f32,
    brush_resize_drag_active: bool,
}

impl Default for ToolState {
    fn default() -> Self {
        Self {
            last_pos: None,
            last_precise_pos: None,
            distance_remainder: 0.0,
            last_brush_pos: None,
            using_secondary_color: false,
            smooth_pos: None,
            current_pressure: 1.0,
            brush_resize_drag_origin: None,
            brush_resize_drag_start_size: 10.0,
            brush_resize_drag_active: false,
        }
    }
}

/// Tracks stroke state for undo/redo integration
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum StrokeTarget {
    #[default]
    LayerPixels,
    LayerMask,
}

#[derive(Default)]
pub struct StrokeTracker {
    /// Whether a stroke is currently in progress
    pub is_active: bool,
    /// The layer index being modified
    pub layer_index: usize,
    /// Accumulated bounding rect of all modifications in this stroke
    pub bounds: Option<Rect>,
    /// For direct-edit tools (Eraser): Full layer snapshot at stroke start
    pub layer_snapshot: Option<TiledImage>,
    /// For preview-based tools (Brush, Line): We capture before right before commit
    pub uses_preview_layer: bool,
    /// Whether this stroke targets layer pixels or the active layer mask.
    pub target: StrokeTarget,
    /// Description of the stroke (e.g., "Brush Stroke", "Eraser Stroke")
    pub description: String,
    /// Pre-stroke mask snapshot for mask-targeted strokes.
    pub mask_before: Option<TiledImage>,
    pub mask_before_enabled: bool,
}

impl StrokeTracker {
    /// Start tracking a new stroke for a tool that uses preview layer (Brush, Line).
    pub fn start_preview_tool(&mut self, layer_index: usize, description: &str) {
        self.is_active = true;
        self.layer_index = layer_index;
        self.bounds = None;
        self.layer_snapshot = None;
        self.uses_preview_layer = true;
        self.target = StrokeTarget::LayerPixels;
        self.description = description.to_string();
        self.mask_before = None;
        self.mask_before_enabled = true;
    }

    pub fn start_preview_mask_tool(
        &mut self,
        layer_index: usize,
        description: &str,
        before_mask: Option<TiledImage>,
        before_enabled: bool,
    ) {
        self.is_active = true;
        self.layer_index = layer_index;
        self.bounds = None;
        self.layer_snapshot = None;
        self.uses_preview_layer = true;
        self.target = StrokeTarget::LayerMask;
        self.description = description.to_string();
        self.mask_before = before_mask;
        self.mask_before_enabled = before_enabled;
    }

    /// Start tracking a new stroke for a direct-edit tool (Eraser).
    pub fn start_direct_tool(
        &mut self,
        layer_index: usize,
        description: &str,
        layer_pixels: &TiledImage,
    ) {
        self.is_active = true;
        self.layer_index = layer_index;
        self.bounds = None;
        self.layer_snapshot = Some(layer_pixels.clone());
        self.uses_preview_layer = false;
        self.target = StrokeTarget::LayerPixels;
        self.description = description.to_string();
        self.mask_before = None;
        self.mask_before_enabled = true;
    }

    pub fn expand_bounds(&mut self, rect: Rect) {
        self.bounds = Some(match self.bounds {
            Some(existing) => existing.union(rect),
            None => rect,
        });
    }

    /// Get the "before" pixels for the given bounds
    /// For preview tools: extract from current canvas state (unchanged during stroke)
    /// For direct tools: extract from our saved layer snapshot
    pub fn get_before_patch(&self, canvas: &CanvasState, bounds: Rect) -> Option<PixelPatch> {
        // Add generous padding to handle brush radius and anti-aliasing
        let padding = 10.0; // Extra padding beyond the tracked bounds
        let padded_bounds = bounds.expand(padding);

        if self.uses_preview_layer {
            // Active layer hasn't been modified yet - capture directly
            Some(PixelPatch::capture(canvas, self.layer_index, padded_bounds))
        } else {
            // Extract from our saved snapshot
            self.layer_snapshot.as_ref().map(|snapshot| {
                PixelPatch::from_image(
                    snapshot,
                    self.layer_index,
                    padded_bounds,
                    canvas.width,
                    canvas.height,
                )
            })
        }
    }

    pub fn finish(&mut self, canvas: &CanvasState) -> Option<StrokeEvent> {
        if !self.is_active {
            return None;
        }

        let event = self.bounds.map(|bounds| {
            let before_snapshot = self.get_before_patch(canvas, bounds);
            StrokeEvent {
                layer_index: self.layer_index,
                bounds,
                before_snapshot,
                description: self.description.clone(),
                target: self.target,
                before_mask: self.mask_before.clone(),
                before_mask_enabled: self.mask_before_enabled,
            }
        });

        // Reset state
        self.is_active = false;
        self.layer_index = 0;
        self.bounds = None;
        self.layer_snapshot = None;
        self.uses_preview_layer = false;
        self.target = StrokeTarget::LayerPixels;
        self.description.clear();
        self.mask_before = None;
        self.mask_before_enabled = true;

        event
    }

    pub fn cancel(&mut self) {
        self.is_active = false;
        self.layer_index = 0;
        self.bounds = None;
        self.layer_snapshot = None;
        self.uses_preview_layer = false;
        self.target = StrokeTarget::LayerPixels;
        self.description.clear();
        self.mask_before = None;
        self.mask_before_enabled = true;
    }
}

/// Event emitted when a stroke completes
pub struct StrokeEvent {
    pub layer_index: usize,
    pub bounds: Rect,
    pub before_snapshot: Option<PixelPatch>,
    pub description: String,
    pub target: StrokeTarget,
    pub before_mask: Option<TiledImage>,
    pub before_mask_enabled: bool,
}

#[derive(Default)]
pub struct LineState {
    pub line_tool: LineToolState,
}

/// Tracks active selection drag state.
#[derive(Clone, Debug, Default)]
pub struct SelectionToolState {
    pub dragging: bool,
    pub drag_start: Option<Pos2>,
    pub drag_end: Option<Pos2>,
    pub mode: SelectionMode,
    /// True when initiated with right-click (forces Subtract).
    pub right_click_drag: bool,
    /// Effective mode locked at drag-start (may differ from `mode` when modifier keys override).
    pub drag_effective_mode: SelectionMode,
}

#[derive(Clone, Debug)]
struct FlatLayerCache {
    layer_index: usize,
    gpu_generation: u64,
    width: u32,
    height: u32,
    data: Arc<[u8]>,
}

#[derive(Clone, Debug)]
pub(crate) struct ThresholdRegionIndex {
    width: u32,
    height: u32,
    distances: Arc<[u8]>,
    buckets: Arc<Vec<Vec<u32>>>,
    cumulative_bboxes: Arc<Vec<Option<(u32, u32, u32, u32)>>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MagicWandScope {
    Contiguous,
    Global,
}

#[derive(Clone, Debug)]
struct MagicWandOperation {
    start_x: u32,
    start_y: u32,
    target_color: Rgba<u8>,
    combine_mode: SelectionMode,
    scope: MagicWandScope,
    distance_mode: WandDistanceMode,
    connectivity: FloodConnectivity,
    region_index: ThresholdRegionIndex,
}

#[derive(Clone, Debug)]
struct PendingMagicWandOperation {
    request_id: u64,
    start_x: u32,
    start_y: u32,
    target_color: Rgba<u8>,
    combine_mode: SelectionMode,
    scope: MagicWandScope,
    distance_mode: WandDistanceMode,
    connectivity: FloodConnectivity,
}

/// Async distance map result delivered from rayon background thread.
enum MagicWandAsyncResult {
    Ready {
        request_id: u64,
        index: ThresholdRegionIndex,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WandDistanceMode {
    LegacyRgba,
    Perceptual,
}

impl WandDistanceMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::LegacyRgba => "Legacy RGBA",
            Self::Perceptual => "Perceptual",
        }
    }

    pub fn all() -> &'static [Self] {
        &[Self::Perceptual, Self::LegacyRgba]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FloodConnectivity {
    Four,
    Eight,
}

impl FloodConnectivity {
    pub fn label(self) -> &'static str {
        match self {
            Self::Four => "4-neighbor",
            Self::Eight => "8-neighbor",
        }
    }

    pub fn all() -> &'static [Self] {
        &[Self::Four, Self::Eight]
    }
}

#[derive(Clone, Debug)]
pub(crate) struct ActiveFillRegion {
    start_x: u32,
    start_y: u32,
    layer_idx: usize,
    target_color: Rgba<u8>,
    region_index: Option<ThresholdRegionIndex>,
    fill_mask: Vec<u8>,
    fill_bbox: Option<(u32, u32, u32, u32)>,
    last_threshold: Option<u8>,
}

struct FillPreviewResult {
    request_id: u64,
    region_index: Option<ThresholdRegionIndex>,
    start_x: u32,
    start_y: u32,
    target_color: Rgba<u8>,
    threshold: u8,
    fill_mask: Vec<u8>,
    fill_bbox: Option<(u32, u32, u32, u32)>,
    dirty_bbox: Option<(u32, u32, u32, u32)>,
    preview_region: Vec<u8>,
    preview_region_w: u32,
    preview_region_h: u32,
}

#[derive(Clone, Copy, Debug)]
struct FillPreviewSpan {
    y: u32,
    x0: u32,
    x1: u32,
}

impl ThresholdRegionIndex {
    fn from_distances(distances: Vec<u8>, width: u32, height: u32) -> Self {
        let mut buckets = vec![Vec::new(); 256];
        let mut per_distance_bbox: Vec<Option<(u32, u32, u32, u32)>> = vec![None; 256];

        for (idx, &distance) in distances.iter().enumerate() {
            buckets[distance as usize].push(idx as u32);

            let x = (idx as u32) % width;
            let y = (idx as u32) / width;
            let bbox: &mut Option<(u32, u32, u32, u32)> = &mut per_distance_bbox[distance as usize];
            *bbox = Some(match *bbox {
                Some((x0, y0, x1, y1)) => (x0.min(x), y0.min(y), x1.max(x), y1.max(y)),
                None => (x, y, x, y),
            });
        }

        let mut cumulative_bboxes = vec![None; 256];
        let mut cumulative: Option<(u32, u32, u32, u32)> = None;
        for (distance, bbox) in per_distance_bbox.into_iter().enumerate() {
            cumulative = match (cumulative, bbox) {
                (Some((ax0, ay0, ax1, ay1)), Some((bx0, by0, bx1, by1))) => {
                    Some((ax0.min(bx0), ay0.min(by0), ax1.max(bx1), ay1.max(by1)))
                }
                (Some(existing), None) => Some(existing),
                (None, Some(new_bbox)) => Some(new_bbox),
                (None, None) => None,
            };
            cumulative_bboxes[distance] = cumulative;
        }

        Self {
            width,
            height,
            distances: Arc::from(distances.into_boxed_slice()),
            buckets: Arc::new(buckets),
            cumulative_bboxes: Arc::new(cumulative_bboxes),
        }
    }

    fn threshold_bbox(&self, threshold: u8) -> Option<(u32, u32, u32, u32)> {
        self.cumulative_bboxes[threshold as usize]
    }
}

/// State for Magic Wand tool (color-based selection)
#[derive(Debug)]
pub struct MagicWandState {
    pub tolerance: f32,
    pub anti_aliased: bool,
    pub distance_mode: WandDistanceMode,
    pub connectivity: FloodConnectivity,
    pub global_select: bool,
    pub session_before_mask: Option<GrayImage>,
    operations: Vec<MagicWandOperation>,
    pending_operation: Option<PendingMagicWandOperation>,
    /// Tracks the last tolerance + anti-alias that was applied, to skip no-op updates.
    pub last_applied_tolerance: f32,
    pub last_applied_aa: bool,
    /// Channel receiver for async Dijkstra/global distance map computation.
    async_rx: Option<std::sync::mpsc::Receiver<MagicWandAsyncResult>>,
    pub computing: bool,
    preview_pending: bool,
    tolerance_changed_at: Option<Instant>,
    next_request_id: u64,
    /// Cached flat RGBA snapshot of the active layer, keyed by layer generation.
    cached_flat_rgba: Option<FlatLayerCache>,
}

impl Clone for MagicWandState {
    fn clone(&self) -> Self {
        Self {
            tolerance: self.tolerance,
            anti_aliased: self.anti_aliased,
            distance_mode: self.distance_mode,
            connectivity: self.connectivity,
            global_select: self.global_select,
            session_before_mask: self.session_before_mask.clone(),
            operations: self.operations.clone(),
            pending_operation: self.pending_operation.clone(),
            last_applied_tolerance: self.last_applied_tolerance,
            last_applied_aa: self.last_applied_aa,
            async_rx: None,
            computing: false,
            preview_pending: false,
            tolerance_changed_at: self.tolerance_changed_at,
            next_request_id: self.next_request_id,
            cached_flat_rgba: self.cached_flat_rgba.clone(),
        }
    }
}

impl Default for MagicWandState {
    fn default() -> Self {
        Self {
            tolerance: 5.0,
            anti_aliased: true,
            distance_mode: WandDistanceMode::Perceptual,
            connectivity: FloodConnectivity::Four,
            global_select: false,
            session_before_mask: None,
            operations: Vec::new(),
            pending_operation: None,
            last_applied_tolerance: -1.0,
            last_applied_aa: false,
            async_rx: None,
            computing: false,
            preview_pending: false,
            tolerance_changed_at: None,
            next_request_id: 0,
            cached_flat_rgba: None,
        }
    }
}

/// State for Fill tool (flood fill)
pub struct FillToolState {
    /// Tolerance for color matching (0 = exact, 100 = all).
    pub tolerance: f32,
    pub anti_aliased: bool,
    pub distance_mode: WandDistanceMode,
    pub connectivity: FloodConnectivity,
    pub active_fill: Option<ActiveFillRegion>,
    pub last_preview_tolerance: f32,
    pub fill_color_u8: Option<Rgba<u8>>,
    pub use_secondary_color: bool,
    /// When true, fills ALL pixels matching the target color (Shift+click).
    pub global_fill: bool,
    pub tolerance_changed_at: Option<Instant>,
    pub recalc_pending: bool,
    pub last_preview_aa: bool,
    async_rx: Option<std::sync::mpsc::Receiver<FillPreviewResult>>,
    preview_request_id: u64,
    preview_in_flight: bool,
    /// Cached flat RGBA snapshot of the active layer for preview recalculation.
    cached_flat_rgba: Option<FlatLayerCache>,
    gpu_preview_region: Vec<u8>,
    /// Keep the previous preview visible until the next fill preview patch is ready.
    pub defer_preview_clear: bool,
    /// Optional deadline for clearing a committed fill preview overlay.
    pub preview_clear_at: Option<Instant>,
    /// Queued click parameters for rapid fill taps during async preview work.
    pub pending_clicks: VecDeque<FillPendingClick>,
}

/// Parameters for a queued fill click (used when async preview is in flight).
#[derive(Clone, Copy)]
pub struct FillPendingClick {
    pub pos: (u32, u32),
    pub use_secondary: bool,
    pub global_fill: bool,
}

impl Clone for FillToolState {
    fn clone(&self) -> Self {
        Self {
            tolerance: self.tolerance,
            anti_aliased: self.anti_aliased,
            distance_mode: self.distance_mode,
            connectivity: self.connectivity,
            active_fill: self.active_fill.clone(),
            last_preview_tolerance: self.last_preview_tolerance,
            fill_color_u8: self.fill_color_u8,
            use_secondary_color: self.use_secondary_color,
            global_fill: self.global_fill,
            tolerance_changed_at: self.tolerance_changed_at,
            recalc_pending: self.recalc_pending,
            last_preview_aa: self.last_preview_aa,
            async_rx: None,
            preview_request_id: self.preview_request_id,
            preview_in_flight: false,
            cached_flat_rgba: self.cached_flat_rgba.clone(),
            gpu_preview_region: self.gpu_preview_region.clone(),
            defer_preview_clear: self.defer_preview_clear,
            preview_clear_at: self.preview_clear_at,
            pending_clicks: self.pending_clicks.clone(),
        }
    }
}

impl Default for FillToolState {
    fn default() -> Self {
        Self {
            tolerance: 5.0,
            anti_aliased: false,
            distance_mode: WandDistanceMode::LegacyRgba,
            connectivity: FloodConnectivity::Four,
            active_fill: None,
            last_preview_tolerance: 5.0,
            fill_color_u8: None,
            use_secondary_color: false,
            global_fill: false,
            tolerance_changed_at: None,
            recalc_pending: false,
            last_preview_aa: true,
            async_rx: None,
            preview_request_id: 0,
            preview_in_flight: false,
            cached_flat_rgba: None,
            gpu_preview_region: Vec::new(),
            defer_preview_clear: false,
            preview_clear_at: None,
            pending_clicks: VecDeque::new(),
        }
    }
}

// ============================================================================
// GRADIENT TOOL
// ============================================================================

/// Shape of the gradient.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GradientShape {
    Linear,
    LinearReflected,
    Radial,
    Diamond,
}

impl GradientShape {
    pub fn label(&self) -> String {
        match self {
            GradientShape::Linear => t!("gradient_shape.linear"),
            GradientShape::LinearReflected => t!("gradient_shape.linear_reflected"),
            GradientShape::Radial => t!("gradient_shape.radial"),
            GradientShape::Diamond => t!("gradient_shape.diamond"),
        }
    }
    pub fn all() -> &'static [GradientShape] {
        &[
            GradientShape::Linear,
            GradientShape::LinearReflected,
            GradientShape::Radial,
            GradientShape::Diamond,
        ]
    }
}

/// Color vs transparency mode.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GradientMode {
    Color,
    Transparency,
}

impl GradientMode {
    pub fn label(&self) -> String {
        match self {
            GradientMode::Color => t!("gradient_mode.color"),
            GradientMode::Transparency => t!("gradient_mode.transparency"),
        }
    }
}

/// A single color stop in the gradient.
#[derive(Clone, Debug)]
pub struct GradientStop {
    /// Position along the gradient, 0.0 = start, 1.0 = end
    pub position: f32,
    /// RGBA color (un-premultiplied)
    pub color: [u8; 4],
    /// HSV state to avoid roundtrip drift (h, s, v each 0.0..1.0)
    pub hsv: [f32; 3],
}

impl GradientStop {
    pub fn new(position: f32, color: [u8; 4]) -> Self {
        let c32 = Color32::from_rgba_unmultiplied(color[0], color[1], color[2], color[3]);
        let hsv = crate::components::colors::color_to_hsv(c32);
        Self {
            position,
            color,
            hsv,
        }
    }
}

/// Preset gradient configurations.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GradientPreset {
    PrimarySecondary,
    BlackWhite,
    ForegroundTransparent,
    Rainbow,
    Custom,
}

impl GradientPreset {
    pub fn label(&self) -> String {
        match self {
            GradientPreset::PrimarySecondary => t!("gradient_preset.primary_secondary"),
            GradientPreset::BlackWhite => t!("gradient_preset.black_white"),
            GradientPreset::ForegroundTransparent => t!("gradient_preset.foreground_transparent"),
            GradientPreset::Rainbow => t!("gradient_preset.rainbow"),
            GradientPreset::Custom => t!("gradient_preset.custom"),
        }
    }
    pub fn all() -> &'static [GradientPreset] {
        &[
            GradientPreset::PrimarySecondary,
            GradientPreset::BlackWhite,
            GradientPreset::ForegroundTransparent,
            GradientPreset::Rainbow,
        ]
    }
}

/// Tool state for the gradient tool.
#[derive(Clone, Debug)]
pub struct GradientToolState {
    pub shape: GradientShape,
    pub mode: GradientMode,
    pub preset: GradientPreset,
    pub stops: Vec<GradientStop>,
    /// Canvas-coordinate start point
    pub drag_start: Option<Pos2>,
    /// Canvas-coordinate end point
    pub drag_end: Option<Pos2>,
    pub dragging: bool,
    /// Which handle is being dragged (0 = start, 1 = end), None = new gradient drag
    pub dragging_handle: Option<usize>,
    /// Repeat/tile gradient beyond the handle range
    pub repeat: bool,
    /// Which stop is selected for color editing (index)
    pub selected_stop: Option<usize>,
    /// Pre-computed LUT for fast per-pixel lookup (256 entries × 4 channels)
    lut: Vec<u8>,
    lut_dirty: bool,
    pub preview_dirty: bool,
    pub cached_primary: Option<[u8; 4]>,
    /// Reusable buffer for GPU readback (avoids per-frame allocation)
    gpu_readback_buf: Vec<u8>,
    /// Deferred commit — runs one frame after requested (lets the loading bar render first).
    pub commit_pending: bool,
    pub commit_pending_frame: u8,
    /// Layer index where the current gradient preview session started.
    pub source_layer_index: Option<usize>,
}

impl Default for GradientToolState {
    fn default() -> Self {
        let stops = vec![
            GradientStop::new(0.0, [0, 0, 0, 255]),
            GradientStop::new(1.0, [255, 255, 255, 255]),
        ];
        Self {
            shape: GradientShape::Linear,
            mode: GradientMode::Color,
            preset: GradientPreset::PrimarySecondary,
            stops,
            drag_start: None,
            drag_end: None,
            dragging: false,
            dragging_handle: None,
            repeat: false,
            selected_stop: None,
            lut: Vec::new(),
            lut_dirty: true,
            preview_dirty: false,
            cached_primary: None,
            gpu_readback_buf: Vec::new(),
            commit_pending: false,
            commit_pending_frame: 0,
            source_layer_index: None,
        }
    }
}

impl GradientToolState {
    /// Build the 256-entry RGBA lookup table from current stops.
    pub fn rebuild_lut(&mut self) {
        self.lut.resize(256 * 4, 0);
        let stops = &self.stops;
        if stops.is_empty() {
            self.lut.fill(0);
            self.lut_dirty = false;
            return;
        }
        if stops.len() == 1 {
            let c = &stops[0].color;
            for i in 0..256 {
                let off = i * 4;
                self.lut[off] = c[0];
                self.lut[off + 1] = c[1];
                self.lut[off + 2] = c[2];
                self.lut[off + 3] = c[3];
            }
            self.lut_dirty = false;
            return;
        }
        // Sort stops by position
        let mut sorted: Vec<(f32, [u8; 4])> = stops.iter().map(|s| (s.position, s.color)).collect();
        sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        for i in 0..256 {
            let t = i as f32 / 255.0;
            let off = i * 4;
            // Find the two surrounding stops
            if t <= sorted[0].0 {
                let c = &sorted[0].1;
                self.lut[off] = c[0];
                self.lut[off + 1] = c[1];
                self.lut[off + 2] = c[2];
                self.lut[off + 3] = c[3];
            } else if t >= sorted.last().unwrap().0 {
                let c = &sorted.last().unwrap().1;
                self.lut[off] = c[0];
                self.lut[off + 1] = c[1];
                self.lut[off + 2] = c[2];
                self.lut[off + 3] = c[3];
            } else {
                // Linear interpolation between surrounding stops
                let mut left = &sorted[0];
                let mut right = &sorted[sorted.len() - 1];
                for j in 0..sorted.len() - 1 {
                    if sorted[j].0 <= t && sorted[j + 1].0 >= t {
                        left = &sorted[j];
                        right = &sorted[j + 1];
                        break;
                    }
                }
                let span = right.0 - left.0;
                let local_t = if span > 0.0 { (t - left.0) / span } else { 0.0 };
                let inv = 1.0 - local_t;
                self.lut[off] =
                    (left.1[0] as f32 * inv + right.1[0] as f32 * local_t).round() as u8;
                self.lut[off + 1] =
                    (left.1[1] as f32 * inv + right.1[1] as f32 * local_t).round() as u8;
                self.lut[off + 2] =
                    (left.1[2] as f32 * inv + right.1[2] as f32 * local_t).round() as u8;
                self.lut[off + 3] =
                    (left.1[3] as f32 * inv + right.1[3] as f32 * local_t).round() as u8;
            }
        }
        self.lut_dirty = false;
    }

    /// Sample the gradient LUT at position t (0.0..1.0 clamped).
    #[inline(always)]
    pub fn sample_lut(&self, t: f32) -> [u8; 4] {
        let idx = (t.clamp(0.0, 1.0) * 255.0).round() as usize;
        let off = idx * 4;
        [
            self.lut[off],
            self.lut[off + 1],
            self.lut[off + 2],
            self.lut[off + 3],
        ]
    }

    /// Apply stops from a preset, given current primary/secondary colors.
    pub fn apply_preset(&mut self, preset: GradientPreset, primary: [u8; 4], secondary: [u8; 4]) {
        self.preset = preset;
        self.stops = match preset {
            GradientPreset::PrimarySecondary => vec![
                GradientStop::new(0.0, primary),
                GradientStop::new(1.0, secondary),
            ],
            GradientPreset::BlackWhite => vec![
                GradientStop::new(0.0, [0, 0, 0, 255]),
                GradientStop::new(1.0, [255, 255, 255, 255]),
            ],
            GradientPreset::ForegroundTransparent => vec![
                GradientStop::new(0.0, primary),
                GradientStop::new(1.0, [primary[0], primary[1], primary[2], 0]),
            ],
            GradientPreset::Rainbow => vec![
                GradientStop::new(0.0, [255, 0, 0, 255]),
                GradientStop::new(0.17, [255, 165, 0, 255]),
                GradientStop::new(0.33, [255, 255, 0, 255]),
                GradientStop::new(0.5, [0, 200, 0, 255]),
                GradientStop::new(0.67, [0, 100, 255, 255]),
                GradientStop::new(0.83, [75, 0, 130, 255]),
                GradientStop::new(1.0, [148, 0, 211, 255]),
            ],
            GradientPreset::Custom => return, // don't modify stops
        };
        self.lut_dirty = true;
    }

    /// Compute the gradient parameter `t` for a pixel, given start/end and shape.
    #[inline(always)]
    pub fn compute_t(&self, px: f32, py: f32, ax: f32, ay: f32, bx: f32, by: f32) -> f32 {
        let dx = bx - ax;
        let dy = by - ay;
        let len_sq = dx * dx + dy * dy;
        if len_sq < 1e-6 {
            return 0.0;
        }

        match self.shape {
            GradientShape::Linear => {
                let t = ((px - ax) * dx + (py - ay) * dy) / len_sq;
                if self.repeat {
                    t.rem_euclid(1.0)
                } else {
                    t.clamp(0.0, 1.0)
                }
            }
            GradientShape::LinearReflected => {
                let t = ((px - ax) * dx + (py - ay) * dy) / len_sq;

                if self.repeat {
                    let t_mod = t.rem_euclid(2.0);
                    if t_mod > 1.0 { 2.0 - t_mod } else { t_mod }
                } else {
                    1.0 - (2.0 * t.clamp(0.0, 1.0) - 1.0).abs()
                }
            }
            GradientShape::Radial => {
                let inv_len = 1.0 / len_sq.sqrt();
                let dist = ((px - ax) * (px - ax) + (py - ay) * (py - ay)).sqrt() * inv_len;
                if self.repeat {
                    dist.rem_euclid(1.0)
                } else {
                    dist.clamp(0.0, 1.0)
                }
            }
            GradientShape::Diamond => {
                let len = len_sq.sqrt();
                let inv_len = 1.0 / len;
                // Unit vectors along and perpendicular to the gradient direction
                let ux = dx * inv_len;
                let uy = dy * inv_len;
                let rx = px - ax;
                let ry = py - ay;
                let proj = (rx * ux + ry * uy).abs() * inv_len;
                let perp = (rx * (-uy) + ry * ux).abs() * inv_len;
                let dist = proj + perp;
                if self.repeat {
                    dist.rem_euclid(1.0)
                } else {
                    dist.clamp(0.0, 1.0)
                }
            }
        }
    }
}

// ============================================================================
// TEXT TOOL STATE
// ============================================================================

/// Drag interaction type for glyph edit mode.
#[derive(Clone, Debug)]
pub struct GlyphDragState {
    /// Index of the glyph being dragged.
    pub glyph_index: usize,
    /// Type of drag: Move, Scale, or Rotate.
    pub drag_type: GlyphDragType,
    /// Starting mouse position in canvas coords.
    pub start_pos: [f32; 2],
    /// Original override values at drag start.
    pub original_offset: [f32; 2],
    pub original_rotation: f32,
    pub original_scale: f32,
}

/// Type of glyph drag interaction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GlyphDragType {
    Move,
    Scale,
    Rotate,
}

/// Type of text box drag interaction (resize, rotate).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextBoxDragType {
    /// Dragging the left edge (changes max_width and origin X).
    ResizeLeft,
    /// Dragging the right edge (changes max_width).
    ResizeRight,
    /// Dragging top-left corner.
    ResizeTopLeft,
    /// Dragging top-right corner.
    ResizeTopRight,
    /// Dragging bottom-left corner.
    ResizeBottomLeft,
    /// Dragging bottom-right corner.
    ResizeBottomRight,
    /// Dragging the rotation handle.
    Rotate,
}

/// Saved raster text tool style, preserved across text-layer editing sessions.
#[derive(Debug, Clone)]
struct SavedRasterStyle {
    font_family: String,
    font_size: f32,
    font_weight: u16,
    bold: bool,
    italic: bool,
    underline: bool,
    strikethrough: bool,
    letter_spacing: f32,
    width_scale: f32,
    height_scale: f32,
    line_spacing: f32,
    alignment: crate::ops::text::TextAlignment,
    last_color: [u8; 4],
}

#[derive(Default)]
pub struct FontPreviewTextureCache(pub std::collections::HashMap<String, egui::TextureHandle>);

impl std::fmt::Debug for FontPreviewTextureCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FontPreviewTextureCache")
            .field("len", &self.0.len())
            .finish()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FontPreviewInk(pub [u8; 4]);

impl Default for FontPreviewInk {
    fn default() -> Self {
        Self([0, 0, 0, 255])
    }
}

impl Default for SavedRasterStyle {
    fn default() -> Self {
        Self {
            font_family: crate::ops::text::preferred_default_font_family(),
            font_size: 23.0,
            font_weight: 400,
            bold: false,
            italic: false,
            underline: false,
            strikethrough: false,
            letter_spacing: 0.0,
            width_scale: 1.0,
            height_scale: 1.0,
            line_spacing: 1.0,
            alignment: crate::ops::text::TextAlignment::Left,
            last_color: [0, 0, 0, 255],
        }
    }
}

#[derive(Debug, Clone)]
pub enum TextStyleUpdate {
    FontFamily(String),
    FontWeight(u16),
    FontSize(f32),
    Bold { enabled: bool, base_weight: u16 },
    Italic(bool),
    Underline(bool),
    Strikethrough(bool),
    LetterSpacing(f32),
    WidthScale(f32),
    HeightScale(f32),
}

/// State for the Text tool.
#[derive(Debug)]
pub struct TextToolState {
    pub text: String,
    pub cursor_pos: usize,
    pub origin: Option<[f32; 2]>,
    pub is_editing: bool,
    pub font_family: String,
    pub font_size: f32,
    pub font_weight: u16,
    pub bold: bool,
    pub italic: bool,
    pub underline: bool,
    pub strikethrough: bool,
    pub anti_alias: bool,
    pub alignment: crate::ops::text::TextAlignment,
    /// Extra spacing between characters (px). Per-block, synced with TextStyle.letter_spacing.
    pub letter_spacing: f32,
    /// Horizontal glyph scale multiplier (1.0 = normal). Synced with TextStyle.width_scale.
    pub width_scale: f32,
    /// Vertical glyph scale multiplier (1.0 = normal). Synced with TextStyle.height_scale.
    pub height_scale: f32,
    /// Line height multiplier. Per-block, synced with ParagraphStyle.line_spacing.
    pub line_spacing: f32,
    pub available_fonts: Vec<String>,
    pub font_search: String,
    /// True while the font-family popup is open (used to avoid stealing input focus).
    pub font_popup_open: bool,
    pub loaded_font: Option<ab_glyph::FontArc>,
    pub loaded_font_key: String,
    pub commit_pending: bool,
    pub commit_pending_frame: u8,
    pub preview_dirty: bool,
    pub cursor_blink_timer: f64,
    /// Track last rendered color to detect changes from color widget
    pub last_color: [u8; 4],
    /// Whether the mouse is hovering over the move handle
    pub hovering_handle: bool,
    /// Whether the move handle is being dragged
    pub dragging_handle: bool,
    /// Offset from origin to mouse at drag start
    pub drag_offset: [f32; 2],
    /// Reusable coverage buffer for text rasterization (avoids per-keystroke allocation)
    pub coverage_buf: Vec<f32>,
    /// Cached per-line cursor advance values from last rasterization
    pub cached_line_advances: Vec<Vec<f32>>,
    /// Cached line height from last rasterization
    pub cached_line_height: f32,
    /// Available weight variants for the currently selected font family
    pub available_weights: Vec<(String, u16)>,
    /// Async font list: receiver from background thread
    pub fonts_loading_rx: Option<std::sync::mpsc::Receiver<Vec<String>>>,
    /// Cached rasterized text buffer for fast move (avoid re-rasterization during drag)
    pub cached_raster_buf: Vec<u8>,
    pub cached_raster_w: u32,
    pub cached_raster_h: u32,
    /// The raster offset (pixel position) used when originally cached
    pub cached_raster_off_x: i32,
    pub cached_raster_off_y: i32,
    /// The origin offset relative to raster origin used when caching
    pub cached_raster_origin: Option<[f32; 2]>,
    /// Whether the cached raster is valid for the current text/font/style
    pub cached_raster_key: String,
    /// Per-glyph pixel cache to avoid re-outlining unchanged glyphs (Opt 4)
    pub glyph_cache: crate::ops::text::GlyphPixelCache,
    /// Async font preview cache: maps font family name -> loaded FontArc (or None if load failed)
    pub font_preview_cache: std::collections::HashMap<String, Option<ab_glyph::FontArc>>,
    /// Cached egui textures for rasterized font previews shown in the dropdown.
    pub font_preview_textures: FontPreviewTextureCache,
    /// Last theme-aware ink color used to build preview textures.
    pub font_preview_ink: FontPreviewInk,
    /// Channel receiver for async font preview loading results
    pub font_preview_rx:
        Option<std::sync::mpsc::Receiver<Vec<(String, Option<ab_glyph::FontArc>)>>>,
    /// Font names currently queued for async preview loading
    pub font_preview_pending: std::collections::HashSet<String>,
    /// Whether we are editing a text layer's TextLayerData (vs stamping on a raster layer)
    pub editing_text_layer: bool,
    /// Layer index that owns the active text edit session.
    pub editing_layer_index: Option<usize>,
    /// Selection within the active block (Batch 3: per-run formatting)
    pub selection: crate::ops::text_layer::TextSelection,
    /// ID of the currently active text block (Batch 4: multi-block)
    pub active_block_id: Option<u64>,
    /// Set by context bar when style properties change; consumed by handle_input
    /// to apply to selection in text layer mode.
    pub ctx_bar_style_dirty: bool,
    pub pending_ctx_style_update: Option<TextStyleUpdate>,
    /// Cached text effects for the current text layer (synced with TextLayerData).
    pub text_effects: crate::ops::text_layer::TextEffects,
    /// Whether text effects have been modified by the UI and need to be written back.
    pub text_effects_dirty: bool,
    /// Cached text warp for the current text layer block (synced with TextBlock).
    pub text_warp: crate::ops::text_layer::TextWarp,
    /// Whether text warp has been modified by the UI and needs to be written back.
    pub text_warp_dirty: bool,
    // --- Text Box Resize/Rotate ---
    /// Active text box drag interaction (resize handle or rotation handle).
    pub text_box_drag: Option<TextBoxDragType>,
    /// Canvas coordinates of mouse at drag start.
    pub text_box_drag_start_mouse: [f32; 2],
    /// max_width value at drag start (for resize).
    pub text_box_drag_start_width: Option<f32>,
    /// max_height value at drag start (for vertical resize).
    pub text_box_drag_start_height: Option<f32>,
    /// Block position at drag start (for resize from left side).
    pub text_box_drag_start_origin: [f32; 2],
    /// Block rotation at drag start (for rotation handle).
    pub text_box_drag_start_rotation: f32,
    /// The max_width for the active text block (None = natural width).
    pub active_block_max_width: Option<f32>,
    /// The max_height for the active text block (None = auto-height from content).
    pub active_block_max_height: Option<f32>,
    /// Cached bounding rect height for the active block (auto from text reflow).
    pub active_block_height: f32,
    /// Prevents click-to-place from firing on the same mouse-up as a handle interaction.
    pub text_box_click_guard: bool,
    /// True while the user is dragging to select text within a text layer block.
    pub text_select_dragging: bool,
    /// Whether the cursor is hovering the rotation handle (for cursor icon).
    pub hovering_rotation_handle: bool,
    /// Which resize handle the cursor is hovering (for resize cursor icon).
    pub hovering_resize_handle: Option<TextBoxDragType>,
    // --- Glyph Edit Mode (Phase 5 — Batch 9) ---
    /// Whether glyph edit mode is active (per-glyph select/move/rotate/scale).
    pub glyph_edit_mode: bool,
    /// Indices of the currently selected glyphs.
    pub selected_glyphs: Vec<usize>,
    /// Cached glyph bounding boxes for the active block (recomputed on text change).
    pub cached_glyph_bounds: Vec<crate::ops::text_layer::GlyphBounds>,
    /// Whether glyph bounds need recomputation.
    pub glyph_bounds_dirty: bool,
    /// Glyph drag state: which glyph is being dragged and the drag type.
    pub glyph_drag: Option<GlyphDragState>,
    /// Cached glyph overrides synced from the active block (written back on commit).
    pub glyph_overrides: Vec<crate::ops::text_layer::GlyphOverride>,
    /// Whether glyph overrides have been modified and need to be synced back.
    pub glyph_overrides_dirty: bool,
    // --- Text Layer Undo (Phase 6 — Batch 10) ---
    /// Snapshot of the TextLayerData before editing started (for TextLayerEditCommand).
    /// Captured when `load_text_layer_block` is called, consumed by `commit_text_layer`.
    pub text_layer_before: Option<(usize, crate::ops::text_layer::TextLayerData)>,
    /// Whether the text layer drag is using the fast cached preview path.
    /// When true, layer.pixels shows all blocks EXCEPT the dragging one,
    /// and the dragging block's pixels are in cached_raster_buf.
    pub text_layer_drag_cached: bool,
    /// Style properties saved before entering text-layer editing mode, restored when
    /// starting a new raster text session so text-layer styles don't bleed through.
    saved_raster_style: SavedRasterStyle,
}

impl Default for TextToolState {
    fn default() -> Self {
        Self {
            text: String::new(),
            cursor_pos: 0,
            origin: None,
            is_editing: false,
            font_family: crate::ops::text::preferred_default_font_family(),
            font_size: 23.0,
            font_weight: 400,
            bold: false,
            italic: false,
            underline: false,
            strikethrough: false,
            anti_alias: true,
            alignment: crate::ops::text::TextAlignment::Left,
            letter_spacing: 0.0,
            width_scale: 1.0,
            height_scale: 1.0,
            line_spacing: 1.0,
            available_fonts: Vec::new(),
            font_search: String::new(),
            font_popup_open: false,
            loaded_font: None,
            loaded_font_key: String::new(),
            commit_pending: false,
            commit_pending_frame: 0,
            preview_dirty: false,
            cursor_blink_timer: 0.0,
            last_color: [0, 0, 0, 255],
            hovering_handle: false,
            dragging_handle: false,
            drag_offset: [0.0, 0.0],
            coverage_buf: Vec::new(),
            cached_line_advances: Vec::new(),
            cached_line_height: 0.0,
            available_weights: vec![("Regular".to_string(), 400)],
            fonts_loading_rx: None,
            cached_raster_buf: Vec::new(),
            cached_raster_w: 0,
            cached_raster_h: 0,
            cached_raster_off_x: 0,
            cached_raster_off_y: 0,
            cached_raster_origin: None,
            cached_raster_key: String::new(),
            glyph_cache: std::collections::HashMap::new(),
            font_preview_cache: std::collections::HashMap::new(),
            font_preview_textures: FontPreviewTextureCache::default(),
            font_preview_ink: FontPreviewInk::default(),
            font_preview_rx: None,
            font_preview_pending: std::collections::HashSet::new(),
            editing_text_layer: false,
            editing_layer_index: None,
            selection: crate::ops::text_layer::TextSelection::default(),
            active_block_id: None,
            ctx_bar_style_dirty: false,
            pending_ctx_style_update: None,
            text_effects: crate::ops::text_layer::TextEffects::default(),
            text_effects_dirty: false,
            text_warp: crate::ops::text_layer::TextWarp::None,
            text_warp_dirty: false,
            text_box_drag: None,
            text_box_drag_start_mouse: [0.0; 2],
            text_box_drag_start_width: None,
            text_box_drag_start_height: None,
            text_box_drag_start_origin: [0.0; 2],
            text_box_drag_start_rotation: 0.0,
            active_block_max_width: None,
            active_block_max_height: None,
            active_block_height: 0.0,
            text_box_click_guard: false,
            text_select_dragging: false,
            hovering_rotation_handle: false,
            hovering_resize_handle: None,
            glyph_edit_mode: false,
            selected_glyphs: Vec::new(),
            cached_glyph_bounds: Vec::new(),
            glyph_bounds_dirty: true,
            glyph_drag: None,
            glyph_overrides: Vec::new(),
            glyph_overrides_dirty: false,
            text_layer_before: None,
            text_layer_drag_cached: false,
            saved_raster_style: SavedRasterStyle::default(),
        }
    }
}

// ============================================================================
// LIQUIFY TOOL STATE
// ============================================================================

/// Liquify warp mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LiquifyMode {
    Push,
    Expand,
    Contract,
    TwirlCW,
    TwirlCCW,
}

impl LiquifyMode {
    pub fn label(&self) -> String {
        match self {
            LiquifyMode::Push => t!("liquify_mode.push"),
            LiquifyMode::Expand => t!("liquify_mode.expand"),
            LiquifyMode::Contract => t!("liquify_mode.contract"),
            LiquifyMode::TwirlCW => t!("liquify_mode.twirl_cw"),
            LiquifyMode::TwirlCCW => t!("liquify_mode.twirl_ccw"),
        }
    }
    pub fn all() -> &'static [LiquifyMode] {
        &[
            LiquifyMode::Push,
            LiquifyMode::Expand,
            LiquifyMode::Contract,
            LiquifyMode::TwirlCW,
            LiquifyMode::TwirlCCW,
        ]
    }
}

/// State for the Liquify tool.
#[derive(Clone, Debug)]
pub struct LiquifyToolState {
    pub displacement: Option<crate::ops::transform::DisplacementField>,
    pub is_active: bool,
    pub last_pos: Option<[f32; 2]>,
    pub is_dragging: bool,
    pub strength: f32,
    pub mode: LiquifyMode,
    pub dirty_rect: Option<[i32; 4]>,
    pub source_snapshot: Option<image::RgbaImage>,
    /// Layer index captured with `source_snapshot`.
    pub source_layer_index: Option<usize>,
    pub warp_buffer: Vec<u8>,
    pub commit_pending: bool,
    pub commit_pending_frame: u8,
}

impl Default for LiquifyToolState {
    fn default() -> Self {
        Self {
            displacement: None,
            is_active: false,
            last_pos: None,
            is_dragging: false,
            strength: 0.5,
            mode: LiquifyMode::Push,
            dirty_rect: None,
            source_snapshot: None,
            source_layer_index: None,
            warp_buffer: Vec::new(),
            commit_pending: false,
            commit_pending_frame: 0,
        }
    }
}

// ============================================================================
// MESH WARP TOOL STATE
// ============================================================================

/// State for the Mesh Warp tool.
#[derive(Clone, Debug)]
pub struct MeshWarpToolState {
    pub grid_cols: usize,
    pub grid_rows: usize,
    pub points: Vec<[f32; 2]>,
    pub original_points: Vec<[f32; 2]>,
    pub is_active: bool,
    pub dragging_index: Option<usize>,
    pub hover_index: Option<usize>,
    pub source_snapshot: Option<image::RgbaImage>,
    pub warp_buffer: Vec<u8>,
    pub commit_pending: bool,
    pub commit_pending_frame: u8,
    /// True when grid size changed and needs auto-reinit on next frame.
    pub needs_reinit: bool,
    /// Layer index at the time the source snapshot was taken (for staleness detection).
    pub snapshot_layer_index: usize,
    /// Layer gpu_generation at the time the source snapshot was taken.
    pub snapshot_generation: u64,
}

impl Default for MeshWarpToolState {
    fn default() -> Self {
        Self {
            grid_cols: 4,
            grid_rows: 4,
            points: Vec::new(),
            original_points: Vec::new(),
            is_active: false,
            dragging_index: None,
            hover_index: None,
            source_snapshot: None,
            warp_buffer: Vec::new(),
            commit_pending: false,
            commit_pending_frame: 0,
            needs_reinit: false,
            snapshot_layer_index: 0,
            snapshot_generation: 0,
        }
    }
}

// ============================================================================
// COLOR REMOVER TOOL STATE
// ============================================================================

/// Request for an async color removal operation (consumed by app.rs).
#[derive(Clone, Debug)]
pub struct ColorRemovalRequest {
    pub click_x: u32,
    pub click_y: u32,
    pub tolerance: f32,
    pub smoothness: u32,
    pub contiguous: bool,
    pub layer_idx: usize,
    pub selection_mask: Option<image::GrayImage>,
}

/// State for the Color Remover tool.
#[derive(Clone, Debug)]
pub struct ColorRemoverToolState {
    pub tolerance: f32,
    pub smoothness: u32,
    pub contiguous: bool,
    pub commit_pending: bool,
    pub commit_pending_frame: u8,
}

impl Default for ColorRemoverToolState {
    fn default() -> Self {
        Self {
            tolerance: 5.0,
            smoothness: 3,
            contiguous: true,
            commit_pending: false,
            commit_pending_frame: 0,
        }
    }
}

// ============================================================================
// SMUDGE TOOL STATE
// ============================================================================

/// State for the Smudge / finger-painting tool.
#[derive(Clone, Debug)]
pub struct SmudgeState {
    /// How much of the existing pixel is picked up each stamp (0.0–1.0).
    pub strength: f32,
    /// Accumulated pickup color (RGBA f32 0–255).
    pub pickup_color: [f32; 4],
    /// Whether a stroke is currently in progress.
    pub is_stroking: bool,
}

impl Default for SmudgeState {
    fn default() -> Self {
        Self {
            strength: 0.6,
            pickup_color: [0.0; 4],
            is_stroking: false,
        }
    }
}

// ============================================================================
// SHAPES TOOL STATE
// ============================================================================

/// State for the Shapes tool.
#[derive(Clone, Debug)]
pub struct ShapesToolState {
    pub selected_shape: crate::ops::shapes::ShapeKind,
    pub selected_custom_shape: Option<String>,
    pub selected_custom_shape_data: Option<crate::ops::shapes::CustomShapeRenderData>,
    pub draw_start: Option<[f32; 2]>,
    pub draw_end: Option<[f32; 2]>,
    pub is_drawing: bool,
    pub placed: Option<crate::ops::shapes::PlacedShape>,
    pub fill_mode: crate::ops::shapes::ShapeFillMode,
    pub anti_alias: bool,
    pub corner_radius: f32,
    pub commit_pending: bool,
    pub commit_pending_frame: u8,
    /// Reusable RGBA buffer for rasterized shape — avoids per-frame allocation.
    pub cached_shape_buf: Vec<u8>,
    /// Layer index where the current shape preview session started.
    pub source_layer_index: Option<usize>,
}

impl Default for ShapesToolState {
    fn default() -> Self {
        Self {
            selected_shape: crate::ops::shapes::ShapeKind::Rectangle,
            selected_custom_shape: None,
            selected_custom_shape_data: None,
            draw_start: None,
            draw_end: None,
            is_drawing: false,
            placed: None,
            fill_mode: crate::ops::shapes::ShapeFillMode::Filled,
            anti_alias: true,
            corner_radius: 10.0,
            commit_pending: false,
            commit_pending_frame: 0,
            cached_shape_buf: Vec::new(),
            source_layer_index: None,
        }
    }
}

pub struct ToolsPanel {
    pub active_tool: Tool,
    pub properties: ToolProperties,
    tool_state: ToolState,
    pub line_state: LineState,
    pub stroke_tracker: StrokeTracker,
    /// Pending stroke event — processed by app.rs for undo/redo.
    pending_stroke_event: Option<StrokeEvent>,
    pub selection_state: SelectionToolState,
    pub magic_wand_state: MagicWandState,
    pub fill_state: FillToolState,
    /// Last color picked this frame plus its target swatch; None if color picker not used.
    pub last_picked_color: Option<(Color32, bool)>,
    pub lasso_state: LassoState,
    pub perspective_crop_state: PerspectiveCropState,
    pub zoom_tool_state: ZoomToolState,
    /// Consumed by Canvas each frame.
    pub zoom_pan_action: ZoomPanAction,
    pub clone_stamp_state: CloneStampState,
    pub content_aware_state: ContentAwareBrushState,
    pub gradient_state: GradientToolState,
    pub text_state: TextToolState,
    pub liquify_state: LiquifyToolState,
    pub mesh_warp_state: MeshWarpToolState,
    pub color_remover_state: ColorRemoverToolState,
    pub smudge_state: SmudgeState,
    /// Pending async color removal request — consumed by app.rs for spawn_filter_job.
    pending_color_removal: Option<ColorRemovalRequest>,
    pub shapes_state: ShapesToolState,
    pub pending_open_add_shape: bool,
    pub pending_delete_shape: Option<String>,
    last_tracked_layer_index: usize,
    last_tracked_layer_count: usize,
    /// B6: Pre-computed brush alpha LUT indexed by squared distance ratio.
    /// `lut[i]` = alpha for `dist_sq / radius_sq == i / 255.0`.
    /// Eliminates per-pixel `sqrt()` + `smoothstep` computation.
    brush_alpha_lut: [u8; 256],
    /// Parameters that were used to build the current LUT (to detect changes).
    lut_params: (f32, f32, bool), // (size, hardness, anti_aliased)
    /// Cached rescaled tip mask for current (tip, size) combo
    pub(crate) brush_tip_mask: Vec<u8>,
    /// Side length of cached mask
    pub(crate) brush_tip_mask_size: u32,
    /// Cache key for brush tip mask: (tip_name, rounded_size, hardness_pct)
    brush_tip_cache_key: (String, u32, u32),
    /// Monotonic stamp counter, used to seed per-stamp random rotation.
    stamp_counter: u32,
    /// The effective rotation angle (degrees) used for cursor overlay display.
    /// Updated each frame: equals `tip_rotation` in fixed mode, 0.0 in random mode.
    pub(crate) active_tip_rotation_deg: f32,
    /// Pending history commands from tool operations (e.g., perspective crop)
    /// Consumed by app.rs each frame.
    pub pending_history_commands: Vec<Box<dyn crate::components::history::Command>>,
    /// When set, the active text layer at this index needs to be rasterized
    /// before a destructive tool operation can proceed. Consumed by app.rs.
    pub pending_auto_rasterize: Option<usize>,
    /// Saved tool before auto-switching to Text for a text layer.
    /// Restored when switching away from a text layer.
    tool_before_text_layer: Option<Tool>,
    /// Pixel amount for selection feather/expand/contract (context bar spinner).
    pub sel_modify_radius: f32,
    /// Pending selection modification set from context bar; consumed by app.rs.
    pub pending_sel_modify: Option<SelectionModifyOp>,
    /// Tool usage hint — displayed at the bottom-left of the app window.
    /// Set each frame based on the active tool.
    pub tool_hint: String,
    active_layer_rgba_prewarm_rx: Option<std::sync::mpsc::Receiver<FlatLayerCache>>,
    active_layer_rgba_prewarm_key: Option<(usize, u64, u32, u32)>,
    pub brush_resize_drag_binding: KeyCombo,
    pub injected_enter_pressed: bool,
    pub injected_escape_pressed: bool,
    /// Set to true when user clicks "New..." in brush tip picker.
    /// Consumed by app.rs to open AddBrushTipDialog.
    pub pending_open_add_brush_tip: bool,
    /// Set to a brush tip name when user right-clicks and selects "Delete".
    /// Consumed by app.rs to remove the tip from assets.
    pub pending_delete_brush_tip: Option<String>,
    /// Right-click context menu state for brush tip picker: (tip_name, screen_pos_x, screen_pos_y)
    pub brush_tip_context_menu: Option<(String, f32, f32)>,
}

impl Default for ToolsPanel {
    fn default() -> Self {
        Self {
            active_tool: Tool::default(),
            properties: ToolProperties::default(),
            tool_state: ToolState::default(),
            line_state: LineState::default(),
            stroke_tracker: StrokeTracker::default(),
            pending_stroke_event: None,
            selection_state: SelectionToolState::default(),
            magic_wand_state: MagicWandState::default(),
            fill_state: FillToolState::default(),
            last_picked_color: None,
            lasso_state: LassoState::default(),
            perspective_crop_state: PerspectiveCropState::default(),
            zoom_tool_state: ZoomToolState::default(),
            zoom_pan_action: ZoomPanAction::default(),
            clone_stamp_state: CloneStampState::default(),
            content_aware_state: ContentAwareBrushState::default(),
            gradient_state: GradientToolState::default(),
            text_state: TextToolState::default(),
            liquify_state: LiquifyToolState::default(),
            mesh_warp_state: MeshWarpToolState::default(),
            color_remover_state: ColorRemoverToolState::default(),
            smudge_state: SmudgeState::default(),
            pending_color_removal: None,
            shapes_state: ShapesToolState::default(),
            pending_open_add_shape: false,
            pending_delete_shape: None,
            last_tracked_layer_index: 0,
            last_tracked_layer_count: 0,
            brush_alpha_lut: [0u8; 256],
            lut_params: (0.0, 0.0, false),
            brush_tip_mask: Vec::new(),
            brush_tip_mask_size: 0,
            brush_tip_cache_key: (String::new(), 0, 0),
            stamp_counter: 0,
            active_tip_rotation_deg: 0.0,
            pending_history_commands: Vec::new(),
            pending_auto_rasterize: None,
            tool_before_text_layer: None,
            sel_modify_radius: 5.0,
            pending_sel_modify: None,
            tool_hint: String::new(),
            active_layer_rgba_prewarm_rx: None,
            active_layer_rgba_prewarm_key: None,
            brush_resize_drag_binding: KeyCombo::modifiers_only(false, true, false),
            injected_enter_pressed: false,
            injected_escape_pressed: false,
            pending_open_add_brush_tip: false,
            pending_delete_brush_tip: None,
            brush_tip_context_menu: None,
        }
    }
}

/// Action returned from tools panel
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum ToolsPanelAction {
    None,
    OpenColors,
    SwapColors,
}

/// Pending selection modification triggered from the context bar.
#[derive(Clone, Copy, Debug)]
pub enum SelectionModifyOp {
    Feather(f32),  // blur radius in pixels
    Expand(i32),   // dilation radius in pixels
    Contract(i32), // erosion radius in pixels
}

/// Action for zoom/pan tools — read by Canvas after handle_input().
#[derive(Clone, Copy, Debug, Default)]
pub enum ZoomPanAction {
    #[default]
    None,
    /// Zoom in, anchored at a canvas-coordinate point.
    ZoomIn {
        canvas_x: f32,
        canvas_y: f32,
    },
    /// Zoom out, anchored at a canvas-coordinate point.
    ZoomOut {
        canvas_x: f32,
        canvas_y: f32,
    },
    /// Zoom to fit a canvas-space rectangle into the viewport.
    ZoomToRect {
        min_x: f32,
        min_y: f32,
        max_x: f32,
        max_y: f32,
    },
    Pan {
        dx: f32,
        dy: f32,
    },
}

/// State for the Lasso (freeform) selection tool.
#[derive(Clone, Debug, Default)]
pub struct LassoState {
    /// Accumulated polygon points in **canvas** pixel coordinates.
    pub points: Vec<Pos2>,
    /// True while dragging to collect points.
    pub dragging: bool,
    /// The selection combination mode.
    pub mode: SelectionMode,
    /// True when drag started with right-click (forces Subtract).
    pub right_click_drag: bool,
    /// Effective mode locked at drag-start.
    pub drag_effective_mode: SelectionMode,
}

/// State for the Perspective Crop tool.
#[derive(Clone, Debug)]
pub struct PerspectiveCropState {
    /// The four corner handles in canvas pixel coordinates
    /// (top-left, top-right, bottom-right, bottom-left).
    pub corners: [Pos2; 4],
    /// Which corner (0-3) is being dragged, if any.
    pub dragging_corner: Option<usize>,
    /// Whether the crop quad has been placed.
    pub active: bool,
    /// Set by change_tool; consumed by handle_input to auto-init with canvas dims.
    pub needs_auto_init: bool,
}

impl Default for PerspectiveCropState {
    fn default() -> Self {
        Self {
            corners: [Pos2::ZERO; 4],
            dragging_corner: None,
            active: false,
            needs_auto_init: false,
        }
    }
}

/// State for the Zoom tool drag-rect.
#[derive(Clone, Debug, Default)]
pub struct ZoomToolState {
    pub drag_start: Option<Pos2>,
    pub drag_end: Option<Pos2>,
    pub dragging: bool,
    /// When true, single tap/click zooms OUT instead of in.
    /// Right-click always does the opposite.  Touch-friendly toggle.
    pub zoom_out_mode: bool,
}

/// State for the Clone Stamp tool.
#[derive(Clone, Debug, Default)]
pub struct CloneStampState {
    /// Source point in canvas coordinates (set via Alt+Click).
    pub source: Option<Pos2>,
    /// Offset from source to current paint position, locked on first stroke.
    pub offset: Option<Vec2>,
    /// Whether the offset has been locked for the current stroke.
    pub offset_locked: bool,
}

/// State for the Content Aware Brush (healing brush).
#[derive(Clone, Debug)]
pub struct ContentAwareBrushState {
    /// Radius of the sampling ring for the Instant mode.
    pub sample_radius: f32,
    /// Fill quality / algorithm selection.
    pub quality: crate::ops::inpaint::ContentAwareQuality,
    /// Patch side-length for PatchMatch (Balanced / HQ).
    pub patch_size: u32,
    /// Collected painted points for the current stroke.
    pub stroke_points: Vec<Pos2>,
    /// Original layer pixels before the stroke started (for Balanced/HQ async job).
    pub stroke_original: Option<image::RgbaImage>,
    /// Accumulated hole mask: every painted pixel in the current stroke.
    pub hole_mask: Option<GrayImage>,
    /// Set on mouse-release for Balanced/HQ; consumed by app.rs.
    pub pending_inpaint: Option<crate::ops::inpaint::InpaintRequest>,
}

impl Default for ContentAwareBrushState {
    fn default() -> Self {
        Self {
            sample_radius: 30.0,
            quality: crate::ops::inpaint::ContentAwareQuality::Instant,
            patch_size: 5,
            stroke_points: Vec::new(),
            stroke_original: None,
            hole_mask: None,
            pending_inpaint: None,
        }
    }
}

