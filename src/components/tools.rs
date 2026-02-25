use crate::assets::{Assets, BRUSH_SIZE_PRESETS, Icon};
use crate::canvas::{
    BlendMode, CHUNK_SIZE, CanvasState, SelectionMode, SelectionShape, TiledImage,
};
use crate::components::history::PixelPatch;
use eframe::egui;
use egui::{Color32, Pos2, Rect, Vec2};
use image::{GrayImage, Luma, Rgba};
use rayon::prelude::*;
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
}

impl Default for LineOptions {
    fn default() -> Self {
        Self {
            pattern: LinePattern::Solid,
            cap_style: CapStyle::Round,
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
        }
    }
}

#[derive(Default)]
pub struct ToolState {
    last_pos: Option<(u32, u32)>,
    last_precise_pos: Option<Pos2>, // Float position for sub-pixel spacing tracking
    distance_remainder: f32,        // How far we've moved since the last stamp
    last_brush_pos: Option<(u32, u32)>, // For Shift+Click straight lines
    using_secondary_color: bool,
    /// EMA-smoothed brush position for rounding off angular corners
    /// during fast mouse movement.
    smooth_pos: Option<Pos2>,
}

/// Tracks stroke state for undo/redo integration
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
    /// Description of the stroke (e.g., "Brush Stroke", "Eraser Stroke")
    pub description: String,
}

impl StrokeTracker {
    /// Start tracking a new stroke for a tool that uses preview layer (Brush, Line).
    pub fn start_preview_tool(&mut self, layer_index: usize, description: &str) {
        self.is_active = true;
        self.layer_index = layer_index;
        self.bounds = None;
        self.layer_snapshot = None;
        self.uses_preview_layer = true;
        self.description = description.to_string();
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
        self.description = description.to_string();
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
            }
        });

        // Reset state
        self.is_active = false;
        self.layer_index = 0;
        self.bounds = None;
        self.layer_snapshot = None;
        self.uses_preview_layer = false;
        self.description.clear();

        event
    }

    pub fn cancel(&mut self) {
        self.is_active = false;
        self.layer_index = 0;
        self.bounds = None;
        self.layer_snapshot = None;
        self.uses_preview_layer = false;
        self.description.clear();
    }
}

/// Event emitted when a stroke completes
pub struct StrokeEvent {
    pub layer_index: usize,
    pub bounds: Rect,
    pub before_snapshot: Option<PixelPatch>,
    pub description: String,
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

/// State for Magic Wand tool (color-based selection)
#[derive(Clone, Debug)]
pub struct MagicWandState {
    pub tolerance: f32,
    pub active_selection: Option<(u32, u32, Rgba<u8>, Vec<(u32, u32)>)>,
    pub last_preview_tolerance: f32,
    pub tolerance_changed_at: Option<Instant>,
    pub recalc_pending: bool,
    /// Effective selection mode for the current/pending click.
    pub effective_mode: SelectionMode,
    pub global_select: bool,
    pub base_selection_mask: Option<GrayImage>,
}

impl Default for MagicWandState {
    fn default() -> Self {
        Self {
            tolerance: 5.0,
            active_selection: None,
            last_preview_tolerance: 5.0,
            tolerance_changed_at: None,
            recalc_pending: false,
            effective_mode: SelectionMode::Replace,
            global_select: false,
            base_selection_mask: None,
        }
    }
}

/// State for Fill tool (flood fill)
#[derive(Clone, Debug)]
pub struct FillToolState {
    /// Tolerance for color matching (0 = exact, 100 = all).
    pub tolerance: f32,
    pub anti_aliased: bool,
    /// (start_x, start_y, target_color, flat_fill_mask)
    /// flat_fill_mask is width*height bytes: 255=filled, 0=not filled.
    pub active_fill: Option<(u32, u32, Rgba<u8>, Vec<u8>)>,
    pub last_preview_tolerance: f32,
    pub fill_color_u8: Option<Rgba<u8>>,
    pub use_secondary_color: bool,
    pub tolerance_changed_at: Option<Instant>,
    pub recalc_pending: bool,
    pub last_preview_aa: bool,
}

impl Default for FillToolState {
    fn default() -> Self {
        Self {
            tolerance: 5.0,
            anti_aliased: true,
            active_fill: None,
            last_preview_tolerance: 5.0,
            fill_color_u8: None,
            use_secondary_color: false,
            tolerance_changed_at: None,
            recalc_pending: false,
            last_preview_aa: true,
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
        let hsv = super::colors::color_to_hsv(c32);
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
    pub available_fonts: Vec<String>,
    pub font_search: String,
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
    /// Channel receiver for async font preview loading results
    pub font_preview_rx:
        Option<std::sync::mpsc::Receiver<Vec<(String, Option<ab_glyph::FontArc>)>>>,
    /// Font names currently queued for async preview loading
    pub font_preview_pending: std::collections::HashSet<String>,
}

impl Default for TextToolState {
    fn default() -> Self {
        Self {
            text: String::new(),
            cursor_pos: 0,
            origin: None,
            is_editing: false,
            font_family: "Arial".to_string(),
            font_size: 23.0,
            font_weight: 400,
            bold: false,
            italic: false,
            underline: false,
            strikethrough: false,
            anti_alias: true,
            alignment: crate::ops::text::TextAlignment::Left,
            available_fonts: Vec::new(),
            font_search: String::new(),
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
            font_preview_rx: None,
            font_preview_pending: std::collections::HashSet::new(),
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
}

impl Default for ShapesToolState {
    fn default() -> Self {
        Self {
            selected_shape: crate::ops::shapes::ShapeKind::Rectangle,
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
    /// Last color picked this frame; None if color picker not used.
    pub last_picked_color: Option<Color32>,
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
    /// Pixel amount for selection feather/expand/contract (context bar spinner).
    pub sel_modify_radius: f32,
    /// Pending selection modification set from context bar; consumed by app.rs.
    pub pending_sel_modify: Option<SelectionModifyOp>,
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
            sel_modify_radius: 5.0,
            pending_sel_modify: None,
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

impl ToolsPanel {
    /// Take the pending stroke event (if any) for processing
    pub fn take_stroke_event(&mut self) -> Option<StrokeEvent> {
        self.pending_stroke_event.take()
    }

    /// Take the pending color removal request (if any) for async dispatch
    pub fn take_pending_color_removal(&mut self) -> Option<ColorRemovalRequest> {
        self.pending_color_removal.take()
    }

    /// Take the pending inpaint request (if any) for async dispatch
    pub fn take_pending_inpaint(&mut self) -> Option<crate::ops::inpaint::InpaintRequest> {
        self.content_aware_state.pending_inpaint.take()
    }

    /// Returns true if the user is actively painting (brush/eraser stroke in progress).
    /// Useful for triggering continuous repaint during brush strokes.
    pub fn is_stroke_active(&self) -> bool {
        self.stroke_tracker.is_active || self.tool_state.last_pos.is_some()
    }

    /// Returns true if any tool has an uncommitted preview/in-progress state.
    pub fn has_active_tool_preview(&self) -> bool {
        match self.active_tool {
            Tool::Brush | Tool::Pencil | Tool::Eraser => {
                self.stroke_tracker.is_active || self.tool_state.last_pos.is_some()
            }
            Tool::Line => {
                matches!(
                    self.line_state.line_tool.stage,
                    LineStage::Dragging | LineStage::Editing
                )
            }
            Tool::Gradient => self.gradient_state.drag_start.is_some(),
            Tool::Shapes => self.shapes_state.placed.is_some() || self.shapes_state.is_drawing,
            Tool::Text => self.text_state.is_editing,
            Tool::Fill => self.fill_state.active_fill.is_some(),
            Tool::Liquify => self.liquify_state.is_active,
            Tool::MeshWarp => self.mesh_warp_state.is_active,
            _ => false,
        }
    }

    /// Cancel the current tool's in-progress operation without committing.
    /// Returns true if something was cancelled, false if there was nothing to cancel.
    pub fn cancel_active_tool(&mut self, canvas_state: &mut CanvasState) -> bool {
        match self.active_tool {
            Tool::Brush | Tool::Pencil | Tool::Eraser => {
                if self.stroke_tracker.is_active || self.tool_state.last_pos.is_some() {
                    self.stroke_tracker.cancel();
                    self.tool_state.last_pos = None;
                    self.tool_state.last_precise_pos = None;
                    self.tool_state.smooth_pos = None;
                    canvas_state.clear_preview_state();
                    canvas_state.mark_dirty(None);
                    true
                } else {
                    false
                }
            }
            Tool::Line => match self.line_state.line_tool.stage {
                LineStage::Dragging | LineStage::Editing => {
                    self.stroke_tracker.cancel();
                    canvas_state.clear_preview_state();
                    if let Some(bounds) = self.line_state.line_tool.last_bounds {
                        canvas_state.mark_dirty(Some(bounds));
                    } else {
                        canvas_state.mark_dirty(None);
                    }
                    self.line_state.line_tool.stage = LineStage::Idle;
                    self.line_state.line_tool.last_bounds = None;
                    self.line_state.line_tool.require_mouse_release = true;
                    self.line_state.line_tool.initial_mouse_pos = None;
                    self.line_state.line_tool.dragging_handle = None;
                    true
                }
                _ => false,
            },
            Tool::Gradient => {
                if self.gradient_state.drag_start.is_some() {
                    self.cancel_gradient(canvas_state);
                    true
                } else {
                    false
                }
            }
            Tool::Shapes => {
                if self.shapes_state.placed.is_some() {
                    self.shapes_state.placed = None;
                    canvas_state.clear_preview_state();
                    canvas_state.mark_dirty(None);
                    true
                } else if self.shapes_state.is_drawing {
                    self.shapes_state.is_drawing = false;
                    self.shapes_state.draw_start = None;
                    self.shapes_state.draw_end = None;
                    canvas_state.clear_preview_state();
                    canvas_state.mark_dirty(None);
                    true
                } else {
                    false
                }
            }
            Tool::Text => {
                if self.text_state.is_editing {
                    self.text_state.text.clear();
                    self.text_state.cursor_pos = 0;
                    self.text_state.is_editing = false;
                    self.text_state.origin = None;
                    self.text_state.dragging_handle = false;
                    canvas_state.clear_preview_state();
                    canvas_state.mark_dirty(None);
                    true
                } else {
                    false
                }
            }
            Tool::Fill => {
                if self.fill_state.active_fill.is_some() {
                    self.fill_state.active_fill = None;
                    self.fill_state.fill_color_u8 = None;
                    canvas_state.clear_preview_state();
                    self.stroke_tracker.cancel();
                    true
                } else {
                    false
                }
            }
            Tool::Liquify => {
                if self.liquify_state.is_active {
                    self.liquify_state.displacement = None;
                    self.liquify_state.is_active = false;
                    self.liquify_state.source_snapshot = None;
                    self.liquify_state.warp_buffer.clear();
                    self.liquify_state.dirty_rect = None;
                    canvas_state.clear_preview_state();
                    canvas_state.mark_dirty(None);
                    true
                } else {
                    false
                }
            }
            Tool::MeshWarp => {
                if self.mesh_warp_state.is_active {
                    self.mesh_warp_state.is_active = false;
                    self.mesh_warp_state.source_snapshot = None;
                    self.mesh_warp_state.warp_buffer.clear();
                    self.stroke_tracker.cancel();
                    canvas_state.clear_preview_state();
                    canvas_state.mark_dirty(None);
                    true
                } else {
                    false
                }
            }
            _ => false,
        }
    }

    /// Compact vertical tool strip for floating window
    /// Returns an action indicating what the user clicked
    pub fn show_compact(
        &mut self,
        ui: &mut egui::Ui,
        assets: &Assets,
        primary_color: egui::Color32,
        secondary_color: egui::Color32,
        keybindings: &crate::assets::KeyBindings,
    ) -> ToolsPanelAction {
        let mut action = ToolsPanelAction::None;

        let btn_size = 26.0; // visual button size
        let cols = 3;
        let gap = 6.0; // 1px gap between buttons

        ui.spacing_mut().item_spacing = egui::vec2(0.0, 0.0);
        ui.spacing_mut().button_padding = egui::vec2(2.0, 2.0);

        ui.add_space(4.0);

        // Separator color for between-group dividers
        let sep_color = if ui.visuals().dark_mode {
            egui::Color32::from_gray(60)
        } else {
            egui::Color32::from_gray(200)
        };

        // Tool groups — 4 groups of 6, each 2 rows × 3 cols, with separators
        // PAINT: Core painting & fill tools
        let paint_tools: Vec<(Icon, Tool)> = vec![
            (Icon::Brush, Tool::Brush),
            (Icon::Pencil, Tool::Pencil),
            (Icon::Eraser, Tool::Eraser),
            (Icon::Line, Tool::Line),
            (Icon::Fill, Tool::Fill),
            (Icon::Gradient, Tool::Gradient),
        ];
        // SELECT: Region selection & movement
        let select_tools: Vec<(Icon, Tool)> = vec![
            (Icon::RectSelect, Tool::RectangleSelect),
            (Icon::EllipseSelect, Tool::EllipseSelect),
            (Icon::Lasso, Tool::Lasso),
            (Icon::MagicWand, Tool::MagicWand),
            (Icon::MovePixels, Tool::MovePixels),
            (Icon::MoveSelection, Tool::MoveSelection),
        ];
        // RETOUCH & WARP: Repair/clone + distort/transform
        let retouch_tools: Vec<(Icon, Tool)> = vec![
            (Icon::CloneStamp, Tool::CloneStamp),
            (Icon::ContentAwareBrush, Tool::ContentAwareBrush),
            (Icon::ColorRemover, Tool::ColorRemover),
            (Icon::Liquify, Tool::Liquify),
            (Icon::MeshWarp, Tool::MeshWarp),
            (Icon::PerspectiveCrop, Tool::PerspectiveCrop),
        ];
        // UTILITY: Sample, create, navigate
        let utility_tools: Vec<(Icon, Tool)> = vec![
            (Icon::ColorPicker, Tool::ColorPicker),
            (Icon::Text, Tool::Text),
            (Icon::Zoom, Tool::Zoom),
            (Icon::Pan, Tool::Pan),
        ];

        let groups: Vec<&Vec<(Icon, Tool)>> =
            vec![&paint_tools, &select_tools, &retouch_tools, &utility_tools];
        let sep_gap = 11.0; // vertical space for separator lines between groups (5px padding each side + 1px line)
        let grid_w = cols as f32 * btn_size + (cols - 1) as f32 * gap;

        // Calculate total height: all tool rows + separators between groups
        let total_tool_rows: usize = groups.iter().map(|g| g.len().div_ceil(cols)).sum();
        let num_separators = groups.len() - 1;
        let grid_h = total_tool_rows as f32 * btn_size
            + (total_tool_rows - 1) as f32 * gap
            + num_separators as f32 * sep_gap;

        // Allocate exact space for entire tool grid (all groups + separators)
        let (grid_rect, _) =
            ui.allocate_exact_size(egui::vec2(grid_w, grid_h), egui::Sense::hover());

        let mut current_y = grid_rect.min.y;
        let dark_mode = ui.visuals().dark_mode;

        // In dark mode, use a noticeably darker fill than the panel to create a recessed look
        let tool_btn_fill = if dark_mode {
            egui::Color32::from_gray(18)
        } else {
            ui.visuals().widgets.inactive.bg_fill
        };

        for (gi, group) in groups.iter().enumerate() {
            let group_rows = group.len().div_ceil(cols);

            for (i, (icon, tool)) in group.iter().enumerate() {
                let col = i % cols;
                let row = i / cols;
                let x = grid_rect.min.x + col as f32 * (btn_size + gap);
                let y = current_y + row as f32 * (btn_size + gap);
                let btn_rect =
                    egui::Rect::from_min_size(egui::pos2(x, y), egui::vec2(btn_size, btn_size));

                let selected = self.active_tool == *tool;

                // Manual painting (like Shapes button) for full control over fill/tint
                let resp = ui.allocate_rect(btn_rect, egui::Sense::click());
                let hovered = resp.hovered();

                // Background fill — selected > hovered > recessed default
                let fill = if selected {
                    ui.visuals().selection.bg_fill
                } else if hovered {
                    ui.visuals().widgets.hovered.bg_fill
                } else {
                    tool_btn_fill
                };
                ui.painter().rect_filled(btn_rect, 4.0, fill);

                // Border
                if selected {
                    ui.painter().rect_stroke(
                        btn_rect,
                        2.0,
                        egui::Stroke::new(2.0, ui.visuals().selection.bg_fill),
                    );
                } else if hovered {
                    ui.painter()
                        .rect_stroke(btn_rect, 4.0, ui.visuals().widgets.hovered.bg_stroke);
                }

                // Draw icon image or emoji fallback
                if let Some(texture) = assets.get_texture(*icon) {
                    let sized_texture = egui::load::SizedTexture::from_handle(texture);
                    let img_size = egui::vec2(btn_size * 0.75, btn_size * 0.75);
                    // In dark mode, tint icons full white for contrast (inverted icons are white-on-transparent)
                    let img = if dark_mode && !selected {
                        egui::Image::from_texture(sized_texture)
                            .fit_to_exact_size(img_size)
                            .tint(egui::Color32::WHITE)
                    } else {
                        egui::Image::from_texture(sized_texture).fit_to_exact_size(img_size)
                    };
                    let img_rect = egui::Rect::from_center_size(btn_rect.center(), img_size);
                    img.paint_at(ui, img_rect);
                } else {
                    let text_color = if selected {
                        egui::Color32::WHITE
                    } else {
                        ui.visuals().text_color()
                    };
                    ui.painter().text(
                        btn_rect.center(),
                        egui::Align2::CENTER_CENTER,
                        icon.emoji(),
                        egui::FontId::proportional(btn_size * 0.42),
                        text_color,
                    );
                }

                if resp
                    .on_hover_text(icon.tooltip_with_keybind(keybindings))
                    .clicked()
                {
                    self.change_tool(*tool);
                }
            }

            // Advance Y past this group's rows
            current_y += group_rows as f32 * btn_size + (group_rows - 1) as f32 * gap;

            // Draw separator line between groups (not after the last group)
            if gi < groups.len() - 1 {
                let sep_y = current_y + sep_gap * 0.5;
                ui.painter().line_segment(
                    [
                        egui::pos2(grid_rect.min.x, sep_y),
                        egui::pos2(grid_rect.min.x + grid_w, sep_y),
                    ],
                    egui::Stroke::new(1.0, sep_color),
                );
                current_y += sep_gap;
            }
        }

        // Shapes — double-wide, sits in the last row of the utility group
        {
            // The utility group has 4 tools → row0: 3 tools, row1: Pan only (col 0)
            // Shapes goes in col 1-2 of that last row
            let shape_x = grid_rect.min.x + 1.0 * (btn_size + gap);
            let shape_y = current_y - btn_size; // last row Y (Pan's row)
            let remaining_cols = cols - 1; // 2 columns
            let shape_w = remaining_cols as f32 * btn_size + (remaining_cols - 1) as f32 * gap;
            let shape_rect = egui::Rect::from_min_size(
                egui::pos2(shape_x, shape_y),
                egui::vec2(shape_w, btn_size),
            );

            let icon = Icon::Shapes;
            let is_shapes = self.active_tool == Tool::Shapes;

            let resp = ui.allocate_rect(shape_rect, egui::Sense::click());
            let hovered = resp.hovered();

            let fill = if is_shapes {
                ui.visuals().selection.bg_fill
            } else if hovered {
                ui.visuals().widgets.hovered.bg_fill
            } else {
                tool_btn_fill
            };
            ui.painter().rect_filled(shape_rect, 4.0, fill);

            if is_shapes {
                ui.painter().rect_stroke(
                    shape_rect,
                    2.0,
                    egui::Stroke::new(2.0, ui.visuals().selection.bg_fill),
                );
            } else if hovered {
                ui.painter()
                    .rect_stroke(shape_rect, 4.0, ui.visuals().widgets.hovered.bg_stroke);
            }

            if let Some(texture) = assets.get_texture(icon) {
                let sized_texture = egui::load::SizedTexture::from_handle(texture);
                let img = egui::Image::from_texture(sized_texture)
                    .fit_to_exact_size(egui::vec2(shape_w * 0.75, btn_size * 0.75));
                let img = if dark_mode && !is_shapes {
                    img.tint(egui::Color32::WHITE)
                } else {
                    img
                };
                let img_rect = egui::Rect::from_center_size(
                    shape_rect.center(),
                    egui::vec2(shape_w * 0.75, btn_size * 0.75),
                );
                img.paint_at(ui, img_rect);
            } else {
                let text_color = if is_shapes {
                    egui::Color32::WHITE
                } else {
                    ui.visuals().text_color()
                };
                ui.painter().text(
                    shape_rect.center(),
                    egui::Align2::CENTER_CENTER,
                    icon.emoji(),
                    egui::FontId::proportional(btn_size * 0.42),
                    text_color,
                );
            }

            if resp
                .on_hover_text(icon.tooltip_with_keybind(keybindings))
                .clicked()
            {
                self.change_tool(Tool::Shapes);
            }
        }

        ui.add_space(8.0);

        // Separator line before color swatches
        let sep_width = grid_w;
        let (sep_rect, _) =
            ui.allocate_exact_size(egui::vec2(sep_width, 1.0), egui::Sense::hover());
        ui.painter().line_segment(
            [sep_rect.left_center(), sep_rect.right_center()],
            egui::Stroke::new(1.0, sep_color),
        );
        ui.add_space(10.0);

        // Color swatches — centered in panel
        ui.horizontal(|ui| {
            ui.add_space(28.0);
            if Self::draw_color_swatch_compact(ui, primary_color, secondary_color) {
                action = ToolsPanelAction::OpenColors;
            }
        });

        ui.add_space(-8.0);

        // Swap button — centered in panel (frameless)
        ui.horizontal(|ui| {
            ui.add_space(28.0);
            let clicked = if let Some(texture) = assets.get_texture(Icon::SwapColors) {
                let sized_texture = egui::load::SizedTexture::from_handle(texture);
                let img = egui::Image::from_texture(sized_texture)
                    .fit_to_exact_size(egui::vec2(16.0, 16.0));
                ui.add(egui::ImageButton::new(img).frame(false))
                    .on_hover_text(Icon::SwapColors.tooltip())
                    .clicked()
            } else {
                assets.small_icon_button(ui, Icon::SwapColors).clicked()
            };
            if clicked {
                action = ToolsPanelAction::SwapColors;
            }
        });

        ui.add_space(4.0);

        action
    }

    /// Draw compact overlapping primary/secondary color swatch
    fn draw_color_swatch_compact(
        ui: &mut egui::Ui,
        primary: egui::Color32,
        secondary: egui::Color32,
    ) -> bool {
        let swatch_size = 24.0;
        let offset = 8.0;
        let total_size = egui::vec2(swatch_size + offset, swatch_size + offset);

        let (rect, response) = ui.allocate_exact_size(total_size, egui::Sense::click());

        if ui.is_rect_visible(rect) {
            let painter = ui.painter();

            // Secondary color (back, offset down-right)
            let secondary_rect = egui::Rect::from_min_size(
                rect.min + egui::vec2(offset, offset),
                egui::vec2(swatch_size, swatch_size),
            );
            painter.rect_filled(secondary_rect, 2.0, secondary);
            painter.rect_stroke(
                secondary_rect,
                2.0,
                egui::Stroke::new(1.0, egui::Color32::from_gray(100)),
            );

            // Primary color (front, top-left)
            let primary_rect =
                egui::Rect::from_min_size(rect.min, egui::vec2(swatch_size, swatch_size));
            painter.rect_filled(primary_rect, 2.0, primary);
            painter.rect_stroke(
                primary_rect,
                2.0,
                egui::Stroke::new(1.0, egui::Color32::from_gray(100)),
            );
        }

        response.on_hover_text("Click to open Colors").clicked()
    }

    /// Draw larger overlapping primary/secondary color swatch (centered)
    /// Returns true if clicked (to open colors panel)
    fn draw_color_swatch_large(
        ui: &mut egui::Ui,
        primary: egui::Color32,
        secondary: egui::Color32,
    ) -> bool {
        let swatch_size = 32.0;
        let offset = 10.0;
        let total_size = egui::vec2(swatch_size + offset, swatch_size + offset);

        let (rect, response) = ui.allocate_exact_size(total_size, egui::Sense::click());

        if ui.is_rect_visible(rect) {
            let painter = ui.painter();

            // Secondary color (back, offset down-right)
            let secondary_rect = egui::Rect::from_min_size(
                rect.min + egui::vec2(offset, offset),
                egui::vec2(swatch_size, swatch_size),
            );
            painter.rect_filled(secondary_rect, 3.0, secondary);
            painter.rect_stroke(
                secondary_rect,
                3.0,
                egui::Stroke::new(1.0, egui::Color32::from_gray(80)),
            );

            // Primary color (front, top-left)
            let primary_rect =
                egui::Rect::from_min_size(rect.min, egui::vec2(swatch_size, swatch_size));
            painter.rect_filled(primary_rect, 3.0, primary);
            painter.rect_stroke(
                primary_rect,
                3.0,
                egui::Stroke::new(1.0, egui::Color32::from_gray(80)),
            );
        }

        response.on_hover_text("Click to open Colors").clicked()
    }

    /// Draw overlapping primary/secondary color swatch
    /// Returns true if clicked (to open colors panel)
    pub fn draw_color_swatch(
        ui: &mut egui::Ui,
        primary: egui::Color32,
        secondary: egui::Color32,
    ) -> bool {
        let swatch_size = 24.0;
        let offset = 8.0;
        let total_size = egui::vec2(swatch_size + offset, swatch_size + offset);

        let (rect, response) = ui.allocate_exact_size(total_size, egui::Sense::click());

        if ui.is_rect_visible(rect) {
            let painter = ui.painter();

            // Secondary color (back, offset down-right)
            let secondary_rect = egui::Rect::from_min_size(
                rect.min + egui::vec2(offset, offset),
                egui::vec2(swatch_size, swatch_size),
            );
            painter.rect_filled(secondary_rect, 2.0, secondary);
            painter.rect_stroke(
                secondary_rect,
                2.0,
                egui::Stroke::new(1.0, egui::Color32::from_gray(80)),
            );

            // Primary color (front, top-left)
            let primary_rect =
                egui::Rect::from_min_size(rect.min, egui::vec2(swatch_size, swatch_size));
            painter.rect_filled(primary_rect, 2.0, primary);
            painter.rect_stroke(
                primary_rect,
                2.0,
                egui::Stroke::new(1.0, egui::Color32::from_gray(80)),
            );
        }

        response.on_hover_text("Click to open Colors").clicked()
    }

    /// Original full show method for sidebar (kept for compatibility)
    pub fn show(&mut self, ui: &mut egui::Ui, assets: &Assets) {
        ui.vertical(|ui| {
            ui.heading("Tools");
            ui.separator();

            // Large icon-style tool buttons
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 4.0;

                let is_brush = self.active_tool == Tool::Brush;
                if assets.icon_selectable(ui, Icon::Brush, is_brush) {
                    self.change_tool(Tool::Brush);
                }

                let is_eraser = self.active_tool == Tool::Eraser;
                if assets.icon_selectable(ui, Icon::Eraser, is_eraser) {
                    self.change_tool(Tool::Eraser);
                }

                let is_line = self.active_tool == Tool::Line;
                if assets.icon_selectable(ui, Icon::Line, is_line) {
                    self.change_tool(Tool::Line);
                }

                let is_rect_sel = self.active_tool == Tool::RectangleSelect;
                if assets.icon_selectable(ui, Icon::RectSelect, is_rect_sel) {
                    self.change_tool(Tool::RectangleSelect);
                }

                let is_ellipse_sel = self.active_tool == Tool::EllipseSelect;
                if assets.icon_selectable(ui, Icon::EllipseSelect, is_ellipse_sel) {
                    self.change_tool(Tool::EllipseSelect);
                }

                let is_move_px = self.active_tool == Tool::MovePixels;
                if assets.icon_selectable(ui, Icon::MovePixels, is_move_px) {
                    self.change_tool(Tool::MovePixels);
                }

                let is_move_sel = self.active_tool == Tool::MoveSelection;
                if assets.icon_selectable(ui, Icon::MoveSelection, is_move_sel) {
                    self.change_tool(Tool::MoveSelection);
                }
            });

            ui.separator();

            // Tool name label
            let tool_name = match self.active_tool {
                Tool::Brush => t!("tool.brush"),
                Tool::Eraser => t!("tool.eraser"),
                Tool::Pencil => t!("tool.pencil"),
                Tool::Line => t!("tool.line"),
                Tool::RectangleSelect => t!("tool.rectangle_select"),
                Tool::EllipseSelect => t!("tool.ellipse_select"),
                Tool::MovePixels => t!("tool.move_pixels"),
                Tool::MoveSelection => t!("tool.move_selection"),
                Tool::MagicWand => t!("tool.magic_wand"),
                Tool::Fill => t!("tool.fill"),
                Tool::ColorPicker => t!("tool.color_picker"),
                Tool::Gradient => t!("tool.gradient"),
                Tool::ContentAwareBrush => t!("tool.content_aware_fill"),
                Tool::Liquify => t!("tool.liquify"),
                Tool::MeshWarp => t!("tool.mesh_warp"),
                Tool::ColorRemover => t!("tool.color_remover"),
                Tool::Smudge => "Smudge".to_string(),
                Tool::CloneStamp => t!("tool.clone_stamp"),
                Tool::Text => t!("tool.text"),
                Tool::PerspectiveCrop => t!("tool.perspective_crop"),
                Tool::Lasso => t!("tool.lasso"),
                Tool::Zoom => t!("tool.zoom"),
                Tool::Pan => t!("tool.pan"),
                Tool::Shapes => t!("tool.shapes"),
            };
            ui.label(egui::RichText::new(tool_name).strong());
        });
    }

    /// Change tool and handle any cleanup needed (like committing active Bézier line)
    pub fn change_tool(&mut self, new_tool: Tool) {
        if self.active_tool != new_tool {
            // Deactivate perspective crop when switching away
            if self.active_tool == Tool::PerspectiveCrop {
                self.perspective_crop_state.active = false;
                self.perspective_crop_state.dragging_corner = None;
            }
            self.active_tool = new_tool;
            // Auto-init perspective crop — need canvas dims, so flag for
            // lazy init in handle_input on next frame.
            if new_tool == Tool::PerspectiveCrop {
                self.perspective_crop_state.needs_auto_init = true;
            }
            // Note: Actual commitment will be handled in handle_input with canvas_state access
        }
    }

    /// Get the name of the active tool for display in context bar
    pub fn active_tool_name(&self) -> String {
        match self.active_tool {
            Tool::Brush => t!("tool.brush"),
            Tool::Eraser => t!("tool.eraser"),
            Tool::Pencil => t!("tool.pencil"),
            Tool::Line => t!("tool.line"),
            Tool::RectangleSelect => t!("tool.rectangle_select"),
            Tool::EllipseSelect => t!("tool.ellipse_select"),
            Tool::MovePixels => t!("tool.move_pixels"),
            Tool::MoveSelection => t!("tool.move_selection"),
            Tool::MagicWand => t!("tool.magic_wand"),
            Tool::Fill => t!("tool.fill"),
            Tool::ColorPicker => t!("tool.color_picker"),
            Tool::Gradient => t!("tool.gradient"),
            Tool::ContentAwareBrush => t!("tool.content_aware_fill"),
            Tool::Liquify => t!("tool.liquify"),
            Tool::MeshWarp => t!("tool.mesh_warp"),
            Tool::ColorRemover => t!("tool.color_remover"),
            Tool::Smudge => "Smudge".to_string(),
            Tool::CloneStamp => t!("tool.clone_stamp"),
            Tool::Text => t!("tool.text"),
            Tool::PerspectiveCrop => t!("tool.perspective_crop"),
            Tool::Lasso => t!("tool.lasso"),
            Tool::Zoom => t!("tool.zoom"),
            Tool::Pan => t!("tool.pan"),
            Tool::Shapes => t!("tool.shapes"),
        }
    }

    /// Dynamic context bar that shows options based on active tool
    /// This replaces show_properties_toolbar for the floating window layout
    pub fn show_context_bar(
        &mut self,
        ui: &mut egui::Ui,
        assets: &Assets,
        primary_color: Color32,
        secondary_color: Color32,
    ) {
        ui.horizontal(|ui| {
            // Tool name label
            ui.label(egui::RichText::new(self.active_tool_name()).strong());
            ui.separator();

            // Rebuild tip mask cache for brush/eraser tools
            // Called both before AND after tool options UI so picker changes
            // take effect in the same frame (picker runs inside show_*_options).
            match self.active_tool {
                Tool::Brush | Tool::Eraser => {
                    self.rebuild_tip_mask(assets);
                }
                _ => {}
            }

            match self.active_tool {
                Tool::Brush | Tool::Pencil => {
                    self.show_brush_options(ui, assets);
                }
                Tool::Line => {
                    self.show_line_options(ui);
                }
                Tool::Eraser => {
                    self.show_eraser_options(ui, assets);
                }
                _ => {}
            }
            // Re-run rebuild_tip_mask after options UI so that picker changes
            // (tip/size/hardness) take effect this frame, not next frame.
            match self.active_tool {
                Tool::Brush | Tool::Eraser => {
                    self.rebuild_tip_mask(assets);
                }
                _ => {}
            }
            match self.active_tool {
                // Already handled above
                Tool::Brush | Tool::Pencil | Tool::Line | Tool::Eraser => {}
                Tool::RectangleSelect | Tool::EllipseSelect => {
                    self.show_selection_options(ui);
                }
                Tool::MovePixels => {
                    ui.label(
                        "Click + drag to move selected pixels. No selection = move entire image.",
                    );
                }
                Tool::MoveSelection => {
                    ui.label("Click + drag to move the selection boundary.");
                }
                Tool::MagicWand => {
                    self.show_magic_wand_options(ui);
                }
                Tool::Fill => {
                    self.show_fill_options(ui);
                }
                Tool::ColorPicker => {
                    ui.label("Left-click to pick Primary color. Right-click for Secondary color.");
                }
                Tool::Lasso => {
                    self.show_lasso_options(ui);
                }
                Tool::Zoom => {
                    // Toggle button for zoom direction (touch-friendly)
                    let label = if self.zoom_tool_state.zoom_out_mode {
                        "\u{1F50D}\u{2796} Zoom Out"
                    } else {
                        "\u{1F50D}\u{2795} Zoom In"
                    };
                    if ui
                        .selectable_label(self.zoom_tool_state.zoom_out_mode, label)
                        .clicked()
                    {
                        self.zoom_tool_state.zoom_out_mode = !self.zoom_tool_state.zoom_out_mode;
                    }
                    ui.separator();
                    ui.label("Click/tap to zoom. Drag a rectangle to zoom to area.");
                }
                Tool::Pan => {
                    ui.label("Click and drag to pan the canvas viewport.");
                }
                Tool::PerspectiveCrop => {
                    self.show_perspective_crop_options(ui);
                }
                Tool::Gradient => {
                    self.show_gradient_options(ui, assets, primary_color, secondary_color);
                }
                Tool::Liquify => {
                    self.show_liquify_options(ui);
                }
                Tool::MeshWarp => {
                    self.show_mesh_warp_options(ui);
                }
                Tool::ColorRemover => {
                    self.show_color_remover_options(ui);
                }
                Tool::Smudge => {
                    self.show_smudge_options(ui);
                }
                Tool::Text => {
                    self.show_text_options(ui);
                }
                Tool::Shapes => {
                    self.show_shapes_options(ui, assets);
                }
                Tool::CloneStamp => {
                    self.show_clone_stamp_options(ui);
                }
                Tool::ContentAwareBrush => {
                    self.show_content_aware_options(ui);
                }
            }
        });
    }

    /// Show brush tip picker dropdown (grid popup with categories, matching shapes tool pattern)
    fn show_brush_tip_picker(&mut self, ui: &mut egui::Ui, assets: &Assets) {
        ui.label("Tip:");

        let popup_id = ui.make_persistent_id("brush_tip_grid_popup");
        let display_name = self.properties.brush_tip.display_name().to_string();

        // Button showing current tip icon + name
        let btn_response = {
            if let BrushTip::Image(ref name) = self.properties.brush_tip {
                if let Some(tex) = assets.get_brush_tip_texture(name) {
                    let sized = egui::load::SizedTexture::from_handle(tex);
                    let img =
                        egui::Image::from_texture(sized).fit_to_exact_size(egui::Vec2::splat(16.0));
                    let btn = egui::Button::image_and_text(img, &display_name);
                    ui.add(btn)
                } else {
                    ui.button(&display_name)
                }
            } else {
                // Circle — draw a small filled circle icon on the button
                let btn = ui.button(format!("      {}", display_name));
                let rect = btn.rect;
                let circle_x = rect.left() + 14.0;
                let circle_y = rect.center().y;
                ui.painter().circle_filled(
                    egui::Pos2::new(circle_x, circle_y),
                    5.0,
                    ui.visuals().text_color(),
                );
                btn
            }
        };
        if btn_response.clicked() {
            ui.memory_mut(|m| m.toggle_popup(popup_id));
        }

        egui::popup_below_widget(ui, popup_id, &btn_response, |ui| {
            ui.set_min_width(240.0);

            let cols = 5;
            let icon_size = egui::Vec2::splat(36.0);
            let accent = ui.visuals().hyperlink_color;

            // "Basic" category header always first, with Circle as first item
            ui.label(egui::RichText::new("Basic").strong().size(11.0));
            egui::Grid::new("brush_tip_basic_grid")
                .spacing(egui::Vec2::splat(2.0))
                .show(ui, |ui| {
                    // Circle (built-in, always first)
                    let selected = self.properties.brush_tip.is_circle();
                    let (rect, response) = ui.allocate_exact_size(icon_size, egui::Sense::click());
                    if selected {
                        ui.painter().rect_filled(
                            rect,
                            4.0,
                            egui::Color32::from_rgba_premultiplied(
                                accent.r(),
                                accent.g(),
                                accent.b(),
                                60,
                            ),
                        );
                    }
                    if response.hovered() {
                        ui.painter()
                            .rect_filled(rect, 4.0, ui.visuals().widgets.hovered.bg_fill);
                    }
                    // Draw a circle icon
                    let center = rect.center();
                    let r = icon_size.x * 0.3;
                    let stroke_color = ui.visuals().text_color();
                    ui.painter().circle_filled(center, r, stroke_color);
                    if response.clicked() {
                        self.properties.brush_tip = BrushTip::Circle;
                    }
                    response.on_hover_text("Circle");

                    // Other tips in the "Basic" category
                    let mut col = 1;
                    let categories = assets.brush_tip_categories();
                    if let Some(basic_cat) = categories.iter().find(|c| c.name == "Basic") {
                        for tip_name in &basic_cat.tips {
                            let is_selected =
                                self.properties.brush_tip == BrushTip::Image(tip_name.clone());
                            let (rect, response) =
                                ui.allocate_exact_size(icon_size, egui::Sense::click());
                            if is_selected {
                                ui.painter().rect_filled(
                                    rect,
                                    4.0,
                                    egui::Color32::from_rgba_premultiplied(
                                        accent.r(),
                                        accent.g(),
                                        accent.b(),
                                        60,
                                    ),
                                );
                            }
                            if response.hovered() {
                                ui.painter().rect_filled(
                                    rect,
                                    4.0,
                                    ui.visuals().widgets.hovered.bg_fill,
                                );
                            }
                            if let Some(tex) = assets.get_brush_tip_texture(tip_name) {
                                let sized = egui::load::SizedTexture::from_handle(tex);
                                let img = egui::Image::from_texture(sized)
                                    .fit_to_exact_size(icon_size * 0.8);
                                let inner_rect = rect.shrink(icon_size.x * 0.1);
                                img.paint_at(ui, inner_rect);
                            } else {
                                ui.painter().text(
                                    rect.center(),
                                    egui::Align2::CENTER_CENTER,
                                    &tip_name[..2.min(tip_name.len())],
                                    egui::FontId::proportional(11.0),
                                    ui.visuals().text_color(),
                                );
                            }
                            if response.clicked() {
                                self.properties.brush_tip = BrushTip::Image(tip_name.clone());
                            }
                            response.on_hover_text(tip_name);
                            col += 1;
                            if col % cols == 0 {
                                ui.end_row();
                            }
                        }
                    }
                });

            // Remaining categories
            let categories = assets.brush_tip_categories();
            for cat in categories.iter().filter(|c| c.name != "Basic") {
                ui.add_space(4.0);
                ui.label(egui::RichText::new(&cat.name).strong().size(11.0));
                egui::Grid::new(format!("brush_tip_{}_grid", cat.name))
                    .spacing(egui::Vec2::splat(2.0))
                    .show(ui, |ui| {
                        for (i, tip_name) in cat.tips.iter().enumerate() {
                            let is_selected =
                                self.properties.brush_tip == BrushTip::Image(tip_name.clone());
                            let (rect, response) =
                                ui.allocate_exact_size(icon_size, egui::Sense::click());
                            if is_selected {
                                ui.painter().rect_filled(
                                    rect,
                                    4.0,
                                    egui::Color32::from_rgba_premultiplied(
                                        accent.r(),
                                        accent.g(),
                                        accent.b(),
                                        60,
                                    ),
                                );
                            }
                            if response.hovered() {
                                ui.painter().rect_filled(
                                    rect,
                                    4.0,
                                    ui.visuals().widgets.hovered.bg_fill,
                                );
                            }
                            if let Some(tex) = assets.get_brush_tip_texture(tip_name) {
                                let sized = egui::load::SizedTexture::from_handle(tex);
                                let img = egui::Image::from_texture(sized)
                                    .fit_to_exact_size(icon_size * 0.8);
                                let inner_rect = rect.shrink(icon_size.x * 0.1);
                                img.paint_at(ui, inner_rect);
                            } else {
                                ui.painter().text(
                                    rect.center(),
                                    egui::Align2::CENTER_CENTER,
                                    &tip_name[..2.min(tip_name.len())],
                                    egui::FontId::proportional(11.0),
                                    ui.visuals().text_color(),
                                );
                            }
                            if response.clicked() {
                                self.properties.brush_tip = BrushTip::Image(tip_name.clone());
                            }
                            response.on_hover_text(tip_name);
                            if (i + 1) % cols == 0 {
                                ui.end_row();
                            }
                        }
                    });
            }
        });
    }

    /// Show brush-specific options (size, hardness, blend mode)
    /// For Pencil tool, skip size and hardness since it always paints single pixels
    fn show_brush_options(&mut self, ui: &mut egui::Ui, assets: &Assets) {
        // Brush tip picker (skip for Pencil — always pixel)
        if self.active_tool != Tool::Pencil {
            self.show_brush_tip_picker(ui, assets);
            ui.separator();
        }

        // Size - skip for Pencil tool since it always paints single pixels
        if self.active_tool != Tool::Pencil {
            ui.label(t!("ctx.size"));
            let size_text = format!("{:.0}px", self.properties.size);
            egui::ComboBox::from_id_source("ctx_brush_size")
                .selected_text(&size_text)
                .width(65.0)
                .show_ui(ui, |ui| {
                    for &preset in BRUSH_SIZE_PRESETS.iter() {
                        let label = format!("{:.0}px", preset);
                        if ui
                            .selectable_label((self.properties.size - preset).abs() < 0.1, &label)
                            .clicked()
                        {
                            self.properties.size = preset;
                        }
                    }
                });
            ui.horizontal(|ui| {
                if ui.button("-").clicked() {
                    self.properties.size = (self.properties.size - 1.0).max(1.0);
                }
                ui.add(
                    egui::DragValue::new(&mut self.properties.size)
                        .speed(0.5)
                        .clamp_range(1.0..=256.0)
                        .suffix("px"),
                )
                .on_hover_text(t!("ctx.size_drag_tooltip"));
                if ui.button("+").clicked() {
                    self.properties.size = (self.properties.size + 1.0).min(256.0);
                }
            });

            ui.separator();
        }

        // Hardness - skip for Pencil tool since it always paints single pixels and Line tool
        if self.active_tool != Tool::Pencil && self.active_tool != Tool::Line {
            ui.label(t!("ctx.hardness"));
            let mut hardness_pct = (self.properties.hardness * 100.0).round();
            if ui
                .add(
                    egui::DragValue::new(&mut hardness_pct)
                        .speed(1.0)
                        .clamp_range(0.0..=100.0)
                        .suffix("%"),
                )
                .on_hover_text(t!("ctx.hardness_brush_tooltip"))
                .changed()
            {
                self.properties.hardness = hardness_pct / 100.0;
            }

            ui.separator();
        }

        // Blend Mode
        ui.label(t!("ctx.blend"));
        let current_mode = self.properties.blending_mode;
        egui::ComboBox::from_id_source("ctx_blend_mode")
            .selected_text(current_mode.name())
            .width(90.0)
            .show_ui(ui, |ui| {
                for &mode in BlendMode::all() {
                    if ui
                        .selectable_label(mode == current_mode, mode.name())
                        .clicked()
                    {
                        self.properties.blending_mode = mode;
                    }
                }
            });

        ui.separator();

        // Anti-aliasing toggle (compact "AA" matching Text tool)
        let aa_resp = ui.selectable_label(self.properties.anti_aliased, t!("ctx.anti_alias"));
        if aa_resp.clicked() {
            self.properties.anti_aliased = !self.properties.anti_aliased;
        }
        aa_resp.on_hover_text(t!("ctx.anti_alias_tooltip"));

        // Brush Mode (Normal/Dodge/Burn/Sponge) - only for Brush tool (disabled for now)
        if self.active_tool == Tool::Brush {
            ui.separator();
            ui.add_enabled_ui(false, |ui| {
                ui.label("Mode:");
                let current_bm = self.properties.brush_mode;
                egui::ComboBox::from_id_source("ctx_brush_mode")
                    .selected_text(current_bm.label())
                    .width(70.0)
                    .show_ui(ui, |ui| {
                        for &mode in BrushMode::all() {
                            let _ = ui.selectable_label(mode == current_bm, mode.label());
                        }
                    });
            });
        }

        // Dynamics popup: Scatter, Color Jitter
        ui.separator();
        let dyn_active = self.properties.scatter > 0.01
            || self.properties.hue_jitter > 0.01
            || self.properties.brightness_jitter > 0.01;
        let dyn_popup_id = ui.make_persistent_id("brush_dyn_popup");
        let dyn_resp = assets.icon_button(ui, Icon::UiBrushDynamics, egui::Vec2::splat(20.0));
        if dyn_active {
            let dot = egui::pos2(dyn_resp.rect.max.x - 3.5, dyn_resp.rect.min.y + 3.5);
            ui.painter()
                .circle_filled(dot, 2.5, ui.visuals().hyperlink_color);
        }
        if dyn_resp.clicked() {
            ui.memory_mut(|m| m.toggle_popup(dyn_popup_id));
        }
        egui::popup_below_widget(ui, dyn_popup_id, &dyn_resp, |ui| {
            ui.set_min_width(220.0);
            egui::Grid::new("brush_dynamics_popup")
                .num_columns(2)
                .spacing([8.0, 4.0])
                .show(ui, |ui| {
                    ui.label("Scatter");
                    let mut scatter_pct = (self.properties.scatter * 200.0).round();
                    if ui
                        .add(
                            egui::Slider::new(&mut scatter_pct, 0.0..=200.0)
                                .suffix("%")
                                .max_decimals(0),
                        )
                        .changed()
                    {
                        self.properties.scatter = scatter_pct / 200.0;
                    }
                    ui.end_row();
                    ui.label("Hue Jitter");
                    let mut hj_pct = (self.properties.hue_jitter * 100.0).round();
                    if ui
                        .add(
                            egui::Slider::new(&mut hj_pct, 0.0..=100.0)
                                .suffix("%")
                                .max_decimals(0),
                        )
                        .changed()
                    {
                        self.properties.hue_jitter = hj_pct / 100.0;
                    }
                    ui.end_row();
                    ui.label("Brightness");
                    let mut bj_pct = (self.properties.brightness_jitter * 100.0).round();
                    if ui
                        .add(
                            egui::Slider::new(&mut bj_pct, 0.0..=100.0)
                                .suffix("%")
                                .max_decimals(0),
                        )
                        .changed()
                    {
                        self.properties.brightness_jitter = bj_pct / 100.0;
                    }
                    ui.end_row();
                });
        });

        // Spacing slider (only for image tips, not circle)
        if !self.properties.brush_tip.is_circle() && self.active_tool != Tool::Pencil {
            ui.separator();
            ui.label(t!("ctx.spacing"));
            let mut spacing_pct = (self.properties.spacing * 100.0).round();
            if ui
                .add(
                    egui::DragValue::new(&mut spacing_pct)
                        .speed(1.0)
                        .clamp_range(1.0..=200.0)
                        .suffix("%"),
                )
                .on_hover_text(t!("ctx.spacing_tooltip"))
                .changed()
            {
                self.properties.spacing = spacing_pct / 100.0;
            }

            // Rotation controls
            self.show_tip_rotation_controls(ui);
        }
    }

    /// Show rotation controls for non-circle brush tips.
    /// Provides a fixed-angle slider OR a random-range double-slider, plus a checkbox to toggle.
    fn show_tip_rotation_controls(&mut self, ui: &mut egui::Ui) {
        ui.separator();

        // Always show "Angle:" label first
        ui.label(t!("ctx.angle"));

        if self.properties.tip_random_rotation {
            // --- Random rotation mode: range controls ---
            let mut lo = self.properties.tip_rotation_range.0;
            let mut hi = self.properties.tip_rotation_range.1;

            // Min handle
            if ui
                .add(
                    egui::DragValue::new(&mut lo)
                        .speed(1.0)
                        .clamp_range(0.0..=360.0)
                        .suffix("°"),
                )
                .changed()
                && lo > hi
            {
                hi = lo;
            }

            // Painted range bar
            let bar_width = 100.0;
            let bar_height = 14.0;
            let (bar_rect, _bar_resp) = ui
                .allocate_exact_size(egui::Vec2::new(bar_width, bar_height), egui::Sense::hover());

            // Background track
            let track_color = ui.visuals().widgets.inactive.bg_fill;
            ui.painter().rect_filled(bar_rect, 3.0, track_color);

            // Highlighted portion
            let frac_lo = lo / 360.0;
            let frac_hi = hi / 360.0;
            let fill_left = bar_rect.left() + frac_lo * bar_width;
            let fill_right = bar_rect.left() + frac_hi * bar_width;
            if fill_right > fill_left + 1.0 {
                let fill_rect = egui::Rect::from_min_max(
                    egui::Pos2::new(fill_left, bar_rect.top()),
                    egui::Pos2::new(fill_right, bar_rect.bottom()),
                );
                let accent = ui.visuals().hyperlink_color;
                let fill_color =
                    egui::Color32::from_rgba_unmultiplied(accent.r(), accent.g(), accent.b(), 140);
                ui.painter().rect_filled(fill_rect, 3.0, fill_color);
            }

            // Handle markers at the edges of the range
            let handle_r = 4.0;
            let handle_color = ui.visuals().text_color();
            ui.painter().circle_filled(
                egui::Pos2::new(fill_left, bar_rect.center().y),
                handle_r,
                handle_color,
            );
            ui.painter().circle_filled(
                egui::Pos2::new(fill_right, bar_rect.center().y),
                handle_r,
                handle_color,
            );

            // Max handle
            if ui
                .add(
                    egui::DragValue::new(&mut hi)
                        .speed(1.0)
                        .clamp_range(0.0..=360.0)
                        .suffix("°"),
                )
                .changed()
                && hi < lo
            {
                lo = hi;
            }

            self.properties.tip_rotation_range = (lo, hi);
            // In random mode, the cursor doesn't show a specific rotation
            self.active_tip_rotation_deg = 0.0;
        } else {
            // --- Fixed rotation mode: single angle value ---
            ui.add(
                egui::DragValue::new(&mut self.properties.tip_rotation)
                    .speed(1.0)
                    .clamp_range(0.0..=359.0)
                    .suffix("°"),
            )
            .on_hover_text(t!("ctx.rotation_tooltip"));
            self.active_tip_rotation_deg = self.properties.tip_rotation;
        }

        // Random checkbox — to the right of the angle controls
        let rnd_resp = ui.selectable_label(self.properties.tip_random_rotation, t!("ctx.random"));
        if rnd_resp.clicked() {
            self.properties.tip_random_rotation = !self.properties.tip_random_rotation;
        }
        rnd_resp.on_hover_text(t!("ctx.random_tooltip"));
    }

    /// Show line tool-specific options (size, cap style, pattern, blend mode)
    fn show_line_options(&mut self, ui: &mut egui::Ui) {
        // Size
        ui.label(t!("ctx.size"));
        let size_text = format!("{:.0}px", self.properties.size);
        egui::ComboBox::from_id_source("ctx_line_size")
            .selected_text(&size_text)
            .width(65.0)
            .show_ui(ui, |ui| {
                for &preset in BRUSH_SIZE_PRESETS.iter() {
                    let label = format!("{:.0}px", preset);
                    if ui
                        .selectable_label((self.properties.size - preset).abs() < 0.1, &label)
                        .clicked()
                    {
                        self.properties.size = preset;
                    }
                }
            });
        ui.horizontal(|ui| {
            if ui.button("-").clicked() {
                self.properties.size = (self.properties.size - 1.0).max(1.0);
            }
            ui.add(
                egui::DragValue::new(&mut self.properties.size)
                    .speed(0.5)
                    .clamp_range(1.0..=256.0)
                    .suffix("px"),
            )
            .on_hover_text(t!("ctx.size_drag_tooltip"));
            if ui.button("+").clicked() {
                self.properties.size = (self.properties.size + 1.0).min(256.0);
            }
        });

        ui.separator();

        // Line Pattern
        ui.label(t!("ctx.pattern"));
        let current_pattern = self.line_state.line_tool.options.pattern;
        egui::ComboBox::from_id_source("ctx_line_pattern")
            .selected_text(current_pattern.label())
            .width(70.0)
            .show_ui(ui, |ui| {
                for &pattern in LinePattern::all() {
                    if ui
                        .selectable_label(pattern == current_pattern, pattern.label())
                        .clicked()
                    {
                        self.line_state.line_tool.options.pattern = pattern;
                    }
                }
            });

        ui.separator();

        // Blend Mode
        ui.label(t!("ctx.blend"));
        let current_mode = self.properties.blending_mode;
        egui::ComboBox::from_id_source("ctx_line_blend_mode")
            .selected_text(current_mode.name())
            .width(90.0)
            .show_ui(ui, |ui| {
                for &mode in BlendMode::all() {
                    if ui
                        .selectable_label(mode == current_mode, mode.name())
                        .clicked()
                    {
                        self.properties.blending_mode = mode;
                    }
                }
            });
    }

    /// Show eraser-specific options (size, hardness - opacity from color alpha)
    fn show_eraser_options(&mut self, ui: &mut egui::Ui, assets: &Assets) {
        // Brush tip picker
        self.show_brush_tip_picker(ui, assets);
        ui.separator();

        // Size
        ui.label(t!("ctx.size"));
        let size_text = format!("{:.0}px", self.properties.size);
        egui::ComboBox::from_id_source("ctx_eraser_size")
            .selected_text(&size_text)
            .width(65.0)
            .show_ui(ui, |ui| {
                for &preset in BRUSH_SIZE_PRESETS.iter() {
                    let label = format!("{:.0}px", preset);
                    if ui
                        .selectable_label((self.properties.size - preset).abs() < 0.1, &label)
                        .clicked()
                    {
                        self.properties.size = preset;
                    }
                }
            });
        ui.add(
            egui::DragValue::new(&mut self.properties.size)
                .speed(0.5)
                .clamp_range(1.0..=256.0)
                .suffix("px"),
        )
        .on_hover_text(t!("ctx.size_drag_tooltip"));

        ui.separator();

        // Hardness
        ui.label(t!("ctx.hardness"));
        let mut hardness_pct = (self.properties.hardness * 100.0).round();
        if ui
            .add(
                egui::DragValue::new(&mut hardness_pct)
                    .speed(1.0)
                    .clamp_range(0.0..=100.0)
                    .suffix("%"),
            )
            .on_hover_text(t!("ctx.hardness_eraser_tooltip"))
            .changed()
        {
            self.properties.hardness = hardness_pct / 100.0;
        }

        ui.separator();

        // Anti-aliasing toggle (compact "AA" matching Text tool)
        let aa_resp = ui.selectable_label(self.properties.anti_aliased, t!("ctx.anti_alias"));
        if aa_resp.clicked() {
            self.properties.anti_aliased = !self.properties.anti_aliased;
        }
        aa_resp.on_hover_text(t!("ctx.anti_alias_tooltip"));

        // Dynamics popup for eraser: Scatter only
        ui.separator();
        let dyn_active_eraser = self.properties.scatter > 0.01;
        let dyn_popup_id_e = ui.make_persistent_id("eraser_dyn_popup");
        let dyn_resp_e = assets.icon_button(ui, Icon::UiBrushDynamics, egui::Vec2::splat(20.0));
        if dyn_active_eraser {
            let dot_e = egui::pos2(dyn_resp_e.rect.max.x - 3.5, dyn_resp_e.rect.min.y + 3.5);
            ui.painter()
                .circle_filled(dot_e, 2.5, ui.visuals().hyperlink_color);
        }
        if dyn_resp_e.clicked() {
            ui.memory_mut(|m| m.toggle_popup(dyn_popup_id_e));
        }
        egui::popup_below_widget(ui, dyn_popup_id_e, &dyn_resp_e, |ui| {
            ui.set_min_width(180.0);
            egui::Grid::new("eraser_dynamics_popup")
                .num_columns(2)
                .spacing([8.0, 4.0])
                .show(ui, |ui| {
                    ui.label("Scatter");
                    let mut scatter_pct = (self.properties.scatter * 200.0).round();
                    if ui
                        .add(
                            egui::Slider::new(&mut scatter_pct, 0.0..=200.0)
                                .suffix("%")
                                .max_decimals(0),
                        )
                        .changed()
                    {
                        self.properties.scatter = scatter_pct / 200.0;
                    }
                    ui.end_row();
                });
        });

        ui.separator();

        // Show current eraser opacity from color alpha
        let opacity_pct = (self.properties.color.a() as f32 / 255.0 * 100.0).round();
        ui.label(t!("ctx.opacity_label").replace("{0}", &format!("{:.0}", opacity_pct)))
            .on_hover_text(t!("ctx.eraser_opacity_tooltip"));

        // Spacing slider (only for image tips)
        if !self.properties.brush_tip.is_circle() {
            ui.separator();
            ui.label(t!("ctx.spacing"));
            let mut spacing_pct = (self.properties.spacing * 100.0).round();
            if ui
                .add(
                    egui::DragValue::new(&mut spacing_pct)
                        .speed(1.0)
                        .clamp_range(1.0..=200.0)
                        .suffix("%"),
                )
                .on_hover_text(t!("ctx.spacing_tooltip"))
                .changed()
            {
                self.properties.spacing = spacing_pct / 100.0;
            }

            // Rotation controls (shared helper)
            self.show_tip_rotation_controls(ui);
        }
    }

    /// Show selection-tool-specific options (mode dropdown)
    fn show_selection_options(&mut self, ui: &mut egui::Ui) {
        ui.label(t!("ctx.mode"));
        let current = self.selection_state.mode;
        egui::ComboBox::from_id_source("ctx_sel_mode")
            .selected_text(current.label())
            .width(90.0)
            .show_ui(ui, |ui| {
                for &mode in SelectionMode::all() {
                    if ui.selectable_label(mode == current, mode.label()).clicked() {
                        self.selection_state.mode = mode;
                    }
                }
            });

        ui.separator();

        self.show_sel_modify_controls(ui);

        ui.separator();
        ui.label(t!("ctx.selection_hint"));
    }

    /// Show lasso-tool-specific options (mode dropdown)
    fn show_lasso_options(&mut self, ui: &mut egui::Ui) {
        ui.label(t!("ctx.mode"));
        let current = self.lasso_state.mode;
        egui::ComboBox::from_id_source("ctx_lasso_mode")
            .selected_text(current.label())
            .width(90.0)
            .show_ui(ui, |ui| {
                for &mode in SelectionMode::all() {
                    if ui.selectable_label(mode == current, mode.label()).clicked() {
                        self.lasso_state.mode = mode;
                    }
                }
            });
        ui.separator();
        self.show_sel_modify_controls(ui);
        ui.separator();
        ui.label(t!("ctx.lasso_hint"));
    }

    /// Inline Feather / Expand / Contract controls for all selection tool context bars.
    fn show_sel_modify_controls(&mut self, ui: &mut egui::Ui) {
        ui.label("Modify:");
        ui.add(
            egui::DragValue::new(&mut self.sel_modify_radius)
                .clamp_range(1.0..=200.0)
                .speed(0.5)
                .suffix("px")
                .max_decimals(0),
        )
        .on_hover_text("Radius in pixels for Feather / Expand / Contract");
        if ui
            .button("Feather")
            .on_hover_text("Blur (soften) selection edge by radius")
            .clicked()
        {
            self.pending_sel_modify = Some(SelectionModifyOp::Feather(self.sel_modify_radius));
        }
        if ui
            .button("Expand")
            .on_hover_text("Grow selection by radius")
            .clicked()
        {
            self.pending_sel_modify = Some(SelectionModifyOp::Expand(
                self.sel_modify_radius.round() as i32,
            ));
        }
        if ui
            .button("Contract")
            .on_hover_text("Shrink selection by radius")
            .clicked()
        {
            self.pending_sel_modify = Some(SelectionModifyOp::Contract(
                self.sel_modify_radius.round() as i32,
            ));
        }
    }

    /// Show perspective crop options
    fn show_perspective_crop_options(&mut self, ui: &mut egui::Ui) {
        if self.perspective_crop_state.active {
            ui.label(t!("ctx.perspective_crop_active"));
            if ui.button(t!("ctx.apply")).clicked() {
                // Set a flag — actual cropping happens in handle_input
                // which has canvas_state access
            }
            if ui.button(t!("ctx.cancel")).clicked() {
                self.perspective_crop_state.active = false;
            }
        } else {
            ui.label(t!("ctx.perspective_crop_inactive"));
        }
    }

    fn show_clone_stamp_options(&mut self, ui: &mut egui::Ui) {
        // Size
        ui.label(t!("ctx.size"));
        ui.add(
            egui::DragValue::new(&mut self.properties.size)
                .clamp_range(1.0..=256.0)
                .speed(0.5)
                .suffix("px"),
        );
        ui.separator();

        // Hardness
        ui.label(t!("ctx.hardness"));
        let mut hardness_pct = self.properties.hardness * 100.0;
        if ui
            .add(
                egui::DragValue::new(&mut hardness_pct)
                    .clamp_range(0.0..=100.0)
                    .speed(0.5)
                    .suffix("%"),
            )
            .changed()
        {
            self.properties.hardness = hardness_pct / 100.0;
        }
        ui.separator();

        // Source indicator
        if let Some(src) = self.clone_stamp_state.source {
            ui.label(
                t!("ctx.clone_stamp.source")
                    .replace("{0}", &format!("{:.0}", src.x))
                    .replace("{1}", &format!("{:.0}", src.y)),
            );
        } else {
            ui.label(t!("ctx.clone_stamp.set_source"));
        }
    }

    fn show_content_aware_options(&mut self, ui: &mut egui::Ui) {
        use crate::ops::inpaint::ContentAwareQuality;

        // Size
        ui.label(t!("ctx.size"));
        ui.add(
            egui::DragValue::new(&mut self.properties.size)
                .clamp_range(1.0..=256.0)
                .speed(0.5)
                .suffix("px"),
        );
        ui.separator();

        // Hardness
        ui.label(t!("ctx.hardness"));
        let mut hardness_pct = self.properties.hardness * 100.0;
        if ui
            .add(
                egui::DragValue::new(&mut hardness_pct)
                    .clamp_range(0.0..=100.0)
                    .speed(0.5)
                    .suffix("%"),
            )
            .changed()
        {
            self.properties.hardness = hardness_pct / 100.0;
        }
        ui.separator();

        // Quality dropdown
        ui.label("Quality:");
        let cur_q = self.content_aware_state.quality;
        egui::ComboBox::from_id_source("ca_quality")
            .selected_text(cur_q.label())
            .width(100.0)
            .show_ui(ui, |ui| {
                for &q in ContentAwareQuality::all() {
                    if ui.selectable_label(q == cur_q, q.label()).clicked() {
                        self.content_aware_state.quality = q;
                    }
                }
            });
        ui.separator();

        // Sample radius (Instant only)
        if cur_q == ContentAwareQuality::Instant {
            ui.label(t!("ctx.content_aware.sample"));
            ui.add(
                egui::DragValue::new(&mut self.content_aware_state.sample_radius)
                    .clamp_range(10.0..=150.0)
                    .speed(0.5)
                    .suffix("px"),
            );
            ui.separator();
        }

        // Patch size (Balanced / HQ only)
        if cur_q.is_async() {
            ui.label("Patch:");
            ui.add(
                egui::DragValue::new(&mut self.content_aware_state.patch_size)
                    .clamp_range(3_u32..=11_u32)
                    .speed(0.5)
                    .suffix("px"),
            );
            // Keep patch_size odd
            if self.content_aware_state.patch_size.is_multiple_of(2) {
                self.content_aware_state.patch_size += 1;
            }
            ui.separator();
        }

        if cur_q == ContentAwareQuality::Instant {
            ui.label(t!("ctx.content_aware.hint"));
        } else {
            ui.label("Paint to preview, then release to run inpaint.");
        }
    }

    /// Draw a custom tolerance slider with +/- buttons, wide track, and vertical handle.
    /// Returns `Some(new_value)` if changed, `None` otherwise.
    fn tolerance_slider(ui: &mut egui::Ui, id_salt: &str, value: f32) -> Option<f32> {
        let mut new_value = value;
        let mut changed = false;

        let vis = ui.visuals().clone();
        let is_dark = vis.dark_mode;

        // Colors that work in both dark and light modes
        let track_bg = if is_dark {
            Color32::from_gray(50)
        } else {
            Color32::from_gray(190)
        };
        // Use theme accent color for the filled portion
        let track_fill = vis.selection.bg_fill;
        let handle_color = if is_dark {
            Color32::from_gray(220)
        } else {
            Color32::from_gray(255)
        };
        let handle_border = if is_dark {
            Color32::from_gray(140)
        } else {
            Color32::from_gray(80)
        };
        let btn_bg = if is_dark {
            Color32::from_gray(60)
        } else {
            Color32::from_gray(210)
        };
        let btn_hover = if is_dark {
            Color32::from_gray(80)
        } else {
            Color32::from_gray(225)
        };
        let btn_text = if is_dark {
            Color32::from_gray(220)
        } else {
            Color32::from_gray(30)
        };
        let value_text_color = if is_dark {
            Color32::from_gray(200)
        } else {
            Color32::from_gray(40)
        };

        ui.horizontal(|ui| {
            // Minus button
            let btn_size = egui::vec2(20.0, 20.0);
            let (minus_rect, minus_resp) = ui.allocate_exact_size(btn_size, egui::Sense::click());
            let minus_bg = if minus_resp.hovered() {
                btn_hover
            } else {
                btn_bg
            };
            ui.painter().rect_filled(minus_rect, 3.0, minus_bg);
            ui.painter().text(
                minus_rect.center(),
                egui::Align2::CENTER_CENTER,
                "−",
                egui::FontId::proportional(14.0),
                btn_text,
            );
            if minus_resp.clicked() {
                new_value = (new_value - 1.0).max(0.0);
                changed = true;
            }

            ui.add_space(4.0);

            // Slider track
            let slider_width = 140.0;
            let slider_height = 20.0;
            let (slider_rect, slider_resp) = ui.allocate_exact_size(
                egui::vec2(slider_width, slider_height),
                egui::Sense::click_and_drag(),
            );

            let rounding = slider_height / 2.0;

            // Draw track background
            ui.painter().rect_filled(slider_rect, rounding, track_bg);

            // Draw filled portion
            let fill_fraction = new_value / 100.0;
            let fill_width = slider_rect.width() * fill_fraction;
            if fill_width > 0.5 {
                let fill_rect =
                    Rect::from_min_size(slider_rect.min, egui::vec2(fill_width, slider_height));
                ui.painter().rect_filled(fill_rect, rounding, track_fill);
            }

            // Draw vertical handle
            let handle_x = slider_rect.min.x + fill_width;
            let handle_width = 4.0;
            let handle_height = slider_height + 2.0;
            let handle_rect = Rect::from_center_size(
                egui::pos2(handle_x, slider_rect.center().y),
                egui::vec2(handle_width, handle_height),
            );
            ui.painter().rect_filled(handle_rect, 2.0, handle_color);
            ui.painter()
                .rect_stroke(handle_rect, 2.0, (1.0, handle_border));

            // Draw value text centered on track
            let text = format!("{}%", new_value.round() as i32);
            ui.painter().text(
                slider_rect.center(),
                egui::Align2::CENTER_CENTER,
                &text,
                egui::FontId::proportional(11.0),
                value_text_color,
            );

            // Handle interaction
            if (slider_resp.dragged() || slider_resp.clicked())
                && let Some(pos) = slider_resp.interact_pointer_pos()
            {
                let frac = ((pos.x - slider_rect.min.x) / slider_rect.width()).clamp(0.0, 1.0);
                new_value = (frac * 100.0).round();
                changed = true;
            }

            // Tooltip
            if slider_resp.hovered() {
                slider_resp.on_hover_text(t!("ctx.tolerance_tooltip"));
            }

            ui.add_space(4.0);

            // Plus button
            let (plus_rect, plus_resp) = ui.allocate_exact_size(btn_size, egui::Sense::click());
            let plus_bg = if plus_resp.hovered() {
                btn_hover
            } else {
                btn_bg
            };
            ui.painter().rect_filled(plus_rect, 3.0, plus_bg);
            ui.painter().text(
                plus_rect.center(),
                egui::Align2::CENTER_CENTER,
                "+",
                egui::FontId::proportional(14.0),
                btn_text,
            );
            if plus_resp.clicked() {
                new_value = (new_value + 1.0).min(100.0);
                changed = true;
            }
        });

        // Ensure we allocate a unique id for internal egui state
        let _ = ui.id().with(id_salt);

        if changed { Some(new_value) } else { None }
    }

    fn show_magic_wand_options(&mut self, ui: &mut egui::Ui) {
        ui.label(t!("ctx.tolerance"));
        if let Some(new_val) = Self::tolerance_slider(ui, "mw_tol", self.magic_wand_state.tolerance)
        {
            self.magic_wand_state.tolerance = new_val;
            self.magic_wand_state.tolerance_changed_at = Some(Instant::now());
            self.magic_wand_state.recalc_pending = true;
            ui.ctx().request_repaint();
        }

        ui.separator();

        ui.label(t!("ctx.mode"));
        let current = self.selection_state.mode;
        egui::ComboBox::from_id_source("ctx_magic_wand_mode")
            .selected_text(current.label())
            .width(90.0)
            .show_ui(ui, |ui| {
                for &mode in SelectionMode::all() {
                    if ui.selectable_label(mode == current, mode.label()).clicked() {
                        self.selection_state.mode = mode;
                    }
                }
            });

        ui.separator();
        self.show_sel_modify_controls(ui);
    }

    fn show_fill_options(&mut self, ui: &mut egui::Ui) {
        ui.label(t!("ctx.tolerance"));
        if let Some(new_val) = Self::tolerance_slider(ui, "fill_tol", self.fill_state.tolerance) {
            self.fill_state.tolerance = new_val;
            self.fill_state.tolerance_changed_at = Some(Instant::now());
            self.fill_state.recalc_pending = true;
            ui.ctx().request_repaint();
        }

        ui.separator();

        let old_aa = self.fill_state.anti_aliased;
        let aa_resp = ui.selectable_label(self.fill_state.anti_aliased, t!("ctx.anti_alias"));
        if aa_resp.clicked() {
            self.fill_state.anti_aliased = !self.fill_state.anti_aliased;
        }
        aa_resp.on_hover_text(t!("ctx.anti_alias_tooltip"));
        if self.fill_state.anti_aliased != old_aa {
            // Anti-alias setting changed, trigger preview refresh
            self.fill_state.tolerance_changed_at = Some(Instant::now());
            self.fill_state.recalc_pending = true;
            ui.ctx().request_repaint();
        }
    }

    pub fn show_properties_toolbar(&mut self, ui: &mut egui::Ui, _assets: &Assets) {
        // Size ComboBox with presets
        ui.label(t!("ctx.size"));

        let size_text = format!("{:.0}px", self.properties.size);
        egui::ComboBox::from_id_source("brush_size_presets")
            .selected_text(&size_text)
            .width(65.0)
            .show_ui(ui, |ui| {
                for &preset in BRUSH_SIZE_PRESETS.iter() {
                    let label = format!("{:.0}px", preset);
                    if ui
                        .selectable_label((self.properties.size - preset).abs() < 0.1, &label)
                        .clicked()
                    {
                        self.properties.size = preset;
                    }
                }
            });

        // Fine-tune with drag value
        ui.add(
            egui::DragValue::new(&mut self.properties.size)
                .speed(0.5)
                .clamp_range(1.0..=256.0)
                .suffix("px"),
        )
        .on_hover_text(t!("ctx.size_drag_tooltip"));

        ui.separator();

        // Hardness as percentage
        ui.label(t!("ctx.hardness"));
        let mut hardness_pct = (self.properties.hardness * 100.0).round();
        if ui
            .add(
                egui::DragValue::new(&mut hardness_pct)
                    .speed(1.0)
                    .clamp_range(0.0..=100.0)
                    .suffix("%"),
            )
            .on_hover_text(t!("ctx.hardness_brush_tooltip"))
            .changed()
        {
            self.properties.hardness = hardness_pct / 100.0;
        }

        ui.separator();

        // Blend Mode Dropdown (for brush painting)
        ui.label(t!("ctx.blend"));
        let current_mode = self.properties.blending_mode;
        egui::ComboBox::from_id_source("tool_blend_mode")
            .selected_text(current_mode.name())
            .width(90.0)
            .show_ui(ui, |ui: &mut egui::Ui| {
                for &mode in BlendMode::all() {
                    if ui
                        .selectable_label(mode == current_mode, mode.name())
                        .clicked()
                    {
                        self.properties.blending_mode = mode;
                    }
                }
            });
    }

    pub fn handle_input(
        &mut self,
        ui: &egui::Ui,
        canvas_state: &mut CanvasState,
        canvas_pos: Option<(u32, u32)>,
        canvas_pos_f32: Option<(f32, f32)>,
        canvas_pos_f32_clamped: Option<(f32, f32)>,
        raw_motion_events: &[(f32, f32)],
        painter: &egui::Painter,
        canvas_rect: Rect,
        zoom: f32,
        primary_color_f32: [f32; 4],
        secondary_color_f32: [f32; 4],
        mut gpu_renderer: Option<&mut crate::gpu::GpuRenderer>,
    ) {
        let mut stroke_event: Option<StrokeEvent> = None;

        // -- Auto-commit on layer change ---------------------------
        // If the active layer index or layer count changed (user clicked a
        // different layer, reordered, deleted, etc.), commit any tool that
        // has uncommitted state — mirrors the auto-commit-on-tool-switch
        // logic below.
        let layer_changed = canvas_state.active_layer_index != self.last_tracked_layer_index
            || canvas_state.layers.len() != self.last_tracked_layer_count;
        if layer_changed {
            self.last_tracked_layer_index = canvas_state.active_layer_index;
            self.last_tracked_layer_count = canvas_state.layers.len();

            // Line
            if self.active_tool == Tool::Line
                && self.line_state.line_tool.stage == LineStage::Editing
            {
                if let Some(final_bounds) = self.line_state.line_tool.last_bounds {
                    self.stroke_tracker.expand_bounds(final_bounds);
                }
                let auto_commit_event = self.stroke_tracker.finish(canvas_state);
                let mirror_bounds = canvas_state.mirror_preview_layer();
                self.commit_bezier_to_layer(canvas_state, secondary_color_f32);
                let base_dirty = self.line_state.line_tool.last_bounds;
                let combined = match (base_dirty, mirror_bounds) {
                    (Some(b), Some(m)) => Some(b.union(m)),
                    (Some(b), None) => Some(b),
                    (None, Some(m)) => Some(m),
                    (None, None) => None,
                };
                if let Some(dirty) = combined {
                    canvas_state.mark_dirty(Some(dirty));
                } else {
                    self.mark_full_dirty(canvas_state);
                }
                canvas_state.clear_preview_state();
                self.line_state.line_tool.stage = LineStage::Idle;
                self.line_state.line_tool.last_bounds = None;
                stroke_event = auto_commit_event;
            }
            // Fill
            if self.active_tool == Tool::Fill && self.fill_state.active_fill.is_some() {
                self.commit_fill_preview(canvas_state);
                canvas_state.clear_preview_state();
                self.fill_state.active_fill = None;
                self.fill_state.fill_color_u8 = None;
                canvas_state.mark_dirty(None);
            }
            // Gradient
            if self.active_tool == Tool::Gradient && self.gradient_state.drag_start.is_some() {
                self.gradient_state.commit_pending = true;
                self.gradient_state.commit_pending_frame = 0;
            }
            // Text
            if self.active_tool == Tool::Text && self.text_state.is_editing {
                if !self.text_state.text.is_empty() {
                    self.stroke_tracker
                        .start_preview_tool(canvas_state.active_layer_index, "Text");
                    self.text_state.commit_pending = true;
                    self.text_state.commit_pending_frame = 0;
                } else {
                    self.text_state.is_editing = false;
                    self.text_state.origin = None;
                    canvas_state.clear_preview_state();
                }
            }
            // Liquify
            if self.active_tool == Tool::Liquify && self.liquify_state.is_active {
                self.liquify_state.commit_pending = true;
                self.liquify_state.commit_pending_frame = 0;
            }
            // Mesh Warp
            if self.active_tool == Tool::MeshWarp && self.mesh_warp_state.is_active {
                self.mesh_warp_state.commit_pending = true;
                self.mesh_warp_state.commit_pending_frame = 0;
            }
            // Shape
            if self.active_tool == Tool::Shapes && self.shapes_state.placed.is_some() {
                self.commit_shape(canvas_state);
            }
        }

        // -- Deferred gradient commit --
        // When commit_pending is true, we defer the actual commit by one
        // frame so the loading bar ("Committing: Gradient") has a chance
        // to render before the synchronous work blocks the UI thread.
        if self.gradient_state.commit_pending {
            if self.gradient_state.commit_pending_frame == 0 {
                // Frame 0: loading bar will render; advance counter.
                self.gradient_state.commit_pending_frame = 1;
                ui.ctx().request_repaint();
                return;
            } else {
                // Frame 1+: actually commit now (loading bar is visible).
                self.gradient_state.commit_pending = false;
                self.gradient_state.commit_pending_frame = 0;
                self.commit_gradient(canvas_state);
                ui.ctx().request_repaint();
            }
        }

        // -- Deferred text commit --
        if self.text_state.commit_pending {
            if self.text_state.commit_pending_frame == 0 {
                self.text_state.commit_pending_frame = 1;
                ui.ctx().request_repaint();
                return;
            } else {
                self.text_state.commit_pending = false;
                self.text_state.commit_pending_frame = 0;
                self.commit_text(canvas_state);
                ui.ctx().request_repaint();
            }
        }

        // -- Deferred liquify commit --
        if self.liquify_state.commit_pending {
            if self.liquify_state.commit_pending_frame == 0 {
                self.liquify_state.commit_pending_frame = 1;
                ui.ctx().request_repaint();
                return;
            } else {
                self.liquify_state.commit_pending = false;
                self.liquify_state.commit_pending_frame = 0;
                self.commit_liquify(canvas_state);
                ui.ctx().request_repaint();
            }
        }

        // -- Deferred mesh warp commit --
        if self.mesh_warp_state.commit_pending {
            if self.mesh_warp_state.commit_pending_frame == 0 {
                self.mesh_warp_state.commit_pending_frame = 1;
                ui.ctx().request_repaint();
                return;
            } else {
                self.mesh_warp_state.commit_pending = false;
                self.mesh_warp_state.commit_pending_frame = 0;
                self.commit_mesh_warp(canvas_state);
                ui.ctx().request_repaint();
            }
        }

        // Auto-commit Bézier line if tool changed away from Line tool
        if self.active_tool != Tool::Line && self.line_state.line_tool.stage == LineStage::Editing {
            // Capture "before" for undo (layer still unchanged, line is in preview)
            if let Some(final_bounds) = self.line_state.line_tool.last_bounds {
                self.stroke_tracker.expand_bounds(final_bounds);
            }
            let auto_commit_event = self.stroke_tracker.finish(canvas_state);

            let mirror_bounds = canvas_state.mirror_preview_layer();
            self.commit_bezier_to_layer(canvas_state, secondary_color_f32);

            // Mark dirty: combine original line bounds with mirrored bounds
            let base_dirty = self.line_state.line_tool.last_bounds;
            let combined = match (base_dirty, mirror_bounds) {
                (Some(b), Some(m)) => Some(b.union(m)),
                (Some(b), None) => Some(b),
                (None, Some(m)) => Some(m),
                (None, None) => None,
            };
            if let Some(dirty) = combined {
                canvas_state.mark_dirty(Some(dirty));
            } else {
                self.mark_full_dirty(canvas_state);
            }

            canvas_state.clear_preview_state();
            self.line_state.line_tool.stage = LineStage::Idle;
            self.line_state.line_tool.last_bounds = None; // Reset bounds

            // Store the event for pickup by app.rs
            stroke_event = auto_commit_event;
        }

        // Auto-commit Fill if tool changed away from Fill tool
        if self.active_tool != Tool::Fill && self.fill_state.active_fill.is_some() {
            self.commit_fill_preview(canvas_state);
            canvas_state.clear_preview_state();
            self.fill_state.active_fill = None;
            self.fill_state.fill_color_u8 = None;
            canvas_state.mark_dirty(None);
        }

        // Auto-commit Magic Wand if tool changed away from Magic Wand tool
        if self.active_tool != Tool::MagicWand && self.magic_wand_state.active_selection.is_some() {
            // Selection is already in canvas_state.selection_mask, just clear state
            self.magic_wand_state.active_selection = None;
            canvas_state.mark_dirty(None);
        }

        // Auto-commit Gradient if tool changed away from Gradient tool
        if self.active_tool != Tool::Gradient && self.gradient_state.drag_start.is_some() {
            self.gradient_state.commit_pending = true;
            self.gradient_state.commit_pending_frame = 0;
        }

        // Auto-commit Text if tool changed away
        if self.active_tool != Tool::Text && self.text_state.is_editing {
            if !self.text_state.text.is_empty() {
                self.stroke_tracker
                    .start_preview_tool(canvas_state.active_layer_index, "Text");
                self.commit_text(canvas_state);
            } else {
                self.text_state.is_editing = false;
                self.text_state.origin = None;
                canvas_state.clear_preview_state();
            }
        }

        // Auto-commit Liquify if tool changed away
        if self.active_tool != Tool::Liquify && self.liquify_state.is_active {
            self.liquify_state.commit_pending = true;
            self.liquify_state.commit_pending_frame = 0;
        }

        // Auto-commit Mesh Warp if tool changed away
        if self.active_tool != Tool::MeshWarp && self.mesh_warp_state.is_active {
            self.mesh_warp_state.commit_pending = true;
            self.mesh_warp_state.commit_pending_frame = 0;
        }

        // Auto-commit Shape if tool changed away
        if self.active_tool != Tool::Shapes && self.shapes_state.placed.is_some() {
            self.commit_shape(canvas_state);
        }

        let is_primary_down = ui.input(|i| i.pointer.primary_down());
        let is_primary_released = ui.input(|i| i.pointer.primary_released());
        let is_primary_clicked = ui.input(|i| i.pointer.primary_clicked());
        let is_primary_pressed = ui.input(|i| i.pointer.primary_pressed()); // Just pressed this frame
        let is_secondary_down = ui.input(|i| i.pointer.secondary_down());
        let is_secondary_released = ui.input(|i| i.pointer.secondary_released());
        let is_secondary_clicked = ui.input(|i| i.pointer.secondary_clicked());
        let shift_held = ui.input(|i| i.modifiers.shift);
        let enter_pressed = ui.input(|i| i.key_pressed(egui::Key::Enter));

        match self.active_tool {
            Tool::Brush | Tool::Eraser | Tool::Pencil => {
                let is_painting = is_primary_down || is_secondary_down;

                // TASK 2: Shift+Click straight line
                // Trigger when mouse is pressed (not dragged) with Shift held
                if is_primary_pressed
                    && shift_held
                    && self.tool_state.last_brush_pos.is_some()
                    && let (Some(last), Some(current)) =
                        (self.tool_state.last_brush_pos, canvas_pos)
                {
                    let is_eraser = self.active_tool == Tool::Eraser;

                    // Start stroke tracking for undo/redo BEFORE any modifications
                    if is_eraser {
                        // Eraser modifies layer directly - snapshot the layer NOW
                        if let Some(layer) =
                            canvas_state.layers.get(canvas_state.active_layer_index)
                        {
                            self.stroke_tracker.start_direct_tool(
                                canvas_state.active_layer_index,
                                "Eraser Line",
                                &layer.pixels,
                            );
                        }
                    } else {
                        // Brush/Pencil uses preview layer - we'll capture before right before commit
                        let description = if self.active_tool == Tool::Pencil {
                            "Pencil Line"
                        } else {
                            "Brush Line"
                        };
                        self.stroke_tracker
                            .start_preview_tool(canvas_state.active_layer_index, description);
                    }

                    // For Brush/Pencil: Initialize/clear preview layer
                    if !is_eraser {
                        if canvas_state.preview_layer.is_none()
                            || canvas_state.preview_layer.as_ref().unwrap().width()
                                != canvas_state.width
                            || canvas_state.preview_layer.as_ref().unwrap().height()
                                != canvas_state.height
                        {
                            canvas_state.preview_layer =
                                Some(TiledImage::new(canvas_state.width, canvas_state.height));
                        } else {
                            // Clear existing preview layer
                            if let Some(ref mut preview) = canvas_state.preview_layer {
                                preview.clear();
                            }
                        }
                        // Set preview blend mode to match tool's blending mode
                        canvas_state.preview_blend_mode = self.properties.blending_mode;
                    }

                    // Draw straight line (convert integer positions to float)
                    // With mirror support
                    let mirror = canvas_state.mirror_mode;
                    let mw = canvas_state.width;
                    let mh = canvas_state.height;
                    let start_f = (last.0 as f32, last.1 as f32);
                    let end_f = (current.0 as f32, current.1 as f32);
                    let start_mirrors = mirror.mirror_positions(start_f.0, start_f.1, mw, mh);
                    let end_mirrors = mirror.mirror_positions(end_f.0, end_f.1, mw, mh);
                    let mut modified_rect = Rect::NOTHING;
                    for i in 0..end_mirrors.len {
                        let s = start_mirrors.data[i];
                        let e = end_mirrors.data[i];
                        let r = if self.active_tool == Tool::Pencil {
                            self.draw_pixel_line_and_get_bounds(
                                canvas_state,
                                s,
                                e,
                                false,
                                primary_color_f32,
                                secondary_color_f32,
                            )
                        } else {
                            self.draw_line_and_get_bounds(
                                canvas_state,
                                s,
                                e,
                                is_eraser,
                                false,
                                primary_color_f32,
                                secondary_color_f32,
                            )
                        };
                        modified_rect = modified_rect.union(r);
                    }

                    // Track the line bounds for undo
                    self.stroke_tracker.expand_bounds(modified_rect);

                    // For Brush/Pencil: Capture "before" then commit preview layer to active layer
                    if !is_eraser {
                        // Finish stroke tracking BEFORE commit - captures "before" from unchanged layer
                        stroke_event = self.stroke_tracker.finish(canvas_state);

                        self.commit_bezier_to_layer(canvas_state, primary_color_f32);
                        canvas_state.clear_preview_state();
                        // Mark only stroke bounds dirty (not full canvas)
                        if let Some(ref ev) = stroke_event {
                            canvas_state.mark_dirty(Some(ev.bounds.expand(12.0)));
                        } else {
                            self.mark_full_dirty(canvas_state);
                        }
                    } else {
                        // Eraser: Layer was already modified, finish with saved snapshot
                        stroke_event = self.stroke_tracker.finish(canvas_state);
                        canvas_state.mark_dirty(Some(modified_rect));
                    }

                    // Store stroke event for app.rs to pick up (before returning)
                    if stroke_event.is_some() {
                        self.pending_stroke_event = stroke_event;
                    }

                    self.tool_state.last_brush_pos = Some(current);
                    self.tool_state.last_precise_pos =
                        Some(Pos2::new(current.0 as f32, current.1 as f32));
                    ui.ctx().request_repaint();
                    return; // Don't process as normal painting
                }

                if is_painting {
                    // Use float position for smooth sub-pixel drawing; fall back
                    // to integer pos converted to float if unavailable.
                    let current_f32 =
                        canvas_pos_f32.or_else(|| canvas_pos.map(|(x, y)| (x as f32, y as f32)));
                    if let (Some(cf), Some(current_pos)) = (current_f32, canvas_pos) {
                        // Initialize on first paint
                        if self.tool_state.last_pos.is_none() {
                            self.tool_state.using_secondary_color = is_secondary_down;
                            self.tool_state.last_precise_pos = Some(Pos2::new(cf.0, cf.1));
                            self.tool_state.distance_remainder = 0.0;
                            self.tool_state.smooth_pos = Some(Pos2::new(cf.0, cf.1));

                            // Start stroke tracking for Undo/Redo
                            let is_eraser = self.active_tool == Tool::Eraser;
                            if is_eraser {
                                // Eraser uses preview layer as an erase mask — capture before at commit time
                                self.stroke_tracker.start_preview_tool(
                                    canvas_state.active_layer_index,
                                    "Eraser Stroke",
                                );
                            } else {
                                // Brush/Pencil uses preview layer - we'll capture before right before commit
                                let description = if self.active_tool == Tool::Pencil {
                                    "Pencil Stroke"
                                } else {
                                    "Brush Stroke"
                                };
                                self.stroke_tracker.start_preview_tool(
                                    canvas_state.active_layer_index,
                                    description,
                                );
                            }

                            // Initialize/clear preview layer for stroke compositing (all tools: Brush, Pencil, Eraser)
                            if canvas_state.preview_layer.is_none()
                                || canvas_state.preview_layer.as_ref().unwrap().width()
                                    != canvas_state.width
                                || canvas_state.preview_layer.as_ref().unwrap().height()
                                    != canvas_state.height
                            {
                                canvas_state.preview_layer =
                                    Some(TiledImage::new(canvas_state.width, canvas_state.height));
                            } else {
                                // Clear existing preview layer
                                if let Some(ref mut preview) = canvas_state.preview_layer {
                                    preview.clear();
                                }
                            }
                            if is_eraser {
                                // Mark preview as eraser mask mode
                                canvas_state.preview_is_eraser = true;
                                canvas_state.preview_force_composite = true;
                            } else {
                                // Set preview blend mode to match tool's blending mode
                                canvas_state.preview_blend_mode = self.properties.blending_mode;
                            }
                        }

                        let is_eraser = self.active_tool == Tool::Eraser;
                        let use_secondary = self.tool_state.using_secondary_color;

                        // ============================================================
                        // SUB-FRAME EVENT PROCESSING
                        // Process ALL PointerMoved events that occurred since last
                        // frame. At 21 FPS with 1000Hz mouse, this captures ~48
                        // intermediate positions per frame instead of just 1.
                        // All painting is CPU-only; GPU upload happens ONCE at the end.
                        // ============================================================

                        // Build the list of positions to paint through this frame.
                        // Use raw_motion_events if available, otherwise fall back
                        // to the single current position.
                        let positions: Vec<(f32, f32)> = if !raw_motion_events.is_empty() {
                            raw_motion_events.to_vec()
                        } else {
                            vec![cf]
                        };

                        // ============================================================
                        // SPEED-ADAPTIVE EMA SMOOTHING
                        // Applies an exponential moving average to each raw
                        // mouse position before painting.  This directly rounds
                        // off angular corners caused by straight-line segments
                        // between sparse/distant mouse samples.
                        //
                        // Speed-adaptive alpha:
                        //   Close movement (< 1.5 px) → alpha = 1.0  (raw, precise)
                        //   Far movement   (> ~20 px) → alpha ≈ 0.55 (strong smoothing)
                        //
                        // At 1000 Hz sub-frame input the per-sample distance is
                        // small, so the smoothing is gentle — but it accumulates
                        // across several consecutive direction changes, naturally
                        // rounding corners.  At frame-rate input (big jumps) the
                        // smoothing is stronger, eliminating visible polygon edges.
                        // ============================================================
                        let smoothed_positions: Vec<(f32, f32)> = {
                            let mut result = Vec::with_capacity(positions.len());
                            for &pos in &positions {
                                let raw = Pos2::new(pos.0, pos.1);
                                let smoothed = if let Some(prev) = self.tool_state.smooth_pos {
                                    let dx = raw.x - prev.x;
                                    let dy = raw.y - prev.y;
                                    let dist = (dx * dx + dy * dy).sqrt();
                                    // Speed-adaptive alpha:
                                    // dist < 1.5 → 1.0  (no smoothing)
                                    // dist → ∞   → 0.55 (max smoothing)
                                    let alpha = if dist < 1.5 {
                                        1.0
                                    } else {
                                        (0.55 + 1.8 / (dist + 1.8)).min(1.0)
                                    };
                                    Pos2::new(prev.x + alpha * dx, prev.y + alpha * dy)
                                } else {
                                    raw
                                };
                                self.tool_state.smooth_pos = Some(smoothed);
                                result.push((smoothed.x, smoothed.y));
                            }
                            result
                        };

                        // Accumulate a single dirty rect for the entire frame
                        let mut frame_dirty_rect = Rect::NOTHING;

                        // Mirror mode: generate mirrored positions for drawing
                        let mirror = canvas_state.mirror_mode;
                        let mw = canvas_state.width;
                        let mh = canvas_state.height;

                        for &pos in &smoothed_positions {
                            // Generate all positions (original + mirrored)
                            let positions_to_draw = mirror.mirror_positions(pos.0, pos.1, mw, mh);

                            // Also mirror the start position for line drawing
                            let start_precise = self.tool_state.last_precise_pos;

                            for &mpos in positions_to_draw.iter() {
                                // CPU-only paint step: lerp from last_precise_pos → pos
                                let modified_rect = if self.active_tool == Tool::Pencil {
                                    // Pencil: draw single pixel at each position (no line interpolation)
                                    self.draw_pixel_and_get_bounds(
                                        canvas_state,
                                        mpos,
                                        use_secondary,
                                        primary_color_f32,
                                        secondary_color_f32,
                                    )
                                } else if let Some(start_p) = start_precise {
                                    // Mirror the start point to match this arm
                                    let start_mirrors =
                                        mirror.mirror_positions(start_p.x, start_p.y, mw, mh);
                                    let arm_idx = positions_to_draw
                                        .iter()
                                        .position(|p| *p == mpos)
                                        .unwrap_or(0);
                                    let start_m = if arm_idx < start_mirrors.len {
                                        start_mirrors.data[arm_idx]
                                    } else {
                                        (start_p.x, start_p.y)
                                    };
                                    self.draw_line_and_get_bounds(
                                        canvas_state,
                                        start_m,
                                        mpos,
                                        is_eraser,
                                        use_secondary,
                                        primary_color_f32,
                                        secondary_color_f32,
                                    )
                                } else {
                                    self.draw_circle_and_get_bounds(
                                        canvas_state,
                                        mpos,
                                        is_eraser,
                                        use_secondary,
                                        primary_color_f32,
                                        secondary_color_f32,
                                    )
                                };

                                // Grow the frame's union dirty rect (CPU only, no GPU calls)
                                frame_dirty_rect = frame_dirty_rect.union(modified_rect);
                            }

                            // Update last position for next paint step (always track the original, not mirrored)
                            self.tool_state.last_precise_pos = Some(Pos2::new(pos.0, pos.1));
                        }

                        // Track stroke bounds for undo/redo
                        self.stroke_tracker.expand_bounds(frame_dirty_rect);

                        // ============================================================
                        // SINGLE GPU UPLOAD: Mark dirty ONCE for all sub-frame events
                        // ============================================================
                        if frame_dirty_rect.is_positive() {
                            // All tools (Brush, Pencil, Eraser) use the preview path now
                            canvas_state.mark_preview_changed_rect(frame_dirty_rect);
                        }

                        self.tool_state.last_pos = Some(current_pos);
                        self.tool_state.last_brush_pos = Some(current_pos);
                        ui.ctx().request_repaint();
                    }
                } else {
                    // Mouse released - commit and reset state
                    if self.tool_state.last_pos.is_some() {
                        let is_eraser = self.active_tool == Tool::Eraser;

                        // Capture "before" NOW (layer still unchanged), then commit
                        // Finish stroke tracking BEFORE commit - this captures "before" from unchanged layer
                        stroke_event = self.stroke_tracker.finish(canvas_state);

                        if is_eraser {
                            // Commit the eraser mask to the active layer
                            self.commit_eraser_to_layer(canvas_state);
                        } else {
                            // Commit the preview layer to the active layer (Brush/Pencil)
                            self.commit_bezier_to_layer(canvas_state, primary_color_f32);
                        }
                        // Clear preview layer
                        canvas_state.clear_preview_state();
                        // Mark only stroke bounds dirty (not full canvas)
                        if let Some(ref ev) = stroke_event {
                            canvas_state.mark_dirty(Some(ev.bounds.expand(12.0)));
                        } else {
                            self.mark_full_dirty(canvas_state);
                        }
                    }

                    self.tool_state.last_pos = None;
                    self.tool_state.last_precise_pos = None;
                    self.tool_state.distance_remainder = 0.0;
                    self.tool_state.using_secondary_color = false;
                    self.tool_state.smooth_pos = None;
                }
            }
            Tool::Line => {
                // TASK 1: Advanced Bézier Line Tool
                match self.line_state.line_tool.stage {
                    LineStage::Idle => {
                        // Clear the require_mouse_release flag when mouse is released
                        if is_primary_released {
                            self.line_state.line_tool.require_mouse_release = false;
                        }

                        if !self.line_state.line_tool.require_mouse_release {
                            // Normal behavior: start line on click (or click and drag)
                            if is_primary_pressed && let Some(pos) = canvas_pos {
                                let pos2 = Pos2::new(pos.0 as f32, pos.1 as f32);
                                self.line_state.line_tool.control_points = [pos2; 4];
                                self.line_state.line_tool.stage = LineStage::Dragging;
                                self.line_state.line_tool.last_bounds = None;
                            }
                        } else {
                            // After committing a line: require drag or release+click
                            if is_primary_pressed && let Some(pos) = canvas_pos {
                                // Store initial position to detect drag
                                let pos2 = Pos2::new(pos.0 as f32, pos.1 as f32);
                                self.line_state.line_tool.initial_mouse_pos = Some(pos2);
                                self.line_state.line_tool.control_points = [pos2; 4];
                            }

                            // Transition to Dragging only if the mouse moved while pressed
                            if is_primary_down
                                && self.line_state.line_tool.initial_mouse_pos.is_some()
                                && let Some(pos) = canvas_pos
                            {
                                let current_pos = Pos2::new(pos.0 as f32, pos.1 as f32);
                                let initial_pos =
                                    self.line_state.line_tool.initial_mouse_pos.unwrap();

                                // Check if mouse moved enough to be considered a drag (more than 2 pixels)
                                if (current_pos - initial_pos).length() > 2.0 {
                                    self.line_state.line_tool.stage = LineStage::Dragging;
                                    self.line_state.line_tool.last_bounds = None;
                                    self.line_state.line_tool.require_mouse_release = false;
                                }
                            }

                            // If mouse was released without dragging, reset
                            if is_primary_released
                                && self.line_state.line_tool.initial_mouse_pos.is_some()
                            {
                                self.line_state.line_tool.initial_mouse_pos = None;
                            }
                        }
                    }
                    LineStage::Dragging => {
                        if let Some(pos) = canvas_pos {
                            let raw_pos = Pos2::new(pos.0 as f32, pos.1 as f32);
                            let p0 = self.line_state.line_tool.control_points[0];
                            // Snap to nearest 45° angle when Shift is held
                            let pos2 = if shift_held {
                                let dx = raw_pos.x - p0.x;
                                let dy = raw_pos.y - p0.y;
                                let angle = dy.atan2(dx);
                                let snap = std::f32::consts::PI / 4.0; // 45°
                                let snapped_angle = (angle / snap).round() * snap;
                                let dist = (dx * dx + dy * dy).sqrt();
                                Pos2::new(
                                    p0.x + dist * snapped_angle.cos(),
                                    p0.y + dist * snapped_angle.sin(),
                                )
                            } else {
                                raw_pos
                            };
                            // Update end point
                            self.line_state.line_tool.control_points[3] = pos2;
                            let midpoint = Pos2::new((p0.x + pos2.x) / 2.0, (p0.y + pos2.y) / 2.0);
                            self.line_state.line_tool.control_points[1] = midpoint;
                            self.line_state.line_tool.control_points[2] = midpoint;

                            // OPTIMIZATION: Calculate bounds and use smart dirty rects
                            let current_bounds = self.get_bezier_bounds(
                                self.line_state.line_tool.control_points,
                                canvas_state.width,
                                canvas_state.height,
                            );

                            // Rasterize preview with focused clearing
                            self.rasterize_bezier(
                                canvas_state,
                                self.line_state.line_tool.control_points,
                                self.properties.color,
                                self.line_state.line_tool.options.pattern,
                                self.line_state.line_tool.options.cap_style,
                                self.line_state.line_tool.last_bounds,
                            );

                            // Preview layer changed - use combined bounds for partial GPU upload
                            let dirty_bounds = match self.line_state.line_tool.last_bounds {
                                Some(last) => last.union(current_bounds),
                                None => current_bounds,
                            };
                            canvas_state.mark_preview_changed_rect(dirty_bounds);

                            // Track bounds for next frame
                            self.line_state.line_tool.last_bounds = Some(current_bounds);
                        }

                        if is_primary_released {
                            // Enter editing stage - start tracking for undo/redo
                            self.line_state.line_tool.stage = LineStage::Editing;

                            // Initialize tracking values for change detection
                            self.line_state.line_tool.last_size = self.properties.size;
                            self.line_state.line_tool.last_pattern =
                                self.line_state.line_tool.options.pattern;
                            self.line_state.line_tool.last_cap_style =
                                self.line_state.line_tool.options.cap_style;

                            // Start stroke tracking (line uses preview layer like Brush)
                            self.stroke_tracker
                                .start_preview_tool(canvas_state.active_layer_index, "Line");
                            if let Some(bounds) = self.line_state.line_tool.last_bounds {
                                self.stroke_tracker.expand_bounds(bounds);
                            }
                        }
                    }
                    LineStage::Editing => {
                        let mouse_pos = ui.input(|i| i.pointer.interact_pos());

                        // Check if settings changed and re-render if needed
                        let settings_changed =
                            (self.properties.size - self.line_state.line_tool.last_size).abs()
                                > 0.1
                                || self.line_state.line_tool.options.pattern
                                    != self.line_state.line_tool.last_pattern
                                || self.line_state.line_tool.options.cap_style
                                    != self.line_state.line_tool.last_cap_style;

                        if settings_changed {
                            // Update tracked values
                            self.line_state.line_tool.last_size = self.properties.size;
                            self.line_state.line_tool.last_pattern =
                                self.line_state.line_tool.options.pattern;
                            self.line_state.line_tool.last_cap_style =
                                self.line_state.line_tool.options.cap_style;

                            // Re-calculate bounds
                            let current_bounds = self.get_bezier_bounds(
                                self.line_state.line_tool.control_points,
                                canvas_state.width,
                                canvas_state.height,
                            );

                            // Re-rasterize with new settings
                            self.rasterize_bezier(
                                canvas_state,
                                self.line_state.line_tool.control_points,
                                self.properties.color,
                                self.line_state.line_tool.options.pattern,
                                self.line_state.line_tool.options.cap_style,
                                self.line_state.line_tool.last_bounds,
                            );

                            // Mark dirty for preview update
                            let dirty_bounds = match self.line_state.line_tool.last_bounds {
                                Some(last) => last.union(current_bounds),
                                None => current_bounds,
                            };
                            canvas_state.mark_preview_changed_rect(dirty_bounds);
                            self.line_state.line_tool.last_bounds = Some(current_bounds);

                            ui.ctx().request_repaint();
                        }

                        // Handle Enter key or tool change to commit
                        if enter_pressed {
                            // Capture "before" for undo (layer still unchanged, line is in preview)
                            if let Some(final_bounds) = self.line_state.line_tool.last_bounds {
                                self.stroke_tracker.expand_bounds(final_bounds);
                            }
                            stroke_event = self.stroke_tracker.finish(canvas_state);

                            // Commit the line to the actual layer
                            let mirror_bounds = canvas_state.mirror_preview_layer();
                            self.commit_bezier_to_layer(canvas_state, secondary_color_f32);

                            // Mark dirty: combine original line bounds with mirrored bounds
                            let base_dirty = self.line_state.line_tool.last_bounds;
                            let combined = match (base_dirty, mirror_bounds) {
                                (Some(b), Some(m)) => Some(b.union(m)),
                                (Some(b), None) => Some(b),
                                (None, Some(m)) => Some(m),
                                (None, None) => None,
                            };
                            if let Some(dirty) = combined {
                                canvas_state.mark_dirty(Some(dirty));
                            } else {
                                self.mark_full_dirty(canvas_state);
                            }

                            canvas_state.clear_preview_state();
                            self.line_state.line_tool.stage = LineStage::Idle;
                            self.line_state.line_tool.last_bounds = None; // Reset bounds
                            self.line_state.line_tool.require_mouse_release = false; // Allow new line after Enter
                            self.line_state.line_tool.initial_mouse_pos = None;
                            ui.ctx().request_repaint();
                        } else if let Some(screen_pos) = mouse_pos {
                            let handle_radius = 8.0; // Screen pixels

                            // Check if starting to drag a handle
                            // -- Compute pan handle screen position --
                            // Pan handle sits 22px above-left of P0 in screen space so it
                            // doesn't overlap the start (green) endpoint handle.
                            let p0_screen = self.canvas_pos2_to_screen(
                                self.line_state.line_tool.control_points[0],
                                canvas_rect,
                                zoom,
                            );
                            let pan_offset_screen = egui::Vec2::new(-22.0, -22.0);
                            let pan_handle_screen = p0_screen + pan_offset_screen;
                            let pan_hit_radius = 10.0f32;

                            // -- Hover tracking for pan handle (cursor icon) --
                            let hovering_pan =
                                (pan_handle_screen - screen_pos).length() < pan_hit_radius;
                            self.line_state.line_tool.pan_handle_hovering = hovering_pan;

                            if is_primary_down
                                && self.line_state.line_tool.dragging_handle.is_none()
                                && !self.line_state.line_tool.pan_handle_dragging
                            {
                                // -- Check pan handle first --
                                if is_primary_pressed && hovering_pan {
                                    let canvas_pos_now =
                                        self.screen_to_canvas_pos2(screen_pos, canvas_rect, zoom);
                                    self.line_state.line_tool.pan_handle_dragging = true;
                                    self.line_state.line_tool.pan_drag_canvas_start =
                                        Some(canvas_pos_now);
                                } else {
                                    // -- Check individual endpoint / control handles --
                                    let mut clicked_handle = false;
                                    for i in 0..4 {
                                        let handle_screen = self.canvas_pos2_to_screen(
                                            self.line_state.line_tool.control_points[i],
                                            canvas_rect,
                                            zoom,
                                        );
                                        if (handle_screen - screen_pos).length() < handle_radius {
                                            self.line_state.line_tool.dragging_handle = Some(i);
                                            clicked_handle = true;
                                            break;
                                        }
                                    }

                                    // If clicked but didn't hit any handle, commit current line
                                    if !clicked_handle && is_primary_pressed {
                                        // Capture "before" for undo (layer still unchanged, line is in preview)
                                        if let Some(final_bounds) =
                                            self.line_state.line_tool.last_bounds
                                        {
                                            self.stroke_tracker.expand_bounds(final_bounds);
                                        }
                                        stroke_event = self.stroke_tracker.finish(canvas_state);

                                        // Commit the current line
                                        let mirror_bounds = canvas_state.mirror_preview_layer();
                                        self.commit_bezier_to_layer(
                                            canvas_state,
                                            secondary_color_f32,
                                        );

                                        // Mark dirty: combine original line bounds with mirrored bounds
                                        let base_dirty = self.line_state.line_tool.last_bounds;
                                        let combined = match (base_dirty, mirror_bounds) {
                                            (Some(b), Some(m)) => Some(b.union(m)),
                                            (Some(b), None) => Some(b),
                                            (None, Some(m)) => Some(m),
                                            (None, None) => None,
                                        };
                                        if let Some(dirty) = combined {
                                            canvas_state.mark_dirty(Some(dirty));
                                        } else {
                                            self.mark_full_dirty(canvas_state);
                                        }

                                        canvas_state.clear_preview_state();
                                        self.line_state.line_tool.stage = LineStage::Idle;
                                        self.line_state.line_tool.last_bounds = None;
                                        self.line_state.line_tool.require_mouse_release = true;
                                        self.line_state.line_tool.initial_mouse_pos = None;
                                        self.line_state.line_tool.pan_handle_hovering = false;

                                        ui.ctx().request_repaint();
                                    }
                                }
                            }

                            // -- Pan drag: translate all 4 control points together --
                            if is_primary_down && self.line_state.line_tool.pan_handle_dragging {
                                let canvas_pos_now =
                                    self.screen_to_canvas_pos2(screen_pos, canvas_rect, zoom);
                                if let Some(drag_start) =
                                    self.line_state.line_tool.pan_drag_canvas_start
                                {
                                    let delta = canvas_pos_now - drag_start;
                                    if delta.length() > 0.0 {
                                        for pt in
                                            self.line_state.line_tool.control_points.iter_mut()
                                        {
                                            pt.x = (pt.x + delta.x)
                                                .clamp(0.0, canvas_state.width as f32 - 1.0);
                                            pt.y = (pt.y + delta.y)
                                                .clamp(0.0, canvas_state.height as f32 - 1.0);
                                        }
                                        self.line_state.line_tool.pan_drag_canvas_start =
                                            Some(canvas_pos_now);

                                        let current_bounds = self.get_bezier_bounds(
                                            self.line_state.line_tool.control_points,
                                            canvas_state.width,
                                            canvas_state.height,
                                        );
                                        self.stroke_tracker.expand_bounds(current_bounds);
                                        self.rasterize_bezier(
                                            canvas_state,
                                            self.line_state.line_tool.control_points,
                                            self.properties.color,
                                            self.line_state.line_tool.options.pattern,
                                            self.line_state.line_tool.options.cap_style,
                                            self.line_state.line_tool.last_bounds,
                                        );
                                        let dirty_bounds =
                                            match self.line_state.line_tool.last_bounds {
                                                Some(last) => last.union(current_bounds),
                                                None => current_bounds,
                                            };
                                        canvas_state.mark_preview_changed_rect(dirty_bounds);
                                        self.line_state.line_tool.last_bounds =
                                            Some(current_bounds);
                                        ui.ctx().request_repaint();
                                    }
                                }
                            }

                            // -- Individual handle drag --
                            if is_primary_down
                                && !self.line_state.line_tool.pan_handle_dragging
                                && let Some(handle_idx) = self.line_state.line_tool.dragging_handle
                            {
                                let canvas_pos_float =
                                    self.screen_to_canvas_pos2(screen_pos, canvas_rect, zoom);
                                // Clamp to canvas bounds
                                let clamped = Pos2::new(
                                    canvas_pos_float
                                        .x
                                        .clamp(0.0, canvas_state.width as f32 - 1.0),
                                    canvas_pos_float
                                        .y
                                        .clamp(0.0, canvas_state.height as f32 - 1.0),
                                );
                                self.line_state.line_tool.control_points[handle_idx] = clamped;

                                // OPTIMIZATION: Calculate bounds and use smart dirty rects
                                let current_bounds = self.get_bezier_bounds(
                                    self.line_state.line_tool.control_points,
                                    canvas_state.width,
                                    canvas_state.height,
                                );

                                // Track bounds for undo/redo
                                self.stroke_tracker.expand_bounds(current_bounds);

                                // Re-rasterize the curve with focused clearing
                                self.rasterize_bezier(
                                    canvas_state,
                                    self.line_state.line_tool.control_points,
                                    self.properties.color,
                                    self.line_state.line_tool.options.pattern,
                                    self.line_state.line_tool.options.cap_style,
                                    self.line_state.line_tool.last_bounds,
                                );

                                // Preview layer changed - use combined bounds for partial GPU upload
                                let dirty_bounds = match self.line_state.line_tool.last_bounds {
                                    Some(last) => last.union(current_bounds),
                                    None => current_bounds,
                                };
                                canvas_state.mark_preview_changed_rect(dirty_bounds);

                                // Track bounds for next frame
                                self.line_state.line_tool.last_bounds = Some(current_bounds);

                                ui.ctx().request_repaint();
                            }

                            // -- Release all handles --------------------------------------------------
                            if is_primary_released {
                                self.line_state.line_tool.dragging_handle = None;
                                self.line_state.line_tool.pan_handle_dragging = false;
                                self.line_state.line_tool.pan_drag_canvas_start = None;
                            }

                            // Draw handles and control lines
                            self.draw_bezier_handles(painter, canvas_rect, zoom);
                        }
                    }
                }
            }

            Tool::RectangleSelect | Tool::EllipseSelect => {
                let esc_pressed = ui.input(|i| i.key_pressed(egui::Key::Escape));
                let alt_held = ui.input(|i| i.modifiers.alt);
                let is_secondary_pressed =
                    ui.input(|i| i.pointer.button_pressed(egui::PointerButton::Secondary));

                // Esc or Enter: deselect
                if esc_pressed || enter_pressed {
                    canvas_state.clear_selection();
                    self.selection_state.dragging = false;
                    self.selection_state.drag_start = None;
                    self.selection_state.drag_end = None;
                    self.selection_state.right_click_drag = false;
                    canvas_state.mark_dirty(None);
                    ui.ctx().request_repaint();
                }

                // Start drag on primary OR secondary press.
                // At drag start, lock the effective mode from modifier keys:
                //   Shift+Alt → Intersect
                //   Shift      → Add
                //   Alt        → Subtract
                //   Right-click → Subtract (when context bar mode is Replace)
                //   else       → context bar mode
                if (is_primary_pressed || is_secondary_pressed)
                    && !self.selection_state.dragging
                    && let Some(pos) = canvas_pos
                {
                    let pos2 = Pos2::new(pos.0 as f32, pos.1 as f32);
                    self.selection_state.dragging = true;
                    self.selection_state.drag_start = Some(pos2);
                    self.selection_state.drag_end = Some(pos2);
                    self.selection_state.right_click_drag = is_secondary_pressed;
                    // Determine and lock effective mode
                    self.selection_state.drag_effective_mode = if is_secondary_pressed {
                        SelectionMode::Subtract
                    } else if shift_held && alt_held {
                        SelectionMode::Intersect
                    } else if shift_held {
                        SelectionMode::Add
                    } else if alt_held {
                        SelectionMode::Subtract
                    } else {
                        self.selection_state.mode
                    };
                }

                let any_button_down = is_primary_down || is_secondary_down;
                let any_button_released = is_primary_released || is_secondary_released;

                if any_button_down
                    && self.selection_state.dragging
                    && let Some(pos) = canvas_pos
                {
                    let mut end = Pos2::new(pos.0 as f32, pos.1 as f32);

                    // Shift => constrain to 1:1 aspect ratio (square / circle)
                    if shift_held && let Some(start) = self.selection_state.drag_start {
                        let dx = end.x - start.x;
                        let dy = end.y - start.y;
                        let side = dx.abs().max(dy.abs());
                        end = Pos2::new(start.x + side * dx.signum(), start.y + side * dy.signum());
                    }
                    self.selection_state.drag_end = Some(end);
                    ui.ctx().request_repaint();
                }

                if any_button_released && self.selection_state.dragging {
                    let effective_mode = self.selection_state.drag_effective_mode;
                    self.selection_state.dragging = false;
                    self.selection_state.right_click_drag = false;

                    if let (Some(start), Some(end)) = (
                        self.selection_state.drag_start,
                        self.selection_state.drag_end,
                    ) {
                        let min_x = start.x.min(end.x).max(0.0) as u32;
                        let min_y = start.y.min(end.y).max(0.0) as u32;
                        let max_x =
                            (start.x.max(end.x) as u32).min(canvas_state.width.saturating_sub(1));
                        let max_y =
                            (start.y.max(end.y) as u32).min(canvas_state.height.saturating_sub(1));

                        // Ignore tiny accidental clicks (< 2px)
                        if max_x.saturating_sub(min_x) > 1 && max_y.saturating_sub(min_y) > 1 {
                            let shape = match self.active_tool {
                                Tool::RectangleSelect => SelectionShape::Rectangle {
                                    min_x,
                                    min_y,
                                    max_x,
                                    max_y,
                                },
                                Tool::EllipseSelect => {
                                    let cx = (min_x as f32 + max_x as f32) / 2.0;
                                    let cy = (min_y as f32 + max_y as f32) / 2.0;
                                    let rx = (max_x as f32 - min_x as f32) / 2.0;
                                    let ry = (max_y as f32 - min_y as f32) / 2.0;
                                    SelectionShape::Ellipse { cx, cy, rx, ry }
                                }
                                _ => unreachable!(),
                            };

                            canvas_state.apply_selection_shape(&shape, effective_mode);
                            canvas_state.mark_dirty(None);
                        } else {
                            // Tiny click => deselect
                            canvas_state.clear_selection();
                            canvas_state.mark_dirty(None);
                        }
                    }

                    self.selection_state.drag_start = None;
                    self.selection_state.drag_end = None;
                    ui.ctx().request_repaint();
                }
            }

            // Move tools are handled by app.rs via PasteOverlay / mask shifting.
            Tool::MovePixels | Tool::MoveSelection => {}

            // Magic Wand tool - color-based selection with live preview
            Tool::MagicWand => {
                let esc_pressed = ui.input(|i| i.key_pressed(egui::Key::Escape));
                let shift_held_mw = ui.input(|i| i.modifiers.shift);
                let alt_held_mw = ui.input(|i| i.modifiers.alt);
                let ctrl_held_mw = ui.input(|i| i.modifiers.ctrl);
                // Ctrl+Shift = global select (all matching pixels across entire canvas)
                let global_select = ctrl_held_mw && shift_held_mw;
                // Determine click-time SelectionMode from modifier keys
                let click_mode = if is_primary_clicked || is_secondary_clicked {
                    if is_secondary_clicked {
                        SelectionMode::Subtract
                    } else if shift_held_mw && alt_held_mw {
                        SelectionMode::Intersect
                    } else if shift_held_mw {
                        SelectionMode::Add
                    } else if alt_held_mw {
                        SelectionMode::Subtract
                    } else {
                        self.magic_wand_state.effective_mode
                    }
                } else {
                    self.magic_wand_state.effective_mode
                };

                // Commit on Enter or cancel on Escape
                if enter_pressed || esc_pressed {
                    if self.magic_wand_state.active_selection.is_some() {
                        self.magic_wand_state.active_selection = None;
                        canvas_state.mark_dirty(None);
                        ui.ctx().request_repaint();
                    }
                    if esc_pressed {
                        canvas_state.clear_selection();
                    }
                }

                // Click to start new selection (or commit current + start new)
                if (is_primary_clicked || is_secondary_clicked)
                    && let Some(pos) = canvas_pos
                {
                    // If there's an active selection preview, commit it first
                    if self.magic_wand_state.active_selection.is_some() {
                        self.magic_wand_state.active_selection = None;
                    }
                    // Perform new selection
                    self.magic_wand_state.global_select = global_select;
                    self.perform_magic_wand_selection(canvas_state, pos, click_mode);
                }

                // Update preview if tolerance changed (while selection is active)
                if self.magic_wand_state.active_selection.is_some()
                    && self.active_tool == Tool::MagicWand
                {
                    self.update_magic_wand_preview(canvas_state);
                    if self.magic_wand_state.recalc_pending {
                        ui.ctx().request_repaint();
                    }
                }
            }

            // Fill tool - flood fill with live preview
            Tool::Fill => {
                let esc_pressed = ui.input(|i| i.key_pressed(egui::Key::Escape));

                // Commit on Enter or cancel on Escape
                if enter_pressed || esc_pressed {
                    if self.fill_state.active_fill.is_some() && !esc_pressed {
                        self.commit_fill_preview(canvas_state);
                    } else if self.fill_state.active_fill.is_some() {
                        // Cancel preview on Escape
                        self.fill_state.active_fill = None;
                        self.fill_state.fill_color_u8 = None;
                        canvas_state.clear_preview_state();
                        self.stroke_tracker.cancel();
                    }
                }

                // Click to fill: commit current fill (if any) then start new one
                if (is_primary_clicked || is_secondary_clicked)
                    && let Some(pos) = canvas_pos
                {
                    // If there's an active fill, commit it first
                    if self.fill_state.active_fill.is_some() {
                        self.commit_fill_preview(canvas_state);
                    }
                    // Start new fill
                    let use_secondary = is_secondary_clicked;
                    self.perform_flood_fill(
                        canvas_state,
                        pos,
                        use_secondary,
                        primary_color_f32,
                        secondary_color_f32,
                    );
                }

                // Update preview if tolerance/AA/color changed
                if self.fill_state.active_fill.is_some() && self.active_tool == Tool::Fill {
                    // Detect color/opacity change from the colors panel
                    let current_color_f32 = if self.fill_state.use_secondary_color {
                        secondary_color_f32
                    } else {
                        primary_color_f32
                    };
                    let current_color_u8 = Rgba([
                        (current_color_f32[0] * 255.0) as u8,
                        (current_color_f32[1] * 255.0) as u8,
                        (current_color_f32[2] * 255.0) as u8,
                        (current_color_f32[3] * 255.0) as u8,
                    ]);
                    if self.fill_state.fill_color_u8 != Some(current_color_u8) {
                        self.fill_state.fill_color_u8 = Some(current_color_u8);
                        self.fill_state.tolerance_changed_at = Some(Instant::now());
                        self.fill_state.recalc_pending = true;
                    }

                    self.update_fill_preview(canvas_state);

                    if self.fill_state.recalc_pending {
                        ui.ctx().request_repaint();
                    }
                }
            }

            // Color picker tool - sample colors
            Tool::ColorPicker => {
                if (is_primary_clicked || is_secondary_clicked)
                    && let Some(pos) = canvas_pos
                {
                    // TODO: Implement color picker logic
                    self.pick_color_at_position(canvas_state, pos, is_secondary_clicked);
                }
            }

            // ================================================================
            // TEXT TOOL — click to place cursor, type to add text
            // ================================================================
            Tool::Text => {
                let escape_pressed = ui.input(|i| i.key_pressed(egui::Key::Escape));
                let ctrl_enter =
                    ui.input(|i| i.key_pressed(egui::Key::Enter) && i.modifiers.command);

                // Escape: cancel editing
                if escape_pressed && self.text_state.is_editing {
                    self.text_state.text.clear();
                    self.text_state.cursor_pos = 0;
                    self.text_state.is_editing = false;
                    self.text_state.origin = None;
                    self.text_state.dragging_handle = false;
                    canvas_state.clear_preview_state();
                    canvas_state.mark_dirty(None);
                }

                // Ctrl+Enter: commit text
                if ctrl_enter && self.text_state.is_editing && !self.text_state.text.is_empty() {
                    // Use deferred commit for loading bar
                    self.stroke_tracker
                        .start_preview_tool(canvas_state.active_layer_index, "Text");
                    self.text_state.commit_pending = true;
                    self.text_state.commit_pending_frame = 0;
                }

                // Enter (without Ctrl): insert newline
                if enter_pressed && !ctrl_enter && self.text_state.is_editing {
                    self.text_state
                        .text
                        .insert(self.text_state.cursor_pos, '\n');
                    self.text_state.cursor_pos += 1;
                    self.text_state.preview_dirty = true;
                }

                // Compute move handle position — alignment-aware
                let handle_radius_screen = 6.0;
                let handle_offset_canvas = 10.0 / zoom; // close distance for fine control
                let handle_canvas_pos = self.text_state.origin.map(|o| {
                    use crate::ops::text::TextAlignment;
                    match self.text_state.alignment {
                        TextAlignment::Left => {
                            // Handle to the left
                            (
                                o[0] - handle_offset_canvas,
                                o[1] + self.text_state.font_size * 0.5,
                            )
                        }
                        TextAlignment::Center => {
                            // Handle above, centered horizontally
                            (o[0], o[1] - handle_offset_canvas)
                        }
                        TextAlignment::Right => {
                            // Handle to the right
                            (
                                o[0] + handle_offset_canvas,
                                o[1] + self.text_state.font_size * 0.5,
                            )
                        }
                    }
                });

                // Detect hover over the move handle (for cursor icon)
                self.text_state.hovering_handle = if self.text_state.is_editing {
                    if let (Some(pos_f), Some(hp)) = (canvas_pos_f32, handle_canvas_pos) {
                        let dx = (pos_f.0 - hp.0) * zoom;
                        let dy = (pos_f.1 - hp.1) * zoom;
                        (dx * dx + dy * dy).sqrt() < handle_radius_screen + 4.0
                    } else {
                        false
                    }
                } else {
                    false
                };

                // Handle dragging the move handle
                if is_primary_pressed
                    && self.text_state.is_editing
                    && let (Some(pos_f), Some(hp)) = (canvas_pos_f32, handle_canvas_pos)
                {
                    let dx = (pos_f.0 - hp.0) * zoom;
                    let dy = (pos_f.1 - hp.1) * zoom;
                    let dist = (dx * dx + dy * dy).sqrt();
                    if dist < handle_radius_screen + 4.0 {
                        self.text_state.dragging_handle = true;
                        if let Some(origin) = self.text_state.origin {
                            self.text_state.drag_offset =
                                [pos_f.0 - origin[0], pos_f.1 - origin[1]];
                        }
                    }
                }

                if self.text_state.dragging_handle
                    && is_primary_down
                    && let Some(pos_f) = canvas_pos_f32
                {
                    let new_x = pos_f.0 - self.text_state.drag_offset[0];
                    let new_y = pos_f.1 - self.text_state.drag_offset[1];
                    self.text_state.origin = Some([new_x, new_y]);
                    // During drag, skip full re-rasterization: reuse cached raster buffer
                    // and just re-blit at new origin offset.
                    if self.text_state.cached_raster_w > 0 && self.text_state.cached_raster_h > 0 {
                        if let Some(cached_origin) = self.text_state.cached_raster_origin {
                            let dx = new_x - cached_origin[0];
                            let dy = new_y - cached_origin[1];
                            let off_x = self.text_state.cached_raster_off_x + dx as i32;
                            let off_y = self.text_state.cached_raster_off_y + dy as i32;
                            let buf_w = self.text_state.cached_raster_w;
                            let buf_h = self.text_state.cached_raster_h;

                            let mut preview =
                                TiledImage::new(canvas_state.width, canvas_state.height);
                            preview.blit_rgba_at(
                                off_x,
                                off_y,
                                buf_w,
                                buf_h,
                                &self.text_state.cached_raster_buf,
                            );

                            canvas_state.preview_layer = Some(preview);
                            canvas_state.preview_blend_mode = self.properties.blending_mode;
                            canvas_state.preview_force_composite =
                                self.properties.blending_mode != BlendMode::Normal;
                            canvas_state.preview_is_eraser = false;
                            canvas_state.preview_downscale = 1;
                            canvas_state.preview_flat_ready = false;
                            canvas_state.preview_stroke_bounds = Some(egui::Rect::from_min_max(
                                egui::pos2(off_x as f32, off_y as f32),
                                egui::pos2(
                                    (off_x + buf_w as i32) as f32,
                                    (off_y + buf_h as i32) as f32,
                                ),
                            ));
                            if canvas_state.preview_texture_cache.is_some() {
                                canvas_state.preview_dirty_rect = Some(egui::Rect::from_min_max(
                                    egui::pos2(off_x as f32, off_y as f32),
                                    egui::pos2(
                                        (off_x + buf_w as i32) as f32,
                                        (off_y + buf_h as i32) as f32,
                                    ),
                                ));
                            } else {
                                canvas_state.preview_texture_cache = None;
                            }
                            canvas_state.mark_dirty(None);
                            self.text_state.preview_dirty = false;
                        } else {
                            self.text_state.preview_dirty = true;
                        }
                    } else {
                        self.text_state.preview_dirty = true;
                    }
                }

                if is_primary_released {
                    self.text_state.dragging_handle = false;
                }

                // Click to place origin (or commit existing + start new)
                // Only if not dragging the handle
                if is_primary_clicked && !self.text_state.dragging_handle {
                    // Check if click is on the handle — if so, skip placement
                    let on_handle =
                        if let (Some(pos_f), Some(hp)) = (canvas_pos_f32, handle_canvas_pos) {
                            let dx = (pos_f.0 - hp.0) * zoom;
                            let dy = (pos_f.1 - hp.1) * zoom;
                            (dx * dx + dy * dy).sqrt() < handle_radius_screen + 4.0
                        } else {
                            false
                        };

                    if !on_handle && let Some(pos_f) = canvas_pos_f32 {
                        if self.text_state.is_editing && !self.text_state.text.is_empty() {
                            // Commit current text first
                            self.stroke_tracker
                                .start_preview_tool(canvas_state.active_layer_index, "Text");
                            self.commit_text(canvas_state);
                            if self.pending_stroke_event.is_none()
                                && let Some(evt) = stroke_event.take()
                            {
                                self.pending_stroke_event = Some(evt);
                            }
                        }
                        // Start new text at this position
                        self.text_state.origin = Some([pos_f.0, pos_f.1]);
                        self.text_state.is_editing = true;
                        self.text_state.text.clear();
                        self.text_state.cursor_pos = 0;
                        self.text_state.preview_dirty = true;
                        self.stroke_tracker
                            .start_preview_tool(canvas_state.active_layer_index, "Text");
                    }
                }

                // Detect color changes from the color widget
                let current_color = [
                    (primary_color_f32[0] * 255.0) as u8,
                    (primary_color_f32[1] * 255.0) as u8,
                    (primary_color_f32[2] * 255.0) as u8,
                    (primary_color_f32[3] * 255.0) as u8,
                ];
                if self.text_state.is_editing && self.text_state.last_color != current_color {
                    self.text_state.last_color = current_color;
                    self.text_state.preview_dirty = true;
                }

                // Capture text input
                if self.text_state.is_editing {
                    let events: Vec<egui::Event> = ui.input(|i| i.events.clone());
                    for event in &events {
                        match event {
                            egui::Event::Text(t) => {
                                self.text_state
                                    .text
                                    .insert_str(self.text_state.cursor_pos, t);
                                self.text_state.cursor_pos += t.len();
                                self.text_state.preview_dirty = true;
                            }
                            egui::Event::Key {
                                key: egui::Key::Backspace,
                                pressed: true,
                                ..
                            } => {
                                if self.text_state.cursor_pos > 0 {
                                    // Remove one char before cursor
                                    let mut chars: Vec<char> =
                                        self.text_state.text.chars().collect();
                                    let char_idx = self.text_state.text
                                        [..self.text_state.cursor_pos]
                                        .chars()
                                        .count();
                                    if char_idx > 0 {
                                        let remove_idx = char_idx - 1;
                                        chars.remove(remove_idx);
                                        self.text_state.text = chars.into_iter().collect();
                                        // Adjust cursor
                                        let new_char_pos = remove_idx;
                                        self.text_state.cursor_pos = self
                                            .text_state
                                            .text
                                            .chars()
                                            .take(new_char_pos)
                                            .map(|c| c.len_utf8())
                                            .sum();
                                    }
                                    self.text_state.preview_dirty = true;
                                }
                            }
                            egui::Event::Key {
                                key: egui::Key::ArrowLeft,
                                pressed: true,
                                ..
                            } => {
                                if self.text_state.cursor_pos > 0 {
                                    let chars: Vec<char> = self.text_state.text
                                        [..self.text_state.cursor_pos]
                                        .chars()
                                        .collect();
                                    if let Some(last) = chars.last() {
                                        self.text_state.cursor_pos -= last.len_utf8();
                                    }
                                }
                            }
                            egui::Event::Key {
                                key: egui::Key::ArrowRight,
                                pressed: true,
                                ..
                            } => {
                                if self.text_state.cursor_pos < self.text_state.text.len()
                                    && let Some(c) = self.text_state.text
                                        [self.text_state.cursor_pos..]
                                        .chars()
                                        .next()
                                {
                                    self.text_state.cursor_pos += c.len_utf8();
                                }
                            }
                            egui::Event::Key {
                                key: egui::Key::Delete,
                                pressed: true,
                                ..
                            } => {
                                if self.text_state.cursor_pos < self.text_state.text.len() {
                                    let mut chars: Vec<char> =
                                        self.text_state.text.chars().collect();
                                    let char_idx = self.text_state.text
                                        [..self.text_state.cursor_pos]
                                        .chars()
                                        .count();
                                    if char_idx < chars.len() {
                                        chars.remove(char_idx);
                                        self.text_state.text = chars.into_iter().collect();
                                    }
                                    self.text_state.preview_dirty = true;
                                }
                            }
                            egui::Event::Key {
                                key: egui::Key::Home,
                                pressed: true,
                                ..
                            } => {
                                // Move cursor to start of current line
                                let text_before =
                                    &self.text_state.text[..self.text_state.cursor_pos];
                                if let Some(nl) = text_before.rfind('\n') {
                                    self.text_state.cursor_pos = nl + 1;
                                } else {
                                    self.text_state.cursor_pos = 0;
                                }
                            }
                            egui::Event::Key {
                                key: egui::Key::End,
                                pressed: true,
                                ..
                            } => {
                                // Move cursor to end of current line
                                let text_after =
                                    &self.text_state.text[self.text_state.cursor_pos..];
                                if let Some(nl) = text_after.find('\n') {
                                    self.text_state.cursor_pos += nl;
                                } else {
                                    self.text_state.cursor_pos = self.text_state.text.len();
                                }
                            }
                            egui::Event::Paste(t) => {
                                // Filter out any non-printable chars except newlines
                                let filtered: String = t
                                    .chars()
                                    .filter(|c| !c.is_control() || *c == '\n')
                                    .collect();
                                if !filtered.is_empty() {
                                    self.text_state
                                        .text
                                        .insert_str(self.text_state.cursor_pos, &filtered);
                                    self.text_state.cursor_pos += filtered.len();
                                    self.text_state.preview_dirty = true;
                                }
                            }
                            _ => {}
                        }
                    }

                    // Re-render preview if dirty
                    if self.text_state.preview_dirty {
                        self.render_text_preview(canvas_state, primary_color_f32);
                    }

                    // Draw text overlay: outline border, move handle, blinking cursor
                    if let Some(origin) = self.text_state.origin {
                        let time = ui.input(|i| i.time);
                        self.text_state.cursor_blink_timer = time;

                        // Compute text bounds for border outline
                        let font_size = self.text_state.font_size;
                        self.ensure_text_font_loaded();
                        let line_height = if let Some(ref font) = self.text_state.loaded_font {
                            use ab_glyph::{Font as _, ScaleFont as _};
                            font.as_scaled(font_size).height()
                        } else {
                            font_size * 1.2
                        };

                        let lines: Vec<&str> = self.text_state.text.split('\n').collect();
                        let num_lines = lines.len().max(1);

                        // Compute max line width
                        let max_width = if let Some(ref font) = self.text_state.loaded_font {
                            use ab_glyph::{Font as _, ScaleFont as _};
                            let scaled = font.as_scaled(font_size);
                            let mut max_w = font_size * 2.0; // minimum width
                            for line in &lines {
                                let mut w = 0.0f32;
                                let mut prev = None;
                                for ch in line.chars() {
                                    let gid = font.glyph_id(ch);
                                    if let Some(prev_id) = prev {
                                        w += scaled.kern(prev_id, gid);
                                    }
                                    w += scaled.h_advance(gid);
                                    prev = Some(gid);
                                }
                                max_w = max_w.max(w);
                            }
                            max_w
                        } else {
                            font_size * 2.0
                        };

                        let text_h = num_lines as f32 * line_height;
                        let pad = 4.0; // canvas pixels of padding

                        // Border rect in canvas coords — alignment-aware
                        let border_x = {
                            use crate::ops::text::TextAlignment;
                            match self.text_state.alignment {
                                TextAlignment::Left => origin[0] - pad,
                                TextAlignment::Center => origin[0] - max_width * 0.5 - pad,
                                TextAlignment::Right => origin[0] - max_width - pad,
                            }
                        };
                        let border_y = origin[1] - pad;
                        let border_w = max_width + pad * 2.0;
                        let border_h = text_h + pad * 2.0;

                        // Convert to screen
                        let sx = canvas_rect.min.x + border_x * zoom;
                        let sy = canvas_rect.min.y + border_y * zoom;
                        let sw = border_w * zoom;
                        let sh = border_h * zoom;

                        let border_rect =
                            egui::Rect::from_min_size(Pos2::new(sx, sy), egui::vec2(sw, sh));

                        // Use theme accent color for border/handle
                        let accent = ui.visuals().selection.bg_fill;
                        let accent_semi = Color32::from_rgba_unmultiplied(
                            accent.r(),
                            accent.g(),
                            accent.b(),
                            180,
                        );
                        let accent_fill = Color32::from_rgba_unmultiplied(
                            accent.r(),
                            accent.g(),
                            accent.b(),
                            200,
                        );
                        let accent_faint = Color32::from_rgba_unmultiplied(
                            accent.r(),
                            accent.g(),
                            accent.b(),
                            120,
                        );

                        // Draw dashed border outline
                        let border_stroke = egui::Stroke::new(1.0, accent_semi);
                        painter.rect_stroke(border_rect, 0.0, border_stroke);

                        // Draw move handle — alignment-aware position
                        let (handle_screen_x, handle_screen_y) = {
                            use crate::ops::text::TextAlignment;
                            match self.text_state.alignment {
                                TextAlignment::Left => {
                                    let hx = canvas_rect.min.x
                                        + (origin[0] - handle_offset_canvas) * zoom;
                                    let hy =
                                        canvas_rect.min.y + (origin[1] + font_size * 0.5) * zoom;
                                    (hx, hy)
                                }
                                TextAlignment::Center => {
                                    let hx = canvas_rect.min.x + origin[0] * zoom;
                                    let hy = canvas_rect.min.y
                                        + (origin[1] - handle_offset_canvas) * zoom;
                                    (hx, hy)
                                }
                                TextAlignment::Right => {
                                    let hx = canvas_rect.min.x
                                        + (origin[0] + handle_offset_canvas) * zoom;
                                    let hy =
                                        canvas_rect.min.y + (origin[1] + font_size * 0.5) * zoom;
                                    (hx, hy)
                                }
                            }
                        };
                        let handle_pos = Pos2::new(handle_screen_x, handle_screen_y);

                        painter.circle_filled(handle_pos, handle_radius_screen, accent_fill);
                        painter.circle_stroke(
                            handle_pos,
                            handle_radius_screen,
                            egui::Stroke::new(1.5, Color32::WHITE),
                        );
                        // Draw crosshair inside handle
                        let hs = 3.0;
                        let cross_stroke = egui::Stroke::new(1.0, Color32::WHITE);
                        painter.line_segment(
                            [
                                Pos2::new(handle_pos.x - hs, handle_pos.y),
                                Pos2::new(handle_pos.x + hs, handle_pos.y),
                            ],
                            cross_stroke,
                        );
                        painter.line_segment(
                            [
                                Pos2::new(handle_pos.x, handle_pos.y - hs),
                                Pos2::new(handle_pos.x, handle_pos.y + hs),
                            ],
                            cross_stroke,
                        );
                        // Line from handle to nearest edge of the border rect
                        let connector_target = {
                            use crate::ops::text::TextAlignment;
                            match self.text_state.alignment {
                                TextAlignment::Left => {
                                    // Connect to left edge midpoint
                                    Pos2::new(
                                        border_rect.min.x,
                                        canvas_rect.min.y + (origin[1] + font_size * 0.5) * zoom,
                                    )
                                }
                                TextAlignment::Center => {
                                    // Connect to top edge midpoint
                                    Pos2::new(
                                        canvas_rect.min.x + origin[0] * zoom,
                                        border_rect.min.y,
                                    )
                                }
                                TextAlignment::Right => {
                                    // Connect to right edge midpoint
                                    Pos2::new(
                                        border_rect.max.x,
                                        canvas_rect.min.y + (origin[1] + font_size * 0.5) * zoom,
                                    )
                                }
                            }
                        };
                        painter.line_segment(
                            [handle_pos, connector_target],
                            egui::Stroke::new(1.0, accent_faint),
                        );

                        // Draw blinking cursor
                        let blink_on = ((time * 2.0) as i32) % 2 == 0;
                        if blink_on {
                            // Opt 7: Use cached line advances from rasterization instead of recomputing
                            let (cursor_x_offset, cursor_line) = if !self.text_state.text.is_empty()
                                && self.text_state.cursor_pos > 0
                            {
                                let text_before =
                                    &self.text_state.text[..self.text_state.cursor_pos];
                                let newlines_before = text_before.matches('\n').count();
                                let last_line =
                                    text_before.rsplit('\n').next().unwrap_or(text_before);
                                let char_count = last_line.chars().count();

                                let x_off = if !self.text_state.cached_line_advances.is_empty() {
                                    self.text_state
                                        .cached_line_advances
                                        .get(newlines_before)
                                        .and_then(|advances| advances.get(char_count).copied())
                                        .unwrap_or(0.0)
                                } else if let Some(ref font) = self.text_state.loaded_font {
                                    // Fallback if cache empty
                                    use ab_glyph::{Font as _, ScaleFont as _};
                                    let scaled = font.as_scaled(font_size);
                                    let mut x = 0.0f32;
                                    let mut prev_glyph_id = None;
                                    for ch in last_line.chars() {
                                        let glyph_id = font.glyph_id(ch);
                                        if let Some(prev) = prev_glyph_id {
                                            x += scaled.kern(prev, glyph_id);
                                        }
                                        x += scaled.h_advance(glyph_id);
                                        prev_glyph_id = Some(glyph_id);
                                    }
                                    x
                                } else {
                                    0.0
                                };
                                (x_off, newlines_before)
                            } else {
                                (0.0, 0)
                            };

                            let cached_lh = if self.text_state.cached_line_height > 0.0 {
                                self.text_state.cached_line_height
                            } else {
                                line_height
                            };
                            let cursor_y_offset = cursor_line as f32 * cached_lh;

                            // Apply alignment offset to cursor position
                            let align_offset = {
                                use crate::ops::text::TextAlignment;
                                if self.text_state.alignment == TextAlignment::Left {
                                    0.0
                                } else {
                                    // Get the full width of the current line
                                    let current_line_text = lines.get(cursor_line).unwrap_or(&"");
                                    let line_w = if let Some(ref font) = self.text_state.loaded_font
                                    {
                                        use ab_glyph::{Font as _, ScaleFont as _};
                                        let scaled = font.as_scaled(font_size);
                                        let mut w = 0.0f32;
                                        let mut prev = None;
                                        for ch in current_line_text.chars() {
                                            let gid = font.glyph_id(ch);
                                            if let Some(prev_id) = prev {
                                                w += scaled.kern(prev_id, gid);
                                            }
                                            w += scaled.h_advance(gid);
                                            prev = Some(gid);
                                        }
                                        w
                                    } else {
                                        0.0
                                    };
                                    match self.text_state.alignment {
                                        TextAlignment::Center => -line_w * 0.5,
                                        TextAlignment::Right => -line_w,
                                        _ => 0.0,
                                    }
                                }
                            };

                            let cx = origin[0] + cursor_x_offset + align_offset;
                            let cy = origin[1] + cursor_y_offset;

                            let csx = canvas_rect.min.x + cx * zoom;
                            let csy = canvas_rect.min.y + cy * zoom;
                            let cursor_h = font_size * zoom;

                            painter.line_segment(
                                [Pos2::new(csx, csy), Pos2::new(csx, csy + cursor_h)],
                                egui::Stroke::new(1.5, Color32::BLACK),
                            );
                            painter.line_segment(
                                [
                                    Pos2::new(csx + 1.0, csy),
                                    Pos2::new(csx + 1.0, csy + cursor_h),
                                ],
                                egui::Stroke::new(0.5, Color32::WHITE),
                            );
                        }
                        // Opt 6: Throttle repaint to cursor blink rate (2Hz) instead of 60fps
                        ui.ctx()
                            .request_repaint_after(std::time::Duration::from_millis(500));
                    }
                }
            }

            // ================================================================
            // LIQUIFY TOOL — click+drag to push/pull pixels
            // ================================================================
            Tool::Liquify => {
                let escape_pressed = ui.input(|i| i.key_pressed(egui::Key::Escape));

                // Escape: reset
                if escape_pressed && self.liquify_state.is_active {
                    self.liquify_state.displacement = None;
                    self.liquify_state.is_active = false;
                    self.liquify_state.source_snapshot = None;
                    self.liquify_state.warp_buffer.clear();
                    self.liquify_state.dirty_rect = None;
                    canvas_state.clear_preview_state();
                    canvas_state.mark_dirty(None);
                }

                // Enter: commit
                if enter_pressed && self.liquify_state.is_active {
                    self.liquify_state.commit_pending = true;
                    self.liquify_state.commit_pending_frame = 0;
                }

                // Mouse drag: apply displacement
                if is_primary_down {
                    if let Some(pos_f) = canvas_pos_f32 {
                        // Initialize on first use
                        if !self.liquify_state.is_active {
                            self.liquify_state.is_active = true;
                            if let Some(layer) =
                                canvas_state.layers.get(canvas_state.active_layer_index)
                            {
                                self.liquify_state.source_snapshot =
                                    Some(layer.pixels.to_rgba_image());
                            }
                            self.liquify_state.displacement =
                                Some(crate::ops::transform::DisplacementField::new(
                                    canvas_state.width,
                                    canvas_state.height,
                                ));
                            self.stroke_tracker
                                .start_preview_tool(canvas_state.active_layer_index, "Liquify");
                            // Tell GPU pipeline the source snapshot changed
                            if let Some(ref mut gpu) = gpu_renderer {
                                gpu.liquify_pipeline.invalidate_source();
                            }
                        }

                        if let Some(ref mut disp) = self.liquify_state.displacement {
                            let radius = self.properties.size;
                            let cx = pos_f.0;
                            let cy = pos_f.1;

                            match self.liquify_state.mode {
                                LiquifyMode::Push => {
                                    // Get delta from last position
                                    if let Some(last) = self.liquify_state.last_pos {
                                        let dx = cx - last[0];
                                        let dy = cy - last[1];
                                        if dx.abs() > 0.1 || dy.abs() > 0.1 {
                                            disp.apply_push(
                                                cx,
                                                cy,
                                                dx * self.liquify_state.strength,
                                                dy * self.liquify_state.strength,
                                                radius,
                                                self.liquify_state.strength,
                                            );
                                        }
                                    }
                                }
                                LiquifyMode::Expand => {
                                    disp.apply_expand(
                                        cx,
                                        cy,
                                        radius,
                                        self.liquify_state.strength * 2.0,
                                    );
                                }
                                LiquifyMode::Contract => {
                                    disp.apply_contract(
                                        cx,
                                        cy,
                                        radius,
                                        self.liquify_state.strength * 2.0,
                                    );
                                }
                                LiquifyMode::TwirlCW => {
                                    disp.apply_twirl(
                                        cx,
                                        cy,
                                        radius,
                                        self.liquify_state.strength * 0.05,
                                        true,
                                    );
                                }
                                LiquifyMode::TwirlCCW => {
                                    disp.apply_twirl(
                                        cx,
                                        cy,
                                        radius,
                                        self.liquify_state.strength * 0.05,
                                        false,
                                    );
                                }
                            }

                            // Update dirty rect
                            let r = radius as i32 + 2;
                            let new_dirty =
                                [cx as i32 - r, cy as i32 - r, cx as i32 + r, cy as i32 + r];
                            self.liquify_state.dirty_rect =
                                Some(match self.liquify_state.dirty_rect {
                                    Some(d) => [
                                        d[0].min(new_dirty[0]),
                                        d[1].min(new_dirty[1]),
                                        d[2].max(new_dirty[2]),
                                        d[3].max(new_dirty[3]),
                                    ],
                                    None => new_dirty,
                                });

                            self.liquify_state.last_pos = Some([cx, cy]);
                            self.render_liquify_preview(canvas_state, gpu_renderer.as_deref_mut());
                        }
                    }
                } else if self.liquify_state.last_pos.is_some() {
                    self.liquify_state.last_pos = None;
                    self.liquify_state.dirty_rect = None;
                }
            }

            // ================================================================
            // MESH WARP TOOL — drag control points to warp image
            // ================================================================
            Tool::MeshWarp => {
                let escape_pressed = ui.input(|i| i.key_pressed(egui::Key::Escape));

                // Escape: reset to original
                if escape_pressed && self.mesh_warp_state.is_active {
                    self.mesh_warp_state.points = self.mesh_warp_state.original_points.clone();
                }

                // Enter: commit
                if enter_pressed && self.mesh_warp_state.is_active {
                    self.mesh_warp_state.commit_pending = true;
                    self.mesh_warp_state.commit_pending_frame = 0;
                }

                // Auto-initialize grid as soon as tool is selected
                if !self.mesh_warp_state.is_active {
                    self.init_mesh_warp_grid(canvas_state);
                    self.stroke_tracker
                        .start_preview_tool(canvas_state.active_layer_index, "Mesh Warp");
                } else {
                    // Detect stale snapshot (layer changed since snapshot was taken)
                    let cur_layer_idx = canvas_state.active_layer_index;
                    let cur_gen = canvas_state
                        .layers
                        .get(cur_layer_idx)
                        .map(|l| l.gpu_generation)
                        .unwrap_or(0);
                    if cur_layer_idx != self.mesh_warp_state.snapshot_layer_index
                        || cur_gen != self.mesh_warp_state.snapshot_generation
                    {
                        // Re-snapshot: layer content or active layer changed
                        self.init_mesh_warp_grid(canvas_state);
                        self.stroke_tracker
                            .start_preview_tool(cur_layer_idx, "Mesh Warp");
                    }
                }
                if self.mesh_warp_state.is_active {
                    // Find nearest control point for hover/drag
                    // Use clamped coordinates so edge/corner handles are reachable
                    let mesh_pos = canvas_pos_f32_clamped.or(canvas_pos_f32);
                    if let Some(pos_f) = mesh_pos {
                        let hit_radius = 12.0 / zoom; // screen pixels → canvas pixels

                        if is_primary_pressed {
                            // Start drag
                            let mut best_idx = None;
                            let mut best_dist = f32::MAX;
                            for (i, pt) in self.mesh_warp_state.points.iter().enumerate() {
                                let dx = pos_f.0 - pt[0];
                                let dy = pos_f.1 - pt[1];
                                let dist = (dx * dx + dy * dy).sqrt();
                                if dist < hit_radius && dist < best_dist {
                                    best_dist = dist;
                                    best_idx = Some(i);
                                }
                            }
                            self.mesh_warp_state.dragging_index = best_idx;
                        }

                        if is_primary_down && let Some(idx) = self.mesh_warp_state.dragging_index {
                            self.mesh_warp_state.points[idx] = [pos_f.0, pos_f.1];
                        }

                        if is_primary_released {
                            self.mesh_warp_state.dragging_index = None;
                        }

                        // Hover detection
                        if !is_primary_down {
                            let mut hover = None;
                            let mut best_dist = f32::MAX;
                            for (i, pt) in self.mesh_warp_state.points.iter().enumerate() {
                                let dx = pos_f.0 - pt[0];
                                let dy = pos_f.1 - pt[1];
                                let dist = (dx * dx + dy * dy).sqrt();
                                if dist < hit_radius && dist < best_dist {
                                    best_dist = dist;
                                    hover = Some(i);
                                }
                            }
                            self.mesh_warp_state.hover_index = hover;
                        }
                    }

                    // Draw overlay (grid + handles) — no live preview, warp only on commit
                    self.draw_mesh_warp_overlay(ui, painter, canvas_rect, zoom, canvas_state);
                }
            }

            // ================================================================
            // COLOR REMOVER TOOL — click to remove color
            // ================================================================
            Tool::ColorRemover => {
                if is_primary_clicked && let Some(pos) = canvas_pos {
                    self.commit_color_removal(canvas_state, pos.0, pos.1);
                }
            }

            // ================================================================
            // SMUDGE TOOL — drag to smear/blend canvas pixels
            // ================================================================
            Tool::Smudge => {
                if is_primary_pressed {
                    // Stroke start: pick up the color at the start position
                    if let Some(pos) = canvas_pos {
                        if let Some(layer) =
                            canvas_state.layers.get(canvas_state.active_layer_index)
                        {
                            let px = layer.pixels.get_pixel(
                                pos.0.min(canvas_state.width.saturating_sub(1)),
                                pos.1.min(canvas_state.height.saturating_sub(1)),
                            );
                            self.smudge_state.pickup_color =
                                [px[0] as f32, px[1] as f32, px[2] as f32, px[3] as f32];
                        }
                        self.smudge_state.is_stroking = true;
                        let layer_pixels = canvas_state
                            .layers
                            .get(canvas_state.active_layer_index)
                            .map(|l| l.pixels.clone());
                        if let Some(pixels) = layer_pixels {
                            self.stroke_tracker.start_direct_tool(
                                canvas_state.active_layer_index,
                                "Smudge",
                                &pixels,
                            );
                        }
                    }
                }
                if is_primary_down
                    && self.smudge_state.is_stroking
                    && let Some(pos) = canvas_pos
                {
                    let radius = (self.properties.size * 0.5).max(1.0);
                    self.draw_smudge_no_dirty(canvas_state, pos.0, pos.1);
                    let dirty = egui::Rect::from_min_max(
                        egui::pos2(
                            (pos.0 as f32 - radius - 2.0).max(0.0),
                            (pos.1 as f32 - radius - 2.0).max(0.0),
                        ),
                        egui::pos2(pos.0 as f32 + radius + 2.0, pos.1 as f32 + radius + 2.0),
                    );
                    self.stroke_tracker.expand_bounds(dirty);
                    canvas_state.mark_dirty(Some(dirty));
                }
                if is_primary_released && self.smudge_state.is_stroking {
                    self.smudge_state.is_stroking = false;
                    if let Some(se) = self.stroke_tracker.finish(canvas_state) {
                        self.pending_stroke_event = Some(se);
                    }
                }
            }

            // ================================================================
            // SHAPES TOOL — click+drag to draw, then adjust
            // ================================================================
            Tool::Shapes => {
                let escape_pressed = ui.input(|i| i.key_pressed(egui::Key::Escape));

                // Escape: cancel
                if escape_pressed {
                    if self.shapes_state.placed.is_some() {
                        self.shapes_state.placed = None;
                        canvas_state.clear_preview_state();
                        canvas_state.mark_dirty(None);
                    } else if self.shapes_state.is_drawing {
                        self.shapes_state.is_drawing = false;
                        self.shapes_state.draw_start = None;
                        self.shapes_state.draw_end = None;
                        canvas_state.clear_preview_state();
                        canvas_state.mark_dirty(None);
                    }
                }

                // Enter: commit placed shape
                if enter_pressed && self.shapes_state.placed.is_some() {
                    self.stroke_tracker
                        .start_preview_tool(canvas_state.active_layer_index, "Shape");
                    self.commit_shape(canvas_state);
                }

                if let Some(pos_f) = canvas_pos_f32 {
                    if self.shapes_state.placed.is_some() {
                        // Shape is placed — handle move/resize/rotate
                        let hit_radius = 10.0 / zoom;

                        // Determine what was hit on press
                        let mut should_commit = false;
                        let mut need_preview = false;

                        if is_primary_pressed {
                            use crate::ops::shapes::ShapeHandle;
                            let placed = self.shapes_state.placed.as_ref().unwrap();
                            let cos_r = placed.rotation.cos();
                            let sin_r = placed.rotation.sin();

                            let corners_local = [
                                (-placed.hw, -placed.hh),
                                (placed.hw, -placed.hh),
                                (placed.hw, placed.hh),
                                (-placed.hw, placed.hh),
                            ];
                            let corners_canvas: Vec<(f32, f32)> = corners_local
                                .iter()
                                .map(|(cx, cy)| {
                                    (
                                        cx * cos_r - cy * sin_r + placed.cx,
                                        cx * sin_r + cy * cos_r + placed.cy,
                                    )
                                })
                                .collect();

                            let top_mid_x = (corners_canvas[0].0 + corners_canvas[1].0) * 0.5;
                            let top_mid_y = (corners_canvas[0].1 + corners_canvas[1].1) * 0.5;
                            let rot_offset = 20.0 / zoom;
                            let dir_x = top_mid_x - placed.cx;
                            let dir_y = top_mid_y - placed.cy;
                            let dir_len = (dir_x * dir_x + dir_y * dir_y).sqrt().max(0.001);
                            let rot_handle_x = top_mid_x + (dir_x / dir_len) * rot_offset;
                            let rot_handle_y = top_mid_y + (dir_y / dir_len) * rot_offset;

                            let handle_names = [
                                ShapeHandle::TopLeft,
                                ShapeHandle::TopRight,
                                ShapeHandle::BottomRight,
                                ShapeHandle::BottomLeft,
                            ];
                            let mut hit: Option<ShapeHandle> = None;

                            // Check rotation handle
                            let dx = pos_f.0 - rot_handle_x;
                            let dy = pos_f.1 - rot_handle_y;
                            if (dx * dx + dy * dy).sqrt() < hit_radius {
                                hit = Some(ShapeHandle::Rotate);
                            }

                            // Check corners
                            if hit.is_none() {
                                for (i, &(cx, cy)) in corners_canvas.iter().enumerate() {
                                    let dx = pos_f.0 - cx;
                                    let dy = pos_f.1 - cy;
                                    if (dx * dx + dy * dy).sqrt() < hit_radius {
                                        hit = Some(handle_names[i]);
                                        break;
                                    }
                                }
                            }

                            // Check inside shape for move
                            if hit.is_none() {
                                let dx = pos_f.0 - placed.cx;
                                let dy = pos_f.1 - placed.cy;
                                let lx = (dx * cos_r + dy * sin_r).abs();
                                let ly = (-dx * sin_r + dy * cos_r).abs();
                                if lx <= placed.hw + hit_radius && ly <= placed.hh + hit_radius {
                                    hit = Some(ShapeHandle::Move);
                                }
                            }

                            let pcx = placed.cx;
                            let pcy = placed.cy;
                            let prot = placed.rotation;
                            // Drop the immutable borrow before mutable access below
                            let _ = placed;

                            if let Some(h) = hit {
                                let p = self.shapes_state.placed.as_mut().unwrap();
                                p.handle_dragging = Some(h);
                                p.drag_offset = [pos_f.0 - pcx, pos_f.1 - pcy];
                                if h == ShapeHandle::Rotate {
                                    p.rotate_start_angle = (pos_f.1 - pcy).atan2(pos_f.0 - pcx);
                                    p.rotate_start_rotation = prot;
                                }
                                // For corner resize: compute anchor = opposite corner in canvas coords
                                let cos_r = prot.cos();
                                let sin_r = prot.sin();
                                let (anchor_lx, anchor_ly) = match h {
                                    ShapeHandle::TopLeft => (p.hw, p.hh),
                                    ShapeHandle::TopRight => (-p.hw, p.hh),
                                    ShapeHandle::BottomRight => (-p.hw, -p.hh),
                                    ShapeHandle::BottomLeft => (p.hw, -p.hh),
                                    _ => (0.0, 0.0),
                                };
                                p.drag_anchor = [
                                    anchor_lx * cos_r - anchor_ly * sin_r + pcx,
                                    anchor_lx * sin_r + anchor_ly * cos_r + pcy,
                                ];
                            } else {
                                should_commit = true;
                            }
                        }

                        if should_commit {
                            self.stroke_tracker
                                .start_preview_tool(canvas_state.active_layer_index, "Shape");
                            self.commit_shape(canvas_state);
                            if self.pending_stroke_event.is_none()
                                && let Some(evt) = stroke_event.take()
                            {
                                self.pending_stroke_event = Some(evt);
                            }
                        }

                        if is_primary_down && !should_commit {
                            use crate::ops::shapes::ShapeHandle;
                            if let Some(ref mut p) = self.shapes_state.placed {
                                match p.handle_dragging {
                                    Some(ShapeHandle::Move) => {
                                        p.cx = pos_f.0 - p.drag_offset[0];
                                        p.cy = pos_f.1 - p.drag_offset[1];
                                        need_preview = true;
                                    }
                                    Some(ShapeHandle::Rotate) => {
                                        let angle = (pos_f.1 - p.cy).atan2(pos_f.0 - p.cx);
                                        let mut new_rot = p.rotate_start_rotation
                                            + (angle - p.rotate_start_angle);
                                        // Shift: snap to 45° increments
                                        if shift_held {
                                            let snap = std::f32::consts::FRAC_PI_4; // 45°
                                            new_rot = (new_rot / snap).round() * snap;
                                        }
                                        p.rotation = new_rot;
                                        need_preview = true;
                                    }
                                    Some(_handle) => {
                                        // Corner resize: anchor is opposite corner
                                        let ax = p.drag_anchor[0];
                                        let ay = p.drag_anchor[1];
                                        let mx = pos_f.0;
                                        let my = pos_f.1;
                                        // Half-sizes in local (rotated) space
                                        let cos_r = p.rotation.cos();
                                        let sin_r = p.rotation.sin();
                                        let dx = (mx - ax) * 0.5;
                                        let dy = (my - ay) * 0.5;
                                        let mut lx = (dx * cos_r + dy * sin_r).abs();
                                        let mut ly = (-dx * sin_r + dy * cos_r).abs();
                                        // Shift: constrain to 1:1 aspect ratio
                                        if shift_held {
                                            let side = lx.max(ly);
                                            lx = side;
                                            ly = side;
                                        }
                                        // Recompute center from anchor + constrained size
                                        // The dragged corner in local space is the negation of the anchor's local offset
                                        let (anchor_lx, anchor_ly) = match _handle {
                                            ShapeHandle::TopLeft => (lx, ly),
                                            ShapeHandle::TopRight => (-lx, ly),
                                            ShapeHandle::BottomRight => (-lx, -ly),
                                            ShapeHandle::BottomLeft => (lx, -ly),
                                            _ => (0.0, 0.0),
                                        };
                                        // Anchor canvas pos is known; center = anchor + rotated(-anchor_local)
                                        let neg_lx = -anchor_lx;
                                        let _neg_ly = -anchor_ly;
                                        let half_diag_x = (anchor_lx + neg_lx) * 0.5; // = 0
                                        let _ = half_diag_x;
                                        // Center = anchor + rotate((-anchor_lx + lx_sign*lx), ...) * 0.5  ... simpler:
                                        // anchor is at (anchor_lx) in local; center is at (0,0) in local
                                        // so center_canvas = anchor_canvas + rotate(-anchor_lx, -anchor_ly)
                                        let new_cx =
                                            ax + (-anchor_lx) * cos_r - (-anchor_ly) * sin_r;
                                        let new_cy =
                                            ay + (-anchor_lx) * sin_r + (-anchor_ly) * cos_r;
                                        p.cx = new_cx;
                                        p.cy = new_cy;
                                        p.hw = lx.max(2.0);
                                        p.hh = ly.max(2.0);
                                        need_preview = true;
                                    }
                                    None => {}
                                }
                            }
                        }

                        if need_preview {
                            self.render_shape_preview(
                                canvas_state,
                                primary_color_f32,
                                secondary_color_f32,
                            );
                        }

                        if is_primary_released && let Some(ref mut p) = self.shapes_state.placed {
                            p.handle_dragging = None;
                        }
                    } else {
                        // Drawing mode
                        if is_primary_pressed && !self.shapes_state.is_drawing {
                            self.shapes_state.is_drawing = true;
                            self.shapes_state.draw_start = Some([pos_f.0, pos_f.1]);
                            self.shapes_state.draw_end = Some([pos_f.0, pos_f.1]);
                            self.stroke_tracker
                                .start_preview_tool(canvas_state.active_layer_index, "Shape");
                        }

                        if is_primary_down && self.shapes_state.is_drawing {
                            let mut end = [pos_f.0, pos_f.1];
                            // Shift: constrain to square/circle
                            if shift_held && let Some(start) = self.shapes_state.draw_start {
                                let dx = (end[0] - start[0]).abs();
                                let dy = (end[1] - start[1]).abs();
                                let side = dx.max(dy);
                                end[0] = start[0] + side * (end[0] - start[0]).signum();
                                end[1] = start[1] + side * (end[1] - start[1]).signum();
                            }
                            self.shapes_state.draw_end = Some(end);
                            self.render_shape_preview(
                                canvas_state,
                                primary_color_f32,
                                secondary_color_f32,
                            );
                        }

                        if is_primary_released && self.shapes_state.is_drawing {
                            self.shapes_state.is_drawing = false;
                            // Convert to placed shape for manipulation
                            if let (Some(start), Some(end)) =
                                (self.shapes_state.draw_start, self.shapes_state.draw_end)
                            {
                                let cx = (start[0] + end[0]) * 0.5;
                                let cy = (start[1] + end[1]) * 0.5;
                                let hw = ((end[0] - start[0]) * 0.5).abs();
                                let hh = ((end[1] - start[1]) * 0.5).abs();
                                if hw > 2.0 && hh > 2.0 {
                                    let primary = [
                                        (primary_color_f32[0] * 255.0) as u8,
                                        (primary_color_f32[1] * 255.0) as u8,
                                        (primary_color_f32[2] * 255.0) as u8,
                                        (primary_color_f32[3] * 255.0) as u8,
                                    ];
                                    let secondary = [
                                        (secondary_color_f32[0] * 255.0) as u8,
                                        (secondary_color_f32[1] * 255.0) as u8,
                                        (secondary_color_f32[2] * 255.0) as u8,
                                        (secondary_color_f32[3] * 255.0) as u8,
                                    ];
                                    self.shapes_state.placed =
                                        Some(crate::ops::shapes::PlacedShape {
                                            cx,
                                            cy,
                                            hw,
                                            hh,
                                            rotation: 0.0,
                                            kind: self.shapes_state.selected_shape,
                                            fill_mode: self.shapes_state.fill_mode,
                                            outline_width: self.properties.size,
                                            primary_color: primary,
                                            secondary_color: secondary,
                                            anti_alias: self.shapes_state.anti_alias,
                                            corner_radius: self.shapes_state.corner_radius,
                                            handle_dragging: None,
                                            drag_offset: [0.0, 0.0],
                                            drag_anchor: [0.0, 0.0],
                                            rotate_start_angle: 0.0,
                                            rotate_start_rotation: 0.0,
                                        });
                                    self.render_shape_preview(
                                        canvas_state,
                                        primary_color_f32,
                                        secondary_color_f32,
                                    );
                                } else {
                                    self.shapes_state.draw_start = None;
                                    self.shapes_state.draw_end = None;
                                    canvas_state.clear_preview_state();
                                }
                            }
                        }
                    }
                }

                // Property sync is handled by update_shape_if_dirty() called from canvas update loop

                // Draw shape overlay (bounding box, handles)
                if self.shapes_state.placed.is_some() {
                    self.draw_shape_overlay(painter, canvas_rect, zoom);
                }
            }

            // ================================================================
            // GRADIENT TOOL — click+drag to define gradient direction
            // ================================================================
            Tool::Gradient => {
                let escape_pressed = ui.input(|i| i.key_pressed(egui::Key::Escape));

                // Escape cancels active gradient
                if escape_pressed && self.gradient_state.drag_start.is_some() {
                    self.cancel_gradient(canvas_state);
                }

                // Enter commits active gradient  (deferred for loading bar)
                if enter_pressed
                    && self.gradient_state.drag_start.is_some()
                    && !self.gradient_state.dragging
                {
                    self.gradient_state.commit_pending = true;
                    self.gradient_state.commit_pending_frame = 0;
                }

                // Update stops from primary/secondary if using PrimarySecondary preset
                // and colors have changed
                if self.gradient_state.preset == GradientPreset::PrimarySecondary {
                    let p = [
                        (primary_color_f32[0] * 255.0) as u8,
                        (primary_color_f32[1] * 255.0) as u8,
                        (primary_color_f32[2] * 255.0) as u8,
                        (primary_color_f32[3] * 255.0) as u8,
                    ];
                    let s = [
                        (secondary_color_f32[0] * 255.0) as u8,
                        (secondary_color_f32[1] * 255.0) as u8,
                        (secondary_color_f32[2] * 255.0) as u8,
                        (secondary_color_f32[3] * 255.0) as u8,
                    ];
                    if self.gradient_state.stops.len() >= 2
                        && (self.gradient_state.stops[0].color != p
                            || self.gradient_state.stops[self.gradient_state.stops.len() - 1].color
                                != s)
                    {
                        self.gradient_state.stops[0] =
                            GradientStop::new(self.gradient_state.stops[0].position, p);
                        let last = self.gradient_state.stops.len() - 1;
                        self.gradient_state.stops[last] =
                            GradientStop::new(self.gradient_state.stops[last].position, s);
                        self.gradient_state.lut_dirty = true;
                    }
                } else if self.gradient_state.preset == GradientPreset::ForegroundTransparent {
                    let p = [
                        (primary_color_f32[0] * 255.0) as u8,
                        (primary_color_f32[1] * 255.0) as u8,
                        (primary_color_f32[2] * 255.0) as u8,
                        (primary_color_f32[3] * 255.0) as u8,
                    ];
                    if self.gradient_state.stops.len() >= 2
                        && self.gradient_state.stops[0].color != p
                    {
                        self.gradient_state.stops[0] =
                            GradientStop::new(self.gradient_state.stops[0].position, p);
                        let last = self.gradient_state.stops.len() - 1;
                        self.gradient_state.stops[last] = GradientStop::new(
                            self.gradient_state.stops[last].position,
                            [p[0], p[1], p[2], 0],
                        );
                        self.gradient_state.lut_dirty = true;
                    }
                }

                // Only start/grab gradient on clicks that are actually on the canvas
                let pointer_on_canvas = ui.input(|i| {
                    i.pointer
                        .interact_pos()
                        .is_some_and(|pos| canvas_rect.contains(pos))
                });

                if is_primary_pressed
                    && pointer_on_canvas
                    && let Some(pos_f) = canvas_pos_f32
                {
                    let click_pos = Pos2::new(pos_f.0, pos_f.1);

                    // Check if clicking near an existing handle first
                    let handle_grab_radius = (12.0 / zoom).max(6.0); // screen-space ~12px
                    let mut grabbed_handle: Option<usize> = None;

                    if let Some(start) = self.gradient_state.drag_start
                        && (click_pos - start).length() < handle_grab_radius
                    {
                        grabbed_handle = Some(0);
                    }
                    if let Some(end) = self.gradient_state.drag_end
                        && (click_pos - end).length() < handle_grab_radius
                    {
                        grabbed_handle = Some(1);
                    }

                    if let Some(handle_idx) = grabbed_handle {
                        // Grab existing handle for repositioning
                        self.gradient_state.dragging = true;
                        self.gradient_state.dragging_handle = Some(handle_idx);
                    } else {
                        // No handle hit — commit previous and start new gradient.
                        // Immediate commit here (no defer) because the new
                        // gradient's drag_start is set on the same frame and
                        // commit_gradient clears it.  The old preview was
                        // already rendered at full res on mouse release.
                        if self.gradient_state.drag_start.is_some() {
                            self.commit_gradient(canvas_state);
                        }

                        self.gradient_state.drag_start = Some(click_pos);
                        self.gradient_state.drag_end = Some(click_pos);
                        self.gradient_state.dragging = true;
                        self.gradient_state.dragging_handle = None;

                        // Start stroke tracking
                        self.stroke_tracker
                            .start_preview_tool(canvas_state.active_layer_index, "Gradient");
                        self.stroke_tracker.expand_bounds(egui::Rect::from_min_max(
                            egui::pos2(0.0, 0.0),
                            egui::pos2(canvas_state.width as f32, canvas_state.height as f32),
                        ));

                        if self.gradient_state.lut_dirty {
                            self.gradient_state.rebuild_lut();
                        }
                    }
                }

                if is_primary_down
                    && self.gradient_state.dragging
                    && let Some(pos_f) = canvas_pos_f32
                {
                    let new_pos = Pos2::new(pos_f.0, pos_f.1);
                    match self.gradient_state.dragging_handle {
                        Some(0) => self.gradient_state.drag_start = Some(new_pos),
                        Some(1) => self.gradient_state.drag_end = Some(new_pos),
                        _ => self.gradient_state.drag_end = Some(new_pos),
                    }
                    self.render_gradient_to_preview(canvas_state, gpu_renderer.as_deref_mut());
                    ui.ctx().request_repaint();
                }

                if is_primary_released && self.gradient_state.dragging {
                    // Save handle index BEFORE clearing it so we update the correct handle
                    let released_handle = self.gradient_state.dragging_handle;
                    self.gradient_state.dragging = false;
                    self.gradient_state.dragging_handle = None;
                    if let Some(pos_f) = canvas_pos_f32 {
                        let new_pos = Pos2::new(pos_f.0, pos_f.1);
                        match released_handle {
                            Some(0) => self.gradient_state.drag_start = Some(new_pos),
                            Some(1) => self.gradient_state.drag_end = Some(new_pos),
                            _ => self.gradient_state.drag_end = Some(new_pos),
                        }
                        self.render_gradient_to_preview(canvas_state, gpu_renderer.as_deref_mut());
                    }
                }

                // If gradient is active (not dragging) and LUT changed, re-render
                if !self.gradient_state.dragging
                    && self.gradient_state.drag_start.is_some()
                    && self.gradient_state.lut_dirty
                {
                    self.render_gradient_to_preview(canvas_state, gpu_renderer);
                    ui.ctx().request_repaint();
                }
                // preview_dirty is handled separately via update_gradient_if_dirty()
                // so it works even when handle_input is blocked by UI interactions

                // Draw overlay handles
                self.draw_gradient_overlay(painter, canvas_rect, zoom, canvas_state);
            }

            // ================================================================
            // CLONE STAMP — Alt+click to set source, then paint from offset
            // ================================================================
            Tool::CloneStamp => {
                let is_painting = is_primary_down || is_secondary_down;
                let alt_held = ui.input(|i| i.modifiers.alt);

                // Enter/Escape: clear source point
                if enter_pressed || ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                    self.clone_stamp_state.source = None;
                    self.clone_stamp_state.offset = None;
                    self.clone_stamp_state.offset_locked = false;
                }

                // Alt+Click: set source point
                if alt_held && is_primary_clicked {
                    if let Some(pos_f) = canvas_pos_f32 {
                        self.clone_stamp_state.source = Some(Pos2::new(pos_f.0, pos_f.1));
                        self.clone_stamp_state.offset = None;
                        self.clone_stamp_state.offset_locked = false;
                    }
                } else if !alt_held && self.clone_stamp_state.source.is_some() {
                    // Normal painting with clone source
                    if is_painting {
                        let current_f32 = canvas_pos_f32
                            .or_else(|| canvas_pos.map(|(x, y)| (x as f32, y as f32)));
                        if let (Some(cf), Some(_current_pos)) = (current_f32, canvas_pos) {
                            let current = Pos2::new(cf.0, cf.1);

                            // Lock offset on first paint of this stroke
                            if !self.clone_stamp_state.offset_locked {
                                let src = self.clone_stamp_state.source.unwrap();
                                self.clone_stamp_state.offset =
                                    Some(Vec2::new(src.x - current.x, src.y - current.y));
                                self.clone_stamp_state.offset_locked = true;
                            }

                            // Initialize stroke on first frame
                            if self.tool_state.last_pos.is_none() {
                                self.tool_state.last_precise_pos = Some(current);
                                self.tool_state.distance_remainder = 0.0;
                                self.tool_state.smooth_pos = Some(current);

                                self.stroke_tracker.start_preview_tool(
                                    canvas_state.active_layer_index,
                                    "Clone Stamp",
                                );

                                // Init preview layer
                                if canvas_state.preview_layer.is_none()
                                    || canvas_state.preview_layer.as_ref().unwrap().width()
                                        != canvas_state.width
                                    || canvas_state.preview_layer.as_ref().unwrap().height()
                                        != canvas_state.height
                                {
                                    canvas_state.preview_layer = Some(TiledImage::new(
                                        canvas_state.width,
                                        canvas_state.height,
                                    ));
                                } else if let Some(ref mut preview) = canvas_state.preview_layer {
                                    preview.clear();
                                }
                                canvas_state.preview_blend_mode = BlendMode::Normal;
                            }

                            // Sub-frame events
                            let positions: Vec<(f32, f32)> = if !raw_motion_events.is_empty() {
                                raw_motion_events.to_vec()
                            } else {
                                vec![cf]
                            };

                            // EMA smoothing (same as brush)
                            let smoothed_positions: Vec<(f32, f32)> = {
                                let mut result = Vec::with_capacity(positions.len());
                                for &pos in &positions {
                                    let raw = Pos2::new(pos.0, pos.1);
                                    let smoothed = if let Some(prev) = self.tool_state.smooth_pos {
                                        let dx = raw.x - prev.x;
                                        let dy = raw.y - prev.y;
                                        let dist = (dx * dx + dy * dy).sqrt();
                                        let alpha = if dist < 1.5 {
                                            1.0
                                        } else {
                                            (0.55 + 1.8 / (dist + 1.8)).min(1.0)
                                        };
                                        Pos2::new(prev.x + alpha * dx, prev.y + alpha * dy)
                                    } else {
                                        raw
                                    };
                                    self.tool_state.smooth_pos = Some(smoothed);
                                    result.push((smoothed.x, smoothed.y));
                                }
                                result
                            };

                            let offset = self.clone_stamp_state.offset.unwrap();
                            let mut frame_dirty_rect = Rect::NOTHING;

                            for &pos in &smoothed_positions {
                                let start_precise = self.tool_state.last_precise_pos;
                                let modified_rect = if let Some(start_p) = start_precise {
                                    self.clone_stamp_line(
                                        canvas_state,
                                        (start_p.x, start_p.y),
                                        (pos.0, pos.1),
                                        offset,
                                    )
                                } else {
                                    self.clone_stamp_circle(canvas_state, (pos.0, pos.1), offset)
                                };
                                frame_dirty_rect = frame_dirty_rect.union(modified_rect);
                                self.tool_state.last_precise_pos = Some(Pos2::new(pos.0, pos.1));
                            }

                            self.stroke_tracker.expand_bounds(frame_dirty_rect);
                            if frame_dirty_rect.is_positive() {
                                canvas_state.mark_preview_changed_rect(frame_dirty_rect);
                            }

                            self.tool_state.last_pos = Some(canvas_pos.unwrap());
                            self.tool_state.last_brush_pos = Some(canvas_pos.unwrap());
                            ui.ctx().request_repaint();
                        }
                    } else {
                        // Mouse released — commit
                        if self.tool_state.last_pos.is_some() {
                            stroke_event = self.stroke_tracker.finish(canvas_state);
                            self.commit_bezier_to_layer(canvas_state, primary_color_f32);
                            canvas_state.clear_preview_state();
                            if let Some(ref ev) = stroke_event {
                                canvas_state.mark_dirty(Some(ev.bounds.expand(12.0)));
                            } else {
                                self.mark_full_dirty(canvas_state);
                            }
                        }
                        self.tool_state.last_pos = None;
                        self.tool_state.last_precise_pos = None;
                        self.tool_state.distance_remainder = 0.0;
                        self.tool_state.smooth_pos = None;
                        self.clone_stamp_state.offset_locked = false;
                    }
                }
            }

            // ================================================================
            // CONTENT AWARE BRUSH — healing brush, samples surrounding texture
            // ================================================================
            Tool::ContentAwareBrush => {
                let is_painting = is_primary_down || is_secondary_down;
                if is_painting {
                    let current_f32 =
                        canvas_pos_f32.or_else(|| canvas_pos.map(|(x, y)| (x as f32, y as f32)));
                    if let (Some(cf), Some(_current_pos)) = (current_f32, canvas_pos) {
                        let current = Pos2::new(cf.0, cf.1);

                        // Initialize stroke
                        if self.tool_state.last_pos.is_none() {
                            self.tool_state.last_precise_pos = Some(current);
                            self.tool_state.distance_remainder = 0.0;
                            self.tool_state.smooth_pos = Some(current);
                            self.content_aware_state.stroke_points.clear();

                            // For async modes snapshot original pixels + init hole mask
                            self.content_aware_state.stroke_original = None;
                            self.content_aware_state.hole_mask = None;
                            if self.content_aware_state.quality.is_async() {
                                let idx = canvas_state.active_layer_index;
                                if idx < canvas_state.layers.len() {
                                    self.content_aware_state.stroke_original =
                                        Some(canvas_state.layers[idx].pixels.to_rgba_image());
                                    self.content_aware_state.hole_mask = Some(GrayImage::new(
                                        canvas_state.width,
                                        canvas_state.height,
                                    ));
                                }
                            }

                            self.stroke_tracker
                                .start_preview_tool(canvas_state.active_layer_index, "Heal Brush");

                            // Init preview layer
                            if canvas_state.preview_layer.is_none()
                                || canvas_state.preview_layer.as_ref().unwrap().width()
                                    != canvas_state.width
                                || canvas_state.preview_layer.as_ref().unwrap().height()
                                    != canvas_state.height
                            {
                                canvas_state.preview_layer =
                                    Some(TiledImage::new(canvas_state.width, canvas_state.height));
                            } else if let Some(ref mut preview) = canvas_state.preview_layer {
                                preview.clear();
                            }
                            canvas_state.preview_blend_mode = BlendMode::Normal;
                        }

                        // Sub-frame events
                        let positions: Vec<(f32, f32)> = if !raw_motion_events.is_empty() {
                            raw_motion_events.to_vec()
                        } else {
                            vec![cf]
                        };

                        // EMA smoothing
                        let smoothed_positions: Vec<(f32, f32)> = {
                            let mut result = Vec::with_capacity(positions.len());
                            for &pos in &positions {
                                let raw = Pos2::new(pos.0, pos.1);
                                let smoothed = if let Some(prev) = self.tool_state.smooth_pos {
                                    let dx = raw.x - prev.x;
                                    let dy = raw.y - prev.y;
                                    let dist = (dx * dx + dy * dy).sqrt();
                                    let alpha = if dist < 1.5 {
                                        1.0
                                    } else {
                                        (0.55 + 1.8 / (dist + 1.8)).min(1.0)
                                    };
                                    Pos2::new(prev.x + alpha * dx, prev.y + alpha * dy)
                                } else {
                                    raw
                                };
                                self.tool_state.smooth_pos = Some(smoothed);
                                result.push((smoothed.x, smoothed.y));
                            }
                            result
                        };

                        let mut frame_dirty_rect = Rect::NOTHING;

                        for &pos in &smoothed_positions {
                            self.content_aware_state
                                .stroke_points
                                .push(Pos2::new(pos.0, pos.1));

                            // Mark brush footprint in hole_mask for async modes
                            if let Some(ref mut mask) = self.content_aware_state.hole_mask {
                                let r = (self.properties.size / 2.0).max(1.0);
                                let ir = r as u32 + 1;
                                let x0 = (pos.0 - r).max(0.0) as u32;
                                let x1 =
                                    ((pos.0 + r) as u32).min(canvas_state.width.saturating_sub(1));
                                let y0 = (pos.1 - r).max(0.0) as u32;
                                let y1 =
                                    ((pos.1 + r) as u32).min(canvas_state.height.saturating_sub(1));
                                let _ = ir;
                                for py in y0..=y1 {
                                    for px in x0..=x1 {
                                        let ddx = px as f32 - pos.0;
                                        let ddy = py as f32 - pos.1;
                                        if ddx * ddx + ddy * ddy <= r * r {
                                            mask.put_pixel(px, py, image::Luma([255u8]));
                                        }
                                    }
                                }
                            }

                            let start_precise = self.tool_state.last_precise_pos;
                            let modified_rect = if let Some(start_p) = start_precise {
                                self.heal_line(canvas_state, (start_p.x, start_p.y), (pos.0, pos.1))
                            } else {
                                self.heal_circle(canvas_state, (pos.0, pos.1))
                            };
                            frame_dirty_rect = frame_dirty_rect.union(modified_rect);
                            self.tool_state.last_precise_pos = Some(Pos2::new(pos.0, pos.1));
                        }

                        self.stroke_tracker.expand_bounds(frame_dirty_rect);
                        if frame_dirty_rect.is_positive() {
                            canvas_state.mark_preview_changed_rect(frame_dirty_rect);
                        }

                        self.tool_state.last_pos = Some(canvas_pos.unwrap());
                        self.tool_state.last_brush_pos = Some(canvas_pos.unwrap());
                        ui.ctx().request_repaint();
                    }
                } else {
                    // Mouse released — commit
                    if self.tool_state.last_pos.is_some() {
                        // Schedule async PatchMatch job if quality requires it
                        if self.content_aware_state.quality.is_async()
                            && let (Some(orig), Some(hmask)) = (
                                self.content_aware_state.stroke_original.take(),
                                self.content_aware_state.hole_mask.take(),
                            )
                        {
                            self.content_aware_state.pending_inpaint =
                                Some(crate::ops::inpaint::InpaintRequest {
                                    original_flat: orig,
                                    hole_mask: hmask,
                                    patch_size: self.content_aware_state.patch_size,
                                    iterations: self.content_aware_state.quality.patchmatch_iters(),
                                    layer_idx: canvas_state.active_layer_index,
                                });
                        }

                        stroke_event = self.stroke_tracker.finish(canvas_state);
                        self.commit_bezier_to_layer(canvas_state, primary_color_f32);
                        canvas_state.clear_preview_state();
                        if let Some(ref ev) = stroke_event {
                            canvas_state.mark_dirty(Some(ev.bounds.expand(12.0)));
                        } else {
                            self.mark_full_dirty(canvas_state);
                        }
                    }
                    self.tool_state.last_pos = None;
                    self.tool_state.last_precise_pos = None;
                    self.tool_state.distance_remainder = 0.0;
                    self.tool_state.smooth_pos = None;
                    self.content_aware_state.stroke_points.clear();
                }
            }

            // ================================================================
            // LASSO SELECT — freeform polygon selection
            // ================================================================
            Tool::Lasso => {
                let esc_pressed = ui.input(|i| i.key_pressed(egui::Key::Escape));
                let alt_held_l = ui.input(|i| i.modifiers.alt);
                let is_secondary_pressed =
                    ui.input(|i| i.pointer.button_pressed(egui::PointerButton::Secondary));

                // Esc / Enter: deselect / cancel
                if esc_pressed || enter_pressed {
                    if self.lasso_state.dragging {
                        self.lasso_state.dragging = false;
                        self.lasso_state.points.clear();
                    } else {
                        canvas_state.clear_selection();
                    }
                    canvas_state.mark_dirty(None);
                    ui.ctx().request_repaint();
                }

                // Start lasso drag — lock effective mode from modifier keys at drag start
                if (is_primary_pressed || is_secondary_pressed)
                    && !self.lasso_state.dragging
                    && let Some(pos) = canvas_pos
                {
                    self.lasso_state.dragging = true;
                    self.lasso_state.right_click_drag = is_secondary_pressed;
                    self.lasso_state.drag_effective_mode = if is_secondary_pressed {
                        SelectionMode::Subtract
                    } else if shift_held && alt_held_l {
                        SelectionMode::Intersect
                    } else if shift_held {
                        SelectionMode::Add
                    } else if alt_held_l {
                        SelectionMode::Subtract
                    } else {
                        self.lasso_state.mode
                    };
                    self.lasso_state.points.clear();
                    self.lasso_state
                        .points
                        .push(Pos2::new(pos.0 as f32, pos.1 as f32));
                }

                // Accumulate points while dragging
                let any_button_down = is_primary_down || is_secondary_down;
                if any_button_down && self.lasso_state.dragging {
                    if let Some(pos) = canvas_pos {
                        let p = Pos2::new(pos.0 as f32, pos.1 as f32);
                        // Only add if moved at least 1px from last point
                        if let Some(last) = self.lasso_state.points.last() {
                            let d = (*last - p).length();
                            if d >= 1.0 {
                                self.lasso_state.points.push(p);
                            }
                        }
                    }
                    ui.ctx().request_repaint();
                }

                // Draw lasso preview path
                if self.lasso_state.dragging && self.lasso_state.points.len() >= 2 {
                    let screen_pts: Vec<Pos2> = self
                        .lasso_state
                        .points
                        .iter()
                        .map(|cp| {
                            Pos2::new(
                                canvas_rect.min.x + cp.x * zoom,
                                canvas_rect.min.y + cp.y * zoom,
                            )
                        })
                        .collect();
                    // Draw the path
                    painter.add(egui::Shape::line(
                        screen_pts.clone(),
                        egui::Stroke::new(1.5, Color32::WHITE),
                    ));
                    painter.add(egui::Shape::line(
                        screen_pts,
                        egui::Stroke::new(0.8, Color32::from_black_alpha(150)),
                    ));
                }

                // Finish on release — rasterize polygon into selection mask
                let any_button_released = is_primary_released || is_secondary_released;
                if any_button_released && self.lasso_state.dragging {
                    let effective_mode = self.lasso_state.drag_effective_mode;
                    self.lasso_state.dragging = false;
                    let pts = std::mem::take(&mut self.lasso_state.points);
                    if pts.len() >= 3 {
                        // Scanline-fill the polygon into the selection mask
                        Self::apply_lasso_selection(canvas_state, &pts, effective_mode);
                        canvas_state.mark_dirty(None);
                    } else {
                        // Tiny lasso → deselect
                        canvas_state.clear_selection();
                        canvas_state.mark_dirty(None);
                    }
                    ui.ctx().request_repaint();
                }
            }

            // ================================================================
            // ZOOM TOOL — click/drag to zoom
            // ================================================================
            Tool::Zoom => {
                self.zoom_pan_action = ZoomPanAction::None;
                let min_drag_screen_px = 30.0; // minimum screen-pixel drag before zoom-to-rect

                // Track potential drag start (but don't commit to dragging yet)
                if is_primary_pressed
                    && !self.zoom_tool_state.dragging
                    && let Some(pos_f) = canvas_pos_f32
                {
                    let p = Pos2::new(pos_f.0, pos_f.1);
                    self.zoom_tool_state.drag_start = Some(p);
                    self.zoom_tool_state.drag_end = Some(p);
                }

                // While held, update end point; only enter drag mode once threshold exceeded
                if is_primary_down
                    && self.zoom_tool_state.drag_start.is_some()
                    && let Some(pos_f) = canvas_pos_f32
                {
                    let end = Pos2::new(pos_f.0, pos_f.1);
                    self.zoom_tool_state.drag_end = Some(end);

                    if !self.zoom_tool_state.dragging
                        && let Some(s) = self.zoom_tool_state.drag_start
                    {
                        let dx = (end.x - s.x) * zoom;
                        let dy = (end.y - s.y) * zoom;
                        if dx.abs().max(dy.abs()) >= min_drag_screen_px {
                            self.zoom_tool_state.dragging = true;
                        }
                    }
                    ui.ctx().request_repaint();
                }

                // Draw zoom rect preview only after threshold exceeded
                if self.zoom_tool_state.dragging
                    && let (Some(s), Some(e)) = (
                        self.zoom_tool_state.drag_start,
                        self.zoom_tool_state.drag_end,
                    )
                {
                    let accent = ui.visuals().selection.bg_fill;
                    let min = Pos2::new(s.x.min(e.x), s.y.min(e.y));
                    let max = Pos2::new(s.x.max(e.x), s.y.max(e.y));
                    let screen_min = Pos2::new(
                        canvas_rect.min.x + min.x * zoom,
                        canvas_rect.min.y + min.y * zoom,
                    );
                    let screen_max = Pos2::new(
                        canvas_rect.min.x + max.x * zoom,
                        canvas_rect.min.y + max.y * zoom,
                    );
                    let r = Rect::from_min_max(screen_min, screen_max);
                    let accent_faint =
                        Color32::from_rgba_unmultiplied(accent.r(), accent.g(), accent.b(), 30);
                    painter.rect_filled(r, 0.0, accent_faint);
                    painter.rect_stroke(r, 0.0, egui::Stroke::new(1.0, accent));
                }

                // On release: zoom-to-rect if dragging, otherwise simple click zoom
                if is_primary_released {
                    if self.zoom_tool_state.dragging {
                        self.zoom_tool_state.dragging = false;
                        if let (Some(s), Some(e)) = (
                            self.zoom_tool_state.drag_start,
                            self.zoom_tool_state.drag_end,
                        ) {
                            self.zoom_pan_action = ZoomPanAction::ZoomToRect {
                                min_x: s.x.min(e.x),
                                min_y: s.y.min(e.y),
                                max_x: s.x.max(e.x),
                                max_y: s.y.max(e.y),
                            };
                        }
                    } else {
                        // Simple click — zoom direction based on toggle
                        if let Some(pos_f) = canvas_pos_f32 {
                            if self.zoom_tool_state.zoom_out_mode {
                                self.zoom_pan_action = ZoomPanAction::ZoomOut {
                                    canvas_x: pos_f.0,
                                    canvas_y: pos_f.1,
                                };
                            } else {
                                self.zoom_pan_action = ZoomPanAction::ZoomIn {
                                    canvas_x: pos_f.0,
                                    canvas_y: pos_f.1,
                                };
                            }
                        }
                    }
                    self.zoom_tool_state.drag_start = None;
                    self.zoom_tool_state.drag_end = None;
                }

                // Right-click: always does the opposite of the toggle
                if is_secondary_clicked && let Some(pos_f) = canvas_pos_f32 {
                    if self.zoom_tool_state.zoom_out_mode {
                        self.zoom_pan_action = ZoomPanAction::ZoomIn {
                            canvas_x: pos_f.0,
                            canvas_y: pos_f.1,
                        };
                    } else {
                        self.zoom_pan_action = ZoomPanAction::ZoomOut {
                            canvas_x: pos_f.0,
                            canvas_y: pos_f.1,
                        };
                    }
                }
            }

            // ================================================================
            // PAN TOOL — click+drag to pan viewport
            // ================================================================
            Tool::Pan => {
                self.zoom_pan_action = ZoomPanAction::None;
                // Drag delta in screen coordinates → pass directly to Canvas pan_offset
                if is_primary_down {
                    let drag_delta = ui.input(|i| i.pointer.delta());
                    if drag_delta.length_sq() > 0.0 {
                        self.zoom_pan_action = ZoomPanAction::Pan {
                            dx: drag_delta.x,
                            dy: drag_delta.y,
                        };
                    }
                    ui.ctx().request_repaint();
                }
            }

            // ================================================================
            // PERSPECTIVE CROP — 4-corner quad, drag handles
            // ================================================================
            Tool::PerspectiveCrop => {
                let esc_pressed = ui.input(|i| i.key_pressed(egui::Key::Escape));

                if esc_pressed {
                    self.perspective_crop_state.active = false;
                    self.perspective_crop_state.dragging_corner = None;
                    ui.ctx().request_repaint();
                }

                // Auto-init: place quad immediately when tool is selected
                if self.perspective_crop_state.needs_auto_init {
                    self.perspective_crop_state.needs_auto_init = false;
                    let w = canvas_state.width as f32;
                    let h = canvas_state.height as f32;
                    let m = 0.1; // 10% inset
                    self.perspective_crop_state.corners = [
                        Pos2::new(w * m, h * m),                 // top-left
                        Pos2::new(w * (1.0 - m), h * m),         // top-right
                        Pos2::new(w * (1.0 - m), h * (1.0 - m)), // bottom-right
                        Pos2::new(w * m, h * (1.0 - m)),         // bottom-left
                    ];
                    self.perspective_crop_state.active = true;
                    ui.ctx().request_repaint();
                }

                if self.perspective_crop_state.active {
                    // Enter: apply crop
                    if enter_pressed {
                        if let Some(cmd) = Self::apply_perspective_crop(
                            canvas_state,
                            &self.perspective_crop_state.corners,
                        ) {
                            self.pending_history_commands.push(cmd);
                        }
                        self.perspective_crop_state.active = false;
                        self.perspective_crop_state.dragging_corner = None;
                        canvas_state.mark_dirty(None);
                        ui.ctx().request_repaint();
                    }

                    let handle_r = 6.0 / zoom; // handles stay the same screen size

                    // Hit-test corners
                    if is_primary_pressed && let Some(pos_f) = canvas_pos_f32 {
                        let mp = Pos2::new(pos_f.0, pos_f.1);
                        for (i, &corner) in self.perspective_crop_state.corners.iter().enumerate() {
                            if (corner - mp).length() < handle_r + 4.0 / zoom {
                                self.perspective_crop_state.dragging_corner = Some(i);
                                break;
                            }
                        }
                    }

                    // Drag selected corner
                    if is_primary_down
                        && let Some(idx) = self.perspective_crop_state.dragging_corner
                        && let Some(pos_f) = canvas_pos_f32
                    {
                        let clamped = Pos2::new(
                            pos_f.0.clamp(0.0, canvas_state.width as f32 - 1.0),
                            pos_f.1.clamp(0.0, canvas_state.height as f32 - 1.0),
                        );
                        self.perspective_crop_state.corners[idx] = clamped;
                        ui.ctx().request_repaint();
                    }

                    if is_primary_released {
                        self.perspective_crop_state.dragging_corner = None;
                    }

                    // Draw the quad outline and handles
                    let corners = &self.perspective_crop_state.corners;
                    let screen: Vec<Pos2> = corners
                        .iter()
                        .map(|c| {
                            Pos2::new(
                                canvas_rect.min.x + c.x * zoom,
                                canvas_rect.min.y + c.y * zoom,
                            )
                        })
                        .collect();

                    // Dim area outside quad
                    painter.rect_filled(canvas_rect, 0.0, Color32::from_black_alpha(80));

                    // Draw filled quad (clear the dimming inside)
                    let mut quad_mesh = egui::Mesh::default();
                    for &sp in &screen {
                        quad_mesh.colored_vertex(sp, Color32::TRANSPARENT);
                    }
                    quad_mesh.add_triangle(0, 1, 2);
                    quad_mesh.add_triangle(0, 2, 3);
                    // We can't clear the dimming easily, so draw quad outline only

                    // Quad outline
                    let accent = ui.visuals().selection.bg_fill;
                    for i in 0..4 {
                        let a = screen[i];
                        let b = screen[(i + 1) % 4];
                        painter.line_segment([a, b], egui::Stroke::new(2.0, Color32::WHITE));
                        painter.line_segment([a, b], egui::Stroke::new(1.0, accent));
                    }

                    // Corner handles
                    let screen_handle_r = 5.0;
                    for (i, &sp) in screen.iter().enumerate() {
                        let is_active = self.perspective_crop_state.dragging_corner == Some(i);
                        let fill = if is_active { accent } else { Color32::WHITE };
                        painter.circle_filled(sp, screen_handle_r, fill);
                        painter.circle_stroke(
                            sp,
                            screen_handle_r,
                            egui::Stroke::new(1.5, Color32::from_black_alpha(120)),
                        );
                    }
                }
            }
        }

        // Store stroke event for app.rs to pick up
        if stroke_event.is_some() {
            self.pending_stroke_event = stroke_event;
        }
    }

    /// Draw interactive handles for Bézier curve editing
    fn draw_bezier_handles(&self, painter: &egui::Painter, canvas_rect: Rect, zoom: f32) {
        let points = self.line_state.line_tool.control_points;

        // Draw control lines (dashed)
        let p0_screen = self.canvas_pos2_to_screen(points[0], canvas_rect, zoom);
        let p1_screen = self.canvas_pos2_to_screen(points[1], canvas_rect, zoom);
        let p2_screen = self.canvas_pos2_to_screen(points[2], canvas_rect, zoom);
        let p3_screen = self.canvas_pos2_to_screen(points[3], canvas_rect, zoom);

        // Draw dashed control lines with transparency
        painter.line_segment(
            [p0_screen, p1_screen],
            (1.0, Color32::from_rgba_unmultiplied(150, 150, 150, 180)),
        );
        painter.line_segment(
            [p2_screen, p3_screen],
            (1.0, Color32::from_rgba_unmultiplied(150, 150, 150, 180)),
        );

        // Draw handles with high transparency (30% opaque) to see through them
        let handle_colors = [
            Color32::from_rgba_unmultiplied(0, 200, 0, 77), // Start - green, 30% opaque
            Color32::from_rgba_unmultiplied(100, 150, 255, 77), // Control1 - light blue, 30% opaque
            Color32::from_rgba_unmultiplied(100, 150, 255, 77), // Control2 - light blue, 30% opaque
            Color32::from_rgba_unmultiplied(200, 0, 0, 77), // End - red, 30% opaque
        ];

        for (i, &point) in points.iter().enumerate() {
            let screen_pos = self.canvas_pos2_to_screen(point, canvas_rect, zoom);
            painter.circle_filled(screen_pos, 6.0, handle_colors[i]);
            painter.circle_stroke(
                screen_pos,
                6.0,
                (2.0, Color32::from_rgba_unmultiplied(255, 255, 255, 100)),
            );
        }

        // ── Pan handle — drawn as a diamond with a 4-arrow cross, offset from P0 ──
        // Positioned 22px up-left of P0 in screen space. Distinct from the regular
        // endpoint handles: different shape, colour and stronger outline so it clearly
        // communicates "move the whole line".
        let pan_offset = egui::Vec2::new(-22.0, -22.0);
        let ph = p0_screen + pan_offset;

        // Dashed connector line from P0 to pan handle so it's visually anchored
        painter.line_segment(
            [p0_screen, ph],
            (1.0, Color32::from_rgba_unmultiplied(200, 200, 200, 100)),
        );

        // Choose colour: brighter when hovering or dragging
        let is_active = self.line_state.line_tool.pan_handle_dragging
            || self.line_state.line_tool.pan_handle_hovering;
        let fill_alpha: u8 = if is_active { 210 } else { 130 };
        let ring_alpha: u8 = if is_active { 255 } else { 160 };
        let fill = Color32::from_rgba_unmultiplied(255, 200, 50, fill_alpha); // amber
        let ring = Color32::from_rgba_unmultiplied(255, 255, 255, ring_alpha);
        let outline = Color32::from_rgba_unmultiplied(80, 60, 0, fill_alpha); // dark amber border

        // Diamond shape (rotate square 45°)
        let r = if is_active { 8.0f32 } else { 7.0f32 };
        let diamond = vec![
            ph + egui::Vec2::new(0.0, -r),
            ph + egui::Vec2::new(r, 0.0),
            ph + egui::Vec2::new(0.0, r),
            ph + egui::Vec2::new(-r, 0.0),
        ];
        painter.add(egui::Shape::convex_polygon(
            diamond.clone(),
            fill,
            egui::Stroke::new(1.5, outline),
        ));
        // White ring for contrast
        painter.add(egui::Shape::closed_line(
            diamond,
            egui::Stroke::new(0.8, ring),
        ));

        // 4-arrow cross drawn as four small filled triangles pointing outward
        let arm = r * 0.55;
        let tip = r * 0.90;
        let w = r * 0.28;
        let arrow_color = Color32::from_rgba_unmultiplied(60, 40, 0, fill_alpha);
        for (ax, ay) in [(0.0f32, -1.0f32), (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)] {
            // perpendicular
            let (px_dir, py_dir) = (-ay, ax);
            let tri = vec![
                ph + egui::Vec2::new(ax * tip, ay * tip),
                ph + egui::Vec2::new(ax * arm + px_dir * w, ay * arm + py_dir * w),
                ph + egui::Vec2::new(ax * arm - px_dir * w, ay * arm - py_dir * w),
            ];
            painter.add(egui::Shape::convex_polygon(
                tri,
                arrow_color,
                egui::Stroke::NONE,
            ));
        }
    }

    /// Commit the Bézier curve from preview layer to the actual active layer.
    /// Only iterates populated preview chunks (~5-20 chunks) instead of all W×H pixels.
    fn commit_bezier_to_layer(&self, canvas_state: &mut CanvasState, _color_f32: [f32; 4]) {
        let width = canvas_state.width;
        let height = canvas_state.height;
        let blend_mode = self.properties.blending_mode;

        // Extract only populated chunk data from preview (clone to release borrow)
        let preview_chunks: Vec<(u32, u32, image::RgbaImage)> = match &canvas_state.preview_layer {
            Some(preview) => preview
                .chunk_keys()
                .filter_map(|(cx, cy)| preview.get_chunk(cx, cy).map(|c| (cx, cy, c.clone())))
                .collect(),
            None => return,
        };

        // Extract selection mask pointer before mutable borrow of active layer
        let mask_ptr = canvas_state
            .selection_mask
            .as_ref()
            .map(|m| m as *const GrayImage);

        if let Some(active_layer) = canvas_state.get_active_layer_mut() {
            let mask_ref = mask_ptr.map(|p| unsafe { &*p });
            for (cx, cy, preview_chunk) in &preview_chunks {
                let base_x = cx * CHUNK_SIZE;
                let base_y = cy * CHUNK_SIZE;
                let cw = CHUNK_SIZE.min(width.saturating_sub(base_x));
                let ch = CHUNK_SIZE.min(height.saturating_sub(base_y));

                for ly in 0..ch {
                    for lx in 0..cw {
                        // Skip pixels outside the selection mask
                        if let Some(mask) = mask_ref {
                            let gx = base_x + lx;
                            let gy = base_y + ly;
                            if gx < mask.width() && gy < mask.height() {
                                if mask.get_pixel(gx, gy).0[0] == 0 {
                                    continue;
                                }
                            } else {
                                continue;
                            }
                        }

                        let preview_pixel = *preview_chunk.get_pixel(lx, ly);
                        if preview_pixel[3] > 0 {
                            let layer_pixel =
                                active_layer.pixels.get_pixel_mut(base_x + lx, base_y + ly);
                            *layer_pixel = CanvasState::blend_pixel_static(
                                *layer_pixel,
                                preview_pixel,
                                blend_mode,
                                1.0,
                            );
                        }
                    }
                }
            }
        }
    }

    /// Commit the eraser mask from preview layer to the actual active layer.
    /// Applies the mask's alpha channel as erase strength (reducing the layer's alpha).
    /// Only iterates populated preview chunks for efficiency.
    fn commit_eraser_to_layer(&self, canvas_state: &mut CanvasState) {
        // Eraser already draws at mirrored positions per-frame, so no
        // mirror_preview_layer() needed here (would double-mirror).

        let width = canvas_state.width;
        let height = canvas_state.height;

        // Extract only populated chunk data from preview (clone to release borrow)
        let preview_chunks: Vec<(u32, u32, image::RgbaImage)> = match &canvas_state.preview_layer {
            Some(preview) => preview
                .chunk_keys()
                .filter_map(|(cx, cy)| preview.get_chunk(cx, cy).map(|c| (cx, cy, c.clone())))
                .collect(),
            None => return,
        };

        // Extract selection mask pointer before mutable borrow of active layer
        let mask_ptr = canvas_state
            .selection_mask
            .as_ref()
            .map(|m| m as *const GrayImage);

        if let Some(active_layer) = canvas_state.get_active_layer_mut() {
            let mask_ref = mask_ptr.map(|p| unsafe { &*p });
            for (cx, cy, preview_chunk) in &preview_chunks {
                let base_x = cx * CHUNK_SIZE;
                let base_y = cy * CHUNK_SIZE;
                let cw = CHUNK_SIZE.min(width.saturating_sub(base_x));
                let ch = CHUNK_SIZE.min(height.saturating_sub(base_y));

                for ly in 0..ch {
                    for lx in 0..cw {
                        // Skip pixels outside the selection mask
                        if let Some(mask) = mask_ref {
                            let gx = base_x + lx;
                            let gy = base_y + ly;
                            if gx < mask.width() && gy < mask.height() {
                                if mask.get_pixel(gx, gy).0[0] == 0 {
                                    continue;
                                }
                            } else {
                                continue;
                            }
                        }

                        let mask_pixel = *preview_chunk.get_pixel(lx, ly);
                        if mask_pixel[3] > 0 {
                            let layer_pixel =
                                active_layer.pixels.get_pixel_mut(base_x + lx, base_y + ly);
                            // Reduce the layer pixel's alpha by the mask strength
                            let mask_strength = mask_pixel[3] as f32 / 255.0;
                            let current_a = layer_pixel[3] as f32 / 255.0;
                            let new_a = (current_a * (1.0 - mask_strength)).max(0.0);
                            layer_pixel[3] = (new_a * 255.0) as u8;
                        }
                    }
                }
            }
        }
    }

    // ========================================================================
    // SPEED-ADAPTIVE EMA SMOOTHING
    // ========================================================================
    // The brush position is smoothed using an exponential moving average (EMA)
    // applied inline in the painting loop above.  This replaces the previous
    // Catmull-Rom spline approach which was ineffective because 1000 Hz
    // sub-frame events produce segments too short (~2 px) for meaningful
    // spline curvature.  The EMA naturally accumulates across consecutive
    // direction changes, rounding off corners regardless of input density.

    /// B6: Rebuild brush alpha LUT when brush properties change.
    /// The LUT maps squared-distance ratio (0..255 → 0.0..1.0 of `dist_sq/radius_sq`)
    /// to alpha (0..255).  Eliminates per-pixel `sqrt` + `smoothstep`.
    fn rebuild_brush_lut(&mut self) {
        let params = (
            self.properties.size,
            self.properties.hardness,
            self.properties.anti_aliased,
        );
        if params == self.lut_params {
            return;
        }
        self.lut_params = params;

        let radius = self.properties.size / 2.0;
        if radius < 0.001 {
            self.brush_alpha_lut = [0u8; 256];
            return;
        }

        for i in 0..256 {
            let t_sq = i as f32 / 255.0; // squared distance ratio
            let dist = t_sq.sqrt() * radius; // linear distance
            let alpha = self.compute_brush_alpha(dist, radius);
            self.brush_alpha_lut[i] = (alpha * 255.0).round().min(255.0) as u8;
        }
    }

    /// Compute brush alpha with optional smoothstep interpolation
    /// When anti_aliased is false, returns hard binary 0.0 or 1.0 based on radius only
    fn compute_brush_alpha(&self, dist: f32, radius: f32) -> f32 {
        // Non-AA mode: hard binary edge at radius
        if !self.properties.anti_aliased {
            return if dist <= radius { 1.0 } else { 0.0 };
        }

        // Anti-aliased mode with hardness-based soft edge:
        // 1. Remap hardness range for much softer low-hardness brushes.
        //    UI 0% → 0.02 internal (extremely soft, airbrush-like)
        //    UI 100% → 1.0 internal (hard edge)
        let remapped_hardness = 0.02 + (self.properties.hardness * 0.98);
        let safe_hardness = remapped_hardness.clamp(0.0, 0.99); // Clamp to prevent div by zero

        // For small brushes (radius < 3), force extra AA by extending beyond nominal radius
        let (effective_radius, fade_width) = if radius < 3.0 {
            // For tiny brushes, ensure at least 1.5px of AA range
            let aa_extend = 1.5;
            let extended_radius = radius + aa_extend;
            let fade = aa_extend + (radius * (1.0 - safe_hardness));
            (extended_radius, fade)
        } else {
            // Normal brushes: fade within the brush radius
            let fade = (radius * (1.0 - safe_hardness)).max(1.0);
            (radius, fade)
        };

        let solid_radius = effective_radius - fade_width;

        if dist <= solid_radius {
            return 1.0;
        } else if dist >= effective_radius {
            return 0.0;
        }

        // 2. Normalize distance within the fade region (0.0 to 1.0)
        let t = (dist - solid_radius) / fade_width;

        // 3. Apply Smoothstep: t * t * (3 - 2t)
        // Invert t first because we want 1.0 at the inner edge and 0.0 at outer
        let x = 1.0 - t.clamp(0.0, 1.0);
        x * x * (3.0 - 2.0 * x)
    }

    /// Compute alpha specifically for line tool with forced soft edges
    fn compute_line_alpha(&self, dist: f32, radius: f32, forced_hardness: f32) -> f32 {
        // Always use anti-aliasing for lines
        let safe_hardness = forced_hardness.clamp(0.0, 0.99);

        // For small brushes (radius < 3), force extra AA by extending beyond nominal radius
        let (effective_radius, fade_width) = if radius < 3.0 {
            // For tiny brushes, ensure at least 1.5px of AA range
            let aa_extend = 1.5;
            let extended_radius = radius + aa_extend;
            let fade = aa_extend + (radius * (1.0 - safe_hardness));
            (extended_radius, fade)
        } else {
            // Normal brushes: fade within the brush radius, but with larger fade for softness
            let fade = (radius * (1.0 - safe_hardness)).max(2.0); // Min 2px fade for softness
            (radius, fade)
        };

        let solid_radius = effective_radius - fade_width;

        if dist <= solid_radius {
            return 1.0;
        } else if dist >= effective_radius {
            return 0.0;
        }

        // Normalize distance within the fade region (0.0 to 1.0)
        let t = (dist - solid_radius) / fade_width;

        // Apply Smoothstep: t * t * (3 - 2t)
        // Invert t first because we want 1.0 at the inner edge and 0.0 at outer
        let x = 1.0 - t.clamp(0.0, 1.0);
        x * x * (3.0 - 2.0 * x)
    }

    fn draw_circle_no_dirty(
        &self,
        target_image: &mut TiledImage,
        width: u32,
        height: u32,
        pos: (f32, f32),
        is_eraser: bool,
        use_secondary: bool,
        primary_color_f32: [f32; 4],
        secondary_color_f32: [f32; 4],
        selection_mask: Option<&GrayImage>,
    ) {
        // Dispatch to image tip path if active
        if !self.properties.brush_tip.is_circle() {
            // Compute rotation angle for this stamp
            let rotation_deg = if self.properties.tip_random_rotation {
                // Hash position to get a deterministic-but-random angle
                let (lo, hi) = self.properties.tip_rotation_range;
                let range = hi - lo;
                if range.abs() < 0.01 {
                    lo
                } else {
                    // Simple hash of position for pseudorandom per-stamp rotation
                    let hash = Self::stamp_hash(pos.0, pos.1, self.stamp_counter);
                    lo + (hash % 10000) as f32 / 10000.0 * range
                }
            } else {
                self.properties.tip_rotation
            };
            self.draw_image_tip_no_dirty(
                target_image,
                width,
                height,
                pos,
                is_eraser,
                use_secondary,
                primary_color_f32,
                secondary_color_f32,
                selection_mask,
                rotation_deg,
            );
            return;
        }

        // Scatter: randomize stamp position by up to scatter*diameter
        let (cx, cy) = {
            let (px, py) = pos;
            if self.properties.scatter > 0.01 {
                let diam = self.properties.size;
                let h1 = Self::stamp_hash(px, py, self.stamp_counter) as f32 / u32::MAX as f32;
                let h2 = Self::stamp_hash(py, px, self.stamp_counter.wrapping_add(99991)) as f32
                    / u32::MAX as f32;
                let ox = (h1 * 2.0 - 1.0) * self.properties.scatter * diam;
                let oy = (h2 * 2.0 - 1.0) * self.properties.scatter * diam;
                (px + ox, py + oy)
            } else {
                (px, py)
            }
        };
        let radius = self.properties.size / 2.0;
        let radius_sq = radius * radius;
        if radius_sq < 0.001 {
            return;
        }
        let inv_radius_sq = 1.0 / radius_sq;

        let _r_ceil = radius.ceil() as i32;
        let min_x = ((cx - radius).max(0.0)) as u32;
        let max_x = ((cx + radius) as u32).min(width.saturating_sub(1));
        let min_y = ((cy - radius).max(0.0)) as u32;
        let max_y = ((cy + radius) as u32).min(height.saturating_sub(1));
        if min_x > max_x || min_y > max_y {
            return;
        }

        // Determine brush color (high-precision unmultiplied)
        let brush_color_f32 = if use_secondary {
            secondary_color_f32
        } else {
            primary_color_f32
        };
        let [src_r, src_g, src_b, src_a] = brush_color_f32;
        let base_r8 = (src_r * 255.0) as u8;
        let base_g8 = (src_g * 255.0) as u8;
        let base_b8 = (src_b * 255.0) as u8;
        // Color jitter: per-stamp HSL perturbation
        let (src_r8, src_g8, src_b8) =
            if self.properties.hue_jitter > 0.01 || self.properties.brightness_jitter > 0.01 {
                let (mut h, s, mut l) = crate::ops::adjustments::rgb_to_hsl(src_r, src_g, src_b);
                if self.properties.hue_jitter > 0.01 {
                    let hh = Self::stamp_hash(
                        pos.0 + 0.1,
                        pos.1 + 0.2,
                        self.stamp_counter.wrapping_add(777),
                    ) as f32
                        / u32::MAX as f32;
                    h = (h + (hh * 2.0 - 1.0) * self.properties.hue_jitter * 0.5).fract();
                    if h < 0.0 {
                        h += 1.0;
                    }
                }
                if self.properties.brightness_jitter > 0.01 {
                    let bh = Self::stamp_hash(
                        pos.0 + 0.3,
                        pos.1 + 0.4,
                        self.stamp_counter.wrapping_add(555),
                    ) as f32
                        / u32::MAX as f32;
                    l = (l + (bh * 2.0 - 1.0) * self.properties.brightness_jitter * 0.5)
                        .clamp(0.0, 1.0);
                }
                let (nr, ng, nb) = crate::ops::adjustments::hsl_to_rgb(h, s, l);
                ((nr * 255.0) as u8, (ng * 255.0) as u8, (nb * 255.0) as u8)
            } else {
                (base_r8, base_g8, base_b8)
            };

        let lut = &self.brush_alpha_lut;
        let cs = crate::canvas::CHUNK_SIZE;

        // Determine which chunks overlap the brush bounding box
        let chunk_x0 = min_x / cs;
        let chunk_y0 = min_y / cs;
        let chunk_x1 = max_x / cs;
        let chunk_y1 = max_y / cs;

        for chunk_cy in chunk_y0..=chunk_y1 {
            for chunk_cx in chunk_x0..=chunk_x1 {
                let chunk_base_x = chunk_cx * cs;
                let chunk_base_y = chunk_cy * cs;

                // Local pixel range within this chunk (clamped to brush bbox & canvas)
                let lx0 = min_x.saturating_sub(chunk_base_x);
                let ly0 = min_y.saturating_sub(chunk_base_y);
                let lx1 = (max_x + 1 - chunk_base_x).min(cs).min(width - chunk_base_x);
                let ly1 = (max_y + 1 - chunk_base_y)
                    .min(cs)
                    .min(height - chunk_base_y);
                if lx0 >= lx1 || ly0 >= ly1 {
                    continue;
                }

                // Quick check: does ANY pixel in this chunk-local range fall within the circle?
                // Test the closest point of the local rect to the circle center
                let near_x = (cx).clamp(
                    chunk_base_x as f32 + lx0 as f32,
                    chunk_base_x as f32 + lx1 as f32 - 1.0,
                );
                let near_y = (cy).clamp(
                    chunk_base_y as f32 + ly0 as f32,
                    chunk_base_y as f32 + ly1 as f32 - 1.0,
                );
                let nd = (near_x - cx) * (near_x - cx) + (near_y - cy) * (near_y - cy);
                if nd > radius_sq {
                    continue;
                }

                // Get or create chunk (COW-safe via ensure_chunk_mut)
                let chunk = target_image.ensure_chunk_mut(chunk_cx, chunk_cy);
                let chunk_raw = chunk.as_mut();
                let chunk_stride = cs as usize * 4;

                for ly in ly0..ly1 {
                    let global_y = chunk_base_y + ly;
                    let dy = global_y as f32 - cy;
                    let dy_sq = dy * dy;
                    let row_off = ly as usize * chunk_stride;

                    for lx in lx0..lx1 {
                        let global_x = chunk_base_x + lx;

                        // Selection mask check
                        if let Some(mask) = selection_mask {
                            if global_x < mask.width() && global_y < mask.height() {
                                if mask.get_pixel(global_x, global_y).0[0] == 0 {
                                    continue;
                                }
                            } else {
                                continue;
                            }
                        }

                        let dx = global_x as f32 - cx;
                        let dist_sq = dx * dx + dy_sq;
                        if dist_sq > radius_sq {
                            continue;
                        }

                        // B6: LUT lookup — replaces sqrt + smoothstep
                        let lut_idx = (dist_sq * inv_radius_sq * 255.0).min(255.0) as usize;
                        let geom_alpha_u8 = lut[lut_idx];
                        if geom_alpha_u8 == 0 {
                            continue;
                        }
                        let geom_alpha = geom_alpha_u8 as f32 / 255.0;

                        let px_off = row_off + lx as usize * 4;

                        if is_eraser {
                            let erase_strength = geom_alpha * src_a * self.properties.flow;
                            if erase_strength < 0.01 {
                                continue;
                            }
                            let old_mask = chunk_raw[px_off + 3] as f32 / 255.0;
                            if erase_strength > old_mask {
                                chunk_raw[px_off] = 0;
                                chunk_raw[px_off + 1] = 0;
                                chunk_raw[px_off + 2] = 0;
                                chunk_raw[px_off + 3] = (erase_strength * 255.0) as u8;
                            }
                        } else {
                            let brush_alpha = geom_alpha * src_a * self.properties.flow;
                            if brush_alpha < 0.01 {
                                continue;
                            }
                            match self.properties.brush_mode {
                                BrushMode::Normal => {
                                    let brush_alpha_u8 = (brush_alpha * 255.0) as u8;
                                    let old_alpha = chunk_raw[px_off + 3];
                                    // Max-alpha stamping: only update if increasing opacity
                                    if brush_alpha_u8 >= old_alpha {
                                        chunk_raw[px_off] = src_r8;
                                        chunk_raw[px_off + 1] = src_g8;
                                        chunk_raw[px_off + 2] = src_b8;
                                        chunk_raw[px_off + 3] = brush_alpha_u8;
                                    }
                                }
                                BrushMode::Dodge | BrushMode::Burn | BrushMode::Sponge => {
                                    // Read existing pixel, modify in HSL space, write back
                                    let old_r = chunk_raw[px_off] as f32 / 255.0;
                                    let old_g = chunk_raw[px_off + 1] as f32 / 255.0;
                                    let old_b = chunk_raw[px_off + 2] as f32 / 255.0;
                                    let (h, mut s, mut l) =
                                        crate::ops::adjustments::rgb_to_hsl(old_r, old_g, old_b);
                                    let strength = brush_alpha * 0.5;
                                    match self.properties.brush_mode {
                                        BrushMode::Dodge => l = (l + strength).clamp(0.0, 1.0),
                                        BrushMode::Burn => l = (l - strength).clamp(0.0, 1.0),
                                        BrushMode::Sponge => s = (s - strength).clamp(0.0, 1.0),
                                        _ => {}
                                    }
                                    let (nr, ng, nb) = crate::ops::adjustments::hsl_to_rgb(h, s, l);
                                    chunk_raw[px_off] = (nr * 255.0) as u8;
                                    chunk_raw[px_off + 1] = (ng * 255.0) as u8;
                                    chunk_raw[px_off + 2] = (nb * 255.0) as u8;
                                    // alpha unchanged
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Rescale the brush tip mask to the current brush size.
    /// Only rebuilds if tip or size changed. Called alongside rebuild_brush_lut.
    fn rebuild_tip_mask(&mut self, assets: &Assets) {
        if self.properties.brush_tip.is_circle() {
            return;
        }
        let tip_name = match &self.properties.brush_tip {
            BrushTip::Image(name) => name.clone(),
            BrushTip::Circle => return,
        };
        let target_size = (self.properties.size.ceil() as u32).max(1);
        let hardness_key = (self.properties.hardness * 100.0).round() as u32;
        let key = (tip_name.clone(), target_size, hardness_key);
        if key == self.brush_tip_cache_key {
            return;
        }
        self.brush_tip_cache_key = key;

        if let Some(data) = assets.get_brush_tip_data(&tip_name) {
            let src = &data.mask;
            let src_size = data.mask_size;
            if src_size == 0 {
                return;
            }

            let dst_size = target_size;
            self.brush_tip_mask
                .resize((dst_size * dst_size) as usize, 0);
            self.brush_tip_mask_size = dst_size;

            // Bilinear interpolation from source mask to target size
            let scale = src_size as f32 / dst_size as f32;
            for dy in 0..dst_size {
                for dx in 0..dst_size {
                    let sx = dx as f32 * scale;
                    let sy = dy as f32 * scale;
                    let sx0 = sx.floor() as u32;
                    let sy0 = sy.floor() as u32;
                    let sx1 = (sx0 + 1).min(src_size - 1);
                    let sy1 = (sy0 + 1).min(src_size - 1);
                    let fx = sx - sx0 as f32;
                    let fy = sy - sy0 as f32;

                    let v00 = src[(sy0 * src_size + sx0) as usize] as f32;
                    let v10 = src[(sy0 * src_size + sx1) as usize] as f32;
                    let v01 = src[(sy1 * src_size + sx0) as usize] as f32;
                    let v11 = src[(sy1 * src_size + sx1) as usize] as f32;

                    let top = v00 * (1.0 - fx) + v10 * fx;
                    let bot = v01 * (1.0 - fx) + v11 * fx;
                    let val = top * (1.0 - fy) + bot * fy;
                    self.brush_tip_mask[(dy * dst_size + dx) as usize] =
                        val.round().min(255.0) as u8;
                }
            }

            // Apply hardness as contrast modifier:
            // hardness 1.0 → use as-is
            // hardness 0.0 → heavily feathered (only brightest survive)
            let h = self.properties.hardness;
            if h < 0.99 {
                let threshold = (1.0 - h) * 0.6; // 0..0.6 range
                let range = 1.0 - threshold;
                for v in self.brush_tip_mask.iter_mut() {
                    let norm = *v as f32 / 255.0;
                    let adj = ((norm - threshold) / range).clamp(0.0, 1.0);
                    *v = (adj * 255.0).round() as u8;
                }
            }

            // Anti-alias pass: when downscaling significantly (src >> dst),
            // bilinear alone can't capture enough of the source detail and edges
            // look blocky. Apply a small box blur whose radius scales with the
            // downscale ratio. This smooths staircased edges while preserving shape.
            if dst_size < src_size && dst_size >= 3 {
                let ratio = src_size as f32 / dst_size as f32;
                // blur radius: 1 pass at 2-4× downscale, 2 at higher ratios
                let passes: usize = if ratio > 4.0 {
                    2
                } else if ratio > 1.5 {
                    1
                } else {
                    0
                };
                for _ in 0..passes {
                    // Horizontal pass
                    let mut tmp = self.brush_tip_mask.clone();
                    for y in 0..dst_size {
                        for x in 0..dst_size {
                            let idx = (y * dst_size + x) as usize;
                            let mut sum = self.brush_tip_mask[idx] as u32;
                            let mut count = 1u32;
                            if x > 0 {
                                sum += self.brush_tip_mask[(y * dst_size + x - 1) as usize] as u32;
                                count += 1;
                            }
                            if x + 1 < dst_size {
                                sum += self.brush_tip_mask[(y * dst_size + x + 1) as usize] as u32;
                                count += 1;
                            }
                            tmp[idx] = (sum / count) as u8;
                        }
                    }
                    // Vertical pass
                    for y in 0..dst_size {
                        for x in 0..dst_size {
                            let idx = (y * dst_size + x) as usize;
                            let mut sum = tmp[idx] as u32;
                            let mut count = 1u32;
                            if y > 0 {
                                sum += tmp[((y - 1) * dst_size + x) as usize] as u32;
                                count += 1;
                            }
                            if y + 1 < dst_size {
                                sum += tmp[((y + 1) * dst_size + x) as usize] as u32;
                                count += 1;
                            }
                            self.brush_tip_mask[idx] = (sum / count) as u8;
                        }
                    }
                }
            }
        } else {
            self.brush_tip_mask.clear();
            self.brush_tip_mask_size = 0;
        }
    }

    /// Stamp an image-based brush tip at the given position.
    /// Uses the pre-scaled tip mask from `brush_tip_mask`.
    /// `rotation_deg` applies rotation (degrees) to the mask sampling.
    fn draw_image_tip_no_dirty(
        &self,
        target_image: &mut TiledImage,
        width: u32,
        height: u32,
        pos: (f32, f32),
        is_eraser: bool,
        use_secondary: bool,
        primary_color_f32: [f32; 4],
        secondary_color_f32: [f32; 4],
        selection_mask: Option<&GrayImage>,
        rotation_deg: f32,
    ) {
        let mask_size = self.brush_tip_mask_size;
        if mask_size == 0 || self.brush_tip_mask.is_empty() {
            return;
        }

        // Scatter: randomize stamp position
        let (cx, cy) = {
            let (px, py) = pos;
            if self.properties.scatter > 0.01 {
                let diam = self.properties.size;
                let h1 = Self::stamp_hash(px, py, self.stamp_counter) as f32 / u32::MAX as f32;
                let h2 = Self::stamp_hash(py, px, self.stamp_counter.wrapping_add(99991)) as f32
                    / u32::MAX as f32;
                let ox = (h1 * 2.0 - 1.0) * self.properties.scatter * diam;
                let oy = (h2 * 2.0 - 1.0) * self.properties.scatter * diam;
                (px + ox, py + oy)
            } else {
                (px, py)
            }
        };
        let half = mask_size as f32 / 2.0;

        // When rotated, the bounding box of the stamp expands.
        // The diagonal of the original square mask is half*sqrt(2) from center.
        let rotated = rotation_deg.abs() > 0.01;
        let (cos_a, sin_a) = if rotated {
            let rad = -rotation_deg.to_radians(); // negative = inverse rotation for sampling
            (rad.cos(), rad.sin())
        } else {
            (1.0, 0.0)
        };
        let effective_half = if rotated {
            half * std::f32::consts::SQRT_2
        } else {
            half
        };

        // Bounding box of the stamp in canvas coordinates
        let stamp_min_x = (cx - effective_half).max(0.0) as u32;
        let stamp_min_y = (cy - effective_half).max(0.0) as u32;
        let stamp_max_x = ((cx + effective_half) as u32).min(width.saturating_sub(1));
        let stamp_max_y = ((cy + effective_half) as u32).min(height.saturating_sub(1));
        if stamp_min_x > stamp_max_x || stamp_min_y > stamp_max_y {
            return;
        }

        // Brush color
        let brush_color_f32 = if use_secondary {
            secondary_color_f32
        } else {
            primary_color_f32
        };
        let [src_r, src_g, src_b, src_a] = brush_color_f32;
        let base_r8 = (src_r * 255.0) as u8;
        let base_g8 = (src_g * 255.0) as u8;
        let base_b8 = (src_b * 255.0) as u8;
        // Color jitter
        let (src_r8, src_g8, src_b8) =
            if self.properties.hue_jitter > 0.01 || self.properties.brightness_jitter > 0.01 {
                let (mut h, s, mut l) = crate::ops::adjustments::rgb_to_hsl(src_r, src_g, src_b);
                if self.properties.hue_jitter > 0.01 {
                    let hh = Self::stamp_hash(
                        pos.0 + 0.1,
                        pos.1 + 0.2,
                        self.stamp_counter.wrapping_add(777),
                    ) as f32
                        / u32::MAX as f32;
                    h = (h + (hh * 2.0 - 1.0) * self.properties.hue_jitter * 0.5).fract();
                    if h < 0.0 {
                        h += 1.0;
                    }
                }
                if self.properties.brightness_jitter > 0.01 {
                    let bh = Self::stamp_hash(
                        pos.0 + 0.3,
                        pos.1 + 0.4,
                        self.stamp_counter.wrapping_add(555),
                    ) as f32
                        / u32::MAX as f32;
                    l = (l + (bh * 2.0 - 1.0) * self.properties.brightness_jitter * 0.5)
                        .clamp(0.0, 1.0);
                }
                let (nr, ng, nb) = crate::ops::adjustments::hsl_to_rgb(h, s, l);
                ((nr * 255.0) as u8, (ng * 255.0) as u8, (nb * 255.0) as u8)
            } else {
                (base_r8, base_g8, base_b8)
            };

        let cs = crate::canvas::CHUNK_SIZE;
        let mask = &self.brush_tip_mask;

        // Chunk iteration (same pattern as draw_circle_no_dirty)
        let chunk_x0 = stamp_min_x / cs;
        let chunk_y0 = stamp_min_y / cs;
        let chunk_x1 = stamp_max_x / cs;
        let chunk_y1 = stamp_max_y / cs;

        for chunk_cy in chunk_y0..=chunk_y1 {
            for chunk_cx in chunk_x0..=chunk_x1 {
                let chunk_base_x = chunk_cx * cs;
                let chunk_base_y = chunk_cy * cs;

                let lx0 = stamp_min_x.saturating_sub(chunk_base_x);
                let ly0 = stamp_min_y.saturating_sub(chunk_base_y);
                let lx1 = (stamp_max_x + 1 - chunk_base_x)
                    .min(cs)
                    .min(width - chunk_base_x);
                let ly1 = (stamp_max_y + 1 - chunk_base_y)
                    .min(cs)
                    .min(height - chunk_base_y);
                if lx0 >= lx1 || ly0 >= ly1 {
                    continue;
                }

                let chunk = target_image.ensure_chunk_mut(chunk_cx, chunk_cy);
                let chunk_raw = chunk.as_mut();
                let chunk_stride = cs as usize * 4;

                for ly in ly0..ly1 {
                    let global_y = chunk_base_y + ly;
                    let row_off = ly as usize * chunk_stride;

                    for lx in lx0..lx1 {
                        let global_x = chunk_base_x + lx;

                        // Selection mask check
                        if let Some(mask_img) = selection_mask {
                            if global_x < mask_img.width() && global_y < mask_img.height() {
                                if mask_img.get_pixel(global_x, global_y).0[0] == 0 {
                                    continue;
                                }
                            } else {
                                continue;
                            }
                        }

                        // Map global pixel to mask coordinate, applying inverse rotation
                        let rel_x = global_x as f32 - cx;
                        let rel_y = global_y as f32 - cy;

                        let geom_alpha_u8 = if rotated {
                            // Inverse-rotate to find source mask position
                            let rot_x = rel_x * cos_a - rel_y * sin_a + half;
                            let rot_y = rel_x * sin_a + rel_y * cos_a + half;
                            // Bilinear sample from the unrotated mask
                            if rot_x < -0.5
                                || rot_y < -0.5
                                || rot_x >= mask_size as f32 - 0.5
                                || rot_y >= mask_size as f32 - 0.5
                            {
                                continue;
                            }
                            let sx = rot_x.max(0.0);
                            let sy = rot_y.max(0.0);
                            let sx0 = sx.floor() as u32;
                            let sy0 = sy.floor() as u32;
                            let sx1 = (sx0 + 1).min(mask_size - 1);
                            let sy1 = (sy0 + 1).min(mask_size - 1);
                            let fx = sx - sx0 as f32;
                            let fy = sy - sy0 as f32;
                            let v00 = mask[(sy0 * mask_size + sx0) as usize] as f32;
                            let v10 = mask[(sy0 * mask_size + sx1) as usize] as f32;
                            let v01 = mask[(sy1 * mask_size + sx0) as usize] as f32;
                            let v11 = mask[(sy1 * mask_size + sx1) as usize] as f32;
                            let top = v00 * (1.0 - fx) + v10 * fx;
                            let bot = v01 * (1.0 - fx) + v11 * fx;
                            let val = top * (1.0 - fy) + bot * fy;
                            val.round().min(255.0) as u8
                        } else {
                            let mask_x = (rel_x + half).round() as i32;
                            let mask_y = (rel_y + half).round() as i32;
                            if mask_x < 0
                                || mask_y < 0
                                || mask_x >= mask_size as i32
                                || mask_y >= mask_size as i32
                            {
                                continue;
                            }
                            mask[(mask_y as u32 * mask_size + mask_x as u32) as usize]
                        };
                        if geom_alpha_u8 == 0 {
                            continue;
                        }
                        let geom_alpha = geom_alpha_u8 as f32 / 255.0;

                        let px_off = row_off + lx as usize * 4;

                        if is_eraser {
                            let erase_strength = geom_alpha * src_a * self.properties.flow;
                            if erase_strength < 0.01 {
                                continue;
                            }
                            let old_mask = chunk_raw[px_off + 3] as f32 / 255.0;
                            if erase_strength > old_mask {
                                chunk_raw[px_off] = 0;
                                chunk_raw[px_off + 1] = 0;
                                chunk_raw[px_off + 2] = 0;
                                chunk_raw[px_off + 3] = (erase_strength * 255.0) as u8;
                            }
                        } else {
                            let brush_alpha = geom_alpha * src_a * self.properties.flow;
                            let brush_alpha_u8 = (brush_alpha * 255.0) as u8;
                            let old_alpha = chunk_raw[px_off + 3];
                            if brush_alpha_u8 >= old_alpha {
                                chunk_raw[px_off] = src_r8;
                                chunk_raw[px_off + 1] = src_g8;
                                chunk_raw[px_off + 2] = src_b8;
                                chunk_raw[px_off + 3] = brush_alpha_u8;
                            }
                        }
                    }
                }
            }
        }
    }

    fn draw_line_no_dirty(
        &mut self,
        target_image: &mut TiledImage,
        width: u32,
        height: u32,
        start: (f32, f32),
        end: (f32, f32),
        is_eraser: bool,
        use_secondary: bool,
        primary_color_f32: [f32; 4],
        secondary_color_f32: [f32; 4],
        selection_mask: Option<&GrayImage>,
    ) {
        // Dense sub-pixel stepping for smooth lines
        let x0 = start.0;
        let y0 = start.1;
        let x1 = end.0;
        let y1 = end.1;

        let dx = x1 - x0;
        let dy = y1 - y0;
        let distance = (dx * dx + dy * dy).sqrt();

        if distance < 0.1 {
            // Just draw one circle at start
            if start.0 >= 0.0
                && (start.0 as u32) < width
                && start.1 >= 0.0
                && (start.1 as u32) < height
            {
                self.draw_circle_no_dirty(
                    target_image,
                    width,
                    height,
                    start,
                    is_eraser,
                    use_secondary,
                    primary_color_f32,
                    secondary_color_f32,
                    selection_mask,
                );
            }
            return;
        }

        // For image tips, use spacing-based stepping; for circle tips, dense per-pixel stepping
        let step = if self.properties.brush_tip.is_circle() {
            1.0
        } else {
            (self.properties.size * self.properties.spacing).max(1.0)
        };
        let steps = (distance / step).ceil() as usize;

        for i in 0..=steps {
            let t = i as f32 / steps as f32;
            let x = x0 + dx * t;
            let y = y0 + dy * t;

            // Pass float position directly — no rounding — for sub-pixel smooth circles
            if x >= 0.0 && (x as u32) < width && y >= 0.0 && (y as u32) < height {
                self.draw_circle_no_dirty(
                    target_image,
                    width,
                    height,
                    (x, y),
                    is_eraser,
                    use_secondary,
                    primary_color_f32,
                    secondary_color_f32,
                    selection_mask,
                );
            }
        }
    }

    fn mark_full_dirty(&self, canvas_state: &mut CanvasState) {
        canvas_state.dirty_rect = Some(Rect::from_min_max(
            Pos2::ZERO,
            Pos2::new(canvas_state.width as f32, canvas_state.height as f32),
        ));
    }

    /// Simple positional hash for pseudorandom per-stamp rotation.
    /// Produces a deterministic u32 from floating-point position + counter.
    fn stamp_hash(x: f32, y: f32, counter: u32) -> u32 {
        let ix = (x * 100.0) as u32;
        let iy = (y * 100.0) as u32;
        let mut h = ix
            .wrapping_mul(374761393)
            .wrapping_add(iy.wrapping_mul(668265263))
            .wrapping_add(counter.wrapping_mul(1013904223));
        h ^= h >> 13;
        h = h.wrapping_mul(1274126177);
        h ^= h >> 16;
        h
    }

    /// Draw a circle immediately to pixels and return its bounding box
    fn draw_circle_and_get_bounds(
        &mut self,
        canvas_state: &mut CanvasState,
        pos: (f32, f32),
        is_eraser: bool,
        use_secondary: bool,
        primary_color_f32: [f32; 4],
        secondary_color_f32: [f32; 4],
    ) -> Rect {
        // B6: Ensure brush alpha LUT is up-to-date
        self.rebuild_brush_lut();
        // Increment stamp counter for random rotation seeding
        self.stamp_counter = self.stamp_counter.wrapping_add(1);

        let (cx, cy) = pos;
        let radius = self.properties.size / 2.0;

        // Calculate bounds — expanded by max scatter offset so the dirty rect
        // covers stamps that landed far from the nominal position.
        let width = canvas_state.width;
        let height = canvas_state.height;
        // scatter moves center by up to scatter * size per axis
        let scatter_pad = self.properties.scatter * self.properties.size;

        let min_x = (cx - radius - scatter_pad).max(0.0) as u32;
        let max_x = ((cx + radius + scatter_pad) as u32).min(width - 1);
        let min_y = (cy - radius - scatter_pad).max(0.0) as u32;
        let max_y = ((cy + radius + scatter_pad) as u32).min(height - 1);

        // All tools (Brush, Pencil, Eraser) write to the preview layer.
        // The eraser writes an erase-strength mask; the compositor handles the rest.
        {
            let mask_ptr = canvas_state
                .selection_mask
                .as_ref()
                .map(|m| m as *const GrayImage);
            if let Some(ref mut preview) = canvas_state.preview_layer {
                let mask_ref = mask_ptr.map(|p| unsafe { &*p });
                self.draw_circle_no_dirty(
                    preview,
                    width,
                    height,
                    pos,
                    is_eraser,
                    use_secondary,
                    primary_color_f32,
                    secondary_color_f32,
                    mask_ref,
                );
            }
        }

        // Return the bounding box (add 1 pixel padding)
        Rect::from_min_max(
            Pos2::new(
                min_x.saturating_sub(1) as f32,
                min_y.saturating_sub(1) as f32,
            ),
            Pos2::new(
                (max_x + 2).min(width) as f32,
                (max_y + 2).min(height) as f32,
            ),
        )
    }

    // ================================================================
    // CLONE STAMP helpers
    // ================================================================

    /// Stamp a single circle for clone stamp: sample from source layer at offset
    fn clone_stamp_circle(
        &self,
        canvas_state: &mut CanvasState,
        pos: (f32, f32),
        offset: Vec2,
    ) -> Rect {
        let (cx, cy) = pos;
        let radius = self.properties.size / 2.0;
        let width = canvas_state.width;
        let height = canvas_state.height;

        let min_x = (cx - radius).max(0.0) as u32;
        let max_x = ((cx + radius) as u32).min(width.saturating_sub(1));
        let min_y = (cy - radius).max(0.0) as u32;
        let max_y = ((cy + radius) as u32).min(height.saturating_sub(1));

        // Get source layer pointer for safe aliasing
        let layer = match canvas_state.layers.get(canvas_state.active_layer_index) {
            Some(l) => l,
            None => return Rect::NOTHING,
        };
        let src_ptr = &layer.pixels as *const TiledImage;
        let sel_ptr = canvas_state
            .selection_mask
            .as_ref()
            .map(|m| m as *const GrayImage);

        if let Some(ref mut preview) = canvas_state.preview_layer {
            for y in min_y..=max_y {
                for x in min_x..=max_x {
                    // Selection mask check
                    if let Some(mask_p) = sel_ptr {
                        let mask = unsafe { &*mask_p };
                        if x < mask.width() && y < mask.height() {
                            if mask.get_pixel(x, y).0[0] == 0 {
                                continue;
                            }
                        } else {
                            continue;
                        }
                    }

                    let dx = x as f32 - cx;
                    let dy = y as f32 - cy;
                    let dist = (dx * dx + dy * dy).sqrt();
                    if dist > radius {
                        continue;
                    }

                    let geom_alpha = self.compute_brush_alpha(dist, radius);
                    if geom_alpha < 0.01 {
                        continue;
                    }

                    // Source coordinates
                    let sx = (x as f32 + offset.x).round() as i32;
                    let sy = (y as f32 + offset.y).round() as i32;
                    if sx < 0 || sx >= width as i32 || sy < 0 || sy >= height as i32 {
                        continue;
                    }

                    let src_pixel = unsafe { &*src_ptr }.get_pixel(sx as u32, sy as u32);
                    let sr = src_pixel.0[0] as f32 / 255.0;
                    let sg = src_pixel.0[1] as f32 / 255.0;
                    let sb = src_pixel.0[2] as f32 / 255.0;
                    let sa = src_pixel.0[3] as f32 / 255.0;

                    let brush_alpha = geom_alpha * sa;
                    let old_alpha = preview.get_pixel(x, y).0[3] as f32 / 255.0;

                    if brush_alpha >= old_alpha {
                        *preview.get_pixel_mut(x, y) = Rgba([
                            (sr * 255.0) as u8,
                            (sg * 255.0) as u8,
                            (sb * 255.0) as u8,
                            (brush_alpha * 255.0) as u8,
                        ]);
                    }
                }
            }
        }

        Rect::from_min_max(
            Pos2::new(
                min_x.saturating_sub(1) as f32,
                min_y.saturating_sub(1) as f32,
            ),
            Pos2::new(
                (max_x + 2).min(width) as f32,
                (max_y + 2).min(height) as f32,
            ),
        )
    }

    /// Draw a line for clone stamp — dense stepping like draw_line_no_dirty
    fn clone_stamp_line(
        &self,
        canvas_state: &mut CanvasState,
        start: (f32, f32),
        end: (f32, f32),
        offset: Vec2,
    ) -> Rect {
        let dx = end.0 - start.0;
        let dy = end.1 - start.1;
        let distance = (dx * dx + dy * dy).sqrt();

        if distance < 0.1 {
            return self.clone_stamp_circle(canvas_state, start, offset);
        }

        let steps = distance.ceil() as usize;
        let mut dirty = Rect::NOTHING;

        for i in 0..=steps {
            let t = i as f32 / steps as f32;
            let x = start.0 + dx * t;
            let y = start.1 + dy * t;
            if x >= 0.0
                && (x as u32) < canvas_state.width
                && y >= 0.0
                && (y as u32) < canvas_state.height
            {
                dirty = dirty.union(self.clone_stamp_circle(canvas_state, (x, y), offset));
            }
        }
        dirty
    }

    // ================================================================
    // CONTENT AWARE BRUSH (healing) helpers
    // ================================================================

    /// Heal a single circle: for each pixel inside the brush, sample surrounding
    /// pixels from a ring and replace with their average. This effectively
    /// "fills in" the area with surrounding texture.
    fn heal_circle(&self, canvas_state: &mut CanvasState, pos: (f32, f32)) -> Rect {
        let (cx, cy) = pos;
        let radius = self.properties.size / 2.0;
        let sample_radius = self.content_aware_state.sample_radius;
        let hardness = self.properties.hardness;
        let width = canvas_state.width;
        let height = canvas_state.height;

        let min_x = (cx - radius).max(0.0) as u32;
        let max_x = ((cx + radius) as u32).min(width.saturating_sub(1));
        let min_y = (cy - radius).max(0.0) as u32;
        let max_y = ((cy + radius) as u32).min(height.saturating_sub(1));

        let layer = match canvas_state.layers.get(canvas_state.active_layer_index) {
            Some(l) => l,
            None => return Rect::NOTHING,
        };
        let src_ptr = &layer.pixels as *const TiledImage;
        let sel_ptr = canvas_state
            .selection_mask
            .as_ref()
            .map(|m| m as *const GrayImage);

        // Sample at two rings (75% and 100% of sample_radius) with per-pixel
        // angle randomisation to avoid visible sampling-grid artifacts.
        let num_samples: usize = 24;

        if let Some(ref mut preview) = canvas_state.preview_layer {
            let src = unsafe { &*src_ptr };

            for y in min_y..=max_y {
                for x in min_x..=max_x {
                    // Selection mask
                    if let Some(mask_p) = sel_ptr {
                        let mask = unsafe { &*mask_p };
                        if x < mask.width() && y < mask.height() {
                            if mask.get_pixel(x, y).0[0] == 0 {
                                continue;
                            }
                        } else {
                            continue;
                        }
                    }

                    let dx = x as f32 - cx;
                    let dy = y as f32 - cy;
                    let dist = (dx * dx + dy * dy).sqrt();
                    if dist > radius {
                        continue;
                    }

                    // Hardness-aware brush alpha
                    let t = (dist / radius).clamp(0.0, 1.0);
                    let hard_t = (hardness * 0.9 + 0.1).clamp(0.0, 1.0);
                    let geom_alpha = if t < hard_t {
                        1.0
                    } else {
                        let s = (t - hard_t) / (1.0 - hard_t + 1e-6);
                        1.0 - s * s * (3.0 - 2.0 * s)
                    };
                    if geom_alpha < 0.01 {
                        continue;
                    }

                    // Per-pixel angle offset to break up ring-sampling grid artifacts
                    let angle_seed = (x.wrapping_mul(1619)).wrapping_add(y.wrapping_mul(3929));
                    let angle_offset = angle_seed as f32 / u32::MAX as f32 * std::f32::consts::TAU;

                    let mut sum_r = 0.0_f32;
                    let mut sum_g = 0.0_f32;
                    let mut sum_b = 0.0_f32;
                    let mut count = 0.0_f32;

                    for i in 0..num_samples {
                        let angle =
                            angle_offset + (i as f32 / num_samples as f32) * std::f32::consts::TAU;
                        // Sample at two concentric rings for smoother coverage
                        for &rr in &[sample_radius * 0.75, sample_radius] {
                            let sx = (x as f32 + angle.cos() * rr).round() as i32;
                            let sy = (y as f32 + angle.sin() * rr).round() as i32;
                            if sx < 0 || sx >= width as i32 || sy < 0 || sy >= height as i32 {
                                continue;
                            }
                            let sp = src.get_pixel(sx as u32, sy as u32);
                            sum_r += sp.0[0] as f32;
                            sum_g += sp.0[1] as f32;
                            sum_b += sp.0[2] as f32;
                            count += 1.0;
                        }
                    }

                    if count < 1.0 {
                        continue;
                    }

                    let old_alpha = preview.get_pixel(x, y).0[3] as f32 / 255.0;
                    if geom_alpha >= old_alpha {
                        *preview.get_pixel_mut(x, y) = Rgba([
                            (sum_r / count) as u8,
                            (sum_g / count) as u8,
                            (sum_b / count) as u8,
                            (geom_alpha * 255.0) as u8,
                        ]);
                    }
                }
            }
        }

        Rect::from_min_max(
            Pos2::new(
                min_x.saturating_sub(1) as f32,
                min_y.saturating_sub(1) as f32,
            ),
            Pos2::new(
                (max_x + 2).min(width) as f32,
                (max_y + 2).min(height) as f32,
            ),
        )
    }

    /// Draw a line for heal brush — dense stepping
    fn heal_line(
        &self,
        canvas_state: &mut CanvasState,
        start: (f32, f32),
        end: (f32, f32),
    ) -> Rect {
        let dx = end.0 - start.0;
        let dy = end.1 - start.1;
        let distance = (dx * dx + dy * dy).sqrt();

        if distance < 0.1 {
            return self.heal_circle(canvas_state, start);
        }

        let steps = distance.ceil() as usize;
        let mut dirty = Rect::NOTHING;

        for i in 0..=steps {
            let t = i as f32 / steps as f32;
            let x = start.0 + dx * t;
            let y = start.1 + dy * t;
            if x >= 0.0
                && (x as u32) < canvas_state.width
                && y >= 0.0
                && (y as u32) < canvas_state.height
            {
                dirty = dirty.union(self.heal_circle(canvas_state, (x, y)));
            }
        }
        dirty
    }

    /// Draw a single pixel immediately to pixels and return its bounding box
    fn draw_pixel_and_get_bounds(
        &mut self,
        canvas_state: &mut CanvasState,
        pos: (f32, f32),
        use_secondary: bool,
        primary_color_f32: [f32; 4],
        secondary_color_f32: [f32; 4],
    ) -> Rect {
        let (cx, cy) = pos;

        // Calculate bounds (single pixel)
        let width = canvas_state.width;
        let height = canvas_state.height;

        let x = cx.floor() as u32;
        let y = cy.floor() as u32;

        // Check bounds
        if x >= width || y >= height {
            return Rect::NOTHING;
        }

        // Determine which color to use (high-precision unmultiplied)
        let pixel_color_f32 = if use_secondary {
            secondary_color_f32
        } else {
            primary_color_f32
        };

        // Unpack high-precision source
        let [src_r, src_g, src_b, src_a] = pixel_color_f32;

        // Skip pixels outside the selection mask
        if let Some(mask) = &canvas_state.selection_mask {
            if x < mask.width() && y < mask.height() {
                if mask.get_pixel(x, y).0[0] == 0 {
                    return Rect::NOTHING;
                }
            } else {
                return Rect::NOTHING;
            }
        }

        // Get the target image (preview layer for pencil)
        if let Some(ref mut preview) = canvas_state.preview_layer {
            let pixel = preview.get_pixel_mut(x, y);

            // For pencil, use Max-Alpha blending to prevent opacity accumulation
            // when dragging over the same area. This is the same technique the
            // brush uses to prevent opacity stacking.
            let pencil_alpha = src_a; // 1.0 for pencil ink (full opacity color)

            // Read existing alpha from the preview layer
            let old_alpha = pixel.0[3] as f32 / 255.0;

            // Only update pixel if we're increasing opacity (or equal)
            // This prevents less-opaque strokes from overwriting more-opaque ones
            if pencil_alpha >= old_alpha {
                *pixel = Rgba([
                    (src_r * 255.0) as u8,
                    (src_g * 255.0) as u8,
                    (src_b * 255.0) as u8,
                    (pencil_alpha * 255.0) as u8,
                ]);
            }
        }

        // Return the bounding box (single pixel with 1 pixel padding)
        Rect::from_min_max(
            Pos2::new(x.saturating_sub(1) as f32, y.saturating_sub(1) as f32),
            Pos2::new((x + 2).min(width) as f32, (y + 2).min(height) as f32),
        )
    }

    /// Draw a line of single pixels and return its bounding box
    fn draw_pixel_line_and_get_bounds(
        &mut self,
        canvas_state: &mut CanvasState,
        start: (f32, f32),
        end: (f32, f32),
        use_secondary: bool,
        primary_color_f32: [f32; 4],
        secondary_color_f32: [f32; 4],
    ) -> Rect {
        let width = canvas_state.width;
        let height = canvas_state.height;

        // Calculate line bounds
        let _min_x = start.0.min(end.0) as u32;
        let _max_x = end.0.max(start.0) as u32;
        let _min_y = start.1.min(end.1) as u32;
        let _max_y = end.1.max(start.1) as u32;

        // Use Bresenham's line algorithm to draw pixel-perfect line
        let mut x0 = start.0.floor() as i32;
        let mut y0 = start.1.floor() as i32;
        let x1 = end.0.floor() as i32;
        let y1 = end.1.floor() as i32;

        let dx = (x1 - x0).abs();
        let dy = (y1 - y0).abs();
        let sx = if x0 < x1 { 1 } else { -1 };
        let sy = if y0 < y1 { 1 } else { -1 };
        let mut err = dx - dy;

        let mut bounds = Rect::NOTHING;

        loop {
            // Check bounds
            if x0 >= 0 && x0 < width as i32 && y0 >= 0 && y0 < height as i32 {
                let pixel_rect = self.draw_pixel_and_get_bounds(
                    canvas_state,
                    (x0 as f32, y0 as f32),
                    use_secondary,
                    primary_color_f32,
                    secondary_color_f32,
                );
                bounds = bounds.union(pixel_rect);
            }

            if x0 == x1 && y0 == y1 {
                break;
            }

            let e2 = 2 * err;
            if e2 > -dy {
                err -= dy;
                x0 += sx;
            }
            if e2 < dx {
                err += dx;
                y0 += sy;
            }
        }

        bounds
    }

    /// Draw a line immediately to pixels and return its bounding box
    fn draw_line_and_get_bounds(
        &mut self,
        canvas_state: &mut CanvasState,
        start: (f32, f32),
        end: (f32, f32),
        is_eraser: bool,
        use_secondary: bool,
        primary_color_f32: [f32; 4],
        secondary_color_f32: [f32; 4],
    ) -> Rect {
        // B6: Ensure brush alpha LUT is up-to-date
        self.rebuild_brush_lut();
        // Increment stamp counter for random rotation seeding (for the whole line)
        self.stamp_counter = self.stamp_counter.wrapping_add(1);

        let radius = self.properties.size / 2.0;
        let width = canvas_state.width;
        let height = canvas_state.height;

        // Calculate bounding box of the line + brush radius + max scatter offset
        let scatter_pad = self.properties.scatter * self.properties.size;
        let min_x = (start.0.min(end.0) - radius - scatter_pad).max(0.0) as u32;
        let max_x = ((start.0.max(end.0) + radius + scatter_pad) as u32).min(width);
        let min_y = (start.1.min(end.1) - radius - scatter_pad).max(0.0) as u32;
        let max_y = ((start.1.max(end.1) + radius + scatter_pad) as u32).min(height);

        // All tools (Brush, Pencil, Eraser) write to the preview layer.
        // The eraser writes an erase-strength mask; the compositor handles the rest.
        {
            let mask_ptr = canvas_state
                .selection_mask
                .as_ref()
                .map(|m| m as *const GrayImage);
            if let Some(ref mut preview) = canvas_state.preview_layer {
                let mask_ref = mask_ptr.map(|p| unsafe { &*p });
                self.draw_line_no_dirty(
                    preview,
                    width,
                    height,
                    start,
                    end,
                    is_eraser,
                    use_secondary,
                    primary_color_f32,
                    secondary_color_f32,
                    mask_ref,
                );
            }
        }

        // Return the bounding box (add 1 pixel padding)
        Rect::from_min_max(
            Pos2::new(
                min_x.saturating_sub(1) as f32,
                min_y.saturating_sub(1) as f32,
            ),
            Pos2::new(
                (max_x + 2).min(width) as f32,
                (max_y + 2).min(height) as f32,
            ),
        )
    }

    fn canvas_to_screen(&self, canvas_pos: (u32, u32), canvas_rect: Rect, zoom: f32) -> Pos2 {
        let (x, y) = canvas_pos;
        Pos2::new(
            canvas_rect.min.x + x as f32 * zoom,
            canvas_rect.min.y + y as f32 * zoom,
        )
    }

    /// Convert canvas pixel coordinates to screen Pos2 (float version)
    fn canvas_pos2_to_screen(&self, canvas_pos: Pos2, canvas_rect: Rect, zoom: f32) -> Pos2 {
        Pos2::new(
            canvas_rect.min.x + canvas_pos.x * zoom,
            canvas_rect.min.y + canvas_pos.y * zoom,
        )
    }

    /// Convert screen Pos2 to canvas pixel coordinates
    fn screen_to_canvas_pos2(&self, screen_pos: Pos2, canvas_rect: Rect, zoom: f32) -> Pos2 {
        Pos2::new(
            (screen_pos.x - canvas_rect.min.x) / zoom,
            (screen_pos.y - canvas_rect.min.y) / zoom,
        )
    }

    /// Calculate cubic Bézier point at t (0.0 to 1.0)
    fn bezier_point(&self, p0: Pos2, p1: Pos2, p2: Pos2, p3: Pos2, t: f32) -> Pos2 {
        let t2 = t * t;
        let t3 = t2 * t;
        let mt = 1.0 - t;
        let mt2 = mt * mt;
        let mt3 = mt2 * mt;

        Pos2::new(
            mt3 * p0.x + 3.0 * mt2 * t * p1.x + 3.0 * mt * t2 * p2.x + t3 * p3.x,
            mt3 * p0.y + 3.0 * mt2 * t * p1.y + 3.0 * mt * t2 * p2.y + t3 * p3.y,
        )
    }

    /// Calculate bounding box for Bézier curve based on control points
    pub fn get_bezier_bounds(
        &self,
        control_points: [Pos2; 4],
        canvas_width: u32,
        canvas_height: u32,
    ) -> Rect {
        // Find min/max X and Y of control points
        let mut min_x = control_points[0].x;
        let mut max_x = control_points[0].x;
        let mut min_y = control_points[0].y;
        let mut max_y = control_points[0].y;

        for point in &control_points[1..] {
            min_x = min_x.min(point.x);
            max_x = max_x.max(point.x);
            min_y = min_y.min(point.y);
            max_y = max_y.max(point.y);
        }

        // Expand by brush size + padding
        let padding = self.properties.size / 2.0 + 2.0;
        min_x = (min_x - padding).max(0.0);
        min_y = (min_y - padding).max(0.0);
        max_x = (max_x + padding).min(canvas_width as f32);
        max_y = (max_y + padding).min(canvas_height as f32);

        Rect::from_min_max(Pos2::new(min_x, min_y), Pos2::new(max_x, max_y))
    }

    /// Rasterize a cubic Bézier curve to the preview layer
    pub fn rasterize_bezier(
        &self,
        canvas_state: &mut CanvasState,
        control_points: [Pos2; 4],
        color: Color32,
        pattern: LinePattern,
        cap_style: CapStyle,
        last_bounds: Option<Rect>, // Previous frame's bounds for focused clearing
    ) {
        // Convert Color32 to high-precision f32 for drawing
        let color_f32 = [
            color.r() as f32 / 255.0,
            color.g() as f32 / 255.0,
            color.b() as f32 / 255.0,
            color.a() as f32 / 255.0,
        ];
        // Create or clear preview layer
        if canvas_state.preview_layer.is_none()
            || canvas_state.preview_layer.as_ref().unwrap().width() != canvas_state.width
            || canvas_state.preview_layer.as_ref().unwrap().height() != canvas_state.height
        {
            canvas_state.preview_layer =
                Some(TiledImage::new(canvas_state.width, canvas_state.height));
        }
        // Line tool uses the tool's blending mode for preview
        canvas_state.preview_blend_mode = self.properties.blending_mode;

        // OPTIMIZATION: Only clear the previous frame's bounding box instead of entire image
        if let Some(ref mut preview) = canvas_state.preview_layer {
            if let Some(bounds) = last_bounds {
                // Focused clear: drop chunks that overlap the previous bounds
                let min_x = bounds.min.x.floor().max(0.0) as u32;
                let max_x = bounds.max.x.ceil().min(canvas_state.width as f32) as u32;
                let min_y = bounds.min.y.floor().max(0.0) as u32;
                let max_y = bounds.max.y.ceil().min(canvas_state.height as f32) as u32;
                preview.clear_region(min_x, min_y, max_x, max_y);
            } else {
                // First frame: clear entire layer
                preview.clear();
            }

            let [p0, p1, p2, p3] = control_points;

            // Much tighter spacing for smooth lines - use 10% of size for better quality
            let spacing = (self.properties.size * 0.1).max(0.5);

            // Adaptive sampling based on curve length estimate
            let chord_len = (p3 - p0).length();
            let control_net_len = (p1 - p0).length() + (p2 - p1).length() + (p3 - p2).length();
            let total_length = control_net_len + chord_len;

            // Calculate steps based on tighter spacing for smooth rendering
            let steps = (total_length / spacing).ceil() as usize;
            let steps = steps.clamp(20, 5000); // Higher max for quality

            // Track cumulative distance for pattern rendering
            let mut cumulative_distance = 0.0;
            let mut last_pos: Option<Pos2> = None;

            // Pattern parameters (in pixels)
            let (on_length, off_length) = match pattern {
                LinePattern::Solid => (0.0, 0.0), // No pattern
                LinePattern::Dotted => (self.properties.size * 0.5, self.properties.size * 1.5), // Dot size, larger gap
                LinePattern::Dashed => (self.properties.size * 2.0, self.properties.size * 1.5), // Dash, gap
            };
            let pattern_cycle = on_length + off_length;

            // Collect all points first for cap style processing
            let mut line_points = Vec::new();

            for i in 0..=steps {
                let t = i as f32 / steps as f32;
                let pos = self.bezier_point(p0, p1, p2, p3, t);

                // Update cumulative distance
                if let Some(prev_pos) = last_pos {
                    cumulative_distance += (pos - prev_pos).length();
                }
                last_pos = Some(pos);

                let fx = pos.x;
                let fy = pos.y;

                if fx >= 0.0
                    && fy >= 0.0
                    && (fx as u32) < canvas_state.width
                    && (fy as u32) < canvas_state.height
                {
                    // Check pattern based on cumulative distance
                    let should_draw = match pattern {
                        LinePattern::Solid => true,
                        LinePattern::Dotted | LinePattern::Dashed => {
                            let pos_in_cycle = cumulative_distance % pattern_cycle;
                            pos_in_cycle < on_length
                        }
                    };

                    if should_draw {
                        line_points.push((fx, fy, i == 0, i == steps));
                    }
                }
            }

            // Draw all points
            for &(fx, fy, is_start, is_end) in &line_points {
                // For flat caps, skip drawing circles at the very endpoints
                // This creates a clean, flat termination of the line
                if cap_style == CapStyle::Flat && (is_start || is_end) {
                    // Skip the endpoint entirely for flat caps - line just stops
                    continue;
                }

                // Normal drawing for round caps or middle points
                self.draw_bezier_dot(preview, (fx, fy), color_f32);
            }
        }
    }

    /// Draw a segment as part of Bézier curve with spacing
    fn draw_bezier_segment(
        &self,
        preview: &mut TiledImage,
        start: (f32, f32),
        end: (f32, f32),
        color_f32: [f32; 4],
    ) {
        let (x0, y0) = start;
        let (x1, y1) = end;

        // Calculate spacing (25% of brush diameter)
        let spacing = (self.properties.size * 0.25).max(1.0);
        let radius = (self.properties.size / 2.0).max(1.0);

        // Calculate distance
        let dx = x1 - x0;
        let dy = y1 - y0;
        let distance = (dx * dx + dy * dy).sqrt();

        // If distance is too small, just draw one circle
        if distance < 0.1 {
            if (start.0 as u32) < preview.width() && (start.1 as u32) < preview.height() {
                self.draw_bezier_circle_with_hardness(preview, start, color_f32, radius, 0.3);
            }
            return;
        }

        // Calculate number of steps
        let num_steps = (distance / spacing).ceil() as usize;

        // Draw circles at spaced intervals
        for i in 0..=num_steps {
            let t = (i as f32 * spacing / distance).min(1.0);
            let x = x0 + dx * t;
            let y = y0 + dy * t;

            if (x as u32) < preview.width() && (y as u32) < preview.height() {
                self.draw_bezier_circle_with_hardness(preview, (x, y), color_f32, radius, 0.3);
            }
        }

        // Do NOT force-draw the end point - only draw at spacing intervals
    }

    /// Draw a single dot for Bézier curve (sub-pixel position for smooth AA)
    fn draw_bezier_dot(&self, preview: &mut TiledImage, pos: (f32, f32), color_f32: [f32; 4]) {
        let radius = (self.properties.size / 2.0).max(1.0);
        // Force softer rendering for line tool by using low hardness (0.3 = very soft edges)
        self.draw_bezier_circle_with_hardness(preview, pos, color_f32, radius, 0.3);
    }

    /// Draw a circle on the preview layer for Bézier curves with custom hardness override
    /// Uses sub-pixel (f32) center for smooth anti-aliasing without grid artifacts
    fn draw_bezier_circle_with_hardness(
        &self,
        preview: &mut TiledImage,
        pos: (f32, f32),
        color_f32: [f32; 4],
        radius: f32,
        forced_hardness: f32,
    ) {
        let (cx, cy) = pos;

        // Extend the sampling area beyond the nominal radius so the soft fade
        // region actually gets drawn.  For softness 0.3 the fade extends ~70% of
        // radius beyond the solid core; we add a generous 2px on top for tiny
        // brushes.
        let aa_pad = (radius * (1.0 - forced_hardness)).max(2.0) + 2.0;
        let outer_radius = radius + aa_pad;

        let min_x = (cx - outer_radius).max(0.0) as u32;
        let max_x = ((cx + outer_radius).ceil() as u32).min(preview.width() - 1);
        let min_y = (cy - outer_radius).max(0.0) as u32;
        let max_y = ((cy + outer_radius).ceil() as u32).min(preview.height() - 1);

        // Unpack high-precision source
        let [src_r, src_g, src_b, src_a] = color_f32;

        for y in min_y..=max_y {
            for x in min_x..=max_x {
                // Sub-pixel distance from float center for smooth AA
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let dist = (dx * dx + dy * dy).sqrt();

                // Compute alpha — this returns 0.0 for pixels beyond the
                // effective (soft) radius, so no explicit dist guard needed.
                let alpha = self.compute_line_alpha(dist, radius, forced_hardness);
                if alpha <= 0.0 {
                    continue;
                }

                let pixel_alpha = alpha * src_a;

                // MAX blending: keep the highest alpha at each pixel.
                // Since every stamp uses the same RGB colour, taking max(alpha)
                // produces perfectly smooth edges with no scalloping from
                // overlapping circle stamps.
                let pixel = preview.get_pixel_mut(x, y);
                let base_a = pixel[3] as f32 / 255.0;

                if pixel_alpha > base_a {
                    *pixel = Rgba([
                        (src_r * 255.0) as u8,
                        (src_g * 255.0) as u8,
                        (src_b * 255.0) as u8,
                        (pixel_alpha * 255.0) as u8,
                    ]);
                }
            }
        }
    }

    /// Magic Wand selection - flood select based on color tolerance
    fn perform_magic_wand_selection(
        &mut self,
        canvas_state: &mut CanvasState,
        start_pos: (u32, u32),
        mode: SelectionMode,
    ) {
        // Bounds check
        if start_pos.0 >= canvas_state.width || start_pos.1 >= canvas_state.height {
            return;
        }

        // Get the color at the clicked position from the active layer
        let active_layer = match canvas_state.layers.get(canvas_state.active_layer_index) {
            Some(layer) => layer,
            None => return,
        };

        let target_color = active_layer.pixels.get_pixel(start_pos.0, start_pos.1);

        // Save mode and base mask for merging
        self.magic_wand_state.effective_mode = mode;
        if mode != SelectionMode::Replace {
            // Save the current selection mask as the base to merge with
            self.magic_wand_state.base_selection_mask = canvas_state.selection_mask.clone();
        } else {
            self.magic_wand_state.base_selection_mask = None;
        }

        // Store the selection state for live preview
        self.magic_wand_state.active_selection =
            Some((start_pos.0, start_pos.1, *target_color, Vec::new()));
        self.magic_wand_state.last_preview_tolerance = self.magic_wand_state.tolerance;

        // Force immediate recalculation
        self.magic_wand_state.recalc_pending = true;
        self.magic_wand_state.last_preview_tolerance = -1.0; // Force update
        self.update_magic_wand_preview(canvas_state);
    }

    /// Update magic wand preview if tolerance changed (debounced)
    fn update_magic_wand_preview(&mut self, canvas_state: &mut CanvasState) {
        // Only recalculate if a change is pending
        if !self.magic_wand_state.recalc_pending {
            return;
        }

        // Debounce: wait 50ms after last change before recalculating
        if let Some(changed_at) = self.magic_wand_state.tolerance_changed_at
            && changed_at.elapsed().as_millis() < 50
        {
            return; // Too soon, wait for the user to stop adjusting
        }

        // Check if tolerance actually changed
        if (self.magic_wand_state.tolerance - self.magic_wand_state.last_preview_tolerance).abs()
            < 0.01
        {
            self.magic_wand_state.recalc_pending = false;
            return;
        }

        // Get active selection data
        let (start_x, start_y, target_color, _) = match &self.magic_wand_state.active_selection {
            Some(data) => data,
            None => return,
        };

        // Get the active layer
        let active_layer = match canvas_state.layers.get(canvas_state.active_layer_index) {
            Some(layer) => layer,
            None => return,
        };

        // Convert tolerance (0-100%) to distance threshold
        // colors_match uses max-component distance (range 0-255), so map tolerance to that range
        let tolerance_threshold = (self.magic_wand_state.tolerance / 100.0) * 255.0;

        // Perform flood fill to find all connected pixels, or global select for all matching pixels
        let selected_pixels = {
            let pixels = &active_layer.pixels;
            if self.magic_wand_state.global_select {
                self.global_color_select(
                    pixels,
                    target_color,
                    tolerance_threshold,
                    canvas_state.width,
                    canvas_state.height,
                )
            } else {
                self.flood_fill_selection(
                    pixels,
                    (*start_x, *start_y),
                    target_color,
                    tolerance_threshold,
                    canvas_state.width,
                    canvas_state.height,
                )
            }
        };

        // Update the stored selection
        self.magic_wand_state.active_selection =
            Some((*start_x, *start_y, *target_color, selected_pixels.clone()));
        self.magic_wand_state.last_preview_tolerance = self.magic_wand_state.tolerance;
        self.magic_wand_state.recalc_pending = false;
        self.magic_wand_state.tolerance_changed_at = None;

        // Create a selection mask from the flood fill result
        let w = canvas_state.width;
        let h = canvas_state.height;

        // Build the new selection from flood fill result
        let mut new_mask = GrayImage::new(w, h);
        for (x, y) in selected_pixels.iter() {
            new_mask.put_pixel(*x, *y, Luma([255]));
        }

        let effective_mode = self.magic_wand_state.effective_mode;
        let final_mask = match effective_mode {
            SelectionMode::Replace => new_mask,
            SelectionMode::Add => {
                let mut base = self
                    .magic_wand_state
                    .base_selection_mask
                    .clone()
                    .unwrap_or_else(|| GrayImage::new(w, h));
                for (x, y) in selected_pixels.iter() {
                    base.put_pixel(*x, *y, Luma([255]));
                }
                base
            }
            SelectionMode::Subtract => {
                let mut base = self
                    .magic_wand_state
                    .base_selection_mask
                    .clone()
                    .unwrap_or_else(|| GrayImage::new(w, h));
                for (x, y) in selected_pixels.iter() {
                    base.put_pixel(*x, *y, Luma([0]));
                }
                base
            }
            SelectionMode::Intersect => {
                let base = self
                    .magic_wand_state
                    .base_selection_mask
                    .clone()
                    .unwrap_or_else(|| GrayImage::new(w, h));
                let selected_set: std::collections::HashSet<(u32, u32)> =
                    selected_pixels.iter().copied().collect();
                let mut intersect = GrayImage::new(w, h);
                for (x, y, pixel) in base.enumerate_pixels() {
                    if pixel[0] > 0 && selected_set.contains(&(x, y)) {
                        intersect.put_pixel(x, y, Luma([255]));
                    }
                }
                intersect
            }
        };

        // Apply the selection to the canvas state
        canvas_state.selection_mask = Some(final_mask);
        canvas_state.invalidate_selection_overlay();
        canvas_state.mark_dirty(None);
    }

    /// Fill tool - flood fill with preview
    fn perform_flood_fill(
        &mut self,
        canvas_state: &mut CanvasState,
        start_pos: (u32, u32),
        use_secondary: bool,
        primary_color_f32: [f32; 4],
        secondary_color_f32: [f32; 4],
    ) {
        // Bounds check
        if start_pos.0 >= canvas_state.width || start_pos.1 >= canvas_state.height {
            return;
        }

        // Get the active layer to determine target color
        let active_layer = match canvas_state.layers.get(canvas_state.active_layer_index) {
            Some(layer) => layer,
            None => return,
        };

        let target_color = *active_layer.pixels.get_pixel(start_pos.0, start_pos.1);

        // Determine fill color from primary or secondary
        let fill_color = if use_secondary {
            secondary_color_f32
        } else {
            primary_color_f32
        };
        let fill_color_u8 = Rgba([
            (fill_color[0] * 255.0) as u8,
            (fill_color[1] * 255.0) as u8,
            (fill_color[2] * 255.0) as u8,
            (fill_color[3] * 255.0) as u8,
        ]);

        let tolerance_threshold = (self.fill_state.tolerance / 100.0) * 255.0;

        // Pre-extract layer as flat RGBA — avoids per-pixel TiledImage chunk lookups
        let flat_rgba = active_layer.pixels.to_rgba_image();
        let flat_slice = flat_rgba.as_raw();

        let (fill_mask, bbox) = Self::flood_fill_fast(
            flat_slice,
            start_pos.0,
            start_pos.1,
            &target_color,
            tolerance_threshold,
            canvas_state.width,
            canvas_state.height,
        );

        // Store active fill state for preview updates
        self.fill_state.active_fill =
            Some((start_pos.0, start_pos.1, target_color, fill_mask.clone()));
        self.fill_state.last_preview_tolerance = self.fill_state.tolerance;
        self.fill_state.last_preview_aa = self.fill_state.anti_aliased;
        self.fill_state.fill_color_u8 = Some(fill_color_u8);
        self.fill_state.use_secondary_color = use_secondary;

        // Initialize preview layer
        if canvas_state.preview_layer.is_none()
            || canvas_state.preview_layer.as_ref().unwrap().width() != canvas_state.width
            || canvas_state.preview_layer.as_ref().unwrap().height() != canvas_state.height
        {
            canvas_state.preview_layer =
                Some(TiledImage::new(canvas_state.width, canvas_state.height));
        } else if let Some(ref mut preview) = canvas_state.preview_layer {
            preview.clear();
        }

        let anti_aliased = self.fill_state.anti_aliased;
        let active_layer_index = canvas_state.active_layer_index;

        // Bbox already computed by flood_fill_fast
        let fill_bounds = bbox.map(|(fx0, fy0, fx1, fy1)| {
            let pad = if anti_aliased { 1u32 } else { 0 };
            egui::Rect::from_min_max(
                egui::pos2(
                    fx0.saturating_sub(pad) as f32,
                    fy0.saturating_sub(pad) as f32,
                ),
                egui::pos2(
                    (fx1 + 1 + pad).min(canvas_state.width) as f32,
                    (fy1 + 1 + pad).min(canvas_state.height) as f32,
                ),
            )
        });

        // Render preview: fill pixels on preview layer with optional AA
        if let Some(ref mut preview) = canvas_state.preview_layer
            && let Some(bbox) = bbox
        {
            self.render_fill_to_preview(
                preview,
                &fill_mask,
                bbox,
                fill_color_u8,
                canvas_state.selection_mask.as_ref(),
                anti_aliased,
                canvas_state.width,
                canvas_state.height,
                Some(&canvas_state.layers),
                active_layer_index,
            );
        }

        // Start stroke tracking for undo/redo
        if let Some(_layer) = canvas_state.layers.get(canvas_state.active_layer_index) {
            self.stroke_tracker
                .start_preview_tool(canvas_state.active_layer_index, "Fill");
            // Track the fill bounds so that finish() produces a StrokeEvent for undo
            if let Some(bounds) = fill_bounds {
                self.stroke_tracker.expand_bounds(bounds);
            }
        }

        canvas_state.preview_blend_mode = BlendMode::Normal;
        // Force blend-aware composite when fill colour has transparency,
        // so the preview accurately shows the semi-transparent fill on
        // top of the layer data instead of on top of everything.
        canvas_state.preview_force_composite = fill_color_u8.0[3] < 255;
        // Set stroke bounds to the fill area for efficient texture upload
        canvas_state.preview_stroke_bounds = fill_bounds;
        canvas_state.mark_preview_changed();
    }

    /// Update fill preview if tolerance, anti-alias, or color changed (debounced)
    fn update_fill_preview(&mut self, canvas_state: &mut CanvasState) {
        // Only recalculate if a change is pending
        if !self.fill_state.recalc_pending {
            return;
        }

        // Debounce: wait 50ms after last change before recalculating
        if let Some(changed_at) = self.fill_state.tolerance_changed_at
            && changed_at.elapsed().as_millis() < 50
        {
            return; // Too soon, wait for the user to stop adjusting
        }

        // Check if tolerance, anti-alias, or color actually changed
        let tol_changed =
            (self.fill_state.last_preview_tolerance - self.fill_state.tolerance).abs() >= 0.1;
        let aa_changed = self.fill_state.last_preview_aa != self.fill_state.anti_aliased;
        // Color change is already detected in handle_input and sets recalc_pending,
        // so if we reach here with recalc_pending=true and neither tol nor aa changed,
        // it must be a color change — always proceed.
        let color_only_change = !tol_changed && !aa_changed;
        if color_only_change {
            // Still proceed — color changed. Don't early-return.
        }

        if let Some((start_x, start_y, target_color, _original_mask)) =
            &self.fill_state.active_fill.clone()
        {
            // Only re-run flood fill if tolerance actually changed
            let (fill_mask, new_bbox) = if tol_changed {
                let active_layer = match canvas_state.layers.get(canvas_state.active_layer_index) {
                    Some(layer) => layer,
                    None => return,
                };
                let tolerance_threshold = (self.fill_state.tolerance / 100.0) * 255.0;
                let flat_rgba = active_layer.pixels.to_rgba_image();
                Self::flood_fill_fast(
                    flat_rgba.as_raw(),
                    *start_x,
                    *start_y,
                    target_color,
                    tolerance_threshold,
                    canvas_state.width,
                    canvas_state.height,
                )
            } else {
                // Reuse existing mask — compute bbox from it so we don't need to store it
                let mask = _original_mask.clone();
                let mut bbox: Option<(u32, u32, u32, u32)> = None;
                let w = canvas_state.width as usize;
                for (i, &b) in mask.iter().enumerate() {
                    if b != 0 {
                        let x = (i % w) as u32;
                        let y = (i / w) as u32;
                        bbox = Some(match bbox {
                            Some((x0, y0, x1, y1)) => (x0.min(x), y0.min(y), x1.max(x), y1.max(y)),
                            None => (x, y, x, y),
                        });
                    }
                }
                (mask, bbox)
            };

            // Update stored fill state
            self.fill_state.active_fill =
                Some((*start_x, *start_y, *target_color, fill_mask.clone()));
            self.fill_state.last_preview_tolerance = self.fill_state.tolerance;
            self.fill_state.last_preview_aa = self.fill_state.anti_aliased;
            self.fill_state.recalc_pending = false;
            self.fill_state.tolerance_changed_at = None;

            let anti_aliased = self.fill_state.anti_aliased;
            let active_layer_index = canvas_state.active_layer_index;

            let fill_bounds = new_bbox.map(|(fx0, fy0, fx1, fy1)| {
                let pad = if anti_aliased { 1u32 } else { 0 };
                egui::Rect::from_min_max(
                    egui::pos2(
                        fx0.saturating_sub(pad) as f32,
                        fy0.saturating_sub(pad) as f32,
                    ),
                    egui::pos2(
                        (fx1 + 1 + pad).min(canvas_state.width) as f32,
                        (fy1 + 1 + pad).min(canvas_state.height) as f32,
                    ),
                )
            });

            // Update stroke tracker bounds to cover the new fill area for correct undo
            if let Some(bounds) = fill_bounds {
                self.stroke_tracker.expand_bounds(bounds);
            }

            // Update preview_force_composite based on current fill color alpha
            if let Some(fill_color_u8) = self.fill_state.fill_color_u8 {
                canvas_state.preview_force_composite = fill_color_u8.0[3] < 255;
            }

            // Clear and redraw preview layer with new tolerance/color
            if let Some(ref mut preview) = canvas_state.preview_layer {
                preview.clear();

                if let Some(fill_color_u8) = self.fill_state.fill_color_u8
                    && let Some(bbox) = new_bbox
                {
                    self.render_fill_to_preview(
                        preview,
                        &fill_mask,
                        bbox,
                        fill_color_u8,
                        canvas_state.selection_mask.as_ref(),
                        anti_aliased,
                        canvas_state.width,
                        canvas_state.height,
                        Some(&canvas_state.layers),
                        active_layer_index,
                    );
                }
            }

            // Reset stroke bounds to the new fill area
            canvas_state.preview_stroke_bounds = fill_bounds;
            canvas_state.preview_texture_cache = None; // force full re-upload
            canvas_state.mark_preview_changed();
        }
    }

    /// Render filled pixels to preview layer, with optional anti-aliasing on edges
    fn render_fill_to_preview(
        &self,
        preview: &mut TiledImage,
        fill_mask: &[u8], // flat width*height bytes: 255=filled, 0=empty
        fill_bbox: (u32, u32, u32, u32), // (min_x, min_y, max_x, max_y)
        fill_color: Rgba<u8>,
        selection_mask: Option<&GrayImage>,
        anti_aliased: bool,
        width: u32,
        height: u32,
        canvas_state_layers: Option<&[crate::canvas::Layer]>,
        active_layer_index: usize,
    ) {
        let (bx0, by0, bx1, by1) = fill_bbox;
        let wu = width as usize;
        // Iterate only within the bounding box — avoids scanning entire canvas
        for y in by0..=by1.min(height.saturating_sub(1)) {
            for x in bx0..=bx1.min(width.saturating_sub(1)) {
                let idx = y as usize * wu + x as usize;
                if fill_mask[idx] == 0 {
                    continue;
                }

                // Check if pixel is in selection mask (if one exists)
                let should_fill = if let Some(mask) = selection_mask {
                    mask.get_pixel(x, y).0[0] > 0
                } else {
                    true
                };

                if !should_fill {
                    continue;
                }

                if anti_aliased {
                    // Count how many of 8 neighbors are also in the fill mask
                    let mut neighbor_fill_count = 0u8;
                    let mut total_neighbors = 0u8;
                    for dy in -1i32..=1 {
                        for dx in -1i32..=1 {
                            if dx == 0 && dy == 0 {
                                continue;
                            }
                            let nx = x as i32 + dx;
                            let ny = y as i32 + dy;
                            if nx >= 0 && nx < width as i32 && ny >= 0 && ny < height as i32 {
                                total_neighbors += 1;
                                if fill_mask[ny as usize * wu + nx as usize] != 0 {
                                    neighbor_fill_count += 1;
                                }
                            }
                        }
                    }

                    // If pixel is on the boundary (not all neighbors are filled),
                    // blend fill color with underlying pixel color for feathered edge
                    if total_neighbors > 0 && neighbor_fill_count < total_neighbors {
                        let ratio = neighbor_fill_count as f32 / total_neighbors as f32;
                        // Smoothstep curve for nicer edge falloff
                        let t = ratio * ratio * (3.0 - 2.0 * ratio);

                        // Get the underlying pixel color from the active layer
                        let bg_color = if let Some(layers) = canvas_state_layers {
                            if let Some(layer) = layers.get(active_layer_index) {
                                *layer.pixels.get_pixel(x, y)
                            } else {
                                Rgba([0, 0, 0, 0])
                            }
                        } else {
                            Rgba([0, 0, 0, 0])
                        };

                        // Blend fill color with background color based on edge factor
                        // t=1.0 means fully interior → full fill color
                        // t=0.0 means fully edge → mostly background color
                        let blend = |fc: u8, bc: u8, factor: f32| -> u8 {
                            (fc as f32 * factor + bc as f32 * (1.0 - factor)).round() as u8
                        };

                        let blended = Rgba([
                            blend(fill_color.0[0], bg_color.0[0], t),
                            blend(fill_color.0[1], bg_color.0[1], t),
                            blend(fill_color.0[2], bg_color.0[2], t),
                            // Alpha: scale fill alpha by edge factor so AA edges
                            // fade proportionally even for semi-transparent fills.
                            // Previously blended fill_alpha with bg_alpha, which
                            // made edges invisible when bg was fully transparent.
                            (fill_color.0[3] as f32 * t).round() as u8,
                        ]);

                        let pixel = preview.get_pixel_mut(x, y);
                        *pixel = blended;
                    } else {
                        let pixel = preview.get_pixel_mut(x, y);
                        *pixel = fill_color;
                    }
                } else {
                    let pixel = preview.get_pixel_mut(x, y);
                    *pixel = fill_color;
                }
            }
        }
    }

    /// Commit the fill preview to the actual layer
    fn commit_fill_preview(&mut self, canvas_state: &mut CanvasState) {
        if self.fill_state.active_fill.is_none() {
            return;
        }

        // Clear active fill state
        self.fill_state.active_fill = None;

        let blend_mode = self.properties.blending_mode;

        // IMPORTANT: Capture "before" snapshot BEFORE modifying the layer
        // For preview-based tools, the layer is still unmodified at this point
        let stroke_event = self.stroke_tracker.finish(canvas_state);

        // Commit preview to actual layer with proper blend mode
        if let Some(ref preview) = canvas_state.preview_layer {
            // Collect populated chunk data before mutating the layer
            let chunk_data: Vec<(u32, u32, image::RgbaImage)> = preview
                .chunk_keys()
                .filter_map(|(cx, cy)| {
                    preview
                        .get_chunk(cx, cy)
                        .map(|chunk| (cx, cy, chunk.clone()))
                })
                .collect();

            let chunk_size = crate::canvas::CHUNK_SIZE;

            if let Some(active_layer) = canvas_state.layers.get_mut(canvas_state.active_layer_index)
            {
                for (cx, cy, chunk) in &chunk_data {
                    let base_x = cx * chunk_size;
                    let base_y = cy * chunk_size;
                    let cw = chunk_size.min(canvas_state.width.saturating_sub(base_x));
                    let ch = chunk_size.min(canvas_state.height.saturating_sub(base_y));
                    for ly in 0..ch {
                        for lx in 0..cw {
                            let gx = base_x + lx;
                            let gy = base_y + ly;
                            let src = *chunk.get_pixel(lx, ly);
                            if src.0[3] == 0 {
                                continue;
                            }
                            let dst = active_layer.pixels.get_pixel_mut(gx, gy);
                            *dst = CanvasState::blend_pixel_static(*dst, src, blend_mode, 1.0);
                        }
                    }
                }
            }
        }

        // Mark dirty and clear preview
        canvas_state.clear_preview_state();
        canvas_state.mark_dirty(None);

        // Store event for history
        if stroke_event.is_some() {
            self.pending_stroke_event = stroke_event;
        }
    }

    /// Color picker - sample color from canvas
    fn pick_color_at_position(
        &mut self,
        canvas_state: &mut CanvasState,
        pos: (u32, u32),
        _use_secondary: bool,
    ) {
        // Bounds check
        if pos.0 >= canvas_state.width || pos.1 >= canvas_state.height {
            return;
        }

        // Get the color from the active layer
        let active_layer = match canvas_state.layers.get(canvas_state.active_layer_index) {
            Some(layer) => layer,
            None => return,
        };

        let pixel = active_layer.pixels.get_pixel(pos.0, pos.1);
        let color_32 =
            Color32::from_rgba_unmultiplied(pixel.0[0], pixel.0[1], pixel.0[2], pixel.0[3]);

        // Store the picked color
        self.last_picked_color = Some(color_32);

        // Update the appropriate color in ToolsPanel
        self.properties.color = color_32;

        // Note: The app.rs code will synchronize this to the ColorsPanel
        // We can't directly modify ColorsPanel from here due to borrow constraints
    }

    /// Fast flood fill using a DFS Vec-stack on a pre-extracted flat RGBA buffer.
    /// Returns (flat_mask, bbox) where flat_mask is width*height bytes (255=filled)
    /// and bbox is (min_x, min_y, max_x, max_y).  Returns None bbox when nothing filled.
    fn flood_fill_fast(
        flat_pixels: &[u8],
        start_x: u32,
        start_y: u32,
        target_color: &Rgba<u8>,
        tolerance: f32,
        canvas_w: u32,
        canvas_h: u32,
    ) -> (Vec<u8>, Option<(u32, u32, u32, u32)>) {
        let wu = canvas_w as usize;
        let hu = canvas_h as usize;
        // mask doubles as the visited array and the output
        let mut mask = vec![0u8; wu * hu];

        if start_x >= canvas_w || start_y >= canvas_h {
            return (mask, None);
        }

        let tc = target_color.0;
        let tol = tolerance;

        // Inline pixel fetch from flat RGBA buffer
        #[inline(always)]
        fn pix(flat: &[u8], idx: usize) -> [u8; 4] {
            let o = idx * 4;
            [flat[o], flat[o + 1], flat[o + 2], flat[o + 3]]
        }

        // Inline tight color match (same logic as colors_match but on raw arrays)
        #[inline(always)]
        fn matches(p: [u8; 4], tc: [u8; 4], tol: f32) -> bool {
            if tc[3] == 0 && p[3] == 0 {
                return true;
            }
            if tc[3] == 0 || p[3] == 0 {
                return (tc[3] as f32 - p[3] as f32).abs() <= tol;
            }
            let r = (tc[0] as f32 - p[0] as f32).abs();
            let g = (tc[1] as f32 - p[1] as f32).abs();
            let b = (tc[2] as f32 - p[2] as f32).abs();
            let a = (tc[3] as f32 - p[3] as f32).abs();
            r.max(g).max(b).max(a) <= tol
        }

        // Seed check
        let seed_idx = start_y as usize * wu + start_x as usize;
        if !matches(pix(flat_pixels, seed_idx), tc, tol) {
            return (mask, None);
        }

        // Bounding box
        let mut min_x = start_x;
        let mut min_y = start_y;
        let mut max_x = start_x;
        let mut max_y = start_y;

        // DFS stack stores packed flat indices to avoid (u32,u32) tuple overhead.
        // A flat index = y * canvas_w + x, max value < 4K*4K = 16M < u32::MAX.
        let mut stack: Vec<u32> = Vec::with_capacity(4096);
        mask[seed_idx] = 255;
        stack.push(seed_idx as u32);

        while let Some(idx) = stack.pop() {
            let x = (idx as usize % wu) as u32;
            let y = (idx as usize / wu) as u32;

            // Update bbox
            if x < min_x {
                min_x = x;
            }
            if x > max_x {
                max_x = x;
            }
            if y < min_y {
                min_y = y;
            }
            if y > max_y {
                max_y = y;
            }

            // Check 4 neighbors, push unvisited matching ones
            // Left
            if x > 0 {
                let ni = idx as usize - 1;
                if mask[ni] == 0 && matches(pix(flat_pixels, ni), tc, tol) {
                    mask[ni] = 255;
                    stack.push(ni as u32);
                }
            }
            // Right
            if x + 1 < canvas_w {
                let ni = idx as usize + 1;
                if mask[ni] == 0 && matches(pix(flat_pixels, ni), tc, tol) {
                    mask[ni] = 255;
                    stack.push(ni as u32);
                }
            }
            // Up
            if y > 0 {
                let ni = idx as usize - wu;
                if mask[ni] == 0 && matches(pix(flat_pixels, ni), tc, tol) {
                    mask[ni] = 255;
                    stack.push(ni as u32);
                }
            }
            // Down
            if y + 1 < canvas_h {
                let ni = idx as usize + wu;
                if mask[ni] == 0 && matches(pix(flat_pixels, ni), tc, tol) {
                    mask[ni] = 255;
                    stack.push(ni as u32);
                }
            }
        }

        let bbox = if max_x >= min_x {
            Some((min_x, min_y, max_x, max_y))
        } else {
            None
        };
        (mask, bbox)
    }

    /// Flood fill algorithm using queue-based traversal
    /// Returns a vector of (x, y) coordinates of all filled pixels
    fn flood_fill_selection(
        &self,
        pixels: &TiledImage,
        start_pos: (u32, u32),
        target_color: &Rgba<u8>,
        tolerance_threshold: f32,
        width: u32,
        height: u32,
    ) -> Vec<(u32, u32)> {
        let mut result = Vec::new();
        let w = width as usize;
        let mut visited = vec![false; w * height as usize];
        let mut queue = std::collections::VecDeque::new();

        queue.push_back(start_pos);
        visited[start_pos.1 as usize * w + start_pos.0 as usize] = true;

        while let Some((x, y)) = queue.pop_front() {
            result.push((x, y));

            // Check all 4 neighbors (up, down, left, right)
            let neighbors = [
                (x.saturating_sub(1), y),
                (x.saturating_add(1), y),
                (x, y.saturating_sub(1)),
                (x, y.saturating_add(1)),
            ];

            for (nx, ny) in neighbors.iter() {
                // Bounds check
                if *nx >= width || *ny >= height {
                    continue;
                }

                // Skip if already visited
                let vi = *ny as usize * w + *nx as usize;
                if visited[vi] {
                    continue;
                }

                visited[vi] = true;

                // Check color similarity
                let neighbor_color = pixels.get_pixel(*nx, *ny);
                if self.colors_match(target_color, neighbor_color, tolerance_threshold) {
                    queue.push_back((*nx, *ny));
                }
            }
        }

        result
    }

    /// Check if two colors match within tolerance
    /// Tolerance is the maximum allowed Euclidean distance in RGB space
    fn colors_match(&self, color1: &Rgba<u8>, color2: &Rgba<u8>, tolerance: f32) -> bool {
        // Both transparent → match (allows flood filling transparent areas)
        if color1.0[3] == 0 && color2.0[3] == 0 {
            return true;
        }
        // One transparent, one not → only match if tolerance covers alpha gap
        if color1.0[3] == 0 || color2.0[3] == 0 {
            let alpha_diff = (color1.0[3] as f32 - color2.0[3] as f32).abs();
            return alpha_diff <= tolerance;
        }

        // Convert to f32 and compute max component distance
        let r = (color1.0[0] as f32 - color2.0[0] as f32).abs();
        let g = (color1.0[1] as f32 - color2.0[1] as f32).abs();
        let b = (color1.0[2] as f32 - color2.0[2] as f32).abs();
        let a = (color1.0[3] as f32 - color2.0[3] as f32).abs();

        // Use maximum component distance including alpha
        let dist = r.max(g).max(b).max(a);

        dist <= tolerance
    }

    /// Global color select: find ALL pixels matching the target color across the
    /// entire canvas (not just connected/flood-fill). Respects tolerance.
    fn global_color_select(
        &self,
        pixels: &TiledImage,
        target_color: &Rgba<u8>,
        tolerance_threshold: f32,
        width: u32,
        height: u32,
    ) -> Vec<(u32, u32)> {
        let mut result = Vec::new();

        // Iterate every pixel in the canvas
        for y in 0..height {
            for x in 0..width {
                let pixel_color = pixels.get_pixel(x, y);
                if self.colors_match(target_color, pixel_color, tolerance_threshold) {
                    result.push((x, y));
                }
            }
        }

        result
    }

    // ================================================================
    // Lasso: scanline polygon rasterization into selection mask
    // ================================================================
    fn apply_lasso_selection(canvas_state: &mut CanvasState, points: &[Pos2], mode: SelectionMode) {
        let w = canvas_state.width;
        let h = canvas_state.height;

        // Build a fresh mask from polygon
        let mut lasso_mask = image::GrayImage::new(w, h);

        // Scanline fill: for each row, find intersection x-coords with polygon edges
        let n = points.len();
        for y in 0..h {
            let yf = y as f32 + 0.5; // centre of pixel row
            let mut nodes: Vec<f32> = Vec::new();
            // Walk polygon edges (including closing edge n-1 → 0)
            for i in 0..n {
                let j = (i + 1) % n;
                let yi = points[i].y;
                let yj = points[j].y;
                // Check if this edge crosses the scanline
                if (yi < yf && yj >= yf) || (yj < yf && yi >= yf) {
                    // x-intercept
                    let t = (yf - yi) / (yj - yi);
                    let x = points[i].x + t * (points[j].x - points[i].x);
                    nodes.push(x);
                }
            }
            nodes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            // Fill between pairs of intersections
            let mut k = 0;
            while k + 1 < nodes.len() {
                let x_start = (nodes[k].max(0.0) as u32).min(w);
                let x_end = ((nodes[k + 1] + 1.0).max(0.0) as u32).min(w);
                for x in x_start..x_end {
                    lasso_mask.put_pixel(x, y, image::Luma([255u8]));
                }
                k += 2;
            }
        }

        // Merge into the existing selection mask
        match mode {
            SelectionMode::Replace => {
                canvas_state.selection_mask = Some(lasso_mask);
            }
            SelectionMode::Add => {
                if let Some(ref mut existing) = canvas_state.selection_mask {
                    for y in 0..h {
                        for x in 0..w {
                            if lasso_mask.get_pixel(x, y).0[0] > 0 {
                                existing.put_pixel(x, y, image::Luma([255u8]));
                            }
                        }
                    }
                } else {
                    canvas_state.selection_mask = Some(lasso_mask);
                }
            }
            SelectionMode::Subtract => {
                if let Some(ref mut existing) = canvas_state.selection_mask {
                    for y in 0..h {
                        for x in 0..w {
                            if lasso_mask.get_pixel(x, y).0[0] > 0 {
                                existing.put_pixel(x, y, image::Luma([0u8]));
                            }
                        }
                    }
                }
                // If no existing mask, subtracting from nothing → no-op
            }
            SelectionMode::Intersect => {
                if let Some(ref existing) = canvas_state.selection_mask {
                    let mut result = image::GrayImage::new(w, h);
                    for y in 0..h {
                        for x in 0..w {
                            let new_val = lasso_mask.get_pixel(x, y).0[0];
                            let old_val = existing.get_pixel(x, y).0[0];
                            if new_val > 0 && old_val > 0 {
                                result.put_pixel(x, y, image::Luma([new_val.min(old_val)]));
                            }
                        }
                    }
                    canvas_state.selection_mask = Some(result);
                }
                // No existing mask → intersection with nothing = empty
            }
        }

        canvas_state.invalidate_selection_overlay();
    }

    // ================================================================
    // Perspective Crop: apply perspective transform + crop
    // ================================================================
    fn apply_perspective_crop(
        canvas_state: &mut CanvasState,
        corners: &[Pos2; 4],
    ) -> Option<Box<dyn crate::components::history::Command>> {
        // corners: [TL, TR, BR, BL] in canvas coords
        // Compute bounding box of the quad to determine output size
        let min_x = corners
            .iter()
            .map(|c| c.x)
            .fold(f32::INFINITY, f32::min)
            .max(0.0);
        let min_y = corners
            .iter()
            .map(|c| c.y)
            .fold(f32::INFINITY, f32::min)
            .max(0.0);
        let max_x = corners
            .iter()
            .map(|c| c.x)
            .fold(f32::NEG_INFINITY, f32::max)
            .min(canvas_state.width as f32);
        let max_y = corners
            .iter()
            .map(|c| c.y)
            .fold(f32::NEG_INFINITY, f32::max)
            .min(canvas_state.height as f32);

        let out_w = (max_x - min_x).round() as u32;
        let out_h = (max_y - min_y).round() as u32;
        if out_w < 2 || out_h < 2 {
            return None;
        }

        // Snapshot before crop for undo (multi-layer + dimension change = full snapshot)
        let snap_before = crate::components::history::SnapshotCommand::new(
            "Perspective Crop".to_string(),
            canvas_state,
        );

        let src_w = canvas_state.width;
        let src_h = canvas_state.height;

        // Apply perspective transform to each layer
        for layer in &mut canvas_state.layers {
            let src = layer.pixels.clone();
            let mut new_pixels = crate::canvas::TiledImage::new(out_w, out_h);

            for oy in 0..out_h {
                let v = (oy as f32 + 0.5) / out_h as f32;
                for ox in 0..out_w {
                    let u = (ox as f32 + 0.5) / out_w as f32;
                    // Bilinear interpolation of quad corners [TL, TR, BR, BL]
                    let src_x = (1.0 - u) * (1.0 - v) * corners[0].x
                        + u * (1.0 - v) * corners[1].x
                        + u * v * corners[2].x
                        + (1.0 - u) * v * corners[3].x;
                    let src_y = (1.0 - u) * (1.0 - v) * corners[0].y
                        + u * (1.0 - v) * corners[1].y
                        + u * v * corners[2].y
                        + (1.0 - u) * v * corners[3].y;

                    let pixel = Self::bilinear_sample(&src, src_x, src_y, src_w, src_h);
                    new_pixels.put_pixel(ox, oy, pixel);
                }
            }
            layer.pixels = new_pixels;
        }

        // Update canvas dimensions
        canvas_state.width = out_w;
        canvas_state.height = out_h;

        // Clear selection (doesn't make sense after a crop)
        canvas_state.clear_selection();
        canvas_state.mark_dirty(None);

        // Return undo command
        let mut snap = snap_before;
        snap.set_after(canvas_state);
        Some(Box::new(snap))
    }

    /// Bilinear sample from a TiledImage at floating point coordinates
    fn bilinear_sample(
        img: &crate::canvas::TiledImage,
        x: f32,
        y: f32,
        w: u32,
        h: u32,
    ) -> Rgba<u8> {
        let x0 = (x.floor() as i32).max(0).min(w as i32 - 1) as u32;
        let y0 = (y.floor() as i32).max(0).min(h as i32 - 1) as u32;
        let x1 = (x0 + 1).min(w - 1);
        let y1 = (y0 + 1).min(h - 1);
        let fx = x - x.floor();
        let fy = y - y.floor();

        let p00 = img.get_pixel(x0, y0);
        let p10 = img.get_pixel(x1, y0);
        let p01 = img.get_pixel(x0, y1);
        let p11 = img.get_pixel(x1, y1);

        let lerp = |a: u8, b: u8, t: f32| -> u8 {
            (a as f32 * (1.0 - t) + b as f32 * t)
                .round()
                .clamp(0.0, 255.0) as u8
        };

        let r = lerp(
            lerp(p00.0[0], p10.0[0], fx),
            lerp(p01.0[0], p11.0[0], fx),
            fy,
        );
        let g = lerp(
            lerp(p00.0[1], p10.0[1], fx),
            lerp(p01.0[1], p11.0[1], fx),
            fy,
        );
        let b_ch = lerp(
            lerp(p00.0[2], p10.0[2], fx),
            lerp(p01.0[2], p11.0[2], fx),
            fy,
        );
        let a = lerp(
            lerp(p00.0[3], p10.0[3], fx),
            lerp(p01.0[3], p11.0[3], fx),
            fy,
        );

        Rgba([r, g, b_ch, a])
    }

    // ========================================================================
    // GRADIENT TOOL — rasterizer, preview, commit, overlay, context bar
    // ========================================================================

    /// Rasterize the gradient into the canvas preview_layer.
    /// This is the hot path — called every frame during drag.
    /// Uses rayon parallel rows + pre-computed LUT for maximum throughput.
    /// When `dragging` is true, renders at reduced resolution for responsiveness,
    /// then renders full-res on release.
    fn render_gradient_to_preview(
        &mut self,
        canvas_state: &mut CanvasState,
        gpu_renderer: Option<&mut crate::gpu::GpuRenderer>,
    ) {
        let (start, end) = match (self.gradient_state.drag_start, self.gradient_state.drag_end) {
            (Some(s), Some(e)) => (s, e),
            _ => return,
        };

        // Ensure LUT is current
        if self.gradient_state.lut_dirty {
            self.gradient_state.rebuild_lut();
        }

        let w = canvas_state.width;
        let h = canvas_state.height;

        let shape = self.gradient_state.shape;
        let repeat = self.gradient_state.repeat;
        let mode = self.gradient_state.mode;
        let is_eraser = mode == GradientMode::Transparency;

        // Selection mask for clipping (applied as CPU post-pass)
        let sel_mask = canvas_state.selection_mask.as_ref();
        let has_selection = sel_mask.is_some();

        // Pre-determine fast/slow path so we know whether to downscale.
        let active_layer_normal_pre = canvas_state
            .layers
            .get(canvas_state.active_layer_index)
            .map(|l| l.blend_mode == BlendMode::Normal && l.opacity >= 1.0)
            .unwrap_or(false);
        let has_layers_above_pre = canvas_state
            .layers
            .iter()
            .skip(canvas_state.active_layer_index + 1)
            .any(|l| l.visible);
        let can_fast_path_pre = active_layer_normal_pre && !has_layers_above_pre && !is_eraser;

        // Downscale factor: only for slow path during interactive drag at >1080p.
        // The gradient preview is generated at reduced resolution and
        // composited via composite_partial_downscaled — ~4–16× fewer pixels.
        // On release / commit the gradient re-renders at full resolution.
        // Applied to BOTH fast and slow paths: the GPU readback + premultiply +
        // texture upload cost is significant even on the fast path at 4K.
        // Threshold ~1.5M pixels so 1080p (2M) gets scale 2 → ~960×540.
        let preview_scale: u32 = if self.gradient_state.dragging {
            let total_pixels = w as u64 * h as u64;
            if total_pixels > 1_500_000 {
                ((total_pixels as f64 / 1_000_000.0).sqrt().ceil() as u32).max(2)
            } else {
                1
            }
        } else {
            1
        };
        let gen_w = if preview_scale > 1 {
            w.div_ceil(preview_scale)
        } else {
            w
        };
        let gen_h = if preview_scale > 1 {
            h.div_ceil(preview_scale)
        } else {
            h
        };

        // ------------------------------------------------------------------
        // GPU PATH
        // ------------------------------------------------------------------
        let mut full_buf = if let Some(gpu) = gpu_renderer {
            let shape_u32 = match shape {
                GradientShape::Linear => 0u32,
                GradientShape::LinearReflected => 1,
                GradientShape::Radial => 2,
                GradientShape::Diamond => 3,
            };

            let params = crate::gpu::GradientGpuParams {
                start_x: start.x / preview_scale as f32,
                start_y: start.y / preview_scale as f32,
                end_x: end.x / preview_scale as f32,
                end_y: end.y / preview_scale as f32,
                width: gen_w,
                height: gen_h,
                shape: shape_u32,
                repeat: if repeat { 1 } else { 0 },
                is_eraser: if is_eraser { 1 } else { 0 },
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            };

            // Pass the raw LUT — the shader handles eraser-mode baking internally
            gpu.gradient_pipeline.generate_into(
                &gpu.ctx,
                &params,
                &self.gradient_state.lut,
                &mut self.gradient_state.gpu_readback_buf,
            );
            let mut buf = std::mem::take(&mut self.gradient_state.gpu_readback_buf);

            // Apply selection mask as CPU post-pass (parallelised with rayon)
            if has_selection && let Some(mask) = sel_mask {
                let mw = mask.width();
                let mh = mask.height();
                let row_bytes = gen_w as usize * 4;
                let ps = preview_scale;
                buf.par_chunks_mut(row_bytes)
                    .enumerate()
                    .for_each(|(y, row)| {
                        let canvas_y = (y as u32) * ps;
                        if canvas_y >= mh {
                            return;
                        }
                        for x in 0..gen_w as usize {
                            let canvas_x = (x as u32) * ps;
                            if canvas_x >= mw {
                                continue;
                            }
                            let sel_alpha = mask.get_pixel(canvas_x, canvas_y).0[0];
                            let off = x * 4;
                            if sel_alpha == 0 {
                                row[off] = 0;
                                row[off + 1] = 0;
                                row[off + 2] = 0;
                                row[off + 3] = 0;
                            } else if sel_alpha < 255 {
                                row[off + 3] =
                                    ((row[off + 3] as u16 * sel_alpha as u16) / 255) as u8;
                            }
                        }
                    });
            }

            buf
        } else {
            // ------------------------------------------------------------------
            // CPU FALLBACK PATH — with downscale during drag
            // ------------------------------------------------------------------
            let ax = start.x;
            let ay = start.y;
            let bx = end.x;
            let by = end.y;

            // Pre-bake the LUT for eraser mode on CPU
            let lut: Vec<u8> = if is_eraser {
                let src = &self.gradient_state.lut;
                let mut baked = vec![0u8; 256 * 4];
                for i in 0..256 {
                    let off = i * 4;
                    let lum = (0.299 * src[off] as f32
                        + 0.587 * src[off + 1] as f32
                        + 0.114 * src[off + 2] as f32) as u8;
                    let a = ((lum as u16 * src[off + 3] as u16) / 255) as u8;
                    baked[off] = 255;
                    baked[off + 1] = 255;
                    baked[off + 2] = 255;
                    baked[off + 3] = a;
                }
                baked
            } else {
                self.gradient_state.lut.clone()
            };

            // Pre-compute direction vectors
            let dx = bx - ax;
            let dy = by - ay;
            let len_sq = dx * dx + dy * dy;
            let len = len_sq.sqrt();
            let inv_len = if len > 1e-6 { 1.0 / len } else { 0.0 };
            let inv_len_sq = if len_sq > 1e-6 { 1.0 / len_sq } else { 0.0 };
            let ux = dx * inv_len;
            let uy = dy * inv_len;

            // Downscale during drag for responsiveness
            let scale: u32 = if self.gradient_state.dragging {
                let pixels = w as u64 * h as u64;
                if pixels > 4_000_000 {
                    4
                } else if pixels > 1_000_000 {
                    2
                } else {
                    1
                }
            } else {
                1
            };
            let rw = w.div_ceil(scale);
            let rh = h.div_ceil(scale);
            let scale_f = scale as f32;

            let row_stride = rw as usize * 4;
            let mut flat_buf = vec![0u8; row_stride * rh as usize];

            flat_buf
                .par_chunks_mut(row_stride)
                .enumerate()
                .for_each(|(y, row)| {
                    let py = y as f32 * scale_f + scale_f * 0.5;
                    let ry = py - ay;
                    let dot_y = ry * dy;
                    let dist_y_sq = ry * ry;
                    let proj_y_component = ry * uy;
                    let perp_y_component = ry * ux;
                    let gy = (y as u32) * scale;

                    for x in 0..rw as usize {
                        let px = x as f32 * scale_f + scale_f * 0.5;
                        let gx = (x as u32) * scale;

                        if has_selection
                            && let Some(mask) = sel_mask
                            && (gx as usize) < mask.width() as usize
                            && (gy as usize) < mask.height() as usize
                            && mask.get_pixel(gx, gy).0[0] == 0
                        {
                            continue;
                        }

                        let rx = px - ax;
                        let t = match shape {
                            GradientShape::Linear => {
                                let raw = (rx * dx + dot_y) * inv_len_sq;
                                if repeat {
                                    raw.rem_euclid(1.0)
                                } else {
                                    raw.clamp(0.0, 1.0)
                                }
                            }
                            GradientShape::LinearReflected => {
                                let raw = (rx * dx + dot_y) * inv_len_sq;
                                if repeat {
                                    let t_mod = raw.rem_euclid(2.0);
                                    if t_mod > 1.0 { 2.0 - t_mod } else { t_mod }
                                } else {
                                    1.0 - (2.0 * raw.clamp(0.0, 1.0) - 1.0).abs()
                                }
                            }
                            GradientShape::Radial => {
                                let dist_sq = rx * rx + dist_y_sq;
                                let dist = dist_sq.sqrt() * inv_len;
                                if repeat {
                                    dist.rem_euclid(1.0)
                                } else {
                                    dist.clamp(0.0, 1.0)
                                }
                            }
                            GradientShape::Diamond => {
                                let proj = (rx * ux + proj_y_component).abs() * inv_len;
                                let perp = (rx * (-uy) + perp_y_component).abs() * inv_len;
                                let dist = proj + perp;
                                if repeat {
                                    dist.rem_euclid(1.0)
                                } else {
                                    dist.clamp(0.0, 1.0)
                                }
                            }
                        };

                        let idx = (t * 255.0) as usize;
                        let loff = idx * 4;
                        let mut a = lut[loff + 3];

                        if has_selection
                            && let Some(mask) = sel_mask
                            && (gx as usize) < mask.width() as usize
                            && (gy as usize) < mask.height() as usize
                        {
                            let sel_alpha = mask.get_pixel(gx, gy).0[0];
                            if sel_alpha < 255 {
                                a = ((a as u16 * sel_alpha as u16) / 255) as u8;
                            }
                        }

                        if a > 0 {
                            let off = x * 4;
                            row[off] = lut[loff];
                            row[off + 1] = lut[loff + 1];
                            row[off + 2] = lut[loff + 2];
                            row[off + 3] = a;
                        }
                    }
                });

            // Upscale if needed — but NOT when preview_scale > 1 (slow path
            // will use composite_partial_downscaled at reduced resolution)
            if scale > 1 && preview_scale == 1 {
                let mut full = vec![0u8; w as usize * 4 * h as usize];
                let full_stride = w as usize * 4;
                for fy in 0..h as usize {
                    let sy = fy / scale as usize;
                    let src_row = &flat_buf[sy * row_stride..(sy + 1) * row_stride];
                    let dst_row = &mut full[fy * full_stride..(fy + 1) * full_stride];
                    for fx in 0..w as usize {
                        let sx = fx / scale as usize;
                        let s_off = sx * 4;
                        let d_off = fx * 4;
                        dst_row[d_off] = src_row[s_off];
                        dst_row[d_off + 1] = src_row[s_off + 1];
                        dst_row[d_off + 2] = src_row[s_off + 2];
                        dst_row[d_off + 3] = src_row[s_off + 3];
                    }
                }
                full
            } else {
                flat_buf
            }
        };

        // Determine whether we can use the fast overlay path (no full-stack
        // composite needed).  Already pre-computed above as can_fast_path_pre.
        let can_fast_path = can_fast_path_pre;

        if can_fast_path {
            // ── Fast path: skip TiledImage entirely ──────────────────
            // Pre-multiply alpha and store directly in preview_flat_buffer.
            // The display code will detect preview_flat_ready and use this
            // buffer directly, skipping extract_region + premultiply.
            let buf_len = full_buf.len();
            canvas_state.preview_flat_buffer.resize(buf_len, 0);
            // Premultiply in parallel (rayon) into the target buffer
            canvas_state
                .preview_flat_buffer
                .par_chunks_exact_mut(4)
                .zip(full_buf.par_chunks_exact(4))
                .for_each(|(dst, src)| {
                    let a = src[3] as u16;
                    dst[0] = ((src[0] as u16 * a + 128) / 255) as u8;
                    dst[1] = ((src[1] as u16 * a + 128) / 255) as u8;
                    dst[2] = ((src[2] as u16 * a + 128) / 255) as u8;
                    dst[3] = src[3];
                });
            canvas_state.preview_flat_ready = true;
            // Populate preview_layer with full-res straight-alpha data so that
            // commit_gradient (which reads preview_layer) works correctly.
            // Display still uses preview_flat_buffer (premultiplied fast path).
            canvas_state.preview_layer = Some(TiledImage::from_raw_rgba(w, h, &full_buf));
            canvas_state.preview_force_composite = false;
            canvas_state.preview_downscale = preview_scale;
        } else {
            // ── Slow path: need TiledImage for composite_partial ─────
            // When preview_scale > 1 the buffer is at reduced resolution;
            // composite_partial_downscaled will sample layers at matching
            // stride so the output is a smaller texture that egui stretches.
            let pw = if preview_scale > 1 { gen_w } else { w };
            let ph = if preview_scale > 1 { gen_h } else { h };
            let preview = TiledImage::from_raw_rgba(pw, ph, &full_buf);
            canvas_state.preview_layer = Some(preview);
            canvas_state.preview_force_composite = true;
            canvas_state.preview_flat_ready = false;
            canvas_state.preview_downscale = preview_scale;
        }

        // Configure preview compositing
        canvas_state.preview_blend_mode = BlendMode::Normal;
        canvas_state.preview_is_eraser = is_eraser;
        canvas_state.preview_stroke_bounds = Some(egui::Rect::from_min_max(
            egui::pos2(0.0, 0.0),
            egui::pos2(w as f32, h as f32),
        ));
        canvas_state.mark_preview_changed();
        self.gradient_state.preview_dirty = false;

        // Return the full_buf to the reusable readback buffer (keeps capacity)
        full_buf.clear();
        self.gradient_state.gpu_readback_buf = full_buf;
    }

    /// Called every frame (even when handle_input is blocked by UI) to
    /// re-render the gradient preview when toolbar settings change.
    pub fn update_gradient_if_dirty(
        &mut self,
        canvas_state: &mut CanvasState,
        gpu_renderer: Option<&mut crate::gpu::GpuRenderer>,
    ) {
        if self.gradient_state.preview_dirty
            && self.gradient_state.drag_start.is_some()
            && !self.gradient_state.dragging
        {
            self.render_gradient_to_preview(canvas_state, gpu_renderer);
        }
        self.gradient_state.preview_dirty = false;
    }

    /// Re-render text preview if context bar changed properties (color, alignment, etc.)
    /// Called outside the allow_input gate so it works when interacting with UI.
    pub fn update_text_if_dirty(
        &mut self,
        canvas_state: &mut CanvasState,
        primary_color_f32: [f32; 4],
    ) {
        if !self.text_state.is_editing || self.text_state.text.is_empty() {
            return;
        }
        // Detect color change
        let current_color = [
            (primary_color_f32[0] * 255.0) as u8,
            (primary_color_f32[1] * 255.0) as u8,
            (primary_color_f32[2] * 255.0) as u8,
            (primary_color_f32[3] * 255.0) as u8,
        ];
        if self.text_state.last_color != current_color {
            self.text_state.last_color = current_color;
            self.text_state.preview_dirty = true;
        }
        if self.text_state.preview_dirty {
            self.render_text_preview(canvas_state, primary_color_f32);
        }
    }

    /// Re-render shape preview if context bar / color widget changed properties.
    /// Called outside the allow_input gate so it works when interacting with UI.
    pub fn update_shape_if_dirty(
        &mut self,
        canvas_state: &mut CanvasState,
        primary_color_f32: [f32; 4],
        secondary_color_f32: [f32; 4],
    ) {
        if self.shapes_state.placed.is_none() {
            return;
        }
        let p = self.shapes_state.placed.as_mut().unwrap();
        let current_primary = [
            (primary_color_f32[0] * 255.0) as u8,
            (primary_color_f32[1] * 255.0) as u8,
            (primary_color_f32[2] * 255.0) as u8,
            (primary_color_f32[3] * 255.0) as u8,
        ];
        let current_secondary = [
            (secondary_color_f32[0] * 255.0) as u8,
            (secondary_color_f32[1] * 255.0) as u8,
            (secondary_color_f32[2] * 255.0) as u8,
            (secondary_color_f32[3] * 255.0) as u8,
        ];
        let changed = p.kind != self.shapes_state.selected_shape
            || p.fill_mode != self.shapes_state.fill_mode
            || p.outline_width != self.properties.size
            || p.anti_alias != self.shapes_state.anti_alias
            || p.corner_radius != self.shapes_state.corner_radius
            || p.primary_color != current_primary
            || p.secondary_color != current_secondary;
        if changed {
            p.kind = self.shapes_state.selected_shape;
            p.fill_mode = self.shapes_state.fill_mode;
            p.outline_width = self.properties.size;
            p.anti_alias = self.shapes_state.anti_alias;
            p.corner_radius = self.shapes_state.corner_radius;
            p.primary_color = current_primary;
            p.secondary_color = current_secondary;
            self.render_shape_preview(canvas_state, primary_color_f32, secondary_color_f32);
        }
    }

    /// Commit the gradient preview to the active layer.
    fn commit_gradient(&mut self, canvas_state: &mut CanvasState) {
        if self.gradient_state.drag_start.is_none() {
            return;
        }

        // Capture "before" snapshot BEFORE modifying the layer
        let stroke_event = self.stroke_tracker.finish(canvas_state);

        let is_eraser = self.gradient_state.mode == GradientMode::Transparency;

        // Commit preview to actual layer with proper alpha blending
        if let Some(ref preview) = canvas_state.preview_layer {
            let chunk_data: Vec<(u32, u32, image::RgbaImage)> = preview
                .chunk_keys()
                .filter_map(|(cx, cy)| {
                    preview
                        .get_chunk(cx, cy)
                        .map(|chunk| (cx, cy, chunk.clone()))
                })
                .collect();

            let chunk_size = CHUNK_SIZE;

            if let Some(active_layer) = canvas_state.layers.get_mut(canvas_state.active_layer_index)
            {
                for (cx, cy, chunk) in &chunk_data {
                    let base_x = cx * chunk_size;
                    let base_y = cy * chunk_size;
                    let cw = chunk_size.min(canvas_state.width.saturating_sub(base_x));
                    let ch = chunk_size.min(canvas_state.height.saturating_sub(base_y));
                    for ly in 0..ch {
                        for lx in 0..cw {
                            let gx = base_x + lx;
                            let gy = base_y + ly;
                            let src = chunk.get_pixel(lx, ly);
                            if src.0[3] == 0 {
                                continue;
                            }
                            let dst = active_layer.pixels.get_pixel_mut(gx, gy);

                            if is_eraser {
                                // Eraser: reduce layer alpha by mask strength
                                let mask_strength = src.0[3] as f32 / 255.0;
                                let current_a = dst.0[3] as f32 / 255.0;
                                let new_a = (current_a * (1.0 - mask_strength)).max(0.0);
                                dst.0[3] = (new_a * 255.0).round() as u8;
                            } else {
                                // Normal alpha-over blend
                                let sa = src.0[3] as f32 / 255.0;
                                let da = dst.0[3] as f32 / 255.0;
                                let out_a = sa + da * (1.0 - sa);
                                if out_a > 0.0 {
                                    let r = (src.0[0] as f32 * sa
                                        + dst.0[0] as f32 * da * (1.0 - sa))
                                        / out_a;
                                    let g = (src.0[1] as f32 * sa
                                        + dst.0[1] as f32 * da * (1.0 - sa))
                                        / out_a;
                                    let b = (src.0[2] as f32 * sa
                                        + dst.0[2] as f32 * da * (1.0 - sa))
                                        / out_a;
                                    *dst = Rgba([
                                        r.round() as u8,
                                        g.round() as u8,
                                        b.round() as u8,
                                        (out_a * 255.0).round() as u8,
                                    ]);
                                }
                            }
                        }
                    }
                }
                active_layer.invalidate_lod();
                active_layer.gpu_generation += 1;
            }
        }

        // Clear gradient state
        self.gradient_state.drag_start = None;
        self.gradient_state.drag_end = None;
        self.gradient_state.dragging = false;
        self.gradient_state.dragging_handle = None;

        // Mark dirty and clear preview
        canvas_state.clear_preview_state();
        canvas_state.mark_dirty(None);

        // Store event for history
        if stroke_event.is_some() {
            self.pending_stroke_event = stroke_event;
        }
    }

    /// Cancel gradient operation without committing.
    fn cancel_gradient(&mut self, canvas_state: &mut CanvasState) {
        self.gradient_state.drag_start = None;
        self.gradient_state.drag_end = None;
        self.gradient_state.dragging = false;
        self.gradient_state.dragging_handle = None;
        self.stroke_tracker.cancel();
        canvas_state.clear_preview_state();
        canvas_state.mark_dirty(None);
    }

    /// Draw gradient handle overlay (start/end points and connecting line).
    fn draw_gradient_overlay(
        &self,
        painter: &egui::Painter,
        canvas_rect: Rect,
        zoom: f32,
        canvas_state: &CanvasState,
    ) {
        let (start, end) = match (self.gradient_state.drag_start, self.gradient_state.drag_end) {
            (Some(s), Some(e)) => (s, e),
            _ => return,
        };

        // Convert canvas coords to screen coords
        let to_screen = |cx: f32, cy: f32| -> Pos2 {
            let image_rect = egui::Rect::from_min_size(
                canvas_rect.min,
                egui::vec2(
                    canvas_state.width as f32 * zoom,
                    canvas_state.height as f32 * zoom,
                ),
            );
            Pos2::new(image_rect.min.x + cx * zoom, image_rect.min.y + cy * zoom)
        };

        let screen_start = to_screen(start.x, start.y);
        let screen_end = to_screen(end.x, end.y);

        // Dashed line connecting start ↔ end
        let dash_len = 6.0;
        let gap_len = 4.0;
        let line_vec = screen_end - screen_start;
        let line_len = line_vec.length();
        if line_len > 1.0 {
            let dir = line_vec / line_len;
            let mut d = 0.0;
            while d < line_len {
                let seg_start = screen_start + dir * d;
                let seg_end = screen_start + dir * (d + dash_len).min(line_len);
                painter.line_segment([seg_start, seg_end], egui::Stroke::new(1.5, Color32::WHITE));
                painter.line_segment([seg_start, seg_end], egui::Stroke::new(0.5, Color32::BLACK));
                d += dash_len + gap_len;
            }
        }

        // Start handle — filled circle with outline
        let handle_r = 5.0;
        painter.circle_filled(screen_start, handle_r + 1.0, Color32::BLACK);
        painter.circle_filled(screen_start, handle_r, Color32::WHITE);
        painter.circle_filled(
            screen_start,
            handle_r - 2.0,
            Color32::from_rgb(100, 180, 255),
        );

        // End handle — filled circle with outline
        painter.circle_filled(screen_end, handle_r + 1.0, Color32::BLACK);
        painter.circle_filled(screen_end, handle_r, Color32::WHITE);
        painter.circle_filled(screen_end, handle_r - 2.0, Color32::from_rgb(255, 100, 100));
    }

    /// Context bar: gradient shape, mode, preset, repeat, and gradient strip.
    fn show_gradient_options(
        &mut self,
        ui: &mut egui::Ui,
        assets: &Assets,
        primary_color: Color32,
        secondary_color: Color32,
    ) {
        // Cache primary color for the "Use Primary" button in the gradient bar
        self.gradient_state.cached_primary = Some([
            primary_color.r(),
            primary_color.g(),
            primary_color.b(),
            primary_color.a(),
        ]);

        // Shape dropdown
        ui.label(t!("ctx.shapes.shape"));
        let current_shape = self.gradient_state.shape;
        egui::ComboBox::from_id_source("gradient_shape")
            .selected_text(current_shape.label())
            .width(120.0)
            .show_ui(ui, |ui| {
                for &shape in GradientShape::all() {
                    if ui
                        .selectable_label(shape == current_shape, shape.label())
                        .clicked()
                    {
                        self.gradient_state.shape = shape;
                        self.gradient_state.preview_dirty = true;
                    }
                }
            });

        ui.separator();

        // Mode toggle
        ui.label(t!("ctx.mode"));
        let mode = self.gradient_state.mode;
        if ui
            .selectable_label(mode == GradientMode::Color, GradientMode::Color.label())
            .clicked()
        {
            self.gradient_state.mode = GradientMode::Color;
            self.gradient_state.lut_dirty = true;
            self.gradient_state.preview_dirty = true;
        }
        if ui
            .selectable_label(
                mode == GradientMode::Transparency,
                GradientMode::Transparency.label(),
            )
            .clicked()
        {
            self.gradient_state.mode = GradientMode::Transparency;
            self.gradient_state.lut_dirty = true;
            self.gradient_state.preview_dirty = true;
        }

        ui.separator();

        // Preset dropdown
        ui.label(t!("ctx.gradient.preset"));
        let current_preset = self.gradient_state.preset;
        let primary_u8 = [
            primary_color.r(),
            primary_color.g(),
            primary_color.b(),
            primary_color.a(),
        ];
        let secondary_u8 = [
            secondary_color.r(),
            secondary_color.g(),
            secondary_color.b(),
            secondary_color.a(),
        ];
        egui::ComboBox::from_id_source("gradient_preset")
            .selected_text(current_preset.label())
            .width(150.0)
            .show_ui(ui, |ui| {
                for &preset in GradientPreset::all() {
                    if ui
                        .selectable_label(preset == current_preset, preset.label())
                        .clicked()
                    {
                        self.gradient_state
                            .apply_preset(preset, primary_u8, secondary_u8);
                        self.gradient_state.preview_dirty = true;
                    }
                }
            });

        ui.separator();

        // Repeat checkbox
        let mut repeat = self.gradient_state.repeat;
        if ui
            .checkbox(&mut repeat, t!("ctx.gradient.repeat"))
            .changed()
        {
            self.gradient_state.repeat = repeat;
            self.gradient_state.preview_dirty = true;
        }

        ui.separator();

        // Gradient strip preview + stop editor
        self.show_gradient_bar(ui, assets);
    }

    /// Draw the interactive gradient bar with draggable color stops.
    fn show_gradient_bar(&mut self, ui: &mut egui::Ui, assets: &Assets) {
        let bar_width = 200.0f32;
        let bar_height = 18.0f32;
        let stop_radius = 5.0f32;
        let h_pad = stop_radius + 2.0; // horizontal padding so edge handles aren't clipped

        let (response, painter) = ui.allocate_painter(
            egui::vec2(
                bar_width + h_pad * 2.0,
                bar_height + stop_radius * 2.0 + 4.0,
            ),
            egui::Sense::click_and_drag(),
        );
        let bar_rect = egui::Rect::from_min_size(
            response.rect.min + egui::vec2(h_pad, stop_radius + 2.0),
            egui::vec2(bar_width, bar_height),
        );

        // Draw checkerboard behind bar (for transparency visualization)
        let check_size = 4.0;
        let cols = (bar_width / check_size).ceil() as usize;
        let rows = (bar_height / check_size).ceil() as usize;
        for row in 0..rows {
            for col in 0..cols {
                let color = if (row + col) % 2 == 0 {
                    Color32::from_gray(200)
                } else {
                    Color32::from_gray(255)
                };
                let r = egui::Rect::from_min_size(
                    bar_rect.min + egui::vec2(col as f32 * check_size, row as f32 * check_size),
                    egui::vec2(check_size, check_size),
                )
                .intersect(bar_rect);
                painter.rect_filled(r, 0.0, color);
            }
        }

        // Rebuild LUT if needed for display
        if self.gradient_state.lut_dirty {
            self.gradient_state.rebuild_lut();
        }

        // Draw gradient bar using LUT
        let lut = &self.gradient_state.lut;
        for x in 0..bar_width as usize {
            let t = x as f32 / (bar_width - 1.0);
            let idx = (t * 255.0).round() as usize;
            let off = idx * 4;
            let color =
                Color32::from_rgba_unmultiplied(lut[off], lut[off + 1], lut[off + 2], lut[off + 3]);
            let line_rect = egui::Rect::from_min_size(
                bar_rect.min + egui::vec2(x as f32, 0.0),
                egui::vec2(1.0, bar_height),
            );
            painter.rect_filled(line_rect, 0.0, color);
        }

        // Outline
        painter.rect_stroke(bar_rect, 1.0, egui::Stroke::new(1.0, Color32::DARK_GRAY));

        // Draw stop handles
        let stop_y = bar_rect.max.y + stop_radius + 1.0;
        for (i, stop) in self.gradient_state.stops.iter().enumerate() {
            let stop_x = bar_rect.min.x + stop.position * bar_width;
            let _pos = Pos2::new(stop_x, stop_y);
            let is_selected = self.gradient_state.selected_stop == Some(i);

            // Triangle pointing up at the bar
            let tri_top = Pos2::new(stop_x, bar_rect.max.y + 1.0);
            let tri_left = Pos2::new(stop_x - stop_radius, stop_y);
            let tri_right = Pos2::new(stop_x + stop_radius, stop_y);

            // Outline
            let outline_color = if is_selected {
                Color32::WHITE
            } else {
                Color32::DARK_GRAY
            };
            painter.add(egui::Shape::convex_polygon(
                vec![tri_top, tri_right, tri_left],
                Color32::from_rgba_unmultiplied(
                    stop.color[0],
                    stop.color[1],
                    stop.color[2],
                    stop.color[3],
                ),
                egui::Stroke::new(if is_selected { 2.0 } else { 1.0 }, outline_color),
            ));
        }

        // Handle interactions
        let pointer_pos = response.interact_pointer_pos();
        if let Some(pp) = pointer_pos {
            let t_at_pointer = ((pp.x - bar_rect.min.x) / bar_width).clamp(0.0, 1.0);

            if response.drag_started() {
                // Check if clicking near an existing stop
                let mut hit_stop: Option<usize> = None;
                for (i, stop) in self.gradient_state.stops.iter().enumerate() {
                    let stop_screen_x = bar_rect.min.x + stop.position * bar_width;
                    if (pp.x - stop_screen_x).abs() < stop_radius * 2.0 {
                        hit_stop = Some(i);
                        break;
                    }
                }
                if let Some(idx) = hit_stop {
                    self.gradient_state.selected_stop = Some(idx);
                } else {
                    // Add new stop at click position
                    let new_color = self.gradient_state.sample_lut(t_at_pointer);
                    self.gradient_state
                        .stops
                        .push(GradientStop::new(t_at_pointer, new_color));
                    self.gradient_state
                        .stops
                        .sort_by(|a, b| a.position.partial_cmp(&b.position).unwrap());
                    // Find the newly added stop index
                    for (i, s) in self.gradient_state.stops.iter().enumerate() {
                        if (s.position - t_at_pointer).abs() < 0.001 {
                            self.gradient_state.selected_stop = Some(i);
                            break;
                        }
                    }
                    self.gradient_state.preset = GradientPreset::Custom;
                    self.gradient_state.lut_dirty = true;
                    self.gradient_state.preview_dirty = true;
                }
            }

            // Drag selected stop
            if response.dragged()
                && let Some(sel) = self.gradient_state.selected_stop
                && sel < self.gradient_state.stops.len()
            {
                // Don't allow dragging first/last stops past each other
                let new_pos = t_at_pointer.clamp(0.0, 1.0);
                self.gradient_state.stops[sel].position = new_pos;
                // Re-sort and update selected index
                let sel_stop_pos = self.gradient_state.stops[sel].position;
                let sel_stop_color = self.gradient_state.stops[sel].color;
                self.gradient_state
                    .stops
                    .sort_by(|a, b| a.position.partial_cmp(&b.position).unwrap());
                for (i, s) in self.gradient_state.stops.iter().enumerate() {
                    if (s.position - sel_stop_pos).abs() < 1e-6 && s.color == sel_stop_color {
                        self.gradient_state.selected_stop = Some(i);
                        break;
                    }
                }
                self.gradient_state.preset = GradientPreset::Custom;
                self.gradient_state.lut_dirty = true;
                self.gradient_state.preview_dirty = true;
            }
        }

        // Right-click to delete a stop (if more than 2)
        if response.secondary_clicked()
            && let Some(pp) = ui.input(|i| i.pointer.latest_pos())
            && self.gradient_state.stops.len() > 2
        {
            let mut closest_idx: Option<usize> = None;
            let mut closest_dist = f32::MAX;
            for (i, stop) in self.gradient_state.stops.iter().enumerate() {
                let stop_screen_x = bar_rect.min.x + stop.position * bar_width;
                let d = (pp.x - stop_screen_x).abs();
                if d < stop_radius * 3.0 && d < closest_dist {
                    closest_dist = d;
                    closest_idx = Some(i);
                }
            }
            if let Some(idx) = closest_idx {
                self.gradient_state.stops.remove(idx);
                self.gradient_state.selected_stop = None;
                self.gradient_state.preset = GradientPreset::Custom;
                self.gradient_state.lut_dirty = true;
                self.gradient_state.preview_dirty = true;
            }
        }

        // Color edit for selected stop — compact swatch + "Use Primary" button
        if let Some(sel) = self.gradient_state.selected_stop
            && sel < self.gradient_state.stops.len()
        {
            ui.separator();

            let stop_color = self.gradient_state.stops[sel].color;
            let preview_color = Color32::from_rgba_unmultiplied(
                stop_color[0],
                stop_color[1],
                stop_color[2],
                stop_color[3],
            );

            ui.horizontal(|ui| {
                // Color swatch (small)
                let (swatch_rect, _) =
                    ui.allocate_exact_size(egui::vec2(16.0, 16.0), egui::Sense::hover());
                let p = ui.painter_at(swatch_rect);
                let cs = 4.0;
                for row in 0..4 {
                    for col in 0..4 {
                        let c = if (row + col) % 2 == 0 {
                            Color32::from_gray(200)
                        } else {
                            Color32::WHITE
                        };
                        let r = egui::Rect::from_min_size(
                            swatch_rect.min + egui::vec2(col as f32 * cs, row as f32 * cs),
                            egui::vec2(cs, cs),
                        )
                        .intersect(swatch_rect);
                        p.rect_filled(r, 0.0, c);
                    }
                }
                p.rect_filled(swatch_rect, 2.0, preview_color);
                p.rect_stroke(swatch_rect, 2.0, egui::Stroke::new(1.0, Color32::DARK_GRAY));

                // Hex label
                ui.label(format!(
                    "#{:02X}{:02X}{:02X}{:02X}",
                    stop_color[0], stop_color[1], stop_color[2], stop_color[3],
                ));

                // "Use Primary" button — applies the current primary color to this stop
                if assets
                    .menu_item(ui, Icon::ApplyPrimary, &t!("ctx.gradient.apply_primary"))
                    .clicked()
                    && let Some(pc) = self.gradient_state.cached_primary
                {
                    self.gradient_state.stops[sel].color = pc;
                    let c32 = Color32::from_rgba_unmultiplied(pc[0], pc[1], pc[2], pc[3]);
                    self.gradient_state.stops[sel].hsv = super::colors::color_to_hsv(c32);
                    self.gradient_state.preset = GradientPreset::Custom;
                    self.gradient_state.lut_dirty = true;
                    self.gradient_state.preview_dirty = true;
                }
            });

            // Hint for the user
            ui.label(
                egui::RichText::new(t!("ctx.gradient.color_hint"))
                    .weak()
                    .small(),
            );
        }
    }

    // ====================================================================
    // TEXT TOOL — context bar, preview, commit
    // ====================================================================

    fn show_text_options(&mut self, ui: &mut egui::Ui) {
        // Async font loading: kick off background thread on first access
        if self.text_state.available_fonts.is_empty() && self.text_state.fonts_loading_rx.is_none()
        {
            let (tx, rx) = std::sync::mpsc::channel();
            self.text_state.fonts_loading_rx = Some(rx);
            std::thread::spawn(move || {
                let fonts = crate::ops::text::enumerate_system_fonts();
                let _ = tx.send(fonts);
            });
        }

        // Check if async font list is ready
        if let Some(ref rx) = self.text_state.fonts_loading_rx
            && let Ok(fonts) = rx.try_recv()
        {
            self.text_state.available_fonts = fonts;
            self.text_state.fonts_loading_rx = None;
            // Refresh weights for current family
            self.text_state.available_weights =
                crate::ops::text::enumerate_font_weights(&self.text_state.font_family);
        }

        // Poll async font preview results
        if let Some(ref rx) = self.text_state.font_preview_rx {
            while let Ok(batch) = rx.try_recv() {
                for (name, font) in batch {
                    self.text_state.font_preview_pending.remove(&name);
                    self.text_state.font_preview_cache.insert(name, font);
                }
            }
        }

        ui.label(t!("ctx.text.font"));
        let family_label = self.text_state.font_family.clone();
        let popup_id = ui.make_persistent_id("ctx_text_font_popup");
        let is_open = ui.memory(|mem| mem.is_popup_open(popup_id));

        if self.text_state.available_fonts.is_empty() {
            // Still loading
            ui.add(
                egui::Button::new(t!("ctx.text.loading_fonts")).min_size(egui::vec2(140.0, 0.0)),
            );
            ui.ctx()
                .request_repaint_after(std::time::Duration::from_millis(100));
        } else {
            let button_response = ui.add(
                egui::Button::new(egui::RichText::new(if family_label.len() > 20 {
                    &family_label[..20]
                } else {
                    &family_label
                }))
                .min_size(egui::vec2(140.0, 0.0)),
            );
            if button_response.clicked() {
                ui.memory_mut(|mem| mem.toggle_popup(popup_id));
            }
            if is_open {
                let below = button_response.rect;
                egui::Area::new(popup_id)
                    .order(egui::Order::Foreground)
                    .fixed_pos(egui::pos2(below.min.x, below.max.y))
                    .show(ui.ctx(), |ui| {
                        egui::Frame::popup(ui.style()).show(ui, |ui| {
                            ui.set_min_width(250.0);
                            let te_resp = ui.add(
                                egui::TextEdit::singleline(&mut self.text_state.font_search)
                                    .hint_text(t!("ctx.text.search"))
                                    .desired_width(230.0),
                            );
                            if !te_resp.has_focus() && self.text_state.font_search.is_empty() {
                                te_resp.request_focus();
                            }
                            let search = self.text_state.font_search.to_lowercase();
                            let fonts: Vec<&String> = self
                                .text_state
                                .available_fonts
                                .iter()
                                .filter(|f| search.is_empty() || f.to_lowercase().contains(&search))
                                .collect();

                            // Collect fonts needing preview loading from visible area
                            let mut to_load: Vec<String> = Vec::new();

                            let row_height = 22.0;
                            let total_rows = fonts.len().min(200);
                            egui::ScrollArea::vertical().max_height(300.0).show_rows(
                                ui,
                                row_height,
                                total_rows,
                                |ui, row_range| {
                                    for idx in row_range {
                                        if idx >= fonts.len() {
                                            break;
                                        }
                                        let font_name = &fonts[idx];
                                        let is_selected =
                                            self.text_state.font_family == **font_name;

                                        // Queue font for preview loading if not cached
                                        if !self
                                            .text_state
                                            .font_preview_cache
                                            .contains_key(font_name.as_str())
                                            && !self
                                                .text_state
                                                .font_preview_pending
                                                .contains(font_name.as_str())
                                        {
                                            to_load.push((*font_name).clone());
                                        }

                                        // Layout: left-aligned font name + preview
                                        let resp = ui.horizontal(|ui| {
                                            let preview_text = "Abc";
                                            let preview_width = 60.0;
                                            let name_width = 250.0 - preview_width - 16.0;

                                            // Font name (left-aligned, truncated)
                                            let display_name = if font_name.len() > 22 {
                                                format!("{}…", &font_name[..21])
                                            } else {
                                                (*font_name).clone()
                                            };

                                            let label_resp = ui
                                                .with_layout(
                                                    egui::Layout::left_to_right(
                                                        egui::Align::Center,
                                                    )
                                                    .with_main_wrap(false),
                                                    |ui| {
                                                        ui.set_min_size(egui::vec2(
                                                            name_width, row_height,
                                                        ));
                                                        ui.set_max_size(egui::vec2(
                                                            name_width, row_height,
                                                        ));
                                                        ui.selectable_label(
                                                            is_selected,
                                                            &display_name,
                                                        )
                                                    },
                                                )
                                                .inner;

                                            // Font preview (rendered with ab_glyph if cached)
                                            if let Some(Some(preview_font)) = self
                                                .text_state
                                                .font_preview_cache
                                                .get(font_name.as_str())
                                            {
                                                use ab_glyph::{Font as AbFont, ScaleFont as _};
                                                let preview_size = 14.0;
                                                let scaled = preview_font.as_scaled(preview_size);
                                                let (rect, painter) = ui.allocate_painter(
                                                    egui::vec2(preview_width, row_height),
                                                    egui::Sense::hover(),
                                                );
                                                let text_color = ui.visuals().text_color();
                                                let base_y = rect.rect.min.y
                                                    + scaled.ascent()
                                                    + (row_height - preview_size) * 0.5;
                                                let mut cx = rect.rect.min.x;
                                                let mut prev_glyph: Option<ab_glyph::GlyphId> =
                                                    None;
                                                for ch in preview_text.chars() {
                                                    let gid = preview_font.glyph_id(ch);
                                                    if let Some(prev) = prev_glyph {
                                                        cx += scaled.kern(prev, gid);
                                                    }
                                                    let glyph = gid.with_scale_and_position(
                                                        preview_size,
                                                        ab_glyph::point(cx, base_y),
                                                    );
                                                    if let Some(outlined) =
                                                        preview_font.outline_glyph(glyph)
                                                    {
                                                        let bounds = outlined.px_bounds();
                                                        outlined.draw(|px, py, cov| {
                                                            if cov > 0.1 {
                                                                let x = bounds.min.x + px as f32;
                                                                let y = bounds.min.y + py as f32;
                                                                let alpha = (cov
                                                                    * text_color.a() as f32)
                                                                    .round()
                                                                    .min(255.0)
                                                                    as u8;
                                                                let c =
                                                                    Color32::from_rgba_unmultiplied(
                                                                        text_color.r(),
                                                                        text_color.g(),
                                                                        text_color.b(),
                                                                        alpha,
                                                                    );
                                                                painter.rect_filled(
                                                                    egui::Rect::from_min_size(
                                                                        egui::pos2(x, y),
                                                                        egui::vec2(1.0, 1.0),
                                                                    ),
                                                                    0.0,
                                                                    c,
                                                                );
                                                            }
                                                        });
                                                    }
                                                    cx += scaled.h_advance(gid);
                                                    prev_glyph = Some(gid);
                                                }
                                            } else {
                                                // Show placeholder while loading
                                                ui.add_sized(
                                                    [preview_width, row_height],
                                                    egui::Label::new(
                                                        egui::RichText::new("Abc").weak().italics(),
                                                    ),
                                                );
                                            }

                                            label_resp
                                        });

                                        if resp.inner.clicked() {
                                            self.text_state.font_family = (*font_name).clone();
                                            self.text_state.loaded_font = None;
                                            self.text_state.preview_dirty = true;
                                            self.text_state.cached_raster_key.clear();
                                            // Refresh available weights for new family
                                            self.text_state.available_weights =
                                                crate::ops::text::enumerate_font_weights(
                                                    &self.text_state.font_family,
                                                );
                                            if !self
                                                .text_state
                                                .available_weights
                                                .iter()
                                                .any(|w| w.1 == self.text_state.font_weight)
                                            {
                                                self.text_state.font_weight = 400;
                                            }
                                            ui.memory_mut(|mem| mem.close_popup());
                                        }
                                    }
                                },
                            );

                            // Launch async loading for visible fonts that need previews
                            if !to_load.is_empty() {
                                for name in &to_load {
                                    self.text_state.font_preview_pending.insert(name.clone());
                                }
                                let (tx, rx) = std::sync::mpsc::channel();
                                // Replace old receiver with new one
                                self.text_state.font_preview_rx = Some(rx);
                                std::thread::spawn(move || {
                                    let batch: Vec<(String, Option<ab_glyph::FontArc>)> = to_load
                                        .into_iter()
                                        .map(|name| {
                                            let font = crate::ops::text::load_system_font(
                                                &name, 400, false,
                                            );
                                            (name, font)
                                        })
                                        .collect();
                                    let _ = tx.send(batch);
                                });
                                ui.ctx()
                                    .request_repaint_after(std::time::Duration::from_millis(50));
                            }
                        });
                    });
                // Close popup if user clicks elsewhere
                let popup_rect = ui.ctx().memory(|mem| mem.area_rect(popup_id));
                if let Some(popup_rect) = popup_rect {
                    let pointer_pos = ui.input(|i| i.pointer.interact_pos());
                    let clicked_outside = ui.input(|i| i.pointer.any_click());
                    if clicked_outside
                        && let Some(pos) = pointer_pos
                        && !popup_rect.contains(pos)
                        && !button_response.rect.contains(pos)
                    {
                        ui.memory_mut(|mem| mem.close_popup());
                    }
                }
            }
        }

        // Weight dropdown (only show if more than one weight available)
        if self.text_state.available_weights.len() > 1 {
            ui.separator();
            ui.label(t!("ctx.text.weight"));
            let current_weight_label = self
                .text_state
                .available_weights
                .iter()
                .find(|w| w.1 == self.text_state.font_weight)
                .map(|w| w.0.as_str())
                .unwrap_or("Regular");
            egui::ComboBox::from_id_source("ctx_text_weight")
                .selected_text(current_weight_label)
                .width(90.0)
                .show_ui(ui, |ui| {
                    for (name, val) in &self.text_state.available_weights {
                        if ui
                            .selectable_label(self.text_state.font_weight == *val, name)
                            .clicked()
                        {
                            self.text_state.font_weight = *val;
                            self.text_state.loaded_font = None;
                            self.text_state.preview_dirty = true;
                            self.text_state.cached_raster_key.clear();
                        }
                    }
                });
        }

        ui.separator();
        ui.label(t!("ctx.size"));
        // Minus button for decrementing size
        if ui
            .add(egui::Button::new("−").min_size(egui::vec2(18.0, 18.0)))
            .clicked()
        {
            self.text_state.font_size = (self.text_state.font_size - 1.0).max(6.0);
            self.text_state.preview_dirty = true;
            self.text_state.glyph_cache.clear();
        }
        if ui
            .add(
                egui::DragValue::new(&mut self.text_state.font_size)
                    .speed(0.5)
                    .clamp_range(6.0..=500.0)
                    .suffix("px"),
            )
            .changed()
        {
            self.text_state.preview_dirty = true;
            self.text_state.glyph_cache.clear();
        }
        // Plus button for incrementing size
        if ui
            .add(egui::Button::new("+").min_size(egui::vec2(18.0, 18.0)))
            .clicked()
        {
            self.text_state.font_size = (self.text_state.font_size + 1.0).min(500.0);
            self.text_state.preview_dirty = true;
            self.text_state.glyph_cache.clear();
        }

        ui.separator();
        if ui
            .selectable_label(self.text_state.bold, t!("ctx.text.bold"))
            .clicked()
        {
            self.text_state.bold = !self.text_state.bold;
            self.text_state.loaded_font = None;
            self.text_state.preview_dirty = true;
        }
        if ui
            .selectable_label(self.text_state.italic, t!("ctx.text.italic"))
            .clicked()
        {
            self.text_state.italic = !self.text_state.italic;
            self.text_state.loaded_font = None;
            self.text_state.preview_dirty = true;
        }
        if ui
            .selectable_label(self.text_state.underline, t!("ctx.text.underline"))
            .clicked()
        {
            self.text_state.underline = !self.text_state.underline;
            self.text_state.preview_dirty = true;
        }
        if ui
            .selectable_label(self.text_state.strikethrough, t!("ctx.text.strikethrough"))
            .clicked()
        {
            self.text_state.strikethrough = !self.text_state.strikethrough;
            self.text_state.preview_dirty = true;
        }

        ui.separator();
        // Alignment: single cycle-toggle button
        let align_label = self.text_state.alignment.label();
        if ui
            .add(egui::Button::new(align_label).min_size(egui::vec2(50.0, 0.0)))
            .clicked()
        {
            self.text_state.alignment = match self.text_state.alignment {
                crate::ops::text::TextAlignment::Left => crate::ops::text::TextAlignment::Center,
                crate::ops::text::TextAlignment::Center => crate::ops::text::TextAlignment::Right,
                crate::ops::text::TextAlignment::Right => crate::ops::text::TextAlignment::Left,
            };
            self.text_state.preview_dirty = true;
        }

        ui.separator();
        // Anti-alias: compact "AA" toggle with tooltip
        let aa_resp = ui.selectable_label(self.text_state.anti_alias, t!("ctx.anti_alias"));
        if aa_resp.clicked() {
            self.text_state.anti_alias = !self.text_state.anti_alias;
            self.text_state.preview_dirty = true;
        }
        aa_resp.on_hover_text(t!("ctx.anti_alias_tooltip"));

        ui.separator();
        ui.label(t!("ctx.blend"));
        egui::ComboBox::from_id_source("ctx_text_blend")
            .selected_text(self.properties.blending_mode.name())
            .width(80.0)
            .show_ui(ui, |ui| {
                for mode in BlendMode::all() {
                    if ui
                        .selectable_label(self.properties.blending_mode == *mode, mode.name())
                        .clicked()
                    {
                        self.properties.blending_mode = *mode;
                    }
                }
            });
    }

    fn ensure_text_font_loaded(&mut self) {
        // Compute effective weight: if bold toggled and weight is normal, use Bold weight
        let effective_weight = if self.text_state.bold {
            if self.text_state.font_weight < 600 {
                700u16
            } else {
                (self.text_state.font_weight + 200).min(900)
            }
        } else {
            self.text_state.font_weight
        };
        let key = format!(
            "{}:{}:{}",
            self.text_state.font_family, effective_weight, self.text_state.italic
        );
        if self.text_state.loaded_font_key != key || self.text_state.loaded_font.is_none() {
            self.text_state.loaded_font = crate::ops::text::load_system_font(
                &self.text_state.font_family,
                effective_weight,
                self.text_state.italic,
            );
            self.text_state.loaded_font_key = key;
            // Clear glyph pixel cache when font changes (different outlines)
            self.text_state.glyph_cache.clear();
        }
    }

    fn render_text_preview(&mut self, canvas_state: &mut CanvasState, primary_color_f32: [f32; 4]) {
        let origin = match self.text_state.origin {
            Some(o) => o,
            None => return,
        };
        if self.text_state.text.is_empty() {
            canvas_state.clear_preview_state();
            return;
        }

        self.ensure_text_font_loaded();
        let font = match &self.text_state.loaded_font {
            Some(f) => f.clone(),
            None => return,
        };

        let color = [
            (primary_color_f32[0] * 255.0) as u8,
            (primary_color_f32[1] * 255.0) as u8,
            (primary_color_f32[2] * 255.0) as u8,
            (primary_color_f32[3] * 255.0) as u8,
        ];

        let result = crate::ops::text::rasterize_text(
            &font,
            &self.text_state.text,
            self.text_state.font_size,
            self.text_state.alignment,
            origin[0],
            origin[1],
            color,
            self.text_state.anti_alias,
            self.text_state.bold,
            self.text_state.italic,
            self.text_state.underline,
            self.text_state.strikethrough,
            canvas_state.width,
            canvas_state.height,
            &mut self.text_state.coverage_buf,
            &mut self.text_state.glyph_cache,
        );

        // Cache cursor metrics from rasterization (Opt 7)
        self.text_state.cached_line_advances = result.line_advances;
        self.text_state.cached_line_height = result.line_height;

        if result.buf_w == 0 || result.buf_h == 0 {
            canvas_state.clear_preview_state();
            return;
        }

        let off_x = result.off_x;
        let off_y = result.off_y;
        let buf_w = result.buf_w;
        let buf_h = result.buf_h;

        // Cache the rasterized buffer for fast drag moves
        self.text_state.cached_raster_buf = result.buf.clone();
        self.text_state.cached_raster_w = buf_w;
        self.text_state.cached_raster_h = buf_h;
        self.text_state.cached_raster_off_x = off_x;
        self.text_state.cached_raster_off_y = off_y;
        self.text_state.cached_raster_origin = self.text_state.origin;

        // Opt 1: Build TiledImage efficiently using bulk chunk copies instead of per-pixel put_pixel.
        let mut preview = TiledImage::new(canvas_state.width, canvas_state.height);
        preview.blit_rgba_at(off_x, off_y, buf_w, buf_h, &result.buf);

        canvas_state.preview_layer = Some(preview);
        canvas_state.preview_blend_mode = self.properties.blending_mode;
        // Opt 2: Only force full composite when blend mode isn't Normal
        canvas_state.preview_force_composite = self.properties.blending_mode != BlendMode::Normal;
        canvas_state.preview_is_eraser = false;
        canvas_state.preview_downscale = 1;
        canvas_state.preview_flat_ready = false;
        // Opt 2: Set stroke bounds to limit composite/extraction to text region only
        canvas_state.preview_stroke_bounds = Some(egui::Rect::from_min_max(
            egui::pos2(off_x as f32, off_y as f32),
            egui::pos2((off_x + buf_w as i32) as f32, (off_y + buf_h as i32) as f32),
        ));
        // Opt 3: Use dirty_rect to signal texture update needed instead of
        // full cache invalidation. This lets the display path do a set()
        // instead of creating a brand new texture handle.
        if canvas_state.preview_texture_cache.is_some() {
            canvas_state.preview_dirty_rect = Some(egui::Rect::from_min_max(
                egui::pos2(off_x as f32, off_y as f32),
                egui::pos2((off_x + buf_w as i32) as f32, (off_y + buf_h as i32) as f32),
            ));
        } else {
            // First time: force texture creation
            canvas_state.preview_texture_cache = None;
        }
        canvas_state.mark_dirty(None);
        self.text_state.preview_dirty = false;
    }

    fn commit_text(&mut self, canvas_state: &mut CanvasState) {
        if self.text_state.text.is_empty() || self.text_state.origin.is_none() {
            self.text_state.is_editing = false;
            self.text_state.origin = None;
            canvas_state.clear_preview_state();
            return;
        }

        // Apply mirror to text preview before committing
        canvas_state.mirror_preview_layer();

        // Set stroke bounds from preview so undo/redo captures the affected region
        if let Some(bounds) = canvas_state.preview_stroke_bounds {
            self.stroke_tracker.expand_bounds(bounds);
        }

        let stroke_event = self.stroke_tracker.finish(canvas_state);

        let blend_mode = self.properties.blending_mode;
        if let Some(ref preview) = canvas_state.preview_layer
            && let Some(active_layer) = canvas_state.layers.get_mut(canvas_state.active_layer_index)
        {
            let chunk_data: Vec<(u32, u32, image::RgbaImage)> = preview
                .chunk_keys()
                .filter_map(|(cx, cy)| preview.get_chunk(cx, cy).map(|c| (cx, cy, c.clone())))
                .collect();
            let chunk_size = crate::canvas::CHUNK_SIZE;
            for (cx, cy, chunk) in &chunk_data {
                let base_x = cx * chunk_size;
                let base_y = cy * chunk_size;
                let cw = chunk_size.min(canvas_state.width.saturating_sub(base_x));
                let ch = chunk_size.min(canvas_state.height.saturating_sub(base_y));
                for ly in 0..ch {
                    for lx in 0..cw {
                        let gx = base_x + lx;
                        let gy = base_y + ly;
                        let src = *chunk.get_pixel(lx, ly);
                        if src.0[3] == 0 {
                            continue;
                        }
                        let dst = active_layer.pixels.get_pixel_mut(gx, gy);
                        *dst = CanvasState::blend_pixel_static(*dst, src, blend_mode, 1.0);
                    }
                }
            }
            active_layer.invalidate_lod();
            active_layer.gpu_generation += 1;
        }

        self.text_state.text.clear();
        self.text_state.cursor_pos = 0;
        self.text_state.is_editing = false;
        self.text_state.origin = None;
        canvas_state.clear_preview_state();
        canvas_state.mark_dirty(None);

        if stroke_event.is_some() {
            self.pending_stroke_event = stroke_event;
        }
    }

    // ====================================================================
    // LIQUIFY TOOL — context bar, preview, commit
    // ====================================================================

    fn show_liquify_options(&mut self, ui: &mut egui::Ui) {
        ui.label(t!("ctx.mode"));
        egui::ComboBox::from_id_source("ctx_liquify_mode")
            .selected_text(self.liquify_state.mode.label())
            .width(90.0)
            .show_ui(ui, |ui| {
                for mode in LiquifyMode::all() {
                    if ui
                        .selectable_label(self.liquify_state.mode == *mode, mode.label())
                        .clicked()
                    {
                        self.liquify_state.mode = *mode;
                    }
                }
            });

        ui.separator();
        ui.label(t!("ctx.liquify.strength"));
        let mut strength_pct = (self.liquify_state.strength * 100.0).round() as i32;
        if ui
            .add(
                egui::DragValue::new(&mut strength_pct)
                    .speed(1)
                    .clamp_range(1..=100)
                    .suffix("%"),
            )
            .changed()
        {
            self.liquify_state.strength = strength_pct as f32 / 100.0;
        }

        ui.separator();
        ui.label(t!("ctx.size"));
        ui.add(
            egui::DragValue::new(&mut self.properties.size)
                .speed(0.5)
                .clamp_range(5.0..=500.0)
                .suffix("px"),
        );

        if self.liquify_state.is_active {
            ui.separator();
            ui.label(
                egui::RichText::new(t!("ctx.press_enter_to_apply"))
                    .weak()
                    .italics(),
            );
        }
    }

    fn commit_liquify(&mut self, canvas_state: &mut CanvasState) {
        let displacement = match &self.liquify_state.displacement {
            Some(d) => d,
            None => return,
        };
        let source = match &self.liquify_state.source_snapshot {
            Some(s) => s,
            None => return,
        };

        // Expand undo/redo bounds to cover the full canvas (liquify affects everything)
        self.stroke_tracker.expand_bounds(egui::Rect::from_min_max(
            egui::pos2(0.0, 0.0),
            egui::pos2(displacement.width as f32, displacement.height as f32),
        ));

        let stroke_event = self.stroke_tracker.finish(canvas_state);
        let warped = crate::ops::transform::warp_displacement_full(source, displacement);

        if let Some(active_layer) = canvas_state.layers.get_mut(canvas_state.active_layer_index) {
            let w = canvas_state.width.min(warped.width());
            let h = canvas_state.height.min(warped.height());
            for y in 0..h {
                for x in 0..w {
                    active_layer.pixels.put_pixel(x, y, *warped.get_pixel(x, y));
                }
            }
            active_layer.invalidate_lod();
            active_layer.gpu_generation += 1;
        }

        self.liquify_state.displacement = None;
        self.liquify_state.is_active = false;
        self.liquify_state.source_snapshot = None;
        self.liquify_state.warp_buffer.clear();
        self.liquify_state.dirty_rect = None;
        canvas_state.clear_preview_state();
        canvas_state.mark_dirty(None);

        if stroke_event.is_some() {
            self.pending_stroke_event = stroke_event;
        }
    }

    fn render_liquify_preview(
        &mut self,
        canvas_state: &mut CanvasState,
        gpu_renderer: Option<&mut crate::gpu::GpuRenderer>,
    ) {
        let displacement = match &self.liquify_state.displacement {
            Some(d) => d,
            None => return,
        };
        let source = match &self.liquify_state.source_snapshot {
            Some(s) => s,
            None => return,
        };

        let w = canvas_state.width;
        let h = canvas_state.height;
        let total_pixels = w as usize * h as usize;

        // ── GPU path ─────────────────────────────────────────────
        if let Some(gpu) = gpu_renderer {
            gpu.liquify_pipeline.warp_into(
                &gpu.ctx,
                source.as_raw(),
                &displacement.data,
                w,
                h,
                &mut self.liquify_state.warp_buffer,
            );
        } else {
            // ── CPU fallback ─────────────────────────────────────
            let dirty = self
                .liquify_state
                .dirty_rect
                .unwrap_or([0, 0, w as i32, h as i32]);
            let pad = self.properties.size as i32 + 20;
            let dr = [
                (dirty[0] - pad).max(0),
                (dirty[1] - pad).max(0),
                (dirty[2] + pad).min(w as i32),
                (dirty[3] + pad).min(h as i32),
            ];

            if self.liquify_state.warp_buffer.len() != total_pixels * 4 {
                self.liquify_state.warp_buffer = source.as_raw().clone();
            }

            let warped = crate::ops::transform::warp_displacement_region(
                source,
                displacement,
                &self.liquify_state.warp_buffer,
                (dr[0], dr[1], dr[2], dr[3]),
                w,
                h,
            );
            self.liquify_state.warp_buffer = warped;
        }

        let preview = TiledImage::from_raw_rgba(w, h, &self.liquify_state.warp_buffer);
        canvas_state.preview_layer = Some(preview);
        canvas_state.preview_blend_mode = BlendMode::Normal;
        canvas_state.preview_force_composite = true;
        canvas_state.preview_is_eraser = false;
        canvas_state.preview_replaces_layer = true;
        canvas_state.preview_downscale = 1;
        canvas_state.preview_stroke_bounds = Some(egui::Rect::from_min_max(
            egui::pos2(0.0, 0.0),
            egui::pos2(w as f32, h as f32),
        ));
        canvas_state.preview_texture_cache = None; // Force re-upload
        canvas_state.mark_dirty(None);
    }

    // ====================================================================
    // MESH WARP TOOL — context bar, preview, commit, overlay
    // ====================================================================

    fn show_mesh_warp_options(&mut self, ui: &mut egui::Ui) {
        ui.label(t!("ctx.mesh_warp.grid"));
        let grid_label = format!(
            "{}×{}",
            self.mesh_warp_state.grid_cols, self.mesh_warp_state.grid_rows
        );
        egui::ComboBox::from_id_source("ctx_meshwarp_grid")
            .selected_text(&grid_label)
            .width(60.0)
            .show_ui(ui, |ui| {
                for n in &[2usize, 3, 4, 5, 6] {
                    let label = format!("{}×{}", n, n);
                    if ui
                        .selectable_label(self.mesh_warp_state.grid_cols == *n, &label)
                        .clicked()
                    {
                        self.mesh_warp_state.grid_cols = *n;
                        self.mesh_warp_state.grid_rows = *n;
                        self.mesh_warp_state.is_active = false;
                        self.mesh_warp_state.points.clear();
                        self.mesh_warp_state.original_points.clear();
                        self.mesh_warp_state.source_snapshot = None;
                        self.mesh_warp_state.warp_buffer.clear();
                        self.mesh_warp_state.needs_reinit = true;
                    }
                }
            });

        if self.mesh_warp_state.is_active {
            ui.separator();
            ui.label(
                egui::RichText::new(t!("ctx.press_enter_to_apply"))
                    .weak()
                    .italics(),
            );
        }
    }

    fn init_mesh_warp_grid(&mut self, canvas_state: &CanvasState) {
        let cols = self.mesh_warp_state.grid_cols;
        let rows = self.mesh_warp_state.grid_rows;
        let w = canvas_state.width as f32;
        let h = canvas_state.height as f32;

        let mut points = Vec::with_capacity((cols + 1) * (rows + 1));
        for r in 0..=rows {
            for c in 0..=cols {
                points.push([c as f32 / cols as f32 * w, r as f32 / rows as f32 * h]);
            }
        }
        self.mesh_warp_state.original_points = points.clone();
        self.mesh_warp_state.points = points;
        self.mesh_warp_state.is_active = true;

        let idx = canvas_state.active_layer_index;
        if let Some(layer) = canvas_state.layers.get(idx) {
            self.mesh_warp_state.source_snapshot = Some(layer.pixels.to_rgba_image());
            self.mesh_warp_state.snapshot_layer_index = idx;
            self.mesh_warp_state.snapshot_generation = layer.gpu_generation;
        }
    }

    fn commit_mesh_warp(&mut self, canvas_state: &mut CanvasState) {
        let source = match &self.mesh_warp_state.source_snapshot {
            Some(s) => s,
            None => return,
        };

        // Expand undo/redo bounds to cover the full canvas
        self.stroke_tracker.expand_bounds(egui::Rect::from_min_max(
            egui::pos2(0.0, 0.0),
            egui::pos2(canvas_state.width as f32, canvas_state.height as f32),
        ));

        let stroke_event = self.stroke_tracker.finish(canvas_state);
        let warped = crate::ops::transform::warp_mesh_catmull_rom(
            source,
            &self.mesh_warp_state.original_points,
            &self.mesh_warp_state.points,
            self.mesh_warp_state.grid_cols,
            self.mesh_warp_state.grid_rows,
            canvas_state.width,
            canvas_state.height,
        );

        if let Some(active_layer) = canvas_state.layers.get_mut(canvas_state.active_layer_index) {
            let w = canvas_state.width.min(warped.width());
            let h = canvas_state.height.min(warped.height());
            for y in 0..h {
                for x in 0..w {
                    active_layer.pixels.put_pixel(x, y, *warped.get_pixel(x, y));
                }
            }
            active_layer.invalidate_lod();
            active_layer.gpu_generation += 1;
        }

        self.mesh_warp_state.is_active = false;
        self.mesh_warp_state.points.clear();
        self.mesh_warp_state.original_points.clear();
        self.mesh_warp_state.source_snapshot = None;
        self.mesh_warp_state.warp_buffer.clear();
        canvas_state.clear_preview_state();
        canvas_state.mark_dirty(None);

        if stroke_event.is_some() {
            self.pending_stroke_event = stroke_event;
        }
    }

    fn draw_mesh_warp_overlay(
        &self,
        ui: &egui::Ui,
        painter: &egui::Painter,
        canvas_rect: Rect,
        zoom: f32,
        _canvas_state: &CanvasState,
    ) {
        if !self.mesh_warp_state.is_active {
            return;
        }

        let to_screen = |cx: f32, cy: f32| -> Pos2 {
            Pos2::new(canvas_rect.min.x + cx * zoom, canvas_rect.min.y + cy * zoom)
        };

        let accent_color = ui.visuals().hyperlink_color;
        let cols = self.mesh_warp_state.grid_cols;
        let rows = self.mesh_warp_state.grid_rows;
        let pts_per_row = cols + 1;
        let grid_stroke = egui::Stroke::new(
            1.0,
            Color32::from_rgba_unmultiplied(
                accent_color.r(),
                accent_color.g(),
                accent_color.b(),
                160,
            ),
        );
        let curve_segments = 12; // Sub-segments per grid edge for smooth curves

        // Draw smooth horizontal Catmull-Rom curves
        for r in 0..=rows {
            // Collect row control points
            let row_pts: Vec<[f32; 2]> = (0..pts_per_row)
                .map(|c| self.mesh_warp_state.points[r * pts_per_row + c])
                .collect();
            let n = row_pts.len();
            if n < 2 {
                continue;
            }
            let total_segs = (n - 1) * curve_segments;
            let mut screen_pts = Vec::with_capacity(total_segs + 1);
            for s in 0..=total_segs {
                let t = s as f32 / curve_segments as f32;
                let p = crate::ops::transform::catmull_rom_curve_point(&row_pts, t);
                screen_pts.push(to_screen(p[0], p[1]));
            }
            for w in screen_pts.windows(2) {
                painter.line_segment([w[0], w[1]], grid_stroke);
            }
        }

        // Draw smooth vertical Catmull-Rom curves
        for c in 0..=cols {
            // Collect column control points
            let col_pts: Vec<[f32; 2]> = (0..=rows)
                .map(|r| self.mesh_warp_state.points[r * pts_per_row + c])
                .collect();
            let n = col_pts.len();
            if n < 2 {
                continue;
            }
            let total_segs = (n - 1) * curve_segments;
            let mut screen_pts = Vec::with_capacity(total_segs + 1);
            for s in 0..=total_segs {
                let t = s as f32 / curve_segments as f32;
                let p = crate::ops::transform::catmull_rom_curve_point(&col_pts, t);
                screen_pts.push(to_screen(p[0], p[1]));
            }
            for w in screen_pts.windows(2) {
                painter.line_segment([w[0], w[1]], grid_stroke);
            }
        }

        // Draw control point handles
        for (i, pt) in self.mesh_warp_state.points.iter().enumerate() {
            let sp = to_screen(pt[0], pt[1]);
            let is_hover = self.mesh_warp_state.hover_index == Some(i);
            let is_drag = self.mesh_warp_state.dragging_index == Some(i);
            let r = if is_hover || is_drag { 6.0 } else { 4.0 };
            let fill = if is_drag || is_hover {
                accent_color
            } else {
                Color32::WHITE
            };
            painter.circle_filled(sp, r + 1.0, Color32::BLACK);
            painter.circle_filled(sp, r, fill);
        }
    }

    // ====================================================================
    // COLOR REMOVER TOOL — context bar and commit
    // ====================================================================

    fn show_color_remover_options(&mut self, ui: &mut egui::Ui) {
        ui.label(t!("ctx.tolerance"));
        if let Some(new_val) =
            Self::tolerance_slider(ui, "cr_tol", self.color_remover_state.tolerance)
        {
            self.color_remover_state.tolerance = new_val;
        }

        ui.separator();
        ui.label(t!("ctx.color_remover.smoothness"));
        ui.add(
            egui::DragValue::new(&mut self.color_remover_state.smoothness)
                .speed(0.2)
                .clamp_range(0..=20)
                .suffix("px"),
        );

        ui.separator();
        let mut contiguous = self.color_remover_state.contiguous;
        if ui
            .checkbox(&mut contiguous, t!("ctx.color_remover.contiguous"))
            .changed()
        {
            self.color_remover_state.contiguous = contiguous;
        }
    }

    fn commit_color_removal(&mut self, canvas_state: &mut CanvasState, click_x: u32, click_y: u32) {
        // Store request for async dispatch by app.rs (shows loading bar)
        self.pending_color_removal = Some(ColorRemovalRequest {
            click_x,
            click_y,
            tolerance: self.color_remover_state.tolerance,
            smoothness: self.color_remover_state.smoothness,
            contiguous: self.color_remover_state.contiguous,
            layer_idx: canvas_state.active_layer_index,
            selection_mask: canvas_state.selection_mask.clone(),
        });
    }

    // ====================================================================
    // SMUDGE TOOL — context bar + per-pixel smear operation
    // ====================================================================

    fn show_smudge_options(&mut self, ui: &mut egui::Ui) {
        ui.label("Size:");
        ui.add(
            egui::DragValue::new(&mut self.properties.size)
                .speed(0.5)
                .clamp_range(1.0..=500.0)
                .suffix("px"),
        );
        ui.separator();
        ui.label("Strength:");
        ui.add(egui::Slider::new(&mut self.smudge_state.strength, 0.01..=1.0).fixed_decimals(2));
    }

    /// Smudge one brush circle at (cx, cy): picks up color from canvas and
    /// repaints with the accumulated pickup colour, then updates pickup.
    fn draw_smudge_no_dirty(&mut self, canvas_state: &mut CanvasState, cx: u32, cy: u32) {
        let idx = canvas_state.active_layer_index;
        let w = canvas_state.width;
        let h = canvas_state.height;
        if idx >= canvas_state.layers.len() {
            return;
        }
        let radius = (self.properties.size * 0.5).max(1.0);
        let strength = self.smudge_state.strength;

        let r_int = radius.ceil() as i32;
        let x0 = (cx as i32 - r_int).max(0) as u32;
        let y0 = (cy as i32 - r_int).max(0) as u32;
        let x1 = ((cx as i32 + r_int).min(w as i32 - 1)).max(0) as u32;
        let y1 = ((cy as i32 + r_int).min(h as i32 - 1)).max(0) as u32;

        // Collect modifications to avoid re-borrow conflict
        let mut writes: Vec<(u32, u32, [u8; 4])> = Vec::new();
        let layer = &canvas_state.layers[idx];

        for py in y0..=y1 {
            for px in x0..=x1 {
                let dx = px as f32 - cx as f32;
                let dy = py as f32 - cy as f32;
                let dist = (dx * dx + dy * dy).sqrt();
                if dist > radius {
                    continue;
                }

                let t = 1.0 - (dist / radius).powi(2);
                let alpha = (t * strength).clamp(0.0, 1.0);

                let orig = layer.pixels.get_pixel(px, py);
                let existing = [
                    orig[0] as f32,
                    orig[1] as f32,
                    orig[2] as f32,
                    orig[3] as f32,
                ];

                let new_r = (self.smudge_state.pickup_color[0] * alpha
                    + existing[0] * (1.0 - alpha))
                    .clamp(0.0, 255.0) as u8;
                let new_g = (self.smudge_state.pickup_color[1] * alpha
                    + existing[1] * (1.0 - alpha))
                    .clamp(0.0, 255.0) as u8;
                let new_b = (self.smudge_state.pickup_color[2] * alpha
                    + existing[2] * (1.0 - alpha))
                    .clamp(0.0, 255.0) as u8;
                let new_a = (self.smudge_state.pickup_color[3] * alpha
                    + existing[3] * (1.0 - alpha))
                    .clamp(0.0, 255.0) as u8;

                // Blend existing pixel into pickup for a trailing smear
                let pickup_blend = alpha * 0.2;
                self.smudge_state.pickup_color[0] = (self.smudge_state.pickup_color[0]
                    * (1.0 - pickup_blend)
                    + existing[0] * pickup_blend)
                    .clamp(0.0, 255.0);
                self.smudge_state.pickup_color[1] = (self.smudge_state.pickup_color[1]
                    * (1.0 - pickup_blend)
                    + existing[1] * pickup_blend)
                    .clamp(0.0, 255.0);
                self.smudge_state.pickup_color[2] = (self.smudge_state.pickup_color[2]
                    * (1.0 - pickup_blend)
                    + existing[2] * pickup_blend)
                    .clamp(0.0, 255.0);
                self.smudge_state.pickup_color[3] = (self.smudge_state.pickup_color[3]
                    * (1.0 - pickup_blend)
                    + existing[3] * pickup_blend)
                    .clamp(0.0, 255.0);

                writes.push((px, py, [new_r, new_g, new_b, new_a]));
            }
        }

        let layer_mut = &mut canvas_state.layers[idx];
        for (px, py, rgba) in writes {
            layer_mut.pixels.put_pixel(px, py, image::Rgba(rgba));
        }
    }

    // ====================================================================
    // SHAPES TOOL — context bar, preview, commit, overlay
    // ====================================================================

    fn show_shapes_options(&mut self, ui: &mut egui::Ui, assets: &Assets) {
        use crate::ops::shapes::{ShapeFillMode, ShapeKind};

        ui.label(t!("ctx.shapes.shape"));

        // Button showing current shape icon + name, opens grid popup
        let popup_id = ui.make_persistent_id("shapes_grid_popup");
        let btn_response = {
            let label_text = self.shapes_state.selected_shape.label();
            if let Some(tex) = assets.get_shape_texture(self.shapes_state.selected_shape) {
                let sized = egui::load::SizedTexture::from_handle(tex);
                let img =
                    egui::Image::from_texture(sized).fit_to_exact_size(egui::Vec2::splat(16.0));
                let btn = egui::Button::image_and_text(img, label_text);
                ui.add(btn)
            } else {
                ui.button(label_text)
            }
        };
        if btn_response.clicked() {
            ui.memory_mut(|m| m.toggle_popup(popup_id));
        }

        egui::popup_below_widget(ui, popup_id, &btn_response, |ui| {
            ui.set_min_width(180.0);
            let picker_shapes = ShapeKind::picker_shapes();
            let cols = 5;
            let icon_size = egui::Vec2::splat(28.0);
            let accent = ui.visuals().hyperlink_color;

            egui::Grid::new("shapes_icon_grid")
                .spacing(egui::Vec2::splat(2.0))
                .show(ui, |ui| {
                    for (i, kind) in picker_shapes.iter().enumerate() {
                        let selected = self.shapes_state.selected_shape == *kind;
                        let (rect, response) =
                            ui.allocate_exact_size(icon_size, egui::Sense::click());

                        // Highlight selected shape
                        if selected {
                            ui.painter().rect_filled(
                                rect,
                                4.0,
                                egui::Color32::from_rgba_premultiplied(
                                    accent.r(),
                                    accent.g(),
                                    accent.b(),
                                    60,
                                ),
                            );
                        }
                        if response.hovered() {
                            ui.painter().rect_filled(
                                rect,
                                4.0,
                                ui.visuals().widgets.hovered.bg_fill,
                            );
                        }

                        // Draw shape icon
                        if let Some(tex) = assets.get_shape_texture(*kind) {
                            let sized = egui::load::SizedTexture::from_handle(tex);
                            let img =
                                egui::Image::from_texture(sized).fit_to_exact_size(icon_size * 0.8);
                            let inner_rect = rect.shrink(icon_size.x * 0.1);
                            img.paint_at(ui, inner_rect);
                        } else {
                            let lbl = kind.label();
                            let short: String = lbl.chars().take(2).collect();
                            ui.painter().text(
                                rect.center(),
                                egui::Align2::CENTER_CENTER,
                                &short,
                                egui::FontId::proportional(11.0),
                                ui.visuals().text_color(),
                            );
                        }

                        if response.clicked() {
                            self.shapes_state.selected_shape = *kind;
                        }
                        response.on_hover_text(kind.label());

                        if (i + 1) % cols == 0 {
                            ui.end_row();
                        }
                    }
                });
        });

        ui.separator();
        ui.label(t!("ctx.mode"));
        egui::ComboBox::from_id_source("ctx_shapes_fill")
            .selected_text(self.shapes_state.fill_mode.label())
            .width(70.0)
            .show_ui(ui, |ui| {
                for mode in ShapeFillMode::all() {
                    if ui
                        .selectable_label(self.shapes_state.fill_mode == *mode, mode.label())
                        .clicked()
                    {
                        self.shapes_state.fill_mode = *mode;
                    }
                }
            });

        ui.separator();
        ui.label(t!("ctx.shapes.width"));
        ui.add(
            egui::DragValue::new(&mut self.properties.size)
                .speed(0.5)
                .clamp_range(1.0..=100.0)
                .suffix("px"),
        );

        if self.shapes_state.selected_shape == ShapeKind::RoundedRect {
            ui.separator();
            ui.label(t!("ctx.shapes.radius"));
            ui.add(
                egui::DragValue::new(&mut self.shapes_state.corner_radius)
                    .speed(0.5)
                    .clamp_range(0.0..=500.0)
                    .suffix("px"),
            );
        }

        ui.separator();
        let aa_resp = ui.selectable_label(self.shapes_state.anti_alias, t!("ctx.anti_alias"));
        if aa_resp.clicked() {
            self.shapes_state.anti_alias = !self.shapes_state.anti_alias;
        }
        aa_resp.on_hover_text(t!("ctx.anti_alias_tooltip"));

        ui.separator();
        ui.label(t!("ctx.blend"));
        egui::ComboBox::from_id_source("ctx_shapes_blend")
            .selected_text(self.properties.blending_mode.name())
            .width(80.0)
            .show_ui(ui, |ui| {
                for mode in BlendMode::all() {
                    if ui
                        .selectable_label(self.properties.blending_mode == *mode, mode.name())
                        .clicked()
                    {
                        self.properties.blending_mode = *mode;
                    }
                }
            });
    }

    pub fn render_shape_preview(
        &mut self,
        canvas_state: &mut CanvasState,
        primary_color_f32: [f32; 4],
        secondary_color_f32: [f32; 4],
    ) {
        use crate::ops::shapes::PlacedShape;

        let placed = if let Some(ref p) = self.shapes_state.placed {
            p.clone()
        } else if let (Some(start), Some(end)) =
            (self.shapes_state.draw_start, self.shapes_state.draw_end)
        {
            let primary = [
                (primary_color_f32[0] * 255.0) as u8,
                (primary_color_f32[1] * 255.0) as u8,
                (primary_color_f32[2] * 255.0) as u8,
                (primary_color_f32[3] * 255.0) as u8,
            ];
            let secondary = [
                (secondary_color_f32[0] * 255.0) as u8,
                (secondary_color_f32[1] * 255.0) as u8,
                (secondary_color_f32[2] * 255.0) as u8,
                (secondary_color_f32[3] * 255.0) as u8,
            ];
            let cx = (start[0] + end[0]) * 0.5;
            let cy = (start[1] + end[1]) * 0.5;
            let hw = ((end[0] - start[0]) * 0.5).abs();
            let hh = ((end[1] - start[1]) * 0.5).abs();
            PlacedShape {
                cx,
                cy,
                hw,
                hh,
                rotation: 0.0,
                kind: self.shapes_state.selected_shape,
                fill_mode: self.shapes_state.fill_mode,
                outline_width: self.properties.size,
                primary_color: primary,
                secondary_color: secondary,
                anti_alias: self.shapes_state.anti_alias,
                corner_radius: self.shapes_state.corner_radius,
                handle_dragging: None,
                drag_offset: [0.0, 0.0],
                drag_anchor: [0.0, 0.0],
                rotate_start_angle: 0.0,
                rotate_start_rotation: 0.0,
            }
        } else {
            canvas_state.clear_preview_state();
            return;
        };

        let (buf_w, buf_h, off_x, off_y) = crate::ops::shapes::rasterize_shape_into(
            &placed,
            canvas_state.width,
            canvas_state.height,
            &mut self.shapes_state.cached_shape_buf,
        );

        if buf_w == 0 || buf_h == 0 {
            canvas_state.clear_preview_state();
            return;
        }

        let preview = TiledImage::from_region_rgba(
            canvas_state.width,
            canvas_state.height,
            &self.shapes_state.cached_shape_buf,
            buf_w,
            buf_h,
            off_x,
            off_y,
        );

        canvas_state.preview_layer = Some(preview);
        canvas_state.preview_blend_mode = self.properties.blending_mode;
        canvas_state.preview_force_composite = true;
        canvas_state.preview_is_eraser = false;
        canvas_state.preview_downscale = 1;
        canvas_state.preview_stroke_bounds = Some(egui::Rect::from_min_max(
            egui::pos2(off_x as f32, off_y as f32),
            egui::pos2((off_x + buf_w as i32) as f32, (off_y + buf_h as i32) as f32),
        ));
        canvas_state.preview_texture_cache = None; // Force re-upload
        canvas_state.mark_dirty(None);
    }

    fn commit_shape(&mut self, canvas_state: &mut CanvasState) {
        if self.shapes_state.placed.is_none() && self.shapes_state.draw_start.is_none() {
            return;
        }

        // Apply mirror to shape preview before committing
        canvas_state.mirror_preview_layer();

        let blend_mode = self.properties.blending_mode;
        let stroke_event = self.stroke_tracker.finish(canvas_state);

        if let Some(ref preview) = canvas_state.preview_layer
            && let Some(active_layer) = canvas_state.layers.get_mut(canvas_state.active_layer_index)
        {
            let chunk_data: Vec<(u32, u32, image::RgbaImage)> = preview
                .chunk_keys()
                .filter_map(|(cx, cy)| preview.get_chunk(cx, cy).map(|c| (cx, cy, c.clone())))
                .collect();
            let chunk_size = crate::canvas::CHUNK_SIZE;
            for (cx, cy, chunk) in &chunk_data {
                let base_x = cx * chunk_size;
                let base_y = cy * chunk_size;
                let cw = chunk_size.min(canvas_state.width.saturating_sub(base_x));
                let ch = chunk_size.min(canvas_state.height.saturating_sub(base_y));
                for ly in 0..ch {
                    for lx in 0..cw {
                        let gx = base_x + lx;
                        let gy = base_y + ly;
                        let src = *chunk.get_pixel(lx, ly);
                        if src.0[3] == 0 {
                            continue;
                        }
                        let dst = active_layer.pixels.get_pixel_mut(gx, gy);
                        *dst = CanvasState::blend_pixel_static(*dst, src, blend_mode, 1.0);
                    }
                }
            }
            active_layer.invalidate_lod();
            active_layer.gpu_generation += 1;
        }

        self.shapes_state.placed = None;
        self.shapes_state.draw_start = None;
        self.shapes_state.draw_end = None;
        self.shapes_state.is_drawing = false;
        canvas_state.clear_preview_state();
        canvas_state.mark_dirty(None);

        if stroke_event.is_some() {
            self.pending_stroke_event = stroke_event;
        }
    }

    fn draw_shape_overlay(&self, painter: &egui::Painter, canvas_rect: Rect, zoom: f32) {
        let placed = match &self.shapes_state.placed {
            Some(p) => p,
            None => return,
        };

        let to_screen = |cx: f32, cy: f32| -> Pos2 {
            Pos2::new(canvas_rect.min.x + cx * zoom, canvas_rect.min.y + cy * zoom)
        };

        let cos_r = placed.rotation.cos();
        let sin_r = placed.rotation.sin();
        let corners = [
            (-placed.hw, -placed.hh),
            (placed.hw, -placed.hh),
            (placed.hw, placed.hh),
            (-placed.hw, placed.hh),
        ];
        let screen_corners: Vec<Pos2> = corners
            .iter()
            .map(|(cx, cy)| {
                to_screen(
                    cx * cos_r - cy * sin_r + placed.cx,
                    cx * sin_r + cy * cos_r + placed.cy,
                )
            })
            .collect();

        // Use accent color for the selection frame
        let accent = Color32::from_rgb(
            painter.ctx().style().visuals.hyperlink_color.r(),
            painter.ctx().style().visuals.hyperlink_color.g(),
            painter.ctx().style().visuals.hyperlink_color.b(),
        );
        let accent_semi = Color32::from_rgba_unmultiplied(accent.r(), accent.g(), accent.b(), 200);
        let bb_stroke = egui::Stroke::new(1.0, accent_semi);
        for i in 0..4 {
            painter.line_segment([screen_corners[i], screen_corners[(i + 1) % 4]], bb_stroke);
        }

        let hs = 4.0;
        for corner in &screen_corners {
            painter.rect_filled(
                egui::Rect::from_center_size(*corner, egui::vec2(hs * 2.0, hs * 2.0)),
                1.0,
                accent,
            );
            painter.rect_stroke(
                egui::Rect::from_center_size(*corner, egui::vec2(hs * 2.0, hs * 2.0)),
                1.0,
                egui::Stroke::new(1.0, Color32::BLACK),
            );
        }

        // Rotation handle: faces outward from center through the top edge midpoint
        let top_mid = Pos2::new(
            (screen_corners[0].x + screen_corners[1].x) * 0.5,
            (screen_corners[0].y + screen_corners[1].y) * 0.5,
        );
        let center = to_screen(placed.cx, placed.cy);
        let dir_x = top_mid.x - center.x;
        let dir_y = top_mid.y - center.y;
        let dir_len = (dir_x * dir_x + dir_y * dir_y).sqrt().max(0.001);
        let rot_handle = Pos2::new(
            top_mid.x + (dir_x / dir_len) * 20.0,
            top_mid.y + (dir_y / dir_len) * 20.0,
        );
        painter.line_segment([top_mid, rot_handle], egui::Stroke::new(1.0, accent_semi));
        painter.circle_filled(rot_handle, 4.0, accent);
        painter.circle_stroke(rot_handle, 4.0, egui::Stroke::new(1.0, Color32::BLACK));
    }
}
