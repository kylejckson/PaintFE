use ab_glyph::{Font, FontArc, ScaleFont};
use image::{Rgba, RgbaImage};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::canvas::TiledImage;
use crate::ops::text::{self, GlyphPixelCache};

// ---------------------------------------------------------------------------
// Core data types
// ---------------------------------------------------------------------------

/// Complete text layer state — everything needed to re-rasterize.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TextLayerData {
    /// Independent text blocks within this layer.
    pub blocks: Vec<TextBlock>,
    /// Layer-level effects applied after rasterization.
    pub effects: TextEffects,
    /// Cache invalidation counter — bumped on every edit.
    #[serde(skip)]
    pub cache_generation: u64,
    /// Generation that was last rasterized into `layer.pixels`.
    #[serde(skip)]
    pub raster_generation: u64,
    /// Counter for generating unique block IDs.
    #[serde(skip)]
    pub next_block_id: u64,
    /// Cached pre-effects RGBA buffer (text only, no effects).
    /// When only effect parameters change, skip text re-rasterization.
    #[serde(skip)]
    pub cached_text_rgba: Vec<u8>,
    /// Generation counter for the text content cache (separate from effects).
    #[serde(skip)]
    pub text_content_generation: u64,
    /// The generation at which `cached_text_rgba` was last computed.
    #[serde(skip)]
    pub cached_text_generation: u64,
    /// Dimensions of the cached text RGBA buffer.
    #[serde(skip)]
    pub cached_text_w: u32,
    #[serde(skip)]
    pub cached_text_h: u32,
    /// Generation counter for block position changes (not content).
    #[serde(skip)]
    pub position_generation: u64,
    /// The position generation at which `cached_text_rgba` was last computed.
    #[serde(skip)]
    pub cached_position_generation: u64,
}

/// Cached rasterization result for a single text block.
/// Stores the tight pixel buffer and the origin used to compute offsets.
/// On position-only change, new offsets are derived without re-rasterizing.
#[derive(Clone, Debug)]
pub struct CachedBlockRaster {
    pub buf: Vec<u8>,
    pub buf_w: u32,
    pub buf_h: u32,
    pub off_x: i32,
    pub off_y: i32,
    pub origin: [f32; 2],
    pub content_generation: u64,
}

/// A single text block (paragraph/text field) within a text layer.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TextBlock {
    pub id: u64,
    /// Position of the text block origin on the canvas (pixels).
    pub position: [f32; 2],
    /// Rotation in radians around the block origin.
    pub rotation: f32,
    /// The rich text content (attributed runs).
    pub runs: Vec<TextRun>,
    /// Paragraph-level properties.
    pub paragraph: ParagraphStyle,
    /// Optional bounding box width for text wrapping (None = no wrap).
    pub max_width: Option<f32>,
    /// Optional bounding box height (None = auto-height from content).
    /// When set, the visual box height is max(content_height, max_height).
    #[serde(default)]
    pub max_height: Option<f32>,
    /// Geometric warp applied to this block (Phase 4 — Batches 7+8).
    #[serde(default)]
    pub warp: TextWarp,
    /// Per-glyph vertex overrides for manual glyph editing (Phase 5 — Batch 9).
    #[serde(default)]
    pub glyph_overrides: Vec<GlyphOverride>,
    /// Cached rasterized buffer for fast position-only updates.
    #[serde(skip)]
    pub cached_raster: Option<CachedBlockRaster>,
}

// ---------------------------------------------------------------------------
// Per-Glyph Vertex Override (Phase 5 — Batch 9)
// ---------------------------------------------------------------------------

/// Per-glyph vertex override for manual glyph-level editing.
/// Stores position offset, rotation, and scale for an individual glyph.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GlyphOverride {
    /// Index into the laid-out glyph sequence (flat across all runs).
    pub glyph_index: usize,
    /// Position offset in pixels (added to the computed glyph position).
    pub position_offset: [f32; 2],
    /// Per-glyph rotation in radians around the glyph center.
    pub rotation: f32,
    /// Per-glyph uniform scale factor (1.0 = normal).
    pub scale: f32,
}

impl Default for GlyphOverride {
    fn default() -> Self {
        Self {
            glyph_index: 0,
            position_offset: [0.0, 0.0],
            rotation: 0.0,
            scale: 1.0,
        }
    }
}

impl GlyphOverride {
    /// Returns true if this override is effectively identity (no visible change).
    pub fn is_identity(&self) -> bool {
        self.position_offset[0].abs() < 0.001
            && self.position_offset[1].abs() < 0.001
            && self.rotation.abs() < 0.001
            && (self.scale - 1.0).abs() < 0.001
    }
}

/// A contiguous run of text with uniform formatting.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TextRun {
    pub text: String,
    pub style: TextStyle,
}

/// Character-level formatting attributes.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TextStyle {
    pub font_family: String,
    pub font_weight: u16,
    pub font_size: f32,
    pub italic: bool,
    pub underline: bool,
    pub strikethrough: bool,
    pub color: [u8; 4],
    pub letter_spacing: f32,
    pub baseline_offset: f32,
    pub width_scale: f32,
    pub height_scale: f32,
}

/// Paragraph-level formatting.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ParagraphStyle {
    pub alignment: TextAlignment,
    pub line_spacing: f32,
    pub indent: f32,
}

/// Text alignment for paragraphs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TextAlignment {
    Left,
    Center,
    Right,
}

// ---------------------------------------------------------------------------
// Geometric Warps (Phase 4 — Batches 7+8)
// ---------------------------------------------------------------------------

/// Geometric warp for a text block.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub enum TextWarp {
    #[default]
    None,
    /// Text bent along an arc (cup/bulge).
    Arc(ArcWarp),
    /// Text arranged in a circle.
    Circular(CircularWarp),
    /// Text follows a Bézier curve path.
    PathFollow(PathFollowWarp),
    /// Text warped via an envelope (top + bottom curves).
    Envelope(EnvelopeWarp),
}

impl TextWarp {
    /// Returns a display name for the warp type.
    pub fn name(&self) -> &'static str {
        match self {
            TextWarp::None => "None",
            TextWarp::Arc(_) => "Arc",
            TextWarp::Circular(_) => "Circular",
            TextWarp::PathFollow(_) => "Path Follow",
            TextWarp::Envelope(_) => "Envelope",
        }
    }

    /// Returns a list of all warp type names (for UI dropdown).
    pub fn all_names() -> &'static [&'static str] {
        &["None", "Arc", "Circular", "Path Follow", "Envelope"]
    }

    /// Create a warp variant from its name string.
    pub fn from_name(name: &str) -> Self {
        match name {
            "Arc" => TextWarp::Arc(ArcWarp::default()),
            "Circular" => TextWarp::Circular(CircularWarp::default()),
            "Path Follow" => TextWarp::PathFollow(PathFollowWarp::default()),
            "Envelope" => TextWarp::Envelope(EnvelopeWarp::default()),
            _ => TextWarp::None,
        }
    }
}

impl PartialEq for TextWarp {
    fn eq(&self, other: &Self) -> bool {
        std::mem::discriminant(self) == std::mem::discriminant(other)
    }
}

/// Arc warp: bends text along an arc.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ArcWarp {
    /// Bend amount: -1.0 = concave, 0.0 = flat, 1.0 = convex.
    pub bend: f32,
    /// Horizontal distortion (-1..1).
    pub horizontal_distortion: f32,
    /// Vertical distortion (-1..1).
    pub vertical_distortion: f32,
}

impl Default for ArcWarp {
    fn default() -> Self {
        Self {
            bend: 0.5,
            horizontal_distortion: 0.0,
            vertical_distortion: 0.0,
        }
    }
}

/// Circular warp: arranges text around a circle.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircularWarp {
    /// Radius of the circle baseline (pixels).
    pub radius: f32,
    /// Start angle (radians).
    pub start_angle: f32,
    /// Whether text goes clockwise.
    pub clockwise: bool,
}

impl Default for CircularWarp {
    fn default() -> Self {
        Self {
            radius: 150.0,
            start_angle: -std::f32::consts::FRAC_PI_2,
            clockwise: true,
        }
    }
}

/// Path follow warp: text follows a Bézier curve.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PathFollowWarp {
    /// Cubic Bézier control points defining the baseline path.
    /// Groups of 4: [start, cp1, cp2, end] per segment.
    pub control_points: Vec<[f32; 2]>,
}

impl Default for PathFollowWarp {
    fn default() -> Self {
        // Default: a simple horizontal path with slight curve
        Self {
            control_points: vec![[0.0, 0.0], [100.0, -50.0], [200.0, -50.0], [300.0, 0.0]],
        }
    }
}

/// Envelope warp: deforms text between two Bézier curves.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EnvelopeWarp {
    /// Top boundary curve control points (cubic Bézier, groups of 4).
    pub top_curve: Vec<[f32; 2]>,
    /// Bottom boundary curve control points (cubic Bézier, groups of 4).
    pub bottom_curve: Vec<[f32; 2]>,
}

impl Default for EnvelopeWarp {
    fn default() -> Self {
        Self {
            top_curve: vec![[0.0, -30.0], [100.0, -60.0], [200.0, -60.0], [300.0, -30.0]],
            bottom_curve: vec![[0.0, 30.0], [100.0, 60.0], [200.0, 60.0], [300.0, 30.0]],
        }
    }
}

/// Layer-level text effects (Batch 5+6).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct TextEffects {
    pub outline: Option<OutlineEffect>,
    pub shadow: Option<ShadowEffect>,
    pub inner_shadow: Option<InnerShadowEffect>,
    pub gradient_fill: Option<GradientFillEffect>,
    pub texture_fill: Option<TextureFillEffect>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OutlineEffect {
    pub color: [u8; 4],
    pub width: f32,
    pub position: OutlinePosition,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutlinePosition {
    Inside,
    Outside,
    Center,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShadowEffect {
    pub color: [u8; 4],
    pub offset_x: f32,
    pub offset_y: f32,
    pub blur_radius: f32,
    pub spread: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InnerShadowEffect {
    pub color: [u8; 4],
    pub offset_x: f32,
    pub offset_y: f32,
    pub blur_radius: f32,
}

/// Texture fill effect: fills text glyphs with a tiled image pattern.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TextureFillEffect {
    /// Embedded texture image data (PNG bytes).
    pub texture_data: Vec<u8>,
    /// Decoded texture dimensions (cached on load).
    pub texture_width: u32,
    pub texture_height: u32,
    /// Scale factor for texture tiling (1.0 = 1:1).
    pub scale: f32,
    /// Offset for texture tiling origin.
    pub offset: [f32; 2],
}

/// Gradient fill effect: fills text glyphs with a configurable linear gradient.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GradientFillEffect {
    pub start_color: [u8; 4],
    pub end_color: [u8; 4],
    /// Gradient angle in degrees.
    pub angle_degrees: f32,
    /// Distance in pixels from start to end color.
    pub scale: f32,
    /// Translation of the gradient field in canvas pixels.
    pub offset: [f32; 2],
    pub repeat: bool,
}

impl Default for OutlineEffect {
    fn default() -> Self {
        Self {
            color: [0, 0, 0, 255],
            width: 2.0,
            position: OutlinePosition::Outside,
        }
    }
}

impl Default for ShadowEffect {
    fn default() -> Self {
        Self {
            color: [0, 0, 0, 180],
            offset_x: 4.0,
            offset_y: 4.0,
            blur_radius: 5.0,
            spread: 0.0,
        }
    }
}

impl Default for InnerShadowEffect {
    fn default() -> Self {
        Self {
            color: [0, 0, 0, 128],
            offset_x: 2.0,
            offset_y: 2.0,
            blur_radius: 3.0,
        }
    }
}

impl Default for TextureFillEffect {
    fn default() -> Self {
        Self {
            texture_data: Vec::new(),
            texture_width: 0,
            texture_height: 0,
            scale: 1.0,
            offset: [0.0, 0.0],
        }
    }
}

impl Default for GradientFillEffect {
    fn default() -> Self {
        Self {
            start_color: [255, 255, 255, 255],
            end_color: [0, 0, 0, 255],
            angle_degrees: 0.0,
            scale: 200.0,
            offset: [0.0, 0.0],
            repeat: false,
        }
    }
}

impl TextEffects {
    /// Returns true if any effect is enabled.
    pub fn has_any(&self) -> bool {
        self.outline.is_some()
            || self.shadow.is_some()
            || self.inner_shadow.is_some()
            || self.gradient_fill.is_some()
            || self.texture_fill.is_some()
    }
}

// ---------------------------------------------------------------------------
// Text Selection Model (Batch 3)
// ---------------------------------------------------------------------------

/// A position within a block's run-based text model.
/// `run_index` identifies which run, `byte_offset` identifies where within that run's text.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RunPosition {
    pub run_index: usize,
    pub byte_offset: usize,
}

/// Selection within a single text block. Anchor is where the selection started,
/// cursor is where it currently is. If anchor == cursor, there is no selection
/// (just a caret).
#[derive(Clone, Copy, Debug, Default)]
pub struct TextSelection {
    pub anchor: RunPosition,
    pub cursor: RunPosition,
}

/// Per-run layout result — bounding info for hit-testing and cursor positioning.
#[derive(Clone, Debug)]
pub struct RunLayoutInfo {
    /// Per-character cumulative x-advance from the line start (after alignment offset).
    /// Length = number of chars + 1 (entry 0 = 0.0).
    pub char_advances: Vec<f32>,
    /// Baseline y offset from block origin for this line segment.
    pub baseline_y: f32,
    /// Line height for this run's font/size.
    pub line_height: f32,
    /// Ascent for this run's font/size.
    pub ascent: f32,
}

/// Result of multi-run layout computation.
#[derive(Clone, Debug)]
pub struct BlockLayout {
    /// Per-run layout metrics.
    pub runs: Vec<RunLayoutInfo>,
    /// Per-line info: (line_start_run_idx, line_widths, max_ascent, max_line_height)
    pub lines: Vec<LineInfo>,
    /// Total bounding width of the block.
    pub total_width: f32,
    /// Total bounding height of the block.
    pub total_height: f32,
}

/// Info about a single line in the block layout.
#[derive(Clone, Debug)]
pub struct LineInfo {
    pub width: f32,
    pub max_ascent: f32,
    pub max_line_height: f32,
}

// ---------------------------------------------------------------------------
// Defaults
// ---------------------------------------------------------------------------

impl Default for TextStyle {
    fn default() -> Self {
        Self {
            font_family: "Arial".to_string(),
            font_weight: 400,
            font_size: 48.0,
            italic: false,
            underline: false,
            strikethrough: false,
            color: [255, 255, 255, 255],
            letter_spacing: 0.0,
            baseline_offset: 0.0,
            width_scale: 1.0,
            height_scale: 1.0,
        }
    }
}

impl Default for ParagraphStyle {
    fn default() -> Self {
        Self {
            alignment: TextAlignment::Left,
            line_spacing: 1.0,
            indent: 0.0,
        }
    }
}

impl Default for TextLayerData {
    fn default() -> Self {
        Self {
            blocks: vec![TextBlock {
                id: 1,
                position: [100.0, 100.0],
                rotation: 0.0,
                runs: vec![TextRun {
                    text: String::new(),
                    style: TextStyle::default(),
                }],
                paragraph: ParagraphStyle::default(),
                max_width: None,
                max_height: None,
                warp: TextWarp::None,
                glyph_overrides: Vec::new(),
                cached_raster: None,
            }],
            effects: TextEffects::default(),
            cache_generation: 1,
            raster_generation: 0,
            next_block_id: 2,
            cached_text_rgba: Vec::new(),
            text_content_generation: 0,
            cached_text_generation: 0,
            cached_text_w: 0,
            cached_text_h: 0,
            position_generation: 0,
            cached_position_generation: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Rasterization
// ---------------------------------------------------------------------------

impl TextLayerData {
    /// Returns true if the cached rasterized pixels are stale.
    pub fn needs_rasterize(&self) -> bool {
        self.cache_generation != self.raster_generation
    }

    /// Mark text data as dirty (call after any text content edit).
    pub fn mark_dirty(&mut self) {
        self.cache_generation = self.cache_generation.wrapping_add(1);
        self.text_content_generation = self.text_content_generation.wrapping_add(1);
        self.position_generation = self.position_generation.wrapping_add(1);
    }

    /// Mark only effects as dirty (text content unchanged — can reuse cached text RGBA).
    pub fn mark_effects_dirty(&mut self) {
        self.cache_generation = self.cache_generation.wrapping_add(1);
    }

    /// Mark as dirty due to position-only change (preserves text content cache).
    /// The glyph pixel cache will NOT be cleared since text_content_generation is unchanged.
    pub fn mark_position_dirty(&mut self) {
        self.cache_generation = self.cache_generation.wrapping_add(1);
        self.position_generation = self.position_generation.wrapping_add(1);
    }

    /// Concatenate all text from all blocks into a single flat string
    /// (for simple single-block display).
    pub fn flat_text(&self) -> String {
        let mut out = String::new();
        for block in &self.blocks {
            for run in &block.runs {
                out.push_str(&run.text);
            }
        }
        out
    }

    /// Get the primary style (from the first run of the first block).
    pub fn primary_style(&self) -> &TextStyle {
        self.blocks
            .first()
            .and_then(|b| b.runs.first())
            .map(|r| &r.style)
            .unwrap_or_else(|| {
                static DEFAULT: std::sync::OnceLock<TextStyle> = std::sync::OnceLock::new();
                DEFAULT.get_or_init(TextStyle::default)
            })
    }

    /// Create a new block at the given position, returns its ID.
    pub fn add_block(&mut self, x: f32, y: f32) -> u64 {
        let id = self.next_block_id;
        self.next_block_id += 1;
        self.blocks.push(TextBlock {
            id,
            position: [x, y],
            rotation: 0.0,
            runs: vec![TextRun {
                text: String::new(),
                style: self.primary_style().clone(),
            }],
            paragraph: ParagraphStyle::default(),
            max_width: None,
            max_height: None,
            warp: TextWarp::None,
            glyph_overrides: Vec::new(),
            cached_raster: None,
        });
        self.mark_dirty();
        id
    }

    /// Remove empty blocks (blocks whose runs all have empty text).
    /// Returns the number of blocks removed.
    pub fn remove_empty_blocks(&mut self) -> usize {
        let before = self.blocks.len();
        self.blocks
            .retain(|b| b.runs.iter().any(|r| !r.text.is_empty()));
        let removed = before - self.blocks.len();
        if removed > 0 {
            self.mark_dirty();
        }
        removed
    }

    /// Find block index by ID.
    pub fn block_index_by_id(&self, id: u64) -> Option<usize> {
        self.blocks.iter().position(|b| b.id == id)
    }

    /// Rasterize all text blocks into a `TiledImage`, applying any active effects.
    pub fn rasterize(
        &mut self,
        canvas_w: u32,
        canvas_h: u32,
        coverage_buf: &mut Vec<f32>,
        glyph_cache: &mut GlyphPixelCache,
    ) -> TiledImage {
        let has_effects = self.effects.has_any();

        // If text content AND positions haven't changed, reuse cached text RGBA.
        let text_rgba_valid = self.cached_text_generation == self.text_content_generation
            && self.cached_position_generation == self.position_generation
            && !self.cached_text_rgba.is_empty()
            && self.cached_text_w == canvas_w
            && self.cached_text_h == canvas_h;

        if has_effects {
            // We need the raw text RGBA (no effects) to apply effects on top.
            if !text_rgba_valid {
                // Rasterize text into a flat RGBA buffer
                let mut text_tiled = TiledImage::new(canvas_w, canvas_h);
                let content_gen = self.text_content_generation;
                for i in 0..self.blocks.len() {
                    rasterize_block_multirun(
                        &mut self.blocks[i],
                        content_gen,
                        canvas_w,
                        canvas_h,
                        &mut text_tiled,
                        coverage_buf,
                        glyph_cache,
                    );
                }
                self.cached_text_rgba = text_tiled.extract_region_rgba(0, 0, canvas_w, canvas_h);
                self.cached_text_generation = self.text_content_generation;
                self.cached_position_generation = self.position_generation;
                self.cached_text_w = canvas_w;
                self.cached_text_h = canvas_h;
            }

            // Compute the maximum padding required by active effects.
            // Shadow: offset + blur_radius + spread.  Outline: width.
            let mut pad = 0.0f32;
            if let Some(ref shadow) = self.effects.shadow {
                pad = pad.max(
                    shadow.offset_x.abs()
                        + shadow.offset_y.abs()
                        + shadow.blur_radius * 3.0
                        + shadow.spread,
                );
            }
            if let Some(ref outline) = self.effects.outline {
                pad = pad.max(outline.width.ceil() + 2.0);
            }
            if let Some(ref inner) = self.effects.inner_shadow {
                pad =
                    pad.max(inner.offset_x.abs() + inner.offset_y.abs() + inner.blur_radius * 3.0);
            }
            let pad_px = (pad.ceil() as u32).max(4);

            // Find tight AABB of non-transparent text pixels
            let (bx, by, bw, bh, _) =
                find_tight_bounds_rgba(&self.cached_text_rgba, canvas_w, canvas_h);

            if bw == 0 || bh == 0 {
                // No visible text — return empty
                return TiledImage::new(canvas_w, canvas_h);
            }

            // Expand bounds by effect padding, clamped to canvas
            let rx = (bx as i32 - pad_px as i32).max(0) as u32;
            let ry = (by as i32 - pad_px as i32).max(0) as u32;
            let rx2 = (bx + bw + pad_px).min(canvas_w);
            let ry2 = (by + bh + pad_px).min(canvas_h);
            let rw = rx2 - rx;
            let rh = ry2 - ry;

            // If the region covers most of the canvas, fall back to full-canvas path
            let region_area = (rw as u64) * (rh as u64);
            let canvas_area = (canvas_w as u64) * (canvas_h as u64);
            if region_area * 2 >= canvas_area {
                let final_rgba =
                    apply_text_effects(&self.cached_text_rgba, canvas_w, canvas_h, &self.effects);
                TiledImage::from_raw_rgba(canvas_w, canvas_h, &final_rgba)
            } else {
                // Extract tight region from cached RGBA
                let cw = canvas_w as usize;
                let mut region_in = vec![0u8; (rw as usize) * (rh as usize) * 4];
                for row in 0..rh as usize {
                    let src_off = ((ry as usize + row) * cw + rx as usize) * 4;
                    let dst_off = row * rw as usize * 4;
                    let len = rw as usize * 4;
                    region_in[dst_off..dst_off + len]
                        .copy_from_slice(&self.cached_text_rgba[src_off..src_off + len]);
                }

                let region_out = apply_text_effects(&region_in, rw, rh, &self.effects);

                // Blit the effects result back into a TiledImage at the correct offset
                let mut result = TiledImage::new(canvas_w, canvas_h);
                result.blit_rgba_at(rx as i32, ry as i32, rw, rh, &region_out);
                result
            }
        } else {
            // No effects — standard rasterization.
            // Skip the full-canvas cache extraction (it's only needed when
            // effects are added later, and will be populated on demand).
            let mut result = TiledImage::new(canvas_w, canvas_h);
            let content_gen = self.text_content_generation;
            for i in 0..self.blocks.len() {
                rasterize_block_multirun(
                    &mut self.blocks[i],
                    content_gen,
                    canvas_w,
                    canvas_h,
                    &mut result,
                    coverage_buf,
                    glyph_cache,
                );
            }
            // Invalidate cached text RGBA — it will be recomputed if effects are
            // added later. This avoids a full-canvas extract_region_rgba every frame.
            self.cached_text_rgba.clear();
            self.cached_text_generation = 0;
            self.cached_position_generation = 0;
            self.cached_text_w = 0;
            self.cached_text_h = 0;
            result
        }
    }
}

// ---------------------------------------------------------------------------
// TextBlock helpers
// ---------------------------------------------------------------------------

impl TextBlock {
    /// Get the total flat text across all runs.
    pub fn flat_text(&self) -> String {
        self.runs.iter().map(|r| r.text.as_str()).collect()
    }

    /// Total character count across all runs.
    pub fn char_count(&self) -> usize {
        self.runs.iter().map(|r| r.text.chars().count()).sum()
    }

    /// Convert a flat byte offset (into the concatenated text) to a `RunPosition`.
    pub fn flat_offset_to_run_pos(&self, flat_byte: usize) -> RunPosition {
        let mut remaining = flat_byte;
        for (ri, run) in self.runs.iter().enumerate() {
            let len = run.text.len();
            if remaining <= len {
                return RunPosition {
                    run_index: ri,
                    byte_offset: remaining,
                };
            }
            remaining -= len;
        }
        // Past end — clamp to end of last run
        RunPosition {
            run_index: self.runs.len().saturating_sub(1),
            byte_offset: self.runs.last().map_or(0, |r| r.text.len()),
        }
    }

    /// Convert a `RunPosition` to a flat byte offset.
    pub fn run_pos_to_flat_offset(&self, pos: RunPosition) -> usize {
        let mut offset = 0;
        for (ri, run) in self.runs.iter().enumerate() {
            if ri == pos.run_index {
                return offset + pos.byte_offset.min(run.text.len());
            }
            offset += run.text.len();
        }
        offset
    }

    /// Apply a `TextStyle` to the range `[start_byte..end_byte)` (flat byte offsets).
    /// Splits runs at boundaries as needed and applies the style to the middle portion.
    /// Returns the new run positions for the affected range.
    pub fn apply_style_to_range(
        &mut self,
        start_byte: usize,
        end_byte: usize,
        apply: impl Fn(&mut TextStyle),
    ) {
        if start_byte >= end_byte {
            return;
        }

        // Strategy: split runs at start and end boundaries, then apply style to
        // all runs fully inside the range.
        self.split_at_flat_offset(end_byte);
        self.split_at_flat_offset(start_byte);

        // Now apply the style to runs within [start_byte .. end_byte)
        let mut offset = 0usize;
        for run in &mut self.runs {
            let run_end = offset + run.text.len();
            if offset >= start_byte && run_end <= end_byte && !run.text.is_empty() {
                apply(&mut run.style);
            }
            offset = run_end;
        }

        self.merge_adjacent_runs();
    }

    /// Split runs so that there is a run boundary at the given flat byte offset.
    fn split_at_flat_offset(&mut self, flat_byte: usize) {
        let mut offset = 0usize;
        for i in 0..self.runs.len() {
            let run_len = self.runs[i].text.len();
            if offset == flat_byte || offset + run_len <= flat_byte {
                offset += run_len;
                continue;
            }
            // flat_byte falls inside this run — split it
            let local = flat_byte - offset;
            if local > 0 && local < run_len {
                let tail_text = self.runs[i].text[local..].to_string();
                let tail_style = self.runs[i].style.clone();
                self.runs[i].text.truncate(local);
                self.runs.insert(
                    i + 1,
                    TextRun {
                        text: tail_text,
                        style: tail_style,
                    },
                );
            }
            return;
        }
    }

    /// Merge adjacent runs that have identical styles.
    pub fn merge_adjacent_runs(&mut self) {
        let mut i = 0;
        while i + 1 < self.runs.len() {
            if self.runs[i].style == self.runs[i + 1].style {
                let next_text = self.runs[i + 1].text.clone();
                self.runs[i].text.push_str(&next_text);
                self.runs.remove(i + 1);
            } else if self.runs[i].text.is_empty() {
                self.runs.remove(i);
            } else {
                i += 1;
            }
        }
        // Remove trailing empty runs (keep at least one)
        while self.runs.len() > 1 && self.runs.last().is_some_and(|r| r.text.is_empty()) {
            self.runs.pop();
        }
    }

    /// Insert text at a flat byte offset, inheriting the style of the run at that position.
    pub fn insert_text_at(&mut self, flat_byte: usize, text: &str) {
        let pos = self.flat_offset_to_run_pos(flat_byte);
        if pos.run_index < self.runs.len() {
            self.runs[pos.run_index]
                .text
                .insert_str(pos.byte_offset, text);
        } else if let Some(last) = self.runs.last_mut() {
            last.text.push_str(text);
        }
    }

    /// Delete text in the range `[start_byte..end_byte)` (flat byte offsets).
    pub fn delete_range(&mut self, start_byte: usize, end_byte: usize) {
        if start_byte >= end_byte {
            return;
        }
        // Walk runs and delete the overlapping portion from each
        let mut offset = 0usize;
        let mut i = 0;
        while i < self.runs.len() {
            let run_len = self.runs[i].text.len();
            let run_start = offset;
            let run_end = offset + run_len;

            if run_end <= start_byte || run_start >= end_byte {
                // No overlap
                offset = run_end;
                i += 1;
                continue;
            }

            let del_start = start_byte.max(run_start) - run_start;
            let del_end = end_byte.min(run_end) - run_start;

            // Remove the range from this run's text
            let before = &self.runs[i].text[..del_start];
            let after = &self.runs[i].text[del_end..];
            self.runs[i].text = format!("{}{}", before, after);

            offset = run_start + self.runs[i].text.len();
            i += 1;
        }
        self.merge_adjacent_runs();
    }

    /// Find the override for a specific glyph index, if any.
    pub fn get_glyph_override(&self, glyph_index: usize) -> Option<&GlyphOverride> {
        self.glyph_overrides
            .iter()
            .find(|o| o.glyph_index == glyph_index)
    }

    /// Get or insert a mutable override for a glyph index.
    pub fn ensure_glyph_override(&mut self, glyph_index: usize) -> &mut GlyphOverride {
        if let Some(pos) = self
            .glyph_overrides
            .iter()
            .position(|o| o.glyph_index == glyph_index)
        {
            &mut self.glyph_overrides[pos]
        } else {
            self.glyph_overrides.push(GlyphOverride {
                glyph_index,
                ..Default::default()
            });
            self.glyph_overrides.last_mut().unwrap()
        }
    }

    /// Remove identity overrides (cleanup after editing).
    pub fn cleanup_glyph_overrides(&mut self) {
        self.glyph_overrides.retain(|o| !o.is_identity());
    }

    /// Clear all glyph overrides (reset to default positioning).
    pub fn clear_glyph_overrides(&mut self) {
        self.glyph_overrides.clear();
    }

    /// Returns true if any glyph overrides are present.
    pub fn has_glyph_overrides(&self) -> bool {
        !self.glyph_overrides.is_empty()
    }
}

impl PartialEq for TextStyle {
    fn eq(&self, other: &Self) -> bool {
        self.font_family == other.font_family
            && self.font_weight == other.font_weight
            && self.font_size.to_bits() == other.font_size.to_bits()
            && self.italic == other.italic
            && self.underline == other.underline
            && self.strikethrough == other.strikethrough
            && self.color == other.color
            && self.letter_spacing.to_bits() == other.letter_spacing.to_bits()
            && self.baseline_offset.to_bits() == other.baseline_offset.to_bits()
    }
}

impl Eq for TextStyle {}

// ---------------------------------------------------------------------------
// TextSelection helpers
// ---------------------------------------------------------------------------

impl TextSelection {
    /// Whether there is an active selection (anchor != cursor).
    pub fn has_selection(&self) -> bool {
        self.anchor != self.cursor
    }

    /// Get the ordered (start, end) flat byte offsets within a block.
    pub fn ordered_flat_offsets(&self, block: &TextBlock) -> (usize, usize) {
        let a = block.run_pos_to_flat_offset(self.anchor);
        let c = block.run_pos_to_flat_offset(self.cursor);
        if a <= c { (a, c) } else { (c, a) }
    }

    /// Collapse selection to cursor position (deselect).
    pub fn collapse_to_cursor(&mut self) {
        self.anchor = self.cursor;
    }
}

impl RunPosition {
    /// Compare two positions using flat byte offsets.
    pub fn cmp_in(&self, other: &RunPosition, block: &TextBlock) -> std::cmp::Ordering {
        let a = block.run_pos_to_flat_offset(*self);
        let b = block.run_pos_to_flat_offset(*other);
        a.cmp(&b)
    }
}

// ---------------------------------------------------------------------------
// Multi-run rasterization (Batch 3)
// ---------------------------------------------------------------------------

/// Compute the rotation pivot for a `TextBlock` in canvas-pixel coordinates.
/// This must match the pivot used by the UI overlay so that the rendered text
/// and the overlay handles rotate around the same point.
fn block_rotation_pivot(block: &TextBlock, layout: &BlockLayout) -> (f32, f32) {
    let display_w = block.max_width.unwrap_or(layout.total_width).max(1.0);
    let display_h = block
        .max_height
        .map(|mh| mh.max(layout.total_height))
        .unwrap_or(layout.total_height)
        .max(1.0);
    (
        block.position[0] + display_w * 0.5,
        block.position[1] + display_h * 0.5,
    )
}

/// Optionally rotate an RGBA buffer by `rotation` radians and then blit the result
/// onto the target `TiledImage`.  If `rotation` is near-zero, blits directly.
///
/// `pivot_canvas` — if `Some((px, py))`, rotation happens around that point
/// (in canvas pixel coordinates); the buffer-local pivot is derived from
/// `(px - off_x, py - off_y)`.  If `None`, the buffer center is used.
fn maybe_rotate_and_blit(
    target: &mut TiledImage,
    buf: &[u8],
    buf_w: u32,
    buf_h: u32,
    off_x: i32,
    off_y: i32,
    rotation: f32,
    canvas_w: u32,
    canvas_h: u32,
    pivot_canvas: Option<(f32, f32)>,
) {
    if rotation.abs() > 0.001 {
        let local_pivot = pivot_canvas.map(|(px, py)| (px - off_x as f32, py - off_y as f32));
        let (rotated, rw, rh, rx_off, ry_off) =
            rotate_glyph_buffer(buf, buf_w, buf_h, rotation, local_pivot);
        blit_rgba_buffer(
            target,
            &rotated,
            rw,
            rh,
            off_x + rx_off,
            off_y + ry_off,
            canvas_w,
            canvas_h,
        );
    } else {
        blit_rgba_buffer(target, buf, buf_w, buf_h, off_x, off_y, canvas_w, canvas_h);
    }
}

/// Public wrapper to rasterize a single block (by mutable ref) into a target TiledImage.
/// Used by the text layer drag optimization to extract a single block's pixels.
pub fn rasterize_single_block(
    block: &mut TextBlock,
    content_generation: u64,
    canvas_w: u32,
    canvas_h: u32,
    target: &mut crate::canvas::TiledImage,
    coverage_buf: &mut Vec<f32>,
    glyph_cache: &mut GlyphPixelCache,
) {
    rasterize_block_multirun(
        block,
        content_generation,
        canvas_w,
        canvas_h,
        target,
        coverage_buf,
        glyph_cache,
    );
}

/// Find the tight AABB of non-transparent pixels in an RGBA buffer.
/// Returns (x, y, w, h, cropped_buf). Returns (0, 0, 0, 0, vec![]) if empty.
pub fn find_tight_bounds_rgba(data: &[u8], w: u32, h: u32) -> (u32, u32, u32, u32, Vec<u8>) {
    let (w_us, h_us) = (w as usize, h as usize);
    let mut min_x = w_us;
    let mut min_y = h_us;
    let mut max_x = 0usize;
    let mut max_y = 0usize;
    for y in 0..h_us {
        for x in 0..w_us {
            if data[(y * w_us + x) * 4 + 3] > 0 {
                min_x = min_x.min(x);
                min_y = min_y.min(y);
                max_x = max_x.max(x);
                max_y = max_y.max(y);
            }
        }
    }
    if max_x < min_x || max_y < min_y {
        return (0, 0, 0, 0, Vec::new());
    }
    let tw = max_x - min_x + 1;
    let th = max_y - min_y + 1;
    let mut out = vec![0u8; tw * th * 4];
    for y in 0..th {
        let src_off = ((min_y + y) * w_us + min_x) * 4;
        let dst_off = y * tw * 4;
        out[dst_off..dst_off + tw * 4].copy_from_slice(&data[src_off..src_off + tw * 4]);
    }
    (min_x as u32, min_y as u32, tw as u32, th as u32, out)
}

/// Rasterize a single block with multi-run support.
/// Each run can have a different font/size/weight/color. Runs on the same line
/// share a common baseline (derived from the tallest ascent on that line).
fn rasterize_block_multirun(
    block: &mut TextBlock,
    content_generation: u64,
    canvas_w: u32,
    canvas_h: u32,
    target: &mut TiledImage,
    coverage_buf: &mut Vec<f32>,
    glyph_cache: &mut GlyphPixelCache,
) {
    // If there's only one run, or all runs share the same style, use the simple path
    // for maximum compatibility with the existing rasterizer.
    let all_text: String = block.runs.iter().map(|r| r.text.as_str()).collect();
    if all_text.is_empty() {
        return;
    }

    let has_warp = !matches!(block.warp, TextWarp::None);
    let has_glyph_overrides = block.has_glyph_overrides();
    let has_rotation = block.rotation.abs() > 0.001;

    // Precompute rotation pivot (matches UI overlay center)
    let rot_pivot = if has_rotation {
        let layout = compute_block_layout(&*block);
        Some(block_rotation_pivot(&*block, &layout))
    } else {
        None
    };

    // If glyph overrides exist, use per-glyph rasterization path
    if has_glyph_overrides {
        if has_rotation {
            // Per-glyph into temp, then rotate the whole block result
            let mut temp = TiledImage::new(canvas_w, canvas_h);
            rasterize_block_per_glyph(
                &*block,
                canvas_w,
                canvas_h,
                &mut temp,
                coverage_buf,
                glyph_cache,
            );
            let max_font = block
                .runs
                .iter()
                .map(|r| r.style.font_size)
                .fold(0.0f32, f32::max);
            let margin = (max_font * 2.0) as u32 + 20;
            let bx = (block.position[0] as i32 - margin as i32).max(0) as u32;
            let by = (block.position[1] as i32 - margin as i32).max(0) as u32;
            let bx2 = (block.position[0] as u32 + canvas_w / 2 + margin).min(canvas_w);
            let by2 = (block.position[1] as u32 + canvas_h / 2 + margin).min(canvas_h);
            let bw = bx2.saturating_sub(bx);
            let bh = by2.saturating_sub(by);
            if bw > 0 && bh > 0 {
                let region = temp.extract_region_rgba(bx, by, bw, bh);
                if let Some((tx, ty, tw, th, trimmed)) = trim_to_content(&region, bw, bh) {
                    maybe_rotate_and_blit(
                        target,
                        &trimmed,
                        tw,
                        th,
                        bx as i32 + tx as i32,
                        by as i32 + ty as i32,
                        block.rotation,
                        canvas_w,
                        canvas_h,
                        rot_pivot,
                    );
                }
            }
        } else {
            rasterize_block_per_glyph(
                &*block,
                canvas_w,
                canvas_h,
                target,
                coverage_buf,
                glyph_cache,
            );
        }
        return;
    }

    let single_style =
        block.runs.len() <= 1 || block.runs.windows(2).all(|w| w[0].style == w[1].style);

    if single_style {
        // Fast path: check per-block cache for position-only changes
        if let Some(ref cached) = block.cached_raster
            && cached.content_generation == content_generation
            && cached.buf_w > 0
            && cached.buf_h > 0
        {
            let dx = (block.position[0] - cached.origin[0]).round() as i32;
            let dy = (block.position[1] - cached.origin[1]).round() as i32;
            let adj_off_x = cached.off_x + dx;
            let adj_off_y = cached.off_y + dy;
            if has_warp {
                if let Some((warped, ww, wh, wox, woy)) =
                    apply_block_warp(&cached.buf, cached.buf_w, cached.buf_h, &block.warp)
                {
                    maybe_rotate_and_blit(
                        target,
                        &warped,
                        ww,
                        wh,
                        adj_off_x + wox,
                        adj_off_y + woy,
                        block.rotation,
                        canvas_w,
                        canvas_h,
                        rot_pivot,
                    );
                }
            } else {
                maybe_rotate_and_blit(
                    target,
                    &cached.buf,
                    cached.buf_w,
                    cached.buf_h,
                    adj_off_x,
                    adj_off_y,
                    block.rotation,
                    canvas_w,
                    canvas_h,
                    rot_pivot,
                );
            }
            return;
        }

        // Cache miss: full rasterization
        let style = block.runs.first().map(|r| &r.style).unwrap_or_else(|| {
            static DEFAULT: std::sync::OnceLock<TextStyle> = std::sync::OnceLock::new();
            DEFAULT.get_or_init(TextStyle::default)
        });

        let font = match load_font_for_style(style) {
            Some(f) => f,
            None => return,
        };

        let alignment = match block.paragraph.alignment {
            TextAlignment::Left => text::TextAlignment::Left,
            TextAlignment::Center => text::TextAlignment::Center,
            TextAlignment::Right => text::TextAlignment::Right,
        };

        let rasterized = text::rasterize_text(
            &font,
            &all_text,
            style.font_size,
            alignment,
            block.position[0],
            block.position[1],
            style.color,
            true,
            style.font_weight >= 700,
            style.italic,
            style.underline,
            style.strikethrough,
            canvas_w,
            canvas_h,
            coverage_buf,
            glyph_cache,
            block.max_width,
            style.letter_spacing,
            block.paragraph.line_spacing,
            style.width_scale,
            style.height_scale,
        );

        // Cache the rasterized buffer for fast position-only updates
        if rasterized.buf_w > 0 && rasterized.buf_h > 0 {
            block.cached_raster = Some(CachedBlockRaster {
                buf: rasterized.buf.clone(),
                buf_w: rasterized.buf_w,
                buf_h: rasterized.buf_h,
                off_x: rasterized.off_x,
                off_y: rasterized.off_y,
                origin: block.position,
                content_generation,
            });
        }

        if rasterized.buf_w > 0 && rasterized.buf_h > 0 {
            if has_warp {
                if let Some((warped, ww, wh, wox, woy)) = apply_block_warp(
                    &rasterized.buf,
                    rasterized.buf_w,
                    rasterized.buf_h,
                    &block.warp,
                ) {
                    maybe_rotate_and_blit(
                        target,
                        &warped,
                        ww,
                        wh,
                        rasterized.off_x + wox,
                        rasterized.off_y + woy,
                        block.rotation,
                        canvas_w,
                        canvas_h,
                        rot_pivot,
                    );
                }
            } else {
                maybe_rotate_and_blit(
                    target,
                    &rasterized.buf,
                    rasterized.buf_w,
                    rasterized.buf_h,
                    rasterized.off_x,
                    rasterized.off_y,
                    block.rotation,
                    canvas_w,
                    canvas_h,
                    rot_pivot,
                );
            }
        }
        return;
    }

    // Multi-run path: lay out each run segment with its own font/size,
    // sharing a baseline per line (tallest ascent wins).

    // Check per-block cache for position-only changes (multi-run).
    // The cached buffer stores the tight pre-warp/pre-rotation raster.
    if let Some(ref cached) = block.cached_raster
        && cached.content_generation == content_generation
        && cached.buf_w > 0
        && cached.buf_h > 0
    {
        let dx = (block.position[0] - cached.origin[0]).round() as i32;
        let dy = (block.position[1] - cached.origin[1]).round() as i32;
        let adj_off_x = cached.off_x + dx;
        let adj_off_y = cached.off_y + dy;
        if has_warp {
            if let Some((warped, ww, wh, wox, woy)) =
                apply_block_warp(&cached.buf, cached.buf_w, cached.buf_h, &block.warp)
            {
                maybe_rotate_and_blit(
                    target,
                    &warped,
                    ww,
                    wh,
                    adj_off_x + wox,
                    adj_off_y + woy,
                    block.rotation,
                    canvas_w,
                    canvas_h,
                    rot_pivot,
                );
            }
        } else {
            maybe_rotate_and_blit(
                target,
                &cached.buf,
                cached.buf_w,
                cached.buf_h,
                adj_off_x,
                adj_off_y,
                block.rotation,
                canvas_w,
                canvas_h,
                rot_pivot,
            );
        }
        return;
    }

    // Cache miss — full multi-run rasterization, then cache the tight result.
    let rasterize_multirun_and_extract = |block: &TextBlock,
                                          canvas_w: u32,
                                          canvas_h: u32,
                                          coverage_buf: &mut Vec<f32>,
                                          glyph_cache: &mut GlyphPixelCache|
     -> Option<(Vec<u8>, u32, u32, i32, i32)> {
        let mut temp = TiledImage::new(canvas_w, canvas_h);
        rasterize_block_multirun_slow(
            block,
            canvas_w,
            canvas_h,
            &mut temp,
            coverage_buf,
            glyph_cache,
        );
        let max_font = block
            .runs
            .iter()
            .map(|r| r.style.font_size)
            .fold(0.0f32, f32::max);
        let margin = (max_font * 2.0) as u32 + 20;
        let bx = (block.position[0] as i32 - margin as i32).max(0) as u32;
        let by = (block.position[1] as i32 - margin as i32).max(0) as u32;
        let bx2 = (block.position[0] as u32 + canvas_w / 2 + margin).min(canvas_w);
        let by2 = (block.position[1] as u32 + canvas_h / 2 + margin).min(canvas_h);
        let bw = bx2.saturating_sub(bx);
        let bh = by2.saturating_sub(by);
        if bw == 0 || bh == 0 {
            return None;
        }
        let region = temp.extract_region_rgba(bx, by, bw, bh);
        if let Some((tx, ty, tw, th, trimmed)) = trim_to_content(&region, bw, bh) {
            Some((
                trimmed,
                tw,
                th,
                bx as i32 + tx as i32,
                by as i32 + ty as i32,
            ))
        } else {
            None
        }
    };

    if let Some((buf, tw, th, off_x, off_y)) =
        rasterize_multirun_and_extract(&*block, canvas_w, canvas_h, coverage_buf, glyph_cache)
    {
        // Cache the tight buffer for fast position-only updates
        block.cached_raster = Some(CachedBlockRaster {
            buf: buf.clone(),
            buf_w: tw,
            buf_h: th,
            off_x,
            off_y,
            origin: block.position,
            content_generation,
        });

        if has_warp {
            if let Some((warped, ww, wh, wox, woy)) = apply_block_warp(&buf, tw, th, &block.warp) {
                maybe_rotate_and_blit(
                    target,
                    &warped,
                    ww,
                    wh,
                    off_x + wox,
                    off_y + woy,
                    block.rotation,
                    canvas_w,
                    canvas_h,
                    rot_pivot,
                );
            }
        } else {
            maybe_rotate_and_blit(
                target,
                &buf,
                tw,
                th,
                off_x,
                off_y,
                block.rotation,
                canvas_w,
                canvas_h,
                rot_pivot,
            );
        }
    }
}

/// Segment of text within a single run that belongs to a single line.
struct RunSegment {
    run_index: usize,
    text: String,
    style: TextStyle,
    font: FontArc,
    ascent: f32,
    line_height: f32,
    advance: f32,
}

// ---------------------------------------------------------------------------
// Per-Glyph Rasterization (Phase 5 — Batch 9: glyph overrides)
// ---------------------------------------------------------------------------

/// Rasterize a block with per-glyph overrides.
/// Each glyph is rasterized individually and then transformed (offset, rotated, scaled)
/// according to its GlyphOverride entry before compositing into the target.
fn rasterize_block_per_glyph(
    block: &TextBlock,
    canvas_w: u32,
    canvas_h: u32,
    target: &mut TiledImage,
    coverage_buf: &mut Vec<f32>,
    glyph_cache: &mut GlyphPixelCache,
) {
    // Compute glyph positions (same layout as compute_glyph_bounds)
    let bounds = compute_glyph_bounds(block);
    if bounds.is_empty() {
        return;
    }

    // Rasterize each glyph individually
    for gb in &bounds {
        // Find the style for this glyph:
        // Walk runs to determine which run contains this glyph_index
        let (style, font) = match find_glyph_style(block, gb.glyph_index) {
            Some(sf) => sf,
            None => continue,
        };

        // Get override for this glyph
        let ovr = block.get_glyph_override(gb.glyph_index);
        let offset_x = ovr.map_or(0.0, |o| o.position_offset[0]);
        let offset_y = ovr.map_or(0.0, |o| o.position_offset[1]);
        let rotation = ovr.map_or(0.0, |o| o.rotation);
        let scale = ovr.map_or(1.0, |o| o.scale);

        // Compute the glyph's canvas position with offset
        let glyph_origin_x = gb.x + offset_x;
        let glyph_origin_y = gb.y + offset_y;

        // Rasterize this single character
        let ch_str = gb.ch.to_string();
        let rasterized = text::rasterize_text(
            &font,
            &ch_str,
            style.font_size * scale,
            text::TextAlignment::Left,
            glyph_origin_x,
            glyph_origin_y,
            style.color,
            true,
            style.font_weight >= 700,
            style.italic,
            style.underline,
            style.strikethrough,
            canvas_w,
            canvas_h,
            coverage_buf,
            glyph_cache,
            None,
            style.letter_spacing,
            1.0,
            style.width_scale,
            style.height_scale,
        );

        if rasterized.buf_w == 0 || rasterized.buf_h == 0 {
            continue;
        }

        // If rotation is needed, rotate the glyph buffer around its center
        if rotation.abs() > 0.001 {
            let (rotated, rw, rh, rx_off, ry_off) = rotate_glyph_buffer(
                &rasterized.buf,
                rasterized.buf_w,
                rasterized.buf_h,
                rotation,
                None,
            );
            blit_rgba_buffer(
                target,
                &rotated,
                rw,
                rh,
                rasterized.off_x + rx_off,
                rasterized.off_y + ry_off,
                canvas_w,
                canvas_h,
            );
        } else {
            blit_rgba_buffer(
                target,
                &rasterized.buf,
                rasterized.buf_w,
                rasterized.buf_h,
                rasterized.off_x,
                rasterized.off_y,
                canvas_w,
                canvas_h,
            );
        }
    }
}

/// Find the TextStyle and font for a specific glyph index within a block.
/// Walks the runs to determine which run contains the glyph at the given
/// flat index (newlines excluded from the count).
fn find_glyph_style(block: &TextBlock, glyph_index: usize) -> Option<(TextStyle, FontArc)> {
    let mut idx = 0usize;
    for run in &block.runs {
        for ch in run.text.chars() {
            if ch == '\n' {
                continue;
            }
            if idx == glyph_index {
                let font = load_font_for_style(&run.style)?;
                return Some((run.style.clone(), font));
            }
            idx += 1;
        }
    }
    None
}

/// Rotate an RGBA buffer around its center by the given angle (radians).
/// Returns (rotated_buffer, new_width, new_height, x_offset, y_offset)
/// where offsets are adjustments to the original blit position.
fn rotate_glyph_buffer(
    buf: &[u8],
    w: u32,
    h: u32,
    angle: f32,
    pivot: Option<(f32, f32)>,
) -> (Vec<u8>, u32, u32, i32, i32) {
    let cos_a = angle.cos();
    let sin_a = angle.sin();

    let fw = w as f32;
    let fh = h as f32;

    // Rotation pivot: custom or buffer center
    let (cx, cy) = pivot.unwrap_or((fw * 0.5, fh * 0.5));

    // Compute new bounding box for the rotated image
    let corners = [(0.0f32, 0.0f32), (fw, 0.0), (0.0, fh), (fw, fh)];

    let mut min_x = f32::MAX;
    let mut min_y = f32::MAX;
    let mut max_x = f32::MIN;
    let mut max_y = f32::MIN;

    for &(px, py) in &corners {
        let dx = px - cx;
        let dy = py - cy;
        let rx = dx * cos_a - dy * sin_a + cx;
        let ry = dx * sin_a + dy * cos_a + cy;
        min_x = min_x.min(rx);
        min_y = min_y.min(ry);
        max_x = max_x.max(rx);
        max_y = max_y.max(ry);
    }

    let new_w = (max_x - min_x).ceil() as u32 + 1;
    let new_h = (max_y - min_y).ceil() as u32 + 1;
    // The pivot's location in the output buffer = (cx - min_x, cy - min_y)
    let new_cx = cx - min_x.floor();
    let new_cy = cy - min_y.floor();

    let mut out = vec![0u8; (new_w * new_h * 4) as usize];

    // Inverse transform: for each output pixel, find source pixel
    for oy in 0..new_h {
        for ox in 0..new_w {
            let dx = ox as f32 - new_cx;
            let dy = oy as f32 - new_cy;
            // Rotate back (inverse rotation)
            let sx = dx * cos_a + dy * sin_a + cx;
            let sy = -dx * sin_a + dy * cos_a + cy;

            // Bilinear sample from source
            if sx >= 0.0 && sx < fw && sy >= 0.0 && sy < fh {
                let sx0 = sx.floor() as u32;
                let sy0 = sy.floor() as u32;
                let sx1 = (sx0 + 1).min(w - 1);
                let sy1 = (sy0 + 1).min(h - 1);
                let fx = sx - sx0 as f32;
                let fy = sy - sy0 as f32;

                let sample = |x: u32, y: u32| -> [f32; 4] {
                    let i = (y * w + x) as usize * 4;
                    [
                        buf[i] as f32,
                        buf[i + 1] as f32,
                        buf[i + 2] as f32,
                        buf[i + 3] as f32,
                    ]
                };

                let p00 = sample(sx0, sy0);
                let p10 = sample(sx1, sy0);
                let p01 = sample(sx0, sy1);
                let p11 = sample(sx1, sy1);

                let oi = (oy * new_w + ox) as usize * 4;
                for c in 0..4 {
                    let val = p00[c] * (1.0 - fx) * (1.0 - fy)
                        + p10[c] * fx * (1.0 - fy)
                        + p01[c] * (1.0 - fx) * fy
                        + p11[c] * fx * fy;
                    out[oi + c] = val.round().clamp(0.0, 255.0) as u8;
                }
            }
        }
    }

    let x_off = (min_x.floor()) as i32;
    let y_off = (min_y.floor()) as i32;

    (out, new_w, new_h, x_off, y_off)
}

/// Multi-run rasterization: each run may have a different font/size.
/// We split by newlines first, determine per-line metrics (max ascent, max line height),
/// then rasterize each segment at the correct baseline.
fn rasterize_block_multirun_slow(
    block: &TextBlock,
    canvas_w: u32,
    canvas_h: u32,
    target: &mut TiledImage,
    coverage_buf: &mut Vec<f32>,
    glyph_cache: &mut GlyphPixelCache,
) {
    // Build segments: split each run by '\n' into per-line pieces
    let mut lines: Vec<Vec<RunSegment>> = vec![Vec::new()];

    for (run_idx, run) in block.runs.iter().enumerate() {
        let font = match load_font_for_style(&run.style) {
            Some(f) => f,
            None => continue,
        };
        let scaled = font.as_scaled(ab_glyph::PxScale {
            x: run.style.font_size * run.style.width_scale,
            y: run.style.font_size * run.style.height_scale,
        });

        let parts: Vec<&str> = run.text.split('\n').collect();
        for (pi, part) in parts.iter().enumerate() {
            if pi > 0 {
                // Start a new line
                lines.push(Vec::new());
            }
            if part.is_empty() {
                // Even empty segments contribute to line metrics
                lines.last_mut().unwrap().push(RunSegment {
                    run_index: run_idx,
                    text: String::new(),
                    style: run.style.clone(),
                    font: font.clone(),
                    ascent: scaled.ascent(),
                    line_height: scaled.height(),
                    advance: 0.0,
                });
                continue;
            }
            // Compute advance for this segment
            let mut advance = 0.0f32;
            let mut prev_glyph: Option<ab_glyph::GlyphId> = None;
            for ch in part.chars() {
                let gid = font.glyph_id(ch);
                if let Some(prev) = prev_glyph {
                    advance += scaled.kern(prev, gid);
                    advance += run.style.letter_spacing;
                }
                advance += scaled.h_advance(gid);
                prev_glyph = Some(gid);
            }
            lines.last_mut().unwrap().push(RunSegment {
                run_index: run_idx,
                text: part.to_string(),
                style: run.style.clone(),
                font: font.clone(),
                ascent: scaled.ascent(),
                line_height: scaled.height(),
                advance,
            });
        }
    }

    // Compute per-line metrics
    let line_metrics: Vec<(f32, f32, f32)> = lines
        .iter()
        .map(|segs| {
            let max_ascent = segs.iter().map(|s| s.ascent).fold(0.0f32, f32::max);
            let max_lh = segs.iter().map(|s| s.line_height).fold(0.0f32, f32::max);
            let total_width: f32 = segs.iter().map(|s| s.advance).sum();
            (total_width, max_ascent, max_lh)
        })
        .collect();

    // Rasterize each segment individually
    let mut y_pos = 0.0f32;
    for (line_idx, segs) in lines.iter().enumerate() {
        let (line_width, max_ascent, max_lh) = line_metrics[line_idx];

        // Alignment offset for this line
        let align_off = match block.paragraph.alignment {
            TextAlignment::Left => 0.0,
            TextAlignment::Center => -line_width * 0.5,
            TextAlignment::Right => -line_width,
        };

        let mut x_cursor = align_off;

        for seg in segs {
            if seg.text.is_empty() {
                continue;
            }

            // This segment's baseline y = block.position[1] + y_pos + max_ascent
            // But since rasterize_text takes origin as baseline-start, we pass
            // (block.position[0] + x_cursor, block.position[1] + y_pos)
            let seg_origin_x = block.position[0] + x_cursor;
            let seg_origin_y = block.position[1] + y_pos + (max_ascent - seg.ascent);

            let rasterized = text::rasterize_text(
                &seg.font,
                &seg.text,
                seg.style.font_size,
                text::TextAlignment::Left, // alignment handled by x_cursor
                seg_origin_x,
                seg_origin_y,
                seg.style.color,
                true,
                seg.style.font_weight >= 700,
                seg.style.italic,
                seg.style.underline,
                seg.style.strikethrough,
                canvas_w,
                canvas_h,
                coverage_buf,
                glyph_cache,
                None, // word wrap handled at segment level
                seg.style.letter_spacing,
                block.paragraph.line_spacing,
                seg.style.width_scale,
                seg.style.height_scale,
            );

            if rasterized.buf_w > 0 && rasterized.buf_h > 0 {
                blit_rgba_buffer(
                    target,
                    &rasterized.buf,
                    rasterized.buf_w,
                    rasterized.buf_h,
                    rasterized.off_x,
                    rasterized.off_y,
                    canvas_w,
                    canvas_h,
                );
            }

            x_cursor += seg.advance;
        }

        y_pos += max_lh * block.paragraph.line_spacing;
    }
}

/// Compute layout metrics for a block (for cursor positioning and hit testing).
/// Returns per-run character advances and per-line metrics.
pub fn compute_block_layout(block: &TextBlock) -> BlockLayout {
    let mut run_infos = Vec::new();
    let mut line_infos: Vec<LineInfo> = Vec::new();

    // Split runs by newlines into per-line segments, same as rasterizer
    struct SegInfo {
        run_index: usize,
        text: String,
        ascent: f32,
        line_height: f32,
        char_advances: Vec<f32>,
    }

    let mut lines: Vec<Vec<SegInfo>> = vec![Vec::new()];

    for (run_idx, run) in block.runs.iter().enumerate() {
        let font = match load_font_for_style(&run.style) {
            Some(f) => f,
            None => {
                // Can't load font — provide dummy metrics
                run_infos.push(RunLayoutInfo {
                    char_advances: vec![0.0; run.text.chars().count() + 1],
                    baseline_y: 0.0,
                    line_height: run.style.font_size,
                    ascent: run.style.font_size * 0.8,
                });
                continue;
            }
        };
        let scaled = font.as_scaled(ab_glyph::PxScale {
            x: run.style.font_size * run.style.width_scale,
            y: run.style.font_size * run.style.height_scale,
        });
        let ascent = scaled.ascent();
        let lh = scaled.height();

        let parts: Vec<&str> = run.text.split('\n').collect();
        let mut run_advances = vec![0.0f32]; // entry 0 = 0.0

        for (pi, part) in parts.iter().enumerate() {
            if pi > 0 {
                lines.push(Vec::new());
                run_advances.push(*run_advances.last().unwrap()); // newline doesn't advance x
            }
            let mut seg_advances = vec![0.0f32];
            let mut cursor_x = 0.0f32;
            let mut prev_glyph: Option<ab_glyph::GlyphId> = None;
            for ch in part.chars() {
                let gid = font.glyph_id(ch);
                if let Some(prev) = prev_glyph {
                    cursor_x += scaled.kern(prev, gid);
                    cursor_x += run.style.letter_spacing;
                }
                cursor_x += scaled.h_advance(gid);
                seg_advances.push(cursor_x);
                prev_glyph = Some(gid);
            }
            // Extend the flat run_advances with this segment's advances (offset from line start)
            for &a in seg_advances.iter().skip(1) {
                run_advances.push(
                    *run_advances.last().unwrap_or(&0.0) + a
                        - seg_advances[seg_advances.len().saturating_sub(1)
                            - (seg_advances.len()
                                - 1
                                - (seg_advances.iter().position(|&x| x == a).unwrap_or(0)))]
                        .min(a),
                );
            }

            lines.last_mut().unwrap().push(SegInfo {
                run_index: run_idx,
                text: part.to_string(),
                ascent,
                line_height: lh,
                char_advances: seg_advances,
            });
        }

        // Simplified: just store flat char advances for this run
        let mut flat_advances = vec![0.0f32];
        let mut cx = 0.0f32;
        let mut prev_g: Option<ab_glyph::GlyphId> = None;
        for ch in run.text.chars() {
            if ch == '\n' {
                flat_advances.push(cx);
                prev_g = None;
                continue;
            }
            let gid = font.glyph_id(ch);
            if let Some(prev) = prev_g {
                cx += scaled.kern(prev, gid);
                cx += run.style.letter_spacing;
            }
            cx += scaled.h_advance(gid);
            flat_advances.push(cx);
            prev_g = Some(gid);
        }

        run_infos.push(RunLayoutInfo {
            char_advances: flat_advances,
            baseline_y: 0.0, // filled in below per-line
            line_height: lh,
            ascent,
        });
    }

    // Compute per-line metrics
    let mut total_width = 0.0f32;
    let mut total_height = 0.0f32;
    let ls = block.paragraph.line_spacing;
    for segs in &lines {
        let max_ascent = segs.iter().map(|s| s.ascent).fold(0.0f32, f32::max);
        let max_lh = segs.iter().map(|s| s.line_height).fold(0.0f32, f32::max);
        let width: f32 = segs
            .iter()
            .map(|s| *s.char_advances.last().unwrap_or(&0.0))
            .sum();
        total_width = total_width.max(width);
        total_height += max_lh * ls;
        line_infos.push(LineInfo {
            width,
            max_ascent,
            max_line_height: max_lh * ls,
        });
    }

    // If max_width is set, adjust layout for word wrapping
    if let Some(mw) = block.max_width {
        total_width = mw;
        // For single-run blocks, recompute height with word wrapping
        if block.runs.len() == 1 {
            let run = &block.runs[0];
            if let Some(font) = load_font_for_style(&run.style) {
                let wrapped_lines: Vec<String> = run
                    .text
                    .split('\n')
                    .flat_map(|line| {
                        text::word_wrap_line(
                            line,
                            &font,
                            run.style.font_size,
                            mw,
                            run.style.letter_spacing,
                            run.style.width_scale,
                            run.style.height_scale,
                        )
                    })
                    .collect();
                let scaled = font.as_scaled(ab_glyph::PxScale {
                    x: run.style.font_size * run.style.width_scale,
                    y: run.style.font_size * run.style.height_scale,
                });
                total_height = wrapped_lines.len().max(1) as f32
                    * scaled.height()
                    * block.paragraph.line_spacing;
            }
        }
    }

    BlockLayout {
        runs: run_infos,
        lines: line_infos,
        total_width,
        total_height,
    }
}

// ---------------------------------------------------------------------------
// Per-Glyph Bounding Boxes (Phase 5 — Batch 9: glyph vertex editing)
// ---------------------------------------------------------------------------

/// Bounding box for a single glyph, in canvas coordinates.
#[derive(Clone, Debug)]
pub struct GlyphBounds {
    /// Index of this glyph in the flat sequence (across all runs in a block).
    pub glyph_index: usize,
    /// The character this glyph represents.
    pub ch: char,
    /// Top-left corner in canvas coords.
    pub x: f32,
    pub y: f32,
    /// Width and height of the glyph's bounding box.
    pub w: f32,
    pub h: f32,
    /// Center point (convenience, = x + w/2, y + h/2).
    pub cx: f32,
    pub cy: f32,
}

/// Compute per-glyph bounding boxes for a text block, in canvas coordinates.
/// Newline characters are excluded from the returned list.
/// Respects alignment and multi-run layout.
pub fn compute_glyph_bounds(block: &TextBlock) -> Vec<GlyphBounds> {
    let mut result = Vec::new();
    let mut glyph_idx = 0usize;

    // Split runs into per-line segments (same logic as rasterize_block_multirun_slow)
    struct GlyphSeg {
        text: String,
        font: FontArc,
        font_size: f32,
        width_scale: f32,
        height_scale: f32,
        ascent: f32,
        line_height: f32,
        advance: f32,
    }

    let mut lines: Vec<Vec<GlyphSeg>> = vec![Vec::new()];

    for run in &block.runs {
        let font = match load_font_for_style(&run.style) {
            Some(f) => f,
            None => continue,
        };
        let scaled = font.as_scaled(ab_glyph::PxScale {
            x: run.style.font_size * run.style.width_scale,
            y: run.style.font_size * run.style.height_scale,
        });
        let parts: Vec<&str> = run.text.split('\n').collect();
        for (pi, part) in parts.iter().enumerate() {
            if pi > 0 {
                lines.push(Vec::new());
            }
            let mut advance = 0.0f32;
            let mut prev_glyph: Option<ab_glyph::GlyphId> = None;
            for ch in part.chars() {
                let gid = font.glyph_id(ch);
                if let Some(prev) = prev_glyph {
                    advance += scaled.kern(prev, gid);
                }
                advance += scaled.h_advance(gid);
                prev_glyph = Some(gid);
            }
            lines.last_mut().unwrap().push(GlyphSeg {
                text: part.to_string(),
                font: font.clone(),
                font_size: run.style.font_size,
                width_scale: run.style.width_scale,
                height_scale: run.style.height_scale,
                ascent: scaled.ascent(),
                line_height: scaled.height(),
                advance,
            });
        }
    }

    // Per-line metrics
    let line_metrics: Vec<(f32, f32, f32)> = lines
        .iter()
        .map(|segs| {
            let max_ascent = segs.iter().map(|s| s.ascent).fold(0.0f32, f32::max);
            let max_lh = segs.iter().map(|s| s.line_height).fold(0.0f32, f32::max);
            let total_width: f32 = segs.iter().map(|s| s.advance).sum();
            (total_width, max_ascent, max_lh)
        })
        .collect();

    let mut y_pos = 0.0f32;
    for (line_idx, segs) in lines.iter().enumerate() {
        let (line_width, max_ascent, max_lh) = line_metrics[line_idx];
        let align_off = match block.paragraph.alignment {
            TextAlignment::Left => 0.0,
            TextAlignment::Center => -line_width * 0.5,
            TextAlignment::Right => -line_width,
        };
        let mut x_cursor = align_off;

        for seg in segs {
            let font = &seg.font;
            let scaled = font.as_scaled(ab_glyph::PxScale {
                x: seg.font_size * seg.width_scale,
                y: seg.font_size * seg.height_scale,
            });
            let baseline_y = block.position[1] + y_pos + max_ascent;
            let seg_ascent = seg.ascent;
            let mut prev_glyph: Option<ab_glyph::GlyphId> = None;

            for ch in seg.text.chars() {
                let gid = font.glyph_id(ch);
                if let Some(prev) = prev_glyph {
                    x_cursor += scaled.kern(prev, gid);
                }
                let h_advance = scaled.h_advance(gid);

                // Glyph box: use the tight glyph bounds from the font
                let glyph_x = block.position[0] + x_cursor;
                let glyph_y = baseline_y - seg_ascent;
                let glyph_w = h_advance;
                let glyph_h = seg.line_height;

                result.push(GlyphBounds {
                    glyph_index: glyph_idx,
                    ch,
                    x: glyph_x,
                    y: glyph_y,
                    w: glyph_w,
                    h: glyph_h,
                    cx: glyph_x + glyph_w * 0.5,
                    cy: glyph_y + glyph_h * 0.5,
                });

                x_cursor += h_advance;
                prev_glyph = Some(gid);
                glyph_idx += 1;
            }
        }

        y_pos += max_lh;
    }

    result
}

/// Hit-test a point (in canvas coordinates) against all blocks in a TextLayerData.
/// Returns the block index if the point falls within a block's bounding box.
pub fn hit_test_blocks(data: &TextLayerData, x: f32, y: f32) -> Option<usize> {
    for (idx, block) in data.blocks.iter().enumerate() {
        let layout = compute_block_layout(block);
        let bx = block.position[0];
        let by = block.position[1];
        // Use max_width for box width if set, otherwise use natural text width
        let box_w = block.max_width.unwrap_or(layout.total_width);
        let box_h = if let Some(mh) = block.max_height {
            mh.max(layout.total_height)
        } else {
            layout.total_height
        };
        let margin = 10.0;
        if x >= bx - margin
            && x <= bx + box_w + margin
            && y >= by - margin
            && y <= by + box_h + margin
        {
            return Some(idx);
        }
    }
    None
}

/// Load a font from the system matching the given TextStyle.
fn load_font_for_style(style: &TextStyle) -> Option<FontArc> {
    text::load_system_font(&style.font_family, style.font_weight, style.italic)
}

/// Trim an RGBA buffer to the smallest rectangle containing non-transparent pixels.
/// Returns `(x, y, w, h, trimmed_data)` or `None` if fully transparent.
fn trim_to_content(data: &[u8], w: u32, h: u32) -> Option<(u32, u32, u32, u32, Vec<u8>)> {
    let (w_us, h_us) = (w as usize, h as usize);
    let mut min_x = w_us;
    let mut min_y = h_us;
    let mut max_x = 0usize;
    let mut max_y = 0usize;
    for y in 0..h_us {
        for x in 0..w_us {
            let a = data[(y * w_us + x) * 4 + 3];
            if a > 0 {
                min_x = min_x.min(x);
                min_y = min_y.min(y);
                max_x = max_x.max(x);
                max_y = max_y.max(y);
            }
        }
    }
    if max_x < min_x || max_y < min_y {
        return None;
    }
    let tw = max_x - min_x + 1;
    let th = max_y - min_y + 1;
    let mut out = vec![0u8; tw * th * 4];
    for y in 0..th {
        let src_off = ((min_y + y) * w_us + min_x) * 4;
        let dst_off = y * tw * 4;
        out[dst_off..dst_off + tw * 4].copy_from_slice(&data[src_off..src_off + tw * 4]);
    }
    Some((min_x as u32, min_y as u32, tw as u32, th as u32, out))
}

/// Blit an RGBA buffer onto a TiledImage using alpha compositing.
fn blit_rgba_buffer(
    target: &mut TiledImage,
    buf: &[u8],
    buf_w: u32,
    buf_h: u32,
    off_x: i32,
    off_y: i32,
    canvas_w: u32,
    canvas_h: u32,
) {
    for py in 0..buf_h {
        let cy = off_y + py as i32;
        if cy < 0 || cy >= canvas_h as i32 {
            continue;
        }
        for px in 0..buf_w {
            let cx = off_x + px as i32;
            if cx < 0 || cx >= canvas_w as i32 {
                continue;
            }
            let src_idx = (py as usize * buf_w as usize + px as usize) * 4;
            let sa = buf[src_idx + 3];
            if sa == 0 {
                continue;
            }
            let sr = buf[src_idx];
            let sg = buf[src_idx + 1];
            let sb = buf[src_idx + 2];
            target.put_pixel(cx as u32, cy as u32, Rgba([sr, sg, sb, sa]));
        }
    }
}

// ===========================================================================
// Text Warp Application (Phase 4 — Batches 7+8)
// ===========================================================================

/// Apply a geometric warp to a rasterized text block.
/// Takes the flat-rasterized block RGBA and returns a warped version.
/// The warp is applied as a pixel-level displacement: for each output pixel,
/// compute where it came from in the source, and bilinear-sample.
fn apply_block_warp(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    warp: &TextWarp,
) -> Option<(Vec<u8>, u32, u32, i32, i32)> {
    match warp {
        TextWarp::None => None,
        TextWarp::Arc(arc) => Some(apply_arc_warp(src, src_w, src_h, arc)),
        TextWarp::Circular(circ) => Some(apply_circular_warp(src, src_w, src_h, circ)),
        TextWarp::PathFollow(pf) => Some(apply_path_follow_warp(src, src_w, src_h, pf)),
        TextWarp::Envelope(env) => Some(apply_envelope_warp(src, src_w, src_h, env)),
    }
}

/// Arc warp: bend text along an arc.
/// bend > 0 curves upward (convex), bend < 0 curves downward (concave).
fn apply_arc_warp(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    arc: &ArcWarp,
) -> (Vec<u8>, u32, u32, i32, i32) {
    if arc.bend.abs() < 0.001 {
        return (src.to_vec(), src_w, src_h, 0, 0);
    }

    let w = src_w as f32;
    let h = src_h as f32;

    // The arc angle spans from bend * PI (at full bend = semicircle)
    let angle = arc.bend * std::f32::consts::PI;
    let radius = if angle.abs() > 0.01 {
        w / (2.0 * (angle / 2.0).sin())
    } else {
        w * 100.0 // effectively flat
    };

    // Compute output bounds by sampling corners and midpoints
    let mut min_x = f32::MAX;
    let mut max_x = f32::MIN;
    let mut min_y = f32::MAX;
    let mut max_y = f32::MIN;

    let samples = 32;
    for i in 0..=samples {
        let t = i as f32 / samples as f32;
        let sx = t * w;
        for sy in [0.0, h] {
            let (dx, dy) = arc_map_point(sx, sy, w, h, radius, angle, arc);
            min_x = min_x.min(dx);
            max_x = max_x.max(dx);
            min_y = min_y.min(dy);
            max_y = max_y.max(dy);
        }
    }

    let margin = 2.0;
    min_x -= margin;
    min_y -= margin;
    max_x += margin;
    max_y += margin;

    let out_w = (max_x - min_x).ceil() as u32;
    let out_h = (max_y - min_y).ceil() as u32;
    if out_w == 0 || out_h == 0 || out_w > 8192 || out_h > 8192 {
        return (src.to_vec(), src_w, src_h, 0, 0);
    }

    let off_x = min_x.floor() as i32;
    let off_y = min_y.floor() as i32;

    let mut out = vec![0u8; (out_w * out_h * 4) as usize];
    let sw = src_w as usize;

    out.par_chunks_mut((out_w * 4) as usize)
        .enumerate()
        .for_each(|(oy, row)| {
            for ox in 0..out_w as usize {
                let dx = ox as f32 + min_x;
                let dy = oy as f32 + min_y;
                // Inverse map: find source pixel for this output pixel
                if let Some((sx, sy)) = arc_inverse_map(dx, dy, w, h, radius, angle, arc)
                    && sx >= 0.0
                    && sx < w
                    && sy >= 0.0
                    && sy < h
                {
                    let pixel = bilinear_sample(src, sw, src_h as usize, sx, sy);
                    let pi = ox * 4;
                    row[pi] = pixel[0];
                    row[pi + 1] = pixel[1];
                    row[pi + 2] = pixel[2];
                    row[pi + 3] = pixel[3];
                }
            }
        });

    (out, out_w, out_h, off_x, off_y)
}

/// Forward map: source(sx,sy) -> destination(dx,dy) for arc warp.
fn arc_map_point(
    sx: f32,
    sy: f32,
    w: f32,
    h: f32,
    radius: f32,
    angle: f32,
    arc: &ArcWarp,
) -> (f32, f32) {
    let cx = w / 2.0;

    // Normalized position along text width [-1, 1]
    let t = (sx - cx) / (w / 2.0);

    // Angle for this position
    let theta = t * angle / 2.0;

    // Center of curvature is below (positive bend) or above (negative bend)
    let r_sign = if angle > 0.0 { 1.0 } else { -1.0 };
    let r_abs = radius.abs();

    // Distance from baseline (sy=0 at top, h at bottom)
    let sy_norm = sy / h; // 0..1

    // Radial position: baseline at radius, top of text further from center
    let r = r_abs - (1.0 - sy_norm) * h * r_sign;

    let dx = cx + r * theta.sin();
    let dy = r_abs - r * theta.cos() * r_sign;

    // Apply distortion
    let hdist = arc.horizontal_distortion;
    let vdist = arc.vertical_distortion;
    let dx = dx + (dx - cx) * hdist;
    let dy = dy + (dy - h / 2.0) * vdist;

    (dx, dy)
}

/// Inverse map: destination(dx,dy) -> source(sx,sy) for arc warp.
fn arc_inverse_map(
    dx: f32,
    dy: f32,
    w: f32,
    h: f32,
    radius: f32,
    angle: f32,
    arc: &ArcWarp,
) -> Option<(f32, f32)> {
    let cx = w / 2.0;
    let r_sign = if angle > 0.0 { 1.0 } else { -1.0 };
    let r_abs = radius.abs();

    // Undo distortion
    let hdist = arc.horizontal_distortion;
    let vdist = arc.vertical_distortion;
    let dx = if hdist.abs() > 0.001 {
        cx + (dx - cx) / (1.0 + hdist)
    } else {
        dx
    };
    let dy = if vdist.abs() > 0.001 {
        h / 2.0 + (dy - h / 2.0) / (1.0 + vdist)
    } else {
        dy
    };

    // Compute polar coordinates relative to center of curvature
    let rel_x = dx - cx;
    let rel_y = r_abs - dy * r_sign;

    let r = (rel_x * rel_x + (rel_y * r_sign) * (rel_y * r_sign)).sqrt();
    let theta = rel_x.atan2(rel_y * r_sign);

    // Check if the angle is within the arc range
    if angle.abs() > 0.01 && theta.abs() > (angle / 2.0).abs() + 0.1 {
        return None;
    }

    // Compute source x from angle
    let t = if angle.abs() > 0.01 {
        theta / (angle / 2.0)
    } else {
        (dx - cx) / (w / 2.0)
    };
    let sx = cx + t * w / 2.0;

    // Compute source y from radial distance
    let sy_norm = 1.0 - (r_abs - r) / (h * r_sign);
    let sy = sy_norm * h;

    Some((sx, sy))
}

/// Circular warp: arrange text around a circle.
fn apply_circular_warp(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    circ: &CircularWarp,
) -> (Vec<u8>, u32, u32, i32, i32) {
    let w = src_w as f32;
    let h = src_h as f32;
    let r = circ.radius.max(10.0);

    // Total angle subtended by the text
    let _total_angle = w / r;
    let dir = if circ.clockwise { 1.0 } else { -1.0 };

    // Output bounds: circle bounding box with margin
    let r_outer = r + h;
    let out_size = (r_outer * 2.0 + 4.0).ceil() as u32;
    let out_cx = out_size as f32 / 2.0;
    let out_cy = out_size as f32 / 2.0;

    // Offset relative to source buffer: center circle output on source center
    // (same pattern as arc warp: src_center - out_center)
    let off_x = (w / 2.0 - out_cx).round() as i32;
    let off_y = (h / 2.0 - out_cy).round() as i32;

    let mut out = vec![0u8; (out_size * out_size * 4) as usize];
    let sw = src_w as usize;

    out.par_chunks_mut((out_size * 4) as usize)
        .enumerate()
        .for_each(|(oy, row)| {
            for ox in 0..out_size as usize {
                let px = ox as f32 - out_cx;
                let py = oy as f32 - out_cy;
                let dist = (px * px + py * py).sqrt();

                // Check if within the annular ring
                if dist < r || dist > r_outer {
                    continue;
                }

                // Angle of this pixel
                let pixel_angle = py.atan2(px);
                // Relative angle from start angle
                let mut rel_angle = (pixel_angle - circ.start_angle) * dir;
                // Normalize to [0, 2*PI)
                while rel_angle < 0.0 {
                    rel_angle += std::f32::consts::TAU;
                }
                while rel_angle >= std::f32::consts::TAU {
                    rel_angle -= std::f32::consts::TAU;
                }

                // Map angle to source x
                let sx = rel_angle * r;
                if sx < 0.0 || sx >= w {
                    continue;
                }

                // Map radial distance to source y (outer = top of text)
                let sy = r_outer - dist;
                if sy < 0.0 || sy >= h {
                    continue;
                }

                let pixel = bilinear_sample(src, sw, src_h as usize, sx, sy);
                let pi = ox * 4;
                row[pi] = pixel[0];
                row[pi + 1] = pixel[1];
                row[pi + 2] = pixel[2];
                row[pi + 3] = pixel[3];
            }
        });

    (out, out_size, out_size, off_x, off_y)
}

/// Path follow warp: place text along a Bézier curve.
fn apply_path_follow_warp(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    pf: &PathFollowWarp,
) -> (Vec<u8>, u32, u32, i32, i32) {
    if pf.control_points.len() < 4 {
        return (src.to_vec(), src_w, src_h, 0, 0);
    }

    let w = src_w as f32;
    let h = src_h as f32;

    // Build arc-length table for the cubic Bézier
    let lut_size = 256;
    let (arc_lengths, total_arc_len) = build_arc_length_table(&pf.control_points[0..4], lut_size);

    // Compute output bounds by sampling along the path
    let mut min_x = f32::MAX;
    let mut max_x = f32::MIN;
    let mut min_y = f32::MAX;
    let mut max_y = f32::MIN;

    let samples = 64;
    for i in 0..=samples {
        let s = (i as f32 / samples as f32) * total_arc_len;
        let t = arc_length_to_t(s, &arc_lengths, total_arc_len);
        let (px, py) = eval_cubic_bezier(&pf.control_points[0..4], t);
        let (_, ny) = eval_cubic_bezier_tangent(&pf.control_points[0..4], t);
        // Extend bounds by text height in the normal direction
        for offset in [-h, 0.0, h] {
            min_x = min_x.min(px - offset.abs());
            max_x = max_x.max(px + offset.abs());
            min_y = min_y.min(py + offset + ny.abs() * h);
            max_y = max_y.max(py - offset - ny.abs() * h);
        }
    }

    let margin = h + 10.0;
    min_x -= margin;
    min_y -= margin;
    max_x += margin;
    max_y += margin;

    let out_w = ((max_x - min_x).ceil() as u32).min(4096);
    let out_h = ((max_y - min_y).ceil() as u32).min(4096);
    if out_w == 0 || out_h == 0 {
        return (src.to_vec(), src_w, src_h, 0, 0);
    }

    let off_x = min_x.floor() as i32;
    let off_y = min_y.floor() as i32;

    let sw = src_w as usize;
    let mut out = vec![0u8; (out_w * out_h * 4) as usize];

    // For each output pixel, find the closest point on the path,
    // compute distance along path (= source x) and perpendicular distance (= source y)
    out.par_chunks_mut((out_w * 4) as usize)
        .enumerate()
        .for_each(|(oy, row)| {
            for ox in 0..out_w as usize {
                let px = ox as f32 + min_x;
                let py = oy as f32 + min_y;

                // Find closest t on the curve using iterative refinement
                if let Some((sx, sy)) = inverse_path_follow(
                    px,
                    py,
                    &pf.control_points[0..4],
                    &arc_lengths,
                    total_arc_len,
                    h,
                ) && sx >= 0.0
                    && sx < w
                    && sy >= 0.0
                    && sy < h
                {
                    let pixel = bilinear_sample(src, sw, src_h as usize, sx, sy);
                    let pi = ox * 4;
                    row[pi] = pixel[0];
                    row[pi + 1] = pixel[1];
                    row[pi + 2] = pixel[2];
                    row[pi + 3] = pixel[3];
                }
            }
        });

    (out, out_w, out_h, off_x, off_y)
}

/// Envelope warp: deform text between two Bézier curves.
fn apply_envelope_warp(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    env: &EnvelopeWarp,
) -> (Vec<u8>, u32, u32, i32, i32) {
    if env.top_curve.len() < 4 || env.bottom_curve.len() < 4 {
        return (src.to_vec(), src_w, src_h, 0, 0);
    }

    let w = src_w as f32;
    let h = src_h as f32;

    // Compute output bounds from both curves
    let mut min_x = f32::MAX;
    let mut max_x = f32::MIN;
    let mut min_y = f32::MAX;
    let mut max_y = f32::MIN;

    let samples = 64;
    for i in 0..=samples {
        let t = i as f32 / samples as f32;
        let (tx, ty) = eval_cubic_bezier(&env.top_curve[0..4], t);
        let (bx, by) = eval_cubic_bezier(&env.bottom_curve[0..4], t);
        min_x = min_x.min(tx).min(bx);
        max_x = max_x.max(tx).max(bx);
        min_y = min_y.min(ty).min(by);
        max_y = max_y.max(ty).max(by);
    }

    let margin = 4.0;
    min_x -= margin;
    min_y -= margin;
    max_x += margin;
    max_y += margin;

    let out_w = ((max_x - min_x).ceil() as u32).min(4096);
    let out_h = ((max_y - min_y).ceil() as u32).min(4096);
    if out_w == 0 || out_h == 0 {
        return (src.to_vec(), src_w, src_h, 0, 0);
    }

    let off_x = min_x.floor() as i32;
    let off_y = min_y.floor() as i32;

    let sw = src_w as usize;
    let mut out = vec![0u8; (out_w * out_h * 4) as usize];

    out.par_chunks_mut((out_w * 4) as usize)
        .enumerate()
        .for_each(|(oy, row)| {
            for ox in 0..out_w as usize {
                let px = ox as f32 + min_x;
                let py = oy as f32 + min_y;

                // Find t such that px is between top_curve(t).x and bottom_curve(t).x
                // Simple approach: use normalized x position as t
                let t = (px - min_x) / (max_x - min_x - 2.0 * margin).max(1.0);
                if !(0.0..=1.0).contains(&t) {
                    continue;
                }

                let (_, top_y) = eval_cubic_bezier(&env.top_curve[0..4], t);
                let (_, bot_y) = eval_cubic_bezier(&env.bottom_curve[0..4], t);

                let span = bot_y - top_y;
                if span.abs() < 0.001 {
                    continue;
                }

                // Vertical interpolation: where does py fall between top and bottom?
                let v = (py - top_y) / span;
                if !(0.0..=1.0).contains(&v) {
                    continue;
                }

                // Map to source coordinates
                let sx = t * w;
                let sy = v * h;

                if sx >= 0.0 && sx < w && sy >= 0.0 && sy < h {
                    let pixel = bilinear_sample(src, sw, src_h as usize, sx, sy);
                    let pi = ox * 4;
                    row[pi] = pixel[0];
                    row[pi + 1] = pixel[1];
                    row[pi + 2] = pixel[2];
                    row[pi + 3] = pixel[3];
                }
            }
        });

    (out, out_w, out_h, off_x, off_y)
}

// ---------------------------------------------------------------------------
// Bézier curve helpers
// ---------------------------------------------------------------------------

/// Evaluate a cubic Bézier at parameter t.
fn eval_cubic_bezier(pts: &[[f32; 2]], t: f32) -> (f32, f32) {
    let u = 1.0 - t;
    let u2 = u * u;
    let t2 = t * t;
    let x = u2 * u * pts[0][0]
        + 3.0 * u2 * t * pts[1][0]
        + 3.0 * u * t2 * pts[2][0]
        + t2 * t * pts[3][0];
    let y = u2 * u * pts[0][1]
        + 3.0 * u2 * t * pts[1][1]
        + 3.0 * u * t2 * pts[2][1]
        + t2 * t * pts[3][1];
    (x, y)
}

/// Evaluate the tangent (first derivative) of a cubic Bézier at parameter t.
fn eval_cubic_bezier_tangent(pts: &[[f32; 2]], t: f32) -> (f32, f32) {
    let u = 1.0 - t;
    let dx = 3.0 * u * u * (pts[1][0] - pts[0][0])
        + 6.0 * u * t * (pts[2][0] - pts[1][0])
        + 3.0 * t * t * (pts[3][0] - pts[2][0]);
    let dy = 3.0 * u * u * (pts[1][1] - pts[0][1])
        + 6.0 * u * t * (pts[2][1] - pts[1][1])
        + 3.0 * t * t * (pts[3][1] - pts[2][1]);
    (dx, dy)
}

/// Build an arc-length lookup table for a cubic Bézier curve.
/// Returns (cumulative_lengths, total_length).
fn build_arc_length_table(pts: &[[f32; 2]], steps: usize) -> (Vec<f32>, f32) {
    let mut lengths = Vec::with_capacity(steps + 1);
    lengths.push(0.0);
    let mut total = 0.0;
    let (mut prev_x, mut prev_y) = eval_cubic_bezier(pts, 0.0);
    for i in 1..=steps {
        let t = i as f32 / steps as f32;
        let (x, y) = eval_cubic_bezier(pts, t);
        let dx = x - prev_x;
        let dy = y - prev_y;
        total += (dx * dx + dy * dy).sqrt();
        lengths.push(total);
        prev_x = x;
        prev_y = y;
    }
    (lengths, total)
}

/// Convert an arc-length distance to a Bézier parameter t using the LUT.
fn arc_length_to_t(s: f32, lengths: &[f32], total: f32) -> f32 {
    if s <= 0.0 {
        return 0.0;
    }
    if s >= total {
        return 1.0;
    }
    // Binary search
    let n = lengths.len() - 1;
    let mut lo = 0usize;
    let mut hi = n;
    while lo < hi {
        let mid = (lo + hi) / 2;
        if lengths[mid] < s {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    if lo == 0 {
        return 0.0;
    }
    let seg_len = lengths[lo] - lengths[lo - 1];
    let frac = if seg_len > 0.0 {
        (s - lengths[lo - 1]) / seg_len
    } else {
        0.0
    };
    ((lo - 1) as f32 + frac) / n as f32
}

/// Inverse path-follow mapping: given a point (px, py), find the corresponding
/// source coordinates (sx, sy) where sx = arc-length along curve, sy = perpendicular distance.
fn inverse_path_follow(
    px: f32,
    py: f32,
    pts: &[[f32; 2]],
    arc_lengths: &[f32],
    total_arc_len: f32,
    text_height: f32,
) -> Option<(f32, f32)> {
    // Coarse search: find closest t
    let coarse_steps = 64;
    let mut best_t = 0.0f32;
    let mut best_dist_sq = f32::MAX;

    for i in 0..=coarse_steps {
        let t = i as f32 / coarse_steps as f32;
        let (cx, cy) = eval_cubic_bezier(pts, t);
        let dist_sq = (px - cx) * (px - cx) + (py - cy) * (py - cy);
        if dist_sq < best_dist_sq {
            best_dist_sq = dist_sq;
            best_t = t;
        }
    }

    // Fine refinement around the best coarse t
    let step = 1.0 / coarse_steps as f32;
    let mut t_lo = (best_t - step).max(0.0);
    let mut t_hi = (best_t + step).min(1.0);
    for _ in 0..8 {
        let t_mid = (t_lo + t_hi) / 2.0;
        let t_a = (t_lo + t_mid) / 2.0;
        let t_b = (t_mid + t_hi) / 2.0;
        let (ax, ay) = eval_cubic_bezier(pts, t_a);
        let (bx, by) = eval_cubic_bezier(pts, t_b);
        let da = (px - ax) * (px - ax) + (py - ay) * (py - ay);
        let db = (px - bx) * (px - bx) + (py - by) * (py - by);
        if da < db {
            t_hi = t_mid;
        } else {
            t_lo = t_mid;
        }
    }
    let t = (t_lo + t_hi) / 2.0;

    // Get curve point and tangent
    let (cx, cy) = eval_cubic_bezier(pts, t);
    let (tx, ty) = eval_cubic_bezier_tangent(pts, t);
    let tangent_len = (tx * tx + ty * ty).sqrt();
    if tangent_len < 0.0001 {
        return None;
    }

    // Normal vector (perpendicular to tangent, pointing "up" from curve)
    let nx = -ty / tangent_len;
    let ny = tx / tangent_len;

    // Signed perpendicular distance from curve
    let rel_x = px - cx;
    let rel_y = py - cy;
    let perp_dist = rel_x * nx + rel_y * ny;

    // Source x = arc length at this t
    let sx = arc_length_to_t_inverse(t, arc_lengths, total_arc_len);

    // Source y = perpendicular distance mapped to text height
    // perp_dist > 0 means above curve, maps to top of text (sy=0)
    let sy = text_height / 2.0 - perp_dist;

    Some((sx, sy))
}

/// Convert a Bézier t parameter back to arc-length distance.
fn arc_length_to_t_inverse(t: f32, lengths: &[f32], _total: f32) -> f32 {
    let n = lengths.len() - 1;
    let idx_f = t * n as f32;
    let idx = (idx_f as usize).min(n - 1);
    let frac = idx_f - idx as f32;
    lengths[idx] + frac * (lengths[idx + 1] - lengths[idx])
}

/// Bilinear sample from an RGBA buffer.
fn bilinear_sample(src: &[u8], w: usize, h: usize, x: f32, y: f32) -> [u8; 4] {
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;

    let sample = |sx: i32, sy: i32| -> [f32; 4] {
        if sx < 0 || sy < 0 || sx >= w as i32 || sy >= h as i32 {
            return [0.0; 4];
        }
        let idx = (sy as usize * w + sx as usize) * 4;
        [
            src[idx] as f32,
            src[idx + 1] as f32,
            src[idx + 2] as f32,
            src[idx + 3] as f32,
        ]
    };

    let p00 = sample(x0, y0);
    let p10 = sample(x0 + 1, y0);
    let p01 = sample(x0, y0 + 1);
    let p11 = sample(x0 + 1, y0 + 1);

    let mut result = [0u8; 4];
    for c in 0..4 {
        let v = p00[c] * (1.0 - fx) * (1.0 - fy)
            + p10[c] * fx * (1.0 - fy)
            + p01[c] * (1.0 - fx) * fy
            + p11[c] * fx * fy;
        result[c] = v.round().clamp(0.0, 255.0) as u8;
    }
    result
}

// ===========================================================================
// Text Effects Rendering (Batch 5+6)
// ===========================================================================

/// Extract an alpha coverage mask from an RGBA buffer.
/// Returns a Vec<f32> with values in [0.0, 1.0], one per pixel.
fn extract_coverage_mask(rgba: &[u8], w: u32, h: u32) -> Vec<f32> {
    let count = (w as usize) * (h as usize);
    let mut mask = vec![0.0f32; count];
    mask.par_chunks_mut(w as usize)
        .enumerate()
        .for_each(|(y, row)| {
            let row_off = y * w as usize * 4;
            for x in 0..w as usize {
                let a = rgba[row_off + x * 4 + 3];
                row[x] = a as f32 / 255.0;
            }
        });
    mask
}

/// Apply all enabled text effects to a base text RGBA buffer.
/// Returns a new RGBA buffer with effects composited.
fn apply_text_effects(text_rgba: &[u8], w: u32, h: u32, effects: &TextEffects) -> Vec<u8> {
    let count = (w as usize) * (h as usize);
    let coverage = extract_coverage_mask(text_rgba, w, h);

    // Start with a transparent output buffer
    let mut output = vec![0u8; count * 4];

    // 1. Shadow (behind everything)
    if let Some(ref shadow) = effects.shadow {
        render_shadow(&coverage, w, h, shadow, &mut output);
    }

    // 2. Outline (behind filled text, in front of shadow)
    if let Some(ref outline) = effects.outline
        && (outline.position == OutlinePosition::Outside
            || outline.position == OutlinePosition::Center)
    {
        render_outline(&coverage, w, h, outline, &mut output);
    }

    // 3. Text fill (flat color, gradient, or texture)
    if let Some(ref gradient) = effects.gradient_fill {
        render_gradient_fill(&coverage, w, h, gradient, &mut output);
    } else if let Some(ref tex) = effects.texture_fill {
        render_texture_fill(text_rgba, &coverage, w, h, tex, &mut output);
    } else {
        // Normal text fill — composite text onto output
        composite_over(text_rgba, &mut output, count);
    }

    // 4. Inside outline (on top of text fill)
    if let Some(ref outline) = effects.outline
        && outline.position == OutlinePosition::Inside
    {
        render_outline_inside(&coverage, w, h, outline, &mut output);
    }

    // 5. Inner shadow (on top of everything, clipped to text shape)
    if let Some(ref inner) = effects.inner_shadow {
        render_inner_shadow(&coverage, w, h, inner, &mut output);
    }

    output
}

/// Composite src over dst (premultiplied-aware alpha blending on straight-alpha buffers).
fn composite_over(src: &[u8], dst: &mut [u8], pixel_count: usize) {
    for i in 0..pixel_count {
        let si = i * 4;
        let sa = src[si + 3] as u32;
        if sa == 0 {
            continue;
        }
        if sa == 255 {
            dst[si..si + 4].copy_from_slice(&src[si..si + 4]);
            continue;
        }
        let da = dst[si + 3] as u32;
        let inv_sa = 255 - sa;
        let out_a = sa + (da * inv_sa) / 255;
        if out_a == 0 {
            continue;
        }
        for c in 0..3 {
            let sc = src[si + c] as u32;
            let dc = dst[si + c] as u32;
            dst[si + c] = ((sc * sa + dc * da * inv_sa / 255) / out_a).min(255) as u8;
        }
        dst[si + 3] = out_a.min(255) as u8;
    }
}

// ---------------------------------------------------------------------------
// Outline Effect
// ---------------------------------------------------------------------------

/// Compute a distance field from a coverage mask, then render the outline.
/// For Outside: dilated mask minus original = outline ring.
/// For Center: half-dilated in both directions.
fn render_outline(coverage: &[f32], w: u32, h: u32, outline: &OutlineEffect, output: &mut [u8]) {
    let radius = match outline.position {
        OutlinePosition::Outside => outline.width,
        OutlinePosition::Center => outline.width * 0.5,
        OutlinePosition::Inside => return, // handled separately
    };
    if radius <= 0.0 {
        return;
    }

    // Dilate the coverage mask by `radius` using a distance-field approach.
    let dilated = dilate_mask(coverage, w, h, radius);
    let count = (w as usize) * (h as usize);
    let [or, og, ob, oa] = outline.color;

    // Outline = dilated - original coverage (smooth edges)
    for i in 0..count {
        let outline_alpha = (dilated[i] - coverage[i]).clamp(0.0, 1.0) * (oa as f32 / 255.0);
        if outline_alpha < 1.0 / 255.0 {
            continue;
        }
        let sa = (outline_alpha * 255.0).round() as u32;
        let si = i * 4;
        let da = output[si + 3] as u32;
        let inv_sa = 255 - sa;
        let out_a = sa + (da * inv_sa) / 255;
        if out_a == 0 {
            continue;
        }
        for (c, &sc) in [or, og, ob].iter().enumerate() {
            let dc = output[si + c] as u32;
            output[si + c] = ((sc as u32 * sa + dc * da * inv_sa / 255) / out_a).min(255) as u8;
        }
        output[si + 3] = out_a.min(255) as u8;
    }
}

/// Render an inside outline: erode the mask, then the outline is original minus eroded.
fn render_outline_inside(
    coverage: &[f32],
    w: u32,
    h: u32,
    outline: &OutlineEffect,
    output: &mut [u8],
) {
    let radius = match outline.position {
        OutlinePosition::Inside => outline.width,
        OutlinePosition::Center => outline.width * 0.5,
        _ => return,
    };
    if radius <= 0.0 {
        return;
    }

    // Erode the mask: dilate the inverted mask, then invert back
    let count = (w as usize) * (h as usize);
    let inverted: Vec<f32> = coverage.iter().map(|&c| 1.0 - c).collect();
    let dilated_inv = dilate_mask(&inverted, w, h, radius);
    let eroded: Vec<f32> = dilated_inv.iter().map(|&d| (1.0 - d).max(0.0)).collect();

    let [or, og, ob, oa] = outline.color;

    // Inside outline = original - eroded, clipped to text shape
    for i in 0..count {
        let outline_alpha = (coverage[i] - eroded[i]).clamp(0.0, 1.0) * (oa as f32 / 255.0);
        if outline_alpha < 1.0 / 255.0 {
            continue;
        }
        let sa = (outline_alpha * 255.0).round() as u32;
        let si = i * 4;
        let da = output[si + 3] as u32;
        let inv_sa = 255 - sa;
        let out_a = sa + (da * inv_sa) / 255;
        if out_a == 0 {
            continue;
        }
        for (c, &sc) in [or, og, ob].iter().enumerate() {
            let dc = output[si + c] as u32;
            output[si + c] = ((sc as u32 * sa + dc * da * inv_sa / 255) / out_a).min(255) as u8;
        }
        output[si + 3] = out_a.min(255) as u8;
    }
}

/// Dilate a coverage mask by the given radius (in pixels).
/// Uses a fast box-filter approximation (3-pass box blur on the binary mask)
/// which gives a good approximation to a circular dilation for moderate radii.
fn dilate_mask(mask: &[f32], w: u32, h: u32, radius: f32) -> Vec<f32> {
    let ww = w as usize;
    let hh = h as usize;
    let count = ww * hh;

    if radius <= 0.0 {
        return mask.to_vec();
    }

    let int_radius = radius.ceil() as usize;
    if int_radius == 0 {
        return mask.to_vec();
    }

    let r_sq = radius * radius;

    // Euclidean-distance circular dilation: for each pixel, find the maximum
    // coverage value within a circular kernel of the given radius. This
    // preserves the original anti-aliased coverage values, producing smooth
    // outlines without binary thresholding or rectangular artifacts.
    let mut out = vec![0.0f32; count];
    out.par_chunks_mut(ww).enumerate().for_each(|(y, row_out)| {
        let y_start = y.saturating_sub(int_radius);
        let y_end = (y + int_radius + 1).min(hh);
        for (x, out_val) in row_out.iter_mut().enumerate() {
            let x_start = x.saturating_sub(int_radius);
            let x_end = (x + int_radius + 1).min(ww);
            let mut max_val = 0.0f32;
            for sy in y_start..y_end {
                let dy = sy as f32 - y as f32;
                let dy_sq = dy * dy;
                if dy_sq > r_sq {
                    continue;
                }
                let row_off = sy * ww;
                for sx in x_start..x_end {
                    let dx = sx as f32 - x as f32;
                    if dx * dx + dy_sq <= r_sq {
                        max_val = max_val.max(mask[row_off + sx]);
                    }
                }
            }
            *out_val = max_val;
        }
    });

    out
}

// ---------------------------------------------------------------------------
// Shadow Effect
// ---------------------------------------------------------------------------

/// Render a drop shadow: offset the coverage mask, optionally spread (dilate),
/// then apply Gaussian blur, tinted with the shadow color.
fn render_shadow(coverage: &[f32], w: u32, h: u32, shadow: &ShadowEffect, output: &mut [u8]) {
    let ww = w as usize;
    let hh = h as usize;
    let count = ww * hh;

    // Create an offset mask
    let mut offset_mask = vec![0.0f32; count];
    let dx = shadow.offset_x.round() as i32;
    let dy = shadow.offset_y.round() as i32;

    for y in 0..hh {
        let sy = y as i32 - dy;
        if sy < 0 || sy >= hh as i32 {
            continue;
        }
        for x in 0..ww {
            let sx = x as i32 - dx;
            if sx < 0 || sx >= ww as i32 {
                continue;
            }
            offset_mask[y * ww + x] = coverage[sy as usize * ww + sx as usize];
        }
    }

    // Spread (dilate) if needed
    if shadow.spread > 0.5 {
        offset_mask = dilate_mask(&offset_mask, w, h, shadow.spread);
    }

    // Apply Gaussian blur using image crate
    if shadow.blur_radius > 0.5 {
        // Convert mask to RGBA for the blur function
        let [sr, sg, sb, sa] = shadow.color;
        let mut shadow_rgba = vec![0u8; count * 4];
        for (i, &mask_val) in offset_mask.iter().enumerate().take(count) {
            let alpha = (mask_val * sa as f32).round().clamp(0.0, 255.0) as u8;
            shadow_rgba[i * 4] = sr;
            shadow_rgba[i * 4 + 1] = sg;
            shadow_rgba[i * 4 + 2] = sb;
            shadow_rgba[i * 4 + 3] = alpha;
        }
        let shadow_img = RgbaImage::from_raw(w, h, shadow_rgba).unwrap();
        let blurred =
            crate::ops::filters::parallel_gaussian_blur_pub(&shadow_img, shadow.blur_radius);
        let blurred_raw = blurred.as_raw();

        // Composite blurred shadow onto output
        composite_over(blurred_raw, output, count);
    } else {
        // No blur — just composite the solid shadow color with offset mask alpha
        let [sr, sg, sb, sa] = shadow.color;
        for (i, &mask_val) in offset_mask.iter().enumerate().take(count) {
            let alpha = (mask_val * sa as f32).round().clamp(0.0, 255.0) as u32;
            if alpha == 0 {
                continue;
            }
            let si = i * 4;
            let da = output[si + 3] as u32;
            let inv_sa = 255 - alpha;
            let out_a = alpha + (da * inv_sa) / 255;
            if out_a == 0 {
                continue;
            }
            for (c, &sc) in [sr, sg, sb].iter().enumerate() {
                let dc = output[si + c] as u32;
                output[si + c] =
                    ((sc as u32 * alpha + dc * da * inv_sa / 255) / out_a).min(255) as u8;
            }
            output[si + 3] = out_a.min(255) as u8;
        }
    }
}

// ---------------------------------------------------------------------------
// Inner Shadow Effect
// ---------------------------------------------------------------------------

/// Render an inner shadow: invert mask → offset → blur → clip to original mask.
fn render_inner_shadow(
    coverage: &[f32],
    w: u32,
    h: u32,
    inner: &InnerShadowEffect,
    output: &mut [u8],
) {
    let ww = w as usize;
    let hh = h as usize;
    let count = ww * hh;

    // Invert the mask, then offset
    let mut inv_offset = vec![0.0f32; count];
    let dx = inner.offset_x.round() as i32;
    let dy = inner.offset_y.round() as i32;

    for y in 0..hh {
        let sy = y as i32 - dy;
        if sy < 0 || sy >= hh as i32 {
            // Outside original → inverted mask is 1.0
            inv_offset[y * ww..y * ww + ww].fill(1.0);
            continue;
        }
        for x in 0..ww {
            let sx = x as i32 - dx;
            if sx < 0 || sx >= ww as i32 {
                inv_offset[y * ww + x] = 1.0;
            } else {
                inv_offset[y * ww + x] = 1.0 - coverage[sy as usize * ww + sx as usize];
            }
        }
    }

    // Apply Gaussian blur
    let [ir, ig, ib, ia] = inner.color;
    if inner.blur_radius > 0.5 {
        let mut shadow_rgba = vec![0u8; count * 4];
        for i in 0..count {
            let alpha = (inv_offset[i] * ia as f32).round().clamp(0.0, 255.0) as u8;
            shadow_rgba[i * 4] = ir;
            shadow_rgba[i * 4 + 1] = ig;
            shadow_rgba[i * 4 + 2] = ib;
            shadow_rgba[i * 4 + 3] = alpha;
        }
        let shadow_img = RgbaImage::from_raw(w, h, shadow_rgba).unwrap();
        let blurred =
            crate::ops::filters::parallel_gaussian_blur_pub(&shadow_img, inner.blur_radius);
        let blurred_raw = blurred.as_raw();

        // Clip to original text shape and composite onto output
        for (i, &clip_alpha) in coverage.iter().enumerate().take(count) {
            if clip_alpha < 1.0 / 255.0 {
                continue;
            }
            let bi = i * 4;
            let ba = blurred_raw[bi + 3] as f32 * clip_alpha;
            let sa = ba.round() as u32;
            if sa == 0 {
                continue;
            }
            let si = i * 4;
            let da = output[si + 3] as u32;
            let inv_sa = 255u32.saturating_sub(sa);
            let out_a = sa + (da * inv_sa) / 255;
            if out_a == 0 {
                continue;
            }
            for (c, &sc) in [blurred_raw[bi], blurred_raw[bi + 1], blurred_raw[bi + 2]]
                .iter()
                .enumerate()
            {
                let dc = output[si + c] as u32;
                output[si + c] = ((sc as u32 * sa + dc * da * inv_sa / 255) / out_a).min(255) as u8;
            }
            output[si + 3] = out_a.min(255) as u8;
        }
    } else {
        // No blur
        for i in 0..count {
            let clip_alpha = coverage[i];
            if clip_alpha < 1.0 / 255.0 {
                continue;
            }
            let alpha = (inv_offset[i] * ia as f32 * clip_alpha)
                .round()
                .clamp(0.0, 255.0) as u32;
            if alpha == 0 {
                continue;
            }
            let si = i * 4;
            let da = output[si + 3] as u32;
            let inv_sa = 255u32.saturating_sub(alpha);
            let out_a = alpha + (da * inv_sa) / 255;
            if out_a == 0 {
                continue;
            }
            for (c, &sc) in [ir, ig, ib].iter().enumerate() {
                let dc = output[si + c] as u32;
                output[si + c] =
                    ((sc as u32 * alpha + dc * da * inv_sa / 255) / out_a).min(255) as u8;
            }
            output[si + 3] = out_a.min(255) as u8;
        }
    }
}

// ---------------------------------------------------------------------------
// Gradient Fill Effect
// ---------------------------------------------------------------------------

fn render_gradient_fill(
    coverage: &[f32],
    w: u32,
    h: u32,
    gradient: &GradientFillEffect,
    output: &mut [u8],
) {
    let ww = w as usize;
    let hh = h as usize;
    let count = ww * hh;
    let angle = gradient.angle_degrees.to_radians();
    let dir_x = angle.cos();
    let dir_y = angle.sin();
    let scale = gradient.scale.max(1.0);
    let off_x = gradient.offset[0];
    let off_y = gradient.offset[1];
    let [sr, sg, sb, sa] = gradient.start_color;
    let [er, eg, eb, ea] = gradient.end_color;

    let mut filled = vec![0u8; count * 4];
    filled
        .par_chunks_mut(ww * 4)
        .enumerate()
        .for_each(|(y, row)| {
            for x in 0..ww {
                let cov = coverage[y * ww + x];
                if cov < 1.0 / 255.0 {
                    continue;
                }
                let proj = ((x as f32 - off_x) * dir_x + (y as f32 - off_y) * dir_y) / scale;
                let t = if gradient.repeat {
                    proj.rem_euclid(1.0)
                } else {
                    proj.clamp(0.0, 1.0)
                };
                let inv_t = 1.0 - t;
                let idx = x * 4;
                row[idx] = (sr as f32 * inv_t + er as f32 * t)
                    .round()
                    .clamp(0.0, 255.0) as u8;
                row[idx + 1] = (sg as f32 * inv_t + eg as f32 * t)
                    .round()
                    .clamp(0.0, 255.0) as u8;
                row[idx + 2] = (sb as f32 * inv_t + eb as f32 * t)
                    .round()
                    .clamp(0.0, 255.0) as u8;
                let grad_alpha = sa as f32 * inv_t + ea as f32 * t;
                row[idx + 3] = (grad_alpha * cov).round().clamp(0.0, 255.0) as u8;
            }
        });

    composite_over(&filled, output, count);
}

// ---------------------------------------------------------------------------
// Texture Fill Effect (Batch 6)
// ---------------------------------------------------------------------------

/// Fill text with a tiled texture pattern.
fn render_texture_fill(
    text_rgba: &[u8],
    coverage: &[f32],
    w: u32,
    h: u32,
    tex: &TextureFillEffect,
    output: &mut [u8],
) {
    // Decode texture image from embedded data
    if tex.texture_data.is_empty() || tex.texture_width == 0 || tex.texture_height == 0 {
        // No valid texture — fall back to normal text fill
        composite_over(text_rgba, output, (w as usize) * (h as usize));
        return;
    }

    // Try to decode the texture
    let tex_img = match image::load_from_memory(&tex.texture_data) {
        Ok(img) => img.to_rgba8(),
        Err(_) => {
            // Decode failed — fall back to normal text fill
            composite_over(text_rgba, output, (w as usize) * (h as usize));
            return;
        }
    };

    let tw = tex_img.width() as usize;
    let th = tex_img.height() as usize;
    let tex_raw = tex_img.as_raw();
    let ww = w as usize;
    let hh = h as usize;
    let count = ww * hh;
    let scale = tex.scale.max(0.01);
    let off_x = tex.offset[0];
    let off_y = tex.offset[1];

    // Build a textured fill: for each pixel with text coverage, sample the tiled texture
    let mut textured = vec![0u8; count * 4];
    textured
        .par_chunks_mut(ww * 4)
        .enumerate()
        .for_each(|(y, row)| {
            for x in 0..ww {
                let cov = coverage[y * ww + x];
                if cov < 1.0 / 255.0 {
                    continue;
                }
                // Sample tiled texture at scaled+offset coordinates
                let tx_f = ((x as f32 - off_x) / scale) % tw as f32;
                let ty_f = ((y as f32 - off_y) / scale) % th as f32;
                let tx = ((tx_f + tw as f32) as usize) % tw;
                let ty = ((ty_f + th as f32) as usize) % th;
                let ti = (ty * tw + tx) * 4;
                let alpha = (cov * 255.0).round().clamp(0.0, 255.0) as u8;
                let pi = x * 4;
                row[pi] = tex_raw[ti];
                row[pi + 1] = tex_raw[ti + 1];
                row[pi + 2] = tex_raw[ti + 2];
                row[pi + 3] = alpha.min(tex_raw[ti + 3]);
            }
        });

    composite_over(&textured, output, count);
}
