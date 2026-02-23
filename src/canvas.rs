use std::sync::Arc;
use std::collections::VecDeque;
use eframe::egui;
use egui::{Color32, ColorImage, ImageData, Pos2, Rect, TextureOptions, TextureFilter, Vec2};
use image::{RgbaImage, GrayImage, Rgba, Luma};
use rayon::prelude::*;

/// Maximum longest-edge dimension for the LOD cache thumbnail.
const LOD_MAX_EDGE: u32 = 1024;

// ============================================================================
// SELECTION SYSTEM
// ============================================================================

/// How a new selection shape interacts with the existing mask.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum SelectionMode {
    /// Clear any existing selection, then set the new shape.
    #[default]
    Replace,
    /// Union – add to the existing mask.
    Add,
    /// Difference – subtract from the existing mask.
    Subtract,
    /// Keep only pixels present in both the existing mask AND the new shape.
    Intersect,
}

impl SelectionMode {
    pub fn label(&self) -> String {
        match self {
            SelectionMode::Replace   => t!("selection_mode.normal"),
            SelectionMode::Add       => t!("selection_mode.add"),
            SelectionMode::Subtract  => t!("selection_mode.subtract"),
            SelectionMode::Intersect => t!("selection_mode.intersect"),
        }
    }

    pub fn all() -> &'static [SelectionMode] {
        &[SelectionMode::Replace, SelectionMode::Add, SelectionMode::Subtract, SelectionMode::Intersect]
    }
}

/// Shape used during a selection drag.
#[derive(Clone, Debug)]
pub enum SelectionShape {
    Rectangle { min_x: u32, min_y: u32, max_x: u32, max_y: u32 },
    Ellipse   { cx: f32, cy: f32, rx: f32, ry: f32 },
}

impl SelectionShape {
    /// Returns 255 if the pixel (x, y) is inside the shape, 0 otherwise.
    pub fn contains(&self, x: u32, y: u32) -> u8 {
        match self {
            SelectionShape::Rectangle { min_x, min_y, max_x, max_y } => {
                if x >= *min_x && x <= *max_x && y >= *min_y && y <= *max_y {
                    255
                } else {
                    0
                }
            }
            SelectionShape::Ellipse { cx, cy, rx, ry } => {
                if *rx <= 0.0 || *ry <= 0.0 {
                    return 0;
                }
                let dx = (x as f32 - cx) / rx;
                let dy = (y as f32 - cy) / ry;
                if dx * dx + dy * dy <= 1.0 { 255 } else { 0 }
            }
        }
    }

    /// Bounding box in pixel coordinates (clamped to canvas).
    pub fn bounds(&self, canvas_w: u32, canvas_h: u32) -> (u32, u32, u32, u32) {
        match self {
            SelectionShape::Rectangle { min_x, min_y, max_x, max_y } => {
                (*min_x, *min_y, (*max_x).min(canvas_w.saturating_sub(1)), (*max_y).min(canvas_h.saturating_sub(1)))
            }
            SelectionShape::Ellipse { cx, cy, rx, ry } => {
                let min_x = (cx - rx).max(0.0).floor() as u32;
                let min_y = (cy - ry).max(0.0).floor() as u32;
                let max_x = ((cx + rx).ceil() as u32).min(canvas_w.saturating_sub(1));
                let max_y = ((cy + ry).ceil() as u32).min(canvas_h.saturating_sub(1));
                (min_x, min_y, max_x, max_y)
            }
        }
    }
}

// ============================================================================
// TILED IMAGE – sparse 64×64 chunk storage (Vec-indexed for speed)
// ============================================================================

pub const CHUNK_SIZE: u32 = 64;

/// A pixel with zero alpha, returned by reference for missing chunks.
static TRANSPARENT_PIXEL: Rgba<u8> = Rgba([0, 0, 0, 0]);

/// Sparse tiled image backed by a flat `Vec<Option<Arc<RgbaImage>>>`.
/// Chunk coordinates are mapped to a flat index via `cy * chunks_per_row + cx`,
/// giving O(1) access with zero hashing overhead.
///
/// Chunks are wrapped in `Arc` for copy-on-write semantics: `clone()` only
/// bumps reference counts (~32KB at 4K), and mutations via `put_pixel` /
/// `get_pixel_mut` use `Arc::make_mut` to COW-clone only the touched chunk.
#[derive(Clone)]
pub struct TiledImage {
    pub width: u32,
    pub height: u32,
    chunks_per_row: u32,
    chunks: Vec<Option<Arc<RgbaImage>>>,
}

impl TiledImage {
    // ---- construction -------------------------------------------------------

    /// Create an empty (fully transparent) tiled image.
    pub fn new(width: u32, height: u32) -> Self {
        // Sanity: clamp dimensions to prevent overflow (max ~256 megapixels)
        let (width, height) = {
            let total = (width as u64) * (height as u64);
            if total > 256_000_000 || width == 0 || height == 0 {
                eprintln!("TiledImage::new: dimensions {}×{} exceed 256M pixels, clamped to 1×1", width, height);
                (1, 1)
            } else {
                (width, height)
            }
        };
        let chunks_per_row = (width + CHUNK_SIZE - 1) / CHUNK_SIZE;
        let chunks_per_col = (height + CHUNK_SIZE - 1) / CHUNK_SIZE;
        let total = (chunks_per_row * chunks_per_col) as usize;
        Self {
            width,
            height,
            chunks_per_row,
            chunks: vec![None; total],
        }
    }

    /// Fill the entire image with `color`.  Chunks with `alpha == 0` are
    /// skipped so a transparent fill costs nothing.
    pub fn new_filled(width: u32, height: u32, color: Rgba<u8>) -> Self {
        let mut img = Self::new(width, height);
        if color[3] > 0 {
            img.fill(color);
        }
        img
    }

    /// Import from a flat `RgbaImage`.  Only non-transparent chunks are stored.
    /// Chunk conversion is parallelised with rayon for faster import of large images.
    pub fn from_rgba_image(src: &RgbaImage) -> Self {
        let width = src.width();
        let height = src.height();
        let mut img = Self::new(width, height);

        let chunks_x = img.chunks_per_row as usize;
        let chunks_y = ((height + CHUNK_SIZE - 1) / CHUNK_SIZE) as usize;
        let total_chunks = chunks_x * chunks_y;
        let src_raw = src.as_raw();

        let chunk_results: Vec<(usize, Option<Arc<RgbaImage>>)> = (0..total_chunks)
            .into_par_iter()
            .map(|flat| {
                let cx = (flat % chunks_x) as u32;
                let cy = (flat / chunks_x) as u32;
                let base_x = cx * CHUNK_SIZE;
                let base_y = cy * CHUNK_SIZE;

                let cw = CHUNK_SIZE.min(width - base_x);
                let ch = CHUNK_SIZE.min(height - base_y);
                let chunk_stride = CHUNK_SIZE as usize * 4;
                let mut chunk_data = vec![0u8; chunk_stride * CHUNK_SIZE as usize];
                let mut has_content = false;

                for ly in 0..ch {
                    let src_start = ((base_y + ly) * width + base_x) as usize * 4;
                    let dst_start = ly as usize * chunk_stride;
                    let byte_len = cw as usize * 4;
                    chunk_data[dst_start..dst_start + byte_len]
                        .copy_from_slice(&src_raw[src_start..src_start + byte_len]);

                    if !has_content {
                        for lx in 0..cw as usize {
                            if chunk_data[dst_start + lx * 4 + 3] != 0 {
                                has_content = true;
                                break;
                            }
                        }
                    }
                }

                if has_content {
                    let chunk = RgbaImage::from_raw(CHUNK_SIZE, CHUNK_SIZE, chunk_data).unwrap();
                    (flat, Some(Arc::new(chunk)))
                } else {
                    (flat, None)
                }
            })
            .collect();

        for (idx, chunk) in chunk_results {
            img.chunks[idx] = chunk;
        }
        img
    }

    /// Import from a flat RGBA byte slice without creating an intermediate `RgbaImage`.
    /// `data` must be exactly `width * height * 4` bytes (row-major, RGBA).
    /// Only non-transparent chunks are stored.  Parallelised with rayon.
    pub fn from_raw_rgba(width: u32, height: u32, data: &[u8]) -> Self {
        debug_assert_eq!(data.len(), (width as usize) * (height as usize) * 4);
        let mut img = Self::new(width, height);

        let chunks_x = img.chunks_per_row as usize;
        let chunks_y = ((height + CHUNK_SIZE - 1) / CHUNK_SIZE) as usize;
        let total_chunks = chunks_x * chunks_y;

        let chunk_results: Vec<(usize, Option<Arc<RgbaImage>>)> = (0..total_chunks)
            .into_par_iter()
            .map(|flat| {
                let cx = (flat % chunks_x) as u32;
                let cy = (flat / chunks_x) as u32;
                let base_x = cx * CHUNK_SIZE;
                let base_y = cy * CHUNK_SIZE;

                let cw = CHUNK_SIZE.min(width - base_x);
                let ch = CHUNK_SIZE.min(height - base_y);
                let chunk_stride = CHUNK_SIZE as usize * 4;
                let mut chunk_data = vec![0u8; chunk_stride * CHUNK_SIZE as usize];
                let mut has_content = false;

                for ly in 0..ch {
                    let src_start = ((base_y + ly) * width + base_x) as usize * 4;
                    let dst_start = ly as usize * chunk_stride;
                    let byte_len = cw as usize * 4;
                    chunk_data[dst_start..dst_start + byte_len]
                        .copy_from_slice(&data[src_start..src_start + byte_len]);

                    if !has_content {
                        for lx in 0..cw as usize {
                            if chunk_data[dst_start + lx * 4 + 3] != 0 {
                                has_content = true;
                                break;
                            }
                        }
                    }
                }

                if has_content {
                    let chunk = RgbaImage::from_raw(CHUNK_SIZE, CHUNK_SIZE, chunk_data).unwrap();
                    (flat, Some(Arc::new(chunk)))
                } else {
                    (flat, None)
                }
            })
            .collect();

        for (idx, chunk) in chunk_results {
            img.chunks[idx] = chunk;
        }
        img
    }

    /// Build a TiledImage from a sub-region RGBA buffer.
    ///
    /// `data` is `region_w * region_h * 4` bytes, positioned at `(off_x, off_y)`
    /// within a `canvas_w × canvas_h` canvas. Only chunks overlapping the region
    /// are created (parallelized with rayon). Much faster than per-pixel `put_pixel`.
    pub fn from_region_rgba(
        canvas_w: u32, canvas_h: u32,
        data: &[u8],
        region_w: u32, region_h: u32,
        off_x: i32, off_y: i32,
    ) -> Self {
        let mut img = Self::new(canvas_w, canvas_h);
        if region_w == 0 || region_h == 0 || data.is_empty() {
            return img;
        }

        let chunks_x = img.chunks_per_row as usize;

        // Compute chunk coordinate range that overlaps the region
        let rx0 = off_x.max(0) as u32;
        let ry0 = off_y.max(0) as u32;
        let rx1 = ((off_x + region_w as i32) as u32).min(canvas_w);
        let ry1 = ((off_y + region_h as i32) as u32).min(canvas_h);

        let cx_start = (rx0 / CHUNK_SIZE) as usize;
        let cx_end = ((rx1 + CHUNK_SIZE - 1) / CHUNK_SIZE) as usize;
        let cy_start = (ry0 / CHUNK_SIZE) as usize;
        let cy_end = ((ry1 + CHUNK_SIZE - 1) / CHUNK_SIZE) as usize;

        // Collect all chunk indices that overlap the region
        let mut overlapping: Vec<usize> = Vec::with_capacity((cx_end - cx_start) * (cy_end - cy_start));
        for cy in cy_start..cy_end {
            for cx in cx_start..cx_end {
                overlapping.push(cy * chunks_x + cx);
            }
        }

        let region_stride = region_w as usize * 4;
        let chunk_results: Vec<(usize, Option<Arc<RgbaImage>>)> = overlapping
            .into_par_iter()
            .map(|flat| {
                let cx = (flat % chunks_x) as u32;
                let cy = (flat / chunks_x) as u32;
                let base_x = cx * CHUNK_SIZE;
                let base_y = cy * CHUNK_SIZE;

                let cw = CHUNK_SIZE.min(canvas_w - base_x);
                let ch = CHUNK_SIZE.min(canvas_h - base_y);
                let chunk_stride = CHUNK_SIZE as usize * 4;
                let mut chunk_data = vec![0u8; chunk_stride * CHUNK_SIZE as usize];
                let mut has_content = false;

                for ly in 0..ch {
                    let gy = base_y + ly;
                    let ry = gy as i32 - off_y;
                    if ry < 0 || ry >= region_h as i32 { continue; }

                    // Compute horizontal overlap
                    let gx_start = base_x;
                    let gx_end = base_x + cw;
                    let rx_start = (gx_start as i32 - off_x).max(0) as u32;
                    let rx_end = ((gx_end as i32 - off_x) as u32).min(region_w);
                    if rx_start >= rx_end { continue; }

                    let lx_start = (off_x + rx_start as i32) as u32 - base_x;
                    let copy_w = rx_end - rx_start;

                    let src_start = ry as usize * region_stride + rx_start as usize * 4;
                    let dst_start = ly as usize * chunk_stride + lx_start as usize * 4;
                    let byte_len = copy_w as usize * 4;

                    chunk_data[dst_start..dst_start + byte_len]
                        .copy_from_slice(&data[src_start..src_start + byte_len]);

                    if !has_content {
                        for px in 0..copy_w as usize {
                            if chunk_data[dst_start + px * 4 + 3] != 0 {
                                has_content = true;
                                break;
                            }
                        }
                    }
                }

                if has_content {
                    let chunk = RgbaImage::from_raw(CHUNK_SIZE, CHUNK_SIZE, chunk_data).unwrap();
                    (flat, Some(Arc::new(chunk)))
                } else {
                    (flat, None)
                }
            })
            .collect();

        for (idx, chunk) in chunk_results {
            img.chunks[idx] = chunk;
        }
        img
    }

    /// Flatten back to a contiguous `RgbaImage`.
    pub fn to_rgba_image(&self) -> RgbaImage {
        let mut out = RgbaImage::new(self.width, self.height);
        let out_raw = out.as_mut();
        let out_stride = self.width as usize * 4;
        for (cx, cy) in self.chunk_keys() {
            if let Some(chunk) = self.get_chunk(cx, cy) {
                let base_x = cx * CHUNK_SIZE;
                let base_y = cy * CHUNK_SIZE;
                let cw = (CHUNK_SIZE.min(self.width.saturating_sub(base_x))) as usize;
                let ch = CHUNK_SIZE.min(self.height.saturating_sub(base_y));
                let chunk_raw = chunk.as_raw();
                let chunk_stride = CHUNK_SIZE as usize * 4;
                for ly in 0..ch as usize {
                    let src_start = ly * chunk_stride;
                    let src_end = src_start + cw * 4;
                    let dst_start = (base_y as usize + ly) * out_stride + base_x as usize * 4;
                    let dst_end = dst_start + cw * 4;
                    out_raw[dst_start..dst_end].copy_from_slice(&chunk_raw[src_start..src_end]);
                }
            }
        }
        out
    }

    // ---- chunk-level flip / rotate (avoid full-image materialisation) --------

    /// Flip horizontally without materialising the full image.
    /// Iterates source chunks and writes transformed pixels to a new chunk array.
    pub fn flip_horizontal_chunked(&mut self) {
        let cs = CHUNK_SIZE;
        let w = self.width;
        let cpr = self.chunks_per_row;
        let total = self.chunks.len();
        let mut dst: Vec<Option<Arc<RgbaImage>>> = vec![None; total];

        for (src_idx, slot) in self.chunks.iter().enumerate() {
            if let Some(chunk) = slot {
                let src_cx = (src_idx as u32) % cpr;
                let src_cy = (src_idx as u32) / cpr;
                let base_x = src_cx * cs;
                let base_y = src_cy * cs;
                let cw = cs.min(w - base_x);
                let ch = cs.min(self.height - base_y);
                let raw = chunk.as_raw();
                let stride = cs as usize * 4;

                for ly in 0..ch {
                    let row_off = ly as usize * stride;
                    for lx in 0..cw {
                        let off = row_off + lx as usize * 4;
                        if raw[off + 3] == 0 { continue; }

                        let dst_x = w - 1 - (base_x + lx);
                        let dst_cx = dst_x / cs;
                        let dst_lx = dst_x % cs;
                        let dst_i = (src_cy * cpr + dst_cx) as usize;

                        let dc = Arc::make_mut(dst[dst_i]
                            .get_or_insert_with(|| Arc::new(RgbaImage::new(cs, cs))));
                        dc.put_pixel(dst_lx, ly, Rgba([
                            raw[off], raw[off + 1], raw[off + 2], raw[off + 3],
                        ]));
                    }
                }
            }
        }
        self.chunks = dst;
    }

    /// Flip vertically without materialising the full image.
    pub fn flip_vertical_chunked(&mut self) {
        let cs = CHUNK_SIZE;
        let h = self.height;
        let cpr = self.chunks_per_row;
        let total = self.chunks.len();
        let mut dst: Vec<Option<Arc<RgbaImage>>> = vec![None; total];

        for (src_idx, slot) in self.chunks.iter().enumerate() {
            if let Some(chunk) = slot {
                let src_cx = (src_idx as u32) % cpr;
                let src_cy = (src_idx as u32) / cpr;
                let base_x = src_cx * cs;
                let base_y = src_cy * cs;
                let cw = cs.min(self.width - base_x);
                let ch = cs.min(h - base_y);
                let raw = chunk.as_raw();
                let stride = cs as usize * 4;

                for ly in 0..ch {
                    let row_off = ly as usize * stride;
                    let dst_y = h - 1 - (base_y + ly);
                    let dst_cy = dst_y / cs;
                    let dst_ly = dst_y % cs;
                    let dst_i = (dst_cy * cpr + src_cx) as usize;

                    for lx in 0..cw {
                        let off = row_off + lx as usize * 4;
                        if raw[off + 3] == 0 { continue; }

                        let dc = Arc::make_mut(dst[dst_i]
                            .get_or_insert_with(|| Arc::new(RgbaImage::new(cs, cs))));
                        dc.put_pixel(lx, dst_ly, Rgba([
                            raw[off], raw[off + 1], raw[off + 2], raw[off + 3],
                        ]));
                    }
                }
            }
        }
        self.chunks = dst;
    }

    /// Rotate 180° without materialising the full image (= H flip + V flip combined).
    pub fn rotate_180_chunked(&mut self) {
        let cs = CHUNK_SIZE;
        let w = self.width;
        let h = self.height;
        let cpr = self.chunks_per_row;
        let total = self.chunks.len();
        let mut dst: Vec<Option<Arc<RgbaImage>>> = vec![None; total];

        for (src_idx, slot) in self.chunks.iter().enumerate() {
            if let Some(chunk) = slot {
                let src_cx = (src_idx as u32) % cpr;
                let src_cy = (src_idx as u32) / cpr;
                let base_x = src_cx * cs;
                let base_y = src_cy * cs;
                let cw = cs.min(w - base_x);
                let ch = cs.min(h - base_y);
                let raw = chunk.as_raw();
                let stride = cs as usize * 4;

                for ly in 0..ch {
                    let row_off = ly as usize * stride;
                    let dst_y = h - 1 - (base_y + ly);
                    let dst_cy = dst_y / cs;
                    let dst_ly = dst_y % cs;

                    for lx in 0..cw {
                        let off = row_off + lx as usize * 4;
                        if raw[off + 3] == 0 { continue; }

                        let dst_x = w - 1 - (base_x + lx);
                        let dst_cx = dst_x / cs;
                        let dst_lx = dst_x % cs;
                        let dst_i = (dst_cy * cpr + dst_cx) as usize;

                        let dc = Arc::make_mut(dst[dst_i]
                            .get_or_insert_with(|| Arc::new(RgbaImage::new(cs, cs))));
                        dc.put_pixel(dst_lx, dst_ly, Rgba([
                            raw[off], raw[off + 1], raw[off + 2], raw[off + 3],
                        ]));
                    }
                }
            }
        }
        self.chunks = dst;
    }

    /// Rotate 90° CW without materialising the full image.
    /// Returns a new TiledImage with swapped dimensions (W×H → H×W).
    pub fn rotate_90cw_chunked(&self) -> TiledImage {
        let cs = CHUNK_SIZE;
        let old_w = self.width;
        let old_h = self.height;
        let new_w = old_h;
        let new_h = old_w;
        let new_cpr = (new_w + cs - 1) / cs;
        let new_cpc = (new_h + cs - 1) / cs;
        let total = (new_cpr * new_cpc) as usize;
        let mut dst: Vec<Option<Arc<RgbaImage>>> = vec![None; total];

        for (src_idx, slot) in self.chunks.iter().enumerate() {
            if let Some(chunk) = slot {
                let src_cx = (src_idx as u32) % self.chunks_per_row;
                let src_cy = (src_idx as u32) / self.chunks_per_row;
                let base_x = src_cx * cs;
                let base_y = src_cy * cs;
                let cw = cs.min(old_w - base_x);
                let ch = cs.min(old_h - base_y);
                let raw = chunk.as_raw();
                let stride = cs as usize * 4;

                for ly in 0..ch {
                    let row_off = ly as usize * stride;
                    for lx in 0..cw {
                        let off = row_off + lx as usize * 4;
                        if raw[off + 3] == 0 { continue; }
                        // 90° CW: (x, y) → (old_h - 1 - y, x)
                        let dx = old_h - 1 - (base_y + ly);
                        let dy = base_x + lx;
                        let dcx = dx / cs;
                        let dcy = dy / cs;
                        let dlx = dx % cs;
                        let dly = dy % cs;
                        let di = (dcy * new_cpr + dcx) as usize;

                        let dc = Arc::make_mut(dst[di]
                            .get_or_insert_with(|| Arc::new(RgbaImage::new(cs, cs))));
                        dc.put_pixel(dlx, dly, Rgba([
                            raw[off], raw[off + 1], raw[off + 2], raw[off + 3],
                        ]));
                    }
                }
            }
        }

        TiledImage { width: new_w, height: new_h, chunks_per_row: new_cpr, chunks: dst }
    }

    /// Rotate 90° CCW without materialising the full image.
    /// Returns a new TiledImage with swapped dimensions (W×H → H×W).
    pub fn rotate_90ccw_chunked(&self) -> TiledImage {
        let cs = CHUNK_SIZE;
        let old_w = self.width;
        let old_h = self.height;
        let new_w = old_h;
        let new_h = old_w;
        let new_cpr = (new_w + cs - 1) / cs;
        let new_cpc = (new_h + cs - 1) / cs;
        let total = (new_cpr * new_cpc) as usize;
        let mut dst: Vec<Option<Arc<RgbaImage>>> = vec![None; total];

        for (src_idx, slot) in self.chunks.iter().enumerate() {
            if let Some(chunk) = slot {
                let src_cx = (src_idx as u32) % self.chunks_per_row;
                let src_cy = (src_idx as u32) / self.chunks_per_row;
                let base_x = src_cx * cs;
                let base_y = src_cy * cs;
                let cw = cs.min(old_w - base_x);
                let ch = cs.min(old_h - base_y);
                let raw = chunk.as_raw();
                let stride = cs as usize * 4;

                for ly in 0..ch {
                    let row_off = ly as usize * stride;
                    for lx in 0..cw {
                        let off = row_off + lx as usize * 4;
                        if raw[off + 3] == 0 { continue; }
                        // 90° CCW: (x, y) → (y, old_w - 1 - x)
                        let dx = base_y + ly;
                        let dy = old_w - 1 - (base_x + lx);
                        let dcx = dx / cs;
                        let dcy = dy / cs;
                        let dlx = dx % cs;
                        let dly = dy % cs;
                        let di = (dcy * new_cpr + dcx) as usize;

                        let dc = Arc::make_mut(dst[di]
                            .get_or_insert_with(|| Arc::new(RgbaImage::new(cs, cs))));
                        dc.put_pixel(dlx, dly, Rgba([
                            raw[off], raw[off + 1], raw[off + 2], raw[off + 3],
                        ]));
                    }
                }
            }
        }

        TiledImage { width: new_w, height: new_h, chunks_per_row: new_cpr, chunks: dst }
    }

    /// Extract a sub-rectangle of the image as raw RGBA bytes (tightly packed,
    /// `rect_w * rect_h * 4` bytes).  Used for partial GPU texture uploads.
    pub fn extract_region_rgba(&self, rx: u32, ry: u32, rw: u32, rh: u32) -> Vec<u8> {
        let size = (rw as u64) * (rh as u64) * 4;
        if size > 1_073_741_824 { return Vec::new(); } // 1GB sanity limit
        let mut buf = vec![0u8; size as usize];
        for y in 0..rh {
            let iy = ry + y;
            if iy >= self.height { continue; }
            for x in 0..rw {
                let ix = rx + x;
                if ix >= self.width { continue; }
                let px = self.get_pixel(ix, iy);
                let off = ((y * rw + x) * 4) as usize;
                buf[off]     = px[0];
                buf[off + 1] = px[1];
                buf[off + 2] = px[2];
                buf[off + 3] = px[3];
            }
        }
        buf
    }

    /// Fast chunk-aware region extraction.  Instead of per-pixel `get_pixel()`
    /// calls, iterates only overlapping chunks and copies rows via `memcpy`.
    /// For a 500×500 stroke region on a 5000×5000 canvas this is ~100× faster
    /// than `extract_region_rgba` / `to_rgba_image`.
    pub fn extract_region_rgba_fast(&self, rx: u32, ry: u32, rw: u32, rh: u32, buf: &mut Vec<u8>) {
        let size = (rw as u64) * (rh as u64) * 4;
        if size > 1_073_741_824 { buf.clear(); return; } // 1GB sanity limit
        let needed = size as usize;
        buf.resize(needed, 0);
        // Zero the buffer (un-populated chunks must be transparent)
        // Using a fast fill instead of allocating a new Vec each frame.
        for b in buf.iter_mut() { *b = 0; }

        let cx_start = rx / CHUNK_SIZE;
        let cx_end = (rx + rw + CHUNK_SIZE - 1) / CHUNK_SIZE;
        let cy_start = ry / CHUNK_SIZE;
        let cy_end = (ry + rh + CHUNK_SIZE - 1) / CHUNK_SIZE;

        for cy in cy_start..cy_end {
            for cx in cx_start..cx_end {
                let chunk = match self.get_chunk(cx, cy) {
                    Some(c) => c,
                    None => continue, // transparent – already zeroed
                };
                let base_x = cx * CHUNK_SIZE;
                let base_y = cy * CHUNK_SIZE;

                // Overlap between this chunk and the requested region
                let ox  = rx.max(base_x);
                let oy  = ry.max(base_y);
                let ox2 = (rx + rw).min(base_x + CHUNK_SIZE).min(self.width);
                let oy2 = (ry + rh).min(base_y + CHUNK_SIZE).min(self.height);
                if ox >= ox2 || oy >= oy2 { continue; }

                let ow = (ox2 - ox) as usize;
                let chunk_raw = chunk.as_raw();
                let chunk_stride = CHUNK_SIZE as usize * 4;

                for sy in oy..oy2 {
                    let lx = (ox - base_x) as usize;
                    let ly = (sy - base_y) as usize;
                    let src_start = ly * chunk_stride + lx * 4;
                    let src_end   = src_start + ow * 4;

                    let dx = (ox - rx) as usize;
                    let dy = (sy - ry) as usize;
                    let dst_start = dy * (rw as usize) * 4 + dx * 4;
                    let dst_end   = dst_start + ow * 4;

                    buf[dst_start..dst_end].copy_from_slice(&chunk_raw[src_start..src_end]);
                }
            }
        }
    }

    // ---- indexing helpers ----------------------------------------------------

    #[inline(always)]
    fn flat_index(&self, cx: u32, cy: u32) -> usize {
        (cy * self.chunks_per_row + cx) as usize
    }

    #[inline(always)]
    fn chunk_coord(x: u32, y: u32) -> (u32, u32) { (x / CHUNK_SIZE, y / CHUNK_SIZE) }

    #[inline(always)]
    fn local(x: u32, y: u32) -> (u32, u32) { (x % CHUNK_SIZE, y % CHUNK_SIZE) }

    // ---- pixel access -------------------------------------------------------

    /// Read a pixel (returns `&TRANSPARENT_PIXEL` for missing chunks).
    #[inline]
    pub fn get_pixel(&self, x: u32, y: u32) -> &Rgba<u8> {
        if x >= self.width || y >= self.height { return &TRANSPARENT_PIXEL; }
        let (cx, cy) = Self::chunk_coord(x, y);
        let (lx, ly) = Self::local(x, y);
        let idx = self.flat_index(cx, cy);
        self.chunks[idx].as_ref()
            .map(|c| c.get_pixel(lx, ly))
            .unwrap_or(&TRANSPARENT_PIXEL)
    }

    /// Write a pixel (creates the chunk on demand, COW-clones if shared).
    #[inline]
    pub fn put_pixel(&mut self, x: u32, y: u32, pixel: Rgba<u8>) {
        if x >= self.width || y >= self.height { return; }
        let (cx, cy) = Self::chunk_coord(x, y);
        let (lx, ly) = Self::local(x, y);
        let idx = self.flat_index(cx, cy);
        let arc = self.chunks[idx]
            .get_or_insert_with(|| Arc::new(RgbaImage::new(CHUNK_SIZE, CHUNK_SIZE)));
        Arc::make_mut(arc).put_pixel(lx, ly, pixel);
    }

    /// Blit an RGBA sub-image at a given position using bulk chunk row copies.
    /// Much faster than per-pixel put_pixel for large regions.
    pub fn blit_rgba_at(&mut self, dst_x: i32, dst_y: i32, src_w: u32, src_h: u32, data: &[u8]) {
        debug_assert_eq!(data.len(), src_w as usize * src_h as usize * 4);
        let cs = CHUNK_SIZE;

        for sy in 0..src_h {
            let gy = dst_y + sy as i32;
            if gy < 0 || gy as u32 >= self.height { continue; }
            let gy = gy as u32;

            let src_row_start = sy as usize * src_w as usize * 4;

            // Process contiguous runs of pixels in this row
            let mut sx = 0u32;
            while sx < src_w {
                let gx = dst_x + sx as i32;
                if gx < 0 { sx += 1; continue; }
                let gx = gx as u32;
                if gx >= self.width { break; }

                let (cx, cy) = Self::chunk_coord(gx, gy);
                let (lx, ly) = Self::local(gx, gy);
                let idx = self.flat_index(cx, cy);

                // How many pixels can we write into this chunk row?
                let chunk_remaining = (cs - lx) as u32;
                let src_remaining = src_w - sx;
                let canvas_remaining = self.width - gx;
                let run = chunk_remaining.min(src_remaining).min(canvas_remaining);

                // Check if this run has any non-transparent pixels
                let src_off = src_row_start + sx as usize * 4;
                let byte_len = run as usize * 4;
                let has_content = data[src_off..src_off + byte_len]
                    .chunks_exact(4)
                    .any(|px| px[3] != 0);

                if has_content {
                    let arc = self.chunks[idx]
                        .get_or_insert_with(|| Arc::new(RgbaImage::new(cs, cs)));
                    let chunk = Arc::make_mut(arc);
                    let dst_off = (ly as usize * cs as usize + lx as usize) * 4;
                    chunk.as_mut()[dst_off..dst_off + byte_len]
                        .copy_from_slice(&data[src_off..src_off + byte_len]);
                }

                sx += run;
            }
        }
    }

    /// Mutable reference to a pixel (creates the chunk on demand, COW-clones if shared).
    #[inline]
    pub fn get_pixel_mut(&mut self, x: u32, y: u32) -> &mut Rgba<u8> {
        let (cx, cy) = Self::chunk_coord(x, y);
        let (lx, ly) = Self::local(x, y);
        let idx = self.flat_index(cx, cy);
        let arc = self.chunks[idx]
            .get_or_insert_with(|| Arc::new(RgbaImage::new(CHUNK_SIZE, CHUNK_SIZE)));
        Arc::make_mut(arc).get_pixel_mut(lx, ly)
    }

    /// Read-only access to a chunk (if it exists).
    pub fn get_chunk(&self, cx: u32, cy: u32) -> Option<&RgbaImage> {
        let idx = self.flat_index(cx, cy);
        self.chunks.get(idx).and_then(|c| c.as_deref())
    }

    /// Mutable access to an existing chunk (COW-clones if shared).
    pub fn get_chunk_mut(&mut self, cx: u32, cy: u32) -> Option<&mut RgbaImage> {
        let idx = self.flat_index(cx, cy);
        self.chunks.get_mut(idx)
            .and_then(|slot| slot.as_mut())
            .map(|arc| Arc::make_mut(arc))
    }

    /// Get or create a chunk, returning a mutable reference (COW-safe).
    pub fn ensure_chunk_mut(&mut self, cx: u32, cy: u32) -> &mut RgbaImage {
        let idx = self.flat_index(cx, cy);
        let arc = self.chunks[idx]
            .get_or_insert_with(|| Arc::new(RgbaImage::new(CHUNK_SIZE, CHUNK_SIZE)));
        Arc::make_mut(arc)
    }

    /// Place a fully-built chunk at the given chunk coordinate.
    pub fn set_chunk(&mut self, cx: u32, cy: u32, chunk: RgbaImage) {
        let idx = self.flat_index(cx, cy);
        if idx < self.chunks.len() {
            self.chunks[idx] = Some(Arc::new(chunk));
        }
    }

    /// Iterator over populated chunk coordinates.
    pub fn chunk_keys(&self) -> impl Iterator<Item = (u32, u32)> + '_ {
        let cpr = self.chunks_per_row;
        self.chunks.iter().enumerate().filter_map(move |(i, slot)| {
            if slot.is_some() {
                Some(((i as u32) % cpr, (i as u32) / cpr))
            } else {
                None
            }
        })
    }

    /// Number of populated chunks.
    pub fn chunk_count(&self) -> usize {
        self.chunks.iter().filter(|c| c.is_some()).count()
    }

    // ---- bulk operations ----------------------------------------------------

    /// Fill every pixel with `color`.
    pub fn fill(&mut self, color: Rgba<u8>) {
        for slot in &mut self.chunks {
            let arc = slot.get_or_insert_with(|| Arc::new(RgbaImage::new(CHUNK_SIZE, CHUNK_SIZE)));
            let chunk = Arc::make_mut(arc);
            for pixel in chunk.pixels_mut() {
                *pixel = color;
            }
        }
    }

    /// Drop all chunks (make the image fully transparent).
    pub fn clear(&mut self) {
        for slot in &mut self.chunks {
            *slot = None;
        }
    }

    /// Drop chunks whose coordinates overlap the given pixel-space rectangle.
    pub fn clear_region(&mut self, min_x: u32, min_y: u32, max_x: u32, max_y: u32) {
        let min_cx = min_x / CHUNK_SIZE;
        let max_cx = (max_x + CHUNK_SIZE - 1) / CHUNK_SIZE;
        let min_cy = min_y / CHUNK_SIZE;
        let max_cy = (max_y + CHUNK_SIZE - 1) / CHUNK_SIZE;
        let total_cx = self.chunks_per_row;
        let total_cy = (self.height + CHUNK_SIZE - 1) / CHUNK_SIZE;
        for cy in min_cy..max_cy.min(total_cy) {
            for cx in min_cx..max_cx.min(total_cx) {
                let idx = (cy * total_cx + cx) as usize;
                if idx < self.chunks.len() {
                    self.chunks[idx] = None;
                }
            }
        }
    }

    /// Width accessor (matches `RgbaImage::width()`).
    pub fn width(&self) -> u32 { self.width }

    /// Height accessor (matches `RgbaImage::height()`).
    pub fn height(&self) -> u32 { self.height }

    /// Approximate memory usage in bytes.
    /// Shared (COW) chunks are counted at minimal cost (Arc pointer only)
    /// since their pixel data is shared with undo snapshots.
    pub fn memory_bytes(&self) -> usize {
        let chunk_byte_size = (CHUNK_SIZE * CHUNK_SIZE * 4) as usize;
        self.chunks.iter()
            .filter_map(|c| c.as_ref())
            .map(|arc| {
                if Arc::strong_count(arc) == 1 {
                    chunk_byte_size
                } else {
                    // Shared with snapshots — only count the Arc pointer overhead
                    std::mem::size_of::<usize>() * 2
                }
            })
            .sum()
    }

    /// Total pixel memory owned by this image (ignoring sharing).
    /// Used for diagnostic display.
    pub fn memory_bytes_total(&self) -> usize {
        self.chunks.iter().filter(|c| c.is_some()).count() * (CHUNK_SIZE * CHUNK_SIZE * 4) as usize
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum BlendMode {
    #[default]
    Normal,
    Multiply,
    Screen,
    Additive,
    Reflect,
    Glow,
    ColorBurn,
    ColorDodge,
    Overlay,
    Difference,
    Negation,
    Lighten,
    Darken,
    Xor,
    Overwrite,
    HardLight,
    SoftLight,
    Exclusion,
    Subtract,
    Divide,
    LinearBurn,
    VividLight,
    LinearLight,
    PinLight,
    HardMix,
}

impl BlendMode {
    /// Returns all blend modes for UI display
    pub fn all() -> &'static [BlendMode] {
        &[
            BlendMode::Normal,
            BlendMode::Multiply,
            BlendMode::Screen,
            BlendMode::Additive,
            BlendMode::Overlay,
            BlendMode::HardLight,
            BlendMode::SoftLight,
            BlendMode::Lighten,
            BlendMode::Darken,
            BlendMode::ColorBurn,
            BlendMode::ColorDodge,
            BlendMode::Difference,
            BlendMode::Exclusion,
            BlendMode::Negation,
            BlendMode::Reflect,
            BlendMode::Glow,
            BlendMode::Subtract,
            BlendMode::Divide,
            BlendMode::LinearBurn,
            BlendMode::VividLight,
            BlendMode::LinearLight,
            BlendMode::PinLight,
            BlendMode::HardMix,
            BlendMode::Xor,
            BlendMode::Overwrite,
        ]
    }
    
    pub fn name(&self) -> &'static str {
        match self {
            BlendMode::Normal => "Normal",
            BlendMode::Multiply => "Multiply",
            BlendMode::Screen => "Screen",
            BlendMode::Additive => "Additive",
            BlendMode::Reflect => "Reflect",
            BlendMode::Glow => "Glow",
            BlendMode::ColorBurn => "Color Burn",
            BlendMode::ColorDodge => "Color Dodge",
            BlendMode::Overlay => "Overlay",
            BlendMode::Difference => "Difference",
            BlendMode::Negation => "Negation",
            BlendMode::Lighten => "Lighten",
            BlendMode::Darken => "Darken",
            BlendMode::Xor => "Xor",
            BlendMode::Overwrite => "Overwrite",
            BlendMode::HardLight => "Hard Light",
            BlendMode::SoftLight => "Soft Light",
            BlendMode::Exclusion => "Exclusion",
            BlendMode::Subtract => "Subtract",
            BlendMode::Divide => "Divide",
            BlendMode::LinearBurn => "Linear Burn",
            BlendMode::VividLight => "Vivid Light",
            BlendMode::LinearLight => "Linear Light",
            BlendMode::PinLight => "Pin Light",
            BlendMode::HardMix => "Hard Mix",
        }
    }

    /// Returns the localized display name for UI rendering
    pub fn display_name(&self) -> String {
        match self {
            BlendMode::Normal => t!("blend.normal"),
            BlendMode::Multiply => t!("blend.multiply"),
            BlendMode::Screen => t!("blend.screen"),
            BlendMode::Additive => t!("blend.additive"),
            BlendMode::Reflect => t!("blend.reflect"),
            BlendMode::Glow => t!("blend.glow"),
            BlendMode::ColorBurn => t!("blend.color_burn"),
            BlendMode::ColorDodge => t!("blend.color_dodge"),
            BlendMode::Overlay => t!("blend.overlay"),
            BlendMode::Difference => t!("blend.difference"),
            BlendMode::Negation => t!("blend.negation"),
            BlendMode::Lighten => t!("blend.lighten"),
            BlendMode::Darken => t!("blend.darken"),
            BlendMode::Xor => t!("blend.xor"),
            BlendMode::Overwrite => t!("blend.overwrite"),
            BlendMode::HardLight => t!("blend.hard_light"),
            BlendMode::SoftLight => t!("blend.soft_light"),
            BlendMode::Exclusion => t!("blend.exclusion"),
            BlendMode::Subtract => t!("blend.subtract"),
            BlendMode::Divide => t!("blend.divide"),
            BlendMode::LinearBurn => t!("blend.linear_burn"),
            BlendMode::VividLight => t!("blend.vivid_light"),
            BlendMode::LinearLight => t!("blend.linear_light"),
            BlendMode::PinLight => t!("blend.pin_light"),
            BlendMode::HardMix => t!("blend.hard_mix"),
        }
    }

    /// Convert to a stable u8 for binary serialization
    pub fn to_u8(&self) -> u8 {
        match self {
            BlendMode::Normal => 0,
            BlendMode::Multiply => 1,
            BlendMode::Screen => 2,
            BlendMode::Additive => 3,
            BlendMode::Reflect => 4,
            BlendMode::Glow => 5,
            BlendMode::ColorBurn => 6,
            BlendMode::ColorDodge => 7,
            BlendMode::Overlay => 8,
            BlendMode::Difference => 9,
            BlendMode::Negation => 10,
            BlendMode::Lighten => 11,
            BlendMode::Darken => 12,
            BlendMode::Xor => 13,
            BlendMode::Overwrite => 14,
            BlendMode::HardLight => 15,
            BlendMode::SoftLight => 16,
            BlendMode::Exclusion => 17,
            BlendMode::Subtract => 18,
            BlendMode::Divide => 19,
            BlendMode::LinearBurn => 20,
            BlendMode::VividLight => 21,
            BlendMode::LinearLight => 22,
            BlendMode::PinLight => 23,
            BlendMode::HardMix => 24,
        }
    }

    /// Reconstruct from a u8 (defaults to Normal for unknown values)
    pub fn from_u8(v: u8) -> Self {
        match v {
            0 => BlendMode::Normal,
            1 => BlendMode::Multiply,
            2 => BlendMode::Screen,
            3 => BlendMode::Additive,
            4 => BlendMode::Reflect,
            5 => BlendMode::Glow,
            6 => BlendMode::ColorBurn,
            7 => BlendMode::ColorDodge,
            8 => BlendMode::Overlay,
            9 => BlendMode::Difference,
            10 => BlendMode::Negation,
            11 => BlendMode::Lighten,
            12 => BlendMode::Darken,
            13 => BlendMode::Xor,
            14 => BlendMode::Overwrite,
            15 => BlendMode::HardLight,
            16 => BlendMode::SoftLight,
            17 => BlendMode::Exclusion,
            18 => BlendMode::Subtract,
            19 => BlendMode::Divide,
            20 => BlendMode::LinearBurn,
            21 => BlendMode::VividLight,
            22 => BlendMode::LinearLight,
            23 => BlendMode::PinLight,
            24 => BlendMode::HardMix,
            _ => BlendMode::Normal,
        }
    }
}

pub struct Layer {
    pub name: String,
    pub visible: bool,
    pub opacity: f32,
    pub blend_mode: BlendMode,
    pub pixels: TiledImage,
    /// Downscaled cache (max 1024px longest edge) for zoomed-out rendering.
    /// Not serialized — rebuilt on demand.
    pub lod_cache: Option<Arc<RgbaImage>>,
    /// Per-layer generation counter for GPU texture synchronisation.
    /// Bumped only when THIS layer's pixels are modified, so unchanged
    /// layers are never re-uploaded to the GPU.
    pub gpu_generation: u64,
}

impl Layer {
    pub fn new(name: String, width: u32, height: u32, fill_color: Rgba<u8>) -> Self {
        let pixels = TiledImage::new_filled(width, height, fill_color);
        
        Self {
            name,
            visible: true,
            opacity: 1.0,
            blend_mode: BlendMode::Normal,
            pixels,
            lod_cache: None,
            gpu_generation: 0,
        }
    }

    /// Invalidate the LOD cache (call after any pixel modification).
    pub fn invalidate_lod(&mut self) {
        self.lod_cache = None;
    }

    /// Return a reference to the downscaled LOD image, generating it lazily.
    /// The thumbnail is at most `LOD_MAX_EDGE` pixels on its longest side.
    pub fn get_lod_image(&mut self) -> Arc<RgbaImage> {
        if let Some(ref cached) = self.lod_cache {
            return Arc::clone(cached);
        }
        let (w, h) = (self.pixels.width(), self.pixels.height());
        let longest = w.max(h);
        let (nw, nh) = if longest <= LOD_MAX_EDGE {
            (w, h) // Already small enough
        } else {
            let scale = LOD_MAX_EDGE as f32 / longest as f32;
            (((w as f32 * scale).round() as u32).max(1),
             ((h as f32 * scale).round() as u32).max(1))
        };
        let flat = self.pixels.to_rgba_image();
        let thumb = image::imageops::resize(&flat, nw, nh, image::imageops::FilterType::Triangle);
        let arc = Arc::new(thumb);
        self.lod_cache = Some(Arc::clone(&arc));
        arc
    }
}

/// Mirror symmetry mode for the canvas.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum MirrorMode {
    #[default]
    None,
    /// Left↔Right symmetry (vertical axis at center)
    Horizontal,
    /// Top↔Bottom symmetry (horizontal axis at center)
    Vertical,
    /// 4-way symmetry (both axes)
    Quarters,
}

impl MirrorMode {
    /// Cycle to the next mode.
    pub fn next(self) -> Self {
        match self {
            MirrorMode::None => MirrorMode::Horizontal,
            MirrorMode::Horizontal => MirrorMode::Vertical,
            MirrorMode::Vertical => MirrorMode::Quarters,
            MirrorMode::Quarters => MirrorMode::None,
        }
    }

    pub fn is_active(self) -> bool {
        self != MirrorMode::None
    }

    /// Produce mirrored positions for a given canvas coordinate.
    /// Always includes the original position first.
    /// Uses ArrayVec-style inline storage (no heap allocation).
    pub fn mirror_positions(self, x: f32, y: f32, w: u32, h: u32) -> MirrorPositions {
        let wf = w as f32 - 1.0;
        let hf = h as f32 - 1.0;
        match self {
            MirrorMode::None => MirrorPositions { data: [(x, y), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)], len: 1 },
            MirrorMode::Horizontal => MirrorPositions { data: [(x, y), (wf - x, y), (0.0, 0.0), (0.0, 0.0)], len: 2 },
            MirrorMode::Vertical => MirrorPositions { data: [(x, y), (x, hf - y), (0.0, 0.0), (0.0, 0.0)], len: 2 },
            MirrorMode::Quarters => MirrorPositions { data: [(x, y), (wf - x, y), (x, hf - y), (wf - x, hf - y)], len: 4 },
        }
    }

    /// Produce mirrored positions for integer coordinates.
    pub fn mirror_positions_u32(self, x: u32, y: u32, w: u32, h: u32) -> MirrorPositionsU32 {
        let wx = w.saturating_sub(1).saturating_sub(x);
        let hy = h.saturating_sub(1).saturating_sub(y);
        match self {
            MirrorMode::None => MirrorPositionsU32 { data: [(x, y), (0, 0), (0, 0), (0, 0)], len: 1 },
            MirrorMode::Horizontal => MirrorPositionsU32 { data: [(x, y), (wx, y), (0, 0), (0, 0)], len: 2 },
            MirrorMode::Vertical => MirrorPositionsU32 { data: [(x, y), (x, hy), (0, 0), (0, 0)], len: 2 },
            MirrorMode::Quarters => MirrorPositionsU32 { data: [(x, y), (wx, y), (x, hy), (wx, hy)], len: 4 },
        }
    }
}

/// Inline array of up to 4 mirrored positions (no heap allocation).
pub struct MirrorPositions {
    pub data: [(f32, f32); 4],
    pub len: usize,
}

impl MirrorPositions {
    pub fn iter(&self) -> impl Iterator<Item = &(f32, f32)> {
        self.data[..self.len].iter()
    }
}

/// Inline array of up to 4 mirrored positions (integer coordinates).
pub struct MirrorPositionsU32 {
    pub data: [(u32, u32); 4],
    pub len: usize,
}

impl MirrorPositionsU32 {
    pub fn iter(&self) -> impl Iterator<Item = &(u32, u32)> {
        self.data[..self.len].iter()
    }
}

pub struct CanvasState {
    pub layers: Vec<Layer>,
    pub active_layer_index: usize,
    pub width: u32,
    pub height: u32,
    pub composite_cache: Option<egui::TextureHandle>,
    pub dirty_rect: Option<egui::Rect>,
    pub show_pixel_grid: bool, // Toggle for pixel grid overlay
    pub show_guidelines: bool, // Toggle for center/thirds guidelines overlay
    pub mirror_mode: MirrorMode, // Symmetry mirror mode
    pub preview_layer: Option<TiledImage>, // For non-destructive tool previews (e.g., Bézier curves)
    pub preview_blend_mode: BlendMode, // Blend mode for the preview layer
    /// When true, forces the blend-aware composite path for the preview overlay,
    /// even when blend modes are Normal.  Needed for tools like Fill that write
    /// semi-transparent pixels which must be composited properly with the layer stack.
    pub preview_force_composite: bool,
    /// When true, the preview layer is an eraser mask: each pixel's alpha represents
    /// how much to erase (reduce alpha) from the active layer during compositing.
    /// The mask uses max-alpha stamping to prevent opacity stacking.
    pub preview_is_eraser: bool,
    /// When true, the preview layer fully replaces the active layer pixels
    /// rather than blending on top.  Used by Liquify and Mesh Warp whose
    /// previews contain the entire warped layer.
    pub preview_replaces_layer: bool,
    /// Monotonically increasing counter, bumped on each mark_dirty call
    pub dirty_generation: u64,
    /// Selection mask – 0 = unselected, 255 = fully selected.
    /// Dimensions must match (width, height).
    pub selection_mask: Option<GrayImage>,
    /// LOD composite texture for zoomed-out rendering (zoom < 0.5).
    pub lod_composite_cache: Option<egui::TextureHandle>,
    /// Generation counter for LOD cache validity.
    pub lod_generation: u64,
    /// Dirty rect for preview layer only (CPU-side, no GPU involvement)
    pub preview_dirty_rect: Option<egui::Rect>,
    /// CPU-rendered preview overlay texture (bypasses GPU entirely during strokes)
    pub preview_texture_cache: Option<egui::TextureHandle>,
    /// Generation counter for preview texture validity
    pub preview_generation: u64,
    /// Accumulated bounding box of ALL paint in the current stroke.
    /// Used to crop the preview texture to just the painted area instead
    /// of uploading the entire canvas every frame.
    pub preview_stroke_bounds: Option<egui::Rect>,
    /// Persistent flat RGBA buffer for the preview overlay.
    /// Sized to `stroke_bounds` dimensions.  Reused across frames
    /// to avoid per-frame allocation.
    pub preview_flat_buffer: Vec<u8>,
    /// When true, `preview_flat_buffer` already contains premultiplied RGBA
    /// data at full canvas size, ready for direct texture upload.
    /// Skips the TiledImage → extract → premultiply pipeline entirely.
    pub preview_flat_ready: bool,
    /// Downscale factor for the preview layer during interactive drag at high
    /// resolution.  1 = full resolution, 2 = half, etc.  When > 1 the
    /// preview_layer is at reduced dimensions and composite_partial_downscaled
    /// is used for the blend-aware display path.
    pub preview_downscale: u32,

    // -- Persistent GPU composite CPU buffer ----------------
    /// Full-canvas Color32 buffer mirroring the GPU composite result.
    /// Updated incrementally via dirty-rect readback (Plan E).
    /// Prevents full-canvas readback on every stroke commit.
    pub composite_cpu_buffer: Vec<Color32>,
    /// Back buffer for zero-copy swap with composite_cpu_buffer during upload.
    pub composite_cpu_buffer_back: Vec<Color32>,
    /// Reusable buffer for GPU partial texture uploads.
    pub region_extract_buf: Vec<u8>,
    /// Reusable buffer for composite_layers_above (avoids 33MB alloc per frame during paste).
    pub composite_above_buffer: Vec<Color32>,

    // -- Incremental preview cache (brush fast path) ----------
    /// Persistent premultiplied Color32 cache covering the current
    /// `preview_stroke_bounds`.  Updated incrementally: only the
    /// per-frame dirty rect is re-extracted and blitted in, reducing
    /// per-frame work from O(stroke_bounds) to O(brush_size).
    pub preview_premul_cache: Vec<Color32>,
    /// The (rx, ry, rw, rh) rectangle that `preview_premul_cache` covers.
    pub preview_cache_rect: Option<(u32, u32, u32, u32)>,

    // -- Selection overlay GPU cache ----------------------
    /// Cached selection overlay texture (crosshatch pattern + border glow).
    /// Rebuilt only when selection_mask changes or animation ticks.
    pub selection_overlay_texture: Option<egui::TextureHandle>,
    /// Generation counter — bumped whenever the selection mask changes.
    pub selection_overlay_generation: u64,
    /// The generation value that was last used to build the cached texture.
    pub selection_overlay_built_generation: u64,
    /// Last animation offset baked into the cached texture.
    pub selection_overlay_anim_offset: f32,
    /// Bounding box of the selected region in canvas coordinates
    /// (used to position the overlay texture).
    pub selection_overlay_bounds: Option<(u32, u32, u32, u32)>,
    /// Cached border segments in canvas coordinates: (line_pos, seg_start, seg_end).
    /// Horizontal segments: y_line, x_start, x_end.
    /// Vertical segments: x_line, y_start, y_end.
    pub selection_border_h_segs: Vec<(u32, u32, u32)>,
    pub selection_border_v_segs: Vec<(u32, u32, u32)>,
    /// Generation at which border segments were last computed.
    pub selection_border_built_generation: u64,
}

impl CanvasState {
    pub fn new(width: u32, height: u32) -> Self {
        let white = Rgba([255, 255, 255, 255]);
        let background = Layer::new("Background".to_string(), width, height, white);
        
        Self {
            layers: vec![background],
            active_layer_index: 0,
            width,
            height,
            composite_cache: None,
            dirty_rect: None,
            show_pixel_grid: true, // Enable by default
            show_guidelines: false, // Disabled by default
            mirror_mode: MirrorMode::None,
            preview_layer: None,
            preview_blend_mode: BlendMode::Normal,
            preview_force_composite: false,
            preview_is_eraser: false,
            preview_replaces_layer: false,
            dirty_generation: 0,
            selection_mask: None,
            lod_composite_cache: None,
            lod_generation: 0,
            preview_dirty_rect: None,
            preview_texture_cache: None,
            preview_generation: 0,
            preview_stroke_bounds: None,
            preview_flat_buffer: Vec::new(),
            preview_flat_ready: false,
            preview_downscale: 1,
            composite_cpu_buffer: Vec::new(),
            composite_cpu_buffer_back: Vec::new(),
            region_extract_buf: Vec::new(),
            composite_above_buffer: Vec::new(),
            preview_premul_cache: Vec::new(),
            preview_cache_rect: None,
            selection_overlay_texture: None,
            selection_overlay_generation: 0,
            selection_overlay_built_generation: 0,
            selection_overlay_anim_offset: -1.0,
            selection_overlay_bounds: None,
            selection_border_h_segs: Vec::new(),
            selection_border_v_segs: Vec::new(),
            selection_border_built_generation: u64::MAX, // force first compute
        }
    }

    /// Reset all preview-related state (call whenever preview_layer is cleared).
    pub fn clear_preview_state(&mut self) {
        self.preview_layer = None;
        self.preview_dirty_rect = None;
        self.preview_texture_cache = None;
        self.preview_stroke_bounds = None;
        self.preview_flat_buffer.clear();
        self.preview_flat_ready = false;
        self.preview_downscale = 1;
        self.preview_premul_cache.clear();
        self.preview_cache_rect = None;
        self.preview_force_composite = false;
        self.preview_is_eraser = false;
        self.preview_replaces_layer = false;
    }

    /// Mirror all populated pixels in the preview layer according to the
    /// current mirror mode.  Called after generating preview content for
    /// tools like Fill, Text, Shapes, Gradient, etc.
    /// Returns the expanded bounding rect that covers all mirrored regions.
    pub fn mirror_preview_layer(&mut self) -> Option<egui::Rect> {
        let mode = self.mirror_mode;
        if mode == MirrorMode::None { return None; }
        let w = self.width;
        let h = self.height;
        
        let preview = match self.preview_layer.as_mut() {
            Some(p) => p,
            None => return None,
        };
        
        // Collect all existing non-empty pixels from populated chunks
        let chunk_data: Vec<(u32, u32, Vec<(u32, u32, image::Rgba<u8>)>)> = preview.chunk_keys()
            .filter_map(|(cx, cy)| {
                preview.get_chunk(cx, cy).map(|chunk| {
                    let base_x = cx * CHUNK_SIZE;
                    let base_y = cy * CHUNK_SIZE;
                    let cw = CHUNK_SIZE.min(w.saturating_sub(base_x));
                    let ch = CHUNK_SIZE.min(h.saturating_sub(base_y));
                    let mut pixels = Vec::new();
                    for ly in 0..ch {
                        for lx in 0..cw {
                            let px = *chunk.get_pixel(lx, ly);
                            if px[3] > 0 {
                                pixels.push((base_x + lx, base_y + ly, px));
                            }
                        }
                    }
                    (cx, cy, pixels)
                })
            })
            .collect();
        
        let mut expanded_bounds: Option<egui::Rect> = None;
        
        // Write mirrored pixels (skip arm 0 which is the original)
        for (_cx, _cy, pixels) in &chunk_data {
            for &(gx, gy, px) in pixels {
                let mirrors = mode.mirror_positions_u32(gx, gy, w, h);
                // Start from arm 1 to skip the original position
                for i in 1..mirrors.len {
                    let (mx, my) = mirrors.data[i];
                    if mx < w && my < h {
                        // Use max-alpha stamping to avoid double-writes at the center axis
                        let existing = preview.get_pixel(mx, my);
                        if px[3] > existing[3] {
                            preview.put_pixel(mx, my, px);
                        }
                        // Expand bounds
                        let r = egui::Rect::from_min_max(
                            egui::pos2(mx as f32, my as f32),
                            egui::pos2(mx as f32 + 1.0, my as f32 + 1.0),
                        );
                        expanded_bounds = Some(match expanded_bounds {
                            Some(b) => b.union(r),
                            None => r,
                        });
                    }
                }
            }
        }
        
        expanded_bounds
    }
    
    pub fn composite(&self) -> RgbaImage {
        self.composite_viewport(None)
    }

    /// Produce a downscaled composite for LOD rendering (max 1024px longest edge).
    /// Uses the full composite then resizes for simplicity; the result is cached
    /// in `lod_composite_cache` by the rendering code.
    pub fn composite_lod(&self) -> RgbaImage {
        let full = self.composite();
        let (w, h) = (full.width(), full.height());
        let longest = w.max(h);
        if longest <= LOD_MAX_EDGE {
            return full;
        }
        let scale = LOD_MAX_EDGE as f32 / longest as f32;
        let nw = ((w as f32 * scale).round() as u32).max(1);
        let nh = ((h as f32 * scale).round() as u32).max(1);
        image::imageops::resize(&full, nw, nh, image::imageops::FilterType::Triangle)
    }
    
    /// Composite with optional viewport clipping for optimization
    /// If viewport is Some, only pixels within that rect will be computed.
    /// Uses chunk-based iteration with rayon parallelism.
    pub fn composite_viewport(&self, viewport: Option<Rect>) -> RgbaImage {
        let mut result = RgbaImage::new(self.width, self.height);
        
        // Calculate chunk range to process
        let (min_cx, min_cy, max_cx, max_cy) = if let Some(vp) = viewport {
            let min_cx = (vp.min.x.floor().max(0.0) as u32) / CHUNK_SIZE;
            let min_cy = (vp.min.y.floor().max(0.0) as u32) / CHUNK_SIZE;
            let max_cx = ((vp.max.x.ceil() as u32).min(self.width) + CHUNK_SIZE - 1) / CHUNK_SIZE;
            let max_cy = ((vp.max.y.ceil() as u32).min(self.height) + CHUNK_SIZE - 1) / CHUNK_SIZE;
            (min_cx, min_cy, max_cx, max_cy)
        } else {
            (0, 0, (self.width + CHUNK_SIZE - 1) / CHUNK_SIZE, (self.height + CHUNK_SIZE - 1) / CHUNK_SIZE)
        };
        
        // Collect unique chunk keys from visible layers within viewport
        let mut active_chunks: Vec<(u32, u32)> = Vec::new();
        for layer in &self.layers {
            if !layer.visible { continue; }
            for key in layer.pixels.chunk_keys() {
                let (cx, cy) = key;
                if cx >= min_cx && cx < max_cx && cy >= min_cy && cy < max_cy {
                    active_chunks.push(key);
                }
            }
        }
        if let Some(ref preview) = self.preview_layer {
            for key in preview.chunk_keys() {
                let (cx, cy) = key;
                if cx >= min_cx && cx < max_cx && cy >= min_cy && cy < max_cy {
                    active_chunks.push(key);
                }
            }
        }
        active_chunks.sort_unstable();
        active_chunks.dedup();
        
        let layers = &self.layers;
        let preview = &self.preview_layer;
        let preview_blend = self.preview_blend_mode;
        let preview_is_eraser = self.preview_is_eraser;
        let preview_replaces = self.preview_replaces_layer;
        let active_idx = self.active_layer_index;
        let img_w = self.width;
        let img_h = self.height;
        
        // Process chunks in parallel
        let chunk_results: Vec<_> = active_chunks.par_iter().map(|&(cx, cy)| {
            let base_x = cx * CHUNK_SIZE;
            let base_y = cy * CHUNK_SIZE;
            let cw = CHUNK_SIZE.min(img_w.saturating_sub(base_x));
            let ch = CHUNK_SIZE.min(img_h.saturating_sub(base_y));
            
            let mut pixels = vec![Rgba([0u8, 0, 0, 0]); (cw * ch) as usize];
            
            for (li, layer) in layers.iter().enumerate() {
                if !layer.visible { continue; }
                
                let is_active = li == active_idx;
                let layer_chunk = layer.pixels.get_chunk(cx, cy);
                let preview_chunk = if is_active {
                    preview.as_ref().and_then(|p| p.get_chunk(cx, cy))
                } else {
                    None
                };
                
                // Skip if neither the layer nor the preview has data here
                if layer_chunk.is_none() && preview_chunk.is_none() {
                    continue;
                }
                
                // Occlusion opt only safe when no preview is modifying pixels
                let opaque_overwrite = layer.blend_mode == BlendMode::Normal
                    && layer.opacity >= 1.0
                    && preview_chunk.is_none();
                
                for ly in 0..ch {
                    for lx in 0..cw {
                        let idx = (ly * cw + lx) as usize;
                        
                        let mut top = if let Some(chunk) = layer_chunk {
                            *chunk.get_pixel(lx, ly)
                        } else {
                            Rgba([0u8, 0, 0, 0])
                        };
                        
                        // Apply preview into the layer pixel BEFORE compositing,
                        // so the preview inherits the layer's blend mode & opacity
                        if let Some(pchunk) = preview_chunk {
                            let pp = *pchunk.get_pixel(lx, ly);
                            if preview_replaces {
                                top = pp;
                            } else if pp[3] > 0 {
                                if preview_is_eraser {
                                    // Eraser mask: reduce the layer pixel's alpha by the mask value
                                    let mask_strength = pp[3] as f32 / 255.0;
                                    let current_a = top[3] as f32 / 255.0;
                                    let new_a = (current_a * (1.0 - mask_strength)).max(0.0);
                                    top[3] = (new_a * 255.0) as u8;
                                } else {
                                    top = Self::blend_pixel_static(top, pp, preview_blend, 1.0);
                                }
                            }
                        }
                        
                        if opaque_overwrite && top[3] == 255 {
                            pixels[idx] = top;
                        } else {
                            pixels[idx] = Self::blend_pixel_static(pixels[idx], top, layer.blend_mode, layer.opacity);
                        }
                    }
                }
            }
            
            (cx, cy, cw, ch, pixels)
        }).collect();
        
        // Write results to output image
        for (cx, cy, cw, ch, pixels) in chunk_results {
            let base_x = cx * CHUNK_SIZE;
            let base_y = cy * CHUNK_SIZE;
            for ly in 0..ch {
                for lx in 0..cw {
                    let idx = (ly * cw + lx) as usize;
                    result.put_pixel(base_x + lx, base_y + ly, pixels[idx]);
                }
            }
        }
        
        result
    }

    /// Downscaled version of `composite_partial` for interactive previews at
    /// high resolution.  Produces a `ColorImage` at `(target_w × target_h)`
    /// by sampling every `scale`-th pixel from the layer stack and the preview
    /// layer (which is already at the reduced resolution).
    ///
    /// The caller paints the resulting texture at the full `image_rect` — egui
    /// stretches it to fill.
    pub fn composite_partial_downscaled(
        &self,
        rect: Rect,
        scale: u32,
    ) -> (ColorImage, [usize; 2]) {
        let min_x = (rect.min.x.floor() as u32).max(0);
        let min_y = (rect.min.y.floor() as u32).max(0);
        let max_x = (rect.max.x.ceil() as u32).min(self.width);
        let max_y = (rect.max.y.ceil() as u32).min(self.height);

        let source_w = max_x - min_x;
        let source_h = max_y - min_y;
        let target_w = (source_w + scale - 1) / scale;
        let target_h = (source_h + scale - 1) / scale;

        let layers = &self.layers;
        let preview_layer = &self.preview_layer;
        let preview_blend_mode = self.preview_blend_mode;
        let preview_is_eraser = self.preview_is_eraser;
        let preview_replaces = self.preview_replaces_layer;
        let active_layer_index = self.active_layer_index;

        let row_len = target_w as usize;
        let mut pixels = vec![Color32::TRANSPARENT; (target_w * target_h) as usize];

        pixels
            .par_chunks_mut(row_len)
            .enumerate()
            .for_each(|(row_idx, row)| {
                // Map output row → source canvas coordinate
                let y = min_y + (row_idx as u32) * scale;
                for out_x in 0..target_w as usize {
                    let x = min_x + (out_x as u32) * scale;

                    // Opaque-base optimisation: find the deepest fully-opaque
                    // normal-blend pixel to skip layers beneath it
                    let mut start_layer_idx = 0;
                    for (idx, layer) in layers.iter().enumerate().rev() {
                        if !layer.visible { continue; }
                        if preview_is_eraser && preview_layer.is_some()
                            && idx == active_layer_index
                        {
                            continue;
                        }
                        if layer.blend_mode == BlendMode::Normal && layer.opacity >= 1.0 {
                            let pixel = layer.pixels.get_pixel(x, y);
                            if pixel[3] == 255 {
                                start_layer_idx = idx;
                                break;
                            }
                        }
                    }

                    let mut base = Rgba([0, 0, 0, 0]);

                    for (li, layer) in layers.iter().enumerate().skip(start_layer_idx) {
                        if !layer.visible { continue; }
                        let mut top = *layer.pixels.get_pixel(x, y);

                        if li == active_layer_index {
                            if let Some(preview) = preview_layer {
                                // Preview is at reduced resolution — sample
                                // at the output coordinate directly.
                                let pp = *preview.get_pixel(out_x as u32, row_idx as u32);
                                if preview_replaces {
                                    top = pp;
                                } else if pp[3] > 0 {
                                    if preview_is_eraser {
                                        let mask_strength = pp[3] as f32 / 255.0;
                                        let current_a = top[3] as f32 / 255.0;
                                        let new_a = (current_a * (1.0 - mask_strength)).max(0.0);
                                        top[3] = (new_a * 255.0) as u8;
                                    } else {
                                        top = Self::blend_pixel_static(top, pp, preview_blend_mode, 1.0);
                                    }
                                }
                            }
                        }

                        base = Self::blend_pixel_static(base, top, layer.blend_mode, layer.opacity);
                    }

                    let a = base[3];
                    if a == 255 || a == 0 {
                        row[out_x] = Color32::from_rgba_premultiplied(base[0], base[1], base[2], a);
                    } else {
                        let pm_r = ((base[0] as u16 * a as u16 + 127) / 255) as u8;
                        let pm_g = ((base[1] as u16 * a as u16 + 127) / 255) as u8;
                        let pm_b = ((base[2] as u16 * a as u16 + 127) / 255) as u8;
                        row[out_x] = Color32::from_rgba_premultiplied(pm_r, pm_g, pm_b, a);
                    }
                }
            });

        (
            ColorImage {
                size: [target_w as usize, target_h as usize],
                pixels,
            },
            [min_x as usize, min_y as usize]
        )
    }
    
    /// Render ONLY the modified area for partial texture updates.
    /// Processes chunk-by-chunk within the dirty rect for efficient batch access.
    pub fn composite_partial(&self, rect: Rect) -> (ColorImage, [usize; 2]) {
        // 1. Clamp rect to canvas bounds
        let min_x = (rect.min.x.floor() as u32).max(0);
        let min_y = (rect.min.y.floor() as u32).max(0);
        let max_x = (rect.max.x.ceil() as u32).min(self.width);
        let max_y = (rect.max.y.ceil() as u32).min(self.height);
        
        let width = max_x - min_x;
        let height = max_y - min_y;

        // Capture shared state for the parallel closure
        let layers = &self.layers;
        let preview_layer = &self.preview_layer;
        let preview_blend_mode = self.preview_blend_mode;
        let preview_is_eraser = self.preview_is_eraser;
        let preview_replaces = self.preview_replaces_layer;
        let active_layer_index = self.active_layer_index;

        // 2. Pre-allocate output and parallelise row-by-row in-place
        let row_len = width as usize;
        let mut pixels = vec![Color32::TRANSPARENT; (width * height) as usize];

        // Chunk column range covering [min_x, max_x)
        let first_cx = min_x / CHUNK_SIZE;
        let last_cx = max_x.saturating_sub(1) / CHUNK_SIZE;
        let n_layers = layers.len();

        pixels
            .par_chunks_mut(row_len)
            .enumerate()
            .for_each(|(row_idx, row)| {
                let y = min_y + row_idx as u32;
                let cy = y / CHUNK_SIZE;
                let ly = y % CHUNK_SIZE;

                // Per-row buffer for chunk raw pointers (reused across chunk columns)
                let mut chunk_raws: Vec<Option<&[u8]>> = Vec::with_capacity(n_layers);

                for cx in first_cx..=last_cx {
                    let span_start = (cx * CHUNK_SIZE).max(min_x);
                    let span_end = ((cx + 1) * CHUNK_SIZE).min(max_x);

                    // Pre-fetch chunk raw data for ALL layers at (cx, cy) — O(layers) per chunk column
                    chunk_raws.clear();
                    for layer in layers.iter() {
                        chunk_raws.push(
                            layer.pixels.get_chunk(cx, cy).map(|c| c.as_raw().as_slice())
                        );
                    }

                    // Pre-fetch preview layer chunk
                    let preview_raw = preview_layer.as_ref().and_then(|p| {
                        p.get_chunk(cx, cy).map(|c| c.as_raw().as_slice())
                    });

                    let row_byte_off = (ly * CHUNK_SIZE * 4) as usize;

                    for x in span_start..span_end {
                        let lx = x - cx * CHUNK_SIZE;
                        let px_off = row_byte_off + (lx as usize) * 4;
                        let out_idx = (x - min_x) as usize;

                        // Opaque-base optimisation: find deepest opaque Normal pixel
                        let mut start_layer_idx = 0;
                        for (idx, layer) in layers.iter().enumerate().rev() {
                            if !layer.visible { continue; }
                            if preview_is_eraser && preview_layer.is_some()
                                && idx == active_layer_index
                            {
                                continue;
                            }
                            if layer.blend_mode == BlendMode::Normal && layer.opacity >= 1.0 {
                                if let Some(raw) = chunk_raws[idx] {
                                    if raw[px_off + 3] == 255 {
                                        start_layer_idx = idx;
                                        break;
                                    }
                                }
                            }
                        }

                        let mut base = Rgba([0, 0, 0, 0]);

                        for (li, layer) in layers.iter().enumerate().skip(start_layer_idx) {
                            if !layer.visible { continue; }
                            let mut top = match chunk_raws[li] {
                                Some(raw) => Rgba([raw[px_off], raw[px_off+1], raw[px_off+2], raw[px_off+3]]),
                                None => Rgba([0, 0, 0, 0]),
                            };

                            if li == active_layer_index {
                                if preview_layer.is_some() {
                                    let pp = match preview_raw {
                                        Some(raw) => Rgba([raw[px_off], raw[px_off+1], raw[px_off+2], raw[px_off+3]]),
                                        None => Rgba([0, 0, 0, 0]),
                                    };
                                    if preview_replaces {
                                        top = pp;
                                    } else if pp[3] > 0 {
                                        if preview_is_eraser {
                                            let mask_strength = pp[3] as f32 / 255.0;
                                            let current_a = top[3] as f32 / 255.0;
                                            let new_a = (current_a * (1.0 - mask_strength)).max(0.0);
                                            top[3] = (new_a * 255.0) as u8;
                                        } else {
                                            top = Self::blend_pixel_static(top, pp, preview_blend_mode, 1.0);
                                        }
                                    }
                                }
                            }

                            base = Self::blend_pixel_static(base, top, layer.blend_mode, layer.opacity);
                        }

                        let a = base[3];
                        if a == 255 || a == 0 {
                            row[out_idx] = Color32::from_rgba_premultiplied(base[0], base[1], base[2], a);
                        } else {
                            let pm_r = ((base[0] as u16 * a as u16 + 127) / 255) as u8;
                            let pm_g = ((base[1] as u16 * a as u16 + 127) / 255) as u8;
                            let pm_b = ((base[2] as u16 * a as u16 + 127) / 255) as u8;
                            row[out_idx] = Color32::from_rgba_premultiplied(pm_r, pm_g, pm_b, a);
                        }
                    }
                }
            });

        (
            ColorImage {
                size: [width as usize, height as usize],
                pixels,
            },
            [min_x as usize, min_y as usize]
        )
    }

    /// Composite only the visible layers ABOVE `active_layer_index` against a
    /// transparent background.  Returns a premultiplied RGBA `ColorImage`
    /// covering the full canvas (or `None` if there are no visible layers above).
    /// Used to overlay layers-above on top of a paste preview.
    pub fn composite_layers_above(&mut self) -> Option<ColorImage> {
        let above_start = self.active_layer_index + 1;
        let has_any = self.layers.iter().skip(above_start).any(|l| l.visible);
        if !has_any {
            return None;
        }

        let w = self.width as usize;
        let h = self.height as usize;
        let needed = w * h;
        // A12: reuse persistent buffer instead of allocating per frame
        self.composite_above_buffer.resize(needed, Color32::TRANSPARENT);
        self.composite_above_buffer.fill(Color32::TRANSPARENT);

        self.composite_above_buffer
            .par_chunks_mut(w)
            .enumerate()
            .for_each(|(y, row)| {
                for x in 0..w {
                    let mut base = Rgba([0u8, 0, 0, 0]);
                    for layer in self.layers.iter().skip(above_start) {
                        if !layer.visible { continue; }
                        let top = *layer.pixels.get_pixel(x as u32, y as u32);
                        base = Self::blend_pixel_static(base, top, layer.blend_mode, layer.opacity);
                    }
                    let a = base[3];
                    if a == 0 {
                        row[x] = Color32::TRANSPARENT;
                    } else if a == 255 {
                        row[x] = Color32::from_rgba_premultiplied(base[0], base[1], base[2], 255);
                    } else {
                        let pm_r = ((base[0] as u16 * a as u16 + 127) / 255) as u8;
                        let pm_g = ((base[1] as u16 * a as u16 + 127) / 255) as u8;
                        let pm_b = ((base[2] as u16 * a as u16 + 127) / 255) as u8;
                        row[x] = Color32::from_rgba_premultiplied(pm_r, pm_g, pm_b, a);
                    }
                }
            });

        let pixels = self.composite_above_buffer.clone();
        Some(ColorImage { size: [w, h], pixels })
    }

    fn blend_pixel(&self, base: Rgba<u8>, top: Rgba<u8>, mode: BlendMode, opacity: f32) -> Rgba<u8> {
        Self::blend_pixel_static(base, top, mode, opacity)
    }
    
    /// Static version of blend_pixel for use when self is borrowed mutably elsewhere
    pub fn blend_pixel_static(base: Rgba<u8>, top: Rgba<u8>, mode: BlendMode, opacity: f32) -> Rgba<u8> {
        // Fast path: fully transparent top pixel — nothing to blend
        if top[3] == 0 {
            return base;
        }

        // Fast path: Normal blend, full opacity, fully opaque top pixel — just overwrite
        if matches!(mode, BlendMode::Normal) && opacity >= 1.0 && top[3] == 255 {
            return top;
        }

        let opacity = opacity.clamp(0.0, 1.0);
        
        let base_r = base[0] as f32 / 255.0;
        let base_g = base[1] as f32 / 255.0;
        let base_b = base[2] as f32 / 255.0;
        let base_a = base[3] as f32 / 255.0;
        
        let top_r = top[0] as f32 / 255.0;
        let top_g = top[1] as f32 / 255.0;
        let top_b = top[2] as f32 / 255.0;
        let top_a = (top[3] as f32 / 255.0) * opacity;
        
        match mode {
            BlendMode::Overwrite => {
                return Rgba([
                    (top_r * 255.0) as u8,
                    (top_g * 255.0) as u8,
                    (top_b * 255.0) as u8,
                    (top_a * 255.0) as u8,
                ]);
            }
            BlendMode::Xor => {
                let xor_a = base_a * (1.0 - top_a) + top_a * (1.0 - base_a);
                if xor_a == 0.0 {
                    return Rgba([0, 0, 0, 0]);
                }
                let xor_r = (base_r * base_a * (1.0 - top_a) + top_r * top_a * (1.0 - base_a)) / xor_a;
                let xor_g = (base_g * base_a * (1.0 - top_a) + top_g * top_a * (1.0 - base_a)) / xor_a;
                let xor_b = (base_b * base_a * (1.0 - top_a) + top_b * top_a * (1.0 - base_a)) / xor_a;
                return Rgba([
                    (xor_r * 255.0).clamp(0.0, 255.0) as u8,
                    (xor_g * 255.0).clamp(0.0, 255.0) as u8,
                    (xor_b * 255.0).clamp(0.0, 255.0) as u8,
                    (xor_a * 255.0).clamp(0.0, 255.0) as u8,
                ]);
            }
            _ => {}
        }
        
        let (r, g, b) = match mode {
            BlendMode::Normal => (top_r, top_g, top_b),
            BlendMode::Multiply => (base_r * top_r, base_g * top_g, base_b * top_b),
            BlendMode::Screen => (
                1.0 - (1.0 - base_r) * (1.0 - top_r),
                1.0 - (1.0 - base_g) * (1.0 - top_g),
                1.0 - (1.0 - base_b) * (1.0 - top_b),
            ),
            BlendMode::Additive => (
                (base_r + top_r).min(1.0),
                (base_g + top_g).min(1.0),
                (base_b + top_b).min(1.0),
            ),
            BlendMode::Overlay => (
                Self::overlay_channel(base_r, top_r),
                Self::overlay_channel(base_g, top_g),
                Self::overlay_channel(base_b, top_b),
            ),
            BlendMode::Lighten => (base_r.max(top_r), base_g.max(top_g), base_b.max(top_b)),
            BlendMode::Darken => (base_r.min(top_r), base_g.min(top_g), base_b.min(top_b)),
            BlendMode::Difference => ((base_r - top_r).abs(), (base_g - top_g).abs(), (base_b - top_b).abs()),
            BlendMode::Negation => (
                1.0 - (1.0 - base_r - top_r).abs(),
                1.0 - (1.0 - base_g - top_g).abs(),
                1.0 - (1.0 - base_b - top_b).abs(),
            ),
            BlendMode::ColorBurn => (
                Self::color_burn_channel(base_r, top_r),
                Self::color_burn_channel(base_g, top_g),
                Self::color_burn_channel(base_b, top_b),
            ),
            BlendMode::ColorDodge => (
                Self::color_dodge_channel(base_r, top_r),
                Self::color_dodge_channel(base_g, top_g),
                Self::color_dodge_channel(base_b, top_b),
            ),
            BlendMode::Reflect => (
                Self::reflect_channel(base_r, top_r),
                Self::reflect_channel(base_g, top_g),
                Self::reflect_channel(base_b, top_b),
            ),
            BlendMode::Glow => (
                Self::reflect_channel(top_r, base_r),
                Self::reflect_channel(top_g, base_g),
                Self::reflect_channel(top_b, base_b),
            ),
            BlendMode::HardLight => (
                Self::overlay_channel(top_r, base_r),
                Self::overlay_channel(top_g, base_g),
                Self::overlay_channel(top_b, base_b),
            ),
            BlendMode::SoftLight => (
                Self::soft_light_channel(base_r, top_r),
                Self::soft_light_channel(base_g, top_g),
                Self::soft_light_channel(base_b, top_b),
            ),
            BlendMode::Exclusion => (
                base_r + top_r - 2.0 * base_r * top_r,
                base_g + top_g - 2.0 * base_g * top_g,
                base_b + top_b - 2.0 * base_b * top_b,
            ),
            BlendMode::Subtract => (
                (base_r - top_r).max(0.0),
                (base_g - top_g).max(0.0),
                (base_b - top_b).max(0.0),
            ),
            BlendMode::Divide => (
                Self::divide_channel(base_r, top_r),
                Self::divide_channel(base_g, top_g),
                Self::divide_channel(base_b, top_b),
            ),
            BlendMode::LinearBurn => (
                (base_r + top_r - 1.0).max(0.0),
                (base_g + top_g - 1.0).max(0.0),
                (base_b + top_b - 1.0).max(0.0),
            ),
            BlendMode::VividLight => (
                Self::vivid_light_channel(base_r, top_r),
                Self::vivid_light_channel(base_g, top_g),
                Self::vivid_light_channel(base_b, top_b),
            ),
            BlendMode::LinearLight => (
                (base_r + 2.0 * top_r - 1.0).clamp(0.0, 1.0),
                (base_g + 2.0 * top_g - 1.0).clamp(0.0, 1.0),
                (base_b + 2.0 * top_b - 1.0).clamp(0.0, 1.0),
            ),
            BlendMode::PinLight => (
                Self::pin_light_channel(base_r, top_r),
                Self::pin_light_channel(base_g, top_g),
                Self::pin_light_channel(base_b, top_b),
            ),
            BlendMode::HardMix => (
                if base_r + top_r >= 1.0 { 1.0 } else { 0.0 },
                if base_g + top_g >= 1.0 { 1.0 } else { 0.0 },
                if base_b + top_b >= 1.0 { 1.0 } else { 0.0 },
            ),
            BlendMode::Xor | BlendMode::Overwrite => unreachable!(),
        };
        
        let out_a = top_a + base_a * (1.0 - top_a);
        if out_a == 0.0 {
            return Rgba([0, 0, 0, 0]);
        }
        
        let out_r = (r * top_a + base_r * base_a * (1.0 - top_a)) / out_a;
        let out_g = (g * top_a + base_g * base_a * (1.0 - top_a)) / out_a;
        let out_b = (b * top_a + base_b * base_a * (1.0 - top_a)) / out_a;
        
        Rgba([
            (out_r * 255.0).clamp(0.0, 255.0) as u8,
            (out_g * 255.0).clamp(0.0, 255.0) as u8,
            (out_b * 255.0).clamp(0.0, 255.0) as u8,
            (out_a * 255.0).clamp(0.0, 255.0) as u8,
        ])
    }
    
    // Blend mode helper functions
    fn overlay_channel(base: f32, top: f32) -> f32 {
        if base < 0.5 {
            2.0 * base * top
        } else {
            1.0 - 2.0 * (1.0 - base) * (1.0 - top)
        }
    }
    
    fn color_burn_channel(base: f32, top: f32) -> f32 {
        if top == 0.0 {
            0.0
        } else {
            (1.0 - (1.0 - base) / top).max(0.0)
        }
    }
    
    fn color_dodge_channel(base: f32, top: f32) -> f32 {
        if top >= 1.0 {
            1.0
        } else {
            (base / (1.0 - top)).min(1.0)
        }
    }
    
    fn reflect_channel(base: f32, top: f32) -> f32 {
        if top >= 1.0 {
            1.0
        } else {
            (base * base / (1.0 - top)).min(1.0)
        }
    }

    /// W3C Soft Light formula.
    fn soft_light_channel(base: f32, top: f32) -> f32 {
        if top <= 0.5 {
            base - (1.0 - 2.0 * top) * base * (1.0 - base)
        } else {
            let d = if base <= 0.25 {
                ((16.0 * base - 12.0) * base + 4.0) * base
            } else {
                base.sqrt()
            };
            base + (2.0 * top - 1.0) * (d - base)
        }
    }

    fn divide_channel(base: f32, top: f32) -> f32 {
        if top <= 0.0 { 1.0 } else { (base / top).min(1.0) }
    }

    fn vivid_light_channel(base: f32, top: f32) -> f32 {
        if top <= 0.5 {
            // Color Burn with 2*top
            let t2 = 2.0 * top;
            if t2 <= 0.0 { 0.0 } else { (1.0 - (1.0 - base) / t2).max(0.0) }
        } else {
            // Color Dodge with 2*(top-0.5)
            let t2 = 2.0 * (top - 0.5);
            if t2 >= 1.0 { 1.0 } else { (base / (1.0 - t2)).min(1.0) }
        }
    }

    fn pin_light_channel(base: f32, top: f32) -> f32 {
        if top <= 0.5 {
            base.min(2.0 * top)
        } else {
            base.max(2.0 * (top - 0.5))
        }
    }
    
    pub fn get_active_layer_mut(&mut self) -> Option<&mut Layer> {
        self.layers.get_mut(self.active_layer_index)
    }
    
    pub fn mark_dirty(&mut self, rect: Option<egui::Rect>) {
        let full = egui::Rect::from_min_max(
            egui::pos2(0.0, 0.0),
            egui::pos2(self.width as f32, self.height as f32),
        );
        let new_rect = rect.unwrap_or(full);
        // Merge with any existing dirty rect so we never lose pending updates
        self.dirty_rect = Some(match self.dirty_rect {
            Some(existing) => existing.union(new_rect),
            None => new_rect,
        });
        self.dirty_generation = self.dirty_generation.wrapping_add(1);

        // Per-layer GPU generation tracking: if a specific rect is provided,
        // only the active layer changed.  If None (full-image dirty), bump
        // ALL layers (e.g. flatten, resize, filter that changed everything).
        if rect.is_some() {
            // Only the active layer was modified (brush stroke, etc.)
            if let Some(layer) = self.layers.get_mut(self.active_layer_index) {
                layer.gpu_generation = layer.gpu_generation.wrapping_add(1);
            }
        } else {
            // Full-image operation — every layer may have changed
            for layer in &mut self.layers {
                layer.gpu_generation = layer.gpu_generation.wrapping_add(1);
            }
        }

        // A8: Only invalidate LOD cache for the layer(s) that actually changed
        if rect.is_some() {
            // Only the active layer was modified — only its thumbnail is stale
            if let Some(layer) = self.layers.get_mut(self.active_layer_index) {
                layer.invalidate_lod();
            }
        } else {
            // Full-image operation — all layer thumbnails may be stale
            for layer in &mut self.layers {
                layer.invalidate_lod();
            }
        }
    }
    
    /// Mark that only the preview layer changed.
    /// This is CPU-only: it does NOT trigger GPU recomposite.
    /// The preview layer is rendered as a separate egui texture overlay.
    pub fn mark_preview_changed_rect(&mut self, rect: egui::Rect) {
        // Accumulate preview dirty rect for CPU texture upload
        self.preview_dirty_rect = Some(match self.preview_dirty_rect {
            Some(existing) => existing.union(rect),
            None => rect,
        });
        // Also expand the stroke-lifetime bounding box (union of ALL frames)
        self.preview_stroke_bounds = Some(match self.preview_stroke_bounds {
            Some(existing) => existing.union(rect),
            None => rect,
        });
        self.preview_generation = self.preview_generation.wrapping_add(1);
        // NOTE: We do NOT set dirty_rect here. The GPU composite is unchanged.
        // The preview is rendered as a CPU-side overlay on top of the cached
        // GPU composite. This eliminates ALL GPU work during brush strokes.
    }
    
    /// Mark that only the preview layer changed (full layer, for backwards compatibility)
    pub fn mark_preview_changed(&mut self) {
        let full = egui::Rect::from_min_max(
            egui::pos2(0.0, 0.0),
            egui::pos2(self.width as f32, self.height as f32),
        );
        self.mark_preview_changed_rect(full);
    }

    // ========================================================================
    // SELECTION HELPERS
    // ========================================================================

    /// Clear (remove) the current selection.
    pub fn clear_selection(&mut self) {
        self.selection_mask = None;
        self.invalidate_selection_overlay();
        self.selection_overlay_texture = None;
        self.selection_overlay_bounds = None;
    }

    /// Invalidate the cached selection overlay texture so it gets rebuilt
    /// on the next frame.  Call this whenever `selection_mask` changes.
    pub fn invalidate_selection_overlay(&mut self) {
        self.selection_overlay_generation = self.selection_overlay_generation.wrapping_add(1);
    }

    /// Returns `true` when there is an active selection mask.
    pub fn has_selection(&self) -> bool {
        self.selection_mask.is_some()
    }

    /// Translate the selection mask by (dx, dy) pixels.
    /// The mask is shifted; pixels that move off-canvas are clipped,
    /// and newly-exposed areas are unselected (0).
    pub fn translate_selection(&mut self, dx: i32, dy: i32) {
        let mask = match self.selection_mask.as_ref() {
            Some(m) => m,
            None => return,
        };
        if dx == 0 && dy == 0 { return; }

        let w = mask.width();
        let h = mask.height();
        let mut new_mask = GrayImage::new(w, h);

        for y in 0..h {
            for x in 0..w {
                let src_x = x as i32 - dx;
                let src_y = y as i32 - dy;
                if src_x >= 0 && src_x < w as i32 && src_y >= 0 && src_y < h as i32 {
                    let v = mask.get_pixel(src_x as u32, src_y as u32).0[0];
                    if v > 0 {
                        new_mask.put_pixel(x, y, Luma([v]));
                    }
                }
            }
        }

        self.selection_mask = Some(new_mask);
        self.invalidate_selection_overlay();
        self.mark_dirty(None);
    }

    /// Apply a selection shape to the mask according to `mode`.
    pub fn apply_selection_shape(&mut self, shape: &SelectionShape, mode: SelectionMode) {
        let w = self.width;
        let h = self.height;

        // Ensure a mask exists (for Add/Subtract we start from the current one).
        let mask = self.selection_mask.get_or_insert_with(|| GrayImage::new(w, h));

        // Resize if canvas dimensions changed.
        if mask.width() != w || mask.height() != h {
            *mask = GrayImage::new(w, h);
        }

        let (bx0, by0, bx1, by1) = shape.bounds(w, h);

        match mode {
            SelectionMode::Replace => {
                // Zero the whole mask first, then fill the shape.
                for p in mask.pixels_mut() {
                    *p = Luma([0]);
                }
                for y in by0..=by1 {
                    for x in bx0..=bx1 {
                        let v = shape.contains(x, y);
                        if v > 0 {
                            mask.put_pixel(x, y, Luma([v]));
                        }
                    }
                }
            }
            SelectionMode::Add => {
                for y in by0..=by1 {
                    for x in bx0..=bx1 {
                        let new_val = shape.contains(x, y);
                        if new_val > 0 {
                            let old = mask.get_pixel(x, y).0[0];
                            mask.put_pixel(x, y, Luma([old.max(new_val)]));
                        }
                    }
                }
            }
            SelectionMode::Subtract => {
                for y in by0..=by1 {
                    for x in bx0..=bx1 {
                        let sub_val = shape.contains(x, y);
                        if sub_val > 0 {
                            let old = mask.get_pixel(x, y).0[0];
                            mask.put_pixel(x, y, Luma([old.saturating_sub(sub_val)]));
                        }
                    }
                }
            }
            SelectionMode::Intersect => {
                // Keep only pixels inside BOTH the existing mask and the new shape.
                // Clone the existing state so we can read while writing.
                let old_mask = mask.clone();
                // Zero entire mask first
                for p in mask.pixels_mut() { *p = Luma([0]); }
                // Then restore pixels that exist in both
                for y in by0..=by1 {
                    for x in bx0..=bx1 {
                        let shape_val = shape.contains(x, y);
                        if shape_val > 0 {
                            let old_val = old_mask.get_pixel(x, y).0[0];
                            if old_val > 0 {
                                mask.put_pixel(x, y, Luma([shape_val.min(old_val)]));
                            }
                        }
                    }
                }
            }
        }
        self.invalidate_selection_overlay();
    }

    /// Delete (make transparent) the selected pixels on the active layer.
    pub fn delete_selected_pixels(&mut self) {
        let mask = match &self.selection_mask {
            Some(m) => m.clone(),
            None => return,
        };

        if let Some(layer) = self.layers.get_mut(self.active_layer_index) {
            let w = self.width;
            let h = self.height;
            for y in 0..h {
                for x in 0..w {
                    let sel = mask.get_pixel(x, y).0[0];
                    if sel > 0 {
                        let p = layer.pixels.get_pixel_mut(x, y);
                        if sel == 255 {
                            *p = Rgba([0, 0, 0, 0]);
                        } else {
                            // Partial selection: reduce alpha proportionally.
                            let factor = 1.0 - (sel as f32 / 255.0);
                            p[3] = (p[3] as f32 * factor).round() as u8;
                        }
                    }
                }
            }
            self.mark_dirty(None);
        }
    }

    /// Fill the selected area on the active layer with a solid colour.
    pub fn fill_selected_pixels(&mut self, color: Rgba<u8>) {
        let mask = match &self.selection_mask {
            Some(m) => m.clone(),
            None => return,
        };

        if let Some(layer) = self.layers.get_mut(self.active_layer_index) {
            let w = self.width;
            let h = self.height;
            for y in 0..h {
                for x in 0..w {
                    let sel = mask.get_pixel(x, y).0[0];
                    if sel > 0 {
                        let p = layer.pixels.get_pixel_mut(x, y);
                        if sel == 255 {
                            *p = color;
                        } else {
                            // Blend proportionally.
                            let t = sel as f32 / 255.0;
                            let blend = |old: u8, new: u8| -> u8 {
                                ((old as f32) * (1.0 - t) + (new as f32) * t).round() as u8
                            };
                            *p = Rgba([
                                blend(p[0], color[0]),
                                blend(p[1], color[1]),
                                blend(p[2], color[2]),
                                blend(p[3], color[3]),
                            ]);
                        }
                    }
                }
            }
            self.mark_dirty(None);
        }
    }
}

pub struct Canvas {
    pub zoom: f32,
    pan_offset: Vec2,
    last_filter_was_linear: Option<bool>, // Track last filter state to detect changes
    pub last_canvas_rect: Option<Rect>,
    /// Accent color for selection outlines (set from theme).
    pub selection_stroke: Color32,
    /// Faint accent for selection fill overlay (set from theme).
    pub selection_fill: Color32,
    /// Contrasting color for selection dashes (white in dark mode, black in light).
    pub selection_contrast: Color32,
    /// Result from paste overlay context menu: Some(true)=commit, Some(false)=cancel.
    pub paste_context_action: Option<bool>,
    /// When true, the paste overlay context menu should auto-open on next frame.
    pub open_paste_menu: bool,
    /// GPU renderer (always initialised — uses software fallback if no hardware).
    pub gpu_renderer: crate::gpu::GpuRenderer,
    /// FPS tracking: recent frame times for averaging
    frame_times: VecDeque<f64>,
    /// Cached FPS value (updated periodically)
    pub fps: f32,
    /// True while the fill tool is recalculating its preview (shown in loading bar).
    pub fill_recalc_active: bool,
    /// True while a gradient commit is in progress (shown in loading bar).
    pub gradient_commit_active: bool,
    /// Cached texture for layers above the active layer during paste preview.
    /// Rebuilt each frame while a paste overlay is active and layers above exist.
    paste_layers_above_cache: Option<egui::TextureHandle>,
    /// Cached texture for brush tip cursor overlay (reused across frames).
    brush_tip_cursor_tex: Option<egui::TextureHandle>,
    /// Second pass texture (inverted) for visibility on all backgrounds.
    brush_tip_cursor_tex_inv: Option<egui::TextureHandle>,
    /// Cache key for brush tip cursor: (tip_name, mask_size, hardness_pct)
    brush_tip_cursor_key: (String, u32, u32),
    /// Tool icon texture for custom cursor overlay (set from app.rs each frame).
    pub tool_cursor_icon: Option<egui::TextureHandle>,
}

impl Canvas {
    pub fn new(preferred_gpu: &str) -> Self {
        let gpu_renderer = crate::gpu::GpuRenderer::new(preferred_gpu);
        Self {
            zoom: 1.0,
            pan_offset: Vec2::ZERO,
            last_filter_was_linear: None,
            last_canvas_rect: None,
            selection_stroke: Color32::from_rgb(66, 133, 244),
            selection_fill: Color32::from_rgba_unmultiplied(66, 133, 244, 50),
            selection_contrast: Color32::WHITE,
            paste_context_action: None,
            open_paste_menu: false,
            gpu_renderer,
            frame_times: VecDeque::with_capacity(60),
            fps: 0.0,
            fill_recalc_active: false,
            gradient_commit_active: false,
            paste_layers_above_cache: None,
            brush_tip_cursor_tex: None,
            brush_tip_cursor_tex_inv: None,
            brush_tip_cursor_key: (String::new(), 0, 0),
            tool_cursor_icon: None,
        }
    }
    
    pub fn new_without_state() -> Self {
        // Default init with high performance GPU request
        Self::new("high performance")
    }

    /// Reinitialise the GPU renderer with a different preferred adapter.
    pub fn init_gpu(&mut self, preferred_gpu: &str) {
        self.gpu_renderer = crate::gpu::GpuRenderer::new(preferred_gpu);
    }

    /// Returns `true` — GPU rendering is always available (software fallback).
    pub fn has_gpu(&self) -> bool {
        true
    }

    /// GPU adapter name (for status bar display).
    pub fn gpu_adapter_name(&self) -> &str {
        self.gpu_renderer.adapter_name()
    }

    /// Notify the GPU renderer that a layer was deleted so its texture can
    /// be recycled and remaining layer indices are shifted down.
    pub fn gpu_remove_layer(&mut self, layer_idx: usize) {
        self.gpu_renderer.remove_layer_and_reindex(layer_idx);
    }

    /// Clear all GPU layer textures (e.g., project switch).
    pub fn gpu_clear_layers(&mut self) {
        self.gpu_renderer.clear_layers();
    }

    pub fn show(&mut self, ui: &mut egui::Ui, state: &mut CanvasState) {
        // Default colors when called without color info
        let default_primary = [0.0, 0.0, 0.0, 1.0]; // Black
        let default_secondary = [1.0, 1.0, 1.0, 1.0]; // White
        let default_bg = Color32::from_gray(250); // Light grey default
        let default_settings = crate::assets::AppSettings::default();
        let default_accent = Color32::from_rgb(66, 133, 244); // Default blue
        self.show_with_state(ui, state, None, default_primary, default_secondary, default_bg, None, false, &default_settings, 0, 0, default_accent, None, None, &String::new());
    }
    
    pub fn show_with_state(
        &mut self,
        ui: &mut egui::Ui,
        state: &mut CanvasState,
        tools: Option<&mut crate::components::tools::ToolsPanel>,
        primary_color_f32: [f32; 4],
        secondary_color_f32: [f32; 4],
        bg_color: Color32,
        mut paste_overlay: Option<&mut crate::ops::clipboard::PasteOverlay>,
        modal_open: bool,
        debug_settings: &crate::assets::AppSettings,
        pending_filter_jobs: usize,
        pending_io_ops: usize,
        accent_color: Color32,
        filter_ops_start_time: Option<f64>,
        io_ops_start_time: Option<f64>,
        filter_status_description: &str,
    ) {
        // FPS tracking: measure time since last frame
        let current_time = ui.input(|i| i.time);
        self.frame_times.push_back(current_time);
        // Keep last 60 frames for averaging
        while self.frame_times.len() > 60 {
            self.frame_times.pop_front();
        }
        // Calculate FPS from frame times
        if self.frame_times.len() >= 2 {
            let oldest = *self.frame_times.front().unwrap();
            let newest = *self.frame_times.back().unwrap();
            let elapsed = newest - oldest;
            if elapsed > 0.0 {
                self.fps = (self.frame_times.len() - 1) as f32 / elapsed as f32;
            }
        }
        
        let available_size = ui.available_size();
        
        // Allocate canvas area
        let sense = egui::Sense::click_and_drag().union(egui::Sense::hover());
        let (response, painter) = ui.allocate_painter(available_size, sense);
        let canvas_rect = response.rect;
        self.last_canvas_rect = Some(canvas_rect);
        
        // Handle panning with middle mouse button
        if response.dragged() && ui.input(|i| i.pointer.middle_down()) {
            self.pan_offset += response.drag_delta();
        }
        
        // Determine correct texture filter based on zoom level and user settings.
        // User can choose between Linear (smooth) and Nearest (sharp) for zoomed-out views.
        let use_linear_filter = match debug_settings.zoom_filter_mode {
            crate::assets::ZoomFilterMode::Linear => self.zoom < 2.0,
            crate::assets::ZoomFilterMode::Nearest => false,
        };
        let texture_options = if use_linear_filter {
            TextureOptions {
                magnification: TextureFilter::Linear,
                minification: TextureFilter::Linear,
                ..Default::default()
            }
        } else {
            TextureOptions {
                magnification: TextureFilter::Nearest,
                minification: TextureFilter::Nearest,
                ..Default::default()
            }
        };
        
        // Check if filter changed (zoom crossed the 2.0 threshold)
        let prev_filter_was_linear = self.last_filter_was_linear;
        let filter_changed = prev_filter_was_linear.map_or(false, |last| last != use_linear_filter);
        self.last_filter_was_linear = Some(use_linear_filter);

        // ---- GPU COMPOSITING (committed layers only) ----
        // Plans C+D+E+A: Optimised composite → readback → display pipeline.
        //   C: Reuse TextureHandle via tex.set() (no allocation churn)
        //   D: bytemuck cast_slice for zero-copy Color32 conversion
        //   E: Dirty-rect only GPU readback + persistent CPU pixel buffer
        //   A: Avoid recomposite when only filter mode changes (tex options only)
        {
            let gpu = &mut self.gpu_renderer;
            let pixels_dirty = state.dirty_rect.is_some()
                || state.composite_cache.is_none();

            // Plan A: filter_changed (zoom crosses 2.0× threshold) doesn't
            // require GPU recomposite — the pixels haven't changed, only the
            // texture sampling mode.  Re-upload from CPU buffer with new options.
            let needs_reupload_only = filter_changed && !pixels_dirty
                && !state.composite_cpu_buffer.is_empty();

            if pixels_dirty {
                // Sync each real layer to the GPU — per-layer generation
                // tracking ensures only actually-modified layers are
                // re-uploaded, and dirty-rect partial uploads avoid copying
                // the entire image when only a small brush stroke changed.
                let dirty_rect_opt = state.dirty_rect.clone();
                for (idx, layer) in state.layers.iter().enumerate() {
                    if !layer.visible { continue; }

                    if gpu.layer_is_current(idx, layer.gpu_generation) {
                        continue;
                    }

                    let did_partial = if let Some(ref dr) = dirty_rect_opt {
                        let cw = layer.pixels.width();
                        let ch = layer.pixels.height();
                        let rx = (dr.min.x.max(0.0) as u32).min(cw.saturating_sub(1));
                        let ry = (dr.min.y.max(0.0) as u32).min(ch.saturating_sub(1));
                        let rx2 = (dr.max.x.ceil() as u32).min(cw);
                        let ry2 = (dr.max.y.ceil() as u32).min(ch);
                        let rw = rx2.saturating_sub(rx);
                        let rh = ry2.saturating_sub(ry);
                        let is_subregion = rw > 0 && rh > 0 && (rw < cw || rh < ch);
                        if is_subregion && gpu.layer_has_texture(idx, cw, ch) {
                            let (ax, ay, aw, ah) = crate::gpu::align_dirty_rect(rx, ry, rw, rh, cw, ch);
                            if aw > 0 && ah > 0 {
                                // A7: use fast chunk-aware extraction with reusable buffer
                                let mut buf = std::mem::take(&mut state.region_extract_buf);
                                layer.pixels.extract_region_rgba_fast(ax, ay, aw, ah, &mut buf);
                                let region = crate::gpu::renderer::DirtyRegion {
                                    x: ax, y: ay, width: aw, height: ah,
                                };
                                gpu.update_layer_rect(idx, &region, &buf);
                                state.region_extract_buf = buf;
                                gpu.set_layer_generation(idx, layer.gpu_generation);
                                true
                            } else {
                                false
                            }
                        } else {
                            false
                        }
                    } else {
                        false
                    };

                    if !did_partial {
                        let flat = layer.pixels.to_rgba_image();
                        gpu.ensure_layer_texture(
                            idx,
                            layer.pixels.width(),
                            layer.pixels.height(),
                            flat.as_raw(),
                            layer.gpu_generation,
                        );
                    }
                }

                let preview_gpu_idx = usize::MAX;
                gpu.remove_layer(preview_gpu_idx);

                let mut layer_info: Vec<(usize, f32, bool, u8)> = Vec::with_capacity(state.layers.len());
                for (i, l) in state.layers.iter().enumerate() {
                    layer_info.push((i, l.opacity, l.visible, l.blend_mode.to_u8()));
                }

                // B1: Async double-buffered readback.  GPU composites the full
                // canvas (fast, stays on GPU) and copies to a staging buffer.
                // We read from the PREVIOUS frame's staging buffer (non-blocking,
                // 1-frame latency — imperceptible at 60fps).
                let have_existing_buffer = state.composite_cpu_buffer.len()
                    == (state.width as usize * state.height as usize);
                let dr_for_readback = if have_existing_buffer {
                    state.dirty_rect.clone()
                } else {
                    None // Force full readback on first frame or after resize
                };

                // On the very first dirty frame we won't have previous data yet,
                // so fall back to the synchronous path to avoid a blank frame.
                // Also use sync when no preview overlay is active (i.e. brush
                // commit or filter apply) — the preview layer masks the 1-frame
                // async latency during interactive strokes, but without it the
                // user would see a flicker (old composite without the committed
                // pixels for one frame).
                let use_sync = !have_existing_buffer || state.preview_layer.is_none();
                let result = if use_sync {
                    // Sync readback — immediate result, no flicker
                    // Cancel any pending async read to prevent stale data
                    // from being applied on a subsequent frame.
                    gpu.async_readback.cancel_pending();
                    gpu.composite_dirty_readback(
                        state.width, state.height,
                        &layer_info,
                        dr_for_readback,
                    )
                } else {
                    // Interactive stroke with preview overlay — async is safe
                    // (preview masks the 1-frame latency)
                    gpu.composite_dirty_readback_async(
                        state.width, state.height,
                        &layer_info,
                        dr_for_readback,
                    )
                };

                if let Some((pixels, rx, ry, rw, rh, is_full)) = result {
                    // Plan D: bytemuck zero-copy cast from &[u8] to &[Color32]
                    let src: &[Color32] = bytemuck::cast_slice(&pixels);

                    if is_full {
                        // Full readback — replace entire CPU buffer.
                        // Create ColorImage directly from src to avoid a
                        // redundant 33MB clone at 4K.
                        let color_image = ColorImage {
                            size: [state.width as usize, state.height as usize],
                            pixels: src.to_vec(),
                        };
                        // Update persistent CPU buffer from the same source
                        state.composite_cpu_buffer.clear();
                        state.composite_cpu_buffer.extend_from_slice(src);

                        let image_data = ImageData::Color(Arc::new(color_image));
                        if let Some(ref mut tex) = state.composite_cache {
                            tex.set(image_data, texture_options);
                        } else {
                            state.composite_cache = Some(ui.ctx().load_texture(
                                "canvas_composite",
                                image_data,
                                texture_options,
                            ));
                        }
                    } else {
                        // Partial readback — patch dirty region into persistent CPU buffer
                        let canvas_w = state.width as usize;
                        let region_w = rw as usize;
                        for row in 0..rh as usize {
                            let dst_y = ry as usize + row;
                            let dst_start = dst_y * canvas_w + rx as usize;
                            let src_start = row * region_w;
                            state.composite_cpu_buffer[dst_start..dst_start + region_w]
                                .copy_from_slice(&src[src_start..src_start + region_w]);
                        }

                        // B2: Partial upload — only send the dirty region to egui.
                        // This avoids cloning the entire 33MB CPU buffer at 4K.
                        // A brush stroke (e.g. 40×40px) uploads ~6KB instead of ~33MB.
                        let region_pixels: Vec<Color32> = src.to_vec();
                        let region_image = ColorImage {
                            size: [rw as usize, rh as usize],
                            pixels: region_pixels,
                        };
                        let region_data = ImageData::Color(Arc::new(region_image));
                        if let Some(ref mut tex) = state.composite_cache {
                            tex.set_partial(
                                [rx as usize, ry as usize],
                                region_data,
                                texture_options,
                            );
                        } else {
                            // No texture yet — need full upload. Fall back to
                            // full buffer (shouldn't happen: partial readback
                            // requires an existing buffer).
                            let color_image = ColorImage {
                                size: [state.width as usize, state.height as usize],
                                pixels: state.composite_cpu_buffer.clone(),
                            };
                            let image_data = ImageData::Color(Arc::new(color_image));
                            state.composite_cache = Some(ui.ctx().load_texture(
                                "canvas_composite",
                                image_data,
                                texture_options,
                            ));
                        }
                    }
                }
                state.dirty_rect = None;
            } else if gpu.async_readback.read_pending {
                // B1 fix: The async readback from the previous dirty frame may
                // have completed.  Poll it even though no new dirty rect exists,
                // otherwise the committed stroke won't appear until the NEXT
                // dirty frame (causing the "one-stroke-behind" bug).
                if let Some((pixels, rx, ry, rw, rh, is_full)) = gpu.async_readback.try_read(&gpu.ctx.device) {
                    let src: &[Color32] = bytemuck::cast_slice(&pixels);
                    if is_full {
                        let color_image = ColorImage {
                            size: [state.width as usize, state.height as usize],
                            pixels: src.to_vec(),
                        };
                        state.composite_cpu_buffer.clear();
                        state.composite_cpu_buffer.extend_from_slice(src);
                        let image_data = ImageData::Color(Arc::new(color_image));
                        if let Some(ref mut tex) = state.composite_cache {
                            tex.set(image_data, texture_options);
                        } else {
                            state.composite_cache = Some(ui.ctx().load_texture(
                                "canvas_composite",
                                image_data,
                                texture_options,
                            ));
                        }
                    } else {
                        let canvas_w = state.width as usize;
                        let region_w = rw as usize;
                        for row in 0..rh as usize {
                            let dst_y = ry as usize + row;
                            let dst_start = dst_y * canvas_w + rx as usize;
                            let src_start = row * region_w;
                            state.composite_cpu_buffer[dst_start..dst_start + region_w]
                                .copy_from_slice(&src[src_start..src_start + region_w]);
                        }
                        let region_pixels: Vec<Color32> = src.to_vec();
                        let region_image = ColorImage {
                            size: [rw as usize, rh as usize],
                            pixels: region_pixels,
                        };
                        let region_data = ImageData::Color(Arc::new(region_image));
                        if let Some(ref mut tex) = state.composite_cache {
                            tex.set_partial(
                                [rx as usize, ry as usize],
                                region_data,
                                texture_options,
                            );
                        }
                    }
                } else {
                    // Data not ready yet — request another repaint so we poll again next frame
                    ui.ctx().request_repaint();
                }
            } else if needs_reupload_only {
                // Plan A: filter mode changed but pixels didn't — re-upload
                // from existing CPU buffer with new texture options (no GPU work).
                let color_image = ColorImage {
                    size: [state.width as usize, state.height as usize],
                    pixels: state.composite_cpu_buffer.clone(),
                };
                let image_data = ImageData::Color(Arc::new(color_image));
                if let Some(ref mut tex) = state.composite_cache {
                    tex.set(image_data, texture_options);
                } else {
                    state.composite_cache = Some(ui.ctx().load_texture(
                        "canvas_composite",
                        image_data,
                        texture_options,
                    ));
                }
            }
        }
        
        // Calculate image dimensions with zoom
        let image_width = state.width as f32 * self.zoom;
        let image_height = state.height as f32 * self.zoom;
        
        // Center the image with pan offset
        let center_x = canvas_rect.center().x + self.pan_offset.x;
        let center_y = canvas_rect.center().y + self.pan_offset.y;
        
        let temp_rect = Rect::from_center_size(
            Pos2::new(center_x, center_y),
            Vec2::new(image_width, image_height),
        );
        
        // Round to pixel boundaries to prevent sub-pixel rendering gaps
        let image_rect = Rect::from_min_max(
            Pos2::new(temp_rect.min.x.round(), temp_rect.min.y.round()),
            Pos2::new(temp_rect.max.x.round(), temp_rect.max.y.round()),
        );
        
        // Fill background with theme color
        painter.rect_filled(canvas_rect, 0.0, bg_color);
        
        // Draw subtle glow/shadow around the canvas (emanating from all sides)
        let glow_layers = 6;
        let max_glow_distance = 12.0;
        for i in 0..glow_layers {
            let distance = (i as f32 + 1.0) * (max_glow_distance / glow_layers as f32);
            let alpha = (((8 - i) * 2) as f32 * 0.6) as u8; // 40% more subtle
            let glow_rect = image_rect.expand(distance);
            painter.rect_filled(glow_rect, 3.0, Color32::from_black_alpha(alpha));
        }
        
        // Draw checkerboard background (clipped to visible canvas area)
        self.draw_checkerboard(&painter, image_rect, canvas_rect, debug_settings.checkerboard_brightness);
        
        // Draw the composite image
        let uv = Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0));
        if let Some(texture) = &state.composite_cache {
            painter.image(texture.id(), image_rect, uv, Color32::WHITE);
        }

        // ====================================================================
        // CPU PREVIEW OVERLAY  (brush / line / eraser strokes in progress)
        // ====================================================================
        // The preview layer is composited with the full layer stack so that
        // both the tool's blend mode (preview_blend_mode) and each layer's
        // own blend mode + opacity are respected.  For
        // Normal+Normal+opacity=1.0 we use the fast raw-extraction path
        // (single-layer memcpy) since alpha-over on top of the GPU
        // composite is already correct in that case.
        if state.preview_layer.is_some() {
            // Check if zoom crossed the filtering threshold - invalidate cache to force recreation
            let current_filter_is_linear = match debug_settings.zoom_filter_mode {
                crate::assets::ZoomFilterMode::Linear if self.zoom < 2.0 => true,
                _ => false,
            };
            if prev_filter_was_linear != Some(current_filter_is_linear) {
                state.preview_texture_cache = None;
            }
            
            let needs_upload = state.preview_texture_cache.is_none()
                || state.preview_dirty_rect.is_some();

            if needs_upload {
                // Determine the stroke's accumulated bounding box.
                let sb = state.preview_stroke_bounds.unwrap_or_else(|| {
                    egui::Rect::from_min_max(
                        egui::pos2(0.0, 0.0),
                        egui::pos2(state.width as f32, state.height as f32),
                    )
                });

                // Zoom-adaptive filtering based on user preference.
                // Linear (smooth) or Nearest (sharp) for zoomed-out pixel art.
                let preview_filter = match debug_settings.zoom_filter_mode {
                    crate::assets::ZoomFilterMode::Linear if self.zoom < 2.0 => TextureFilter::Linear,
                    _ => TextureFilter::Nearest,
                };
                let tex_options = TextureOptions {
                    magnification: preview_filter,
                    minification: preview_filter,
                    ..Default::default()
                };

                // Decide whether the fast raw-extraction path is valid.
                // It is correct when both blend modes are Normal and
                // the active layer is fully opaque, because plain
                // alpha-over in the egui painter is equivalent.
                // Force the blend-aware path when preview_force_composite is set
                // (e.g. fill tool with semi-transparent colours needs proper
                // layer-stack compositing to preview accurately).
                // Also force blend-aware path when there are visible layers above
                // the active layer, so they get composited correctly.
                let active_layer_normal = state.layers
                    .get(state.active_layer_index)
                    .map(|l| l.blend_mode == BlendMode::Normal && l.opacity >= 1.0)
                    .unwrap_or(false);
                let has_layers_above = state.layers
                    .iter()
                    .skip(state.active_layer_index + 1)
                    .any(|l| l.visible);
                let use_fast_path = state.preview_blend_mode == BlendMode::Normal
                    && active_layer_normal
                    && !state.preview_force_composite
                    && !has_layers_above
                    && !state.preview_is_eraser;

                if use_fast_path {
                    if state.preview_flat_ready {
                        // -- Ultra-fast path: buffer already premultiplied --
                        // Transmute &[u8] → &[Color32] (both are 4 bytes, same
                        // layout) to avoid the copy that from_rgba_premultiplied
                        // would do.  bytemuck guarantees safety.
                        // When preview_downscale > 1 the buffer is at reduced
                        // resolution — use scaled dimensions so egui stretches.
                        let pixels: &[Color32] = bytemuck::cast_slice(&state.preview_flat_buffer);
                        let ds = state.preview_downscale;
                        let pw = if ds > 1 { ((state.width + ds - 1) / ds) as usize } else { state.width as usize };
                        let ph = if ds > 1 { ((state.height + ds - 1) / ds) as usize } else { state.height as usize };
                        let color_image = ColorImage {
                            size: [pw, ph],
                            pixels: pixels.to_vec(),
                        };
                        let image_data = ImageData::Color(Arc::new(color_image));
                        if let Some(ref mut tex) = state.preview_texture_cache {
                            tex.set(image_data, tex_options);
                        } else {
                            state.preview_texture_cache = Some(ui.ctx().load_texture(
                                "preview_overlay",
                                image_data,
                                tex_options,
                            ));
                        }
                    } else {
                        // -- Incremental fast path: only re-extract dirty rect --
                        // The brush uses max-alpha stamping, so previously
                        // written pixels never decrease — their premultiplied
                        // values in the cache stay valid.  Each frame we only
                        // re-extract the small dirty region (brush-sized) and
                        // blit it into the persistent cache, reducing per-frame
                        // work from O(stroke_bounds) to O(brush_size).
                        let preview = state.preview_layer.as_ref().unwrap();
                        let rx = (sb.min.x.max(0.0) as u32).min(preview.width().saturating_sub(1));
                        let ry = (sb.min.y.max(0.0) as u32).min(preview.height().saturating_sub(1));
                        let rx2 = (sb.max.x.ceil() as u32).min(preview.width());
                        let ry2 = (sb.max.y.ceil() as u32).min(preview.height());
                        let rw = rx2.saturating_sub(rx).max(1);
                        let rh = ry2.saturating_sub(ry).max(1);

                        let old_cache = state.preview_cache_rect;
                        let bounds_match = old_cache.map_or(false, |(ox, oy, ow, oh)| {
                            ox == rx && oy == ry && ow == rw && oh == rh
                        });

                        if bounds_match && state.preview_dirty_rect.is_some() {
                            // -- Incremental update: extract only the dirty rect --
                            let dr = state.preview_dirty_rect.unwrap();
                            let dx = (dr.min.x.max(0.0).floor() as u32).max(rx);
                            let dy = (dr.min.y.max(0.0).floor() as u32).max(ry);
                            let dx2 = (dr.max.x.ceil() as u32).min(rx + rw);
                            let dy2 = (dr.max.y.ceil() as u32).min(ry + rh);
                            let dw = dx2.saturating_sub(dx);
                            let dh = dy2.saturating_sub(dy);

                            if dw > 0 && dh > 0 {
                                preview.extract_region_rgba_fast(dx, dy, dw, dh, &mut state.preview_flat_buffer);

                                // Premultiply the small dirty region in-place
                                for px in state.preview_flat_buffer.chunks_exact_mut(4) {
                                    let a = px[3] as u16;
                                    px[0] = ((px[0] as u16 * a + 128) / 255) as u8;
                                    px[1] = ((px[1] as u16 * a + 128) / 255) as u8;
                                    px[2] = ((px[2] as u16 * a + 128) / 255) as u8;
                                }

                                // Blit dirty pixels into persistent cache
                                let dirty_pixels: &[Color32] = bytemuck::cast_slice(&state.preview_flat_buffer);
                                let cache_w = rw as usize;
                                let off_x = (dx - rx) as usize;
                                let off_y = (dy - ry) as usize;
                                for row in 0..dh as usize {
                                    let src_start = row * dw as usize;
                                    let dst_start = (off_y + row) * cache_w + off_x;
                                    state.preview_premul_cache[dst_start..dst_start + dw as usize]
                                        .copy_from_slice(&dirty_pixels[src_start..src_start + dw as usize]);
                                }

                                // B2: Partial upload — only send the dirty rect to egui.
                                // During brush strokes this uploads ~brush_size² pixels
                                // instead of cloning the full stroke_bounds cache.
                                let region_pixels: Vec<Color32> = dirty_pixels.to_vec();
                                let region_image = ColorImage {
                                    size: [dw as usize, dh as usize],
                                    pixels: region_pixels,
                                };
                                let region_data = ImageData::Color(Arc::new(region_image));
                                if let Some(ref mut tex) = state.preview_texture_cache {
                                    tex.set_partial(
                                        [off_x, off_y],
                                        region_data,
                                        tex_options,
                                    );
                                } else {
                                    // No texture yet — fall back to full upload
                                    let color_image = ColorImage {
                                        size: [rw as usize, rh as usize],
                                        pixels: state.preview_premul_cache.clone(),
                                    };
                                    let image_data = ImageData::Color(Arc::new(color_image));
                                    state.preview_texture_cache = Some(ui.ctx().load_texture(
                                        "preview_overlay",
                                        image_data,
                                        tex_options,
                                    ));
                                }
                            }
                        } else if old_cache.is_some() && !bounds_match {
                            // -- Bounds changed: resize cache, preserve overlapping data --
                            let (ox, oy, ow, oh) = old_cache.unwrap();
                            let new_size = (rw as usize) * (rh as usize);
                            let mut new_cache = vec![Color32::TRANSPARENT; new_size];

                            // Compute the intersection of old and new bounds
                            // to safely copy only the overlapping region.
                            // Bounds can shrink or shift (e.g. text alignment
                            // change), not just grow.
                            let inter_x0 = ox.max(rx);
                            let inter_y0 = oy.max(ry);
                            let inter_x1 = (ox + ow).min(rx + rw);
                            let inter_y1 = (oy + oh).min(ry + rh);
                            let new_w = rw as usize;
                            let old_w = ow as usize;
                            if inter_x1 > inter_x0 && inter_y1 > inter_y0 {
                                let copy_w = (inter_x1 - inter_x0) as usize;
                                let copy_h = (inter_y1 - inter_y0) as usize;
                                let src_off_x = (inter_x0 - ox) as usize;
                                let src_off_y = (inter_y0 - oy) as usize;
                                let dst_off_x = (inter_x0 - rx) as usize;
                                let dst_off_y = (inter_y0 - ry) as usize;
                                for row in 0..copy_h {
                                    let src_start = (src_off_y + row) * old_w + src_off_x;
                                    let dst_start = (dst_off_y + row) * new_w + dst_off_x;
                                    new_cache[dst_start..dst_start + copy_w]
                                        .copy_from_slice(&state.preview_premul_cache[src_start..src_start + copy_w]);
                                }
                            }

                            // Extract and premultiply only the dirty rect
                            if let Some(dr) = state.preview_dirty_rect {
                                let dx = (dr.min.x.max(0.0).floor() as u32).max(rx);
                                let dy = (dr.min.y.max(0.0).floor() as u32).max(ry);
                                let dx2 = (dr.max.x.ceil() as u32).min(rx + rw);
                                let dy2 = (dr.max.y.ceil() as u32).min(ry + rh);
                                let dw = dx2.saturating_sub(dx);
                                let dh = dy2.saturating_sub(dy);

                                if dw > 0 && dh > 0 {
                                    preview.extract_region_rgba_fast(dx, dy, dw, dh, &mut state.preview_flat_buffer);
                                    for px in state.preview_flat_buffer.chunks_exact_mut(4) {
                                        let a = px[3] as u16;
                                        px[0] = ((px[0] as u16 * a + 128) / 255) as u8;
                                        px[1] = ((px[1] as u16 * a + 128) / 255) as u8;
                                        px[2] = ((px[2] as u16 * a + 128) / 255) as u8;
                                    }
                                    let dirty_pixels: &[Color32] = bytemuck::cast_slice(&state.preview_flat_buffer);
                                    let off_x = (dx - rx) as usize;
                                    let off_y = (dy - ry) as usize;
                                    for row in 0..dh as usize {
                                        let src_start = row * dw as usize;
                                        let dst_start = (off_y + row) * new_w + off_x;
                                        new_cache[dst_start..dst_start + dw as usize]
                                            .copy_from_slice(&dirty_pixels[src_start..src_start + dw as usize]);
                                    }
                                }
                            }

                            state.preview_premul_cache = new_cache;
                            state.preview_cache_rect = Some((rx, ry, rw, rh));

                            // Bounds changed — must do full texture upload
                            let color_image = ColorImage {
                                size: [rw as usize, rh as usize],
                                pixels: state.preview_premul_cache.clone(),
                            };
                            let image_data = ImageData::Color(Arc::new(color_image));
                            if let Some(ref mut tex) = state.preview_texture_cache {
                                tex.set(image_data, tex_options);
                            } else {
                                state.preview_texture_cache = Some(ui.ctx().load_texture(
                                    "preview_overlay",
                                    image_data,
                                    tex_options,
                                ));
                            }
                        } else {
                            // -- First frame or full rebuild --
                            preview.extract_region_rgba_fast(rx, ry, rw, rh, &mut state.preview_flat_buffer);

                            for px in state.preview_flat_buffer.chunks_exact_mut(4) {
                                let a = px[3] as u16;
                                px[0] = ((px[0] as u16 * a + 128) / 255) as u8;
                                px[1] = ((px[1] as u16 * a + 128) / 255) as u8;
                                px[2] = ((px[2] as u16 * a + 128) / 255) as u8;
                            }

                            let pixels: &[Color32] = bytemuck::cast_slice(&state.preview_flat_buffer);
                            state.preview_premul_cache = pixels.to_vec();
                            state.preview_cache_rect = Some((rx, ry, rw, rh));

                            // First frame — must do full texture upload
                            let color_image = ColorImage {
                                size: [rw as usize, rh as usize],
                                pixels: state.preview_premul_cache.clone(),
                            };
                            let image_data = ImageData::Color(Arc::new(color_image));
                            if let Some(ref mut tex) = state.preview_texture_cache {
                                tex.set(image_data, tex_options);
                            } else {
                                state.preview_texture_cache = Some(ui.ctx().load_texture(
                                    "preview_overlay",
                                    image_data,
                                    tex_options,
                                ));
                            }
                        }
                    }
                } else {
                    // -- Blend-aware path: full-stack composite --
                    // composite_partial blends the preview into the
                    // active layer (via preview_blend_mode) and then
                    // composites every layer with its own blend mode
                    // and opacity, giving a pixel-accurate preview.
                    // When preview_downscale > 1, sample at reduced rate
                    // for interactive responsiveness (egui stretches the
                    // smaller texture to fill the canvas rect).
                    let ds = state.preview_downscale;
                    let (mut color_image, _offset) = if ds > 1 {
                        state.composite_partial_downscaled(sb, ds)
                    } else {
                        state.composite_partial(sb)
                    };

                    // For eraser: bake the checkerboard pattern into any
                    // semi-transparent pixels so the preview texture is fully
                    // opaque.  This lets us paint it directly over the base
                    // composite with no extra rects underneath — eliminating
                    // anti-aliased edge artifacts that caused a visible seam.
                    if state.preview_is_eraser {
                        // Scale checker size proportionally when downscaled so
                        // the pattern looks consistent at any preview resolution.
                        let checker_canvas = if ds > 1 { (10 / ds).max(2) } else { 10u32 };
                        let brightness = debug_settings.checkerboard_brightness;
                        let light = (220.0 * brightness).clamp(0.0, 255.0);
                        let dark  = (180.0 * brightness).clamp(0.0, 255.0);
                        let cw = color_image.size[0] as u32;
                        let ch = color_image.size[1] as u32;
                        let origin_x = (sb.min.x.max(0.0).floor() as u32) / ds;
                        let origin_y = (sb.min.y.max(0.0).floor() as u32) / ds;
                        for py in 0..ch {
                            for px in 0..cw {
                                let idx = (py * cw + px) as usize;
                                let pixel = color_image.pixels[idx];
                                let a = pixel.a();
                                if a == 255 { continue; } // fully opaque — no checkerboard needed
                                let cx = (origin_x + px) / checker_canvas;
                                let cy = (origin_y + py) / checker_canvas;
                                let bg = if (cx + cy) % 2 == 0 { light } else { dark };
                                let bg_u8 = bg as u8;
                                if a == 0 {
                                    color_image.pixels[idx] = Color32::from_rgb(bg_u8, bg_u8, bg_u8);
                                } else {
                                    // Color32 stores premultiplied alpha — use
                                    // premultiplied compositing: result = src + bg*(1-a)
                                    // (NOT src*a + bg*(1-a), which would double-multiply).
                                    let inv = 1.0 - a as f32 / 255.0;
                                    let r = (pixel.r() as f32 + bg * inv).min(255.0) as u8;
                                    let g = (pixel.g() as f32 + bg * inv).min(255.0) as u8;
                                    let b = (pixel.b() as f32 + bg * inv).min(255.0) as u8;
                                    color_image.pixels[idx] = Color32::from_rgb(r, g, b);
                                }
                            }
                        }
                    }

                    let image_data = ImageData::Color(Arc::new(color_image));
                    if let Some(ref mut tex) = state.preview_texture_cache {
                        tex.set(image_data, tex_options);
                    } else {
                        state.preview_texture_cache = Some(ui.ctx().load_texture(
                            "preview_overlay",
                            image_data,
                            tex_options,
                        ));
                    }
                }

                state.preview_dirty_rect = None;
            }

            // Paint the cropped preview at the correct position.
            // Use image_rect-derived proportional positioning so the preview
            // aligns exactly with the GPU composite.  image_rect is
            // independently rounded from temp_rect, so its effective zoom
            // (image_rect.width / state.width) can differ slightly from
            // self.zoom.  Using fractional offsets ensures pixel-perfect
            // registration between the preview overlay and the base
            // composite texture.
            if let Some(ref tex) = state.preview_texture_cache {
                if let Some(sb) = state.preview_stroke_bounds {
                    let off_x = sb.min.x.max(0.0).floor();
                    let off_y = sb.min.y.max(0.0).floor();
                    let sw = (sb.max.x.ceil().min(state.width as f32) - off_x).max(1.0);
                    let sh = (sb.max.y.ceil().min(state.height as f32) - off_y).max(1.0);
                    let canvas_w = state.width as f32;
                    let canvas_h = state.height as f32;
                    let sub_rect = Rect::from_min_size(
                        Pos2::new(
                            image_rect.min.x + off_x / canvas_w * image_rect.width(),
                            image_rect.min.y + off_y / canvas_h * image_rect.height(),
                        ),
                        Vec2::new(
                            sw / canvas_w * image_rect.width(),
                            sh / canvas_h * image_rect.height(),
                        ),
                    );
                    // Eraser preview has checkerboard baked into the texture
                    // (fully opaque), so no extra rects are needed — just paint it.
                    painter.image(tex.id(), sub_rect, uv, Color32::WHITE);
                } else {
                    painter.image(tex.id(), image_rect, uv, Color32::WHITE);
                }
            }
        } else {
            // No preview layer — drop cached texture and buffer.
            state.preview_texture_cache = None;
            // Don't clear preview_flat_buffer here to avoid dealloc; it will
            // be reused on the next stroke.
        }
        
        // Draw pixel grid overlay when zoomed in
        if state.show_pixel_grid && self.zoom >= 8.0 {
            self.draw_pixel_grid(&painter, image_rect, state, canvas_rect);
        }

        // Draw center/thirds guidelines overlay
        if state.show_guidelines {
            self.draw_guidelines(&painter, image_rect, state, canvas_rect);
        }

        // Draw mirror axis overlay
        if state.mirror_mode.is_active() {
            self.draw_mirror_overlay(&painter, image_rect, state, canvas_rect);
        }

        // ====================================================================
        // SELECTION OVERLAY  (above layers, below tool cursor)
        // ====================================================================
        // Animated time value for marching ants.
        let anim_time = ui.input(|i| i.time);

        // Keep repainting so marching ants stay animated even when mouse is idle.
        let has_selection_mask = state.selection_mask.is_some();
        let has_selection_drag = tools.as_ref().map_or(false, |t| t.selection_state.dragging);
        if has_selection_mask || has_selection_drag {
            ui.ctx().request_repaint();
        }

        // 1. Draw the committed selection mask (if any).
        //    Temporarily take the mask to avoid borrow conflict with `&mut state`
        //    (needed for the overlay texture cache).  Put it back afterwards.
        if let Some(mask) = state.selection_mask.take() {
            // Switch to "tool-active" visual mode when a drawing tool is
            // selected (not a selection tool).  This hides the hatch fill so
            // the user can see what they're painting.
            let tool_active = tools.as_ref().map_or(false, |t| {
                matches!(t.active_tool,
                    crate::components::tools::Tool::Brush
                    | crate::components::tools::Tool::Eraser
                    | crate::components::tools::Tool::Pencil
                    | crate::components::tools::Tool::Line
                )
            });
            self.draw_selection_overlay(&painter, image_rect, &mask, anim_time, tool_active, state, ui.ctx());
            state.selection_mask = Some(mask); // put it back
        }

        // 2. Draw the in-progress drag preview (while dragging a selection tool).
        // We peek at the tools reference to see if a selection drag is active.
        {
            // We need to check tools *before* we pass ownership via Option,
            // so we borrow it here (tools is &mut Option).
            let sel_preview = tools.as_ref().and_then(|t| {
                if !t.selection_state.dragging { return None; }
                let start = t.selection_state.drag_start?;
                let end = t.selection_state.drag_end?;
                Some((t.active_tool, start, end))
            });

            if let Some((tool, start, end)) = sel_preview {
                let min_x = start.x.min(end.x);
                let min_y = start.y.min(end.y);
                let max_x = start.x.max(end.x);
                let max_y = start.y.max(end.y);

                // Convert canvas coords => screen coords
                let screen_min = Pos2::new(
                    image_rect.min.x + min_x * self.zoom,
                    image_rect.min.y + min_y * self.zoom,
                );
                let screen_max = Pos2::new(
                    image_rect.min.x + max_x * self.zoom,
                    image_rect.min.y + max_y * self.zoom,
                );
                let sel_rect = Rect::from_min_max(screen_min, screen_max);

                // Faint fill (derived from theme accent)
                let fill_color = self.selection_fill;

                match tool {
                    crate::components::tools::Tool::EllipseSelect => {
                        // Draw filled ellipse approximation with marching ants
                        self.draw_ellipse_overlay(&painter, sel_rect, fill_color, anim_time);
                    }
                    _ => {
                        painter.rect_filled(sel_rect, 0.0, fill_color);
                        // Marching ants border
                        self.draw_marching_rect(&painter, sel_rect, anim_time);
                    }
                }
            }
        }
        
        // ====================================================================
        // PASTE OVERLAY  (above selection, interactive handles)
        // ====================================================================
        let mut paste_consumed_input = false;
        let mut paste_context_result: Option<bool> = None;
        if let Some(ref mut overlay) = paste_overlay {
            let is_dark = ui.visuals().dark_mode;
            let accent = self.selection_stroke; // theme accent colour
            
            // Upload/re-upload the source texture when scale changes (once).
            // The GPU handles rotation + translation via a textured mesh.
            // Pass interaction state so large images use a lower-res preview
            // during drag to keep the UI responsive.
            let is_interacting = overlay.active_handle.is_some();
            overlay.ensure_gpu_texture(ui.ctx(), is_interacting);

            // Draw the transformed paste image via GPU mesh, clipped to canvas bounds
            // so pasted images larger than the canvas don't extend beyond it.
            let clipped_painter = painter.with_clip_rect(image_rect);
            overlay.draw_gpu(&clipped_painter, image_rect, self.zoom);

            // If there are visible layers above the active layer, composite them
            // and overlay on top of the paste so those layers render correctly.
            let has_layers_above = state.layers
                .iter()
                .skip(state.active_layer_index + 1)
                .any(|l| l.visible);
            if has_layers_above {
                if let Some(above_image) = state.composite_layers_above() {
                    let tex = if let Some(ref mut existing) = self.paste_layers_above_cache {
                        existing.set(above_image, TextureOptions {
                            magnification: TextureFilter::Nearest,
                            minification: TextureFilter::Nearest,
                            ..Default::default()
                        });
                        existing.id()
                    } else {
                        let handle = ui.ctx().load_texture(
                            "paste_layers_above",
                            above_image,
                            TextureOptions {
                                magnification: TextureFilter::Nearest,
                                minification: TextureFilter::Nearest,
                                ..Default::default()
                            },
                        );
                        let id = handle.id();
                        self.paste_layers_above_cache = Some(handle);
                        id
                    };
                    let uv = Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0));
                    clipped_painter.image(tex, image_rect, uv, Color32::WHITE);
                }
            } else {
                // No layers above — drop cache
                self.paste_layers_above_cache = None;
            }

            // Clear CPU preview layer so the composite doesn't double-draw.
            if state.preview_layer.is_some() {
                let tool_using = tools.as_ref().map_or(false, |t| t.stroke_tracker.uses_preview_layer);
                if !tool_using {
                    state.clear_preview_state();
                    state.mark_dirty(None);
                }
            }
            ui.ctx().request_repaint();
            
            // Draw handles/border on top of the composite.
            overlay.draw(&painter, image_rect, self.zoom, is_dark, accent);
            
            // Handle interaction (move, resize, rotate).
            paste_consumed_input = overlay.handle_input(ui, image_rect, self.zoom);
            
            // Auto-open context menu when requested (e.g. right after paste).
            if self.open_paste_menu {
                let ctx_id = response.id.with("__egui_context_menu");
                ui.memory_mut(|mem| mem.open_popup(ctx_id));
                self.open_paste_menu = false;
            }
            
            // Right-click context menu on the paste overlay.
            response.context_menu(|ui| {
                // Interpolation filter selector.
                ui.label("Filter:");
                for interp in crate::ops::transform::Interpolation::all() {
                    if ui.selectable_label(overlay.interpolation == *interp, interp.label()).clicked() {
                        overlay.interpolation = *interp;
                        ui.close_menu();
                    }
                }
                ui.separator();
                if ui.button("Reset Transform").clicked() {
                    overlay.rotation = 0.0;
                    overlay.scale_x = 1.0;
                    overlay.scale_y = 1.0;
                    overlay.anchor_offset = Vec2::ZERO;
                    ui.close_menu();
                }
                if ui.button("Center Anchor").clicked() {
                    overlay.anchor_offset = Vec2::ZERO;
                    ui.close_menu();
                }
                ui.separator();
                if ui.button("✓ Commit   (Enter)").clicked() {
                    paste_context_result = Some(true);
                    ui.close_menu();
                }
                if ui.button("✗ Cancel   (Esc)").clicked() {
                    paste_context_result = Some(false);
                    ui.close_menu();
                }
            });
        } else {
            // No paste overlay — clear layers-above cache.
            self.paste_layers_above_cache = None;
            // Close any orphaned paste context-menu popup.
            // If commit/cancel happened while the context menu was still open,
            // the popup ID remains registered in egui memory and permanently
            // makes `any_popup_open()` return true, blocking all tool input.
            let ctx_id = response.id.with("__egui_context_menu");
            if ui.memory(|mem| mem.is_popup_open(ctx_id)) {
                ui.memory_mut(|mem| mem.toggle_popup(ctx_id));
            }
            // Clear preview layer if no paste overlay is active.
            if state.preview_layer.is_some() {
                // Only clear if tools aren't using it.
                // Check if a tool stroke is active — if so, leave preview_layer alone.
                let tool_using_preview = tools.as_ref().map_or(false, |t| t.stroke_tracker.uses_preview_layer);
                if !tool_using_preview {
                    state.clear_preview_state();
                }
            }
        }
        self.paste_context_action = paste_context_result;

        // ====================================================================
        // EXTRACT DEBUG INFO before tools is consumed
        // ====================================================================
        // Selection drag info (for debug display).
        let sel_drag_info: Option<(f32, f32)> = tools.as_ref().and_then(|t| {
            if !t.selection_state.dragging { return None; }
            let s = t.selection_state.drag_start?;
            let e = t.selection_state.drag_end?;
            let w = (e.x - s.x).abs();
            let h = (e.y - s.y).abs();
            Some((w, h))
        });

        // Paste overlay info (for debug display).
        let paste_info: Option<(f32, f32, f32, f32, f32, f32)> = paste_overlay.as_ref().map(|o| {
            let sw = o.source.width() as f32 * o.scale_x;
            let sh = o.source.height() as f32 * o.scale_y;
            (o.center.x, o.center.y, sw, sh, o.rotation.to_degrees(), o.scale_x * 100.0)
        });

        // Check if Pencil tool is active before consuming tools
        let pencil_active = tools.as_ref().map_or(false, |t| t.active_tool == crate::components::tools::Tool::Pencil);
        // Check if Fill tool is active before consuming tools
        let fill_active = tools.as_ref().map_or(false, |t| t.active_tool == crate::components::tools::Tool::Fill);
        // Check if Color Picker tool is active before consuming tools
        let color_picker_active = tools.as_ref().map_or(false, |t| t.active_tool == crate::components::tools::Tool::ColorPicker);
        // Extract active tool for cursor icon mapping
        let active_tool_for_cursor = tools.as_ref().map(|t| t.active_tool);
        // Extract text tool handle state for cursor icon
        let text_hovering_handle = tools.as_ref().map_or(false, |t| t.text_state.hovering_handle);
        let text_dragging_handle = tools.as_ref().map_or(false, |t| t.text_state.dragging_handle);
        // Extract line tool pan-handle state for cursor icon
        let line_pan_hovering = tools.as_ref().map_or(false, |t| {
            t.active_tool == crate::components::tools::Tool::Line
                && t.line_state.line_tool.stage == crate::components::tools::LineStage::Editing
                && t.line_state.line_tool.pan_handle_hovering
        });
        let line_pan_dragging = tools.as_ref().map_or(false, |t| {
            t.active_tool == crate::components::tools::Tool::Line
                && t.line_state.line_tool.pan_handle_dragging
        });
        // Extract zoom tool out-mode for cursor icon
        let _zoom_out_mode = tools.as_ref().map_or(false, |t| t.zoom_tool_state.zoom_out_mode);
        // Extract brush cursor info before tools ref is consumed
        // (size, clone_source, is_circle, mask_data_for_cursor, rotation_deg)
        let brush_cursor_info: Option<(f32, Option<Pos2>, bool, Option<(Vec<u8>, u32)>, f32)> = tools.as_ref().and_then(|t| {
            use crate::components::tools::Tool;
            match t.active_tool {
                Tool::Brush | Tool::Eraser | Tool::CloneStamp | Tool::ContentAwareBrush | Tool::Liquify => {
                    let clone_source = if t.active_tool == Tool::CloneStamp {
                        t.clone_stamp_state.source
                    } else {
                        None
                    };
                    let is_circle_tip = t.properties.brush_tip.is_circle();
                    // For image tips, pass the mask so cursor can draw shape overlay
                    let mask_info = if !is_circle_tip && !t.brush_tip_mask.is_empty() {
                        Some((t.brush_tip_mask.clone(), t.brush_tip_mask_size))
                    } else {
                        None
                    };
                    // For fixed rotation, pass angle to cursor; for random, 0 (no preview)
                    let cursor_rotation = t.active_tip_rotation_deg;
                    Some((t.properties.size, clone_source, is_circle_tip, mask_info, cursor_rotation))
                }
                _ => None,
            }
        });
        let tip_name_for_cursor: Option<String> = tools.as_ref().and_then(|t| {
            match &t.properties.brush_tip {
                crate::components::tools::BrushTip::Image(name) => Some(name.clone()),
                _ => None,
            }
        });

        // Handle tool input - Call every frame while mouse button is held
        if let Some(tools) = tools {
            // Only block input if there's a modal window/popup or pointer is over any UI element
            let ui_blocking = ui.ctx().memory(|mem| mem.any_popup_open()) || ui.ctx().is_pointer_over_area();
            
            // Get mouse position and check if over canvas
            let mouse_pos = ui.input(|i| i.pointer.interact_pos());
            let pointer_over_canvas = mouse_pos.map_or(false, |pos| canvas_rect.contains(pos));
            
            // Get canvas position (will be None if not over image)
            let canvas_pos = mouse_pos.and_then(|pos| self.screen_to_canvas(pos, canvas_rect, state));
            let canvas_pos_f32 = mouse_pos.and_then(|pos| self.screen_to_canvas_f32(pos, canvas_rect, state));
            // Extended version for overlay tools (mesh warp) — allows clicks slightly outside canvas bounds
            let canvas_pos_f32_clamped = mouse_pos.and_then(|pos| self.screen_to_canvas_f32_clamped(pos, canvas_rect, state, 16.0));
            
            // Check if we're in the middle of a stroke (mouse button is held)
            let is_painting = ui.input(|i| i.pointer.primary_down() || i.pointer.secondary_down());
            
            // Collect ALL sub-frame PointerMoved events for smooth strokes.
            // At 21 FPS with a 1000Hz mouse, there can be ~48 intermediate
            // positions between frames. Processing all of them eliminates gaps.
            let raw_motion_events: Vec<(f32, f32)> = if is_painting {
                ui.input(|i| {
                    i.events.iter().filter_map(|e| {
                        if let egui::Event::PointerMoved(screen_pos) = e {
                            // Convert screen coordinates to canvas coordinates
                            // Inline the conversion to avoid borrow issues
                            let image_width = state.width as f32 * self.zoom;
                            let image_height = state.height as f32 * self.zoom;
                            let center_x = canvas_rect.center().x + self.pan_offset.x;
                            let center_y = canvas_rect.center().y + self.pan_offset.y;
                            let ir = Rect::from_center_size(
                                Pos2::new(center_x, center_y),
                                Vec2::new(image_width, image_height),
                            );
                            if !ir.contains(*screen_pos) {
                                return None;
                            }
                            let rel_x = (screen_pos.x - ir.min.x) / self.zoom;
                            let rel_y = (screen_pos.y - ir.min.y) / self.zoom;
                            if rel_x >= 0.0 && rel_x < state.width as f32
                                && rel_y >= 0.0 && rel_y < state.height as f32
                            {
                                Some((rel_x, rel_y))
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    }).collect()
                })
            } else {
                Vec::new()
            };
            
            // Only process if not blocked by UI AND (pointer over canvas OR actively painting)
            // AND paste overlay didn't consume the input
            let allow_input = !modal_open && !ui_blocking && !paste_consumed_input && (pointer_over_canvas || is_painting);
            
            if allow_input {
                tools.handle_input(ui, state, canvas_pos, canvas_pos_f32, canvas_pos_f32_clamped, &raw_motion_events, &painter, image_rect, self.zoom, primary_color_f32, secondary_color_f32, Some(&mut self.gpu_renderer));

                // Consume zoom/pan action emitted by the Zoom or Pan tool
                use crate::components::tools::ZoomPanAction;
                let action = std::mem::replace(&mut tools.zoom_pan_action, ZoomPanAction::None);
                match action {
                    ZoomPanAction::None => {}
                    ZoomPanAction::ZoomIn { canvas_x, canvas_y } => {
                        // Zoom in anchored at the click point
                        let screen_x = canvas_rect.min.x + self.pan_offset.x + canvas_rect.width() / 2.0
                            - (state.width as f32 * self.zoom / 2.0) + canvas_x * self.zoom;
                        let screen_y = canvas_rect.min.y + self.pan_offset.y + canvas_rect.height() / 2.0
                            - (state.height as f32 * self.zoom / 2.0) + canvas_y * self.zoom;
                        let anchor = Pos2::new(screen_x, screen_y);
                        self.zoom_around_screen_point(1.2, anchor, canvas_rect);
                    }
                    ZoomPanAction::ZoomOut { canvas_x, canvas_y } => {
                        let screen_x = canvas_rect.min.x + self.pan_offset.x + canvas_rect.width() / 2.0
                            - (state.width as f32 * self.zoom / 2.0) + canvas_x * self.zoom;
                        let screen_y = canvas_rect.min.y + self.pan_offset.y + canvas_rect.height() / 2.0
                            - (state.height as f32 * self.zoom / 2.0) + canvas_y * self.zoom;
                        let anchor = Pos2::new(screen_x, screen_y);
                        self.zoom_around_screen_point(1.0 / 1.2, anchor, canvas_rect);
                    }
                    ZoomPanAction::Pan { dx, dy } => {
                        self.pan_by(Vec2::new(dx, dy));
                    }
                    ZoomPanAction::ZoomToRect { min_x, min_y, max_x, max_y } => {
                        // min/max are in canvas pixel coordinates
                        let rect_w = (max_x - min_x).max(1.0);
                        let rect_h = (max_y - min_y).max(1.0);
                        let viewport_w = canvas_rect.width();
                        let viewport_h = canvas_rect.height();
                        // Desired zoom: fit the selected region into the viewport
                        let desired_zoom = (viewport_w / rect_w).min(viewport_h / rect_h).clamp(0.1, 100.0);
                        self.zoom = desired_zoom;
                        // Pan so that the rect center is at the viewport center
                        // Image center is at canvas_rect.center() + pan_offset
                        // Canvas point (cx,cy) is at screen: image_center - image_size/2 + cx*zoom
                        // We want rect_center to map to canvas_rect.center():
                        //   canvas_rect.center().x + pan.x - (w*zoom)/2 + rcx*zoom = canvas_rect.center().x
                        //   pan.x = (w*zoom)/2 - rcx*zoom
                        let rcx = (min_x + max_x) / 2.0;
                        let rcy = (min_y + max_y) / 2.0;
                        let image_w = state.width as f32 * self.zoom;
                        let image_h = state.height as f32 * self.zoom;
                        self.pan_offset = Vec2::new(
                            image_w / 2.0 - rcx * self.zoom,
                            image_h / 2.0 - rcy * self.zoom,
                        );
                    }
                }
            }
            
            // Always check if gradient settings changed and re-render preview,
            // even when handle_input is blocked by UI (e.g. context bar interactions)
            if tools.active_tool == crate::components::tools::Tool::Gradient {
                tools.update_gradient_if_dirty(state, Some(&mut self.gpu_renderer));
            }

            // Always check if text/shape properties changed (color picker, context bar)
            if tools.active_tool == crate::components::tools::Tool::Text {
                tools.update_text_if_dirty(state, primary_color_f32);
            }
            if tools.active_tool == crate::components::tools::Tool::Shapes {
                tools.update_shape_if_dirty(state, primary_color_f32, secondary_color_f32);
            }

            // Track fill recalculation state for the loading bar
            self.fill_recalc_active = tools.fill_state.recalc_pending
                && tools.fill_state.active_fill.is_some();

            // Track gradient commit state for the loading bar
            self.gradient_commit_active = tools.gradient_state.commit_pending
                || tools.text_state.commit_pending
                || tools.liquify_state.commit_pending
                || tools.mesh_warp_state.commit_pending;
            
            // ====================================================================
            // TOOL-SPECIFIC CURSOR ICON
            // ====================================================================
            {
                let mouse_pos = ui.input(|i| i.pointer.interact_pos());
                let over_image = mouse_pos.map_or(false, |pos| image_rect.contains(pos));
                // Only override cursor when mouse is truly over just the canvas —
                // not when a dialog, menu, popup, or floating panel is on top.
                if over_image && !modal_open
                    && !ui.ctx().memory(|mem| mem.any_popup_open())
                    && !ui.ctx().is_pointer_over_area()
                {
                    if let Some(tool) = active_tool_for_cursor {
                        use crate::components::tools::Tool;
                        let is_dragging = ui.input(|i| i.pointer.primary_down());
                        let cursor = match tool {
                            // Tools with custom icon cursor — hide OS cursor
                            Tool::Pencil | Tool::Fill | Tool::ColorPicker
                            | Tool::Zoom | Tool::Pan => egui::CursorIcon::None,
                            // Remaining precision tools — crosshair
                            Tool::RectangleSelect | Tool::EllipseSelect
                            | Tool::Lasso | Tool::PerspectiveCrop | Tool::MagicWand
                            | Tool::ColorRemover | Tool::Shapes | Tool::Gradient => egui::CursorIcon::Crosshair,
                            // Line tool: move cursor when on pan handle, crosshair otherwise
                            Tool::Line => {
                                if line_pan_dragging {
                                    egui::CursorIcon::Grabbing
                                } else if line_pan_hovering {
                                    egui::CursorIcon::Move
                                } else {
                                    egui::CursorIcon::Crosshair
                                }
                            }
                            // Brush-type tools — None (custom circle overlay drawn)
                            Tool::Brush | Tool::Eraser | Tool::CloneStamp
                            | Tool::ContentAwareBrush | Tool::Liquify | Tool::Smudge => egui::CursorIcon::None,
                            // Move tools — move/grabbing
                            Tool::MovePixels | Tool::MoveSelection => {
                                if is_dragging { egui::CursorIcon::Grabbing } else { egui::CursorIcon::Move }
                            }
                            // Mesh warp — crosshair for precise control point dragging
                            Tool::MeshWarp => egui::CursorIcon::Crosshair,
                            // Text — move cursor when hovering handle, text caret otherwise
                            Tool::Text => {
                                if text_dragging_handle {
                                    egui::CursorIcon::Grabbing
                                } else if text_hovering_handle {
                                    egui::CursorIcon::Move
                                } else {
                                    egui::CursorIcon::Text
                                }
                            }
                            // Zoom / Pan now handled above (custom icon cursor)
                            // (unreachable — already matched in first arm)
                        };
                        ui.ctx().set_cursor_icon(cursor);
                    }
                }
            }

            // ====================================================================
            // FILL CURSOR OVERLAY (shows exact pixel that will be flood-filled)
            // ====================================================================
            if fill_active {
                let mouse_pos = ui.input(|i| i.pointer.interact_pos());
                if let Some(pos) = mouse_pos {
                    if canvas_rect.contains(pos) {
                        if let Some((canvas_x_f32, canvas_y_f32)) = self.screen_to_canvas_f32(pos, canvas_rect, state) {
                            let canvas_x = canvas_x_f32.floor() as u32;
                            let canvas_y = canvas_y_f32.floor() as u32;
                            let pixel_screen_x = image_rect.min.x + canvas_x as f32 * self.zoom;
                            let pixel_screen_y = image_rect.min.y + canvas_y as f32 * self.zoom;
                            let pixel_rect = Rect::from_min_size(
                                Pos2::new(pixel_screen_x, pixel_screen_y),
                                Vec2::new(self.zoom.max(1.0), self.zoom.max(1.0)),
                            );
                            painter.rect_stroke(pixel_rect, 0.0, egui::Stroke::new(1.0, Color32::from_black_alpha(180)));
                            painter.rect_stroke(pixel_rect, 0.0, egui::Stroke::new(0.5, Color32::from_white_alpha(200)));
                        }
                    }
                }
            }

            // ====================================================================
            // PENCIL CURSOR OVERLAY (shows exact pixel that will be painted)
            // ====================================================================
            // Only show when Pencil tool is active, mouse is over canvas, and not painting
            if pencil_active {
                let mouse_pos = ui.input(|i| i.pointer.interact_pos());
                let is_painting = ui.input(|i| i.pointer.primary_down() || i.pointer.secondary_down());
                
                if let Some(pos) = mouse_pos {
                    if canvas_rect.contains(pos) && !is_painting {
                        // Convert screen position to canvas position (floating point)
                        if let Some((canvas_x_f32, canvas_y_f32)) = self.screen_to_canvas_f32(pos, canvas_rect, state) {
                            // Floor to get the pixel that contains the cursor (not nearest pixel)
                            let canvas_x = canvas_x_f32.floor() as u32;
                            let canvas_y = canvas_y_f32.floor() as u32;
                            
                            // Convert back to screen coordinates for the exact pixel
                            let pixel_screen_x = image_rect.min.x + canvas_x as f32 * self.zoom;
                            let pixel_screen_y = image_rect.min.y + canvas_y as f32 * self.zoom;
                            
                            // Draw a subtle 1x1 pixel outline to show which pixel will be painted
                            let pixel_rect = Rect::from_min_size(
                                Pos2::new(pixel_screen_x, pixel_screen_y),
                                Vec2::new(self.zoom.max(1.0), self.zoom.max(1.0)),
                            );
                            
                            // Use a contrasting color for visibility
                            let cursor_color = if ui.visuals().dark_mode {
                                Color32::from_rgb(255, 255, 255) // White in dark mode
                            } else {
                                Color32::from_rgb(0, 0, 0) // Black in light mode
                            };
                            
                            painter.rect_stroke(pixel_rect, 0.0, egui::Stroke::new(1.0, cursor_color));
                        }
                    }
                }
            }
            
            // ====================================================================
            // COLOR PICKER CURSOR OVERLAY (shows sampled pixel)
            // ====================================================================
            if color_picker_active {
                let mouse_pos = ui.input(|i| i.pointer.interact_pos());
                
                if let Some(pos) = mouse_pos {
                    if canvas_rect.contains(pos) {
                        // Convert screen position to canvas position (floating point)
                        if let Some((canvas_x_f32, canvas_y_f32)) = self.screen_to_canvas_f32(pos, canvas_rect, state) {
                            // Floor to get the pixel that contains the cursor
                            let canvas_x = canvas_x_f32.floor() as u32;
                            let canvas_y = canvas_y_f32.floor() as u32;
                            
                            // Convert back to screen coordinates for the exact pixel
                            let pixel_screen_x = image_rect.min.x + canvas_x as f32 * self.zoom;
                            let pixel_screen_y = image_rect.min.y + canvas_y as f32 * self.zoom;
                            
                            // Draw the pixel outline for color picker
                            let pixel_rect = Rect::from_min_size(
                                Pos2::new(pixel_screen_x, pixel_screen_y),
                                Vec2::new(self.zoom.max(1.0), self.zoom.max(1.0)),
                            );
                            
                            // Use a contrasting color for visibility (brighter cyan/yellow)
                            let cursor_color = if ui.visuals().dark_mode {
                                Color32::from_rgb(0, 255, 255) // Cyan in dark mode
                            } else {
                                Color32::from_rgb(255, 200, 0) // Yellow in light mode
                            };
                            
                            painter.rect_stroke(pixel_rect, 0.0, egui::Stroke::new(1.0, cursor_color));
                        }
                    }
                }
            }

            // ====================================================================
            // TOOL ICON CURSOR OVERLAY
            // Draws the active tool's icon near the cursor for Pencil, Fill,
            // ColorPicker, Zoom, Pan.
            // Crisp border: white icon scaled +3px behind, full-dark icon on top.
            // ====================================================================
            {
                let needs_icon_cursor = active_tool_for_cursor.map_or(false, |t| {
                    use crate::components::tools::Tool;
                    matches!(t, Tool::Pencil | Tool::Fill | Tool::ColorPicker
                        | Tool::Zoom | Tool::Pan)
                });
                if needs_icon_cursor {
                if let Some(ref icon_tex) = self.tool_cursor_icon {
                    let mouse_pos = ui.input(|i| i.pointer.interact_pos());
                    if let Some(pos) = mouse_pos {
                        if canvas_rect.contains(pos) {
                            use crate::components::tools::Tool;
                            let icon_sz = 18.0_f32;
                            // Hot-spot: tip tools anchor bottom-left (tip at cursor),
                            // spatial tools anchor center.
                            let center = match active_tool_for_cursor {
                                Some(Tool::Zoom) | Some(Tool::Pan) =>
                                    Pos2::new(pos.x, pos.y),
                                _ => Pos2::new(pos.x + icon_sz * 0.5 + 1.0, pos.y - icon_sz * 0.5),
                            };
                            let uv = egui::Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(1.0, 1.0));
                            // White border: draw icon 3px larger (centered same point)
                            let border_sz = icon_sz + 4.0;
                            let border_rect = Rect::from_center_size(center, Vec2::splat(border_sz));
                            painter.image(icon_tex.id(), border_rect, uv, Color32::from_white_alpha(230));
                            // Dark icon on top at normal size
                            let icon_rect = Rect::from_center_size(center, Vec2::splat(icon_sz));
                            painter.image(icon_tex.id(), icon_rect, uv, Color32::from_black_alpha(230));
                        }
                    }
                } // if let icon_tex
                } // if needs_icon_cursor
            }

            // ====================================================================
            // BRUSH CURSOR OVERLAY (circle outline matching brush size)
            // Shown for Brush, Eraser, Clone Stamp, Content Aware Brush
            // ====================================================================
            if let Some((brush_size, clone_source, is_circle_tip, ref mask_info, cursor_rotation_deg)) = brush_cursor_info {
                let mouse_pos = ui.input(|i| i.pointer.interact_pos());
                let is_painting = ui.input(|i| i.pointer.primary_down() || i.pointer.secondary_down());

                if let Some(pos) = mouse_pos {
                    if canvas_rect.contains(pos) {
                        if let Some((canvas_x, canvas_y)) = self.screen_to_canvas_f32(pos, canvas_rect, state) {
                            // Brush center in screen coords
                            let screen_cx = image_rect.min.x + canvas_x * self.zoom;
                            let screen_cy = image_rect.min.y + canvas_y * self.zoom;
                            let screen_radius = (brush_size / 2.0) * self.zoom;

                            // Draw brush outline — circle for circle tip, texture overlay for image tips
                            if screen_radius > 1.5 {
                                if is_circle_tip {
                                    // Draw two circles (black + white) for visibility on any background
                                    let stroke_outer = egui::Stroke::new(1.5, Color32::from_black_alpha(160));
                                    let stroke_inner = egui::Stroke::new(0.75, Color32::from_white_alpha(200));
                                    painter.circle_stroke(
                                        Pos2::new(screen_cx, screen_cy),
                                        screen_radius,
                                        stroke_outer,
                                    );
                                    painter.circle_stroke(
                                        Pos2::new(screen_cx, screen_cy),
                                        screen_radius,
                                        stroke_inner,
                                    );
                                } else if let Some((mask_data, mask_sz)) = mask_info {
                                    // Image tip — overlay the mask shape twice (white + black)
                                    // for visibility on any background colour.
                                    let ms = *mask_sz;
                                    let tip_name = match &tip_name_for_cursor {
                                        Some(n) => n.as_str(),
                                        None => "",
                                    };
                                    // Include mask content hash via simple checksum for staleness
                                    let content_hash = mask_data.iter().step_by(7).fold(0u32, |acc, &b| acc.wrapping_add(b as u32));
                                    let key = (tip_name.to_string(), ms, content_hash);
                                    if key != self.brush_tip_cursor_key || self.brush_tip_cursor_tex.is_none() {
                                        // Rebuild cursor textures from mask
                                        let n = (ms * ms) as usize;
                                        let mut rgba_white = vec![0u8; n * 4];
                                        let mut rgba_black = vec![0u8; n * 4];
                                        for i in 0..n {
                                            let a = mask_data[i];
                                            // White version
                                            rgba_white[i * 4]     = 255;
                                            rgba_white[i * 4 + 1] = 255;
                                            rgba_white[i * 4 + 2] = 255;
                                            rgba_white[i * 4 + 3] = a;
                                            // Black version
                                            rgba_black[i * 4]     = 0;
                                            rgba_black[i * 4 + 1] = 0;
                                            rgba_black[i * 4 + 2] = 0;
                                            rgba_black[i * 4 + 3] = a;
                                        }
                                        let opts = egui::TextureOptions {
                                            magnification: egui::TextureFilter::Linear,
                                            minification: egui::TextureFilter::Linear,
                                            ..Default::default()
                                        };
                                        let ci_w = egui::ColorImage::from_rgba_unmultiplied(
                                            [ms as usize, ms as usize], &rgba_white,
                                        );
                                        let ci_b = egui::ColorImage::from_rgba_unmultiplied(
                                            [ms as usize, ms as usize], &rgba_black,
                                        );
                                        if let Some(ref mut tex) = self.brush_tip_cursor_tex {
                                            tex.set(ci_w, opts);
                                        } else {
                                            self.brush_tip_cursor_tex = Some(
                                                ui.ctx().load_texture("brush_tip_cursor_w", ci_w, opts),
                                            );
                                        }
                                        if let Some(ref mut tex) = self.brush_tip_cursor_tex_inv {
                                            tex.set(ci_b, opts);
                                        } else {
                                            self.brush_tip_cursor_tex_inv = Some(
                                                ui.ctx().load_texture("brush_tip_cursor_b", ci_b, opts),
                                            );
                                        }
                                        self.brush_tip_cursor_key = key;
                                    }
                                    let screen_sz = brush_size * self.zoom;
                                    let center = Pos2::new(screen_cx, screen_cy);

                                    // Check if we need rotated drawing
                                    let has_rotation = cursor_rotation_deg.abs() > 0.01;

                                    if has_rotation {
                                        // Rotated cursor: draw a rotated quad using a mesh
                                        let half_sz = screen_sz / 2.0;
                                        let rad = cursor_rotation_deg.to_radians();
                                        let cos_r = rad.cos();
                                        let sin_r = rad.sin();

                                        // Rotated corners: TL, TR, BR, BL
                                        let corners = [
                                            (-half_sz, -half_sz),
                                            ( half_sz, -half_sz),
                                            ( half_sz,  half_sz),
                                            (-half_sz,  half_sz),
                                        ];
                                        let rotated: Vec<Pos2> = corners.iter().map(|&(dx, dy)| {
                                            Pos2::new(
                                                center.x + dx * cos_r - dy * sin_r,
                                                center.y + dx * sin_r + dy * cos_r,
                                            )
                                        }).collect();

                                        let uv_corners = [
                                            Pos2::new(0.0, 0.0),
                                            Pos2::new(1.0, 0.0),
                                            Pos2::new(1.0, 1.0),
                                            Pos2::new(0.0, 1.0),
                                        ];

                                        // Draw white pass
                                        if let Some(ref tex_w) = self.brush_tip_cursor_tex {
                                            let tint_w = Color32::from_rgba_unmultiplied(255, 255, 255, 100);
                                            let mut mesh_w = egui::Mesh::with_texture(tex_w.id());
                                            for i in 0..4 {
                                                mesh_w.vertices.push(egui::epaint::Vertex {
                                                    pos: rotated[i],
                                                    uv: uv_corners[i],
                                                    color: tint_w,
                                                });
                                            }
                                            mesh_w.indices = vec![0, 1, 2, 0, 2, 3];
                                            painter.add(egui::Shape::mesh(mesh_w));
                                        }
                                        // Draw black pass
                                        if let Some(ref tex_b) = self.brush_tip_cursor_tex_inv {
                                            let tint_b = Color32::from_rgba_unmultiplied(255, 255, 255, 100);
                                            let mut mesh_b = egui::Mesh::with_texture(tex_b.id());
                                            for i in 0..4 {
                                                mesh_b.vertices.push(egui::epaint::Vertex {
                                                    pos: rotated[i],
                                                    uv: uv_corners[i],
                                                    color: tint_b,
                                                });
                                            }
                                            mesh_b.indices = vec![0, 1, 2, 0, 2, 3];
                                            painter.add(egui::Shape::mesh(mesh_b));
                                        }
                                    } else {
                                        // No rotation — use simple axis-aligned image draw
                                        let cursor_rect = Rect::from_center_size(
                                            center,
                                            egui::Vec2::splat(screen_sz),
                                        );
                                        let uv = Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0));
                                        // Draw white pass then black pass at low opacity.
                                        // On dark backgrounds the white shows; on light the black shows.
                                        if let Some(ref tex_w) = self.brush_tip_cursor_tex {
                                            let tint_w = Color32::from_rgba_unmultiplied(255, 255, 255, 100);
                                            painter.image(tex_w.id(), cursor_rect, uv, tint_w);
                                        }
                                        if let Some(ref tex_b) = self.brush_tip_cursor_tex_inv {
                                            let tint_b = Color32::from_rgba_unmultiplied(255, 255, 255, 100);
                                            painter.image(tex_b.id(), cursor_rect, uv, tint_b);
                                        }
                                    }
                                } else {
                                    // Fallback: square outline (dual stroke for visibility)
                                    let stamp_rect = Rect::from_center_size(
                                        Pos2::new(screen_cx, screen_cy),
                                        egui::Vec2::splat(screen_radius * 2.0),
                                    );
                                    painter.rect_stroke(stamp_rect, 0.0, egui::Stroke::new(1.5, Color32::from_black_alpha(160)));
                                    painter.rect_stroke(stamp_rect, 0.0, egui::Stroke::new(0.75, Color32::from_white_alpha(200)));
                                }
                            } else {
                                // Very small brush — draw crosshair (dual stroke for visibility)
                                let s = 4.0;
                                let stroke_o = egui::Stroke::new(1.5, Color32::from_black_alpha(160));
                                let stroke_i = egui::Stroke::new(0.75, Color32::from_white_alpha(200));
                                painter.line_segment(
                                    [Pos2::new(screen_cx - s, screen_cy), Pos2::new(screen_cx + s, screen_cy)],
                                    stroke_o,
                                );
                                painter.line_segment(
                                    [Pos2::new(screen_cx - s, screen_cy), Pos2::new(screen_cx + s, screen_cy)],
                                    stroke_i,
                                );
                                painter.line_segment(
                                    [Pos2::new(screen_cx, screen_cy - s), Pos2::new(screen_cx, screen_cy + s)],
                                    stroke_o,
                                );
                                painter.line_segment(
                                    [Pos2::new(screen_cx, screen_cy - s), Pos2::new(screen_cx, screen_cy + s)],
                                    stroke_i,
                                );
                            }

                            // Draw clone stamp source crosshair
                            if let Some(src) = clone_source {
                                let src_sx = image_rect.min.x + src.x * self.zoom;
                                let src_sy = image_rect.min.y + src.y * self.zoom;
                                let cross_size = 8.0;
                                let src_color = Color32::from_rgb(255, 100, 100);
                                let src_stroke = egui::Stroke::new(1.5, src_color);
                                painter.line_segment(
                                    [Pos2::new(src_sx - cross_size, src_sy), Pos2::new(src_sx + cross_size, src_sy)],
                                    src_stroke,
                                );
                                painter.line_segment(
                                    [Pos2::new(src_sx, src_sy - cross_size), Pos2::new(src_sx, src_sy + cross_size)],
                                    src_stroke,
                                );
                                // Also draw source brush circle if painting
                                if is_painting && screen_radius > 1.5 {
                                    // Compute current offset
                                    let offset_x = src.x - canvas_x;
                                    let offset_y = src.y - canvas_y;
                                    let src_brush_cx = image_rect.min.x + (canvas_x + offset_x) * self.zoom;
                                    let src_brush_cy = image_rect.min.y + (canvas_y + offset_y) * self.zoom;
                                    painter.circle_stroke(
                                        Pos2::new(src_brush_cx, src_brush_cy),
                                        screen_radius,
                                        egui::Stroke::new(1.0, Color32::from_rgba_unmultiplied(255, 100, 100, 120)),
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // ====================================================================
        // DYNAMIC DEBUG PANEL  (bottom-right, context-sensitive)
        // ====================================================================
        if debug_settings.show_debug_panel {
            let mouse_pos = ui.input(|i| i.pointer.interact_pos());
            let canvas_pos = mouse_pos.and_then(|pos| self.screen_to_canvas(pos, canvas_rect, state));

            let debug_text = if let Some((cx, cy, sw, sh, rot_deg, scale_pct)) = paste_info {
                // Paste overlay active — show paste-specific info.
                format!(
                    "Paste: {:.0}×{:.0} | Pos: ({:.0}, {:.0}) | Rot: {:.1}° | Scale: {:.0}%",
                    sw, sh, cx, cy, rot_deg, scale_pct
                )
            } else if let Some((w, h)) = sel_drag_info {
                // Selection being drawn — show selection dimensions.
                let mouse_info = if let Some((mx, my)) = canvas_pos {
                    format!(" | Cursor: {}, {}", mx, my)
                } else {
                    String::new()
                };
                format!("Selection: {:.0}×{:.0}{}", w, h, mouse_info)
            } else {
                // Default — build custom info based on settings
                let mut parts = Vec::new();
                
                if debug_settings.debug_show_canvas_size {
                    parts.push(format!("{}×{}", state.width, state.height));
                }
                
                if debug_settings.debug_show_zoom {
                    parts.push(format!("{:.0}%", self.zoom * 100.0));
                }
                
                if let Some((x, y)) = canvas_pos {
                    parts.push(format!("{}, {}", x, y));
                }
                
                if debug_settings.debug_show_fps {
                    parts.push(format!("{:.0} FPS", self.fps));
                }
                
                if debug_settings.debug_show_gpu && debug_settings.gpu_acceleration {
                    let gpu_name = &self.gpu_renderer.ctx.adapter_name;
                    // Shorten GPU name if too long
                    let display_name = if gpu_name.len() > 25 {
                        format!("{}...", &gpu_name[..22])
                    } else {
                        gpu_name.clone()
                    };
                    parts.push(display_name);
                }
                
                parts.join(" | ")
            };

            let text_galley = ui.painter().layout_no_wrap(
                debug_text,
                egui::FontId::monospace(9.0),
                Color32::from_gray(200),
            );
            
            let text_size = text_galley.size();
            let debug_pos = canvas_rect.right_bottom() - Vec2::new(text_size.x + 10.0, text_size.y + 10.0);
            let text_rect = egui::Align2::LEFT_TOP.anchor_rect(egui::Rect::from_min_size(
                debug_pos,
                text_size,
            ));
            
            // Draw semi-transparent black background (more subtle)
            painter.rect_filled(
                text_rect.expand(4.0),
                2.0,
                Color32::from_black_alpha(120),
            );
            
            // Draw text
            painter.galley(debug_pos, text_galley);
            
            // ====================================================================
            // LOADING OPERATIONS BAR (floating above debug panel)
            // Show always for filter/IO ops (user-visible progress),
            // show for fill/gradient only when debug panel is enabled.
            // ====================================================================
            let has_user_ops = pending_filter_jobs > 0 || pending_io_ops > 0;
            let has_debug_ops = self.fill_recalc_active || self.gradient_commit_active;
            if has_user_ops || (debug_settings.debug_show_operations && has_debug_ops) {
                let current_time = ui.input(|i| i.time);
                let mut ops_parts = Vec::new();
                
                if pending_filter_jobs > 0 {
                    let elapsed = if let Some(start) = filter_ops_start_time {
                        format!("{:.1}s", current_time - start)
                    } else {
                        "0.0s".to_string()
                    };
                    let label = if !filter_status_description.is_empty() {
                        filter_status_description.to_string()
                    } else {
                        "Filter".to_string()
                    };
                    ops_parts.push(format!("Processing: {} ({})", label, elapsed));
                } else if pending_io_ops > 0 {
                    let elapsed = if let Some(start) = io_ops_start_time {
                        format!("{:.1}s", current_time - start)
                    } else {
                        "0.0s".to_string()
                    };
                    ops_parts.push(format!("Processing: I/O ({})", elapsed));
                } else if self.fill_recalc_active {
                    ops_parts.push("Updating: Fill Preview".to_string());
                } else if self.gradient_commit_active {
                    ops_parts.push("Committing: Tool".to_string());
                }
                
                let ops_text = ops_parts.join(" | ");
                let ops_galley = ui.painter().layout_no_wrap(
                    ops_text,
                    egui::FontId::monospace(9.0),
                    Color32::from_gray(220),
                );
                
                // Position above the debug panel (if visible), right-aligned
                let debug_panel_height = if debug_settings.debug_show_operations { text_size.y + 10.0 } else { 0.0 };
                let vertical_offset = debug_panel_height + 10.0 + ops_galley.size().y + 4.0;
                let ops_pos = canvas_rect.right_bottom() - Vec2::new(ops_galley.size().x + 10.0, vertical_offset);
                let ops_rect = egui::Align2::LEFT_TOP.anchor_rect(egui::Rect::from_min_size(
                    ops_pos,
                    egui::vec2(ops_galley.size().x, ops_galley.size().y + 4.0),
                ));
                
                // Draw semi-transparent background with slight accent
                painter.rect_filled(
                    ops_rect.expand(4.0),
                    2.0,
                    Color32::from_black_alpha(140),
                );
                
                // Draw animated progress bar
                let time = ui.input(|i| i.time);
                let progress = ((time * 2.0).sin() + 1.0) / 2.0;
                let bar_width = ops_rect.width();
                let bar_rect = egui::Rect::from_min_size(
                    egui::pos2(ops_rect.min.x, ops_rect.max.y - 4.0),
                    egui::vec2(bar_width * progress as f32, 4.0),
                );
                painter.rect_filled(
                    bar_rect,
                    0.0,
                    accent_color,
                );
                
                // Draw text
                painter.galley(ops_pos, ops_galley);
            }
        }
    }
    
    pub fn zoom_in(&mut self) {
        self.zoom = (self.zoom * 1.2).min(100.0);
    }
    
    pub fn zoom_out(&mut self) {
        self.zoom = (self.zoom / 1.2).max(0.1);
    }
    
    pub fn reset_zoom(&mut self) {
        self.zoom = 1.0;
        self.pan_offset = Vec2::ZERO;
    }
    
    pub fn apply_zoom(&mut self, zoom_factor: f32) {
        self.zoom = (self.zoom * zoom_factor).clamp(0.1, 100.0);
    }

    /// Zoom while keeping a screen-space point fixed (e.g. under the mouse cursor).
    /// `anchor` is in screen coordinates, `canvas_rect` is the viewport rect.
    pub fn zoom_around_screen_point(&mut self, zoom_factor: f32, anchor: Pos2, canvas_rect: Rect) {
        let old_zoom = self.zoom;
        self.zoom = (self.zoom * zoom_factor).clamp(0.1, 100.0);
        let actual_factor = self.zoom / old_zoom;
        // The image center in screen space is canvas_rect.center() + pan_offset.
        // After scaling, the anchor point would shift unless we compensate pan_offset.
        // offset from anchor to image-center: (center + pan) - anchor
        // After zoom, that offset scales by actual_factor, so:
        //   new_center = anchor + (old_center - anchor) * factor
        //   new_pan = new_center - canvas_rect.center()
        let old_center = canvas_rect.center() + self.pan_offset;
        let new_center_x = anchor.x + (old_center.x - anchor.x) * actual_factor;
        let new_center_y = anchor.y + (old_center.y - anchor.y) * actual_factor;
        self.pan_offset = Vec2::new(
            new_center_x - canvas_rect.center().x,
            new_center_y - canvas_rect.center().y,
        );
    }

    /// Pan the viewport by a screen-space delta (used by the Pan tool)
    pub fn pan_by(&mut self, delta: Vec2) {
        self.pan_offset += delta;
    }
    
    fn draw_pixel_grid(&self, painter: &egui::Painter, image_rect: Rect, state: &CanvasState, viewport: Rect) {
        let pixel_size = self.zoom;
        
        // Only draw grid if pixels are large enough to see
        if pixel_size < 4.0 {
            return;
        }
        
        // Calculate visible range in image coordinates
        let visible_rect = image_rect.intersect(viewport);
        
        // Convert to pixel coordinates
        let start_x = ((visible_rect.min.x - image_rect.min.x) / pixel_size).floor().max(0.0) as u32;
        let end_x = ((visible_rect.max.x - image_rect.min.x) / pixel_size).ceil().min(state.width as f32) as u32;
        let start_y = ((visible_rect.min.y - image_rect.min.y) / pixel_size).floor().max(0.0) as u32;
        let end_y = ((visible_rect.max.y - image_rect.min.y) / pixel_size).ceil().min(state.height as f32) as u32;
        
        // Grid colors using difference blend approach:
        // Draw black outline + white center line for visibility on any background
        let grid_outline = Color32::from_black_alpha(90);
        let grid_center = Color32::from_white_alpha(100);
        
        // Adaptive stroke width: thinner lines as zoom increases for less visual clutter
        // At zoom 8 (minimum grid display): base thickness (1.2 and 0.6, which is 40% smaller than original 2.0 and 1.0)
        // At zoom 20+ (far zoomed in): minimal thickness (0.5 and 0.3)
        let base_outline = 1.2;
        let base_center = 0.6;
        let reference_zoom = 8.0; // zoom level where we use base thickness
        let outline_stroke = (base_outline * reference_zoom / pixel_size).max(0.5).min(base_outline);
        let center_stroke = (base_center * reference_zoom / pixel_size).max(0.3).min(base_center);
        
        // Draw vertical lines with dual-stroke (black outline + white center)
        for x in start_x..=end_x {
            let screen_x = image_rect.min.x + x as f32 * pixel_size;
            if screen_x >= visible_rect.min.x && screen_x <= visible_rect.max.x {
                let p0 = Pos2::new(screen_x, visible_rect.min.y.max(image_rect.min.y));
                let p1 = Pos2::new(screen_x, visible_rect.max.y.min(image_rect.max.y));
                // Draw black outline first
                painter.line_segment([p0, p1], (outline_stroke, grid_outline));
                // Draw white center line on top
                painter.line_segment([p0, p1], (center_stroke, grid_center));
            }
        }
        
        // Draw horizontal lines with dual-stroke (black outline + white center)
        for y in start_y..=end_y {
            let screen_y = image_rect.min.y + y as f32 * pixel_size;
            if screen_y >= visible_rect.min.y && screen_y <= visible_rect.max.y {
                let p0 = Pos2::new(visible_rect.min.x.max(image_rect.min.x), screen_y);
                let p1 = Pos2::new(visible_rect.max.x.min(image_rect.max.x), screen_y);
                // Draw black outline first
                painter.line_segment([p0, p1], (outline_stroke, grid_outline));
                // Draw white center line on top
                painter.line_segment([p0, p1], (center_stroke, grid_center));
            }
        }
    }

    // ========================================================================
    // GUIDELINES OVERLAY  – center cross + rule-of-thirds
    // ========================================================================

    fn draw_guidelines(&self, painter: &egui::Painter, image_rect: Rect, state: &CanvasState, viewport: Rect) {
        let visible = image_rect.intersect(viewport);
        if visible.width() <= 0.0 || visible.height() <= 0.0 {
            return;
        }

        // Dual-stroke colors for visibility on any background
        let outline_color = Color32::from_black_alpha(100);
        let center_color = Color32::from_white_alpha(160);
        let outline_w = 1.5;
        let center_w = 0.7;

        let w = state.width as f32;
        let h = state.height as f32;

        // Helper: canvas X → screen X
        let sx = |cx: f32| image_rect.min.x + cx * self.zoom;
        // Helper: canvas Y → screen Y
        let sy = |cy: f32| image_rect.min.y + cy * self.zoom;

        // Clamp helpers for the visible viewport
        let clamp_x = |x: f32| x.max(visible.min.x).min(visible.max.x);
        let clamp_y = |y: f32| y.max(visible.min.y).min(visible.max.y);

        // Draw a single dual-stroke line (clipped to visible area)
        let draw_h = |cy: f32| {
            let screen_y = sy(cy);
            if screen_y >= visible.min.y && screen_y <= visible.max.y {
                let p0 = Pos2::new(clamp_x(image_rect.min.x), screen_y);
                let p1 = Pos2::new(clamp_x(image_rect.max.x), screen_y);
                painter.line_segment([p0, p1], (outline_w, outline_color));
                painter.line_segment([p0, p1], (center_w, center_color));
            }
        };
        let draw_v = |cx: f32| {
            let screen_x = sx(cx);
            if screen_x >= visible.min.x && screen_x <= visible.max.x {
                let p0 = Pos2::new(screen_x, clamp_y(image_rect.min.y));
                let p1 = Pos2::new(screen_x, clamp_y(image_rect.max.y));
                painter.line_segment([p0, p1], (outline_w, outline_color));
                painter.line_segment([p0, p1], (center_w, center_color));
            }
        };

        // Center lines
        draw_h(h / 2.0);
        draw_v(w / 2.0);

        // Rule-of-thirds lines
        draw_h(h / 3.0);
        draw_h(h * 2.0 / 3.0);
        draw_v(w / 3.0);
        draw_v(w * 2.0 / 3.0);
    }

    // ========================================================================
    // MIRROR AXIS OVERLAY  – dashed symmetry lines
    // ========================================================================

    fn draw_mirror_overlay(&self, painter: &egui::Painter, image_rect: Rect, state: &CanvasState, viewport: Rect) {
        let visible = image_rect.intersect(viewport);
        if visible.width() <= 0.0 || visible.height() <= 0.0 {
            return;
        }

        let w = state.width as f32;
        let h = state.height as f32;

        // Mirror axis colors — cyan/magenta for high contrast on any content
        let outline_color = Color32::from_rgba_premultiplied(0, 0, 0, 140);
        let line_color_h = Color32::from_rgb(0, 200, 220);   // cyan for horizontal (vertical axis)
        let line_color_v = Color32::from_rgb(220, 80, 220);   // magenta for vertical (horizontal axis)
        let outline_w = 2.5;
        let center_w = 1.2;

        let sx = |cx: f32| image_rect.min.x + cx * self.zoom;
        let sy = |cy: f32| image_rect.min.y + cy * self.zoom;
        let clamp_x = |x: f32| x.max(visible.min.x).min(visible.max.x);
        let clamp_y = |y: f32| y.max(visible.min.y).min(visible.max.y);

        let draw_dashed_h = |cy: f32, color: Color32| {
            let screen_y = sy(cy);
            if screen_y < visible.min.y || screen_y > visible.max.y { return; }
            let x_start = clamp_x(image_rect.min.x);
            let x_end = clamp_x(image_rect.max.x);
            let dash_len = 8.0_f32;
            let gap_len = 4.0_f32;
            let total = dash_len + gap_len;
            let mut x = x_start;
            while x < x_end {
                let x1 = (x + dash_len).min(x_end);
                let p0 = Pos2::new(x, screen_y);
                let p1 = Pos2::new(x1, screen_y);
                painter.line_segment([p0, p1], (outline_w, outline_color));
                painter.line_segment([p0, p1], (center_w, color));
                x += total;
            }
        };

        let draw_dashed_v = |cx: f32, color: Color32| {
            let screen_x = sx(cx);
            if screen_x < visible.min.x || screen_x > visible.max.x { return; }
            let y_start = clamp_y(image_rect.min.y);
            let y_end = clamp_y(image_rect.max.y);
            let dash_len = 8.0_f32;
            let gap_len = 4.0_f32;
            let total = dash_len + gap_len;
            let mut y = y_start;
            while y < y_end {
                let y1 = (y + dash_len).min(y_end);
                let p0 = Pos2::new(screen_x, y);
                let p1 = Pos2::new(screen_x, y1);
                painter.line_segment([p0, p1], (outline_w, outline_color));
                painter.line_segment([p0, p1], (center_w, color));
                y += total;
            }
        };

        match state.mirror_mode {
            MirrorMode::Horizontal => {
                draw_dashed_v(w / 2.0, line_color_h);
            }
            MirrorMode::Vertical => {
                draw_dashed_h(h / 2.0, line_color_v);
            }
            MirrorMode::Quarters => {
                draw_dashed_v(w / 2.0, line_color_h);
                draw_dashed_h(h / 2.0, line_color_v);
            }
            MirrorMode::None => {}
        }
    }

    // ========================================================================
    // SELECTION VISUALIZATION  – marching ants / glow overlay
    // ========================================================================

    /// Build (or reuse) the cached selection overlay RGBA texture, then
    /// draw it as a single GPU-composited quad.  The border segments are
    /// still drawn with immediate-mode lines since they're already
    /// merged into relatively few segments.
    fn draw_selection_overlay(
        &self,
        painter: &egui::Painter,
        image_rect: Rect,
        mask: &GrayImage,
        time: f64,
        tool_active: bool,
        state: &mut CanvasState,
        ctx: &egui::Context,
    ) {
        let w = mask.width();
        let h = mask.height();
        let zoom = self.zoom;

        // --- 1. Find bounding box of selected pixels to limit work. ----------
        let mask_raw = mask.as_raw();
        let stride = w as usize;

        let (min_x, min_y, max_x, max_y) = if let Some(b) = state.selection_overlay_bounds {
            // Fast path: reuse previously computed bounds if generation hasn't changed
            if state.selection_overlay_built_generation == state.selection_overlay_generation {
                b
            } else {
                Self::compute_mask_bounds(mask_raw, w, h, stride)
            }
        } else {
            Self::compute_mask_bounds(mask_raw, w, h, stride)
        };

        if min_x > max_x || min_y > max_y {
            return; // Nothing selected
        }

        let sel = |x: u32, y: u32| -> bool {
            mask_raw[(y as usize) * stride + (x as usize)] > 0
        };

        // --- 2. Selection interior overlay via GPU-cached texture. -----------
        // The crosshatch pattern is baked into an RGBA texture at canvas-pixel
        // resolution (cropped to the selection bounding box).  The texture is
        // rebuilt only when the selection mask changes or the animation offset
        // ticks forward.  The GPU handles zoom/display for free.
        if !tool_active {
            // Animation: slowly scroll pattern.  One pixel of scroll every ~150ms.
            let band_period = 8u32; // canvas-pixel diagonal period
            let anim_offset = ((time * 6.5) as u32 % band_period) as f32;

            let generation_changed = state.selection_overlay_built_generation != state.selection_overlay_generation;
            let anim_changed = (anim_offset - state.selection_overlay_anim_offset).abs() > 0.01;
            let needs_rebuild = state.selection_overlay_texture.is_none()
                || generation_changed
                || anim_changed;

            if needs_rebuild {
                let bw = (max_x - min_x + 1) as usize;
                let bh = (max_y - min_y + 1) as usize;
                let buf_len = bw * bh * 4;

                // Build RGBA buffer with crosshatch pattern
                let mut buf = vec![0u8; buf_len];
                let anim_i = anim_offset as i32;
                for dy in 0..bh {
                    let cy = min_y + dy as u32; // canvas y
                    let row_off = dy * bw * 4;
                    for dx in 0..bw {
                        let cx = min_x + dx as u32; // canvas x
                        if mask_raw[(cy as usize) * stride + (cx as usize)] == 0 {
                            continue; // transparent (unselected)
                        }
                        // Diagonal coordinate in canvas pixels, with animation offset
                        let diag = (cx as i32 + cy as i32 + anim_i) % (band_period as i32 * 2);
                        let diag = if diag < 0 { diag + (band_period as i32 * 2) } else { diag };
                        let is_dark = diag < band_period as i32;
                        let px = row_off + dx * 4;
                        if is_dark {
                            buf[px]     = 0;
                            buf[px + 1] = 0;
                            buf[px + 2] = 0;
                            buf[px + 3] = 40;
                        } else {
                            buf[px]     = 255;
                            buf[px + 1] = 255;
                            buf[px + 2] = 255;
                            buf[px + 3] = 40;
                        }
                    }
                }

                let color_image = ColorImage::from_rgba_unmultiplied(
                    [bw, bh],
                    &buf,
                );
                // Use Nearest filtering so individual pixels stay crisp at high zoom
                let tex_options = TextureOptions {
                    magnification: TextureFilter::Nearest,
                    minification: TextureFilter::Linear,
                    ..Default::default()
                };
                state.selection_overlay_texture = Some(ctx.load_texture(
                    "selection_overlay",
                    ImageData::Color(Arc::new(color_image)),
                    tex_options,
                ));
                state.selection_overlay_built_generation = state.selection_overlay_generation;
                state.selection_overlay_anim_offset = anim_offset;
                state.selection_overlay_bounds = Some((min_x, min_y, max_x, max_y));
            }

            // Paint the cached texture at the correct position.
            if let Some(ref tex) = state.selection_overlay_texture {
                if let Some((bx0, by0, bx1, by1)) = state.selection_overlay_bounds {
                    let screen_x = image_rect.min.x + bx0 as f32 * zoom;
                    let screen_y = image_rect.min.y + by0 as f32 * zoom;
                    let screen_w = (bx1 - bx0 + 1) as f32 * zoom;
                    let screen_h = (by1 - by0 + 1) as f32 * zoom;
                    let sub_rect = Rect::from_min_size(
                        Pos2::new(screen_x, screen_y),
                        Vec2::new(screen_w, screen_h),
                    );
                    let uv = Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0));
                    painter.image(tex.id(), sub_rect, uv, Color32::WHITE);
                }
            }
        } // end if !tool_active

        // --- 3. Collect boundary edges (cached in canvas coordinates). -------
        if state.selection_border_built_generation != state.selection_overlay_generation {
            state.selection_border_h_segs.clear();
            state.selection_border_v_segs.clear();

            // Horizontal edges
            for y_line in min_y..=max_y + 1 {
                let mut seg_x: Option<u32> = None;
                for x in min_x..=max_x {
                    let above = y_line > 0 && y_line - 1 < h && sel(x, y_line - 1);
                    let below = y_line < h && sel(x, y_line);
                    let boundary = above != below;
                    if boundary && seg_x.is_none() {
                        seg_x = Some(x);
                    } else if !boundary {
                        if let Some(sx) = seg_x {
                            state.selection_border_h_segs.push((y_line, sx, x));
                            seg_x = None;
                        }
                    }
                }
                if let Some(sx) = seg_x {
                    state.selection_border_h_segs.push((y_line, sx, max_x + 1));
                }
            }

            // Vertical edges
            for x_line in min_x..=max_x + 1 {
                let mut seg_y: Option<u32> = None;
                for y in min_y..=max_y {
                    let left  = x_line > 0 && x_line - 1 < w && sel(x_line - 1, y);
                    let right = x_line < w && sel(x_line, y);
                    let boundary = left != right;
                    if boundary && seg_y.is_none() {
                        seg_y = Some(y);
                    } else if !boundary {
                        if let Some(sy) = seg_y {
                            state.selection_border_v_segs.push((x_line, sy, y));
                            seg_y = None;
                        }
                    }
                }
                if let Some(sy) = seg_y {
                    state.selection_border_v_segs.push((x_line, sy, max_y + 1));
                }
            }

            state.selection_border_built_generation = state.selection_overlay_generation;
        }

        // --- 4. Draw selection border from cached segments. ------------------
        let accent = self.selection_stroke;
        let stroke_width = 1.5;
        let [sr, sg, sb, _] = accent.to_array();
        let glow_alpha = if tool_active {
            let pulse = ((time * 3.0).sin() * 0.5 + 0.5) as f32;
            (40.0 + pulse * 100.0) as u8
        } else {
            80u8
        };
        let glow = Color32::from_rgba_unmultiplied(sr, sg, sb, glow_alpha);
        let glow_width = if tool_active { stroke_width + 4.0 } else { stroke_width + 2.0 };

        let total_segs = state.selection_border_h_segs.len() + state.selection_border_v_segs.len();
        let skip_glow = total_segs > 10_000;

        // Draw horizontal border segments
        for &(y_line, x0, x1) in &state.selection_border_h_segs {
            let sy = image_rect.min.y + y_line as f32 * zoom;
            let sx0 = image_rect.min.x + x0 as f32 * zoom;
            let sx1 = image_rect.min.x + x1 as f32 * zoom;
            let a = Pos2::new(sx0, sy);
            let b = Pos2::new(sx1, sy);
            if !skip_glow {
                painter.line_segment([a, b], egui::Stroke::new(glow_width, glow));
            }
            painter.line_segment([a, b], egui::Stroke::new(stroke_width + 0.5, accent));
        }

        // Draw vertical border segments
        for &(x_line, y0, y1) in &state.selection_border_v_segs {
            let sx = image_rect.min.x + x_line as f32 * zoom;
            let sy0 = image_rect.min.y + y0 as f32 * zoom;
            let sy1 = image_rect.min.y + y1 as f32 * zoom;
            let a = Pos2::new(sx, sy0);
            let b = Pos2::new(sx, sy1);
            if !skip_glow {
                painter.line_segment([a, b], egui::Stroke::new(glow_width, glow));
            }
            painter.line_segment([a, b], egui::Stroke::new(stroke_width + 0.5, accent));
        }
    }

    /// Compute the bounding box of selected pixels in a mask.
    fn compute_mask_bounds(mask_raw: &[u8], w: u32, h: u32, stride: usize) -> (u32, u32, u32, u32) {
        let mut min_x = w;
        let mut min_y = h;
        let mut max_x = 0u32;
        let mut max_y = 0u32;

        for y in 0..h {
            let row_offset = y as usize * stride;
            let row = &mask_raw[row_offset..row_offset + stride];
            let mut found_in_row = false;
            for (x_idx, &val) in row.iter().enumerate() {
                if val > 0 {
                    let x = x_idx as u32;
                    min_x = min_x.min(x);
                    max_x = max_x.max(x);
                    found_in_row = true;
                }
            }
            if found_in_row {
                min_y = min_y.min(y);
                max_y = max_y.max(y);
            }
        }

        (min_x, min_y, max_x, max_y)
    }

    /// Draw an animated "marching ants" rectangle border using theme colours.
    fn draw_marching_rect(&self, painter: &egui::Painter, rect: Rect, time: f64) {
        let accent = self.selection_stroke;
        let contrast = self.selection_contrast;

        let dash = 6.0f32;
        let gap  = 4.0f32;
        let pattern = dash + gap;
        let offset = ((time * 30.0) % pattern as f64) as f32;
        let stroke_width = 1.5;

        let edges: [(Pos2, Pos2); 4] = [
            (rect.left_top(), rect.right_top()),
            (rect.right_top(), rect.right_bottom()),
            (rect.right_bottom(), rect.left_bottom()),
            (rect.left_bottom(), rect.left_top()),
        ];

        for (start, end) in &edges {
            self.draw_dashed_line(painter, *start, *end, dash, gap, offset, stroke_width, contrast);
            let offset2 = (offset + dash) % pattern;
            self.draw_dashed_line(painter, *start, *end, gap, dash, offset2, stroke_width, accent);
        }
    }

    /// Draw a dashed line segment.
    fn draw_dashed_line(
        &self,
        painter: &egui::Painter,
        a: Pos2,
        b: Pos2,
        dash: f32,
        gap: f32,
        offset: f32,
        width: f32,
        color: Color32,
    ) {
        let dir = b - a;
        let total = dir.length();
        if total < 0.1 { return; }
        let unit = dir / total;
        let pattern = dash + gap;
        let mut t = -offset; // start offset for animation
        while t < total {
            let seg_start = t.max(0.0);
            let seg_end = (t + dash).min(total);
            if seg_start < seg_end {
                let p0 = a + unit * seg_start;
                let p1 = a + unit * seg_end;
                painter.line_segment([p0, p1], egui::Stroke::new(width, color));
            }
            t += pattern;
        }
    }

    /// Draw an ellipse selection preview (filled + dashed border).
    fn draw_ellipse_overlay(
        &self,
        painter: &egui::Painter,
        rect: Rect,
        fill: Color32,
        time: f64,
    ) {
        let center = rect.center();
        let rx = rect.width() / 2.0;
        let ry = rect.height() / 2.0;

        // Approximate the ellipse with a polygon for fill.
        let segments = 64;
        let mut points = Vec::with_capacity(segments);
        for i in 0..segments {
            let angle = 2.0 * std::f32::consts::PI * (i as f32) / (segments as f32);
            points.push(Pos2::new(
                center.x + rx * angle.cos(),
                center.y + ry * angle.sin(),
            ));
        }

        // Fill with triangle fan
        if points.len() >= 3 {
            let mut mesh = egui::Mesh::default();
            for pt in &points {
                mesh.colored_vertex(*pt, fill);
            }
            for i in 1..(points.len() as u32 - 1) {
                mesh.add_triangle(0, i, i + 1);
            }
            painter.add(egui::Shape::mesh(mesh));
        }

        // Dashed border (walk the perimeter) using theme accent + contrast.
        let accent = self.selection_stroke;
        let contrast_col = self.selection_contrast;
        let stroke_width = 1.5;
        let dash = 6.0f32;
        let gap  = 4.0f32;
        let pattern = dash + gap;
        let anim_speed = 30.0;
        let offset = ((time * anim_speed as f64) % pattern as f64) as f32;

        for i in 0..points.len() {
            let a = points[i];
            let b = points[(i + 1) % points.len()];
            let seg_len = (b - a).length();
            if seg_len < 0.1 { continue; }

            self.draw_dashed_line(painter, a, b, dash, gap, offset, stroke_width, contrast_col);
            let offset2 = (offset + dash) % pattern;
            self.draw_dashed_line(painter, a, b, gap, dash, offset2, stroke_width, accent);
        }
    }

    fn draw_checkerboard(&self, painter: &egui::Painter, rect: Rect, clip: Rect, brightness: f32) {
        let checker_size = 10.0 * self.zoom;
        // Apply brightness multiplier to the base grayscale values
        let base_light = 220.0 * brightness;
        let base_dark = 180.0 * brightness;
        let light = Color32::from_gray(base_light.clamp(0.0, 255.0) as u8);
        let dark = Color32::from_gray(base_dark.clamp(0.0, 255.0) as u8);

        // When zoomed out far enough that cells are too small to distinguish,
        // just draw a solid average color.  This avoids generating tens of
        // thousands of rect_filled calls that overwhelm egui's tessellator
        // (e.g. ~41K rects for a 4K canvas at any zoom level).
        if checker_size < 3.0 {
            let avg = ((base_light + base_dark) * 0.5).clamp(0.0, 255.0) as u8;
            painter.rect_filled(rect, 0.0, Color32::from_gray(avg));
            return;
        }
        
        // Clip iteration to the visible viewport to avoid drawing off-screen
        // cells.  At 4× zoom on a 4K image, the full grid is ~384×216 = 83K
        // cells, but only the viewport-visible subset (~50×30) is drawn.
        // This eliminates the need for checker_size inflation that would
        // mismatch the baked checkerboard in eraser preview textures.
        let visible = rect.intersect(clip);
        if visible.is_negative() || visible.width() < 1.0 || visible.height() < 1.0 {
            return;
        }
        
        // Paint the light colour as one solid background rect.  Then draw
        // only the dark squares on top.  Any sub-pixel gap between dark
        // rects now reveals the *light* background — the correct colour —
        // instead of the canvas background colour bleeding through.
        painter.rect_filled(rect, 0.0, light);
        
        // Determine which cell range is visible, anchored to image origin
        // so the pattern aligns with canvas pixel coordinates (matches the
        // baked checkerboard in the eraser preview texture).
        let start_x = ((visible.min.x - rect.min.x) / checker_size).floor() as i32;
        let start_y = ((visible.min.y - rect.min.y) / checker_size).floor() as i32;
        let end_x = ((visible.max.x - rect.min.x) / checker_size).ceil() as i32;
        let end_y = ((visible.max.y - rect.min.y) / checker_size).ceil() as i32;
        
        for y in start_y..end_y {
            for x in start_x..end_x {
                if (x + y) % 2 == 0 { continue; } // light square — already painted
                let checker_rect = Rect::from_min_size(
                    Pos2::new(rect.min.x + x as f32 * checker_size, rect.min.y + y as f32 * checker_size),
                    Vec2::splat(checker_size),
                );
                
                // Only draw if within image bounds
                let intersection = checker_rect.intersect(rect);
                if !intersection.is_negative() {
                    painter.rect_filled(intersection, 0.0, dark);
                }
            }
        }
    }
    
    /// Converts screen position to canvas pixel coordinates
    fn screen_to_canvas(&self, screen_pos: Pos2, canvas_rect: Rect, state: &CanvasState) -> Option<(u32, u32)> {
        let image_width = state.width as f32 * self.zoom;
        let image_height = state.height as f32 * self.zoom;
        
        let center_x = canvas_rect.center().x + self.pan_offset.x;
        let center_y = canvas_rect.center().y + self.pan_offset.y;
        
        let image_rect = Rect::from_center_size(
            Pos2::new(center_x, center_y),
            Vec2::new(image_width, image_height),
        );
        
        if !image_rect.contains(screen_pos) {
            return None;
        }
        
        // Convert to image coordinates
        let rel_x = (screen_pos.x - image_rect.min.x) / self.zoom;
        let rel_y = (screen_pos.y - image_rect.min.y) / self.zoom;
        
        let pixel_x = rel_x as u32;
        let pixel_y = rel_y as u32;
        
        if pixel_x < state.width && pixel_y < state.height {
            Some((pixel_x, pixel_y))
        } else {
            None
        }
    }

    /// Public wrapper for screen_to_canvas conversion (used by app.rs for move tools).
    pub fn screen_to_canvas_pub(&self, screen_pos: Pos2, canvas_rect: Rect, state: &CanvasState) -> Option<(u32, u32)> {
        self.screen_to_canvas(screen_pos, canvas_rect, state)
    }

    /// Public wrapper for sub-pixel screen_to_canvas (used by app.rs for paste-at-cursor).
    pub fn screen_to_canvas_f32_pub(&self, screen_pos: Pos2, canvas_rect: Rect, state: &CanvasState) -> Option<(f32, f32)> {
        self.screen_to_canvas_f32(screen_pos, canvas_rect, state)
    }

    /// Like `screen_to_canvas` but returns sub-pixel float coordinates for
    /// smooth brush strokes.
    fn screen_to_canvas_f32(&self, screen_pos: Pos2, canvas_rect: Rect, state: &CanvasState) -> Option<(f32, f32)> {
        let image_width = state.width as f32 * self.zoom;
        let image_height = state.height as f32 * self.zoom;

        let center_x = canvas_rect.center().x + self.pan_offset.x;
        let center_y = canvas_rect.center().y + self.pan_offset.y;

        let image_rect = Rect::from_center_size(
            Pos2::new(center_x, center_y),
            Vec2::new(image_width, image_height),
        );

        if !image_rect.contains(screen_pos) {
            return None;
        }

        let rel_x = (screen_pos.x - image_rect.min.x) / self.zoom;
        let rel_y = (screen_pos.y - image_rect.min.y) / self.zoom;

        if rel_x >= 0.0 && rel_x < state.width as f32 && rel_y >= 0.0 && rel_y < state.height as f32 {
            Some((rel_x, rel_y))
        } else {
            None
        }
    }

    /// Like `screen_to_canvas_f32` but allows coordinates up to and slightly
    /// beyond canvas bounds (clamped to `[0, width]` × `[0, height]`).
    /// Used by overlay tools (mesh warp) whose control points sit on canvas edges.
    fn screen_to_canvas_f32_clamped(&self, screen_pos: Pos2, canvas_rect: Rect, state: &CanvasState, margin_px: f32) -> Option<(f32, f32)> {
        let image_width = state.width as f32 * self.zoom;
        let image_height = state.height as f32 * self.zoom;

        let center_x = canvas_rect.center().x + self.pan_offset.x;
        let center_y = canvas_rect.center().y + self.pan_offset.y;

        let image_rect = Rect::from_center_size(
            Pos2::new(center_x, center_y),
            Vec2::new(image_width, image_height),
        );

        // Expand the image rect by the margin (screen pixels) so clicks
        // slightly outside the canvas edge are still accepted.
        let expanded = image_rect.expand(margin_px);
        if !expanded.contains(screen_pos) {
            return None;
        }

        let rel_x = (screen_pos.x - image_rect.min.x) / self.zoom;
        let rel_y = (screen_pos.y - image_rect.min.y) / self.zoom;

        // Clamp to inclusive canvas bounds [0, width] × [0, height]
        let cx = rel_x.clamp(0.0, state.width as f32);
        let cy = rel_y.clamp(0.0, state.height as f32);
        Some((cx, cy))
    }
}

/// Converts an RgbaImage to egui's ColorImage format
fn rgba_image_to_color_image(img: &RgbaImage) -> ColorImage {
    let size = [img.width() as usize, img.height() as usize];
    let pixels = img.as_flat_samples();
    let rgba_data = pixels.as_slice();
    
    let color_pixels: Vec<Color32> = rgba_data
        .chunks_exact(4)
        .map(|chunk| Color32::from_rgba_unmultiplied(chunk[0], chunk[1], chunk[2], chunk[3]))
        .collect();
    
    ColorImage {
        size,
        pixels: color_pixels,
    }
}
