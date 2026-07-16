pub struct FillCommitOverlay {
    pub bounds: (u32, u32, u32, u32),
    pub rgba: Vec<u8>,
    pub started_at: crate::time_compat::Instant,
    pub duration: std::time::Duration,
    pub texture: Option<egui::TextureHandle>,
}

pub struct CanvasState {
    pub layers: Vec<Layer>,
    pub layer_folders: Vec<LayerFolder>,
    pub next_layer_folder_id: u64,
    pub active_layer_index: usize,
    pub edit_layer_mask: bool,
    pub width: u32,
    pub height: u32,
    pub composite_cache: Option<egui::TextureHandle>,
    pub dirty_rect: Option<egui::Rect>,
    pub commit_composite_flush_rect: Option<egui::Rect>,
    pub show_pixel_grid: bool,             // Toggle for pixel grid overlay
    pub show_guidelines: bool,             // Toggle for center/thirds guidelines overlay
    pub mirror_mode: MirrorMode,           // Symmetry mirror mode
    pub show_wrap_preview: bool,           // Toggle for 4-side seamless edge preview
    pub preview_layer: Option<TiledImage>, // For non-destructive tool previews (e.g., Bézier curves)
    pub preview_blend_mode: BlendMode,     // Blend mode for the preview layer
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
    /// When true, preview alpha represents edits to the active layer's mask
    /// (conceal/reveal) rather than paint blended directly into layer pixels.
    pub preview_targets_mask: bool,
    /// Meaningful only when `preview_targets_mask` is true.
    /// false = increase conceal (paint mask), true = decrease conceal (erase/reveal mask).
    pub preview_mask_reveal: bool,
    /// Monotonically increasing counter, bumped on each mark_dirty call
    pub dirty_generation: u64,
    /// Selection mask – 0 = unselected, 255 = fully selected.
    /// Dimensions must match (width, height).
    pub selection_mask: Option<GrayImage>,
    /// Semantic full-canvas selection. Avoids allocating/scanning a canvas-sized mask.
    pub selection_all: bool,
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
    /// Bounds-only preview used while interactively resizing a selection.
    pub selection_transform_preview_bounds: Option<(u32, u32, u32, u32)>,
    /// Short-lived fill overlays used to visually bridge preview commits.
    pub fill_commit_overlays: Vec<FillCommitOverlay>,
    /// Cached border segments in canvas coordinates: (line_pos, seg_start, seg_end).
    /// Horizontal segments: y_line, x_start, x_end.
    /// Vertical segments: x_line, y_start, y_end.
    pub selection_border_h_segs: Vec<(u32, u32, u32)>,
    pub selection_border_v_segs: Vec<(u32, u32, u32)>,
    /// Generation at which border segments were last computed.
    pub selection_border_built_generation: u64,
    /// When true, the display applies an RGB→CMYK→RGB round-trip (soft proof)
    /// so the user can preview how the image will look when printed in CMYK.
    /// Does not modify actual pixel data — display-only.
    pub cmyk_preview: bool,

    // -- Text layer rasterization caches ----------------------
    /// Reusable coverage buffer for text rasterization (avoids per-rasterize alloc).
    pub text_coverage_buf: Vec<f32>,
    /// Reusable glyph pixel cache for text rasterization.
    pub text_glyph_cache: crate::ops::text::GlyphPixelCache,
    /// The layer index currently being edited by the text tool (skip rasterization for it).
    pub text_editing_layer: Option<usize>,
    /// Widget ID of the canvas painter (set by Canvas::show_with_state).
    /// Used by tools to detect when a non-canvas widget has keyboard focus.
    pub canvas_widget_id: Option<egui::Id>,
}

impl CanvasState {
    pub fn new(width: u32, height: u32) -> Self {
        let white = Rgba([255, 255, 255, 255]);
        let background = Layer::new("Background".to_string(), width, height, white);

        Self {
            layers: vec![background],
            layer_folders: Vec::new(),
            next_layer_folder_id: 1,
            active_layer_index: 0,
            edit_layer_mask: false,
            width,
            height,
            composite_cache: None,
            dirty_rect: None,
            commit_composite_flush_rect: None,
            show_pixel_grid: true,  // Enable by default
            show_guidelines: false, // Disabled by default
            mirror_mode: MirrorMode::None,
            show_wrap_preview: false,
            preview_layer: None,
            preview_blend_mode: BlendMode::Normal,
            preview_force_composite: false,
            preview_is_eraser: false,
            preview_replaces_layer: false,
            preview_targets_mask: false,
            preview_mask_reveal: false,
            dirty_generation: 0,
            selection_mask: None,
            selection_all: false,
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
            selection_transform_preview_bounds: None,
            fill_commit_overlays: Vec::new(),
            selection_border_h_segs: Vec::new(),
            selection_border_v_segs: Vec::new(),
            selection_border_built_generation: u64::MAX, // force first compute
            cmyk_preview: false,
            text_coverage_buf: Vec::new(),
            text_glyph_cache: Default::default(),
            text_editing_layer: None,
            canvas_widget_id: None,
        }
    }

    pub fn layer_folder(&self, folder_id: u64) -> Option<&LayerFolder> {
        self.layer_folders
            .iter()
            .find(|folder| folder.id == folder_id)
    }

    pub fn layer_folder_mut(&mut self, folder_id: u64) -> Option<&mut LayerFolder> {
        self.layer_folders
            .iter_mut()
            .find(|folder| folder.id == folder_id)
    }

    pub fn layer_effectively_visible(&self, layer_idx: usize) -> bool {
        let Some(layer) = self.layers.get(layer_idx) else {
            return false;
        };
        if !layer.visible {
            return false;
        }
        layer
            .folder_id
            .and_then(|id| self.layer_folder(id))
            .is_none_or(|folder| folder.visible)
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
        self.preview_targets_mask = false;
        self.preview_mask_reveal = false;
        self.selection_transform_preview_bounds = None;
    }

    pub fn push_fill_commit_overlay(
        &mut self,
        bounds: (u32, u32, u32, u32),
        rgba: Vec<u8>,
        duration: std::time::Duration,
    ) {
        let width = bounds.2.saturating_sub(bounds.0) + 1;
        let height = bounds.3.saturating_sub(bounds.1) + 1;
        if width == 0 || height == 0 || rgba.is_empty() {
            return;
        }
        self.fill_commit_overlays.push(FillCommitOverlay {
            bounds,
            rgba,
            started_at: crate::time_compat::Instant::now(),
            duration,
            texture: None,
        });
    }

    /// Mirror all populated pixels in the preview layer according to the
    /// current mirror mode.  Called after generating preview content for
    /// tools like Fill, Text, Shapes, Gradient, etc.
    /// Returns the expanded bounding rect that covers all mirrored regions.
    pub fn mirror_preview_layer(&mut self) -> Option<egui::Rect> {
        let mode = self.mirror_mode;
        if mode == MirrorMode::None {
            return None;
        }
        let w = self.width;
        let h = self.height;

        let preview = self.preview_layer.as_mut()?;

        // Collect all existing non-empty pixels from populated chunks
        let chunk_data: Vec<(u32, u32, Vec<(u32, u32, image::Rgba<u8>)>)> = preview
            .chunk_keys()
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

    /// Ensure all text layers have up-to-date rasterized pixels.
    /// Call before compositing so that text layer `.pixels` reflect current text data.
    pub fn ensure_text_layers_rasterized(&mut self) {
        let w = self.width;
        let h = self.height;
        for layer in &mut self.layers {
            if let LayerContent::Text(ref text_data) = layer.content
                && text_data.needs_rasterize()
            {
                // We need a mutable borrow of `text_data` inside `layer.content`,
                // plus the shared rasterization caches on `self`. To avoid borrow
                // conflicts we take ownership of the caches for the duration of
                // the rasterization, then put them back.
                //
                // Because this loop also borrows `self.layers` mutably, we can't
                // directly access `self.text_coverage_buf` etc. So we do a
                // two-pass: collect indices that need rasterization, then rasterize
                // in a second loop.
                //
                // Actually, we restructure below to avoid the double-loop — see
                // the for-index loop that follows.
                break; // fall through to index-based loop
            }
        }

        // Index-based loop to allow mutable access to both layer and caches.
        let len = self.layers.len();
        for i in 0..len {
            // Skip the layer being actively edited by the text tool — it has its
            // own preview pipeline and rasterizing it every frame is expensive.
            if self.text_editing_layer == Some(i) {
                continue;
            }
            let needs = if let LayerContent::Text(ref td) = self.layers[i].content {
                td.needs_rasterize()
            } else {
                false
            };
            if needs {
                // Clear glyph cache when text content changed (font switch
                // produces different outlines for the same GlyphId).
                let text_changed = if let LayerContent::Text(ref td) = self.layers[i].content {
                    td.text_content_generation != td.cached_text_generation
                } else {
                    false
                };
                if text_changed {
                    self.text_glyph_cache.clear();
                }
                // Temporarily take caches out of self
                let mut cov = std::mem::take(&mut self.text_coverage_buf);
                let mut gc = std::mem::take(&mut self.text_glyph_cache);

                if let LayerContent::Text(ref mut text_data) = self.layers[i].content {
                    let new_pixels = text_data.rasterize(w, h, &mut cov, &mut gc);
                    text_data.raster_generation = text_data.cache_generation;
                    self.layers[i].pixels = new_pixels;
                    self.layers[i].invalidate_lod();
                    self.layers[i].gpu_generation += 1;
                }

                // Put caches back
                self.text_coverage_buf = cov;
                self.text_glyph_cache = gc;
            }
        }
    }

    /// Ensure ALL text layers are rasterized, including the one being
    /// actively edited.  Use before save / export / print so that the
    /// editing layer's pixels are guaranteed to be up-to-date in the
    /// composite.  This does NOT alter TextLayerData — it only refreshes
    /// `layer.pixels`, so editing can continue normally afterwards.
    pub fn ensure_all_text_layers_rasterized(&mut self) {
        // Unconditionally rasterize the editing layer — bypasses
        // needs_rasterize() because the previous frame already synced
        // generations, but we must guarantee layer.pixels is populated
        // for the save-dialog composite.
        if let Some(idx) = self.text_editing_layer {
            let w = self.width;
            let h = self.height;
            let is_text = matches!(
                self.layers.get(idx).map(|l| &l.content),
                Some(LayerContent::Text(_))
            );
            if is_text {
                let mut cov = std::mem::take(&mut self.text_coverage_buf);
                let mut gc = std::mem::take(&mut self.text_glyph_cache);
                if let LayerContent::Text(ref mut text_data) = self.layers[idx].content {
                    let new_pixels = text_data.rasterize(w, h, &mut cov, &mut gc);
                    text_data.raster_generation = text_data.cache_generation;
                    self.layers[idx].pixels = new_pixels;
                    self.layers[idx].invalidate_lod();
                    self.layers[idx].gpu_generation += 1;
                }
                self.text_coverage_buf = cov;
                self.text_glyph_cache = gc;
            }
        }
        let saved = self.text_editing_layer.take();
        self.ensure_text_layers_rasterized();
        self.text_editing_layer = saved;
    }

    /// Force-rasterize a specific text layer by index, even if it is the
    /// currently-editing layer.  Used to show live effects/warp changes
    /// during text editing.
    pub fn force_rasterize_text_layer(&mut self, layer_idx: usize) {
        let w = self.width;
        let h = self.height;
        let needs = if let Some(layer) = self.layers.get(layer_idx)
            && let LayerContent::Text(ref td) = layer.content
        {
            td.needs_rasterize()
        } else {
            false
        };
        if needs {
            // Clear glyph cache when text content changed (font switch produces
            // different outlines for the same GlyphId, which the cache keys
            // cannot distinguish).
            let text_changed = if let Some(layer) = self.layers.get(layer_idx)
                && let LayerContent::Text(ref td) = layer.content
            {
                td.text_content_generation != td.cached_text_generation
            } else {
                false
            };
            if text_changed {
                self.text_glyph_cache.clear();
            }
            let mut cov = std::mem::take(&mut self.text_coverage_buf);
            let mut gc = std::mem::take(&mut self.text_glyph_cache);
            if let LayerContent::Text(ref mut text_data) = self.layers[layer_idx].content {
                let new_pixels = text_data.rasterize(w, h, &mut cov, &mut gc);
                text_data.raster_generation = text_data.cache_generation;
                self.layers[layer_idx].pixels = new_pixels;
                self.layers[layer_idx].invalidate_lod();
                self.layers[layer_idx].gpu_generation += 1;
            }
            self.text_coverage_buf = cov;
            self.text_glyph_cache = gc;
        }
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
            let max_cx = (vp.max.x.ceil() as u32)
                .min(self.width)
                .div_ceil(CHUNK_SIZE);
            let max_cy = (vp.max.y.ceil() as u32)
                .min(self.height)
                .div_ceil(CHUNK_SIZE);
            (min_cx, min_cy, max_cx, max_cy)
        } else {
            (
                0,
                0,
                self.width.div_ceil(CHUNK_SIZE),
                self.height.div_ceil(CHUNK_SIZE),
            )
        };

        // Collect unique chunk keys from visible layers within viewport
        let mut active_chunks: Vec<(u32, u32)> = Vec::new();
        for (idx, layer) in self.layers.iter().enumerate() {
            if !self.layer_effectively_visible(idx) {
                continue;
            }
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

        let layer_visibility: Vec<bool> = (0..self.layers.len())
            .map(|idx| self.layer_effectively_visible(idx))
            .collect();
        let layers = &self.layers;
        let preview = &self.preview_layer;
        let preview_blend = self.preview_blend_mode;
        let preview_is_eraser = self.preview_is_eraser;
        let preview_replaces = self.preview_replaces_layer;
        let active_idx = self.active_layer_index;
        let img_w = self.width;
        let img_h = self.height;

        // Process chunks in parallel
        let chunk_results: Vec<_> = active_chunks
            .par_iter()
            .map(|&(cx, cy)| {
                let base_x = cx * CHUNK_SIZE;
                let base_y = cy * CHUNK_SIZE;
                let cw = CHUNK_SIZE.min(img_w.saturating_sub(base_x));
                let ch = CHUNK_SIZE.min(img_h.saturating_sub(base_y));

                let mut pixels = vec![Rgba([0u8, 0, 0, 0]); (cw * ch) as usize];

                for (li, layer) in layers.iter().enumerate() {
                    if !layer_visibility.get(li).copied().unwrap_or(false) {
                        continue;
                    }
                    if let LayerContent::Adjustment(adj) = &layer.content {
                        for px in pixels.iter_mut() {
                            *px = adj.apply_to_pixel_with_opacity(*px, layer.opacity);
                        }
                        continue;
                    }

                    let is_active = li == active_idx;
                    let layer_chunk = layer.pixels.get_chunk(cx, cy);
                    let mask_chunk = if layer.mask_enabled {
                        layer.mask.as_ref().and_then(|m| m.get_chunk(cx, cy))
                    } else {
                        None
                    };
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
                                    } else if matches!(
                                        preview_blend,
                                        BlendMode::Overwrite | BlendMode::Xor
                                    ) {
                                        // Coverage-weighted lerp: smoothly transition
                                        // from original layer pixel to Overwrite result
                                        // using the preview pixel's alpha as coverage.
                                        // Hardness, flow, and brush geometry are all
                                        // already baked into pp[3] via the brush LUT,
                                        // so edge pixels (low coverage) barely change
                                        // the original while interior pixels strongly
                                        // shift toward the Overwrite/Xor result.
                                        let ow =
                                            Self::blend_pixel_static(top, pp, preview_blend, 1.0);
                                        let cov = pp[3] as f32 / 255.0;
                                        let inv = 1.0 - cov;
                                        top = Rgba([
                                            (top[0] as f32 * inv + ow[0] as f32 * cov + 0.5) as u8,
                                            (top[1] as f32 * inv + ow[1] as f32 * cov + 0.5) as u8,
                                            (top[2] as f32 * inv + ow[2] as f32 * cov + 0.5) as u8,
                                            (top[3] as f32 * inv + ow[3] as f32 * cov + 0.5) as u8,
                                        ]);
                                    } else {
                                        top = Self::blend_pixel_static(top, pp, preview_blend, 1.0);
                                    }
                                }
                            }

                            if let Some(mchunk) = mask_chunk {
                                let conceal = mchunk.get_pixel(lx, ly)[3];
                                if conceal > 0 {
                                    top[3] = ((top[3] as u32 * (255 - conceal as u32)) / 255) as u8;
                                }
                            }

                            if opaque_overwrite && top[3] == 255 {
                                pixels[idx] = top;
                            } else {
                                pixels[idx] = Self::blend_pixel_static(
                                    pixels[idx],
                                    top,
                                    layer.blend_mode,
                                    layer.opacity,
                                );
                            }
                        }
                    }
                }

                (cx, cy, cw, ch, pixels)
            })
            .collect();

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
    pub fn composite_partial_downscaled(&self, rect: Rect, scale: u32) -> (ColorImage, [usize; 2]) {
        let min_x = rect.min.x.floor() as u32;
        let min_y = rect.min.y.floor() as u32;
        let max_x = (rect.max.x.ceil() as u32).min(self.width);
        let max_y = (rect.max.y.ceil() as u32).min(self.height);

        let source_w = max_x - min_x;
        let source_h = max_y - min_y;
        let target_w = source_w.div_ceil(scale);
        let target_h = source_h.div_ceil(scale);

        let layer_visibility: Vec<bool> = (0..self.layers.len())
            .map(|idx| self.layer_effectively_visible(idx))
            .collect();
        let layers = &self.layers;
        let preview_layer = &self.preview_layer;
        let preview_blend_mode = self.preview_blend_mode;
        let preview_is_eraser = self.preview_is_eraser;
        let preview_replaces = self.preview_replaces_layer;
        let preview_targets_mask = self.preview_targets_mask;
        let preview_mask_reveal = self.preview_mask_reveal;
        let active_layer_index = self.active_layer_index;

        let row_len = target_w as usize;
        let mut pixels = vec![Color32::TRANSPARENT; (target_w * target_h) as usize];

        pixels
            .par_chunks_mut(row_len)
            .enumerate()
            .for_each(|(row_idx, row)| {
                // Map output row → source canvas coordinate
                let y = min_y + (row_idx as u32) * scale;
                for (out_x, pixel) in row.iter_mut().enumerate().take(target_w as usize) {
                    let x = min_x + (out_x as u32) * scale;

                    // Opaque-base optimisation: find the deepest fully-opaque
                    // normal-blend pixel to skip layers beneath it
                    let mut start_layer_idx = 0;
                    if !preview_targets_mask {
                        for (idx, layer) in layers.iter().enumerate().rev() {
                            if !layer_visibility.get(idx).copied().unwrap_or(false) {
                                continue;
                            }
                            if preview_is_eraser
                                && preview_layer.is_some()
                                && idx == active_layer_index
                            {
                                continue;
                            }
                            if layer.blend_mode == BlendMode::Normal && layer.opacity >= 1.0 {
                                let mut effective_a = layer.pixels.get_pixel(x, y)[3];
                                if effective_a == 255
                                    && layer.mask_enabled
                                    && let Some(mask) = &layer.mask
                                {
                                    let conceal = mask.get_pixel(x, y)[3];
                                    if conceal > 0 {
                                        effective_a =
                                            ((effective_a as u32 * (255 - conceal as u32)) / 255)
                                                as u8;
                                    }
                                }
                                if effective_a == 255 {
                                    start_layer_idx = idx;
                                    break;
                                }
                            }
                        }
                    }

                    let mut base = Rgba([0, 0, 0, 0]);

                    for (li, layer) in layers.iter().enumerate().skip(start_layer_idx) {
                        if !layer_visibility.get(li).copied().unwrap_or(false) {
                            continue;
                        }
                        if let LayerContent::Adjustment(adj) = &layer.content {
                            base = adj.apply_to_pixel_with_opacity(base, layer.opacity);
                            continue;
                        }
                        let mut top = *layer.pixels.get_pixel(x, y);
                        let mut preview_conceal_override: Option<u8> = None;

                        if li == active_layer_index
                            && let Some(preview) = preview_layer
                        {
                            // Preview is at reduced resolution — sample
                            // at the output coordinate directly.
                            let pp = *preview.get_pixel(out_x as u32, row_idx as u32);
                            if preview_targets_mask {
                                if pp[3] > 0 {
                                    let old_conceal = if layer.mask_enabled {
                                        layer.mask.as_ref().map_or(0, |m| m.get_pixel(x, y)[3])
                                    } else {
                                        0
                                    };
                                    let old = old_conceal as u32;
                                    let s = pp[3] as u32;
                                    let new_conceal = if preview_mask_reveal {
                                        ((old * (255 - s)) / 255) as u8
                                    } else {
                                        (old + ((255 - old) * s) / 255) as u8
                                    };
                                    preview_conceal_override = Some(new_conceal);
                                }
                            } else if preview_replaces {
                                top = pp;
                            } else if pp[3] > 0 {
                                if preview_is_eraser {
                                    let mask_strength = pp[3] as f32 / 255.0;
                                    let current_a = top[3] as f32 / 255.0;
                                    let new_a = (current_a * (1.0 - mask_strength)).max(0.0);
                                    top[3] = (new_a * 255.0) as u8;
                                } else if matches!(
                                    preview_blend_mode,
                                    BlendMode::Overwrite | BlendMode::Xor
                                ) {
                                    let ow =
                                        Self::blend_pixel_static(top, pp, preview_blend_mode, 1.0);
                                    let cov = pp[3] as f32 / 255.0;
                                    let inv = 1.0 - cov;
                                    top = Rgba([
                                        (top[0] as f32 * inv + ow[0] as f32 * cov + 0.5) as u8,
                                        (top[1] as f32 * inv + ow[1] as f32 * cov + 0.5) as u8,
                                        (top[2] as f32 * inv + ow[2] as f32 * cov + 0.5) as u8,
                                        (top[3] as f32 * inv + ow[3] as f32 * cov + 0.5) as u8,
                                    ]);
                                } else {
                                    top =
                                        Self::blend_pixel_static(top, pp, preview_blend_mode, 1.0);
                                }
                            }
                        }

                        let conceal = if let Some(c) = preview_conceal_override {
                            c
                        } else if layer.mask_enabled {
                            layer.mask.as_ref().map_or(0, |m| m.get_pixel(x, y)[3])
                        } else {
                            0
                        };
                        if conceal > 0 {
                            top[3] = ((top[3] as u32 * (255 - conceal as u32)) / 255) as u8;
                        }

                        base = Self::blend_pixel_static(base, top, layer.blend_mode, layer.opacity);
                    }

                    let a = base[3];
                    if a == 255 || a == 0 {
                        *pixel = Color32::from_rgba_premultiplied(base[0], base[1], base[2], a);
                    } else {
                        let pm_r = ((base[0] as u16 * a as u16 + 127) / 255) as u8;
                        let pm_g = ((base[1] as u16 * a as u16 + 127) / 255) as u8;
                        let pm_b = ((base[2] as u16 * a as u16 + 127) / 255) as u8;
                        *pixel = Color32::from_rgba_premultiplied(pm_r, pm_g, pm_b, a);
                    }
                }
            });

        (
            ColorImage {
                size: [target_w as usize, target_h as usize],
                source_size: egui::Vec2::new(target_w as f32, target_h as f32),
                pixels,
            },
            [min_x as usize, min_y as usize],
        )
    }

    /// Render ONLY the modified area for partial texture updates.
    /// Processes chunk-by-chunk within the dirty rect for efficient batch access.
    pub fn composite_partial(&self, rect: Rect) -> (ColorImage, [usize; 2]) {
        // 1. Clamp rect to canvas bounds
        let min_x = rect.min.x.floor() as u32;
        let min_y = rect.min.y.floor() as u32;
        let max_x = (rect.max.x.ceil() as u32).min(self.width);
        let max_y = (rect.max.y.ceil() as u32).min(self.height);

        let width = max_x - min_x;
        let height = max_y - min_y;

        // Capture shared state for the parallel closure
        let layer_visibility: Vec<bool> = (0..self.layers.len())
            .map(|idx| self.layer_effectively_visible(idx))
            .collect();
        let layers = &self.layers;
        let preview_layer = &self.preview_layer;
        let preview_blend_mode = self.preview_blend_mode;
        let preview_is_eraser = self.preview_is_eraser;
        let preview_replaces = self.preview_replaces_layer;
        let preview_targets_mask = self.preview_targets_mask;
        let preview_mask_reveal = self.preview_mask_reveal;
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
                            layer
                                .pixels
                                .get_chunk(cx, cy)
                                .map(|c| c.as_raw().as_slice()),
                        );
                    }

                    // Pre-fetch preview layer chunk
                    let preview_raw = preview_layer
                        .as_ref()
                        .and_then(|p| p.get_chunk(cx, cy).map(|c| c.as_raw().as_slice()));

                    let row_byte_off = (ly * CHUNK_SIZE * 4) as usize;

                    for x in span_start..span_end {
                        let lx = x - cx * CHUNK_SIZE;
                        let px_off = row_byte_off + (lx as usize) * 4;
                        let out_idx = (x - min_x) as usize;

                        // Opaque-base optimisation: find deepest opaque Normal pixel
                        let mut start_layer_idx = 0;
                        if !preview_targets_mask {
                            for (idx, layer) in layers.iter().enumerate().rev() {
                                if !layer_visibility.get(idx).copied().unwrap_or(false) {
                                    continue;
                                }
                                if preview_is_eraser
                                    && preview_layer.is_some()
                                    && idx == active_layer_index
                                {
                                    continue;
                                }
                                if layer.blend_mode == BlendMode::Normal
                                    && layer.opacity >= 1.0
                                    && let Some(raw) = chunk_raws[idx]
                                {
                                    let mut effective_a = raw[px_off + 3];
                                    if effective_a == 255
                                        && layer.mask_enabled
                                        && let Some(mask) = &layer.mask
                                    {
                                        let conceal = mask.get_pixel(x, y)[3];
                                        if conceal > 0 {
                                            effective_a = ((effective_a as u32
                                                * (255 - conceal as u32))
                                                / 255)
                                                as u8;
                                        }
                                    }
                                    if effective_a == 255 {
                                        start_layer_idx = idx;
                                        break;
                                    }
                                }
                            }
                        }

                        let mut base = Rgba([0, 0, 0, 0]);

                        for (li, layer) in layers.iter().enumerate().skip(start_layer_idx) {
                            if !layer_visibility.get(li).copied().unwrap_or(false) {
                                continue;
                            }
                            if let LayerContent::Adjustment(adj) = &layer.content {
                                base = adj.apply_to_pixel_with_opacity(base, layer.opacity);
                                continue;
                            }
                            let mut top = match chunk_raws[li] {
                                Some(raw) => Rgba([
                                    raw[px_off],
                                    raw[px_off + 1],
                                    raw[px_off + 2],
                                    raw[px_off + 3],
                                ]),
                                None => Rgba([0, 0, 0, 0]),
                            };
                            let mut preview_conceal_override: Option<u8> = None;

                            if li == active_layer_index && preview_layer.is_some() {
                                let pp = match preview_raw {
                                    Some(raw) => Rgba([
                                        raw[px_off],
                                        raw[px_off + 1],
                                        raw[px_off + 2],
                                        raw[px_off + 3],
                                    ]),
                                    None => Rgba([0, 0, 0, 0]),
                                };
                                if preview_targets_mask {
                                    if pp[3] > 0 {
                                        let old_conceal = if layer.mask_enabled {
                                            layer.mask.as_ref().map_or(0, |m| m.get_pixel(x, y)[3])
                                        } else {
                                            0
                                        };
                                        let old = old_conceal as u32;
                                        let s = pp[3] as u32;
                                        let new_conceal = if preview_mask_reveal {
                                            ((old * (255 - s)) / 255) as u8
                                        } else {
                                            (old + ((255 - old) * s) / 255) as u8
                                        };
                                        preview_conceal_override = Some(new_conceal);
                                    }
                                } else if preview_replaces {
                                    top = pp;
                                } else if pp[3] > 0 {
                                    if preview_is_eraser {
                                        let mask_strength = pp[3] as f32 / 255.0;
                                        let current_a = top[3] as f32 / 255.0;
                                        let new_a = (current_a * (1.0 - mask_strength)).max(0.0);
                                        top[3] = (new_a * 255.0) as u8;
                                    } else {
                                        top = Self::blend_pixel_static(
                                            top,
                                            pp,
                                            preview_blend_mode,
                                            1.0,
                                        );
                                    }
                                }
                            }

                            let conceal = if let Some(c) = preview_conceal_override {
                                c
                            } else if layer.mask_enabled {
                                layer.mask.as_ref().map_or(0, |m| m.get_pixel(x, y)[3])
                            } else {
                                0
                            };
                            if conceal > 0 {
                                top[3] = ((top[3] as u32 * (255 - conceal as u32)) / 255) as u8;
                            }

                            base = Self::blend_pixel_static(
                                base,
                                top,
                                layer.blend_mode,
                                layer.opacity,
                            );
                        }

                        let a = base[3];
                        if a == 255 || a == 0 {
                            row[out_idx] =
                                Color32::from_rgba_premultiplied(base[0], base[1], base[2], a);
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
                source_size: egui::Vec2::new(width as f32, height as f32),
                pixels,
            },
            [min_x as usize, min_y as usize],
        )
    }

    /// Composite only the visible layers ABOVE `active_layer_index` against a
    /// transparent background.  Returns a premultiplied RGBA `ColorImage`
    /// covering the full canvas (or `None` if there are no visible layers above).
    /// Used to overlay layers-above on top of a paste preview.
    pub fn composite_layers_above(&mut self) -> Option<ColorImage> {
        let above_start = self.active_layer_index + 1;
        let layer_visibility: Vec<bool> = (0..self.layers.len())
            .map(|idx| self.layer_effectively_visible(idx))
            .collect();
        let has_any = layer_visibility.iter().skip(above_start).any(|v| *v);
        if !has_any {
            return None;
        }

        let w = self.width as usize;
        let h = self.height as usize;
        let needed = w * h;
        // A12: reuse persistent buffer instead of allocating per frame
        self.composite_above_buffer
            .resize(needed, Color32::TRANSPARENT);
        self.composite_above_buffer.fill(Color32::TRANSPARENT);

        self.composite_above_buffer
            .par_chunks_mut(w)
            .enumerate()
            .for_each(|(y, row)| {
                for (x, pixel) in row.iter_mut().enumerate() {
                    let mut base = Rgba([0u8, 0, 0, 0]);
                    for (idx, layer) in self.layers.iter().enumerate().skip(above_start) {
                        if !layer_visibility.get(idx).copied().unwrap_or(false) {
                            continue;
                        }
                        if let LayerContent::Adjustment(adj) = &layer.content {
                            base = adj.apply_to_pixel_with_opacity(base, layer.opacity);
                            continue;
                        }
                        let mut top = *layer.pixels.get_pixel(x as u32, y as u32);
                        if layer.mask_enabled
                            && let Some(mask) = &layer.mask
                        {
                            let conceal = mask.get_pixel(x as u32, y as u32)[3];
                            if conceal > 0 {
                                top[3] = ((top[3] as u32 * (255 - conceal as u32)) / 255) as u8;
                            }
                        }
                        base = Self::blend_pixel_static(base, top, layer.blend_mode, layer.opacity);
                    }
                    let a = base[3];
                    if a == 0 {
                        *pixel = Color32::TRANSPARENT;
                    } else if a == 255 {
                        *pixel = Color32::from_rgba_premultiplied(base[0], base[1], base[2], 255);
                    } else {
                        let pm_r = ((base[0] as u16 * a as u16 + 127) / 255) as u8;
                        let pm_g = ((base[1] as u16 * a as u16 + 127) / 255) as u8;
                        let pm_b = ((base[2] as u16 * a as u16 + 127) / 255) as u8;
                        *pixel = Color32::from_rgba_premultiplied(pm_r, pm_g, pm_b, a);
                    }
                }
            });

        let pixels = self.composite_above_buffer.clone();
        Some(ColorImage {
            size: [w, h],
            source_size: egui::Vec2::new(w as f32, h as f32),
            pixels,
        })
    }

    /// Composite only the visible layers BELOW `active_layer_index` against a
    /// transparent background. Returns a premultiplied full-canvas image or
    /// `None` when there are no visible layers below.
    pub fn composite_layers_below_active(&mut self) -> Option<ColorImage> {
        let layer_visibility: Vec<bool> = (0..self.layers.len())
            .map(|idx| self.layer_effectively_visible(idx))
            .collect();
        let has_any = layer_visibility
            .iter()
            .take(self.active_layer_index)
            .any(|v| *v);
        if !has_any {
            return None;
        }

        let w = self.width as usize;
        let h = self.height as usize;
        let needed = w * h;
        self.composite_above_buffer
            .resize(needed, Color32::TRANSPARENT);
        self.composite_above_buffer.fill(Color32::TRANSPARENT);

        self.composite_above_buffer
            .par_chunks_mut(w)
            .enumerate()
            .for_each(|(y, row)| {
                for (x, pixel) in row.iter_mut().enumerate() {
                    let mut base = Rgba([0u8, 0, 0, 0]);
                    for (idx, layer) in self.layers.iter().enumerate().take(self.active_layer_index)
                    {
                        if !layer_visibility.get(idx).copied().unwrap_or(false) {
                            continue;
                        }
                        if let LayerContent::Adjustment(adj) = &layer.content {
                            base = adj.apply_to_pixel_with_opacity(base, layer.opacity);
                            continue;
                        }
                        let mut top = *layer.pixels.get_pixel(x as u32, y as u32);
                        if layer.mask_enabled
                            && let Some(mask) = &layer.mask
                        {
                            let conceal = mask.get_pixel(x as u32, y as u32)[3];
                            if conceal > 0 {
                                top[3] = ((top[3] as u32 * (255 - conceal as u32)) / 255) as u8;
                            }
                        }
                        base = Self::blend_pixel_static(base, top, layer.blend_mode, layer.opacity);
                    }
                    let a = base[3];
                    if a == 0 {
                        *pixel = Color32::TRANSPARENT;
                    } else if a == 255 {
                        *pixel = Color32::from_rgba_premultiplied(base[0], base[1], base[2], 255);
                    } else {
                        let pm_r = ((base[0] as u16 * a as u16 + 127) / 255) as u8;
                        let pm_g = ((base[1] as u16 * a as u16 + 127) / 255) as u8;
                        let pm_b = ((base[2] as u16 * a as u16 + 127) / 255) as u8;
                        *pixel = Color32::from_rgba_premultiplied(pm_r, pm_g, pm_b, a);
                    }
                }
            });

        let pixels = self.composite_above_buffer.clone();
        Some(ColorImage {
            size: [w, h],
            source_size: egui::Vec2::new(w as f32, h as f32),
            pixels,
        })
    }

    fn blend_pixel(
        &self,
        base: Rgba<u8>,
        top: Rgba<u8>,
        mode: BlendMode,
        opacity: f32,
    ) -> Rgba<u8> {
        Self::blend_pixel_static(base, top, mode, opacity)
    }

    /// Static version of blend_pixel for use when self is borrowed mutably elsewhere
    pub fn blend_pixel_static(
        base: Rgba<u8>,
        top: Rgba<u8>,
        mode: BlendMode,
        opacity: f32,
    ) -> Rgba<u8> {
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
                let xor_r =
                    (base_r * base_a * (1.0 - top_a) + top_r * top_a * (1.0 - base_a)) / xor_a;
                let xor_g =
                    (base_g * base_a * (1.0 - top_a) + top_g * top_a * (1.0 - base_a)) / xor_a;
                let xor_b =
                    (base_b * base_a * (1.0 - top_a) + top_b * top_a * (1.0 - base_a)) / xor_a;
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
            BlendMode::Difference => (
                (base_r - top_r).abs(),
                (base_g - top_g).abs(),
                (base_b - top_b).abs(),
            ),
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
        if top <= 0.0 {
            1.0
        } else {
            (base / top).min(1.0)
        }
    }

    fn vivid_light_channel(base: f32, top: f32) -> f32 {
        if top <= 0.5 {
            // Color Burn with 2*top
            let t2 = 2.0 * top;
            if t2 <= 0.0 {
                0.0
            } else {
                (1.0 - (1.0 - base) / t2).max(0.0)
            }
        } else {
            // Color Dodge with 2*(top-0.5)
            let t2 = 2.0 * (top - 0.5);
            if t2 >= 1.0 {
                1.0
            } else {
                (base / (1.0 - t2)).min(1.0)
            }
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
                let r = new_rect;
                let x0 = r.min.x.floor().max(0.0) as u32;
                let y0 = r.min.y.floor().max(0.0) as u32;
                let x1 = r.max.x.ceil().max(0.0) as u32;
                let y1 = r.max.y.ceil().max(0.0) as u32;
                layer.sync_deep_pixels_from_preview_region(x0, y0, x1, y1);
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

    pub fn request_commit_composite_flush(&mut self, rect: egui::Rect) {
        self.commit_composite_flush_rect = Some(match self.commit_composite_flush_rect {
            Some(existing) => existing.union(rect),
            None => rect,
        });
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
        self.selection_all = false;
        self.invalidate_selection_overlay();
        self.selection_overlay_texture = None;
        self.selection_overlay_bounds = None;
        self.selection_transform_preview_bounds = None;
    }

    /// Invalidate the cached selection overlay texture so it gets rebuilt
    /// on the next frame.  Call this whenever `selection_mask` changes.
    pub fn invalidate_selection_overlay(&mut self) {
        if self.selection_mask.is_some() {
            self.selection_all = false;
        }
        self.selection_overlay_generation = self.selection_overlay_generation.wrapping_add(1);
    }

    /// Returns `true` when there is an active selection mask.
    pub fn has_selection(&self) -> bool {
        self.selection_all || self.selection_mask.is_some()
    }

    /// Materialize the semantic full selection only for operations that need pixels.
    pub fn materialize_full_selection(&mut self) {
        if self.selection_all {
            self.selection_mask = Some(GrayImage::from_pixel(self.width, self.height, Luma([255])));
            self.selection_all = false;
        }
    }

    /// Return tight bounds of the current selection mask as (min_x, min_y, max_x, max_y).
    pub fn selection_mask_bounds(&self) -> Option<(u32, u32, u32, u32)> {
        if self.selection_all {
            return (self.width > 0 && self.height > 0).then_some((
                0,
                0,
                self.width - 1,
                self.height - 1,
            ));
        }
        let mask = self.selection_mask.as_ref()?;
        let w = mask.width();
        let h = mask.height();
        if w == 0 || h == 0 {
            return None;
        }

        let mut min_x = w;
        let mut min_y = h;
        let mut max_x = 0u32;
        let mut max_y = 0u32;
        let mut found = false;

        for y in 0..h {
            for x in 0..w {
                if mask.get_pixel(x, y).0[0] == 0 {
                    continue;
                }
                found = true;
                min_x = min_x.min(x);
                min_y = min_y.min(y);
                max_x = max_x.max(x);
                max_y = max_y.max(y);
            }
        }

        if found {
            Some((min_x, min_y, max_x, max_y))
        } else {
            None
        }
    }

    /// Translate the selection mask by (dx, dy) pixels.
    /// The mask is shifted; pixels that move off-canvas are clipped,
    /// and newly-exposed areas are unselected (0).
    pub fn translate_selection(&mut self, dx: i32, dy: i32) {
        if self.selection_all {
            self.selection_mask = Some(GrayImage::from_pixel(self.width, self.height, Luma([255])));
            self.selection_all = false;
        }
        let mask = match self.selection_mask.as_ref() {
            Some(m) => m,
            None => return,
        };
        if dx == 0 && dy == 0 {
            return;
        }

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

        if self.selection_all {
            match mode {
                SelectionMode::Replace => {
                    self.selection_mask = None;
                    self.selection_all = false;
                }
                SelectionMode::Add => return,
                SelectionMode::Subtract | SelectionMode::Intersect => {
                    self.selection_mask = Some(GrayImage::from_pixel(w, h, Luma([255])));
                    self.selection_all = false;
                }
            }
        }

        // Ensure a mask exists (for Add/Subtract we start from the current one).
        let mask = self
            .selection_mask
            .get_or_insert_with(|| GrayImage::new(w, h));

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
                for p in mask.pixels_mut() {
                    *p = Luma([0]);
                }
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
        if self.selection_all {
            if let Some(layer) = self.layers.get_mut(self.active_layer_index) {
                layer.pixels.clear();
                layer.sync_all_deep_pixels_from_preview();
                self.mark_dirty(None);
            }
            return;
        }
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
            layer.sync_all_deep_pixels_from_preview();
            self.mark_dirty(None);
        }
    }

    /// Fill the selected area on the active layer with a solid colour.
    pub fn fill_selected_pixels(&mut self, color: Rgba<u8>) {
        if self.selection_all {
            if let Some(layer) = self.layers.get_mut(self.active_layer_index) {
                layer.pixels.fill(color);
                layer.sync_all_deep_pixels_from_preview();
                self.mark_dirty(None);
            }
            return;
        }
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
            layer.sync_all_deep_pixels_from_preview();
            self.mark_dirty(None);
        }
    }
}
