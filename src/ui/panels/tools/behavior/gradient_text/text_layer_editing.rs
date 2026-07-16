impl ToolsPanel {
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

    /// Sync tool-state styling (color, font, etc.) to the active text block's
    /// single run so that `force_rasterize_text_layer` renders with the
    /// user-facing properties, not the stored defaults. Only applies to
    /// single-run blocks; multi-run blocks maintain per-run formatting.
    fn sync_text_layer_run_style(&self, canvas_state: &mut CanvasState) {
        let bid = match self.text_state.active_block_id {
            Some(id) => id,
            None => return,
        };
        let aidx = canvas_state.active_layer_index;
        if let Some(layer) = canvas_state.layers.get_mut(aidx)
            && let crate::canvas::LayerContent::Text(ref mut td) = layer.content
            && let Some(block) = td.blocks.iter_mut().find(|b| b.id == bid)
        {
            let mut changed = false;
            // Sync paragraph-level properties (line_spacing)
            if block.paragraph.line_spacing.to_bits() != self.text_state.line_spacing.to_bits() {
                block.paragraph.line_spacing = self.text_state.line_spacing;
                changed = true;
            }
            // Sync run-level properties (single-run blocks only)
            if block.runs.len() <= 1
                && let Some(run) = block.runs.first_mut()
            {
                if run.style.color != self.text_state.last_color {
                    run.style.color = self.text_state.last_color;
                    changed = true;
                }
                if run.style.font_family != self.text_state.font_family {
                    run.style.font_family = self.text_state.font_family.clone();
                    changed = true;
                }
                // Use effective bold weight: if bold toggle is ON, use 700 (or boosted)
                let effective_weight = if self.text_state.bold {
                    if self.text_state.font_weight < 600 {
                        700u16
                    } else {
                        (self.text_state.font_weight + 200).min(900)
                    }
                } else {
                    self.text_state.font_weight
                };
                if run.style.font_weight != effective_weight {
                    run.style.font_weight = effective_weight;
                    changed = true;
                }
                if run.style.font_size != self.text_state.font_size {
                    run.style.font_size = self.text_state.font_size;
                    changed = true;
                }
                if run.style.italic != self.text_state.italic {
                    run.style.italic = self.text_state.italic;
                    changed = true;
                }
                if run.style.underline != self.text_state.underline {
                    run.style.underline = self.text_state.underline;
                    changed = true;
                }
                if run.style.strikethrough != self.text_state.strikethrough {
                    run.style.strikethrough = self.text_state.strikethrough;
                    changed = true;
                }
                if run.style.letter_spacing.to_bits() != self.text_state.letter_spacing.to_bits() {
                    run.style.letter_spacing = self.text_state.letter_spacing;
                    changed = true;
                }
                if run.style.width_scale.to_bits() != self.text_state.width_scale.to_bits() {
                    run.style.width_scale = self.text_state.width_scale;
                    changed = true;
                }
                if run.style.height_scale.to_bits() != self.text_state.height_scale.to_bits() {
                    run.style.height_scale = self.text_state.height_scale;
                    changed = true;
                }
            }
            if changed {
                td.mark_dirty();
            }
        }
    }

    fn render_text_preview(&mut self, canvas_state: &mut CanvasState, primary_color_f32: [f32; 4]) {
        let origin = match self.text_state.origin {
            Some(o) => o,
            None => return,
        };
        if self.text_state.text.is_empty() {
            canvas_state.clear_preview_state();
            // For text layers, force-rasterize so the layer reflects the empty state
            // (e.g. after deleting all text the old pixels are cleared).
            if self.text_state.editing_text_layer {
                self.sync_text_layer_run_style(canvas_state);
                let idx = canvas_state.active_layer_index;
                canvas_state.force_rasterize_text_layer(idx);
                canvas_state.mark_dirty(None);
            }
            self.text_state.preview_dirty = false;
            return;
        }

        self.ensure_text_font_loaded();
        let font = match &self.text_state.loaded_font {
            Some(f) => f.clone(),
            None => return,
        };

        // For text layer editing: skip full pixel rasterization ÔÇö only compute
        // lightweight layout metrics for cursor/overlay, then force-rasterize
        // the layer via TextLayerData pipeline. This avoids a double full
        // rasterization per keystroke (was: rasterize_text + force_rasterize).
        if self.text_state.editing_text_layer {
            let active_max_width = self.text_state.active_block_max_width;
            let metrics = crate::ops::text::compute_text_layout_metrics(
                &font,
                &self.text_state.text,
                self.text_state.font_size,
                active_max_width,
                self.text_state.letter_spacing,
                self.text_state.line_spacing,
                self.text_state.width_scale,
                self.text_state.height_scale,
            );
            let preview_width = active_max_width.unwrap_or_else(|| {
                metrics
                    .line_advances
                    .iter()
                    .map(|line| line.iter().sum::<f32>())
                    .fold(self.text_state.font_size * 2.0, f32::max)
            });
            let visual_line_count = metrics
                .line_advances
                .iter()
                .map(Vec::len)
                .sum::<usize>()
                .max(1);
            let preview_height = metrics.line_height * visual_line_count as f32;
            self.text_state.cached_line_advances = metrics.line_advances;
            self.text_state.cached_line_height = metrics.line_height;
            canvas_state.clear_preview_state();
            self.sync_text_layer_run_style(canvas_state);
            let idx = canvas_state.active_layer_index;
            canvas_state.force_rasterize_text_layer(idx);
            // Text edits affect a block-sized region, not the entire canvas.
            // Include generous effect/rotation padding and the previous dirty
            // area so removed glyph pixels are also recomposited.
            let pad = self.text_state.font_size * 3.0 + 8.0;
            let min_x = match self.text_state.alignment {
                crate::ops::text::TextAlignment::Left => origin[0],
                crate::ops::text::TextAlignment::Center => origin[0] - preview_width * 0.5,
                crate::ops::text::TextAlignment::Right => origin[0] - preview_width,
            };
            let current_bounds = egui::Rect::from_min_size(
                egui::pos2(min_x - pad, origin[1] - pad),
                egui::vec2(preview_width + pad * 2.0, preview_height + pad * 2.0),
            );
            let dirty = canvas_state
                .preview_stroke_bounds
                .map_or(current_bounds, |old| old.union(current_bounds));
            canvas_state.preview_stroke_bounds = Some(current_bounds);
            canvas_state.mark_dirty(Some(dirty));
            self.text_state.preview_dirty = false;
            return;
        }

        let color = [
            (primary_color_f32[0] * 255.0) as u8,
            (primary_color_f32[1] * 255.0) as u8,
            (primary_color_f32[2] * 255.0) as u8,
            (primary_color_f32[3] * 255.0) as u8,
        ];

        // Pass max_width for word wrapping when editing a text layer block
        let active_max_width = if self.text_state.editing_text_layer {
            self.text_state.active_block_max_width
        } else {
            None
        };
        let result = crate::ops::text::rasterize_text(
            &font,
            crate::ops::text::font_cache_key(
                &self.text_state.font_family,
                if self.text_state.bold {
                    if self.text_state.font_weight < 600 {
                        700u16
                    } else {
                        (self.text_state.font_weight + 200).min(900)
                    }
                } else {
                    self.text_state.font_weight
                },
                self.text_state.italic,
            ),
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
            active_max_width,
            self.text_state.letter_spacing,
            self.text_state.line_spacing,
            self.text_state.width_scale,
            self.text_state.height_scale,
        );

        // Cache cursor metrics from rasterization (Opt 7)
        self.text_state.cached_line_advances = result.line_advances;
        self.text_state.cached_line_height = result.line_height;

        if result.buf_w == 0 || result.buf_h == 0 {
            canvas_state.clear_preview_state();
            self.text_state.preview_dirty = false;
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

        // Raster-stamp text editing: build preview overlay for non-text-layer editing
        // (text layer editing has already returned above via the lightweight metrics path)
        let mut preview = TiledImage::new(canvas_state.width, canvas_state.height);
        preview.blit_rgba_at(off_x, off_y, buf_w, buf_h, &result.buf);
        Self::adjust_preview_region_for_selection(
            &mut preview,
            off_x,
            off_y,
            buf_w,
            buf_h,
            canvas_state.selection_mask.as_ref(),
            0.0,
        );

        canvas_state.preview_layer = Some(preview);
        canvas_state.preview_blend_mode = self.properties.blending_mode;
        // Opt 2: Only force full composite when blend mode isn't Normal
        canvas_state.preview_force_composite = self.properties.blending_mode != BlendMode::Normal;
        canvas_state.preview_is_eraser = false;
        canvas_state.preview_downscale = 1;
        canvas_state.preview_flat_ready = false;
        // Opt 2: Limit composite/extraction to the visible portion of the text region.
        let visible_bounds = Self::clip_preview_bounds(canvas_state, off_x, off_y, buf_w, buf_h);
        canvas_state.preview_stroke_bounds = visible_bounds;
        // Opt 3: Use dirty_rect to signal texture update needed instead of
        // full cache invalidation. This lets the display path do a set()
        // instead of creating a brand new texture handle.
        if canvas_state.preview_texture_cache.is_some() {
            canvas_state.preview_dirty_rect = visible_bounds;
        } else {
            // First time: force texture creation
            canvas_state.preview_texture_cache = None;
        }
        canvas_state.mark_dirty(None);
        self.text_state.preview_dirty = false;
    }

    /// Load a text layer's data into the text tool state for editing.
    fn load_text_layer_for_editing(&mut self, canvas_state: &mut CanvasState) {
        self.load_text_layer_block(canvas_state, None, None);
    }

    /// Restore the raster text tool style that was saved before entering text-layer editing.
    /// Call this when starting a new raster text session to prevent text-layer properties
    /// from bleeding into the raster text tool.
    fn restore_raster_style(&mut self) {
        let s = self.text_state.saved_raster_style.clone();
        self.text_state.font_family = s.font_family;
        self.text_state.font_size = s.font_size;
        self.text_state.font_weight = s.font_weight;
        self.text_state.bold = s.bold;
        self.text_state.italic = s.italic;
        self.text_state.underline = s.underline;
        self.text_state.strikethrough = s.strikethrough;
        self.text_state.letter_spacing = s.letter_spacing;
        self.text_state.width_scale = s.width_scale;
        self.text_state.height_scale = s.height_scale;
        self.text_state.line_spacing = s.line_spacing;
        self.text_state.alignment = s.alignment;
        self.text_state.last_color = s.last_color;
        // Force font reload for the restored family
        self.text_state.loaded_font_key.clear();
        self.text_state.loaded_font = None;
    }

    /// Load a specific block from a text layer for editing.
    /// If `block_id` is None, loads the first block.
    /// If `click_pos` is provided, hit-tests blocks or creates a new one.
    fn load_text_layer_block(
        &mut self,
        canvas_state: &mut CanvasState,
        block_id: Option<u64>,
        click_pos: Option<[f32; 2]>,
    ) {
        let layer_idx = canvas_state.active_layer_index;
        let layer = match canvas_state.layers.get(layer_idx) {
            Some(l) => l,
            None => return,
        };
        let text_data = match &layer.content {
            crate::canvas::LayerContent::Text(td) => td,
            _ => return,
        };

        // Capture "before" snapshot for TextLayerEditCommand undo (only on first load per session).
        // Subsequent block switches within the same text layer session keep the original snapshot.
        if self.text_state.text_layer_before.is_none() {
            self.text_state.text_layer_before = Some((layer_idx, text_data.clone()));
        }

        // Determine which block to edit
        let target_block_id = if let Some(bid) = block_id {
            Some(bid)
        } else if let Some(pos) = click_pos {
            // Hit-test: find block at click position
            crate::ops::text_layer::hit_test_blocks(text_data, pos[0], pos[1])
                .map(|idx| text_data.blocks[idx].id)
        } else {
            // Default: first block
            text_data.blocks.first().map(|b| b.id)
        };

        // Get the block data (or create new block)
        let (
            block_text,
            style,
            position,
            bid,
            alignment,
            block_max_width,
            block_max_height,
            block_line_spacing,
        );
        if let Some(bid_val) = target_block_id {
            if let Some(block) = text_data.blocks.iter().find(|b| b.id == bid_val) {
                block_text = block.flat_text();
                style = block
                    .runs
                    .first()
                    .map(|r| r.style.clone())
                    .unwrap_or_default();
                position = block.position;
                bid = bid_val;
                alignment = block.paragraph.alignment;
                block_max_width = block.max_width;
                block_max_height = block.max_height;
                block_line_spacing = block.paragraph.line_spacing;
            } else {
                return;
            }
        } else if let Some(pos) = click_pos {
            // Create a new block at click position ÔÇö need mutable access
            let layer_mut = match canvas_state.layers.get_mut(canvas_state.active_layer_index) {
                Some(l) => l,
                None => return,
            };
            let td = match &mut layer_mut.content {
                crate::canvas::LayerContent::Text(td) => td,
                _ => return,
            };
            let new_id = td.add_block(pos[0], pos[1]);
            block_text = String::new();
            style = td.primary_style().clone();
            position = pos;
            bid = new_id;
            alignment = crate::ops::text_layer::TextAlignment::Left;
            block_max_width = None;
            block_max_height = None;
            block_line_spacing = 1.0;
        } else {
            return;
        };

        // Save raster text style before overwriting with text-layer properties, so we can
        // restore it when the user switches back to a raster layer.
        if !self.text_state.editing_text_layer {
            self.text_state.saved_raster_style = SavedRasterStyle {
                font_family: self.text_state.font_family.clone(),
                font_size: self.text_state.font_size,
                font_weight: self.text_state.font_weight,
                bold: self.text_state.bold,
                italic: self.text_state.italic,
                underline: self.text_state.underline,
                strikethrough: self.text_state.strikethrough,
                letter_spacing: self.text_state.letter_spacing,
                width_scale: self.text_state.width_scale,
                height_scale: self.text_state.height_scale,
                line_spacing: self.text_state.line_spacing,
                alignment: self.text_state.alignment,
                last_color: self.text_state.last_color,
            };
        }

        self.text_state.origin = Some(position);
        self.text_state.text = block_text.clone();
        self.text_state.cursor_pos = block_text.len();
        self.text_state.is_editing = true;
        self.text_state.editing_text_layer = true;
        self.text_state.editing_layer_index = Some(canvas_state.active_layer_index);
        self.text_state.active_block_id = Some(bid);
        self.text_state.selection = crate::ops::text_layer::TextSelection::default();
        self.text_state.font_family = style.font_family.clone();
        self.text_state.font_size = style.font_size;
        self.text_state.font_weight = style.font_weight;
        self.text_state.bold = style.font_weight >= 700;
        self.text_state.italic = style.italic;
        self.text_state.underline = style.underline;
        self.text_state.strikethrough = style.strikethrough;
        self.text_state.last_color = style.color;
        self.text_state.letter_spacing = style.letter_spacing;
        self.text_state.width_scale = style.width_scale;
        self.text_state.height_scale = style.height_scale;
        self.text_state.line_spacing = block_line_spacing;
        self.text_state.preview_dirty = true;
        self.text_state.active_block_max_width = block_max_width;
        self.text_state.active_block_max_height = block_max_height;
        self.text_state.text_box_drag = None;
        self.text_state.active_block_height = 0.0;

        // Convert alignment
        self.text_state.alignment = match alignment {
            crate::ops::text_layer::TextAlignment::Left => crate::ops::text::TextAlignment::Left,
            crate::ops::text_layer::TextAlignment::Center => {
                crate::ops::text::TextAlignment::Center
            }
            crate::ops::text_layer::TextAlignment::Right => crate::ops::text::TextAlignment::Right,
        };

        // Force font re-load for the new family
        self.text_state.loaded_font_key.clear();
        self.text_state.loaded_font = None;

        // Load layer effects into tool state for the effects panel
        if let Some(layer) = canvas_state.layers.get(canvas_state.active_layer_index)
            && let crate::canvas::LayerContent::Text(ref td) = layer.content
        {
            self.text_state.text_effects = td.effects.clone();
            // Load warp from the active block
            if let Some(bid) = self.text_state.active_block_id
                && let Some(block) = td.blocks.iter().find(|b| b.id == bid)
            {
                self.text_state.text_warp = block.warp.clone();
                self.text_state.glyph_overrides = block.glyph_overrides.clone();
            }
        }
        self.text_state.text_effects_dirty = false;
        self.text_state.text_warp_dirty = false;
        self.text_state.glyph_edit_mode = false;
        self.text_state.selected_glyphs.clear();
        self.text_state.cached_glyph_bounds.clear();
        self.text_state.glyph_bounds_dirty = true;
        self.text_state.glyph_drag = None;
        self.text_state.glyph_overrides_dirty = false;

        // Set the editing layer marker so ensure_text_layers_rasterized skips
        // this layer (we handle rasterization explicitly via force_rasterize).
        canvas_state.text_editing_layer = Some(canvas_state.active_layer_index);
        // Force-rasterize so the layer pixels are up-to-date (e.g. after commit
        // of a previous block, the committed text is visible).
        let idx = canvas_state.active_layer_index;
        canvas_state.force_rasterize_text_layer(idx);
        canvas_state.mark_dirty(None);

        self.stroke_tracker
            .start_preview_tool(canvas_state.active_layer_index, "Text");
    }

    fn commit_text(&mut self, canvas_state: &mut CanvasState) {
        if self.text_state.text.is_empty() || self.text_state.origin.is_none() {
            self.text_state.is_editing = false;
            self.text_state.editing_text_layer = false;
            self.text_state.editing_layer_index = None;
            self.text_state.active_block_id = None;
            self.text_state.selection = crate::ops::text_layer::TextSelection::default();
            self.text_state.origin = None;
            canvas_state.text_editing_layer = None;
            canvas_state.clear_preview_state();
            return;
        }

        // Text layer mode: write back to TextLayerData instead of blending into pixels
        if self.text_state.editing_text_layer {
            self.commit_text_layer(canvas_state);
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
        let selection_mask = canvas_state.selection_mask.clone();
        let target_layer_idx = self
            .text_state
            .editing_layer_index
            .unwrap_or(canvas_state.active_layer_index);
        if let Some(ref preview) = canvas_state.preview_layer
            && let Some(active_layer) = canvas_state.layers.get_mut(target_layer_idx)
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
                        if !Self::selection_allows(selection_mask.as_ref(), gx, gy) {
                            continue;
                        }
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
        self.text_state.editing_text_layer = false;
        self.text_state.editing_layer_index = None;
        self.text_state.active_block_id = None;
        self.text_state.selection = crate::ops::text_layer::TextSelection::default();
        self.text_state.origin = None;
        canvas_state.clear_preview_state();
        canvas_state.mark_dirty(None);

        if stroke_event.is_some() {
            self.pending_stroke_event = stroke_event;
        }
    }

    // ====================================================================
    // TEXT LAYER EDITING HELPERS (Batch 3+4: multi-run, multi-block)
    // ====================================================================

    /// Insert text at cursor position in the active text layer block.
    /// Handles selection deletion before insertion.
    fn text_layer_insert_text(&mut self, canvas_state: &mut CanvasState, text: &str) {
        let bid = match self.text_state.active_block_id {
            Some(id) => id,
            None => return,
        };

        // Delete selection first if any
        if self.text_state.selection.has_selection() {
            self.text_layer_delete_selection(canvas_state);
        }

        let cursor = self.text_state.cursor_pos;

        if let Some(layer) = canvas_state.layers.get_mut(canvas_state.active_layer_index)
            && let crate::canvas::LayerContent::Text(ref mut td) = layer.content
            && let Some(idx) = td.blocks.iter().position(|b| b.id == bid)
        {
            td.blocks[idx].insert_text_at(cursor, text);
            td.mark_dirty();
            self.text_state.text = td.blocks[idx].flat_text();
            self.text_state.cursor_pos = cursor + text.len();
        }

        self.text_state.selection = crate::ops::text_layer::TextSelection::default();
        self.sync_text_toolbar_to_selection(canvas_state);
        self.text_state.preview_dirty = true;
    }

    /// Backspace in text layer mode: delete selection or char before cursor.
    fn text_layer_backspace(&mut self, canvas_state: &mut CanvasState) {
        if self.text_state.selection.has_selection() {
            self.text_layer_delete_selection(canvas_state);
            return;
        }

        let bid = match self.text_state.active_block_id {
            Some(id) => id,
            None => return,
        };
        let cursor = self.text_state.cursor_pos;
        if cursor == 0 {
            return;
        }

        // Find the byte length of the char before cursor
        let prev_char_len = self.text_state.text[..cursor]
            .chars()
            .last()
            .map(|c| c.len_utf8())
            .unwrap_or(0);
        if prev_char_len == 0 {
            return;
        }

        let del_start = cursor - prev_char_len;
        let del_end = cursor;

        if let Some(layer) = canvas_state.layers.get_mut(canvas_state.active_layer_index)
            && let crate::canvas::LayerContent::Text(ref mut td) = layer.content
            && let Some(idx) = td.blocks.iter().position(|b| b.id == bid)
        {
            td.blocks[idx].delete_range(del_start, del_end);
            td.mark_dirty();
            self.text_state.text = td.blocks[idx].flat_text();
            self.text_state.cursor_pos = del_start;
        }

        self.text_state.selection = crate::ops::text_layer::TextSelection::default();
        self.text_state.preview_dirty = true;
    }

    /// Delete key in text layer mode: delete selection or char after cursor.
    fn text_layer_delete(&mut self, canvas_state: &mut CanvasState) {
        if self.text_state.selection.has_selection() {
            self.text_layer_delete_selection(canvas_state);
            return;
        }

        let bid = match self.text_state.active_block_id {
            Some(id) => id,
            None => return,
        };
        let cursor = self.text_state.cursor_pos;
        if cursor >= self.text_state.text.len() {
            return;
        }

        let next_char_len = self.text_state.text[cursor..]
            .chars()
            .next()
            .map(|c| c.len_utf8())
            .unwrap_or(0);
        if next_char_len == 0 {
            return;
        }

        let del_start = cursor;
        let del_end = cursor + next_char_len;

        if let Some(layer) = canvas_state.layers.get_mut(canvas_state.active_layer_index)
            && let crate::canvas::LayerContent::Text(ref mut td) = layer.content
            && let Some(idx) = td.blocks.iter().position(|b| b.id == bid)
        {
            td.blocks[idx].delete_range(del_start, del_end);
            td.mark_dirty();
            self.text_state.text = td.blocks[idx].flat_text();
        }

        self.text_state.selection = crate::ops::text_layer::TextSelection::default();
        self.text_state.preview_dirty = true;
    }

    /// Move the text cursor up or down by `direction` visual lines (-1 = up, +1 = down).
    /// Tries to preserve the x-position (character column) across lines.
    fn text_move_cursor_vertical(
        &mut self,
        direction: i32,
        shift_held: bool,
        canvas_state: &mut CanvasState,
    ) {
        let text = &self.text_state.text;
        if text.is_empty() {
            return;
        }
        let font_size = self.text_state.font_size;
        let ls = self.text_state.letter_spacing;

        // Determine current visual line and char offset
        let (cur_vis_line, cur_vis_char) = if let (Some(mw), Some(font)) = (
            self.text_state.active_block_max_width,
            &self.text_state.loaded_font,
        ) {
            Self::byte_pos_to_visual(
                text,
                self.text_state.cursor_pos,
                font,
                font_size,
                mw,
                ls,
                self.text_state.width_scale,
                self.text_state.height_scale,
            )
        } else {
            // No wrapping ÔÇö use logical lines
            let before = &text[..self.text_state.cursor_pos];
            let line_idx = before.matches('\n').count();
            let last_line = before.rsplit('\n').next().unwrap_or(before);
            (line_idx, last_line.chars().count())
        };

        // Compute total visual line count
        let total_visual_lines = if let (Some(mw), Some(font)) = (
            self.text_state.active_block_max_width,
            &self.text_state.loaded_font,
        ) {
            text.split('\n')
                .flat_map(|line| {
                    crate::ops::text::word_wrap_line(
                        line,
                        font,
                        font_size,
                        mw,
                        ls,
                        self.text_state.width_scale,
                        self.text_state.height_scale,
                    )
                })
                .count()
        } else {
            text.split('\n').count()
        };

        // Compute target visual line
        let target_line = if direction < 0 {
            if cur_vis_line == 0 {
                // Already on first line ÔÇö move cursor to start of text
                self.text_state.cursor_pos = 0;
                if self.text_state.editing_text_layer {
                    self.text_layer_update_selection(canvas_state, shift_held);
                }
                return;
            }
            cur_vis_line - 1
        } else {
            if cur_vis_line >= total_visual_lines.saturating_sub(1) {
                // Already on last line ÔÇö move cursor to end of text
                self.text_state.cursor_pos = text.len();
                if self.text_state.editing_text_layer {
                    self.text_layer_update_selection(canvas_state, shift_held);
                }
                return;
            }
            cur_vis_line + 1
        };

        // Get the x-position of the cursor on the current line (in pixels)
        // to find the closest char on the target line
        let cursor_x = self
            .text_state
            .cached_line_advances
            .get(cur_vis_line)
            .and_then(|adv| adv.get(cur_vis_char).copied())
            .unwrap_or(0.0);

        // Find the closest char position on the target line
        let target_char =
            if let Some(advances) = self.text_state.cached_line_advances.get(target_line) {
                let mut best = 0;
                let mut best_dist = f32::MAX;
                for (ci, &adv) in advances.iter().enumerate() {
                    let dist = (adv - cursor_x).abs();
                    if dist < best_dist {
                        best_dist = dist;
                        best = ci;
                    }
                }
                // Clamp to line char count (advances has count+1 entries)
                best.min(advances.len().saturating_sub(1))
            } else {
                cur_vis_char
            };

        // Convert (target_line, target_char) back to byte position
        let new_byte_pos = if let (Some(mw), Some(font)) = (
            self.text_state.active_block_max_width,
            &self.text_state.loaded_font,
        ) {
            Self::visual_to_byte_pos(
                text,
                target_line,
                target_char,
                font,
                font_size,
                mw,
                ls,
                self.text_state.width_scale,
                self.text_state.height_scale,
            )
        } else {
            // No wrapping ÔÇö use logical lines
            let lines: Vec<&str> = text.split('\n').collect();
            let clamped = target_line.min(lines.len().saturating_sub(1));
            let line_start: usize = lines.iter().take(clamped).map(|l| l.len() + 1).sum();
            let line_text = lines[clamped];
            let clamped_char = target_char.min(line_text.chars().count());
            let byte_offset: usize = line_text
                .chars()
                .take(clamped_char)
                .map(|c| c.len_utf8())
                .sum();
            line_start + byte_offset
        };

        self.text_state.cursor_pos = new_byte_pos;
        if self.text_state.editing_text_layer {
            self.text_layer_update_selection(canvas_state, shift_held);
        }
    }

    /// Delete the currently selected range in the active text layer block.
    fn text_layer_delete_selection(&mut self, canvas_state: &mut CanvasState) {
        let bid = match self.text_state.active_block_id {
            Some(id) => id,
            None => return,
        };

        if !self.text_state.selection.has_selection() {
            return;
        }

        // Get ordered byte offsets
        let (start, end) = if let Some(layer) =
            canvas_state.layers.get(canvas_state.active_layer_index)
            && let crate::canvas::LayerContent::Text(ref td) = layer.content
            && let Some(block) = td.blocks.iter().find(|b| b.id == bid)
        {
            self.text_state.selection.ordered_flat_offsets(block)
        } else {
            return;
        };

        if let Some(layer) = canvas_state.layers.get_mut(canvas_state.active_layer_index)
            && let crate::canvas::LayerContent::Text(ref mut td) = layer.content
            && let Some(idx) = td.blocks.iter().position(|b| b.id == bid)
        {
            td.blocks[idx].delete_range(start, end);
            td.mark_dirty();
            self.text_state.text = td.blocks[idx].flat_text();
            self.text_state.cursor_pos = start;
        }

        self.text_state.selection = crate::ops::text_layer::TextSelection::default();
        self.text_state.preview_dirty = true;
    }

    /// Update selection after cursor movement.
    /// If shift is held, extend selection; otherwise collapse.
    fn text_layer_update_selection(&mut self, canvas_state: &mut CanvasState, shift_held: bool) {
        let bid = match self.text_state.active_block_id {
            Some(id) => id,
            None => return,
        };

        if let Some(layer) = canvas_state.layers.get(canvas_state.active_layer_index)
            && let crate::canvas::LayerContent::Text(ref td) = layer.content
            && let Some(block) = td.blocks.iter().find(|b| b.id == bid)
        {
            let new_pos = block.flat_offset_to_run_pos(self.text_state.cursor_pos);
            self.text_state.selection.cursor = new_pos;
            if !shift_held {
                self.text_state.selection.anchor = new_pos;
            }
        }
        self.sync_text_toolbar_to_selection(canvas_state);
    }

    fn sync_text_toolbar_to_selection(&mut self, canvas_state: &CanvasState) {
        let bid = match self.text_state.active_block_id {
            Some(id) => id,
            None => return,
        };

        let Some(layer) = canvas_state.layers.get(canvas_state.active_layer_index) else {
            return;
        };
        let crate::canvas::LayerContent::Text(ref td) = layer.content else {
            return;
        };
        let Some(block) = td.blocks.iter().find(|b| b.id == bid) else {
            return;
        };

        let mut first_style = None;
        let mut all_bold = true;
        let mut all_italic = true;
        let mut all_underline = true;
        let mut all_strikethrough = true;

        if self.text_state.selection.has_selection() {
            let (start, end) = self.text_state.selection.ordered_flat_offsets(block);
            let mut offset = 0usize;
            for run in &block.runs {
                let run_end = offset + run.text.len();
                if run_end > start && offset < end && !run.text.is_empty() {
                    first_style.get_or_insert_with(|| run.style.clone());
                    all_bold &= run.style.font_weight >= 700;
                    all_italic &= run.style.italic;
                    all_underline &= run.style.underline;
                    all_strikethrough &= run.style.strikethrough;
                }
                offset = run_end;
            }
        } else {
            let cursor = self.text_state.cursor_pos.min(block.flat_text().len());
            let mut offset = 0usize;
            for (idx, run) in block.runs.iter().enumerate() {
                let run_end = offset + run.text.len();
                let at_run = cursor < run_end
                    || (cursor == run_end
                        && (idx + 1 == block.runs.len() || cursor > offset));
                if at_run {
                    first_style = Some(run.style.clone());
                    all_bold = run.style.font_weight >= 700;
                    all_italic = run.style.italic;
                    all_underline = run.style.underline;
                    all_strikethrough = run.style.strikethrough;
                    break;
                }
                offset = run_end;
            }
        }

        if let Some(style) = first_style {
            if !self.text_state.selection.has_selection() {
                self.text_state.font_family = style.font_family;
                self.text_state.font_weight = style.font_weight;
                self.text_state.font_size = style.font_size;
                self.text_state.letter_spacing = style.letter_spacing;
                self.text_state.width_scale = style.width_scale;
                self.text_state.height_scale = style.height_scale;
            }
            self.text_state.bold = all_bold;
            self.text_state.italic = all_italic;
            self.text_state.underline = all_underline;
            self.text_state.strikethrough = all_strikethrough;
        }
    }

    fn text_layer_byte_pos_from_point(
        &self,
        canvas_state: &CanvasState,
        point: (f32, f32),
    ) -> Option<usize> {
        let layer = canvas_state.layers.get(canvas_state.active_layer_index)?;
        let crate::canvas::LayerContent::Text(ref td) = layer.content else {
            return None;
        };
        let bid = self.text_state.active_block_id?;
        let block = td.blocks.iter().find(|b| b.id == bid)?;
        let bounds = crate::ops::text_layer::compute_glyph_bounds(block);
        if bounds.is_empty() {
            return Some(0);
        }

        let mut best_line_y = f32::MAX;
        let mut line: Vec<_> = Vec::new();
        for gb in &bounds {
            let contains_y = point.1 >= gb.y && point.1 <= gb.y + gb.h;
            let distance_y = if contains_y {
                0.0
            } else {
                (point.1 - (gb.y + gb.h * 0.5)).abs()
            };
            if distance_y + 0.001 < best_line_y {
                best_line_y = distance_y;
                line.clear();
                line.push(gb);
            } else if (distance_y - best_line_y).abs() < 0.001 {
                line.push(gb);
            }
        }

        line.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap_or(std::cmp::Ordering::Equal));
        if let Some(first) = line.first()
            && point.0 <= first.x + first.w * 0.5
        {
            return Some(first.flat_start);
        }
        for pair in line.windows(2) {
            let left = pair[0];
            let right = pair[1];
            let boundary = (left.x + left.w + right.x) * 0.5;
            if point.0 <= boundary {
                return Some(left.flat_end);
            }
        }
        line.last().map(|gb| gb.flat_end)
    }

    /// Apply text-box-wide style properties to every run in the active text block.
    fn text_layer_apply_block_style(
        &mut self,
        canvas_state: &mut CanvasState,
        apply: impl Fn(&mut crate::ops::text_layer::TextStyle),
    ) {
        let bid = match self.text_state.active_block_id {
            Some(id) => id,
            None => return,
        };

        if let Some(layer) = canvas_state.layers.get_mut(canvas_state.active_layer_index)
            && let crate::canvas::LayerContent::Text(ref mut td) = layer.content
            && let Some(idx) = td.blocks.iter().position(|b| b.id == bid)
        {
            for run in &mut td.blocks[idx].runs {
                apply(&mut run.style);
            }
            td.blocks[idx].merge_adjacent_runs();
            td.mark_dirty();
            self.text_state.text = td.blocks[idx].flat_text();
        }

        self.text_state.preview_dirty = true;
        self.sync_text_toolbar_to_selection(canvas_state);
    }

    /// Apply a style closure to the selection (or cursor run if no selection)
    /// in a text layer block.
    fn text_layer_toggle_style(
        &mut self,
        canvas_state: &mut CanvasState,
        apply: impl Fn(&mut crate::ops::text_layer::TextStyle),
    ) {
        let bid = match self.text_state.active_block_id {
            Some(id) => id,
            None => return,
        };

        // Determine byte range: selection if present, otherwise cursor run
        // Also capture flat byte offsets for anchor/cursor BEFORE apply_style_to_range
        // splits runs, so we can refresh the RunPosition values afterward.
        let (start, end, anchor_flat, cursor_flat) = if self.text_state.selection.has_selection() {
            if let Some(layer) =
                canvas_state.layers.get(canvas_state.active_layer_index)
                && let crate::canvas::LayerContent::Text(ref td) = layer.content
                && let Some(block) = td.blocks.iter().find(|b| b.id == bid)
            {
                let a = block.run_pos_to_flat_offset(self.text_state.selection.anchor);
                let c = block.run_pos_to_flat_offset(self.text_state.selection.cursor);
                if a <= c {
                    (a, c, a, c)
                } else {
                    (c, a, a, c)
                }
            } else {
                return;
            }
        } else {
            // No selection: apply to the run at cursor position
            let cursor = self.text_state.cursor_pos;
            if let Some(layer) =
                canvas_state.layers.get(canvas_state.active_layer_index)
                && let crate::canvas::LayerContent::Text(ref td) = layer.content
                && let Some(block) = td.blocks.iter().find(|b| b.id == bid)
            {
                let run_pos = block.flat_offset_to_run_pos(cursor);
                let run_start = block.run_pos_to_flat_offset(run_pos);
                let run_end = if run_pos.run_index + 1 < block.runs.len() {
                    block.run_pos_to_flat_offset(crate::ops::text_layer::RunPosition {
                        run_index: run_pos.run_index + 1,
                        byte_offset: 0,
                    })
                } else {
                    block.flat_text().len()
                };
                (run_start, run_end, cursor, cursor)
            } else {
                return;
            }
        };

        if let Some(layer) = canvas_state.layers.get_mut(canvas_state.active_layer_index)
            && let crate::canvas::LayerContent::Text(ref mut td) = layer.content
            && let Some(idx) = td.blocks.iter().position(|b| b.id == bid)
        {
            td.blocks[idx].apply_style_to_range(start, end, apply);
            td.mark_dirty();
            // Refresh flat text (run splitting might have changed byte offsets)
            self.text_state.text = td.blocks[idx].flat_text();
            // Refresh selection RunPosition values — after apply_style_to_range splits
            // runs, the old RunPosition{run_index, byte_offset} values are stale and
            // would produce wrong flat offsets on the next call. Convert the flat byte
            // offsets we captured before the split back to fresh RunPosition values.
            self.text_state.selection.anchor =
                td.blocks[idx].flat_offset_to_run_pos(anchor_flat);
            self.text_state.selection.cursor =
                td.blocks[idx].flat_offset_to_run_pos(cursor_flat);
        }
        self.sync_text_toolbar_to_selection(canvas_state);

        self.text_state.preview_dirty = true;
        // Immediately recompute cached_line_advances so overlay/cursor positioning
        // uses correct glyph widths (bold/italic changes advance metrics).
        if let Some(ref font) = self.text_state.loaded_font {
            let metrics = crate::ops::text::compute_text_layout_metrics(
                font,
                &self.text_state.text,
                self.text_state.font_size,
                self.text_state.active_block_max_width,
                self.text_state.letter_spacing,
                self.text_state.line_spacing,
                self.text_state.width_scale,
                self.text_state.height_scale,
            );
            self.text_state.cached_line_advances = metrics.line_advances;
            self.text_state.cached_line_height = metrics.line_height;
        }
    }

    /// Cycle to the next/previous text block (Tab / Shift+Tab).
    fn text_layer_cycle_block(&mut self, canvas_state: &mut CanvasState, reverse: bool) {
        let bid = match self.text_state.active_block_id {
            Some(id) => id,
            None => return,
        };

        // First, sync the current block (commit position/text)
        if let Some(layer) = canvas_state.layers.get(canvas_state.active_layer_index)
            && let crate::canvas::LayerContent::Text(ref td) = layer.content
        {
            let block_count = td.blocks.len();
            if block_count <= 1 {
                return;
            }

            let current_idx = td.blocks.iter().position(|b| b.id == bid).unwrap_or(0);
            let next_idx = if reverse {
                if current_idx == 0 {
                    block_count - 1
                } else {
                    current_idx - 1
                }
            } else {
                (current_idx + 1) % block_count
            };
            let next_id = td.blocks[next_idx].id;

            // Load the next block
            self.load_text_layer_block(canvas_state, Some(next_id), None);
        }
    }

    /// Commit text edits back to the active text layer's `TextLayerData`.
    fn commit_text_layer(&mut self, canvas_state: &mut CanvasState) {
        use crate::ops::text_layer::{TextAlignment as TLA, TextStyle as TLS};

        let origin = self.text_state.origin.unwrap_or([100.0, 100.0]);
        let text = self.text_state.text.clone();
        let color = self.text_state.last_color;

        let alignment = match self.text_state.alignment {
            crate::ops::text::TextAlignment::Left => TLA::Left,
            crate::ops::text::TextAlignment::Center => TLA::Center,
            crate::ops::text::TextAlignment::Right => TLA::Right,
        };

        // Use effective bold weight: if bold toggle is ON, use 700 (or boosted)
        let effective_weight = if self.text_state.bold {
            if self.text_state.font_weight < 600 {
                700u16
            } else {
                (self.text_state.font_weight + 200).min(900)
            }
        } else {
            self.text_state.font_weight
        };
        let new_style = TLS {
            font_family: self.text_state.font_family.clone(),
            font_weight: effective_weight,
            font_size: self.text_state.font_size,
            italic: self.text_state.italic,
            underline: self.text_state.underline,
            strikethrough: self.text_state.strikethrough,
            color,
            letter_spacing: self.text_state.letter_spacing,
            baseline_offset: 0.0,
            width_scale: self.text_state.width_scale,
            height_scale: self.text_state.height_scale,
        };

        // Update the TextLayerData on the active layer
        if let Some(layer) = canvas_state.layers.get_mut(canvas_state.active_layer_index)
            && let crate::canvas::LayerContent::Text(ref mut td) = layer.content
        {
            if let Some(block_id) = self.text_state.active_block_id {
                // Multi-block mode: update the specific active block
                if let Some(block) = td.blocks.iter_mut().find(|b| b.id == block_id) {
                    block.position = origin;
                    block.paragraph.alignment = alignment;
                    block.paragraph.line_spacing = self.text_state.line_spacing;
                    block.max_width = self.text_state.active_block_max_width;
                    block.max_height = self.text_state.active_block_max_height;
                    // If the block has only one run (no per-run formatting),
                    // update its text and style directly
                    if block.runs.len() <= 1
                        && let Some(run) = block.runs.first_mut()
                    {
                        run.text = text;
                        run.style = new_style;
                    }
                    // If multi-run: text is already synced during editing,
                    // just update position and alignment
                }
            } else {
                // Legacy path: update the first block
                if let Some(block) = td.blocks.first_mut() {
                    block.position = origin;
                    block.paragraph.alignment = alignment;
                    if let Some(run) = block.runs.first_mut() {
                        run.text = text;
                        run.style = new_style;
                    }
                }
            }
            // Empty blocks are kept as placeholders ÔÇö use delete (├ù) button to remove
            // td.remove_empty_blocks();
            // Note: effects and warp are NOT overwritten here.
            // The layer settings dialog writes directly to TextLayerData,
            // so we must not clobber those values with stale tool-state copies.
            // Write glyph overrides back to the active block
            if let Some(block_id) = self.text_state.active_block_id
                && let Some(block) = td.blocks.iter_mut().find(|b| b.id == block_id)
            {
                block.glyph_overrides = self.text_state.glyph_overrides.clone();
                block.cleanup_glyph_overrides();
            }
            td.mark_dirty();
        }

        // --- Text Layer Undo via TextLayerEditCommand ---
        // Use the captured "before" snapshot and current state for efficient vector-data undo.
        // Cancel the stroke tracker (it was only used for preview positioning).
        self.stroke_tracker.cancel();
        if let Some((layer_idx, before_td)) = self.text_state.text_layer_before.take() {
            // Capture "after" state
            if let Some(layer) = canvas_state.layers.get(layer_idx)
                && let crate::canvas::LayerContent::Text(ref after_td) = layer.content
            {
                let mut cmd = crate::components::history::TextLayerEditCommand::new_from(
                    "Text Edit".to_string(),
                    layer_idx,
                    before_td,
                );
                cmd.set_after_from(after_td.clone());
                self.pending_history_commands.push(Box::new(cmd));
            }
        }

        // Clean up editing state
        self.text_state.text.clear();
        self.text_state.cursor_pos = 0;
        self.text_state.is_editing = false;
        self.text_state.editing_text_layer = false;
        self.text_state.editing_layer_index = None;
        self.text_state.active_block_id = None;
        self.text_state.selection = crate::ops::text_layer::TextSelection::default();
        self.text_state.text_effects = crate::ops::text_layer::TextEffects::default();
        self.text_state.text_effects_dirty = false;
        self.text_state.text_warp = crate::ops::text_layer::TextWarp::None;
        self.text_state.text_warp_dirty = false;
        self.text_state.text_box_drag = None;
        self.text_state.active_block_max_width = None;
        self.text_state.active_block_max_height = None;
        self.text_state.active_block_height = 0.0;
        self.text_state.glyph_edit_mode = false;
        self.text_state.selected_glyphs.clear();
        self.text_state.cached_glyph_bounds.clear();
        self.text_state.glyph_bounds_dirty = true;
        self.text_state.glyph_drag = None;
        self.text_state.glyph_overrides.clear();
        self.text_state.glyph_overrides_dirty = false;
        self.text_state.text_layer_drag_cached = false;
        self.text_state.text_layer_before = None;
        self.text_state.origin = None;
        canvas_state.text_editing_layer = None;
        canvas_state.clear_preview_state();
        canvas_state.mark_dirty(None);
    }

}

