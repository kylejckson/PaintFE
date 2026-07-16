impl ToolsPanel {
    fn clear_text_preview_cache(&mut self, canvas_state: &mut CanvasState) {
        self.text_state.cached_raster_buf.clear();
        self.text_state.cached_raster_w = 0;
        self.text_state.cached_raster_h = 0;
        self.text_state.cached_raster_off_x = 0;
        self.text_state.cached_raster_off_y = 0;
        self.text_state.cached_raster_origin = None;
        self.text_state.cached_raster_key.clear();
        self.text_state.preview_dirty = true;
        canvas_state.clear_preview_state();
    }

    #[allow(clippy::too_many_arguments, unused_variables)]
    fn handle_text_tool_input(
        &mut self,
        ui: &egui::Ui,
        canvas_state: &mut CanvasState,
        canvas_pos: Option<(u32, u32)>,
        canvas_pos_f32: Option<(f32, f32)>,
        canvas_pos_f32_clamped: Option<(f32, f32)>,
        canvas_pos_unclamped: Option<(f32, f32)>,
        raw_motion_events: &[(f32, f32)],
        painter: &egui::Painter,
        canvas_rect: Rect,
        workspace_rect: Rect,
        zoom: f32,
        primary_color_f32: [f32; 4],
        secondary_color_f32: [f32; 4],
        gpu_renderer: &mut Option<&mut crate::gpu::GpuRenderer>,
        stroke_event: &mut Option<StrokeEvent>,
        is_primary_down: bool,
        is_primary_released: bool,
        is_primary_clicked: bool,
        is_primary_pressed: bool,
        is_secondary_down: bool,
        is_secondary_pressed: bool,
        is_secondary_released: bool,
        is_secondary_clicked: bool,
        shift_held: bool,
        enter_pressed: bool,
        escape_pressed_global: bool,
    ) {
        if self.active_tool == Tool::Text {
            let escape_pressed = escape_pressed_global;
            let ctrl_enter = ui.input(|i| i.key_pressed(egui::Key::Enter) && i.modifiers.command);
            // Handle-grab checks below use `canvas_pos_unclamped`, which maps
            // the pointer to canvas space unconditionally (so handles remain
            // grabbable when dragged off-canvas) — but that means a press on
            // the top dock/context bar can coincidentally land "near" a handle
            // that's scrolled/zoomed into a similar screen position, and get
            // misread as a handle grab instead of a UI click. Require the
            // press to actually be within the canvas viewport on screen (a
            // signal `is_pointer_over_egui()` can't give here, since it
            // doesn't reliably detect panel content sharing the canvas's
            // Background z-order).
            let press_over_canvas_viewport = ui
                .input(|i| i.pointer.interact_pos())
                .is_some_and(|p| workspace_rect.contains(p));

            // Sync effects changes from UI panel to the text layer
            if self.text_state.text_effects_dirty && self.text_state.editing_text_layer {
                self.text_state.text_effects_dirty = false;
                if let Some(layer) = canvas_state.layers.get_mut(canvas_state.active_layer_index)
                    && let crate::canvas::LayerContent::Text(ref mut td) = layer.content
                {
                    td.effects = self.text_state.text_effects.clone();
                    td.mark_effects_dirty();
                }
                // Force-rasterize the editing layer so the change is visible immediately
                let idx = canvas_state.active_layer_index;
                canvas_state.force_rasterize_text_layer(idx);
                canvas_state.mark_dirty(None);
            }

            // Sync warp changes from UI panel to the text layer block
            if self.text_state.text_warp_dirty && self.text_state.editing_text_layer {
                self.text_state.text_warp_dirty = false;
                if let Some(layer) = canvas_state.layers.get_mut(canvas_state.active_layer_index)
                    && let crate::canvas::LayerContent::Text(ref mut td) = layer.content
                {
                    if let Some(bid) = self.text_state.active_block_id
                        && let Some(block) = td.blocks.iter_mut().find(|b| b.id == bid)
                    {
                        block.warp = self.text_state.text_warp.clone();
                    }
                    td.mark_dirty();
                }
                // Force-rasterize the editing layer so the change is visible immediately
                let idx = canvas_state.active_layer_index;
                canvas_state.force_rasterize_text_layer(idx);
                canvas_state.mark_dirty(None);
            }

            // Sync glyph overrides from glyph edit mode to the text layer block
            if self.text_state.glyph_overrides_dirty && self.text_state.editing_text_layer {
                self.text_state.glyph_overrides_dirty = false;
                if let Some(layer) = canvas_state.layers.get_mut(canvas_state.active_layer_index)
                    && let crate::canvas::LayerContent::Text(ref mut td) = layer.content
                {
                    if let Some(bid) = self.text_state.active_block_id
                        && let Some(block) = td.blocks.iter_mut().find(|b| b.id == bid)
                    {
                        block.glyph_overrides = self.text_state.glyph_overrides.clone();
                        block.cleanup_glyph_overrides();
                    }
                    td.mark_dirty();
                }
                // Force-rasterize the editing layer so the change is visible immediately
                let idx = canvas_state.active_layer_index;
                canvas_state.force_rasterize_text_layer(idx);
                canvas_state.mark_dirty(None);
                self.text_state.preview_dirty = true;
            }

            // Escape: cancel editing
            if escape_pressed && self.text_state.is_editing {
                self.text_state.text.clear();
                self.clear_text_preview_cache(canvas_state);
                self.text_state.cursor_pos = 0;
                self.text_state.is_editing = false;
                self.text_state.editing_text_layer = false;
                self.text_state.active_block_id = None;
                self.text_state.selection = crate::ops::text_layer::TextSelection::default();
                self.text_state.origin = None;
                self.text_state.dragging_handle = false;
                canvas_state.text_editing_layer = None;
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
                if self.text_state.editing_text_layer {
                    // Text layer mode: insert into the active block's runs
                    self.text_layer_insert_text(canvas_state, "\n");
                } else {
                    self.text_state
                        .text
                        .insert(self.text_state.cursor_pos, '\n');
                    self.text_state.cursor_pos += 1;
                    self.text_state.preview_dirty = true;
                }
            }

            // Compute move handle position - alignment-aware
            let handle_radius_screen = 6.0;
            let handle_canvas_pos = self.text_state.origin.map(|origin| {
                text_move_handle_canvas_pos(
                    origin,
                    self.text_state.alignment,
                    self.text_state.font_size,
                    zoom,
                )
            });

            // Detect hover over the move handle (for cursor icon)
            // Use canvas_pos_unclamped so the handle is reachable even when text is off-canvas
            self.text_state.hovering_handle = if self.text_state.is_editing {
                if let (Some(pos_f), Some(hp)) = (canvas_pos_unclamped, handle_canvas_pos) {
                    let dx = (pos_f.0 - hp.0) * zoom;
                    let dy = (pos_f.1 - hp.1) * zoom;
                    (dx * dx + dy * dy).sqrt() < handle_radius_screen + 4.0
                } else {
                    false
                }
            } else {
                false
            };

            // Detect hover over rotation / resize handles (for cursor icon)
            self.text_state.hovering_rotation_handle = false;
            self.text_state.hovering_resize_handle = None;
            if self.text_state.is_editing
                && self.text_state.editing_text_layer
                && !self.text_state.dragging_handle
                && self.text_state.text_box_drag.is_none()
                && let Some(pos_f) = canvas_pos_unclamped
                && let Some(origin) = self.text_state.origin
            {
                let font_size = self.text_state.font_size;
                let display_width = if let Some(mw) = self.text_state.active_block_max_width {
                    mw.max(font_size * 2.0)
                } else if let Some(ref font) = self.text_state.loaded_font {
                    use ab_glyph::{Font as _, ScaleFont as _};
                    let scaled = font.as_scaled(ab_glyph::PxScale {
                        x: font_size * self.text_state.width_scale,
                        y: font_size * self.text_state.height_scale,
                    });
                    let lines: Vec<&str> = self.text_state.text.split('\n').collect();
                    let mut max_w = font_size * 2.0;
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
                let text_h = self.text_state.active_block_height;
                let visual_h = if let Some(mh) = self.text_state.active_block_max_height {
                    mh.max(text_h)
                } else {
                    text_h
                };
                let pad = 4.0;
                let bx = {
                    use crate::ops::text::TextAlignment;
                    match self.text_state.alignment {
                        TextAlignment::Left => origin[0] - pad,
                        TextAlignment::Center => origin[0] - display_width * 0.5 - pad,
                        TextAlignment::Right => origin[0] - display_width - pad,
                    }
                };
                let by = origin[1] - pad;
                let bw = display_width + pad * 2.0;
                let _bh = visual_h + pad * 2.0;
                let s_left = canvas_rect.min.x + bx * zoom;
                let s_top = canvas_rect.min.y + by * zoom;
                let s_right = s_left + bw * zoom;

                let mx = pos_f.0 * zoom + canvas_rect.min.x;
                let my = pos_f.1 * zoom + canvas_rect.min.y;

                // Get block rotation and inverse-rotate the mouse to axis-aligned space
                let blk_rot = if let Some(layer) =
                    canvas_state.layers.get(canvas_state.active_layer_index)
                    && let crate::canvas::LayerContent::Text(ref td) = layer.content
                    && let Some(bid) = self.text_state.active_block_id
                    && let Some(block) = td.blocks.iter().find(|b| b.id == bid)
                {
                    block.rotation
                } else {
                    0.0
                };

                let s_bottom = s_top + (visual_h + pad * 2.0) * zoom;
                let pivot = Pos2::new((s_left + s_right) * 0.5, (s_top + s_bottom) * 0.5);
                let mp = rotate_screen_point(Pos2::new(mx, my), pivot, -blk_rot);
                let sx = mp.x;
                let sy = mp.y;

                let rot_handle_offset = 20.0;
                let rot_cx = (s_left + s_right) * 0.5;
                let rot_cy = s_top - rot_handle_offset;
                let rot_dist = ((sx - rot_cx).powi(2) + (sy - rot_cy).powi(2)).sqrt();
                if rot_dist < 10.0 {
                    self.text_state.hovering_rotation_handle = true;
                } else {
                    // Check corner resize handles hover
                    let hit_r = 8.0;
                    let bh_total = (visual_h + pad * 2.0) * zoom;
                    let corners = [
                        (s_left, s_top, TextBoxDragType::ResizeTopLeft),
                        (s_right, s_top, TextBoxDragType::ResizeTopRight),
                        (s_left, s_top + bh_total, TextBoxDragType::ResizeBottomLeft),
                        (
                            s_right,
                            s_top + bh_total,
                            TextBoxDragType::ResizeBottomRight,
                        ),
                    ];
                    for &(cx, cy, drag_type) in &corners {
                        if (sx - cx).abs() < hit_r && (sy - cy).abs() < hit_r {
                            self.text_state.hovering_resize_handle = Some(drag_type);
                            break;
                        }
                    }
                    // Check edge resize handles hover (midpoints of left/right edges)
                    if self.text_state.hovering_resize_handle.is_none() {
                        let mid_y = s_top + bh_total * 0.5;
                        if (sx - s_left).abs() < hit_r && (sy - mid_y).abs() < bh_total * 0.5 {
                            self.text_state.hovering_resize_handle =
                                Some(TextBoxDragType::ResizeLeft);
                        } else if (sx - s_right).abs() < hit_r
                            && (sy - mid_y).abs() < bh_total * 0.5
                        {
                            self.text_state.hovering_resize_handle =
                                Some(TextBoxDragType::ResizeRight);
                        }
                    }
                }
            }

            // Handle dragging the move handle
            // Use canvas_pos_unclamped so the handle is grabbable even when text is off-canvas
            if is_primary_pressed
                && self.text_state.is_editing
                && self.text_state.text_box_drag.is_none()
                && press_over_canvas_viewport
                && let (Some(pos_f), Some(hp)) = (canvas_pos_unclamped, handle_canvas_pos)
            {
                let dx = (pos_f.0 - hp.0) * zoom;
                let dy = (pos_f.1 - hp.1) * zoom;
                let dist = (dx * dx + dy * dy).sqrt();
                if dist < handle_radius_screen + 4.0 {
                    self.text_state.dragging_handle = true;
                    if let Some(origin) = self.text_state.origin {
                        self.text_state.drag_offset = [pos_f.0 - origin[0], pos_f.1 - origin[1]];
                    }
                    // --- Text layer drag optimization: cache block pixels ---
                    // At drag start, extract the active block's rasterized pixels
                    // into cached_raster_buf and re-rasterize the layer without
                    // the active block. During drag, we blit the cached block at
                    // the new offset into preview_layer (avoiding per-frame
                    // re-rasterization of the entire text layer).
                    self.text_state.text_layer_drag_cached = false;
                    if self.text_state.editing_text_layer
                        && let Some(bid) = self.text_state.active_block_id
                        && let Some(layer) =
                            canvas_state.layers.get_mut(canvas_state.active_layer_index)
                        && let crate::canvas::LayerContent::Text(ref mut td) = layer.content
                        && let Some(block_idx) = td.blocks.iter().position(|b| b.id == bid)
                    {
                        // 1. Ensure the layer is fully rasterized first
                        let w = canvas_state.width;
                        let h = canvas_state.height;

                        // 2. Extract the active block's tight pixel buffer.
                        //    Use the per-block cached_raster if available,
                        //    otherwise rasterize just this block into a temp buffer.
                        let block = &td.blocks[block_idx];
                        let (buf, buf_w, buf_h, off_x, off_y, origin) = if let Some(ref cached) =
                            block.cached_raster
                            && cached.buf_w > 0
                            && cached.buf_h > 0
                        {
                            let dx = (block.position[0] - cached.origin[0]).round() as i32;
                            let dy = (block.position[1] - cached.origin[1]).round() as i32;
                            (
                                cached.buf.clone(),
                                cached.buf_w,
                                cached.buf_h,
                                cached.off_x + dx,
                                cached.off_y + dy,
                                block.position,
                            )
                        } else {
                            // No per-block cache - rasterize just this block
                            let mut temp = crate::canvas::TiledImage::new(w, h);
                            let mut cov = std::mem::take(&mut canvas_state.text_coverage_buf);
                            let mut gc = std::mem::take(&mut canvas_state.text_glyph_cache);
                            crate::ops::text_layer::rasterize_single_block(
                                &mut td.blocks[block_idx],
                                td.text_content_generation,
                                w,
                                h,
                                &mut temp,
                                &mut cov,
                                &mut gc,
                            );
                            canvas_state.text_coverage_buf = cov;
                            canvas_state.text_glyph_cache = gc;
                            // Extract tight bounds from the temp tiled image
                            let block_pos = td.blocks[block_idx].position;
                            let raw = temp.extract_region_rgba(0, 0, w, h);
                            // Find tight AABB of non-transparent pixels
                            let (bx, by, bw, bh, tight) =
                                crate::ops::text_layer::find_tight_bounds_rgba(&raw, w, h);
                            if bw > 0 && bh > 0 {
                                (tight, bw, bh, bx as i32, by as i32, block_pos)
                            } else {
                                (Vec::new(), 0, 0, 0, 0, block_pos)
                            }
                        };

                        if buf_w > 0 && buf_h > 0 {
                            self.text_state.cached_raster_buf = buf;
                            self.text_state.cached_raster_w = buf_w;
                            self.text_state.cached_raster_h = buf_h;
                            self.text_state.cached_raster_off_x = off_x;
                            self.text_state.cached_raster_off_y = off_y;
                            self.text_state.cached_raster_origin = Some(origin);

                            // 3. Re-rasterize the layer WITHOUT the active block.
                            //    Temporarily empty the block's runs, rasterize, restore.
                            let saved_runs = std::mem::take(&mut td.blocks[block_idx].runs);
                            td.mark_dirty();
                            {
                                let mut cov = std::mem::take(&mut canvas_state.text_coverage_buf);
                                let mut gc = std::mem::take(&mut canvas_state.text_glyph_cache);
                                let new_pixels = td.rasterize(w, h, &mut cov, &mut gc);
                                td.raster_generation = td.cache_generation;
                                layer.pixels = new_pixels;
                                layer.invalidate_lod();
                                layer.gpu_generation += 1;
                                canvas_state.text_coverage_buf = cov;
                                canvas_state.text_glyph_cache = gc;
                            }
                            // Restore the block's runs
                            if let crate::canvas::LayerContent::Text(ref mut td2) =
                                canvas_state.layers[canvas_state.active_layer_index].content
                            {
                                td2.blocks[block_idx].runs = saved_runs;
                                // Mark dirty again so the full layer will re-rasterize on drag end
                                td2.mark_position_dirty();
                            }

                            self.text_state.text_layer_drag_cached = true;
                        }
                    }
                }
            }

            // --- Resize / Rotate / Delete handle interactions ---
            // Use canvas_pos_unclamped because handles (rotation, delete) can be outside image bounds
            if is_primary_pressed
                && self.text_state.is_editing
                && !self.text_state.dragging_handle
                && self.text_state.text_box_drag.is_none()
                && self.text_state.editing_text_layer
                && press_over_canvas_viewport
                && let Some(pos_f) = canvas_pos_unclamped
                && let Some(origin) = self.text_state.origin
            {
                let font_size = self.text_state.font_size;
                let display_width = if let Some(mw) = self.text_state.active_block_max_width {
                    mw.max(font_size * 2.0)
                } else {
                    // Compute natural width for handle positioning
                    if let Some(ref font) = self.text_state.loaded_font {
                        use ab_glyph::{Font as _, ScaleFont as _};
                        let scaled = font.as_scaled(ab_glyph::PxScale {
                            x: font_size * self.text_state.width_scale,
                            y: font_size * self.text_state.height_scale,
                        });
                        let lines: Vec<&str> = self.text_state.text.split('\n').collect();
                        let mut max_w = font_size * 2.0;
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
                    }
                };
                let text_h = self.text_state.active_block_height;
                let visual_h = if let Some(mh) = self.text_state.active_block_max_height {
                    mh.max(text_h)
                } else {
                    text_h
                };
                let pad = 4.0;
                let bx = {
                    use crate::ops::text::TextAlignment;
                    match self.text_state.alignment {
                        TextAlignment::Left => origin[0] - pad,
                        TextAlignment::Center => origin[0] - display_width * 0.5 - pad,
                        TextAlignment::Right => origin[0] - display_width - pad,
                    }
                };
                let by = origin[1] - pad;
                let bw = display_width + pad * 2.0;
                let bh = visual_h + pad * 2.0;

                // Screen-space corners
                let s_left = canvas_rect.min.x + bx * zoom;
                let s_top = canvas_rect.min.y + by * zoom;
                let s_right = s_left + bw * zoom;
                let s_bottom = s_top + bh * zoom;

                // Inverse-rotate mouse position into axis-aligned space
                let blk_rot = if let Some(layer) =
                    canvas_state.layers.get(canvas_state.active_layer_index)
                    && let crate::canvas::LayerContent::Text(ref td) = layer.content
                    && let Some(bid) = self.text_state.active_block_id
                    && let Some(block) = td.blocks.iter().find(|b| b.id == bid)
                {
                    block.rotation
                } else {
                    0.0
                };
                let pivot = Pos2::new((s_left + s_right) * 0.5, (s_top + s_bottom) * 0.5);
                let raw_sx = pos_f.0 * zoom + canvas_rect.min.x;
                let raw_sy = pos_f.1 * zoom + canvas_rect.min.y;
                let mp = rotate_screen_point(Pos2::new(raw_sx, raw_sy), pivot, -blk_rot);
                let sx = mp.x;
                let sy = mp.y;
                let hit_r = 8.0; // screen pixel hit radius

                // Check delete button first (top-right outside)
                let del_offset = 14.0;
                let del_cx = s_right + del_offset;
                let del_cy = s_top - del_offset;
                let del_dist = ((sx - del_cx).powi(2) + (sy - del_cy).powi(2)).sqrt();
                if del_dist < 12.0 {
                    // Delete this block
                    self.text_state.text_box_click_guard = true;
                    if let Some(bid) = self.text_state.active_block_id {
                        if let Some(layer) =
                            canvas_state.layers.get_mut(canvas_state.active_layer_index)
                            && let crate::canvas::LayerContent::Text(ref mut td) = layer.content
                        {
                            td.blocks.retain(|b| b.id != bid);
                            td.mark_dirty();
                        }
                        self.text_state.text.clear();
                        self.clear_text_preview_cache(canvas_state);
                        self.text_state.cursor_pos = 0;
                        self.text_state.is_editing = false;
                        self.text_state.editing_text_layer = false;
                        self.text_state.active_block_id = None;
                        self.text_state.origin = None;
                        self.text_state.active_block_max_width = None;
                        self.text_state.active_block_max_height = None;
                        canvas_state.text_editing_layer = None;
                        canvas_state.clear_preview_state();
                        let idx = canvas_state.active_layer_index;
                        canvas_state.force_rasterize_text_layer(idx);
                        canvas_state.mark_dirty(None);
                    }
                }
                // Check rotation handle (above top-center)
                else {
                    let rot_handle_offset = 20.0;
                    let rot_cx = (s_left + s_right) * 0.5;
                    let rot_cy = s_top - rot_handle_offset;
                    let rot_dist = ((sx - rot_cx).powi(2) + (sy - rot_cy).powi(2)).sqrt();
                    if rot_dist < 10.0 {
                        self.text_state.text_box_click_guard = true;
                        self.text_state.text_box_drag = Some(TextBoxDragType::Rotate);
                        self.text_state.text_box_drag_start_mouse = [pos_f.0, pos_f.1];
                        // Get current block rotation
                        let cur_rot = if let Some(layer) =
                            canvas_state.layers.get(canvas_state.active_layer_index)
                            && let crate::canvas::LayerContent::Text(ref td) = layer.content
                            && let Some(bid) = self.text_state.active_block_id
                            && let Some(block) = td.blocks.iter().find(|b| b.id == bid)
                        {
                            block.rotation
                        } else {
                            0.0
                        };
                        self.text_state.text_box_drag_start_rotation = cur_rot;
                        self.text_state.text_box_drag_start_origin = origin;
                    } else {
                        // Check corner resize handles
                        let corners_canvas = [
                            (bx, by, TextBoxDragType::ResizeTopLeft),
                            (bx + bw, by, TextBoxDragType::ResizeTopRight),
                            (bx, by + bh, TextBoxDragType::ResizeBottomLeft),
                            (bx + bw, by + bh, TextBoxDragType::ResizeBottomRight),
                        ];
                        for &(cx, cy, drag_type) in &corners_canvas {
                            let scx = canvas_rect.min.x + cx * zoom;
                            let scy = canvas_rect.min.y + cy * zoom;
                            if (sx - scx).abs() < hit_r && (sy - scy).abs() < hit_r {
                                self.text_state.text_box_click_guard = true;
                                self.text_state.text_box_drag = Some(drag_type);
                                self.text_state.text_box_drag_start_mouse = [pos_f.0, pos_f.1];
                                self.text_state.text_box_drag_start_width =
                                    self.text_state.active_block_max_width;
                                self.text_state.text_box_drag_start_height = Some(
                                    self.text_state
                                        .active_block_max_height
                                        .map(|mh| mh.max(self.text_state.active_block_height))
                                        .unwrap_or(self.text_state.active_block_height),
                                );
                                self.text_state.text_box_drag_start_origin = origin;
                                break;
                            }
                        }
                    }
                }
            }

            // --- Drag-to-select text within the active block ---
            // If primary pressed within the text content area (not on any handle),
            // set the selection anchor and start dragging.
            if is_primary_pressed
                && self.text_state.is_editing
                && self.text_state.editing_text_layer
                && !self.text_state.dragging_handle
                && self.text_state.text_box_drag.is_none()
                && !self.text_state.text_box_click_guard
                && let Some(pos_f) = canvas_pos_f32
                && let Some(origin) = self.text_state.origin
            {
                // Check if click is within the text box bounds (not on handles)
                let font_size = self.text_state.font_size;
                let display_w = if let Some(mw) = self.text_state.active_block_max_width {
                    mw.max(font_size * 2.0)
                } else {
                    // Compute natural width from font metrics (same as handle positioning)
                    if let Some(ref font) = self.text_state.loaded_font {
                        use ab_glyph::{Font as _, ScaleFont as _};
                        let scaled = font.as_scaled(ab_glyph::PxScale {
                            x: font_size * self.text_state.width_scale,
                            y: font_size * self.text_state.height_scale,
                        });
                        let lines: Vec<&str> = self.text_state.text.split('\n').collect();
                        let mut max_w = font_size * 2.0;
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
                    }
                };
                let text_h = self.text_state.active_block_height;
                let visual_h = self
                    .text_state
                    .active_block_max_height
                    .map(|mh| mh.max(text_h))
                    .unwrap_or(text_h);
                let pad = 4.0;
                let bx = {
                    use crate::ops::text::TextAlignment;
                    match self.text_state.alignment {
                        TextAlignment::Left => origin[0] - pad,
                        TextAlignment::Center => origin[0] - display_w * 0.5 - pad,
                        TextAlignment::Right => origin[0] - display_w - pad,
                    }
                };
                let by = origin[1] - pad;
                let bw = display_w + pad * 2.0;
                let bh = visual_h + pad * 2.0;

                if pos_f.0 >= bx && pos_f.0 <= bx + bw && pos_f.1 >= by && pos_f.1 <= by + bh {
                    // Inside text box - set anchor at click position
                    self.text_state.text_select_dragging = true;
                    // Position cursor at click point (same logic as same-block click)
                    let rel_x = pos_f.0 - origin[0];
                    let rel_y = pos_f.1 - origin[1];
                    let lh = self.text_state.cached_line_height.max(1.0);
                    let visual_line_count = if let (Some(mw), Some(font)) = (
                        self.text_state.active_block_max_width,
                        &self.text_state.loaded_font,
                    ) {
                        let ls = self.text_state.letter_spacing;
                        let vlines: Vec<String> = self
                            .text_state
                            .text
                            .split('\n')
                            .flat_map(|line| {
                                crate::ops::text::word_wrap_line(
                                    line,
                                    font,
                                    self.text_state.font_size,
                                    mw,
                                    ls,
                                    self.text_state.width_scale,
                                    self.text_state.height_scale,
                                )
                            })
                            .collect();
                        vlines.len()
                    } else {
                        self.text_state.text.split('\n').count()
                    };
                    let line_idx =
                        ((rel_y / lh).floor() as usize).min(visual_line_count.saturating_sub(1));
                    let mut best_pos = 0;
                    if !self.text_state.cached_line_advances.is_empty()
                        && line_idx < self.text_state.cached_line_advances.len()
                    {
                        let advances = &self.text_state.cached_line_advances[line_idx];
                        let mut best_dist = f32::MAX;
                        for (ci, &adv) in advances.iter().enumerate() {
                            let dist = (adv - rel_x).abs();
                            if dist < best_dist {
                                best_dist = dist;
                                best_pos = ci;
                            }
                        }
                        let max_chars = advances.len().saturating_sub(1);
                        best_pos = best_pos.min(max_chars);
                    }
                    let byte_pos = if let Some(pos) =
                        self.text_layer_byte_pos_from_point(canvas_state, pos_f)
                    {
                        pos
                    } else if let (Some(mw), Some(font)) = (
                        self.text_state.active_block_max_width,
                        &self.text_state.loaded_font,
                    ) {
                        let ls = self.text_state.letter_spacing;
                        Self::visual_to_byte_pos(
                            &self.text_state.text,
                            line_idx,
                            best_pos,
                            font,
                            self.text_state.font_size,
                            mw,
                            ls,
                            self.text_state.width_scale,
                            self.text_state.height_scale,
                        )
                    } else {
                        let lines: Vec<&str> = self.text_state.text.split('\n').collect();
                        let clamped = line_idx.min(lines.len().saturating_sub(1));
                        let line_start: usize =
                            lines.iter().take(clamped).map(|l| l.len() + 1).sum();
                        let line_text = lines[clamped];
                        let clamped_pos = best_pos.min(line_text.chars().count());
                        let byte_offset: usize = line_text
                            .chars()
                            .take(clamped_pos)
                            .map(|c| c.len_utf8())
                            .sum();
                        line_start + byte_offset
                    };
                    self.text_state.cursor_pos = byte_pos;
                    // Set both anchor and cursor to same position (start of drag)
                    self.text_state.selection.anchor = if let Some(layer) =
                        canvas_state.layers.get(canvas_state.active_layer_index)
                        && let crate::canvas::LayerContent::Text(ref td) = layer.content
                        && let Some(bid) = self.text_state.active_block_id
                        && let Some(block) = td.blocks.iter().find(|b| b.id == bid)
                    {
                        block.flat_offset_to_run_pos(byte_pos)
                    } else {
                        self.text_state.selection.anchor
                    };
                    self.text_state.selection.cursor = self.text_state.selection.anchor;
                    self.text_state.preview_dirty = true;
                }
            }

            // Process text box drag (resize / rotate)
            if let Some(drag_type) = self.text_state.text_box_drag {
                if is_primary_down && let Some(pos_f) = canvas_pos_unclamped {
                    let raw_dx = pos_f.0 - self.text_state.text_box_drag_start_mouse[0];
                    let raw_dy = pos_f.1 - self.text_state.text_box_drag_start_mouse[1];
                    let start_origin = self.text_state.text_box_drag_start_origin;
                    let font_size = self.text_state.font_size;
                    let now = ui.input(|i| i.time);
                    // Keep handle geometry at event rate while pacing expensive
                    // text-layer raster/composite work. A final quality raster is
                    // always performed on release below.
                    let live_raster_due = self.text_state.text_box_last_raster_time == 0.0
                        || now - self.text_state.text_box_last_raster_time >= 0.08;

                    // Project drag delta onto the box's local axes (rotation-aware)
                    let blk_rot = if let Some(layer) =
                        canvas_state.layers.get(canvas_state.active_layer_index)
                        && let crate::canvas::LayerContent::Text(ref td) = layer.content
                        && let Some(bid) = self.text_state.active_block_id
                        && let Some(block) = td.blocks.iter().find(|b| b.id == bid)
                    {
                        block.rotation
                    } else {
                        0.0
                    };
                    let cos_r = blk_rot.cos();
                    let sin_r = blk_rot.sin();
                    // Local-X and local-Y components of the drag delta
                    let dx = raw_dx * cos_r + raw_dy * sin_r;
                    let dy = -raw_dx * sin_r + raw_dy * cos_r;

                    let content_height = self.text_state.active_block_height;

                    // Helper closure: compute natural text width from font metrics
                    let compute_natural_width = || -> f32 {
                        if let Some(ref font) = self.text_state.loaded_font {
                            use ab_glyph::{Font as _, ScaleFont as _};
                            let scaled = font.as_scaled(ab_glyph::PxScale {
                                x: font_size * self.text_state.width_scale,
                                y: font_size * self.text_state.height_scale,
                            });
                            let ls = self.text_state.letter_spacing;
                            let lines: Vec<&str> = self.text_state.text.split('\n').collect();
                            let mut max_w = font_size * 2.0;
                            for line in &lines {
                                let mut w = 0.0f32;
                                let mut prev = None;
                                for ch in line.chars() {
                                    let gid = font.glyph_id(ch);
                                    if let Some(prev_id) = prev {
                                        w += scaled.kern(prev_id, gid);
                                        w += ls;
                                    }
                                    w += scaled.h_advance(gid);
                                    prev = Some(gid);
                                }
                                max_w = max_w.max(w);
                            }
                            max_w
                        } else {
                            font_size * 2.0
                        }
                    };

                    // Determine width change
                    let is_right = matches!(
                        drag_type,
                        TextBoxDragType::ResizeRight
                            | TextBoxDragType::ResizeTopRight
                            | TextBoxDragType::ResizeBottomRight
                    );
                    let is_left = matches!(
                        drag_type,
                        TextBoxDragType::ResizeLeft
                            | TextBoxDragType::ResizeTopLeft
                            | TextBoxDragType::ResizeBottomLeft
                    );
                    let is_top = matches!(
                        drag_type,
                        TextBoxDragType::ResizeTopLeft | TextBoxDragType::ResizeTopRight
                    );
                    let is_bottom = matches!(
                        drag_type,
                        TextBoxDragType::ResizeBottomLeft | TextBoxDragType::ResizeBottomRight
                    );

                    if is_right || is_left {
                        let start_w = self
                            .text_state
                            .text_box_drag_start_width
                            .unwrap_or_else(&compute_natural_width);
                        let new_w = if is_right {
                            (start_w + dx).max(font_size * 2.0)
                        } else {
                            (start_w - dx).max(font_size * 2.0)
                        };
                        self.text_state.active_block_max_width = Some(new_w);

                        // For left-side handles, shift origin along local-X to keep right edge fixed
                        let mut new_origin_x = start_origin[0];
                        let mut new_origin_y = start_origin[1];
                        if is_left {
                            let width_delta = start_w - new_w;
                            new_origin_x += width_delta * cos_r;
                            new_origin_y += width_delta * sin_r;
                        }

                        // For top handles, shift origin along local-Y to keep bottom edge fixed
                        if is_top {
                            let start_h = self
                                .text_state
                                .text_box_drag_start_height
                                .unwrap_or(content_height);
                            let new_h = (start_h - dy).max(font_size);
                            let h_delta = start_h - new_h;
                            // Shift origin along local-Y direction
                            new_origin_x += h_delta * (-sin_r);
                            new_origin_y += h_delta * cos_r;
                            self.text_state.active_block_max_height = Some(new_h);
                        }

                        // For bottom handles, set max_height from drag
                        if is_bottom {
                            let start_h = self
                                .text_state
                                .text_box_drag_start_height
                                .unwrap_or(content_height);
                            let new_h = (start_h + dy).max(font_size);
                            self.text_state.active_block_max_height = Some(new_h);
                        }

                        if is_left || is_top {
                            self.text_state.origin = Some([new_origin_x, new_origin_y]);
                        }

                        // Write to TextBlock immediately
                        if let Some(layer) =
                            canvas_state.layers.get_mut(canvas_state.active_layer_index)
                            && let crate::canvas::LayerContent::Text(ref mut td) = layer.content
                            && let Some(bid) = self.text_state.active_block_id
                            && let Some(block) = td.blocks.iter_mut().find(|b| b.id == bid)
                        {
                            block.max_width = Some(new_w);
                            if is_top || is_bottom {
                                block.max_height = self.text_state.active_block_max_height;
                            }
                            if is_left || is_top {
                                block.position = [new_origin_x, new_origin_y];
                            }
                            if live_raster_due {
                                td.mark_dirty();
                            }
                        }
                        if live_raster_due {
                            self.text_state.text_box_last_raster_time = now;
                            let idx = canvas_state.active_layer_index;
                            canvas_state.force_rasterize_text_layer(idx);
                            let display_h = self
                                .text_state
                                .active_block_max_height
                                .unwrap_or(self.text_state.active_block_height)
                                .max(self.text_state.active_block_height);
                            let radius = (new_w.hypot(display_h) * 0.5 + font_size * 3.0)
                                .max(font_size * 3.0);
                            let current_origin = self.text_state.origin.unwrap_or(start_origin);
                            let current_center_x = match self.text_state.alignment {
                                crate::ops::text::TextAlignment::Left => {
                                    current_origin[0] + new_w * 0.5
                                }
                                crate::ops::text::TextAlignment::Center => current_origin[0],
                                crate::ops::text::TextAlignment::Right => {
                                    current_origin[0] - new_w * 0.5
                                }
                            };
                            let current_center =
                                egui::pos2(current_center_x, current_origin[1] + display_h * 0.5);
                            let start_w =
                                self.text_state.text_box_drag_start_width.unwrap_or(new_w);
                            let start_h = self
                                .text_state
                                .text_box_drag_start_height
                                .unwrap_or(display_h);
                            let start_radius = (start_w.hypot(start_h) * 0.5 + font_size * 3.0)
                                .max(font_size * 3.0);
                            let start_center_x = match self.text_state.alignment {
                                crate::ops::text::TextAlignment::Left => {
                                    start_origin[0] + start_w * 0.5
                                }
                                crate::ops::text::TextAlignment::Center => start_origin[0],
                                crate::ops::text::TextAlignment::Right => {
                                    start_origin[0] - start_w * 0.5
                                }
                            };
                            let start_center =
                                egui::pos2(start_center_x, start_origin[1] + start_h * 0.5);
                            let dirty = egui::Rect::from_center_size(
                                current_center,
                                egui::vec2(radius * 2.0, radius * 2.0),
                            )
                            .union(egui::Rect::from_center_size(
                                start_center,
                                egui::vec2(start_radius * 2.0, start_radius * 2.0),
                            ));
                            canvas_state.mark_dirty(Some(dirty));
                        }
                    }

                    if matches!(drag_type, TextBoxDragType::Rotate) {
                        // Compute rotation angle from mouse position relative to box center
                        // Box center must match the overlay rotation pivot
                        let display_w = self
                            .text_state
                            .active_block_max_width
                            .unwrap_or(font_size * 2.0)
                            .max(font_size * 2.0);
                        let text_h = self.text_state.active_block_height;
                        let visual_h = self
                            .text_state
                            .active_block_max_height
                            .map(|mh| mh.max(text_h))
                            .unwrap_or(text_h);
                        let box_center_x = {
                            use crate::ops::text::TextAlignment;
                            match self.text_state.alignment {
                                TextAlignment::Left => start_origin[0] + display_w * 0.5,
                                TextAlignment::Center => start_origin[0],
                                TextAlignment::Right => start_origin[0] - display_w * 0.5,
                            }
                        };
                        let box_center_y = start_origin[1] + visual_h * 0.5;
                        let start_angle = (self.text_state.text_box_drag_start_mouse[1]
                            - box_center_y)
                            .atan2(self.text_state.text_box_drag_start_mouse[0] - box_center_x);
                        let current_angle = (pos_f.1 - box_center_y).atan2(pos_f.0 - box_center_x);
                        let delta = current_angle - start_angle;
                        let new_rotation = self.text_state.text_box_drag_start_rotation + delta;
                        // Write rotation to TextBlock
                        if let Some(layer) =
                            canvas_state.layers.get_mut(canvas_state.active_layer_index)
                            && let crate::canvas::LayerContent::Text(ref mut td) = layer.content
                            && let Some(bid) = self.text_state.active_block_id
                            && let Some(block) = td.blocks.iter_mut().find(|b| b.id == bid)
                        {
                            block.rotation = new_rotation;
                            if live_raster_due {
                                td.mark_dirty();
                            }
                        }
                        if live_raster_due {
                            self.text_state.text_box_last_raster_time = now;
                            let idx = canvas_state.active_layer_index;
                            canvas_state.force_rasterize_text_layer(idx);
                            let display_w = self
                                .text_state
                                .active_block_max_width
                                .unwrap_or(font_size * 2.0)
                                .max(font_size * 2.0);
                            let display_h = self
                                .text_state
                                .active_block_max_height
                                .unwrap_or(self.text_state.active_block_height)
                                .max(self.text_state.active_block_height);
                            let radius = (display_w.hypot(display_h) * 0.5 + font_size * 3.0)
                                .max(font_size * 3.0);
                            let center_x = match self.text_state.alignment {
                                crate::ops::text::TextAlignment::Left => {
                                    start_origin[0] + display_w * 0.5
                                }
                                crate::ops::text::TextAlignment::Center => start_origin[0],
                                crate::ops::text::TextAlignment::Right => {
                                    start_origin[0] - display_w * 0.5
                                }
                            };
                            let center = egui::pos2(center_x, start_origin[1] + display_h * 0.5);
                            canvas_state.mark_dirty(Some(egui::Rect::from_center_size(
                                center,
                                egui::vec2(radius * 2.0, radius * 2.0),
                            )));
                        }
                    }
                }
                if is_primary_released {
                    if let Some(layer) =
                        canvas_state.layers.get_mut(canvas_state.active_layer_index)
                        && let crate::canvas::LayerContent::Text(ref mut td) = layer.content
                    {
                        td.mark_dirty();
                    }
                    let idx = canvas_state.active_layer_index;
                    canvas_state.force_rasterize_text_layer(idx);
                    canvas_state.mark_dirty(None);
                    self.text_state.text_box_last_raster_time = 0.0;
                    self.text_state.text_box_drag = None;
                }
            }

            // --- Drag-to-select: update cursor position during drag ---
            if self.text_state.text_select_dragging
                && is_primary_down
                && self.text_state.editing_text_layer
                && let Some(pos_f) = canvas_pos_f32
                && let Some(origin) = self.text_state.origin
            {
                // Same cursor-positioning logic as same-block click
                let rel_x = pos_f.0 - origin[0];
                let rel_y = pos_f.1 - origin[1];
                let lh = self.text_state.cached_line_height.max(1.0);
                let visual_line_count = if let (Some(mw), Some(font)) = (
                    self.text_state.active_block_max_width,
                    &self.text_state.loaded_font,
                ) {
                    let ls = self.text_state.letter_spacing;
                    let vlines: Vec<String> = self
                        .text_state
                        .text
                        .split('\n')
                        .flat_map(|line| {
                            crate::ops::text::word_wrap_line(
                                line,
                                font,
                                self.text_state.font_size,
                                mw,
                                ls,
                                self.text_state.width_scale,
                                self.text_state.height_scale,
                            )
                        })
                        .collect();
                    vlines.len()
                } else {
                    self.text_state.text.split('\n').count()
                };
                let line_idx =
                    ((rel_y / lh).floor() as usize).min(visual_line_count.saturating_sub(1));
                let mut best_pos = 0;
                if !self.text_state.cached_line_advances.is_empty()
                    && line_idx < self.text_state.cached_line_advances.len()
                {
                    let advances = &self.text_state.cached_line_advances[line_idx];
                    let mut best_dist = f32::MAX;
                    for (ci, &adv) in advances.iter().enumerate() {
                        let dist = (adv - rel_x).abs();
                        if dist < best_dist {
                            best_dist = dist;
                            best_pos = ci;
                        }
                    }
                    let max_chars = advances.len().saturating_sub(1);
                    best_pos = best_pos.min(max_chars);
                }
                let byte_pos = if let Some(pos) =
                    self.text_layer_byte_pos_from_point(canvas_state, pos_f)
                {
                    pos
                } else if let (Some(mw), Some(font)) = (
                    self.text_state.active_block_max_width,
                    &self.text_state.loaded_font,
                ) {
                    let ls = self.text_state.letter_spacing;
                    Self::visual_to_byte_pos(
                        &self.text_state.text,
                        line_idx,
                        best_pos,
                        font,
                        self.text_state.font_size,
                        mw,
                        ls,
                        self.text_state.width_scale,
                        self.text_state.height_scale,
                    )
                } else {
                    let lines: Vec<&str> = self.text_state.text.split('\n').collect();
                    let clamped = line_idx.min(lines.len().saturating_sub(1));
                    let line_start: usize = lines.iter().take(clamped).map(|l| l.len() + 1).sum();
                    let line_text = lines[clamped];
                    let clamped_pos = best_pos.min(line_text.chars().count());
                    let byte_offset: usize = line_text
                        .chars()
                        .take(clamped_pos)
                        .map(|c| c.len_utf8())
                        .sum();
                    line_start + byte_offset
                };
                self.text_state.cursor_pos = byte_pos;
                // Extend selection (shift-hold behavior: update cursor but keep anchor)
                self.text_layer_update_selection(canvas_state, true);
                self.text_state.preview_dirty = true;
            }

            if self.text_state.dragging_handle
                && is_primary_down
                && let Some(pos_f) = canvas_pos_unclamped
            {
                let new_x = pos_f.0 - self.text_state.drag_offset[0];
                let new_y = pos_f.1 - self.text_state.drag_offset[1];
                self.text_state.origin = Some([new_x, new_y]);

                if self.text_state.editing_text_layer && self.text_state.text_layer_drag_cached {
                    // Fast path: use cached block pixels + preview layer.
                    // layer.pixels already shows all blocks EXCEPT the dragging one
                    // (set up at drag start). We just blit the cached block at the
                    // new offset - zero re-rasterization per frame.
                    if let Some(cached_origin) = self.text_state.cached_raster_origin {
                        let dx = new_x - cached_origin[0];
                        let dy = new_y - cached_origin[1];
                        let off_x = self.text_state.cached_raster_off_x + dx as i32;
                        let off_y = self.text_state.cached_raster_off_y + dy as i32;
                        let buf_w = self.text_state.cached_raster_w;
                        let buf_h = self.text_state.cached_raster_h;

                        let mut preview = TiledImage::new(canvas_state.width, canvas_state.height);
                        preview.blit_rgba_at(
                            off_x,
                            off_y,
                            buf_w,
                            buf_h,
                            &self.text_state.cached_raster_buf,
                        );

                        canvas_state.preview_layer = Some(preview);
                        canvas_state.preview_blend_mode = BlendMode::Normal;
                        canvas_state.preview_force_composite = false;
                        canvas_state.preview_is_eraser = false;
                        canvas_state.preview_downscale = 1;
                        canvas_state.preview_flat_ready = false;
                        let visible_bounds =
                            Self::clip_preview_bounds(canvas_state, off_x, off_y, buf_w, buf_h);
                        // Merge old + new preview bounds so both regions get recomposited
                        let combined_bounds = if let Some(old) = canvas_state.preview_stroke_bounds
                        {
                            Some(old.union(visible_bounds.unwrap_or(old)))
                        } else {
                            visible_bounds
                        };
                        canvas_state.preview_stroke_bounds = combined_bounds;
                        if canvas_state.preview_texture_cache.is_some() {
                            canvas_state.preview_dirty_rect = combined_bounds;
                        } else {
                            canvas_state.preview_texture_cache = None;
                        }
                        // Use targeted dirty rect instead of full-canvas dirty
                        if let Some(bounds) = combined_bounds {
                            canvas_state.mark_dirty(Some(bounds));
                        } else {
                            canvas_state.mark_dirty(None);
                        }
                    }
                    self.text_state.preview_dirty = false;
                } else if self.text_state.editing_text_layer {
                    // Fallback: force-rasterize (only if cached drag setup failed)
                    if let Some(layer) =
                        canvas_state.layers.get_mut(canvas_state.active_layer_index)
                        && let crate::canvas::LayerContent::Text(ref mut td) = layer.content
                        && let Some(bid) = self.text_state.active_block_id
                        && let Some(block) = td.blocks.iter_mut().find(|b| b.id == bid)
                    {
                        block.position = [new_x, new_y];
                        td.mark_position_dirty();
                    }
                    let idx = canvas_state.active_layer_index;
                    canvas_state.force_rasterize_text_layer(idx);
                    canvas_state.mark_dirty(None);
                    self.text_state.preview_dirty = false;
                } else {
                    // Raster text: reuse cached raster buffer and re-blit at new origin offset.
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
                            canvas_state.preview_force_composite =
                                self.properties.blending_mode != BlendMode::Normal;
                            canvas_state.preview_is_eraser = false;
                            canvas_state.preview_downscale = 1;
                            canvas_state.preview_flat_ready = false;
                            let visible_bounds =
                                Self::clip_preview_bounds(canvas_state, off_x, off_y, buf_w, buf_h);
                            canvas_state.preview_stroke_bounds = visible_bounds;
                            if canvas_state.preview_texture_cache.is_some() {
                                canvas_state.preview_dirty_rect = visible_bounds;
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
            }

            if is_primary_released {
                // Finalize text layer drag: apply final position, re-rasterize once
                if self.text_state.dragging_handle
                    && self.text_state.text_layer_drag_cached
                    && self.text_state.editing_text_layer
                {
                    if let Some(origin) = self.text_state.origin
                        && let Some(layer) =
                            canvas_state.layers.get_mut(canvas_state.active_layer_index)
                        && let crate::canvas::LayerContent::Text(ref mut td) = layer.content
                        && let Some(bid) = self.text_state.active_block_id
                        && let Some(block) = td.blocks.iter_mut().find(|b| b.id == bid)
                    {
                        block.position = [origin[0], origin[1]];
                        td.mark_position_dirty();
                    }
                    let idx = canvas_state.active_layer_index;
                    canvas_state.force_rasterize_text_layer(idx);
                    canvas_state.preview_layer = None;
                    canvas_state.mark_dirty(None);
                    self.text_state.text_layer_drag_cached = false;
                }
                self.text_state.dragging_handle = false;
                // Finish glyph drag
                if self.text_state.glyph_drag.is_some() {
                    self.text_state.glyph_drag = None;
                }
            }

            // Click to place origin (or commit existing + start new)
            // Only if not dragging the handle and not finishing a drag-to-select
            let any_popup_open = egui::Popup::is_any_open(ui.ctx());
            if is_primary_clicked
                && !self.text_state.dragging_handle
                && !self.text_state.text_box_click_guard
                && !self.text_state.text_select_dragging
                && !any_popup_open
            {
                // Check if click is on the handle - if so, skip placement
                let on_handle =
                    if let (Some(pos_f), Some(hp)) = (canvas_pos_unclamped, handle_canvas_pos) {
                        let dx = (pos_f.0 - hp.0) * zoom;
                        let dy = (pos_f.1 - hp.1) * zoom;
                        (dx * dx + dy * dy).sqrt() < handle_radius_screen + 4.0
                    } else {
                        false
                    };

                if !on_handle && let Some(pos_f) = canvas_pos_f32 {
                    if self.text_state.is_editing && !self.text_state.text.is_empty() {
                        if self.text_state.editing_text_layer {
                            // Text layer mode: check if clicking on the same vs different block
                            let is_text_layer = canvas_state
                                .layers
                                .get(canvas_state.active_layer_index)
                                .is_some_and(|l| l.is_text_layer());
                            if is_text_layer {
                                // Hit-test to find which block the click is on
                                let clicked_block_id = if let Some(layer) =
                                    canvas_state.layers.get(canvas_state.active_layer_index)
                                    && let crate::canvas::LayerContent::Text(ref td) = layer.content
                                {
                                    crate::ops::text_layer::hit_test_blocks(td, pos_f.0, pos_f.1)
                                        .map(|idx| td.blocks[idx].id)
                                } else {
                                    None
                                };

                                let same_block = self.text_state.active_block_id.is_some()
                                    && clicked_block_id == self.text_state.active_block_id;

                                if same_block {
                                    // Click in the same block - move cursor, don't commit
                                    // Use cached line advances and preview origin for position->cursor mapping
                                    if let Some(origin) = self.text_state.origin {
                                        let rel_x = pos_f.0 - origin[0];
                                        let rel_y = pos_f.1 - origin[1];
                                        let lh = self.text_state.cached_line_height.max(1.0);

                                        // Compute visual lines (word-wrapped) for correct mapping
                                        let visual_line_count = if let (Some(mw), Some(font)) = (
                                            self.text_state.active_block_max_width,
                                            &self.text_state.loaded_font,
                                        ) {
                                            let ls = self.text_state.letter_spacing;
                                            let vlines: Vec<String> = self
                                                .text_state
                                                .text
                                                .split('\n')
                                                .flat_map(|line| {
                                                    crate::ops::text::word_wrap_line(
                                                        line,
                                                        font,
                                                        self.text_state.font_size,
                                                        mw,
                                                        ls,
                                                        self.text_state.width_scale,
                                                        self.text_state.height_scale,
                                                    )
                                                })
                                                .collect();
                                            vlines.len()
                                        } else {
                                            self.text_state.text.split('\n').count()
                                        };
                                        let line_idx = ((rel_y / lh).floor() as usize)
                                            .min(visual_line_count.saturating_sub(1));

                                        // Find approximate char position within visual line using cached_line_advances
                                        let mut best_pos = 0;
                                        if !self.text_state.cached_line_advances.is_empty()
                                            && line_idx < self.text_state.cached_line_advances.len()
                                        {
                                            let advances =
                                                &self.text_state.cached_line_advances[line_idx];
                                            let mut best_dist = f32::MAX;
                                            for (ci, &adv) in advances.iter().enumerate() {
                                                let dist = (adv - rel_x).abs();
                                                if dist < best_dist {
                                                    best_dist = dist;
                                                    best_pos = ci;
                                                }
                                            }
                                            // Clamp to visual line char count
                                            let max_chars = advances.len().saturating_sub(1);
                                            best_pos = best_pos.min(max_chars);
                                        }

                                        // Convert (visual_line, char_in_line) back to byte offset
                                        let byte_pos = if let (Some(mw), Some(font)) = (
                                            self.text_state.active_block_max_width,
                                            &self.text_state.loaded_font,
                                        ) {
                                            let ls = self.text_state.letter_spacing;
                                            Self::visual_to_byte_pos(
                                                &self.text_state.text,
                                                line_idx,
                                                best_pos,
                                                font,
                                                self.text_state.font_size,
                                                mw,
                                                ls,
                                                self.text_state.width_scale,
                                                self.text_state.height_scale,
                                            )
                                        } else {
                                            // No wrapping - use logical line mapping
                                            let lines: Vec<&str> =
                                                self.text_state.text.split('\n').collect();
                                            let clamped =
                                                line_idx.min(lines.len().saturating_sub(1));
                                            let line_start: usize = lines
                                                .iter()
                                                .take(clamped)
                                                .map(|l| l.len() + 1)
                                                .sum();
                                            let line_text = lines[clamped];
                                            let clamped_pos =
                                                best_pos.min(line_text.chars().count());
                                            let byte_offset: usize = line_text
                                                .chars()
                                                .take(clamped_pos)
                                                .map(|c| c.len_utf8())
                                                .sum();
                                            line_start + byte_offset
                                        };
                                        self.text_state.cursor_pos = byte_pos;
                                        // Update selection
                                        let shift_held = ui.input(|i| i.modifiers.shift);
                                        self.text_layer_update_selection(canvas_state, shift_held);
                                        self.text_state.preview_dirty = true;
                                    }
                                } else {
                                    // Different block or empty area - commit and load new
                                    self.stroke_tracker.start_preview_tool(
                                        canvas_state.active_layer_index,
                                        "Text",
                                    );
                                    self.commit_text(canvas_state);
                                    if self.pending_stroke_event.is_none()
                                        && let Some(evt) = stroke_event.take()
                                    {
                                        self.pending_stroke_event = Some(evt);
                                    }
                                    // Load block at click position (or create new)
                                    self.load_text_layer_block(
                                        canvas_state,
                                        None,
                                        Some([pos_f.0, pos_f.1]),
                                    );
                                }
                                // Skip the default click-to-place behavior
                            } else {
                                // Commit raster text and start new
                                self.stroke_tracker
                                    .start_preview_tool(canvas_state.active_layer_index, "Text");
                                self.commit_text(canvas_state);
                                if self.pending_stroke_event.is_none()
                                    && let Some(evt) = stroke_event.take()
                                {
                                    self.pending_stroke_event = Some(evt);
                                }
                                self.text_state.origin = Some([pos_f.0, pos_f.1]);
                                self.text_state.is_editing = true;
                                self.text_state.editing_text_layer = false;
                                self.text_state.editing_layer_index =
                                    Some(canvas_state.active_layer_index);
                                self.text_state.active_block_id = None;
                                self.text_state.text.clear();
                                self.clear_text_preview_cache(canvas_state);
                                self.text_state.cursor_pos = 0;
                                self.text_state.preview_dirty = true;
                                self.restore_raster_style();
                                self.stroke_tracker
                                    .start_preview_tool(canvas_state.active_layer_index, "Text");
                            }
                        } else {
                            // Raster text mode: commit current text first
                            self.stroke_tracker
                                .start_preview_tool(canvas_state.active_layer_index, "Text");
                            self.commit_text(canvas_state);
                            if self.pending_stroke_event.is_none()
                                && let Some(evt) = stroke_event.take()
                            {
                                self.pending_stroke_event = Some(evt);
                            }
                            self.text_state.origin = Some([pos_f.0, pos_f.1]);
                            self.text_state.is_editing = true;
                            self.text_state.editing_text_layer = false;
                            self.text_state.editing_layer_index =
                                Some(canvas_state.active_layer_index);
                            self.text_state.active_block_id = None;
                            self.text_state.text.clear();
                            self.clear_text_preview_cache(canvas_state);
                            self.text_state.cursor_pos = 0;
                            self.text_state.preview_dirty = true;
                            self.restore_raster_style();
                            self.stroke_tracker
                                .start_preview_tool(canvas_state.active_layer_index, "Text");
                        }
                    } else if self.text_state.is_editing && self.text_state.text.is_empty() {
                        // Already editing but empty - for text layers, switch block
                        if self.text_state.editing_text_layer {
                            self.commit_text(canvas_state);
                            if self.pending_stroke_event.is_none()
                                && let Some(evt) = stroke_event.take()
                            {
                                self.pending_stroke_event = Some(evt);
                            }
                            self.load_text_layer_block(
                                canvas_state,
                                None,
                                Some([pos_f.0, pos_f.1]),
                            );
                        } else {
                            // Move the empty text origin
                            self.text_state.origin = Some([pos_f.0, pos_f.1]);
                            self.text_state.preview_dirty = true;
                        }
                    } else {
                        // Check if active layer is a text layer
                        let is_text_layer = canvas_state
                            .layers
                            .get(canvas_state.active_layer_index)
                            .is_some_and(|l| l.is_text_layer());

                        if is_text_layer {
                            // Load text layer data with hit-test at click position
                            self.load_text_layer_block(
                                canvas_state,
                                None,
                                Some([pos_f.0, pos_f.1]),
                            );
                        } else {
                            // Start new text at this position (raster stamp mode)
                            self.text_state.origin = Some([pos_f.0, pos_f.1]);
                            self.text_state.is_editing = true;
                            self.text_state.editing_text_layer = false;
                            self.text_state.editing_layer_index =
                                Some(canvas_state.active_layer_index);
                            self.text_state.active_block_id = None;
                            self.text_state.text.clear();
                            self.clear_text_preview_cache(canvas_state);
                            self.text_state.cursor_pos = 0;
                            self.text_state.preview_dirty = true;
                            self.restore_raster_style();
                            self.stroke_tracker
                                .start_preview_tool(canvas_state.active_layer_index, "Text");
                        }
                    }
                }
            }

            // Clear the click guard and drag-to-select on mouse release (after the click handler above)
            if is_primary_released {
                self.text_state.text_box_click_guard = false;
                self.text_state.text_select_dragging = false;
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

            // Capture text input - but only when the canvas widget has
            // focus.  When a DragValue or other UI widget is focused the
            // same keystrokes would otherwise leak into the text layer.
            let canvas_has_focus = ui.ctx().memory(|m| {
                match m.focused() {
                    Some(id) => canvas_state.canvas_widget_id == Some(id),
                    None => true, // no widget focused - canvas owns input
                }
            });
            let allow_text_capture = !self.text_state.font_popup_open
                && (canvas_has_focus || !ui.ctx().egui_wants_keyboard_input());
            if self.text_state.is_editing && allow_text_capture {
                // Tell egui's web backend where the caret is so it can
                // move/focus the hidden `<input>` element that bridges
                // real browser keyboard/IME events into `Event::Text`.
                // Without this, typed characters never reach `events`
                // below on web (native routes keys directly and doesn't
                // need it).
                if let Some(origin) = self.text_state.origin {
                    let font_size = self.text_state.font_size.max(8.0);
                    let screen_x = canvas_rect.min.x + origin[0] * zoom;
                    let screen_y = canvas_rect.min.y + origin[1] * zoom;
                    let ime_rect = egui::Rect::from_min_size(
                        egui::pos2(screen_x, screen_y),
                        egui::vec2((font_size * zoom).max(1.0), (font_size * zoom).max(1.0)),
                    );
                    ui.ctx().output_mut(|o| {
                        o.ime = Some(egui::output::IMEOutput {
                            rect: ime_rect,
                            cursor_rect: ime_rect,
                            should_interrupt_composition: false,
                        });
                    });
                }

                let events: Vec<egui::Event> = ui.input(|i| i.events.clone());
                let shift_held = ui.input(|i| i.modifiers.shift);
                let ctrl_held = ui.input(|i| i.modifiers.command);

                // Ctrl+B / Ctrl+I / Ctrl+U formatting shortcuts (text layer only)
                if self.text_state.editing_text_layer && ctrl_held {
                    let b_pressed = ui.input(|i| i.key_pressed(egui::Key::B));
                    let i_pressed = ui.input(|i| i.key_pressed(egui::Key::I));
                    let u_pressed = ui.input(|i| i.key_pressed(egui::Key::U));
                    let a_pressed = ui.input(|i| i.key_pressed(egui::Key::A));

                    if b_pressed {
                        self.text_layer_toggle_style(canvas_state, |s| {
                            s.font_weight = if s.font_weight >= 700 { 400 } else { 700 }
                        });
                    }
                    if i_pressed {
                        self.text_layer_toggle_style(canvas_state, |s| s.italic = !s.italic);
                    }
                    if u_pressed {
                        self.text_layer_toggle_style(canvas_state, |s| s.underline = !s.underline);
                    }
                    if a_pressed {
                        // Select all text in active block
                        self.text_state.selection.anchor = crate::ops::text_layer::RunPosition {
                            run_index: 0,
                            byte_offset: 0,
                        };
                        let len = self.text_state.text.len();
                        self.text_state.cursor_pos = len;
                        // Get run position for end of text
                        if let Some(layer) =
                            canvas_state.layers.get(canvas_state.active_layer_index)
                            && let crate::canvas::LayerContent::Text(ref td) = layer.content
                            && let Some(bid) = self.text_state.active_block_id
                            && let Some(block) = td.blocks.iter().find(|b| b.id == bid)
                        {
                            self.text_state.selection.cursor = block.flat_offset_to_run_pos(len);
                        } else {
                            self.text_state.selection.cursor =
                                crate::ops::text_layer::RunPosition {
                                    run_index: 0,
                                    byte_offset: len,
                                };
                        }
                        self.sync_text_toolbar_to_selection(canvas_state);
                    }
                }

                // Tab: cycle between blocks (text layer only)
                if self.text_state.editing_text_layer {
                    let tab_pressed = ui.input(|i| i.key_pressed(egui::Key::Tab));
                    if tab_pressed {
                        self.text_layer_cycle_block(canvas_state, shift_held);
                    }
                }

                for event in &events {
                    match event {
                        egui::Event::Text(t) => {
                            if self.text_state.editing_text_layer {
                                self.text_layer_insert_text(canvas_state, t);
                            } else {
                                // Delete selection first if any
                                self.text_state
                                    .text
                                    .insert_str(self.text_state.cursor_pos, t);
                                self.text_state.cursor_pos += t.len();
                                self.text_state.preview_dirty = true;
                            }
                        }
                        egui::Event::Key {
                            key: egui::Key::Backspace,
                            pressed: true,
                            ..
                        } => {
                            if self.text_state.editing_text_layer {
                                self.text_layer_backspace(canvas_state);
                            } else if self.text_state.cursor_pos > 0 {
                                // Remove one char before cursor
                                let mut chars: Vec<char> = self.text_state.text.chars().collect();
                                let char_idx = self.text_state.text[..self.text_state.cursor_pos]
                                    .chars()
                                    .count();
                                if char_idx > 0 {
                                    let remove_idx = char_idx - 1;
                                    chars.remove(remove_idx);
                                    self.text_state.text = chars.into_iter().collect();
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
                            // Selection: update or collapse
                            if self.text_state.editing_text_layer {
                                self.text_layer_update_selection(canvas_state, shift_held);
                            }
                        }
                        egui::Event::Key {
                            key: egui::Key::ArrowRight,
                            pressed: true,
                            ..
                        } => {
                            if self.text_state.cursor_pos < self.text_state.text.len()
                                && let Some(c) = self.text_state.text[self.text_state.cursor_pos..]
                                    .chars()
                                    .next()
                            {
                                self.text_state.cursor_pos += c.len_utf8();
                            }
                            if self.text_state.editing_text_layer {
                                self.text_layer_update_selection(canvas_state, shift_held);
                            }
                        }
                        egui::Event::Key {
                            key: egui::Key::ArrowUp,
                            pressed: true,
                            ..
                        } => {
                            // Move cursor to same x-position on the previous visual line
                            self.text_move_cursor_vertical(-1, shift_held, canvas_state);
                        }
                        egui::Event::Key {
                            key: egui::Key::ArrowDown,
                            pressed: true,
                            ..
                        } => {
                            // Move cursor to same x-position on the next visual line
                            self.text_move_cursor_vertical(1, shift_held, canvas_state);
                        }
                        egui::Event::Key {
                            key: egui::Key::Delete,
                            pressed: true,
                            ..
                        } => {
                            if self.text_state.editing_text_layer {
                                self.text_layer_delete(canvas_state);
                            } else if self.text_state.cursor_pos < self.text_state.text.len() {
                                let mut chars: Vec<char> = self.text_state.text.chars().collect();
                                let char_idx = self.text_state.text[..self.text_state.cursor_pos]
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
                            let text_before = &self.text_state.text[..self.text_state.cursor_pos];
                            if let Some(nl) = text_before.rfind('\n') {
                                self.text_state.cursor_pos = nl + 1;
                            } else {
                                self.text_state.cursor_pos = 0;
                            }
                            if self.text_state.editing_text_layer {
                                self.text_layer_update_selection(canvas_state, shift_held);
                            }
                        }
                        egui::Event::Key {
                            key: egui::Key::End,
                            pressed: true,
                            ..
                        } => {
                            let text_after = &self.text_state.text[self.text_state.cursor_pos..];
                            if let Some(nl) = text_after.find('\n') {
                                self.text_state.cursor_pos += nl;
                            } else {
                                self.text_state.cursor_pos = self.text_state.text.len();
                            }
                            if self.text_state.editing_text_layer {
                                self.text_layer_update_selection(canvas_state, shift_held);
                            }
                        }
                        egui::Event::Paste(t) => {
                            let filtered: String = t
                                .chars()
                                .filter(|c| !c.is_control() || *c == '\n')
                                .collect();
                            if !filtered.is_empty() {
                                if self.text_state.editing_text_layer {
                                    self.text_layer_insert_text(canvas_state, &filtered);
                                } else {
                                    self.text_state
                                        .text
                                        .insert_str(self.text_state.cursor_pos, &filtered);
                                    self.text_state.cursor_pos += filtered.len();
                                    self.text_state.preview_dirty = true;
                                }
                            }
                        }
                        _ => {}
                    }
                }

                // Re-render preview if dirty
                if self.text_state.preview_dirty {
                    self.render_text_preview(canvas_state, primary_color_f32);
                }

                // Text overlay drawn by canvas.rs after handle_input returns
            }
        }
    }
}
