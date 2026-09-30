impl ToolsPanel {
    #[allow(clippy::too_many_arguments, unused_variables)]
    fn handle_stroke_tools_input(
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
        match self.active_tool {
            Tool::Brush | Tool::Eraser | Tool::Pencil => {
                // Guard: auto-rasterize text layers before destructive drawing
                if (is_primary_pressed || is_secondary_down)
                    && let Some(layer) = canvas_state.layers.get(canvas_state.active_layer_index)
                    && layer.is_text_layer()
                {
                    self.pending_auto_rasterize = Some(canvas_state.active_layer_index);
                    // Return early - app.rs will rasterize and we'll continue next frame
                    return;
                }

                let is_painting = is_primary_down || is_secondary_down;
                let editing_mask = canvas_state.edit_layer_mask
                    && canvas_state
                        .layers
                        .get(canvas_state.active_layer_index)
                        .is_some_and(|l| l.has_live_mask());
                let resize_modifier_held =
                    self.active_tool == Tool::Brush && self.brush_resize_drag_modifier_held(ui);
                let resize_drag_threshold = 4.0;

                if self.active_tool == Tool::Brush
                    && is_primary_pressed
                    && resize_modifier_held
                    && let Some((x, y)) = canvas_pos_unclamped
                        .or(canvas_pos_f32)
                        .or(canvas_pos.map(|(x, y)| (x as f32, y as f32)))
                {
                    self.tool_state.brush_resize_drag_origin = Some(Pos2::new(x, y));
                    self.tool_state.brush_resize_drag_start_size = self.properties.size;
                    self.tool_state.brush_resize_drag_active = false;
                    self.reset_brush_pointer_state();
                    return;
                }

                if self.active_tool == Tool::Brush
                    && let Some(origin) = self.tool_state.brush_resize_drag_origin
                {
                    let current_drag_pos = canvas_pos_unclamped
                        .or(canvas_pos_f32)
                        .or(canvas_pos.map(|(x, y)| (x as f32, y as f32)));
                    if is_primary_down {
                        if let Some((x, y)) = current_drag_pos {
                            let current = Pos2::new(x, y);
                            let delta = current - origin;
                            if self.tool_state.brush_resize_drag_active
                                || delta.length() >= resize_drag_threshold
                            {
                                self.tool_state.brush_resize_drag_active = true;
                                self.properties.size =
                                    (self.tool_state.brush_resize_drag_start_size + delta.x)
                                        .clamp(1.0, 500.0);
                                self.reset_brush_pointer_state();
                                ui.ctx().request_repaint();
                            }
                        }
                        return;
                    }
                    if is_primary_released {
                        let was_active = self.tool_state.brush_resize_drag_active;
                        self.tool_state.brush_resize_drag_origin = None;
                        self.tool_state.brush_resize_drag_active = false;
                        if was_active {
                            return;
                        }
                        if resize_modifier_held
                            && self.tool_state.last_brush_pos.is_some()
                            && let (Some(last), Some(current)) =
                                (self.tool_state.last_brush_pos, canvas_pos)
                        {
                            *stroke_event = self.commit_brush_straight_line(
                                canvas_state,
                                last,
                                current,
                                primary_color_f32,
                                secondary_color_f32,
                            );
                            if stroke_event.is_some() {
                                self.pending_stroke_event = stroke_event.take();
                            }
                            self.tool_state.last_brush_pos = Some(current);
                            self.tool_state.last_precise_pos =
                                Some(Pos2::new(current.0 as f32, current.1 as f32));
                            ui.ctx().request_repaint();
                            return;
                        }
                    }
                }

                // TASK 2: Shift+Click straight line
                // Trigger when mouse is pressed (not dragged) with Shift held
                if is_primary_pressed
                    && shift_held
                    && !resize_modifier_held
                    && self.tool_state.last_brush_pos.is_some()
                    && let (Some(last), Some(current)) =
                        (self.tool_state.last_brush_pos, canvas_pos)
                {
                    *stroke_event = self.commit_brush_straight_line(
                        canvas_state,
                        last,
                        current,
                        primary_color_f32,
                        secondary_color_f32,
                    );

                    // Store stroke event for app.rs to pick up (before returning)
                    if stroke_event.is_some() {
                        self.pending_stroke_event = stroke_event.take();
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
                    // Also allow off-canvas start: use raw unclamped coords so the
                    // stroke tracks the real pointer position. Draw functions clip
                    // to canvas bounds internally — off-canvas positions produce
                    // no pixels but maintain natural stroke flow onto the canvas.
                    let current_f32 = canvas_pos_f32.or(
                        canvas_pos_unclamped
                    ).or_else(|| canvas_pos.map(|(x, y)| (x as f32, y as f32)));
                    let current_pos = canvas_pos.or_else(|| {
                        canvas_pos_unclamped.map(|(x, y)| (x as u32, y as u32))
                    });
                    if let (Some(cf), Some(current_pos)) = (current_f32, current_pos) {
                        // Initialize on first paint
                        if self.tool_state.last_pos.is_none() {
                            self.tool_state.using_secondary_color = is_secondary_down;
                            self.tool_state.last_precise_pos = Some(Pos2::new(cf.0, cf.1));
                            self.tool_state.distance_remainder = 0.0;
                            self.tool_state.smooth_pos = Some(Pos2::new(cf.0, cf.1));

                            // Start stroke tracking for Undo/Redo
                            let is_eraser = self.active_tool == Tool::Eraser;
                            if editing_mask {
                                let layer_idx = canvas_state.active_layer_index;
                                let (before_mask, before_enabled) = canvas_state
                                    .layers
                                    .get(layer_idx)
                                    .map(|l| (l.mask.clone(), l.mask_enabled))
                                    .unwrap_or((None, true));
                                self.stroke_tracker.start_preview_mask_tool(
                                    layer_idx,
                                    if is_eraser {
                                        "Layer Mask Erase"
                                    } else {
                                        "Layer Mask Paint"
                                    },
                                    before_mask,
                                    before_enabled,
                                );
                            } else if is_eraser {
                                // Eraser uses preview layer as an erase mask - capture before at commit time
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
                                canvas_state.preview_targets_mask = false;
                                canvas_state.preview_mask_reveal = false;
                            } else {
                                // Set preview blend mode to match tool's blending mode
                                canvas_state.preview_blend_mode = self.properties.blending_mode;
                                canvas_state.preview_targets_mask = false;
                                canvas_state.preview_mask_reveal = false;
                            }

                            if editing_mask {
                                // In mask edit mode, preview alpha represents conceal/reveal delta.
                                canvas_state.preview_force_composite = true;
                                canvas_state.preview_targets_mask = true;
                                canvas_state.preview_mask_reveal = is_eraser;
                                canvas_state.preview_is_eraser = false;
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

                        // Stroke stabilization (EMA smoothing of raw pointer
                        // samples; 0 = raw, Pencil always raw).
                        let smoothed_positions = self.stabilize_positions(&positions);

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
                                // CPU-only paint step: lerp from last_precise_pos -> pos
                                let modified_rect = if self.active_tool == Tool::Pencil {
                                    if let Some(start_p) = start_precise {
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
                                        self.draw_pixel_line_and_get_bounds(
                                            canvas_state,
                                            start_m,
                                            mpos,
                                            use_secondary,
                                            primary_color_f32,
                                            secondary_color_f32,
                                        )
                                    } else {
                                        self.draw_pixel_and_get_bounds(
                                            canvas_state,
                                            mpos,
                                            use_secondary,
                                            primary_color_f32,
                                            secondary_color_f32,
                                        )
                                    }
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
                            if editing_mask {
                                // Keep edits in preview during drag; commit once on release.
                                // This avoids per-frame mask writes + full masked-layer uploads.
                                canvas_state.mark_preview_changed_rect(frame_dirty_rect);
                            } else {
                                // All tools (Brush, Pencil, Eraser) use the preview path now
                                canvas_state.mark_preview_changed_rect(frame_dirty_rect);
                            }
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
                        *stroke_event = self.stroke_tracker.finish(canvas_state);

                        if editing_mask {
                            // Commit mask once at stroke end for smooth interactive dragging.
                            self.commit_preview_to_layer_mask(canvas_state, is_eraser);
                            canvas_state.defer_preview_clear();
                        } else if is_eraser {
                            // Commit the eraser mask to the active layer
                            self.commit_eraser_to_layer(canvas_state);
                        } else {
                            // Commit the preview layer to the active layer (Brush/Pencil)
                            self.commit_bezier_to_layer(canvas_state, primary_color_f32);
                        }
                        // Clear preview layer
                        canvas_state.defer_preview_clear();
                        // Mark only stroke bounds dirty (not full canvas)
                        if let Some(ev) = stroke_event.as_ref() {
                            let dirty = ev.bounds.expand(12.0);
                            canvas_state.request_commit_composite_flush(dirty);
                            canvas_state.mark_dirty(Some(dirty));
                        } else {
                            self.mark_full_dirty(canvas_state);
                        }
                    }

                    self.reset_brush_pointer_state();
                }
            }
            Tool::Line => {
                let line_editing_mask = canvas_state.edit_layer_mask
                    && canvas_state
                        .layers
                        .get(canvas_state.active_layer_index)
                        .is_some_and(|l| l.has_live_mask());

                // Guard: auto-rasterize text layers before destructive line tool
                if is_primary_pressed
                    && let Some(layer) = canvas_state.layers.get(canvas_state.active_layer_index)
                    && layer.is_text_layer()
                {
                    self.pending_auto_rasterize = Some(canvas_state.active_layer_index);
                    return;
                }

                // TASK 1: Advanced Bezier Line Tool
                match self.line_state.line_tool.stage {
                    LineStage::Idle => {
                        // Clear the require_mouse_release flag when mouse is released
                        if is_primary_released {
                            self.line_state.line_tool.require_mouse_release = false;
                        }

                        // Allow off-canvas start: use raw unclamped f32 coords so the line
                        // tracks the real pointer. Rasterization clips to canvas bounds.
                        // Use f32 directly — u32 cast breaks for negative (left/top off-canvas).
                        let line_pos_f32 = canvas_pos_f32.or(
                            canvas_pos_unclamped
                        );

                        if !self.line_state.line_tool.require_mouse_release {
                            // Normal behavior: start line on click (or click and drag)
                            if is_primary_pressed && let Some(pos) = line_pos_f32 {
                                let pos2 = Pos2::new(pos.0, pos.1);
                                self.line_state.line_tool.control_points = [pos2; 4];
                                self.line_state.line_tool.stage = LineStage::Dragging;
                                self.line_state.line_tool.last_bounds = None;
                            }
                        } else {
                            // After committing a line: require drag or release+click
                            if is_primary_pressed && let Some(pos) = line_pos_f32 {
                                // Store initial position to detect drag
                                let pos2 = Pos2::new(pos.0, pos.1);
                                self.line_state.line_tool.initial_mouse_pos = Some(pos2);
                                self.line_state.line_tool.control_points = [pos2; 4];
                            }

                            // Transition to Dragging only if the mouse moved while pressed
                            if is_primary_down
                                && self.line_state.line_tool.initial_mouse_pos.is_some()
                                && let Some(pos) = line_pos_f32
                            {
                                let current_pos = Pos2::new(pos.0, pos.1);
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
                        // Allow off-canvas drag: use raw unclamped f32 coords.
                        let drag_pos_f32 = canvas_pos_f32.or(
                            canvas_pos_unclamped
                        );
                        if let Some(pos) = drag_pos_f32 {
                            let raw_pos = Pos2::new(pos.0, pos.1);
                            let p0 = self.line_state.line_tool.control_points[0];
                            // Snap to nearest 45┬░ angle when Shift is held
                            let pos2 = if shift_held {
                                let dx = raw_pos.x - p0.x;
                                let dy = raw_pos.y - p0.y;
                                let angle = dy.atan2(dx);
                                let snap = std::f32::consts::PI / 4.0; // 45┬░
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
                            self.line_state.line_tool.last_end_shape =
                                self.line_state.line_tool.options.end_shape;
                            self.line_state.line_tool.last_arrow_side =
                                self.line_state.line_tool.options.arrow_side;
                            self.line_state.line_tool.last_anti_alias =
                                self.line_state.line_tool.options.anti_alias;

                            // Start stroke tracking (line uses preview layer like Brush)
                            if line_editing_mask {
                                let layer_idx = canvas_state.active_layer_index;
                                let (before_mask, before_enabled) = canvas_state
                                    .layers
                                    .get(layer_idx)
                                    .map(|l| (l.mask.clone(), l.mask_enabled))
                                    .unwrap_or((None, true));
                                self.stroke_tracker.start_preview_mask_tool(
                                    layer_idx,
                                    "Line Mask",
                                    before_mask,
                                    before_enabled,
                                );
                            } else {
                                self.stroke_tracker
                                    .start_preview_tool(canvas_state.active_layer_index, "Line");
                            }
                            if let Some(bounds) = self.line_state.line_tool.last_bounds {
                                self.stroke_tracker.expand_bounds(bounds);
                            }
                        }
                    }
                    LineStage::Editing => {
                        let mouse_pos = ui.input(|i| {
                            i.pointer.interact_pos().or_else(|| {
                                i.events.iter().rev().find_map(|e| match e {
                                    egui::Event::PointerButton { pressed: true, pos, .. } => Some(*pos),
                                    egui::Event::Touch {
                                        phase: egui::TouchPhase::Start | egui::TouchPhase::Move,
                                        pos,
                                        ..
                                    } => Some(*pos),
                                    _ => None,
                                })
                            })
                        });

                        // Check if settings changed and re-render if needed
                        let settings_changed =
                            (self.properties.size - self.line_state.line_tool.last_size).abs()
                                > 0.1
                                || self.line_state.line_tool.options.pattern
                                    != self.line_state.line_tool.last_pattern
                                || self.line_state.line_tool.options.cap_style
                                    != self.line_state.line_tool.last_cap_style
                                || self.line_state.line_tool.options.end_shape
                                    != self.line_state.line_tool.last_end_shape
                                || self.line_state.line_tool.options.arrow_side
                                    != self.line_state.line_tool.last_arrow_side
                                || self.line_state.line_tool.options.anti_alias
                                    != self.line_state.line_tool.last_anti_alias;

                        if settings_changed {
                            // Update tracked values
                            self.line_state.line_tool.last_size = self.properties.size;
                            self.line_state.line_tool.last_pattern =
                                self.line_state.line_tool.options.pattern;
                            self.line_state.line_tool.last_cap_style =
                                self.line_state.line_tool.options.cap_style;
                            self.line_state.line_tool.last_end_shape =
                                self.line_state.line_tool.options.end_shape;
                            self.line_state.line_tool.last_arrow_side =
                                self.line_state.line_tool.options.arrow_side;
                            self.line_state.line_tool.last_anti_alias =
                                self.line_state.line_tool.options.anti_alias;

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
                            *stroke_event = self.stroke_tracker.finish(canvas_state);

                            // Commit the line to the actual layer
                            let mirror_bounds = canvas_state.mirror_preview_layer();
                            if line_editing_mask {
                                self.commit_preview_to_layer_mask(canvas_state, false);
                            } else {
                                self.commit_bezier_to_layer(canvas_state, secondary_color_f32);
                            }

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

                            canvas_state.defer_preview_clear();
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
                                        *stroke_event = self.stroke_tracker.finish(canvas_state);

                                        // Commit the current line
                                        let mirror_bounds = canvas_state.mirror_preview_layer();
                                        if line_editing_mask {
                                            self.commit_preview_to_layer_mask(canvas_state, false);
                                        } else {
                                            self.commit_bezier_to_layer(
                                                canvas_state,
                                                secondary_color_f32,
                                            );
                                        }

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

                                        canvas_state.defer_preview_clear();
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
                                            pt.x += delta.x;
                                            pt.y += delta.y;
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
                                // Allow handles off-canvas — rasterizer clips to bounds.
                                self.line_state.line_tool.control_points[handle_idx] = canvas_pos_float;

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

            _ => {}
        }
    }
}

