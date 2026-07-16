impl ToolsPanel {
    #[allow(clippy::too_many_arguments, unused_variables)]
    fn handle_utility_navigation_input(
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
            // CLONE STAMP - Alt+click to set source, then paint from offset
            // ================================================================
            Tool::CloneStamp => {
                // Guard: auto-rasterize text layers before destructive clone stamp
                if (is_primary_pressed || is_secondary_down)
                    && let Some(layer) = canvas_state.layers.get(canvas_state.active_layer_index)
                    && layer.is_text_layer()
                {
                    self.pending_auto_rasterize = Some(canvas_state.active_layer_index);
                    return;
                }

                let is_painting = is_primary_down || is_secondary_down;
                let alt_held = ui.input(|i| i.modifiers.alt);

                // Enter/Escape: clear source point
                if enter_pressed || escape_pressed_global {
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
                        // Mouse released - commit
                        if self.tool_state.last_pos.is_some() {
                            *stroke_event = self.stroke_tracker.finish(canvas_state);
                            self.commit_bezier_to_layer(canvas_state, primary_color_f32);
                            canvas_state.clear_preview_state();
                            if let Some(ev) = stroke_event.as_ref() {
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
            // CONTENT AWARE BRUSH - healing brush, samples surrounding texture
            // ================================================================
            Tool::ContentAwareBrush => {
                // Auto-rasterize guard: if active layer is a text layer, rasterize first
                if (is_primary_pressed || is_secondary_down)
                    && canvas_state.active_layer_index < canvas_state.layers.len()
                    && canvas_state.layers[canvas_state.active_layer_index].is_text_layer()
                {
                    self.pending_auto_rasterize = Some(canvas_state.active_layer_index);
                    return;
                }

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
                    // Mouse released - commit
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

                        *stroke_event = self.stroke_tracker.finish(canvas_state);
                        self.commit_bezier_to_layer(canvas_state, primary_color_f32);
                        canvas_state.clear_preview_state();
                        if let Some(ev) = stroke_event.as_ref() {
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
            // LASSO SELECT - freeform polygon selection
            // ================================================================
            Tool::Lasso => {
                let esc_pressed = escape_pressed_global;
                let alt_held_l = ui.input(|i| i.modifiers.alt);
                let is_secondary_pressed =
                    ui.input(|i| i.pointer.button_pressed(egui::PointerButton::Secondary));

                // Esc / Enter: clear selection and cancel any in-progress lasso path.
                if esc_pressed || enter_pressed {
                    if self.lasso_state.dragging {
                        self.lasso_state.dragging = false;
                        self.lasso_state.points.clear();
                    }
                    canvas_state.clear_selection();
                    canvas_state.mark_dirty(None);
                    ui.ctx().request_repaint();
                }

                // Start lasso drag - lock effective mode from modifier keys at drag start
                let ctrl_held_l = ui.input(|i| i.modifiers.command);
                if (is_primary_pressed || is_secondary_pressed)
                    && !self.lasso_state.dragging
                    && let Some(pos_f) = canvas_pos_unclamped
                {
                    self.lasso_state.dragging = true;
                    self.lasso_state.right_click_drag = is_secondary_pressed;
                    self.lasso_state.drag_effective_mode = if is_secondary_pressed {
                        SelectionMode::Subtract
                    } else if shift_held && alt_held_l {
                        SelectionMode::Intersect
                    } else if ctrl_held_l {
                        SelectionMode::Add
                    } else if alt_held_l {
                        SelectionMode::Subtract
                    } else {
                        self.lasso_state.mode
                    };
                    self.lasso_state.points.clear();
                    self.lasso_state.points.push(Pos2::new(pos_f.0, pos_f.1));
                }

                // Accumulate points while dragging
                let any_button_down = is_primary_down || is_secondary_down;
                if any_button_down && self.lasso_state.dragging {
                    if let Some(pos_f) = canvas_pos_unclamped {
                        let p = Pos2::new(pos_f.0, pos_f.1);
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

                // Finish on release - rasterize polygon into selection mask
                let any_button_released = is_primary_released || is_secondary_released;
                if any_button_released && self.lasso_state.dragging {
                    let effective_mode = self.lasso_state.drag_effective_mode;
                    self.lasso_state.dragging = false;
                    let pts = std::mem::take(&mut self.lasso_state.points);
                    let sel_before = canvas_state.selection_mask.clone();
                    let sel_before_all = canvas_state.selection_all;
                    if pts.len() >= 3 {
                        // Scanline-fill the polygon into the selection mask
                        Self::apply_lasso_selection(canvas_state, &pts, effective_mode);
                    } else {
                        // Tiny lasso -> deselect
                        canvas_state.clear_selection();
                    }
                    let sel_after = canvas_state.selection_mask.clone();
                    let sel_after_all = canvas_state.selection_all;
                    self.pending_history_commands
                        .push(Box::new(SelectionCommand::new_states(
                            "Lasso Select",
                            sel_before,
                            sel_before_all,
                            sel_after,
                            sel_after_all,
                        )));
                    ui.ctx().request_repaint();
                }
            }

            // ================================================================
            // ZOOM TOOL - click/drag to zoom
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
                    painter.rect_stroke(
                        r,
                        0.0,
                        egui::Stroke::new(1.0, accent),
                        egui::StrokeKind::Middle,
                    );
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
                        // Simple click - zoom direction based on toggle
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
            // PAN TOOL - click+drag to pan viewport
            // ================================================================
            Tool::Pan => {
                self.zoom_pan_action = ZoomPanAction::None;
                // Drag delta in screen coordinates -> pass directly to Canvas pan_offset
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
            // PERSPECTIVE CROP - 4-corner quad, drag handles
            // ================================================================
            Tool::PerspectiveCrop => {
                let esc_pressed = escape_pressed_global;

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
            _ => {}
        }
    }
}
