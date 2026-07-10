impl ToolsPanel {
    fn pointer_over_off_canvas_edit_handle(
        &self,
        screen_pos: egui::Pos2,
        canvas_rect: Rect,
        zoom: f32,
    ) -> bool {
        let to_screen = |cx: f32, cy: f32| -> egui::Pos2 {
            egui::pos2(canvas_rect.min.x + cx * zoom, canvas_rect.min.y + cy * zoom)
        };

        if self.active_tool == Tool::Line && self.line_state.line_tool.stage == LineStage::Editing
        {
            let handle_radius = 8.0;
            for point in self.line_state.line_tool.control_points {
                if (to_screen(point.x, point.y) - screen_pos).length() < handle_radius {
                    return true;
                }
            }

            let p0 = self.line_state.line_tool.control_points[0];
            let pan_handle = to_screen(p0.x, p0.y) + egui::vec2(-22.0, -22.0);
            return (pan_handle - screen_pos).length() < 10.0;
        }

        if self.active_tool == Tool::Shapes
            && let Some(placed) = self.shapes_state.placed.as_ref()
        {
            let cos_r = placed.rotation.cos();
            let sin_r = placed.rotation.sin();
            let corners_local = [
                (-placed.hw, -placed.hh),
                (placed.hw, -placed.hh),
                (placed.hw, placed.hh),
                (-placed.hw, placed.hh),
            ];
            let corners: Vec<egui::Pos2> = corners_local
                .iter()
                .map(|(cx, cy)| {
                    to_screen(
                        cx * cos_r - cy * sin_r + placed.cx,
                        cx * sin_r + cy * cos_r + placed.cy,
                    )
                })
                .collect();

            let handle_radius = 10.0;
            if corners
                .iter()
                .any(|corner| (*corner - screen_pos).length() < handle_radius)
            {
                return true;
            }

            let edge_handles = [
                egui::pos2(
                    (corners[0].x + corners[1].x) * 0.5,
                    (corners[0].y + corners[1].y) * 0.5,
                ),
                egui::pos2(
                    (corners[1].x + corners[2].x) * 0.5,
                    (corners[1].y + corners[2].y) * 0.5,
                ),
                egui::pos2(
                    (corners[2].x + corners[3].x) * 0.5,
                    (corners[2].y + corners[3].y) * 0.5,
                ),
                egui::pos2(
                    (corners[3].x + corners[0].x) * 0.5,
                    (corners[3].y + corners[0].y) * 0.5,
                ),
            ];
            if edge_handles
                .iter()
                .any(|handle| (*handle - screen_pos).length() < handle_radius)
            {
                return true;
            }

            let top_mid = edge_handles[0];
            let center = to_screen(placed.cx, placed.cy);
            let dir = top_mid - center;
            let dir_len = dir.length().max(0.001);
            let rot_handle = top_mid + dir / dir_len * 20.0;
            return (rot_handle - screen_pos).length() < handle_radius;
        }

        false
    }

    pub fn handle_input(
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
        mut gpu_renderer: Option<&mut crate::gpu::GpuRenderer>,
        ui_blocks_canvas_input: bool,
    ) {
        let mut stroke_event: Option<StrokeEvent> = None;

        self.poll_active_layer_rgba_prewarm();
        self.maybe_prewarm_active_layer_rgba(canvas_state);

        // -- Auto-commit on layer change ---------------------------
        // If the active layer index or layer count changed (user clicked a
        // different layer, reordered, deleted, etc.), commit any tool that
        // has uncommitted state - mirrors the auto-commit-on-tool-switch
        // logic below.
        let prev_layer_index = self.last_tracked_layer_index;
        let layer_changed = canvas_state.active_layer_index != self.last_tracked_layer_index
            || canvas_state.layers.len() != self.last_tracked_layer_count;
        if layer_changed {
            // Line
            if self.active_tool == Tool::Line
                && self.line_state.line_tool.stage == LineStage::Editing
            {
                let line_editing_mask = canvas_state.edit_layer_mask
                    && canvas_state
                        .layers
                        .get(canvas_state.active_layer_index)
                        .is_some_and(|l| l.has_live_mask());
                if let Some(final_bounds) = self.line_state.line_tool.last_bounds {
                    self.stroke_tracker.expand_bounds(final_bounds);
                }
                let auto_commit_event = self.stroke_tracker.finish(canvas_state);
                let mirror_bounds = canvas_state.mirror_preview_layer();
                if line_editing_mask {
                    self.commit_preview_to_layer_mask(canvas_state, false);
                } else {
                    self.commit_bezier_to_layer(canvas_state, secondary_color_f32);
                }
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
                self.clear_fill_preview_state();
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
                    self.text_state.editing_text_layer = false;
                    self.text_state.origin = None;
                    canvas_state.text_editing_layer = None;
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
                if self.shapes_state.source_layer_index.is_none() {
                    self.shapes_state.source_layer_index = Some(prev_layer_index);
                }
                self.commit_shape(canvas_state);
            }

            self.last_tracked_layer_index = canvas_state.active_layer_index;
            self.last_tracked_layer_count = canvas_state.layers.len();

            // Auto-switch tool for text layers (also called from app.rs
            // immediately after layers panel for zero-delay switching).
            self.auto_switch_tool_for_layer(canvas_state);
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

        // Auto-commit Bezier line if tool changed away from Line tool
        if self.active_tool != Tool::Line && self.line_state.line_tool.stage == LineStage::Editing {
            let line_editing_mask = canvas_state.edit_layer_mask
                && canvas_state
                    .layers
                    .get(canvas_state.active_layer_index)
                    .is_some_and(|l| l.has_live_mask());
            // Capture "before" for undo (layer still unchanged, line is in preview)
            if let Some(final_bounds) = self.line_state.line_tool.last_bounds {
                self.stroke_tracker.expand_bounds(final_bounds);
            }
            let auto_commit_event = self.stroke_tracker.finish(canvas_state);

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
            self.clear_fill_preview_state();
            canvas_state.mark_dirty(None);
        }

        // Auto-commit Magic Wand if tool changed away from Magic Wand tool
        if self.active_tool != Tool::MagicWand && self.magic_wand_has_session() {
            self.finalize_magic_wand_session(canvas_state);
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
                self.text_state.editing_text_layer = false;
                self.text_state.origin = None;
                canvas_state.text_editing_layer = None;
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

        // On Wayland the stylus may fire Touch events instead of Pointer events,
        // so augment each pointer query with a Touch-event fallback.
        let mut is_primary_down = ui.input(|i| {
            i.pointer.primary_down()
                || i.events.iter().any(|e| {
                    matches!(
                        e,
                        egui::Event::Touch {
                            phase: egui::TouchPhase::Start | egui::TouchPhase::Move,
                            ..
                        }
                    )
                })
        });
        let mut is_primary_released = ui.input(|i| {
            i.pointer.primary_released()
                || i.events.iter().any(|e| {
                    matches!(
                        e,
                        egui::Event::Touch {
                            phase: egui::TouchPhase::End,
                            ..
                        }
                    )
                })
        });
        let mut is_primary_clicked = ui.input(|i| i.pointer.primary_clicked());
        let mut is_primary_pressed = ui.input(|i| {
            i.pointer.primary_pressed()
                || i.events.iter().any(|e| {
                    matches!(
                        e,
                        egui::Event::Touch {
                            phase: egui::TouchPhase::Start,
                            ..
                        }
                    )
                })
        }); // Just pressed this frame
        let mut is_secondary_down = ui.input(|i| i.pointer.secondary_down());
        let mut is_secondary_pressed =
            ui.input(|i| i.pointer.button_pressed(egui::PointerButton::Secondary));
        let mut is_secondary_released = ui.input(|i| i.pointer.secondary_released());
        let mut is_secondary_clicked = ui.input(|i| i.pointer.secondary_clicked());
        let pointer_over_canvas = ui.input(|i| {
            i.pointer
                .hover_pos()
                .or_else(|| i.pointer.interact_pos())
                .is_some_and(|p| canvas_rect.contains(p))
        });
        let pointer_over_egui = ui_blocks_canvas_input;
        let pointer_over_edit_handle = ui.input(|i| {
            i.pointer
                .hover_pos()
                .or_else(|| i.pointer.interact_pos())
                .is_some_and(|p| self.pointer_over_off_canvas_edit_handle(p, canvas_rect, zoom))
        });
        if is_primary_pressed || is_secondary_pressed {
            self.canvas_pointer_active =
                (pointer_over_canvas || pointer_over_edit_handle) && !pointer_over_egui;
        }
        if ui_blocks_canvas_input {
            self.canvas_pointer_active = false;
        }
        let pointer_allowed = self.canvas_pointer_active
            || ((pointer_over_canvas || pointer_over_edit_handle) && !pointer_over_egui);
        if !pointer_allowed {
            is_primary_down = false;
            is_primary_released = false;
            is_primary_clicked = false;
            is_primary_pressed = false;
            is_secondary_down = false;
            is_secondary_pressed = false;
            is_secondary_released = false;
            is_secondary_clicked = false;
        }
        let shift_held = ui.input(|i| i.modifiers.shift);
        let enter_pressed =
            ui.input(|i| i.key_pressed(egui::Key::Enter)) || self.injected_enter_pressed;
        let escape_pressed_global =
            ui.input(|i| i.key_pressed(egui::Key::Escape)) || self.injected_escape_pressed;
        self.injected_enter_pressed = false;
        self.injected_escape_pressed = false;

        // Read pen/touch pressure from egui touch events (Apple Pencil, Wacom, etc.)
        // Uses the latest Touch event's force value; falls back to 1.0 (no pen).
        let touch_pressure = ui.input(|i| {
            let mut pressure = None;
            for ev in &i.events {
                if let egui::Event::Touch {
                    force: Some(f),
                    phase,
                    ..
                } = ev
                    && matches!(phase, egui::TouchPhase::Start | egui::TouchPhase::Move)
                {
                    pressure = Some(*f);
                }
            }
            pressure
        });
        if let Some(p) = touch_pressure {
            self.tool_state.current_pressure = p.clamp(0.0, 1.0);
        } else if !is_primary_down && !is_secondary_down {
            // Reset to full pressure when not actively drawing
            self.tool_state.current_pressure = 1.0;
        }

        match self.active_tool {
            Tool::Brush | Tool::Eraser | Tool::Pencil | Tool::Line => self.handle_stroke_tools_input(
                ui,
                canvas_state,
                canvas_pos,
                canvas_pos_f32,
                canvas_pos_f32_clamped,
                canvas_pos_unclamped,
                raw_motion_events,
                painter,
                canvas_rect,
                zoom,
                primary_color_f32,
                secondary_color_f32,
                &mut gpu_renderer,
                &mut stroke_event,
                is_primary_down,
                is_primary_released,
                is_primary_clicked,
                is_primary_pressed,
                is_secondary_down,
                is_secondary_pressed,
                is_secondary_released,
                is_secondary_clicked,
                shift_held,
                enter_pressed,
                escape_pressed_global,
            ),
            Tool::RectangleSelect
            | Tool::EllipseSelect
            | Tool::MovePixels
            | Tool::MoveSelection
            | Tool::MagicWand
            | Tool::Fill
            | Tool::ColorPicker => self.handle_selection_tools_input(
                ui,
                canvas_state,
                canvas_pos,
                canvas_pos_f32,
                canvas_pos_f32_clamped,
                canvas_pos_unclamped,
                raw_motion_events,
                painter,
                canvas_rect,
                zoom,
                primary_color_f32,
                secondary_color_f32,
                &mut gpu_renderer,
                &mut stroke_event,
                is_primary_down,
                is_primary_released,
                is_primary_clicked,
                is_primary_pressed,
                is_secondary_down,
                is_secondary_pressed,
                is_secondary_released,
                is_secondary_clicked,
                shift_held,
                enter_pressed,
                escape_pressed_global,
            ),
            Tool::Text => self.handle_text_tool_input(
                ui,
                canvas_state,
                canvas_pos,
                canvas_pos_f32,
                canvas_pos_f32_clamped,
                canvas_pos_unclamped,
                raw_motion_events,
                painter,
                canvas_rect,
                zoom,
                primary_color_f32,
                secondary_color_f32,
                &mut gpu_renderer,
                &mut stroke_event,
                is_primary_down,
                is_primary_released,
                is_primary_clicked,
                is_primary_pressed,
                is_secondary_down,
                is_secondary_pressed,
                is_secondary_released,
                is_secondary_clicked,
                shift_held,
                enter_pressed,
                escape_pressed_global,
            ),
            Tool::Liquify
            | Tool::MeshWarp
            | Tool::ColorRemover
            | Tool::Smudge
            | Tool::Shapes
            | Tool::Gradient => self.handle_surface_transform_input(
                ui,
                canvas_state,
                canvas_pos,
                canvas_pos_f32,
                canvas_pos_f32_clamped,
                canvas_pos_unclamped,
                raw_motion_events,
                painter,
                canvas_rect,
                zoom,
                primary_color_f32,
                secondary_color_f32,
                &mut gpu_renderer,
                &mut stroke_event,
                is_primary_down,
                is_primary_released,
                is_primary_clicked,
                is_primary_pressed,
                is_secondary_down,
                is_secondary_pressed,
                is_secondary_released,
                is_secondary_clicked,
                shift_held,
                enter_pressed,
                escape_pressed_global,
            ),
            Tool::CloneStamp
            | Tool::ContentAwareBrush
            | Tool::Lasso
            | Tool::Zoom
            | Tool::Pan
            | Tool::PerspectiveCrop => self.handle_utility_navigation_input(
                ui,
                canvas_state,
                canvas_pos,
                canvas_pos_f32,
                canvas_pos_f32_clamped,
                canvas_pos_unclamped,
                raw_motion_events,
                painter,
                canvas_rect,
                zoom,
                primary_color_f32,
                secondary_color_f32,
                &mut gpu_renderer,
                &mut stroke_event,
                is_primary_down,
                is_primary_released,
                is_primary_clicked,
                is_primary_pressed,
                is_secondary_down,
                is_secondary_pressed,
                is_secondary_released,
                is_secondary_clicked,
                shift_held,
                enter_pressed,
                escape_pressed_global,
            ),
        }

        if stroke_event.is_some() {
            self.pending_stroke_event = stroke_event;
        }

        if is_primary_released || is_secondary_released || (!is_primary_down && !is_secondary_down) {
            self.canvas_pointer_active = false;
        }
    }
}

include!("handle_input/brush_line_input.rs");
include!("handle_input/selection_fill_input.rs");
include!("handle_input/text_tool_input.rs");
include!("handle_input/surface_transform_input.rs");
include!("handle_input/utility_navigation_input.rs");

