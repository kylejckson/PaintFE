impl ToolsPanel {
    #[allow(clippy::too_many_arguments, unused_variables)]
    fn handle_selection_tools_input(
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
            Tool::RectangleSelect | Tool::EllipseSelect => {
                let esc_pressed = escape_pressed_global;
                let alt_held = ui.input(|i| i.modifiers.alt);
                let ctrl_held = ui.input(|i| i.modifiers.command);
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
                //   Shift+Alt -> Intersect
                //   Ctrl      -> Add
                //   Alt       -> Subtract
                //   Right-click -> Subtract (when context bar mode is Replace)
                //   else      -> context bar mode
                if (is_primary_pressed || is_secondary_pressed)
                    && !self.selection_state.dragging
                    && let Some(pos_f) = canvas_pos_unclamped
                {
                    let pos2 = Pos2::new(pos_f.0, pos_f.1);
                    self.selection_state.dragging = true;
                    self.selection_state.drag_start = Some(pos2);
                    self.selection_state.drag_end = Some(pos2);
                    self.selection_state.right_click_drag = is_secondary_pressed;
                    // Determine and lock effective mode
                    self.selection_state.drag_effective_mode = if is_secondary_pressed {
                        SelectionMode::Subtract
                    } else if shift_held && alt_held {
                        SelectionMode::Intersect
                    } else if ctrl_held {
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
                    && let Some(pos_f) = canvas_pos_unclamped
                {
                    let mut end = Pos2::new(pos_f.0, pos_f.1);

                    if let Some(start) = self.selection_state.drag_start {
                        let aspect_ratio = if shift_held {
                            Some(1.0)
                        } else if self.active_tool == Tool::RectangleSelect {
                            parse_selection_aspect_ratio(&self.selection_state.aspect_ratio_input)
                                .ok()
                                .flatten()
                                .map(|ratio| match ratio {
                                    SelectionAspectRatio::Fixed(ratio) => ratio,
                                    SelectionAspectRatio::Image => {
                                        canvas_state.width as f32
                                            / canvas_state.height.max(1) as f32
                                    }
                                })
                        } else {
                            None
                        };
                        if let Some(aspect_ratio) = aspect_ratio {
                            end = constrain_selection_aspect(
                                start,
                                end,
                                aspect_ratio,
                                canvas_state.width,
                                canvas_state.height,
                            );
                        }
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
                        let raw_min_x = start.x.min(end.x).max(0.0);
                        let raw_min_y = start.y.min(end.y).max(0.0);
                        let raw_max_x = start.x.max(end.x).max(0.0);
                        let raw_max_y = start.y.max(end.y).max(0.0);
                        let min_x = (raw_min_x as u32).min(canvas_state.width.saturating_sub(1));
                        let min_y = (raw_min_y as u32).min(canvas_state.height.saturating_sub(1));
                        let max_x = (raw_max_x as u32).min(canvas_state.width.saturating_sub(1));
                        let max_y = (raw_max_y as u32).min(canvas_state.height.saturating_sub(1));

                        let sel_before = canvas_state.selection_mask.clone();
                        let sel_before_all = canvas_state.selection_all;

                        // Allow single-pixel selections for pixel art accuracy.
                        // Deselect only happens on a zero-size drag (start == end,
                        // which resolves to a single point).
                        if max_x > min_x || max_y > min_y {
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
                        } else {
                            // Zero-size click => deselect
                            canvas_state.clear_selection();
                        }

                        let sel_after = canvas_state.selection_mask.clone();
                        let sel_after_all = canvas_state.selection_all;
                        let tool_name = match self.active_tool {
                            Tool::RectangleSelect => "Rectangle Select",
                            _ => "Ellipse Select",
                        };
                        self.pending_history_commands
                            .push(Box::new(SelectionCommand::new_states(
                                tool_name,
                                sel_before,
                                sel_before_all,
                                sel_after,
                                sel_after_all,
                            )));
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
                let esc_pressed = escape_pressed_global;
                let shift_held_mw = ui.input(|i| i.modifiers.shift);
                let alt_held_mw = ui.input(|i| i.modifiers.alt);
                let ctrl_held_mw = ui.input(|i| i.modifiers.command);
                let click_scope =
                    if (ctrl_held_mw && shift_held_mw) || self.magic_wand_state.global_select {
                        MagicWandScope::Global
                    } else {
                        MagicWandScope::Contiguous
                    };
                let click_mode = if is_primary_clicked || is_secondary_clicked {
                    if is_secondary_clicked {
                        SelectionMode::Subtract
                    } else if ctrl_held_mw && shift_held_mw {
                        SelectionMode::Add
                    } else if shift_held_mw && alt_held_mw {
                        SelectionMode::Intersect
                    } else if ctrl_held_mw {
                        SelectionMode::Add
                    } else if alt_held_mw {
                        SelectionMode::Subtract
                    } else {
                        self.selection_state.mode
                    }
                } else {
                    self.selection_state.mode
                };

                // Poll async distance map computation
                if let Some(rx) = &self.magic_wand_state.async_rx {
                    if let Ok(result) = rx.try_recv() {
                        match result {
                            MagicWandAsyncResult::Ready { request_id, index } => {
                                if let Some(pending) =
                                    self.magic_wand_state.pending_operation.take()
                                    && pending.request_id == request_id
                                {
                                    self.magic_wand_state.operations.push(MagicWandOperation {
                                        start_x: pending.start_x,
                                        start_y: pending.start_y,
                                        target_color: pending.target_color,
                                        combine_mode: pending.combine_mode,
                                        scope: pending.scope,
                                        distance_mode: pending.distance_mode,
                                        connectivity: pending.connectivity,
                                        region_index: index,
                                    });
                                }
                            }
                        }
                        self.magic_wand_state.async_rx = None;
                        self.magic_wand_state.computing = false;
                        self.magic_wand_state.last_applied_tolerance = -1.0;
                        self.magic_wand_state.last_applied_aa = !self.magic_wand_state.anti_aliased;
                        self.magic_wand_state.preview_pending = true;
                        self.magic_wand_state.tolerance_changed_at = None;
                        ui.ctx().request_repaint();
                    } else {
                        // Still computing - keep repainting to poll
                        ui.ctx().request_repaint();
                    }
                }

                // Commit on Enter or clear selection on Escape.
                // Escape should behave like deselect for selection tools and
                // must not restore a prior mask from the wand session cache.
                if enter_pressed || esc_pressed {
                    if esc_pressed {
                        canvas_state.clear_selection();
                        canvas_state.mark_dirty(None);
                        self.clear_magic_wand_async_state();
                        ui.ctx().request_repaint();
                    } else if self.magic_wand_has_session() {
                        self.finalize_magic_wand_session(canvas_state);
                        ui.ctx().request_repaint();
                    }
                }

                // Click to start new selection (or commit current + start new)
                if (is_primary_clicked || is_secondary_clicked)
                    && let Some(pos) = canvas_pos
                {
                    if click_mode == SelectionMode::Replace && self.magic_wand_has_session() {
                        self.finalize_magic_wand_session(canvas_state);
                    }
                    if click_mode == SelectionMode::Replace {
                        self.clear_magic_wand_async_state();
                        self.begin_magic_wand_session_if_needed(canvas_state);
                        canvas_state.selection_mask = None;
                        canvas_state.invalidate_selection_overlay();
                        canvas_state.mark_dirty(None);
                    } else {
                        self.begin_magic_wand_session_if_needed(canvas_state);
                    }

                    self.perform_magic_wand_selection(
                        canvas_state,
                        pos,
                        click_mode,
                        click_scope,
                        gpu_renderer.as_deref_mut(),
                    );
                    ui.ctx().request_repaint();
                }

                // Re-threshold the distance map only when tolerance or anti-alias changed
                let has_map = !self.magic_wand_state.operations.is_empty();
                if has_map
                    && self.active_tool == Tool::MagicWand
                    && (self.magic_wand_state.preview_pending
                        || (self.magic_wand_state.tolerance
                            - self.magic_wand_state.last_applied_tolerance)
                            .abs()
                            > 0.001
                        || self.magic_wand_state.anti_aliased
                            != self.magic_wand_state.last_applied_aa)
                {
                    self.magic_wand_state.preview_pending = true;
                    self.maybe_spawn_magic_wand_preview(canvas_state, gpu_renderer.as_deref_mut());
                    ui.ctx().request_repaint();
                }
            }

            // Fill tool - direct bucket fill from a fresh sample every click
            Tool::Fill => {
                let esc_pressed = escape_pressed_global;
                let fill_triggered = !self.fill_state.pending_clicks.is_empty()
                    || is_primary_clicked
                    || is_secondary_clicked
                    || is_primary_pressed
                    || is_secondary_pressed;

                // Guard: auto-rasterize text layers before destructive fill
                if fill_triggered
                    && let Some(layer) = canvas_state.layers.get(canvas_state.active_layer_index)
                    && layer.is_text_layer()
                {
                    self.pending_auto_rasterize = Some(canvas_state.active_layer_index);
                    return;
                }

                // Commit on Enter or cancel on Escape
                if enter_pressed || esc_pressed {
                    self.clear_fill_preview_state();
                    canvas_state.clear_preview_state();
                }

                if let Some(pending) = self.fill_state.pending_clicks.pop_front() {
                    self.handle_fill_click(
                        canvas_state,
                        pending.pos,
                        pending.use_secondary,
                        pending.global_fill,
                        primary_color_f32,
                        secondary_color_f32,
                        gpu_renderer.as_deref_mut(),
                    );
                    ui.ctx().request_repaint();
                }
            }

            // Color picker tool - sample colors
            Tool::ColorPicker => {
                let pointer_moved_at_least_one_pixel = ui.input(|i| {
                    let delta = i.pointer.delta();
                    delta.x.abs() >= 1.0 || delta.y.abs() >= 1.0
                });
                if is_primary_down
                    && pointer_moved_at_least_one_pixel
                    && let Some(pos) = canvas_pos
                {
                    self.pick_color_at_position(canvas_state, pos, false);
                }
            }

            // ================================================================
            // TEXT TOOL - click to place cursor, type to add text
            // ================================================================
            _ => {}
        }
    }
}

fn constrain_selection_aspect(
    start: Pos2,
    end: Pos2,
    aspect_ratio: f32,
    canvas_width: u32,
    canvas_height: u32,
) -> Pos2 {
    let dx = end.x - start.x;
    let dy = end.y - start.y;
    if dx == 0.0 && dy == 0.0 {
        return end;
    }

    let (mut width, mut height) = (dx.abs(), dy.abs());
    if width / aspect_ratio >= height {
        height = width / aspect_ratio;
    } else {
        width = height * aspect_ratio;
    }

    let max_x = canvas_width.saturating_sub(1) as f32;
    let max_y = canvas_height.saturating_sub(1) as f32;
    if start.x >= 0.0 && start.x <= max_x && start.y >= 0.0 && start.y <= max_y {
        let available_width = if dx >= 0.0 { max_x - start.x } else { start.x };
        let available_height = if dy >= 0.0 { max_y - start.y } else { start.y };
        let scale = (available_width / width)
            .min(available_height / height)
            .min(1.0);
        if scale.is_finite() {
            width *= scale.max(0.0);
            height *= scale.max(0.0);
        }
    }

    Pos2::new(
        start.x + width * dx.signum(),
        start.y + height * dy.signum(),
    )
}

#[cfg(test)]
mod selection_aspect_ratio_tests {
    use super::*;

    #[test]
    fn parses_selection_aspect_ratios() {
        assert_eq!(parse_selection_aspect_ratio(""), Ok(None));
        assert_eq!(
            parse_selection_aspect_ratio(" 16 : 9 "),
            Ok(Some(SelectionAspectRatio::Fixed(16.0 / 9.0)))
        );
        assert_eq!(
            parse_selection_aspect_ratio("IMAGE"),
            Ok(Some(SelectionAspectRatio::Image))
        );
        assert!(parse_selection_aspect_ratio("16x9").is_err());
        assert!(parse_selection_aspect_ratio("0:9").is_err());
    }

    #[test]
    fn constrains_selection_aspect_and_canvas_edge() {
        let end = constrain_selection_aspect(
            Pos2::new(10.0, 10.0),
            Pos2::new(90.0, 20.0),
            4.0 / 3.0,
            100,
            100,
        );
        assert_eq!(end, Pos2::new(90.0, 70.0));

        let end = constrain_selection_aspect(
            Pos2::new(80.0, 80.0),
            Pos2::new(99.0, 99.0),
            16.0 / 9.0,
            100,
            100,
        );
        assert!((end.x - 99.0).abs() < f32::EPSILON);
        assert!((end.y - 90.6875).abs() < f32::EPSILON);
    }
}
