impl ToolsPanel {
    pub(crate) fn finish_paint_for_action(
        &mut self,
        canvas: &mut CanvasState,
        primary: [f32; 4],
        secondary: [f32; 4],
    ) -> bool {
        if let Some(event) = self.finish_paint_stroke(canvas, primary, secondary, true) {
            self.pending_stroke_event = Some(event);
            true
        } else {
            false
        }
    }

    /// Finish against the originating layer, even if the selected layer changed.
    pub(crate) fn finish_paint_stroke(
        &mut self,
        canvas_state: &mut CanvasState,
        primary_color_f32: [f32; 4],
        secondary_color_f32: [f32; 4],
        require_release: bool,
    ) -> Option<StrokeEvent> {
        if !matches!(self.active_tool, Tool::Brush | Tool::Pencil | Tool::Eraser)
            || !self.is_stroke_active()
        {
            return None;
        }
        let selected = canvas_state.active_layer_index;
        let selected_mask = canvas_state.edit_layer_mask;
        let Some((target, editing_mask)) = self.paint_target.or_else(|| {
            canvas_state
                .layers
                .get(self.stroke_tracker.layer_index)
                .map(|layer| {
                    (
                        layer.runtime_id,
                        self.stroke_tracker.target == StrokeTarget::LayerMask,
                    )
                })
        }) else {
            self.stroke_tracker.cancel();
            self.reset_brush_pointer_state();
            canvas_state.clear_preview_state();
            self.paint_requires_release = require_release;
            return None;
        };
        let Some(index) = canvas_state
            .layers
            .iter()
            .position(|layer| layer.runtime_id == target)
        else {
            self.stroke_tracker.cancel();
            self.reset_brush_pointer_state();
            canvas_state.clear_preview_state();
            self.paint_requires_release = require_release;
            return None;
        };
        canvas_state.active_layer_index = index;
        self.stroke_tracker.layer_index = index;
        canvas_state.edit_layer_mask = editing_mask;
        let is_eraser = self.active_tool == Tool::Eraser;

        // Finish the short tail left after the last distance-spaced deposit.
        if self.active_tool == Tool::Brush
            && matches!(
                self.properties.brush_mode,
                BrushMode::Normal | BrushMode::BuildUp
            )
            && self.tool_state.distance_remainder > 1e-5
            && let Some(end) = self.tool_state.last_precise_pos
        {
            let positions = canvas_state.mirror_mode.mirror_positions(
                end.x,
                end.y,
                canvas_state.width,
                canvas_state.height,
            );
            let mut bounds = Rect::NOTHING;
            for (arm, &pos) in positions.iter().enumerate() {
                self.brush_motion_arm = arm;
                bounds = bounds.union(self.draw_circle_and_get_bounds(
                    canvas_state,
                    pos,
                    false,
                    self.tool_state.using_secondary_color,
                    primary_color_f32,
                    secondary_color_f32,
                ));
            }
            self.stroke_tracker.expand_bounds(bounds);
            if bounds.is_positive() {
                canvas_state.mark_preview_changed_rect(bounds);
            }
        }

        // Capture "before" NOW (layer still unchanged), then commit
        // Finish stroke tracking BEFORE commit - this captures "before" from unchanged layer
        let stroke_event = self.stroke_tracker.finish(canvas_state);

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
        canvas_state.active_layer_index = selected;
        canvas_state.edit_layer_mask = selected_mask;
        self.reset_brush_pointer_state();
        self.paint_requires_release = require_release;
        stroke_event
    }
}
