impl ToolsPanel {
    fn brush_resize_drag_modifier_held(&self, ui: &egui::Ui) -> bool {
        let binding = &self.brush_resize_drag_binding;
        ui.input(|i| {
            i.modifiers.command == binding.ctrl
                && i.modifiers.shift == binding.shift
                && i.modifiers.alt == binding.alt
        })
    }

    fn reset_brush_pointer_state(&mut self) {
        self.brush_coverage.clear();
        self.tool_state.last_pos = None;
        self.tool_state.last_precise_pos = None;
        self.tool_state.distance_remainder = 0.0;
        self.tool_state.using_secondary_color = false;
        self.tool_state.smooth_pos = None;
    }

    fn commit_brush_straight_line(
        &mut self,
        canvas_state: &mut CanvasState,
        last: (u32, u32),
        current: (u32, u32),
        primary_color_f32: [f32; 4],
        secondary_color_f32: [f32; 4],
    ) -> Option<StrokeEvent> {
        let is_eraser = self.active_tool == Tool::Eraser;
        let editing_mask = canvas_state.edit_layer_mask
            && canvas_state
                .layers
                .get(canvas_state.active_layer_index)
                .is_some_and(|l| l.has_live_mask());

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
            if let Some(layer) = canvas_state.layers.get(canvas_state.active_layer_index) {
                self.stroke_tracker.start_direct_tool(
                    canvas_state.active_layer_index,
                    "Eraser Line",
                    &layer.pixels,
                );
            }
        } else {
            let description = if self.active_tool == Tool::Pencil {
                "Pencil Line"
            } else {
                "Brush Line"
            };
            self.stroke_tracker
                .start_preview_tool(canvas_state.active_layer_index, description);
        }

        if !is_eraser || editing_mask {
            if canvas_state.preview_layer.is_none()
                || canvas_state.preview_layer.as_ref().unwrap().width() != canvas_state.width
                || canvas_state.preview_layer.as_ref().unwrap().height() != canvas_state.height
            {
                canvas_state.preview_layer =
                    Some(TiledImage::new(canvas_state.width, canvas_state.height));
            } else if let Some(ref mut preview) = canvas_state.preview_layer {
                preview.clear();
            }
            canvas_state.preview_blend_mode = self.properties.blending_mode;
        }

        let mirror = canvas_state.mirror_mode;
        let mw = canvas_state.width;
        let mh = canvas_state.height;
        let start_f = (last.0 as f32, last.1 as f32);
        let end_f = (current.0 as f32, current.1 as f32);
        let start_mirrors = mirror.mirror_positions(start_f.0, start_f.1, mw, mh);
        let end_mirrors = mirror.mirror_positions(end_f.0, end_f.1, mw, mh);
        let mut modified_rect = Rect::NOTHING;
        for i in 0..end_mirrors.len {
            let s = start_mirrors.data[i];
            let e = end_mirrors.data[i];
            let r = if self.active_tool == Tool::Pencil {
                self.draw_pixel_line_and_get_bounds(
                    canvas_state,
                    s,
                    e,
                    false,
                    primary_color_f32,
                    secondary_color_f32,
                )
            } else {
                self.draw_line_and_get_bounds(
                    canvas_state,
                    s,
                    e,
                    is_eraser,
                    false,
                    primary_color_f32,
                    secondary_color_f32,
                )
            };
            modified_rect = modified_rect.union(r);
        }

        self.stroke_tracker.expand_bounds(modified_rect);

        if editing_mask {
            let stroke_event = self.stroke_tracker.finish(canvas_state);
            self.commit_preview_to_layer_mask(canvas_state, is_eraser);
            canvas_state.clear_preview_state();
            canvas_state.mark_dirty(Some(modified_rect.expand(12.0)));
            stroke_event
        } else if !is_eraser {
            let stroke_event = self.stroke_tracker.finish(canvas_state);
            self.commit_bezier_to_layer(canvas_state, primary_color_f32);
            canvas_state.clear_preview_state();
            if let Some(ref ev) = stroke_event {
                canvas_state.mark_dirty(Some(ev.bounds.expand(12.0)));
            } else {
                self.mark_full_dirty(canvas_state);
            }
            stroke_event
        } else {
            let stroke_event = self.stroke_tracker.finish(canvas_state);
            canvas_state.mark_dirty(Some(modified_rect));
            stroke_event
        }
    }

    /// Auto-switch to Text tool when active layer is a text layer, and
    /// restore the previous tool when switching away. Called immediately
    /// after layer selection changes so there is no 1-frame delay.
    pub fn auto_switch_tool_for_layer(&mut self, canvas_state: &CanvasState) {
        let is_text = canvas_state
            .layers
            .get(canvas_state.active_layer_index)
            .is_some_and(|l| l.is_text_layer());
        if is_text && self.active_tool != Tool::Text {
            self.tool_before_text_layer = Some(self.active_tool);
            self.active_tool = Tool::Text;
        } else if !is_text
            && let Some(prev) = self.tool_before_text_layer.take()
            && self.active_tool == Tool::Text
        {
            self.active_tool = prev;
        }
    }

    /// Take the pending stroke event (if any) for processing
    pub fn take_stroke_event(&mut self) -> Option<StrokeEvent> {
        self.pending_stroke_event.take()
    }

    /// Take the pending color removal request (if any) for async dispatch
    pub fn take_pending_color_removal(&mut self) -> Option<ColorRemovalRequest> {
        self.pending_color_removal.take()
    }

    /// Take the pending inpaint request (if any) for async dispatch
    pub fn take_pending_inpaint(&mut self) -> Option<crate::ops::inpaint::InpaintRequest> {
        self.content_aware_state.pending_inpaint.take()
    }

    /// Cancel any active text editing session without committing.
    /// Call this whenever the layer being edited is rasterized or otherwise
    /// converted away from `LayerContent::Text`, so that the text tool no
    /// longer holds stale `is_editing` / `editing_text_layer` state that
    /// would otherwise absorb canvas clicks and prevent other tools from
    /// working.
    pub fn cancel_text_editing(&mut self, canvas_state: &mut crate::canvas::CanvasState) {
        if !self.text_state.is_editing && !self.text_state.editing_text_layer {
            return;
        }
        self.stroke_tracker.cancel();
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

    /// Returns true if the user is actively painting (brush/eraser stroke in progress).
    /// Useful for triggering continuous repaint during brush strokes.
    pub fn is_stroke_active(&self) -> bool {
        self.stroke_tracker.is_active || self.tool_state.last_pos.is_some()
    }

    /// Returns true if any tool has an uncommitted preview/in-progress state.
    pub fn has_active_tool_preview(&self) -> bool {
        match self.active_tool {
            Tool::Brush | Tool::Pencil | Tool::Eraser => {
                self.stroke_tracker.is_active || self.tool_state.last_pos.is_some()
            }
            Tool::Line => {
                matches!(
                    self.line_state.line_tool.stage,
                    LineStage::Dragging | LineStage::Editing
                )
            }
            Tool::Gradient => self.gradient_state.drag_start.is_some(),
            Tool::Shapes => self.shapes_state.placed.is_some() || self.shapes_state.is_drawing,
            Tool::Text => self.text_state.is_editing,
            Tool::Fill => false,
            Tool::Liquify => self.liquify_state.is_active,
            Tool::MeshWarp => self.mesh_warp_state.is_active,
            _ => false,
        }
    }

    pub fn debug_operation_label(&self) -> Option<String> {
        if self.active_layer_rgba_prewarm_rx.is_some() {
            return Some(match self.active_tool {
                Tool::MagicWand => "Building: Magic Wand Map".to_string(),
                _ => "Building: Tool Map".to_string(),
            });
        }

        if self.active_tool == Tool::MagicWand && self.magic_wand_state.computing {
            return Some("Building: Magic Wand Map".to_string());
        }

        None
    }

    /// Cancel the current tool's in-progress operation without committing.
    /// Returns true if something was cancelled, false if there was nothing to cancel.
    pub fn cancel_active_tool(&mut self, canvas_state: &mut CanvasState) -> bool {
        match self.active_tool {
            Tool::Brush | Tool::Pencil | Tool::Eraser
                if self.stroke_tracker.is_active || self.tool_state.last_pos.is_some() =>
            {
                self.stroke_tracker.cancel();
                self.tool_state.last_pos = None;
                self.tool_state.last_precise_pos = None;
                self.tool_state.smooth_pos = None;
                canvas_state.clear_preview_state();
                canvas_state.mark_dirty(None);
                true
            }
            Tool::Brush | Tool::Pencil | Tool::Eraser => false,
            Tool::Line => match self.line_state.line_tool.stage {
                LineStage::Dragging | LineStage::Editing => {
                    self.stroke_tracker.cancel();
                    canvas_state.clear_preview_state();
                    if let Some(bounds) = self.line_state.line_tool.last_bounds {
                        canvas_state.mark_dirty(Some(bounds));
                    } else {
                        canvas_state.mark_dirty(None);
                    }
                    self.line_state.line_tool.stage = LineStage::Idle;
                    self.line_state.line_tool.last_bounds = None;
                    self.line_state.line_tool.require_mouse_release = true;
                    self.line_state.line_tool.initial_mouse_pos = None;
                    self.line_state.line_tool.dragging_handle = None;
                    true
                }
                _ => false,
            },
            Tool::Gradient if self.gradient_state.drag_start.is_some() => {
                self.cancel_gradient(canvas_state);
                true
            }
            Tool::Gradient => false,
            Tool::Shapes => {
                if self.shapes_state.placed.is_some() {
                    self.shapes_state.placed = None;
                    canvas_state.clear_preview_state();
                    canvas_state.mark_dirty(None);
                    true
                } else if self.shapes_state.is_drawing {
                    self.shapes_state.is_drawing = false;
                    self.shapes_state.draw_start = None;
                    self.shapes_state.draw_end = None;
                    canvas_state.clear_preview_state();
                    canvas_state.mark_dirty(None);
                    true
                } else {
                    false
                }
            }
            Tool::Text if self.text_state.is_editing => {
                self.text_state.text.clear();
                self.text_state.cursor_pos = 0;
                self.text_state.is_editing = false;
                self.text_state.editing_text_layer = false;
                self.text_state.origin = None;
                self.text_state.dragging_handle = false;
                canvas_state.clear_preview_state();
                canvas_state.mark_dirty(None);
                true
            }
            Tool::Text => false,
            Tool::Fill
                if self.fill_state.active_fill.is_some()
                    || !self.fill_state.pending_clicks.is_empty()
                    || self.fill_state.preview_in_flight =>
            {
                self.clear_fill_preview_state();
                canvas_state.clear_preview_state();
                self.stroke_tracker.cancel();
                true
            }
            Tool::Fill => false,
            Tool::Liquify if self.liquify_state.is_active => {
                self.liquify_state.displacement = None;
                self.liquify_state.is_active = false;
                self.liquify_state.source_snapshot = None;
                self.liquify_state.warp_buffer.clear();
                self.liquify_state.dirty_rect = None;
                canvas_state.clear_preview_state();
                canvas_state.mark_dirty(None);
                true
            }
            Tool::Liquify => false,
            Tool::MeshWarp if self.mesh_warp_state.is_active => {
                self.mesh_warp_state.is_active = false;
                self.mesh_warp_state.source_snapshot = None;
                self.mesh_warp_state.warp_buffer.clear();
                self.stroke_tracker.cancel();
                canvas_state.clear_preview_state();
                canvas_state.mark_dirty(None);
                true
            }
            Tool::MeshWarp => false,
            _ => false,
        }
    }

    /// Commit the current tool's in-progress preview immediately.
    /// Returns true if a commit occurred.
    pub fn commit_active_tool_preview(
        &mut self,
        canvas_state: &mut CanvasState,
        secondary_color_f32: [f32; 4],
    ) -> bool {
        match self.active_tool {
            Tool::Line if self.line_state.line_tool.stage == LineStage::Editing => {
                let line_editing_mask = canvas_state.edit_layer_mask
                    && canvas_state
                        .layers
                        .get(canvas_state.active_layer_index)
                        .is_some_and(|l| l.has_live_mask());
                if let Some(final_bounds) = self.line_state.line_tool.last_bounds {
                    self.stroke_tracker.expand_bounds(final_bounds);
                }
                let stroke_event = self.stroke_tracker.finish(canvas_state);
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
                if stroke_event.is_some() {
                    self.pending_stroke_event = stroke_event;
                }
                true
            }
            Tool::Fill if self.fill_state.active_fill.is_some() => {
                self.commit_fill_preview(canvas_state);
                true
            }
            Tool::Gradient if self.gradient_state.drag_start.is_some() => {
                self.commit_gradient(canvas_state);
                true
            }
            Tool::Text if self.text_state.is_editing => {
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
                true
            }
            Tool::Liquify if self.liquify_state.is_active => {
                self.commit_liquify(canvas_state);
                true
            }
            Tool::MeshWarp if self.mesh_warp_state.is_active => {
                self.commit_mesh_warp(canvas_state);
                true
            }
            Tool::Shapes if self.shapes_state.placed.is_some() => {
                self.commit_shape(canvas_state);
                true
            }
            _ => false,
        }
    }
}
