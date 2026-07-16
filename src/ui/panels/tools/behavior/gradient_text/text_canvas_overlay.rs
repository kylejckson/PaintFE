fn text_move_handle_canvas_pos(
    origin: [f32; 2],
    alignment: crate::ops::text::TextAlignment,
    font_size: f32,
    zoom: f32,
) -> (f32, f32) {
    // Radius is 6 px and the box padding reaches 4 px left of the origin.
    // A 17 px center offset leaves the requested 7 px visual gap.
    let offset = 17.0 / zoom.max(0.001);
    match alignment {
        crate::ops::text::TextAlignment::Left => (origin[0] - offset, origin[1] + font_size * 0.5),
        crate::ops::text::TextAlignment::Center => (origin[0], origin[1] - offset),
        crate::ops::text::TextAlignment::Right => (origin[0] + offset, origin[1] + font_size * 0.5),
    }
}

impl ToolsPanel {
    pub fn update_gradient_if_dirty(
        &mut self,
        canvas_state: &mut CanvasState,
        gpu_renderer: Option<&mut crate::gpu::GpuRenderer>,
    ) {
        if self.gradient_state.preview_dirty
            && self.gradient_state.drag_start.is_some()
            && !self.gradient_state.dragging
        {
            self.render_gradient_to_preview(canvas_state, gpu_renderer);
        }
        self.gradient_state.preview_dirty = false;
    }

    /// Re-render text preview if context bar changed properties (color, alignment, etc.)
    /// Called outside the allow_input gate so it works when interacting with UI.
    pub fn update_text_if_dirty(
        &mut self,
        canvas_state: &mut CanvasState,
        primary_color_f32: [f32; 4],
    ) {
        if !self.text_state.is_editing || self.text_state.text.is_empty() {
            return;
        }
        // Apply context bar style changes (bold, italic, etc.) to the selection.
        // This MUST run outside the allow_input gate because when the user clicks
        // Bold in the context bar (egui UI), ui_blocking=true and text_drag_override=false
        // (due to pointer_over_egui_with_touch), so handle_input is skipped.
        // If we don't process ctx_bar_style_dirty here, sync_text_layer_run_style
        // (called from render_text_preview) writes the effective weight to the
        // single run, making the ENTIRE block bold instead of just the selection.
        if self.text_state.ctx_bar_style_dirty && self.text_state.editing_text_layer {
            self.text_state.ctx_bar_style_dirty = false;
            if let Some(update) = self.text_state.pending_ctx_style_update.take() {
                match update {
                    TextStyleUpdate::FontFamily(font_family) => {
                        self.text_layer_apply_block_style(canvas_state, |style| {
                            style.font_family.clone_from(&font_family);
                        });
                    }
                    TextStyleUpdate::FontWeight(font_weight) => {
                        self.text_layer_apply_block_style(canvas_state, |style| {
                            style.font_weight = font_weight;
                        });
                    }
                    TextStyleUpdate::FontSize(font_size) => {
                        self.text_layer_apply_block_style(canvas_state, |style| {
                            style.font_size = font_size;
                        });
                    }
                    TextStyleUpdate::Bold {
                        enabled,
                        base_weight,
                    } => {
                        let effective_weight = if enabled {
                            if base_weight < 600 {
                                700u16
                            } else {
                                (base_weight + 200).min(900)
                            }
                        } else if base_weight >= 600 {
                            400
                        } else {
                            base_weight
                        };
                        self.text_layer_toggle_style(canvas_state, |style| {
                            style.font_weight = effective_weight;
                        });
                    }
                    TextStyleUpdate::Italic(italic) => {
                        self.text_layer_toggle_style(canvas_state, |style| {
                            style.italic = italic;
                        });
                    }
                    TextStyleUpdate::Underline(underline) => {
                        self.text_layer_toggle_style(canvas_state, |style| {
                            style.underline = underline;
                        });
                    }
                    TextStyleUpdate::Strikethrough(strikethrough) => {
                        self.text_layer_toggle_style(canvas_state, |style| {
                            style.strikethrough = strikethrough;
                        });
                    }
                    TextStyleUpdate::LetterSpacing(letter_spacing) => {
                        self.text_layer_apply_block_style(canvas_state, |style| {
                            style.letter_spacing = letter_spacing;
                        });
                    }
                    TextStyleUpdate::WidthScale(width_scale) => {
                        self.text_layer_apply_block_style(canvas_state, |style| {
                            style.width_scale = width_scale;
                        });
                    }
                    TextStyleUpdate::HeightScale(height_scale) => {
                        self.text_layer_apply_block_style(canvas_state, |style| {
                            style.height_scale = height_scale;
                        });
                    }
                }
            }
        }
        // Detect color change
        let current_color = [
            (primary_color_f32[0] * 255.0) as u8,
            (primary_color_f32[1] * 255.0) as u8,
            (primary_color_f32[2] * 255.0) as u8,
            (primary_color_f32[3] * 255.0) as u8,
        ];
        if self.text_state.last_color != current_color {
            self.text_state.last_color = current_color;
            self.text_state.preview_dirty = true;
        }
        if self.text_state.preview_dirty {
            self.render_text_preview(canvas_state, primary_color_f32);
        }
    }

    /// Map a char offset within a logical line to (visual_line_index, char_in_visual_line)
    /// when the text has been word-wrapped.
    /// `logical_line` is the original text (one logical line, no `\n`).
    /// `char_pos` is the char position within that logical line.
    /// `visual_lines` are the output of `word_wrap_line`.
    fn map_char_to_visual_line(
        logical_line: &str,
        char_pos: usize,
        visual_lines: &[String],
    ) -> (usize, usize) {
        if visual_lines.is_empty() {
            return (0, 0);
        }
        if visual_lines.len() == 1 {
            return (0, char_pos.min(visual_lines[0].chars().count()));
        }
        let logical_chars: Vec<char> = logical_line.chars().collect();
        let mut consumed = 0usize;
        for (vi, vline) in visual_lines.iter().enumerate() {
            let vlen = vline.chars().count();
            let line_end = consumed + vlen;
            if char_pos >= consumed && char_pos < line_end {
                return (vi, char_pos - consumed);
            }
            // Count whitespace gap consumed between this visual line and next
            let mut next_start = line_end;
            if vi < visual_lines.len() - 1 {
                while next_start < logical_chars.len() && logical_chars[next_start].is_whitespace()
                {
                    next_start += 1;
                }
            }
            // Cursor in trimmed whitespace ÔåÆ end of this visual line
            if char_pos >= line_end && char_pos < next_start {
                return (vi, vlen);
            }
            consumed = next_start.max(line_end);
        }
        let last = visual_lines.len() - 1;
        (last, visual_lines[last].chars().count())
    }

    /// Inverse of `map_char_to_visual_line`: given a visual line index and char offset,
    /// return the char position within the original logical line.
    fn map_visual_line_to_char(
        logical_line: &str,
        visual_lines: &[String],
        visual_line: usize,
        char_in_line: usize,
    ) -> usize {
        if visual_lines.is_empty() {
            return 0;
        }
        let logical_chars: Vec<char> = logical_line.chars().collect();
        let mut consumed = 0usize;
        for (vi, vline) in visual_lines.iter().enumerate() {
            let vlen = vline.chars().count();
            if vi == visual_line {
                return consumed + char_in_line.min(vlen);
            }
            let line_end = consumed + vlen;
            let mut next_start = line_end;
            while next_start < logical_chars.len() && logical_chars[next_start].is_whitespace() {
                next_start += 1;
            }
            consumed = next_start.max(line_end);
        }
        logical_chars.len()
    }

    /// Given the full text (with `\n`), a byte position, and wrap parameters,
    /// return `(visual_line_index, char_in_visual_line)` accounting for word wrapping.
    fn byte_pos_to_visual(
        text: &str,
        byte_pos: usize,
        font: &ab_glyph::FontArc,
        font_size: f32,
        max_width: f32,
        letter_spacing: f32,
        width_scale: f32,
        height_scale: f32,
    ) -> (usize, usize) {
        let logical_lines: Vec<&str> = text.split('\n').collect();
        let mut logical_byte_start = 0usize;
        let mut visual_line_offset = 0usize;
        for logical_line in &logical_lines {
            let logical_byte_end = logical_byte_start + logical_line.len();
            if byte_pos <= logical_byte_end {
                let char_in_logical = text[logical_byte_start..byte_pos].chars().count();
                let visual = crate::ops::text::word_wrap_line(
                    logical_line,
                    font,
                    font_size,
                    max_width,
                    letter_spacing,
                    width_scale,
                    height_scale,
                );
                let (vl, vc) =
                    Self::map_char_to_visual_line(logical_line, char_in_logical, &visual);
                return (visual_line_offset + vl, vc);
            }
            let visual = crate::ops::text::word_wrap_line(
                logical_line,
                font,
                font_size,
                max_width,
                letter_spacing,
                width_scale,
                height_scale,
            );
            visual_line_offset += visual.len();
            logical_byte_start = logical_byte_end + 1;
        }
        (visual_line_offset.saturating_sub(1), 0)
    }

    /// Given a visual line index and char offset, return the byte position in the original text.
    /// The inverse of `byte_pos_to_visual`.
    fn visual_to_byte_pos(
        text: &str,
        visual_line: usize,
        char_in_line: usize,
        font: &ab_glyph::FontArc,
        font_size: f32,
        max_width: f32,
        letter_spacing: f32,
        width_scale: f32,
        height_scale: f32,
    ) -> usize {
        let logical_lines: Vec<&str> = text.split('\n').collect();
        let mut logical_byte_start = 0usize;
        let mut visual_line_offset = 0usize;
        for logical_line in &logical_lines {
            let visual = crate::ops::text::word_wrap_line(
                logical_line,
                font,
                font_size,
                max_width,
                letter_spacing,
                width_scale,
                height_scale,
            );
            let visual_count = visual.len();
            if visual_line < visual_line_offset + visual_count {
                let local_visual = visual_line - visual_line_offset;
                let char_in_logical = Self::map_visual_line_to_char(
                    logical_line,
                    &visual,
                    local_visual,
                    char_in_line,
                );
                // Convert char offset to byte offset within logical line
                let byte_in_logical: usize = logical_line
                    .chars()
                    .take(char_in_logical)
                    .map(|c| c.len_utf8())
                    .sum();
                return logical_byte_start + byte_in_logical;
            }
            visual_line_offset += visual_count;
            logical_byte_start += logical_line.len() + 1;
        }
        text.len()
    }

    /// Compute visual lines for the full text (with `\n`) using word wrapping.
    /// Returns `Vec<(String, usize)>` where each entry is `(visual_line_text, byte_start_in_original)`.
    fn compute_visual_lines_with_byte_offsets(
        text: &str,
        font: &ab_glyph::FontArc,
        font_size: f32,
        max_width: f32,
        letter_spacing: f32,
        width_scale: f32,
        height_scale: f32,
    ) -> Vec<(String, usize)> {
        let mut result = Vec::new();
        let logical_lines: Vec<&str> = text.split('\n').collect();
        let mut byte_start = 0usize;
        for logical_line in &logical_lines {
            let visual = crate::ops::text::word_wrap_line(
                logical_line,
                font,
                font_size,
                max_width,
                letter_spacing,
                width_scale,
                height_scale,
            );
            let logical_chars: Vec<char> = logical_line.chars().collect();
            let mut char_consumed = 0usize;
            for (vi, vline) in visual.iter().enumerate() {
                // byte_start + char_consumed chars converted to bytes
                let byte_off: usize = logical_chars
                    .iter()
                    .take(char_consumed)
                    .map(|c| c.len_utf8())
                    .sum();
                result.push((vline.clone(), byte_start + byte_off));
                let vlen = vline.chars().count();
                let line_end = char_consumed + vlen;
                // Skip whitespace gap
                let mut next = line_end;
                if vi < visual.len() - 1 {
                    while next < logical_chars.len() && logical_chars[next].is_whitespace() {
                        next += 1;
                    }
                }
                char_consumed = next.max(line_end);
            }
            byte_start += logical_line.len() + 1;
        }
        result
    }

    /// Draw the text tool overlay (border, move handle, blinking cursor, selection
    /// highlight, glyph overlay) when text editing is active.
    /// Also draws faint dotted outlines for non-active text blocks on text layers.
    /// Called outside the `allow_input` gate so it persists when the pointer leaves
    /// the canvas (e.g. while interacting with UI panels).
    pub fn draw_text_overlay(
        &mut self,
        ui: &egui::Ui,
        canvas_state: &CanvasState,
        painter: &egui::Painter,
        canvas_rect: Rect,
        zoom: f32,
    ) {
        let accent = ui.visuals().selection.bg_fill;

        // --- Phase 1: Draw dotted outlines for non-active blocks on text layers ---
        // Show whenever the Text tool is active on a text layer (not just when editing)
        if let Some(layer) = canvas_state.layers.get(canvas_state.active_layer_index)
            && let crate::canvas::LayerContent::Text(ref td) = layer.content
        {
            for block in &td.blocks {
                if Some(block.id) == self.text_state.active_block_id {
                    continue; // Active block gets full overlay below
                }
                // Skip truly empty blocks that have no explicit box size
                // (freshly created via click, not yet typed into or resized)
                let has_text = block.runs.iter().any(|r| !r.text.is_empty());
                if !has_text && block.max_width.is_none() {
                    continue;
                }
                let layout = crate::ops::text_layer::compute_block_layout(block);
                let bx = block.position[0];
                let by = block.position[1];
                let bw = if let Some(mw) = block.max_width {
                    mw
                } else {
                    layout.total_width
                };
                let bh = if let Some(mh) = block.max_height {
                    mh.max(layout.total_height)
                } else {
                    layout.total_height
                };
                if bw < 1.0 && bh < 1.0 {
                    continue;
                }
                let pad = 4.0;
                let sx = canvas_rect.min.x + (bx - pad) * zoom;
                let sy = canvas_rect.min.y + (by - pad) * zoom;
                let sw = (bw + pad * 2.0) * zoom;
                let sh = (bh + pad * 2.0) * zoom;
                let r = egui::Rect::from_min_size(Pos2::new(sx, sy), egui::vec2(sw, sh));
                let halo = Color32::from_rgba_premultiplied(0, 0, 0, 210);
                let light = Color32::from_rgba_premultiplied(255, 255, 255, 235);
                if block.rotation.abs() > 0.001 {
                    let center = r.center();
                    let corners = [
                        rotate_screen_point(r.left_top(), center, block.rotation),
                        rotate_screen_point(r.right_top(), center, block.rotation),
                        rotate_screen_point(r.right_bottom(), center, block.rotation),
                        rotate_screen_point(r.left_bottom(), center, block.rotation),
                    ];
                    draw_dotted_quad(painter, corners, halo, 3.0, 4.0, 3.0);
                    draw_dotted_quad(painter, corners, light, 1.0, 4.0, 3.0);
                } else {
                    // Draw dotted outline
                    draw_dotted_rect(painter, r, halo, 3.0, 4.0, 3.0);
                    draw_dotted_rect(painter, r, light, 1.0, 4.0, 3.0);
                }
            }
        }

        // --- Phase 2: Active block overlay (full handles, cursor, etc.) ---
        if !self.text_state.is_editing {
            return;
        }
        let origin = match self.text_state.origin {
            Some(o) => o,
            None => return,
        };

        let time = ui.input(|i| i.time);
        self.text_state.cursor_blink_timer = time;

        let font_size = self.text_state.font_size;
        self.ensure_text_font_loaded();
        let ls = self.text_state.letter_spacing;
        let line_height = if let Some(ref font) = self.text_state.loaded_font {
            use ab_glyph::{Font as _, ScaleFont as _};
            font.as_scaled(ab_glyph::PxScale {
                x: font_size * self.text_state.width_scale,
                y: font_size * self.text_state.height_scale,
            })
            .height()
                * self.text_state.line_spacing
        } else {
            font_size * 1.2 * self.text_state.line_spacing
        };

        let lines: Vec<&str> = self.text_state.text.split('\n').collect();
        let num_lines = lines.len().max(1);

        // Compute natural text width (from font metrics)
        let natural_width = if let Some(ref font) = self.text_state.loaded_font {
            use ab_glyph::{Font as _, ScaleFont as _};
            let scaled = font.as_scaled(ab_glyph::PxScale {
                x: font_size * self.text_state.width_scale,
                y: font_size * self.text_state.height_scale,
            });
            // If max_width is set, word-wrap to compute visual lines
            if let Some(mw) = self.text_state.active_block_max_width {
                let visual_lines: Vec<String> = lines
                    .iter()
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
                    .collect();
                let mut max_w = font_size * 2.0;
                for line in &visual_lines {
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
            }
        } else {
            font_size * 2.0
        };

        // Display width: use max_width if set, otherwise natural width
        let display_width = if let Some(mw) = self.text_state.active_block_max_width {
            mw.max(font_size * 2.0)
        } else {
            natural_width
        };

        // Compute visual line count (considering word wrap)
        let visual_num_lines = if let (Some(mw), Some(font)) = (
            self.text_state.active_block_max_width,
            &self.text_state.loaded_font,
        ) {
            let visual_lines: Vec<String> = lines
                .iter()
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
                .collect();
            visual_lines.len().max(1)
        } else {
            num_lines
        };

        let text_h = visual_num_lines as f32 * line_height;
        // Cache the active block height for input handling
        self.text_state.active_block_height = text_h;
        // Visual box height: use max_height if set (but never smaller than content)
        let visual_h = if let Some(mh) = self.text_state.active_block_max_height {
            mh.max(text_h)
        } else {
            text_h
        };
        let pad = 4.0;

        let border_x = {
            use crate::ops::text::TextAlignment;
            match self.text_state.alignment {
                TextAlignment::Left => origin[0] - pad,
                TextAlignment::Center => origin[0] - display_width * 0.5 - pad,
                TextAlignment::Right => origin[0] - display_width - pad,
            }
        };
        let border_y = origin[1] - pad;
        let border_w = display_width + pad * 2.0;
        let border_h = visual_h + pad * 2.0;

        let sx = canvas_rect.min.x + border_x * zoom;
        let sy = canvas_rect.min.y + border_y * zoom;
        let sw = border_w * zoom;
        let sh = border_h * zoom;

        let border_rect = egui::Rect::from_min_size(Pos2::new(sx, sy), egui::vec2(sw, sh));

        // Get block rotation for transforming the overlay
        let block_rotation = if let Some(layer) =
            canvas_state.layers.get(canvas_state.active_layer_index)
            && let crate::canvas::LayerContent::Text(ref td) = layer.content
            && let Some(bid) = self.text_state.active_block_id
            && let Some(block) = td.blocks.iter().find(|b| b.id == bid)
        {
            block.rotation
        } else {
            0.0
        };
        let has_rot = block_rotation.abs() > 0.001;
        let rot_pivot = border_rect.center(); // screen-space rotation center

        let accent_semi = Color32::from_rgba_unmultiplied(accent.r(), accent.g(), accent.b(), 180);
        let accent_fill = Color32::from_rgba_unmultiplied(accent.r(), accent.g(), accent.b(), 200);

        // Border outline (rotated if needed)
        if has_rot {
            let c = [
                rotate_screen_point(border_rect.left_top(), rot_pivot, block_rotation),
                rotate_screen_point(border_rect.right_top(), rot_pivot, block_rotation),
                rotate_screen_point(border_rect.right_bottom(), rot_pivot, block_rotation),
                rotate_screen_point(border_rect.left_bottom(), rot_pivot, block_rotation),
            ];
            for stroke in [
                egui::Stroke::new(4.0, Color32::from_rgba_premultiplied(0, 0, 0, 210)),
                egui::Stroke::new(2.0, Color32::WHITE),
                egui::Stroke::new(1.0, accent_semi),
            ] {
                painter.line_segment([c[0], c[1]], stroke);
                painter.line_segment([c[1], c[2]], stroke);
                painter.line_segment([c[2], c[3]], stroke);
                painter.line_segment([c[3], c[0]], stroke);
            }
        } else {
            for stroke in [
                egui::Stroke::new(4.0, Color32::from_rgba_premultiplied(0, 0, 0, 210)),
                egui::Stroke::new(2.0, Color32::WHITE),
                egui::Stroke::new(1.0, accent_semi),
            ] {
                painter.rect_stroke(border_rect, 0.0, stroke, egui::StrokeKind::Middle);
            }
        }

        // --- Resize handles, rotation handle, delete button (text layer only) ---
        if self.text_state.editing_text_layer {
            // --- Resize handles (4 corners) ---
            let handle_size = 5.0; // screen pixels
            let handle_stroke = egui::Stroke::new(1.0, Color32::WHITE);
            let corners_raw = [
                border_rect.left_top(),
                border_rect.right_top(),
                border_rect.left_bottom(),
                border_rect.right_bottom(),
            ];
            for &corner in &corners_raw {
                let c = if has_rot {
                    rotate_screen_point(corner, rot_pivot, block_rotation)
                } else {
                    corner
                };
                let hr = egui::Rect::from_center_size(
                    c,
                    egui::vec2(handle_size * 2.0, handle_size * 2.0),
                );
                painter.rect_filled(
                    hr.expand(2.0),
                    2.0,
                    Color32::from_rgba_premultiplied(0, 0, 0, 220),
                );
                painter.rect_filled(hr, 1.0, accent_fill);
                painter.rect_stroke(hr, 1.0, handle_stroke, egui::StrokeKind::Middle);
            }

            // --- Rotation handle (circle above top-center with connector line) ---
            let rot_handle_offset = 20.0; // screen pixels above the box
            let rot_handle_base = border_rect.center_top();
            let rot_center_raw =
                Pos2::new(rot_handle_base.x, rot_handle_base.y - rot_handle_offset);
            let rot_center = if has_rot {
                rotate_screen_point(rot_center_raw, rot_pivot, block_rotation)
            } else {
                rot_center_raw
            };
            let rot_base = if has_rot {
                rotate_screen_point(rot_handle_base, rot_pivot, block_rotation)
            } else {
                rot_handle_base
            };
            painter.line_segment(
                [rot_base, rot_center],
                egui::Stroke::new(3.0, Color32::from_rgba_premultiplied(0, 0, 0, 210)),
            );
            painter.line_segment(
                [rot_base, rot_center],
                egui::Stroke::new(1.0, Color32::WHITE),
            );
            painter.circle_filled(
                rot_center,
                7.5,
                Color32::from_rgba_premultiplied(0, 0, 0, 220),
            );
            painter.circle_filled(rot_center, 5.0, accent_fill);
            painter.circle_stroke(rot_center, 5.0, egui::Stroke::new(1.0, Color32::WHITE));
            // Small rotation icon (curved arrow hint)
            let arc_r = 3.0;
            painter.line_segment(
                [
                    Pos2::new(rot_center.x - arc_r, rot_center.y - 1.0),
                    Pos2::new(rot_center.x + arc_r, rot_center.y - 1.0),
                ],
                egui::Stroke::new(1.0, Color32::WHITE),
            );

            // --- Delete button (├ù at top-right corner, outside the box) ---
            let del_offset = 14.0; // screen pixels outside top-right
            let del_center_raw = Pos2::new(
                border_rect.max.x + del_offset,
                border_rect.min.y - del_offset,
            );
            let del_center = if has_rot {
                rotate_screen_point(del_center_raw, rot_pivot, block_rotation)
            } else {
                del_center_raw
            };
            let del_radius = 8.0;
            let del_bg = Color32::from_rgba_unmultiplied(200, 60, 60, 220);
            painter.circle_filled(del_center, del_radius, del_bg);
            painter.circle_stroke(
                del_center,
                del_radius,
                egui::Stroke::new(1.0, Color32::WHITE),
            );
            let xr = 3.5;
            let x_stroke = egui::Stroke::new(1.5, Color32::WHITE);
            painter.line_segment(
                [
                    Pos2::new(del_center.x - xr, del_center.y - xr),
                    Pos2::new(del_center.x + xr, del_center.y + xr),
                ],
                x_stroke,
            );
            painter.line_segment(
                [
                    Pos2::new(del_center.x + xr, del_center.y - xr),
                    Pos2::new(del_center.x - xr, del_center.y + xr),
                ],
                x_stroke,
            );
        } // end editing_text_layer handles

        // Move handle (cross circle ÔÇö existing)
        let handle_radius_screen = 6.0;
        let (handle_canvas_x, handle_canvas_y) =
            text_move_handle_canvas_pos(origin, self.text_state.alignment, font_size, zoom);
        let handle_screen_x = canvas_rect.min.x + handle_canvas_x * zoom;
        let handle_screen_y = canvas_rect.min.y + handle_canvas_y * zoom;
        let handle_pos_raw = Pos2::new(handle_screen_x, handle_screen_y);
        let handle_pos = if has_rot {
            rotate_screen_point(handle_pos_raw, rot_pivot, block_rotation)
        } else {
            handle_pos_raw
        };

        painter.circle_filled(
            handle_pos,
            handle_radius_screen + 2.5,
            Color32::from_rgba_premultiplied(0, 0, 0, 220),
        );
        painter.circle_filled(handle_pos, handle_radius_screen, accent_fill);
        painter.circle_stroke(
            handle_pos,
            handle_radius_screen,
            egui::Stroke::new(1.5, Color32::WHITE),
        );
        let hs = 3.0;
        let cross_stroke = egui::Stroke::new(1.0, Color32::WHITE);
        painter.line_segment(
            [
                Pos2::new(handle_pos.x - hs, handle_pos.y),
                Pos2::new(handle_pos.x + hs, handle_pos.y),
            ],
            cross_stroke,
        );
        painter.line_segment(
            [
                Pos2::new(handle_pos.x, handle_pos.y - hs),
                Pos2::new(handle_pos.x, handle_pos.y + hs),
            ],
            cross_stroke,
        );

        // Connector line from handle to border edge
        let connector_target_raw = {
            use crate::ops::text::TextAlignment;
            match self.text_state.alignment {
                TextAlignment::Left => Pos2::new(
                    border_rect.min.x,
                    canvas_rect.min.y + (origin[1] + font_size * 0.5) * zoom,
                ),
                TextAlignment::Center => {
                    Pos2::new(canvas_rect.min.x + origin[0] * zoom, border_rect.min.y)
                }
                TextAlignment::Right => Pos2::new(
                    border_rect.max.x,
                    canvas_rect.min.y + (origin[1] + font_size * 0.5) * zoom,
                ),
            }
        };
        let connector_target = if has_rot {
            rotate_screen_point(connector_target_raw, rot_pivot, block_rotation)
        } else {
            connector_target_raw
        };
        painter.line_segment(
            [handle_pos, connector_target],
            egui::Stroke::new(3.0, Color32::from_rgba_premultiplied(0, 0, 0, 210)),
        );
        painter.line_segment(
            [handle_pos, connector_target],
            egui::Stroke::new(1.0, Color32::WHITE),
        );

        // Selection highlight (text layer only)
        if self.text_state.editing_text_layer && self.text_state.selection.has_selection() {
            let sel_anchor_flat = {
                let a = self.text_state.selection.anchor;
                let mut off = 0usize;
                if let Some(layer) = canvas_state.layers.get(canvas_state.active_layer_index)
                    && let crate::canvas::LayerContent::Text(ref td) = layer.content
                    && let Some(bid) = self.text_state.active_block_id
                    && let Some(block) = td.blocks.iter().find(|b| b.id == bid)
                {
                    off = block.run_pos_to_flat_offset(a);
                }
                off
            };
            let sel_cursor_flat = self.text_state.cursor_pos;
            let (sel_start, sel_end) = if sel_anchor_flat <= sel_cursor_flat {
                (sel_anchor_flat, sel_cursor_flat)
            } else {
                (sel_cursor_flat, sel_anchor_flat)
            };

            let full_text = &self.text_state.text;
            let cached_lh_sel = if self.text_state.cached_line_height > 0.0 {
                self.text_state.cached_line_height
            } else {
                line_height
            };
            let sel_color = Color32::from_rgba_unmultiplied(accent.r(), accent.g(), accent.b(), 80);
            let mut drew_glyph_selection = false;

            if let Some(layer) = canvas_state.layers.get(canvas_state.active_layer_index)
                && let crate::canvas::LayerContent::Text(ref td) = layer.content
                && let Some(bid) = self.text_state.active_block_id
                && let Some(block) = td.blocks.iter().find(|b| b.id == bid)
            {
                for gb in crate::ops::text_layer::compute_glyph_bounds(block) {
                    if gb.flat_end <= sel_start || gb.flat_start >= sel_end {
                        continue;
                    }
                    let r = egui::Rect::from_min_max(
                        Pos2::new(
                            canvas_rect.min.x + gb.x * zoom,
                            canvas_rect.min.y + gb.y * zoom,
                        ),
                        Pos2::new(
                            canvas_rect.min.x + (gb.x + gb.w) * zoom,
                            canvas_rect.min.y + (gb.y + gb.h) * zoom,
                        ),
                    );
                    if has_rot {
                        let corners = [
                            rotate_screen_point(r.left_top(), rot_pivot, block_rotation),
                            rotate_screen_point(r.right_top(), rot_pivot, block_rotation),
                            rotate_screen_point(r.right_bottom(), rot_pivot, block_rotation),
                            rotate_screen_point(r.left_bottom(), rot_pivot, block_rotation),
                        ];
                        let mut mesh = egui::Mesh::with_texture(egui::TextureId::Managed(0));
                        let uv = Pos2::ZERO;
                        for &c in &corners {
                            mesh.vertices.push(egui::epaint::Vertex {
                                pos: c,
                                uv,
                                color: sel_color,
                            });
                        }
                        mesh.indices.extend_from_slice(&[0, 1, 2, 0, 2, 3]);
                        painter.add(egui::Shape::mesh(mesh));
                    } else {
                        painter.rect_filled(r, 0.0, sel_color);
                    }
                    drew_glyph_selection = true;
                }
            }

            // Compute visual lines with byte offsets for wrap-aware selection
            let visual_with_offsets: Vec<(String, usize)> = if drew_glyph_selection {
                Vec::new()
            } else if let (Some(mw), Some(font)) = (
                self.text_state.active_block_max_width,
                &self.text_state.loaded_font,
            ) {
                Self::compute_visual_lines_with_byte_offsets(
                    full_text,
                    font,
                    font_size,
                    mw,
                    ls,
                    self.text_state.width_scale,
                    self.text_state.height_scale,
                )
            } else {
                // No wrapping ÔÇö one visual line per logical line
                let mut result = Vec::new();
                let mut byte_start = 0usize;
                for line_text in full_text.split('\n') {
                    result.push((line_text.to_string(), byte_start));
                    byte_start += line_text.len() + 1;
                }
                result
            };

            for (line_idx, (line_text, line_byte_start)) in visual_with_offsets.iter().enumerate() {
                let line_byte_end = line_byte_start + line_text.len();

                let hl_start = sel_start.max(*line_byte_start);
                let hl_end = sel_end.min(line_byte_end);

                if hl_start < hl_end {
                    let compute_x = |byte_in_line: usize| -> f32 {
                        let chars_before = line_text[..byte_in_line].chars().count();
                        if !self.text_state.cached_line_advances.is_empty() {
                            self.text_state
                                .cached_line_advances
                                .get(line_idx)
                                .and_then(|a| a.get(chars_before).copied())
                                .unwrap_or(0.0)
                        } else if let Some(ref font) = self.text_state.loaded_font {
                            use ab_glyph::{Font as _, ScaleFont as _};
                            let scaled = font.as_scaled(ab_glyph::PxScale {
                                x: font_size * self.text_state.width_scale,
                                y: font_size * self.text_state.height_scale,
                            });
                            let mut x = 0.0f32;
                            let mut prev = None;
                            for ch in line_text[..byte_in_line].chars() {
                                let gid = font.glyph_id(ch);
                                if let Some(p) = prev {
                                    x += scaled.kern(p, gid);
                                }
                                x += scaled.h_advance(gid);
                                prev = Some(gid);
                            }
                            x
                        } else {
                            0.0
                        }
                    };

                    let x1 = compute_x(hl_start - line_byte_start);
                    let x2 = compute_x(hl_end - line_byte_start);

                    let line_align = {
                        use crate::ops::text::TextAlignment;
                        if self.text_state.alignment == TextAlignment::Left {
                            0.0
                        } else {
                            let line_w = if let Some(ref font) = self.text_state.loaded_font {
                                use ab_glyph::{Font as _, ScaleFont as _};
                                let scaled = font.as_scaled(ab_glyph::PxScale {
                                    x: font_size * self.text_state.width_scale,
                                    y: font_size * self.text_state.height_scale,
                                });
                                let mut w = 0.0f32;
                                let mut prev = None;
                                for ch in line_text.chars() {
                                    let gid = font.glyph_id(ch);
                                    if let Some(p) = prev {
                                        w += scaled.kern(p, gid);
                                    }
                                    w += scaled.h_advance(gid);
                                    prev = Some(gid);
                                }
                                w
                            } else {
                                0.0
                            };
                            match self.text_state.alignment {
                                TextAlignment::Center => -line_w * 0.5,
                                TextAlignment::Right => -line_w,
                                _ => 0.0,
                            }
                        }
                    };

                    let y_off = line_idx as f32 * cached_lh_sel;
                    let r = egui::Rect::from_min_max(
                        Pos2::new(
                            canvas_rect.min.x + (origin[0] + x1 + line_align) * zoom,
                            canvas_rect.min.y + (origin[1] + y_off) * zoom,
                        ),
                        Pos2::new(
                            canvas_rect.min.x + (origin[0] + x2 + line_align) * zoom,
                            canvas_rect.min.y + (origin[1] + y_off + cached_lh_sel) * zoom,
                        ),
                    );
                    if has_rot {
                        // Draw rotated selection quad
                        let corners = [
                            rotate_screen_point(r.left_top(), rot_pivot, block_rotation),
                            rotate_screen_point(r.right_top(), rot_pivot, block_rotation),
                            rotate_screen_point(r.right_bottom(), rot_pivot, block_rotation),
                            rotate_screen_point(r.left_bottom(), rot_pivot, block_rotation),
                        ];
                        let mesh = egui::Mesh::with_texture(egui::TextureId::Managed(0));
                        let mut mesh = mesh;
                        let uv = Pos2::ZERO;
                        for &c in &corners {
                            mesh.vertices.push(egui::epaint::Vertex {
                                pos: c,
                                uv,
                                color: sel_color,
                            });
                        }
                        mesh.indices.extend_from_slice(&[0, 1, 2, 0, 2, 3]);
                        painter.add(egui::Shape::mesh(mesh));
                    } else {
                        painter.rect_filled(r, 0.0, sel_color);
                    }
                }
            }
        }

        // Blinking cursor
        let blink_on = ((time * 2.0) as i32) % 2 == 0;
        if blink_on {
            let (cursor_x_offset, cursor_line) =
                if !self.text_state.text.is_empty() && self.text_state.cursor_pos > 0 {
                    // Use visual (word-wrapped) line mapping when max_width is set
                    let (vis_line, vis_char) = if let (Some(mw), Some(font)) = (
                        self.text_state.active_block_max_width,
                        &self.text_state.loaded_font,
                    ) {
                        Self::byte_pos_to_visual(
                            &self.text_state.text,
                            self.text_state.cursor_pos,
                            font,
                            font_size,
                            mw,
                            ls,
                            self.text_state.width_scale,
                            self.text_state.height_scale,
                        )
                    } else {
                        // No wrapping ÔÇö use logical line mapping
                        let text_before = &self.text_state.text[..self.text_state.cursor_pos];
                        let newlines_before = text_before.matches('\n').count();
                        let last_line = text_before.rsplit('\n').next().unwrap_or(text_before);
                        (newlines_before, last_line.chars().count())
                    };

                    let x_off = if !self.text_state.cached_line_advances.is_empty() {
                        self.text_state
                            .cached_line_advances
                            .get(vis_line)
                            .and_then(|advances| advances.get(vis_char).copied())
                            .unwrap_or(0.0)
                    } else if let Some(ref font) = self.text_state.loaded_font {
                        // Fallback: compute from font metrics for the visual line text
                        use ab_glyph::{Font as _, ScaleFont as _};
                        let scaled = font.as_scaled(ab_glyph::PxScale {
                            x: font_size * self.text_state.width_scale,
                            y: font_size * self.text_state.height_scale,
                        });
                        // Get the visual line text to measure
                        let text_before = &self.text_state.text[..self.text_state.cursor_pos];
                        let last_line = text_before.rsplit('\n').next().unwrap_or(text_before);
                        let mut x = 0.0f32;
                        let mut prev_glyph_id = None;
                        for ch in last_line.chars() {
                            let glyph_id = font.glyph_id(ch);
                            if let Some(prev) = prev_glyph_id {
                                x += scaled.kern(prev, glyph_id);
                            }
                            x += scaled.h_advance(glyph_id);
                            prev_glyph_id = Some(glyph_id);
                        }
                        x
                    } else {
                        0.0
                    };
                    (x_off, vis_line)
                } else {
                    (0.0, 0)
                };

            let cached_lh = if self.text_state.cached_line_height > 0.0 {
                self.text_state.cached_line_height
            } else {
                line_height
            };
            let cursor_y_offset = cursor_line as f32 * cached_lh;

            let align_offset = {
                use crate::ops::text::TextAlignment;
                if self.text_state.alignment == TextAlignment::Left {
                    0.0
                } else {
                    // Get current visual line text for alignment measurement
                    let current_line_text: String = if let (Some(mw), Some(font)) = (
                        self.text_state.active_block_max_width,
                        &self.text_state.loaded_font,
                    ) {
                        let all_visual: Vec<String> = self
                            .text_state
                            .text
                            .split('\n')
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
                            .collect();
                        all_visual.get(cursor_line).cloned().unwrap_or_default()
                    } else {
                        lines.get(cursor_line).unwrap_or(&"").to_string()
                    };
                    let line_w = if let Some(ref font) = self.text_state.loaded_font {
                        use ab_glyph::{Font as _, ScaleFont as _};
                        let scaled = font.as_scaled(ab_glyph::PxScale {
                            x: font_size * self.text_state.width_scale,
                            y: font_size * self.text_state.height_scale,
                        });
                        let mut w = 0.0f32;
                        let mut prev = None;
                        for ch in current_line_text.chars() {
                            let gid = font.glyph_id(ch);
                            if let Some(prev_id) = prev {
                                w += scaled.kern(prev_id, gid);
                            }
                            w += scaled.h_advance(gid);
                            prev = Some(gid);
                        }
                        w
                    } else {
                        0.0
                    };
                    match self.text_state.alignment {
                        TextAlignment::Center => -line_w * 0.5,
                        TextAlignment::Right => -line_w,
                        _ => 0.0,
                    }
                }
            };

            let cx = origin[0] + cursor_x_offset + align_offset;
            let cy = origin[1] + cursor_y_offset;

            let csx = canvas_rect.min.x + cx * zoom;
            let csy = canvas_rect.min.y + cy * zoom;
            let cursor_h = font_size * self.text_state.height_scale * zoom;

            let cursor_top = Pos2::new(csx, csy);
            let cursor_bot = Pos2::new(csx, csy + cursor_h);
            let cursor_top2 = Pos2::new(csx + 1.0, csy);
            let cursor_bot2 = Pos2::new(csx + 1.0, csy + cursor_h);

            let (ct, cb, ct2, cb2) = if has_rot {
                (
                    rotate_screen_point(cursor_top, rot_pivot, block_rotation),
                    rotate_screen_point(cursor_bot, rot_pivot, block_rotation),
                    rotate_screen_point(cursor_top2, rot_pivot, block_rotation),
                    rotate_screen_point(cursor_bot2, rot_pivot, block_rotation),
                )
            } else {
                (cursor_top, cursor_bot, cursor_top2, cursor_bot2)
            };

            painter.line_segment([ct, cb], egui::Stroke::new(1.5, Color32::BLACK));
            painter.line_segment([ct2, cb2], egui::Stroke::new(0.5, Color32::WHITE));
        }
        // Throttle repaint to cursor blink rate (2Hz)
        ui.ctx()
            .request_repaint_after(std::time::Duration::from_millis(500));
    }
}
