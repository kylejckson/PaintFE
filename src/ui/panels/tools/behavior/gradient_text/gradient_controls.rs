impl ToolsPanel {
    pub fn update_shape_if_dirty(
        &mut self,
        canvas_state: &mut CanvasState,
        primary_color_f32: [f32; 4],
        secondary_color_f32: [f32; 4],
    ) {
        if self.shapes_state.placed.is_none() {
            return;
        }
        let p = self.shapes_state.placed.as_mut().unwrap();
        let current_primary = [
            (primary_color_f32[0] * 255.0) as u8,
            (primary_color_f32[1] * 255.0) as u8,
            (primary_color_f32[2] * 255.0) as u8,
            (primary_color_f32[3] * 255.0) as u8,
        ];
        let current_secondary = [
            (secondary_color_f32[0] * 255.0) as u8,
            (secondary_color_f32[1] * 255.0) as u8,
            (secondary_color_f32[2] * 255.0) as u8,
            (secondary_color_f32[3] * 255.0) as u8,
        ];
        let changed = p.kind != self.shapes_state.selected_shape
            || p.fill_mode != self.shapes_state.fill_mode
            || p.outline_width != self.properties.size
            || p.anti_alias != self.shapes_state.anti_alias
            || p.corner_radius != self.shapes_state.corner_radius
            || p.primary_color != current_primary
            || p.secondary_color != current_secondary;
        if changed {
            p.kind = self.shapes_state.selected_shape;
            p.fill_mode = self.shapes_state.fill_mode;
            p.outline_width = self.properties.size;
            p.anti_alias = self.shapes_state.anti_alias;
            p.corner_radius = self.shapes_state.corner_radius;
            p.primary_color = current_primary;
            p.secondary_color = current_secondary;
            self.render_shape_preview(canvas_state, primary_color_f32, secondary_color_f32);
        }
    }

    /// Commit the gradient preview to the active layer.
    fn commit_gradient(&mut self, canvas_state: &mut CanvasState) {
        if self.gradient_state.drag_start.is_none() {
            return;
        }

        let Some(target_layer_idx) = self.gradient_state.source_layer_index else {
            return;
        };
        if target_layer_idx >= canvas_state.layers.len() {
            self.cancel_gradient(canvas_state);
            return;
        }

        // Capture "before" snapshot BEFORE modifying the layer
        let stroke_event = self.stroke_tracker.finish(canvas_state);

        let is_eraser = self.gradient_state.mode == GradientMode::Transparency;

        // Commit preview to actual layer with proper alpha blending
        if let Some(ref preview) = canvas_state.preview_layer {
            let chunk_data: Vec<(u32, u32, image::RgbaImage)> = preview
                .chunk_keys()
                .filter_map(|(cx, cy)| {
                    preview
                        .get_chunk(cx, cy)
                        .map(|chunk| (cx, cy, chunk.clone()))
                })
                .collect();

            let chunk_size = CHUNK_SIZE;

            if let Some(active_layer) = canvas_state.layers.get_mut(target_layer_idx) {
                for (cx, cy, chunk) in &chunk_data {
                    let base_x = cx * chunk_size;
                    let base_y = cy * chunk_size;
                    let cw = chunk_size.min(canvas_state.width.saturating_sub(base_x));
                    let ch = chunk_size.min(canvas_state.height.saturating_sub(base_y));
                    for ly in 0..ch {
                        for lx in 0..cw {
                            let gx = base_x + lx;
                            let gy = base_y + ly;
                            let src = chunk.get_pixel(lx, ly);
                            if src.0[3] == 0 {
                                continue;
                            }
                            let dst = active_layer.pixels.get_pixel_mut(gx, gy);

                            if is_eraser {
                                // Eraser: reduce layer alpha by mask strength
                                let mask_strength = src.0[3] as f32 / 255.0;
                                let current_a = dst.0[3] as f32 / 255.0;
                                let new_a = (current_a * (1.0 - mask_strength)).max(0.0);
                                dst.0[3] = (new_a * 255.0).round() as u8;
                            } else {
                                // Normal alpha-over blend
                                let sa = src.0[3] as f32 / 255.0;
                                let da = dst.0[3] as f32 / 255.0;
                                let out_a = sa + da * (1.0 - sa);
                                if out_a > 0.0 {
                                    let r = (src.0[0] as f32 * sa
                                        + dst.0[0] as f32 * da * (1.0 - sa))
                                        / out_a;
                                    let g = (src.0[1] as f32 * sa
                                        + dst.0[1] as f32 * da * (1.0 - sa))
                                        / out_a;
                                    let b = (src.0[2] as f32 * sa
                                        + dst.0[2] as f32 * da * (1.0 - sa))
                                        / out_a;
                                    *dst = Rgba([
                                        r.round() as u8,
                                        g.round() as u8,
                                        b.round() as u8,
                                        (out_a * 255.0).round() as u8,
                                    ]);
                                }
                            }
                        }
                    }
                }
                active_layer.invalidate_lod();
                active_layer.gpu_generation += 1;
            }
        }

        // Clear gradient state
        self.gradient_state.drag_start = None;
        self.gradient_state.drag_end = None;
        self.gradient_state.dragging = false;
        self.gradient_state.dragging_handle = None;
        self.gradient_state.source_layer_index = None;

        // Mark dirty and clear preview
        canvas_state.clear_preview_state();
        canvas_state.mark_dirty(None);

        // Store event for history
        if stroke_event.is_some() {
            self.pending_stroke_event = stroke_event;
        }
    }

    /// Cancel gradient operation without committing.
    fn cancel_gradient(&mut self, canvas_state: &mut CanvasState) {
        self.gradient_state.drag_start = None;
        self.gradient_state.drag_end = None;
        self.gradient_state.dragging = false;
        self.gradient_state.dragging_handle = None;
        self.gradient_state.source_layer_index = None;
        self.stroke_tracker.cancel();
        canvas_state.clear_preview_state();
        canvas_state.mark_dirty(None);
    }

    /// Draw gradient handle overlay (start/end points and connecting line).
    fn draw_gradient_overlay(
        &self,
        painter: &egui::Painter,
        canvas_rect: Rect,
        zoom: f32,
        canvas_state: &CanvasState,
    ) {
        let (start, end) = match (self.gradient_state.drag_start, self.gradient_state.drag_end) {
            (Some(s), Some(e)) => (s, e),
            _ => return,
        };

        // Convert canvas coords to screen coords
        let to_screen = |cx: f32, cy: f32| -> Pos2 {
            let image_rect = egui::Rect::from_min_size(
                canvas_rect.min,
                egui::vec2(
                    canvas_state.width as f32 * zoom,
                    canvas_state.height as f32 * zoom,
                ),
            );
            Pos2::new(image_rect.min.x + cx * zoom, image_rect.min.y + cy * zoom)
        };

        let screen_start = to_screen(start.x, start.y);
        let screen_end = to_screen(end.x, end.y);

        // Dashed line connecting start Ôåö end
        let dash_len = 6.0;
        let gap_len = 4.0;
        let line_vec = screen_end - screen_start;
        let line_len = line_vec.length();
        if line_len > 1.0 {
            let dir = line_vec / line_len;
            let mut d = 0.0;
            while d < line_len {
                let seg_start = screen_start + dir * d;
                let seg_end = screen_start + dir * (d + dash_len).min(line_len);
                painter.line_segment([seg_start, seg_end], egui::Stroke::new(1.5, Color32::WHITE));
                painter.line_segment([seg_start, seg_end], egui::Stroke::new(0.5, Color32::BLACK));
                d += dash_len + gap_len;
            }
        }

        // Start handle ÔÇö filled circle with outline
        let handle_r = 5.0;
        painter.circle_filled(screen_start, handle_r + 1.0, Color32::BLACK);
        painter.circle_filled(screen_start, handle_r, Color32::WHITE);
        painter.circle_filled(
            screen_start,
            handle_r - 2.0,
            Color32::from_rgb(100, 180, 255),
        );

        // End handle ÔÇö filled circle with outline
        painter.circle_filled(screen_end, handle_r + 1.0, Color32::BLACK);
        painter.circle_filled(screen_end, handle_r, Color32::WHITE);
        painter.circle_filled(screen_end, handle_r - 2.0, Color32::from_rgb(255, 100, 100));
    }

    /// Context bar: gradient shape, mode, preset, repeat, and gradient strip.
    fn show_gradient_options(
        &mut self,
        ui: &mut egui::Ui,
        assets: &Assets,
        primary_color: Color32,
        secondary_color: Color32,
    ) {
        // Cache primary color for the "Use Primary" button in the gradient bar
        self.gradient_state.cached_primary = Some([
            primary_color.r(),
            primary_color.g(),
            primary_color.b(),
            primary_color.a(),
        ]);

        // Shape dropdown
        ui.label(t!("ctx.shapes.shape"));
        let current_shape = self.gradient_state.shape;
        egui::ComboBox::from_id_salt("gradient_shape")
            .selected_text(crate::i18n::translate_english(&current_shape.label()))
            .width(120.0)
            .show_ui(ui, |ui| {
                for &shape in GradientShape::all() {
                    if ui
                        .selectable_label(shape == current_shape, crate::i18n::translate_english(&shape.label()))
                        .clicked()
                    {
                        self.gradient_state.shape = shape;
                        self.gradient_state.preview_dirty = true;
                    }
                }
            });

        ui.separator();

        // Mode toggle
        ui.label(t!("ctx.mode"));
        let mode = self.gradient_state.mode;
        if ui
            .selectable_label(mode == GradientMode::Color, crate::i18n::translate_english(&GradientMode::Color.label()))
            .clicked()
        {
            self.gradient_state.mode = GradientMode::Color;
            self.gradient_state.lut_dirty = true;
            self.gradient_state.preview_dirty = true;
        }
        if ui
            .selectable_label(
                mode == GradientMode::Transparency,
                crate::i18n::translate_english(&GradientMode::Transparency.label()),
            )
            .clicked()
        {
            self.gradient_state.mode = GradientMode::Transparency;
            self.gradient_state.lut_dirty = true;
            self.gradient_state.preview_dirty = true;
        }

        ui.separator();

        // Preset dropdown
        ui.label(t!("ctx.gradient.preset"));
        let current_preset = self.gradient_state.preset;
        let primary_u8 = [
            primary_color.r(),
            primary_color.g(),
            primary_color.b(),
            primary_color.a(),
        ];
        let secondary_u8 = [
            secondary_color.r(),
            secondary_color.g(),
            secondary_color.b(),
            secondary_color.a(),
        ];
        egui::ComboBox::from_id_salt("gradient_preset")
            .selected_text(crate::i18n::translate_english(&current_preset.label()))
            .width(150.0)
            .show_ui(ui, |ui| {
                for &preset in GradientPreset::all() {
                    if ui
                        .selectable_label(preset == current_preset, crate::i18n::translate_english(&preset.label()))
                        .clicked()
                    {
                        self.gradient_state
                            .apply_preset(preset, primary_u8, secondary_u8);
                        self.gradient_state.preview_dirty = true;
                    }
                }
            });

        ui.separator();

        // Repeat checkbox
        let mut repeat = self.gradient_state.repeat;
        if ui
            .checkbox(&mut repeat, t!("ctx.gradient.repeat"))
            .changed()
        {
            self.gradient_state.repeat = repeat;
            self.gradient_state.preview_dirty = true;
        }

        ui.separator();

        // Gradient strip preview + stop editor
        self.show_gradient_bar(ui, assets);
    }

    /// Draw the interactive gradient bar with draggable color stops.
    fn show_gradient_bar(&mut self, ui: &mut egui::Ui, assets: &Assets) {
        let bar_width = 200.0f32;
        let bar_height = 12.0f32;
        let stop_radius = 4.0f32;
        let h_pad = stop_radius + 2.0; // horizontal padding so edge handles aren't clipped

        let (response, painter) = ui.allocate_painter(
            egui::vec2(bar_width + h_pad * 2.0, bar_height + stop_radius + 4.0),
            egui::Sense::click_and_drag(),
        );
        let bar_rect = egui::Rect::from_min_size(
            response.rect.min + egui::vec2(h_pad, 1.0),
            egui::vec2(bar_width, bar_height),
        );

        // Draw checkerboard behind bar (for transparency visualization)
        let check_size = 4.0;
        let cols = (bar_width / check_size).ceil() as usize;
        let rows = (bar_height / check_size).ceil() as usize;
        for row in 0..rows {
            for col in 0..cols {
                let color = if (row + col) % 2 == 0 {
                    Color32::from_gray(200)
                } else {
                    Color32::from_gray(255)
                };
                let r = egui::Rect::from_min_size(
                    bar_rect.min + egui::vec2(col as f32 * check_size, row as f32 * check_size),
                    egui::vec2(check_size, check_size),
                )
                .intersect(bar_rect);
                painter.rect_filled(r, 0.0, color);
            }
        }

        // Rebuild LUT if needed for display
        if self.gradient_state.lut_dirty {
            self.gradient_state.rebuild_lut();
        }

        // Draw gradient bar using LUT
        let lut = &self.gradient_state.lut;
        for x in 0..bar_width as usize {
            let t = x as f32 / (bar_width - 1.0);
            let idx = (t * 255.0).round() as usize;
            let off = idx * 4;
            let color =
                Color32::from_rgba_unmultiplied(lut[off], lut[off + 1], lut[off + 2], lut[off + 3]);
            let line_rect = egui::Rect::from_min_size(
                bar_rect.min + egui::vec2(x as f32, 0.0),
                egui::vec2(1.0, bar_height),
            );
            painter.rect_filled(line_rect, 0.0, color);
        }

        // Outline
        painter.rect_stroke(
            bar_rect,
            1.0,
            egui::Stroke::new(1.0, Color32::DARK_GRAY),
            egui::StrokeKind::Middle,
        );

        // Draw stop handles
        let stop_y = bar_rect.max.y + stop_radius + 1.0;
        for (i, stop) in self.gradient_state.stops.iter().enumerate() {
            let stop_x = bar_rect.min.x + stop.position * bar_width;
            let _pos = Pos2::new(stop_x, stop_y);
            let is_selected = self.gradient_state.selected_stop == Some(i);

            // Triangle pointing up at the bar
            let tri_top = Pos2::new(stop_x, bar_rect.max.y + 1.0);
            let tri_left = Pos2::new(stop_x - stop_radius, stop_y);
            let tri_right = Pos2::new(stop_x + stop_radius, stop_y);

            // Outline
            let outline_color = if is_selected {
                Color32::WHITE
            } else {
                Color32::DARK_GRAY
            };
            painter.add(egui::Shape::convex_polygon(
                vec![tri_top, tri_right, tri_left],
                Color32::from_rgba_unmultiplied(
                    stop.color[0],
                    stop.color[1],
                    stop.color[2],
                    stop.color[3],
                ),
                egui::Stroke::new(if is_selected { 2.0 } else { 1.0 }, outline_color),
            ));
        }

        // Handle interactions
        let pointer_pos = response.interact_pointer_pos();
        if let Some(pp) = pointer_pos {
            let t_at_pointer = ((pp.x - bar_rect.min.x) / bar_width).clamp(0.0, 1.0);

            if response.drag_started() {
                // Check if clicking near an existing stop
                let mut hit_stop: Option<usize> = None;
                for (i, stop) in self.gradient_state.stops.iter().enumerate() {
                    let stop_screen_x = bar_rect.min.x + stop.position * bar_width;
                    if (pp.x - stop_screen_x).abs() < stop_radius * 2.0 {
                        hit_stop = Some(i);
                        break;
                    }
                }
                if let Some(idx) = hit_stop {
                    self.gradient_state.selected_stop = Some(idx);
                } else {
                    // Add new stop at click position
                    let new_color = self.gradient_state.sample_lut(t_at_pointer);
                    self.gradient_state
                        .stops
                        .push(GradientStop::new(t_at_pointer, new_color));
                    self.gradient_state
                        .stops
                        .sort_by(|a, b| a.position.partial_cmp(&b.position).unwrap());
                    // Find the newly added stop index
                    for (i, s) in self.gradient_state.stops.iter().enumerate() {
                        if (s.position - t_at_pointer).abs() < 0.001 {
                            self.gradient_state.selected_stop = Some(i);
                            break;
                        }
                    }
                    self.gradient_state.preset = GradientPreset::Custom;
                    self.gradient_state.lut_dirty = true;
                    self.gradient_state.preview_dirty = true;
                }
            }

            // Drag selected stop
            if response.dragged()
                && let Some(sel) = self.gradient_state.selected_stop
                && sel < self.gradient_state.stops.len()
            {
                // Don't allow dragging first/last stops past each other
                let new_pos = t_at_pointer.clamp(0.0, 1.0);
                self.gradient_state.stops[sel].position = new_pos;
                // Re-sort and update selected index
                let sel_stop_pos = self.gradient_state.stops[sel].position;
                let sel_stop_color = self.gradient_state.stops[sel].color;
                self.gradient_state
                    .stops
                    .sort_by(|a, b| a.position.partial_cmp(&b.position).unwrap());
                for (i, s) in self.gradient_state.stops.iter().enumerate() {
                    if (s.position - sel_stop_pos).abs() < 1e-6 && s.color == sel_stop_color {
                        self.gradient_state.selected_stop = Some(i);
                        break;
                    }
                }
                self.gradient_state.preset = GradientPreset::Custom;
                self.gradient_state.lut_dirty = true;
                self.gradient_state.preview_dirty = true;
            }
        }

        // Right-click to delete a stop (if more than 2)
        if response.secondary_clicked()
            && let Some(pp) = ui.input(|i| i.pointer.latest_pos())
            && self.gradient_state.stops.len() > 2
        {
            let mut closest_idx: Option<usize> = None;
            let mut closest_dist = f32::MAX;
            for (i, stop) in self.gradient_state.stops.iter().enumerate() {
                let stop_screen_x = bar_rect.min.x + stop.position * bar_width;
                let d = (pp.x - stop_screen_x).abs();
                if d < stop_radius * 3.0 && d < closest_dist {
                    closest_dist = d;
                    closest_idx = Some(i);
                }
            }
            if let Some(idx) = closest_idx {
                self.gradient_state.stops.remove(idx);
                self.gradient_state.selected_stop = None;
                self.gradient_state.preset = GradientPreset::Custom;
                self.gradient_state.lut_dirty = true;
                self.gradient_state.preview_dirty = true;
            }
        }

        // Color edit for selected stop ÔÇö compact swatch + "Use Primary" button
        if let Some(sel) = self.gradient_state.selected_stop
            && sel < self.gradient_state.stops.len()
        {
            ui.separator();

            let stop_color = self.gradient_state.stops[sel].color;
            let preview_color = Color32::from_rgba_unmultiplied(
                stop_color[0],
                stop_color[1],
                stop_color[2],
                stop_color[3],
            );

            ui.horizontal(|ui| {
                // Color swatch (small)
                let (swatch_rect, _) =
                    ui.allocate_exact_size(egui::vec2(16.0, 16.0), egui::Sense::hover());
                let p = ui.painter_at(swatch_rect);
                let cs = 4.0;
                for row in 0..4 {
                    for col in 0..4 {
                        let c = if (row + col) % 2 == 0 {
                            Color32::from_gray(200)
                        } else {
                            Color32::WHITE
                        };
                        let r = egui::Rect::from_min_size(
                            swatch_rect.min + egui::vec2(col as f32 * cs, row as f32 * cs),
                            egui::vec2(cs, cs),
                        )
                        .intersect(swatch_rect);
                        p.rect_filled(r, 0.0, c);
                    }
                }
                p.rect_filled(swatch_rect, 2.0, preview_color);
                p.rect_stroke(
                    swatch_rect,
                    2.0,
                    egui::Stroke::new(1.0, Color32::DARK_GRAY),
                    egui::StrokeKind::Middle,
                );

                // Hex label
                ui.label(format!(
                    "#{:02X}{:02X}{:02X}{:02X}",
                    stop_color[0], stop_color[1], stop_color[2], stop_color[3],
                ));

                // "Use Primary" button ÔÇö applies the current primary color to this stop
                if assets
                    .menu_item(ui, Icon::ApplyPrimary, &t!("ctx.gradient.apply_primary"))
                    .clicked()
                    && let Some(pc) = self.gradient_state.cached_primary
                {
                    self.gradient_state.stops[sel].color = pc;
                    let c32 = Color32::from_rgba_unmultiplied(pc[0], pc[1], pc[2], pc[3]);
                    self.gradient_state.stops[sel].hsv = crate::components::colors::color_to_hsv(c32);
                    self.gradient_state.preset = GradientPreset::Custom;
                    self.gradient_state.lut_dirty = true;
                    self.gradient_state.preview_dirty = true;
                }
            });

            // Hint for the user
            ui.label(
                egui::RichText::new(t!("ctx.gradient.color_hint"))
                    .weak()
                    .small(),
            );
        }
    }
}

