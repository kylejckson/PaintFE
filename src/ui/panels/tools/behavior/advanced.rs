impl ToolsPanel {
    fn show_liquify_options(&mut self, ui: &mut egui::Ui) {
        ui.label(t!("ctx.mode"));
        egui::ComboBox::from_id_salt("ctx_liquify_mode")
            .selected_text(self.liquify_state.mode.label())
            .width(90.0)
            .show_ui(ui, |ui| {
                for mode in LiquifyMode::all() {
                    if ui
                        .selectable_label(self.liquify_state.mode == *mode, mode.label())
                        .clicked()
                    {
                        self.liquify_state.mode = *mode;
                    }
                }
            });

        ui.separator();
        ui.label(t!("ctx.liquify.strength"));
        let mut strength_pct = (self.liquify_state.strength * 100.0).round() as i32;
        if ui
            .add(
                egui::DragValue::new(&mut strength_pct)
                    .speed(1)
                    .range(1..=100)
                    .suffix("%"),
            )
            .changed()
        {
            self.liquify_state.strength = strength_pct as f32 / 100.0;
        }

        ui.separator();
        ui.label(t!("ctx.size"));
        ui.add(
            egui::DragValue::new(&mut self.properties.size)
                .speed(0.5)
                .range(5.0..=500.0)
                .suffix("px"),
        );

        if self.liquify_state.is_active {
            ui.separator();
            ui.label(
                egui::RichText::new(t!("ctx.press_enter_to_apply"))
                    .weak()
                    .italics(),
            );
        }
    }

    fn commit_liquify(&mut self, canvas_state: &mut CanvasState) {
        let displacement = match &self.liquify_state.displacement {
            Some(d) => d,
            None => {
                self.stroke_tracker.cancel();
                return;
            }
        };
        let source = match &self.liquify_state.source_snapshot {
            Some(s) => s,
            None => {
                self.stroke_tracker.cancel();
                return;
            }
        };
        let Some(target_layer_idx) = self.liquify_state.source_layer_index else {
            self.stroke_tracker.cancel();
            return;
        };
        if target_layer_idx >= canvas_state.layers.len() {
            self.liquify_state.displacement = None;
            self.liquify_state.is_active = false;
            self.liquify_state.source_snapshot = None;
            self.liquify_state.source_layer_index = None;
            self.liquify_state.warp_buffer.clear();
            self.liquify_state.dirty_rect = None;
            self.stroke_tracker.cancel();
            canvas_state.clear_preview_state();
            return;
        }
        let selection_mask = canvas_state.selection_mask.clone();
        let mut warped = crate::ops::transform::warp_displacement_full(source, displacement);
        if let Some(mask) = selection_mask.as_ref() {
            let w = warped.width().min(source.width());
            let h = warped.height().min(source.height());
            for y in 0..h {
                for x in 0..w {
                    if x >= mask.width() || y >= mask.height() || mask.get_pixel(x, y).0[0] == 0 {
                        *warped.get_pixel_mut(x, y) = *source.get_pixel(x, y);
                    }
                }
            }
        }
        let mut history_cmd = crate::components::history::SingleLayerSnapshotCommand::new_for_layer(
            "Liquify".to_string(),
            canvas_state,
            target_layer_idx,
        );
        let changed = warped.as_raw() != source.as_raw();

        if changed && let Some(active_layer) = canvas_state.layers.get_mut(target_layer_idx) {
            let w = canvas_state.width.min(warped.width());
            let h = canvas_state.height.min(warped.height());
            for y in 0..h {
                for x in 0..w {
                    active_layer.pixels.put_pixel(x, y, *warped.get_pixel(x, y));
                }
            }
            active_layer.invalidate_lod();
            active_layer.gpu_generation += 1;
        }

        self.liquify_state.displacement = None;
        self.liquify_state.is_active = false;
        self.liquify_state.source_snapshot = None;
        self.liquify_state.source_layer_index = None;
        self.liquify_state.warp_buffer.clear();
        self.liquify_state.dirty_rect = None;
        self.stroke_tracker.cancel();
        canvas_state.clear_preview_state();
        if changed {
            canvas_state.mark_dirty(None);
            history_cmd.set_after(canvas_state);
            self.pending_history_commands.push(Box::new(history_cmd));
        }

        self.pending_stroke_event = None;
    }

    fn render_liquify_preview(
        &mut self,
        canvas_state: &mut CanvasState,
        gpu_renderer: Option<&mut crate::gpu::GpuRenderer>,
    ) {
        let displacement = match &self.liquify_state.displacement {
            Some(d) => d,
            None => return,
        };
        let source = match &self.liquify_state.source_snapshot {
            Some(s) => s,
            None => return,
        };

        let w = canvas_state.width;
        let h = canvas_state.height;
        let total_pixels = w as usize * h as usize;

        // ── GPU path ─────────────────────────────────────────────
        if let Some(gpu) = gpu_renderer {
            gpu.liquify_pipeline.warp_into(
                &gpu.ctx,
                source.as_raw(),
                &displacement.data,
                w,
                h,
                &mut self.liquify_state.warp_buffer,
            );
        } else {
            // ── CPU fallback ─────────────────────────────────────
            let dirty = self
                .liquify_state
                .dirty_rect
                .unwrap_or([0, 0, w as i32, h as i32]);
            let pad = self.properties.size as i32 + 20;
            let dr = [
                (dirty[0] - pad).max(0),
                (dirty[1] - pad).max(0),
                (dirty[2] + pad).min(w as i32),
                (dirty[3] + pad).min(h as i32),
            ];

            if self.liquify_state.warp_buffer.len() != total_pixels * 4 {
                self.liquify_state.warp_buffer = source.as_raw().clone();
            }

            let warped = crate::ops::transform::warp_displacement_region(
                source,
                displacement,
                &self.liquify_state.warp_buffer,
                (dr[0], dr[1], dr[2], dr[3]),
                w,
                h,
            );
            self.liquify_state.warp_buffer = warped;
        }

        if let Some(mask) = canvas_state.selection_mask.as_ref() {
            let src_raw = source.as_raw();
            for y in 0..h {
                for x in 0..w {
                    if x < mask.width() && y < mask.height() && mask.get_pixel(x, y).0[0] != 0 {
                        continue;
                    }
                    let idx = ((y * w + x) * 4) as usize;
                    self.liquify_state.warp_buffer[idx..idx + 4]
                        .copy_from_slice(&src_raw[idx..idx + 4]);
                }
            }
        }

        let preview = TiledImage::from_raw_rgba(w, h, &self.liquify_state.warp_buffer);
        canvas_state.preview_layer = Some(preview);
        canvas_state.preview_blend_mode = BlendMode::Normal;
        canvas_state.preview_force_composite = true;
        canvas_state.preview_is_eraser = false;
        canvas_state.preview_replaces_layer = true;
        canvas_state.preview_downscale = 1;
        canvas_state.preview_stroke_bounds = Some(egui::Rect::from_min_max(
            egui::pos2(0.0, 0.0),
            egui::pos2(w as f32, h as f32),
        ));
        canvas_state.preview_texture_cache = None; // Force re-upload
        canvas_state.mark_dirty(None);
    }

    // ====================================================================
    // MESH WARP TOOL — context bar, preview, commit, overlay
    // ====================================================================

    fn show_mesh_warp_options(&mut self, ui: &mut egui::Ui) {
        ui.label(t!("ctx.mesh_warp.grid"));
        let grid_label = format!(
            "{}×{}",
            self.mesh_warp_state.grid_cols, self.mesh_warp_state.grid_rows
        );
        egui::ComboBox::from_id_salt("ctx_meshwarp_grid")
            .selected_text(&grid_label)
            .width(60.0)
            .show_ui(ui, |ui| {
                for n in &[2usize, 3, 4, 5, 6] {
                    let label = format!("{}×{}", n, n);
                    if ui
                        .selectable_label(self.mesh_warp_state.grid_cols == *n, &label)
                        .clicked()
                    {
                        self.mesh_warp_state.grid_cols = *n;
                        self.mesh_warp_state.grid_rows = *n;
                        self.mesh_warp_state.is_active = false;
                        self.mesh_warp_state.points.clear();
                        self.mesh_warp_state.original_points.clear();
                        self.mesh_warp_state.source_snapshot = None;
                        self.mesh_warp_state.warp_buffer.clear();
                        self.mesh_warp_state.needs_reinit = true;
                    }
                }
            });

        if self.mesh_warp_state.is_active {
            ui.separator();
            ui.label(
                egui::RichText::new(t!("ctx.press_enter_to_apply"))
                    .weak()
                    .italics(),
            );
        }
    }

    fn init_mesh_warp_grid(&mut self, canvas_state: &CanvasState) {
        let cols = self.mesh_warp_state.grid_cols;
        let rows = self.mesh_warp_state.grid_rows;
        let w = canvas_state.width as f32;
        let h = canvas_state.height as f32;

        let mut points = Vec::with_capacity((cols + 1) * (rows + 1));
        for r in 0..=rows {
            for c in 0..=cols {
                points.push([c as f32 / cols as f32 * w, r as f32 / rows as f32 * h]);
            }
        }
        self.mesh_warp_state.original_points = points.clone();
        self.mesh_warp_state.points = points;
        self.mesh_warp_state.is_active = true;

        let idx = canvas_state.active_layer_index;
        if let Some(layer) = canvas_state.layers.get(idx) {
            self.mesh_warp_state.source_snapshot = Some(layer.pixels.to_rgba_image());
            self.mesh_warp_state.snapshot_layer_index = idx;
            self.mesh_warp_state.snapshot_generation = layer.gpu_generation;
        }
    }

    fn commit_mesh_warp(&mut self, canvas_state: &mut CanvasState) {
        let source = match &self.mesh_warp_state.source_snapshot {
            Some(s) => s,
            None => return,
        };
        let target_layer_idx = self.mesh_warp_state.snapshot_layer_index;
        if target_layer_idx >= canvas_state.layers.len() {
            self.mesh_warp_state.is_active = false;
            self.mesh_warp_state.points.clear();
            self.mesh_warp_state.original_points.clear();
            self.mesh_warp_state.source_snapshot = None;
            self.mesh_warp_state.warp_buffer.clear();
            canvas_state.clear_preview_state();
            return;
        }

        // Expand undo/redo bounds to cover the full canvas
        self.stroke_tracker.expand_bounds(egui::Rect::from_min_max(
            egui::pos2(0.0, 0.0),
            egui::pos2(canvas_state.width as f32, canvas_state.height as f32),
        ));

        let stroke_event = self.stroke_tracker.finish(canvas_state);
        let warped = crate::ops::transform::warp_mesh_catmull_rom(
            source,
            &self.mesh_warp_state.original_points,
            &self.mesh_warp_state.points,
            self.mesh_warp_state.grid_cols,
            self.mesh_warp_state.grid_rows,
            canvas_state.width,
            canvas_state.height,
        );

        if let Some(active_layer) = canvas_state.layers.get_mut(target_layer_idx) {
            let w = canvas_state.width.min(warped.width());
            let h = canvas_state.height.min(warped.height());
            for y in 0..h {
                for x in 0..w {
                    active_layer.pixels.put_pixel(x, y, *warped.get_pixel(x, y));
                }
            }
            active_layer.invalidate_lod();
            active_layer.gpu_generation += 1;
        }

        self.mesh_warp_state.is_active = false;
        self.mesh_warp_state.points.clear();
        self.mesh_warp_state.original_points.clear();
        self.mesh_warp_state.source_snapshot = None;
        self.mesh_warp_state.warp_buffer.clear();
        canvas_state.clear_preview_state();
        canvas_state.mark_dirty(None);

        if stroke_event.is_some() {
            self.pending_stroke_event = stroke_event;
        }
    }

    fn draw_mesh_warp_overlay(
        &self,
        ui: &egui::Ui,
        painter: &egui::Painter,
        canvas_rect: Rect,
        zoom: f32,
        _canvas_state: &CanvasState,
    ) {
        if !self.mesh_warp_state.is_active {
            return;
        }

        let to_screen = |cx: f32, cy: f32| -> Pos2 {
            Pos2::new(canvas_rect.min.x + cx * zoom, canvas_rect.min.y + cy * zoom)
        };

        let accent_color = ui.visuals().hyperlink_color;
        let cols = self.mesh_warp_state.grid_cols;
        let rows = self.mesh_warp_state.grid_rows;
        let pts_per_row = cols + 1;
        let grid_stroke = egui::Stroke::new(
            1.0,
            Color32::from_rgba_unmultiplied(
                accent_color.r(),
                accent_color.g(),
                accent_color.b(),
                160,
            ),
        );
        let curve_segments = 12; // Sub-segments per grid edge for smooth curves

        // Draw smooth horizontal Catmull-Rom curves
        for r in 0..=rows {
            // Collect row control points
            let row_pts: Vec<[f32; 2]> = (0..pts_per_row)
                .map(|c| self.mesh_warp_state.points[r * pts_per_row + c])
                .collect();
            let n = row_pts.len();
            if n < 2 {
                continue;
            }
            let total_segs = (n - 1) * curve_segments;
            let mut screen_pts = Vec::with_capacity(total_segs + 1);
            for s in 0..=total_segs {
                let t = s as f32 / curve_segments as f32;
                let p = crate::ops::transform::catmull_rom_curve_point(&row_pts, t);
                screen_pts.push(to_screen(p[0], p[1]));
            }
            for w in screen_pts.windows(2) {
                painter.line_segment([w[0], w[1]], grid_stroke);
            }
        }

        // Draw smooth vertical Catmull-Rom curves
        for c in 0..=cols {
            // Collect column control points
            let col_pts: Vec<[f32; 2]> = (0..=rows)
                .map(|r| self.mesh_warp_state.points[r * pts_per_row + c])
                .collect();
            let n = col_pts.len();
            if n < 2 {
                continue;
            }
            let total_segs = (n - 1) * curve_segments;
            let mut screen_pts = Vec::with_capacity(total_segs + 1);
            for s in 0..=total_segs {
                let t = s as f32 / curve_segments as f32;
                let p = crate::ops::transform::catmull_rom_curve_point(&col_pts, t);
                screen_pts.push(to_screen(p[0], p[1]));
            }
            for w in screen_pts.windows(2) {
                painter.line_segment([w[0], w[1]], grid_stroke);
            }
        }

        // Draw control point handles
        for (i, pt) in self.mesh_warp_state.points.iter().enumerate() {
            let sp = to_screen(pt[0], pt[1]);
            let is_hover = self.mesh_warp_state.hover_index == Some(i);
            let is_drag = self.mesh_warp_state.dragging_index == Some(i);
            let r = if is_hover || is_drag { 6.0 } else { 4.0 };
            let fill = if is_drag || is_hover {
                accent_color
            } else {
                Color32::WHITE
            };
            painter.circle_filled(sp, r + 1.0, Color32::BLACK);
            painter.circle_filled(sp, r, fill);
        }
    }

    // ====================================================================
    // COLOR REMOVER TOOL — context bar and commit
    // ====================================================================

    fn show_color_remover_options(&mut self, ui: &mut egui::Ui) {
        ui.label(t!("ctx.tolerance"));
        if let Some(new_val) =
            Self::tolerance_slider(ui, "cr_tol", self.color_remover_state.tolerance)
        {
            self.color_remover_state.tolerance = new_val;
        }

        ui.separator();
        ui.label(t!("ctx.color_remover.smoothness"));
        ui.add(
            egui::DragValue::new(&mut self.color_remover_state.smoothness)
                .speed(0.2)
                .range(0..=20)
                .suffix("px"),
        );

        ui.separator();
        let mut contiguous = self.color_remover_state.contiguous;
        if ui
            .checkbox(&mut contiguous, t!("ctx.color_remover.contiguous"))
            .changed()
        {
            self.color_remover_state.contiguous = contiguous;
        }
    }

    fn commit_color_removal(&mut self, canvas_state: &mut CanvasState, click_x: u32, click_y: u32) {
        // Store request for async dispatch by app.rs (shows loading bar)
        self.pending_color_removal = Some(ColorRemovalRequest {
            click_x,
            click_y,
            tolerance: self.color_remover_state.tolerance,
            smoothness: self.color_remover_state.smoothness,
            contiguous: self.color_remover_state.contiguous,
            layer_idx: canvas_state.active_layer_index,
            selection_mask: canvas_state.selection_mask.clone(),
        });
    }

    // ====================================================================
    // SMUDGE TOOL — context bar + per-pixel smear operation
    // ====================================================================

    fn show_smudge_options(&mut self, ui: &mut egui::Ui) {
        ui.label("Size:");
        ui.add(
            egui::DragValue::new(&mut self.properties.size)
                .speed(0.5)
                .range(1.0..=500.0)
                .suffix("px"),
        );
        ui.separator();
        ui.label("Strength:");
        ui.add(egui::Slider::new(&mut self.smudge_state.strength, 0.01..=1.0).fixed_decimals(2));
    }

    /// Smudge one brush circle at (cx, cy): picks up color from canvas and
    /// repaints with the accumulated pickup colour, then updates pickup.
    fn draw_smudge_no_dirty(&mut self, canvas_state: &mut CanvasState, cx: u32, cy: u32) {
        let idx = canvas_state.active_layer_index;
        let w = canvas_state.width;
        let h = canvas_state.height;
        if idx >= canvas_state.layers.len() {
            return;
        }
        let radius = (self.properties.size * 0.5).max(1.0);
        let strength = self.smudge_state.strength;

        let r_int = radius.ceil() as i32;
        let x0 = (cx as i32 - r_int).max(0) as u32;
        let y0 = (cy as i32 - r_int).max(0) as u32;
        let x1 = ((cx as i32 + r_int).min(w as i32 - 1)).max(0) as u32;
        let y1 = ((cy as i32 + r_int).min(h as i32 - 1)).max(0) as u32;

        // Collect modifications to avoid re-borrow conflict
        let mut writes: Vec<(u32, u32, [u8; 4])> = Vec::new();
        let layer = &canvas_state.layers[idx];

        for py in y0..=y1 {
            for px in x0..=x1 {
                let dx = px as f32 - cx as f32;
                let dy = py as f32 - cy as f32;
                let dist = (dx * dx + dy * dy).sqrt();
                if dist > radius {
                    continue;
                }

                let t = 1.0 - (dist / radius).powi(2);
                let alpha = (t * strength).clamp(0.0, 1.0);

                let orig = layer.pixels.get_pixel(px, py);
                let existing = [
                    orig[0] as f32,
                    orig[1] as f32,
                    orig[2] as f32,
                    orig[3] as f32,
                ];

                let new_r = (self.smudge_state.pickup_color[0] * alpha
                    + existing[0] * (1.0 - alpha))
                    .clamp(0.0, 255.0) as u8;
                let new_g = (self.smudge_state.pickup_color[1] * alpha
                    + existing[1] * (1.0 - alpha))
                    .clamp(0.0, 255.0) as u8;
                let new_b = (self.smudge_state.pickup_color[2] * alpha
                    + existing[2] * (1.0 - alpha))
                    .clamp(0.0, 255.0) as u8;
                let new_a = (self.smudge_state.pickup_color[3] * alpha
                    + existing[3] * (1.0 - alpha))
                    .clamp(0.0, 255.0) as u8;

                // Blend existing pixel into pickup for a trailing smear
                let pickup_blend = alpha * 0.2;
                self.smudge_state.pickup_color[0] = (self.smudge_state.pickup_color[0]
                    * (1.0 - pickup_blend)
                    + existing[0] * pickup_blend)
                    .clamp(0.0, 255.0);
                self.smudge_state.pickup_color[1] = (self.smudge_state.pickup_color[1]
                    * (1.0 - pickup_blend)
                    + existing[1] * pickup_blend)
                    .clamp(0.0, 255.0);
                self.smudge_state.pickup_color[2] = (self.smudge_state.pickup_color[2]
                    * (1.0 - pickup_blend)
                    + existing[2] * pickup_blend)
                    .clamp(0.0, 255.0);
                self.smudge_state.pickup_color[3] = (self.smudge_state.pickup_color[3]
                    * (1.0 - pickup_blend)
                    + existing[3] * pickup_blend)
                    .clamp(0.0, 255.0);

                writes.push((px, py, [new_r, new_g, new_b, new_a]));
            }
        }

        let layer_mut = &mut canvas_state.layers[idx];
        for (px, py, rgba) in writes {
            layer_mut.pixels.put_pixel(px, py, image::Rgba(rgba));
        }
    }

    // ====================================================================
    // SHAPES TOOL — context bar, preview, commit, overlay
    // ====================================================================

    fn show_shapes_options(&mut self, ui: &mut egui::Ui, assets: &Assets) {
        use crate::ops::shapes::{ShapeFillMode, ShapeKind};

        ui.label(t!("ctx.shapes.shape"));

        // Button showing current shape icon + name, opens grid popup
        let popup_id = ui.make_persistent_id("shapes_grid_popup");
        let btn_response = {
            let mut label_text = self
                .shapes_state
                .selected_custom_shape
                .clone()
                .unwrap_or_else(|| self.shapes_state.selected_shape.label());
            if label_text.chars().count() > 12 {
                label_text = format!("{}...", label_text.chars().take(12).collect::<String>());
            }
            let custom_tex = self
                .shapes_state
                .selected_custom_shape
                .as_ref()
                .and_then(|n| assets.get_custom_shape_texture(n));
            if let Some(tex) = custom_tex.or_else(|| assets.get_shape_texture(self.shapes_state.selected_shape)) {
                let sized = egui::load::SizedTexture::from_handle(tex);
                let img =
                    egui::Image::from_texture(sized).fit_to_exact_size(egui::Vec2::splat(16.0));
                let btn = egui::Button::image_and_text(img, label_text);
                ui.add_sized([116.0, 24.0], btn)
            } else {
                ui.add_sized([116.0, 24.0], egui::Button::new(label_text))
            }
        };
        if btn_response.clicked() {
            egui::Popup::toggle_id(ui.ctx(), popup_id);
        }

        egui::Popup::new(
            popup_id,
            ui.ctx().clone(),
            egui::PopupAnchor::from(&btn_response),
            ui.layer_id(),
        )
        .open_memory(None::<egui::SetOpenCommand>)
        .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
        .show(|ui| {
            ui.set_width(156.0);
            ui.set_min_width(156.0);
            ui.set_max_width(156.0);
            let picker_shapes = ShapeKind::picker_shapes();
            let cols = 5;
            let icon_size = egui::Vec2::splat(28.0);
            let accent = ui.visuals().hyperlink_color;
            let selected_stroke = egui::Stroke::new(1.0, contrast_text_color(accent));
            let selected_alpha = if color_luminance(accent) > 0.6 {
                34
            } else {
                70
            };

            if ui.add_sized([54.0, 20.0], egui::Button::new("New...")).clicked() {
                self.pending_open_add_shape = true;
                ui.close();
            }

            egui::Grid::new("shapes_icon_grid")
                .spacing(egui::Vec2::splat(2.0))
                .show(ui, |ui| {
                    for (i, kind) in picker_shapes.iter().enumerate() {
                        let selected = self.shapes_state.selected_custom_shape.is_none()
                            && self.shapes_state.selected_shape == *kind;
                        let (rect, response) =
                            ui.allocate_exact_size(icon_size, egui::Sense::click());

                        // Highlight selected shape
                        if selected {
                            ui.painter().rect_filled(
                                rect,
                                4.0,
                                egui::Color32::from_rgba_premultiplied(
                                    accent.r(),
                                    accent.g(),
                                    accent.b(),
                                    selected_alpha,
                                ),
                            );
                            ui.painter().rect_stroke(
                                rect,
                                4.0,
                                selected_stroke,
                                egui::StrokeKind::Middle,
                            );
                        }
                        if response.hovered() {
                            ui.painter().rect_filled(
                                rect,
                                4.0,
                                ui.visuals().widgets.hovered.bg_fill,
                            );
                        }

                        // Draw shape icon
                        if let Some(tex) = assets.get_shape_texture(*kind) {
                            let sized = egui::load::SizedTexture::from_handle(tex);
                            let img =
                                egui::Image::from_texture(sized).fit_to_exact_size(icon_size * 0.8);
                            let inner_rect = rect.shrink(icon_size.x * 0.1);
                            img.paint_at(ui, inner_rect);
                        } else {
                            let lbl = kind.label();
                            let short: String = lbl.chars().take(2).collect();
                            ui.painter().text(
                                rect.center(),
                                egui::Align2::CENTER_CENTER,
                                &short,
                                egui::FontId::proportional(11.0),
                                ui.visuals().text_color(),
                            );
                        }

                        if response.clicked() {
                            self.shapes_state.selected_shape = *kind;
                            self.shapes_state.selected_custom_shape = None;
                            self.shapes_state.selected_custom_shape_data = None;
                        }
                        response.on_hover_text(kind.label());

                        if (i + 1) % cols == 0 {
                            ui.end_row();
                        }
                    }
                });

            for cat in assets.custom_shape_categories() {
                ui.add_space(4.0);
                egui::Grid::new(format!("custom_shapes_{}_grid", cat.name))
                    .spacing(egui::Vec2::splat(2.0))
                    .show(ui, |ui| {
                        for (i, name) in cat.shapes.iter().enumerate() {
                            let selected = self.shapes_state.selected_custom_shape.as_ref() == Some(name);
                            let (rect, response) = ui.allocate_exact_size(icon_size, egui::Sense::click());
                            if selected {
                                ui.painter().rect_filled(
                                    rect,
                                    4.0,
                                    egui::Color32::from_rgba_premultiplied(
                                        accent.r(), accent.g(), accent.b(), selected_alpha,
                                    ),
                                );
                                ui.painter().rect_stroke(rect, 4.0, selected_stroke, egui::StrokeKind::Middle);
                            }
                            if response.hovered() {
                                ui.painter().rect_filled(rect, 4.0, ui.visuals().widgets.hovered.bg_fill);
                            }
                            if let Some(tex) = assets.get_custom_shape_texture(name) {
                                let sized = egui::load::SizedTexture::from_handle(tex);
                                egui::Image::from_texture(sized)
                                    .fit_to_exact_size(icon_size * 0.8)
                                    .paint_at(ui, rect.shrink(icon_size.x * 0.1));
                            }
                            if response.clicked() {
                                self.shapes_state.selected_custom_shape = Some(name.clone());
                                self.shapes_state.selected_custom_shape_data =
                                    assets.get_custom_shape_data(name).map(Into::into);
                            }
                            if response.secondary_clicked() {
                                self.pending_delete_shape = Some(name.clone());
                            }
                            response.on_hover_text(name);
                            if (i + 1) % cols == 0 {
                                ui.end_row();
                            }
                        }
                    });
            }
        });

        ui.separator();
        ui.label(t!("ctx.mode"));

        // Merged icon + dropdown arrow (like brush size widget)
        let popup_id = ui.make_persistent_id("ctx_shapes_fill_popup");
        let fill_icon = self.shapes_state.fill_mode.icon();
        let icon_size = egui::Vec2::splat(16.0);
        let inactive = ui.visuals().widgets.inactive;

        let (icon_resp, arrow_resp) = egui::Frame::NONE
            .fill(inactive.bg_fill)
            .stroke(inactive.bg_stroke)
            .corner_radius(inactive.corner_radius)
            .inner_margin(egui::Margin::same(0))
            .show(ui, |ui| {
                ui.spacing_mut().item_spacing.x = 0.0;
                let vis = ui.visuals_mut();
                vis.widgets.inactive.bg_fill = Color32::TRANSPARENT;
                vis.widgets.inactive.bg_stroke = egui::Stroke::NONE;
                vis.widgets.hovered.bg_fill = Color32::TRANSPARENT;
                vis.widgets.hovered.bg_stroke = egui::Stroke::NONE;
                vis.widgets.active.bg_fill = Color32::TRANSPARENT;
                vis.widgets.active.bg_stroke = egui::Stroke::NONE;

                // Icon button — direct click toggles to next mode
                let i_resp = if let Some(tex) = assets.get_texture(fill_icon) {
                    let sized = egui::load::SizedTexture::from_handle(tex);
                    let img = egui::Image::from_texture(sized).fit_to_exact_size(icon_size);
                    ui.add(egui::Button::image(img).min_size(icon_size))
                } else {
                    ui.add(
                        egui::Button::new(egui::RichText::new(fill_icon.emoji()).size(11.0))
                            .min_size(icon_size),
                    )
                };
                let i_height = i_resp.rect.height();

                // Thin divider
                let sep_x = ui.cursor().left();
                ui.painter().vline(
                    sep_x,
                    i_resp.rect.top() + 3.0..=i_resp.rect.bottom() - 3.0,
                    egui::Stroke::new(1.0, inactive.bg_stroke.color.linear_multiply(0.4)),
                );

                // Dropdown arrow
                let a_resp = if let Some(tex) = assets.get_texture(Icon::DropDown) {
                    let sized = egui::load::SizedTexture::from_handle(tex);
                    let img = egui::Image::from_texture(sized)
                        .fit_to_exact_size(egui::vec2(12.0, 12.0));
                    ui.add(
                        egui::Button::image(img)
                            .min_size(egui::vec2(14.0, i_height)),
                    )
                } else {
                    ui.add(
                        egui::Button::new(egui::RichText::new("\u{25BE}").size(9.0))
                            .min_size(egui::vec2(14.0, i_height)),
                    )
                };
                (i_resp, a_resp)
            })
            .inner;

        // Icon click → toggle to next mode
        if icon_resp.clicked() {
            self.shapes_state.fill_mode = self.shapes_state.fill_mode.next();
        }
        icon_resp.on_hover_text(self.shapes_state.fill_mode.label());

        // Arrow click → open dropdown popup
        if arrow_resp.clicked() {
            egui::Popup::toggle_id(ui.ctx(), popup_id);
        }

        // Popup with all fill mode options (icon + label per row)
        egui::Popup::new(
            popup_id,
            ui.ctx().clone(),
            egui::PopupAnchor::from(&arrow_resp),
            ui.layer_id(),
        )
        .open_memory(None::<egui::SetOpenCommand>)
        .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
        .show(|ui| {
            ui.set_min_width(120.0);
            for mode in ShapeFillMode::all() {
                let mode_icon = mode.icon();
                let label = mode.label();
                let is_selected = self.shapes_state.fill_mode == *mode;

                let resp = if let Some(tex) = assets.get_texture(mode_icon) {
                    let sized = egui::load::SizedTexture::from_handle(tex);
                    let img = egui::Image::from_texture(sized)
                        .fit_to_exact_size(egui::vec2(14.0, 14.0));
                    ui.add(
                        egui::Button::image_and_text(img, label)
                            .min_size(egui::vec2(100.0, 22.0)),
                    )
                } else {
                    ui.add(
                        egui::Button::new(
                            egui::RichText::new(format!("{} {}", mode_icon.emoji(), label))
                                .size(12.0),
                        )
                        .min_size(egui::vec2(100.0, 22.0)),
                    )
                };

                if is_selected {
                    ui.painter().rect_filled(
                        resp.rect,
                        2.0,
                        ui.visuals().selection.bg_fill,
                    );
                }

                if resp.clicked() {
                    self.shapes_state.fill_mode = *mode;
                    egui::Popup::close_id(ui.ctx(), popup_id);
                }
            }
        });

        ui.separator();
        ui.label(t!("ctx.shapes.width"));
        ui.add(
            egui::DragValue::new(&mut self.properties.size)
                .speed(0.5)
                .range(1.0..=100.0)
                .suffix("px"),
        );

        if self.shapes_state.selected_shape == ShapeKind::RoundedRect {
            ui.separator();
            ui.label(t!("ctx.shapes.radius"));
            ui.add(
                egui::DragValue::new(&mut self.shapes_state.corner_radius)
                    .speed(0.5)
                    .range(0.0..=500.0)
                    .suffix("px"),
            );
        }

        ui.separator();
        let aa_resp = ui.selectable_label(self.shapes_state.anti_alias, t!("ctx.anti_alias"));
        if aa_resp.clicked() {
            self.shapes_state.anti_alias = !self.shapes_state.anti_alias;
        }
        aa_resp.on_hover_text(t!("ctx.anti_alias_tooltip"));

        ui.separator();
        ui.label(t!("ctx.blend"));
        egui::ComboBox::from_id_salt("ctx_shapes_blend")
            .selected_text(self.properties.blending_mode.name())
            .width(80.0)
            .show_ui(ui, |ui| {
                for mode in BlendMode::all() {
                    if ui
                        .selectable_label(self.properties.blending_mode == *mode, mode.name())
                        .clicked()
                    {
                        self.properties.blending_mode = *mode;
                    }
                }
            });
    }

    pub fn render_shape_preview(
        &mut self,
        canvas_state: &mut CanvasState,
        primary_color_f32: [f32; 4],
        secondary_color_f32: [f32; 4],
    ) {
        use crate::ops::shapes::PlacedShape;

        let placed = if let Some(ref p) = self.shapes_state.placed {
            p.clone()
        } else if let (Some(start), Some(end)) =
            (self.shapes_state.draw_start, self.shapes_state.draw_end)
        {
            let primary = [
                (primary_color_f32[0] * 255.0) as u8,
                (primary_color_f32[1] * 255.0) as u8,
                (primary_color_f32[2] * 255.0) as u8,
                (primary_color_f32[3] * 255.0) as u8,
            ];
            let secondary = [
                (secondary_color_f32[0] * 255.0) as u8,
                (secondary_color_f32[1] * 255.0) as u8,
                (secondary_color_f32[2] * 255.0) as u8,
                (secondary_color_f32[3] * 255.0) as u8,
            ];
            let cx = (start[0] + end[0]) * 0.5;
            let cy = (start[1] + end[1]) * 0.5;
            let hw = ((end[0] - start[0]) * 0.5).abs();
            let hh = ((end[1] - start[1]) * 0.5).abs();
            PlacedShape {
                cx,
                cy,
                hw,
                hh,
                rotation: 0.0,
                kind: self.shapes_state.selected_shape,
                custom_shape: self.shapes_state.selected_custom_shape.clone(),
                custom_shape_data: self.shapes_state.selected_custom_shape_data.clone(),
                fill_mode: self.shapes_state.fill_mode,
                outline_width: self.properties.size,
                primary_color: primary,
                secondary_color: secondary,
                anti_alias: self.shapes_state.anti_alias,
                corner_radius: self.shapes_state.corner_radius,
                handle_dragging: None,
                drag_offset: [0.0, 0.0],
                drag_anchor: [0.0, 0.0],
                rotate_start_angle: 0.0,
                rotate_start_rotation: 0.0,
            }
        } else {
            canvas_state.clear_preview_state();
            return;
        };

        let (buf_w, buf_h, off_x, off_y) = crate::ops::shapes::rasterize_shape_into(
            &placed,
            canvas_state.width,
            canvas_state.height,
            &mut self.shapes_state.cached_shape_buf,
        );

        if buf_w == 0 || buf_h == 0 {
            canvas_state.clear_preview_state();
            return;
        }

        let preview = TiledImage::from_region_rgba(
            canvas_state.width,
            canvas_state.height,
            &self.shapes_state.cached_shape_buf,
            buf_w,
            buf_h,
            off_x,
            off_y,
        );
        let mut preview = preview;
        Self::adjust_preview_region_for_selection(
            &mut preview,
            off_x,
            off_y,
            buf_w,
            buf_h,
            canvas_state.selection_mask.as_ref(),
            0.3,
        );

        canvas_state.preview_layer = Some(preview);
        canvas_state.preview_blend_mode = self.properties.blending_mode;
        canvas_state.preview_force_composite = true;
        canvas_state.preview_is_eraser = false;
        canvas_state.preview_downscale = 1;
        canvas_state.preview_stroke_bounds = Some(egui::Rect::from_min_max(
            egui::pos2(off_x as f32, off_y as f32),
            egui::pos2((off_x + buf_w as i32) as f32, (off_y + buf_h as i32) as f32),
        ));
        canvas_state.preview_texture_cache = None; // Force re-upload
        canvas_state.mark_dirty(None);
    }

    fn commit_shape(&mut self, canvas_state: &mut CanvasState) {
        if self.shapes_state.placed.is_none() && self.shapes_state.draw_start.is_none() {
            return;
        }

        let Some(target_layer_idx) = self.shapes_state.source_layer_index else {
            return;
        };
        if target_layer_idx >= canvas_state.layers.len() {
            self.shapes_state.placed = None;
            self.shapes_state.draw_start = None;
            self.shapes_state.draw_end = None;
            self.shapes_state.is_drawing = false;
            self.shapes_state.source_layer_index = None;
            canvas_state.clear_preview_state();
            return;
        }

        // Apply mirror to shape preview before committing
        let mirror_bounds = canvas_state.mirror_preview_layer();
        if let Some(bounds) = canvas_state.preview_stroke_bounds {
            self.stroke_tracker.expand_bounds(bounds);
        }
        if let Some(bounds) = mirror_bounds {
            self.stroke_tracker.expand_bounds(bounds);
        }

        let blend_mode = self.properties.blending_mode;
        let stroke_event = self.stroke_tracker.finish(canvas_state);

        if let Some(ref preview) = canvas_state.preview_layer
            && let Some(active_layer) = canvas_state.layers.get_mut(target_layer_idx)
        {
            let selection_mask = canvas_state.selection_mask.clone();
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

        self.shapes_state.placed = None;
        self.shapes_state.draw_start = None;
        self.shapes_state.draw_end = None;
        self.shapes_state.is_drawing = false;
        self.shapes_state.source_layer_index = None;
        canvas_state.clear_preview_state();
        canvas_state.mark_dirty(None);

        if stroke_event.is_some() {
            self.pending_stroke_event = stroke_event;
        }
    }

    fn draw_shape_overlay(&self, painter: &egui::Painter, canvas_rect: Rect, zoom: f32) {
        let placed = match &self.shapes_state.placed {
            Some(p) => p,
            None => return,
        };

        let to_screen = |cx: f32, cy: f32| -> Pos2 {
            Pos2::new(canvas_rect.min.x + cx * zoom, canvas_rect.min.y + cy * zoom)
        };

        let cos_r = placed.rotation.cos();
        let sin_r = placed.rotation.sin();
        let corners = [
            (-placed.hw, -placed.hh),
            (placed.hw, -placed.hh),
            (placed.hw, placed.hh),
            (-placed.hw, placed.hh),
        ];
        let screen_corners: Vec<Pos2> = corners
            .iter()
            .map(|(cx, cy)| {
                to_screen(
                    cx * cos_r - cy * sin_r + placed.cx,
                    cx * sin_r + cy * cos_r + placed.cy,
                )
            })
            .collect();

        // Use accent color for the selection frame
        let style = painter.ctx().style_of(painter.ctx().theme());
        let accent = Color32::from_rgb(
            style.visuals.hyperlink_color.r(),
            style.visuals.hyperlink_color.g(),
            style.visuals.hyperlink_color.b(),
        );
        let accent_semi = Color32::from_rgba_unmultiplied(accent.r(), accent.g(), accent.b(), 200);
        let bb_stroke = egui::Stroke::new(1.0, accent_semi);
        for i in 0..4 {
            painter.line_segment([screen_corners[i], screen_corners[(i + 1) % 4]], bb_stroke);
        }

        let hs = 4.0;
        for corner in &screen_corners {
            painter.rect_filled(
                egui::Rect::from_center_size(*corner, egui::vec2(hs * 2.0, hs * 2.0)),
                1.0,
                accent,
            );
            painter.rect_stroke(
                egui::Rect::from_center_size(*corner, egui::vec2(hs * 2.0, hs * 2.0)),
                1.0,
                egui::Stroke::new(1.0, Color32::BLACK),
                egui::StrokeKind::Middle,
            );
        }

        let edge_handles = [
            Pos2::new(
                (screen_corners[0].x + screen_corners[1].x) * 0.5,
                (screen_corners[0].y + screen_corners[1].y) * 0.5,
            ),
            Pos2::new(
                (screen_corners[1].x + screen_corners[2].x) * 0.5,
                (screen_corners[1].y + screen_corners[2].y) * 0.5,
            ),
            Pos2::new(
                (screen_corners[2].x + screen_corners[3].x) * 0.5,
                (screen_corners[2].y + screen_corners[3].y) * 0.5,
            ),
            Pos2::new(
                (screen_corners[3].x + screen_corners[0].x) * 0.5,
                (screen_corners[3].y + screen_corners[0].y) * 0.5,
            ),
        ];
        let ehs = 3.0;
        for p in edge_handles {
            painter.rect_filled(
                egui::Rect::from_center_size(p, egui::vec2(ehs * 2.0, ehs * 2.0)),
                1.0,
                accent,
            );
            painter.rect_stroke(
                egui::Rect::from_center_size(p, egui::vec2(ehs * 2.0, ehs * 2.0)),
                1.0,
                egui::Stroke::new(1.0, Color32::BLACK),
                egui::StrokeKind::Middle,
            );
        }

        // Rotation handle: faces outward from center through the top edge midpoint
        let top_mid = Pos2::new(
            (screen_corners[0].x + screen_corners[1].x) * 0.5,
            (screen_corners[0].y + screen_corners[1].y) * 0.5,
        );
        let center = to_screen(placed.cx, placed.cy);
        let dir_x = top_mid.x - center.x;
        let dir_y = top_mid.y - center.y;
        let dir_len = (dir_x * dir_x + dir_y * dir_y).sqrt().max(0.001);
        let rot_handle = Pos2::new(
            top_mid.x + (dir_x / dir_len) * 20.0,
            top_mid.y + (dir_y / dir_len) * 20.0,
        );
        painter.line_segment([top_mid, rot_handle], egui::Stroke::new(1.0, accent_semi));
        painter.circle_filled(rot_handle, 4.0, accent);
        painter.circle_stroke(rot_handle, 4.0, egui::Stroke::new(1.0, Color32::BLACK));
    }

}

