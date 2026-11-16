impl ToolsPanel {
    /// Thin aliased patterns count raster cells rather than snapping regularly
    /// spaced geometric dots into occasionally adjacent pixels.
    fn draw_thin_pixel_pattern(
        &self,
        image: &mut TiledImage,
        points: [Pos2; 4],
        steps: usize,
        pattern: LinePattern,
        color: [f32; 4],
        mask: Option<&GrayImage>,
    ) -> bool {
        if self.line_state.line_tool.options.anti_alias
            || self.properties.size > 1.001
            || pattern == LinePattern::Solid
        {
            return false;
        }
        let [a, b, c, d] = points;
        let length = (b - a).length() + (c - b).length() + (d - c).length();
        if !length.is_finite() || length > 1_000_000. {
            return false;
        }
        let straight = length - (d - a).length() < 0.001;
        let samples = if straight { 1 } else { steps };
        let mut path: Vec<(i32, i32)> = Vec::new();
        let mut previous = (a.x.floor() as i32, a.y.floor() as i32);
        path.push(previous);
        for sample in 1..=samples {
            let p = self.bezier_point(a, b, c, d, sample as f32 / samples as f32);
            let end = (p.x.floor() as i32, p.y.floor() as i32);
            let (mut x, mut y) = previous;
            let dx = (i64::from(end.0) - i64::from(x)).abs();
            let dy = -(i64::from(end.1) - i64::from(y)).abs();
            let sx = if x < end.0 { 1 } else { -1 };
            let sy = if y < end.1 { 1 } else { -1 };
            let mut error = dx + dy;
            while (x, y) != end {
                let twice = error * 2;
                if twice >= dy {
                    error += dy;
                    x += sx;
                }
                if twice <= dx {
                    error += dx;
                    y += sy;
                }
                // Collapse subpixel stair corners into one diagonal raster cell.
                if !straight && path.len() >= 2 {
                    let prior = path[path.len() - 2];
                    if prior != (x, y)
                        && (i64::from(prior.0) - i64::from(x)).abs() <= 1
                        && (i64::from(prior.1) - i64::from(y)).abs() <= 1
                    {
                        path.pop();
                    }
                }
                if path.last() != Some(&(x, y)) {
                    path.push((x, y));
                }
            }
            previous = end;
        }
        let rgba = Rgba(color.map(|value| (value * 255.).round() as u8));
        for (index, (x, y)) in path.into_iter().enumerate() {
            let on = match pattern {
                LinePattern::Dotted => index % 2 == 0,
                _ => index % 4 < 2,
            };
            if on
                && x >= 0
                && y >= 0
                && (x as u32) < image.width()
                && (y as u32) < image.height()
                && Self::selection_allows(mask, x as u32, y as u32)
            {
                image.put_pixel(x as u32, y as u32, rgba);
            }
        }
        true
    }

    pub(crate) fn line_handle_cursor(
        &self,
        pos: Pos2,
        image: Rect,
        zoom: f32,
    ) -> Option<egui::CursorIcon> {
        if self.active_tool != Tool::Line || self.line_state.line_tool.stage != LineStage::Editing {
            return None;
        }
        if self.line_state.line_tool.dragging_handle.is_some()
            || self.line_state.line_tool.pan_handle_dragging
        {
            Some(egui::CursorIcon::Grabbing)
        } else if self.pointer_over_off_canvas_edit_handle(pos, image, zoom) {
            Some(egui::CursorIcon::Grab)
        } else {
            None
        }
    }

    pub(crate) fn refresh_line_preview(&mut self, canvas: &mut CanvasState, color: Color32) {
        if (self.active_tool != Tool::Line && self.temporary_picker_return != Some(Tool::Line))
            || self.line_state.line_tool.stage != LineStage::Editing
            || self.line_target_layer(canvas).is_none()
        {
            return;
        }
        let line = &self.line_state.line_tool;
        let changed = line.last_size != self.properties.size
            || line.last_color != color
            || line.last_blend != self.properties.blending_mode
            || line.last_pattern != line.options.pattern
            || line.last_cap_style != line.options.cap_style
            || line.last_end_shape != line.options.end_shape
            || line.last_arrow_side != line.options.arrow_side
            || line.last_anti_alias != line.options.anti_alias;
        if !changed {
            return;
        }
        self.properties.color = color;
        let line = &mut self.line_state.line_tool;
        line.last_size = self.properties.size;
        line.last_color = color;
        line.last_blend = self.properties.blending_mode;
        line.last_pattern = line.options.pattern;
        line.last_cap_style = line.options.cap_style;
        line.last_end_shape = line.options.end_shape;
        line.last_arrow_side = line.options.arrow_side;
        line.last_anti_alias = line.options.anti_alias;
        let points = line.control_points;
        let last = line.last_bounds;
        let current = self.get_bezier_bounds(points, canvas.width, canvas.height);
        self.rasterize_bezier(
            canvas,
            points,
            color,
            self.line_state.line_tool.options.pattern,
            self.line_state.line_tool.options.cap_style,
            last,
        );
        canvas.mark_preview_changed_rect(last.map_or(current, |old| old.union(current)));
        self.line_state.line_tool.last_bounds = Some(current);
    }

    #[allow(clippy::too_many_arguments)]
    fn draw_line_capsule(
        &self,
        image: &mut TiledImage,
        start: Pos2,
        end: Pos2,
        color: [f32; 4],
        mask: Option<&GrayImage>,
        flat_start: bool,
        flat_end: bool,
    ) {
        let aa = self.line_state.line_tool.options.anti_alias;
        let align = |p: Pos2| {
            let (x, y) = crate::ui::panels::tools::pixel_aligned_brush_center(
                (p.x + 0.0001, p.y + 0.0001),
                self.properties.size,
                !aa,
            );
            Pos2::new(x, y)
        };
        let start = align(start);
        let end = align(end);
        let delta = end - start;
        let length_sq = delta.length_sq();
        let radius = self.properties.size * 0.5;
        let pad = radius + if aa { 0.5 } else { 0. };
        let x0 = (start.x.min(end.x) - pad).floor().max(0.) as u32;
        let y0 = (start.y.min(end.y) - pad).floor().max(0.) as u32;
        let x1 = (start.x.max(end.x) + pad).ceil().min(image.width() as f32) as u32;
        let y1 = (start.y.max(end.y) + pad).ceil().min(image.height() as f32) as u32;
        for y in y0..y1 {
            for x in x0..x1 {
                if !Self::selection_allows(mask, x, y) {
                    continue;
                }
                let pixel = Pos2::new(x as f32 + 0.5, y as f32 + 0.5);
                let t = if length_sq > 0.000001 {
                    (pixel - start).dot(delta) / length_sq
                } else {
                    0.
                };
                if (flat_start && t < 0.) || (flat_end && t > 1.) {
                    continue;
                }
                let distance = (pixel - (start + delta * t.clamp(0., 1.))).length();
                let coverage = if aa {
                    (radius + 0.5 - distance).clamp(0., 1.)
                } else if distance <= radius {
                    1.
                } else {
                    0.
                };
                let alpha = (coverage * color[3] * 255.).round() as u8;
                if alpha > image.get_pixel(x, y)[3] {
                    image.put_pixel(
                        x,
                        y,
                        Rgba([
                            (color[0] * 255.).round() as u8,
                            (color[1] * 255.).round() as u8,
                            (color[2] * 255.).round() as u8,
                            alpha,
                        ]),
                    );
                }
            }
        }
    }

    fn canvas_to_screen(&self, canvas_pos: (u32, u32), canvas_rect: Rect, zoom: f32) -> Pos2 {
        let (x, y) = canvas_pos;
        Pos2::new(
            canvas_rect.min.x + x as f32 * zoom,
            canvas_rect.min.y + y as f32 * zoom,
        )
    }

    /// Convert canvas pixel coordinates to screen Pos2 (float version)
    fn canvas_pos2_to_screen(&self, canvas_pos: Pos2, canvas_rect: Rect, zoom: f32) -> Pos2 {
        Pos2::new(
            canvas_rect.min.x + canvas_pos.x * zoom,
            canvas_rect.min.y + canvas_pos.y * zoom,
        )
    }

    /// Convert screen Pos2 to canvas pixel coordinates
    fn screen_to_canvas_pos2(&self, screen_pos: Pos2, canvas_rect: Rect, zoom: f32) -> Pos2 {
        Pos2::new(
            (screen_pos.x - canvas_rect.min.x) / zoom,
            (screen_pos.y - canvas_rect.min.y) / zoom,
        )
    }

    /// Calculate cubic Bézier point at t (0.0 to 1.0)
    fn bezier_point(&self, p0: Pos2, p1: Pos2, p2: Pos2, p3: Pos2, t: f32) -> Pos2 {
        let t2 = t * t;
        let t3 = t2 * t;
        let mt = 1.0 - t;
        let mt2 = mt * mt;
        let mt3 = mt2 * mt;

        Pos2::new(
            mt3 * p0.x + 3.0 * mt2 * t * p1.x + 3.0 * mt * t2 * p2.x + t3 * p3.x,
            mt3 * p0.y + 3.0 * mt2 * t * p1.y + 3.0 * mt * t2 * p2.y + t3 * p3.y,
        )
    }

    /// Calculate bounding box for Bézier curve based on control points
    pub fn get_bezier_bounds(
        &self,
        control_points: [Pos2; 4],
        canvas_width: u32,
        canvas_height: u32,
    ) -> Rect {
        // Find min/max X and Y of control points
        let mut min_x = control_points[0].x;
        let mut max_x = control_points[0].x;
        let mut min_y = control_points[0].y;
        let mut max_y = control_points[0].y;

        for point in &control_points[1..] {
            min_x = min_x.min(point.x);
            max_x = max_x.max(point.x);
            min_y = min_y.min(point.y);
            max_y = max_y.max(point.y);
        }

        // Padding includes brush radius plus extra space for arrowheads
        let arrow_extra = if self.line_state.line_tool.options.end_shape == LineEndShape::Arrow {
            (self.properties.size * 3.0).max(8.0) + (self.properties.size * 1.5).max(4.0)
        } else {
            0.0
        };
        let padding = self.properties.size / 2.0 + 2.0 + arrow_extra;
        min_x = (min_x - padding).max(0.0);
        min_y = (min_y - padding).max(0.0);
        max_x = (max_x + padding).min(canvas_width as f32);
        max_y = (max_y + padding).min(canvas_height as f32);

        Rect::from_min_max(Pos2::new(min_x, min_y), Pos2::new(max_x, max_y))
    }

    /// Rasterize a cubic Bézier curve to the preview layer
    pub fn rasterize_bezier(
        &self,
        canvas_state: &mut CanvasState,
        control_points: [Pos2; 4],
        color: Color32,
        pattern: LinePattern,
        cap_style: CapStyle,
        last_bounds: Option<Rect>, // Previous frame's bounds for focused clearing
    ) {
        // Convert Color32 to high-precision f32 for drawing
        let color_f32 = color
            .to_srgba_unmultiplied()
            .map(|channel| channel as f32 / 255.);
        // Create or clear preview layer
        if canvas_state.preview_layer.is_none()
            || canvas_state.preview_layer.as_ref().unwrap().width() != canvas_state.width
            || canvas_state.preview_layer.as_ref().unwrap().height() != canvas_state.height
        {
            canvas_state.preview_layer =
                Some(TiledImage::new(canvas_state.width, canvas_state.height));
        }
        // Line tool uses the tool's blending mode for preview
        canvas_state.preview_blend_mode = self.properties.blending_mode;
        if canvas_state.edit_layer_mask
            && canvas_state
                .layers
                .get(canvas_state.active_layer_index)
                .is_some_and(|l| l.has_live_mask())
        {
            canvas_state.preview_force_composite = true;
            canvas_state.preview_targets_mask = true;
            canvas_state.preview_mask_reveal = false;
            canvas_state.preview_is_eraser = false;
        } else {
            canvas_state.preview_targets_mask = false;
            canvas_state.preview_mask_reveal = false;
        }

        // OPTIMIZATION: Only clear the previous frame's bounding box instead of entire image
        if let Some(ref mut preview) = canvas_state.preview_layer {
            if let Some(bounds) = last_bounds {
                // Focused clear: drop chunks that overlap the previous bounds
                let min_x = bounds.min.x.floor().max(0.0) as u32;
                let max_x = bounds.max.x.ceil().min(canvas_state.width as f32) as u32;
                let min_y = bounds.min.y.floor().max(0.0) as u32;
                let max_y = bounds.max.y.ceil().min(canvas_state.height as f32) as u32;
                preview.clear_region(min_x, min_y, max_x, max_y);
            } else {
                // First frame: clear entire layer
                preview.clear();
            }

            let [p0, p1, p2, p3] = control_points;

            // Much tighter spacing for smooth lines - use 10% of size for better quality
            let spacing = (self.properties.size * 0.1).max(0.5);

            // Adaptive sampling based on curve length estimate
            let chord_len = (p3 - p0).length();
            let control_net_len = (p1 - p0).length() + (p2 - p1).length() + (p3 - p2).length();
            let total_length = control_net_len + chord_len;

            // Calculate steps based on tighter spacing for smooth rendering
            let steps = (total_length / spacing).ceil() as usize;
            let steps = steps.clamp(20, 5000); // Higher max for quality

            let selection_mask = canvas_state.selection_mask.as_ref();
            let pixel_pattern = self.draw_thin_pixel_pattern(
                preview,
                control_points,
                steps,
                pattern,
                color_f32,
                selection_mask,
            );
            let mut previous = p0;
            let mut distance = 0.0_f32;
            let mut next_dot = 0.0_f32;
            let size = self.properties.size.max(1.);
            let on = (size * 2.).round().max(1.);
            let off = (size * 1.5).round().max(1.);
            let cycle = on + off;
            for i in 1..=if pixel_pattern { 0 } else { steps } {
                let point = self.bezier_point(p0, p1, p2, p3, i as f32 / steps as f32);
                let length = (point - previous).length();
                if length <= 0.00001 {
                    continue;
                }
                let end_distance = distance + length;
                match pattern {
                    LinePattern::Dotted => {
                        while next_dot <= end_distance + 0.00001 {
                            let t = ((next_dot - distance) / length).clamp(0., 1.);
                            let dot = previous + (point - previous) * t;
                            self.draw_bezier_dot(
                                preview,
                                (dot.x, dot.y),
                                color_f32,
                                selection_mask,
                            );
                            next_dot += size * 2.;
                        }
                    }
                    LinePattern::Solid => self.draw_line_capsule(
                        preview,
                        previous,
                        point,
                        color_f32,
                        selection_mask,
                        cap_style == CapStyle::Flat && i == 1,
                        cap_style == CapStyle::Flat && i == steps,
                    ),
                    LinePattern::Dashed => {
                        let mut position = distance;
                        while position < end_distance - 0.00001 {
                            let phase = position.rem_euclid(cycle);
                            let drawing = phase < on - 0.00001;
                            let remaining = if drawing { on - phase } else { cycle - phase };
                            let finish = (position + remaining.max(0.00001)).min(end_distance);
                            if drawing {
                                let start = previous
                                    + (point - previous) * ((position - distance) / length);
                                let end =
                                    previous + (point - previous) * ((finish - distance) / length);
                                self.draw_line_capsule(
                                    preview,
                                    start,
                                    end,
                                    color_f32,
                                    selection_mask,
                                    cap_style == CapStyle::Flat && phase < 0.0001,
                                    cap_style == CapStyle::Flat
                                        && (finish - position - remaining).abs() < 0.0001,
                                );
                            }
                            position = finish;
                        }
                    }
                }
                distance = end_distance;
                previous = point;
            }
            if !pixel_pattern && distance < 0.00001 {
                self.draw_bezier_dot(preview, (p0.x, p0.y), color_f32, selection_mask);
            }
            let end_shape = self.line_state.line_tool.options.end_shape;
            let arrow_side = self.line_state.line_tool.options.arrow_side;

            // Draw arrowheads if enabled
            if end_shape == LineEndShape::Arrow {
                let arrow_length = (self.properties.size * 3.0).max(8.0);
                let arrow_half_width = (self.properties.size * 1.5).max(4.0);
                // Push tip forward so the triangle fully covers the line endpoint
                let tip_advance = self.properties.size + self.properties.size / 2.0;

                // End arrow
                if arrow_side == ArrowSide::End || arrow_side == ArrowSide::Both {
                    // Tangent at t=1: B'(1) = 3(P3 - P2)
                    let tx = 3.0 * (p3.x - p2.x);
                    let ty = 3.0 * (p3.y - p2.y);
                    let len = (tx * tx + ty * ty).sqrt().max(0.001);
                    let dx = tx / len;
                    let dy = ty / len;
                    let tip = Pos2::new(p3.x + dx * tip_advance, p3.y + dy * tip_advance);
                    let base_center =
                        Pos2::new(tip.x - dx * arrow_length, tip.y - dy * arrow_length);
                    let px = -dy;
                    let py = dx;
                    let wing1 = Pos2::new(
                        base_center.x + px * arrow_half_width,
                        base_center.y + py * arrow_half_width,
                    );
                    let wing2 = Pos2::new(
                        base_center.x - px * arrow_half_width,
                        base_center.y - py * arrow_half_width,
                    );
                    self.draw_filled_triangle(
                        preview,
                        tip,
                        wing1,
                        wing2,
                        color_f32,
                        canvas_state.width,
                        canvas_state.height,
                        selection_mask,
                    );
                }

                // Start arrow (points backward along the curve)
                if arrow_side == ArrowSide::Start || arrow_side == ArrowSide::Both {
                    // Tangent at t=0: B'(0) = 3(P1 - P0)
                    let tx = 3.0 * (p1.x - p0.x);
                    let ty = 3.0 * (p1.y - p0.y);
                    let len = (tx * tx + ty * ty).sqrt().max(0.001);
                    let dx = tx / len;
                    let dy = ty / len;
                    let tip = Pos2::new(p0.x - dx * tip_advance, p0.y - dy * tip_advance);
                    let base_center =
                        Pos2::new(tip.x + dx * arrow_length, tip.y + dy * arrow_length);
                    let px = -dy;
                    let py = dx;
                    let wing1 = Pos2::new(
                        base_center.x + px * arrow_half_width,
                        base_center.y + py * arrow_half_width,
                    );
                    let wing2 = Pos2::new(
                        base_center.x - px * arrow_half_width,
                        base_center.y - py * arrow_half_width,
                    );
                    self.draw_filled_triangle(
                        preview,
                        tip,
                        wing1,
                        wing2,
                        color_f32,
                        canvas_state.width,
                        canvas_state.height,
                        selection_mask,
                    );
                }
            }
        }
    }

    /// Draw a filled anti-aliased triangle on the preview layer (for arrowheads).
    fn draw_filled_triangle(
        &self,
        preview: &mut TiledImage,
        a: Pos2,
        b: Pos2,
        c: Pos2,
        color_f32: [f32; 4],
        canvas_w: u32,
        canvas_h: u32,
        selection_mask: Option<&image::GrayImage>,
    ) {
        // 1px AA fade for crisp edges
        let fade_px = 1.0_f32;

        // Bounding box expanded by fade zone
        let min_x = (a.x.min(b.x).min(c.x) - fade_px).floor().max(0.0) as u32;
        let max_x =
            ((a.x.max(b.x).max(c.x) + fade_px).ceil() as u32).min(canvas_w.saturating_sub(1));
        let min_y = (a.y.min(b.y).min(c.y) - fade_px).floor().max(0.0) as u32;
        let max_y =
            ((a.y.max(b.y).max(c.y) + fade_px).ceil() as u32).min(canvas_h.saturating_sub(1));

        let [src_r, src_g, src_b, src_a] = color_f32;

        // Signed pixel distance from point to edge (positive = inside)
        #[inline]
        fn edge_dist(v0: Pos2, v1: Pos2, px: f32, py: f32) -> f32 {
            let ex = v1.x - v0.x;
            let ey = v1.y - v0.y;
            let len = (ex * ex + ey * ey).sqrt().max(0.001);
            ((ex) * (py - v0.y) - (ey) * (px - v0.x)) / len
        }

        // Determine winding direction
        let area_sign = {
            let ex = b.x - a.x;
            let ey = b.y - a.y;
            let cross = ex * (c.y - a.y) - ey * (c.x - a.x);
            if cross >= 0.0 { 1.0f32 } else { -1.0f32 }
        };

        for y in min_y..=max_y {
            for x in min_x..=max_x {
                if !Self::selection_allows(selection_mask, x, y) {
                    continue;
                }
                let px = x as f32 + 0.5;
                let py = y as f32 + 0.5;

                let d0 = edge_dist(a, b, px, py) * area_sign;
                let d1 = edge_dist(b, c, px, py) * area_sign;
                let d2 = edge_dist(c, a, px, py) * area_sign;

                let min_dist = d0.min(d1).min(d2);

                if min_dist < -fade_px {
                    continue;
                }

                // Smoothstep AA fade
                let alpha = if min_dist >= fade_px {
                    src_a
                } else {
                    let t = ((min_dist + fade_px) / (2.0 * fade_px)).clamp(0.0, 1.0);
                    let smooth = t * t * (3.0 - 2.0 * t);
                    smooth * src_a
                };

                if alpha <= 0.0 {
                    continue;
                }

                let pixel = preview.get_pixel_mut(x, y);
                let base_a = pixel[3] as f32 / 255.0;
                if alpha > base_a {
                    *pixel = image::Rgba([
                        (src_r * 255.0) as u8,
                        (src_g * 255.0) as u8,
                        (src_b * 255.0) as u8,
                        (alpha * 255.0) as u8,
                    ]);
                }
            }
        }
    }

    /// Draw a single dot for Bézier curve (sub-pixel position for smooth AA)
    fn draw_bezier_dot(
        &self,
        preview: &mut TiledImage,
        pos: (f32, f32),
        color_f32: [f32; 4],
        selection_mask: Option<&image::GrayImage>,
    ) {
        let radius = self.properties.size / 2.0;
        let anti_alias = self.line_state.line_tool.options.anti_alias;
        // High hardness for crisp edges with ~2px AA fade (matching arrow sharpness)
        self.draw_bezier_circle_with_hardness(
            preview,
            pos,
            color_f32,
            radius,
            0.95,
            anti_alias,
            selection_mask,
        );
    }

    /// Draw a circle on the preview layer for Bézier curves with custom hardness override
    /// Uses sub-pixel (f32) center for smooth anti-aliasing without grid artifacts
    fn draw_bezier_circle_with_hardness(
        &self,
        preview: &mut TiledImage,
        pos: (f32, f32),
        color_f32: [f32; 4],
        radius: f32,
        forced_hardness: f32,
        anti_alias: bool,
        selection_mask: Option<&image::GrayImage>,
    ) {
        let (cx, cy) = crate::ui::panels::tools::pixel_aligned_brush_center(
            (pos.0 + 0.0001, pos.1 + 0.0001),
            radius * 2.,
            !anti_alias,
        );

        // Extend the sampling area beyond the nominal radius so the soft fade
        // region actually gets drawn.  For tiny radii (< 1.5) we only need 1px
        // of AA feather; for larger radii the fade scales with softness.
        let aa_pad = if anti_alias {
            if radius < 1.5 {
                1.5 // 1px AA feather for thin lines
            } else {
                (radius * (1.0 - forced_hardness)).max(2.0) + 2.0
            }
        } else {
            1.0
        };
        let outer_radius = radius + aa_pad;

        let min_x = (cx - outer_radius).max(0.0) as u32;
        let max_x = ((cx + outer_radius).ceil() as u32).min(preview.width() - 1);
        let min_y = (cy - outer_radius).max(0.0) as u32;
        let max_y = ((cy + outer_radius).ceil() as u32).min(preview.height() - 1);

        // Unpack high-precision source
        let [src_r, src_g, src_b, src_a] = color_f32;

        for y in min_y..=max_y {
            for x in min_x..=max_x {
                if !Self::selection_allows(selection_mask, x, y) {
                    continue;
                }
                // Sub-pixel distance from float center for smooth AA
                let dx = x as f32 + 0.5 - cx;
                let dy = y as f32 + 0.5 - cy;
                let dist = (dx * dx + dy * dy).sqrt();

                // Compute alpha — this returns 0.0 for pixels beyond the
                // effective (soft) radius, so no explicit dist guard needed.
                let alpha = self.compute_line_alpha(dist, radius, forced_hardness, anti_alias);
                if alpha <= 0.0 {
                    continue;
                }

                let pixel_alpha = alpha * src_a;

                // MAX blending: keep the highest alpha at each pixel.
                // Since every stamp uses the same RGB colour, taking max(alpha)
                // produces perfectly smooth edges with no scalloping from
                // overlapping circle stamps.
                let pixel = preview.get_pixel_mut(x, y);
                let base_a = pixel[3] as f32 / 255.0;

                if pixel_alpha > base_a {
                    *pixel = Rgba([
                        (src_r * 255.0) as u8,
                        (src_g * 255.0) as u8,
                        (src_b * 255.0) as u8,
                        (pixel_alpha * 255.0) as u8,
                    ]);
                }
            }
        }
    }

    /// Magic Wand selection — spawn async Dijkstra minimax distance map computation.
    /// Result arrives via channel; tolerance changes re-threshold instantly.
    fn cached_active_layer_rgba(
        cache: &mut Option<FlatLayerCache>,
        canvas_state: &CanvasState,
    ) -> Option<Arc<[u8]>> {
        let layer_index = canvas_state.active_layer_index;
        let layer = canvas_state.layers.get(layer_index)?;

        let needs_refresh = cache.as_ref().is_none_or(|entry| {
            entry.layer_index != layer_index
                || entry.gpu_generation != layer.gpu_generation
                || entry.width != canvas_state.width
                || entry.height != canvas_state.height
        });

        if needs_refresh {
            let data: Arc<[u8]> =
                Arc::from(layer.pixels.to_rgba_image().into_raw().into_boxed_slice());
            *cache = Some(FlatLayerCache {
                layer_index,
                gpu_generation: layer.gpu_generation,
                width: canvas_state.width,
                height: canvas_state.height,
                data: data.clone(),
            });
            Some(data)
        } else {
            cache.as_ref().map(|entry| entry.data.clone())
        }
    }

    #[inline]
    fn selection_allows(selection_mask: Option<&image::GrayImage>, x: u32, y: u32) -> bool {
        selection_mask.is_none_or(|mask| {
            x < mask.width() && y < mask.height() && mask.get_pixel(x, y).0[0] != 0
        })
    }

    fn adjust_preview_region_for_selection(
        preview: &mut TiledImage,
        off_x: i32,
        off_y: i32,
        region_w: u32,
        region_h: u32,
        selection_mask: Option<&image::GrayImage>,
        outside_alpha_scale: f32,
    ) {
        let Some(mask) = selection_mask else {
            return;
        };

        for row in 0..region_h {
            let gy = off_y + row as i32;
            if gy < 0 || gy >= preview.height() as i32 {
                continue;
            }
            for col in 0..region_w {
                let gx = off_x + col as i32;
                if gx < 0 || gx >= preview.width() as i32 {
                    continue;
                }
                let x = gx as u32;
                let y = gy as u32;
                if Self::selection_allows(Some(mask), x, y) {
                    continue;
                }
                let pixel = preview.get_pixel_mut(x, y);
                if pixel[3] == 0 {
                    continue;
                }
                if outside_alpha_scale <= 0.0 {
                    pixel[3] = 0;
                } else {
                    pixel[0] = ((pixel[0] as u16 + 128) / 2) as u8;
                    pixel[1] = ((pixel[1] as u16 + 128) / 2) as u8;
                    pixel[2] = ((pixel[2] as u16 + 128) / 2) as u8;
                    pixel[3] = ((pixel[3] as f32) * outside_alpha_scale).round() as u8;
                }
            }
        }
    }
}

#[cfg(test)]
mod line_refinement_tests {
    use super::*;
    #[test]
    fn one_pixel_patterns_repeat_in_raster_cells_at_all_slopes() {
        for (start, end) in [
            ((4., 4.), (55., 55.)),
            ((55., 4.), (4., 55.)),
            ((10., 4.), (30., 60.)),
            ((4., 10.), (60., 30.)),
            ((55., 55.), (4., 4.)),
        ] {
            for pattern in [LinePattern::Dotted, LinePattern::Dashed] {
                let mut tools = ToolsPanel::default();
                tools.properties.size = 1.;
                tools.line_state.line_tool.options.anti_alias = false;
                let mut canvas = CanvasState::new(64, 64);
                let a = Pos2::new(start.0, start.1);
                let b = Pos2::new(end.0, end.1);
                tools.rasterize_bezier(
                    &mut canvas,
                    [a, a + (b - a) / 3., a + (b - a) * 2. / 3., b],
                    Color32::BLACK,
                    pattern,
                    CapStyle::Round,
                    None,
                );
                let image = canvas.preview_layer.as_ref().unwrap();
                let x_major = (b.x - a.x).abs() >= (b.y - a.y).abs();
                let length = if x_major {
                    (b.x - a.x).abs()
                } else {
                    (b.y - a.y).abs()
                } as i32;
                for index in 0..=length {
                    let axis = if x_major {
                        a.x as i32 + index * (if b.x >= a.x { 1 } else { -1 })
                    } else {
                        a.y as i32 + index * (if b.y >= a.y { 1 } else { -1 })
                    };
                    let painted = (0..64)
                        .filter(|other| {
                            if x_major {
                                image.get_pixel(axis as u32, *other)[3] != 0
                            } else {
                                image.get_pixel(*other, axis as u32)[3] != 0
                            }
                        })
                        .count();
                    let on = if pattern == LinePattern::Dotted {
                        index % 2 == 0
                    } else {
                        index % 4 < 2
                    };
                    assert_eq!(
                        painted,
                        usize::from(on),
                        "{pattern:?}, {start:?}->{end:?}, step {index}"
                    );
                }
            }
        }
    }
    #[test]
    fn small_aliased_patterns_have_uniform_arc_length_spacing() {
        for size in [1_f32, 2.] {
            for pattern in [LinePattern::Dotted, LinePattern::Dashed] {
                let mut tools = ToolsPanel::default();
                tools.properties.size = size;
                tools.line_state.line_tool.options.anti_alias = false;
                let mut state = CanvasState::new(160, 32);
                tools.rasterize_bezier(
                    &mut state,
                    [
                        Pos2::new(8., 16.),
                        Pos2::new(48., 16.),
                        Pos2::new(88., 16.),
                        Pos2::new(128., 16.),
                    ],
                    Color32::RED,
                    pattern,
                    CapStyle::Round,
                    None,
                );
                let pixels = state.preview_layer.as_ref().unwrap();
                let period = if pattern == LinePattern::Dotted {
                    (size * 2.) as u32
                } else {
                    ((size * 2.).round() + (size * 1.5).round()) as u32
                };
                assert!(
                    pixels
                        .to_rgba_image()
                        .pixels()
                        .all(|p| p[3] == 0 || p[3] == 255)
                );
                for x in 20..100 {
                    for y in 13..19 {
                        assert_eq!(
                            pixels.get_pixel(x, y),
                            pixels.get_pixel(x + period, y),
                            "size {size}, pattern {pattern:?}, x {x}, y {y}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn toolbar_changes_refresh_pending_line_without_pointer_input_or_baking_pixels() {
        let mut tools = ToolsPanel {
            active_tool: Tool::Line,
            ..Default::default()
        };
        let mut state = CanvasState::new(64, 32);
        let original = state.layers[0].pixels.to_rgba_image();
        tools.capture_line_target(&state);
        tools.line_state.line_tool.stage = LineStage::Editing;
        tools.line_state.line_tool.control_points = [
            Pos2::new(8., 16.),
            Pos2::new(20., 16.),
            Pos2::new(32., 16.),
            Pos2::new(44., 16.),
        ];
        tools.properties.size = 1.;
        tools.line_state.line_tool.options.anti_alias = false;
        tools.refresh_line_preview(&mut state, Color32::RED);
        let before = state.preview_layer.as_ref().unwrap().to_rgba_image();
        tools.properties.size = 4.;
        tools.properties.blending_mode = BlendMode::HardLight;
        tools.line_state.line_tool.options.pattern = LinePattern::Dashed;
        tools.refresh_line_preview(&mut state, Color32::BLUE);
        assert_ne!(
            before,
            state.preview_layer.as_ref().unwrap().to_rgba_image()
        );
        assert_eq!(tools.line_state.line_tool.last_color, Color32::BLUE);
        assert_eq!(tools.line_state.line_tool.last_blend, BlendMode::HardLight);
        assert!(
            state
                .preview_layer
                .as_ref()
                .unwrap()
                .to_rgba_image()
                .pixels()
                .any(|p| p[2] == 255 && p[3] > 0)
        );
        assert_eq!(state.layers[0].pixels.to_rgba_image(), original);
        tools.commit_pending_line_edit(&mut state);
        assert_eq!(tools.line_state.line_tool.stage, LineStage::Idle);
        assert!(
            state.layers[0]
                .pixels
                .to_rgba_image()
                .pixels()
                .any(|p| p[2] > 0)
        );
    }
}
