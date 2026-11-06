impl ToolsPanel {
    // CLONE STAMP helpers
    // ================================================================

    /// Stamp a single circle for clone stamp: sample from source layer at offset
    fn clone_stamp_circle(
        &self,
        canvas_state: &mut CanvasState,
        pos: (f32, f32),
        offset: Vec2,
    ) -> Rect {
        let (cx, cy) = pos;
        let radius = self.properties.size / 2.0;
        let width = canvas_state.width;
        let height = canvas_state.height;

        let min_x = (cx - radius).max(0.0) as u32;
        let max_x = ((cx + radius) as u32).min(width.saturating_sub(1));
        let min_y = (cy - radius).max(0.0) as u32;
        let max_y = ((cy + radius) as u32).min(height.saturating_sub(1));

        // Get source layer pointer for safe aliasing
        let layer = match canvas_state.layers.get(canvas_state.active_layer_index) {
            Some(l) => l,
            None => return Rect::NOTHING,
        };
        let src_ptr = &layer.pixels as *const TiledImage;
        let sel_ptr = canvas_state
            .selection_mask
            .as_ref()
            .map(|m| m as *const GrayImage);

        if let Some(ref mut preview) = canvas_state.preview_layer {
            for y in min_y..=max_y {
                for x in min_x..=max_x {
                    // Selection mask check
                    if let Some(mask_p) = sel_ptr {
                        let mask = unsafe { &*mask_p };
                        if x < mask.width() && y < mask.height() {
                            if mask.get_pixel(x, y).0[0] == 0 {
                                continue;
                            }
                        } else {
                            continue;
                        }
                    }

                    let dx = x as f32 - cx;
                    let dy = y as f32 - cy;
                    let dist = (dx * dx + dy * dy).sqrt();
                    if dist > radius {
                        continue;
                    }

                    let geom_alpha = self.compute_brush_alpha(dist, radius);
                    if geom_alpha < 0.01 {
                        continue;
                    }

                    // Source coordinates
                    let sx = (x as f32 + offset.x).round() as i32;
                    let sy = (y as f32 + offset.y).round() as i32;
                    if sx < 0 || sx >= width as i32 || sy < 0 || sy >= height as i32 {
                        continue;
                    }

                    let src_pixel = unsafe { &*src_ptr }.get_pixel(sx as u32, sy as u32);
                    let sr = src_pixel.0[0] as f32 / 255.0;
                    let sg = src_pixel.0[1] as f32 / 255.0;
                    let sb = src_pixel.0[2] as f32 / 255.0;
                    let sa = src_pixel.0[3] as f32 / 255.0;

                    let brush_alpha = geom_alpha * sa;
                    let old_alpha = preview.get_pixel(x, y).0[3] as f32 / 255.0;

                    if brush_alpha >= old_alpha {
                        *preview.get_pixel_mut(x, y) = Rgba([
                            (sr * 255.0) as u8,
                            (sg * 255.0) as u8,
                            (sb * 255.0) as u8,
                            (brush_alpha * 255.0) as u8,
                        ]);
                    }
                }
            }
        }

        Rect::from_min_max(
            Pos2::new(
                min_x.saturating_sub(1) as f32,
                min_y.saturating_sub(1) as f32,
            ),
            Pos2::new(
                (max_x + 2).min(width) as f32,
                (max_y + 2).min(height) as f32,
            ),
        )
    }

    /// Draw a line for clone stamp — dense stepping like draw_line_no_dirty
    fn clone_stamp_line(
        &self,
        canvas_state: &mut CanvasState,
        start: (f32, f32),
        end: (f32, f32),
        offset: Vec2,
    ) -> Rect {
        let dx = end.0 - start.0;
        let dy = end.1 - start.1;
        let distance = (dx * dx + dy * dy).sqrt();

        if distance < 0.1 {
            return self.clone_stamp_circle(canvas_state, start, offset);
        }

        let steps = distance.ceil() as usize;
        let mut dirty = Rect::NOTHING;

        for i in 0..=steps {
            let t = i as f32 / steps as f32;
            let x = start.0 + dx * t;
            let y = start.1 + dy * t;
            if x >= 0.0
                && (x as u32) < canvas_state.width
                && y >= 0.0
                && (y as u32) < canvas_state.height
            {
                dirty = dirty.union(self.clone_stamp_circle(canvas_state, (x, y), offset));
            }
        }
        dirty
    }

    // ================================================================
    // CONTENT AWARE BRUSH (healing) helpers
    // ================================================================

    /// Heal a single circle: for each pixel inside the brush, sample surrounding
    /// pixels from a ring and replace with their average. This effectively
    /// "fills in" the area with surrounding texture.
    fn heal_circle(&self, canvas_state: &mut CanvasState, pos: (f32, f32)) -> Rect {
        let (cx, cy) = pos;
        let radius = self.properties.size / 2.0;
        let sample_radius = self.content_aware_state.sample_radius;
        let hardness = self.properties.hardness;
        let width = canvas_state.width;
        let height = canvas_state.height;

        let min_x = (cx - radius).max(0.0) as u32;
        let max_x = ((cx + radius) as u32).min(width.saturating_sub(1));
        let min_y = (cy - radius).max(0.0) as u32;
        let max_y = ((cy + radius) as u32).min(height.saturating_sub(1));

        let layer = match canvas_state.layers.get(canvas_state.active_layer_index) {
            Some(l) => l,
            None => return Rect::NOTHING,
        };
        let src_ptr = &layer.pixels as *const TiledImage;
        let sel_ptr = canvas_state
            .selection_mask
            .as_ref()
            .map(|m| m as *const GrayImage);

        // Sample at two rings (75% and 100% of sample_radius) with per-pixel
        // angle randomisation to avoid visible sampling-grid artifacts.
        let num_samples: usize = 24;

        if let Some(ref mut preview) = canvas_state.preview_layer {
            let src = unsafe { &*src_ptr };

            for y in min_y..=max_y {
                for x in min_x..=max_x {
                    // Selection mask
                    if let Some(mask_p) = sel_ptr {
                        let mask = unsafe { &*mask_p };
                        if x < mask.width() && y < mask.height() {
                            if mask.get_pixel(x, y).0[0] == 0 {
                                continue;
                            }
                        } else {
                            continue;
                        }
                    }

                    let dx = x as f32 - cx;
                    let dy = y as f32 - cy;
                    let dist = (dx * dx + dy * dy).sqrt();
                    if dist > radius {
                        continue;
                    }

                    // Hardness-aware brush alpha
                    let t = (dist / radius).clamp(0.0, 1.0);
                    let hard_t = (hardness * 0.9 + 0.1).clamp(0.0, 1.0);
                    let geom_alpha = if t < hard_t {
                        1.0
                    } else {
                        let s = (t - hard_t) / (1.0 - hard_t + 1e-6);
                        1.0 - s * s * (3.0 - 2.0 * s)
                    };
                    if geom_alpha < 0.01 {
                        continue;
                    }

                    // Per-pixel angle offset to break up ring-sampling grid artifacts
                    let angle_seed = (x.wrapping_mul(1619)).wrapping_add(y.wrapping_mul(3929));
                    let angle_offset = angle_seed as f32 / u32::MAX as f32 * std::f32::consts::TAU;

                    let mut sum_r = 0.0_f32;
                    let mut sum_g = 0.0_f32;
                    let mut sum_b = 0.0_f32;
                    let mut count = 0.0_f32;

                    for i in 0..num_samples {
                        let angle =
                            angle_offset + (i as f32 / num_samples as f32) * std::f32::consts::TAU;
                        // Sample at two concentric rings for smoother coverage
                        for &rr in &[sample_radius * 0.75, sample_radius] {
                            let sx = (x as f32 + angle.cos() * rr).round() as i32;
                            let sy = (y as f32 + angle.sin() * rr).round() as i32;
                            if sx < 0 || sx >= width as i32 || sy < 0 || sy >= height as i32 {
                                continue;
                            }
                            let sp = src.get_pixel(sx as u32, sy as u32);
                            sum_r += sp.0[0] as f32;
                            sum_g += sp.0[1] as f32;
                            sum_b += sp.0[2] as f32;
                            count += 1.0;
                        }
                    }

                    if count < 1.0 {
                        continue;
                    }

                    let old_alpha = preview.get_pixel(x, y).0[3] as f32 / 255.0;
                    if geom_alpha >= old_alpha {
                        *preview.get_pixel_mut(x, y) = Rgba([
                            (sum_r / count) as u8,
                            (sum_g / count) as u8,
                            (sum_b / count) as u8,
                            (geom_alpha * 255.0) as u8,
                        ]);
                    }
                }
            }
        }

        Rect::from_min_max(
            Pos2::new(
                min_x.saturating_sub(1) as f32,
                min_y.saturating_sub(1) as f32,
            ),
            Pos2::new(
                (max_x + 2).min(width) as f32,
                (max_y + 2).min(height) as f32,
            ),
        )
    }

    /// Draw a line for heal brush — dense stepping
    fn heal_line(
        &self,
        canvas_state: &mut CanvasState,
        start: (f32, f32),
        end: (f32, f32),
    ) -> Rect {
        let dx = end.0 - start.0;
        let dy = end.1 - start.1;
        let distance = (dx * dx + dy * dy).sqrt();

        if distance < 0.1 {
            return self.heal_circle(canvas_state, start);
        }

        let steps = distance.ceil() as usize;
        let mut dirty = Rect::NOTHING;

        for i in 0..=steps {
            let t = i as f32 / steps as f32;
            let x = start.0 + dx * t;
            let y = start.1 + dy * t;
            if x >= 0.0
                && (x as u32) < canvas_state.width
                && y >= 0.0
                && (y as u32) < canvas_state.height
            {
                dirty = dirty.union(self.heal_circle(canvas_state, (x, y)));
            }
        }
        dirty
    }

    /// Draw a single pixel immediately to pixels and return its bounding box
    fn draw_pixel_and_get_bounds(
        &mut self,
        canvas_state: &mut CanvasState,
        pos: (f32, f32),
        use_secondary: bool,
        primary_color_f32: [f32; 4],
        secondary_color_f32: [f32; 4],
    ) -> Rect {
        let (cx, cy) = pos;

        // Calculate bounds (single pixel)
        let width = canvas_state.width;
        let height = canvas_state.height;

        let x = cx.floor() as u32;
        let y = cy.floor() as u32;

        // Check bounds
        if x >= width || y >= height {
            return Rect::NOTHING;
        }

        // Determine which color to use (high-precision unmultiplied)
        let pixel_color_f32 = if use_secondary {
            secondary_color_f32
        } else {
            primary_color_f32
        };

        // Unpack high-precision source
        let [src_r, src_g, src_b, src_a] = pixel_color_f32;

        // Skip pixels outside the selection mask
        if let Some(mask) = &canvas_state.selection_mask {
            if x < mask.width() && y < mask.height() {
                if mask.get_pixel(x, y).0[0] == 0 {
                    return Rect::NOTHING;
                }
            } else {
                return Rect::NOTHING;
            }
        }

        // Get the target image (preview layer for pencil)
        if let Some(ref mut preview) = canvas_state.preview_layer {
            let pixel = preview.get_pixel_mut(x, y);

            // For pencil, use Max-Alpha blending to prevent opacity accumulation
            // when dragging over the same area. This is the same technique the
            // brush uses to prevent opacity stacking.
            let pencil_alpha = src_a; // 1.0 for pencil ink (full opacity color)

            // Read existing alpha from the preview layer
            let old_alpha = pixel.0[3] as f32 / 255.0;

            // Only update pixel if we're increasing opacity (or equal)
            // This prevents less-opaque strokes from overwriting more-opaque ones
            if pencil_alpha >= old_alpha {
                *pixel = Rgba([
                    (src_r * 255.0) as u8,
                    (src_g * 255.0) as u8,
                    (src_b * 255.0) as u8,
                    (pencil_alpha * 255.0) as u8,
                ]);
            }
        }

        // Return the bounding box (single pixel with 1 pixel padding)
        Rect::from_min_max(
            Pos2::new(x.saturating_sub(1) as f32, y.saturating_sub(1) as f32),
            Pos2::new((x + 2).min(width) as f32, (y + 2).min(height) as f32),
        )
    }

    /// Draw a line of single pixels and return its bounding box
    fn draw_pixel_line_and_get_bounds(
        &mut self,
        canvas_state: &mut CanvasState,
        start: (f32, f32),
        end: (f32, f32),
        use_secondary: bool,
        primary_color_f32: [f32; 4],
        secondary_color_f32: [f32; 4],
    ) -> Rect {
        let width = canvas_state.width;
        let height = canvas_state.height;

        // Calculate line bounds
        let _min_x = start.0.min(end.0) as u32;
        let _max_x = end.0.max(start.0) as u32;
        let _min_y = start.1.min(end.1) as u32;
        let _max_y = end.1.max(start.1) as u32;

        // Use Bresenham's line algorithm to draw pixel-perfect line
        let mut x0 = start.0.floor() as i32;
        let mut y0 = start.1.floor() as i32;
        let x1 = end.0.floor() as i32;
        let y1 = end.1.floor() as i32;

        let dx = (x1 - x0).abs();
        let dy = (y1 - y0).abs();
        let sx = if x0 < x1 { 1 } else { -1 };
        let sy = if y0 < y1 { 1 } else { -1 };
        let mut err = dx - dy;

        let mut bounds = Rect::NOTHING;

        loop {
            // Check bounds
            if x0 >= 0 && x0 < width as i32 && y0 >= 0 && y0 < height as i32 {
                let pixel_rect = self.draw_pixel_and_get_bounds(
                    canvas_state,
                    (x0 as f32, y0 as f32),
                    use_secondary,
                    primary_color_f32,
                    secondary_color_f32,
                );
                bounds = bounds.union(pixel_rect);
            }

            if x0 == x1 && y0 == y1 {
                break;
            }

            let e2 = 2 * err;
            if e2 > -dy {
                err -= dy;
                x0 += sx;
            }
            if e2 < dx {
                err += dx;
                y0 += sy;
            }
        }

        bounds
    }

    /// Draw a line immediately to pixels and return its bounding box
    fn draw_line_and_get_bounds(
        &mut self,
        canvas_state: &mut CanvasState,
        start: (f32, f32),
        end: (f32, f32),
        is_eraser: bool,
        use_secondary: bool,
        primary_color_f32: [f32; 4],
        secondary_color_f32: [f32; 4],
    ) -> Rect {
        // B6: Ensure brush alpha LUT is up-to-date
        self.rebuild_brush_lut();
        // Increment stamp counter for random rotation seeding (for the whole line)
        self.stamp_counter = self.stamp_counter.wrapping_add(1);

        let radius = self.maximum_brush_radius();
        let width = canvas_state.width;
        let height = canvas_state.height;

        // Calculate bounding box of the line + brush radius + max scatter offset
        let scatter_pad = self.properties.scatter * self.pressure_size();
        let min_x = (start.0.min(end.0) - radius - scatter_pad).max(0.0) as u32;
        let max_x = ((start.0.max(end.0) + radius + scatter_pad) as u32).min(width);
        let min_y = (start.1.min(end.1) - radius - scatter_pad).max(0.0) as u32;
        let max_y = ((start.1.max(end.1) + radius + scatter_pad) as u32).min(height);

        // All tools (Brush, Pencil, Eraser) write to the preview layer.
        // The eraser writes an erase-strength mask; the compositor handles the rest.
        {
            let mask_ptr = canvas_state
                .selection_mask
                .as_ref()
                .map(|m| m as *const GrayImage);
            if let Some(ref mut preview) = canvas_state.preview_layer {
                let mask_ref = mask_ptr.map(|p| unsafe { &*p });
                self.draw_line_no_dirty(
                    preview,
                    width,
                    height,
                    start,
                    end,
                    is_eraser,
                    use_secondary,
                    primary_color_f32,
                    secondary_color_f32,
                    mask_ref,
                );
            }
        }

        // Return the bounding box (add 1 pixel padding)
        Rect::from_min_max(
            Pos2::new(
                min_x.saturating_sub(1) as f32,
                min_y.saturating_sub(1) as f32,
            ),
            Pos2::new(
                (max_x + 2).min(width) as f32,
                (max_y + 2).min(height) as f32,
            ),
        )
    }
}
