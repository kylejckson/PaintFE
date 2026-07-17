impl Canvas {
    fn draw_pixel_grid(
        &self,
        painter: &egui::Painter,
        image_rect: Rect,
        state: &CanvasState,
        viewport: Rect,
        settings: &crate::assets::AppSettings,
    ) {
        let pixel_size = self.zoom;

        // Only draw grid if pixels are large enough to see
        if pixel_size < 4.0 {
            return;
        }

        // Calculate visible range in image coordinates
        let visible_rect = image_rect.intersect(viewport);

        // Convert to pixel coordinates
        let start_x = ((visible_rect.min.x - image_rect.min.x) / pixel_size)
            .floor()
            .max(0.0) as u32;
        let end_x = ((visible_rect.max.x - image_rect.min.x) / pixel_size)
            .ceil()
            .min(state.width as f32) as u32;
        let start_y = ((visible_rect.min.y - image_rect.min.y) / pixel_size)
            .floor()
            .max(0.0) as u32;
        let end_y = ((visible_rect.max.y - image_rect.min.y) / pixel_size)
            .ceil()
            .min(state.height as f32) as u32;

        // Grid colors from user settings (dual-stroke: outline + center)
        let grid_outline = settings.pixel_grid_outline_color;
        let grid_center = settings.pixel_grid_center_color;

        // Adaptive stroke width: thinner lines as zoom increases for less visual clutter
        // At zoom 8 (minimum grid display): base thickness (1.2 and 0.6, which is 40% smaller than original 2.0 and 1.0)
        // At zoom 20+ (far zoomed in): minimal thickness (0.5 and 0.3)
        let base_outline = 1.2;
        let base_center = 0.6;
        let reference_zoom = 8.0; // zoom level where we use base thickness
        let outline_stroke = (base_outline * reference_zoom / pixel_size)
            .max(0.5)
            .min(base_outline);
        let center_stroke = (base_center * reference_zoom / pixel_size)
            .max(0.3)
            .min(base_center);

        // Draw vertical lines with dual-stroke (black outline + white center)
        for x in start_x..=end_x {
            let screen_x = image_rect.min.x + x as f32 * pixel_size;
            if screen_x >= visible_rect.min.x && screen_x <= visible_rect.max.x {
                let p0 = Pos2::new(screen_x, visible_rect.min.y.max(image_rect.min.y));
                let p1 = Pos2::new(screen_x, visible_rect.max.y.min(image_rect.max.y));
                // Draw black outline first
                painter.line_segment([p0, p1], (outline_stroke, grid_outline));
                // Draw white center line on top
                painter.line_segment([p0, p1], (center_stroke, grid_center));
            }
        }

        // Draw horizontal lines with dual-stroke (black outline + white center)
        for y in start_y..=end_y {
            let screen_y = image_rect.min.y + y as f32 * pixel_size;
            if screen_y >= visible_rect.min.y && screen_y <= visible_rect.max.y {
                let p0 = Pos2::new(visible_rect.min.x.max(image_rect.min.x), screen_y);
                let p1 = Pos2::new(visible_rect.max.x.min(image_rect.max.x), screen_y);
                // Draw black outline first
                painter.line_segment([p0, p1], (outline_stroke, grid_outline));
                // Draw white center line on top
                painter.line_segment([p0, p1], (center_stroke, grid_center));
            }
        }
    }

    // ========================================================================
    // GUIDELINES OVERLAY  – center cross + rule-of-thirds
    // ========================================================================

    fn draw_guidelines(
        &self,
        painter: &egui::Painter,
        image_rect: Rect,
        state: &CanvasState,
        viewport: Rect,
    ) {
        let visible = image_rect.intersect(viewport);
        if visible.width() <= 0.0 || visible.height() <= 0.0 {
            return;
        }

        // Dual-stroke colors for visibility on any background
        let outline_color = Color32::from_black_alpha(100);
        let center_color = Color32::from_white_alpha(160);
        let outline_w = 1.5;
        let center_w = 0.7;

        let w = state.width as f32;
        let h = state.height as f32;

        // Helper: canvas X → screen X
        let sx = |cx: f32| image_rect.min.x + cx * self.zoom;
        // Helper: canvas Y → screen Y
        let sy = |cy: f32| image_rect.min.y + cy * self.zoom;

        // Clamp helpers for the visible viewport
        let clamp_x = |x: f32| x.max(visible.min.x).min(visible.max.x);
        let clamp_y = |y: f32| y.max(visible.min.y).min(visible.max.y);

        // Draw a single dual-stroke line (clipped to visible area)
        let draw_h = |cy: f32| {
            let screen_y = sy(cy);
            if screen_y >= visible.min.y && screen_y <= visible.max.y {
                let p0 = Pos2::new(clamp_x(image_rect.min.x), screen_y);
                let p1 = Pos2::new(clamp_x(image_rect.max.x), screen_y);
                painter.line_segment([p0, p1], (outline_w, outline_color));
                painter.line_segment([p0, p1], (center_w, center_color));
            }
        };
        let draw_v = |cx: f32| {
            let screen_x = sx(cx);
            if screen_x >= visible.min.x && screen_x <= visible.max.x {
                let p0 = Pos2::new(screen_x, clamp_y(image_rect.min.y));
                let p1 = Pos2::new(screen_x, clamp_y(image_rect.max.y));
                painter.line_segment([p0, p1], (outline_w, outline_color));
                painter.line_segment([p0, p1], (center_w, center_color));
            }
        };

        // Center lines
        draw_h(h / 2.0);
        draw_v(w / 2.0);

        // Rule-of-thirds lines
        draw_h(h / 3.0);
        draw_h(h * 2.0 / 3.0);
        draw_v(w / 3.0);
        draw_v(w * 2.0 / 3.0);
    }

    fn paint_composite_texture(
        &self,
        painter: &egui::Painter,
        image_rect: Rect,
        viewport: Rect,
        state: &CanvasState,
        tint: Color32,
    ) {
        if tint.a() == 0 {
            return;
        }
        let visible = image_rect.intersect(viewport);
        if visible.width() <= 0.0 || visible.height() <= 0.0 {
            return;
        }

        let clipped_painter = painter.with_clip_rect(viewport);
        let uv = Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0));
        if let Some(texture) = self.native_composite_texture {
            clipped_painter.image(texture, image_rect, uv, tint);
        } else if let Some(texture) = &state.composite_cache {
            clipped_painter.image(texture.id(), image_rect, uv, tint);
        }
    }

    fn paint_preview_texture(
        &self,
        painter: &egui::Painter,
        image_rect: Rect,
        viewport: Rect,
        state: &CanvasState,
        tint: Color32,
        draw_eraser_checkerboard: bool,
    ) {
        if tint.a() == 0 {
            return;
        }
        let visible = image_rect.intersect(viewport);
        if visible.width() <= 0.0 || visible.height() <= 0.0 {
            return;
        }

        let Some(tex) = state.preview_texture_cache.as_ref() else {
            return;
        };

        let clipped_painter = painter.with_clip_rect(viewport);
        let uv = Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0));

        // Use image_rect-derived proportional positioning so the preview
        // aligns exactly with the GPU composite. image_rect is independently
        // rounded from temp_rect, so its effective zoom can differ slightly.
        if let Some(sb) = state.preview_stroke_bounds {
            let off_x = sb.min.x.max(0.0).floor();
            let off_y = sb.min.y.max(0.0).floor();
            let sw = (sb.max.x.ceil().min(state.width as f32) - off_x).max(1.0);
            let sh = (sb.max.y.ceil().min(state.height as f32) - off_y).max(1.0);
            let canvas_w = state.width as f32;
            let canvas_h = state.height as f32;
            let sub_rect = Rect::from_min_size(
                Pos2::new(
                    image_rect.min.x + off_x / canvas_w * image_rect.width(),
                    image_rect.min.y + off_y / canvas_h * image_rect.height(),
                ),
                Vec2::new(
                    sw / canvas_w * image_rect.width(),
                    sh / canvas_h * image_rect.height(),
                ),
            );

            if draw_eraser_checkerboard
                && (state.preview_is_eraser
                    || matches!(
                        state.preview_blend_mode,
                        BlendMode::Overwrite | BlendMode::Xor
                    ))
                && let Some(ref checker_tex) = self.checkerboard_texture
            {
                let cell = 10.0_f32;
                let (cols, rows) = self.checkerboard_cached_size;
                let grid_ox = (viewport.min.x / cell).floor() * cell;
                let grid_oy = (viewport.min.y / cell).floor() * cell;
                let tex_w = cols as f32 * cell;
                let tex_h = rows as f32 * cell;
                let checker_uv = Rect::from_min_max(
                    Pos2::new(
                        (sub_rect.min.x - grid_ox) / tex_w,
                        (sub_rect.min.y - grid_oy) / tex_h,
                    ),
                    Pos2::new(
                        (sub_rect.max.x - grid_ox) / tex_w,
                        (sub_rect.max.y - grid_oy) / tex_h,
                    ),
                );
                clipped_painter.image(checker_tex.id(), sub_rect, checker_uv, tint);
            }

            clipped_painter.image(tex.id(), sub_rect, uv, tint);
        } else {
            if draw_eraser_checkerboard
                && (state.preview_is_eraser
                    || matches!(
                        state.preview_blend_mode,
                        BlendMode::Overwrite | BlendMode::Xor
                    ))
                && let Some(ref checker_tex) = self.checkerboard_texture
            {
                let cell = 10.0_f32;
                let (cols, rows) = self.checkerboard_cached_size;
                let grid_ox = (viewport.min.x / cell).floor() * cell;
                let grid_oy = (viewport.min.y / cell).floor() * cell;
                let tex_w = cols as f32 * cell;
                let tex_h = rows as f32 * cell;
                let checker_uv = Rect::from_min_max(
                    Pos2::new(
                        (image_rect.min.x - grid_ox) / tex_w,
                        (image_rect.min.y - grid_oy) / tex_h,
                    ),
                    Pos2::new(
                        (image_rect.max.x - grid_ox) / tex_w,
                        (image_rect.max.y - grid_oy) / tex_h,
                    ),
                );
                clipped_painter.image(checker_tex.id(), image_rect, checker_uv, tint);
            }

            clipped_painter.image(tex.id(), image_rect, uv, tint);
        }
    }

    fn draw_wrap_preview(
        &self,
        painter: &egui::Painter,
        image_rect: Rect,
        viewport: Rect,
        state: &CanvasState,
    ) {
        let tint = Color32::from_white_alpha(112);
        let clipped_painter = painter.with_clip_rect(viewport);
        let border_outer = egui::Stroke::new(1.5, Color32::from_black_alpha(72));
        let border_inner = egui::Stroke::new(0.75, Color32::from_white_alpha(52));
        let offsets = [
            Vec2::new(-image_rect.width(), 0.0),
            Vec2::new(image_rect.width(), 0.0),
            Vec2::new(0.0, -image_rect.height()),
            Vec2::new(0.0, image_rect.height()),
            Vec2::new(-image_rect.width(), -image_rect.height()),
            Vec2::new(image_rect.width(), -image_rect.height()),
            Vec2::new(-image_rect.width(), image_rect.height()),
            Vec2::new(image_rect.width(), image_rect.height()),
        ];

        for offset in offsets {
            let ghost_rect = image_rect.translate(offset);
            let visible = ghost_rect.intersect(viewport);
            if visible.width() <= 0.0 || visible.height() <= 0.0 {
                continue;
            }

            self.paint_composite_texture(painter, ghost_rect, viewport, state, tint);
            self.paint_preview_texture(painter, ghost_rect, viewport, state, tint, false);
            clipped_painter.rect_stroke(ghost_rect, 0.0, border_outer, egui::StrokeKind::Middle);
            clipped_painter.rect_stroke(
                ghost_rect.shrink(0.5),
                0.0,
                border_inner,
                egui::StrokeKind::Middle,
            );
        }
    }

    // ========================================================================
    // MIRROR AXIS OVERLAY  – dashed symmetry lines
    // ========================================================================

    fn draw_mirror_overlay(
        &self,
        painter: &egui::Painter,
        image_rect: Rect,
        state: &CanvasState,
        viewport: Rect,
    ) {
        let visible = image_rect.intersect(viewport);
        if visible.width() <= 0.0 || visible.height() <= 0.0 {
            return;
        }

        let w = state.width as f32;
        let h = state.height as f32;

        // Mirror axis colors — cyan/magenta for high contrast on any content
        let outline_color = Color32::from_rgba_premultiplied(0, 0, 0, 140);
        let line_color_h = Color32::from_rgb(0, 200, 220); // cyan for horizontal (vertical axis)
        let line_color_v = Color32::from_rgb(220, 80, 220); // magenta for vertical (horizontal axis)
        let outline_w = 2.5;
        let center_w = 1.2;

        let sx = |cx: f32| image_rect.min.x + cx * self.zoom;
        let sy = |cy: f32| image_rect.min.y + cy * self.zoom;
        let clamp_x = |x: f32| x.max(visible.min.x).min(visible.max.x);
        let clamp_y = |y: f32| y.max(visible.min.y).min(visible.max.y);

        let draw_dashed_h = |cy: f32, color: Color32| {
            let screen_y = sy(cy);
            if screen_y < visible.min.y || screen_y > visible.max.y {
                return;
            }
            let x_start = clamp_x(image_rect.min.x);
            let x_end = clamp_x(image_rect.max.x);
            let dash_len = 8.0_f32;
            let gap_len = 4.0_f32;
            let total = dash_len + gap_len;
            let mut x = x_start;
            while x < x_end {
                let x1 = (x + dash_len).min(x_end);
                let p0 = Pos2::new(x, screen_y);
                let p1 = Pos2::new(x1, screen_y);
                painter.line_segment([p0, p1], (outline_w, outline_color));
                painter.line_segment([p0, p1], (center_w, color));
                x += total;
            }
        };

        let draw_dashed_v = |cx: f32, color: Color32| {
            let screen_x = sx(cx);
            if screen_x < visible.min.x || screen_x > visible.max.x {
                return;
            }
            let y_start = clamp_y(image_rect.min.y);
            let y_end = clamp_y(image_rect.max.y);
            let dash_len = 8.0_f32;
            let gap_len = 4.0_f32;
            let total = dash_len + gap_len;
            let mut y = y_start;
            while y < y_end {
                let y1 = (y + dash_len).min(y_end);
                let p0 = Pos2::new(screen_x, y);
                let p1 = Pos2::new(screen_x, y1);
                painter.line_segment([p0, p1], (outline_w, outline_color));
                painter.line_segment([p0, p1], (center_w, color));
                y += total;
            }
        };

        match state.mirror_mode {
            MirrorMode::Horizontal => {
                draw_dashed_v((w - 1.0) * 0.5, line_color_h);
            }
            MirrorMode::Vertical => {
                draw_dashed_h((h - 1.0) * 0.5, line_color_v);
            }
            MirrorMode::Quarters => {
                draw_dashed_v((w - 1.0) * 0.5, line_color_h);
                draw_dashed_h((h - 1.0) * 0.5, line_color_v);
            }
            MirrorMode::None => {}
        }
    }

    // ========================================================================
    // SELECTION VISUALIZATION  – marching ants / glow overlay
    // ========================================================================

    /// Build (or reuse) the cached selection overlay RGBA texture, then
    /// draw it as a single GPU-composited quad.  The border segments are
    /// still drawn with immediate-mode lines since they're already
    /// merged into relatively few segments.
    fn draw_selection_transform_preview(
        &self,
        painter: &egui::Painter,
        image_rect: Rect,
        bounds: (u32, u32, u32, u32),
    ) {
        let (x0, y0, x1, y1) = bounds;
        let rect = Rect::from_min_max(
            Pos2::new(
                image_rect.min.x + x0 as f32 * self.zoom,
                image_rect.min.y + y0 as f32 * self.zoom,
            ),
            Pos2::new(
                image_rect.min.x + (x1 + 1) as f32 * self.zoom,
                image_rect.min.y + (y1 + 1) as f32 * self.zoom,
            ),
        );
        painter.rect_filled(
            rect,
            0.0,
            Color32::from_rgba_premultiplied(255, 255, 255, 12),
        );
        painter.rect_stroke(
            rect,
            0.0,
            egui::Stroke::new(3.5, Color32::from_rgba_premultiplied(0, 0, 0, 210)),
            egui::StrokeKind::Middle,
        );
        painter.rect_stroke(
            rect,
            0.0,
            egui::Stroke::new(1.5, Color32::WHITE),
            egui::StrokeKind::Middle,
        );
    }

    fn draw_full_selection_overlay(
        &self,
        painter: &egui::Painter,
        image_rect: Rect,
        state: &mut CanvasState,
    ) {
        let rect = image_rect.intersect(painter.clip_rect());
        if rect.is_negative() {
            return;
        }
        painter.rect_filled(
            rect,
            0.0,
            Color32::from_rgba_premultiplied(255, 255, 255, 12),
        );
        painter.rect_stroke(
            image_rect,
            0.0,
            egui::Stroke::new(3.5, Color32::from_rgba_premultiplied(0, 0, 0, 210)),
            egui::StrokeKind::Inside,
        );
        painter.rect_stroke(
            image_rect,
            0.0,
            egui::Stroke::new(1.5, Color32::WHITE),
            egui::StrokeKind::Inside,
        );
        state.selection_overlay_bounds = (state.width > 0 && state.height > 0).then_some((
            0,
            0,
            state.width - 1,
            state.height - 1,
        ));
        state.selection_overlay_built_generation = state.selection_overlay_generation;
        state.selection_border_built_generation = state.selection_overlay_generation;
        state.selection_overlay_texture = None;
        state.selection_border_h_segs.clear();
        state.selection_border_v_segs.clear();
    }

    fn draw_selection_overlay(
        &self,
        painter: &egui::Painter,
        image_rect: Rect,
        mask: &GrayImage,
        time: f64,
        tool_active: bool,
        state: &mut CanvasState,
        ctx: &egui::Context,
    ) {
        let w = mask.width();
        let h = mask.height();
        let zoom = self.zoom;

        // --- 1. Find bounding box of selected pixels to limit work. ----------
        let mask_raw = mask.as_raw();
        let stride = w as usize;

        let (min_x, min_y, max_x, max_y) = if let Some(b) = state.selection_overlay_bounds {
            // Fast path: reuse previously computed bounds if generation hasn't changed
            if state.selection_overlay_built_generation == state.selection_overlay_generation {
                b
            } else {
                Self::compute_mask_bounds(mask_raw, w, h, stride)
            }
        } else {
            Self::compute_mask_bounds(mask_raw, w, h, stride)
        };

        if min_x > max_x || min_y > max_y {
            return; // Nothing selected
        }

        let should_animate_interior =
            selection_overlay_should_animate(Some((min_x, min_y, max_x, max_y)));
        let clip_rect = painter.clip_rect();

        let sel = |x: u32, y: u32| -> bool { mask_raw[(y as usize) * stride + (x as usize)] > 0 };

        // --- 2. Selection interior overlay via GPU-cached texture. -----------
        // The crosshatch pattern is baked into an RGBA texture at canvas-pixel
        // resolution (cropped to the selection bounding box).  The texture is
        // rebuilt only when the selection mask changes or the animation offset
        // ticks forward.  The GPU handles zoom/display for free.
        if !tool_active {
            // Animation: smoothly scroll pattern at ~1.5 canvas-pixels per second.
            // Using a float modulo (no integer cast) so the offset is continuous and
            // the texture rebuilds in small sub-pixel increments instead of whole-pixel
            // jumps, eliminating the jitter visible at high zoom levels.
            let band_period = 8u32; // canvas-pixel diagonal period
            let period_f = (band_period * 2) as f32;
            let anim_offset = ((time * 3.0) % (period_f as f64)) as f32;

            let generation_changed =
                state.selection_overlay_built_generation != state.selection_overlay_generation;
            // Rebuild when the fractional offset shifts by ≥0.15 canvas pixels
            // (~10 rebuilds/sec at 1.5 px/s) — enough for smooth motion without
            // rebuilding a potentially large texture on every single frame.
            let anim_changed = should_animate_interior
                && (anim_offset - state.selection_overlay_anim_offset).abs() > 0.15;
            let needs_rebuild =
                state.selection_overlay_texture.is_none() || generation_changed || anim_changed;

            if needs_rebuild {
                let bw = (max_x - min_x + 1) as usize;
                let bh = (max_y - min_y + 1) as usize;
                let buf_len = bw * bh * 4;

                // Build RGBA buffer with diagonal-stripe pattern.
                // Low alpha values keep the image behind clearly readable; the slight
                // alpha difference between the two bands creates a very gentle stripe
                // without harsh contrast.
                let mut buf = vec![0u8; buf_len];
                let stripe = ctx.data_mut(|d| {
                    d.get_persisted::<Color32>(egui::Id::new("paintfe_selection_stripe_color"))
                        .unwrap_or(Color32::from_rgba_premultiplied(255, 255, 255, 255))
                });
                let stripe_alpha = ctx.data_mut(|d| {
                    d.get_persisted::<u8>(egui::Id::new("paintfe_selection_stripe_alpha"))
                        .unwrap_or(22)
                });
                for dy in 0..bh {
                    let cy = min_y + dy as u32; // canvas y
                    let row_off = dy * bw * 4;
                    for dx in 0..bw {
                        let cx = min_x + dx as u32; // canvas x
                        if mask_raw[(cy as usize) * stride + (cx as usize)] == 0 {
                            continue; // transparent (unselected)
                        }
                        // Continuous float diagonal coordinate with smooth animation offset
                        let diag = (cx as f32 + cy as f32 + anim_offset).rem_euclid(period_f);
                        let is_dark = diag < band_period as f32;
                        let px = row_off + dx * 4;
                        if is_dark {
                            buf[px] = stripe.r();
                            buf[px + 1] = stripe.g();
                            buf[px + 2] = stripe.b();
                            buf[px + 3] = stripe_alpha;
                        } else {
                            buf[px] = 255u8.saturating_sub(stripe.r());
                            buf[px + 1] = 255u8.saturating_sub(stripe.g());
                            buf[px + 2] = 255u8.saturating_sub(stripe.b());
                            buf[px + 3] = stripe_alpha / 2;
                        }
                    }
                }

                let color_image = ColorImage::from_rgba_unmultiplied([bw, bh], &buf);
                // Use Nearest filtering so individual pixels stay crisp at high zoom
                let tex_options = TextureOptions {
                    magnification: TextureFilter::Nearest,
                    minification: TextureFilter::Linear,
                    ..Default::default()
                };
                if let Some(ref mut tex) = state.selection_overlay_texture {
                    tex.set(ImageData::Color(Arc::new(color_image)), tex_options);
                } else {
                    state.selection_overlay_texture = Some(ctx.load_texture(
                        "selection_overlay",
                        ImageData::Color(Arc::new(color_image)),
                        tex_options,
                    ));
                }
                state.selection_overlay_built_generation = state.selection_overlay_generation;
                state.selection_overlay_anim_offset = if should_animate_interior {
                    anim_offset
                } else {
                    0.0
                };
                state.selection_overlay_bounds = Some((min_x, min_y, max_x, max_y));
            }

            // Paint the cached texture at the correct position (pixel-snapped).
            if let Some(ref tex) = state.selection_overlay_texture
                && let Some((bx0, by0, bx1, by1)) = state.selection_overlay_bounds
            {
                let screen_x = (image_rect.min.x + bx0 as f32 * zoom).round();
                let screen_y = (image_rect.min.y + by0 as f32 * zoom).round();
                let screen_x1 = (image_rect.min.x + (bx1 + 1) as f32 * zoom).round();
                let screen_y1 = (image_rect.min.y + (by1 + 1) as f32 * zoom).round();
                let sub_rect = Rect::from_min_max(
                    Pos2::new(screen_x, screen_y),
                    Pos2::new(screen_x1, screen_y1),
                );
                let uv = Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0));
                painter.image(tex.id(), sub_rect, uv, Color32::WHITE);
            }
        } // end if !tool_active

        // --- 3. Collect boundary edges (cached in canvas coordinates). -------
        if state.selection_border_built_generation != state.selection_overlay_generation {
            state.selection_border_h_segs.clear();
            state.selection_border_v_segs.clear();

            // Horizontal edges
            for y_line in min_y..=max_y + 1 {
                let mut seg_x: Option<u32> = None;
                for x in min_x..=max_x {
                    let above = y_line > 0 && y_line - 1 < h && sel(x, y_line - 1);
                    let below = y_line < h && sel(x, y_line);
                    let boundary = above != below;
                    if boundary && seg_x.is_none() {
                        seg_x = Some(x);
                    } else if !boundary && let Some(sx) = seg_x {
                        state.selection_border_h_segs.push((y_line, sx, x));
                        seg_x = None;
                    }
                }
                if let Some(sx) = seg_x {
                    state.selection_border_h_segs.push((y_line, sx, max_x + 1));
                }
            }

            // Vertical edges
            for x_line in min_x..=max_x + 1 {
                let mut seg_y: Option<u32> = None;
                for y in min_y..=max_y {
                    let left = x_line > 0 && x_line - 1 < w && sel(x_line - 1, y);
                    let right = x_line < w && sel(x_line, y);
                    let boundary = left != right;
                    if boundary && seg_y.is_none() {
                        seg_y = Some(y);
                    } else if !boundary && let Some(sy) = seg_y {
                        state.selection_border_v_segs.push((x_line, sy, y));
                        seg_y = None;
                    }
                }
                if let Some(sy) = seg_y {
                    state.selection_border_v_segs.push((x_line, sy, max_y + 1));
                }
            }

            state.selection_border_built_generation = state.selection_overlay_generation;
        }

        // --- 4. Draw selection border from cached segments. ------------------
        let accent = self.selection_stroke;
        let stroke_width = 1.5;
        let [sr, sg, sb, _] = accent.to_array();
        let glow_alpha = if tool_active {
            let pulse = ((time * 3.0).sin() * 0.5 + 0.5) as f32;
            (40.0 + pulse * 100.0) as u8
        } else {
            80u8
        };
        let glow = Color32::from_rgba_unmultiplied(sr, sg, sb, glow_alpha);
        let glow_width = if tool_active {
            stroke_width + 4.0
        } else {
            stroke_width + 2.0
        };

        let total_segs = state.selection_border_h_segs.len() + state.selection_border_v_segs.len();
        let skip_glow = total_segs > 10_000;

        // Draw horizontal border segments (pixel-snapped)
        for &(y_line, x0, x1) in &state.selection_border_h_segs {
            let sy = (image_rect.min.y + y_line as f32 * zoom).round();
            let sx0 = (image_rect.min.x + x0 as f32 * zoom).round();
            let sx1 = (image_rect.min.x + x1 as f32 * zoom).round();
            if sy < clip_rect.min.y - glow_width || sy > clip_rect.max.y + glow_width {
                continue;
            }
            if sx1 < clip_rect.min.x - glow_width || sx0 > clip_rect.max.x + glow_width {
                continue;
            }
            let a = Pos2::new(sx0, sy);
            let b = Pos2::new(sx1, sy);
            if !skip_glow {
                painter.line_segment([a, b], egui::Stroke::new(glow_width, glow));
            }
            painter.line_segment([a, b], egui::Stroke::new(stroke_width + 0.5, accent));
        }

        // Draw vertical border segments (pixel-snapped)
        for &(x_line, y0, y1) in &state.selection_border_v_segs {
            let sx = (image_rect.min.x + x_line as f32 * zoom).round();
            let sy0 = (image_rect.min.y + y0 as f32 * zoom).round();
            let sy1 = (image_rect.min.y + y1 as f32 * zoom).round();
            if sx < clip_rect.min.x - glow_width || sx > clip_rect.max.x + glow_width {
                continue;
            }
            if sy1 < clip_rect.min.y - glow_width || sy0 > clip_rect.max.y + glow_width {
                continue;
            }
            let a = Pos2::new(sx, sy0);
            let b = Pos2::new(sx, sy1);
            if !skip_glow {
                painter.line_segment([a, b], egui::Stroke::new(glow_width, glow));
            }
            painter.line_segment([a, b], egui::Stroke::new(stroke_width + 0.5, accent));
        }
    }

    /// Compute the bounding box of selected pixels in a mask.
    fn compute_mask_bounds(mask_raw: &[u8], w: u32, h: u32, stride: usize) -> (u32, u32, u32, u32) {
        let mut min_x = w;
        let mut min_y = h;
        let mut max_x = 0u32;
        let mut max_y = 0u32;

        for y in 0..h {
            let row_offset = y as usize * stride;
            let row = &mask_raw[row_offset..row_offset + stride];
            let mut found_in_row = false;
            for (x_idx, &val) in row.iter().enumerate() {
                if val > 0 {
                    let x = x_idx as u32;
                    min_x = min_x.min(x);
                    max_x = max_x.max(x);
                    found_in_row = true;
                }
            }
            if found_in_row {
                min_y = min_y.min(y);
                max_y = max_y.max(y);
            }
        }

        (min_x, min_y, max_x, max_y)
    }

    /// Draw an animated "marching ants" rectangle border using theme colours.
    fn draw_marching_rect(&self, painter: &egui::Painter, rect: Rect, time: f64) {
        let accent = self.selection_stroke;
        let contrast = self.selection_contrast;

        let dash = 6.0f32;
        let gap = 4.0f32;
        let pattern = dash + gap;
        let offset = ((time * 30.0) % pattern as f64) as f32;
        let stroke_width = 1.5;

        let edges: [(Pos2, Pos2); 4] = [
            (rect.left_top(), rect.right_top()),
            (rect.right_top(), rect.right_bottom()),
            (rect.right_bottom(), rect.left_bottom()),
            (rect.left_bottom(), rect.left_top()),
        ];

        for (start, end) in &edges {
            self.draw_dashed_line(
                painter,
                *start,
                *end,
                dash,
                gap,
                offset,
                stroke_width,
                contrast,
            );
            let offset2 = (offset + dash) % pattern;
            self.draw_dashed_line(
                painter,
                *start,
                *end,
                gap,
                dash,
                offset2,
                stroke_width,
                accent,
            );
        }
    }

    /// Draw a dashed line segment.
    fn draw_dashed_line(
        &self,
        painter: &egui::Painter,
        a: Pos2,
        b: Pos2,
        dash: f32,
        gap: f32,
        offset: f32,
        width: f32,
        color: Color32,
    ) {
        let dir = b - a;
        let total = dir.length();
        if total < 0.1 {
            return;
        }
        let unit = dir / total;
        let pattern = dash + gap;
        let mut t = -offset; // start offset for animation
        while t < total {
            let seg_start = t.max(0.0);
            let seg_end = (t + dash).min(total);
            if seg_start < seg_end {
                let p0 = a + unit * seg_start;
                let p1 = a + unit * seg_end;
                painter.line_segment([p0, p1], egui::Stroke::new(width, color));
            }
            t += pattern;
        }
    }

    /// Draw an ellipse selection preview (filled + dashed border).
    fn draw_ellipse_overlay(&self, painter: &egui::Painter, rect: Rect, fill: Color32, time: f64) {
        let center = rect.center();
        let rx = rect.width() / 2.0;
        let ry = rect.height() / 2.0;

        // Approximate the ellipse with a polygon for fill.
        let segments = 64;
        let mut points = Vec::with_capacity(segments);
        for i in 0..segments {
            let angle = 2.0 * std::f32::consts::PI * (i as f32) / (segments as f32);
            points.push(Pos2::new(
                center.x + rx * angle.cos(),
                center.y + ry * angle.sin(),
            ));
        }

        // Fill with triangle fan
        if points.len() >= 3 {
            let mut mesh = egui::Mesh::default();
            for pt in &points {
                mesh.colored_vertex(*pt, fill);
            }
            for i in 1..(points.len() as u32 - 1) {
                mesh.add_triangle(0, i, i + 1);
            }
            painter.add(egui::Shape::mesh(mesh));
        }

        // Dashed border (walk the perimeter) using theme accent + contrast.
        let accent = self.selection_stroke;
        let contrast_col = self.selection_contrast;
        let stroke_width = 1.5;
        let dash = 6.0f32;
        let gap = 4.0f32;
        let pattern = dash + gap;
        let anim_speed = 30.0;
        let offset = ((time * anim_speed) % pattern as f64) as f32;

        for i in 0..points.len() {
            let a = points[i];
            let b = points[(i + 1) % points.len()];
            let seg_len = (b - a).length();
            if seg_len < 0.1 {
                continue;
            }

            self.draw_dashed_line(painter, a, b, dash, gap, offset, stroke_width, contrast_col);
            let offset2 = (offset + dash) % pattern;
            self.draw_dashed_line(painter, a, b, gap, dash, offset2, stroke_width, accent);
        }
    }

    /// Draw a static screen-space checkerboard behind the canvas image.
    ///
    /// The pattern uses a fixed 10px screen-pixel cell size (independent of
    /// zoom) and is anchored to screen coordinates — panning and zooming move
    /// the image content over the pattern, matching Paint.NET / Photoshop.
    ///
    /// Implementation: a small `ColorImage` (1 pixel per cell, Nearest filter)
    /// is cached and reused across frames.  Only 1 textured quad is drawn per
    /// frame regardless of canvas size or zoom level.
    fn draw_checkerboard(
        &mut self,
        painter: &egui::Painter,
        rect: Rect,
        clip: Rect,
        brightness: f32,
        ctx: &egui::Context,
    ) {
        let visible = rect.intersect(clip);
        if visible.is_negative() || visible.width() < 1.0 || visible.height() < 1.0 {
            return;
        }

        let cell = 10.0_f32; // fixed screen-pixel cell size

        // How many cells span the full viewport (canvas_rect).
        // We build the texture to cover the viewport so the pattern is stable
        // as the image pans within it.  +2 padding cells avoids edge gaps.
        let cols = (clip.width() / cell).ceil() as usize + 2;
        let rows = (clip.height() / cell).ceil() as usize + 2;

        // Rebuild texture if brightness changed or viewport cell count changed.
        if self.checkerboard_texture.is_none()
            || (self.checkerboard_brightness_cached - brightness).abs() > 0.001
            || self.checkerboard_cached_size != (cols, rows)
        {
            let light_val = (220.0 * brightness).clamp(0.0, 255.0) as u8;
            let dark_val = (180.0 * brightness).clamp(0.0, 255.0) as u8;
            let light = Color32::from_gray(light_val);
            let dark = Color32::from_gray(dark_val);

            let mut pixels = vec![light; cols * rows];
            for y in 0..rows {
                for x in 0..cols {
                    if (x + y) % 2 != 0 {
                        pixels[y * cols + x] = dark;
                    }
                }
            }

            let image = ColorImage {
                size: [cols, rows],
                source_size: egui::Vec2::new(cols as f32, rows as f32),
                pixels,
            };
            let tex_options = TextureOptions {
                magnification: TextureFilter::Nearest,
                minification: TextureFilter::Nearest,
                ..Default::default()
            };
            if let Some(ref mut tex) = self.checkerboard_texture {
                tex.set(ImageData::Color(Arc::new(image)), tex_options);
            } else {
                self.checkerboard_texture = Some(ctx.load_texture(
                    "checkerboard_bg",
                    ImageData::Color(Arc::new(image)),
                    tex_options,
                ));
            }
            self.checkerboard_brightness_cached = brightness;
            self.checkerboard_cached_size = (cols, rows);
        }

        let tex = self.checkerboard_texture.as_ref().unwrap();

        // The texture covers `cols × rows` cells starting from the viewport
        // top-left corner (clip.min), snapped to the cell grid so the pattern
        // is perfectly static on screen regardless of pan position.
        let grid_origin_x = (clip.min.x / cell).floor() * cell;
        let grid_origin_y = (clip.min.y / cell).floor() * cell;

        // Map the image_rect portion of the grid into UV coordinates.
        // u = (screen_x - grid_origin) / (cols * cell)  (similarly for v).
        let tex_w = cols as f32 * cell;
        let tex_h = rows as f32 * cell;
        let u_min = (rect.min.x - grid_origin_x) / tex_w;
        let v_min = (rect.min.y - grid_origin_y) / tex_h;
        let u_max = (rect.max.x - grid_origin_x) / tex_w;
        let v_max = (rect.max.y - grid_origin_y) / tex_h;
        let uv = Rect::from_min_max(Pos2::new(u_min, v_min), Pos2::new(u_max, v_max));

        painter.image(tex.id(), rect, uv, Color32::WHITE);
    }

    fn draw_fill_commit_overlays(
        &self,
        painter: &egui::Painter,
        image_rect: Rect,
        state: &mut CanvasState,
        ctx: &egui::Context,
    ) {
        let now = crate::time_compat::Instant::now();
        state.fill_commit_overlays.retain_mut(|overlay| {
            let elapsed = now.saturating_duration_since(overlay.started_at);
            if elapsed >= overlay.duration {
                return false;
            }

            let width = overlay.bounds.2.saturating_sub(overlay.bounds.0) + 1;
            let height = overlay.bounds.3.saturating_sub(overlay.bounds.1) + 1;
            if width == 0 || height == 0 || overlay.rgba.is_empty() {
                return false;
            }

            if overlay.texture.is_none() {
                let color_image = ColorImage::from_rgba_unmultiplied(
                    [width as usize, height as usize],
                    &overlay.rgba,
                );
                overlay.texture = Some(ctx.load_texture(
                    format!(
                        "fill_commit_overlay_{}_{}_{}",
                        overlay.started_at.elapsed().as_nanos(),
                        overlay.bounds.0,
                        overlay.bounds.1
                    ),
                    ImageData::Color(Arc::new(color_image)),
                    TextureOptions {
                        magnification: TextureFilter::Nearest,
                        minification: TextureFilter::Linear,
                        ..Default::default()
                    },
                ));
            }

            let fade =
                1.0 - (elapsed.as_secs_f32() / overlay.duration.as_secs_f32()).clamp(0.0, 1.0);
            let alpha = (255.0 * fade + 0.5) as u8;
            let tint = Color32::from_rgba_premultiplied(alpha, alpha, alpha, alpha);

            if let Some(tex) = &overlay.texture {
                let screen_x0 = (image_rect.min.x + overlay.bounds.0 as f32 * self.zoom).round();
                let screen_y0 = (image_rect.min.y + overlay.bounds.1 as f32 * self.zoom).round();
                let screen_x1 =
                    (image_rect.min.x + (overlay.bounds.2 + 1) as f32 * self.zoom).round();
                let screen_y1 =
                    (image_rect.min.y + (overlay.bounds.3 + 1) as f32 * self.zoom).round();
                painter.image(
                    tex.id(),
                    Rect::from_min_max(
                        Pos2::new(screen_x0, screen_y0),
                        Pos2::new(screen_x1, screen_y1),
                    ),
                    Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
                    tint,
                );
            }

            ctx.request_repaint();
            true
        });
    }

    /// Converts screen position to canvas pixel coordinates
    fn screen_to_canvas(
        &self,
        screen_pos: Pos2,
        canvas_rect: Rect,
        state: &CanvasState,
    ) -> Option<(u32, u32)> {
        let image_width = state.width as f32 * self.zoom;
        let image_height = state.height as f32 * self.zoom;

        let center_x = canvas_rect.center().x + self.pan_offset.x;
        let center_y = canvas_rect.center().y + self.pan_offset.y;

        let image_rect = Rect::from_center_size(
            Pos2::new(center_x, center_y),
            Vec2::new(image_width, image_height),
        );

        if !image_rect.contains(screen_pos) {
            return None;
        }

        // Convert to image coordinates
        let rel_x = (screen_pos.x - image_rect.min.x) / self.zoom;
        let rel_y = (screen_pos.y - image_rect.min.y) / self.zoom;

        let pixel_x = rel_x as u32;
        let pixel_y = rel_y as u32;

        if pixel_x < state.width && pixel_y < state.height {
            Some((pixel_x, pixel_y))
        } else {
            None
        }
    }

    /// Public wrapper for screen_to_canvas conversion (used by app.rs for move tools).
    pub fn screen_to_canvas_pub(
        &self,
        screen_pos: Pos2,
        canvas_rect: Rect,
        state: &CanvasState,
    ) -> Option<(u32, u32)> {
        self.screen_to_canvas(screen_pos, canvas_rect, state)
    }

    /// Public wrapper for sub-pixel screen_to_canvas (used by app.rs for paste-at-cursor).
    pub fn screen_to_canvas_f32_pub(
        &self,
        screen_pos: Pos2,
        canvas_rect: Rect,
        state: &CanvasState,
    ) -> Option<(f32, f32)> {
        self.screen_to_canvas_f32(screen_pos, canvas_rect, state)
    }

    /// Converts any screen position to canvas-space float coordinates without
    /// bounds checking.  Returns coordinates even when the pointer is outside the
    /// canvas image — values can be negative or exceed `(width, height)`.
    /// Used by selection tools and gradient handles so the user can start/drag
    /// from outside the canvas area.
    fn screen_to_canvas_unclamped(
        &self,
        screen_pos: Pos2,
        canvas_rect: Rect,
        state: &CanvasState,
    ) -> (f32, f32) {
        let image_width = state.width as f32 * self.zoom;
        let image_height = state.height as f32 * self.zoom;

        let center_x = canvas_rect.center().x + self.pan_offset.x;
        let center_y = canvas_rect.center().y + self.pan_offset.y;

        let image_rect = Rect::from_center_size(
            Pos2::new(center_x, center_y),
            Vec2::new(image_width, image_height),
        );

        let rel_x = (screen_pos.x - image_rect.min.x) / self.zoom;
        let rel_y = (screen_pos.y - image_rect.min.y) / self.zoom;
        (rel_x, rel_y)
    }

    /// Like `screen_to_canvas` but returns sub-pixel float coordinates for
    /// smooth brush strokes.
    fn screen_to_canvas_f32(
        &self,
        screen_pos: Pos2,
        canvas_rect: Rect,
        state: &CanvasState,
    ) -> Option<(f32, f32)> {
        let image_width = state.width as f32 * self.zoom;
        let image_height = state.height as f32 * self.zoom;

        let center_x = canvas_rect.center().x + self.pan_offset.x;
        let center_y = canvas_rect.center().y + self.pan_offset.y;

        let image_rect = Rect::from_center_size(
            Pos2::new(center_x, center_y),
            Vec2::new(image_width, image_height),
        );

        if !image_rect.contains(screen_pos) {
            return None;
        }

        let rel_x = (screen_pos.x - image_rect.min.x) / self.zoom;
        let rel_y = (screen_pos.y - image_rect.min.y) / self.zoom;

        if rel_x >= 0.0 && rel_x < state.width as f32 && rel_y >= 0.0 && rel_y < state.height as f32
        {
            Some((rel_x, rel_y))
        } else {
            None
        }
    }

    /// Like `screen_to_canvas_f32` but allows coordinates up to and slightly
    /// beyond canvas bounds (clamped to `[0, width]` × `[0, height]`).
    /// Used by overlay tools (mesh warp) whose control points sit on canvas edges.
    fn screen_to_canvas_f32_clamped(
        &self,
        screen_pos: Pos2,
        canvas_rect: Rect,
        state: &CanvasState,
        margin_px: f32,
    ) -> Option<(f32, f32)> {
        let image_width = state.width as f32 * self.zoom;
        let image_height = state.height as f32 * self.zoom;

        let center_x = canvas_rect.center().x + self.pan_offset.x;
        let center_y = canvas_rect.center().y + self.pan_offset.y;

        let image_rect = Rect::from_center_size(
            Pos2::new(center_x, center_y),
            Vec2::new(image_width, image_height),
        );

        // Expand the image rect by the margin (screen pixels) so clicks
        // slightly outside the canvas edge are still accepted.
        let expanded = image_rect.expand(margin_px);
        if !expanded.contains(screen_pos) {
            return None;
        }

        let rel_x = (screen_pos.x - image_rect.min.x) / self.zoom;
        let rel_y = (screen_pos.y - image_rect.min.y) / self.zoom;

        // Clamp to inclusive canvas bounds [0, width] × [0, height]
        let cx = rel_x.clamp(0.0, state.width as f32);
        let cy = rel_y.clamp(0.0, state.height as f32);
        Some((cx, cy))
    }
}
