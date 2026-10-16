impl ToolsPanel {
    fn apply_lasso_selection(canvas_state: &mut CanvasState, points: &[Pos2], mode: SelectionMode) {
        let w = canvas_state.width;
        let h = canvas_state.height;

        // Build a fresh mask from polygon
        let mut lasso_mask = image::GrayImage::new(w, h);

        // Scanline fill: for each row, find intersection x-coords with polygon edges
        let n = points.len();
        for y in 0..h {
            let yf = y as f32 + 0.5; // centre of pixel row
            let mut nodes: Vec<f32> = Vec::new();
            // Walk polygon edges (including closing edge n-1 → 0)
            for i in 0..n {
                let j = (i + 1) % n;
                let yi = points[i].y;
                let yj = points[j].y;
                // Check if this edge crosses the scanline
                if (yi < yf && yj >= yf) || (yj < yf && yi >= yf) {
                    // x-intercept
                    let t = (yf - yi) / (yj - yi);
                    let x = points[i].x + t * (points[j].x - points[i].x);
                    nodes.push(x);
                }
            }
            nodes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            // Fill between pairs of intersections
            let mut k = 0;
            while k + 1 < nodes.len() {
                let x_start = (nodes[k].max(0.0) as u32).min(w);
                let x_end = ((nodes[k + 1] + 1.0).max(0.0) as u32).min(w);
                for x in x_start..x_end {
                    lasso_mask.put_pixel(x, y, image::Luma([255u8]));
                }
                k += 2;
            }
        }

        // Merge into the existing selection mask
        match mode {
            SelectionMode::Replace => {
                canvas_state.selection_mask = Some(lasso_mask);
            }
            SelectionMode::Add => {
                if let Some(ref mut existing) = canvas_state.selection_mask {
                    for y in 0..h {
                        for x in 0..w {
                            if lasso_mask.get_pixel(x, y).0[0] > 0 {
                                existing.put_pixel(x, y, image::Luma([255u8]));
                            }
                        }
                    }
                } else {
                    canvas_state.selection_mask = Some(lasso_mask);
                }
            }
            SelectionMode::Subtract => {
                if let Some(ref mut existing) = canvas_state.selection_mask {
                    for y in 0..h {
                        for x in 0..w {
                            if lasso_mask.get_pixel(x, y).0[0] > 0 {
                                existing.put_pixel(x, y, image::Luma([0u8]));
                            }
                        }
                    }
                }
                // If no existing mask, subtracting from nothing → no-op
            }
            SelectionMode::Intersect => {
                if let Some(ref existing) = canvas_state.selection_mask {
                    let mut result = image::GrayImage::new(w, h);
                    for y in 0..h {
                        for x in 0..w {
                            let new_val = lasso_mask.get_pixel(x, y).0[0];
                            let old_val = existing.get_pixel(x, y).0[0];
                            if new_val > 0 && old_val > 0 {
                                result.put_pixel(x, y, image::Luma([new_val.min(old_val)]));
                            }
                        }
                    }
                    canvas_state.selection_mask = Some(result);
                }
                // No existing mask → intersection with nothing = empty
            }
        }

        canvas_state.invalidate_selection_overlay();
    }

    // ================================================================
    // Perspective Crop: apply perspective transform + crop
    // ================================================================
    fn apply_perspective_crop(
        canvas_state: &mut CanvasState,
        corners: &[Pos2; 4],
    ) -> Option<Box<dyn crate::components::history::Command>> {
        // corners: [TL, TR, BR, BL] in canvas coords
        // Compute bounding box of the quad to determine output size
        let min_x = corners
            .iter()
            .map(|c| c.x)
            .fold(f32::INFINITY, f32::min)
            .max(0.0);
        let min_y = corners
            .iter()
            .map(|c| c.y)
            .fold(f32::INFINITY, f32::min)
            .max(0.0);
        let max_x = corners
            .iter()
            .map(|c| c.x)
            .fold(f32::NEG_INFINITY, f32::max)
            .min(canvas_state.width as f32);
        let max_y = corners
            .iter()
            .map(|c| c.y)
            .fold(f32::NEG_INFINITY, f32::max)
            .min(canvas_state.height as f32);

        let out_w = (max_x - min_x).round() as u32;
        let out_h = (max_y - min_y).round() as u32;
        if out_w < 2 || out_h < 2 {
            return None;
        }

        // Snapshot before crop for undo (multi-layer + dimension change = full snapshot)
        let snap_before = crate::components::history::SnapshotCommand::new(
            HistoryAction::PerspectiveCrop.label("Perspective Crop".to_string()),
            canvas_state,
        );

        // Rasterize and convert text layers to raster before warping —
        // perspective warp destroys vector editability
        canvas_state.ensure_text_layers_rasterized();
        for layer in &mut canvas_state.layers {
            if layer.is_text_layer() {
                layer.content = crate::canvas::LayerContent::Raster;
            }
        }

        let src_w = canvas_state.width;
        let src_h = canvas_state.height;

        // Apply perspective transform to each layer
        for layer in &mut canvas_state.layers {
            let src = layer.pixels.clone();
            let mut new_pixels = crate::canvas::TiledImage::new(out_w, out_h);

            for oy in 0..out_h {
                let v = (oy as f32 + 0.5) / out_h as f32;
                for ox in 0..out_w {
                    let u = (ox as f32 + 0.5) / out_w as f32;
                    // Bilinear interpolation of quad corners [TL, TR, BR, BL]
                    let src_x = (1.0 - u) * (1.0 - v) * corners[0].x
                        + u * (1.0 - v) * corners[1].x
                        + u * v * corners[2].x
                        + (1.0 - u) * v * corners[3].x;
                    let src_y = (1.0 - u) * (1.0 - v) * corners[0].y
                        + u * (1.0 - v) * corners[1].y
                        + u * v * corners[2].y
                        + (1.0 - u) * v * corners[3].y;

                    let pixel = Self::bilinear_sample(&src, src_x, src_y, src_w, src_h);
                    new_pixels.put_pixel(ox, oy, pixel);
                }
            }
            layer.pixels = new_pixels;
        }

        // Update canvas dimensions
        canvas_state.width = out_w;
        canvas_state.height = out_h;

        // Clear selection (doesn't make sense after a crop)
        canvas_state.clear_selection();
        canvas_state.mark_dirty(None);

        // Return undo command
        let mut snap = snap_before;
        snap.set_after(canvas_state);
        Some(Box::new(snap))
    }

    /// Bilinear sample from a TiledImage at floating point coordinates
    fn bilinear_sample(
        img: &crate::canvas::TiledImage,
        x: f32,
        y: f32,
        w: u32,
        h: u32,
    ) -> Rgba<u8> {
        let x0 = (x.floor() as i32).max(0).min(w as i32 - 1) as u32;
        let y0 = (y.floor() as i32).max(0).min(h as i32 - 1) as u32;
        let x1 = (x0 + 1).min(w - 1);
        let y1 = (y0 + 1).min(h - 1);
        let fx = x - x.floor();
        let fy = y - y.floor();

        let p00 = img.get_pixel(x0, y0);
        let p10 = img.get_pixel(x1, y0);
        let p01 = img.get_pixel(x0, y1);
        let p11 = img.get_pixel(x1, y1);

        let lerp = |a: u8, b: u8, t: f32| -> u8 {
            (a as f32 * (1.0 - t) + b as f32 * t)
                .round()
                .clamp(0.0, 255.0) as u8
        };

        let r = lerp(
            lerp(p00.0[0], p10.0[0], fx),
            lerp(p01.0[0], p11.0[0], fx),
            fy,
        );
        let g = lerp(
            lerp(p00.0[1], p10.0[1], fx),
            lerp(p01.0[1], p11.0[1], fx),
            fy,
        );
        let b_ch = lerp(
            lerp(p00.0[2], p10.0[2], fx),
            lerp(p01.0[2], p11.0[2], fx),
            fy,
        );
        let a = lerp(
            lerp(p00.0[3], p10.0[3], fx),
            lerp(p01.0[3], p11.0[3], fx),
            fy,
        );

        Rgba([r, g, b_ch, a])
    }

    // ========================================================================
    // GRADIENT TOOL — rasterizer, preview, commit, overlay, context bar
    // ========================================================================

    /// Rasterize the gradient into the canvas preview_layer.
    /// This is the hot path — called every frame during drag.
    /// Uses rayon parallel rows + pre-computed LUT for maximum throughput.
    /// When `dragging` is true, renders at reduced resolution for responsiveness,
    /// then renders full-res on release.
    fn render_gradient_to_preview(
        &mut self,
        canvas_state: &mut CanvasState,
        gpu_renderer: Option<&mut crate::gpu::GpuRenderer>,
    ) {
        let (start, end) = match (self.gradient_state.drag_start, self.gradient_state.drag_end) {
            (Some(s), Some(e)) => (s, e),
            _ => return,
        };

        // Ensure LUT is current
        if self.gradient_state.lut_dirty {
            self.gradient_state.rebuild_lut();
        }

        let w = canvas_state.width;
        let h = canvas_state.height;

        let shape = self.gradient_state.shape;
        let repeat = self.gradient_state.repeat;
        let mode = self.gradient_state.mode;
        let is_eraser = mode == GradientMode::Transparency;

        // Selection mask for clipping (applied as CPU post-pass)
        let sel_mask = canvas_state.selection_mask.as_ref();
        let has_selection = sel_mask.is_some();

        // Pre-determine fast/slow path so we know whether to downscale.
        let active_layer_normal_pre = canvas_state
            .layers
            .get(canvas_state.active_layer_index)
            .map(|l| l.blend_mode == BlendMode::Normal && l.opacity >= 1.0)
            .unwrap_or(false);
        let has_layers_above_pre = canvas_state
            .layers
            .iter()
            .skip(canvas_state.active_layer_index + 1)
            .any(|l| l.visible);
        let can_fast_path_pre = active_layer_normal_pre && !has_layers_above_pre && !is_eraser;

        // Downscale factor: only for slow path during interactive drag at >1080p.
        // The gradient preview is generated at reduced resolution and
        // composited via composite_partial_downscaled — ~4–16× fewer pixels.
        // On release / commit the gradient re-renders at full resolution.
        // Applied to BOTH fast and slow paths: the GPU readback + premultiply +
        // texture upload cost is significant even on the fast path at 4K.
        // Threshold ~1.5M pixels so 1080p (2M) gets scale 2 → ~960×540.
        let preview_scale: u32 = if self.gradient_state.dragging {
            let total_pixels = w as u64 * h as u64;
            if total_pixels > 1_500_000 {
                ((total_pixels as f64 / 1_000_000.0).sqrt().ceil() as u32).max(2)
            } else {
                1
            }
        } else {
            1
        };
        let gen_w = if preview_scale > 1 {
            w.div_ceil(preview_scale)
        } else {
            w
        };
        let gen_h = if preview_scale > 1 {
            h.div_ceil(preview_scale)
        } else {
            h
        };

        // ------------------------------------------------------------------
        // GPU PATH
        // ------------------------------------------------------------------
        // (On web, `gpu_renderer` is already `None` here — see
        // `Canvas::gpu_renderer_for_sync_readback` — because `generate_into`
        // blocks on a GPU readback that can't complete synchronously in the
        // browser. This just takes the existing CPU fallback below.)
        let mut full_buf = if let Some(gpu) = gpu_renderer {
            let shape_u32 = match shape {
                GradientShape::Linear => 0u32,
                GradientShape::LinearReflected => 1,
                GradientShape::Radial => 2,
                GradientShape::Diamond => 3,
            };

            let params = crate::gpu::GradientGpuParams {
                start_x: start.x / preview_scale as f32,
                start_y: start.y / preview_scale as f32,
                end_x: end.x / preview_scale as f32,
                end_y: end.y / preview_scale as f32,
                width: gen_w,
                height: gen_h,
                shape: shape_u32,
                repeat: if repeat { 1 } else { 0 },
                is_eraser: if is_eraser { 1 } else { 0 },
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            };

            // Pass the raw LUT — the shader handles eraser-mode baking internally
            gpu.gradient_pipeline.generate_into(
                &gpu.ctx,
                &params,
                &self.gradient_state.lut,
                &mut self.gradient_state.gpu_readback_buf,
            );
            let mut buf = std::mem::take(&mut self.gradient_state.gpu_readback_buf);

            // Apply selection mask as CPU post-pass (parallelised with rayon)
            if has_selection && let Some(mask) = sel_mask {
                let mw = mask.width();
                let mh = mask.height();
                let row_bytes = gen_w as usize * 4;
                let ps = preview_scale;
                buf.par_chunks_mut(row_bytes)
                    .enumerate()
                    .for_each(|(y, row)| {
                        let canvas_y = (y as u32) * ps;
                        if canvas_y >= mh {
                            return;
                        }
                        for x in 0..gen_w as usize {
                            let canvas_x = (x as u32) * ps;
                            if canvas_x >= mw {
                                continue;
                            }
                            let sel_alpha = mask.get_pixel(canvas_x, canvas_y).0[0];
                            let off = x * 4;
                            if sel_alpha == 0 {
                                row[off] = 0;
                                row[off + 1] = 0;
                                row[off + 2] = 0;
                                row[off + 3] = 0;
                            } else if sel_alpha < 255 {
                                row[off + 3] =
                                    ((row[off + 3] as u16 * sel_alpha as u16) / 255) as u8;
                            }
                        }
                    });
            }

            buf
        } else {
            // ------------------------------------------------------------------
            // CPU FALLBACK PATH — with downscale during drag
            // ------------------------------------------------------------------
            let ax = start.x;
            let ay = start.y;
            let bx = end.x;
            let by = end.y;

            // Pre-bake the LUT for eraser mode on CPU
            let lut: Vec<u8> = if is_eraser {
                let src = &self.gradient_state.lut;
                let mut baked = vec![0u8; 256 * 4];
                for i in 0..256 {
                    let off = i * 4;
                    let lum = (0.299 * src[off] as f32
                        + 0.587 * src[off + 1] as f32
                        + 0.114 * src[off + 2] as f32) as u8;
                    let a = ((lum as u16 * src[off + 3] as u16) / 255) as u8;
                    baked[off] = 255;
                    baked[off + 1] = 255;
                    baked[off + 2] = 255;
                    baked[off + 3] = a;
                }
                baked
            } else {
                self.gradient_state.lut.clone()
            };

            // Pre-compute direction vectors
            let dx = bx - ax;
            let dy = by - ay;
            let len_sq = dx * dx + dy * dy;
            let len = len_sq.sqrt();
            let inv_len = if len > 1e-6 { 1.0 / len } else { 0.0 };
            let inv_len_sq = if len_sq > 1e-6 { 1.0 / len_sq } else { 0.0 };
            let ux = dx * inv_len;
            let uy = dy * inv_len;

            // Downscale during drag for responsiveness — use the
            // same preview_scale as the GPU path so buffer dimensions
            // are consistent across both paths.
            let scale: u32 = preview_scale;
            let rw = gen_w;
            let rh = gen_h;
            let scale_f = scale as f32;

            let row_stride = rw as usize * 4;
            let mut flat_buf = vec![0u8; row_stride * rh as usize];

            flat_buf
                .par_chunks_mut(row_stride)
                .enumerate()
                .for_each(|(y, row)| {
                    let py = y as f32 * scale_f + scale_f * 0.5;
                    let ry = py - ay;
                    let dot_y = ry * dy;
                    let dist_y_sq = ry * ry;
                    let proj_y_component = ry * uy;
                    let perp_y_component = ry * ux;
                    let gy = (y as u32) * scale;

                    for x in 0..rw as usize {
                        let px = x as f32 * scale_f + scale_f * 0.5;
                        let gx = (x as u32) * scale;

                        if has_selection
                            && let Some(mask) = sel_mask
                            && (gx as usize) < mask.width() as usize
                            && (gy as usize) < mask.height() as usize
                            && mask.get_pixel(gx, gy).0[0] == 0
                        {
                            continue;
                        }

                        let rx = px - ax;
                        let t = match shape {
                            GradientShape::Linear => {
                                let raw = (rx * dx + dot_y) * inv_len_sq;
                                if repeat {
                                    raw.rem_euclid(1.0)
                                } else {
                                    raw.clamp(0.0, 1.0)
                                }
                            }
                            GradientShape::LinearReflected => {
                                let raw = (rx * dx + dot_y) * inv_len_sq;
                                if repeat {
                                    let t_mod = raw.rem_euclid(2.0);
                                    if t_mod > 1.0 { 2.0 - t_mod } else { t_mod }
                                } else {
                                    1.0 - (2.0 * raw.clamp(0.0, 1.0) - 1.0).abs()
                                }
                            }
                            GradientShape::Radial => {
                                let dist_sq = rx * rx + dist_y_sq;
                                let dist = dist_sq.sqrt() * inv_len;
                                if repeat {
                                    dist.rem_euclid(1.0)
                                } else {
                                    dist.clamp(0.0, 1.0)
                                }
                            }
                            GradientShape::Diamond => {
                                let proj = (rx * ux + proj_y_component).abs() * inv_len;
                                let perp = (rx * (-uy) + perp_y_component).abs() * inv_len;
                                let dist = proj + perp;
                                if repeat {
                                    dist.rem_euclid(1.0)
                                } else {
                                    dist.clamp(0.0, 1.0)
                                }
                            }
                        };

                        let idx = (t * 255.0) as usize;
                        let loff = idx * 4;
                        let mut a = lut[loff + 3];

                        if has_selection
                            && let Some(mask) = sel_mask
                            && (gx as usize) < mask.width() as usize
                            && (gy as usize) < mask.height() as usize
                        {
                            let sel_alpha = mask.get_pixel(gx, gy).0[0];
                            if sel_alpha < 255 {
                                a = ((a as u16 * sel_alpha as u16) / 255) as u8;
                            }
                        }

                        if a > 0 {
                            let off = x * 4;
                            row[off] = lut[loff];
                            row[off + 1] = lut[loff + 1];
                            row[off + 2] = lut[loff + 2];
                            row[off + 3] = a;
                        }
                    }
                });

            // Upscale if needed — but NOT when preview_scale > 1 (slow path
            // will use composite_partial_downscaled at reduced resolution)
            if scale > 1 && preview_scale == 1 {
                let mut full = vec![0u8; w as usize * 4 * h as usize];
                let full_stride = w as usize * 4;
                for fy in 0..h as usize {
                    let sy = fy / scale as usize;
                    let src_row = &flat_buf[sy * row_stride..(sy + 1) * row_stride];
                    let dst_row = &mut full[fy * full_stride..(fy + 1) * full_stride];
                    for fx in 0..w as usize {
                        let sx = fx / scale as usize;
                        let s_off = sx * 4;
                        let d_off = fx * 4;
                        dst_row[d_off] = src_row[s_off];
                        dst_row[d_off + 1] = src_row[s_off + 1];
                        dst_row[d_off + 2] = src_row[s_off + 2];
                        dst_row[d_off + 3] = src_row[s_off + 3];
                    }
                }
                full
            } else {
                flat_buf
            }
        };

        // Determine whether we can use the fast overlay path (no full-stack
        // composite needed).  Already pre-computed above as can_fast_path_pre.
        let can_fast_path = can_fast_path_pre;

        if can_fast_path {
            // ── Fast path: skip TiledImage entirely ──────────────────
            // Pre-multiply alpha and store directly in preview_flat_buffer.
            // The display code will detect preview_flat_ready and use this
            // buffer directly, skipping extract_region + premultiply.
            let buf_len = full_buf.len();
            canvas_state.preview_flat_buffer.resize(buf_len, 0);
            // Premultiply in parallel (rayon) into the target buffer
            canvas_state
                .preview_flat_buffer
                .par_chunks_exact_mut(4)
                .zip(full_buf.par_chunks_exact(4))
                .for_each(|(dst, src)| {
                    let a = src[3] as u16;
                    dst[0] = ((src[0] as u16 * a + 128) / 255) as u8;
                    dst[1] = ((src[1] as u16 * a + 128) / 255) as u8;
                    dst[2] = ((src[2] as u16 * a + 128) / 255) as u8;
                    dst[3] = src[3];
                });
            canvas_state.preview_flat_ready = true;
            // Populate preview_layer so that commit_gradient works correctly.
            // When downscaled (preview_scale > 1), use the actual buffer
            // dimensions — commit will re-render at full res on mouse release
            // (dragging=false → preview_scale=1) before commit_gradient runs.
            let pl_w = if preview_scale > 1 { gen_w } else { w };
            let pl_h = if preview_scale > 1 { gen_h } else { h };
            canvas_state.preview_layer = Some(TiledImage::from_raw_rgba(pl_w, pl_h, &full_buf));
            canvas_state.preview_force_composite = false;
            canvas_state.preview_downscale = preview_scale;
        } else {
            // ── Slow path: need TiledImage for composite_partial ─────
            // When preview_scale > 1 the buffer is at reduced resolution;
            // composite_partial_downscaled will sample layers at matching
            // stride so the output is a smaller texture that egui stretches.
            let pw = if preview_scale > 1 { gen_w } else { w };
            let ph = if preview_scale > 1 { gen_h } else { h };
            let preview = TiledImage::from_raw_rgba(pw, ph, &full_buf);
            canvas_state.preview_layer = Some(preview);
            canvas_state.preview_force_composite = true;
            canvas_state.preview_flat_ready = false;
            canvas_state.preview_downscale = preview_scale;
        }

        // Configure preview compositing
        canvas_state.preview_blend_mode = BlendMode::Normal;
        canvas_state.preview_is_eraser = is_eraser;
        canvas_state.preview_stroke_bounds = Some(egui::Rect::from_min_max(
            egui::pos2(0.0, 0.0),
            egui::pos2(w as f32, h as f32),
        ));
        canvas_state.mark_preview_changed();
        self.gradient_state.preview_dirty = false;

        // Return the full_buf to the reusable readback buffer (keeps capacity)
        full_buf.clear();
        self.gradient_state.gpu_readback_buf = full_buf;
    }
}
