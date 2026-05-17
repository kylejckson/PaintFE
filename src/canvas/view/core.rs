impl Canvas {
    pub fn new(preferred_gpu: &str) -> Self {
        let gpu_renderer = crate::gpu::GpuRenderer::new(preferred_gpu);
        Self {
            zoom: 1.0,
            pan_offset: Vec2::ZERO,
            last_filter_was_linear: None,
            last_canvas_rect: None,
            selection_stroke: Color32::from_rgb(66, 133, 244),
            selection_fill: Color32::from_rgba_unmultiplied(66, 133, 244, 50),
            selection_contrast: Color32::WHITE,
            paste_context_action: None,
            open_paste_menu: false,
            gpu_renderer,
            frame_times: VecDeque::with_capacity(60),
            fps: 0.0,
            fill_recalc_active: false,
            tool_map_build_label: None,
            gradient_commit_active: false,
            paste_layers_above_cache: None,
            paste_layers_below_cache: None,
            paste_overwrite_preview_cache: None,
            brush_tip_cursor_tex: None,
            brush_tip_cursor_tex_inv: None,
            brush_tip_cursor_key: (String::new(), 0, 0),
            tool_cursor_icon: None,
            canvas_widget_id: None,
            checkerboard_texture: None,
            checkerboard_brightness_cached: 0.0,
            checkerboard_cached_size: (0, 0),
        }
    }

    pub fn new_without_state() -> Self {
        // Default init with high performance GPU request
        Self::new("high performance")
    }

    /// Reinitialise the GPU renderer with a different preferred adapter.
    pub fn init_gpu(&mut self, preferred_gpu: &str) {
        self.gpu_renderer = crate::gpu::GpuRenderer::new(preferred_gpu);
    }

    /// Returns `true` ÔÇö GPU rendering is always available (software fallback).
    pub fn has_gpu(&self) -> bool {
        true
    }

    /// GPU adapter name (for status bar display).
    pub fn gpu_adapter_name(&self) -> &str {
        self.gpu_renderer.adapter_name()
    }

    /// Notify the GPU renderer that a layer was deleted so its texture can
    /// be recycled and remaining layer indices are shifted down.
    pub fn gpu_remove_layer(&mut self, layer_idx: usize) {
        self.gpu_renderer.remove_layer_and_reindex(layer_idx);
    }

    /// Clear all GPU layer textures (e.g., project switch).
    pub fn gpu_clear_layers(&mut self) {
        self.gpu_renderer.clear_layers();
    }

    pub fn show(&mut self, ui: &mut egui::Ui, state: &mut CanvasState) {
        // Default colors when called without color info
        let default_primary = [0.0, 0.0, 0.0, 1.0]; // Black
        let default_secondary = [1.0, 1.0, 1.0, 1.0]; // White
        let default_bg = Color32::from_gray(250); // Light grey default
        let default_settings = crate::assets::AppSettings::default();
        let default_accent = Color32::from_rgb(66, 133, 244); // Default blue
        self.show_with_state(
            ui,
            state,
            None,
            default_primary,
            default_secondary,
            default_bg,
            None,
            false,
            &default_settings,
            0,
            0,
            default_accent,
            None,
            None,
            "",
            false,
        );
    }

    pub fn show_with_state(
        &mut self,
        ui: &mut egui::Ui,
        state: &mut CanvasState,
        tools: Option<&mut crate::components::tools::ToolsPanel>,
        primary_color_f32: [f32; 4],
        secondary_color_f32: [f32; 4],
        bg_color: Color32,
        mut paste_overlay: Option<&mut crate::ops::clipboard::PasteOverlay>,
        modal_open: bool,
        debug_settings: &crate::assets::AppSettings,
        pending_filter_jobs: usize,
        pending_io_ops: usize,
        accent_color: Color32,
        filter_ops_start_time: Option<f64>,
        io_ops_start_time: Option<f64>,
        filter_status_description: &str,
        live_window_resize: bool,
    ) {
        // FPS tracking: measure time since last frame
        let current_time = ui.input(|i| i.time);
        self.frame_times.push_back(current_time);
        // Keep last 60 frames for averaging
        while self.frame_times.len() > 60 {
            self.frame_times.pop_front();
        }
        // Calculate FPS from frame times
        if self.frame_times.len() >= 2 {
            let oldest = *self.frame_times.front().unwrap();
            let newest = *self.frame_times.back().unwrap();
            let elapsed = newest - oldest;
            if elapsed > 0.0 {
                self.fps = (self.frame_times.len() - 1) as f32 / elapsed as f32;
            }
        }

        // Ensure text layers are up-to-date before any compositing/display.
        state.ensure_text_layers_rasterized();

        let available_size = ui.available_size();

        // Allocate canvas area with focusable sense so egui claims keyboard input,
        // preventing Windows from playing error sounds for unhandled key events.
        let sense = egui::Sense::click_and_drag()
            .union(egui::Sense::hover())
            .union(egui::Sense::focusable_noninteractive());
        let (response, painter) = ui.allocate_painter(available_size, sense);
        self.canvas_widget_id = Some(response.id);
        state.canvas_widget_id = Some(response.id);
        // Keep focus on the canvas when no other UI widget is active.
        // Also always reclaim focus when the text tool is editing ÔÇö this
        // prevents DragValues in floating panels (color, etc.) from holding
        // focus and silently swallowing keystrokes meant for the text layer.
        let force_canvas_focus_for_text = tools
            .as_ref()
            .map(|t| t.text_state.is_editing && !t.text_state.font_popup_open)
            .unwrap_or(false);
        if !ui.ctx().memory(|m| m.focused().is_some())
            || response.clicked()
            || force_canvas_focus_for_text
        {
            response.request_focus();
        }
        let canvas_rect = response.rect;
        self.last_canvas_rect = Some(canvas_rect);

        // Handle panning with middle mouse button
        if response.dragged() && ui.input(|i| i.pointer.middle_down()) {
            self.pan_offset += response.drag_delta();
        }

        // Determine correct texture filter based on zoom level and user settings.
        // User can choose between Linear (smooth) and Nearest (sharp) for zoomed-out views.
        let use_linear_filter = match debug_settings.zoom_filter_mode {
            crate::assets::ZoomFilterMode::Linear => self.zoom < 2.0,
            crate::assets::ZoomFilterMode::Nearest => false,
        };
        let texture_options = if use_linear_filter {
            TextureOptions {
                magnification: TextureFilter::Linear,
                minification: TextureFilter::Linear,
                ..Default::default()
            }
        } else {
            TextureOptions {
                magnification: TextureFilter::Nearest,
                minification: TextureFilter::Nearest,
                ..Default::default()
            }
        };

        // Check if filter changed (zoom crossed the 2.0 threshold)
        let prev_filter_was_linear = self.last_filter_was_linear;
        let filter_changed = prev_filter_was_linear.is_some_and(|last| last != use_linear_filter);
        self.last_filter_was_linear = Some(use_linear_filter);

        // ---- GPU COMPOSITING (committed layers only) ----
        // Plans C+D+E+A: Optimised composite ÔåÆ readback ÔåÆ display pipeline.
        //   C: Reuse TextureHandle via tex.set() (no allocation churn)
        //   D: bytemuck cast_slice for zero-copy Color32 conversion
        //   E: Dirty-rect only GPU readback + persistent CPU pixel buffer
        //   A: Avoid recomposite when only filter mode changes (tex options only)
        {
            let gpu = &mut self.gpu_renderer;
            let pixels_dirty = state.dirty_rect.is_some() || state.composite_cache.is_none();
            let defer_live_resize_composite = live_window_resize && state.composite_cache.is_some();

            // Plan A: filter_changed (zoom crosses 2.0├ù threshold) doesn't
            // require GPU recomposite ÔÇö the pixels haven't changed, only the
            // texture sampling mode.  Re-upload from CPU buffer with new options.
            let needs_reupload_only =
                filter_changed && !pixels_dirty && !state.composite_cpu_buffer.is_empty();

            if pixels_dirty && !defer_live_resize_composite {
                // Sync each real layer to the GPU ÔÇö per-layer generation
                // tracking ensures only actually-modified layers are
                // re-uploaded, and dirty-rect partial uploads avoid copying
                // the entire image when only a small brush stroke changed.
                let dirty_rect_opt = state.dirty_rect;
                for (idx, layer) in state.layers.iter().enumerate() {
                    if !layer.visible {
                        continue;
                    }

                    let has_live_mask = layer.mask_enabled && layer.mask.is_some();

                    if gpu.layer_is_current(idx, layer.gpu_generation) {
                        continue;
                    }

                    let did_partial = if let Some(ref dr) = dirty_rect_opt {
                        let cw = layer.pixels.width();
                        let ch = layer.pixels.height();
                        let rx = (dr.min.x.max(0.0) as u32).min(cw.saturating_sub(1));
                        let ry = (dr.min.y.max(0.0) as u32).min(ch.saturating_sub(1));
                        let rx2 = (dr.max.x.ceil() as u32).min(cw);
                        let ry2 = (dr.max.y.ceil() as u32).min(ch);
                        let rw = rx2.saturating_sub(rx);
                        let rh = ry2.saturating_sub(ry);
                        let is_subregion = rw > 0 && rh > 0 && (rw < cw || rh < ch);
                        if is_subregion && gpu.layer_has_texture(idx, cw, ch) {
                            let (ax, ay, aw, ah) =
                                crate::gpu::align_dirty_rect(rx, ry, rw, rh, cw, ch);
                            if aw > 0 && ah > 0 {
                                let mut buf = std::mem::take(&mut state.region_extract_buf);
                                layer
                                    .pixels
                                    .extract_region_rgba_fast(ax, ay, aw, ah, &mut buf);

                                // For masked layers, fold conceal alpha into the extracted
                                // region so we can still do partial uploads.
                                if has_live_mask && let Some(mask) = &layer.mask {
                                    for y in 0..ah {
                                        for x in 0..aw {
                                            let conceal = mask.get_pixel(ax + x, ay + y)[3];
                                            if conceal == 0 {
                                                continue;
                                            }
                                            let off = ((y * aw + x) * 4 + 3) as usize;
                                            let a = buf[off] as u32;
                                            buf[off] = ((a * (255 - conceal as u32)) / 255) as u8;
                                        }
                                    }
                                }

                                let region = crate::gpu::renderer::DirtyRegion {
                                    x: ax,
                                    y: ay,
                                    width: aw,
                                    height: ah,
                                };
                                gpu.update_layer_rect(idx, &region, &buf);
                                state.region_extract_buf = buf;
                                gpu.set_layer_generation(idx, layer.gpu_generation);
                                true
                            } else {
                                false
                            }
                        } else {
                            false
                        }
                    } else {
                        false
                    };

                    if !did_partial {
                        let flat = layer.to_masked_rgba_image();
                        gpu.ensure_layer_texture(
                            idx,
                            layer.pixels.width(),
                            layer.pixels.height(),
                            flat.as_raw(),
                            layer.gpu_generation,
                        );
                    }
                }

                let preview_gpu_idx = usize::MAX;
                gpu.remove_layer(preview_gpu_idx);

                let mut layer_info: Vec<(usize, f32, bool, u8)> =
                    Vec::with_capacity(state.layers.len());
                for (i, l) in state.layers.iter().enumerate() {
                    layer_info.push((i, l.opacity, l.visible, l.blend_mode.to_u8()));
                }

                // B1: Async double-buffered readback.  GPU composites the full
                // canvas (fast, stays on GPU) and copies to a staging buffer.
                // We read from the PREVIOUS frame's staging buffer (non-blocking,
                // 1-frame latency ÔÇö imperceptible at 60fps).
                let have_existing_buffer = state.composite_cpu_buffer.len()
                    == (state.width as usize * state.height as usize);
                let dr_for_readback = if have_existing_buffer {
                    state.dirty_rect
                } else {
                    None // Force full readback on first frame or after resize
                };

                // Always use async readback so the UI thread never blocks on
                // GPU completion during repaint-heavy events like live window
                // resizing. While the async copy finishes we keep displaying
                // the previous composite.
                let result = gpu.composite_dirty_readback_async(
                    state.width,
                    state.height,
                    &layer_info,
                    dr_for_readback,
                );

                if let Some((pixels, rx, ry, rw, rh, is_full)) = result {
                    // Plan D: bytemuck zero-copy cast from &[u8] to &[Color32]
                    let src: &[Color32] = bytemuck::cast_slice(&pixels);
                    let cmyk = state.cmyk_preview;

                    if is_full {
                        // Full readback ÔÇö replace entire CPU buffer.
                        // Create ColorImage directly from src to avoid a
                        // redundant 33MB clone at 4K.
                        let display_pixels = if cmyk {
                            apply_cmyk_soft_proof(src)
                        } else {
                            src.to_vec()
                        };
                        let color_image = ColorImage {
                            size: [state.width as usize, state.height as usize],
                            source_size: egui::Vec2::new(state.width as f32, state.height as f32),
                            pixels: display_pixels,
                        };
                        // Update persistent CPU buffer from the same source
                        // (un-proofed ÔÇö true pixel values)
                        state.composite_cpu_buffer.clear();
                        state.composite_cpu_buffer.extend_from_slice(src);

                        let image_data = ImageData::Color(Arc::new(color_image));
                        if let Some(ref mut tex) = state.composite_cache {
                            tex.set(image_data, texture_options);
                        } else {
                            state.composite_cache = Some(ui.ctx().load_texture(
                                "canvas_composite",
                                image_data,
                                texture_options,
                            ));
                        }
                    } else {
                        // Partial readback ÔÇö patch dirty region into persistent CPU buffer
                        let canvas_w = state.width as usize;
                        let region_w = rw as usize;
                        for row in 0..rh as usize {
                            let dst_y = ry as usize + row;
                            let dst_start = dst_y * canvas_w + rx as usize;
                            let src_start = row * region_w;
                            state.composite_cpu_buffer[dst_start..dst_start + region_w]
                                .copy_from_slice(&src[src_start..src_start + region_w]);
                        }

                        // B2: Partial upload ÔÇö only send the dirty region to egui.
                        // This avoids cloning the entire 33MB CPU buffer at 4K.
                        // A brush stroke (e.g. 40├ù40px) uploads ~6KB instead of ~33MB.
                        let region_pixels = if cmyk {
                            apply_cmyk_soft_proof(src)
                        } else {
                            src.to_vec()
                        };
                        let region_image = ColorImage {
                            size: [rw as usize, rh as usize],
                            source_size: egui::Vec2::new(rw as f32, rh as f32),
                            pixels: region_pixels,
                        };
                        let region_data = ImageData::Color(Arc::new(region_image));
                        if let Some(ref mut tex) = state.composite_cache {
                            tex.set_partial(
                                [rx as usize, ry as usize],
                                region_data,
                                texture_options,
                            );
                        } else {
                            // No texture yet ÔÇö need full upload. Fall back to
                            // full buffer (shouldn't happen: partial readback
                            // requires an existing buffer).
                            let display_pixels = if cmyk {
                                apply_cmyk_soft_proof(&state.composite_cpu_buffer)
                            } else {
                                state.composite_cpu_buffer.clone()
                            };
                            let color_image = ColorImage {
                                size: [state.width as usize, state.height as usize],
                                source_size: egui::Vec2::new(
                                    state.width as f32,
                                    state.height as f32,
                                ),
                                pixels: display_pixels,
                            };
                            let image_data = ImageData::Color(Arc::new(color_image));
                            state.composite_cache = Some(ui.ctx().load_texture(
                                "canvas_composite",
                                image_data,
                                texture_options,
                            ));
                        }
                    }
                } else if gpu.async_readback.read_pending {
                    // First async frame after a dirty update: keep the last
                    // composite visible and poll again next frame.
                    ui.ctx().request_repaint();
                }
                state.dirty_rect = None;
            } else if pixels_dirty && defer_live_resize_composite {
                if let Some((pixels, rx, ry, rw, rh, is_full)) =
                    gpu.async_readback.try_read(&gpu.ctx.device)
                {
                    let src: &[Color32] = bytemuck::cast_slice(&pixels);
                    let cmyk = state.cmyk_preview;
                    if is_full {
                        let display_pixels = if cmyk {
                            apply_cmyk_soft_proof(src)
                        } else {
                            src.to_vec()
                        };
                        let color_image = ColorImage {
                            size: [state.width as usize, state.height as usize],
                            source_size: egui::Vec2::new(state.width as f32, state.height as f32),
                            pixels: display_pixels,
                        };
                        state.composite_cpu_buffer.clear();
                        state.composite_cpu_buffer.extend_from_slice(src);
                        let image_data = ImageData::Color(Arc::new(color_image));
                        if let Some(ref mut tex) = state.composite_cache {
                            tex.set(image_data, texture_options);
                        }
                    } else {
                        let canvas_w = state.width as usize;
                        let region_w = rw as usize;
                        for row in 0..rh as usize {
                            let dst_y = ry as usize + row;
                            let dst_start = dst_y * canvas_w + rx as usize;
                            let src_start = row * region_w;
                            state.composite_cpu_buffer[dst_start..dst_start + region_w]
                                .copy_from_slice(&src[src_start..src_start + region_w]);
                        }
                        let region_pixels = if cmyk {
                            apply_cmyk_soft_proof(src)
                        } else {
                            src.to_vec()
                        };
                        let region_image = ColorImage {
                            size: [rw as usize, rh as usize],
                            source_size: egui::Vec2::new(rw as f32, rh as f32),
                            pixels: region_pixels,
                        };
                        let region_data = ImageData::Color(Arc::new(region_image));
                        if let Some(ref mut tex) = state.composite_cache {
                            tex.set_partial(
                                [rx as usize, ry as usize],
                                region_data,
                                texture_options,
                            );
                        }
                    }
                }
            } else if gpu.async_readback.read_pending {
                // B1 fix: The async readback from the previous dirty frame may
                // have completed.  Poll it even though no new dirty rect exists,
                // otherwise the committed stroke won't appear until the NEXT
                // dirty frame (causing the "one-stroke-behind" bug).
                if let Some((pixels, rx, ry, rw, rh, is_full)) =
                    gpu.async_readback.try_read(&gpu.ctx.device)
                {
                    let src: &[Color32] = bytemuck::cast_slice(&pixels);
                    let cmyk = state.cmyk_preview;
                    if is_full {
                        let display_pixels = if cmyk {
                            apply_cmyk_soft_proof(src)
                        } else {
                            src.to_vec()
                        };
                        let color_image = ColorImage {
                            size: [state.width as usize, state.height as usize],
                            source_size: egui::Vec2::new(state.width as f32, state.height as f32),
                            pixels: display_pixels,
                        };
                        state.composite_cpu_buffer.clear();
                        state.composite_cpu_buffer.extend_from_slice(src);
                        let image_data = ImageData::Color(Arc::new(color_image));
                        if let Some(ref mut tex) = state.composite_cache {
                            tex.set(image_data, texture_options);
                        } else {
                            state.composite_cache = Some(ui.ctx().load_texture(
                                "canvas_composite",
                                image_data,
                                texture_options,
                            ));
                        }
                    } else {
                        let canvas_w = state.width as usize;
                        let region_w = rw as usize;
                        for row in 0..rh as usize {
                            let dst_y = ry as usize + row;
                            let dst_start = dst_y * canvas_w + rx as usize;
                            let src_start = row * region_w;
                            state.composite_cpu_buffer[dst_start..dst_start + region_w]
                                .copy_from_slice(&src[src_start..src_start + region_w]);
                        }
                        let region_pixels = if cmyk {
                            apply_cmyk_soft_proof(src)
                        } else {
                            src.to_vec()
                        };
                        let region_image = ColorImage {
                            size: [rw as usize, rh as usize],
                            source_size: egui::Vec2::new(rw as f32, rh as f32),
                            pixels: region_pixels,
                        };
                        let region_data = ImageData::Color(Arc::new(region_image));
                        if let Some(ref mut tex) = state.composite_cache {
                            tex.set_partial(
                                [rx as usize, ry as usize],
                                region_data,
                                texture_options,
                            );
                        }
                    }
                } else {
                    // Data not ready yet ÔÇö request another repaint so we poll again next frame
                    if !live_window_resize {
                        ui.ctx().request_repaint();
                    }
                }
            } else if needs_reupload_only {
                // Plan A: filter mode changed but pixels didn't ÔÇö re-upload
                // from existing CPU buffer with new texture options (no GPU work).
                let display_pixels = if state.cmyk_preview {
                    apply_cmyk_soft_proof(&state.composite_cpu_buffer)
                } else {
                    state.composite_cpu_buffer.clone()
                };
                let color_image = ColorImage {
                    size: [state.width as usize, state.height as usize],
                    source_size: egui::Vec2::new(state.width as f32, state.height as f32),
                    pixels: display_pixels,
                };
                let image_data = ImageData::Color(Arc::new(color_image));
                if let Some(ref mut tex) = state.composite_cache {
                    tex.set(image_data, texture_options);
                } else {
                    state.composite_cache = Some(ui.ctx().load_texture(
                        "canvas_composite",
                        image_data,
                        texture_options,
                    ));
                }
            }
        }

        // Calculate image dimensions with zoom
        let image_width = state.width as f32 * self.zoom;
        let image_height = state.height as f32 * self.zoom;

        // Center the image with pan offset
        let center_x = canvas_rect.center().x + self.pan_offset.x;
        let center_y = canvas_rect.center().y + self.pan_offset.y;

        let temp_rect = Rect::from_center_size(
            Pos2::new(center_x, center_y),
            Vec2::new(image_width, image_height),
        );

        // Round to pixel boundaries to prevent sub-pixel rendering gaps
        let image_rect = Rect::from_min_max(
            Pos2::new(temp_rect.min.x.round(), temp_rect.min.y.round()),
            Pos2::new(temp_rect.max.x.round(), temp_rect.max.y.round()),
        );

        // Fill background with theme color
        painter.rect_filled(canvas_rect, 0.0, bg_color);

        // Draw subtle grid texture on canvas background (Signal Grid pattern)
        // Only draw when grid cells would be visible (> 5px on screen) and grid is enabled
        if debug_settings.canvas_grid_visible {
            let grid_cell = 40.0; // matches website's .grid-bg
            if grid_cell > 5.0 {
                let base_alpha = debug_settings.canvas_grid_opacity;
                let grid_color = if bg_color.r() < 128 {
                    // Dark mode: blue-tinted gray (not pure white) for subtle contrast
                    Color32::from_rgba_unmultiplied(120, 120, 145, (6.0 * base_alpha) as u8)
                } else {
                    // Light mode: dark blue-black, visible on white bg
                    Color32::from_rgba_unmultiplied(0, 0, 20, (18.0 * base_alpha) as u8)
                };
                crate::signal_draw::draw_grid_texture(&painter, canvas_rect, grid_cell, grid_color);
            }
        }

        // Draw accent-tinted under-glow + depth shadow around the canvas image
        {
            let gi = debug_settings.glow_intensity;
            let ss = debug_settings.shadow_strength;
            let glow_alpha_outer = (1.5 * gi).min(255.0) as u8;
            let glow_alpha_inner = (3.0 * gi).min(255.0) as u8;
            if glow_alpha_outer > 0 {
                painter.rect_filled(
                    image_rect.expand(5.0),
                    4.0,
                    Color32::from_rgba_unmultiplied(
                        accent_color.r(),
                        accent_color.g(),
                        accent_color.b(),
                        glow_alpha_outer,
                    ),
                );
            }
            if glow_alpha_inner > 0 {
                painter.rect_filled(
                    image_rect.expand(2.0),
                    3.0,
                    Color32::from_rgba_unmultiplied(
                        accent_color.r(),
                        accent_color.g(),
                        accent_color.b(),
                        glow_alpha_inner,
                    ),
                );
            }
            // Dark depth layers
            for i in 0..4u32 {
                let distance = i as f32 * 2.0 + 2.0;
                let alpha = (((4 - i) * 5) as f32 * ss).min(255.0) as u8;
                if alpha > 0 {
                    painter.rect_filled(
                        image_rect.expand(distance),
                        4.0,
                        Color32::from_black_alpha(alpha),
                    );
                }
            }
        }

        // Draw checkerboard background (clipped to visible canvas area).
        // Static screen-space pattern: 1 textured quad, O(1) cost.
        self.draw_checkerboard(
            &painter,
            image_rect,
            canvas_rect,
            debug_settings.checkerboard_brightness,
            ui.ctx(),
        );

        // When in overwrite-paste preview mode we skip the GPU composite here;
        // the paste overlay section below draws below-layers + replacement + above-layers
        // directly over the checkerboard, correctly revealing transparency through
        // transparent paste areas without the original active layer contaminating the view.
        let is_overwrite_paste = paste_overlay
            .as_ref()
            .is_some_and(|o| o.overwrite_transparent_pixels);
        if !is_overwrite_paste {
            self.paint_composite_texture(&painter, image_rect, canvas_rect, state, Color32::WHITE);
        }

        // ====================================================================
        // CPU PREVIEW OVERLAY  (brush / line / eraser strokes in progress)
        // ====================================================================
        // The preview layer is composited with the full layer stack so that
        // both the tool's blend mode (preview_blend_mode) and each layer's
        // own blend mode + opacity are respected.  For
        // Normal+Normal+opacity=1.0 we use the fast raw-extraction path
        // (single-layer memcpy) since alpha-over on top of the GPU
        // composite is already correct in that case.
        if state.preview_layer.is_some() {
            // Check if zoom crossed the filtering threshold - invalidate cache to force recreation
            let current_filter_is_linear = matches!(
                debug_settings.zoom_filter_mode,
                crate::assets::ZoomFilterMode::Linear
            ) && self.zoom < 2.0;
            if prev_filter_was_linear != Some(current_filter_is_linear) {
                state.preview_texture_cache = None;
            }

            let needs_upload =
                state.preview_texture_cache.is_none() || state.preview_dirty_rect.is_some();

            if needs_upload {
                // Determine the stroke's accumulated bounding box.
                let sb = state.preview_stroke_bounds.unwrap_or_else(|| {
                    egui::Rect::from_min_max(
                        egui::pos2(0.0, 0.0),
                        egui::pos2(state.width as f32, state.height as f32),
                    )
                });

                // Zoom-adaptive filtering based on user preference.
                // Linear (smooth) or Nearest (sharp) for zoomed-out pixel art.
                let preview_filter = match debug_settings.zoom_filter_mode {
                    crate::assets::ZoomFilterMode::Linear if self.zoom < 2.0 => {
                        TextureFilter::Linear
                    }
                    _ => TextureFilter::Nearest,
                };
                let tex_options = TextureOptions {
                    magnification: preview_filter,
                    minification: preview_filter,
                    ..Default::default()
                };

                // Decide whether the fast raw-extraction path is valid.
                // It is correct when both blend modes are Normal and
                // the active layer is fully opaque, because plain
                // alpha-over in the egui painter is equivalent.
                // Force the blend-aware path when preview_force_composite is set
                // (e.g. fill tool with semi-transparent colours needs proper
                // layer-stack compositing to preview accurately).
                // Also force blend-aware path when there are visible layers above
                // the active layer, so they get composited correctly.
                let active_layer_normal = state
                    .layers
                    .get(state.active_layer_index)
                    .map(|l| l.blend_mode == BlendMode::Normal && l.opacity >= 1.0)
                    .unwrap_or(false);
                let active_layer_has_live_mask = state
                    .layers
                    .get(state.active_layer_index)
                    .is_some_and(|l| l.mask_enabled && l.has_live_mask());
                let has_layers_above = state
                    .layers
                    .iter()
                    .skip(state.active_layer_index + 1)
                    .any(|l| l.visible);
                let use_fast_path = state.preview_blend_mode == BlendMode::Normal
                    && active_layer_normal
                    && !active_layer_has_live_mask
                    && !state.preview_targets_mask
                    && !state.preview_force_composite
                    && !has_layers_above
                    && !state.preview_is_eraser;

                if use_fast_path {
                    if state.preview_flat_ready {
                        // -- Ultra-fast path: buffer already premultiplied --
                        // Transmute &[u8] ÔåÆ &[Color32] (both are 4 bytes, same
                        // layout) to avoid the copy that from_rgba_premultiplied
                        // would do.  bytemuck guarantees safety.
                        // When preview_downscale > 1 the buffer is at reduced
                        // resolution ÔÇö use scaled dimensions so egui stretches.
                        let pixels: &[Color32] = bytemuck::cast_slice(&state.preview_flat_buffer);
                        let ds = state.preview_downscale;
                        let pw = if ds > 1 {
                            state.width.div_ceil(ds) as usize
                        } else {
                            state.width as usize
                        };
                        let ph = if ds > 1 {
                            state.height.div_ceil(ds) as usize
                        } else {
                            state.height as usize
                        };
                        let color_image = ColorImage {
                            size: [pw, ph],
                            source_size: egui::Vec2::new(pw as f32, ph as f32),
                            pixels: pixels.to_vec(),
                        };
                        let image_data = ImageData::Color(Arc::new(color_image));
                        if let Some(ref mut tex) = state.preview_texture_cache {
                            tex.set(image_data, tex_options);
                        } else {
                            state.preview_texture_cache = Some(ui.ctx().load_texture(
                                "preview_overlay",
                                image_data,
                                tex_options,
                            ));
                        }
                    } else {
                        // -- Incremental fast path: only re-extract dirty rect --
                        // The brush uses max-alpha stamping, so previously
                        // written pixels never decrease ÔÇö their premultiplied
                        // values in the cache stay valid.  Each frame we only
                        // re-extract the small dirty region (brush-sized) and
                        // blit it into the persistent cache, reducing per-frame
                        // work from O(stroke_bounds) to O(brush_size).
                        let preview = state.preview_layer.as_ref().unwrap();
                        let rx = (sb.min.x.max(0.0) as u32).min(preview.width().saturating_sub(1));
                        let ry = (sb.min.y.max(0.0) as u32).min(preview.height().saturating_sub(1));
                        let rx2 = (sb.max.x.ceil() as u32).min(preview.width());
                        let ry2 = (sb.max.y.ceil() as u32).min(preview.height());
                        let rw = rx2.saturating_sub(rx).max(1);
                        let rh = ry2.saturating_sub(ry).max(1);

                        let old_cache = state.preview_cache_rect;
                        let bounds_match = old_cache.is_some_and(|(ox, oy, ow, oh)| {
                            ox == rx && oy == ry && ow == rw && oh == rh
                        });

                        if bounds_match && state.preview_dirty_rect.is_some() {
                            // -- Incremental update: extract only the dirty rect --
                            let dr = state.preview_dirty_rect.unwrap();
                            let dx = (dr.min.x.max(0.0).floor() as u32).max(rx);
                            let dy = (dr.min.y.max(0.0).floor() as u32).max(ry);
                            let dx2 = (dr.max.x.ceil() as u32).min(rx + rw);
                            let dy2 = (dr.max.y.ceil() as u32).min(ry + rh);
                            let dw = dx2.saturating_sub(dx);
                            let dh = dy2.saturating_sub(dy);

                            if dw > 0 && dh > 0 {
                                preview.extract_region_rgba_fast(
                                    dx,
                                    dy,
                                    dw,
                                    dh,
                                    &mut state.preview_flat_buffer,
                                );

                                // Premultiply the small dirty region in-place
                                for px in state.preview_flat_buffer.chunks_exact_mut(4) {
                                    let a = px[3] as u16;
                                    px[0] = ((px[0] as u16 * a + 128) / 255) as u8;
                                    px[1] = ((px[1] as u16 * a + 128) / 255) as u8;
                                    px[2] = ((px[2] as u16 * a + 128) / 255) as u8;
                                }

                                // Blit dirty pixels into persistent cache
                                let dirty_pixels: &[Color32] =
                                    bytemuck::cast_slice(&state.preview_flat_buffer);
                                let cache_w = rw as usize;
                                let off_x = (dx - rx) as usize;
                                let off_y = (dy - ry) as usize;
                                for row in 0..dh as usize {
                                    let src_start = row * dw as usize;
                                    let dst_start = (off_y + row) * cache_w + off_x;
                                    state.preview_premul_cache[dst_start..dst_start + dw as usize]
                                        .copy_from_slice(
                                            &dirty_pixels[src_start..src_start + dw as usize],
                                        );
                                }

                                // B2: Partial upload ÔÇö only send the dirty rect to egui.
                                // During brush strokes this uploads ~brush_size┬▓ pixels
                                // instead of cloning the full stroke_bounds cache.
                                let region_pixels: Vec<Color32> = dirty_pixels.to_vec();
                                let region_image = ColorImage {
                                    size: [dw as usize, dh as usize],
                                    source_size: egui::Vec2::new(dw as f32, dh as f32),
                                    pixels: region_pixels,
                                };
                                let region_data = ImageData::Color(Arc::new(region_image));
                                if let Some(ref mut tex) = state.preview_texture_cache {
                                    tex.set_partial([off_x, off_y], region_data, tex_options);
                                } else {
                                    // No texture yet ÔÇö fall back to full upload
                                    let color_image = ColorImage {
                                        size: [rw as usize, rh as usize],
                                        source_size: egui::Vec2::new(rw as f32, rh as f32),
                                        pixels: state.preview_premul_cache.clone(),
                                    };
                                    let image_data = ImageData::Color(Arc::new(color_image));
                                    state.preview_texture_cache = Some(ui.ctx().load_texture(
                                        "preview_overlay",
                                        image_data,
                                        tex_options,
                                    ));
                                }
                            }
                        } else if old_cache.is_some() && !bounds_match {
                            // -- Bounds changed: resize cache, preserve overlapping data --
                            let (ox, oy, ow, oh) = old_cache.unwrap();
                            let new_size = (rw as usize) * (rh as usize);
                            let mut new_cache = vec![Color32::TRANSPARENT; new_size];

                            // Compute the intersection of old and new bounds
                            // to safely copy only the overlapping region.
                            // Bounds can shrink or shift (e.g. text alignment
                            // change), not just grow.
                            let inter_x0 = ox.max(rx);
                            let inter_y0 = oy.max(ry);
                            let inter_x1 = (ox + ow).min(rx + rw);
                            let inter_y1 = (oy + oh).min(ry + rh);
                            let new_w = rw as usize;
                            let old_w = ow as usize;
                            if inter_x1 > inter_x0 && inter_y1 > inter_y0 {
                                let copy_w = (inter_x1 - inter_x0) as usize;
                                let copy_h = (inter_y1 - inter_y0) as usize;
                                let src_off_x = (inter_x0 - ox) as usize;
                                let src_off_y = (inter_y0 - oy) as usize;
                                let dst_off_x = (inter_x0 - rx) as usize;
                                let dst_off_y = (inter_y0 - ry) as usize;
                                for row in 0..copy_h {
                                    let src_start = (src_off_y + row) * old_w + src_off_x;
                                    let dst_start = (dst_off_y + row) * new_w + dst_off_x;
                                    new_cache[dst_start..dst_start + copy_w].copy_from_slice(
                                        &state.preview_premul_cache[src_start..src_start + copy_w],
                                    );
                                }
                            }

                            // Extract and premultiply only the dirty rect
                            if let Some(dr) = state.preview_dirty_rect {
                                let dx = (dr.min.x.max(0.0).floor() as u32).max(rx);
                                let dy = (dr.min.y.max(0.0).floor() as u32).max(ry);
                                let dx2 = (dr.max.x.ceil() as u32).min(rx + rw);
                                let dy2 = (dr.max.y.ceil() as u32).min(ry + rh);
                                let dw = dx2.saturating_sub(dx);
                                let dh = dy2.saturating_sub(dy);

                                if dw > 0 && dh > 0 {
                                    preview.extract_region_rgba_fast(
                                        dx,
                                        dy,
                                        dw,
                                        dh,
                                        &mut state.preview_flat_buffer,
                                    );
                                    for px in state.preview_flat_buffer.chunks_exact_mut(4) {
                                        let a = px[3] as u16;
                                        px[0] = ((px[0] as u16 * a + 128) / 255) as u8;
                                        px[1] = ((px[1] as u16 * a + 128) / 255) as u8;
                                        px[2] = ((px[2] as u16 * a + 128) / 255) as u8;
                                    }
                                    let dirty_pixels: &[Color32] =
                                        bytemuck::cast_slice(&state.preview_flat_buffer);
                                    let off_x = (dx - rx) as usize;
                                    let off_y = (dy - ry) as usize;
                                    for row in 0..dh as usize {
                                        let src_start = row * dw as usize;
                                        let dst_start = (off_y + row) * new_w + off_x;
                                        new_cache[dst_start..dst_start + dw as usize]
                                            .copy_from_slice(
                                                &dirty_pixels[src_start..src_start + dw as usize],
                                            );
                                    }
                                }
                            }

                            state.preview_premul_cache = new_cache;
                            state.preview_cache_rect = Some((rx, ry, rw, rh));

                            // Bounds changed ÔÇö must do full texture upload
                            let color_image = ColorImage {
                                size: [rw as usize, rh as usize],
                                source_size: egui::Vec2::new(rw as f32, rh as f32),
                                pixels: state.preview_premul_cache.clone(),
                            };
                            let image_data = ImageData::Color(Arc::new(color_image));
                            if let Some(ref mut tex) = state.preview_texture_cache {
                                tex.set(image_data, tex_options);
                            } else {
                                state.preview_texture_cache = Some(ui.ctx().load_texture(
                                    "preview_overlay",
                                    image_data,
                                    tex_options,
                                ));
                            }
                        } else {
                            // -- First frame or full rebuild --
                            preview.extract_region_rgba_fast(
                                rx,
                                ry,
                                rw,
                                rh,
                                &mut state.preview_flat_buffer,
                            );

                            for px in state.preview_flat_buffer.chunks_exact_mut(4) {
                                let a = px[3] as u16;
                                px[0] = ((px[0] as u16 * a + 128) / 255) as u8;
                                px[1] = ((px[1] as u16 * a + 128) / 255) as u8;
                                px[2] = ((px[2] as u16 * a + 128) / 255) as u8;
                            }

                            let pixels: &[Color32] =
                                bytemuck::cast_slice(&state.preview_flat_buffer);
                            state.preview_premul_cache = pixels.to_vec();
                            state.preview_cache_rect = Some((rx, ry, rw, rh));

                            // First frame ÔÇö must do full texture upload
                            let color_image = ColorImage {
                                size: [rw as usize, rh as usize],
                                source_size: egui::Vec2::new(rw as f32, rh as f32),
                                pixels: state.preview_premul_cache.clone(),
                            };
                            let image_data = ImageData::Color(Arc::new(color_image));
                            if let Some(ref mut tex) = state.preview_texture_cache {
                                tex.set(image_data, tex_options);
                            } else {
                                state.preview_texture_cache = Some(ui.ctx().load_texture(
                                    "preview_overlay",
                                    image_data,
                                    tex_options,
                                ));
                            }
                        }
                    }
                } else {
                    // -- Blend-aware path: full-stack composite --
                    // composite_partial blends the preview into the
                    // active layer (via preview_blend_mode) and then
                    // composites every layer with its own blend mode
                    // and opacity, giving a pixel-accurate preview.
                    // When preview_downscale > 1, sample at reduced rate
                    // for interactive responsiveness (egui stretches the
                    // smaller texture to fill the canvas rect).
                    let ds = state.preview_downscale;
                    let (color_image, _offset) = if ds > 1 {
                        state.composite_partial_downscaled(sb, ds)
                    } else {
                        state.composite_partial(sb)
                    };

                    let image_data = ImageData::Color(Arc::new(color_image));
                    if let Some(ref mut tex) = state.preview_texture_cache {
                        tex.set(image_data, tex_options);
                    } else {
                        state.preview_texture_cache = Some(ui.ctx().load_texture(
                            "preview_overlay",
                            image_data,
                            tex_options,
                        ));
                    }
                }

                state.preview_dirty_rect = None;
            }

            self.paint_preview_texture(
                &painter,
                image_rect,
                canvas_rect,
                state,
                Color32::WHITE,
                true,
            );
        } else {
            // No preview layer ÔÇö drop cached texture and buffer.
            state.preview_texture_cache = None;
            // Don't clear preview_flat_buffer here to avoid dealloc; it will
            // be reused on the next stroke.
        }

        if state.show_wrap_preview {
            self.draw_wrap_preview(&painter, image_rect, canvas_rect, state);
        }

        // Draw pixel grid overlay when zoomed in
        if state.show_pixel_grid && self.zoom >= 8.0 {
            self.draw_pixel_grid(&painter, image_rect, state, canvas_rect, debug_settings);
        }

        // Draw mirror axis overlay
        if state.mirror_mode.is_active() {
            self.draw_mirror_overlay(&painter, image_rect, state, canvas_rect);
        }

        // ====================================================================
        // SELECTION OVERLAY  (above layers, below tool cursor)
        // ====================================================================
        // Animated time value for marching ants.
        let anim_time = ui.input(|i| i.time);

        // Keep repainting so marching ants stay animated even when mouse is idle.
        let has_selection_mask = state.selection_mask.is_some();
        let has_selection_drag = tools.as_ref().is_some_and(|t| t.selection_state.dragging);
        let selection_needs_animation = has_selection_mask
            && (state.selection_overlay_built_generation != state.selection_overlay_generation
                || selection_overlay_should_animate(state.selection_overlay_bounds));
        if selection_needs_animation || has_selection_drag {
            ui.ctx().request_repaint();
        }

        // 1. Draw the committed selection mask (if any).
        //    Temporarily take the mask to avoid borrow conflict with `&mut state`
        //    (needed for the overlay texture cache).  Put it back afterwards.
        if let Some(mask) = state.selection_mask.take() {
            // Switch to "tool-active" visual mode when a drawing tool is
            // selected (not a selection tool).  This hides the hatch fill so
            // the user can see what they're painting.
            let tool_active = tools.as_ref().is_some_and(|t| {
                matches!(
                    t.active_tool,
                    crate::components::tools::Tool::Brush
                        | crate::components::tools::Tool::Eraser
                        | crate::components::tools::Tool::Pencil
                        | crate::components::tools::Tool::Line
                )
            });
            self.draw_selection_overlay(
                &painter,
                image_rect,
                &mask,
                anim_time,
                tool_active,
                state,
                ui.ctx(),
            );
            state.selection_mask = Some(mask); // put it back
        }

        if tools
            .as_ref()
            .is_some_and(|t| t.active_tool == crate::components::tools::Tool::MoveSelection)
            && let Some(mask) = state.selection_mask.as_ref()
            && let Some((x0, y0, x1, y1)) = selection_mask_bounds(mask)
        {
            let min = Pos2::new(
                (image_rect.min.x + x0 as f32 * self.zoom).round(),
                (image_rect.min.y + y0 as f32 * self.zoom).round(),
            );
            let max = Pos2::new(
                (image_rect.min.x + x1 as f32 * self.zoom).round(),
                (image_rect.min.y + y1 as f32 * self.zoom).round(),
            );
            let rect = Rect::from_min_max(min, max);
            painter.rect_stroke(
                rect,
                egui::CornerRadius::ZERO,
                egui::Stroke::new(1.0, self.selection_stroke),
                egui::StrokeKind::Middle,
            );
            let handle = (8.0_f32).max((self.zoom * 0.6).min(12.0));
            let points = [
                rect.left_top(),
                Pos2::new(rect.center().x, rect.top()),
                rect.right_top(),
                Pos2::new(rect.right(), rect.center().y),
                rect.right_bottom(),
                Pos2::new(rect.center().x, rect.bottom()),
                rect.left_bottom(),
                Pos2::new(rect.left(), rect.center().y),
            ];
            for p in points {
                let hr = Rect::from_center_size(p, Vec2::splat(handle));
                painter.rect_filled(hr, 2.0, self.selection_stroke);
                painter.rect_stroke(
                    hr,
                    2.0,
                    egui::Stroke::new(1.0, egui::Color32::WHITE),
                    egui::StrokeKind::Middle,
                );
            }
        }

        if !state.fill_commit_overlays.is_empty() {
            self.draw_fill_commit_overlays(&painter, image_rect, state, ui.ctx());
        }

        // 2. Draw the in-progress drag preview (while dragging a selection tool).
        // We peek at the tools reference to see if a selection drag is active.
        {
            // We need to check tools *before* we pass ownership via Option,
            // so we borrow it here (tools is &mut Option).
            let sel_preview = tools.as_ref().and_then(|t| {
                if !t.selection_state.dragging {
                    return None;
                }
                let start = t.selection_state.drag_start?;
                let end = t.selection_state.drag_end?;
                Some((t.active_tool, start, end))
            });

            if let Some((tool, start, end)) = sel_preview {
                let min_x = start.x.min(end.x);
                let min_y = start.y.min(end.y);
                let max_x = start.x.max(end.x);
                let max_y = start.y.max(end.y);

                // Snap to pixel grid: the selection of pixel (x,y) covers the
                // screen region [x*zoom, (x+1)*zoom), so floor the min and
                // ceil(max+1) to ensure the overlay aligns with pixel edges.
                let screen_min = Pos2::new(
                    (image_rect.min.x + min_x.floor() * self.zoom).round(),
                    (image_rect.min.y + min_y.floor() * self.zoom).round(),
                );
                let screen_max = Pos2::new(
                    (image_rect.min.x + (max_x.floor() + 1.0) * self.zoom).round(),
                    (image_rect.min.y + (max_y.floor() + 1.0) * self.zoom).round(),
                );
                let sel_rect = Rect::from_min_max(screen_min, screen_max);

                // Faint fill (derived from theme accent)
                let fill_color = self.selection_fill;

                match tool {
                    crate::components::tools::Tool::EllipseSelect => {
                        // Draw filled ellipse approximation with marching ants
                        self.draw_ellipse_overlay(&painter, sel_rect, fill_color, anim_time);
                    }
                    _ => {
                        painter.rect_filled(sel_rect, 0.0, fill_color);
                        // Marching ants border
                        self.draw_marching_rect(&painter, sel_rect, anim_time);
                    }
                }
            }
        }

        // ====================================================================
        // PASTE OVERLAY  (above selection, interactive handles)
        // ====================================================================
        let mut paste_consumed_input = false;
        let mut paste_context_result: Option<PasteAction> = None;
        if let Some(ref mut overlay) = paste_overlay {
            let is_dark = ui.visuals().dark_mode;
            let accent = self.selection_stroke; // theme accent colour
            let clipped_painter = painter.with_clip_rect(image_rect);
            if overlay.overwrite_transparent_pixels {
                if let Some(below_image) = state.composite_layers_below_active() {
                    let tex = if let Some(ref mut existing) = self.paste_layers_below_cache {
                        existing.set(
                            below_image,
                            TextureOptions {
                                magnification: TextureFilter::Nearest,
                                minification: TextureFilter::Nearest,
                                ..Default::default()
                            },
                        );
                        existing.id()
                    } else {
                        let handle = ui.ctx().load_texture(
                            "paste_layers_below",
                            below_image,
                            TextureOptions {
                                magnification: TextureFilter::Nearest,
                                minification: TextureFilter::Nearest,
                                ..Default::default()
                            },
                        );
                        let id = handle.id();
                        self.paste_layers_below_cache = Some(handle);
                        id
                    };
                    let uv = Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0));
                    clipped_painter.image(tex, image_rect, uv, Color32::WHITE);
                } else {
                    self.paste_layers_below_cache = None;
                }

                if let Some(active_layer) = state.layers.get(state.active_layer_index) {
                    let base = active_layer.pixels.to_rgba_image();
                    let preview = overlay.render_replacement_preview(&base);
                    let color_image = egui::ColorImage::from_rgba_unmultiplied(
                        [state.width as usize, state.height as usize],
                        preview.as_raw(),
                    );
                    let tex = if let Some(ref mut existing) = self.paste_overwrite_preview_cache {
                        existing.set(
                            color_image,
                            TextureOptions {
                                magnification: TextureFilter::Nearest,
                                minification: TextureFilter::Nearest,
                                ..Default::default()
                            },
                        );
                        existing.id()
                    } else {
                        let handle = ui.ctx().load_texture(
                            "paste_overwrite_preview",
                            color_image,
                            TextureOptions {
                                magnification: TextureFilter::Nearest,
                                minification: TextureFilter::Nearest,
                                ..Default::default()
                            },
                        );
                        let id = handle.id();
                        self.paste_overwrite_preview_cache = Some(handle);
                        id
                    };
                    let uv = Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0));
                    clipped_painter.image(tex, image_rect, uv, Color32::WHITE);
                }
            } else {
                self.paste_layers_below_cache = None;
                self.paste_overwrite_preview_cache = None;
                // Upload/re-upload the source texture when scale changes (once).
                // The GPU handles rotation + translation via a textured mesh.
                // Pass interaction state so large images use a lower-res preview
                // during drag to keep the UI responsive.
                let is_interacting = overlay.active_handle.is_some();
                overlay.ensure_gpu_texture(ui.ctx(), is_interacting);

                // Draw the transformed paste image via GPU mesh, clipped to canvas bounds
                // so pasted images larger than the canvas don't extend beyond it.
                overlay.draw_gpu(&clipped_painter, image_rect, self.zoom);
            }

            // If there are visible layers above the active layer, composite them
            // and overlay on top of the paste so those layers render correctly.
            let has_layers_above = state
                .layers
                .iter()
                .skip(state.active_layer_index + 1)
                .any(|l| l.visible);
            if has_layers_above {
                if let Some(above_image) = state.composite_layers_above() {
                    let tex = if let Some(ref mut existing) = self.paste_layers_above_cache {
                        existing.set(
                            above_image,
                            TextureOptions {
                                magnification: TextureFilter::Nearest,
                                minification: TextureFilter::Nearest,
                                ..Default::default()
                            },
                        );
                        existing.id()
                    } else {
                        let handle = ui.ctx().load_texture(
                            "paste_layers_above",
                            above_image,
                            TextureOptions {
                                magnification: TextureFilter::Nearest,
                                minification: TextureFilter::Nearest,
                                ..Default::default()
                            },
                        );
                        let id = handle.id();
                        self.paste_layers_above_cache = Some(handle);
                        id
                    };
                    let uv = Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0));
                    clipped_painter.image(tex, image_rect, uv, Color32::WHITE);
                }
            } else {
                // No layers above ÔÇö drop cache
                self.paste_layers_above_cache = None;
            }

            // Clear CPU preview layer so the composite doesn't double-draw.
            if state.preview_layer.is_some() {
                let tool_using = tools
                    .as_ref()
                    .is_some_and(|t| {
                        t.stroke_tracker.uses_preview_layer || t.should_keep_fill_preview_overlay()
                    });
                if !tool_using {
                    state.clear_preview_state();
                    state.mark_dirty(None);
                }
            }
            ui.ctx().request_repaint();

            // Draw handles/border on top of the composite.
            if state.show_pixel_grid && self.zoom >= 8.0 {
                self.draw_pixel_grid(&painter, image_rect, state, canvas_rect, debug_settings);
            }
            overlay.draw(&painter, image_rect, self.zoom, is_dark, accent);
            if state.show_pixel_grid
                && self.zoom >= 8.0
                && let Some((x0, y0, x1, y1)) = overlay.transformed_bounds(state.width, state.height)
            {
                let rect = Rect::from_min_max(
                    Pos2::new(image_rect.min.x + x0 as f32 * self.zoom, image_rect.min.y + y0 as f32 * self.zoom),
                    Pos2::new(image_rect.min.x + x1 as f32 * self.zoom, image_rect.min.y + y1 as f32 * self.zoom),
                );
                let grid_color = debug_settings.pixel_grid_outline_color;
                let stroke = egui::Stroke::new(1.0, grid_color);
                let mut x = rect.min.x;
                while x <= rect.max.x {
                    painter.line_segment([Pos2::new(x, rect.min.y), Pos2::new(x, rect.max.y)], stroke);
                    x += self.zoom;
                }
                let mut y = rect.min.y;
                while y <= rect.max.y {
                    painter.line_segment([Pos2::new(rect.min.x, y), Pos2::new(rect.max.x, y)], stroke);
                    y += self.zoom;
                }
            }

            // Handle interaction (move, resize, rotate).
            paste_consumed_input = overlay.handle_input(ui, image_rect, self.zoom);

            // Auto-open context menu when requested (e.g. right after paste).
            if self.open_paste_menu {
                let ctx_id = response.id.with("__egui_context_menu");
                egui::Popup::open_id(ui.ctx(), ctx_id);
                self.open_paste_menu = false;
            }

            // Right-click context menu on the paste overlay.
            response.context_menu(|ui| {
                // Interpolation filter selector.
                ui.label("Filter:");
                for interp in crate::ops::transform::Interpolation::all() {
                    if ui
                        .selectable_label(overlay.interpolation == *interp, interp.label())
                        .clicked()
                    {
                        overlay.interpolation = *interp;
                        ui.close();
                    }
                }
                ui.separator();
                if ui.button("Reset Transform").clicked() {
                    overlay.rotation = 0.0;
                    overlay.scale_x = 1.0;
                    overlay.scale_y = 1.0;
                    overlay.anchor_offset = Vec2::ZERO;
                    ui.close();
                }
                if ui.button("Center Anchor").clicked() {
                    overlay.anchor_offset = Vec2::ZERO;
                    ui.close();
                }
                ui.separator();
                if ui.button("Commit (Enter)").clicked() {
                    paste_context_result = Some(PasteAction::Commit);
                    ui.close();
                }
                if ui.button("Commit & Select").clicked() {
                    paste_context_result = Some(PasteAction::CommitAndSelect);
                    ui.close();
                }
                if ui.button("Cancel (Esc)").clicked() {
                    paste_context_result = Some(PasteAction::Cancel);
                    ui.close();
                }
            });
        } else {
            // No paste overlay ÔÇö clear layers-above cache.
            self.paste_layers_above_cache = None;
            self.paste_layers_below_cache = None;
            self.paste_overwrite_preview_cache = None;
            // Close any orphaned paste context-menu popup.
            // If commit/cancel happened while the context menu was still open,
            // the popup ID remains registered in egui memory and permanently
            // makes `any_popup_open()` return true, blocking all tool input.
            let ctx_id = response.id.with("__egui_context_menu");
            if egui::Popup::is_id_open(ui.ctx(), ctx_id) {
                egui::Popup::toggle_id(ui.ctx(), ctx_id);
            }
            // Clear preview layer if no paste overlay is active.
            if state.preview_layer.is_some() {
                // Only clear if tools aren't using it.
                // Check if a tool stroke is active ÔÇö if so, leave preview_layer alone.
                let tool_using_preview = tools
                    .as_ref()
                    .is_some_and(|t| {
                        t.stroke_tracker.uses_preview_layer || t.should_keep_fill_preview_overlay()
                    });
                if !tool_using_preview {
                    state.clear_preview_state();
                }
            }
        }
        self.paste_context_action = paste_context_result;

        // Draw center/thirds guidelines above pasted overlays for easier
        // placement while transforming clipboard content.
        if state.show_guidelines {
            self.draw_guidelines(&painter, image_rect, state, canvas_rect);
        }

        // ====================================================================
        // EXTRACT DEBUG INFO before tools is consumed
        // ====================================================================
        // Selection drag info (for debug display).
        let sel_drag_info: Option<(f32, f32)> = tools.as_ref().and_then(|t| {
            if !t.selection_state.dragging {
                return None;
            }
            let s = t.selection_state.drag_start?;
            let e = t.selection_state.drag_end?;
            let w = (e.x - s.x).abs();
            let h = (e.y - s.y).abs();
            Some((w, h))
        });

        // Paste overlay info (for debug display).
        let paste_info: Option<(f32, f32, f32, f32, f32, f32)> = paste_overlay.as_ref().map(|o| {
            let sw = o.source.width() as f32 * o.scale_x;
            let sh = o.source.height() as f32 * o.scale_y;
            (
                o.center.x,
                o.center.y,
                sw,
                sh,
                o.rotation.to_degrees(),
                o.scale_x * 100.0,
            )
        });

        // Tool info data (for bottom-left info overlay).
        let tool_info_data: Option<String> = tools.as_ref().and_then(|t| {
            match t.active_tool {
                crate::components::tools::Tool::Line => {
                    let lt = &t.line_state.line_tool;
                    if lt.stage == crate::components::tools::LineStage::Dragging
                        || lt.stage == crate::components::tools::LineStage::Editing
                    {
                        let p0 = lt.control_points[0];
                        let p3 = lt.control_points[3];
                        let dx = p3.x - p0.x;
                        let dy = p3.y - p0.y;
                        let length = (dx * dx + dy * dy).sqrt();
                        Some(format!(
                            "Line: {:.1}px | Pos: ({:.0}, {:.0}) \u{2192} ({:.0}, {:.0})",
                            length, p0.x, p0.y, p3.x, p3.y
                        ))
                    } else {
                        None
                    }
                }
                crate::components::tools::Tool::RectangleSelect
                | crate::components::tools::Tool::EllipseSelect => {
                    if t.selection_state.dragging {
                        if let (Some(s), Some(e)) =
                            (t.selection_state.drag_start, t.selection_state.drag_end)
                        {
                            let w = (e.x - s.x).abs();
                            let h = (e.y - s.y).abs();
                            let cx = (s.x + e.x) / 2.0;
                            let cy = (s.y + e.y) / 2.0;
                            let label = if t.active_tool
                                == crate::components::tools::Tool::RectangleSelect
                            {
                                "Rect"
                            } else {
                                "Ellipse"
                            };
                            Some(format!(
                                "{}: {:.0}x{:.0} | Center: ({:.0}, {:.0})",
                                label, w, h, cx, cy
                            ))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                }
                crate::components::tools::Tool::Shapes => {
                    let ss = &t.shapes_state;
                    if ss.is_drawing {
                        if let (Some(start), Some(end)) = (ss.draw_start, ss.draw_end) {
                            let w = (end[0] - start[0]).abs();
                            let h = (end[1] - start[1]).abs();
                            let cx = (start[0] + end[0]) / 2.0;
                            let cy = (start[1] + end[1]) / 2.0;
                            Some(format!(
                                "Shape: {:.0}x{:.0} | Center: ({:.0}, {:.0})",
                                w, h, cx, cy
                            ))
                        } else {
                            None
                        }
                    } else if let Some(ref placed) = ss.placed {
                        let w = placed.hw * 2.0;
                        let h = placed.hh * 2.0;
                        let mut s = format!(
                            "Shape: {:.0}x{:.0} | Center: ({:.0}, {:.0})",
                            w, h, placed.cx, placed.cy
                        );
                        if placed.rotation != 0.0 {
                            s.push_str(&format!(" | Rot: {:.1}\u{b0}", placed.rotation.to_degrees()));
                        }
                        Some(s)
                    } else {
                        None
                    }
                }
                _ => None,
            }
        });

        // Check if Pencil tool is active before consuming tools
        let pencil_active = tools
            .as_ref()
            .is_some_and(|t| t.active_tool == crate::components::tools::Tool::Pencil);
        // Check if Fill tool is active before consuming tools
        let fill_active = tools
            .as_ref()
            .is_some_and(|t| t.active_tool == crate::components::tools::Tool::Fill);
        let magic_wand_active = tools
            .as_ref()
            .is_some_and(|t| t.active_tool == crate::components::tools::Tool::MagicWand);
        // Check if Color Picker tool is active before consuming tools
        let color_picker_active = tools
            .as_ref()
            .is_some_and(|t| t.active_tool == crate::components::tools::Tool::ColorPicker);
        // Extract active tool for cursor icon mapping
        let active_tool_for_cursor = tools.as_ref().map(|t| t.active_tool);
        // Extract text tool handle state for cursor icon
        let text_hovering_handle = tools.as_ref().is_some_and(|t| t.text_state.hovering_handle);
        let text_dragging_handle = tools.as_ref().is_some_and(|t| t.text_state.dragging_handle);
        let text_hovering_rotation = tools
            .as_ref()
            .is_some_and(|t| t.text_state.hovering_rotation_handle);
        let text_hovering_resize = tools
            .as_ref()
            .and_then(|t| t.text_state.hovering_resize_handle);
        let text_resizing = tools
            .as_ref()
            .and_then(|t| match t.text_state.text_box_drag {
                Some(
                    dt @ (crate::components::tools::TextBoxDragType::ResizeLeft
                    | crate::components::tools::TextBoxDragType::ResizeRight
                    | crate::components::tools::TextBoxDragType::ResizeTopLeft
                    | crate::components::tools::TextBoxDragType::ResizeTopRight
                    | crate::components::tools::TextBoxDragType::ResizeBottomLeft
                    | crate::components::tools::TextBoxDragType::ResizeBottomRight),
                ) => Some(dt),
                _ => None,
            });
        let text_rotating = tools.as_ref().is_some_and(|t| {
            matches!(
                t.text_state.text_box_drag,
                Some(crate::components::tools::TextBoxDragType::Rotate)
            )
        });
        // Extract line tool pan-handle state for cursor icon
        let line_pan_hovering = tools.as_ref().is_some_and(|t| {
            t.active_tool == crate::components::tools::Tool::Line
                && t.line_state.line_tool.stage == crate::components::tools::LineStage::Editing
                && t.line_state.line_tool.pan_handle_hovering
        });
        let line_pan_dragging = tools.as_ref().is_some_and(|t| {
            t.active_tool == crate::components::tools::Tool::Line
                && t.line_state.line_tool.pan_handle_dragging
        });
        // Extract zoom tool out-mode for cursor icon
        let _zoom_out_mode = tools
            .as_ref()
            .is_some_and(|t| t.zoom_tool_state.zoom_out_mode);
        // Extract brush cursor info before tools ref is consumed
        // (size, clone_source, is_circle, mask_data_for_cursor, rotation_deg)
        let brush_cursor_info: Option<(f32, Option<Pos2>, bool, Option<(Vec<u8>, u32)>, f32)> =
            tools.as_ref().and_then(|t| {
                use crate::components::tools::Tool;
                match t.active_tool {
                    Tool::Brush
                    | Tool::Eraser
                    | Tool::CloneStamp
                    | Tool::ContentAwareBrush
                    | Tool::Liquify => {
                        let clone_source = if t.active_tool == Tool::CloneStamp {
                            t.clone_stamp_state.source
                        } else {
                            None
                        };
                        let is_circle_tip = t.properties.brush_tip.is_circle();
                        // For image tips, pass the mask so cursor can draw shape overlay
                        let mask_info = if !is_circle_tip && !t.brush_tip_mask.is_empty() {
                            Some((t.brush_tip_mask.clone(), t.brush_tip_mask_size))
                        } else {
                            None
                        };
                        // For fixed rotation, pass angle to cursor; for random, 0 (no preview)
                        let cursor_rotation = t.active_tip_rotation_deg;
                        Some((
                            t.pressure_size(),
                            clone_source,
                            is_circle_tip,
                            mask_info,
                            cursor_rotation,
                        ))
                    }
                    _ => None,
                }
            });
        let tip_name_for_cursor: Option<String> =
            tools.as_ref().and_then(|t| match &t.properties.brush_tip {
                crate::components::tools::BrushTip::Image(name) => Some(name.clone()),
                _ => None,
            });

        // Handle tool input - Call every frame while mouse button is held
        if let Some(tools) = tools {
            // Only block input if there's a modal window/popup or pointer is over any UI element
            let pointer_over_egui = ui.ctx().is_pointer_over_egui();
            // On Wayland the stylus may fire Touch events instead of Pointer events,
            // so also check for active Touch events when determining if pointer is over egui.
            let pointer_over_egui_with_touch = pointer_over_egui
                || ui.input(|i| {
                    i.events.iter().any(|e| {
                        matches!(
                            e,
                            egui::Event::Touch {
                                phase: egui::TouchPhase::Start | egui::TouchPhase::Move,
                                ..
                            }
                        )
                    })
                });
            let ui_blocking = egui::Popup::is_any_open(ui.ctx()) || pointer_over_egui_with_touch;

            // Get mouse position and check if over canvas.
            // On Wayland with a graphics tablet the stylus may route events as
            // Touch rather than Pointer, so fall back to the most recent touch
            // position when the pointer position is unavailable.
            let mouse_pos = ui.input(|i| {
                i.pointer.interact_pos().or_else(|| {
                    i.events.iter().rev().find_map(|e| match e {
                        egui::Event::PointerButton {
                            pressed: true, pos, ..
                        } => Some(*pos),
                        egui::Event::Touch {
                            phase: egui::TouchPhase::Start | egui::TouchPhase::Move,
                            pos,
                            ..
                        } => Some(*pos),
                        _ => None,
                    })
                })
            });
            let pointer_over_canvas = mouse_pos.is_some_and(|pos| canvas_rect.contains(pos));
            let raw_pointer_button_on_canvas = ui.input(|i| {
                if pointer_over_egui {
                    return false;
                }
                i.events.iter().any(|e| match e {
                    egui::Event::PointerButton {
                        pressed: true, pos, ..
                    } => image_rect.contains(*pos),
                    egui::Event::Touch {
                        phase: egui::TouchPhase::Start,
                        pos,
                        ..
                    } => image_rect.contains(*pos),
                    _ => false,
                })
            });
            let shift_held_now = ui.input(|i| i.modifiers.shift);
            let raw_fill_clicks: Vec<crate::components::tools::FillPendingClick> =
                if tools.active_tool == crate::components::tools::Tool::Fill && !pointer_over_egui
                {
                    ui.input(|i| {
                        i.events
                            .iter()
                            .filter_map(|e| match e {
                                egui::Event::PointerButton {
                                    pressed: true,
                                    pos,
                                    button,
                                    ..
                                } if matches!(
                                    button,
                                    egui::PointerButton::Primary
                                        | egui::PointerButton::Secondary
                                ) && image_rect.contains(*pos) =>
                                {
                                    self.screen_to_canvas(*pos, canvas_rect, state).map(
                                        |canvas_pos| crate::components::tools::FillPendingClick {
                                            pos: canvas_pos,
                                            use_secondary: *button
                                                == egui::PointerButton::Secondary,
                                            global_fill: shift_held_now,
                                        },
                                    )
                                }
                                egui::Event::Touch {
                                    phase: egui::TouchPhase::Start,
                                    pos,
                                    ..
                                } if image_rect.contains(*pos) => {
                                    self.screen_to_canvas(*pos, canvas_rect, state).map(
                                        |canvas_pos| crate::components::tools::FillPendingClick {
                                            pos: canvas_pos,
                                            use_secondary: false,
                                            global_fill: shift_held_now,
                                        },
                                    )
                                }
                                _ => None,
                            })
                            .collect()
                    })
                } else {
                    Vec::new()
                };

            // Get canvas position (will be None if not over image)
            let canvas_pos =
                mouse_pos.and_then(|pos| self.screen_to_canvas(pos, canvas_rect, state));
            let canvas_pos_f32 =
                mouse_pos.and_then(|pos| self.screen_to_canvas_f32(pos, canvas_rect, state));
            // Extended version for overlay tools (mesh warp) ÔÇö allows clicks slightly outside canvas bounds
            let canvas_pos_f32_clamped = mouse_pos
                .and_then(|pos| self.screen_to_canvas_f32_clamped(pos, canvas_rect, state, 16.0));
            // Unclamped version for selection/gradient: always returns coords even outside canvas
            let canvas_pos_unclamped =
                mouse_pos.map(|pos| self.screen_to_canvas_unclamped(pos, canvas_rect, state));

            // Check if we're in the middle of a stroke (mouse/touch button is held)
            let is_painting = ui.input(|i| {
                i.pointer.primary_down()
                    || i.pointer.secondary_down()
                    || i.events.iter().any(|e| {
                        matches!(
                            e,
                            egui::Event::Touch {
                                phase: egui::TouchPhase::Start | egui::TouchPhase::Move,
                                ..
                            }
                        )
                    })
            });

            // Collect ALL sub-frame PointerMoved events for smooth strokes.
            // At 21 FPS with a 1000Hz mouse, there can be ~48 intermediate
            // positions between frames. Processing all of them eliminates gaps.
            // When painting, use unclamped conversion so strokes can start from
            // off-canvas and still track motion back onto the canvas.
            let raw_motion_events: Vec<(f32, f32)> = if is_painting {
                ui.input(|i| {
                    i.events
                        .iter()
                        .filter_map(|e| {
                            let screen_pos = match e {
                                egui::Event::PointerMoved(pos) => *pos,
                                egui::Event::Touch {
                                    phase: egui::TouchPhase::Move,
                                    pos,
                                    ..
                                } => *pos,
                                _ => return None,
                            };
                            // Convert screen coordinates to canvas coordinates
                            // Inline the conversion to avoid borrow issues
                            let image_width = state.width as f32 * self.zoom;
                            let image_height = state.height as f32 * self.zoom;
                            let center_x = canvas_rect.center().x + self.pan_offset.x;
                            let center_y = canvas_rect.center().y + self.pan_offset.y;
                            let ir = Rect::from_center_size(
                                Pos2::new(center_x, center_y),
                                Vec2::new(image_width, image_height),
                            );
                            let rel_x = (screen_pos.x - ir.min.x) / self.zoom;
                            let rel_y = (screen_pos.y - ir.min.y) / self.zoom;
                            // Pass raw coordinates — draw functions clip to canvas
                            // bounds internally. Off-canvas positions produce no
                            // pixels but maintain natural stroke flow.
                            Some((rel_x, rel_y))
                        })
                        .collect()
                })
            } else {
                Vec::new()
            };

            // Only process if not blocked by UI AND (pointer over canvas OR actively painting
            // OR a tool drag is in progress that should allow off-canvas tracking)
            // AND paste overlay didn't consume the input
            let tool_drag_active = tools.selection_state.dragging
                || tools.lasso_state.dragging
                || tools.gradient_state.dragging
                || tools.text_state.text_box_drag.is_some()
                || tools.text_state.dragging_handle
                // Magic Wand reacts to a single click; block only if pointer genuinely
                // left the canvas, not because a UI panel is "over" the viewport rect.
                || tools.active_tool == crate::components::tools::Tool::MagicWand
                // Brush/Pencil/Eraser/Line: allow off-canvas drag so users can start
                // strokes from outside the canvas for better edge coverage.
                || tools.line_state.line_tool.stage != crate::components::tools::LineStage::Idle
                || tools.is_stroke_active();
            // When editing a text layer block, handles (rotation, delete) can be drawn outside
            // canvas bounds ÔÇö allow input so the user can click/drag them.
            // Do NOT override ui_blocking when pointer is over egui UI (e.g. tool shelf),
            // otherwise clicking Bold/Italic buttons passes through to the canvas.
            let text_handles_active =
                tools.text_state.is_editing && tools.text_state.editing_text_layer;
            let text_drag_override =
                (text_handles_active || tools.text_state.text_box_drag.is_some())
                    && !pointer_over_egui_with_touch;
            let keyboard_finalize_pressed =
                ui.input(|i| i.key_pressed(egui::Key::Enter) || i.key_pressed(egui::Key::Escape));
            let pending_tool_commit = tools.mesh_warp_state.commit_pending
                || tools.liquify_state.commit_pending
                || tools.gradient_state.commit_pending
                || tools.text_state.commit_pending;
            let keyboard_tool_override = keyboard_finalize_pressed
                && (tools.mesh_warp_state.is_active
                    || tools.liquify_state.is_active
                    || tools.shapes_state.placed.is_some()
                    || tools.perspective_crop_state.active);
            let allow_input = !modal_open
                && !paste_consumed_input
                && (!ui_blocking
                    || raw_pointer_button_on_canvas
                    || text_drag_override
                    || keyboard_tool_override
                    || pending_tool_commit)
                && (pointer_over_canvas
                    || raw_pointer_button_on_canvas
                    || is_painting
                    || tool_drag_active
                    || text_handles_active
                    || keyboard_tool_override
                    || pending_tool_commit);

            if allow_input {
                if tools.active_tool == crate::components::tools::Tool::Fill {
                    for click in raw_fill_clicks {
                        if tools.fill_state.pending_clicks.len() >= 64 {
                            let _ = tools.fill_state.pending_clicks.pop_front();
                        }
                        tools.fill_state.pending_clicks.push_back(click);
                    }
                }
                tools.handle_input(
                    ui,
                    state,
                    canvas_pos,
                    canvas_pos_f32,
                    canvas_pos_f32_clamped,
                    canvas_pos_unclamped,
                    &raw_motion_events,
                    &painter,
                    image_rect,
                    self.zoom,
                    primary_color_f32,
                    secondary_color_f32,
                    Some(&mut self.gpu_renderer),
                );

                // Consume zoom/pan action emitted by the Zoom or Pan tool
                use crate::components::tools::ZoomPanAction;
                let action = std::mem::replace(&mut tools.zoom_pan_action, ZoomPanAction::None);
                match action {
                    ZoomPanAction::None => {}
                    ZoomPanAction::ZoomIn { canvas_x, canvas_y } => {
                        // Zoom in anchored at the click point
                        let screen_x =
                            canvas_rect.min.x + self.pan_offset.x + canvas_rect.width() / 2.0
                                - (state.width as f32 * self.zoom / 2.0)
                                + canvas_x * self.zoom;
                        let screen_y =
                            canvas_rect.min.y + self.pan_offset.y + canvas_rect.height() / 2.0
                                - (state.height as f32 * self.zoom / 2.0)
                                + canvas_y * self.zoom;
                        let anchor = Pos2::new(screen_x, screen_y);
                        self.zoom_around_screen_point(1.2, anchor, canvas_rect);
                    }
                    ZoomPanAction::ZoomOut { canvas_x, canvas_y } => {
                        let screen_x =
                            canvas_rect.min.x + self.pan_offset.x + canvas_rect.width() / 2.0
                                - (state.width as f32 * self.zoom / 2.0)
                                + canvas_x * self.zoom;
                        let screen_y =
                            canvas_rect.min.y + self.pan_offset.y + canvas_rect.height() / 2.0
                                - (state.height as f32 * self.zoom / 2.0)
                                + canvas_y * self.zoom;
                        let anchor = Pos2::new(screen_x, screen_y);
                        self.zoom_around_screen_point(1.0 / 1.2, anchor, canvas_rect);
                    }
                    ZoomPanAction::Pan { dx, dy } => {
                        self.pan_by(Vec2::new(dx, dy));
                    }
                    ZoomPanAction::ZoomToRect {
                        min_x,
                        min_y,
                        max_x,
                        max_y,
                    } => {
                        // min/max are in canvas pixel coordinates
                        let rect_w = (max_x - min_x).max(1.0);
                        let rect_h = (max_y - min_y).max(1.0);
                        let viewport_w = canvas_rect.width();
                        let viewport_h = canvas_rect.height();
                        // Desired zoom: fit the selected region into the viewport
                        let desired_zoom = (viewport_w / rect_w)
                            .min(viewport_h / rect_h)
                            .clamp(0.1, 100.0);
                        self.zoom = desired_zoom;
                        // Pan so that the rect center is at the viewport center
                        // Image center is at canvas_rect.center() + pan_offset
                        // Canvas point (cx,cy) is at screen: image_center - image_size/2 + cx*zoom
                        // We want rect_center to map to canvas_rect.center():
                        //   canvas_rect.center().x + pan.x - (w*zoom)/2 + rcx*zoom = canvas_rect.center().x
                        //   pan.x = (w*zoom)/2 - rcx*zoom
                        let rcx = (min_x + max_x) / 2.0;
                        let rcy = (min_y + max_y) / 2.0;
                        let image_w = state.width as f32 * self.zoom;
                        let image_h = state.height as f32 * self.zoom;
                        self.pan_offset = Vec2::new(
                            image_w / 2.0 - rcx * self.zoom,
                            image_h / 2.0 - rcy * self.zoom,
                        );
                    }
                }
            }

            // Always check if gradient settings changed and re-render preview,
            // even when handle_input is blocked by UI (e.g. context bar interactions)
            if tools.active_tool == crate::components::tools::Tool::Gradient {
                tools.update_gradient_if_dirty(state, Some(&mut self.gpu_renderer));
            }

            // Always check if text/shape properties changed (color picker, context bar)
            if tools.active_tool == crate::components::tools::Tool::Text {
                tools.update_text_if_dirty(state, primary_color_f32);
                // Draw overlay (border, handle, cursor) even when pointer is off-canvas
                tools.draw_text_overlay(ui, state, &painter, image_rect, self.zoom);
            }
            if tools.active_tool == crate::components::tools::Tool::Shapes {
                tools.update_shape_if_dirty(state, primary_color_f32, secondary_color_f32);
            }

            // Track fill recalculation state for the loading bar
            self.fill_recalc_active =
                tools.fill_state.recalc_pending && tools.fill_state.active_fill.is_some();

            // Track tool map warmup/build state for the loading bar
            self.tool_map_build_label = tools.debug_operation_label();

            // Track gradient commit state for the loading bar
            self.gradient_commit_active = tools.gradient_state.commit_pending
                || tools.text_state.commit_pending
                || tools.liquify_state.commit_pending
                || tools.mesh_warp_state.commit_pending;

            // ====================================================================
            // TOOL-SPECIFIC CURSOR ICON
            // ====================================================================
            {
                let over_image = mouse_pos.is_some_and(|pos| image_rect.contains(pos));
                // Only override cursor when mouse is truly over just the canvas ÔÇö
                // not when a dialog, menu, popup, or floating panel is on top.
                // Exception: text rotation/resize handles may be drawn outside image bounds.
                let text_handle_cursor = text_hovering_rotation
                    || text_rotating
                    || text_hovering_resize.is_some()
                    || text_resizing.is_some();
                if (over_image || text_handle_cursor)
                    && !modal_open
                    && !egui::Popup::is_any_open(ui.ctx())
                    && (!pointer_over_egui_with_touch || text_handle_cursor)
                    && let Some(tool) = active_tool_for_cursor
                {
                    use crate::components::tools::Tool;
                    let is_dragging = ui.input(|i| {
                        i.pointer.primary_down()
                            || i.events.iter().any(|e| {
                                matches!(
                                    e,
                                    egui::Event::Touch {
                                        phase: egui::TouchPhase::Start | egui::TouchPhase::Move,
                                        ..
                                    }
                                )
                            })
                    });
                    let cursor = match tool {
                        // Tools with custom icon cursor ÔÇö hide OS cursor
                        Tool::Pencil | Tool::Fill | Tool::ColorPicker | Tool::Zoom | Tool::Pan => {
                            egui::CursorIcon::None
                        }
                        // Remaining precision tools ÔÇö crosshair
                        Tool::RectangleSelect
                        | Tool::EllipseSelect
                        | Tool::Lasso
                        | Tool::PerspectiveCrop
                        | Tool::MagicWand
                        | Tool::ColorRemover
                        | Tool::Shapes
                        | Tool::Gradient => egui::CursorIcon::Crosshair,
                        // Line tool: move cursor when on pan handle, crosshair otherwise
                        Tool::Line => {
                            if line_pan_dragging {
                                egui::CursorIcon::Grabbing
                            } else if line_pan_hovering {
                                egui::CursorIcon::Move
                            } else {
                                egui::CursorIcon::Crosshair
                            }
                        }
                        // Brush-type tools ÔÇö None (custom circle overlay drawn)
                        Tool::Brush
                        | Tool::Eraser
                        | Tool::CloneStamp
                        | Tool::ContentAwareBrush
                        | Tool::Liquify
                        | Tool::Smudge => egui::CursorIcon::None,
                        // Move tools ÔÇö move/grabbing
                        Tool::MovePixels | Tool::MoveSelection => {
                            if is_dragging {
                                egui::CursorIcon::Grabbing
                            } else {
                                egui::CursorIcon::Move
                            }
                        }
                        // Mesh warp ÔÇö crosshair for precise control point dragging
                        Tool::MeshWarp => egui::CursorIcon::Crosshair,
                        // Text ÔÇö context-dependent cursor for handles
                        Tool::Text => {
                            // Map resize drag type to appropriate system resize cursor
                            let resize_cursor = |dt: crate::components::tools::TextBoxDragType| -> egui::CursorIcon {
                                use crate::components::tools::TextBoxDragType;
                                match dt {
                                    TextBoxDragType::ResizeLeft | TextBoxDragType::ResizeRight => egui::CursorIcon::ResizeHorizontal,
                                    TextBoxDragType::ResizeTopLeft | TextBoxDragType::ResizeBottomRight => egui::CursorIcon::ResizeNwSe,
                                    TextBoxDragType::ResizeTopRight | TextBoxDragType::ResizeBottomLeft => egui::CursorIcon::ResizeNeSw,
                                    TextBoxDragType::Rotate => egui::CursorIcon::Alias,
                                }
                            };
                            if let Some(dt) = text_resizing {
                                resize_cursor(dt)
                            } else if text_dragging_handle {
                                egui::CursorIcon::Grabbing
                            } else if text_rotating || text_hovering_rotation {
                                egui::CursorIcon::Alias
                            } else if let Some(dt) = text_hovering_resize {
                                resize_cursor(dt)
                            } else if text_hovering_handle {
                                egui::CursorIcon::Move
                            } else {
                                egui::CursorIcon::Text
                            }
                        } // Zoom / Pan now handled above (custom icon cursor)
                          // (unreachable ÔÇö already matched in first arm)
                    };
                    ui.ctx().set_cursor_icon(cursor);
                }
            }

            // ====================================================================
            // FILL CURSOR OVERLAY (shows exact pixel that will be flood-filled)
            // ====================================================================
            if fill_active
                && let Some(pos) = mouse_pos
                && canvas_rect.contains(pos)
                && let Some((canvas_x_f32, canvas_y_f32)) =
                    self.screen_to_canvas_f32(pos, canvas_rect, state)
            {
                let canvas_x = canvas_x_f32.floor() as u32;
                let canvas_y = canvas_y_f32.floor() as u32;
                let pixel_screen_x = image_rect.min.x + canvas_x as f32 * self.zoom;
                let pixel_screen_y = image_rect.min.y + canvas_y as f32 * self.zoom;
                let pixel_rect = Rect::from_min_size(
                    Pos2::new(pixel_screen_x, pixel_screen_y),
                    Vec2::new(self.zoom.max(1.0), self.zoom.max(1.0)),
                );
                painter.rect_stroke(
                    pixel_rect,
                    0.0,
                    egui::Stroke::new(1.0, Color32::from_black_alpha(180)),
                    egui::StrokeKind::Middle,
                );
                painter.rect_stroke(
                    pixel_rect,
                    0.0,
                    egui::Stroke::new(0.5, Color32::from_white_alpha(200)),
                    egui::StrokeKind::Middle,
                );
            }

            // ====================================================================
            // MAGIC WAND CURSOR OVERLAY (shows exact seed pixel)
            // ====================================================================
            if magic_wand_active
                && let Some(pos) = mouse_pos
                && canvas_rect.contains(pos)
                && let Some((canvas_x_f32, canvas_y_f32)) =
                    self.screen_to_canvas_f32(pos, canvas_rect, state)
            {
                let canvas_x = canvas_x_f32.floor() as u32;
                let canvas_y = canvas_y_f32.floor() as u32;
                let pixel_screen_x = image_rect.min.x + canvas_x as f32 * self.zoom;
                let pixel_screen_y = image_rect.min.y + canvas_y as f32 * self.zoom;
                let pixel_rect = Rect::from_min_size(
                    Pos2::new(pixel_screen_x, pixel_screen_y),
                    Vec2::new(self.zoom.max(1.0), self.zoom.max(1.0)),
                );
                painter.rect_stroke(
                    pixel_rect,
                    0.0,
                    egui::Stroke::new(1.0, Color32::from_black_alpha(180)),
                    egui::StrokeKind::Middle,
                );
                painter.rect_stroke(
                    pixel_rect,
                    0.0,
                    egui::Stroke::new(0.5, Color32::from_white_alpha(200)),
                    egui::StrokeKind::Middle,
                );
            }

            // ====================================================================
            // PENCIL CURSOR OVERLAY (shows exact pixel that will be painted)
            // ====================================================================
            // Only show when Pencil tool is active, mouse is over canvas, and not painting
            if pencil_active
                && let Some(pos) = mouse_pos
                && canvas_rect.contains(pos)
                && !is_painting
            {
                // Convert screen position to canvas position (floating point)
                if let Some((canvas_x_f32, canvas_y_f32)) =
                    self.screen_to_canvas_f32(pos, canvas_rect, state)
                {
                    // Floor to get the pixel that contains the cursor (not nearest pixel)
                    let canvas_x = canvas_x_f32.floor() as u32;
                    let canvas_y = canvas_y_f32.floor() as u32;

                    // Convert back to screen coordinates for the exact pixel
                    let pixel_screen_x = image_rect.min.x + canvas_x as f32 * self.zoom;
                    let pixel_screen_y = image_rect.min.y + canvas_y as f32 * self.zoom;

                    // Draw a subtle 1x1 pixel outline to show which pixel will be painted
                    let pixel_rect = Rect::from_min_size(
                        Pos2::new(pixel_screen_x, pixel_screen_y),
                        Vec2::new(self.zoom.max(1.0), self.zoom.max(1.0)),
                    );

                    // Use a contrasting color for visibility
                    let cursor_color = if ui.visuals().dark_mode {
                        Color32::from_rgb(255, 255, 255) // White in dark mode
                    } else {
                        Color32::from_rgb(0, 0, 0) // Black in light mode
                    };

                    painter.rect_stroke(
                        pixel_rect,
                        0.0,
                        egui::Stroke::new(1.0, cursor_color),
                        egui::StrokeKind::Middle,
                    );
                }
            }

            // ====================================================================
            // COLOR PICKER CURSOR OVERLAY (shows sampled pixel)
            // ====================================================================
            if color_picker_active
                && let Some(pos) = mouse_pos
                && canvas_rect.contains(pos)
            {
                // Convert screen position to canvas position (floating point)
                if let Some((canvas_x_f32, canvas_y_f32)) =
                    self.screen_to_canvas_f32(pos, canvas_rect, state)
                {
                    // Floor to get the pixel that contains the cursor
                    let canvas_x = canvas_x_f32.floor() as u32;
                    let canvas_y = canvas_y_f32.floor() as u32;

                    // Convert back to screen coordinates for the exact pixel
                    let pixel_screen_x = image_rect.min.x + canvas_x as f32 * self.zoom;
                    let pixel_screen_y = image_rect.min.y + canvas_y as f32 * self.zoom;

                    // Draw the pixel outline for color picker
                    let pixel_rect = Rect::from_min_size(
                        Pos2::new(pixel_screen_x, pixel_screen_y),
                        Vec2::new(self.zoom.max(1.0), self.zoom.max(1.0)),
                    );

                    // Use a contrasting color for visibility (brighter cyan/yellow)
                    let cursor_color = if ui.visuals().dark_mode {
                        Color32::from_rgb(0, 255, 255) // Cyan in dark mode
                    } else {
                        Color32::from_rgb(255, 200, 0) // Yellow in light mode
                    };

                    painter.rect_stroke(
                        pixel_rect,
                        0.0,
                        egui::Stroke::new(1.0, cursor_color),
                        egui::StrokeKind::Middle,
                    );
                }
            }

            // ====================================================================
            // TOOL ICON CURSOR OVERLAY
            // Draws the active tool's icon near the cursor for Pencil, Fill,
            // ColorPicker, Zoom, Pan.
            // Crisp border: white icon scaled +3px behind, full-dark icon on top.
            // ====================================================================
            {
                let needs_icon_cursor = active_tool_for_cursor.is_some_and(|t| {
                    use crate::components::tools::Tool;
                    matches!(
                        t,
                        Tool::Pencil | Tool::Fill | Tool::ColorPicker | Tool::Zoom | Tool::Pan
                    )
                });
                if needs_icon_cursor
                    && let Some(ref icon_tex) = self.tool_cursor_icon
                    && let Some(pos) = mouse_pos
                    && canvas_rect.contains(pos)
                {
                    use crate::components::tools::Tool;
                    let icon_sz = 18.0_f32;
                    // Hot-spot: tip tools anchor bottom-left (tip at cursor),
                    // spatial tools anchor center.
                    let center = match active_tool_for_cursor {
                        Some(Tool::Zoom) | Some(Tool::Pan) => Pos2::new(pos.x, pos.y),
                        _ => Pos2::new(pos.x + icon_sz * 0.5 + 1.0, pos.y - icon_sz * 0.5),
                    };
                    let uv = egui::Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(1.0, 1.0));
                    // White border: draw icon 3px larger (centered same point)
                    let border_sz = icon_sz + 4.0;
                    let border_rect = Rect::from_center_size(center, Vec2::splat(border_sz));
                    painter.image(
                        icon_tex.id(),
                        border_rect,
                        uv,
                        Color32::from_white_alpha(230),
                    );
                    // Dark icon on top at normal size
                    let icon_rect = Rect::from_center_size(center, Vec2::splat(icon_sz));
                    painter.image(icon_tex.id(), icon_rect, uv, Color32::from_black_alpha(230));
                } // if let icon_tex
                // if needs_icon_cursor
            }

            // ====================================================================
            // BRUSH CURSOR OVERLAY (circle outline matching brush size)
            // Shown for Brush, Eraser, Clone Stamp, Content Aware Brush
            // ====================================================================
            if let Some((
                brush_size,
                clone_source,
                is_circle_tip,
                ref mask_info,
                cursor_rotation_deg,
            )) = brush_cursor_info
                && let Some(pos) = mouse_pos
                && canvas_rect.contains(pos)
                && let Some((canvas_x, canvas_y)) =
                    self.screen_to_canvas_f32(pos, canvas_rect, state)
            {
                // Brush center in screen coords
                let screen_cx = image_rect.min.x + canvas_x * self.zoom;
                let screen_cy = image_rect.min.y + canvas_y * self.zoom;
                let screen_radius = (brush_size / 2.0) * self.zoom;

                // Draw brush outline ÔÇö circle for circle tip, texture overlay for image tips
                if screen_radius > 1.5 {
                    if is_circle_tip {
                        // Draw two circles (black + white) for visibility on any background
                        let stroke_outer = egui::Stroke::new(1.5, Color32::from_black_alpha(160));
                        let stroke_inner = egui::Stroke::new(0.75, Color32::from_white_alpha(200));
                        painter.circle_stroke(
                            Pos2::new(screen_cx, screen_cy),
                            screen_radius,
                            stroke_outer,
                        );
                        painter.circle_stroke(
                            Pos2::new(screen_cx, screen_cy),
                            screen_radius,
                            stroke_inner,
                        );
                    } else if let Some((mask_data, mask_sz)) = mask_info {
                        // Image tip ÔÇö overlay the mask shape twice (white + black)
                        // for visibility on any background colour.
                        let ms = *mask_sz;
                        let tip_name = match &tip_name_for_cursor {
                            Some(n) => n.as_str(),
                            None => "",
                        };
                        // Include mask content hash via simple checksum for staleness
                        let content_hash = mask_data
                            .iter()
                            .step_by(7)
                            .fold(0u32, |acc, &b| acc.wrapping_add(b as u32));
                        let key = (tip_name.to_string(), ms, content_hash);
                        if key != self.brush_tip_cursor_key || self.brush_tip_cursor_tex.is_none() {
                            // Rebuild cursor textures from mask
                            let n = (ms * ms) as usize;
                            let mut rgba_white = vec![0u8; n * 4];
                            let mut rgba_black = vec![0u8; n * 4];
                            for i in 0..n {
                                let a = mask_data[i];
                                // White version
                                rgba_white[i * 4] = 255;
                                rgba_white[i * 4 + 1] = 255;
                                rgba_white[i * 4 + 2] = 255;
                                rgba_white[i * 4 + 3] = a;
                                // Black version
                                rgba_black[i * 4] = 0;
                                rgba_black[i * 4 + 1] = 0;
                                rgba_black[i * 4 + 2] = 0;
                                rgba_black[i * 4 + 3] = a;
                            }
                            let opts = egui::TextureOptions {
                                magnification: egui::TextureFilter::Linear,
                                minification: egui::TextureFilter::Linear,
                                ..Default::default()
                            };
                            let ci_w = egui::ColorImage::from_rgba_unmultiplied(
                                [ms as usize, ms as usize],
                                &rgba_white,
                            );
                            let ci_b = egui::ColorImage::from_rgba_unmultiplied(
                                [ms as usize, ms as usize],
                                &rgba_black,
                            );
                            if let Some(ref mut tex) = self.brush_tip_cursor_tex {
                                tex.set(ci_w, opts);
                            } else {
                                self.brush_tip_cursor_tex =
                                    Some(ui.ctx().load_texture("brush_tip_cursor_w", ci_w, opts));
                            }
                            if let Some(ref mut tex) = self.brush_tip_cursor_tex_inv {
                                tex.set(ci_b, opts);
                            } else {
                                self.brush_tip_cursor_tex_inv =
                                    Some(ui.ctx().load_texture("brush_tip_cursor_b", ci_b, opts));
                            }
                            self.brush_tip_cursor_key = key;
                        }
                        let screen_sz = brush_size * self.zoom;
                        let center = Pos2::new(screen_cx, screen_cy);

                        // Check if we need rotated drawing
                        let has_rotation = cursor_rotation_deg.abs() > 0.01;

                        if has_rotation {
                            // Rotated cursor: draw a rotated quad using a mesh
                            let half_sz = screen_sz / 2.0;
                            let rad = cursor_rotation_deg.to_radians();
                            let cos_r = rad.cos();
                            let sin_r = rad.sin();

                            // Rotated corners: TL, TR, BR, BL
                            let corners = [
                                (-half_sz, -half_sz),
                                (half_sz, -half_sz),
                                (half_sz, half_sz),
                                (-half_sz, half_sz),
                            ];
                            let rotated: Vec<Pos2> = corners
                                .iter()
                                .map(|&(dx, dy)| {
                                    Pos2::new(
                                        center.x + dx * cos_r - dy * sin_r,
                                        center.y + dx * sin_r + dy * cos_r,
                                    )
                                })
                                .collect();

                            let uv_corners = [
                                Pos2::new(0.0, 0.0),
                                Pos2::new(1.0, 0.0),
                                Pos2::new(1.0, 1.0),
                                Pos2::new(0.0, 1.0),
                            ];

                            // Draw white pass
                            if let Some(ref tex_w) = self.brush_tip_cursor_tex {
                                let tint_w = Color32::from_rgba_unmultiplied(255, 255, 255, 100);
                                let mut mesh_w = egui::Mesh::with_texture(tex_w.id());
                                for i in 0..4 {
                                    mesh_w.vertices.push(egui::epaint::Vertex {
                                        pos: rotated[i],
                                        uv: uv_corners[i],
                                        color: tint_w,
                                    });
                                }
                                mesh_w.indices = vec![0, 1, 2, 0, 2, 3];
                                painter.add(egui::Shape::mesh(mesh_w));
                            }
                            // Draw black pass
                            if let Some(ref tex_b) = self.brush_tip_cursor_tex_inv {
                                let tint_b = Color32::from_rgba_unmultiplied(255, 255, 255, 100);
                                let mut mesh_b = egui::Mesh::with_texture(tex_b.id());
                                for i in 0..4 {
                                    mesh_b.vertices.push(egui::epaint::Vertex {
                                        pos: rotated[i],
                                        uv: uv_corners[i],
                                        color: tint_b,
                                    });
                                }
                                mesh_b.indices = vec![0, 1, 2, 0, 2, 3];
                                painter.add(egui::Shape::mesh(mesh_b));
                            }
                        } else {
                            // No rotation ÔÇö use simple axis-aligned image draw
                            let cursor_rect =
                                Rect::from_center_size(center, egui::Vec2::splat(screen_sz));
                            let uv = Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0));
                            // Draw white pass then black pass at low opacity.
                            // On dark backgrounds the white shows; on light the black shows.
                            if let Some(ref tex_w) = self.brush_tip_cursor_tex {
                                let tint_w = Color32::from_rgba_unmultiplied(255, 255, 255, 100);
                                painter.image(tex_w.id(), cursor_rect, uv, tint_w);
                            }
                            if let Some(ref tex_b) = self.brush_tip_cursor_tex_inv {
                                let tint_b = Color32::from_rgba_unmultiplied(255, 255, 255, 100);
                                painter.image(tex_b.id(), cursor_rect, uv, tint_b);
                            }
                        }
                    } else {
                        // Fallback: square outline (dual stroke for visibility)
                        let stamp_rect = Rect::from_center_size(
                            Pos2::new(screen_cx, screen_cy),
                            egui::Vec2::splat(screen_radius * 2.0),
                        );
                        painter.rect_stroke(
                            stamp_rect,
                            0.0,
                            egui::Stroke::new(1.5, Color32::from_black_alpha(160)),
                            egui::StrokeKind::Middle,
                        );
                        painter.rect_stroke(
                            stamp_rect,
                            0.0,
                            egui::Stroke::new(0.75, Color32::from_white_alpha(200)),
                            egui::StrokeKind::Middle,
                        );
                    }
                } else {
                    // Very small brush ÔÇö draw crosshair (dual stroke for visibility)
                    let s = 4.0;
                    let stroke_o = egui::Stroke::new(1.5, Color32::from_black_alpha(160));
                    let stroke_i = egui::Stroke::new(0.75, Color32::from_white_alpha(200));
                    painter.line_segment(
                        [
                            Pos2::new(screen_cx - s, screen_cy),
                            Pos2::new(screen_cx + s, screen_cy),
                        ],
                        stroke_o,
                    );
                    painter.line_segment(
                        [
                            Pos2::new(screen_cx - s, screen_cy),
                            Pos2::new(screen_cx + s, screen_cy),
                        ],
                        stroke_i,
                    );
                    painter.line_segment(
                        [
                            Pos2::new(screen_cx, screen_cy - s),
                            Pos2::new(screen_cx, screen_cy + s),
                        ],
                        stroke_o,
                    );
                    painter.line_segment(
                        [
                            Pos2::new(screen_cx, screen_cy - s),
                            Pos2::new(screen_cx, screen_cy + s),
                        ],
                        stroke_i,
                    );
                }

                // Draw clone stamp source crosshair
                if let Some(src) = clone_source {
                    let src_sx = image_rect.min.x + src.x * self.zoom;
                    let src_sy = image_rect.min.y + src.y * self.zoom;
                    let cross_size = 8.0;
                    let src_color = Color32::from_rgb(255, 100, 100);
                    let src_stroke = egui::Stroke::new(1.5, src_color);
                    painter.line_segment(
                        [
                            Pos2::new(src_sx - cross_size, src_sy),
                            Pos2::new(src_sx + cross_size, src_sy),
                        ],
                        src_stroke,
                    );
                    painter.line_segment(
                        [
                            Pos2::new(src_sx, src_sy - cross_size),
                            Pos2::new(src_sx, src_sy + cross_size),
                        ],
                        src_stroke,
                    );
                    // Also draw source brush circle if painting
                    if is_painting && screen_radius > 1.5 {
                        // Compute current offset
                        let offset_x = src.x - canvas_x;
                        let offset_y = src.y - canvas_y;
                        let src_brush_cx = image_rect.min.x + (canvas_x + offset_x) * self.zoom;
                        let src_brush_cy = image_rect.min.y + (canvas_y + offset_y) * self.zoom;
                        painter.circle_stroke(
                            Pos2::new(src_brush_cx, src_brush_cy),
                            screen_radius,
                            egui::Stroke::new(
                                1.0,
                                Color32::from_rgba_unmultiplied(255, 100, 100, 120),
                            ),
                        );
                    }
                }
            }
        }

        // ====================================================================
        // DYNAMIC DEBUG PANEL  (bottom-right, context-sensitive)
        // ====================================================================
        if debug_settings.show_debug_panel {
            let mouse_pos = ui.input(|i| i.pointer.interact_pos());
            let canvas_pos =
                mouse_pos.and_then(|pos| self.screen_to_canvas(pos, canvas_rect, state));

            let debug_text = if let Some((cx, cy, sw, sh, rot_deg, scale_pct)) = paste_info {
                // Paste overlay active — show paste-specific info.
                format!(
                    "Paste Preview: {:.0}x{:.0} | Pos: ({:.0}, {:.0}) | Rot: {:.1}deg | Scale: {:.0}%",
                    sw, sh, cx, cy, rot_deg, scale_pct
                )
            } else if let Some((w, h)) = sel_drag_info {
                // Selection being drawn — show selection dimensions.
                let mouse_info = if let Some((mx, my)) = canvas_pos {
                    format!(" | Cursor: {}, {}", mx, my)
                } else {
                    String::new()
                };
                format!("Selection: {:.0}x{:.0}{}", w, h, mouse_info)
            } else {
                // Default — build custom info based on settings
                let mut parts = Vec::new();

                if debug_settings.debug_show_canvas_size {
                    parts.push(format!("{}x{}", state.width, state.height));
                }

                if debug_settings.debug_show_zoom {
                    parts.push(format!("{:.0}%", self.zoom * 100.0));
                }

                if let Some((x, y)) = canvas_pos {
                    parts.push(format!("{}, {}", x, y));
                }

                if debug_settings.debug_show_fps {
                    parts.push(format!("{:.0} FPS", self.fps));
                }

                if debug_settings.debug_show_gpu {
                    let gpu_name = &self.gpu_renderer.ctx.adapter_name;
                    // Shorten GPU name if too long
                    let display_name = if gpu_name.len() > 25 {
                        format!("{}...", &gpu_name[..22])
                    } else {
                        gpu_name.clone()
                    };
                    parts.push(display_name);
                }

                parts.join(" | ")
            };

            let text_galley = ui.painter().layout_no_wrap(
                debug_text,
                egui::FontId::monospace(9.0),
                Color32::from_gray(200),
            );

            let text_size = text_galley.size();
            let debug_pos =
                canvas_rect.right_bottom() - Vec2::new(text_size.x + 10.0, text_size.y + 10.0);
            let text_rect =
                egui::Align2::LEFT_TOP.anchor_rect(egui::Rect::from_min_size(debug_pos, text_size));

            // Draw semi-transparent black background (more subtle)
            painter.rect_filled(text_rect.expand(4.0), 2.0, Color32::from_black_alpha(120));

            // Draw text
            painter.galley(debug_pos, text_galley, egui::Color32::TRANSPARENT);

            // ====================================================================
            // LOADING OPERATIONS BAR (floating above debug panel)
            // Show always for filter/IO ops (user-visible progress),
            // show for fill/gradient only when debug panel is enabled.
            // ====================================================================
            let has_user_ops = pending_filter_jobs > 0 || pending_io_ops > 0;
            let has_debug_ops = self.fill_recalc_active
                || self.gradient_commit_active
                || self.tool_map_build_label.is_some();
            if has_user_ops || (debug_settings.debug_show_operations && has_debug_ops) {
                let current_time = ui.input(|i| i.time);
                let mut ops_parts = Vec::new();

                if pending_filter_jobs > 0 {
                    let elapsed = if let Some(start) = filter_ops_start_time {
                        format!("{:.1}s", current_time - start)
                    } else {
                        "0.0s".to_string()
                    };
                    let label = if !filter_status_description.is_empty() {
                        filter_status_description.to_string()
                    } else {
                        "Filter".to_string()
                    };
                    ops_parts.push(format!("Processing: {} ({})", label, elapsed));
                } else if pending_io_ops > 0 {
                    let elapsed = if let Some(start) = io_ops_start_time {
                        format!("{:.1}s", current_time - start)
                    } else {
                        "0.0s".to_string()
                    };
                    ops_parts.push(format!("Processing: I/O ({})", elapsed));
                } else if self.fill_recalc_active {
                    ops_parts.push("Updating: Fill Preview".to_string());
                } else if let Some(label) = &self.tool_map_build_label {
                    ops_parts.push(label.clone());
                } else if self.gradient_commit_active {
                    ops_parts.push("Committing: Tool".to_string());
                }

                let ops_text = ops_parts.join(" | ");
                let ops_galley = ui.painter().layout_no_wrap(
                    ops_text,
                    egui::FontId::monospace(9.0),
                    Color32::from_gray(220),
                );

                // Position above the debug panel (if visible), right-aligned
                let debug_panel_height = if debug_settings.debug_show_operations {
                    text_size.y + 10.0
                } else {
                    0.0
                };
                let vertical_offset = debug_panel_height + 10.0 + ops_galley.size().y + 4.0;
                let ops_pos = canvas_rect.right_bottom()
                    - Vec2::new(ops_galley.size().x + 10.0, vertical_offset);
                let ops_rect = egui::Align2::LEFT_TOP.anchor_rect(egui::Rect::from_min_size(
                    ops_pos,
                    egui::vec2(ops_galley.size().x, ops_galley.size().y + 4.0),
                ));

                // Draw semi-transparent background with slight accent
                painter.rect_filled(ops_rect.expand(4.0), 2.0, Color32::from_black_alpha(140));

                // Draw animated progress bar
                let time = ui.input(|i| i.time);
                let progress = ((time * 2.0).sin() + 1.0) / 2.0;
                let bar_width = ops_rect.width();
                let bar_rect = egui::Rect::from_min_size(
                    egui::pos2(ops_rect.min.x, ops_rect.max.y - 4.0),
                    egui::vec2(bar_width * progress as f32, 4.0),
                );
                painter.rect_filled(bar_rect, 0.0, accent_color);

                // Draw text
                painter.galley(ops_pos, ops_galley, egui::Color32::TRANSPARENT);
            }

            // ====================================================================
            // TOOL INFO OVERLAY  (bottom-left, context-sensitive, single line)
            // ====================================================================
            if debug_settings.show_tool_info && let Some(ref info_text) = tool_info_data {
                    let font_id = egui::FontId::monospace(9.0);
                    let text_color = Color32::from_gray(200);

                    let galley = ui.painter().layout_no_wrap(
                        info_text.clone(),
                        font_id,
                        text_color,
                    );

                    // Position at bottom-left of canvas
                    let info_pos = canvas_rect.left_bottom()
                        + egui::vec2(10.0, -(galley.size().y + 10.0));
                    let info_rect = egui::Align2::LEFT_TOP.anchor_rect(
                        egui::Rect::from_min_size(info_pos, galley.size()),
                    );

                    // Draw semi-transparent black background
                    painter.rect_filled(info_rect.expand(4.0), 2.0, Color32::from_black_alpha(120));

                    // Draw text
                    painter.galley(info_pos, galley, egui::Color32::TRANSPARENT);
            }

            // ====================================================================
            // PASTE PREVIEW BADGE  (bottom-left, below tool info, always visible)
            // ====================================================================
            if paste_overlay.is_some() {
                let badge_text = "PASTE PREVIEW";
                let font_id = egui::FontId::monospace(10.0);
                let badge_color = Color32::from_rgb(255, 200, 50); // amber/gold

                let galley = ui.painter().layout_no_wrap(
                    badge_text.to_string(),
                    font_id,
                    badge_color,
                );

                // Position below tool info (or at bottom-left if no tool info)
                let tool_info_height = if debug_settings.show_tool_info && tool_info_data.is_some() {
                    ui.painter().layout_no_wrap(
                        "X".to_string(),
                        egui::FontId::monospace(9.0),
                        Color32::from_gray(200),
                    ).size().y + 14.0
                } else {
                    0.0
                };
                let badge_pos = canvas_rect.left_bottom()
                    + egui::vec2(10.0, -(tool_info_height + galley.size().y + 10.0));
                let badge_rect = egui::Align2::LEFT_TOP.anchor_rect(
                    egui::Rect::from_min_size(badge_pos, galley.size()),
                );

                // Draw semi-transparent dark background
                painter.rect_filled(badge_rect.expand(4.0), 2.0, Color32::from_black_alpha(160));

                // Draw text
                painter.galley(badge_pos, galley, egui::Color32::TRANSPARENT);
            }
        }
    }

    pub fn zoom_in(&mut self) {
        self.zoom = (self.zoom * 1.2).min(100.0);
    }

    pub fn zoom_out(&mut self) {
        self.zoom = (self.zoom / 1.2).max(0.1);
    }

    pub fn reset_zoom(&mut self) {
        self.zoom = 1.0;
        self.pan_offset = Vec2::ZERO;
    }

    pub fn view_state(&self) -> (f32, Vec2) {
        (self.zoom, self.pan_offset)
    }

    pub fn set_view_state(&mut self, zoom: f32, pan_offset: Vec2) {
        self.zoom = zoom.clamp(0.1, 100.0);
        self.pan_offset = pan_offset;
    }

    pub fn apply_zoom(&mut self, zoom_factor: f32) {
        self.zoom = (self.zoom * zoom_factor).clamp(0.1, 100.0);
    }

    /// Zoom while keeping a screen-space point fixed (e.g. under the mouse cursor).
    /// `anchor` is in screen coordinates, `canvas_rect` is the viewport rect.
    pub fn zoom_around_screen_point(&mut self, zoom_factor: f32, anchor: Pos2, canvas_rect: Rect) {
        let old_zoom = self.zoom;
        self.zoom = (self.zoom * zoom_factor).clamp(0.1, 100.0);
        let actual_factor = self.zoom / old_zoom;
        // The image center in screen space is canvas_rect.center() + pan_offset.
        // After scaling, the anchor point would shift unless we compensate pan_offset.
        // offset from anchor to image-center: (center + pan) - anchor
        // After zoom, that offset scales by actual_factor, so:
        //   new_center = anchor + (old_center - anchor) * factor
        //   new_pan = new_center - canvas_rect.center()
        let old_center = canvas_rect.center() + self.pan_offset;
        let new_center_x = anchor.x + (old_center.x - anchor.x) * actual_factor;
        let new_center_y = anchor.y + (old_center.y - anchor.y) * actual_factor;
        self.pan_offset = Vec2::new(
            new_center_x - canvas_rect.center().x,
            new_center_y - canvas_rect.center().y,
        );
    }

    /// Pan the viewport by a screen-space delta (used by the Pan tool)
    pub fn pan_by(&mut self, delta: Vec2) {
        self.pan_offset += delta;
    }

}

fn selection_mask_bounds(mask: &image::GrayImage) -> Option<(u32, u32, u32, u32)> {
    let (w, h) = mask.dimensions();
    let mut min_x = w;
    let mut min_y = h;
    let mut max_x = 0;
    let mut max_y = 0;
    for y in 0..h {
        for x in 0..w {
            if mask.get_pixel(x, y)[0] > 0 {
                min_x = min_x.min(x);
                min_y = min_y.min(y);
                max_x = max_x.max(x + 1);
                max_y = max_y.max(y + 1);
            }
        }
    }
    (min_x < max_x && min_y < max_y).then_some((min_x, min_y, max_x, max_y))
}

