impl PaintFEApp {
    /// Spawn a filter job on a background thread.
    ///
    /// `description`: undo entry label (e.g. "Gaussian Blur").
    /// `layer_idx`: which layer to operate on.
    /// `original_pixels`: clone of layer pixels before the filter.
    /// `original_flat`: pre-flattened RGBA for the filter function.
    /// `filter_fn`: closure that takes the flat image and returns the processed image.
    ///
    /// The closure runs on `rayon::spawn`; when done it sends a `FilterResult`
    /// back via the channel.  The main thread polls the channel in `update()`.
    fn spawn_filter_job(
        &mut self,
        current_time: f64,
        description: String,
        layer_idx: usize,
        original_pixels: TiledImage,
        original_flat: image::RgbaImage,
        filter_fn: impl FnOnce(&image::RgbaImage) -> image::RgbaImage + Send + 'static,
    ) {
        self.spawn_filter_job_internal(
            current_time,
            description,
            layer_idx,
            original_pixels,
            original_flat,
            0,
            None,
            filter_fn,
        );
    }

    /// Spawn a live-preview filter job. The token is incremented so any in-flight
    /// job from a previous slider position is automatically discarded on arrival.
    fn spawn_preview_job(
        &mut self,
        current_time: f64,
        description: String,
        layer_idx: usize,
        original_pixels: TiledImage,
        original_flat: image::RgbaImage,
        filter_fn: impl FnOnce(&image::RgbaImage) -> image::RgbaImage + Send + 'static,
    ) {
        // Cancel any in-flight preview job before spawning the new one
        self.filter_cancel
            .store(true, std::sync::atomic::Ordering::Relaxed);
        self.filter_cancel = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let cancel = std::sync::Arc::clone(&self.filter_cancel);
        self.preview_job_token = self.preview_job_token.wrapping_add(1);
        let token = self.preview_job_token;
        self.spawn_filter_job_internal(
            current_time,
            description,
            layer_idx,
            original_pixels,
            original_flat,
            token,
            Some(cancel),
            filter_fn,
        );
    }

    fn spawn_filter_job_with_token(
        &mut self,
        current_time: f64,
        description: String,
        layer_idx: usize,
        original_pixels: TiledImage,
        original_flat: image::RgbaImage,
        preview_token: u64,
        filter_fn: impl FnOnce(&image::RgbaImage) -> image::RgbaImage + Send + 'static,
    ) {
        self.spawn_filter_job_internal(
            current_time,
            description,
            layer_idx,
            original_pixels,
            original_flat,
            preview_token,
            None,
            filter_fn,
        );
    }

    fn spawn_filter_job_internal(
        &mut self,
        current_time: f64,
        description: String,
        layer_idx: usize,
        original_pixels: TiledImage,
        original_flat: image::RgbaImage,
        preview_token: u64,
        cancel: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
        filter_fn: impl FnOnce(&image::RgbaImage) -> image::RgbaImage + Send + 'static,
    ) {
        let sender = self.filter_sender.clone();
        let project_index = self.active_project_index;
        if self.pending_filter_jobs == 0 {
            self.filter_ops_start_time = Some(current_time);
        }
        self.filter_status_description = description.clone();
        self.pending_filter_jobs += 1;
        crate::par_compat::spawn(move || {
            let result_flat = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                // Check cancellation before starting expensive work
                if cancel
                    .as_ref()
                    .is_some_and(|c| c.load(std::sync::atomic::Ordering::Relaxed))
                {
                    return original_flat.clone();
                }
                filter_fn(&original_flat)
            }));
            match result_flat {
                Ok(processed) => {
                    let result_tiled = TiledImage::from_rgba_image(&processed);
                    let _ = sender.send(FilterResult {
                        project_index,
                        layer_idx,
                        original_pixels,
                        result_pixels: result_tiled,
                        description,
                        preview_token,
                    });
                }
                Err(panic_info) => {
                    // Panic in filter — revert to original (no-op: don't send)
                    let msg = if let Some(s) = panic_info.downcast_ref::<&str>() {
                        s.to_string()
                    } else if let Some(s) = panic_info.downcast_ref::<String>() {
                        s.to_string()
                    } else {
                        "unknown panic payload".to_string()
                    };
                    eprintln!("Filter '{}' panicked: {}", description, msg);
                    let _ = sender.send(FilterResult {
                        project_index,
                        layer_idx,
                        original_pixels: original_pixels.clone(),
                        result_pixels: original_pixels,
                        description,
                        preview_token,
                    });
                }
            }
        });
    }

    /// Perform a canvas operation with full-snapshot undo.
    /// The closure receives `&mut CanvasState` and should apply the operation.
    fn do_snapshot_op(&mut self, description: &str, op: impl FnOnce(&mut CanvasState)) {
        if let Some(project) = self.active_project_mut() {
            // Rasterize all text layers before any canvas-wide destructive op
            project.canvas_state.ensure_all_text_layers_rasterized();
            for layer in &mut project.canvas_state.layers {
                if layer.is_text_layer() {
                    layer.content = crate::canvas::LayerContent::Raster;
                }
            }
            let mut cmd = SnapshotCommand::new(description.to_string(), &project.canvas_state);
            op(&mut project.canvas_state);
            cmd.set_after(&project.canvas_state);
            project.history.push(Box::new(cmd));
            project.mark_dirty();
        }
    }

    fn cut_selection_with_history(&mut self, transparent_cutout: bool) -> bool {
        let Some(project) = self.active_project_mut() else {
            return false;
        };
        let Some(mut command) =
            crate::components::history::CutSelectionCommand::new(&project.canvas_state)
        else {
            return false;
        };

        let idx = project.canvas_state.active_layer_index;
        if project
            .canvas_state
            .layers
            .get(idx)
            .is_some_and(|layer| layer.is_text_layer())
        {
            project.canvas_state.force_rasterize_text_layer(idx);
            if let Some(layer) = project.canvas_state.layers.get_mut(idx) {
                layer.content = crate::canvas::LayerContent::Raster;
            }
        }

        if !crate::ops::clipboard::cut_selection(&mut project.canvas_state, transparent_cutout) {
            return false;
        }
        command.set_after(&project.canvas_state);
        project.history.push(Box::new(command));
        project.mark_dirty();
        true
    }

    /// Commit the active paste overlay.
    fn commit_paste_overlay(&mut self) {
        if let Some(overlay) = self.paste_overlay.take() {
            self.paste_transform_undo.clear();
            self.paste_transform_redo.clear();
            let desc = if self.is_move_pixels_active {
                "Move Pixels"
            } else {
                "Paste"
            };
            let move_before = if self.is_move_pixels_active {
                self.move_pixels_before.take()
            } else {
                None
            };
            if let Some(project) = self.active_project_mut() {
                let before = move_before.unwrap_or_else(|| {
                    crate::components::history::CanvasSnapshot::capture(&project.canvas_state)
                });
                project.canvas_state.clear_preview_state();
                overlay.commit(&mut project.canvas_state);
                let after =
                    crate::components::history::CanvasSnapshot::capture(&project.canvas_state);
                project.history.push(Box::new(
                    crate::components::history::SnapshotCommand::from_snapshots(
                        desc.to_string(),
                        before,
                        after,
                    ),
                ));
                project.mark_dirty();
            }
            self.is_move_pixels_active = false;
        }
    }

    /// Cancel the active paste overlay.
    /// If MovePixels is active, restore the pre-extraction snapshot without pushing history.
    fn cancel_paste_overlay(&mut self) {
        self.paste_overlay = None;
        self.paste_transform_undo.clear();
        self.paste_transform_redo.clear();
        if self.is_move_pixels_active {
            let before = self.move_pixels_before.take();
            if let Some(project) = self.active_project_mut() {
                if let Some(before) = before {
                    before.restore_into(&mut project.canvas_state);
                }
                project.canvas_state.clear_preview_state();
            }
            self.is_move_pixels_active = false;
        } else if let Some(project) = self.active_project_mut() {
            project.canvas_state.clear_preview_state();
            project.canvas_state.mark_dirty(None);
        }
    }

    fn commit_pending_tool_history(&mut self) {
        let pending_cmds: Vec<_> = self
            .tools_panel
            .pending_history_commands
            .drain(..)
            .collect();
        for cmd in pending_cmds {
            if let Some(project) = self.active_project_mut() {
                project.history.push(cmd);
                project.mark_dirty();
            }
        }

        if let Some(stroke_event) = self.tools_panel.take_stroke_event()
            && let Some(project) = self.active_project_mut()
        {
            match stroke_event.target {
                crate::components::tools::StrokeTarget::LayerPixels => {
                    let after_patch = history::PixelPatch::capture(
                        &project.canvas_state,
                        stroke_event.layer_index,
                        stroke_event.bounds.expand(10.0),
                    );

                    if let Some(before_patch) = stroke_event.before_snapshot {
                        let command = history::BrushCommand::new(
                            stroke_event.description,
                            before_patch,
                            after_patch,
                        );
                        project.history.push(Box::new(command));
                        project.mark_dirty();
                    }
                }
                crate::components::tools::StrokeTarget::LayerMask => {
                    if let Some(layer) = project.canvas_state.layers.get(stroke_event.layer_index) {
                        let command = history::LayerMaskCommand::new(
                            stroke_event.description,
                            stroke_event.layer_index,
                            stroke_event.before_mask,
                            layer.mask.clone(),
                            stroke_event.before_mask_enabled,
                            layer.mask_enabled,
                        );
                        project.history.push(Box::new(command));
                        project.mark_dirty();
                    }
                }
            }
        }
    }

    /// Same as do_snapshot_op, but only captures the active layer (not all layers).
    /// Much more memory-efficient for single-layer operations.
    fn do_layer_snapshot_op(&mut self, description: &str, op: impl FnOnce(&mut CanvasState)) {
        if let Some(project) = self.active_project_mut() {
            // Auto-rasterize the active text layer before any destructive single-layer op
            let idx = project.canvas_state.active_layer_index;
            if idx < project.canvas_state.layers.len()
                && project.canvas_state.layers[idx].is_text_layer()
            {
                project.canvas_state.ensure_all_text_layers_rasterized();
                project.canvas_state.layers[idx].content = crate::canvas::LayerContent::Raster;
            }
            let mut cmd =
                SingleLayerSnapshotCommand::new(description.to_string(), &project.canvas_state);
            op(&mut project.canvas_state);
            cmd.set_after(&project.canvas_state);
            project.history.push(Box::new(cmd));
            project.mark_dirty();
        }
    }

    fn select_all_canvas(&mut self) {
        let secondary = self.colors_panel.get_secondary_color_f32();
        let project_idx = self.active_project_index;

        if self.paste_overlay.is_some() {
            self.commit_paste_overlay();
        }

        let tools_panel = &mut self.tools_panel;
        if let Some(project) = self.projects.get_mut(project_idx) {
            tools_panel.commit_active_tool_preview(&mut project.canvas_state, secondary);

            let w = project.canvas_state.width;
            let h = project.canvas_state.height;
            let sel_before = project.canvas_state.selection_mask.clone();
            let sel_before_all = project.canvas_state.selection_all;

            if sel_before_all {
                return;
            }

            project.canvas_state.selection_mask = None;
            project.canvas_state.selection_all = true;
            project.canvas_state.invalidate_selection_overlay();
            project.canvas_state.selection_overlay_bounds =
                (w > 0 && h > 0).then_some((0, 0, w - 1, h - 1));
            project.history.push(Box::new(SelectionCommand::new_all(
                "Select All",
                sel_before,
                sel_before_all,
            )));
        }
    }

    /// Like `do_layer_snapshot_op`, but splits the borrow so the closure can also
    /// access the GPU renderer for compute-shader operations.
    fn do_gpu_snapshot_op(
        &mut self,
        description: &str,
        op: impl FnOnce(&mut CanvasState, &crate::gpu::GpuRenderer),
    ) {
        let active_idx = self.active_project_index;
        if let Some(project) = self.projects.get_mut(active_idx) {
            // Auto-rasterize the active text layer before any destructive GPU op
            let idx = project.canvas_state.active_layer_index;
            if idx < project.canvas_state.layers.len()
                && project.canvas_state.layers[idx].is_text_layer()
            {
                project.canvas_state.ensure_all_text_layers_rasterized();
                project.canvas_state.layers[idx].content = crate::canvas::LayerContent::Raster;
            }
            let mut cmd =
                SingleLayerSnapshotCommand::new(description.to_string(), &project.canvas_state);
            op(&mut project.canvas_state, &self.canvas.gpu_renderer);
            cmd.set_after(&project.canvas_state);
            project.history.push(Box::new(cmd));
            project.mark_dirty();
        }
    }

    /// Downscale an image for fast low-res live preview.
    /// Returns (downscaled_image, scale_factor).  If the image is already small
    /// enough the original is returned with factor 1.0.
    fn make_preview_image(flat: &RgbaImage, max_edge: u32) -> (RgbaImage, f32) {
        let (w, h) = (flat.width(), flat.height());
        let longest = w.max(h);
        if longest <= max_edge {
            return (flat.clone(), 1.0);
        }
        let scale = max_edge as f32 / longest as f32;
        let nw = ((w as f32 * scale).round() as u32).max(1);
        let nh = ((h as f32 * scale).round() as u32).max(1);
        let small = image::imageops::resize(flat, nw, nh, image::imageops::FilterType::Triangle);
        (small, scale)
    }

    /// Upscale a processed preview back to the original layer dimensions then
    /// write it into the layer for on-screen display.
    fn apply_preview_to_layer(
        state: &mut CanvasState,
        layer_idx: usize,
        preview: &RgbaImage,
        orig_w: u32,
        orig_h: u32,
    ) {
        if layer_idx >= state.layers.len() {
            return;
        }
        let upscaled = if preview.width() == orig_w && preview.height() == orig_h {
            preview.clone()
        } else {
            image::imageops::resize(
                preview,
                orig_w,
                orig_h,
                image::imageops::FilterType::Triangle,
            )
        };
        state.layers[layer_idx].pixels = TiledImage::from_rgba_image(&upscaled);
        state.mark_dirty(None);
    }

    /// Apply a low‑res effect preview: runs the effect on `preview_flat_small`,
    /// then upscales to the original layer dimensions.
    /// `effect_fn` receives the small image and returns the processed result.
    fn apply_lowres_preview<F>(
        state: &mut CanvasState,
        layer_idx: usize,
        preview_small: &RgbaImage,
        orig_flat: Option<&RgbaImage>,
        effect_fn: F,
    ) where
        F: FnOnce(&RgbaImage) -> RgbaImage,
    {
        let result = effect_fn(preview_small);
        let (orig_w, orig_h) = orig_flat
            .map(|f| (f.width(), f.height()))
            .unwrap_or((preview_small.width(), preview_small.height()));
        Self::apply_preview_to_layer(state, layer_idx, &result, orig_w, orig_h);
    }

    /// Compute a low-res preview (run effect on the small image then upscale)
    /// but do NOT write it to the layer.  Returns the upscaled result so the
    /// caller can drive progressive reveal.
    fn compute_lowres_preview<F>(
        preview_small: &RgbaImage,
        orig_flat: Option<&RgbaImage>,
        effect_fn: F,
    ) -> RgbaImage
    where
        F: FnOnce(&RgbaImage) -> RgbaImage,
    {
        let result = effect_fn(preview_small);
        let (orig_w, orig_h) = orig_flat
            .map(|f| (f.width(), f.height()))
            .unwrap_or((preview_small.width(), preview_small.height()));
        if result.width() == orig_w && result.height() == orig_h {
            result
        } else {
            image::imageops::resize(
                &result,
                orig_w,
                orig_h,
                image::imageops::FilterType::Triangle,
            )
        }
    }

    /// Apply an effect at full resolution and commit it to the layer.
    /// Used when the user clicks OK.
    fn apply_fullres_effect<F>(
        state: &mut CanvasState,
        layer_idx: usize,
        original_flat: &RgbaImage,
        effect_fn: F,
    ) where
        F: FnOnce(&RgbaImage) -> RgbaImage,
    {
        if layer_idx >= state.layers.len() {
            return;
        }
        // Convert text layer to raster so the effect isn't overwritten by re-rasterization
        if state.layers[layer_idx].is_text_layer() {
            state.layers[layer_idx].content = crate::canvas::LayerContent::Raster;
        }
        let result = effect_fn(original_flat);
        state.layers[layer_idx].pixels = TiledImage::from_rgba_image(&result);
        state.mark_dirty(None);
    }
}
