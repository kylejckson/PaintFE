impl PaintFEApp {
    fn process_canvas_and_transform_dialog(&mut self, ctx: &egui::Context, dialog: &mut ActiveDialog) -> bool {
        let matched = matches!(dialog, ActiveDialog::None | ActiveDialog::ResizeImage(_) | ActiveDialog::ResizeCanvas(_) | ActiveDialog::AlignLayer(_) | ActiveDialog::GaussianBlur(_) | ActiveDialog::LayerTransform(_));
        if !matched {
            return false;
        }

        match dialog {

            ActiveDialog::None => {}

            ActiveDialog::ResizeImage(dlg) => match dlg.show(ctx) {
                DialogResult::Ok((w, h, options)) => {
                    self.settings.persist_resize_lock_aspect = dlg.lock_aspect;
                    self.active_dialog = ActiveDialog::None;
                    if let Some(project) = self.active_project_mut() {
                        // Rasterize text layers before extracting pixels for resize
                        project.canvas_state.ensure_all_text_layers_rasterized();
                        for layer in &mut project.canvas_state.layers {
                            if layer.is_text_layer() {
                                layer.content = crate::canvas::LayerContent::Raster;
                            }
                        }
                        let before = CanvasSnapshot::capture(&project.canvas_state);
                        let flat_layers: Vec<RgbaImage> = project
                            .canvas_state
                            .layers
                            .iter()
                            .map(|l| l.pixels.to_rgba_image())
                            .collect();
                        let protection_mask = if options.method
                            == crate::ops::transform::ResizeMethod::PixelArtRetarget
                            && options.retarget.use_selection_as_protection
                        {
                            project.canvas_state.selection_mask.clone()
                        } else {
                            None
                        };
                        let sender = self.canvas_op_sender.clone();
                        let project_index = self.active_project_index;
                        let current_time = ctx.input(|i| i.time);
                        if self.pending_filter_jobs == 0 {
                            self.filter_ops_start_time = Some(current_time);
                        }
                        self.filter_status_description = "Resize Image".to_string();
                        self.pending_filter_jobs += 1;
                        crate::par_compat::spawn(move || {
                            let result_layers = crate::ops::transform::resize_layers_with_options(
                                flat_layers,
                                w,
                                h,
                                options,
                                protection_mask,
                            );
                            let _ = sender.send(CanvasOpResult {
                                project_index,
                                before,
                                result_layers,
                                new_width: w,
                                new_height: h,
                                description: "Resize Image".to_string(),
                            });
                        });
                    }
                    return true;
                }
                DialogResult::Cancel => {
                    self.settings.persist_resize_lock_aspect = dlg.lock_aspect;
                    self.active_dialog = ActiveDialog::None;
                    return true;
                }
                _ => {}
            },

            ActiveDialog::ResizeCanvas(dlg) => {
                let secondary = self.colors_panel.get_secondary_color_f32();
                match dlg.show(ctx, secondary) {
                    DialogResult::Ok((w, h, anchor, fill)) => {
                        self.settings.persist_resize_lock_aspect = dlg.lock_aspect;
                        self.active_dialog = ActiveDialog::None;
                        let fill = if dlg.fill_transparent {
                            image::Rgba([0, 0, 0, 0])
                        } else {
                            fill
                        };
                        if let Some(project) = self.active_project_mut() {
                            // Rasterize text layers before extracting pixels for canvas resize
                            project.canvas_state.ensure_all_text_layers_rasterized();
                            for layer in &mut project.canvas_state.layers {
                                if layer.is_text_layer() {
                                    layer.content = crate::canvas::LayerContent::Raster;
                                }
                            }
                            let before = CanvasSnapshot::capture(&project.canvas_state);
                            let old_w = project.canvas_state.width;
                            let old_h = project.canvas_state.height;
                            let flat_layers: Vec<RgbaImage> = project
                                .canvas_state
                                .layers
                                .iter()
                                .map(|l| l.pixels.to_rgba_image())
                                .collect();
                            let sender = self.canvas_op_sender.clone();
                            let project_index = self.active_project_index;
                            let current_time = ctx.input(|i| i.time);
                            if self.pending_filter_jobs == 0 {
                                self.filter_ops_start_time = Some(current_time);
                            }
                            self.filter_status_description = "Resize Canvas".to_string();
                            self.pending_filter_jobs += 1;
                            crate::par_compat::spawn(move || {
                                let result_layers = crate::ops::transform::resize_canvas_layers(
                                    flat_layers,
                                    old_w,
                                    old_h,
                                    w,
                                    h,
                                    anchor,
                                    fill,
                                );
                                let _ = sender.send(CanvasOpResult {
                                    project_index,
                                    before,
                                    result_layers,
                                    new_width: w,
                                    new_height: h,
                                    description: "Resize Canvas".to_string(),
                                });
                            });
                        }
                        return true;
                    }
                    DialogResult::Cancel => {
                        self.settings.persist_resize_lock_aspect = dlg.lock_aspect;
                        self.active_dialog = ActiveDialog::None;
                        return true;
                    }
                    _ => {}
                }
            }

            ActiveDialog::AlignLayer(dlg) => match dlg.show(ctx) {
                DialogResult::Changed => {
                    let idx = dlg.layer_idx;
                    if let Some(flat) = &dlg.original_flat
                        && let Some(project) = self.active_project_mut()
                        && idx < project.canvas_state.layers.len()
                        && !project.canvas_state.layers[idx].is_text_layer()
                    {
                        let target_bounds = if dlg.align_to_selection {
                            project.canvas_state.selection_mask_bounds()
                        } else {
                            None
                        };
                        crate::ops::transform::align_layer_to_anchor_from_flat(
                            &mut project.canvas_state,
                            idx,
                            (dlg.anchor_x, dlg.anchor_y),
                            flat,
                            target_bounds,
                        );
                    }
                }
                DialogResult::Ok((_ax, _ay, _selection_target)) => {
                    self.active_dialog = ActiveDialog::None;
                    if let Some(project) = self.active_project_mut() {
                        let idx = dlg.layer_idx;
                        if let Some(original) = &dlg.original_pixels
                            && idx < project.canvas_state.layers.len()
                        {
                            let aligned = project.canvas_state.layers[idx].pixels.clone();
                            project.canvas_state.layers[idx].pixels = original.clone();
                            let mut cmd = SingleLayerSnapshotCommand::new_for_layer(
                                "Align Layer".to_string(),
                                &project.canvas_state,
                                idx,
                            );
                            project.canvas_state.layers[idx].pixels = aligned;
                            cmd.set_after(&project.canvas_state);
                            project.history.push(Box::new(cmd));
                        }
                        project.mark_dirty();
                    }
                    return true;
                }
                DialogResult::Cancel => {
                    self.filter_cancel
                        .store(true, std::sync::atomic::Ordering::Relaxed);
                    let idx = dlg.layer_idx;
                    if let Some(original) = &dlg.original_pixels
                        && let Some(project) = self.active_project_mut()
                    {
                        if let Some(layer) = project.canvas_state.layers.get_mut(idx) {
                            layer.pixels = original.clone();
                        }
                        project.canvas_state.mark_dirty(None);
                    }
                    self.active_dialog = ActiveDialog::None;
                    return true;
                }
                _ => {}
            },

            ActiveDialog::GaussianBlur(dlg) => {
                match dlg.show(ctx) {
                    DialogResult::Changed => {
                        // Spawn async preview job (token-tracked, interrupts stale previews)
                        let idx = dlg.layer_idx;
                        if let (Some(original), Some(flat)) =
                            (&dlg.original_pixels, &dlg.original_flat)
                        {
                            let sigma = dlg.sigma;
                            let sel_mask = self
                                .active_project()
                                .and_then(|p| p.canvas_state.selection_mask.clone());
                            self.spawn_preview_job(
                                ctx.input(|i| i.time),
                                "Gaussian Blur".to_string(),
                                idx,
                                original.clone(),
                                flat.clone(),
                                move |img| {
                                    crate::ops::filters::blur_with_selection_pub(
                                        img,
                                        sigma,
                                        sel_mask.as_ref(),
                                    )
                                },
                            );
                        }
                    }
                    DialogResult::Ok(_sigma) => {
                        self.preview_job_token = self.preview_job_token.wrapping_add(1);
                        // Spawn full-resolution blur on background thread.
                        self.filter_cancel
                            .store(true, std::sync::atomic::Ordering::Relaxed);
                        let sigma = dlg.sigma;
                        let idx = dlg.layer_idx;
                        self.active_dialog = ActiveDialog::None;
                        if let (Some(original_pixels), Some(original_flat)) =
                            (&dlg.original_pixels, &dlg.original_flat)
                        {
                            // Restore originals so the layer isn't left with preview data
                            if let Some(project) = self.active_project_mut()
                                && idx < project.canvas_state.layers.len()
                            {
                                project.canvas_state.layers[idx].pixels = original_pixels.clone();
                                project.canvas_state.mark_dirty(None);
                            }
                            let orig_clone = original_pixels.clone();
                            let flat_clone = original_flat.clone();
                            let sel_mask = self
                                .active_project()
                                .and_then(|p| p.canvas_state.selection_mask.clone());
                            self.spawn_filter_job(
                                ctx.input(|i| i.time),
                                "Gaussian Blur".to_string(),
                                idx,
                                orig_clone,
                                flat_clone,
                                move |flat| {
                                    crate::ops::filters::blur_with_selection_pub(
                                        flat,
                                        sigma,
                                        sel_mask.as_ref(),
                                    )
                                },
                            );
                        }
                        return true;
                    }
                    DialogResult::Cancel => {
                        self.preview_job_token = self.preview_job_token.wrapping_add(1);
                        // Restore original pixels.
                        let idx = dlg.layer_idx;
                        self.filter_cancel
                            .store(true, std::sync::atomic::Ordering::Relaxed);
                        if let Some(original) = &dlg.original_pixels
                            && let Some(project) = self.active_project_mut()
                        {
                            if let Some(layer) = project.canvas_state.layers.get_mut(idx) {
                                layer.pixels = original.clone();
                            }
                            project.canvas_state.mark_dirty(None);
                        }
                        self.active_dialog = ActiveDialog::None;
                        return true;
                    }
                    _ => {}
                }
            }

            ActiveDialog::LayerTransform(dlg) => {
                match dlg.show(ctx) {
                    DialogResult::Changed => {
                        // Live preview: use pre-flattened original to skip
                        // the expensive clone -> flatten round-trip.
                        let rz = dlg.rotation_z;
                        let rx = dlg.rotation_x;
                        let ry = dlg.rotation_y;
                        let scale = dlg.scale_percent / 100.0;
                        let offset = (dlg.offset_x, dlg.offset_y);
                        let idx = dlg.layer_idx;
                        if let Some(flat) = &dlg.original_flat
                            && let Some(project) = self.active_project_mut()
                        {
                            crate::ops::transform::affine_transform_layer_from_flat(
                                &mut project.canvas_state,
                                idx,
                                rz,
                                rx,
                                ry,
                                scale,
                                offset,
                                flat,
                            );
                        }
                    }
                    DialogResult::Ok((_rot_z, _rot_x, _rot_y, _scale, _offset)) => {
                        // Accept current preview - push undo.
                        self.active_dialog = ActiveDialog::None;
                        if let Some(project) = self.active_project_mut() {
                            let idx = dlg.layer_idx;
                            if let Some(original) = &dlg.original_pixels
                                && idx < project.canvas_state.layers.len()
                            {
                                let transformed = project.canvas_state.layers[idx].pixels.clone();
                                project.canvas_state.layers[idx].pixels = original.clone();
                                let mut cmd = SingleLayerSnapshotCommand::new_for_layer(
                                    "Layer Transform".to_string(),
                                    &project.canvas_state,
                                    idx,
                                );
                                project.canvas_state.layers[idx].pixels = transformed;
                                cmd.set_after(&project.canvas_state);
                                project.history.push(Box::new(cmd));
                            }
                            project.mark_dirty();
                        }
                        return true;
                    }
                    DialogResult::Cancel => {
                        // Restore original pixels.
                        self.filter_cancel
                            .store(true, std::sync::atomic::Ordering::Relaxed);
                        let idx = dlg.layer_idx;
                        if let Some(original) = &dlg.original_pixels
                            && let Some(project) = self.active_project_mut()
                        {
                            if let Some(layer) = project.canvas_state.layers.get_mut(idx) {
                                layer.pixels = original.clone();
                            }
                            project.canvas_state.mark_dirty(None);
                        }
                        self.active_dialog = ActiveDialog::None;
                        return true;
                    }
                    _ => {}
                }
            }

            // ================================================================
            // BRIGHTNESS / CONTRAST
            // ================================================================

            _ => unreachable!(),
        }

        self.active_dialog = std::mem::replace(dialog, ActiveDialog::None);
        true
    }
}


