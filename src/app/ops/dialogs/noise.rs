impl PaintFEApp {
    fn process_noise_dialog(&mut self, ctx: &egui::Context, dialog: &mut ActiveDialog) -> bool {
        let matched = matches!(dialog, ActiveDialog::AddNoise(_) | ActiveDialog::ReduceNoise(_) | ActiveDialog::Median(_));
        if !matched {
            return false;
        }

        match dialog {

            ActiveDialog::AddNoise(dlg) => match dlg.show(ctx) {
                DialogResult::Changed => {
                    dlg.first_open = false;
                    let idx = dlg.layer_idx;
                    if let (Some(original), Some(flat)) = (&dlg.original_pixels, &dlg.original_flat)
                    {
                        let amount = dlg.amount;
                        let noise_type = dlg.noise_type();
                        let monochrome = dlg.monochrome;
                        let seed = dlg.seed;
                        let noise_scale = dlg.scale;
                        let octaves = dlg.octaves as u32;
                        let selection_mask = self
                            .active_project()
                            .and_then(|p| p.canvas_state.selection_mask.clone());
                        self.spawn_preview_job(
                            ctx.input(|i| i.time),
                            "Add Noise".to_string(),
                            idx,
                            original.clone(),
                            flat.clone(),
                            move |img| {
                                crate::ops::effects::add_noise_core(
                                    img,
                                    amount,
                                    noise_type,
                                    monochrome,
                                    seed,
                                    noise_scale,
                                    octaves,
                                    selection_mask.as_ref(),
                                )
                            },
                        );
                    }
                }
                DialogResult::Ok(_) => {
                    self.preview_job_token = self.preview_job_token.wrapping_add(1);
                    let idx = dlg.layer_idx;
                    if let Some(flat) = &dlg.original_flat {
                        let amount = dlg.amount;
                        let noise_type = dlg.noise_type();
                        let monochrome = dlg.monochrome;
                        let seed = dlg.seed;
                        let noise_scale = dlg.scale;
                        let octaves = dlg.octaves as u32;
                        let selection_mask = self
                            .active_project()
                            .and_then(|p| p.canvas_state.selection_mask.clone());
                        if let Some(project) = self.active_project_mut() {
                            Self::apply_fullres_effect(
                                &mut project.canvas_state,
                                idx,
                                flat,
                                |img| {
                                    crate::ops::effects::add_noise_core(
                                        img,
                                        amount,
                                        noise_type,
                                        monochrome,
                                        seed,
                                        noise_scale,
                                        octaves,
                                        selection_mask.as_ref(),
                                    )
                                },
                            );
                        }
                    }
                    self.active_dialog = ActiveDialog::None;
                    if let Some(project) = self.active_project_mut() {
                        let idx = dlg.layer_idx;
                        if let Some(original) = &dlg.original_pixels
                            && idx < project.canvas_state.layers.len()
                        {
                            let adjusted = project.canvas_state.layers[idx].pixels.clone();
                            project.canvas_state.layers[idx].pixels = original.clone();
                            let mut cmd = SingleLayerSnapshotCommand::new_for_layer(
                                "Add Noise".to_string(),
                                &project.canvas_state,
                                idx,
                            );
                            project.canvas_state.layers[idx].pixels = adjusted;
                            cmd.set_after(&project.canvas_state);
                            project.history.push(Box::new(cmd));
                        }
                        project.mark_dirty();
                    }
                    return true;
                }
                DialogResult::Cancel => {
                    self.preview_job_token = self.preview_job_token.wrapping_add(1);
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
                _ => {
                    dlg.poll_flat();
                    if dlg.first_open && dlg.live_preview && dlg.original_flat.is_some() {
                        dlg.first_open = false;
                        let idx = dlg.layer_idx;
                        if let (Some(original), Some(flat)) =
                            (&dlg.original_pixels, &dlg.original_flat)
                        {
                            let amount = dlg.amount;
                            let noise_type = dlg.noise_type();
                            let monochrome = dlg.monochrome;
                            let seed = dlg.seed;
                            let noise_scale = dlg.scale;
                            let octaves = dlg.octaves as u32;
                            let selection_mask = self
                                .active_project()
                                .and_then(|p| p.canvas_state.selection_mask.clone());
                            self.spawn_preview_job(
                                ctx.input(|i| i.time),
                                "Add Noise".to_string(),
                                idx,
                                original.clone(),
                                flat.clone(),
                                move |img| {
                                    crate::ops::effects::add_noise_core(
                                        img,
                                        amount,
                                        noise_type,
                                        monochrome,
                                        seed,
                                        noise_scale,
                                        octaves,
                                        selection_mask.as_ref(),
                                    )
                                },
                            );
                        }
                    }
                }
            },

            ActiveDialog::ReduceNoise(dlg) => match dlg.show(ctx) {
                DialogResult::Changed => {
                    dlg.first_open = false;
                    let idx = dlg.layer_idx;
                    if let (Some(original), Some(flat)) = (&dlg.original_pixels, &dlg.original_flat)
                    {
                        let (strength, radius) = (dlg.strength, dlg.radius as u32);
                        let selection_mask = self
                            .active_project()
                            .and_then(|p| p.canvas_state.selection_mask.clone());
                        self.spawn_preview_job(
                            ctx.input(|i| i.time),
                            "Reduce Noise".to_string(),
                            idx,
                            original.clone(),
                            flat.clone(),
                            move |img| {
                                crate::ops::effects::reduce_noise_core(
                                    img,
                                    strength,
                                    radius,
                                    selection_mask.as_ref(),
                                )
                            },
                        );
                    }
                }
                DialogResult::Ok(_) => {
                    self.preview_job_token = self.preview_job_token.wrapping_add(1);
                    let idx = dlg.layer_idx;
                    if let Some(flat) = &dlg.original_flat {
                        let strength = dlg.strength;
                        let radius = dlg.radius as u32;
                        let selection_mask = self
                            .active_project()
                            .and_then(|p| p.canvas_state.selection_mask.clone());
                        if let Some(project) = self.active_project_mut() {
                            Self::apply_fullres_effect(
                                &mut project.canvas_state,
                                idx,
                                flat,
                                |img| {
                                    crate::ops::effects::reduce_noise_core(
                                        img,
                                        strength,
                                        radius,
                                        selection_mask.as_ref(),
                                    )
                                },
                            );
                        }
                    }
                    self.active_dialog = ActiveDialog::None;
                    if let Some(project) = self.active_project_mut() {
                        let idx = dlg.layer_idx;
                        if let Some(original) = &dlg.original_pixels
                            && idx < project.canvas_state.layers.len()
                        {
                            let adjusted = project.canvas_state.layers[idx].pixels.clone();
                            project.canvas_state.layers[idx].pixels = original.clone();
                            let mut cmd = SingleLayerSnapshotCommand::new_for_layer(
                                "Reduce Noise".to_string(),
                                &project.canvas_state,
                                idx,
                            );
                            project.canvas_state.layers[idx].pixels = adjusted;
                            cmd.set_after(&project.canvas_state);
                            project.history.push(Box::new(cmd));
                        }
                        project.mark_dirty();
                    }
                    return true;
                }
                DialogResult::Cancel => {
                    self.preview_job_token = self.preview_job_token.wrapping_add(1);
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
                _ => {
                    dlg.poll_flat();
                    if dlg.first_open && dlg.live_preview && dlg.original_flat.is_some() {
                        dlg.first_open = false;
                        let idx = dlg.layer_idx;
                        if let (Some(original), Some(flat)) =
                            (&dlg.original_pixels, &dlg.original_flat)
                        {
                            let (strength, radius) = (dlg.strength, dlg.radius as u32);
                            let selection_mask = self
                                .active_project()
                                .and_then(|p| p.canvas_state.selection_mask.clone());
                            self.spawn_preview_job(
                                ctx.input(|i| i.time),
                                "Reduce Noise".to_string(),
                                idx,
                                original.clone(),
                                flat.clone(),
                                move |img| {
                                    crate::ops::effects::reduce_noise_core(
                                        img,
                                        strength,
                                        radius,
                                        selection_mask.as_ref(),
                                    )
                                },
                            );
                        }
                    }
                }
            },

            ActiveDialog::Median(dlg) => {
                match dlg.show(ctx) {
                    DialogResult::Changed => {
                        // Spawn async preview job (token-tracked, interrupts stale previews)
                        let idx = dlg.layer_idx;
                        if let (Some(original), Some(flat)) =
                            (&dlg.original_pixels, &dlg.original_flat)
                        {
                            let radius = dlg.radius as u32;
                            let selection_mask = self
                                .active_project()
                                .and_then(|p| p.canvas_state.selection_mask.clone());
                            self.spawn_preview_job(
                                ctx.input(|i| i.time),
                                "Median Filter".to_string(),
                                idx,
                                original.clone(),
                                flat.clone(),
                                move |img| {
                                    crate::ops::effects::median_core(
                                        img,
                                        radius,
                                        selection_mask.as_ref(),
                                    )
                                },
                            );
                        }
                    }
                    DialogResult::Ok(_) => {
                        self.preview_job_token = self.preview_job_token.wrapping_add(1);
                        // Use GPU-accelerated median at full resolution
                        let idx = dlg.layer_idx;
                        let radius = dlg.radius as u32;
                        if let Some(flat) = &dlg.original_flat {
                            let active_idx = self.active_project_index;
                            if let Some(project) = self.projects.get_mut(active_idx) {
                                crate::ops::effects::median_filter_from_flat_gpu(
                                    &mut project.canvas_state,
                                    idx,
                                    radius,
                                    flat,
                                    &self.canvas.gpu_renderer,
                                );
                            }
                        }
                        self.active_dialog = ActiveDialog::None;
                        if let Some(project) = self.active_project_mut() {
                            let idx = dlg.layer_idx;
                            if let Some(original) = &dlg.original_pixels
                                && idx < project.canvas_state.layers.len()
                            {
                                let adjusted = project.canvas_state.layers[idx].pixels.clone();
                                project.canvas_state.layers[idx].pixels = original.clone();
                                let mut cmd = SingleLayerSnapshotCommand::new_for_layer(
                                    "Median Filter".to_string(),
                                    &project.canvas_state,
                                    idx,
                                );
                                project.canvas_state.layers[idx].pixels = adjusted;
                                cmd.set_after(&project.canvas_state);
                                project.history.push(Box::new(cmd));
                            }
                            project.mark_dirty();
                        }
                        return true;
                    }
                    DialogResult::Cancel => {
                        self.preview_job_token = self.preview_job_token.wrapping_add(1);
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


            _ => unreachable!(),
        }

        self.active_dialog = std::mem::replace(dialog, ActiveDialog::None);
        true
    }
}

