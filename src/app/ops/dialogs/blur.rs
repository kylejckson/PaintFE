impl PaintFEApp {
    fn process_blur_dialog(&mut self, ctx: &egui::Context, dialog: &mut ActiveDialog) -> bool {
        let matched = matches!(dialog, ActiveDialog::BokehBlur(_) | ActiveDialog::MotionBlur(_) | ActiveDialog::BoxBlur(_) | ActiveDialog::ZoomBlur(_));
        if !matched {
            return false;
        }

        match dialog {

            ActiveDialog::BokehBlur(dlg) => {
                match dlg.show(ctx) {
                    DialogResult::Changed => {
                        // Spawn async preview job (token-tracked, interrupts stale previews)
                        dlg.poll_flat();
                        let idx = dlg.layer_idx;
                        if let (Some(original), Some(flat)) =
                            (&dlg.original_pixels, &dlg.original_flat)
                        {
                            let radius = dlg.radius;
                            let selection_mask = self
                                .active_project()
                                .and_then(|p| p.canvas_state.selection_mask.clone());
                            self.spawn_preview_job(
                                ctx.input(|i| i.time),
                                "Bokeh Blur".to_string(),
                                idx,
                                original.clone(),
                                flat.clone(),
                                move |img| {
                                    crate::ops::effects::bokeh_blur_core(
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
                        // Apply at full resolution
                        let idx = dlg.layer_idx;
                        if let Some(flat) = &dlg.original_flat {
                            let radius = dlg.radius;
                            let selection_mask = self
                                .active_project()
                                .and_then(|p| p.canvas_state.selection_mask.clone());
                            if let Some(project) = self.active_project_mut() {
                                Self::apply_fullres_effect(
                                    &mut project.canvas_state,
                                    idx,
                                    flat,
                                    |img| {
                                        crate::ops::effects::bokeh_blur_core(
                                            img,
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
                                    "Bokeh Blur".to_string(),
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
                    }
                }
            }

            ActiveDialog::MotionBlur(dlg) => {
                match dlg.show(ctx) {
                    DialogResult::Changed => {
                        // Spawn async preview job (token-tracked, interrupts stale previews)
                        dlg.poll_flat();
                        let idx = dlg.layer_idx;
                        if let (Some(original), Some(flat)) =
                            (&dlg.original_pixels, &dlg.original_flat)
                        {
                            let (angle, distance) = (dlg.angle, dlg.distance);
                            let selection_mask = self
                                .active_project()
                                .and_then(|p| p.canvas_state.selection_mask.clone());
                            self.spawn_preview_job(
                                ctx.input(|i| i.time),
                                "Motion Blur".to_string(),
                                idx,
                                original.clone(),
                                flat.clone(),
                                move |img| {
                                    crate::ops::effects::motion_blur_core(
                                        img,
                                        angle,
                                        distance,
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
                            let (angle, distance) = (dlg.angle, dlg.distance);
                            let selection_mask = self
                                .active_project()
                                .and_then(|p| p.canvas_state.selection_mask.clone());
                            if let Some(project) = self.active_project_mut() {
                                Self::apply_fullres_effect(
                                    &mut project.canvas_state,
                                    idx,
                                    flat,
                                    |img| {
                                    crate::ops::effects::motion_blur_core(
                                            img,
                                            angle,
                                            distance,
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
                                    "Motion Blur".to_string(),
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
                    }
                }
            }

            ActiveDialog::BoxBlur(dlg) => {
                match dlg.show(ctx) {
                    DialogResult::Changed => {
                        // Spawn async preview job (token-tracked, interrupts stale previews)
                        dlg.poll_flat();
                        let idx = dlg.layer_idx;
                        if let (Some(original), Some(flat)) =
                            (&dlg.original_pixels, &dlg.original_flat)
                        {
                            let radius = dlg.radius;
                            let selection_mask = self
                                .active_project()
                                .and_then(|p| p.canvas_state.selection_mask.clone());
                            self.spawn_preview_job(
                                ctx.input(|i| i.time),
                                "Box Blur".to_string(),
                                idx,
                                original.clone(),
                                flat.clone(),
                                move |img| {
                                    crate::ops::effects::box_blur_core(
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
                        let idx = dlg.layer_idx;
                        if let Some(flat) = &dlg.original_flat {
                            let radius = dlg.radius;
                            let selection_mask = self
                                .active_project()
                                .and_then(|p| p.canvas_state.selection_mask.clone());
                            if let Some(project) = self.active_project_mut() {
                                Self::apply_fullres_effect(
                                    &mut project.canvas_state,
                                    idx,
                                    flat,
                                    |img| {
                                        crate::ops::effects::box_blur_core(
                                            img,
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
                                    "Box Blur".to_string(),
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
                    }
                }
            }

            ActiveDialog::ZoomBlur(dlg) => match dlg.show(ctx) {
                DialogResult::Changed => {
                    dlg.poll_flat();
                    let idx = dlg.layer_idx;
                    if let (Some(original), Some(flat)) = (&dlg.original_pixels, &dlg.original_flat)
                    {
                        let (cx, cy, strength, samples, tint, ts) = dlg.current_params();
                        let selection_mask = self
                            .active_project()
                            .and_then(|p| p.canvas_state.selection_mask.clone());
                        self.spawn_preview_job(
                            ctx.input(|i| i.time),
                            "Zoom Blur".to_string(),
                            idx,
                            original.clone(),
                            flat.clone(),
                            move |img| {
                                crate::ops::effects::zoom_blur_core(
                                    img,
                                    cx,
                                    cy,
                                    strength,
                                    samples,
                                    tint,
                                    ts,
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
                        let (cx, cy, strength, samples, tint, ts) = dlg.current_params();
                        let selection_mask = self
                            .active_project()
                            .and_then(|p| p.canvas_state.selection_mask.clone());
                        if let Some(project) = self.active_project_mut() {
                            Self::apply_fullres_effect(
                                &mut project.canvas_state,
                                idx,
                                flat,
                                |img| {
                                    crate::ops::effects::zoom_blur_core(
                                        img,
                                        cx,
                                        cy,
                                        strength,
                                        samples,
                                        tint,
                                        ts,
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
                                "Zoom Blur".to_string(),
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
                }
            },


            _ => unreachable!(),
        }

        self.active_dialog = std::mem::replace(dialog, ActiveDialog::None);
        true
    }
}

