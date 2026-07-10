impl PaintFEApp {
    fn process_stylize_dialog(&mut self, ctx: &egui::Context, dialog: &mut ActiveDialog) -> bool {
        let matched = matches!(dialog, ActiveDialog::Glow(_) | ActiveDialog::Sharpen(_) | ActiveDialog::Vignette(_) | ActiveDialog::Halftone(_));
        if !matched {
            return false;
        }

        match dialog {

            ActiveDialog::Glow(dlg) => match dlg.show(ctx) {
                DialogResult::Changed => {
                    dlg.first_open = false;
                    let idx = dlg.layer_idx;
                    if let (Some(original), Some(flat)) = (&dlg.original_pixels, &dlg.original_flat)
                    {
                        let (radius, intensity) = (dlg.radius, dlg.intensity);
                        let selection_mask = self
                            .active_project()
                            .and_then(|p| p.canvas_state.selection_mask.clone());
                        self.spawn_preview_job(
                            ctx.input(|i| i.time),
                            "Glow".to_string(),
                            idx,
                            original.clone(),
                            flat.clone(),
                            move |img| {
                                crate::ops::effects::glow_core(
                                    img,
                                    radius,
                                    intensity,
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
                        let (radius, intensity) = (dlg.radius, dlg.intensity);
                        let selection_mask = self
                            .active_project()
                            .and_then(|p| p.canvas_state.selection_mask.clone());
                        if let Some(project) = self.active_project_mut() {
                            Self::apply_fullres_effect(
                                &mut project.canvas_state,
                                idx,
                                flat,
                                |img| {
                                    crate::ops::effects::glow_core(
                                        img,
                                        radius,
                                        intensity,
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
                                "Glow".to_string(),
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
                            let (radius, intensity) = (dlg.radius, dlg.intensity);
                            let selection_mask = self
                                .active_project()
                                .and_then(|p| p.canvas_state.selection_mask.clone());
                            self.spawn_preview_job(
                                ctx.input(|i| i.time),
                                "Glow".to_string(),
                                idx,
                                original.clone(),
                                flat.clone(),
                                move |img| {
                                    crate::ops::effects::glow_core(
                                        img,
                                        radius,
                                        intensity,
                                        selection_mask.as_ref(),
                                    )
                                },
                            );
                        }
                    }
                }
            },

            ActiveDialog::Sharpen(dlg) => match dlg.show(ctx) {
                DialogResult::Changed => {
                    dlg.first_open = false;
                    let idx = dlg.layer_idx;
                    if let (Some(original), Some(flat)) = (&dlg.original_pixels, &dlg.original_flat)
                    {
                        let (amount, radius) = (dlg.amount, dlg.radius);
                        let selection_mask = self
                            .active_project()
                            .and_then(|p| p.canvas_state.selection_mask.clone());
                        self.spawn_preview_job(
                            ctx.input(|i| i.time),
                            "Sharpen".to_string(),
                            idx,
                            original.clone(),
                            flat.clone(),
                            move |img| {
                                crate::ops::effects::sharpen_core(
                                    img,
                                    amount,
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
                        let (amount, radius) = (dlg.amount, dlg.radius);
                        let selection_mask = self
                            .active_project()
                            .and_then(|p| p.canvas_state.selection_mask.clone());
                        if let Some(project) = self.active_project_mut() {
                            Self::apply_fullres_effect(
                                &mut project.canvas_state,
                                idx,
                                flat,
                                |img| {
                                    crate::ops::effects::sharpen_core(
                                        img,
                                        amount,
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
                                "Sharpen".to_string(),
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
                            let (amount, radius) = (dlg.amount, dlg.radius);
                            let selection_mask = self
                                .active_project()
                                .and_then(|p| p.canvas_state.selection_mask.clone());
                            self.spawn_preview_job(
                                ctx.input(|i| i.time),
                                "Sharpen".to_string(),
                                idx,
                                original.clone(),
                                flat.clone(),
                                move |img| {
                                    crate::ops::effects::sharpen_core(
                                        img,
                                        amount,
                                        radius,
                                        selection_mask.as_ref(),
                                    )
                                },
                            );
                        }
                    }
                }
            },

            ActiveDialog::Vignette(dlg) => match dlg.show(ctx) {
                DialogResult::Changed => {
                    dlg.first_open = false;
                    let idx = dlg.layer_idx;
                    if let (Some(original), Some(flat)) = (&dlg.original_pixels, &dlg.original_flat)
                    {
                        let (amount, softness) = (dlg.amount, dlg.softness);
                        let selection_mask = self
                            .active_project()
                            .and_then(|p| p.canvas_state.selection_mask.clone());
                        self.spawn_preview_job(
                            ctx.input(|i| i.time),
                            "Vignette".to_string(),
                            idx,
                            original.clone(),
                            flat.clone(),
                            move |img| {
                                crate::ops::effects::vignette_core(
                                    img,
                                    amount,
                                    softness,
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
                        let (amount, softness) = (dlg.amount, dlg.softness);
                        let selection_mask = self
                            .active_project()
                            .and_then(|p| p.canvas_state.selection_mask.clone());
                        if let Some(project) = self.active_project_mut() {
                            Self::apply_fullres_effect(
                                &mut project.canvas_state,
                                idx,
                                flat,
                                |img| {
                                    crate::ops::effects::vignette_core(
                                        img,
                                        amount,
                                        softness,
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
                                "Vignette".to_string(),
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
                            let (amount, softness) = (dlg.amount, dlg.softness);
                            let selection_mask = self
                                .active_project()
                                .and_then(|p| p.canvas_state.selection_mask.clone());
                            self.spawn_preview_job(
                                ctx.input(|i| i.time),
                                "Vignette".to_string(),
                                idx,
                                original.clone(),
                                flat.clone(),
                                move |img| {
                                    crate::ops::effects::vignette_core(
                                        img,
                                        amount,
                                        softness,
                                        selection_mask.as_ref(),
                                    )
                                },
                            );
                        }
                    }
                }
            },

            ActiveDialog::Halftone(dlg) => match dlg.show(ctx) {
                DialogResult::Changed => {
                    dlg.first_open = false;
                    let idx = dlg.layer_idx;
                    if let (Some(original), Some(flat)) = (&dlg.original_pixels, &dlg.original_flat)
                    {
                        let (dot_size, angle) = (dlg.dot_size, dlg.angle);
                        let shape = dlg.halftone_shape();
                        let selection_mask = self
                            .active_project()
                            .and_then(|p| p.canvas_state.selection_mask.clone());
                        self.spawn_preview_job(
                            ctx.input(|i| i.time),
                            "Halftone".to_string(),
                            idx,
                            original.clone(),
                            flat.clone(),
                            move |img| {
                                crate::ops::effects::halftone_core(
                                    img,
                                    dot_size,
                                    angle,
                                    shape,
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
                        let (dot_size, angle) = (dlg.dot_size, dlg.angle);
                        let shape = dlg.halftone_shape();
                        let selection_mask = self
                            .active_project()
                            .and_then(|p| p.canvas_state.selection_mask.clone());
                        if let Some(project) = self.active_project_mut() {
                            Self::apply_fullres_effect(
                                &mut project.canvas_state,
                                idx,
                                flat,
                                |img| {
                                    crate::ops::effects::halftone_core(
                                        img,
                                        dot_size,
                                        angle,
                                        shape,
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
                                "Halftone".to_string(),
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
                            let (dot_size, angle) = (dlg.dot_size, dlg.angle);
                            let shape = dlg.halftone_shape();
                            let selection_mask = self
                                .active_project()
                                .and_then(|p| p.canvas_state.selection_mask.clone());
                            self.spawn_preview_job(
                                ctx.input(|i| i.time),
                                "Halftone".to_string(),
                                idx,
                                original.clone(),
                                flat.clone(),
                                move |img| {
                                    crate::ops::effects::halftone_core(
                                        img,
                                        dot_size,
                                        angle,
                                        shape,
                                        selection_mask.as_ref(),
                                    )
                                },
                            );
                        }
                    }
                }
            },


            _ => unreachable!(),
        }

        self.active_dialog = std::mem::replace(dialog, ActiveDialog::None);
        true
    }
}

