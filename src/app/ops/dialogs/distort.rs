impl PaintFEApp {
    fn process_distort_dialog(&mut self, ctx: &egui::Context, dialog: &mut ActiveDialog) -> bool {
        let matched = matches!(dialog, ActiveDialog::Crystallize(_) | ActiveDialog::Dents(_) | ActiveDialog::Pixelate(_) | ActiveDialog::Bulge(_) | ActiveDialog::Twist(_));
        if !matched {
            return false;
        }

        match dialog {

            ActiveDialog::Crystallize(dlg) => match dlg.show(ctx) {
                DialogResult::Changed => {
                    dlg.first_open = false;
                    let idx = dlg.layer_idx;
                    if let (Some(original), Some(flat)) = (&dlg.original_pixels, &dlg.original_flat)
                    {
                        let (cell_size, seed) = (dlg.cell_size, dlg.seed);
                        let selection_mask = self
                            .active_project()
                            .and_then(|p| p.canvas_state.selection_mask.clone());
                        self.spawn_preview_job(
                            ctx.input(|i| i.time),
                            "Crystallize".to_string(),
                            idx,
                            original.clone(),
                            flat.clone(),
                            move |img| {
                                crate::ops::effects::crystallize_core(
                                    img,
                                    cell_size,
                                    seed,
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
                        let (cell_size, seed) = (dlg.cell_size, dlg.seed);
                        let selection_mask = self
                            .active_project()
                            .and_then(|p| p.canvas_state.selection_mask.clone());
                        if let Some(project) = self.active_project_mut() {
                            Self::apply_fullres_effect(
                                &mut project.canvas_state,
                                idx,
                                flat,
                                |img| {
                                    crate::ops::effects::crystallize_core(
                                        img,
                                        cell_size,
                                        seed,
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
                                "Crystallize".to_string(),
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
                            let (cell_size, seed) = (dlg.cell_size, dlg.seed);
                            let selection_mask = self
                                .active_project()
                                .and_then(|p| p.canvas_state.selection_mask.clone());
                            self.spawn_preview_job(
                                ctx.input(|i| i.time),
                                "Crystallize".to_string(),
                                idx,
                                original.clone(),
                                flat.clone(),
                                move |img| {
                                    crate::ops::effects::crystallize_core(
                                        img,
                                        cell_size,
                                        seed,
                                        selection_mask.as_ref(),
                                    )
                                },
                            );
                        }
                    }
                }
            },

            ActiveDialog::Dents(dlg) => match dlg.show(ctx) {
                DialogResult::Changed => {
                    dlg.first_open = false;
                    let idx = dlg.layer_idx;
                    if let (Some(original), Some(flat)) = (&dlg.original_pixels, &dlg.original_flat)
                    {
                        let (scale_p, amount, seed) = (dlg.scale, dlg.amount, dlg.seed);
                        let (octaves, roughness) = (dlg.octaves as u32, dlg.roughness);
                        let (pinch, wrap) = (dlg.pinch, dlg.wrap);
                        let selection_mask = self
                            .active_project()
                            .and_then(|p| p.canvas_state.selection_mask.clone());
                        self.spawn_preview_job(
                            ctx.input(|i| i.time),
                            "Dents".to_string(),
                            idx,
                            original.clone(),
                            flat.clone(),
                            move |img| {
                                crate::ops::effects::dents_core(
                                    img, scale_p, amount, seed, octaves, roughness, pinch, wrap,
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
                        let (scale_p, amount, seed) = (dlg.scale, dlg.amount, dlg.seed);
                        let (octaves, roughness) = (dlg.octaves as u32, dlg.roughness);
                        let (pinch, wrap) = (dlg.pinch, dlg.wrap);
                        let selection_mask = self
                            .active_project()
                            .and_then(|p| p.canvas_state.selection_mask.clone());
                        if let Some(project) = self.active_project_mut() {
                            Self::apply_fullres_effect(
                                &mut project.canvas_state,
                                idx,
                                flat,
                                |img| {
                                    crate::ops::effects::dents_core(
                                        img, scale_p, amount, seed, octaves, roughness, pinch,
                                        wrap,
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
                                "Dents".to_string(),
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
                            let (scale_p, amount, seed) = (dlg.scale, dlg.amount, dlg.seed);
                            let (octaves, roughness) = (dlg.octaves as u32, dlg.roughness);
                            let (pinch, wrap) = (dlg.pinch, dlg.wrap);
                            let selection_mask = self
                                .active_project()
                                .and_then(|p| p.canvas_state.selection_mask.clone());
                            self.spawn_preview_job(
                                ctx.input(|i| i.time),
                                "Dents".to_string(),
                                idx,
                                original.clone(),
                                flat.clone(),
                                move |img| {
                                    crate::ops::effects::dents_core(
                                        img, scale_p, amount, seed, octaves, roughness, pinch,
                                        wrap,
                                        selection_mask.as_ref(),
                                    )
                                },
                            );
                        }
                    }
                }
            },

            ActiveDialog::Pixelate(dlg) => match dlg.show(ctx) {
                DialogResult::Changed => {
                    dlg.first_open = false;
                    let idx = dlg.layer_idx;
                    if let (Some(original), Some(flat)) = (&dlg.original_pixels, &dlg.original_flat)
                    {
                        let block_size = dlg.block_size as u32;
                        let selection_mask = self
                            .active_project()
                            .and_then(|p| p.canvas_state.selection_mask.clone());
                        self.spawn_preview_job(
                            ctx.input(|i| i.time),
                            "Pixelate".to_string(),
                            idx,
                            original.clone(),
                            flat.clone(),
                            move |img| {
                                crate::ops::effects::pixelate_core(
                                    img,
                                    block_size,
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
                        let block_size = dlg.block_size as u32;
                        let selection_mask = self
                            .active_project()
                            .and_then(|p| p.canvas_state.selection_mask.clone());
                        if let Some(project) = self.active_project_mut() {
                            Self::apply_fullres_effect(
                                &mut project.canvas_state,
                                idx,
                                flat,
                                |img| {
                                    crate::ops::effects::pixelate_core(
                                        img,
                                        block_size,
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
                                "Pixelate".to_string(),
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
                            let block_size = dlg.block_size as u32;
                            let selection_mask = self
                                .active_project()
                                .and_then(|p| p.canvas_state.selection_mask.clone());
                            self.spawn_preview_job(
                                ctx.input(|i| i.time),
                                "Pixelate".to_string(),
                                idx,
                                original.clone(),
                                flat.clone(),
                                move |img| {
                                    crate::ops::effects::pixelate_core(
                                        img,
                                        block_size,
                                        selection_mask.as_ref(),
                                    )
                                },
                            );
                        }
                    }
                }
            },

            ActiveDialog::Bulge(dlg) => match dlg.show(ctx) {
                DialogResult::Changed => {
                    dlg.first_open = false;
                    let idx = dlg.layer_idx;
                    if let (Some(original), Some(flat)) = (&dlg.original_pixels, &dlg.original_flat)
                    {
                        let amount = dlg.amount;
                        let origin = (dlg.origin_x / 100.0, dlg.origin_y / 100.0);
                        let selection_mask = self
                            .active_project()
                            .and_then(|p| p.canvas_state.selection_mask.clone());
                        self.spawn_preview_job(
                            ctx.input(|i| i.time),
                            "Bulge".to_string(),
                            idx,
                            original.clone(),
                            flat.clone(),
                            move |img| {
                                crate::ops::effects::bulge_core_at(
                                    img,
                                    amount,
                                    origin,
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
                        let origin = (dlg.origin_x / 100.0, dlg.origin_y / 100.0);
                        let selection_mask = self
                            .active_project()
                            .and_then(|p| p.canvas_state.selection_mask.clone());
                        if let Some(project) = self.active_project_mut() {
                            Self::apply_fullres_effect(
                                &mut project.canvas_state,
                                idx,
                                flat,
                                |img| {
                                    crate::ops::effects::bulge_core_at(
                                        img,
                                        amount,
                                        origin,
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
                                "Bulge".to_string(),
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
                            let origin = (dlg.origin_x / 100.0, dlg.origin_y / 100.0);
                            let selection_mask = self
                                .active_project()
                                .and_then(|p| p.canvas_state.selection_mask.clone());
                            self.spawn_preview_job(
                                ctx.input(|i| i.time),
                                "Bulge".to_string(),
                                idx,
                                original.clone(),
                                flat.clone(),
                                move |img| {
                                    crate::ops::effects::bulge_core_at(
                                        img,
                                        amount,
                                        origin,
                                        selection_mask.as_ref(),
                                    )
                                },
                            );
                        }
                    }
                }
            },

            ActiveDialog::Twist(dlg) => match dlg.show(ctx) {
                DialogResult::Changed => {
                    dlg.first_open = false;
                    let idx = dlg.layer_idx;
                    if let (Some(original), Some(flat)) = (&dlg.original_pixels, &dlg.original_flat)
                    {
                        let angle = dlg.angle;
                        let origin = (dlg.origin_x / 100.0, dlg.origin_y / 100.0);
                        let selection_mask = self
                            .active_project()
                            .and_then(|p| p.canvas_state.selection_mask.clone());
                        self.spawn_preview_job(
                            ctx.input(|i| i.time),
                            "Twist".to_string(),
                            idx,
                            original.clone(),
                            flat.clone(),
                            move |img| {
                                crate::ops::effects::twist_core_at(
                                    img,
                                    angle,
                                    origin,
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
                        let angle = dlg.angle;
                        let origin = (dlg.origin_x / 100.0, dlg.origin_y / 100.0);
                        let selection_mask = self
                            .active_project()
                            .and_then(|p| p.canvas_state.selection_mask.clone());
                        if let Some(project) = self.active_project_mut() {
                            Self::apply_fullres_effect(
                                &mut project.canvas_state,
                                idx,
                                flat,
                                |img| {
                                    crate::ops::effects::twist_core_at(
                                        img,
                                        angle,
                                        origin,
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
                                "Twist".to_string(),
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
                            let angle = dlg.angle;
                            let origin = (dlg.origin_x / 100.0, dlg.origin_y / 100.0);
                            let selection_mask = self
                                .active_project()
                                .and_then(|p| p.canvas_state.selection_mask.clone());
                            self.spawn_preview_job(
                                ctx.input(|i| i.time),
                                "Twist".to_string(),
                                idx,
                                original.clone(),
                                flat.clone(),
                                move |img| {
                                    crate::ops::effects::twist_core_at(
                                        img,
                                        angle,
                                        origin,
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

