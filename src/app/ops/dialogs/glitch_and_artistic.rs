impl PaintFEApp {
    fn process_glitch_and_artistic_dialog(&mut self, ctx: &egui::Context, dialog: &mut ActiveDialog) -> bool {
        let matched = matches!(dialog, ActiveDialog::PixelDrag(_) | ActiveDialog::RgbDisplace(_) | ActiveDialog::Ink(_) | ActiveDialog::OilPainting(_) | ActiveDialog::ColorFilter(_) | ActiveDialog::ColorToAlpha(_) | ActiveDialog::Contours(_));
        if !matched {
            return false;
        }

        match dialog {

            ActiveDialog::PixelDrag(dlg) => match dlg.show(ctx) {
                DialogResult::Changed => {
                    dlg.first_open = false;
                    let idx = dlg.layer_idx;
                    if let (Some(original), Some(flat)) = (&dlg.original_pixels, &dlg.original_flat)
                    {
                        let (seed, amount) = (dlg.seed, dlg.amount);
                        let distance = dlg.distance as u32;
                        let direction = dlg.direction;
                        self.spawn_preview_job(
                            ctx.input(|i| i.time),
                            "Pixel Drag".to_string(),
                            idx,
                            original.clone(),
                            flat.clone(),
                            move |img| {
                                crate::ops::effects::pixel_drag_core(
                                    img, seed, amount, distance, direction, None,
                                )
                            },
                        );
                    }
                }
                DialogResult::Ok(_) => {
                    self.preview_job_token = self.preview_job_token.wrapping_add(1);
                    let idx = dlg.layer_idx;
                    if let Some(flat) = &dlg.original_flat {
                        let (seed, amount) = (dlg.seed, dlg.amount);
                        let distance = dlg.distance as u32;
                        let direction = dlg.direction;
                        if let Some(project) = self.active_project_mut() {
                            Self::apply_fullres_effect(
                                &mut project.canvas_state,
                                idx,
                                flat,
                                |img| {
                                    crate::ops::effects::pixel_drag_core(
                                        img, seed, amount, distance, direction, None,
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
                                "Pixel Drag".to_string(),
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
                            let (seed, amount) = (dlg.seed, dlg.amount);
                            let distance = dlg.distance as u32;
                            let direction = dlg.direction;
                            self.spawn_preview_job(
                                ctx.input(|i| i.time),
                                "Pixel Drag".to_string(),
                                idx,
                                original.clone(),
                                flat.clone(),
                                move |img| {
                                    crate::ops::effects::pixel_drag_core(
                                        img, seed, amount, distance, direction, None,
                                    )
                                },
                            );
                        }
                    }
                }
            },

            ActiveDialog::RgbDisplace(dlg) => match dlg.show(ctx) {
                DialogResult::Changed => {
                    dlg.first_open = false;
                    let idx = dlg.layer_idx;
                    if let (Some(original), Some(flat)) = (&dlg.original_pixels, &dlg.original_flat)
                    {
                        let r_off = (dlg.r_x as i32, dlg.r_y as i32);
                        let g_off = (dlg.g_x as i32, dlg.g_y as i32);
                        let b_off = (dlg.b_x as i32, dlg.b_y as i32);
                        self.spawn_preview_job(
                            ctx.input(|i| i.time),
                            "RGB Displace".to_string(),
                            idx,
                            original.clone(),
                            flat.clone(),
                            move |img| {
                                crate::ops::effects::rgb_displace_core(
                                    img, r_off, g_off, b_off, None,
                                )
                            },
                        );
                    }
                }
                DialogResult::Ok(_) => {
                    self.preview_job_token = self.preview_job_token.wrapping_add(1);
                    let idx = dlg.layer_idx;
                    if let Some(flat) = &dlg.original_flat {
                        let r_off = (dlg.r_x as i32, dlg.r_y as i32);
                        let g_off = (dlg.g_x as i32, dlg.g_y as i32);
                        let b_off = (dlg.b_x as i32, dlg.b_y as i32);
                        if let Some(project) = self.active_project_mut() {
                            Self::apply_fullres_effect(
                                &mut project.canvas_state,
                                idx,
                                flat,
                                |img| {
                                    crate::ops::effects::rgb_displace_core(
                                        img, r_off, g_off, b_off, None,
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
                                "RGB Displace".to_string(),
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
                            let r_off = (dlg.r_x as i32, dlg.r_y as i32);
                            let g_off = (dlg.g_x as i32, dlg.g_y as i32);
                            let b_off = (dlg.b_x as i32, dlg.b_y as i32);
                            self.spawn_preview_job(
                                ctx.input(|i| i.time),
                                "RGB Displace".to_string(),
                                idx,
                                original.clone(),
                                flat.clone(),
                                move |img| {
                                    crate::ops::effects::rgb_displace_core(
                                        img, r_off, g_off, b_off, None,
                                    )
                                },
                            );
                        }
                    }
                }
            },

            ActiveDialog::Ink(dlg) => match dlg.show(ctx) {
                DialogResult::Changed => {
                    dlg.first_open = false;
                    let idx = dlg.layer_idx;
                    if let (Some(original), Some(flat)) = (&dlg.original_pixels, &dlg.original_flat)
                    {
                        let (edge_strength, threshold) = (dlg.edge_strength, dlg.threshold);
                        self.spawn_preview_job(
                            ctx.input(|i| i.time),
                            "Ink".to_string(),
                            idx,
                            original.clone(),
                            flat.clone(),
                            move |img| {
                                crate::ops::effects::ink_core(img, edge_strength, threshold, None)
                            },
                        );
                    }
                }
                DialogResult::Ok(_) => {
                    self.preview_job_token = self.preview_job_token.wrapping_add(1);
                    let idx = dlg.layer_idx;
                    if let Some(flat) = &dlg.original_flat {
                        let (edge_strength, threshold) = (dlg.edge_strength, dlg.threshold);
                        if let Some(project) = self.active_project_mut() {
                            Self::apply_fullres_effect(
                                &mut project.canvas_state,
                                idx,
                                flat,
                                |img| {
                                    crate::ops::effects::ink_core(
                                        img,
                                        edge_strength,
                                        threshold,
                                        None,
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
                                "Ink".to_string(),
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
                            let (edge_strength, threshold) = (dlg.edge_strength, dlg.threshold);
                            self.spawn_preview_job(
                                ctx.input(|i| i.time),
                                "Ink".to_string(),
                                idx,
                                original.clone(),
                                flat.clone(),
                                move |img| {
                                    crate::ops::effects::ink_core(
                                        img,
                                        edge_strength,
                                        threshold,
                                        None,
                                    )
                                },
                            );
                        }
                    }
                }
            },

            ActiveDialog::OilPainting(dlg) => match dlg.show(ctx) {
                DialogResult::Changed => {
                    dlg.first_open = false;
                    let idx = dlg.layer_idx;
                    if let (Some(original), Some(flat)) = (&dlg.original_pixels, &dlg.original_flat)
                    {
                        let (radius, levels) = (dlg.radius as u32, dlg.levels as u32);
                        self.spawn_preview_job(
                            ctx.input(|i| i.time),
                            "Oil Painting".to_string(),
                            idx,
                            original.clone(),
                            flat.clone(),
                            move |img| {
                                crate::ops::effects::oil_painting_core(img, radius, levels, None)
                            },
                        );
                    }
                }
                DialogResult::Ok(_) => {
                    self.preview_job_token = self.preview_job_token.wrapping_add(1);
                    let idx = dlg.layer_idx;
                    if let Some(flat) = &dlg.original_flat {
                        let (radius, levels) = (dlg.radius as u32, dlg.levels as u32);
                        if let Some(project) = self.active_project_mut() {
                            Self::apply_fullres_effect(
                                &mut project.canvas_state,
                                idx,
                                flat,
                                |img| {
                                    crate::ops::effects::oil_painting_core(
                                        img, radius, levels, None,
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
                                "Oil Painting".to_string(),
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
                            let (radius, levels) = (dlg.radius as u32, dlg.levels as u32);
                            self.spawn_preview_job(
                                ctx.input(|i| i.time),
                                "Oil Painting".to_string(),
                                idx,
                                original.clone(),
                                flat.clone(),
                                move |img| {
                                    crate::ops::effects::oil_painting_core(
                                        img, radius, levels, None,
                                    )
                                },
                            );
                        }
                    }
                }
            },

            ActiveDialog::ColorFilter(dlg) => match dlg.show(ctx) {
                DialogResult::Changed => {
                    dlg.first_open = false;
                    let idx = dlg.layer_idx;
                    if let (Some(original), Some(flat)) = (&dlg.original_pixels, &dlg.original_flat)
                    {
                        let fc = [
                            (dlg.color[0] * 255.0) as u8,
                            (dlg.color[1] * 255.0) as u8,
                            (dlg.color[2] * 255.0) as u8,
                            255,
                        ];
                        let intensity = dlg.intensity;
                        let mode = dlg.filter_mode();
                        self.spawn_preview_job(
                            ctx.input(|i| i.time),
                            "Color Filter".to_string(),
                            idx,
                            original.clone(),
                            flat.clone(),
                            move |img| {
                                crate::ops::effects::color_filter_core(
                                    img, fc, intensity, mode, None,
                                )
                            },
                        );
                    }
                }
                DialogResult::Ok(_) => {
                    self.preview_job_token = self.preview_job_token.wrapping_add(1);
                    let idx = dlg.layer_idx;
                    if let Some(flat) = &dlg.original_flat {
                        let fc = [
                            (dlg.color[0] * 255.0) as u8,
                            (dlg.color[1] * 255.0) as u8,
                            (dlg.color[2] * 255.0) as u8,
                            255,
                        ];
                        let intensity = dlg.intensity;
                        let mode = dlg.filter_mode();
                        if let Some(project) = self.active_project_mut() {
                            Self::apply_fullres_effect(
                                &mut project.canvas_state,
                                idx,
                                flat,
                                |img| {
                                    crate::ops::effects::color_filter_core(
                                        img, fc, intensity, mode, None,
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
                                "Color Filter".to_string(),
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
                            let fc = [
                                (dlg.color[0] * 255.0) as u8,
                                (dlg.color[1] * 255.0) as u8,
                                (dlg.color[2] * 255.0) as u8,
                                255,
                            ];
                            let intensity = dlg.intensity;
                            let mode = dlg.filter_mode();
                            self.spawn_preview_job(
                                ctx.input(|i| i.time),
                                "Color Filter".to_string(),
                                idx,
                                original.clone(),
                                flat.clone(),
                                move |img| {
                                    crate::ops::effects::color_filter_core(
                                        img, fc, intensity, mode, None,
                                    )
                                },
                            );
                        }
                    }
                }
            },

            ActiveDialog::ColorToAlpha(dlg) => match dlg.show(
                ctx,
                self.assets.icon_texture(crate::assets::Icon::ColorRemover),
            ) {
                DialogResult::Changed => {
                    dlg.first_open = false;
                    let idx = dlg.layer_idx;
                    if let (Some(original), Some(flat)) = (&dlg.original_pixels, &dlg.original_flat)
                    {
                        let settings = dlg.settings();
                        let selection_mask = self
                            .active_project()
                            .and_then(|project| project.canvas_state.selection_mask.clone());
                        self.spawn_preview_job(
                            ctx.input(|i| i.time),
                            "Color to Alpha".to_string(),
                            idx,
                            original.clone(),
                            flat.clone(),
                            move |img| {
                                crate::ops::color_removal::color_to_alpha_core(
                                    img,
                                    &settings,
                                    selection_mask.as_ref(),
                                )
                            },
                        );
                    }
                }
                DialogResult::Ok(settings) => {
                    self.preview_job_token = self.preview_job_token.wrapping_add(1);
                    let idx = dlg.layer_idx;
                    let selection_mask = self
                        .active_project()
                        .and_then(|project| project.canvas_state.selection_mask.clone());
                    if let Some(flat) = &dlg.original_flat
                        && let Some(project) = self.active_project_mut()
                    {
                        Self::apply_fullres_effect(
                            &mut project.canvas_state,
                            idx,
                            flat,
                            |img| {
                                crate::ops::color_removal::color_to_alpha_core(
                                    img,
                                    &settings,
                                    selection_mask.as_ref(),
                                )
                            },
                        );
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
                                "Color to Alpha".to_string(),
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
                            let settings = dlg.settings();
                            let selection_mask = self
                                .active_project()
                                .and_then(|project| project.canvas_state.selection_mask.clone());
                            self.spawn_preview_job(
                                ctx.input(|i| i.time),
                                "Color to Alpha".to_string(),
                                idx,
                                original.clone(),
                                flat.clone(),
                                move |img| {
                                    crate::ops::color_removal::color_to_alpha_core(
                                        img,
                                        &settings,
                                        selection_mask.as_ref(),
                                    )
                                },
                            );
                        }
                    }
                }
            },

            ActiveDialog::Contours(dlg) => match dlg.show(ctx) {
                DialogResult::Changed => {
                    dlg.first_open = false;
                    let idx = dlg.layer_idx;
                    if let (Some(original), Some(flat)) = (&dlg.original_pixels, &dlg.original_flat)
                    {
                        let (contour_scale, frequency, line_width) =
                            (dlg.scale, dlg.frequency, dlg.line_width);
                        let lc = [
                            (dlg.line_color[0] * 255.0) as u8,
                            (dlg.line_color[1] * 255.0) as u8,
                            (dlg.line_color[2] * 255.0) as u8,
                            255u8,
                        ];
                        let (seed, octaves, blend) = (dlg.seed, dlg.octaves as u32, dlg.blend);
                        self.spawn_preview_job(
                            ctx.input(|i| i.time),
                            "Contours".to_string(),
                            idx,
                            original.clone(),
                            flat.clone(),
                            move |img| {
                                crate::ops::effects::contours_core(
                                    img,
                                    contour_scale,
                                    frequency,
                                    line_width,
                                    lc,
                                    seed,
                                    octaves,
                                    blend,
                                    None,
                                )
                            },
                        );
                    }
                }
                DialogResult::Ok(_) => {
                    self.preview_job_token = self.preview_job_token.wrapping_add(1);
                    let idx = dlg.layer_idx;
                    if let Some(flat) = &dlg.original_flat {
                        let (contour_scale, frequency, line_width) =
                            (dlg.scale, dlg.frequency, dlg.line_width);
                        let lc = [
                            (dlg.line_color[0] * 255.0) as u8,
                            (dlg.line_color[1] * 255.0) as u8,
                            (dlg.line_color[2] * 255.0) as u8,
                            255u8,
                        ];
                        let (seed, octaves, blend) = (dlg.seed, dlg.octaves as u32, dlg.blend);
                        if let Some(project) = self.active_project_mut() {
                            Self::apply_fullres_effect(
                                &mut project.canvas_state,
                                idx,
                                flat,
                                |img| {
                                    crate::ops::effects::contours_core(
                                        img,
                                        contour_scale,
                                        frequency,
                                        line_width,
                                        lc,
                                        seed,
                                        octaves,
                                        blend,
                                        None,
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
                                "Contours".to_string(),
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
                            let (contour_scale, frequency, line_width) =
                                (dlg.scale, dlg.frequency, dlg.line_width);
                            let lc = [
                                (dlg.line_color[0] * 255.0) as u8,
                                (dlg.line_color[1] * 255.0) as u8,
                                (dlg.line_color[2] * 255.0) as u8,
                                255u8,
                            ];
                            let (seed, octaves, blend) = (dlg.seed, dlg.octaves as u32, dlg.blend);
                            self.spawn_preview_job(
                                ctx.input(|i| i.time),
                                "Contours".to_string(),
                                idx,
                                original.clone(),
                                flat.clone(),
                                move |img| {
                                    crate::ops::effects::contours_core(
                                        img,
                                        contour_scale,
                                        frequency,
                                        line_width,
                                        lc,
                                        seed,
                                        octaves,
                                        blend,
                                        None,
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

