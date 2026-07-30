impl PaintFEApp {
    fn process_render_dialog(&mut self, ctx: &egui::Context, dialog: &mut ActiveDialog) -> bool {
        let matched = matches!(dialog, ActiveDialog::Grid(_) | ActiveDialog::DropShadow(_) | ActiveDialog::Outline(_) | ActiveDialog::CanvasBorder(_) | ActiveDialog::SeamlessTexture(_));
        if !matched {
            return false;
        }

        if let ActiveDialog::SeamlessTexture(dlg) = dialog {
            if !dlg.wrap_preview_enabled {
                dlg.previous_wrap_preview = self.active_project()
                    .is_some_and(|project| project.canvas_state.show_wrap_preview);
                dlg.wrap_preview_enabled = true;
            }
            if let Some(project) = self.active_project_mut() {
                project.canvas_state.show_wrap_preview = true;
            }
        }

        match dialog {

            ActiveDialog::Grid(dlg) => match dlg.show(ctx) {
                DialogResult::Changed => {
                    dlg.first_open = false;
                    let idx = dlg.layer_idx;
                    if let (Some(original), Some(flat)) = (&dlg.original_pixels, &dlg.original_flat)
                    {
                        let cw = dlg.cell_w as u32;
                        let ch = dlg.cell_h as u32;
                        let lw = dlg.line_width as u32;
                        let c = [
                            (dlg.color[0] * 255.0) as u8,
                            (dlg.color[1] * 255.0) as u8,
                            (dlg.color[2] * 255.0) as u8,
                            255,
                        ];
                        let style = dlg.grid_style();
                        let opacity = dlg.opacity;
                        let selection_mask = self
                            .active_project()
                            .and_then(|p| p.canvas_state.selection_mask.clone());
                        self.spawn_preview_job(
                            ctx.input(|i| i.time),
                            "Grid".to_string(),
                            idx,
                            original.clone(),
                            flat.clone(),
                            move |img| {
                                crate::ops::effects::grid_core(
                                    img,
                                    cw,
                                    ch,
                                    lw,
                                    c,
                                    style,
                                    opacity,
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
                        let cw = dlg.cell_w as u32;
                        let ch = dlg.cell_h as u32;
                        let lw = dlg.line_width as u32;
                        let c = [
                            (dlg.color[0] * 255.0) as u8,
                            (dlg.color[1] * 255.0) as u8,
                            (dlg.color[2] * 255.0) as u8,
                            255,
                        ];
                        let style = dlg.grid_style();
                        let opacity = dlg.opacity;
                        let selection_mask = self
                            .active_project()
                            .and_then(|p| p.canvas_state.selection_mask.clone());
                        if let Some(project) = self.active_project_mut() {
                            Self::apply_fullres_effect(
                                &mut project.canvas_state,
                                idx,
                                flat,
                                |img| {
                                    crate::ops::effects::grid_core(
                                        img,
                                        cw,
                                        ch,
                                        lw,
                                        c,
                                        style,
                                        opacity,
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
                                "Grid".to_string(),
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
                            let cw = dlg.cell_w as u32;
                            let ch = dlg.cell_h as u32;
                            let lw = dlg.line_width as u32;
                            let c = [
                                (dlg.color[0] * 255.0) as u8,
                                (dlg.color[1] * 255.0) as u8,
                                (dlg.color[2] * 255.0) as u8,
                                255,
                            ];
                            let style = dlg.grid_style();
                            let opacity = dlg.opacity;
                            let selection_mask = self
                                .active_project()
                                .and_then(|p| p.canvas_state.selection_mask.clone());
                            self.spawn_preview_job(
                                ctx.input(|i| i.time),
                                "Grid".to_string(),
                                idx,
                                original.clone(),
                                flat.clone(),
                                move |img| {
                                    crate::ops::effects::grid_core(
                                        img,
                                        cw,
                                        ch,
                                        lw,
                                        c,
                                        style,
                                        opacity,
                                        selection_mask.as_ref(),
                                    )
                                },
                            );
                        }
                    }
                }
            },

            ActiveDialog::DropShadow(dlg) => match dlg.show(ctx) {
                DialogResult::Changed => {
                    dlg.first_open = false;
                    let idx = dlg.layer_idx;
                    if let (Some(original), Some(flat)) = (&dlg.original_pixels, &dlg.original_flat)
                    {
                        let ox = dlg.offset_x as i32;
                        let oy = dlg.offset_y as i32;
                        let br = dlg.blur_radius;
                        let widen = dlg.widen_radius;
                        let c = [
                            (dlg.color[0] * 255.0).round() as u8,
                            (dlg.color[1] * 255.0).round() as u8,
                            (dlg.color[2] * 255.0).round() as u8,
                            255,
                        ];
                        let opacity = dlg.opacity;
                        let selection_mask = self
                            .active_project()
                            .and_then(|p| p.canvas_state.selection_mask.clone());
                        self.spawn_preview_job(
                            ctx.input(|i| i.time),
                            "Drop Shadow".to_string(),
                            idx,
                            original.clone(),
                            flat.clone(),
                            move |img| {
                                crate::ops::effects::shadow_core(
                                    img,
                                    ox,
                                    oy,
                                    br,
                                    widen,
                                    c,
                                    opacity,
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
                        let ox = dlg.offset_x as i32;
                        let oy = dlg.offset_y as i32;
                        let br = dlg.blur_radius;
                        let widen = dlg.widen_radius;
                        let c = [
                            (dlg.color[0] * 255.0).round() as u8,
                            (dlg.color[1] * 255.0).round() as u8,
                            (dlg.color[2] * 255.0).round() as u8,
                            255,
                        ];
                        let opacity = dlg.opacity;
                        let selection_mask = self
                            .active_project()
                            .and_then(|p| p.canvas_state.selection_mask.clone());
                        if let Some(project) = self.active_project_mut() {
                            Self::apply_fullres_effect(
                                &mut project.canvas_state,
                                idx,
                                flat,
                                |img| {
                                    crate::ops::effects::shadow_core(
                                        img,
                                        ox,
                                        oy,
                                        br,
                                        widen,
                                        c,
                                        opacity,
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
                                "Drop Shadow".to_string(),
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
                            let ox = dlg.offset_x as i32;
                            let oy = dlg.offset_y as i32;
                            let br = dlg.blur_radius;
                            let widen = dlg.widen_radius;
                            let c = [
                                (dlg.color[0] * 255.0).round() as u8,
                                (dlg.color[1] * 255.0).round() as u8,
                                (dlg.color[2] * 255.0).round() as u8,
                                255,
                            ];
                            let opacity = dlg.opacity;
                            let selection_mask = self
                                .active_project()
                                .and_then(|p| p.canvas_state.selection_mask.clone());
                            self.spawn_preview_job(
                                ctx.input(|i| i.time),
                                "Drop Shadow".to_string(),
                                idx,
                                original.clone(),
                                flat.clone(),
                                move |img| {
                                    crate::ops::effects::shadow_core(
                                        img,
                                        ox,
                                        oy,
                                        br,
                                        widen,
                                        c,
                                        opacity,
                                        selection_mask.as_ref(),
                                    )
                                },
                            );
                        }
                    }
                }
            },

            ActiveDialog::Outline(dlg) => match dlg.show(ctx) {
                DialogResult::Changed => {
                    dlg.first_open = false;
                    let idx = dlg.layer_idx;
                    if let (Some(original), Some(flat)) = (&dlg.original_pixels, &dlg.original_flat)
                    {
                        let width = dlg.width as u32;
                        let c = [
                            (dlg.color[0] * 255.0) as u8,
                            (dlg.color[1] * 255.0) as u8,
                            (dlg.color[2] * 255.0) as u8,
                            255,
                        ];
                        let mode = dlg.outline_mode();
                        let anti_alias = dlg.anti_alias;
                        let selection_mask = self
                            .active_project()
                            .and_then(|p| p.canvas_state.selection_mask.clone());
                        self.spawn_preview_job(
                            ctx.input(|i| i.time),
                            "Outline".to_string(),
                            idx,
                            original.clone(),
                            flat.clone(),
                            move |img| {
                                crate::ops::effects::outline_core(
                                    img,
                                    width,
                                    c,
                                    mode,
                                    anti_alias,
                                    selection_mask.as_ref(),
                                )
                            },
                        );
                    }
                }
                DialogResult::Ok(_) => {
                    self.preview_job_token = self.preview_job_token.wrapping_add(1);
                    self.filter_cancel
                        .store(true, std::sync::atomic::Ordering::Relaxed);
                    let idx = dlg.layer_idx;
                    if let (Some(original), Some(flat)) = (&dlg.original_pixels, &dlg.original_flat)
                    {
                        let width = dlg.width as u32;
                        let c = [
                            (dlg.color[0] * 255.0) as u8,
                            (dlg.color[1] * 255.0) as u8,
                            (dlg.color[2] * 255.0) as u8,
                            255,
                        ];
                        let mode = dlg.outline_mode();
                        let anti_alias = dlg.anti_alias;
                        let selection_mask = self
                            .active_project()
                            .and_then(|p| p.canvas_state.selection_mask.clone());
                        self.spawn_filter_job(
                            ctx.input(|i| i.time),
                            "Outline".to_string(),
                            idx,
                            original.clone(),
                            flat.clone(),
                            move |img| {
                                crate::ops::effects::outline_core(
                                    img,
                                    width,
                                    c,
                                    mode,
                                    anti_alias,
                                    selection_mask.as_ref(),
                                )
                            },
                        );
                    }
                    self.active_dialog = ActiveDialog::None;
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
                            let width = dlg.width as u32;
                            let c = [
                                (dlg.color[0] * 255.0) as u8,
                                (dlg.color[1] * 255.0) as u8,
                                (dlg.color[2] * 255.0) as u8,
                                255,
                            ];
                            let mode = dlg.outline_mode();
                            let anti_alias = dlg.anti_alias;
                            let selection_mask = self
                                .active_project()
                                .and_then(|p| p.canvas_state.selection_mask.clone());
                            self.spawn_preview_job(
                                ctx.input(|i| i.time),
                                "Outline".to_string(),
                                idx,
                                original.clone(),
                                flat.clone(),
                                move |img| {
                                    crate::ops::effects::outline_core(
                                        img,
                                        width,
                                        c,
                                        mode,
                                        anti_alias,
                                        selection_mask.as_ref(),
                                    )
                                },
                            );
                        }
                    }
                }
            },

            ActiveDialog::SeamlessTexture(dlg) => match dlg.show(ctx) {
                DialogResult::Changed => {
                    dlg.first_open = false;
                    let idx = dlg.layer_idx;
                    if let (Some(original), Some(flat)) = (&dlg.original_pixels, &dlg.original_flat) {
                        let blend_px = dlg.blend_px as u32;
                        let strength = dlg.strength;
                        let horizontal = dlg.horizontal;
                        let vertical = dlg.vertical;
                        let profile = dlg.profile();
                        let organicity = dlg.organicity;
                        let dent_size = dlg.dent_size;
                        let seed = dlg.seed;
                        self.spawn_preview_job(ctx.input(|i| i.time), "Make Seamless Texture".to_string(), idx, original.clone(), flat.clone(), move |img| {
                            crate::ops::effects::seamless_texture_core(img, blend_px, strength, horizontal, vertical, profile, organicity, dent_size, seed)
                        });
                    }
                }
                DialogResult::Ok(_) => {
                    self.preview_job_token = self.preview_job_token.wrapping_add(1);
                    let idx = dlg.layer_idx;
                    if let Some(flat) = &dlg.original_flat {
                        let blend_px = dlg.blend_px as u32;
                        let strength = dlg.strength;
                        let horizontal = dlg.horizontal;
                        let vertical = dlg.vertical;
                        let profile = dlg.profile();
                        let organicity = dlg.organicity;
                        let dent_size = dlg.dent_size;
                        let seed = dlg.seed;
                        if let Some(project) = self.active_project_mut() {
                            Self::apply_fullres_effect(&mut project.canvas_state, idx, flat, |img| {
                                crate::ops::effects::seamless_texture_core(img, blend_px, strength, horizontal, vertical, profile, organicity, dent_size, seed)
                            });
                            project.canvas_state.show_wrap_preview = dlg.previous_wrap_preview;
                        }
                    }
                    self.active_dialog = ActiveDialog::None;
                    if let Some(project) = self.active_project_mut() {
                        if let Some(original) = &dlg.original_pixels && idx < project.canvas_state.layers.len() {
                            let adjusted = project.canvas_state.layers[idx].pixels.clone();
                            project.canvas_state.layers[idx].pixels = original.clone();
                            let mut cmd = SingleLayerSnapshotCommand::new_for_layer("Make Seamless Texture".to_string(), &project.canvas_state, idx);
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
                    self.filter_cancel.store(true, std::sync::atomic::Ordering::Relaxed);
                    let idx = dlg.layer_idx;
                    if let Some(project) = self.active_project_mut() {
                        if let Some(original) = &dlg.original_pixels
                            && let Some(layer) = project.canvas_state.layers.get_mut(idx) {
                            layer.pixels = original.clone();
                        }
                        project.canvas_state.show_wrap_preview = dlg.previous_wrap_preview;
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
                        if let (Some(original), Some(flat)) = (&dlg.original_pixels, &dlg.original_flat) {
                            let blend_px = dlg.blend_px as u32;
                            let strength = dlg.strength;
                            let horizontal = dlg.horizontal;
                            let vertical = dlg.vertical;
                            let profile = dlg.profile();
                            let organicity = dlg.organicity;
                            let dent_size = dlg.dent_size;
                            let seed = dlg.seed;
                            self.spawn_preview_job(ctx.input(|i| i.time), "Make Seamless Texture".to_string(), idx, original.clone(), flat.clone(), move |img| {
                                crate::ops::effects::seamless_texture_core(img, blend_px, strength, horizontal, vertical, profile, organicity, dent_size, seed)
                            });
                        }
                    }
                }
            },

            ActiveDialog::CanvasBorder(dlg) => match dlg.show(ctx) {
                DialogResult::Changed => {
                    dlg.first_open = false;
                    let idx = dlg.layer_idx;
                    if let (Some(original), Some(flat)) = (&dlg.original_pixels, &dlg.original_flat)
                    {
                        let width = dlg.width as u32;
                        let primary = self.colors_panel.get_primary_color_f32();
                        let c = [
                            (primary[0] * 255.0) as u8,
                            (primary[1] * 255.0) as u8,
                            (primary[2] * 255.0) as u8,
                            255,
                        ];
                        let selection_mask = self
                            .active_project()
                            .and_then(|p| p.canvas_state.selection_mask.clone());
                        self.spawn_preview_job(
                            ctx.input(|i| i.time),
                            "Canvas Border".to_string(),
                            idx,
                            original.clone(),
                            flat.clone(),
                            move |img| {
                                crate::ops::effects::canvas_border_core(
                                    img,
                                    width,
                                    c,
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
                        let width = dlg.width as u32;
                        let primary = self.colors_panel.get_primary_color_f32();
                        let c = [
                            (primary[0] * 255.0) as u8,
                            (primary[1] * 255.0) as u8,
                            (primary[2] * 255.0) as u8,
                            255,
                        ];
                        let selection_mask = self
                            .active_project()
                            .and_then(|p| p.canvas_state.selection_mask.clone());
                        if let Some(project) = self.active_project_mut() {
                            Self::apply_fullres_effect(
                                &mut project.canvas_state,
                                idx,
                                flat,
                                |img| {
                                    crate::ops::effects::canvas_border_core(
                                        img,
                                        width,
                                        c,
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
                                "Canvas Border".to_string(),
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
                            let width = dlg.width as u32;
                            let primary = self.colors_panel.get_primary_color_f32();
                            let c = [
                                (primary[0] * 255.0) as u8,
                                (primary[1] * 255.0) as u8,
                                (primary[2] * 255.0) as u8,
                                255,
                            ];
                            let selection_mask = self
                                .active_project()
                                .and_then(|p| p.canvas_state.selection_mask.clone());
                            self.spawn_preview_job(
                                ctx.input(|i| i.time),
                                "Canvas Border".to_string(),
                                idx,
                                original.clone(),
                                flat.clone(),
                                move |img| {
                                    crate::ops::effects::canvas_border_core(
                                        img,
                                        width,
                                        c,
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

