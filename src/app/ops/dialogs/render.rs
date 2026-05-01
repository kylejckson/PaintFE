impl PaintFEApp {
    fn process_render_dialog(&mut self, ctx: &egui::Context, dialog: &mut ActiveDialog) -> bool {
        let matched = matches!(dialog, ActiveDialog::Grid(_) | ActiveDialog::DropShadow(_) | ActiveDialog::Outline(_) | ActiveDialog::CanvasBorder(_));
        if !matched {
            return false;
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
                        self.spawn_preview_job(
                            ctx.input(|i| i.time),
                            "Grid".to_string(),
                            idx,
                            original.clone(),
                            flat.clone(),
                            move |img| {
                                crate::ops::effects::grid_core(
                                    img, cw, ch, lw, c, style, opacity, None,
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
                        if let Some(project) = self.active_project_mut() {
                            Self::apply_fullres_effect(
                                &mut project.canvas_state,
                                idx,
                                flat,
                                |img| {
                                    crate::ops::effects::grid_core(
                                        img, cw, ch, lw, c, style, opacity, None,
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
                            self.spawn_preview_job(
                                ctx.input(|i| i.time),
                                "Grid".to_string(),
                                idx,
                                original.clone(),
                                flat.clone(),
                                move |img| {
                                    crate::ops::effects::grid_core(
                                        img, cw, ch, lw, c, style, opacity, None,
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
                        self.spawn_preview_job(
                            ctx.input(|i| i.time),
                            "Drop Shadow".to_string(),
                            idx,
                            original.clone(),
                            flat.clone(),
                            move |img| {
                                crate::ops::effects::shadow_core(
                                    img, ox, oy, br, widen, c, opacity, None,
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
                        if let Some(project) = self.active_project_mut() {
                            Self::apply_fullres_effect(
                                &mut project.canvas_state,
                                idx,
                                flat,
                                |img| {
                                    crate::ops::effects::shadow_core(
                                        img, ox, oy, br, widen, c, opacity, None,
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
                            self.spawn_preview_job(
                                ctx.input(|i| i.time),
                                "Drop Shadow".to_string(),
                                idx,
                                original.clone(),
                                flat.clone(),
                                move |img| {
                                    crate::ops::effects::shadow_core(
                                        img, ox, oy, br, widen, c, opacity, None,
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
                        self.spawn_preview_job(
                            ctx.input(|i| i.time),
                            "Outline".to_string(),
                            idx,
                            original.clone(),
                            flat.clone(),
                            move |img| {
                                crate::ops::effects::outline_core(
                                    img, width, c, mode, anti_alias, None,
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
                        self.spawn_filter_job(
                            ctx.input(|i| i.time),
                            "Outline".to_string(),
                            idx,
                            original.clone(),
                            flat.clone(),
                            move |img| {
                                crate::ops::effects::outline_core(
                                    img, width, c, mode, anti_alias, None,
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
                            self.spawn_preview_job(
                                ctx.input(|i| i.time),
                                "Outline".to_string(),
                                idx,
                                original.clone(),
                                flat.clone(),
                                move |img| {
                                    crate::ops::effects::outline_core(
                                        img, width, c, mode, anti_alias, None,
                                    )
                                },
                            );
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
                        self.spawn_preview_job(
                            ctx.input(|i| i.time),
                            "Canvas Border".to_string(),
                            idx,
                            original.clone(),
                            flat.clone(),
                            move |img| crate::ops::effects::canvas_border_core(img, width, c, None),
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
                        if let Some(project) = self.active_project_mut() {
                            Self::apply_fullres_effect(
                                &mut project.canvas_state,
                                idx,
                                flat,
                                |img| crate::ops::effects::canvas_border_core(img, width, c, None),
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
                            self.spawn_preview_job(
                                ctx.input(|i| i.time),
                                "Canvas Border".to_string(),
                                idx,
                                original.clone(),
                                flat.clone(),
                                move |img| {
                                    crate::ops::effects::canvas_border_core(img, width, c, None)
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

