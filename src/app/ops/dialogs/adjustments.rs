impl PaintFEApp {
    fn process_adjustments_dialog(
        &mut self,
        ctx: &egui::Context,
        dialog: &mut ActiveDialog,
    ) -> bool {
        let matched = matches!(
            dialog,
            ActiveDialog::BrightnessContrast(_)
                | ActiveDialog::HueSaturation(_)
                | ActiveDialog::Exposure(_)
                | ActiveDialog::HighlightsShadows(_)
                | ActiveDialog::Levels(_)
                | ActiveDialog::Curves(_)
                | ActiveDialog::TemperatureTint(_)
                | ActiveDialog::Threshold(_)
                | ActiveDialog::Posterize(_)
                | ActiveDialog::ColorBalance(_)
                | ActiveDialog::GradientMap(_)
                | ActiveDialog::BlackAndWhite(_)
                | ActiveDialog::Vibrance(_)
        );
        if !matched {
            return false;
        }

        match dialog {
            ActiveDialog::BrightnessContrast(dlg) => {
                match crate::ui::dialogs::remember::show_result(ctx, dlg.show(ctx)) {
                    DialogResult::Changed => {
                        let brightness = dlg.brightness;
                        let contrast = dlg.contrast;
                        let idx = dlg.layer_idx;
                        if let Some(flat) = &dlg.original_flat {
                            let active_idx = self.active_project_index;
                            if let Some(project) = self.projects.get_mut(active_idx) {
                                crate::ops::adjustments::brightness_contrast_from_flat_gpu(
                                    &mut project.canvas_state,
                                    idx,
                                    brightness,
                                    contrast,
                                    flat,
                                    &self.canvas.gpu_renderer,
                                );
                            }
                        }
                    }
                    DialogResult::Ok(_) => {
                        crate::ui::dialogs::remember::mark_applied(ctx);
                        self.active_dialog = ActiveDialog::None;
                        if let Some(project) = self.active_project_mut() {
                            let idx = dlg.layer_idx;
                            if let Some(original) = &dlg.original_pixels
                                && idx < project.canvas_state.layers.len()
                            {
                                let adjusted = project.canvas_state.layers[idx].pixels.clone();
                                project.canvas_state.layers[idx].pixels = original.clone();
                                let mut cmd = SingleLayerSnapshotCommand::new_for_layer(
                                    HistoryAction::BrightnessContrast
                                        .label("Brightness/Contrast".to_string()),
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
            // HUE / SATURATION
            // ================================================================
            ActiveDialog::HueSaturation(dlg) => {
                match crate::ui::dialogs::remember::show_result(ctx, dlg.show(ctx)) {
                    DialogResult::Changed => {
                        let hue = dlg.hue;
                        let sat = dlg.saturation;
                        let light = dlg.lightness;
                        let idx = dlg.layer_idx;
                        let per_band = dlg.per_band;
                        let bands = dlg.bands;
                        if let Some(flat) = &dlg.original_flat {
                            let active_idx = self.active_project_index;
                            if let Some(project) = self.projects.get_mut(active_idx) {
                                if per_band {
                                    crate::ops::adjustments::hue_saturation_per_band_from_flat(
                                        &mut project.canvas_state,
                                        idx,
                                        hue,
                                        sat,
                                        light,
                                        &bands,
                                        flat,
                                    );
                                } else {
                                    crate::ops::adjustments::hue_saturation_lightness_from_flat_gpu(
                                        &mut project.canvas_state,
                                        idx,
                                        hue,
                                        sat,
                                        light,
                                        flat,
                                        &self.canvas.gpu_renderer,
                                    );
                                }
                            }
                        }
                    }
                    DialogResult::Ok(_) => {
                        crate::ui::dialogs::remember::mark_applied(ctx);
                        self.active_dialog = ActiveDialog::None;
                        if let Some(project) = self.active_project_mut() {
                            let idx = dlg.layer_idx;
                            let label = if dlg.per_band {
                                HistoryAction::HueSaturation.label("Hue/Saturation (Per Band)")
                            } else {
                                HistoryAction::HueSaturation.label("Hue/Saturation")
                            };
                            if let Some(original) = &dlg.original_pixels
                                && idx < project.canvas_state.layers.len()
                            {
                                let adjusted = project.canvas_state.layers[idx].pixels.clone();
                                project.canvas_state.layers[idx].pixels = original.clone();
                                let mut cmd = SingleLayerSnapshotCommand::new_for_layer(
                                    label,
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
            // EXPOSURE
            // ================================================================
            ActiveDialog::Exposure(dlg) => {
                match crate::ui::dialogs::remember::show_result(ctx, dlg.show(ctx)) {
                    DialogResult::Changed => {
                        let exposure = dlg.exposure;
                        let idx = dlg.layer_idx;
                        if let Some(flat) = &dlg.original_flat
                            && let Some(project) = self.active_project_mut()
                        {
                            crate::ops::adjustments::exposure_from_flat(
                                &mut project.canvas_state,
                                idx,
                                exposure,
                                flat,
                            );
                        }
                    }
                    DialogResult::Ok(_) => {
                        crate::ui::dialogs::remember::mark_applied(ctx);
                        self.active_dialog = ActiveDialog::None;
                        if let Some(project) = self.active_project_mut() {
                            let idx = dlg.layer_idx;
                            if let Some(original) = &dlg.original_pixels
                                && idx < project.canvas_state.layers.len()
                            {
                                let adjusted = project.canvas_state.layers[idx].pixels.clone();
                                project.canvas_state.layers[idx].pixels = original.clone();
                                let mut cmd = SingleLayerSnapshotCommand::new_for_layer(
                                    HistoryAction::Exposure.label("Exposure".to_string()),
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
            // HIGHLIGHTS / SHADOWS
            // ================================================================
            ActiveDialog::HighlightsShadows(dlg) => {
                match crate::ui::dialogs::remember::show_result(ctx, dlg.show(ctx)) {
                    DialogResult::Changed => {
                        let shadows = dlg.shadows;
                        let highlights = dlg.highlights;
                        let idx = dlg.layer_idx;
                        if let Some(flat) = &dlg.original_flat
                            && let Some(project) = self.active_project_mut()
                        {
                            crate::ops::adjustments::highlights_shadows_from_flat(
                                &mut project.canvas_state,
                                idx,
                                shadows,
                                highlights,
                                flat,
                            );
                        }
                    }
                    DialogResult::Ok(_) => {
                        crate::ui::dialogs::remember::mark_applied(ctx);
                        self.active_dialog = ActiveDialog::None;
                        if let Some(project) = self.active_project_mut() {
                            let idx = dlg.layer_idx;
                            if let Some(original) = &dlg.original_pixels
                                && idx < project.canvas_state.layers.len()
                            {
                                let adjusted = project.canvas_state.layers[idx].pixels.clone();
                                project.canvas_state.layers[idx].pixels = original.clone();
                                let mut cmd = SingleLayerSnapshotCommand::new_for_layer(
                                    HistoryAction::HighlightsShadows
                                        .label("Highlights/Shadows".to_string()),
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
            // LEVELS
            // ================================================================
            ActiveDialog::Levels(dlg) => {
                match crate::ui::dialogs::remember::show_result(ctx, dlg.show(ctx)) {
                    DialogResult::Changed => {
                        let res = dlg.as_result();
                        let idx = dlg.layer_idx;
                        if let Some(flat) = &dlg.original_flat
                            && let Some(project) = self.active_project_mut()
                        {
                            crate::ops::adjustments::levels_from_flat_per_channel(
                                &mut project.canvas_state,
                                idx,
                                res.master,
                                res.r_ch,
                                res.g_ch,
                                res.b_ch,
                                flat,
                            );
                        }
                    }
                    DialogResult::Ok(_) => {
                        crate::ui::dialogs::remember::mark_applied(ctx);
                        self.active_dialog = ActiveDialog::None;
                        if let Some(project) = self.active_project_mut() {
                            let idx = dlg.layer_idx;
                            if let Some(original) = &dlg.original_pixels
                                && idx < project.canvas_state.layers.len()
                            {
                                let adjusted = project.canvas_state.layers[idx].pixels.clone();
                                project.canvas_state.layers[idx].pixels = original.clone();
                                let mut cmd = SingleLayerSnapshotCommand::new_for_layer(
                                    HistoryAction::Levels.label("Levels".to_string()),
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
            // CURVES
            // ================================================================
            ActiveDialog::Curves(dlg) => {
                match crate::ui::dialogs::remember::show_result(ctx, dlg.show(ctx)) {
                    DialogResult::Changed => {
                        let ch_data: [(&[(f32, f32)], bool); 5] = [
                            (&dlg.channels[0].points, dlg.channels[0].enabled),
                            (&dlg.channels[1].points, dlg.channels[1].enabled),
                            (&dlg.channels[2].points, dlg.channels[2].enabled),
                            (&dlg.channels[3].points, dlg.channels[3].enabled),
                            (&dlg.channels[4].points, dlg.channels[4].enabled),
                        ];
                        let idx = dlg.layer_idx;
                        if let Some(flat) = &dlg.original_flat
                            && let Some(project) = self.active_project_mut()
                        {
                            crate::ops::adjustments::curves_from_flat_multi(
                                &mut project.canvas_state,
                                idx,
                                &ch_data,
                                flat,
                            );
                        }
                    }
                    DialogResult::Ok(_) => {
                        crate::ui::dialogs::remember::mark_applied(ctx);
                        self.active_dialog = ActiveDialog::None;
                        if let Some(project) = self.active_project_mut() {
                            let idx = dlg.layer_idx;
                            if let Some(original) = &dlg.original_pixels
                                && idx < project.canvas_state.layers.len()
                            {
                                let adjusted = project.canvas_state.layers[idx].pixels.clone();
                                project.canvas_state.layers[idx].pixels = original.clone();
                                let mut cmd = SingleLayerSnapshotCommand::new_for_layer(
                                    HistoryAction::Curves.label("Curves".to_string()),
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
            // TEMPERATURE / TINT
            // ================================================================
            ActiveDialog::TemperatureTint(dlg) => {
                match crate::ui::dialogs::remember::show_result(ctx, dlg.show(ctx)) {
                    DialogResult::Changed => {
                        let temperature = dlg.temperature;
                        let tint = dlg.tint;
                        let idx = dlg.layer_idx;
                        if let Some(flat) = &dlg.original_flat
                            && let Some(project) = self.active_project_mut()
                        {
                            crate::ops::adjustments::temperature_tint_from_flat(
                                &mut project.canvas_state,
                                idx,
                                temperature,
                                tint,
                                flat,
                            );
                        }
                    }
                    DialogResult::Ok(_) => {
                        crate::ui::dialogs::remember::mark_applied(ctx);
                        self.active_dialog = ActiveDialog::None;
                        if let Some(project) = self.active_project_mut() {
                            let idx = dlg.layer_idx;
                            if let Some(original) = &dlg.original_pixels
                                && idx < project.canvas_state.layers.len()
                            {
                                let adjusted = project.canvas_state.layers[idx].pixels.clone();
                                project.canvas_state.layers[idx].pixels = original.clone();
                                let mut cmd = SingleLayerSnapshotCommand::new_for_layer(
                                    HistoryAction::TemperatureTint
                                        .label("Temperature/Tint".to_string()),
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

            ActiveDialog::Threshold(dlg) => {
                match crate::ui::dialogs::remember::show_result(ctx, dlg.show(ctx)) {
                    DialogResult::Changed => {
                        let idx = dlg.layer_idx;
                        if let Some(flat) = &dlg.original_flat
                            && let Some(project) = self.active_project_mut()
                        {
                            crate::ops::adjustments::threshold_from_flat(
                                &mut project.canvas_state,
                                idx,
                                dlg.level,
                                flat,
                            );
                        }
                    }
                    DialogResult::Ok(_) => {
                        crate::ui::dialogs::remember::mark_applied(ctx);
                        self.active_dialog = ActiveDialog::None;
                        if let Some(project) = self.active_project_mut() {
                            let idx = dlg.layer_idx;
                            if let Some(original) = &dlg.original_pixels
                                && idx < project.canvas_state.layers.len()
                            {
                                let adjusted = project.canvas_state.layers[idx].pixels.clone();
                                project.canvas_state.layers[idx].pixels = original.clone();
                                let mut cmd = SingleLayerSnapshotCommand::new_for_layer(
                                    HistoryAction::Threshold.label("Threshold".to_string()),
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

            ActiveDialog::Posterize(dlg) => {
                match crate::ui::dialogs::remember::show_result(ctx, dlg.show(ctx)) {
                    DialogResult::Changed => {
                        let idx = dlg.layer_idx;
                        if let Some(flat) = &dlg.original_flat
                            && let Some(project) = self.active_project_mut()
                        {
                            crate::ops::adjustments::posterize_from_flat(
                                &mut project.canvas_state,
                                idx,
                                dlg.levels,
                                flat,
                            );
                        }
                    }
                    DialogResult::Ok(_) => {
                        crate::ui::dialogs::remember::mark_applied(ctx);
                        self.active_dialog = ActiveDialog::None;
                        if let Some(project) = self.active_project_mut() {
                            let idx = dlg.layer_idx;
                            if let Some(original) = &dlg.original_pixels
                                && idx < project.canvas_state.layers.len()
                            {
                                let adjusted = project.canvas_state.layers[idx].pixels.clone();
                                project.canvas_state.layers[idx].pixels = original.clone();
                                let mut cmd = SingleLayerSnapshotCommand::new_for_layer(
                                    HistoryAction::Posterize.label("Posterize".to_string()),
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

            ActiveDialog::ColorBalance(dlg) => {
                match crate::ui::dialogs::remember::show_result(ctx, dlg.show(ctx)) {
                    DialogResult::Changed => {
                        let idx = dlg.layer_idx;
                        if let Some(flat) = &dlg.original_flat
                            && let Some(project) = self.active_project_mut()
                        {
                            crate::ops::adjustments::color_balance_from_flat(
                                &mut project.canvas_state,
                                idx,
                                dlg.shadows,
                                dlg.midtones,
                                dlg.highlights,
                                flat,
                            );
                        }
                    }
                    DialogResult::Ok(_) => {
                        crate::ui::dialogs::remember::mark_applied(ctx);
                        self.active_dialog = ActiveDialog::None;
                        if let Some(project) = self.active_project_mut() {
                            let idx = dlg.layer_idx;
                            if let Some(original) = &dlg.original_pixels
                                && idx < project.canvas_state.layers.len()
                            {
                                let adjusted = project.canvas_state.layers[idx].pixels.clone();
                                project.canvas_state.layers[idx].pixels = original.clone();
                                let mut cmd = SingleLayerSnapshotCommand::new_for_layer(
                                    HistoryAction::ColorBalance.label("Color Balance".to_string()),
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

            ActiveDialog::GradientMap(dlg) => {
                match crate::ui::dialogs::remember::show_result(ctx, dlg.show(ctx)) {
                    DialogResult::Changed => {
                        let idx = dlg.layer_idx;
                        let lut = dlg.build_lut();
                        if let Some(flat) = &dlg.original_flat
                            && let Some(project) = self.active_project_mut()
                        {
                            crate::ops::adjustments::gradient_map_from_flat(
                                &mut project.canvas_state,
                                idx,
                                &lut,
                                flat,
                            );
                        }
                    }
                    DialogResult::Ok(_) => {
                        crate::ui::dialogs::remember::mark_applied(ctx);
                        self.active_dialog = ActiveDialog::None;
                        if let Some(project) = self.active_project_mut() {
                            let idx = dlg.layer_idx;
                            if let Some(original) = &dlg.original_pixels
                                && idx < project.canvas_state.layers.len()
                            {
                                let adjusted = project.canvas_state.layers[idx].pixels.clone();
                                project.canvas_state.layers[idx].pixels = original.clone();
                                let mut cmd = SingleLayerSnapshotCommand::new_for_layer(
                                    HistoryAction::GradientMap.label("Gradient Map".to_string()),
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

            ActiveDialog::BlackAndWhite(dlg) => {
                match crate::ui::dialogs::remember::show_result(ctx, dlg.show(ctx)) {
                    DialogResult::Changed => {
                        let idx = dlg.layer_idx;
                        if let Some(flat) = &dlg.original_flat
                            && let Some(project) = self.active_project_mut()
                        {
                            crate::ops::adjustments::black_and_white_from_flat(
                                &mut project.canvas_state,
                                idx,
                                dlg.r_weight,
                                dlg.g_weight,
                                dlg.b_weight,
                                flat,
                            );
                        }
                    }
                    DialogResult::Ok(_) => {
                        crate::ui::dialogs::remember::mark_applied(ctx);
                        self.active_dialog = ActiveDialog::None;
                        if let Some(project) = self.active_project_mut() {
                            let idx = dlg.layer_idx;
                            if let Some(original) = &dlg.original_pixels
                                && idx < project.canvas_state.layers.len()
                            {
                                let adjusted = project.canvas_state.layers[idx].pixels.clone();
                                project.canvas_state.layers[idx].pixels = original.clone();
                                let mut cmd = SingleLayerSnapshotCommand::new_for_layer(
                                    HistoryAction::BlackAndWhite.label("Black & White".to_string()),
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

            ActiveDialog::Vibrance(dlg) => {
                match crate::ui::dialogs::remember::show_result(ctx, dlg.show(ctx)) {
                    DialogResult::Changed => {
                        let idx = dlg.layer_idx;
                        if let Some(flat) = &dlg.original_flat
                            && let Some(project) = self.active_project_mut()
                        {
                            crate::ops::adjustments::vibrance_from_flat(
                                &mut project.canvas_state,
                                idx,
                                dlg.amount,
                                flat,
                            );
                        }
                    }
                    DialogResult::Ok(_) => {
                        crate::ui::dialogs::remember::mark_applied(ctx);
                        self.active_dialog = ActiveDialog::None;
                        if let Some(project) = self.active_project_mut() {
                            let idx = dlg.layer_idx;
                            if let Some(original) = &dlg.original_pixels
                                && idx < project.canvas_state.layers.len()
                            {
                                let adjusted = project.canvas_state.layers[idx].pixels.clone();
                                project.canvas_state.layers[idx].pixels = original.clone();
                                let mut cmd = SingleLayerSnapshotCommand::new_for_layer(
                                    HistoryAction::Vibrance.label("Vibrance".to_string()),
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

            _ => {}
        }

        self.active_dialog = std::mem::replace(dialog, ActiveDialog::None);
        true
    }
}
