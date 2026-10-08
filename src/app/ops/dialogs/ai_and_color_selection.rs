impl PaintFEApp {
    fn process_ai_and_color_selection_dialog(&mut self, ctx: &egui::Context, dialog: &mut ActiveDialog) -> bool {
        let matched = matches!(dialog, ActiveDialog::RemoveBackground(_) | ActiveDialog::ColorRange(_));
        if !matched {
            return false;
        }

        match dialog {

            ActiveDialog::RemoveBackground(dlg) => match dlg.show(ctx) {
                DialogResult::Ok(settings) => {
                    self.active_dialog = ActiveDialog::None;
                    if self.background_removal_pending { return true; }
                    if let Some(project) = self.active_project() {
                        let layer_idx = project.canvas_state.active_layer_index;
                        let original_pixels = project.canvas_state.layers[layer_idx].pixels.clone();
                        let original_flat = original_pixels.to_rgba_image();
                        let dll_path = self.settings.onnx_runtime_path.clone();
                        let model_path = self.settings.birefnet_model_path.clone();

                        let options = crate::ops::ai::InferenceOptions {
                            graph_optimization: self.settings.onnx_graph_optimization,
                            cpu_memory_arena: self.settings.onnx_cpu_memory_arena,
                        };
                        self.background_removal_pending = true;
                        self.filter_status_description = t!("status.remove_background");
                        self.spawn_fallible_filter_job(
                            ctx.input(|i| i.time),
                            "Remove Background".to_string(),
                            layer_idx,
                            original_pixels,
                            original_flat,
                            move |input_img| crate::ops::ai::remove_background_with_options(
                                &dll_path, &model_path, input_img, &settings, &options,
                            ).map_err(|error| error.to_string()),
                        );
                    }
                    return true;
                }
                DialogResult::Cancel => {
                    self.active_dialog = ActiveDialog::None;
                    return true;
                }
                _ => {}
            },

            ActiveDialog::ColorRange(dlg) => match dlg.show(ctx) {
                DialogResult::Changed => {
                    if let Some(project) = self.active_project_mut() {
                        project.canvas_state.selection_mask = dlg.original_selection.clone();
                        crate::ops::adjustments::select_color_range(
                            &mut project.canvas_state,
                            dlg.hue_center,
                            dlg.hue_tolerance,
                            dlg.sat_min,
                            dlg.fuzziness,
                            dlg.mode,
                        );
                        project.canvas_state.invalidate_selection_overlay();
                        project.canvas_state.mark_dirty(None);
                    }
                }
                DialogResult::Ok(_) => {
                    self.active_dialog = ActiveDialog::None;
                    return true;
                }
                DialogResult::Cancel => {
                    self.filter_cancel
                        .store(true, std::sync::atomic::Ordering::Relaxed);
                    if let Some(project) = self.active_project_mut() {
                        project.canvas_state.selection_mask = dlg.original_selection.clone();
                        project.canvas_state.invalidate_selection_overlay();
                        project.canvas_state.mark_dirty(None);
                    }
                    self.active_dialog = ActiveDialog::None;
                    return true;
                }
                _ => {}
            },

            _ => unreachable!(),
        }

        self.active_dialog = std::mem::replace(dialog, ActiveDialog::None);
        true
    }
}

