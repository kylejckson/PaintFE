impl PaintFEApp {
    fn process_paintdotnet_plugin_dialog(
        &mut self,
        ctx: &egui::Context,
        dialog: &mut ActiveDialog,
    ) -> bool {
        let ActiveDialog::PaintDotNetPlugin(plugin_dialog) = dialog else {
            return false;
        };

        match plugin_dialog.show(ctx) {
            DialogResult::Changed => {
                let plugin = plugin_dialog.plugin.clone();
                let parameters = plugin_dialog.values.clone();
                let selection = plugin_dialog.selection.clone();
                let errors = std::sync::Arc::clone(&plugin_dialog.render_error);
                self.spawn_preview_job(
                    ctx.input(|i| i.time),
                    plugin.name.clone(),
                    plugin_dialog.layer_idx,
                    plugin_dialog.original_pixels.clone(),
                    plugin_dialog.original_flat.clone(),
                    move |image| match crate::paintdotnet_plugins::render(
                        &plugin,
                        image,
                        &parameters,
                        selection.as_ref(),
                    ) {
                        Ok(result) => result,
                        Err(error) => {
                            crate::paintdotnet_plugins::record_plugin_error(&plugin, &error);
                            if let Ok(mut slot) = errors.lock() {
                                *slot = Some(error);
                            }
                            image.clone()
                        }
                    },
                );
            }
            DialogResult::Ok(()) => {
                self.preview_job_token = self.preview_job_token.wrapping_add(1);
                let plugin = plugin_dialog.plugin.clone();
                let parameters = plugin_dialog.values.clone();
                let selection = plugin_dialog.selection.clone();
                let layer_idx = plugin_dialog.layer_idx;
                if let Some(project) = self.active_project_mut()
                    && let Some(layer) = project.canvas_state.layers.get_mut(layer_idx)
                    && layer.is_text_layer()
                {
                    layer.content = crate::canvas::LayerContent::Raster;
                }
                self.spawn_filter_job(
                    ctx.input(|i| i.time),
                    plugin.name.clone(),
                    layer_idx,
                    plugin_dialog.original_pixels.clone(),
                    plugin_dialog.original_flat.clone(),
                    move |image| {
                        crate::paintdotnet_plugins::render(
                            &plugin,
                            image,
                            &parameters,
                            selection.as_ref(),
                        )
                        .unwrap_or_else(|error| {
                            crate::paintdotnet_plugins::record_plugin_error(&plugin, &error);
                            eprintln!("Paint.NET plugin render failed: {error}");
                            image.clone()
                        })
                    },
                );
                self.active_dialog = ActiveDialog::None;
                return true;
            }
            DialogResult::Cancel => {
                self.preview_job_token = self.preview_job_token.wrapping_add(1);
                self.filter_cancel.store(true, std::sync::atomic::Ordering::Relaxed);
                if let Some(project) = self.active_project_mut()
                    && let Some(layer) = project.canvas_state.layers.get_mut(plugin_dialog.layer_idx)
                {
                    layer.pixels = plugin_dialog.original_pixels.clone();
                    project.canvas_state.mark_dirty(None);
                }
                self.active_dialog = ActiveDialog::None;
                return true;
            }
            DialogResult::Open => {}
        }
        false
    }
}
