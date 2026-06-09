impl PaintFEApp {
    fn handle_runtime_modal_flow(&mut self, ctx: &egui::Context) -> bool {
        self.settings_window
            .show(ctx, &mut self.settings, &mut self.theme, &self.assets);

        let current_paths = (
            self.settings.onnx_runtime_path.clone(),
            self.settings.birefnet_model_path.clone(),
        );
        if current_paths != self.onnx_last_probed_paths {
            self.onnx_last_probed_paths = current_paths;
            self.onnx_available = if !self.settings.onnx_runtime_path.is_empty()
                && !self.settings.birefnet_model_path.is_empty()
            {
                crate::ops::ai::probe_onnx_runtime(&self.settings.onnx_runtime_path).is_ok()
                    && std::path::Path::new(&self.settings.birefnet_model_path).exists()
            } else {
                false
            };
        }

        self.process_active_dialog(ctx);

        // Paste size confirmation (when clipboard image exceeds current canvas bounds)
        if let Some(req) = self.pending_paste_request.as_ref() {
            let mut do_resize = false;
            let mut do_keep = false;
            let mut do_cancel = false;
            egui::Window::new("Paste Image")
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .show(ctx, |ui| {
                    ui.vertical_centered(|ui| {
                        ui.label(format!(
                            "Clipboard image is larger than the current canvas ({}x{}).",
                            req.image.width(),
                            req.image.height()
                        ));
                        ui.label("Expand canvas to fit the pasted image?");
                    });
                    ui.add_space(8.0);
                    ui.horizontal(|ui| {
                        let btn_size = egui::vec2(136.0, 28.0);
                        if ui
                            .add(egui::Button::new("Expand Canvas").min_size(btn_size))
                            .clicked()
                        {
                            do_resize = true;
                        }
                        if ui
                            .add(egui::Button::new("Keep Canvas").min_size(btn_size))
                            .clicked()
                        {
                            do_keep = true;
                        }
                        if ui
                            .add(egui::Button::new("Cancel").min_size(btn_size))
                            .clicked()
                        {
                            do_cancel = true;
                        }
                    });
                });

            if do_resize && let Some(request) = self.pending_paste_request.take() {
                self.apply_pending_paste_request(request, true);
            }
            if do_keep && let Some(request) = self.pending_paste_request.take() {
                self.apply_pending_paste_request(request, false);
            }
            if do_cancel {
                self.pending_paste_request = None;
            }
        }

        if let Some(close_idx) = self.pending_close_index {
            let name = self
                .projects
                .get(close_idx)
                .map(|p| p.name.clone())
                .unwrap_or_default();
            let mut do_save = false;
            let mut do_discard = false;
            let mut do_cancel = false;
            egui::Window::new("Unsaved Changes")
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .show(ctx, |ui| {
                    ui.label(format!("\"{}\" has unsaved changes.", name));
                    ui.label("Do you want to save before closing?");
                    ui.add_space(8.0);
                    ui.horizontal(|ui| {
                        let btn_size = egui::vec2(100.0, 28.0);
                        if ui
                            .add(
                                egui::Button::new(egui::RichText::new("Save").strong())
                                    .min_size(btn_size),
                            )
                            .clicked()
                        {
                            do_save = true;
                        }
                        if ui
                            .add(
                                egui::Button::new(egui::RichText::new("Don't Save").strong())
                                    .min_size(btn_size),
                            )
                            .clicked()
                        {
                            do_discard = true;
                        }
                        if ui
                            .add(
                                egui::Button::new(egui::RichText::new("Cancel").strong())
                                    .min_size(btn_size),
                            )
                            .clicked()
                        {
                            do_cancel = true;
                        }
                    });
                });
            if do_save {
                self.open_save_as_for_project(close_idx);
                self.pending_close_index = None;
            }
            if do_discard {
                self.pending_close_index = None;
                self.force_close_project(close_idx);
            }
            if do_cancel {
                self.pending_close_index = None;
            }
        }

        if self.pending_exit {
            let dirty_projects: Vec<String> = self
                .projects
                .iter()
                .filter(|p| p.is_dirty)
                .map(|p| p.name.clone())
                .collect();

            let mut do_save = false;
            let mut do_exit = false;
            let mut do_cancel = false;

            egui::Window::new("Exit PaintFE")
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .min_width(380.0)
                .show(ctx, |ui| {
                    if dirty_projects.is_empty() {
                        do_exit = true;
                    }

                    if !dirty_projects.is_empty() {
                        const SHOW_MAX: usize = 3;
                        let overflow = dirty_projects.len().saturating_sub(SHOW_MAX);

                        ui.vertical_centered(|ui| {
                            if dirty_projects.len() == 1 {
                                ui.label(format!("\"{}\" has unsaved changes.", dirty_projects[0]));
                            } else {
                                ui.label(format!(
                                    "{} projects have unsaved changes:",
                                    dirty_projects.len()
                                ));
                                ui.add_space(4.0);
                                for name in dirty_projects.iter().take(SHOW_MAX) {
                                    ui.label(format!("\u{2022}  {}", name));
                                }
                                if overflow > 0 {
                                    ui.label(
                                        egui::RichText::new(format!("...and {} more", overflow))
                                            .weak()
                                            .italics(),
                                    );
                                }
                            }

                            ui.add_space(8.0);
                            ui.label("Do you want to save before exiting?");
                            ui.add_space(12.0);

                            let is_dark = ui.visuals().dark_mode;
                            let (danger_fill, danger_text) = if is_dark {
                                (
                                    egui::Color32::from_rgb(170, 35, 35),
                                    egui::Color32::from_rgb(255, 220, 220),
                                )
                            } else {
                                (egui::Color32::from_rgb(192, 38, 38), egui::Color32::WHITE)
                            };

                            let btn_size = egui::vec2(110.0, 26.0);
                            let total_w = btn_size.x * 3.0 + ui.spacing().item_spacing.x * 2.0;
                            let avail = ui.available_width();
                            let pad = ((avail - total_w) / 2.0).max(0.0);
                            ui.horizontal(|ui| {
                                ui.add_space(pad);
                                if ui
                                    .add(
                                        egui::Button::new(egui::RichText::new("Save").strong())
                                            .min_size(btn_size),
                                    )
                                    .clicked()
                                {
                                    do_save = true;
                                }
                                if ui
                                    .add(
                                        egui::Button::new(
                                            egui::RichText::new("Exit Without")
                                                .strong()
                                                .color(danger_text),
                                        )
                                        .fill(danger_fill)
                                        .min_size(btn_size),
                                    )
                                    .clicked()
                                {
                                    do_exit = true;
                                }
                                if ui
                                    .add(
                                        egui::Button::new(egui::RichText::new("Cancel").strong())
                                            .min_size(btn_size),
                                    )
                                    .clicked()
                                {
                                    do_cancel = true;
                                }
                            });

                            ui.add_space(6.0);
                        });
                    }
                });

            if do_save {
                let current_time = ctx.input(|i| i.time);
                self.handle_save_all(current_time);
                let untitled_dirty: Vec<usize> = self
                    .projects
                    .iter()
                    .enumerate()
                    .filter(|(_, p)| p.is_dirty && !p.file_handler.has_current_path())
                    .map(|(i, _)| i)
                    .collect();
                self.pending_exit = false;
                if untitled_dirty.is_empty() {
                    self.force_exit = true;
                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                } else {
                    self.exit_save_queue = untitled_dirty;
                    self.exit_save_active = true;
                    let first = self.exit_save_queue.remove(0);
                    self.open_save_as_for_project(first);
                }
            }
            if do_exit {
                self.pending_exit = false;
                self.force_exit = true;
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            }
            if do_cancel {
                self.pending_exit = false;
            }
        }

        self.settings.persist_new_file_lock_aspect = self.new_file_dialog.lock_aspect_ratio();
        if let Some((width, height)) = self.new_file_dialog.show(ctx) {
            self.new_project(width, height);
        }
        self.settings.persist_new_file_lock_aspect = self.new_file_dialog.lock_aspect_ratio();

        let save_dialog_was_open = self.save_file_dialog.open;
        let mut save_dialog_confirmed = false;
        if let Some(action) = self.save_file_dialog.show(ctx) {
            save_dialog_confirmed = true;
            let project_index = self.active_project_index;
            if project_index < self.projects.len() {
                if action.format == SaveFormat::Pfe {
                    let project = &mut self.projects[project_index];
                    project.canvas_state.ensure_all_text_layers_rasterized();
                    let pfe_data = crate::io::build_pfe(&project.canvas_state);
                    let path = action.path.clone();

                    let sender = self.io_sender.clone();
                    if self.pending_io_ops == 0 {
                        self.io_ops_start_time = Some(ctx.input(|i| i.time));
                    }
                    self.pending_io_ops += 1;

                    rayon::spawn(move || match crate::io::write_pfe(&pfe_data, &path) {
                        Ok(()) => {
                            let _ = sender.send(IoResult::SaveComplete {
                                project_index,
                                path,
                                format: SaveFormat::Pfe,
                                quality: 100,
                                tiff_compression: TiffCompression::None,
                                update_project_path: true,
                            });
                        }
                        Err(e) => {
                            let _ = sender.send(IoResult::SaveFailed {
                                project_index,
                                error: format!("{}", e),
                            });
                        }
                    });
                } else if action.animated && action.format.supports_animation() {
                    let project = &mut self.projects[project_index];
                    project.canvas_state.ensure_all_text_layers_rasterized();
                    let frames: Vec<image::RgbaImage> = project
                        .canvas_state
                        .layers
                        .iter()
                        .map(|l| l.pixels.to_rgba_image())
                        .collect();

                    let path = action.path.clone();
                    let format = action.format;
                    let quality = action.quality;
                    let tiff_compression = action.tiff_compression;
                    let fps = action.animation_fps;
                    let gif_colors = action.gif_colors;
                    let gif_dither = action.gif_dither;

                    project.file_handler.last_animated = true;
                    project.file_handler.last_animation_fps = fps;
                    project.file_handler.last_gif_colors = gif_colors;
                    project.file_handler.last_gif_dither = gif_dither;
                    project.was_animated = true;
                    project.animation_fps = fps;

                    let sender = self.io_sender.clone();
                    if self.pending_io_ops == 0 {
                        self.io_ops_start_time = Some(ctx.input(|i| i.time));
                    }
                    self.pending_io_ops += 1;

                    rayon::spawn(move || {
                        let result = match format {
                            SaveFormat::Gif => crate::io::encode_animated_gif(
                                &frames, fps, gif_colors, gif_dither, &path,
                            ),
                            SaveFormat::Png => crate::io::encode_animated_png(&frames, fps, &path),
                            _ => Err("Format does not support animation".to_string()),
                        };
                        match result {
                            Ok(()) => {
                                let _ = sender.send(IoResult::SaveComplete {
                                    project_index,
                                    path,
                                    format,
                                    quality,
                                    tiff_compression,
                                    update_project_path: true,
                                });
                            }
                            Err(e) => {
                                let _ = sender.send(IoResult::SaveFailed {
                                    project_index,
                                    error: e,
                                });
                            }
                        }
                    });
                } else {
                    let project = &mut self.projects[project_index];
                    project.canvas_state.ensure_all_text_layers_rasterized();
                    let export_image = crate::io::prepare_export_image(&project.canvas_state);
                    let path = action.path.clone();
                    let format = action.format;
                    let quality = action.quality;
                    let tiff_compression = action.tiff_compression;

                    project.file_handler.last_animated = false;

                    let sender = self.io_sender.clone();
                    if self.pending_io_ops == 0 {
                        self.io_ops_start_time = Some(ctx.input(|i| i.time));
                    }
                    self.pending_io_ops += 1;

                    rayon::spawn(move || {
                        match crate::io::encode_prepared_and_write(
                            export_image,
                            &path,
                            format,
                            quality,
                            tiff_compression,
                        ) {
                            Ok(()) => {
                                let _ = sender.send(IoResult::SaveComplete {
                                    project_index,
                                    path,
                                    format,
                                    quality,
                                    tiff_compression,
                                    update_project_path: true,
                                });
                            }
                            Err(e) => {
                                let _ = sender.send(IoResult::SaveFailed {
                                    project_index,
                                    error: format!("{}", e),
                                });
                            }
                        }
                    });
                }
            }
        }

        if self.exit_save_active {
            if save_dialog_confirmed {
                if self.exit_save_queue.is_empty() {
                    self.exit_save_active = false;
                    self.force_exit = true;
                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                } else {
                    let next = self.exit_save_queue.remove(0);
                    self.open_save_as_for_project(next);
                }
            } else if save_dialog_was_open && !self.save_file_dialog.open {
                self.exit_save_queue.clear();
                self.exit_save_active = false;
            }
        }

        self.save_file_dialog.open
            || self.new_file_dialog.open
            || !matches!(self.active_dialog, ActiveDialog::None)
            || self.pending_paste_request.is_some()
    }
}
