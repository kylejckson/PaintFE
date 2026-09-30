impl PaintFEApp {
    fn handle_runtime_modal_flow(&mut self, ctx: &egui::Context) -> bool {
        #[cfg(target_arch = "wasm32")]
        self.show_welcome_popup_window(ctx);

        let settings_window_rect = self
            .settings_window
            .show(ctx, &mut self.settings, &mut self.theme, &self.assets);
        // Keep the runtime stroke stabilization in sync with the settings
        // slider (the slider writes AppSettings only).
        self.tools_panel.stroke_stabilization = self.settings.persisted_stroke_stabilization;

        // Register the settings window as input-blocking: without this the
        // canvas behind it keeps receiving wheel zoom / clicks / strokes while
        // the pointer is over the window.
        if let Some(rect) = settings_window_rect {
            self.remember_ui_cursor_rect(rect);
        }

        // Pixel Art preset: also update the live tool state (the settings
        // window only writes AppSettings).
        if self.settings_window.pending_pixel_art_preset {
            self.settings_window.pending_pixel_art_preset = false;
            self.tools_panel.properties.hardness = 1.0;
            self.tools_panel.stroke_stabilization = 0.0;
        }

        // Icon pack changes from the Settings window (needs &mut Assets).
        let dark = matches!(self.theme.mode, crate::theme::ThemeMode::Dark);
        if let Some(path) = self.settings_window.pending_icon_pack_load.take() {
            self.assets
                .set_icon_pack_invert_mismatch(self.settings.icon_pack_invert_mismatch);
            match self.assets.load_icon_pack(&path) {
                Ok(name) => {
                    log_info!("Icon pack loaded: {name} ({})", path.display());
                    self.settings.icon_pack_path = path.display().to_string();
                }
                Err(e) => {
                    log_info!("Icon pack load failed: {e}");
                }
            }
            self.assets.reload_icons(ctx, dark);
            self.settings.save();
        }
        if self.settings_window.pending_icon_pack_clear {
            self.settings_window.pending_icon_pack_clear = false;
            self.assets.clear_icon_pack();
            self.settings.icon_pack_path.clear();
            self.assets.reload_icons(ctx, dark);
            self.settings.save();
        }
        if self.settings_window.pending_icon_pack_reload {
            self.settings_window.pending_icon_pack_reload = false;
            self.assets
                .set_icon_pack_invert_mismatch(self.settings.icon_pack_invert_mismatch);
            self.assets.reload_icons(ctx, dark);
        }

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
            let unsaved_changes_window = egui::Window::new("Unsaved Changes")
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
            if let Some(response) = unsaved_changes_window {
                self.remember_ui_cursor_rect(response.response.rect);
            }
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

            let exit_window = egui::Window::new("Exit PaintFE")
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
            if let Some(response) = exit_window {
                self.remember_ui_cursor_rect(response.response.rect);
            }

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

                    crate::par_compat::spawn(move || match crate::io::write_pfe(&pfe_data, &path) {
                        Ok(()) => {
                            let _ = sender.send(IoResult::SaveComplete {
                                project_index,
                                path,
                                format: SaveFormat::Pfe,
                                quality: 100,
                                webp_lossless: true,
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
                    let webp_lossless = action.webp_lossless;
                    let tiff_compression = action.tiff_compression;
                    let fps = action.animation_fps;
                    let gif_colors = action.gif_colors;
                    let gif_dither = action.gif_dither;
                    let frame_modes: Vec<_> = project
                        .canvas_state
                        .layers
                        .iter()
                        .map(|l| l.webp_frame_compression)
                        .collect();

                    project.file_handler.last_animated = true;
                    project.file_handler.last_webp_lossless = webp_lossless;
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

                    crate::par_compat::spawn(move || {
                        let result = match format {
                            SaveFormat::Gif => crate::io::encode_animated_gif(
                                &frames, fps, gif_colors, gif_dither, &path,
                            ),
                            SaveFormat::Png => crate::io::encode_animated_png(&frames, fps, &path),
                            SaveFormat::Webp => crate::io::encode_animated_webp(
                                &frames,
                                &frame_modes,
                                fps,
                                quality,
                                &path,
                            ),
                            _ => Err("Format does not support animation".to_string()),
                        };
                        match result {
                            Ok(()) => {
                                let _ = sender.send(IoResult::SaveComplete {
                                    project_index,
                                    path,
                                    format,
                                    quality,
                                    webp_lossless,
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
                    let webp_lossless = action.webp_lossless;
                    let tiff_compression = action.tiff_compression;

                    project.file_handler.last_animated = false;
                    project.file_handler.last_webp_lossless = webp_lossless;

                    let sender = self.io_sender.clone();
                    if self.pending_io_ops == 0 {
                        self.io_ops_start_time = Some(ctx.input(|i| i.time));
                    }
                    self.pending_io_ops += 1;

                    crate::par_compat::spawn(move || {
                        match crate::io::encode_prepared_and_write(
                            export_image,
                            &path,
                            format,
                            quality,
                            tiff_compression,
                            webp_lossless,
                        ) {
                            Ok(()) => {
                                let _ = sender.send(IoResult::SaveComplete {
                                    project_index,
                                    path,
                                    format,
                                    quality,
                                    webp_lossless,
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

        #[cfg(target_arch = "wasm32")]
        let welcome_open = self.show_welcome_popup;
        #[cfg(not(target_arch = "wasm32"))]
        let welcome_open = false;

        welcome_open
            || self.save_file_dialog.open
            || self.new_file_dialog.open
            || !matches!(self.active_dialog, ActiveDialog::None)
            || self.pending_paste_request.is_some()
    }

    /// First-run welcome / beta-disclaimer popup (web only). Shown once per
    /// browser; dismissal is persisted to localStorage so it never shows
    /// again on that browser, even across page reloads.
    #[cfg(target_arch = "wasm32")]
    fn show_welcome_popup_window(&mut self, ctx: &egui::Context) {
        if !self.show_welcome_popup {
            return;
        }
        let mut dismiss = false;
        let colors = &self.theme;
        let is_mobile = crate::web_storage::is_mobile_device();
        // Translucent tint of a color (for subtle highlight backgrounds),
        // independent of the color's own alpha.
        let tint = |c: egui::Color32, alpha: u8| {
            let scale = |channel: u8| (channel as u16 * alpha as u16 / 255) as u8;
            egui::Color32::from_rgba_premultiplied(
                scale(c.r()),
                scale(c.g()),
                scale(c.b()),
                alpha,
            )
        };

        egui::Window::new("welcome_popup")
            .title_bar(false)
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
            .frame(
                egui::Frame::window(&ctx.global_style())
                    .fill(colors.window_bg)
                    .corner_radius(12.0),
            )
            .show(ctx, |ui| {
                ui.set_width(400.0);
                ui.spacing_mut().item_spacing.y = 0.0;

                // -- Header ------------------------------------------------
                ui.vertical_centered(|ui| {
                    ui.add_space(20.0);
                    ui.label(
                        egui::RichText::new("PaintFE").strong().size(24.0).color(colors.text_color),
                    );
                    ui.add_space(4.0);
                    egui::Frame::NONE
                        .fill(tint(colors.accent, 40))
                        .corner_radius(10.0)
                        .inner_margin(egui::Margin::symmetric(10, 3))
                        .show(ui, |ui| {
                            ui.label(
                                egui::RichText::new("WEB • VERSION 1.0 • EXPERIMENTAL BETA")
                                    .size(10.5)
                                    .strong()
                                    .color(colors.accent_strong),
                            );
                        });
                    ui.add_space(16.0);
                });

                ui.separator();
                ui.add_space(14.0);

                // -- Body ----------------------------------------------------
                ui.label(
                    egui::RichText::new(
                        "Welcome! This is a beta of PaintFE running entirely in your \
                         browser. Everything you do stays on this device; nothing is \
                         uploaded anywhere.",
                    )
                    .color(colors.text_color),
                );
                ui.add_space(10.0);
                ui.label(
                    egui::RichText::new(
                        "A few things work differently here than on desktop: font \
                         and clipboard access are more limited by browser security, \
                         and a couple of native only features (RAW camera import, \
                         printing through the OS) are unavailable or work \
                         differently. See Settings for details.",
                    )
                    .color(colors.text_muted),
                );

                // -- Mobile notice (only shown if actually on one) -----------
                if is_mobile {
                    ui.add_space(14.0);
                    egui::Frame::NONE
                        .fill(tint(colors.accent, 30))
                        .stroke(egui::Stroke::new(1.0, tint(colors.accent, 110)))
                        .corner_radius(8.0)
                        .inner_margin(egui::Margin::same(10))
                        .show(ui, |ui| {
                            ui.horizontal_wrapped(|ui| {
                                ui.label(
                                    egui::RichText::new("Heads up:")
                                        .strong()
                                        .color(colors.accent_strong),
                                );
                                ui.label(
                                    egui::RichText::new(
                                        "PaintFE - Web is built for a desktop experience \
                                         (mouse and keyboard, larger screen) and may not \
                                         work well on a phone or tablet. Mobile support \
                                         isn't planned.",
                                    )
                                    .color(colors.text_color),
                                );
                            });
                        });
                }

                // -- Desktop upsell -------------------------------------------
                ui.add_space(14.0);
                egui::Frame::NONE
                    .fill(colors.bg2)
                    .corner_radius(8.0)
                    .inner_margin(egui::Margin::same(12))
                    .show(ui, |ui| {
                        ui.label(
                            egui::RichText::new("Want the full experience?")
                                .strong()
                                .color(colors.text_color),
                        );
                        ui.add_space(3.0);
                        ui.label(
                            egui::RichText::new(
                                "The desktop app has full feature access, better \
                                 performance, and GPU acceleration for filters and \
                                 effects.",
                            )
                            .small()
                            .color(colors.text_muted),
                        );
                        ui.add_space(8.0);
                        let dl_btn = egui::Button::new(
                            egui::RichText::new("Download Desktop App")
                                .strong()
                                .color(egui::Color32::WHITE),
                        )
                        .fill(colors.accent)
                        .corner_radius(6.0);
                        if ui.add(dl_btn).clicked() {
                            crate::ops::open_url_in_new_tab("https://paintfe.com/download.html");
                        }
                    });

                ui.add_space(16.0);
                ui.horizontal(|ui| {
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.button("Got it").clicked() {
                            dismiss = true;
                        }
                    });
                });
                ui.add_space(4.0);
            });
        if dismiss {
            self.show_welcome_popup = false;
            crate::web_storage::mark_welcome_seen();
        }
    }
}
