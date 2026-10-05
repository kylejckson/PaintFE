impl PaintFEApp {
    fn handle_runtime_modal_flow(&mut self, ctx: &egui::Context) -> bool {
        #[cfg(target_arch = "wasm32")]
        self.show_welcome_popup_window(ctx);

        let settings_window_rect =
            self.settings_window
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
        let bundled_icons_changed =
            self.assets.bundled_icon_style() != self.settings.bundled_icon_style;
        if bundled_icons_changed
            && let Err(e) = self.assets.set_bundled_icon_style(self.settings.bundled_icon_style)
        {
            log_info!("Bundled icon style load failed: {e}");
            self.settings.bundled_icon_style = self.assets.bundled_icon_style();
        }
        if self.settings_window.pending_icon_pack_clear || bundled_icons_changed {
            if self.settings_window.pending_icon_pack_clear {
                self.settings_window.pending_icon_pack_clear = false;
                self.assets.clear_icon_pack();
                self.settings.icon_pack_path.clear();
            }
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
        // ---- Import image: one dialog per dropped image ----
        // Only handle an oversized dialog that was already pending this frame:
        // Enter on "Add as Layer" must not also accept "Expand Canvas".
        let oversized_at_frame_start = self.pending_oversized_import.is_some();
        if !oversized_at_frame_start {
            self.process_import_batch();
            if self.pending_oversized_import.is_some() {
                ctx.request_repaint();
            }
        }
        let mut import_choice: Option<usize> = None;
        if self.pending_oversized_import.is_none()
            && let Some(item) = self.pending_import_queue.first().cloned()
        {
            use crate::ops::dialogs::{DialogAction, action_list, dialog_card_header};
            let total = self.pending_import_queue.len();
            let caption = if total > 1 {
                Some(format!("{} · {} more waiting", item.name, total - 1))
            } else {
                Some(item.name.clone())
            };
            let actions = [
                DialogAction::new(
                    crate::assets::Icon::DialogOpenImage,
                    "Open",
                    "Open the image in a new document.",
                )
                .recommended(),
                DialogAction::new(
                    crate::assets::Icon::DialogAddLayer,
                    "Add as Layer",
                    "Place the image into the current canvas as a new layer.",
                ),
                DialogAction::new(
                    crate::assets::Icon::DialogCancel,
                    "Cancel",
                    "Don't import the image.",
                ),
            ];
            let mut apply_all = self.import_apply_to_all;
            let window_response = crate::ui::polish::window(ctx, "Import Image")
                .title_bar(false)
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .frame(self.theme.dialog_frame())
                .show(ctx, |ui| {
                    ui.set_min_width(380.0);
                    if dialog_card_header(
                        ui,
                        &self.assets,
                        crate::assets::Icon::DialogOpenImage,
                        "Import image",
                    ) {
                        import_choice = Some(2);
                    }
                    if let Some(idx) = action_list(
                        ui,
                        &self.assets,
                        "What would you like to do with this file?",
                        caption.as_deref(),
                        &actions,
                    ) {
                        import_choice = Some(idx);
                    }
                    if total > 1 {
                        ui.add_space(8.0);
                        ui.checkbox(&mut apply_all, "Apply to all remaining");
                    }
                });
            if let Some(response) = window_response {
                self.remember_ui_cursor_rect(response.response.rect);
            }
            self.import_apply_to_all = apply_all;

            // Keyboard: Enter = recommended, Esc = cancel.
            if import_choice.is_none() {
                if ctx.input(|i| i.key_pressed(egui::Key::Enter)) {
                    import_choice = Some(0);
                } else if ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
                    import_choice = Some(2);
                }
            }

            if let Some(choice) = import_choice {
                if self.import_apply_to_all {
                    self.import_batch_choice = Some(choice);
                    self.process_import_batch();
                } else {
                    self.pending_import_queue.remove(0);
                    self.apply_import_choice(&item, choice);
                }
                self.import_apply_to_all = false;
                ctx.request_repaint();
            }
        }

        // ---- Oversized import (image being added as a layer is too big) ----
        let mut oversize_choice: Option<usize> = None;
        if oversized_at_frame_start && let Some(item) = self.pending_oversized_import.clone() {
            use crate::ops::dialogs::{DialogAction, action_list, dialog_card_header};
            let (iw, ih) = (item.width, item.height);
            let (cw, ch) = self
                .projects
                .get(self.active_project_index)
                .map(|p| (p.canvas_state.width, p.canvas_state.height))
                .unwrap_or((iw, ih));
            let actions = [
                DialogAction::new(
                    crate::assets::Icon::DialogExpandCanvas,
                    "Expand Canvas",
                    "Resize canvas to fit the image.",
                )
                .recommended(),
                DialogAction::new(
                    crate::assets::Icon::DialogKeepCanvas,
                    "Keep Canvas",
                    "Place the image as is (may be cropped).",
                ),
                DialogAction::new(
                    crate::assets::Icon::DialogCancel,
                    "Cancel",
                    "Don't import the image.",
                ),
            ];
            let window_response = crate::ui::polish::window(ctx, "Oversized Import")
                .title_bar(false)
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .frame(self.theme.dialog_frame())
                .show(ctx, |ui| {
                    ui.set_min_width(380.0);
                    if dialog_card_header(
                        ui,
                        &self.assets,
                        crate::assets::Icon::DialogOpenImage,
                        "Image is larger than canvas",
                    ) {
                        oversize_choice = Some(2);
                    }
                    let question = format!(
                        "The image is {} \u{D7} {}, which is larger than the current canvas ({} \u{D7} {}). How would you like to proceed?",
                        iw, ih, cw, ch
                    );
                    if let Some(idx) = action_list(ui, &self.assets, &question, None, &actions) {
                        oversize_choice = Some(idx);
                    }
                });
            if let Some(response) = window_response {
                self.remember_ui_cursor_rect(response.response.rect);
            }
            if oversize_choice.is_none() {
                if ctx.input(|i| i.key_pressed(egui::Key::Enter)) {
                    oversize_choice = Some(0);
                } else if ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
                    oversize_choice = Some(2);
                }
            }
            if let Some(choice) = oversize_choice {
                self.pending_oversized_import = None;
                ctx.request_repaint();
                match choice {
                    0 => {
                        if let Ok(img) = image::load_from_memory(&item.bytes) {
                            self.import_image_as_layer(&img.to_rgba8(), &item.name, true);
                        }
                    }
                    1 => {
                        if let Ok(img) = image::load_from_memory(&item.bytes) {
                            self.import_image_as_layer(&img.to_rgba8(), &item.name, false);
                        }
                    }
                    _ => {}
                }
            }
        } else if let Some(req) = self.pending_paste_request.as_ref() {
            use crate::ops::dialogs::{DialogAction, action_list, dialog_card_header};
            let (iw, ih) = (req.image.width(), req.image.height());
            let (cw, ch) = self
                .projects
                .get(self.active_project_index)
                .map(|p| (p.canvas_state.width, p.canvas_state.height))
                .unwrap_or((iw, ih));
            let mut choice: Option<usize> = None;
            let actions = [
                DialogAction::new(
                    crate::assets::Icon::DialogExpandCanvas,
                    "Expand Canvas",
                    "Resize canvas to fit the pasted image.",
                )
                .recommended(),
                DialogAction::new(
                    crate::assets::Icon::DialogKeepCanvas,
                    "Keep Canvas",
                    "Paste the image as is (may be cropped).",
                ),
                DialogAction::new(
                    crate::assets::Icon::DialogCancel,
                    "Cancel",
                    "Don't paste the image.",
                ),
            ];
            let window_response = crate::ui::polish::window(ctx, "Paste Image")
                .title_bar(false)
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .frame(self.theme.dialog_frame())
                .show(ctx, |ui| {
                    ui.set_min_width(380.0);
                    if dialog_card_header(
                        ui,
                        &self.assets,
                        crate::assets::Icon::DialogOpenImage,
                        "Image is larger than canvas",
                    ) {
                        choice = Some(2);
                    }
                    let question = format!(
                        "The clipboard image is {} \u{D7} {}, which is larger than the current canvas ({} \u{D7} {}). How would you like to proceed?",
                        iw, ih, cw, ch
                    );
                    if let Some(idx) = action_list(ui, &self.assets, &question, None, &actions) {
                        choice = Some(idx);
                    }
                });
            if let Some(response) = window_response {
                self.remember_ui_cursor_rect(response.response.rect);
            }
            if choice.is_none() {
                if ctx.input(|i| i.key_pressed(egui::Key::Enter)) {
                    choice = Some(0);
                } else if ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
                    choice = Some(2);
                }
            }
            match choice {
                Some(0) => {
                    if let Some(request) = self.pending_paste_request.take() {
                        self.apply_pending_paste_request(request, true);
                    }
                }
                Some(1) => {
                    if let Some(request) = self.pending_paste_request.take() {
                        self.apply_pending_paste_request(request, false);
                    }
                }
                Some(_) => {
                    self.pending_paste_request = None;
                }
                None => {}
            }
        }

        if let Some(close_idx) = self.pending_close_index {
            use crate::ops::dialogs::{ConfirmButton, confirm_row, dialog_card_header};
            let name = self
                .projects
                .get(close_idx)
                .map(|p| p.name.clone())
                .unwrap_or_default();
            let mut choice: Option<usize> = None;
            let buttons = [
                ConfirmButton::new("Save")
                    .icon(crate::assets::Icon::DialogSave)
                    .primary(),
                ConfirmButton::new("Don't Save"),
                ConfirmButton::new("Cancel"),
            ];
            let window_response = crate::ui::polish::window(ctx, "Unsaved Changes")
                .title_bar(false)
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .frame(self.theme.dialog_frame())
                .show(ctx, |ui| {
                    ui.set_min_width(380.0);
                    if dialog_card_header(
                        ui,
                        &self.assets,
                        crate::assets::Icon::DialogUnsavedWarning,
                        "Unsaved changes",
                    ) {
                        choice = Some(2);
                    }
                    ui.label(format!("\u{201C}{}\u{201D} has unsaved changes.", name));
                    ui.add_space(2.0);
                    ui.label(egui::RichText::new("Do you want to save before closing?").weak());
                    if let Some(idx) = confirm_row(ui, &self.assets, &buttons) {
                        choice = Some(idx);
                    }
                });
            if let Some(response) = window_response {
                self.remember_ui_cursor_rect(response.response.rect);
            }
            // Keyboard: Enter = Save, Esc = Cancel.
            if choice.is_none() {
                if ctx.input(|i| i.key_pressed(egui::Key::Enter)) {
                    choice = Some(0);
                } else if ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
                    choice = Some(2);
                }
            }
            match choice {
                Some(0) => {
                    self.open_save_as_for_project(close_idx);
                    self.pending_close_index = None;
                }
                Some(1) => {
                    self.pending_close_index = None;
                    self.force_close_project(close_idx);
                }
                Some(_) => {
                    self.pending_close_index = None;
                }
                None => {}
            }
        }

        if self.pending_exit {
            use crate::ops::dialogs::{ConfirmButton, confirm_row, dialog_card_header};
            let dirty_projects: Vec<String> = self
                .projects
                .iter()
                .filter(|p| p.is_dirty)
                .map(|p| p.name.clone())
                .collect();
            if dirty_projects.is_empty() {
                self.pending_exit = false;
                self.force_exit = true;
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            } else {
                let mut choice: Option<usize> = None;
                let buttons = [
                    ConfirmButton::new("Save")
                        .icon(crate::assets::Icon::DialogSave)
                        .primary(),
                    ConfirmButton::new("Don't Save"),
                    ConfirmButton::new("Cancel"),
                ];
                let window_response = crate::ui::polish::window(ctx, "Exit PaintFE")
                    .title_bar(false)
                    .collapsible(false)
                    .resizable(false)
                    .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                    .frame(self.theme.dialog_frame())
                    .show(ctx, |ui| {
                        ui.set_min_width(380.0);
                        if dialog_card_header(
                            ui,
                            &self.assets,
                            crate::assets::Icon::DialogUnsavedWarning,
                            "Unsaved changes",
                        ) {
                            choice = Some(2);
                        }
                        if dirty_projects.len() == 1 {
                            ui.label(format!(
                                "\u{201C}{}\u{201D} has unsaved changes.",
                                dirty_projects[0]
                            ));
                        } else {
                            const SHOW_MAX: usize = 4;
                            ui.label(format!(
                                "{} projects have unsaved changes.",
                                dirty_projects.len()
                            ));
                            ui.add_space(4.0);
                            for name in dirty_projects.iter().take(SHOW_MAX) {
                                ui.label(format!("\u{2022}  {}", name));
                            }
                            let overflow = dirty_projects.len().saturating_sub(SHOW_MAX);
                            if overflow > 0 {
                                ui.label(
                                    egui::RichText::new(format!("({} more)…", overflow))
                                        .weak()
                                        .italics(),
                                );
                            }
                        }
                        ui.add_space(2.0);
                        ui.label(egui::RichText::new("Do you want to save before exiting?").weak());
                        if let Some(idx) = confirm_row(ui, &self.assets, &buttons) {
                            choice = Some(idx);
                        }
                    });
                if let Some(response) = window_response {
                    self.remember_ui_cursor_rect(response.response.rect);
                }
                // Keyboard: Enter = Save, Esc = Cancel.
                if choice.is_none() {
                    if ctx.input(|i| i.key_pressed(egui::Key::Enter)) {
                        choice = Some(0);
                    } else if ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
                        choice = Some(2);
                    }
                }
                match choice {
                    Some(0) => {
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
                    Some(1) => {
                        self.pending_exit = false;
                        self.force_exit = true;
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                    Some(_) => {
                        self.pending_exit = false;
                    }
                    None => {}
                }
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

                    crate::par_compat::spawn(move || {
                        match crate::io::write_pfe(&pfe_data, &path) {
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
            || !self.pending_import_queue.is_empty()
            || self.pending_oversized_import.is_some()
            || self.pending_exit
            || self.pending_close_index.is_some()
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
            egui::Color32::from_rgba_premultiplied(scale(c.r()), scale(c.g()), scale(c.b()), alpha)
        };

        crate::ui::polish::window(ctx, "welcome_popup")
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
                        egui::RichText::new("PaintFE")
                            .strong()
                            .size(24.0)
                            .color(colors.text_color),
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
