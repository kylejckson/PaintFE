impl PaintFEApp {
    fn update_runtime_lifecycle_async(&mut self, ctx: &egui::Context) {
        // --- Dynamic window title: "PaintFE - <project name>[*]" ---
        {
            let title = if let Some(project) = self.projects.get(self.active_project_index) {
                let dirty = if project.is_dirty { "*" } else { "" };
                format!("PaintFE - {}{}", project.name, dirty)
            } else {
                "PaintFE".to_string()
            };
            ctx.send_viewport_cmd(egui::ViewportCommand::Title(title));
        }

        // --- Intercept OS window-close button ---
        // If the user hasn't yet confirmed the exit dialog, cancel the close and show it.
        if ctx.input(|i| i.viewport().close_requested()) {
            let current_time = ctx.input(|i| i.time);
            self.persist_window_state_if_changed(current_time, true);
            if self.force_exit {
                // Already confirmed — let eframe proceed with the close.
            } else if self.settings.confirm_on_exit {
                let dirty_count = self.projects.iter().filter(|p| p.is_dirty).count();
                if dirty_count > 0 {
                    ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
                    self.pending_exit = true;
                }
            }
        }

        // --- Sync icon colors with theme (catches settings window, menu toggle, etc.) ---
        let is_dark = matches!(self.theme.mode, crate::theme::ThemeMode::Dark);
        self.assets.update_theme(ctx, is_dark);

        // --- Sync OS window chrome (title bar) with app theme on Windows/macOS ---
        let system_theme = if is_dark {
            egui::SystemTheme::Dark
        } else {
            egui::SystemTheme::Light
        };
        ctx.send_viewport_cmd(egui::ViewportCommand::SetTheme(system_theme));

        // --- First frame startup file processing ---
        if self.first_frame {
            self.first_frame = false;

            if self.settings.persist_window_maximized {
                ctx.send_viewport_cmd(egui::ViewportCommand::Maximized(true));
            }

            // Open files passed as positional arguments (e.g. right-click → "Open with PaintFE")
            let files = std::mem::take(&mut self.pending_startup_files);
            let current_time = ctx.input(|i| i.time);
            for path in files {
                self.open_file_by_path(path, current_time);
            }
        }

        // --- Single-instance IPC: open files sent from other PaintFE invocations ---
        {
            let current_time = ctx.input(|i| i.time);
            while let Ok(path) = self.ipc_receiver.try_recv() {
                self.open_file_by_path(path, current_time);
                // Bring our window to the foreground (the sender already tried via
                // FindWindow, but this handles the case where it didn't have permission).
                ctx.send_viewport_cmd(egui::ViewportCommand::Focus);
            }
        }

        // --- Auto-save tick ---
        // Saves every open project as a .autosave.pfe in the platform data dir.
        // Controlled by settings.auto_save_minutes (0 = disabled).
        {
            let interval_secs = (self.settings.auto_save_minutes as u64) * 60;
            if interval_secs > 0 && self.last_autosave.elapsed().as_secs() >= interval_secs {
                self.last_autosave = std::time::Instant::now();
                if let Some(dir) = crate::io::autosave_dir() {
                    let _ = std::fs::create_dir_all(&dir);
                    for project in &self.projects {
                        // Sanitize project name into a safe filename component
                        let safe_name: String = project
                            .name
                            .chars()
                            .map(|c| {
                                if c.is_alphanumeric() || c == '-' || c == '_' {
                                    c
                                } else {
                                    '_'
                                }
                            })
                            .collect();
                        let path = dir.join(format!("{}.autosave.pfe", safe_name));
                        let pfe_data = crate::io::build_pfe(&project.canvas_state);
                        let proj_name = project.name.clone();
                        rayon::spawn(move || match crate::io::write_pfe(&pfe_data, &path) {
                            Ok(()) => {
                                crate::logger::write(
                                    "INFO",
                                    &format!(
                                        "Auto-save OK  \"{}\"  →  {}",
                                        proj_name,
                                        path.display()
                                    ),
                                );
                            }
                            Err(e) => {
                                crate::logger::write(
                                    "ERROR",
                                    &format!("Auto-save FAILED for \"{}\": {}", proj_name, e),
                                );
                            }
                        });
                    }
                }
            }
        }

        // --- Poll async filter results ---
        while let Ok(result) = self.filter_receiver.try_recv() {
            self.pending_filter_jobs = self.pending_filter_jobs.saturating_sub(1);
            if self.pending_filter_jobs == 0 {
                self.filter_ops_start_time = None;
                self.filter_status_description.clear();
            }
            // Discard stale live-preview results (token mismatch = superseded by newer job)
            if result.preview_token != 0 && result.preview_token != self.preview_job_token {
                continue;
            }
            if result.project_index < self.projects.len()
                && let Some(project) = self.projects.get_mut(result.project_index)
            {
                let idx = result.layer_idx;
                if idx < project.canvas_state.layers.len() {
                    if result.preview_token != 0 {
                        project.canvas_state.layers[idx].pixels = result.result_pixels;
                        project.canvas_state.mark_dirty(None);
                        continue;
                    }
                    // Swap original back for "before" snapshot, then install result
                    project.canvas_state.layers[idx].pixels = result.original_pixels;
                    let mut cmd = SingleLayerSnapshotCommand::new_for_layer(
                        result.description,
                        &project.canvas_state,
                        idx,
                    );
                    project.canvas_state.layers[idx].pixels = result.result_pixels;
                    cmd.set_after(&project.canvas_state);
                    project.history.push(Box::new(cmd));
                    project.canvas_state.mark_dirty(None);
                    project.mark_dirty();
                }
            }
        }

        // --- Poll async canvas-wide operation results (resize image/canvas) ---
        while let Ok(result) = self.canvas_op_receiver.try_recv() {
            self.pending_filter_jobs = self.pending_filter_jobs.saturating_sub(1);
            if self.pending_filter_jobs == 0 {
                self.filter_ops_start_time = None;
                self.filter_status_description.clear();
            }
            if result.project_index < self.projects.len()
                && let Some(project) = self.projects.get_mut(result.project_index)
            {
                let state = &mut project.canvas_state;
                // Restore before-state so SnapshotCommand captures it correctly
                result.before.restore_into(state);
                let mut cmd = SnapshotCommand::new(result.description, state);
                // Apply result layers
                for (i, tiled) in result.result_layers.into_iter().enumerate() {
                    if i < state.layers.len() {
                        state.layers[i].pixels = tiled;
                        state.layers[i].invalidate_lod();
                        state.layers[i].gpu_generation += 1;
                    }
                }
                state.width = result.new_width;
                state.height = result.new_height;
                state.composite_cache = None;
                state.clear_preview_state();
                state.mark_dirty(None);
                cmd.set_after(state);
                project.history.push(Box::new(cmd));
                project.mark_dirty();
            }
        }

        // Request a repaint while filter jobs are pending so polling stays active
        if self.pending_filter_jobs > 0 {
            ctx.request_repaint();
        }

        // --- Poll async script results ---
        while let Ok(msg) = self.script_receiver.try_recv() {
            match msg {
                ScriptMessage::Completed {
                    project_index,
                    layer_idx,
                    original_pixels,
                    result_pixels,
                    width,
                    height,
                    console_output,
                    elapsed_ms,
                    canvas_ops,
                } => {
                    self.script_editor.is_running = false;
                    self.script_editor.progress = None;
                    // Clear spinner status
                    self.pending_filter_jobs = self.pending_filter_jobs.saturating_sub(1);
                    if self.pending_filter_jobs == 0 {
                        self.filter_ops_start_time = None;
                        self.filter_status_description.clear();
                    }
                    for line in console_output {
                        self.script_editor.add_console_line(
                            line,
                            crate::components::script_editor::ConsoleLineKind::Output,
                        );
                    }
                    self.script_editor.add_console_line(
                        format!("Completed in {}ms", elapsed_ms),
                        crate::components::script_editor::ConsoleLineKind::Info,
                    );
                    // Apply result to canvas with undo — route to the project that spawned the script
                    if project_index < self.projects.len()
                        && let Some(project) = self.projects.get_mut(project_index)
                        && layer_idx < project.canvas_state.layers.len()
                    {
                        // Restore original pixels so snapshot captures the true before-state
                        project.canvas_state.layers[layer_idx].pixels = original_pixels;

                        if canvas_ops.is_empty() {
                            // Layer-only script: lightweight single-layer snapshot
                            let mut cmd = SingleLayerSnapshotCommand::new_for_layer(
                                "Script".to_string(),
                                &project.canvas_state,
                                layer_idx,
                            );
                            let result_tiled =
                                TiledImage::from_raw_rgba(width, height, &result_pixels);
                            project.canvas_state.layers[layer_idx].pixels = result_tiled;
                            cmd.set_after(&project.canvas_state);
                            project.history.push(Box::new(cmd));
                        } else {
                            // Canvas-wide transform: full snapshot (dimensions may change)
                            let mut cmd =
                                SnapshotCommand::new("Script".to_string(), &project.canvas_state);

                            // Apply result to active layer
                            let result_tiled =
                                TiledImage::from_raw_rgba(width, height, &result_pixels);
                            project.canvas_state.layers[layer_idx].pixels = result_tiled;

                            // Replay canvas ops on all other layers via shared helper
                            // (also updates state.width / state.height to final dims)
                            apply_canvas_ops(&mut project.canvas_state, layer_idx, &canvas_ops);

                            project.canvas_state.composite_cache = None;
                            project.canvas_state.clear_preview_state();
                            cmd.set_after(&project.canvas_state);
                            project.history.push(Box::new(cmd));
                        }

                        project.canvas_state.mark_dirty(None);
                        project.mark_dirty();
                    }
                    self.script_original_pixels = None;
                }
                ScriptMessage::Error {
                    error,
                    console_output,
                } => {
                    self.script_editor.is_running = false;
                    self.script_editor.progress = None;
                    // Clear spinner status
                    self.pending_filter_jobs = self.pending_filter_jobs.saturating_sub(1);
                    if self.pending_filter_jobs == 0 {
                        self.filter_ops_start_time = None;
                        self.filter_status_description.clear();
                    }
                    for line in console_output {
                        self.script_editor.add_console_line(
                            line,
                            crate::components::script_editor::ConsoleLineKind::Output,
                        );
                    }
                    // Show error with tips
                    let friendly = error.friendly_message();
                    for err_line in friendly.lines() {
                        self.script_editor.add_console_line(
                            err_line.to_string(),
                            crate::components::script_editor::ConsoleLineKind::Error,
                        );
                    }
                    // Restore original layer pixels if preview had modified them
                    if let Some((proj_idx, layer_idx, original)) =
                        self.script_original_pixels.take()
                        && let Some(project) = self.projects.get_mut(proj_idx)
                        && layer_idx < project.canvas_state.layers.len()
                    {
                        project.canvas_state.layers[layer_idx].pixels = original;
                        project.canvas_state.mark_dirty(None);
                    }
                    // Highlight the error line in the code editor
                    self.script_editor.error_line = error.line;
                    // Auto-expand console so user sees the error
                    self.script_editor.console_expanded = true;
                }
                ScriptMessage::ConsoleOutput(line) => {
                    self.script_editor.add_console_line(
                        line,
                        crate::components::script_editor::ConsoleLineKind::Output,
                    );
                }
                ScriptMessage::Progress(p) => {
                    self.script_editor.progress = Some(p);
                }
                ScriptMessage::Preview {
                    project_index,
                    pixels,
                    width,
                    height,
                } => {
                    if self.script_editor.live_preview
                        && let Some(project) = self.projects.get_mut(project_index)
                    {
                        let layer_idx = project.canvas_state.active_layer_index;
                        if layer_idx < project.canvas_state.layers.len() {
                            let preview_tiled = TiledImage::from_raw_rgba(width, height, &pixels);
                            project.canvas_state.layers[layer_idx].pixels = preview_tiled;
                            project.canvas_state.mark_dirty(None);
                        }
                    }
                }
            }
        }
        if self.script_editor.is_running {
            ctx.request_repaint();
        }

        // --- Poll async IO results (image load / save) ---
        while let Ok(result) = self.io_receiver.try_recv() {
            self.pending_io_ops = self.pending_io_ops.saturating_sub(1);
            if self.pending_io_ops == 0 {
                self.io_ops_start_time = None;
            }
            match result {
                IoResult::ImageLoaded {
                    tiled,
                    width,
                    height,
                    path,
                    format,
                } => {
                    self.pending_open_paths
                        .remove(&Self::normalize_open_path(&path));
                    let mut canvas_state = CanvasState::new(width, height);
                    if let Some(layer) = canvas_state.layers.first_mut() {
                        layer.pixels = tiled;
                    }
                    canvas_state.composite_cache = None;
                    canvas_state.mark_dirty(None);

                    let file_handler = FileHandler {
                        current_path: Some(path.clone()),
                        last_format: format,
                        last_quality: 90,
                        last_tiff_compression: TiffCompression::None,
                        last_animated: false,
                        last_animation_fps: 10.0,
                        last_gif_colors: 256,
                        last_gif_dither: true,
                    };
                    let project = Project::from_file(path, canvas_state, file_handler);
                    self.projects.push(project);
                    self.persist_active_project_view();
                    self.active_project_index = self.projects.len() - 1;
                    self.restore_active_project_view();
                    // Clear GPU layer cache for the new project
                    self.canvas.gpu_clear_layers();
                    self.maybe_close_initial_blank();
                }
                IoResult::LoadFailed { path, error } => {
                    if let Some(path) = path {
                        self.pending_open_paths
                            .remove(&Self::normalize_open_path(&path));
                    }
                    log_info!("FileIO: load failed — {}", error);
                    eprintln!("Failed to open image: {}", error);
                }
                IoResult::SaveComplete {
                    project_index,
                    path,
                    format,
                    quality,
                    tiff_compression,
                    update_project_path,
                } => {
                    log_info!(
                        "FileIO: save complete — project={} path={:?} format={:?}",
                        project_index,
                        path,
                        format
                    );
                    if let Some(project) = self.projects.get_mut(project_index) {
                        project.file_handler.current_path = Some(path.clone());
                        project.file_handler.last_format = format;
                        project.file_handler.last_quality = quality;
                        project.file_handler.last_tiff_compression = tiff_compression;
                        if update_project_path {
                            project.path = Some(path);
                            project.update_name_from_path();
                        }
                        project.mark_clean();
                    }
                }
                IoResult::SaveFailed {
                    project_index: _,
                    error,
                } => {
                    log_info!("FileIO: save failed — {}", error);
                    eprintln!("Failed to save: {}", error);
                }
                IoResult::AnimatedLoaded {
                    tiled,
                    width,
                    height,
                    path,
                    format,
                    fps,
                    frame_count: _,
                } => {
                    self.pending_open_paths
                        .remove(&Self::normalize_open_path(&path));
                    log_info!(
                        "FileIO: animated loaded — path={:?} format={:?} fps={}",
                        path,
                        format,
                        fps
                    );
                    let mut canvas_state = CanvasState::new(width, height);
                    if let Some(layer) = canvas_state.layers.first_mut() {
                        layer.pixels = tiled;
                        layer.name = "Frame 1".to_string();
                    }
                    canvas_state.composite_cache = None;
                    canvas_state.mark_dirty(None);

                    let file_handler = FileHandler {
                        current_path: Some(path.clone()),
                        last_format: format,
                        last_quality: 90,
                        last_tiff_compression: TiffCompression::None,
                        last_animated: true,
                        last_animation_fps: fps,
                        last_gif_colors: 256,
                        last_gif_dither: true,
                    };
                    let mut project = Project::from_file(path, canvas_state, file_handler);
                    project.was_animated = true;
                    project.animation_fps = fps;
                    self.projects.push(project);
                    self.persist_active_project_view();
                    self.active_project_index = self.projects.len() - 1;
                    self.restore_active_project_view();
                    self.canvas.gpu_clear_layers();
                    self.maybe_close_initial_blank();
                }
                IoResult::AnimatedFramesLoaded { path, frames } => {
                    // Find the project that was created by the AnimatedLoaded handler
                    if let Some(project) = self
                        .projects
                        .iter_mut()
                        .find(|p| p.path.as_deref() == Some(path.as_path()))
                    {
                        for (tiled, name) in frames {
                            let layer = Layer {
                                name,
                                visible: true,
                                folder_id: None,
                                opacity: 1.0,
                                blend_mode: BlendMode::Normal,
                                pixels: tiled,
                                mask: None,
                                mask_enabled: true,
                                lod_cache: None,
                                gpu_generation: 0,
                                content: crate::canvas::LayerContent::Raster,
                                pixel_format: crate::canvas::PixelFormat::RgbaU8,
                                hdr_metadata: crate::canvas::HdrMetadata::default(),
                                source_metadata: crate::canvas::ImageMetadata::default(),
                                deep_pixels: None,
                            };
                            project.canvas_state.layers.push(layer);
                        }
                        project.canvas_state.composite_cache = None;
                        project.canvas_state.mark_dirty(None);
                        self.canvas.gpu_clear_layers();
                    }
                }
                IoResult::PfeLoaded {
                    mut canvas_state,
                    path,
                } => {
                    self.pending_open_paths
                        .remove(&Self::normalize_open_path(&path));
                    log_info!("FileIO: pfe loaded — path={:?}", path);
                    canvas_state.composite_cache = None;
                    canvas_state.mark_dirty(None);

                    let file_handler = FileHandler {
                        current_path: Some(path.clone()),
                        last_format: SaveFormat::Pfe,
                        last_quality: 90,
                        last_tiff_compression: TiffCompression::None,
                        last_animated: false,
                        last_animation_fps: 10.0,
                        last_gif_colors: 256,
                        last_gif_dither: true,
                    };
                    let project = Project::from_file(path, canvas_state, file_handler);
                    self.projects.push(project);
                    self.persist_active_project_view();
                    self.active_project_index = self.projects.len() - 1;
                    self.restore_active_project_view();
                    self.canvas.gpu_clear_layers();
                    self.maybe_close_initial_blank();
                }
            }
        }
        if self.pending_io_ops > 0 {
            ctx.request_repaint();
        }

        // --- Drag-and-Drop: open dropped image files as new projects ---
        {
            let shortcut_paste_present = ctx.input(|i| {
                i.events.iter().any(|e| {
                    matches!(
                        e,
                        egui::Event::Key {
                            key: egui::Key::V,
                            pressed: true,
                            modifiers,
                            ..
                        } if modifiers.command || modifiers.ctrl
                    )
                })
            });
            if !shortcut_paste_present {
                let dropped: Vec<egui::DroppedFile> = ctx.input(|i| i.raw.dropped_files.clone());
                for file in dropped {
                    if let Some(path) = file.path.clone() {
                        if path.is_file() {
                            self.open_file_by_path(path, ctx.input(|i| i.time));
                            continue;
                        }

                        // Some Linux/Wayland stacks can surface a dropped URI-like
                        // string in the path field (e.g. file:///...).
                        let parsed = Self::parse_file_uri_list(&path.to_string_lossy());
                        if !parsed.is_empty() {
                            for path in parsed {
                                self.open_file_by_path(path, ctx.input(|i| i.time));
                            }
                            continue;
                        }
                    }

                    if !file.name.is_empty() {
                        let parsed = Self::parse_file_uri_list(&file.name);
                        if !parsed.is_empty() {
                            for path in parsed {
                                self.open_file_by_path(path, ctx.input(|i| i.time));
                            }
                            continue;
                        }

                        let named_path = PathBuf::from(file.name.clone());
                        if named_path.is_file() {
                            self.open_file_by_path(named_path, ctx.input(|i| i.time));
                            continue;
                        }
                    }

                    if let Some(bytes) = file.bytes.as_ref() {
                        let name_hint = if file.name.is_empty() {
                            None
                        } else {
                            Some(file.name.clone())
                        };
                        self.open_image_from_bytes(bytes.as_ref(), name_hint);
                    }
                }
            }
        }

        // Some Linux/Wayland desktop flows surface file drags as text/uri-list
        // paste events instead of dropped file paths.
        self.handle_file_uri_paste_events(ctx);

        // Determine if a modal dialog is open — block all shortcuts and canvas interaction.
        let modal_open = self.save_file_dialog.open
            || self.new_file_dialog.open
            || !matches!(self.active_dialog, ActiveDialog::None)
            || self.pending_paste_request.is_some();

        let _global_probe = ctx.input(|i| {
            let cmd = i.modifiers.ctrl || i.modifiers.command;
            let c_down = i.key_down(egui::Key::C);
            let x_down = i.key_down(egui::Key::X);
            let v_down = i.key_down(egui::Key::V);
            let enter_down = i.key_down(egui::Key::Enter);
            let escape_down = i.key_down(egui::Key::Escape);
            let events_len = i.events.len();
            let copy_evt = i.events.iter().any(|e| matches!(e, egui::Event::Copy));
            let cut_evt = i.events.iter().any(|e| matches!(e, egui::Event::Cut));
            let paste_evt = i.events.iter().any(|e| matches!(e, egui::Event::Paste(_)));
            (
                cmd,
                c_down,
                x_down,
                v_down,
                enter_down,
                escape_down,
                events_len,
                copy_evt,
                cut_evt,
                paste_evt,
                i.pointer.hover_pos().is_some(),
            )
        });
        let _vk_probe = crate::windows_key_probe::snapshot();
        let _linux_probe = crate::linux_key_probe::snapshot();

        // Pre-claim canvas focus when the text tool is actively editing text.
        // This must happen BEFORE any context-bar or panel widgets render (they render
        // before the CentralPanel canvas).  Without this, DragValues in the text tool
        // options bar (letter spacing, scale, etc.) can enter text-edit mode and consume
        // Event::Text keystokes before handle_input ever sees them.
        if !modal_open
            && self.tools_panel.text_state.is_editing
            && !self.tools_panel.text_state.font_popup_open
            && let Some(canvas_id) = self.canvas.canvas_widget_id
        {
            ctx.memory_mut(|m| m.request_focus(canvas_id));
        }

        // Handle scroll wheel zoom — only when mouse is over the canvas and NOT over a widget

        let mut should_zoom = false;
        let mut zoom_amount = 0.0;

        // Check if any floating window/widget is under the pointer.
        // On Wayland with a tablet, also check for active Touch events since
        // is_pointer_over_egui() relies on interact_pos() which isn't updated by Touch.
        let pointer_over_widget = ctx.is_pointer_over_egui()
            || ctx.input(|i| {
                i.events.iter().any(|e| {
                    matches!(
                        e,
                        egui::Event::Touch {
                            phase: egui::TouchPhase::Start | egui::TouchPhase::Move,
                            ..
                        }
                    )
                })
            });

        if !modal_open {
            ctx.input_mut(|i| {
                if i.smooth_scroll_delta.y.abs() > 0.1 {
                    let mouse_over_canvas = i.pointer.hover_pos().is_some_and(|pos| {
                        self.canvas
                            .last_canvas_rect
                            .is_some_and(|rect| rect.contains(pos))
                    });
                    if mouse_over_canvas && !pointer_over_widget {
                        should_zoom = true;
                        zoom_amount = i.smooth_scroll_delta.y;
                        i.smooth_scroll_delta.y = 0.0;
                    }
                }
            });
        }

        if should_zoom {
            let zoom_factor = 1.0 + zoom_amount * 0.005;
            // Zoom around the mouse cursor so the point under the pointer stays fixed
            let mouse_pos = ctx.input(|i| i.pointer.hover_pos());
            if let (Some(pos), Some(rect)) = (mouse_pos, self.canvas.last_canvas_rect) {
                self.canvas.zoom_around_screen_point(zoom_factor, pos, rect);
            } else {
                self.canvas.apply_zoom(zoom_factor);
            }
        }
    }
}
