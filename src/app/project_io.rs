impl PaintFEApp {
    fn percent_decode_path_component(input: &str) -> String {
        let bytes = input.as_bytes();
        let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
        let mut i = 0usize;
        while i < bytes.len() {
            if bytes[i] == b'%' && i + 2 < bytes.len() {
                let h1 = bytes[i + 1];
                let h2 = bytes[i + 2];
                let hex = |c: u8| -> Option<u8> {
                    match c {
                        b'0'..=b'9' => Some(c - b'0'),
                        b'a'..=b'f' => Some(c - b'a' + 10),
                        b'A'..=b'F' => Some(c - b'A' + 10),
                        _ => None,
                    }
                };
                if let (Some(a), Some(b)) = (hex(h1), hex(h2)) {
                    out.push((a << 4) | b);
                    i += 3;
                    continue;
                }
            }
            out.push(bytes[i]);
            i += 1;
        }
        String::from_utf8_lossy(&out).to_string()
    }

    fn parse_file_uri_list(text: &str) -> Vec<PathBuf> {
        let mut paths = Vec::new();
        for raw_line in text.split(['\n', '\0']) {
            let line = raw_line.trim().trim_end_matches('\r');
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            // Some desktop environments prepend an action line before URIs.
            if line.eq_ignore_ascii_case("copy") || line.eq_ignore_ascii_case("cut") {
                continue;
            }

            if let Some(mut rest) = line.strip_prefix("file://") {
                if let Some(after_localhost) = rest.strip_prefix("localhost/") {
                    rest = after_localhost;
                } else if let Some(idx) = rest.find('/') {
                    // file://<host>/<path> -> keep absolute path part
                    rest = &rest[idx + 1..];
                }

                let decoded = Self::percent_decode_path_component(rest);
                #[cfg(target_os = "windows")]
                let candidate = {
                    let normalized = decoded.replace('/', "\\");
                    PathBuf::from(normalized)
                };
                #[cfg(not(target_os = "windows"))]
                let candidate = PathBuf::from(format!("/{}", decoded));

                if candidate.is_file() {
                    paths.push(candidate);
                }
                continue;
            }

            let direct = PathBuf::from(line);
            if direct.is_file() {
                paths.push(direct);
            }
        }
        paths
    }

    fn handle_file_uri_paste_events(&mut self, ctx: &egui::Context) {
        // If dropped files are available this frame, prefer the dedicated
        // drag-and-drop path and avoid double-processing URI payloads.
        if ctx.input(|i| !i.raw.dropped_files.is_empty()) {
            return;
        }

        // If this frame looks like an explicit paste shortcut (Ctrl/Cmd+V),
        // let the normal clipboard paste path handle it so behavior matches
        // Windows (paste into current project with transform overlay).
        let shortcut_paste = ctx.input(|i| {
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
        if shortcut_paste {
            return;
        }

        let cursor_canvas = if self.active_project_index < self.projects.len() {
            ctx.input(|i| i.pointer.latest_pos())
                .and_then(|screen_pos| {
                    self.canvas.last_canvas_rect.and_then(|rect| {
                        let state = &self.projects[self.active_project_index].canvas_state;
                        self.canvas
                            .screen_to_canvas_f32_pub(screen_pos, rect, state)
                    })
                })
        } else {
            None
        };

        let decoded_images = ctx.input_mut(|i| {
            let mut images = Vec::new();
            i.events.retain(|e| {
                if let egui::Event::Paste(text) = e {
                    let parsed = Self::parse_file_uri_list(text);
                    if !parsed.is_empty() {
                        let mut decoded_any = false;
                        for path in parsed {
                            if let Ok((w, h)) = image::image_dimensions(&path)
                                && crate::io::validate_open_dimensions(w, h).is_err()
                            {
                                continue;
                            }
                            if let Ok(decoded) = image::open(&path) {
                                images.push(decoded.to_rgba8());
                                decoded_any = true;
                                break;
                            }
                        }
                        if decoded_any {
                            return false;
                        }
                    }
                }
                true
            });
            images
        });

        for img in decoded_images {
            self.queue_paste_image(img, cursor_canvas, None, false, false, None);
        }
    }

    fn open_image_from_bytes(&mut self, bytes: &[u8], name_hint: Option<String>) {
        let Ok(decoded) = image::load_from_memory(bytes) else {
            return;
        };

        let image = decoded.to_rgba8();
        let width = image.width();
        let height = image.height();
        if crate::io::validate_open_dimensions(width, height).is_err() {
            return;
        }

        self.persist_active_project_view();
        self.untitled_counter += 1;

        let mut project = Project::new_untitled(self.untitled_counter, width, height);
        if let Some(layer) = project.canvas_state.layers.first_mut() {
            layer.pixels = TiledImage::from_rgba_image(&image);
        }
        project.canvas_state.composite_cache = None;
        project.canvas_state.mark_dirty(None);

        if let Some(name) = name_hint {
            let trimmed = name.trim();
            if !trimmed.is_empty() {
                project.name = trimmed.to_string();
            }
        }

        self.projects.push(project);
        self.active_project_index = self.projects.len() - 1;
        self.restore_active_project_view();
        self.canvas.gpu_clear_layers();
        self.maybe_close_initial_blank();
    }

    /// Handle opening one or more files - creates a new project tab for each.
    /// Uses multi-select file dialog so Linux/Wayland users can open multiple files
    /// without relying on drag-and-drop.
    fn handle_open_file(&mut self, current_time: f64) {
        let paths = self.file_handler.pick_file_paths();
        for path in paths {
            self.open_file_by_path(path, current_time);
        }
    }

    /// Open a file by path — creates a new project tab.
    /// Used by both "Open…" menu and drag-and-drop.
    fn open_file_by_path(&mut self, path: std::path::PathBuf, current_time: f64) {
        // Check if file is already open in another tab
        if self
            .projects
            .iter()
            .any(|p| p.path.as_ref().is_some_and(|pp| *pp == path))
        {
            // Already open — just switch to that tab
            if let Some(idx) = self
                .projects
                .iter()
                .position(|p| p.path.as_ref().is_some_and(|pp| *pp == path))
            {
                self.switch_to_project(idx);
            }
            return;
        }

        let is_pfe = path
            .extension()
            .map(|ext| ext.to_string_lossy().to_lowercase() == "pfe")
            .unwrap_or(false);

        if is_pfe {
            // Open .pfe project file in background (preserves layers)
            let sender = self.io_sender.clone();
            if self.pending_io_ops == 0 {
                self.io_ops_start_time = Some(current_time);
            }
            self.pending_io_ops += 1;
            rayon::spawn(move || match crate::io::load_pfe(&path) {
                Ok(canvas_state) => {
                    let _ = sender.send(IoResult::PfeLoaded { canvas_state, path });
                }
                Err(e) => {
                    let _ = sender.send(IoResult::LoadFailed(format!(
                        "Failed to open project: {}",
                        e
                    )));
                }
            });
        } else {
            // Open as flat image on background thread (decode + tile in parallel)
            // Check for animated GIF/APNG first
            let sender = self.io_sender.clone();
            if self.pending_io_ops == 0 {
                self.io_ops_start_time = Some(current_time);
            }
            self.pending_io_ops += 1;
            rayon::spawn(move || {
                let ext = path
                    .extension()
                    .map(|e| e.to_string_lossy().to_lowercase())
                    .unwrap_or_default();
                let is_potential_animation = ext == "gif" || ext == "png";

                if is_potential_animation {
                    let anim_info = crate::io::detect_animation(&path);
                    if anim_info.is_animated && anim_info.frame_count > 1 {
                        // Animated file: decode all frames
                        let decode_result = match ext.as_str() {
                            "gif" => crate::io::decode_gif_frames(&path),
                            "png" => crate::io::decode_apng_frames(&path),
                            _ => unreachable!(),
                        };

                        match decode_result {
                            Ok(frames) if !frames.is_empty() => {
                                let format = if ext == "gif" {
                                    SaveFormat::Gif
                                } else {
                                    SaveFormat::Png
                                };
                                let width = frames[0].0.width();
                                let height = frames[0].0.height();

                                // Calculate FPS from average delay
                                let avg_delay = anim_info.avg_delay_ms.max(10) as f32;
                                let fps = (1000.0 / avg_delay).clamp(1.0, 60.0);

                                // First frame becomes the base layer
                                let first_tiled = TiledImage::from_rgba_image(&frames[0].0);

                                // Send first frame immediately so the project opens fast
                                let _ = sender.send(IoResult::AnimatedLoaded {
                                    tiled: first_tiled,
                                    width,
                                    height,
                                    path: path.clone(),
                                    format,
                                    fps,
                                    frame_count: frames.len() as u32,
                                });

                                // Remaining frames converted to TiledImages in background
                                if frames.len() > 1 {
                                    // We need to get the project_id back, but since it hasn't been
                                    // created yet, we use a second spawn that waits a frame.
                                    // Instead, we pack all remaining frames into one message.
                                    let remaining: Vec<(TiledImage, String)> = frames[1..]
                                        .iter()
                                        .enumerate()
                                        .map(|(i, (img, _delay))| {
                                            let tiled = TiledImage::from_rgba_image(img);
                                            let name = format!("Frame {}", i + 2);
                                            (tiled, name)
                                        })
                                        .collect();

                                    let _ = sender.send(IoResult::AnimatedFramesLoaded {
                                        path: path.clone(),
                                        frames: remaining,
                                    });
                                }
                            }
                            Ok(_) => {
                                let _ = sender.send(IoResult::LoadFailed(
                                    "Animated file contains no frames".to_string(),
                                ));
                            }
                            Err(e) => {
                                let _ = sender.send(IoResult::LoadFailed(e));
                            }
                        }
                        return;
                    }
                }

                // Non-animated or non-GIF/PNG: standard single-image load
                // Check for RAW camera files first
                let is_raw = crate::io::is_raw_extension(&ext);
                if is_raw {
                    match crate::io::decode_raw_image(&path) {
                        Ok(image) => {
                            let width = image.width();
                            let height = image.height();
                            let tiled = TiledImage::from_rgba_image(&image);
                            // RAW files open as PNG (user must Save As to choose format)
                            let _ = sender.send(IoResult::ImageLoaded {
                                tiled,
                                width,
                                height,
                                path,
                                format: SaveFormat::Png,
                            });
                        }
                        Err(e) => {
                            let _ = sender.send(IoResult::LoadFailed(format!("RAW: {}", e)));
                        }
                    }
                    return;
                }

                if let Ok((w, h)) = image::image_dimensions(&path)
                    && let Err(e) = crate::io::validate_open_dimensions(w, h)
                {
                    let _ = sender.send(IoResult::LoadFailed(e));
                    return;
                }

                match image::open(&path) {
                    Ok(img) => {
                        if crate::io::validate_open_dimensions(img.width(), img.height()).is_err() {
                            let _ = sender.send(IoResult::LoadFailed(format!(
                                "Image size {}x{} exceeds maximum allowed {}x{}",
                                img.width(),
                                img.height(),
                                crate::io::MAX_OPEN_IMAGE_DIM,
                                crate::io::MAX_OPEN_IMAGE_DIM
                            )));
                            return;
                        }
                        let image = img.to_rgba8();
                        let width = image.width();
                        let height = image.height();
                        let tiled = TiledImage::from_rgba_image(&image);

                        let format = path
                            .extension()
                            .map(|ext| match ext.to_string_lossy().to_lowercase().as_str() {
                                "jpg" | "jpeg" => SaveFormat::Jpeg,
                                "webp" => SaveFormat::Webp,
                                "bmp" => SaveFormat::Bmp,
                                "tga" => SaveFormat::Tga,
                                "ico" => SaveFormat::Ico,
                                "tiff" | "tif" => SaveFormat::Tiff,
                                "gif" => SaveFormat::Gif,
                                _ => SaveFormat::Png,
                            })
                            .unwrap_or(SaveFormat::Png);

                        let _ = sender.send(IoResult::ImageLoaded {
                            tiled,
                            width,
                            height,
                            path,
                            format,
                        });
                    }
                    Err(e) => {
                        let _ = sender.send(IoResult::LoadFailed(format!("{}", e)));
                    }
                }
            });
        }
    }

    /// Handle save (quick save or open dialog) for the active project.
    /// For image formats, the encoding runs on a background thread so the UI
    /// stays responsive on large canvases.
    fn handle_save(&mut self, current_time: f64) {
        let idx = self.active_project_index;
        if idx >= self.projects.len() {
            return;
        }

        let has_path = self.projects[idx].file_handler.has_current_path();

        if has_path {
            let project = &mut self.projects[idx];
            let is_pfe = project.file_handler.last_format == SaveFormat::Pfe;
            let is_animated = project.file_handler.last_animated
                && project.file_handler.last_format.supports_animation();

            if is_pfe {
                // PFE save — build data snapshot, serialize in background
                project.canvas_state.ensure_all_text_layers_rasterized();
                if let Some(path) = project.file_handler.current_path.clone() {
                    let pfe_data = crate::io::build_pfe(&project.canvas_state);
                    let sender = self.io_sender.clone();
                    if self.pending_io_ops == 0 {
                        self.io_ops_start_time = Some(current_time);
                    }
                    self.pending_io_ops += 1;
                    rayon::spawn(move || match crate::io::write_pfe(&pfe_data, &path) {
                        Ok(()) => {
                            let _ = sender.send(IoResult::SaveComplete {
                                project_index: idx,
                                path,
                                format: SaveFormat::Pfe,
                                quality: 100,
                                tiff_compression: TiffCompression::None,
                                update_project_path: false,
                            });
                        }
                        Err(e) => {
                            let _ = sender.send(IoResult::SaveFailed {
                                project_index: idx,
                                error: format!("{}", e),
                            });
                        }
                    });
                }
            } else if is_animated {
                // Quick-save animated format (include all layers, even hidden)
                project.canvas_state.ensure_all_text_layers_rasterized();
                let frames: Vec<image::RgbaImage> = project
                    .canvas_state
                    .layers
                    .iter()
                    .map(|l| l.pixels.to_rgba_image())
                    .collect();
                let path = project.file_handler.current_path.clone().unwrap();
                let format = project.file_handler.last_format;
                let quality = project.file_handler.last_quality;
                let tiff_compression = project.file_handler.last_tiff_compression;
                let fps = project.file_handler.last_animation_fps;
                let gif_colors = project.file_handler.last_gif_colors;
                let gif_dither = project.file_handler.last_gif_dither;
                let sender = self.io_sender.clone();
                if self.pending_io_ops == 0 {
                    self.io_ops_start_time = Some(current_time);
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
                                project_index: idx,
                                path,
                                format,
                                quality,
                                tiff_compression,
                                update_project_path: false,
                            });
                        }
                        Err(e) => {
                            let _ = sender.send(IoResult::SaveFailed {
                                project_index: idx,
                                error: e,
                            });
                        }
                    }
                });
            } else {
                // Static image save — composite on main thread (usually cached),
                // encode + write on background thread.
                project.canvas_state.ensure_all_text_layers_rasterized();
                let composite = project.canvas_state.composite();
                let path = project.file_handler.current_path.clone().unwrap();
                let format = project.file_handler.last_format;
                let quality = project.file_handler.last_quality;
                let tiff_compression = project.file_handler.last_tiff_compression;
                let sender = self.io_sender.clone();
                if self.pending_io_ops == 0 {
                    self.io_ops_start_time = Some(current_time);
                }
                self.pending_io_ops += 1;
                rayon::spawn(move || {
                    match crate::io::encode_and_write(
                        &composite,
                        &path,
                        format,
                        quality,
                        tiff_compression,
                    ) {
                        Ok(()) => {
                            let _ = sender.send(IoResult::SaveComplete {
                                project_index: idx,
                                path,
                                format,
                                quality,
                                tiff_compression,
                                update_project_path: false,
                            });
                        }
                        Err(e) => {
                            let _ = sender.send(IoResult::SaveFailed {
                                project_index: idx,
                                error: format!("{}", e),
                            });
                        }
                    }
                });
            }
        } else {
            // No existing path — open Save As dialog
            self.open_save_as_for_project(idx);
        }
    }

    /// Open a Save As dialog for the project at `idx`, switching to that tab first.
    /// Used by both Ctrl+Shift+S and the exit-save queue.
    fn open_save_as_for_project(&mut self, idx: usize) {
        if idx >= self.projects.len() {
            return;
        }
        self.switch_to_project(idx);
        let project = &mut self.projects[idx];
        project.canvas_state.ensure_all_text_layers_rasterized();
        let composite = project.canvas_state.composite();
        let was_animated = project.was_animated;
        let animation_fps = project.animation_fps;
        let frame_images: Option<Vec<image::RgbaImage>> = if project.canvas_state.layers.len() > 1 {
            Some(
                project
                    .canvas_state
                    .layers
                    .iter()
                    .map(|l| l.pixels.to_rgba_image())
                    .collect(),
            )
        } else {
            None
        };
        self.save_file_dialog.reset();
        self.save_file_dialog.set_source_image(&composite);
        if let Some(frames) = frame_images.as_ref() {
            self.save_file_dialog
                .set_source_animated(frames, was_animated, animation_fps);
        }
        self.save_file_dialog.open = true;
    }

    /// Save a specific project by index (only if it has a path — skips dialog).
    /// Returns true if a background save was launched.
    fn handle_save_project(&mut self, idx: usize, current_time: f64) -> bool {
        if idx >= self.projects.len() {
            return false;
        }
        if !self.projects[idx].file_handler.has_current_path() {
            return false;
        }
        if !self.projects[idx].is_dirty {
            return false;
        }

        let project = &mut self.projects[idx];
        let is_pfe = project.file_handler.last_format == SaveFormat::Pfe;
        let is_animated = project.file_handler.last_animated
            && project.file_handler.last_format.supports_animation();

        if is_pfe {
            project.canvas_state.ensure_all_text_layers_rasterized();
            if let Some(path) = project.file_handler.current_path.clone() {
                let pfe_data = crate::io::build_pfe(&project.canvas_state);
                let sender = self.io_sender.clone();
                if self.pending_io_ops == 0 {
                    self.io_ops_start_time = Some(current_time);
                }
                self.pending_io_ops += 1;
                rayon::spawn(move || match crate::io::write_pfe(&pfe_data, &path) {
                    Ok(()) => {
                        let _ = sender.send(IoResult::SaveComplete {
                            project_index: idx,
                            path,
                            format: SaveFormat::Pfe,
                            quality: 100,
                            tiff_compression: TiffCompression::None,
                            update_project_path: false,
                        });
                    }
                    Err(e) => {
                        let _ = sender.send(IoResult::SaveFailed {
                            project_index: idx,
                            error: format!("{}", e),
                        });
                    }
                });
            }
        } else if is_animated {
            project.canvas_state.ensure_all_text_layers_rasterized();
            let frames: Vec<image::RgbaImage> = project
                .canvas_state
                .layers
                .iter()
                .map(|l| l.pixels.to_rgba_image())
                .collect();
            let path = project.file_handler.current_path.clone().unwrap();
            let format = project.file_handler.last_format;
            let quality = project.file_handler.last_quality;
            let tiff_compression = project.file_handler.last_tiff_compression;
            let fps = project.file_handler.last_animation_fps;
            let gif_colors = project.file_handler.last_gif_colors;
            let gif_dither = project.file_handler.last_gif_dither;
            let sender = self.io_sender.clone();
            if self.pending_io_ops == 0 {
                self.io_ops_start_time = Some(current_time);
            }
            self.pending_io_ops += 1;
            rayon::spawn(move || {
                let result = match format {
                    SaveFormat::Gif => {
                        crate::io::encode_animated_gif(&frames, fps, gif_colors, gif_dither, &path)
                    }
                    SaveFormat::Png => crate::io::encode_animated_png(&frames, fps, &path),
                    _ => Err("Format does not support animation".to_string()),
                };
                match result {
                    Ok(()) => {
                        let _ = sender.send(IoResult::SaveComplete {
                            project_index: idx,
                            path,
                            format,
                            quality,
                            tiff_compression,
                            update_project_path: false,
                        });
                    }
                    Err(e) => {
                        let _ = sender.send(IoResult::SaveFailed {
                            project_index: idx,
                            error: e,
                        });
                    }
                }
            });
        } else {
            project.canvas_state.ensure_all_text_layers_rasterized();
            let composite = project.canvas_state.composite();
            let path = project.file_handler.current_path.clone().unwrap();
            let format = project.file_handler.last_format;
            let quality = project.file_handler.last_quality;
            let tiff_compression = project.file_handler.last_tiff_compression;
            let sender = self.io_sender.clone();
            if self.pending_io_ops == 0 {
                self.io_ops_start_time = Some(current_time);
            }
            self.pending_io_ops += 1;
            rayon::spawn(move || {
                match crate::io::encode_and_write(
                    &composite,
                    &path,
                    format,
                    quality,
                    tiff_compression,
                ) {
                    Ok(()) => {
                        let _ = sender.send(IoResult::SaveComplete {
                            project_index: idx,
                            path,
                            format,
                            quality,
                            tiff_compression,
                            update_project_path: false,
                        });
                    }
                    Err(e) => {
                        let _ = sender.send(IoResult::SaveFailed {
                            project_index: idx,
                            error: format!("{}", e),
                        });
                    }
                }
            });
        }
        true
    }

    /// Save all dirty projects that already have a file path.
    fn handle_save_all(&mut self, current_time: f64) {
        let count = self.projects.len();
        for i in 0..count {
            self.handle_save_project(i, current_time);
        }
    }
}

// --- Operation Helpers (snapshot undo for menus) ---
