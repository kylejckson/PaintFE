impl PaintFEApp {
    pub fn new(
        cc: &eframe::CreationContext<'_>,
        startup_files: Vec<PathBuf>,
        ipc_receiver: mpsc::Receiver<PathBuf>,
    ) -> Self {
        // Initialize settings from disk (or defaults if no saved file)
        let mut settings = AppSettings::load();

        // Apply saved language preference (or auto-detect on first boot)
        if settings.language.is_empty() {
            let detected = crate::i18n::detect_system_language();
            crate::i18n::set_language(&detected);
        } else {
            crate::i18n::set_language(&settings.language);
        }

        // -- Font configuration ------------------------------------------------
        // Proportional: DM Sans (primary, matches website) → Noto Sans (Cyrillic/
        //   Greek/Thai fallback) → system CJK → egui defaults
        // Monospace: JetBrains Mono (matches website badges/tags) → egui defaults
        {
            let mut fonts = egui::FontDefinitions::default();

            // DM Sans — primary proportional UI font (~47 KB, Latin + Latin Ext)
            fonts.font_data.insert(
                "dm_sans".to_owned(),
                egui::FontData::from_static(include_bytes!("../../assets/fonts/DMSans-Regular.ttf"))
                    .into(),
            );

            // Noto Sans — fallback for Cyrillic, Greek, Thai (~556 KB)
            fonts.font_data.insert(
                "noto_sans".to_owned(),
                egui::FontData::from_static(include_bytes!("../../assets/fonts/NotoSans-Regular.ttf"))
                    .into(),
            );

            // JetBrains Mono — monospace for badges, tags, script editor (~110 KB)
            fonts.font_data.insert(
                "jetbrains_mono".to_owned(),
                egui::FontData::from_static(include_bytes!(
                    "../../assets/fonts/JetBrainsMono-Regular.ttf"
                ))
                .into(),
            );

            // Proportional family: DM Sans → Noto Sans → egui defaults
            let proportional = fonts
                .families
                .entry(egui::FontFamily::Proportional)
                .or_default();
            proportional.insert(0, "noto_sans".to_owned());
            proportional.insert(0, "dm_sans".to_owned()); // push to front

            // Monospace family: JetBrains Mono → egui defaults (Hack)
            let monospace = fonts
                .families
                .entry(egui::FontFamily::Monospace)
                .or_default();
            monospace.insert(0, "jetbrains_mono".to_owned());

            // Try to discover a system CJK font at runtime for JP/KO/ZH support
            if let Some((name, data)) = discover_system_cjk_font() {
                fonts
                    .font_data
                    .insert(name.clone(), egui::FontData::from_owned(data).into());
                // Insert CJK after DM Sans + Noto Sans but before egui defaults
                let proportional = fonts
                    .families
                    .entry(egui::FontFamily::Proportional)
                    .or_default();
                proportional.insert(2, name);
            }

            cc.egui_ctx.set_fonts(fonts);
        }

        // Initialize theme from settings with accent and apply immediately
        settings.persisted_text_font_family =
            crate::ops::text::resolve_font_family_preference(&settings.persisted_text_font_family);

        let accent = if settings.theme_preset == crate::theme::ThemePreset::Custom {
            settings.custom_accent
        } else {
            settings.theme_preset.accent_colors()
        };
        let mut theme = match settings.theme_mode {
            crate::theme::ThemeMode::Dark => Theme::dark_with_accent(settings.theme_preset, accent),
            crate::theme::ThemeMode::Light => {
                Theme::light_with_accent(settings.theme_preset, accent)
            }
        };
        let ov = settings.build_theme_overrides();
        theme.apply_overrides(&ov);
        theme.apply(&cc.egui_ctx);
        // Disable egui's built-in Ctrl+/Ctrl- keyboard zoom so it doesn't
        // intercept Ctrl++ before our canvas-zoom keybind handler fires.
        cc.egui_ctx.options_mut(|o| o.zoom_with_keyboard = false);

        // Initialize with one default project (or empty if disabled in settings)
        let (initial_projects, initial_counter) = if settings.create_canvas_on_startup {
            let w = settings.default_canvas_width.max(1);
            let h = settings.default_canvas_height.max(1);
            (vec![Project::new_untitled(1, w, h)], 1usize)
        } else {
            (Vec::new(), 0usize)
        };

        // Initialize assets
        let mut assets = Assets::new();
        assets.init(&cc.egui_ctx);

        let (filter_sender, filter_receiver) = mpsc::channel();
        let (io_sender, io_receiver) = mpsc::channel();
        let (script_sender, script_receiver) = mpsc::channel();
        let (canvas_op_sender, canvas_op_receiver) = mpsc::channel();

        // Probe ONNX Runtime availability
        let onnx_available =
            if !settings.onnx_runtime_path.is_empty() && !settings.birefnet_model_path.is_empty() {
                crate::ops::ai::probe_onnx_runtime(&settings.onnx_runtime_path).is_ok()
                    && std::path::Path::new(&settings.birefnet_model_path).exists()
            } else {
                false
            };
        let onnx_last_probed_paths = (
            settings.onnx_runtime_path.clone(),
            settings.birefnet_model_path.clone(),
        );

        let canvas = Canvas::new(&settings.preferred_gpu);
        let create_canvas_on_startup = settings.create_canvas_on_startup;

        let mut app = Self {
            projects: initial_projects,
            active_project_index: 0,
            untitled_counter: initial_counter,
            canvas,
            file_handler: FileHandler::new(),
            tools_panel: tools::ToolsPanel::default(),
            layers_panel: layers::LayersPanel::default(),
            colors_panel: colors::ColorsPanel::default(),
            palette_panel: palette::PalettePanel::default(),
            history_panel: history::HistoryPanel::default(),
            new_file_dialog: NewFileDialog::default(),
            save_file_dialog: SaveFileDialog::default(),
            settings_window: SettingsWindow::default(),
            assets,
            settings,
            theme,
            window_visibility: WindowVisibility::new(),
            active_dialog: ActiveDialog::default(),
            paste_overlay: None,
            pending_paste_request: None,
            paste_transform_undo: Vec::new(),
            paste_transform_redo: Vec::new(),
            move_pixels_before: None,
            move_sel_dragging: false,
            move_sel_last_canvas: None,
            move_sel_handle: None,
            move_sel_start_mask: None,
            move_sel_start_bounds: None,
            pending_selection_reassert: None,
            layers_panel_right_offset: None,
            layers_panel_size: None,
            history_panel_right_offset: None,
            history_panel_size: None,
            colors_panel_left_offset: None,
            palette_panel_pos: None,
            tools_panel_pos: None,
            last_screen_size: (0.0, 0.0),
            ui_cursor_blocking_rects: Vec::new(),
            ui_cursor_blocking_rects_next: Vec::new(),
            ui_pointer_capture_active: false,
            is_move_pixels_active: false,
            is_pointer_over_layers_panel: false,
            filter_sender,
            filter_receiver,
            pending_filter_jobs: 0,
            filter_ops_start_time: None,
            filter_status_description: String::new(),
            preview_job_token: 1,
            filter_cancel: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            canvas_op_sender,
            canvas_op_receiver,
            io_sender,
            io_receiver,
            pending_io_ops: 0,
            io_ops_start_time: None,
            onnx_available,
            onnx_last_probed_paths,
            script_editor: {
                let mut se = script_editor::ScriptEditorPanel::default();
                se.load_saved_scripts();
                se
            },
            script_right_offset: None,
            script_sender,
            script_receiver,
            custom_scripts: script_editor::load_custom_effects(),
            script_original_pixels: None,
            pending_close_index: None,
            pending_exit: false,
            force_exit: false,
            exit_save_queue: Vec::new(),
            exit_save_active: false,
            last_autosave: std::time::Instant::now(),
            first_frame: true,
            ipc_receiver,
            close_initial_blank: !startup_files.is_empty() && create_canvas_on_startup,
            pending_startup_files: startup_files,
            pending_open_paths: HashSet::new(),
            prev_ctrl_c_down: false,
            prev_ctrl_x_down: false,
            prev_ctrl_v_down: false,
            prev_enter_down: false,
            prev_escape_down: false,
            prev_vk_c_press_count: 0,
            prev_vk_x_press_count: 0,
            prev_vk_v_press_count: 0,
            prev_vk_enter_press_count: 0,
            prev_vk_escape_press_count: 0,
            recent_color_project_id: None,
            recent_color_undo_count: 0,
            palette_reposition_settle_frames: 8,
            palette_startup_target_pos: None,
            last_tool_settings_fingerprint: 0,
            last_window_state_fingerprint: 0,
            last_window_state_observed_fingerprint: 0,
            window_state_dirty: false,
            last_window_state_change_time: 0.0,
            last_paste_trigger_time: -1.0,
        };

        app.window_visibility.tools = app.settings.persist_tools_visible;
        app.window_visibility.layers = app.settings.persist_layers_visible;
        app.window_visibility.history = app.settings.persist_history_visible;
        app.window_visibility.colors = app.settings.persist_colors_visible;
        app.window_visibility.palette = app.settings.persist_palette_visible;
        app.window_visibility.script_editor = app.settings.persist_script_editor_visible;
        app.tools_panel_pos = app.settings.persist_tools_panel_pos;
        app.layers_panel_right_offset = app.settings.persist_layers_panel_right_offset;
        app.layers_panel_size = app.settings.persist_layers_panel_size;
        app.history_panel_right_offset = app.settings.persist_history_panel_right_offset;
        app.history_panel_size = app.settings.persist_history_panel_size;
        app.colors_panel_left_offset = app.settings.persist_colors_panel_left_offset;
        app.palette_panel_pos = app.settings.persist_palette_panel_pos.or_else(|| {
            app.settings
                .persist_palette_panel_right_offset
                .map(|(right, bottom)| {
                    (
                        app.settings.persist_window_width - right,
                        app.settings.persist_window_height - bottom,
                    )
                })
                .or_else(|| {
                    app.settings
                        .persist_palette_panel_left_offset
                        .map(|(x, bottom)| (x, app.settings.persist_window_height - bottom))
                })
        });
        app.palette_startup_target_pos = app.palette_panel_pos;
        app.script_right_offset = app.settings.persist_script_right_offset;
        app.colors_panel
            .set_expanded(app.settings.persist_colors_panel_expanded);
        app.new_file_dialog
            .set_lock_aspect_ratio(app.settings.persist_new_file_lock_aspect);
        app.palette_panel
            .load_recent_colors_from_serialized(&app.settings.persist_palette_recent_colors);

        // Reload custom brush tips from persisted settings
        {
            use base64::Engine;
            for (name, cat, b64) in &app.settings.custom_brush_tips {
                if let Ok(png_data) = base64::engine::general_purpose::STANDARD.decode(b64) {
                    app.assets.load_brush_tip(&cc.egui_ctx, name, cat, &png_data);
                }
            }
            for (name, cat, b64) in &app.settings.custom_shapes {
                if let Ok(path_data) = base64::engine::general_purpose::STANDARD.decode(b64)
                    && let Ok(path_data) = String::from_utf8(path_data)
                {
                    let _ = app.assets.load_custom_shape(&cc.egui_ctx, name, cat, &path_data);
                }
            }
        }
        app.apply_persisted_tool_settings();
        app.last_tool_settings_fingerprint = app.compute_tool_settings_fingerprint();
        if let Some((project_id, undo_count)) = app
            .active_project()
            .map(|project| (project.id, project.history.undo_count()))
        {
            app.recent_color_project_id = Some(project_id);
            app.recent_color_undo_count = undo_count;
        }
        app.last_window_state_fingerprint = app.compute_window_state_fingerprint();
        app.last_window_state_observed_fingerprint = app.last_window_state_fingerprint;
        log_info!("Startup: PaintFE initialized (theme={:?} preset={:?} projects={})", app.theme.mode, app.settings.theme_preset, app.projects.len());
        app
    }

    /// Get a reference to the active project
    fn active_project(&self) -> Option<&Project> {
        self.projects.get(self.active_project_index)
    }

    /// Get a mutable reference to the active project
    fn active_project_mut(&mut self) -> Option<&mut Project> {
        self.projects.get_mut(self.active_project_index)
    }

    fn persist_active_project_view(&mut self) {
        let (zoom, pan_offset) = self.canvas.view_state();
        if let Some(project) = self.projects.get_mut(self.active_project_index) {
            project.view_zoom = zoom;
            project.view_pan_offset = pan_offset;
        }
    }

    fn restore_active_project_view(&mut self) {
        if let Some(project) = self.projects.get(self.active_project_index) {
            self.canvas
                .set_view_state(project.view_zoom, project.view_pan_offset);
        } else {
            self.canvas.reset_zoom();
        }
    }

    /// Create a new untitled project and switch to it
    fn new_project(&mut self, width: u32, height: u32) {
        if self.paste_overlay.is_some() {
            self.commit_paste_overlay();
        }
        self.persist_active_project_view();
        self.untitled_counter += 1;
        let project = Project::new_untitled(self.untitled_counter, width, height);
        self.projects.push(project);
        self.active_project_index = self.projects.len() - 1;
        self.restore_active_project_view();
    }

    fn copy_active_selection_or_overlay(&self) -> bool {
        if let Some(overlay) = self.paste_overlay.as_ref() {
            crate::ops::clipboard::copy_overlay(overlay)
        } else if let Some(project) = self.active_project() {
            crate::ops::clipboard::copy_selection(
                &project.canvas_state,
                self.settings.clipboard_copy_transparent_cutout,
            )
        } else {
            false
        }
    }

    /// If the initial blank project should be auto-closed (because we opened a
    /// file from the command line or IPC), close it now. Called once after the
    /// first real file finishes loading.
    fn maybe_close_initial_blank(&mut self) {
        if !self.close_initial_blank {
            return;
        }
        self.close_initial_blank = false;

        // The initial blank project is always at index 0.  Only auto-close it
        // if it's still untitled, unmodified, and there's now at least one
        // other project.
        if self.projects.len() > 1 {
            let is_blank = self.projects[0].path.is_none() && !self.projects[0].is_dirty;
            if is_blank {
                self.projects.remove(0);
                // The newly loaded project was pushed to the end; adjust index.
                self.active_project_index = self.projects.len() - 1;
                self.restore_active_project_view();
            }
        }
    }

    /// Close a project by index, with dirty check
    fn close_project(&mut self, index: usize) {
        if index >= self.projects.len() {
            return;
        }

        // If closing the active project and there's a paste overlay, commit it first.
        if index == self.active_project_index && self.paste_overlay.is_some() {
            self.commit_paste_overlay();
        }

        let project = &self.projects[index];
        if project.is_dirty {
            // Defer: show unsaved-changes dialog
            self.pending_close_index = Some(index);
            return;
        }

        self.persist_active_project_view();
        self.projects.remove(index);

        // Adjust active index — allow empty state
        if self.projects.is_empty() {
            self.active_project_index = 0;
        } else if self.active_project_index >= self.projects.len() {
            self.active_project_index = self.projects.len() - 1;
        } else if index < self.active_project_index {
            self.active_project_index -= 1;
        }

        self.restore_active_project_view();
    }

    /// Close a project by index unconditionally (no dirty check).
    /// Used after the user has confirmed they want to discard changes.
    fn force_close_project(&mut self, index: usize) {
        if index >= self.projects.len() {
            return;
        }
        if index == self.active_project_index && self.paste_overlay.is_some() {
            self.commit_paste_overlay();
        }
        self.persist_active_project_view();
        self.projects.remove(index);
        if self.projects.is_empty() {
            self.active_project_index = 0;
        } else if self.active_project_index >= self.projects.len() {
            self.active_project_index = self.projects.len() - 1;
        } else if index < self.active_project_index {
            self.active_project_index -= 1;
        }
        self.restore_active_project_view();
    }

    /// Switch to a different project tab
    fn switch_to_project(&mut self, index: usize) {
        if index < self.projects.len() && index != self.active_project_index {
            self.persist_active_project_view();
            // Commit any active paste overlay before switching tabs.
            // The overlay belongs to the current project's canvas — switching
            // without committing would leave it orphaned.
            if self.paste_overlay.is_some() {
                self.commit_paste_overlay();
            }
            // Clear move-selection drag state
            self.move_sel_dragging = false;
            self.move_sel_last_canvas = None;
            self.move_sel_handle = None;
            self.move_sel_start_mask = None;
            self.move_sel_start_bounds = None;
            self.pending_selection_reassert = None;
            self.is_move_pixels_active = false;

            // Clear GPU layer textures — different project, different layers.
            self.canvas.gpu_clear_layers();

            self.active_project_index = index;
            self.restore_active_project_view();
        }
    }

    fn queue_paste_image(
        &mut self,
        image: RgbaImage,
        cursor_canvas: Option<(f32, f32)>,
        source_center: Option<egui::Pos2>,
        use_source_center: bool,
        overwrite_transparent_pixels: bool,
        overwrite_mask: Option<image::GrayImage>,
    ) {
        if self.paste_overlay.is_some() {
            self.commit_paste_overlay();
        }

        let Some(project) = self.active_project() else {
            return;
        };

        let request = PendingPasteRequest {
            image,
            target_project_id: project.id,
            cursor_canvas,
            source_center,
            use_source_center,
            overwrite_transparent_pixels,
            overwrite_mask,
        };

        if request.image.width() > project.canvas_state.width
            || request.image.height() > project.canvas_state.height
        {
            self.pending_paste_request = Some(request);
        } else {
            self.apply_pending_paste_request(request, false);
        }
    }

    fn queue_paste_from_clipboard(&mut self, cursor_canvas: Option<(f32, f32)>) {
        let cutout_enabled = self.settings.clipboard_copy_transparent_cutout;
        let payload = if let Some(payload) = crate::ops::clipboard::get_clipboard_image_for_paste()
        {
            payload
        } else if let Some(img) = crate::ops::clipboard::get_clipboard_image_pub() {
                crate::ops::clipboard::ClipboardImageForPaste {
                    image: img,
                    source: ClipboardImageSource::Internal,
                    origin_center: None,
                    overwrite_transparent_pixels: true,
                    overwrite_mask: None,
                }
            } else {
                return;
            };

        let overwrite = match payload.source {
            ClipboardImageSource::Internal => payload.overwrite_transparent_pixels,
            ClipboardImageSource::External => {
                payload.overwrite_transparent_pixels && cutout_enabled
            }
        };

        let use_source_center =
            payload.source == ClipboardImageSource::Internal && payload.origin_center.is_some();
        self.queue_paste_image(
            payload.image,
            cursor_canvas,
            payload.origin_center,
            use_source_center,
            overwrite,
            payload.overwrite_mask,
        );
    }

    fn apply_pending_paste_request(&mut self, request: PendingPasteRequest, resize_canvas: bool) {
        let Some(target_idx) = self
            .projects
            .iter()
            .position(|p| p.id == request.target_project_id)
        else {
            return;
        };

        self.switch_to_project(target_idx);

        if resize_canvas {
            let target_w = request.image.width();
            let target_h = request.image.height();
            self.do_snapshot_op("Resize Canvas to Fit Paste", |s| {
                let new_w = s.width.max(target_w);
                let new_h = s.height.max(target_h);
                // Center the existing canvas content within the new expanded canvas
                crate::ops::transform::resize_canvas(
                    s,
                    new_w,
                    new_h,
                    (1, 1),
                    image::Rgba([0, 0, 0, 0]),
                );
            });
        }

        let move_interpolation = self.tools_panel.move_interpolation;
        let move_anti_aliasing = self.tools_panel.move_anti_aliasing;
        if let Some(project) = self.active_project_mut() {
            let (cw, ch) = (project.canvas_state.width, project.canvas_state.height);
            // Performance: if overwrite mode is requested but the source image has no
            // actual transparency (all alpha = 255), skip the expensive CPU overwrite
            // path and use the GPU-accelerated path instead (which supports preview
            // downscaling during interaction).
            let needs_overwrite = request.overwrite_transparent_pixels
                && (request.overwrite_mask.is_some()
                    || request.image.pixels().any(|p| p[3] < 255));
            let mut overlay = if request.use_source_center {
                let center = request
                    .source_center
                    .unwrap_or(egui::Pos2::new(cw as f32 / 2.0, ch as f32 / 2.0));
                let mut o = PasteOverlay::from_image_at(request.image, cw, ch, center);
                o.overwrite_transparent_pixels = needs_overwrite;
                o.overwrite_mask = request.overwrite_mask;
                o
            } else if let Some((cx, cy)) = request.cursor_canvas {
                let mut o = PasteOverlay::from_image_at(request.image, cw, ch, egui::Pos2::new(cx, cy));
                o.overwrite_transparent_pixels = needs_overwrite;
                o.overwrite_mask = request.overwrite_mask;
                o
            } else {
                let mut o = PasteOverlay::from_image(request.image, cw, ch);
                o.overwrite_transparent_pixels = needs_overwrite;
                o.overwrite_mask = request.overwrite_mask;
                o
            };
            overlay.interpolation = move_interpolation;
            overlay.anti_aliasing = move_anti_aliasing;

            project.canvas_state.clear_selection();
            self.paste_overlay = Some(overlay);
            self.paste_transform_undo.clear();
            self.paste_transform_redo.clear();
            self.canvas.open_paste_menu = true;
        }
    }

    fn tool_to_key(tool: tools::Tool) -> &'static str {
        match tool {
            tools::Tool::Brush => "brush",
            tools::Tool::Eraser => "eraser",
            tools::Tool::Pencil => "pencil",
            tools::Tool::Line => "line",
            tools::Tool::RectangleSelect => "rect_select",
            tools::Tool::EllipseSelect => "ellipse_select",
            tools::Tool::MovePixels => "move_pixels",
            tools::Tool::MoveSelection => "move_selection",
            tools::Tool::MagicWand => "magic_wand",
            tools::Tool::Fill => "fill",
            tools::Tool::ColorPicker => "color_picker",
            tools::Tool::Gradient => "gradient",
            tools::Tool::ContentAwareBrush => "content_aware_brush",
            tools::Tool::Liquify => "liquify",
            tools::Tool::MeshWarp => "mesh_warp",
            tools::Tool::ColorRemover => "color_remover",
            tools::Tool::Smudge => "smudge",
            tools::Tool::CloneStamp => "clone_stamp",
            tools::Tool::Text => "text",
            tools::Tool::PerspectiveCrop => "perspective_crop",
            tools::Tool::Lasso => "lasso",
            tools::Tool::Zoom => "zoom",
            tools::Tool::Pan => "pan",
            tools::Tool::Shapes => "shapes",
        }
    }

    fn key_to_tool(key: &str) -> tools::Tool {
        match key {
            "eraser" => tools::Tool::Eraser,
            "pencil" => tools::Tool::Pencil,
            "line" => tools::Tool::Line,
            "rect_select" => tools::Tool::RectangleSelect,
            "ellipse_select" => tools::Tool::EllipseSelect,
            "move_pixels" => tools::Tool::MovePixels,
            "move_selection" => tools::Tool::MoveSelection,
            "magic_wand" => tools::Tool::MagicWand,
            "fill" => tools::Tool::Fill,
            "color_picker" => tools::Tool::ColorPicker,
            "gradient" => tools::Tool::Gradient,
            "content_aware_brush" => tools::Tool::ContentAwareBrush,
            "liquify" => tools::Tool::Liquify,
            "mesh_warp" => tools::Tool::MeshWarp,
            "color_remover" => tools::Tool::ColorRemover,
            "smudge" => tools::Tool::Smudge,
            "clone_stamp" => tools::Tool::CloneStamp,
            "text" => tools::Tool::Text,
            "perspective_crop" => tools::Tool::PerspectiveCrop,
            "lasso" => tools::Tool::Lasso,
            "zoom" => tools::Tool::Zoom,
            "pan" => tools::Tool::Pan,
            "shapes" => tools::Tool::Shapes,
            _ => tools::Tool::Brush,
        }
    }

    fn apply_persisted_tool_settings(&mut self) {
        self.tools_panel.active_tool = Self::key_to_tool(&self.settings.persisted_active_tool);

        self.tools_panel.properties.size = self.settings.persisted_brush_size.clamp(1.0, 1024.0);
        self.tools_panel.properties.hardness =
            self.settings.persisted_brush_hardness.clamp(0.0, 1.0);
        self.tools_panel.properties.flow = self.settings.persisted_brush_flow.clamp(0.0, 1.0);
        self.tools_panel.properties.spacing =
            self.settings.persisted_brush_spacing.clamp(0.01, 2.0);
        self.tools_panel.properties.scatter = self.settings.persisted_brush_scatter.clamp(0.0, 1.0);
        self.tools_panel.properties.hue_jitter =
            self.settings.persisted_brush_hue_jitter.clamp(0.0, 1.0);
        self.tools_panel.properties.brightness_jitter = self
            .settings
            .persisted_brush_brightness_jitter
            .clamp(0.0, 1.0);
        self.tools_panel.properties.anti_aliased = self.settings.persisted_brush_anti_aliased;
        self.tools_panel.properties.pressure_size = self.settings.persisted_pressure_size;
        self.tools_panel.properties.pressure_opacity = self.settings.persisted_pressure_opacity;
        self.tools_panel.properties.pressure_min_size =
            self.settings.persisted_pressure_min_size.clamp(0.0, 1.0);
        self.tools_panel.properties.pressure_min_opacity =
            self.settings.persisted_pressure_min_opacity.clamp(0.0, 1.0);

        self.tools_panel.properties.brush_mode = match self.settings.persisted_brush_mode.as_str() {
            "dodge" => tools::BrushMode::Dodge,
            "burn" => tools::BrushMode::Burn,
            "sponge" => tools::BrushMode::Sponge,
            _ => tools::BrushMode::Normal,
        };

        self.tools_panel.properties.brush_tip = if self.settings.persisted_brush_tip.is_empty() {
            tools::BrushTip::Circle
        } else {
            tools::BrushTip::Image(self.settings.persisted_brush_tip.clone())
        };

        self.tools_panel.fill_state.tolerance =
            self.settings.persisted_fill_tolerance.clamp(0.0, 100.0);
        self.tools_panel.fill_state.anti_aliased = self.settings.persisted_fill_anti_aliased;
        self.tools_panel.fill_state.global_fill = self.settings.persisted_fill_global;

        self.tools_panel.magic_wand_state.tolerance =
            self.settings.persisted_wand_tolerance.clamp(0.0, 100.0);
        self.tools_panel.magic_wand_state.anti_aliased = self.settings.persisted_wand_anti_aliased;
        self.tools_panel.magic_wand_state.global_select = self.settings.persisted_wand_global;

        self.tools_panel.color_remover_state.tolerance = self
            .settings
            .persisted_color_remover_tolerance
            .clamp(0.0, 100.0);
        self.tools_panel.color_remover_state.smoothness = self
            .settings
            .persisted_color_remover_smoothness
            .clamp(1, 64);
        self.tools_panel.color_remover_state.contiguous =
            self.settings.persisted_color_remover_contiguous;

        self.tools_panel.smudge_state.strength =
            self.settings.persisted_smudge_strength.clamp(0.0, 1.0);

        self.tools_panel.shapes_state.fill_mode =
            match self.settings.persisted_shapes_fill_mode.as_str() {
                "outline" => crate::ops::shapes::ShapeFillMode::Outline,
                "both" => crate::ops::shapes::ShapeFillMode::Both,
                _ => crate::ops::shapes::ShapeFillMode::Filled,
            };
        self.tools_panel.shapes_state.anti_alias = self.settings.persisted_shapes_anti_alias;
        self.tools_panel.shapes_state.corner_radius = self
            .settings
            .persisted_shapes_corner_radius
            .clamp(0.0, 1000.0);
        self.tools_panel.move_interpolation =
            match self.settings.persisted_move_interpolation.as_str() {
                "nearest" => crate::ops::transform::Interpolation::Nearest,
                "bicubic" => crate::ops::transform::Interpolation::Bicubic,
                "lanczos3" => crate::ops::transform::Interpolation::Lanczos3,
                _ => crate::ops::transform::Interpolation::Bilinear,
            };
        self.tools_panel.move_anti_aliasing = self.settings.persisted_move_anti_aliasing;
        self.tools_panel.text_state.font_family = crate::ops::text::resolve_font_family_preference(
            &self.settings.persisted_text_font_family,
        );
        self.tools_panel.text_state.loaded_font = None;
        self.tools_panel.text_state.loaded_font_key.clear();
    }

    fn compute_tool_settings_fingerprint(&self) -> u64 {
        let mut hasher = DefaultHasher::new();

        Self::tool_to_key(self.tools_panel.active_tool).hash(&mut hasher);
        self.tools_panel.properties.size.to_bits().hash(&mut hasher);
        self.tools_panel
            .properties
            .hardness
            .to_bits()
            .hash(&mut hasher);
        self.tools_panel.properties.flow.to_bits().hash(&mut hasher);
        self.tools_panel
            .properties
            .spacing
            .to_bits()
            .hash(&mut hasher);
        self.tools_panel
            .properties
            .scatter
            .to_bits()
            .hash(&mut hasher);
        self.tools_panel
            .properties
            .hue_jitter
            .to_bits()
            .hash(&mut hasher);
        self.tools_panel
            .properties
            .brightness_jitter
            .to_bits()
            .hash(&mut hasher);
        self.tools_panel.properties.anti_aliased.hash(&mut hasher);
        self.tools_panel.properties.pressure_size.hash(&mut hasher);
        self.tools_panel
            .properties
            .pressure_opacity
            .hash(&mut hasher);
        self.tools_panel
            .properties
            .pressure_min_size
            .to_bits()
            .hash(&mut hasher);
        self.tools_panel
            .properties
            .pressure_min_opacity
            .to_bits()
            .hash(&mut hasher);
        match self.tools_panel.properties.brush_mode {
            tools::BrushMode::Normal => 0u8,
            tools::BrushMode::Dodge => 1u8,
            tools::BrushMode::Burn => 2u8,
            tools::BrushMode::Sponge => 3u8,
        }
        .hash(&mut hasher);
        match &self.tools_panel.properties.brush_tip {
            tools::BrushTip::Circle => "".hash(&mut hasher),
            tools::BrushTip::Image(name) => name.hash(&mut hasher),
        }

        self.tools_panel
            .fill_state
            .tolerance
            .to_bits()
            .hash(&mut hasher);
        self.tools_panel.fill_state.anti_aliased.hash(&mut hasher);
        self.tools_panel.fill_state.global_fill.hash(&mut hasher);

        self.tools_panel
            .magic_wand_state
            .tolerance
            .to_bits()
            .hash(&mut hasher);
        self.tools_panel
            .magic_wand_state
            .anti_aliased
            .hash(&mut hasher);
        self.tools_panel
            .magic_wand_state
            .global_select
            .hash(&mut hasher);

        self.tools_panel
            .color_remover_state
            .tolerance
            .to_bits()
            .hash(&mut hasher);
        self.tools_panel
            .color_remover_state
            .smoothness
            .hash(&mut hasher);
        self.tools_panel
            .color_remover_state
            .contiguous
            .hash(&mut hasher);

        self.tools_panel
            .smudge_state
            .strength
            .to_bits()
            .hash(&mut hasher);
        match self.tools_panel.shapes_state.fill_mode {
            crate::ops::shapes::ShapeFillMode::Outline => 0u8,
            crate::ops::shapes::ShapeFillMode::Filled => 1u8,
            crate::ops::shapes::ShapeFillMode::Both => 2u8,
        }
        .hash(&mut hasher);
        self.tools_panel.shapes_state.anti_alias.hash(&mut hasher);
        self.tools_panel
            .shapes_state
            .corner_radius
            .to_bits()
            .hash(&mut hasher);
        match self.tools_panel.move_interpolation {
            crate::ops::transform::Interpolation::Nearest => 0u8,
            crate::ops::transform::Interpolation::Bilinear => 1u8,
            crate::ops::transform::Interpolation::Bicubic => 2u8,
            crate::ops::transform::Interpolation::Lanczos3 => 3u8,
        }
        .hash(&mut hasher);
        self.tools_panel.move_anti_aliasing.hash(&mut hasher);
        self.tools_panel.text_state.font_family.hash(&mut hasher);

        hasher.finish()
    }

    fn compute_window_state_fingerprint(&self) -> u64 {
        fn hash_opt_pair(v: Option<(f32, f32)>, hasher: &mut DefaultHasher) {
            match v {
                Some((x, y)) => {
                    true.hash(hasher);
                    x.to_bits().hash(hasher);
                    y.to_bits().hash(hasher);
                }
                None => false.hash(hasher),
            }
        }

        let mut hasher = DefaultHasher::new();
        self.window_visibility.tools.hash(&mut hasher);
        self.window_visibility.layers.hash(&mut hasher);
        self.window_visibility.history.hash(&mut hasher);
        self.window_visibility.colors.hash(&mut hasher);
        self.window_visibility.palette.hash(&mut hasher);
        self.window_visibility.script_editor.hash(&mut hasher);
        self.settings
            .persist_window_width
            .to_bits()
            .hash(&mut hasher);
        self.settings
            .persist_window_height
            .to_bits()
            .hash(&mut hasher);
        hash_opt_pair(self.settings.persist_window_pos, &mut hasher);
        self.settings.persist_window_maximized.hash(&mut hasher);
        self.palette_panel
            .serialize_recent_colors()
            .hash(&mut hasher);
        hash_opt_pair(self.tools_panel_pos, &mut hasher);
        hash_opt_pair(self.layers_panel_right_offset, &mut hasher);
        hash_opt_pair(self.layers_panel_size, &mut hasher);
        hash_opt_pair(self.history_panel_right_offset, &mut hasher);
        hash_opt_pair(self.history_panel_size, &mut hasher);
        hash_opt_pair(self.colors_panel_left_offset, &mut hasher);
        hash_opt_pair(self.palette_panel_pos, &mut hasher);
        hash_opt_pair(self.script_right_offset, &mut hasher);
        self.colors_panel.is_expanded().hash(&mut hasher);
        self.new_file_dialog.lock_aspect_ratio().hash(&mut hasher);

        let resize_lock = match &self.active_dialog {
            ActiveDialog::ResizeImage(d) => d.lock_aspect,
            ActiveDialog::ResizeCanvas(d) => d.lock_aspect,
            _ => self.settings.persist_resize_lock_aspect,
        };
        resize_lock.hash(&mut hasher);
        hasher.finish()
    }

    fn persist_window_state_if_changed(&mut self, current_time: f64, force: bool) {
        let resize_lock = match &self.active_dialog {
            ActiveDialog::ResizeImage(d) => d.lock_aspect,
            ActiveDialog::ResizeCanvas(d) => d.lock_aspect,
            _ => self.settings.persist_resize_lock_aspect,
        };

        self.settings.persist_tools_visible = self.window_visibility.tools;
        self.settings.persist_layers_visible = self.window_visibility.layers;
        self.settings.persist_history_visible = self.window_visibility.history;
        self.settings.persist_colors_visible = self.window_visibility.colors;
        self.settings.persist_palette_visible = self.window_visibility.palette;
        self.settings.persist_script_editor_visible = self.window_visibility.script_editor;
        self.settings.persist_tools_panel_pos = self.tools_panel_pos;
        self.settings.persist_layers_panel_right_offset = self.layers_panel_right_offset;
        self.settings.persist_layers_panel_size = self.layers_panel_size;
        self.settings.persist_history_panel_right_offset = self.history_panel_right_offset;
        self.settings.persist_history_panel_size = self.history_panel_size;
        self.settings.persist_colors_panel_left_offset = self.colors_panel_left_offset;
        self.settings.persist_palette_panel_pos = self.palette_panel_pos;
        self.settings.persist_palette_recent_colors = self.palette_panel.serialize_recent_colors();
        self.settings.persist_script_right_offset = self.script_right_offset;
        self.settings.persist_colors_panel_expanded = self.colors_panel.is_expanded();
        self.settings.persist_new_file_lock_aspect = self.new_file_dialog.lock_aspect_ratio();
        self.settings.persist_resize_lock_aspect = resize_lock;

        let fp = self.compute_window_state_fingerprint();
        if fp == self.last_window_state_fingerprint {
            self.last_window_state_observed_fingerprint = fp;
            self.window_state_dirty = false;
            return;
        }

        if fp != self.last_window_state_observed_fingerprint {
            self.last_window_state_observed_fingerprint = fp;
            self.last_window_state_change_time = current_time;
        }

        if !self.window_state_dirty {
            self.window_state_dirty = true;
        }

        if !force && current_time - self.last_window_state_change_time < 0.4 {
            return;
        }

        self.last_window_state_fingerprint = fp;
        self.window_state_dirty = false;
        self.settings.save();
    }

    fn persist_tool_settings_if_changed(&mut self) {
        let fp = self.compute_tool_settings_fingerprint();
        if fp == self.last_tool_settings_fingerprint {
            return;
        }
        self.last_tool_settings_fingerprint = fp;

        self.settings.persisted_active_tool =
            Self::tool_to_key(self.tools_panel.active_tool).to_string();
        self.settings.persisted_brush_size = self.tools_panel.properties.size;
        self.settings.persisted_brush_hardness = self.tools_panel.properties.hardness;
        self.settings.persisted_brush_flow = self.tools_panel.properties.flow;
        self.settings.persisted_brush_spacing = self.tools_panel.properties.spacing;
        self.settings.persisted_brush_scatter = self.tools_panel.properties.scatter;
        self.settings.persisted_brush_hue_jitter = self.tools_panel.properties.hue_jitter;
        self.settings.persisted_brush_brightness_jitter =
            self.tools_panel.properties.brightness_jitter;
        self.settings.persisted_brush_anti_aliased = self.tools_panel.properties.anti_aliased;
        self.settings.persisted_pressure_size = self.tools_panel.properties.pressure_size;
        self.settings.persisted_pressure_opacity = self.tools_panel.properties.pressure_opacity;
        self.settings.persisted_pressure_min_size = self.tools_panel.properties.pressure_min_size;
        self.settings.persisted_pressure_min_opacity =
            self.tools_panel.properties.pressure_min_opacity;
        self.settings.persisted_brush_mode = match self.tools_panel.properties.brush_mode {
            tools::BrushMode::Normal => "normal",
            tools::BrushMode::Dodge => "dodge",
            tools::BrushMode::Burn => "burn",
            tools::BrushMode::Sponge => "sponge",
        }
        .to_string();
        self.settings.persisted_brush_tip = match &self.tools_panel.properties.brush_tip {
            tools::BrushTip::Circle => String::new(),
            tools::BrushTip::Image(name) => name.clone(),
        };

        self.settings.persisted_fill_tolerance = self.tools_panel.fill_state.tolerance;
        self.settings.persisted_fill_anti_aliased = self.tools_panel.fill_state.anti_aliased;
        self.settings.persisted_fill_global = self.tools_panel.fill_state.global_fill;

        self.settings.persisted_wand_tolerance = self.tools_panel.magic_wand_state.tolerance;
        self.settings.persisted_wand_anti_aliased = self.tools_panel.magic_wand_state.anti_aliased;
        self.settings.persisted_wand_global = self.tools_panel.magic_wand_state.global_select;

        self.settings.persisted_color_remover_tolerance =
            self.tools_panel.color_remover_state.tolerance;
        self.settings.persisted_color_remover_smoothness =
            self.tools_panel.color_remover_state.smoothness;
        self.settings.persisted_color_remover_contiguous =
            self.tools_panel.color_remover_state.contiguous;
        self.settings.persisted_smudge_strength = self.tools_panel.smudge_state.strength;
        self.settings.persisted_shapes_fill_mode = match self.tools_panel.shapes_state.fill_mode {
            crate::ops::shapes::ShapeFillMode::Outline => "outline",
            crate::ops::shapes::ShapeFillMode::Filled => "filled",
            crate::ops::shapes::ShapeFillMode::Both => "both",
        }
        .to_string();
        self.settings.persisted_shapes_anti_alias = self.tools_panel.shapes_state.anti_alias;
        self.settings.persisted_shapes_corner_radius = self.tools_panel.shapes_state.corner_radius;
        self.settings.persisted_move_interpolation = match self.tools_panel.move_interpolation {
            crate::ops::transform::Interpolation::Nearest => "nearest",
            crate::ops::transform::Interpolation::Bilinear => "bilinear",
            crate::ops::transform::Interpolation::Bicubic => "bicubic",
            crate::ops::transform::Interpolation::Lanczos3 => "lanczos3",
        }
        .to_string();
        self.settings.persisted_move_anti_aliasing = self.tools_panel.move_anti_aliasing;
        self.settings.persisted_text_font_family = self.tools_panel.text_state.font_family.clone();

        self.settings.save();
    }
}


