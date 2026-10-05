impl PaintFEApp {
    fn apply_workspace_command(&mut self, ctx: &egui::Context) {
        use crate::ui::workspace::Command;
        let Some(command) = crate::ui::workspace::take_command(ctx) else {
            return;
        };
        let viewport = ctx.content_rect();
        crate::ui::workspace::clear_gestures(ctx);
        match command {
            Command::Reset => {
                self.tools_panel_pos = Some((12.0, 128.0));
                self.layers_panel_right_offset = Some((264.0, 128.0));
                self.layers_panel_size = Some((240.0, 200.0));
                self.history_panel_right_offset = Some((224.0, 380.0));
                self.history_panel_size = Some((200.0, 200.0));
                self.colors_panel_pos = Some((12.0, 530.0));
                self.palette_panel_pos = Some((280.0, 128.0));
                self.palette_panel_size = Some((300.0, 108.0));
                self.script_right_offset = Some((540.0, 128.0));
                self.window_visibility.tools = true;
                self.window_visibility.layers = true;
                self.window_visibility.history = true;
                self.window_visibility.colors = true;
                self.window_visibility.palette = true;
                self.window_visibility.script_editor = false;
                for (panel, size) in [
                    ("Layers", egui::vec2(240.0, 200.0)),
                    ("History", egui::vec2(200.0, 200.0)),
                    ("Palette", egui::vec2(300.0, 108.0)),
                    ("ScriptEditor", egui::vec2(520.0, 500.0)),
                ] {
                    ctx.data_mut(|d| {
                        d.insert_temp(egui::Id::new(("workspace_size_request", panel)), size)
                    });
                }
            }
            Command::Apply(mut layout) => {
                if layout.validate().is_err() {
                    return;
                }
                self.window_visibility.tools = false;
                self.window_visibility.layers = false;
                self.window_visibility.history = false;
                self.window_visibility.colors = false;
                self.window_visibility.palette = false;
                self.window_visibility.script_editor = false;
                for panel in &layout.panels {
                    let pos = layout.panel_position(panel, viewport);
                    let size = egui::vec2(panel.size[0], panel.size[1]);
                    match panel.panel.as_str() {
                        "Tools" => {
                            self.tools_panel_pos = Some((pos.x, pos.y));
                            self.window_visibility.tools = panel.visible;
                        }
                        "Layers" => {
                            self.layers_panel_right_offset =
                                Some((viewport.right() - pos.x, pos.y));
                            self.layers_panel_size = Some((size.x, size.y));
                            self.window_visibility.layers = panel.visible;
                        }
                        "History" => {
                            self.history_panel_right_offset =
                                Some((viewport.right() - pos.x, pos.y));
                            self.history_panel_size = Some((size.x, size.y));
                            self.window_visibility.history = panel.visible;
                        }
                        "Colors" => {
                            self.colors_panel_pos = Some((pos.x, pos.y));
                            self.window_visibility.colors = panel.visible;
                            if let Some(expanded) = panel.expanded {
                                self.colors_panel.set_expanded(expanded);
                            }
                            if let Some(mask) = panel.section_mask {
                                self.colors_panel.load_section_mask(mask);
                            }
                        }
                        "Palette" => {
                            self.palette_panel_pos = Some((pos.x, pos.y));
                            self.palette_panel_size = Some((size.x, size.y));
                            self.window_visibility.palette = panel.visible;
                        }
                        "ScriptEditor" => {
                            self.script_right_offset = Some((viewport.right() - pos.x, pos.y));
                            self.window_visibility.script_editor = panel.visible;
                        }
                        _ => {}
                    }
                    if matches!(
                        panel.panel.as_str(),
                        "Layers" | "History" | "Palette" | "ScriptEditor"
                    ) {
                        ctx.data_mut(|d| {
                            d.insert_temp(
                                egui::Id::new(("workspace_size_request", panel.panel.as_str())),
                                size.max(egui::vec2(80.0, 60.0)),
                            )
                        });
                    }
                    ctx.data_mut(|d| {
                        d.remove::<egui::Vec2>(egui::Id::new((
                            "floating_header_drag",
                            panel.panel.as_str(),
                        )));
                        d.remove::<egui::Pos2>(egui::Id::new((
                            "floating_header_pointer",
                            panel.panel.as_str(),
                        )));
                    });
                }
            }
        }
        ctx.request_repaint();
        self.persist_window_state_if_changed(ctx.input(|i| i.time), true);
    }

    fn reset_ui_cursor_blocking_rects(&mut self) {
        self.ui_cursor_blocking_rects_next.clear();
    }

    fn remember_ui_cursor_rect(&mut self, rect: egui::Rect) {
        if rect.is_positive() {
            self.ui_cursor_blocking_rects_next.push(rect.expand(12.0));
        }
    }

    fn publish_ui_cursor_blocking_rects(&mut self) {
        std::mem::swap(
            &mut self.ui_cursor_blocking_rects,
            &mut self.ui_cursor_blocking_rects_next,
        );
        self.ui_cursor_blocking_rects_next.clear();
    }

    fn pointer_over_cursor_blocking_ui(&self, ctx: &egui::Context) -> bool {
        if crate::ui::polish::popup_blocks_input(ctx) {
            return true;
        }
        ctx.input(|i| {
            i.pointer
                .hover_pos()
                .or_else(|| i.pointer.interact_pos())
                .is_some_and(|pos| {
                    self.ui_cursor_blocking_rects
                        .iter()
                        .chain(self.ui_cursor_blocking_rects_next.iter())
                        .any(|rect| rect.contains(pos))
                })
        })
    }

    fn update_ui_pointer_capture(&mut self, ctx: &egui::Context) -> bool {
        let popup_blocks_input = crate::ui::polish::popup_blocks_input(ctx);
        let started_over_ui = ctx.input(|i| {
            i.events.iter().any(|event| {
                if let egui::Event::PointerButton {
                    pos, pressed: true, ..
                } = event
                {
                    popup_blocks_input
                        || self
                            .ui_cursor_blocking_rects
                            .iter()
                            .chain(self.ui_cursor_blocking_rects_next.iter())
                            .any(|rect| rect.contains(*pos))
                } else {
                    false
                }
            })
        });
        if started_over_ui {
            self.ui_pointer_capture_active = true;
        }
        let any_down = ctx.input(|i| i.pointer.any_down());
        if !any_down && !started_over_ui {
            self.ui_pointer_capture_active = false;
            return false;
        }

        let pointer_started = ctx.input(|i| i.pointer.any_pressed());
        let over_blocking_ui = self.pointer_over_cursor_blocking_ui(ctx);

        if pointer_started && over_blocking_ui {
            self.ui_pointer_capture_active = true;
        }

        self.ui_pointer_capture_active || over_blocking_ui
    }

    fn clamp_floating_pos(
        pos_x: f32,
        pos_y: f32,
        size: egui::Vec2,
        screen_rect: egui::Rect,
    ) -> egui::Pos2 {
        let margin = 8.0;
        let min_x = screen_rect.min.x + margin;
        let min_y = screen_rect.min.y + margin;
        let max_x = (screen_rect.max.x - size.x - margin).max(min_x);
        let max_y = (screen_rect.max.y - size.y - margin).max(min_y);
        egui::pos2(pos_x.clamp(min_x, max_x), pos_y.clamp(min_y, max_y))
    }

    /// Show the floating Tools panel (minimalist vertical strip) - anchored to left edge
    fn show_floating_tools_panel(&mut self, ctx: &egui::Context, _screen_size_changed: bool) {
        let _profile = crate::ui::perf::Scope::new(8);
        self.tools_panel.cursor_blocking_rect = None;
        let mut show = self.window_visibility.tools;
        let mut close_clicked = false;

        let _first_show = self.tools_panel_pos.is_none();
        let screen_rect = ctx.content_rect();
        let (pos_x, pos_y) = self.tools_panel_pos.unwrap_or((12.0, 128.0));

        let hover_id = egui::Id::new("ToolsStrip_hover");
        let hover_t = crate::ui::polish::panel_hover(ctx, hover_id);
        let mut window = crate::ui::polish::window(ctx, "ToolsStrip")
            .open(&mut show)
            .movable(false)
            .resizable(false)
            .collapsible(false)
            .default_size(egui::vec2(120.0, 400.0))
            .max_width(114.0)
            .title_bar(false)
            .frame(self.theme.floating_window_frame_animated(hover_t));

        // Always enforce anchored position so the panel stays put on resize
        {
            let clamped =
                Self::clamp_floating_pos(pos_x, pos_y, egui::vec2(114.0, 400.0), screen_rect);
            window = window.current_pos(clamped);
        }

        window = window.current_pos(crate::ui::workspace::place(
            ctx,
            "Tools",
            egui::pos2(pos_x, pos_y),
            egui::vec2(114.0, 400.0),
        ));
        if let Some(size) = ctx.data_mut(|d| {
            d.remove_temp::<egui::Vec2>(egui::Id::new(("workspace_size_request", "Tools")))
        }) {
            window = window.fixed_size(size);
        }
        let resp = window.show(ctx, |ui| {
            // Constrain content width to match the tool grid (3×26 + 2×6 = 90px)
            // so the header doesn't inflate the window wider than the buttons.
            ui.set_max_width(90.0);

            // Signal Grid panel header
            if signal_widgets::panel_header(
                ui,
                &self.theme,
                "Tools",
                Some(("TOOLS", self.theme.accent3)),
            ) {
                close_clicked = true;
            }
            // Make all text in this window slightly smaller
            ui.style_mut().override_text_style = Some(egui::TextStyle::Small);
            let primary = self.colors_panel.get_primary_color();
            let secondary = self.colors_panel.get_secondary_color();

            let is_text_layer = self
                .projects
                .get(self.active_project_index)
                .map(|p| {
                    p.canvas_state
                        .layers
                        .get(p.canvas_state.active_layer_index)
                        .is_some_and(|l| l.is_text_layer())
                })
                .unwrap_or(false);

            let action = self.tools_panel.show_compact(
                ui,
                &self.assets,
                primary,
                secondary,
                &self.settings.keybindings,
                is_text_layer,
            );

            match action {
                tools::ToolsPanelAction::OpenColors => {
                    self.window_visibility.colors = !self.window_visibility.colors;
                }
                tools::ToolsPanelAction::SwapColors => {
                    self.colors_panel.swap_colors();
                }
                tools::ToolsPanelAction::None => {}
            }
        });

        if let Some(inner_resp) = resp {
            let win_rect = inner_resp.response.rect;
            crate::ui::workspace::register(ctx, "Tools", win_rect);
            self.remember_ui_cursor_rect(win_rect);
            self.tools_panel_pos = Some((win_rect.min.x, win_rect.min.y));
            let hovered =
                ctx.input(|i| i.pointer.hover_pos().is_some_and(|p| win_rect.contains(p)));
            crate::ui::polish::remember_hover(ctx, hover_id, hovered);
        }
        if let Some(rect) = self.tools_panel.cursor_blocking_rect {
            self.remember_ui_cursor_rect(rect);
        }

        if close_clicked {
            show = false;
        }
        self.window_visibility.tools = show;
    }

    /// Show the floating Layers panel
    fn show_floating_layers_panel(&mut self, ctx: &egui::Context, _screen_size_changed: bool) {
        let _profile = crate::ui::perf::Scope::new(9);
        self.layers_panel.settings_popup_rect = None;
        let mut show = self.window_visibility.layers;
        let mut close_clicked = false;
        self.is_pointer_over_layers_panel = false;

        let screen_rect = ctx.content_rect();
        let screen_w = screen_rect.max.x;

        let _first_show = self.layers_panel_right_offset.is_none();

        // Default: 12px from right edge, 12px below menu bar
        let (right_off, y_pos) = self.layers_panel_right_offset.unwrap_or((264.0, 128.0));
        let pos_x = screen_w - right_off;
        let panel_size = self.layers_panel_size.unwrap_or((240.0, 200.0));

        let hover_id = egui::Id::new("Layers_hover");
        let hover_t = crate::ui::polish::panel_hover(ctx, hover_id);
        let mut window = crate::ui::polish::window(ctx, "Layers")
            .open(&mut show)
            .movable(false)
            .resizable(true)
            .collapsible(false)
            .default_size(egui::vec2(panel_size.0, panel_size.1))
            .min_width(180.0)
            .min_height(200.0)
            .title_bar(false)
            .frame(self.theme.floating_window_frame_animated(hover_t));

        // Always enforce anchored position so the panel stays put on resize
        {
            let clamped =
                Self::clamp_floating_pos(pos_x, y_pos, egui::vec2(180.0, 200.0), screen_rect);
            window = window.current_pos(clamped);
        }

        window = window.current_pos(crate::ui::workspace::place(
            ctx,
            "Layers",
            egui::pos2(pos_x, y_pos),
            egui::vec2(panel_size.0, panel_size.1),
        ));
        if let Some(size) = ctx.data_mut(|d| {
            d.remove_temp::<egui::Vec2>(egui::Id::new(("workspace_size_request", "Layers")))
        }) {
            window = window.fixed_size(size);
        }
        let resp = window.show(ctx, |ui| {
            // Signal Grid panel header
            if signal_widgets::panel_header(
                ui,
                &self.theme,
                "Layers",
                Some(("LAYERS", self.theme.accent3)),
            ) {
                close_clicked = true;
            }

            if let Some(project) = self.projects.get_mut(self.active_project_index) {
                self.layers_panel.set_thumbnail_project(project.id);
                self.layers_panel.show(
                    ui,
                    &mut project.canvas_state,
                    &mut self.assets,
                    &self.settings,
                    &mut project.history,
                );

                // If the layer being text-edited was rasterized inside show()
                // (e.g. via right-click → "Rasterize Text Layer"), cancel any
                // stale text editing state so tools don't remain locked.
                if project.canvas_state.text_editing_layer.is_some_and(|idx| {
                    project
                        .canvas_state
                        .layers
                        .get(idx)
                        .is_some_and(|l| !l.is_text_layer())
                }) {
                    self.tools_panel
                        .cancel_text_editing(&mut project.canvas_state);
                }

                if let Some(deleted_idx) = self.layers_panel.pending_deleted_layer
                    && self.tools_panel.text_state.is_editing
                    && !self.tools_panel.text_state.editing_text_layer
                {
                    match self.tools_panel.text_state.editing_layer_index {
                        Some(idx) if idx == deleted_idx => {
                            self.tools_panel
                                .cancel_text_editing(&mut project.canvas_state);
                        }
                        Some(idx) if deleted_idx < idx => {
                            self.tools_panel.text_state.editing_layer_index = Some(idx - 1);
                        }
                        _ => {}
                    }
                }

                if self.tools_panel.text_state.is_editing
                    && !self.tools_panel.text_state.editing_text_layer
                    && self
                        .tools_panel
                        .text_state
                        .editing_layer_index
                        .is_none_or(|idx| idx >= project.canvas_state.layers.len())
                {
                    self.tools_panel
                        .cancel_text_editing(&mut project.canvas_state);
                }

                // Auto-switch tool immediately when layer selection changes
                // (same frame as click, no 1-frame delay).
                self.tools_panel
                    .auto_switch_tool_for_layer(&project.canvas_state);
            }

            // Cancel paste overlay if the layer it was on was deleted.
            // Must be after layers_panel.show() so pending_deleted_layer is set.
            if self.paste_overlay.is_some() && self.layers_panel.pending_deleted_layer.is_some() {
                self.cancel_paste_overlay();
            }
            // Drain pending GPU delete from the layers panel.
            self.layers_panel.pending_deleted_layer = None;
            if let Some(del_idx) = self.layers_panel.pending_gpu_delete.take() {
                self.canvas.gpu_remove_layer(del_idx);
            }
            if self.layers_panel.pending_gpu_clear {
                self.layers_panel.pending_gpu_clear = false;
                self.canvas.gpu_clear_layers();
            }
            // Handle pending app-level actions from layers context menu
            if let Some(action) = self.layers_panel.pending_app_action.take() {
                match action {
                    // Not available on web — drag & drop an image onto the
                    // canvas to import it as a new layer instead.
                    crate::components::layers::LayerAppAction::ImportFromFile => {
                        #[cfg(not(target_arch = "wasm32"))]
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter(
                                "Image",
                                &[
                                    "png", "jpg", "jpeg", "bmp", "gif", "webp", "tiff", "tif",
                                    "tga", "ico",
                                ],
                            )
                            .pick_file()
                            && let Ok(img) = image::open(&path)
                        {
                            let rgba = img.to_rgba8();
                            let name = path
                                .file_stem()
                                .map(|s| s.to_string_lossy().to_string())
                                .unwrap_or_else(|| "Imported".to_string());
                            self.do_snapshot_op("Import Layer", |s| {
                                crate::ops::adjustments::import_layer_from_image(s, &rgba, &name);
                            });
                        }
                    }
                    crate::components::layers::LayerAppAction::FlipHorizontal => {
                        self.do_layer_snapshot_op("Flip Layer H", |s| {
                            let idx = s.active_layer_index;
                            crate::ops::transform::flip_layer_horizontal(s, idx);
                        });
                    }
                    crate::components::layers::LayerAppAction::FlipVertical => {
                        self.do_layer_snapshot_op("Flip Layer V", |s| {
                            let idx = s.active_layer_index;
                            crate::ops::transform::flip_layer_vertical(s, idx);
                        });
                    }
                    crate::components::layers::LayerAppAction::RotateScale => {
                        if let Some(project) = self.projects.get(self.active_project_index) {
                            self.active_dialog = ActiveDialog::LayerTransform(
                                crate::ops::dialogs::LayerTransformDialog::new(
                                    &project.canvas_state,
                                ),
                            );
                        }
                    }
                    crate::components::layers::LayerAppAction::AlignLayer => {
                        if let Some(project) = self.projects.get(self.active_project_index) {
                            self.active_dialog = ActiveDialog::AlignLayer(
                                crate::ops::dialogs::AlignLayerDialog::new(&project.canvas_state),
                            );
                        }
                    }
                    crate::components::layers::LayerAppAction::MergeDownAsMask(layer_idx) => {
                        self.do_snapshot_op("Merge Down as Mono-Mask", |s| {
                            crate::ops::canvas_ops::merge_down_as_mask(s, layer_idx);
                        });
                        self.layers_panel.pending_gpu_clear = true;
                    }
                    crate::components::layers::LayerAppAction::AddLayerMaskRevealAll(layer_idx) => {
                        self.do_layer_snapshot_op("Add Layer Mask", |s| {
                            crate::ops::canvas_ops::add_layer_mask_reveal_all(s, layer_idx);
                        });
                        self.layers_panel.pending_gpu_clear = true;
                    }
                    crate::components::layers::LayerAppAction::AddLayerMaskFromSelection(
                        layer_idx,
                    ) => {
                        self.do_layer_snapshot_op("Mask From Selection", |s| {
                            crate::ops::canvas_ops::add_layer_mask_from_selection(s, layer_idx);
                        });
                        self.layers_panel.pending_gpu_clear = true;
                    }
                    crate::components::layers::LayerAppAction::ToggleLayerMaskEdit(layer_idx) => {
                        if let Some(project) = self.projects.get_mut(self.active_project_index)
                            && layer_idx < project.canvas_state.layers.len()
                        {
                            project.canvas_state.active_layer_index = layer_idx;
                            let has_mask = project.canvas_state.layers[layer_idx].has_live_mask();
                            if has_mask {
                                let currently_editing = project.canvas_state.edit_layer_mask
                                    && project.canvas_state.active_layer_index == layer_idx;
                                project.canvas_state.edit_layer_mask = !currently_editing;
                            } else {
                                project.canvas_state.edit_layer_mask = false;
                            }
                        }
                    }
                    crate::components::layers::LayerAppAction::ToggleLayerMask(layer_idx) => {
                        self.do_layer_snapshot_op("Toggle Layer Mask", |s| {
                            crate::ops::canvas_ops::toggle_layer_mask(s, layer_idx);
                        });
                        self.layers_panel.pending_gpu_clear = true;
                    }
                    crate::components::layers::LayerAppAction::InvertLayerMask(layer_idx) => {
                        self.do_layer_snapshot_op("Invert Layer Mask", |s| {
                            crate::ops::canvas_ops::invert_layer_mask(s, layer_idx);
                        });
                        self.layers_panel.pending_gpu_clear = true;
                    }
                    crate::components::layers::LayerAppAction::ApplyLayerMask(layer_idx) => {
                        self.do_layer_snapshot_op("Apply Layer Mask", |s| {
                            crate::ops::canvas_ops::apply_layer_mask(s, layer_idx);
                        });
                        self.layers_panel.pending_gpu_clear = true;
                    }
                    crate::components::layers::LayerAppAction::DeleteLayerMask(layer_idx) => {
                        self.do_layer_snapshot_op("Delete Layer Mask", |s| {
                            crate::ops::canvas_ops::delete_layer_mask(s, layer_idx);
                        });
                        self.layers_panel.pending_gpu_clear = true;
                    }
                    crate::components::layers::LayerAppAction::RasterizeTextLayer(layer_idx) => {
                        if let Some(project) = self.projects.get_mut(self.active_project_index) {
                            self.layers_panel.rasterize_text_layer_from_app(
                                layer_idx,
                                &mut project.canvas_state,
                                &mut project.history,
                            );
                            // Clean up any active text editing session so the text tool
                            // does not remain in editing mode on the now-raster layer.
                            self.tools_panel
                                .cancel_text_editing(&mut project.canvas_state);
                        }
                    }
                }
            }
        });

        // Update the stored right-edge offset from the window's actual position
        // so that user drags are remembered and window resizes keep the offset.
        if let Some(inner_resp) = resp {
            let win_rect = inner_resp.response.rect;
            crate::ui::workspace::register(ctx, "Layers", win_rect);
            if let Some(size) = crate::signal_widgets::floating_resize(
                ctx,
                &inner_resp.response,
                "Layers",
                egui::vec2(180.0, 140.0),
            ) {
                ctx.data_mut(|d| {
                    d.insert_temp(egui::Id::new(("workspace_size_request", "Layers")), size)
                });
                ctx.request_repaint();
            }
            self.remember_ui_cursor_rect(win_rect);
            self.layers_panel_right_offset = Some((screen_w - win_rect.min.x, win_rect.min.y));
            let observed_size = (win_rect.width().round(), win_rect.height().round());
            let pointer_down = ctx.input(|i| i.pointer.primary_down());
            let should_store_size = pointer_down
                && self.layers_panel_size.is_none_or(|old| {
                    (old.0 - observed_size.0).abs() >= 4.0 || (old.1 - observed_size.1).abs() >= 4.0
                });
            if should_store_size {
                self.layers_panel_size = Some(observed_size);
            }
            let hovered =
                ctx.input(|i| i.pointer.hover_pos().is_some_and(|p| win_rect.contains(p)));
            self.is_pointer_over_layers_panel = hovered;
            crate::ui::polish::remember_hover(ctx, hover_id, hovered);
        }
        if let Some(rect) = self.layers_panel.settings_popup_rect {
            self.remember_ui_cursor_rect(rect);
        }

        if close_clicked {
            show = false;
        }
        self.window_visibility.layers = show;
    }

    /// Show the floating History panel
    fn show_floating_history_panel(&mut self, ctx: &egui::Context, screen_size_changed: bool) {
        let _profile = crate::ui::perf::Scope::new(10);
        let mut show = self.window_visibility.history;
        let mut close_clicked = false;

        let screen_rect = ctx.content_rect();
        let screen_w = screen_rect.max.x;
        let screen_h = screen_rect.max.y;

        let first_show = self.history_panel_right_offset.is_none();
        let saved_size = self.history_panel_size.unwrap_or((200.0, 200.0));
        let panel_size = egui::vec2(saved_size.0, saved_size.1);
        let (right_off, mut y_pos) = self
            .history_panel_right_offset
            .unwrap_or((12.0, screen_rect.center().y - panel_size.y * 0.5));
        let mut pos_x = screen_w - right_off;
        let drag_delta = egui::Vec2::ZERO;
        pos_x += drag_delta.x;
        y_pos += drag_delta.y;

        let hover_id = egui::Id::new("History_hover");
        let hover_t = crate::ui::polish::panel_hover(ctx, hover_id);
        let mut window = crate::ui::polish::window(ctx, "History")
            .open(&mut show)
            .movable(false)
            .resizable(true)
            .collapsible(false)
            .min_width(200.0)
            .min_height(150.0)
            .max_width(screen_w * 0.5)
            .max_height(screen_h * 0.7)
            .default_size(panel_size)
            .title_bar(false)
            .frame(self.theme.floating_window_frame_animated(hover_t));

        if let Some(size) =
            ctx.data_mut(|d| d.remove_temp::<egui::Vec2>(egui::Id::new("history_resize_request")))
        {
            window = window.fixed_size(size);
        }

        if first_show || screen_size_changed || drag_delta != egui::Vec2::ZERO {
            let clamped = Self::clamp_floating_pos(pos_x, y_pos, panel_size, screen_rect);
            window = window.current_pos(clamped);
        }

        window = window.current_pos(crate::ui::workspace::place(
            ctx,
            "History",
            egui::pos2(pos_x, y_pos),
            panel_size,
        ));
        if let Some(size) = ctx.data_mut(|d| {
            d.remove_temp::<egui::Vec2>(egui::Id::new(("workspace_size_request", "History")))
        }) {
            window = window.fixed_size(size);
        }
        let resp = window.show(ctx, |ui| {
            // Signal Grid panel header
            if signal_widgets::panel_header(
                ui,
                &self.theme,
                "History",
                Some(("HISTORY", self.theme.accent4)),
            ) {
                close_clicked = true;
            }
            ui.style_mut().override_text_style = Some(egui::TextStyle::Small);
            if let Some(project) = self.projects.get_mut(self.active_project_index) {
                self.history_panel.show_interactive(
                    ui,
                    &mut project.history,
                    &mut project.canvas_state,
                    &self.assets,
                    &self.theme,
                );
            }
        });

        if let Some(inner_resp) = resp {
            let win_rect = inner_resp.response.rect;
            crate::ui::workspace::register(ctx, "History", win_rect);
            self.remember_ui_cursor_rect(win_rect);
            self.history_panel_right_offset = Some((screen_w - win_rect.min.x, win_rect.min.y));
            if let Some(size) = signal_widgets::floating_resize(
                ctx,
                &inner_resp.response,
                "History",
                egui::vec2(200.0, 150.0),
            ) {
                self.history_panel_size = Some((size.x, size.y));
                ctx.data_mut(|d| d.insert_temp(egui::Id::new("history_resize_request"), size));
                ctx.request_repaint();
            }
            let observed_size = (win_rect.width().round(), win_rect.height().round());
            let pointer_down = ctx.input(|i| i.pointer.primary_down());
            let should_store_size = pointer_down
                && self.history_panel_size.is_none_or(|old| {
                    (old.0 - observed_size.0).abs() >= 4.0 || (old.1 - observed_size.1).abs() >= 4.0
                });
            if should_store_size {
                self.history_panel_size = Some(observed_size);
            }
            let hovered =
                ctx.input(|i| i.pointer.hover_pos().is_some_and(|p| win_rect.contains(p)));
            crate::ui::polish::remember_hover(ctx, hover_id, hovered);
        }

        if close_clicked {
            show = false;
        }
        self.window_visibility.history = show;
    }

    /// Show the floating Colors panel - anchored below tools
    fn show_floating_colors_panel(&mut self, ctx: &egui::Context, screen_size_changed: bool) {
        let _profile = crate::ui::perf::Scope::new(11);
        let mut show = self.window_visibility.colors;
        let mut close_clicked = false;

        let screen_rect = ctx.content_rect();
        let screen_h = screen_rect.max.y;

        let panel_size = egui::vec2(220.0, 310.0);
        let first_show = self.colors_panel_pos.is_none();
        let (mut x_off, mut pos_y) = self.colors_panel_pos.unwrap_or_else(|| {
            let (x, bottom) = self
                .settings
                .persist_colors_panel_left_offset
                .unwrap_or((12.0, 12.0));
            (x, screen_h - bottom - panel_size.y)
        });

        let drag_delta = egui::Vec2::ZERO;
        x_off += drag_delta.x;
        pos_y += drag_delta.y;
        let hover_id = egui::Id::new("Colors_hover");
        let hover_t = crate::ui::polish::panel_hover(ctx, hover_id);
        let mut window = crate::ui::polish::window(ctx, "Colors")
            .open(&mut show)
            .movable(false)
            // Our top-left clamp and body scroll bounds own positioning. egui's
            // size-based constraint otherwise moves the header when Advanced grows.
            .constrain(false)
            .resizable(false)
            .collapsible(false)
            .auto_sized()
            .title_bar(false)
            .frame(self.theme.floating_window_frame_animated(hover_t));

        let clamped = Self::clamp_floating_pos(x_off, pos_y, panel_size, screen_rect);
        window = window.default_pos(clamped);
        if first_show || screen_size_changed || drag_delta != egui::Vec2::ZERO {
            window = window.current_pos(clamped);
        }

        window = window.current_pos(crate::ui::workspace::place(
            ctx,
            "Colors",
            egui::pos2(x_off, pos_y),
            panel_size,
        ));
        if let Some(size) = ctx.data_mut(|d| {
            d.remove_temp::<egui::Vec2>(egui::Id::new(("workspace_size_request", "Colors")))
        }) {
            window = window.fixed_size(size);
        }
        let resp = window.show(ctx, |ui| {
            // Reserve the body width before the header; fixed-size windows can
            // otherwise grow later when the color controls are laid out.
            ui.set_width(panel_size.x);
            // Signal Grid panel header
            if signal_widgets::panel_header(
                ui,
                &self.theme,
                "Colors",
                Some(("COLOR", self.theme.accent)),
            ) {
                close_clicked = true;
            }
            ui.style_mut().override_text_style = Some(egui::TextStyle::Small);
            egui::ScrollArea::vertical()
                .id_salt("color_panel_scroll")
                .max_height((screen_rect.max.y - clamped.y - 50.0).max(100.0))
                .show(ui, |ui| {
                    self.colors_panel.show(ui, &self.assets);
                });
        });

        if let Some(inner_resp) = resp {
            let win_rect = inner_resp.response.rect;
            crate::ui::workspace::register(ctx, "Colors", win_rect);
            self.remember_ui_cursor_rect(win_rect);
            self.colors_panel_pos = Some((win_rect.min.x, win_rect.min.y));
            let hovered =
                ctx.input(|i| i.pointer.hover_pos().is_some_and(|p| win_rect.contains(p)));
            crate::ui::polish::remember_hover(ctx, hover_id, hovered);
        }

        if close_clicked {
            show = false;
        }
        self.window_visibility.colors = show;
    }

    /// Show the floating Script Editor panel
    fn show_floating_palette_panel(&mut self, ctx: &egui::Context, screen_size_changed: bool) {
        let _profile = crate::ui::perf::Scope::new(12);
        let mut show = self.window_visibility.palette;
        let mut close_clicked = false;

        let screen_rect = ctx
            .input(|i| i.viewport().inner_rect)
            .unwrap_or_else(|| ctx.content_rect());

        let saved_size = self.palette_panel_size.unwrap_or((302.0, 108.0));
        let panel_size = egui::vec2(saved_size.0, saved_size.1);
        let first_show = self.palette_panel_pos.is_none();
        let source_pos = if self.palette_reposition_settle_frames > 0 {
            self.palette_startup_target_pos.or(self.palette_panel_pos)
        } else {
            self.palette_panel_pos
        };
        let (mut pos_x, mut pos_y) = source_pos.unwrap_or((
            screen_rect.max.x - panel_size.x - 8.0,
            screen_rect.max.y - panel_size.y - 8.0,
        ));

        let drag_delta = egui::Vec2::ZERO;
        pos_x += drag_delta.x;
        pos_y += drag_delta.y;
        let hover_id = egui::Id::new("Palette_hover");
        let hover_t = crate::ui::polish::panel_hover(ctx, hover_id);
        let mut window = crate::ui::polish::window(ctx, "Palette")
            .open(&mut show)
            .movable(false)
            .resizable(true)
            .collapsible(false)
            .min_size(egui::vec2(280.0, 108.0))
            .default_size(panel_size)
            .title_bar(false)
            .frame(self.theme.floating_window_frame_animated(hover_t));

        if let Some(size) =
            ctx.data_mut(|d| d.remove_temp::<egui::Vec2>(egui::Id::new("palette_resize_request")))
        {
            window = window.fixed_size(size);
        }

        let should_reposition = first_show
            || screen_size_changed
            || self.palette_reposition_settle_frames > 0
            || drag_delta != egui::Vec2::ZERO;

        if should_reposition {
            // Keep palette where user placed it. Only apply a relaxed clamp so it remains
            // reachable if monitor/layout changed, without forcing it inward each launch.
            let min_x = screen_rect.min.x - panel_size.x + 24.0;
            let min_y = screen_rect.min.y;
            let max_x = screen_rect.max.x - 24.0;
            let max_y = screen_rect.max.y - 24.0;
            let clamped = egui::pos2(pos_x.clamp(min_x, max_x), pos_y.clamp(min_y, max_y));
            window = window.current_pos(clamped);
        }

        window = window.current_pos(crate::ui::workspace::place(
            ctx,
            "Palette",
            egui::pos2(pos_x, pos_y),
            panel_size,
        ));
        if let Some(size) = ctx.data_mut(|d| {
            d.remove_temp::<egui::Vec2>(egui::Id::new(("workspace_size_request", "Palette")))
        }) {
            window = window.fixed_size(size);
        }
        let resp = window.show(ctx, |ui| {
            if signal_widgets::panel_header(
                ui,
                &self.theme,
                "Palette",
                Some(("PALETTE", self.theme.accent4)),
            ) {
                close_clicked = true;
            }
            ui.style_mut().override_text_style = Some(egui::TextStyle::Small);
            if let Some((color, secondary)) = self.palette_panel.show(
                ui,
                &self.assets,
                self.colors_panel.get_primary_color(),
                self.colors_panel.get_secondary_color(),
                self.colors_panel.editing_secondary(),
            ) {
                if secondary {
                    self.colors_panel.set_secondary_color(color);
                } else {
                    self.colors_panel.set_primary_color(color);
                }
            }
        });

        if let Some(inner_resp) = resp {
            inner_resp.response.context_menu(|ui| {
                if ui.button("Save Palette").clicked() {
                    self.palette_panel.save_palette_dialog();
                    ui.close();
                }
                if ui.button("Load Palette").clicked() {
                    self.palette_panel.load_palette_dialog();
                    ui.close();
                }
                if ui.button("Reset Palette").clicked() {
                    self.palette_panel.reset_palette_default();
                    ui.close();
                }
                if ui.button("Reset Recents").clicked() {
                    self.palette_panel.reset_recent_default();
                    ui.close();
                }
            });

            let win_rect = inner_resp.response.rect;
            crate::ui::workspace::register(ctx, "Palette", win_rect);
            if let Some(size) = signal_widgets::floating_resize(
                ctx,
                &inner_resp.response,
                "Palette",
                egui::vec2(280.0, 108.0),
            ) {
                self.palette_panel_size = Some((size.x, size.y));
                ctx.data_mut(|d| d.insert_temp(egui::Id::new("palette_resize_request"), size));
                ctx.request_repaint();
            }
            if ctx.input(|i| i.pointer.primary_down()) {
                self.palette_panel_size = Some((win_rect.width(), win_rect.height()));
            }
            self.remember_ui_cursor_rect(win_rect);
            if self.palette_reposition_settle_frames == 0 {
                self.palette_panel_pos = Some((win_rect.min.x, win_rect.min.y));
                self.palette_startup_target_pos = self.palette_panel_pos;
            }
            let hovered =
                ctx.input(|i| i.pointer.hover_pos().is_some_and(|p| win_rect.contains(p)));
            crate::ui::polish::remember_hover(ctx, hover_id, hovered);
        }

        if close_clicked {
            show = false;
        }
        self.window_visibility.palette = show;
    }

    /// Show the floating Script Editor panel
    fn show_floating_script_editor(&mut self, ctx: &egui::Context, screen_size_changed: bool) {
        let _profile = crate::ui::perf::Scope::new(13);
        let mut show = self.window_visibility.script_editor;
        if !show {
            return;
        }

        let screen_rect = ctx.content_rect();
        let screen_w = screen_rect.max.x;
        let screen_h = screen_rect.max.y;

        let first_show = self.script_right_offset.is_none();

        // Default: centered-ish position
        let (right_off, top_off) = self
            .script_right_offset
            .unwrap_or((screen_w * 0.3, screen_h * 0.15));
        let pos_x = screen_w - right_off;
        let pos_y = top_off;

        let hover_id = egui::Id::new("ScriptEditor_hover");
        let hover_t = crate::ui::polish::panel_hover(ctx, hover_id);
        let mut window = crate::ui::polish::window(ctx, "ScriptEditor")
            .open(&mut show)
            .movable(false)
            .resizable(true)
            .collapsible(false)
            .min_width(400.0)
            .min_height(350.0)
            .default_size(egui::vec2(520.0, 500.0))
            .title_bar(false)
            .frame(self.theme.floating_window_frame_animated(hover_t));

        if first_show || screen_size_changed {
            let clamped =
                Self::clamp_floating_pos(pos_x, pos_y, egui::vec2(400.0, 350.0), screen_rect);
            window = window.current_pos(clamped);
        }

        let theme_copy = self.theme.clone();
        window = window.current_pos(crate::ui::workspace::place(
            ctx,
            "ScriptEditor",
            egui::pos2(pos_x, pos_y),
            egui::vec2(520.0, 500.0),
        ));
        if let Some(size) = ctx.data_mut(|d| {
            d.remove_temp::<egui::Vec2>(egui::Id::new(("workspace_size_request", "ScriptEditor")))
        }) {
            window = window.fixed_size(size);
        }
        let resp = window.show(ctx, |ui| {
            self.script_editor.show(ui, &theme_copy);

            // Handle run request
            if self.script_editor.run_requested {
                self.run_script();
            }
        });

        // Handle "Add to Filters" request from the editor
        if let Some((name, code)) = self.script_editor.pending_add_effect.take() {
            let effect = script_editor::CustomScriptEffect { name, code };
            script_editor::save_custom_effect(&effect);
            // Avoid duplicates by name
            self.custom_scripts.retain(|e| e.name != effect.name);
            self.custom_scripts.push(effect);
            self.custom_scripts.sort_by_key(|a| a.name.to_lowercase());
        }

        if let Some(inner_resp) = resp {
            let win_rect = inner_resp.response.rect;
            crate::ui::workspace::register(ctx, "ScriptEditor", win_rect);
            if let Some(size) = crate::signal_widgets::floating_resize(
                ctx,
                &inner_resp.response,
                "ScriptEditor",
                egui::vec2(320.0, 240.0),
            ) {
                ctx.data_mut(|d| {
                    d.insert_temp(
                        egui::Id::new(("workspace_size_request", "ScriptEditor")),
                        size,
                    )
                });
                ctx.request_repaint();
            }
            self.remember_ui_cursor_rect(win_rect);
            self.script_right_offset = Some((screen_w - win_rect.min.x, win_rect.min.y));
            let hovered =
                ctx.input(|i| i.pointer.hover_pos().is_some_and(|p| win_rect.contains(p)));
            crate::ui::polish::remember_hover(ctx, hover_id, hovered);
        }

        if self.script_editor.close_requested {
            show = false;
        }
        self.window_visibility.script_editor = show;
    }

    /// Execute the current script on the active layer
    fn run_script(&mut self) {
        if self.script_editor.is_running {
            return;
        }

        let project = match self.projects.get(self.active_project_index) {
            Some(p) => p,
            None => {
                self.script_editor.add_console_line(
                    "No active project".to_string(),
                    crate::components::script_editor::ConsoleLineKind::Error,
                );
                return;
            }
        };

        let state = &project.canvas_state;
        let layer_idx = state.active_layer_index;
        if layer_idx >= state.layers.len() {
            self.script_editor.add_console_line(
                "No active layer".to_string(),
                crate::components::script_editor::ConsoleLineKind::Error,
            );
            return;
        }

        let layer = &state.layers[layer_idx];
        let w = state.width;
        let h = state.height;

        // Extract flat pixels from tiled image
        let flat_pixels = layer.pixels.extract_region_rgba(0, 0, w, h);

        // Clone original for undo
        let original_pixels = layer.pixels.clone();

        // Get selection mask if any
        let mask = state.selection_mask.as_ref().map(|m| m.as_raw().clone());

        // Reset cancel flag
        let cancel_flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        self.script_editor.cancel_flag = cancel_flag.clone();

        // Save backup of layer pixels for restore on error
        self.script_original_pixels =
            Some((self.active_project_index, layer_idx, layer.pixels.clone()));

        self.script_editor.is_running = true;
        self.script_editor.progress = None;
        self.script_editor.error_line = None; // Clear previous error highlight
        self.script_editor.add_console_line(
            "Running script...".to_string(),
            crate::components::script_editor::ConsoleLineKind::Info,
        );

        crate::ops::scripting::execute_script(
            self.script_editor.code.clone(),
            self.active_project_index,
            layer_idx,
            original_pixels,
            flat_pixels,
            w,
            h,
            mask,
            cancel_flag,
            self.script_sender.clone(),
        );
    }

    /// Execute a custom script effect from Filter > Custom (no editor needed)
    fn run_custom_script(&mut self, code: String, name: String) {
        if self.script_editor.is_running {
            return;
        }

        let project = match self.projects.get(self.active_project_index) {
            Some(p) => p,
            None => return,
        };

        let state = &project.canvas_state;
        let layer_idx = state.active_layer_index;
        if layer_idx >= state.layers.len() {
            return;
        }

        let layer = &state.layers[layer_idx];
        let w = state.width;
        let h = state.height;

        let flat_pixels = layer.pixels.extract_region_rgba(0, 0, w, h);
        let original_pixels = layer.pixels.clone();
        let mask = state.selection_mask.as_ref().map(|m| m.as_raw().clone());

        let cancel_flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        self.script_editor.cancel_flag = cancel_flag.clone();
        // Save backup of layer pixels for restore on error
        self.script_original_pixels =
            Some((self.active_project_index, layer_idx, layer.pixels.clone()));

        self.script_editor.is_running = true;
        self.script_editor.progress = None;
        self.script_editor.error_line = None;
        self.script_editor.console_output.clear();
        self.script_editor.add_console_line(
            "Running custom effect...".to_string(),
            crate::components::script_editor::ConsoleLineKind::Info,
        );

        // Show spinner in status bar
        self.pending_filter_jobs += 1;
        self.filter_status_description = format!("Script: {}", name);

        crate::ops::scripting::execute_script(
            code,
            self.active_project_index,
            layer_idx,
            original_pixels,
            flat_pixels,
            w,
            h,
            mask,
            cancel_flag,
            self.script_sender.clone(),
        );
    }
}
