impl PaintFEApp {
    fn show_runtime_canvas_tail(&mut self, ctx: &egui::Context, modal_open: bool) {
        if let Some(mask) = self.pending_selection_reassert.take()
            && let Some(project) = self.active_project_mut()
        {
            project.canvas_state.selection_mask = Some(mask);
            project.canvas_state.invalidate_selection_overlay();
            project.canvas_state.mark_dirty(None);
            ctx.request_repaint();
        }

        let has_project = !self.projects.is_empty();
        let live_window_resize = crate::windows_key_probe::is_live_resize();
        let was_live_window_resize = ctx.data_mut(|d| {
            let id = egui::Id::new("paintfe_live_window_resize");
            let was = d.get_temp::<bool>(id).unwrap_or(false);
            d.insert_temp(id, live_window_resize);
            was
        });
        if was_live_window_resize && !live_window_resize {
            ctx.request_repaint();
        }

        // --- Floating Tool Shelf (replaces docked context bar) ---
        // Keep the strip itself transparent so the canvas/app backdrop remains
        // visible behind the floating shelf container.
        let shelf_margin = 6.0;
        #[allow(deprecated)]
        egui::Panel::top("tool_shelf_strip")
            .frame(egui::Frame::NONE.inner_margin(egui::Margin::same(shelf_margin as i8)))
            .min_size(30.0) // Allow growth so controls don't get vertically clipped on newer egui metrics
            .show(ctx, |ui| {
                let shelf_frame = self.theme.tool_shelf_frame();
                shelf_frame.show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    // Context bar label styling
                    ui.style_mut().override_font_id =
                        Some(egui::FontId::proportional(crate::theme::Theme::FONT_LABEL));
                    ui.visuals_mut().override_text_color = Some(self.theme.text_color);
                    ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                        if !has_project {
                            ui.disable();
                        }
                        if let Some(ref mut overlay) = self.paste_overlay {
                            // --- Paste overlay context bar ---
                            crate::signal_widgets::tool_shelf_tag(ui, "PASTE", self.theme.accent, &self.theme);
                            ui.add_space(6.0);

                            // Filter mode
                            ui.label("Filter:");
                            let current_interp = overlay.interpolation;
                            egui::ComboBox::from_id_salt("ctx_paste_filter")
                                .selected_text(current_interp.label())
                                .width(110.0)
                                .show_ui(ui, |ui| {
                                    for interp in crate::ops::transform::Interpolation::all() {
                                        if ui
                                            .selectable_label(
                                                *interp == current_interp,
                                                interp.label(),
                                            )
                                            .clicked()
                                        {
                                            overlay.interpolation = *interp;
                                        }
                                    }
                                });

                            ui.add_space(4.0);

                            // Anti-aliasing toggle
                            ui.checkbox(&mut overlay.anti_aliasing, "Anti-aliasing");

                            ui.add_space(4.0);

                            // Position info
                            ui.label(format!(
                                "X: {:.0}  Y: {:.0}  W: {:.0}  H: {:.0}  Rot: {:.1}°",
                                overlay.center.x
                                    - overlay.source.width() as f32 * overlay.scale_x / 2.0,
                                overlay.center.y
                                    - overlay.source.height() as f32 * overlay.scale_y / 2.0,
                                overlay.source.width() as f32 * overlay.scale_x,
                                overlay.source.height() as f32 * overlay.scale_y,
                                overlay.rotation.to_degrees(),
                            ));

                            ui.add_space(4.0);

                            // Quick actions
                            if ui
                                .button("Reset")
                                .on_hover_text("Reset all transforms")
                                .clicked()
                            {
                                overlay.rotation = 0.0;
                                overlay.scale_x = 1.0;
                                overlay.scale_y = 1.0;
                                overlay.anchor_offset = egui::Vec2::ZERO;
                            }
                        } else {
                            let ctx_primary = self.colors_panel.get_primary_color();
                            let ctx_secondary = self.colors_panel.get_secondary_color();
                            self.tools_panel.show_context_bar(
                                ui,
                                &self.assets,
                                ctx_primary,
                                ctx_secondary,
                                &self.theme,
                            );
                        }
                    });
                });
            });

        // Process pending brush tip actions from context bar
        if self.tools_panel.pending_open_add_brush_tip {
            self.tools_panel.pending_open_add_brush_tip = false;
            // Build category list from assets
            let cats: Vec<String> = self.assets.brush_tip_categories()
                .iter()
                .map(|c| c.name.clone())
                .collect();
            let mut dlg = crate::ui::dialogs::core::AddBrushTipDialog::new(&cats);
            dlg.brush_icon_texture = self.assets.get_texture(crate::config::icons::Icon::Brush).cloned();
            dlg.open_dialog();
            self.active_dialog = crate::ui::dialogs::core::ActiveDialog::AddBrushTip(dlg);
        }

        if let Some(tip_name) = self.tools_panel.pending_delete_brush_tip.take() {
            // If currently using this tip, reset to Circle
            if !self.tools_panel.properties.brush_tip.is_circle()
                && let crate::components::tools::BrushTip::Image(ref name) =
                    self.tools_panel.properties.brush_tip
                && name == &tip_name
            {
                self.tools_panel.properties.brush_tip =
                    crate::components::tools::BrushTip::Circle;
            }
            self.assets.remove_brush_tip(&tip_name);
            // Also remove from persisted settings
            self.settings.custom_brush_tips.retain(|(n, _, _)| n != &tip_name);
            self.settings.save();
        }

        if self.tools_panel.pending_open_add_shape {
            self.tools_panel.pending_open_add_shape = false;
            let cats: Vec<String> = self.assets.custom_shape_categories()
                .iter()
                .map(|c| c.name.clone())
                .collect();
            let mut dlg = crate::ui::dialogs::core::AddShapeDialog::new(&cats);
            dlg.open_dialog();
            self.active_dialog = crate::ui::dialogs::core::ActiveDialog::AddShape(dlg);
        }

        if let Some(shape_name) = self.tools_panel.pending_delete_shape.take() {
            if self.tools_panel.shapes_state.selected_custom_shape.as_ref() == Some(&shape_name) {
                self.tools_panel.shapes_state.selected_custom_shape = None;
                self.tools_panel.shapes_state.selected_custom_shape_data = None;
            }
            self.assets.remove_custom_shape(&shape_name);
            self.settings.custom_shapes.retain(|(n, _, _)| n != &shape_name);
            self.settings.save();
        }

        // Process pending selection modification from context bar
        if let Some(op) = self.tools_panel.pending_sel_modify.take()
            && let Some(project) = self.active_project_mut()
            && project.canvas_state.has_selection()
        {
            use crate::components::tools::SelectionModifyOp;
            match op {
                SelectionModifyOp::Feather(r) => {
                    crate::ops::adjustments::feather_selection(&mut project.canvas_state, r)
                }
                SelectionModifyOp::Expand(r) => {
                    crate::ops::adjustments::expand_selection(&mut project.canvas_state, r)
                }
                SelectionModifyOp::Contract(r) => {
                    crate::ops::adjustments::contract_selection(&mut project.canvas_state, r)
                }
            }
        }

        // --- Full-Screen Canvas (CentralPanel fills remaining space) ---
        let canvas_bg_top = self.theme.canvas_bg_top;
        let canvas_bg_bottom = self.theme.canvas_bg_bottom;

        #[allow(deprecated)]
        egui::CentralPanel::default()
            .frame(egui::Frame {
                fill: canvas_bg_bottom,
                ..Default::default()
            })
            .show(ctx, |ui| {
                // Draw subtle gradient background over the solid fill
                let rect = ui.max_rect();
                let painter = ui.painter();

                // Vertical gradient from top to bottom
                let mesh = {
                    let mut mesh = egui::Mesh::default();
                    mesh.colored_vertex(rect.left_top(), canvas_bg_top);
                    mesh.colored_vertex(rect.right_top(), canvas_bg_top);
                    mesh.colored_vertex(rect.left_bottom(), canvas_bg_bottom);
                    mesh.colored_vertex(rect.right_bottom(), canvas_bg_bottom);
                    mesh.add_triangle(0, 1, 2);
                    mesh.add_triangle(1, 2, 3);
                    mesh
                };
                painter.add(egui::Shape::mesh(mesh));

                if let Some(project) = self.projects.get_mut(self.active_project_index) {
                    let primary_color_f32 = self.colors_panel.get_primary_color_f32();
                    let secondary_color_f32 = self.colors_panel.get_secondary_color_f32();
                    // Push theme accent colours into canvas for selection rendering.
                    self.canvas.selection_stroke = self.theme.accent;
                    self.canvas.selection_fill = {
                        let [r, g, b, _] = self.theme.accent.to_array();
                        egui::Color32::from_rgba_unmultiplied(r, g, b, 25)
                    };
                    self.canvas.selection_contrast = match self.theme.mode {
                        crate::theme::ThemeMode::Dark => egui::Color32::BLACK,
                        crate::theme::ThemeMode::Light => egui::Color32::WHITE,
                    };
                    // Set tool icon cursor texture for the canvas overlay.
                    {
                        use crate::assets::Icon;
                        use crate::components::tools::Tool;
                        let icon_for_cursor: Option<Icon> = match self.tools_panel.active_tool {
                            Tool::Pencil => Some(Icon::Pencil),
                            Tool::Fill => Some(Icon::Fill),
                            Tool::ColorPicker => Some(Icon::ColorPicker),
                            Tool::Zoom => Some(Icon::Zoom),
                            Tool::Pan => Some(Icon::Pan),
                            _ => None,
                        };
                        self.canvas.tool_cursor_icon =
                            icon_for_cursor.and_then(|ic| self.assets.get_texture(ic).cloned());
                    }
                    ctx.data_mut(|d| {
                        d.insert_persisted(
                            egui::Id::new("paintfe_selection_stripe_color"),
                            self.settings.selection_stripe_color,
                        );
                        d.insert_persisted(
                            egui::Id::new("paintfe_selection_stripe_alpha"),
                            self.settings.selection_stripe_alpha,
                        );
                    });
                    self.canvas.show_with_state(
                        ui,
                        &mut project.canvas_state,
                        Some(&mut self.tools_panel),
                        primary_color_f32,
                        secondary_color_f32,
                        canvas_bg_bottom,
                        self.paste_overlay.as_mut(),
                        modal_open,
                        &self.settings,
                        self.pending_filter_jobs,
                        self.pending_io_ops,
                        self.theme.accent,
                        self.filter_ops_start_time,
                        self.io_ops_start_time,
                        &self.filter_status_description,
                        live_window_resize,
                    );
                    if let Some(overlay) = self.paste_overlay.as_mut()
                        && let Some((before, _after)) = overlay.pending_transform_checkpoint.take()
                    {
                        self.paste_transform_undo.push(before);
                        self.paste_transform_redo.clear();
                        project.history.push(Box::new(
                            crate::components::history::MarkerCommand::new("Move Paste"),
                        ));
                    }

                    // Handle paste overlay context menu results.
                    if let Some(action) = self.canvas.paste_context_action.take() {
                        match action {
                            crate::canvas::PasteAction::Commit
                            | crate::canvas::PasteAction::CommitAndSelect => {
                                // Commit — always a fresh snapshot (extraction is already in history).
                                let select_mask = self.paste_overlay.as_ref().map(|overlay| {
                                    overlay.solid_bounds_selection_mask(
                                        project.canvas_state.width,
                                        project.canvas_state.height,
                                    )
                                });
                                let select_bounds = self.paste_overlay.as_ref().and_then(|overlay| {
                                    overlay.transformed_bounds(
                                        project.canvas_state.width,
                                        project.canvas_state.height,
                                    )
                                });
                                if let Some(overlay) = self.paste_overlay.take() {
                                    self.paste_transform_undo.clear();
                                    self.paste_transform_redo.clear();
                                    let desc = if self.is_move_pixels_active {
                                        "Move Pixels"
                                    } else {
                                        "Paste"
                                    };
                                    let mut cmd =
                                        SnapshotCommand::new(desc.to_string(), &project.canvas_state);
                                    project.canvas_state.clear_preview_state();
                                    overlay.commit(&mut project.canvas_state);
                                    cmd.set_after(&project.canvas_state);
                                    project.history.push(Box::new(cmd));
                                    project.mark_dirty();
                                }
                                self.is_move_pixels_active = false;

                                if action == crate::canvas::PasteAction::CommitAndSelect {
                                    project.canvas_state.selection_mask = select_mask;
                                    if let Some(mask) = project.canvas_state.selection_mask.as_mut()
                                        && let Some((x0, y0, x1, y1)) = select_bounds
                                    {
                                        for y in y0..y1 {
                                            for x in x0..x1 {
                                                mask.put_pixel(x, y, image::Luma([255u8]));
                                            }
                                        }
                                    }
                                    project.canvas_state.invalidate_selection_overlay();
                                    project.canvas_state.mark_dirty(None);
                                    self.pending_selection_reassert =
                                        project.canvas_state.selection_mask.clone();
                                    self.tools_panel.selection_state.mode =
                                        crate::canvas::SelectionMode::Replace;
                                }
                            }
                            crate::canvas::PasteAction::Cancel => {
                                // Cancel.
                                self.paste_overlay = None;
                                self.paste_transform_undo.clear();
                                self.paste_transform_redo.clear();
                                if self.is_move_pixels_active {
                                    // MovePixels: undo the extraction entry we already pushed
                                    project.history.undo(&mut project.canvas_state);
                                    self.is_move_pixels_active = false;
                                }
                                project.canvas_state.clear_preview_state();
                                project.canvas_state.mark_dirty(None);
                            }
                        }
                    }
                }
            });

        // --- Floating Panels ---
        // Detect screen size changes ONCE before any panel renders,
        // so all panels see the same change flag.
        let screen_rect = ctx
            .input(|i| i.viewport().inner_rect)
            .unwrap_or_else(|| ctx.content_rect());
        let screen_w = screen_rect.max.x;
        let screen_h = screen_rect.max.y;
        let initial_layout_pass = self.last_screen_size.0 <= 0.0 || self.last_screen_size.1 <= 0.0;
        let screen_size_changed = initial_layout_pass
            || ((screen_w - self.last_screen_size.0).abs() > 0.5
                || (screen_h - self.last_screen_size.1).abs() > 0.5);

        self.is_pointer_over_layers_panel = false;
        self.show_floating_tools_panel(ctx, screen_size_changed);
        self.show_floating_layers_panel(ctx, screen_size_changed);
        self.show_floating_history_panel(ctx, screen_size_changed);
        self.show_floating_colors_panel(ctx, screen_size_changed);
        self.show_floating_palette_panel(ctx, screen_size_changed);
        self.show_floating_script_editor(ctx, screen_size_changed);
        if self.palette_reposition_settle_frames > 0 {
            self.palette_reposition_settle_frames -= 1;
        }

        // Persist last non-trivial window content size for next launch.
        let is_maximized = ctx.input(|i| i.viewport().maximized).unwrap_or(false);
        self.settings.persist_window_maximized = is_maximized;
        if !is_maximized && !live_window_resize {
            if screen_w >= 640.0 && screen_h >= 480.0 {
                self.settings.persist_window_width = screen_w;
                self.settings.persist_window_height = screen_h;
            }
            if let Some(outer_rect) = ctx.input(|i| i.viewport().outer_rect) {
                self.settings.persist_window_pos = Some((outer_rect.min.x, outer_rect.min.y));
            }
        }
        let current_time = ctx.input(|i| i.time);
        self.persist_window_state_if_changed(
            current_time,
            was_live_window_resize && !live_window_resize,
        );

        // --- Tool Hint (bottom-left status text) ---
        // Subtle text showing what the current tool does, visible at the bottom-left.
        {
            let hint = &self.tools_panel.tool_hint;
            if !hint.is_empty() {
                let screen_rect = ctx.content_rect();
                let painter = ctx.layer_painter(egui::LayerId::new(
                    egui::Order::Foreground,
                    egui::Id::new("tool_hint_overlay"),
                ));
                let text_color = match self.theme.mode {
                    crate::theme::ThemeMode::Dark => egui::Color32::from_white_alpha(60),
                    crate::theme::ThemeMode::Light => egui::Color32::from_black_alpha(80),
                };
                let font = egui::FontId::proportional(11.0);
                let pos = egui::pos2(10.0, screen_rect.max.y - 22.0);
                painter.text(pos, egui::Align2::LEFT_CENTER, hint, font, text_color);
            }
        }

        // Update last_screen_size AFTER all panels have used the flag
        self.last_screen_size = (screen_w, screen_h);

        self.commit_pending_tool_history();

        // --- Auto-rasterize text layers when destructive tools attempt to paint on them ---
        if let Some(layer_idx) = self.tools_panel.pending_auto_rasterize.take() {
            let active_idx = self.active_project_index;
            if active_idx < self.projects.len()
                && layer_idx < self.projects[active_idx].canvas_state.layers.len()
                && self.projects[active_idx].canvas_state.layers[layer_idx].is_text_layer()
            {
                {
                    let project = &mut self.projects[active_idx];
                    // Snapshot before rasterization for undo
                    let mut cmd =
                        crate::components::history::SingleLayerSnapshotCommand::new_for_layer(
                            "Rasterize Text Layer".to_string(),
                            &project.canvas_state,
                            layer_idx,
                        );
                    // Rasterize in place — convert Text→Raster, pixels are already up-to-date
                    project.canvas_state.layers[layer_idx].content =
                        crate::canvas::LayerContent::Raster;
                    // Clear canvas-level text editing marker for this layer
                    if project.canvas_state.text_editing_layer == Some(layer_idx) {
                        project.canvas_state.text_editing_layer = None;
                        project.canvas_state.clear_preview_state();
                    }
                    // Capture after state
                    cmd.set_after(&project.canvas_state);
                    project.history.push(Box::new(cmd));
                    project.mark_dirty();
                } // `project` borrow ends here — allows split-borrow below
                // Cancel any stale text editing session (different field from projects)
                self.tools_panel
                    .cancel_text_editing(&mut self.projects[active_idx].canvas_state);
            }
        }

        // --- Async Color Removal ---
        // Check if a color removal was requested and dispatch via spawn_filter_job
        if let Some(req) = self.tools_panel.take_pending_color_removal()
            && let Some(project) = self.projects.get(self.active_project_index)
        {
            let idx = req.layer_idx;
            if idx < project.canvas_state.layers.len() {
                let original_pixels = project.canvas_state.layers[idx].pixels.clone();
                let original_flat = original_pixels.to_rgba_image();
                let current_time = ctx.input(|i| i.time);
                self.spawn_filter_job(
                    current_time,
                    "Color Remover".to_string(),
                    idx,
                    original_pixels,
                    original_flat,
                    move |img| {
                        let changes = crate::ops::color_removal::compute_color_removal(
                            img,
                            req.click_x,
                            req.click_y,
                            req.tolerance,
                            req.smoothness,
                            req.contiguous,
                            req.selection_mask.as_ref(),
                        );
                        let mut result = img.clone();
                        crate::ops::color_removal::apply_color_removal(&mut result, &changes);
                        result
                    },
                );
            }
        }

        // --- Async Content-Aware Inpaint (Balanced / High Quality) ---
        if let Some(req) = self.tools_panel.take_pending_inpaint()
            && let Some(project) = self.projects.get(self.active_project_index)
        {
            let idx = req.layer_idx;
            if idx < project.canvas_state.layers.len() {
                let original_pixels = project.canvas_state.layers[idx].pixels.clone();
                let original_flat = req.original_flat;
                let hole_mask = req.hole_mask;
                let patch_size = req.patch_size;
                let iterations = req.iterations;
                let current_time = ctx.input(|i| i.time);
                self.spawn_filter_job(
                    current_time,
                    "Content-Aware Brush".to_string(),
                    idx,
                    original_pixels,
                    original_flat,
                    move |img| {
                        crate::ops::inpaint::fill_region_patchmatch(
                            img, &hole_mask, patch_size, iterations,
                        )
                    },
                );
            }
        }

        // --- Continuous Repaint While Painting ---
        // Request repaint during active brush/eraser strokes for smooth 60fps.
        // This ensures we don't miss mouse input events and get jittery results.
    }
}
