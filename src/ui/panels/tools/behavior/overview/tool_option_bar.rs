impl ToolsPanel {
    pub fn show_context_bar(
        &mut self,
        ui: &mut egui::Ui,
        assets: &Assets,
        primary_color: Color32,
        secondary_color: Color32,
        theme: &crate::theme::Theme,
    ) {
        ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
            // Tool name tag badge (Signal Grid style)
            crate::signal_widgets::tool_shelf_tag(
                ui,
                &self.active_tool_name().to_uppercase(),
                ui.visuals().widgets.active.bg_stroke.color,
                theme,
            );
            ui.add_space(6.0);

            // Rebuild tip mask cache for brush/eraser tools
            // Called both before AND after tool options UI so picker changes
            // take effect in the same frame (picker runs inside show_*_options).
            match self.active_tool {
                Tool::Brush | Tool::Eraser => {
                    self.rebuild_tip_mask(assets);
                }
                _ => {}
            }

            match self.active_tool {
                Tool::Brush | Tool::Pencil => {
                    self.show_brush_options(ui, assets);
                }
                Tool::Line => {
                    self.show_line_options(ui, assets);
                }
                Tool::Eraser => {
                    self.show_eraser_options(ui, assets);
                }
                _ => {}
            }
            // Re-run rebuild_tip_mask after options UI so that picker changes
            // (tip/size/hardness) take effect this frame, not next frame.
            match self.active_tool {
                Tool::Brush | Tool::Eraser => {
                    self.rebuild_tip_mask(assets);
                }
                _ => {}
            }
            match self.active_tool {
                // Already handled above
                Tool::Brush | Tool::Pencil | Tool::Line | Tool::Eraser => {}
                Tool::RectangleSelect | Tool::EllipseSelect => {
                    self.show_selection_options(ui);
                }
                Tool::MovePixels => {
                    self.show_move_options(ui);
                }
                Tool::MoveSelection => {
                    self.show_move_options(ui);
                }
                Tool::MagicWand => {
                    self.show_magic_wand_options(ui);
                }
                Tool::Fill => {
                    self.show_fill_options(ui);
                }
                Tool::ColorPicker => {
                    // Hint only ÔÇö no options
                }
                Tool::Lasso => {
                    self.show_lasso_options(ui);
                }
                Tool::Zoom => {
                    // Toggle button for zoom direction (touch-friendly)
                    let label = if self.zoom_tool_state.zoom_out_mode {
                        "\u{1F50D}\u{2796} Zoom Out"
                    } else {
                        "\u{1F50D}\u{2795} Zoom In"
                    };
                    if ui
                        .selectable_label(self.zoom_tool_state.zoom_out_mode, label)
                        .clicked()
                    {
                        self.zoom_tool_state.zoom_out_mode = !self.zoom_tool_state.zoom_out_mode;
                    }
                    // Removed inline label ÔÇö zoom hint is in tool_hint
                }
                Tool::Pan => {
                    // Hint only ÔÇö no options
                }
                Tool::PerspectiveCrop => {
                    self.show_perspective_crop_options(ui);
                }
                Tool::Gradient => {
                    self.show_gradient_options(ui, assets, primary_color, secondary_color);
                }
                Tool::Liquify => {
                    self.show_liquify_options(ui);
                }
                Tool::MeshWarp => {
                    self.show_mesh_warp_options(ui);
                }
                Tool::ColorRemover => {
                    self.show_color_remover_options(ui);
                }
                Tool::Smudge => {
                    self.show_smudge_options(ui);
                }
                Tool::Text => {
                    self.show_text_options(ui, assets);
                }
                Tool::Shapes => {
                    self.show_shapes_options(ui, assets);
                }
                Tool::CloneStamp => {
                    self.show_clone_stamp_options(ui);
                }
                Tool::ContentAwareBrush => {
                    self.show_content_aware_options(ui);
                }
            }
        });
    }

    fn show_move_options(&mut self, ui: &mut egui::Ui) {
        ui.label("Filter:");
        let current = self.move_interpolation;
        egui::ComboBox::from_id_salt("move_tool_filter")
            .selected_text(current.label())
            .width(110.0)
            .show_ui(ui, |ui| {
                for interp in crate::ops::transform::Interpolation::all() {
                    ui.selectable_value(&mut self.move_interpolation, *interp, interp.label());
                }
            });
        ui.checkbox(&mut self.move_anti_aliasing, "Anti-aliasing");
    }

    /// Show brush tip picker dropdown (grid popup with categories, matching shapes tool pattern)
    fn show_brush_tip_picker(&mut self, ui: &mut egui::Ui, assets: &Assets) {
        ui.label("Tip:");

        let popup_id = ui.make_persistent_id("brush_tip_grid_popup");
        let display_name = self.properties.brush_tip.display_name().to_string();

        // Button showing current tip icon + name
        let btn_response = {
            if let BrushTip::Image(ref name) = self.properties.brush_tip {
                if let Some(tex) = assets.get_brush_tip_texture(name) {
                    let sized = egui::load::SizedTexture::from_handle(tex);
                    let img =
                        egui::Image::from_texture(sized).fit_to_exact_size(egui::Vec2::splat(16.0));
                    let btn = egui::Button::image_and_text(img, &display_name);
                    ui.add(btn)
                } else {
                    ui.button(&display_name)
                }
            } else {
                // Circle ÔÇö draw a small filled circle icon on the button
                let btn = ui.button(format!("      {}", display_name));
                let rect = btn.rect;
                let circle_x = rect.left() + 14.0;
                let circle_y = rect.center().y;
                ui.painter().circle_filled(
                    egui::Pos2::new(circle_x, circle_y),
                    5.0,
                    ui.visuals().text_color(),
                );
                btn
            }
        };
        if btn_response.clicked() {
            egui::Popup::toggle_id(ui.ctx(), popup_id);
        }

        egui::Popup::new(
            popup_id,
            ui.ctx().clone(),
            egui::PopupAnchor::from(&btn_response),
            ui.layer_id(),
        )
        .open_memory(None::<egui::SetOpenCommand>)
        .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
        .show(|ui| {
            ui.set_min_width(240.0);
            ui.set_max_width(260.0);

            let cols = 5;
            let icon_size = egui::Vec2::splat(36.0);
            let accent = ui.visuals().hyperlink_color;

            // "New..." button in top-right corner
            ui.horizontal(|ui| {
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("New...").clicked() {
                        self.pending_open_add_brush_tip = true;
                        ui.close();
                    }
                });
            });

            // "Basic" category header always first, with Circle as first item
            ui.label(egui::RichText::new("Basic").strong().size(11.0));
            egui::Grid::new("brush_tip_basic_grid")
                .spacing(egui::Vec2::splat(2.0))
                .show(ui, |ui| {
                    // Circle (built-in, always first)
                    let selected = self.properties.brush_tip.is_circle();
                    let (rect, response) = ui.allocate_exact_size(icon_size, egui::Sense::click());
                    if selected {
                        ui.painter().rect_filled(
                            rect,
                            4.0,
                            egui::Color32::from_rgba_premultiplied(
                                accent.r(),
                                accent.g(),
                                accent.b(),
                                60,
                            ),
                        );
                    }
                    if response.hovered() {
                        ui.painter()
                            .rect_filled(rect, 4.0, ui.visuals().widgets.hovered.bg_fill);
                    }
                    // Draw a circle icon
                    let center = rect.center();
                    let r = icon_size.x * 0.3;
                    let stroke_color = ui.visuals().text_color();
                    ui.painter().circle_filled(center, r, stroke_color);
                    if response.clicked() {
                        self.properties.brush_tip = BrushTip::Circle;
                    }
                    response.on_hover_text("Circle");

                    // Other tips in the "Basic" category
                    let mut col = 1;
                    let categories = assets.brush_tip_categories();
                    if let Some(basic_cat) = categories.iter().find(|c| c.name == "Basic") {
                        for tip_name in &basic_cat.tips {
                            let is_selected =
                                self.properties.brush_tip == BrushTip::Image(tip_name.clone());
                            let (rect, response) =
                                ui.allocate_exact_size(icon_size, egui::Sense::click());
                            if is_selected {
                                ui.painter().rect_filled(
                                    rect,
                                    4.0,
                                    egui::Color32::from_rgba_premultiplied(
                                        accent.r(),
                                        accent.g(),
                                        accent.b(),
                                        60,
                                    ),
                                );
                            }
                            if response.hovered() {
                                ui.painter().rect_filled(
                                    rect,
                                    4.0,
                                    ui.visuals().widgets.hovered.bg_fill,
                                );
                            }
                            if let Some(tex) = assets.get_brush_tip_texture(tip_name) {
                                let sized = egui::load::SizedTexture::from_handle(tex);
                                let img = egui::Image::from_texture(sized)
                                    .fit_to_exact_size(icon_size * 0.8);
                                let inner_rect = rect.shrink(icon_size.x * 0.1);
                                img.paint_at(ui, inner_rect);
                            } else {
                                ui.painter().text(
                                    rect.center(),
                                    egui::Align2::CENTER_CENTER,
                                    &tip_name[..2.min(tip_name.len())],
                                    egui::FontId::proportional(11.0),
                                    ui.visuals().text_color(),
                                );
                            }
                            if response.clicked() {
                                self.properties.brush_tip = BrushTip::Image(tip_name.clone());
                            }
                            response.on_hover_text(tip_name);
                            col += 1;
                            if col % cols == 0 {
                                ui.end_row();
                            }
                        }
                    }
                });

            // Remaining categories
            let categories = assets.brush_tip_categories();
            for cat in categories.iter().filter(|c| c.name != "Basic") {
                ui.add_space(4.0);
                ui.label(egui::RichText::new(&cat.name).strong().size(11.0));
                egui::Grid::new(format!("brush_tip_{}_grid", cat.name))
                    .spacing(egui::Vec2::splat(2.0))
                    .show(ui, |ui| {
                        for (i, tip_name) in cat.tips.iter().enumerate() {
                            let is_selected =
                                self.properties.brush_tip == BrushTip::Image(tip_name.clone());
                            let (rect, response) =
                                ui.allocate_exact_size(icon_size, egui::Sense::click());
                            if is_selected {
                                ui.painter().rect_filled(
                                    rect,
                                    4.0,
                                    egui::Color32::from_rgba_premultiplied(
                                        accent.r(),
                                        accent.g(),
                                        accent.b(),
                                        60,
                                    ),
                                );
                            }
                            if response.hovered() {
                                ui.painter().rect_filled(
                                    rect,
                                    4.0,
                                    ui.visuals().widgets.hovered.bg_fill,
                                );
                            }
                            if let Some(tex) = assets.get_brush_tip_texture(tip_name) {
                                let sized = egui::load::SizedTexture::from_handle(tex);
                                let img = egui::Image::from_texture(sized)
                                    .fit_to_exact_size(icon_size * 0.8);
                                let inner_rect = rect.shrink(icon_size.x * 0.1);
                                img.paint_at(ui, inner_rect);
                            } else {
                                ui.painter().text(
                                    rect.center(),
                                    egui::Align2::CENTER_CENTER,
                                    &tip_name[..2.min(tip_name.len())],
                                    egui::FontId::proportional(11.0),
                                    ui.visuals().text_color(),
                                );
                            }
                            if response.clicked() {
                                self.properties.brush_tip = BrushTip::Image(tip_name.clone());
                            }
                            // Right-click opens context menu (only non-default tips)
                            let is_default_cat = cat.name == "Basic"
                                || cat.name == "Texture"
                                || cat.name == "Vegetation";
                            if response.secondary_clicked()
                                && !is_default_cat
                                && let Some(pos) = ui.input(|i| i.pointer.latest_pos())
                            {
                                self.brush_tip_context_menu =
                                    Some((tip_name.clone(), pos.x, pos.y));
                            }
                            response.on_hover_text(tip_name);
                            if (i + 1) % cols == 0 {
                                ui.end_row();
                            }
                        }
                    });
            }

            // Show right-click context menu if active
            self.cursor_blocking_rect = None;
            if let Some((ref ctx_tip, cx, cy)) = self.brush_tip_context_menu.clone() {
                let ctx_id = ui.make_persistent_id("brush_tip_ctx_menu");
                let ctx_rect = egui::Rect::from_min_size(
                    egui::pos2(cx, cy),
                    egui::vec2(120.0, 0.0),
                );
                let menu = egui::Area::new(ctx_id)
                    .fixed_pos(ctx_rect.min)
                    .order(egui::Order::Foreground)
                    .show(ui.ctx(), |ui| {
                        let frame = egui::Frame::popup(ui.style());
                        frame.show(ui, |ui| {
                            ui.set_min_width(120.0);
                            if ui.button("Delete").clicked() {
                                self.pending_delete_brush_tip = Some(ctx_tip.clone());
                                self.brush_tip_context_menu = None;
                                ui.close();
                            }
                        });
                    });
                let menu_rect = menu.response.rect;
                self.cursor_blocking_rect = Some(menu_rect);
                if ui
                    .ctx()
                    .input(|i| i.pointer.hover_pos().is_some_and(|pos| menu_rect.contains(pos)))
                {
                    ui.ctx().set_cursor_icon(egui::CursorIcon::Default);
                }
                // Close context menu on any click outside
                if ui.input(|i| i.pointer.any_click())
                    && let Some(pointer_pos) = ui.input(|i| i.pointer.latest_pos())
                    && !menu_rect.contains(pointer_pos)
                {
                    self.brush_tip_context_menu = None;
                }
            }
        });
    }

    /// Size widget with +/- buttons, drag value, and preset dropdown.
    /// The DragValue and dropdown arrow are merged into a single bordered control.
    fn show_size_widget(&mut self, ui: &mut egui::Ui, combo_id: &str, assets: &Assets) {
        ui.label(t!("ctx.size"));
        let popup_id = ui.make_persistent_id(combo_id);

        if ui.small_button("\u{2212}").clicked() {
            self.properties.size = (self.properties.size - 1.0).max(1.0);
        }

        // Merged DragValue + dropdown arrow in one frame
        let inactive = ui.visuals().widgets.inactive;
        let frame_resp = egui::Frame::NONE
            .fill(inactive.bg_fill)
            .stroke(inactive.bg_stroke)
            .corner_radius(inactive.corner_radius)
            .inner_margin(egui::Margin::same(0))
            .show(ui, |ui| {
                ui.spacing_mut().item_spacing.x = 0.0;
                // Make inner widgets frameless ÔÇö the outer Frame provides the border
                let vis = ui.visuals_mut();
                vis.widgets.inactive.bg_fill = Color32::TRANSPARENT;
                vis.widgets.inactive.bg_stroke = egui::Stroke::NONE;
                vis.widgets.hovered.bg_fill = Color32::TRANSPARENT;
                vis.widgets.hovered.bg_stroke = egui::Stroke::NONE;
                vis.widgets.active.bg_fill = Color32::TRANSPARENT;
                vis.widgets.active.bg_stroke = egui::Stroke::NONE;

                let dv_resp = ui.add(
                    egui::DragValue::new(&mut self.properties.size)
                        .speed(0.5)
                        .range(1.0..=256.0)
                        .suffix("px"),
                );
                let dv_rect = dv_resp.rect;
                let dv_height = dv_rect.height();
                dv_resp.on_hover_text(t!("ctx.size_drag_tooltip"));

                // Thin internal divider
                let sep_x = ui.cursor().left();
                ui.painter().vline(
                    sep_x,
                    dv_rect.top() + 3.0..=dv_rect.bottom() - 3.0,
                    egui::Stroke::new(1.0, inactive.bg_stroke.color.linear_multiply(0.4)),
                );

                if let Some(tex) = assets.get_texture(Icon::DropDown) {
                    let sized = egui::load::SizedTexture::from_handle(tex);
                    let img =
                        egui::Image::from_texture(sized).fit_to_exact_size(egui::vec2(12.0, 12.0));
                    ui.add(egui::Button::image(img).min_size(egui::vec2(14.0, dv_height)))
                } else {
                    ui.add(
                        egui::Button::new(egui::RichText::new("\u{25BE}").size(9.0))
                            .min_size(egui::vec2(14.0, dv_height)),
                    )
                }
            });

        let arrow_resp = frame_resp.inner;
        if arrow_resp.clicked() {
            egui::Popup::toggle_id(ui.ctx(), popup_id);
        }
        // Anchor popup below the whole merged control, not just the arrow
        egui::Popup::new(
            popup_id,
            ui.ctx().clone(),
            egui::PopupAnchor::from(&frame_resp.response),
            ui.layer_id(),
        )
        .open_memory(None::<egui::SetOpenCommand>)
        .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
        .show(|ui| {
            ui.set_min_width(80.0);
            for &preset in BRUSH_SIZE_PRESETS.iter() {
                let label = format!("{:.0} px", preset);
                if ui
                    .selectable_label((self.properties.size - preset).abs() < 0.1, &label)
                    .clicked()
                {
                    self.properties.size = preset;
                    egui::Popup::close_id(ui.ctx(), popup_id);
                }
            }
        });
        if ui.small_button("+").clicked() {
            self.properties.size = (self.properties.size + 1.0).min(256.0);
        }
    }

    /// Show text font-size widget: merged DragValue + dropdown (same as brush size).
    fn show_text_size_widget(&mut self, ui: &mut egui::Ui, combo_id: &str, assets: &Assets) {
        ui.label(t!("ctx.size"));
        let popup_id = ui.make_persistent_id(combo_id);

        if ui.small_button("\u{2212}").clicked() {
            self.text_state.font_size = (self.text_state.font_size - 1.0).max(6.0);
            self.text_state.preview_dirty = true;
            self.text_state.glyph_cache.clear();
            self.text_state.pending_ctx_style_update =
                Some(TextStyleUpdate::FontSize(self.text_state.font_size));
            self.text_state.ctx_bar_style_dirty = true;
        }

        let inactive = ui.visuals().widgets.inactive;
        let frame_resp = egui::Frame::NONE
            .fill(inactive.bg_fill)
            .stroke(inactive.bg_stroke)
            .corner_radius(inactive.corner_radius)
            .inner_margin(egui::Margin::same(0))
            .show(ui, |ui| {
                ui.spacing_mut().item_spacing.x = 0.0;
                let vis = ui.visuals_mut();
                vis.widgets.inactive.bg_fill = Color32::TRANSPARENT;
                vis.widgets.inactive.bg_stroke = egui::Stroke::NONE;
                vis.widgets.hovered.bg_fill = Color32::TRANSPARENT;
                vis.widgets.hovered.bg_stroke = egui::Stroke::NONE;
                vis.widgets.active.bg_fill = Color32::TRANSPARENT;
                vis.widgets.active.bg_stroke = egui::Stroke::NONE;

                let dv_resp = ui.add(
                    egui::DragValue::new(&mut self.text_state.font_size)
                        .speed(0.5)
                        .range(6.0..=f32::MAX)
                        .suffix("px"),
                );
                if dv_resp.changed() {
                    self.text_state.preview_dirty = true;
                    self.text_state.glyph_cache.clear();
                    self.text_state.pending_ctx_style_update =
                        Some(TextStyleUpdate::FontSize(self.text_state.font_size));
                    self.text_state.ctx_bar_style_dirty = true;
                }
                let dv_rect = dv_resp.rect;
                let dv_height = dv_rect.height();
                dv_resp.on_hover_text(t!("ctx.size_drag_tooltip"));

                let sep_x = ui.cursor().left();
                ui.painter().vline(
                    sep_x,
                    dv_rect.top() + 3.0..=dv_rect.bottom() - 3.0,
                    egui::Stroke::new(1.0, inactive.bg_stroke.color.linear_multiply(0.4)),
                );

                if let Some(tex) = assets.get_texture(Icon::DropDown) {
                    let sized = egui::load::SizedTexture::from_handle(tex);
                    let img =
                        egui::Image::from_texture(sized).fit_to_exact_size(egui::vec2(12.0, 12.0));
                    ui.add(egui::Button::image(img).min_size(egui::vec2(14.0, dv_height)))
                } else {
                    ui.add(
                        egui::Button::new(egui::RichText::new("\u{25BE}").size(9.0))
                            .min_size(egui::vec2(14.0, dv_height)),
                    )
                }
            });

        let arrow_resp = frame_resp.inner;
        if arrow_resp.clicked() {
            egui::Popup::toggle_id(ui.ctx(), popup_id);
        }
        egui::Popup::new(
            popup_id,
            ui.ctx().clone(),
            egui::PopupAnchor::from(&frame_resp.response),
            ui.layer_id(),
        )
        .open_memory(None::<egui::SetOpenCommand>)
        .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
        .show(|ui| {
            ui.set_min_width(80.0);
            for &preset in TEXT_SIZE_PRESETS.iter() {
                let label = format!("{:.0} px", preset);
                if ui
                    .selectable_label((self.text_state.font_size - preset).abs() < 0.1, &label)
                    .clicked()
                {
                    self.text_state.font_size = preset;
                    self.text_state.preview_dirty = true;
                    self.text_state.glyph_cache.clear();
                    self.text_state.pending_ctx_style_update =
                        Some(TextStyleUpdate::FontSize(self.text_state.font_size));
                    self.text_state.ctx_bar_style_dirty = true;
                    egui::Popup::close_id(ui.ctx(), popup_id);
                }
            }
        });
        if ui.small_button("+").clicked() {
            self.text_state.font_size += 1.0;
            self.text_state.preview_dirty = true;
            self.text_state.glyph_cache.clear();
            self.text_state.pending_ctx_style_update =
                Some(TextStyleUpdate::FontSize(self.text_state.font_size));
            self.text_state.ctx_bar_style_dirty = true;
        }
    }

    /// Show brush-specific options (size, hardness, blend mode)
    /// For Pencil tool, skip size and hardness since it always paints single pixels
    fn show_brush_options(&mut self, ui: &mut egui::Ui, assets: &Assets) {
        // Brush tip picker (skip for Pencil ÔÇö always pixel)
        if self.active_tool != Tool::Pencil {
            self.show_brush_tip_picker(ui, assets);
            ui.separator();
        }

        // Size - skip for Pencil tool since it always paints single pixels
        if self.active_tool != Tool::Pencil {
            self.show_size_widget(ui, "ctx_brush_size", assets);

            ui.separator();
        }

        // Hardness - skip for Pencil tool since it always paints single pixels and Line tool
        if self.active_tool != Tool::Pencil && self.active_tool != Tool::Line {
            ui.label(t!("ctx.hardness"));
            let mut hardness_pct = (self.properties.hardness * 100.0).round();
            if ui
                .add(
                    egui::DragValue::new(&mut hardness_pct)
                        .speed(1.0)
                        .range(0.0..=100.0)
                        .suffix("%"),
                )
                .on_hover_text(t!("ctx.hardness_brush_tooltip"))
                .changed()
            {
                self.properties.hardness = hardness_pct / 100.0;
            }

            ui.separator();
        }

        // Blend Mode
        ui.label(t!("ctx.blend"));
        let current_mode = self.properties.blending_mode;
        egui::ComboBox::from_id_salt("ctx_blend_mode")
            .selected_text(current_mode.name())
            .width(90.0)
            .show_ui(ui, |ui| {
                for &mode in BlendMode::all() {
                    if ui
                        .selectable_label(mode == current_mode, mode.name())
                        .clicked()
                    {
                        self.properties.blending_mode = mode;
                    }
                }
            });

        ui.separator();

        // Anti-aliasing toggle (compact "AA" matching Text tool)
        let aa_resp = ui.selectable_label(self.properties.anti_aliased, t!("ctx.anti_alias"));
        if aa_resp.clicked() {
            self.properties.anti_aliased = !self.properties.anti_aliased;
        }
        aa_resp.on_hover_text(t!("ctx.anti_alias_tooltip"));

        // Brush Mode (Normal/Dodge/Burn/Sponge) - only for Brush tool (disabled for now)
        if self.active_tool == Tool::Brush {
            ui.separator();
            ui.add_enabled_ui(false, |ui| {
                ui.label("Mode:");
                let current_bm = self.properties.brush_mode;
                egui::ComboBox::from_id_salt("ctx_brush_mode")
                    .selected_text(current_bm.label())
                    .width(70.0)
                    .show_ui(ui, |ui| {
                        for &mode in BrushMode::all() {
                            let _ = ui.selectable_label(mode == current_bm, mode.label());
                        }
                    });
            });
        }

        // Dynamics popup: Scatter, Color Jitter
        ui.separator();
        let dyn_active = self.properties.scatter > 0.01
            || self.properties.hue_jitter > 0.01
            || self.properties.brightness_jitter > 0.01
            || self.properties.pressure_size
            || self.properties.pressure_opacity;
        let dyn_popup_id = ui.make_persistent_id("brush_dyn_popup");
        let dyn_resp = assets.icon_button(ui, Icon::UiBrushDynamics, egui::Vec2::splat(20.0));
        if dyn_active {
            let dot = egui::pos2(dyn_resp.rect.max.x - 3.5, dyn_resp.rect.min.y + 3.5);
            ui.painter()
                .circle_filled(dot, 2.5, ui.visuals().hyperlink_color);
        }
        if dyn_resp.clicked() {
            egui::Popup::toggle_id(ui.ctx(), dyn_popup_id);
        }
        egui::Popup::new(
            dyn_popup_id,
            ui.ctx().clone(),
            egui::PopupAnchor::from(&dyn_resp),
            ui.layer_id(),
        )
        .open_memory(None::<egui::SetOpenCommand>)
        .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
        .show(|ui| {
            ui.set_min_width(220.0);
            egui::Grid::new("brush_dynamics_popup")
                .num_columns(2)
                .spacing([8.0, 4.0])
                .show(ui, |ui| {
                    ui.label("Scatter");
                    let mut scatter_pct = (self.properties.scatter * 200.0).round();
                    if ui
                        .add(
                            egui::Slider::new(&mut scatter_pct, 0.0..=200.0)
                                .suffix("%")
                                .max_decimals(0),
                        )
                        .changed()
                    {
                        self.properties.scatter = scatter_pct / 200.0;
                    }
                    ui.end_row();
                    ui.label("Hue Jitter");
                    let mut hj_pct = (self.properties.hue_jitter * 100.0).round();
                    if ui
                        .add(
                            egui::Slider::new(&mut hj_pct, 0.0..=100.0)
                                .suffix("%")
                                .max_decimals(0),
                        )
                        .changed()
                    {
                        self.properties.hue_jitter = hj_pct / 100.0;
                    }
                    ui.end_row();
                    ui.label("Brightness");
                    let mut bj_pct = (self.properties.brightness_jitter * 100.0).round();
                    if ui
                        .add(
                            egui::Slider::new(&mut bj_pct, 0.0..=100.0)
                                .suffix("%")
                                .max_decimals(0),
                        )
                        .changed()
                    {
                        self.properties.brightness_jitter = bj_pct / 100.0;
                    }
                    ui.end_row();

                    // Pen pressure sensitivity
                    ui.separator();
                    ui.separator();
                    ui.end_row();

                    ui.label("Pen Pressure");
                    ui.label("");
                    ui.end_row();

                    ui.checkbox(&mut self.properties.pressure_size, "Size")
                        .on_hover_text("Pen pressure controls brush size");
                    let mut min_size_pct = (self.properties.pressure_min_size * 100.0).round();
                    if ui
                        .add_enabled(
                            self.properties.pressure_size,
                            egui::Slider::new(&mut min_size_pct, 1.0..=100.0)
                                .suffix("% min")
                                .max_decimals(0),
                        )
                        .on_hover_text("Minimum brush size at zero pressure")
                        .changed()
                    {
                        self.properties.pressure_min_size = min_size_pct / 100.0;
                    }
                    ui.end_row();

                    ui.checkbox(&mut self.properties.pressure_opacity, "Opacity")
                        .on_hover_text("Pen pressure controls brush opacity");
                    let mut min_opacity_pct =
                        (self.properties.pressure_min_opacity * 100.0).round();
                    if ui
                        .add_enabled(
                            self.properties.pressure_opacity,
                            egui::Slider::new(&mut min_opacity_pct, 1.0..=100.0)
                                .suffix("% min")
                                .max_decimals(0),
                        )
                        .on_hover_text("Minimum opacity at zero pressure")
                        .changed()
                    {
                        self.properties.pressure_min_opacity = min_opacity_pct / 100.0;
                    }
                    ui.end_row();
                });
        });

        // Spacing slider (only for image tips, not circle)
        if !self.properties.brush_tip.is_circle() && self.active_tool != Tool::Pencil {
            ui.separator();
            ui.label(t!("ctx.spacing"));
            let mut spacing_pct = (self.properties.spacing * 100.0).round();
            if ui
                .add(
                    egui::DragValue::new(&mut spacing_pct)
                        .speed(1.0)
                        .range(1.0..=200.0)
                        .suffix("%"),
                )
                .on_hover_text(t!("ctx.spacing_tooltip"))
                .changed()
            {
                self.properties.spacing = spacing_pct / 100.0;
            }

            // Rotation controls
            self.show_tip_rotation_controls(ui);
        }
    }

    /// Show rotation controls for non-circle brush tips.
    /// Provides a fixed-angle slider OR a random-range double-slider, plus a checkbox to toggle.
    fn show_tip_rotation_controls(&mut self, ui: &mut egui::Ui) {
        ui.separator();

        // Always show "Angle:" label first
        ui.label(t!("ctx.angle"));

        if self.properties.tip_random_rotation {
            // --- Random rotation mode: range controls ---
            let mut lo = self.properties.tip_rotation_range.0;
            let mut hi = self.properties.tip_rotation_range.1;

            // Min handle
            if ui
                .add(
                    egui::DragValue::new(&mut lo)
                        .speed(1.0)
                        .range(0.0..=360.0)
                        .suffix(" deg"),
                )
                .changed()
                && lo > hi
            {
                hi = lo;
            }

            // Painted range bar
            let bar_width = 100.0;
            let bar_height = 14.0;
            let (bar_rect, _bar_resp) = ui
                .allocate_exact_size(egui::Vec2::new(bar_width, bar_height), egui::Sense::hover());

            // Background track
            let track_color = ui.visuals().widgets.inactive.bg_fill;
            ui.painter().rect_filled(bar_rect, 3.0, track_color);

            // Highlighted portion
            let frac_lo = lo / 360.0;
            let frac_hi = hi / 360.0;
            let fill_left = bar_rect.left() + frac_lo * bar_width;
            let fill_right = bar_rect.left() + frac_hi * bar_width;
            if fill_right > fill_left + 1.0 {
                let fill_rect = egui::Rect::from_min_max(
                    egui::Pos2::new(fill_left, bar_rect.top()),
                    egui::Pos2::new(fill_right, bar_rect.bottom()),
                );
                let accent = ui.visuals().hyperlink_color;
                let fill_color =
                    egui::Color32::from_rgba_unmultiplied(accent.r(), accent.g(), accent.b(), 140);
                ui.painter().rect_filled(fill_rect, 3.0, fill_color);
            }

            // Handle markers at the edges of the range
            let handle_r = 4.0;
            let handle_color = ui.visuals().text_color();
            ui.painter().circle_filled(
                egui::Pos2::new(fill_left, bar_rect.center().y),
                handle_r,
                handle_color,
            );
            ui.painter().circle_filled(
                egui::Pos2::new(fill_right, bar_rect.center().y),
                handle_r,
                handle_color,
            );

            // Max handle
            if ui
                .add(
                    egui::DragValue::new(&mut hi)
                        .speed(1.0)
                        .range(0.0..=360.0)
                        .suffix(" deg"),
                )
                .changed()
                && hi < lo
            {
                lo = hi;
            }

            self.properties.tip_rotation_range = (lo, hi);
            // In random mode, the cursor doesn't show a specific rotation
            self.active_tip_rotation_deg = 0.0;
        } else {
            // --- Fixed rotation mode: single angle value ---
            ui.add(
                egui::DragValue::new(&mut self.properties.tip_rotation)
                    .speed(1.0)
                    .range(0.0..=359.0)
                    .suffix(" deg"),
            )
            .on_hover_text(t!("ctx.rotation_tooltip"));
            self.active_tip_rotation_deg = self.properties.tip_rotation;
        }

        // Random checkbox ÔÇö to the right of the angle controls
        let rnd_resp = ui.selectable_label(self.properties.tip_random_rotation, t!("ctx.random"));
        if rnd_resp.clicked() {
            self.properties.tip_random_rotation = !self.properties.tip_random_rotation;
        }
        rnd_resp.on_hover_text(t!("ctx.random_tooltip"));
    }

    /// Show line tool-specific options (size, cap style, pattern, blend mode)
    fn show_line_options(&mut self, ui: &mut egui::Ui, assets: &Assets) {
        // Size
        self.show_size_widget(ui, "ctx_line_size", assets);

        ui.separator();

        // Line Pattern
        ui.label(t!("ctx.pattern"));
        let current_pattern = self.line_state.line_tool.options.pattern;
        egui::ComboBox::from_id_salt("ctx_line_pattern")
            .selected_text(current_pattern.label())
            .width(70.0)
            .show_ui(ui, |ui| {
                for &pattern in LinePattern::all() {
                    if ui
                        .selectable_label(pattern == current_pattern, pattern.label())
                        .clicked()
                    {
                        self.line_state.line_tool.options.pattern = pattern;
                    }
                }
            });

        ui.separator();

        // End Shape
        ui.label(t!("ctx.ends"));
        let current_end_shape = self.line_state.line_tool.options.end_shape;
        egui::ComboBox::from_id_salt("ctx_line_end_shape")
            .selected_text(current_end_shape.label())
            .width(60.0)
            .show_ui(ui, |ui| {
                for &shape in LineEndShape::all() {
                    if ui
                        .selectable_label(shape == current_end_shape, shape.label())
                        .clicked()
                    {
                        self.line_state.line_tool.options.end_shape = shape;
                    }
                }
            });

        // Arrow Side (only visible when Arrow is selected)
        if self.line_state.line_tool.options.end_shape == LineEndShape::Arrow {
            let current_arrow_side = self.line_state.line_tool.options.arrow_side;
            egui::ComboBox::from_id_salt("ctx_line_arrow_side")
                .selected_text(current_arrow_side.label())
                .width(55.0)
                .show_ui(ui, |ui| {
                    for &side in ArrowSide::all() {
                        if ui
                            .selectable_label(side == current_arrow_side, side.label())
                            .clicked()
                        {
                            self.line_state.line_tool.options.arrow_side = side;
                        }
                    }
                });
        }

        ui.separator();

        // Anti-aliasing toggle (matches brush/eraser style)
        let aa_resp = ui.selectable_label(
            self.line_state.line_tool.options.anti_alias,
            t!("ctx.anti_alias"),
        );
        if aa_resp.clicked() {
            self.line_state.line_tool.options.anti_alias =
                !self.line_state.line_tool.options.anti_alias;
        }
        aa_resp.on_hover_text(t!("ctx.anti_alias_tooltip"));

        ui.separator();

        // Blend Mode
        ui.label(t!("ctx.blend"));
        let current_mode = self.properties.blending_mode;
        egui::ComboBox::from_id_salt("ctx_line_blend_mode")
            .selected_text(current_mode.name())
            .width(90.0)
            .show_ui(ui, |ui| {
                for &mode in BlendMode::all() {
                    if ui
                        .selectable_label(mode == current_mode, mode.name())
                        .clicked()
                    {
                        self.properties.blending_mode = mode;
                    }
                }
            });
    }

    /// Show eraser-specific options (size, hardness - opacity from color alpha)
    fn show_eraser_options(&mut self, ui: &mut egui::Ui, assets: &Assets) {
        // Brush tip picker
        self.show_brush_tip_picker(ui, assets);
        ui.separator();

        // Size
        self.show_size_widget(ui, "ctx_eraser_size", assets);

        ui.separator();

        // Hardness
        ui.label(t!("ctx.hardness"));
        let mut hardness_pct = (self.properties.hardness * 100.0).round();
        if ui
            .add(
                egui::DragValue::new(&mut hardness_pct)
                    .speed(1.0)
                    .range(0.0..=100.0)
                    .suffix("%"),
            )
            .on_hover_text(t!("ctx.hardness_eraser_tooltip"))
            .changed()
        {
            self.properties.hardness = hardness_pct / 100.0;
        }

        ui.separator();

        // Anti-aliasing toggle (compact "AA" matching Text tool)
        let aa_resp = ui.selectable_label(self.properties.anti_aliased, t!("ctx.anti_alias"));
        if aa_resp.clicked() {
            self.properties.anti_aliased = !self.properties.anti_aliased;
        }
        aa_resp.on_hover_text(t!("ctx.anti_alias_tooltip"));

        // Dynamics popup for eraser: Scatter only
        ui.separator();
        let dyn_active_eraser = self.properties.scatter > 0.01;
        let dyn_popup_id_e = ui.make_persistent_id("eraser_dyn_popup");
        let dyn_resp_e = assets.icon_button(ui, Icon::UiBrushDynamics, egui::Vec2::splat(20.0));
        if dyn_active_eraser {
            let dot_e = egui::pos2(dyn_resp_e.rect.max.x - 3.5, dyn_resp_e.rect.min.y + 3.5);
            ui.painter()
                .circle_filled(dot_e, 2.5, ui.visuals().hyperlink_color);
        }
        if dyn_resp_e.clicked() {
            egui::Popup::toggle_id(ui.ctx(), dyn_popup_id_e);
        }
        egui::Popup::new(
            dyn_popup_id_e,
            ui.ctx().clone(),
            egui::PopupAnchor::from(&dyn_resp_e),
            ui.layer_id(),
        )
        .open_memory(None::<egui::SetOpenCommand>)
        .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
        .show(|ui| {
            ui.set_min_width(180.0);
            egui::Grid::new("eraser_dynamics_popup")
                .num_columns(2)
                .spacing([8.0, 4.0])
                .show(ui, |ui| {
                    ui.label("Scatter");
                    let mut scatter_pct = (self.properties.scatter * 200.0).round();
                    if ui
                        .add(
                            egui::Slider::new(&mut scatter_pct, 0.0..=200.0)
                                .suffix("%")
                                .max_decimals(0),
                        )
                        .changed()
                    {
                        self.properties.scatter = scatter_pct / 200.0;
                    }
                    ui.end_row();
                });
        });

        ui.separator();

        // Show current eraser opacity from color alpha
        let opacity_pct = (self.properties.color.a() as f32 / 255.0 * 100.0).round();
        ui.label(t!("ctx.opacity_label").replace("{0}", &format!("{:.0}", opacity_pct)))
            .on_hover_text(t!("ctx.eraser_opacity_tooltip"));

        // Spacing slider (only for image tips)
        if !self.properties.brush_tip.is_circle() {
            ui.separator();
            ui.label(t!("ctx.spacing"));
            let mut spacing_pct = (self.properties.spacing * 100.0).round();
            if ui
                .add(
                    egui::DragValue::new(&mut spacing_pct)
                        .speed(1.0)
                        .range(1.0..=200.0)
                        .suffix("%"),
                )
                .on_hover_text(t!("ctx.spacing_tooltip"))
                .changed()
            {
                self.properties.spacing = spacing_pct / 100.0;
            }

            // Rotation controls (shared helper)
            self.show_tip_rotation_controls(ui);
        }
    }

    /// Show selection-tool-specific options (mode dropdown)
    fn show_selection_options(&mut self, ui: &mut egui::Ui) {
        ui.label(t!("ctx.mode"));
        let current = self.selection_state.mode;
        egui::ComboBox::from_id_salt("ctx_sel_mode")
            .selected_text(current.label())
            .width(90.0)
            .show_ui(ui, |ui| {
                for &mode in SelectionMode::all() {
                    if ui.selectable_label(mode == current, mode.label()).clicked() {
                        self.selection_state.mode = mode;
                    }
                }
            });

        if self.active_tool == Tool::RectangleSelect {
            ui.separator();

            ui.label("Aspect ratio:");
            ui.add(
                egui::TextEdit::singleline(&mut self.selection_state.aspect_ratio_input)
                    .desired_width(72.0)
                    .hint_text("3:4"),
            )
            .on_hover_text(
                "Lock rectangular selections to width:height. Use image for the canvas ratio.",
            );
            if let Err(error) = parse_selection_aspect_ratio(&self.selection_state.aspect_ratio_input) {
                ui.colored_label(ui.visuals().error_fg_color, error);
            }
        }

        ui.separator();

        self.show_sel_modify_controls(ui);

        ui.separator();
        ui.label(t!("ctx.selection_hint"));
    }

    /// Show lasso-tool-specific options (mode dropdown)
    fn show_lasso_options(&mut self, ui: &mut egui::Ui) {
        ui.label(t!("ctx.mode"));
        let current = self.lasso_state.mode;
        egui::ComboBox::from_id_salt("ctx_lasso_mode")
            .selected_text(current.label())
            .width(90.0)
            .show_ui(ui, |ui| {
                for &mode in SelectionMode::all() {
                    if ui.selectable_label(mode == current, mode.label()).clicked() {
                        self.lasso_state.mode = mode;
                    }
                }
            });
        ui.separator();
        self.show_sel_modify_controls(ui);
        ui.separator();
        ui.label(t!("ctx.lasso_hint"));
    }

    /// Inline Feather / Expand / Contract controls for all selection tool context bars.
    fn show_sel_modify_controls(&mut self, ui: &mut egui::Ui) {
        ui.label("Modify:");
        ui.add(
            egui::DragValue::new(&mut self.sel_modify_radius)
                .range(1.0..=200.0)
                .speed(0.5)
                .suffix("px")
                .max_decimals(0),
        )
        .on_hover_text("Radius in pixels for Feather / Expand / Contract");
        if ui
            .button("Feather")
            .on_hover_text("Blur (soften) selection edge by radius")
            .clicked()
        {
            self.pending_sel_modify = Some(SelectionModifyOp::Feather(self.sel_modify_radius));
        }
        if ui
            .button("Expand")
            .on_hover_text("Grow selection by radius")
            .clicked()
        {
            self.pending_sel_modify = Some(SelectionModifyOp::Expand(
                self.sel_modify_radius.round() as i32,
            ));
        }
        if ui
            .button("Contract")
            .on_hover_text("Shrink selection by radius")
            .clicked()
        {
            self.pending_sel_modify = Some(SelectionModifyOp::Contract(
                self.sel_modify_radius.round() as i32,
            ));
        }
    }

    /// Show perspective crop options
    fn show_perspective_crop_options(&mut self, ui: &mut egui::Ui) {
        if self.perspective_crop_state.active {
            ui.label(t!("ctx.perspective_crop_active"));
            if ui.button(t!("ctx.apply")).clicked() {
                // Set a flag ÔÇö actual cropping happens in handle_input
                // which has canvas_state access
            }
            if ui.button(t!("ctx.cancel")).clicked() {
                self.perspective_crop_state.active = false;
            }
        } else {
            ui.label(t!("ctx.perspective_crop_inactive"));
        }
    }

    fn show_clone_stamp_options(&mut self, ui: &mut egui::Ui) {
        // Size
        ui.label(t!("ctx.size"));
        ui.add(
            egui::DragValue::new(&mut self.properties.size)
                .range(1.0..=256.0)
                .speed(0.5)
                .suffix("px"),
        );
        ui.separator();

        // Hardness
        ui.label(t!("ctx.hardness"));
        let mut hardness_pct = self.properties.hardness * 100.0;
        if ui
            .add(
                egui::DragValue::new(&mut hardness_pct)
                    .range(0.0..=100.0)
                    .speed(0.5)
                    .suffix("%"),
            )
            .changed()
        {
            self.properties.hardness = hardness_pct / 100.0;
        }
        ui.separator();

        // Source indicator
        if let Some(src) = self.clone_stamp_state.source {
            ui.label(
                t!("ctx.clone_stamp.source")
                    .replace("{0}", &format!("{:.0}", src.x))
                    .replace("{1}", &format!("{:.0}", src.y)),
            );
        } else {
            ui.label(t!("ctx.clone_stamp.set_source"));
        }
    }

    fn show_content_aware_options(&mut self, ui: &mut egui::Ui) {
        use crate::ops::inpaint::ContentAwareQuality;

        // Size
        ui.label(t!("ctx.size"));
        ui.add(
            egui::DragValue::new(&mut self.properties.size)
                .range(1.0..=256.0)
                .speed(0.5)
                .suffix("px"),
        );
        ui.separator();

        // Hardness
        ui.label(t!("ctx.hardness"));
        let mut hardness_pct = self.properties.hardness * 100.0;
        if ui
            .add(
                egui::DragValue::new(&mut hardness_pct)
                    .range(0.0..=100.0)
                    .speed(0.5)
                    .suffix("%"),
            )
            .changed()
        {
            self.properties.hardness = hardness_pct / 100.0;
        }
        ui.separator();

        // Quality dropdown
        ui.label("Quality:");
        let cur_q = self.content_aware_state.quality;
        egui::ComboBox::from_id_salt("ca_quality")
            .selected_text(cur_q.label())
            .width(100.0)
            .show_ui(ui, |ui| {
                for &q in ContentAwareQuality::all() {
                    if ui.selectable_label(q == cur_q, q.label()).clicked() {
                        self.content_aware_state.quality = q;
                    }
                }
            });
        ui.separator();

        // Sample radius (Instant only)
        if cur_q == ContentAwareQuality::Instant {
            ui.label(t!("ctx.content_aware.sample"));
            ui.add(
                egui::DragValue::new(&mut self.content_aware_state.sample_radius)
                    .range(10.0..=150.0)
                    .speed(0.5)
                    .suffix("px"),
            );
            ui.separator();
        }

        // Patch size (Balanced / HQ only)
        if cur_q.is_async() {
            ui.label("Patch:");
            ui.add(
                egui::DragValue::new(&mut self.content_aware_state.patch_size)
                    .range(3_u32..=11_u32)
                    .speed(0.5)
                    .suffix("px"),
            );
            // Keep patch_size odd
            if self.content_aware_state.patch_size.is_multiple_of(2) {
                self.content_aware_state.patch_size += 1;
            }
            ui.separator();
        }

        if cur_q == ContentAwareQuality::Instant {
            ui.label(t!("ctx.content_aware.hint"));
        } else {
            ui.label("Paint to preview, then release to run inpaint.");
        }
    }

    /// Draw a custom tolerance slider with +/- buttons, wide track, and vertical handle.
    /// Returns `Some(new_value)` if changed, `None` otherwise.
    fn tolerance_slider(ui: &mut egui::Ui, id_salt: &str, value: f32) -> Option<f32> {
        let mut new_value = value;
        let mut changed = false;

        let vis = ui.visuals().clone();
        let is_dark = vis.dark_mode;

        // Colors that work in both dark and light modes
        let track_bg = if is_dark {
            Color32::from_gray(50)
        } else {
            Color32::from_gray(190)
        };
        // Use theme accent color for the filled portion
        let track_fill = vis.selection.bg_fill;
        let handle_color = if is_dark {
            Color32::from_gray(220)
        } else {
            Color32::from_gray(255)
        };
        let handle_border = if is_dark {
            Color32::from_gray(140)
        } else {
            Color32::from_gray(80)
        };
        let btn_bg = if is_dark {
            Color32::from_gray(60)
        } else {
            Color32::from_gray(210)
        };
        let btn_hover = if is_dark {
            Color32::from_gray(80)
        } else {
            Color32::from_gray(225)
        };
        let btn_text = if is_dark {
            Color32::from_gray(220)
        } else {
            Color32::from_gray(30)
        };
        let value_text_color = if is_dark {
            Color32::from_gray(200)
        } else {
            Color32::from_gray(40)
        };

        ui.horizontal(|ui| {
            // Minus button
            let btn_size = egui::vec2(20.0, 20.0);
            let (minus_rect, minus_resp) = ui.allocate_exact_size(btn_size, egui::Sense::click());
            let minus_bg = if minus_resp.hovered() {
                btn_hover
            } else {
                btn_bg
            };
            ui.painter().rect_filled(minus_rect, 3.0, minus_bg);
            ui.painter().text(
                minus_rect.center(),
                egui::Align2::CENTER_CENTER,
                "-",
                egui::FontId::proportional(14.0),
                btn_text,
            );
            if minus_resp.clicked() {
                new_value = (new_value - 1.0).max(0.0);
                changed = true;
            }

            ui.add_space(4.0);

            // Slider track
            let slider_width = 140.0;
            let slider_height = 20.0;
            let (slider_rect, slider_resp) = ui.allocate_exact_size(
                egui::vec2(slider_width, slider_height),
                egui::Sense::click_and_drag(),
            );

            let rounding = slider_height / 2.0;

            // Draw track background
            ui.painter().rect_filled(slider_rect, rounding, track_bg);

            // Draw filled portion
            let fill_fraction = new_value / 100.0;
            let fill_width = slider_rect.width() * fill_fraction;
            if fill_width > 0.5 {
                let fill_rect =
                    Rect::from_min_size(slider_rect.min, egui::vec2(fill_width, slider_height));
                ui.painter().rect_filled(fill_rect, rounding, track_fill);
            }

            // Draw vertical handle
            let handle_x = slider_rect.min.x + fill_width;
            let handle_width = 4.0;
            let handle_height = slider_height + 2.0;
            let handle_rect = Rect::from_center_size(
                egui::pos2(handle_x, slider_rect.center().y),
                egui::vec2(handle_width, handle_height),
            );
            ui.painter().rect_filled(handle_rect, 2.0, handle_color);
            ui.painter().rect_stroke(
                handle_rect,
                2.0,
                (1.0, handle_border),
                egui::StrokeKind::Middle,
            );

            // Draw value text centered on track
            let text = format!("{}%", new_value.round() as i32);
            ui.painter().text(
                slider_rect.center(),
                egui::Align2::CENTER_CENTER,
                &text,
                egui::FontId::proportional(11.0),
                value_text_color,
            );

            // Handle interaction
            if (slider_resp.dragged() || slider_resp.clicked())
                && let Some(pos) = slider_resp.interact_pointer_pos()
            {
                let frac = ((pos.x - slider_rect.min.x) / slider_rect.width()).clamp(0.0, 1.0);
                new_value = (frac * 100.0).round();
                changed = true;
            }

            // Tooltip
            if slider_resp.hovered() {
                slider_resp.on_hover_text(t!("ctx.tolerance_tooltip"));
            }

            ui.add_space(4.0);

            // Plus button
            let (plus_rect, plus_resp) = ui.allocate_exact_size(btn_size, egui::Sense::click());
            let plus_bg = if plus_resp.hovered() {
                btn_hover
            } else {
                btn_bg
            };
            ui.painter().rect_filled(plus_rect, 3.0, plus_bg);
            ui.painter().text(
                plus_rect.center(),
                egui::Align2::CENTER_CENTER,
                "+",
                egui::FontId::proportional(14.0),
                btn_text,
            );
            if plus_resp.clicked() {
                new_value = (new_value + 1.0).min(100.0);
                changed = true;
            }
        });

        // Ensure we allocate a unique id for internal egui state
        let _ = ui.id().with(id_salt);

        if changed { Some(new_value) } else { None }
    }

    fn show_magic_wand_options(&mut self, ui: &mut egui::Ui) {
        ui.label(t!("ctx.tolerance"));
        if let Some(new_tolerance) =
            Self::tolerance_slider(ui, "mw_tol", self.magic_wand_state.tolerance)
        {
            self.magic_wand_state.tolerance = new_tolerance;
            self.magic_wand_state.tolerance_changed_at = Some(Instant::now());
            self.magic_wand_state.preview_pending = true;
            // Instant re-threshold ÔÇö no debounce needed with distance-map approach
            ui.ctx().request_repaint();
        }

        ui.separator();

        let old_aa = self.magic_wand_state.anti_aliased;
        let aa_resp = ui.selectable_label(self.magic_wand_state.anti_aliased, t!("ctx.anti_alias"));
        if aa_resp.clicked() {
            self.magic_wand_state.anti_aliased = !self.magic_wand_state.anti_aliased;
        }
        aa_resp.on_hover_text(t!("ctx.anti_alias_tooltip"));
        if self.magic_wand_state.anti_aliased != old_aa {
            self.magic_wand_state.tolerance_changed_at = Some(Instant::now());
            self.magic_wand_state.preview_pending = true;
            ui.ctx().request_repaint();
        }

        ui.separator();

        ui.label("Compare");
        let prev_distance_mode = self.magic_wand_state.distance_mode;
        egui::ComboBox::from_id_salt("ctx_magic_wand_distance_mode")
            .selected_text(self.magic_wand_state.distance_mode.label())
            .width(120.0)
            .show_ui(ui, |ui| {
                for &mode in WandDistanceMode::all() {
                    if ui
                        .selectable_label(mode == self.magic_wand_state.distance_mode, mode.label())
                        .clicked()
                    {
                        self.magic_wand_state.distance_mode = mode;
                    }
                }
            });
        if self.magic_wand_state.distance_mode != prev_distance_mode {
            self.clear_magic_wand_async_state();
            ui.ctx().request_repaint();
        }

        ui.separator();

        ui.label("Connectivity");
        let prev_connectivity = self.magic_wand_state.connectivity;
        egui::ComboBox::from_id_salt("ctx_magic_wand_connectivity")
            .selected_text(self.magic_wand_state.connectivity.label())
            .width(120.0)
            .show_ui(ui, |ui| {
                for &mode in FloodConnectivity::all() {
                    if ui
                        .selectable_label(mode == self.magic_wand_state.connectivity, mode.label())
                        .clicked()
                    {
                        self.magic_wand_state.connectivity = mode;
                    }
                }
            });
        if self.magic_wand_state.connectivity != prev_connectivity {
            self.clear_magic_wand_async_state();
            ui.ctx().request_repaint();
        }

        ui.separator();

        ui.label(t!("ctx.mode"));
        let current = self.selection_state.mode;
        egui::ComboBox::from_id_salt("ctx_magic_wand_mode")
            .selected_text(current.label())
            .width(90.0)
            .show_ui(ui, |ui| {
                for &mode in SelectionMode::all() {
                    if ui.selectable_label(mode == current, mode.label()).clicked() {
                        self.selection_state.mode = mode;
                    }
                }
            });

        ui.separator();
        self.show_sel_modify_controls(ui);
    }

    fn show_fill_options(&mut self, ui: &mut egui::Ui) {
        ui.label(t!("ctx.tolerance"));
        if let Some(new_val) = Self::tolerance_slider(ui, "fill_tol", self.fill_state.tolerance) {
            self.fill_state.tolerance = new_val;
            self.fill_state.tolerance_changed_at = Some(Instant::now());
            self.fill_state.recalc_pending = true;
            ui.ctx().request_repaint();
        }

        ui.separator();

        let old_aa = self.fill_state.anti_aliased;
        let aa_resp = ui.selectable_label(self.fill_state.anti_aliased, t!("ctx.anti_alias"));
        if aa_resp.clicked() {
            self.fill_state.anti_aliased = !self.fill_state.anti_aliased;
        }
        aa_resp.on_hover_text(t!("ctx.anti_alias_tooltip"));
        if self.fill_state.anti_aliased != old_aa {
            // Anti-alias setting changed, trigger preview refresh
            self.fill_state.tolerance_changed_at = Some(Instant::now());
            self.fill_state.recalc_pending = true;
            ui.ctx().request_repaint();
        }
    }

    pub fn show_properties_toolbar(&mut self, ui: &mut egui::Ui, assets: &Assets) {
        // Size
        self.show_size_widget(ui, "brush_size_presets", assets);

        ui.separator();

        // Hardness as percentage
        ui.label(t!("ctx.hardness"));
        let mut hardness_pct = (self.properties.hardness * 100.0).round();
        if ui
            .add(
                egui::DragValue::new(&mut hardness_pct)
                    .speed(1.0)
                    .range(0.0..=100.0)
                    .suffix("%"),
            )
            .on_hover_text(t!("ctx.hardness_brush_tooltip"))
            .changed()
        {
            self.properties.hardness = hardness_pct / 100.0;
        }

        ui.separator();

        // Blend Mode Dropdown (for brush painting)
        ui.label(t!("ctx.blend"));
        let current_mode = self.properties.blending_mode;
        egui::ComboBox::from_id_salt("tool_blend_mode")
            .selected_text(current_mode.name())
            .width(90.0)
            .show_ui(ui, |ui: &mut egui::Ui| {
                for &mode in BlendMode::all() {
                    if ui
                        .selectable_label(mode == current_mode, mode.name())
                        .clicked()
                    {
                        self.properties.blending_mode = mode;
                    }
                }
            });
    }
}
