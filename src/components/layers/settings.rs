impl LayersPanel {
    /// Show the layer settings popup window (Options menu).
    /// For text layers this includes Effects and Warp tabs.
    fn show_layer_settings_popup(&mut self, ui: &mut egui::Ui, canvas_state: &mut CanvasState) {
        self.settings_popup_rect = None;
        if let Some(layer_idx) = self.settings_state.editing_layer {
            if layer_idx >= canvas_state.layers.len() {
                self.settings_state.editing_layer = None;
                return;
            }

            let is_text = matches!(
                canvas_state.layers[layer_idx].content,
                LayerContent::Text(_)
            );

            let mut open = true;
            let title = if is_text {
                t!("layer.text_options_title")
            } else {
                t!("layer.options_title")
            };
            let win_width = if is_text { 340.0 } else { 280.0 };

            let popup = egui::Window::new("layer_settings_popup_win")
                .id(Id::new("layer_settings_popup"))
                .title_bar(false)
                .collapsible(false)
                .resizable(true)
                .default_width(win_width)
                .show(ui.ctx(), |ui| {
                    // Header with close button
                    let colors = DialogColors::from_ctx(ui.ctx());
                    let available_w = ui.available_width();
                    let header_height = 32.0;
                    let (header_rect, _) = ui
                        .allocate_exact_size(Vec2::new(available_w, header_height), Sense::hover());
                    let painter = ui.painter();
                    painter.rect_filled(
                        header_rect,
                        egui::CornerRadius::same(4),
                        colors.accent_faint,
                    );
                    painter.rect_filled(
                        Rect::from_min_size(header_rect.min, Vec2::new(3.0, header_height)),
                        egui::CornerRadius::ZERO,
                        colors.accent,
                    );
                    painter.text(
                        Pos2::new(header_rect.min.x + 12.0, header_rect.center().y),
                        egui::Align2::LEFT_CENTER,
                        format!("\u{2699} {title}"),
                        egui::FontId::proportional(14.0),
                        colors.accent_strong,
                    );
                    // Close button (×) — matches panel_header ghost-style
                    let close_size = 18.0;
                    let close_rect = Rect::from_center_size(
                        Pos2::new(header_rect.right() - 14.0, header_rect.center().y),
                        Vec2::splat(close_size),
                    );
                    let close_resp =
                        ui.interact(close_rect, Id::new("ls_close_btn"), Sense::click());
                    if ui.is_rect_visible(close_rect) {
                        let hovered = close_resp.hovered();
                        if hovered {
                            painter.rect_filled(
                                close_rect,
                                egui::CornerRadius::same(4),
                                colors.accent_faint,
                            );
                        }
                        let color = if hovered {
                            colors.accent_strong
                        } else {
                            colors.text_muted
                        };
                        let font = egui::FontId::proportional(13.0);
                        let galley = painter.layout_no_wrap("\u{00D7}".to_string(), font, color);
                        let gpos = close_rect.center() - galley.size() / 2.0;
                        painter.galley(
                            Pos2::new(gpos.x, gpos.y),
                            galley,
                            egui::Color32::TRANSPARENT,
                        );
                    }
                    if close_resp.clicked() {
                        open = false;
                    }
                    ui.add_space(4.0);
                    // --- Tab bar for text layers ---
                    if is_text {
                        ui.horizontal(|ui| {
                            ui.spacing_mut().item_spacing.x = 0.0;
                            let tabs = [
                                (LayerSettingsTab::General, t!("layer.tab.general")),
                                (LayerSettingsTab::Effects, t!("layer.tab.effects")),
                                (LayerSettingsTab::Warp, t!("layer.tab.warp")),
                            ];
                            for (tab, label) in &tabs {
                                let selected = self.settings_state.tab == *tab;
                                let btn = egui::Button::new(
                                    egui::RichText::new(label.as_str())
                                        .strong()
                                        .size(13.0)
                                        .color(if selected {
                                            ui.visuals().strong_text_color()
                                        } else {
                                            ui.visuals().text_color()
                                        }),
                                )
                                .fill(if selected {
                                    ui.visuals().selection.bg_fill
                                } else {
                                    Color32::TRANSPARENT
                                })
                                .corner_radius(egui::CornerRadius {
                                    nw: 4,
                                    ne: 4,
                                    sw: 0,
                                    se: 0,
                                })
                                .min_size(Vec2::new(80.0, 24.0));
                                if ui.add(btn).clicked() {
                                    self.settings_state.tab = *tab;
                                }
                            }
                        });
                        // Underline for current tab
                        let accent = ui.visuals().selection.bg_fill;
                        let r = ui.available_rect_before_wrap();
                        ui.painter().line_segment(
                            [Pos2::new(r.left(), r.top()), Pos2::new(r.right(), r.top())],
                            egui::Stroke::new(1.0, accent.linear_multiply(0.3)),
                        );
                        ui.add_space(4.0);
                    }

                    match self.settings_state.tab {
                        LayerSettingsTab::General => {
                            self.show_settings_general_tab(ui, layer_idx, canvas_state);
                        }
                        LayerSettingsTab::Effects if is_text => {
                            self.show_settings_effects_tab(ui, layer_idx, canvas_state);
                        }
                        LayerSettingsTab::Warp if is_text => {
                            self.show_settings_warp_tab(ui, layer_idx, canvas_state);
                        }
                        _ => {
                            // Non-text layers only have General
                            self.show_settings_general_tab(ui, layer_idx, canvas_state);
                        }
                    }

                    // Rasterize button for text layers
                    if is_text {
                        ui.add_space(8.0);
                        ui.separator();
                        ui.add_space(4.0);
                        if ui
                            .button(
                                egui::RichText::new(t!("layer.rasterize_text_layer")).size(13.0),
                            )
                            .clicked()
                        {
                            self.pending_app_action =
                                Some(LayerAppAction::RasterizeTextLayer(layer_idx));
                            open = false;
                        }
                    }
                });

            if let Some(popup) = popup {
                let rect = popup.response.rect;
                self.settings_popup_rect = Some(rect);
                if ui
                    .ctx()
                    .input(|i| i.pointer.hover_pos().is_some_and(|pos| rect.contains(pos)))
                {
                    ui.ctx().set_cursor_icon(CursorIcon::Default);
                }
            }

            if !open {
                self.settings_state.editing_layer = None;
            }
        }
    }

    /// General tab: name, opacity, blend mode.
    fn show_settings_general_tab(
        &mut self,
        ui: &mut egui::Ui,
        layer_idx: usize,
        canvas_state: &mut CanvasState,
    ) {
        ui.horizontal(|ui| {
            ui.label(t!("layer.name"));
            if ui
                .text_edit_singleline(&mut self.settings_state.editing_name)
                .changed()
                && !self.settings_state.editing_name.is_empty()
            {
                canvas_state.layers[layer_idx].name = self.settings_state.editing_name.clone();
            }
        });

        ui.add_space(8.0);

        ui.horizontal(|ui| {
            ui.label(t!("layer.opacity"));
            if ui
                .add(
                    egui::Slider::new(&mut self.settings_state.editing_opacity, 0.0..=1.0)
                        .fixed_decimals(2)
                        .show_value(true),
                )
                .changed()
            {
                canvas_state.layers[layer_idx].opacity = self.settings_state.editing_opacity;
                self.mark_full_dirty(canvas_state);
            }
        });

        ui.add_space(8.0);

        ui.horizontal(|ui| {
            ui.label(t!("layer.blend"));
            egui::ComboBox::from_id_salt("blend_mode_combo_ls")
                .selected_text(self.settings_state.editing_blend_mode.display_name())
                .width(120.0)
                .show_ui(ui, |ui: &mut egui::Ui| {
                    for &mode in BlendMode::all() {
                        if ui
                            .selectable_label(
                                mode == self.settings_state.editing_blend_mode,
                                mode.display_name(),
                            )
                            .clicked()
                        {
                            self.settings_state.editing_blend_mode = mode;
                            canvas_state.layers[layer_idx].blend_mode = mode;
                            self.mark_full_dirty(canvas_state);
                        }
                    }
                });
        });

        if matches!(
            canvas_state.layers[layer_idx].content,
            LayerContent::Adjustment(_)
        ) {
            self.show_adjustment_settings(ui, layer_idx, canvas_state);
        }

        ui.add_space(8.0);
        ui.separator();
        ui.add_space(4.0);
        ui.label(egui::RichText::new("Image data").strong().size(12.0));
        ui.horizontal(|ui| {
            ui.label("Pixel format");
            let mut fmt = canvas_state.layers[layer_idx].pixel_format;
            egui::ComboBox::from_id_salt(("pixel_format_combo_ls", layer_idx))
                .selected_text(fmt.name())
                .width(120.0)
                .show_ui(ui, |ui| {
                    for next in [
                        crate::canvas::PixelFormat::RgbaU8,
                        crate::canvas::PixelFormat::RgbaU16,
                        crate::canvas::PixelFormat::RgbaF16,
                        crate::canvas::PixelFormat::RgbaF32,
                    ] {
                        if ui.selectable_label(fmt == next, next.name()).clicked() {
                            fmt = next;
                        }
                    }
                });
            if fmt != canvas_state.layers[layer_idx].pixel_format {
                canvas_state.layers[layer_idx].pixel_format = fmt;
            }
        });
        ui.horizontal(|ui| {
            ui.label("WebP frame");
            let mut mode = canvas_state.layers[layer_idx].webp_frame_compression;
            egui::ComboBox::from_id_salt(("webp_frame_mode_ls", layer_idx))
                .selected_text(mode.label())
                .width(120.0)
                .show_ui(ui, |ui| {
                    ui.selectable_value(
                        &mut mode,
                        crate::canvas::WebpFrameCompression::Lossless,
                        "Lossless",
                    );
                    ui.selectable_value(
                        &mut mode,
                        crate::canvas::WebpFrameCompression::Lossy,
                        "Lossy",
                    );
                });
            if mode != canvas_state.layers[layer_idx].webp_frame_compression {
                canvas_state.layers[layer_idx].webp_frame_compression = mode;
                canvas_state.mark_dirty(None);
            }
        });
        ui.label(
            egui::RichText::new("Only used for animated WebP export.")
                .size(11.0)
                .color(ui.visuals().weak_text_color()),
        );
        let mut hdr_enabled = canvas_state.layers[layer_idx].hdr_metadata.enabled;
        if ui.checkbox(&mut hdr_enabled, "HDR metadata").changed() {
            canvas_state.layers[layer_idx].hdr_metadata.enabled = hdr_enabled;
        }
        if hdr_enabled {
            let mut max_nits = canvas_state.layers[layer_idx]
                .hdr_metadata
                .max_luminance_nits
                .unwrap_or(1000.0);
            ui.horizontal(|ui| {
                ui.label("Max nits");
                if ui
                    .add(
                        egui::DragValue::new(&mut max_nits)
                            .range(80.0..=10000.0)
                            .speed(10.0),
                    )
                    .changed()
                {
                    canvas_state.layers[layer_idx]
                        .hdr_metadata
                        .max_luminance_nits = Some(max_nits);
                }
            });
        }
        let meta = &canvas_state.layers[layer_idx].source_metadata;
        if meta.source_format.is_some()
            || !meta.png_text_chunks.is_empty()
            || !meta.raw_png_chunks.is_empty()
        {
            ui.label(
                egui::RichText::new(format!(
                    "Metadata: {} text, {} raw chunks",
                    meta.png_text_chunks.len(),
                    meta.raw_png_chunks.len()
                ))
                .size(11.0),
            );
        }
    }

    fn show_adjustment_settings(
        &mut self,
        ui: &mut egui::Ui,
        layer_idx: usize,
        canvas_state: &mut CanvasState,
    ) {
        ui.add_space(8.0);
        ui.separator();
        ui.add_space(4.0);
        ui.label(egui::RichText::new("Adjustment").strong().size(12.0));

        let mut changed = false;
        let layer = &mut canvas_state.layers[layer_idx];
        let LayerContent::Adjustment(ref mut adj) = layer.content else {
            return;
        };

        let selected = match adj.kind {
            crate::canvas::AdjustmentKind::Exposure { .. } => "Exposure",
            crate::canvas::AdjustmentKind::BrightnessContrast { .. } => "Brightness / Contrast",
            crate::canvas::AdjustmentKind::Invert => "Invert",
            crate::canvas::AdjustmentKind::ChannelMixer { .. } => "Channel Mixer",
        };
        egui::ComboBox::from_id_salt(("adjustment_kind", layer_idx))
            .selected_text(selected)
            .width(180.0)
            .show_ui(ui, |ui| {
                if ui
                    .selectable_label(selected == "Exposure", "Exposure")
                    .clicked()
                {
                    adj.kind = crate::canvas::AdjustmentKind::Exposure { ev: 0.5 };
                    changed = true;
                }
                if ui
                    .selectable_label(selected == "Brightness / Contrast", "Brightness / Contrast")
                    .clicked()
                {
                    adj.kind = crate::canvas::AdjustmentKind::BrightnessContrast {
                        brightness: 0.0,
                        contrast: 0.0,
                    };
                    changed = true;
                }
                if ui
                    .selectable_label(selected == "Invert", "Invert")
                    .clicked()
                {
                    adj.kind = crate::canvas::AdjustmentKind::Invert;
                    changed = true;
                }
                if ui
                    .selectable_label(selected == "Channel Mixer", "Channel Mixer")
                    .clicked()
                {
                    adj.kind = crate::canvas::AdjustmentKind::ChannelMixer {
                        red: [1.0, 0.0, 0.0, 0.0],
                        green: [0.0, 1.0, 0.0, 0.0],
                        blue: [0.0, 0.0, 1.0, 0.0],
                        alpha: [0.0, 0.0, 0.0, 1.0],
                    };
                    changed = true;
                }
            });

        match &mut adj.kind {
            crate::canvas::AdjustmentKind::Exposure { ev } => {
                changed |= ui
                    .add(
                        egui::Slider::new(ev, -4.0..=4.0)
                            .text("EV")
                            .fixed_decimals(2),
                    )
                    .changed();
            }
            crate::canvas::AdjustmentKind::BrightnessContrast {
                brightness,
                contrast,
            } => {
                changed |= ui
                    .add(egui::Slider::new(brightness, -255.0..=255.0).text("Brightness"))
                    .changed();
                changed |= ui
                    .add(egui::Slider::new(contrast, -255.0..=255.0).text("Contrast"))
                    .changed();
            }
            crate::canvas::AdjustmentKind::Invert => {
                ui.label("Inverts RGB; alpha is preserved.");
            }
            crate::canvas::AdjustmentKind::ChannelMixer {
                red,
                green,
                blue,
                alpha,
            } => {
                for (label, row) in [
                    ("Red", red),
                    ("Green", green),
                    ("Blue", blue),
                    ("Alpha", alpha),
                ] {
                    ui.label(label);
                    ui.horizontal(|ui| {
                        for (i, coeff) in row.iter_mut().enumerate() {
                            let name = ["R", "G", "B", "A"][i];
                            changed |= ui
                                .add(
                                    egui::DragValue::new(coeff)
                                        .range(-2.0..=2.0)
                                        .speed(0.05)
                                        .prefix(name),
                                )
                                .changed();
                        }
                    });
                }
            }
        }

        if changed {
            layer.gpu_generation = layer.gpu_generation.wrapping_add(1);
            self.mark_full_dirty(canvas_state);
        }
    }

    /// Effects tab (text layers only): outline, shadow, inner shadow, texture fill.
    fn show_settings_effects_tab(
        &mut self,
        ui: &mut egui::Ui,
        layer_idx: usize,
        canvas_state: &mut CanvasState,
    ) {
        let mut changed = false;

        // --- Outline ---
        {
            let has_outline = self.settings_state.text_effects.outline.is_some();
            let mut outline_on = has_outline;
            if ui
                .checkbox(&mut outline_on, t!("ctx.text.effects.outline"))
                .changed()
            {
                changed = true;
            }
            if outline_on && !has_outline {
                self.settings_state.text_effects.outline = Some(OutlineEffect::default());
                changed = true;
            } else if !outline_on && has_outline {
                self.settings_state.text_effects.outline = None;
                changed = true;
            }
            if let Some(ref mut outline) = self.settings_state.text_effects.outline {
                ui.indent("ls_outline", |ui| {
                    ui.horizontal(|ui| {
                        ui.label(t!("ctx.text.effects.outline.color"));
                        let mut c = Color32::from_rgba_unmultiplied(
                            outline.color[0],
                            outline.color[1],
                            outline.color[2],
                            outline.color[3],
                        );
                        if ui.color_edit_button_srgba(&mut c).changed() {
                            outline.color = [c.r(), c.g(), c.b(), c.a()];
                            changed = true;
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label(t!("ctx.text.effects.outline.width"));
                        if ui
                            .add(
                                egui::DragValue::new(&mut outline.width)
                                    .speed(0.1)
                                    .range(0.5..=50.0)
                                    .suffix("px"),
                            )
                            .changed()
                        {
                            changed = true;
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label(t!("ctx.text.effects.outline.position"));
                        let positions = [
                            (
                                OutlinePosition::Outside,
                                t!("ctx.text.effects.outline.outside"),
                            ),
                            (
                                OutlinePosition::Inside,
                                t!("ctx.text.effects.outline.inside"),
                            ),
                            (
                                OutlinePosition::Center,
                                t!("ctx.text.effects.outline.center"),
                            ),
                        ];
                        for (pos, label) in &positions {
                            if ui
                                .selectable_label(outline.position == *pos, label.as_str())
                                .clicked()
                            {
                                outline.position = *pos;
                                changed = true;
                            }
                        }
                    });
                });
            }
        }

        ui.add_space(4.0);

        // --- Drop Shadow ---
        {
            let has_shadow = self.settings_state.text_effects.shadow.is_some();
            let mut shadow_on = has_shadow;
            if ui
                .checkbox(&mut shadow_on, t!("ctx.text.effects.shadow"))
                .changed()
            {
                changed = true;
            }
            if shadow_on && !has_shadow {
                self.settings_state.text_effects.shadow = Some(ShadowEffect::default());
                changed = true;
            } else if !shadow_on && has_shadow {
                self.settings_state.text_effects.shadow = None;
                changed = true;
            }
            if let Some(ref mut shadow) = self.settings_state.text_effects.shadow {
                ui.indent("ls_shadow", |ui| {
                    ui.horizontal(|ui| {
                        ui.label(t!("ctx.text.effects.shadow.color"));
                        let mut c = Color32::from_rgba_unmultiplied(
                            shadow.color[0],
                            shadow.color[1],
                            shadow.color[2],
                            shadow.color[3],
                        );
                        if ui.color_edit_button_srgba(&mut c).changed() {
                            shadow.color = [c.r(), c.g(), c.b(), c.a()];
                            changed = true;
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label(t!("ctx.text.effects.shadow.offset_x"));
                        if ui
                            .add(
                                egui::DragValue::new(&mut shadow.offset_x)
                                    .speed(0.5)
                                    .range(-100.0..=100.0)
                                    .suffix("px"),
                            )
                            .changed()
                        {
                            changed = true;
                        }
                        ui.label(t!("ctx.text.effects.shadow.offset_y"));
                        if ui
                            .add(
                                egui::DragValue::new(&mut shadow.offset_y)
                                    .speed(0.5)
                                    .range(-100.0..=100.0)
                                    .suffix("px"),
                            )
                            .changed()
                        {
                            changed = true;
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label(t!("ctx.text.effects.shadow.blur"));
                        let r = ui.add(
                            egui::Slider::new(&mut shadow.blur_radius, 0.0..=50.0)
                                .suffix(" px")
                                .max_decimals(1),
                        );
                        if r.changed() {
                            changed = true;
                        }
                        ui.label(t!("ctx.text.effects.shadow.spread"));
                        if ui
                            .add(
                                egui::DragValue::new(&mut shadow.spread)
                                    .speed(0.2)
                                    .range(0.0..=30.0)
                                    .suffix("px"),
                            )
                            .changed()
                        {
                            changed = true;
                        }
                    });
                });
            }
        }

        ui.add_space(4.0);

        // --- Inner Shadow ---
        {
            let has_inner = self.settings_state.text_effects.inner_shadow.is_some();
            let mut inner_on = has_inner;
            if ui
                .checkbox(&mut inner_on, t!("ctx.text.effects.inner_shadow"))
                .changed()
            {
                changed = true;
            }
            if inner_on && !has_inner {
                self.settings_state.text_effects.inner_shadow = Some(InnerShadowEffect::default());
                changed = true;
            } else if !inner_on && has_inner {
                self.settings_state.text_effects.inner_shadow = None;
                changed = true;
            }
            if let Some(ref mut inner) = self.settings_state.text_effects.inner_shadow {
                ui.indent("ls_inner_shadow", |ui| {
                    ui.horizontal(|ui| {
                        ui.label(t!("ctx.text.effects.inner_shadow.color"));
                        let mut c = Color32::from_rgba_unmultiplied(
                            inner.color[0],
                            inner.color[1],
                            inner.color[2],
                            inner.color[3],
                        );
                        if ui.color_edit_button_srgba(&mut c).changed() {
                            inner.color = [c.r(), c.g(), c.b(), c.a()];
                            changed = true;
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label(t!("ctx.text.effects.inner_shadow.offset_x"));
                        if ui
                            .add(
                                egui::DragValue::new(&mut inner.offset_x)
                                    .speed(0.5)
                                    .range(-100.0..=100.0)
                                    .suffix("px"),
                            )
                            .changed()
                        {
                            changed = true;
                        }
                        ui.label(t!("ctx.text.effects.inner_shadow.offset_y"));
                        if ui
                            .add(
                                egui::DragValue::new(&mut inner.offset_y)
                                    .speed(0.5)
                                    .range(-100.0..=100.0)
                                    .suffix("px"),
                            )
                            .changed()
                        {
                            changed = true;
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label(t!("ctx.text.effects.inner_shadow.blur"));
                        if ui
                            .add(
                                egui::DragValue::new(&mut inner.blur_radius)
                                    .speed(0.2)
                                    .range(0.0..=50.0)
                                    .suffix("px"),
                            )
                            .changed()
                        {
                            changed = true;
                        }
                    });
                });
            }
        }

        ui.add_space(4.0);

        // --- Gradient Fill ---
        {
            let has_gradient = self.settings_state.text_effects.gradient_fill.is_some();
            let mut gradient_on = has_gradient;
            if ui
                .checkbox(&mut gradient_on, t!("ctx.text.effects.gradient"))
                .changed()
            {
                changed = true;
            }
            if gradient_on && !has_gradient {
                self.settings_state.text_effects.gradient_fill =
                    Some(GradientFillEffect::default());
                self.settings_state.text_effects.texture_fill = None;
                changed = true;
            } else if !gradient_on && has_gradient {
                self.settings_state.text_effects.gradient_fill = None;
                changed = true;
            }

            if let Some(ref mut gradient) = self.settings_state.text_effects.gradient_fill {
                ui.indent("ls_gradient_fill", |ui| {
                    ui.horizontal(|ui| {
                        ui.label(t!("ctx.text.effects.gradient.start"));
                        let mut c = Color32::from_rgba_unmultiplied(
                            gradient.start_color[0],
                            gradient.start_color[1],
                            gradient.start_color[2],
                            gradient.start_color[3],
                        );
                        if ui.color_edit_button_srgba(&mut c).changed() {
                            gradient.start_color = [c.r(), c.g(), c.b(), c.a()];
                            changed = true;
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label(t!("ctx.text.effects.gradient.end"));
                        let mut c = Color32::from_rgba_unmultiplied(
                            gradient.end_color[0],
                            gradient.end_color[1],
                            gradient.end_color[2],
                            gradient.end_color[3],
                        );
                        if ui.color_edit_button_srgba(&mut c).changed() {
                            gradient.end_color = [c.r(), c.g(), c.b(), c.a()];
                            changed = true;
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label(t!("ctx.text.effects.gradient.angle"));
                        if ui
                            .add(
                                egui::DragValue::new(&mut gradient.angle_degrees)
                                    .speed(1.0)
                                    .range(-360.0..=360.0)
                                    .suffix("deg"),
                            )
                            .changed()
                        {
                            changed = true;
                        }
                        ui.label(t!("ctx.text.effects.gradient.scale"));
                        if ui
                            .add(
                                egui::DragValue::new(&mut gradient.scale)
                                    .speed(1.0)
                                    .range(1.0..=5000.0)
                                    .suffix("px"),
                            )
                            .changed()
                        {
                            changed = true;
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label(t!("ctx.text.effects.gradient.offset_x"));
                        if ui
                            .add(
                                egui::DragValue::new(&mut gradient.offset[0])
                                    .speed(0.5)
                                    .range(-5000.0..=5000.0),
                            )
                            .changed()
                        {
                            changed = true;
                        }
                        ui.label(t!("ctx.text.effects.gradient.offset_y"));
                        if ui
                            .add(
                                egui::DragValue::new(&mut gradient.offset[1])
                                    .speed(0.5)
                                    .range(-5000.0..=5000.0),
                            )
                            .changed()
                        {
                            changed = true;
                        }
                    });
                    if ui
                        .checkbox(&mut gradient.repeat, t!("ctx.text.effects.gradient.repeat"))
                        .changed()
                    {
                        changed = true;
                    }
                });
            }
        }

        ui.add_space(4.0);

        // --- Texture Fill ---
        {
            let has_texture = self.settings_state.text_effects.texture_fill.is_some();
            let mut texture_on = has_texture;
            if ui
                .checkbox(&mut texture_on, t!("ctx.text.effects.texture"))
                .changed()
            {
                changed = true;
            }
            if texture_on && !has_texture {
                self.settings_state.text_effects.texture_fill = Some(TextureFillEffect::default());
                self.settings_state.text_effects.gradient_fill = None;
                changed = true;
            } else if !texture_on && has_texture {
                self.settings_state.text_effects.texture_fill = None;
                changed = true;
            }

            // Poll for async texture load
            if let Some(ref rx) = self.settings_state.texture_load_rx
                && let Ok(data) = rx.try_recv()
            {
                if let Ok(img) = image::load_from_memory(&data)
                    && let Some(ref mut tex) = self.settings_state.text_effects.texture_fill
                {
                    tex.texture_data = data;
                    tex.texture_width = img.width();
                    tex.texture_height = img.height();
                    changed = true;
                }
                self.settings_state.texture_load_rx = None;
            }
            // Poll for async texture load from the browser file picker.
            #[cfg(target_arch = "wasm32")]
            if let Some((_name, data)) =
                crate::web_bridge::drain_pending("texture_fill").into_iter().next()
                && let Ok(img) = image::load_from_memory(&data)
                && let Some(ref mut tex) = self.settings_state.text_effects.texture_fill
            {
                tex.texture_data = data;
                tex.texture_width = img.width();
                tex.texture_height = img.height();
                changed = true;
            }

            let mut remove_texture = false;
            let mut spawn_texture_dialog = false;
            if let Some(ref mut tex) = self.settings_state.text_effects.texture_fill {
                ui.indent("ls_texture", |ui| {
                    ui.horizontal(|ui| {
                        if ui.button(t!("ctx.text.effects.texture.load")).clicked() {
                            spawn_texture_dialog = true;
                        }
                        if tex.texture_width > 0 {
                            ui.label(format!("{}×{}", tex.texture_width, tex.texture_height));
                        }
                    });
                    if tex.texture_width > 0 {
                        ui.horizontal(|ui| {
                            ui.label(t!("ctx.text.effects.texture.scale"));
                            if ui
                                .add(
                                    egui::DragValue::new(&mut tex.scale)
                                        .speed(0.01)
                                        .range(0.1..=10.0),
                                )
                                .changed()
                            {
                                changed = true;
                            }
                        });
                        ui.horizontal(|ui| {
                            ui.label(t!("ctx.text.effects.texture.offset_x"));
                            if ui
                                .add(
                                    egui::DragValue::new(&mut tex.offset[0])
                                        .speed(0.5)
                                        .range(-1000.0..=1000.0),
                                )
                                .changed()
                            {
                                changed = true;
                            }
                            ui.label(t!("ctx.text.effects.texture.offset_y"));
                            if ui
                                .add(
                                    egui::DragValue::new(&mut tex.offset[1])
                                        .speed(0.5)
                                        .range(-1000.0..=1000.0),
                                )
                                .changed()
                            {
                                changed = true;
                            }
                        });
                        if ui.button(t!("ctx.text.effects.texture.remove")).clicked() {
                            remove_texture = true;
                            changed = true;
                        }
                    }
                });
            }
            if remove_texture {
                self.settings_state.text_effects.texture_fill = None;
            }
            // Texture-fill file picker runs on a background thread and uses a
            // native dialog on desktop; on web it opens the browser's
            // `<input type=file>` picker instead (see web_bridge).
            #[cfg(not(target_arch = "wasm32"))]
            if spawn_texture_dialog {
                let (tx, rx) = std::sync::mpsc::channel();
                self.settings_state.texture_load_rx = Some(rx);
                std::thread::spawn(move || {
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("Images", &["png", "jpg", "jpeg", "bmp", "webp"])
                        .pick_file()
                        && let Ok(data) = std::fs::read(&path)
                    {
                        let _ = tx.send(data);
                    }
                });
            }
            #[cfg(target_arch = "wasm32")]
            if spawn_texture_dialog {
                crate::web_bridge::open_picker("texture_fill", "image/*", false);
            }
        }

        // Live-commit: write effects to TextLayerData and rasterize immediately
        if changed {
            if let Some(layer) = canvas_state.layers.get_mut(layer_idx)
                && let LayerContent::Text(ref mut td) = layer.content
            {
                td.effects = self.settings_state.text_effects.clone();
                td.mark_effects_dirty();
            }
            canvas_state.force_rasterize_text_layer(layer_idx);
            canvas_state.mark_dirty(None);
        }
    }

    /// Warp tab (text layers only): warp type selector and per-type controls.
    fn show_settings_warp_tab(
        &mut self,
        ui: &mut egui::Ui,
        layer_idx: usize,
        canvas_state: &mut CanvasState,
    ) {
        let mut changed = false;

        let current_name = self.settings_state.text_warp.name().to_string();
        ui.horizontal(|ui| {
            ui.label(t!("ctx.text.warp.type"));
            egui::ComboBox::from_id_salt("ls_text_warp_type")
                .selected_text(&current_name)
                .width(130.0)
                .show_ui(ui, |ui| {
                    for name in TextWarp::all_names() {
                        if *name == "Path Follow" {
                            continue; // hidden for now
                        }
                        if ui.selectable_label(current_name == *name, *name).clicked() {
                            let new_warp = TextWarp::from_name(name);
                            if self.settings_state.text_warp != new_warp {
                                self.settings_state.text_warp = new_warp;
                                changed = true;
                            }
                        }
                    }
                });
        });

        ui.add_space(4.0);

        match &mut self.settings_state.text_warp {
            TextWarp::None => {}
            TextWarp::Arc(arc) => {
                ui.horizontal(|ui| {
                    ui.label(t!("ctx.text.warp.arc.bend"));
                    if ui
                        .add(
                            egui::DragValue::new(&mut arc.bend)
                                .speed(0.01)
                                .range(-1.0..=1.0),
                        )
                        .changed()
                    {
                        changed = true;
                    }
                });
                ui.horizontal(|ui| {
                    ui.label(t!("ctx.text.warp.arc.hdist"));
                    if ui
                        .add(
                            egui::DragValue::new(&mut arc.horizontal_distortion)
                                .speed(0.01)
                                .range(-1.0..=1.0),
                        )
                        .changed()
                    {
                        changed = true;
                    }
                });
                ui.horizontal(|ui| {
                    ui.label(t!("ctx.text.warp.arc.vdist"));
                    if ui
                        .add(
                            egui::DragValue::new(&mut arc.vertical_distortion)
                                .speed(0.01)
                                .range(-1.0..=1.0),
                        )
                        .changed()
                    {
                        changed = true;
                    }
                });
            }
            TextWarp::Circular(circ) => {
                ui.horizontal(|ui| {
                    ui.label(t!("ctx.text.warp.circular.radius"));
                    if ui
                        .add(
                            egui::DragValue::new(&mut circ.radius)
                                .speed(1.0)
                                .range(20.0..=2000.0)
                                .suffix("px"),
                        )
                        .changed()
                    {
                        changed = true;
                    }
                });
                ui.horizontal(|ui| {
                    ui.label(t!("ctx.text.warp.circular.start_angle"));
                    let mut degrees = circ.start_angle.to_degrees();
                    if ui
                        .add(
                            egui::DragValue::new(&mut degrees)
                                .speed(1.0)
                                .range(-360.0..=360.0)
                                .suffix("°"),
                        )
                        .changed()
                    {
                        circ.start_angle = degrees.to_radians();
                        changed = true;
                    }
                });
                ui.horizontal(|ui| {
                    ui.label(t!("ctx.text.warp.circular.clockwise"));
                    if ui.checkbox(&mut circ.clockwise, "").changed() {
                        changed = true;
                    }
                });
            }
            TextWarp::PathFollow(_) => {
                // Hidden from UI for now; no controls shown.
            }
            TextWarp::Envelope(env) => {
                ui.label(t!("ctx.text.warp.envelope.top"));
                let mut any_changed = false;
                for (i, pt) in env.top_curve.iter_mut().enumerate() {
                    ui.horizontal(|ui| {
                        ui.label(format!("T{i}:"));
                        if ui
                            .add(egui::DragValue::new(&mut pt[0]).speed(1.0).prefix("x: "))
                            .changed()
                        {
                            any_changed = true;
                        }
                        if ui
                            .add(egui::DragValue::new(&mut pt[1]).speed(1.0).prefix("y: "))
                            .changed()
                        {
                            any_changed = true;
                        }
                    });
                }
                ui.add_space(2.0);
                ui.label(t!("ctx.text.warp.envelope.bottom"));
                for (i, pt) in env.bottom_curve.iter_mut().enumerate() {
                    ui.horizontal(|ui| {
                        ui.label(format!("B{i}:"));
                        if ui
                            .add(egui::DragValue::new(&mut pt[0]).speed(1.0).prefix("x: "))
                            .changed()
                        {
                            any_changed = true;
                        }
                        if ui
                            .add(egui::DragValue::new(&mut pt[1]).speed(1.0).prefix("y: "))
                            .changed()
                        {
                            any_changed = true;
                        }
                    });
                }
                if any_changed {
                    changed = true;
                }
                if ui.button(t!("ctx.text.warp.reset")).clicked() {
                    *env = EnvelopeWarp::default();
                    changed = true;
                }
            }
        }

        // Live-commit: write warp to ALL blocks and rasterize immediately
        if changed {
            if let Some(layer) = canvas_state.layers.get_mut(layer_idx)
                && let LayerContent::Text(ref mut td) = layer.content
            {
                for block in &mut td.blocks {
                    block.warp = self.settings_state.text_warp.clone();
                }
                td.mark_dirty();
            }
            canvas_state.force_rasterize_text_layer(layer_idx);
            canvas_state.mark_dirty(None);
        }
    }

    // === Layer Operations ===
}
