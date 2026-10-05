impl ToolsPanel {
    fn show_text_options(&mut self, ui: &mut egui::Ui, assets: &Assets) {
        // Web only: user-uploaded/Google Fonts can be added at any time from
        // Settings. Detect that via a generation counter and force the
        // font-list + preview caches below to recompute.
        #[cfg(target_arch = "wasm32")]
        {
            let font_gen = crate::ops::text::custom_fonts::generation();
            if font_gen != self.text_state.custom_fonts_seen_gen {
                self.text_state.custom_fonts_seen_gen = font_gen;
                self.text_state.available_fonts.clear();
                self.text_state.font_preview_cache.clear();
            }
        }

        // Async font loading: kick off background thread on first access
        if self.text_state.available_fonts.is_empty() && self.text_state.fonts_loading_rx.is_none()
        {
            let (tx, rx) = std::sync::mpsc::channel();
            self.text_state.fonts_loading_rx = Some(rx);
            crate::par_compat::spawn(move || {
                let fonts = crate::ops::text::enumerate_system_fonts();
                let _ = tx.send(fonts);
            });
        }

        // Check if async font list is ready
        if let Some(ref rx) = self.text_state.fonts_loading_rx
            && let Ok(fonts) = rx.try_recv()
        {
            self.text_state.available_fonts = fonts;
            self.text_state.fonts_loading_rx = None;
            // Refresh weights for current family
            self.text_state.available_weights =
                crate::ops::text::enumerate_font_weights(&self.text_state.font_family);
        }

        // Poll async font preview results. The loader sends multiple batches
        // until all family previews are cached.
        if let Some(rx) = self.text_state.font_preview_rx.take() {
            let mut keep_rx = true;
            loop {
                match rx.try_recv() {
                    Ok(batch) => {
                        for (name, font) in batch {
                            self.text_state.font_preview_pending.remove(&name);
                            self.text_state.font_preview_cache.insert(name, font);
                        }
                    }
                    Err(std::sync::mpsc::TryRecvError::Empty) => break,
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                        keep_rx = false;
                        break;
                    }
                }
            }
            if keep_rx {
                self.text_state.font_preview_rx = Some(rx);
            }
        }

        // Ensure all font previews are loaded in the background with one
        // persistent batch stream (egui virtualized rows can then render from cache).
        if !self.text_state.available_fonts.is_empty()
            && self.text_state.font_preview_rx.is_none()
            && self.text_state.font_preview_pending.is_empty()
            && self.text_state.font_preview_cache.len() < self.text_state.available_fonts.len()
        {
            let missing: Vec<String> = self
                .text_state
                .available_fonts
                .iter()
                .filter(|name| {
                    !self
                        .text_state
                        .font_preview_cache
                        .contains_key(name.as_str())
                })
                .cloned()
                .collect();
            if !missing.is_empty() {
                for name in &missing {
                    self.text_state.font_preview_pending.insert(name.clone());
                }
                let (tx, rx) = std::sync::mpsc::channel();
                self.text_state.font_preview_rx = Some(rx);
                crate::par_compat::spawn(move || {
                    const BATCH: usize = 24;
                    let mut out: Vec<(String, Option<ab_glyph::FontArc>)> =
                        Vec::with_capacity(BATCH);
                    for name in missing {
                        let font = crate::ops::text::load_system_font(&name, 400, false);
                        out.push((name, font));
                        if out.len() >= BATCH {
                            if tx.send(out).is_err() {
                                return;
                            }
                            out = Vec::with_capacity(BATCH);
                        }
                    }
                    if !out.is_empty() {
                        let _ = tx.send(out);
                    }
                });
            }
        }

        ui.label(t!("ctx.text.font"));
        let family_label = self.text_state.font_family.clone();
        let popup_id = ui.make_persistent_id("ctx_text_font_popup");
        self.text_state.font_popup_open = egui::Popup::is_id_open(ui.ctx(), popup_id);

        if self.text_state.available_fonts.is_empty() {
            // Still loading
            ui.add(
                egui::Button::new(t!("ctx.text.loading_fonts")).min_size(egui::vec2(140.0, 0.0)),
            );
            ui.ctx()
                .request_repaint_after(std::time::Duration::from_millis(100));
        } else {
            let button_response = ui.add(
                egui::Button::new(egui::RichText::new(if family_label.len() > 20 {
                    &family_label[..20]
                } else {
                    &family_label
                }))
                .min_size(egui::vec2(140.0, 0.0)),
            );
            if button_response.clicked() {
                egui::Popup::toggle_id(ui.ctx(), popup_id);
            }
            egui::Popup::new(
                popup_id,
                ui.ctx().clone(),
                egui::PopupAnchor::from(&button_response),
                ui.layer_id(),
            )
            .open_memory(None::<egui::SetOpenCommand>)
            .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
            .show(|ui| {
                egui::Frame::popup(ui.style()).show(ui, |ui| {
                    const FONT_LIST_HEIGHT: f32 = 300.0;
                    const FONT_POPUP_WIDTH: f32 = 360.0;
                    const FONT_LIST_WIDTH: f32 = FONT_POPUP_WIDTH - 18.0;
                    let preview_ink = {
                        let color = contrast_text_color(ui.visuals().window_fill());
                        [color.r(), color.g(), color.b(), color.a()]
                    };
                    if self.text_state.font_preview_ink.0 != preview_ink {
                        self.text_state.font_preview_ink = FontPreviewInk(preview_ink);
                        self.text_state.font_preview_textures.0.clear();
                    }
                    ui.set_min_width(FONT_POPUP_WIDTH);
                    ui.set_max_width(FONT_POPUP_WIDTH);
                    // Keep popup height stable so clearing search restores a tall list.
                    ui.set_min_height(FONT_LIST_HEIGHT + 38.0);
                    let te_resp = ui.add(
                        egui::TextEdit::singleline(&mut self.text_state.font_search)
                            .hint_text(t!("ctx.text.search"))
                            .desired_width(FONT_LIST_WIDTH - 6.0),
                    );
                    if !te_resp.has_focus() && self.text_state.font_search.is_empty() {
                        te_resp.request_focus();
                    }
                    let search = self.text_state.font_search.to_lowercase();
                    let fonts: Vec<&String> = self
                        .text_state
                        .available_fonts
                        .iter()
                        .filter(|f| search.is_empty() || f.to_lowercase().contains(&search))
                        .collect();

                    let row_height = 22.0;
                    let total_rows = fonts.len().min(200);
                    let list_width = FONT_LIST_WIDTH;
                    egui::ScrollArea::vertical()
                        .auto_shrink([false, false])
                        .min_scrolled_width(list_width)
                        .max_height(FONT_LIST_HEIGHT)
                        .show_rows(ui, row_height, total_rows, |ui, row_range| {
                            for idx in row_range {
                                if idx >= fonts.len() {
                                    break;
                                }
                                let font_name = &fonts[idx];
                                let is_selected = self.text_state.font_family == **font_name;
                                let preview_text = "Abc";
                                let row_width = list_width;
                                let preview_width = 118.0;
                                let gap = 8.0;
                                let name_width = (row_width - preview_width - gap).max(120.0);

                                let (row_rect, row_resp) = ui.allocate_exact_size(
                                    egui::vec2(row_width, row_height),
                                    egui::Sense::click(),
                                );
                                let name_rect = egui::Rect::from_min_max(
                                    row_rect.min,
                                    egui::pos2(row_rect.min.x + name_width, row_rect.max.y),
                                );
                                let preview_rect = egui::Rect::from_min_max(
                                    egui::pos2(name_rect.max.x + gap, row_rect.min.y),
                                    row_rect.max,
                                );
                                let row_clip = row_rect.intersect(ui.clip_rect());
                                let painter = ui.painter().with_clip_rect(row_clip);
                                let row_fully_visible = row_clip.height() >= (row_height - 0.5);

                                if row_resp.hovered() && row_fully_visible {
                                    painter.rect_filled(
                                        row_rect.shrink2(egui::vec2(1.0, 1.0)),
                                        0.0,
                                        ui.visuals().widgets.hovered.weak_bg_fill,
                                    );
                                }

                                if is_selected && row_fully_visible {
                                    painter.rect_filled(
                                        name_rect.shrink2(egui::vec2(2.0, 2.0)),
                                        0.0,
                                        ui.visuals().selection.bg_fill,
                                    );
                                }

                                let display_name = if font_name.len() > 22 {
                                    format!("{}...", &font_name[..21])
                                } else {
                                    (*font_name).clone()
                                };
                                painter.text(
                                    egui::pos2(name_rect.min.x + 6.0, name_rect.center().y),
                                    egui::Align2::LEFT_CENTER,
                                    display_name,
                                    egui::FontId::default(),
                                    ui.visuals().text_color(),
                                );

                                match self.text_state.font_preview_cache.get(font_name.as_str()) {
                                    Some(Some(preview_font)) => {
                                        if !self
                                            .text_state
                                            .font_preview_textures
                                            .0
                                            .contains_key(font_name.as_str())
                                        {
                                            let mut preview_coverage = Vec::new();
                                            let mut preview_glyph_cache =
                                                crate::ops::text::GlyphPixelCache::default();
                                            let preview = crate::ops::text::rasterize_text(
                                                preview_font,
                                                crate::ops::text::font_cache_key(
                                                    font_name.as_str(),
                                                    400,
                                                    false,
                                                ),
                                                preview_text,
                                                18.0,
                                                crate::ops::text::TextAlignment::Left,
                                                0.0,
                                                0.0,
                                                self.text_state.font_preview_ink.0,
                                                true,
                                                false,
                                                false,
                                                false,
                                                false,
                                                256,
                                                64,
                                                &mut preview_coverage,
                                                &mut preview_glyph_cache,
                                                None,
                                                0.0,
                                                1.0,
                                                1.0,
                                                1.0,
                                            );
                                            if preview.buf_w > 0 && preview.buf_h > 0 {
                                                let color_image =
                                                    egui::ColorImage::from_rgba_unmultiplied(
                                                        [
                                                            preview.buf_w as usize,
                                                            preview.buf_h as usize,
                                                        ],
                                                        &preview.buf,
                                                    );
                                                let texture = ui.ctx().load_texture(
                                                    format!("font_preview_tex::{}", font_name),
                                                    color_image,
                                                    egui::TextureOptions::LINEAR,
                                                );
                                                self.text_state
                                                    .font_preview_textures
                                                    .0
                                                    .insert((*font_name).clone(), texture);
                                            }
                                        }

                                        if let Some(texture) = self
                                            .text_state
                                            .font_preview_textures
                                            .0
                                            .get(font_name.as_str())
                                        {
                                            let tex_size = texture.size_vec2();
                                            let max_w = (preview_rect.width() - 24.0).max(8.0);
                                            let max_h = (preview_rect.height() - 4.0).max(8.0);
                                            let scale = (max_w / tex_size.x)
                                                .min(max_h / tex_size.y)
                                                .max(0.01);
                                            let draw_size = tex_size * scale;
                                            let image_rect = egui::Rect::from_min_size(
                                                egui::pos2(
                                                    preview_rect.min.x + 4.0,
                                                    preview_rect.center().y - draw_size.y * 0.5,
                                                ),
                                                draw_size,
                                            );
                                            painter.image(
                                                texture.id(),
                                                image_rect,
                                                egui::Rect::from_min_max(
                                                    egui::pos2(0.0, 0.0),
                                                    egui::pos2(1.0, 1.0),
                                                ),
                                                Color32::WHITE,
                                            );
                                        } else {
                                            painter.text(
                                                egui::pos2(
                                                    preview_rect.min.x + 4.0,
                                                    preview_rect.center().y,
                                                ),
                                                egui::Align2::LEFT_CENTER,
                                                "No preview",
                                                egui::FontId::default(),
                                                ui.visuals().weak_text_color(),
                                            );
                                        }
                                    }
                                    Some(None) => {
                                        painter.text(
                                            egui::pos2(
                                                preview_rect.min.x + 4.0,
                                                preview_rect.center().y,
                                            ),
                                            egui::Align2::LEFT_CENTER,
                                            "No preview",
                                            egui::FontId::default(),
                                            ui.visuals().weak_text_color(),
                                        );
                                    }
                                    None => {
                                        let pending = self
                                            .text_state
                                            .font_preview_pending
                                            .contains(font_name.as_str());
                                        painter.text(
                                            egui::pos2(
                                                preview_rect.min.x + 4.0,
                                                preview_rect.center().y,
                                            ),
                                            egui::Align2::LEFT_CENTER,
                                            if pending { "Loading" } else { "Queued" },
                                            egui::FontId::default(),
                                            ui.visuals().weak_text_color(),
                                        );
                                    }
                                }

                                if row_resp.clicked() {
                                    self.text_state.font_family = (*font_name).clone();
                                    self.text_state.loaded_font = None;
                                    self.text_state.preview_dirty = true;
                                    self.text_state.pending_ctx_style_update =
                                        Some(TextStyleUpdate::FontFamily(
                                            self.text_state.font_family.clone(),
                                        ));
                                    self.text_state.ctx_bar_style_dirty = true;
                                    self.text_state.cached_raster_key.clear();
                                    // Refresh available weights for new family
                                    self.text_state.available_weights =
                                        crate::ops::text::enumerate_font_weights(
                                            &self.text_state.font_family,
                                        );
                                    if !self
                                        .text_state
                                        .available_weights
                                        .iter()
                                        .any(|w| w.1 == self.text_state.font_weight)
                                    {
                                        self.text_state.font_weight = 400;
                                    }
                                    egui::Popup::close_id(ui.ctx(), popup_id);
                                }
                            }
                        });

                    if !self.text_state.font_preview_pending.is_empty() && total_rows > 0 {
                        ui.ctx()
                            .request_repaint_after(std::time::Duration::from_millis(16));
                    }
                });
            });
            self.text_state.font_popup_open = egui::Popup::is_id_open(ui.ctx(), popup_id);
        }

        // Weight dropdown (only show if more than one weight available)
        if self.text_state.available_weights.len() > 1 {
            ui.separator();
            ui.label(t!("ctx.text.weight"));
            let current_weight_label = self
                .text_state
                .available_weights
                .iter()
                .find(|w| w.1 == self.text_state.font_weight)
                .map(|w| w.0.as_str())
                .unwrap_or("Regular");
            egui::ComboBox::from_id_salt("ctx_text_weight")
                .selected_text(current_weight_label)
                .width(90.0)
                .show_ui(ui, |ui| {
                    for (name, val) in &self.text_state.available_weights {
                        if ui
                            .selectable_label(self.text_state.font_weight == *val, name)
                            .clicked()
                        {
                            self.text_state.font_weight = *val;
                            self.text_state.loaded_font = None;
                            self.text_state.preview_dirty = true;
                            self.text_state.pending_ctx_style_update =
                                Some(TextStyleUpdate::FontWeight(*val));
                            self.text_state.ctx_bar_style_dirty = true;
                            self.text_state.cached_raster_key.clear();
                        }
                    }
                });
        }

        ui.separator();
        // Font size ÔÇö merged DragValue + dropdown (same pattern as brush size widget)
        self.show_text_size_widget(ui, "ctx_text_size", assets);

        ui.separator();
        if ui
            .selectable_label(self.text_state.bold, t!("ctx.text.bold"))
            .clicked()
        {
            self.text_state.bold = !self.text_state.bold;
            self.text_state.loaded_font = None;
            self.text_state.preview_dirty = true;
            self.text_state.pending_ctx_style_update = Some(TextStyleUpdate::Bold {
                enabled: self.text_state.bold,
                base_weight: self.text_state.font_weight,
            });
            self.text_state.ctx_bar_style_dirty = true;
        }
        if ui
            .selectable_label(self.text_state.italic, t!("ctx.text.italic"))
            .clicked()
        {
            self.text_state.italic = !self.text_state.italic;
            self.text_state.loaded_font = None;
            self.text_state.preview_dirty = true;
            self.text_state.pending_ctx_style_update =
                Some(TextStyleUpdate::Italic(self.text_state.italic));
            self.text_state.ctx_bar_style_dirty = true;
        }
        if ui
            .selectable_label(self.text_state.underline, t!("ctx.text.underline"))
            .clicked()
        {
            self.text_state.underline = !self.text_state.underline;
            self.text_state.preview_dirty = true;
            self.text_state.pending_ctx_style_update =
                Some(TextStyleUpdate::Underline(self.text_state.underline));
            self.text_state.ctx_bar_style_dirty = true;
        }
        if ui
            .selectable_label(self.text_state.strikethrough, t!("ctx.text.strikethrough"))
            .clicked()
        {
            self.text_state.strikethrough = !self.text_state.strikethrough;
            self.text_state.preview_dirty = true;
            self.text_state.pending_ctx_style_update = Some(TextStyleUpdate::Strikethrough(
                self.text_state.strikethrough,
            ));
            self.text_state.ctx_bar_style_dirty = true;
        }

        ui.separator();
        // Alignment: single cycle-toggle button
        let align_label = self.text_state.alignment.label();
        if ui
            .add(egui::Button::new(align_label).min_size(egui::vec2(50.0, 0.0)))
            .clicked()
        {
            self.text_state.alignment = match self.text_state.alignment {
                crate::ops::text::TextAlignment::Left => crate::ops::text::TextAlignment::Center,
                crate::ops::text::TextAlignment::Center => crate::ops::text::TextAlignment::Right,
                crate::ops::text::TextAlignment::Right => crate::ops::text::TextAlignment::Left,
            };
            self.text_state.preview_dirty = true;
        }

        ui.separator();
        ui.label(t!("ctx.text.letter_spacing"));
        if ui
            .add(
                crate::ui::numeric::Numeric::new(&mut self.text_state.letter_spacing)
                    .speed(0.1)
                    .suffix("px"),
            )
            .changed()
        {
            self.text_state.preview_dirty = true;
            self.text_state.pending_ctx_style_update = Some(TextStyleUpdate::LetterSpacing(
                self.text_state.letter_spacing,
            ));
            self.text_state.ctx_bar_style_dirty = true;
        }

        ui.separator();
        ui.label("Letter Width");
        if ui
            .add(
                crate::ui::numeric::Numeric::new(&mut self.text_state.width_scale)
                    .speed(0.01)
                    .range(0.01..=10.0)
                    .suffix("x"),
            )
            .changed()
        {
            self.text_state.preview_dirty = true;
            self.text_state.pending_ctx_style_update =
                Some(TextStyleUpdate::WidthScale(self.text_state.width_scale));
            self.text_state.ctx_bar_style_dirty = true;
        }

        ui.label("Letter Height");
        if ui
            .add(
                crate::ui::numeric::Numeric::new(&mut self.text_state.height_scale)
                    .speed(0.01)
                    .range(0.01..=10.0)
                    .suffix("x"),
            )
            .changed()
        {
            self.text_state.preview_dirty = true;
            self.text_state.pending_ctx_style_update =
                Some(TextStyleUpdate::HeightScale(self.text_state.height_scale));
            self.text_state.ctx_bar_style_dirty = true;
        }

        ui.separator();
        ui.label(t!("ctx.text.line_spacing"));
        if ui
            .add(
                crate::ui::numeric::Numeric::new(&mut self.text_state.line_spacing)
                    .speed(0.01)
                    .range(0.5..=5.0)
                    .suffix("x"),
            )
            .changed()
        {
            self.text_state.preview_dirty = true;
        }

        ui.separator();
        // Anti-alias: compact "AA" toggle with tooltip
        let aa_resp = ui.selectable_label(self.text_state.anti_alias, t!("ctx.anti_alias"));
        if aa_resp.clicked() {
            self.text_state.anti_alias = !self.text_state.anti_alias;
            self.text_state.preview_dirty = true;
        }
        aa_resp.on_hover_text(t!("ctx.anti_alias_tooltip"));

        ui.separator();
        ui.label(t!("ctx.blend"));
        egui::ComboBox::from_id_salt("ctx_text_blend")
            .selected_text(self.properties.blending_mode.name())
            .width(80.0)
            .show_ui(ui, |ui| {
                for mode in BlendMode::all() {
                    if ui
                        .selectable_label(self.properties.blending_mode == *mode, mode.name())
                        .clicked()
                    {
                        self.properties.blending_mode = *mode;
                    }
                }
            });
    }
}
