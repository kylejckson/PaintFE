//! Responsive Save As presentation built on the shared export model.
use super::*;
use crate::ops::dialogs::{DialogColors, accent_separator, section_label};

impl SaveFileDialog {
    fn show_preview_status(&self, ui: &mut egui::Ui) {
        let height = ui.text_style_height(&egui::TextStyle::Body) * 2.0 + 6.0;
        ui.allocate_ui(egui::vec2(ui.available_width(), height), |ui| {
            if self.anim_playing && self.animated {
                ui.label(crate::t!("ui.animation"));
                ui.weak(crate::t!("export.preview_estimate_note"));
                return;
            }
            if let Some(error) = &self.preview_error {
                ui.add(
                    egui::Label::new(
                        egui::RichText::new(crate::t!("export.preview_failed"))
                            .color(ui.visuals().error_fg_color),
                    )
                    .truncate(),
                )
                .on_hover_text(error);
            } else if self.preview_completed != Some(self.preview_key()) {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label(crate::t!("export.updating"));
                });
            } else {
                ui.label(crate::t!("export.preview_ready"));
            }
            if self.preview_completed == Some(self.preview_key())
                && self.preview_error.is_none()
                && self.format != SaveFormat::Pfe
                && !self.animated
            {
                ui.weak(crate::i18n::format_translation(
                    "export.estimated_size",
                    &[("0", Self::format_file_size(self.estimated_full_size))],
                ));
            } else {
                ui.weak(crate::t!("export.preview_estimate_note"));
            }
        });
    }

    fn show_output_description(&self, ui: &mut egui::Ui) {
        use crate::io::export::png::{Depth, Mode};
        if self.format == SaveFormat::Pfe {
            ui.weak(crate::t!("export.editable_document"));
            return;
        }
        let mut bits = if matches!(self.format, SaveFormat::Jxl(_) | SaveFormat::Tiff)
            && self.source_depth == 16
        {
            16
        } else {
            8
        };
        if self.format == SaveFormat::Tiff && self.source_depth == 32 {
            bits = 32;
        }
        let mut alpha = !matches!(self.format, SaveFormat::Jpeg);
        if self.format == SaveFormat::Png && !self.animated {
            bits = if self.options.png.mode != Mode::Indexed
                && (self.options.png.depth == Depth::Sixteen
                    || (self.options.png.depth == Depth::Automatic && self.source_depth == 16))
            {
                16
            } else {
                8
            };
            alpha = match self.options.png.mode {
                Mode::Rgb | Mode::Gray => false,
                Mode::Automatic => self.source_has_alpha,
                Mode::Indexed => self.options.png.preserve_palette_alpha,
                _ => true,
            };
        }
        let alpha_label = if alpha {
            crate::t!("export.with_alpha")
        } else {
            crate::t!("export.without_alpha")
        };
        ui.weak(crate::i18n::format_translation(
            "export.actual_output",
            &[("0", bits.to_string()), ("1", alpha_label)],
        ));
    }

    fn show_png_options(&mut self, ui: &mut egui::Ui) {
        use crate::io::export::png::{Compression, Depth, Mode};
        if self.format != SaveFormat::Png {
            return;
        }
        ui.separator();
        if self.animated {
            // APNG currently writes RGBA8 frames. Do not display controls it cannot honour.
            ui.weak(crate::t!("export.apng_options"));
            return;
        }
        fn combo<T: Copy + PartialEq>(
            ui: &mut egui::Ui,
            id: &str,
            value: &mut T,
            items: &[(T, &str)],
        ) {
            let selected = items.iter().find(|(v, _)| v == value).unwrap().1;
            egui::ComboBox::from_id_salt(id)
                .wrap_mode(egui::TextWrapMode::Truncate)
                .width((ui.available_width() - 12.0).max(100.0))
                .selected_text(crate::i18n::translate(selected))
                .show_ui(ui, |ui| {
                    for &(option, label) in items {
                        ui.selectable_value(value, option, crate::i18n::translate(label));
                    }
                });
        }
        ui.label(crate::t!("export.png_mode"));
        combo(
            ui,
            "save_png_mode",
            &mut self.options.png.mode,
            &[
                (Mode::Automatic, "export.automatic"),
                (Mode::Rgba, "export.rgba"),
                (Mode::Rgb, "export.rgb"),
                (Mode::Gray, "export.gray"),
                (Mode::GrayAlpha, "export.gray_alpha"),
                (Mode::Indexed, "export.indexed"),
            ],
        );
        if self.options.png.mode == Mode::Indexed {
            ui.label(crate::t!("ui.colors"));
            ui.add(
                crate::ui::controls::Slider::new(&mut self.options.png.palette_colors, 2..=256)
                    .id_salt("save_png_colors")
                    .reset_to(256),
            );
            ui.checkbox(&mut self.options.png.dither, crate::t!("ui.dither"));
            ui.checkbox(
                &mut self.options.png.preserve_palette_alpha,
                crate::t!("export.keep_alpha"),
            );
            ui.weak(crate::t!("export.indexed_note"));
        } else {
            ui.label(crate::t!("export.png_depth"));
            combo(
                ui,
                "save_png_depth",
                &mut self.options.png.depth,
                &[
                    (Depth::Automatic, "export.automatic"),
                    (Depth::Eight, "export.depth_8"),
                    (Depth::Sixteen, "export.depth_16"),
                ],
            );
        }
        ui.label(crate::t!("ui.compression"));
        combo(
            ui,
            "save_png_compression",
            &mut self.options.png.compression,
            &[
                (Compression::Fast, "export.fast"),
                (Compression::Balanced, "export.balanced"),
                (Compression::Smaller, "export.smaller"),
            ],
        );
        ui.weak(crate::t!("export.compression_note"));
        if self.source_has_color && matches!(self.options.png.mode, Mode::Gray | Mode::GrayAlpha) {
            ui.weak(crate::t!("export.loss_color"));
        }
    }
    pub(super) fn show_export_body(
        &mut self,
        ui: &mut egui::Ui,
        ctx: &egui::Context,
        colors: &DialogColors,
        show_anim_controls: bool,
    ) {
        let width = ui.available_width();
        let height = (ui.available_height() - 64.0).max(96.0);
        let settings_width = 270.0 * (ui.text_style_height(&egui::TextStyle::Body) / 14.0).max(1.0);
        let wide = width >= settings_width + 280.0;
        ui.allocate_ui_with_layout(
            egui::vec2(width, height),
            egui::Layout::top_down(egui::Align::Min),
            |ui| {
                if wide {
                    ui.horizontal_top(|ui| {
                        let preview_width = (width - settings_width - 24.0).max(200.0);
                        ui.allocate_ui_with_layout(
                            egui::vec2(preview_width, height),
                            egui::Layout::top_down(egui::Align::Min),
                            |ui| {
                                self.show_export_preview(
                                    ui,
                                    ctx,
                                    colors,
                                    show_anim_controls,
                                    egui::vec2(
                                        preview_width,
                                        (height
                                            - (ui.text_style_height(&egui::TextStyle::Body) * 6.0
                                                + 34.0))
                                            .max(48.0),
                                    ),
                                );
                            },
                        );
                        ui.separator();
                        ui.allocate_ui_with_layout(
                            egui::vec2(settings_width, height),
                            egui::Layout::top_down(egui::Align::Min),
                            |ui| {
                                egui::ScrollArea::vertical()
                                    .id_salt("save_as_settings")
                                    .max_height(height)
                                    .auto_shrink([false, false])
                                    .show(ui, |ui| {
                                        ui.set_width(settings_width - 12.0);
                                        crate::ui::controls::without_wheel_adjustment(ui, |ui| {
                                            self.show_export_settings(ui, ctx, colors)
                                        });
                                    });
                            },
                        );
                    });
                } else {
                    egui::ScrollArea::vertical()
                        .id_salt("save_as_stacked")
                        .max_height(height)
                        .auto_shrink([false, false])
                        .show(ui, |ui| {
                            ui.set_width(width);
                            self.show_export_preview(
                                ui,
                                ctx,
                                colors,
                                show_anim_controls,
                                egui::vec2(
                                    (width - 12.0).max(80.0),
                                    (height * 0.4).clamp(80.0, 220.0),
                                ),
                            );
                            accent_separator(ui, colors);
                            crate::ui::controls::without_wheel_adjustment(ui, |ui| {
                                self.show_export_settings(ui, ctx, colors)
                            });
                        });
                }
            },
        );
    }
    fn show_export_preview(
        &mut self,
        ui: &mut egui::Ui,
        ctx: &egui::Context,
        colors: &DialogColors,
        show_anim_controls: bool,
        size: egui::Vec2,
    ) {
        let preview_size = size;
        let (rect, response) = ui.allocate_exact_size(preview_size, egui::Sense::click_and_drag());

        // Choose texture filter based on zoom level
        let use_nearest = self.preview_zoom > 2.0;
        let tex_opts = if use_nearest {
            TextureOptions::NEAREST
        } else {
            TextureOptions::LINEAR
        };

        // Reload textures if filter changed
        if use_nearest != self.preview_texture_is_nearest {
            if let (Some(texture), Some(encoded)) =
                (self.preview_texture.as_mut(), self.encoded_preview.as_ref())
            {
                texture.set(encoded.clone(), tex_opts);
            }
            for (texture, thumbnail) in self.frame_textures.iter_mut().zip(&self.frame_thumbnails) {
                if let Some(texture) = texture {
                    texture.set(rgba_to_color_image(thumbnail), tex_opts);
                }
            }
            self.preview_texture_is_nearest = use_nearest;
        }

        // Load still preview texture on demand
        if !show_anim_controls
            && self.preview_texture.is_none()
            && let Some(thumb) = &self.source_thumbnail
        {
            let color_image = rgba_to_color_image(thumb);
            self.preview_texture = Some(ctx.load_texture("save_preview", color_image, tex_opts));
        }

        // Determine which texture to show
        let display_texture = if show_anim_controls
            && (self.anim_playing
                || self.preview_texture.is_none()
                || self.preview_completed != Some(self.preview_key()))
        {
            let idx = self
                .anim_current_frame
                .min(self.frame_thumbnails.len().saturating_sub(1));
            if self.frame_textures[idx].is_none() {
                let color_image = rgba_to_color_image(&self.frame_thumbnails[idx]);
                self.frame_textures[idx] =
                    Some(ctx.load_texture(format!("frame_preview_{}", idx), color_image, tex_opts));
            }
            self.frame_textures[idx].as_ref()
        } else {
            self.preview_texture.as_ref()
        };

        // Auto-fit: compute initial zoom when zoom == 0.0
        if let Some(texture) = display_texture {
            let tex_size = texture.size_vec2();
            if self.preview_zoom == 0.0 {
                let fit = (preview_size.x / tex_size.x).min(preview_size.y / tex_size.y);
                self.preview_zoom = if fit < 1.0 { fit } else { fit.min(32.0) };
                let vis_w = preview_size.x / self.preview_zoom;
                let vis_h = preview_size.y / self.preview_zoom;
                self.preview_pan =
                    egui::Vec2::new((tex_size.x - vis_w) / 2.0, (tex_size.y - vis_h) / 2.0);
            }
        }

        // Drag to pan
        if response.dragged() {
            self.preview_pan -= response.drag_delta() / self.preview_zoom;
        }

        // Scroll to zoom (centered on cursor)
        let scroll_dy = ctx.input(|i| i.smooth_scroll_delta.y);
        let hover_pos = ctx.input(|i| i.pointer.hover_pos());
        if rect.contains(hover_pos.unwrap_or(egui::Pos2::ZERO)) && scroll_dy != 0.0 {
            let old_zoom = self.preview_zoom;
            let factor = (scroll_dy * 0.002).exp();
            self.preview_zoom = (self.preview_zoom * factor).clamp(0.05, 64.0);
            if let Some(cursor) = hover_pos {
                let off = cursor - rect.min;
                let src_x = self.preview_pan.x + off.x / old_zoom;
                let src_y = self.preview_pan.y + off.y / old_zoom;
                self.preview_pan = egui::Vec2::new(
                    src_x - off.x / self.preview_zoom,
                    src_y - off.y / self.preview_zoom,
                );
            }
            ctx.request_repaint();
        }

        // Double-click: reset to auto-fit
        if response.double_clicked() {
            if let Some(texture) = display_texture {
                let size = texture.size_vec2();
                self.preview_zoom = (preview_size.x / size.x)
                    .min(preview_size.y / size.y)
                    .clamp(0.05, 32.0);
            }
            self.preview_pan = egui::Vec2::ZERO;
        }

        // Clamp pan to valid range
        if let Some(texture) = display_texture {
            let ts = texture.size_vec2();
            let vis_w = preview_size.x / self.preview_zoom;
            let vis_h = preview_size.y / self.preview_zoom;
            self.preview_pan.x = self
                .preview_pan
                .x
                .clamp((0.0_f32).min(ts.x - vis_w), (ts.x - vis_w).max(0.0));
            self.preview_pan.y = self
                .preview_pan
                .y
                .clamp((0.0_f32).min(ts.y - vis_h), (ts.y - vis_h).max(0.0));
        }

        // Checkerboard background
        let painter = ui.painter_at(rect);
        let grid_size = 8.0;
        let light = Color32::from_gray(200);
        let dark = Color32::from_gray(160);
        for row in 0..((rect.height() / grid_size).ceil() as i32) {
            for col in 0..((rect.width() / grid_size).ceil() as i32) {
                let color = if (col + row) % 2 == 0 { light } else { dark };
                let cell_rect = egui::Rect::from_min_size(
                    rect.min + egui::vec2(col as f32 * grid_size, row as f32 * grid_size),
                    egui::vec2(grid_size, grid_size),
                )
                .intersect(rect);
                painter.rect_filled(cell_rect, 0.0, color);
            }
        }

        // Draw only the texture-backed portion of the viewport. At
        // auto-fit, one source axis extends beyond the image to
        // create letterboxing; drawing that full UV range would make
        // clamp-to-edge repeat the image's border pixels.
        if let Some(texture) = display_texture {
            let ts = texture.size_vec2();
            let vis_w = preview_size.x / self.preview_zoom;
            let vis_h = preview_size.y / self.preview_zoom;
            if let Some((destination, uv)) = clipped_preview_rects(
                rect,
                ts,
                egui::pos2(self.preview_pan.x, self.preview_pan.y),
                egui::vec2(vis_w, vis_h),
            ) {
                painter.image(texture.id(), destination, uv, Color32::WHITE);
            }

            // Thin scrollbar indicators
            let bar_color = Color32::from_rgba_premultiplied(120, 120, 120, 160);
            let bar_w = 4.0;
            if vis_w < ts.x {
                let ratio = self.preview_pan.x / (ts.x - vis_w).max(1.0);
                let bar_len = (preview_size.x * (vis_w / ts.x)).max(16.0);
                let bar_x = rect.min.x + ratio * (preview_size.x - bar_len);
                painter.rect_filled(
                    egui::Rect::from_min_size(
                        egui::pos2(bar_x, rect.max.y - bar_w - 1.0),
                        egui::vec2(bar_len, bar_w),
                    ),
                    2.0,
                    bar_color,
                );
            }
            if vis_h < ts.y {
                let ratio = self.preview_pan.y / (ts.y - vis_h).max(1.0);
                let bar_len = (preview_size.y * (vis_h / ts.y)).max(16.0);
                let bar_y = rect.min.y + ratio * (preview_size.y - bar_len);
                painter.rect_filled(
                    egui::Rect::from_min_size(
                        egui::pos2(rect.max.x - bar_w - 1.0, bar_y),
                        egui::vec2(bar_w, bar_len),
                    ),
                    2.0,
                    bar_color,
                );
            }
        }

        // Zoom controls strip
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            if ui.small_button("\u{2212}").clicked() {
                self.preview_zoom = (self.preview_zoom / 1.25).max(0.05);
            }
            let mut zoom_pct = (self.preview_zoom * 100.0).round();
            let zoom_response = ui.add(
                crate::ui::controls::Numeric::new(&mut zoom_pct)
                    .range(5.0..=6400.0)
                    .speed(1.0)
                    .suffix("%"),
            );
            #[cfg(test)]
            ctx.data_mut(|d| d.insert_temp(egui::Id::new("save_as_zoom_rect"), zoom_response.rect));
            if zoom_response.changed() {
                self.preview_zoom = zoom_pct / 100.0;
                if let Some(texture) = display_texture {
                    let ts = texture.size_vec2();
                    let vis_w = preview_size.x / self.preview_zoom;
                    let vis_h = preview_size.y / self.preview_zoom;
                    self.preview_pan = egui::Vec2::new((ts.x - vis_w) / 2.0, (ts.y - vis_h) / 2.0);
                }
            }
            if ui.small_button("\u{2B}").clicked() {
                self.preview_zoom = (self.preview_zoom * 1.25).min(64.0);
            }
            if ui.small_button(crate::t!("ui.fit")).clicked() {
                self.preview_zoom = 0.0;
                self.preview_pan = egui::Vec2::ZERO;
                ctx.request_repaint();
            }
        });

        // Animation playback controls
        if show_anim_controls {
            ui.add_space(2.0);
            ui.horizontal(|ui| {
                if ui.small_button("\u{23EE}").clicked() {
                    self.anim_current_frame = 0;
                    self.anim_playing = false;
                }
                if ui.small_button("\u{23EA}").clicked() {
                    if self.anim_current_frame > 0 {
                        self.anim_current_frame -= 1;
                    } else {
                        self.anim_current_frame = self.frame_thumbnails.len() - 1;
                    }
                    self.anim_playing = false;
                }
                let play_label = if self.anim_playing {
                    "\u{23F8}"
                } else {
                    "\u{25B6}"
                };
                if ui.small_button(play_label).clicked() {
                    self.anim_playing = !self.anim_playing;
                    if self.anim_playing {
                        self.anim_last_frame_time = ctx.input(|i| i.time);
                    }
                }
                if ui.small_button("\u{23E9}").clicked() {
                    self.anim_current_frame =
                        (self.anim_current_frame + 1) % self.frame_thumbnails.len();
                    self.anim_playing = false;
                }
                if ui.small_button("\u{23ED}").clicked() {
                    self.anim_current_frame = self.frame_thumbnails.len() - 1;
                    self.anim_playing = false;
                }
            });
            ui.small(crate::i18n::format_translation(
                "ui.frame_0_1",
                &[
                    ("0", format!("{}", self.anim_current_frame + 1)),
                    ("1", format!("{}", self.frame_thumbnails.len())),
                ],
            ));
        }
        self.show_preview_status(ui);

        let dim_text = format!("{}×{}", self.source_dimensions.0, self.source_dimensions.1);
        ui.label(
            egui::RichText::new(dim_text)
                .size(11.0)
                .color(colors.text_muted),
        );
    }
    fn show_export_settings(
        &mut self,
        ui: &mut egui::Ui,
        ctx: &egui::Context,
        colors: &DialogColors,
    ) {
        // ── FORMAT ────────────────────────────────────────
        section_label(ui, colors, &crate::t!("ui.format"));
        egui::ComboBox::from_id_salt("format_combo")
            .wrap_mode(egui::TextWrapMode::Truncate)
            .width((ui.available_width() - 12.0).max(120.0))
            .selected_text(crate::i18n::translate_english(&self.format.label()))
            .show_ui(ui, |ui| {
                for format in SaveFormat::all() {
                    if ui
                        .selectable_label(
                            self.format.extension() == format.extension(),
                            crate::i18n::translate_english(&format.label()),
                        )
                        .clicked()
                        && self.format.extension() != format.extension()
                    {
                        self.format_options
                            .insert(self.format.extension().into(), self.options);
                        self.options = self
                            .format_options
                            .get(format.extension())
                            .copied()
                            .unwrap_or(crate::io::export::ExportOptions {
                                format: *format,
                                ..Default::default()
                            });
                    }
                }
            });

        #[cfg(not(target_arch = "wasm32"))]
        if matches!(self.format, SaveFormat::Jxl(_))
            && !crate::services::optional_packs::is_present(
                crate::services::optional_packs::Pack::Jxl,
            )
        {
            ui.label(crate::t!("packs.jxl_missing"));
            if ui.button(crate::t!("packs.install_jxl")).clicked() {
                crate::services::optional_packs::request_jxl();
            }
        }
        #[cfg(not(target_arch = "wasm32"))]
        if self.format == SaveFormat::Heic {
            ui.weak(crate::t!("heic.export_note"));
            ui.label(crate::io::heic::size_limit_message(
                self.source_dimensions.0,
                self.source_dimensions.1,
            ));
            ui.label(crate::io::heic::setup_message());
        }
        #[cfg(not(target_arch = "wasm32"))]
        if self.format == SaveFormat::Avif {
            ui.weak(crate::t!("avif.export_note"));
            if !crate::services::optional_packs::is_present(
                crate::services::optional_packs::Pack::Avif,
            ) {
                ui.label(crate::t!("packs.avif_missing"));
                if ui.button(crate::t!("packs.install_avif")).clicked() {
                    crate::services::optional_packs::request_avif();
                }
            }
        }
        self.show_png_options(ui);
        self.show_output_description(ui);
        if ui.button(crate::t!("export.reset")).clicked() {
            self.options = crate::io::export::ExportOptions {
                format: match self.format {
                    SaveFormat::Jxl(_) => SaveFormat::Jxl(Default::default()),
                    format => format,
                },
                ..Default::default()
            };
        }
        if let SaveFormat::Jxl(options) = &mut self.format {
            accent_separator(ui, colors);
            ui.checkbox(&mut options.lossless, crate::t!("ui.lossless"));
            ui.label(crate::t!("jxl.effort"));
            ui.add(crate::ui::controls::Slider::new(&mut options.effort, 1..=9).reset_to(5));
        }
        // ── QUALITY (JPEG / WebP) ─────────────────────────
        if self.format.supports_quality() {
            accent_separator(ui, colors);
            if self.format == SaveFormat::Webp {
                section_label(ui, colors, &crate::t!("ui.webp"));
                if ui
                    .checkbox(&mut self.webp_lossless, crate::t!("ui.lossless"))
                    .changed()
                {
                    self.needs_preview_update = true;
                    ctx.request_repaint();
                }
            }
            if !matches!(self.format, SaveFormat::Jxl(options) if options.lossless)
                && (self.format != SaveFormat::Webp || !self.webp_lossless)
            {
                section_label(ui, colors, &crate::t!("dialog.resize_image.quality"));
                let mut quality = self.quality as f32;
                if crate::ui::dialogs::core::numeric_field_with_buttons(
                    ui,
                    &mut quality,
                    1.0,
                    1.0..=100.0,
                    "%",
                    1.0,
                ) {
                    self.quality = quality.round() as u8;
                    ctx.request_repaint();
                }
            }
            if !matches!(self.format, SaveFormat::Jxl(options) if options.lossless)
                && (self.format != SaveFormat::Webp || !self.webp_lossless)
            {
                let hint = match self.quality {
                    1..=30 => "Very Low",
                    31..=50 => "Low",
                    51..=70 => "Medium",
                    71..=85 => "Good",
                    86..=95 => "High",
                    _ => "Maximum",
                };
                ui.label(
                    egui::RichText::new(hint)
                        .size(11.0)
                        .color(colors.text_muted),
                );
            } else {
                ui.label(
                    egui::RichText::new(crate::t!("ui.exact_pixels_larger_than_lossy"))
                        .size(11.0)
                        .color(colors.text_muted),
                );
            }
            if self.format == SaveFormat::Webp && self.animated {
                ui.label(
                    egui::RichText::new(crate::t!(
                        "ui.animated_webp_can_use_per_frame_layer_options"
                    ))
                    .size(11.0)
                    .color(colors.text_muted),
                );
            }
        }

        // ── COMPRESSION (TIFF) ────────────────────────────
        if self.format == SaveFormat::Tiff {
            accent_separator(ui, colors);
            section_label(ui, colors, &crate::t!("ui.compression"));
            egui::ComboBox::from_id_salt("tiff_compression_combo")
                .width(160.0)
                .selected_text(match self.tiff_compression {
                    TiffCompression::None => crate::t!("line_end.none"),
                    TiffCompression::Lzw => "LZW".to_string(),
                    TiffCompression::Deflate => "Deflate".to_string(),
                })
                .show_ui(ui, |ui| {
                    ui.selectable_value(
                        &mut self.tiff_compression,
                        TiffCompression::None,
                        crate::t!("line_end.none"),
                    );
                    ui.selectable_value(&mut self.tiff_compression, TiffCompression::Lzw, "LZW");
                    ui.selectable_value(
                        &mut self.tiff_compression,
                        TiffCompression::Deflate,
                        "Deflate",
                    );
                });
        }

        // ── ICO warning ───────────────────────────────────
        if self.format == SaveFormat::Ico {
            accent_separator(ui, colors);
            ui.add_space(4.0);
            let (w, h) = self.source_dimensions;
            if w > 256 || h > 256 {
                ui.colored_label(
                    Color32::from_rgb(255, 180, 50),
                    crate::i18n::format_translation(
                        "ui.0_1_exceeds_256_256_and_will_be_scaled_down",
                        &[("0", format!("{}", w)), ("1", format!("{}", h))],
                    ),
                );
            } else {
                ui.label(
                    egui::RichText::new(crate::t!("ui.ico_supports_up_to_256_256_per_entry"))
                        .size(11.0)
                        .color(colors.text_muted),
                );
            }
        }

        // ── ANIMATION (PNG / GIF with multiple layers) ────
        if self.format.supports_animation() && self.layer_count > 1 {
            accent_separator(ui, colors);
            section_label(ui, colors, &crate::t!("ui.animation"));
            ui.checkbox(&mut self.animated, crate::t!("ui.animated"));

            if self.animated {
                ui.add_space(4.0);
                egui::Grid::new("anim_options_grid")
                    .num_columns(2)
                    .min_col_width(60.0)
                    .spacing([8.0, 4.0])
                    .show(ui, |ui| {
                        ui.label(crate::t!("ui.fps"));
                        ui.add(
                            crate::ui::controls::Slider::new(&mut self.animation_fps, 1.0..=60.0)
                                .double_click_reset_to(10.0)
                                .step_by(1.0)
                                .suffix(" fps"),
                        );
                        ui.end_row();
                    });

                if self.layer_count > 1 {
                    ui.label(
                        egui::RichText::new(crate::i18n::format_translation(
                            "ui.0_frames_1_fps_2_s",
                            &[
                                ("0", format!("{}", self.layer_count)),
                                ("1", format!("{:.0}", self.animation_fps)),
                                (
                                    "2",
                                    format!("{:.1}", self.layer_count as f32 / self.animation_fps),
                                ),
                            ],
                        ))
                        .size(11.0)
                        .color(colors.text_muted),
                    );
                }
            }
        }
        if self.format == SaveFormat::Gif {
            ui.label(crate::t!("ui.colors"));
            ui.add(
                crate::ui::controls::Slider::new(&mut self.options.gif_colors, 2..=256)
                    .id_salt("save_gif_colors")
                    .double_click_reset_to(256)
                    .logarithmic(true),
            );

            ui.checkbox(&mut self.gif_dither, crate::t!("ui.dither"));
        }
    }
}
