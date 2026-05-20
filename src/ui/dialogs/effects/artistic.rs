impl InkDialog {
    pub fn show(&mut self, ctx: &egui::Context) -> DialogResult<(f32, f32)> {
        let mut result = DialogResult::Open;
        let colors = DialogColors::from_ctx(ctx);

        egui::Window::new("dialog_ink")
            .title_bar(false)
            .collapsible(false)
            .resizable(false)
            .default_pos(egui::pos2(ctx.content_rect().center().x - 175.0, 60.0))
            .show(ctx, |ui| {
                ui.set_min_width(350.0);
                if paint_dialog_header(ui, &colors, "\u{1F58B}", &t!("dialog.ink")) { result = DialogResult::Cancel; }
                ui.add_space(4.0);
                section_label(ui, &colors, "INK SETTINGS");

                let mut changed = false;
                egui::Grid::new("ink_params")
                    .num_columns(2)
                    .spacing([8.0, 6.0])
                    .show(ui, |ui| {
                        ui.label("Edge Strength");
                        let r = ui.add(
                            egui::Slider::new(&mut self.edge_strength, 10.0..=300.0)
                                .max_decimals(0),
                        );
                        if track_slider(&r, &mut self.dragging) {
                            changed = true;
                        }
                        ui.end_row();

                        ui.label("Threshold");
                        let r = ui.add(
                            egui::Slider::new(&mut self.threshold, 0.05..=1.0).max_decimals(2),
                        );
                        if track_slider(&r, &mut self.dragging) {
                            changed = true;
                        }
                        ui.end_row();
                    });

                accent_separator(ui, &colors);
                let manual = preview_controls(ui, &colors, &mut self.live_preview);
                if (changed && self.live_preview) || manual {
                    result = DialogResult::Changed;
                }

                let (ok, cancel) = dialog_footer(ui, &colors);
                if ok {
                    result = DialogResult::Ok((self.edge_strength, self.threshold));
                }
                if cancel {
                    result = DialogResult::Cancel;
                }
            });
        result
    }
}

// -------

effect_dialog_base!(OilPaintingDialog {
    radius: f32 = 0.0,
    levels: f32 = 20.0,
    first_open: bool = true
});

impl OilPaintingDialog {
    pub fn show(&mut self, ctx: &egui::Context) -> DialogResult<(u32, u32)> {
        let mut result = DialogResult::Open;
        let colors = DialogColors::from_ctx(ctx);

        egui::Window::new("dialog_oil_painting")
            .title_bar(false)
            .collapsible(false)
            .resizable(false)
            .default_pos(egui::pos2(ctx.content_rect().center().x - 175.0, 60.0))
            .show(ctx, |ui| {
                ui.set_min_width(360.0);
                if paint_dialog_header(ui, &colors, "\u{1F3A8}", &t!("dialog.oil_painting")) { result = DialogResult::Cancel; }
                ui.add_space(4.0);
                section_label(ui, &colors, "PAINTING SETTINGS");

                let mut changed = false;
                egui::Grid::new("oil_params")
                    .num_columns(2)
                    .spacing([8.0, 6.0])
                    .show(ui, |ui| {
                        ui.label("Brush Radius");
                        if numeric_field_with_buttons(
                            ui,
                            &mut self.radius,
                            0.5,
                            1.0..=10.0,
                            " px",
                            1.0,
                        ) {
                            changed = true;
                        }
                        ui.end_row();

                        ui.label("Intensity Levels");
                        let r =
                            ui.add(egui::Slider::new(&mut self.levels, 4.0..=64.0).max_decimals(0));
                        if track_slider(&r, &mut self.dragging) {
                            changed = true;
                        }
                        ui.end_row();
                    });

                accent_separator(ui, &colors);
                let manual = preview_controls(ui, &colors, &mut self.live_preview);
                if (changed && self.live_preview) || manual {
                    result = DialogResult::Changed;
                }

                let (ok, cancel) = dialog_footer(ui, &colors);
                if ok {
                    result = DialogResult::Ok((self.radius as u32, self.levels as u32));
                }
                if cancel {
                    result = DialogResult::Cancel;
                }
            });
        result
    }
}

// -------

effect_dialog_base!(ColorFilterDialog {
    color: [f32; 3] = [1.0, 0.8, 0.4],
    intensity: f32 = 0.0,
    mode_idx: usize = 0,
    first_open: bool = true
});

impl ColorFilterDialog {
    pub fn filter_mode(&self) -> ColorFilterMode {
        match self.mode_idx {
            1 => ColorFilterMode::Screen,
            2 => ColorFilterMode::Overlay,
            3 => ColorFilterMode::SoftLight,
            _ => ColorFilterMode::Multiply,
        }
    }

    pub fn show(&mut self, ctx: &egui::Context) -> DialogResult<([u8; 4], f32, ColorFilterMode)> {
        let mut result = DialogResult::Open;
        let colors = DialogColors::from_ctx(ctx);

        egui::Window::new("dialog_color_filter")
            .title_bar(false)
            .collapsible(false)
            .resizable(false)
            .default_pos(egui::pos2(ctx.content_rect().center().x - 175.0, 60.0))
            .show(ctx, |ui| {
                ui.set_min_width(380.0);
                if paint_dialog_header(ui, &colors, "\u{1F3AD}", &t!("dialog.color_filter")) { result = DialogResult::Cancel; }
                ui.add_space(4.0);
                section_label(ui, &colors, "FILTER SETTINGS");

                let mut changed = false;
                egui::Grid::new("cfilter_params")
                    .num_columns(2)
                    .spacing([8.0, 6.0])
                    .show(ui, |ui| {
                        ui.label("Color");
                        if ui.color_edit_button_rgb(&mut self.color).changed() {
                            changed = true;
                        }
                        ui.end_row();

                        ui.label("Intensity");
                        let r = ui
                            .add(egui::Slider::new(&mut self.intensity, 0.0..=1.0).max_decimals(2));
                        if track_slider(&r, &mut self.dragging) {
                            changed = true;
                        }
                        ui.end_row();

                        ui.label("Quick");
                        ui.horizontal(|ui| {
                            ui.spacing_mut().item_spacing.x = 3.0;
                            let presets: [(&str, [f32; 3]); 5] = [
                                ("Warm", [1.0, 0.85, 0.6]),
                                ("Cool", [0.6, 0.8, 1.0]),
                                ("Sepia", [0.94, 0.82, 0.63]),
                                ("Rose", [1.0, 0.7, 0.75]),
                                ("Cyan", [0.5, 0.95, 0.95]),
                            ];
                            for (label, c) in &presets {
                                let preview_col = Color32::from_rgb(
                                    (c[0] * 255.0) as u8,
                                    (c[1] * 255.0) as u8,
                                    (c[2] * 255.0) as u8,
                                );
                                let btn = egui::Button::new(
                                    egui::RichText::new(*label).size(10.5).color(
                                        if c[0] > 0.8 && c[1] > 0.8 && c[2] > 0.8 {
                                            Color32::BLACK
                                        } else {
                                            Color32::WHITE
                                        },
                                    ),
                                )
                                .fill(preview_col);
                                if ui.add(btn).clicked() {
                                    self.color = *c;
                                    changed = true;
                                }
                            }
                        });
                        ui.end_row();

                        ui.label("Blend Mode");
                        ui.horizontal(|ui| {
                            ui.spacing_mut().item_spacing.x = 4.0;
                            for (i, label) in ["Multiply", "Screen", "Overlay", "Soft Light"]
                                .iter()
                                .enumerate()
                            {
                                let btn = if self.mode_idx == i {
                                    egui::Button::new(
                                        egui::RichText::new(*label).strong().size(11.0),
                                    )
                                    .fill(colors.accent_faint)
                                } else {
                                    egui::Button::new(egui::RichText::new(*label).size(11.0))
                                };
                                if ui.add(btn).clicked() {
                                    self.mode_idx = i;
                                    changed = true;
                                }
                            }
                        });
                        ui.end_row();
                    });

                accent_separator(ui, &colors);
                let manual = preview_controls(ui, &colors, &mut self.live_preview);
                if (changed && self.live_preview) || manual {
                    result = DialogResult::Changed;
                }

                let (ok, cancel) = dialog_footer(ui, &colors);
                if ok {
                    let c = [
                        (self.color[0] * 255.0) as u8,
                        (self.color[1] * 255.0) as u8,
                        (self.color[2] * 255.0) as u8,
                        255,
                    ];
                    result = DialogResult::Ok((c, self.intensity, self.filter_mode()));
                }
                if cancel {
                    result = DialogResult::Cancel;
                }
            });
        result
    }
}

effect_dialog_base!(ColorToAlphaDialog {
    target_color: [f32; 3] = [1.0, 0.0, 0.0],
    tolerance: f32 = 18.0,
    softness: f32 = 35.0,
    strength: f32 = 1.0,
    spill_suppression: f32 = 0.35,
    alpha_floor: f32 = 0.0,
    alpha_ceiling: f32 = 1.0,
    protect_luminance: f32 = 0.15,
    sample_x: u32 = 0,
    sample_y: u32 = 0,
    first_open: bool = true
});

impl ColorToAlphaDialog {
    pub fn new_with_target(state: &CanvasState, target: Color32) -> Self {
        let mut dlg = Self::new(state);
        dlg.target_color = [
            target.r() as f32 / 255.0,
            target.g() as f32 / 255.0,
            target.b() as f32 / 255.0,
        ];
        dlg
    }

    pub fn settings(&self) -> crate::ops::color_removal::ColorToAlphaSettings {
        crate::ops::color_removal::ColorToAlphaSettings {
            target: [
                (self.target_color[0] * 255.0).round().clamp(0.0, 255.0) as u8,
                (self.target_color[1] * 255.0).round().clamp(0.0, 255.0) as u8,
                (self.target_color[2] * 255.0).round().clamp(0.0, 255.0) as u8,
            ],
            tolerance: self.tolerance,
            softness: self.softness,
            strength: self.strength,
            spill_suppression: self.spill_suppression,
            alpha_floor: self.alpha_floor,
            alpha_ceiling: self.alpha_ceiling,
            protect_luminance: self.protect_luminance,
        }
    }

    fn set_preset(&mut self, color: [f32; 3]) {
        self.target_color = color;
        self.tolerance = 18.0;
        self.softness = 35.0;
        self.strength = 1.0;
        self.spill_suppression = 0.35;
        self.alpha_floor = 0.0;
        self.alpha_ceiling = 1.0;
        self.protect_luminance = 0.15;
    }

    pub fn show(
        &mut self,
        ctx: &egui::Context,
        icon_texture: Option<&egui::TextureHandle>,
    ) -> DialogResult<crate::ops::color_removal::ColorToAlphaSettings> {
        let mut result = DialogResult::Open;
        let colors = DialogColors::from_ctx(ctx);

        egui::Window::new("dialog_color_to_alpha")
            .title_bar(false)
            .collapsible(false)
            .resizable(false)
            .default_pos(egui::pos2(ctx.content_rect().center().x - 200.0, 60.0))
            .show(ctx, |ui| {
                ui.set_min_width(420.0);
                let close_clicked = if icon_texture.is_some() {
                    paint_dialog_header_with_texture(
                        ui,
                        &colors,
                        icon_texture,
                        &t!("dialog.color_to_alpha"),
                    )
                } else {
                    paint_dialog_header(
                        ui,
                        &colors,
                        crate::assets::Icon::ColorRemover.emoji(),
                        &t!("dialog.color_to_alpha"),
                    )
                };
                if close_clicked {
                    result = DialogResult::Cancel;
                }
                ui.add_space(4.0);
                section_label(ui, &colors, "TARGET COLOR");

                let mut changed = false;
                egui::Grid::new("color_to_alpha_target")
                    .num_columns(2)
                    .spacing([8.0, 6.0])
                    .show(ui, |ui| {
                        ui.label("Color");
                        if ui.color_edit_button_rgb(&mut self.target_color).changed() {
                            changed = true;
                        }
                        ui.end_row();

                        ui.label("Sample");
                        ui.horizontal(|ui| {
                            let max_x = self
                                .original_flat
                                .as_ref()
                                .map_or(0, |img| img.width().saturating_sub(1));
                            let max_y = self
                                .original_flat
                                .as_ref()
                                .map_or(0, |img| img.height().saturating_sub(1));
                            ui.add(egui::DragValue::new(&mut self.sample_x).range(0..=max_x));
                            ui.label("x");
                            ui.add(egui::DragValue::new(&mut self.sample_y).range(0..=max_y));
                            if ui.small_button("Pick").clicked()
                                && let Some(flat) = &self.original_flat
                            {
                                let x = self.sample_x.min(flat.width().saturating_sub(1));
                                let y = self.sample_y.min(flat.height().saturating_sub(1));
                                let p = flat.get_pixel(x, y);
                                self.target_color = [
                                    p[0] as f32 / 255.0,
                                    p[1] as f32 / 255.0,
                                    p[2] as f32 / 255.0,
                                ];
                                changed = true;
                            }
                        });
                        ui.end_row();

                        ui.label("Reset");
                        if ui.small_button("\u{21BA}").clicked() {
                            self.set_preset([1.0, 0.0, 0.0]);
                            changed = true;
                        }
                        ui.end_row();
                    });

                ui.add_space(4.0);
                section_label(ui, &colors, "REMOVAL SETTINGS");
                egui::Grid::new("color_to_alpha_params")
                    .num_columns(2)
                    .spacing([8.0, 6.0])
                    .show(ui, |ui| {
                        ui.label("Tolerance");
                        if dialog_slider(ui, &mut self.tolerance, 0.0..=128.0, 1.0, "", 0) {
                            changed = true;
                        }
                        ui.end_row();

                        ui.label("Softness");
                        if dialog_slider(ui, &mut self.softness, 0.0..=255.0, 1.0, "", 0) {
                            changed = true;
                        }
                        ui.end_row();

                        ui.label("Strength");
                        if dialog_slider(ui, &mut self.strength, 0.0..=1.0, 0.01, "", 2) {
                            changed = true;
                        }
                        ui.end_row();

                        ui.label("Spill Suppression");
                        if dialog_slider(
                            ui,
                            &mut self.spill_suppression,
                            0.0..=1.0,
                            0.01,
                            "",
                            2,
                        ) {
                            changed = true;
                        }
                        ui.end_row();

                        ui.label("Alpha Floor");
                        if dialog_slider(ui, &mut self.alpha_floor, 0.0..=1.0, 0.01, "", 2) {
                            self.alpha_ceiling = self.alpha_ceiling.max(self.alpha_floor);
                            changed = true;
                        }
                        ui.end_row();

                        ui.label("Alpha Ceiling");
                        if dialog_slider(ui, &mut self.alpha_ceiling, 0.0..=1.0, 0.01, "", 2) {
                            self.alpha_floor = self.alpha_floor.min(self.alpha_ceiling);
                            changed = true;
                        }
                        ui.end_row();

                        ui.label("Protect Luminance");
                        if dialog_slider(
                            ui,
                            &mut self.protect_luminance,
                            0.0..=1.0,
                            0.01,
                            "",
                            2,
                        ) {
                            changed = true;
                        }
                        ui.end_row();
                    });

                accent_separator(ui, &colors);
                let manual = preview_controls(ui, &colors, &mut self.live_preview);
                if (changed && self.live_preview) || manual {
                    result = DialogResult::Changed;
                }

                let (ok, cancel) = dialog_footer(ui, &colors);
                if ok {
                    result = DialogResult::Ok(self.settings());
                }
                if cancel {
                    result = DialogResult::Cancel;
                }
            });
        result
    }
}

// ============================================================================
// RENDER — CONTOURS DIALOG
// ============================================================================

effect_dialog_base!(ContoursDialog {
    scale: f32 = 30.0,
    frequency: f32 = 8.0,
    line_width: f32 = 1.5,
    line_color: [f32; 3] = [0.0, 0.0, 0.0],
    seed: u32 = 42,
    octaves: f32 = 3.0,
    blend: f32 = 0.0,
    first_open: bool = true
});

impl ContoursDialog {
    pub fn show(
        &mut self,
        ctx: &egui::Context,
    ) -> DialogResult<(f32, f32, f32, [u8; 4], u32, u32, f32)> {
        let mut result = DialogResult::Open;
        let colors = DialogColors::from_ctx(ctx);

        egui::Window::new("dialog_contours")
            .title_bar(false)
            .collapsible(false)
            .resizable(false)
            .default_pos(egui::pos2(ctx.content_rect().center().x - 190.0, 60.0))
            .show(ctx, |ui| {
                ui.set_min_width(400.0);
                if paint_dialog_header(ui, &colors, "\u{1F5FA}", &t!("dialog.contours")) { result = DialogResult::Cancel; }
                ui.add_space(4.0);
                section_label(ui, &colors, "CONTOUR SETTINGS");

                let mut changed = false;
                egui::Grid::new("contour_params")
                    .num_columns(2)
                    .spacing([8.0, 6.0])
                    .show(ui, |ui| {
                        ui.label("Scale");
                        if dialog_slider(ui, &mut self.scale, 5.0..=400.0, 1.0, " px", 0) {
                            changed = true;
                        }
                        ui.end_row();
                        ui.label("");
                        ui.label(
                            egui::RichText::new("Size of the noise pattern")
                                .size(10.0)
                                .color(colors.text_muted),
                        );
                        ui.end_row();

                        ui.label("Frequency");
                        if dialog_slider(ui, &mut self.frequency, 1.0..=30.0, 0.1, "", 1) {
                            changed = true;
                        }
                        ui.end_row();
                        ui.label("");
                        ui.label(
                            egui::RichText::new("Number of contour levels")
                                .size(10.0)
                                .color(colors.text_muted),
                        );
                        ui.end_row();

                        ui.label("Line Width");
                        if dialog_slider(ui, &mut self.line_width, 0.5..=8.0, 0.1, " px", 1) {
                            changed = true;
                        }
                        ui.end_row();

                        ui.label("Line Color");
                        ui.horizontal(|ui| {
                            let mut c32 = Color32::from_rgb(
                                (self.line_color[0] * 255.0) as u8,
                                (self.line_color[1] * 255.0) as u8,
                                (self.line_color[2] * 255.0) as u8,
                            );
                            if ui.color_edit_button_srgba(&mut c32).changed() {
                                self.line_color = [
                                    c32.r() as f32 / 255.0,
                                    c32.g() as f32 / 255.0,
                                    c32.b() as f32 / 255.0,
                                ];
                                changed = true;
                            }
                            ui.spacing_mut().item_spacing.x = 3.0;
                            if ui.small_button("Black").clicked() {
                                self.line_color = [0.0, 0.0, 0.0];
                                changed = true;
                            }
                            if ui.small_button("White").clicked() {
                                self.line_color = [1.0, 1.0, 1.0];
                                changed = true;
                            }
                            if ui.small_button("Brown").clicked() {
                                self.line_color = [0.55, 0.35, 0.17];
                                changed = true;
                            }
                            if ui.small_button("Blue").clicked() {
                                self.line_color = [0.15, 0.35, 0.7];
                                changed = true;
                            }
                        });
                        ui.end_row();

                        ui.label("Blend");
                        if dialog_slider(ui, &mut self.blend, 0.0..=1.0, 0.01, "", 2) {
                            changed = true;
                        }
                        ui.end_row();
                    });

                ui.add_space(4.0);
                section_label(ui, &colors, "NOISE FIELD");

                egui::Grid::new("contour_noise")
                    .num_columns(2)
                    .spacing([8.0, 6.0])
                    .show(ui, |ui| {
                        ui.label("Octaves");
                        if dialog_slider(ui, &mut self.octaves, 1.0..=6.0, 1.0, "", 0) {
                            self.octaves = self.octaves.round().clamp(1.0, 6.0);
                            changed = true;
                        }
                        ui.end_row();

                        ui.label("Seed");
                        ui.horizontal(|ui| {
                            let mut seed_f = self.seed as f32;
                            if ui
                                .add(
                                    egui::DragValue::new(&mut seed_f)
                                        .speed(1.0)
                                        .range(0.0..=9999.0),
                                )
                                .changed()
                            {
                                self.seed = seed_f as u32;
                                changed = true;
                            }
                            if ui.small_button("\u{1F3B2}").clicked() {
                                self.seed =
                                    (self.seed.wrapping_mul(1103515245).wrapping_add(12345))
                                        % 10000;
                                changed = true;
                            }
                        });
                        ui.end_row();
                    });

                ui.add_space(4.0);
                section_label(ui, &colors, "PRESETS");
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 4.0;
                    if ui
                        .button(egui::RichText::new("Topo Map").size(11.0))
                        .clicked()
                    {
                        self.scale = 40.0;
                        self.frequency = 10.0;
                        self.line_width = 1.0;
                        self.line_color = [0.55, 0.35, 0.17];
                        self.octaves = 4.0;
                        self.blend = 0.8;
                        changed = true;
                    }
                    if ui
                        .button(egui::RichText::new("Fine Lines").size(11.0))
                        .clicked()
                    {
                        self.scale = 15.0;
                        self.frequency = 20.0;
                        self.line_width = 0.5;
                        self.line_color = [0.0, 0.0, 0.0];
                        self.octaves = 2.0;
                        self.blend = 0.5;
                        changed = true;
                    }
                    if ui.button(egui::RichText::new("Bold").size(11.0)).clicked() {
                        self.scale = 60.0;
                        self.frequency = 5.0;
                        self.line_width = 4.0;
                        self.line_color = [0.0, 0.0, 0.0];
                        self.octaves = 3.0;
                        self.blend = 1.0;
                        changed = true;
                    }
                    if ui.button(egui::RichText::new("Ocean").size(11.0)).clicked() {
                        self.scale = 50.0;
                        self.frequency = 12.0;
                        self.line_width = 1.5;
                        self.line_color = [0.15, 0.35, 0.7];
                        self.octaves = 5.0;
                        self.blend = 0.7;
                        changed = true;
                    }
                });

                accent_separator(ui, &colors);
                let manual = preview_controls(ui, &colors, &mut self.live_preview);
                if (changed && self.live_preview) || manual {
                    result = DialogResult::Changed;
                }

                let (ok, cancel) = dialog_footer(ui, &colors);
                if ok {
                    let c = [
                        (self.line_color[0] * 255.0) as u8,
                        (self.line_color[1] * 255.0) as u8,
                        (self.line_color[2] * 255.0) as u8,
                        255,
                    ];
                    result = DialogResult::Ok((
                        self.scale,
                        self.frequency,
                        self.line_width,
                        c,
                        self.seed,
                        self.octaves as u32,
                        self.blend,
                    ));
                }
                if cancel {
                    result = DialogResult::Cancel;
                }
            });
        result
    }
}


