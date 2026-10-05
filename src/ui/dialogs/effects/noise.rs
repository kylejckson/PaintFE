impl AddNoiseDialog {
    pub fn noise_type(&self) -> NoiseType {
        match self.noise_type_idx {
            0 => NoiseType::Uniform,
            1 => NoiseType::Gaussian,
            _ => NoiseType::Perlin,
        }
    }

    pub fn show(
        &mut self,
        ctx: &egui::Context,
    ) -> DialogResult<(f32, NoiseType, bool, u32, f32, u32)> {
        let mut result = DialogResult::Open;
        let colors = DialogColors::from_ctx(ctx);

        crate::ui::polish::window(ctx, "dialog_add_noise")
            .title_bar(false)
            .collapsible(false)
            .resizable(false)
            .default_pos(egui::pos2(ctx.content_rect().center().x - 185.0, 60.0))
            .show(ctx, |ui| {
                ui.set_min_width(380.0);
                if paint_dialog_header(ui, &colors, "\u{1F4A5}", &t!("dialog.add_noise")) {
                    result = DialogResult::Cancel;
                }
                ui.add_space(4.0);
                section_label(ui, &colors, "NOISE SETTINGS");

                let mut changed = false;
                egui::Grid::new("noise_params")
                    .num_columns(2)
                    .spacing([8.0, 6.0])
                    .show(ui, |ui| {
                        ui.label("Amount");
                        let r = ui.add(
                            egui::Slider::new(&mut self.amount, 1.0..=100.0)
                                .suffix("%")
                                .max_decimals(0),
                        );
                        if track_slider(&r, &mut self.dragging) {
                            changed = true;
                        }
                        ui.end_row();

                        ui.label("Type");
                        ui.horizontal(|ui| {
                            ui.spacing_mut().item_spacing.x = 4.0;
                            let types = ["Uniform", "Gaussian", "Perlin"];
                            for (i, label) in types.iter().enumerate() {
                                let btn = if self.noise_type_idx == i {
                                    egui::Button::new(
                                        egui::RichText::new(*label).strong().size(11.0),
                                    )
                                    .fill(colors.accent_faint)
                                } else {
                                    egui::Button::new(egui::RichText::new(*label).size(11.0))
                                };
                                if ui.add(btn).clicked() {
                                    self.noise_type_idx = i;
                                    changed = true;
                                }
                            }
                        });
                        ui.end_row();

                        ui.label("Color");
                        if ui.checkbox(&mut self.monochrome, "Monochrome").changed() {
                            changed = true;
                        }
                        ui.end_row();

                        ui.label("Seed");
                        ui.horizontal(|ui| {
                            let mut seed_f = self.seed as f32;
                            if ui
                                .add(
                                    crate::ui::numeric::Numeric::new(&mut seed_f)
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
                section_label(ui, &colors, "GRAIN CONTROL");

                egui::Grid::new("noise_grain")
                    .num_columns(2)
                    .spacing([8.0, 6.0])
                    .show(ui, |ui| {
                        ui.label("Scale");
                        let r = ui.add(
                            egui::Slider::new(&mut self.scale, 0.5..=100.0)
                                .suffix(" px")
                                .max_decimals(1)
                                .logarithmic(true)
                                .clamping(egui::SliderClamping::Never),
                        );
                        if track_slider(&r, &mut self.dragging) {
                            changed = true;
                        }
                        ui.end_row();
                        ui.label("");
                        ui.label(
                            egui::RichText::new("Size of each noise grain (1 = single pixel)")
                                .size(10.0)
                                .color(colors.text_muted),
                        );
                        ui.end_row();

                        // Octaves only for Perlin
                        if self.noise_type_idx == 2 {
                            ui.label("Octaves");
                            let r = ui.add(
                                egui::Slider::new(&mut self.octaves, 1.0..=8.0)
                                    .max_decimals(0)
                                    .integer(),
                            );
                            if track_slider(&r, &mut self.dragging) {
                                changed = true;
                            }
                            ui.end_row();
                            ui.label("");
                            ui.label(
                                egui::RichText::new("More octaves = finer detail layers")
                                    .size(10.0)
                                    .color(colors.text_muted),
                            );
                            ui.end_row();
                        }
                    });

                accent_separator(ui, &colors);
                let manual = preview_controls(ui, &colors, &mut self.live_preview);
                if (changed && self.live_preview) || manual {
                    result = DialogResult::Changed;
                }

                let (ok, cancel) = dialog_footer(ui, &colors);
                if ok {
                    result = DialogResult::Ok((
                        self.amount,
                        self.noise_type(),
                        self.monochrome,
                        self.seed,
                        self.scale,
                        self.octaves as u32,
                    ));
                }
                if cancel {
                    result = DialogResult::Cancel;
                }
            });
        result
    }
}

// -------

effect_dialog_base!(ReduceNoiseDialog {
    strength: f32 = 0.0,
    radius: f32 = 3.0,
    first_open: bool = true
});

impl ReduceNoiseDialog {
    pub fn show(&mut self, ctx: &egui::Context) -> DialogResult<(f32, u32)> {
        let mut result = DialogResult::Open;
        let colors = DialogColors::from_ctx(ctx);

        crate::ui::polish::window(ctx, "dialog_reduce_noise")
            .title_bar(false)
            .collapsible(false)
            .resizable(false)
            .default_pos(egui::pos2(ctx.content_rect().center().x - 175.0, 60.0))
            .show(ctx, |ui| {
                ui.set_min_width(350.0);
                if paint_dialog_header(ui, &colors, "\u{1F50A}", &t!("dialog.reduce_noise")) {
                    result = DialogResult::Cancel;
                }
                ui.add_space(4.0);
                section_label(ui, &colors, "DENOISE SETTINGS");

                let mut changed = false;
                egui::Grid::new("denoise_params")
                    .num_columns(2)
                    .spacing([8.0, 6.0])
                    .show(ui, |ui| {
                        ui.label("Strength");
                        let r = ui.add(
                            egui::Slider::new(&mut self.strength, 5.0..=100.0).max_decimals(0),
                        );
                        if track_slider(&r, &mut self.dragging) {
                            changed = true;
                        }
                        ui.end_row();

                        ui.label("Radius");
                        if numeric_field_with_buttons(
                            ui,
                            &mut self.radius,
                            0.5,
                            1.0..=8.0,
                            " px",
                            1.0,
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
                    result = DialogResult::Ok((self.strength, self.radius as u32));
                }
                if cancel {
                    result = DialogResult::Cancel;
                }
            });
        result
    }
}

// -------

effect_dialog_base!(MedianDialog { radius: f32 = 0.0 });

impl MedianDialog {
    pub fn show(&mut self, ctx: &egui::Context) -> DialogResult<u32> {
        let mut result = DialogResult::Open;
        let colors = DialogColors::from_ctx(ctx);

        crate::ui::polish::window(ctx, "dialog_median")
            .title_bar(false)
            .collapsible(false)
            .resizable(false)
            .default_pos(egui::pos2(ctx.content_rect().center().x - 175.0, 60.0))
            .show(ctx, |ui| {
                ui.set_min_width(350.0);
                if paint_dialog_header(ui, &colors, "\u{1F4CA}", &t!("dialog.median_filter")) {
                    result = DialogResult::Cancel;
                }
                ui.add_space(4.0);
                section_label(ui, &colors, "FILTER SETTINGS");

                let mut changed = false;
                egui::Grid::new("median_params")
                    .num_columns(2)
                    .spacing([8.0, 6.0])
                    .show(ui, |ui| {
                        ui.label("Radius");
                        if numeric_field_with_buttons(
                            ui,
                            &mut self.radius,
                            0.5,
                            1.0..=8.0,
                            " px",
                            1.0,
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
                    result = DialogResult::Ok(self.radius as u32);
                }
                if cancel {
                    result = DialogResult::Cancel;
                }
            });
        result
    }
}

// ============================================================================
// STYLIZE DIALOGS
// ============================================================================

effect_dialog_base!(GlowDialog {
    radius: f32 = 0.0,
    intensity: f32 = 0.0,
    first_open: bool = true
});
