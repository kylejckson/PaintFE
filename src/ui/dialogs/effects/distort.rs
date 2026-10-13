impl CrystallizeDialog {
    pub fn show(&mut self, ctx: &egui::Context) -> DialogResult<(f32, u32)> {
        let mut result = DialogResult::Open;
        let colors = DialogColors::from_ctx(ctx);

        crate::ui::polish::window(ctx, "dialog_crystallize")
            .title_bar(false)
            .collapsible(false)
            .resizable(false)
            .default_pos(egui::pos2(ctx.content_rect().center().x - 175.0, 60.0))
            .show(ctx, |ui| {
                ui.set_min_width(350.0);
                if paint_dialog_header(ui, &colors, "\u{1F48E}", &t!("dialog.crystallize")) {
                    result = DialogResult::Cancel;
                }
                ui.add_space(4.0);
                section_label(ui, &colors, &crate::t!("ui.voronoi_settings"));

                let mut changed = false;
                egui::Grid::new("crystal_params")
                    .num_columns(2)
                    .spacing([8.0, 6.0])
                    .show(ui, |ui| {
                        ui.label(crate::t!("ui.cell_size"));
                        if numeric_field_with_buttons(
                            ui,
                            &mut self.cell_size,
                            0.5,
                            2.0..=100.0,
                            " px",
                            2.0,
                        ) {
                            changed = true;
                        }
                        ui.end_row();

                        ui.label(crate::t!("ui.seed"));
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

                        ui.label(crate::t!("ui.quick"));
                        ui.horizontal(|ui| {
                            ui.spacing_mut().item_spacing.x = 4.0;
                            for &(label, val) in &[
                                ("Fine", 5.0),
                                ("Medium", 15.0),
                                ("Coarse", 30.0),
                                ("Chunky", 60.0),
                            ] {
                                let btn = if (self.cell_size - val).abs() < 2.0 {
                                    egui::Button::new(
                                        egui::RichText::new(crate::i18n::translate_english(label)).strong().size(11.0),
                                    )
                                    .fill(colors.accent_faint)
                                } else {
                                    egui::Button::new(egui::RichText::new(crate::i18n::translate_english(label)).size(11.0))
                                };
                                if ui.add(btn).clicked() {
                                    self.cell_size = val;
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
                    result = DialogResult::Ok((self.cell_size, self.seed));
                }
                if cancel {
                    result = DialogResult::Cancel;
                }
            });
        result
    }
}

// -------

effect_dialog_base!(DentsDialog {
    scale: f32 = 8.0,
    amount: f32 = 0.0,
    seed: u32 = 42,
    octaves: f32 = 1.0,
    roughness: f32 = 0.5,
    pinch: bool = false,
    wrap: bool = false,
    first_open: bool = true
});

impl DentsDialog {
    pub fn show(
        &mut self,
        ctx: &egui::Context,
    ) -> DialogResult<(f32, f32, u32, u32, f32, bool, bool)> {
        let mut result = DialogResult::Open;
        let colors = DialogColors::from_ctx(ctx);

        crate::ui::polish::window(ctx, "dialog_dents")
            .title_bar(false)
            .collapsible(false)
            .resizable(false)
            .default_pos(egui::pos2(ctx.content_rect().center().x - 175.0, 60.0))
            .show(ctx, |ui| {
                ui.set_min_width(370.0);
                if paint_dialog_header(ui, &colors, "\u{1F30A}", &t!("dialog.dents")) {
                    result = DialogResult::Cancel;
                }
                ui.add_space(4.0);
                section_label(ui, &colors, &crate::t!("ui.distortion_settings"));

                let mut changed = false;
                egui::Grid::new("dents_params")
                    .num_columns(2)
                    .spacing([8.0, 6.0])
                    .show(ui, |ui| {
                        ui.label(crate::t!("dialog.resize_image.scale"));
                        let r =
                            ui.add(egui::Slider::new(&mut self.scale, 1.0..=80.0).max_decimals(1));
                        if track_slider(&r, &mut self.dragging) {
                            changed = true;
                        }
                        ui.end_row();

                        ui.label(crate::t!("ui.amount"));
                        let r =
                            ui.add(egui::Slider::new(&mut self.amount, 0.5..=30.0).max_decimals(1));
                        if track_slider(&r, &mut self.dragging) {
                            changed = true;
                        }
                        ui.end_row();

                        ui.label(crate::t!("ui.seed"));
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
                section_label(ui, &colors, &crate::t!("ui.turbulence"));

                egui::Grid::new("dents_turb")
                    .num_columns(2)
                    .spacing([8.0, 6.0])
                    .show(ui, |ui| {
                        ui.label(crate::t!("ui.octaves"));
                        let r = ui.add(
                            egui::Slider::new(&mut self.octaves, 1.0..=8.0)
                                .max_decimals(0)
                                .integer(),
                        );
                        if track_slider(&r, &mut self.dragging) {
                            changed = true;
                        }
                        ui.end_row();

                        ui.label(crate::t!("ui.roughness"));
                        let r = ui
                            .add(egui::Slider::new(&mut self.roughness, 0.1..=1.0).max_decimals(2));
                        if track_slider(&r, &mut self.dragging) {
                            changed = true;
                        }
                        ui.end_row();
                    });

                ui.add_space(4.0);
                section_label(ui, &colors, &crate::t!("ui.options"));

                egui::Grid::new("dents_opts")
                    .num_columns(2)
                    .spacing([8.0, 6.0])
                    .show(ui, |ui| {
                        ui.label(crate::t!("ui.pinch"));
                        if ui.checkbox(&mut self.pinch, crate::t!("ui.inward_bias")).changed() {
                            changed = true;
                        }
                        ui.end_row();

                        ui.label(crate::t!("ui.wrap"));
                        if ui.checkbox(&mut self.wrap, crate::t!("ui.tile_edges")).changed() {
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
                    result = DialogResult::Ok((
                        self.scale,
                        self.amount,
                        self.seed,
                        self.octaves as u32,
                        self.roughness,
                        self.pinch,
                        self.wrap,
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

effect_dialog_base!(PixelateDialog {
    block_size: f32 = 1.0,
    first_open: bool = true
});

impl PixelateDialog {
    pub fn show(&mut self, ctx: &egui::Context) -> DialogResult<u32> {
        let mut result = DialogResult::Open;
        let colors = DialogColors::from_ctx(ctx);

        crate::ui::polish::window(ctx, "dialog_pixelate")
            .title_bar(false)
            .collapsible(false)
            .resizable(false)
            .default_pos(egui::pos2(ctx.content_rect().center().x - 175.0, 60.0))
            .show(ctx, |ui| {
                ui.set_min_width(350.0);
                if paint_dialog_header(ui, &colors, "\u{1F9E9}", &t!("dialog.pixelate")) {
                    result = DialogResult::Cancel;
                }
                ui.add_space(4.0);
                section_label(ui, &colors, &crate::t!("ui.pixel_settings"));

                let mut changed = false;
                egui::Grid::new("pixelate_params")
                    .num_columns(2)
                    .spacing([8.0, 6.0])
                    .show(ui, |ui| {
                        ui.label(crate::t!("ui.block_size"));
                        if numeric_field_with_buttons(
                            ui,
                            &mut self.block_size,
                            0.5,
                            2.0..=64.0,
                            " px",
                            1.0,
                        ) {
                            changed = true;
                        }
                        ui.end_row();

                        ui.label(crate::t!("ui.quick"));
                        ui.horizontal(|ui| {
                            ui.spacing_mut().item_spacing.x = 4.0;
                            for &(label, val) in &[
                                ("Small", 4.0),
                                ("Medium", 8.0),
                                ("Large", 16.0),
                                ("Huge", 32.0),
                            ] {
                                let btn = if (self.block_size - val).abs() < 1.0 {
                                    egui::Button::new(
                                        egui::RichText::new(crate::i18n::translate_english(label)).strong().size(11.0),
                                    )
                                    .fill(colors.accent_faint)
                                } else {
                                    egui::Button::new(egui::RichText::new(crate::i18n::translate_english(label)).size(11.0))
                                };
                                if ui.add(btn).clicked() {
                                    self.block_size = val;
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
                    result = DialogResult::Ok(self.block_size as u32);
                }
                if cancel {
                    result = DialogResult::Cancel;
                }
            });
        result
    }
}

// -------

effect_dialog_base!(BulgeDialog {
    amount: f32 = 0.0,
    origin_x: f32 = 50.0,
    origin_y: f32 = 50.0,
    first_open: bool = true
});

impl BulgeDialog {
    pub fn show(&mut self, ctx: &egui::Context) -> DialogResult<(f32, f32, f32)> {
        let mut result = DialogResult::Open;
        let colors = DialogColors::from_ctx(ctx);

        crate::ui::polish::window(ctx, "dialog_bulge")
            .title_bar(false)
            .collapsible(false)
            .resizable(false)
            .default_pos(egui::pos2(ctx.content_rect().center().x - 175.0, 60.0))
            .show(ctx, |ui| {
                ui.set_min_width(350.0);
                if paint_dialog_header(ui, &colors, "\u{1F534}", &t!("dialog.bulge_pinch")) {
                    result = DialogResult::Cancel;
                }
                ui.add_space(4.0);
                section_label(ui, &colors, &crate::t!("ui.distortion_settings"));

                let mut changed = false;
                egui::Grid::new("bulge_params")
                    .num_columns(2)
                    .spacing([8.0, 6.0])
                    .show(ui, |ui| {
                        ui.label(crate::t!("ui.amount"));
                        let r =
                            ui.add(egui::Slider::new(&mut self.amount, -3.0..=3.0).max_decimals(2));
                        if track_slider(&r, &mut self.dragging) {
                            changed = true;
                        }
                        ui.end_row();
                        ui.label("");
                        ui.label(
                            egui::RichText::new(crate::t!("ui.positive_bulge_negative_pinch"))
                                .size(10.0)
                                .color(colors.text_muted),
                        );
                        ui.end_row();

                        ui.label(crate::t!("ui.origin_x"));
                        let r = ui.add(
                            egui::Slider::new(&mut self.origin_x, 0.0..=100.0)
                                .suffix("%")
                                .max_decimals(0),
                        );
                        if track_slider(&r, &mut self.dragging) {
                            changed = true;
                        }
                        ui.end_row();

                        ui.label(crate::t!("ui.origin_y"));
                        let r = ui.add(
                            egui::Slider::new(&mut self.origin_y, 0.0..=100.0)
                                .suffix("%")
                                .max_decimals(0),
                        );
                        if track_slider(&r, &mut self.dragging) {
                            changed = true;
                        }
                        ui.end_row();

                        ui.label(crate::t!("ui.origin"));
                        if ui.small_button(crate::t!("text_alignment.center")).clicked() {
                            self.origin_x = 50.0;
                            self.origin_y = 50.0;
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
                    result = DialogResult::Ok((
                        self.amount,
                        self.origin_x / 100.0,
                        self.origin_y / 100.0,
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

effect_dialog_base!(TwistDialog {
    angle: f32 = 0.0,
    origin_x: f32 = 50.0,
    origin_y: f32 = 50.0,
    first_open: bool = true
});

impl TwistDialog {
    pub fn show(&mut self, ctx: &egui::Context) -> DialogResult<(f32, f32, f32)> {
        let mut result = DialogResult::Open;
        let colors = DialogColors::from_ctx(ctx);

        crate::ui::polish::window(ctx, "dialog_twist")
            .title_bar(false)
            .collapsible(false)
            .resizable(false)
            .default_pos(egui::pos2(ctx.content_rect().center().x - 175.0, 60.0))
            .show(ctx, |ui| {
                ui.set_min_width(350.0);
                if paint_dialog_header(ui, &colors, "\u{1F300}", &t!("dialog.twist")) {
                    result = DialogResult::Cancel;
                }
                ui.add_space(4.0);
                section_label(ui, &colors, &crate::t!("ui.twist_settings"));

                let mut changed = false;
                egui::Grid::new("twist_params")
                    .num_columns(2)
                    .spacing([8.0, 6.0])
                    .show(ui, |ui| {
                        ui.label(crate::t!("ui.angle"));
                        let r = ui.add(
                            egui::Slider::new(&mut self.angle, -720.0..=720.0)
                                .suffix("°")
                                .max_decimals(0),
                        );
                        if track_slider(&r, &mut self.dragging) {
                            changed = true;
                        }
                        ui.end_row();

                        ui.label(crate::t!("ui.quick"));
                        ui.horizontal(|ui| {
                            ui.spacing_mut().item_spacing.x = 4.0;
                            for &(label, val) in &[
                                ("Mild", 45.0),
                                ("Half", 180.0),
                                ("Full", 360.0),
                                ("Double", 720.0),
                            ] {
                                let btn = if (self.angle - val).abs() < 5.0 {
                                    egui::Button::new(
                                        egui::RichText::new(crate::i18n::translate_english(label)).strong().size(11.0),
                                    )
                                    .fill(colors.accent_faint)
                                } else {
                                    egui::Button::new(egui::RichText::new(crate::i18n::translate_english(label)).size(11.0))
                                };
                                if ui.add(btn).clicked() {
                                    self.angle = val;
                                    changed = true;
                                }
                            }
                        });
                        ui.end_row();

                        ui.label(crate::t!("ui.origin_x"));
                        let r = ui.add(
                            egui::Slider::new(&mut self.origin_x, 0.0..=100.0)
                                .suffix("%")
                                .max_decimals(0),
                        );
                        if track_slider(&r, &mut self.dragging) {
                            changed = true;
                        }
                        ui.end_row();

                        ui.label(crate::t!("ui.origin_y"));
                        let r = ui.add(
                            egui::Slider::new(&mut self.origin_y, 0.0..=100.0)
                                .suffix("%")
                                .max_decimals(0),
                        );
                        if track_slider(&r, &mut self.dragging) {
                            changed = true;
                        }
                        ui.end_row();

                        ui.label(crate::t!("ui.origin"));
                        if ui.small_button(crate::t!("text_alignment.center")).clicked() {
                            self.origin_x = 50.0;
                            self.origin_y = 50.0;
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
                    result = DialogResult::Ok((
                        self.angle,
                        self.origin_x / 100.0,
                        self.origin_y / 100.0,
                    ));
                }
                if cancel {
                    result = DialogResult::Cancel;
                }
            });
        result
    }
}

// ============================================================================
// NOISE DIALOGS
// ============================================================================

effect_dialog_base!(AddNoiseDialog {
    amount: f32 = 0.0,
    noise_type_idx: usize = 0,
    monochrome: bool = false,
    seed: u32 = 42,
    scale: f32 = 1.0,
    octaves: f32 = 1.0,
    first_open: bool = true
});
