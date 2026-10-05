impl BokehBlurDialog {
    pub fn show(&mut self, ctx: &egui::Context) -> DialogResult<f32> {
        let mut result = DialogResult::Open;
        let colors = DialogColors::from_ctx(ctx);

        crate::ui::polish::window(ctx, "dialog_bokeh_blur")
            .title_bar(false)
            .collapsible(false)
            .resizable(false)
            .default_pos(egui::pos2(ctx.content_rect().center().x - 175.0, 60.0))
            .show(ctx, |ui| {
                ui.set_min_width(350.0);
                if paint_dialog_header(ui, &colors, "\u{2B55}", &t!("dialog.bokeh_blur")) {
                    result = DialogResult::Cancel;
                }
                ui.add_space(4.0);
                section_label(ui, &colors, "BLUR SETTINGS");

                let mut changed = false;
                let slider_max = if self.advanced_blur { 100.0 } else { 10.0 };

                egui::Grid::new("bokeh_params")
                    .num_columns(2)
                    .spacing([8.0, 6.0])
                    .show(ui, |ui| {
                        ui.label("Radius");
                        ui.horizontal(|ui| {
                            ui.spacing_mut().item_spacing.x = 4.0;
                            if self.advanced_blur {
                                // Advanced: editable DragValue (up to 100)
                                let r = ui.add(
                                    crate::ui::numeric::Numeric::new(&mut self.radius)
                                        .speed(0.2)
                                        .range(1.0..=100.0)
                                        .max_decimals(1),
                                );
                                if track_slider(&r, &mut self.dragging) {
                                    changed = true;
                                }
                            } else {
                                // Normal: slider capped at 10
                                let r = ui.add(
                                    egui::Slider::new(&mut self.radius, 1.0..=slider_max)
                                        .max_decimals(1),
                                );
                                if track_slider(&r, &mut self.dragging) {
                                    changed = true;
                                }
                            }
                        });
                        ui.end_row();

                        ui.label("Quick");
                        ui.horizontal(|ui| {
                            ui.spacing_mut().item_spacing.x = 4.0;
                            let presets: &[(&str, f32)] = if self.advanced_blur {
                                &[
                                    ("Subtle", 3.0),
                                    ("Soft", 8.0),
                                    ("Medium", 20.0),
                                    ("Dreamy", 40.0),
                                    ("Max", 80.0),
                                ]
                            } else {
                                &[
                                    ("Subtle", 1.5),
                                    ("Soft", 3.0),
                                    ("Medium", 6.0),
                                    ("Dreamy", 10.0),
                                ]
                            };
                            for &(label, val) in presets {
                                let is_close = (self.radius - val).abs() < 1.0;
                                let btn = if is_close {
                                    egui::Button::new(
                                        egui::RichText::new(label).strong().size(11.0),
                                    )
                                    .fill(colors.accent_faint)
                                } else {
                                    egui::Button::new(egui::RichText::new(label).size(11.0))
                                };
                                if ui.add(btn).clicked() {
                                    self.radius = val;
                                    changed = true;
                                }
                            }
                        });
                        ui.end_row();

                        // Advanced blur toggle
                        ui.label("");
                        ui.horizontal(|ui| {
                            if ui
                                .checkbox(&mut self.advanced_blur, "Advanced (up to 100)")
                                .changed()
                            {
                                // Clamp radius when switching back to normal mode
                                if !self.advanced_blur && self.radius > 10.0 {
                                    self.radius = 10.0;
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
                    result = DialogResult::Ok(self.radius);
                }
                if cancel {
                    result = DialogResult::Cancel;
                }
            });
        result
    }
}

// -------

effect_dialog_base!(MotionBlurDialog {
    angle: f32 = 0.0,
    distance: f32 = 0.0
});

impl MotionBlurDialog {
    pub fn show(&mut self, ctx: &egui::Context) -> DialogResult<(f32, f32)> {
        let mut result = DialogResult::Open;
        let colors = DialogColors::from_ctx(ctx);

        crate::ui::polish::window(ctx, "dialog_motion_blur")
            .title_bar(false)
            .collapsible(false)
            .resizable(false)
            .default_pos(egui::pos2(ctx.content_rect().center().x - 175.0, 60.0))
            .show(ctx, |ui| {
                ui.set_min_width(350.0);
                if paint_dialog_header(ui, &colors, "\u{27A1}", &t!("dialog.motion_blur")) {
                    result = DialogResult::Cancel;
                }
                ui.add_space(4.0);
                section_label(ui, &colors, "MOTION SETTINGS");

                let mut changed = false;
                egui::Grid::new("motion_params")
                    .num_columns(2)
                    .spacing([8.0, 6.0])
                    .show(ui, |ui| {
                        ui.label("Angle");
                        let r = ui.add(
                            egui::Slider::new(&mut self.angle, -180.0..=180.0)
                                .suffix("°")
                                .max_decimals(0),
                        );
                        if track_slider(&r, &mut self.dragging) {
                            changed = true;
                        }
                        ui.end_row();

                        ui.label("Distance");
                        let r = ui.add(
                            egui::Slider::new(&mut self.distance, 1.0..=100.0)
                                .suffix(" px")
                                .max_decimals(0),
                        );
                        if track_slider(&r, &mut self.dragging) {
                            changed = true;
                        }
                        ui.end_row();

                        ui.label("Direction");
                        ui.horizontal(|ui| {
                            ui.spacing_mut().item_spacing.x = 4.0;
                            for &(label, val) in
                                &[("→", 0.0), ("↗", -45.0), ("↑", -90.0), ("↖", -135.0)]
                            {
                                let btn = if (self.angle - val).abs() < 1.0 {
                                    egui::Button::new(egui::RichText::new(label).strong())
                                        .fill(colors.accent_faint)
                                } else {
                                    egui::Button::new(label)
                                };
                                if ui.add(btn).clicked() {
                                    self.angle = val;
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
                    result = DialogResult::Ok((self.angle, self.distance));
                }
                if cancel {
                    result = DialogResult::Cancel;
                }
            });
        result
    }
}

// -------

effect_dialog_base!(BoxBlurDialog { radius: f32 = 0.0 });

impl BoxBlurDialog {
    pub fn show(&mut self, ctx: &egui::Context) -> DialogResult<f32> {
        let mut result = DialogResult::Open;
        let colors = DialogColors::from_ctx(ctx);

        crate::ui::polish::window(ctx, "dialog_box_blur")
            .title_bar(false)
            .collapsible(false)
            .resizable(false)
            .default_pos(egui::pos2(ctx.content_rect().center().x - 175.0, 60.0))
            .show(ctx, |ui| {
                ui.set_min_width(350.0);
                if paint_dialog_header(ui, &colors, "\u{25A3}", &t!("dialog.box_blur")) {
                    result = DialogResult::Cancel;
                }
                ui.add_space(4.0);
                section_label(ui, &colors, "BLUR SETTINGS");

                let mut changed = false;
                egui::Grid::new("box_params")
                    .num_columns(2)
                    .spacing([8.0, 6.0])
                    .show(ui, |ui| {
                        ui.label("Radius");
                        let r =
                            ui.add(egui::Slider::new(&mut self.radius, 1.0..=50.0).max_decimals(1));
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
                    result = DialogResult::Ok(self.radius);
                }
                if cancel {
                    result = DialogResult::Cancel;
                }
            });
        result
    }
}

// -------

effect_dialog_base!(ZoomBlurDialog {
    center_x: f32 = 0.5,
    center_y: f32 = 0.5,
    intensity: f32 = 30.0, // 1–100, mapped to 0.01–0.99 strength
    quality: u8 = 1,       // 0=Fast(8), 1=Normal(16), 2=High(32)
    tint_enabled: bool = false,
    tint_r: f32 = 1.0,
    tint_g: f32 = 0.6,
    tint_b: f32 = 0.1,
    tint_mix: f32 = 30.0, // 0–100
    first_open: bool = true,
});

fn zoom_quality_samples(q: u8) -> u32 {
    match q {
        0 => 8,
        1 => 16,
        _ => 32,
    }
}

impl ZoomBlurDialog {
    /// Returns (center_x, center_y, strength, samples, tint_color, tint_strength)
    pub fn current_params(&self) -> (f32, f32, f32, u32, [f32; 4], f32) {
        let strength = (self.intensity / 100.0).clamp(0.01, 0.99);
        let samples = zoom_quality_samples(self.quality);
        let tint = [self.tint_r, self.tint_g, self.tint_b, 1.0];
        let ts = if self.tint_enabled {
            self.tint_mix / 100.0
        } else {
            0.0
        };
        (self.center_x, self.center_y, strength, samples, tint, ts)
    }

    pub fn show(&mut self, ctx: &egui::Context) -> DialogResult<()> {
        let mut result = DialogResult::Open;
        let colors = DialogColors::from_ctx(ctx);

        crate::ui::polish::window(ctx, "dialog_zoom_blur")
            .title_bar(false)
            .collapsible(false)
            .resizable(false)
            .default_pos(egui::pos2(ctx.content_rect().center().x - 190.0, 60.0))
            .show(ctx, |ui| {
                ui.set_min_width(380.0);
                if paint_dialog_header(ui, &colors, "\u{25CE}", &t!("dialog.zoom_blur")) {
                    result = DialogResult::Cancel;
                }
                ui.add_space(4.0);

                let mut changed = false;

                // Trigger an initial preview on the very first frame the dialog is shown.
                if self.first_open {
                    self.first_open = false;
                    changed = true;
                }

                // --------------------------------------------------
                // ZOOM ORIGIN
                // --------------------------------------------------
                section_label(ui, &colors, "ZOOM ORIGIN");
                egui::Grid::new("zoom_origin_params")
                    .num_columns(2)
                    .spacing([8.0, 6.0])
                    .show(ui, |ui| {
                        ui.label("Center X");
                        let r = ui.add(
                            egui::Slider::new(&mut self.center_x, 0.0..=1.0)
                                .custom_formatter(|v, _| format!("{:.0}%", v * 100.0))
                                .custom_parser(|s| {
                                    s.trim_end_matches('%')
                                        .parse::<f64>()
                                        .ok()
                                        .map(|v| v / 100.0)
                                }),
                        );
                        if track_slider(&r, &mut self.dragging) {
                            changed = true;
                        }
                        ui.end_row();

                        ui.label("Center Y");
                        let r = ui.add(
                            egui::Slider::new(&mut self.center_y, 0.0..=1.0)
                                .custom_formatter(|v, _| format!("{:.0}%", v * 100.0))
                                .custom_parser(|s| {
                                    s.trim_end_matches('%')
                                        .parse::<f64>()
                                        .ok()
                                        .map(|v| v / 100.0)
                                }),
                        );
                        if track_slider(&r, &mut self.dragging) {
                            changed = true;
                        }
                        ui.end_row();

                        ui.label("Preset");
                        ui.horizontal_wrapped(|ui| {
                            ui.spacing_mut().item_spacing = egui::vec2(3.0, 3.0);
                            let presets: &[(&str, f32, f32)] = &[
                                ("↖", 0.0, 0.0),
                                ("↑", 0.5, 0.0),
                                ("↗", 1.0, 0.0),
                                ("←", 0.0, 0.5),
                                ("⊙", 0.5, 0.5),
                                ("→", 1.0, 0.5),
                                ("↙", 0.0, 1.0),
                                ("↓", 0.5, 1.0),
                                ("↘", 1.0, 1.0),
                            ];
                            for &(lbl, px, py) in presets {
                                let active = (self.center_x - px).abs() < 0.01
                                    && (self.center_y - py).abs() < 0.01;
                                let btn = if active {
                                    egui::Button::new(egui::RichText::new(lbl).strong())
                                        .fill(colors.accent_faint)
                                } else {
                                    egui::Button::new(lbl)
                                };
                                if ui.add_sized([28.0, 22.0], btn).clicked() {
                                    self.center_x = px;
                                    self.center_y = py;
                                    changed = true;
                                }
                            }
                        });
                        ui.end_row();
                    });

                accent_separator(ui, &colors);

                // --------------------------------------------------
                // BLUR SETTINGS
                // --------------------------------------------------
                section_label(ui, &colors, "BLUR SETTINGS");
                egui::Grid::new("zoom_blur_params")
                    .num_columns(2)
                    .spacing([8.0, 6.0])
                    .show(ui, |ui| {
                        ui.label("Intensity");
                        let r = ui.add(
                            egui::Slider::new(&mut self.intensity, 1.0..=100.0)
                                .max_decimals(0)
                                .suffix("%"),
                        );
                        if track_slider(&r, &mut self.dragging) {
                            changed = true;
                        }
                        ui.end_row();

                        ui.label("Quality");
                        egui::ComboBox::from_id_salt("zoom_quality")
                            .selected_text(match self.quality {
                                0 => "Fast",
                                1 => "Normal",
                                _ => "High",
                            })
                            .show_ui(ui, |ui| {
                                if ui
                                    .selectable_value(&mut self.quality, 0, "Fast  (8 samples)")
                                    .changed()
                                {
                                    changed = true;
                                }
                                if ui
                                    .selectable_value(&mut self.quality, 1, "Normal (16 samples)")
                                    .changed()
                                {
                                    changed = true;
                                }
                                if ui
                                    .selectable_value(&mut self.quality, 2, "High  (32 samples)")
                                    .changed()
                                {
                                    changed = true;
                                }
                            });
                        ui.end_row();

                        ui.label("Quick");
                        ui.horizontal(|ui| {
                            ui.spacing_mut().item_spacing.x = 4.0;
                            for &(lbl, val) in &[
                                ("Subtle", 10.0f32),
                                ("Soft", 25.0),
                                ("Strong", 55.0),
                                ("Max", 90.0),
                            ] {
                                let active = (self.intensity - val).abs() < 2.0;
                                let btn = if active {
                                    egui::Button::new(egui::RichText::new(lbl).strong().size(11.0))
                                        .fill(colors.accent_faint)
                                } else {
                                    egui::Button::new(egui::RichText::new(lbl).size(11.0))
                                };
                                if ui.add(btn).clicked() {
                                    self.intensity = val;
                                    changed = true;
                                }
                            }
                        });
                        ui.end_row();
                    });

                accent_separator(ui, &colors);

                // --------------------------------------------------
                // COLOR TINT
                // --------------------------------------------------
                section_label(ui, &colors, "COLOR TINT");
                egui::Grid::new("zoom_tint_params")
                    .num_columns(2)
                    .spacing([8.0, 6.0])
                    .show(ui, |ui| {
                        ui.label("Enable");
                        if ui
                            .checkbox(&mut self.tint_enabled, "Radial color tint at origin")
                            .changed()
                        {
                            changed = true;
                        }
                        ui.end_row();

                        if self.tint_enabled {
                            ui.label("Tint Color");
                            ui.horizontal(|ui| {
                                let mut col = egui::Color32::from_rgb(
                                    (self.tint_r * 255.0) as u8,
                                    (self.tint_g * 255.0) as u8,
                                    (self.tint_b * 255.0) as u8,
                                );
                                if ui.color_edit_button_srgba(&mut col).changed() {
                                    self.tint_r = col.r() as f32 / 255.0;
                                    self.tint_g = col.g() as f32 / 255.0;
                                    self.tint_b = col.b() as f32 / 255.0;
                                    changed = true;
                                }
                                ui.spacing_mut().item_spacing.x = 4.0;
                                for &(lbl, r, g, b) in &[
                                    ("White", 1.0f32, 1.0f32, 1.0f32),
                                    ("Black", 0.0, 0.0, 0.0),
                                    ("Warm", 1.0, 0.6, 0.1),
                                    ("Cool", 0.2, 0.5, 1.0),
                                    ("Fire", 1.0, 0.2, 0.0),
                                ] {
                                    let active = (self.tint_r - r).abs() < 0.05
                                        && (self.tint_g - g).abs() < 0.05
                                        && (self.tint_b - b).abs() < 0.05;
                                    let swatch = egui::Button::new("  ")
                                        .fill(egui::Color32::from_rgb(
                                            (r * 255.0) as u8,
                                            (g * 255.0) as u8,
                                            (b * 255.0) as u8,
                                        ))
                                        .stroke(if active {
                                            egui::Stroke::new(2.0, colors.accent)
                                        } else {
                                            egui::Stroke::new(1.0, egui::Color32::GRAY)
                                        });
                                    if ui
                                        .add_sized([20.0, 20.0], swatch)
                                        .on_hover_text(lbl)
                                        .clicked()
                                    {
                                        self.tint_r = r;
                                        self.tint_g = g;
                                        self.tint_b = b;
                                        changed = true;
                                    }
                                }
                            });
                            ui.end_row();

                            ui.label("Mix");
                            let r = ui.add(
                                egui::Slider::new(&mut self.tint_mix, 0.0..=100.0)
                                    .max_decimals(0)
                                    .suffix("%"),
                            );
                            if track_slider(&r, &mut self.dragging) {
                                changed = true;
                            }
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
                    result = DialogResult::Ok(());
                }
                if cancel {
                    result = DialogResult::Cancel;
                }
            });
        result
    }
}

// ============================================================================
// DISTORTION DIALOGS
// ============================================================================

effect_dialog_base!(CrystallizeDialog {
    cell_size: f32 = 1.0,
    seed: u32 = 42,
    first_open: bool = true
});
