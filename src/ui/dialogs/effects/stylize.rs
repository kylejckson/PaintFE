impl GlowDialog {
    pub fn show(&mut self, ctx: &egui::Context) -> DialogResult<(f32, f32)> {
        let mut result = DialogResult::Open;
        let colors = DialogColors::from_ctx(ctx);

        crate::ui::polish::window(ctx, "dialog_glow")
            .title_bar(false)
            .collapsible(false)
            .resizable(false)
            .default_pos(egui::pos2(ctx.content_rect().center().x - 175.0, 60.0))
            .show(ctx, |ui| {
                ui.set_min_width(350.0);
                if paint_dialog_header(ui, &colors, "\u{2728}", &t!("dialog.glow")) {
                    result = DialogResult::Cancel;
                }
                ui.add_space(4.0);
                section_label(ui, &colors, "GLOW SETTINGS");

                let mut changed = false;
                egui::Grid::new("glow_params")
                    .num_columns(2)
                    .spacing([8.0, 6.0])
                    .show(ui, |ui| {
                        ui.label("Radius");
                        let r =
                            ui.add(egui::Slider::new(&mut self.radius, 1.0..=30.0).max_decimals(1));
                        if track_slider(&r, &mut self.dragging) {
                            changed = true;
                        }
                        ui.end_row();

                        ui.label("Intensity");
                        let r = ui
                            .add(egui::Slider::new(&mut self.intensity, 0.0..=2.0).max_decimals(2));
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
                    result = DialogResult::Ok((self.radius, self.intensity));
                }
                if cancel {
                    result = DialogResult::Cancel;
                }
            });
        result
    }
}

// -------

effect_dialog_base!(SharpenDialog {
    amount: f32 = 0.0,
    radius: f32 = 1.5,
    first_open: bool = true
});

impl SharpenDialog {
    pub fn show(&mut self, ctx: &egui::Context) -> DialogResult<(f32, f32)> {
        let mut result = DialogResult::Open;
        let colors = DialogColors::from_ctx(ctx);

        crate::ui::polish::window(ctx, "dialog_sharpen")
            .title_bar(false)
            .collapsible(false)
            .resizable(false)
            .default_pos(egui::pos2(ctx.content_rect().center().x - 175.0, 60.0))
            .show(ctx, |ui| {
                ui.set_min_width(350.0);
                if paint_dialog_header(ui, &colors, "\u{1F4CC}", &t!("dialog.sharpen")) {
                    result = DialogResult::Cancel;
                }
                ui.add_space(4.0);
                section_label(ui, &colors, "SHARPEN SETTINGS");

                let mut changed = false;
                egui::Grid::new("sharpen_params")
                    .num_columns(2)
                    .spacing([8.0, 6.0])
                    .show(ui, |ui| {
                        ui.label("Amount");
                        let r =
                            ui.add(egui::Slider::new(&mut self.amount, 0.1..=5.0).max_decimals(1));
                        if track_slider(&r, &mut self.dragging) {
                            changed = true;
                        }
                        ui.end_row();

                        ui.label("Radius");
                        let r =
                            ui.add(egui::Slider::new(&mut self.radius, 0.5..=10.0).max_decimals(1));
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
                    result = DialogResult::Ok((self.amount, self.radius));
                }
                if cancel {
                    result = DialogResult::Cancel;
                }
            });
        result
    }
}

// -------

effect_dialog_base!(VignetteDialog {
    amount: f32 = 0.0,
    softness: f32 = 0.6,
    first_open: bool = true
});

impl VignetteDialog {
    pub fn show(&mut self, ctx: &egui::Context) -> DialogResult<(f32, f32)> {
        let mut result = DialogResult::Open;
        let colors = DialogColors::from_ctx(ctx);

        crate::ui::polish::window(ctx, "dialog_vignette")
            .title_bar(false)
            .collapsible(false)
            .resizable(false)
            .default_pos(egui::pos2(ctx.content_rect().center().x - 175.0, 60.0))
            .show(ctx, |ui| {
                ui.set_min_width(350.0);
                if paint_dialog_header(ui, &colors, "\u{1F311}", &t!("dialog.vignette")) {
                    result = DialogResult::Cancel;
                }
                ui.add_space(4.0);
                section_label(ui, &colors, "VIGNETTE SETTINGS");

                let mut changed = false;
                egui::Grid::new("vignette_params")
                    .num_columns(2)
                    .spacing([8.0, 6.0])
                    .show(ui, |ui| {
                        ui.label("Amount");
                        let r =
                            ui.add(egui::Slider::new(&mut self.amount, 0.0..=2.0).max_decimals(2));
                        if track_slider(&r, &mut self.dragging) {
                            changed = true;
                        }
                        ui.end_row();

                        ui.label("Softness");
                        let r = ui
                            .add(egui::Slider::new(&mut self.softness, 0.1..=1.5).max_decimals(2));
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
                    result = DialogResult::Ok((self.amount, self.softness));
                }
                if cancel {
                    result = DialogResult::Cancel;
                }
            });
        result
    }
}

// -------

effect_dialog_base!(HalftoneDialog {
    dot_size: f32 = 1.0,
    angle: f32 = 45.0,
    shape_idx: usize = 0,
    first_open: bool = true
});

impl HalftoneDialog {
    pub fn halftone_shape(&self) -> HalftoneShape {
        match self.shape_idx {
            1 => HalftoneShape::Square,
            2 => HalftoneShape::Diamond,
            3 => HalftoneShape::Line,
            _ => HalftoneShape::Circle,
        }
    }

    pub fn show(&mut self, ctx: &egui::Context) -> DialogResult<(f32, f32, HalftoneShape)> {
        let mut result = DialogResult::Open;
        let colors = DialogColors::from_ctx(ctx);

        crate::ui::polish::window(ctx, "dialog_halftone")
            .title_bar(false)
            .collapsible(false)
            .resizable(false)
            .default_pos(egui::pos2(ctx.content_rect().center().x - 175.0, 60.0))
            .show(ctx, |ui| {
                ui.set_min_width(370.0);
                if paint_dialog_header(ui, &colors, "\u{25CF}", &t!("dialog.halftone")) {
                    result = DialogResult::Cancel;
                }
                ui.add_space(4.0);
                section_label(ui, &colors, "HALFTONE SETTINGS");

                let mut changed = false;
                egui::Grid::new("halftone_params")
                    .num_columns(2)
                    .spacing([8.0, 6.0])
                    .show(ui, |ui| {
                        ui.label("Dot Size");
                        let r = ui.add(
                            egui::Slider::new(&mut self.dot_size, 2.0..=30.0)
                                .suffix(" px")
                                .max_decimals(0),
                        );
                        if track_slider(&r, &mut self.dragging) {
                            changed = true;
                        }
                        ui.end_row();

                        ui.label("Angle");
                        let r = ui.add(
                            egui::Slider::new(&mut self.angle, -90.0..=90.0)
                                .suffix("°")
                                .max_decimals(0),
                        );
                        if track_slider(&r, &mut self.dragging) {
                            changed = true;
                        }
                        ui.end_row();

                        ui.label("Shape");
                        ui.horizontal(|ui| {
                            ui.spacing_mut().item_spacing.x = 4.0;
                            let shapes = [
                                ("\u{25CF}", "Circle"),
                                ("\u{25A0}", "Square"),
                                ("\u{25C6}", "Diamond"),
                                ("\u{2550}", "Line"),
                            ];
                            for (i, (icon, label)) in shapes.iter().enumerate() {
                                let text = format!("{} {}", icon, label);
                                let btn = if self.shape_idx == i {
                                    egui::Button::new(
                                        egui::RichText::new(&text).strong().size(11.0),
                                    )
                                    .fill(colors.accent_faint)
                                } else {
                                    egui::Button::new(egui::RichText::new(&text).size(11.0))
                                };
                                if ui.add(btn).clicked() {
                                    self.shape_idx = i;
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
                    result = DialogResult::Ok((self.dot_size, self.angle, self.halftone_shape()));
                }
                if cancel {
                    result = DialogResult::Cancel;
                }
            });
        result
    }
}

// ============================================================================
// RENDER DIALOGS
// ============================================================================

effect_dialog_base!(GridDialog {
    cell_w: f32 = 32.0,
    cell_h: f32 = 32.0,
    line_width: f32 = 1.0,
    color: [f32; 3] = [0.0, 0.0, 0.0],
    opacity: f32 = 0.0,
    style_idx: usize = 0,
    first_open: bool = true
});
