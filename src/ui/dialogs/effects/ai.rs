pub struct RemoveBackgroundDialog {
    pub threshold: f32,
    pub edge_feather: f32,
    pub mask_expansion: i32,
    pub smooth_edges: bool,
    pub fill_holes: u32,
}

impl RemoveBackgroundDialog {
    pub fn new() -> Self {
        Self {
            threshold: 0.5,
            edge_feather: 0.0,
            mask_expansion: 0,
            smooth_edges: true,
            fill_holes: 0,
        }
    }

    /// Show the settings dialog. Returns Ok(RemoveBgSettings) when the user
    /// clicks "Run", or Cancel if they dismiss.
    pub fn show(&mut self, ctx: &egui::Context) -> DialogResult<crate::ops::ai::RemoveBgSettings> {
        let mut result = DialogResult::Open;
        let colors = DialogColors::from_ctx(ctx);

        crate::ui::polish::window(ctx, "dialog_remove_bg")
            .title_bar(false)
            .collapsible(false)
            .resizable(false)
            .default_pos(egui::pos2(ctx.content_rect().center().x - 190.0, 80.0))
            .show(ctx, |ui| {
                ui.set_min_width(380.0);
                if paint_dialog_header(ui, &colors, "\u{2728}", &t!("dialog.remove_background")) {
                    result = DialogResult::Cancel;
                }
                ui.add_space(4.0);

                section_label(ui, &colors, "MASK SETTINGS");
                ui.add_space(2.0);

                egui::Grid::new("remove_bg_grid")
                    .num_columns(2)
                    .spacing([8.0, 4.0])
                    .show(ui, |ui| {
                        // Threshold
                        ui.label(egui::RichText::new("Threshold").size(11.0));
                        ui.horizontal(|ui| {
                            ui.add(
                                egui::Slider::new(&mut self.threshold, 0.05..=0.95)
                                    .step_by(0.01)
                                    .fixed_decimals(2)
                                    .text(""),
                            );
                        });
                        ui.end_row();

                        // Threshold help text
                        ui.label("");
                        ui.label(
                            egui::RichText::new(if self.threshold > 0.7 {
                                "Aggressive \u{2014} removes more of the background"
                            } else if self.threshold > 0.5 {
                                "Strict \u{2014} only keeps confident foreground"
                            } else if self.threshold > 0.3 {
                                "Balanced \u{2014} standard cutoff"
                            } else {
                                "Conservative \u{2014} keeps more of the subject"
                            })
                            .size(9.5)
                            .color(colors.text_muted),
                        );
                        ui.end_row();

                        // Smooth edges
                        ui.label(egui::RichText::new("Smooth Edges").size(11.0));
                        ui.checkbox(&mut self.smooth_edges, "Soft alpha transitions");
                        ui.end_row();

                        // Fill holes
                        ui.label(egui::RichText::new("Fill Holes").size(11.0));
                        ui.horizontal(|ui| {
                            let mut fill_val = self.fill_holes as i32;
                            ui.add(
                                egui::Slider::new(&mut fill_val, 0..=20)
                                    .suffix(" px")
                                    .text(""),
                            );
                            self.fill_holes = fill_val.max(0) as u32;
                        });
                        ui.end_row();

                        // Fill holes help text
                        ui.label("");
                        ui.label(
                            egui::RichText::new(if self.fill_holes > 0 {
                                "Fills gaps inside the subject (nose, teeth, etc.)"
                            } else {
                                "Off \u{2014} enable to fix holes in foreground"
                            })
                            .size(9.5)
                            .color(colors.text_muted),
                        );
                        ui.end_row();

                        // Edge feather
                        ui.label(egui::RichText::new("Edge Feather").size(11.0));
                        ui.horizontal(|ui| {
                            ui.add(
                                egui::Slider::new(&mut self.edge_feather, 0.0..=20.0)
                                    .step_by(0.5)
                                    .fixed_decimals(1)
                                    .suffix(" px")
                                    .text(""),
                            );
                        });
                        ui.end_row();

                        // Mask expansion
                        ui.label(egui::RichText::new("Mask Expansion").size(11.0));
                        ui.horizontal(|ui| {
                            ui.add(
                                egui::Slider::new(&mut self.mask_expansion, -10..=10)
                                    .suffix(" px")
                                    .text(""),
                            );
                        });
                        ui.end_row();

                        // Expansion help text
                        ui.label("");
                        ui.label(
                            egui::RichText::new(if self.mask_expansion > 0 {
                                "Expand: grows foreground mask outward"
                            } else if self.mask_expansion < 0 {
                                "Contract: shrinks foreground mask inward"
                            } else {
                                "No change to mask boundary"
                            })
                            .size(9.5)
                            .color(colors.text_muted),
                        );
                        ui.end_row();
                    });

                accent_separator(ui, &colors);

                section_label(ui, &colors, "PRESETS");
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 4.0;
                    if ui
                        .button(egui::RichText::new("Default").size(11.0))
                        .clicked()
                    {
                        self.threshold = 0.5;
                        self.edge_feather = 0.0;
                        self.mask_expansion = 0;
                        self.smooth_edges = true;
                        self.fill_holes = 0;
                    }
                    if ui
                        .button(egui::RichText::new("Portrait").size(11.0))
                        .clicked()
                    {
                        self.threshold = 0.5;
                        self.edge_feather = 1.0;
                        self.mask_expansion = 1;
                        self.smooth_edges = true;
                        self.fill_holes = 8;
                    }
                    if ui
                        .button(egui::RichText::new("Conservative").size(11.0))
                        .clicked()
                    {
                        self.threshold = 0.3;
                        self.edge_feather = 1.0;
                        self.mask_expansion = 2;
                        self.smooth_edges = true;
                        self.fill_holes = 5;
                    }
                    if ui
                        .button(egui::RichText::new("Aggressive").size(11.0))
                        .clicked()
                    {
                        self.threshold = 0.7;
                        self.edge_feather = 0.0;
                        self.mask_expansion = -1;
                        self.smooth_edges = false;
                        self.fill_holes = 0;
                    }
                });

                accent_separator(ui, &colors);

                // Footer: Run + Cancel
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let run_btn = ui.add(
                            egui::Button::new(
                                egui::RichText::new("\u{25B6} Run")
                                    .size(12.0)
                                    .color(contrast_text_color(colors.accent)),
                            )
                            .fill(colors.accent)
                            .min_size(egui::vec2(80.0, 24.0))
                            .corner_radius(4.0),
                        );
                        if run_btn.clicked() {
                            result = DialogResult::Ok(crate::ops::ai::RemoveBgSettings {
                                threshold: self.threshold,
                                edge_feather: self.edge_feather,
                                mask_expansion: self.mask_expansion,
                                smooth_edges: self.smooth_edges,
                                fill_holes: self.fill_holes,
                            });
                        }

                        if ui
                            .add(
                                egui::Button::new(
                                    egui::RichText::new(t!("common.cancel")).size(12.0),
                                )
                                .min_size(egui::vec2(80.0, 24.0))
                                .corner_radius(4.0),
                            )
                            .clicked()
                        {
                            result = DialogResult::Cancel;
                        }
                    });
                });
                ui.add_space(4.0);
            });
        result
    }
}
