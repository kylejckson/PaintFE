impl GridDialog {
    pub fn grid_style(&self) -> GridStyle {
        match self.style_idx {
            1 => GridStyle::Checkerboard,
            _ => GridStyle::Lines,
        }
    }

    pub fn show(
        &mut self,
        ctx: &egui::Context,
    ) -> DialogResult<(u32, u32, u32, [u8; 4], GridStyle, f32)> {
        let mut result = DialogResult::Open;
        let colors = DialogColors::from_ctx(ctx);

        egui::Window::new("dialog_grid")
            .title_bar(false)
            .collapsible(false)
            .resizable(false)
            .default_pos(egui::pos2(ctx.content_rect().center().x - 175.0, 60.0))
            .show(ctx, |ui| {
                ui.set_min_width(370.0);
                if paint_dialog_header(ui, &colors, "\u{1F4D0}", &t!("dialog.grid")) { result = DialogResult::Cancel; }
                ui.add_space(4.0);
                section_label(ui, &colors, "GRID SETTINGS");

                let mut changed = false;
                egui::Grid::new("grid_params")
                    .num_columns(2)
                    .spacing([8.0, 6.0])
                    .show(ui, |ui| {
                        ui.label("Cell Width");
                        if numeric_field_with_buttons(
                            ui,
                            &mut self.cell_w,
                            1.0,
                            4.0..=256.0,
                            " px",
                            4.0,
                        ) {
                            changed = true;
                        }
                        ui.end_row();

                        ui.label("Cell Height");
                        if numeric_field_with_buttons(
                            ui,
                            &mut self.cell_h,
                            1.0,
                            4.0..=256.0,
                            " px",
                            4.0,
                        ) {
                            changed = true;
                        }
                        ui.end_row();

                        ui.label("Line Width");
                        if numeric_field_with_buttons(
                            ui,
                            &mut self.line_width,
                            0.5,
                            1.0..=8.0,
                            " px",
                            1.0,
                        ) {
                            changed = true;
                        }
                        ui.end_row();

                        ui.label("Color");
                        if ui.color_edit_button_rgb(&mut self.color).changed() {
                            changed = true;
                        }
                        ui.end_row();

                        ui.label("Opacity");
                        if dialog_slider(ui, &mut self.opacity, 0.0..=1.0, 0.01, "", 2) {
                            changed = true;
                        }
                        ui.end_row();

                        ui.label("Style");
                        ui.horizontal(|ui| {
                            ui.spacing_mut().item_spacing.x = 4.0;
                            for (i, label) in ["Lines", "Checkerboard"].iter().enumerate() {
                                let btn = if self.style_idx == i {
                                    egui::Button::new(
                                        egui::RichText::new(*label).strong().size(11.0),
                                    )
                                    .fill(colors.accent_faint)
                                } else {
                                    egui::Button::new(egui::RichText::new(*label).size(11.0))
                                };
                                if ui.add(btn).clicked() {
                                    self.style_idx = i;
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
                        (self.color[0] * 255.0).round() as u8,
                        (self.color[1] * 255.0).round() as u8,
                        (self.color[2] * 255.0).round() as u8,
                        255,
                    ];
                    result = DialogResult::Ok((
                        self.cell_w as u32,
                        self.cell_h as u32,
                        self.line_width as u32,
                        c,
                        self.grid_style(),
                        self.opacity,
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

effect_dialog_base!(DropShadowDialog {
    offset_x: f32 = 0.0,
    offset_y: f32 = 0.0,
    blur_radius: f32 = 0.0,
    widen_radius: bool = false,
    color: [f32; 3] = [0.0, 0.0, 0.0],
    opacity: f32 = 1.0,
    first_open: bool = true
});

impl DropShadowDialog {
    pub fn show(
        &mut self,
        ctx: &egui::Context,
    ) -> DialogResult<(i32, i32, f32, bool, [u8; 4], f32)> {
        let mut result = DialogResult::Open;
        let colors = DialogColors::from_ctx(ctx);

        egui::Window::new("dialog_drop_shadow")
            .title_bar(false)
            .collapsible(false)
            .resizable(false)
            .default_pos(egui::pos2(ctx.content_rect().center().x - 175.0, 60.0))
            .show(ctx, |ui| {
                ui.set_min_width(380.0);
                if paint_dialog_header(ui, &colors, "\u{1F4A4}", &t!("dialog.drop_shadow")) { result = DialogResult::Cancel; }
                ui.add_space(4.0);
                section_label(ui, &colors, "SHADOW SETTINGS");

                let mut changed = false;
                egui::Grid::new("shadow_params")
                    .num_columns(2)
                    .spacing([8.0, 6.0])
                    .show(ui, |ui| {
                        ui.label("Offset X");
                        if dialog_slider(ui, &mut self.offset_x, -50.0..=50.0, 1.0, " px", 0) {
                            changed = true;
                        }
                        ui.end_row();

                        ui.label("Offset Y");
                        if dialog_slider(ui, &mut self.offset_y, -50.0..=50.0, 1.0, " px", 0) {
                            changed = true;
                        }
                        ui.end_row();

                        ui.label("Radius");
                        if dialog_slider(ui, &mut self.blur_radius, 0.0..=30.0, 0.1, " px", 1) {
                            changed = true;
                        }
                        ui.end_row();

                        ui.label("Widen Radius");
                        if ui
                            .checkbox(&mut self.widen_radius, "Expand spread before blur")
                            .changed()
                        {
                            changed = true;
                        }
                        ui.end_row();

                        ui.label("Color");
                        if ui.color_edit_button_rgb(&mut self.color).changed() {
                            changed = true;
                        }
                        ui.end_row();

                        ui.label("Opacity");
                        if dialog_slider(ui, &mut self.opacity, 0.0..=1.0, 0.01, "", 2) {
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
                    let c = [
                        (self.color[0] * 255.0).round() as u8,
                        (self.color[1] * 255.0).round() as u8,
                        (self.color[2] * 255.0).round() as u8,
                        255,
                    ];
                    result = DialogResult::Ok((
                        self.offset_x as i32,
                        self.offset_y as i32,
                        self.blur_radius,
                        self.widen_radius,
                        c,
                        self.opacity,
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

effect_dialog_base!(OutlineDialog {
    width: f32 = 4.0,
    color: [f32; 3] = [0.0, 0.0, 0.0],
    mode_idx: usize = 0,
    anti_alias: bool = true,
    first_open: bool = true
});

impl OutlineDialog {
    pub fn outline_mode(&self) -> OutlineMode {
        match self.mode_idx {
            1 => OutlineMode::Inside,
            2 => OutlineMode::Center,
            _ => OutlineMode::Outside,
        }
    }

    pub fn show(&mut self, ctx: &egui::Context) -> DialogResult<(u32, [u8; 4], OutlineMode, bool)> {
        let mut result = DialogResult::Open;
        let colors = DialogColors::from_ctx(ctx);

        egui::Window::new("dialog_outline")
            .title_bar(false)
            .collapsible(false)
            .resizable(false)
            .default_pos(egui::pos2(ctx.content_rect().center().x - 175.0, 60.0))
            .show(ctx, |ui| {
                ui.set_min_width(360.0);
                if paint_dialog_header(ui, &colors, "\u{1F58A}", &t!("dialog.outline")) { result = DialogResult::Cancel; }
                ui.add_space(4.0);
                section_label(ui, &colors, "OUTLINE SETTINGS");

                let mut changed = false;
                egui::Grid::new("outline_params")
                    .num_columns(2)
                    .spacing([8.0, 6.0])
                    .show(ui, |ui| {
                        ui.label("Width");
                        if numeric_field_with_buttons(
                            ui,
                            &mut self.width,
                            0.5,
                            1.0..=4096.0,
                            " px",
                            1.0,
                        ) {
                            changed = true;
                        }
                        ui.end_row();

                        ui.label("Color");
                        if ui.color_edit_button_rgb(&mut self.color).changed() {
                            changed = true;
                        }
                        ui.end_row();

                        ui.label("Mode");
                        ui.horizontal(|ui| {
                            ui.spacing_mut().item_spacing.x = 4.0;
                            for (i, label) in ["Outside", "Inside", "Center"].iter().enumerate() {
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

                        ui.label(t!("ctx.anti_alias"));
                        if ui
                            .checkbox(&mut self.anti_alias, t!("ctx.anti_alias"))
                            .changed()
                        {
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
                    let c = [
                        (self.color[0] * 255.0) as u8,
                        (self.color[1] * 255.0) as u8,
                        (self.color[2] * 255.0) as u8,
                        255,
                    ];
                    result = DialogResult::Ok((
                        self.width as u32,
                        c,
                        self.outline_mode(),
                        self.anti_alias,
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

effect_dialog_base!(CanvasBorderDialog {
    width: f32 = 8.0,
    first_open: bool = true
});

impl CanvasBorderDialog {
    pub fn show(&mut self, ctx: &egui::Context) -> DialogResult<u32> {
        let mut result = DialogResult::Open;
        let colors = DialogColors::from_ctx(ctx);

        egui::Window::new("dialog_canvas_border")
            .title_bar(false)
            .collapsible(false)
            .resizable(false)
            .default_pos(egui::pos2(ctx.content_rect().center().x - 175.0, 70.0))
            .show(ctx, |ui| {
                ui.set_min_width(360.0);
                if paint_dialog_header(ui, &colors, "\u{25A3}", &t!("dialog.canvas_border")) { result = DialogResult::Cancel; }
                ui.add_space(4.0);
                section_label(ui, &colors, "BORDER SETTINGS");

                let mut changed = false;
                egui::Grid::new("canvas_border_params")
                    .num_columns(2)
                    .spacing([8.0, 6.0])
                    .show(ui, |ui| {
                        ui.label("Width");
                        if numeric_field_with_buttons(
                            ui,
                            &mut self.width,
                            1.0,
                            1.0..=512.0,
                            " px",
                            8.0,
                        ) {
                            changed = true;
                        }
                        ui.end_row();

                        ui.label("Color");
                        ui.label("Uses Primary Color");
                        ui.end_row();
                    });

                accent_separator(ui, &colors);
                let manual = preview_controls(ui, &colors, &mut self.live_preview);
                if (changed && self.live_preview) || manual {
                    result = DialogResult::Changed;
                }

                let (ok, cancel, reset) = dialog_footer_with_reset(ui, &colors);
                if ok {
                    result = DialogResult::Ok(self.width as u32);
                }
                if cancel {
                    result = DialogResult::Cancel;
                }
                if reset {
                    self.width = 8.0;
                    if self.live_preview {
                        result = DialogResult::Changed;
                    }
                }
            });

        result
    }
}

// ============================================================================
// GLITCH DIALOGS
// ============================================================================

effect_dialog_base!(PixelDragDialog {
    seed: u32 = 42,
    amount: f32 = 0.0,
    distance: f32 = 0.0,
    direction: f32 = 0.0,
    first_open: bool = true
});

impl PixelDragDialog {
    pub fn show(&mut self, ctx: &egui::Context) -> DialogResult<(u32, f32, u32, f32)> {
        let mut result = DialogResult::Open;
        let colors = DialogColors::from_ctx(ctx);

        egui::Window::new("dialog_pixel_drag")
            .title_bar(false)
            .collapsible(false)
            .resizable(false)
            .default_pos(egui::pos2(ctx.content_rect().center().x - 175.0, 60.0))
            .show(ctx, |ui| {
                ui.set_min_width(370.0);
                if paint_dialog_header(ui, &colors, "\u{1F4A2}", &t!("dialog.pixel_drag")) { result = DialogResult::Cancel; }
                ui.add_space(4.0);
                section_label(ui, &colors, "GLITCH SETTINGS");

                let mut changed = false;
                egui::Grid::new("pixdrag_params")
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

                        ui.label("Distance");
                        let r = ui.add(
                            egui::Slider::new(&mut self.distance, 5.0..=200.0)
                                .suffix(" px")
                                .max_decimals(0),
                        );
                        if track_slider(&r, &mut self.dragging) {
                            changed = true;
                        }
                        ui.end_row();

                        ui.label("Direction");
                        let r = ui.add(
                            egui::Slider::new(&mut self.direction, -180.0..=180.0)
                                .suffix("°")
                                .max_decimals(0),
                        );
                        if track_slider(&r, &mut self.dragging) {
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

                accent_separator(ui, &colors);
                let manual = preview_controls(ui, &colors, &mut self.live_preview);
                if (changed && self.live_preview) || manual {
                    result = DialogResult::Changed;
                }

                let (ok, cancel) = dialog_footer(ui, &colors);
                if ok {
                    result = DialogResult::Ok((
                        self.seed,
                        self.amount,
                        self.distance as u32,
                        self.direction,
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

effect_dialog_base!(RgbDisplaceDialog {
    r_x: f32 = 0.0,
    r_y: f32 = 0.0,
    g_x: f32 = 0.0,
    g_y: f32 = 0.0,
    b_x: f32 = 0.0,
    b_y: f32 = 0.0,
    first_open: bool = true
});

impl RgbDisplaceDialog {
    pub fn show(
        &mut self,
        ctx: &egui::Context,
    ) -> DialogResult<((i32, i32), (i32, i32), (i32, i32))> {
        let mut result = DialogResult::Open;
        let colors = DialogColors::from_ctx(ctx);

        egui::Window::new("dialog_rgb_displace")
            .title_bar(false)
            .collapsible(false)
            .resizable(false)
            .default_pos(egui::pos2(ctx.content_rect().center().x - 200.0, 60.0))
            .show(ctx, |ui| {
                ui.set_min_width(420.0);
                if paint_dialog_header(ui, &colors, "\u{1F308}", &t!("dialog.rgb_displace")) { result = DialogResult::Cancel; }
                ui.add_space(4.0);
                section_label(ui, &colors, "CHANNEL OFFSETS");

                let mut changed = false;

                // Red channel
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new("Red")
                            .color(Color32::from_rgb(220, 50, 50))
                            .strong()
                            .size(12.0),
                    );
                });
                egui::Grid::new("rgb_r")
                    .num_columns(2)
                    .spacing([8.0, 4.0])
                    .show(ui, |ui| {
                        ui.label("  X");
                        let r = ui.add(
                            egui::Slider::new(&mut self.r_x, -100.0..=100.0)
                                .suffix(" px")
                                .max_decimals(0),
                        );
                        if track_slider(&r, &mut self.dragging) {
                            changed = true;
                        }
                        ui.end_row();
                        ui.label("  Y");
                        let r = ui.add(
                            egui::Slider::new(&mut self.r_y, -100.0..=100.0)
                                .suffix(" px")
                                .max_decimals(0),
                        );
                        if track_slider(&r, &mut self.dragging) {
                            changed = true;
                        }
                        ui.end_row();
                    });

                ui.add_space(2.0);

                // Green channel
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new("Green")
                            .color(Color32::from_rgb(50, 180, 50))
                            .strong()
                            .size(12.0),
                    );
                });
                egui::Grid::new("rgb_g")
                    .num_columns(2)
                    .spacing([8.0, 4.0])
                    .show(ui, |ui| {
                        ui.label("  X");
                        let r = ui.add(
                            egui::Slider::new(&mut self.g_x, -100.0..=100.0)
                                .suffix(" px")
                                .max_decimals(0),
                        );
                        if track_slider(&r, &mut self.dragging) {
                            changed = true;
                        }
                        ui.end_row();
                        ui.label("  Y");
                        let r = ui.add(
                            egui::Slider::new(&mut self.g_y, -100.0..=100.0)
                                .suffix(" px")
                                .max_decimals(0),
                        );
                        if track_slider(&r, &mut self.dragging) {
                            changed = true;
                        }
                        ui.end_row();
                    });

                ui.add_space(2.0);

                // Blue channel
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new("Blue")
                            .color(Color32::from_rgb(50, 100, 220))
                            .strong()
                            .size(12.0),
                    );
                });
                egui::Grid::new("rgb_b")
                    .num_columns(2)
                    .spacing([8.0, 4.0])
                    .show(ui, |ui| {
                        ui.label("  X");
                        let r = ui.add(
                            egui::Slider::new(&mut self.b_x, -100.0..=100.0)
                                .suffix(" px")
                                .max_decimals(0),
                        );
                        if track_slider(&r, &mut self.dragging) {
                            changed = true;
                        }
                        ui.end_row();
                        ui.label("  Y");
                        let r = ui.add(
                            egui::Slider::new(&mut self.b_y, -100.0..=100.0)
                                .suffix(" px")
                                .max_decimals(0),
                        );
                        if track_slider(&r, &mut self.dragging) {
                            changed = true;
                        }
                        ui.end_row();
                    });

                ui.add_space(4.0);
                section_label(ui, &colors, "PRESETS");
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 4.0;
                    if ui
                        .button(egui::RichText::new("Subtle").size(11.0))
                        .clicked()
                    {
                        self.r_x = 2.0;
                        self.r_y = 0.0;
                        self.g_x = -2.0;
                        self.g_y = 0.0;
                        self.b_x = 0.0;
                        self.b_y = 2.0;
                        changed = true;
                    }
                    if ui.button(egui::RichText::new("VHS").size(11.0)).clicked() {
                        self.r_x = 8.0;
                        self.r_y = 1.0;
                        self.g_x = -4.0;
                        self.g_y = -1.0;
                        self.b_x = 0.0;
                        self.b_y = -3.0;
                        changed = true;
                    }
                    if ui.button(egui::RichText::new("Heavy").size(11.0)).clicked() {
                        self.r_x = 20.0;
                        self.r_y = -5.0;
                        self.g_x = -20.0;
                        self.g_y = 5.0;
                        self.b_x = 5.0;
                        self.b_y = 20.0;
                        changed = true;
                    }
                    if ui.button(egui::RichText::new("Reset").size(11.0)).clicked() {
                        self.r_x = 0.0;
                        self.r_y = 0.0;
                        self.g_x = 0.0;
                        self.g_y = 0.0;
                        self.b_x = 0.0;
                        self.b_y = 0.0;
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
                    result = DialogResult::Ok((
                        (self.r_x as i32, self.r_y as i32),
                        (self.g_x as i32, self.g_y as i32),
                        (self.b_x as i32, self.b_y as i32),
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
// ARTISTIC DIALOGS
// ============================================================================

effect_dialog_base!(InkDialog {
    edge_strength: f32 = 0.0,
    threshold: f32 = 0.3,
    first_open: bool = true
});


