impl InkDialog {
    pub fn show(&mut self, ctx: &egui::Context) -> DialogResult<(f32, f32)> {
        let mut result = DialogResult::Open;
        let colors = DialogColors::from_ctx(ctx);

        crate::ui::polish::window(ctx, "dialog_ink")
            .title_bar(false)
            .collapsible(false)
            .resizable(false)
            .default_pos(egui::pos2(ctx.content_rect().center().x - 175.0, 60.0))
            .show(ctx, |ui| {
                ui.set_min_width(350.0);
                if paint_dialog_header(ui, &colors, "\u{1F58B}", &t!("dialog.ink")) {
                    result = DialogResult::Cancel;
                }
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

        crate::ui::polish::window(ctx, "dialog_oil_painting")
            .title_bar(false)
            .collapsible(false)
            .resizable(false)
            .default_pos(egui::pos2(ctx.content_rect().center().x - 175.0, 60.0))
            .show(ctx, |ui| {
                ui.set_min_width(360.0);
                if paint_dialog_header(ui, &colors, "\u{1F3A8}", &t!("dialog.oil_painting")) {
                    result = DialogResult::Cancel;
                }
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

        crate::ui::polish::window(ctx, "dialog_color_filter")
            .title_bar(false)
            .collapsible(false)
            .resizable(false)
            .default_pos(egui::pos2(ctx.content_rect().center().x - 175.0, 60.0))
            .show(ctx, |ui| {
                ui.set_min_width(380.0);
                if paint_dialog_header(ui, &colors, "\u{1F3AD}", &t!("dialog.color_filter")) {
                    result = DialogResult::Cancel;
                }
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

        crate::ui::polish::window(ctx, "dialog_color_to_alpha")
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
                            ui.add(
                                crate::ui::numeric::Numeric::new(&mut self.sample_x)
                                    .range(0..=max_x),
                            );
                            ui.label("x");
                            ui.add(
                                crate::ui::numeric::Numeric::new(&mut self.sample_y)
                                    .range(0..=max_y),
                            );
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
                        if dialog_slider(ui, &mut self.spill_suppression, 0.0..=1.0, 0.01, "", 2) {
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
                        if dialog_slider(ui, &mut self.protect_luminance, 0.0..=1.0, 0.01, "", 2) {
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

effect_dialog_base!(RecoverTransparencyDialog {
    background_color: [f32; 3] = [1.0, 0.0, 1.0],
    auto_sample_edges: bool = true,
    sample_depth: f32 = 2.0,
    noise_tolerance: f32 = 4.0,
    edge_width: f32 = 2.0,
    eight_connected: bool = false,
    preserve_hard_pixels: bool = true,
    transparent_snap: f32 = 0.025,
    opaque_snap: f32 = 0.97,
    foreground_influence: f32 = 0.8,
    interior_mode: crate::ops::color_removal::InteriorRecoveryMode = crate::ops::color_removal::InteriorRecoveryMode::Off,
    island_max_size: f32 = 24.0,
    island_tolerance: f32 = 0.65,
    island_max_depth: f32 = 12.0,
    bridge_gaps: f32 = 0.0,
    island_x: u32 = 0,
    island_y: u32 = 0,
    remove_seeds: Vec<(u32, u32)> = Vec::new(),
    protect_seeds: Vec<(u32, u32)> = Vec::new(),
    sample_x: u32 = 0,
    sample_y: u32 = 0,
    preview_mode: i32 = 0,
    first_open: bool = true
});

impl RecoverTransparencyDialog {
    pub fn new_with_target(state: &CanvasState, target: Color32) -> Self {
        let mut dlg = Self::new(state);
        dlg.background_color = [
            target.r() as f32 / 255.0,
            target.g() as f32 / 255.0,
            target.b() as f32 / 255.0,
        ];
        dlg
    }

    pub fn settings(&self) -> crate::ops::color_removal::RecoverTransparencySettings {
        crate::ops::color_removal::RecoverTransparencySettings {
            background: [
                (self.background_color[0] * 255.0).round().clamp(0.0, 255.0) as u8,
                (self.background_color[1] * 255.0).round().clamp(0.0, 255.0) as u8,
                (self.background_color[2] * 255.0).round().clamp(0.0, 255.0) as u8,
            ],
            auto_sample_edges: self.auto_sample_edges,
            sample_depth: self.sample_depth.round().clamp(1.0, 8.0) as u32,
            noise_tolerance: self.noise_tolerance,
            edge_width: self.edge_width.round().clamp(0.0, 6.0) as u32,
            eight_connected: self.eight_connected,
            preserve_hard_pixels: self.preserve_hard_pixels,
            transparent_snap: self.transparent_snap,
            opaque_snap: self.opaque_snap,
            foreground_influence: self.foreground_influence,
            interior_mode: self.interior_mode,
            island_max_size: self.island_max_size.round().clamp(1.0, 4096.0) as u32,
            island_tolerance: self.island_tolerance,
            island_max_depth: self.island_max_depth.round().clamp(1.0, 256.0) as u32,
            bridge_gaps: self.bridge_gaps.round().clamp(0.0, 4.0) as u32,
            remove_seeds: self.remove_seeds.clone(),
            protect_seeds: self.protect_seeds.clone(),
        }
    }

    pub fn preview_kind(&self) -> crate::ops::color_removal::RecoverTransparencyPreview {
        match self.preview_mode {
            1 => crate::ops::color_removal::RecoverTransparencyPreview::Alpha,
            2 => crate::ops::color_removal::RecoverTransparencyPreview::ReconstructionError,
            _ => crate::ops::color_removal::RecoverTransparencyPreview::Result,
        }
    }

    pub fn show(
        &mut self,
        ctx: &egui::Context,
        icon_texture: Option<&egui::TextureHandle>,
    ) -> DialogResult<crate::ops::color_removal::RecoverTransparencySettings> {
        let mut result = DialogResult::Open;
        let colors = DialogColors::from_ctx(ctx);

        crate::ui::polish::window(ctx, "dialog_recover_transparency")
            .title_bar(false)
            .collapsible(false)
            .resizable(true)
            .default_size(egui::vec2(460.0, 600.0))
            .max_size(ctx.content_rect().size() - egui::vec2(24.0, 24.0))
            .default_pos(egui::pos2(ctx.content_rect().center().x - 220.0, 48.0))
            .show(ctx, |ui| {
                ui.set_min_width(440.0);
                let close_clicked = if icon_texture.is_some() {
                    paint_dialog_header_with_texture(
                        ui,
                        &colors,
                        icon_texture,
                        &t!("dialog.recover_transparency"),
                    )
                } else {
                    paint_dialog_header(
                        ui,
                        &colors,
                        crate::assets::Icon::ColorRemover.emoji(),
                        &t!("dialog.recover_transparency"),
                    )
                };
                if close_clicked {
                    result = DialogResult::Cancel;
                }

                egui::ScrollArea::vertical()
                    .id_salt("recover_transparency_body")
                    .max_height((ui.available_height() - 52.0).max(100.0))
                    .show(ui, |ui| {
                let mut changed = false;
                ui.add_space(4.0);
                ui.label("Recover alpha and edge colors from artwork flattened over a noisy solid background.");
                ui.add_space(4.0);
                section_label(ui, &colors, "BACKGROUND MODEL");
                egui::Grid::new("recover_transparency_background")
                    .num_columns(2)
                    .spacing([8.0, 6.0])
                    .show(ui, |ui| {
                        ui.label("Automatic");
                        if ui
                            .checkbox(&mut self.auto_sample_edges, "Estimate from canvas edges")
                            .changed()
                        {
                            changed = true;
                        }
                        ui.end_row();

                        ui.label("Fallback / Manual");
                        if ui.color_edit_button_rgb(&mut self.background_color).changed() {
                            self.auto_sample_edges = false;
                            changed = true;
                        }
                        ui.end_row();

                        ui.label("Sample Pixel");
                        ui.horizontal(|ui| {
                            let max_x = self
                                .original_flat
                                .as_ref()
                                .map_or(0, |img| img.width().saturating_sub(1));
                            let max_y = self
                                .original_flat
                                .as_ref()
                                .map_or(0, |img| img.height().saturating_sub(1));
                            ui.add(crate::ui::numeric::Numeric::new(&mut self.sample_x).range(0..=max_x));
                            ui.label("x");
                            ui.add(crate::ui::numeric::Numeric::new(&mut self.sample_y).range(0..=max_y));
                            if ui.small_button("Pick").clicked()
                                && let Some(flat) = &self.original_flat
                            {
                                let p = flat.get_pixel(
                                    self.sample_x.min(flat.width().saturating_sub(1)),
                                    self.sample_y.min(flat.height().saturating_sub(1)),
                                );
                                self.background_color = [
                                    p[0] as f32 / 255.0,
                                    p[1] as f32 / 255.0,
                                    p[2] as f32 / 255.0,
                                ];
                                self.auto_sample_edges = false;
                                changed = true;
                            }
                        });
                        ui.end_row();

                        ui.label("Edge Sample Depth");
                        if dialog_slider(ui, &mut self.sample_depth, 1.0..=8.0, 1.0, " px", 0) {
                            changed = true;
                        }
                        ui.end_row();

                        ui.label("Noise Tolerance");
                        if dialog_slider(ui, &mut self.noise_tolerance, 0.5..=32.0, 0.5, "", 1) {
                            changed = true;
                        }
                        ui.end_row();

                        ui.label("Connectivity");
                        egui::ComboBox::from_id_salt("recover_connectivity")
                            .selected_text(if self.eight_connected { "8-way" } else { "4-way (pixel art)" })
                            .show_ui(ui, |ui| {
                                changed |= ui.selectable_value(&mut self.eight_connected, false, "4-way (pixel art)").changed();
                                changed |= ui.selectable_value(&mut self.eight_connected, true, "8-way").changed();
                            });
                        ui.end_row();
                    });

                ui.add_space(4.0);
                section_label(ui, &colors, "EDGE RECOVERY");
                egui::Grid::new("recover_transparency_edges")
                    .num_columns(2)
                    .spacing([8.0, 6.0])
                    .show(ui, |ui| {
                        ui.label("Recover Band");
                        if dialog_slider(ui, &mut self.edge_width, 0.0..=6.0, 1.0, " px", 0) {
                            changed = true;
                        }
                        ui.end_row();

                        ui.label("Preserve Hard Pixels");
                        if ui.checkbox(&mut self.preserve_hard_pixels, "Snap near-opaque edges").changed() {
                            changed = true;
                        }
                        ui.end_row();

                        ui.label("Transparent Snap");
                        if dialog_slider(ui, &mut self.transparent_snap, 0.0..=0.25, 0.005, "", 3) {
                            self.opaque_snap = self.opaque_snap.max(self.transparent_snap);
                            changed = true;
                        }
                        ui.end_row();

                        ui.label("Opaque Snap");
                        if dialog_slider(ui, &mut self.opaque_snap, 0.75..=1.0, 0.005, "", 3) {
                            self.transparent_snap = self.transparent_snap.min(self.opaque_snap);
                            changed = true;
                        }
                        ui.end_row();

                        ui.label("Color Preservation");
                        if dialog_slider(ui, &mut self.foreground_influence, 0.0..=1.0, 0.05, "", 2) {
                            changed = true;
                        }
                        ui.end_row();
                    });

                ui.add_space(4.0);
                section_label(ui, &colors, "INTERIOR BACKGROUND RECOVERY");
                egui::Grid::new("recover_transparency_islands")
                    .num_columns(2)
                    .spacing([8.0, 6.0])
                    .show(ui, |ui| {
                        use crate::ops::color_removal::InteriorRecoveryMode;
                        ui.label("Enclosed Regions");
                        egui::ComboBox::from_id_salt("recover_island_mode")
                            .selected_text(match self.interior_mode {
                                InteriorRecoveryMode::Off => "Off (safest)",
                                InteriorRecoveryMode::SmallIslands => "Small Islands",
                                InteriorRecoveryMode::NearExterior => "Near Exterior",
                                InteriorRecoveryMode::AllMatching => "All Matching",
                            })
                            .show_ui(ui, |ui| {
                                changed |= ui.selectable_value(&mut self.interior_mode, InteriorRecoveryMode::Off, "Off (safest)").changed();
                                changed |= ui.selectable_value(&mut self.interior_mode, InteriorRecoveryMode::SmallIslands, "Small Islands").changed();
                                changed |= ui.selectable_value(&mut self.interior_mode, InteriorRecoveryMode::NearExterior, "Near Exterior").changed();
                                changed |= ui.selectable_value(&mut self.interior_mode, InteriorRecoveryMode::AllMatching, "All Matching").changed();
                            });
                        ui.end_row();

                        ui.label("Maximum Island Size");
                        if dialog_slider(ui, &mut self.island_max_size, 1.0..=256.0, 1.0, " px", 0) {
                            changed = true;
                        }
                        ui.end_row();

                        ui.label("Island Strictness");
                        if dialog_slider(ui, &mut self.island_tolerance, 0.1..=1.25, 0.05, "", 2) {
                            changed = true;
                        }
                        ui.end_row();

                        ui.label("Maximum Depth");
                        if dialog_slider(ui, &mut self.island_max_depth, 1.0..=64.0, 1.0, " px", 0) {
                            changed = true;
                        }
                        ui.end_row();

                        ui.label("Bridge Tiny Gaps");
                        if dialog_slider(ui, &mut self.bridge_gaps, 0.0..=4.0, 1.0, " px", 0) {
                            changed = true;
                        }
                        ui.end_row();

                        ui.label("Manual Component Seed");
                        ui.horizontal(|ui| {
                            let max_x = self.original_flat.as_ref().map_or(0, |img| img.width().saturating_sub(1));
                            let max_y = self.original_flat.as_ref().map_or(0, |img| img.height().saturating_sub(1));
                            ui.add(crate::ui::numeric::Numeric::new(&mut self.island_x).range(0..=max_x));
                            ui.label("x");
                            ui.add(crate::ui::numeric::Numeric::new(&mut self.island_y).range(0..=max_y));
                        });
                        ui.end_row();

                        ui.label("Component Override");
                        ui.horizontal(|ui| {
                            if ui.small_button("Remove").clicked() {
                                let seed = (self.island_x, self.island_y);
                                self.protect_seeds.retain(|&p| p != seed);
                                if !self.remove_seeds.contains(&seed) {
                                    self.remove_seeds.push(seed);
                                }
                                changed = true;
                            }
                            if ui.small_button("Protect").clicked() {
                                let seed = (self.island_x, self.island_y);
                                self.remove_seeds.retain(|&p| p != seed);
                                if !self.protect_seeds.contains(&seed) {
                                    self.protect_seeds.push(seed);
                                }
                                changed = true;
                            }
                            if ui.small_button("Clear").clicked() {
                                self.remove_seeds.clear();
                                self.protect_seeds.clear();
                                changed = true;
                            }
                        });
                        ui.end_row();
                    });
                ui.small(format!(
                    "Manual overrides: {} remove, {} protect. Interior strictness is relative to the measured background noise.",
                    self.remove_seeds.len(),
                    self.protect_seeds.len()
                ));

                ui.add_space(4.0);
                section_label(ui, &colors, "DIAGNOSTIC PREVIEW");
                ui.horizontal(|ui| {
                    ui.label("View");
                    egui::ComboBox::from_id_salt("recover_preview_mode")
                        .selected_text(match self.preview_mode {
                            1 => "Alpha Matte",
                            2 => "Reconstruction Error",
                            _ => "Recovered Result",
                        })
                        .show_ui(ui, |ui| {
                            changed |= ui.selectable_value(&mut self.preview_mode, 0, "Recovered Result").changed();
                            changed |= ui.selectable_value(&mut self.preview_mode, 1, "Alpha Matte").changed();
                            changed |= ui.selectable_value(&mut self.preview_mode, 2, "Reconstruction Error").changed();
                        });
                });
                ui.small("Reconstruction Error is amplified 8×; black means the recovered pixels reproduce the source backing.");

                accent_separator(ui, &colors);
                let manual = preview_controls(ui, &colors, &mut self.live_preview);
                if (changed && self.live_preview) || manual {
                    result = DialogResult::Changed;
                }
                    });
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

        crate::ui::polish::window(ctx, "dialog_contours")
            .title_bar(false)
            .collapsible(false)
            .resizable(false)
            .default_pos(egui::pos2(ctx.content_rect().center().x - 190.0, 60.0))
            .show(ctx, |ui| {
                ui.set_min_width(400.0);
                if paint_dialog_header(ui, &colors, "\u{1F5FA}", &t!("dialog.contours")) {
                    result = DialogResult::Cancel;
                }
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
