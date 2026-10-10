pub struct ResizeImageDialog {
    pub width: f32,
    pub height: f32,
    width_input: String,
    height_input: String,
    pub scale_percent: f32,
    pub lock_aspect: bool,
    aspect_ratio: f32,
    pub interpolation: Interpolation,
    pub resize_method: ResizeMethod,
    pub retarget: PixelRetargetSettings,
    pub preset: ResizePreset,
    original_w: u32,
    original_h: u32,
    focus_width_on_open: bool,
    replace_width_on_first_edit: bool,
    preview_source: Option<TiledImage>,
    comparison_previews: Vec<(String, egui::TextureHandle)>,
}

#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum ResizePreset {
    #[default]
    Custom,
    Hd1920x1080,
    Square1080,
    Uhd4K,
    A4At300Dpi,
}

impl ResizePreset {
    pub fn label(&self) -> String {
        match self {
            ResizePreset::Custom => t!("resize_preset.custom"),
            ResizePreset::Hd1920x1080 => t!("resize_preset.hd"),
            ResizePreset::Square1080 => t!("resize_preset.square"),
            ResizePreset::Uhd4K => t!("resize_preset.uhd4k"),
            ResizePreset::A4At300Dpi => t!("resize_preset.a4_300dpi"),
        }
    }
    pub fn dims(&self) -> Option<(u32, u32)> {
        match self {
            ResizePreset::Custom => None,
            ResizePreset::Hd1920x1080 => Some((1920, 1080)),
            ResizePreset::Square1080 => Some((1080, 1080)),
            ResizePreset::Uhd4K => Some((3840, 2160)),
            ResizePreset::A4At300Dpi => Some((2480, 3508)),
        }
    }
    pub fn all() -> &'static [ResizePreset] {
        &[
            ResizePreset::Custom,
            ResizePreset::Hd1920x1080,
            ResizePreset::Square1080,
            ResizePreset::Uhd4K,
            ResizePreset::A4At300Dpi,
        ]
    }
}

impl ResizeImageDialog {
    pub fn new(state: &CanvasState) -> Self {
        Self {
            width: state.width as f32,
            height: state.height as f32,
            width_input: format_dimension_value(state.width as f32),
            height_input: format_dimension_value(state.height as f32),
            scale_percent: 100.0,
            lock_aspect: true,
            aspect_ratio: state.width as f32 / state.height.max(1) as f32,
            interpolation: Interpolation::default(),
            resize_method: ResizeMethod::Standard,
            retarget: PixelRetargetSettings::default(),
            preset: ResizePreset::default(),
            original_w: state.width,
            original_h: state.height,
            focus_width_on_open: true,
            replace_width_on_first_edit: true,
            preview_source: state
                .layers
                .get(state.active_layer_index)
                .map(|l| l.pixels.clone()),
            comparison_previews: Vec::new(),
        }
    }

    fn sync_inputs_from_values(&mut self) {
        self.width_input = format_dimension_value(self.width);
        self.height_input = format_dimension_value(self.height);
    }

    fn commit_width_input(&mut self) {
        let old_width = self.width;
        if let Some(new_width) = evaluate_dimension_expression(&self.width_input) {
            self.width = new_width.round().clamp(1.0, 20000.0);
            self.width_input = format_dimension_value(self.width);
            self.preset = ResizePreset::Custom;
            if self.lock_aspect && old_width > 0.0 {
                self.height = (self.width / self.aspect_ratio).round().clamp(1.0, 20000.0);
                self.height_input = format_dimension_value(self.height);
            } else {
                self.aspect_ratio = self.width / self.height.max(1.0);
            }
            self.scale_percent = self.width / self.original_w.max(1) as f32 * 100.0;
        } else {
            self.width_input = format_dimension_value(self.width);
        }
    }

    fn commit_height_input(&mut self) {
        let old_height = self.height;
        if let Some(new_height) = evaluate_dimension_expression(&self.height_input) {
            self.height = new_height.round().clamp(1.0, 20000.0);
            self.height_input = format_dimension_value(self.height);
            self.preset = ResizePreset::Custom;
            if self.lock_aspect && old_height > 0.0 {
                self.width = (self.height * self.aspect_ratio)
                    .round()
                    .clamp(1.0, 20000.0);
                self.width_input = format_dimension_value(self.width);
            } else {
                self.aspect_ratio = self.width / self.height.max(1.0);
            }
            self.scale_percent = self.height / self.original_h.max(1) as f32 * 100.0;
        } else {
            self.height_input = format_dimension_value(self.height);
        }
    }

    fn commit_inputs(&mut self) {
        self.commit_width_input();
        self.commit_height_input();
    }

    fn generate_comparison_previews(&mut self, ctx: &egui::Context) {
        let Some(source) = &self.preview_source else {
            return;
        };
        let mut flat = source.to_rgba_image();
        let max_dim = flat.width().max(flat.height());
        if max_dim > 192 {
            let scale = 192.0 / max_dim as f32;
            let w = (flat.width() as f32 * scale).round().max(1.0) as u32;
            let h = (flat.height() as f32 * scale).round().max(1.0) as u32;
            flat = image::imageops::resize(&flat, w, h, image::imageops::FilterType::Nearest);
        }
        let ratio_x = self.width / self.original_w.max(1) as f32;
        let ratio_y = self.height / self.original_h.max(1) as f32;
        let target_w = (flat.width() as f32 * ratio_x).round().max(1.0) as u32;
        let target_h = (flat.height() as f32 * ratio_y).round().max(1.0) as u32;
        let nearest = image::imageops::resize(
            &flat,
            target_w,
            target_h,
            image::imageops::FilterType::Nearest,
        );
        let mut candidates = vec![("Nearest".to_string(), nearest)];
        for (label, mode) in [
            ("Palette", PixelRetargetMode::PaletteAware),
            ("Structure", PixelRetargetMode::StructurePreserving),
            ("Topology", PixelRetargetMode::TopologyPreserving),
        ] {
            let mut settings = self.retarget;
            settings.mode = mode;
            settings.optimize_phase = false;
            candidates.push((
                label.to_string(),
                crate::ops::transform::pixel_art_retarget(
                    &flat, target_w, target_h, &settings, None,
                ),
            ));
        }
        self.comparison_previews = candidates
            .into_iter()
            .enumerate()
            .map(|(index, (label, image))| {
                let color_image = egui::ColorImage::from_rgba_unmultiplied(
                    [image.width() as usize, image.height() as usize],
                    image.as_raw(),
                );
                let texture = ctx.load_texture(
                    format!("retarget_preview_{index}"),
                    color_image,
                    egui::TextureOptions::NEAREST,
                );
                (label, texture)
            })
            .collect();
    }

    pub fn show(&mut self, ctx: &egui::Context) -> DialogResult<(u32, u32, ResizeImageOptions)> {
        let mut result = DialogResult::Open;
        let colors = DialogColors::from_ctx(ctx);
        let mut ok_pressed = false;

        crate::ui::polish::window(ctx, "dialog_resize_image")
            .title_bar(false)
            .collapsible(false)
            .resizable(false)
            .default_pos(egui::pos2(
                ctx.content_rect().center().x - 175.0,
                ctx.content_rect().center().y - 160.0,
            ))
            .show(ctx, |ui| {
                ui.set_min_width(350.0);

                if paint_dialog_header(ui, &colors, "\u{1F4D0}", &t!("dialog.resize_image")) {
                    result = DialogResult::Cancel;
                }
                ui.add_space(4.0);

                // -- Preset (own grid so it doesn't misalign the dims columns) --
                section_label(ui, &colors, &t!("dialog.resize_image.dimensions"));

                egui::Grid::new("resize_img_preset")
                    .num_columns(2)
                    .min_col_width(80.0)
                    .spacing([8.0, 4.0])
                    .show(ui, |ui| {
                        ui.label(t!("dialog.resize_image.preset"));
                        let preset_was_open = egui::ComboBox::is_open(ctx, ui.make_persistent_id(egui::IdSalt::new("resize_preset")));
                        egui::ComboBox::from_id_salt("resize_preset")
                            .width(210.0)
                            .selected_text(self.preset.label())
                            .show_ui(ui, |ui| {
                                let choices = ResizePreset::all();
                                let choice_id = ui.id().with("keyboard_choice");
                                let mut choice = ctx.data(|d| d.get_temp::<usize>(choice_id))
                                    .filter(|_| preset_was_open)
                                    .unwrap_or_else(|| choices.iter().position(|p| *p == self.preset).unwrap_or(0));
                                let (down, up, enter) = if preset_was_open {
                                    ui.input_mut(|i| (
                                        i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowDown),
                                        i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowUp),
                                        i.consume_key(egui::Modifiers::NONE, egui::Key::Enter),
                                    ))
                                } else { (false, false, false) };
                                if down { choice = (choice + 1).min(choices.len() - 1); }
                                if up { choice = choice.saturating_sub(1); }
                                if down || up {
                                    ui.memory_mut(|memory| memory.move_focus(egui::FocusDirection::None));
                                }
                                let mut picked = enter.then_some(choice);
                                for (index, p) in choices.iter().enumerate() {
                                    let response = ui.selectable_label(index == choice, p.label());
                                    if index == choice && (!preset_was_open || down || up) {
                                        response.request_focus();
                                    }
                                    if response.clicked() { picked = Some(index); }
                                }
                                ctx.data_mut(|d| d.insert_temp(choice_id, choice));
                                if let Some(index) = picked {
                                    self.preset = choices[index];
                                    ui.close();
                                    if let Some((w, h)) = self.preset.dims() {
                                        self.width = w as f32;
                                        self.height = h as f32;
                                        self.scale_percent =
                                            self.width / self.original_w.max(1) as f32 * 100.0;
                                        self.aspect_ratio = self.width / self.height.max(1.0);
                                        self.sync_inputs_from_values();
                                    }
                                }
                            });
                        ui.end_row();
                    });

                ui.add_space(2.0);

                // -- Width / Height / Lock / Scale grid --
                egui::Grid::new("resize_img_dims")
                    .num_columns(3)
                    .min_col_width(80.0)
                    .spacing([8.0, 5.0])
                    .show(ui, |ui| {
                        // Width row
                        ui.label(t!("dialog.resize_image.width"));
                        ui.horizontal(|ui| {
                            ui.spacing_mut().item_spacing.x = 4.0;
                            let previous_input = self.width_input.clone();
                            let width_response = ui.add(
                                egui::TextEdit::singleline(&mut self.width_input)
                                    .desired_width(96.0),
                            );
                            if self.focus_width_on_open {
                                width_response.request_focus();
                                if let Some(mut state) = egui::TextEdit::load_state(ctx, width_response.id) {
                                    state.cursor.set_char_range(Some(egui::text::CCursorRange::two(
                                        egui::text::CCursor::new(0),
                                        egui::text::CCursor::new(self.width_input.chars().count()),
                                    )));
                                    state.store(ctx, width_response.id);
                                }
                                self.focus_width_on_open = false;
                                self.replace_width_on_first_edit = false;
                            }
                            if self.replace_width_on_first_edit && width_response.changed() {
                                if self.width_input.starts_with(&previous_input)
                                    && self.width_input.len() > previous_input.len()
                                {
                                    let suffix = &self.width_input[previous_input.len()..];
                                    let suffix_trimmed = suffix.trim_start();
                                    if !matches!(
                                        suffix_trimmed.chars().next(),
                                        Some('+') | Some('-') | Some('*') | Some('/')
                                    ) {
                                        self.width_input = suffix.to_string();
                                    }
                                }
                                self.replace_width_on_first_edit = false;
                            }
                            let width_commit = width_response.lost_focus()
                                || (width_response.has_focus()
                                    && ui.input(|i| i.key_pressed(egui::Key::Tab)));
                            if width_commit {
                                self.commit_width_input();
                            }
                            let drag = ui.add(egui::Label::new("px ↔").sense(egui::Sense::DRAG))
                                .on_hover_text("Drag to change width; click the number to enter a value or expression.");
                            if drag.dragged() {
                                self.width_input = (self.width + drag.drag_delta().x).round().clamp(1.0, 32768.0).to_string();
                                self.commit_width_input();
                            }
                        });
                        ui.end_row();

                        // Height row
                        ui.label(t!("dialog.resize_image.height"));
                        ui.horizontal(|ui| {
                            ui.spacing_mut().item_spacing.x = 4.0;
                            let height_response = ui.add(
                                egui::TextEdit::singleline(&mut self.height_input)
                                    .desired_width(96.0),
                            );
                            if height_response.gained_focus()
                                && let Some(mut state) = egui::TextEdit::load_state(ctx, height_response.id)
                            {
                                state.cursor.set_char_range(Some(egui::text::CCursorRange::two(
                                    egui::text::CCursor::new(0), egui::text::CCursor::new(self.height_input.chars().count()))));
                                state.store(ctx, height_response.id);
                            }
                            let height_commit = height_response.lost_focus()
                                || (height_response.has_focus()
                                    && ui.input(|i| i.key_pressed(egui::Key::Tab)));
                            if height_commit {
                                self.commit_height_input();
                            }
                            let drag = ui.add(egui::Label::new("px ↔").sense(egui::Sense::DRAG))
                                .on_hover_text("Drag to change height; click the number to enter a value or expression.");
                            if drag.dragged() {
                                self.height_input = (self.height + drag.drag_delta().x).round().clamp(1.0, 32768.0).to_string();
                                self.commit_height_input();
                            }
                        });
                        ui.end_row();

                        // Lock aspect ratio (inline between H and Scale)
                        ui.label("");
                        let lock_resp = ui
                            .checkbox(&mut self.lock_aspect, t!("dialog.resize_image.lock_aspect"));
                        if lock_resp.changed() && self.lock_aspect {
                            self.aspect_ratio = self.width / self.height.max(1.0);
                        }
                        ui.label("");
                        ui.end_row();

                        // Scale row
                        ui.label(t!("dialog.resize_image.scale"));
                        let s_changed = numeric_field_with_buttons(
                            ui,
                            &mut self.scale_percent,
                            0.5,
                            1.0..=10000.0,
                            "%",
                            5.0,
                        );
                        ui.label("");
                        ui.end_row();

                        if s_changed {
                            self.preset = ResizePreset::Custom;
                            self.width =
                                (self.original_w as f32 * self.scale_percent / 100.0).round();
                            self.height =
                                (self.original_h as f32 * self.scale_percent / 100.0).round();
                            self.aspect_ratio = self.width / self.height.max(1.0);
                            self.sync_inputs_from_values();
                        }
                    });

                // -- Quality section --
                accent_separator(ui, &colors);
                section_label(ui, &colors, &t!("dialog.resize_image.quality"));

                egui::Grid::new("resize_img_quality")
                    .num_columns(2)
                    .spacing([8.0, 4.0])
                    .show(ui, |ui| {
                        ui.label("Resize Method");
                        egui::ComboBox::from_id_salt("resize_method")
                            .width(220.0)
                            .selected_text(self.resize_method.label())
                            .show_ui(ui, |ui| {
                                ui.selectable_value(
                                    &mut self.resize_method,
                                    ResizeMethod::Standard,
                                    ResizeMethod::Standard.label(),
                                );
                                ui.selectable_value(
                                    &mut self.resize_method,
                                    ResizeMethod::PixelArtRetarget,
                                    ResizeMethod::PixelArtRetarget.label(),
                                );
                            });
                        ui.end_row();

                        if self.resize_method == ResizeMethod::Standard {
                            ui.label(t!("dialog.resize_image.interpolation"));
                            egui::ComboBox::from_id_salt("resize_interp")
                                .width(220.0)
                                .selected_text(self.interpolation.label())
                                .show_ui(ui, |ui| {
                                    for i in Interpolation::all() {
                                        ui.selectable_value(&mut self.interpolation, *i, i.label());
                                    }
                                });
                            ui.end_row();
                        } else {
                            ui.label("Retarget Mode");
                            egui::ComboBox::from_id_salt("pixel_retarget_mode")
                                .width(220.0)
                                .selected_text(self.retarget.mode.label())
                                .show_ui(ui, |ui| {
                                    ui.selectable_value(
                                        &mut self.retarget.mode,
                                        PixelRetargetMode::PaletteAware,
                                        PixelRetargetMode::PaletteAware.label(),
                                    );
                                    ui.selectable_value(
                                        &mut self.retarget.mode,
                                        PixelRetargetMode::StructurePreserving,
                                        PixelRetargetMode::StructurePreserving.label(),
                                    );
                                    ui.selectable_value(
                                        &mut self.retarget.mode,
                                        PixelRetargetMode::TopologyPreserving,
                                        PixelRetargetMode::TopologyPreserving.label(),
                                    );
                                });
                            ui.end_row();
                        }
                    });

                if self.resize_method == ResizeMethod::PixelArtRetarget {
                    ui.add_space(4.0);
                    egui::Grid::new("pixel_retarget_options")
                        .num_columns(2)
                        .spacing([8.0, 4.0])
                        .show(ui, |ui| {
                            ui.label("Palette");
                            ui.checkbox(&mut self.retarget.preserve_palette, "Preserve source colors");
                            ui.end_row();

                            ui.label("Silhouette");
                            ui.checkbox(&mut self.retarget.preserve_silhouette, "Protect alpha edges");
                            ui.end_row();

                            ui.label("Thin Features");
                            ui.checkbox(&mut self.retarget.protect_thin_features, "Protect one-pixel details");
                            ui.end_row();

                            ui.label("Connectivity");
                            egui::ComboBox::from_id_salt("retarget_connectivity")
                                .selected_text(if self.retarget.eight_connected { "8-way" } else { "4-way" })
                                .show_ui(ui, |ui| {
                                    ui.selectable_value(&mut self.retarget.eight_connected, false, "4-way");
                                    ui.selectable_value(&mut self.retarget.eight_connected, true, "8-way");
                                });
                            ui.end_row();

                            ui.label("Detail Priority");
                            dialog_slider(ui, &mut self.retarget.detail_priority, 0.0..=1.0, 0.05, "", 2);
                            ui.end_row();

                            ui.label("Geometry / Topology");
                            ui.horizontal(|ui| {
                                ui.label("Geometry");
                                ui.add(egui::Slider::new(&mut self.retarget.topology_priority, 0.0..=1.0).show_value(false));
                                ui.label("Topology");
                            });
                            ui.end_row();

                            ui.label("Sampling Phase");
                            ui.checkbox(&mut self.retarget.optimize_phase, "Optimize automatically");
                            ui.end_row();

                            ui.label("Protection Mask");
                            ui.checkbox(&mut self.retarget.use_selection_as_protection, "Use current selection");
                            ui.end_row();
                        });

                    let target_ratio = (self.width / self.original_w.max(1) as f32)
                        .min(self.height / self.original_h.max(1) as f32);
                    if target_ratio < 0.5 {
                        ui.colored_label(
                            Color32::from_rgb(220, 145, 45),
                            "Severe reduction: some detail must be simplified. Use a protection selection for critical pixels.",
                        );
                    } else if self.width > self.original_w as f32
                        || self.height > self.original_h as f32
                    {
                        ui.colored_label(
                            colors.text_muted,
                            "Retarget is designed for reduction; enlargement uses nearest-neighbor pixels.",
                        );
                    }

                    ui.add_space(4.0);
                    if ui.button("Generate Comparison Preview").clicked() {
                        self.generate_comparison_previews(ctx);
                    }
                    if !self.comparison_previews.is_empty() {
                        ui.horizontal_wrapped(|ui| {
                            for (label, texture) in &self.comparison_previews {
                                ui.vertical(|ui| {
                                    ui.label(egui::RichText::new(label).size(10.0));
                                    ui.add(
                                        egui::Image::new(texture)
                                            .fit_to_exact_size(egui::vec2(80.0, 80.0)),
                                    );
                                });
                            }
                        });
                    }
                }

                // -- Info bar --
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    ui.add_space(4.0);
                    let new_w = self.width.round() as u32;
                    let new_h = self.height.round() as u32;
                    let info = format!(
                        "{}x{} \u{2192} {}x{}",
                        self.original_w, self.original_h, new_w, new_h
                    );
                    ui.label(
                        egui::RichText::new(info)
                            .size(11.0)
                            .color(colors.text_muted),
                    );
                });

                // -- Footer --
                let (ok, cancel) = dialog_footer(ui, &colors);
                if ok {
                    ok_pressed = true;
                }
                if cancel {
                    result = DialogResult::Cancel;
                }
            });

        if ok_pressed && matches!(result, DialogResult::Open) {
            self.commit_inputs();
            let w = (self.width.round() as u32).max(1);
            let h = (self.height.round() as u32).max(1);
            result = DialogResult::Ok((
                w,
                h,
                ResizeImageOptions {
                    method: self.resize_method,
                    interpolation: self.interpolation,
                    retarget: self.retarget,
                },
            ));
        }

        result
    }
}

// ============================================================================
// RESIZE CANVAS DIALOG
// ============================================================================

pub struct ResizeCanvasDialog {
    pub width: f32,
    pub height: f32,
    width_input: String,
    height_input: String,
    pub scale_percent: f32,
    pub lock_aspect: bool,
    aspect_ratio: f32,
    /// Anchor as (col, row) each 0..=2.
    pub anchor: (u32, u32),
    pub fill_transparent: bool,
    original_w: u32,
    original_h: u32,
    focus_width_on_open: bool,
    replace_width_on_first_edit: bool,
}

impl ResizeCanvasDialog {
    pub fn new(state: &CanvasState) -> Self {
        Self {
            width: state.width as f32,
            height: state.height as f32,
            width_input: format_dimension_value(state.width as f32),
            height_input: format_dimension_value(state.height as f32),
            scale_percent: 100.0,
            lock_aspect: true,
            aspect_ratio: state.width as f32 / state.height.max(1) as f32,
            anchor: (1, 1), // center
            fill_transparent: true,
            original_w: state.width,
            original_h: state.height,
            focus_width_on_open: true,
            replace_width_on_first_edit: true,
        }
    }

    fn sync_inputs_from_values(&mut self) {
        self.width_input = format_dimension_value(self.width);
        self.height_input = format_dimension_value(self.height);
    }

    fn commit_width_input(&mut self) {
        let old_width = self.width;
        if let Some(new_width) = evaluate_dimension_expression(&self.width_input) {
            self.width = new_width.round().clamp(1.0, 20000.0);
            self.width_input = format_dimension_value(self.width);
            if self.lock_aspect && old_width > 0.0 {
                self.height = (self.width / self.aspect_ratio).round().clamp(1.0, 20000.0);
                self.height_input = format_dimension_value(self.height);
            } else {
                self.aspect_ratio = self.width / self.height.max(1.0);
            }
            self.scale_percent = self.width / self.original_w.max(1) as f32 * 100.0;
        } else {
            self.width_input = format_dimension_value(self.width);
        }
    }

    fn commit_height_input(&mut self) {
        let old_height = self.height;
        if let Some(new_height) = evaluate_dimension_expression(&self.height_input) {
            self.height = new_height.round().clamp(1.0, 20000.0);
            self.height_input = format_dimension_value(self.height);
            if self.lock_aspect && old_height > 0.0 {
                self.width = (self.height * self.aspect_ratio)
                    .round()
                    .clamp(1.0, 20000.0);
                self.width_input = format_dimension_value(self.width);
            } else {
                self.aspect_ratio = self.width / self.height.max(1.0);
            }
            self.scale_percent = self.height / self.original_h.max(1) as f32 * 100.0;
        } else {
            self.height_input = format_dimension_value(self.height);
        }
    }

    fn commit_inputs(&mut self) {
        self.commit_width_input();
        self.commit_height_input();
    }

    pub fn show(
        &mut self,
        ctx: &egui::Context,
        secondary_color: [f32; 4],
    ) -> DialogResult<(u32, u32, (u32, u32), Rgba<u8>)> {
        let mut result = DialogResult::Open;
        let colors = DialogColors::from_ctx(ctx);
        let mut ok_pressed = false;

        crate::ui::polish::window(ctx, "dialog_resize_canvas")
            .title_bar(false)
            .collapsible(false)
            .resizable(false)
            .default_pos(egui::pos2(
                ctx.content_rect().center().x - 170.0,
                ctx.content_rect().center().y - 180.0,
            ))
            .show(ctx, |ui| {
                ui.set_min_width(340.0);

                if paint_dialog_header(ui, &colors, "\u{1F532}", &t!("dialog.resize_canvas")) {
                    result = DialogResult::Cancel;
                }
                ui.add_space(4.0);

                // -- Dimensions section --
                section_label(ui, &colors, &t!("dialog.resize_canvas.canvas_size"));

                egui::Grid::new("resize_canvas_dims")
                    .num_columns(3)
                    .spacing([8.0, 4.0])
                    .show(ui, |ui| {
                        ui.label(t!("dialog.resize_image.width"));
                        ui.horizontal(|ui| {
                            ui.spacing_mut().item_spacing.x = 4.0;
                            let previous_input = self.width_input.clone();
                            let width_response = ui.add(
                                egui::TextEdit::singleline(&mut self.width_input)
                                    .desired_width(96.0),
                            );
                            if self.focus_width_on_open {
                                width_response.request_focus();
                                self.focus_width_on_open = false;
                            }
                            if self.replace_width_on_first_edit && width_response.changed() {
                                if self.width_input.starts_with(&previous_input)
                                    && self.width_input.len() > previous_input.len()
                                {
                                    let suffix = &self.width_input[previous_input.len()..];
                                    let suffix_trimmed = suffix.trim_start();
                                    if !matches!(
                                        suffix_trimmed.chars().next(),
                                        Some('+') | Some('-') | Some('*') | Some('/')
                                    ) {
                                        self.width_input = suffix.to_string();
                                    }
                                }
                                self.replace_width_on_first_edit = false;
                            }
                            let width_commit = width_response.lost_focus()
                                || (width_response.has_focus()
                                    && ui.input(|i| i.key_pressed(egui::Key::Tab)));
                            if width_commit {
                                self.commit_width_input();
                            }
                            ui.label("px");
                        });
                        ui.end_row();

                        ui.label(t!("dialog.resize_image.height"));
                        ui.horizontal(|ui| {
                            ui.spacing_mut().item_spacing.x = 4.0;
                            let height_response = ui.add(
                                egui::TextEdit::singleline(&mut self.height_input)
                                    .desired_width(96.0),
                            );
                            let height_commit = height_response.lost_focus()
                                || (height_response.has_focus()
                                    && ui.input(|i| i.key_pressed(egui::Key::Tab)));
                            if height_commit {
                                self.commit_height_input();
                            }
                            ui.label("px");
                        });
                        ui.end_row();

                        ui.label("");
                        let lock_resp = ui
                            .checkbox(&mut self.lock_aspect, t!("dialog.resize_image.lock_aspect"));
                        if lock_resp.changed() && self.lock_aspect {
                            self.aspect_ratio = self.width / self.height.max(1.0);
                        }
                        ui.label("");
                        ui.end_row();

                        ui.label(t!("dialog.resize_image.scale"));
                        let s_changed = numeric_field_with_buttons(
                            ui,
                            &mut self.scale_percent,
                            0.5,
                            1.0..=10000.0,
                            "%",
                            5.0,
                        );
                        ui.label("");
                        ui.end_row();

                        if s_changed {
                            self.width =
                                (self.original_w as f32 * self.scale_percent / 100.0).round();
                            self.height =
                                (self.original_h as f32 * self.scale_percent / 100.0).round();
                            self.aspect_ratio = self.width / self.height.max(1.0);
                            self.sync_inputs_from_values();
                        }
                    });

                // -- Anchor section --
                accent_separator(ui, &colors);
                section_label(ui, &colors, &t!("dialog.resize_canvas.anchor_position"));

                ui.add_space(2.0);

                // Visual anchor grid with canvas preview
                let new_w = self.width.round() as u32;
                let new_h = self.height.round() as u32;
                self.draw_anchor_grid(ui, &colors, new_w, new_h);

                ui.add_space(4.0);

                // -- Fill section --
                accent_separator(ui, &colors);
                section_label(ui, &colors, &t!("dialog.resize_canvas.fill"));
                ui.horizontal(|ui| {
                    ui.add_space(4.0);
                    ui.checkbox(
                        &mut self.fill_transparent,
                        t!("dialog.resize_canvas.fill_transparent"),
                    );
                });
                if !self.fill_transparent {
                    ui.horizontal(|ui| {
                        ui.add_space(4.0);
                        ui.label(
                            egui::RichText::new(t!("dialog.resize_canvas.uses_secondary_color"))
                                .size(11.0)
                                .color(colors.text_muted),
                        );
                    });
                }

                // -- Info bar --
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    ui.add_space(4.0);
                    let dw = new_w as i32 - self.original_w as i32;
                    let dh = new_h as i32 - self.original_h as i32;
                    let sign_w = if dw >= 0 { "+" } else { "" };
                    let sign_h = if dh >= 0 { "+" } else { "" };
                    let info = format!(
                        "{}x{} \u{2192} {}x{}  ({}{}px, {}{}px)",
                        self.original_w, self.original_h, new_w, new_h, sign_w, dw, sign_h, dh,
                    );
                    ui.label(
                        egui::RichText::new(info)
                            .size(11.0)
                            .color(colors.text_muted),
                    );
                });

                // -- Footer --
                let (ok, cancel) = dialog_footer(ui, &colors);
                if ok {
                    ok_pressed = true;
                }
                if cancel {
                    result = DialogResult::Cancel;
                }
            });

        if ok_pressed && matches!(result, DialogResult::Open) {
            self.commit_inputs();
            let w = (self.width.round() as u32).max(1);
            let h = (self.height.round() as u32).max(1);
            let fill = if self.fill_transparent {
                Rgba([0, 0, 0, 0])
            } else {
                Rgba([
                    (secondary_color[0] * 255.0) as u8,
                    (secondary_color[1] * 255.0) as u8,
                    (secondary_color[2] * 255.0) as u8,
                    (secondary_color[3] * 255.0) as u8,
                ])
            };
            result = DialogResult::Ok((w, h, self.anchor, fill));
        }

        result
    }

    /// Draw an interactive 3x3 anchor grid with a mini canvas preview.
    fn draw_anchor_grid(
        &mut self,
        ui: &mut egui::Ui,
        colors: &DialogColors,
        new_w: u32,
        new_h: u32,
    ) {
        let grid_size = 120.0f32;
        let cell_size = grid_size / 3.0;

        // Center the grid
        ui.horizontal(|ui| {
            let avail = ui.available_width();
            let pad = ((avail - grid_size) / 2.0).max(0.0);
            ui.add_space(pad);

            let (grid_rect, response) =
                ui.allocate_exact_size(Vec2::splat(grid_size), Sense::click());
            let painter = ui.painter();

            // Background
            let grid_bg = if colors.is_dark {
                Color32::from_gray(25)
            } else {
                Color32::from_gray(245)
            };
            painter.rect_filled(grid_rect, 4.0, grid_bg);
            painter.rect_stroke(
                grid_rect,
                4.0,
                Stroke::new(1.0, colors.separator),
                egui::StrokeKind::Middle,
            );

            // Compute canvas preview rect (showing where the original sits in the new canvas)
            let max_dim = (new_w as f32).max(new_h as f32).max(1.0);
            let preview_w = (new_w as f32 / max_dim) * (grid_size - 8.0);
            let preview_h = (new_h as f32 / max_dim) * (grid_size - 8.0);
            let offset_x = (grid_size - preview_w) / 2.0;
            let offset_y = (grid_size - preview_h) / 2.0;

            let canvas_w = (self.original_w as f32 / max_dim) * (grid_size - 8.0);
            let canvas_h = (self.original_h as f32 / max_dim) * (grid_size - 8.0);

            // Where canvas sits based on anchor
            let cx = match self.anchor.0 {
                0 => grid_rect.min.x + offset_x + 4.0,
                2 => grid_rect.min.x + offset_x + 4.0 + (preview_w - canvas_w),
                _ => grid_rect.min.x + offset_x + 4.0 + (preview_w - canvas_w) / 2.0,
            };
            let cy = match self.anchor.1 {
                0 => grid_rect.min.y + offset_y + 4.0,
                2 => grid_rect.min.y + offset_y + 4.0 + (preview_h - canvas_h),
                _ => grid_rect.min.y + offset_y + 4.0 + (preview_h - canvas_h) / 2.0,
            };

            // Draw new canvas area (faint)
            let new_canvas_rect = Rect::from_min_size(
                Pos2::new(
                    grid_rect.min.x + offset_x + 4.0,
                    grid_rect.min.y + offset_y + 4.0,
                ),
                Vec2::new(preview_w, preview_h),
            );
            let new_area_color = if colors.is_dark {
                Color32::from_gray(40)
            } else {
                Color32::from_gray(225)
            };
            painter.rect_filled(new_canvas_rect, 2.0, new_area_color);

            // Draw original canvas position (accent colored)
            let orig_rect = Rect::from_min_size(Pos2::new(cx, cy), Vec2::new(canvas_w, canvas_h));
            painter.rect_filled(orig_rect, 1.0, colors.accent_faint);
            painter.rect_stroke(
                orig_rect,
                1.0,
                Stroke::new(1.5, colors.accent),
                egui::StrokeKind::Middle,
            );

            // Handle click on the grid to set anchor
            if response.clicked()
                && let Some(pos) = response.interact_pointer_pos()
            {
                let local_x = pos.x - grid_rect.min.x;
                let local_y = pos.y - grid_rect.min.y;
                let col = ((local_x / cell_size).floor() as u32).min(2);
                let row = ((local_y / cell_size).floor() as u32).min(2);
                self.anchor = (col, row);
            }

            // Draw anchor dots aligned to the corners/edges of the new-canvas preview rect.
            // Previously these used uniform cell-centre positions which did not match the
            // accent-bordered original-canvas rect when the canvas was non-square.
            let dot_xs = [
                new_canvas_rect.min.x,
                new_canvas_rect.center().x,
                new_canvas_rect.max.x,
            ];
            let dot_ys = [
                new_canvas_rect.min.y,
                new_canvas_rect.center().y,
                new_canvas_rect.max.y,
            ];
            for row in 0..3u32 {
                for col in 0..3u32 {
                    let center = Pos2::new(dot_xs[col as usize], dot_ys[row as usize]);
                    let selected = self.anchor == (col, row);

                    let dot_radius = if selected { 6.0 } else { 3.5 };
                    let dot_color = if selected {
                        colors.accent
                    } else {
                        colors.text_muted
                    };

                    if selected {
                        // Ring around selected dot
                        painter.circle_stroke(
                            center,
                            dot_radius + 2.0,
                            Stroke::new(1.5, colors.accent),
                        );
                    }
                    painter.circle_filled(center, dot_radius, dot_color);
                }
            }
        });
    }
}

// ============================================================================
// GAUSSIAN BLUR DIALOG - with live preview
// ============================================================================

pub struct GaussianBlurDialog {
    pub sigma: f32,
    /// Snapshot of the original layer pixels before any preview blur.
    pub original_pixels: Option<TiledImage>,
    /// Pre-flattened original pixels (avoids re-flattening every frame).
    pub original_flat: Option<image::RgbaImage>,
    /// The sigma value currently applied to the preview (-1 = none yet).
    pub applied_sigma: f32,
    /// Layer index being blurred.
    pub layer_idx: usize,
    /// Whether live preview is enabled (vs manual preview button).
    pub live_preview: bool,
    /// Advanced mode: unlocks higher radius values (up to 100).
    pub advanced_blur: bool,
    /// Slider currently being dragged.
    pub dragging: bool,
}

impl GaussianBlurDialog {
    pub fn new(state: &CanvasState) -> Self {
        let idx = state.active_layer_index;
        let original = state.layers.get(idx).map(|l| l.pixels.clone());
        let flat = state.layers.get(idx).map(|l| l.pixels.to_rgba_image());
        Self {
            sigma: 0.0,
            original_pixels: original,
            original_flat: flat,
            applied_sigma: -1.0,
            layer_idx: idx,
            live_preview: true,
            advanced_blur: false,
            dragging: false,
        }
    }

    pub fn show(&mut self, ctx: &egui::Context) -> DialogResult<f32> {
        let mut result = DialogResult::Open;
        let colors = DialogColors::from_ctx(ctx);

        crate::ui::polish::window(ctx, "dialog_gaussian_blur")
            .title_bar(false)
            .collapsible(false)
            .resizable(false)
            .default_pos(egui::pos2(ctx.content_rect().center().x - 175.0, 60.0))
            .show(ctx, |ui| {
                ui.set_min_width(350.0);

                if paint_dialog_header(ui, &colors, "\u{1F4A7}", &t!("dialog.gaussian_blur")) {
                    result = DialogResult::Cancel;
                }
                ui.add_space(4.0);

                // -- Parameters --
                section_label(ui, &colors, "BLUR SETTINGS");

                let mut sigma_changed = false;

                let slider_max = if self.advanced_blur { 100.0 } else { 10.0 };

                egui::Grid::new("blur_params")
                    .num_columns(2)
                    .spacing([8.0, 6.0])
                    .show(ui, |ui| {
                        ui.label("Radius");
                        ui.horizontal(|ui| {
                            ui.spacing_mut().item_spacing.x = 4.0;
                            if self.advanced_blur {
                                // Advanced: editable DragValue (up to 100)
                                let r = ui.add(
                                    crate::ui::numeric::Numeric::new(&mut self.sigma)
                                        .speed(0.2)
                                        .range(0.1..=100.0)
                                        .max_decimals(1),
                                );
                                if track_slider(&r, &mut self.dragging) {
                                    sigma_changed = true;
                                }
                            } else {
                                // Normal: slider capped at 10
                                if dialog_slider(ui, &mut self.sigma, 0.1..=slider_max, 0.1, "", 1)
                                {
                                    sigma_changed = true;
                                }
                            }
                        });
                        ui.end_row();

                        // Quick presets
                        ui.label("Quick");
                        ui.horizontal(|ui| {
                            ui.spacing_mut().item_spacing.x = 4.0;
                            let presets: &[(&str, f32)] = if self.advanced_blur {
                                &[
                                    ("Subtle", 1.0),
                                    ("Light", 3.0),
                                    ("Medium", 8.0),
                                    ("Heavy", 25.0),
                                    ("Max", 80.0),
                                ]
                            } else {
                                &[
                                    ("Subtle", 0.5),
                                    ("Light", 1.5),
                                    ("Medium", 3.0),
                                    ("Strong", 6.0),
                                    ("Max", 10.0),
                                ]
                            };
                            for &(label, val) in presets {
                                let is_close = (self.sigma - val).abs() < 0.3;
                                let btn = if is_close {
                                    egui::Button::new(
                                        egui::RichText::new(label).strong().size(11.0),
                                    )
                                    .fill(colors.accent_faint)
                                } else {
                                    egui::Button::new(egui::RichText::new(label).size(11.0))
                                };
                                if ui.add(btn).clicked() {
                                    self.sigma = val;
                                    sigma_changed = true;
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
                                // Clamp sigma when switching back to normal mode
                                if !self.advanced_blur && self.sigma > 10.0 {
                                    self.sigma = 10.0;
                                    sigma_changed = true;
                                }
                            }
                        });
                        ui.end_row();
                    });

                // -- Preview controls --
                accent_separator(ui, &colors);
                let manual_preview = preview_controls(ui, &colors, &mut self.live_preview);

                if sigma_changed && self.live_preview {
                    result = DialogResult::Changed;
                }
                if manual_preview {
                    result = DialogResult::Changed;
                }

                // -- Footer --
                let (ok, cancel) = dialog_footer(ui, &colors);
                if ok {
                    result = DialogResult::Ok(self.sigma);
                }
                if cancel {
                    result = DialogResult::Cancel;
                }
            });

        result
    }
}

// ============================================================================
// ADD BRUSH TIP DIALOG - import a PNG as a custom brush tip
// ============================================================================

/// Result from the AddBrushTipDialog
pub struct AddBrushTipResult {
    pub name: String,
    pub category: String,
    pub png_data: Vec<u8>,
}

pub struct AddBrushTipDialog {
    pub open: bool,
    pub name: String,
    pub category: String,
    pub categories: Vec<String>,
    pub selected_path: Option<std::path::PathBuf>,
    pub png_data: Option<Vec<u8>>,
    pub preview_texture: Option<egui::TextureHandle>,
    pub valid: bool,
    pub error_message: String,
    pub image_width: u32,
    pub image_height: u32,
    pub brush_icon_texture: Option<egui::TextureHandle>,
}

impl AddBrushTipDialog {
    pub fn new(categories: &[String]) -> Self {
        let mut cats = categories.to_vec();
        if !cats.iter().any(|c| c == "Custom") {
            cats.push("Custom".to_string());
        }
        Self {
            open: false,
            name: String::new(),
            category: "Custom".to_string(),
            categories: cats,
            selected_path: None,
            png_data: None,
            preview_texture: None,
            valid: false,
            error_message: String::new(),
            image_width: 0,
            image_height: 0,
            brush_icon_texture: None,
        }
    }

    pub fn open_dialog(&mut self) {
        self.open = true;
        self.name.clear();
        self.category = "Custom".to_string();
        self.selected_path = None;
        self.png_data = None;
        self.preview_texture = None;
        self.valid = false;
        self.error_message.clear();
        self.image_width = 0;
        self.image_height = 0;
    }

    /// Validate the selected file. Returns true if valid.
    #[cfg(not(target_arch = "wasm32"))]
    fn validate_file(&mut self, path: &std::path::Path) -> bool {
        self.error_message.clear();
        self.valid = false;
        self.png_data = None;
        self.preview_texture = None;
        self.image_width = 0;
        self.image_height = 0;

        // Check extension
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_lowercase())
            .unwrap_or_default();
        if ext != "png" {
            self.error_message = "Only PNG files are supported.".to_string();
            return false;
        }

        // Read file
        let data = match std::fs::read(path) {
            Ok(d) => d,
            Err(e) => {
                self.error_message = format!("Cannot read file: {}", e);
                return false;
            }
        };

        self.validate_bytes(data)
    }

    /// Validate already-in-memory PNG bytes (used on web, where files arrive
    /// as bytes from the browser's file picker rather than a real path).
    fn validate_bytes(&mut self, data: Vec<u8>) -> bool {
        self.error_message.clear();
        self.valid = false;
        self.png_data = None;
        self.preview_texture = None;
        self.image_width = 0;
        self.image_height = 0;

        // Decode PNG
        match image::load_from_memory(&data) {
            Ok(img) => {
                self.image_width = img.width();
                self.image_height = img.height();
                self.png_data = Some(data);

                // Check size recommendation
                if img.width() > 256 || img.height() > 256 {
                    self.error_message = format!(
                        "Image is {}x{} — large images may impact performance. Recommended: 128x128.",
                        img.width(),
                        img.height()
                    );
                    // Still valid, just warn
                }

                self.valid = true;
                true
            }
            Err(e) => {
                self.error_message = format!("Invalid PNG: {}", e);
                false
            }
        }
    }

    pub fn show(&mut self, ctx: &egui::Context) -> Option<AddBrushTipResult> {
        if !self.open {
            return None;
        }

        #[cfg(target_arch = "wasm32")]
        if let Some((name, data)) = crate::web_bridge::drain_pending("brush_tip")
            .into_iter()
            .next()
        {
            self.selected_path = Some(std::path::PathBuf::from(&name));
            if self.name.is_empty()
                && let Some(stem) = std::path::Path::new(&name)
                    .file_stem()
                    .and_then(|s| s.to_str())
            {
                self.name = stem.to_string();
            }
            self.validate_bytes(data);
        }

        let mut result: Option<AddBrushTipResult> = None;
        let colors = super::DialogColors::from_ctx(ctx);

        crate::ui::polish::window(ctx, "dialog_add_brush_tip")
            .title_bar(false)
            .collapsible(false)
            .resizable(false)
            .default_pos(egui::pos2(ctx.content_rect().center().x - 200.0, 80.0))
            .show(ctx, |ui| {
                ui.set_min_width(400.0);

                if super::paint_dialog_header_with_texture(
                    ui,
                    &colors,
                    self.brush_icon_texture.as_ref(),
                    "New Brush Tip",
                ) {
                    self.open = false;
                    return;
                }
                ui.add_space(4.0);

                // Info section
                super::section_label(ui, &colors, "INFORMATION");
                ui.add_space(2.0);
                ui.label(
                    egui::RichText::new(
                        "Recommended file size: 128\u{00D7}128 px. Black pixels are treated as \
                         transparent; white pixels define the brush area.",
                    )
                    .size(12.0)
                    .color(colors.text_muted),
                );
                ui.add_space(8.0);

                // Name field
                super::section_label(ui, &colors, "BRUSH NAME");
                ui.add_space(2.0);
                ui.horizontal(|ui| {
                    ui.add_sized(
                        [280.0, 22.0],
                        egui::TextEdit::singleline(&mut self.name)
                            .hint_text("Enter brush tip name..."),
                    );
                });
                ui.add_space(6.0);

                // File browse
                super::section_label(ui, &colors, "IMAGE FILE");
                ui.add_space(2.0);
                ui.horizontal(|ui| {
                    let path_text = self
                        .selected_path
                        .as_ref()
                        .and_then(|p| p.file_name())
                        .and_then(|n| n.to_str())
                        .unwrap_or("No file selected");
                    ui.add_sized(
                        [250.0, 22.0],
                        egui::Label::new(
                            egui::RichText::new(path_text)
                                .size(12.0)
                                .color(colors.text_muted),
                        ),
                    );
                    #[cfg(not(target_arch = "wasm32"))]
                    if ui.button("Browse...").clicked()
                        && let Some(path) = rfd::FileDialog::new()
                            .add_filter("PNG Image", &["png"])
                            .pick_file()
                    {
                        self.selected_path = Some(path.clone());
                        self.validate_file(&path);
                        // Auto-fill name from filename
                        if self.name.is_empty()
                            && let Some(stem) = path.file_stem().and_then(|s| s.to_str())
                        {
                            self.name = stem.to_string();
                        }
                    }
                    #[cfg(target_arch = "wasm32")]
                    if ui.button("Browse...").clicked() {
                        crate::web_bridge::open_picker("brush_tip", "image/png", false);
                    }
                });
                ui.add_space(6.0);

                // Category
                super::section_label(ui, &colors, "CATEGORY");
                ui.add_space(2.0);
                egui::ComboBox::from_id_salt("brush_tip_category")
                    .selected_text(&self.category)
                    .width(200.0)
                    .show_ui(ui, |ui| {
                        for cat in &self.categories {
                            ui.selectable_value(&mut self.category, cat.clone(), cat.as_str());
                        }
                    });
                ui.add_space(8.0);

                // Preview / validation
                if self.selected_path.is_some() {
                    super::accent_separator(ui, &colors);
                    ui.add_space(4.0);
                    super::section_label(ui, &colors, "PREVIEW");

                    if self.valid {
                        // Load preview texture if needed
                        if self.preview_texture.is_none()
                            && let Some(png_data) = &self.png_data
                            && let Ok(img) = image::load_from_memory(png_data)
                        {
                            let rgba = img.to_rgba8();
                            let (w, h) = rgba.dimensions();
                            let size = [w as usize, h as usize];
                            let pixels = rgba.into_raw();
                            let ci = egui::ColorImage::from_rgba_unmultiplied(size, &pixels);
                            let tex = ctx.load_texture(
                                "brush_tip_preview",
                                ci,
                                egui::TextureOptions::LINEAR,
                            );
                            self.preview_texture = Some(tex);
                        }

                        ui.horizontal(|ui| {
                            if let Some(tex) = &self.preview_texture {
                                let max_preview = 128.0_f32;
                                let scale = (max_preview / self.image_width as f32)
                                    .min(max_preview / self.image_height as f32)
                                    .min(1.0);
                                let preview_size = egui::vec2(
                                    self.image_width as f32 * scale,
                                    self.image_height as f32 * scale,
                                );
                                let sized = egui::load::SizedTexture::from_handle(tex);
                                let preview_rect =
                                    ui.allocate_exact_size(preview_size, egui::Sense::hover()).0;
                                egui::Image::from_texture(sized)
                                    .fit_to_exact_size(preview_size)
                                    .paint_at(ui, preview_rect);
                            }
                            ui.add_space(8.0);
                            ui.vertical(|ui| {
                                ui.label(
                                    egui::RichText::new("\u{2705} Valid")
                                        .color(egui::Color32::from_rgb(80, 200, 80)),
                                );
                                ui.label(format!("{}x{} px", self.image_width, self.image_height));
                                if !self.error_message.is_empty() {
                                    ui.label(
                                        egui::RichText::new(&self.error_message)
                                            .size(11.0)
                                            .color(egui::Color32::from_rgb(220, 180, 50)),
                                    );
                                }
                            });
                        });
                    } else {
                        ui.horizontal(|ui| {
                            ui.label(
                                egui::RichText::new("\u{274C} Invalid")
                                    .color(egui::Color32::from_rgb(220, 60, 60)),
                            );
                            if !self.error_message.is_empty() {
                                ui.label(
                                    egui::RichText::new(&self.error_message)
                                        .size(11.0)
                                        .color(egui::Color32::from_rgb(220, 60, 60)),
                                );
                            }
                        });
                    }
                    ui.add_space(4.0);
                }

                // Footer
                let (ok, cancel) = super::dialog_footer(ui, &colors);
                if ok {
                    let name_trimmed = self.name.trim().to_string();
                    if name_trimmed.is_empty() {
                        self.error_message = "Please enter a brush name.".to_string();
                    } else if !self.valid {
                        self.error_message = "Please select a valid PNG file.".to_string();
                    } else if let Some(png_data) = self.png_data.clone() {
                        result = Some(AddBrushTipResult {
                            name: name_trimmed,
                            category: self.category.clone(),
                            png_data,
                        });
                        self.open = false;
                    }
                }
                if cancel {
                    self.open = false;
                }
            });

        result
    }
}

pub struct AddShapeResult {
    pub name: String,
    pub svg_path_data: String,
}

pub struct AddShapeDialog {
    pub open: bool,
    pub name: String,
    pub categories: Vec<String>,
    pub selected_path: Option<std::path::PathBuf>,
    pub svg_path_data: Option<String>,
    pub preview_texture: Option<egui::TextureHandle>,
    pub valid: bool,
    pub error_message: String,
}

impl AddShapeDialog {
    pub fn new(categories: &[String]) -> Self {
        let mut cats = categories.to_vec();
        if !cats.iter().any(|c| c == "Custom") {
            cats.push("Custom".to_string());
        }
        Self {
            open: false,
            name: String::new(),
            categories: cats,
            selected_path: None,
            svg_path_data: None,
            preview_texture: None,
            valid: false,
            error_message: String::new(),
        }
    }

    pub fn open_dialog(&mut self) {
        self.open = true;
        self.name.clear();
        self.selected_path = None;
        self.svg_path_data = None;
        self.preview_texture = None;
        self.valid = false;
        self.error_message.clear();
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn validate_file(&mut self, ctx: &egui::Context, path: &std::path::Path) {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_lowercase())
            .unwrap_or_default();
        if ext != "svg" {
            self.error_message.clear();
            self.valid = false;
            self.svg_path_data = None;
            self.preview_texture = None;
            self.error_message = "Only SVG files are supported.".to_string();
            return;
        }
        let text = match std::fs::read_to_string(path) {
            Ok(t) => t,
            Err(e) => {
                self.error_message = format!("Cannot read file: {e}");
                return;
            }
        };
        self.validate_text(ctx, &text);
    }

    /// Validate already-in-memory SVG text (used on web, where files arrive
    /// as bytes from the browser's file picker rather than a real path).
    fn validate_text(&mut self, ctx: &egui::Context, text: &str) {
        self.error_message.clear();
        self.valid = false;
        self.svg_path_data = None;
        self.preview_texture = None;
        let path_data = match crate::ops::shapes::extract_svg_path_data(text).and_then(|d| {
            crate::ops::shapes::parse_custom_shape("Preview", "Custom", &d).map(|_| d)
        }) {
            Ok(d) => d,
            Err(e) => {
                self.error_message = e;
                return;
            }
        };
        let preview = crate::ops::shapes::parse_custom_shape("Preview", "Custom", &path_data)
            .map(|s| {
                crate::ops::shapes::render_custom_shape_icon(
                    &s,
                    96,
                    ctx.global_style().visuals.dark_mode,
                )
            })
            .ok();
        if let Some(pixels) = preview {
            let ci = egui::ColorImage::from_rgba_unmultiplied([96, 96], &pixels);
            self.preview_texture =
                Some(ctx.load_texture("shape_preview", ci, egui::TextureOptions::LINEAR));
        }
        self.svg_path_data = Some(path_data);
        self.valid = true;
    }

    pub fn show(&mut self, ctx: &egui::Context) -> Option<AddShapeResult> {
        if !self.open {
            return None;
        }

        #[cfg(target_arch = "wasm32")]
        if let Some((name, data)) = crate::web_bridge::drain_pending("custom_shape")
            .into_iter()
            .next()
        {
            self.selected_path = Some(std::path::PathBuf::from(&name));
            if self.name.is_empty()
                && let Some(stem) = std::path::Path::new(&name)
                    .file_stem()
                    .and_then(|s| s.to_str())
            {
                self.name = stem.to_string();
            }
            match String::from_utf8(data) {
                Ok(text) => self.validate_text(ctx, &text),
                Err(_) => self.error_message = "File is not valid UTF-8 text.".to_string(),
            }
        }

        let mut result = None;
        let colors = super::DialogColors::from_ctx(ctx);
        crate::ui::polish::window(ctx, "dialog_add_shape")
            .title_bar(false)
            .collapsible(false)
            .resizable(false)
            .default_pos(egui::pos2(ctx.content_rect().center().x - 200.0, 80.0))
            .show(ctx, |ui| {
                ui.set_min_width(400.0);
                if super::paint_dialog_header(ui, &colors, "◇", "New Shape") {
                    self.open = false;
                    return;
                }
                ui.add_space(4.0);
                super::section_label(ui, &colors, "INFORMATION");
                ui.label(egui::RichText::new("Import an SVG containing path geometry. Use the example as a template; embedded raster images are not supported.").size(12.0).color(colors.text_muted));
                ui.add_space(8.0);
                super::section_label(ui, &colors, "SHAPE NAME");
                ui.add_sized([280.0, 22.0], egui::TextEdit::singleline(&mut self.name).hint_text("Enter shape name..."));
                ui.add_space(6.0);
                super::section_label(ui, &colors, "SVG FILE");
                ui.horizontal(|ui| {
                    let path_text = self.selected_path.as_ref().and_then(|p| p.file_name()).and_then(|n| n.to_str()).unwrap_or("No file selected");
                    ui.add_sized([250.0, 22.0], egui::Label::new(egui::RichText::new(path_text).size(12.0).color(colors.text_muted)));
                    #[cfg(not(target_arch = "wasm32"))]
                    if ui.button("Browse...").clicked()
                        && let Some(path) = rfd::FileDialog::new().add_filter("SVG Shape", &["svg"]).pick_file()
                    {
                        self.selected_path = Some(path.clone());
                        self.validate_file(ctx, &path);
                        if self.name.is_empty()
                            && let Some(stem) = path.file_stem().and_then(|s| s.to_str())
                        {
                            self.name = stem.to_string();
                        }
                    }
                    #[cfg(target_arch = "wasm32")]
                    if ui.button("Browse...").clicked() {
                        crate::web_bridge::open_picker("custom_shape", ".svg,image/svg+xml", false);
                    }
                    let example_svg = r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="-50 -50 100 100">
  <path d="M 0 -48 L 45.65 -14.83 L 28.21 38.83 L -28.21 38.83 L -45.65 -14.83 Z"/>
</svg>
"#;
                    #[cfg(target_arch = "wasm32")]
                    if ui.button("Export example...").clicked() {
                        crate::web_fs::trigger_download(
                            "paintfe_pentagon_shape.svg",
                            example_svg.as_bytes(),
                        );
                    }
                    #[cfg(not(target_arch = "wasm32"))]
                    if ui.button("Export example...").clicked()
                        && let Some(path) = rfd::FileDialog::new()
                            .set_file_name("paintfe_pentagon_shape.svg")
                            .add_filter("SVG Shape", &["svg"])
                            .save_file()
                        && let Err(e) = std::fs::write(&path, example_svg)
                    {
                        self.error_message = format!("Cannot write example: {e}");
                    }
                });
                if self.selected_path.is_some() {
                    super::accent_separator(ui, &colors);
                    super::section_label(ui, &colors, "PREVIEW");
                    if self.valid {
                        ui.horizontal(|ui| {
                            if let Some(tex) = &self.preview_texture {
                                let sized = egui::load::SizedTexture::from_handle(tex);
                                let rect = ui.allocate_exact_size(egui::Vec2::splat(96.0), egui::Sense::hover()).0;
                                egui::Image::from_texture(sized).fit_to_exact_size(egui::Vec2::splat(96.0)).paint_at(ui, rect);
                            }
                            ui.label(egui::RichText::new("Valid SVG path").color(egui::Color32::from_rgb(80, 200, 80)));
                        });
                    } else {
                        ui.label(egui::RichText::new(&self.error_message).color(egui::Color32::from_rgb(220, 60, 60)));
                    }
                }
                let (ok, cancel) = super::dialog_footer(ui, &colors);
                if ok {
                    let name = self.name.trim().to_string();
                    if name.is_empty() {
                        self.error_message = "Please enter a shape name.".to_string();
                    } else if !self.valid {
                        self.error_message = "Please select a valid SVG file.".to_string();
                    } else if let Some(svg_path_data) = self.svg_path_data.clone() {
                        result = Some(AddShapeResult {
                            name,
                            svg_path_data,
                        });
                        self.open = false;
                    }
                }
                if cancel {
                    self.open = false;
                }
            });
        result
    }
}
