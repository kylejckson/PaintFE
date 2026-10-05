pub struct BrightnessContrastDialog {
    pub brightness: f32,
    pub contrast: f32,
    pub original_pixels: Option<TiledImage>,
    pub original_flat: Option<image::RgbaImage>,
    pub layer_idx: usize,
    pub live_preview: bool,
    applied_vals: (f32, f32),
}

impl BrightnessContrastDialog {
    pub fn new(state: &CanvasState) -> Self {
        let idx = state.active_layer_index;
        Self {
            brightness: 0.0,
            contrast: 0.0,
            original_pixels: state.layers.get(idx).map(|l| l.pixels.clone()),
            original_flat: state.layers.get(idx).map(|l| l.pixels.to_rgba_image()),
            layer_idx: idx,
            live_preview: true,
            applied_vals: (f32::NAN, f32::NAN),
        }
    }

    pub fn show(&mut self, ctx: &egui::Context) -> DialogResult<(f32, f32)> {
        let mut result = DialogResult::Open;
        let colors = DialogColors::from_ctx(ctx);

        crate::ui::polish::window(ctx, "dialog_brightness_contrast")
            .title_bar(false)
            .collapsible(false)
            .resizable(false)
            .default_pos(egui::pos2(ctx.content_rect().center().x - 175.0, 60.0))
            .show(ctx, |ui| {
                ui.set_min_width(350.0);
                if paint_dialog_header(ui, &colors, "☀", &t!("dialog.brightness_contrast")) {
                    result = DialogResult::Cancel;
                }
                ui.add_space(4.0);

                section_label(ui, &colors, "ADJUSTMENTS");
                let mut changed = false;

                egui::Grid::new("bc_params")
                    .num_columns(2)
                    .spacing([8.0, 6.0])
                    .show(ui, |ui| {
                        ui.label("Brightness");
                        if dialog_slider(ui, &mut self.brightness, -100.0..=100.0, 1.0, "", 0) {
                            changed = true;
                        }
                        ui.end_row();

                        ui.label("Contrast");
                        if dialog_slider(ui, &mut self.contrast, -100.0..=100.0, 1.0, "", 0) {
                            changed = true;
                        }
                        ui.end_row();
                    });

                // Visual indicator bar
                ui.add_space(4.0);
                let bar_rect = ui.allocate_space(Vec2::new(ui.available_width(), 8.0)).1;
                let painter = ui.painter();
                // Draw gradient bar showing current adjustment effect
                let steps = 32;
                let bar_w = bar_rect.width() / steps as f32;
                let factor = (259.0 * (self.contrast + 255.0)) / (255.0 * (259.0 - self.contrast));
                for i in 0..steps {
                    let t = i as f32 / (steps - 1) as f32;
                    let v = t * 255.0;
                    let adjusted =
                        (factor * (v + self.brightness - 128.0) + 128.0).clamp(0.0, 255.0) as u8;
                    let color = Color32::from_gray(adjusted);
                    let r = Rect::from_min_size(
                        Pos2::new(bar_rect.min.x + i as f32 * bar_w, bar_rect.min.y),
                        Vec2::new(bar_w + 0.5, 8.0),
                    );
                    painter.rect_filled(r, 0.0, color);
                }
                painter.rect_stroke(
                    bar_rect,
                    CornerRadius::same(2),
                    Stroke::new(1.0, colors.separator),
                    egui::StrokeKind::Middle,
                );

                accent_separator(ui, &colors);
                let manual_preview = preview_controls(ui, &colors, &mut self.live_preview);
                if changed && self.live_preview {
                    result = DialogResult::Changed;
                }
                if manual_preview {
                    result = DialogResult::Changed;
                }

                let (ok, cancel, reset) = dialog_footer_with_reset(ui, &colors);
                if reset {
                    self.brightness = 0.0;
                    self.contrast = 0.0;
                    result = DialogResult::Changed;
                }
                if ok {
                    result = DialogResult::Ok((self.brightness, self.contrast));
                }
                if cancel {
                    result = DialogResult::Cancel;
                }
            });
        result
    }
}

// ============================================================================
// HUE / SATURATION DIALOG
// ============================================================================

pub struct HueSaturationDialog {
    pub hue: f32,
    pub saturation: f32,
    pub lightness: f32,
    pub original_pixels: Option<TiledImage>,
    pub original_flat: Option<image::RgbaImage>,
    pub layer_idx: usize,
    pub live_preview: bool,
    // Per-band mode
    pub per_band: bool,
    pub selected_band: usize, // 0=Reds 1=Yellows 2=Greens 3=Cyans 4=Blues 5=Magentas
    pub bands: [crate::ops::adjustments::HueBandAdjust; 6],
}

impl HueSaturationDialog {
    pub fn new(state: &CanvasState) -> Self {
        let idx = state.active_layer_index;
        Self {
            hue: 0.0,
            saturation: 0.0,
            lightness: 0.0,
            original_pixels: state.layers.get(idx).map(|l| l.pixels.clone()),
            original_flat: state.layers.get(idx).map(|l| l.pixels.to_rgba_image()),
            layer_idx: idx,
            live_preview: true,
            per_band: false,
            selected_band: 0,
            bands: Default::default(),
        }
    }

    pub fn show(&mut self, ctx: &egui::Context) -> DialogResult<()> {
        let mut result = DialogResult::Open;
        let colors = DialogColors::from_ctx(ctx);

        crate::ui::polish::window(ctx, "dialog_hue_saturation")
            .title_bar(false)
            .collapsible(false)
            .resizable(false)
            .default_pos(egui::pos2(ctx.content_rect().center().x - 175.0, 60.0))
            .show(ctx, |ui| {
                ui.set_min_width(380.0);
                if paint_dialog_header(ui, &colors, "🎨", &t!("dialog.hue_saturation")) {
                    result = DialogResult::Cancel;
                }
                ui.add_space(4.0);

                section_label(ui, &colors, "GLOBAL ADJUSTMENTS");
                let mut changed = false;

                egui::Grid::new("hs_params")
                    .num_columns(2)
                    .spacing([8.0, 6.0])
                    .show(ui, |ui| {
                        ui.label("Hue");
                        if dialog_slider(ui, &mut self.hue, -180.0..=180.0, 1.0, "°", 0) {
                            changed = true;
                        }
                        ui.end_row();

                        ui.label("Saturation");
                        if dialog_slider(ui, &mut self.saturation, -100.0..=100.0, 1.0, "", 0) {
                            changed = true;
                        }
                        ui.end_row();

                        ui.label("Lightness");
                        if dialog_slider(ui, &mut self.lightness, -100.0..=100.0, 1.0, "", 0) {
                            changed = true;
                        }
                        ui.end_row();
                    });

                // Hue spectrum bar
                ui.add_space(4.0);
                let bar_rect = ui.allocate_space(Vec2::new(ui.available_width(), 12.0)).1;
                let painter = ui.painter();
                let steps = 64;
                let bar_w = bar_rect.width() / steps as f32;
                for i in 0..steps {
                    let hue_deg = (i as f32 / steps as f32) * 360.0 + self.hue;
                    let hue_norm = ((hue_deg % 360.0) + 360.0) % 360.0;
                    let (r, g, b) = hsv_to_rgb_simple(hue_norm, 0.8, 0.9);
                    let color =
                        Color32::from_rgb((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8);
                    let r = Rect::from_min_size(
                        Pos2::new(bar_rect.min.x + i as f32 * bar_w, bar_rect.min.y),
                        Vec2::new(bar_w + 0.5, 12.0),
                    );
                    painter.rect_filled(r, 0.0, color);
                }
                painter.rect_stroke(
                    bar_rect,
                    CornerRadius::same(2),
                    Stroke::new(1.0, colors.separator),
                    egui::StrokeKind::Middle,
                );

                // Per-band toggle
                ui.add_space(6.0);
                if ui.checkbox(&mut self.per_band, "Per Band").changed() {
                    changed = true;
                }

                if self.per_band {
                    ui.add_space(4.0);
                    // Band selector row
                    const BAND_NAMES: [&str; 6] = ["R", "Y", "G", "C", "B", "M"];
                    const BAND_HUES: [f32; 6] = [0.0, 60.0, 120.0, 180.0, 240.0, 300.0];
                    ui.horizontal(|ui| {
                        ui.label("Band:");
                        for i in 0..6 {
                            let band = &self.bands[i];
                            let is_nondefault =
                                band.hue != 0.0 || band.saturation != 0.0 || band.lightness != 0.0;
                            let (hr, hg, hb) = hsv_to_rgb_simple(BAND_HUES[i], 0.75, 0.85);
                            let band_col = Color32::from_rgb(
                                (hr * 255.0) as u8,
                                (hg * 255.0) as u8,
                                (hb * 255.0) as u8,
                            );
                            let selected = self.selected_band == i;
                            let btn_text = if is_nondefault {
                                format!("{}●", BAND_NAMES[i])
                            } else {
                                BAND_NAMES[i].to_string()
                            };
                            let btn = egui::Button::new(
                                egui::RichText::new(&btn_text).color(if selected {
                                    Color32::WHITE
                                } else {
                                    band_col
                                }),
                            )
                            .fill(if selected {
                                band_col.linear_multiply(0.5)
                            } else {
                                Color32::TRANSPARENT
                            })
                            .stroke(egui::Stroke::new(1.5, band_col));
                            if ui.add_sized([32.0, 22.0], btn).clicked() {
                                self.selected_band = i;
                            }
                        }
                    });

                    // Per-band sliders for selected band
                    let band_label = ["Reds", "Yellows", "Greens", "Cyans", "Blues", "Magentas"]
                        [self.selected_band];
                    ui.add_space(4.0);
                    section_label(ui, &colors, &format!("{} BAND", band_label.to_uppercase()));

                    let b = &mut self.bands[self.selected_band];
                    egui::Grid::new("hs_band_params")
                        .num_columns(2)
                        .spacing([8.0, 6.0])
                        .show(ui, |ui| {
                            ui.label("Hue");
                            if dialog_slider(ui, &mut b.hue, -180.0..=180.0, 1.0, "°", 0) {
                                changed = true;
                            }
                            ui.end_row();
                            ui.label("Saturation");
                            if dialog_slider(ui, &mut b.saturation, -100.0..=100.0, 1.0, "", 0) {
                                changed = true;
                            }
                            ui.end_row();
                            ui.label("Lightness");
                            if dialog_slider(ui, &mut b.lightness, -100.0..=100.0, 1.0, "", 0) {
                                changed = true;
                            }
                            ui.end_row();
                        });

                    // Reset band button
                    if ui.small_button("Reset Band").clicked() {
                        self.bands[self.selected_band] = Default::default();
                        changed = true;
                    }
                }

                accent_separator(ui, &colors);
                let manual_preview = preview_controls(ui, &colors, &mut self.live_preview);
                if changed && self.live_preview {
                    result = DialogResult::Changed;
                }
                if manual_preview {
                    result = DialogResult::Changed;
                }

                let (ok, cancel, reset) = dialog_footer_with_reset(ui, &colors);
                if reset {
                    self.hue = 0.0;
                    self.saturation = 0.0;
                    self.lightness = 0.0;
                    self.bands = Default::default();
                    result = DialogResult::Changed;
                }
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

/// HSV (H: 0..360, S: 0..1, V: 0..1) → RGB (0..1) for UI display
fn hsv_to_rgb_simple(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
    let c = v * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = v - c;
    let (r, g, b) = if h < 60.0 {
        (c, x, 0.0)
    } else if h < 120.0 {
        (x, c, 0.0)
    } else if h < 180.0 {
        (0.0, c, x)
    } else if h < 240.0 {
        (0.0, x, c)
    } else if h < 300.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };
    (r + m, g + m, b + m)
}

// ============================================================================
// EXPOSURE DIALOG
// ============================================================================

pub struct ExposureDialog {
    pub exposure: f32,
    pub original_pixels: Option<TiledImage>,
    pub original_flat: Option<image::RgbaImage>,
    pub layer_idx: usize,
    pub live_preview: bool,
}

impl ExposureDialog {
    pub fn new(state: &CanvasState) -> Self {
        let idx = state.active_layer_index;
        Self {
            exposure: 0.0,
            original_pixels: state.layers.get(idx).map(|l| l.pixels.clone()),
            original_flat: state.layers.get(idx).map(|l| l.pixels.to_rgba_image()),
            layer_idx: idx,
            live_preview: true,
        }
    }

    pub fn show(&mut self, ctx: &egui::Context) -> DialogResult<f32> {
        let mut result = DialogResult::Open;
        let colors = DialogColors::from_ctx(ctx);

        crate::ui::polish::window(ctx, "dialog_exposure")
            .title_bar(false)
            .collapsible(false)
            .resizable(false)
            .default_pos(egui::pos2(ctx.content_rect().center().x - 160.0, 60.0))
            .show(ctx, |ui| {
                ui.set_min_width(320.0);
                if paint_dialog_header(ui, &colors, "📷", &t!("dialog.exposure")) {
                    result = DialogResult::Cancel;
                }
                ui.add_space(4.0);

                section_label(ui, &colors, "EXPOSURE");
                let mut changed = false;

                egui::Grid::new("exp_params")
                    .num_columns(2)
                    .spacing([8.0, 6.0])
                    .show(ui, |ui| {
                        ui.label("EV Stops");
                        if dialog_slider(ui, &mut self.exposure, -5.0..=5.0, 0.1, " EV", 2) {
                            changed = true;
                        }
                        ui.end_row();

                        ui.label("Quick");
                        ui.horizontal(|ui| {
                            ui.spacing_mut().item_spacing.x = 4.0;
                            for &(label, val) in &[
                                ("-2", -2.0),
                                ("-1", -1.0),
                                ("0", 0.0),
                                ("+1", 1.0),
                                ("+2", 2.0),
                            ] {
                                let is_close = (self.exposure - val).abs() < 0.1;
                                let btn = if is_close {
                                    egui::Button::new(
                                        egui::RichText::new(label).strong().size(11.0),
                                    )
                                    .fill(colors.accent_faint)
                                } else {
                                    egui::Button::new(egui::RichText::new(label).size(11.0))
                                };
                                if ui.add(btn).clicked() {
                                    self.exposure = val;
                                    changed = true;
                                }
                            }
                        });
                        ui.end_row();
                    });

                // Exposure preview bar
                ui.add_space(4.0);
                let bar_rect = ui.allocate_space(Vec2::new(ui.available_width(), 8.0)).1;
                let painter = ui.painter();
                let gain = 2.0f32.powf(self.exposure);
                let steps = 32;
                let bar_w = bar_rect.width() / steps as f32;
                for i in 0..steps {
                    let t = i as f32 / (steps - 1) as f32;
                    let v = (t * 255.0 * gain).clamp(0.0, 255.0) as u8;
                    let r = Rect::from_min_size(
                        Pos2::new(bar_rect.min.x + i as f32 * bar_w, bar_rect.min.y),
                        Vec2::new(bar_w + 0.5, 8.0),
                    );
                    painter.rect_filled(r, 0.0, Color32::from_gray(v));
                }
                painter.rect_stroke(
                    bar_rect,
                    CornerRadius::same(2),
                    Stroke::new(1.0, colors.separator),
                    egui::StrokeKind::Middle,
                );

                accent_separator(ui, &colors);
                let manual_preview = preview_controls(ui, &colors, &mut self.live_preview);
                if changed && self.live_preview {
                    result = DialogResult::Changed;
                }
                if manual_preview {
                    result = DialogResult::Changed;
                }

                let (ok, cancel, reset) = dialog_footer_with_reset(ui, &colors);
                if reset {
                    self.exposure = 0.0;
                    result = DialogResult::Changed;
                }
                if ok {
                    result = DialogResult::Ok(self.exposure);
                }
                if cancel {
                    result = DialogResult::Cancel;
                }
            });
        result
    }
}

// ============================================================================
// HIGHLIGHTS / SHADOWS DIALOG
// ============================================================================

pub struct HighlightsShadowsDialog {
    pub shadows: f32,
    pub highlights: f32,
    pub original_pixels: Option<TiledImage>,
    pub original_flat: Option<image::RgbaImage>,
    pub layer_idx: usize,
    pub live_preview: bool,
}

impl HighlightsShadowsDialog {
    pub fn new(state: &CanvasState) -> Self {
        let idx = state.active_layer_index;
        Self {
            shadows: 0.0,
            highlights: 0.0,
            original_pixels: state.layers.get(idx).map(|l| l.pixels.clone()),
            original_flat: state.layers.get(idx).map(|l| l.pixels.to_rgba_image()),
            layer_idx: idx,
            live_preview: true,
        }
    }

    pub fn show(&mut self, ctx: &egui::Context) -> DialogResult<(f32, f32)> {
        let mut result = DialogResult::Open;
        let colors = DialogColors::from_ctx(ctx);

        crate::ui::polish::window(ctx, "dialog_highlights_shadows")
            .title_bar(false)
            .collapsible(false)
            .resizable(false)
            .default_pos(egui::pos2(ctx.content_rect().center().x - 175.0, 60.0))
            .show(ctx, |ui| {
                ui.set_min_width(350.0);
                if paint_dialog_header(ui, &colors, "◑", &t!("dialog.highlights_shadows")) {
                    result = DialogResult::Cancel;
                }
                ui.add_space(4.0);

                section_label(ui, &colors, "TONAL ADJUSTMENTS");
                let mut changed = false;

                egui::Grid::new("hs_params_grid")
                    .num_columns(2)
                    .spacing([8.0, 6.0])
                    .show(ui, |ui| {
                        ui.label("Shadows");
                        if dialog_slider(ui, &mut self.shadows, -100.0..=100.0, 1.0, "", 0) {
                            changed = true;
                        }
                        ui.end_row();

                        ui.label("Highlights");
                        if dialog_slider(ui, &mut self.highlights, -100.0..=100.0, 1.0, "", 0) {
                            changed = true;
                        }
                        ui.end_row();
                    });

                // Visual: shadow/highlight regions
                ui.add_space(4.0);
                let bar_rect = ui.allocate_space(Vec2::new(ui.available_width(), 12.0)).1;
                let painter = ui.painter();
                let half = bar_rect.width() / 2.0;
                // Left half: shadows (dark → mid)
                let shadow_brightness = (self.shadows / 100.0 * 0.4 + 0.15).clamp(0.0, 0.5);
                let shadow_color = Color32::from_rgb(
                    (shadow_brightness * 255.0) as u8,
                    (shadow_brightness * 255.0) as u8,
                    (shadow_brightness * 255.0 * 1.1).min(255.0) as u8,
                );
                painter.rect_filled(
                    Rect::from_min_size(bar_rect.min, Vec2::new(half, 12.0)),
                    CornerRadius::ZERO,
                    shadow_color,
                );
                // Right half: highlights (mid → bright)
                let highlight_brightness = (0.7 + self.highlights / 100.0 * 0.3).clamp(0.5, 1.0);
                let highlight_color = Color32::from_rgb(
                    (highlight_brightness * 255.0) as u8,
                    (highlight_brightness * 255.0) as u8,
                    (highlight_brightness * 240.0) as u8,
                );
                painter.rect_filled(
                    Rect::from_min_size(
                        Pos2::new(bar_rect.min.x + half, bar_rect.min.y),
                        Vec2::new(half, 12.0),
                    ),
                    CornerRadius::ZERO,
                    highlight_color,
                );
                painter.rect_stroke(
                    bar_rect,
                    CornerRadius::same(2),
                    Stroke::new(1.0, colors.separator),
                    egui::StrokeKind::Middle,
                );
                // Labels
                painter.text(
                    Pos2::new(bar_rect.min.x + half * 0.5, bar_rect.center().y),
                    egui::Align2::CENTER_CENTER,
                    "Shadows",
                    egui::FontId::proportional(9.0),
                    if shadow_brightness > 0.3 {
                        Color32::BLACK
                    } else {
                        Color32::WHITE
                    },
                );
                painter.text(
                    Pos2::new(bar_rect.min.x + half * 1.5, bar_rect.center().y),
                    egui::Align2::CENTER_CENTER,
                    "Highlights",
                    egui::FontId::proportional(9.0),
                    if highlight_brightness > 0.7 {
                        Color32::BLACK
                    } else {
                        Color32::WHITE
                    },
                );

                accent_separator(ui, &colors);
                let manual_preview = preview_controls(ui, &colors, &mut self.live_preview);
                if changed && self.live_preview {
                    result = DialogResult::Changed;
                }
                if manual_preview {
                    result = DialogResult::Changed;
                }

                let (ok, cancel, reset) = dialog_footer_with_reset(ui, &colors);
                if reset {
                    self.shadows = 0.0;
                    self.highlights = 0.0;
                    result = DialogResult::Changed;
                }
                if ok {
                    result = DialogResult::Ok((self.shadows, self.highlights));
                }
                if cancel {
                    result = DialogResult::Cancel;
                }
            });
        result
    }
}

// ============================================================================
// LEVELS DIALOG — with histogram visualization
// ============================================================================

/// Per-channel levels result returned when dialog is confirmed.
pub struct LevelsDialogResult {
    /// (in_black, in_white, gamma, out_black, out_white) for Master & R/G/B channels.
    pub master: (f32, f32, f32, f32, f32),
    pub r_ch: (f32, f32, f32, f32, f32),
    pub g_ch: (f32, f32, f32, f32, f32),
    pub b_ch: (f32, f32, f32, f32, f32),
}

pub struct LevelsDialog {
    /// Per-channel input/output level params: index 0=Master, 1=R, 2=G, 3=B
    pub ch_input_black: [f32; 4],
    pub ch_input_white: [f32; 4],
    pub ch_gamma: [f32; 4],
    pub ch_output_black: [f32; 4],
    pub ch_output_white: [f32; 4],
    /// Which channel is currently shown in the editor (0=Master, 1=R, 2=G, 3=B)
    pub active_channel: usize,
    pub original_pixels: Option<TiledImage>,
    pub original_flat: Option<image::RgbaImage>,
    pub layer_idx: usize,
    pub live_preview: bool,
    /// Histograms: [Luminance, R, G, B]
    pub ch_histograms: [[u32; 256]; 4],
    pub ch_hist_max: [u32; 4],
}

impl LevelsDialog {
    pub fn new(state: &CanvasState) -> Self {
        let idx = state.active_layer_index;
        let (hr, hg, hb, hl) = crate::ops::adjustments::compute_histogram(state, idx);
        let hist_max = |h: &[u32; 256]| h.iter().copied().max().unwrap_or(1).max(1);
        Self {
            ch_input_black: [0.0; 4],
            ch_input_white: [255.0; 4],
            ch_gamma: [1.0; 4],
            ch_output_black: [0.0; 4],
            ch_output_white: [255.0; 4],
            active_channel: 0,
            original_pixels: state.layers.get(idx).map(|l| l.pixels.clone()),
            original_flat: state.layers.get(idx).map(|l| l.pixels.to_rgba_image()),
            layer_idx: idx,
            live_preview: true,
            ch_histograms: [hl, hr, hg, hb],
            ch_hist_max: [hist_max(&hl), hist_max(&hr), hist_max(&hg), hist_max(&hb)],
        }
    }

    fn ch_color(ch: usize, colors: &DialogColors) -> Color32 {
        match ch {
            1 => Color32::from_rgb(220, 60, 60),
            2 => Color32::from_rgb(60, 180, 60),
            3 => Color32::from_rgb(70, 100, 220),
            _ => colors.accent,
        }
    }

    pub fn current_params(&self) -> (f32, f32, f32, f32, f32) {
        let c = self.active_channel;
        (
            self.ch_input_black[c],
            self.ch_input_white[c],
            self.ch_gamma[c],
            self.ch_output_black[c],
            self.ch_output_white[c],
        )
    }

    pub fn as_result(&self) -> LevelsDialogResult {
        let p = |c: usize| {
            (
                self.ch_input_black[c],
                self.ch_input_white[c],
                self.ch_gamma[c],
                self.ch_output_black[c],
                self.ch_output_white[c],
            )
        };
        LevelsDialogResult {
            master: p(0),
            r_ch: p(1),
            g_ch: p(2),
            b_ch: p(3),
        }
    }

    pub fn show(&mut self, ctx: &egui::Context) -> DialogResult<LevelsDialogResult> {
        let mut result = DialogResult::Open;
        let colors = DialogColors::from_ctx(ctx);

        crate::ui::polish::window(ctx, "dialog_levels")
            .title_bar(false)
            .collapsible(false)
            .resizable(false)
            .default_pos(egui::pos2(ctx.content_rect().center().x - 200.0, 40.0))
            .show(ctx, |ui| {
                ui.set_min_width(400.0);
                if paint_dialog_header(ui, &colors, "📊", &t!("dialog.levels")) {
                    result = DialogResult::Cancel;
                }
                ui.add_space(4.0);

                // --- Channel selector ---
                let ch_labels = ["Master", "Red", "Green", "Blue"];
                ui.horizontal(|ui| {
                    for (i, label) in ch_labels.iter().enumerate() {
                        let selected = self.active_channel == i;
                        let btn_color = if selected {
                            Self::ch_color(i, &colors)
                        } else {
                            colors.text
                        };
                        if ui
                            .add(
                                egui::Button::new(
                                    egui::RichText::new(*label).color(btn_color).strong(),
                                )
                                .fill(if selected {
                                    Self::ch_color(i, &colors).linear_multiply(0.2)
                                } else {
                                    Color32::TRANSPARENT
                                }),
                            )
                            .clicked()
                        {
                            self.active_channel = i;
                        }
                    }
                });
                ui.add_space(4.0);

                // --- Histogram ---
                let ci = self.active_channel;
                let histogram = &self.ch_histograms[ci];
                let hist_max = self.ch_hist_max[ci];
                section_label(ui, &colors, "HISTOGRAM");
                let hist_height = 80.0;
                let hist_rect = ui
                    .allocate_space(Vec2::new(ui.available_width(), hist_height))
                    .1;
                let painter = ui.painter();
                // Background
                painter.rect_filled(
                    hist_rect,
                    CornerRadius::same(3),
                    if colors.is_dark {
                        Color32::from_gray(30)
                    } else {
                        Color32::from_gray(240)
                    },
                );
                // Bars
                let bar_w = hist_rect.width() / 256.0;
                let log_max = (hist_max as f32).ln().max(1.0);
                let bar_color = Self::ch_color(ci, &colors);
                for (i, &count) in histogram.iter().enumerate() {
                    if count == 0 {
                        continue;
                    }
                    let h = ((count as f32).ln() / log_max * hist_height).min(hist_height);
                    let x = hist_rect.min.x + i as f32 * bar_w;
                    let bar = Rect::from_min_max(
                        Pos2::new(x, hist_rect.max.y - h),
                        Pos2::new(x + bar_w.max(1.0), hist_rect.max.y),
                    );
                    painter.rect_filled(bar, 0.0, bar_color.linear_multiply(0.7));
                }
                // Input range markers
                let ib_x = hist_rect.min.x + self.ch_input_black[ci] / 255.0 * hist_rect.width();
                let iw_x = hist_rect.min.x + self.ch_input_white[ci] / 255.0 * hist_rect.width();
                painter.vline(ib_x, hist_rect.y_range(), Stroke::new(2.0, Color32::BLACK));
                painter.vline(iw_x, hist_rect.y_range(), Stroke::new(2.0, Color32::WHITE));
                painter.rect_stroke(
                    hist_rect,
                    CornerRadius::same(3),
                    Stroke::new(1.0, colors.separator),
                    egui::StrokeKind::Middle,
                );

                ui.add_space(4.0);

                // --- Input levels ---
                section_label(ui, &colors, "INPUT LEVELS");
                let mut changed = false;

                egui::Grid::new("levels_input")
                    .num_columns(2)
                    .spacing([8.0, 6.0])
                    .show(ui, |ui| {
                        ui.label("Black Point");
                        if numeric_field_with_buttons(
                            ui,
                            &mut self.ch_input_black[ci],
                            1.0,
                            0.0..=254.0,
                            "",
                            1.0,
                        ) {
                            if self.ch_input_black[ci] >= self.ch_input_white[ci] {
                                self.ch_input_black[ci] = self.ch_input_white[ci] - 1.0;
                            }
                            changed = true;
                        }
                        ui.end_row();

                        ui.label("Gamma");
                        if dialog_slider(ui, &mut self.ch_gamma[ci], 0.1..=10.0, 0.05, "", 2) {
                            changed = true;
                        }
                        ui.end_row();

                        ui.label("White Point");
                        if numeric_field_with_buttons(
                            ui,
                            &mut self.ch_input_white[ci],
                            1.0,
                            1.0..=255.0,
                            "",
                            1.0,
                        ) {
                            if self.ch_input_white[ci] <= self.ch_input_black[ci] {
                                self.ch_input_white[ci] = self.ch_input_black[ci] + 1.0;
                            }
                            changed = true;
                        }
                        ui.end_row();
                    });

                accent_separator(ui, &colors);
                section_label(ui, &colors, "OUTPUT LEVELS");

                egui::Grid::new("levels_output")
                    .num_columns(2)
                    .spacing([8.0, 6.0])
                    .show(ui, |ui| {
                        ui.label("Output Black");
                        if numeric_field_with_buttons(
                            ui,
                            &mut self.ch_output_black[ci],
                            1.0,
                            0.0..=255.0,
                            "",
                            1.0,
                        ) {
                            changed = true;
                        }
                        ui.end_row();

                        ui.label("Output White");
                        if numeric_field_with_buttons(
                            ui,
                            &mut self.ch_output_white[ci],
                            1.0,
                            0.0..=255.0,
                            "",
                            1.0,
                        ) {
                            changed = true;
                        }
                        ui.end_row();
                    });

                // Output gradient bar
                ui.add_space(4.0);
                let bar_rect = ui.allocate_space(Vec2::new(ui.available_width(), 8.0)).1;
                let painter = ui.painter();
                let steps = 32;
                let bar_w = bar_rect.width() / steps as f32;
                for i in 0..steps {
                    let t = i as f32 / (steps - 1) as f32;
                    let v = (self.ch_output_black[ci]
                        + t * (self.ch_output_white[ci] - self.ch_output_black[ci]))
                        .clamp(0.0, 255.0) as u8;
                    let r = Rect::from_min_size(
                        Pos2::new(bar_rect.min.x + i as f32 * bar_w, bar_rect.min.y),
                        Vec2::new(bar_w + 0.5, 8.0),
                    );
                    painter.rect_filled(r, 0.0, Color32::from_gray(v));
                }
                painter.rect_stroke(
                    bar_rect,
                    CornerRadius::same(2),
                    Stroke::new(1.0, colors.separator),
                    egui::StrokeKind::Middle,
                );

                accent_separator(ui, &colors);
                let manual_preview = preview_controls(ui, &colors, &mut self.live_preview);
                if changed && self.live_preview {
                    result = DialogResult::Changed;
                }
                if manual_preview {
                    result = DialogResult::Changed;
                }

                let (ok, cancel, reset) = dialog_footer_with_reset(ui, &colors);
                if reset {
                    let c = self.active_channel;
                    self.ch_input_black[c] = 0.0;
                    self.ch_input_white[c] = 255.0;
                    self.ch_gamma[c] = 1.0;
                    self.ch_output_black[c] = 0.0;
                    self.ch_output_white[c] = 255.0;
                    result = DialogResult::Changed;
                }
                if ok {
                    result = DialogResult::Ok(self.as_result());
                }
                if cancel {
                    result = DialogResult::Cancel;
                }
            });
        result
    }
}

// ============================================================================
// CURVES DIALOG — interactive multi-channel point-based curve editor
// ============================================================================

/// Index constants for curve channels.
const CH_RGB: usize = 0;
const CH_RED: usize = 1;
const CH_GREEN: usize = 2;
const CH_BLUE: usize = 3;
const CH_ALPHA: usize = 4;
const CHANNEL_COUNT: usize = 5;

/// Per-channel curve data.
#[derive(Clone)]
pub struct CurveChannel {
    /// Control points [(input, output)] in 0..255, sorted by input.
    pub points: Vec<(f32, f32)>,
    /// Whether this channel's curve is displayed on the graph.
    pub visible: bool,
    /// Whether this channel's curve is applied during preview/commit.
    pub enabled: bool,
}

impl Default for CurveChannel {
    fn default() -> Self {
        Self {
            points: vec![(0.0, 0.0), (255.0, 255.0)],
            visible: true,
            enabled: true,
        }
    }
}

/// Multi-channel curves result data: [RGB, R, G, B, A] LUTs.
pub type CurvesChannelData = [CurveChannel; CHANNEL_COUNT];

pub struct CurvesDialog {
    /// Per-channel curve data: [RGB, R, G, B, A].
    pub channels: [CurveChannel; CHANNEL_COUNT],
    /// Which channel is actively being edited.
    pub active_channel: usize,
    pub original_pixels: Option<TiledImage>,
    pub original_flat: Option<image::RgbaImage>,
    pub layer_idx: usize,
    pub live_preview: bool,
    /// Index of the point currently being dragged (None if idle).
    dragging_point: Option<usize>,
}

impl CurvesDialog {
    pub fn new(state: &CanvasState) -> Self {
        let idx = state.active_layer_index;
        Self {
            channels: Default::default(),
            active_channel: CH_RGB,
            original_pixels: state.layers.get(idx).map(|l| l.pixels.clone()),
            original_flat: state.layers.get(idx).map(|l| l.pixels.to_rgba_image()),
            layer_idx: idx,
            live_preview: true,
            dragging_point: None,
        }
    }

    /// Get the color associated with a channel index.
    fn channel_color(ch: usize, colors: &DialogColors) -> Color32 {
        match ch {
            CH_RGB => colors.accent,
            CH_RED => Color32::from_rgb(220, 60, 60),
            CH_GREEN => Color32::from_rgb(60, 180, 60),
            CH_BLUE => Color32::from_rgb(70, 100, 220),
            CH_ALPHA => Color32::from_gray(160),
            _ => colors.accent,
        }
    }

    fn channel_label(ch: usize) -> &'static str {
        match ch {
            CH_RGB => "RGB",
            CH_RED => "Red",
            CH_GREEN => "Green",
            CH_BLUE => "Blue",
            CH_ALPHA => "Alpha",
            _ => "?",
        }
    }

    pub fn show(&mut self, ctx: &egui::Context) -> DialogResult<CurvesChannelData> {
        let mut result = DialogResult::Open;
        let colors = DialogColors::from_ctx(ctx);

        crate::ui::polish::window(ctx, "dialog_curves")
            .title_bar(false)
            .collapsible(false)
            .resizable(false)
            .default_pos(egui::pos2(ctx.content_rect().center().x - 195.0, 40.0))
            .show(ctx, |ui| {
                ui.set_min_width(390.0);
                if paint_dialog_header(ui, &colors, "📈", &t!("dialog.curves")) {
                    result = DialogResult::Cancel;
                }
                ui.add_space(4.0);

                // --- Channel selector row ---
                section_label(ui, &colors, "CHANNEL");
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 3.0;
                    for ch in 0..CHANNEL_COUNT {
                        let label = Self::channel_label(ch);
                        let ch_color = Self::channel_color(ch, &colors);
                        let is_active = self.active_channel == ch;
                        let is_enabled = self.channels[ch].enabled;
                        let _is_visible = self.channels[ch].visible;

                        // Channel button: click = select, shows active state
                        let text = egui::RichText::new(label).size(11.0).color(if !is_enabled {
                            colors.text_muted
                        } else if is_active {
                            Color32::WHITE
                        } else {
                            ch_color
                        });
                        let btn = if is_active {
                            egui::Button::new(text.strong()).fill(ch_color.linear_multiply(0.7))
                        } else {
                            egui::Button::new(text)
                        };
                        if ui.add(btn).clicked() {
                            self.active_channel = ch;
                        }
                    }
                    ui.add_space(8.0);
                    // Visibility/enable toggles for current channel
                    let ch = self.active_channel;
                    let eye_text = if self.channels[ch].visible {
                        "👁"
                    } else {
                        "○"
                    };
                    if ui
                        .small_button(eye_text)
                        .on_hover_text("Toggle curve visibility on graph")
                        .clicked()
                    {
                        self.channels[ch].visible = !self.channels[ch].visible;
                    }
                    let en_text = if self.channels[ch].enabled {
                        "✓"
                    } else {
                        "✗"
                    };
                    let en_color = if self.channels[ch].enabled {
                        colors.accent
                    } else {
                        colors.text_muted
                    };
                    if ui
                        .add(
                            egui::Button::new(
                                egui::RichText::new(en_text).size(12.0).color(en_color),
                            )
                            .small(),
                        )
                        .on_hover_text("Toggle channel enabled/disabled")
                        .clicked()
                    {
                        self.channels[ch].enabled = !self.channels[ch].enabled;
                        result = DialogResult::Changed;
                    }
                });
                ui.add_space(2.0);

                // Instructions
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 4.0;
                    ui.label(
                        egui::RichText::new(
                            "Click to add points • Right-click to remove • Drag to adjust",
                        )
                        .size(10.0)
                        .color(colors.text_muted),
                    );
                });
                ui.add_space(4.0);

                // --- Curve editor canvas ---
                let canvas_size = 256.0;
                let (response, painter) = ui.allocate_painter(
                    Vec2::new(canvas_size + 2.0, canvas_size + 2.0),
                    Sense::click_and_drag(),
                );
                let canvas_rect = Rect::from_min_size(
                    response.rect.min + Vec2::new(1.0, 1.0),
                    Vec2::new(canvas_size, canvas_size),
                );

                // Background
                painter.rect_filled(
                    canvas_rect,
                    CornerRadius::same(2),
                    if colors.is_dark {
                        Color32::from_gray(25)
                    } else {
                        Color32::from_gray(245)
                    },
                );

                // Grid lines
                let grid_color = if colors.is_dark {
                    Color32::from_gray(50)
                } else {
                    Color32::from_gray(210)
                };
                for i in 1..4 {
                    let t = i as f32 / 4.0;
                    let x = canvas_rect.min.x + t * canvas_size;
                    let y = canvas_rect.min.y + t * canvas_size;
                    painter.line_segment(
                        [
                            Pos2::new(x, canvas_rect.min.y),
                            Pos2::new(x, canvas_rect.max.y),
                        ],
                        Stroke::new(0.5, grid_color),
                    );
                    painter.line_segment(
                        [
                            Pos2::new(canvas_rect.min.x, y),
                            Pos2::new(canvas_rect.max.x, y),
                        ],
                        Stroke::new(0.5, grid_color),
                    );
                }

                // Diagonal reference line
                let diag_color = if colors.is_dark {
                    Color32::from_gray(60)
                } else {
                    Color32::from_gray(180)
                };
                painter.line_segment(
                    [
                        Pos2::new(canvas_rect.min.x, canvas_rect.max.y),
                        Pos2::new(canvas_rect.max.x, canvas_rect.min.y),
                    ],
                    Stroke::new(1.0, diag_color),
                );

                // Draw all visible channel curves (inactive ones first, active on top)
                let draw_order: Vec<usize> = (0..CHANNEL_COUNT)
                    .filter(|&ch| ch != self.active_channel && self.channels[ch].visible)
                    .chain(
                        std::iter::once(self.active_channel)
                            .filter(|_| self.channels[self.active_channel].visible),
                    )
                    .collect();

                for &ch in &draw_order {
                    let lut =
                        crate::ops::adjustments::build_curves_lut_pub(&self.channels[ch].points);
                    let ch_color = Self::channel_color(ch, &colors);
                    let is_active = ch == self.active_channel;
                    let stroke_w = if is_active { 2.0 } else { 1.2 };
                    let alpha = if is_active { 1.0 } else { 0.45 };

                    let mut curve_points = Vec::with_capacity(256);
                    for (i, &lv) in lut.iter().enumerate() {
                        let x = canvas_rect.min.x + i as f32 / 255.0 * canvas_size;
                        let y = canvas_rect.max.y - lv as f32 / 255.0 * canvas_size;
                        curve_points.push(Pos2::new(x, y));
                    }
                    for w in curve_points.windows(2) {
                        painter.line_segment(
                            [w[0], w[1]],
                            Stroke::new(stroke_w, ch_color.linear_multiply(alpha)),
                        );
                    }
                }

                // Draw control points for active channel only
                let active_ch = self.active_channel;
                let point_color = Self::channel_color(active_ch, &colors);
                let point_outline = if colors.is_dark {
                    Color32::WHITE
                } else {
                    Color32::BLACK
                };
                for (i, &(px, py)) in self.channels[active_ch].points.iter().enumerate() {
                    let screen_x = canvas_rect.min.x + px / 255.0 * canvas_size;
                    let screen_y = canvas_rect.max.y - py / 255.0 * canvas_size;
                    let radius = if self.dragging_point == Some(i) {
                        6.0
                    } else {
                        5.0
                    };
                    painter.circle_filled(Pos2::new(screen_x, screen_y), radius, point_color);
                    painter.circle_stroke(
                        Pos2::new(screen_x, screen_y),
                        radius,
                        Stroke::new(1.5, point_outline),
                    );
                }

                // Handle interaction — operates on active channel
                let mut changed = false;
                let points = &mut self.channels[active_ch].points;
                if let Some(pos) = response.interact_pointer_pos() {
                    let rel_x =
                        ((pos.x - canvas_rect.min.x) / canvas_size * 255.0).clamp(0.0, 255.0);
                    let rel_y =
                        ((canvas_rect.max.y - pos.y) / canvas_size * 255.0).clamp(0.0, 255.0);

                    if response.drag_started() {
                        // Check if clicking near an existing point
                        let mut closest = None;
                        let mut closest_dist = f32::MAX;
                        for (i, &(px, py)) in points.iter().enumerate() {
                            let dx = rel_x - px;
                            let dy = rel_y - py;
                            let dist = (dx * dx + dy * dy).sqrt();
                            if dist < closest_dist {
                                closest_dist = dist;
                                closest = Some(i);
                            }
                        }
                        if closest_dist < 15.0 {
                            self.dragging_point = closest;
                        } else {
                            // Add new point
                            points.push((rel_x, rel_y));
                            points.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                            for (i, &(px, _)) in points.iter().enumerate() {
                                if (px - rel_x).abs() < 0.5 {
                                    self.dragging_point = Some(i);
                                    break;
                                }
                            }
                            changed = true;
                        }
                    }

                    if response.dragged()
                        && let Some(idx) = self.dragging_point
                    {
                        if idx > 0 && idx < points.len() - 1 {
                            points[idx] = (
                                rel_x.clamp(points[idx - 1].0 + 1.0, points[idx + 1].0 - 1.0),
                                rel_y,
                            );
                        } else {
                            points[idx].1 = rel_y;
                        }
                        changed = true;
                    }
                }

                if response.drag_stopped() {
                    self.dragging_point = None;
                }

                // Right-click to remove point
                if response.secondary_clicked()
                    && let Some(pos) = response.interact_pointer_pos()
                {
                    let rel_x =
                        ((pos.x - canvas_rect.min.x) / canvas_size * 255.0).clamp(0.0, 255.0);
                    let rel_y =
                        ((canvas_rect.max.y - pos.y) / canvas_size * 255.0).clamp(0.0, 255.0);
                    let mut closest = None;
                    let mut closest_dist = f32::MAX;
                    for (i, &(px, py)) in points.iter().enumerate() {
                        if i == 0 || i == points.len() - 1 {
                            continue;
                        }
                        let dx = rel_x - px;
                        let dy = rel_y - py;
                        let dist = (dx * dx + dy * dy).sqrt();
                        if dist < closest_dist {
                            closest_dist = dist;
                            closest = Some(i);
                        }
                    }
                    if closest_dist < 15.0
                        && let Some(idx) = closest
                    {
                        points.remove(idx);
                        changed = true;
                    }
                }

                painter.rect_stroke(
                    canvas_rect,
                    CornerRadius::same(2),
                    Stroke::new(1.0, colors.separator),
                    egui::StrokeKind::Middle,
                );

                // Channel info
                ui.add_space(2.0);
                let ch_label = Self::channel_label(active_ch);
                ui.label(
                    egui::RichText::new(format!(
                        "{} — {} control points",
                        ch_label,
                        self.channels[active_ch].points.len()
                    ))
                    .size(10.0)
                    .color(colors.text_muted),
                );

                // Presets (apply to active channel)
                ui.add_space(2.0);
                section_label(ui, &colors, "PRESETS");
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 4.0;
                    if ui
                        .button(egui::RichText::new("Linear").size(11.0))
                        .clicked()
                    {
                        self.channels[active_ch].points = vec![(0.0, 0.0), (255.0, 255.0)];
                        changed = true;
                    }
                    if ui
                        .button(egui::RichText::new("Brighten").size(11.0))
                        .clicked()
                    {
                        self.channels[active_ch].points =
                            vec![(0.0, 0.0), (128.0, 180.0), (255.0, 255.0)];
                        changed = true;
                    }
                    if ui
                        .button(egui::RichText::new("Darken").size(11.0))
                        .clicked()
                    {
                        self.channels[active_ch].points =
                            vec![(0.0, 0.0), (128.0, 80.0), (255.0, 255.0)];
                        changed = true;
                    }
                    if ui
                        .button(egui::RichText::new("S-Curve").size(11.0))
                        .clicked()
                    {
                        self.channels[active_ch].points =
                            vec![(0.0, 0.0), (64.0, 40.0), (192.0, 215.0), (255.0, 255.0)];
                        changed = true;
                    }
                    if ui
                        .button(egui::RichText::new("Negative").size(11.0))
                        .clicked()
                    {
                        self.channels[active_ch].points = vec![(0.0, 255.0), (255.0, 0.0)];
                        changed = true;
                    }
                });

                accent_separator(ui, &colors);
                let manual_preview = preview_controls(ui, &colors, &mut self.live_preview);
                if changed && self.live_preview {
                    result = DialogResult::Changed;
                }
                if manual_preview {
                    result = DialogResult::Changed;
                }

                let (ok, cancel, reset) = dialog_footer_with_reset(ui, &colors);
                if reset {
                    for ch in &mut self.channels {
                        ch.points = vec![(0.0, 0.0), (255.0, 255.0)];
                        ch.enabled = true;
                        ch.visible = true;
                    }
                    self.active_channel = CH_RGB;
                    result = DialogResult::Changed;
                }
                if ok {
                    result = DialogResult::Ok(self.channels.clone());
                }
                if cancel {
                    result = DialogResult::Cancel;
                }
            });
        result
    }
}

// ============================================================================
// TEMPERATURE / TINT DIALOG
// ============================================================================

pub struct TemperatureTintDialog {
    pub temperature: f32,
    pub tint: f32,
    pub original_pixels: Option<TiledImage>,
    pub original_flat: Option<image::RgbaImage>,
    pub layer_idx: usize,
    pub live_preview: bool,
}

impl TemperatureTintDialog {
    pub fn new(state: &CanvasState) -> Self {
        let idx = state.active_layer_index;
        Self {
            temperature: 0.0,
            tint: 0.0,
            original_pixels: state.layers.get(idx).map(|l| l.pixels.clone()),
            original_flat: state.layers.get(idx).map(|l| l.pixels.to_rgba_image()),
            layer_idx: idx,
            live_preview: true,
        }
    }

    pub fn show(&mut self, ctx: &egui::Context) -> DialogResult<(f32, f32)> {
        let mut result = DialogResult::Open;
        let colors = DialogColors::from_ctx(ctx);

        crate::ui::polish::window(ctx, "dialog_temperature_tint")
            .title_bar(false)
            .collapsible(false)
            .resizable(false)
            .default_pos(egui::pos2(ctx.content_rect().center().x - 175.0, 60.0))
            .show(ctx, |ui| {
                ui.set_min_width(360.0);
                if paint_dialog_header(ui, &colors, "☀", &t!("dialog.temperature_tint")) {
                    result = DialogResult::Cancel;
                }
                ui.add_space(4.0);

                section_label(ui, &colors, "WHITE BALANCE");
                let mut changed = false;

                egui::Grid::new("tt_params")
                    .num_columns(2)
                    .spacing([8.0, 6.0])
                    .show(ui, |ui| {
                        ui.label("Temperature");
                        if dialog_slider(ui, &mut self.temperature, -100.0..=100.0, 1.0, "", 0) {
                            changed = true;
                        }
                        ui.end_row();

                        ui.label("Tint");
                        if dialog_slider(ui, &mut self.tint, -100.0..=100.0, 1.0, "", 0) {
                            changed = true;
                        }
                        ui.end_row();
                    });

                // Temperature gradient bar: blue ← → orange  (interactive)
                ui.add_space(4.0);
                let bar_rect = ui.allocate_space(Vec2::new(ui.available_width(), 18.0)).1;
                let temp_bar_response =
                    ui.interact(bar_rect, ui.id().with("temp_bar"), Sense::click_and_drag());
                let steps = 48;
                let bar_w = bar_rect.width() / steps as f32;
                // Handle drag on temperature bar
                if (temp_bar_response.dragged() || temp_bar_response.clicked())
                    && let Some(pos) = temp_bar_response.interact_pointer_pos()
                {
                    let t = ((pos.x - bar_rect.min.x) / bar_rect.width()).clamp(0.0, 1.0);
                    self.temperature = (t - 0.5) * 200.0; // map 0..1 → -100..100
                    changed = true;
                }
                {
                    let painter = ui.painter();
                    for i in 0..steps {
                        let t = i as f32 / (steps - 1) as f32; // 0..1
                        // Blue → White → Orange
                        let (r, g, b) = if t < 0.5 {
                            let s = t * 2.0; // 0..1
                            (100.0 + s * 155.0, 140.0 + s * 115.0, 255.0)
                        } else {
                            let s = (t - 0.5) * 2.0; // 0..1
                            (255.0, 255.0 - s * 100.0, 255.0 - s * 200.0)
                        };
                        let color = Color32::from_rgb(r as u8, g as u8, b.max(0.0) as u8);
                        let rect = Rect::from_min_size(
                            Pos2::new(bar_rect.min.x + i as f32 * bar_w, bar_rect.min.y),
                            Vec2::new(bar_w + 0.5, bar_rect.height()),
                        );
                        painter.rect_filled(rect, 0.0, color);
                    }
                    // Position indicator (thumb)
                    let indicator_x =
                        bar_rect.min.x + (self.temperature / 200.0 + 0.5) * bar_rect.width();
                    let ix = indicator_x.clamp(bar_rect.min.x, bar_rect.max.x);
                    painter.rect_filled(
                        Rect::from_center_size(
                            Pos2::new(ix, bar_rect.center().y),
                            Vec2::new(4.0, bar_rect.height() + 4.0),
                        ),
                        CornerRadius::same(2),
                        Color32::WHITE,
                    );
                    painter.rect_stroke(
                        Rect::from_center_size(
                            Pos2::new(ix, bar_rect.center().y),
                            Vec2::new(4.0, bar_rect.height() + 4.0),
                        ),
                        CornerRadius::same(2),
                        Stroke::new(1.0, Color32::BLACK),
                        egui::StrokeKind::Middle,
                    );
                    painter.rect_stroke(
                        bar_rect,
                        CornerRadius::same(2),
                        Stroke::new(1.0, colors.separator),
                        egui::StrokeKind::Middle,
                    );
                }

                // Tint gradient bar: green ← → magenta  (interactive)
                ui.add_space(4.0);
                let bar_rect2 = ui.allocate_space(Vec2::new(ui.available_width(), 18.0)).1;
                let tint_bar_response =
                    ui.interact(bar_rect2, ui.id().with("tint_bar"), Sense::click_and_drag());
                // Handle drag on tint bar
                if (tint_bar_response.dragged() || tint_bar_response.clicked())
                    && let Some(pos) = tint_bar_response.interact_pointer_pos()
                {
                    let t = ((pos.x - bar_rect2.min.x) / bar_rect2.width()).clamp(0.0, 1.0);
                    self.tint = (t - 0.5) * 200.0;
                    changed = true;
                }
                {
                    let painter = ui.painter();
                    for i in 0..steps {
                        let t = i as f32 / (steps - 1) as f32;
                        let (r, g, b) = if t < 0.5 {
                            let s = t * 2.0;
                            (100.0 + s * 155.0, 200.0, 100.0 + s * 155.0)
                        } else {
                            let s = (t - 0.5) * 2.0;
                            (255.0, 200.0 - s * 100.0, 255.0)
                        };
                        let color = Color32::from_rgb(r as u8, g as u8, b as u8);
                        let rect = Rect::from_min_size(
                            Pos2::new(bar_rect2.min.x + i as f32 * bar_w, bar_rect2.min.y),
                            Vec2::new(bar_w + 0.5, bar_rect2.height()),
                        );
                        painter.rect_filled(rect, 0.0, color);
                    }
                    let tint_x = bar_rect2.min.x + (self.tint / 200.0 + 0.5) * bar_rect2.width();
                    let tx = tint_x.clamp(bar_rect2.min.x, bar_rect2.max.x);
                    painter.rect_filled(
                        Rect::from_center_size(
                            Pos2::new(tx, bar_rect2.center().y),
                            Vec2::new(4.0, bar_rect2.height() + 4.0),
                        ),
                        CornerRadius::same(2),
                        Color32::WHITE,
                    );
                    painter.rect_stroke(
                        Rect::from_center_size(
                            Pos2::new(tx, bar_rect2.center().y),
                            Vec2::new(4.0, bar_rect2.height() + 4.0),
                        ),
                        CornerRadius::same(2),
                        Stroke::new(1.0, Color32::BLACK),
                        egui::StrokeKind::Middle,
                    );
                    painter.rect_stroke(
                        bar_rect2,
                        CornerRadius::same(2),
                        Stroke::new(1.0, colors.separator),
                        egui::StrokeKind::Middle,
                    );
                }

                accent_separator(ui, &colors);
                let manual_preview = preview_controls(ui, &colors, &mut self.live_preview);
                if changed && self.live_preview {
                    result = DialogResult::Changed;
                }
                if manual_preview {
                    result = DialogResult::Changed;
                }

                let (ok, cancel, reset) = dialog_footer_with_reset(ui, &colors);
                if reset {
                    self.temperature = 0.0;
                    self.tint = 0.0;
                    result = DialogResult::Changed;
                }
                if ok {
                    result = DialogResult::Ok((self.temperature, self.tint));
                }
                if cancel {
                    result = DialogResult::Cancel;
                }
            });
        result
    }
}

// ============================================================================
// THRESHOLD DIALOG
// ============================================================================

pub struct ThresholdDialog {
    pub level: f32,
    pub original_pixels: Option<TiledImage>,
    pub original_flat: Option<image::RgbaImage>,
    pub layer_idx: usize,
    pub live_preview: bool,
}

impl ThresholdDialog {
    pub fn new(state: &CanvasState) -> Self {
        let idx = state.active_layer_index;
        Self {
            level: 128.0,
            original_pixels: state.layers.get(idx).map(|l| l.pixels.clone()),
            original_flat: state.layers.get(idx).map(|l| l.pixels.to_rgba_image()),
            layer_idx: idx,
            live_preview: true,
        }
    }

    pub fn show(&mut self, ctx: &egui::Context) -> DialogResult<f32> {
        let mut result = DialogResult::Open;
        let colors = DialogColors::from_ctx(ctx);

        crate::ui::polish::window(ctx, "dialog_threshold")
            .title_bar(false)
            .collapsible(false)
            .resizable(false)
            .default_pos(egui::pos2(ctx.content_rect().center().x - 160.0, 60.0))
            .show(ctx, |ui| {
                ui.set_min_width(320.0);
                if paint_dialog_header(ui, &colors, "◑", &t!("dialog.threshold")) {
                    result = DialogResult::Cancel;
                }
                ui.add_space(4.0);

                section_label(ui, &colors, "THRESHOLD");
                let mut changed = false;

                egui::Grid::new("threshold_params")
                    .num_columns(2)
                    .spacing([8.0, 6.0])
                    .show(ui, |ui| {
                        ui.label("Level");
                        if dialog_slider(ui, &mut self.level, 0.0..=255.0, 1.0, "", 0) {
                            changed = true;
                        }
                        ui.end_row();
                    });

                // Threshold bar visualization
                ui.add_space(4.0);
                let bar_rect = ui.allocate_space(Vec2::new(ui.available_width(), 8.0)).1;
                let painter = ui.painter();
                let split = self.level / 255.0;
                painter.rect_filled(
                    Rect::from_min_size(
                        bar_rect.min,
                        Vec2::new(bar_rect.width() * split + 0.5, 8.0),
                    ),
                    CornerRadius::ZERO,
                    Color32::BLACK,
                );
                painter.rect_filled(
                    Rect::from_min_size(
                        Pos2::new(bar_rect.min.x + bar_rect.width() * split, bar_rect.min.y),
                        Vec2::new(bar_rect.width() * (1.0 - split), 8.0),
                    ),
                    CornerRadius::ZERO,
                    Color32::WHITE,
                );
                painter.rect_stroke(
                    bar_rect,
                    CornerRadius::same(2),
                    Stroke::new(1.0, colors.separator),
                    egui::StrokeKind::Middle,
                );

                accent_separator(ui, &colors);
                let manual_preview = preview_controls(ui, &colors, &mut self.live_preview);
                if changed && self.live_preview {
                    result = DialogResult::Changed;
                }
                if manual_preview {
                    result = DialogResult::Changed;
                }

                let (ok, cancel, reset) = dialog_footer_with_reset(ui, &colors);
                if reset {
                    self.level = 128.0;
                    result = DialogResult::Changed;
                }
                if ok {
                    result = DialogResult::Ok(self.level);
                }
                if cancel {
                    result = DialogResult::Cancel;
                }
            });
        result
    }
}

// ============================================================================
// POSTERIZE DIALOG
// ============================================================================

pub struct PosterizeDialog {
    pub levels: u32,
    pub original_pixels: Option<TiledImage>,
    pub original_flat: Option<image::RgbaImage>,
    pub layer_idx: usize,
    pub live_preview: bool,
}

impl PosterizeDialog {
    pub fn new(state: &CanvasState) -> Self {
        let idx = state.active_layer_index;
        Self {
            levels: 4,
            original_pixels: state.layers.get(idx).map(|l| l.pixels.clone()),
            original_flat: state.layers.get(idx).map(|l| l.pixels.to_rgba_image()),
            layer_idx: idx,
            live_preview: true,
        }
    }

    pub fn show(&mut self, ctx: &egui::Context) -> DialogResult<u32> {
        let mut result = DialogResult::Open;
        let colors = DialogColors::from_ctx(ctx);

        crate::ui::polish::window(ctx, "dialog_posterize")
            .title_bar(false)
            .collapsible(false)
            .resizable(false)
            .default_pos(egui::pos2(ctx.content_rect().center().x - 160.0, 60.0))
            .show(ctx, |ui| {
                ui.set_min_width(320.0);
                if paint_dialog_header(ui, &colors, "🎨", &t!("dialog.posterize")) {
                    result = DialogResult::Cancel;
                }
                ui.add_space(4.0);

                section_label(ui, &colors, "POSTERIZE");
                let mut changed = false;
                let mut levels_i32 = self.levels as i32;

                egui::Grid::new("posterize_params")
                    .num_columns(2)
                    .spacing([8.0, 6.0])
                    .show(ui, |ui| {
                        ui.label("Levels");
                        {
                            let mut lf = levels_i32 as f32;
                            if dialog_slider(ui, &mut lf, 2.0..=16.0, 1.0, "", 0) {
                                levels_i32 = lf.round() as i32;
                                self.levels = levels_i32 as u32;
                                changed = true;
                            }
                        }
                        ui.end_row();
                    });

                // Posterize preview bar
                ui.add_space(4.0);
                let bar_rect = ui.allocate_space(Vec2::new(ui.available_width(), 8.0)).1;
                let painter = ui.painter();
                let steps = self.levels as usize;
                let bar_w = bar_rect.width() / steps as f32;
                for i in 0..steps {
                    let v = (i as f32 / (steps - 1).max(1) as f32 * 255.0) as u8;
                    let r = Rect::from_min_size(
                        Pos2::new(bar_rect.min.x + i as f32 * bar_w, bar_rect.min.y),
                        Vec2::new(bar_w + 0.5, 8.0),
                    );
                    painter.rect_filled(r, 0.0, Color32::from_gray(v));
                }
                painter.rect_stroke(
                    bar_rect,
                    CornerRadius::same(2),
                    Stroke::new(1.0, colors.separator),
                    egui::StrokeKind::Middle,
                );

                accent_separator(ui, &colors);
                let manual_preview = preview_controls(ui, &colors, &mut self.live_preview);
                if changed && self.live_preview {
                    result = DialogResult::Changed;
                }
                if manual_preview {
                    result = DialogResult::Changed;
                }

                let (ok, cancel, reset) = dialog_footer_with_reset(ui, &colors);
                if reset {
                    self.levels = 4;
                    result = DialogResult::Changed;
                }
                if ok {
                    result = DialogResult::Ok(self.levels);
                }
                if cancel {
                    result = DialogResult::Cancel;
                }
            });
        result
    }
}

// ============================================================================
// COLOR BALANCE DIALOG
// ============================================================================

pub struct ColorBalanceDialog {
    pub shadows: [f32; 3],
    pub midtones: [f32; 3],
    pub highlights: [f32; 3],
    pub active_zone: u8, // 0=Shadows 1=Midtones 2=Highlights
    pub original_pixels: Option<TiledImage>,
    pub original_flat: Option<image::RgbaImage>,
    pub layer_idx: usize,
    pub live_preview: bool,
}

impl ColorBalanceDialog {
    pub fn new(state: &CanvasState) -> Self {
        let idx = state.active_layer_index;
        Self {
            shadows: [0.0; 3],
            midtones: [0.0; 3],
            highlights: [0.0; 3],
            active_zone: 1,
            original_pixels: state.layers.get(idx).map(|l| l.pixels.clone()),
            original_flat: state.layers.get(idx).map(|l| l.pixels.to_rgba_image()),
            layer_idx: idx,
            live_preview: true,
        }
    }

    pub fn show(&mut self, ctx: &egui::Context) -> DialogResult<([f32; 3], [f32; 3], [f32; 3])> {
        let mut result = DialogResult::Open;
        let colors = DialogColors::from_ctx(ctx);

        crate::ui::polish::window(ctx, "dialog_color_balance")
            .title_bar(false)
            .collapsible(false)
            .resizable(false)
            .default_pos(egui::pos2(ctx.content_rect().center().x - 185.0, 60.0))
            .show(ctx, |ui| {
                ui.set_min_width(370.0);
                if paint_dialog_header(ui, &colors, "⚖", &t!("dialog.color_balance")) {
                    result = DialogResult::Cancel;
                }
                ui.add_space(4.0);

                // Zone selector
                ui.horizontal(|ui| {
                    for (i, label) in ["Shadows", "Midtones", "Highlights"].iter().enumerate() {
                        let active = self.active_zone == i as u8;
                        let btn = if active {
                            egui::Button::new(*label).fill(colors.accent_faint)
                        } else {
                            egui::Button::new(*label)
                        };
                        if ui.add(btn).clicked() {
                            self.active_zone = i as u8;
                        }
                    }
                });
                ui.add_space(4.0);

                let zone = match self.active_zone {
                    0 => &mut self.shadows,
                    2 => &mut self.highlights,
                    _ => &mut self.midtones,
                };

                section_label(ui, &colors, "RGB BALANCE");
                let mut changed = false;

                egui::Grid::new("cb_params")
                    .num_columns(2)
                    .spacing([8.0, 6.0])
                    .show(ui, |ui| {
                        ui.label(
                            egui::RichText::new("Cyan — Red")
                                .color(Color32::from_rgb(200, 100, 100)),
                        );
                        if dialog_slider(ui, &mut zone[0], -100.0..=100.0, 1.0, "", 0) {
                            changed = true;
                        }
                        ui.end_row();

                        ui.label(
                            egui::RichText::new("Magenta — Green")
                                .color(Color32::from_rgb(100, 180, 100)),
                        );
                        if dialog_slider(ui, &mut zone[1], -100.0..=100.0, 1.0, "", 0) {
                            changed = true;
                        }
                        ui.end_row();

                        ui.label(
                            egui::RichText::new("Yellow — Blue")
                                .color(Color32::from_rgb(100, 140, 220)),
                        );
                        if dialog_slider(ui, &mut zone[2], -100.0..=100.0, 1.0, "", 0) {
                            changed = true;
                        }
                        ui.end_row();
                    });

                accent_separator(ui, &colors);
                let manual_preview = preview_controls(ui, &colors, &mut self.live_preview);
                if changed && self.live_preview {
                    result = DialogResult::Changed;
                }
                if manual_preview {
                    result = DialogResult::Changed;
                }

                let (ok, cancel, reset) = dialog_footer_with_reset(ui, &colors);
                if reset {
                    self.shadows = [0.0; 3];
                    self.midtones = [0.0; 3];
                    self.highlights = [0.0; 3];
                    result = DialogResult::Changed;
                }
                if ok {
                    result = DialogResult::Ok((self.shadows, self.midtones, self.highlights));
                }
                if cancel {
                    result = DialogResult::Cancel;
                }
            });
        result
    }
}

// ============================================================================
// GRADIENT MAP DIALOG
// ============================================================================

/// Presets for the gradient map
#[derive(Clone, Copy, PartialEq)]
pub enum GradientMapPreset {
    BlackToWhite,
    WhiteToBlack,
    BlackToColor,
    Sepia,
    Infrared,
    Custom,
}

pub struct GradientMapDialog {
    pub shadow_color: [u8; 3],
    pub highlight_color: [u8; 3],
    pub preset: GradientMapPreset,
    pub original_pixels: Option<TiledImage>,
    pub original_flat: Option<image::RgbaImage>,
    pub layer_idx: usize,
    pub live_preview: bool,
}

impl GradientMapDialog {
    pub fn new(state: &CanvasState) -> Self {
        let idx = state.active_layer_index;
        Self {
            shadow_color: [0, 0, 0],
            highlight_color: [255, 255, 255],
            preset: GradientMapPreset::BlackToWhite,
            original_pixels: state.layers.get(idx).map(|l| l.pixels.clone()),
            original_flat: state.layers.get(idx).map(|l| l.pixels.to_rgba_image()),
            layer_idx: idx,
            live_preview: true,
        }
    }

    pub fn build_lut(&self) -> [[u8; 4]; 256] {
        let mut lut = [[0u8; 4]; 256];
        let [sr, sg, sb] = self.shadow_color;
        let [hr, hg, hb] = self.highlight_color;
        for (i, item) in lut.iter_mut().enumerate() {
            let t = i as f32 / 255.0;
            *item = [
                (sr as f32 + (hr as f32 - sr as f32) * t) as u8,
                (sg as f32 + (hg as f32 - sg as f32) * t) as u8,
                (sb as f32 + (hb as f32 - sb as f32) * t) as u8,
                255,
            ];
        }
        lut
    }

    fn apply_preset(&mut self, preset: GradientMapPreset) {
        self.preset = preset;
        match preset {
            GradientMapPreset::BlackToWhite => {
                self.shadow_color = [0, 0, 0];
                self.highlight_color = [255, 255, 255];
            }
            GradientMapPreset::WhiteToBlack => {
                self.shadow_color = [255, 255, 255];
                self.highlight_color = [0, 0, 0];
            }
            GradientMapPreset::BlackToColor => {
                self.shadow_color = [0, 0, 0];
                self.highlight_color = [80, 180, 255];
            }
            GradientMapPreset::Sepia => {
                self.shadow_color = [60, 30, 0];
                self.highlight_color = [230, 210, 170];
            }
            GradientMapPreset::Infrared => {
                self.shadow_color = [0, 0, 80];
                self.highlight_color = [255, 50, 0];
            }
            GradientMapPreset::Custom => {}
        }
    }

    pub fn show(&mut self, ctx: &egui::Context) -> DialogResult<[[u8; 4]; 256]> {
        let mut result = DialogResult::Open;
        let colors = DialogColors::from_ctx(ctx);

        crate::ui::polish::window(ctx, "dialog_gradient_map")
            .title_bar(false)
            .collapsible(false)
            .resizable(false)
            .default_pos(egui::pos2(ctx.content_rect().center().x - 185.0, 60.0))
            .show(ctx, |ui| {
                ui.set_min_width(370.0);
                if paint_dialog_header(ui, &colors, "🌈", &t!("dialog.gradient_map")) {
                    result = DialogResult::Cancel;
                }
                ui.add_space(4.0);

                // Preset buttons
                section_label(ui, &colors, "PRESETS");
                ui.horizontal_wrapped(|ui| {
                    let presets = [
                        (GradientMapPreset::BlackToWhite, "B→W"),
                        (GradientMapPreset::WhiteToBlack, "W→B"),
                        (GradientMapPreset::BlackToColor, "B→Blue"),
                        (GradientMapPreset::Sepia, "Sepia"),
                        (GradientMapPreset::Infrared, "Infrared"),
                    ];
                    let mut changed = false;
                    for (preset, label) in presets {
                        let active = self.preset == preset;
                        let btn = if active {
                            egui::Button::new(label).fill(colors.accent_faint)
                        } else {
                            egui::Button::new(label)
                        };
                        if ui.add(btn).clicked() && !active {
                            self.apply_preset(preset);
                            changed = true;
                        }
                        if changed && self.live_preview {
                            result = DialogResult::Changed;
                        }
                    }
                });

                // Gradient preview strip
                ui.add_space(4.0);
                let lut = self.build_lut();
                let bar_rect = ui.allocate_space(Vec2::new(ui.available_width(), 16.0)).1;
                let painter = ui.painter();
                let steps = 64usize;
                let bar_w = bar_rect.width() / steps as f32;
                for i in 0..steps {
                    let idx = (i * 255 / steps.max(1)).min(255);
                    let [r, g, b, _] = lut[idx];
                    let rect = Rect::from_min_size(
                        Pos2::new(bar_rect.min.x + i as f32 * bar_w, bar_rect.min.y),
                        Vec2::new(bar_w + 0.5, 16.0),
                    );
                    painter.rect_filled(rect, 0.0, Color32::from_rgb(r, g, b));
                }
                painter.rect_stroke(
                    bar_rect,
                    CornerRadius::same(2),
                    Stroke::new(1.0, colors.separator),
                    egui::StrokeKind::Middle,
                );

                // Shadow / Highlight color pickers
                ui.add_space(4.0);
                section_label(ui, &colors, "COLORS");
                let mut changed_color = false;
                egui::Grid::new("gm_colors")
                    .num_columns(2)
                    .spacing([8.0, 6.0])
                    .show(ui, |ui| {
                        ui.label("Shadows");
                        let mut c = Color32::from_rgb(
                            self.shadow_color[0],
                            self.shadow_color[1],
                            self.shadow_color[2],
                        );
                        if ui.color_edit_button_srgba(&mut c).changed() {
                            self.shadow_color = [c.r(), c.g(), c.b()];
                            self.preset = GradientMapPreset::Custom;
                            changed_color = true;
                        }
                        ui.end_row();

                        ui.label("Highlights");
                        let mut c = Color32::from_rgb(
                            self.highlight_color[0],
                            self.highlight_color[1],
                            self.highlight_color[2],
                        );
                        if ui.color_edit_button_srgba(&mut c).changed() {
                            self.highlight_color = [c.r(), c.g(), c.b()];
                            self.preset = GradientMapPreset::Custom;
                            changed_color = true;
                        }
                        ui.end_row();
                    });
                if changed_color && self.live_preview {
                    result = DialogResult::Changed;
                }

                accent_separator(ui, &colors);
                let manual_preview = preview_controls(ui, &colors, &mut self.live_preview);
                if manual_preview {
                    result = DialogResult::Changed;
                }

                let (ok, cancel, reset) = dialog_footer_with_reset(ui, &colors);
                if reset {
                    self.apply_preset(GradientMapPreset::BlackToWhite);
                    result = DialogResult::Changed;
                }
                if ok {
                    result = DialogResult::Ok(self.build_lut());
                }
                if cancel {
                    result = DialogResult::Cancel;
                }
            });
        result
    }
}

// ============================================================================
// BLACK AND WHITE DIALOG
// ============================================================================

pub struct BlackAndWhiteDialog {
    pub r_weight: f32,
    pub g_weight: f32,
    pub b_weight: f32,
    pub original_pixels: Option<TiledImage>,
    pub original_flat: Option<image::RgbaImage>,
    pub layer_idx: usize,
    pub live_preview: bool,
}

impl BlackAndWhiteDialog {
    pub fn new(state: &CanvasState) -> Self {
        let idx = state.active_layer_index;
        Self {
            r_weight: 21.26,
            g_weight: 71.52,
            b_weight: 7.22,
            original_pixels: state.layers.get(idx).map(|l| l.pixels.clone()),
            original_flat: state.layers.get(idx).map(|l| l.pixels.to_rgba_image()),
            layer_idx: idx,
            live_preview: true,
        }
    }

    pub fn show(&mut self, ctx: &egui::Context) -> DialogResult<(f32, f32, f32)> {
        let mut result = DialogResult::Open;
        let colors = DialogColors::from_ctx(ctx);

        crate::ui::polish::window(ctx, "dialog_black_and_white")
            .title_bar(false)
            .collapsible(false)
            .resizable(false)
            .default_pos(egui::pos2(ctx.content_rect().center().x - 185.0, 60.0))
            .show(ctx, |ui| {
                ui.set_min_width(370.0);
                if paint_dialog_header(ui, &colors, "🎨", &t!("dialog.black_and_white")) {
                    result = DialogResult::Cancel;
                }
                ui.add_space(4.0);

                // Quick presets
                ui.horizontal(|ui| {
                    if ui.small_button("Luminosity").clicked() {
                        self.r_weight = 21.26;
                        self.g_weight = 71.52;
                        self.b_weight = 7.22;
                        if self.live_preview {
                            result = DialogResult::Changed;
                        }
                    }
                    if ui.small_button("Natural").clicked() {
                        self.r_weight = 40.0;
                        self.g_weight = 40.0;
                        self.b_weight = 20.0;
                        if self.live_preview {
                            result = DialogResult::Changed;
                        }
                    }
                    if ui.small_button("Infrared").clicked() {
                        self.r_weight = 60.0;
                        self.g_weight = 40.0;
                        self.b_weight = 0.0;
                        if self.live_preview {
                            result = DialogResult::Changed;
                        }
                    }
                    if ui.small_button("Flat").clicked() {
                        self.r_weight = 33.33;
                        self.g_weight = 33.33;
                        self.b_weight = 33.33;
                        if self.live_preview {
                            result = DialogResult::Changed;
                        }
                    }
                });

                section_label(ui, &colors, "CHANNEL WEIGHTS");
                let mut changed = false;

                egui::Grid::new("bw_params")
                    .num_columns(2)
                    .spacing([8.0, 6.0])
                    .show(ui, |ui| {
                        ui.label(egui::RichText::new("Reds").color(Color32::from_rgb(220, 80, 80)));
                        if dialog_slider(ui, &mut self.r_weight, 0.0..=200.0, 1.0, "", 1) {
                            changed = true;
                        }
                        ui.end_row();

                        ui.label(
                            egui::RichText::new("Greens").color(Color32::from_rgb(80, 180, 80)),
                        );
                        if dialog_slider(ui, &mut self.g_weight, 0.0..=200.0, 1.0, "", 1) {
                            changed = true;
                        }
                        ui.end_row();

                        ui.label(
                            egui::RichText::new("Blues").color(Color32::from_rgb(80, 140, 220)),
                        );
                        if dialog_slider(ui, &mut self.b_weight, 0.0..=200.0, 1.0, "", 1) {
                            changed = true;
                        }
                        ui.end_row();
                    });

                // Grayscale preview bar showing channel mix
                ui.add_space(4.0);
                let bar_rect = ui.allocate_space(Vec2::new(ui.available_width(), 8.0)).1;
                let painter = ui.painter();
                let steps = 32usize;
                let bar_w = bar_rect.width() / (steps * 3) as f32;
                // Show R, G, B contribution strips
                for i in 0..steps {
                    let t = i as f32 / (steps - 1).max(1) as f32;
                    let vr = (t * 255.0 * self.r_weight / 100.0).clamp(0.0, 255.0) as u8;
                    let vg = (t * 255.0 * self.g_weight / 100.0).clamp(0.0, 255.0) as u8;
                    let vb = (t * 255.0 * self.b_weight / 100.0).clamp(0.0, 255.0) as u8;
                    let colors_strips = [
                        Color32::from_rgb(vr, 0, 0),
                        Color32::from_rgb(0, vg, 0),
                        Color32::from_rgb(0, 0, vb),
                    ];
                    for (ci, col) in colors_strips.iter().enumerate() {
                        let rect = Rect::from_min_size(
                            Pos2::new(bar_rect.min.x + (i * 3 + ci) as f32 * bar_w, bar_rect.min.y),
                            Vec2::new(bar_w + 0.5, 8.0),
                        );
                        painter.rect_filled(rect, 0.0, *col);
                    }
                }
                painter.rect_stroke(
                    bar_rect,
                    CornerRadius::same(2),
                    Stroke::new(1.0, colors.separator),
                    egui::StrokeKind::Middle,
                );

                accent_separator(ui, &colors);
                let manual_preview = preview_controls(ui, &colors, &mut self.live_preview);
                if changed && self.live_preview {
                    result = DialogResult::Changed;
                }
                if manual_preview {
                    result = DialogResult::Changed;
                }

                let (ok, cancel, reset) = dialog_footer_with_reset(ui, &colors);
                if reset {
                    self.r_weight = 21.26;
                    self.g_weight = 71.52;
                    self.b_weight = 7.22;
                    result = DialogResult::Changed;
                }
                if ok {
                    result = DialogResult::Ok((self.r_weight, self.g_weight, self.b_weight));
                }
                if cancel {
                    result = DialogResult::Cancel;
                }
            });
        result
    }
}

// ============================================================================
// VIBRANCE DIALOG
// ============================================================================

pub struct VibranceDialog {
    pub amount: f32,
    pub original_pixels: Option<TiledImage>,
    pub original_flat: Option<image::RgbaImage>,
    pub layer_idx: usize,
    pub live_preview: bool,
}

impl VibranceDialog {
    pub fn new(state: &CanvasState) -> Self {
        let idx = state.active_layer_index;
        Self {
            amount: 0.0,
            original_pixels: state.layers.get(idx).map(|l| l.pixels.clone()),
            original_flat: state.layers.get(idx).map(|l| l.pixels.to_rgba_image()),
            layer_idx: idx,
            live_preview: true,
        }
    }

    pub fn show(&mut self, ctx: &egui::Context) -> DialogResult<f32> {
        let mut result = DialogResult::Open;
        let colors = DialogColors::from_ctx(ctx);

        crate::ui::polish::window(ctx, "dialog_vibrance")
            .title_bar(false)
            .collapsible(false)
            .resizable(false)
            .default_pos(egui::pos2(ctx.content_rect().center().x - 175.0, 60.0))
            .show(ctx, |ui| {
                ui.set_min_width(350.0);
                if paint_dialog_header(ui, &colors, "✨", &t!("dialog.vibrance")) {
                    result = DialogResult::Cancel;
                }
                ui.add_space(4.0);

                section_label(ui, &colors, "VIBRANCE");
                let mut changed = false;

                egui::Grid::new("vibrance_params")
                    .num_columns(2)
                    .spacing([8.0, 6.0])
                    .show(ui, |ui| {
                        ui.label("Vibrance");
                        if dialog_slider(ui, &mut self.amount, -100.0..=100.0, 1.0, "", 0) {
                            changed = true;
                        }
                        ui.end_row();
                    });

                // Saturation preview bar
                ui.add_space(4.0);
                let bar_rect = ui.allocate_space(Vec2::new(ui.available_width(), 8.0)).1;
                let painter = ui.painter();
                let steps = 32usize;
                let bar_w = bar_rect.width() / steps as f32;
                for i in 0..steps {
                    let hue = (i as f32 / steps as f32) * 360.0;
                    let sat = (0.5 + self.amount / 200.0).clamp(0.0, 1.0);
                    let (r, g, b) = hsv_to_rgb_simple(hue, sat, 0.9);
                    let rect = Rect::from_min_size(
                        Pos2::new(bar_rect.min.x + i as f32 * bar_w, bar_rect.min.y),
                        Vec2::new(bar_w + 0.5, 8.0),
                    );
                    painter.rect_filled(
                        rect,
                        0.0,
                        Color32::from_rgb((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8),
                    );
                }
                painter.rect_stroke(
                    bar_rect,
                    CornerRadius::same(2),
                    Stroke::new(1.0, colors.separator),
                    egui::StrokeKind::Middle,
                );

                accent_separator(ui, &colors);
                let manual_preview = preview_controls(ui, &colors, &mut self.live_preview);
                if changed && self.live_preview {
                    result = DialogResult::Changed;
                }
                if manual_preview {
                    result = DialogResult::Changed;
                }

                let (ok, cancel, reset) = dialog_footer_with_reset(ui, &colors);
                if reset {
                    self.amount = 0.0;
                    result = DialogResult::Changed;
                }
                if ok {
                    result = DialogResult::Ok(self.amount);
                }
                if cancel {
                    result = DialogResult::Cancel;
                }
            });
        result
    }
}

// ============================================================================
