pub struct ColorRangeDialog {
    /// Center hue in degrees (0–360).
    pub hue_center: f32,
    /// Hue tolerance in degrees; larger = wider selection.
    pub hue_tolerance: f32,
    /// Minimum saturation (0–1) to be included.
    pub sat_min: f32,
    /// Edge softness / fuzziness (0–1).
    pub fuzziness: f32,
    /// How this selection merges with the existing mask.
    pub mode: crate::canvas::SelectionMode,
    /// Selection mask saved before dialog opened (for Cancel/live-preview).
    pub original_selection: Option<image::GrayImage>,
}

impl Default for ColorRangeDialog {
    fn default() -> Self {
        Self {
            hue_center: 0.0,
            hue_tolerance: 30.0,
            sat_min: 0.1,
            fuzziness: 0.3,
            mode: crate::canvas::SelectionMode::Replace,
            original_selection: None,
        }
    }
}

impl ColorRangeDialog {
    pub fn new(state: &crate::canvas::CanvasState) -> Self {
        Self {
            original_selection: state.selection_mask.clone(),
            ..Default::default()
        }
    }

    pub fn show(&mut self, ctx: &egui::Context) -> DialogResult<()> {
        let mut result = DialogResult::Open;
        let colors = DialogColors::from_ctx(ctx);

        crate::ui::polish::window(ctx, "dialog_color_range")
            .title_bar(false)
            .collapsible(false)
            .resizable(false)
            .default_pos(egui::pos2(ctx.content_rect().center().x - 170.0, 40.0))
            .show(ctx, |ui| {
                ui.set_min_width(340.0);
                if paint_dialog_header(ui, &colors, "🎨", "Select Color Range") {
                    result = DialogResult::Cancel;
                }
                ui.add_space(4.0);

                let mut changed = false;

                egui::Grid::new("color_range_grid")
                    .num_columns(2)
                    .spacing([8.0, 6.0])
                    .show(ui, |ui| {
                        ui.label("Hue Center");
                        if dialog_slider(ui, &mut self.hue_center, 0.0..=360.0, 1.0, "°", 1) {
                            changed = true;
                        }
                        ui.end_row();

                        ui.label("Tolerance");
                        if dialog_slider(ui, &mut self.hue_tolerance, 1.0..=180.0, 1.0, "°", 1) {
                            changed = true;
                        }
                        ui.end_row();

                        ui.label("Min Saturation");
                        if dialog_slider(ui, &mut self.sat_min, 0.0..=1.0, 0.01, "", 2) {
                            changed = true;
                        }
                        ui.end_row();

                        ui.label("Fuzziness");
                        if dialog_slider(ui, &mut self.fuzziness, 0.0..=1.0, 0.01, "", 2) {
                            changed = true;
                        }
                        ui.end_row();

                        ui.label("Mode");
                        ui.horizontal(|ui| {
                            for mode in crate::canvas::SelectionMode::all() {
                                let selected = self.mode == *mode;
                                if ui.selectable_label(selected, mode.label()).clicked() {
                                    self.mode = *mode;
                                    changed = true;
                                }
                            }
                        });
                        ui.end_row();
                    });

                // Hue preview bar
                ui.add_space(4.0);
                let bar_rect = ui.allocate_space(Vec2::new(ui.available_width(), 12.0)).1;
                let painter = ui.painter();
                let steps = 360usize;
                let bar_w = bar_rect.width() / steps as f32;
                for deg in 0..steps {
                    let (r, g, b) =
                        crate::ops::adjustments::hsl_to_rgb(deg as f32 / 360.0, 1.0, 0.5);
                    let rect = Rect::from_min_size(
                        Pos2::new(bar_rect.min.x + deg as f32 * bar_w, bar_rect.min.y),
                        Vec2::new(bar_w + 0.5, 12.0),
                    );
                    painter.rect_filled(
                        rect,
                        0.0,
                        Color32::from_rgb((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8),
                    );
                }
                let marker_x = bar_rect.min.x + self.hue_center * (bar_rect.width() / 360.0);
                painter.vline(
                    marker_x,
                    bar_rect.y_range(),
                    Stroke::new(2.0, Color32::WHITE),
                );
                let bw = bar_rect.width() / 360.0;
                let lo_x =
                    bar_rect.min.x + (self.hue_center - self.hue_tolerance).rem_euclid(360.0) * bw;
                let hi_x =
                    bar_rect.min.x + (self.hue_center + self.hue_tolerance).rem_euclid(360.0) * bw;
                let fade = Color32::from_rgba_premultiplied(255, 255, 255, 140);
                painter.vline(
                    lo_x.clamp(bar_rect.min.x, bar_rect.max.x),
                    bar_rect.y_range(),
                    Stroke::new(1.0, fade),
                );
                painter.vline(
                    hi_x.clamp(bar_rect.min.x, bar_rect.max.x),
                    bar_rect.y_range(),
                    Stroke::new(1.0, fade),
                );
                painter.rect_stroke(
                    bar_rect,
                    CornerRadius::same(2),
                    Stroke::new(1.0, colors.separator),
                    egui::StrokeKind::Middle,
                );

                if changed {
                    result = DialogResult::Changed;
                }

                accent_separator(ui, &colors);
                let (ok, cancel, reset) = dialog_footer_with_reset(ui, &colors);
                if reset {
                    self.hue_tolerance = 30.0;
                    self.sat_min = 0.1;
                    self.fuzziness = 0.3;
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
