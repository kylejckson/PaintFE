pub struct LayerTransformDialog {
    /// Z-axis rotation (normal 2D rotation), degrees.
    pub rotation_z: f32,
    /// X-axis perspective tilt, degrees.
    pub rotation_x: f32,
    /// Y-axis perspective tilt, degrees.
    pub rotation_y: f32,
    pub scale_percent: f32,
    pub offset_x: f32,
    pub offset_y: f32,
    /// Snapshot of original layer pixels for preview restore.
    pub original_pixels: Option<TiledImage>,
    /// Pre-flattened original pixels (avoids re-flattening every frame).
    pub original_flat: Option<image::RgbaImage>,
    /// Layer index being transformed.
    pub layer_idx: usize,
    /// Live preview toggle.
    pub live_preview: bool,
    /// Which axis the gizmo is currently dragging (None if idle).
    gizmo_drag_axis: Option<GizmoAxis>,
    /// Where the drag started (for relative calculation).
    gizmo_drag_start: Option<Pos2>,
    /// Values at drag start (for relative changes).
    gizmo_start_vals: (f32, f32, f32),
}

#[derive(Clone, Copy, PartialEq)]
enum GizmoAxis {
    Z, // rotation around Z (normal 2D rotation)
    X, // tilt around X axis
    Y, // tilt around Y axis
}

impl LayerTransformDialog {
    pub fn new(state: &CanvasState) -> Self {
        let idx = state.active_layer_index;
        let original = state.layers.get(idx).map(|l| l.pixels.clone());
        let flat = state.layers.get(idx).map(|l| l.pixels.to_rgba_image());
        Self {
            rotation_z: 0.0,
            rotation_x: 0.0,
            rotation_y: 0.0,
            scale_percent: 100.0,
            offset_x: 0.0,
            offset_y: 0.0,
            original_pixels: original,
            original_flat: flat,
            layer_idx: idx,
            live_preview: true,
            gizmo_drag_axis: None,
            gizmo_drag_start: None,
            gizmo_start_vals: (0.0, 0.0, 0.0),
        }
    }

    pub fn show(&mut self, ctx: &egui::Context) -> DialogResult<(f32, f32, f32, f32, (f32, f32))> {
        let mut result = DialogResult::Open;
        let mut changed = false;
        let colors = DialogColors::from_ctx(ctx);

        crate::ui::polish::window(ctx, "dialog_layer_transform")
            .title_bar(false)
            .collapsible(false)
            .resizable(false)
            .default_pos(egui::pos2(ctx.content_rect().center().x - 185.0, 50.0))
            .show(ctx, |ui| {
                ui.set_min_width(370.0);

                if paint_dialog_header(ui, &colors, "\u{1F504}", &t!("dialog.layer_transform")) {
                    result = DialogResult::Cancel;
                }
                ui.add_space(4.0);

                // -- Interactive Rotation Gizmo --
                section_label(ui, &colors, "ROTATION \u{2014} drag rings to rotate");
                ui.add_space(2.0);

                let gizmo_changed = self.draw_rotation_gizmo(ui, &colors);
                if gizmo_changed {
                    changed = true;
                }

                // -- Precise numeric controls --
                accent_separator(ui, &colors);
                section_label(ui, &colors, "PRECISE VALUES");

                egui::Grid::new("transform_values")
                    .num_columns(2)
                    .spacing([8.0, 4.0])
                    .show(ui, |ui| {
                        ui.label("Rotation");
                        if numeric_field_with_buttons(
                            ui,
                            &mut self.rotation_z,
                            0.5,
                            -180.0..=180.0,
                            "\u{00B0}",
                            1.0,
                        ) {
                            changed = true;
                        }
                        ui.end_row();

                        ui.label("Tilt X");
                        if numeric_field_with_buttons(
                            ui,
                            &mut self.rotation_x,
                            0.5,
                            -80.0..=80.0,
                            "\u{00B0}",
                            1.0,
                        ) {
                            changed = true;
                        }
                        ui.end_row();

                        ui.label("Tilt Y");
                        if numeric_field_with_buttons(
                            ui,
                            &mut self.rotation_y,
                            0.5,
                            -80.0..=80.0,
                            "\u{00B0}",
                            1.0,
                        ) {
                            changed = true;
                        }
                        ui.end_row();

                        ui.label("Scale");
                        if numeric_field_with_buttons(
                            ui,
                            &mut self.scale_percent,
                            0.5,
                            1.0..=500.0,
                            "%",
                            5.0,
                        ) {
                            changed = true;
                        }
                        ui.end_row();
                    });

                // -- Offset --
                accent_separator(ui, &colors);
                section_label(ui, &colors, "OFFSET");

                egui::Grid::new("transform_offset")
                    .num_columns(4)
                    .spacing([4.0, 4.0])
                    .show(ui, |ui| {
                        ui.label("X");
                        if ui
                            .add(crate::ui::numeric::Numeric::new(&mut self.offset_x).speed(1.0))
                            .changed()
                        {
                            changed = true;
                        }
                        ui.label("Y");
                        if ui
                            .add(crate::ui::numeric::Numeric::new(&mut self.offset_y).speed(1.0))
                            .changed()
                        {
                            changed = true;
                        }
                        ui.end_row();
                    });

                // -- Preview controls --
                accent_separator(ui, &colors);
                let manual_preview = preview_controls(ui, &colors, &mut self.live_preview);

                if changed && self.live_preview {
                    result = DialogResult::Changed;
                }
                if manual_preview {
                    result = DialogResult::Changed;
                }

                // -- Footer --
                let (ok, cancel, reset) = dialog_footer_with_reset(ui, &colors);
                if ok {
                    result = DialogResult::Ok((
                        self.rotation_z,
                        self.rotation_x,
                        self.rotation_y,
                        self.scale_percent / 100.0,
                        (self.offset_x, self.offset_y),
                    ));
                }
                if cancel {
                    result = DialogResult::Cancel;
                }
                if reset {
                    self.rotation_z = 0.0;
                    self.rotation_x = 0.0;
                    self.rotation_y = 0.0;
                    self.scale_percent = 100.0;
                    self.offset_x = 0.0;
                    self.offset_y = 0.0;
                    if self.live_preview {
                        result = DialogResult::Changed;
                    }
                }
            });

        result
    }

    /// Draw a 3-ring rotation gizmo that the user can drag.
    /// Returns true if any value changed.
    fn draw_rotation_gizmo(&mut self, ui: &mut egui::Ui, colors: &DialogColors) -> bool {
        let mut changed = false;
        let gizmo_size = 160.0;
        let half = gizmo_size / 2.0;

        // Center the gizmo
        ui.horizontal(|ui| {
            let avail = ui.available_width();
            let pad = ((avail - gizmo_size) / 2.0).max(0.0);
            ui.add_space(pad);

            let (gizmo_rect, response) =
                ui.allocate_exact_size(Vec2::splat(gizmo_size), Sense::click_and_drag());
            let center = gizmo_rect.center();
            let painter = ui.painter();

            // Background circle (subtle depth)
            let bg = if colors.is_dark {
                Color32::from_gray(22)
            } else {
                Color32::from_gray(248)
            };
            painter.circle_filled(center, half - 2.0, bg);
            painter.circle_stroke(center, half - 2.0, Stroke::new(1.0, colors.separator));

            // Ring radii
            let r_z = half - 8.0; // outermost - Z rotation
            let r_y = half - 28.0; // middle - Y tilt
            let r_x = half - 28.0; // inner - X tilt (drawn as ellipse)

            // Colors per axis
            let z_color = colors.accent;
            let x_color = Color32::from_rgb(230, 80, 80); // red-ish
            let y_color = Color32::from_rgb(80, 200, 80); // green-ish
            let active_axis = self.gizmo_drag_axis;

            // Draw Z-ring (outermost circle)
            let z_alpha = if active_axis == Some(GizmoAxis::Z) {
                255
            } else {
                180
            };
            let z_stroke = Stroke::new(
                if active_axis == Some(GizmoAxis::Z) {
                    3.0
                } else {
                    2.0
                },
                Color32::from_rgba_unmultiplied(z_color.r(), z_color.g(), z_color.b(), z_alpha),
            );
            painter.circle_stroke(center, r_z, z_stroke);

            // Draw Z rotation indicator (line from center to current angle)
            let z_rad = self.rotation_z.to_radians();
            let z_tip = Pos2::new(center.x + z_rad.cos() * r_z, center.y - z_rad.sin() * r_z);
            painter.line_segment(
                [center, z_tip],
                Stroke::new(
                    2.0,
                    Color32::from_rgba_unmultiplied(z_color.r(), z_color.g(), z_color.b(), z_alpha),
                ),
            );
            painter.circle_filled(z_tip, 4.0, z_color);

            // Draw Y-ring (vertical ellipse - tilt around Y makes it look like a vertical disc)
            let y_squash = (1.0 - (self.rotation_y.to_radians().sin().abs() * 0.6)).max(0.3);
            let y_alpha = if active_axis == Some(GizmoAxis::Y) {
                255
            } else {
                160
            };
            let y_stroke_w = if active_axis == Some(GizmoAxis::Y) {
                2.5
            } else {
                1.5
            };
            let y_col =
                Color32::from_rgba_unmultiplied(y_color.r(), y_color.g(), y_color.b(), y_alpha);
            // Approximate ellipse with line segments
            let n_segs: usize = 48;
            for i in 0..n_segs {
                let a0 = (i as f32 / n_segs as f32) * std::f32::consts::TAU;
                let a1 = ((i + 1) as f32 / n_segs as f32) * std::f32::consts::TAU;
                let p0 = Pos2::new(
                    center.x + a0.cos() * r_y * y_squash,
                    center.y + a0.sin() * r_y,
                );
                let p1 = Pos2::new(
                    center.x + a1.cos() * r_y * y_squash,
                    center.y + a1.sin() * r_y,
                );
                painter.line_segment([p0, p1], Stroke::new(y_stroke_w, y_col));
            }

            // Draw X-ring (horizontal ellipse - tilt around X)
            let x_squash = (1.0 - (self.rotation_x.to_radians().sin().abs() * 0.6)).max(0.3);
            let x_alpha = if active_axis == Some(GizmoAxis::X) {
                255
            } else {
                160
            };
            let x_stroke_w = if active_axis == Some(GizmoAxis::X) {
                2.5
            } else {
                1.5
            };
            let x_col =
                Color32::from_rgba_unmultiplied(x_color.r(), x_color.g(), x_color.b(), x_alpha);
            for i in 0..n_segs {
                let a0 = (i as f32 / n_segs as f32) * std::f32::consts::TAU;
                let a1 = ((i + 1) as f32 / n_segs as f32) * std::f32::consts::TAU;
                let p0 = Pos2::new(
                    center.x + a0.cos() * r_x,
                    center.y + a0.sin() * r_x * x_squash,
                );
                let p1 = Pos2::new(
                    center.x + a1.cos() * r_x,
                    center.y + a1.sin() * r_x * x_squash,
                );
                painter.line_segment([p0, p1], Stroke::new(x_stroke_w, x_col));
            }

            // Center crosshair
            let cross_len = 6.0;
            let cross_col = colors.text_muted;
            painter.line_segment(
                [
                    Pos2::new(center.x - cross_len, center.y),
                    Pos2::new(center.x + cross_len, center.y),
                ],
                Stroke::new(1.0, cross_col),
            );
            painter.line_segment(
                [
                    Pos2::new(center.x, center.y - cross_len),
                    Pos2::new(center.x, center.y + cross_len),
                ],
                Stroke::new(1.0, cross_col),
            );

            // Axis labels
            painter.text(
                Pos2::new(center.x + r_z + 2.0, center.y - 8.0),
                egui::Align2::LEFT_CENTER,
                "Z",
                egui::FontId::proportional(10.0),
                z_color,
            );
            painter.text(
                Pos2::new(center.x, center.y - r_y - 6.0),
                egui::Align2::CENTER_BOTTOM,
                "X",
                egui::FontId::proportional(10.0),
                x_color,
            );
            painter.text(
                Pos2::new(center.x - r_y * y_squash - 6.0, center.y),
                egui::Align2::RIGHT_CENTER,
                "Y",
                egui::FontId::proportional(10.0),
                y_color,
            );

            // -- Drag interaction --
            if response.drag_started()
                && let Some(pos) = response.interact_pointer_pos()
            {
                let d = (pos - center).length();
                // Determine which ring was clicked based on distance from center
                let axis = if (d - r_z).abs() < 15.0 {
                    Some(GizmoAxis::Z)
                } else if d < r_y + 10.0 {
                    // Inner region: determine X vs Y based on position
                    let dx = (pos.x - center.x).abs();
                    let dy = (pos.y - center.y).abs();
                    if dy > dx {
                        Some(GizmoAxis::X)
                    } else {
                        Some(GizmoAxis::Y)
                    }
                } else {
                    Some(GizmoAxis::Z) // default to Z for outer region
                };

                self.gizmo_drag_axis = axis;
                self.gizmo_drag_start = Some(pos);
                self.gizmo_start_vals = (self.rotation_z, self.rotation_x, self.rotation_y);
            }

            if response.dragged()
                && let (Some(axis), Some(start), Some(current)) = (
                    self.gizmo_drag_axis,
                    self.gizmo_drag_start,
                    response.interact_pointer_pos(),
                )
            {
                match axis {
                    GizmoAxis::Z => {
                        // Compute angle change from center
                        let a_start = (start.y - center.y).atan2(start.x - center.x);
                        let a_now = (current.y - center.y).atan2(current.x - center.x);
                        let delta = (a_start - a_now).to_degrees();
                        self.rotation_z = (self.gizmo_start_vals.0 + delta).clamp(-180.0, 180.0);
                        changed = true;
                    }
                    GizmoAxis::X => {
                        let delta = (start.y - current.y) * 0.5;
                        self.rotation_x = (self.gizmo_start_vals.1 + delta).clamp(-80.0, 80.0);
                        changed = true;
                    }
                    GizmoAxis::Y => {
                        let delta = (current.x - start.x) * 0.5;
                        self.rotation_y = (self.gizmo_start_vals.2 + delta).clamp(-80.0, 80.0);
                        changed = true;
                    }
                }
            }

            if response.drag_stopped() {
                self.gizmo_drag_axis = None;
                self.gizmo_drag_start = None;
            }
        });

        changed
    }
}

// ============================================================================
// ALIGN LAYER DIALOG
// ============================================================================

pub struct AlignLayerDialog {
    pub anchor_x: u32,
    pub anchor_y: u32,
    pub align_to_selection: bool,
    pub has_selection: bool,
    pub original_pixels: Option<TiledImage>,
    pub original_flat: Option<image::RgbaImage>,
    pub layer_idx: usize,
    pub live_preview: bool,
}

impl AlignLayerDialog {
    pub fn new(state: &CanvasState) -> Self {
        let idx = state.active_layer_index;
        let original = state.layers.get(idx).map(|l| l.pixels.clone());
        let flat = state.layers.get(idx).map(|l| l.pixels.to_rgba_image());
        let has_selection = state.selection_mask_bounds().is_some();
        Self {
            anchor_x: 1,
            anchor_y: 1,
            align_to_selection: has_selection,
            has_selection,
            original_pixels: original,
            original_flat: flat,
            layer_idx: idx,
            live_preview: true,
        }
    }

    pub fn show(&mut self, ctx: &egui::Context) -> DialogResult<(u32, u32, bool)> {
        let mut result = DialogResult::Open;
        let mut changed = false;
        let colors = DialogColors::from_ctx(ctx);

        crate::ui::polish::window(ctx, "dialog_align_layer")
            .title_bar(false)
            .collapsible(false)
            .resizable(false)
            .default_pos(egui::pos2(ctx.content_rect().center().x - 170.0, 70.0))
            .show(ctx, |ui| {
                ui.set_min_width(340.0);
                if paint_dialog_header(ui, &colors, "\u{2B1A}", &t!("dialog.align_layer")) {
                    result = DialogResult::Cancel;
                }
                ui.add_space(4.0);
                section_label(ui, &colors, "ALIGN TO CANVAS");
                ui.label("Moves the active raster layer using its non-transparent bounds.");
                ui.add_space(4.0);

                ui.horizontal(|ui| {
                    ui.label("Target:");
                    if ui
                        .selectable_label(!self.align_to_selection, "Canvas")
                        .clicked()
                    {
                        self.align_to_selection = false;
                        changed = true;
                    }
                    ui.add_enabled_ui(self.has_selection, |ui| {
                        if ui
                            .selectable_label(self.align_to_selection, "Selection")
                            .clicked()
                        {
                            self.align_to_selection = true;
                            changed = true;
                        }
                    });
                    if !self.has_selection {
                        ui.label(egui::RichText::new("(no selection)").weak());
                    }
                });
                ui.add_space(2.0);

                let labels = [
                    ["\u{2196}", "\u{2191}", "\u{2197}"],
                    ["\u{2190}", "\u{2299}", "\u{2192}"],
                    ["\u{2199}", "\u{2193}", "\u{2198}"],
                ];

                egui::Grid::new("align_anchor_grid")
                    .num_columns(3)
                    .spacing([6.0, 6.0])
                    .show(ui, |ui| {
                        for (y, row) in labels.iter().enumerate() {
                            for (x, label) in row.iter().enumerate() {
                                let selected =
                                    self.anchor_x == x as u32 && self.anchor_y == y as u32;
                                let mut btn = egui::Button::new(
                                    egui::RichText::new(*label).size(18.0).strong(),
                                )
                                .min_size(egui::vec2(42.0, 34.0));
                                if selected {
                                    btn = btn.fill(colors.accent_faint);
                                }

                                if ui.add(btn).clicked() {
                                    self.anchor_x = x as u32;
                                    self.anchor_y = y as u32;
                                    changed = true;
                                }
                            }
                            ui.end_row();
                        }
                    });

                accent_separator(ui, &colors);
                let manual_preview = preview_controls(ui, &colors, &mut self.live_preview);
                if (changed && self.live_preview) || manual_preview {
                    result = DialogResult::Changed;
                }

                let (ok, cancel, reset) = dialog_footer_with_reset(ui, &colors);
                if ok {
                    result =
                        DialogResult::Ok((self.anchor_x, self.anchor_y, self.align_to_selection));
                }
                if cancel {
                    result = DialogResult::Cancel;
                }
                if reset {
                    self.anchor_x = 1;
                    self.anchor_y = 1;
                    self.align_to_selection = false;
                    if self.live_preview {
                        result = DialogResult::Changed;
                    }
                }
            });

        result
    }
}

// ============================================================================
// BRIGHTNESS / CONTRAST DIALOG
// ============================================================================
