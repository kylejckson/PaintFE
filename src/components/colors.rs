use super::color_widgets as cw;
use crate::assets::Assets;
use eframe::egui;
use egui::{Color32, Pos2, Stroke, Vec2};

const TAU: f32 = std::f32::consts::TAU;

// ============================================================================
// Interaction zone for the combined ring + triangle widget
// ============================================================================

#[derive(Clone, Copy, PartialEq, Default)]
enum DragZone {
    #[default]
    None,
    HueRing,
    SvTriangle,
}

// ============================================================================
// ColorsPanel — Hue-Ring / SV-Triangle Color Picker
// ============================================================================

pub struct ColorsPanel {
    pub primary_color: Color32,
    pub secondary_color: Color32,
    editing_primary: bool,
    expanded: bool,
    primary_hsv: [f32; 3],
    secondary_hsv: [f32; 3],
    drag_zone: DragZone,
    hex_buffer: String,
    hex_editing: bool,
    hex_invalid: bool,
    sections: [bool; 3],
}

impl Default for ColorsPanel {
    fn default() -> Self {
        Self {
            primary_color: Color32::BLACK,
            secondary_color: Color32::WHITE,
            editing_primary: true,
            expanded: false,
            primary_hsv: [0.0, 0.0, 0.0],   // black
            secondary_hsv: [0.0, 0.0, 1.0], // white
            drag_zone: DragZone::None,
            hex_buffer: String::new(),
            hex_editing: false,
            hex_invalid: false,
            sections: [true, false, false],
        }
    }
}

// ============================================================================
// Public API  (contract unchanged from old implementation)
// ============================================================================

impl ColorsPanel {
    pub fn show(&mut self, ui: &mut egui::Ui, assets: &Assets) {
        self.show_content(ui, assets);
    }

    pub fn show_compact(&mut self, ui: &mut egui::Ui, assets: &Assets) {
        self.show_content(ui, assets);
    }

    pub fn editing_secondary(&self) -> bool {
        !self.editing_primary
    }

    pub fn get_primary_color(&self) -> Color32 {
        self.primary_color
    }

    pub fn get_secondary_color(&self) -> Color32 {
        self.secondary_color
    }

    pub fn set_primary_color(&mut self, color: Color32) {
        self.primary_color = color;
        self.primary_hsv = preserve_hue(color, self.primary_hsv);
        self.editing_primary = true;
    }

    pub fn set_secondary_color(&mut self, color: Color32) {
        self.secondary_color = color;
        self.secondary_hsv = preserve_hue(color, self.secondary_hsv);
        self.editing_primary = false;
    }

    pub fn swap_colors(&mut self) {
        std::mem::swap(&mut self.primary_color, &mut self.secondary_color);
        std::mem::swap(&mut self.primary_hsv, &mut self.secondary_hsv);
    }

    pub fn section_mask(&self) -> u8 {
        self.sections
            .iter()
            .enumerate()
            .fold(0, |mask, (i, open)| mask | (u8::from(*open) << i))
    }
    pub fn load_section_mask(&mut self, mask: u8) {
        self.sections = [mask & 1 != 0, mask & 2 != 0, mask & 4 != 0];
    }
    pub fn is_expanded(&self) -> bool {
        self.expanded
    }

    pub fn is_hex_editing(&self) -> bool {
        self.hex_editing
    }

    pub fn set_expanded(&mut self, expanded: bool) {
        self.expanded = expanded;
    }

    /// (r, g, b, a) as 0.0–1.0 f32, un-multiplied.  RGB reconstructed from
    /// stored HSV for maximum precision.
    pub fn get_primary_color_f32(&self) -> [f32; 4] {
        let [h, s, v] = self.primary_hsv;
        let c = hsv_to_color(h, s, v, 255);
        let a = self.primary_color.a() as f32 / 255.0;
        [
            c.r() as f32 / 255.0,
            c.g() as f32 / 255.0,
            c.b() as f32 / 255.0,
            a,
        ]
    }

    pub fn get_secondary_color_f32(&self) -> [f32; 4] {
        let [h, s, v] = self.secondary_hsv;
        let c = hsv_to_color(h, s, v, 255);
        let a = self.secondary_color.a() as f32 / 255.0;
        [
            c.r() as f32 / 255.0,
            c.g() as f32 / 255.0,
            c.b() as f32 / 255.0,
            a,
        ]
    }
}

// ============================================================================
// Internal — layout, sync, widgets
// ============================================================================

impl ColorsPanel {
    // -- HSV sync (handles externally-set colours) -------------------------
    fn sync_hsv_from_colors(&mut self) {
        let exp = hsv_to_color(
            self.primary_hsv[0],
            self.primary_hsv[1],
            self.primary_hsv[2],
            self.primary_color.a(),
        );
        if (exp.r() as i16 - self.primary_color.r() as i16).abs() > 2
            || (exp.g() as i16 - self.primary_color.g() as i16).abs() > 2
            || (exp.b() as i16 - self.primary_color.b() as i16).abs() > 2
        {
            self.primary_hsv = preserve_hue(self.primary_color, self.primary_hsv);
        }
        let exp = hsv_to_color(
            self.secondary_hsv[0],
            self.secondary_hsv[1],
            self.secondary_hsv[2],
            self.secondary_color.a(),
        );
        if (exp.r() as i16 - self.secondary_color.r() as i16).abs() > 2
            || (exp.g() as i16 - self.secondary_color.g() as i16).abs() > 2
            || (exp.b() as i16 - self.secondary_color.b() as i16).abs() > 2
        {
            self.secondary_hsv = preserve_hue(self.secondary_color, self.secondary_hsv);
        }
    }

    // -- Top-level layout ---------------------------------------
    fn show_content(&mut self, ui: &mut egui::Ui, _assets: &Assets) {
        self.sync_hsv_from_colors();
        ui.spacing_mut().item_spacing =
            egui::vec2(5.0, 5.0) * crate::ui::polish::settings(ui.ctx()).spacing_scale;
        ui.style_mut().animation_time =
            crate::ui::polish::duration(ui.ctx(), crate::ui::polish::MotionKind::Expansion);
        let mut mode = usize::from(self.expanded);
        cw::tabs(ui, "color_modes", &["Color", "Advanced"], &mut mode);
        self.expanded = mode == 1;
        ui.add_space(2.0);
        ui.horizontal(|ui| {
            ui.add_space((ui.available_width() - 168.0).max(0.0) * 0.5);
            self.draw_hue_ring_and_sv_triangle(ui);
        });
        let [h, s, v] = self.active_hsv();
        let mut alpha = self.active_color().a() as f32;
        if cw::slider(
            ui,
            "opacity",
            "",
            &mut alpha,
            255.0,
            |t| hsv_to_color(h, s, v, (t * 255.0).round() as u8),
            true,
        ) {
            self.apply_hsv([h, s, v], alpha.round() as u8);
        }
        ui.horizontal(|ui| {
            for (primary, title) in [(true, "Primary"), (false, "Secondary")] {
                ui.vertical(|ui| {
                    ui.label(
                        egui::RichText::new(title)
                            .size(9.0)
                            .color(ui.visuals().weak_text_color()),
                    );
                    let color = if primary {
                        self.primary_color
                    } else {
                        self.secondary_color
                    };
                    if cw::swatch(ui, title, color, 29.0, self.editing_primary == primary).clicked()
                    {
                        self.editing_primary = primary;
                        self.hex_editing = false;
                    }
                });
                if primary {
                    ui.vertical(|ui| {
                        ui.add_space(17.0);
                        if cw::icon_button(ui, "Swap Primary and Secondary", false).clicked() {
                            self.swap_colors();
                        }
                    });
                }
            }
            ui.vertical(|ui| {
                ui.add_space(17.0);
                self.draw_hex_row(ui);
            });
        });
        let id = ui.id().with("advanced_rollout");
        let mut rollout =
            egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, false);
        rollout.set_open(self.expanded);
        rollout.show_body_unindented(ui, |ui| {
            ui.add_space(5.0);
            self.section(ui, "HSV", true, 0);
            self.section(ui, "RGB", false, 1);
            self.section(ui, "HSL", false, 2);
        });
    }

    fn active_hsv(&self) -> [f32; 3] {
        if self.editing_primary {
            self.primary_hsv
        } else {
            self.secondary_hsv
        }
    }
    fn active_color(&self) -> Color32 {
        if self.editing_primary {
            self.primary_color
        } else {
            self.secondary_color
        }
    }
    fn apply_hsv(&mut self, hsv: [f32; 3], alpha: u8) {
        let color = hsv_to_color(hsv[0], hsv[1], hsv[2], alpha);
        if self.editing_primary {
            self.primary_hsv = hsv;
            self.primary_color = color;
        } else {
            self.secondary_hsv = hsv;
            self.secondary_color = color;
        }
    }
    fn draw_hex_row(&mut self, ui: &mut egui::Ui) {
        let [h, s, v] = self.active_hsv();
        let current = cw::hex(hsv_to_color(h, s, v, 255));
        if !self.hex_editing {
            self.hex_buffer.clone_from(&current);
        }
        egui::Frame::new()
            .fill(cw::field_fill(ui))
            .corner_radius(cw::widget_radius(ui))
            .inner_margin(egui::Margin {
                left: 8,
                right: 2,
                top: 2,
                bottom: 2,
            })
            .show(ui, |ui| {
                ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                    ui.set_min_height(20.0);
                    // TextEdit consumes Escape itself; capture navigation intent first.
                    let escape_pressed = ui.input(|i| i.key_pressed(egui::Key::Escape));
                    let enter_pressed = ui.input(|i| i.key_pressed(egui::Key::Enter));
                    let response = ui.add(
                        egui::TextEdit::singleline(&mut self.hex_buffer)
                            .id_source("color_hex")
                            .desired_width(58.0)
                            .font(egui::FontId::monospace(10.0))
                            .frame(egui::Frame::NONE),
                    );
                    let was_editing = self.hex_editing;
                    self.hex_editing = response.has_focus();
                    if (was_editing || response.has_focus() || response.lost_focus())
                        && escape_pressed
                    {
                        self.hex_buffer.clone_from(&current);
                        self.hex_invalid = false;
                        response.surrender_focus();
                        self.hex_editing = false;
                    } else if response.lost_focus() || (response.has_focus() && enter_pressed) {
                        if let Some([r, g, b]) = cw::parse_hex(&self.hex_buffer) {
                            let hsv = preserve_hue(cw::rgba(r, g, b, 255), self.active_hsv());
                            self.apply_hsv(hsv, self.active_color().a());
                            self.hex_invalid = false;
                            response.surrender_focus();
                            self.hex_editing = false;
                        } else {
                            self.hex_invalid = true;
                            response.request_focus();
                            self.hex_editing = true;
                        }
                    }
                    if cw::icon_button(ui, "Copy hex color", true).clicked() {
                        ui.ctx().copy_text(current);
                    }
                });
            });
        if self.hex_invalid {
            ui.label(
                egui::RichText::new("Use #RRGGBB")
                    .size(9.0)
                    .color(ui.visuals().error_fg_color),
            );
        }
    }
    fn section(&mut self, ui: &mut egui::Ui, label: &str, default_open: bool, space: usize) {
        let id = ui.id().with(label);
        let mut state = egui::collapsing_header::CollapsingState::load_with_default_open(
            ui.ctx(),
            id,
            default_open,
        );
        state.set_open(self.sections[space]);
        let open = state.is_open();
        let (rect, response) =
            ui.allocate_exact_size(egui::vec2(ui.available_width(), 25.0), egui::Sense::click());
        response.widget_info(|| {
            egui::WidgetInfo::labeled(egui::WidgetType::Button, ui.is_enabled(), label)
        });
        ui.painter()
            .rect_filled(rect, cw::widget_radius(ui), cw::field_fill(ui));
        ui.painter().text(
            rect.left_center() + egui::vec2(21.0, 0.0),
            egui::Align2::LEFT_CENTER,
            label,
            egui::FontId::proportional(10.0),
            ui.visuals().text_color(),
        );
        let center = rect.left_center() + egui::vec2(9.0, 0.0);
        crate::ui::polish::control(ui, &response, false, cw::widget_radius(ui));
        let angle = crate::ui::polish::animate(
            ui.ctx(),
            id.with("chevron"),
            if open {
                std::f32::consts::FRAC_PI_2
            } else {
                0.0
            },
            crate::ui::polish::MotionKind::Expansion,
        );
        let rotate = |p: egui::Vec2| {
            center
                + egui::vec2(
                    p.x * angle.cos() - p.y * angle.sin(),
                    p.x * angle.sin() + p.y * angle.cos(),
                )
        };
        let points = vec![
            rotate(egui::vec2(-1.5, -3.0)),
            rotate(egui::vec2(1.5, 0.0)),
            rotate(egui::vec2(-1.5, 3.0)),
        ];
        ui.painter().add(egui::Shape::line(
            points,
            Stroke::new(1.2, ui.visuals().weak_text_color()),
        ));
        if response.clicked() {
            state.toggle(ui);
        }
        self.sections[space] = state.is_open();
        state.show_body_unindented(ui, |ui| self.draw_channels(ui, space));
    }
    fn draw_channels(&mut self, ui: &mut egui::Ui, space: usize) {
        let hsv = self.active_hsv();
        let rgb = cw::straight(hsv_to_color(hsv[0], hsv[1], hsv[2], 255));
        let mut values = match space {
            0 => [hsv[0] * 360.0, hsv[1] * 100.0, hsv[2] * 100.0],
            1 => [rgb[0] as f32, rgb[1] as f32, rgb[2] as f32],
            _ => {
                let [h, s, l] = hsv_to_hsl(hsv);
                [h * 360.0, s * 100.0, l * 100.0]
            }
        };
        let labels = match space {
            0 => ["H", "S", "V"],
            1 => ["R", "G", "B"],
            _ => ["H", "S", "L"],
        };
        let mut changed = false;
        for i in 0..3 {
            let max = if space == 1 {
                255.0
            } else if i == 0 {
                360.0
            } else {
                100.0
            };
            let base = values;
            changed |= cw::slider(
                ui,
                labels[i],
                labels[i],
                &mut values[i],
                max,
                |t| {
                    let mut c = base;
                    c[i] = t * max;
                    match space {
                        0 => {
                            if i == 0 {
                                hsv_to_color(t, 1.0, 1.0, 255)
                            } else if i == 1 {
                                hsv_to_color(c[0] / 360.0, t, 1.0, 255)
                            } else {
                                hsv_to_color(c[0] / 360.0, c[1] / 100.0, t, 255)
                            }
                        }
                        1 => cw::rgba(
                            c[0].round() as u8,
                            c[1].round() as u8,
                            c[2].round() as u8,
                            255,
                        ),
                        _ => {
                            let h = hsl_to_hsv([c[0] / 360.0, c[1] / 100.0, c[2] / 100.0]);
                            hsv_to_color(h[0], h[1], h[2], 255)
                        }
                    }
                },
                false,
            );
        }
        if changed {
            let mut next = match space {
                0 => [values[0] / 360.0, values[1] / 100.0, values[2] / 100.0],
                1 => preserve_hue(
                    cw::rgba(
                        values[0].round() as u8,
                        values[1].round() as u8,
                        values[2].round() as u8,
                        255,
                    ),
                    hsv,
                ),
                _ => hsl_to_hsv([values[0] / 360.0, values[1] / 100.0, values[2] / 100.0]),
            };
            next[0] = next[0].rem_euclid(1.0);
            self.apply_hsv(next, self.active_color().a());
        }
    }

    fn draw_hue_ring_and_sv_triangle(&mut self, ui: &mut egui::Ui) {
        // -- geometry --
        let outer_r: f32 = 78.0;
        let ring_w: f32 = 16.0;
        let inner_r = outer_r - ring_w;
        let tri_r = inner_r - 3.0; // small gap between ring & triangle

        let widget_size = Vec2::splat(outer_r * 2.0 + 12.0); // room for indicators
        let (rect, response) = ui.allocate_exact_size(widget_size, egui::Sense::click_and_drag());
        let center = rect.center();

        // -- copy out current values (avoids borrow issues) --
        let mut h = if self.editing_primary {
            self.primary_hsv[0]
        } else {
            self.secondary_hsv[0]
        };
        let mut s = if self.editing_primary {
            self.primary_hsv[1]
        } else {
            self.secondary_hsv[1]
        };
        let mut v = if self.editing_primary {
            self.primary_hsv[2]
        } else {
            self.secondary_hsv[2]
        };
        let alpha = if self.editing_primary {
            self.primary_color.a()
        } else {
            self.secondary_color.a()
        };
        let mut drag_zone = self.drag_zone;
        let mut changed = false;

        // -- triangle vertex positions (rotate with hue) --
        let hue_angle = h * TAU;
        let vert_a = Pos2::new(
            center.x + hue_angle.cos() * tri_r,
            center.y + hue_angle.sin() * tri_r,
        );
        let angle_b = hue_angle + TAU / 3.0;
        let vert_b = Pos2::new(
            center.x + angle_b.cos() * tri_r,
            center.y + angle_b.sin() * tri_r,
        );
        let angle_c = hue_angle + 2.0 * TAU / 3.0;
        let vert_c = Pos2::new(
            center.x + angle_c.cos() * tri_r,
            center.y + angle_c.sin() * tri_r,
        );

        // -- rendering --
        if ui.is_rect_visible(rect) {
            let p = ui.painter();

            // 1) Hue ring  (annular mesh, 96 segments + AA fringe)
            let segs: u32 = 96;
            let aa_w: f32 = 1.2; // anti-alias fringe width
            let mut ring = egui::Mesh::default();
            for i in 0..segs {
                let a0 = (i as f32 / segs as f32) * TAU;
                let a1 = ((i + 1) as f32 / segs as f32) * TAU;
                let c0 = hsv_to_color(i as f32 / segs as f32, 1.0, 1.0, 255);
                let c1 = hsv_to_color((i + 1) as f32 / segs as f32, 1.0, 1.0, 255);
                let c0t = cw::rgba(c0.r(), c0.g(), c0.b(), 0);
                let c1t = cw::rgba(c1.r(), c1.g(), c1.b(), 0);
                let b = ring.vertices.len() as u32;
                // outer AA fringe (transparent → opaque)
                ring.colored_vertex(
                    Pos2::new(
                        center.x + a0.cos() * (outer_r + aa_w),
                        center.y + a0.sin() * (outer_r + aa_w),
                    ),
                    c0t,
                );
                ring.colored_vertex(
                    Pos2::new(
                        center.x + a1.cos() * (outer_r + aa_w),
                        center.y + a1.sin() * (outer_r + aa_w),
                    ),
                    c1t,
                );
                // outer solid edge
                ring.colored_vertex(
                    Pos2::new(center.x + a0.cos() * outer_r, center.y + a0.sin() * outer_r),
                    c0,
                );
                ring.colored_vertex(
                    Pos2::new(center.x + a1.cos() * outer_r, center.y + a1.sin() * outer_r),
                    c1,
                );
                // inner solid edge
                ring.colored_vertex(
                    Pos2::new(center.x + a0.cos() * inner_r, center.y + a0.sin() * inner_r),
                    c0,
                );
                ring.colored_vertex(
                    Pos2::new(center.x + a1.cos() * inner_r, center.y + a1.sin() * inner_r),
                    c1,
                );
                // inner AA fringe (opaque → transparent)
                ring.colored_vertex(
                    Pos2::new(
                        center.x + a0.cos() * (inner_r - aa_w),
                        center.y + a0.sin() * (inner_r - aa_w),
                    ),
                    c0t,
                );
                ring.colored_vertex(
                    Pos2::new(
                        center.x + a1.cos() * (inner_r - aa_w),
                        center.y + a1.sin() * (inner_r - aa_w),
                    ),
                    c1t,
                );
                // outer fringe quad
                ring.add_triangle(b, b + 1, b + 3);
                ring.add_triangle(b, b + 3, b + 2);
                // solid body quad
                ring.add_triangle(b + 2, b + 3, b + 5);
                ring.add_triangle(b + 2, b + 5, b + 4);
                // inner fringe quad
                ring.add_triangle(b + 4, b + 5, b + 7);
                ring.add_triangle(b + 4, b + 7, b + 6);
            }
            p.add(egui::Shape::mesh(ring));

            // 2) SV triangle  (3-vertex core + AA fringe edges)
            let pure_col = hsv_to_color(h, 1.0, 1.0, 255);
            let mut tri = egui::Mesh::default();
            tri.colored_vertex(vert_a, pure_col); // pure hue
            tri.colored_vertex(vert_b, Color32::WHITE); // white
            tri.colored_vertex(vert_c, Color32::BLACK); // black
            tri.add_triangle(0, 1, 2);
            p.add(egui::Shape::mesh(tri));

            // Triangle edge AA fringe — extrude each edge outward by aa_w
            let tri_verts = [
                (vert_a, pure_col),
                (vert_b, Color32::WHITE),
                (vert_c, Color32::BLACK),
            ];
            for i in 0..3 {
                let (p0, c0) = tri_verts[i];
                let (p1, c1) = tri_verts[(i + 1) % 3];
                let edge = Vec2::new(p1.x - p0.x, p1.y - p0.y);
                let n = Vec2::new(-edge.y, edge.x).normalized() * aa_w;
                // determine outward direction (away from opposite vertex)
                let (p2, _) = tri_verts[(i + 2) % 3];
                let mid = Pos2::new((p0.x + p1.x) / 2.0, (p0.y + p1.y) / 2.0);
                let to_opp = Vec2::new(p2.x - mid.x, p2.y - mid.y);
                let n = if n.x * to_opp.x + n.y * to_opp.y > 0.0 {
                    -n
                } else {
                    n
                };
                let c0t = cw::rgba(c0.r(), c0.g(), c0.b(), 0);
                let c1t = cw::rgba(c1.r(), c1.g(), c1.b(), 0);
                let mut fringe = egui::Mesh::default();
                fringe.colored_vertex(p0, c0); // 0
                fringe.colored_vertex(p1, c1); // 1
                fringe.colored_vertex(Pos2::new(p1.x + n.x, p1.y + n.y), c1t); // 2
                fringe.colored_vertex(Pos2::new(p0.x + n.x, p0.y + n.y), c0t); // 3
                fringe.add_triangle(0, 1, 2);
                fringe.add_triangle(0, 2, 3);
                p.add(egui::Shape::mesh(fringe));
            }

            // 3) SV indicator dot
            let w_a = s * v;
            let w_b = v * (1.0 - s);
            let w_c = 1.0 - v;
            let sv_pos = Pos2::new(
                w_a * vert_a.x + w_b * vert_b.x + w_c * vert_c.x,
                w_a * vert_a.y + w_b * vert_b.y + w_c * vert_c.y,
            );
            // outer halo → colour fill → white ring
            p.circle_stroke(sv_pos, 7.5, Stroke::new(1.0, Color32::from_black_alpha(45)));
            p.circle_filled(sv_pos, 6.0, hsv_to_color(h, s, v, 255));
            p.circle_stroke(sv_pos, 6.0, Stroke::new(2.0, Color32::WHITE));

            // 4) Hue ring indicator — white radial line across ring width
            let line_inner = Pos2::new(
                center.x + hue_angle.cos() * (inner_r - 1.0),
                center.y + hue_angle.sin() * (inner_r - 1.0),
            );
            let line_outer = Pos2::new(
                center.x + hue_angle.cos() * (outer_r + 1.0),
                center.y + hue_angle.sin() * (outer_r + 1.0),
            );
            // dark outline for contrast
            p.line_segment(
                [line_inner, line_outer],
                Stroke::new(4.0, Color32::from_black_alpha(80)),
            );
            // white core line
            p.line_segment([line_inner, line_outer], Stroke::new(2.0, Color32::WHITE));
        }

        // Capture the press zone once and continue through an outside release.
        let (start, position, active) = cw::drag_motion(ui, rect, response.id);
        if let Some(start) = start {
            let dist = (start - center).length();
            drag_zone = if dist >= inner_r - 6.0 && dist <= outer_r + 6.0 {
                DragZone::HueRing
            } else if point_in_triangle(start, vert_a, vert_b, vert_c) || dist < inner_r {
                DragZone::SvTriangle
            } else {
                DragZone::None
            };
        }
        if let Some(mp) = position {
            match drag_zone {
                DragZone::HueRing => {
                    let delta = mp - center;
                    h = (delta.y.atan2(delta.x) / TAU).rem_euclid(1.0);
                    changed = true;
                }
                DragZone::SvTriangle => {
                    let (wa, wb, wc) = barycentric(mp, vert_a, vert_b, vert_c);
                    let wa = wa.max(0.0);
                    let wb = wb.max(0.0);
                    let wc = wc.max(0.0);
                    let sum = wa + wb + wc;
                    if sum > 0.001 {
                        let wa = wa / sum;
                        let wb = wb / sum;
                        v = (wa + wb).clamp(0.0, 1.0);
                        s = if v > 0.001 {
                            (wa / v).clamp(0.0, 1.0)
                        } else {
                            s
                        };
                        changed = true;
                    }
                }
                DragZone::None => {}
            }
        }
        if !active {
            drag_zone = DragZone::None;
        }

        // -- write back --
        self.drag_zone = drag_zone;
        if changed {
            if self.editing_primary {
                self.primary_hsv = [h, s, v];
                self.primary_color = hsv_to_color(h, s, v, alpha);
            } else {
                self.secondary_hsv = [h, s, v];
                self.secondary_color = hsv_to_color(h, s, v, alpha);
            }
        }
    }
}

fn barycentric(p: Pos2, a: Pos2, b: Pos2, c: Pos2) -> (f32, f32, f32) {
    let v0 = Vec2::new(c.x - a.x, c.y - a.y);
    let v1 = Vec2::new(b.x - a.x, b.y - a.y);
    let v2 = Vec2::new(p.x - a.x, p.y - a.y);
    let d00 = v0.x * v0.x + v0.y * v0.y;
    let d01 = v0.x * v1.x + v0.y * v1.y;
    let d02 = v0.x * v2.x + v0.y * v2.y;
    let d11 = v1.x * v1.x + v1.y * v1.y;
    let d12 = v1.x * v2.x + v1.y * v2.y;
    let denom = d00 * d11 - d01 * d01;
    if denom.abs() < 1e-10 {
        return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0);
    }
    let inv = 1.0 / denom;
    let u = (d11 * d02 - d01 * d12) * inv; // weight for c
    let v = (d00 * d12 - d01 * d02) * inv; // weight for b
    let w = 1.0 - u - v; // weight for a
    (w, v, u)
}

/// True when `p` is inside triangle (a, b, c).
fn point_in_triangle(p: Pos2, a: Pos2, b: Pos2, c: Pos2) -> bool {
    let (wa, wb, wc) = barycentric(p, a, b, c);
    wa >= 0.0 && wb >= 0.0 && wc >= 0.0
}

// -- Colour-space conversions -----------------------------------

pub(crate) fn color_to_hsv(color: Color32) -> [f32; 3] {
    let [r, g, b, _] = cw::straight(color);
    let r = r as f32 / 255.0;
    let g = g as f32 / 255.0;
    let b = b as f32 / 255.0;
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let d = max - min;

    let h = if d == 0.0 {
        0.0
    } else if max == r {
        ((g - b) / d % 6.0) / 6.0
    } else if max == g {
        (((b - r) / d) + 2.0) / 6.0
    } else {
        (((r - g) / d) + 4.0) / 6.0
    };
    let h = if h < 0.0 { h + 1.0 } else { h };
    let s = if max == 0.0 { 0.0 } else { d / max };
    [h, s, max]
}

pub(crate) fn hsv_to_color(h: f32, s: f32, v: f32, a: u8) -> Color32 {
    let h6 = h.rem_euclid(1.0) * 6.0;
    let c = v * s;
    let x = c * (1.0 - ((h6 % 2.0) - 1.0).abs());
    let m = v - c;
    let (r, g, b) = match h6 as i32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    cw::rgba(
        ((r + m) * 255.0).round() as u8,
        ((g + m) * 255.0).round() as u8,
        ((b + m) * 255.0).round() as u8,
        a,
    )
}

fn preserve_hue(color: Color32, previous: [f32; 3]) -> [f32; 3] {
    if color.a() == 0 {
        return previous;
    }
    let mut hsv = color_to_hsv(color);
    if hsv[1] <= 0.0001 {
        hsv[0] = previous[0];
    }
    hsv
}
fn hsv_to_hsl([h, s, v]: [f32; 3]) -> [f32; 3] {
    let l = v * (1.0 - s * 0.5);
    let sl = if l <= 0.0 || l >= 1.0 {
        0.0
    } else {
        (v - l) / l.min(1.0 - l)
    };
    [h, sl, l]
}
fn hsl_to_hsv([h, s, l]: [f32; 3]) -> [f32; 3] {
    let v = l + s * l.min(1.0 - l);
    [h, if v <= 0.0 { 0.0 } else { 2.0 * (1.0 - l / v) }, v]
}
#[cfg(test)]
mod redesign_tests {
    use super::*;
    #[test]
    fn escape_restores_invalid_hex_even_after_focus_is_cleared() {
        let ctx = egui::Context::default();
        let mut panel = ColorsPanel {
            hex_buffer: "#g".into(),
            hex_editing: true,
            hex_invalid: true,
            ..Default::default()
        };
        let input = egui::RawInput {
            events: vec![egui::Event::Key {
                key: egui::Key::Escape,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: egui::Modifiers::NONE,
            }],
            ..Default::default()
        };
        let _ = ctx.run_ui(input, |ui| panel.draw_hex_row(ui));
        assert_eq!(panel.hex_buffer, "#000000");
        assert!(!panel.hex_invalid && !panel.hex_editing);
    }
    #[test]
    fn hue_wrap_and_hsl_roundtrip() {
        for rgb in [[51, 102, 153], [30, 32, 31], [255, 119, 13]] {
            let color = cw::rgba(rgb[0], rgb[1], rgb[2], 255);
            let [h, s, v] = color_to_hsv(color);
            assert_eq!(hsv_to_color(h, s, v, 255), color);
        }
        assert_eq!(
            hsv_to_color(1.0, 1.0, 1.0, 255),
            hsv_to_color(0.0, 1.0, 1.0, 255)
        );
        for hsv in [[0.3, 0.7, 0.8], [0.9, 0.0, 0.4], [0.5, 1.0, 0.0]] {
            let got = hsl_to_hsv(hsv_to_hsl(hsv));
            assert!((got[2] - hsv[2]).abs() < 0.0001);
            if hsv[2] > 0.0 {
                assert!((got[1] - hsv[1]).abs() < 0.0001);
            }
        }
    }
    #[test]
    fn alpha_and_achromatic_keep_color_intent() {
        let mut panel = ColorsPanel::default();
        panel.apply_hsv([0.73, 0.8, 0.9], 0);
        panel.sync_hsv_from_colors();
        assert_eq!(panel.active_hsv(), [0.73, 0.8, 0.9]);
        panel.apply_hsv(panel.active_hsv(), 128);
        let hsv = color_to_hsv(panel.get_primary_color());
        assert!((hsv[2] - 0.9).abs() < 0.015);
        panel.set_primary_color(Color32::GRAY);
        assert!((panel.active_hsv()[0] - 0.73).abs() < 0.0001);
    }
}
