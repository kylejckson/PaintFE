use crate::assets::{Assets, Icon};
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

    pub fn get_primary_color(&self) -> Color32 {
        self.primary_color
    }

    pub fn get_secondary_color(&self) -> Color32 {
        self.secondary_color
    }

    pub fn swap_colors(&mut self) {
        std::mem::swap(&mut self.primary_color, &mut self.secondary_color);
        std::mem::swap(&mut self.primary_hsv, &mut self.secondary_hsv);
    }

    pub fn is_expanded(&self) -> bool {
        self.expanded
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
            self.primary_hsv = color_to_hsv(self.primary_color);
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
            self.secondary_hsv = color_to_hsv(self.secondary_color);
        }
    }

    // -- Top-level layout ---------------------------------------
    fn show_content(&mut self, ui: &mut egui::Ui, assets: &Assets) {
        self.sync_hsv_from_colors();

        if self.expanded {
            ui.horizontal(|ui| {
                // -- Left column (same as compact, width-constrained) --
                ui.vertical(|ui| {
                    ui.set_max_width(180.0);
                    self.draw_swatches(ui, assets);
                    ui.add_space(4.0);
                    self.draw_hue_ring_and_sv_triangle(ui);
                    ui.add_space(6.0);
                    self.draw_alpha_slider(ui);
                    ui.add_space(4.0);
                    self.draw_hex_row(ui, assets);
                });

                ui.add_space(6.0);
                ui.separator();
                ui.add_space(2.0);

                // -- Right column (sliders) --
                ui.vertical(|ui| {
                    ui.set_min_width(180.0);
                    ui.add_space(34.0); // align visually with ring area
                    self.draw_rgb_sliders(ui);
                    ui.add_space(10.0);
                    self.draw_hsv_sliders(ui);
                });
            });
        } else {
            self.draw_swatches(ui, assets);
            ui.add_space(4.0);
            self.draw_hue_ring_and_sv_triangle(ui);
            ui.add_space(6.0);
            self.draw_alpha_slider(ui);
            ui.add_space(4.0);
            self.draw_hex_row(ui, assets);
        }
    }

    // ====================================================================
    // WIDGET: Colour swatches + swap + expand toggle
    // ====================================================================

    fn draw_swatches(&mut self, ui: &mut egui::Ui, assets: &Assets) {
        ui.horizontal(|ui| {
            let pri_size = Vec2::new(30.0, 30.0);
            let sec_size = Vec2::new(24.0, 24.0);

            // -- primary swatch --
            let (pri_rect, pri_resp) = ui.allocate_exact_size(pri_size, egui::Sense::click());
            if ui.is_rect_visible(pri_rect) {
                let p = ui.painter();
                draw_checkerboard(p, pri_rect, 5.0);
                p.rect_filled(pri_rect, 3.0, self.primary_color);
                let border = if self.editing_primary {
                    Stroke::new(2.0, ui.visuals().selection.stroke.color)
                } else {
                    Stroke::new(1.0, ui.visuals().widgets.noninteractive.bg_stroke.color)
                };
                p.rect_stroke(pri_rect, 3.0, border);
            }
            if pri_resp.clicked() {
                self.editing_primary = true;
            }

            // -- secondary swatch --
            let (sec_rect, sec_resp) = ui.allocate_exact_size(sec_size, egui::Sense::click());
            if ui.is_rect_visible(sec_rect) {
                let p = ui.painter();
                draw_checkerboard(p, sec_rect, 4.0);
                p.rect_filled(sec_rect, 3.0, self.secondary_color);
                let border = if !self.editing_primary {
                    Stroke::new(2.0, ui.visuals().selection.stroke.color)
                } else {
                    Stroke::new(1.0, ui.visuals().widgets.noninteractive.bg_stroke.color)
                };
                p.rect_stroke(sec_rect, 3.0, border);
            }
            if sec_resp.clicked() {
                self.editing_primary = false;
            }

            // -- swap button --
            if assets
                .small_icon_button_frameless(ui, Icon::SwapColors)
                .clicked()
            {
                self.swap_colors();
            }

            // -- spacer + expand / collapse toggle  (right-aligned) --
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let icon = if self.expanded {
                    Icon::Collapse
                } else {
                    Icon::Expand
                };
                let tip = if self.expanded {
                    "Compact mode"
                } else {
                    "Show sliders"
                };
                if assets
                    .small_icon_button(ui, icon)
                    .on_hover_text(tip)
                    .clicked()
                {
                    self.expanded = !self.expanded;
                }
            });
        });
    }

    // ====================================================================
    // WIDGET: Hue ring + SV triangle  (the centrepiece)
    // ====================================================================

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
                let c0t = Color32::from_rgba_unmultiplied(c0.r(), c0.g(), c0.b(), 0);
                let c1t = Color32::from_rgba_unmultiplied(c1.r(), c1.g(), c1.b(), 0);
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
                let c0t = Color32::from_rgba_unmultiplied(c0.r(), c0.g(), c0.b(), 0);
                let c1t = Color32::from_rgba_unmultiplied(c1.r(), c1.g(), c1.b(), 0);
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

        // -- interaction --
        if let Some(mp) = response.interact_pointer_pos() {
            let delta = mp - center;
            let dist = delta.length();

            // decide zone on first contact
            if response.drag_started() || (response.clicked() && drag_zone == DragZone::None) {
                if dist >= (inner_r - 6.0) && dist <= (outer_r + 6.0) {
                    drag_zone = DragZone::HueRing;
                } else if point_in_triangle(mp, vert_a, vert_b, vert_c) || dist < inner_r {
                    drag_zone = DragZone::SvTriangle;
                }
            }

            if response.dragged() || response.clicked() {
                match drag_zone {
                    DragZone::HueRing => {
                        let mut new_h = delta.y.atan2(delta.x) / TAU;
                        if new_h < 0.0 {
                            new_h += 1.0;
                        }
                        h = new_h;
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
                            let new_v = (wa + wb).clamp(0.0, 1.0);
                            let new_s = if new_v > 0.001 {
                                (wa / new_v).clamp(0.0, 1.0)
                            } else {
                                0.0
                            };
                            s = new_s;
                            v = new_v;
                            changed = true;
                        }
                    }
                    DragZone::None => {}
                }
            }
        }

        if !response.dragged() {
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

    // ====================================================================
    // WIDGET: Alpha slider  (horizontal, full-width)
    // ====================================================================

    fn draw_alpha_slider(&mut self, ui: &mut egui::Ui) {
        let editing = self.editing_primary;
        let hsv = if editing {
            self.primary_hsv
        } else {
            self.secondary_hsv
        };
        let color = if editing {
            self.primary_color
        } else {
            self.secondary_color
        };
        let (h, s, v) = (hsv[0], hsv[1], hsv[2]);
        let mut a = color.a() as f32 / 255.0;
        let mut changed = false;

        let bar_h = 14.0;

        ui.horizontal(|ui| {
            ui.add_sized(
                [14.0, bar_h + 4.0],
                egui::Label::new(egui::RichText::new("A").small().strong()),
            );

            let bar_w = ui.available_width() - 40.0;
            let desired = Vec2::new(bar_w, bar_h + 4.0);
            let (rect, resp) = ui.allocate_exact_size(desired, egui::Sense::click_and_drag());
            let bar = egui::Rect::from_min_size(
                Pos2::new(rect.min.x, rect.center().y - bar_h / 2.0),
                Vec2::new(rect.width(), bar_h),
            );

            if ui.is_rect_visible(rect) {
                let p = ui.painter();

                // checkerboard behind slider
                draw_checkerboard(p, bar, 4.0);

                // alpha gradient
                let base = hsv_to_color(h, s, v, 255);
                let steps = 48;
                for i in 0..steps {
                    let t = i as f32 / steps as f32;
                    let x0 = bar.min.x + t * bar.width();
                    let x1 = bar.min.x + (i + 1) as f32 / steps as f32 * bar.width() + 0.5;
                    p.rect_filled(
                        egui::Rect::from_min_max(
                            Pos2::new(x0, bar.min.y),
                            Pos2::new(x1, bar.max.y),
                        ),
                        0.0,
                        Color32::from_rgba_unmultiplied(
                            base.r(),
                            base.g(),
                            base.b(),
                            (t * 255.0) as u8,
                        ),
                    );
                }

                // border
                p.rect_stroke(bar, 2.0, Stroke::new(1.0, Color32::from_black_alpha(45)));

                // thumb
                let tx = bar.min.x + a * bar.width();
                let ty = bar.center().y;
                let tr = bar_h / 2.0 + 1.0;
                p.circle_filled(Pos2::new(tx, ty + 0.5), tr, Color32::from_black_alpha(18));
                p.circle_filled(Pos2::new(tx, ty), tr, Color32::WHITE);
                p.circle_stroke(
                    Pos2::new(tx, ty),
                    tr,
                    Stroke::new(1.0, Color32::from_black_alpha(70)),
                );
            }

            if (resp.dragged() || resp.clicked())
                && let Some(mp) = resp.interact_pointer_pos()
            {
                a = ((mp.x - bar.min.x) / bar.width()).clamp(0.0, 1.0);
                changed = true;
            }

            // alpha 0-255 drag-value
            let mut alpha_val = (a * 255.0).round() as u32;
            if ui
                .add_sized(
                    [32.0, 16.0],
                    egui::DragValue::new(&mut alpha_val)
                        .clamp_range(0..=255)
                        .speed(1),
                )
                .changed()
            {
                a = alpha_val as f32 / 255.0;
                changed = true;
            }
        });

        if changed {
            let new_a = (a * 255.0).round() as u8;
            // Reconstruct from HSV — avoids premultiplied-alpha round-trip corruption
            let [ch, cs, cv] = if editing {
                self.primary_hsv
            } else {
                self.secondary_hsv
            };
            if editing {
                self.primary_color = hsv_to_color(ch, cs, cv, new_a);
            } else {
                self.secondary_color = hsv_to_color(ch, cs, cv, new_a);
            }
        }
    }

    // ====================================================================
    // WIDGET: Hex input row
    // ====================================================================

    fn draw_hex_row(&mut self, ui: &mut egui::Ui, assets: &Assets) {
        let editing = self.editing_primary;
        let a = if editing {
            self.primary_color.a()
        } else {
            self.secondary_color.a()
        };
        // Derive un-premultiplied RGB from HSV (Color32 stores premultiplied)
        let hsv = if editing {
            self.primary_hsv
        } else {
            self.secondary_hsv
        };
        let opaque = hsv_to_color(hsv[0], hsv[1], hsv[2], 255);
        let (r, g, b) = (opaque.r(), opaque.g(), opaque.b());

        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("#").monospace().strong());

            let mut hex = format!("{:02X}{:02X}{:02X}", r, g, b);
            if ui
                .add_sized(
                    [52.0, 18.0],
                    egui::TextEdit::singleline(&mut hex).font(egui::TextStyle::Monospace),
                )
                .changed()
                && hex.len() == 6
                && let Ok(val) = u32::from_str_radix(&hex, 16)
            {
                let nr = ((val >> 16) & 0xFF) as u8;
                let ng = ((val >> 8) & 0xFF) as u8;
                let nb = (val & 0xFF) as u8;
                let nc = Color32::from_rgba_unmultiplied(nr, ng, nb, a);
                if editing {
                    self.primary_color = nc;
                    self.primary_hsv = color_to_hsv(nc);
                } else {
                    self.secondary_color = nc;
                    self.secondary_hsv = color_to_hsv(nc);
                }
            }

            if assets
                .small_icon_button_frameless(ui, Icon::CopyHex)
                .clicked()
            {
                ui.output_mut(|o| o.copied_text = format!("{:02X}{:02X}{:02X}", r, g, b));
            }
        });
    }

    // ====================================================================
    // WIDGET: RGB gradient sliders  (expanded mode)
    // ====================================================================

    fn draw_rgb_sliders(&mut self, ui: &mut egui::Ui) {
        let editing = self.editing_primary;
        let a = if editing {
            self.primary_color.a()
        } else {
            self.secondary_color.a()
        };
        // Derive un-premultiplied RGB from HSV (Color32 stores premultiplied)
        let hsv_cur = if editing {
            self.primary_hsv
        } else {
            self.secondary_hsv
        };
        let opaque = hsv_to_color(hsv_cur[0], hsv_cur[1], hsv_cur[2], 255);
        let mut r = opaque.r() as f32 / 255.0;
        let mut g = opaque.g() as f32 / 255.0;
        let mut b = opaque.b() as f32 / 255.0;
        let mut changed = false;

        let sw = 120.0;
        let sh = 12.0;

        ui.label(egui::RichText::new(t!("color_panel.rgb")).small().strong());
        ui.add_space(2.0);

        // -- R --
        ui.horizontal(|ui| {
            ui.add_sized(
                [12.0, 16.0],
                egui::Label::new(
                    egui::RichText::new("R")
                        .small()
                        .color(Color32::from_rgb(210, 90, 90)),
                ),
            );
            let g2 = g;
            let b2 = b;
            if Self::gradient_bar(
                ui,
                &mut r,
                |t| Color32::from_rgb((t * 255.0) as u8, (g2 * 255.0) as u8, (b2 * 255.0) as u8),
                sw,
                sh,
            ) {
                changed = true;
            }
            let mut ri = (r * 255.0).round() as u32;
            if ui
                .add_sized(
                    [32.0, 16.0],
                    egui::DragValue::new(&mut ri).clamp_range(0..=255).speed(1),
                )
                .changed()
            {
                r = ri as f32 / 255.0;
                changed = true;
            }
        });

        // -- G --
        ui.horizontal(|ui| {
            ui.add_sized(
                [12.0, 16.0],
                egui::Label::new(
                    egui::RichText::new("G")
                        .small()
                        .color(Color32::from_rgb(80, 190, 80)),
                ),
            );
            let r2 = r;
            let b2 = b;
            if Self::gradient_bar(
                ui,
                &mut g,
                |t| Color32::from_rgb((r2 * 255.0) as u8, (t * 255.0) as u8, (b2 * 255.0) as u8),
                sw,
                sh,
            ) {
                changed = true;
            }
            let mut gi = (g * 255.0).round() as u32;
            if ui
                .add_sized(
                    [32.0, 16.0],
                    egui::DragValue::new(&mut gi).clamp_range(0..=255).speed(1),
                )
                .changed()
            {
                g = gi as f32 / 255.0;
                changed = true;
            }
        });

        // -- B --
        ui.horizontal(|ui| {
            ui.add_sized(
                [12.0, 16.0],
                egui::Label::new(
                    egui::RichText::new("B")
                        .small()
                        .color(Color32::from_rgb(90, 120, 220)),
                ),
            );
            let r2 = r;
            let g2 = g;
            if Self::gradient_bar(
                ui,
                &mut b,
                |t| Color32::from_rgb((r2 * 255.0) as u8, (g2 * 255.0) as u8, (t * 255.0) as u8),
                sw,
                sh,
            ) {
                changed = true;
            }
            let mut bi = (b * 255.0).round() as u32;
            if ui
                .add_sized(
                    [32.0, 16.0],
                    egui::DragValue::new(&mut bi).clamp_range(0..=255).speed(1),
                )
                .changed()
            {
                b = bi as f32 / 255.0;
                changed = true;
            }
        });

        if changed {
            let nc = Color32::from_rgba_unmultiplied(
                (r * 255.0).round() as u8,
                (g * 255.0).round() as u8,
                (b * 255.0).round() as u8,
                a,
            );
            // Derive HSV from opaque color to avoid premultiplied-alpha issues
            let opaque_nc = Color32::from_rgb(
                (r * 255.0).round() as u8,
                (g * 255.0).round() as u8,
                (b * 255.0).round() as u8,
            );
            if editing {
                self.primary_color = nc;
                self.primary_hsv = color_to_hsv(opaque_nc);
            } else {
                self.secondary_color = nc;
                self.secondary_hsv = color_to_hsv(opaque_nc);
            }
        }
    }

    // ====================================================================
    // WIDGET: HSV gradient sliders  (expanded mode)
    // ====================================================================

    fn draw_hsv_sliders(&mut self, ui: &mut egui::Ui) {
        let editing = self.editing_primary;
        let hsv = if editing {
            self.primary_hsv
        } else {
            self.secondary_hsv
        };
        let alpha = if editing {
            self.primary_color.a()
        } else {
            self.secondary_color.a()
        };
        let mut h = hsv[0];
        let mut s = hsv[1];
        let mut v = hsv[2];
        let mut changed = false;

        let sw = 120.0;
        let sh = 12.0;

        ui.label(egui::RichText::new(t!("color_panel.hsv")).small().strong());
        ui.add_space(2.0);

        // -- H --
        ui.horizontal(|ui| {
            ui.add_sized(
                [12.0, 16.0],
                egui::Label::new(egui::RichText::new("H").small()),
            );
            if Self::gradient_bar(ui, &mut h, |t| hsv_to_color(t, 1.0, 1.0, 255), sw, sh) {
                changed = true;
            }
            let mut hi = (h * 360.0).round() as u32;
            if ui
                .add_sized(
                    [32.0, 16.0],
                    egui::DragValue::new(&mut hi)
                        .clamp_range(0..=360)
                        .speed(1)
                        .suffix("°"),
                )
                .changed()
            {
                h = hi as f32 / 360.0;
                changed = true;
            }
        });

        // -- S --
        ui.horizontal(|ui| {
            ui.add_sized(
                [12.0, 16.0],
                egui::Label::new(egui::RichText::new("S").small()),
            );
            let h2 = h;
            let v2 = v;
            if Self::gradient_bar(ui, &mut s, |t| hsv_to_color(h2, t, v2, 255), sw, sh) {
                changed = true;
            }
            let mut si = (s * 100.0).round() as u32;
            if ui
                .add_sized(
                    [32.0, 16.0],
                    egui::DragValue::new(&mut si)
                        .clamp_range(0..=100)
                        .speed(1)
                        .suffix("%"),
                )
                .changed()
            {
                s = si as f32 / 100.0;
                changed = true;
            }
        });

        // -- V --
        ui.horizontal(|ui| {
            ui.add_sized(
                [12.0, 16.0],
                egui::Label::new(egui::RichText::new("V").small()),
            );
            let h2 = h;
            let s2 = s;
            if Self::gradient_bar(ui, &mut v, |t| hsv_to_color(h2, s2, t, 255), sw, sh) {
                changed = true;
            }
            let mut vi = (v * 100.0).round() as u32;
            if ui
                .add_sized(
                    [32.0, 16.0],
                    egui::DragValue::new(&mut vi)
                        .clamp_range(0..=100)
                        .speed(1)
                        .suffix("%"),
                )
                .changed()
            {
                v = vi as f32 / 100.0;
                changed = true;
            }
        });

        if changed {
            let nc = hsv_to_color(h, s, v, alpha);
            if editing {
                self.primary_hsv = [h, s, v];
                self.primary_color = nc;
            } else {
                self.secondary_hsv = [h, s, v];
                self.secondary_color = nc;
            }
        }
    }

    // ====================================================================
    // Reusable gradient bar  (static — no self access)
    // ====================================================================

    fn gradient_bar(
        ui: &mut egui::Ui,
        value: &mut f32, // normalised 0..1
        gradient_fn: impl Fn(f32) -> Color32,
        width: f32,
        height: f32,
    ) -> bool {
        let pad = 4.0; // extra height for thumb overhang
        let desired = Vec2::new(width, height + pad);
        let (rect, resp) = ui.allocate_exact_size(desired, egui::Sense::click_and_drag());
        let bar = egui::Rect::from_min_size(
            Pos2::new(rect.min.x, rect.center().y - height / 2.0),
            Vec2::new(rect.width(), height),
        );
        let mut did_change = false;

        if ui.is_rect_visible(rect) {
            let p = ui.painter();

            // gradient fill  (48 steps — smooth on all monitors)
            let steps = 48u32;
            for i in 0..steps {
                let t = i as f32 / steps as f32;
                let x0 = bar.min.x + t * bar.width();
                let x1 = bar.min.x + (i + 1) as f32 / steps as f32 * bar.width() + 0.5;
                p.rect_filled(
                    egui::Rect::from_min_max(Pos2::new(x0, bar.min.y), Pos2::new(x1, bar.max.y)),
                    0.0,
                    gradient_fn(t),
                );
            }

            // border
            p.rect_stroke(bar, 2.0, Stroke::new(1.0, Color32::from_black_alpha(50)));

            // circular thumb
            let tx = bar.min.x + *value * bar.width();
            let ty = bar.center().y;
            let tr = height / 2.0 + 1.0;
            // shadow
            p.circle_filled(Pos2::new(tx, ty + 0.5), tr, Color32::from_black_alpha(18));
            // body
            p.circle_filled(Pos2::new(tx, ty), tr, Color32::WHITE);
            // outline
            p.circle_stroke(
                Pos2::new(tx, ty),
                tr,
                Stroke::new(1.0, Color32::from_black_alpha(70)),
            );
        }

        if (resp.dragged() || resp.clicked())
            && let Some(mp) = resp.interact_pointer_pos()
        {
            *value = ((mp.x - bar.min.x) / bar.width()).clamp(0.0, 1.0);
            did_change = true;
        }
        did_change
    }
}

// ============================================================================
// Free helper functions
// ============================================================================

/// Draw a checkerboard pattern inside `rect` (for transparency preview).
fn draw_checkerboard(painter: &egui::Painter, rect: egui::Rect, cell: f32) {
    painter.rect_filled(rect, 0.0, Color32::WHITE);
    let cols = (rect.width() / cell).ceil() as i32;
    let rows = (rect.height() / cell).ceil() as i32;
    for row in 0..rows {
        for col in 0..cols {
            if (row + col) % 2 == 1 {
                let cr = egui::Rect::from_min_size(
                    Pos2::new(
                        rect.min.x + col as f32 * cell,
                        rect.min.y + row as f32 * cell,
                    ),
                    Vec2::new(cell, cell),
                )
                .intersect(rect);
                painter.rect_filled(cr, 0.0, Color32::from_gray(200));
            }
        }
    }
}

/// Barycentric coordinates of `p` w.r.t. triangle (a, b, c).
/// Returns (weight_a, weight_b, weight_c).
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
    let r = color.r() as f32 / 255.0;
    let g = color.g() as f32 / 255.0;
    let b = color.b() as f32 / 255.0;
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
    let h6 = h * 6.0;
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
    Color32::from_rgba_unmultiplied(
        ((r + m) * 255.0) as u8,
        ((g + m) * 255.0) as u8,
        ((b + m) * 255.0) as u8,
        a,
    )
}
