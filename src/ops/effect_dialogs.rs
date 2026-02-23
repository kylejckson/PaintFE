// ============================================================================
// EFFECT DIALOGS â€” Modal dialog UIs for all 24 effects
// ============================================================================
//
// Each dialog follows the standard pattern:
//   - struct with effect params + original_pixels + original_flat + layer_idx + live_preview
//   - new(&CanvasState) constructor
//   - show(&mut self, ctx) -> DialogResult<params>
//   - Accent header, Grid params, preview controls, OK/Cancel footer
// ============================================================================

use eframe::egui;
use egui::Color32;
use image::RgbaImage;

use crate::canvas::{CanvasState, TiledImage};
use super::dialogs::{DialogResult, paint_dialog_header, section_label, accent_separator,
                     numeric_field_with_buttons, preview_controls, dialog_footer,
                     DialogColors};
use super::effects::{NoiseType, HalftoneShape, GridStyle, OutlineMode, ColorFilterMode};

// Create common effect dialog fields
macro_rules! effect_dialog_base {
    ($name:ident { $($field:ident : $ty:ty = $default:expr),* $(,)? }) => {
        #[allow(dead_code)]
        pub struct $name {
            pub original_pixels: Option<TiledImage>,
            pub original_flat: Option<RgbaImage>,
            /// Unused legacy fields kept for API compatibility.
            pub preview_flat_small: Option<RgbaImage>,
            pub preview_scale: f32,
            pub layer_idx: usize,
            pub live_preview: bool,
            /// Set true when a slider value changes; cleared when the effect is applied.
            pub needs_apply: bool,
            /// True on any frame where a slider is actively being dragged.
            pub dragging: bool,
            /// Fully-computed effect result at preview scale, used for progressive reveal.
            pub processed_preview: Option<RgbaImage>,
            /// Next row to reveal in the progressive top-down display.
            pub progressive_row: u32,
            /// Background flat-extraction job: populated by rayon, polled each frame.
            pub pending_flat: Option<std::sync::Arc<std::sync::Mutex<Option<RgbaImage>>>>,
            $(pub $field: $ty,)*
        }

        impl $name {
            pub fn new(state: &CanvasState) -> Self {
                let idx = state.active_layer_index;
                // Clone original pixels (fast — COW Arc clones only).
                let original_pixels = state.layers.get(idx).map(|l| l.pixels.clone());
                // Defer the expensive to_rgba_image() to a rayon thread so the dialog
                // opens on the very next frame.  poll_flat() will resolve it.
                let pending_flat = if original_pixels.is_some() {
                    let arc: std::sync::Arc<std::sync::Mutex<Option<RgbaImage>>> =
                        std::sync::Arc::new(std::sync::Mutex::new(None));
                    let arc_clone = arc.clone();
                    let tiled = original_pixels.clone().unwrap();
                    rayon::spawn(move || {
                        let flat = tiled.to_rgba_image();
                        if let Ok(mut guard) = arc_clone.lock() {
                            *guard = Some(flat);
                        }
                    });
                    Some(arc)
                } else {
                    None
                };
                Self {
                    original_pixels,
                    original_flat: None, // populated by poll_flat()
                    preview_flat_small: None,
                    preview_scale: 1.0,
                    layer_idx: idx,
                    live_preview: true,
                    needs_apply: false,
                    dragging: false,
                    processed_preview: None,
                    progressive_row: 0,
                    pending_flat,
                    $($field: $default,)*
                }
            }

            /// Call each frame while the dialog is open.  Returns `true` the first
            /// time `original_flat` becomes available (i.e. the background job just
            /// finished).  Callers should use the `true` return to trigger an initial
            /// preview render.
            pub fn poll_flat(&mut self) -> bool {
                if self.original_flat.is_some() {
                    return false; // already resolved
                }
                // Extract the flat image without holding a borrow on self.pending_flat
                // (so we can immediately assign self.original_flat / self.pending_flat).
                let maybe_flat = self.pending_flat.as_ref().and_then(|arc| {
                    arc.try_lock().ok().and_then(|mut guard| guard.take())
                });
                if let Some(flat) = maybe_flat {
                    self.original_flat = Some(flat);
                    self.pending_flat = None;
                    return true;
                }
                false
            }
        }
    };
}

/// Track a slider for deferred preview: the slider value updates instantly
/// (no lag) but the effect is only applied when the slider is *released*.
/// Returns `true` when the slider was just released after being dragged.
pub fn track_slider(response: &egui::Response, dragging: &mut bool) -> bool {
    if response.dragged() || response.changed() {
        *dragging = true;
    }
    if response.drag_released() {
        *dragging = false;
        return true; // "apply now"
    }
    false
}

// ============================================================================
// BLUR DIALOGS
// ============================================================================

effect_dialog_base!(BokehBlurDialog { radius: f32 = 0.0, advanced_blur: bool = false });

impl BokehBlurDialog {
    pub fn show(&mut self, ctx: &egui::Context) -> DialogResult<f32> {
        let mut result = DialogResult::Open;
        let colors = DialogColors::from_ctx(ctx);

        egui::Window::new("dialog_bokeh_blur")
            .title_bar(false).collapsible(false).resizable(false)
            .default_pos(egui::pos2(ctx.screen_rect().center().x - 175.0, 60.0))
            .show(ctx, |ui| {
                ui.set_min_width(350.0);
                paint_dialog_header(ui, &colors, "\u{2B55}", &t!("dialog.bokeh_blur"));
                ui.add_space(4.0);
                section_label(ui, &colors, "BLUR SETTINGS");

                let mut changed = false;
                let slider_max = if self.advanced_blur { 100.0 } else { 10.0 };
                
                egui::Grid::new("bokeh_params").num_columns(2).spacing([8.0, 6.0]).show(ui, |ui| {
                    ui.label("Radius");
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 4.0;
                        if self.advanced_blur {
                            // Advanced: editable DragValue (up to 100)
                            let r = ui.add(
                                egui::DragValue::new(&mut self.radius)
                                    .speed(0.2)
                                    .clamp_range(1.0..=100.0)
                                    .max_decimals(1)
                            );
                            if track_slider(&r, &mut self.dragging) {
                                changed = true;
                            }
                        } else {
                            // Normal: slider capped at 10
                            let r = ui.add(egui::Slider::new(&mut self.radius, 1.0..=slider_max).max_decimals(1));
                            if track_slider(&r, &mut self.dragging) {
                                changed = true;
                            }
                        }
                    });
                    ui.end_row();

                    ui.label("Quick");
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 4.0;
                        let presets: &[(&str, f32)] = if self.advanced_blur {
                            &[("Subtle", 3.0), ("Soft", 8.0), ("Medium", 20.0), ("Dreamy", 40.0), ("Max", 80.0)]
                        } else {
                            &[("Subtle", 1.5), ("Soft", 3.0), ("Medium", 6.0), ("Dreamy", 10.0)]
                        };
                        for &(label, val) in presets {
                            let is_close = (self.radius - val).abs() < 1.0;
                            let btn = if is_close {
                                egui::Button::new(egui::RichText::new(label).strong().size(11.0)).fill(colors.accent_faint)
                            } else {
                                egui::Button::new(egui::RichText::new(label).size(11.0))
                            };
                            if ui.add(btn).clicked() { self.radius = val; changed = true; }
                        }
                    });
                    ui.end_row();

                    // Advanced blur toggle
                    ui.label("");
                    ui.horizontal(|ui| {
                        if ui.checkbox(&mut self.advanced_blur, "Advanced (up to 100)").changed() {
                            // Clamp radius when switching back to normal mode
                            if !self.advanced_blur && self.radius > 10.0 {
                                self.radius = 10.0;
                                changed = true;
                            }
                        }
                    });
                    ui.end_row();
                });

                accent_separator(ui, &colors);
                
                let manual = preview_controls(ui, &colors, &mut self.live_preview);
                if (changed && self.live_preview) || manual { result = DialogResult::Changed; }

                let (ok, cancel) = dialog_footer(ui, &colors);
                if ok { result = DialogResult::Ok(self.radius); }
                if cancel { result = DialogResult::Cancel; }
            });
        result
    }
}

// -------

effect_dialog_base!(MotionBlurDialog { angle: f32 = 0.0, distance: f32 = 0.0 });

impl MotionBlurDialog {
    pub fn show(&mut self, ctx: &egui::Context) -> DialogResult<(f32, f32)> {
        let mut result = DialogResult::Open;
        let colors = DialogColors::from_ctx(ctx);

        egui::Window::new("dialog_motion_blur")
            .title_bar(false).collapsible(false).resizable(false)
            .default_pos(egui::pos2(ctx.screen_rect().center().x - 175.0, 60.0))
            .show(ctx, |ui| {
                ui.set_min_width(350.0);
                paint_dialog_header(ui, &colors, "\u{27A1}", &t!("dialog.motion_blur"));
                ui.add_space(4.0);
                section_label(ui, &colors, "MOTION SETTINGS");

                let mut changed = false;
                egui::Grid::new("motion_params").num_columns(2).spacing([8.0, 6.0]).show(ui, |ui| {
                    ui.label("Angle");
                    let r = ui.add(egui::Slider::new(&mut self.angle, -180.0..=180.0).suffix("Â°").max_decimals(0)); if track_slider(&r, &mut self.dragging) {
                        changed = true;
                    }
                    ui.end_row();

                    ui.label("Distance");
                    let r = ui.add(egui::Slider::new(&mut self.distance, 1.0..=100.0).suffix(" px").max_decimals(0)); if track_slider(&r, &mut self.dragging) {
                        changed = true;
                    }
                    ui.end_row();

                    ui.label("Direction");
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 4.0;
                        for &(label, val) in &[("â†’", 0.0), ("â†—", -45.0), ("â†‘", -90.0), ("â†–", -135.0)] {
                            let btn = if (self.angle - val).abs() < 1.0 {
                                egui::Button::new(egui::RichText::new(label).strong()).fill(colors.accent_faint)
                            } else {
                                egui::Button::new(label)
                            };
                            if ui.add(btn).clicked() { self.angle = val; changed = true; }
                        }
                    });
                    ui.end_row();
                });

                accent_separator(ui, &colors);
                
                let manual = preview_controls(ui, &colors, &mut self.live_preview);
                if (changed && self.live_preview) || manual { result = DialogResult::Changed; }

                let (ok, cancel) = dialog_footer(ui, &colors);
                if ok { result = DialogResult::Ok((self.angle, self.distance)); }
                if cancel { result = DialogResult::Cancel; }
            });
        result
    }
}

// -------

effect_dialog_base!(BoxBlurDialog { radius: f32 = 0.0 });

impl BoxBlurDialog {
    pub fn show(&mut self, ctx: &egui::Context) -> DialogResult<f32> {
        let mut result = DialogResult::Open;
        let colors = DialogColors::from_ctx(ctx);

        egui::Window::new("dialog_box_blur")
            .title_bar(false).collapsible(false).resizable(false)
            .default_pos(egui::pos2(ctx.screen_rect().center().x - 175.0, 60.0))
            .show(ctx, |ui| {
                ui.set_min_width(350.0);
                paint_dialog_header(ui, &colors, "\u{25A3}", &t!("dialog.box_blur"));
                ui.add_space(4.0);
                section_label(ui, &colors, "BLUR SETTINGS");

                let mut changed = false;
                egui::Grid::new("box_params").num_columns(2).spacing([8.0, 6.0]).show(ui, |ui| {
                    ui.label("Radius");
                    let r = ui.add(egui::Slider::new(&mut self.radius, 1.0..=50.0).max_decimals(1)); if track_slider(&r, &mut self.dragging) {
                        changed = true;
                    }
                    ui.end_row();
                });

                accent_separator(ui, &colors);
                
                let manual = preview_controls(ui, &colors, &mut self.live_preview);
                if (changed && self.live_preview) || manual { result = DialogResult::Changed; }

                let (ok, cancel) = dialog_footer(ui, &colors);
                if ok { result = DialogResult::Ok(self.radius); }
                if cancel { result = DialogResult::Cancel; }
            });
        result
    }
}

// -------

effect_dialog_base!(ZoomBlurDialog {
    center_x: f32 = 0.5,
    center_y: f32 = 0.5,
    intensity: f32 = 30.0,   // 1–100, mapped to 0.01–0.99 strength
    quality: u8 = 1,          // 0=Fast(8), 1=Normal(16), 2=High(32)
    tint_enabled: bool = false,
    tint_r: f32 = 1.0,
    tint_g: f32 = 0.6,
    tint_b: f32 = 0.1,
    tint_mix: f32 = 30.0,    // 0–100
    first_open: bool = true,
});

fn zoom_quality_samples(q: u8) -> u32 {
    match q { 0 => 8, 1 => 16, _ => 32 }
}

impl ZoomBlurDialog {
    /// Returns (center_x, center_y, strength, samples, tint_color, tint_strength)
    pub fn current_params(&self) -> (f32, f32, f32, u32, [f32; 4], f32) {
        let strength = (self.intensity / 100.0).clamp(0.01, 0.99);
        let samples  = zoom_quality_samples(self.quality);
        let tint     = [self.tint_r, self.tint_g, self.tint_b, 1.0];
        let ts       = if self.tint_enabled { self.tint_mix / 100.0 } else { 0.0 };
        (self.center_x, self.center_y, strength, samples, tint, ts)
    }

    pub fn show(&mut self, ctx: &egui::Context) -> DialogResult<()> {
        let mut result = DialogResult::Open;
        let colors = DialogColors::from_ctx(ctx);

        egui::Window::new("dialog_zoom_blur")
            .title_bar(false).collapsible(false).resizable(false)
            .default_pos(egui::pos2(ctx.screen_rect().center().x - 190.0, 60.0))
            .show(ctx, |ui| {
                ui.set_min_width(380.0);
                paint_dialog_header(ui, &colors, "\u{25CE}", &t!("dialog.zoom_blur"));
                ui.add_space(4.0);

                let mut changed = false;

                // Trigger an initial preview on the very first frame the dialog is shown.
                if self.first_open {
                    self.first_open = false;
                    changed = true;
                }

                // --------------------------------------------------
                // ZOOM ORIGIN
                // --------------------------------------------------
                section_label(ui, &colors, "ZOOM ORIGIN");
                egui::Grid::new("zoom_origin_params").num_columns(2).spacing([8.0, 6.0]).show(ui, |ui| {
                    ui.label("Center X");
                    let r = ui.add(egui::Slider::new(&mut self.center_x, 0.0..=1.0)
                        .custom_formatter(|v, _| format!("{:.0}%", v * 100.0))
                        .custom_parser(|s| s.trim_end_matches('%').parse::<f64>().ok().map(|v| v / 100.0)));
                    if track_slider(&r, &mut self.dragging) { changed = true; }
                    ui.end_row();

                    ui.label("Center Y");
                    let r = ui.add(egui::Slider::new(&mut self.center_y, 0.0..=1.0)
                        .custom_formatter(|v, _| format!("{:.0}%", v * 100.0))
                        .custom_parser(|s| s.trim_end_matches('%').parse::<f64>().ok().map(|v| v / 100.0)));
                    if track_slider(&r, &mut self.dragging) { changed = true; }
                    ui.end_row();

                    ui.label("Preset");
                    ui.horizontal_wrapped(|ui| {
                        ui.spacing_mut().item_spacing = egui::vec2(3.0, 3.0);
                        let presets: &[(&str, f32, f32)] = &[
                            ("↖", 0.0, 0.0), ("↑", 0.5, 0.0), ("↗", 1.0, 0.0),
                            ("←", 0.0, 0.5), ("⊙", 0.5, 0.5), ("→", 1.0, 0.5),
                            ("↙", 0.0, 1.0), ("↓", 0.5, 1.0), ("↘", 1.0, 1.0),
                        ];
                        for &(lbl, px, py) in presets {
                            let active = (self.center_x - px).abs() < 0.01 && (self.center_y - py).abs() < 0.01;
                            let btn = if active {
                                egui::Button::new(egui::RichText::new(lbl).strong()).fill(colors.accent_faint)
                            } else {
                                egui::Button::new(lbl)
                            };
                            if ui.add_sized([28.0, 22.0], btn).clicked() {
                                self.center_x = px; self.center_y = py; changed = true;
                            }
                        }
                    });
                    ui.end_row();
                });

                accent_separator(ui, &colors);

                // --------------------------------------------------
                // BLUR SETTINGS
                // --------------------------------------------------
                section_label(ui, &colors, "BLUR SETTINGS");
                egui::Grid::new("zoom_blur_params").num_columns(2).spacing([8.0, 6.0]).show(ui, |ui| {
                    ui.label("Intensity");
                    let r = ui.add(egui::Slider::new(&mut self.intensity, 1.0..=100.0)
                        .max_decimals(0).suffix("%"));
                    if track_slider(&r, &mut self.dragging) { changed = true; }
                    ui.end_row();

                    ui.label("Quality");
                    egui::ComboBox::from_id_source("zoom_quality")
                        .selected_text(match self.quality { 0 => "Fast", 1 => "Normal", _ => "High" })
                        .show_ui(ui, |ui| {
                            if ui.selectable_value(&mut self.quality, 0, "Fast  (8 samples)").changed()   { changed = true; }
                            if ui.selectable_value(&mut self.quality, 1, "Normal (16 samples)").changed() { changed = true; }
                            if ui.selectable_value(&mut self.quality, 2, "High  (32 samples)").changed()  { changed = true; }
                        });
                    ui.end_row();

                    ui.label("Quick");
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 4.0;
                        for &(lbl, val) in &[("Subtle", 10.0f32), ("Soft", 25.0), ("Strong", 55.0), ("Max", 90.0)] {
                            let active = (self.intensity - val).abs() < 2.0;
                            let btn = if active {
                                egui::Button::new(egui::RichText::new(lbl).strong().size(11.0)).fill(colors.accent_faint)
                            } else {
                                egui::Button::new(egui::RichText::new(lbl).size(11.0))
                            };
                            if ui.add(btn).clicked() { self.intensity = val; changed = true; }
                        }
                    });
                    ui.end_row();
                });

                accent_separator(ui, &colors);

                // --------------------------------------------------
                // COLOR TINT
                // --------------------------------------------------
                section_label(ui, &colors, "COLOR TINT");
                egui::Grid::new("zoom_tint_params").num_columns(2).spacing([8.0, 6.0]).show(ui, |ui| {
                    ui.label("Enable");
                    if ui.checkbox(&mut self.tint_enabled, "Radial color tint at origin").changed() {
                        changed = true;
                    }
                    ui.end_row();

                    if self.tint_enabled {
                        ui.label("Tint Color");
                        ui.horizontal(|ui| {
                            let mut col = egui::Color32::from_rgb(
                                (self.tint_r * 255.0) as u8,
                                (self.tint_g * 255.0) as u8,
                                (self.tint_b * 255.0) as u8,
                            );
                            if ui.color_edit_button_srgba(&mut col).changed() {
                                self.tint_r = col.r() as f32 / 255.0;
                                self.tint_g = col.g() as f32 / 255.0;
                                self.tint_b = col.b() as f32 / 255.0;
                                changed = true;
                            }
                            ui.spacing_mut().item_spacing.x = 4.0;
                            for &(lbl, r, g, b) in &[
                                ("White",  1.0f32, 1.0f32, 1.0f32),
                                ("Black",  0.0,    0.0,    0.0   ),
                                ("Warm",   1.0,    0.6,    0.1   ),
                                ("Cool",   0.2,    0.5,    1.0   ),
                                ("Fire",   1.0,    0.2,    0.0   ),
                            ] {
                                let active = (self.tint_r - r).abs() < 0.05
                                          && (self.tint_g - g).abs() < 0.05
                                          && (self.tint_b - b).abs() < 0.05;
                                let swatch = egui::Button::new("  ")
                                    .fill(egui::Color32::from_rgb((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8))
                                    .stroke(if active {
                                        egui::Stroke::new(2.0, colors.accent)
                                    } else {
                                        egui::Stroke::new(1.0, egui::Color32::GRAY)
                                    });
                                if ui.add_sized([20.0, 20.0], swatch).on_hover_text(lbl).clicked() {
                                    self.tint_r = r; self.tint_g = g; self.tint_b = b; changed = true;
                                }
                            }
                        });
                        ui.end_row();

                        ui.label("Mix");
                        let r = ui.add(egui::Slider::new(&mut self.tint_mix, 0.0..=100.0)
                            .max_decimals(0).suffix("%"));
                        if track_slider(&r, &mut self.dragging) { changed = true; }
                        ui.end_row();
                    }
                });

                accent_separator(ui, &colors);

                let manual = preview_controls(ui, &colors, &mut self.live_preview);
                if (changed && self.live_preview) || manual { result = DialogResult::Changed; }

                let (ok, cancel) = dialog_footer(ui, &colors);
                if ok     { result = DialogResult::Ok(()); }
                if cancel { result = DialogResult::Cancel; }
            });
        result
    }
}

// ============================================================================
// DISTORTION DIALOGS
// ============================================================================

effect_dialog_base!(CrystallizeDialog { cell_size: f32 = 1.0, seed: u32 = 42, first_open: bool = true });

impl CrystallizeDialog {
    pub fn show(&mut self, ctx: &egui::Context) -> DialogResult<(f32, u32)> {
        let mut result = DialogResult::Open;
        let colors = DialogColors::from_ctx(ctx);

        egui::Window::new("dialog_crystallize")
            .title_bar(false).collapsible(false).resizable(false)
            .default_pos(egui::pos2(ctx.screen_rect().center().x - 175.0, 60.0))
            .show(ctx, |ui| {
                ui.set_min_width(350.0);
                paint_dialog_header(ui, &colors, "\u{1F48E}", &t!("dialog.crystallize"));
                ui.add_space(4.0);
                section_label(ui, &colors, "VORONOI SETTINGS");

                let mut changed = false;
                egui::Grid::new("crystal_params").num_columns(2).spacing([8.0, 6.0]).show(ui, |ui| {
                    ui.label("Cell Size");
                    if numeric_field_with_buttons(ui, &mut self.cell_size, 0.5, 2.0..=100.0, " px", 2.0) {
                        changed = true;
                    }
                    ui.end_row();

                    ui.label("Seed");
                    ui.horizontal(|ui| {
                        let mut seed_f = self.seed as f32;
                        if ui.add(egui::DragValue::new(&mut seed_f).speed(1.0).clamp_range(0.0..=9999.0)).changed() {
                            self.seed = seed_f as u32; changed = true;
                        }
                        if ui.small_button("\u{1F3B2}").clicked() {
                            self.seed = (self.seed.wrapping_mul(1103515245).wrapping_add(12345)) % 10000;
                            changed = true;
                        }
                    });
                    ui.end_row();

                    ui.label("Quick");
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 4.0;
                        for &(label, val) in &[("Fine", 5.0), ("Medium", 15.0), ("Coarse", 30.0), ("Chunky", 60.0)] {
                            let btn = if (self.cell_size - val).abs() < 2.0 {
                                egui::Button::new(egui::RichText::new(label).strong().size(11.0)).fill(colors.accent_faint)
                            } else {
                                egui::Button::new(egui::RichText::new(label).size(11.0))
                            };
                            if ui.add(btn).clicked() { self.cell_size = val; changed = true; }
                        }
                    });
                    ui.end_row();
                });

                accent_separator(ui, &colors);
                let manual = preview_controls(ui, &colors, &mut self.live_preview);
                if (changed && self.live_preview) || manual { result = DialogResult::Changed; }

                let (ok, cancel) = dialog_footer(ui, &colors);
                if ok { result = DialogResult::Ok((self.cell_size, self.seed)); }
                if cancel { result = DialogResult::Cancel; }
            });
        result
    }
}

// -------

effect_dialog_base!(DentsDialog {
    scale: f32 = 8.0, amount: f32 = 0.0, seed: u32 = 42,
    octaves: f32 = 1.0, roughness: f32 = 0.5,
    pinch: bool = false, wrap: bool = false,
    first_open: bool = true
});

impl DentsDialog {
    pub fn show(&mut self, ctx: &egui::Context) -> DialogResult<(f32, f32, u32, u32, f32, bool, bool)> {
        let mut result = DialogResult::Open;
        let colors = DialogColors::from_ctx(ctx);

        egui::Window::new("dialog_dents")
            .title_bar(false).collapsible(false).resizable(false)
            .default_pos(egui::pos2(ctx.screen_rect().center().x - 175.0, 60.0))
            .show(ctx, |ui| {
                ui.set_min_width(370.0);
                paint_dialog_header(ui, &colors, "\u{1F30A}", &t!("dialog.dents"));
                ui.add_space(4.0);
                section_label(ui, &colors, "DISTORTION SETTINGS");

                let mut changed = false;
                egui::Grid::new("dents_params").num_columns(2).spacing([8.0, 6.0]).show(ui, |ui| {
                    ui.label("Scale");
                    let r = ui.add(egui::Slider::new(&mut self.scale, 1.0..=80.0).max_decimals(1)); if track_slider(&r, &mut self.dragging) { changed = true; }
                    ui.end_row();

                    ui.label("Amount");
                    let r = ui.add(egui::Slider::new(&mut self.amount, 0.5..=30.0).max_decimals(1)); if track_slider(&r, &mut self.dragging) { changed = true; }
                    ui.end_row();

                    ui.label("Seed");
                    ui.horizontal(|ui| {
                        let mut seed_f = self.seed as f32;
                        if ui.add(egui::DragValue::new(&mut seed_f).speed(1.0).clamp_range(0.0..=9999.0)).changed() {
                            self.seed = seed_f as u32;
                            changed = true;
                        }
                        if ui.small_button("\u{1F3B2}").clicked() {
                            self.seed = (self.seed.wrapping_mul(1103515245).wrapping_add(12345)) % 10000;
                            changed = true;
                        }
                    });
                    ui.end_row();
                });

                ui.add_space(4.0);
                section_label(ui, &colors, "TURBULENCE");

                egui::Grid::new("dents_turb").num_columns(2).spacing([8.0, 6.0]).show(ui, |ui| {
                    ui.label("Octaves");
                    let r = ui.add(egui::Slider::new(&mut self.octaves, 1.0..=8.0).max_decimals(0).integer()); if track_slider(&r, &mut self.dragging) { changed = true; }
                    ui.end_row();

                    ui.label("Roughness");
                    let r = ui.add(egui::Slider::new(&mut self.roughness, 0.1..=1.0).max_decimals(2)); if track_slider(&r, &mut self.dragging) { changed = true; }
                    ui.end_row();
                });

                ui.add_space(4.0);
                section_label(ui, &colors, "OPTIONS");

                egui::Grid::new("dents_opts").num_columns(2).spacing([8.0, 6.0]).show(ui, |ui| {
                    ui.label("Pinch");
                    if ui.checkbox(&mut self.pinch, "Inward bias").changed() { changed = true; }
                    ui.end_row();

                    ui.label("Wrap");
                    if ui.checkbox(&mut self.wrap, "Tile edges").changed() { changed = true; }
                    ui.end_row();
                });

                accent_separator(ui, &colors);
                let manual = preview_controls(ui, &colors, &mut self.live_preview);
                if (changed && self.live_preview) || manual { result = DialogResult::Changed; }

                let (ok, cancel) = dialog_footer(ui, &colors);
                if ok { result = DialogResult::Ok((self.scale, self.amount, self.seed, self.octaves as u32, self.roughness, self.pinch, self.wrap)); }
                if cancel { result = DialogResult::Cancel; }
            });
        result
    }
}

// -------

effect_dialog_base!(PixelateDialog { block_size: f32 = 1.0, first_open: bool = true });

impl PixelateDialog {
    pub fn show(&mut self, ctx: &egui::Context) -> DialogResult<u32> {
        let mut result = DialogResult::Open;
        let colors = DialogColors::from_ctx(ctx);

        egui::Window::new("dialog_pixelate")
            .title_bar(false).collapsible(false).resizable(false)
            .default_pos(egui::pos2(ctx.screen_rect().center().x - 175.0, 60.0))
            .show(ctx, |ui| {
                ui.set_min_width(350.0);
                paint_dialog_header(ui, &colors, "\u{1F9E9}", &t!("dialog.pixelate"));
                ui.add_space(4.0);
                section_label(ui, &colors, "PIXEL SETTINGS");

                let mut changed = false;
                egui::Grid::new("pixelate_params").num_columns(2).spacing([8.0, 6.0]).show(ui, |ui| {
                    ui.label("Block Size");
                    if numeric_field_with_buttons(ui, &mut self.block_size, 0.5, 2.0..=64.0, " px", 1.0) {
                        changed = true;
                    }
                    ui.end_row();

                    ui.label("Quick");
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 4.0;
                        for &(label, val) in &[("Small", 4.0), ("Medium", 8.0), ("Large", 16.0), ("Huge", 32.0)] {
                            let btn = if (self.block_size - val).abs() < 1.0 {
                                egui::Button::new(egui::RichText::new(label).strong().size(11.0)).fill(colors.accent_faint)
                            } else {
                                egui::Button::new(egui::RichText::new(label).size(11.0))
                            };
                            if ui.add(btn).clicked() { self.block_size = val; changed = true; }
                        }
                    });
                    ui.end_row();
                });

                accent_separator(ui, &colors);
                let manual = preview_controls(ui, &colors, &mut self.live_preview);
                if (changed && self.live_preview) || manual { result = DialogResult::Changed; }

                let (ok, cancel) = dialog_footer(ui, &colors);
                if ok { result = DialogResult::Ok(self.block_size as u32); }
                if cancel { result = DialogResult::Cancel; }
            });
        result
    }
}

// -------

effect_dialog_base!(BulgeDialog { amount: f32 = 0.0, first_open: bool = true });

impl BulgeDialog {
    pub fn show(&mut self, ctx: &egui::Context) -> DialogResult<f32> {
        let mut result = DialogResult::Open;
        let colors = DialogColors::from_ctx(ctx);

        egui::Window::new("dialog_bulge")
            .title_bar(false).collapsible(false).resizable(false)
            .default_pos(egui::pos2(ctx.screen_rect().center().x - 175.0, 60.0))
            .show(ctx, |ui| {
                ui.set_min_width(350.0);
                paint_dialog_header(ui, &colors, "\u{1F534}", &t!("dialog.bulge_pinch"));
                ui.add_space(4.0);
                section_label(ui, &colors, "DISTORTION SETTINGS");

                let mut changed = false;
                egui::Grid::new("bulge_params").num_columns(2).spacing([8.0, 6.0]).show(ui, |ui| {
                    ui.label("Amount");
                    let r = ui.add(egui::Slider::new(&mut self.amount, -3.0..=3.0).max_decimals(2)); if track_slider(&r, &mut self.dragging) {
                        changed = true;
                    }
                    ui.end_row();
                    ui.label("");
                    ui.label(egui::RichText::new("Positive = bulge, Negative = pinch").size(10.0).color(colors.text_muted));
                    ui.end_row();
                });

                accent_separator(ui, &colors);
                let manual = preview_controls(ui, &colors, &mut self.live_preview);
                if (changed && self.live_preview) || manual { result = DialogResult::Changed; }

                let (ok, cancel) = dialog_footer(ui, &colors);
                if ok { result = DialogResult::Ok(self.amount); }
                if cancel { result = DialogResult::Cancel; }
            });
        result
    }
}

// -------

effect_dialog_base!(TwistDialog { angle: f32 = 0.0, first_open: bool = true });

impl TwistDialog {
    pub fn show(&mut self, ctx: &egui::Context) -> DialogResult<f32> {
        let mut result = DialogResult::Open;
        let colors = DialogColors::from_ctx(ctx);

        egui::Window::new("dialog_twist")
            .title_bar(false).collapsible(false).resizable(false)
            .default_pos(egui::pos2(ctx.screen_rect().center().x - 175.0, 60.0))
            .show(ctx, |ui| {
                ui.set_min_width(350.0);
                paint_dialog_header(ui, &colors, "\u{1F300}", &t!("dialog.twist"));
                ui.add_space(4.0);
                section_label(ui, &colors, "TWIST SETTINGS");

                let mut changed = false;
                egui::Grid::new("twist_params").num_columns(2).spacing([8.0, 6.0]).show(ui, |ui| {
                    ui.label("Angle");
                    let r = ui.add(egui::Slider::new(&mut self.angle, -720.0..=720.0).suffix("Â°").max_decimals(0)); if track_slider(&r, &mut self.dragging) {
                        changed = true;
                    }
                    ui.end_row();

                    ui.label("Quick");
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 4.0;
                        for &(label, val) in &[("Mild", 45.0), ("Half", 180.0), ("Full", 360.0), ("Double", 720.0)] {
                            let btn = if (self.angle - val).abs() < 5.0 {
                                egui::Button::new(egui::RichText::new(label).strong().size(11.0)).fill(colors.accent_faint)
                            } else {
                                egui::Button::new(egui::RichText::new(label).size(11.0))
                            };
                            if ui.add(btn).clicked() { self.angle = val; changed = true; }
                        }
                    });
                    ui.end_row();
                });

                accent_separator(ui, &colors);
                let manual = preview_controls(ui, &colors, &mut self.live_preview);
                if (changed && self.live_preview) || manual { result = DialogResult::Changed; }

                let (ok, cancel) = dialog_footer(ui, &colors);
                if ok { result = DialogResult::Ok(self.angle); }
                if cancel { result = DialogResult::Cancel; }
            });
        result
    }
}

// ============================================================================
// NOISE DIALOGS
// ============================================================================

effect_dialog_base!(AddNoiseDialog {
    amount: f32 = 0.0,
    noise_type_idx: usize = 0,
    monochrome: bool = false,
    seed: u32 = 42,
    scale: f32 = 1.0,
    octaves: f32 = 1.0,
    first_open: bool = true
});

impl AddNoiseDialog {
    pub fn noise_type(&self) -> NoiseType {
        match self.noise_type_idx {
            0 => NoiseType::Uniform,
            1 => NoiseType::Gaussian,
            _ => NoiseType::Perlin,
        }
    }

    pub fn show(&mut self, ctx: &egui::Context) -> DialogResult<(f32, NoiseType, bool, u32, f32, u32)> {
        let mut result = DialogResult::Open;
        let colors = DialogColors::from_ctx(ctx);

        egui::Window::new("dialog_add_noise")
            .title_bar(false).collapsible(false).resizable(false)
            .default_pos(egui::pos2(ctx.screen_rect().center().x - 185.0, 60.0))
            .show(ctx, |ui| {
                ui.set_min_width(380.0);
                paint_dialog_header(ui, &colors, "\u{1F4A5}", &t!("dialog.add_noise"));
                ui.add_space(4.0);
                section_label(ui, &colors, "NOISE SETTINGS");

                let mut changed = false;
                egui::Grid::new("noise_params").num_columns(2).spacing([8.0, 6.0]).show(ui, |ui| {
                    ui.label("Amount");
                    let r = ui.add(egui::Slider::new(&mut self.amount, 1.0..=100.0).suffix("%").max_decimals(0)); if track_slider(&r, &mut self.dragging) {
                        changed = true;
                    }
                    ui.end_row();

                    ui.label("Type");
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 4.0;
                        let types = ["Uniform", "Gaussian", "Perlin"];
                        for (i, label) in types.iter().enumerate() {
                            let btn = if self.noise_type_idx == i {
                                egui::Button::new(egui::RichText::new(*label).strong().size(11.0)).fill(colors.accent_faint)
                            } else {
                                egui::Button::new(egui::RichText::new(*label).size(11.0))
                            };
                            if ui.add(btn).clicked() { self.noise_type_idx = i; changed = true; }
                        }
                    });
                    ui.end_row();

                    ui.label("Color");
                    if ui.checkbox(&mut self.monochrome, "Monochrome").changed() { changed = true; }
                    ui.end_row();

                    ui.label("Seed");
                    ui.horizontal(|ui| {
                        let mut seed_f = self.seed as f32;
                        if ui.add(egui::DragValue::new(&mut seed_f).speed(1.0).clamp_range(0.0..=9999.0)).changed() {
                            self.seed = seed_f as u32; changed = true;
                        }
                        if ui.small_button("\u{1F3B2}").clicked() {
                            self.seed = (self.seed.wrapping_mul(1103515245).wrapping_add(12345)) % 10000;
                            changed = true;
                        }
                    });
                    ui.end_row();
                });

                ui.add_space(4.0);
                section_label(ui, &colors, "GRAIN CONTROL");

                egui::Grid::new("noise_grain").num_columns(2).spacing([8.0, 6.0]).show(ui, |ui| {
                    ui.label("Scale");
                    let r = ui.add(egui::Slider::new(&mut self.scale, 0.5..=100.0).suffix(" px").max_decimals(1)
                              .logarithmic(true).clamp_to_range(false)); if track_slider(&r, &mut self.dragging) {
                        changed = true;
                    }
                    ui.end_row();
                    ui.label("");
                    ui.label(egui::RichText::new("Size of each noise grain (1 = single pixel)").size(10.0).color(colors.text_muted));
                    ui.end_row();

                    // Octaves only for Perlin
                    if self.noise_type_idx == 2 {
                        ui.label("Octaves");
                        let r = ui.add(egui::Slider::new(&mut self.octaves, 1.0..=8.0).max_decimals(0).integer()); if track_slider(&r, &mut self.dragging) {
                            changed = true;
                        }
                        ui.end_row();
                        ui.label("");
                        ui.label(egui::RichText::new("More octaves = finer detail layers").size(10.0).color(colors.text_muted));
                        ui.end_row();
                    }
                });

                accent_separator(ui, &colors);
                let manual = preview_controls(ui, &colors, &mut self.live_preview);
                if (changed && self.live_preview) || manual { result = DialogResult::Changed; }

                let (ok, cancel) = dialog_footer(ui, &colors);
                if ok { result = DialogResult::Ok((self.amount, self.noise_type(), self.monochrome, self.seed, self.scale, self.octaves as u32)); }
                if cancel { result = DialogResult::Cancel; }
            });
        result
    }
}

// -------

effect_dialog_base!(ReduceNoiseDialog { strength: f32 = 0.0, radius: f32 = 3.0, first_open: bool = true });

impl ReduceNoiseDialog {
    pub fn show(&mut self, ctx: &egui::Context) -> DialogResult<(f32, u32)> {
        let mut result = DialogResult::Open;
        let colors = DialogColors::from_ctx(ctx);

        egui::Window::new("dialog_reduce_noise")
            .title_bar(false).collapsible(false).resizable(false)
            .default_pos(egui::pos2(ctx.screen_rect().center().x - 175.0, 60.0))
            .show(ctx, |ui| {
                ui.set_min_width(350.0);
                paint_dialog_header(ui, &colors, "\u{1F50A}", &t!("dialog.reduce_noise"));
                ui.add_space(4.0);
                section_label(ui, &colors, "DENOISE SETTINGS");

                let mut changed = false;
                egui::Grid::new("denoise_params").num_columns(2).spacing([8.0, 6.0]).show(ui, |ui| {
                    ui.label("Strength");
                    let r = ui.add(egui::Slider::new(&mut self.strength, 5.0..=100.0).max_decimals(0)); if track_slider(&r, &mut self.dragging) { changed = true; }
                    ui.end_row();

                    ui.label("Radius");
                    if numeric_field_with_buttons(ui, &mut self.radius, 0.5, 1.0..=8.0, " px", 1.0) { changed = true; }
                    ui.end_row();
                });

                accent_separator(ui, &colors);
                let manual = preview_controls(ui, &colors, &mut self.live_preview);
                if (changed && self.live_preview) || manual { result = DialogResult::Changed; }

                let (ok, cancel) = dialog_footer(ui, &colors);
                if ok { result = DialogResult::Ok((self.strength, self.radius as u32)); }
                if cancel { result = DialogResult::Cancel; }
            });
        result
    }
}

// -------

effect_dialog_base!(MedianDialog { radius: f32 = 0.0 });

impl MedianDialog {
    pub fn show(&mut self, ctx: &egui::Context) -> DialogResult<u32> {
        let mut result = DialogResult::Open;
        let colors = DialogColors::from_ctx(ctx);

        egui::Window::new("dialog_median")
            .title_bar(false).collapsible(false).resizable(false)
            .default_pos(egui::pos2(ctx.screen_rect().center().x - 175.0, 60.0))
            .show(ctx, |ui| {
                ui.set_min_width(350.0);
                paint_dialog_header(ui, &colors, "\u{1F4CA}", &t!("dialog.median_filter"));
                ui.add_space(4.0);
                section_label(ui, &colors, "FILTER SETTINGS");

                let mut changed = false;
                egui::Grid::new("median_params").num_columns(2).spacing([8.0, 6.0]).show(ui, |ui| {
                    ui.label("Radius");
                    if numeric_field_with_buttons(ui, &mut self.radius, 0.5, 1.0..=8.0, " px", 1.0) { changed = true; }
                    ui.end_row();
                });

                accent_separator(ui, &colors);
                
                let manual = preview_controls(ui, &colors, &mut self.live_preview);
                if (changed && self.live_preview) || manual { result = DialogResult::Changed; }

                let (ok, cancel) = dialog_footer(ui, &colors);
                if ok { result = DialogResult::Ok(self.radius as u32); }
                if cancel { result = DialogResult::Cancel; }
            });
        result
    }
}

// ============================================================================
// STYLIZE DIALOGS
// ============================================================================

effect_dialog_base!(GlowDialog { radius: f32 = 0.0, intensity: f32 = 0.0, first_open: bool = true });

impl GlowDialog {
    pub fn show(&mut self, ctx: &egui::Context) -> DialogResult<(f32, f32)> {
        let mut result = DialogResult::Open;
        let colors = DialogColors::from_ctx(ctx);

        egui::Window::new("dialog_glow")
            .title_bar(false).collapsible(false).resizable(false)
            .default_pos(egui::pos2(ctx.screen_rect().center().x - 175.0, 60.0))
            .show(ctx, |ui| {
                ui.set_min_width(350.0);
                paint_dialog_header(ui, &colors, "\u{2728}", &t!("dialog.glow"));
                ui.add_space(4.0);
                section_label(ui, &colors, "GLOW SETTINGS");

                let mut changed = false;
                egui::Grid::new("glow_params").num_columns(2).spacing([8.0, 6.0]).show(ui, |ui| {
                    ui.label("Radius");
                    let r = ui.add(egui::Slider::new(&mut self.radius, 1.0..=30.0).max_decimals(1)); if track_slider(&r, &mut self.dragging) { changed = true; }
                    ui.end_row();

                    ui.label("Intensity");
                    let r = ui.add(egui::Slider::new(&mut self.intensity, 0.0..=2.0).max_decimals(2)); if track_slider(&r, &mut self.dragging) { changed = true; }
                    ui.end_row();
                });

                accent_separator(ui, &colors);
                let manual = preview_controls(ui, &colors, &mut self.live_preview);
                if (changed && self.live_preview) || manual { result = DialogResult::Changed; }

                let (ok, cancel) = dialog_footer(ui, &colors);
                if ok { result = DialogResult::Ok((self.radius, self.intensity)); }
                if cancel { result = DialogResult::Cancel; }
            });
        result
    }
}

// -------

effect_dialog_base!(SharpenDialog { amount: f32 = 0.0, radius: f32 = 1.5, first_open: bool = true });

impl SharpenDialog {
    pub fn show(&mut self, ctx: &egui::Context) -> DialogResult<(f32, f32)> {
        let mut result = DialogResult::Open;
        let colors = DialogColors::from_ctx(ctx);

        egui::Window::new("dialog_sharpen")
            .title_bar(false).collapsible(false).resizable(false)
            .default_pos(egui::pos2(ctx.screen_rect().center().x - 175.0, 60.0))
            .show(ctx, |ui| {
                ui.set_min_width(350.0);
                paint_dialog_header(ui, &colors, "\u{1F4CC}", &t!("dialog.sharpen"));
                ui.add_space(4.0);
                section_label(ui, &colors, "SHARPEN SETTINGS");

                let mut changed = false;
                egui::Grid::new("sharpen_params").num_columns(2).spacing([8.0, 6.0]).show(ui, |ui| {
                    ui.label("Amount");
                    let r = ui.add(egui::Slider::new(&mut self.amount, 0.1..=5.0).max_decimals(1)); if track_slider(&r, &mut self.dragging) { changed = true; }
                    ui.end_row();

                    ui.label("Radius");
                    let r = ui.add(egui::Slider::new(&mut self.radius, 0.5..=10.0).max_decimals(1)); if track_slider(&r, &mut self.dragging) { changed = true; }
                    ui.end_row();
                });

                accent_separator(ui, &colors);
                let manual = preview_controls(ui, &colors, &mut self.live_preview);
                if (changed && self.live_preview) || manual { result = DialogResult::Changed; }

                let (ok, cancel) = dialog_footer(ui, &colors);
                if ok { result = DialogResult::Ok((self.amount, self.radius)); }
                if cancel { result = DialogResult::Cancel; }
            });
        result
    }
}

// -------

effect_dialog_base!(VignetteDialog { amount: f32 = 0.0, softness: f32 = 0.6, first_open: bool = true });

impl VignetteDialog {
    pub fn show(&mut self, ctx: &egui::Context) -> DialogResult<(f32, f32)> {
        let mut result = DialogResult::Open;
        let colors = DialogColors::from_ctx(ctx);

        egui::Window::new("dialog_vignette")
            .title_bar(false).collapsible(false).resizable(false)
            .default_pos(egui::pos2(ctx.screen_rect().center().x - 175.0, 60.0))
            .show(ctx, |ui| {
                ui.set_min_width(350.0);
                paint_dialog_header(ui, &colors, "\u{1F311}", &t!("dialog.vignette"));
                ui.add_space(4.0);
                section_label(ui, &colors, "VIGNETTE SETTINGS");

                let mut changed = false;
                egui::Grid::new("vignette_params").num_columns(2).spacing([8.0, 6.0]).show(ui, |ui| {
                    ui.label("Amount");
                    let r = ui.add(egui::Slider::new(&mut self.amount, 0.0..=2.0).max_decimals(2)); if track_slider(&r, &mut self.dragging) { changed = true; }
                    ui.end_row();

                    ui.label("Softness");
                    let r = ui.add(egui::Slider::new(&mut self.softness, 0.1..=1.5).max_decimals(2)); if track_slider(&r, &mut self.dragging) { changed = true; }
                    ui.end_row();
                });

                accent_separator(ui, &colors);
                let manual = preview_controls(ui, &colors, &mut self.live_preview);
                if (changed && self.live_preview) || manual { result = DialogResult::Changed; }

                let (ok, cancel) = dialog_footer(ui, &colors);
                if ok { result = DialogResult::Ok((self.amount, self.softness)); }
                if cancel { result = DialogResult::Cancel; }
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

        egui::Window::new("dialog_halftone")
            .title_bar(false).collapsible(false).resizable(false)
            .default_pos(egui::pos2(ctx.screen_rect().center().x - 175.0, 60.0))
            .show(ctx, |ui| {
                ui.set_min_width(370.0);
                paint_dialog_header(ui, &colors, "\u{25CF}", &t!("dialog.halftone"));
                ui.add_space(4.0);
                section_label(ui, &colors, "HALFTONE SETTINGS");

                let mut changed = false;
                egui::Grid::new("halftone_params").num_columns(2).spacing([8.0, 6.0]).show(ui, |ui| {
                    ui.label("Dot Size");
                    let r = ui.add(egui::Slider::new(&mut self.dot_size, 2.0..=30.0).suffix(" px").max_decimals(0)); if track_slider(&r, &mut self.dragging) { changed = true; }
                    ui.end_row();

                    ui.label("Angle");
                    let r = ui.add(egui::Slider::new(&mut self.angle, -90.0..=90.0).suffix("Â°").max_decimals(0)); if track_slider(&r, &mut self.dragging) { changed = true; }
                    ui.end_row();

                    ui.label("Shape");
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 4.0;
                        let shapes = [("\u{25CF}", "Circle"), ("\u{25A0}", "Square"), ("\u{25C6}", "Diamond"), ("\u{2550}", "Line")];
                        for (i, (icon, label)) in shapes.iter().enumerate() {
                            let text = format!("{} {}", icon, label);
                            let btn = if self.shape_idx == i {
                                egui::Button::new(egui::RichText::new(&text).strong().size(11.0)).fill(colors.accent_faint)
                            } else {
                                egui::Button::new(egui::RichText::new(&text).size(11.0))
                            };
                            if ui.add(btn).clicked() { self.shape_idx = i; changed = true; }
                        }
                    });
                    ui.end_row();
                });

                accent_separator(ui, &colors);
                let manual = preview_controls(ui, &colors, &mut self.live_preview);
                if (changed && self.live_preview) || manual { result = DialogResult::Changed; }

                let (ok, cancel) = dialog_footer(ui, &colors);
                if ok { result = DialogResult::Ok((self.dot_size, self.angle, self.halftone_shape())); }
                if cancel { result = DialogResult::Cancel; }
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

impl GridDialog {
    pub fn grid_style(&self) -> GridStyle {
        match self.style_idx { 1 => GridStyle::Checkerboard, _ => GridStyle::Lines }
    }

    pub fn show(&mut self, ctx: &egui::Context) -> DialogResult<(u32, u32, u32, [u8; 4], GridStyle, f32)> {
        let mut result = DialogResult::Open;
        let colors = DialogColors::from_ctx(ctx);

        egui::Window::new("dialog_grid")
            .title_bar(false).collapsible(false).resizable(false)
            .default_pos(egui::pos2(ctx.screen_rect().center().x - 175.0, 60.0))
            .show(ctx, |ui| {
                ui.set_min_width(370.0);
                paint_dialog_header(ui, &colors, "\u{1F4D0}", &t!("dialog.grid"));
                ui.add_space(4.0);
                section_label(ui, &colors, "GRID SETTINGS");

                let mut changed = false;
                egui::Grid::new("grid_params").num_columns(2).spacing([8.0, 6.0]).show(ui, |ui| {
                    ui.label("Cell Width");
                    if numeric_field_with_buttons(ui, &mut self.cell_w, 1.0, 4.0..=256.0, " px", 4.0) { changed = true; }
                    ui.end_row();

                    ui.label("Cell Height");
                    if numeric_field_with_buttons(ui, &mut self.cell_h, 1.0, 4.0..=256.0, " px", 4.0) { changed = true; }
                    ui.end_row();

                    ui.label("Line Width");
                    if numeric_field_with_buttons(ui, &mut self.line_width, 0.5, 1.0..=8.0, " px", 1.0) { changed = true; }
                    ui.end_row();

                    ui.label("Color");
                    if ui.color_edit_button_rgb(&mut self.color).changed() { changed = true; }
                    ui.end_row();

                    ui.label("Opacity");
                    let r = ui.add(egui::Slider::new(&mut self.opacity, 0.0..=1.0).max_decimals(2)); if track_slider(&r, &mut self.dragging) { changed = true; }
                    ui.end_row();

                    ui.label("Style");
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 4.0;
                        for (i, label) in ["Lines", "Checkerboard"].iter().enumerate() {
                            let btn = if self.style_idx == i {
                                egui::Button::new(egui::RichText::new(*label).strong().size(11.0)).fill(colors.accent_faint)
                            } else {
                                egui::Button::new(egui::RichText::new(*label).size(11.0))
                            };
                            if ui.add(btn).clicked() { self.style_idx = i; changed = true; }
                        }
                    });
                    ui.end_row();
                });

                accent_separator(ui, &colors);
                let manual = preview_controls(ui, &colors, &mut self.live_preview);
                if (changed && self.live_preview) || manual { result = DialogResult::Changed; }

                let (ok, cancel) = dialog_footer(ui, &colors);
                if ok {
                    let c = [(self.color[0] * 255.0) as u8, (self.color[1] * 255.0) as u8, (self.color[2] * 255.0) as u8, 255];
                    result = DialogResult::Ok((self.cell_w as u32, self.cell_h as u32, self.line_width as u32, c, self.grid_style(), self.opacity));
                }
                if cancel { result = DialogResult::Cancel; }
            });
        result
    }
}

// -------

effect_dialog_base!(DropShadowDialog {
    offset_x: f32 = 0.0,
    offset_y: f32 = 0.0,
    blur_radius: f32 = 0.0,
    color: [f32; 3] = [0.0, 0.0, 0.0],
    opacity: f32 = 0.0,
    first_open: bool = true
});

impl DropShadowDialog {
    pub fn show(&mut self, ctx: &egui::Context) -> DialogResult<(i32, i32, f32, [u8; 4], f32)> {
        let mut result = DialogResult::Open;
        let colors = DialogColors::from_ctx(ctx);

        egui::Window::new("dialog_drop_shadow")
            .title_bar(false).collapsible(false).resizable(false)
            .default_pos(egui::pos2(ctx.screen_rect().center().x - 175.0, 60.0))
            .show(ctx, |ui| {
                ui.set_min_width(380.0);
                paint_dialog_header(ui, &colors, "\u{1F4A4}", &t!("dialog.drop_shadow"));
                ui.add_space(4.0);
                section_label(ui, &colors, "SHADOW SETTINGS");

                let mut changed = false;
                egui::Grid::new("shadow_params").num_columns(2).spacing([8.0, 6.0]).show(ui, |ui| {
                    ui.label("Offset X");
                    let r = ui.add(egui::Slider::new(&mut self.offset_x, -50.0..=50.0).suffix(" px").max_decimals(0)); if track_slider(&r, &mut self.dragging) { changed = true; }
                    ui.end_row();

                    ui.label("Offset Y");
                    let r = ui.add(egui::Slider::new(&mut self.offset_y, -50.0..=50.0).suffix(" px").max_decimals(0)); if track_slider(&r, &mut self.dragging) { changed = true; }
                    ui.end_row();

                    ui.label("Blur");
                    let r = ui.add(egui::Slider::new(&mut self.blur_radius, 0.0..=30.0).max_decimals(1)); if track_slider(&r, &mut self.dragging) { changed = true; }
                    ui.end_row();

                    ui.label("Color");
                    if ui.color_edit_button_rgb(&mut self.color).changed() { changed = true; }
                    ui.end_row();

                    ui.label("Opacity");
                    let r = ui.add(egui::Slider::new(&mut self.opacity, 0.0..=1.0).max_decimals(2)); if track_slider(&r, &mut self.dragging) { changed = true; }
                    ui.end_row();
                });

                accent_separator(ui, &colors);
                let manual = preview_controls(ui, &colors, &mut self.live_preview);
                if (changed && self.live_preview) || manual { result = DialogResult::Changed; }

                let (ok, cancel) = dialog_footer(ui, &colors);
                if ok {
                    let c = [(self.color[0] * 255.0) as u8, (self.color[1] * 255.0) as u8, (self.color[2] * 255.0) as u8, 255];
                    result = DialogResult::Ok((self.offset_x as i32, self.offset_y as i32, self.blur_radius, c, self.opacity));
                }
                if cancel { result = DialogResult::Cancel; }
            });
        result
    }
}

// -------

effect_dialog_base!(OutlineDialog {
    width: f32 = 0.0,
    color: [f32; 3] = [0.0, 0.0, 0.0],
    mode_idx: usize = 0,
    first_open: bool = true
});

impl OutlineDialog {
    pub fn outline_mode(&self) -> OutlineMode {
        match self.mode_idx { 1 => OutlineMode::Inside, 2 => OutlineMode::Center, _ => OutlineMode::Outside }
    }

    pub fn show(&mut self, ctx: &egui::Context) -> DialogResult<(u32, [u8; 4], OutlineMode)> {
        let mut result = DialogResult::Open;
        let colors = DialogColors::from_ctx(ctx);

        egui::Window::new("dialog_outline")
            .title_bar(false).collapsible(false).resizable(false)
            .default_pos(egui::pos2(ctx.screen_rect().center().x - 175.0, 60.0))
            .show(ctx, |ui| {
                ui.set_min_width(360.0);
                paint_dialog_header(ui, &colors, "\u{1F58A}", &t!("dialog.outline"));
                ui.add_space(4.0);
                section_label(ui, &colors, "OUTLINE SETTINGS");

                let mut changed = false;
                egui::Grid::new("outline_params").num_columns(2).spacing([8.0, 6.0]).show(ui, |ui| {
                    ui.label("Width");
                    if numeric_field_with_buttons(ui, &mut self.width, 0.5, 1.0..=20.0, " px", 1.0) { changed = true; }
                    ui.end_row();

                    ui.label("Color");
                    if ui.color_edit_button_rgb(&mut self.color).changed() { changed = true; }
                    ui.end_row();

                    ui.label("Mode");
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 4.0;
                        for (i, label) in ["Outside", "Inside", "Center"].iter().enumerate() {
                            let btn = if self.mode_idx == i {
                                egui::Button::new(egui::RichText::new(*label).strong().size(11.0)).fill(colors.accent_faint)
                            } else {
                                egui::Button::new(egui::RichText::new(*label).size(11.0))
                            };
                            if ui.add(btn).clicked() { self.mode_idx = i; changed = true; }
                        }
                    });
                    ui.end_row();
                });

                accent_separator(ui, &colors);
                let manual = preview_controls(ui, &colors, &mut self.live_preview);
                if (changed && self.live_preview) || manual { result = DialogResult::Changed; }

                let (ok, cancel) = dialog_footer(ui, &colors);
                if ok {
                    let c = [(self.color[0] * 255.0) as u8, (self.color[1] * 255.0) as u8, (self.color[2] * 255.0) as u8, 255];
                    result = DialogResult::Ok((self.width as u32, c, self.outline_mode()));
                }
                if cancel { result = DialogResult::Cancel; }
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
            .title_bar(false).collapsible(false).resizable(false)
            .default_pos(egui::pos2(ctx.screen_rect().center().x - 175.0, 60.0))
            .show(ctx, |ui| {
                ui.set_min_width(370.0);
                paint_dialog_header(ui, &colors, "\u{1F4A2}", &t!("dialog.pixel_drag"));
                ui.add_space(4.0);
                section_label(ui, &colors, "GLITCH SETTINGS");

                let mut changed = false;
                egui::Grid::new("pixdrag_params").num_columns(2).spacing([8.0, 6.0]).show(ui, |ui| {
                    ui.label("Amount");
                    let r = ui.add(egui::Slider::new(&mut self.amount, 1.0..=100.0).suffix("%").max_decimals(0)); if track_slider(&r, &mut self.dragging) { changed = true; }
                    ui.end_row();

                    ui.label("Distance");
                    let r = ui.add(egui::Slider::new(&mut self.distance, 5.0..=200.0).suffix(" px").max_decimals(0)); if track_slider(&r, &mut self.dragging) { changed = true; }
                    ui.end_row();

                    ui.label("Direction");
                    let r = ui.add(egui::Slider::new(&mut self.direction, -180.0..=180.0).suffix("Â°").max_decimals(0)); if track_slider(&r, &mut self.dragging) { changed = true; }
                    ui.end_row();

                    ui.label("Seed");
                    ui.horizontal(|ui| {
                        let mut seed_f = self.seed as f32;
                        if ui.add(egui::DragValue::new(&mut seed_f).speed(1.0).clamp_range(0.0..=9999.0)).changed() {
                            self.seed = seed_f as u32; changed = true;
                        }
                        if ui.small_button("\u{1F3B2}").clicked() {
                            self.seed = (self.seed.wrapping_mul(1103515245).wrapping_add(12345)) % 10000;
                            changed = true;
                        }
                    });
                    ui.end_row();
                });

                accent_separator(ui, &colors);
                let manual = preview_controls(ui, &colors, &mut self.live_preview);
                if (changed && self.live_preview) || manual { result = DialogResult::Changed; }

                let (ok, cancel) = dialog_footer(ui, &colors);
                if ok { result = DialogResult::Ok((self.seed, self.amount, self.distance as u32, self.direction)); }
                if cancel { result = DialogResult::Cancel; }
            });
        result
    }
}

// -------

effect_dialog_base!(RgbDisplaceDialog {
    r_x: f32 = 0.0, r_y: f32 = 0.0,
    g_x: f32 = 0.0, g_y: f32 = 0.0,
    b_x: f32 = 0.0, b_y: f32 = 0.0,
    first_open: bool = true
});

impl RgbDisplaceDialog {
    pub fn show(&mut self, ctx: &egui::Context) -> DialogResult<((i32, i32), (i32, i32), (i32, i32))> {
        let mut result = DialogResult::Open;
        let colors = DialogColors::from_ctx(ctx);

        egui::Window::new("dialog_rgb_displace")
            .title_bar(false).collapsible(false).resizable(false)
            .default_pos(egui::pos2(ctx.screen_rect().center().x - 200.0, 60.0))
            .show(ctx, |ui| {
                ui.set_min_width(420.0);
                paint_dialog_header(ui, &colors, "\u{1F308}", &t!("dialog.rgb_displace"));
                ui.add_space(4.0);
                section_label(ui, &colors, "CHANNEL OFFSETS");

                let mut changed = false;

                // Red channel
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("Red").color(Color32::from_rgb(220, 50, 50)).strong().size(12.0));
                });
                egui::Grid::new("rgb_r").num_columns(2).spacing([8.0, 4.0]).show(ui, |ui| {
                    ui.label("  X");
                    let r = ui.add(egui::Slider::new(&mut self.r_x, -100.0..=100.0).suffix(" px").max_decimals(0)); if track_slider(&r, &mut self.dragging) { changed = true; }
                    ui.end_row();
                    ui.label("  Y");
                    let r = ui.add(egui::Slider::new(&mut self.r_y, -100.0..=100.0).suffix(" px").max_decimals(0)); if track_slider(&r, &mut self.dragging) { changed = true; }
                    ui.end_row();
                });

                ui.add_space(2.0);

                // Green channel
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("Green").color(Color32::from_rgb(50, 180, 50)).strong().size(12.0));
                });
                egui::Grid::new("rgb_g").num_columns(2).spacing([8.0, 4.0]).show(ui, |ui| {
                    ui.label("  X");
                    let r = ui.add(egui::Slider::new(&mut self.g_x, -100.0..=100.0).suffix(" px").max_decimals(0)); if track_slider(&r, &mut self.dragging) { changed = true; }
                    ui.end_row();
                    ui.label("  Y");
                    let r = ui.add(egui::Slider::new(&mut self.g_y, -100.0..=100.0).suffix(" px").max_decimals(0)); if track_slider(&r, &mut self.dragging) { changed = true; }
                    ui.end_row();
                });

                ui.add_space(2.0);

                // Blue channel
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("Blue").color(Color32::from_rgb(50, 100, 220)).strong().size(12.0));
                });
                egui::Grid::new("rgb_b").num_columns(2).spacing([8.0, 4.0]).show(ui, |ui| {
                    ui.label("  X");
                    let r = ui.add(egui::Slider::new(&mut self.b_x, -100.0..=100.0).suffix(" px").max_decimals(0)); if track_slider(&r, &mut self.dragging) { changed = true; }
                    ui.end_row();
                    ui.label("  Y");
                    let r = ui.add(egui::Slider::new(&mut self.b_y, -100.0..=100.0).suffix(" px").max_decimals(0)); if track_slider(&r, &mut self.dragging) { changed = true; }
                    ui.end_row();
                });

                ui.add_space(4.0);
                section_label(ui, &colors, "PRESETS");
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 4.0;
                    if ui.button(egui::RichText::new("Subtle").size(11.0)).clicked() {
                        self.r_x = 2.0; self.r_y = 0.0; self.g_x = -2.0; self.g_y = 0.0; self.b_x = 0.0; self.b_y = 2.0;
                        changed = true;
                    }
                    if ui.button(egui::RichText::new("VHS").size(11.0)).clicked() {
                        self.r_x = 8.0; self.r_y = 1.0; self.g_x = -4.0; self.g_y = -1.0; self.b_x = 0.0; self.b_y = -3.0;
                        changed = true;
                    }
                    if ui.button(egui::RichText::new("Heavy").size(11.0)).clicked() {
                        self.r_x = 20.0; self.r_y = -5.0; self.g_x = -20.0; self.g_y = 5.0; self.b_x = 5.0; self.b_y = 20.0;
                        changed = true;
                    }
                    if ui.button(egui::RichText::new("Reset").size(11.0)).clicked() {
                        self.r_x = 0.0; self.r_y = 0.0; self.g_x = 0.0; self.g_y = 0.0; self.b_x = 0.0; self.b_y = 0.0;
                        changed = true;
                    }
                });

                accent_separator(ui, &colors);
                let manual = preview_controls(ui, &colors, &mut self.live_preview);
                if (changed && self.live_preview) || manual { result = DialogResult::Changed; }

                let (ok, cancel) = dialog_footer(ui, &colors);
                if ok {
                    result = DialogResult::Ok((
                        (self.r_x as i32, self.r_y as i32),
                        (self.g_x as i32, self.g_y as i32),
                        (self.b_x as i32, self.b_y as i32),
                    ));
                }
                if cancel { result = DialogResult::Cancel; }
            });
        result
    }
}

// ============================================================================
// ARTISTIC DIALOGS
// ============================================================================

effect_dialog_base!(InkDialog { edge_strength: f32 = 0.0, threshold: f32 = 0.3, first_open: bool = true });

impl InkDialog {
    pub fn show(&mut self, ctx: &egui::Context) -> DialogResult<(f32, f32)> {
        let mut result = DialogResult::Open;
        let colors = DialogColors::from_ctx(ctx);

        egui::Window::new("dialog_ink")
            .title_bar(false).collapsible(false).resizable(false)
            .default_pos(egui::pos2(ctx.screen_rect().center().x - 175.0, 60.0))
            .show(ctx, |ui| {
                ui.set_min_width(350.0);
                paint_dialog_header(ui, &colors, "\u{1F58B}", &t!("dialog.ink"));
                ui.add_space(4.0);
                section_label(ui, &colors, "INK SETTINGS");

                let mut changed = false;
                egui::Grid::new("ink_params").num_columns(2).spacing([8.0, 6.0]).show(ui, |ui| {
                    ui.label("Edge Strength");
                    let r = ui.add(egui::Slider::new(&mut self.edge_strength, 10.0..=300.0).max_decimals(0)); if track_slider(&r, &mut self.dragging) { changed = true; }
                    ui.end_row();

                    ui.label("Threshold");
                    let r = ui.add(egui::Slider::new(&mut self.threshold, 0.05..=1.0).max_decimals(2)); if track_slider(&r, &mut self.dragging) { changed = true; }
                    ui.end_row();
                });

                accent_separator(ui, &colors);
                let manual = preview_controls(ui, &colors, &mut self.live_preview);
                if (changed && self.live_preview) || manual { result = DialogResult::Changed; }

                let (ok, cancel) = dialog_footer(ui, &colors);
                if ok { result = DialogResult::Ok((self.edge_strength, self.threshold)); }
                if cancel { result = DialogResult::Cancel; }
            });
        result
    }
}

// -------

effect_dialog_base!(OilPaintingDialog { radius: f32 = 0.0, levels: f32 = 20.0, first_open: bool = true });

impl OilPaintingDialog {
    pub fn show(&mut self, ctx: &egui::Context) -> DialogResult<(u32, u32)> {
        let mut result = DialogResult::Open;
        let colors = DialogColors::from_ctx(ctx);

        egui::Window::new("dialog_oil_painting")
            .title_bar(false).collapsible(false).resizable(false)
            .default_pos(egui::pos2(ctx.screen_rect().center().x - 175.0, 60.0))
            .show(ctx, |ui| {
                ui.set_min_width(360.0);
                paint_dialog_header(ui, &colors, "\u{1F3A8}", &t!("dialog.oil_painting"));
                ui.add_space(4.0);
                section_label(ui, &colors, "PAINTING SETTINGS");

                let mut changed = false;
                egui::Grid::new("oil_params").num_columns(2).spacing([8.0, 6.0]).show(ui, |ui| {
                    ui.label("Brush Radius");
                    if numeric_field_with_buttons(ui, &mut self.radius, 0.5, 1.0..=10.0, " px", 1.0) { changed = true; }
                    ui.end_row();

                    ui.label("Intensity Levels");
                    let r = ui.add(egui::Slider::new(&mut self.levels, 4.0..=64.0).max_decimals(0)); if track_slider(&r, &mut self.dragging) { changed = true; }
                    ui.end_row();
                });

                accent_separator(ui, &colors);
                let manual = preview_controls(ui, &colors, &mut self.live_preview);
                if (changed && self.live_preview) || manual { result = DialogResult::Changed; }

                let (ok, cancel) = dialog_footer(ui, &colors);
                if ok { result = DialogResult::Ok((self.radius as u32, self.levels as u32)); }
                if cancel { result = DialogResult::Cancel; }
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

        egui::Window::new("dialog_color_filter")
            .title_bar(false).collapsible(false).resizable(false)
            .default_pos(egui::pos2(ctx.screen_rect().center().x - 175.0, 60.0))
            .show(ctx, |ui| {
                ui.set_min_width(380.0);
                paint_dialog_header(ui, &colors, "\u{1F3AD}", &t!("dialog.color_filter"));
                ui.add_space(4.0);
                section_label(ui, &colors, "FILTER SETTINGS");

                let mut changed = false;
                egui::Grid::new("cfilter_params").num_columns(2).spacing([8.0, 6.0]).show(ui, |ui| {
                    ui.label("Color");
                    if ui.color_edit_button_rgb(&mut self.color).changed() { changed = true; }
                    ui.end_row();

                    ui.label("Intensity");
                    let r = ui.add(egui::Slider::new(&mut self.intensity, 0.0..=1.0).max_decimals(2)); if track_slider(&r, &mut self.dragging) { changed = true; }
                    ui.end_row();

                    ui.label("Quick");
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 3.0;
                        let presets: [(&str, [f32;3]); 5] = [
                            ("Warm", [1.0, 0.85, 0.6]),
                            ("Cool", [0.6, 0.8, 1.0]),
                            ("Sepia", [0.94, 0.82, 0.63]),
                            ("Rose", [1.0, 0.7, 0.75]),
                            ("Cyan", [0.5, 0.95, 0.95]),
                        ];
                        for (label, c) in &presets {
                            let preview_col = Color32::from_rgb((c[0]*255.0) as u8, (c[1]*255.0) as u8, (c[2]*255.0) as u8);
                            let btn = egui::Button::new(egui::RichText::new(*label).size(10.5).color(
                                if c[0] > 0.8 && c[1] > 0.8 && c[2] > 0.8 { Color32::BLACK } else { Color32::WHITE }
                            )).fill(preview_col);
                            if ui.add(btn).clicked() { self.color = *c; changed = true; }
                        }
                    });
                    ui.end_row();

                    ui.label("Blend Mode");
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 4.0;
                        for (i, label) in ["Multiply", "Screen", "Overlay", "Soft Light"].iter().enumerate() {
                            let btn = if self.mode_idx == i {
                                egui::Button::new(egui::RichText::new(*label).strong().size(11.0)).fill(colors.accent_faint)
                            } else {
                                egui::Button::new(egui::RichText::new(*label).size(11.0))
                            };
                            if ui.add(btn).clicked() { self.mode_idx = i; changed = true; }
                        }
                    });
                    ui.end_row();
                });

                accent_separator(ui, &colors);
                let manual = preview_controls(ui, &colors, &mut self.live_preview);
                if (changed && self.live_preview) || manual { result = DialogResult::Changed; }

                let (ok, cancel) = dialog_footer(ui, &colors);
                if ok {
                    let c = [(self.color[0] * 255.0) as u8, (self.color[1] * 255.0) as u8, (self.color[2] * 255.0) as u8, 255];
                    result = DialogResult::Ok((c, self.intensity, self.filter_mode()));
                }
                if cancel { result = DialogResult::Cancel; }
            });
        result
    }
}

// ============================================================================
// RENDER â€” CONTOURS DIALOG
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
    pub fn show(&mut self, ctx: &egui::Context) -> DialogResult<(f32, f32, f32, [u8; 4], u32, u32, f32)> {
        let mut result = DialogResult::Open;
        let colors = DialogColors::from_ctx(ctx);

        egui::Window::new("dialog_contours")
            .title_bar(false).collapsible(false).resizable(false)
            .default_pos(egui::pos2(ctx.screen_rect().center().x - 190.0, 60.0))
            .show(ctx, |ui| {
                ui.set_min_width(400.0);
                paint_dialog_header(ui, &colors, "\u{1F5FA}", &t!("dialog.contours"));
                ui.add_space(4.0);
                section_label(ui, &colors, "CONTOUR SETTINGS");

                let mut changed = false;
                egui::Grid::new("contour_params").num_columns(2).spacing([8.0, 6.0]).show(ui, |ui| {
                    ui.label("Scale");
                    let r = ui.add(egui::Slider::new(&mut self.scale, 5.0..=400.0).suffix(" px").max_decimals(0)
                              .logarithmic(true).clamp_to_range(false)); if track_slider(&r, &mut self.dragging) { changed = true; }
                    ui.end_row();
                    ui.label("");
                    ui.label(egui::RichText::new("Size of the noise pattern").size(10.0).color(colors.text_muted));
                    ui.end_row();

                    ui.label("Frequency");
                    let r = ui.add(egui::Slider::new(&mut self.frequency, 1.0..=30.0).max_decimals(1)); if track_slider(&r, &mut self.dragging) { changed = true; }
                    ui.end_row();
                    ui.label("");
                    ui.label(egui::RichText::new("Number of contour levels").size(10.0).color(colors.text_muted));
                    ui.end_row();

                    ui.label("Line Width");
                    let r = ui.add(egui::Slider::new(&mut self.line_width, 0.5..=8.0).suffix(" px").max_decimals(1)); if track_slider(&r, &mut self.dragging) { changed = true; }
                    ui.end_row();

                    ui.label("Line Color");
                    ui.horizontal(|ui| {
                        let mut c32 = Color32::from_rgb(
                            (self.line_color[0] * 255.0) as u8,
                            (self.line_color[1] * 255.0) as u8,
                            (self.line_color[2] * 255.0) as u8,
                        );
                        if ui.color_edit_button_srgba(&mut c32).changed() {
                            self.line_color = [c32.r() as f32 / 255.0, c32.g() as f32 / 255.0, c32.b() as f32 / 255.0];
                            changed = true;
                        }
                        ui.spacing_mut().item_spacing.x = 3.0;
                        if ui.small_button("Black").clicked() { self.line_color = [0.0, 0.0, 0.0]; changed = true; }
                        if ui.small_button("White").clicked() { self.line_color = [1.0, 1.0, 1.0]; changed = true; }
                        if ui.small_button("Brown").clicked() { self.line_color = [0.55, 0.35, 0.17]; changed = true; }
                        if ui.small_button("Blue").clicked() { self.line_color = [0.15, 0.35, 0.7]; changed = true; }
                    });
                    ui.end_row();

                    ui.label("Blend");
                    let r = ui.add(egui::Slider::new(&mut self.blend, 0.0..=1.0).max_decimals(2)); if track_slider(&r, &mut self.dragging) { changed = true; }
                    ui.end_row();
                });

                ui.add_space(4.0);
                section_label(ui, &colors, "NOISE FIELD");

                egui::Grid::new("contour_noise").num_columns(2).spacing([8.0, 6.0]).show(ui, |ui| {
                    ui.label("Octaves");
                    let r = ui.add(egui::Slider::new(&mut self.octaves, 1.0..=6.0).max_decimals(0).integer()); if track_slider(&r, &mut self.dragging) { changed = true; }
                    ui.end_row();

                    ui.label("Seed");
                    ui.horizontal(|ui| {
                        let mut seed_f = self.seed as f32;
                        if ui.add(egui::DragValue::new(&mut seed_f).speed(1.0).clamp_range(0.0..=9999.0)).changed() {
                            self.seed = seed_f as u32; changed = true;
                        }
                        if ui.small_button("\u{1F3B2}").clicked() {
                            self.seed = (self.seed.wrapping_mul(1103515245).wrapping_add(12345)) % 10000;
                            changed = true;
                        }
                    });
                    ui.end_row();
                });

                ui.add_space(4.0);
                section_label(ui, &colors, "PRESETS");
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 4.0;
                    if ui.button(egui::RichText::new("Topo Map").size(11.0)).clicked() {
                        self.scale = 40.0; self.frequency = 10.0; self.line_width = 1.0;
                        self.line_color = [0.55, 0.35, 0.17]; self.octaves = 4.0; self.blend = 0.8;
                        changed = true;
                    }
                    if ui.button(egui::RichText::new("Fine Lines").size(11.0)).clicked() {
                        self.scale = 15.0; self.frequency = 20.0; self.line_width = 0.5;
                        self.line_color = [0.0, 0.0, 0.0]; self.octaves = 2.0; self.blend = 0.5;
                        changed = true;
                    }
                    if ui.button(egui::RichText::new("Bold").size(11.0)).clicked() {
                        self.scale = 60.0; self.frequency = 5.0; self.line_width = 4.0;
                        self.line_color = [0.0, 0.0, 0.0]; self.octaves = 3.0; self.blend = 1.0;
                        changed = true;
                    }
                    if ui.button(egui::RichText::new("Ocean").size(11.0)).clicked() {
                        self.scale = 50.0; self.frequency = 12.0; self.line_width = 1.5;
                        self.line_color = [0.15, 0.35, 0.7]; self.octaves = 5.0; self.blend = 0.7;
                        changed = true;
                    }
                });

                accent_separator(ui, &colors);
                let manual = preview_controls(ui, &colors, &mut self.live_preview);
                if (changed && self.live_preview) || manual { result = DialogResult::Changed; }

                let (ok, cancel) = dialog_footer(ui, &colors);
                if ok {
                    let c = [(self.line_color[0] * 255.0) as u8, (self.line_color[1] * 255.0) as u8,
                             (self.line_color[2] * 255.0) as u8, 255];
                    result = DialogResult::Ok((self.scale, self.frequency, self.line_width, c, self.seed, self.octaves as u32, self.blend));
                }
                if cancel { result = DialogResult::Cancel; }
            });
        result
    }
}

// ============================================================================
// REMOVE BACKGROUND DIALOG (AI / ONNX)
// ============================================================================

/// Settings dialog shown before running Remove Background.
/// This is NOT a preview dialog — it configures post-processing
/// parameters and then launches the background job.
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

        egui::Window::new("dialog_remove_bg")
            .title_bar(false).collapsible(false).resizable(false)
            .default_pos(egui::pos2(ctx.screen_rect().center().x - 190.0, 80.0))
            .show(ctx, |ui| {
                ui.set_min_width(380.0);
                paint_dialog_header(ui, &colors, "\u{2728}", &t!("dialog.remove_background"));
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
                            ui.add(egui::Slider::new(&mut self.threshold, 0.05..=0.95)
                                .step_by(0.01)
                                .fixed_decimals(2)
                                .text(""));
                        });
                        ui.end_row();

                        // Threshold help text
                        ui.label("");
                        ui.label(egui::RichText::new(
                            if self.threshold > 0.7 {
                                "Aggressive \u{2014} removes more of the background"
                            } else if self.threshold > 0.5 {
                                "Strict \u{2014} only keeps confident foreground"
                            } else if self.threshold > 0.3 {
                                "Balanced \u{2014} standard cutoff"
                            } else {
                                "Conservative \u{2014} keeps more of the subject"
                            }
                        ).size(9.5).color(colors.text_muted));
                        ui.end_row();

                        // Smooth edges
                        ui.label(egui::RichText::new("Smooth Edges").size(11.0));
                        ui.checkbox(&mut self.smooth_edges, "Soft alpha transitions");
                        ui.end_row();

                        // Fill holes
                        ui.label(egui::RichText::new("Fill Holes").size(11.0));
                        ui.horizontal(|ui| {
                            let mut fill_val = self.fill_holes as i32;
                            ui.add(egui::Slider::new(&mut fill_val, 0..=20)
                                .suffix(" px")
                                .text(""));
                            self.fill_holes = fill_val.max(0) as u32;
                        });
                        ui.end_row();

                        // Fill holes help text
                        ui.label("");
                        ui.label(egui::RichText::new(
                            if self.fill_holes > 0 {
                                "Fills gaps inside the subject (nose, teeth, etc.)"
                            } else {
                                "Off \u{2014} enable to fix holes in foreground"
                            }
                        ).size(9.5).color(colors.text_muted));
                        ui.end_row();

                        // Edge feather
                        ui.label(egui::RichText::new("Edge Feather").size(11.0));
                        ui.horizontal(|ui| {
                            ui.add(egui::Slider::new(&mut self.edge_feather, 0.0..=20.0)
                                .step_by(0.5)
                                .fixed_decimals(1)
                                .suffix(" px")
                                .text(""));
                        });
                        ui.end_row();

                        // Mask expansion
                        ui.label(egui::RichText::new("Mask Expansion").size(11.0));
                        ui.horizontal(|ui| {
                            ui.add(egui::Slider::new(&mut self.mask_expansion, -10..=10)
                                .suffix(" px")
                                .text(""));
                        });
                        ui.end_row();

                        // Expansion help text
                        ui.label("");
                        ui.label(egui::RichText::new(
                            if self.mask_expansion > 0 {
                                "Expand: grows foreground mask outward"
                            } else if self.mask_expansion < 0 {
                                "Contract: shrinks foreground mask inward"
                            } else {
                                "No change to mask boundary"
                            }
                        ).size(9.5).color(colors.text_muted));
                        ui.end_row();
                    });

                accent_separator(ui, &colors);

                section_label(ui, &colors, "PRESETS");
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 4.0;
                    if ui.button(egui::RichText::new("Default").size(11.0)).clicked() {
                        self.threshold = 0.5;
                        self.edge_feather = 0.0;
                        self.mask_expansion = 0;
                        self.smooth_edges = true;
                        self.fill_holes = 0;
                    }
                    if ui.button(egui::RichText::new("Portrait").size(11.0)).clicked() {
                        self.threshold = 0.5;
                        self.edge_feather = 1.0;
                        self.mask_expansion = 1;
                        self.smooth_edges = true;
                        self.fill_holes = 8;
                    }
                    if ui.button(egui::RichText::new("Conservative").size(11.0)).clicked() {
                        self.threshold = 0.3;
                        self.edge_feather = 1.0;
                        self.mask_expansion = 2;
                        self.smooth_edges = true;
                        self.fill_holes = 5;
                    }
                    if ui.button(egui::RichText::new("Aggressive").size(11.0)).clicked() {
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
                                egui::RichText::new("\u{25B6} Run").size(12.0).color(Color32::WHITE),
                            )
                            .fill(colors.accent)
                            .min_size(egui::vec2(80.0, 24.0))
                            .rounding(4.0),
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

                        if ui.add(
                            egui::Button::new(egui::RichText::new(t!("common.cancel")).size(12.0))
                                .min_size(egui::vec2(80.0, 24.0))
                                .rounding(4.0),
                        ).clicked() {
                            result = DialogResult::Cancel;
                        }
                    });
                });
                ui.add_space(4.0);
            });
        result
    }
}