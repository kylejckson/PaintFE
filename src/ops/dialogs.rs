// ============================================================================
// OPS DIALOG SYSTEM - modern modal dialogs for Image, Layer, and Effects
// ============================================================================
//
// Design principles:
//   - Accent-colored header strip with icon + title
//   - Consistent section layout with labeled groups
//   - Live preview toggle with manual Preview button fallback
//   - +/- increment buttons next to numeric fields
//   - Theme-aware colors throughout
//   - Grid-aligned layouts using egui::Grid
// ============================================================================

use eframe::egui;
use egui::{Color32, Pos2, Rect, Rounding, Sense, Stroke, Vec2};
use image::Rgba;

use crate::canvas::{CanvasState, TiledImage};
use super::transform::Interpolation;

use super::effect_dialogs::*;

// ============================================================================
// ACTIVE-DIALOG ENUM - at most one modal dialog is open at a time
// ============================================================================

#[derive(Default)]
pub enum ActiveDialog {
    #[default]
    None,
    ResizeImage(ResizeImageDialog),
    ResizeCanvas(ResizeCanvasDialog),
    GaussianBlur(GaussianBlurDialog),
    LayerTransform(LayerTransformDialog),
    BrightnessContrast(BrightnessContrastDialog),
    HueSaturation(HueSaturationDialog),
    Exposure(ExposureDialog),
    HighlightsShadows(HighlightsShadowsDialog),
    Levels(LevelsDialog),
    Curves(CurvesDialog),
    TemperatureTint(TemperatureTintDialog),
    // Blur effects
    BokehBlur(BokehBlurDialog),
    MotionBlur(MotionBlurDialog),
    BoxBlur(BoxBlurDialog),
    ZoomBlur(ZoomBlurDialog),
    // Distortion effects
    Crystallize(CrystallizeDialog),
    Dents(DentsDialog),
    Pixelate(PixelateDialog),
    Bulge(BulgeDialog),
    Twist(TwistDialog),
    // Noise effects
    AddNoise(AddNoiseDialog),
    ReduceNoise(ReduceNoiseDialog),
    Median(MedianDialog),
    // Stylize effects
    Glow(GlowDialog),
    Sharpen(SharpenDialog),
    Vignette(VignetteDialog),
    Halftone(HalftoneDialog),
    // Render effects
    Grid(GridDialog),
    DropShadow(DropShadowDialog),
    Outline(OutlineDialog),
    // Glitch effects
    PixelDrag(PixelDragDialog),
    RgbDisplace(RgbDisplaceDialog),
    // Artistic effects
    Ink(InkDialog),
    OilPainting(OilPaintingDialog),
    ColorFilter(ColorFilterDialog),
    // Render effects (additional)
    Contours(ContoursDialog),
    // AI
    RemoveBackground(RemoveBackgroundDialog),
    // Color adjustments (new)
    Threshold(ThresholdDialog),
    Posterize(PosterizeDialog),
    ColorBalance(ColorBalanceDialog),
    GradientMap(GradientMapDialog),
    BlackAndWhite(BlackAndWhiteDialog),
    Vibrance(VibranceDialog),
    // Selection
    ColorRange(ColorRangeDialog),
}

impl ActiveDialog {
    /// Returns true if no dialog is currently open.
    pub fn is_none(&self) -> bool {
        matches!(self, ActiveDialog::None)
    }
}

/// Result returned by each dialog's `show()` method every frame.
pub enum DialogResult<T> {
    /// Dialog is still open, no action needed this frame.
    Open,
    /// A parameter changed - caller should apply live preview.
    Changed,
    /// User clicked OK - contains the final values.
    Ok(T),
    /// User clicked Cancel.
    Cancel,
}

// ============================================================================
// SHARED DIALOG STYLING HELPERS
// ============================================================================

/// Colors extracted from the current egui visuals for dialog rendering.
pub(crate) struct DialogColors {
    pub accent: Color32,
    pub accent_strong: Color32,
    pub accent_faint: Color32,
    pub text: Color32,
    pub text_muted: Color32,
    #[allow(dead_code)]
    pub bg: Color32,
    pub separator: Color32,
    pub is_dark: bool,
}

impl DialogColors {
    pub(crate) fn from_ctx(ctx: &egui::Context) -> Self {
        let v = ctx.style().visuals.clone();
        let accent = v.selection.stroke.color;
        let is_dark = v.dark_mode;
        let accent_faint = if is_dark {
            Color32::from_rgba_unmultiplied(accent.r(), accent.g(), accent.b(), 35)
        } else {
            Color32::from_rgba_unmultiplied(accent.r(), accent.g(), accent.b(), 25)
        };
        // In dark mode, boost muted text so labels stay readable
        let text_muted = if is_dark {
            Color32::from_gray(160)
        } else {
            v.weak_text_color()
        };
        Self {
            accent,
            accent_strong: v.selection.stroke.color,
            accent_faint,
            text: v.text_color(),
            text_muted,
            bg: v.window_fill(),
            separator: v.widgets.noninteractive.bg_stroke.color,
            is_dark,
        }
    }
}

/// Paint the accent header bar with icon + title.
pub(crate) fn paint_dialog_header(ui: &mut egui::Ui, colors: &DialogColors, icon: &str, title: &str) {
    let available_width = ui.available_width();
    let header_height = 32.0;
    let (rect, _) = ui.allocate_exact_size(
        Vec2::new(available_width, header_height),
        Sense::hover(),
    );

    let painter = ui.painter();
    // Gradient-like header: solid accent left fading to faint right
    painter.rect_filled(rect, Rounding::ZERO, colors.accent_faint);
    // Left accent bar
    painter.rect_filled(
        Rect::from_min_size(rect.min, Vec2::new(3.0, header_height)),
        Rounding::ZERO,
        colors.accent,
    );

    // Icon + title
    let text_pos = Pos2::new(rect.min.x + 12.0, rect.center().y);
    painter.text(
        text_pos,
        egui::Align2::LEFT_CENTER,
        format!("{} {}", icon, title),
        egui::FontId::proportional(14.0),
        colors.accent_strong,
    );
}

/// Styled section label (small caps feel).
pub(crate) fn section_label(ui: &mut egui::Ui, colors: &DialogColors, text: &str) {
    ui.add_space(6.0);
    ui.horizontal(|ui| {
        ui.add_space(2.0);
        ui.label(egui::RichText::new(text).size(11.0).color(colors.text_muted).strong());
    });
    ui.add_space(2.0);
}

/// Thin separator line using accent color (very faint).
pub(crate) fn accent_separator(ui: &mut egui::Ui, colors: &DialogColors) {
    let available_width = ui.available_width();
    let (rect, _) = ui.allocate_exact_size(Vec2::new(available_width, 1.0), Sense::hover());
    ui.painter().rect_filled(rect, 0.0, colors.accent_faint);
}

/// Draw a numeric field with +/- buttons.  Returns true if the value changed.
pub(crate) fn numeric_field_with_buttons(
    ui: &mut egui::Ui,
    value: &mut f32,
    speed: f32,
    range: std::ops::RangeInclusive<f32>,
    suffix: &str,
    step: f32,
) -> bool {
    let mut changed = false;
    let range_start = *range.start();
    let range_end = *range.end();
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 2.0;
        if ui.small_button("\u{2212}").clicked() {
            *value = (*value - step).max(range_start);
            changed = true;
        }
        let dv = egui::DragValue::new(value)
            .speed(speed)
            .clamp_range(range);
        let dv = if !suffix.is_empty() { dv.suffix(suffix) } else { dv };
        if ui.add(dv).changed() {
            changed = true;
        }
        if ui.small_button("+").clicked() {
            *value = (*value + step).min(range_end);
            changed = true;
        }
    });
    changed
}

/// Styled OK / Cancel footer. Returns (ok_clicked, cancel_clicked).
pub(crate) fn dialog_footer(ui: &mut egui::Ui, colors: &DialogColors) -> (bool, bool) {
    let mut ok = false;
    let mut cancel = false;
    ui.add_space(4.0);
    accent_separator(ui, colors);
    ui.add_space(6.0);
    ui.horizontal(|ui| {
        // Right-align buttons
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui.button(t!("common.cancel")).clicked() {
                cancel = true;
            }
            // Styled accent OK button
            let ok_label = format!("  {}  ", t!("common.ok"));
            let ok_btn = egui::Button::new(
                egui::RichText::new(ok_label).color(Color32::WHITE).strong()
            ).fill(colors.accent);
            if ui.add(ok_btn).clicked() {
                ok = true;
            }
        });
    });
    (ok, cancel)
}

/// Styled OK / Cancel / Reset footer. Returns (ok, cancel, reset).
pub(crate) fn dialog_footer_with_reset(ui: &mut egui::Ui, colors: &DialogColors) -> (bool, bool, bool) {
    let mut ok = false;
    let mut cancel = false;
    let mut reset = false;
    ui.add_space(4.0);
    accent_separator(ui, colors);
    ui.add_space(6.0);
    ui.horizontal(|ui| {
        // Reset on the left
        if ui.button(t!("common.reset")).clicked() {
            reset = true;
        }
        // Right-align OK/Cancel
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui.button(t!("common.cancel")).clicked() {
                cancel = true;
            }
            let ok_label = format!("  {}  ", t!("common.ok"));
            let ok_btn = egui::Button::new(
                egui::RichText::new(ok_label).color(Color32::WHITE).strong()
            ).fill(colors.accent);
            if ui.add(ok_btn).clicked() {
                ok = true;
            }
        });
    });
    (ok, cancel, reset)
}

/// Live preview toggle + manual preview button.
/// Returns true if manual preview was clicked.
pub(crate) fn preview_controls(
    ui: &mut egui::Ui,
    _colors: &DialogColors,
    live_preview: &mut bool,
) -> bool {
    let mut preview_clicked = false;
    ui.add_space(2.0);
    ui.horizontal(|ui| {
        ui.checkbox(live_preview, t!("common.live_preview"));
        if !*live_preview {
            if ui.button(t!("common.preview")).clicked() {
                preview_clicked = true;
            }
        }
    });
    preview_clicked
}


// ============================================================================
// RESIZE IMAGE DIALOG
// ============================================================================

pub struct ResizeImageDialog {
    pub width: f32,
    pub height: f32,
    pub scale_percent: f32,
    pub lock_aspect: bool,
    aspect_ratio: f32,
    pub interpolation: Interpolation,
    pub preset: ResizePreset,
    original_w: u32,
    original_h: u32,
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
            ResizePreset::Custom      => t!("resize_preset.custom"),
            ResizePreset::Hd1920x1080 => t!("resize_preset.hd"),
            ResizePreset::Square1080  => t!("resize_preset.square"),
            ResizePreset::Uhd4K       => t!("resize_preset.uhd4k"),
            ResizePreset::A4At300Dpi  => t!("resize_preset.a4_300dpi"),
        }
    }
    pub fn dims(&self) -> Option<(u32, u32)> {
        match self {
            ResizePreset::Custom      => None,
            ResizePreset::Hd1920x1080 => Some((1920, 1080)),
            ResizePreset::Square1080  => Some((1080, 1080)),
            ResizePreset::Uhd4K       => Some((3840, 2160)),
            ResizePreset::A4At300Dpi  => Some((2480, 3508)),
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
            scale_percent: 100.0,
            lock_aspect: true,
            aspect_ratio: state.width as f32 / state.height.max(1) as f32,
            interpolation: Interpolation::default(),
            preset: ResizePreset::default(),
            original_w: state.width,
            original_h: state.height,
        }
    }

    pub fn show(&mut self, ctx: &egui::Context) -> DialogResult<(u32, u32, Interpolation)> {
        let mut result = DialogResult::Open;
        let colors = DialogColors::from_ctx(ctx);
        let mut ok_pressed = false;

        egui::Window::new("dialog_resize_image")
            .title_bar(false)
            .collapsible(false)
            .resizable(false)
            .default_pos(egui::pos2(
                ctx.screen_rect().center().x - 175.0,
                ctx.screen_rect().center().y - 160.0,
            ))
            .show(ctx, |ui| {
                ui.set_min_width(350.0);

                paint_dialog_header(ui, &colors, "\u{1F4D0}", &t!("dialog.resize_image"));
                ui.add_space(4.0);

                // -- Preset (own grid so it doesn't misalign the dims columns) --
                section_label(ui, &colors, &t!("dialog.resize_image.dimensions"));

                egui::Grid::new("resize_img_preset")
                    .num_columns(2)
                    .min_col_width(80.0)
                    .spacing([8.0, 4.0])
                    .show(ui, |ui| {
                        ui.label(t!("dialog.resize_image.preset"));
                        egui::ComboBox::from_id_source("resize_preset")
                            .width(210.0)
                            .selected_text(self.preset.label())
                            .show_ui(ui, |ui| {
                                for p in ResizePreset::all() {
                                    if ui.selectable_value(&mut self.preset, *p, p.label()).clicked() {
                                        if let Some((w, h)) = p.dims() {
                                            self.width = w as f32;
                                            self.height = h as f32;
                                            self.scale_percent = self.width / self.original_w.max(1) as f32 * 100.0;
                                            self.aspect_ratio = self.width / self.height.max(1.0);
                                        }
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
                        let w_changed = numeric_field_with_buttons(
                            ui, &mut self.width, 1.0, 1.0..=20000.0, "", 1.0,
                        );
                        ui.label("px");
                        ui.end_row();

                        if w_changed {
                            self.preset = ResizePreset::Custom;
                            if self.lock_aspect {
                                self.height = (self.width / self.aspect_ratio).round();
                            } else {
                                self.aspect_ratio = self.width / self.height.max(1.0);
                            }
                            self.scale_percent = self.width / self.original_w.max(1) as f32 * 100.0;
                        }

                        // Height row
                        ui.label(t!("dialog.resize_image.height"));
                        let h_changed = numeric_field_with_buttons(
                            ui, &mut self.height, 1.0, 1.0..=20000.0, "", 1.0,
                        );
                        ui.label("px");
                        ui.end_row();

                        if h_changed {
                            self.preset = ResizePreset::Custom;
                            if self.lock_aspect {
                                self.width = (self.height * self.aspect_ratio).round();
                            } else {
                                self.aspect_ratio = self.width / self.height.max(1.0);
                            }
                            self.scale_percent = self.height / self.original_h.max(1) as f32 * 100.0;
                        }

                        // Lock aspect ratio (inline between H and Scale)
                        ui.label("");
                        let lock_icon = if self.lock_aspect { "\u{1F517}" } else { "\u{25CB}" };
                        if ui.selectable_label(
                            self.lock_aspect,
                            format!("{} {}", lock_icon, t!("dialog.resize_image.lock_aspect")),
                        ).clicked() {
                            self.lock_aspect = !self.lock_aspect;
                            if self.lock_aspect {
                                self.aspect_ratio = self.width / self.height.max(1.0);
                            }
                        }
                        ui.label("");
                        ui.end_row();

                        // Scale row
                        ui.label(t!("dialog.resize_image.scale"));
                        let s_changed = numeric_field_with_buttons(
                            ui, &mut self.scale_percent, 0.5, 1.0..=10000.0, "%", 5.0,
                        );
                        ui.label("");
                        ui.end_row();

                        if s_changed {
                            self.preset = ResizePreset::Custom;
                            self.width = (self.original_w as f32 * self.scale_percent / 100.0).round();
                            self.height = (self.original_h as f32 * self.scale_percent / 100.0).round();
                            self.aspect_ratio = self.width / self.height.max(1.0);
                        }
                    });

                // -- Quality section --
                accent_separator(ui, &colors);
                section_label(ui, &colors, &t!("dialog.resize_image.quality"));

                egui::Grid::new("resize_img_quality")
                    .num_columns(2)
                    .spacing([8.0, 4.0])
                    .show(ui, |ui| {
                        ui.label(t!("dialog.resize_image.interpolation"));
                        egui::ComboBox::from_id_source("resize_interp")
                            .width(160.0)
                            .selected_text(self.interpolation.label())
                            .show_ui(ui, |ui| {
                                for i in Interpolation::all() {
                                    ui.selectable_value(&mut self.interpolation, *i, i.label());
                                }
                            });
                        ui.end_row();
                    });

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
                    ui.label(egui::RichText::new(info).size(11.0).color(colors.text_muted));
                });

                // -- Footer --
                let (ok, cancel) = dialog_footer(ui, &colors);
                if ok { ok_pressed = true; }
                if cancel { result = DialogResult::Cancel; }
            });

        // Keyboard shortcuts
        if matches!(result, DialogResult::Open) {
            if ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::Enter)) {
                ok_pressed = true;
            }
            if ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::Escape)) {
                result = DialogResult::Cancel;
            }
        }
        if ok_pressed && matches!(result, DialogResult::Open) {
            let w = (self.width.round() as u32).max(1);
            let h = (self.height.round() as u32).max(1);
            result = DialogResult::Ok((w, h, self.interpolation));
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
    pub scale_percent: f32,
    /// Anchor as (col, row) each 0..=2.
    pub anchor: (u32, u32),
    pub fill_transparent: bool,
    original_w: u32,
    original_h: u32,
}

impl ResizeCanvasDialog {
    pub fn new(state: &CanvasState) -> Self {
        Self {
            width: state.width as f32,
            height: state.height as f32,
            scale_percent: 100.0,
            anchor: (1, 1), // center
            fill_transparent: true,
            original_w: state.width,
            original_h: state.height,
        }
    }

    pub fn show(&mut self, ctx: &egui::Context, secondary_color: [f32; 4]) -> DialogResult<(u32, u32, (u32, u32), Rgba<u8>)> {
        let mut result = DialogResult::Open;
        let colors = DialogColors::from_ctx(ctx);
        let mut ok_pressed = false;

        egui::Window::new("dialog_resize_canvas")
            .title_bar(false)
            .collapsible(false)
            .resizable(false)
            .default_pos(egui::pos2(
                ctx.screen_rect().center().x - 170.0,
                ctx.screen_rect().center().y - 180.0,
            ))
            .show(ctx, |ui| {
                ui.set_min_width(340.0);

                paint_dialog_header(ui, &colors, "\u{1F532}", &t!("dialog.resize_canvas"));
                ui.add_space(4.0);

                // -- Dimensions section --
                section_label(ui, &colors, &t!("dialog.resize_canvas.canvas_size"));

                egui::Grid::new("resize_canvas_dims")
                    .num_columns(3)
                    .spacing([8.0, 4.0])
                    .show(ui, |ui| {
                        ui.label(t!("dialog.resize_image.width"));
                        let w_changed = numeric_field_with_buttons(
                            ui, &mut self.width, 1.0, 1.0..=20000.0, "", 1.0,
                        );
                        ui.label("px");
                        ui.end_row();

                        if w_changed {
                            self.scale_percent = self.width / self.original_w.max(1) as f32 * 100.0;
                        }

                        ui.label(t!("dialog.resize_image.height"));
                        let h_changed = numeric_field_with_buttons(
                            ui, &mut self.height, 1.0, 1.0..=20000.0, "", 1.0,
                        );
                        ui.label("px");
                        ui.end_row();

                        if h_changed {
                            self.scale_percent = self.height / self.original_h.max(1) as f32 * 100.0;
                        }

                        ui.label(t!("dialog.resize_image.scale"));
                        let s_changed = numeric_field_with_buttons(
                            ui, &mut self.scale_percent, 0.5, 1.0..=10000.0, "%", 5.0,
                        );
                        ui.label("");
                        ui.end_row();

                        if s_changed {
                            self.width = (self.original_w as f32 * self.scale_percent / 100.0).round();
                            self.height = (self.original_h as f32 * self.scale_percent / 100.0).round();
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
                    ui.checkbox(&mut self.fill_transparent, t!("dialog.resize_canvas.fill_transparent"));
                });
                if !self.fill_transparent {
                    ui.horizontal(|ui| {
                        ui.add_space(4.0);
                        ui.label(egui::RichText::new(t!("dialog.resize_canvas.uses_secondary_color")).size(11.0).color(colors.text_muted));
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
                        self.original_w, self.original_h, new_w, new_h,
                        sign_w, dw, sign_h, dh,
                    );
                    ui.label(egui::RichText::new(info).size(11.0).color(colors.text_muted));
                });

                // -- Footer --
                let (ok, cancel) = dialog_footer(ui, &colors);
                if ok { ok_pressed = true; }
                if cancel { result = DialogResult::Cancel; }
            });

        // Keyboard shortcuts
        if matches!(result, DialogResult::Open) {
            if ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::Enter)) {
                ok_pressed = true;
            }
            if ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::Escape)) {
                result = DialogResult::Cancel;
            }
        }
        if ok_pressed && matches!(result, DialogResult::Open) {
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
    fn draw_anchor_grid(&mut self, ui: &mut egui::Ui, colors: &DialogColors, new_w: u32, new_h: u32) {
        let grid_size = 120.0f32;
        let cell_size = grid_size / 3.0;

        // Center the grid
        ui.horizontal(|ui| {
            let avail = ui.available_width();
            let pad = ((avail - grid_size) / 2.0).max(0.0);
            ui.add_space(pad);

            let (grid_rect, response) = ui.allocate_exact_size(Vec2::splat(grid_size), Sense::click());
            let painter = ui.painter();

            // Background
            let grid_bg = if colors.is_dark {
                Color32::from_gray(25)
            } else {
                Color32::from_gray(245)
            };
            painter.rect_filled(grid_rect, 4.0, grid_bg);
            painter.rect_stroke(grid_rect, 4.0, Stroke::new(1.0, colors.separator));

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
                Pos2::new(grid_rect.min.x + offset_x + 4.0, grid_rect.min.y + offset_y + 4.0),
                Vec2::new(preview_w, preview_h),
            );
            let new_area_color = if colors.is_dark {
                Color32::from_gray(40)
            } else {
                Color32::from_gray(225)
            };
            painter.rect_filled(new_canvas_rect, 2.0, new_area_color);

            // Draw original canvas position (accent colored)
            let orig_rect = Rect::from_min_size(
                Pos2::new(cx, cy),
                Vec2::new(canvas_w, canvas_h),
            );
            painter.rect_filled(orig_rect, 1.0, colors.accent_faint);
            painter.rect_stroke(orig_rect, 1.0, Stroke::new(1.5, colors.accent));

            // Handle click on the grid to set anchor
            if response.clicked() {
                if let Some(pos) = response.interact_pointer_pos() {
                    let local_x = pos.x - grid_rect.min.x;
                    let local_y = pos.y - grid_rect.min.y;
                    let col = ((local_x / cell_size).floor() as u32).min(2);
                    let row = ((local_y / cell_size).floor() as u32).min(2);
                    self.anchor = (col, row);
                }
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
                    let dot_color = if selected { colors.accent } else { colors.text_muted };

                    if selected {
                        // Ring around selected dot
                        painter.circle_stroke(center, dot_radius + 2.0, Stroke::new(1.5, colors.accent));
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

        egui::Window::new("dialog_gaussian_blur")
            .title_bar(false)
            .collapsible(false)
            .resizable(false)
            .default_pos(egui::pos2(
                ctx.screen_rect().center().x - 175.0,
                60.0,
            ))
            .show(ctx, |ui| {
                ui.set_min_width(350.0);

                paint_dialog_header(ui, &colors, "\u{1F4A7}", &t!("dialog.gaussian_blur"));
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
                                    egui::DragValue::new(&mut self.sigma)
                                        .speed(0.2)
                                        .clamp_range(0.1..=100.0)
                                        .max_decimals(1)
                                );
                                if track_slider(&r, &mut self.dragging) {
                                    sigma_changed = true;
                                }
                            } else {
                                // Normal: slider capped at 10
                                let r = ui.add(
                                    egui::Slider::new(&mut self.sigma, 0.1..=slider_max)
                                        .max_decimals(1)
                                );
                                if track_slider(&r, &mut self.dragging) {
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
                                &[("Subtle", 1.0), ("Light", 3.0), ("Medium", 8.0), ("Heavy", 25.0), ("Max", 80.0)]
                            } else {
                                &[("Subtle", 0.5), ("Light", 1.5), ("Medium", 3.0), ("Strong", 6.0), ("Max", 10.0)]
                            };
                            for &(label, val) in presets {
                                let is_close = (self.sigma - val).abs() < 0.3;
                                let btn = if is_close {
                                    egui::Button::new(egui::RichText::new(label).strong().size(11.0))
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
                            if ui.checkbox(&mut self.advanced_blur, "Advanced (up to 100)").changed() {
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
// LAYER TRANSFORM DIALOG - with interactive rotation gizmo
// ============================================================================

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
    Z,  // rotation around Z (normal 2D rotation)
    X,  // tilt around X axis
    Y,  // tilt around Y axis
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

        egui::Window::new("dialog_layer_transform")
            .title_bar(false)
            .collapsible(false)
            .resizable(false)
            .default_pos(egui::pos2(
                ctx.screen_rect().center().x - 185.0,
                50.0,
            ))
            .show(ctx, |ui| {
                ui.set_min_width(370.0);

                paint_dialog_header(ui, &colors, "\u{1F504}", &t!("dialog.layer_transform"));
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
                            ui, &mut self.rotation_z, 0.5, -180.0..=180.0, "\u{00B0}", 1.0,
                        ) { changed = true; }
                        ui.end_row();

                        ui.label("Tilt X");
                        if numeric_field_with_buttons(
                            ui, &mut self.rotation_x, 0.5, -80.0..=80.0, "\u{00B0}", 1.0,
                        ) { changed = true; }
                        ui.end_row();

                        ui.label("Tilt Y");
                        if numeric_field_with_buttons(
                            ui, &mut self.rotation_y, 0.5, -80.0..=80.0, "\u{00B0}", 1.0,
                        ) { changed = true; }
                        ui.end_row();

                        ui.label("Scale");
                        if numeric_field_with_buttons(
                            ui, &mut self.scale_percent, 0.5, 1.0..=500.0, "%", 5.0,
                        ) { changed = true; }
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
                        if ui.add(egui::DragValue::new(&mut self.offset_x).speed(1.0)).changed() {
                            changed = true;
                        }
                        ui.label("Y");
                        if ui.add(egui::DragValue::new(&mut self.offset_y).speed(1.0)).changed() {
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

            let (gizmo_rect, response) = ui.allocate_exact_size(
                Vec2::splat(gizmo_size),
                Sense::click_and_drag(),
            );
            let center = gizmo_rect.center();
            let painter = ui.painter();

            // Background circle (subtle depth)
            let bg = if colors.is_dark { Color32::from_gray(22) } else { Color32::from_gray(248) };
            painter.circle_filled(center, half - 2.0, bg);
            painter.circle_stroke(center, half - 2.0, Stroke::new(1.0, colors.separator));

            // Ring radii
            let r_z = half - 8.0;   // outermost - Z rotation
            let r_y = half - 28.0;  // middle - Y tilt
            let r_x = half - 28.0;  // inner - X tilt (drawn as ellipse)

            // Colors per axis
            let z_color = colors.accent;
            let x_color = Color32::from_rgb(230, 80, 80);   // red-ish
            let y_color = Color32::from_rgb(80, 200, 80);    // green-ish
            let active_axis = self.gizmo_drag_axis;

            // Draw Z-ring (outermost circle)
            let z_alpha = if active_axis == Some(GizmoAxis::Z) { 255 } else { 180 };
            let z_stroke = Stroke::new(
                if active_axis == Some(GizmoAxis::Z) { 3.0 } else { 2.0 },
                Color32::from_rgba_unmultiplied(z_color.r(), z_color.g(), z_color.b(), z_alpha),
            );
            painter.circle_stroke(center, r_z, z_stroke);

            // Draw Z rotation indicator (line from center to current angle)
            let z_rad = self.rotation_z.to_radians();
            let z_tip = Pos2::new(
                center.x + z_rad.cos() * r_z,
                center.y - z_rad.sin() * r_z,
            );
            painter.line_segment(
                [center, z_tip],
                Stroke::new(2.0, Color32::from_rgba_unmultiplied(z_color.r(), z_color.g(), z_color.b(), z_alpha)),
            );
            painter.circle_filled(z_tip, 4.0, z_color);

            // Draw Y-ring (vertical ellipse - tilt around Y makes it look like a vertical disc)
            let y_squash = (1.0 - (self.rotation_y.to_radians().sin().abs() * 0.6)).max(0.3);
            let y_alpha = if active_axis == Some(GizmoAxis::Y) { 255 } else { 160 };
            let y_stroke_w = if active_axis == Some(GizmoAxis::Y) { 2.5 } else { 1.5 };
            let y_col = Color32::from_rgba_unmultiplied(y_color.r(), y_color.g(), y_color.b(), y_alpha);
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
            let x_alpha = if active_axis == Some(GizmoAxis::X) { 255 } else { 160 };
            let x_stroke_w = if active_axis == Some(GizmoAxis::X) { 2.5 } else { 1.5 };
            let x_col = Color32::from_rgba_unmultiplied(x_color.r(), x_color.g(), x_color.b(), x_alpha);
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
                [Pos2::new(center.x - cross_len, center.y), Pos2::new(center.x + cross_len, center.y)],
                Stroke::new(1.0, cross_col),
            );
            painter.line_segment(
                [Pos2::new(center.x, center.y - cross_len), Pos2::new(center.x, center.y + cross_len)],
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
            if response.drag_started() {
                if let Some(pos) = response.interact_pointer_pos() {
                    let d = (pos - center).length();
                    // Determine which ring was clicked based on distance from center
                    let axis = if (d - r_z).abs() < 15.0 {
                        Some(GizmoAxis::Z)
                    } else if d < r_y + 10.0 {
                        // Inner region: determine X vs Y based on position
                        let dx = (pos.x - center.x).abs();
                        let dy = (pos.y - center.y).abs();
                        if dy > dx { Some(GizmoAxis::X) } else { Some(GizmoAxis::Y) }
                    } else {
                        Some(GizmoAxis::Z) // default to Z for outer region
                    };

                    self.gizmo_drag_axis = axis;
                    self.gizmo_drag_start = Some(pos);
                    self.gizmo_start_vals = (self.rotation_z, self.rotation_x, self.rotation_y);
                }
            }

            if response.dragged() {
                if let (Some(axis), Some(start), Some(current)) = (
                    self.gizmo_drag_axis,
                    self.gizmo_drag_start,
                    response.interact_pointer_pos(),
                ) {
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
            }

            if response.drag_released() {
                self.gizmo_drag_axis = None;
                self.gizmo_drag_start = None;
            }
        });

        changed
    }
}

// ============================================================================
// BRIGHTNESS / CONTRAST DIALOG
// ============================================================================

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

        egui::Window::new("dialog_brightness_contrast")
            .title_bar(false)
            .collapsible(false)
            .resizable(false)
            .default_pos(egui::pos2(ctx.screen_rect().center().x - 175.0, 60.0))
            .show(ctx, |ui| {
                ui.set_min_width(350.0);
                paint_dialog_header(ui, &colors, "☀", &t!("dialog.brightness_contrast"));
                ui.add_space(4.0);

                section_label(ui, &colors, "ADJUSTMENTS");
                let mut changed = false;

                egui::Grid::new("bc_params")
                    .num_columns(2)
                    .spacing([8.0, 6.0])
                    .show(ui, |ui| {
                        ui.label("Brightness");
                        if ui.add(egui::Slider::new(&mut self.brightness, -100.0..=100.0)
                            .max_decimals(0)).changed() {
                            changed = true;
                        }
                        ui.end_row();

                        ui.label("Contrast");
                        if ui.add(egui::Slider::new(&mut self.contrast, -100.0..=100.0)
                            .max_decimals(0)).changed() {
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
                    let adjusted = (factor * (v + self.brightness - 128.0) + 128.0).clamp(0.0, 255.0) as u8;
                    let color = Color32::from_gray(adjusted);
                    let r = Rect::from_min_size(
                        Pos2::new(bar_rect.min.x + i as f32 * bar_w, bar_rect.min.y),
                        Vec2::new(bar_w + 0.5, 8.0),
                    );
                    painter.rect_filled(r, 0.0, color);
                }
                painter.rect_stroke(bar_rect, Rounding::same(2.0), Stroke::new(1.0, colors.separator));

                accent_separator(ui, &colors);
                let manual_preview = preview_controls(ui, &colors, &mut self.live_preview);
                if changed && self.live_preview { result = DialogResult::Changed; }
                if manual_preview { result = DialogResult::Changed; }

                let (ok, cancel, reset) = dialog_footer_with_reset(ui, &colors);
                if reset {
                    self.brightness = 0.0;
                    self.contrast = 0.0;
                    result = DialogResult::Changed;
                }
                if ok { result = DialogResult::Ok((self.brightness, self.contrast)); }
                if cancel { result = DialogResult::Cancel; }
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

        egui::Window::new("dialog_hue_saturation")
            .title_bar(false)
            .collapsible(false)
            .resizable(false)
            .default_pos(egui::pos2(ctx.screen_rect().center().x - 175.0, 60.0))
            .show(ctx, |ui| {
                ui.set_min_width(380.0);
                paint_dialog_header(ui, &colors, "🎨", &t!("dialog.hue_saturation"));
                ui.add_space(4.0);

                section_label(ui, &colors, "GLOBAL ADJUSTMENTS");
                let mut changed = false;

                egui::Grid::new("hs_params")
                    .num_columns(2)
                    .spacing([8.0, 6.0])
                    .show(ui, |ui| {
                        ui.label("Hue");
                        if ui.add(egui::Slider::new(&mut self.hue, -180.0..=180.0)
                            .suffix("°").max_decimals(0)).changed() {
                            changed = true;
                        }
                        ui.end_row();

                        ui.label("Saturation");
                        if ui.add(egui::Slider::new(&mut self.saturation, -100.0..=100.0)
                            .max_decimals(0)).changed() {
                            changed = true;
                        }
                        ui.end_row();

                        ui.label("Lightness");
                        if ui.add(egui::Slider::new(&mut self.lightness, -100.0..=100.0)
                            .max_decimals(0)).changed() {
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
                    let color = Color32::from_rgb(
                        (r * 255.0) as u8,
                        (g * 255.0) as u8,
                        (b * 255.0) as u8,
                    );
                    let r = Rect::from_min_size(
                        Pos2::new(bar_rect.min.x + i as f32 * bar_w, bar_rect.min.y),
                        Vec2::new(bar_w + 0.5, 12.0),
                    );
                    painter.rect_filled(r, 0.0, color);
                }
                painter.rect_stroke(bar_rect, Rounding::same(2.0), Stroke::new(1.0, colors.separator));

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
                            let is_nondefault = band.hue != 0.0 || band.saturation != 0.0 || band.lightness != 0.0;
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
                                egui::RichText::new(&btn_text).color(if selected { Color32::WHITE } else { band_col })
                            )
                            .fill(if selected { band_col.linear_multiply(0.5) } else { Color32::TRANSPARENT })
                            .stroke(egui::Stroke::new(1.5, band_col));
                            if ui.add_sized([32.0, 22.0], btn).clicked() {
                                self.selected_band = i;
                            }
                        }
                    });

                    // Per-band sliders for selected band
                    let band_label = ["Reds", "Yellows", "Greens", "Cyans", "Blues", "Magentas"][self.selected_band];
                    ui.add_space(4.0);
                    section_label(ui, &colors, &format!("{} BAND", band_label.to_uppercase()));

                    let b = &mut self.bands[self.selected_band];
                    egui::Grid::new("hs_band_params")
                        .num_columns(2)
                        .spacing([8.0, 6.0])
                        .show(ui, |ui| {
                            ui.label("Hue");
                            if ui.add(egui::Slider::new(&mut b.hue, -180.0..=180.0)
                                .suffix("°").max_decimals(0)).changed() {
                                changed = true;
                            }
                            ui.end_row();
                            ui.label("Saturation");
                            if ui.add(egui::Slider::new(&mut b.saturation, -100.0..=100.0)
                                .max_decimals(0)).changed() {
                                changed = true;
                            }
                            ui.end_row();
                            ui.label("Lightness");
                            if ui.add(egui::Slider::new(&mut b.lightness, -100.0..=100.0)
                                .max_decimals(0)).changed() {
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
                if changed && self.live_preview { result = DialogResult::Changed; }
                if manual_preview { result = DialogResult::Changed; }

                let (ok, cancel, reset) = dialog_footer_with_reset(ui, &colors);
                if reset {
                    self.hue = 0.0; self.saturation = 0.0; self.lightness = 0.0;
                    self.bands = Default::default();
                    result = DialogResult::Changed;
                }
                if ok { result = DialogResult::Ok(()); }
                if cancel { result = DialogResult::Cancel; }
            });
        result
    }
}

/// HSV (H: 0..360, S: 0..1, V: 0..1) → RGB (0..1) for UI display
fn hsv_to_rgb_simple(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
    let c = v * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = v - c;
    let (r, g, b) = if h < 60.0 { (c, x, 0.0) }
        else if h < 120.0 { (x, c, 0.0) }
        else if h < 180.0 { (0.0, c, x) }
        else if h < 240.0 { (0.0, x, c) }
        else if h < 300.0 { (x, 0.0, c) }
        else { (c, 0.0, x) };
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

        egui::Window::new("dialog_exposure")
            .title_bar(false)
            .collapsible(false)
            .resizable(false)
            .default_pos(egui::pos2(ctx.screen_rect().center().x - 160.0, 60.0))
            .show(ctx, |ui| {
                ui.set_min_width(320.0);
                paint_dialog_header(ui, &colors, "📷", &t!("dialog.exposure"));
                ui.add_space(4.0);

                section_label(ui, &colors, "EXPOSURE");
                let mut changed = false;

                egui::Grid::new("exp_params")
                    .num_columns(2)
                    .spacing([8.0, 6.0])
                    .show(ui, |ui| {
                        ui.label("EV Stops");
                        if ui.add(egui::Slider::new(&mut self.exposure, -5.0..=5.0)
                            .max_decimals(2).suffix(" EV")).changed() {
                            changed = true;
                        }
                        ui.end_row();

                        ui.label("Quick");
                        ui.horizontal(|ui| {
                            ui.spacing_mut().item_spacing.x = 4.0;
                            for &(label, val) in &[("-2", -2.0), ("-1", -1.0), ("0", 0.0), ("+1", 1.0), ("+2", 2.0)] {
                                let is_close = (self.exposure - val).abs() < 0.1;
                                let btn = if is_close {
                                    egui::Button::new(egui::RichText::new(label).strong().size(11.0))
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
                painter.rect_stroke(bar_rect, Rounding::same(2.0), Stroke::new(1.0, colors.separator));

                accent_separator(ui, &colors);
                let manual_preview = preview_controls(ui, &colors, &mut self.live_preview);
                if changed && self.live_preview { result = DialogResult::Changed; }
                if manual_preview { result = DialogResult::Changed; }

                let (ok, cancel, reset) = dialog_footer_with_reset(ui, &colors);
                if reset { self.exposure = 0.0; result = DialogResult::Changed; }
                if ok { result = DialogResult::Ok(self.exposure); }
                if cancel { result = DialogResult::Cancel; }
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

        egui::Window::new("dialog_highlights_shadows")
            .title_bar(false)
            .collapsible(false)
            .resizable(false)
            .default_pos(egui::pos2(ctx.screen_rect().center().x - 175.0, 60.0))
            .show(ctx, |ui| {
                ui.set_min_width(350.0);
                paint_dialog_header(ui, &colors, "◑", &t!("dialog.highlights_shadows"));
                ui.add_space(4.0);

                section_label(ui, &colors, "TONAL ADJUSTMENTS");
                let mut changed = false;

                egui::Grid::new("hs_params_grid")
                    .num_columns(2)
                    .spacing([8.0, 6.0])
                    .show(ui, |ui| {
                        ui.label("Shadows");
                        if ui.add(egui::Slider::new(&mut self.shadows, -100.0..=100.0)
                            .max_decimals(0)).changed() {
                            changed = true;
                        }
                        ui.end_row();

                        ui.label("Highlights");
                        if ui.add(egui::Slider::new(&mut self.highlights, -100.0..=100.0)
                            .max_decimals(0)).changed() {
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
                    Rounding::ZERO, shadow_color,
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
                    Rounding::ZERO, highlight_color,
                );
                painter.rect_stroke(bar_rect, Rounding::same(2.0), Stroke::new(1.0, colors.separator));
                // Labels
                painter.text(
                    Pos2::new(bar_rect.min.x + half * 0.5, bar_rect.center().y),
                    egui::Align2::CENTER_CENTER,
                    "Shadows", egui::FontId::proportional(9.0),
                    if shadow_brightness > 0.3 { Color32::BLACK } else { Color32::WHITE },
                );
                painter.text(
                    Pos2::new(bar_rect.min.x + half * 1.5, bar_rect.center().y),
                    egui::Align2::CENTER_CENTER,
                    "Highlights", egui::FontId::proportional(9.0),
                    if highlight_brightness > 0.7 { Color32::BLACK } else { Color32::WHITE },
                );

                accent_separator(ui, &colors);
                let manual_preview = preview_controls(ui, &colors, &mut self.live_preview);
                if changed && self.live_preview { result = DialogResult::Changed; }
                if manual_preview { result = DialogResult::Changed; }

                let (ok, cancel, reset) = dialog_footer_with_reset(ui, &colors);
                if reset { self.shadows = 0.0; self.highlights = 0.0; result = DialogResult::Changed; }
                if ok { result = DialogResult::Ok((self.shadows, self.highlights)); }
                if cancel { result = DialogResult::Cancel; }
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
    pub r_ch:   (f32, f32, f32, f32, f32),
    pub g_ch:   (f32, f32, f32, f32, f32),
    pub b_ch:   (f32, f32, f32, f32, f32),
}

pub struct LevelsDialog {
    /// Per-channel input/output level params: index 0=Master, 1=R, 2=G, 3=B
    pub ch_input_black:  [f32; 4],
    pub ch_input_white:  [f32; 4],
    pub ch_gamma:        [f32; 4],
    pub ch_output_black: [f32; 4],
    pub ch_output_white: [f32; 4],
    /// Which channel is currently shown in the editor (0=Master, 1=R, 2=G, 3=B)
    pub active_channel:  usize,
    pub original_pixels: Option<TiledImage>,
    pub original_flat:   Option<image::RgbaImage>,
    pub layer_idx:       usize,
    pub live_preview:    bool,
    /// Histograms: [Luminance, R, G, B]
    pub ch_histograms: [[u32; 256]; 4],
    pub ch_hist_max:   [u32; 4],
}

impl LevelsDialog {
    pub fn new(state: &CanvasState) -> Self {
        let idx = state.active_layer_index;
        let (hr, hg, hb, hl) = crate::ops::adjustments::compute_histogram(state, idx);
        let hist_max = |h: &[u32; 256]| h.iter().copied().max().unwrap_or(1).max(1);
        Self {
            ch_input_black:  [0.0; 4],
            ch_input_white:  [255.0; 4],
            ch_gamma:        [1.0; 4],
            ch_output_black: [0.0; 4],
            ch_output_white: [255.0; 4],
            active_channel:  0,
            original_pixels: state.layers.get(idx).map(|l| l.pixels.clone()),
            original_flat:   state.layers.get(idx).map(|l| l.pixels.to_rgba_image()),
            layer_idx:       idx,
            live_preview:    true,
            ch_histograms:   [hl, hr, hg, hb],
            ch_hist_max:     [hist_max(&hl), hist_max(&hr), hist_max(&hg), hist_max(&hb)],
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
        (self.ch_input_black[c], self.ch_input_white[c], self.ch_gamma[c],
         self.ch_output_black[c], self.ch_output_white[c])
    }

    pub fn as_result(&self) -> LevelsDialogResult {
        let p = |c: usize| (self.ch_input_black[c], self.ch_input_white[c],
                             self.ch_gamma[c], self.ch_output_black[c], self.ch_output_white[c]);
        LevelsDialogResult { master: p(0), r_ch: p(1), g_ch: p(2), b_ch: p(3) }
    }

    pub fn show(&mut self, ctx: &egui::Context) -> DialogResult<LevelsDialogResult> {
        let mut result = DialogResult::Open;
        let colors = DialogColors::from_ctx(ctx);

        egui::Window::new("dialog_levels")
            .title_bar(false)
            .collapsible(false)
            .resizable(false)
            .default_pos(egui::pos2(ctx.screen_rect().center().x - 200.0, 40.0))
            .show(ctx, |ui| {
                ui.set_min_width(400.0);
                paint_dialog_header(ui, &colors, "📊", &t!("dialog.levels"));
                ui.add_space(4.0);

                // --- Channel selector ---
                let ch_labels = ["Master", "Red", "Green", "Blue"];
                ui.horizontal(|ui| {
                    for (i, label) in ch_labels.iter().enumerate() {
                        let selected = self.active_channel == i;
                        let btn_color = if selected { Self::ch_color(i, &colors) } else { colors.text };
                        if ui.add(egui::Button::new(egui::RichText::new(*label).color(btn_color).strong())
                            .fill(if selected {
                                Self::ch_color(i, &colors).linear_multiply(0.2)
                            } else {
                                Color32::TRANSPARENT
                            })
                        ).clicked() {
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
                let hist_rect = ui.allocate_space(Vec2::new(ui.available_width(), hist_height)).1;
                let painter = ui.painter();
                // Background
                painter.rect_filled(hist_rect, Rounding::same(3.0),
                    if colors.is_dark { Color32::from_gray(30) } else { Color32::from_gray(240) });
                // Bars
                let bar_w = hist_rect.width() / 256.0;
                let log_max = (hist_max as f32).ln().max(1.0);
                let bar_color = Self::ch_color(ci, &colors);
                for i in 0..256 {
                    let count = histogram[i];
                    if count == 0 { continue; }
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
                painter.rect_stroke(hist_rect, Rounding::same(3.0), Stroke::new(1.0, colors.separator));

                ui.add_space(4.0);

                // --- Input levels ---
                section_label(ui, &colors, "INPUT LEVELS");
                let mut changed = false;

                egui::Grid::new("levels_input")
                    .num_columns(2)
                    .spacing([8.0, 6.0])
                    .show(ui, |ui| {
                        ui.label("Black Point");
                        if numeric_field_with_buttons(ui, &mut self.ch_input_black[ci], 1.0, 0.0..=254.0, "", 1.0) {
                            if self.ch_input_black[ci] >= self.ch_input_white[ci] {
                                self.ch_input_black[ci] = self.ch_input_white[ci] - 1.0;
                            }
                            changed = true;
                        }
                        ui.end_row();

                        ui.label("Gamma");
                        if ui.add(egui::Slider::new(&mut self.ch_gamma[ci], 0.1..=10.0)
                            .logarithmic(true).max_decimals(2)).changed() {
                            changed = true;
                        }
                        ui.end_row();

                        ui.label("White Point");
                        if numeric_field_with_buttons(ui, &mut self.ch_input_white[ci], 1.0, 1.0..=255.0, "", 1.0) {
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
                        if numeric_field_with_buttons(ui, &mut self.ch_output_black[ci], 1.0, 0.0..=255.0, "", 1.0) {
                            changed = true;
                        }
                        ui.end_row();

                        ui.label("Output White");
                        if numeric_field_with_buttons(ui, &mut self.ch_output_white[ci], 1.0, 0.0..=255.0, "", 1.0) {
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
                    let v = (self.ch_output_black[ci] + t * (self.ch_output_white[ci] - self.ch_output_black[ci])).clamp(0.0, 255.0) as u8;
                    let r = Rect::from_min_size(
                        Pos2::new(bar_rect.min.x + i as f32 * bar_w, bar_rect.min.y),
                        Vec2::new(bar_w + 0.5, 8.0),
                    );
                    painter.rect_filled(r, 0.0, Color32::from_gray(v));
                }
                painter.rect_stroke(bar_rect, Rounding::same(2.0), Stroke::new(1.0, colors.separator));

                accent_separator(ui, &colors);
                let manual_preview = preview_controls(ui, &colors, &mut self.live_preview);
                if changed && self.live_preview { result = DialogResult::Changed; }
                if manual_preview { result = DialogResult::Changed; }

                let (ok, cancel, reset) = dialog_footer_with_reset(ui, &colors);
                if reset {
                    let c = self.active_channel;
                    self.ch_input_black[c]  = 0.0;
                    self.ch_input_white[c]  = 255.0;
                    self.ch_gamma[c]        = 1.0;
                    self.ch_output_black[c] = 0.0;
                    self.ch_output_white[c] = 255.0;
                    result = DialogResult::Changed;
                }
                if ok     { result = DialogResult::Ok(self.as_result()); }
                if cancel { result = DialogResult::Cancel; }
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

        egui::Window::new("dialog_curves")
            .title_bar(false)
            .collapsible(false)
            .resizable(false)
            .default_pos(egui::pos2(ctx.screen_rect().center().x - 195.0, 40.0))
            .show(ctx, |ui| {
                ui.set_min_width(390.0);
                paint_dialog_header(ui, &colors, "📈", &t!("dialog.curves"));
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
                        let text = egui::RichText::new(label).size(11.0)
                            .color(if !is_enabled { colors.text_muted } else if is_active { Color32::WHITE } else { ch_color });
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
                    let eye_text = if self.channels[ch].visible { "👁" } else { "○" };
                    if ui.small_button(eye_text).on_hover_text("Toggle curve visibility on graph").clicked() {
                        self.channels[ch].visible = !self.channels[ch].visible;
                    }
                    let en_text = if self.channels[ch].enabled { "✓" } else { "✗" };
                    let en_color = if self.channels[ch].enabled { colors.accent } else { colors.text_muted };
                    if ui.add(egui::Button::new(egui::RichText::new(en_text).size(12.0).color(en_color)).small())
                        .on_hover_text("Toggle channel enabled/disabled").clicked() {
                        self.channels[ch].enabled = !self.channels[ch].enabled;
                        result = DialogResult::Changed;
                    }
                });
                ui.add_space(2.0);

                // Instructions
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 4.0;
                    ui.label(egui::RichText::new("Click to add points • Right-click to remove • Drag to adjust").size(10.0).color(colors.text_muted));
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
                painter.rect_filled(canvas_rect, Rounding::same(2.0),
                    if colors.is_dark { Color32::from_gray(25) } else { Color32::from_gray(245) });

                // Grid lines
                let grid_color = if colors.is_dark { Color32::from_gray(50) } else { Color32::from_gray(210) };
                for i in 1..4 {
                    let t = i as f32 / 4.0;
                    let x = canvas_rect.min.x + t * canvas_size;
                    let y = canvas_rect.min.y + t * canvas_size;
                    painter.line_segment(
                        [Pos2::new(x, canvas_rect.min.y), Pos2::new(x, canvas_rect.max.y)],
                        Stroke::new(0.5, grid_color),
                    );
                    painter.line_segment(
                        [Pos2::new(canvas_rect.min.x, y), Pos2::new(canvas_rect.max.x, y)],
                        Stroke::new(0.5, grid_color),
                    );
                }

                // Diagonal reference line
                let diag_color = if colors.is_dark { Color32::from_gray(60) } else { Color32::from_gray(180) };
                painter.line_segment(
                    [Pos2::new(canvas_rect.min.x, canvas_rect.max.y),
                     Pos2::new(canvas_rect.max.x, canvas_rect.min.y)],
                    Stroke::new(1.0, diag_color),
                );

                // Draw all visible channel curves (inactive ones first, active on top)
                let draw_order: Vec<usize> = (0..CHANNEL_COUNT)
                    .filter(|&ch| ch != self.active_channel && self.channels[ch].visible)
                    .chain(std::iter::once(self.active_channel).filter(|_| self.channels[self.active_channel].visible))
                    .collect();

                for &ch in &draw_order {
                    let lut = crate::ops::adjustments::build_curves_lut_pub(&self.channels[ch].points);
                    let ch_color = Self::channel_color(ch, &colors);
                    let is_active = ch == self.active_channel;
                    let stroke_w = if is_active { 2.0 } else { 1.2 };
                    let alpha = if is_active { 1.0 } else { 0.45 };

                    let mut curve_points = Vec::with_capacity(256);
                    for i in 0..256 {
                        let x = canvas_rect.min.x + i as f32 / 255.0 * canvas_size;
                        let y = canvas_rect.max.y - lut[i] as f32 / 255.0 * canvas_size;
                        curve_points.push(Pos2::new(x, y));
                    }
                    for w in curve_points.windows(2) {
                        painter.line_segment([w[0], w[1]], Stroke::new(stroke_w, ch_color.linear_multiply(alpha)));
                    }
                }

                // Draw control points for active channel only
                let active_ch = self.active_channel;
                let point_color = Self::channel_color(active_ch, &colors);
                let point_outline = if colors.is_dark { Color32::WHITE } else { Color32::BLACK };
                for (i, &(px, py)) in self.channels[active_ch].points.iter().enumerate() {
                    let screen_x = canvas_rect.min.x + px / 255.0 * canvas_size;
                    let screen_y = canvas_rect.max.y - py / 255.0 * canvas_size;
                    let radius = if self.dragging_point == Some(i) { 6.0 } else { 5.0 };
                    painter.circle_filled(Pos2::new(screen_x, screen_y), radius, point_color);
                    painter.circle_stroke(Pos2::new(screen_x, screen_y), radius, Stroke::new(1.5, point_outline));
                }

                // Handle interaction — operates on active channel
                let mut changed = false;
                let points = &mut self.channels[active_ch].points;
                if let Some(pos) = response.interact_pointer_pos() {
                    let rel_x = ((pos.x - canvas_rect.min.x) / canvas_size * 255.0).clamp(0.0, 255.0);
                    let rel_y = ((canvas_rect.max.y - pos.y) / canvas_size * 255.0).clamp(0.0, 255.0);

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
                                if (px - rel_x).abs() < 0.5 { self.dragging_point = Some(i); break; }
                            }
                            changed = true;
                        }
                    }

                    if response.dragged() {
                        if let Some(idx) = self.dragging_point {
                            if idx > 0 && idx < points.len() - 1 {
                                points[idx] = (rel_x.clamp(
                                    points[idx - 1].0 + 1.0,
                                    points[idx + 1].0 - 1.0,
                                ), rel_y);
                            } else {
                                points[idx].1 = rel_y;
                            }
                            changed = true;
                        }
                    }
                }

                if response.drag_released() {
                    self.dragging_point = None;
                }

                // Right-click to remove point
                if response.secondary_clicked() {
                    if let Some(pos) = response.interact_pointer_pos() {
                        let rel_x = ((pos.x - canvas_rect.min.x) / canvas_size * 255.0).clamp(0.0, 255.0);
                        let rel_y = ((canvas_rect.max.y - pos.y) / canvas_size * 255.0).clamp(0.0, 255.0);
                        let mut closest = None;
                        let mut closest_dist = f32::MAX;
                        for (i, &(px, py)) in points.iter().enumerate() {
                            if i == 0 || i == points.len() - 1 { continue; }
                            let dx = rel_x - px;
                            let dy = rel_y - py;
                            let dist = (dx * dx + dy * dy).sqrt();
                            if dist < closest_dist { closest_dist = dist; closest = Some(i); }
                        }
                        if closest_dist < 15.0 {
                            if let Some(idx) = closest {
                                points.remove(idx);
                                changed = true;
                            }
                        }
                    }
                }

                painter.rect_stroke(canvas_rect, Rounding::same(2.0), Stroke::new(1.0, colors.separator));

                // Channel info
                ui.add_space(2.0);
                let ch_label = Self::channel_label(active_ch);
                ui.label(egui::RichText::new(format!("{} — {} control points", ch_label, self.channels[active_ch].points.len())).size(10.0).color(colors.text_muted));

                // Presets (apply to active channel)
                ui.add_space(2.0);
                section_label(ui, &colors, "PRESETS");
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 4.0;
                    if ui.button(egui::RichText::new("Linear").size(11.0)).clicked() {
                        self.channels[active_ch].points = vec![(0.0, 0.0), (255.0, 255.0)];
                        changed = true;
                    }
                    if ui.button(egui::RichText::new("Brighten").size(11.0)).clicked() {
                        self.channels[active_ch].points = vec![(0.0, 0.0), (128.0, 180.0), (255.0, 255.0)];
                        changed = true;
                    }
                    if ui.button(egui::RichText::new("Darken").size(11.0)).clicked() {
                        self.channels[active_ch].points = vec![(0.0, 0.0), (128.0, 80.0), (255.0, 255.0)];
                        changed = true;
                    }
                    if ui.button(egui::RichText::new("S-Curve").size(11.0)).clicked() {
                        self.channels[active_ch].points = vec![(0.0, 0.0), (64.0, 40.0), (192.0, 215.0), (255.0, 255.0)];
                        changed = true;
                    }
                    if ui.button(egui::RichText::new("Negative").size(11.0)).clicked() {
                        self.channels[active_ch].points = vec![(0.0, 255.0), (255.0, 0.0)];
                        changed = true;
                    }
                });

                accent_separator(ui, &colors);
                let manual_preview = preview_controls(ui, &colors, &mut self.live_preview);
                if changed && self.live_preview { result = DialogResult::Changed; }
                if manual_preview { result = DialogResult::Changed; }

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
                if ok { result = DialogResult::Ok(self.channels.clone()); }
                if cancel { result = DialogResult::Cancel; }
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

        egui::Window::new("dialog_temperature_tint")
            .title_bar(false)
            .collapsible(false)
            .resizable(false)
            .default_pos(egui::pos2(ctx.screen_rect().center().x - 175.0, 60.0))
            .show(ctx, |ui| {
                ui.set_min_width(360.0);
                paint_dialog_header(ui, &colors, "☀", &t!("dialog.temperature_tint"));
                ui.add_space(4.0);

                section_label(ui, &colors, "WHITE BALANCE");
                let mut changed = false;

                egui::Grid::new("tt_params")
                    .num_columns(2)
                    .spacing([8.0, 6.0])
                    .show(ui, |ui| {
                        ui.label("Temperature");
                        if ui.add(egui::Slider::new(&mut self.temperature, -100.0..=100.0)
                            .max_decimals(0)).changed() {
                            changed = true;
                        }
                        ui.end_row();

                        ui.label("Tint");
                        if ui.add(egui::Slider::new(&mut self.tint, -100.0..=100.0)
                            .max_decimals(0)).changed() {
                            changed = true;
                        }
                        ui.end_row();
                    });

                // Temperature gradient bar: blue ← → orange  (interactive)
                ui.add_space(4.0);
                let bar_rect = ui.allocate_space(Vec2::new(ui.available_width(), 18.0)).1;
                let temp_bar_response = ui.interact(bar_rect, ui.id().with("temp_bar"), Sense::click_and_drag());
                let steps = 48;
                let bar_w = bar_rect.width() / steps as f32;
                // Handle drag on temperature bar
                if temp_bar_response.dragged() || temp_bar_response.clicked() {
                    if let Some(pos) = temp_bar_response.interact_pointer_pos() {
                        let t = ((pos.x - bar_rect.min.x) / bar_rect.width()).clamp(0.0, 1.0);
                        self.temperature = (t - 0.5) * 200.0; // map 0..1 → -100..100
                        changed = true;
                    }
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
                    let indicator_x = bar_rect.min.x + (self.temperature / 200.0 + 0.5) * bar_rect.width();
                    let ix = indicator_x.clamp(bar_rect.min.x, bar_rect.max.x);
                    painter.rect_filled(
                        Rect::from_center_size(Pos2::new(ix, bar_rect.center().y), Vec2::new(4.0, bar_rect.height() + 4.0)),
                        Rounding::same(2.0), Color32::WHITE,
                    );
                    painter.rect_stroke(
                        Rect::from_center_size(Pos2::new(ix, bar_rect.center().y), Vec2::new(4.0, bar_rect.height() + 4.0)),
                        Rounding::same(2.0), Stroke::new(1.0, Color32::BLACK),
                    );
                    painter.rect_stroke(bar_rect, Rounding::same(2.0), Stroke::new(1.0, colors.separator));
                }

                // Tint gradient bar: green ← → magenta  (interactive)
                ui.add_space(4.0);
                let bar_rect2 = ui.allocate_space(Vec2::new(ui.available_width(), 18.0)).1;
                let tint_bar_response = ui.interact(bar_rect2, ui.id().with("tint_bar"), Sense::click_and_drag());
                // Handle drag on tint bar
                if tint_bar_response.dragged() || tint_bar_response.clicked() {
                    if let Some(pos) = tint_bar_response.interact_pointer_pos() {
                        let t = ((pos.x - bar_rect2.min.x) / bar_rect2.width()).clamp(0.0, 1.0);
                        self.tint = (t - 0.5) * 200.0;
                        changed = true;
                    }
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
                        Rect::from_center_size(Pos2::new(tx, bar_rect2.center().y), Vec2::new(4.0, bar_rect2.height() + 4.0)),
                        Rounding::same(2.0), Color32::WHITE,
                    );
                    painter.rect_stroke(
                        Rect::from_center_size(Pos2::new(tx, bar_rect2.center().y), Vec2::new(4.0, bar_rect2.height() + 4.0)),
                        Rounding::same(2.0), Stroke::new(1.0, Color32::BLACK),
                    );
                    painter.rect_stroke(bar_rect2, Rounding::same(2.0), Stroke::new(1.0, colors.separator));
                }

                accent_separator(ui, &colors);
                let manual_preview = preview_controls(ui, &colors, &mut self.live_preview);
                if changed && self.live_preview { result = DialogResult::Changed; }
                if manual_preview { result = DialogResult::Changed; }

                let (ok, cancel, reset) = dialog_footer_with_reset(ui, &colors);
                if reset { self.temperature = 0.0; self.tint = 0.0; result = DialogResult::Changed; }
                if ok { result = DialogResult::Ok((self.temperature, self.tint)); }
                if cancel { result = DialogResult::Cancel; }
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

        egui::Window::new("dialog_threshold")
            .title_bar(false)
            .collapsible(false)
            .resizable(false)
            .default_pos(egui::pos2(ctx.screen_rect().center().x - 160.0, 60.0))
            .show(ctx, |ui| {
                ui.set_min_width(320.0);
                paint_dialog_header(ui, &colors, "◑", &t!("dialog.threshold"));
                ui.add_space(4.0);

                section_label(ui, &colors, "THRESHOLD");
                let mut changed = false;

                egui::Grid::new("threshold_params")
                    .num_columns(2)
                    .spacing([8.0, 6.0])
                    .show(ui, |ui| {
                        ui.label("Level");
                        if ui.add(egui::Slider::new(&mut self.level, 0.0..=255.0)
                            .max_decimals(0)).changed() {
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
                    Rect::from_min_size(bar_rect.min, Vec2::new(bar_rect.width() * split + 0.5, 8.0)),
                    Rounding::ZERO, Color32::BLACK,
                );
                painter.rect_filled(
                    Rect::from_min_size(
                        Pos2::new(bar_rect.min.x + bar_rect.width() * split, bar_rect.min.y),
                        Vec2::new(bar_rect.width() * (1.0 - split), 8.0),
                    ),
                    Rounding::ZERO, Color32::WHITE,
                );
                painter.rect_stroke(bar_rect, Rounding::same(2.0), Stroke::new(1.0, colors.separator));

                accent_separator(ui, &colors);
                let manual_preview = preview_controls(ui, &colors, &mut self.live_preview);
                if changed && self.live_preview { result = DialogResult::Changed; }
                if manual_preview { result = DialogResult::Changed; }

                let (ok, cancel, reset) = dialog_footer_with_reset(ui, &colors);
                if reset { self.level = 128.0; result = DialogResult::Changed; }
                if ok { result = DialogResult::Ok(self.level); }
                if cancel { result = DialogResult::Cancel; }
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

        egui::Window::new("dialog_posterize")
            .title_bar(false)
            .collapsible(false)
            .resizable(false)
            .default_pos(egui::pos2(ctx.screen_rect().center().x - 160.0, 60.0))
            .show(ctx, |ui| {
                ui.set_min_width(320.0);
                paint_dialog_header(ui, &colors, "🎨", &t!("dialog.posterize"));
                ui.add_space(4.0);

                section_label(ui, &colors, "POSTERIZE");
                let mut changed = false;
                let mut levels_i32 = self.levels as i32;

                egui::Grid::new("posterize_params")
                    .num_columns(2)
                    .spacing([8.0, 6.0])
                    .show(ui, |ui| {
                        ui.label("Levels");
                        if ui.add(egui::Slider::new(&mut levels_i32, 2..=16)).changed() {
                            self.levels = levels_i32 as u32;
                            changed = true;
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
                painter.rect_stroke(bar_rect, Rounding::same(2.0), Stroke::new(1.0, colors.separator));

                accent_separator(ui, &colors);
                let manual_preview = preview_controls(ui, &colors, &mut self.live_preview);
                if changed && self.live_preview { result = DialogResult::Changed; }
                if manual_preview { result = DialogResult::Changed; }

                let (ok, cancel, reset) = dialog_footer_with_reset(ui, &colors);
                if reset { self.levels = 4; result = DialogResult::Changed; }
                if ok { result = DialogResult::Ok(self.levels); }
                if cancel { result = DialogResult::Cancel; }
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

        egui::Window::new("dialog_color_balance")
            .title_bar(false)
            .collapsible(false)
            .resizable(false)
            .default_pos(egui::pos2(ctx.screen_rect().center().x - 185.0, 60.0))
            .show(ctx, |ui| {
                ui.set_min_width(370.0);
                paint_dialog_header(ui, &colors, "⚖", &t!("dialog.color_balance"));
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
                        ui.label(egui::RichText::new("Cyan — Red").color(Color32::from_rgb(200, 100, 100)));
                        if ui.add(egui::Slider::new(&mut zone[0], -100.0..=100.0).max_decimals(0)).changed() { changed = true; }
                        ui.end_row();

                        ui.label(egui::RichText::new("Magenta — Green").color(Color32::from_rgb(100, 180, 100)));
                        if ui.add(egui::Slider::new(&mut zone[1], -100.0..=100.0).max_decimals(0)).changed() { changed = true; }
                        ui.end_row();

                        ui.label(egui::RichText::new("Yellow — Blue").color(Color32::from_rgb(100, 140, 220)));
                        if ui.add(egui::Slider::new(&mut zone[2], -100.0..=100.0).max_decimals(0)).changed() { changed = true; }
                        ui.end_row();
                    });

                accent_separator(ui, &colors);
                let manual_preview = preview_controls(ui, &colors, &mut self.live_preview);
                if changed && self.live_preview { result = DialogResult::Changed; }
                if manual_preview { result = DialogResult::Changed; }

                let (ok, cancel, reset) = dialog_footer_with_reset(ui, &colors);
                if reset {
                    self.shadows = [0.0; 3];
                    self.midtones = [0.0; 3];
                    self.highlights = [0.0; 3];
                    result = DialogResult::Changed;
                }
                if ok { result = DialogResult::Ok((self.shadows, self.midtones, self.highlights)); }
                if cancel { result = DialogResult::Cancel; }
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
        for i in 0..256 {
            let t = i as f32 / 255.0;
            lut[i] = [
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
                self.shadow_color = [0, 0, 0]; self.highlight_color = [255, 255, 255];
            }
            GradientMapPreset::WhiteToBlack => {
                self.shadow_color = [255, 255, 255]; self.highlight_color = [0, 0, 0];
            }
            GradientMapPreset::BlackToColor => {
                self.shadow_color = [0, 0, 0]; self.highlight_color = [80, 180, 255];
            }
            GradientMapPreset::Sepia => {
                self.shadow_color = [60, 30, 0]; self.highlight_color = [230, 210, 170];
            }
            GradientMapPreset::Infrared => {
                self.shadow_color = [0, 0, 80]; self.highlight_color = [255, 50, 0];
            }
            GradientMapPreset::Custom => {}
        }
    }

    pub fn show(&mut self, ctx: &egui::Context) -> DialogResult<[[u8; 4]; 256]> {
        let mut result = DialogResult::Open;
        let colors = DialogColors::from_ctx(ctx);

        egui::Window::new("dialog_gradient_map")
            .title_bar(false)
            .collapsible(false)
            .resizable(false)
            .default_pos(egui::pos2(ctx.screen_rect().center().x - 185.0, 60.0))
            .show(ctx, |ui| {
                ui.set_min_width(370.0);
                paint_dialog_header(ui, &colors, "🌈", &t!("dialog.gradient_map"));
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
                painter.rect_stroke(bar_rect, Rounding::same(2.0), Stroke::new(1.0, colors.separator));

                // Shadow / Highlight color pickers
                ui.add_space(4.0);
                section_label(ui, &colors, "COLORS");
                let mut changed_color = false;
                egui::Grid::new("gm_colors")
                    .num_columns(2)
                    .spacing([8.0, 6.0])
                    .show(ui, |ui| {
                        ui.label("Shadows");
                        let mut c = Color32::from_rgb(self.shadow_color[0], self.shadow_color[1], self.shadow_color[2]);
                        if ui.color_edit_button_srgba(&mut c).changed() {
                            self.shadow_color = [c.r(), c.g(), c.b()];
                            self.preset = GradientMapPreset::Custom;
                            changed_color = true;
                        }
                        ui.end_row();

                        ui.label("Highlights");
                        let mut c = Color32::from_rgb(self.highlight_color[0], self.highlight_color[1], self.highlight_color[2]);
                        if ui.color_edit_button_srgba(&mut c).changed() {
                            self.highlight_color = [c.r(), c.g(), c.b()];
                            self.preset = GradientMapPreset::Custom;
                            changed_color = true;
                        }
                        ui.end_row();
                    });
                if changed_color && self.live_preview { result = DialogResult::Changed; }

                accent_separator(ui, &colors);
                let manual_preview = preview_controls(ui, &colors, &mut self.live_preview);
                if manual_preview { result = DialogResult::Changed; }

                let (ok, cancel, reset) = dialog_footer_with_reset(ui, &colors);
                if reset {
                    self.apply_preset(GradientMapPreset::BlackToWhite);
                    result = DialogResult::Changed;
                }
                if ok { result = DialogResult::Ok(self.build_lut()); }
                if cancel { result = DialogResult::Cancel; }
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

        egui::Window::new("dialog_black_and_white")
            .title_bar(false)
            .collapsible(false)
            .resizable(false)
            .default_pos(egui::pos2(ctx.screen_rect().center().x - 185.0, 60.0))
            .show(ctx, |ui| {
                ui.set_min_width(370.0);
                paint_dialog_header(ui, &colors, "🎨", &t!("dialog.black_and_white"));
                ui.add_space(4.0);

                // Quick presets
                ui.horizontal(|ui| {
                    if ui.small_button("Luminosity").clicked() {
                        self.r_weight = 21.26; self.g_weight = 71.52; self.b_weight = 7.22;
                        if self.live_preview { result = DialogResult::Changed; }
                    }
                    if ui.small_button("Natural").clicked() {
                        self.r_weight = 40.0; self.g_weight = 40.0; self.b_weight = 20.0;
                        if self.live_preview { result = DialogResult::Changed; }
                    }
                    if ui.small_button("Infrared").clicked() {
                        self.r_weight = 60.0; self.g_weight = 40.0; self.b_weight = 0.0;
                        if self.live_preview { result = DialogResult::Changed; }
                    }
                    if ui.small_button("Flat").clicked() {
                        self.r_weight = 33.33; self.g_weight = 33.33; self.b_weight = 33.33;
                        if self.live_preview { result = DialogResult::Changed; }
                    }
                });

                section_label(ui, &colors, "CHANNEL WEIGHTS");
                let mut changed = false;

                egui::Grid::new("bw_params")
                    .num_columns(2)
                    .spacing([8.0, 6.0])
                    .show(ui, |ui| {
                        ui.label(egui::RichText::new("Reds").color(Color32::from_rgb(220, 80, 80)));
                        if ui.add(egui::Slider::new(&mut self.r_weight, 0.0..=200.0).max_decimals(1)).changed() { changed = true; }
                        ui.end_row();

                        ui.label(egui::RichText::new("Greens").color(Color32::from_rgb(80, 180, 80)));
                        if ui.add(egui::Slider::new(&mut self.g_weight, 0.0..=200.0).max_decimals(1)).changed() { changed = true; }
                        ui.end_row();

                        ui.label(egui::RichText::new("Blues").color(Color32::from_rgb(80, 140, 220)));
                        if ui.add(egui::Slider::new(&mut self.b_weight, 0.0..=200.0).max_decimals(1)).changed() { changed = true; }
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
                painter.rect_stroke(bar_rect, Rounding::same(2.0), Stroke::new(1.0, colors.separator));

                accent_separator(ui, &colors);
                let manual_preview = preview_controls(ui, &colors, &mut self.live_preview);
                if changed && self.live_preview { result = DialogResult::Changed; }
                if manual_preview { result = DialogResult::Changed; }

                let (ok, cancel, reset) = dialog_footer_with_reset(ui, &colors);
                if reset {
                    self.r_weight = 21.26; self.g_weight = 71.52; self.b_weight = 7.22;
                    result = DialogResult::Changed;
                }
                if ok { result = DialogResult::Ok((self.r_weight, self.g_weight, self.b_weight)); }
                if cancel { result = DialogResult::Cancel; }
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

        egui::Window::new("dialog_vibrance")
            .title_bar(false)
            .collapsible(false)
            .resizable(false)
            .default_pos(egui::pos2(ctx.screen_rect().center().x - 175.0, 60.0))
            .show(ctx, |ui| {
                ui.set_min_width(350.0);
                paint_dialog_header(ui, &colors, "✨", &t!("dialog.vibrance"));
                ui.add_space(4.0);

                section_label(ui, &colors, "VIBRANCE");
                let mut changed = false;

                egui::Grid::new("vibrance_params")
                    .num_columns(2)
                    .spacing([8.0, 6.0])
                    .show(ui, |ui| {
                        ui.label("Vibrance");
                        if ui.add(egui::Slider::new(&mut self.amount, -100.0..=100.0).max_decimals(0)).changed() {
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
                    painter.rect_filled(rect, 0.0, Color32::from_rgb((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8));
                }
                painter.rect_stroke(bar_rect, Rounding::same(2.0), Stroke::new(1.0, colors.separator));

                accent_separator(ui, &colors);
                let manual_preview = preview_controls(ui, &colors, &mut self.live_preview);
                if changed && self.live_preview { result = DialogResult::Changed; }
                if manual_preview { result = DialogResult::Changed; }

                let (ok, cancel, reset) = dialog_footer_with_reset(ui, &colors);
                if reset { self.amount = 0.0; result = DialogResult::Changed; }
                if ok { result = DialogResult::Ok(self.amount); }
                if cancel { result = DialogResult::Cancel; }
            });
        result
    }
}

// ============================================================================
// COLOR RANGE DIALOG — select pixels by HSL hue / saturation proximity
// ============================================================================

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

        egui::Window::new("dialog_color_range")
            .title_bar(false)
            .collapsible(false)
            .resizable(false)
            .default_pos(egui::pos2(ctx.screen_rect().center().x - 170.0, 40.0))
            .show(ctx, |ui| {
                ui.set_min_width(340.0);
                paint_dialog_header(ui, &colors, "🎨", "Select Color Range");
                ui.add_space(4.0);

                let mut changed = false;

                egui::Grid::new("color_range_grid")
                    .num_columns(2)
                    .spacing([8.0, 6.0])
                    .show(ui, |ui| {
                        ui.label("Hue Center");
                        if ui.add(egui::Slider::new(&mut self.hue_center, 0.0_f32..=360.0_f32)
                            .suffix("°").fixed_decimals(1)).changed() { changed = true; }
                        ui.end_row();

                        ui.label("Tolerance");
                        if ui.add(egui::Slider::new(&mut self.hue_tolerance, 1.0_f32..=180.0_f32)
                            .suffix("°").fixed_decimals(1)).changed() { changed = true; }
                        ui.end_row();

                        ui.label("Min Saturation");
                        if ui.add(egui::Slider::new(&mut self.sat_min, 0.0_f32..=1.0_f32)
                            .fixed_decimals(2)).changed() { changed = true; }
                        ui.end_row();

                        ui.label("Fuzziness");
                        if ui.add(egui::Slider::new(&mut self.fuzziness, 0.0_f32..=1.0_f32)
                            .fixed_decimals(2)).changed() { changed = true; }
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
                    let (r, g, b) = crate::ops::adjustments::hsl_to_rgb(deg as f32 / 360.0, 1.0, 0.5);
                    let rect = Rect::from_min_size(
                        Pos2::new(bar_rect.min.x + deg as f32 * bar_w, bar_rect.min.y),
                        Vec2::new(bar_w + 0.5, 12.0),
                    );
                    painter.rect_filled(rect, 0.0, Color32::from_rgb(
                        (r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8,
                    ));
                }
                let marker_x = bar_rect.min.x + self.hue_center * (bar_rect.width() / 360.0);
                painter.vline(marker_x, bar_rect.y_range(), Stroke::new(2.0, Color32::WHITE));
                let bw = bar_rect.width() / 360.0;
                let lo_x = bar_rect.min.x + (self.hue_center - self.hue_tolerance).rem_euclid(360.0) * bw;
                let hi_x = bar_rect.min.x + (self.hue_center + self.hue_tolerance).rem_euclid(360.0) * bw;
                let fade = Color32::from_rgba_premultiplied(255, 255, 255, 140);
                painter.vline(lo_x.clamp(bar_rect.min.x, bar_rect.max.x), bar_rect.y_range(), Stroke::new(1.0, fade));
                painter.vline(hi_x.clamp(bar_rect.min.x, bar_rect.max.x), bar_rect.y_range(), Stroke::new(1.0, fade));
                painter.rect_stroke(bar_rect, Rounding::same(2.0), Stroke::new(1.0, colors.separator));

                if changed { result = DialogResult::Changed; }

                accent_separator(ui, &colors);
                let (ok, cancel, reset) = dialog_footer_with_reset(ui, &colors);
                if reset {
                    self.hue_tolerance = 30.0;
                    self.sat_min = 0.1;
                    self.fuzziness = 0.3;
                    result = DialogResult::Changed;
                }
                if ok     { result = DialogResult::Ok(()); }
                if cancel { result = DialogResult::Cancel; }
            });
        result
    }
}
