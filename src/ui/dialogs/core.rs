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

use ::image::Rgba;
use eframe::egui;
use egui::{Color32, CornerRadius, Pos2, Rect, Sense, Stroke, Vec2};

use crate::canvas::{CanvasState, TiledImage};
use crate::ops::transform::Interpolation;

use super::effects::*;

// ============================================================================
// ACTIVE-DIALOG ENUM - at most one modal dialog is open at a time
// ============================================================================

#[derive(Default)]
pub enum ActiveDialog {
    #[default]
    None,
    AddBrushTip(AddBrushTipDialog),
    AddShape(AddShapeDialog),
    ResizeImage(ResizeImageDialog),
    ResizeCanvas(ResizeCanvasDialog),
    AlignLayer(AlignLayerDialog),
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
    ColorToAlpha(ColorToAlphaDialog),
    CanvasBorder(CanvasBorderDialog),
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

    /// Returns a human-readable name for the active dialog (for logging).
    pub fn name(&self) -> &'static str {
        match self {
            ActiveDialog::None => "None",
            ActiveDialog::AddBrushTip(_) => "AddBrushTip",
            ActiveDialog::AddShape(_) => "AddShape",
            ActiveDialog::ResizeImage(_) => "ResizeImage",
            ActiveDialog::ResizeCanvas(_) => "ResizeCanvas",
            ActiveDialog::AlignLayer(_) => "AlignLayer",
            ActiveDialog::GaussianBlur(_) => "GaussianBlur",
            ActiveDialog::LayerTransform(_) => "LayerTransform",
            ActiveDialog::BrightnessContrast(_) => "BrightnessContrast",
            ActiveDialog::HueSaturation(_) => "HueSaturation",
            ActiveDialog::Exposure(_) => "Exposure",
            ActiveDialog::HighlightsShadows(_) => "HighlightsShadows",
            ActiveDialog::Levels(_) => "Levels",
            ActiveDialog::Curves(_) => "Curves",
            ActiveDialog::TemperatureTint(_) => "TemperatureTint",
            ActiveDialog::BokehBlur(_) => "BokehBlur",
            ActiveDialog::MotionBlur(_) => "MotionBlur",
            ActiveDialog::BoxBlur(_) => "BoxBlur",
            ActiveDialog::ZoomBlur(_) => "ZoomBlur",
            ActiveDialog::Crystallize(_) => "Crystallize",
            ActiveDialog::Dents(_) => "Dents",
            ActiveDialog::Pixelate(_) => "Pixelate",
            ActiveDialog::Bulge(_) => "Bulge",
            ActiveDialog::Twist(_) => "Twist",
            ActiveDialog::AddNoise(_) => "AddNoise",
            ActiveDialog::ReduceNoise(_) => "ReduceNoise",
            ActiveDialog::Median(_) => "Median",
            ActiveDialog::Glow(_) => "Glow",
            ActiveDialog::Sharpen(_) => "Sharpen",
            ActiveDialog::Vignette(_) => "Vignette",
            ActiveDialog::Halftone(_) => "Halftone",
            ActiveDialog::Grid(_) => "Grid",
            ActiveDialog::DropShadow(_) => "DropShadow",
            ActiveDialog::Outline(_) => "Outline",
            ActiveDialog::PixelDrag(_) => "PixelDrag",
            ActiveDialog::RgbDisplace(_) => "RgbDisplace",
            ActiveDialog::Ink(_) => "Ink",
            ActiveDialog::OilPainting(_) => "OilPainting",
            ActiveDialog::ColorFilter(_) => "ColorFilter",
            ActiveDialog::ColorToAlpha(_) => "ColorToAlpha",
            ActiveDialog::CanvasBorder(_) => "CanvasBorder",
            ActiveDialog::Contours(_) => "Contours",
            ActiveDialog::RemoveBackground(_) => "RemoveBackground",
            ActiveDialog::Threshold(_) => "Threshold",
            ActiveDialog::Posterize(_) => "Posterize",
            ActiveDialog::ColorBalance(_) => "ColorBalance",
            ActiveDialog::GradientMap(_) => "GradientMap",
            ActiveDialog::BlackAndWhite(_) => "BlackAndWhite",
            ActiveDialog::Vibrance(_) => "Vibrance",
            ActiveDialog::ColorRange(_) => "ColorRange",
        }
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
        let v = ctx.global_style().visuals.clone();
        let accent = v.selection.stroke.color;
        let is_dark = v.dark_mode;
        let accent_faint = if is_dark {
            Color32::from_rgba_unmultiplied(accent.r(), accent.g(), accent.b(), 35)
        } else {
            Color32::from_rgba_unmultiplied(accent.r(), accent.g(), accent.b(), 25)
        };
        // Blue-tinted muted text matching Signal Grid palette
        let text_muted = if is_dark {
            Color32::from_rgb(122, 122, 144) // #7a7a90
        } else {
            Color32::from_rgb(85, 85, 110) // #55556e
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

fn srgb_to_linear_component(c: u8) -> f32 {
    let s = c as f32 / 255.0;
    if s <= 0.04045 {
        s / 12.92
    } else {
        ((s + 0.055) / 1.055).powf(2.4)
    }
}

/// Returns either black or white text based on contrast against `fill`.
pub(crate) fn contrast_text_color(fill: Color32) -> Color32 {
    let r = srgb_to_linear_component(fill.r());
    let g = srgb_to_linear_component(fill.g());
    let b = srgb_to_linear_component(fill.b());
    let luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b;

    let contrast_with_black = (luminance + 0.05) / 0.05;
    let contrast_with_white = 1.05 / (luminance + 0.05);
    if contrast_with_black >= contrast_with_white {
        Color32::BLACK
    } else {
        Color32::WHITE
    }
}

/// Paint the accent header bar with icon + title (Signal Grid style).
/// If `texture_icon` is provided, renders it as an image instead of text.
pub(crate) fn paint_dialog_header(
    ui: &mut egui::Ui,
    colors: &DialogColors,
    icon: &str,
    title: &str,
) -> bool {
    paint_dialog_header_impl(ui, colors, icon, title, None)
}

/// Same as `paint_dialog_header` but with an optional texture handle for the icon.
pub(crate) fn paint_dialog_header_with_texture(
    ui: &mut egui::Ui,
    colors: &DialogColors,
    texture_icon: Option<&egui::TextureHandle>,
    title: &str,
) -> bool {
    paint_dialog_header_impl(ui, colors, "", title, texture_icon)
}

fn paint_dialog_header_impl(
    ui: &mut egui::Ui,
    colors: &DialogColors,
    icon: &str,
    title: &str,
    texture_icon: Option<&egui::TextureHandle>,
) -> bool {
    let available_width = ui.available_width();
    let header_height = 32.0;
    let (rect, response) =
        ui.allocate_exact_size(Vec2::new(available_width, header_height), Sense::click());

    let painter = ui.painter();
    // Gradient-like header: accent faint fill with rounded top corners
    painter.rect_filled(rect, CornerRadius::same(4), colors.accent_faint);
    // Left accent bar (3px, full accent color)
    painter.rect_filled(
        Rect::from_min_size(rect.min, Vec2::new(3.0, header_height)),
        CornerRadius::ZERO,
        colors.accent,
    );

    // Icon + title
    let text_pos = Pos2::new(rect.min.x + 12.0, rect.center().y);
    if let Some(tex) = texture_icon {
        let sized = egui::load::SizedTexture::from_handle(tex);
        let img = egui::Image::from_texture(sized).fit_to_exact_size(egui::vec2(16.0, 16.0));
        img.paint_at(
            ui,
            egui::Rect::from_center_size(
                Pos2::new(text_pos.x + 8.0, text_pos.y),
                egui::vec2(16.0, 16.0),
            ),
        );
        painter.text(
            Pos2::new(text_pos.x + 22.0, text_pos.y),
            egui::Align2::LEFT_CENTER,
            title,
            egui::FontId::proportional(14.0),
            colors.accent_strong,
        );
    } else {
        painter.text(
            text_pos,
            egui::Align2::LEFT_CENTER,
            format!("{icon} {title}"),
            egui::FontId::proportional(14.0),
            colors.accent_strong,
        );
    }

    let close_size = Vec2::new(header_height, header_height);
    let close_rect = Rect::from_center_size(
        Pos2::new(rect.max.x - close_size.x * 0.5, rect.center().y),
        close_size,
    );
    let close_response = ui.interact(
        close_rect,
        response.id.with("dialog_header_close"),
        Sense::click(),
    );
    painter.text(
        close_rect.center(),
        egui::Align2::CENTER_CENTER,
        "×",
        egui::FontId::proportional(14.0),
        if close_response.hovered() {
            colors.accent_strong
        } else {
            colors.accent
        },
    );

    close_response.clicked()
}

/// Styled section label with monospace uppercase styling.
pub(crate) fn section_label(ui: &mut egui::Ui, colors: &DialogColors, text: &str) {
    ui.add_space(6.0);
    ui.horizontal(|ui| {
        ui.add_space(2.0);
        ui.label(
            egui::RichText::new(text.to_uppercase())
                .size(11.0)
                .color(colors.text_muted)
                .strong()
                .monospace(),
        );
    });
    ui.add_space(2.0);
}

/// Thin separator line (subtle monotone).
pub(crate) fn accent_separator(ui: &mut egui::Ui, colors: &DialogColors) {
    let available_width = ui.available_width();
    let (rect, _) = ui.allocate_exact_size(Vec2::new(available_width, 1.0), Sense::hover());
    ui.painter().rect_filled(rect, 0.0, colors.separator);
}

fn themed_stepper_button(ui: &mut egui::Ui, label: &str) -> egui::Response {
    let stepper_bg = crate::theme::Theme::stepper_button_bg_for(ui);
    ui.scope(|ui| {
        ui.visuals_mut().widgets.inactive.bg_fill = stepper_bg;
        ui.visuals_mut().widgets.inactive.weak_bg_fill = stepper_bg;
        ui.small_button(label)
    })
    .inner
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
    numeric_field_with_buttons_focus(ui, value, speed, range, suffix, step, false)
}

/// Like `numeric_field_with_buttons`, but when `request_focus` is true the
/// DragValue is given keyboard focus on this frame (used to auto-select
/// the first field when a dialog opens).
pub(crate) fn numeric_field_with_buttons_focus(
    ui: &mut egui::Ui,
    value: &mut f32,
    speed: f32,
    range: std::ops::RangeInclusive<f32>,
    suffix: &str,
    step: f32,
    request_focus: bool,
) -> bool {
    let mut changed = false;
    let range_start = *range.start();
    let range_end = *range.end();
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 2.0;
        if themed_stepper_button(ui, "-").clicked() {
            *value = (*value - step).max(range_start);
            changed = true;
        }
        let dv = egui::DragValue::new(value).speed(speed).range(range);
        let dv = if !suffix.is_empty() {
            dv.suffix(suffix)
        } else {
            dv
        };
        let response = ui.add(dv);
        if request_focus {
            response.request_focus();
        }
        if response.changed() {
            changed = true;
        }
        if themed_stepper_button(ui, "+").clicked() {
            *value = (*value + step).min(range_end);
            changed = true;
        }
    });
    changed
}

/// Painter-based slider with a downward-pointing arrow thumb + flanking −/+ buttons.
/// Matches the visual style of the colour-panel gradient bars.
/// `step` is the increment for the buttons; `suffix` is appended to the value label.
/// Returns `true` if the value changed.
pub(crate) fn dialog_slider(
    ui: &mut egui::Ui,
    value: &mut f32,
    range: std::ops::RangeInclusive<f32>,
    step: f32,
    suffix: &str,
    decimals: usize,
) -> bool {
    let range_start = *range.start();
    let range_end = *range.end();
    let default_value = if range_start <= 0.0 && range_end >= 0.0 {
        0.0
    } else {
        range_start
    };
    let mut changed = false;

    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 2.0;

        // Custom painted slider track + arrow thumb
        let bar_h = 6.0_f32;
        let arrow_h = 14.0_f32;
        let bar_w = 140.0_f32;
        let desired = egui::Vec2::new(bar_w, bar_h + arrow_h + 1.0);
        let (rect, resp) = ui.allocate_exact_size(desired, egui::Sense::click_and_drag());
        let bar = egui::Rect::from_min_size(
            egui::Pos2::new(rect.min.x, rect.max.y - bar_h),
            egui::Vec2::new(rect.width(), bar_h),
        );

        if ui.is_rect_visible(rect) {
            let p = ui.painter();
            let vis = ui.visuals();
            let track_col = vis.extreme_bg_color;
            let filled_col = vis.selection.bg_fill;
            let border_col = if vis.widgets.inactive.bg_stroke.color == egui::Color32::TRANSPARENT {
                egui::Color32::from_black_alpha(40)
            } else {
                vis.widgets.inactive.bg_stroke.color
            };

            // Filled portion (left of thumb)
            let t = (*value - range_start) / (range_end - range_start).max(f32::EPSILON);
            let fill_x = bar.min.x + t * bar.width();
            if fill_x > bar.min.x {
                p.rect_filled(
                    egui::Rect::from_min_max(bar.min, egui::Pos2::new(fill_x, bar.max.y)),
                    egui::CornerRadius::same(2),
                    filled_col.linear_multiply(0.6),
                );
            }
            // Unfilled portion
            if fill_x < bar.max.x {
                p.rect_filled(
                    egui::Rect::from_min_max(egui::Pos2::new(fill_x, bar.min.y), bar.max),
                    egui::CornerRadius::same(2),
                    track_col,
                );
            }
            p.rect_stroke(
                bar,
                egui::CornerRadius::same(2),
                egui::Stroke::new(1.0, border_col),
                egui::StrokeKind::Middle,
            );

            // Arrow thumb — small dark triangle, white outline (Paint.NET style)
            let tx = bar.min.x + t * bar.width();
            let aw = 5.5_f32;
            let ah = 9.0_f32;
            let tip = egui::Pos2::new(tx, bar.center().y);
            let base_y = tip.y - ah;
            let bl = egui::Pos2::new(tx - aw, base_y);
            let br = egui::Pos2::new(tx + aw, base_y);
            p.add(egui::Shape::convex_polygon(
                vec![tip, bl, br],
                egui::Color32::from_gray(20),
                egui::Stroke::NONE,
            ));
            p.add(egui::Shape::convex_polygon(
                vec![tip, bl, br],
                egui::Color32::TRANSPARENT,
                egui::Stroke::new(1.5, egui::Color32::WHITE),
            ));
        }

        // Drag / click interaction
        if (resp.dragged() || resp.clicked())
            && let Some(mp) = resp.interact_pointer_pos()
        {
            let t = ((mp.x - bar.min.x) / bar.width()).clamp(0.0, 1.0);
            let new_val = range_start + t * (range_end - range_start);
            // Snap to step grid
            let snapped = (new_val / step).round() * step;
            *value = snapped.clamp(range_start, range_end);
            changed = true;
        }

        // Stepper + manual input cluster (buttons flank the numeric field).
        ui.allocate_ui_with_layout(
            egui::Vec2::new(146.0, 22.0),
            egui::Layout::left_to_right(egui::Align::Center),
            |ui| {
                ui.spacing_mut().item_spacing.x = 2.0;

                if ui.add_sized([20.0, 18.0], egui::Button::new("-")).clicked() {
                    *value = (*value - step).max(range_start);
                    changed = true;
                }

                let dv = egui::DragValue::new(value)
                    .speed(step * 0.5)
                    .range(range)
                    .max_decimals(decimals);
                let dv = if !suffix.is_empty() {
                    dv.suffix(suffix)
                } else {
                    dv
                };
                if ui.add_sized([60.0, 18.0], dv).changed() {
                    changed = true;
                }

                if ui.add_sized([20.0, 18.0], egui::Button::new("+")).clicked() {
                    *value = (*value + step).min(range_end);
                    changed = true;
                }

                let is_default = (*value - default_value).abs() <= step.abs().max(0.0001) * 0.5;
                let reset_resp = ui
                    .add_enabled(!is_default, egui::Button::new("↺"))
                    .on_hover_text("Reset to default");
                if reset_resp.clicked() {
                    *value = default_value;
                    changed = true;
                }
            },
        );
    });
    changed
}

fn format_dimension_value(value: f32) -> String {
    if (value.round() - value).abs() < 0.0001 {
        format!("{}", value.round() as i64)
    } else {
        let mut s = format!("{value:.4}");
        while s.contains('.') && s.ends_with('0') {
            s.pop();
        }
        if s.ends_with('.') {
            s.pop();
        }
        s
    }
}

fn evaluate_dimension_expression(input: &str) -> Option<f32> {
    #[derive(Clone, Copy)]
    struct Parser<'a> {
        bytes: &'a [u8],
        pos: usize,
    }

    impl<'a> Parser<'a> {
        fn new(src: &'a str) -> Self {
            Self {
                bytes: src.as_bytes(),
                pos: 0,
            }
        }

        fn skip_ws(&mut self) {
            while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_whitespace() {
                self.pos += 1;
            }
        }

        fn parse_expr(&mut self) -> Option<f32> {
            let mut value = self.parse_term()?;
            loop {
                self.skip_ws();
                let Some(&op) = self.bytes.get(self.pos) else {
                    break;
                };
                if op != b'+' && op != b'-' {
                    break;
                }
                self.pos += 1;
                let rhs = self.parse_term()?;
                value = if op == b'+' { value + rhs } else { value - rhs };
            }
            Some(value)
        }

        fn parse_term(&mut self) -> Option<f32> {
            let mut value = self.parse_factor()?;
            loop {
                self.skip_ws();
                let Some(&op) = self.bytes.get(self.pos) else {
                    break;
                };
                if op != b'*' && op != b'/' {
                    break;
                }
                self.pos += 1;
                let rhs = self.parse_factor()?;
                value = if op == b'*' {
                    value * rhs
                } else {
                    if rhs.abs() < f32::EPSILON {
                        return None;
                    }
                    value / rhs
                };
            }
            Some(value)
        }

        fn parse_factor(&mut self) -> Option<f32> {
            self.skip_ws();
            if let Some(&b'+') = self.bytes.get(self.pos) {
                self.pos += 1;
                return self.parse_factor();
            }
            if let Some(&b'-') = self.bytes.get(self.pos) {
                self.pos += 1;
                return self.parse_factor().map(|v| -v);
            }
            if let Some(&b'(') = self.bytes.get(self.pos) {
                self.pos += 1;
                let value = self.parse_expr()?;
                self.skip_ws();
                if self.bytes.get(self.pos) == Some(&b')') {
                    self.pos += 1;
                    return Some(value);
                }
                return None;
            }
            self.parse_number()
        }

        fn parse_number(&mut self) -> Option<f32> {
            self.skip_ws();
            let start = self.pos;
            let mut seen_digit = false;
            let mut seen_dot = false;
            while let Some(&ch) = self.bytes.get(self.pos) {
                if ch.is_ascii_digit() {
                    seen_digit = true;
                    self.pos += 1;
                } else if ch == b'.' && !seen_dot {
                    seen_dot = true;
                    self.pos += 1;
                } else {
                    break;
                }
            }
            if !seen_digit {
                return None;
            }
            std::str::from_utf8(&self.bytes[start..self.pos])
                .ok()?
                .parse::<f32>()
                .ok()
        }
    }

    let trimmed = input.trim();
    if trimmed.is_empty() {
        return None;
    }
    let mut parser = Parser::new(trimmed);
    let value = parser.parse_expr()?;
    parser.skip_ws();
    if parser.pos != parser.bytes.len() || !value.is_finite() {
        return None;
    }
    Some(value)
}

/// Styled OK / Cancel footer. Returns (ok_clicked, cancel_clicked).
pub(crate) fn dialog_footer(ui: &mut egui::Ui, colors: &DialogColors) -> (bool, bool) {
    let mut ok = false;
    let mut cancel = false;

    // Global footer keyboard behavior for all dialogs that use this helper.
    if ui.input(|i| i.key_pressed(egui::Key::Enter)) {
        ok = true;
    }
    if ui.input(|i| i.key_pressed(egui::Key::Escape)) {
        cancel = true;
    }

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
                egui::RichText::new(ok_label)
                    .color(contrast_text_color(colors.accent))
                    .strong(),
            )
            .fill(colors.accent);
            if ui.add(ok_btn).clicked() {
                ok = true;
            }
        });
    });
    (ok, cancel)
}

/// Styled OK / Cancel / Reset footer. Returns (ok, cancel, reset).
pub(crate) fn dialog_footer_with_reset(
    ui: &mut egui::Ui,
    colors: &DialogColors,
) -> (bool, bool, bool) {
    let mut ok = false;
    let mut cancel = false;
    let mut reset = false;

    // Global footer keyboard behavior for all dialogs that use this helper.
    if ui.input(|i| i.key_pressed(egui::Key::Enter)) {
        ok = true;
    }
    if ui.input(|i| i.key_pressed(egui::Key::Escape)) {
        cancel = true;
    }

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
                egui::RichText::new(ok_label)
                    .color(contrast_text_color(colors.accent))
                    .strong(),
            )
            .fill(colors.accent);
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
        if !*live_preview && ui.button(t!("common.preview")).clicked() {
            preview_clicked = true;
        }
    });
    preview_clicked
}

// ============================================================================
// RESIZE IMAGE DIALOG
// ============================================================================

mod image_dialogs {
    use super::*;
    include!("core/image.rs");
}
pub use image_dialogs::*;

mod transform {
    use super::*;
    include!("core/transform.rs");
}
pub use transform::*;

mod adjustments {
    use super::*;
    include!("core/adjustments.rs");
}
pub use adjustments::*;

mod selection {
    use super::*;
    include!("core/selection.rs");
}
pub use selection::*;
