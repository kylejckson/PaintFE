use eframe::egui;
use egui::{Color32, ColorImage, TextureHandle, TextureOptions};
use image::ImageEncoder;
use image::codecs::bmp::BmpEncoder;
use image::codecs::jpeg::JpegEncoder;
use image::codecs::png::PngEncoder;
use image::codecs::tiff::TiffEncoder;
use image::imageops::FilterType;
use image::{DynamicImage, RgbaImage};
use std::io::Cursor;
use std::path::PathBuf;
use std::sync::mpsc;

// ============================================================================
// UNITS ENUM
// ============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum SizeUnit {
    #[default]
    Pixels,
    Inches,
    Centimeters,
}

impl SizeUnit {
    pub fn label(&self) -> String {
        match self {
            SizeUnit::Pixels => t!("unit.px_suffix"),
            SizeUnit::Inches => t!("unit.in_suffix"),
            SizeUnit::Centimeters => t!("unit.cm_suffix"),
        }
    }
}

// ============================================================================
// SIZE PRESETS
// ============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum SizePreset {
    #[default]
    Custom,
    Size800x600,
    Size1280x720,
    Size1920x1080,
    Size2560x1440,
    Size4K,
    A4At300Ppi,
    A4At72Ppi,
    LetterAt300Ppi,
}

impl SizePreset {
    pub fn label(&self) -> String {
        match self {
            SizePreset::Custom => t!("new_file_preset.custom"),
            SizePreset::Size800x600 => t!("new_file_preset.800x600"),
            SizePreset::Size1280x720 => t!("new_file_preset.1280x720"),
            SizePreset::Size1920x1080 => t!("new_file_preset.1920x1080"),
            SizePreset::Size2560x1440 => t!("new_file_preset.2560x1440"),
            SizePreset::Size4K => t!("new_file_preset.4k"),
            SizePreset::A4At300Ppi => t!("new_file_preset.a4_300ppi"),
            SizePreset::A4At72Ppi => t!("new_file_preset.a4_72ppi"),
            SizePreset::LetterAt300Ppi => t!("new_file_preset.letter_300ppi"),
        }
    }

    /// Returns (width, height, ppi) for the preset
    pub fn dimensions(&self) -> Option<(u32, u32, f32)> {
        match self {
            SizePreset::Custom => None,
            SizePreset::Size800x600 => Some((800, 600, 72.0)),
            SizePreset::Size1280x720 => Some((1280, 720, 72.0)),
            SizePreset::Size1920x1080 => Some((1920, 1080, 72.0)),
            SizePreset::Size2560x1440 => Some((2560, 1440, 72.0)),
            SizePreset::Size4K => Some((3840, 2160, 72.0)),
            // A4 is 210mm × 297mm = 8.27in × 11.69in
            SizePreset::A4At300Ppi => Some((2480, 3508, 300.0)),
            SizePreset::A4At72Ppi => Some((595, 842, 72.0)),
            // US Letter is 8.5in × 11in
            SizePreset::LetterAt300Ppi => Some((2550, 3300, 300.0)),
        }
    }

    fn all() -> &'static [SizePreset] {
        &[
            SizePreset::Custom,
            SizePreset::Size800x600,
            SizePreset::Size1280x720,
            SizePreset::Size1920x1080,
            SizePreset::Size2560x1440,
            SizePreset::Size4K,
            SizePreset::A4At300Ppi,
            SizePreset::A4At72Ppi,
            SizePreset::LetterAt300Ppi,
        ]
    }
}

// ============================================================================
// NEW FILE DIALOG
// ============================================================================

pub struct NewFileDialog {
    pub open: bool,
    width: f32,
    height: f32,
    width_input: String,
    height_input: String,
    unit: SizeUnit,
    ppi: f32,
    lock_aspect_ratio: bool,
    aspect_ratio: f32,
    preset: SizePreset,
    focus_width_on_open: bool,
    clipboard_dims_rx: Option<mpsc::Receiver<Option<(u32, u32)>>>,
}

impl Default for NewFileDialog {
    fn default() -> Self {
        Self {
            open: false,
            width: 800.0,
            height: 600.0,
            width_input: "800".to_string(),
            height_input: "600".to_string(),
            unit: SizeUnit::Pixels,
            ppi: 72.0,
            lock_aspect_ratio: true,
            aspect_ratio: 800.0 / 600.0,
            preset: SizePreset::Size800x600,
            focus_width_on_open: false,
            clipboard_dims_rx: None,
        }
    }
}

impl NewFileDialog {
    pub fn set_lock_aspect_ratio(&mut self, lock: bool) {
        self.lock_aspect_ratio = lock;
        if lock {
            self.aspect_ratio = self.width / self.height.max(1.0);
        }
    }

    pub fn lock_aspect_ratio(&self) -> bool {
        self.lock_aspect_ratio
    }

    pub fn open_dialog(&mut self) {
        self.sync_inputs_from_values();
        self.open = true;
        self.focus_width_on_open = true;
    }

    /// Pre-populate width and height from a clipboard image if available.
    /// Called when the dialog opens so newly-created canvases match the
    /// clipboard image dimensions by default.
    pub fn load_clipboard_dimensions(&mut self) {
        // Fast path: in-app clipboard dimensions are available immediately.
        if let Some((w, h)) = crate::ops::clipboard::get_internal_clipboard_image_dimensions() {
            self.width = w as f32;
            self.height = h as f32;
            self.aspect_ratio = self.width / self.height.max(1.0);
            self.preset = SizePreset::Custom;
            self.unit = SizeUnit::Pixels;
            self.sync_inputs_from_values();
        }

        // Slow path: probe system clipboard on a background thread so opening the
        // dialog remains instant even when clipboard providers are sluggish.
        // Not available on web (no threads) — the system clipboard isn't
        // reachable from wasm anyway, so this is skipped entirely there.
        #[cfg(not(target_arch = "wasm32"))]
        {
            let (tx, rx) = mpsc::channel();
            self.clipboard_dims_rx = Some(rx);
            std::thread::spawn(move || {
                let dims = crate::ops::clipboard::get_clipboard_image_dimensions();
                let _ = tx.send(dims);
            });
        }
    }

    fn sync_inputs_from_values(&mut self) {
        self.width_input = format_dimension_value(self.width);
        self.height_input = format_dimension_value(self.height);
    }

    fn preview_values(&self) -> (f32, f32) {
        let width = evaluate_dimension_expression(&self.width_input).unwrap_or(self.width);
        let height = evaluate_dimension_expression(&self.height_input).unwrap_or(self.height);
        (width.max(1.0), height.max(1.0))
    }

    fn preview_pixels(&self) -> (u32, u32) {
        let (width, height) = self.preview_values();
        let (w, h) = match self.unit {
            SizeUnit::Pixels => (width, height),
            SizeUnit::Inches => (width * self.ppi, height * self.ppi),
            SizeUnit::Centimeters => {
                let inches_w = width / 2.54;
                let inches_h = height / 2.54;
                (inches_w * self.ppi, inches_h * self.ppi)
            }
        };
        (w.round().max(1.0) as u32, h.round().max(1.0) as u32)
    }

    fn commit_width_input(&mut self) {
        let old_width = self.width;
        if let Some(new_width) = evaluate_dimension_expression(&self.width_input) {
            let new_width = new_width.round().clamp(1.0, 20000.0);
            self.width = new_width;
            self.width_input = format_dimension_value(self.width);
            self.preset = SizePreset::Custom;
            if self.lock_aspect_ratio && old_width > 0.0 {
                self.height = (self.width / self.aspect_ratio).round().clamp(1.0, 20000.0);
                self.height_input = format_dimension_value(self.height);
            } else {
                self.aspect_ratio = self.width / self.height.max(1.0);
            }
        } else {
            self.width_input = format_dimension_value(self.width);
        }
    }

    fn commit_height_input(&mut self) {
        let old_height = self.height;
        if let Some(new_height) = evaluate_dimension_expression(&self.height_input) {
            let new_height = new_height.round().clamp(1.0, 20000.0);
            self.height = new_height;
            self.height_input = format_dimension_value(self.height);
            self.preset = SizePreset::Custom;
            if self.lock_aspect_ratio && old_height > 0.0 {
                self.width = (self.height * self.aspect_ratio)
                    .round()
                    .clamp(1.0, 20000.0);
                self.width_input = format_dimension_value(self.width);
            } else {
                self.aspect_ratio = self.width / self.height.max(1.0);
            }
        } else {
            self.height_input = format_dimension_value(self.height);
        }
    }

    fn commit_inputs(&mut self) {
        self.commit_width_input();
        self.commit_height_input();
    }

    /// Convert current width/height from current unit to pixels
    fn to_pixels(&self) -> (u32, u32) {
        let (w, h) = match self.unit {
            SizeUnit::Pixels => (self.width, self.height),
            SizeUnit::Inches => (self.width * self.ppi, self.height * self.ppi),
            SizeUnit::Centimeters => {
                // 1 inch = 2.54 cm
                let inches_w = self.width / 2.54;
                let inches_h = self.height / 2.54;
                (inches_w * self.ppi, inches_h * self.ppi)
            }
        };
        (w.round().max(1.0) as u32, h.round().max(1.0) as u32)
    }

    /// Show the dialog and return Some((width, height)) if user clicks Create
    pub fn show(&mut self, ctx: &egui::Context) -> Option<(u32, u32)> {
        use crate::ops::dialogs::{
            DialogColors, accent_separator, contrast_text_color, paint_dialog_header, section_label,
        };

        let mut result = None;
        let mut should_close = false;

        if self.open {
            if let Some(rx) = &self.clipboard_dims_rx {
                match rx.try_recv() {
                    Ok(Some((w, h))) => {
                        self.width = w as f32;
                        self.height = h as f32;
                        self.aspect_ratio = self.width / self.height.max(1.0);
                        self.preset = SizePreset::Custom;
                        self.unit = SizeUnit::Pixels;
                        self.sync_inputs_from_values();
                        self.clipboard_dims_rx = None;
                    }
                    Ok(None) => {
                        self.clipboard_dims_rx = None;
                    }
                    Err(mpsc::TryRecvError::Disconnected) => {
                        self.clipboard_dims_rx = None;
                    }
                    Err(mpsc::TryRecvError::Empty) => {}
                }
            }

            crate::ui::polish::window(ctx, "new_file_dialog_internal")
                .title_bar(false)
                .collapsible(false)
                .resizable(false)
                .heading_drag(
                    ctx,
                    egui::Id::new("new_image_heading_position"),
                    ctx.content_rect().center() - egui::vec2(200.0, 200.0),
                )
                .show(ctx, |ui| {
                    ui.set_min_width(360.0);

                    let colors = DialogColors::from_ctx(ctx);
                    if paint_dialog_header(ui, &colors, "\u{1F4C4}", &t!("dialog.new_file")) {
                        should_close = true;
                    }
                    ui.add_space(6.0);

                    // ── Preset row ──────────────────────────────────────────
                    section_label(ui, &colors, &t!("common.preset"));
                    egui::Grid::new("new_file_preset_grid")
                        .num_columns(2)
                        .min_col_width(80.0)
                        .spacing([8.0, 4.0])
                        .show(ui, |ui| {
                            ui.label(t!("common.preset"));
                            egui::ComboBox::from_id_salt("preset_combo")
                                .width(224.0)
                                .selected_text(crate::i18n::translate_english(&self.preset.label()))
                                .show_ui(ui, |ui| {
                                    for preset in SizePreset::all() {
                                        if ui
                                            .selectable_value(
                                                &mut self.preset,
                                                *preset,
                                                crate::i18n::translate_english(&preset.label()),
                                            )
                                            .clicked()
                                            && let Some((w, h, ppi)) = preset.dimensions()
                                        {
                                            self.width = w as f32;
                                            self.height = h as f32;
                                            self.ppi = ppi;
                                            self.unit = SizeUnit::Pixels;
                                            self.aspect_ratio = self.width / self.height;
                                            self.sync_inputs_from_values();
                                        }
                                    }
                                });
                            ui.end_row();
                        });

                    // ── Dimensions grid ──────────────────────────────────────
                    accent_separator(ui, &colors);
                    section_label(ui, &colors, &t!("dialog.resize_image.dimensions"));

                    egui::Grid::new("new_file_dims_grid")
                        .num_columns(2)
                        .min_col_width(80.0)
                        .spacing([8.0, 6.0])
                        .show(ui, |ui| {
                            ui.label(t!("common.width"));
                            let mut width = self.width;
                            let width_response = ui.add(
                                crate::ui::controls::Numeric::dimension(&mut width)
                                    .focus_on_open(self.focus_width_on_open)
                                    .id_salt("new_width")
                                    .suffix(format!(
                                        " {}",
                                        crate::i18n::translate_english(&self.unit.label())
                                    )),
                            );
                            self.focus_width_on_open = false;
                            if width_response.changed() {
                                self.width_input = format_dimension_value(width);
                                self.commit_width_input();
                            }
                            ui.end_row();
                            ui.label(t!("common.height"));
                            let mut height = self.height;
                            if ui
                                .add(
                                    crate::ui::controls::Numeric::dimension(&mut height)
                                        .id_salt("new_height")
                                        .suffix(format!(
                                            " {}",
                                            crate::i18n::translate_english(&self.unit.label())
                                        )),
                                )
                                .changed()
                            {
                                self.height_input = format_dimension_value(height);
                                self.commit_height_input();
                            }
                            ui.end_row();

                            ui.label("");
                            let checkbox_resp = ui.checkbox(
                                &mut self.lock_aspect_ratio,
                                t!("common.lock_aspect_ratio"),
                            );
                            if checkbox_resp.changed() && self.lock_aspect_ratio {
                                self.aspect_ratio = self.width / self.height.max(1.0);
                            }
                            ui.end_row();
                        });

                    // ── Options (units / resolution) ─────────────────────────
                    accent_separator(ui, &colors);
                    section_label(ui, &colors, &t!("common.units"));

                    egui::Grid::new("new_file_units_grid")
                        .num_columns(2)
                        .min_col_width(80.0)
                        .spacing([8.0, 4.0])
                        .show(ui, |ui| {
                            ui.label(t!("common.units"));
                            let old_unit = self.unit;
                            let unit_label = match self.unit {
                                SizeUnit::Pixels => t!("unit.pixels"),
                                SizeUnit::Inches => t!("unit.inches"),
                                SizeUnit::Centimeters => t!("unit.centimeters"),
                            };
                            egui::ComboBox::from_id_salt("units_combo")
                                .width(160.0)
                                .selected_text(unit_label)
                                .show_ui(ui, |ui| {
                                    if ui
                                        .selectable_value(
                                            &mut self.unit,
                                            SizeUnit::Pixels,
                                            t!("unit.pixels"),
                                        )
                                        .clicked()
                                    {
                                        self.convert_unit(old_unit, SizeUnit::Pixels);
                                    }
                                    if ui
                                        .selectable_value(
                                            &mut self.unit,
                                            SizeUnit::Inches,
                                            t!("unit.inches"),
                                        )
                                        .clicked()
                                    {
                                        self.convert_unit(old_unit, SizeUnit::Inches);
                                    }
                                    if ui
                                        .selectable_value(
                                            &mut self.unit,
                                            SizeUnit::Centimeters,
                                            t!("unit.centimeters"),
                                        )
                                        .clicked()
                                    {
                                        self.convert_unit(old_unit, SizeUnit::Centimeters);
                                    }
                                });
                            ui.end_row();

                            if self.unit != SizeUnit::Pixels {
                                ui.label(t!("common.resolution"));
                                ui.horizontal(|ui| {
                                    ui.add(
                                        crate::ui::controls::Numeric::new(&mut self.ppi)
                                            .speed(1.0)
                                            .range(1.0..=1200.0),
                                    );
                                    ui.label(t!("unit.ppi"));
                                });
                                ui.end_row();
                            }
                        });

                    // ── Final pixel size info ────────────────────────────────
                    ui.add_space(4.0);
                    let (px_w, px_h) = self.preview_pixels();
                    ui.horizontal(|ui| {
                        ui.add_space(4.0);
                        ui.label(
                            egui::RichText::new(
                                t!("common.final_size")
                                    .replace("{0}", &px_w.to_string())
                                    .replace("{1}", &px_h.to_string()),
                            )
                            .size(11.0)
                            .color(colors.text_muted),
                        );
                    });

                    // ── Footer ───────────────────────────────────────────────
                    let (enter, esc) = crate::ui::dialogs::core::dialog_footer_keys(ctx);
                    should_close |= esc;
                    ui.add_space(4.0);
                    accent_separator(ui, &colors);
                    ui.add_space(6.0);
                    ui.horizontal(|ui| {
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui.button(t!("common.cancel")).clicked() {
                                should_close = true;
                            }
                            let create_btn = egui::Button::new(
                                egui::RichText::new(format!("  {}  ", t!("common.create")))
                                    .color(contrast_text_color(colors.accent))
                                    .strong(),
                            )
                            .fill(colors.accent);
                            if ui.add(create_btn).clicked() || enter {
                                self.commit_inputs();
                                result = Some(self.to_pixels());
                                should_close = true;
                            }
                        });
                    });
                });
        }

        if should_close {
            self.open = false;
        }

        result
    }

    /// Convert width/height values between units
    fn convert_unit(&mut self, from: SizeUnit, to: SizeUnit) {
        if from == to {
            return;
        }

        // First convert to pixels
        let (px_w, px_h) = match from {
            SizeUnit::Pixels => (self.width, self.height),
            SizeUnit::Inches => (self.width * self.ppi, self.height * self.ppi),
            SizeUnit::Centimeters => {
                let in_w = self.width / 2.54;
                let in_h = self.height / 2.54;
                (in_w * self.ppi, in_h * self.ppi)
            }
        };

        // Then convert to target unit
        let (new_w, new_h) = match to {
            SizeUnit::Pixels => (px_w, px_h),
            SizeUnit::Inches => (px_w / self.ppi, px_h / self.ppi),
            SizeUnit::Centimeters => {
                let in_w = px_w / self.ppi;
                let in_h = px_h / self.ppi;
                (in_w * 2.54, in_h * 2.54)
            }
        };

        self.width = new_w.round().max(1.0);
        self.height = new_h.round().max(1.0);
        self.sync_inputs_from_values();
    }
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

use crate::ui::numeric::evaluate_dimension_expression;
mod export_ui;

// ============================================================================
// IMAGE FORMAT ENUM
// ============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub enum SaveFormat {
    #[default]
    Png,
    Jpeg,
    Jxl(crate::io::jxl::JxlOptions),
    Webp,
    Avif,
    Heic,
    Bmp,
    Tga,
    Ico,
    Tiff,
    Gif,
    Pfe,
}

/// Compression options for TIFF format
#[derive(Clone, Copy, Debug, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub enum TiffCompression {
    #[default]
    None,
    Lzw,
    Deflate,
}

impl SaveFormat {
    pub fn extension(&self) -> &'static str {
        match self {
            SaveFormat::Png => "png",
            SaveFormat::Jpeg => "jpg",
            SaveFormat::Jxl(_) => "jxl",
            SaveFormat::Avif => "avif",
            SaveFormat::Heic => "heic",
            SaveFormat::Webp => "webp",
            SaveFormat::Bmp => "bmp",
            SaveFormat::Tga => "tga",
            SaveFormat::Ico => "ico",
            SaveFormat::Tiff => "tiff",
            SaveFormat::Gif => "gif",
            SaveFormat::Pfe => "pfe",
        }
    }

    pub fn label(&self) -> String {
        match self {
            SaveFormat::Png => t!("format.png"),
            SaveFormat::Jpeg => t!("format.jpeg"),
            SaveFormat::Jxl(_) => t!("format.jxl"),
            SaveFormat::Avif => t!("format.avif"),
            SaveFormat::Heic => t!("format.heic"),
            SaveFormat::Webp => t!("format.webp"),
            SaveFormat::Bmp => t!("format.bmp"),
            SaveFormat::Tga => t!("format.tga"),
            SaveFormat::Ico => t!("format.ico"),
            SaveFormat::Tiff => t!("format.tiff"),
            SaveFormat::Gif => t!("format.gif"),
            SaveFormat::Pfe => t!("format.pfe"),
        }
    }

    pub fn supports_quality(&self) -> bool {
        matches!(
            self,
            SaveFormat::Jpeg
                | SaveFormat::Webp
                | SaveFormat::Jxl(_)
                | SaveFormat::Avif
                | SaveFormat::Heic
        )
    }

    /// Returns true if this format supports animated (multi-frame) output.
    pub fn supports_animation(&self) -> bool {
        matches!(self, SaveFormat::Png | SaveFormat::Gif | SaveFormat::Webp)
    }

    fn all() -> &'static [SaveFormat] {
        &[
            SaveFormat::Pfe,
            SaveFormat::Png,
            SaveFormat::Jpeg,
            #[cfg(not(target_arch = "wasm32"))]
            SaveFormat::Jxl(crate::io::jxl::JxlOptions {
                lossless: true,
                effort: 5,
            }),
            #[cfg(not(target_arch = "wasm32"))]
            SaveFormat::Avif,
            #[cfg(not(target_arch = "wasm32"))]
            SaveFormat::Heic,
            SaveFormat::Webp,
            SaveFormat::Bmp,
            SaveFormat::Tga,
            SaveFormat::Ico,
            SaveFormat::Tiff,
            SaveFormat::Gif,
        ]
    }
}

// ============================================================================
// PREVIEW GENERATION
// ============================================================================

/// Result of encoding an image for preview purposes
pub struct PreviewResult {
    /// The encoded file size in bytes
    pub file_size: usize,
    /// The decoded preview image (showing compression artifacts)
    pub preview_image: RgbaImage,
}

/// Encode an image to a format with quality settings and return size + decoded preview
/// This is used to show what the saved image will look like
pub fn generate_preview(
    image: &RgbaImage,
    format: SaveFormat,
    quality: u8,
    webp_lossless: bool,
) -> Option<PreviewResult> {
    let mut buffer = Vec::new();

    match format {
        SaveFormat::Png => {
            // PNG is lossless, just get the size
            let mut cursor = Cursor::new(&mut buffer);
            let encoder = PngEncoder::new(&mut cursor);
            encoder
                .write_image(
                    image.as_raw(),
                    image.width(),
                    image.height(),
                    image::ExtendedColorType::Rgba8,
                )
                .ok()?;

            Some(PreviewResult {
                file_size: buffer.len(),
                preview_image: image.clone(),
            })
        }
        SaveFormat::Jpeg => {
            // JPEG is lossy - encode then decode to show artifacts
            let rgb_image = DynamicImage::ImageRgba8(image.clone()).to_rgb8();
            let mut cursor = Cursor::new(&mut buffer);
            let mut encoder = JpegEncoder::new_with_quality(&mut cursor, quality);
            encoder
                .encode(
                    rgb_image.as_raw(),
                    rgb_image.width(),
                    rgb_image.height(),
                    image::ColorType::Rgb8.into(),
                )
                .ok()?;

            // Decode back to show compression artifacts
            let decoded = image::load_from_memory(&buffer).ok()?;
            let preview = decoded.to_rgba8();

            Some(PreviewResult {
                file_size: buffer.len(),
                preview_image: preview,
            })
        }
        SaveFormat::Heic => {
            let (bytes, preview_image) = crate::io::heic::preview(image, quality).ok()?;
            Some(PreviewResult {
                file_size: bytes.len(),
                preview_image,
            })
        }
        SaveFormat::Avif => {
            let (bytes, preview_image) = crate::io::avif::preview(image, quality).ok()?;
            Some(PreviewResult {
                file_size: bytes.len(),
                preview_image,
            })
        }
        SaveFormat::Jxl(options) => {
            let bytes = crate::io::jxl::encode_rgba8(image, quality, options).ok()?;
            crate::io::register_codecs();
            let preview_image = crate::io::decode_image_bytes(&bytes).ok()?.to_rgba8();
            Some(PreviewResult {
                file_size: bytes.len(),
                preview_image,
            })
        }
        SaveFormat::Webp => {
            #[cfg(not(target_arch = "wasm32"))]
            {
                let encoder =
                    webp::Encoder::from_rgba(image.as_raw(), image.width(), image.height());
                let encoded = if webp_lossless {
                    encoder.encode_lossless()
                } else {
                    encoder.encode(quality.clamp(1, 100) as f32)
                };
                buffer.extend_from_slice(&encoded);
            }
            // Web fallback: pure-Rust lossless-only WebP encoder (no libwebp).
            #[cfg(target_arch = "wasm32")]
            {
                use image::ImageEncoder;
                let _ = webp_lossless;
                let _ = quality;
                let encoder = image::codecs::webp::WebPEncoder::new_lossless(&mut buffer);
                encoder
                    .write_image(
                        image.as_raw(),
                        image.width(),
                        image.height(),
                        image::ExtendedColorType::Rgba8,
                    )
                    .ok()?;
            }

            // Decode back
            let decoded = image::load_from_memory(&buffer).ok()?;
            let preview = decoded.to_rgba8();

            Some(PreviewResult {
                file_size: buffer.len(),
                preview_image: preview,
            })
        }
        SaveFormat::Bmp => {
            // BMP is lossless
            let mut cursor = Cursor::new(&mut buffer);
            let mut encoder = BmpEncoder::new(&mut cursor);
            encoder
                .encode(
                    image.as_raw(),
                    image.width(),
                    image.height(),
                    image::ColorType::Rgba8.into(),
                )
                .ok()?;

            Some(PreviewResult {
                file_size: buffer.len(),
                preview_image: image.clone(),
            })
        }
        SaveFormat::Tga => {
            // TGA - estimate size (lossless)
            // TGA with RGBA is roughly width * height * 4 + header
            let estimated_size = (image.width() * image.height() * 4) as usize + 18;

            Some(PreviewResult {
                file_size: estimated_size,
                preview_image: image.clone(),
            })
        }
        SaveFormat::Ico => {
            // ICO is lossless (PNG-encoded inside)
            // Estimate size based on the image data
            let mut buffer = Vec::new();
            let mut cursor = Cursor::new(&mut buffer);
            let encoder = PngEncoder::new(&mut cursor);
            #[allow(deprecated)]
            encoder
                .write_image(
                    image.as_raw(),
                    image.width(),
                    image.height(),
                    image::ExtendedColorType::Rgba8,
                )
                .ok()?;
            // ICO overhead: ~22 bytes header + 16 bytes per entry + PNG data
            let estimated_size = buffer.len() + 38;

            Some(PreviewResult {
                file_size: estimated_size,
                preview_image: image.clone(),
            })
        }
        SaveFormat::Tiff => {
            // TIFF is lossless
            let mut buffer = Vec::new();
            let mut cursor = Cursor::new(&mut buffer);
            let encoder = TiffEncoder::new(&mut cursor);
            encoder
                .write_image(
                    image.as_raw(),
                    image.width(),
                    image.height(),
                    image::ExtendedColorType::Rgba8,
                )
                .ok()?;

            Some(PreviewResult {
                file_size: buffer.len(),
                preview_image: image.clone(),
            })
        }
        SaveFormat::Pfe => {
            // PFE is a project format — no single-image preview needed
            // Estimate a rough size (header + raw pixels)
            let estimated_size = (image.width() * image.height() * 4) as usize + 64;

            Some(PreviewResult {
                file_size: estimated_size,
                preview_image: image.clone(),
            })
        }
        SaveFormat::Gif => {
            // GIF is lossy (256 colors max) — estimate size
            // Rough estimate: width * height (indexed color) + overhead
            let estimated_size = (image.width() * image.height()) as usize + 800;

            Some(PreviewResult {
                file_size: estimated_size,
                preview_image: image.clone(),
            })
        }
    }
}

/// Create a thumbnail of an image for preview display
pub fn create_thumbnail(image: &RgbaImage, max_size: u32) -> RgbaImage {
    let (width, height) = image.dimensions();

    if width <= max_size && height <= max_size {
        return image.clone();
    }

    let scale = (max_size as f32 / width.max(height) as f32).min(1.0);
    let new_width = ((width as f32 * scale) as u32).max(1);
    let new_height = ((height as f32 * scale) as u32).max(1);

    image::imageops::resize(image, new_width, new_height, FilterType::Triangle)
}

/// Convert RgbaImage to egui ColorImage
fn rgba_to_color_image(img: &RgbaImage) -> ColorImage {
    let size = [img.width() as usize, img.height() as usize];
    let pixels: Vec<Color32> = img
        .pixels()
        .map(|p| Color32::from_rgba_unmultiplied(p[0], p[1], p[2], p[3]))
        .collect();
    ColorImage {
        size,
        source_size: egui::Vec2::new(img.width() as f32, img.height() as f32),
        pixels,
    }
}

// ============================================================================
// SAVE ACTION
// ============================================================================

#[derive(Clone, Debug)]
pub struct SaveAction {
    pub path: PathBuf,
    pub options: crate::io::export::ExportOptions,
}
impl std::ops::Deref for SaveAction {
    type Target = crate::io::export::ExportOptions;
    fn deref(&self) -> &Self::Target {
        &self.options
    }
}
impl std::ops::DerefMut for SaveAction {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.options
    }
}

// ============================================================================
// SAVE FILE DIALOG
// ============================================================================

/// Maximum size for preview thumbnail (keeps preview generation fast).
/// 512 gives enough resolution to zoom in meaningfully on large images.
const PREVIEW_MAX_SIZE: u32 = 512;

/// Return destination and UV rectangles for the texture-backed portion of a
/// preview viewport, leaving out-of-bounds areas for the checkerboard.
fn clipped_preview_rects(
    panel: egui::Rect,
    texture_size: egui::Vec2,
    source_min: egui::Pos2,
    source_size: egui::Vec2,
) -> Option<(egui::Rect, egui::Rect)> {
    if texture_size.x <= 0.0
        || texture_size.y <= 0.0
        || source_size.x <= 0.0
        || source_size.y <= 0.0
    {
        return None;
    }

    let source_max = source_min + source_size;
    let clipped_min = egui::pos2(source_min.x.max(0.0), source_min.y.max(0.0));
    let clipped_max = egui::pos2(
        source_max.x.min(texture_size.x),
        source_max.y.min(texture_size.y),
    );
    if clipped_min.x >= clipped_max.x || clipped_min.y >= clipped_max.y {
        return None;
    }

    let destination = egui::Rect::from_min_max(
        panel.min
            + egui::vec2(
                (clipped_min.x - source_min.x) / source_size.x * panel.width(),
                (clipped_min.y - source_min.y) / source_size.y * panel.height(),
            ),
        panel.min
            + egui::vec2(
                (clipped_max.x - source_min.x) / source_size.x * panel.width(),
                (clipped_max.y - source_min.y) / source_size.y * panel.height(),
            ),
    );
    let uv = egui::Rect::from_min_max(
        egui::pos2(
            clipped_min.x / texture_size.x,
            clipped_min.y / texture_size.y,
        ),
        egui::pos2(
            clipped_max.x / texture_size.x,
            clipped_max.y / texture_size.y,
        ),
    );
    Some((destination, uv))
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct PreviewKey {
    options: crate::io::export::ExportOptions,
    generation: u64,
    frame: usize,
}

pub struct SaveFileDialog {
    encoded_preview: Option<ColorImage>,
    pub open: bool,
    options: crate::io::export::ExportOptions,
    filename: String,
    pub target_directory: Option<PathBuf>,

    // Preview state
    source_thumbnail: Option<RgbaImage>,
    source_dimensions: (u32, u32), // Full source image dimensions
    preview_texture: Option<TextureHandle>,
    preview_file_size: usize,   // Thumbnail-based file size
    estimated_full_size: usize, // Estimated full-resolution file size
    source_generation: u64,
    source_depth: u8,
    source_has_alpha: bool,
    source_has_color: bool,
    prepared_thumbnail: Option<crate::io::PreparedExportImage>,
    preview_requested: Option<(PreviewKey, f64)>,
    preview_completed: Option<PreviewKey>,
    preview_active: Option<PreviewKey>,
    preview_error: Option<String>,
    format_options: std::collections::BTreeMap<String, crate::io::export::ExportOptions>,
    needs_preview_update: bool,
    jxl_preview_job: Option<std::sync::mpsc::Receiver<(PreviewKey, Result<PreviewResult, String>)>>,

    // Interactive preview pan/zoom
    /// 0.0 = not yet computed (will auto-fit on first frame)
    preview_zoom: f32,
    /// Top-left visible corner in source-pixel coordinates
    preview_pan: egui::Vec2,
    /// (start screen pos, start pan) when a drag is in progress
    preview_drag_start: Option<(egui::Pos2, egui::Vec2)>,
    /// Whether the texture was last uploaded with NEAREST filter
    preview_texture_is_nearest: bool,

    // Animation options
    layer_count: usize, // number of layers/frames available
    was_animated: bool, // whether source was animated on import

    // Animation preview playback
    anim_playing: bool,                         // play/pause state
    anim_current_frame: usize,                  // which frame is displayed (0-based)
    anim_last_frame_time: f64,                  // timestamp of last frame advance
    frame_thumbnails: Vec<RgbaImage>,           // per-frame thumbnails
    frame_textures: Vec<Option<TextureHandle>>, // cached per-frame textures
}

impl Default for SaveFileDialog {
    fn default() -> Self {
        Self {
            encoded_preview: None,
            open: false,
            options: Default::default(),
            filename: "untitled".to_string(),
            target_directory: None,
            source_thumbnail: None,
            source_dimensions: (0, 0),
            preview_texture: None,
            preview_file_size: 0,
            estimated_full_size: 0,
            source_generation: 0,
            source_depth: 8,
            source_has_alpha: false,
            source_has_color: false,
            prepared_thumbnail: None,
            preview_requested: None,
            preview_completed: None,
            preview_active: None,
            preview_error: None,
            format_options: Default::default(),
            needs_preview_update: true,
            jxl_preview_job: None,
            preview_zoom: 0.0,
            preview_pan: egui::Vec2::ZERO,
            preview_drag_start: None,
            preview_texture_is_nearest: false,
            // Animation
            layer_count: 1,
            was_animated: false,
            anim_playing: false,
            anim_current_frame: 0,
            anim_last_frame_time: 0.0,
            frame_thumbnails: Vec::new(),
            frame_textures: Vec::new(),
        }
    }
}

impl std::ops::Deref for SaveFileDialog {
    type Target = crate::io::export::ExportOptions;
    fn deref(&self) -> &Self::Target {
        &self.options
    }
}
impl std::ops::DerefMut for SaveFileDialog {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.options
    }
}

impl SaveFileDialog {
    pub fn format(&self) -> SaveFormat {
        self.format
    }

    #[cfg(test)]
    pub(crate) fn filename(&self) -> &str {
        &self.filename
    }
    /// Reset dialog state for a fresh "Save As"
    pub fn reset(&mut self) {
        self.encoded_preview = None;
        self.filename = "untitled".to_string();
        self.target_directory = None;
        self.source_thumbnail = None;
        self.source_generation = self.source_generation.wrapping_add(1);
        self.prepared_thumbnail = None;
        self.preview_requested = None;
        self.preview_completed = None;
        self.preview_error = None;
        self.source_dimensions = (0, 0);
        self.preview_texture = None;
        self.preview_file_size = 0;
        self.estimated_full_size = 0;
        self.needs_preview_update = true;
        self.webp_lossless = true;
        self.preview_zoom = 0.0;
        self.preview_pan = egui::Vec2::ZERO;
        self.preview_drag_start = None;
        self.preview_texture_is_nearest = false;
        // Animation reset
        self.animated = false;
        self.animation_fps = 10.0;
        self.gif_colors = 256;
        self.gif_dither = true;
        self.layer_count = 1;
        self.was_animated = false;
        self.anim_playing = false;
        self.anim_current_frame = 0;
        self.anim_last_frame_time = 0.0;
        self.frame_thumbnails.clear();
        self.frame_textures.clear();
    }

    /// Set the source image for preview generation
    /// Creates a thumbnail for efficient preview rendering
    pub fn set_source_image(&mut self, image: &RgbaImage) {
        self.source_generation = self.source_generation.wrapping_add(1);
        self.prepared_thumbnail = None;
        self.preview_requested = None;
        self.preview_completed = None;
        self.preview_error = None;
        self.source_dimensions = (image.width(), image.height());
        self.source_depth = 8;
        self.source_has_alpha = image.pixels().any(|p| p[3] != 255);
        self.source_has_color = image.pixels().any(|p| p[0] != p[1] || p[1] != p[2]);
        self.source_thumbnail = Some(create_thumbnail(image, PREVIEW_MAX_SIZE));
        self.needs_preview_update = true;
        // Reset pan/zoom so auto-fit recalculates for the new image.
        self.preview_zoom = 0.0;
        self.preview_pan = egui::Vec2::ZERO;
        self.preview_drag_start = None;
        self.preview_texture_is_nearest = false;
    }

    /// Set animation info and per-frame thumbnails for the save dialog.
    /// `frame_images`: one RgbaImage per frame/layer (composite of that layer alone).
    /// `was_animated`: whether the source file was already animated.
    /// `fps`: animation fps from import or default.
    pub fn set_source_animated(
        &mut self,
        frame_images: &[RgbaImage],
        was_animated: bool,
        fps: f32,
    ) {
        self.layer_count = frame_images.len();
        self.was_animated = was_animated;
        self.animated = was_animated;
        self.animation_fps = fps;
        self.anim_current_frame = 0;
        self.anim_playing = false;
        self.frame_thumbnails.clear();
        self.frame_textures.clear();

        for img in frame_images {
            let thumb = create_thumbnail(img, PREVIEW_MAX_SIZE);
            self.frame_thumbnails.push(thumb);
            self.frame_textures.push(None);
        }
    }

    /// Set filename from an existing path (for re-saves)
    pub fn set_from_path(&mut self, path: &std::path::Path) {
        if let Some(stem) = path.file_stem() {
            self.filename = stem.to_string_lossy().to_string();
        }
        if let Some(ext) = path.extension() {
            self.format = match ext.to_string_lossy().to_lowercase().as_str() {
                "png" => SaveFormat::Png,
                "jpg" | "jpeg" => SaveFormat::Jpeg,
                "jxl" if cfg!(not(target_arch = "wasm32")) => SaveFormat::Jxl(Default::default()),
                "avif" if cfg!(not(target_arch = "wasm32")) => SaveFormat::Avif,
                "heic" | "heif" | "hif" if cfg!(not(target_arch = "wasm32")) => SaveFormat::Heic,
                "webp" => SaveFormat::Webp,
                "bmp" => SaveFormat::Bmp,
                "tga" => SaveFormat::Tga,
                "ico" => SaveFormat::Ico,
                "tiff" | "tif" => SaveFormat::Tiff,
                "gif" => SaveFormat::Gif,
                "pfe" => SaveFormat::Pfe,
                _ => SaveFormat::Png,
            };
        }
        self.target_directory = path.parent().map(|p| p.to_path_buf());
        self.needs_preview_update = true;
    }

    pub fn restore_options(
        &mut self,
        options: &std::collections::BTreeMap<String, crate::io::export::ExportOptions>,
    ) {
        self.format_options = options.clone();
        if let Some(saved) = options
            .get(self.format.extension())
            .filter(|o| o.format.extension() == self.format.extension() && o.validate().is_ok())
        {
            self.options = *saved;
        }
    }

    pub fn set_prepared_source(&mut self, prepared: &crate::io::PreparedExportImage) {
        use crate::io::PreparedExportImage;
        fn sample<T: Copy>(source: &[T], width: u32, height: u32, tw: u32, th: u32) -> Vec<T> {
            let mut data = Vec::with_capacity(tw as usize * th as usize * 4);
            for y in 0..th {
                for x in 0..tw {
                    let index = ((u64::from(y) * u64::from(height) / u64::from(th))
                        * u64::from(width)
                        + u64::from(x) * u64::from(width) / u64::from(tw))
                        as usize
                        * 4;
                    data.extend_from_slice(&source[index..index + 4]);
                }
            }
            data
        }
        self.source_has_alpha = prepared.has_alpha();
        self.source_has_color = prepared.has_color();
        self.source_generation = self.source_generation.wrapping_add(1);
        self.needs_preview_update = true;
        let (tw, th) = self
            .source_thumbnail
            .as_ref()
            .map(|t| t.dimensions())
            .unwrap_or((1, 1));
        self.source_depth = match prepared {
            PreparedExportImage::Rgba8(_) => 8,
            PreparedExportImage::Rgba16 { .. } => 16,
            PreparedExportImage::RgbaF32 { .. } => 32,
        };
        self.prepared_thumbnail = match prepared {
            PreparedExportImage::Rgba16 {
                width,
                height,
                pixels,
            } => Some(PreparedExportImage::Rgba16 {
                width: tw,
                height: th,
                pixels: sample(pixels, *width, *height, tw, th),
            }),
            PreparedExportImage::RgbaF32 {
                width,
                height,
                pixels,
            } => Some(PreparedExportImage::RgbaF32 {
                width: tw,
                height: th,
                pixels: sample(pixels, *width, *height, tw, th),
            }),
            PreparedExportImage::Rgba8(_) => None,
        };
    }

    fn effective_options(&self) -> crate::io::export::ExportOptions {
        let mut options = self.options;
        options.animated &= options.format.supports_animation();
        if options.animated && options.format == SaveFormat::Png {
            options.png = Default::default();
        }
        options
    }

    fn preview_key(&self) -> PreviewKey {
        PreviewKey {
            options: self.effective_options(),
            generation: self.source_generation,
            frame: if self.animated && self.format.supports_animation() {
                self.anim_current_frame
            } else {
                0
            },
        }
    }

    /// One active job, a debounced latest request, and source-aware stale-result rejection.
    fn update_preview_if_needed(&mut self, ctx: &egui::Context) {
        let key = self.preview_key();
        let now = ctx.input(|i| i.time);
        if self.needs_preview_update || self.preview_requested.is_none_or(|(last, _)| last != key) {
            self.preview_requested = Some((key, now));
            self.needs_preview_update = false;
            self.preview_error = None;
        }
        if let Some(job) = &self.jxl_preview_job {
            match job.try_recv() {
                Ok((finished, result)) => {
                    self.jxl_preview_job = None;
                    self.preview_active = None;
                    if finished == key {
                        self.preview_completed = Some(finished);
                        match result {
                            Ok(result) => self.apply_preview(ctx, result),
                            Err(error) => self.preview_error = Some(error),
                        }
                    }
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    self.jxl_preview_job = None;
                    if self.preview_active.take() == Some(key) {
                        self.preview_completed = Some(key);
                        self.preview_error = Some(crate::t!("export.preview_failed"));
                    }
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {}
            }
        }
        if self.preview_completed == Some(key) {
            return;
        }
        // Playback uses source thumbnails. Encode the selected frame once paused.
        if self.anim_playing && self.animated {
            return;
        }
        ctx.request_repaint_after(std::time::Duration::from_millis(40));
        if self.jxl_preview_job.is_some() || now - self.preview_requested.unwrap().1 < 0.15 {
            return;
        }
        let thumbnail = if self.animated
            && self.format.supports_animation()
            && !self.frame_thumbnails.is_empty()
        {
            Some(crate::io::PreparedExportImage::Rgba8(
                self.frame_thumbnails[key.frame.min(self.frame_thumbnails.len() - 1)].clone(),
            ))
        } else {
            self.prepared_thumbnail.clone().or_else(|| {
                self.source_thumbnail
                    .clone()
                    .map(crate::io::PreparedExportImage::Rgba8)
            })
        };
        let Some(thumbnail) = thumbnail else {
            return;
        };
        let (tx, rx) = std::sync::mpsc::channel();
        let ctx = ctx.clone();
        crate::par_compat::spawn(move || {
            let result = crate::io::export::preview(thumbnail, key.options);
            let _ = tx.send((key, result));
            ctx.request_repaint();
        });
        self.preview_active = Some(key);
        self.jxl_preview_job = Some(rx);
    }

    fn apply_preview(&mut self, ctx: &egui::Context, result: PreviewResult) {
        let color_image = rgba_to_color_image(&result.preview_image);
        self.encoded_preview = Some(color_image.clone());
        self.preview_texture =
            Some(ctx.load_texture("save_preview", color_image, TextureOptions::LINEAR));
        self.preview_file_size = result.file_size;
        let (fw, fh) = self.source_dimensions;
        let thumb_pixels = self
            .source_thumbnail
            .as_ref()
            .map_or(0, |image| image.width() as u64 * image.height() as u64);
        self.estimated_full_size = if thumb_pixels > 0 {
            (result.file_size as f64 * fw as f64 * fh as f64 / thumb_pixels as f64) as usize
        } else {
            result.file_size
        };
    }

    /// Format file size for display
    fn format_file_size(bytes: usize) -> String {
        if bytes < 1024 {
            format!("{} B", bytes)
        } else if bytes < 1024 * 1024 {
            format!("{:.1} KB", bytes as f64 / 1024.0)
        } else {
            format!("{:.2} MB", bytes as f64 / (1024.0 * 1024.0))
        }
    }

    /// Show the dialog and return SaveAction if user confirms
    pub fn show(&mut self, ctx: &egui::Context) -> Option<SaveAction> {
        use crate::ops::dialogs::{
            DialogColors, accent_separator, contrast_text_color, paint_dialog_header,
        };

        let mut result = None;
        let mut should_close = false;

        if self.open {
            // Update preview before showing UI
            self.update_preview_if_needed(ctx);

            // Animation playback: advance frame if playing
            let show_anim_controls = self.format.supports_animation()
                && self.animated
                && self.frame_thumbnails.len() > 1;

            if show_anim_controls && self.anim_playing {
                let now = ctx.input(|i| i.time);
                let frame_duration = 1.0 / self.animation_fps as f64;
                if now - self.anim_last_frame_time >= frame_duration {
                    self.anim_current_frame =
                        (self.anim_current_frame + 1) % self.frame_thumbnails.len();
                    self.anim_last_frame_time = now;
                }
                ctx.request_repaint();
            }

            let position_id = egui::Id::new("save_as_heading_position");
            let position = ctx
                .data(|d| d.get_temp::<egui::Pos2>(position_id))
                .unwrap_or(ctx.content_rect().center() - egui::vec2(390.0, 290.0));
            let mut moved_position = None;
            let response = crate::ui::polish::window(ctx, "save_file_dialog_internal")
                .title_bar(false)
                .movable(false)
                .current_pos(position)
                .collapsible(false)
                .resizable(true)
                .default_size(egui::vec2(780.0, 580.0))
                .min_size(egui::vec2(280.0, 300.0))
                .default_pos(ctx.content_rect().center() - egui::vec2(390.0, 290.0))
                .max_size(ctx.content_rect().size() - egui::vec2(24.0, 24.0))
                .show(ctx, |ui| {
                    let colors = DialogColors::from_ctx(ctx);

                    // ── Header ──────────────────────────────────────────────
                    let header_min = ui.cursor().min;
                    let header_height = 32.0 * crate::ui::polish::settings(ctx).spacing_scale;
                    let header_rect = egui::Rect::from_min_size(
                        header_min,
                        egui::vec2(
                            (ui.available_width() - header_height * 3.0).max(1.0),
                            header_height,
                        ),
                    );
                    should_close |= paint_dialog_header(
                        ui,
                        &colors,
                        "\u{1F4BE}",
                        &crate::t!("keybind.save_as"),
                    );
                    let heading = ui.interact(
                        header_rect,
                        ui.id().with("save_as_heading_drag"),
                        egui::Sense::drag(),
                    );
                    if heading.dragged() {
                        moved_position = Some(position + ui.input(|i| i.pointer.delta()));
                        ctx.request_repaint();
                    }
                    ui.add_space(8.0);

                    self.show_export_body(ui, ctx, &colors, show_anim_controls);
                    // ── Footer (full-width, below both columns) ───────────────
                    let (enter, esc) = crate::ui::dialogs::core::dialog_footer_keys(ctx);
                    should_close |= esc;
                    accent_separator(ui, &colors);
                    ui.add_space(6.0);
                    ui.horizontal(|ui| {
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui.button(crate::t!("ui.cancel")).clicked() {
                                should_close = true;
                            }
                            let save_btn = egui::Button::new(
                                egui::RichText::new(crate::t!("ui.save"))
                                    .color(contrast_text_color(colors.accent))
                                    .strong(),
                            )
                            .fill(colors.accent);
                            let heic_size_ok = self.format != SaveFormat::Heic
                                || crate::io::heic::validate_dimensions(
                                    self.source_dimensions.0,
                                    self.source_dimensions.1,
                                )
                                .is_ok();
                            let save_response = ui.add_enabled(heic_size_ok, save_btn);
                            #[cfg(test)]
                            ctx.data_mut(|d| {
                                d.insert_temp(
                                    egui::Id::new("save_as_save_rect"),
                                    save_response.rect,
                                )
                            });
                            let save_clicked = (save_response.clicked() || enter) && heic_size_ok;
                            #[cfg(not(target_arch = "wasm32"))]
                            let needs_jxl = matches!(self.format, SaveFormat::Jxl(_))
                                && !crate::services::optional_packs::is_present(
                                    crate::services::optional_packs::Pack::Jxl,
                                );
                            #[cfg(target_arch = "wasm32")]
                            let needs_jxl = false;
                            #[cfg(not(target_arch = "wasm32"))]
                            let needs_avif = self.format == SaveFormat::Avif
                                && !crate::services::optional_packs::is_present(
                                    crate::services::optional_packs::Pack::Avif,
                                );
                            #[cfg(target_arch = "wasm32")]
                            let needs_avif = self.format == SaveFormat::Avif;
                            #[cfg(not(target_arch = "wasm32"))]
                            if save_clicked && needs_avif {
                                crate::services::optional_packs::request_avif();
                            }
                            #[cfg(not(target_arch = "wasm32"))]
                            if save_clicked && needs_jxl {
                                crate::services::optional_packs::request_jxl();
                            }
                            #[cfg(not(target_arch = "wasm32"))]
                            let needs_heic = self.format == SaveFormat::Heic
                                && !crate::io::heic::capabilities().encode;
                            #[cfg(target_arch = "wasm32")]
                            let needs_heic = self.format == SaveFormat::Heic;
                            #[cfg(not(target_arch = "wasm32"))]
                            if save_clicked && !needs_jxl && !needs_avif && !needs_heic {
                                let full_filename =
                                    format!("{}.{}", self.filename, self.format.extension());
                                // Web has no native save-location picker: downloads always
                                // go to the browser's default download folder, so we skip
                                // straight to a virtual path carrying just the file name.
                                #[cfg(target_arch = "wasm32")]
                                let picked_path = Some(std::path::PathBuf::from(&full_filename));
                                #[cfg(not(target_arch = "wasm32"))]
                                let picked_path = {
                                    let extensions = if self.format == SaveFormat::Heic {
                                        vec!["heic", "heif"]
                                    } else {
                                        vec![self.format.extension()]
                                    };
                                    let mut dialog = crate::services::file_dialogs::dialog(
                                        crate::services::file_dialogs::Purpose::Documents,
                                        None,
                                    )
                                    .set_file_name(&full_filename)
                                    .add_filter(
                                        crate::i18n::translate_english(&self.format.label()),
                                        &extensions,
                                    );
                                    if let Some(dir) = &self.target_directory {
                                        dialog = dialog.set_directory(dir);
                                    }
                                    dialog.save_file()
                                };
                                if let Some(path) = picked_path {
                                    result = Some(SaveAction {
                                        path,
                                        options: self.effective_options(),
                                    });
                                    should_close = true;
                                }
                            }
                        });
                    });
                });
            if let Some(response) = response {
                #[cfg(test)]
                ctx.data_mut(|d| {
                    d.insert_temp(egui::Id::new("save_as_window_rect"), response.response.rect)
                });
                ctx.data_mut(|d| {
                    d.insert_temp(
                        position_id,
                        moved_position.unwrap_or(response.response.rect.min),
                    )
                });
            }
        }

        if should_close {
            self.open = false;
            self.anim_playing = false;
            self.preview_texture = None;
            self.frame_textures.clear();
        }

        result
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod jxl_preview_tests {
    use super::*;

    fn tick(dialog: &mut SaveFileDialog, ctx: &egui::Context, time: f64) {
        let _ = ctx.run_ui(
            egui::RawInput {
                time: Some(time),
                ..Default::default()
            },
            |_| dialog.update_preview_if_needed(ctx),
        );
    }
    #[test]
    fn asynchronous_preview_discards_stale_options_and_encodes_current_settings() {
        if let Some(path) = std::env::var_os("PAINTFE_TEST_JXL_PACK") {
            crate::services::optional_packs::install(
                std::path::Path::new(&path),
                crate::services::optional_packs::Pack::Jxl,
            )
            .unwrap();
        }
        let ctx = egui::Context::default();
        let image = RgbaImage::from_pixel(12, 9, image::Rgba([83, 142, 217, 128]));
        let mut dialog = SaveFileDialog {
            options: crate::io::export::ExportOptions {
                format: SaveFormat::Jxl(crate::io::jxl::JxlOptions {
                    lossless: true,
                    effort: 1,
                }),
                ..Default::default()
            },
            ..Default::default()
        };
        dialog.set_source_image(&image);
        let (tx, rx) = std::sync::mpsc::channel();
        let stale = PreviewKey {
            options: Default::default(),
            generation: 0,
            frame: 0,
        };
        tx.send((
            stale,
            Ok(PreviewResult {
                file_size: 1,
                preview_image: RgbaImage::new(1, 1),
            }),
        ))
        .unwrap();
        dialog.jxl_preview_job = Some(rx);
        tick(&mut dialog, &ctx, 0.0);
        assert_eq!(dialog.preview_file_size, 0, "stale preview displayed");
        assert!(
            dialog.jxl_preview_job.is_none(),
            "debounce must finish before starting another job"
        );
        tick(&mut dialog, &ctx, 0.2);
        assert!(dialog.jxl_preview_job.is_some());
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while dialog.jxl_preview_job.is_some() {
            assert!(std::time::Instant::now() < deadline, "preview timed out");
            std::thread::sleep(std::time::Duration::from_millis(2));
            tick(&mut dialog, &ctx, 0.4);
        }
        assert!(dialog.preview_error.is_none(), "{:?}", dialog.preview_error);
        let expected = crate::io::export::preview(
            crate::io::PreparedExportImage::Rgba8(image),
            dialog.options,
        )
        .unwrap();
        assert_eq!(dialog.preview_file_size, expected.file_size);
        assert_eq!(dialog.encoded_preview.as_ref().unwrap().size, [12, 9]);
        assert_eq!(dialog.preview_completed, Some(dialog.preview_key()));
    }
    #[test]
    fn reset_keeps_one_running_job_but_rejects_its_old_source() {
        let mut dialog = SaveFileDialog::default();
        let previous = dialog.preview_key();
        let (tx, rx) = std::sync::mpsc::channel();
        dialog.jxl_preview_job = Some(rx);
        dialog.reset();
        assert!(
            dialog.jxl_preview_job.is_some(),
            "dropping a receiver would orphan a running encoder"
        );
        tx.send((
            previous,
            Ok(PreviewResult {
                file_size: 1,
                preview_image: RgbaImage::new(1, 1),
            }),
        ))
        .unwrap();
        tick(&mut dialog, &egui::Context::default(), 0.0);
        assert!(dialog.jxl_preview_job.is_none());
        assert!(dialog.encoded_preview.is_none());
    }
    #[test]
    fn png_and_tiff_changes_invalidate_keys_and_failed_preview_remains_marked() {
        let mut dialog = SaveFileDialog::default();
        let png = dialog.preview_key();
        dialog.png.mode = crate::io::export::png::Mode::Gray;
        assert_ne!(png, dialog.preview_key());
        dialog.format = SaveFormat::Tiff;
        let tiff = dialog.preview_key();
        dialog.tiff_compression = TiffCompression::Lzw;
        assert_ne!(tiff, dialog.preview_key());
        let (tx, rx) = std::sync::mpsc::channel();
        tx.send((dialog.preview_key(), Err("test encoder failure".into())))
            .unwrap();
        dialog.jxl_preview_job = Some(rx);
        tick(&mut dialog, &egui::Context::default(), 0.0);
        assert_eq!(
            dialog.preview_error.as_deref(),
            Some("test encoder failure")
        );
        assert_eq!(dialog.preview_completed, Some(dialog.preview_key()));
    }
}

#[cfg(test)]
mod preview_mapping_tests {
    use super::clipped_preview_rects;

    fn assert_close(actual: f32, expected: f32) {
        assert!((actual - expected).abs() < 0.001, "{actual} != {expected}");
    }

    #[test]
    fn wide_image_leaves_vertical_letterbox_unpainted() {
        let panel = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(320.0, 320.0));
        let (destination, uv) = clipped_preview_rects(
            panel,
            egui::vec2(320.0, 160.0),
            egui::pos2(0.0, -80.0),
            egui::vec2(320.0, 320.0),
        )
        .unwrap();

        assert_close(destination.min.y, 80.0);
        assert_close(destination.max.y, 240.0);
        assert_eq!(
            uv,
            egui::Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1.0, 1.0))
        );
    }

    #[test]
    fn tall_image_leaves_horizontal_letterbox_unpainted() {
        let panel = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(320.0, 320.0));
        let (destination, uv) = clipped_preview_rects(
            panel,
            egui::vec2(160.0, 320.0),
            egui::pos2(-80.0, 0.0),
            egui::vec2(320.0, 320.0),
        )
        .unwrap();

        assert_close(destination.min.x, 80.0);
        assert_close(destination.max.x, 240.0);
        assert_eq!(
            uv,
            egui::Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1.0, 1.0))
        );
    }

    #[test]
    fn zoomed_view_maps_to_bounded_uvs() {
        let panel = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(320.0, 320.0));
        let (destination, uv) = clipped_preview_rects(
            panel,
            egui::vec2(512.0, 256.0),
            egui::pos2(64.0, 32.0),
            egui::vec2(160.0, 160.0),
        )
        .unwrap();

        assert_eq!(destination, panel);
        assert_close(uv.min.x, 0.125);
        assert_close(uv.min.y, 0.125);
        assert_close(uv.max.x, 0.4375);
        assert_close(uv.max.y, 0.75);
    }
}

#[cfg(test)]
mod save_as_controls_tests {
    use super::*;

    #[test]
    fn save_as_corner_drag_changes_both_dimensions_without_hiding_the_footer() {
        crate::i18n::init();
        let ctx = egui::Context::default();
        let mut fonts = egui::FontDefinitions::default();
        fonts.families.insert(
            egui::FontFamily::Name("WidgetTitle".into()),
            fonts.families[&egui::FontFamily::Proportional].clone(),
        );
        ctx.set_fonts(fonts);
        let mut dialog = SaveFileDialog {
            open: true,
            ..Default::default()
        };
        dialog.set_source_image(&RgbaImage::from_pixel(
            32,
            24,
            image::Rgba([20, 80, 180, 127]),
        ));
        for _ in 0..4 {
            frame(&ctx, &mut dialog, Vec::new());
        }
        let id = egui::Id::new("save_as_window_rect");
        let before = ctx.data(|d| d.get_temp::<egui::Rect>(id)).unwrap();
        drag(
            &ctx,
            &mut dialog,
            before.right_bottom() - egui::vec2(3.0, 3.0),
            egui::vec2(-180.0, -130.0),
        );
        for _ in 0..3 {
            frame(&ctx, &mut dialog, Vec::new());
        }
        let after = ctx.data(|d| d.get_temp::<egui::Rect>(id)).unwrap();
        assert!(
            after.width() < before.width() - 80.0,
            "width did not shrink: {before:?} -> {after:?}"
        );
        assert!(
            after.height() < before.height() - 60.0,
            "height did not shrink: {before:?} -> {after:?}"
        );
        let save = ctx
            .data(|d| d.get_temp::<egui::Rect>(egui::Id::new("save_as_save_rect")))
            .unwrap();
        assert!(after.contains_rect(save), "footer escaped resized window");
    }

    #[test]
    fn save_as_footer_remains_visible_in_compact_and_large_text_layouts() {
        crate::i18n::init();
        for (width, height, scale, dark) in [
            (1280.0, 900.0, 1.0, false),
            (700.0, 500.0, 1.0, true),
            (420.0, 620.0, 1.0, false),
            (900.0, 360.0, 1.0, true),
            (1024.0, 768.0, 1.5, true),
        ] {
            let ctx = egui::Context::default();
            ctx.set_visuals(if dark {
                egui::Visuals::dark()
            } else {
                egui::Visuals::light()
            });
            let mut fonts = egui::FontDefinitions::default();
            fonts.families.insert(
                egui::FontFamily::Name("WidgetTitle".into()),
                fonts.families[&egui::FontFamily::Proportional].clone(),
            );
            ctx.set_fonts(fonts);
            ctx.all_styles_mut(|style| {
                for font in style.text_styles.values_mut() {
                    font.size *= scale;
                }
            });
            let mut dialog = SaveFileDialog {
                open: true,
                ..Default::default()
            };
            dialog.set_source_image(&RgbaImage::from_pixel(
                32,
                24,
                image::Rgba([20, 80, 180, 127]),
            ));
            for _ in 0..4 {
                let _ = ctx.run_ui(
                    egui::RawInput {
                        screen_rect: Some(egui::Rect::from_min_size(
                            egui::Pos2::ZERO,
                            egui::vec2(width, height),
                        )),
                        ..Default::default()
                    },
                    |_| {
                        dialog.show(&ctx);
                    },
                );
            }
            let save = ctx
                .data(|d| d.get_temp::<egui::Rect>(egui::Id::new("save_as_save_rect")))
                .unwrap();
            assert!(
                save.bottom() <= height && save.right() <= width && save.left() >= 0.0,
                "save button outside {width}x{height}: {save:?}"
            );
            let zoom = ctx
                .data(|d| d.get_temp::<egui::Rect>(egui::Id::new("save_as_zoom_rect")))
                .unwrap();
            assert!(
                zoom.right() <= width,
                "preview controls overflowed: {zoom:?}"
            );
        }
    }

    #[test]
    fn format_presets_restore_without_changing_the_requested_format() {
        let mut dialog = SaveFileDialog::default();
        let png = crate::io::export::ExportOptions {
            png: crate::io::export::png::PngOptions {
                mode: crate::io::export::png::Mode::GrayAlpha,
                ..Default::default()
            },
            ..Default::default()
        };
        let options = std::collections::BTreeMap::from([("png".into(), png)]);
        dialog.restore_options(&options);
        assert_eq!(dialog.options, png);
        dialog.format = SaveFormat::Jpeg;
        dialog.restore_options(&options);
        assert_eq!(dialog.format, SaveFormat::Jpeg);
    }

    fn frame(ctx: &egui::Context, dialog: &mut SaveFileDialog, events: Vec<egui::Event>) {
        let _ = ctx.run_ui(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(1000.0, 800.0),
                )),
                events,
                ..Default::default()
            },
            |_| {
                dialog.show(ctx);
            },
        );
    }
    fn drag(
        ctx: &egui::Context,
        dialog: &mut SaveFileDialog,
        start: egui::Pos2,
        delta: egui::Vec2,
    ) {
        frame(ctx, dialog, vec![egui::Event::PointerMoved(start)]);
        frame(
            ctx,
            dialog,
            vec![egui::Event::PointerButton {
                pos: start,
                button: egui::PointerButton::Primary,
                pressed: true,
                modifiers: egui::Modifiers::NONE,
            }],
        );
        frame(ctx, dialog, vec![egui::Event::PointerMoved(start + delta)]);
        frame(
            ctx,
            dialog,
            vec![egui::Event::PointerButton {
                pos: start + delta,
                button: egui::PointerButton::Primary,
                pressed: false,
                modifiers: egui::Modifiers::NONE,
            }],
        );
    }

    #[test]
    fn issue_185_save_as_zoom_owns_drag_and_heading_still_moves() {
        let ctx = egui::Context::default();
        let mut fonts = egui::FontDefinitions::default();
        fonts.families.insert(
            egui::FontFamily::Name("WidgetTitle".into()),
            fonts.families[&egui::FontFamily::Proportional].clone(),
        );
        ctx.set_fonts(fonts);
        let mut dialog = SaveFileDialog {
            open: true,
            preview_zoom: 1.0,
            ..Default::default()
        };
        for _ in 0..3 {
            frame(&ctx, &mut dialog, Vec::new());
        }
        let position_id = egui::Id::new("save_as_heading_position");
        let before = ctx.data(|d| d.get_temp::<egui::Pos2>(position_id)).unwrap();
        let zoom_rect = ctx
            .data(|d| d.get_temp::<egui::Rect>(egui::Id::new("save_as_zoom_rect")))
            .unwrap();
        drag(&ctx, &mut dialog, zoom_rect.center(), egui::vec2(32.0, 0.0));
        assert!(
            dialog.preview_zoom > 1.0,
            "zoom control did not receive drag"
        );
        assert_eq!(
            ctx.data(|d| d.get_temp::<egui::Pos2>(position_id)),
            Some(before),
            "percentage drag moved dialog"
        );
        drag(
            &ctx,
            &mut dialog,
            before + egui::vec2(100.0, 20.0),
            egui::vec2(25.0, 15.0),
        );
        let after = ctx.data(|d| d.get_temp::<egui::Pos2>(position_id)).unwrap();
        assert!((after - before - egui::vec2(25.0, 15.0)).length() < 1.0);
    }
}

#[cfg(test)]
mod shared_new_image_tests {
    use super::*;

    #[test]
    fn new_image_dimensions_use_shared_expressions_and_keep_linked_values() {
        let ctx = egui::Context::default();
        let mut fonts = egui::FontDefinitions::default();
        fonts.families.insert(
            egui::FontFamily::Name("WidgetTitle".into()),
            fonts.families[&egui::FontFamily::Proportional].clone(),
        );
        ctx.set_fonts(fonts);
        let mut dialog = NewFileDialog::default();
        dialog.open_dialog();
        fn frame(ctx: &egui::Context, dialog: &mut NewFileDialog, events: Vec<egui::Event>) {
            let _ = ctx.run_ui(
                egui::RawInput {
                    events,
                    ..Default::default()
                },
                |_| {
                    dialog.show(ctx);
                },
            );
        }
        for _ in 0..3 {
            frame(&ctx, &mut dialog, vec![]);
        }
        frame(&ctx, &mut dialog, vec![egui::Event::Text("400/2".into())]);
        frame(
            &ctx,
            &mut dialog,
            vec![egui::Event::Key {
                key: egui::Key::Tab,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: egui::Modifiers::NONE,
            }],
        );
        assert_eq!(dialog.width, 200.0);
        assert_eq!(dialog.height, 150.0);
        assert_eq!(dialog.preview_pixels(), (200, 150));
    }
}
