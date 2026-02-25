use eframe::egui;
use egui::{Color32, ColorImage, TextureHandle, TextureOptions};
use image::codecs::bmp::BmpEncoder;
use image::codecs::jpeg::JpegEncoder;
use image::codecs::png::PngEncoder;
use image::codecs::tiff::TiffEncoder;
use image::imageops::FilterType;
use image::{DynamicImage, RgbaImage};
use std::io::Cursor;
use std::path::PathBuf;

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
    unit: SizeUnit,
    ppi: f32,
    lock_aspect_ratio: bool,
    aspect_ratio: f32,
    preset: SizePreset,
}

impl Default for NewFileDialog {
    fn default() -> Self {
        Self {
            open: false,
            width: 800.0,
            height: 600.0,
            unit: SizeUnit::Pixels,
            ppi: 72.0,
            lock_aspect_ratio: false,
            aspect_ratio: 800.0 / 600.0,
            preset: SizePreset::Size800x600,
        }
    }
}

impl NewFileDialog {
    /// Pre-populate width and height from a clipboard image if available.
    /// Called when the dialog opens so newly-created canvases match the
    /// clipboard image dimensions by default.
    pub fn load_clipboard_dimensions(&mut self) {
        if let Some((w, h)) = crate::ops::clipboard::get_clipboard_image_dimensions() {
            self.width = w as f32;
            self.height = h as f32;
            self.aspect_ratio = self.width / self.height.max(1.0);
            self.preset = SizePreset::Custom;
            self.unit = SizeUnit::Pixels;
        }
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
            DialogColors, accent_separator, paint_dialog_header, section_label,
        };

        let mut result = None;
        let mut should_close = false;

        if self.open {
            // Keyboard: Enter = Create, Esc = Cancel
            let enter = ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::Enter));
            let esc = ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::Escape));
            if enter {
                result = Some(self.to_pixels());
                should_close = true;
            }
            if esc {
                should_close = true;
            }

            egui::Window::new("new_file_dialog_internal")
                .title_bar(false)
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .show(ctx, |ui| {
                    ui.set_min_width(360.0);

                    let colors = DialogColors::from_ctx(ctx);
                    paint_dialog_header(ui, &colors, "\u{1F4C4}", &t!("dialog.new_file"));
                    ui.add_space(6.0);

                    // ── Preset row ──────────────────────────────────────────
                    section_label(ui, &colors, &t!("common.preset"));
                    egui::Grid::new("new_file_preset_grid")
                        .num_columns(2)
                        .min_col_width(80.0)
                        .spacing([8.0, 4.0])
                        .show(ui, |ui| {
                            ui.label(t!("common.preset"));
                            egui::ComboBox::from_id_source("preset_combo")
                                .width(224.0)
                                .selected_text(self.preset.label())
                                .show_ui(ui, |ui| {
                                    for preset in SizePreset::all() {
                                        if ui
                                            .selectable_value(
                                                &mut self.preset,
                                                *preset,
                                                preset.label(),
                                            )
                                            .clicked()
                                            && let Some((w, h, ppi)) = preset.dimensions()
                                        {
                                            self.width = w as f32;
                                            self.height = h as f32;
                                            self.ppi = ppi;
                                            self.unit = SizeUnit::Pixels;
                                            self.aspect_ratio = self.width / self.height;
                                        }
                                    }
                                });
                            ui.end_row();
                        });

                    // ── Dimensions grid ──────────────────────────────────────
                    accent_separator(ui, &colors);
                    section_label(ui, &colors, &t!("dialog.resize_image.dimensions"));

                    egui::Grid::new("new_file_dims_grid")
                        .num_columns(3)
                        .min_col_width(80.0)
                        .spacing([8.0, 6.0])
                        .show(ui, |ui| {
                            // Width row
                            ui.label(t!("common.width"));
                            let old_width = self.width;
                            if ui
                                .add(
                                    egui::DragValue::new(&mut self.width)
                                        .speed(1.0)
                                        .clamp_range(1.0..=20000.0),
                                )
                                .changed()
                            {
                                self.preset = SizePreset::Custom;
                                if self.lock_aspect_ratio && old_width > 0.0 {
                                    self.height = self.width / self.aspect_ratio;
                                } else {
                                    self.aspect_ratio = self.width / self.height.max(1.0);
                                }
                            }
                            ui.label(self.unit.label());
                            ui.end_row();

                            // Height row
                            ui.label(t!("common.height"));
                            let old_height = self.height;
                            if ui
                                .add(
                                    egui::DragValue::new(&mut self.height)
                                        .speed(1.0)
                                        .clamp_range(1.0..=20000.0),
                                )
                                .changed()
                            {
                                self.preset = SizePreset::Custom;
                                if self.lock_aspect_ratio && old_height > 0.0 {
                                    self.width = self.height * self.aspect_ratio;
                                } else {
                                    self.aspect_ratio = self.width / self.height.max(1.0);
                                }
                            }
                            ui.label(self.unit.label());
                            ui.end_row();

                            // Lock aspect ratio (below Height)
                            ui.label("");
                            let lock_icon = if self.lock_aspect_ratio {
                                "\u{1F517}"
                            } else {
                                "\u{25CB}"
                            };
                            let lock_text =
                                format!("{} {}", lock_icon, t!("common.lock_aspect_ratio"));
                            if ui
                                .selectable_label(self.lock_aspect_ratio, lock_text)
                                .clicked()
                            {
                                self.lock_aspect_ratio = !self.lock_aspect_ratio;
                                if self.lock_aspect_ratio {
                                    self.aspect_ratio = self.width / self.height.max(1.0);
                                }
                            }
                            ui.label("");
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
                            egui::ComboBox::from_id_source("units_combo")
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
                                        egui::DragValue::new(&mut self.ppi)
                                            .speed(1.0)
                                            .clamp_range(1.0..=1200.0),
                                    );
                                    ui.label(t!("unit.ppi"));
                                });
                                ui.end_row();
                            }
                        });

                    // ── Final pixel size info ────────────────────────────────
                    ui.add_space(4.0);
                    let (px_w, px_h) = self.to_pixels();
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
                                    .color(Color32::WHITE)
                                    .strong(),
                            )
                            .fill(colors.accent);
                            if ui.add(create_btn).clicked() {
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

        self.width = new_w;
        self.height = new_h;
    }
}

// ============================================================================
// IMAGE FORMAT ENUM
// ============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum SaveFormat {
    #[default]
    Png,
    Jpeg,
    Webp,
    Bmp,
    Tga,
    Ico,
    Tiff,
    Gif,
    Pfe,
}

/// Compression options for TIFF format
#[derive(Clone, Copy, Debug, PartialEq, Default)]
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
        matches!(self, SaveFormat::Jpeg | SaveFormat::Webp)
    }

    /// Returns true if this format supports animated (multi-frame) output.
    pub fn supports_animation(&self) -> bool {
        matches!(self, SaveFormat::Png | SaveFormat::Gif)
    }

    fn all() -> &'static [SaveFormat] {
        &[
            SaveFormat::Pfe,
            SaveFormat::Png,
            SaveFormat::Jpeg,
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
) -> Option<PreviewResult> {
    let mut buffer = Vec::new();

    match format {
        SaveFormat::Png => {
            // PNG is lossless, just get the size
            let mut cursor = Cursor::new(&mut buffer);
            let encoder = PngEncoder::new(&mut cursor);
            #[allow(deprecated)]
            encoder
                .encode(
                    image.as_raw(),
                    image.width(),
                    image.height(),
                    image::ColorType::Rgba8,
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
                    image::ColorType::Rgb8,
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
        SaveFormat::Webp => {
            // WebP - use DynamicImage save for encoding
            // Note: image crate's webp may not support quality directly
            let dyn_img = DynamicImage::ImageRgba8(image.clone());
            let mut cursor = Cursor::new(&mut buffer);
            dyn_img
                .write_to(&mut cursor, image::ImageOutputFormat::WebP)
                .ok()?;

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
                    image::ColorType::Rgba8,
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
                .encode(
                    image.as_raw(),
                    image.width(),
                    image.height(),
                    image::ColorType::Rgba8,
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
                .encode(
                    image.as_raw(),
                    image.width(),
                    image.height(),
                    image::ColorType::Rgba8,
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
    ColorImage { size, pixels }
}

// ============================================================================
// SAVE ACTION
// ============================================================================

#[derive(Clone, Debug)]
pub struct SaveAction {
    pub path: PathBuf,
    pub format: SaveFormat,
    pub quality: u8,
    pub tiff_compression: TiffCompression,
    /// Whether to save as animated (multi-frame)
    pub animated: bool,
    /// Animation FPS (1-60)
    pub animation_fps: f32,
    /// GIF max colors (2-256)
    pub gif_colors: u16,
    /// GIF dithering enabled
    pub gif_dither: bool,
}

// ============================================================================
// SAVE FILE DIALOG
// ============================================================================

/// Maximum size for preview thumbnail (keeps preview generation fast)
const PREVIEW_MAX_SIZE: u32 = 256;

pub struct SaveFileDialog {
    pub open: bool,
    filename: String,
    format: SaveFormat,
    quality: u8,
    tiff_compression: TiffCompression,
    target_directory: Option<PathBuf>,

    // Preview state
    source_thumbnail: Option<RgbaImage>,
    source_dimensions: (u32, u32), // Full source image dimensions
    preview_texture: Option<TextureHandle>,
    preview_file_size: usize,   // Thumbnail-based file size
    estimated_full_size: usize, // Estimated full-resolution file size
    last_preview_format: SaveFormat,
    last_preview_quality: u8,
    needs_preview_update: bool,

    // Animation options
    animated: bool,     // "Animated" checkbox
    animation_fps: f32, // FPS slider (1–60, default 10)
    gif_colors: u16,    // GIF color count (2–256, default 256)
    gif_dither: bool,   // GIF dithering toggle
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
            open: false,
            filename: "untitled".to_string(),
            format: SaveFormat::Png,
            quality: 90,
            tiff_compression: TiffCompression::None,
            target_directory: None,
            source_thumbnail: None,
            source_dimensions: (0, 0),
            preview_texture: None,
            preview_file_size: 0,
            estimated_full_size: 0,
            last_preview_format: SaveFormat::Png,
            last_preview_quality: 90,
            needs_preview_update: true,
            // Animation
            animated: false,
            animation_fps: 10.0,
            gif_colors: 256,
            gif_dither: true,
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

impl SaveFileDialog {
    /// Reset dialog state for a fresh "Save As"
    pub fn reset(&mut self) {
        self.filename = "untitled".to_string();
        self.target_directory = None;
        self.source_thumbnail = None;
        self.source_dimensions = (0, 0);
        self.preview_texture = None;
        self.preview_file_size = 0;
        self.estimated_full_size = 0;
        self.needs_preview_update = true;
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
        self.source_dimensions = (image.width(), image.height());
        self.source_thumbnail = Some(create_thumbnail(image, PREVIEW_MAX_SIZE));
        self.needs_preview_update = true;
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

        // Set the composite preview to the first frame
        if let Some(first) = frame_images.first() {
            self.source_dimensions = (first.width(), first.height());
            self.source_thumbnail = Some(create_thumbnail(first, PREVIEW_MAX_SIZE));
            self.needs_preview_update = true;
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
                "webp" => SaveFormat::Webp,
                "bmp" => SaveFormat::Bmp,
                "tga" => SaveFormat::Tga,
                "ico" => SaveFormat::Ico,
                "tiff" | "tif" => SaveFormat::Tiff,
                "gif" => SaveFormat::Gif,
                _ => SaveFormat::Png,
            };
        }
        self.target_directory = path.parent().map(|p| p.to_path_buf());
        self.needs_preview_update = true;
    }

    /// Update preview if settings have changed
    fn update_preview_if_needed(&mut self, ctx: &egui::Context) {
        // Check if we need to regenerate
        let settings_changed = self.format != self.last_preview_format
            || (self.format.supports_quality() && self.quality != self.last_preview_quality);

        if !self.needs_preview_update && !settings_changed {
            return;
        }

        if let Some(thumbnail) = &self.source_thumbnail {
            let thumb_w = thumbnail.width();
            let thumb_h = thumbnail.height();
            if let Some(preview_result) = generate_preview(thumbnail, self.format, self.quality) {
                // Update texture
                let color_image = rgba_to_color_image(&preview_result.preview_image);
                self.preview_texture =
                    Some(ctx.load_texture("save_preview", color_image, TextureOptions::LINEAR));
                self.preview_file_size = preview_result.file_size;

                // Estimate full-resolution file size by scaling from thumbnail ratio
                let (fw, fh) = self.source_dimensions;
                let full_pixels = fw as f64 * fh as f64;
                let thumb_pixels = thumb_w as f64 * thumb_h as f64;
                if thumb_pixels > 0.0 {
                    let ratio = full_pixels / thumb_pixels;
                    self.estimated_full_size = (preview_result.file_size as f64 * ratio) as usize;
                } else {
                    self.estimated_full_size = preview_result.file_size;
                }
            }
        }

        self.last_preview_format = self.format;
        self.last_preview_quality = self.quality;
        self.needs_preview_update = false;
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
            DialogColors, accent_separator, paint_dialog_header, section_label,
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

            // Keyboard: Enter = Save (opens native picker), Esc = Cancel
            let enter = ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::Enter));
            let esc = ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::Escape));
            if esc {
                should_close = true;
            }

            egui::Window::new("save_file_dialog_internal")
                .title_bar(false)
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .show(ctx, |ui| {
                    let colors = DialogColors::from_ctx(ctx);
                    ui.set_min_width(560.0);

                    // ── Header ──────────────────────────────────────────────
                    paint_dialog_header(ui, &colors, "\u{1F4BE}", "Save As");
                    ui.add_space(8.0);

                    // ── Two-column layout ────────────────────────────────────
                    ui.horizontal(|ui| {
                        // ── LEFT: Preview ────────────────────────────────────
                        ui.vertical(|ui| {
                            ui.set_width(PREVIEW_MAX_SIZE as f32);

                            let preview_size = egui::vec2(PREVIEW_MAX_SIZE as f32, PREVIEW_MAX_SIZE as f32);
                            let (rect, _response) = ui.allocate_exact_size(preview_size, egui::Sense::hover());

                            // Checkerboard background
                            let painter = ui.painter_at(rect);
                            let grid_size = 8.0;
                            let light = Color32::from_gray(200);
                            let dark  = Color32::from_gray(160);
                            for y in 0..((rect.height() / grid_size).ceil() as i32) {
                                for x in 0..((rect.width() / grid_size).ceil() as i32) {
                                    let color = if (x + y) % 2 == 0 { light } else { dark };
                                    let cell_rect = egui::Rect::from_min_size(
                                        rect.min + egui::vec2(x as f32 * grid_size, y as f32 * grid_size),
                                        egui::vec2(grid_size, grid_size),
                                    ).intersect(rect);
                                    painter.rect_filled(cell_rect, 0.0, color);
                                }
                            }

                            // Determine which texture to show
                            let display_texture = if show_anim_controls {
                                let idx = self.anim_current_frame.min(self.frame_thumbnails.len().saturating_sub(1));
                                if self.frame_textures[idx].is_none() {
                                    let color_image = rgba_to_color_image(&self.frame_thumbnails[idx]);
                                    self.frame_textures[idx] = Some(ctx.load_texture(
                                        format!("frame_preview_{}", idx),
                                        color_image,
                                        TextureOptions::LINEAR,
                                    ));
                                }
                                self.frame_textures[idx].as_ref()
                            } else {
                                self.preview_texture.as_ref()
                            };

                            // Centered preview image
                            if let Some(texture) = display_texture {
                                let tex_size = texture.size_vec2();
                                let scale = (preview_size.x / tex_size.x).min(preview_size.y / tex_size.y).min(1.0);
                                let scaled_size = tex_size * scale;
                                let offset = (preview_size - scaled_size) / 2.0;
                                let image_rect = egui::Rect::from_min_size(rect.min + offset, scaled_size);
                                painter.image(
                                    texture.id(),
                                    image_rect,
                                    egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                                    Color32::WHITE,
                                );
                            }

                            // Animation playback controls
                            if show_anim_controls {
                                ui.add_space(4.0);
                                ui.horizontal(|ui| {
                                    if ui.small_button("\u{23EE}").clicked() {
                                        self.anim_current_frame = 0;
                                        self.anim_playing = false;
                                    }
                                    if ui.small_button("\u{23EA}").clicked() {
                                        if self.anim_current_frame > 0 {
                                            self.anim_current_frame -= 1;
                                        } else {
                                            self.anim_current_frame = self.frame_thumbnails.len() - 1;
                                        }
                                        self.anim_playing = false;
                                    }
                                    let play_label = if self.anim_playing { "\u{23F8}" } else { "\u{25B6}" };
                                    if ui.small_button(play_label).clicked() {
                                        self.anim_playing = !self.anim_playing;
                                        if self.anim_playing {
                                            self.anim_last_frame_time = ctx.input(|i| i.time);
                                        }
                                    }
                                    if ui.small_button("\u{23E9}").clicked() {
                                        self.anim_current_frame = (self.anim_current_frame + 1) % self.frame_thumbnails.len();
                                        self.anim_playing = false;
                                    }
                                    if ui.small_button("\u{23ED}").clicked() {
                                        self.anim_current_frame = self.frame_thumbnails.len() - 1;
                                        self.anim_playing = false;
                                    }
                                });
                                ui.small(format!("Frame {} / {}", self.anim_current_frame + 1, self.frame_thumbnails.len()));
                            } else {
                                ui.add_space(4.0);
                                let size_text = format!("Est. {}", Self::format_file_size(self.estimated_full_size));
                                ui.label(egui::RichText::new(size_text).size(11.0).color(colors.text_muted));
                            }

                            let dim_text = format!("{}×{}", self.source_dimensions.0, self.source_dimensions.1);
                            ui.label(egui::RichText::new(dim_text).size(11.0).color(colors.text_muted));
                        });

                        ui.add_space(12.0);
                        ui.separator();
                        ui.add_space(8.0);

                        // ── RIGHT: Format settings ────────────────────────────
                        ui.vertical(|ui| {
                            ui.set_width(240.0);

                            // ── FORMAT ────────────────────────────────────────
                            section_label(ui, &colors, "FORMAT");
                            egui::ComboBox::from_id_source("format_combo")
                                .width(220.0)
                                .selected_text(self.format.label())
                                .show_ui(ui, |ui| {
                                    for format in SaveFormat::all() {
                                        ui.selectable_value(&mut self.format, *format, format.label());
                                    }
                                });

                            // ── QUALITY (JPEG / WebP) ─────────────────────────
                            if self.format.supports_quality() {
                                accent_separator(ui, &colors);
                                section_label(ui, &colors, "QUALITY");
                                if ui.add(egui::Slider::new(&mut self.quality, 1..=100).suffix("%")).changed() {
                                    ctx.request_repaint();
                                }
                                let hint = match self.quality {
                                    1..=30 => "Very Low",
                                    31..=50 => "Low",
                                    51..=70 => "Medium",
                                    71..=85 => "Good",
                                    86..=95 => "High",
                                    _ => "Maximum",
                                };
                                ui.label(egui::RichText::new(hint).size(11.0).color(colors.text_muted));
                            }

                            // ── COMPRESSION (TIFF) ────────────────────────────
                            if self.format == SaveFormat::Tiff {
                                accent_separator(ui, &colors);
                                section_label(ui, &colors, "COMPRESSION");
                                egui::ComboBox::from_id_source("tiff_compression_combo")
                                    .width(160.0)
                                    .selected_text(match self.tiff_compression {
                                        TiffCompression::None    => "None",
                                        TiffCompression::Lzw     => "LZW",
                                        TiffCompression::Deflate => "Deflate",
                                    })
                                    .show_ui(ui, |ui| {
                                        ui.selectable_value(&mut self.tiff_compression, TiffCompression::None,    "None");
                                        ui.selectable_value(&mut self.tiff_compression, TiffCompression::Lzw,     "LZW");
                                        ui.selectable_value(&mut self.tiff_compression, TiffCompression::Deflate, "Deflate");
                                    });
                            }

                            // ── ICO warning ───────────────────────────────────
                            if self.format == SaveFormat::Ico {
                                accent_separator(ui, &colors);
                                ui.add_space(4.0);
                                let (w, h) = self.source_dimensions;
                                if w > 256 || h > 256 {
                                    ui.colored_label(
                                        Color32::from_rgb(255, 180, 50),
                                        format!("\u{26A0} {}×{} exceeds 256×256 and will be scaled down.", w, h),
                                    );
                                } else {
                                    ui.label(egui::RichText::new("ICO supports up to 256×256 per entry.").size(11.0).color(colors.text_muted));
                                }
                            }

                            // ── ANIMATION (PNG / GIF with multiple layers) ────
                            if self.format.supports_animation() && self.layer_count > 1 {
                                accent_separator(ui, &colors);
                                section_label(ui, &colors, "ANIMATION");
                                ui.checkbox(&mut self.animated, "Animated");

                                if self.animated {
                                    ui.add_space(4.0);
                                    egui::Grid::new("anim_options_grid")
                                        .num_columns(2)
                                        .min_col_width(60.0)
                                        .spacing([8.0, 4.0])
                                        .show(ui, |ui| {
                                            ui.label("FPS");
                                            ui.add(egui::Slider::new(&mut self.animation_fps, 1.0..=60.0)
                                                .step_by(1.0)
                                                .suffix(" fps"));
                                            ui.end_row();

                                            if self.format == SaveFormat::Gif {
                                                ui.label("Colors");
                                                let mut colors_f = self.gif_colors as f32;
                                                if ui.add(egui::Slider::new(&mut colors_f, 2.0..=256.0)
                                                    .step_by(1.0)
                                                    .logarithmic(true)
                                                ).changed() {
                                                    self.gif_colors = colors_f as u16;
                                                }
                                                ui.end_row();

                                                ui.label("Dither");
                                                ui.checkbox(&mut self.gif_dither, "");
                                                ui.end_row();
                                            }
                                        });

                                    if self.layer_count > 1 {
                                        ui.label(egui::RichText::new(
                                            format!("{} frames × {:.0} fps = {:.1}s",
                                                self.layer_count, self.animation_fps,
                                                self.layer_count as f32 / self.animation_fps)
                                        ).size(11.0).color(colors.text_muted));
                                    }
                                }
                            }
                        });
                    });

                    // ── Footer (full-width, below both columns) ───────────────
                    accent_separator(ui, &colors);
                    ui.add_space(6.0);
                    ui.horizontal(|ui| {
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui.button("  Cancel  ").clicked() {
                                should_close = true;
                            }
                            let save_btn = egui::Button::new(
                                egui::RichText::new("  Save  ")
                                    .color(Color32::WHITE)
                                    .strong()
                            ).fill(colors.accent);
                            let save_clicked = ui.add(save_btn).clicked() || enter;
                            if save_clicked {
                                let full_filename = format!("{}.{}", self.filename, self.format.extension());
                                let mut dialog = rfd::FileDialog::new()
                                    .set_file_name(&full_filename)
                                    .add_filter(self.format.label(), &[self.format.extension()]);
                                if let Some(dir) = &self.target_directory {
                                    dialog = dialog.set_directory(dir);
                                }
                                if let Some(path) = dialog.save_file() {
                                    result = Some(SaveAction {
                                        path,
                                        format: self.format,
                                        quality: self.quality,
                                        tiff_compression: self.tiff_compression,
                                        animated: self.animated && self.format.supports_animation(),
                                        animation_fps: self.animation_fps,
                                        gif_colors: self.gif_colors,
                                        gif_dither: self.gif_dither,
                                    });
                                    should_close = true;
                                }
                            }
                        });
                    });
                });
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
