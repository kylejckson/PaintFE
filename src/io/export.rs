//! One export-settings snapshot and encoder entry point for previews and saves.
use super::{PreparedExportImage, encode_prepared_and_write};
use crate::components::dialogs::{SaveFormat, TiffCompression};
use serde::{Deserialize, Serialize};
use std::{io::Write, path::Path};

pub mod palette;
pub mod png;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ExportOptions {
    pub format: SaveFormat,
    pub quality: u8,
    pub webp_lossless: bool,
    pub tiff_compression: TiffCompression,
    pub png: png::PngOptions,
    pub animated: bool,
    pub animation_fps: f32,
    pub gif_colors: u16,
    pub gif_dither: bool,
}
impl Default for ExportOptions {
    fn default() -> Self {
        Self {
            format: SaveFormat::Png,
            quality: 90,
            webp_lossless: true,
            tiff_compression: TiffCompression::None,
            png: Default::default(),
            animated: false,
            animation_fps: 10.0,
            gif_colors: 256,
            gif_dither: true,
        }
    }
}
impl ExportOptions {
    pub fn validate(&self) -> Result<(), String> {
        if !(1..=100).contains(&self.quality)
            || !(2..=256).contains(&self.gif_colors)
            || (self.animated
                && self.format.supports_animation()
                && (!self.animation_fps.is_finite() || self.animation_fps <= 0.0))
            || !(2..=256).contains(&self.png.palette_colors)
        {
            return Err("Invalid export settings".into());
        }
        if let SaveFormat::Jxl(options) = self.format
            && !(1..=9).contains(&options.effort)
        {
            return Err("Invalid JPEG XL effort".into());
        }
        Ok(())
    }
}

pub fn write(
    image: PreparedExportImage,
    path: &Path,
    options: ExportOptions,
) -> Result<(), String> {
    options.validate()?;
    match options.format {
        SaveFormat::Png => {
            let mut writer =
                std::io::BufWriter::new(std::fs::File::create(path).map_err(|e| e.to_string())?);
            png::write(&image, options.png, &mut writer)?;
            writer.flush().map_err(|e| e.to_string())
        }
        SaveFormat::Gif => super::encode_static_gif_with_options(
            &image.rgba8(),
            path,
            options.gif_colors,
            options.gif_dither,
        ),
        _ => encode_prepared_and_write(
            image,
            path,
            options.format,
            options.quality,
            options.tiff_compression,
            options.webp_lossless,
        )
        .map_err(|e| e.to_string()),
    }
}

pub fn preview(
    image: PreparedExportImage,
    options: ExportOptions,
) -> Result<crate::components::dialogs::PreviewResult, String> {
    #[cfg(target_arch = "wasm32")]
    {
        if options.format == SaveFormat::Png {
            let mut bytes = Vec::new();
            png::write(&image, options.png, &mut bytes)?;
            let decoded = image::load_from_memory_with_format(&bytes, image::ImageFormat::Png)
                .map_err(|e| e.to_string())?
                .to_rgba8();
            return Ok(crate::components::dialogs::PreviewResult {
                file_size: bytes.len(),
                preview_image: decoded,
            });
        }
        return crate::components::dialogs::generate_preview(
            &image.rgba8(),
            options.format,
            options.quality,
            options.webp_lossless,
        )
        .ok_or_else(|| "Preview encoding failed".into());
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        // Preview files are private, bounded thumbnails. Their encoder is also the final-save encoder.
        let directory =
            std::env::temp_dir().join(format!("paintfe-preview-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir(&directory).map_err(|e| e.to_string())?;
        let path = directory.join(format!("preview.{}", options.format.extension()));
        let result = (|| {
            if options.format == SaveFormat::Pfe {
                return Ok(crate::components::dialogs::PreviewResult {
                    file_size: 0,
                    preview_image: image.rgba8(),
                });
            }
            write(image, &path, options)?;
            let bytes = std::fs::read(&path).map_err(|e| e.to_string())?;
            let decoded = super::decode_image_path(&path)
                .map_err(|e| e.to_string())?
                .to_rgba8();
            Ok(crate::components::dialogs::PreviewResult {
                file_size: bytes.len(),
                preview_image: decoded,
            })
        })();
        let _ = std::fs::remove_file(path);
        let _ = std::fs::remove_dir(directory);
        result
    }
}
