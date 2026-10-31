//! Concrete encoder options shared by final saves and previews.
use super::*;
use image::ImageEncoder;
use serde::{Deserialize, Serialize};
use std::io::BufWriter;
macro_rules! choices {
    ($name:ident, $first:ident $(,$rest:ident)*) => {
        #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
        pub enum $name { #[default] $first, $($rest),* }
    };
}
choices!(JpegSampling, Full, Vertical, Horizontal, Quarter);
choices!(WebpPreset, Default, Photo, Picture, Drawing, Icon, Text);
choices!(RasterDepth, Automatic, Eight, Sixteen, Float);
choices!(RasterMode, Automatic, Rgba, Rgb, Gray, Indexed);
choices!(HeicChroma, Automatic, Full, Horizontal, Quarter);

pub(super) fn write(
    image: &PreparedExportImage,
    path: &Path,
    o: ExportOptions,
    tracked: bool,
) -> Result<bool, String> {
    use crate::components::dialogs::SaveFormat;
    let handled = matches!(
        o.format,
        SaveFormat::Jpeg
            | SaveFormat::Webp
            | SaveFormat::Bmp
            | SaveFormat::Tga
            | SaveFormat::Avif
            | SaveFormat::Heic
    ) || (o.format == SaveFormat::Tiff
        && (o.raster_depth != RasterDepth::Automatic || o.raster_mode != RasterMode::Automatic));
    if !handled {
        return Ok(false);
    }
    // Validate and encode before creating a target where an external helper can fail.
    if o.format == SaveFormat::Avif {
        let bytes = crate::io::avif::encode_options(&image.rgba8(), o, tracked)?;
        std::fs::write(path, bytes).map_err(|e| e.to_string())?;
        return Ok(true);
    }
    if o.format == SaveFormat::Heic {
        let bytes = crate::io::heic::encode_options(&image.rgba8(), o, tracked)?;
        std::fs::write(path, bytes).map_err(|e| e.to_string())?;
        return Ok(true);
    }
    if o.format == SaveFormat::Tiff {
        write_tiff(image, path, o)?;
        return Ok(true);
    }
    let rgba = image.rgba8();
    let mut writer = BufWriter::new(std::fs::File::create(path).map_err(|e| e.to_string())?);
    let result: Result<(), String> = match o.format {
        SaveFormat::Jpeg => encode_jpeg(&rgba, &mut writer, o.quality, o.jpeg_sampling),
        SaveFormat::Webp => {
            #[cfg(not(target_arch = "wasm32"))]
            {
                use libwebp_sys::WebPPreset::*;
                crate::io::validate_webp_dimensions(rgba.width(), rgba.height())
                    .map_err(|e| e.to_string())?;
                let preset = match o.webp_preset {
                    WebpPreset::Default => WEBP_PRESET_DEFAULT,
                    WebpPreset::Photo => WEBP_PRESET_PHOTO,
                    WebpPreset::Picture => WEBP_PRESET_PICTURE,
                    WebpPreset::Drawing => WEBP_PRESET_DRAWING,
                    WebpPreset::Icon => WEBP_PRESET_ICON,
                    WebpPreset::Text => WEBP_PRESET_TEXT,
                };
                let mut config = webp::WebPConfig::new_with_preset(
                    preset,
                    if o.webp_lossless {
                        75.0
                    } else {
                        f32::from(o.quality)
                    },
                )
                .map_err(|_| "Invalid WebP preset")?;
                config.lossless = i32::from(o.webp_lossless);
                config.exact = i32::from(o.webp_lossless);
                let bytes = webp::Encoder::from_rgba(rgba.as_raw(), rgba.width(), rgba.height())
                    .encode_advanced(&config)
                    .map_err(|e| format!("WebP: {e:?}"))?;
                writer.write_all(&bytes).map_err(|e| e.to_string())
            }
            #[cfg(target_arch = "wasm32")]
            {
                image::codecs::webp::WebPEncoder::new_lossless(&mut writer)
                    .write_image(
                        rgba.as_raw(),
                        rgba.width(),
                        rgba.height(),
                        image::ExtendedColorType::Rgba8,
                    )
                    .map_err(|e| e.to_string())
            }
        }
        SaveFormat::Bmp | SaveFormat::Tga => {
            let (w, h) = rgba.dimensions();
            if o.format == SaveFormat::Bmp && o.raster_mode == RasterMode::Indexed {
                let mut rgba = rgba;
                for pixel in rgba.pixels_mut() {
                    pixel[3] = 255;
                }
                let (palette, index) = palette::quantize_with_options(
                    &rgba,
                    256,
                    o.gif_dither,
                    o.quantizer,
                    o.dither_strength,
                );
                let palette: Vec<_> = palette.iter().map(|p| [p[0], p[1], p[2]]).collect();
                image::codecs::bmp::BmpEncoder::new(&mut writer)
                    .encode_with_palette(&index, w, h, image::ExtendedColorType::L8, Some(&palette))
                    .map_err(|e| e.to_string())
            } else {
                let mode = if o.raster_mode == RasterMode::Automatic {
                    RasterMode::Rgba
                } else {
                    o.raster_mode
                };
                let (raw, color) = match mode {
                    RasterMode::Rgb => (
                        image::DynamicImage::ImageRgba8(rgba).to_rgb8().into_raw(),
                        image::ExtendedColorType::Rgb8,
                    ),
                    RasterMode::Gray => (
                        image::DynamicImage::ImageRgba8(rgba).to_luma8().into_raw(),
                        image::ExtendedColorType::L8,
                    ),
                    RasterMode::Indexed => return Err("Indexed TGA output is unavailable".into()),
                    _ => (rgba.into_raw(), image::ExtendedColorType::Rgba8),
                };
                if o.format == SaveFormat::Bmp {
                    image::codecs::bmp::BmpEncoder::new(&mut writer)
                        .write_image(&raw, w, h, color)
                        .map_err(|e| e.to_string())
                } else {
                    let encoder = image::codecs::tga::TgaEncoder::new(&mut writer);
                    let encoder = if o.tga_rle {
                        encoder
                    } else {
                        encoder.disable_rle()
                    };
                    encoder
                        .write_image(&raw, w, h, color)
                        .map_err(|e| e.to_string())
                }
            }
        }
        _ => unreachable!(),
    };
    result?;
    writer.flush().map_err(|e| e.to_string())?;
    Ok(true)
}

pub(crate) fn encode_jpeg(
    rgba: &image::RgbaImage,
    writer: impl std::io::Write,
    quality: u8,
    sampling: JpegSampling,
) -> Result<(), String> {
    let w = u16::try_from(rgba.width()).map_err(|_| "JPEG dimensions exceed 65535")?;
    let h = u16::try_from(rgba.height()).map_err(|_| "JPEG dimensions exceed 65535")?;
    let rgb = image::RgbImage::from_raw(
        rgba.width(),
        rgba.height(),
        rgba.pixels()
            .flat_map(|p| p.0[..3].iter().copied())
            .collect(),
    )
    .ok_or("Invalid JPEG pixels")?;
    let mut encoder = jpeg_encoder::Encoder::new(writer, quality);
    use jpeg_encoder::SamplingFactor::*;
    encoder.set_sampling_factor(match sampling {
        JpegSampling::Full => R_4_4_4,
        JpegSampling::Vertical => R_4_4_0,
        JpegSampling::Horizontal => R_4_2_2,
        JpegSampling::Quarter => R_4_2_0,
    });
    encoder
        .encode(rgb.as_raw(), w, h, jpeg_encoder::ColorType::Rgb)
        .map_err(|e| e.to_string())
}

fn write_tiff(image: &PreparedExportImage, path: &Path, o: ExportOptions) -> Result<(), String> {
    let (w, h, rgba) = match image {
        PreparedExportImage::Rgba8(i) => (
            i.width(),
            i.height(),
            i.as_raw()
                .iter()
                .map(|v| f32::from(*v) / 255.0)
                .collect::<Vec<_>>(),
        ),
        PreparedExportImage::Rgba16 {
            width,
            height,
            pixels,
        } => (
            *width,
            *height,
            pixels.iter().map(|v| f32::from(*v) / 65535.0).collect(),
        ),
        PreparedExportImage::RgbaF32 {
            width,
            height,
            pixels,
        } => (*width, *height, pixels.clone()),
    };
    let mode = if o.raster_mode == RasterMode::Automatic {
        RasterMode::Rgba
    } else {
        o.raster_mode
    };
    if mode == RasterMode::Indexed {
        return Err("Indexed TIFF output is unavailable".into());
    }
    let mut samples = Vec::new();
    for p in rgba.chunks_exact(4) {
        match mode {
            RasterMode::Rgb => samples.extend_from_slice(&p[..3]),
            RasterMode::Gray => samples.push(p[0] * 0.2126 + p[1] * 0.7152 + p[2] * 0.0722),
            _ => samples.extend_from_slice(p),
        }
    }
    let depth = if o.raster_depth == RasterDepth::Automatic {
        match image {
            PreparedExportImage::Rgba16 { .. } => RasterDepth::Sixteen,
            PreparedExportImage::RgbaF32 { .. } => RasterDepth::Float,
            _ => RasterDepth::Eight,
        }
    } else {
        o.raster_depth
    };
    let file = std::fs::File::create(path).map_err(|e| e.to_string())?;
    let mut encoder =
        tiff::encoder::TiffEncoder::new(BufWriter::new(file)).map_err(|e| e.to_string())?;
    macro_rules! encode {
        ($ty:ty,$data:expr) => {
            match o.tiff_compression {
                TiffCompression::None => encoder.write_image::<$ty>(w, h, $data),
                TiffCompression::Lzw => encoder.write_image_with_compression::<$ty, _>(
                    w,
                    h,
                    tiff::encoder::compression::Lzw,
                    $data,
                ),
                TiffCompression::Deflate => encoder.write_image_with_compression::<$ty, _>(
                    w,
                    h,
                    tiff::encoder::compression::Deflate::default(),
                    $data,
                ),
            }
            .map_err(|e| e.to_string())
        };
    }
    use tiff::encoder::colortype::*;
    match depth {
        RasterDepth::Float => match mode {
            RasterMode::Rgb => encode!(RGB32Float, &samples),
            RasterMode::Gray => encode!(Gray32Float, &samples),
            _ => encode!(RGBA32Float, &samples),
        },
        RasterDepth::Sixteen => {
            let data: Vec<u16> = samples
                .iter()
                .map(|v| (v.clamp(0.0, 1.0) * 65535.0).round() as u16)
                .collect();
            match mode {
                RasterMode::Rgb => encode!(RGB16, &data),
                RasterMode::Gray => encode!(Gray16, &data),
                _ => encode!(RGBA16, &data),
            }
        }
        _ => {
            let data: Vec<u8> = samples
                .iter()
                .map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)
                .collect();
            match mode {
                RasterMode::Rgb => encode!(RGB8, &data),
                RasterMode::Gray => encode!(Gray8, &data),
                _ => encode!(RGBA8, &data),
            }
        }
    }
}
