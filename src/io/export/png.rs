use super::PreparedExportImage;
use serde::{Deserialize, Serialize};
use std::io::Write;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Depth {
    #[default]
    Automatic,
    Eight,
    Sixteen,
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Mode {
    #[default]
    Automatic,
    Rgba,
    Rgb,
    Gray,
    GrayAlpha,
    Indexed,
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Compression {
    Fast,
    #[default]
    Balanced,
    Smaller,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct PngOptions {
    pub depth: Depth,
    pub mode: Mode,
    pub compression: Compression,
    pub palette_colors: u16,
    pub dither: bool,
    pub preserve_palette_alpha: bool,
}
impl Default for PngOptions {
    fn default() -> Self {
        Self {
            depth: Depth::Automatic,
            mode: Mode::Automatic,
            compression: Compression::Balanced,
            palette_colors: 256,
            dither: true,
            preserve_palette_alpha: true,
        }
    }
}

pub fn write(
    image: &PreparedExportImage,
    options: PngOptions,
    writer: impl Write,
) -> Result<(), String> {
    if !(2..=256).contains(&options.palette_colors) {
        return Err("PNG palette must contain 2 to 256 colours".into());
    }
    let (width, height) = match image {
        PreparedExportImage::Rgba8(i) => i.dimensions(),
        PreparedExportImage::Rgba16 { width, height, .. }
        | PreparedExportImage::RgbaF32 { width, height, .. } => (*width, *height),
    };
    if width == 0 || height == 0 {
        return Err("PNG dimensions cannot be zero".into());
    }
    let expected = (width as usize)
        .checked_mul(height as usize)
        .and_then(|n| n.checked_mul(4))
        .ok_or("PNG dimensions overflow")?;
    if matches!(image,PreparedExportImage::Rgba16{pixels,..} if pixels.len()!=expected)
        || matches!(image,PreparedExportImage::RgbaF32{pixels,..} if pixels.len()!=expected)
    {
        return Err("PNG pixel buffer does not match its dimensions".into());
    }
    let sixteen = options.mode != Mode::Indexed
        && (options.depth == Depth::Sixteen
            || (options.depth == Depth::Automatic
                && matches!(image, PreparedExportImage::Rgba16 { .. })));
    use std::borrow::Cow;
    let pixels = match image {
        PreparedExportImage::Rgba16 { pixels, .. } if sixteen => Some(pixels.as_slice()),
        _ => None,
    };
    let eight = if pixels.is_some() {
        None
    } else {
        Some(match image {
            PreparedExportImage::Rgba8(image) => Cow::Borrowed(image),
            _ => Cow::Owned(image.rgba8()),
        })
    };
    let alpha = if let Some(pixels) = pixels {
        pixels.chunks_exact(4).any(|p| p[3] != 65535)
    } else {
        eight.as_ref().unwrap().pixels().any(|p| p[3] != 255)
    };
    let mode = if options.mode == Mode::Automatic {
        if alpha { Mode::Rgba } else { Mode::Rgb }
    } else {
        options.mode
    };
    let color = match mode {
        Mode::Rgb => png::ColorType::Rgb,
        Mode::Gray => png::ColorType::Grayscale,
        Mode::GrayAlpha => png::ColorType::GrayscaleAlpha,
        Mode::Indexed => png::ColorType::Indexed,
        _ => png::ColorType::Rgba,
    };
    let mut encoder = png::Encoder::new(writer, width, height);
    encoder.set_color(color);
    encoder.set_depth(if sixteen {
        png::BitDepth::Sixteen
    } else {
        png::BitDepth::Eight
    });
    encoder.set_compression(match options.compression {
        Compression::Fast => png::Compression::Fast,
        Compression::Balanced => png::Compression::Default,
        Compression::Smaller => png::Compression::Best,
    });
    if mode == Mode::Indexed {
        let mut indexed_source = Cow::Borrowed(eight.as_ref().unwrap().as_ref());
        if !options.preserve_palette_alpha {
            for p in indexed_source.to_mut().pixels_mut() {
                p[3] = 255;
            }
        }
        let (palette, indices) = super::palette::quantize(
            &indexed_source,
            usize::from(options.palette_colors),
            options.dither,
        );
        encoder.set_palette(
            palette
                .iter()
                .flat_map(|p| [p[0], p[1], p[2]])
                .collect::<Vec<_>>(),
        );
        encoder.set_trns(palette.iter().map(|p| p[3]).collect::<Vec<_>>());
        let mut writer = encoder.write_header().map_err(|e| e.to_string())?;
        writer
            .write_image_data(&indices)
            .map_err(|e| e.to_string())?;
        return writer.finish().map_err(|e| e.to_string());
    }
    let mut writer = encoder.write_header().map_err(|e| e.to_string())?;
    if !sixteen && mode == Mode::Rgba {
        writer
            .write_image_data(eight.as_ref().unwrap().as_raw())
            .map_err(|e| e.to_string())?;
    } else {
        // Convert one row at a time instead of making a second full-size raster.
        let mut stream = writer
            .stream_writer_with_size(32 * 1024)
            .map_err(|e| e.to_string())?;
        let mut row =
            Vec::with_capacity(width as usize * color.samples() * if sixteen { 2 } else { 1 });
        let append = |data: &mut Vec<u8>, p: [u16; 4]| {
            let gray = ((u64::from(p[0]) * 13933
                + u64::from(p[1]) * 46871
                + u64::from(p[2]) * 4732
                + 32768)
                / 65536) as u16;
            let (channels, length) = match mode {
                Mode::Rgb => ([p[0], p[1], p[2], 0], 3),
                Mode::Gray => ([gray, 0, 0, 0], 1),
                Mode::GrayAlpha => ([gray, p[3], 0, 0], 2),
                _ => (p, 4),
            };
            for &sample in &channels[..length] {
                if sixteen {
                    data.extend_from_slice(&sample.to_be_bytes());
                } else {
                    data.push(sample as u8);
                }
            }
        };
        if let Some(pixels) = pixels {
            for samples in pixels.chunks_exact(width as usize * 4) {
                row.clear();
                for p in samples.chunks_exact(4) {
                    append(&mut row, [p[0], p[1], p[2], p[3]]);
                }
                stream.write_all(&row).map_err(|e| e.to_string())?;
            }
        } else {
            for samples in eight.as_ref().unwrap().rows() {
                row.clear();
                for p in samples {
                    append(
                        &mut row,
                        p.0.map(|v| u16::from(v) * if sixteen { 257 } else { 1 }),
                    );
                }
                stream.write_all(&row).map_err(|e| e.to_string())?;
            }
        }
        stream.finish().map_err(|e| e.to_string())?;
    }
    writer.finish().map_err(|e| e.to_string())
}
