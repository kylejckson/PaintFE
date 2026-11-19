use std::{
    fs::File,
    io::{BufWriter, Write},
    path::Path,
};

use image::RgbaImage;

use super::{BLEND_KEYS, BlendMode, CanvasState, MAX_BYTES, MAX_FILE, Result, TiledImage};
use crate::canvas::{CHUNK_SIZE, LayerContent};

const SIZE_LIMIT: &str = "PSD export exceeds the 256 MiB limit. Save as PFE for larger projects.";

fn flattens_layers(state: &CanvasState) -> bool {
    state.layers.iter().any(|layer| {
        matches!(layer.content, LayerContent::Adjustment(_))
            || blend_key(layer.blend_mode).is_none()
    })
}

pub(crate) fn losses(state: &CanvasState) -> Vec<String> {
    let mut losses = Vec::new();
    if !state.guides.items.is_empty() {
        losses.push(crate::t!("guides.save_loss"));
    }
    if flattens_layers(state) {
        losses.push(crate::t!("save.loss_structure"));
    } else if !state.layer_folders.is_empty()
        || state.layers.iter().any(|layer| layer.is_text_layer())
    {
        losses.push(crate::t!("psd.loss_editability"));
    }
    if state.layers.iter().any(|layer| {
        layer.hdr_metadata.enabled
            || layer.pixel_format != crate::canvas::PixelFormat::RgbaU8
            || layer
                .deep_pixels
                .as_ref()
                .is_some_and(|pixels| !matches!(pixels, crate::experimental::DeepRgbaBuffer::U8(_)))
    }) {
        losses.push(crate::t!("save.loss_depth"));
    }
    losses
}

fn blend_key(mode: BlendMode) -> Option<&'static [u8; 4]> {
    BLEND_KEYS
        .iter()
        .find_map(|&(item, key)| (mode == item).then_some(key))
}

#[derive(Clone, Copy)]
struct Bounds {
    left: u32,
    top: u32,
    right: u32,
    bottom: u32,
}

impl Bounds {
    fn of(image: &TiledImage) -> Self {
        let mut tiles = image.chunk_keys();
        let Some((x, y)) = tiles.next() else {
            return Self {
                left: 0,
                top: 0,
                right: 1,
                bottom: 1,
            };
        };
        let mut bounds = Self {
            left: x,
            top: y,
            right: x,
            bottom: y,
        };
        for (x, y) in tiles {
            bounds.left = bounds.left.min(x);
            bounds.top = bounds.top.min(y);
            bounds.right = bounds.right.max(x);
            bounds.bottom = bounds.bottom.max(y);
        }
        Self {
            left: bounds.left * CHUNK_SIZE,
            top: bounds.top * CHUNK_SIZE,
            right: ((bounds.right + 1) * CHUNK_SIZE).min(image.width()),
            bottom: ((bounds.bottom + 1) * CHUNK_SIZE).min(image.height()),
        }
    }

    fn samples(self) -> u64 {
        u64::from(self.right - self.left) * u64::from(self.bottom - self.top)
    }

    fn tiled_bytes(self) -> u64 {
        u64::from((self.right - self.left).div_ceil(CHUNK_SIZE))
            * u64::from((self.bottom - self.top).div_ceil(CHUNK_SIZE))
            * u64::from(CHUNK_SIZE).pow(2)
            * 4
    }

    fn write(self, buffer: &mut Vec<u8>) {
        for value in [self.top, self.left, self.bottom, self.right] {
            buffer.extend(value.to_be_bytes());
        }
    }
}

struct PlaneSource {
    pixels: TiledImage,
    bounds: Bounds,
}

impl PlaneSource {
    fn new(pixels: &TiledImage) -> Self {
        Self {
            pixels: pixels.clone(),
            bounds: Bounds::of(pixels),
        }
    }

    fn channel_bytes(&self) -> u64 {
        2 + self.bounds.samples()
    }

    fn write_channel(
        &self,
        out: &mut impl Write,
        row: &mut Vec<u8>,
        index: usize,
        mask: bool,
    ) -> std::io::Result<()> {
        out.write_all(&[0, 0])?; // Raw planar channel
        for y in self.bounds.top..self.bounds.bottom {
            row.clear();
            row.extend((self.bounds.left..self.bounds.right).map(|x| {
                let value = self.pixels.get_pixel(x, y)[index];
                if mask { 255 - value } else { value }
            }));
            out.write_all(row)?;
        }
        Ok(())
    }
}

struct ExportLayer {
    pixels: PlaneSource,
    mask: Option<PlaneSource>,
    record: Vec<u8>,
}

impl ExportLayer {
    fn new(layer: &crate::canvas::Layer, visible: bool) -> Self {
        let pixels = PlaneSource::new(&layer.pixels);
        let mask = layer.mask.as_ref().map(PlaneSource::new);
        let mut record = Vec::new();
        pixels.bounds.write(&mut record);
        record.extend(if mask.is_some() { 5u16 } else { 4u16 }.to_be_bytes());
        for id in [0i16, 1, 2, -1] {
            record.extend(id.to_be_bytes());
            record.extend((pixels.channel_bytes() as u32).to_be_bytes());
        }
        if let Some(mask) = &mask {
            record.extend((-2i16).to_be_bytes());
            record.extend((mask.channel_bytes() as u32).to_be_bytes());
        }
        record.extend(b"8BIM");
        record.extend(blend_key(layer.blend_mode).expect("representable PSD layer"));
        record.extend([
            (layer.opacity.clamp(0.0, 1.0) * 255.0).round() as u8,
            0,
            if visible { 0 } else { 2 },
            0,
        ]);

        let mut extra = Vec::new();
        if let Some(mask) = &mask {
            extra.extend(20u32.to_be_bytes());
            mask.bounds.write(&mut extra);
            extra.extend([255, if layer.mask_enabled { 0 } else { 2 }, 0, 0]);
        } else {
            extra.extend([0; 4]);
        }
        extra.extend([0; 4]); // Default blending ranges.
        let short_name: Vec<_> = layer
            .name
            .chars()
            .take(255)
            .map(|ch| {
                if ch.is_ascii() && !ch.is_ascii_control() {
                    ch as u8
                } else {
                    b'?'
                }
            })
            .collect();
        extra.push(short_name.len() as u8);
        extra.extend(short_name);
        extra.resize(extra.len().next_multiple_of(4), 0);
        let name: Vec<_> = layer.name.encode_utf16().collect();
        extra.extend(b"8BIMluni");
        extra.extend((4 + name.len() as u32 * 2).to_be_bytes());
        extra.extend((name.len() as u32).to_be_bytes());
        extra.extend(name.into_iter().flat_map(u16::to_be_bytes));
        record.extend((extra.len() as u32).to_be_bytes());
        record.extend(extra);
        Self {
            pixels,
            mask,
            record,
        }
    }
}

pub(crate) struct Export {
    composite: RgbaImage,
    layers: Vec<ExportLayer>,
    layer_info_bytes: u32,
    layer_info_padding: bool,
    merged_alpha: bool,
    resolution: Option<super::super::resolution::Resolution>,
}

impl Export {
    pub(crate) fn prepare(state: &CanvasState) -> Result<Self> {
        let (width, height) = (state.width, state.height);
        super::super::validate_open_dimensions(width, height)?;
        let composite_bytes = 4 * u64::from(width) * u64::from(height);
        if composite_bytes >= MAX_FILE as u64 {
            return Err(SIZE_LIMIT.into());
        }
        let flatten = flattens_layers(state) || state.layers.is_empty();
        let flattened_composite;
        let flattened;
        let layers = if flatten {
            let composite = state.composite_committed();
            let mut layer = crate::canvas::Layer::new(
                crate::t!("psd.composite"),
                width,
                height,
                image::Rgba([0; 4]),
            );
            layer.pixels = TiledImage::from_rgba_image(&composite);
            flattened_composite = Some(composite);
            flattened = [layer];
            &flattened[..]
        } else {
            flattened_composite = None;
            &state.layers[..]
        };
        if layers.len() > super::super::MAX_LAYERS {
            return Err(
                "Too many layers for PSD export. Save as PFE or reduce the layer count.".into(),
            );
        }
        let mut layer_info = 2u64;
        let mut decoded_tiles = 0;
        let mut export = Vec::with_capacity(layers.len());
        for (idx, layer) in layers.iter().enumerate().rev() {
            if layer.name.encode_utf16().take(4097).count() > 4096 {
                return Err(
                    "Layer names in PSD export must fit within 4096 UTF-16 code units.".into(),
                );
            }
            let visible = flatten || state.layer_effectively_visible(idx);
            let layer = ExportLayer::new(layer, visible);
            layer_info += layer.record.len() as u64 + 4 * layer.pixels.channel_bytes();
            decoded_tiles += layer.pixels.bounds.tiled_bytes();
            let mut temporary_bytes = 5 * layer.pixels.bounds.samples();
            if let Some(mask) = &layer.mask {
                layer_info += mask.channel_bytes();
                decoded_tiles += mask.bounds.tiled_bytes();
                temporary_bytes = temporary_bytes.max(5 * mask.bounds.samples());
            }
            // Match the importer's budgets so files we produce can be reopened here.
            if layer_info + composite_bytes + 76 > MAX_FILE as u64
                || decoded_tiles + temporary_bytes > MAX_BYTES as u64
            {
                return Err(SIZE_LIMIT.into());
            }
            export.push(layer);
        }
        let composite = flattened_composite.unwrap_or_else(|| state.composite_committed());
        let merged_alpha = composite.pixels().any(|p| p[3] != 255);
        Ok(Self {
            composite,
            layers: export,
            layer_info_bytes: layer_info.next_multiple_of(2) as u32,
            layer_info_padding: !layer_info.is_multiple_of(2),
            merged_alpha,
            resolution: state.resolution,
        })
    }

    pub(crate) fn write(&self, path: &Path) -> Result<()> {
        let file = File::create(path).map_err(|e| e.to_string())?;
        self.write_to(&mut BufWriter::new(file))
            .map_err(|e| e.to_string())
    }

    fn write_to(&self, out: &mut impl Write) -> std::io::Result<()> {
        let (width, height) = self.composite.dimensions();
        let channel_count = if self.merged_alpha { 4u16 } else { 3 };
        out.write_all(b"8BPS\0\x01\0\0\0\0\0\0")?;
        out.write_all(&channel_count.to_be_bytes())?;
        out.write_all(&height.to_be_bytes())?;
        out.write_all(&width.to_be_bytes())?;
        out.write_all(&[0, 8, 0, 3])?; // 8-bit RGB
        out.write_all(&[0; 4])?; // Colour mode data
        if let Some(resolution) = self.resolution {
            let [x, y] = resolution.dpi().map(|dpi| (dpi * 65_536.0).round() as u32);
            out.write_all(&28u32.to_be_bytes())?;
            out.write_all(b"8BIM")?;
            out.write_all(&1005u16.to_be_bytes())?;
            out.write_all(&[0; 2])?; // Empty Pascal resource name
            out.write_all(&16u32.to_be_bytes())?;
            for value in [x, y] {
                out.write_all(&value.to_be_bytes())?;
                out.write_all(&[0, 1, 0, 1])?; // Display and physical units: inches
            }
        } else {
            out.write_all(&[0; 4])?;
        }
        out.write_all(&(self.layer_info_bytes + 8).to_be_bytes())?;
        out.write_all(&self.layer_info_bytes.to_be_bytes())?;
        let count = self.layers.len() as i16;
        let count = if self.merged_alpha { -count } else { count };
        out.write_all(&count.to_be_bytes())?;
        for layer in &self.layers {
            out.write_all(&layer.record)?;
        }
        let mut row = Vec::with_capacity(width as usize);
        for layer in &self.layers {
            for index in 0..4 {
                layer.pixels.write_channel(out, &mut row, index, false)?;
            }
            if let Some(mask) = &layer.mask {
                mask.write_channel(out, &mut row, 3, true)?;
            }
        }
        if self.layer_info_padding {
            out.write_all(&[0])?;
        }
        out.write_all(&[0; 4])?; // Global layer mask
        out.write_all(&[0, 0])?; // Raw merged image
        for channel in 0..usize::from(channel_count) {
            for pixels in self.composite.rows() {
                row.clear();
                row.extend(pixels.map(|p| {
                    if self.merged_alpha && channel < 3 {
                        // The PSD merged RGB planes are matted against white.
                        let alpha = u16::from(p[3]);
                        ((u16::from(p[channel]) * alpha + 127) / 255 + 255 - alpha) as u8
                    } else {
                        p[channel]
                    }
                }));
                out.write_all(&row)?;
            }
        }
        out.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgba;

    #[test]
    fn odd_mask_channel_is_padded_and_merged_preview_uses_committed_alpha() {
        let mut state = CanvasState::new(1, 1);
        let layer = &mut state.layers[0];
        layer.pixels.put_pixel(0, 0, Rgba([51, 102, 153, 128]));
        layer.ensure_mask();
        layer.mask_enabled = false;
        let expected = state.composite_committed();
        state.preview_layer = Some(TiledImage::new_filled(1, 1, Rgba([230, 30, 16, 255])));
        assert_ne!(state.composite(), expected);

        let export = Export::prepare(&state).unwrap();
        assert!(export.layer_info_padding);
        assert_eq!(export.composite, expected);
        let mut file = Vec::new();
        export.write_to(&mut file).unwrap();
        assert_eq!(&file[file.len() - 6..], &[0, 0, 153, 178, 204, 128]);
        let loaded = super::super::decode(&file).unwrap();
        assert_eq!(loaded.canvas.composite(), expected);
        assert!(loaded.canvas.layers[0].mask.is_some());
    }
}
