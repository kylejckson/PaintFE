//! Bounded import and export of ordinary 8-bit RGB Photoshop raster documents.
use std::{fs::File, io::Read, path::Path};

use crate::canvas::{BlendMode, CanvasState, Layer, TiledImage};
use image::Rgba;

mod export;
pub(crate) use export::{Export, losses};

type Result<T> = std::result::Result<T, String>;
const MAX_FILE: usize = 256 * 1024 * 1024;
const MAX_BYTES: usize = 512 * 1024 * 1024;
const TRUNCATED: &str = "Truncated Photoshop document";

struct Reader<'a>(&'a [u8]);
impl<'a> Reader<'a> {
    fn take(&mut self, len: usize) -> Result<&'a [u8]> {
        let (head, rest) = self.0.split_at_checked(len).ok_or(TRUNCATED)?;
        self.0 = rest;
        Ok(head)
    }
    fn byte(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_be_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_be_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn i32(&mut self) -> Result<i32> {
        Ok(self.u32()? as i32)
    }
    fn section(&mut self) -> Result<Self> {
        let len = self.u32()? as usize;
        Ok(Self(self.take(len)?))
    }
}

struct Record {
    x: i32,
    y: i32,
    width: u32,
    height: u32,
    channels: Vec<(i16, usize)>,
    name: String,
    visible: bool,
    opacity: u8,
    blend: BlendMode,
    mask: Option<MaskRecord>,
}

struct MaskRecord {
    x: i32,
    y: i32,
    width: u32,
    height: u32,
    enabled: bool,
}

pub struct Import {
    pub canvas: CanvasState,
    pub fallback: Option<FallbackReason>,
}

#[derive(Debug, PartialEq, Eq)]
pub enum FallbackReason {
    NoLayers,
    Channel(i16),
    Blend([u8; 4]),
    Clipping,
    Mask,
    BlendingRanges,
    LayerData,
    LayerTag([u8; 4]),
}

impl std::fmt::Display for FallbackReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let label = |id: &[u8; 4]| {
            id.iter()
                .map(|&b| {
                    if b.is_ascii_graphic() || b == b' ' {
                        char::from(b)
                    } else {
                        '?'
                    }
                })
                .collect::<String>()
        };
        match self {
            Self::NoLayers => f.write_str("PSD: no raster layer records"),
            Self::Channel(id) => write!(f, "PSD channel {id}"),
            Self::Blend(key) => write!(f, "PSD blend mode '{}'", label(key)),
            Self::Clipping => f.write_str("PSD clipping layer"),
            Self::Mask => f.write_str("PSD layer mask layout"),
            Self::BlendingRanges => f.write_str("PSD layer blending ranges"),
            Self::LayerData => f.write_str("PSD layer data"),
            Self::LayerTag(key) => write!(f, "PSD layer feature '{}'", label(key)),
        }
    }
}

pub fn load(path: &Path) -> Result<Import> {
    let file = File::open(path).map_err(|e| e.to_string())?;
    if file.metadata().map_err(|e| e.to_string())?.len() > MAX_FILE as u64 {
        return Err("Photoshop file exceeds the 256 MiB import limit".into());
    }
    let mut data = Vec::new();
    file.take(MAX_FILE as u64 + 1)
        .read_to_end(&mut data)
        .map_err(|e| e.to_string())?;
    if data.len() > MAX_FILE {
        return Err("Photoshop file exceeds the 256 MiB import limit".into());
    }
    decode(&data)
}

const BLEND_KEYS: [(BlendMode, &[u8; 4]); 9] = [
    (BlendMode::Normal, b"norm"),
    (BlendMode::Multiply, b"mul "),
    (BlendMode::Screen, b"scrn"),
    (BlendMode::Overlay, b"over"),
    (BlendMode::Darken, b"dark"),
    (BlendMode::Lighten, b"lite"),
    (BlendMode::Difference, b"diff"),
    (BlendMode::HardLight, b"hLit"),
    (BlendMode::SoftLight, b"sLit"),
];

fn records<'a>(info: &mut Reader<'a>) -> Result<(Vec<Record>, bool, Option<FallbackReason>)> {
    if info.0.is_empty() {
        return Ok((Vec::new(), false, None));
    }
    let count = info.u16()? as i16;
    if usize::from(count.unsigned_abs()) > super::MAX_LAYERS {
        return Err("Too many Photoshop layers".into());
    }
    let mut fallback = None;
    let mut records = Vec::new();
    for _ in 0..count.unsigned_abs() {
        let y = info.i32()?;
        let x = info.i32()?;
        let bottom = info.i32()?;
        let right = info.i32()?;
        let width = u32::try_from(i64::from(right) - i64::from(x))
            .map_err(|_| "Invalid Photoshop layer bounds")?;
        let height = u32::try_from(i64::from(bottom) - i64::from(y))
            .map_err(|_| "Invalid Photoshop layer bounds")?;
        let len = info.u16()?;
        if len > 56 {
            return Err("Too many Photoshop layer channels".into());
        }
        let mut channels = Vec::new();
        for _ in 0..len {
            let id = info.u16()? as i16;
            if !(-2..=2).contains(&id) {
                fallback.get_or_insert(FallbackReason::Channel(id));
            }
            if channels.iter().any(|&(old, _)| old == id) {
                return Err("Duplicate Photoshop layer channel".into());
            }
            channels.push((id, info.u32()? as usize));
        }
        if info.take(4)? != b"8BIM" {
            return Err("Invalid Photoshop layer signature".into());
        }
        let key: &[u8; 4] = info.take(4)?.try_into().unwrap();
        let mode = BLEND_KEYS
            .iter()
            .find_map(|&(mode, code)| (key == code).then_some(mode));
        if mode.is_none() {
            fallback.get_or_insert(FallbackReason::Blend(*key));
        }
        let opacity = info.byte()?;
        if info.byte()? != 0 {
            fallback.get_or_insert(FallbackReason::Clipping);
        }
        let flags = info.byte()?;
        if flags & 0x18 == 0x18 {
            fallback.get_or_insert(FallbackReason::LayerData);
        }
        info.byte()?;
        let mut extra = info.section()?;
        let mut mask_data = extra.section()?;
        let mask = if mask_data.0.len() == 20 {
            let my = mask_data.i32()?;
            let mx = mask_data.i32()?;
            let mh = u32::try_from(i64::from(mask_data.i32()?) - i64::from(my))
                .map_err(|_| "Invalid Photoshop mask bounds")?;
            let mw = u32::try_from(i64::from(mask_data.i32()?) - i64::from(mx))
                .map_err(|_| "Invalid Photoshop mask bounds")?;
            let default = mask_data.byte()?;
            let flags = mask_data.byte()?;
            if default == 255 && flags & !3 == 0 && mw > 0 && mh > 0 {
                let (ox, oy) = if flags & 1 == 1 { (x, y) } else { (0, 0) };
                Some(MaskRecord {
                    x: mx.checked_add(ox).ok_or("Invalid Photoshop mask offset")?,
                    y: my.checked_add(oy).ok_or("Invalid Photoshop mask offset")?,
                    width: mw,
                    height: mh,
                    enabled: flags & 2 == 0,
                })
            } else {
                fallback.get_or_insert(FallbackReason::Mask);
                None
            }
        } else {
            if !mask_data.0.is_empty() {
                fallback.get_or_insert(FallbackReason::Mask);
            }
            None
        };
        if mask.is_some() != channels.iter().any(|&(id, _)| id == -2) {
            fallback.get_or_insert(FallbackReason::Mask);
        }
        let ranges = extra.section()?.0;
        // Default blending ranges have black/white endpoints. Anything else needs
        // Photoshop's compositor and is handled by the embedded composite path.
        if !ranges.chunks_exact(4).remainder().is_empty()
            || ranges.chunks_exact(4).any(|v| v != [0, 0, 255, 255])
        {
            fallback.get_or_insert(FallbackReason::BlendingRanges);
        }
        let len = extra.byte()? as usize;
        let mut name = String::from_utf8_lossy(extra.take(len)?).into_owned();
        extra.take((4 - (len + 1) % 4) % 4)?;
        while extra.0.len() >= 12 {
            if extra.take(4)? != b"8BIM" {
                fallback.get_or_insert(FallbackReason::LayerData);
                break;
            }
            let key: &[u8; 4] = extra.take(4)?.try_into().unwrap();
            let block_len = extra.u32()? as usize;
            let mut block = Reader(extra.take(block_len)?);
            extra.take(block_len % 2)?;
            match key {
                b"luni" => {
                    let len = block.u32()? as usize;
                    if len > 4096 {
                        return Err("Photoshop layer name exceeds the import limit".into());
                    }
                    let bytes = block.take(len.checked_mul(2).ok_or(TRUNCATED)?)?;
                    name = String::from_utf16_lossy(
                        &bytes
                            .chunks_exact(2)
                            .map(|v| u16::from_be_bytes([v[0], v[1]]))
                            .collect::<Vec<_>>(),
                    );
                    name.truncate(name.trim_end_matches('\0').len());
                }
                b"lyid" | b"lclr" | b"lyvr" | b"shmd" | b"fxrp" | b"lspf" => {}
                // Layer name source is editor metadata. These pixel-related flags
                // can be ignored only when they match the default raster behaviour.
                b"lnsr" if block.0.len() == 4 => {}
                b"iOpa" if block.0 == [255] || block.0 == [255, 0, 0, 0] => {}
                b"clbl" if block.0 == [1, 0, 0, 0] => {}
                b"infx" | b"knko" if block.0 == [0, 0, 0, 0] => {}
                _ => {
                    fallback.get_or_insert(FallbackReason::LayerTag(*key));
                }
            }
        }
        records.push(Record {
            x,
            y,
            width,
            height,
            channels,
            name,
            visible: flags & 2 == 0,
            opacity,
            blend: mode.unwrap_or_default(),
            mask,
        });
    }
    Ok((records, count < 0, fallback))
}

fn unpack_row(mut input: Reader<'_>, output: &mut [u8]) -> Result<()> {
    let mut offset: usize = 0;
    while !input.0.is_empty() {
        let control = input.byte()? as i8;
        if control == -128 {
            continue;
        }
        let len = usize::from(control.unsigned_abs()) + 1;
        let end = offset + len;
        let output = output
            .get_mut(offset..end)
            .ok_or("Photoshop RLE row exceeds its width")?;
        if control < 0 {
            output.fill(input.byte()?);
        } else {
            output.copy_from_slice(input.take(len)?);
        }
        offset = end;
    }
    if offset != output.len() {
        return Err("Incomplete Photoshop RLE row".into());
    }
    Ok(())
}

fn planes(data: &mut Reader<'_>, width: usize, rows: usize) -> Result<Vec<u8>> {
    let count = width
        .checked_mul(rows)
        .filter(|&n| n <= MAX_BYTES)
        .ok_or("Photoshop channel exceeds the memory budget")?;
    match data.u16()? {
        0 => Ok(data.take(count)?.to_vec()),
        1 => {
            let mut lengths = Reader(data.take(rows.checked_mul(2).ok_or(TRUNCATED)?)?);
            let mut output = vec![0; count];
            for row in output.chunks_exact_mut(width.max(1)) {
                unpack_row(Reader(data.take(usize::from(lengths.u16()?))?), row)?;
            }
            Ok(output)
        }
        format @ (2 | 3) => {
            let mut zlib = flate2::read::ZlibDecoder::new(data.0);
            let mut output = vec![0; count];
            zlib.read_exact(&mut output)
                .map_err(|_| "Invalid Photoshop ZIP channel")?;
            if zlib
                .read(&mut [0])
                .map_err(|_| "Invalid Photoshop ZIP channel")?
                != 0
            {
                return Err("Photoshop ZIP channel exceeds its dimensions".into());
            }
            data.0 = &[];
            if format == 3 {
                for row in output.chunks_exact_mut(width.max(1)) {
                    for index in 1..row.len() {
                        row[index] = row[index].wrapping_add(row[index - 1]);
                    }
                }
            }
            Ok(output)
        }
        _ => Err("Unsupported Photoshop channel compression".into()),
    }
}

fn read_resolution(mut resources: Reader<'_>) -> Option<super::resolution::Resolution> {
    while !resources.0.is_empty() {
        if resources.take(4).ok()? != b"8BIM" {
            return None;
        }
        let id = resources.u16().ok()?;
        let name = usize::from(resources.byte().ok()?);
        resources.take((name + 1).next_multiple_of(2) - 1).ok()?;
        let mut data = resources.section().ok()?;
        resources.take(data.0.len() % 2).ok()?;
        if id == 1005 && data.0.len() == 16 {
            let x = f64::from(data.u32().ok()?) / 65_536.0;
            data.take(4).ok()?;
            let y = f64::from(data.u32().ok()?) / 65_536.0;
            return super::resolution::Resolution::from_dpi(x, y);
        }
    }
    None
}

fn decode(data: &[u8]) -> Result<Import> {
    let mut src = Reader(data);
    if src.take(4)? != b"8BPS" || src.u16()? != 1 {
        return Err("Expected a Photoshop PSD document (PSB is unsupported)".into());
    }
    src.take(6)?;
    let channel_count = usize::from(src.u16()?);
    let height = src.u32()?;
    let width = src.u32()?;
    super::validate_open_dimensions(width, height)?;
    if src.u16()? != 8 || src.u16()? != 3 || !(3..=56).contains(&channel_count) {
        return Err("Only 8-bit RGB Photoshop documents are supported".into());
    }
    src.section()?; // colour mode data
    let resources = src.section()?;
    let resolution = read_resolution(resources);
    let mut layer_mask = src.section()?;
    let mut layer_info = if layer_mask.0.is_empty() {
        Reader(&[])
    } else {
        layer_mask.section()?
    };
    let (records, merged_alpha, mut fallback) = records(&mut layer_info)?;
    let mut canvas = CanvasState::new(width, height);
    canvas.layers.clear();
    canvas.resolution = resolution;
    if records.is_empty() {
        fallback.get_or_insert(FallbackReason::NoLayers);
    }
    if fallback.is_some() {
        let channels = planes(&mut src, width as usize, height as usize * channel_count)
            .map_err(|e| format!("Could not read the embedded Photoshop composite: {e}"))?;
        let area = width as usize * height as usize;
        let rgba: Vec<_> = (0..area)
            .flat_map(|i| {
                [
                    channels[i],
                    channels[area + i],
                    channels[2 * area + i],
                    if merged_alpha && channel_count >= 4 {
                        channels[3 * area + i]
                    } else {
                        255
                    },
                ]
            })
            .collect();
        let mut layer = Layer::new(crate::t!("psd.composite"), width, height, Rgba([0; 4]));
        layer.pixels = TiledImage::from_raw_rgba(width, height, &rgba);
        canvas.layers.push(layer);
    } else {
        let mut retained_bytes = 0usize;
        for record in records {
            let area = (record.width as usize)
                .checked_mul(record.height as usize)
                .filter(|&n| n <= MAX_BYTES / 8)
                .ok_or("Photoshop layer exceeds the memory budget")?;
            let x0 = record.x.clamp(0, width as i32);
            let y0 = record.y.clamp(0, height as i32);
            let x1 =
                (i64::from(record.x) + i64::from(record.width)).clamp(0, i64::from(width)) as i32;
            let y1 =
                (i64::from(record.y) + i64::from(record.height)).clamp(0, i64::from(height)) as i32;
            let cw = (x1 - x0) as usize;
            let ch = (y1 - y0) as usize;
            let chunk = crate::canvas::CHUNK_SIZE as usize;
            if cw != 0 && ch != 0 {
                retained_bytes += (cw + x0 as usize % chunk).div_ceil(chunk)
                    * (ch + y0 as usize % chunk).div_ceil(chunk)
                    * chunk
                    * chunk
                    * 4;
                if retained_bytes + area * 5 > MAX_BYTES {
                    return Err("Photoshop layers exceed the memory budget".into());
                }
            }
            let mut rgba = vec![255; cw * ch * 4];
            let mut colour_channels = 0u8;
            let mut layer_mask = None;
            for (id, len) in record.channels {
                let mut bytes = Reader(layer_info.take(len)?);
                if id == -2 {
                    let mask = record
                        .mask
                        .as_ref()
                        .ok_or("Missing Photoshop mask metadata")?;
                    layer_mask = Some(decode_mask(
                        &mut bytes,
                        width,
                        height,
                        mask,
                        &mut retained_bytes,
                    )?);
                    continue;
                }
                if rgba.is_empty() {
                    continue;
                }
                let plane = planes(&mut bytes, record.width as usize, record.height as usize)?;
                let channel = if id == -1 { 3 } else { id as usize };
                colour_channels |= 1 << channel;
                let left = (i64::from(x0) - i64::from(record.x)) as usize;
                let top = (i64::from(y0) - i64::from(record.y)) as usize;
                for row in 0..ch {
                    let start = (top + row) * record.width as usize + left;
                    for (col, &value) in plane[start..start + cw].iter().enumerate() {
                        rgba[(row * cw + col) * 4 + channel] = value;
                    }
                }
            }
            if !rgba.is_empty() && colour_channels & 7 != 7 {
                return Err("Photoshop raster layer is missing RGB channels".into());
            }
            let mut layer = Layer::new(record.name, width, height, Rgba([0; 4]));
            if !rgba.is_empty() {
                layer.pixels = TiledImage::from_region_rgba(
                    width, height, &rgba, cw as u32, ch as u32, x0, y0,
                );
            }
            layer.visible = record.visible;
            layer.opacity = f32::from(record.opacity) / 255.0;
            layer.blend_mode = record.blend;
            layer.mask = layer_mask;
            layer.mask_enabled = record.mask.is_some_and(|mask| mask.enabled);
            canvas.layers.push(layer);
        }
        canvas.layers.reverse(); // PSD lists topmost layer first.
    }
    canvas.active_layer_index = canvas.layers.len() - 1;
    Ok(Import { canvas, fallback })
}

fn decode_mask(
    bytes: &mut Reader<'_>,
    width: u32,
    height: u32,
    mask: &MaskRecord,
    retained: &mut usize,
) -> Result<TiledImage> {
    let x = mask.x.clamp(0, width as i32) as usize;
    let y = mask.y.clamp(0, height as i32) as usize;
    let right = (i64::from(mask.x) + i64::from(mask.width)).clamp(0, i64::from(width)) as usize;
    let bottom = (i64::from(mask.y) + i64::from(mask.height)).clamp(0, i64::from(height)) as usize;
    let w = right - x;
    let h = bottom - y;
    if w == 0 || h == 0 {
        return Ok(TiledImage::new(width, height));
    }
    let samples = (mask.width as usize)
        .checked_mul(mask.height as usize)
        .filter(|&n| n <= MAX_BYTES / 8)
        .ok_or("Photoshop mask exceeds the memory budget")?;
    let chunk = crate::canvas::CHUNK_SIZE as usize;
    *retained +=
        (w + x % chunk).div_ceil(chunk) * (h + y % chunk).div_ceil(chunk) * chunk * chunk * 4;
    if *retained + samples + w * h * 4 > MAX_BYTES {
        return Err("Photoshop masks exceed the memory budget".into());
    }
    let plane = planes(bytes, mask.width as usize, mask.height as usize)?;
    let left = (x as i64 - i64::from(mask.x)) as usize;
    let top = (y as i64 - i64::from(mask.y)) as usize;
    let mut pixels = Vec::with_capacity(w * h * 4);
    for row in 0..h {
        let start = (top + row) * mask.width as usize + left;
        for &reveal in &plane[start..start + w] {
            pixels.extend([0, 0, 0, 255 - reveal]);
        }
    }
    Ok(TiledImage::from_region_rgba(
        width, height, &pixels, w as u32, h as u32, x as i32, y as i32,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn section(body: &[u8]) -> Vec<u8> {
        [(body.len() as u32).to_be_bytes().as_slice(), body].concat()
    }

    fn record(
        bounds: [i32; 4],
        name: &[u8],
        key: &[u8; 4],
        opacity: u8,
        flags: u8,
        mask: &[u8],
        tag: Option<(&[u8; 4], &[u8])>,
    ) -> Vec<u8> {
        let mut data: Vec<_> = bounds.into_iter().flat_map(i32::to_be_bytes).collect();
        let ids = if mask.is_empty() {
            vec![0i16, 1, 2, -1]
        } else {
            vec![0, 1, 2, -1, -2]
        };
        data.extend((ids.len() as u16).to_be_bytes());
        for id in ids {
            data.extend(id.to_be_bytes());
            data.extend(4u32.to_be_bytes()); // two raw bytes plus compression code
        }
        data.extend(b"8BIM");
        data.extend(key);
        data.extend([opacity, 0, flags, 0]);
        let mut extra = section(mask);
        extra.extend(section(&[]));
        extra.push(name.len() as u8);
        extra.extend(name);
        extra.resize(extra.len().next_multiple_of(4), 0);
        if let Some((key, data)) = tag {
            extra.extend(b"8BIM");
            extra.extend(key);
            extra.extend(section(data));
            if !data.len().is_multiple_of(2) {
                extra.push(0);
            }
        }
        data.extend(section(&extra));
        data
    }

    fn fixture(unknown_blend: bool) -> Vec<u8> {
        fixture_with_tag(unknown_blend, None)
    }

    fn fixture_with_tag(unknown_blend: bool, tag: Option<(&[u8; 4], &[u8])>) -> Vec<u8> {
        let mut file = b"8BPS\0\x01\0\0\0\0\0\0\0\x03".to_vec();
        file.extend(2u32.to_be_bytes());
        file.extend(3u32.to_be_bytes());
        file.extend([0, 8, 0, 3]);
        file.extend(section(&[]));
        file.extend(section(&[]));
        let mut info = 2u16.to_be_bytes().to_vec();
        let mut mask: Vec<_> = [0i32, 0, 1, 2]
            .into_iter()
            .flat_map(i32::to_be_bytes)
            .collect();
        mask.extend([255, 0, 0, 0]);
        info.extend(record(
            [0, -1, 1, 1],
            b"Top",
            if unknown_blend { b"??? " } else { b"mul " },
            128,
            2,
            &mask,
            tag,
        ));
        info.extend(record([1, 1, 2, 3], b"Base", b"norm", 255, 0, &[], None));
        for values in [
            [10, 20],
            [30, 40],
            [50, 60],
            [70, 80],
            [100, 200],
            [90, 100],
            [110, 120],
            [130, 140],
            [150, 160],
        ] {
            info.extend([0, 0, values[0], values[1]]);
        }
        let mut layers = section(&info);
        layers.extend(section(&[]));
        file.extend(section(&layers));
        file.extend([0, 0]);
        file.extend([7; 6]);
        file.extend([8; 6]);
        file.extend([9; 6]);
        file
    }

    #[test]
    fn photoshop_raster_metadata_keeps_layers_but_active_features_use_composite() {
        let ordinary = decode(&fixture(false)).unwrap();
        for (key, value) in [
            (b"lnsr", &[0, 0, 0, 0][..]),
            (b"iOpa", &[255][..]),
            (b"clbl", &[1, 0, 0, 0][..]),
            (b"knko", &[0, 0, 0, 0][..]),
        ] {
            let imported = decode(&fixture_with_tag(false, Some((key, value)))).unwrap();
            assert!(imported.fallback.is_none(), "{key:?}");
            assert_eq!(imported.canvas.layers.len(), 2);
            assert_eq!(imported.canvas.composite(), ordinary.canvas.composite());
        }
        for (key, value) in [(b"iOpa", &[128][..]), (b"lsct", &[0, 0, 0, 1][..])] {
            let imported = decode(&fixture_with_tag(false, Some((key, value)))).unwrap();
            assert_eq!(imported.fallback, Some(FallbackReason::LayerTag(*key)));
            assert_eq!(imported.canvas.layers.len(), 1);
        }
    }

    #[test]
    fn layer_order_offsets_opacity_visibility_and_live_mask_are_preserved() {
        let import = decode(&fixture(false)).unwrap();
        assert!(import.fallback.is_none());
        let layers = import.canvas.layers;
        assert_eq!(layers.len(), 2);
        assert_eq!(layers[0].name, "Base");
        assert_eq!(layers[0].pixels.get_pixel(2, 1).0, [100, 120, 140, 160]);
        let top = &layers[1];
        assert_eq!(top.name, "Top");
        assert_eq!(top.pixels.get_pixel(0, 0).0, [20, 40, 60, 80]);
        assert_eq!(top.pixels.get_pixel(1, 0).0[3], 0);
        assert!(!top.visible);
        assert_eq!(top.opacity, 128.0 / 255.0);
        assert_eq!(top.blend_mode, BlendMode::Multiply);
        assert!(top.mask_enabled);
        assert_eq!(top.mask.as_ref().unwrap().get_pixel(1, 0).0[3], 55);
    }

    #[test]
    fn unsupported_layer_uses_saved_composite_and_truncated_input_is_rejected() {
        let file = fixture(true);
        let import = decode(&file).unwrap();
        assert_eq!(import.fallback, Some(FallbackReason::Blend(*b"??? ")));
        assert_eq!(import.canvas.layers.len(), 1);
        assert_eq!(
            import.canvas.layers[0].pixels.get_pixel(2, 1).0,
            [7, 8, 9, 255]
        );
        assert!(decode(&file[..file.len() - 1]).is_err());
        let supported = fixture(false);
        assert!(decode(&supported[..supported.len() - 25]).is_err());
    }

    #[test]
    fn rle_and_zip_prediction_decode_to_exact_samples() {
        assert_eq!(
            planes(&mut Reader(&[0, 1, 0, 4, 0, 6, 255, 8]), 3, 1).unwrap(),
            [6, 8, 8]
        );
        assert!(planes(&mut Reader(&[0, 1, 0, 2, 250, 8]), 3, 1).is_err());
        use std::io::Write;
        let mut zip = flate2::write::ZlibEncoder::new(vec![0, 3], flate2::Compression::fast());
        zip.write_all(&[20, 10, 30]).unwrap();
        let compressed = zip.finish().unwrap();
        assert_eq!(
            planes(&mut Reader(&compressed), 3, 1).unwrap(),
            [20, 30, 60]
        );
    }
}
