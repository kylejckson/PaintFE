//! Pure-Rust HEIF/HEIC image decoder (and, later, encoder).
//!
//! ```no_run
//! let bytes = std::fs::read("photo.heic").unwrap();
//! let image = heifer::decode(&bytes).unwrap();
//! let rgba: Vec<u8> = image.to_rgba8();
//! println!("{}x{}, alpha: {}", image.width, image.height, image.has_alpha);
//! ```
//!
//! This crate ties together the container ([`heifer_isobmff`]) and the HEVC codec
//! ([`heifer_hevc_dec`], [`heifer_hevc_enc`]).

pub mod color;
pub mod hdr;
pub mod image;
#[cfg(feature = "image")]
pub mod image_crate;
pub mod metadata;

pub use hdr::{GainMap, HdrImage, decode_hdr, read_gain_map};
pub use heifer_hevc_dec;
pub use heifer_hevc_enc;
pub use heifer_isobmff;
pub use image::Image;
pub use metadata::{Exif, Metadata, item_metadata, read_metadata};

use color::{ColorParams, frame_to_rgba, frame_to_rgba_into};
use heifer_hevc_dec::decoder::{DecodeOptions, decode_picture};
use heifer_hevc_dec::recon::Frame;
use heifer_isobmff::boxes::{ColorInfo, Property};
use heifer_isobmff::{HeifFile, ItemId};

/// Errors produced while decoding a HEIF file.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The HEIF container is invalid.
    #[error("container: {0}")]
    Container(#[from] heifer_isobmff::Error),
    /// An HEVC bitstream could not be decoded.
    #[error("HEVC: {0}")]
    Hevc(#[from] heifer_hevc_dec::Error),
    /// The file uses a feature heifer does not support yet.
    #[error("unsupported: {0}")]
    Unsupported(String),
    /// The file is inconsistent.
    #[error("invalid file: {0}")]
    Invalid(&'static str),
}

/// Limits protecting against malicious files.
const MAX_PIXELS: u64 = 64_000_000;
const MAX_DEPTH: u32 = 4;

/// Decoding options.
#[derive(Debug, Clone, Copy, Default)]
pub struct Options {
    /// Maximum number of threads used to decode grid tiles in parallel. `0` uses all
    /// available cores; `1` decodes sequentially.
    pub max_threads: usize,
    /// Maximum size, in pixels, of any decoded or assembled image (protects against malicious
    /// files). `0` means the default, 64 million pixels.
    pub max_pixels: u64,
}

impl Options {
    fn max_pixels(&self) -> u64 {
        if self.max_pixels == 0 {
            MAX_PIXELS
        } else {
            self.max_pixels
        }
    }

    fn threads(&self) -> usize {
        let available = std::thread::available_parallelism().map_or(1, |n| n.get());
        if self.max_threads == 0 {
            available
        } else {
            self.max_threads.min(available)
        }
    }
}

/// Decodes the primary image of a HEIF/HEIC file, with its alpha channel if any, and applies
/// the crop, rotation and mirror transforms stored in the file. Grid tiles are decoded in
/// parallel on all available cores.
pub fn decode(bytes: &[u8]) -> Result<Image, Error> {
    decode_with_options(bytes, &Options::default())
}

/// Like [`decode`], with options.
pub fn decode_with_options(bytes: &[u8], options: &Options) -> Result<Image, Error> {
    let file = HeifFile::parse(bytes)?;
    decode_item_with(&file, file.primary_id, 0, options)
}

/// Decodes one image item: coded (`hvc1`) or derived (`grid`, `iovl`), then applies its
/// transformative properties.
pub fn decode_item(file: &HeifFile<'_>, id: ItemId, depth: u32) -> Result<Image, Error> {
    decode_item_with(file, id, depth, &Options::default())
}

fn decode_item_with(
    file: &HeifFile<'_>,
    id: ItemId,
    depth: u32,
    options: &Options,
) -> Result<Image, Error> {
    decode_item_mode(file, id, depth, options, false, true)
}

fn decode_item_mode(
    file: &HeifFile<'_>,
    id: ItemId,
    depth: u32,
    options: &Options,
    alpha_samples: bool,
    transform: bool,
) -> Result<Image, Error> {
    if depth > MAX_DEPTH {
        return Err(Error::Invalid("derived images nested too deeply"));
    }
    let item = file.item(id)?;
    let mut image = match &item.item_type.0 {
        b"hvc1" => decode_hvc1(file, id, options, alpha_samples)?,
        b"grid" => decode_grid(file, id, depth, options, alpha_samples)?,
        b"iovl" => decode_overlay(file, id, depth, options, alpha_samples)?,
        _ => {
            return Err(Error::Unsupported(format!(
                "image item type `{}`",
                item.item_type
            )));
        }
    };
    attach_alpha(file, id, &mut image, depth, options)?;
    if !transform {
        return Ok(image);
    }
    // Transformative properties, in association order (clap, then irot, then imir).
    for property in file.item_properties(id)? {
        image = match *property {
            Property::CleanAperture {
                width,
                height,
                horiz_offset,
                vert_offset,
            } => clean_aperture(&image, width, height, horiz_offset, vert_offset)?,
            Property::Rotation(q) => image.rotate_ccw(q),
            Property::Mirror(top_bottom) => image.mirror(top_bottom),
            _ => image,
        };
    }
    Ok(image)
}

/// Colour parameters of an item: its `colr` (nclx) property if present, otherwise those of the
/// HEVC stream (ISO/IEC 23008-12 defers to the coded bitstream).
fn color_params(
    file: &HeifFile<'_>,
    id: ItemId,
    frame: &heifer_hevc_dec::recon::Frame,
) -> Result<ColorParams, Error> {
    Ok(file
        .item_properties(id)?
        .find_map(|p| match p {
            Property::Color(ColorInfo::Nclx {
                matrix, full_range, ..
            }) => Some(ColorParams {
                matrix: *matrix,
                full_range: *full_range,
            }),
            _ => None,
        })
        .unwrap_or_else(|| ColorParams::from_frame(frame)))
}

/// Decodes the HEVC picture of a coded image item.
fn decode_hvc1_frame(file: &HeifFile<'_>, id: ItemId, options: &Options) -> Result<Frame, Error> {
    let hevc_options = DecodeOptions {
        max_pixels: options.max_pixels(),
        ..Default::default()
    };
    Ok(decode_picture(&file.hevc_bitstream(id)?, hevc_options)?)
}

fn decode_hvc1(
    file: &HeifFile<'_>,
    id: ItemId,
    options: &Options,
    alpha_samples: bool,
) -> Result<Image, Error> {
    let frame = decode_hvc1_frame(file, id, options)?;
    let params = color_params(file, id, &frame)?;
    let mut image = if alpha_samples {
        let bits = frame.bit_depth[0];
        let mut image = Image::filled(frame.widths[0], frame.heights[0], bits, [0; 4]);
        let max = image.max_value();
        for (pixel, &sample) in image.data.chunks_exact_mut(4).zip(&frame.planes[0]) {
            let a = alpha_sample(sample, bits, params.full_range);
            pixel.copy_from_slice(&[a, a, a, max]);
        }
        image
    } else {
        frame_to_rgba(&frame, params)
    };
    if let Some((w, h)) = file.image_size(id)?
        && (w, h) != (image.width, image.height)
    {
        if w == 0 || h == 0 || w > image.width || h > image.height {
            return Err(Error::Invalid("spatial extent exceeds coded image"));
        }
        // `ispe` is authoritative; HEVC may code a slightly larger picture.
        image = image.crop(0, 0, w, h);
    }

    Ok(image)
}

fn alpha_sample(sample: u16, bits: u8, full_range: bool) -> u16 {
    let max = (1u32 << bits) - 1;
    if full_range {
        return u32::from(sample).min(max) as u16;
    }
    let factor = 1u32 << (bits - 8);
    let value = u32::from(sample).saturating_sub(16 * factor);
    ((value * max + 219 * factor / 2) / (219 * factor)).min(max) as u16
}

// PaintFE patch: auxiliary alpha belongs to the derived image, not just hvc1.
fn attach_alpha(
    file: &HeifFile<'_>,
    id: ItemId,
    image: &mut Image,
    depth: u32,
    options: &Options,
) -> Result<(), Error> {
    for aux in file.referencing_items(id, b"auxl") {
        let is_alpha = file.item_properties(aux)?.any(|p| matches!(p, Property::AuxiliaryType(t)
            if t == "urn:mpeg:hevc:2015:auxid:1" || t == "urn:mpeg:mpegB:cicp:systems:auxiliary:alpha"));
        if !is_alpha {
            continue;
        }
        // Parent transforms apply to colour and alpha together, as in upstream's coded-alpha path.
        let alpha = decode_item_mode(file, aux, depth + 1, options, true, false)?;
        if alpha.width != image.width || alpha.height != image.height {
            return Err(Error::Invalid("alpha dimensions do not match the image"));
        }
        let source_max = u32::from(alpha.max_value());
        let target_max = u32::from(image.max_value());
        for (pixel, a) in image
            .data
            .chunks_exact_mut(4)
            .zip(alpha.data.chunks_exact(4))
        {
            pixel[3] = ((u32::from(a[0]) * target_max + source_max / 2) / source_max) as u16;
        }
        image.has_alpha = true;
        break;
    }
    Ok(())
}

fn check_size(w: u32, h: u32, max_pixels: u64) -> Result<(), Error> {
    if w == 0 || h == 0 || u64::from(w) * u64::from(h) > max_pixels {
        return Err(Error::Invalid("image size is zero or too large"));
    }
    Ok(())
}

fn decode_grid(
    file: &HeifFile<'_>,
    id: ItemId,
    depth: u32,
    options: &Options,
    alpha_samples: bool,
) -> Result<Image, Error> {
    let grid = file.grid(id)?;
    let tiles = file.referenced_items(id, b"dimg");
    let columns = u32::from(grid.columns);
    if tiles.is_empty() || tiles.len() != usize::from(grid.rows) * usize::from(grid.columns) {
        return Err(Error::Invalid(
            "grid tile count does not match its rows and columns",
        ));
    }
    check_size(grid.output_width, grid.output_height, options.max_pixels())?;
    if let Some(size) = file.image_size(id)?
        && size != (grid.output_width, grid.output_height)
    {
        return Err(Error::Invalid(
            "grid spatial extents do not match its output",
        ));
    }

    // Tile size and bit depth come from the tiles' `ispe` and `hvcC` properties, so that all
    // tiles can be decoded in parallel. Otherwise, decode the first tile to find out.
    let mut first_tile = None;
    let (tw, th, bit_depth) = match (file.image_size(tiles[0])?, file.hevc_config(tiles[0])?) {
        (Some((w, h)), Some(c)) => (w, h, c.bit_depth_luma.max(c.bit_depth_chroma)),
        _ => {
            let tile = decode_item_mode(file, tiles[0], depth + 1, options, alpha_samples, true)?;
            let size = (tile.width, tile.height, tile.bit_depth);
            first_tile = Some(tile);
            size
        }
    };
    check_size(tw, th, options.max_pixels())?;
    if u64::from(grid.output_width) > u64::from(columns) * u64::from(tw)
        || u64::from(grid.output_height) > u64::from(grid.rows) * u64::from(th)
    {
        return Err(Error::Invalid("grid output exceeds tile coverage"));
    }
    let mut canvas = Image::filled(grid.output_width, grid.output_height, bit_depth, [0; 4]);
    let (canvas_w, canvas_h) = (canvas.width, canvas.height);
    let stride = canvas_w as usize * 4;
    // Each tile gets its own row segments of the canvas, so tiles are written concurrently.
    let mut segments: Vec<Vec<&mut [u16]>> = (0..tiles.len()).map(|_| Vec::new()).collect();
    for (y, row) in canvas.data.chunks_exact_mut(stride).enumerate() {
        let first = y / th as usize * columns as usize;
        for (column, segment) in row.chunks_mut(tw as usize * 4).enumerate() {
            if let Some(rows) = segments.get_mut(first + column) {
                rows.push(segment);
            }
        }
    }
    let segments: Vec<std::sync::Mutex<Vec<&mut [u16]>>> =
        segments.into_iter().map(std::sync::Mutex::new).collect();
    let has_alpha = std::sync::atomic::AtomicBool::new(false);
    // Plain tiles are converted straight into the canvas, without an intermediate image.
    let plain = !alpha_samples
        && first_tile.is_none()
        && tiles
            .iter()
            .all(|&t| is_plain_coded_image(file, t).unwrap_or(false));
    let place = |i: usize, tile: Option<&Image>| -> Result<(), Error> {
        let (x0, y0) = ((i as u32 % columns) * tw, (i as u32 / columns) * th);
        let (w, h) = (
            tw.min(canvas_w.saturating_sub(x0)) as usize,
            th.min(canvas_h.saturating_sub(y0)) as usize,
        );
        let tile_image;
        let tile = match tile {
            Some(t) => Some(t),
            None if plain => None,
            None => {
                tile_image =
                    decode_item_mode(file, tiles[i], depth + 1, options, alpha_samples, true)?;
                Some(&tile_image)
            }
        };
        let frame = match tile {
            Some(_) => None,
            None => {
                let frame = decode_hvc1_frame(file, tiles[i], options)?;
                let params = color_params(file, tiles[i], &frame)?;
                Some((frame, params))
            }
        };
        let size_ok = match (tile, &frame) {
            (Some(t), _) => (t.width, t.height, t.bit_depth) == (tw, th, bit_depth),
            (None, Some((f, _))) => {
                f.widths[0] >= tw
                    && f.heights[0] >= th
                    && f.bit_depth[0].max(f.bit_depth[1]) == bit_depth
            }
            (None, None) => false,
        };
        if !size_ok {
            return Err(Error::Invalid(
                "grid tiles have different sizes or bit depths",
            ));
        }
        let mut out = segments[i].lock().unwrap_or_else(|e| e.into_inner());
        match (tile, frame) {
            (Some(tile), _) => {
                // Copy rows (no blending): tiles are opaque unless they carry alpha.
                for (dst, src) in out
                    .iter_mut()
                    .zip(tile.data.chunks_exact(tw as usize * 4))
                    .take(h)
                {
                    dst[..w * 4].copy_from_slice(&src[..w * 4]);
                }
                if tile.has_alpha {
                    has_alpha.store(true, std::sync::atomic::Ordering::Relaxed);
                }
            }
            (None, Some((frame, params))) => frame_to_rgba_into(&frame, params, &mut out, w, h),
            (None, None) => unreachable!("checked above"),
        }
        Ok(())
    };
    let start = if let Some(tile) = &first_tile {
        place(0, Some(tile))?;
        1
    } else {
        0
    };

    // Tiles are independent HEVC streams: decode them on several threads.
    let threads = options.threads().min(tiles.len() - start);
    if threads <= 1 {
        for i in start..tiles.len() {
            place(i, None)?;
        }
    } else {
        let next = std::sync::atomic::AtomicUsize::new(start);
        let error = std::sync::Mutex::new(None);
        std::thread::scope(|s| {
            for _ in 0..threads {
                s.spawn(|| {
                    loop {
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if i >= tiles.len() {
                            break;
                        }
                        if let Err(e) = place(i, None) {
                            error
                                .lock()
                                .unwrap_or_else(|e| e.into_inner())
                                .get_or_insert(e);
                            next.store(tiles.len(), std::sync::atomic::Ordering::Relaxed);
                        }
                    }
                });
            }
        });
        if let Some(e) = error.into_inner().unwrap_or_else(|e| e.into_inner()) {
            return Err(e);
        }
    }
    drop(segments);
    canvas.has_alpha = has_alpha.into_inner();
    Ok(canvas)
}

/// Whether `id` is a coded image without transformative properties or alpha plane, so that
/// its decoded picture can be used as is.
fn is_plain_coded_image(file: &HeifFile<'_>, id: ItemId) -> Result<bool, Error> {
    if file.item(id)?.item_type.0 != *b"hvc1" || !file.referencing_items(id, b"auxl").is_empty() {
        return Ok(false);
    }
    Ok(!file.item_properties(id)?.any(|p| {
        matches!(
            p,
            Property::CleanAperture { .. } | Property::Rotation(_) | Property::Mirror(_)
        )
    }))
}

fn decode_overlay(
    file: &HeifFile<'_>,
    id: ItemId,
    depth: u32,
    options: &Options,
    alpha_samples: bool,
) -> Result<Image, Error> {
    let data = file.item_data(id)?;
    let inputs = file.referenced_items(id, b"dimg");
    let mut r = data.iter().copied();
    let mut next = || r.next().ok_or(Error::Invalid("truncated iovl data"));
    if next()? != 0 {
        return Err(Error::Unsupported("iovl version".into()));
    }
    let large = next()? & 1 != 0;
    let mut u16v = || -> Result<u16, Error> { Ok(u16::from_be_bytes([next()?, next()?])) };
    let fill = [u16v()?, u16v()?, u16v()?, u16v()?];
    drop(u16v);
    let mut field = |signed: bool| -> Result<i64, Error> {
        Ok(if large {
            let v = [next()?, next()?, next()?, next()?];
            if signed {
                i64::from(i32::from_be_bytes(v))
            } else {
                i64::from(u32::from_be_bytes(v))
            }
        } else {
            let v = [next()?, next()?];
            if signed {
                i64::from(i16::from_be_bytes(v))
            } else {
                i64::from(u16::from_be_bytes(v))
            }
        })
    };
    let (w, h) = (field(false)? as u32, field(false)? as u32);
    check_size(w, h, options.max_pixels())?;
    let offsets = (0..inputs.len())
        .map(|_| Ok((field(true)?, field(true)?)))
        .collect::<Result<Vec<_>, Error>>()?;

    let images = inputs
        .iter()
        .map(|&i| decode_item_mode(file, i, depth + 1, options, alpha_samples, true))
        .collect::<Result<Vec<_>, _>>()?;
    let bit_depth = images.iter().map(|i| i.bit_depth).max().unwrap_or(8);
    // canvas_fill_value is given on 16 bits.
    let scale = |v: u16| ((u32::from(v) * ((1u32 << bit_depth) - 1) + 32767) / 65535) as u16;
    let mut canvas = Image::filled(w, h, bit_depth, fill.map(scale));
    canvas.has_alpha = fill[3] != 65535;
    for (img, (x, y)) in images.iter().zip(offsets) {
        canvas.draw(img, x, y);
    }
    let opaque = canvas.max_value();
    canvas.has_alpha = canvas.data.chunks_exact(4).any(|p| p[3] != opaque);
    Ok(canvas)
}

/// Applies a `clap` (clean aperture) crop, centred as specified by ISO/IEC 14496-12.
fn clean_aperture(
    img: &Image,
    width: (u32, u32),
    height: (u32, u32),
    horiz: (i32, u32),
    vert: (i32, u32),
) -> Result<Image, Error> {
    if width.1 == 0 || height.1 == 0 || horiz.1 == 0 || vert.1 == 0 {
        return Err(Error::Invalid("clap with a zero denominator"));
    }
    let cw = f64::from(width.0) / f64::from(width.1);
    let ch = f64::from(height.0) / f64::from(height.1);
    let cx = f64::from(horiz.0) / f64::from(horiz.1) + (f64::from(img.width) - 1.0) / 2.0;
    let cy = f64::from(vert.0) / f64::from(vert.1) + (f64::from(img.height) - 1.0) / 2.0;
    let left = (cx - (cw - 1.0) / 2.0).round().max(0.0) as u32;
    let top = (cy - (ch - 1.0) / 2.0).round().max(0.0) as u32;
    Ok(img.crop(left, top, cw.round() as u32, ch.round() as u32))
}

#[cfg(test)]
mod paintfe_alpha_tests {
    #[test]
    fn alpha_ranges_preserve_upstream_coded_alpha_scaling() {
        assert_eq!(super::alpha_sample(128, 8, true), 128);
        assert_eq!(super::alpha_sample(16, 8, false), 0);
        assert_eq!(super::alpha_sample(235, 8, false), 255);
        assert_eq!(super::alpha_sample(0, 8, false), 0);
        assert_eq!(super::alpha_sample(255, 8, false), 255);
        assert_eq!(super::alpha_sample(64, 10, false), 0);
        assert_eq!(super::alpha_sample(940, 10, false), 1023);
    }
}
