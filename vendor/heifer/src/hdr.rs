//! HDR photos with an Apple gain map.
//!
//! Recent iPhones store a normal (SDR) image plus a *gain map*: a small grayscale auxiliary
//! image (`urn:com:apple:photo:2020:aux:hdrgainmap`) telling how much brighter each area may be
//! shown on an HDR screen. The maximum boost, the *headroom*, comes from the EXIF MakerNote.
//! This module follows Apple's published method ("Applying Apple HDR effect to your photos"):
//!
//! `hdr = sdr × (1 + (headroom − 1) × gain)`, in linear light, where 1.0 is SDR white.
//!
//! Two details follow Apple's own decoder (ImageIO) rather than that document, so that the result
//! matches what Apple devices show: gain map samples are linearized with a 2.2 gamma (the
//! document says Rec.709), and [`decode_hdr`] limits the headroom to [`APPLE_MAX_HEADROOM`].
//! On an iPhone 12 Pro photo, heifer then matches ImageIO's HDR decoding within 0.3% on average.

use heifer_isobmff::HeifFile;
use heifer_isobmff::boxes::Property;

use crate::{Error, Exif, Options, decode_item_with};

/// Auxiliary type of Apple HDR gain maps.
pub const APPLE_GAIN_MAP: &str = "urn:com:apple:photo:2020:aux:hdrgainmap";

/// Headroom limit applied by Apple's decoder (ImageIO) when decoding to HDR.
pub const APPLE_MAX_HEADROOM: f32 = 4.0;

/// An HDR gain map, ready to be applied.
#[derive(Debug, Clone, PartialEq)]
pub struct GainMap {
    /// Width of the gain map (usually half the image width), after rotation and mirroring.
    pub width: u32,
    /// Height of the gain map.
    pub height: u32,
    /// Linear gains in `0.0..=1.0`, one per gain map pixel, row-major.
    pub data: Vec<f32>,
    /// Ratio of the brightest HDR white to SDR white (at least 1), from the photo's metadata.
    pub headroom: f32,
}

impl GainMap {
    /// The gain at image position (`x`, `y`) of an image of size `width`×`height`, with
    /// bilinear interpolation.
    pub fn sample(&self, x: u32, y: u32, width: u32, height: u32) -> f32 {
        let coord = |p: u32, out: u32, size: u32| {
            let f =
                ((p as f32 + 0.5) * size as f32 / out as f32 - 0.5).clamp(0.0, (size - 1) as f32);
            let i = (f as u32).min(size - 1);
            (i, (i + 1).min(size - 1), f - i as f32)
        };
        let (x0, x1, fx) = coord(x, width, self.width);
        let (y0, y1, fy) = coord(y, height, self.height);
        let at = |x: u32, y: u32| self.data[(y * self.width + x) as usize];
        let top = at(x0, y0) * (1.0 - fx) + at(x1, y0) * fx;
        let bottom = at(x0, y1) * (1.0 - fx) + at(x1, y1) * fx;
        top * (1.0 - fy) + bottom * fy
    }
}

/// A decoded HDR image.
#[derive(Debug, Clone, PartialEq)]
pub struct HdrImage {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Peak value of `data` (1.0 when the file has no usable gain map).
    pub headroom: f32,
    /// Linear RGB, 3 values per pixel, row-major. 1.0 is SDR white; highlights go up to
    /// `headroom`. The primaries are those of the image (Display P3 for iPhone photos).
    pub data: Vec<f32>,
}

/// Reads the HDR gain map of the primary image, if the file has a valid Apple gain map.
pub fn read_gain_map(bytes: &[u8]) -> Result<Option<GainMap>, Error> {
    read_gain_map_with_options(bytes, &Options::default())
}

/// Like [`read_gain_map`], with options.
pub fn read_gain_map_with_options(
    bytes: &[u8],
    options: &Options,
) -> Result<Option<GainMap>, Error> {
    gain_map(&HeifFile::parse(bytes)?, options)
}

/// Decodes the primary image in HDR, as Apple's decoder does: the SDR image with its gain map
/// applied, with the headroom limited to [`APPLE_MAX_HEADROOM`]. Files without a usable gain
/// map give the SDR image in linear light, with a headroom of 1.
pub fn decode_hdr(bytes: &[u8]) -> Result<HdrImage, Error> {
    decode_hdr_with_options(bytes, &Options::default(), APPLE_MAX_HEADROOM)
}

/// Like [`decode_hdr`], with options and a headroom limit, for example the headroom of the
/// target display (`f32::INFINITY` for the full headroom stored in the photo).
pub fn decode_hdr_with_options(
    bytes: &[u8],
    options: &Options,
    max_headroom: f32,
) -> Result<HdrImage, Error> {
    let file = HeifFile::parse(bytes)?;
    let sdr = decode_item_with(&file, file.primary_id, 0, options)?;
    let gain = gain_map(&file, options)?.map(|mut g| {
        g.headroom = g.headroom.min(max_headroom).max(1.0);
        g
    });

    // SDR samples to linear light (sRGB transfer, also used by Display P3).
    let max = f32::from(sdr.max_value());
    let lut: Vec<f32> = (0..=sdr.max_value())
        .map(|v| srgb_to_linear(f32::from(v) / max))
        .collect();
    let (w, h) = (sdr.width, sdr.height);
    let mut data = Vec::with_capacity(w as usize * h as usize * 3);
    for (i, px) in sdr.data.chunks_exact(4).enumerate() {
        let factor = gain.as_ref().map_or(1.0, |g| {
            let (x, y) = (i as u32 % w, i as u32 / w);
            1.0 + (g.headroom - 1.0) * g.sample(x, y, w, h)
        });
        data.extend(px[..3].iter().map(|&v| lut[usize::from(v)] * factor));
    }
    Ok(HdrImage {
        width: w,
        height: h,
        headroom: gain.as_ref().map_or(1.0, |g| g.headroom),
        data,
    })
}

fn gain_map(file: &HeifFile<'_>, options: &Options) -> Result<Option<GainMap>, Error> {
    let primary = file.primary_id;
    let mut aux_items = file.referencing_items(primary, b"auxl").into_iter();
    let found = aux_items.try_fold(None, |found, aux| -> Result<_, Error> {
        if found.is_some() {
            return Ok(found);
        }
        let is_gain_map = file
            .item_properties(aux)?
            .any(|p| matches!(p, Property::AuxiliaryType(t) if t == APPLE_GAIN_MAP));
        Ok(is_gain_map.then_some(aux))
    })?;
    let Some(aux) = found else {
        return Ok(None);
    };
    // Apple: the gain map is valid only with the `HDRGainMapVersion` key and both MakerNote tags.
    let has_version = file
        .xmp(aux)?
        .is_some_and(|x| x.windows(17).any(|w| w == b"HDRGainMapVersion"));
    let exif = file.exif(primary)?;
    let headroom = exif
        .as_deref()
        .and_then(Exif::parse)
        .and_then(|e| e.apple_hdr_headroom());
    let (true, Some(headroom)) = (has_version, headroom) else {
        return Ok(None);
    };

    let image = decode_item_with(file, aux, 1, options)?;
    // Grayscale: the red channel holds the gain map samples, encoded with a 2.2 gamma.
    let max = f32::from(image.max_value());
    let data = image
        .data
        .chunks_exact(4)
        .map(|p| (f32::from(p[0]) / max).powf(2.2))
        .collect();
    Ok(Some(GainMap {
        width: image.width,
        height: image.height,
        data,
        headroom,
    }))
}

/// sRGB transfer function to linear light.
fn srgb_to_linear(v: f32) -> f32 {
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transfer_functions() {
        assert!((srgb_to_linear(1.0) - 1.0).abs() < 1e-6);
        assert!((srgb_to_linear(0.5) - 0.214).abs() < 1e-3);
    }

    #[test]
    fn bilinear_sampling() {
        let g = GainMap {
            width: 2,
            height: 1,
            data: vec![0.0, 1.0],
            headroom: 2.0,
        };
        // A 4-pixel-wide image over a 2-pixel gain map: edges clamp, the middle interpolates.
        let row: Vec<f32> = (0..4).map(|x| g.sample(x, 0, 4, 1)).collect();
        assert_eq!(row, vec![0.0, 0.25, 0.75, 1.0]);
    }
}
