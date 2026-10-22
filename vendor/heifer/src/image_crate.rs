//! Integration with the [`image`](https://docs.rs/image) crate (feature `image`).
//!
//! ```no_run
//! heifer::image_crate::register_image_decoder_hooks();
//! let img = image::open("photo.heic").unwrap(); // decoded by heifer
//! ```
//!
//! [`HeifDecoder`] can also be used directly with `image::DynamicImage::from_decoder`.

use std::io::Read;

use image::error::{DecodingError, ImageFormatHint};
use image::metadata::Orientation;
use image::{ColorType, ImageDecoder, ImageError, ImageResult};

use crate::{Image, Metadata, Options};

/// File extensions handled by heifer.
const EXTENSIONS: [&str; 3] = ["heic", "heif", "hif"];

/// HEVC-based HEIF brands recognised from the file content (the `ftyp` major brand).
const BRANDS: [&[u8; 4]; 4] = [b"heic", b"heix", b"heim", b"heis"];

/// Registers heifer with the `image` crate, for the `.heic`, `.heif` and `.hif` extensions and
/// for content detection of HEVC-coded HEIF files. Returns `false` if another decoder was
/// already registered for one of these extensions.
///
/// After this call, `image::open`, `image::ImageReader` and `image::load_from_memory` decode
/// HEIC files with heifer.
pub fn register_image_decoder_hooks() -> bool {
    let mut all = true;
    for ext in EXTENSIONS {
        all &= image::hooks::register_decoding_hook(
            ext.into(),
            Box::new(|reader| Ok(Box::new(HeifDecoder::new(reader)?))),
        );
    }
    // Box size (4 bytes, any value) then `ftyp` and the major brand.
    const MASK: &[u8] = b"\x00\x00\x00\x00\xff\xff\xff\xff\xff\xff\xff\xff";
    static SIGNATURES: [[u8; 12]; 4] = {
        let mut s = [[0u8; 12]; 4];
        let mut i = 0;
        while i < 4 {
            let mut j = 0;
            while j < 4 {
                s[i][4 + j] = b"ftyp"[j];
                s[i][8 + j] = BRANDS[i][j];
                j += 1;
            }
            i += 1;
        }
        s
    };
    for signature in &SIGNATURES {
        image::hooks::register_format_detection_hook("heic".into(), signature, Some(MASK));
    }
    all
}

/// An [`image::ImageDecoder`] for HEIF/HEIC files.
///
/// The image is decoded when the decoder is created; crop, rotation and mirror transforms are
/// already applied, so [`ImageDecoder::orientation`] reports no remaining transform.
pub struct HeifDecoder {
    image: Image,
    metadata: Metadata,
}

impl HeifDecoder {
    /// Reads a whole HEIF file and decodes its primary image.
    pub fn new(reader: impl Read) -> ImageResult<Self> {
        let mut bytes = Vec::new();
        reader
            .take(256 * 1024 * 1024 + 1)
            .read_to_end(&mut bytes)
            .map_err(ImageError::IoError)?;
        if bytes.len() > 256 * 1024 * 1024 {
            return Err(ImageError::IoError(std::io::Error::other(
                "HEIF input exceeds 256 MiB",
            )));
        }
        Self::from_bytes(&bytes, &Options::default())
    }

    /// Decodes a HEIF file held in memory, with heifer options.
    pub fn from_bytes(bytes: &[u8], options: &Options) -> ImageResult<Self> {
        let image = crate::decode_with_options(bytes, options).map_err(decoding_error)?;
        let metadata = crate::read_metadata(bytes).map_err(decoding_error)?;
        Ok(Self { image, metadata })
    }
}

fn decoding_error(e: crate::Error) -> ImageError {
    ImageError::Decoding(DecodingError::new(ImageFormatHint::Name("HEIF".into()), e))
}

impl ImageDecoder for HeifDecoder {
    fn dimensions(&self) -> (u32, u32) {
        (self.image.width, self.image.height)
    }

    fn color_type(&self) -> ColorType {
        match (self.image.bit_depth > 8, self.image.has_alpha) {
            (false, false) => ColorType::Rgb8,
            (false, true) => ColorType::Rgba8,
            (true, false) => ColorType::Rgb16,
            (true, true) => ColorType::Rgba16,
        }
    }

    fn icc_profile(&mut self) -> ImageResult<Option<Vec<u8>>> {
        Ok(self.metadata.icc_profile.clone())
    }

    fn exif_metadata(&mut self) -> ImageResult<Option<Vec<u8>>> {
        Ok(self.metadata.exif.clone())
    }

    fn xmp_metadata(&mut self) -> ImageResult<Option<Vec<u8>>> {
        Ok(self.metadata.xmp.clone())
    }

    /// HEIF transforms (`irot`/`imir`) are already applied, and the EXIF orientation is only
    /// informative in HEIF: nothing is left to do.
    fn orientation(&mut self) -> ImageResult<Orientation> {
        Ok(Orientation::NoTransforms)
    }

    fn set_limits(&mut self, limits: image::Limits) -> ImageResult<()> {
        limits.check_support(&image::LimitSupport::default())?;
        limits.check_dimensions(self.image.width, self.image.height)?;
        Ok(())
    }

    fn read_image(self, buf: &mut [u8]) -> ImageResult<()> {
        let channels = if self.image.has_alpha { 4 } else { 3 };
        let max = u32::from(self.image.max_value());
        let pixels = self.image.data.chunks_exact(4).map(|p| &p[..channels]);
        if self.image.bit_depth > 8 {
            // 16-bit native endian, scaled to the full 0..=65535 range.
            for (out, v) in buf.chunks_exact_mut(2).zip(pixels.flatten()) {
                let v16 = ((u32::from(*v) * 65535 + max / 2) / max) as u16;
                out.copy_from_slice(&v16.to_ne_bytes());
            }
        } else {
            for (out, v) in buf.iter_mut().zip(pixels.flatten()) {
                *out = *v as u8;
            }
        }
        Ok(())
    }

    fn read_image_boxed(self: Box<Self>, buf: &mut [u8]) -> ImageResult<()> {
        (*self).read_image(buf)
    }
}
