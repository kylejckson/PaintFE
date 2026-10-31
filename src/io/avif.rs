//! Optional offline AVIF import/export; the core has no AV1 codec dependency.
use image::{ImageDecoder, ImageError, ImageResult};
use std::io::Read;

fn error(message: impl Into<String>) -> ImageError {
    ImageError::IoError(std::io::Error::other(message.into()))
}
const MAX_PIXELS: u64 = 64_000_000;

fn request(
    op: u8,
    width: u32,
    height: u32,
    quality: u8,
    settings: Option<crate::io::export::ExportOptions>,
    data: &[u8],
    tracked: bool,
) -> ImageResult<Vec<u8>> {
    let mut header = [0; 40];
    header[..8].copy_from_slice(
        if settings.is_some_and(|o| o.avif_lossless || o.avif_lossless_alpha) {
            b"PFEAVF02"
        } else {
            b"PFEAVF01"
        },
    );
    header[8] = op;
    header[12..16].copy_from_slice(&width.to_le_bytes());
    header[16..20].copy_from_slice(&height.to_le_bytes());
    header[20..24].copy_from_slice(&8u32.to_le_bytes());
    header[24] = quality.clamp(1, 100);
    header[25] = settings.map_or(8, |o| o.avif_speed);
    header[26] = settings.map_or(0, |o| {
        u8::from(o.avif_lossless) | (u8::from(o.avif_lossless_alpha) << 1)
    });
    header[32..40].copy_from_slice(&(data.len() as u64).to_le_bytes());
    #[cfg(not(target_arch = "wasm32"))]
    {
        crate::services::optional_packs::avif::exchange(header, data, tracked).map_err(error)
    }
    #[cfg(target_arch = "wasm32")]
    {
        let _ = (header, tracked);
        Err(error("AVIF packs are available in the desktop app"))
    }
}

pub fn is_avif(bytes: &[u8]) -> bool {
    if bytes.len() < 16 || &bytes[4..8] != b"ftyp" {
        return false;
    }
    let size = u32::from_be_bytes(bytes[..4].try_into().unwrap()) as usize;
    if size < 16 || size > bytes.len() {
        return false;
    }
    matches!(&bytes[8..12], b"avif" | b"avis")
        || bytes[16..size]
            .chunks_exact(4)
            .any(|b| matches!(b, b"avif" | b"avis"))
}

fn encode(image: &image::RgbaImage, quality: u8, tracked: bool) -> ImageResult<Vec<u8>> {
    if u64::from(image.width()) * u64::from(image.height()) > MAX_PIXELS {
        return Err(error("AVIF image exceeds 64 million pixels"));
    }
    let bytes = request(
        1,
        image.width(),
        image.height(),
        quality,
        None,
        image.as_raw(),
        tracked,
    )?;
    if !is_avif(&bytes) {
        return Err(error("Invalid AVIF helper output"));
    }
    Ok(bytes)
}
pub fn encode_rgba8(image: &image::RgbaImage, quality: u8) -> ImageResult<Vec<u8>> {
    encode(image, quality, true)
}
pub(crate) fn encode_options(
    image: &image::RgbaImage,
    options: crate::io::export::ExportOptions,
    tracked: bool,
) -> Result<Vec<u8>, String> {
    if u64::from(image.width()) * u64::from(image.height()) > MAX_PIXELS {
        return Err("AVIF image exceeds 64 million pixels".into());
    }
    let output = request(
        1,
        image.width(),
        image.height(),
        options.quality,
        Some(options),
        image.as_raw(),
        tracked,
    )
    .map_err(|e| format!("AVIF encoding failed. For lossless options, update the AVIF pack in Preferences > Optional features. {e}"))?;
    if !is_avif(&output) {
        return Err("Invalid AVIF helper output".into());
    }
    if options.avif_lossless || options.avif_lossless_alpha {
        let decoded = image::DynamicImage::from_decoder(
            Decoder::new_with_tracking(std::io::Cursor::new(&output), tracked)
                .map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?
        .to_rgba8();
        if decoded.dimensions() != image.dimensions()
            || decoded.pixels().zip(image.pixels()).any(|(a, b)| {
                if options.avif_lossless {
                    a != b
                } else {
                    a[3] != b[3]
                }
            })
        {
            return Err("AVIF encoder could not preserve the requested lossless pixels".into());
        }
    }
    Ok(output)
}
pub fn preview(image: &image::RgbaImage, quality: u8) -> ImageResult<(Vec<u8>, image::RgbaImage)> {
    let bytes = encode(image, quality, false)?;
    let decoded = image::DynamicImage::from_decoder(Decoder::new_with_tracking(
        std::io::Cursor::new(&bytes),
        false,
    )?)?
    .to_rgba8();
    Ok((bytes, decoded))
}

pub fn decode(reader: impl Read) -> ImageResult<image::DynamicImage> {
    image::DynamicImage::from_decoder(Decoder::new_with_tracking(reader, true)?)
}

struct Decoder {
    width: u32,
    height: u32,
    bits: u32,
    bytes: Vec<u8>,
}
impl Decoder {
    fn new_with_tracking(reader: impl Read, tracked: bool) -> ImageResult<Self> {
        let mut input = Vec::new();
        reader.take(256 * 1024 * 1024 + 1).read_to_end(&mut input)?;
        if input.len() > 256 * 1024 * 1024 {
            return Err(error("AVIF input exceeds 256 MiB"));
        }
        let reply = request(0, 0, 0, 80, None, &input, tracked)?;
        if reply.len() < 32 || &reply[..8] != b"PFEAVO01" {
            return Err(error("Invalid AVIF pixel header"));
        }
        let field = |offset| u32::from_le_bytes(reply[offset..offset + 4].try_into().unwrap());
        let (width, height, bits) = (field(8), field(12), field(16));
        let pixels = u64::from(width) * u64::from(height);
        if width == 0 || height == 0 || pixels > MAX_PIXELS || !matches!(bits, 8 | 16) {
            return Err(error("Invalid AVIF pixel dimensions or depth"));
        }
        let expected = pixels * 4 * u64::from(bits / 8);
        if expected != (reply.len() - 32) as u64
            || expected != u64::from_le_bytes(reply[24..32].try_into().unwrap())
        {
            return Err(error("Invalid AVIF pixel dimensions or length"));
        }
        let mut bytes = reply;
        bytes.drain(..32);
        if cfg!(target_endian = "big") && bits == 16 {
            for sample in bytes.chunks_exact_mut(2) {
                sample.swap(0, 1);
            }
        }
        Ok(Self {
            width,
            height,
            bits,
            bytes,
        })
    }
}
impl ImageDecoder for Decoder {
    fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }
    fn color_type(&self) -> image::ColorType {
        if self.bits == 16 {
            image::ColorType::Rgba16
        } else {
            image::ColorType::Rgba8
        }
    }
    fn read_image(self, buffer: &mut [u8]) -> ImageResult<()> {
        if buffer.len() != self.bytes.len() {
            return Err(error("AVIF pixel buffer length mismatch"));
        }
        buffer.copy_from_slice(&self.bytes);
        Ok(())
    }
    fn read_image_boxed(self: Box<Self>, buffer: &mut [u8]) -> ImageResult<()> {
        (*self).read_image(buffer)
    }
}
pub fn register() {
    let _ = image::hooks::register_decoding_hook(
        "avif".into(),
        Box::new(|reader| Ok(Box::new(Decoder::new_with_tracking(reader, true)?))),
    );
    for signature in [b"\0\0\0\0ftypavif", b"\0\0\0\0ftypavis"] {
        image::hooks::register_format_detection_hook(
            "avif".into(),
            signature,
            Some(b"\0\0\0\0\xff\xff\xff\xff\xff\xff\xff\xff"),
        );
    }
}
