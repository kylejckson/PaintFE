//! System HEIC codecs only. No HEVC implementation is linked or downloaded.
use image::{DynamicImage, ImageError, ImageResult, RgbaImage};
#[cfg(not(target_arch = "wasm32"))]
mod container;
#[cfg(not(target_arch = "wasm32"))]
mod worker;
#[cfg(not(target_arch = "wasm32"))]
pub use worker::run_worker;
fn error(message: impl Into<String>) -> ImageError {
    ImageError::IoError(std::io::Error::other(message.into()))
}
#[derive(Clone, Debug)]
pub struct Capabilities {
    pub decode: bool,
    pub encode: bool,
    pub detail: String,
}
pub fn capabilities() -> &'static Capabilities {
    static CAPS: std::sync::OnceLock<Capabilities> = std::sync::OnceLock::new();
    CAPS.get_or_init(|| {
        #[cfg(not(target_arch = "wasm32"))]
        let result = worker::native(0, &[], 0, 0, 90);
        #[cfg(target_arch = "wasm32")]
        let result: Result<(Vec<u8>, u32, u32, u32), String> =
            Err("HEIC requires the desktop app and an installed system codec".into());
        match result {
            Ok((bits, _, _, _)) => Capabilities {
                decode: bits.first().is_some_and(|b| b & 1 != 0),
                encode: bits.first().is_some_and(|b| b & 2 != 0),
                detail: String::new(),
            },
            Err(detail) => Capabilities {
                decode: false,
                encode: false,
                detail,
            },
        }
    })
}
pub fn setup_message() -> String {
    crate::t!(if cfg!(target_os = "windows") {
        "heic.setup_windows"
    } else if cfg!(target_os = "macos") {
        "heic.setup_macos"
    } else if cfg!(target_os = "linux") {
        "heic.setup_linux"
    } else {
        "heic.setup_unsupported"
    })
}
pub fn is_heif(bytes: &[u8]) -> bool {
    if bytes.len() < 16 || &bytes[4..8] != b"ftyp" || super::avif::is_avif(bytes) {
        return false;
    }
    let size = u32::from_be_bytes(bytes[..4].try_into().unwrap()) as usize;
    if !(16..=bytes.len()).contains(&size) {
        return false;
    }
    std::iter::once(&bytes[8..12])
        .chain(bytes[16..size].chunks_exact(4))
        .any(|b| {
            matches!(
                b,
                b"heic" | b"heix" | b"hevc" | b"hevx" | b"heim" | b"heis" | b"mif1" | b"msf1"
            )
        })
}
fn request(
    op: u8,
    data: &[u8],
    width: u32,
    height: u32,
    quality: u8,
    tracked: bool,
) -> ImageResult<Vec<u8>> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let caps = capabilities();
        if (op == 1 && !caps.encode) || (op == 2 && !caps.decode) {
            return Err(error(setup_message()));
        }
        let mut header = [0; 40];
        header[..8].copy_from_slice(b"PFEHIC02");
        header[8] = op;
        header[12..16].copy_from_slice(&width.to_le_bytes());
        header[16..20].copy_from_slice(&height.to_le_bytes());
        header[24] = quality.clamp(1, 100);
        header[32..40].copy_from_slice(&(data.len() as u64).to_le_bytes());
        crate::services::optional_packs::avif::exchange_system_heic(header, data, tracked)
            .map_err(error)
    }
    #[cfg(target_arch = "wasm32")]
    {
        let _ = (op, data, width, height, quality, tracked);
        Err(error(setup_message()))
    }
}
fn encode(image: &RgbaImage, quality: u8, tracked: bool) -> ImageResult<Vec<u8>> {
    let pixels = u64::from(image.width()) * u64::from(image.height());
    if pixels == 0 || pixels > 64_000_000 {
        return Err(error("HEIC image exceeds supported dimensions"));
    }
    let output = request(
        1,
        image.as_raw(),
        image.width(),
        image.height(),
        quality,
        tracked,
    )?;
    if !is_heif(&output) {
        return Err(error("System codec returned an invalid HEIC file"));
    }
    Ok(output)
}
pub fn encode_rgba8(image: &RgbaImage, quality: u8) -> ImageResult<Vec<u8>> {
    encode(image, quality, true)
}
pub fn decode(bytes: &[u8]) -> ImageResult<DynamicImage> {
    decode_tracked(bytes, true)
}
fn decode_tracked(bytes: &[u8], tracked: bool) -> ImageResult<DynamicImage> {
    if bytes.len() > 256 * 1024 * 1024 {
        return Err(error("HEIC input exceeds 256 MiB"));
    }
    if !is_heif(bytes) {
        return Err(error("Invalid HEIC container"));
    }
    let output = request(2, bytes, 0, 0, 90, tracked)?;
    if output.len() < 20 || &output[..8] != b"PFEHIO03" {
        return Err(error("Invalid HEIC pixel header"));
    }
    let w = u32::from_le_bytes(output[8..12].try_into().unwrap());
    let h = u32::from_le_bytes(output[12..16].try_into().unwrap());
    let bits = u32::from_le_bytes(output[16..20].try_into().unwrap());
    super::validate_open_dimensions(w, h).map_err(error)?;
    if u64::from(w) * u64::from(h) > 64_000_000 {
        return Err(error("HEIC image exceeds supported dimensions"));
    }
    if !matches!(bits, 8 | 16)
        || (output.len() - 20) as u64 != u64::from(w) * u64::from(h) * 4 * u64::from(bits / 8)
    {
        return Err(error("Invalid HEIC pixel length"));
    }
    if bits == 16 {
        let samples = output[20..]
            .chunks_exact(2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
            .collect::<Vec<_>>();
        return image::ImageBuffer::from_raw(w, h, samples)
            .map(DynamicImage::ImageRgba16)
            .ok_or_else(|| error("Invalid HEIC pixel length"));
    }
    RgbaImage::from_raw(w, h, output[20..].to_vec())
        .map(DynamicImage::ImageRgba8)
        .ok_or_else(|| error("Invalid HEIC pixel length"))
}
pub fn preview(image: &RgbaImage, quality: u8) -> ImageResult<(Vec<u8>, RgbaImage)> {
    let bytes = encode(image, quality, false)?;
    let decoded = decode_tracked(&bytes, false)?.to_rgba8();
    Ok((bytes, decoded))
}

#[cfg(test)]
mod tests {
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn container_metadata_preserves_depth_flags_and_rejects_mismatched_alpha() {
        let bytes = include_bytes!("../../tests/fixtures/heif/RGBA_8__29x100.heif");
        let info = super::container::validate(bytes).unwrap();
        assert!(info.alpha);
        assert!(!info.high_depth);
        let info = super::container::validate(include_bytes!(
            "../../tests/fixtures/heif/RGBA_10__128x128.heif"
        ))
        .unwrap();
        assert!(info.alpha && info.high_depth);
        let mut malformed = bytes.to_vec();
        let position = malformed
            .windows(4)
            .enumerate()
            .filter_map(|(i, b)| (b == b"ispe").then_some(i))
            .next_back()
            .unwrap();
        malformed[position + 12..position + 16].copy_from_slice(&99u32.to_be_bytes());
        assert!(super::container::validate(&malformed).is_err());
    }

    #[test]
    fn container_detection_keeps_avif_separate_and_bounds_malformed_headers() {
        assert!(super::is_heif(include_bytes!(
            "../../tests/fixtures/heif/RGBA_8__29x100.heif"
        )));
        assert!(!super::is_heif(include_bytes!(
            "../../tests/fixtures/avif/rgba.avif"
        )));
        let mut bytes = b"\0\0\0\x18ftypheic\0\0\0\0mif1heic".to_vec();
        for size in 0..16 {
            assert!(!super::is_heif(&bytes[..size]));
        }
        bytes[..4].copy_from_slice(&u32::MAX.to_be_bytes());
        assert!(!super::is_heif(&bytes));
    }
}
