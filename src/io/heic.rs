//! Optional offline HEIC export. HEIC decoding remains bundled.
use image::{ImageError, ImageResult};
fn error(message: impl Into<String>) -> ImageError {
    ImageError::IoError(std::io::Error::other(message.into()))
}
fn encode(image: &image::RgbaImage, quality: u8, tracked: bool) -> ImageResult<Vec<u8>> {
    if image.width() == 0
        || image.height() == 0
        || u64::from(image.width()) * u64::from(image.height()) > 64_000_000
    {
        return Err(error("HEIC image exceeds supported dimensions"));
    }
    let mut header = [0; 40];
    header[..8].copy_from_slice(b"PFEHIC01");
    header[8] = 1;
    header[12..16].copy_from_slice(&image.width().to_le_bytes());
    header[16..20].copy_from_slice(&image.height().to_le_bytes());
    header[20..24].copy_from_slice(&8u32.to_le_bytes());
    header[24] = quality.clamp(1, 100);
    header[25] = 8;
    header[32..40].copy_from_slice(&(image.as_raw().len() as u64).to_le_bytes());
    #[cfg(not(target_arch = "wasm32"))]
    {
        let output = crate::services::optional_packs::avif::exchange_for(
            crate::services::optional_packs::Pack::Heic,
            header,
            image.as_raw(),
            tracked,
        )
        .map_err(error)?;
        if output.len() < 16 || &output[4..8] != b"ftyp" {
            return Err(error("Invalid HEIC helper output"));
        }
        Ok(output)
    }
    #[cfg(target_arch = "wasm32")]
    {
        let _ = (header, tracked);
        Err(error("HEIC export packs are available in the desktop app"))
    }
}
pub fn encode_rgba8(image: &image::RgbaImage, quality: u8) -> ImageResult<Vec<u8>> {
    encode(image, quality, true)
}
pub fn preview(image: &image::RgbaImage, quality: u8) -> ImageResult<(Vec<u8>, image::RgbaImage)> {
    let bytes = encode(image, quality, false)?;
    let decoded = crate::io::decode_image_bytes(&bytes)?.to_rgba8();
    Ok((bytes, decoded))
}
