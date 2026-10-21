//! JPEG XL still images. Built-in Rust decoding and an optional offline export helper.
use image::{ImageError, ImageResult};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JxlOptions {
    pub lossless: bool,
    pub effort: u8,
}
impl Default for JxlOptions {
    fn default() -> Self {
        Self {
            lossless: true,
            effort: 5,
        }
    }
}

/// Register both codestream/container magic and the .jxl extension before any
/// generic image operation (including dimensions and clipboard byte loading).
pub fn register() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        jxl_image_rs_integration::register_image_decoding_hook();
    });
}

fn error(message: impl Into<String>) -> ImageError {
    ImageError::IoError(std::io::Error::other(message.into()))
}

pub fn encode_rgba8(
    image: &image::RgbaImage,
    quality: u8,
    options: JxlOptions,
) -> ImageResult<Vec<u8>> {
    encode_bytes(
        image.width(),
        image.height(),
        image.as_raw(),
        8,
        quality,
        options,
    )
}
pub fn encode_rgba16(
    width: u32,
    height: u32,
    pixels: &[u16],
    quality: u8,
    options: JxlOptions,
) -> ImageResult<Vec<u8>> {
    encode_bytes(
        width,
        height,
        bytemuck::cast_slice(pixels),
        16,
        quality,
        options,
    )
}
fn encode_bytes(
    width: u32,
    height: u32,
    bytes: &[u8],
    bits: u32,
    quality: u8,
    options: JxlOptions,
) -> ImageResult<Vec<u8>> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        crate::services::optional_packs::encode_jxl(width, height, bytes, bits, quality, options)
            .map_err(error)
    }
    #[cfg(target_arch = "wasm32")]
    {
        let _ = (width, height, bytes, bits, quality, options);
        Err(error("JPEG XL export is available in the desktop app"))
    }
}
