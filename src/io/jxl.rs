//! JPEG XL still images. Rust decoding and a statically linked libjxl encoder.
//! No external executable, DLL discovery or network access is used.
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

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use super::*;
    use jxl_sys::*;

    struct Encoder(std::ptr::NonNull<JxlEncoder>);
    impl Drop for Encoder {
        fn drop(&mut self) {
            // SAFETY: unique encoder created by JxlEncoderCreate, destroyed once.
            unsafe { JxlEncoderDestroy(self.0.as_ptr()) }
        }
    }

    fn check(status: JxlEncoderStatus) -> ImageResult<()> {
        if status == JxlEncoderStatus::JXL_ENC_SUCCESS {
            Ok(())
        } else {
            Err(error(format!("JPEG XL encoder error: {status:?}")))
        }
    }

    pub(super) fn encode(
        width: u32,
        height: u32,
        bytes: &[u8],
        bits: u32,
        quality: u8,
        options: JxlOptions,
    ) -> ImageResult<Vec<u8>> {
        crate::io::validate_open_dimensions(width, height).map_err(error)?;
        let expected = (width as usize)
            .checked_mul(height as usize)
            .and_then(|n| n.checked_mul(4 * (bits as usize / 8)))
            .ok_or_else(|| error("JPEG XL image size overflow"))?;
        if bytes.len() != expected || !matches!(bits, 8 | 16) {
            return Err(error("Invalid JPEG XL RGBA buffer"));
        }
        // SAFETY: all native objects remain alive for each call. libjxl copies
        // the input frame before returning. Output pointers refer to the live
        // mutable chunk and are never retained across its allocation changes.
        unsafe {
            let encoder = Encoder(
                std::ptr::NonNull::new(JxlEncoderCreate(std::ptr::null()))
                    .ok_or_else(|| error("Could not create JPEG XL encoder"))?,
            );
            let enc = encoder.0.as_ptr();
            let mut info = JxlBasicInfo::default();
            JxlEncoderInitBasicInfo(&mut info);
            info.xsize = width;
            info.ysize = height;
            info.bits_per_sample = bits;
            info.num_color_channels = 3;
            info.num_extra_channels = 1;
            info.alpha_bits = bits;
            info.alpha_premultiplied = 0;
            info.uses_original_profile = i32::from(options.lossless);
            check(JxlEncoderSetBasicInfo(enc, &info))?;
            check(JxlEncoderUseContainer(enc, 1))?;
            let mut color = JxlColorEncoding::default();
            JxlColorEncodingSetToSRGB(&mut color, 0);
            check(JxlEncoderSetColorEncoding(enc, &color))?;
            let frame = JxlEncoderFrameSettingsCreate(enc, std::ptr::null());
            if frame.is_null() {
                return Err(error("Could not configure JPEG XL frame"));
            }
            check(JxlEncoderSetFrameLossless(
                frame,
                i32::from(options.lossless),
            ))?;
            // A quality of 100 is still a lossy setting. Losslessness is explicit.
            let distance = if options.lossless {
                0.0
            } else {
                JxlEncoderDistanceFromQuality(quality.clamp(1, 100) as f32).max(0.01)
            };
            check(JxlEncoderSetFrameDistance(frame, distance))?;
            check(JxlEncoderSetExtraChannelDistance(frame, 0, 0.0))?;
            check(JxlEncoderFrameSettingsSetOption(
                frame,
                JxlEncoderFrameSettingId::JXL_ENC_FRAME_SETTING_EFFORT,
                options.effort.clamp(1, 9) as i64,
            ))?;
            check(JxlEncoderFrameSettingsSetOption(
                frame,
                JxlEncoderFrameSettingId::JXL_ENC_FRAME_SETTING_KEEP_INVISIBLE,
                1,
            ))?;
            let format = JxlPixelFormat {
                num_channels: 4,
                data_type: if bits == 16 {
                    JxlDataType::JXL_TYPE_UINT16
                } else {
                    JxlDataType::JXL_TYPE_UINT8
                },
                endianness: JxlEndianness::JXL_NATIVE_ENDIAN,
                align: 0,
            };
            check(JxlEncoderAddImageFrame(
                frame,
                &format,
                bytes.as_ptr().cast(),
                bytes.len(),
            ))?;
            JxlEncoderCloseInput(enc);
            // No parallel runner: one encoding job cannot monopolise every CPU.
            let mut output = Vec::new();
            let mut chunk = vec![0u8; 65536];
            loop {
                let mut next = chunk.as_mut_ptr();
                let mut available = chunk.len();
                let status = JxlEncoderProcessOutput(enc, &mut next, &mut available);
                let written = chunk.len() - available;
                output
                    .try_reserve(written)
                    .map_err(|_| error("JPEG XL output allocation failed"))?;
                output.extend_from_slice(&chunk[..written]);
                match status {
                    JxlEncoderStatus::JXL_ENC_SUCCESS => return Ok(output),
                    JxlEncoderStatus::JXL_ENC_NEED_MORE_OUTPUT if written > 0 => {}
                    _ => return Err(error(format!("JPEG XL output failed: {status:?}"))),
                }
            }
        }
    }
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
        native::encode(width, height, bytes, bits, quality, options)
    }
    #[cfg(target_arch = "wasm32")]
    {
        let _ = (width, height, bytes, bits, quality, options);
        Err(error("JPEG XL export is available in the desktop app"))
    }
}
