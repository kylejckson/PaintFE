//! Offline JPEG XL export helper. Protocol 1; raw RGBA enters stdin, JXL leaves stdout.
use jxl_sys::*;
use std::io::{Read, Write};
fn error(message: impl Into<String>) -> String {
    message.into()
}
#[derive(Clone, Copy)]
struct JxlOptions {
    lossless: bool,
    effort: u8,
}
struct Encoder(std::ptr::NonNull<JxlEncoder>);
impl Drop for Encoder {
    fn drop(&mut self) {
        // SAFETY: unique encoder created by JxlEncoderCreate, destroyed once.
        unsafe { JxlEncoderDestroy(self.0.as_ptr()) }
    }
}

fn check(status: JxlEncoderStatus) -> Result<(), String> {
    if status == JxlEncoderStatus::JXL_ENC_SUCCESS {
        Ok(())
    } else {
        Err(error(format!("JPEG XL encoder error: {status:?}")))
    }
}

fn encode(
    width: u32,
    height: u32,
    bytes: &[u8],
    bits: u32,
    quality: u8,
    options: JxlOptions,
) -> Result<Vec<u8>, String> {
    if width == 0 || height == 0 || width > 65535 || height > 65535 {
        return Err(error("Invalid JPEG XL dimensions"));
    }
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
fn run() -> Result<(), String> {
    let mut input = std::io::stdin().lock();
    let mut header = [0u8; 32];
    input.read_exact(&mut header).map_err(|e| e.to_string())?;
    if &header[..8] != b"PFEJXL01" {
        return Err(error("Unsupported JPEG XL protocol"));
    }
    let value = |offset| u32::from_le_bytes(header[offset..offset + 4].try_into().unwrap());
    let (width, height, bits) = (value(8), value(12), value(16));
    let length = u64::from_le_bytes(header[24..32].try_into().unwrap());
    let expected = u64::from(width)
        .checked_mul(u64::from(height))
        .and_then(|n| n.checked_mul(4 * u64::from(bits / 8)));
    if !matches!(bits, 8 | 16)
        || length > 1024 * 1024 * 1024
        || Some(length) != expected
        || header[22] > 1
    {
        return Err(error("Invalid JPEG XL input"));
    }
    // u16-aligned allocation for the native 16-bit encoder; wire samples are little endian.
    let mut words = vec![0u16; (length as usize).div_ceil(2)];
    let bytes =
        unsafe { std::slice::from_raw_parts_mut(words.as_mut_ptr().cast::<u8>(), length as usize) };
    input.read_exact(bytes).map_err(|e| e.to_string())?;
    if input.read(&mut [0u8; 1]).map_err(|e| e.to_string())? != 0 {
        return Err(error("Unexpected trailing input"));
    }
    #[cfg(target_endian = "big")]
    if bits == 16 {
        for word in &mut words {
            *word = u16::from_le(*word);
        }
    }
    let bytes = unsafe { std::slice::from_raw_parts(words.as_ptr().cast::<u8>(), length as usize) };
    let result = encode(
        width,
        height,
        bytes,
        bits,
        header[20],
        JxlOptions {
            effort: header[21],
            lossless: header[22] != 0,
        },
    )?;
    std::io::stdout()
        .lock()
        .write_all(&result)
        .map_err(|e| e.to_string())
}
fn main() {
    if let Err(message) = run() {
        eprintln!("{message}");
        std::process::exit(1);
    }
}
