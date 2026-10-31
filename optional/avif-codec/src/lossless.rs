use std::ffi::{CStr, c_char};
unsafe extern "C" {
    fn pf_aom_encode(
        rgba: *const u8,
        width: u32,
        height: u32,
        quality: u8,
        speed: u8,
        flags: u8,
        color: *mut *mut u8,
        color_size: *mut usize,
        alpha: *mut *mut u8,
        alpha_size: *mut usize,
        error: *mut c_char,
        error_size: usize,
    ) -> i32;
    fn pf_aom_free(pointer: *mut u8);
}
struct Buffer {
    pointer: *mut u8,
    len: usize,
}
impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe { pf_aom_free(self.pointer) }
    }
}
impl Buffer {
    fn bytes(&self) -> &[u8] {
        if self.len == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(self.pointer, self.len) }
        }
    }
}
pub fn encode(
    rgba: &[u8],
    width: u32,
    height: u32,
    quality: u8,
    speed: u8,
    flags: u8,
) -> Result<Vec<u8>, String> {
    if !matches!(u64::from(width) * u64::from(height), 1..=64_000_000)
        || rgba.len() as u64 != u64::from(width) * u64::from(height) * 4
    {
        return Err("Invalid AVIF pixel length".into());
    }
    let mut color = Buffer {
        pointer: std::ptr::null_mut(),
        len: 0,
    };
    let mut alpha = Buffer {
        pointer: std::ptr::null_mut(),
        len: 0,
    };
    let mut error = [0 as c_char; 512];
    let status = unsafe {
        pf_aom_encode(
            rgba.as_ptr(),
            width,
            height,
            quality,
            speed,
            flags,
            &mut color.pointer,
            &mut color.len,
            &mut alpha.pointer,
            &mut alpha.len,
            error.as_mut_ptr(),
            error.len(),
        )
    };
    if status != 0 {
        return Err(unsafe { CStr::from_ptr(error.as_ptr()) }
            .to_string_lossy()
            .into_owned());
    }
    if color.pointer.is_null()
        || color.len > 512 * 1024 * 1024
        || alpha.len > 512 * 1024 * 1024
        || (alpha.len > 0 && alpha.pointer.is_null())
    {
        return Err("Invalid native AVIF encoder output".into());
    }
    let mut container = avif_serialize::Aviffy::new();
    container
        .set_color_primaries(avif_serialize::constants::ColorPrimaries::Bt709)
        .set_transfer_characteristics(avif_serialize::constants::TransferCharacteristics::Srgb)
        .set_matrix_coefficients(avif_serialize::constants::MatrixCoefficients::Rgb)
        .set_full_color_range(true);
    let mut output = Vec::new();
    container
        .write(
            &mut output,
            color.bytes(),
            (!alpha.bytes().is_empty()).then(|| alpha.bytes()),
            width,
            height,
            8,
        )
        .map_err(|e| e.to_string())?;
    Ok(output)
}
