use std::io::{Read, Write};
#[cfg(any(target_os = "windows", target_os = "macos", target_os = "linux"))]
unsafe extern "C" {
    fn pf_system_heic(
        op: u8,
        data: *const u8,
        length: usize,
        width: u32,
        height: u32,
        quality: u8,
        output: *mut *mut u8,
        size: *mut usize,
        out_width: *mut u32,
        out_height: *mut u32,
        out_bits: *mut u32,
        error: *mut std::ffi::c_char,
        error_size: usize,
    ) -> i32;
    fn pf_system_heic_free(bytes: *mut u8);
    fn pf_system_heic_limits(error: *mut std::ffi::c_char, capacity: usize) -> i32;
}
pub(super) fn native(
    op: u8,
    data: &[u8],
    width: u32,
    height: u32,
    quality: u8,
) -> Result<(Vec<u8>, u32, u32, u32), String> {
    #[cfg(debug_assertions)]
    if std::env::var_os("PAINTFE_TEST_NO_SYSTEM_HEIC").is_some() {
        return Err("System HEIC codecs unavailable (isolated test)".into());
    }
    #[cfg(any(target_os = "windows", target_os = "macos", target_os = "linux"))]
    {
        let op = if op == 2 {
            let info = super::container::validate(data)?;
            op | if info.alpha { 0x80 } else { 0 } | if info.high_depth { 0x40 } else { 0 }
        } else {
            op
        };
        let (mut output, mut length, mut w, mut h) = (std::ptr::null_mut(), 0usize, 0u32, 0u32);
        let mut message = [0 as std::ffi::c_char; 1024];
        let mut bits = 8;
        // The bridge catches C++ exceptions and owns the buffer until freed here.
        let result = unsafe {
            pf_system_heic(
                op,
                data.as_ptr(),
                data.len(),
                width,
                height,
                quality,
                &mut output,
                &mut length,
                &mut w,
                &mut h,
                &mut bits,
                message.as_mut_ptr(),
                message.len(),
            )
        };
        if result != 0 {
            return Err(unsafe { std::ffi::CStr::from_ptr(message.as_ptr()) }
                .to_string_lossy()
                .into_owned());
        }
        if output.is_null() || length > 512 * 1024 * 1024 {
            if !output.is_null() {
                unsafe { pf_system_heic_free(output) }
            }
            return Err("Invalid system codec buffer".into());
        }
        let bytes = unsafe { std::slice::from_raw_parts(output, length) }.to_vec();
        unsafe { pf_system_heic_free(output) };
        Ok((bytes, w, h, bits))
    }
    #[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
    {
        let _ = (op, data, width, height, quality);
        Err("No system HEIC backend for this platform".into())
    }
}
fn process() -> Result<(), String> {
    #[cfg(any(target_os = "windows", target_os = "macos", target_os = "linux"))]
    {
        let mut message = [0 as std::ffi::c_char; 1024];
        if unsafe { pf_system_heic_limits(message.as_mut_ptr(), message.len()) } != 0 {
            return Err(unsafe { std::ffi::CStr::from_ptr(message.as_ptr()) }
                .to_string_lossy()
                .into_owned());
        }
    }
    let mut stdin = std::io::stdin().lock();
    let mut header = [0; 40];
    stdin.read_exact(&mut header).map_err(|e| e.to_string())?;
    if &header[..8] != b"PFEHIC02" || !matches!(header[8], 0..=2) {
        return Err("Invalid system HEIC request".into());
    }
    let size = u64::from_le_bytes(header[32..40].try_into().unwrap());
    let w = u32::from_le_bytes(header[12..16].try_into().unwrap());
    let h = u32::from_le_bytes(header[16..20].try_into().unwrap());
    let limit = if header[8] == 0 {
        0
    } else if header[8] == 1 {
        if w == 0 || h == 0 || u64::from(w) * u64::from(h) > 64_000_000 {
            return Err("Invalid HEIC dimensions".into());
        }
        u64::from(w) * u64::from(h) * 4
    } else {
        256 * 1024 * 1024
    };
    if size > limit || (header[8] == 1 && size != limit) {
        return Err("Invalid HEIC input length".into());
    }
    let mut input = vec![0; size as usize];
    stdin.read_exact(&mut input).map_err(|e| e.to_string())?;
    let (output, width, height, bits) = native(header[8], &input, w, h, header[24].clamp(1, 100))?;
    let mut stdout = std::io::stdout().lock();
    if header[8] == 2 {
        stdout
            .write_all(b"PFEHIO03")
            .and_then(|_| stdout.write_all(&width.to_le_bytes()))
            .and_then(|_| stdout.write_all(&height.to_le_bytes()))
            .and_then(|_| stdout.write_all(&bits.to_le_bytes()))
            .map_err(|e| e.to_string())?;
    }
    stdout
        .write_all(&output)
        .and_then(|_| stdout.flush())
        .map_err(|e| e.to_string())
}
pub fn run_worker() -> i32 {
    match process() {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("{e}");
            1
        }
    }
}
