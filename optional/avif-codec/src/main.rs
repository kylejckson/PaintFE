//! Offline AVIF still-image helper. stdin/stdout only; never accesses the network.
use image::{ImageDecoder, ImageEncoder};
use std::io::{Read, Write};
const MAX_PIXELS: u64 = 64_000_000;
const MAX_INPUT: u64 = 512 * 1024 * 1024;

fn memory_limit() -> Result<(), String> {
    #[cfg(target_os = "linux")]
    unsafe {
        let mut inherited: libc::rlimit = std::mem::zeroed();
        if libc::getrlimit(libc::RLIMIT_AS, &mut inherited) != 0 {
            return Err("Could not read codec memory limit".into());
        }
        let maximum = inherited.rlim_max.min(3 * 1024 * 1024 * 1024);
        let limit = libc::rlimit {
            rlim_cur: maximum,
            rlim_max: maximum,
        };
        if libc::setrlimit(libc::RLIMIT_AS, &limit) != 0 {
            return Err("Could not set codec memory limit".into());
        }
    }
    #[cfg(target_os = "macos")]
    {
        // libc already supplies this stable Mach API; keep the helper dependency set small.
        #[allow(deprecated)]
        fn resident_bytes() -> Result<u64, String> {
            unsafe {
                let mut info: libc::mach_task_basic_info = std::mem::zeroed();
                let mut count = libc::MACH_TASK_BASIC_INFO_COUNT;
                if libc::task_info(
                    libc::mach_task_self(),
                    libc::MACH_TASK_BASIC_INFO,
                    &mut info as *mut _ as *mut libc::integer_t,
                    &mut count,
                ) != 0
                {
                    return Err("Could not read codec memory usage".into());
                }
                Ok(info.resident_size)
            }
        }
        resident_bytes()?;
        // macOS maps a large shared cache before main: a small RLIMIT_AS can
        // reject an otherwise empty process. Watch resident memory instead.
        std::thread::Builder::new()
            .name("codec-memory-limit".into())
            .spawn(|| {
                loop {
                    if resident_bytes().map_or(true, |bytes| bytes > 2 * 1024 * 1024 * 1024) {
                        eprintln!("AVIF codec exceeded memory limit");
                        std::process::exit(1);
                    }
                    std::thread::sleep(std::time::Duration::from_millis(20));
                }
            })
            .map_err(|e| e.to_string())?;
    }
    #[cfg(windows)]
    unsafe {
        use windows_sys::Win32::System::{JobObjects::*, Threading::GetCurrentProcess};
        let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
        if job.is_null() {
            return Err("Could not create codec memory limit".into());
        }
        let mut limits: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
        limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_PROCESS_MEMORY;
        limits.ProcessMemoryLimit = 2 * 1024 * 1024 * 1024;
        if SetInformationJobObject(
            job,
            JobObjectExtendedLimitInformation,
            &limits as *const _ as _,
            std::mem::size_of_val(&limits) as u32,
        ) == 0
            || AssignProcessToJobObject(job, GetCurrentProcess()) == 0
        {
            windows_sys::Win32::Foundation::CloseHandle(job);
            return Err("Could not apply codec memory limit".into());
        }
        // Kept until process exit so the limit remains active.
    }
    Ok(())
}

fn dimensions(width: u32, height: u32) -> Result<(), String> {
    if width == 0 || height == 0 || u64::from(width) * u64::from(height) > MAX_PIXELS {
        Err("AVIF image exceeds dimension limits".into())
    } else {
        Ok(())
    }
}

fn run() -> Result<(), String> {
    memory_limit()?;
    let mut input = std::io::stdin().lock();
    let mut header = [0u8; 40];
    input.read_exact(&mut header).map_err(|e| e.to_string())?;
    if &header[..8] != b"PFEAVF01" || header[9..12] != [0; 3] || header[26..32] != [0; 6] {
        return Err("Invalid AVIF protocol header".into());
    }
    let field = |offset| u32::from_le_bytes(header[offset..offset + 4].try_into().unwrap());
    let (width, height, bits) = (field(12), field(16), field(20));
    let length = u64::from_le_bytes(header[32..40].try_into().unwrap());
    if length > MAX_INPUT || header[8] > 1 {
        return Err("Invalid AVIF request length or operation".into());
    }
    if header[8] == 1 {
        dimensions(width, height)?;
        if bits != 8
            || length != u64::from(width) * u64::from(height) * 4
            || !(1..=100).contains(&header[24])
            || !(1..=10).contains(&header[25])
        {
            return Err("Invalid AVIF encoder settings".into());
        }
    } else if length > 256 * 1024 * 1024 {
        return Err("Compressed AVIF input exceeds limit".into());
    }
    let mut bytes = vec![0; length as usize];
    input.read_exact(&mut bytes).map_err(|e| e.to_string())?;
    let mut extra = [0];
    if input.read(&mut extra).map_err(|e| e.to_string())? != 0 {
        return Err("Trailing AVIF request bytes".into());
    }
    let mut output = std::io::stdout().lock();
    if header[8] == 1 {
        image::codecs::avif::AvifEncoder::new_with_speed_quality(
            &mut output,
            header[25],
            header[24],
        )
        .with_num_threads(Some(4))
        .write_image(&bytes, width, height, image::ExtendedColorType::Rgba8)
        .map_err(|e| e.to_string())?;
    } else {
        let decoder = image::codecs::avif::AvifDecoder::new(std::io::Cursor::new(bytes))
            .map_err(|e| e.to_string())?;
        let (width, height) = decoder.dimensions();
        dimensions(width, height)?;
        let deep = matches!(
            decoder.color_type(),
            image::ColorType::Rgb16 | image::ColorType::Rgba16
        );
        let image = image::DynamicImage::from_decoder(decoder).map_err(|e| e.to_string())?;
        let (bits, raw) = if deep {
            let image = image.to_rgba16();
            let mut raw = Vec::with_capacity(image.as_raw().len() * 2);
            for sample in image.into_raw() {
                raw.extend_from_slice(&sample.to_le_bytes());
            }
            (16u32, raw)
        } else {
            (8u32, image.to_rgba8().into_raw())
        };
        let mut header = [0; 32];
        header[..8].copy_from_slice(b"PFEAVO01");
        for (offset, value) in [(8, width), (12, height), (16, bits)] {
            header[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        }
        header[24..32].copy_from_slice(&(raw.len() as u64).to_le_bytes());
        output
            .write_all(&header)
            .and_then(|_| output.write_all(&raw))
            .map_err(|e| e.to_string())?;
    }
    output.flush().map_err(|e| e.to_string())
}

fn main() {
    match std::panic::catch_unwind(run) {
        Ok(Ok(())) => {}
        Ok(Err(error)) => {
            eprintln!("{error}");
            std::process::exit(1);
        }
        Err(_) => {
            eprintln!("AVIF codec failed");
            std::process::exit(2);
        }
    }
}
