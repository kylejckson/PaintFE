//! Real offline HEIC helper, signed-pack and CLI checks.
#![cfg(not(target_arch = "wasm32"))]
use paintfe::{
    components::dialogs::{SaveFormat, TiffCompression},
    io,
    services::optional_packs::{self as packs, Pack},
};
use std::{
    path::PathBuf,
    process::Command,
    time::{Duration, Instant},
};
fn installed() -> bool {
    static INIT: std::sync::Once = std::sync::Once::new();
    let Some(path) = std::env::var_os("PAINTFE_TEST_HEIC_PACK") else {
        eprintln!(
            "Set PAINTFE_TEST_HEIC_PACK and matching build-env.json metadata to exercise the HEIC helper."
        );
        return false;
    };
    INIT.call_once(|| packs::install(&PathBuf::from(path), Pack::Heic).unwrap());
    assert!(packs::installed(Pack::Heic).is_ok());
    true
}
fn directory() -> PathBuf {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target/heic-tests")
        .join(uuid::Uuid::new_v4().to_string());
    std::fs::create_dir_all(&path).unwrap();
    path
}
#[test]
fn real_encoder_roundtrips_odd_and_tiny_images_with_exact_alpha() {
    if !installed() {
        return;
    }
    for (w, h) in [(1, 1), (3, 2), (29, 100), (128, 128)] {
        let source = image::RgbaImage::from_fn(w, h, |x, y| {
            image::Rgba([
                (x * 17) as u8,
                (y * 23) as u8,
                ((x + y) * 11) as u8,
                ((x * 37 + y * 19) % 256) as u8,
            ])
        });
        let bytes = io::heic::encode_rgba8(&source, 90).unwrap();
        let decoded = io::decode_image_bytes(&bytes).unwrap().to_rgba8();
        assert_eq!(decoded.dimensions(), (w, h));
        for (expected, actual) in source.pixels().zip(decoded.pixels()) {
            assert_eq!(actual[3], expected[3], "alpha at {w}x{h}");
        }
    }
    let source = image::RgbaImage::from_pixel(32, 32, image::Rgba([180, 100, 50, 255]));
    let (_, preview) = io::heic::preview(&source, 90).unwrap();
    let actual = preview.get_pixel(16, 16);
    for c in 0..3 {
        assert!(actual[c].abs_diff(source.get_pixel(16, 16)[c]) <= 8);
    }
}
#[test]
fn cli_exports_and_missing_pack_does_not_truncate_existing_output() {
    if !installed() {
        return;
    }
    let dir = directory();
    let input = dir.join("input.png");
    let output = dir.join("output.heic");
    std::fs::write(&input, include_bytes!("fixtures/avif/rgba.png")).unwrap();
    let cli = env!("CARGO_BIN_EXE_PaintFE");
    assert!(
        Command::new(cli)
            .args([
                "--input",
                input.to_str().unwrap(),
                "--output",
                output.to_str().unwrap(),
                "--format",
                "heic"
            ])
            .status()
            .unwrap()
            .success()
    );
    assert_eq!(io::load_image_sync(&output).unwrap().width, 29);
    let reference = image::load_from_memory(include_bytes!("fixtures/avif/rgba.png"))
        .unwrap()
        .to_rgba8();
    let actual = io::decode_image_path(&output).unwrap().to_rgba8();
    for (expected, actual) in reference.pixels().zip(actual.pixels()) {
        assert_eq!(expected[3], actual[3], "CLI must preserve alpha exactly");
    }
    let before = std::fs::read(&output).unwrap();
    assert!(
        !Command::new(cli)
            .env("PAINTFE_TEST_STORAGE_DIR", dir.join("empty"))
            .args([
                "--input",
                input.to_str().unwrap(),
                "--output",
                output.to_str().unwrap(),
                "--format",
                "heic",
                "--overwrite"
            ])
            .status()
            .unwrap()
            .success()
    );
    assert_eq!(std::fs::read(&output).unwrap(), before);
    let source = image::RgbaImage::from_pixel(3, 2, image::Rgba([1, 2, 3, 128]));
    io::encode_and_write(
        &source,
        &output,
        SaveFormat::Heic,
        80,
        TiffCompression::None,
        true,
    )
    .unwrap();
    assert_eq!(io::decode_image_path(&output).unwrap().width(), 3);
    std::fs::remove_dir_all(dir).unwrap();
}
#[test]
fn invalid_protocol_and_cancellation_are_bounded() {
    if !installed() {
        return;
    }
    let mut header = [0; 40];
    header[..8].copy_from_slice(b"PFEHIC01");
    header[8] = 1;
    header[12..16].copy_from_slice(&u32::MAX.to_le_bytes());
    header[16..20].copy_from_slice(&u32::MAX.to_le_bytes());
    use std::io::Write;
    let mut command = Command::new(packs::installed(Pack::Heic).unwrap());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x08000000);
    }
    let mut child = command
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(&header).unwrap();
    assert!(!child.wait().unwrap().success());
    let image = image::RgbaImage::from_fn(1024, 1024, |x, y| {
        image::Rgba([x as u8, y as u8, (x + y) as u8, 255])
    });
    let job = std::thread::spawn(move || io::heic::encode_rgba8(&image, 90));
    let start = Instant::now();
    while !packs::avif::active() && start.elapsed() < Duration::from_secs(5) {
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(packs::avif::active());
    packs::avif::cancel();
    assert!(job.join().unwrap().is_err());
    assert!(start.elapsed() < Duration::from_secs(5));
}
