//! Native system-codec checks. No downloaded HEVC implementation is used.
#![cfg(not(target_arch = "wasm32"))]
use image::{Rgba, RgbaImage};
use paintfe::{
    io,
    services::optional_packs::{self as packs, Pack},
};
use std::{
    io::Write,
    path::PathBuf,
    process::{Command, Stdio},
};
fn directory() -> PathBuf {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target/system-heic-tests")
        .join(uuid::Uuid::new_v4().to_string());
    std::fs::create_dir_all(&path).unwrap();
    path
}
#[test]
fn capabilities_are_reported_without_a_pack_or_network() {
    let caps = io::heic::capabilities();
    println!(
        "System HEIC: decode={}, encode={}, {}",
        caps.decode, caps.encode, caps.detail
    );
    if let Ok(expected) = std::env::var("PAINTFE_EXPECT_SYSTEM_HEIC") {
        assert!(caps.decode, "Installed system HEIC decoder must be usable");
        if expected == "both" {
            assert!(caps.encode, "Installed system HEIC encoder must be usable");
        }
    }
    assert!(!Pack::ALL.contains(&Pack::Heic));
    assert!(packs::installed(Pack::Heic).is_err());
    assert!(packs::download(Pack::Heic).is_err()); // rejected before transport
}
#[test]
fn native_encoder_roundtrips_opaque_pixels_and_reports_unsupported_alpha() {
    if !io::heic::capabilities().encode || !io::heic::capabilities().decode {
        return;
    }
    for (w, h) in [(32, 32), (29, 100), (128, 128)] {
        let source = RgbaImage::from_pixel(w, h, Rgba([180, 100, 50, 255]));
        let bytes = io::heic::encode_rgba8(&source, 90).unwrap();
        assert!(io::heic::is_heif(&bytes));
        let decoded = io::decode_image_bytes(&bytes).unwrap().to_rgba8();
        assert_eq!(decoded.dimensions(), (w, h));
        for (actual, expected) in decoded.pixels().zip(source.pixels()) {
            assert_eq!(actual[3], 255);
            for c in 0..3 {
                assert!(actual[c].abs_diff(expected[c]) <= 12, "{actual:?}");
            }
        }
    }
    let transparent = RgbaImage::from_pixel(32, 32, Rgba([180, 100, 50, 128]));
    match io::heic::encode_rgba8(&transparent, 90) {
        Ok(bytes) => assert!(
            io::heic::decode(&bytes)
                .unwrap()
                .to_rgba8()
                .pixels()
                .all(|p| p[3].abs_diff(128) <= 1),
            "System encoder silently lost transparency"
        ),
        Err(error) => assert!(
            error.to_string().to_lowercase().contains("transparency"),
            "{error}"
        ),
    }
}
#[test]
fn invalid_worker_requests_are_bounded_and_rejected() {
    for (op, w, h, size) in [
        (1, 0, 3, 0),
        (1, 25000, 25000, 0),
        (2, 0, 0, 300 * 1024 * 1024),
        (9, 0, 0, 0),
    ] {
        let mut child = Command::new(env!("CARGO_BIN_EXE_PaintFE"))
            .arg("--paintfe-system-heic")
            .stdin(Stdio::piped())
            .stderr(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .unwrap();
        let mut header = [0; 40];
        header[..8].copy_from_slice(b"PFEHIC02");
        header[8] = op;
        header[12..16].copy_from_slice(&(w as u32).to_le_bytes());
        header[16..20].copy_from_slice(&(h as u32).to_le_bytes());
        header[32..40].copy_from_slice(&(size as u64).to_le_bytes());
        child.stdin.take().unwrap().write_all(&header).unwrap();
        let output = child.wait_with_output().unwrap();
        assert!(!output.status.success());
        assert!(output.stdout.is_empty());
    }
}
#[test]
fn cli_missing_system_codec_preserves_existing_destination() {
    let dir = directory();
    let input = dir.join("source.png");
    let output = dir.join("keep.heic");
    RgbaImage::from_pixel(32, 32, Rgba([120, 80, 40, 255]))
        .save(&input)
        .unwrap();
    std::fs::write(&output, b"original file").unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_PaintFE"))
        .env("PAINTFE_TEST_NO_SYSTEM_HEIC", "1")
        .args([
            "--input",
            input.to_str().unwrap(),
            "--output",
            output.to_str().unwrap(),
            "--format",
            "heic",
            "--overwrite",
        ])
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert_eq!(std::fs::read(output).unwrap(), b"original file");
}
