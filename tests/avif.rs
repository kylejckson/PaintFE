//! Real signed helper tests. The build tool exports the matching trust anchor.
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
    INIT.call_once(|| {
        if let Some(path) = std::env::var_os("PAINTFE_TEST_AVIF_PACK") {
            packs::install(&PathBuf::from(path), Pack::Avif).unwrap();
        }
    });
    if std::env::var_os("PAINTFE_TEST_AVIF_PACK").is_none() {
        eprintln!(
            "AVIF pack not provided; build local packs and export build-env.json to exercise this target."
        );
        return false;
    }
    assert!(packs::installed(Pack::Avif).is_ok());
    true
}
fn directory() -> PathBuf {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target/avif-tests")
        .join(uuid::Uuid::new_v4().to_string());
    std::fs::create_dir_all(&path).unwrap();
    path
}

#[test]
fn advanced_exports_preserve_exact_rgba_and_alpha_across_sizes_and_speeds() {
    if !installed() {
        return;
    }
    let folder = directory();
    for (w, h) in [(1, 1), (2, 3), (37, 29)] {
        let source = image::RgbaImage::from_fn(w, h, |x, y| {
            image::Rgba([
                (x * 17 + y * 3) as u8,
                (x * 7 + y * 29) as u8,
                (x * 31 + y * 11) as u8,
                (x * 19 + y * 23) as u8,
            ])
        });
        for speed in [1, 8, 10] {
            for exact in [false, true] {
                let options = io::export::ExportOptions {
                    format: SaveFormat::Avif,
                    avif_speed: speed,
                    avif_lossless: exact,
                    avif_lossless_alpha: !exact,
                    ..Default::default()
                };
                let path = folder.join("result.avif");
                io::export::write(
                    io::PreparedExportImage::Rgba8(source.clone()),
                    &path,
                    options,
                )
                .unwrap();
                let decoded = io::decode_image_path(&path).unwrap().to_rgba8();
                assert_eq!(decoded.dimensions(), source.dimensions());
                if exact {
                    assert_eq!(decoded, source);
                } else {
                    assert!(
                        decoded
                            .pixels()
                            .zip(source.pixels())
                            .all(|(a, b)| a[3] == b[3])
                    );
                }
            }
        }
    }
    let source = image::RgbaImage::from_pixel(32, 24, image::Rgba([32, 96, 160, 128]));
    let options = io::export::ExportOptions {
        format: SaveFormat::Avif,
        avif_lossless: true,
        ..Default::default()
    };
    let preview =
        io::export::preview(io::PreparedExportImage::Rgba8(source.clone()), options).unwrap();
    assert_eq!(preview.preview_image, source);
    assert!(
        !packs::avif::active(),
        "Preview must not leave a blocking codec operation"
    );
    std::fs::remove_dir_all(folder).unwrap();
}

#[test]
fn cancelling_a_lossless_export_preserves_the_previous_file() {
    if !installed() {
        return;
    }
    let folder = directory();
    let path = folder.join("previous.avif");
    std::fs::write(&path, b"previous contents").unwrap();
    let output = path.clone();
    let source = image::RgbaImage::from_fn(512, 512, |x, y| {
        image::Rgba([
            (x * 7 + y * 3) as u8,
            (x + y * 9) as u8,
            (x * 13 + y) as u8,
            (x + y) as u8,
        ])
    });
    let job = std::thread::spawn(move || {
        io::export::write(
            io::PreparedExportImage::Rgba8(source),
            &output,
            io::export::ExportOptions {
                format: SaveFormat::Avif,
                avif_lossless: true,
                avif_speed: 1,
                ..Default::default()
            },
        )
    });
    let start = Instant::now();
    while !packs::avif::active() && start.elapsed() < Duration::from_secs(5) {
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(packs::avif::active());
    packs::avif::cancel();
    assert!(job.join().unwrap().is_err());
    assert_eq!(std::fs::read(&path).unwrap(), b"previous contents");
    assert!(start.elapsed() < Duration::from_secs(5));
    std::fs::remove_dir_all(folder).unwrap();
}

#[test]
fn older_signed_packs_keep_basic_export_and_reject_new_flags_without_overwriting() {
    let Some(legacy) = std::env::var_os("PAINTFE_TEST_AVIF_LEGACY_PACK") else {
        eprintln!("Legacy AVIF pack fixture not supplied");
        return;
    };
    if !installed() {
        return;
    }
    struct Restore(PathBuf);
    impl Drop for Restore {
        fn drop(&mut self) {
            packs::install(&self.0, Pack::Avif).unwrap();
        }
    }
    let _restore = Restore(PathBuf::from(
        std::env::var_os("PAINTFE_TEST_AVIF_PACK").unwrap(),
    ));
    packs::install(&PathBuf::from(legacy), Pack::Avif).unwrap();
    let folder = directory();
    let path = folder.join("test.avif");
    let source = image::RgbaImage::from_pixel(32, 24, image::Rgba([20, 80, 140, 255]));
    let options = io::export::ExportOptions {
        format: SaveFormat::Avif,
        ..Default::default()
    };
    io::export::write(
        io::PreparedExportImage::Rgba8(source.clone()),
        &path,
        options,
    )
    .unwrap();
    let original = std::fs::read(&path).unwrap();
    let error = io::export::write(
        io::PreparedExportImage::Rgba8(source),
        &path,
        io::export::ExportOptions {
            avif_lossless: true,
            ..options
        },
    )
    .unwrap_err();
    assert!(error.contains("update the AVIF pack"), "{error}");
    assert_eq!(std::fs::read(&path).unwrap(), original);
    std::fs::remove_dir_all(folder).unwrap();
}

#[test]
fn invalid_helper_protocol_requests_fail_without_allocating_claimed_input() {
    if !installed() {
        return;
    }
    for (op, length, width, height) in [
        (2, 0, 0, 0),
        (0, 512 * 1024 * 1024 + 1, 0, 0),
        (1, 0, u32::MAX, u32::MAX),
    ] {
        let mut header = [0; 40];
        header[..8].copy_from_slice(b"PFEAVF01");
        header[8] = op;
        header[12..16].copy_from_slice(&width.to_le_bytes());
        header[16..20].copy_from_slice(&height.to_le_bytes());
        header[32..40].copy_from_slice(&(length as u64).to_le_bytes());
        assert!(packs::avif::exchange(header, &[], false).is_err());
    }
}

#[test]
fn opens_independent_reference_and_roundtrips_through_real_signed_pack() {
    if !installed() {
        return;
    }
    let reference = image::load_from_memory(include_bytes!("fixtures/avif/rgba.png"))
        .unwrap()
        .to_rgba8();
    let actual = io::decode_image_bytes(include_bytes!("fixtures/avif/rgba.avif"))
        .unwrap()
        .to_rgba8();
    assert_eq!(actual.dimensions(), reference.dimensions());
    for (actual, expected) in actual.pixels().zip(reference.pixels()) {
        assert_eq!(actual[3], expected[3]);
        for c in 0..3 {
            assert!(actual[c].abs_diff(expected[c]) <= 1);
        }
    }
    let encoded = io::avif::encode_rgba8(&actual, 90).unwrap();
    assert!(io::avif::is_avif(&encoded));
    let decoded = io::decode_image_bytes(&encoded).unwrap();
    assert_eq!((decoded.width(), decoded.height()), (29, 100));
    assert!(decoded.to_rgba8().pixels().any(|p| p[3] < 255));
    assert!(io::decode_image_bytes(b"\0\0\0\x18ftypavifnot a file").is_err());
}

#[test]
fn cancelled_helper_returns_promptly() {
    if !installed() {
        return;
    }
    let image = image::RgbaImage::from_fn(512, 512, |x, y| {
        image::Rgba([
            (x * 7 + y * 3) as u8,
            (x + y * 9) as u8,
            (x * 13 + y) as u8,
            255,
        ])
    });
    let job = std::thread::spawn(move || io::avif::encode_rgba8(&image, 90));
    let start = Instant::now();
    while !packs::avif::active() && start.elapsed() < Duration::from_secs(5) {
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(packs::avif::active());
    packs::avif::cancel();
    assert!(job.join().unwrap().is_err());
    assert!(start.elapsed() < Duration::from_secs(5));
    assert!(!packs::avif::active());
}

#[test]
fn cli_exports_avif_and_missing_pack_preserves_existing_file() {
    if !installed() {
        return;
    }
    let directory = directory();
    let input = directory.join("input.png");
    std::fs::write(&input, include_bytes!("fixtures/avif/rgba.png")).unwrap();
    let output = directory.join("output.avif");
    let status = Command::new(env!("CARGO_BIN_EXE_PaintFE"))
        .args([
            "--input",
            input.to_str().unwrap(),
            "--output",
            output.to_str().unwrap(),
            "--format",
            "avif",
        ])
        .status()
        .unwrap();
    assert!(status.success());
    assert_eq!(io::load_image_sync(&output).unwrap().width, 29);
    let original = std::fs::read(&output).unwrap();
    let empty = directory.join("empty-storage");
    let status = Command::new(env!("CARGO_BIN_EXE_PaintFE"))
        .env("PAINTFE_TEST_STORAGE_DIR", &empty)
        .args([
            "--overwrite",
            "--input",
            input.to_str().unwrap(),
            "--output",
            output.to_str().unwrap(),
            "--format",
            "avif",
        ])
        .status()
        .unwrap();
    assert!(!status.success());
    assert_eq!(std::fs::read(&output).unwrap(), original);
    // Direct file export also uses the checked helper before opening its destination.
    let source = image::RgbaImage::from_pixel(3, 2, image::Rgba([1, 2, 3, 128]));
    io::encode_and_write(
        &source,
        &output,
        SaveFormat::Avif,
        80,
        TiffCompression::None,
        true,
    )
    .unwrap();
    assert!(io::avif::is_avif(&std::fs::read(&output).unwrap()));
    std::fs::remove_dir_all(directory).unwrap();
}
