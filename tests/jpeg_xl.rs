//! Cross-codec checks: libjxl encodes, the independent Rust decoder reads it.
#![cfg(not(target_arch = "wasm32"))]
use image::{ImageDecoder, Rgba, RgbaImage};
use paintfe::components::dialogs::{SaveFormat, TiffCompression};
use paintfe::io::{
    self, PreparedExportImage,
    jxl::{self, JxlOptions},
};

fn encoder_available() -> bool {
    static SETUP: std::sync::Once = std::sync::Once::new();
    SETUP.call_once(|| {
        if let Some(path) = std::env::var_os("PAINTFE_TEST_JXL_PACK") {
            paintfe::services::optional_packs::install(
                std::path::Path::new(&path),
                paintfe::services::optional_packs::Pack::Jxl,
            )
            .unwrap();
        }
    });
    let available =
        paintfe::services::optional_packs::installed(paintfe::services::optional_packs::Pack::Jxl)
            .is_ok();
    if !available {
        eprintln!(
            "Optional encoder unavailable; run tools/build_optional_packs.py and set PAINTFE_TEST_JXL_PACK to exercise exports."
        );
    }
    available
}
struct Fixture(std::path::PathBuf);
impl Fixture {
    fn new() -> Self {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target/jpeg-xl-tests")
            .join(uuid::Uuid::new_v4().to_string());
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn image() -> RgbaImage {
    RgbaImage::from_fn(23, 17, |x, y| {
        Rgba([
            (x * 11 + y) as u8,
            (y * 15 + x) as u8,
            (x * 7 + y * 9) as u8,
            [0, 1, 128, 254, 255][(x + y) as usize % 5],
        ])
    })
}

#[test]
fn lossless_rgba8_preserves_all_pixels_including_hidden_rgb_and_alpha() {
    if !encoder_available() {
        return;
    }
    jxl::register();
    let source = image();
    for effort in [1, 5, 9] {
        let bytes = jxl::encode_rgba8(
            &source,
            90,
            JxlOptions {
                lossless: true,
                effort,
            },
        )
        .unwrap();
        assert_eq!(&bytes[..12], b"\0\0\0\x0cJXL \r\n\x87\n");
        let decoded = io::decode_image_bytes(&bytes).unwrap().to_rgba8();
        assert_eq!(source, decoded, "effort {effort}");
    }
}

#[test]
fn lossy_quality_100_is_explicit_and_alpha_remains_lossless() {
    if !encoder_available() {
        return;
    }
    jxl::register();
    let source = image();
    for quality in [30, 90, 100] {
        let bytes = jxl::encode_rgba8(
            &source,
            quality,
            JxlOptions {
                lossless: false,
                effort: 3,
            },
        )
        .unwrap();
        let decoded = io::decode_image_bytes(&bytes).unwrap().to_rgba8();
        assert_eq!(decoded.dimensions(), source.dimensions());
        for (a, b) in source.pixels().zip(decoded.pixels()) {
            assert_eq!(a[3], b[3]);
        }
        let opaque_errors: Vec<_> = source
            .pixels()
            .zip(decoded.pixels())
            .filter(|(a, _)| a[3] == 255)
            .flat_map(|(a, b)| (0..3).map(move |c| a[c].abs_diff(b[c]) as f64))
            .collect();
        assert!(opaque_errors.iter().sum::<f64>() / (opaque_errors.len() as f64) < 35.0);
    }
}

#[test]
fn lossless_rgba16_roundtrip_preserves_native_depth_and_samples() {
    if !encoder_available() {
        return;
    }
    jxl::register();
    let pixels: Vec<u16> = (0..11 * 7 * 4).map(|i| (i * 137 + 11) as u16).collect();
    let bytes = jxl::encode_rgba16(11, 7, &pixels, 90, JxlOptions::default()).unwrap();
    let decoded = io::decode_image_bytes(&bytes).unwrap();
    assert_eq!(decoded.color(), image::ColorType::Rgba16);
    assert_eq!(decoded.to_rgba16().as_raw(), &pixels);
    let fixture = Fixture::new();
    let path = fixture.0.join("deep.jxl");
    io::encode_prepared_and_write(
        PreparedExportImage::Rgba16 {
            width: 11,
            height: 7,
            pixels: pixels.clone(),
        },
        &path,
        SaveFormat::Jxl(JxlOptions::default()),
        90,
        TiffCompression::None,
        true,
    )
    .unwrap();
    let loaded = io::load_image_sync(&path).unwrap();
    assert_eq!(
        loaded.layers[0].pixel_format,
        paintfe::canvas::PixelFormat::RgbaU16
    );
    match loaded.layers[0].deep_pixels.as_ref().unwrap() {
        paintfe::experimental::DeepRgbaBuffer::U16(values) => assert_eq!(values, &pixels),
        _ => panic!("16-bit payload discarded"),
    }
    let second = fixture.0.join("saved-again.jxl");
    io::encode_canvas_state_and_write(
        &loaded,
        &second,
        SaveFormat::Jxl(JxlOptions::default()),
        90,
        TiffCompression::None,
        true,
    )
    .unwrap();
    let second = image::open(second).unwrap().to_rgba16();
    assert_eq!(
        second.as_raw(),
        &pixels,
        "document export lost 16-bit precision"
    );
}

#[test]
fn generic_open_detects_jxl_without_a_filename_extension() {
    if !encoder_available() {
        return;
    }
    jxl::register();
    let fixture = Fixture::new();
    let source = image();
    let bytes = jxl::encode_rgba8(&source, 90, JxlOptions::default()).unwrap();
    let path = fixture.0.join("image.data");
    std::fs::write(&path, &bytes).unwrap();
    let reader = image::ImageReader::open(&path)
        .unwrap()
        .with_guessed_format()
        .unwrap();
    assert_eq!(reader.decode().unwrap().to_rgba8(), source);
    assert!(image::image_dimensions(fixture.0.join("missing.jxl")).is_err());
}

#[test]
fn prepared_file_and_quick_save_use_the_same_codec_options() {
    if !encoder_available() {
        return;
    }
    let fixture = Fixture::new();
    let source = image();
    let path = fixture.0.join("test.jxl");
    let format = SaveFormat::Jxl(JxlOptions {
        lossless: true,
        effort: 2,
    });
    let mut handler = io::FileHandler::new();
    handler
        .save_image(&source, &path, format, 80, TiffCompression::None, true)
        .unwrap();
    assert_eq!(handler.last_export_options.format, format);
    handler.quick_save(&source).unwrap();
    assert_eq!(image::open(&path).unwrap().to_rgba8(), source);
    assert_eq!(image::image_dimensions(&path).unwrap(), source.dimensions());
    assert!(format.supports_quality());
    assert!(!format.supports_animation());
}

#[test]
fn malformed_truncated_and_over_limit_inputs_are_rejected() {
    if !encoder_available() {
        return;
    }
    jxl::register();
    let bytes = jxl::encode_rgba8(&image(), 90, JxlOptions::default()).unwrap();
    for end in [0, 1, 2, 12, bytes.len() / 2] {
        assert!(io::decode_image_bytes(&bytes[..end]).is_err());
    }
    assert!(jxl::encode_rgba16(20, 20, &[0; 16], 90, JxlOptions::default()).is_err());
    assert!(jxl::encode_rgba8(&RgbaImage::new(0, 0), 90, JxlOptions::default()).is_err());
    let mut decoder =
        jxl_image_rs_integration::JxlDecoder::new(std::io::Cursor::new(&bytes)).unwrap();
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(10);
    assert!(decoder.set_limits(limits).is_err());
    let mut decoder =
        jxl_image_rs_integration::JxlDecoder::new(std::io::Cursor::new(&bytes)).unwrap();
    let mut limits = image::Limits::default();
    limits.max_alloc = Some(32);
    assert!(decoder.set_limits(limits).is_err());
}

#[test]
fn cli_exports_lossless_and_lossy_jxl_and_imports_it_again() {
    if !encoder_available() {
        return;
    }
    let fixture = Fixture::new();
    let source = fixture.0.join("in.png");
    image().save(&source).unwrap();
    for lossy in [false, true] {
        let output = fixture
            .0
            .join(if lossy { "lossy.jxl" } else { "exact.jxl" });
        let mut cmd = std::process::Command::new(env!("CARGO_BIN_EXE_PaintFE"));
        cmd.args(["--input"])
            .arg(&source)
            .arg("--output")
            .arg(&output)
            .args(["--jxl-effort", "2"]);
        if lossy {
            cmd.arg("--jxl-lossy");
        }
        let result = cmd
            .env(
                "PAINTFE_TEST_STORAGE_DIR",
                std::env::var_os("PAINTFE_TEST_STORAGE_DIR")
                    .expect("Set PAINTFE_TEST_STORAGE_DIR for codec tests"),
            )
            .env("RAYON_NUM_THREADS", "2")
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let loaded = io::load_image_sync(&output).unwrap();
        assert_eq!((loaded.width, loaded.height), image().dimensions());
        let png = fixture.0.join(if lossy {
            "decoded-lossy.png"
        } else {
            "decoded-exact.png"
        });
        let result = std::process::Command::new(env!("CARGO_BIN_EXE_PaintFE"))
            .arg("--input")
            .arg(&output)
            .arg("--output")
            .arg(&png)
            .env(
                "PAINTFE_TEST_STORAGE_DIR",
                std::env::var_os("PAINTFE_TEST_STORAGE_DIR")
                    .expect("Set PAINTFE_TEST_STORAGE_DIR for codec tests"),
            )
            .env("RAYON_NUM_THREADS", "2")
            .output()
            .unwrap();
        assert!(result.status.success());
        if !lossy {
            assert_eq!(image::open(png).unwrap().to_rgba8(), image());
        }
    }
}

#[test]
fn built_in_import_works_without_encoder_or_network() {
    jxl::register();
    let decoded = io::decode_image_bytes(include_bytes!("fixtures/jpeg-xl/rgba8.jxl"))
        .unwrap()
        .to_rgba8();
    assert_eq!(decoded, image());
    let decoded = io::decode_image_bytes(include_bytes!("fixtures/jpeg-xl/rgba16.jxl")).unwrap();
    assert_eq!(decoded.color(), image::ColorType::Rgba16);
    let expected: Vec<u16> = (0..11 * 7 * 4).map(|i| (i * 137 + 11) as u16).collect();
    assert_eq!(decoded.to_rgba16().as_raw(), &expected);
}
#[test]
fn failed_jxl_export_preserves_existing_destination() {
    let fixture = Fixture::new();
    let path = fixture.0.join("existing.jxl");
    std::fs::write(&path, b"previous export").unwrap();
    // Invalid dimensions fail even with an encoder installed.
    assert!(
        io::encode_and_write(
            &RgbaImage::new(0, 0),
            &path,
            SaveFormat::Jxl(Default::default()),
            90,
            TiffCompression::None,
            true
        )
        .is_err()
    );
    assert_eq!(std::fs::read(path).unwrap(), b"previous export");
}

#[test]
fn cli_without_encoder_reports_failure_and_preserves_previous_export() {
    let fixture = Fixture::new();
    let source = fixture.0.join("input.png");
    image().save(&source).unwrap();
    let output = fixture.0.join("existing.jxl");
    std::fs::write(&output, b"previous export").unwrap();
    let result = std::process::Command::new(env!("CARGO_BIN_EXE_PaintFE"))
        .arg("--input")
        .arg(source)
        .arg("--output")
        .arg(&output)
        .arg("--overwrite")
        .env(
            "PAINTFE_TEST_STORAGE_DIR",
            fixture.0.join("no-pack-profile"),
        )
        .env("RAYON_NUM_THREADS", "2")
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert_eq!(std::fs::read(output).unwrap(), b"previous export");
}
