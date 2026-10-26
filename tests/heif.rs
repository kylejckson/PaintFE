//! Independent libheif-generated reference pixels; test fixtures are BSD-3-Clause.
use paintfe::io;

/// Keep the public sample outside the repository. This uses a real native codec
/// and intentionally runs separately from small CI tests because it is 51 MP.
#[test]
#[ignore = "requires PAINTFE_TEST_ISSUE207_HEIF pointing to the 8736x5856 sample1.heif"]
fn issue_207_large_heif_decodes_in_the_bounded_worker() {
    let path =
        std::env::var_os("PAINTFE_TEST_ISSUE207_HEIF").expect("set PAINTFE_TEST_ISSUE207_HEIF");
    let decoded =
        io::decode_image_path(std::path::Path::new(&path)).expect("large HEIF worker decode");
    assert_eq!((decoded.width(), decoded.height()), (8736, 5856));
    let pixels = decoded.as_rgba8().expect("8-bit source must remain 8-bit");
    let samples = [(0, 0), (4368, 2928), (8735, 5855)];
    for (x, y) in samples {
        assert_eq!(pixels.get_pixel(x, y)[3], 255);
    }
    assert_ne!(pixels.get_pixel(0, 0), pixels.get_pixel(4368, 2928));
}

const CASES: &[(&str, &[u8], &[u8], u8)] = &[
    (
        "gray grid alpha",
        include_bytes!("fixtures/heif/LA_8__29x100.heif"),
        include_bytes!("fixtures/heif/LA_8__29x100.heif.png"),
        0,
    ),
    (
        "RGB grid alpha",
        include_bytes!("fixtures/heif/RGBA_8__29x100.heif"),
        include_bytes!("fixtures/heif/RGBA_8__29x100.heif.png"),
        0,
    ),
    (
        "gray coded alpha",
        include_bytes!("fixtures/heif/LA_8__128x128.heif"),
        include_bytes!("fixtures/heif/LA_8__128x128.heif.png"),
        0,
    ),
    (
        "10-bit alpha",
        include_bytes!("fixtures/heif/RGBA_10__128x128.heif"),
        include_bytes!("fixtures/heif/RGBA_10__128x128.heif.png"),
        1,
    ),
    (
        "12-bit alpha",
        include_bytes!("fixtures/heif/RGBA_12__128x128.heif"),
        include_bytes!("fixtures/heif/RGBA_12__128x128.heif.png"),
        1,
    ),
];

#[test]
fn system_decoder_preserves_alpha_or_explicitly_rejects_unsupported_variants() {
    if !io::heic::capabilities().decode {
        eprintln!("System HEIC decoder unavailable");
        return;
    }
    for &(name, input, reference, alpha_tolerance) in CASES {
        let decoded = match io::decode_image_bytes(input) {
            Ok(image) => {
                if name == "10-bit alpha" || name == "12-bit alpha" {
                    assert!(
                        matches!(image, image::DynamicImage::ImageRgba16(_)),
                        "High-depth source must not be silently reduced to 8-bit"
                    );
                }
                image.to_rgba8()
            }
            Err(error)
                if (cfg!(target_os = "windows") || cfg!(target_os = "macos"))
                    && (name == "10-bit alpha" || name == "12-bit alpha") =>
            {
                let message = error.to_string();
                assert!(
                    message.contains("transparency")
                        || message.contains("high-bit-depth")
                        || message.contains("Windows HEIC codec failed"),
                    "{message}"
                );
                continue;
            }
            Err(error) => panic!("{name}: {error}"),
        };
        let reference = image::load_from_memory(reference).unwrap().to_rgba8();
        assert_eq!(decoded.dimensions(), reference.dimensions(), "{name}");
        let mut rgb_error = 0u64;
        for (actual, expected) in decoded.pixels().zip(reference.pixels()) {
            assert!(
                actual[3].abs_diff(expected[3]) <= alpha_tolerance,
                "{name}: {actual:?} != {expected:?}"
            );
            for channel in 0..3 {
                let difference = actual[channel].abs_diff(expected[channel]);
                rgb_error += u64::from(difference);
                // Windows' native YUV/chroma reconstruction differs from the
                // libheif reference on this high-contrast synthetic grid. Keep
                // both peak and mean bounds; alpha still must match exactly.
                assert!(
                    difference <= if cfg!(target_os = "windows") { 48 } else { 8 },
                    "{name}: {actual:?} != {expected:?}"
                );
            }
        }
        let mean = rgb_error as f64 / (decoded.width() as f64 * decoded.height() as f64 * 3.0);
        assert!(
            mean <= if cfg!(target_os = "windows") {
                9.0
            } else {
                4.0
            },
            "{name}: RGB mean difference {mean}"
        );
    }
}

#[test]
fn truncated_input_and_inconsistent_alpha_height_return_errors() {
    let input = CASES[1].1;
    for len in [0, 8, 32, input.len() / 2] {
        assert!(io::decode_image_bytes(&input[..len]).is_err());
    }
    let mut malformed = input.to_vec();
    let alpha = malformed
        .windows(4)
        .enumerate()
        .filter_map(|(i, b)| (b == b"ispe").then_some(i))
        .next_back()
        .unwrap();
    malformed[alpha + 12..alpha + 16].copy_from_slice(&99u32.to_be_bytes());
    assert!(
        io::decode_image_bytes(&malformed).is_err(),
        "Mismatched alpha dimensions must not produce an opaque import"
    );
}

#[test]
fn extension_file_loading_preserves_grid_alpha() {
    if !io::heic::capabilities().decode {
        eprintln!("System HEIC decoder unavailable");
        return;
    }
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/heif/RGBA_8__29x100.heif");
    let state = io::load_image_sync(&path).unwrap();
    assert_eq!((state.width, state.height), (29, 100));
    assert!(state.layers[0].pixels.get_pixel(0, 0)[3] < 255);
}

#[cfg(target_os = "windows")]
#[test]
#[ignore = "requires PAINTFE_TEST_HEIF_LIBRARY pointing to a user-supplied libheif DLL"]
fn user_supplied_windows_libheif_opens_alpha_fixtures_and_can_export() {
    let path = std::env::var("PAINTFE_TEST_HEIF_LIBRARY").expect("set PAINTFE_TEST_HEIF_LIBRARY");
    io::heic::set_library_path(&path);
    let caps = io::heic::capabilities();
    assert!(caps.decode, "{}", caps.detail);
    for &(_, input, reference, alpha_tolerance) in CASES {
        let actual = io::decode_image_bytes(input).unwrap().to_rgba8();
        let expected = image::load_from_memory(reference).unwrap().to_rgba8();
        assert_eq!(actual.dimensions(), expected.dimensions());
        for (actual, expected) in actual.pixels().zip(expected.pixels()) {
            assert!(actual[3].abs_diff(expected[3]) <= alpha_tolerance);
            for c in 0..3 {
                assert!(
                    actual[c].abs_diff(expected[c]) <= 8,
                    "{actual:?} != {expected:?}"
                );
            }
        }
    }
    if caps.encode {
        let image = image::RgbaImage::from_pixel(8, 8, image::Rgba([40, 100, 200, 128]));
        let bytes = io::heic::encode_rgba8(&image, 90).unwrap();
        let restored = io::decode_image_bytes(&bytes).unwrap().to_rgba8();
        assert_eq!(restored.dimensions(), image.dimensions());
        assert!(restored.pixels().all(|pixel| pixel[3].abs_diff(128) <= 1));
    }
    let bad = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join(format!("invalid-heif-{}.dll", uuid::Uuid::new_v4()));
    std::fs::write(&bad, b"not a library").unwrap();
    io::heic::set_library_path(bad.to_str().unwrap());
    let invalid = io::heic::capabilities();
    assert!(!invalid.decode && !invalid.encode);
    assert!(!invalid.detail.is_empty());
    io::heic::set_library_path("");
    std::fs::remove_file(bad).unwrap();
}
