//! Independent libheif-generated reference pixels; test fixtures are BSD-3-Clause.
use paintfe::io;

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
