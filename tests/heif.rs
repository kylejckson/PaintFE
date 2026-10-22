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
fn bundled_decoder_matches_independent_alpha_and_color_references() {
    for &(name, input, reference, alpha_tolerance) in CASES {
        let decoded = io::decode_image_bytes(input)
            .unwrap_or_else(|e| panic!("{name}: {e}"))
            .to_rgba8();
        let reference = image::load_from_memory(reference).unwrap().to_rgba8();
        assert_eq!(decoded.dimensions(), reference.dimensions(), "{name}");
        for (actual, expected) in decoded.pixels().zip(reference.pixels()) {
            assert!(
                actual[3].abs_diff(expected[3]) <= alpha_tolerance,
                "{name}: {actual:?} != {expected:?}"
            );
            for channel in 0..3 {
                assert!(
                    actual[channel].abs_diff(expected[channel]) <= 4,
                    "{name}: {actual:?} != {expected:?}"
                );
            }
        }
    }
}

#[test]
fn truncated_input_and_inconsistent_alpha_height_return_errors() {
    let input = CASES[1].1;
    for len in [0, 8, 32, input.len() / 2] {
        assert!(io::decode_image_bytes(&input[..len]).is_err());
    }
    let mut malformed = input.to_vec();
    let positions: Vec<_> = malformed
        .windows(4)
        .enumerate()
        .filter_map(|(i, b)| (b == b"ispe").then_some(i))
        .collect();
    // The alpha's spatial extent is a separate property in this reviewed fixture.
    let alpha = *positions.last().unwrap();
    malformed[alpha + 12..alpha + 16].copy_from_slice(&99u32.to_be_bytes());
    assert!(io::decode_image_bytes(&malformed).is_err());
    // Also make the auxiliary grid itself 99 rows tall, consistently with its
    // ispe. This reaches the alpha-vs-parent size guard after valid decoding.
    let file = heifer::heifer_isobmff::HeifFile::parse(input).unwrap();
    let auxiliary = file.referencing_items(file.primary_id, b"auxl")[0];
    let grid = file.item_data(auxiliary).unwrap();
    let payload = malformed
        .windows(grid.len())
        .rposition(|bytes| bytes == &grid[..])
        .unwrap();
    assert_eq!(grid.len(), 8);
    malformed[payload + 6..payload + 8].copy_from_slice(&99u16.to_be_bytes());
    let error = io::decode_image_bytes(&malformed).unwrap_err().to_string();
    assert!(error.contains("alpha dimensions"), "{error}");
}

#[test]
fn extension_file_loading_preserves_grid_alpha() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/heif/RGBA_8__29x100.heif");
    let state = io::load_image_sync(&path).unwrap();
    assert_eq!((state.width, state.height), (29, 100));
    assert!(state.layers[0].pixels.get_pixel(0, 0)[3] < 255);
}
