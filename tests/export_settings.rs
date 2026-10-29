use image::{Rgba, RgbaImage};
use paintfe::components::dialogs::{SaveFormat, TiffCompression};
use paintfe::io::{
    PreparedExportImage,
    export::{
        self, ExportOptions,
        png::{self, Compression, Depth, Mode, PngOptions},
    },
};

fn rgba() -> RgbaImage {
    RgbaImage::from_fn(48, 24, |x, y| {
        Rgba([
            (x * 5) as u8,
            (y * 10) as u8,
            ((x + y) * 3) as u8,
            if x == 0 {
                0
            } else if x < 16 {
                127
            } else {
                255
            },
        ])
    })
}
fn encode(image: &PreparedExportImage, options: PngOptions) -> Vec<u8> {
    let mut bytes = Vec::new();
    png::write(image, options, &mut bytes).unwrap();
    bytes
}
fn info(bytes: &[u8]) -> (png::Depth, ::png::ColorType) {
    let decoder = ::png::Decoder::new(std::io::Cursor::new(bytes));
    let reader = decoder.read_info().unwrap();
    let depth = if reader.info().bit_depth == ::png::BitDepth::Sixteen {
        Depth::Sixteen
    } else {
        Depth::Eight
    };
    (depth, reader.info().color_type)
}

#[test]
fn png_automatic_preserves_alpha_and_selects_rgb_for_opaque_images() {
    let source = rgba();
    let bytes = encode(
        &PreparedExportImage::Rgba8(source.clone()),
        Default::default(),
    );
    assert_eq!(info(&bytes), (Depth::Eight, ::png::ColorType::Rgba));
    assert_eq!(image::load_from_memory(&bytes).unwrap().to_rgba8(), source);
    let opaque = RgbaImage::from_pixel(9, 7, Rgba([42, 82, 123, 255]));
    let bytes = encode(
        &PreparedExportImage::Rgba8(opaque.clone()),
        Default::default(),
    );
    assert_eq!(info(&bytes), (Depth::Eight, ::png::ColorType::Rgb));
    assert_eq!(image::load_from_memory(&bytes).unwrap().to_rgba8(), opaque);
}

#[test]
fn png_depth_and_colour_modes_are_real_output_settings() {
    let pixels = vec![12345, 23456, 34567, 45678, 65535, 54321, 43210, 65535];
    let source = PreparedExportImage::Rgba16 {
        width: 2,
        height: 1,
        pixels: pixels.clone(),
    };
    let bytes = encode(&source, Default::default());
    assert_eq!(info(&bytes), (Depth::Sixteen, ::png::ColorType::Rgba));
    assert_eq!(
        image::load_from_memory(&bytes)
            .unwrap()
            .to_rgba16()
            .as_raw(),
        &pixels
    );
    for (mode, color) in [
        (Mode::Rgba, ::png::ColorType::Rgba),
        (Mode::Rgb, ::png::ColorType::Rgb),
        (Mode::Gray, ::png::ColorType::Grayscale),
        (Mode::GrayAlpha, ::png::ColorType::GrayscaleAlpha),
    ] {
        for depth in [Depth::Eight, Depth::Sixteen] {
            let bytes = encode(
                &source,
                PngOptions {
                    mode,
                    depth,
                    ..Default::default()
                },
            );
            assert_eq!(info(&bytes), (depth, color));
            let decoded = image::load_from_memory(&bytes).unwrap().to_rgba16();
            if matches!(mode, Mode::Gray | Mode::GrayAlpha) {
                assert_eq!(decoded.get_pixel(0, 0)[0], decoded.get_pixel(0, 0)[1]);
            }
            if matches!(mode, Mode::Rgb | Mode::Gray) {
                assert_eq!(decoded.get_pixel(0, 0)[3], 65535);
            }
        }
    }
}

#[test]
fn indexed_png_preserves_full_palette_alpha_and_honours_alpha_removal() {
    let source = RgbaImage::from_fn(12, 1, |x, _| {
        Rgba([35, 90, 120, [0, 41, 127, 254, 255][x as usize % 5]])
    });
    let options = PngOptions {
        mode: Mode::Indexed,
        palette_colors: 8,
        ..Default::default()
    };
    let bytes = encode(&PreparedExportImage::Rgba8(source.clone()), options);
    assert_eq!(info(&bytes), (Depth::Eight, ::png::ColorType::Indexed));
    assert_eq!(image::load_from_memory(&bytes).unwrap().to_rgba8(), source);
    let bytes = encode(
        &PreparedExportImage::Rgba8(source),
        PngOptions {
            preserve_palette_alpha: false,
            ..options
        },
    );
    assert!(
        image::load_from_memory(&bytes)
            .unwrap()
            .to_rgba8()
            .pixels()
            .all(|p| p[3] == 255)
    );
}

#[test]
fn compression_changes_size_without_changing_pixels() {
    let source = RgbaImage::from_fn(256, 128, |x, y| {
        Rgba([(x / 16) as u8, (y / 8) as u8, 180, 255])
    });
    let mut sizes = Vec::new();
    for compression in [
        Compression::Fast,
        Compression::Balanced,
        Compression::Smaller,
    ] {
        let bytes = encode(
            &PreparedExportImage::Rgba8(source.clone()),
            PngOptions {
                compression,
                ..Default::default()
            },
        );
        sizes.push(bytes.len());
        assert_eq!(image::load_from_memory(&bytes).unwrap().to_rgba8(), source);
    }
    assert_ne!(sizes[0], sizes[2]);
}

#[test]
fn palette_size_and_dithering_affect_png_and_static_and_animated_gif() {
    let source = rgba();
    let mut png_results = Vec::new();
    let directory = std::env::temp_dir().join(format!("paintfe-export-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir(&directory).unwrap();
    for dither in [false, true] {
        let png = encode(
            &PreparedExportImage::Rgba8(source.clone()),
            PngOptions {
                mode: Mode::Indexed,
                palette_colors: 8,
                dither,
                ..Default::default()
            },
        );
        let reader = ::png::Decoder::new(std::io::Cursor::new(&png))
            .read_info()
            .unwrap();
        assert!(reader.info().palette.as_ref().unwrap().len() / 3 <= 8);
        png_results.push(image::load_from_memory(&png).unwrap().to_rgba8());
        let options = ExportOptions {
            format: SaveFormat::Gif,
            gif_colors: 8,
            gif_dither: dither,
            ..Default::default()
        };
        let path = directory.join(format!("static-{dither}.gif"));
        export::write(PreparedExportImage::Rgba8(source.clone()), &path, options).unwrap();
        let still = image::open(&path).unwrap().to_rgba8();
        assert_eq!(still.get_pixel(0, 0)[3], 0);
        let animated = directory.join(format!("animated-{dither}.gif"));
        paintfe::io::encode_animated_gif(
            &[source.clone(), source.clone()],
            10.0,
            8,
            dither,
            &animated,
        )
        .unwrap();
        assert_eq!(image::open(&animated).unwrap().to_rgba8(), still);
    }
    assert_ne!(png_results[0], png_results[1]);
    assert_ne!(
        image::open(directory.join("static-false.gif"))
            .unwrap()
            .to_rgba8(),
        image::open(directory.join("static-true.gif"))
            .unwrap()
            .to_rgba8()
    );
    for entry in std::fs::read_dir(&directory).unwrap() {
        std::fs::remove_file(entry.unwrap().path()).unwrap();
    }
    std::fs::remove_dir(directory).unwrap();
}

#[test]
fn preview_and_final_output_use_identical_settings_for_png_tiff_and_gif() {
    let directory = std::env::temp_dir().join(format!("paintfe-export-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir(&directory).unwrap();
    for format in [SaveFormat::Png, SaveFormat::Tiff, SaveFormat::Gif] {
        for compression in [
            TiffCompression::None,
            TiffCompression::Lzw,
            TiffCompression::Deflate,
        ] {
            let options = ExportOptions {
                format,
                tiff_compression: compression,
                gif_colors: 8,
                gif_dither: false,
                png: PngOptions {
                    mode: Mode::GrayAlpha,
                    ..Default::default()
                },
                ..Default::default()
            };
            let image = PreparedExportImage::Rgba8(rgba());
            let preview = export::preview(image.clone(), options).unwrap();
            let path = directory.join(format!("output.{}", format.extension()));
            export::write(image, &path, options).unwrap();
            assert_eq!(
                preview.file_size,
                std::fs::metadata(&path).unwrap().len() as usize
            );
            assert_eq!(
                preview.preview_image,
                image::open(&path).unwrap().to_rgba8()
            );
            std::fs::remove_file(path).unwrap();
        }
    }
    std::fs::remove_dir(directory).unwrap();
}

#[test]
fn float_tiff_honours_compression_and_preserves_hdr_samples() {
    let directory = std::env::temp_dir().join(format!("paintfe-tiff-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir(&directory).unwrap();
    let path = directory.join("float.tiff");
    let pixels = vec![0.125f32, 2.5, 0.75, 1.0, -0.25, 0.5, 4.0, 0.25];
    for (compression, tag) in [
        (TiffCompression::None, 1),
        (TiffCompression::Lzw, 5),
        (TiffCompression::Deflate, 8),
    ] {
        let options = ExportOptions {
            format: SaveFormat::Tiff,
            tiff_compression: compression,
            ..Default::default()
        };
        export::write(
            PreparedExportImage::RgbaF32 {
                width: 2,
                height: 1,
                pixels: pixels.clone(),
            },
            &path,
            options,
        )
        .unwrap();
        let mut decoder = tiff::decoder::Decoder::new(std::fs::File::open(&path).unwrap()).unwrap();
        assert_eq!(
            decoder
                .get_tag_unsigned::<u16>(tiff::tags::Tag::Compression)
                .unwrap(),
            tag
        );
        let tiff::decoder::DecodingResult::F32(actual) = decoder.read_image().unwrap() else {
            panic!("not floating-point TIFF")
        };
        assert_eq!(actual, pixels);
    }
    std::fs::remove_file(path).unwrap();
    std::fs::remove_dir(directory).unwrap();
}

#[test]
fn still_export_ignores_source_animation_timing_and_animation_keeps_supported_timing() {
    let directory = std::env::temp_dir().join(format!("paintfe-timing-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir(&directory).unwrap();
    let path = directory.join("still.jpg");
    let options = ExportOptions {
        format: SaveFormat::Jpeg,
        animation_fps: 100.0,
        ..Default::default()
    };
    export::write(PreparedExportImage::Rgba8(rgba()), &path, options).unwrap();
    assert_eq!(image::open(&path).unwrap().width(), 48);
    assert!(
        ExportOptions {
            format: SaveFormat::Gif,
            animated: true,
            animation_fps: 100.0,
            ..Default::default()
        }
        .validate()
        .is_ok()
    );
    std::fs::remove_file(path).unwrap();
    std::fs::remove_dir(directory).unwrap();
}
