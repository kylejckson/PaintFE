use image::{Rgba, RgbaImage};
use paintfe::components::dialogs::SaveFormat;
use paintfe::io::{PreparedExportImage, export::*};
struct Folder(std::path::PathBuf);
impl Folder {
    fn new() -> Self {
        let p =
            std::env::temp_dir().join(format!("paintfe-export-options-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir(&p).unwrap();
        Self(p)
    }
}
impl Drop for Folder {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn fixture() -> RgbaImage {
    RgbaImage::from_fn(37, 29, |x, y| {
        Rgba([
            (x * 7) as u8,
            (y * 9) as u8,
            (x * 11 + y * 3) as u8,
            ((x + y) * 13) as u8,
        ])
    })
}
#[test]
fn jpeg_subsampling_is_written_into_real_sof_headers() {
    let folder = Folder::new();
    let image = PreparedExportImage::Rgba8(fixture());
    for (sampling, hv) in [
        (JpegSampling::Full, 0x11),
        (JpegSampling::Vertical, 0x12),
        (JpegSampling::Horizontal, 0x21),
        (JpegSampling::Quarter, 0x22),
    ] {
        let p = folder.0.join("test.jpg");
        write(
            image.clone(),
            &p,
            ExportOptions {
                format: SaveFormat::Jpeg,
                jpeg_sampling: sampling,
                ..Default::default()
            },
        )
        .unwrap();
        let bytes = std::fs::read(&p).unwrap();
        let mut at = 2;
        let mut found = false;
        while at + 4 < bytes.len() {
            assert_eq!(bytes[at], 255);
            let marker = bytes[at + 1];
            let len = u16::from_be_bytes([bytes[at + 2], bytes[at + 3]]) as usize;
            if marker == 0xc0 {
                assert_eq!(bytes[at + 11], hv);
                found = true;
                break;
            }
            at += 2 + len;
        }
        assert!(found);
        assert_eq!(image::open(&p).unwrap().width(), 37);
    }
}
#[test]
fn adam7_has_real_seven_pass_pixels_for_all_colour_modes_and_depths() {
    let folder = Folder::new();
    for size in [(1, 1), (2, 3), (7, 9), (37, 29)] {
        let source = RgbaImage::from_fn(size.0, size.1, |x, y| {
            Rgba([(x * 7) as u8, (y * 9) as u8, 83, (x * 17 + y * 31) as u8])
        });
        for mode in [
            png::Mode::Rgba,
            png::Mode::Rgb,
            png::Mode::Gray,
            png::Mode::GrayAlpha,
            png::Mode::Indexed,
        ] {
            for depth in [png::Depth::Eight, png::Depth::Sixteen] {
                let mut o = ExportOptions::default();
                o.png.mode = mode;
                o.png.depth = depth;
                let regular = folder.0.join("regular.png");
                write(PreparedExportImage::Rgba8(source.clone()), &regular, o).unwrap();
                o.png.interlaced = true;
                let interlaced = folder.0.join("adam7.png");
                write(PreparedExportImage::Rgba8(source.clone()), &interlaced, o).unwrap();
                let bytes = std::fs::read(&interlaced).unwrap();
                assert_eq!(bytes[28], 1);
                assert_eq!(
                    image::open(regular).unwrap(),
                    image::open(interlaced).unwrap(),
                    "{size:?}/{mode:?}/{depth:?}"
                );
            }
        }
    }
}
#[test]
fn tga_rle_and_bmp_depth_and_palettes_change_real_headers() {
    let folder = Folder::new();
    let source = PreparedExportImage::Rgba8(fixture());
    for (format, mode, bits) in [
        (SaveFormat::Bmp, RasterMode::Rgb, 24),
        (SaveFormat::Bmp, RasterMode::Rgba, 32),
        (SaveFormat::Bmp, RasterMode::Indexed, 8),
        (SaveFormat::Tga, RasterMode::Rgb, 24),
        (SaveFormat::Tga, RasterMode::Rgba, 32),
    ] {
        for rle in [false, true] {
            let p = folder.0.join(format!("test.{}", format.extension()));
            write(
                source.clone(),
                &p,
                ExportOptions {
                    format,
                    raster_mode: mode,
                    tga_rle: rle,
                    ..Default::default()
                },
            )
            .unwrap();
            let bytes = std::fs::read(&p).unwrap();
            if format == SaveFormat::Bmp {
                assert_eq!(u16::from_le_bytes([bytes[28], bytes[29]]), bits);
            } else {
                assert_eq!(bytes[16], bits as u8);
                assert_eq!(bytes[2], if rle { 10 } else { 2 });
            }
            assert_eq!(image::open(p).unwrap().width(), 37);
        }
    }
}
#[test]
fn tiff_explicit_precision_and_colour_are_independently_decodable() {
    let folder = Folder::new();
    let source = PreparedExportImage::Rgba8(fixture());
    for depth in [RasterDepth::Eight, RasterDepth::Sixteen, RasterDepth::Float] {
        for mode in [RasterMode::Rgba, RasterMode::Rgb, RasterMode::Gray] {
            let path = folder.0.join("test.tiff");
            write(
                source.clone(),
                &path,
                ExportOptions {
                    format: SaveFormat::Tiff,
                    raster_depth: depth,
                    raster_mode: mode,
                    ..Default::default()
                },
            )
            .unwrap();
            let mut decoder =
                tiff::decoder::Decoder::new(std::fs::File::open(path).unwrap()).unwrap();
            assert_eq!(decoder.dimensions().unwrap(), (37, 29));
            let bits = match decoder.colortype().unwrap() {
                tiff::ColorType::RGBA(b) | tiff::ColorType::RGB(b) | tiff::ColorType::Gray(b) => b,
                _ => panic!("Unexpected TIFF colour type"),
            };
            assert_eq!(
                bits,
                match depth {
                    RasterDepth::Eight => 8,
                    RasterDepth::Sixteen => 16,
                    _ => 32,
                }
            );
            decoder.read_image().unwrap();
        }
    }
}
#[test]
fn quantizers_dithering_strength_and_gif_threshold_apply_to_static_and_animation() {
    let folder = Folder::new();
    let source = fixture();
    let (_, plain) =
        palette::quantize_with_options(&source, 8, false, palette::Algorithm::MedianCut, 100);
    let (_, zero) =
        palette::quantize_with_options(&source, 8, true, palette::Algorithm::MedianCut, 0);
    assert_eq!(plain, zero);
    let (_, dither) =
        palette::quantize_with_options(&source, 8, true, palette::Algorithm::MedianCut, 100);
    assert_ne!(plain, dither);
    let (median, _) =
        palette::quantize_with_options(&source, 8, false, palette::Algorithm::MedianCut, 100);
    let (neural, _) =
        palette::quantize_with_options(&source, 8, false, palette::Algorithm::NeuQuant, 100);
    assert_ne!(median, neural);
    for threshold in [0, 64, 128, 255] {
        let o = ExportOptions {
            format: SaveFormat::Gif,
            gif_alpha_threshold: threshold,
            ..Default::default()
        };
        for animated in [false, true] {
            let p = folder.0.join("test.gif");
            if animated {
                paintfe::io::encode_animated_gif_options(
                    &[source.clone(), source.clone()],
                    10.0,
                    o,
                    &p,
                )
                .unwrap();
            } else {
                write(PreparedExportImage::Rgba8(source.clone()), &p, o).unwrap();
            }
            let pixels = image::open(p).unwrap().to_rgba8();
            for (a, b) in source.pixels().zip(pixels.pixels()) {
                assert_eq!(b[3], if a[3] < threshold { 0 } else { 255 });
            }
        }
    }
}
#[cfg(not(target_arch = "wasm32"))]
#[test]
fn webp_presets_keep_lossless_pixels_and_produce_valid_lossy_output() {
    let folder = Folder::new();
    let source = fixture();
    for preset in [
        WebpPreset::Default,
        WebpPreset::Photo,
        WebpPreset::Picture,
        WebpPreset::Drawing,
        WebpPreset::Icon,
        WebpPreset::Text,
    ] {
        for lossless in [false, true] {
            let path = folder.0.join("test.webp");
            write(
                PreparedExportImage::Rgba8(source.clone()),
                &path,
                ExportOptions {
                    format: SaveFormat::Webp,
                    webp_preset: preset,
                    webp_lossless: lossless,
                    ..Default::default()
                },
            )
            .unwrap();
            let decoded = image::open(path).unwrap().to_rgba8();
            assert_eq!(decoded.dimensions(), source.dimensions());
            if lossless {
                assert_eq!(decoded, source);
            }
        }
    }
}
#[test]
fn new_options_default_for_old_preferences_and_validate_ranges() {
    let old: ExportOptions = serde_json::from_str("{}").unwrap();
    assert_eq!(old, ExportOptions::default());
    let mut o = old;
    o.dither_strength = 101;
    assert!(o.validate().is_err());
    o.dither_strength = 100;
    o.png.dither_strength = 101;
    assert!(o.validate().is_err());
    o.png.dither_strength = 100;
    o.avif_speed = 0;
    assert!(o.validate().is_err());
    let o = ExportOptions {
        jpeg_sampling: JpegSampling::Quarter,
        avif_lossless: true,
        png: png::PngOptions {
            interlaced: true,
            ..Default::default()
        },
        ..Default::default()
    };
    assert_eq!(
        serde_json::from_str::<ExportOptions>(&serde_json::to_string(&o).unwrap()).unwrap(),
        o
    );
}

#[test]
fn invalid_gif_requests_preserve_existing_files() {
    let folder = Folder::new();
    let path = folder.0.join("existing.bin");
    std::fs::write(&path, b"keep me").unwrap();
    let source = image::RgbaImage::from_pixel(2, 2, image::Rgba([1, 2, 3, 128]));
    for fps in [0.0, f32::NAN, f32::INFINITY] {
        assert!(
            paintfe::io::encode_animated_gif_options(
                std::slice::from_ref(&source),
                fps,
                ExportOptions::default(),
                &path
            )
            .is_err()
        );
        assert_eq!(std::fs::read(&path).unwrap(), b"keep me");
    }
    let other = image::RgbaImage::new(3, 2);
    assert!(
        paintfe::io::encode_animated_gif_options(
            &[source, other],
            10.0,
            ExportOptions::default(),
            &path
        )
        .is_err()
    );
    assert_eq!(std::fs::read(&path).unwrap(), b"keep me");
}

#[test]
fn indexed_bmp_does_not_spend_its_palette_on_discarded_alpha_values() {
    let folder = Folder::new();
    let source = RgbaImage::from_fn(64, 32, |x, y| {
        Rgba([(x * 4) as u8, (x * 4) as u8, 0, (y * 8) as u8])
    });
    for algorithm in [palette::Algorithm::MedianCut, palette::Algorithm::NeuQuant] {
        let path = folder.0.join("indexed.bmp");
        write(
            PreparedExportImage::Rgba8(source.clone()),
            &path,
            ExportOptions {
                format: SaveFormat::Bmp,
                raster_mode: RasterMode::Indexed,
                quantizer: algorithm,
                ..Default::default()
            },
        )
        .unwrap();
        let decoded = image::open(path).unwrap().to_rgba8();
        for (actual, expected) in decoded.pixels().zip(source.pixels()) {
            assert_eq!(actual.0[..3], expected.0[..3]);
            assert_eq!(actual[3], 255);
        }
    }
}
