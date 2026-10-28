use image::RgbaImage;
use paintfe::{
    canvas::{Canvas, TiledImage},
    components::dialogs::{SaveFormat, TiffCompression},
    io,
    ops::transform,
};
use std::sync::atomic::AtomicBool;

#[test]
fn image_budget_is_consistent_across_open_resize_and_tiled_storage() {
    for (w, h) in [(0, 1), (1, 0), (20000, 20000), (16001, 16000)] {
        assert!(io::validate_open_dimensions(w, h).is_err());
        assert!(TiledImage::validate_dimensions(w, h).is_err());
    }
    assert!(io::validate_open_dimensions(16000, 16000).is_ok());
    assert!(TiledImage::validate_dimensions(16000, 16000).is_ok());
    let result = transform::resize_layers_with_options(
        vec![RgbaImage::new(1, 1)],
        20000,
        20000,
        Default::default(),
        None,
        &AtomicBool::new(false),
    );
    assert!(result.err().unwrap().contains("256 million"));
}

#[test]
fn cancelled_resizes_do_not_produce_partial_results() {
    let cancel = AtomicBool::new(true);
    assert!(
        transform::resize_layers_with_options(
            vec![RgbaImage::new(2, 2)],
            8,
            8,
            Default::default(),
            None,
            &cancel
        )
        .is_err()
    );
    assert!(
        transform::resize_canvas_layers(
            vec![RgbaImage::new(2, 2)],
            2,
            2,
            8,
            8,
            (0, 0),
            image::Rgba([0; 4]),
            &cancel
        )
        .is_err()
    );
    let result = transform::resize_layers_with_options(
        vec![RgbaImage::new(2, 2)],
        8,
        8,
        Default::default(),
        None,
        &AtomicBool::new(false),
    )
    .unwrap();
    assert_eq!((result[0].width, result[0].height), (8, 8));
}

#[test]
fn fit_and_every_zoom_entry_can_go_below_ten_percent() {
    let mut canvas = Canvas::new("");
    let viewport = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1000.0, 600.0));
    canvas.last_canvas_rect = Some(viewport);
    canvas.fit_to_window((20000, 20000));
    let (zoom, pan) = canvas.view_state();
    assert!((zoom - 0.0288).abs() < 0.00001);
    assert_eq!(pan, egui::Vec2::ZERO);
    canvas.zoom_out();
    let (smaller, _) = canvas.view_state();
    assert!(smaller < zoom);
    canvas.zoom_in();
    assert!(canvas.view_state().0 > smaller);
    canvas.set_view_state(0.001, egui::vec2(2.0, 3.0));
    assert_eq!(canvas.view_state().0, 0.001);
    canvas.apply_zoom(0.5);
    assert_eq!(canvas.view_state().0, canvas.minimum_zoom());
    for _ in 0..30 {
        canvas.zoom_out();
    }
    assert_eq!(canvas.view_state().0, canvas.minimum_zoom());
}

#[test]
fn zoom_floor_tracks_viewport_and_document_without_enlarging_small_images() {
    let mut canvas = Canvas::new("");
    for size in [(20000, 1000), (1000, 20000), (8000, 8000), (32, 24)] {
        for viewport in [egui::vec2(1000., 600.), egui::vec2(600., 1000.)] {
            canvas.last_canvas_rect = Some(egui::Rect::from_min_size(egui::Pos2::ZERO, viewport));
            canvas.fit_to_window(size);
            let fitted = canvas.view_state().0;
            let expected = (fitted * 0.1).clamp(paintfe::canvas::Canvas::MIN_ZOOM, 1.);
            for _ in 0..30 {
                canvas.zoom_out();
            }
            assert!((canvas.view_state().0 - expected).abs() < 0.000001);
            assert!(canvas.view_state().0 <= 1.);
        }
    }
}

#[test]
fn heic_size_budget_reports_actual_dimensions_and_accepts_both_extensions() {
    paintfe::i18n::init();
    assert_eq!(io::heic::max_pixels(), 64_000_000);
    assert!(io::heic::validate_dimensions(8000, 8000).is_ok());
    for (w, h) in [(0, 1), (1, 0), (8001, 8000), (17472, 11712)] {
        let message = io::heic::validate_dimensions(w, h).unwrap_err();
        assert!(
            message.contains(&w.to_string())
                && message.contains(&h.to_string())
                && message.contains("64"),
            "{message}"
        );
    }
    assert!(io::heic::size_limit_message(17472, 11712).contains("204.6"));
    for extension in ["heic", "HEIF"] {
        assert!(io::is_heif_path(std::path::Path::new(&format!(
            "photo.{extension}"
        ))));
    }
}

#[test]
fn unsupported_webp_does_not_truncate_an_existing_destination() {
    let directory = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target/processing-tests")
        .join(uuid::Uuid::new_v4().to_string());
    std::fs::create_dir_all(&directory).unwrap();
    let path = directory.join("original.webp");
    std::fs::write(&path, b"original").unwrap();
    for lossless in [true, false] {
        assert!(
            io::encode_and_write(
                &RgbaImage::new(16384, 1),
                &path,
                SaveFormat::Webp,
                90,
                TiffCompression::None,
                lossless
            )
            .unwrap_err()
            .to_string()
            .contains("16383")
        );
        assert_eq!(std::fs::read(&path).unwrap(), b"original");
    }
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn fallible_webp_preserves_existing_lossless_compression_settings() {
    let directory = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target/processing-tests")
        .join(uuid::Uuid::new_v4().to_string());
    std::fs::create_dir_all(&directory).unwrap();
    let image = RgbaImage::from_fn(8, 8, |x, y| {
        image::Rgba([(x * 30) as u8, (y * 30) as u8, 130, 180])
    });
    let first = directory.join("first.webp");
    let second = directory.join("second.webp");
    io::encode_and_write(
        &image,
        &first,
        SaveFormat::Webp,
        1,
        TiffCompression::None,
        true,
    )
    .unwrap();
    io::encode_and_write(
        &image,
        &second,
        SaveFormat::Webp,
        100,
        TiffCompression::None,
        true,
    )
    .unwrap();
    assert_eq!(
        std::fs::read(&first).unwrap(),
        std::fs::read(&second).unwrap()
    );
    assert_eq!(io::decode_image_path(&first).unwrap().to_rgba8(), image);
    std::fs::remove_dir_all(directory).unwrap();
}
