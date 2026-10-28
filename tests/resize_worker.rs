#![cfg(not(target_arch = "wasm32"))]
use paintfe::{
    canvas::TiledImage,
    ops::transform::{self, Interpolation, PixelRetargetMode, ResizeImageOptions, ResizeMethod},
    services::resize_worker,
};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::{Duration, Instant};
static NATIVE_TEST: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
fn native_resize_matches_all_interpolators_and_retarget_modes() {
    let _guard = NATIVE_TEST.lock().unwrap();
    let source = image::RgbaImage::from_fn(43, 31, |x, y| {
        image::Rgba([
            (x * 5) as u8,
            (y * 7) as u8,
            97,
            if (x + y) % 5 == 0 { 0 } else { 210 },
        ])
    });
    let mask = image::GrayImage::from_pixel(43, 31, image::Luma([128]));
    let mut options = Vec::new();
    for interpolation in Interpolation::all() {
        options.push(ResizeImageOptions {
            interpolation: *interpolation,
            ..Default::default()
        });
    }
    for mode in [
        PixelRetargetMode::PaletteAware,
        PixelRetargetMode::StructurePreserving,
        PixelRetargetMode::TopologyPreserving,
    ] {
        let mut option = ResizeImageOptions {
            method: ResizeMethod::PixelArtRetarget,
            ..Default::default()
        };
        option.retarget.mode = mode;
        options.push(option);
    }
    for option in options {
        let expected = transform::resize_layers_with_options(
            vec![source.clone()],
            29,
            17,
            option,
            Some(mask.clone()),
            &AtomicBool::new(false),
        )
        .unwrap();
        let actual = resize_worker::resize_layers(
            vec![TiledImage::from_rgba_image(&source)],
            29,
            17,
            option,
            Some(mask.clone()),
            &AtomicBool::new(false),
        )
        .unwrap();
        assert_eq!(
            actual[0].to_rgba_image(),
            expected[0].to_rgba_image(),
            "{:?} {:?}",
            option.method,
            option.interpolation
        );
    }
}
#[test]
fn native_resize_preserves_multiple_layers_edges_and_transparency() {
    let _guard = NATIVE_TEST.lock().unwrap();
    let input = image::RgbaImage::from_fn(259, 263, |x, y| {
        image::Rgba([x as u8, y as u8, 50, if x % 11 == 0 { 0 } else { 255 }])
    });
    let layers = vec![
        TiledImage::from_rgba_image(&input),
        TiledImage::new(259, 263),
    ];
    let output = resize_worker::resize_layers(
        layers,
        521,
        529,
        Default::default(),
        None,
        &AtomicBool::new(false),
    )
    .unwrap();
    let expected = transform::resize_layers_with_options(
        vec![input],
        521,
        529,
        Default::default(),
        None,
        &AtomicBool::new(false),
    )
    .unwrap();
    assert_eq!(output.len(), 2);
    assert_eq!(output[0].to_rgba_image(), expected[0].to_rgba_image());
    assert_eq!(output[1].chunk_count(), 0);
}
#[test]
fn running_resample_is_terminated_promptly_and_next_job_succeeds() {
    let _guard = NATIVE_TEST.lock().unwrap();
    for interpolation in [Interpolation::Bicubic, Interpolation::Lanczos3] {
        let cancel = Arc::new(AtomicBool::new(false));
        let flag = cancel.clone();
        let pixels = TiledImage::from_rgba_image(&image::RgbaImage::from_pixel(
            1024,
            1024,
            image::Rgba([100, 70, 50, 255]),
        ));
        let job = std::thread::spawn(move || {
            resize_worker::resize_layers(
                vec![pixels],
                8192,
                8192,
                ResizeImageOptions {
                    interpolation,
                    ..Default::default()
                },
                None,
                &flag,
            )
        });
        let deadline = Instant::now() + Duration::from_secs(20);
        while resize_worker::active_workers() == 0 {
            assert!(Instant::now() < deadline, "worker never reached resampling");
            assert!(!job.is_finished(), "worker finished before cancellation");
            std::thread::sleep(Duration::from_millis(2));
        }
        let start = Instant::now();
        cancel.store(true, Ordering::Relaxed);
        assert_eq!(job.join().unwrap().err().unwrap(), "Cancelled");
        eprintln!(
            "Resize cancellation {:?}: {:.3}s",
            interpolation,
            start.elapsed().as_secs_f64()
        );
        assert!(start.elapsed() < Duration::from_secs(2));
        assert_eq!(resize_worker::active_workers(), 0);
    }
    let result = resize_worker::resize_layers(
        vec![TiledImage::new_filled(8, 8, image::Rgba([1, 2, 3, 255]))],
        16,
        16,
        Default::default(),
        None,
        &AtomicBool::new(false),
    )
    .unwrap();
    assert_eq!(*result[0].get_pixel(0, 0), image::Rgba([1, 2, 3, 255]));
}
#[test]
fn cancelled_preparation_and_invalid_dimensions_do_not_start_a_worker() {
    let _guard = NATIVE_TEST.lock().unwrap();
    assert!(
        resize_worker::resize_layers(
            vec![TiledImage::new(8, 8)],
            16,
            16,
            Default::default(),
            None,
            &AtomicBool::new(true)
        )
        .is_err()
    );
    assert!(
        resize_worker::resize_layers(
            vec![],
            20000,
            20000,
            Default::default(),
            None,
            &AtomicBool::new(false)
        )
        .is_err()
    );
    assert_eq!(resize_worker::active_workers(), 0);
}
