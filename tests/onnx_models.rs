//! Native runtime checks are opt-in because ONNX Runtime is supplied by the user.
use paintfe::ops::ai::{
    GraphOptimization, InferenceOptions, ModelProfile, RemoveBgSettings,
    remove_background_with_options,
};

fn options() -> InferenceOptions {
    InferenceOptions {
        cpu_threads: Some(2),
        graph_optimization: GraphOptimization::Basic,
        cpu_memory_arena: false,
    }
}

fn profile_from_environment() -> ModelProfile {
    match std::env::var("PAINTFE_MODEL_PROFILE").as_deref() {
        Ok("birefnet") => ModelProfile::BiRefNet,
        Ok("u2net") => ModelProfile::U2Net,
        Ok("clearcut") => ModelProfile::ISNetClearCut,
        Ok("transformers") => ModelProfile::ISNetTransformers,
        Ok("original") => ModelProfile::ISNet,
        Ok("rembg") => ModelProfile::ISNetRembg,
        _ => ModelProfile::Auto,
    }
}

#[test]
#[ignore = "Requires PAINTFE_ONNX_RUNTIME pointing to a local runtime library"]
fn native_runtime_handles_float32_float16_and_uint8_masks() {
    let runtime = std::env::var("PAINTFE_ONNX_RUNTIME").expect("Set PAINTFE_ONNX_RUNTIME");
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/onnx");
    let input = image::RgbaImage::from_pixel(2, 2, image::Rgba([255, 128, 64, 255]));
    for (name, profile) in [
        ("float32", ModelProfile::ISNet),
        ("float16", ModelProfile::ISNet),
        ("uint8", ModelProfile::ISNetTransformers),
    ] {
        let settings = RemoveBgSettings {
            model_profile: profile,
            ..Default::default()
        };
        let result = remove_background_with_options(
            &runtime,
            root.join(format!("{name}.onnx")).to_str().unwrap(),
            &input,
            &settings,
            &options(),
        )
        .unwrap();
        let alpha: Vec<_> = result.pixels().map(|p| p[3]).collect();
        for (&actual, expected) in alpha.iter().zip([0i32, 64, 191, 255]) {
            assert!((actual as i32 - expected).abs() <= 1, "{name}: {alpha:?}");
        }
        assert_eq!(alpha[0], 0, "{name}: transparent endpoint");
        assert_eq!(alpha[3], 255, "{name}: opaque endpoint");
    }
}

#[test]
#[ignore = "Requires local runtime, model, input and a reference mask"]
fn native_isnet_matches_reference_mask() {
    let runtime = std::env::var("PAINTFE_ONNX_RUNTIME").unwrap();
    let model = std::env::var("PAINTFE_MODEL_PATH").unwrap();
    let input = image::open(std::env::var("PAINTFE_MODEL_INPUT").unwrap())
        .unwrap()
        .to_rgba8();
    let expected = image::open(std::env::var("PAINTFE_MODEL_MASK").unwrap())
        .unwrap()
        .to_luma8();
    let profile = profile_from_environment();
    let settings = RemoveBgSettings {
        model_profile: profile,
        ..Default::default()
    };
    let actual =
        remove_background_with_options(&runtime, &model, &input, &settings, &options()).unwrap();
    if let Ok(path) = std::env::var("PAINTFE_MODEL_OUTPUT") {
        actual.save(path).unwrap();
    }
    assert_eq!(actual.dimensions(), expected.dimensions());
    let differences: Vec<_> = actual
        .pixels()
        .zip(expected.pixels())
        .map(|(a, e)| a[3].abs_diff(e[0]) as f64)
        .collect();
    let mean = differences.iter().sum::<f64>() / differences.len() as f64;
    let max = differences.iter().copied().fold(0.0, f64::max);
    eprintln!("Reference alpha difference: mean={mean:.4}/255 max={max}/255");
    assert!(mean <= 1.0, "Reference mismatch: mean {mean}/255");
}

#[test]
#[ignore = "Requires a local runtime, model and external test photo"]
fn native_photo_preserves_rgb_and_has_opaque_foreground() {
    let runtime = std::env::var("PAINTFE_ONNX_RUNTIME").unwrap();
    let model = std::env::var("PAINTFE_MODEL_PATH").unwrap();
    let input = image::open(std::env::var("PAINTFE_MODEL_INPUT").unwrap())
        .unwrap()
        .to_rgba8();
    let settings = RemoveBgSettings {
        model_profile: profile_from_environment(),
        threshold: 0.4,
        edge_feather: 1.0,
        mask_expansion: -1,
        ..Default::default()
    };
    let actual =
        remove_background_with_options(&runtime, &model, &input, &settings, &options()).unwrap();
    assert_eq!(actual.dimensions(), input.dimensions());
    let mut opaque = 0;
    let mut transparent = 0;
    for (a, source) in actual.pixels().zip(input.pixels()) {
        assert_eq!(&a.0[..3], &source.0[..3]);
        assert!(a[3] <= source[3]);
        opaque += usize::from(a[3] == 255);
        transparent += usize::from(a[3] == 0);
    }
    assert!(
        opaque > 0 && transparent > 0,
        "opaque={opaque} transparent={transparent}"
    );
    eprintln!(
        "Photo: opaque={opaque}, transparent={transparent}, total={}",
        input.width() * input.height()
    );
    if let Ok(path) = std::env::var("PAINTFE_MODEL_OUTPUT") {
        actual.save(path).unwrap();
    }
}
