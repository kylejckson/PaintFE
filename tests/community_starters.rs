#![cfg(not(target_arch = "wasm32"))]
use paintfe::{ops::scripting::execute_script_parameters, services::plugins::Manifest};
fn run(name: &str, w: u32, h: u32, pixels: Vec<u8>, overrides: serde_json::Value) -> Vec<u8> {
    let at = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/community-starters")
        .join(name);
    let m: Manifest =
        serde_json::from_str(&std::fs::read_to_string(at.join("manifest.json")).unwrap()).unwrap();
    let effect = &m.effects[0];
    let mut values = effect.defaults();
    if let Some(map) = overrides.as_object() {
        values.extend(map.iter().map(|(k, v)| (k.clone(), v.clone())));
    }
    effect.validate_values(&values).unwrap();
    let source = std::fs::read_to_string(at.join("effect.rhai")).unwrap();
    let (out, ow, oh, _, ops) =
        execute_script_parameters(&source, pixels, w, h, None, &values, Some(0)).unwrap();
    assert_eq!((ow, oh), (w, h));
    assert!(ops.is_empty());
    out
}
#[test]
fn starter_effects_preserve_every_alpha_and_have_neutral_settings() {
    let pixels: Vec<u8> = (0..64)
        .flat_map(|i| {
            [
                (i * 3) as u8,
                (i * 2) as u8,
                (255 - i * 2) as u8,
                (i * 4) as u8,
            ]
        })
        .collect();
    for name in ["two-tone", "paper-grain", "colour-offset", "soft-vignette"] {
        let output = run(name, 8, 8, pixels.clone(), serde_json::json!({}));
        assert_eq!(output.len(), pixels.len());
        assert!(
            output
                .chunks_exact(4)
                .zip(pixels.chunks_exact(4))
                .all(|(a, b)| a[3] == b[3])
        );
        let neutral = match name {
            "two-tone" | "colour-offset" => serde_json::json!({"strength":0.}),
            _ => serde_json::json!({"amount":0.}),
        };
        assert_eq!(run(name, 8, 8, pixels.clone(), neutral), pixels);
    }
}
#[test]
fn two_tone_endpoints_match_selected_colours() {
    assert_eq!(
        run(
            "two-tone",
            2,
            1,
            vec![0, 0, 0, 255, 255, 255, 255, 128],
            serde_json::json!({})
        ),
        vec![25, 43, 67, 255, 245, 188, 135, 128]
    );
}
#[test]
fn colour_offset_samples_original_pixels_without_accumulation() {
    let input = vec![10, 20, 30, 255, 40, 50, 60, 128, 70, 80, 90, 0];
    assert_eq!(
        run(
            "colour-offset",
            3,
            1,
            input,
            serde_json::json!({"horizontal":1})
        ),
        vec![10, 20, 30, 255, 10, 50, 60, 128, 40, 80, 90, 0]
    );
}
#[test]
fn grain_is_repeatable_and_seed_changes_the_result() {
    let input = vec![128; 256];
    let first = run("paper-grain", 8, 8, input.clone(), serde_json::json!({}));
    assert_eq!(
        first,
        run("paper-grain", 8, 8, input.clone(), serde_json::json!({}))
    );
    assert_ne!(
        first,
        run("paper-grain", 8, 8, input, serde_json::json!({"seed":2}))
    );
}
#[test]
fn vignette_preserves_centre_and_darkens_edges() {
    let out = run(
        "soft-vignette",
        9,
        9,
        vec![255; 9 * 9 * 4],
        serde_json::json!({}),
    );
    assert_eq!(out[(4 * 9 + 4) * 4], 255);
    assert!(out[0] < 255);
}

#[test]
fn native_batch_operations_reject_invalid_bounds_without_panicking() {
    for source in [
        "map_luminance([0,0,0,0],[255,255,255,255],100.0,1.0);",
        "apply_seeded_grain(0.2,0,1,true);",
        "offset_channel(0,-9223372036854775808,0,1.0);",
        "apply_vignette_at(0.5,0.5,0.0,0.5,1.0);",
    ] {
        assert!(
            execute_script_parameters(
                source,
                vec![255; 16],
                2,
                2,
                None,
                &Default::default(),
                Some(0)
            )
            .is_err()
        );
    }
}
#[test]
fn native_batch_operations_respect_feathered_selection_once() {
    let source = "map_luminance([200,100,0,255],[200,100,0,255],1.0,1.0);";
    let (out, _, _, _, _) = execute_script_parameters(
        source,
        vec![0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255],
        3,
        1,
        Some(vec![0, 128, 255]),
        &Default::default(),
        Some(0),
    )
    .unwrap();
    assert_eq!(out, vec![0, 0, 0, 255, 100, 50, 0, 255, 200, 100, 0, 255]);
}
