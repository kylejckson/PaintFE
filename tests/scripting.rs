// =============================================================================
// Integration tests — Rhai scripting engine
// =============================================================================
//
// Tests the headless script execution pipeline via `execute_script_sync`.
// These also serve as regression tests for the scripting host API.
//
// Run with GENERATE_GOLDEN=1 to create/update golden files:
//   GENERATE_GOLDEN=1 cargo test --test scripting

mod common;

use common::*;
use image::RgbaImage;
use paintfe::ops::scripting::{ScriptError, execute_script_sync};

/// Run a script on a 64×64 gradient and return the result as RgbaImage.
fn run_script(source: &str) -> Result<(RgbaImage, Vec<String>), ScriptError> {
    let img = create_test_gradient(64, 64);
    let pixels = img.as_raw().clone();
    let (out, w, h, console, _ops) = execute_script_sync(source, pixels, 64, 64, None)?;
    let result = RgbaImage::from_raw(w, h, out).expect("invalid pixel data from script");
    Ok((result, console))
}

// =============================================================================
// Canvas API
// =============================================================================

#[test]
fn script_width_height() {
    let (_, console) = run_script(
        r#"
        let w = width();
        let h = height();
        print_line(`${w}x${h}`);
        "#,
    )
    .unwrap();
    assert_eq!(console.last().unwrap(), "64x64");
}

// =============================================================================
// Pixel API — get/set
// =============================================================================

#[test]
fn script_set_pixel() {
    let (result, _) = run_script(
        r#"
        set_pixel(0, 0, 255, 0, 0, 255);
        set_pixel(1, 0, 0, 255, 0, 128);
        "#,
    )
    .unwrap();
    let p0 = result.get_pixel(0, 0).0;
    assert_eq!(p0, [255, 0, 0, 255]);
    let p1 = result.get_pixel(1, 0).0;
    assert_eq!(p1, [0, 255, 0, 128]);
}

#[test]
fn script_get_pixel_roundtrip() {
    let (result, _) = run_script(
        r#"
        // Read top-left pixel, then write it back shifted by one
        let r = get_r(0, 0);
        let g = get_g(0, 0);
        let b = get_b(0, 0);
        let a = get_a(0, 0);
        set_pixel(1, 1, r, g, b, a);
        "#,
    )
    .unwrap();
    let orig_00 = create_test_gradient(64, 64).get_pixel(0, 0).0;
    let copied = result.get_pixel(1, 1).0;
    assert_eq!(
        orig_00, copied,
        "get/set pixel roundtrip should preserve values"
    );
}

// =============================================================================
// Bulk iteration
// =============================================================================

#[test]
fn script_for_each_pixel_invert() {
    let (result, _) = run_script(
        r#"
        for_each_pixel(|x, y, r, g, b, a| {
            [255 - r, 255 - g, 255 - b, a]
        });
        "#,
    )
    .unwrap();
    assert_golden("scripting", "for_each_pixel_invert", &result);
}

#[test]
fn script_map_channels() {
    let (result, _) = run_script(
        r#"
        map_channels(|r, g, b, a| {
            [255 - r, 255 - g, 255 - b, a]
        });
        "#,
    )
    .unwrap();
    // map_channels invert should match for_each_pixel invert
    assert_golden("scripting", "map_channels_invert", &result);
}

// =============================================================================
// Effect API
// =============================================================================

#[test]
fn script_apply_blur() {
    let (result, _) = run_script("apply_blur(2.0);").unwrap();
    assert_golden("scripting", "apply_blur", &result);
}

#[test]
fn script_apply_invert() {
    let (result, _) = run_script("apply_invert();").unwrap();
    assert_golden("scripting", "apply_invert", &result);
}

#[test]
fn script_apply_sepia() {
    let (result, _) = run_script("apply_sepia();").unwrap();
    assert_golden("scripting", "apply_sepia", &result);
}

#[test]
fn script_apply_desaturate() {
    let (result, _) = run_script("apply_desaturate();").unwrap();
    assert_golden("scripting", "apply_desaturate", &result);
}

#[test]
fn script_apply_brightness_contrast() {
    let (result, _) = run_script("apply_brightness_contrast(20.0, 10.0);").unwrap();
    assert_golden("scripting", "apply_brightness_contrast", &result);
}

#[test]
fn script_apply_pixelate() {
    let (result, _) = run_script("apply_pixelate(4);").unwrap();
    assert_golden("scripting", "apply_pixelate", &result);
}

// =============================================================================
// Transform API (layer-only)
// =============================================================================

#[test]
fn script_flip_horizontal() {
    let (result, _) = run_script("flip_horizontal();").unwrap();
    assert_golden("scripting", "flip_horizontal", &result);
}

#[test]
fn script_flip_vertical() {
    let (result, _) = run_script("flip_vertical();").unwrap();
    assert_golden("scripting", "flip_vertical", &result);
}

#[test]
fn script_flip_roundtrip() {
    let (result, _) = run_script(
        r#"
        flip_horizontal();
        flip_horizontal();
        "#,
    )
    .unwrap();
    let original = create_test_gradient(64, 64);
    assert_eq!(result, original, "script flip × 2 should be identity");
}

// =============================================================================
// Utility API
// =============================================================================

#[test]
fn script_print() {
    let (_, console) = run_script(
        r#"
        print_line("hello world");
        print_line("second line");
        "#,
    )
    .unwrap();
    assert!(console.iter().any(|l| l.contains("hello world")));
    assert!(console.iter().any(|l| l.contains("second line")));
}

#[test]
fn script_math_functions() {
    let (_, console) = run_script(
        r#"
        let v = clamp(300, 0, 255);
        print_line(`${v}`);
        "#,
    )
    .unwrap();
    assert_eq!(console.last().unwrap(), "255");
}

// =============================================================================
// Error handling
// =============================================================================

#[test]
fn script_syntax_error() {
    let err = run_script("let x = ;").unwrap_err();
    assert!(
        err.message.len() > 0,
        "syntax error should produce an error message"
    );
}

#[test]
fn script_runtime_error() {
    let err = run_script("let x = 1 / 0;").unwrap_err();
    assert!(
        err.message.len() > 0,
        "division by zero should produce a runtime error"
    );
}

// =============================================================================
// Equivalence: script invert ≡ native invert
// =============================================================================

#[test]
fn script_invert_matches_native() {
    let (script_result, _) = run_script("apply_invert();").unwrap();
    let img = create_test_gradient(64, 64);
    let native_result = apply_direct_and_extract_adj(&img, |s| {
        paintfe::ops::adjustments::invert_colors(s, 0);
    });
    assert_eq!(
        script_result, native_result,
        "scripted invert should be identical to native invert"
    );
}

/// Helper (identical to visual_adjustments pattern but inlined here).
fn apply_direct_and_extract_adj(
    img: &RgbaImage,
    f: impl FnOnce(&mut paintfe::canvas::CanvasState),
) -> RgbaImage {
    let mut state = canvas_from_image(img);
    f(&mut state);
    extract_layer(&state, 0)
}

// =============================================================================
// Selection API (Strategy 2 — Rhai scripting bridge)
// =============================================================================

#[test]
fn script_select_rect_limits_effect() {
    // Select a rectangle and fill only that area
    let (result, _) = run_script(
        r#"
        select_rect(10, 10, 30, 30);
        fill_selected(255, 0, 0, 255);
        "#,
    )
    .unwrap();
    // Inside the rect should be red
    let inside = result.get_pixel(20, 20);
    assert_eq!(inside[0], 255, "inside rect R should be 255");
    assert_eq!(inside[1], 0, "inside rect G should be 0");
    assert_eq!(inside[2], 0, "inside rect B should be 0");
    // Outside the rect should be the original gradient pixel
    let outside = result.get_pixel(5, 5);
    assert_ne!(outside[0], 255, "outside rect should not be pure red");
}

#[test]
fn script_select_ellipse_limits_effect() {
    let (result, _) = run_script(
        r#"
        select_ellipse(32.0, 32.0, 15.0, 15.0);
        fill_selected(255, 0, 255, 255);
        "#,
    )
    .unwrap();
    // Center should be magenta (255, 0, 255)
    let center = result.get_pixel(32, 32);
    assert_eq!(center[0], 255, "center R should be 255");
    assert_eq!(center[1], 0, "center G should be 0");
    assert_eq!(center[2], 255, "center B should be 255");
    // Corner (0,0) should remain the original gradient value
    // Original gradient: (0,0) → r=0, g=255, b=0
    let corner = result.get_pixel(0, 0);
    assert_eq!(
        corner[0], 0,
        "corner R should be 0 (unchanged from gradient)"
    );
    assert_eq!(
        corner[1], 255,
        "corner G should be 255 (unchanged from gradient)"
    );
}

#[test]
fn script_clear_selection() {
    let (result, _) = run_script(
        r#"
        select_rect(0, 0, 10, 10);
        clear_selection();
        // After clearing, fill_selected should affect everything
        fill_selected(0, 0, 255, 255);
        "#,
    )
    .unwrap();
    // Every pixel should be blue after clearing selection and filling
    let corner = result.get_pixel(50, 50);
    assert_eq!(corner[2], 255, "corner B should be 255 after clear+fill");
}

#[test]
fn script_has_selection() {
    let (_, console) = run_script(
        r#"
        print_line("before: " + has_selection());
        select_rect(0, 0, 10, 10);
        print_line("after: " + has_selection());
        clear_selection();
        print_line("cleared: " + has_selection());
        "#,
    )
    .unwrap();
    assert!(console.iter().any(|l| l.contains("before: false")));
    assert!(console.iter().any(|l| l.contains("after: true")));
    assert!(console.iter().any(|l| l.contains("cleared: false")));
}

#[test]
fn script_invert_selection() {
    let (result, _) = run_script(
        r#"
        select_rect(10, 10, 54, 54);
        invert_selection();
        // Now the border region (outside the original rect) is selected
        fill_selected(255, 0, 255, 255);
        "#,
    )
    .unwrap();
    // The border area should be magenta
    let border = result.get_pixel(0, 0);
    assert_eq!(border[0], 255, "border R should be 255");
    assert_eq!(border[2], 255, "border B should be 255");
    // The inside of the original rect should be unchanged (not magenta)
    let inside = result.get_pixel(32, 32);
    assert_ne!(
        (inside[0], inside[2]),
        (255, 255),
        "inside should not be magenta"
    );
}

#[test]
fn script_delete_selected() {
    let (result, _) = run_script(
        r#"
        select_rect(20, 20, 44, 44);
        delete_selected();
        "#,
    )
    .unwrap();
    // Selected region should be transparent
    let inside = result.get_pixel(32, 32);
    assert_eq!(inside[3], 0, "deleted pixel should be transparent");
    // Outside should be untouched
    let outside = result.get_pixel(5, 5);
    assert!(outside[3] > 0, "outside pixel should still be opaque");
}

#[test]
fn script_select_rect_then_apply_effect() {
    // Verify that the Rhai is_selected check works with selection for effects
    let (result, _) = run_script(
        r#"
        select_rect(0, 0, 32, 64);
        // Manually invert only selected pixels
        for_each_pixel(|x, y, r, g, b, a| {
            if is_selected(x, y) {
                [255 - r, 255 - g, 255 - b, a]
            } else {
                [r, g, b, a]
            }
        });
        "#,
    )
    .unwrap();
    // Left side should be inverted, right side should be original gradient
    let left = result.get_pixel(5, 32);
    let right = result.get_pixel(50, 32);
    // The gradient at x=5 has R channel ~ 5*255/64 ~ 20, inverted = ~235
    assert!(
        left[0] > 200,
        "left R should be high (inverted) got {}",
        left[0]
    );
    // Right side should be low-ish (original gradient, x=50)
    assert!(right[0] > 100, "right R should be moderate (original)");
}
