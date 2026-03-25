// =============================================================================
// Visual regression tests — Blend Modes
// =============================================================================
//
// Tests all 25 blend modes by compositing a gradient foreground over a
// checkerboard background, then comparing the flattened result against
// a golden PNG.
//
// Run with GENERATE_GOLDEN=1 to create/update golden files:
//   GENERATE_GOLDEN=1 cargo test --test visual_blend

mod common;

use common::*;
use image::{Rgba, RgbaImage};
use paintfe::canvas::{BlendMode, CanvasState, Layer, TiledImage};

/// Create a 2-layer test canvas: checkerboard background + translucent gradient foreground.
fn make_blend_test(mode: BlendMode) -> RgbaImage {
    let w = 64;
    let h = 64;

    // Background: checkerboard
    let bg_img = create_test_checkerboard(w, h);

    // Foreground: semi-transparent gradient (alpha varies across width)
    let mut fg_img = RgbaImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let r = ((x as f32 / w as f32) * 255.0) as u8;
            let g = ((y as f32 / h as f32) * 255.0) as u8;
            let b = 128u8;
            let a = (((x + y) as f32 / (w + h - 2) as f32) * 200.0 + 55.0) as u8;
            fg_img.put_pixel(x, y, Rgba([r, g, b, a]));
        }
    }

    let mut state = CanvasState::new(w, h);
    state.layers[0].pixels = TiledImage::from_rgba_image(&bg_img);

    let mut fg = Layer::new("Foreground".into(), w, h, Rgba([0, 0, 0, 0]));
    fg.blend_mode = mode;
    fg.pixels = TiledImage::from_rgba_image(&fg_img);
    state.layers.push(fg);

    state.composite()
}

// -- tests for all 25 blend modes ------------------------------------------------

macro_rules! blend_test {
    ($name:ident, $mode:expr) => {
        #[test]
        fn $name() {
            let result = make_blend_test($mode);
            assert_golden("blend", stringify!($name), &result);
        }
    };
}

blend_test!(normal, BlendMode::Normal);
blend_test!(multiply, BlendMode::Multiply);
blend_test!(screen, BlendMode::Screen);
blend_test!(additive, BlendMode::Additive);
blend_test!(reflect, BlendMode::Reflect);
blend_test!(glow, BlendMode::Glow);
blend_test!(color_burn, BlendMode::ColorBurn);
blend_test!(color_dodge, BlendMode::ColorDodge);
blend_test!(overlay, BlendMode::Overlay);
blend_test!(difference, BlendMode::Difference);
blend_test!(negation, BlendMode::Negation);
blend_test!(lighten, BlendMode::Lighten);
blend_test!(darken, BlendMode::Darken);
blend_test!(xor, BlendMode::Xor);
blend_test!(overwrite, BlendMode::Overwrite);
blend_test!(hard_light, BlendMode::HardLight);
blend_test!(soft_light, BlendMode::SoftLight);
blend_test!(exclusion, BlendMode::Exclusion);
blend_test!(subtract, BlendMode::Subtract);
blend_test!(divide, BlendMode::Divide);
blend_test!(linear_burn, BlendMode::LinearBurn);
blend_test!(vivid_light, BlendMode::VividLight);
blend_test!(linear_light, BlendMode::LinearLight);
blend_test!(pin_light, BlendMode::PinLight);
blend_test!(hard_mix, BlendMode::HardMix);

// -- Opacity test -----------------------------------------------------------------

#[test]
fn normal_half_opacity() {
    let w = 64;
    let h = 64;
    let bg_img = create_test_checkerboard(w, h);
    let fg_img = create_test_gradient(w, h);

    let mut state = CanvasState::new(w, h);
    state.layers[0].pixels = TiledImage::from_rgba_image(&bg_img);

    let mut fg = Layer::new("Foreground".into(), w, h, Rgba([0, 0, 0, 0]));
    fg.opacity = 0.5;
    fg.pixels = TiledImage::from_rgba_image(&fg_img);
    state.layers.push(fg);

    let result = state.composite();
    assert_golden("blend", "normal_half_opacity", &result);
}

// -- Hidden layer test -----------------------------------------------------------

#[test]
fn hidden_layer_invisible() {
    let w = 64;
    let h = 64;
    let bg_img = create_test_checkerboard(w, h);
    let fg_img = create_test_gradient(w, h);

    let mut state = CanvasState::new(w, h);
    state.layers[0].pixels = TiledImage::from_rgba_image(&bg_img);

    let mut fg = Layer::new("Hidden".into(), w, h, Rgba([0, 0, 0, 0]));
    fg.visible = false;
    fg.pixels = TiledImage::from_rgba_image(&fg_img);
    state.layers.push(fg);

    let result = state.composite();
    // Hiding the only overlay means the result equals the background
    let bg_only = {
        let mut s = CanvasState::new(w, h);
        s.layers[0].pixels = TiledImage::from_rgba_image(&bg_img);
        s.composite()
    };
    assert_eq!(
        result, bg_only,
        "hidden layer should not contribute to composite"
    );
}
