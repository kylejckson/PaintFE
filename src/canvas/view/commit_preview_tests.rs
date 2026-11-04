#[test]
fn committed_preview_handover_does_not_blend_the_same_stroke_twice() {
    let ctx = egui::Context::default();
    for &mode in BlendMode::all() {
        for alpha in [96, 255] {
            let mut state = CanvasState::new(8, 8);
            let before = Rgba([90, 130, 170, alpha]);
            let stroke = Rgba([210, 40, 80, 128]);
            state.layers[0].pixels =
                TiledImage::from_rgba_image(&RgbaImage::from_pixel(8, 8, before));
            state.preview_layer = Some(TiledImage::from_rgba_image(&RgbaImage::from_pixel(
                8, 8, stroke,
            )));
            state.preview_blend_mode = mode;
            let rect = Rect::from_min_size(Pos2::ZERO, Vec2::splat(8.));
            let expected = state.composite_partial(rect).0;
            let committed = CanvasState::blend_pixel_static(before, stroke, mode, 1.);
            state.layers[0].pixels =
                TiledImage::from_rgba_image(&RgbaImage::from_pixel(8, 8, committed));
            state.defer_preview_clear();
            state.commit_composite_flush_rect = Some(rect);
            state.composite_cpu_buffer = vec![Color32::TRANSPARENT; 64];
            state.composite_cache =
                Some(ctx.load_texture("handover", expected.clone(), TextureOptions::NEAREST));
            let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
                Canvas::flush_committed_preview_region(ui, &mut state, TextureOptions::NEAREST);
            });
            for (actual, expected) in state.composite_cpu_buffer.iter().zip(expected.pixels) {
                assert!(
                    actual
                        .to_array()
                        .iter()
                        .zip(expected.to_array())
                        .all(|(a, b)| (*a as i16 - b as i16).abs() <= 1),
                    "handover differs for {mode:?}, alpha {alpha}: {actual:?} vs {expected:?}"
                );
            }
        }
    }
}
