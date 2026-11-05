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

#[test]
#[ignore = "manual pixel-grid CPU and tessellation profile"]
fn profile_pixel_grid_frame_cost() {
    use std::time::Instant;
    let mut canvas = Canvas::new_without_state();
    let state = CanvasState::new(800, 600);
    for size in [Vec2::new(1280., 720.), Vec2::new(1920., 1080.)] {
        for dashed in [true, false] {
            for zoom in [6., 8., 12.] {
                let ctx = egui::Context::default();
                canvas.zoom = zoom;
                let settings = crate::config::settings::AppSettings {
                    pixel_grid_dashed: dashed,
                    ..Default::default()
                };
                let viewport = Rect::from_min_size(Pos2::ZERO, size);
                let image_rect =
                    Rect::from_center_size(viewport.center(), Vec2::new(800., 600.) * zoom);
                let mut build = Vec::new();
                let mut tess = Vec::new();
                let mut counts = (0, 0, 0);
                for iteration in 0..9 {
                    let start = Instant::now();
                    let output = ctx.run_ui(
                        egui::RawInput {
                            screen_rect: Some(viewport),
                            ..Default::default()
                        },
                        |ui| {
                            if zoom >= 8. {
                                canvas.draw_pixel_grid(
                                    ui.painter(),
                                    image_rect,
                                    &state,
                                    viewport,
                                    &settings,
                                );
                            }
                        },
                    );
                    let build_ms = start.elapsed().as_secs_f64() * 1000.;
                    let shape_count = output.shapes.len();
                    let start = Instant::now();
                    let jobs = ctx.tessellate(output.shapes, output.pixels_per_point);
                    let tess_ms = start.elapsed().as_secs_f64() * 1000.;
                    let (mut vertices, mut indices) = (0, 0);
                    for job in jobs {
                        if let egui::epaint::Primitive::Mesh(mesh) = job.primitive {
                            vertices += mesh.vertices.len();
                            indices += mesh.indices.len();
                        }
                    }
                    counts = (shape_count, vertices, indices);
                    if iteration >= 3 {
                        build.push(build_ms);
                        tess.push(tess_ms);
                    }
                }
                build.sort_by(f64::total_cmp);
                tess.sort_by(f64::total_cmp);
                println!(
                    "{}",
                    serde_json::json!({"viewport":[size.x,size.y],"zoom":zoom,"dashed":dashed,"build_median_ms":build[build.len()/2],"tessellate_median_ms":tess[tess.len()/2],"shapes":counts.0,"vertices":counts.1,"indices":counts.2})
                );
            }
        }
    }
}

#[test]
fn committed_stroke_overlay_uses_final_pixels_and_replaces_the_presented_region() {
    let ctx = egui::Context::default();
    let mut canvas = Canvas::new_without_state();
    canvas.checkerboard_cached_size = (1, 1);
    canvas.checkerboard_texture = Some(ctx.load_texture(
        "opaque-test-checker",
        ColorImage::filled([2, 2], Color32::GRAY),
        TextureOptions::NEAREST,
    ));
    for &mode in BlendMode::all() {
        for alpha in [0, 96, 255] {
            let mut state = CanvasState::new(8, 8);
            let before = Rgba([90, 130, 170, alpha]);
            let stroke = Rgba([210, 40, 80, 128]);
            let committed = CanvasState::blend_pixel_static(before, stroke, mode, 1.);
            state.layers[0].pixels =
                TiledImage::from_rgba_image(&RgbaImage::from_pixel(8, 8, committed));
            state.preview_layer = Some(TiledImage::from_rgba_image(&RgbaImage::from_pixel(
                8, 8, stroke,
            )));
            state.preview_blend_mode = mode;
            let bounds = Rect::from_min_size(Pos2::ZERO, Vec2::splat(8.));
            state.mark_preview_changed_rect(bounds);
            state.defer_preview_clear();
            let expected = state.composite_committed_downscaled(bounds, 1).0;
            assert_eq!(state.composite_partial(bounds).0.pixels, expected.pixels);
            assert_eq!(
                state.composite_partial_downscaled(bounds, 1).0.pixels,
                expected.pixels
            );
            let pixel = *state.composite().get_pixel(4, 4);
            let actual = Color32::from_rgba_unmultiplied(pixel[0], pixel[1], pixel[2], pixel[3]);
            assert!(
                actual
                    .to_array()
                    .iter()
                    .zip(expected.pixels[36].to_array())
                    .all(|(a, b)| (*a as i16 - b as i16).abs() <= 1)
            );
            let output = ctx.run_ui(egui::RawInput::default(), |ui| {
                canvas.update_preview_overlay(
                    ui,
                    &mut state,
                    &crate::config::settings::AppSettings::default(),
                    Some(false),
                );
                canvas.paint_preview_texture(
                    ui.painter(),
                    bounds,
                    bounds,
                    &state,
                    Color32::WHITE,
                    true,
                );
            });
            assert!(
                state.preview_texture_replaces_composite,
                "{mode:?} alpha {alpha}"
            );
            let texture = state.preview_texture_cache.as_ref().unwrap().id();
            let uploaded = output
                .textures_delta
                .set
                .iter()
                .find(|(id, _)| *id == texture)
                .unwrap();
            let egui::ImageData::Color(image) = &uploaded.1.image;
            assert_eq!(image.pixels, expected.pixels, "{mode:?} alpha {alpha}");

            let textures: Vec<_> = output
                .shapes
                .iter()
                .filter_map(|shape| match &shape.shape {
                    egui::epaint::Shape::Mesh(mesh) => Some(mesh.texture_id),
                    _ => None,
                })
                .collect();
            let checker = canvas.checkerboard_texture.as_ref().unwrap().id();
            assert!(
                textures.windows(2).any(|ids| ids == [checker, texture]),
                "{mode:?} {alpha}: full composite must replace the base over an opaque checker"
            );
            state.clear_preview_state();
            assert!(state.preview_texture_cache.is_none());
        }
    }
}
