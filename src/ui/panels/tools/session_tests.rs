use super::*;

#[test]
fn liquify_batched_path_preserves_turns_and_flushes_a_fast_release() {
    let mut state = LiquifyToolState::default();
    let samples = state.sample_frame_motion(&[(2., 2.), (12., 2.), (12., 12.)], 1., 0.1, true);
    assert!(
        samples
            .iter()
            .any(|(point, _)| point[0] > 2. && point[1] == 2.)
    );
    assert_eq!(samples.last().unwrap().0, [12., 12.]);
    let delta = samples.iter().fold([0., 0.], |sum, (_, delta)| {
        [sum[0] + delta[0], sum[1] + delta[1]]
    });
    assert!((delta[0] - 10.).abs() < 0.001 && (delta[1] - 10.).abs() < 0.001);
    let mut fast = LiquifyToolState::default();
    let samples = fast.sample_frame_motion(&[(2., 2.), (12., 12.)], 1., 0.001, true);
    assert_eq!(samples.last().unwrap(), &([12., 12.], [10., 10.]));
}

#[test]
fn liquify_sampling_is_independent_of_ui_frame_rate() {
    fn samples(fps: usize) -> Vec<([f32; 2], [f32; 2])> {
        let mut state = LiquifyToolState::default();
        let mut result = Vec::new();
        for i in 0..=fps {
            let time = i as f64 / fps as f64;
            let point = [10. + time as f32 * 30., 20.];
            result.extend(state.sample_frame_motion(
                &[(point[0], point[1])],
                time,
                1. / fps as f64,
                i == fps,
            ));
        }
        result
    }
    let reference = samples(60);
    for fps in [15, 30, 144] {
        let actual = samples(fps);
        assert_eq!(actual.len(), reference.len());
        for (a, b) in actual.iter().zip(&reference) {
            for c in 0..2 {
                assert!((a.0[c] - b.0[c]).abs() < 0.0001);
                assert!((a.1[c] - b.1[c]).abs() < 0.0001);
            }
        }
    }
}

#[test]
fn liquify_tile_history_restores_edges_and_branches() {
    let mut field = crate::ops::transform::DisplacementField::new(70, 45);
    let before = field.data.clone();
    let mut history = LiquifyAdjustmentHistory::default();
    history.begin();
    history.capture(&field, [60., 35.], 20.);
    field.apply_push(60., 35., 4., 2., 20., 0.5);
    history.finish(&field);
    let after = field.data.clone();
    assert_ne!(before, after);
    assert!(history.undo(&mut field));
    assert_eq!(field.data, before);
    assert!(history.redo(&mut field));
    assert_eq!(field.data, after);
    assert!(history.undo(&mut field));
    history.begin();
    history.capture(&field, [8., 8.], 4.);
    field.apply_expand(8., 8., 4., 1.);
    history.finish(&field);
    assert!(!history.can_redo());
}

#[test]
fn liquify_strength_is_linear_and_reconstruct_restores_the_center() {
    let mut weak = crate::ops::transform::DisplacementField::new(32, 32);
    let mut strong = weak.clone();
    weak.apply_push(16., 16., 4., 2., 8., 0.25);
    strong.apply_push(16., 16., 4., 2., 8., 0.5);
    for (a, b) in weak.data.iter().zip(&strong.data) {
        assert!((a * 2. - b).abs() < 0.00001);
    }
    strong.reconstruct(16., 16., 8., 1.);
    let index = (16 * 32 + 16) * 2;
    assert_eq!(&strong.data[index..index + 2], &[0., 0.]);
}

#[test]
fn reference_mesh_rejects_crossed_or_collapsed_cells() {
    let state = MeshWarpToolState {
        grid_cols: 1,
        grid_rows: 1,
        original_points: vec![[0., 0.], [10., 0.], [0., 10.], [10., 10.]],
        ..Default::default()
    };
    assert!(state.reference_move_valid(0, [1., 1.]));
    assert!(!state.reference_move_valid(0, [12., 12.]));
    assert!(!state.reference_move_valid(0, [10., 0.]));
}

#[test]
fn local_mesh_history_restores_reference_and_deformed_points() {
    let mut state = MeshWarpToolState {
        grid_cols: 1,
        grid_rows: 1,
        original_points: vec![[0., 0.], [10., 0.], [0., 10.], [10., 10.]],
        ..Default::default()
    };
    state.points = state.original_points.clone();
    let before = state.layout();
    state.drag_before = Some(before.clone());
    state.points[0] = [1., 1.];
    state.finish_drag();
    assert_eq!(state.undo_layouts, vec![before]);
    assert!(state.redo_layouts.is_empty());
}

#[test]
fn liquify_history_evicts_old_strokes_and_handles_one_oversized_stroke() {
    let mut field = crate::ops::transform::DisplacementField::new(64, 32);
    let mut history = LiquifyAdjustmentHistory {
        limit: 16 * 1024 + 128,
        ..Default::default()
    };
    for _ in 0..3 {
        history.begin();
        history.capture(&field, [12., 12.], 4.);
        history.capture(&field, [12., 12.], 4.);
        assert!(!history.limited);
        assert!(history.pending_bytes + history.stored_bytes <= history.limit);
        field.apply_push(12., 12., 1., 0., 4., 0.5);
        history.finish(&field);
        assert_eq!(history.undo.len(), 1);
    }
    history.begin();
    history.capture(&field, [31., 16.], 20.);
    assert!(history.limited);
    assert!(!history.can_undo());
    assert_eq!(history.pending_bytes + history.stored_bytes, 0);
}

#[test]
fn build_up_accumulates_beyond_the_normal_stroke_opacity_cap() {
    let mut alphas = Vec::new();
    for mode in [BrushMode::Normal, BrushMode::BuildUp] {
        let writer = BrushPixelWriter {
            mode,
            flow: 1.,
            opacity: 0.5,
        };
        let mut pixel = [0; 4];
        let mut coverage = 0;
        for _ in 0..12 {
            writer.write(
                &mut pixel,
                0,
                1.,
                1.,
                20,
                30,
                40,
                false,
                Some(&mut coverage),
            );
        }
        alphas.push(pixel[3]);
    }
    assert_eq!(alphas[0], 128);
    assert!(alphas[1] > 250);
}

#[test]
fn hard_circle_stamps_match_an_independent_pixel_center_disk_oracle() {
    for (width, height) in [(1, 1), (32, 32)] {
        for diameter in 1..=9 {
            for point in [(0.1, 0.1), (0.9, 0.9), (8.1, 8.1), (31.9, 16.2)] {
                let mut tools = ToolsPanel::default();
                tools.properties.size = diameter as f32;
                tools.properties.hardness = 1.;
                tools.properties.anti_aliased = false;
                tools.rebuild_brush_lut();
                let mut pixels = TiledImage::new(width, height);
                tools.draw_circle_no_dirty(
                    &mut pixels,
                    width,
                    height,
                    point,
                    false,
                    false,
                    [1., 0., 0., 1.],
                    [0., 0., 0., 1.],
                    None,
                );
                let offset = if (diameter as u32).is_multiple_of(2) {
                    0.
                } else {
                    0.5
                };
                let center = (point.0.floor() + offset, point.1.floor() + offset);
                for y in 0..height {
                    for x in 0..width {
                        let inside = (x as f32 + 0.5 - center.0).powi(2)
                            + (y as f32 + 0.5 - center.1).powi(2)
                            <= (diameter as f32 / 2.).powi(2);
                        assert_eq!(
                            pixels.get_pixel(x, y)[3],
                            if inside { 255 } else { 0 },
                            "{width}x{height}, diameter {diameter}, {point:?}, pixel {x},{y}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn mesh_reset_preserves_the_reference_layout_and_can_be_undone() {
    let mut tools = ToolsPanel::default();
    let state = &mut tools.mesh_warp_state;
    state.original_points = vec![[0., 0.], [10., 0.], [0., 10.], [9., 10.]];
    state.points = state.original_points.clone();
    state.points[3] = [8., 8.];
    let before = state.layout();
    state.commit_pending = true;
    tools.reset_mesh_deformation();
    let state = &tools.mesh_warp_state;
    assert_eq!(state.original_points, before.original);
    assert_eq!(state.points, state.original_points);
    assert_eq!(state.undo_layouts, vec![before]);
    assert!(!state.commit_pending);
}
