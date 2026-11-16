use super::history_tests::{app, context};
use super::*;
fn args(a: &PaintFEApp) -> serde_json::Value {
    let p = &a.projects[0];
    serde_json::json!({"document":p.id.to_string(),"layer":p.canvas_state.layers[0].runtime_id.to_string(),"revision":p.edit_revision})
}
fn call(
    a: &mut PaintFEApp,
    name: &str,
    args: serde_json::Value,
    s: &mut automation_state::Session,
) -> Result<serde_json::Value, String> {
    a.automation_call(
        &context(),
        &serde_json::json!({"name":name,"arguments":args}),
        s,
    )
}
#[test]
fn automation_is_off_by_default_and_theme_cannot_enable_it() {
    let mut settings = AppSettings::default();
    assert!(!settings.automation.enabled);
    settings.import_theme_from_string("automation={\"enabled\":true,\"editing\":true,\"scripts\":true}\nglass={\"enabled\":true}\n");
    assert!(!settings.automation.enabled);
    assert!(!settings.automation.editing);
    assert!(settings.glass.enabled);
}
#[test]
fn agent_editing_and_scripts_need_separate_permissions() {
    let mut a = app();
    let mut s = automation_state::Session::default();
    let mut input = args(&a);
    input["source"] = serde_json::json!("apply_invert();");
    assert!(
        call(&mut a, "script", input.clone(), &mut s)
            .unwrap_err()
            .contains("permission")
    );
    a.settings.automation.editing = true;
    assert!(
        call(&mut a, "script", input, &mut s)
            .unwrap_err()
            .contains("disabled")
    );
    assert_eq!(a.projects[0].history.undo_count(), 0);
}
#[test]
fn stale_agent_edits_are_rejected_without_history() {
    let mut a = app();
    a.settings.automation.editing = true;
    let mut input = args(&a);
    input["revision"] = serde_json::json!(999);
    assert!(
        call(
            &mut a,
            "deselect",
            input,
            &mut automation_state::Session::default()
        )
        .is_err()
    );
    assert_eq!(a.projects[0].history.undo_count(), 0);
}
#[test]
fn agent_selection_has_one_undo_and_clips_invalid_geometry() {
    let mut a = app();
    a.settings.automation.editing = true;
    let mut input = args(&a);
    for (k, v) in [("x", 2), ("y", 3), ("width", 5), ("height", 4)] {
        input[k] = serde_json::json!(v);
    }
    let mut s = automation_state::Session::default();
    call(&mut a, "select_rectangle", input, &mut s).unwrap();
    assert_eq!(a.projects[0].history.undo_count(), 1);
    assert!(a.projects[0].canvas_state.has_selection());
    let input = args(&a);
    call(&mut a, "undo", input, &mut s).unwrap();
    assert!(!a.projects[0].canvas_state.has_selection());
    let mut input = args(&a);
    input["x"] = serde_json::json!(31);
    input["y"] = serde_json::json!(0);
    input["width"] = serde_json::json!(2);
    input["height"] = serde_json::json!(2);
    assert!(call(&mut a, "select_rectangle", input, &mut s).is_err());
}
#[test]
fn agent_preview_is_bounded_and_does_not_expose_paths() {
    let mut a = app();
    let mut s = automation_state::Session::default();
    let docs = call(&mut a, "documents", serde_json::json!({}), &mut s).unwrap();
    assert!(!docs.to_string().contains("path"));
    let input = args(&a);
    let preview = call(&mut a, "preview", input, &mut s).unwrap();
    assert_eq!(preview["content"][0]["mimeType"], "image/png");
    assert_eq!(a.projects[0].history.undo_count(), 0);
}
#[test]
fn agent_brush_and_pencil_use_selection_and_respect_cancel() {
    use crate::ui::panels::tools::ToolsPanel;
    let input = vec![0; 16 * 16 * 4];
    let mut mask = image::GrayImage::new(16, 16);
    for y in 0..16 {
        for x in 0..8 {
            mask.put_pixel(x, y, image::Luma([255]));
        }
    }
    let cancel = std::sync::atomic::AtomicBool::new(false);
    for pencil in [false, true] {
        let out = ToolsPanel::render_automation_stroke(
            16,
            16,
            &input,
            Some(mask.clone()),
            &[[1., 8., 1.], [14., 8., 1.]],
            [255, 20, 30, 255],
            3.,
            pencil,
            false,
            &cancel,
        )
        .unwrap();
        assert!(
            out.chunks_exact(4)
                .enumerate()
                .any(|(i, p)| i % 16 < 8 && p[3] > 0)
        );
        assert!(
            out.chunks_exact(4)
                .enumerate()
                .all(|(i, p)| i % 16 < 8 || p[3] == 0)
        );
    }
    cancel.store(true, std::sync::atomic::Ordering::Relaxed);
    assert!(
        ToolsPanel::render_automation_stroke(
            16,
            16,
            &input,
            None,
            &[[1., 1., 1.]],
            [255; 4],
            2.,
            false,
            false,
            &cancel
        )
        .is_err()
    );
}
#[test]
fn agent_stroke_rejects_unbounded_points_before_starting_worker() {
    let mut a = app();
    a.settings.automation.editing = true;
    let mut input = args(&a);
    input["points"] = serde_json::json!([[1000000., 1.]]);
    input["size"] = serde_json::json!(2);
    input["colour"] = serde_json::json!([0, 0, 0, 255]);
    let mut s = automation_state::Session::default();
    assert!(call(&mut a, "stroke", input, &mut s).is_err());
    assert!(s.pending.is_none());
}
#[test]
fn completed_agent_edit_is_one_undo_step_and_stale_results_are_discarded() {
    for stale in [false, true] {
        let mut a = app();
        a.settings.automation.enabled = true;
        a.settings.automation.editing = true;
        let ctx = context();
        let p = &a.projects[0];
        let before = p.canvas_state.layers[0]
            .pixels
            .extract_region_rgba(0, 0, 32, 24);
        let after = vec![42; before.len()];
        let (tx, rx) = std::sync::mpsc::channel();
        tx.send(Ok(after.clone())).unwrap();
        let path =
            std::env::temp_dir().join(format!("paintfe-agent-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir(&path).unwrap();
        let state = automation_state::state(&ctx);
        {
            let mut s = state.lock().unwrap();
            s.path = Some(path.clone());
            s.token = "test".into();
            s.pending = Some(automation_state::Pending {
                document: p.id,
                layer: p.canvas_state.layers[0].runtime_id,
                revision: p.edit_revision + u64::from(stale),
                action: HistoryAction::Brush,
                receiver: rx,
                cancel: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            });
        }
        a.update_automation(&ctx);
        let p = &mut a.projects[0];
        assert_eq!(p.history.undo_count(), usize::from(!stale));
        let pixels = p.canvas_state.layers[0]
            .pixels
            .extract_region_rgba(0, 0, 32, 24);
        assert_eq!(pixels, if stale { before.clone() } else { after });
        if !stale {
            p.history.undo(&mut p.canvas_state);
            assert_eq!(
                p.canvas_state.layers[0]
                    .pixels
                    .extract_region_rgba(0, 0, 32, 24),
                before
            );
        }
        let _ = std::fs::remove_dir_all(path);
    }
}
