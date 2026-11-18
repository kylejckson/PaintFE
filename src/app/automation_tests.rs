use super::history_tests::{app, context};
use super::*;

#[test]
#[ignore = "Requires PAINTFE_MCP_TEST_EXE and PAINTFE_TEST_PYTHON; runs the real Python client and bridge"]
fn python_client_draws_through_real_bridge() {
    use std::process::{Command, Stdio};
    use std::time::{Duration, Instant};
    let exe =
        std::env::var_os("PAINTFE_MCP_TEST_EXE").expect("Set the built PaintFE executable path");
    let python = std::env::var_os("PAINTFE_TEST_PYTHON").expect("Set the Python executable path");
    assert!(
        std::env::var_os("PAINTFE_TEST_STORAGE_DIR").is_some(),
        "Set a disposable settings profile"
    );
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target/mcp-app-tests")
        .join(uuid::Uuid::new_v4().to_string());
    let directory = root.join("automation");
    std::fs::create_dir_all(&directory).unwrap();
    let ctx = context();
    let mut a = app();
    a.settings.automation.enabled = true;
    a.settings.automation.editing = true;
    let mut setup = args(&a);
    setup["name"] = serde_json::json!("Existing top layer");
    call(
        &mut a,
        "new_layer",
        setup,
        &mut automation_state::Session::default(),
    )
    .unwrap();
    assert_eq!(a.projects[0].canvas_state.active_layer_index, 1);
    let original = a.projects[0].canvas_state.layers[0].pixels.to_rgba_image();
    let baseline = a.projects[0].history.undo_count();
    let state = automation_state::state(&ctx);
    let token = uuid::Uuid::new_v4().to_string();
    std::fs::write(
        directory.join("session.json"),
        serde_json::json!({"token":token}).to_string(),
    )
    .unwrap();
    {
        let mut s = state.lock().unwrap();
        s.path = Some(directory);
        s.token = token;
    }
    let clock = Instant::now();
    for mode in ["inspect", "draw", "brush", "denied"] {
        a.settings.automation.editing = mode != "denied";
        let mut command = Command::new(&python);
        if mode == "brush" {
            command
                .args([
                    "-c",
                    r#"
import sys
sys.path.insert(0, "examples")
from mcp_client import Client, current_document, target
with Client(sys.argv[1]) as c:
    c.connect()
    document = current_document(c)
    c.tool("stroke", {**target(document), "points": [[6, 12], [24, 12]],
                      "colour": [255, 0, 255, 255], "size": 9, "pencil": False})
    c.wait_for_job()
"#,
                ])
                .arg(&exe);
        } else {
            command
                .arg("examples/mcp_client.py")
                .arg("--paintfe")
                .arg(&exe);
            if mode != "inspect" {
                command.arg("--draw");
            }
        }
        command
            .current_dir(env!("CARGO_MANIFEST_DIR"))
            .env("PAINTFE_TEST_STORAGE_DIR", &root)
            .env("PYTHONIOENCODING", "utf-8")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let mut child = command.spawn().unwrap();
        let deadline = Instant::now() + Duration::from_secs(45);
        while child.try_wait().unwrap().is_none() {
            if Instant::now() > deadline {
                let _ = child.kill();
                panic!(
                    "Python example timed out: {:?}",
                    child.wait_with_output().unwrap()
                );
            }
            let _ = ctx.run_ui(
                egui::RawInput {
                    time: Some(clock.elapsed().as_secs_f64()),
                    ..Default::default()
                },
                |ui| a.update_automation(ui.ctx()),
            );
            std::thread::sleep(Duration::from_millis(10));
        }
        let output = child.wait_with_output().unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert_eq!(
            output.status.success(),
            mode != "denied",
            "{mode}: {stdout}\n{stderr}"
        );
        if mode == "inspect" {
            assert!(stdout.contains("Read-only check complete"));
            assert_eq!(a.projects[0].history.undo_count(), baseline);
        } else if mode == "draw" {
            assert!(stdout.contains("Three dots added"));
            assert_eq!(a.projects[0].history.undo_count(), baseline + 4);
            assert_eq!(a.projects[0].canvas_state.layers.len(), 3);
            let pixels = a.projects[0].canvas_state.layers[2].pixels.to_rgba_image();
            for (x, y, colour) in [
                (6, 4, [255, 0, 0, 255]),
                (16, 12, [0, 180, 0, 255]),
                (25, 19, [0, 80, 255, 255]),
            ] {
                assert_eq!(pixels.get_pixel(x, y).0, colour);
            }
        } else if mode == "brush" {
            let p = &a.projects[0];
            assert_eq!(p.history.undo_count(), baseline + 5);
            let pixels = &p.canvas_state.layers[2].pixels;
            for y in 10..14 {
                for x in 10..20 {
                    assert_eq!(pixels.get_pixel(x, y).0, [255, 0, 255, 255]);
                }
            }
            assert_eq!(pixels.get_pixel(14, 4).0[3], 0);
        } else {
            assert!(stderr.contains("permission"), "{stderr}");
            assert_eq!(a.projects[0].history.undo_count(), baseline + 5);
        }
        assert_eq!(
            a.projects[0].canvas_state.layers[0].pixels.to_rgba_image(),
            original
        );
    }
    drop(a);
    drop(ctx);
    drop(state);
    std::fs::remove_dir_all(root).unwrap();
}
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
fn agent_brush_and_eraser_fill_the_interior_of_a_wide_stroke() {
    let cancel = std::sync::atomic::AtomicBool::new(false);
    let mut mask = image::GrayImage::from_pixel(256, 256, image::Luma([255]));
    mask.put_pixel(100, 40, image::Luma([0]));
    for erase in [false, true] {
        let background = if erase { 255 } else { 0 };
        let pixels = crate::ui::panels::tools::ToolsPanel::render_automation_stroke(
            256,
            256,
            &vec![background; 256 * 256 * 4],
            Some(mask.clone()),
            &[[60.0, 40.0, 1.0], [140.0, 40.0, 1.0]],
            [255, 0, 255, 255],
            15.0,
            false,
            erase,
            &cancel,
        )
        .unwrap();
        for y in 35..=44 {
            for x in 75..125 {
                let index = (y * 256 + x) * 4;
                let pixel = &pixels[index..index + 4];
                if (x, y) == (100, 40) {
                    assert_eq!(pixel[3], background);
                } else if erase {
                    assert_eq!(pixel[3], 0);
                } else {
                    assert_eq!(pixel, [255, 0, 255, 255]);
                }
            }
        }
        assert_eq!(pixels[(29 * 256 + 100) * 4 + 3], background);
    }
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
