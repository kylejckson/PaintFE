//! Explicit desktop integration check: actual redraw pacing and clean shutdown.
#![cfg(not(target_arch = "wasm32"))]
use std::{
    path::PathBuf,
    process::{Command, Stdio},
    time::{Duration, Instant},
};

#[test]
#[ignore = "requires a visible desktop; starts two self-closing PaintFE review windows"]
fn native_frame_cap_and_shutdown() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target/frame-pacing-desktop")
        .join(uuid::Uuid::new_v4().to_string());
    for fps in [30, 60] {
        let dir = root.join(fps.to_string());
        std::fs::create_dir_all(&dir).unwrap();
        let exe = dir.join(if cfg!(windows) {
            "PaintFEFrameReview.exe"
        } else {
            "PaintFEFrameReview"
        });
        std::fs::copy(env!("CARGO_BIN_EXE_PaintFE"), &exe).unwrap();
        let config = format!(
            "language=en\npreferred_gpu=Auto\ncreate_canvas_on_startup=true\nconfirm_on_exit=false\nframe_rate_detect_screen=false\nframe_rate_limit={fps}\n"
        );
        for profile in [&dir, &dir.join("portable")] {
            std::fs::create_dir_all(profile).unwrap();
            std::fs::write(profile.join("paintfe_settings.cfg"), &config).unwrap();
            std::fs::write(profile.join("paintfe_storage.cfg"), "portable").unwrap();
        }
        let trace = dir.join("trace.jsonl");
        let log = std::fs::File::create(dir.join("run.log")).unwrap();
        let mut child = Command::new(&exe)
            .current_dir(&dir)
            .env("PAINTFE_UI_PROFILE", &trace)
            .env("PAINTFE_UI_PROFILE_DRIVE", "1")
            .env("PAINTFE_PROFILE_EXIT_AFTER_MS", "10000")
            .env("PAINTFE_TEST_STORAGE_DIR", &dir)
            .stdout(Stdio::null())
            .stderr(log)
            .spawn()
            .unwrap();
        let start = Instant::now();
        let status = loop {
            if let Some(status) = child.try_wait().unwrap() {
                break status;
            }
            if start.elapsed() > Duration::from_secs(35) {
                let _ = child.kill();
                let _ = child.wait();
                panic!(
                    "Paced editor did not shut down cleanly; logs: {}",
                    dir.display()
                );
            }
            std::thread::sleep(Duration::from_millis(30));
        };
        assert!(status.success(), "Editor failed: {}", dir.display());
        let mut gaps: Vec<f64> = std::fs::read_to_string(&trace)
            .unwrap()
            .lines()
            .filter_map(|line| {
                let value: serde_json::Value = serde_json::from_str(line).ok()?;
                (value.get("pass")?.as_u64()? == 0 && value.get("time_ms")?.as_f64()? > 3000.0)
                    .then(|| value.get("gap_ms")?.as_f64())
                    .flatten()
            })
            .filter(|gap| *gap > 0.0)
            .collect();
        assert!(
            gaps.len() >= 5,
            "Missing actual UI frames: {}",
            trace.display()
        );
        gaps.sort_by(f64::total_cmp);
        let median = gaps[gaps.len() / 2];
        assert!(
            median >= 1000.0 / fps as f64 * 0.9,
            "Cap {fps}: median gap {median} ms; {}",
            trace.display()
        );
        eprintln!(
            "FRAME_PACING fps={fps} median_gap_ms={median:.3} frames={} exit=success",
            gaps.len()
        );
    }
}
