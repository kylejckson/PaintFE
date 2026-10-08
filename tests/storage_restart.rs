//! Real process restarts exercise the startup resolver without user-profile I/O.
use paintfe::{assets::AppSettings, services::storage};
use std::path::{Path, PathBuf};

fn child(root: &Path, phase: &str) {
    let result = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "storage_child", "--nocapture"])
        .env("PAINTFE_TEST_STORAGE_DIR", root)
        .env("PAINTFE_STORAGE_TEST_PHASE", phase)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
}

struct Fixture(PathBuf);
impl Fixture {
    fn new() -> Self {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target/storage-restarts")
            .join(stamp.to_string());
        std::fs::create_dir_all(&root).unwrap();
        Self(root)
    }
    fn write(&self, relative: &str, content: &str) {
        let path = self.0.join(relative);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, content).unwrap();
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn storage_child() {
    let Ok(phase) = std::env::var("PAINTFE_STORAGE_TEST_PHASE") else {
        return;
    };
    let active = storage::initialize();
    let mut settings = AppSettings::load();
    match phase.as_str() {
        "default" => {
            assert!(active.portable);
            assert!(!settings.middle_click_close_tabs);
            assert!(active.data_dir().join("scripts/kept.rhai").is_file());
            settings.save();
            storage::request_mode(false, true).unwrap();
            assert!(
                active.portable,
                "changing preferences must wait for restart"
            );
            settings.middle_click_close_tabs = true;
            settings.save();
        }
        "profile" => {
            assert!(!active.portable);
            assert!(
                settings.middle_click_close_tabs,
                "edits after the mode change must migrate"
            );
            assert!(active.data_dir().join("scripts/kept.rhai").is_file());
            storage::request_mode(true, true).unwrap();
            settings.middle_click_close_tabs = false;
            settings.save();
        }
        "portable" => {
            assert!(active.portable);
            assert!(!settings.middle_click_close_tabs);
            assert_eq!(
                Path::new(&settings.birefnet_model_path),
                active.executable_dir.join("models/fixture.onnx")
            );
            assert!(Path::new(&settings.birefnet_model_path).is_file());
        }
        "fallback" => {
            assert!(!active.portable);
            assert!(active.warning.is_some());
            assert!(!settings.middle_click_close_tabs);
        }
        _ => panic!("Unknown test phase"),
    }
}

#[test]
fn defaults_migrate_legacy_data_and_mode_changes_survive_restart_and_relocation() {
    let fixture = Fixture::new();
    let model = fixture.0.join("portable/models/fixture.onnx");
    fixture.write("portable/models/fixture.onnx", "fixture");
    fixture.write(
        "profile/paintfe_settings.cfg",
        &format!(
            "middle_click_close_tabs=false\nbirefnet_model_path={}\n",
            model.display()
        ),
    );
    fixture.write("profile/scripts/kept.rhai", "// preserved");
    child(&fixture.0, "default");
    child(&fixture.0, "profile");
    child(&fixture.0, "portable");
    assert!(fixture.0.join("profile/scripts/kept.rhai").exists());
    let moved = fixture.0.join("relocated");
    std::fs::create_dir(&moved).unwrap();
    std::fs::rename(fixture.0.join("portable"), moved.join("portable")).unwrap();
    child(&moved, "portable");
}

#[test]
fn unwritable_portable_location_falls_back_with_a_visible_reason() {
    let fixture = Fixture::new();
    fixture.write("portable", "this path cannot be a directory");
    fixture.write(
        "profile/paintfe_settings.cfg",
        "middle_click_close_tabs=false\n",
    );
    child(&fixture.0, "fallback");
    assert_eq!(
        std::fs::read_to_string(fixture.0.join("portable")).unwrap(),
        "this path cannot be a directory"
    );
}
