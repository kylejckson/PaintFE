//! Restart persistence is tested in fresh processes with isolated storage.
#![cfg(not(target_arch = "wasm32"))]
use std::{collections::BTreeMap, path::Path, process::Command};

#[test]
fn label_worker() {
    let Ok(stage) = std::env::var("PAINTFE_LABEL_TEST_STAGE") else {
        return;
    };
    paintfe::services::storage::initialize();
    paintfe::i18n::init();
    if stage == "write" {
        paintfe::i18n::save_label_overrides(
            "en",
            BTreeMap::from([
                ("menu.view".into(), "Panels".into()),
                ("keybind_category.view".into(), "Panel shortcuts".into()),
            ]),
        )
        .unwrap();
        paintfe::i18n::save_label_overrides(
            "de",
            BTreeMap::from([("menu.view".into(), "Meine Bereiche".into())]),
        )
        .unwrap();
        assert!(paintfe::i18n::save_label_overrides("../../outside", BTreeMap::new()).is_err());
        assert!(paintfe::i18n::parse_label_overrides("en", r#"{"unknown.key":"no"}"#).is_err());
    } else {
        assert!(paintfe::i18n::load_label_overrides().is_empty());
        assert_eq!(paintfe::i18n::translate("menu.view"), "Panels");
        assert_eq!(paintfe::i18n::translate_english("View"), "Panel shortcuts");
        paintfe::i18n::set_language("de");
        assert_eq!(paintfe::i18n::translate("menu.view"), "Meine Bereiche");
        paintfe::i18n::set_language("fr");
        assert_ne!(paintfe::i18n::translate("menu.view"), "Panels");
        paintfe::i18n::save_label_overrides("en", BTreeMap::new()).unwrap();
        paintfe::i18n::set_language("en");
        assert_eq!(paintfe::i18n::translate("menu.view"), "View");
    }
}

fn child(root: &Path, stage: &str) {
    let status = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "label_worker", "--nocapture"])
        .env("PAINTFE_TEST_STORAGE_DIR", root)
        .env("PAINTFE_LABEL_TEST_STAGE", stage)
        .status()
        .unwrap();
    assert!(status.success(), "{stage}");
}

#[test]
fn customizations_survive_restart_and_reset_without_affecting_other_languages() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target/personal-label-tests")
        .join(uuid::Uuid::new_v4().to_string());
    std::fs::create_dir_all(&root).unwrap();
    child(&root, "write");
    child(&root, "read");
    std::fs::remove_dir_all(root).unwrap();
}
