//! Native file pickers have independent, restart-persistent directories.
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

#[derive(Clone, Copy, Debug)]
pub enum Purpose {
    Documents,
    Themes,
    Runtime,
    Model,
    Scripts,
    Resource,
}
impl Purpose {
    #[cfg(any(not(target_arch = "wasm32"), test))]
    fn adds_to_recent(self) -> bool {
        matches!(self, Self::Documents)
    }
    fn key(self) -> &'static str {
        match self {
            Self::Documents => "documents",
            Self::Themes => "themes",
            Self::Runtime => "runtime",
            Self::Model => "model",
            Self::Scripts => "scripts",
            Self::Resource => "resources",
        }
    }
}

static LOCK: Mutex<()> = Mutex::new(());
pub(crate) const HISTORY_FILE_NAME: &str = "file_dialog_directories.json";

fn load(path: &Path) -> BTreeMap<String, PathBuf> {
    std::fs::read(path)
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
        .unwrap_or_default()
}

fn directory(
    history: &BTreeMap<String, PathBuf>,
    purpose: Purpose,
    current: Option<&Path>,
    fallback: &Path,
) -> PathBuf {
    history
        .get(purpose.key())
        .filter(|p| p.is_dir())
        .cloned()
        .or_else(|| {
            current
                .and_then(Path::parent)
                .filter(|p| p.is_dir())
                .map(Path::to_owned)
        })
        .unwrap_or_else(|| fallback.to_owned())
}

#[cfg(not(target_arch = "wasm32"))]
pub fn dialog(purpose: Purpose, current: Option<&Path>) -> rfd::FileDialog {
    let _lock = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let root = crate::services::storage::data_dir();
    let history = load(&root.join(HISTORY_FILE_NAME));
    rfd::FileDialog::new()
        .set_add_to_recent(purpose.adds_to_recent())
        .set_directory(directory(&history, purpose, current, &root))
}

/// Resource pickers preserve Windows document history for actual documents only.
#[cfg(not(target_arch = "wasm32"))]
pub fn resource_dialog() -> rfd::FileDialog {
    rfd::FileDialog::new().set_add_to_recent(Purpose::Resource.adds_to_recent())
}

/// Explicit cleanup of known configuration resources, never the whole Jump List.
#[cfg(target_os = "windows")]
pub fn remove_recent_resources(paths: &[&str]) -> Result<(), String> {
    use std::os::windows::ffi::OsStrExt;
    unsafe extern "C" {
        fn paintfe_remove_recent_resource(path: *const u16) -> i32;
    }
    for path in paths.iter().filter(|path| !path.trim().is_empty()) {
        if path.contains('\0') {
            return Err("Invalid resource path".into());
        }
        let wide: Vec<u16> = std::ffi::OsStr::new(path)
            .encode_wide()
            .chain(Some(0))
            .collect();
        // The Shell receives a valid, NUL-terminated path during this call only.
        let result = unsafe { paintfe_remove_recent_resource(wide.as_ptr()) };
        if result < 0 {
            return Err(format!(
                "Windows recent-history cleanup failed (0x{:08X})",
                result as u32
            ));
        }
    }
    Ok(())
}

pub fn remember(purpose: Purpose, selected: &Path) {
    let Some(parent) = selected.parent().filter(|p| p.is_dir()) else {
        return;
    };
    let _lock = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let root = crate::services::storage::data_dir();
    let path = root.join(HISTORY_FILE_NAME);
    let mut history = load(&path);
    history.insert(purpose.key().into(), parent.to_owned());
    if let Ok(bytes) = serde_json::to_vec(&history) {
        let _ = std::fs::create_dir_all(&root);
        if let Err(error) = std::fs::write(path, bytes) {
            crate::log_info!("Cannot remember file picker directory: {error}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(target_os = "windows")]
    #[test]
    fn targeted_history_cleanup_accepts_existing_fixture_and_rejects_nul_paths() {
        let path =
            std::env::temp_dir().join(format!("paintfe-picker-{}.onnx", uuid::Uuid::new_v4()));
        std::fs::write(&path, b"picker test resource, not a model").unwrap();
        remove_recent_resources(&[path.to_str().unwrap()]).unwrap();
        assert!(remove_recent_resources(&["bad\0path"]).is_err());
        std::fs::remove_file(path).unwrap();
    }
    #[test]
    fn only_document_pickers_register_windows_recent_documents() {
        assert!(Purpose::Documents.adds_to_recent());
        for purpose in [
            Purpose::Themes,
            Purpose::Runtime,
            Purpose::Model,
            Purpose::Scripts,
            Purpose::Resource,
        ] {
            assert!(!purpose.adds_to_recent());
        }
    }
    #[test]
    fn directory_categories_survive_serialization_without_leaking_into_documents() {
        let root = std::env::temp_dir().join(format!("paintfe-picker-{}", uuid::Uuid::new_v4()));
        let themes = root.join("themes");
        let photos = root.join("photos");
        std::fs::create_dir_all(&themes).unwrap();
        std::fs::create_dir_all(&photos).unwrap();
        let mut history: BTreeMap<String, PathBuf> = BTreeMap::new();
        history.insert(Purpose::Documents.key().into(), photos.clone());
        history.insert(Purpose::Themes.key().into(), themes.clone());
        let path = root.join("directories.json");
        std::fs::write(&path, serde_json::to_vec(&history).unwrap()).unwrap();
        let restored = load(&path);
        assert_eq!(directory(&restored, Purpose::Themes, None, &root), themes);
        assert_eq!(
            directory(&restored, Purpose::Documents, None, &root),
            photos
        );
        for purpose in [Purpose::Runtime, Purpose::Model, Purpose::Scripts] {
            assert_eq!(directory(&restored, purpose, None, &root), root);
        }
        // Every resource picker has its own restart-persistent history.
        for purpose in [Purpose::Runtime, Purpose::Model, Purpose::Scripts] {
            let own = root.join(purpose.key());
            std::fs::create_dir(&own).unwrap();
            history.insert(purpose.key().into(), own);
        }
        std::fs::write(&path, serde_json::to_vec(&history).unwrap()).unwrap();
        let restored = load(&path);
        for purpose in [Purpose::Runtime, Purpose::Model, Purpose::Scripts] {
            assert_eq!(
                directory(&restored, purpose, Some(&photos.join("image.png")), &root),
                root.join(purpose.key())
            );
        }
        // Missing directories and broken history must not poison future pickers.
        let mut missing = BTreeMap::new();
        missing.insert(Purpose::Model.key().into(), root.join("removed"));
        assert_eq!(
            directory(
                &missing,
                Purpose::Model,
                Some(&photos.join("image.png")),
                &root
            ),
            photos
        );
        std::fs::write(&path, b"invalid json").unwrap();
        assert!(load(&path).is_empty());
        std::fs::remove_dir_all(root).unwrap();
    }
}
