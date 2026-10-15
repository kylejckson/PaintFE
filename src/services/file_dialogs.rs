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
}
impl Purpose {
    fn key(self) -> &'static str {
        match self {
            Self::Documents => "documents",
            Self::Themes => "themes",
            Self::Runtime => "runtime",
            Self::Model => "model",
            Self::Scripts => "scripts",
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
    rfd::FileDialog::new().set_directory(directory(&history, purpose, current, &root))
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
