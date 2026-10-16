//! Coalesce Preferences edits and atomically replace the last complete settings file.
use std::{
    collections::HashMap,
    path::PathBuf,
    sync::{LazyLock, Mutex},
    time::{Duration, Instant},
};

#[derive(Default)]
struct Saves {
    deferred: bool,
    pending: HashMap<PathBuf, (String, Instant)>,
    written: HashMap<PathBuf, String>,
    error: Option<String>,
}
static SAVES: LazyLock<Mutex<Saves>> = LazyLock::new(|| Mutex::new(Saves::default()));

pub(crate) fn begin() {
    SAVES.lock().unwrap_or_else(|e| e.into_inner()).deferred = true;
}

#[cfg(not(target_arch = "wasm32"))]
fn replace(path: &std::path::Path, content: &str) -> std::io::Result<()> {
    use std::io::Write;
    let parent = path
        .parent()
        .ok_or_else(|| std::io::Error::other("Missing settings directory"))?;
    std::fs::create_dir_all(parent)?;
    let temporary = path.with_file_name(format!(
        ".{}.{}.tmp",
        path.file_name().unwrap_or_default().to_string_lossy(),
        std::process::id()
    ));
    let result = (|| {
        let mut file = std::fs::File::create(&temporary)?;
        file.write_all(content.as_bytes())?;
        file.sync_all()?;
        drop(file);
        std::fs::rename(&temporary, path)
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&temporary);
    }
    result
}

fn write(state: &mut Saves, path: PathBuf, content: String) {
    #[cfg(not(target_arch = "wasm32"))]
    let result = replace(&path, &content).map_err(|e| e.to_string());
    #[cfg(target_arch = "wasm32")]
    let result = crate::web_storage::save_settings(&content).map_err(|e| e.to_string());
    match result {
        Ok(()) => {
            state.written.insert(path, content);
            state.error = None;
        }
        Err(error) => {
            state.error = Some(error);
        }
    }
}

pub(crate) fn request(path: PathBuf, content: String) {
    let mut state = SAVES.lock().unwrap_or_else(|e| e.into_inner());
    if state
        .pending
        .get(&path)
        .is_some_and(|(old, _)| old == &content)
    {
        return;
    }
    if state.written.get(&path) == Some(&content) {
        state.pending.remove(&path);
        return;
    }
    #[cfg(not(target_arch = "wasm32"))]
    if state.deferred {
        state.pending.insert(path, (content, Instant::now()));
        return;
    }
    state.pending.remove(&path);
    write(&mut state, path, content);
}

/// Returns whether a pending edit needs another repaint to reach its deadline.
pub(crate) fn flush(force: bool) -> bool {
    let mut state = SAVES.lock().unwrap_or_else(|e| e.into_inner());
    state.deferred = false;
    let ready: Vec<_> = state
        .pending
        .iter()
        .filter(|(_, (_, at))| force || at.elapsed() >= Duration::from_millis(250))
        .map(|(p, _)| p.clone())
        .collect();
    for path in ready {
        if let Some((content, _)) = state.pending.remove(&path) {
            write(&mut state, path, content);
        }
    }
    !state.pending.is_empty()
}
pub(crate) fn error() -> Option<String> {
    SAVES
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .error
        .clone()
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    #[test]
    fn atomic_replacement_preserves_previous_file_on_failure() {
        let dir = std::env::current_dir()
            .unwrap()
            .join("target/persistence-tests")
            .join(format!("atomic-{}", uuid::Uuid::new_v4()));
        let path = dir.join("settings.cfg");
        replace(&path, "old").unwrap();
        replace(&path, "new").unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "new");
        let blocked = dir.join("blocked");
        std::fs::create_dir(&blocked).unwrap();
        assert!(replace(&blocked, "bad").is_err());
        assert!(blocked.is_dir());
        assert_eq!(std::fs::read_to_string(path).unwrap(), "new");
        std::fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn deferred_changes_are_coalesced_and_flushed_on_close() {
        let dir = std::env::current_dir()
            .unwrap()
            .join("target/persistence-tests")
            .join(format!("debounce-{}", uuid::Uuid::new_v4()));
        let path = dir.join("settings.cfg");
        begin();
        request(path.clone(), "first".into());
        request(path.clone(), "final".into());
        assert!(!path.exists());
        assert!(flush(false));
        flush(true);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "final");
        let modified = std::fs::metadata(&path).unwrap().modified().unwrap();
        request(path.clone(), "final".into());
        assert_eq!(
            std::fs::metadata(&path).unwrap().modified().unwrap(),
            modified
        );
        std::fs::remove_dir_all(dir).unwrap();
    }
}
