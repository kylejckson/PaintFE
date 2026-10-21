//! Browser local storage stays synchronous; no native filesystem or worker.
use std::{
    collections::HashMap,
    path::PathBuf,
    sync::{LazyLock, Mutex},
};
#[derive(Default)]
struct Saves {
    written: HashMap<PathBuf, String>,
    error: Option<String>,
}
static SAVES: LazyLock<Mutex<Saves>> = LazyLock::new(|| Mutex::new(Saves::default()));
pub(crate) fn begin() {}
pub(crate) fn request(path: PathBuf, content: String) {
    let mut state = SAVES.lock().unwrap_or_else(|e| e.into_inner());
    if state.written.get(&path) == Some(&content) {
        return;
    }
    match crate::web_storage::save_settings(&content) {
        Ok(()) => {
            state.written.insert(path, content);
            state.error = None;
        }
        Err(error) => {
            state.error = Some(error.to_string());
        }
    }
}
pub(crate) fn flush(_: bool) -> bool {
    false
}
pub(crate) fn error() -> Option<String> {
    SAVES
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .error
        .clone()
}
pub fn drain() -> Result<(), String> {
    error().map_or(Ok(()), Err)
}
