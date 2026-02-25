use std::path::PathBuf;
use uuid::Uuid;

use crate::canvas::CanvasState;
use crate::components::history::HistoryManager;
use crate::io::FileHandler;

/// Single open document.
pub struct Project {
    pub id: Uuid,
    pub canvas_state: CanvasState,
    pub history: HistoryManager,
    pub file_handler: FileHandler,
    /// `None` for unsaved/untitled files.
    pub path: Option<PathBuf>,
    pub is_dirty: bool,

    /// Display name (derived from path or "Untitled-X")
    pub name: String,

    /// True if the file was opened from an animated GIF/APNG
    pub was_animated: bool,

    /// Animation FPS preserved from import (default 10.0)
    pub animation_fps: f32,
}

impl Project {
    pub fn new_untitled(untitled_counter: usize, width: u32, height: u32) -> Self {
        let name = format!("Untitled-{}", untitled_counter);

        Self {
            id: Uuid::new_v4(),
            canvas_state: CanvasState::new(width, height),
            history: HistoryManager::new(50), // Default 50 history steps
            file_handler: FileHandler::new(),
            path: None,
            is_dirty: false,
            name,
            was_animated: false,
            animation_fps: 10.0,
        }
    }

    pub fn from_file(path: PathBuf, canvas_state: CanvasState, file_handler: FileHandler) -> Self {
        let name = path
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "Unknown".to_string());

        Self {
            id: Uuid::new_v4(),
            canvas_state,
            history: HistoryManager::new(50),
            file_handler,
            path: Some(path),
            is_dirty: false,
            name,
            was_animated: false,
            animation_fps: 10.0,
        }
    }

    pub fn mark_dirty(&mut self) {
        self.is_dirty = true;
    }

    pub fn mark_clean(&mut self) {
        self.is_dirty = false;
    }

    pub fn update_name_from_path(&mut self) {
        if let Some(ref path) = self.path {
            self.name = path
                .file_name()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| "Unknown".to_string());
        }
    }

    /// Get the display title (name with dirty indicator)
    pub fn display_title(&self) -> String {
        if self.is_dirty {
            format!("{}*", self.name)
        } else {
            self.name.clone()
        }
    }
}
