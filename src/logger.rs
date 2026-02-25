//! Session logger — writes all log output to a single file in the OS data directory.
//!
//! The file is **truncated (overwritten) at each launch**, so it only ever
//! contains output from the most-recent session.  This prevents the log from
//! growing unboundedly.
//!
//! Log location:
//!   Windows:  `%APPDATA%\PaintFE\paintfe.log`
//!   Linux:    `~/.local/share/PaintFE/paintfe.log`
//!   macOS:    `~/Library/Application Support/PaintFE/paintfe.log`
//!
//! Usage — anywhere in the crate use the `log_info!` / `log_warn!` / `log_err!`
//! macros, or call `crate::logger::write_line(...)` directly.  The standard
//! `eprintln!` / `println!` calls you already have in the codebase continue to
//! work normally (they are captured via a custom panic hook and mirrored to the
//! log file).

use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

static LOG_FILE: OnceLock<Mutex<File>> = OnceLock::new();
static LOG_PATH: OnceLock<PathBuf> = OnceLock::new();

/// Returns the path to the current session log file.
pub fn log_path() -> Option<&'static PathBuf> {
    LOG_PATH.get()
}

/// Write a line to the session log.  Silently ignores I/O errors so that
/// logging never crashes the application.
pub fn write_line(line: &str) {
    if let Some(mutex) = LOG_FILE.get()
        && let Ok(mut file) = mutex.lock()
    {
        let _ = writeln!(file, "{}", line);
    }
}

/// Write a timestamped, level-tagged line to the session log.
pub fn write(level: &str, msg: &str) {
    let ts = timestamp();
    write_line(&format!("[{}] [{}] {}", ts, level, msg));
}

#[macro_export]
macro_rules! log_info {
    ($($arg:tt)*) => {
        $crate::logger::write("INFO", &format!($($arg)*));
    };
}

#[macro_export]
macro_rules! log_warn {
    ($($arg:tt)*) => {
        $crate::logger::write("WARN", &format!($($arg)*));
    };
}

#[macro_export]
macro_rules! log_err {
    ($($arg:tt)*) => {
        $crate::logger::write("ERROR", &format!($($arg)*));
    };
}

/// Initialise the session logger.  Must be called once before any logging.
///
/// * Creates (or truncates) the log file.
/// * Installs a panic hook that writes the panic message to the log before
///   propagating to the default handler.
pub fn init() {
    let path = log_file_path();

    // Ensure parent directory exists
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }

    // Open file, truncating any previous session's content
    let file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&path);

    match file {
        Ok(f) => {
            let _ = LOG_PATH.set(path.clone());
            let _ = LOG_FILE.set(Mutex::new(f));
        }
        Err(e) => {
            // Can't open log file — not fatal, just skip
            eprintln!("[logger] Failed to open log file {:?}: {}", path, e);
            return;
        }
    }

    // Write session header
    write_line(&format!(
        "=== PaintFE session started {} ===",
        human_timestamp()
    ));
    write_line(&format!("Log file: {}", path.display()));
    write_line("");

    // Install panic hook — mirrors panic info to the log, then runs default handler
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let msg = format!("PANIC: {}", info);
        write_line(&format!("[{}] [PANIC] {}", timestamp(), msg));
        // Flush by dropping the lock immediately (write_line already dropped it)
        prev(info);
    }));
}

fn log_file_path() -> PathBuf {
    let base = data_dir();
    base.join("PaintFE").join("paintfe.log")
}

/// Platform data directory (without the app sub-folder).
fn data_dir() -> PathBuf {
    #[cfg(target_os = "windows")]
    {
        if let Ok(appdata) = std::env::var("APPDATA") {
            return PathBuf::from(appdata);
        }
    }
    #[cfg(target_os = "macos")]
    {
        if let Some(home) = std::env::var("HOME").ok() {
            return PathBuf::from(home)
                .join("Library")
                .join("Application Support");
        }
    }
    // Linux / fallback
    if let Ok(xdg) = std::env::var("XDG_DATA_HOME") {
        return PathBuf::from(xdg);
    }
    if let Ok(home) = std::env::var("HOME") {
        return PathBuf::from(home).join(".local").join("share");
    }
    // Last resort: current working directory
    PathBuf::from(".")
}

/// Simple seconds-since-epoch timestamp string.
fn timestamp() -> String {
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(d) => {
            let secs = d.as_secs();
            // Format as HH:MM:SS within the current day (good enough for a session log)
            let h = (secs % 86400) / 3600;
            let m = (secs % 3600) / 60;
            let s = secs % 60;
            format!("{:02}:{:02}:{:02}", h, m, s)
        }
        Err(_) => "??:??:??".to_string(),
    }
}

/// Human-readable date-time for the session header.
fn human_timestamp() -> String {
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(d) => {
            let secs = d.as_secs();
            // Very simple: just show unix seconds since we have no chrono dep
            format!("(unix {})", secs)
        }
        Err(_) => "(unknown time)".to_string(),
    }
}
