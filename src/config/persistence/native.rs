use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::{Arc, Condvar, LazyLock, Mutex},
    time::{Duration, Instant},
};

#[derive(Default)]
struct State {
    desired: HashMap<PathBuf, Arc<str>>,
    pending: HashMap<PathBuf, (Arc<str>, Instant)>,
    errors: HashMap<PathBuf, String>,
    busy: bool,
    failed_start: Option<String>,
    #[cfg(test)]
    stop: bool,
}

struct Writer {
    state: Mutex<State>,
    wake: Condvar,
    drained: Condvar,
}

impl Writer {
    fn start(write: impl Fn(&Path, &str) -> Result<(), String> + Send + 'static) -> Arc<Self> {
        let writer = Arc::new(Self {
            state: Mutex::new(State::default()),
            wake: Condvar::new(),
            drained: Condvar::new(),
        });
        let worker = writer.clone();
        if let Err(error) = std::thread::Builder::new()
            .name("paintfe-settings-writer".into())
            .spawn(move || worker.run(write))
        {
            writer
                .state
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .failed_start = Some(error.to_string());
        }
        writer
    }

    fn run(&self, write: impl Fn(&Path, &str) -> Result<(), String>) {
        loop {
            let (path, content) = {
                let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
                loop {
                    #[cfg(test)]
                    if state.stop {
                        return;
                    }
                    let next = state
                        .pending
                        .iter()
                        .min_by_key(|(_, (_, deadline))| *deadline)
                        .map(|(path, (_, deadline))| (path.clone(), *deadline));
                    if let Some((path, deadline)) = next {
                        let now = Instant::now();
                        if deadline <= now {
                            let (content, _) = state.pending.remove(&path).unwrap();
                            state.busy = true;
                            break (path, content);
                        }
                        state = self
                            .wake
                            .wait_timeout(state, deadline - now)
                            .unwrap_or_else(|e| e.into_inner())
                            .0;
                    } else {
                        state = self.wake.wait(state).unwrap_or_else(|e| e.into_inner());
                    }
                }
            };
            // Never hold the queue lock during file creation, fsync or rename.
            let result =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| write(&path, &content)))
                    .unwrap_or_else(|_| Err("Settings writer failed".into()));
            let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
            state.busy = false;
            match result {
                Ok(()) => {
                    state.errors.remove(&path);
                }
                Err(error) => {
                    state.errors.insert(path, error);
                }
            }
            self.drained.notify_all();
        }
    }

    fn request(&self, path: PathBuf, content: String) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if !state.errors.contains_key(&path)
            && state
                .desired
                .get(&path)
                .is_some_and(|s| s.as_ref() == content)
        {
            return;
        }
        let content: Arc<str> = content.into();
        state.desired.insert(path.clone(), content.clone());
        state
            .pending
            .insert(path, (content, Instant::now() + Duration::from_millis(250)));
        self.wake.notify_one();
    }

    fn expedite(&self) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        for (_, deadline) in state.pending.values_mut() {
            *deadline = Instant::now();
        }
        self.wake.notify_one();
    }

    fn drain(&self) -> Result<(), String> {
        self.expedite();
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        while state.busy || !state.pending.is_empty() {
            if let Some(error) = &state.failed_start {
                return Err(error.clone());
            }
            state = self.drained.wait(state).unwrap_or_else(|e| e.into_inner());
        }
        state
            .failed_start
            .clone()
            .or_else(|| state.errors.values().next().cloned())
            .map_or(Ok(()), Err)
    }
}

static WRITER: LazyLock<Arc<Writer>> = LazyLock::new(|| Writer::start(replace));

fn replace(path: &Path, content: &str) -> Result<(), String> {
    let _profile = crate::ui::perf::StartupSpan::new("settings_write_background");
    use std::io::Write;
    let parent = path.parent().ok_or("Missing settings directory")?;
    std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    let temporary = path.with_file_name(format!(
        ".{}.{}.tmp",
        path.file_name().unwrap_or_default().to_string_lossy(),
        std::process::id()
    ));
    let result = (|| -> std::io::Result<()> {
        let mut file = std::fs::File::create(&temporary)?;
        file.write_all(content.as_bytes())?;
        file.sync_all()?;
        drop(file);
        std::fs::rename(&temporary, path)
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&temporary);
    }
    result.map_err(|e| e.to_string())
}

pub(crate) fn begin() {}
pub(crate) fn request(path: PathBuf, content: String) {
    WRITER.request(path, content);
}
/// Expedite on interaction completion; the caller never waits for disk I/O.
pub(crate) fn flush(force: bool) -> bool {
    if force {
        WRITER.expedite();
    }
    false
}
pub(crate) fn error() -> Option<String> {
    let state = WRITER.state.lock().unwrap_or_else(|e| e.into_inner());
    state
        .failed_start
        .clone()
        .or_else(|| state.errors.values().next().cloned())
}
/// Explicit barrier for shutdown and storage migration, never ordinary frames.
pub fn drain() -> Result<(), String> {
    WRITER.drain()
}

#[cfg(test)]
mod tests {
    use super::*;
    fn path() -> PathBuf {
        std::env::current_dir()
            .unwrap()
            .join("target/persistence-tests")
            .join(uuid::Uuid::new_v4().to_string())
            .join("settings.cfg")
    }
    fn stop(writer: &Writer) {
        writer.state.lock().unwrap().stop = true;
        writer.wake.notify_one();
    }
    #[test]
    fn atomic_replacement_preserves_previous_file_on_failure() {
        let path = path();
        replace(&path, "old").unwrap();
        let blocked = path.parent().unwrap().join("blocked");
        std::fs::create_dir(&blocked).unwrap();
        assert!(replace(&blocked, "bad").is_err());
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "old");
    }
    #[test]
    fn background_writes_coalesce_and_shutdown_drains_latest_value() {
        let path = path();
        let writer = Writer::start(replace);
        writer.request(path.clone(), "first".into());
        writer.request(path.clone(), "final".into());
        writer.drain().unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "final");
        let modified = std::fs::metadata(&path).unwrap().modified().unwrap();
        writer.request(path.clone(), "final".into());
        writer.drain().unwrap();
        assert_eq!(
            std::fs::metadata(&path).unwrap().modified().unwrap(),
            modified
        );
        stop(&writer);
    }
    #[test]
    fn failed_write_reports_error_and_same_value_can_be_retried() {
        let path = path();
        let attempts = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let count = attempts.clone();
        let writer = Writer::start(move |path, content| {
            if count.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == 0 {
                Err("simulated disk error".into())
            } else {
                replace(path, content)
            }
        });
        writer.request(path.clone(), "saved".into());
        assert_eq!(writer.drain().unwrap_err(), "simulated disk error");
        writer.request(path.clone(), "saved".into());
        writer.drain().unwrap();
        assert_eq!(std::fs::read_to_string(path).unwrap(), "saved");
        assert_eq!(attempts.load(std::sync::atomic::Ordering::SeqCst), 2);
        stop(&writer);
    }
    #[test]
    fn slow_disk_does_not_hold_ui_queue_lock_and_newest_value_wins() {
        use std::sync::mpsc;
        let path = path();
        let (started, started_rx) = mpsc::channel();
        let (release, release_rx) = mpsc::channel();
        let mut first = true;
        let first_lock = Mutex::new(move || {
            if first {
                first = false;
                started.send(()).unwrap();
                release_rx.recv().unwrap();
            }
        });
        let writer = Writer::start(move |path, text| {
            let mut first = first_lock.lock().unwrap();
            first();
            drop(first);
            replace(path, text)
        });
        writer.request(path.clone(), "original".into());
        writer.expedite();
        started_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        let queued_writer = writer.clone();
        let queued_path = path.clone();
        let (queued, queued_rx) = mpsc::channel();
        std::thread::spawn(move || {
            queued_writer.request(queued_path.clone(), "newest".into());
            queued_writer.request(queued_path, "original".into());
            queued.send(()).unwrap();
        });
        let responsive = queued_rx.recv_timeout(Duration::from_secs(2));
        release.send(()).unwrap();
        assert!(
            responsive.is_ok(),
            "UI queue waited for the blocked disk writer"
        );
        writer.drain().unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "original");
        stop(&writer);
    }
}
