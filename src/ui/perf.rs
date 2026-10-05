//! Opt-in local UI tracing. Set PAINTFE_UI_PROFILE to a JSONL output path.
//! No document content, paths, text input, or credentials are recorded.
use serde::Serialize;
use std::{
    cell::RefCell,
    sync::{OnceLock, mpsc},
    time::Instant,
};

pub const STAGES: [&str; 24] = [
    "lifecycle",
    "motion_policy",
    "os_motion",
    "settings_save",
    "input",
    "menus_dialogs",
    "canvas_tail",
    "canvas_render",
    "tools",
    "layers",
    "history",
    "colors",
    "palette",
    "script",
    "workspace",
    "window_persist",
    "theme",
    "preferences_sync",
    "plugin_scan",
    "preferences_body",
    "preferences_header",
    "preferences_sidebar",
    "preferences_content",
    "total",
];
#[derive(Clone, Serialize)]
struct Sample {
    time_ms: f64,
    gap_ms: f64,
    events: usize,
    previous_cpu_ms: Option<f64>,
    ms: [f64; STAGES.len()],
}
thread_local! { static CURRENT: RefCell<Option<Sample>> = const { RefCell::new(None) }; }
struct Backend {
    sender: mpsc::SyncSender<Sample>,
    origin: Instant,
    drive: bool,
}
static BACKEND: OnceLock<Option<Backend>> = OnceLock::new();

fn backend() -> Option<&'static Backend> {
    BACKEND
        .get_or_init(|| {
            let path = std::env::var_os("PAINTFE_UI_PROFILE")?;
            let path = std::path::PathBuf::from(path);
            let drive = std::env::var_os("PAINTFE_UI_PROFILE_DRIVE").is_some();
            let (sender, receiver) = mpsc::sync_channel::<Sample>(1024);
            std::thread::Builder::new()
                .name("paintfe-ui-profile".into())
                .spawn(move || {
                    use std::io::Write;
                    let Ok(file) = std::fs::File::create(path) else {
                        return;
                    };
                    let mut writer = std::io::BufWriter::new(file);
                    let _ = serde_json::to_writer(
                        &mut writer,
                        &serde_json::json!({"stages": STAGES, "version": 1, "continuous": drive}),
                    );
                    let _ = writeln!(writer);
                    let mut count = 0;
                    loop {
                        match receiver.recv_timeout(std::time::Duration::from_millis(100)) {
                            Ok(sample) => {
                                if count >= 20_000 {
                                    break;
                                }
                                if serde_json::to_writer(&mut writer, &sample).is_err() {
                                    break;
                                }
                                let _ = writeln!(writer);
                                count += 1;
                                if count % 64 == 0 {
                                    let _ = writer.flush();
                                }
                            }
                            Err(mpsc::RecvTimeoutError::Timeout) => {
                                let _ = writer.flush();
                            }
                            Err(mpsc::RecvTimeoutError::Disconnected) => break,
                        }
                    }
                    let _ = writer.flush();
                })
                .ok()?;
            Some(Backend {
                sender,
                origin: Instant::now(),
                drive,
            })
        })
        .as_ref()
}
pub struct Frame(Option<Instant>);
impl Frame {
    pub fn begin(ctx: &egui::Context, previous_cpu: Option<f32>) -> Self {
        let Some(backend) = backend() else {
            return Self(None);
        };
        let now = Instant::now();
        let time_ms = backend.origin.elapsed().as_secs_f64() * 1000.0;
        // A bounded 20-second continuous sample distinguishes presentation gaps
        // from ordinary event-driven idle time. Never active without explicit opt-in.
        if backend.drive && time_ms < 20_000.0 {
            ctx.request_repaint();
        }
        let last = ctx.data(|d| d.get_temp::<f64>(egui::Id::new("paintfe_profile_last_frame")));
        ctx.data_mut(|d| d.insert_temp(egui::Id::new("paintfe_profile_last_frame"), time_ms));
        CURRENT.with_borrow_mut(|slot| {
            *slot = Some(Sample {
                time_ms,
                gap_ms: last.map_or(0.0, |last| time_ms - last),
                events: ctx.input(|i| i.events.len()),
                previous_cpu_ms: previous_cpu.map(|v| v as f64 * 1000.0),
                ms: [0.0; STAGES.len()],
            })
        });
        Self(Some(now))
    }
}
impl Drop for Frame {
    fn drop(&mut self) {
        if let Some(start) = self.0 {
            CURRENT.with_borrow_mut(|slot| {
                if let Some(mut sample) = slot.take() {
                    sample.ms[STAGES.len() - 1] = start.elapsed().as_secs_f64() * 1000.0;
                    if let Some(backend) = BACKEND.get().and_then(Option::as_ref) {
                        let _ = backend.sender.try_send(sample);
                    }
                }
            });
        }
    }
}
pub struct Scope {
    stage: usize,
    start: Option<Instant>,
}
impl Scope {
    pub fn new(stage: usize) -> Self {
        let enabled = BACKEND.get().is_some_and(Option::is_some);
        Self {
            stage,
            start: enabled.then(Instant::now),
        }
    }
}
impl Drop for Scope {
    fn drop(&mut self) {
        if let Some(start) = self.start {
            CURRENT.with_borrow_mut(|slot| {
                if let Some(sample) = slot {
                    sample.ms[self.stage] += start.elapsed().as_secs_f64() * 1000.0;
                }
            });
        }
    }
}
