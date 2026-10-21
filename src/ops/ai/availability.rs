//! Coalesce runtime availability checks on one worker, never on the UI thread.
use crate::time_compat::Instant;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::Arc;
use std::{sync::mpsc, time::Duration};
type Paths = (String, String);
#[cfg(not(target_arch = "wasm32"))]
type Probe = Arc<dyn Fn(&str, &str) -> bool + Send + Sync>;

pub(crate) struct AvailabilityProbe {
    desired: Option<Paths>,
    completed: Option<Paths>,
    receiver: Option<mpsc::Receiver<(Paths, bool)>>,
    not_before: Instant,
    available: bool,
    #[cfg(not(target_arch = "wasm32"))]
    probe: Probe,
}
impl Default for AvailabilityProbe {
    fn default() -> Self {
        Self {
            desired: None,
            completed: None,
            receiver: None,
            not_before: Instant::now(),
            available: false,
            #[cfg(not(target_arch = "wasm32"))]
            probe: Arc::new(|runtime, model| {
                super::probe_onnx_runtime(runtime).is_ok() && std::path::Path::new(model).is_file()
            }),
        }
    }
}
impl AvailabilityProbe {
    pub(crate) fn update(&mut self, ctx: &egui::Context, runtime: &str, model: &str) -> bool {
        if !self
            .desired
            .as_ref()
            .is_some_and(|p| p.0 == runtime && p.1 == model)
        {
            self.desired = Some((runtime.into(), model.into()));
            self.completed = None;
            self.available = false;
            // Allow the first window paint before loading a configured native DLL.
            self.not_before = Instant::now() + Duration::from_millis(100);
        }
        if let Some(receiver) = &self.receiver {
            match receiver.try_recv() {
                Ok((paths, available)) => {
                    self.receiver = None;
                    if self.desired.as_ref() == Some(&paths) {
                        self.completed = Some(paths);
                        self.available = available;
                    }
                }
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.receiver = None;
                    self.completed = self.desired.clone();
                }
                Err(mpsc::TryRecvError::Empty) => {}
            }
        }
        let paths = self.desired.as_ref().unwrap();
        if self.receiver.is_none() && self.completed.as_ref() != Some(paths) {
            if runtime.is_empty() || model.is_empty() {
                self.completed = Some(paths.clone());
            } else if Instant::now() < self.not_before {
                ctx.request_repaint_after(
                    self.not_before.saturating_duration_since(Instant::now()),
                );
            } else {
                #[cfg(not(target_arch = "wasm32"))]
                {
                    let (sender, receiver) = mpsc::channel();
                    let probe = self.probe.clone();
                    let paths = paths.clone();
                    let ctx = ctx.clone();
                    self.receiver = Some(receiver);
                    if std::thread::Builder::new()
                        .name("paintfe-onnx-probe".into())
                        .spawn(move || {
                            let _profile =
                                crate::ui::perf::StartupSpan::new("onnx_probe_background");
                            let available =
                                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                    probe(&paths.0, &paths.1)
                                }))
                                .unwrap_or(false);
                            let _ = sender.send((paths, available));
                            ctx.request_repaint();
                        })
                        .is_err()
                    {
                        self.receiver = None;
                        self.completed = self.desired.clone();
                    }
                }
                #[cfg(target_arch = "wasm32")]
                {
                    self.completed = self.desired.clone();
                }
            }
        }
        self.available
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    fn finish(probe: &mut AvailabilityProbe, ctx: &egui::Context, runtime: &str) -> bool {
        let deadline = Instant::now() + Duration::from_secs(3);
        loop {
            let result = probe.update(ctx, runtime, "model");
            if probe.completed.as_ref() == probe.desired.as_ref() {
                return result;
            }
            assert!(Instant::now() < deadline, "probe did not complete");
            std::thread::yield_now();
        }
    }
    #[test]
    fn pending_probe_never_blocks_ui_and_stale_result_is_ignored() {
        let (started, started_rx) = mpsc::channel();
        let (release, release_rx) = mpsc::channel();
        let gate = std::sync::Mutex::new(release_rx);
        let mut probe = AvailabilityProbe {
            probe: Arc::new(move |runtime, _| {
                started.send(runtime.to_owned()).unwrap();
                if runtime == "old" {
                    gate.lock().unwrap().recv().unwrap();
                }
                runtime == "old"
            }),
            ..Default::default()
        };
        let ctx = egui::Context::default();
        assert!(!probe.update(&ctx, "old", "model"));
        probe.not_before = Instant::now();
        assert!(!probe.update(&ctx, "old", "model"));
        assert_eq!(
            started_rx.recv_timeout(Duration::from_secs(3)).unwrap(),
            "old"
        );
        assert!(!probe.update(&ctx, "new", "model"));
        assert!(
            started_rx.try_recv().is_err(),
            "second worker started before first completed"
        );
        probe.not_before = Instant::now();
        release.send(()).unwrap();
        assert!(!finish(&mut probe, &ctx, "new"));
        assert_eq!(
            started_rx.recv_timeout(Duration::from_secs(3)).unwrap(),
            "new"
        );
        for _ in 0..10 {
            assert!(!probe.update(&ctx, "new", "model"));
        }
        assert!(
            started_rx.try_recv().is_err(),
            "unchanged paths were probed again"
        );
    }
    #[test]
    fn empty_paths_and_panicking_probe_remain_unavailable() {
        let ctx = egui::Context::default();
        let mut probe = AvailabilityProbe {
            probe: Arc::new(|_, _| panic!("simulated invalid runtime")),
            ..Default::default()
        };
        assert!(!probe.update(&ctx, "", ""));
        assert!(probe.receiver.is_none());
        probe.update(&ctx, "runtime", "model");
        probe.not_before = Instant::now();
        assert!(!finish(&mut probe, &ctx, "runtime"));
    }
}
