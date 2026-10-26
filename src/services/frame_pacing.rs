//! Event-driven frame pacing. Input events always reach eframe unchanged.
use std::{
    collections::HashMap,
    sync::{Arc, Mutex, Weak},
    time::{Duration, Instant},
};
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, DeviceId, StartCause, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};

#[derive(Clone, Copy)]
struct Policy {
    detect: bool,
    fps: u32,
}
static POLICY: Mutex<Policy> = Mutex::new(Policy {
    detect: true,
    fps: 60,
});
struct WindowInfo {
    window: Weak<Window>,
    refresh: Option<u32>,
    checked: Instant,
}
static WINDOWS: Mutex<Option<HashMap<WindowId, WindowInfo>>> = Mutex::new(None);

pub fn set_policy(detect: bool, fps: u32) {
    *POLICY.lock().unwrap_or_else(|e| e.into_inner()) = Policy {
        detect,
        fps: fps.clamp(15, 360),
    };
}
pub fn register_window(window: Arc<Window>) {
    WINDOWS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get_or_insert_with(HashMap::new)
        .insert(
            window.id(),
            WindowInfo {
                window: Arc::downgrade(&window),
                refresh: None,
                checked: Instant::now() - Duration::from_secs(2),
            },
        );
}
fn interval(policy: Policy, refresh_millihertz: Option<u32>) -> Duration {
    let rate = if policy.detect {
        refresh_millihertz
            .filter(|r| *r >= 15_000)
            .unwrap_or(60_000) as f64
            / 1000.0
    } else {
        policy.fps.clamp(15, 360) as f64
    };
    Duration::from_secs_f64(1.0 / rate.clamp(15.0, 1000.0))
}
#[derive(Default)]
struct Budget {
    last: Option<Instant>,
    pending: Option<Instant>,
}
impl Budget {
    fn defer(&mut self, now: Instant, interval: Duration) -> bool {
        if let Some(deadline) = self
            .last
            .map(|last| last + interval)
            .filter(|deadline| *deadline > now)
        {
            self.pending = Some(deadline);
            true
        } else {
            self.pending = None;
            // Keep the original cadence when a wake-up is slightly late. Starting
            // a new interval at each wake-up would accumulate timer/render jitter.
            self.last = Some(match self.last {
                Some(last) => {
                    let steps = (now.duration_since(last).as_nanos() / interval.as_nanos()).max(1);
                    last + interval.mul_f64(steps as f64)
                }
                None => now,
            });
            false
        }
    }
}
struct PacedApp<'a> {
    inner: eframe::EframeWinitApplication<'a>,
    budgets: HashMap<WindowId, Budget>,
}
impl PacedApp<'_> {
    fn repaint_due(&mut self) {
        let now = Instant::now();
        let windows = WINDOWS.lock().unwrap_or_else(|e| e.into_inner());
        for (id, budget) in &mut self.budgets {
            if budget.pending.is_some_and(|deadline| deadline <= now) {
                budget.pending = None;
                if let Some(window) = windows
                    .as_ref()
                    .and_then(|windows| windows.get(id))
                    .and_then(|info| info.window.upgrade())
                {
                    window.request_redraw();
                }
            }
        }
    }
}
impl ApplicationHandler<eframe::UserEvent> for PacedApp<'_> {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        self.inner.resumed(el);
    }
    fn suspended(&mut self, el: &ActiveEventLoop) {
        self.inner.suspended(el);
    }
    fn exiting(&mut self, el: &ActiveEventLoop) {
        self.inner.exiting(el);
        *WINDOWS.lock().unwrap_or_else(|e| e.into_inner()) = None;
    }
    fn memory_warning(&mut self, el: &ActiveEventLoop) {
        self.inner.memory_warning(el);
    }
    fn device_event(&mut self, el: &ActiveEventLoop, id: DeviceId, event: DeviceEvent) {
        self.inner.device_event(el, id, event);
    }
    fn user_event(&mut self, el: &ActiveEventLoop, event: eframe::UserEvent) {
        self.inner.user_event(el, event);
    }
    fn new_events(&mut self, el: &ActiveEventLoop, cause: StartCause) {
        self.inner.new_events(el, cause);
        self.repaint_due();
    }
    fn window_event(&mut self, el: &ActiveEventLoop, id: WindowId, event: WindowEvent) {
        if matches!(event, WindowEvent::RedrawRequested) {
            let refresh = WINDOWS
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .as_mut()
                .and_then(|windows| windows.get_mut(&id))
                .map(|info| {
                    if info.checked.elapsed() >= Duration::from_secs(1) {
                        info.refresh = info
                            .window
                            .upgrade()
                            .and_then(|window| window.current_monitor())
                            .and_then(|m| m.refresh_rate_millihertz());
                        info.checked = Instant::now();
                    }
                    info.refresh
                });
            if let Some(refresh) = refresh {
                let policy = *POLICY.lock().unwrap_or_else(|e| e.into_inner());
                if self
                    .budgets
                    .entry(id)
                    .or_default()
                    .defer(Instant::now(), interval(policy, refresh))
                {
                    return;
                }
            }
        }
        if matches!(
            event,
            WindowEvent::Moved(_) | WindowEvent::ScaleFactorChanged { .. }
        ) && let Some(info) = WINDOWS
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_mut()
            .and_then(|w| w.get_mut(&id))
        {
            info.checked = Instant::now() - Duration::from_secs(2);
        }
        if matches!(event, WindowEvent::Destroyed) {
            self.budgets.remove(&id);
        }
        // In particular, never discard/coalesce PointerMoved, pressure, wheel,
        // key or button-release events: the stroke path consumes every sample.
        self.inner.window_event(el, id, event);
    }
    fn about_to_wait(&mut self, el: &ActiveEventLoop) {
        self.inner.about_to_wait(el);
        if let Some(deadline) = self.budgets.values().filter_map(|b| b.pending).min() {
            let deadline = match el.control_flow() {
                ControlFlow::WaitUntil(other) => deadline.min(other),
                _ => deadline,
            };
            el.set_control_flow(ControlFlow::WaitUntil(deadline));
        }
    }
}

pub fn run_native<'a>(
    name: &str,
    mut options: eframe::NativeOptions,
    creator: eframe::AppCreator<'a>,
) -> eframe::Result {
    let mut builder = EventLoop::<eframe::UserEvent>::with_user_event();
    // create_native receives an existing loop, so preserve PaintFE's Windows
    // WM_CHAR/WM_SYSCHAR hook explicitly rather than letting it be skipped.
    if let Some(hook) = options.event_loop_builder.take() {
        hook(&mut builder);
    }
    let event_loop = builder.build()?;
    let inner = eframe::create_native(name, options, creator, &event_loop);
    event_loop.run_app(&mut PacedApp {
        inner,
        budgets: HashMap::new(),
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn scheduling_coalesces_redraws_without_running_an_idle_timer() {
        let now = Instant::now();
        let step = Duration::from_millis(20);
        let mut b = Budget::default();
        assert!(!b.defer(now, step));
        assert!(b.pending.is_none());
        for ms in 1..20 {
            assert!(b.defer(now + Duration::from_millis(ms), step));
            assert_eq!(b.pending, Some(now + step));
        }
        assert!(!b.defer(now + step, step));
        assert!(b.pending.is_none());
        assert!(!b.defer(now + Duration::from_secs(1), step));
    }
    #[test]
    fn late_wakeups_do_not_accumulate_frame_rate_drift_or_catch_up_bursts() {
        let start = Instant::now();
        let interval = Duration::from_nanos(16_666_667);
        let mut budget = Budget::default();
        assert!(!budget.defer(start, interval));
        for frame in 1..=1000 {
            let late = start + interval * frame + Duration::from_micros(550);
            assert!(!budget.defer(late, interval));
            assert_eq!(budget.last, Some(start + interval * frame));
            assert!(budget.defer(late + Duration::from_micros(100), interval));
            assert_eq!(budget.pending, Some(start + interval * (frame + 1)));
        }
        let late = start + interval * 1100 + Duration::from_micros(500);
        assert!(!budget.defer(late, interval));
        assert!(budget.defer(late, interval));
    }
    #[test]
    fn monitor_detection_manual_caps_and_missing_monitor_are_bounded() {
        assert_eq!(
            interval(
                Policy {
                    detect: false,
                    fps: 30
                },
                Some(144_000)
            ),
            Duration::from_secs_f64(1.0 / 30.0)
        );
        assert_eq!(
            interval(
                Policy {
                    detect: true,
                    fps: 30
                },
                Some(144_000)
            ),
            Duration::from_secs_f64(1.0 / 144.0)
        );
        assert_eq!(
            interval(
                Policy {
                    detect: true,
                    fps: 30
                },
                None
            ),
            Duration::from_secs_f64(1.0 / 60.0)
        );
        assert_eq!(
            interval(
                Policy {
                    detect: true,
                    fps: 30
                },
                Some(0)
            ),
            Duration::from_secs_f64(1.0 / 60.0)
        );
    }
}
