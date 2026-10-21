//! Lazy ownership of a pipeline, with optional native background preparation.
use std::{
    ops::{Deref, DerefMut},
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
};

type Pending<T> = (mpsc::Receiver<T>, Arc<AtomicBool>);
pub struct LazyPipeline<T> {
    value: OnceLock<T>,
    device: Arc<wgpu::Device>,
    create: fn(&wgpu::Device) -> T,
    pending: Mutex<Option<Pending<T>>>,
}
impl<T> LazyPipeline<T> {
    pub fn new(device: Arc<wgpu::Device>, create: fn(&wgpu::Device) -> T) -> Self {
        Self {
            value: OnceLock::new(),
            device,
            create,
            pending: Mutex::new(None),
        }
    }
    fn create(&self) -> T {
        let _profile = crate::ui::perf::StartupSpan::new(std::any::type_name::<T>());
        (self.create)(&self.device)
    }
    pub fn is_initialized(&self) -> bool {
        self.value.get().is_some()
    }
}
#[cfg(not(target_arch = "wasm32"))]
impl<T: Send + 'static> LazyPipeline<T> {
    pub(crate) fn background_job(&self) -> Option<Box<dyn FnOnce() + Send>> {
        if self.value.get().is_some() {
            return None;
        }
        let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        if pending.is_some() {
            return None;
        }
        let (sender, receiver) = mpsc::channel();
        let claimed = Arc::new(AtomicBool::new(false));
        *pending = Some((receiver, claimed.clone()));
        let device = self.device.clone();
        let create = self.create;
        Some(Box::new(move || {
            // First use can claim a queued job rather than wait for its turn.
            if !claimed.swap(true, Ordering::AcqRel) {
                let _profile = crate::ui::perf::StartupSpan::new(std::any::type_name::<T>());
                let _ = sender.send(create(&device));
            }
        }))
    }
}
impl<T> Deref for LazyPipeline<T> {
    type Target = T;
    fn deref(&self) -> &T {
        self.value.get_or_init(|| {
            let pending = self
                .pending
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .take();
            if let Some((receiver, claimed)) = pending
                && claimed.swap(true, Ordering::AcqRel)
            {
                return receiver.recv().unwrap_or_else(|_| self.create());
            }
            self.create()
        })
    }
}
impl<T> DerefMut for LazyPipeline<T> {
    fn deref_mut(&mut self) -> &mut T {
        let _ = <Self as Deref>::deref(self);
        self.value.get_mut().unwrap()
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    #[test]
    fn lazy_first_use_background_transfer_and_mutation_keep_one_owned_value() {
        let Some(ctx) = crate::gpu::context::GpuContext::new("") else {
            eprintln!("SKIP: no GPU adapter available for pipeline ownership test");
            return;
        };
        static CREATED: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);
        fn create(_: &wgpu::Device) -> Vec<u8> {
            vec![CREATED.fetch_add(1, Ordering::SeqCst) + 1]
        }
        let mut pipeline = LazyPipeline::new(ctx.device.clone(), create);
        assert!(!pipeline.is_initialized());
        let queued = pipeline.background_job().unwrap();
        assert!(pipeline.background_job().is_none());
        pipeline.push(8); // UI claims a queued job before its worker runs.
        queued();
        assert_eq!(&**pipeline, &[1, 8]);
        assert!(pipeline.background_job().is_none());
        let mut background = LazyPipeline::new(ctx.device.clone(), create);
        background.background_job().unwrap()();
        background.push(9); // Worker transfers ownership; no separate stale copy.
        assert_eq!(&**background, &[2, 9]);
        let failed = LazyPipeline::new(ctx.device, create);
        drop(failed.background_job());
        assert_eq!(&**failed, &[3]);
        assert_eq!(CREATED.load(Ordering::SeqCst), 3);
    }
}
