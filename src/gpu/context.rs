// ============================================================================
// GPU CONTEXT — wgpu Device, Queue, and adapter initialization
// ============================================================================

use std::sync::Arc;

/// Holds the core wgpu resources shared across the entire application.
/// Created once at startup; if creation fails we fall back to CPU rendering.
pub struct GpuContext {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub adapter_name: String,
    /// Maximum texture dimension supported by this device.
    pub max_texture_dim: u32,
}

impl GpuContext {
    /// Attempt to create a GPU context.  Tries hardware first, then falls
    /// back to a software rasterizer (`force_fallback_adapter`) so rendering
    /// always works even without a real GPU.
    ///
    /// We use `pollster::block_on` because eframe doesn't expose its wgpu
    /// device to application code and we need our own for compute + offscreen
    /// composition.
    pub fn new(preferred_gpu: &str) -> Option<Self> {
        // 1. Try hardware adapter.
        if let Some(ctx) = pollster::block_on(Self::new_async(preferred_gpu, false)) {
            return Some(ctx);
        }
        // 2. Fallback: software rasterizer.
        eprintln!("[GPU] Hardware adapter unavailable — trying software fallback");
        pollster::block_on(Self::new_async(preferred_gpu, true))
    }

    async fn new_async(preferred_gpu: &str, force_fallback: bool) -> Option<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        // Pick power preference from settings string.
        let power = match preferred_gpu.to_lowercase().as_str() {
            "low power" | "integrated" => wgpu::PowerPreference::LowPower,
            "high performance" | "discrete" => wgpu::PowerPreference::HighPerformance,
            _ => wgpu::PowerPreference::HighPerformance, // default to fast
        };

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: power,
                compatible_surface: None, // headless — we only need compute + offscreen
                force_fallback_adapter: force_fallback,
            })
            .await?;

        let adapter_name = adapter.get_info().name.clone();
        let limits = adapter.limits();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("PaintFE GPU"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits {
                        max_texture_dimension_2d: limits.max_texture_dimension_2d,
                        max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
                        max_compute_workgroup_size_x: limits.max_compute_workgroup_size_x,
                        max_compute_workgroup_size_y: limits.max_compute_workgroup_size_y,
                        max_compute_workgroup_size_z: limits.max_compute_workgroup_size_z,
                        max_compute_workgroups_per_dimension: limits.max_compute_workgroups_per_dimension,
                        ..wgpu::Limits::downlevel_defaults()
                    },
                },
                None,
            )
            .await
            .ok()?;

        // wgpu 0.20: device loss is handled by polling.  We just proceed.

        Some(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            adapter_name,
            max_texture_dim: limits.max_texture_dimension_2d,
        })
    }

    /// Check if a texture of the given dimensions can be created.
    pub fn supports_size(&self, width: u32, height: u32) -> bool {
        width <= self.max_texture_dim && height <= self.max_texture_dim
    }

    /// Submit a single encoder's commands.
    pub fn submit_one(&self, encoder: wgpu::CommandEncoder) {
        self.queue.submit(std::iter::once(encoder.finish()));
    }
}
