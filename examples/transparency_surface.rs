//! Opt-in native surface capability diagnostic. Creates one hidden disposable window.
#[cfg(target_os = "windows")]
fn main() {
    use std::sync::Arc;
    use winit::{
        application::ApplicationHandler,
        event::WindowEvent,
        event_loop::{ActiveEventLoop, EventLoop},
        window::{Window, WindowId},
    };
    struct Probe;
    impl ApplicationHandler for Probe {
        fn resumed(&mut self, event_loop: &ActiveEventLoop) {
            let window = Arc::new(
                event_loop
                    .create_window(
                        Window::default_attributes()
                            .with_visible(false)
                            .with_transparent(true),
                    )
                    .unwrap(),
            );
            for (label, backend, kind) in [
                (
                    "vulkan",
                    wgpu::Backends::VULKAN,
                    wgpu::Dx12SwapchainKind::DxgiFromHwnd,
                ),
                (
                    "dx12-hwnd",
                    wgpu::Backends::DX12,
                    wgpu::Dx12SwapchainKind::DxgiFromHwnd,
                ),
                (
                    "dx12-composition",
                    wgpu::Backends::DX12,
                    wgpu::Dx12SwapchainKind::DxgiFromVisual,
                ),
            ] {
                let mut descriptor = wgpu::InstanceDescriptor {
                    backends: backend,
                    ..wgpu::InstanceDescriptor::new_without_display_handle()
                };
                descriptor.backend_options.dx12.presentation_system = kind;
                let instance = wgpu::Instance::new(descriptor);
                match instance.create_surface(window.clone()) {
                    Ok(surface) => match pollster::block_on(instance.request_adapter(
                        &wgpu::RequestAdapterOptions {
                            compatible_surface: Some(&surface),
                            ..Default::default()
                        },
                    )) {
                        Ok(adapter) => println!(
                            "{label}: backend={:?} alpha_modes={:?}",
                            adapter.get_info().backend,
                            surface.get_capabilities(&adapter).alpha_modes
                        ),
                        Err(e) => println!("{label}: no compatible adapter: {e}"),
                    },
                    Err(e) => println!("{label}: surface unavailable: {e}"),
                }
            }
            event_loop.exit();
        }
        fn window_event(&mut self, _: &ActiveEventLoop, _: WindowId, _: WindowEvent) {}
    }
    EventLoop::new().unwrap().run_app(&mut Probe).unwrap();
}
#[cfg(not(target_os = "windows"))]
fn main() {}
