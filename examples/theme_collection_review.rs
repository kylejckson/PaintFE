//! Isolated production-app review, without the normal single-instance listener.
fn main() -> eframe::Result {
    paintfe::i18n::init();
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.first().is_some_and(|arg| arg == "--export-icons") {
        let ctx = egui::Context::default();
        let mut assets = paintfe::config::brushes::Assets::new();
        assets.init(&ctx);
        assets
            .export_icon_template(std::path::Path::new(&args[1]))
            .expect("Export icon inventory");
        return Ok(());
    }
    let files = args.into_iter().map(std::path::PathBuf::from).collect();
    let (_sender, receiver) = std::sync::mpsc::channel();
    let mut wgpu_options = eframe::egui_wgpu::WgpuConfiguration::default();
    if let eframe::egui_wgpu::WgpuSetup::CreateNew(setup) = &mut wgpu_options.wgpu_setup {
        setup.instance_descriptor.backends = if cfg!(target_os = "windows") {
            wgpu::Backends::VULKAN
        } else {
            wgpu::Backends::all()
        };
    }
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1360., 880.])
            .with_position([180., 80.])
            .with_transparent(true)
            .with_app_id("io.github.paintfe.ThemeCollectionReview"),
        wgpu_options,
        ..Default::default()
    };
    paintfe::services::frame_pacing::set_policy(false, 60);
    paintfe::services::frame_pacing::run_native(
        "PaintFEThemeCollectionReview",
        options,
        Box::new(move |cc| {
            if let Some(window) = cc.winit_window() {
                paintfe::services::frame_pacing::register_window(window.clone());
            }
            if let Some(state) = &cc.wgpu_render_state {
                paintfe::services::graphics::record_render_state(state);
            }
            Ok(Box::new(paintfe::app::PaintFEApp::new(cc, files, receiver)))
        }),
    )
}
