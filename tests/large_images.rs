use paintfe::{canvas::CanvasState, gpu::GpuRenderer};
#[test]
fn oversized_layer_uploads_are_rejected_before_wgpu_validation() {
    let Some(mut gpu) = GpuRenderer::try_new("Auto") else {
        eprintln!("GPU unavailable");
        return;
    };
    gpu.ctx.max_texture_dim = 8192;
    assert!(gpu.ctx.supports_size(8192, 2));
    for width in [8193, 8736, 25000] {
        let data = vec![255; width as usize * 2 * 4];
        gpu.ensure_layer_texture(0, width, 2, &data, 1);
        assert!(!gpu.layer_has_texture(0, width, 2));
        assert!(gpu.composite(width, 2, &[(0, 1.0, true, 0)]).is_none());
    }
    assert!(!gpu.ctx.supports_size(8192, 8192));
    assert!(!gpu.ctx.supports_size(0, 1));
}
#[test]
fn large_document_adjustments_use_cpu_and_preserve_alpha_and_size() {
    let Some(mut gpu) = GpuRenderer::try_new("Auto") else {
        eprintln!("GPU unavailable");
        return;
    };
    gpu.ctx.max_texture_dim = 8192;
    let mut state = CanvasState::new(8736, 2);
    state.layers[0]
        .pixels
        .put_pixel(8735, 1, image::Rgba([10, 20, 30, 128]));
    paintfe::ops::adjustments::invert_colors_gpu(&mut state, 0, &gpu);
    assert_eq!(
        *state.layers[0].pixels.get_pixel(8735, 1),
        image::Rgba([245, 235, 225, 128])
    );
    assert_eq!((state.width, state.height), (8736, 2));
}
