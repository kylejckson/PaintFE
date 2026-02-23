// ============================================================================
// GPU MODULE — Hardware-accelerated rendering and compute for PaintFE
// ============================================================================
//
// Architecture:
//   context.rs    — wgpu Device, Queue, adapter init
//   shaders.rs    — all WGSL shader source (inline strings)
//   texture.rs    — LayerTexture wrapper with partial upload + mipmaps
//   compositor.rs — render pipeline for alpha-blended layer composition
//   compute.rs    — compute pipeline for Gaussian blur (two-pass separable)
//   pool.rs       — texture recycling pool
//   renderer.rs   — top-level GpuRenderer coordinator
// ============================================================================

pub mod context;
pub mod shaders;
pub mod texture;
pub mod compositor;
pub mod compute;
pub mod pool;
pub mod renderer;

pub use renderer::GpuRenderer;
pub use compute::GradientGpuParams;

/// WGPU requires `bytes_per_row` to be a multiple of 256.
/// This constant is used for aligning dirty rects.
pub const COPY_BYTES_PER_ROW_ALIGNMENT: u32 = 256;

/// Align a dirty rect so that `width * 4` is a multiple of 256 bytes.
/// Expands the rect rightward (and clamps to texture bounds).
///
/// Returns (x, y, aligned_width, height).
pub fn align_dirty_rect(x: u32, y: u32, w: u32, h: u32, texture_width: u32, texture_height: u32) -> (u32, u32, u32, u32) {
    // bytes_per_row = width * 4 must be multiple of 256
    // So width must be multiple of 64 (256 / 4)
    const PIXEL_ALIGNMENT: u32 = 64; // 256 / 4

    let clamped_x = x.min(texture_width.saturating_sub(1));
    let clamped_y = y.min(texture_height.saturating_sub(1));

    // How much space do we have to the right edge?
    let max_width = texture_width - clamped_x;
    let max_height = texture_height - clamped_y;

    let clamped_w = w.min(max_width);
    let clamped_h = h.min(max_height);

    if clamped_w == 0 || clamped_h == 0 {
        return (clamped_x, clamped_y, 0, 0);
    }

    // Align width up to next multiple of PIXEL_ALIGNMENT
    let aligned_w = ((clamped_w + PIXEL_ALIGNMENT - 1) / PIXEL_ALIGNMENT) * PIXEL_ALIGNMENT;

    // Clamp to texture bounds
    let final_w = aligned_w.min(max_width);
    let final_h = clamped_h;

    (clamped_x, clamped_y, final_w, final_h)
}
