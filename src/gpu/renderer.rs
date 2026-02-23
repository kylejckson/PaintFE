// ============================================================================
// GPU RENDERER — top-level coordinator for GPU-accelerated rendering
// ============================================================================

use std::collections::HashMap;
use egui;

use super::context::GpuContext;
use super::compositor::Compositor;

// ============================================================================
// ASYNC GPU READBACK — double-buffered staging for stall-free rendering
// ============================================================================

/// Metadata about what's in a pending readback buffer.
pub(crate) struct ReadbackMeta {
    rx: u32,
    ry: u32,
    rw: u32,
    rh: u32,
    is_full: bool,
    padded_bytes_per_row: u32,
}

/// Double-buffered async GPU readback.  Eliminates `device.poll(Maintain::Wait)`
/// stalls by reading from the PREVIOUS frame's staging buffer while the GPU
/// writes to the current frame's staging buffer.
///
/// Flow per frame:
///   1. `try_read()` — non-blocking poll, read previous frame's data if ready
///   2. GPU composite + copy to `write_buffer()`
///   3. `submit_and_swap()` — map_async on write buffer, swap indices
pub struct AsyncReadback {
    buffers: [Option<wgpu::Buffer>; 2],
    write_idx: usize,
    buf_size: u64,
    /// Whether the read buffer (buffers[1-write_idx]) has a pending map_async
    pub read_pending: bool,
    read_rx: Option<std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>,
    read_meta: Option<ReadbackMeta>,
}

impl AsyncReadback {
    pub fn new() -> Self {
        Self {
            buffers: [None, None],
            write_idx: 0,
            buf_size: 0,
            read_pending: false,
            read_rx: None,
            read_meta: None,
        }
    }

    /// Ensure both staging buffers exist with at least `size` bytes.
    pub fn ensure_buffers(&mut self, device: &wgpu::Device, size: u64) {
        if self.buf_size >= size && self.buffers[0].is_some() && self.buffers[1].is_some() {
            return;
        }
        // If we have a pending read, we can't recreate buffers safely.
        // Cancel the pending read first.
        if self.read_pending {
            // The old buffers will be dropped — any pending map is cancelled.
            self.read_pending = false;
            self.read_rx = None;
            self.read_meta = None;
        }
        for i in 0..2 {
            self.buffers[i] = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(if i == 0 { "async_readback_0" } else { "async_readback_1" }),
                size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }));
        }
        self.buf_size = size;
    }

    /// The staging buffer that the GPU should copy into this frame.
    pub fn write_buffer(&self) -> &wgpu::Buffer {
        self.buffers[self.write_idx].as_ref().unwrap()
    }

    fn read_idx(&self) -> usize {
        1 - self.write_idx
    }

    /// Call after `queue.submit()` with the copy command.  Starts `map_async`
    /// on the write buffer and swaps write/read indices.
    pub fn submit_and_swap(&mut self, meta: ReadbackMeta) {
        let widx = self.write_idx;
        let buf = self.buffers[widx].as_ref().unwrap();
        let slice = buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        self.write_idx = 1 - widx;
        self.read_pending = true;
        self.read_rx = Some(rx);
        self.read_meta = Some(meta);
    }

    /// Non-blocking read of the previous frame's data.  Calls
    /// `device.poll(Poll)` to pump callbacks.  Returns packed RGBA pixels
    /// (alignment padding stripped) if the mapping has completed.
    pub fn try_read(&mut self, device: &wgpu::Device) -> Option<(Vec<u8>, u32, u32, u32, u32, bool)> {
        if !self.read_pending {
            return None;
        }

        device.poll(wgpu::Maintain::Poll);

        let ready = self.read_rx.as_ref()
            .and_then(|rx| rx.try_recv().ok())
            .is_some();

        if !ready {
            return None;
        }

        let ridx = self.read_idx();
        let buf = self.buffers[ridx].as_ref().unwrap();
        let meta = self.read_meta.as_ref().unwrap();

        let slice = buf.slice(..);
        let mapped = slice.get_mapped_range();
        let padded = meta.padded_bytes_per_row as usize;
        let tight = meta.rw as usize * 4;
        let mut result = Vec::with_capacity(tight * meta.rh as usize);
        for row in 0..meta.rh as usize {
            let start = row * padded;
            result.extend_from_slice(&mapped[start..start + tight]);
        }
        drop(mapped);
        buf.unmap();

        let info = (result, meta.rx, meta.ry, meta.rw, meta.rh, meta.is_full);
        self.read_pending = false;
        self.read_rx = None;
        self.read_meta = None;

        Some(info)
    }

    /// Cancel any pending async readback without reading data.
    /// Call this when switching to sync readback to prevent stale data
    /// from being applied on a subsequent frame.
    pub fn cancel_pending(&mut self) {
        if self.read_pending {
            // Dropping the receiver cancels any pending map_async callback.
            let _ = self.read_rx.take();
            self.read_meta = None;
            self.read_pending = false;
        }
    }
}
use super::compute::{
    GpuBlurPipeline, GpuBrightnessContrastPipeline,
    GpuGradientPipeline, GpuHslPipeline, GpuInvertPipeline, GpuLiquifyPipeline, GpuMedianPipeline,
    GpuMeshWarpDisplacementPipeline,
};
use super::texture::{LayerTexture, MipmapPipeline};
use super::pool::TexturePool;

/// Tracks which region of a layer is dirty and needs re-upload.
#[derive(Clone, Debug)]
pub struct DirtyRegion {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

/// Per-layer GPU state, keyed by layer index.
struct GpuLayerState {
    texture: LayerTexture,
    generation: u64,
}

/// The top-level GPU renderer.
pub struct GpuRenderer {
    pub ctx: GpuContext,
    pub compositor: Compositor,
    pub blur_pipeline: GpuBlurPipeline,
    pub bc_pipeline: GpuBrightnessContrastPipeline,
    pub hsl_pipeline: GpuHslPipeline,
    pub invert_pipeline: GpuInvertPipeline,
    pub median_pipeline: GpuMedianPipeline,
    pub gradient_pipeline: GpuGradientPipeline,
    pub liquify_pipeline: GpuLiquifyPipeline,
    pub mesh_warp_disp_pipeline: GpuMeshWarpDisplacementPipeline,
    pub mipmap_pipeline: MipmapPipeline,
    pub texture_pool: TexturePool,
    layer_textures: HashMap<usize, GpuLayerState>,
    /// Output (composited) texture — recycled between frames.
    output_texture: Option<wgpu::Texture>,
    /// Ping-pong pair for uber-compositor.
    ping_pong: [Option<wgpu::Texture>; 2],
    pp_width: u32,
    pp_height: u32,
    output_width: u32,
    output_height: u32,
    /// Cached GPU staging buffer for readback (avoids 33MB alloc per frame)
    cached_staging_buf: Option<(wgpu::Buffer, u64)>,
    /// Double-buffered async readback for stall-free compositing
    pub async_readback: AsyncReadback,
    pub available: bool,
}

impl GpuRenderer {
    /// Create a GPU renderer.  Tries hardware first, then software fallback.
    /// Panics only if even the software rasterizer is unavailable (should not
    /// happen on any modern OS).
    pub fn new(preferred_gpu: &str) -> Self {
        Self::try_new(preferred_gpu)
            .expect("GpuRenderer: neither hardware nor software adapter available")
    }

    pub fn try_new(preferred_gpu: &str) -> Option<Self> {
        let ctx = GpuContext::new(preferred_gpu)?;
        let compositor = Compositor::new(&ctx.device);
        let blur_pipeline = GpuBlurPipeline::new(&ctx.device);
        let bc_pipeline = GpuBrightnessContrastPipeline::new(&ctx.device);
        let hsl_pipeline = GpuHslPipeline::new(&ctx.device);
        let invert_pipeline = GpuInvertPipeline::new(&ctx.device);
        let median_pipeline = GpuMedianPipeline::new(&ctx.device);
        let gradient_pipeline = GpuGradientPipeline::new(&ctx.device);
        let liquify_pipeline = GpuLiquifyPipeline::new(&ctx.device);
        let mesh_warp_disp_pipeline = GpuMeshWarpDisplacementPipeline::new(&ctx.device);
        let mipmap_pipeline = MipmapPipeline::new(&ctx.device);

        Some(Self {
            ctx,
            compositor,
            blur_pipeline,
            bc_pipeline,
            hsl_pipeline,
            invert_pipeline,
            median_pipeline,
            gradient_pipeline,
            liquify_pipeline,
            mesh_warp_disp_pipeline,
            mipmap_pipeline,
            texture_pool: TexturePool::new(),
            layer_textures: HashMap::new(),
            output_texture: None,
            ping_pong: [None, None],
            pp_width: 0,
            pp_height: 0,
            output_width: 0,
            output_height: 0,
            cached_staging_buf: None,
            async_readback: AsyncReadback::new(),
            available: true,
        })
    }

    pub fn adapter_name(&self) -> &str {
        &self.ctx.adapter_name
    }

    // ========================================================================
    // LAYER SYNCHRONISATION
    // ========================================================================

    pub fn ensure_layer_texture(
        &mut self,
        layer_idx: usize,
        width: u32,
        height: u32,
        data: &[u8],
        generation: u64,
    ) {
        if !self.available { return; }

        if let Some(state) = self.layer_textures.get(&layer_idx) {
            if state.generation == generation
                && state.texture.width == width
                && state.texture.height == height
            {
                return;
            }
        }

        let mip_levels = LayerTexture::mip_level_count(width, height);
        let texture = if let Some(recycled) = self.texture_pool.acquire(width, height, mip_levels) {
            let view = recycled.create_view(&wgpu::TextureViewDescriptor::default());
            let sampler = self.compositor.sampler_for_zoom(1.0);
            let bind_group = self.ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("recycled_layer_bg"),
                layout: &self.compositor.texture_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&view) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(sampler) },
                ],
            });
            let lt = LayerTexture {
                texture: recycled,
                view,
                bind_group,
                width,
                height,
                mip_levels,
            };
            lt.upload_full(&self.ctx.queue, data);
            self.mipmap_pipeline.generate(
                &self.ctx.device, &self.ctx.queue,
                &lt.texture, width, height, mip_levels,
            );
            lt
        } else {
            let sampler = self.compositor.sampler_for_zoom(1.0);
            LayerTexture::new(
                &self.ctx.device,
                &self.ctx.queue,
                &self.compositor.texture_bind_group_layout,
                sampler,
                width, height, data,
                Some(&self.mipmap_pipeline),
            )
        };

        if let Some(old) = self.layer_textures.remove(&layer_idx) {
            self.texture_pool.release(
                old.texture.texture,
                old.texture.width,
                old.texture.height,
                old.texture.mip_levels,
            );
        }

        self.layer_textures.insert(layer_idx, GpuLayerState {
            texture,
            generation,
        });
    }

    pub fn update_layer_rect(
        &mut self,
        layer_idx: usize,
        region: &DirtyRegion,
        data: &[u8],
    ) {
        if !self.available { return; }
        if let Some(state) = self.layer_textures.get(&layer_idx) {
            state.texture.update_rect(
                &self.ctx.queue,
                region.x, region.y,
                region.width, region.height,
                data,
            );
        }
    }

    /// Check whether a texture exists for `layer_idx` with a matching generation.
    pub fn layer_is_current(&self, layer_idx: usize, generation: u64) -> bool {
        self.layer_textures.get(&layer_idx).map_or(false, |s| s.generation == generation)
    }

    /// Check whether a texture exists for `layer_idx` at the expected size,
    /// regardless of generation (i.e. it can receive a partial update).
    pub fn layer_has_texture(&self, layer_idx: usize, w: u32, h: u32) -> bool {
        self.layer_textures.get(&layer_idx).map_or(false, |s| {
            s.texture.width == w && s.texture.height == h
        })
    }

    /// Update the stored generation counter for an existing layer texture
    /// (used after a partial `update_layer_rect` so the next frame knows
    /// the texture is up-to-date).
    pub fn set_layer_generation(&mut self, layer_idx: usize, generation: u64) {
        if let Some(state) = self.layer_textures.get_mut(&layer_idx) {
            state.generation = generation;
        }
    }

    pub fn remove_layer(&mut self, layer_idx: usize) {
        if let Some(state) = self.layer_textures.remove(&layer_idx) {
            self.texture_pool.release(
                state.texture.texture,
                state.texture.width,
                state.texture.height,
                state.texture.mip_levels,
            );
        }
    }

    /// Remove a layer and shift all higher-indexed textures down by one,
    /// so GPU texture indices stay in sync with the `Vec<Layer>` after
    /// `Vec::remove(del_idx)`.  Also invalidates generations so the next
    /// `ensure_layer_texture` re-uploads shifted layers.
    pub fn remove_layer_and_reindex(&mut self, del_idx: usize) {
        // 1. Recycle the deleted layer's texture.
        self.remove_layer(del_idx);

        // 2. Shift all entries above del_idx down by one.
        let mut keys_above: Vec<usize> = self.layer_textures
            .keys()
            .copied()
            .filter(|&k| k > del_idx)
            .collect();
        keys_above.sort_unstable();

        for old_key in keys_above {
            if let Some(mut state) = self.layer_textures.remove(&old_key) {
                // Invalidate generation so next ensure_layer_texture re-uploads.
                state.generation = u64::MAX;
                self.layer_textures.insert(old_key - 1, state);
            }
        }
    }

    pub fn clear_layers(&mut self) {
        let keys: Vec<_> = self.layer_textures.keys().cloned().collect();
        for k in keys {
            self.remove_layer(k);
        }
    }

    // ========================================================================
    // PING-PONG TEXTURE MANAGEMENT
    // ========================================================================

    fn ensure_ping_pong(&mut self, w: u32, h: u32) {
        if self.pp_width == w && self.pp_height == h && self.ping_pong[0].is_some() {
            return;
        }
        let usage = wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC;
        for i in 0..2 {
            self.ping_pong[i] = Some(self.ctx.device.create_texture(&wgpu::TextureDescriptor {
                label: Some(if i == 0 { "ping" } else { "pong" }),
                size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: self.compositor.output_format,
                usage,
                view_formats: &[],
            }));
        }
        self.pp_width = w;
        self.pp_height = h;
    }

    // ========================================================================
    // COMPOSITION (with blend modes)
    // ========================================================================

    /// Composite all visible layers with full blend-mode support.
    ///
    /// `layer_info`: `(layer_idx, opacity, visible, blend_mode_u8)`.
    pub fn composite(
        &mut self,
        canvas_w: u32,
        canvas_h: u32,
        layer_info: &[(usize, f32, bool, u8)],
    ) -> Option<Vec<u8>> {
        if !self.available { return None; }

        self.ensure_ping_pong(canvas_w, canvas_h);

        let pp0 = self.ping_pong[0].as_ref().unwrap();
        let pp1 = self.ping_pong[1].as_ref().unwrap();
        let view0 = pp0.create_view(&wgpu::TextureViewDescriptor::default());
        let view1 = pp1.create_view(&wgpu::TextureViewDescriptor::default());

        // Collect visible layers.
        let mut visible_layers: Vec<(f32, u32, &LayerTexture)> = Vec::new();
        for &(idx, opacity, visible, blend_mode) in layer_info.iter() {
            if !visible { continue; }
            if let Some(state) = self.layer_textures.get(&idx) {
                visible_layers.push((opacity, blend_mode as u32, &state.texture));
            }
        }

        let result_idx = self.compositor.composite_layers_blended(
            &self.ctx,
            [&view0, &view1],
            &visible_layers,
            canvas_w,
            canvas_h,
        );

        // Read back from whichever texture holds the result.
        let result_tex = if result_idx == 0 {
            self.ping_pong[0].as_ref().unwrap()
        } else {
            self.ping_pong[1].as_ref().unwrap()
        };

        Some(Compositor::readback_texture(
            &self.ctx,
            result_tex,
            canvas_w,
            canvas_h,
            &mut self.cached_staging_buf,
        ))
    }

    /// Composite all visible layers on GPU, then readback only the dirty region.
    /// Returns `(full_or_region_pixels, region_x, region_y, region_w, region_h, is_full)`.
    /// If `dirty_rect` is None or covers the full canvas, does a full readback.
    pub fn composite_dirty_readback(
        &mut self,
        canvas_w: u32,
        canvas_h: u32,
        layer_info: &[(usize, f32, bool, u8)],
        dirty_rect: Option<egui::Rect>,
    ) -> Option<(Vec<u8>, u32, u32, u32, u32, bool)> {
        if !self.available { return None; }

        self.ensure_ping_pong(canvas_w, canvas_h);

        let pp0 = self.ping_pong[0].as_ref().unwrap();
        let pp1 = self.ping_pong[1].as_ref().unwrap();
        let view0 = pp0.create_view(&wgpu::TextureViewDescriptor::default());
        let view1 = pp1.create_view(&wgpu::TextureViewDescriptor::default());

        let mut visible_layers: Vec<(f32, u32, &LayerTexture)> = Vec::new();
        for &(idx, opacity, visible, blend_mode) in layer_info.iter() {
            if !visible { continue; }
            if let Some(state) = self.layer_textures.get(&idx) {
                visible_layers.push((opacity, blend_mode as u32, &state.texture));
            }
        }

        let result_idx = self.compositor.composite_layers_blended(
            &self.ctx,
            [&view0, &view1],
            &visible_layers,
            canvas_w,
            canvas_h,
        );

        let result_tex = if result_idx == 0 {
            self.ping_pong[0].as_ref().unwrap()
        } else {
            self.ping_pong[1].as_ref().unwrap()
        };

        // Determine if we can do a partial readback
        if let Some(dr) = dirty_rect {
            let rx = (dr.min.x.floor() as u32).max(0).min(canvas_w);
            let ry = (dr.min.y.floor() as u32).max(0).min(canvas_h);
            let rx2 = (dr.max.x.ceil() as u32).min(canvas_w);
            let ry2 = (dr.max.y.ceil() as u32).min(canvas_h);
            let rw = rx2.saturating_sub(rx);
            let rh = ry2.saturating_sub(ry);

            // Only do partial readback if the dirty rect is meaningfully smaller
            let full_pixels = canvas_w as u64 * canvas_h as u64;
            let dirty_pixels = rw as u64 * rh as u64;

            if rw > 0 && rh > 0 && dirty_pixels < full_pixels / 2 {
                let pixels = Compositor::readback_texture_region(
                    &self.ctx, result_tex, rx, ry, rw, rh,
                    &mut self.cached_staging_buf,
                );
                return Some((pixels, rx, ry, rw, rh, false));
            }
        }

        // Full readback
        let pixels = Compositor::readback_texture(
            &self.ctx, result_tex, canvas_w, canvas_h,
            &mut self.cached_staging_buf,
        );
        Some((pixels, 0, 0, canvas_w, canvas_h, true))
    }

    /// Async variant of `composite_dirty_readback`.  Uses double-buffered
    /// staging so the CPU never blocks on `device.poll(Maintain::Wait)`.
    ///
    /// Returns data from the PREVIOUS frame's readback (1-frame latency,
    /// imperceptible at 60fps).  On the very first call, returns `None`.
    ///
    /// **Call order per frame:**
    ///   1. This method tries to read the previous frame first (non-blocking).
    ///   2. Then it composites and submits a new readback via double-buffered staging.
    ///   3. Returns the previous frame's data (if available).
    pub fn composite_dirty_readback_async(
        &mut self,
        canvas_w: u32,
        canvas_h: u32,
        layer_info: &[(usize, f32, bool, u8)],
        dirty_rect: Option<egui::Rect>,
    ) -> Option<(Vec<u8>, u32, u32, u32, u32, bool)> {
        if !self.available { return None; }

        // --- Step 1: Try to read the previous frame's result (non-blocking) ---
        let previous_data = self.async_readback.try_read(&self.ctx.device);

        // --- Step 2: Do GPU composite (same as sync path) ---
        self.ensure_ping_pong(canvas_w, canvas_h);

        let pp0 = self.ping_pong[0].as_ref().unwrap();
        let pp1 = self.ping_pong[1].as_ref().unwrap();
        let view0 = pp0.create_view(&wgpu::TextureViewDescriptor::default());
        let view1 = pp1.create_view(&wgpu::TextureViewDescriptor::default());

        let mut visible_layers: Vec<(f32, u32, &LayerTexture)> = Vec::new();
        for &(idx, opacity, visible, blend_mode) in layer_info.iter() {
            if !visible { continue; }
            if let Some(state) = self.layer_textures.get(&idx) {
                visible_layers.push((opacity, blend_mode as u32, &state.texture));
            }
        }

        let result_idx = self.compositor.composite_layers_blended(
            &self.ctx,
            [&view0, &view1],
            &visible_layers,
            canvas_w,
            canvas_h,
        );

        let result_tex = if result_idx == 0 {
            self.ping_pong[0].as_ref().unwrap()
        } else {
            self.ping_pong[1].as_ref().unwrap()
        };

        // --- Step 3: Determine readback region and submit async copy ---
        let (rx, ry, rw, rh, is_full) = if let Some(dr) = dirty_rect {
            let drx = (dr.min.x.floor() as u32).max(0).min(canvas_w);
            let dry = (dr.min.y.floor() as u32).max(0).min(canvas_h);
            let drx2 = (dr.max.x.ceil() as u32).min(canvas_w);
            let dry2 = (dr.max.y.ceil() as u32).min(canvas_h);
            let drw = drx2.saturating_sub(drx);
            let drh = dry2.saturating_sub(dry);
            let full_pixels = canvas_w as u64 * canvas_h as u64;
            let dirty_pixels = drw as u64 * drh as u64;

            if drw > 0 && drh > 0 && dirty_pixels < full_pixels / 2 {
                (drx, dry, drw, drh, false)
            } else {
                (0, 0, canvas_w, canvas_h, true)
            }
        } else {
            (0, 0, canvas_w, canvas_h, true)
        };

        if rw == 0 || rh == 0 {
            return previous_data;
        }

        let padded_bytes_per_row = Compositor::aligned_bytes_per_row(rw);
        let buffer_size = (padded_bytes_per_row * rh) as u64;
        self.async_readback.ensure_buffers(&self.ctx.device, buffer_size);

        // Encode the copy-to-staging command
        let mut encoder = self.ctx.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("async_readback_copy") },
        );
        let origin = if is_full {
            wgpu::Origin3d::ZERO
        } else {
            wgpu::Origin3d { x: rx, y: ry, z: 0 }
        };
        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: result_tex,
                mip_level: 0,
                origin,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: self.async_readback.write_buffer(),
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(rh),
                },
            },
            wgpu::Extent3d { width: rw, height: rh, depth_or_array_layers: 1 },
        );
        self.ctx.queue.submit(std::iter::once(encoder.finish()));

        // Start async mapping on the write buffer, then swap
        self.async_readback.submit_and_swap(ReadbackMeta {
            rx, ry, rw, rh, is_full, padded_bytes_per_row,
        });

        // Return previous frame's data (1 frame latent, or None on first call)
        previous_data
    }

    /// Composite all visible layers on GPU and return which ping-pong texture
    /// holds the result.  Does NOT readback — the result stays on GPU.
    /// Returns (result_texture_index, view, sampler) for direct rendering.
    pub fn composite_to_gpu(
        &mut self,
        canvas_w: u32,
        canvas_h: u32,
        layer_info: &[(usize, f32, bool, u8)],
        use_linear_filter: bool,
    ) -> Option<(wgpu::TextureView, &wgpu::Sampler)> {
        if !self.available { return None; }

        self.ensure_ping_pong(canvas_w, canvas_h);

        let pp0 = self.ping_pong[0].as_ref().unwrap();
        let pp1 = self.ping_pong[1].as_ref().unwrap();
        let view0 = pp0.create_view(&wgpu::TextureViewDescriptor::default());
        let view1 = pp1.create_view(&wgpu::TextureViewDescriptor::default());

        // Collect visible layers.
        let mut visible_layers: Vec<(f32, u32, &LayerTexture)> = Vec::new();
        for &(idx, opacity, visible, blend_mode) in layer_info.iter() {
            if !visible { continue; }
            if let Some(state) = self.layer_textures.get(&idx) {
                visible_layers.push((opacity, blend_mode as u32, &state.texture));
            }
        }

        let result_idx = self.compositor.composite_layers_blended(
            &self.ctx,
            [&view0, &view1],
            &visible_layers,
            canvas_w,
            canvas_h,
        );

        // Return the view for the result texture
        let result_tex = if result_idx == 0 {
            self.ping_pong[0].as_ref().unwrap()
        } else {
            self.ping_pong[1].as_ref().unwrap()
        };
        let result_view = result_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = if use_linear_filter {
            &self.compositor.sampler_linear
        } else {
            &self.compositor.sampler_nearest
        };

        Some((result_view, sampler))
    }

    /// Get access to context for external rendering
    pub fn context(&self) -> &GpuContext {
        &self.ctx
    }

    /// Get access to compositor for external rendering
    pub fn compositor(&self) -> &Compositor {
        &self.compositor
    }

    // ========================================================================
    // GPU FILTERS
    // ========================================================================

    /// Gaussian blur (CPU ↔ GPU round-trip).
    pub fn blur_rgba(&self, data: &[u8], width: u32, height: u32, sigma: f32) -> Vec<u8> {
        self.blur_pipeline.blur_image(&self.ctx, data, width, height, sigma)
    }

    /// Brightness/Contrast (CPU ↔ GPU round-trip).
    pub fn brightness_contrast_rgba(&self, data: &[u8], w: u32, h: u32, brightness: f32, contrast: f32) -> Vec<u8> {
        self.bc_pipeline.apply(&self.ctx, data, w, h, brightness, contrast)
    }

    /// Hue/Saturation/Lightness (CPU ↔ GPU round-trip).
    pub fn hsl_rgba(&self, data: &[u8], w: u32, h: u32, hue: f32, sat: f32, light: f32) -> Vec<u8> {
        self.hsl_pipeline.apply(&self.ctx, data, w, h, hue, sat, light)
    }

    /// Invert colors (CPU ↔ GPU round-trip).
    pub fn invert_rgba(&self, data: &[u8], w: u32, h: u32) -> Vec<u8> {
        self.invert_pipeline.apply(&self.ctx, data, w, h)
    }

    /// Median filter (CPU ↔ GPU round-trip).  Returns None if radius > 7.
    pub fn median_rgba(&self, data: &[u8], w: u32, h: u32, radius: u32) -> Option<Vec<u8>> {
        self.median_pipeline.apply(&self.ctx, data, w, h, radius)
    }

    // ========================================================================
    // MEMORY / DEBUG
    // ========================================================================

    pub fn active_texture_count(&self) -> usize {
        self.layer_textures.len()
    }

    pub fn active_texture_memory(&self) -> usize {
        self.layer_textures.values().map(|s| {
            (s.texture.width as usize) * (s.texture.height as usize) * 4
        }).sum()
    }

    pub fn pooled_texture_memory(&self) -> usize {
        self.texture_pool.pooled_memory_bytes()
    }
}

impl Drop for GpuRenderer {
    fn drop(&mut self) {
        self.layer_textures.clear();
        self.texture_pool.clear();
        self.output_texture = None;
        self.ping_pong = [None, None];
    }
}
