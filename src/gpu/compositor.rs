// ============================================================================
// COMPOSITOR — GPU-accelerated layer composition pipeline
// ============================================================================
//
// Two compositing paths:
//
//   1. **Legacy (fixed-function blend)** — kept for simple normal-only scenes
//      and for the checkerboard. Uses hardware alpha blending.
//
//   2. **Uber-compositor (custom blend shader)** — supports all 15 blend modes.
//      Uses ping-pong rendering: two textures alternate as read (background)
//      and write (destination).  Each layer is drawn as a full-screen quad;
//      the fragment shader samples both the foreground layer AND the background
//      accumulator, applies the blend mode math, and outputs the composited
//      result.  Hardware blending is DISABLED (Replace).
// ============================================================================

use bytemuck::{Pod, Zeroable};

use super::context::GpuContext;
use super::texture::LayerTexture;

// We need the buffer init descriptor helper from wgpu::util.
use wgpu::util::DeviceExt;

// ============================================================================
// UNIFORM TYPES
// ============================================================================

/// View matrix + per-layer opacity, uploaded as a uniform buffer.
/// Used by the legacy pipeline AND the checkerboard pipeline.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ViewUniforms {
    pub view_proj: [[f32; 4]; 4],
    pub opacity: f32,
    pub _pad: [f32; 3],
}

impl ViewUniforms {
    /// Identity projection for offscreen compositing.
    pub fn identity(opacity: f32) -> Self {
        let view_proj: [[f32; 4]; 4] = [
            [ 2.0,  0.0, 0.0, 0.0],
            [ 0.0, -2.0, 0.0, 0.0],
            [ 0.0,  0.0, 1.0, 0.0],
            [-1.0,  1.0, 0.0, 1.0],
        ];
        Self { view_proj, opacity, _pad: [0.0; 3] }
    }
}

/// Uniforms for the uber-compositor: includes blend_mode.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BlendUniforms {
    pub view_proj:  [[f32; 4]; 4],
    pub opacity:    f32,
    pub blend_mode: u32,
    pub _pad: [f32; 2],
}

impl BlendUniforms {
    pub fn identity(opacity: f32, blend_mode: u32) -> Self {
        let view_proj: [[f32; 4]; 4] = [
            [ 2.0,  0.0, 0.0, 0.0],
            [ 0.0, -2.0, 0.0, 0.0],
            [ 0.0,  0.0, 1.0, 0.0],
            [-1.0,  1.0, 0.0, 1.0],
        ];
        Self { view_proj, opacity, blend_mode, _pad: [0.0; 2] }
    }
}

/// Uniforms for the display shader (hardware pan/zoom).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct DisplayUniforms {
    pub offset: [f32; 2],        // Pan offset in screen pixels
    pub scale: f32,              // Zoom factor
    pub _pad0: f32,
    pub viewport_size: [f32; 2], // Viewport dimensions
    pub image_size: [f32; 2],    // Image dimensions
}

impl DisplayUniforms {
    pub fn new(offset_x: f32, offset_y: f32, scale: f32, vp_w: f32, vp_h: f32, img_w: f32, img_h: f32) -> Self {
        Self {
            offset: [offset_x, offset_y],
            scale,
            _pad0: 0.0,
            viewport_size: [vp_w, vp_h],
            image_size: [img_w, img_h],
        }
    }
}

// ============================================================================
// COMPOSITOR (uber-compositor with blend mode support)
// ============================================================================

pub struct Compositor {
    // ---- Uber-compositor (custom blend shader) ----
    pub uber_pipeline: wgpu::RenderPipeline,
    /// Bind group layout for blend uniforms (group 0).
    pub blend_uniform_bgl: wgpu::BindGroupLayout,
    /// Bind group layout for a texture+sampler pair (group 1 = fg, group 2 = bg).
    pub tex_sampler_bgl: wgpu::BindGroupLayout,

    // ---- Display pipeline (hardware pan/zoom) ----
    pub display_pipeline: wgpu::RenderPipeline,
    /// Bind group layout for display uniforms (group 0).
    pub display_uniform_bgl: wgpu::BindGroupLayout,
    /// Bind group layout for display texture + sampler (group 1).
    pub display_tex_bgl: wgpu::BindGroupLayout,

    // ---- Legacy pipeline (kept for API compat + checkerboard) ----
    pub pipeline: wgpu::RenderPipeline,
    pub view_bind_group_layout: wgpu::BindGroupLayout,
    pub texture_bind_group_layout: wgpu::BindGroupLayout,

    // ---- Samplers ----
    pub sampler_linear: wgpu::Sampler,
    pub sampler_nearest: wgpu::Sampler,

    pub output_format: wgpu::TextureFormat,

    /// Cached per-layer uniform buffers and bind groups for `composite_layers_blended`.
    /// Grows to match layer count; reused across frames via `queue.write_buffer()`.
    cached_blend_slots: Vec<(wgpu::Buffer, wgpu::BindGroup)>,
}

impl Compositor {
    pub fn new(device: &wgpu::Device) -> Self {
        let output_format = wgpu::TextureFormat::Rgba8Unorm;

        // ================================================================
        // LEGACY PIPELINE (old normal-only hardware-blended path)
        // ================================================================
        let legacy_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("composite_shader"),
            source: wgpu::ShaderSource::Wgsl(super::shaders::COMPOSITE_SHADER.into()),
        });

        let view_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("view_bgl"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("layer_tex_bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let legacy_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("composite_pipeline_layout"),
            bind_group_layouts: &[&view_bind_group_layout, &texture_bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("composite_pipeline"),
            layout: Some(&legacy_layout),
            vertex: wgpu::VertexState {
                module: &legacy_shader,
                entry_point: "vs_main",
                buffers: &[],
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &legacy_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: output_format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            multiview: None,
        });

        // ================================================================
        // UBER-COMPOSITOR PIPELINE (custom blend modes, no HW blending)
        // ================================================================
        let uber_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("uber_composite_shader"),
            source: wgpu::ShaderSource::Wgsl(super::shaders::UBER_COMPOSITE_SHADER.into()),
        });

        // Group 0: BlendUniforms (view_proj, opacity, blend_mode)
        let blend_uniform_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("blend_uniform_bgl"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        // Group 1 & 2: texture + sampler (reuse same layout for both fg and bg)
        let tex_sampler_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("tex_sampler_bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let uber_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("uber_composite_pipeline_layout"),
            bind_group_layouts: &[&blend_uniform_bgl, &tex_sampler_bgl, &tex_sampler_bgl],
            push_constant_ranges: &[],
        });

        // NO hardware blending — the fragment shader does all blend math.
        let uber_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("uber_composite_pipeline"),
            layout: Some(&uber_layout),
            vertex: wgpu::VertexState {
                module: &uber_shader,
                entry_point: "vs_blend",
                buffers: &[],
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &uber_shader,
                entry_point: "fs_blend",
                targets: &[Some(wgpu::ColorTargetState {
                    format: output_format,
                    blend: None, // DISABLED — shader handles blending
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            multiview: None,
        });

        // ---- Samplers ----
        let sampler_linear = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("sampler_linear"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let sampler_nearest = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("sampler_nearest"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // ================================================================
        // DISPLAY PIPELINE (hardware pan/zoom)
        // ================================================================
        let display_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("display_shader"),
            source: wgpu::ShaderSource::Wgsl(super::shaders::DISPLAY_SHADER.into()),
        });

        let display_uniform_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("display_uniform_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let display_tex_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("display_tex_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let display_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("display_pipeline_layout"),
            bind_group_layouts: &[&display_uniform_bgl, &display_tex_bgl],
            push_constant_ranges: &[],
        });

        let display_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("display_pipeline"),
            layout: Some(&display_layout),
            vertex: wgpu::VertexState {
                module: &display_shader,
                entry_point: "vs_display",
                buffers: &[],
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &display_shader,
                entry_point: "fs_display",
                targets: &[Some(wgpu::ColorTargetState {
                    format: output_format,
                    // Use premultiplied alpha blending to match the compositor shader
                    // output format. This prevents color desaturation/lightening when
                    // rendering semi-transparent pixels over light backgrounds.
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            multiview: None,
        });

        Self {
            uber_pipeline,
            blend_uniform_bgl,
            tex_sampler_bgl,
            display_pipeline,
            display_uniform_bgl,
            display_tex_bgl,
            pipeline,
            view_bind_group_layout,
            texture_bind_group_layout,
            sampler_linear,
            sampler_nearest,
            output_format,
            cached_blend_slots: Vec::new(),
        }
    }

    pub fn sampler_for_zoom(&self, zoom: f32) -> &wgpu::Sampler {
        // Use nearest-neighbor for zoomed-in (>= 1.5x) for pixel-perfect rendering.
        // For zoomed-out, the user's preference determines the filter mode.
        if zoom >= 1.5 {
            &self.sampler_nearest
        } else {
            // User will control which to use when zoomed out via settings
            &self.sampler_linear
        }
    }

    // ========================================================================
    // UBER-COMPOSITION: ping-pong blend-mode-aware compositing
    // ========================================================================

    /// Composite visible layers with full blend-mode support.
    ///
    /// Uses ping-pong rendering between two textures:
    ///   - `ping`: background accumulator (read)
    ///   - `pong`: destination (write)
    ///   - After each layer, swap ping ↔ pong.
    ///
    /// `layers`: `(opacity, blend_mode_u8, &LayerTexture)` in back-to-front order.
    ///
    /// Returns which of the two ping-pong textures holds the final result
    /// (0 or 1) so the caller knows which to read back.
    pub fn composite_layers_blended(
        &mut self,
        ctx: &GpuContext,
        ping_pong: [&wgpu::TextureView; 2],
        layers: &[(f32, u32, &LayerTexture)],
        _width: u32,
        _height: u32,
    ) -> usize {
        let device = &ctx.device;
        let queue = &ctx.queue;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("uber_composite_encoder"),
        });

        let sampler = &self.sampler_linear;

        // Clear ping (texture 0) to transparent black.
        {
            let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("clear_ping"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: ping_pong[0],
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
        }

        let mut read_idx: usize = 0;  // ping = background (read)
        let mut write_idx: usize = 1; // pong = destination (write)

        for (layer_i, (opacity, blend_mode, layer_tex)) in layers.iter().enumerate() {
            // ---- Uniforms: reuse cached buffer + bind group ----
            let uniforms = BlendUniforms::identity(*opacity, *blend_mode);
            if layer_i >= self.cached_blend_slots.len() {
                // First time seeing this many layers — allocate new slot
                let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("blend_uniform_buf"),
                    contents: bytemuck::bytes_of(&uniforms),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });
                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("blend_uniform_bg"),
                    layout: &self.blend_uniform_bgl,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buf.as_entire_binding(),
                    }],
                });
                self.cached_blend_slots.push((buf, bg));
            } else {
                // Reuse existing buffer — just update contents
                queue.write_buffer(&self.cached_blend_slots[layer_i].0, 0, bytemuck::bytes_of(&uniforms));
            }
            let uniform_bg = &self.cached_blend_slots[layer_i].1;

            // ---- Foreground bind group (group 1) ----
            let fg_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("fg_bg"),
                layout: &self.tex_sampler_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&layer_tex.view) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(sampler) },
                ],
            });

            // ---- Background bind group (group 2) — read from ping ----
            let bg_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bg_bg"),
                layout: &self.tex_sampler_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(ping_pong[read_idx]) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(sampler) },
                ],
            });

            // ---- Render pass: draw to pong ----
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("uber_layer_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: ping_pong[write_idx],
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });

                pass.set_pipeline(&self.uber_pipeline);
                pass.set_bind_group(0, &uniform_bg, &[]);
                pass.set_bind_group(1, &fg_bg, &[]);
                pass.set_bind_group(2, &bg_bg, &[]);
                pass.draw(0..6, 0..1);
            }

            // Swap ping/pong
            std::mem::swap(&mut read_idx, &mut write_idx);
        }

        queue.submit(std::iter::once(encoder.finish()));

        // `read_idx` now points to the texture with the final composited result
        // (because it was the last write_idx before the swap).
        read_idx
    }

    // ========================================================================
    // LEGACY COMPOSITION (kept for backward compat / simple normal-only)
    // ========================================================================

    pub fn create_view_bind_group(
        &self,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        uniforms: &ViewUniforms,
    ) -> (wgpu::BindGroup, wgpu::Buffer) {
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("view_uniform_buf"),
            contents: bytemuck::bytes_of(uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("view_bg"),
            layout: &self.view_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });

        (bind_group, buffer)
    }

    /// Legacy composite (normal blend only, hardware alpha blending).
    pub fn composite_layers(
        &self,
        ctx: &GpuContext,
        output_view: &wgpu::TextureView,
        layers: &[(f32, &LayerTexture)],
    ) {
        let device = &ctx.device;
        let queue = &ctx.queue;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("composite_encoder"),
        });

        {
            let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("clear_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: output_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
        }

        for (opacity, layer_tex) in layers.iter() {
            let uniforms = ViewUniforms::identity(*opacity);
            let (view_bg, _buf) = self.create_view_bind_group(device, queue, &uniforms);

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("layer_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: output_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &view_bg, &[]);
            pass.set_bind_group(1, &layer_tex.bind_group, &[]);
            pass.draw(0..6, 0..1);
        }

        queue.submit(std::iter::once(encoder.finish()));
    }

    /// Read back the composited result from a GPU texture to a CPU buffer.
    pub fn readback_texture(
        ctx: &GpuContext,
        texture: &wgpu::Texture,
        width: u32,
        height: u32,
        cached_staging: &mut Option<(wgpu::Buffer, u64)>,
    ) -> Vec<u8> {
        let device = &ctx.device;
        let queue = &ctx.queue;

        let bytes_per_row = Self::aligned_bytes_per_row(width);
        let buffer_size = (bytes_per_row * height) as u64;

        // A2: Reuse cached staging buffer if large enough
        let need_new = match cached_staging {
            Some((_, sz)) if *sz >= buffer_size => false,
            _ => true,
        };
        if need_new {
            let new_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("readback_staging"),
                size: buffer_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            *cached_staging = Some((new_buf, buffer_size));
        }
        let staging = &cached_staging.as_ref().unwrap().0;

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("readback_encoder"),
        });

        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &staging,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        );

        queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        device.poll(wgpu::Maintain::Wait);
        match rx.recv() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => { eprintln!("[GPU] readback_texture map error: {:?}", e); return vec![]; }
            Err(e) => { eprintln!("[GPU] readback_texture channel error: {:?}", e); return vec![]; }
        }

        let mapped = slice.get_mapped_range();
        let actual_row = width * 4;

        let mut result = Vec::with_capacity((actual_row * height) as usize);
        for y in 0..height {
            let start = (y * bytes_per_row) as usize;
            let end = start + actual_row as usize;
            result.extend_from_slice(&mapped[start..end]);
        }

        drop(mapped);
        staging.unmap();

        result
    }

    pub(crate) fn aligned_bytes_per_row(width: u32) -> u32 {
        let unaligned = width * 4;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        (unaligned + align - 1) / align * align
    }

    /// Read back a sub-region of a texture as packed RGBA bytes.
    /// `src_x`, `src_y` are the origin in the texture; `region_w`, `region_h`
    /// are the dimensions of the region to read.
    pub fn readback_texture_region(
        ctx: &GpuContext,
        texture: &wgpu::Texture,
        src_x: u32,
        src_y: u32,
        region_w: u32,
        region_h: u32,
        cached_staging: &mut Option<(wgpu::Buffer, u64)>,
    ) -> Vec<u8> {
        let device = &ctx.device;
        let queue = &ctx.queue;

        let bytes_per_row = Self::aligned_bytes_per_row(region_w);
        let buffer_size = (bytes_per_row * region_h) as u64;

        // A2: Reuse cached staging buffer if large enough
        let need_new = match cached_staging {
            Some((_, sz)) if *sz >= buffer_size => false,
            _ => true,
        };
        if need_new {
            let new_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("readback_region_staging"),
                size: buffer_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            *cached_staging = Some((new_buf, buffer_size));
        }
        let staging = &cached_staging.as_ref().unwrap().0;

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("readback_region_encoder"),
        });

        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x: src_x, y: src_y, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &staging,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(region_h),
                },
            },
            wgpu::Extent3d { width: region_w, height: region_h, depth_or_array_layers: 1 },
        );

        queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        device.poll(wgpu::Maintain::Wait);
        match rx.recv() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => { eprintln!("[GPU] readback_texture_region map error: {:?}", e); return vec![]; }
            Err(e) => { eprintln!("[GPU] readback_texture_region channel error: {:?}", e); return vec![]; }
        }

        let mapped = slice.get_mapped_range();
        let actual_row = region_w as usize * 4;

        let mut result = Vec::with_capacity(actual_row * region_h as usize);
        for y in 0..region_h {
            let start = (y * bytes_per_row) as usize;
            let end = start + actual_row;
            result.extend_from_slice(&mapped[start..end]);
        }

        drop(mapped);
        staging.unmap();

        result
    }
}
