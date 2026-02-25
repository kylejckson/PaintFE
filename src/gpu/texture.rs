// ============================================================================
// LAYER TEXTURE — GPU-side texture wrapper with partial upload support
// ============================================================================

/// A GPU-side texture representing a single layer's pixel data.
///
/// ### Key optimisation: `update_rect`
/// When the user draws (brush strokes, eraser, etc.), only the modified region
/// is uploaded via `queue.write_texture` — never the full 5K image.
pub struct LayerTexture {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub bind_group: wgpu::BindGroup,
    pub width: u32,
    pub height: u32,
    /// Number of mip levels.
    pub mip_levels: u32,
}

impl LayerTexture {
    /// Create a new GPU texture from RGBA pixel data.
    /// Generates mipmaps via the compute shader if `mip_pipeline` is provided.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bind_group_layout: &wgpu::BindGroupLayout,
        sampler: &wgpu::Sampler,
        width: u32,
        height: u32,
        data: &[u8],
        mip_pipeline: Option<&MipmapPipeline>,
    ) -> Self {
        let mip_levels = if mip_pipeline.is_some() {
            Self::mip_level_count(width, height)
        } else {
            1
        };

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("LayerTexture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: mip_levels,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        // Upload mip-level 0
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4 * width),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        // Generate mipmaps
        if let Some(mip) = mip_pipeline {
            mip.generate(device, queue, &texture, width, height, mip_levels);
        }

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("LayerTexture bind group"),
            layout: bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        });

        Self {
            texture,
            view,
            bind_group,
            width,
            height,
            mip_levels,
        }
    }

    /// **Crucial optimisation**: upload only the modified rectangle.
    ///
    /// `data` must contain `rect_width * rect_height * 4` bytes of RGBA pixels
    /// for the sub-region starting at `(x, y)`.
    pub fn update_rect(
        &self,
        queue: &wgpu::Queue,
        x: u32,
        y: u32,
        rect_width: u32,
        rect_height: u32,
        data: &[u8],
    ) {
        debug_assert_eq!(data.len(), (rect_width * rect_height * 4) as usize);

        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x, y, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4 * rect_width),
                rows_per_image: Some(rect_height),
            },
            wgpu::Extent3d {
                width: rect_width,
                height: rect_height,
                depth_or_array_layers: 1,
            },
        );
    }

    /// Full re-upload of all pixel data (mip 0 only, call mip gen after).
    pub fn upload_full(&self, queue: &wgpu::Queue, data: &[u8]) {
        self.update_rect(queue, 0, 0, self.width, self.height, data);
    }

    /// Calculate how many mip levels are needed for the given dimensions.
    pub fn mip_level_count(width: u32, height: u32) -> u32 {
        let max_dim = width.max(height) as f32;
        (max_dim.log2().floor() as u32 + 1).min(12) // cap at 12 levels
    }
}

// ============================================================================
// MIPMAP GENERATION PIPELINE
// ============================================================================

/// Compute pipeline that generates successive mip levels from level 0.
pub struct MipmapPipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl MipmapPipeline {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mipmap_shader"),
            source: wgpu::ShaderSource::Wgsl(super::shaders::MIPMAP_SHADER.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mipmap_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mipmap_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("mipmap_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "cs_mipmap",
            compilation_options: Default::default(),
        });

        Self {
            pipeline,
            bind_group_layout,
        }
    }

    /// Generate mip levels 1..mip_levels from mip level 0 of the given texture.
    pub fn generate(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture: &wgpu::Texture,
        mut width: u32,
        mut height: u32,
        mip_levels: u32,
    ) {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("mipmap_encoder"),
        });

        for level in 1..mip_levels {
            let src_view = texture.create_view(&wgpu::TextureViewDescriptor {
                base_mip_level: level - 1,
                mip_level_count: Some(1),
                format: Some(wgpu::TextureFormat::Rgba8Unorm),
                ..Default::default()
            });
            let dst_w = (width / 2).max(1);
            let dst_h = (height / 2).max(1);
            let dst_view = texture.create_view(&wgpu::TextureViewDescriptor {
                base_mip_level: level,
                mip_level_count: Some(1),
                format: Some(wgpu::TextureFormat::Rgba8Unorm),
                ..Default::default()
            });

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("mipmap_bg"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&src_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&dst_view),
                    },
                ],
            });

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("mipmap_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(dst_w.div_ceil(16), dst_h.div_ceil(16), 1);
            }

            width = dst_w;
            height = dst_h;
        }

        queue.submit(std::iter::once(encoder.finish()));
    }
}
