// ============================================================================
// GPU COMPUTE FILTERS — Gaussian blur, brightness/contrast, HSL, invert, median
// ============================================================================

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use super::context::GpuContext;

// ============================================================================
// SHARED HELPERS
// ============================================================================

fn create_rw_texture(device: &wgpu::Device, w: u32, h: u32, label: &str) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

fn upload_rgba(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    data: &[u8],
    w: u32,
    h: u32,
    label: &str,
) -> wgpu::Texture {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: &tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        data,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(4 * w),
            rows_per_image: Some(h),
        },
        wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
    );
    tex
}

fn upload_r8(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    data: &[u8],
    w: u32,
    h: u32,
    label: &str,
) -> wgpu::Texture {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: &tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        data,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(w),
            rows_per_image: Some(h),
        },
        wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
    );
    tex
}

/// Standard bind group layout used by most filters: input tex, output storage tex, uniform buf.
fn filter_bgl(device: &wgpu::Device, label: &str) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
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
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

fn dispatch_simple_filter(
    ctx: &GpuContext,
    pipeline: &wgpu::ComputePipeline,
    bgl: &wgpu::BindGroupLayout,
    input_data: &[u8],
    w: u32,
    h: u32,
    params_bytes: &[u8],
    _entry: &str,
) -> Vec<u8> {
    let device = &ctx.device;
    let queue = &ctx.queue;

    let src_tex = upload_rgba(device, queue, input_data, w, h, "filter_src");
    let dst_tex = create_rw_texture(device, w, h, "filter_dst");

    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("filter_params"),
        contents: params_bytes,
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let src_view = src_tex.create_view(&wgpu::TextureViewDescriptor::default());
    let dst_view = dst_tex.create_view(&wgpu::TextureViewDescriptor::default());

    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("filter_bg"),
        layout: bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&src_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&dst_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("filter_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("filter_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(w.div_ceil(16), h.div_ceil(16), 1);
    }
    queue.submit(std::iter::once(encoder.finish()));

    super::compositor::Compositor::readback_texture(ctx, &dst_tex, w, h, &mut None)
}

// ============================================================================
// GAUSSIAN BLUR (shared-memory optimised)
// ============================================================================

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BlurParams {
    radius: u32,
    direction: u32,
    width: u32,
    height: u32,
}

pub struct GpuBlurPipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl GpuBlurPipeline {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("blur_compute_shader"),
            source: wgpu::ShaderSource::Wgsl(super::shaders::GAUSSIAN_BLUR_SHADER.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("blur_bgl"),
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
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("blur_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("blur_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "cs_blur",
            compilation_options: Default::default(),
        });

        Self {
            pipeline,
            bind_group_layout,
        }
    }

    /// Two-pass separable Gaussian blur on a GPU texture.
    /// Updated dispatch for the shared-memory workgroup_size(256,1,1) shader.
    pub fn blur(
        &self,
        ctx: &GpuContext,
        input: &wgpu::Texture,
        width: u32,
        height: u32,
        sigma: f32,
    ) -> wgpu::Texture {
        let device = &ctx.device;
        let queue = &ctx.queue;

        let kernel = Self::build_kernel(sigma);
        // Cap radius to 127 to stay within shared memory limits (MAX_SHARED=512, TILE_W=256, max apron=256+2*127=510)
        let radius = (kernel.len() / 2).min(127) as u32;
        let kernel = if kernel.len() > (radius as usize * 2 + 1) {
            // Truncate kernel to capped radius and renormalize
            let center = kernel.len() / 2;
            let start = center - radius as usize;
            let end = center + radius as usize + 1;
            let mut truncated: Vec<f32> = kernel[start..end].to_vec();
            let sum: f32 = truncated.iter().sum();
            if sum > 0.0 {
                for v in &mut truncated {
                    *v /= sum;
                }
            }
            truncated
        } else {
            kernel
        };

        let kernel_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("blur_kernel_buf"),
            contents: bytemuck::cast_slice(&kernel),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let temp_tex = create_rw_texture(device, width, height, "blur_temp");
        let output_tex = create_rw_texture(device, width, height, "blur_output");

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("blur_encoder"),
        });

        // ---- Pass 1: Horizontal blur (input → temp) ----
        // Dispatch: one workgroup of 256 threads per tile-row, one row per Y.
        {
            let params = BlurParams {
                radius,
                direction: 0,
                width,
                height,
            };
            let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("blur_params_h"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

            let input_view = input.create_view(&wgpu::TextureViewDescriptor {
                format: Some(wgpu::TextureFormat::Rgba8Unorm),
                ..Default::default()
            });
            let temp_view = temp_tex.create_view(&wgpu::TextureViewDescriptor {
                format: Some(wgpu::TextureFormat::Rgba8Unorm),
                ..Default::default()
            });

            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("blur_bg_h"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&input_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&temp_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: kernel_buffer.as_entire_binding(),
                    },
                ],
            });

            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("blur_h_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            // X workgroups: ceil(width / 256), Y: one per row
            pass.dispatch_workgroups(width.div_ceil(256), height, 1);
        }

        // ---- Pass 2: Vertical blur (temp → output) ----
        {
            let params = BlurParams {
                radius,
                direction: 1,
                width,
                height,
            };
            let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("blur_params_v"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

            let temp_view = temp_tex.create_view(&wgpu::TextureViewDescriptor {
                format: Some(wgpu::TextureFormat::Rgba8Unorm),
                ..Default::default()
            });
            let output_view = output_tex.create_view(&wgpu::TextureViewDescriptor {
                format: Some(wgpu::TextureFormat::Rgba8Unorm),
                ..Default::default()
            });

            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("blur_bg_v"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&temp_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&output_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: kernel_buffer.as_entire_binding(),
                    },
                ],
            });

            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("blur_v_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            // Vertical: X workgroups = ceil(height/256), Y = width (one column per Y)
            pass.dispatch_workgroups(height.div_ceil(256), width, 1);
        }

        queue.submit(std::iter::once(encoder.finish()));
        output_tex
    }

    /// CPU ↔ GPU convenience: upload, blur, read back.
    pub fn blur_image(
        &self,
        ctx: &GpuContext,
        input_data: &[u8],
        width: u32,
        height: u32,
        sigma: f32,
    ) -> Vec<u8> {
        let src_tex = upload_rgba(
            &ctx.device,
            &ctx.queue,
            input_data,
            width,
            height,
            "blur_src",
        );
        let output_tex = self.blur(ctx, &src_tex, width, height, sigma);
        super::compositor::Compositor::readback_texture(ctx, &output_tex, width, height, &mut None)
    }

    fn build_kernel(sigma: f32) -> Vec<f32> {
        let radius = (sigma * 3.0).ceil() as usize;
        if radius == 0 {
            return vec![1.0];
        }
        let len = radius * 2 + 1;
        let mut kernel = vec![0.0f32; len];
        let s2 = 2.0 * sigma * sigma;
        let mut sum = 0.0f32;
        for (i, item) in kernel.iter_mut().enumerate() {
            let x = i as f32 - radius as f32;
            let v = (-x * x / s2).exp();
            *item = v;
            sum += v;
        }
        let inv = 1.0 / sum;
        for v in &mut kernel {
            *v *= inv;
        }
        kernel
    }
}

// ============================================================================
// BRIGHTNESS / CONTRAST
// ============================================================================

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BcParams {
    width: u32,
    height: u32,
    brightness: f32,
    contrast: f32,
}

pub struct GpuBrightnessContrastPipeline {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
}

impl GpuBrightnessContrastPipeline {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bc_shader"),
            source: wgpu::ShaderSource::Wgsl(super::shaders::BRIGHTNESS_CONTRAST_SHADER.into()),
        });
        let bgl = filter_bgl(device, "bc_bgl");
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("bc_pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("bc_pipeline"),
            layout: Some(&layout),
            module: &shader,
            entry_point: "cs_brightness_contrast",
            compilation_options: Default::default(),
        });
        Self { pipeline, bgl }
    }

    pub fn apply(
        &self,
        ctx: &GpuContext,
        data: &[u8],
        w: u32,
        h: u32,
        brightness: f32,
        contrast: f32,
    ) -> Vec<u8> {
        let params = BcParams {
            width: w,
            height: h,
            brightness,
            contrast,
        };
        dispatch_simple_filter(
            ctx,
            &self.pipeline,
            &self.bgl,
            data,
            w,
            h,
            bytemuck::bytes_of(&params),
            "cs_brightness_contrast",
        )
    }
}

// ============================================================================
// HUE / SATURATION / LIGHTNESS
// ============================================================================

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct HslParams {
    width: u32,
    height: u32,
    hue_shift: f32,    // normalised: hue_shift_degrees / 360.0
    sat_factor: f32,   // 1.0 + saturation / 100.0
    light_offset: f32, // lightness * 255.0 / 100.0 / 255.0 = lightness / 100.0
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

pub struct GpuHslPipeline {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
}

impl GpuHslPipeline {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("hsl_shader"),
            source: wgpu::ShaderSource::Wgsl(super::shaders::HSL_ADJUST_SHADER.into()),
        });
        let bgl = filter_bgl(device, "hsl_bgl");
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("hsl_pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("hsl_pipeline"),
            layout: Some(&layout),
            module: &shader,
            entry_point: "cs_hsl_adjust",
            compilation_options: Default::default(),
        });
        Self { pipeline, bgl }
    }

    /// `hue_shift`: degrees (-180..180), `saturation`: -100..100, `lightness`: -100..100
    pub fn apply(
        &self,
        ctx: &GpuContext,
        data: &[u8],
        w: u32,
        h: u32,
        hue_shift: f32,
        saturation: f32,
        lightness: f32,
    ) -> Vec<u8> {
        let params = HslParams {
            width: w,
            height: h,
            hue_shift: hue_shift / 360.0,
            sat_factor: 1.0 + saturation / 100.0,
            light_offset: lightness / 100.0,
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
        };
        dispatch_simple_filter(
            ctx,
            &self.pipeline,
            &self.bgl,
            data,
            w,
            h,
            bytemuck::bytes_of(&params),
            "cs_hsl_adjust",
        )
    }
}

// ============================================================================
// INVERT
// ============================================================================

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct InvParams {
    width: u32,
    height: u32,
    _pad0: u32,
    _pad1: u32,
}

pub struct GpuInvertPipeline {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
}

impl GpuInvertPipeline {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("invert_shader"),
            source: wgpu::ShaderSource::Wgsl(super::shaders::INVERT_SHADER.into()),
        });
        let bgl = filter_bgl(device, "invert_bgl");
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("invert_pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("invert_pipeline"),
            layout: Some(&layout),
            module: &shader,
            entry_point: "cs_invert",
            compilation_options: Default::default(),
        });
        Self { pipeline, bgl }
    }

    pub fn apply(&self, ctx: &GpuContext, data: &[u8], w: u32, h: u32) -> Vec<u8> {
        let params = InvParams {
            width: w,
            height: h,
            _pad0: 0,
            _pad1: 0,
        };
        dispatch_simple_filter(
            ctx,
            &self.pipeline,
            &self.bgl,
            data,
            w,
            h,
            bytemuck::bytes_of(&params),
            "cs_invert",
        )
    }
}

// ============================================================================
// MEDIAN FILTER
// ============================================================================

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MedianParams {
    width: u32,
    height: u32,
    radius: u32,
    _pad0: u32,
}

pub struct GpuMedianPipeline {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
}

impl GpuMedianPipeline {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("median_shader"),
            source: wgpu::ShaderSource::Wgsl(super::shaders::MEDIAN_SHADER.into()),
        });
        let bgl = filter_bgl(device, "median_bgl");
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("median_pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("median_pipeline"),
            layout: Some(&layout),
            module: &shader,
            entry_point: "cs_median",
            compilation_options: Default::default(),
        });
        Self { pipeline, bgl }
    }

    /// GPU median filter.  `radius` is clamped to 7 max (window 15×15).
    /// Returns None if radius > 7 (caller should fall back to CPU).
    pub fn apply(
        &self,
        ctx: &GpuContext,
        data: &[u8],
        w: u32,
        h: u32,
        radius: u32,
    ) -> Option<Vec<u8>> {
        if radius > 7 {
            return None;
        }
        let params = MedianParams {
            width: w,
            height: h,
            radius,
            _pad0: 0,
        };
        Some(dispatch_simple_filter(
            ctx,
            &self.pipeline,
            &self.bgl,
            data,
            w,
            h,
            bytemuck::bytes_of(&params),
            "cs_median",
        ))
    }
}

// ============================================================================
// GRADIENT GENERATOR — GPU-accelerated gradient rasterizer
// ============================================================================

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GradientGpuParams {
    pub start_x: f32,
    pub start_y: f32,
    pub end_x: f32,
    pub end_y: f32,
    pub width: u32,
    pub height: u32,
    pub shape: u32,     // 0=Linear, 1=LinearReflected, 2=Radial, 3=Diamond
    pub repeat: u32,    // 0=clamp, 1=repeat
    pub is_eraser: u32, // 0=color, 1=transparency/eraser
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

pub struct GpuGradientPipeline {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    // Cached GPU resources — reused when dimensions match
    cached_output_tex: Option<wgpu::Texture>,
    cached_staging_buf: Option<wgpu::Buffer>,
    cached_params_buf: Option<wgpu::Buffer>,
    cached_lut_buf: Option<wgpu::Buffer>,
    cached_w: u32,
    cached_h: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct MagicWandGpuParams {
    pub width: u32,
    pub height: u32,
    pub threshold: u32,
    pub anti_aliased: u32,
    pub mode: u32,
    pub use_base: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

pub struct GpuMagicWandPipeline {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    cached_distance_tex: Option<wgpu::Texture>,
    cached_base_tex: Option<wgpu::Texture>,
    dummy_base_tex: Option<wgpu::Texture>,
    cached_output_tex: Option<wgpu::Texture>,
    cached_staging_buf: Option<wgpu::Buffer>,
    cached_params_buf: Option<wgpu::Buffer>,
    cached_w: u32,
    cached_h: u32,
    cached_distance_key: usize,
    cached_base_key: Option<usize>,
}

impl GpuMagicWandPipeline {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("magic_wand_mask_shader"),
            source: wgpu::ShaderSource::Wgsl(super::shaders::MAGIC_WAND_MASK_SHADER.into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("magic_wand_mask_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("magic_wand_mask_pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("magic_wand_mask_pipeline"),
            layout: Some(&layout),
            module: &shader,
            entry_point: "cs_magic_wand_mask",
            compilation_options: Default::default(),
        });

        Self {
            pipeline,
            bgl,
            cached_distance_tex: None,
            cached_base_tex: None,
            dummy_base_tex: None,
            cached_output_tex: None,
            cached_staging_buf: None,
            cached_params_buf: None,
            cached_w: 0,
            cached_h: 0,
            cached_distance_key: 0,
            cached_base_key: None,
        }
    }

    fn ensure_cache(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, w: u32, h: u32) {
        if self.cached_w != w || self.cached_h != h {
            self.cached_output_tex = Some(create_rw_texture(device, w, h, "magic_wand_mask_output"));

            let bytes_per_row = super::compositor::Compositor::aligned_bytes_per_row(w);
            let buffer_size = (bytes_per_row * h) as u64;
            self.cached_staging_buf = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("magic_wand_mask_staging"),
                size: buffer_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));

            self.cached_distance_tex = None;
            self.cached_base_tex = None;
            self.cached_distance_key = 0;
            self.cached_base_key = None;
            self.cached_w = w;
            self.cached_h = h;
        }

        if self.dummy_base_tex.is_none() {
            self.dummy_base_tex = Some(upload_r8(device, queue, &[0], 1, 1, "magic_wand_base_dummy"));
        }
    }

    pub fn generate_into(
        &mut self,
        ctx: &GpuContext,
        distances: &[u8],
        distance_key: usize,
        base_mask: Option<&[u8]>,
        base_key: Option<usize>,
        w: u32,
        h: u32,
        threshold: u8,
        anti_aliased: bool,
        mode: u32,
        out: &mut Vec<u8>,
    ) {
        let device = &ctx.device;
        let queue = &ctx.queue;

        self.ensure_cache(device, queue, w, h);

        if self.cached_distance_tex.is_none() || self.cached_distance_key != distance_key {
            self.cached_distance_tex = Some(upload_r8(
                device,
                queue,
                distances,
                w,
                h,
                "magic_wand_distances",
            ));
            self.cached_distance_key = distance_key;
        }

        if let Some(base_data) = base_mask {
            if self.cached_base_tex.is_none() || self.cached_base_key != base_key {
                self.cached_base_tex = Some(upload_r8(device, queue, base_data, w, h, "magic_wand_base_mask"));
                self.cached_base_key = base_key;
            }
        } else {
            self.cached_base_key = None;
        }

        let params = MagicWandGpuParams {
            width: w,
            height: h,
            threshold: threshold as u32,
            anti_aliased: if anti_aliased { 1 } else { 0 },
            mode,
            use_base: if base_mask.is_some() { 1 } else { 0 },
            _pad0: 0,
            _pad1: 0,
        };
        let params_bytes = bytemuck::bytes_of(&params);
        let params_buf = self.cached_params_buf.get_or_insert_with(|| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("magic_wand_mask_params"),
                contents: params_bytes,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
        });
        queue.write_buffer(params_buf, 0, params_bytes);

        let dist_view = self
            .cached_distance_tex
            .as_ref()
            .unwrap()
            .create_view(&wgpu::TextureViewDescriptor::default());
        let base_view = if let Some(base_tex) = self.cached_base_tex.as_ref() {
            base_tex.create_view(&wgpu::TextureViewDescriptor::default())
        } else {
            self.dummy_base_tex
                .as_ref()
                .unwrap()
                .create_view(&wgpu::TextureViewDescriptor::default())
        };
        let out_view = self
            .cached_output_tex
            .as_ref()
            .unwrap()
            .create_view(&wgpu::TextureViewDescriptor::default());

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("magic_wand_mask_bg"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&dist_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&base_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&out_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        let bytes_per_row = super::compositor::Compositor::aligned_bytes_per_row(w);
        let staging = self.cached_staging_buf.as_ref().unwrap();
        let out_tex = self.cached_output_tex.as_ref().unwrap();

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("magic_wand_mask_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("magic_wand_mask_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(w.div_ceil(16), h.div_ceil(16), 1);
        }

        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: out_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: staging,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
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
            Ok(Err(e)) => {
                eprintln!("[GPU] GpuMagicWandPipeline readback map error: {:?}", e);
                return;
            }
            Err(e) => {
                eprintln!("[GPU] GpuMagicWandPipeline readback channel error: {:?}", e);
                return;
            }
        }

        let mapped = slice.get_mapped_range();
        out.clear();
        out.resize((w * h) as usize, 0);
        for y in 0..h as usize {
            let src_row = y * bytes_per_row as usize;
            let dst_row = y * w as usize;
            for x in 0..w as usize {
                out[dst_row + x] = mapped[src_row + x * 4];
            }
        }

        drop(mapped);
        staging.unmap();
    }
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct FillPreviewGpuParams {
    pub fill_color: [f32; 4],
    pub canvas_width: u32,
    pub canvas_height: u32,
    pub region_x: u32,
    pub region_y: u32,
    pub region_width: u32,
    pub region_height: u32,
    pub threshold: u32,
    pub anti_aliased: u32,
    pub use_selection: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

pub struct GpuFillPreviewPipeline {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    cached_distance_tex: Option<wgpu::Texture>,
    cached_background_tex: Option<wgpu::Texture>,
    cached_selection_tex: Option<wgpu::Texture>,
    dummy_selection_tex: Option<wgpu::Texture>,
    cached_output_tex: Option<wgpu::Texture>,
    cached_staging_buf: Option<wgpu::Buffer>,
    cached_params_buf: Option<wgpu::Buffer>,
    cached_canvas_w: u32,
    cached_canvas_h: u32,
    cached_region_w: u32,
    cached_region_h: u32,
    cached_distance_key: usize,
    cached_background_key: usize,
    cached_selection_key: Option<usize>,
}

impl GpuFillPreviewPipeline {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fill_preview_shader"),
            source: wgpu::ShaderSource::Wgsl(super::shaders::FILL_PREVIEW_SHADER.into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fill_preview_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fill_preview_pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("fill_preview_pipeline"),
            layout: Some(&layout),
            module: &shader,
            entry_point: "cs_fill_preview",
            compilation_options: Default::default(),
        });

        Self {
            pipeline,
            bgl,
            cached_distance_tex: None,
            cached_background_tex: None,
            cached_selection_tex: None,
            dummy_selection_tex: None,
            cached_output_tex: None,
            cached_staging_buf: None,
            cached_params_buf: None,
            cached_canvas_w: 0,
            cached_canvas_h: 0,
            cached_region_w: 0,
            cached_region_h: 0,
            cached_distance_key: 0,
            cached_background_key: 0,
            cached_selection_key: None,
        }
    }

    fn ensure_cache(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        canvas_w: u32,
        canvas_h: u32,
        region_w: u32,
        region_h: u32,
    ) {
        if self.cached_canvas_w != canvas_w || self.cached_canvas_h != canvas_h {
            self.cached_distance_tex = None;
            self.cached_background_tex = None;
            self.cached_selection_tex = None;
            self.cached_distance_key = 0;
            self.cached_background_key = 0;
            self.cached_selection_key = None;
            self.cached_canvas_w = canvas_w;
            self.cached_canvas_h = canvas_h;
        }

        if self.cached_region_w != region_w || self.cached_region_h != region_h {
            self.cached_output_tex = Some(create_rw_texture(device, region_w, region_h, "fill_preview_output"));

            let bytes_per_row = super::compositor::Compositor::aligned_bytes_per_row(region_w);
            let buffer_size = (bytes_per_row * region_h) as u64;
            self.cached_staging_buf = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("fill_preview_staging"),
                size: buffer_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));

            self.cached_region_w = region_w;
            self.cached_region_h = region_h;
        }

        if self.dummy_selection_tex.is_none() {
            self.dummy_selection_tex = Some(upload_r8(device, queue, &[255], 1, 1, "fill_preview_selection_dummy"));
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn generate_into(
        &mut self,
        ctx: &GpuContext,
        distances: &[u8],
        distance_key: usize,
        background_rgba: &[u8],
        background_key: usize,
        selection_mask: Option<(&[u8], usize)>,
        canvas_w: u32,
        canvas_h: u32,
        region_x: u32,
        region_y: u32,
        region_w: u32,
        region_h: u32,
        threshold: u8,
        anti_aliased: bool,
        fill_color: [u8; 4],
        out: &mut Vec<u8>,
    ) {
        let device = &ctx.device;
        let queue = &ctx.queue;

        self.ensure_cache(device, queue, canvas_w, canvas_h, region_w, region_h);

        if self.cached_distance_tex.is_none() || self.cached_distance_key != distance_key {
            self.cached_distance_tex = Some(upload_r8(device, queue, distances, canvas_w, canvas_h, "fill_preview_distances"));
            self.cached_distance_key = distance_key;
        }

        if self.cached_background_tex.is_none() || self.cached_background_key != background_key {
            self.cached_background_tex = Some(upload_rgba(
                device,
                queue,
                background_rgba,
                canvas_w,
                canvas_h,
                "fill_preview_background",
            ));
            self.cached_background_key = background_key;
        }

        if let Some((selection_data, selection_key)) = selection_mask {
            if self.cached_selection_tex.is_none() || self.cached_selection_key != Some(selection_key) {
                self.cached_selection_tex = Some(upload_r8(
                    device,
                    queue,
                    selection_data,
                    canvas_w,
                    canvas_h,
                    "fill_preview_selection",
                ));
                self.cached_selection_key = Some(selection_key);
            }
        } else {
            self.cached_selection_key = None;
        }

        let params = FillPreviewGpuParams {
            fill_color: [
                fill_color[0] as f32 / 255.0,
                fill_color[1] as f32 / 255.0,
                fill_color[2] as f32 / 255.0,
                fill_color[3] as f32 / 255.0,
            ],
            canvas_width: canvas_w,
            canvas_height: canvas_h,
            region_x,
            region_y,
            region_width: region_w,
            region_height: region_h,
            threshold: threshold as u32,
            anti_aliased: if anti_aliased { 1 } else { 0 },
            use_selection: if selection_mask.is_some() { 1 } else { 0 },
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };

        let params_bytes = bytemuck::bytes_of(&params);
        let params_buf = self.cached_params_buf.get_or_insert_with(|| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("fill_preview_params"),
                contents: params_bytes,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
        });
        queue.write_buffer(params_buf, 0, params_bytes);

        let dist_view = self
            .cached_distance_tex
            .as_ref()
            .unwrap()
            .create_view(&wgpu::TextureViewDescriptor::default());
        let bg_view = self
            .cached_background_tex
            .as_ref()
            .unwrap()
            .create_view(&wgpu::TextureViewDescriptor::default());
        let sel_view = if let Some(selection_tex) = self.cached_selection_tex.as_ref() {
            selection_tex.create_view(&wgpu::TextureViewDescriptor::default())
        } else {
            self.dummy_selection_tex
                .as_ref()
                .unwrap()
                .create_view(&wgpu::TextureViewDescriptor::default())
        };
        let out_view = self
            .cached_output_tex
            .as_ref()
            .unwrap()
            .create_view(&wgpu::TextureViewDescriptor::default());

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fill_preview_bg"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&dist_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&bg_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&sel_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&out_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        let bytes_per_row = super::compositor::Compositor::aligned_bytes_per_row(region_w);
        let staging = self.cached_staging_buf.as_ref().unwrap();
        let out_tex = self.cached_output_tex.as_ref().unwrap();

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("fill_preview_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("fill_preview_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(region_w.div_ceil(16), region_h.div_ceil(16), 1);
        }

        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: out_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: staging,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(region_h),
                },
            },
            wgpu::Extent3d {
                width: region_w,
                height: region_h,
                depth_or_array_layers: 1,
            },
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
            Ok(Err(e)) => {
                eprintln!("[GPU] GpuFillPreviewPipeline readback map error: {:?}", e);
                return;
            }
            Err(e) => {
                eprintln!("[GPU] GpuFillPreviewPipeline readback channel error: {:?}", e);
                return;
            }
        }

        let mapped = slice.get_mapped_range();
        let actual_row = region_w as usize * 4;
        out.clear();
        out.resize(actual_row * region_h as usize, 0);
        for y in 0..region_h as usize {
            let src_row = y * bytes_per_row as usize;
            let dst_row = y * actual_row;
            out[dst_row..dst_row + actual_row]
                .copy_from_slice(&mapped[src_row..src_row + actual_row]);
        }

        drop(mapped);
        staging.unmap();
    }
}

// ============================================================================
// GPU FLOOD FILL PIPELINE — per-pixel color distance + iterative relaxation
// ============================================================================

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct FloodColorDistGpuParams {
    target_r: u32,
    target_g: u32,
    target_b: u32,
    target_a: u32,
    width: u32,
    height: u32,
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct FloodInitGpuParams {
    seed_x: u32,
    seed_y: u32,
    width: u32,
    height: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct FloodStepGpuParams {
    width: u32,
    height: u32,
    step_size: u32,
    direction: u32,
}

pub struct GpuFloodFillPipeline {
    color_dist_pipeline: wgpu::ComputePipeline,
    color_dist_bgl: wgpu::BindGroupLayout,
    flood_init_pipeline: wgpu::ComputePipeline,
    flood_init_bgl: wgpu::BindGroupLayout,
    flood_step_pipeline: wgpu::ComputePipeline,
    flood_step_bgl: wgpu::BindGroupLayout,
    // Cached GPU resources
    cached_input_tex: Option<wgpu::Texture>,
    cached_color_dist_buf: Option<wgpu::Buffer>,
    cached_flood_a_buf: Option<wgpu::Buffer>,
    cached_flood_b_buf: Option<wgpu::Buffer>,
    cached_staging_buf: Option<wgpu::Buffer>,
    cached_params_buf_cd: Option<wgpu::Buffer>,
    cached_params_buf_init: Option<wgpu::Buffer>,
    cached_w: u32,
    cached_h: u32,
    cached_input_key: usize,
}

impl GpuFloodFillPipeline {
    pub fn new(device: &wgpu::Device) -> Self {
        // --- Color Distance pipeline ---
        let cd_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("flood_color_dist_shader"),
            source: wgpu::ShaderSource::Wgsl(
                super::shaders::FLOOD_COLOR_DISTANCE_SHADER.into(),
            ),
        });

        let color_dist_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("flood_color_dist_bgl"),
                entries: &[
                    // binding 0: input RGBA texture
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float {
                                filterable: false,
                            },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // binding 1: color_dist storage buffer (write)
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // binding 2: uniform params
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let cd_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("flood_color_dist_pl"),
            bind_group_layouts: &[&color_dist_bgl],
            push_constant_ranges: &[],
        });

        let color_dist_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("flood_color_dist_pipeline"),
                layout: Some(&cd_layout),
                module: &cd_shader,
                entry_point: "cs_color_distance",
                compilation_options: Default::default(),
            });

        // --- Flood Init pipeline ---
        let init_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("flood_init_shader"),
            source: wgpu::ShaderSource::Wgsl(super::shaders::FLOOD_INIT_SHADER.into()),
        });

        let flood_init_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("flood_init_bgl"),
                entries: &[
                    // binding 0: color_dist buffer (read)
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // binding 1: flood_dist buffer (read_write)
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // binding 2: uniform
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let init_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("flood_init_pl"),
            bind_group_layouts: &[&flood_init_bgl],
            push_constant_ranges: &[],
        });

        let flood_init_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("flood_init_pipeline"),
                layout: Some(&init_layout),
                module: &init_shader,
                entry_point: "cs_flood_init",
                compilation_options: Default::default(),
            });

        // --- Flood Step pipeline ---
        let step_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("flood_step_shader"),
            source: wgpu::ShaderSource::Wgsl(super::shaders::FLOOD_STEP_SHADER.into()),
        });

        let flood_step_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("flood_step_bgl"),
                entries: &[
                    // binding 0: color_dist (read)
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // binding 1: flood_a (read_write)
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // binding 2: flood_b (read_write)
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // binding 3: uniform
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let step_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("flood_step_pl"),
            bind_group_layouts: &[&flood_step_bgl],
            push_constant_ranges: &[],
        });

        let flood_step_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("flood_step_pipeline"),
                layout: Some(&step_layout),
                module: &step_shader,
                entry_point: "cs_flood_step",
                compilation_options: Default::default(),
            });

        Self {
            color_dist_pipeline,
            color_dist_bgl,
            flood_init_pipeline,
            flood_init_bgl,
            flood_step_pipeline,
            flood_step_bgl,
            cached_input_tex: None,
            cached_color_dist_buf: None,
            cached_flood_a_buf: None,
            cached_flood_b_buf: None,
            cached_staging_buf: None,
            cached_params_buf_cd: None,
            cached_params_buf_init: None,
            cached_w: 0,
            cached_h: 0,
            cached_input_key: 0,
        }
    }

    fn ensure_buffers(&mut self, device: &wgpu::Device, w: u32, h: u32) {
        if self.cached_w == w && self.cached_h == h {
            return;
        }
        let n = (w as u64) * (h as u64);
        let buf_size = n * 4; // u32 per pixel

        self.cached_color_dist_buf = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flood_color_dist_buf"),
            size: buf_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
        self.cached_flood_a_buf = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flood_a_buf"),
            size: buf_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
        self.cached_flood_b_buf = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flood_b_buf"),
            size: buf_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
        self.cached_staging_buf = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flood_staging_buf"),
            size: buf_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        self.cached_input_tex = None;
        self.cached_input_key = 0;
        self.cached_w = w;
        self.cached_h = h;
    }

    /// Compute the flood-fill minimax distance map on GPU.
    /// Returns the distance map as `Vec<u8>` (one byte per pixel, 0–255).
    /// This replaces the CPU `compute_flood_distance_map` Dijkstra.
    pub fn compute_flood_distances(
        &mut self,
        ctx: &GpuContext,
        flat_rgba: &[u8],
        input_key: usize,
        target_color: [u8; 4],
        seed_x: u32,
        seed_y: u32,
        w: u32,
        h: u32,
        out: &mut Vec<u8>,
    ) -> bool {
        let device = &ctx.device;
        let queue = &ctx.queue;

        self.ensure_buffers(device, w, h);

        // Upload input RGBA texture if changed
        if self.cached_input_tex.is_none() || self.cached_input_key != input_key {
            self.cached_input_tex = Some(upload_rgba(
                device,
                queue,
                flat_rgba,
                w,
                h,
                "flood_input_rgba",
            ));
            self.cached_input_key = input_key;
        }

        let wg_x = w.div_ceil(16);
        let wg_y = h.div_ceil(16);

        // === Phase 1: Compute per-pixel color distances ===
        let cd_params = FloodColorDistGpuParams {
            target_r: target_color[0] as u32,
            target_g: target_color[1] as u32,
            target_b: target_color[2] as u32,
            target_a: target_color[3] as u32,
            width: w,
            height: h,
            _pad0: 0,
            _pad1: 0,
        };
        let cd_params_bytes = bytemuck::bytes_of(&cd_params);
        let cd_params_buf = self.cached_params_buf_cd.get_or_insert_with(|| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("flood_cd_params"),
                contents: cd_params_bytes,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
        });
        queue.write_buffer(cd_params_buf, 0, cd_params_bytes);

        let input_view = self
            .cached_input_tex
            .as_ref()
            .unwrap()
            .create_view(&wgpu::TextureViewDescriptor::default());
        let color_dist_buf = self.cached_color_dist_buf.as_ref().unwrap();
        let flood_a_buf = self.cached_flood_a_buf.as_ref().unwrap();
        let flood_b_buf = self.cached_flood_b_buf.as_ref().unwrap();

        let cd_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("flood_cd_bg"),
            layout: &self.color_dist_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&input_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: color_dist_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: cd_params_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("flood_cd_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("flood_cd_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.color_dist_pipeline);
            pass.set_bind_group(0, &cd_bg, &[]);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));

        // === Phase 2: Initialize flood distances ===
        let init_params = FloodInitGpuParams {
            seed_x,
            seed_y,
            width: w,
            height: h,
        };
        let init_params_bytes = bytemuck::bytes_of(&init_params);
        let init_params_buf = self.cached_params_buf_init.get_or_insert_with(|| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("flood_init_params"),
                contents: init_params_bytes,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
        });
        queue.write_buffer(init_params_buf, 0, init_params_bytes);

        let init_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("flood_init_bg"),
            layout: &self.flood_init_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: color_dist_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: flood_a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: init_params_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("flood_init_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("flood_init_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.flood_init_pipeline);
            pass.set_bind_group(0, &init_bg, &[]);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));

        // === Phase 3: Iterative relaxation with step_size=1 ONLY ===
        // Must use step_size=1 for correct 4-connected flood fill.
        // Large step sizes (JFA-style) jump over barriers, connecting
        // disconnected regions and creating power-of-2. grid artifacts.
        // Number of passes = w+h (upper bound on grid graph diameter).
        let num_passes = (w + h) as usize;

        // Pre-create param buffers for both ping-pong directions.
        // These are immutable — no write_buffer needed between passes.
        let params_buf_fwd =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("flood_step_params_fwd"),
                contents: bytemuck::bytes_of(&FloodStepGpuParams {
                    width: w,
                    height: h,
                    step_size: 1,
                    direction: 0,
                }),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let params_buf_bwd =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("flood_step_params_bwd"),
                contents: bytemuck::bytes_of(&FloodStepGpuParams {
                    width: w,
                    height: h,
                    step_size: 1,
                    direction: 1,
                }),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // Two bind groups sharing the same storage buffers but different uniform params.
        let bg_fwd = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("flood_step_bg_fwd"),
            layout: &self.flood_step_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: color_dist_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: flood_a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: flood_b_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buf_fwd.as_entire_binding(),
                },
            ],
        });
        let bg_bwd = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("flood_step_bg_bwd"),
            layout: &self.flood_step_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: color_dist_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: flood_a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: flood_b_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buf_bwd.as_entire_binding(),
                },
            ],
        });

        // Batch passes into encoder submissions.
        // Multiple compute passes per encoder are correct here — wgpu inserts
        // implicit storage-buffer barriers at compute pass boundaries, and we
        // use pre-built bind groups (no write_buffer between passes).
        let batch_size = 1000;
        let mut direction = 0u32;
        for chunk_start in (0..num_passes).step_by(batch_size) {
            let chunk_end = num_passes.min(chunk_start + batch_size);
            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("flood_step_batch"),
                });
            for _ in chunk_start..chunk_end {
                {
                    let mut pass =
                        encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("flood_step_pass"),
                            timestamp_writes: None,
                        });
                    pass.set_pipeline(&self.flood_step_pipeline);
                    if direction == 0 {
                        pass.set_bind_group(0, &bg_fwd, &[]);
                    } else {
                        pass.set_bind_group(0, &bg_bwd, &[]);
                    }
                    pass.dispatch_workgroups(wg_x, wg_y, 1);
                }
                direction ^= 1;
            }
            queue.submit(std::iter::once(encoder.finish()));
        }

        // === Phase 4: Read back result ===
        // The final distances are in flood_a (if direction==0) or flood_b (if direction==1)
        let result_buf = if direction == 0 {
            flood_a_buf
        } else {
            flood_b_buf
        };

        let staging = self.cached_staging_buf.as_ref().unwrap();
        let buf_size = (w as u64) * (h as u64) * 4;

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("flood_readback_encoder"),
        });
        encoder.copy_buffer_to_buffer(result_buf, 0, staging, 0, buf_size);
        queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        device.poll(wgpu::Maintain::Wait);

        match rx.recv() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                eprintln!("[GPU] GpuFloodFillPipeline readback map error: {:?}", e);
                out.clear();
                return false;
            }
            Err(e) => {
                eprintln!("[GPU] GpuFloodFillPipeline readback channel error: {:?}", e);
                out.clear();
                return false;
            }
        }

        let mapped = slice.get_mapped_range();
        let n = (w * h) as usize;
        out.clear();
        out.resize(n, 0);
        // Storage buffer contains u32 per pixel; extract low byte (0–255)
        let src: &[u32] = bytemuck::cast_slice(&mapped[..n * 4]);
        for (i, &val) in src.iter().enumerate() {
            out[i] = val.min(255) as u8;
        }

        drop(mapped);
        staging.unmap();
        true
    }

    /// Compute per-pixel global distance map on GPU (no connectivity, no flood).
    /// Used for Magic Wand global select mode (Ctrl+Shift).
    pub fn compute_global_distances(
        &mut self,
        ctx: &GpuContext,
        flat_rgba: &[u8],
        input_key: usize,
        target_color: [u8; 4],
        w: u32,
        h: u32,
        out: &mut Vec<u8>,
    ) -> bool {
        let device = &ctx.device;
        let queue = &ctx.queue;

        self.ensure_buffers(device, w, h);

        if self.cached_input_tex.is_none() || self.cached_input_key != input_key {
            self.cached_input_tex = Some(upload_rgba(
                device,
                queue,
                flat_rgba,
                w,
                h,
                "flood_input_rgba",
            ));
            self.cached_input_key = input_key;
        }

        let cd_params = FloodColorDistGpuParams {
            target_r: target_color[0] as u32,
            target_g: target_color[1] as u32,
            target_b: target_color[2] as u32,
            target_a: target_color[3] as u32,
            width: w,
            height: h,
            _pad0: 0,
            _pad1: 0,
        };
        let cd_params_bytes = bytemuck::bytes_of(&cd_params);
        let cd_params_buf = self.cached_params_buf_cd.get_or_insert_with(|| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("flood_cd_params"),
                contents: cd_params_bytes,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
        });
        queue.write_buffer(cd_params_buf, 0, cd_params_bytes);

        let input_view = self
            .cached_input_tex
            .as_ref()
            .unwrap()
            .create_view(&wgpu::TextureViewDescriptor::default());
        let color_dist_buf = self.cached_color_dist_buf.as_ref().unwrap();

        let cd_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("flood_cd_bg"),
            layout: &self.color_dist_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&input_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: color_dist_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: cd_params_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("flood_global_cd_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("flood_global_cd_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.color_dist_pipeline);
            pass.set_bind_group(0, &cd_bg, &[]);
            pass.dispatch_workgroups(w.div_ceil(16), h.div_ceil(16), 1);
        }

        // Copy color_dist to staging for readback
        let staging = self.cached_staging_buf.as_ref().unwrap();
        let buf_size = (w as u64) * (h as u64) * 4;
        encoder.copy_buffer_to_buffer(color_dist_buf, 0, staging, 0, buf_size);
        queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        device.poll(wgpu::Maintain::Wait);

        match rx.recv() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                eprintln!("[GPU] GpuFloodFillPipeline global readback map error: {:?}", e);
                out.clear();
                return false;
            }
            Err(e) => {
                eprintln!("[GPU] GpuFloodFillPipeline global readback channel error: {:?}", e);
                out.clear();
                return false;
            }
        }

        let mapped = slice.get_mapped_range();
        let n = (w * h) as usize;
        out.clear();
        out.resize(n, 0);
        let src: &[u32] = bytemuck::cast_slice(&mapped[..n * 4]);
        for (i, &val) in src.iter().enumerate() {
            out[i] = val.min(255) as u8;
        }

        drop(mapped);
        staging.unmap();
        true
    }
}

impl GpuGradientPipeline {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gradient_shader"),
            source: wgpu::ShaderSource::Wgsl(super::shaders::GRADIENT_SHADER.into()),
        });

        // Custom bind group layout: no input texture
        // 0: storage texture (output, write-only)
        // 1: uniform buffer (params)
        // 2: storage buffer (LUT, read-only)
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gradient_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gradient_pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gradient_pipeline"),
            layout: Some(&layout),
            module: &shader,
            entry_point: "cs_gradient",
            compilation_options: Default::default(),
        });

        Self {
            pipeline,
            bgl,
            cached_output_tex: None,
            cached_staging_buf: None,
            cached_params_buf: None,
            cached_lut_buf: None,
            cached_w: 0,
            cached_h: 0,
        }
    }

    /// Ensure cached output texture and staging buffer match the requested size.
    fn ensure_cache(&mut self, device: &wgpu::Device, w: u32, h: u32) {
        if self.cached_w != w || self.cached_h != h {
            self.cached_output_tex = Some(create_rw_texture(device, w, h, "gradient_output"));

            let bytes_per_row = super::compositor::Compositor::aligned_bytes_per_row(w);
            let buffer_size = (bytes_per_row * h) as u64;
            self.cached_staging_buf = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("gradient_staging"),
                size: buffer_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.cached_w = w;
            self.cached_h = h;
        }
    }

    /// Run the gradient compute shader and readback the result into `out`.
    /// `lut_rgba` must be exactly 256 × 4 bytes (un-premultiplied RGBA).
    /// The output buffer is resized and reused across frames to avoid allocation.
    pub fn generate_into(
        &mut self,
        ctx: &GpuContext,
        params: &GradientGpuParams,
        lut_rgba: &[u8], // 256 * 4 = 1024 bytes
        out: &mut Vec<u8>,
    ) {
        let device = &ctx.device;
        let queue = &ctx.queue;
        let w = params.width;
        let h = params.height;

        // Ensure cached output texture + staging buffer are the right size
        self.ensure_cache(device, w, h);

        // Pack the LUT into u32 array (little-endian RGBA)
        let mut lut_packed = [0u32; 256];
        for (i, item) in lut_packed.iter_mut().enumerate() {
            let off = i * 4;
            *item = (lut_rgba[off] as u32)
                | ((lut_rgba[off + 1] as u32) << 8)
                | ((lut_rgba[off + 2] as u32) << 16)
                | ((lut_rgba[off + 3] as u32) << 24);
        }

        let dst_tex = self.cached_output_tex.as_ref().unwrap();
        let staging = self.cached_staging_buf.as_ref().unwrap();

        // Reuse cached GPU buffers for params and LUT
        let params_bytes = bytemuck::bytes_of(params);
        let lut_bytes = bytemuck::cast_slice(&lut_packed);
        let params_buf = self.cached_params_buf.get_or_insert_with(|| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("gradient_params"),
                contents: params_bytes,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
        });
        queue.write_buffer(params_buf, 0, params_bytes);

        let lut_buf = self.cached_lut_buf.get_or_insert_with(|| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("gradient_lut"),
                contents: lut_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            })
        });
        queue.write_buffer(lut_buf, 0, lut_bytes);

        let dst_view = dst_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gradient_bg"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&dst_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: lut_buf.as_entire_binding(),
                },
            ],
        });

        let bytes_per_row = super::compositor::Compositor::aligned_bytes_per_row(w);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gradient_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gradient_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(w.div_ceil(16), h.div_ceil(16), 1);
        }

        // Copy output texture → cached staging buffer
        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: dst_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: staging,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );

        queue.submit(std::iter::once(encoder.finish()));

        // Map and readback directly into the caller's buffer
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        device.poll(wgpu::Maintain::Wait);
        match rx.recv() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                eprintln!("[GPU] GpuGradientPipeline readback map error: {:?}", e);
                return;
            }
            Err(e) => {
                eprintln!("[GPU] GpuGradientPipeline readback channel error: {:?}", e);
                return;
            }
        }

        let mapped = slice.get_mapped_range();
        let actual_row = w as usize * 4;
        let total = actual_row * h as usize;

        out.clear();
        out.resize(total, 0);
        for y in 0..h as usize {
            let src_start = y * bytes_per_row as usize;
            let dst_start = y * actual_row;
            out[dst_start..dst_start + actual_row]
                .copy_from_slice(&mapped[src_start..src_start + actual_row]);
        }

        drop(mapped);
        staging.unmap();
    }
}

// ============================================================================
// LIQUIFY WARP PIPELINE — GPU displacement warp for Liquify tool
// ============================================================================

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct LiquifyGpuParams {
    pub width: u32,
    pub height: u32,
}

pub struct GpuLiquifyPipeline {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    // Cached GPU resources — reused when dimensions match
    cached_source_tex: Option<wgpu::Texture>,
    cached_output_tex: Option<wgpu::Texture>,
    cached_staging_buf: Option<wgpu::Buffer>,
    cached_disp_buf: Option<wgpu::Buffer>,
    cached_params_buf: Option<wgpu::Buffer>,
    cached_w: u32,
    cached_h: u32,
    /// When true, the cached source texture needs re-uploading
    /// (e.g. new stroke started with a fresh snapshot).
    source_dirty: bool,
}

impl GpuLiquifyPipeline {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("liquify_warp_shader"),
            source: wgpu::ShaderSource::Wgsl(super::shaders::LIQUIFY_WARP_SHADER.into()),
        });

        // Bind group layout:
        //  0: source texture (texture_2d<f32>, read)
        //  1: output storage texture (write-only, rgba8unorm)
        //  2: displacement storage buffer (read-only, array<f32>)
        //  3: uniform buffer (LiquifyGpuParams)
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("liquify_bgl"),
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
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("liquify_pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("liquify_pipeline"),
            layout: Some(&layout),
            module: &shader,
            entry_point: "cs_liquify_warp",
            compilation_options: Default::default(),
        });

        Self {
            pipeline,
            bgl,
            cached_source_tex: None,
            cached_output_tex: None,
            cached_staging_buf: None,
            cached_disp_buf: None,
            cached_params_buf: None,
            cached_w: 0,
            cached_h: 0,
            source_dirty: true,
        }
    }

    /// Ensure cached resources match the requested dimensions.
    fn ensure_cache(&mut self, device: &wgpu::Device, w: u32, h: u32) {
        if self.cached_w != w || self.cached_h != h {
            // Source texture (will be written via queue.write_texture)
            self.cached_source_tex = Some(device.create_texture(&wgpu::TextureDescriptor {
                label: Some("liquify_source"),
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            }));

            // Output texture
            self.cached_output_tex = Some(create_rw_texture(device, w, h, "liquify_output"));

            // Staging buffer for readback (256-byte aligned rows)
            let bytes_per_row = super::compositor::Compositor::aligned_bytes_per_row(w);
            let buffer_size = (bytes_per_row * h) as u64;
            self.cached_staging_buf = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("liquify_staging"),
                size: buffer_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));

            // Displacement storage buffer: width * height * 2 floats
            let disp_size = (w as usize * h as usize * 2 * std::mem::size_of::<f32>()) as u64;
            self.cached_disp_buf = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("liquify_disp"),
                size: disp_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));

            // Params buffer will be created/reused on first dispatch
            self.cached_params_buf = None;

            self.cached_w = w;
            self.cached_h = h;
            self.source_dirty = true; // dimensions changed, need re-upload
        }
    }

    /// Mark the source texture as needing re-upload (call when a new stroke starts).
    pub fn invalidate_source(&mut self) {
        self.source_dirty = true;
    }

    /// Run the displacement warp on GPU and write the result into `out`.
    ///
    /// `source_rgba` — raw RGBA pixels of the source snapshot.
    /// `displacement_data` — flat (dx,dy) pairs from DisplacementField.data.
    /// `w, h` — canvas dimensions.
    /// `out` — destination buffer; resized to `w * h * 4`.
    pub fn warp_into(
        &mut self,
        ctx: &GpuContext,
        source_rgba: &[u8],
        displacement_data: &[f32],
        w: u32,
        h: u32,
        out: &mut Vec<u8>,
    ) {
        let device = &ctx.device;
        let queue = &ctx.queue;

        self.ensure_cache(device, w, h);

        let src_tex = match self.cached_source_tex.as_ref() {
            Some(t) => t,
            None => {
                eprintln!("[GPU] GpuLiquifyPipeline: cached_source_tex not initialised — skipping");
                return;
            }
        };
        let dst_tex = match self.cached_output_tex.as_ref() {
            Some(t) => t,
            None => {
                eprintln!("[GPU] GpuLiquifyPipeline: cached_output_tex not initialised — skipping");
                return;
            }
        };
        let staging = match self.cached_staging_buf.as_ref() {
            Some(b) => b,
            None => {
                eprintln!(
                    "[GPU] GpuLiquifyPipeline: cached_staging_buf not initialised — skipping"
                );
                return;
            }
        };
        let disp_buf = match self.cached_disp_buf.as_ref() {
            Some(b) => b,
            None => {
                eprintln!("[GPU] GpuLiquifyPipeline: cached_disp_buf not initialised — skipping");
                return;
            }
        };

        // Upload source image (only when dirty — stays constant during a stroke)
        if self.source_dirty {
            queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture: src_tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                source_rgba,
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(4 * w),
                    rows_per_image: Some(h),
                },
                wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
            );
            self.source_dirty = false;
        }

        // Upload displacement field every frame
        queue.write_buffer(disp_buf, 0, bytemuck::cast_slice(displacement_data));

        // Params uniform
        let params = LiquifyGpuParams {
            width: w,
            height: h,
        };
        let params_bytes = bytemuck::bytes_of(&params);
        let params_buf = self.cached_params_buf.get_or_insert_with(|| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("liquify_params"),
                contents: params_bytes,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
        });
        queue.write_buffer(params_buf, 0, params_bytes);

        // Build bind group
        let src_view = src_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let dst_view = dst_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("liquify_bg"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&src_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&dst_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: disp_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        let bytes_per_row = super::compositor::Compositor::aligned_bytes_per_row(w);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("liquify_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("liquify_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(w.div_ceil(16), h.div_ceil(16), 1);
        }

        // Copy output texture → staging buffer
        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: dst_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: staging,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );

        queue.submit(std::iter::once(encoder.finish()));

        // Map and readback
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        device.poll(wgpu::Maintain::Wait);
        match rx.recv() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                eprintln!("[GPU] GpuLiquifyPipeline readback map error: {:?}", e);
                return;
            }
            Err(e) => {
                eprintln!("[GPU] GpuLiquifyPipeline readback channel error: {:?}", e);
                return;
            }
        }

        let mapped = slice.get_mapped_range();
        let actual_row = w as usize * 4;
        let total = actual_row * h as usize;

        out.clear();
        out.resize(total, 0);
        for y in 0..h as usize {
            let src_start = y * bytes_per_row as usize;
            let dst_start = y * actual_row;
            out[dst_start..dst_start + actual_row]
                .copy_from_slice(&mapped[src_start..src_start + actual_row]);
        }

        drop(mapped);
        staging.unmap();
    }
}

// ============================================================================
// MESH WARP DISPLACEMENT PIPELINE — GPU Catmull-Rom displacement field generation
// ============================================================================

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct MeshWarpGpuParams {
    pub width: u32,
    pub height: u32,
    pub grid_cols: u32,
    pub grid_rows: u32,
}

pub struct GpuMeshWarpDisplacementPipeline {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    cached_points_buf: Option<wgpu::Buffer>,
    cached_disp_out_buf: Option<wgpu::Buffer>,
    cached_staging_buf: Option<wgpu::Buffer>,
    cached_params_buf: Option<wgpu::Buffer>,
    cached_w: u32,
    cached_h: u32,
}

impl GpuMeshWarpDisplacementPipeline {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mesh_warp_displacement_shader"),
            source: wgpu::ShaderSource::Wgsl(super::shaders::MESH_WARP_DISPLACEMENT_SHADER.into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mesh_warp_disp_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mesh_warp_disp_pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("mesh_warp_disp_pipeline"),
            layout: Some(&layout),
            module: &shader,
            entry_point: "cs_mesh_warp_displacement",
            compilation_options: Default::default(),
        });

        Self {
            pipeline,
            bgl,
            cached_points_buf: None,
            cached_disp_out_buf: None,
            cached_staging_buf: None,
            cached_params_buf: None,
            cached_w: 0,
            cached_h: 0,
        }
    }

    fn ensure_cache(&mut self, device: &wgpu::Device, w: u32, h: u32, num_points: usize) {
        if self.cached_w != w || self.cached_h != h {
            let disp_size = (w as usize * h as usize * 2 * std::mem::size_of::<f32>()) as u64;
            self.cached_disp_out_buf = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("mw_disp_out"),
                size: disp_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }));
            self.cached_staging_buf = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("mw_disp_staging"),
                size: disp_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.cached_params_buf = None;
            self.cached_w = w;
            self.cached_h = h;
        }

        let pts_bytes = (num_points * 2 * std::mem::size_of::<f32>()) as u64;
        let need_pts = match &self.cached_points_buf {
            Some(b) => b.size() < pts_bytes,
            None => true,
        };
        if need_pts {
            self.cached_points_buf = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("mw_points"),
                size: pts_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
    }

    /// Generate displacement field on GPU from deformed control points.
    pub fn generate_displacement(
        &mut self,
        ctx: &GpuContext,
        deformed_points: &[[f32; 2]],
        grid_cols: u32,
        grid_rows: u32,
        w: u32,
        h: u32,
        out: &mut Vec<f32>,
    ) {
        let device = &ctx.device;
        let queue = &ctx.queue;
        let num_points = deformed_points.len();

        self.ensure_cache(device, w, h, num_points);

        let pts_buf = match self.cached_points_buf.as_ref() {
            Some(b) => b,
            None => {
                eprintln!(
                    "[GPU] GpuMeshWarpDisplacementPipeline: cached_points_buf not initialised — skipping"
                );
                return;
            }
        };
        let disp_buf = match self.cached_disp_out_buf.as_ref() {
            Some(b) => b,
            None => {
                eprintln!(
                    "[GPU] GpuMeshWarpDisplacementPipeline: cached_disp_out_buf not initialised — skipping"
                );
                return;
            }
        };
        let staging = match self.cached_staging_buf.as_ref() {
            Some(b) => b,
            None => {
                eprintln!(
                    "[GPU] GpuMeshWarpDisplacementPipeline: cached_staging_buf not initialised — skipping"
                );
                return;
            }
        };

        queue.write_buffer(pts_buf, 0, bytemuck::cast_slice(deformed_points));

        let params = MeshWarpGpuParams {
            width: w,
            height: h,
            grid_cols,
            grid_rows,
        };
        let params_bytes = bytemuck::bytes_of(&params);
        let params_buf = self.cached_params_buf.get_or_insert_with(|| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("mw_params"),
                contents: params_bytes,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
        });
        queue.write_buffer(params_buf, 0, params_bytes);

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mw_disp_bg"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: pts_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: disp_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        let disp_byte_size = (w as usize * h as usize * 2 * std::mem::size_of::<f32>()) as u64;

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("mw_disp_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("mw_disp_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(w.div_ceil(16), h.div_ceil(16), 1);
        }
        encoder.copy_buffer_to_buffer(disp_buf, 0, staging, 0, disp_byte_size);
        queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..disp_byte_size);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        device.poll(wgpu::Maintain::Wait);
        match rx.recv() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                eprintln!(
                    "[GPU] GpuMeshWarpDisplacementPipeline readback map error: {:?}",
                    e
                );
                return;
            }
            Err(e) => {
                eprintln!(
                    "[GPU] GpuMeshWarpDisplacementPipeline readback channel error: {:?}",
                    e
                );
                return;
            }
        }

        let mapped = slice.get_mapped_range();
        let float_count = w as usize * h as usize * 2;
        out.resize(float_count, 0.0);
        let src_floats: &[f32] = bytemuck::cast_slice(&mapped);
        out.copy_from_slice(&src_floats[..float_count]);

        drop(mapped);
        staging.unmap();
    }
}
