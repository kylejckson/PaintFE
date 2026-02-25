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
