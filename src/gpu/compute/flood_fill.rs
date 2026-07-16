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
    distance_mode: u32,
    width: u32,
    height: u32,
    _pad0: u32,
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
    connectivity: u32,
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
    cached_changed_buf: Option<wgpu::Buffer>,
    cached_changed_staging_buf: Option<wgpu::Buffer>,
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
            source: wgpu::ShaderSource::Wgsl(super::shaders::FLOOD_COLOR_DISTANCE_SHADER.into()),
        });

        let color_dist_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("flood_color_dist_bgl"),
            entries: &[
                // binding 0: input RGBA texture
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
            bind_group_layouts: &[Some(&color_dist_bgl)],
            immediate_size: 0,
        });

        let color_dist_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("flood_color_dist_pipeline"),
                layout: Some(&cd_layout),
                module: &cd_shader,
                entry_point: Some("cs_color_distance"),
                compilation_options: Default::default(),
                cache: None,
            });

        // --- Flood Init pipeline ---
        let init_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("flood_init_shader"),
            source: wgpu::ShaderSource::Wgsl(super::shaders::FLOOD_INIT_SHADER.into()),
        });

        let flood_init_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
            bind_group_layouts: &[Some(&flood_init_bgl)],
            immediate_size: 0,
        });

        let flood_init_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("flood_init_pipeline"),
                layout: Some(&init_layout),
                module: &init_shader,
                entry_point: Some("cs_flood_init"),
                compilation_options: Default::default(),
                cache: None,
            });

        // --- Flood Step pipeline ---
        let step_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("flood_step_shader"),
            source: wgpu::ShaderSource::Wgsl(super::shaders::FLOOD_STEP_SHADER.into()),
        });

        let flood_step_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
                // binding 4: changed flag (read_write)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let step_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("flood_step_pl"),
            bind_group_layouts: &[Some(&flood_step_bgl)],
            immediate_size: 0,
        });

        let flood_step_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("flood_step_pipeline"),
                layout: Some(&step_layout),
                module: &step_shader,
                entry_point: Some("cs_flood_step"),
                compilation_options: Default::default(),
                cache: None,
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
            cached_changed_buf: None,
            cached_changed_staging_buf: None,
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
        self.cached_changed_buf = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flood_changed_buf"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        self.cached_changed_staging_buf = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flood_changed_staging_buf"),
            size: 4,
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
        distance_mode: WandDistanceMode,
        connectivity: FloodConnectivity,
        out: &mut Vec<u8>,
    ) -> bool {
        // The readback below blocks on `device.poll(Wait)`, which cannot
        // synchronously complete in the browser (no real thread parking).
        // Report "unavailable" so the caller takes its existing CPU fallback
        // path instead of hanging.
        #[cfg(target_arch = "wasm32")]
        {
            let _ = (
                ctx,
                flat_rgba,
                input_key,
                target_color,
                seed_x,
                seed_y,
                w,
                h,
                distance_mode,
                connectivity,
                out,
            );
            false
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
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
                distance_mode: if distance_mode == WandDistanceMode::Perceptual {
                    1
                } else {
                    0
                },
                width: w,
                height: h,
                _pad0: 0,
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
            let changed_buf = self.cached_changed_buf.as_ref().unwrap();

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
            let params_buf_fwd = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("flood_step_params_fwd"),
                contents: bytemuck::bytes_of(&FloodStepGpuParams {
                    width: w,
                    height: h,
                    step_size: 1,
                    direction: 0,
                    connectivity: if connectivity == FloodConnectivity::Eight {
                        8
                    } else {
                        4
                    },
                }),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let params_buf_bwd = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("flood_step_params_bwd"),
                contents: bytemuck::bytes_of(&FloodStepGpuParams {
                    width: w,
                    height: h,
                    step_size: 1,
                    direction: 1,
                    connectivity: if connectivity == FloodConnectivity::Eight {
                        8
                    } else {
                        4
                    },
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
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: changed_buf.as_entire_binding(),
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
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: changed_buf.as_entire_binding(),
                    },
                ],
            });

            // Batch passes into encoder submissions.
            // Multiple compute passes per encoder are correct here — wgpu inserts
            // implicit storage-buffer barriers at compute pass boundaries, and we
            // use pre-built bind groups (no write_buffer between passes).
            let batch_size = 64;
            let mut direction = 0u32;
            let mut chunk_start = 0usize;
            while chunk_start < num_passes {
                let chunk_end = num_passes.min(chunk_start + batch_size);

                // Reset convergence flag for this batch.
                queue.write_buffer(changed_buf, 0, bytemuck::bytes_of(&0u32));

                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("flood_step_batch"),
                });
                for _ in chunk_start..chunk_end {
                    {
                        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
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

                // If nothing changed in this batch, flood distances converged.
                let changed_staging = self.cached_changed_staging_buf.as_ref().unwrap();
                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("flood_changed_readback_encoder"),
                });
                encoder.copy_buffer_to_buffer(changed_buf, 0, changed_staging, 0, 4);
                queue.submit(std::iter::once(encoder.finish()));

                let changed_slice = changed_staging.slice(..);
                let (tx_changed, rx_changed) = std::sync::mpsc::channel();
                changed_slice.map_async(wgpu::MapMode::Read, move |result| {
                    let _ = tx_changed.send(result);
                });
                let _ = device.poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: None,
                });

                let changed_value = match rx_changed.recv() {
                    Ok(Ok(())) => {
                        let mapped = changed_slice.get_mapped_range();
                        let vals: &[u32] = bytemuck::cast_slice(&mapped);
                        let changed = vals.first().copied().unwrap_or(1);
                        drop(mapped);
                        changed_staging.unmap();
                        changed
                    }
                    Ok(Err(_)) | Err(_) => 1,
                };

                if changed_value == 0 {
                    break;
                }

                chunk_start = chunk_end;
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
            let _ = device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            });

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
        distance_mode: WandDistanceMode,
        out: &mut Vec<u8>,
    ) -> bool {
        // See compute_flood_distances above: blocking readback can't work on web.
        #[cfg(target_arch = "wasm32")]
        {
            let _ = (
                ctx,
                flat_rgba,
                input_key,
                target_color,
                w,
                h,
                distance_mode,
                out,
            );
            false
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
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
                distance_mode: if distance_mode == WandDistanceMode::Perceptual {
                    1
                } else {
                    0
                },
                width: w,
                height: h,
                _pad0: 0,
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
            let _ = device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            });

            match rx.recv() {
                Ok(Ok(())) => {}
                Ok(Err(e)) => {
                    eprintln!(
                        "[GPU] GpuFloodFillPipeline global readback map error: {:?}",
                        e
                    );
                    out.clear();
                    return false;
                }
                Err(e) => {
                    eprintln!(
                        "[GPU] GpuFloodFillPipeline global readback channel error: {:?}",
                        e
                    );
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
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gradient_pipeline"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("cs_gradient"),
            compilation_options: Default::default(),
            cache: None,
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
            wgpu::TexelCopyTextureInfo {
                texture: dst_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: staging,
                layout: wgpu::TexelCopyBufferLayout {
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
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
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
