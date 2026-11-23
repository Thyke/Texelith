//! In-place radix-2 FFT for periodic power-of-two grids. No padding or boundary
//! approximation: other grids continue through the Galerkin multigrid path.
use super::*;
use crate::{NodeUniform, executor::common_layout};
use wgpu::util::DeviceExt;

pub(super) fn supports(output: &wgpu::Texture, boundary: Boundary) -> bool {
    boundary == Boundary::Repeat
        && output.width().is_power_of_two()
        && output.height().is_power_of_two()
}

#[derive(Clone)]
struct Pass {
    pipeline: wgpu::ComputePipeline,
    common: wgpu::BindGroup,
    extra: wgpu::BindGroup,
    groups: [u32; 2],
}
impl Pass {
    fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Poisson spectral"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.common, &[]);
        pass.set_bind_group(1, &self.extra, &[]);
        pass.dispatch_workgroups(self.groups[0], self.groups[1], 1);
    }
}
struct Workspace {
    first: Vec<Pass>,
    refine: Vec<Pass>,
    transform: Vec<Pass>,
    accumulate: Pass,
    write: Pass,
    diagnostics: Vec<Pass>,
    certificate: Vec<Pass>,
    result: wgpu::Buffer,
    readback: wgpu::Buffer,
    _reservations: Reservations,
}
// Pipeline objects contain no material operands or ledger allocations. Keep them
// across workspace eviction/rebinding instead of recompiling on every request.
pub(super) struct Kernel {
    common: wgpu::BindGroupLayout,
    extra: wgpu::BindGroupLayout,
    pipelines: Vec<wgpu::ComputePipeline>,
}
impl Kernel {
    fn new(device: &wgpu::Device) -> Self {
        let common = common_layout(device, wgpu::TextureFormat::R32Float);
        let entries: Vec<_> = (0..5)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: if binding == 1 {
                    wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    }
                } else {
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage {
                            read_only: binding == 4,
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    }
                },
                count: None,
            })
            .collect();
        let extra = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Poisson FFT buffers"),
            entries: &entries,
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Poisson spectral"),
            bind_group_layouts: &[Some(&common), Some(&extra)],
            immediate_size: 0,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Poisson spectral"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/poisson_solver/spectral.wgsl").into(),
            ),
        });
        let pipelines: Vec<_> = [
            "initialize",
            "fft",
            "split",
            "divide",
            "write_height",
            "diagnose",
            "reduce",
            "accumulate",
            "transpose",
            "certificate",
        ]
        .into_iter()
        .map(|entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions {
                    // Every consumed shared element is written before the first
                    // barrier. Avoid redundant large-array initialization.
                    zero_initialize_workgroup_memory: false,
                    ..Default::default()
                },
                cache: None,
            })
        })
        .collect();
        Self {
            common,
            extra,
            pipelines,
        }
    }
}
pub(super) struct Solver {
    normal: wgpu::Texture,
    output: wgpu::Texture,
    convention: NormalConvention,
    workspace: Arc<Workspace>,
}
impl Solver {
    pub fn matches(&self, n: &wgpu::Texture, o: &wgpu::Texture, c: NormalConvention) -> bool {
        self.normal == *n && self.output == *o && self.convention == c
    }
    pub fn new(
        solver: &PoissonSolver,
        normal: &wgpu::Texture,
        output: &wgpu::Texture,
        convention: NormalConvention,
    ) -> Result<Self, GpuError> {
        let device = &solver.device;
        let extent = crate::executor::texture_extent(output)?;
        let mut arena = Arena {
            device,
            reservations: Reservations {
                pool: solver.pool.clone(),
                ids: Vec::new(),
            },
        };
        let dummy = arena.texture(Extent::new(1, 1)?)?;
        let mut buffer = |size, usage| -> Result<wgpu::Buffer, GpuError> {
            arena.reserve(size)?;
            Ok(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Poisson spectral scratch"),
                size,
                usage,
                mapped_at_creation: false,
            }))
        };
        let n = u64::from(extent.width) * u64::from(extent.height);
        let data = buffer(n * 8, wgpu::BufferUsages::STORAGE)?;
        let groups = extent.width.div_ceil(8) * extent.height.div_ceil(8);
        let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
        let a = buffer(u64::from(groups) * 16, usage)?;
        let b = buffer(u64::from(groups.div_ceil(64)) * 16, usage)?;
        let readback = buffer(
            32,
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        )?;
        // One canonical table avoids repeated transcendental evaluation in every
        // butterfly and gives identical twiddles to both transform directions.
        let twiddles: Vec<[f32; 2]> = (0..2048)
            .map(|k| {
                let angle = -std::f64::consts::TAU * f64::from(k) / 4096.0;
                [angle.cos() as f32, angle.sin() as f32]
            })
            .collect();
        arena.reserve(2048 * 8)?;
        let twiddles = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Poisson FFT twiddles"),
            contents: bytemuck::cast_slice(&twiddles),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let Kernel {
            common,
            extra,
            pipelines,
        } = solver.spectral_kernel.get_or_init(|| Kernel::new(device));
        let mut prepare = |entry: usize,
                           flags: [u32; 4],
                           dispatch: [u32; 2],
                           count: u32,
                           input: &wgpu::Buffer,
                           out: &wgpu::Buffer|
         -> Result<Pass, GpuError> {
            arena.reserve(64)?;
            let mut node = NodeUniform::new(extent, extent);
            node.flags = flags;
            node.reserved = [
                u32::from(convention == NormalConvention::DirectX),
                count,
                0,
                0,
            ];
            let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Poisson FFT parameters"),
                contents: bytemuck::bytes_of(&node),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            // A sampled output must never alias a storage output in a dispatch.
            let writing = entry == 4;
            let destination =
                if writing { output } else { &dummy }.create_view(&Default::default());
            let height = if writing { &dummy } else { output }.create_view(&Default::default());
            let source = normal.create_view(&Default::default());
            let common = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: common,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&source),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&destination),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: uniform.as_entire_binding(),
                    },
                ],
            });
            let extra = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: extra,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: data.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&height),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: input.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: out.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: twiddles.as_entire_binding(),
                    },
                ],
            });
            Ok(Pass {
                pipeline: pipelines[entry].clone(),
                common,
                extra,
                groups: dispatch,
            })
        };
        let pixels = [extent.width.div_ceil(8), extent.height.div_ceil(8)];
        let first = vec![prepare(0, [0, 0, 0, 0], pixels, 0, &a, &b)?];
        let refine = vec![prepare(0, [0, 0, 0, 1], pixels, 0, &a, &b)?];
        let mut transform = Vec::new();
        let mut forward = Vec::new();
        let square = extent.width == extent.height;
        for (axis, inverse) in [(0, 0), (1, 0), (2, 0), (1, 1), (0, 1)] {
            if axis == 2 {
                forward = transform.clone();
                transform.push(prepare(3, [0; 4], pixels, 0, &a, &b)?);
                continue;
            }
            let length = if axis == 0 {
                extent.width
            } else {
                extent.height
            };
            let lines = if axis == 0 {
                extent.height
            } else {
                extent.width
            };
            if square && ((axis == 1 && inverse == 0) || (axis == 0 && inverse == 1)) {
                transform.push(prepare(8, [0; 4], pixels, 0, &a, &b)?);
            }
            let flags = [if square { 0 } else { axis }, inverse, length, 0];
            if length > 2048 && inverse == 0 {
                transform.push(prepare(2, flags, [length.div_ceil(128), lines], 0, &a, &b)?);
            }
            transform.push(prepare(
                1,
                flags,
                [lines, length.div_ceil(2048)],
                0,
                &a,
                &b,
            )?);
            if length > 2048 && inverse != 0 {
                transform.push(prepare(2, flags, [length.div_ceil(128), lines], 0, &a, &b)?);
            }
        }
        let write = prepare(4, [0; 4], pixels, 0, &a, &b)?;
        let accumulate = prepare(7, [0; 4], pixels, 0, &a, &b)?;
        let mut diagnostics = vec![prepare(5, [0; 4], pixels, 0, &a, &b)?];
        let mut count = groups;
        let mut input = &a;
        let mut out = &b;
        while count > 1 {
            diagnostics.push(prepare(
                6,
                [0; 4],
                [count.div_ceil(64), 1],
                count,
                input,
                out,
            )?);
            count = count.div_ceil(64);
            std::mem::swap(&mut input, &mut out);
        }
        let mut certificate = refine.clone();
        certificate.extend(forward);
        certificate.push(prepare(9, [0; 4], pixels, 0, &a, &b)?);
        certificate.extend(diagnostics.iter().skip(1).cloned());
        Ok(Self {
            normal: normal.clone(),
            output: output.clone(),
            convention,
            workspace: Arc::new(Workspace {
                first,
                refine,
                transform,
                accumulate,
                write,
                diagnostics,
                certificate,
                result: input.clone(),
                readback,
                _reservations: arena.reservations,
            }),
        })
    }
    pub fn solve(
        &self,
        solver: &PoissonSolver,
        p: &PoissonParameters,
        current: &mut dyn FnMut() -> bool,
        submit: &mut dyn FnMut(wgpu::CommandEncoder) -> Result<(), GpuError>,
    ) -> Result<PoissonSolveReport, GpuError> {
        let mut report = PoissonSolveReport::empty();
        let w = &self.workspace;
        for cycle in 0..p.v_cycles {
            let mut encoder = solver.device.create_command_encoder(&Default::default());
            let setup = if cycle == 0 { &w.first } else { &w.refine };
            for pass in setup
                .iter()
                .chain(&w.transform)
                .chain((cycle != 0).then_some(&w.accumulate))
                .chain(std::iter::once(&w.write))
                .chain(&w.diagnostics)
            {
                if !current() {
                    return Ok(report);
                }
                pass.encode(&mut encoder);
            }
            encoder.copy_buffer_to_buffer(&w.result, 0, &w.readback, 0, 16);
            // Certify the actual R32 output, not the pre-publication FFT value.
            for pass in &w.certificate {
                if !current() {
                    return Ok(report);
                }
                pass.encode(&mut encoder);
            }
            encoder.copy_buffer_to_buffer(&w.result, 0, &w.readback, 16, 16);
            let retained = w.clone();
            let result = submit(encoder);
            solver.queue.on_submitted_work_done(move || drop(retained));
            result?;
            let (tx, rx) = std::sync::mpsc::sync_channel(1);
            w.readback
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |r| {
                    let _ = tx.send(r);
                });
            solver
                .device
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: Some(Duration::from_secs(30)),
                })
                .map_err(|e| GpuError(e.to_string()))?;
            rx.recv_timeout(Duration::from_secs(30))
                .map_err(|e| GpuError(e.to_string()))?
                .map_err(|e| GpuError(e.to_string()))?;
            let mapped = w
                .readback
                .slice(..)
                .get_mapped_range()
                .map_err(|e| GpuError(e.to_string()))?;
            let values: Vec<f32> = mapped
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect();
            drop(mapped);
            w.readback.unmap();
            if values.iter().any(|v| !v.is_finite() || *v < 0.0) {
                return Err(GpuError("nonfinite spectral Poisson diagnostic".into()));
            }
            report.rhs_l2 = values[0].sqrt();
            report.residual_l2 = values[1].sqrt();
            report.clamped_pixels = values[2] as u32;
            report.gradient_l2 = Some(values[3].sqrt());
            report.optimality_gap_l2 = Some(values[4].sqrt());
            report.cycles = cycle + 1;
            report.scalar_readbacks += 1;
            report.residual_history.push(report.residual_l2);
            if !current() {
                return Ok(report);
            }
            if converged(report.residual_l2, report.rhs_l2, p)
                || values[4].sqrt() <= p.relative_tolerance * values[3].sqrt()
            {
                report.stop = PoissonStop::Converged;
                return Ok(report);
            }
        }
        report.stop = PoissonStop::MaxCycles;
        Ok(report)
    }
}
