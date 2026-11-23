//! Prepared, non-submitting Poisson primitives. Scalar scratch is R32Float:
//! UV-scaled second differences and mean projection require full precision.
//!
//! Multi-input extension pattern (ADR 0004): group 0 retains the common ABI,
//! with input 0 at binding 0, distinct R32Float output at 1, and the 64-byte
//! `NodeUniform` at 2. Its dimensions describe output/input extents; parameters
//! and flags carry the operator controls. Group 1 adds inputs 1..3 at bindings
//! 0..2, a 16-byte grid uniform at 3, and read-only Galerkin stencils at 4.
//! Grid dimensions are separate because transfers/reductions can have different
//! operand extents. Both point and reduction kernels use 8x8 workgroups.
mod grid;
pub use grid::PoissonGrid;

use crate::{
    Axis, GpuError, NodeUniform,
    executor::{common_layout, texture_extent},
    pool::{AllocationDesc, AllocationId, ResourcePool},
};
use std::{
    num::NonZeroU64,
    sync::{Arc, Mutex},
};
use texelith_core::{Boundary, Extent, NormalConvention};
use wgpu::util::DeviceExt;

#[repr(C, align(16))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GridUniform {
    size: [u32; 4],
}

const _: () = {
    assert!(std::mem::size_of::<GridUniform>() == 16);
    assert!(std::mem::align_of::<GridUniform>() == 16);
};

#[derive(Clone, Copy, Debug)]
pub enum PoissonPass {
    NormalGradient {
        axis: Axis,
        convention: NormalConvention,
    },
    ClampCount,
    Difference(Axis),
    Transpose,
    Laplacian,
    Mean,
    ProjectMean,
    L2Norm,
    Jacobi {
        omega: f32,
    },
    Restrict,
    Prolong,
    Residual,
}

pub struct PoissonOperators {
    device: Arc<wgpu::Device>,
    pool: Arc<Mutex<ResourcePool>>,
    layout: wgpu::BindGroupLayout,
    extra_layout: wgpu::BindGroupLayout,
    point: wgpu::ComputePipeline,
    reduce: wgpu::ComputePipeline,
}

struct Reservations {
    pool: Arc<Mutex<ResourcePool>>,
    ids: Vec<AllocationId>,
}
impl Drop for Reservations {
    fn drop(&mut self) {
        let mut pool = self.pool.lock().unwrap();
        for id in &self.ids {
            let _ = pool.release_transient(*id);
        }
    }
}
struct Step {
    pipeline: wgpu::ComputePipeline,
    binding: wgpu::BindGroup,
    extra_binding: wgpu::BindGroup,
    groups: [u32; 2],
}
struct PreparedInner {
    steps: Vec<Step>,
    _reservations: Reservations,
}

/// Owns bindings and budget reservations. Reuse encoding without allocations.
/// Keep this alive until completion, or call `retain_until_completed` immediately
/// after submission to transfer a retained clone to the queue completion callback.
#[derive(Clone)]
pub struct PreparedPoisson(Arc<PreparedInner>);
impl PreparedPoisson {
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        for step in &self.0.steps {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Poisson operator"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&step.pipeline);
            pass.set_bind_group(0, &step.binding, &[]);
            pass.set_bind_group(1, &step.extra_binding, &[]);
            pass.dispatch_workgroups(step.groups[0], step.groups[1], 1);
        }
    }

    pub fn retain_until_completed(&self, queue: &wgpu::Queue) {
        let retained = self.clone();
        queue.on_submitted_work_done(move || drop(retained));
    }

    pub fn dispatch_count(&self) -> usize {
        self.0.steps.len()
    }
}

impl PoissonOperators {
    pub fn new(
        device: Arc<wgpu::Device>,
        pool: Arc<Mutex<ResourcePool>>,
    ) -> Result<Self, GpuError> {
        let layout = common_layout(&device, wgpu::TextureFormat::R32Float);
        let mut entries: Vec<_> = (0..3)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            })
            .collect();
        for (binding, ty) in [
            (3, wgpu::BufferBindingType::Uniform),
            (4, wgpu::BufferBindingType::Storage { read_only: true }),
        ] {
            entries.push(wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty,
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(16),
                },
                count: None,
            });
        }
        let extra_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Poisson additional inputs and grid"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Poisson"),
            bind_group_layouts: &[Some(&layout), Some(&extra_layout)],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Poisson"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/poisson_operators/operators.wgsl").into(),
            ),
        });
        let pipeline = |entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let point = pipeline("point");
        let reduce = pipeline("reduce");
        Ok(Self {
            device,
            pool,
            layout,
            extra_layout,
            point,
            reduce,
        })
    }

    pub fn prepare(
        &self,
        grid: &PoissonGrid,
        pass: PoissonPass,
        inputs: &[&wgpu::Texture],
        output: &wgpu::Texture,
    ) -> Result<PreparedPoisson, GpuError> {
        let e = grid.extent();
        let half = Extent::new(e.width.div_ceil(2), e.height.div_ceil(2))?;
        let scalar = Extent::new(1, 1)?;
        let reduction = matches!(
            pass,
            PoissonPass::Mean | PoissonPass::L2Norm | PoissonPass::ClampCount
        );
        let normal = matches!(
            pass,
            PoissonPass::NormalGradient { .. } | PoissonPass::ClampCount
        );
        let arity = if matches!(
            pass,
            PoissonPass::Transpose
                | PoissonPass::ProjectMean
                | PoissonPass::Jacobi { .. }
                | PoissonPass::Residual
        ) {
            2
        } else {
            1
        };
        if inputs.len() != arity {
            return Err(GpuError("invalid Poisson operand count".into()));
        }
        if let PoissonPass::Jacobi { omega } = pass
            && (!omega.is_finite() || omega <= 0.0 || omega > 1.0)
        {
            return Err(GpuError("invalid Jacobi omega".into()));
        }
        let target = if reduction {
            scalar
        } else if matches!(pass, PoissonPass::Restrict) {
            half
        } else {
            e
        };
        if texture_extent(output)? != target
            || output.format() != wgpu::TextureFormat::R32Float
            || !output
                .usage()
                .contains(wgpu::TextureUsages::STORAGE_BINDING)
        {
            return Err(GpuError("invalid Poisson output".into()));
        }
        for (i, input) in inputs.iter().enumerate() {
            let expected = if matches!(pass, PoissonPass::Prolong) {
                half
            } else if matches!(pass, PoissonPass::ProjectMean) && i == 1 {
                scalar
            } else {
                e
            };
            let format = if normal {
                wgpu::TextureFormat::Rgba16Float
            } else {
                wgpu::TextureFormat::R32Float
            };
            if *input == output
                || texture_extent(input)? != expected
                || input.format() != format
                || !input.usage().contains(wgpu::TextureUsages::TEXTURE_BINDING)
            {
                return Err(GpuError("invalid or aliased Poisson input".into()));
            }
        }
        let mut builder = Builder {
            owner: self,
            grid,
            steps: Vec::new(),
            reservations: Reservations {
                pool: self.pool.clone(),
                ids: Vec::new(),
            },
        };
        let packed = if grid.depth > 0 {
            grid.packed()
        } else {
            vec![[0.0; 4]]
        };
        let stencil = builder.buffer(bytemuck::cast_slice(&packed), wgpu::BufferUsages::STORAGE)?;
        let (op, axis, dx, omega) = match pass {
            PoissonPass::NormalGradient { axis, convention } => (
                0,
                u32::from(axis == Axis::Y),
                u32::from(convention == NormalConvention::DirectX),
                0.0,
            ),
            PoissonPass::Difference(axis) => (1, u32::from(axis == Axis::Y), 0, 0.0),
            PoissonPass::Transpose => (2, 0, 0, 0.0),
            PoissonPass::Laplacian => (3, 0, 0, 0.0),
            PoissonPass::ProjectMean => (4, 0, 0, 0.0),
            PoissonPass::Jacobi { omega } => (5, 0, 0, omega),
            PoissonPass::Restrict => (6, 0, 0, 0.0),
            PoissonPass::Prolong => (7, 0, 0, 0.0),
            PoissonPass::Residual => (8, 0, 0, 0.0),
            PoissonPass::Mean => (9, 0, 0, 0.0),
            PoissonPass::L2Norm => (10, 0, 0, 0.0),
            PoissonPass::ClampCount => (11, 0, 0, 0.0),
        };
        if reduction {
            let mut input = inputs[0].clone();
            let mut first = true;
            loop {
                let w = input.width().div_ceil(8);
                let h = input.height().div_ceil(8);
                let last = w == 1 && h == 1;
                let target = if last {
                    output.clone()
                } else {
                    builder.texture(Extent::new(w, h)?)?
                };
                builder.step(
                    [&input; 4],
                    &target,
                    &stencil,
                    [if first { op } else { 12 }, 0, 0, op],
                    [
                        0.0,
                        if last { 1.0 } else { 0.0 },
                        (e.width * e.height) as f32,
                        0.0,
                    ],
                    true,
                )?;
                if last {
                    break;
                }
                input = target;
                first = false;
            }
        } else if grid.depth > 0
            && matches!(
                pass,
                PoissonPass::Laplacian | PoissonPass::Jacobi { .. } | PoissonPass::Residual
            )
        {
            let ax = builder.texture(e)?;
            let mx = builder.texture(e)?;
            builder.step(
                [inputs[0]; 4],
                &ax,
                &stencil,
                [13, 0, 0, 0],
                [0.0; 4],
                false,
            )?;
            builder.step(
                [inputs[0]; 4],
                &mx,
                &stencil,
                [14, 0, 0, 0],
                [0.0; 4],
                false,
            )?;
            builder.step(
                [
                    &ax,
                    &mx,
                    inputs[0],
                    inputs.get(1).copied().unwrap_or(inputs[0]),
                ],
                output,
                &stencil,
                [15, 0, 0, op],
                [omega, 0.0, 0.0, 0.0],
                false,
            )?;
        } else {
            builder.step(
                [
                    inputs[0],
                    inputs.get(1).copied().unwrap_or(inputs[0]),
                    inputs[0],
                    inputs[0],
                ],
                output,
                &stencil,
                [op, axis, dx, 0],
                [omega, 0.0, 0.0, 0.0],
                false,
            )?;
        }
        Ok(PreparedPoisson(Arc::new(PreparedInner {
            steps: builder.steps,
            _reservations: builder.reservations,
        })))
    }
}

struct Builder<'a> {
    owner: &'a PoissonOperators,
    grid: &'a PoissonGrid,
    steps: Vec<Step>,
    reservations: Reservations,
}
impl Builder<'_> {
    fn reserve(&mut self, desc: AllocationDesc) -> Result<(), GpuError> {
        let id = self
            .owner
            .pool
            .lock()
            .unwrap()
            .allocate(desc)
            .map_err(|e| GpuError(e.to_string()))?;
        self.reservations.ids.push(id);
        Ok(())
    }
    fn buffer(
        &mut self,
        bytes: &[u8],
        usage: wgpu::BufferUsages,
    ) -> Result<wgpu::Buffer, GpuError> {
        self.reserve(AllocationDesc::Buffer {
            bytes: bytes.len() as u64,
        })?;
        Ok(self
            .owner
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Poisson parameters"),
                contents: bytes,
                usage,
            }))
    }
    fn texture(&mut self, extent: Extent) -> Result<wgpu::Texture, GpuError> {
        let usage = wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING;
        self.reserve(AllocationDesc::Texture {
            extent,
            format: wgpu::TextureFormat::R32Float,
            usage,
        })?;
        Ok(self.owner.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Poisson scratch"),
            size: wgpu::Extent3d {
                width: extent.width,
                height: extent.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage,
            view_formats: &[],
        }))
    }
    fn step(
        &mut self,
        inputs: [&wgpu::Texture; 4],
        output: &wgpu::Texture,
        stencil: &wgpu::Buffer,
        flags: [u32; 4],
        parameters: [f32; 4],
        reduction: bool,
    ) -> Result<(), GpuError> {
        let grid = GridUniform {
            size: [
                self.grid.extent.width,
                self.grid.extent.height,
                u32::from(self.grid.boundary == Boundary::Repeat),
                0,
            ],
        };
        let grid_uniform = self.buffer(bytemuck::bytes_of(&grid), wgpu::BufferUsages::UNIFORM)?;
        let mut node = NodeUniform::new(texture_extent(output)?, texture_extent(inputs[0])?);
        node.flags = flags;
        node.parameters = parameters;
        let uniform = self.buffer(bytemuck::bytes_of(&node), wgpu::BufferUsages::UNIFORM)?;
        let views = inputs.map(|t| t.create_view(&Default::default()));
        let out = output.create_view(&Default::default());
        let binding = self
            .owner
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Poisson common ABI"),
                layout: &self.owner.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&views[0]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&out),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: uniform.as_entire_binding(),
                    },
                ],
            });
        let mut entries: Vec<_> = views[1..]
            .iter()
            .enumerate()
            .map(|(i, v)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: wgpu::BindingResource::TextureView(v),
            })
            .collect();
        entries.push(wgpu::BindGroupEntry {
            binding: 3,
            resource: grid_uniform.as_entire_binding(),
        });
        entries.push(wgpu::BindGroupEntry {
            binding: 4,
            resource: stencil.as_entire_binding(),
        });
        let extra_binding = self
            .owner
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Poisson pass"),
                layout: &self.owner.extra_layout,
                entries: &entries,
            });
        let groups = if reduction {
            [output.width(), output.height()]
        } else {
            [output.width().div_ceil(8), output.height().div_ceil(8)]
        };
        self.steps.push(Step {
            pipeline: if reduction {
                self.owner.reduce.clone()
            } else {
                self.owner.point.clone()
            },
            binding,
            extra_binding,
            groups,
        });
        Ok(())
    }
}
