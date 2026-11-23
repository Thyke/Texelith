//! Signed normal divergence and independently scaled convex/concave planes.
use crate::{
    GpuError, GpuExecutor, NodeUniform, Pass, PreparedPass,
    executor::{common_layout, texture_extent},
    pool::{AllocationDesc, AllocationId, FenceId, ResourcePool},
    workgroups,
};
use std::sync::{Arc, Mutex};
use texelith_core::{Boundary, MaterialNodeParams, NormalConvention, OutputPort};

pub(crate) const SCRATCH_BYTES: u64 = 128;

/// Controls and input convention for one named curvature output.
#[derive(Clone, Copy, Debug)]
pub struct CurvatureParameters {
    pub edge_scale: f32,
    pub cavity_scale: f32,
    pub boundary: Boundary,
    pub convention: NormalConvention,
    pub output: OutputPort,
}
impl CurvatureParameters {
    pub fn from_material(
        parameters: &MaterialNodeParams,
        convention: NormalConvention,
        output: OutputPort,
    ) -> Result<Self, GpuError> {
        parameters.validate()?;
        let MaterialNodeParams::Curvature {
            edge_scale,
            cavity_scale,
            boundary,
        } = *parameters
        else {
            return Err(GpuError("expected curvature parameters".into()));
        };
        if !matches!(
            output,
            OutputPort::Curvature | OutputPort::Edge | OutputPort::Cavity
        ) {
            return Err(GpuError("expected a curvature output port".into()));
        }
        Ok(Self {
            edge_scale,
            cavity_scale,
            boundary,
            convention,
            output,
        })
    }
}

pub(crate) fn pipeline(device: &wgpu::Device) -> wgpu::ComputePipeline {
    let common = common_layout(device, wgpu::TextureFormat::R32Float);
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("curvature"),
        bind_group_layouts: &[Some(&common)],
        immediate_size: 0,
    });
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("curvature"),
        source: wgpu::ShaderSource::Wgsl(
            include_str!("../../shaders/curvature/curvature.wgsl").into(),
        ),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("curvature"),
        layout: Some(&layout),
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

/// Reusable single-plane pass. Texture payloads are owned by the caller; the
/// uniform and its upload staging are admitted together and retained at fences.
pub struct PreparedCurvature {
    passes: [PreparedPass; 1],
    pool: Arc<Mutex<ResourcePool>>,
    allocations: Vec<AllocationId>,
}
impl PreparedCurvature {
    pub fn new(
        executor: &GpuExecutor,
        normal: &wgpu::Texture,
        output: &wgpu::Texture,
        parameters: &CurvatureParameters,
        pool: Arc<Mutex<ResourcePool>>,
    ) -> Result<Self, GpuError> {
        Self::prepare(executor, normal, output, parameters, pool, None)
    }

    pub(crate) fn prepare(
        executor: &GpuExecutor,
        normal: &wgpu::Texture,
        output: &wgpu::Texture,
        parameters: &CurvatureParameters,
        pool: Arc<Mutex<ResourcePool>>,
        cached_pipeline: Option<&wgpu::ComputePipeline>,
    ) -> Result<Self, GpuError> {
        CurvatureParameters::from_material(
            &MaterialNodeParams::Curvature {
                edge_scale: parameters.edge_scale,
                cavity_scale: parameters.cavity_scale,
                boundary: parameters.boundary,
            },
            parameters.convention,
            parameters.output,
        )?;
        let extent = texture_extent(output)?;
        let source = texture_extent(normal)?;
        let valid = |t: &wgpu::Texture| {
            t.sample_count() == 1
                && t.dimension() == wgpu::TextureDimension::D2
                && t.depth_or_array_layers() == 1
                && t.mip_level_count() == 1
        };
        if normal == output
            || source != extent
            || !valid(normal)
            || !valid(output)
            || normal.format() != wgpu::TextureFormat::Rgba16Float
            || output.format() != wgpu::TextureFormat::R32Float
            || !normal
                .usage()
                .contains(wgpu::TextureUsages::TEXTURE_BINDING)
            || !output
                .usage()
                .contains(wgpu::TextureUsages::STORAGE_BINDING)
        {
            return Err(GpuError("invalid curvature textures or extents".into()));
        }
        let pipeline = cached_pipeline
            .cloned()
            .unwrap_or_else(|| pipeline(executor.device()));
        let resources = pool
            .lock()
            .unwrap()
            .allocate_backed(
                executor.device(),
                &[
                    AllocationDesc::Buffer { bytes: 64 },
                    AllocationDesc::Staging { bytes: 64 },
                ],
            )
            .map_err(|e| GpuError(e.to_string()))?;
        let mut uniform = NodeUniform::new(extent, source);
        uniform.parameters = [parameters.edge_scale, parameters.cavity_scale, 0.0, 0.0];
        uniform.flags = [
            u32::from(parameters.boundary == Boundary::Repeat),
            u32::from(parameters.convention == NormalConvention::DirectX),
            match parameters.output {
                OutputPort::Curvature => 0,
                OutputPort::Edge => 1,
                OutputPort::Cavity => 2,
                _ => unreachable!("validated port"),
            },
            0,
        ];
        let buffer = resources[0].1.buffer();
        executor
            .queue
            .write_buffer(buffer, 0, bytemuck::bytes_of(&uniform));
        let input_view = normal.create_view(&Default::default());
        let output_view = output.create_view(&Default::default());
        let common = executor
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("curvature common"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&input_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&output_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: buffer.as_entire_binding(),
                    },
                ],
            });
        Ok(Self {
            passes: [PreparedPass {
                pipeline,
                common,
                buffer: buffer.clone(),
                weights: None,
                queue: executor.queue.clone(),
                operation: Pass::Downsample,
                evaluation: extent,
                source,
                groups: workgroups(extent)?,
                additional: None,
            }],
            pool,
            allocations: resources.iter().map(|(id, _)| *id).collect(),
        })
    }
    pub fn passes(&self) -> &[PreparedPass] {
        &self.passes
    }
    pub fn mark_submitted(&mut self, fence: FenceId) -> Result<(), GpuError> {
        let mut pool = self.pool.lock().unwrap();
        for id in &self.allocations {
            pool.mark_submitted(*id, fence)
                .map_err(|e| GpuError(e.to_string()))?;
        }
        Ok(())
    }
    pub fn complete(&mut self, fence: FenceId) {
        self.pool.lock().unwrap().complete(fence);
    }
}
impl Drop for PreparedCurvature {
    fn drop(&mut self) {
        let mut pool = self.pool.lock().unwrap();
        for id in &self.allocations {
            let _ = pool.release_transient(*id);
        }
    }
}
