//! Linear luminance, curve remapping, numeric masks and final smoothness inversion.
use crate::{
    GpuError, GpuExecutor, NodeUniform, Pass, PreparedPass,
    executor::{common_layout, texture_extent},
    pool::{AllocationDesc, AllocationId, FenceId, ResourcePool},
    workgroups,
};
use std::sync::{Arc, Mutex};
use texelith_core::MaterialNodeParams;

// Each knot occupies a vec4 in a uniform array (16-byte WGSL array stride).
pub(crate) const SCRATCH_BYTES: u64 = 2 * (64 + 256);

pub(crate) fn pipeline(device: &wgpu::Device) -> wgpu::ComputePipeline {
    let common = common_layout(device, wgpu::TextureFormat::R32Float);
    let extra = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("roughness mask and curve ABI"),
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
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(256),
                },
                count: None,
            },
        ],
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("roughness"),
        bind_group_layouts: &[Some(&common), Some(&extra)],
        immediate_size: 0,
    });
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("roughness"),
        source: wgpu::ShaderSource::Wgsl(
            include_str!("../../shaders/roughness/roughness.wgsl").into(),
        ),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("roughness"),
        layout: Some(&layout),
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

/// Reusable single-pass node. Caller owns texture payloads; this node admits its
/// uniforms and upload staging atomically and retains them through real fences.
pub struct PreparedRoughness {
    passes: [PreparedPass; 1],
    pool: Arc<Mutex<ResourcePool>>,
    allocations: Vec<AllocationId>,
}
impl PreparedRoughness {
    pub fn new(
        executor: &GpuExecutor,
        color: &wgpu::Texture,
        mask: Option<&wgpu::Texture>,
        output: &wgpu::Texture,
        parameters: &MaterialNodeParams,
        pool: Arc<Mutex<ResourcePool>>,
    ) -> Result<Self, GpuError> {
        Self::prepare(executor, color, mask, output, parameters, pool, None)
    }

    pub(crate) fn prepare(
        executor: &GpuExecutor,
        color: &wgpu::Texture,
        mask: Option<&wgpu::Texture>,
        output: &wgpu::Texture,
        parameters: &MaterialNodeParams,
        pool: Arc<Mutex<ResourcePool>>,
        cached_pipeline: Option<&wgpu::ComputePipeline>,
    ) -> Result<Self, GpuError> {
        parameters.validate()?;
        let MaterialNodeParams::Roughness {
            curve,
            base,
            invert,
        } = parameters
        else {
            return Err(GpuError("expected roughness parameters".into()));
        };
        let extent = texture_extent(output)?;
        let source = texture_extent(color)?;
        let readable = |t: &wgpu::Texture| {
            t != output
                && t.usage().contains(wgpu::TextureUsages::TEXTURE_BINDING)
                && t.sample_count() == 1
                && t.dimension() == wgpu::TextureDimension::D2
                && t.depth_or_array_layers() == 1
                && t.mip_level_count() == 1
        };
        if !readable(color)
            || color.format() != wgpu::TextureFormat::Rgba16Float
            || output.format() != wgpu::TextureFormat::R32Float
            || !output
                .usage()
                .contains(wgpu::TextureUsages::STORAGE_BINDING)
            || output.sample_count() != 1
            || output.dimension() != wgpu::TextureDimension::D2
            || source.width < extent.width
            || source.height < extent.height
            || mask.is_some_and(|t| {
                !readable(t)
                    || t.format() != wgpu::TextureFormat::R32Float
                    || t.width() != extent.width
                    || t.height() != extent.height
            })
        {
            return Err(GpuError("invalid roughness textures or extents".into()));
        }
        let resources = pool
            .lock()
            .unwrap()
            .allocate_backed(
                executor.device(),
                &[
                    AllocationDesc::Buffer { bytes: 64 },
                    AllocationDesc::Buffer { bytes: 256 },
                    AllocationDesc::Staging { bytes: 320 },
                ],
            )
            .map_err(|e| GpuError(e.to_string()))?;
        let pipeline = cached_pipeline
            .cloned()
            .unwrap_or_else(|| pipeline(executor.device()));
        let mut uniform = NodeUniform::new(extent, source);
        uniform.parameters[0] = *base;
        uniform.flags = [
            curve.len() as u32,
            u32::from(mask.is_some()),
            u32::from(*invert),
            0,
        ];
        let mut knots = [[0.0f32; 4]; 16];
        for (target, [x, y]) in knots.iter_mut().zip(curve) {
            *target = [*x, *y, 0.0, 0.0];
        }
        let buffer = resources[0].1.buffer();
        let curve_buffer = resources[1].1.buffer();
        executor
            .queue
            .write_buffer(buffer, 0, bytemuck::bytes_of(&uniform));
        executor
            .queue
            .write_buffer(curve_buffer, 0, bytemuck::cast_slice(&knots));
        let cv = color.create_view(&Default::default());
        let ov = output.create_view(&Default::default());
        let mv = mask.unwrap_or(color).create_view(&Default::default());
        let common = executor
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("roughness common"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&cv),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&ov),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: buffer.as_entire_binding(),
                    },
                ],
            });
        let extra = executor
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("roughness mask and knots"),
                layout: &pipeline.get_bind_group_layout(1),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&mv),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: curve_buffer.as_entire_binding(),
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
                additional: Some(extra),
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
impl Drop for PreparedRoughness {
    fn drop(&mut self) {
        let mut pool = self.pool.lock().unwrap();
        for id in &self.allocations {
            let _ = pool.release_transient(*id);
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn curve_uniform_abi_and_scratch_include_uploads() {
        assert_eq!(std::mem::size_of::<[[f32; 4]; 16]>(), 256);
        assert_eq!(super::SCRATCH_BYTES, 64 + 256 + 320);
    }
}
