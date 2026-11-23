//! Pointwise linear-RGB metallic selection with optional numeric mask composition.
use crate::{
    GpuError, NodeUniform, Pass, PreparedPass,
    executor::{common_layout, texture_extent},
    workgroups,
};
use std::sync::Arc;
use texelith_core::{MaskMode, MaterialNodeParams};

// One NodeUniform and its queue-write staging, with no texture scratch.
pub(crate) const SCRATCH_BYTES: u64 = 128;

/// A lazily constructed, reusable metallic pipeline. Textures and the 64-byte
/// uniform are caller-owned and must stay alive through queue completion.
pub struct MetallicKernel {
    pipeline: wgpu::ComputePipeline,
    queue: Arc<wgpu::Queue>,
}

impl MetallicKernel {
    pub fn new(device: &wgpu::Device, queue: Arc<wgpu::Queue>) -> Self {
        let common = common_layout(device, wgpu::TextureFormat::R32Float);
        let extra = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("metallic mask ABI"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("metallic"),
            bind_group_layouts: &[Some(&common), Some(&extra)],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("metallic"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/metallic/metallic.wgsl").into(),
            ),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("metallic"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self { pipeline, queue }
    }

    /// Prepare equal-sized linear RGBA16 color, optional R32 mask and distinct
    /// R32 output. Reuse the prepared encoding without allocations. Uniform
    /// writes precede the next submission; use separate uniforms in a batch.
    /// An absent mask is constant one for every composition mode.
    pub fn prepare(
        &self,
        device: &wgpu::Device,
        color: &wgpu::Texture,
        output: &wgpu::Texture,
        mask: Option<&wgpu::Texture>,
        buffer: &wgpu::Buffer,
        parameters: &MaterialNodeParams,
    ) -> Result<PreparedPass, GpuError> {
        if texture_extent(color)? != texture_extent(output)? {
            return Err(GpuError("metallic texture extents must match".into()));
        }
        self.prepare_proxy(device, color, output, mask, buffer, parameters)
    }

    // The worker can restrict source color to its explicitly requested proxy.
    pub(crate) fn prepare_proxy(
        &self,
        device: &wgpu::Device,
        color: &wgpu::Texture,
        output: &wgpu::Texture,
        mask: Option<&wgpu::Texture>,
        buffer: &wgpu::Buffer,
        parameters: &MaterialNodeParams,
    ) -> Result<PreparedPass, GpuError> {
        parameters.validate()?;
        let MaterialNodeParams::Metallic {
            picked_color,
            radius,
            threshold,
            feather,
            mask_mode,
        } = parameters
        else {
            return Err(GpuError("expected metallic parameters".into()));
        };
        let extent = texture_extent(output)?;
        let source = texture_extent(color)?;
        let single = |t: &wgpu::Texture| {
            t.dimension() == wgpu::TextureDimension::D2
                && t.depth_or_array_layers() == 1
                && t.mip_level_count() == 1
                && t.sample_count() == 1
        };
        let readable = |t: &wgpu::Texture| {
            single(t) && t != output && t.usage().contains(wgpu::TextureUsages::TEXTURE_BINDING)
        };
        if !readable(color)
            || color.format() != wgpu::TextureFormat::Rgba16Float
            || !single(output)
            || output.format() != wgpu::TextureFormat::R32Float
            || !output
                .usage()
                .contains(wgpu::TextureUsages::STORAGE_BINDING)
            || source.width < extent.width
            || source.height < extent.height
            || mask.is_some_and(|t| {
                !readable(t)
                    || t.format() != wgpu::TextureFormat::R32Float
                    || t.width() != extent.width
                    || t.height() != extent.height
            })
            || buffer.size() < 64
            || !buffer
                .usage()
                .contains(wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST)
        {
            return Err(GpuError("invalid metallic textures or uniform".into()));
        }
        let mut uniform = NodeUniform::new(extent, source);
        uniform.parameters = [picked_color[0], picked_color[1], picked_color[2], *radius];
        uniform.flags = [
            u32::from(mask.is_some()),
            match mask_mode {
                MaskMode::Replace => 0,
                MaskMode::Add => 1,
                MaskMode::Subtract => 2,
            },
            0,
            0,
        ];
        // Preserve the common 64-byte ABI; these two lanes carry f32 bits.
        uniform.reserved = [threshold.to_bits(), feather.to_bits(), 0, 0];
        self.queue
            .write_buffer(buffer, 0, bytemuck::bytes_of(&uniform));
        let cv = color.create_view(&Default::default());
        let ov = output.create_view(&Default::default());
        let mv = mask.unwrap_or(color).create_view(&Default::default());
        let common = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("metallic common"),
            layout: &self.pipeline.get_bind_group_layout(0),
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
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer,
                        offset: 0,
                        size: wgpu::BufferSize::new(64),
                    }),
                },
            ],
        });
        let extra = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("metallic mask"),
            layout: &self.pipeline.get_bind_group_layout(1),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&mv),
            }],
        });
        Ok(PreparedPass {
            pipeline: self.pipeline.clone(),
            common,
            buffer: buffer.clone(),
            weights: None,
            queue: self.queue.clone(),
            operation: Pass::Downsample,
            evaluation: extent,
            source,
            groups: workgroups(extent)?,
            additional: Some(extra),
        })
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn uniform_and_staging_fit_scratch() {
        assert_eq!(std::mem::size_of::<crate::NodeUniform>(), 64);
        assert_eq!(super::SCRATCH_BYTES, 2 * 64);
    }
}
