//! UV-space gradients of Gaussian height scales, blended before normalization.
use crate::pool::{AllocationDesc, AllocationId, FenceId, ResourcePool};
use crate::{GpuError, GpuExecutor, NodeUniform, PreparedPass, PreparedPyramid, workgroups};
use std::sync::{Arc, Mutex};
use texelith_core::{Boundary, Extent, NormalConvention, NormalFilter, SourceDescriptor};

#[derive(Clone, Debug)]
pub struct NormalParameters {
    pub strength: f32,
    pub filter: NormalFilter,
    pub scale_weights: Vec<f32>,
    pub boundary: Boundary,
    pub convention: NormalConvention,
}
impl NormalParameters {
    pub fn is_legacy(&self) -> bool {
        self.filter == NormalFilter::Central
            && self.boundary == Boundary::Clamp
            && self.convention == NormalConvention::OpenGl
            && self.scale_weights == [1.0]
    }
    fn validate(&self) -> Result<(), GpuError> {
        if !self.strength.is_finite()
            || !(0.0..=10.0).contains(&self.strength)
            || self.scale_weights.is_empty()
            || self.scale_weights.len() > 16
            || self.scale_weights.iter().any(|w| !w.is_finite())
        {
            return Err(GpuError("invalid normal parameters".into()));
        }
        Ok(())
    }
}

/// Conservative transient payload, excluding the input and normal output.
pub(crate) fn scratch_bytes(extent: Extent, scales: usize) -> Result<u64, GpuError> {
    let mut bytes = 256;
    if scales > 1 {
        let extents = crate::pyramid_extents(extent)?;
        for pair in extents.windows(2) {
            bytes += u64::from(pair[0].width) * u64::from(pair[0].height) * 4
                + u64::from(pair[1].width) * u64::from(pair[1].height) * 4
                + 256;
        }
    }
    Ok(bytes)
}

struct Reservations {
    pool: Arc<Mutex<ResourcePool>>,
    ids: Vec<AllocationId>,
}
impl Drop for Reservations {
    fn drop(&mut self) {
        let mut pool = self.pool.lock().unwrap();
        for id in &self.ids {
            let _ = pool.release(*id);
        }
    }
}

/// Persistent bindings and budgeted R32 Gaussian levels. Encoding allocates no
/// resources; the caller drives real fences and cancellation between passes.
pub struct PreparedNormal {
    pyramid: Option<PreparedPyramid>,
    pipeline: wgpu::ComputePipeline,
    common: wgpu::BindGroup,
    additional: wgpu::BindGroup,
    uniform: wgpu::Buffer,
    weights: wgpu::Buffer,
    input: wgpu::Texture,
    boundary: Boundary,
    extent: Extent,
    source: Extent,
    groups: [u32; 3],
    reservations: Reservations,
}
impl PreparedNormal {
    pub fn new(
        executor: &GpuExecutor,
        input: &wgpu::Texture,
        output: &wgpu::Texture,
        source: &SourceDescriptor,
        parameters: &NormalParameters,
        pool: Arc<Mutex<ResourcePool>>,
    ) -> Result<Self, GpuError> {
        parameters.validate()?;
        source.extent.validate()?;
        let extent = Extent::new(input.width(), input.height())?;
        if input == output
            || input.format() != wgpu::TextureFormat::R32Float
            || output.format() != wgpu::TextureFormat::Rgba16Float
            || input.size() != output.size()
            || extent.width > source.extent.width
            || extent.height > source.extent.height
            || input.dimension() != wgpu::TextureDimension::D2
            || input.depth_or_array_layers() != 1
            || input.mip_level_count() != 1
            || output.mip_level_count() != 1
            || input.sample_count() != 1
            || output.sample_count() != 1
            || !input.usage().contains(wgpu::TextureUsages::TEXTURE_BINDING)
            || !output
                .usage()
                .contains(wgpu::TextureUsages::STORAGE_BINDING)
        {
            return Err(GpuError("invalid normal textures".into()));
        }
        let pyramid = if parameters.scale_weights.len() > 1 {
            Some(PreparedPyramid::new(
                executor,
                &SourceDescriptor {
                    content_hash: source.content_hash,
                    extent,
                    interpretation: texelith_core::InputInterpretation::LinearHeight,
                },
                input,
                parameters.boundary,
                pool.clone(),
            )?)
        } else {
            None
        };
        let resources = pool
            .lock()
            .unwrap()
            .allocate_backed(
                executor.device(),
                &[
                    AllocationDesc::Buffer { bytes: 64 },
                    AllocationDesc::Buffer { bytes: 64 },
                    AllocationDesc::Staging { bytes: 128 },
                ],
            )
            .map_err(|e| GpuError(e.to_string()))?;
        let reservations = Reservations {
            pool,
            ids: resources.iter().map(|(id, _)| *id).collect(),
        };
        let pipeline = executor
            .normal_pipeline
            .get_or_init(|| pipeline(executor.device()))
            .clone();
        let uniform = resources[0].1.buffer().clone();
        let weights = resources[1].1.buffer().clone();
        let common = common(executor.device(), &pipeline, input, output, &uniform);
        let levels = pyramid
            .as_ref()
            .map_or(std::slice::from_ref(input), |p| p.levels());
        let views: Vec<_> = (1..=12)
            .map(|i| levels[i.min(levels.len() - 1)].create_view(&Default::default()))
            .collect();
        let mut entries: Vec<_> = views
            .iter()
            .enumerate()
            .map(|(i, v)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: wgpu::BindingResource::TextureView(v),
            })
            .collect();
        entries.push(wgpu::BindGroupEntry {
            binding: 12,
            resource: weights.as_entire_binding(),
        });
        let additional = executor
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("normal scales"),
                layout: &pipeline.get_bind_group_layout(1),
                entries: &entries,
            });
        let prepared = Self {
            pyramid,
            pipeline,
            common,
            additional,
            uniform,
            weights,
            input: input.clone(),
            boundary: parameters.boundary,
            extent,
            source: source.extent,
            groups: workgroups(extent)?,
            reservations,
        };
        prepared.write(executor, parameters)?;
        Ok(prepared)
    }
    fn write(&self, executor: &GpuExecutor, p: &NormalParameters) -> Result<(), GpuError> {
        p.validate()?;
        if p.boundary != self.boundary || (p.scale_weights.len() > 1 && self.pyramid.is_none()) {
            return Err(GpuError("normal scale bindings changed".into()));
        }
        let available = self.pyramid.as_ref().map_or(1, |p| p.levels().len());
        // A proxy already represents a low-pass restriction of the source.
        // Map source-space octaves to its available Gaussian octaves, folding
        // subpixel scales into the finest available level. NPOT aspect rounding
        // is at most one pixel; use the longer-axis restriction factor.
        let restriction = (self.source.width as f64 / self.extent.width as f64)
            .max(self.source.height as f64 / self.extent.height as f64);
        let octave = restriction.log2().round() as usize;
        let count = p
            .scale_weights
            .len()
            .saturating_sub(octave)
            .max(1)
            .min(available);
        let mut u = NodeUniform::new(self.extent, self.source);
        u.parameters[0] = p.strength;
        u.flags = [
            match p.filter {
                NormalFilter::Central => 0,
                NormalFilter::Sobel => 1,
                NormalFilter::Scharr => 2,
            },
            u32::from(p.boundary == Boundary::Repeat),
            u32::from(p.convention == NormalConvention::DirectX),
            count as u32,
        ];
        // Accumulate in f64 so finite signed controls cannot overflow while
        // folding proxy octaves. Reject unrepresentable slots before either
        // uniform is uploaded; Inf * a flat gradient would produce NaN.
        let mut folded = [0.0f64; 16];
        for (i, weight) in p.scale_weights.iter().enumerate() {
            let level = i.saturating_sub(octave);
            if level < available {
                folded[level] += f64::from(*weight);
            }
        }
        if folded.iter().any(|w| w.abs() > f64::from(f32::MAX)) {
            return Err(GpuError("folded normal weights exceed f32 range".into()));
        }
        let weights = folded.map(|w| w as f32);
        executor
            .queue
            .write_buffer(&self.uniform, 0, bytemuck::bytes_of(&u));
        executor
            .queue
            .write_buffer(&self.weights, 0, bytemuck::cast_slice(&weights));
        Ok(())
    }
    pub fn reweight(
        &mut self,
        executor: &GpuExecutor,
        output: &wgpu::Texture,
        p: &NormalParameters,
    ) -> Result<(), GpuError> {
        if output.size() != self.input.size()
            || output.format() != wgpu::TextureFormat::Rgba16Float
            || output == &self.input
        {
            return Err(GpuError("invalid normal output".into()));
        }
        self.write(executor, p)?;
        self.common = common(
            executor.device(),
            &self.pipeline,
            &self.input,
            output,
            &self.uniform,
        );
        Ok(())
    }
    pub fn pyramid_passes(&self) -> &[PreparedPass] {
        self.pyramid.as_ref().map_or(&[], |p| p.passes())
    }
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("normal filters"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.common, &[]);
        pass.set_bind_group(1, &self.additional, &[]);
        pass.dispatch_workgroups(self.groups[0], self.groups[1], 1);
    }
    pub fn mark_submitted(&mut self, fence: FenceId) -> Result<(), GpuError> {
        if let Some(p) = &mut self.pyramid {
            p.mark_submitted(fence)?;
        }
        let mut pool = self.reservations.pool.lock().unwrap();
        for id in &self.reservations.ids {
            pool.mark_submitted(*id, fence)
                .map_err(|e| GpuError(e.to_string()))?;
        }
        Ok(())
    }
    pub fn complete(&mut self, fence: FenceId, ready: bool) {
        self.reservations.pool.lock().unwrap().complete(fence);
        if let Some(p) = &mut self.pyramid {
            p.complete(fence, ready);
        }
    }
    pub fn is_ready(&self) -> bool {
        self.pyramid.as_ref().is_none_or(|p| p.is_ready())
    }
    pub fn has_pyramid(&self) -> bool {
        self.pyramid.is_some()
    }
}

fn common(
    device: &wgpu::Device,
    pipeline: &wgpu::ComputePipeline,
    input: &wgpu::Texture,
    output: &wgpu::Texture,
    uniform: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("normal common"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(
                    &input.create_view(&Default::default()),
                ),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(
                    &output.create_view(&Default::default()),
                ),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: uniform.as_entire_binding(),
            },
        ],
    })
}
fn pipeline(device: &wgpu::Device) -> wgpu::ComputePipeline {
    let sampled = |binding| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    };
    let uniform = |binding| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: std::num::NonZeroU64::new(64),
        },
        count: None,
    };
    let common = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("normal ABI"),
        entries: &[
            sampled(0),
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba16Float,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
            uniform(2),
        ],
    });
    let mut entries: Vec<_> = (0..12).map(sampled).collect();
    entries.push(uniform(12));
    let extra = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("normal pyramid ABI"),
        entries: &entries,
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("normal filters"),
        bind_group_layouts: &[Some(&common), Some(&extra)],
        immediate_size: 0,
    });
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("normal filters"),
        source: wgpu::ShaderSource::Wgsl(
            include_str!("../../shaders/normal_filters/normal.wgsl").into(),
        ),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("normal filters"),
        layout: Some(&layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}
