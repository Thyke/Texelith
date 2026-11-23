//! T-047 height reconstruction and explicit global min/max remapping.
use crate::{
    executor::{common_layout, texture_extent},
    pool::{AllocationDesc, AllocationId, FenceId, ResourcePool},
    *,
};
use std::sync::{Arc, Mutex};
use texelith_core::{Boundary, Extent, SourceDescriptor};

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
impl Reservations {
    fn submitted(&self, fence: FenceId) -> Result<(), GpuError> {
        let mut pool = self.pool.lock().unwrap();
        for id in &self.ids {
            pool.mark_submitted(*id, fence)
                .map_err(|e| GpuError(e.to_string()))?;
        }
        Ok(())
    }
    fn complete(&self, fence: FenceId) {
        self.pool.lock().unwrap().complete(fence);
    }
}
fn texture(extent: Extent) -> AllocationDesc {
    AllocationDesc::Texture {
        extent,
        format: wgpu::TextureFormat::R32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
    }
}
fn sample_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}
impl GpuExecutor {
    fn height_pass(
        &self,
        input: &wgpu::Texture,
        output: &wgpu::Texture,
        extra: [&wgpu::Texture; 2],
        uniform: NodeUniform,
        buffer: &wgpu::Buffer,
    ) -> Result<PreparedPass, GpuError> {
        let pipeline = self
            .height_pipeline
            .get_or_init(|| {
                let common = common_layout(self.device(), wgpu::TextureFormat::R32Float);
                let extra =
                    self.device()
                        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                            label: Some("height inputs"),
                            entries: &[sample_entry(0), sample_entry(1)],
                        });
                let layout =
                    self.device()
                        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                            label: Some("height"),
                            bind_group_layouts: &[Some(&common), Some(&extra)],
                            immediate_size: 0,
                        });
                let module = self
                    .device()
                    .create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("height"),
                        source: wgpu::ShaderSource::Wgsl(
                            concat!(
                                include_str!("../shaders/wrapped.wgsl"),
                                "\n",
                                include_str!("../shaders/height/height.wgsl")
                            )
                            .into(),
                        ),
                    });
                self.device()
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("height"),
                        layout: Some(&layout),
                        module: &module,
                        entry_point: Some("main"),
                        compilation_options: Default::default(),
                        cache: None,
                    })
            })
            .clone();
        self.queue
            .write_buffer(buffer, 0, bytemuck::bytes_of(&uniform));
        let iv = input.create_view(&Default::default());
        let ov = output.create_view(&Default::default());
        let common = self.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("height common"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&iv),
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
        let views = extra.map(|t| t.create_view(&Default::default()));
        let additional = self.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("height extras"),
            layout: &pipeline.get_bind_group_layout(1),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&views[0]),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&views[1]),
                },
            ],
        });
        let evaluation = Extent::new(uniform.dimensions[0], uniform.dimensions[1])?;
        Ok(PreparedPass {
            pipeline,
            common,
            buffer: buffer.clone(),
            weights: None,
            queue: self.queue.clone(),
            operation: Pass::Downsample,
            evaluation,
            source: texture_extent(input)?,
            groups: workgroups(evaluation)?,
            additional: Some(additional),
        })
    }
}
fn uniform(
    input: &wgpu::Texture,
    output: &wgpu::Texture,
    op: u32,
    weight: f32,
    boundary: Boundary,
) -> Result<NodeUniform, GpuError> {
    let mut u = NodeUniform::new(texture_extent(output)?, texture_extent(input)?);
    u.flags = [op, u32::from(boundary == Boundary::Repeat), 0, 0];
    u.parameters[0] = weight;
    Ok(u)
}
fn validate(
    input: &wgpu::Texture,
    output: &wgpu::Texture,
    color: bool,
) -> Result<Extent, GpuError> {
    let source = texture_extent(input)?;
    let extent = texture_extent(output)?;
    if input == output
        || input.format()
            != if color {
                wgpu::TextureFormat::Rgba16Float
            } else {
                wgpu::TextureFormat::R32Float
            }
        || output.format() != wgpu::TextureFormat::R32Float
        || !input.usage().contains(wgpu::TextureUsages::TEXTURE_BINDING)
        || !output
            .usage()
            .contains(wgpu::TextureUsages::STORAGE_BINDING)
        || (!color && source != extent)
        || extent.width > source.width
        || extent.height > source.height
    {
        return Err(GpuError("invalid height input/output textures".into()));
    }
    Ok(extent)
}

/// Weights refer to octaves in the original source, including during proxy work.
#[derive(Clone, Debug)]
pub struct HeightParameters {
    pub band_weights: Vec<f32>,
    pub coarse_weight: f32,
    pub remove_coarse: bool,
    pub boundary: Boundary,
}
impl HeightParameters {
    fn validate(&self) -> Result<(), GpuError> {
        if self.band_weights.is_empty()
            || self.band_weights.len() > 16
            || self.band_weights.iter().any(|w| !w.is_finite())
            || !self.coarse_weight.is_finite()
        {
            return Err(GpuError("invalid height weights".into()));
        }
        Ok(())
    }
}

/// Reusable full node. Every texture, uniform and queue-upload byte is admitted
/// before preparation. Input/output payloads are charged by the caller.
/// Encoding allocates nothing; acknowledge real fences before reweighting.
pub struct PreparedHeight {
    pyramid: PreparedPyramid,
    luminance: wgpu::Texture,
    coarse: wgpu::Texture,
    buffers: Vec<wgpu::Buffer>,
    reservations: Reservations,
    passes: Vec<PreparedPass>,
    prefix: usize,
    source_extent: Extent,
    boundary: Boundary,
    scratch_bytes: u64,
}
impl PreparedHeight {
    pub fn new(
        executor: &GpuExecutor,
        source: &SourceDescriptor,
        input: &wgpu::Texture,
        output: &wgpu::Texture,
        params: &HeightParameters,
        pool: Arc<Mutex<ResourcePool>>,
    ) -> Result<Self, GpuError> {
        params.validate()?;
        let extent = validate(input, output, true)?;
        if texture_extent(input)? != source.extent {
            return Err(GpuError("height source extent mismatch".into()));
        }
        let levels = pyramid_extents(extent)?.len();
        let mut desc = vec![texture(extent), texture(Extent::new(1, 1)?)];
        for _ in 0..levels + 1 {
            desc.extend([
                AllocationDesc::Buffer { bytes: 64 },
                AllocationDesc::Staging { bytes: 64 },
            ]);
        }
        let scratch_bytes = desc.iter().map(|d| d.byte_len().unwrap()).sum::<u64>();
        let resources = pool
            .lock()
            .unwrap()
            .allocate_backed(executor.device(), &desc)
            .map_err(|e| GpuError(e.to_string()))?;
        let reservations = Reservations {
            pool: pool.clone(),
            ids: resources.iter().map(|(id, _)| *id).collect(),
        };
        let luminance = resources[0].1.texture().clone();
        let coarse = resources[1].1.texture().clone();
        let buffers: Vec<_> = resources[2..]
            .as_chunks::<2>()
            .0
            .iter()
            .map(|r| r[0].1.buffer().clone())
            .collect();
        let scalar_source = SourceDescriptor {
            content_hash: source.content_hash,
            extent,
            interpretation: texelith_core::InputInterpretation::LinearHeight,
        };
        let pyramid =
            PreparedPyramid::new(executor, &scalar_source, &luminance, params.boundary, pool)?;
        let scratch_bytes = scratch_bytes + pyramid.scratch_bytes();
        let first = executor.height_pass(
            input,
            &luminance,
            [input, input],
            uniform(input, &luminance, 0, 0.0, params.boundary)?,
            &buffers[0],
        )?;
        let mut passes = vec![first];
        passes.extend_from_slice(pyramid.passes());
        let prefix = passes.len();
        let mut prepared = Self {
            pyramid,
            luminance,
            coarse,
            buffers,
            reservations,
            passes,
            prefix,
            source_extent: source.extent,
            boundary: params.boundary,
            scratch_bytes,
        };
        prepared.reweight(executor, output, params)?;
        Ok(prepared)
    }
    /// Rebind only after all preceding submissions have completed. The Gaussian
    /// chain remains valid when weights change; changing boundary requires a new node.
    pub fn reweight(
        &mut self,
        executor: &GpuExecutor,
        output: &wgpu::Texture,
        params: &HeightParameters,
    ) -> Result<(), GpuError> {
        params.validate()?;
        if params.boundary != self.boundary
            || texture_extent(output)? != texture_extent(&self.luminance)?
            || output.format() != wgpu::TextureFormat::R32Float
            || !output
                .usage()
                .contains(wgpu::TextureUsages::STORAGE_BINDING)
        {
            return Err(GpuError("height reweight extent/boundary mismatch".into()));
        }
        self.passes.truncate(self.prefix);
        let levels = self.pyramid.levels();
        let n = levels.len();
        let coarse = if params.remove_coarse {
            0.0
        } else {
            params.coarse_weight
        };
        let target = if n == 1 { output } else { &self.coarse };
        self.passes.push(executor.height_pass(
            &levels[n - 1],
            target,
            [&levels[n - 1]; 2],
            uniform(&levels[n - 1], target, 1, coarse, self.boundary)?,
            &self.buffers[1],
        )?);
        let ratio = self.source_extent.width.max(self.source_extent.height) as f64
            / self.luminance.width().max(self.luminance.height()) as f64;
        let offset = ratio.log2().round().max(0.0) as usize;
        let mut accumulated = target;
        for l in (0..n - 1).rev() {
            let target = if l == 0 {
                output
            } else {
                &self.pyramid._scratch[l]
            };
            let weight = *params
                .band_weights
                .get(l + offset)
                .unwrap_or(params.band_weights.last().unwrap());
            self.passes.push(executor.height_pass(
                &levels[l],
                target,
                [&levels[l + 1], accumulated],
                uniform(&levels[l], target, 2, weight, self.boundary)?,
                &self.buffers[l + 2],
            )?);
            accumulated = target;
        }
        Ok(())
    }
    pub fn passes(&self) -> &[PreparedPass] {
        &self.passes
    }
    pub fn reconstruction_passes(&self) -> &[PreparedPass] {
        &self.passes[self.prefix..]
    }
    pub fn scratch_bytes(&self) -> u64 {
        self.scratch_bytes
    }
    pub fn levels(&self) -> usize {
        self.pyramid.levels().len()
    }
    pub fn mark_submitted(&mut self, fence: FenceId) -> Result<(), GpuError> {
        self.reservations.submitted(fence)?;
        self.pyramid.mark_submitted(fence)
    }
    pub fn complete(&mut self, fence: FenceId, complete: bool) {
        self.reservations.complete(fence);
        self.pyramid.complete(fence, complete);
    }
    pub fn is_ready(&self) -> bool {
        self.pyramid.is_ready()
    }
}

/// Explicit min/max mapping; equality alone selects midlevel, so adjacent f32
/// values retain their full range. Reduction stays on the GPU.
pub struct PreparedMinMaxRemap {
    passes: Vec<PreparedPass>,
    reservations: Reservations,
}
impl PreparedMinMaxRemap {
    pub(crate) fn scratch_bytes(mut extent: Extent) -> u64 {
        let mut bytes = 128; // Final remap uniform and upload staging.
        loop {
            extent = Extent {
                width: extent.width.div_ceil(8),
                height: extent.height.div_ceil(8),
            };
            bytes += u64::from(extent.width) * u64::from(extent.height) * 8 + 128;
            if extent.width == 1 && extent.height == 1 {
                return bytes;
            }
        }
    }

    pub fn new(
        executor: &GpuExecutor,
        input: &wgpu::Texture,
        output: &wgpu::Texture,
        midlevel: f32,
        pool: Arc<Mutex<ResourcePool>>,
    ) -> Result<Self, GpuError> {
        let extent = validate(input, output, false)?;
        if !midlevel.is_finite() || !(0.0..=1.0).contains(&midlevel) {
            return Err(GpuError("invalid remap midlevel".into()));
        }
        let mut sizes = Vec::new();
        let mut size = extent;
        loop {
            size = Extent::new(size.width.div_ceil(8), size.height.div_ceil(8))?;
            sizes.push(size);
            if size.width == 1 && size.height == 1 {
                break;
            }
        }
        let mut desc: Vec<_> = sizes
            .iter()
            .map(|e| {
                texture(Extent {
                    width: e.width * 2,
                    height: e.height,
                })
            })
            .collect();
        for _ in 0..=sizes.len() {
            desc.extend([
                AllocationDesc::Buffer { bytes: 64 },
                AllocationDesc::Staging { bytes: 64 },
            ]);
        }
        let resources = pool
            .lock()
            .unwrap()
            .allocate_backed(executor.device(), &desc)
            .map_err(|e| GpuError(e.to_string()))?;
        let reservations = Reservations {
            pool,
            ids: resources.iter().map(|(id, _)| *id).collect(),
        };
        let mut passes = Vec::new();
        let mut source = input;
        for (i, size) in sizes.iter().enumerate() {
            let target = resources[i].1.texture();
            let mut u = uniform(
                source,
                target,
                if i == 0 { 3 } else { 4 },
                0.0,
                Boundary::Clamp,
            )?;
            if i > 0 {
                u.dimensions[2] = sizes[i - 1].width;
                u.dimensions[3] = sizes[i - 1].height;
            }
            debug_assert_eq!(target.width(), size.width * 2);
            passes.push(executor.height_pass(
                source,
                target,
                [source; 2],
                u,
                resources[sizes.len() + 2 * i].1.buffer(),
            )?);
            source = target;
        }
        passes.push(executor.height_pass(
            input,
            output,
            [source; 2],
            uniform(input, output, 5, midlevel, Boundary::Clamp)?,
            resources[sizes.len() + 2 * sizes.len()].1.buffer(),
        )?);
        Ok(Self {
            passes,
            reservations,
        })
    }
    pub fn passes(&self) -> &[PreparedPass] {
        &self.passes
    }
    pub fn mark_submitted(&mut self, fence: FenceId) -> Result<(), GpuError> {
        self.reservations.submitted(fence)
    }
    pub fn complete(&mut self, fence: FenceId) {
        self.reservations.complete(fence);
    }
}

/// Conservative full-node scratch, including the persistent Gaussian chain,
/// horizontal filter targets, reconstruction uniforms and upload staging.
pub(crate) fn scratch_bytes(extent: Extent) -> Result<u64, GpuError> {
    let levels = pyramid_extents(extent)?;
    let pixels = |e: Extent| u64::from(e.width) * u64::from(e.height) * 4;
    Ok(pixels(extent)
        + 4
        + levels[1..].iter().copied().map(pixels).sum::<u64>()
        + levels[..levels.len() - 1]
            .iter()
            .copied()
            .map(pixels)
            .sum::<u64>()
        + (levels.len() as u64 + 1) * 128
        + (levels.len() as u64 - 1) * 256)
}
