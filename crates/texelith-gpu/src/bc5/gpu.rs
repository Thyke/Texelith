use crate::{
    GpuError, NodeUniform,
    export_shared::BlockMip,
    pool::{AllocationDesc, AllocationId, FenceId, ResourcePool},
};
use std::sync::{Arc, Mutex};
use texelith_core::Extent;
use wgpu::util::DeviceExt;

/// Bounded endpoint search. Maximum searches the CPU reference's +/-4 range;
/// default tests both palette modes at the extrema for interactive throughput.
#[derive(Clone, Copy, Debug, Default)]
pub enum Bc5Quality {
    #[default]
    Default,
    Maximum,
}

/// DDS mip geometry (floor halving), with offsets into header-free BC5 bytes.
pub fn block_mips(extent: Extent) -> Result<Vec<BlockMip>, GpuError> {
    extent.validate()?;
    let mut e = extent;
    let mut offset = 0;
    let mut mips = Vec::new();
    loop {
        let byte_len = u64::from(e.width.div_ceil(4)) * u64::from(e.height.div_ceil(4)) * 16;
        mips.push(BlockMip {
            extent: e,
            offset,
            byte_len,
        });
        offset += byte_len;
        if e.width == 1 && e.height == 1 {
            break;
        }
        e = Extent::new((e.width / 2).max(1), (e.height / 2).max(1))?;
    }
    Ok(mips)
}

pub struct Bc5Kernel {
    encode: wgpu::ComputePipeline,
    mip: wgpu::ComputePipeline,
    validate: wgpu::ComputePipeline,
}
impl Bc5Kernel {
    pub fn new(device: &wgpu::Device) -> Self {
        let common = crate::executor::common_layout(device, wgpu::TextureFormat::Rgba16Float);
        let extra = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("T-081 BC5 ABI"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(32),
                },
                count: None,
            }],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("T-081 BC5"),
            bind_group_layouts: &[Some(&common), Some(&extra)],
            immediate_size: 0,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("original BC5"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/bc5/codec.wgsl").into()),
        });
        let pipeline = |entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        Self {
            encode: pipeline("encode"),
            mip: pipeline("mip"),
            validate: pipeline("validate"),
        }
    }

    /// Input is a full-resolution encoded XYZ Rgba32Float packing carrier.
    /// `signed_source` validates negative Z before any resampling can hide it.
    /// All scratch, blocks and uniforms are admitted atomically before creation.
    pub fn prepare(
        &self,
        device: &wgpu::Device,
        input: &wgpu::Texture,
        signed_source: Option<&wgpu::Texture>,
        quality: Bc5Quality,
        pool: Arc<Mutex<ResourcePool>>,
    ) -> Result<PreparedBc5, GpuError> {
        let extent = Extent::new(input.width(), input.height())?;
        if input.format() != wgpu::TextureFormat::Rgba32Float
            || !input.usage().contains(wgpu::TextureUsages::TEXTURE_BINDING)
        {
            return Err(GpuError(
                "BC5 requires an encoded XYZ packing carrier".into(),
            ));
        }
        if signed_source.is_some_and(|t| {
            t.format() != wgpu::TextureFormat::Rgba16Float
                || !t.usage().contains(wgpu::TextureUsages::TEXTURE_BINDING)
        }) {
            return Err(GpuError(
                "BC5 validation requires signed normal source".into(),
            ));
        }
        let mips = block_mips(extent)?;
        let bytes = mips.last().unwrap().offset + mips.last().unwrap().byte_len;
        let staging_bytes = (bytes + 16).div_ceil(256) * 256;
        let count = mips.len() * 2 - 1 + usize::from(signed_source.is_some());
        let mut descriptions = vec![AllocationDesc::Staging {
            bytes: staging_bytes,
        }];
        descriptions.push(AllocationDesc::Texture {
            extent: Extent::new(1, 1)?,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING,
        });
        for mip in &mips[1..] {
            descriptions.push(AllocationDesc::Texture {
                extent: mip.extent,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            });
        }
        descriptions.extend((0..count).map(|_| AllocationDesc::Staging { bytes: 64 }));
        let resources = pool
            .lock()
            .unwrap()
            .allocate_backed(device, &descriptions)
            .map_err(|e| GpuError(e.to_string()))?;
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("BC5 blocks and validation status"),
            size: staging_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: true,
        });
        output
            .slice(..16)
            .get_mapped_range_mut()
            .unwrap()
            .copy_from_slice(&[0; 16]);
        output.unmap();
        let blocks = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("BC5 block buffer ABI"),
            layout: &self.encode.get_bind_group_layout(1),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: output.as_entire_binding(),
            }],
        });
        let mut prepared = PreparedBc5 {
            passes: Vec::with_capacity(count),
            output,
            blocks,
            mips,
            staging_bytes,
            pool,
            ids: resources.iter().map(|(id, _)| *id).collect(),
        };
        // Uniforms are initialized through mapped creation, avoiding queue-write
        // staging and allowing replay with no allocations or writes in encode.
        let mut uniform_index = 0;
        let mut add = |pipeline: &wgpu::ComputePipeline,
                       source: &wgpu::Texture,
                       target: &wgpu::Texture,
                       params: [u32; 8],
                       groups: [u32; 2]| {
            let mut node = NodeUniform::new(
                Extent {
                    width: params[0],
                    height: params[1],
                },
                Extent {
                    width: params[2],
                    height: params[3],
                },
            );
            node.flags.copy_from_slice(&params[4..]);
            let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("BC5 params"),
                contents: bytemuck::bytes_of(&node),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            uniform_index += 1;
            let source = source.create_view(&Default::default());
            let target = target.create_view(&Default::default());
            let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("BC5 pass"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&source),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&target),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: uniform.as_entire_binding(),
                    },
                ],
            });
            prepared.passes.push(CodecPass {
                pipeline: pipeline.clone(),
                bindings,
                groups,
            });
        };
        let dummy = resources[1].1.texture();
        if let Some(source) = signed_source {
            add(
                &self.validate,
                source,
                dummy,
                [source.width(), source.height(), 0, 0, 0, 0, 0, 0],
                [source.width().div_ceil(8), source.height().div_ceil(8)],
            );
        }
        for (i, mip) in prepared.mips.clone().iter().enumerate() {
            let e = mip.extent;
            let source = if i == 0 {
                input
            } else {
                resources[i + 1].1.texture()
            };
            if i > 0 {
                let previous = if i == 1 {
                    input
                } else {
                    resources[i].1.texture()
                };
                add(
                    &self.mip,
                    previous,
                    source,
                    [
                        e.width,
                        e.height,
                        previous.width(),
                        previous.height(),
                        0,
                        u32::from(i == 1),
                        0,
                        0,
                    ],
                    [e.width.div_ceil(8), e.height.div_ceil(8)],
                );
            }
            add(
                &self.encode,
                source,
                dummy,
                [
                    e.width,
                    e.height,
                    e.width,
                    e.height,
                    (mip.offset / 4) as u32,
                    u32::from(i == 0),
                    match quality {
                        Bc5Quality::Default => 0,
                        Bc5Quality::Maximum => 4,
                    },
                    0,
                ],
                [
                    e.width.div_ceil(4).div_ceil(8),
                    e.height.div_ceil(4).div_ceil(8),
                ],
            );
        }
        debug_assert_eq!(uniform_index, count);
        Ok(prepared)
    }
}
struct CodecPass {
    pipeline: wgpu::ComputePipeline,
    bindings: wgpu::BindGroup,
    groups: [u32; 2],
}
pub struct PreparedBc5 {
    passes: Vec<CodecPass>,
    output: wgpu::Buffer,
    blocks: wgpu::BindGroup,
    mips: Vec<BlockMip>,
    staging_bytes: u64,
    pool: Arc<Mutex<ResourcePool>>,
    ids: Vec<AllocationId>,
}
impl PreparedBc5 {
    pub fn mips(&self) -> &[BlockMip] {
        &self.mips
    }
    pub fn staging_bytes(&self) -> u64 {
        self.staging_bytes
    }
    pub fn byte_len(&self) -> u64 {
        let m = self.mips.last().unwrap();
        m.offset + m.byte_len
    }
    pub fn pass_count(&self) -> usize {
        self.passes.len()
    }
    pub fn encode_pass(&self, index: usize, encoder: &mut wgpu::CommandEncoder) {
        let p = &self.passes[index];
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&p.pipeline);
        pass.set_bind_group(0, &p.bindings, &[]);
        pass.set_bind_group(1, &self.blocks, &[]);
        pass.dispatch_workgroups(p.groups[0], p.groups[1], 1);
    }
    pub fn encode_readback(&self, encoder: &mut wgpu::CommandEncoder, buffer: &wgpu::Buffer) {
        encoder.copy_buffer_to_buffer(&self.output, 0, buffer, 0, self.staging_bytes);
    }
    pub fn readback_bytes(&self, bytes: &[u8]) -> Result<Vec<u8>, GpuError> {
        if bytes.len() as u64 != self.staging_bytes {
            return Err(GpuError("invalid BC5 readback length".into()));
        }
        if bytes[..4] != [0; 4] {
            return Err(GpuError(
                "negative-Z or invalid normals cannot be exported as BC5; use PNG for full XYZ"
                    .into(),
            ));
        }
        Ok(bytes[16..16 + self.byte_len() as usize].to_vec())
    }
    pub fn mark_submitted(&mut self, fence: FenceId) -> Result<(), GpuError> {
        let mut pool = self.pool.lock().unwrap();
        for id in &self.ids {
            pool.mark_submitted(*id, fence)
                .map_err(|e| GpuError(e.to_string()))?;
        }
        Ok(())
    }
    /// Release scratch only after the final compute fence; blocks remain owned
    /// through copy and CPU consumption by the worker's existing plane charge.
    pub fn finish_compute(&mut self) {
        self.passes.clear();
        let mut pool = self.pool.lock().unwrap();
        for id in self.ids.drain(1..) {
            let _ = pool.release_transient(id);
        }
    }
}
impl Drop for PreparedBc5 {
    fn drop(&mut self) {
        let mut pool = self.pool.lock().unwrap();
        for id in &self.ids {
            let _ = pool.release_transient(*id);
        }
    }
}
