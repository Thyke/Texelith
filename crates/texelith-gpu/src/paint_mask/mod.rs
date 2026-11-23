//! Bounded, ordered UV segment rasterization. Coverage is maximized within each
//! stroke before compositing, so canonical sample spacing cannot build opacity.
use crate::{GpuError, NodeUniform, executor::common_layout};
use std::sync::atomic::{AtomicU64, Ordering};
use texelith_core::{Boundary, BrushSample, BrushStroke, Extent, MaskMode};

pub const MAX_SEGMENTS_PER_BATCH: usize = 256;
pub const SEGMENT_BYTES: u64 = 32;
static CHECKPOINT_HITS: AtomicU64 = AtomicU64::new(0);
static CHECKPOINT_BYTES: AtomicU64 = AtomicU64::new(0);
static PEAK_SEGMENTS: AtomicU64 = AtomicU64::new(0);

/// Process-wide counters for serial verification; never used for scheduling.
pub fn diagnostics() -> (u64, u64, u64) {
    (
        CHECKPOINT_HITS.load(Ordering::Relaxed),
        CHECKPOINT_BYTES.load(Ordering::Relaxed),
        PEAK_SEGMENTS.load(Ordering::Relaxed),
    )
}
pub(crate) fn checkpoint_hit(bytes: u64) {
    CHECKPOINT_HITS.fetch_add(1, Ordering::Relaxed);
    CHECKPOINT_BYTES.fetch_max(bytes, Ordering::Relaxed);
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Segment {
    a: [f32; 4],
    b: [f32; 4],
}
impl Segment {
    fn new(a: BrushSample, b: BrushSample) -> Self {
        Self {
            a: [a.uv[0], a.uv[1], a.pressure, 0.],
            b: [b.uv[0], b.uv[1], b.pressure, 0.],
        }
    }
}

const UNIFORM_OFFSET: u64 = SEGMENT_BYTES * MAX_SEGMENTS_PER_BATCH as u64;
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct BatchUpload {
    segments: [Segment; MAX_SEGMENTS_PER_BATCH],
    uniform: NodeUniform,
}
const _: () = {
    assert!(std::mem::offset_of!(BatchUpload, uniform) as u64 == UNIFORM_OFFSET);
    assert!(std::mem::size_of::<BatchUpload>() as u64 == UNIFORM_OFFSET + 64);
};

/// Reusable pipelines. All uploads are bounded by MAX_SEGMENTS_PER_BATCH.
pub struct PaintKernel {
    pipeline: wgpu::ComputePipeline,
}
impl PaintKernel {
    pub fn new(device: &wgpu::Device) -> Self {
        let common = common_layout(device, wgpu::TextureFormat::R32Float);
        let extra = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("paint storage ABI"),
            entries: &[0, 1].map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage {
                        read_only: binding == 1,
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("paint"),
            bind_group_layouts: &[Some(&common), Some(&extra)],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("paint"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/paint_mask/raster.wgsl").into(),
            ),
        });
        Self {
            pipeline: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("paint"),
                layout: Some(&layout),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            }),
        }
    }
    /// Input is a completed prefix checkpoint (or a distinct zero dummy). Output
    /// and scratch use f32 to preserve the scalar contract across long replays.
    pub fn prepare(
        &self,
        device: &wgpu::Device,
        input: &wgpu::Texture,
        output: &wgpu::Texture,
    ) -> Result<PreparedPaint, GpuError> {
        let extent = Extent::new(output.width(), output.height())?;
        let single = |t: &wgpu::Texture| {
            t.dimension() == wgpu::TextureDimension::D2
                && t.depth_or_array_layers() == 1
                && t.mip_level_count() == 1
                && t.sample_count() == 1
        };
        let pixel_bytes = u64::from(extent.width) * u64::from(extent.height) * 8;
        if input == output
            || !single(input)
            || !single(output)
            || output.format() != wgpu::TextureFormat::R32Float
            || input.format() != wgpu::TextureFormat::R32Float
            || !input.usage().contains(wgpu::TextureUsages::TEXTURE_BINDING)
            || !output
                .usage()
                .contains(wgpu::TextureUsages::STORAGE_BINDING)
            || !((input.width() == extent.width && input.height() == extent.height)
                || (input.width() == 1 && input.height() == 1))
            || pixel_bytes > device.limits().max_storage_buffer_binding_size
        {
            return Err(GpuError(
                "paint requires distinct, compatible R32 textures and supported scratch size"
                    .into(),
            ));
        }
        let buffer = |size, usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("paint scratch"),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let pixels = buffer(
            pixel_bytes,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        // Disjoint storage/uniform ranges share one upload and the same byte budget.
        let upload = buffer(
            UNIFORM_OFFSET + 64,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::UNIFORM
                | wgpu::BufferUsages::COPY_DST,
        );
        let iv = input.create_view(&Default::default());
        let ov = output.create_view(&Default::default());
        let common = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("paint common"),
            layout: &self.pipeline.get_bind_group_layout(0),
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
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &upload,
                        offset: UNIFORM_OFFSET,
                        size: std::num::NonZeroU64::new(64),
                    }),
                },
            ],
        });
        let extra = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("paint scratch"),
            layout: &self.pipeline.get_bind_group_layout(1),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: pixels.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &upload,
                        offset: 0,
                        size: std::num::NonZeroU64::new(UNIFORM_OFFSET),
                    }),
                },
            ],
        });
        Ok(PreparedPaint {
            pipeline: self.pipeline.clone(),
            common,
            extra,
            pixels,
            upload,
            extent,
        })
    }
}

/// Prepared resources reused for all batches and timing iterations. Queue writes
/// must be followed by submission/completion before reusing the upload buffers.
pub struct PreparedPaint {
    pipeline: wgpu::ComputePipeline,
    common: wgpu::BindGroup,
    extra: wgpu::BindGroup,
    pixels: wgpu::Buffer,
    upload: wgpu::Buffer,
    extent: Extent,
}
impl PreparedPaint {
    pub fn initialize(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        checkpoint: bool,
    ) {
        if !checkpoint {
            // Clearing both f32 lanes is exactly the zero-initialization shader.
            encoder.clear_buffer(&self.pixels, 0, None);
            return;
        }
        let mut u = NodeUniform::new(self.extent, self.extent);
        u.flags = [0, u32::from(checkpoint), 0, 0];
        self.encode(queue, encoder, u);
    }
    pub fn finish(&self, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder) {
        let mut u = NodeUniform::new(self.extent, self.extent);
        u.flags[0] = 3;
        self.encode(queue, encoder, u);
    }
    /// Encode one chunk of a validated core stroke. `start` is zero or the
    /// previous return value, and must be less than the stroke point count.
    pub fn batch(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        stroke: &BrushStroke,
        boundary: Boundary,
        start: usize,
    ) -> usize {
        let count = stroke.points.len();
        let end = (start + MAX_SEGMENTS_PER_BATCH).min(count);
        let mut segments = [bytemuck::Zeroable::zeroed(); MAX_SEGMENTS_PER_BATCH];
        for (slot, i) in (start..end).enumerate() {
            let a = stroke.points[i.saturating_sub(1)];
            let b = stroke.points[i];
            segments[slot] = Segment::new(a, b);
        }
        PEAK_SEGMENTS.fetch_max((end - start) as u64, Ordering::Relaxed);
        let mut u = NodeUniform::new(self.extent, self.extent);
        u.parameters = [
            stroke.brush.radius,
            stroke.brush.hardness,
            stroke.brush.opacity,
            0.,
        ];
        u.flags = [
            1,
            (end - start) as u32,
            u32::from(boundary == Boundary::Repeat),
            match stroke.brush.mode {
                MaskMode::Add => 0,
                MaskMode::Subtract => 1,
                MaskMode::Replace => 2,
            },
        ];
        // The final batch composites the maximum accumulated coverage once.
        u.reserved[0] = u32::from(end == count);
        // Dispatch only the union of affected pixel footprints. The final batch
        // also visits earlier batches before applying the stroke exactly once.
        let points = if end == count {
            &stroke.points[..]
        } else {
            &stroke.points[start.saturating_sub(1)..end]
        };
        let mut lo = [1.0f32; 2];
        let mut hi = [0.0f32; 2];
        for p in points {
            for axis in 0..2 {
                lo[axis] = lo[axis].min(p.uv[axis] - stroke.brush.radius);
                hi[axis] = hi[axis].max(p.uv[axis] + stroke.brush.radius);
            }
        }
        // A footprint inside the tile cannot reach another periodic image.
        // Use the direct projection path for these common interior strokes.
        u.flags[2] = u32::from(
            boundary == Boundary::Repeat && (0..2).any(|axis| lo[axis] < 0. || hi[axis] > 1.),
        );
        let size = [self.extent.width, self.extent.height];
        for axis in 0..2 {
            if boundary == Boundary::Repeat && (lo[axis] < 0. || hi[axis] > 1.) {
                lo[axis] = 0.;
                hi[axis] = 1.;
            }
            let low = ((lo[axis].max(0.) * size[axis] as f32).floor() as u32).min(size[axis] - 1);
            let high = ((hi[axis].min(1.) * size[axis] as f32).ceil() as u32)
                .min(size[axis])
                .max(low + 1);
            u.reserved[axis + 1] = low;
            u.dimensions[axis + 2] = high - low;
        }
        queue.write_buffer(
            &self.upload,
            0,
            bytemuck::bytes_of(&BatchUpload {
                segments,
                uniform: u,
            }),
        );
        self.dispatch(encoder, u);
        end
    }
    fn encode(&self, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder, u: NodeUniform) {
        queue.write_buffer(&self.upload, UNIFORM_OFFSET, bytemuck::bytes_of(&u));
        self.dispatch(encoder, u);
    }
    fn dispatch(&self, encoder: &mut wgpu::CommandEncoder, u: NodeUniform) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("paint batch"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.common, &[]);
        pass.set_bind_group(1, &self.extra, &[]);
        pass.dispatch_workgroups(u.dimensions[2].div_ceil(8), u.dimensions[3].div_ceil(8), 1);
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn segment_abi() {
        assert_eq!(
            std::mem::size_of::<super::Segment>(),
            super::SEGMENT_BYTES as usize
        );
    }
}
