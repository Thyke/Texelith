//! T-063 test/harness I/O. Rendering always goes through the production worker.
use std::{
    sync::Arc,
    time::{Duration, Instant},
};
use texelith_core::{
    EvaluationSnapshot, Extent, Graph, MaterialGraphDto, MaterialNodeParams, Node, NodeId,
    NodeParams, ViewState,
};
use texelith_gpu::{
    CompletedMaterial, CompletedPreview, GpuConfig, GpuWorker, JobId, MaterialRequest,
    PreviewRequest, Quality, TypedSourcePixels, TypedSourceUpload, WorkerEvent,
    viewport_shared::PreviewContent,
};

pub type Result<T> = std::result::Result<T, String>;
pub const BUDGET: u64 = 1_073_741_824;
pub struct Gpu {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
}
pub fn require(ok: bool, message: &str) -> Result<()> {
    if ok { Ok(()) } else { Err(message.into()) }
}
pub fn wait<T>(mut next: impl FnMut() -> Result<Option<T>>) -> Result<T> {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        if let Some(value) = next()? {
            return Ok(value);
        }
        if Instant::now() >= deadline {
            return Err("T-063 worker timeout".into());
        }
        std::thread::sleep(Duration::from_millis(1));
    }
}
pub fn no_error(worker: &GpuWorker) -> Result<()> {
    match worker.try_event() {
        None => Ok(()),
        Some(event) => Err(format!("unexpected worker event: {event:?}")),
    }
}
impl Gpu {
    pub fn worker(&self, budget_bytes: u64) -> Result<GpuWorker> {
        GpuWorker::spawn(
            self.device.clone(),
            self.queue.clone(),
            GpuConfig {
                budget_bytes,
                ..Default::default()
            },
        )
        .map_err(|e| e.to_string())
    }
    pub fn finish(&self) -> Result<()> {
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_secs(30)),
            })
            .map_err(|e| e.to_string())?;
        Ok(())
    }
    /// Readback belongs exclusively to verification, never the preview renderer.
    pub fn read_display(&self, texture: &wgpu::Texture) -> Result<Vec<u8>> {
        require(
            texture.format() == wgpu::TextureFormat::Rgba8Unorm,
            "display must contain explicit gamma codes in Rgba8Unorm",
        )?;
        require(
            texture
                .usage()
                .contains(wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::TEXTURE_BINDING),
            "display needs sample/readback usages",
        )?;
        let tight = texture.width() * 4;
        let pitch = tight.div_ceil(256) * 256;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("T-063 verification readback, outside application ledger"),
            size: u64::from(pitch) * u64::from(texture.height()),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(pitch),
                    rows_per_image: Some(texture.height()),
                },
            },
            texture.size(),
        );
        self.queue.submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });
        self.finish()?;
        rx.recv_timeout(Duration::from_secs(30))
            .map_err(|e| e.to_string())?
            .map_err(|e| e.to_string())?;
        let mapped = buffer
            .slice(..)
            .get_mapped_range()
            .map_err(|e| e.to_string())?;
        let bytes = mapped
            .chunks_exact(pitch as usize)
            .flat_map(|row| row[..tight as usize].iter().copied())
            .collect();
        drop(mapped);
        buffer.unmap();
        Ok(bytes)
    }
}

/// Tiny height + normal material, using only T-046's existing typed source path.
pub fn material(
    worker: &GpuWorker,
    job: u64,
    generation: u64,
    serial: u64,
) -> Result<Arc<CompletedMaterial>> {
    let extent = Extent::new(17, 3).map_err(|e| e.to_string())?;
    let pixels = TypedSourcePixels::HeightF32(
        (0..51)
            .map(|i| (i % 17) as f32 / 16.0)
            .collect::<Vec<_>>()
            .into(),
    );
    let descriptor = pixels.descriptor(extent).map_err(|e| e.to_string())?;
    worker
        .install_typed_source(TypedSourceUpload {
            descriptor: descriptor.clone(),
            pixels,
        })
        .map_err(|e| e.to_string())?;
    let ready = wait(|| Ok(worker.try_event()))?;
    require(
        matches!(ready, WorkerEvent::SourceReady { content_hash } if content_hash == descriptor.content_hash),
        "source acknowledgement mismatch",
    )?;
    worker
        .request_material(material_request(descriptor, job, generation, serial)?)
        .map_err(|e| e.to_string())?;
    let bundle = wait(|| {
        no_error(worker)?;
        Ok(worker.try_latest_material())
    })?;
    require(
        bundle.job_id == JobId(job)
            && bundle.generation == generation
            && bundle.request_id == serial
            && bundle.outputs.len() == 2,
        "material must publish a coherent height/normal pair",
    )?;
    Ok(Arc::new(bundle))
}
pub fn material_request(
    descriptor: texelith_core::SourceDescriptor,
    job: u64,
    generation: u64,
    serial: u64,
) -> Result<MaterialRequest> {
    let extent = descriptor.extent;
    let legacy = Graph::new(
        vec![
            Node {
                id: NodeId(1),
                input: None,
                params: NodeParams::Source(descriptor.clone()),
            },
            Node {
                id: NodeId(2),
                input: Some(NodeId(1)),
                params: NodeParams::NormalFromHeight { strength: 1.0 },
            },
        ],
        NodeId(2),
    )
    .map_err(|e| e.to_string())?;
    let mut dto = MaterialGraphDto::from(legacy);
    // The uploaded scalar data uses the typed passthrough, not legacy RGBA decoding.
    dto.graph[0].params = MaterialNodeParams::HeightSource(descriptor);
    let graph = Graph::try_from(dto).map_err(|e| e.to_string())?;
    Ok(MaterialRequest {
        job_id: JobId(job),
        request_id: serial,
        outputs: graph
            .material_outputs()
            .ok_or("missing material outputs")?
            .clone(),
        snapshot: Arc::new(EvaluationSnapshot { generation, graph }),
        extent,
        quality: Quality::Full,
    })
}
pub fn request(
    material: Arc<CompletedMaterial>,
    id: u64,
    extent: Extent,
    content: PreviewContent,
) -> PreviewRequest {
    PreviewRequest {
        job_id: material.job_id,
        generation: material.generation,
        request_id: id,
        material,
        extent,
        view: ViewState::default(),
        content,
    }
}
pub fn complete(worker: &GpuWorker) -> Result<CompletedPreview> {
    wait(|| {
        no_error(worker)?;
        Ok(worker.try_latest_preview())
    })
}
pub fn render(worker: &GpuWorker, request: PreviewRequest) -> Result<CompletedPreview> {
    let id = request.request_id;
    let generation = request.generation;
    let job = request.job_id;
    worker.request_preview(request).map_err(|e| e.to_string())?;
    let frame = complete(worker)?;
    require(
        frame.request_id == id && frame.generation == generation && frame.job_id == job,
        "preview publication identity mismatch",
    )?;
    Ok(frame)
}
/// Independent display oracle: no ACES in this diagnostic linear ramp/clear.
/// Background clear and GUI colors never pass through material tone mapping.
pub fn srgb_code(linear: f64) -> u8 {
    let v = if linear <= 0.0031308 {
        12.92 * linear
    } else {
        1.055 * linear.powf(1.0 / 2.4) - 0.055
    };
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}
pub fn check_clear(bytes: &[u8], color: [f32; 4]) -> Result<()> {
    require(
        !bytes.is_empty() && bytes.len().is_multiple_of(4),
        "empty or malformed display",
    )?;
    let expected = [
        srgb_code(color[0] as f64),
        srgb_code(color[1] as f64),
        srgb_code(color[2] as f64),
        255,
    ];
    require(
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .all(|p| p.iter().zip(expected).all(|(a, b)| a.abs_diff(b) <= 1)),
        "clear color: per-channel error exceeds one display code",
    )
}

pub fn check_ramp(bytes: &[u8], extent: Extent) -> Result<()> {
    require(
        bytes.len() == extent.width as usize * extent.height as usize * 4,
        "wrong ramp display length",
    )?;
    for (i, pixel) in bytes.as_chunks::<4>().0.iter().enumerate() {
        let x = (i % extent.width as usize) as f64;
        let source_x = ((x + 0.5) * 17.0 / f64::from(extent.width) - 0.5).clamp(0.0, 16.0);
        let expected = srgb_code(source_x / 16.0);
        require(
            pixel[..3].iter().all(|c| c.abs_diff(expected) <= 1) && pixel[3] == 255,
            "material ramp: manual bilinear/one sRGB encode differs by more than one code",
        )?;
    }
    Ok(())
}
