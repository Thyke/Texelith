use std::{
    sync::Arc,
    time::{Duration, Instant},
};
use texelith_core::*;
use texelith_gpu::*;
pub type Result<T> = std::result::Result<T, String>;
pub const BUDGET: u64 = 1_073_741_824;
pub struct Gpu {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
}
pub fn require(ok: bool, message: &str) -> Result<()> {
    if ok { Ok(()) } else { Err(message.into()) }
}
pub fn wait<T>(mut f: impl FnMut() -> Result<Option<T>>) -> Result<T> {
    let start = Instant::now();
    loop {
        if let Some(v) = f()? {
            return Ok(v);
        }
        if start.elapsed() > Duration::from_secs(60) {
            return Err("IBL timeout".into());
        }
        std::thread::sleep(Duration::from_millis(1));
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
    pub fn read(&self, texture: &wgpu::Texture, bpp: u32) -> Result<Vec<u8>> {
        let tight = texture.width() * bpp;
        let pitch = tight.div_ceil(256) * 256;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("T-067 verification readback outside app ledger"),
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
        buffer.slice(..).map_async(wgpu::MapMode::Read, move |v| {
            let _ = tx.send(v);
        });
        self.finish()?;
        rx.recv_timeout(Duration::from_secs(30))
            .map_err(|e| e.to_string())?
            .map_err(|e| e.to_string())?;
        let mapped = buffer
            .slice(..)
            .get_mapped_range()
            .map_err(|e| e.to_string())?;
        let out = mapped
            .chunks_exact(pitch as usize)
            .flat_map(|r| r[..tight as usize].iter().copied())
            .collect();
        drop(mapped);
        buffer.unmap();
        Ok(out)
    }
}
pub fn material(worker: &GpuWorker) -> Result<Arc<CompletedMaterial>> {
    let extent = Extent::new(1, 1).map_err(|e| e.to_string())?;
    let pixels = TypedSourcePixels::Rgba8 {
        pixels: Arc::from([200, 155, 95, 255]),
        interpretation: InputInterpretation::SrgbPhoto,
    };
    let descriptor = pixels.descriptor(extent).map_err(|e| e.to_string())?;
    worker
        .install_typed_source(TypedSourceUpload {
            descriptor: descriptor.clone(),
            pixels,
        })
        .map_err(|e| e.to_string())?;
    let event = wait(|| Ok(worker.try_event()))?;
    require(
        matches!(event, WorkerEvent::SourceReady { .. }),
        "source failed",
    )?;
    let output = OutputRef {
        node: NodeId(1),
        port: OutputPort::Albedo,
    };
    let outputs = MaterialOutputs::from([(MapKind::Albedo, output)]);
    let graph = Graph::try_from(MaterialGraphDto {
        schema_version: 2,
        graph: vec![MaterialNode {
            id: NodeId(1),
            inputs: vec![],
            params: MaterialNodeParams::ColorSource(descriptor),
        }],
        outputs: outputs.clone(),
        selected_output: output,
    })
    .map_err(|e| e.to_string())?;
    worker
        .request_material(MaterialRequest {
            job_id: JobId(67),
            request_id: 1,
            snapshot: Arc::new(EvaluationSnapshot {
                generation: 1,
                graph,
            }),
            outputs,
            extent,
            quality: Quality::Full,
        })
        .map_err(|e| e.to_string())?;
    wait(|| {
        no_error(worker)?;
        Ok(worker.try_latest_material().map(Arc::new))
    })
}
pub fn no_error(worker: &GpuWorker) -> Result<()> {
    if let Some(e) = worker.try_event() {
        Err(format!("worker error: {e:?}"))
    } else {
        Ok(())
    }
}
pub fn request(
    material: Arc<CompletedMaterial>,
    id: u64,
    extent: Extent,
    environment: Arc<ibl::Environment>,
    r: f32,
    m: f32,
) -> PreviewRequest {
    PreviewRequest {
        job_id: material.job_id,
        generation: material.generation,
        request_id: id,
        material,
        extent,
        view: ViewState {
            mesh: PreviewMesh::Sphere,
            light: Light {
                intensity: 0.0,
                ..Default::default()
            },
            ..Default::default()
        },
        content: viewport_shared::PreviewContent::Ibl {
            settings: pbr::PbrSettings {
                roughness: Some(r),
                metallic: Some(m),
                ..Default::default()
            },
            environment,
            exposure: 0.0,
        },
    }
}
pub fn complete(worker: &GpuWorker, id: u64) -> Result<CompletedPreview> {
    let frame = wait(|| {
        no_error(worker)?;
        Ok(worker.try_latest_preview())
    })?;
    require(frame.request_id == id, "stale environment publication")?;
    Ok(frame)
}
pub fn render(worker: &GpuWorker, intent: PreviewRequest) -> Result<CompletedPreview> {
    let id = intent.request_id;
    worker.request_preview(intent).map_err(|e| e.to_string())?;
    complete(worker, id)
}
pub fn half(bits: u16) -> f32 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let e = (bits >> 10) & 31;
    let m = bits & 1023;
    sign * if e == 0 {
        (m as f32) * 2.0f32.powi(-24)
    } else if e == 31 {
        f32::INFINITY
    } else {
        (1.0 + m as f32 / 1024.0) * 2.0f32.powi(i32::from(e) - 15)
    }
}
