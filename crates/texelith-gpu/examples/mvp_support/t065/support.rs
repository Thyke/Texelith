use std::{
    sync::Arc,
    time::{Duration, Instant},
};
use texelith_core::*;
use texelith_gpu::{
    CompletedMaterial, CompletedPreview, GpuConfig, GpuWorker, JobId, MaterialRequest,
    PreviewRequest, Quality, TypedSourcePixels, TypedSourceUpload, pbr::PbrSettings,
    viewport_shared::PreviewContent,
};
pub type Result<T> = std::result::Result<T, String>;
pub const BUDGET: u64 = 1_073_741_824;
pub struct Gpu {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
}
#[path = "readback.rs"]
mod readback;
pub fn require(ok: bool, message: &str) -> Result<()> {
    if ok { Ok(()) } else { Err(message.into()) }
}
pub fn wait<T>(mut next: impl FnMut() -> Result<Option<T>>) -> Result<T> {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        if let Some(v) = next()? {
            return Ok(v);
        }
        if Instant::now() >= deadline {
            return Err("T-065 worker timeout".into());
        }
        std::thread::sleep(Duration::from_millis(1));
    }
}
pub fn no_error(worker: &GpuWorker) -> Result<()> {
    match worker.try_event() {
        None => Ok(()),
        Some(e) => Err(format!("worker error: {e:?}")),
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
}
pub fn upload(
    worker: &GpuWorker,
    pixels: TypedSourcePixels,
    extent: Extent,
) -> Result<SourceDescriptor> {
    let descriptor = pixels.descriptor(extent).map_err(|e| e.to_string())?;
    worker
        .install_typed_source(TypedSourceUpload {
            descriptor: descriptor.clone(),
            pixels,
        })
        .map_err(|e| e.to_string())?;
    let event = wait(|| Ok(worker.try_event()))?;
    require(
        matches!(event,texelith_gpu::WorkerEvent::SourceReady {content_hash} if content_hash==descriptor.content_hash),
        "source acknowledgement mismatch",
    )?;
    Ok(descriptor)
}
/// Original fixtures: constant color for analytic checks, small ochre checker
/// for captures. No image assets, IO parsers or another leaf's fixture generator.
pub fn material(
    worker: &GpuWorker,
    job: u64,
    normal: [u8; 3],
    convention: NormalConvention,
    checker: bool,
    with_ao: bool,
) -> Result<Arc<CompletedMaterial>> {
    let extent = Extent::new(32, 32).map_err(|e| e.to_string())?;
    let mut rgba = Vec::new();
    for y in 0..32 {
        for x in 0..32 {
            rgba.extend(if checker && (x / 4 + y / 4) % 2 == 0 {
                [182, 119, 54, 255]
            } else {
                [128, 96, 64, 255]
            });
        }
    }
    let color = upload(
        worker,
        TypedSourcePixels::Rgba8 {
            pixels: rgba.into(),
            interpretation: InputInterpretation::SrgbPhoto,
        },
        extent,
    )?;
    let normals = upload(
        worker,
        TypedSourcePixels::NormalRgb8 {
            pixels: normal.repeat(32 * 32).into(),
            convention,
        },
        extent,
    )?;
    let out = |node, port| OutputRef {
        node: NodeId(node),
        port,
    };
    let color_edge = || {
        vec![InputEdge {
            port: InputPort::Color,
            source: out(1, OutputPort::Albedo),
        }]
    };
    let mut graph = vec![
        MaterialNode {
            id: NodeId(1),
            inputs: vec![],
            params: MaterialNodeParams::ColorSource(color),
        },
        MaterialNode {
            id: NodeId(2),
            inputs: vec![],
            params: MaterialNodeParams::NormalSource {
                source: normals,
                convention,
            },
        },
        MaterialNode {
            id: NodeId(3),
            inputs: color_edge(),
            params: MaterialNodeParams::Roughness {
                curve: vec![[0., 0.45], [1., 0.45]],
                base: 0.45,
                invert: false,
            },
        },
        MaterialNode {
            id: NodeId(4),
            inputs: color_edge(),
            params: MaterialNodeParams::Metallic {
                picked_color: [1.; 3],
                radius: 0.,
                threshold: 0.5,
                feather: 0.,
                // Subtracting the explicit default-one mask yields dielectric
                // zero; Replace would select the default mask (all metal).
                mask_mode: MaskMode::Subtract,
            },
        },
    ];
    let mut outputs = MaterialOutputs::from([
        (MapKind::Albedo, out(1, OutputPort::Albedo)),
        (MapKind::Normal, out(2, OutputPort::Normal)),
        (MapKind::Roughness, out(3, OutputPort::Roughness)),
        (MapKind::Metallic, out(4, OutputPort::Metallic)),
    ]);
    if with_ao {
        let height = upload(
            worker,
            TypedSourcePixels::HeightF32(
                (0..1024)
                    .map(|i| if (i % 32 + i / 32) % 2 == 0 { 1.0 } else { 0.0 })
                    .collect::<Vec<_>>()
                    .into(),
            ),
            extent,
        )?;
        graph.push(MaterialNode {
            id: NodeId(5),
            inputs: vec![],
            params: MaterialNodeParams::HeightSource(height),
        });
        graph.push(MaterialNode {
            id: NodeId(6),
            inputs: vec![InputEdge {
                port: InputPort::Height,
                source: out(5, OutputPort::Height),
            }],
            params: MaterialNodeParams::HorizonAo {
                directions: 8,
                steps: 8,
                radius_source_px: 8.,
                bias: 0.,
                boundary: Boundary::Repeat,
            },
        });
        outputs.insert(MapKind::Ao, out(6, OutputPort::Ao));
    }
    let graph = Graph::try_from(MaterialGraphDto {
        schema_version: 2,
        graph,
        outputs: outputs.clone(),
        selected_output: out(1, OutputPort::Albedo),
    })
    .map_err(|e| e.to_string())?;
    worker
        .request_material(MaterialRequest {
            job_id: JobId(job),
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
    let bundle = wait(|| {
        no_error(worker)?;
        Ok(worker.try_latest_material())
    })?;
    Ok(Arc::new(bundle))
}
pub fn request(
    material: Arc<CompletedMaterial>,
    id: u64,
    extent: Extent,
    mesh: PreviewMesh,
    settings: PbrSettings,
) -> PreviewRequest {
    PreviewRequest {
        job_id: material.job_id,
        generation: material.generation,
        request_id: id,
        material,
        extent,
        view: ViewState {
            mesh,
            light: Light {
                direction: [0.0, 0.0, 1.0],
                intensity: 1.0,
                ..Default::default()
            },
            ..Default::default()
        },
        content: PreviewContent::Pbr(settings),
    }
}
pub fn render(worker: &GpuWorker, request: PreviewRequest) -> Result<CompletedPreview> {
    let id = request.request_id;
    worker.request_preview(request).map_err(|e| e.to_string())?;
    let frame = wait(|| {
        no_error(worker)?;
        Ok(worker.try_latest_preview())
    })?;
    require(frame.request_id == id, "stale preview publication")?;
    Ok(frame)
}
pub fn center(bytes: &[u8], extent: Extent) -> [u8; 4] {
    let i = ((extent.height / 2 * extent.width + extent.width / 2) * 4) as usize;
    bytes[i..i + 4].try_into().unwrap()
}
pub fn diagnostics(worker: &GpuWorker) -> Result<texelith_gpu::WorkerDiagnostics> {
    let d = worker.diagnostics();
    require(
        d.validation_errors == 0 && d.stale_publications == 0,
        "validation/stale publication error",
    )?;
    require(
        d.peak_bytes > 0 && d.peak_bytes <= BUDGET,
        "shared ledger exceeds 1 GiB",
    )?;
    require(
        d.peak_pending <= 1 && d.peak_in_flight <= 1,
        "unbounded worker submission/mailbox",
    )?;
    Ok(d)
}
