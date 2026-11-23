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
#[path = "readback.rs"]
mod readback;
pub fn require(ok: bool, message: &str) -> Result<()> {
    if ok { Ok(()) } else { Err(message.into()) }
}
pub fn wait<T>(mut next: impl FnMut() -> Result<Option<T>>) -> Result<T> {
    let start = Instant::now();
    loop {
        if let Some(v) = next()? {
            return Ok(v);
        }
        if start.elapsed() > Duration::from_secs(60) {
            return Err("T-068 worker timeout".into());
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
}
fn no_error(worker: &GpuWorker) -> Result<()> {
    if let Some(e) = worker.try_event() {
        Err(format!("worker error: {e:?}"))
    } else {
        Ok(())
    }
}
fn upload(
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
        matches!(event, WorkerEvent::SourceReady { content_hash } if content_hash == descriptor.content_hash),
        "source acknowledgement mismatch",
    )?;
    Ok(descriptor)
}
/// Original bounded brick/ramp inputs. Scalar maps are produced by their typed
/// graph nodes; the POM harness never depends on another leaf's test generator.
pub fn material(
    worker: &GpuWorker,
    job: u64,
    constant_height: Option<f32>,
    convention: NormalConvention,
) -> Result<Arc<CompletedMaterial>> {
    let extent = Extent::new(64, 64).map_err(|e| e.to_string())?;
    let mut colors = Vec::new();
    let mut normals = Vec::new();
    let mut heights = Vec::new();
    for y in 0..64 {
        for x in 0..64 {
            let mortar = (x + (y / 16 % 2) * 8) % 32 < 3 || y % 16 < 3;
            colors.extend(if mortar {
                [48, 44, 39, 255]
            } else {
                [170 + (x % 16) as u8 * 3, 102, 50, 255]
            });
            let ny: u8 = if x % 32 < 16 { 156 } else { 99 };
            normals.extend([
                if y % 16 < 8 { 145 } else { 110 },
                if convention == NormalConvention::DirectX {
                    255 - ny
                } else {
                    ny
                },
                250,
            ]);
            heights.push(constant_height.unwrap_or(if mortar { 0.12 } else { 0.85 }));
        }
    }
    let color = upload(
        worker,
        TypedSourcePixels::Rgba8 {
            pixels: colors.into(),
            interpretation: InputInterpretation::SrgbPhoto,
        },
        extent,
    )?;
    let normal = upload(
        worker,
        TypedSourcePixels::NormalRgb8 {
            pixels: normals.into(),
            convention,
        },
        extent,
    )?;
    let height = upload(worker, TypedSourcePixels::HeightF32(heights.into()), extent)?;
    let out = |id, port| OutputRef {
        node: NodeId(id),
        port,
    };
    let color_edge = || {
        vec![InputEdge {
            port: InputPort::Color,
            source: out(1, OutputPort::Albedo),
        }]
    };
    let h_edge = || {
        vec![InputEdge {
            port: InputPort::Height,
            source: out(3, OutputPort::Height),
        }]
    };
    let outputs = MaterialOutputs::from([
        (MapKind::Albedo, out(1, OutputPort::Albedo)),
        (MapKind::Normal, out(2, OutputPort::Normal)),
        (MapKind::Height, out(3, OutputPort::Height)),
        (MapKind::Roughness, out(4, OutputPort::Roughness)),
        (MapKind::Metallic, out(5, OutputPort::Metallic)),
        (MapKind::Ao, out(6, OutputPort::Ao)),
    ]);
    let graph = Graph::try_from(MaterialGraphDto {
        schema_version: 2,
        graph: vec![
            MaterialNode {
                id: NodeId(1),
                inputs: vec![],
                params: MaterialNodeParams::ColorSource(color),
            },
            MaterialNode {
                id: NodeId(2),
                inputs: vec![],
                params: MaterialNodeParams::NormalSource {
                    source: normal,
                    convention,
                },
            },
            MaterialNode {
                id: NodeId(3),
                inputs: vec![],
                params: MaterialNodeParams::HeightSource(height),
            },
            MaterialNode {
                id: NodeId(4),
                inputs: color_edge(),
                params: MaterialNodeParams::Roughness {
                    curve: vec![[0., 0.15], [1., 0.95]],
                    base: 0.5,
                    invert: false,
                },
            },
            MaterialNode {
                id: NodeId(5),
                inputs: color_edge(),
                params: MaterialNodeParams::Metallic {
                    picked_color: [0.5, 0.13, 0.03],
                    radius: 0.5,
                    threshold: 0.3,
                    feather: 0.2,
                    mask_mode: MaskMode::Subtract,
                },
            },
            MaterialNode {
                id: NodeId(6),
                inputs: h_edge(),
                params: MaterialNodeParams::HorizonAo {
                    directions: 8,
                    steps: 8,
                    radius_source_px: 4.,
                    bias: 0.01,
                    boundary: Boundary::Repeat,
                },
            },
        ],
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
    wait(|| {
        no_error(worker)?;
        Ok(worker.try_latest_material().map(Arc::new))
    })
}
pub fn request(
    material: Arc<CompletedMaterial>,
    id: u64,
    extent: Extent,
    scale: f32,
    ibl: bool,
) -> PreviewRequest {
    let settings = pbr::PbrSettings {
        surface: pom::SurfaceSettings {
            midlevel: 0.5,
            boundary: Boundary::Repeat,
        },
        ..Default::default()
    };
    PreviewRequest {
        job_id: material.job_id,
        generation: material.generation,
        request_id: id,
        material,
        extent,
        view: ViewState {
            mesh: PreviewMesh::Plane,
            camera: Camera {
                yaw: 0.8,
                pitch: 0.3,
                distance: 2.8,
                ..Default::default()
            },
            pom: Pom {
                enabled: true,
                scale,
            },
            ..Default::default()
        },
        content: if ibl {
            viewport_shared::PreviewContent::Ibl {
                settings,
                environment: Arc::new(ibl::Environment::procedural()),
                exposure: 0.0,
            }
        } else {
            viewport_shared::PreviewContent::Pbr(settings)
        },
    }
}
pub fn render(worker: &GpuWorker, request: PreviewRequest) -> Result<CompletedPreview> {
    let id = request.request_id;
    worker.request_preview(request).map_err(|e| e.to_string())?;
    let frame = wait(|| {
        no_error(worker)?;
        Ok(worker.try_latest_preview())
    })?;
    require(frame.request_id == id, "stale POM publication")?;
    Ok(frame)
}
pub fn diagnostics(worker: &GpuWorker) -> Result<WorkerDiagnostics> {
    let d = worker.diagnostics();
    require(
        d.validation_errors == 0 && d.stale_publications == 0,
        "POM validation/stale publication error",
    )?;
    require(
        d.peak_bytes > 0 && d.peak_bytes <= BUDGET,
        "POM shared ledger exceeds budget",
    )?;
    require(
        d.peak_pending <= 1 && d.peak_in_flight <= 1,
        "unbounded POM mailbox/submission",
    )?;
    Ok(d)
}
