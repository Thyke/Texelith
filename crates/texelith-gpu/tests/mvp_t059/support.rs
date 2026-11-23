use super::gpu::Gpu;
use std::{
    collections::BTreeMap,
    sync::Arc,
    time::{Duration, Instant},
};
use texelith_core::{
    Boundary, EvaluationSnapshot, Extent, Graph, InputEdge, InputInterpretation, InputPort,
    MapKind, MaterialGraphDto, MaterialNode, MaterialNodeParams as P, NodeId, OutputPort,
    OutputRef, SeamlessBase, SeamlessMethod, SourceDescriptor,
};
use texelith_gpu::{
    GpuWorker, JobId, MaterialRequest, Quality, TypedSourcePixels, TypedSourceUpload, WorkerEvent,
};

#[derive(Clone, Copy)]
pub struct Controls {
    pub base: Base,
    pub strength: f32,
    pub boundary: Boundary,
}
#[derive(Clone, Copy, Debug)]
pub enum Base {
    Offset(f32),
    Splat { seed: u32, grid: u32, overlap: f32 },
}
impl Base {
    pub fn core(self) -> SeamlessBase {
        match self {
            Self::Offset(overlap) => SeamlessBase::OffsetBlend { overlap },
            Self::Splat {
                seed,
                grid,
                overlap,
            } => SeamlessBase::Splat {
                seed,
                grid,
                overlap,
            },
        }
    }
    pub fn method(self) -> SeamlessMethod {
        match self {
            Self::Offset(overlap) => SeamlessMethod::OffsetBlend { overlap },
            Self::Splat {
                seed,
                grid,
                overlap,
            } => SeamlessMethod::Splat {
                seed,
                grid,
                overlap,
            },
        }
    }
}
pub const BASES: [Base; 2] = [
    Base::Offset(0.25),
    Base::Splat {
        seed: 59,
        grid: 4,
        overlap: 0.25,
    },
];
impl Default for Controls {
    fn default() -> Self {
        Self {
            base: BASES[0],
            strength: 1.0,
            boundary: Boundary::Repeat,
        }
    }
}
impl Controls {
    pub fn method(self) -> SeamlessMethod {
        SeamlessMethod::Histogram {
            base: self.base.core(),
            strength: self.strength,
        }
    }
}

pub fn request(
    sources: &[SourceDescriptor],
    serial: u64,
    controls: Option<Controls>,
) -> MaterialRequest {
    request_method(sources, serial, controls.map(|c| (c.method(), c.boundary)))
}

pub fn request_method(
    sources: &[SourceDescriptor],
    serial: u64,
    controls: Option<(SeamlessMethod, Boundary)>,
) -> MaterialRequest {
    let mut graph = Vec::new();
    let mut outputs = BTreeMap::new();
    for (i, source) in sources.iter().enumerate() {
        let color = source.interpretation == InputInterpretation::SrgbPhoto;
        let port = if color {
            OutputPort::Albedo
        } else {
            OutputPort::Height
        };
        let input = OutputRef {
            node: NodeId(i as u64 * 2 + 1),
            port,
        };
        graph.push(MaterialNode {
            id: input.node,
            inputs: vec![],
            params: if color {
                P::ColorSource(source.clone())
            } else {
                P::HeightSource(source.clone())
            },
        });
        let output = if let Some((method, boundary)) = &controls {
            let id = NodeId(input.node.0 + 1);
            graph.push(MaterialNode {
                id,
                inputs: vec![InputEdge {
                    port: if color {
                        InputPort::Color
                    } else {
                        InputPort::Height
                    },
                    source: input,
                }],
                params: P::Seamless {
                    method: method.clone(),
                    boundary: *boundary,
                },
            });
            OutputRef { node: id, port }
        } else {
            input
        };
        assert!(outputs.insert(port.map_kind(), output).is_none());
    }
    let dto = MaterialGraphDto {
        schema_version: 2,
        graph,
        selected_output: *outputs.values().next().unwrap(),
        outputs,
    };
    MaterialRequest {
        job_id: JobId(59),
        request_id: serial,
        extent: sources[0].extent,
        quality: Quality::Full,
        outputs: dto.outputs.clone(),
        snapshot: Arc::new(EvaluationSnapshot {
            generation: serial,
            graph: Graph::try_from(dto).unwrap(),
        }),
    }
}

pub fn wait<T>(mut next: impl FnMut() -> Option<T>) -> T {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        if let Some(value) = next() {
            return value;
        }
        assert!(Instant::now() < deadline, "T-059 worker timed out");
        std::thread::sleep(Duration::from_millis(1));
    }
}

pub fn install(worker: &GpuWorker, extent: Extent, pixels: TypedSourcePixels) -> SourceDescriptor {
    let descriptor = pixels.descriptor(extent).unwrap();
    worker
        .install_typed_source(TypedSourceUpload {
            descriptor: descriptor.clone(),
            pixels,
        })
        .unwrap();
    let event = wait(|| worker.try_event());
    assert!(
        matches!(event, WorkerEvent::SourceReady { content_hash } if content_hash == descriptor.content_hash),
        "source installation: {event:?}"
    );
    descriptor
}
pub fn height(worker: &GpuWorker, extent: Extent, values: Vec<f32>) -> SourceDescriptor {
    install(worker, extent, TypedSourcePixels::HeightF32(values.into()))
}

pub fn evaluate(
    gpu: &Gpu,
    worker: &GpuWorker,
    r: MaterialRequest,
) -> BTreeMap<MapKind, Vec<[f64; 4]>> {
    texelith_gpu::MaterialPlan::new(&r, 1_073_741_824, Default::default())
        .expect("T-059 histogram material must be supported by the planner");
    let outputs: Vec<_> = r.outputs.keys().copied().collect();
    let identity = (
        r.job_id,
        r.request_id,
        r.snapshot.generation,
        r.extent,
        r.quality,
    );
    worker.request_material(r).unwrap();
    let result = wait(|| {
        if let Some(event) = worker.try_event() {
            panic!("T-059 histogram failed: {event:?}");
        }
        worker.try_latest_material()
    });
    assert_eq!(
        (
            result.job_id,
            result.request_id,
            result.generation,
            result.extent,
            result.quality
        ),
        identity
    );
    assert_eq!(result.outputs.keys().copied().collect::<Vec<_>>(), outputs);
    let values = result
        .outputs
        .iter()
        .map(|(kind, lease)| {
            let t = lease.texture();
            assert_eq!(
                (t.width(), t.height()),
                (result.extent.width, result.extent.height)
            );
            assert_eq!(
                t.format(),
                if *kind == MapKind::Albedo {
                    wgpu::TextureFormat::Rgba16Float
                } else {
                    wgpu::TextureFormat::R32Float
                }
            );
            let values = gpu.read(t);
            assert_eq!(
                values.len(),
                (result.extent.width * result.extent.height) as usize
            );
            assert!(values.iter().flatten().all(|v| v.is_finite()));
            (*kind, values)
        })
        .collect();
    // Diagnostics is a lifetime high-water mark; a prior larger request must
    // not be compared with this request's smaller live-resource estimate.
    assert!(worker.diagnostics().peak_bytes <= 1_073_741_824);
    assert_eq!(worker.diagnostics().validation_errors, 0);
    values
}

pub fn compare(actual: impl IntoIterator<Item = f64>, expected: &[f64], tolerance: f64) {
    let actual: Vec<_> = actual.into_iter().collect();
    assert!(!actual.is_empty());
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite() && b.is_finite() && (a - b).abs() <= tolerance,
            "pixel {i}: {a} != {b}; tolerance {tolerance}"
        );
    }
}
