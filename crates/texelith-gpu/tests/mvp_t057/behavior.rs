use super::{
    gpu::{Gpu, with_gpu},
    reference,
};
use std::{
    collections::BTreeMap,
    sync::Arc,
    time::{Duration, Instant},
};
use texelith_core::{
    Boundary, EvaluationSnapshot, Extent, Graph, InputEdge, InputInterpretation, InputPort,
    MapKind, MaterialGraphDto, MaterialNode, MaterialNodeParams as P, NodeId, OutputPort,
    OutputRef, SeamlessMethod, SourceDescriptor,
};
use texelith_gpu::{
    CompletedMaterial, GpuConfig, GpuWorker, JobId, MaterialRequest, Quality, TypedSourcePixels,
    TypedSourceUpload, WorkerEvent,
};

pub fn wait<T>(mut next: impl FnMut() -> Option<T>) -> T {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        if let Some(value) = next() {
            return value;
        }
        assert!(Instant::now() < deadline, "T-057 worker timed out");
        std::thread::sleep(Duration::from_millis(1));
    }
}

pub fn request(
    sources: &[SourceDescriptor],
    serial: u64,
    overlap: Option<f32>,
    boundary: Boundary,
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
        let selected = if let Some(overlap) = overlap {
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
                    method: SeamlessMethod::OffsetBlend { overlap },
                    boundary,
                },
            });
            OutputRef { node: id, port }
        } else {
            input
        };
        assert!(outputs.insert(port.map_kind(), selected).is_none());
    }
    let extent = sources[0].extent;
    let dto = MaterialGraphDto {
        schema_version: 2,
        graph,
        selected_output: *outputs.values().next().unwrap(),
        outputs,
    };
    MaterialRequest {
        job_id: JobId(57),
        request_id: serial,
        outputs: dto.outputs.clone(),
        extent,
        quality: Quality::Full,
        snapshot: Arc::new(EvaluationSnapshot {
            generation: serial,
            graph: Graph::try_from(dto).unwrap(),
        }),
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
        "{event:?}"
    );
    descriptor
}

pub fn height(worker: &GpuWorker, extent: Extent, values: Vec<f32>) -> SourceDescriptor {
    install(worker, extent, TypedSourcePixels::HeightF32(values.into()))
}

pub fn evaluate(
    gpu: &Gpu,
    worker: &GpuWorker,
    request: MaterialRequest,
) -> BTreeMap<MapKind, Vec<[f64; 4]>> {
    let serial = request.request_id;
    let outputs: Vec<_> = request.outputs.keys().copied().collect();
    let extent = request.extent;
    worker.request_material(request).unwrap();
    let completed: CompletedMaterial = wait(|| {
        if let Some(error) = worker.try_event() {
            panic!("seamless evaluation failed: {error:?}");
        }
        worker.try_latest_material()
    });
    assert_eq!(completed.job_id, JobId(57));
    assert_eq!(completed.request_id, serial);
    assert_eq!(completed.generation, serial);
    assert_eq!(completed.extent, extent);
    assert_eq!(
        completed.outputs.keys().copied().collect::<Vec<_>>(),
        outputs
    );
    assert_eq!(completed.quality, Quality::Full);
    let values = completed
        .outputs
        .iter()
        .map(|(kind, lease)| {
            assert_eq!(
                lease.texture().format(),
                if *kind == MapKind::Albedo {
                    wgpu::TextureFormat::Rgba16Float
                } else {
                    wgpu::TextureFormat::R32Float
                }
            );
            let pixels = gpu.read(lease.texture());
            assert_eq!(pixels.len(), (extent.width * extent.height) as usize);
            assert!(pixels.iter().flatten().all(|v| v.is_finite()));
            (*kind, pixels)
        })
        .collect();
    assert_eq!(worker.diagnostics().validation_errors, 0);
    values
}

pub fn compare(actual: impl IntoIterator<Item = f64>, expected: &[f64], tolerance: f64) {
    let actual: Vec<_> = actual.into_iter().collect();
    assert_eq!(actual.len(), expected.len());
    assert!(!actual.is_empty());
    for (i, (a, b)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite() && b.is_finite() && (a - b).abs() <= tolerance,
            "pixel {i}: {a} != {b}, tolerance {tolerance}"
        );
    }
}

#[test]
fn scalar_matches_original_oracle_for_constants_ramps_impulses_npot_singletons_and_overlap_bounds()
{
    with_gpu(|gpu| {
        let worker =
            GpuWorker::spawn(gpu.device.clone(), gpu.queue.clone(), GpuConfig::default()).unwrap();
        let mut serial = 0;
        for (w, h, kind) in [
            (1, 1, "constant"),
            (1, 17, "ramp"),
            (17, 1, "seeded"),
            (7, 5, "constant"),
            (7, 5, "ramp"),
            (17, 9, "impulse"),
            (257, 193, "seeded"),
        ] {
            let e = Extent::new(w, h).unwrap();
            let input = reference::scalar_fixture(e, kind);
            let source = height(&worker, e, input.clone());
            let input: Vec<_> = input.into_iter().map(f64::from).collect();
            for overlap in [0.1f32, 0.25, 0.5] {
                for boundary in [Boundary::Repeat, Boundary::Clamp] {
                    serial += 1;
                    let actual = evaluate(
                        gpu,
                        &worker,
                        request(
                            std::slice::from_ref(&source),
                            serial,
                            Some(overlap),
                            boundary,
                        ),
                    );
                    let expected = reference::render(e, &input, f64::from(overlap), boundary);
                    compare(
                        actual[&MapKind::Height].iter().map(|p| p[0]),
                        &expected,
                        2e-5,
                    );
                    let lo = input.iter().copied().fold(f64::INFINITY, f64::min);
                    let hi = input.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    assert!(
                        actual[&MapKind::Height]
                            .iter()
                            .all(|p| p[0] >= lo - 2e-5 && p[0] <= hi + 2e-5)
                    );
                }
            }
        }
        worker.join().unwrap();
    });
}

#[test]
fn odd_extent_uses_continuous_half_offsets_and_does_not_duplicate_opposite_pixels() {
    with_gpu(|gpu| {
        let worker =
            GpuWorker::spawn(gpu.device.clone(), gpu.queue.clone(), GpuConfig::default()).unwrap();
        let e = Extent::new(7, 5).unwrap();
        let input = reference::scalar_fixture(e, "ramp");
        let source = height(&worker, e, input.clone());
        let actual = evaluate(
            gpu,
            &worker,
            request(&[source], 1, Some(0.25), Boundary::Repeat),
        );
        let actual: Vec<_> = actual[&MapKind::Height].iter().map(|p| p[0]).collect();
        let input: Vec<_> = input.into_iter().map(f64::from).collect();
        compare(
            actual.iter().copied(),
            &reference::render(e, &input, 0.25, Boundary::Repeat),
            2e-5,
        );
        assert!(
            (actual[0] - actual[6]).abs() > 1e-3,
            "periodic samples are pixel centers, not duplicated edges"
        );
        assert!((actual[0] - actual[28]).abs() > 1e-3);
        worker.join().unwrap();
    });
}

#[test]
fn finite_height_range_and_sub_u16_precision_survive_without_remapping_or_transfer() {
    with_gpu(|gpu| {
        let worker =
            GpuWorker::spawn(gpu.device.clone(), gpu.queue.clone(), GpuConfig::default()).unwrap();
        let e = Extent::new(9, 7).unwrap();
        for (i, value) in [-3.25f32, 1.75, 0.500001].into_iter().enumerate() {
            let source = height(&worker, e, vec![value; 63]);
            let actual = evaluate(
                gpu,
                &worker,
                request(&[source], i as u64 + 1, Some(0.25), Boundary::Repeat),
            );
            compare(
                actual[&MapKind::Height].iter().map(|p| p[0]),
                &vec![f64::from(value); 63],
                if (0.0..=1.0).contains(&value) {
                    3e-7
                } else {
                    2e-5
                },
            );
        }
        worker.join().unwrap();
    });
}

#[test]
fn albedo_blends_in_linear_rgb_preserves_alpha_one_and_shares_spatial_weights_with_height() {
    with_gpu(|gpu| {
        let worker =
            GpuWorker::spawn(gpu.device.clone(), gpu.queue.clone(), GpuConfig::default()).unwrap();
        let e = Extent::new(31, 19).unwrap();
        let bytes: Vec<u8> = (0..31 * 19)
            .flat_map(|i| {
                let codes = [0, 10, 11, 64, 128, 192, 255];
                [
                    codes[i % 7],
                    codes[(i * 3 + 1) % 7],
                    codes[(i * 5 + 2) % 7],
                    (i % 256) as u8,
                ]
            })
            .collect();
        let channels: [Vec<f64>; 3] = std::array::from_fn(|c| {
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|p| reference::half(reference::linear(p[c])))
                .collect()
        });
        let scalar: Vec<f32> = channels[0].iter().map(|v| *v as f32).collect();
        let color = install(
            &worker,
            e,
            TypedSourcePixels::Rgba8 {
                pixels: bytes.into(),
                interpretation: InputInterpretation::SrgbPhoto,
            },
        );
        let scalar = height(&worker, e, scalar);
        let mut previous = None;
        for (i, overlap) in [0.1f32, 0.25, 0.5].into_iter().enumerate() {
            let actual = evaluate(
                gpu,
                &worker,
                request(
                    &[color.clone(), scalar.clone()],
                    i as u64 + 1,
                    Some(overlap),
                    Boundary::Repeat,
                ),
            );
            for (c, channel) in channels.iter().enumerate() {
                let expected = reference::render(e, channel, f64::from(overlap), Boundary::Repeat);
                // One binary16 rounding at the output; absolute tolerance stays below one ULP near 1.
                compare(
                    actual[&MapKind::Albedo].iter().map(|p| p[c]),
                    &expected,
                    5e-4,
                );
            }
            compare(
                actual[&MapKind::Albedo].iter().map(|p| p[0]),
                &actual[&MapKind::Height]
                    .iter()
                    .map(|p| p[0])
                    .collect::<Vec<_>>(),
                5e-4,
            );
            assert!(actual[&MapKind::Albedo].iter().all(|p| p[3] == 1.0));
            if let Some(previous) = previous {
                assert_ne!(
                    actual[&MapKind::Height],
                    previous,
                    "overlap edit must change result"
                );
            }
            previous = Some(actual[&MapKind::Height].clone());
        }
        worker.join().unwrap();
    });
}

#[test]
fn constant_albedo_is_invariant_on_npot_and_singleton_axes() {
    with_gpu(|gpu| {
        let worker =
            GpuWorker::spawn(gpu.device.clone(), gpu.queue.clone(), GpuConfig::default()).unwrap();
        let expected = [128, 64, 255].map(|c| reference::half(reference::linear(c)));
        let mut serial = 0;
        for (w, h) in [(1, 1), (1, 17), (17, 1), (257, 193)] {
            let e = Extent::new(w, h).unwrap();
            let source = install(
                &worker,
                e,
                TypedSourcePixels::Rgba8 {
                    pixels: [128, 64, 255, 13].repeat((w * h) as usize).into(),
                    interpretation: InputInterpretation::SrgbPhoto,
                },
            );
            for overlap in [0.1, 0.25, 0.5] {
                serial += 1;
                let actual = evaluate(
                    gpu,
                    &worker,
                    request(
                        std::slice::from_ref(&source),
                        serial,
                        Some(overlap),
                        Boundary::Repeat,
                    ),
                );
                for pixel in &actual[&MapKind::Albedo] {
                    for c in 0..3 {
                        assert!((pixel[c] - expected[c]).abs() <= 2e-5);
                    }
                    assert_eq!(pixel[3], 1.0);
                }
            }
        }
        worker.join().unwrap();
    });
}

#[test]
fn fixed_discontinuous_noise_reduces_seam_rms_by_half_and_retains_half_the_interior_rms() {
    with_gpu(|gpu| {
        let worker =
            GpuWorker::spawn(gpu.device.clone(), gpu.queue.clone(), GpuConfig::default()).unwrap();
        for (i, (w, h)) in [(256, 256), (257, 193)].into_iter().enumerate() {
            let e = Extent::new(w, h).unwrap();
            let input = reference::scalar_fixture(e, "discontinuous_noise");
            let source = height(&worker, e, input.clone());
            let input: Vec<_> = input.into_iter().map(f64::from).collect();
            let result = evaluate(
                gpu,
                &worker,
                request(&[source], i as u64 + 1, Some(0.25), Boundary::Repeat),
            );
            let result: Vec<_> = result[&MapKind::Height].iter().map(|p| p[0]).collect();
            let (before, interior_before) = reference::seam_rms(e, &input);
            let (after, interior_after) = reference::seam_rms(e, &result);
            assert!(
                before > 0.01 && interior_before > 0.01,
                "degenerate fixture"
            );
            assert!(after <= before * 0.5, "seam RMS {before} -> {after}");
            assert!(
                interior_after >= interior_before * 0.5,
                "interior RMS {interior_before} -> {interior_after}"
            );
        }
        worker.join().unwrap();
    });
}

#[test]
fn structured_input_records_composition_change_and_disabled_graph_is_lossless() {
    with_gpu(|gpu| {
        let worker =
            GpuWorker::spawn(gpu.device.clone(), gpu.queue.clone(), GpuConfig::default()).unwrap();
        let e = Extent::new(64, 48).unwrap();
        let input = reference::scalar_fixture(e, "structured");
        let source = height(&worker, e, input.clone());
        let original = evaluate(
            gpu,
            &worker,
            request(std::slice::from_ref(&source), 1, None, Boundary::Repeat),
        );
        assert_eq!(
            original[&MapKind::Height]
                .iter()
                .map(|p| p[0])
                .collect::<Vec<_>>(),
            input.iter().copied().map(f64::from).collect::<Vec<_>>()
        );
        let blended = evaluate(
            gpu,
            &worker,
            request(
                std::slice::from_ref(&source),
                2,
                Some(0.25),
                Boundary::Repeat,
            ),
        );
        let difference = blended[&MapKind::Height]
            .iter()
            .zip(&input)
            .map(|(a, b)| (a[0] - f64::from(*b)).abs())
            .sum::<f64>()
            / input.len() as f64;
        assert!(
            difference > 0.03,
            "structured fixture must show ghosting/composition tradeoff"
        );
        let restored = evaluate(gpu, &worker, request(&[source], 3, None, Boundary::Repeat));
        assert_eq!(original, restored);
        worker.join().unwrap();
    });
}

#[test]
fn repeated_inputs_are_bitwise_deterministic_across_cold_workers_and_warm_cache() {
    with_gpu(|gpu| {
        let e = Extent::new(33, 19).unwrap();
        let mut baseline = None;
        for _ in 0..2 {
            let worker =
                GpuWorker::spawn(gpu.device.clone(), gpu.queue.clone(), GpuConfig::default())
                    .unwrap();
            let source = height(&worker, e, reference::scalar_fixture(e, "seeded"));
            let a = evaluate(
                gpu,
                &worker,
                request(
                    std::slice::from_ref(&source),
                    1,
                    Some(0.25),
                    Boundary::Repeat,
                ),
            );
            let stats = worker.diagnostics();
            let b = evaluate(
                gpu,
                &worker,
                request(&[source], 2, Some(0.25), Boundary::Repeat),
            );
            assert_eq!(a, b);
            assert_eq!(
                stats.submitted,
                worker.diagnostics().submitted,
                "identical content should reuse cache across generations"
            );
            assert!(worker.diagnostics().cache_hits > stats.cache_hits);
            let bits: Vec<_> = a[&MapKind::Height]
                .iter()
                .map(|p| (p[0] as f32).to_bits())
                .collect();
            if let Some(previous) = &baseline {
                assert_eq!(&bits, previous);
            }
            baseline = Some(bits);
            worker.join().unwrap();
        }
    });
}
