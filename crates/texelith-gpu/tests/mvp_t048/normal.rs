use super::{
    gpu::{Gpu, decode, with_gpu},
    reference as cpu,
};
use std::{
    sync::Arc,
    time::{Duration, Instant},
};
use texelith_core::{
    Boundary, EvaluationSnapshot, Extent, Graph, InputEdge, InputPort, MapKind, MaterialGraphDto,
    MaterialNode, MaterialNodeParams as P, NodeId, NormalConvention, NormalFilter, OutputPort,
    OutputRef, SourceDescriptor,
};
use texelith_gpu::{
    GpuConfig, GpuWorker, JobId, MaterialRequest, Pass, Quality, TypedSourcePixels,
    TypedSourceUpload, WorkerEvent,
};

#[derive(Clone)]
pub struct Controls {
    pub strength: f32,
    pub filter: NormalFilter,
    pub weights: Vec<f32>,
    pub boundary: Boundary,
    pub convention: NormalConvention,
}
impl Default for Controls {
    fn default() -> Self {
        Self {
            strength: 1.0,
            filter: NormalFilter::Sobel,
            weights: vec![1.0],
            boundary: Boundary::Repeat,
            convention: NormalConvention::OpenGl,
        }
    }
}
pub fn request(source: &SourceDescriptor, serial: u64, c: &Controls) -> MaterialRequest {
    let height = OutputRef {
        node: NodeId(1),
        port: OutputPort::Height,
    };
    let normal = OutputRef {
        node: NodeId(2),
        port: OutputPort::Normal,
    };
    let dto = MaterialGraphDto {
        schema_version: 2,
        graph: vec![
            MaterialNode {
                id: height.node,
                inputs: vec![],
                params: P::HeightSource(source.clone()),
            },
            MaterialNode {
                id: normal.node,
                inputs: vec![InputEdge {
                    port: InputPort::Height,
                    source: height,
                }],
                params: P::NormalFromHeight {
                    strength: c.strength,
                    filter: c.filter,
                    scale_weights: c.weights.clone(),
                    boundary: c.boundary,
                    convention: c.convention,
                },
            },
        ],
        outputs: [(MapKind::Normal, normal)].into(),
        selected_output: normal,
    };
    MaterialRequest {
        job_id: JobId(48),
        request_id: serial,
        extent: source.extent,
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
        if let Some(v) = next() {
            return v;
        }
        assert!(Instant::now() < deadline, "T-048 worker timeout");
        std::thread::sleep(Duration::from_millis(1));
    }
}
pub fn install(worker: &GpuWorker, e: Extent, pixels: TypedSourcePixels) -> SourceDescriptor {
    let descriptor = pixels.descriptor(e).unwrap();
    worker
        .install_typed_source(TypedSourceUpload {
            descriptor: descriptor.clone(),
            pixels,
        })
        .unwrap();
    let event = wait(|| worker.try_event());
    assert!(
        matches!(event, WorkerEvent::SourceReady {content_hash} if content_hash == descriptor.content_hash),
        "{event:?}"
    );
    descriptor
}
pub fn install_f32(worker: &GpuWorker, e: Extent, data: Vec<f32>) -> SourceDescriptor {
    install(worker, e, TypedSourcePixels::HeightF32(data.into()))
}
pub fn evaluate(
    gpu: &Gpu,
    worker: &GpuWorker,
    r: MaterialRequest,
    convention: NormalConvention,
) -> Vec<u8> {
    let (serial, extent, quality) = (r.request_id, r.extent, r.quality);
    worker.request_material(r).unwrap();
    let bundle = wait(|| {
        if let Some(event) = worker.try_event() {
            panic!("normal evaluation failed: {event:?}");
        }
        worker.try_latest_material()
    });
    assert_eq!(bundle.job_id, JobId(48));
    assert_eq!(bundle.request_id, serial);
    assert_eq!(bundle.generation, serial);
    assert_eq!(bundle.extent, extent);
    assert_eq!(bundle.quality, quality);
    assert_eq!(bundle.normal_convention, Some(convention));
    assert_eq!(bundle.outputs.len(), 1);
    let bytes = gpu.read(bundle.outputs[&MapKind::Normal].texture());
    assert_eq!(bytes.len(), (extent.width * extent.height * 8) as usize);
    bytes
}
fn with_worker(run: impl FnOnce(&Gpu, &GpuWorker)) {
    with_gpu(|gpu| {
        let worker =
            GpuWorker::spawn(gpu.device.clone(), gpu.queue.clone(), GpuConfig::default()).unwrap();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(gpu, &worker)));
        let joined = worker.join();
        if let Err(panic) = result {
            std::panic::resume_unwind(panic);
        }
        joined.unwrap();
        assert_eq!(worker.diagnostics().validation_errors, 0);
    });
}

#[test]
fn legacy_central_clamp_gl_preserves_executor_and_material_results() {
    with_worker(|gpu, worker| {
        let e = Extent::new(19, 13).unwrap();
        let data = cpu::seeded(e);
        let source = install_f32(worker, e, data.clone());
        let c = Controls {
            filter: NormalFilter::Central,
            boundary: Boundary::Clamp,
            strength: 2.0,
            ..Default::default()
        };
        let actual = evaluate(gpu, worker, request(&source, 1, &c), c.convention);
        cpu::compare(
            &decode(&actual),
            &cpu::normals(e, &data, c.filter, c.boundary, c.strength, c.convention),
        );
        let input = gpu.scalar(e, &data);
        let output = gpu.texture(e, wgpu::TextureFormat::Rgba16Float);
        let pass = gpu
            .executor
            .prepare(
                Pass::Normal {
                    strength: c.strength,
                },
                &input,
                &output,
                e,
            )
            .unwrap();
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        pass.encode(&mut encoder);
        gpu.queue.submit([encoder.finish()]);
        assert_eq!(
            actual,
            gpu.read(&output),
            "legacy path must retain its binary16 output"
        );
    });
}

#[test]
fn analytic_uv_ramps_have_correct_axis_scaling_and_gl_dx_y_sign() {
    with_worker(|gpu, worker| {
        let e = Extent::new(37, 23).unwrap();
        let source = install_f32(worker, e, cpu::ramp(e, 0.5, 0.25));
        let mut serial = 0;
        for filter in [
            NormalFilter::Central,
            NormalFilter::Sobel,
            NormalFilter::Scharr,
        ] {
            for convention in [NormalConvention::OpenGl, NormalConvention::DirectX] {
                for strength in [0.0, 0.5, 2.0, 10.0] {
                    serial += 1;
                    let c = Controls {
                        filter,
                        convention,
                        strength,
                        boundary: Boundary::Clamp,
                        ..Default::default()
                    };
                    let output = decode(&evaluate(
                        gpu,
                        worker,
                        request(&source, serial, &c),
                        convention,
                    ));
                    let expected = cpu::unit(0.5, 0.25, f64::from(strength), convention);
                    let interior: Vec<_> = (2..21)
                        .flat_map(|y| (2..35).map(move |x| y * 37 + x))
                        .map(|i| output[i])
                        .collect();
                    cpu::compare(&interior, &vec![expected; interior.len()]);
                }
            }
        }
    });
}

#[test]
fn spatial_oracle_covers_constants_impulses_npot_singletons_both_boundaries_and_filters() {
    with_worker(|gpu, worker| {
        let mut serial = 0;
        for (w, h, kind) in [
            (1, 1, 0),
            (1, 17, 1),
            (17, 1, 1),
            (11, 7, 0),
            (19, 13, 2),
            (257, 193, 3),
        ] {
            let e = Extent::new(w, h).unwrap();
            let data = match kind {
                0 => vec![-2.0; (w * h) as usize],
                1 => cpu::ramp(e, 0.25, -0.5),
                2 => cpu::image(e, |x, y| if x == 0 && y == 0 { 0.0625 } else { 0.0 }),
                _ => cpu::seeded(e),
            };
            let source = install_f32(worker, e, data.clone());
            for filter in [
                NormalFilter::Central,
                NormalFilter::Sobel,
                NormalFilter::Scharr,
            ] {
                for boundary in [Boundary::Clamp, Boundary::Repeat] {
                    for convention in [NormalConvention::OpenGl, NormalConvention::DirectX] {
                        serial += 1;
                        let c = Controls {
                            filter,
                            boundary,
                            convention,
                            ..Default::default()
                        };
                        let output = decode(&evaluate(
                            gpu,
                            worker,
                            request(&source, serial, &c),
                            convention,
                        ));
                        cpu::compare(
                            &output,
                            &cpu::normals(e, &data, filter, boundary, 1.0, convention),
                        );
                        if w == 1 {
                            assert!(output.iter().all(|n| n[0] == 0.0));
                        }
                        if h == 1 {
                            assert!(output.iter().all(|n| n[1] == 0.0));
                        }
                    }
                }
            }
        }
    });
}

#[test]
fn dx_is_exactly_one_y_negation_with_matching_metadata_on_cold_and_cached_results() {
    with_worker(|gpu, worker| {
        let e = Extent::new(31, 17).unwrap();
        let source = install_f32(worker, e, cpu::seeded(e));
        let mut serial = 0;
        for filter in [NormalFilter::Sobel, NormalFilter::Scharr] {
            let mut c = Controls {
                filter,
                weights: vec![0.5, 0.25, -0.125],
                ..Default::default()
            };
            serial += 1;
            let gl = decode(&evaluate(
                gpu,
                worker,
                request(&source, serial, &c),
                c.convention,
            ));
            c.convention = NormalConvention::DirectX;
            serial += 1;
            let dx_bytes = evaluate(gpu, worker, request(&source, serial, &c), c.convention);
            let dx = decode(&dx_bytes);
            assert!(
                gl.iter().any(|n| n[1].abs() > 0.01),
                "Y flip fixture has no signal"
            );
            for (a, b) in gl.iter().zip(dx) {
                assert_eq!(a[0], b[0]);
                assert_eq!(a[1], -b[1]);
                assert_eq!(a[2], b[2]);
                assert_eq!(a[3], b[3]);
            }
            serial += 1;
            assert_eq!(
                dx_bytes,
                evaluate(gpu, worker, request(&source, serial, &c), c.convention)
            );
            c.convention = NormalConvention::OpenGl;
            serial += 1;
            assert_eq!(
                gl,
                decode(&evaluate(
                    gpu,
                    worker,
                    request(&source, serial, &c),
                    c.convention
                ))
            );
        }
    });
}

#[test]
fn multi_scale_blends_unnormalized_gradients_and_each_scale_has_an_effect() {
    with_worker(|gpu, worker| {
        let e = Extent::new(65, 49).unwrap();
        let source = install_f32(
            worker,
            e,
            cpu::image(e, |x, y| {
                let u = (x as f32 + 0.5) / 65.0;
                let v = (y as f32 + 0.5) / 49.0;
                0.07 * (u * std::f32::consts::TAU).sin()
                    + 0.02 * (v * std::f32::consts::TAU * 5.0).cos()
            }),
        );
        let mut serial = 0;
        for filter in [NormalFilter::Sobel, NormalFilter::Scharr] {
            for boundary in [Boundary::Clamp, Boundary::Repeat] {
                let mut c = Controls {
                    filter,
                    boundary,
                    weights: vec![0.0; 3],
                    ..Default::default()
                };
                let mut basis = Vec::new();
                for scale in 0..3 {
                    c.weights = vec![0.0; 3];
                    c.weights[scale] = 1.0;
                    serial += 1;
                    basis.push(decode(&evaluate(
                        gpu,
                        worker,
                        request(&source, serial, &c),
                        c.convention,
                    )));
                }
                for (scale, result) in basis.iter().enumerate() {
                    assert!(
                        result.iter().any(|n| n[0].abs() + n[1].abs() > 0.05),
                        "scale {scale} was ignored"
                    );
                }
                assert!(
                    basis[0]
                        .iter()
                        .zip(&basis[2])
                        .any(|(a, b)| (a[1] - b[1]).abs() > 0.01),
                    "all scales use the same radius"
                );
                for weights in [
                    [0.0, 0.0, 0.0],
                    [0.25, 0.5, -0.125],
                    [2.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0],
                ] {
                    c.weights = weights.to_vec();
                    serial += 1;
                    let actual = decode(&evaluate(
                        gpu,
                        worker,
                        request(&source, serial, &c),
                        c.convention,
                    ));
                    let expected: Vec<_> = (0..actual.len())
                        .map(|i| {
                            let mut du = 0.0;
                            let mut dv = 0.0;
                            for s in 0..3 {
                                du += f64::from(weights[s]) * -basis[s][i][0] / basis[s][i][2];
                                dv += f64::from(weights[s]) * basis[s][i][1] / basis[s][i][2];
                            }
                            cpu::unit(du, dv, 1.0, c.convention)
                        })
                        .collect();
                    // Recovering slopes from half-float basis results still fits the ADR component bound.
                    cpu::compare(&actual, &expected);
                }
            }
        }
    });
}

#[test]
fn maximum_scale_count_and_singleton_axes_stay_finite_and_zero_strength_is_flat() {
    with_worker(|gpu, worker| {
        let mut serial = 0;
        for (w, h) in [(1, 1), (1, 17), (17, 1), (9, 7)] {
            let e = Extent::new(w, h).unwrap();
            for flat in [false, true] {
                let data = if flat {
                    vec![-2.0; (w * h) as usize]
                } else {
                    cpu::ramp(e, 0.5, 0.25)
                };
                let source = install_f32(worker, e, data);
                for filter in [NormalFilter::Sobel, NormalFilter::Scharr] {
                    for boundary in [Boundary::Clamp, Boundary::Repeat] {
                        for strength in [0.0, 1.0] {
                            let c = Controls {
                                filter,
                                boundary,
                                strength,
                                weights: vec![0.25; 16],
                                ..Default::default()
                            };
                            serial += 1;
                            let actual = decode(&evaluate(
                                gpu,
                                worker,
                                request(&source, serial, &c),
                                c.convention,
                            ));
                            for n in &actual {
                                assert!(n.iter().all(|v| v.is_finite()));
                                assert!(
                                    (n[..3].iter().map(|v| v * v).sum::<f64>().sqrt() - 1.0).abs()
                                        <= 2e-3
                                );
                                assert_eq!(n[3], 1.0);
                                if w == 1 {
                                    assert_eq!(n[0], 0.0);
                                }
                                if h == 1 {
                                    assert_eq!(n[1], 0.0);
                                }
                            }
                            if flat || strength == 0.0 {
                                cpu::compare(&actual, &vec![[0.0, 0.0, 1.0]; actual.len()]);
                            }
                        }
                    }
                }
            }
        }
    });
}

#[test]
fn numeric_height_retains_u16_steps_signed_float_values_and_linear_transfer() {
    with_worker(|gpu, worker| {
        let e = Extent::new(256, 1).unwrap();
        let codes: Vec<u16> = (0..256).map(|x| 32700 + x).collect();
        let f32_data: Vec<f32> = codes.iter().map(|v| f32::from(*v) / 65535.0).collect();
        let mut serial = 0;
        for pixels in [
            TypedSourcePixels::HeightU16(codes.into()),
            TypedSourcePixels::HeightF32(f32_data.clone().into()),
        ] {
            let source = install(worker, e, pixels);
            let c = Controls {
                strength: 10.0,
                boundary: Boundary::Clamp,
                ..Default::default()
            };
            serial += 1;
            let actual = decode(&evaluate(
                gpu,
                worker,
                request(&source, serial, &c),
                c.convention,
            ));
            cpu::compare(
                &actual,
                &cpu::normals(e, &f32_data, c.filter, c.boundary, c.strength, c.convention),
            );
            assert!(
                actual[128][0] < -0.03,
                "height was quantized to 8-bit or flattened"
            );
        }
        let e = Extent::new(17, 13).unwrap();
        let data = cpu::ramp(e, 4.0, -3.0);
        assert!(data.iter().any(|v| *v < 0.0) && data.iter().any(|v| *v > 1.0));
        let source = install_f32(worker, e, data.clone());
        let c = Controls {
            strength: 0.5,
            ..Default::default()
        };
        serial += 1;
        cpu::compare(
            &decode(&evaluate(
                gpu,
                worker,
                request(&source, serial, &c),
                c.convention,
            )),
            &cpu::normals(e, &data, c.filter, c.boundary, c.strength, c.convention),
        );
    });
}

#[test]
fn deterministic_results_survive_generation_changes_and_filter_weight_boundary_edits() {
    with_worker(|gpu, worker| {
        let e = Extent::new(33, 19).unwrap();
        let source = install_f32(worker, e, cpu::seeded(e));
        let c = Controls::default();
        let baseline = evaluate(gpu, worker, request(&source, 1, &c), c.convention);
        let submitted = worker.diagnostics().submitted;
        assert_eq!(
            baseline,
            evaluate(gpu, worker, request(&source, 2, &c), c.convention)
        );
        assert_eq!(
            submitted,
            worker.diagnostics().submitted,
            "warm output recomputed"
        );
        for (i, changed) in [
            Controls {
                filter: NormalFilter::Scharr,
                ..c.clone()
            },
            Controls {
                weights: vec![0.5, 0.5],
                ..c.clone()
            },
            Controls {
                boundary: Boundary::Clamp,
                ..c.clone()
            },
        ]
        .iter()
        .enumerate()
        {
            assert_ne!(
                baseline,
                evaluate(
                    gpu,
                    worker,
                    request(&source, 3 + i as u64, changed),
                    changed.convention
                )
            );
        }
        assert_eq!(
            baseline,
            evaluate(gpu, worker, request(&source, 6, &c), c.convention)
        );
    });
}

#[test]
fn proxy_and_full_normals_agree_in_uv_units_for_analytic_ramps_and_multiple_scales() {
    with_worker(|gpu, worker| {
        let e = Extent::new(2048, 128).unwrap();
        let source = install_f32(worker, e, cpu::ramp(e, 0.5, -0.25));
        let mut serial = 0;
        for filter in [
            NormalFilter::Central,
            NormalFilter::Sobel,
            NormalFilter::Scharr,
        ] {
            for weights in [vec![1.0], vec![0.25, 0.5, 0.25]] {
                let c = Controls {
                    filter,
                    weights,
                    boundary: Boundary::Clamp,
                    ..Default::default()
                };
                serial += 1;
                let full = decode(&evaluate(
                    gpu,
                    worker,
                    request(&source, serial, &c),
                    c.convention,
                ));
                let mut r = request(&source, serial + 1, &c);
                r.quality = Quality::Proxy;
                r.extent = texelith_gpu::proxy_extent(e).unwrap();
                serial += 1;
                let proxy = decode(&evaluate(gpu, worker, r, c.convention));
                let mut count = 0;
                for y in 12..52 {
                    for x in (16..1008).step_by(31) {
                        let a = proxy[y * 1024 + x];
                        let b = full[(2 * y) * 2048 + 2 * x];
                        for channel in 0..3 {
                            assert!((a[channel] - b[channel]).abs() <= 2e-3);
                        }
                        let dot = a[..3].iter().zip(&b[..3]).map(|(u, v)| u * v).sum::<f64>();
                        let len = |n: [f64; 4]| n[..3].iter().map(|v| v * v).sum::<f64>().sqrt();
                        assert!(
                            (dot / (len(a) * len(b))).clamp(-1.0, 1.0).acos() < 0.005,
                            "proxy/full angle exceeds 0.005 radians"
                        );
                        count += 1;
                    }
                }
                assert!(count > 100);
            }
        }
    });
}

#[test]
fn reviewer_repeated_strength_edits_evict_normal_state_to_fit_budget() {
    with_gpu(|gpu| {
        let budget = 800_000;
        let worker = GpuWorker::spawn(
            gpu.device.clone(),
            gpu.queue.clone(),
            GpuConfig {
                budget_bytes: budget,
                max_cache_entries: 8,
            },
        )
        .unwrap();
        let e = Extent::new(257, 193).unwrap();
        let data = cpu::ramp(e, 0.5, 0.25);
        let source = install_f32(&worker, e, data.clone());
        for (i, strength) in [1.0, 2.0, 0.5, 3.0].into_iter().enumerate() {
            let c = Controls {
                strength,
                boundary: Boundary::Clamp,
                ..Default::default()
            };
            let r = request(&source, i as u64 + 1, &c);
            let plan = texelith_gpu::MaterialPlan::new(
                &r,
                budget,
                texelith_gpu::ResourceReservations::default(),
            )
            .unwrap();
            assert_eq!(plan.peak_bytes(), 793_872);
            // evaluate drops the completed bundle before the next edit.
            let actual = decode(&evaluate(gpu, &worker, r, c.convention));
            cpu::compare(
                &actual,
                &cpu::normals(e, &data, c.filter, c.boundary, strength, c.convention),
            );
        }
        let diagnostics = worker.diagnostics();
        assert!(diagnostics.peak_bytes <= budget);
        assert!(diagnostics.cache_evictions > 0);
        assert_eq!(diagnostics.validation_errors, 0);
        worker.join().unwrap();
    });
}

#[test]
fn reviewer_extreme_finite_proxy_weights_are_rejected_without_partial_output_and_recover() {
    with_worker(|gpu, worker| {
        let e = Extent::new(2048, 16).unwrap();
        let source = install_f32(worker, e, vec![-2.0; (e.width * e.height) as usize]);
        // At proxy octave one these individually finite weights fold into one
        // slot. Exercise both fresh preparation and cached reweighting.
        for serial in [1, 3] {
            let mut c = Controls {
                weights: vec![3e38, 3e38],
                ..Default::default()
            };
            let mut r = request(&source, serial, &c);
            r.quality = Quality::Proxy;
            r.extent = texelith_gpu::proxy_extent(e).unwrap();
            worker.request_material(r).unwrap();
            let event = wait(|| {
                assert!(
                    worker.try_latest_material().is_none(),
                    "invalid weights published normals"
                );
                worker.try_event()
            });
            assert!(
                matches!(event, WorkerEvent::Error(ref error) if error.0.contains("normal weights")),
                "{event:?}"
            );
            assert!(worker.try_latest_material().is_none());
            c.weights = vec![0.5, 0.5];
            let mut r = request(&source, serial + 1, &c);
            r.quality = Quality::Proxy;
            r.extent = texelith_gpu::proxy_extent(e).unwrap();
            let actual = decode(&evaluate(gpu, worker, r, c.convention));
            cpu::compare(&actual, &vec![[0.0, 0.0, 1.0]; actual.len()]);
        }
    });
}
