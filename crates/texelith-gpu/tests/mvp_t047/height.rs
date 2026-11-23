use super::{
    gpu::{Gpu, with_gpu},
    oracle::compare,
    reference,
};
use std::{
    sync::Arc,
    time::{Duration, Instant},
};
use texelith_core::{
    Boundary, EvaluationSnapshot, Extent, Graph, InputEdge, InputPort, MapKind, MaterialGraphDto,
    MaterialNode, MaterialNodeParams as P, NodeId, OutputPort, OutputRef, SourceDescriptor,
};
use texelith_gpu::{
    CompletedMaterial, GpuConfig, GpuWorker, JobId, MaterialPlan, MaterialRequest, Quality,
    ResourceReservations, TypedSourcePixels, TypedSourceUpload, WorkerEvent,
};

const GIB: u64 = 1024 * 1024 * 1024;

pub fn wait<T>(mut next: impl FnMut() -> Option<T>) -> T {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        if let Some(value) = next() {
            return value;
        }
        assert!(Instant::now() < deadline, "T-047 worker timed out");
        std::thread::sleep(Duration::from_millis(1));
    }
}

pub fn request(
    source: &SourceDescriptor,
    serial: u64,
    weights: &[f32],
    coarse: f32,
    remove: bool,
    boundary: Boundary,
    levels: Option<(f32, f32, f32)>,
) -> MaterialRequest {
    let color = OutputRef {
        node: NodeId(1),
        port: OutputPort::Albedo,
    };
    let height = OutputRef {
        node: NodeId(2),
        port: OutputPort::Height,
    };
    let mut nodes = vec![
        MaterialNode {
            id: color.node,
            inputs: vec![],
            params: P::ColorSource(source.clone()),
        },
        MaterialNode {
            id: height.node,
            inputs: vec![InputEdge {
                port: InputPort::Color,
                source: color,
            }],
            params: P::HeightFromAlbedo {
                band_weights: weights.to_vec(),
                coarse_weight: coarse,
                remove_coarse: remove,
                boundary,
            },
        },
    ];
    let selected = if let Some((black, white, gamma)) = levels {
        nodes.push(MaterialNode {
            id: NodeId(3),
            inputs: vec![InputEdge {
                port: InputPort::Height,
                source: height,
            }],
            params: P::Levels {
                black,
                white,
                gamma,
                boundary: Boundary::Clamp,
            },
        });
        OutputRef {
            node: NodeId(3),
            port: OutputPort::Height,
        }
    } else {
        height
    };
    let dto = MaterialGraphDto {
        schema_version: 2,
        graph: nodes,
        outputs: [(MapKind::Height, selected)].into(),
        selected_output: selected,
    };
    MaterialRequest {
        job_id: JobId(47),
        request_id: serial,
        outputs: dto.outputs.clone(),
        extent: source.extent,
        quality: Quality::Full,
        snapshot: Arc::new(EvaluationSnapshot {
            generation: serial,
            graph: Graph::try_from(dto).unwrap(),
        }),
    }
}

pub fn install(worker: &GpuWorker, extent: Extent, rgba: Vec<u8>) -> SourceDescriptor {
    let pixels = TypedSourcePixels::Rgba8 {
        pixels: rgba.into(),
        interpretation: texelith_core::InputInterpretation::SrgbPhoto,
    };
    let descriptor = pixels.descriptor(extent).unwrap();
    worker
        .install_typed_source(TypedSourceUpload {
            descriptor: descriptor.clone(),
            pixels,
        })
        .unwrap();
    let event = wait(|| worker.try_event());
    assert!(
        matches!(event, WorkerEvent::SourceReady { content_hash }
        if content_hash == descriptor.content_hash),
        "{event:?}"
    );
    descriptor
}

pub fn evaluate(gpu: &Gpu, worker: &GpuWorker, request: MaterialRequest) -> Vec<f32> {
    let serial = request.request_id;
    let extent = request.extent;
    let quality = request.quality;
    worker.request_material(request).unwrap();
    let bundle: CompletedMaterial = wait(|| {
        if let Some(event) = worker.try_event() {
            panic!("height evaluation failed: {event:?}");
        }
        worker.try_latest_material()
    });
    assert_eq!(bundle.request_id, serial);
    assert_eq!(bundle.generation, serial);
    assert_eq!(bundle.job_id, JobId(47));
    assert_eq!(bundle.extent, extent);
    assert_eq!(bundle.quality, quality);
    assert_eq!(bundle.outputs.len(), 1);
    let texture = bundle.outputs[&MapKind::Height].texture();
    assert_eq!(texture.format(), wgpu::TextureFormat::R32Float);
    let values = gpu.read_scalar(texture);
    assert_eq!(values.len(), (extent.width * extent.height) as usize);
    assert!(values.iter().all(|v| v.is_finite()));
    values
}

pub fn fixture(w: u32, h: u32, kind: &str) -> Vec<u8> {
    let mut state = 47u32;
    (0..w * h)
        .flat_map(|i| {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            match kind {
                "flat" => [128, 128, 128, (i % 256) as u8],
                "impulse" => {
                    if i == 0 {
                        [255, 255, 255, 0]
                    } else {
                        [0, 0, 0, 255]
                    }
                }
                "ramp" => {
                    let v = (i * 255 / (w * h - 1).max(1)) as u8;
                    [v, v, v, 255]
                }
                "proxy" => {
                    let v = if ((i % w) / 8).is_multiple_of(2) {
                        64
                    } else {
                        192
                    };
                    [v, v, v, 255]
                }
                _ => [
                    (state >> 24) as u8,
                    (state >> 16) as u8,
                    (state >> 8) as u8,
                    state as u8,
                ],
            }
        })
        .collect()
}

#[test]
fn worker_unit_weights_reconstruct_linear_luminance_on_npot_singletons_and_seeded_inputs() {
    with_gpu(|gpu| {
        let worker =
            GpuWorker::spawn(gpu.device.clone(), gpu.queue.clone(), GpuConfig::default()).unwrap();
        let mut serial = 0;
        for (w, h, kind) in [
            (1, 1, "flat"),
            (1, 17, "ramp"),
            (17, 1, "ramp"),
            (9, 7, "flat"),
            (31, 17, "impulse"),
            (257, 193, "seeded"),
        ] {
            let extent = Extent::new(w, h).unwrap();
            let bytes = fixture(w, h, kind);
            let expected = reference::luminance(extent, &bytes);
            let source = install(&worker, extent, bytes);
            for boundary in [Boundary::Clamp, Boundary::Repeat] {
                for weights in [&[1.0][..], &[1.0; 16][..]] {
                    serial += 1;
                    compare(
                        &evaluate(
                            gpu,
                            &worker,
                            request(&source, serial, weights, 1.0, false, boundary, None),
                        ),
                        &expected.pixels,
                        2e-5,
                    );
                }
            }
        }
        assert_eq!(worker.diagnostics().validation_errors, 0);
        worker.join().unwrap();
    });
}

#[test]
fn worker_band_weight_delta_is_only_the_selected_lifted_band() {
    with_gpu(|gpu| {
        let worker =
            GpuWorker::spawn(gpu.device.clone(), gpu.queue.clone(), GpuConfig::default()).unwrap();
        let extent = Extent::new(33, 19).unwrap();
        let bytes = fixture(33, 19, "seeded");
        let image = reference::luminance(extent, &bytes);
        let source = install(&worker, extent, bytes);
        let mut serial = 0;
        for boundary in [Boundary::Repeat, Boundary::Clamp] {
            serial += 1;
            let unit = evaluate(
                gpu,
                &worker,
                request(&source, serial, &[1.0], 1.0, false, boundary, None),
            );
            let count = reference::pyramid(&image, boundary).len() - 1;
            for selected in 0..count {
                let mut weights = vec![1.0; count];
                weights[selected] = -0.75;
                serial += 1;
                let changed = evaluate(
                    gpu,
                    &worker,
                    request(&source, serial, &weights, 1.0, false, boundary, None),
                );
                let expected = reference::reconstruct(&image, &weights, 1.0, boundary);
                compare(&changed, &expected.pixels, 2e-5);
                let mut isolated = vec![0.0; count];
                isolated[selected] = -1.75;
                let delta = reference::reconstruct(&image, &isolated, 0.0, boundary);
                let actual_delta: Vec<f32> =
                    changed.iter().zip(&unit).map(|(a, b)| a - b).collect();
                compare(&actual_delta, &delta.pixels, 4e-5);
                // For Repeat band 5 the chain is 33x19 -> 17x10 -> 9x5 ->
                // 5x3 -> 3x2 -> 2x1 -> 1x1. The five-tap response is
                // H(w)=cos(w/2)^4: H(2*pi/3)=1/16, H(pi)=0.
                // On the 3x2 level let C be column-mean(0)-column-mean(2).
                // Area restriction 3->2 has row difference (2/3,0,-2/3),
                // hence G5[0]-G5[1]=C/24. Band 5 is +/-C/48. Lifting the
                // alternating [1,-1] through 2->3->5->9->17->33 with the
                // specified pixel centers has infinity gain exactly 2/5.
                // The -1.75 weight delta therefore has amplitude 7*|C|/480,
                // 5.065294783e-6 for this seeded fixture, below 1e-5 by
                // construction. Retain the energy check with that derived
                // lower bound and 1e-12 f64 tolerance; other bands keep 1e-5.
                let minimum = if boundary == Boundary::Repeat && selected == 5 {
                    let levels = reference::pyramid(&image, boundary);
                    let g4 = &levels[4].pixels;
                    let contrast = ((g4[0] + g4[3]) - (g4[2] + g4[5])) * 0.5;
                    let amplitude = 7.0 * contrast.abs() / 480.0;
                    assert!((1e-6..1e-5).contains(&amplitude));
                    let measured = delta.pixels.iter().map(|v| v.abs()).fold(0.0, f64::max);
                    assert!((measured - amplitude).abs() <= 1e-12);
                    amplitude - 1e-12
                } else {
                    1e-5
                };
                assert!(
                    delta.pixels.iter().any(|v| v.abs() > minimum),
                    "ineffective band fixture {selected}"
                );
            }
        }
        worker.join().unwrap();
    });
}

#[test]
fn worker_repeat_band_five_has_a_gpu_contribution_above_oracle_tolerance() {
    with_gpu(|gpu| {
        let worker =
            GpuWorker::spawn(gpu.device.clone(), gpu.queue.clone(), GpuConfig::default()).unwrap();
        let extent = Extent::new(33, 19).unwrap();
        // A broad edge survives to the coarsest band, unlike seeded noise.
        let bytes: Vec<u8> = (0..33 * 19)
            .flat_map(|i| {
                let value = if i % 33 < 16 { 255 } else { 0 };
                [value, value, value, 255]
            })
            .collect();
        let image = reference::luminance(extent, &bytes);
        let source = install(&worker, extent, bytes);
        let baseline = evaluate(
            gpu,
            &worker,
            request(&source, 1, &[1.0], 1.0, false, Boundary::Repeat, None),
        );
        let mut weights = [1.0; 6];
        weights[5] = -0.75;
        let changed = evaluate(
            gpu,
            &worker,
            request(&source, 2, &weights, 1.0, false, Boundary::Repeat, None),
        );
        let expected = reference::reconstruct(&image, &weights, 1.0, Boundary::Repeat);
        compare(&changed, &expected.pixels, 2e-5);
        let mut isolated = [0.0; 6];
        isolated[5] = -1.75;
        let delta = reference::reconstruct(&image, &isolated, 0.0, Boundary::Repeat);
        let amplitude = delta.pixels.iter().map(|v| v.abs()).fold(0.0, f64::max);
        assert!(amplitude > 10.0 * 4e-5, "ineffective band-5 fixture");
        let actual: Vec<_> = changed.iter().zip(&baseline).map(|(a, b)| a - b).collect();
        compare(&actual, &delta.pixels, 4e-5);
        assert!(actual.iter().any(|v| v.abs() > 4e-4));
        worker.join().unwrap();
    });
}

#[test]
fn public_albedo_proxies_publish_declared_sizes_for_single_and_mixed_output_requests() {
    with_gpu(|gpu| {
        let worker =
            GpuWorker::spawn(gpu.device.clone(), gpu.queue.clone(), GpuConfig::default()).unwrap();
        let extent = Extent::new(2048, 16).unwrap();
        let source = install(&worker, extent, fixture(2048, 16, "proxy"));
        let proxy = Extent::new(1024, 8).unwrap();
        assert_eq!(texelith_gpu::proxy_extent(extent).unwrap(), proxy);
        for mixed in [false, true] {
            let serial = if mixed { 2 } else { 1 };
            let mut r = request(&source, serial, &[1.0], 1.0, false, Boundary::Repeat, None);
            r.quality = Quality::Proxy;
            r.extent = proxy;
            if !mixed {
                r.outputs.clear();
            }
            r.outputs.insert(
                MapKind::Albedo,
                OutputRef {
                    node: NodeId(1),
                    port: OutputPort::Albedo,
                },
            );
            let expected_outputs = r.outputs.clone();
            assert!(MaterialPlan::new(&r, GIB, ResourceReservations::default()).is_ok());
            worker.request_material(r).unwrap();
            let bundle = wait(|| {
                if let Some(event) = worker.try_event() {
                    panic!("albedo proxy evaluation failed: {event:?}");
                }
                worker.try_latest_material()
            });
            assert_eq!(bundle.job_id, JobId(47));
            assert_eq!(bundle.request_id, serial);
            assert_eq!(bundle.generation, serial);
            assert_eq!(bundle.quality, Quality::Proxy);
            assert_eq!(bundle.extent, proxy);
            assert_eq!(
                bundle.outputs.keys().copied().collect::<Vec<_>>(),
                expected_outputs.keys().copied().collect::<Vec<_>>()
            );
            for (kind, output) in &bundle.outputs {
                let texture = output.texture();
                assert_eq!(texture.width(), proxy.width, "{kind:?} proxy width");
                assert_eq!(texture.height(), proxy.height, "{kind:?} proxy height");
                assert_eq!(texture.depth_or_array_layers(), 1);
                assert_eq!(
                    texture.format(),
                    if *kind == MapKind::Albedo {
                        wgpu::TextureFormat::Rgba16Float
                    } else {
                        wgpu::TextureFormat::R32Float
                    }
                );
            }
            assert!(worker.try_latest_material().is_none());
        }
        assert!(worker.diagnostics().submitted > 0);
        assert_eq!(worker.diagnostics().validation_errors, 0);
        // Publishing a public proxy must preserve height's internal full color.
        let mut height = request(&source, 3, &[1.0], 1.0, false, Boundary::Repeat, None);
        height.quality = Quality::Proxy;
        height.extent = texelith_gpu::proxy_extent(extent).unwrap();
        assert_eq!(evaluate(gpu, &worker, height).len(), 1024 * 8);
        worker.join().unwrap();
    });
}

#[test]
fn worker_weight_edit_evicts_the_height_chain_to_fit_the_budget() {
    with_gpu(|gpu| {
        // Scale the reviewed 4K / 600 MiB case by sixteen for a small regression.
        let budget = 600 * 1024 * 1024 / 16;
        let worker = GpuWorker::spawn(
            gpu.device.clone(),
            gpu.queue.clone(),
            GpuConfig {
                budget_bytes: budget,
                max_cache_entries: 8,
            },
        )
        .unwrap();
        let extent = Extent::new(1024, 1024).unwrap();
        let source = install(&worker, extent, fixture(1024, 1024, "flat"));
        let flat = reference::luminance(Extent::new(1, 1).unwrap(), &[128; 4]).pixels[0];
        for (serial, weights, coarse, value) in [
            (1, &[1.0][..], 1.0, flat),
            (2, &[0.0][..], 0.0, 0.0),
            (3, &[1.0][..], 1.0, flat),
        ] {
            let r = request(
                &source,
                serial,
                weights,
                coarse,
                false,
                Boundary::Repeat,
                None,
            );
            assert!(MaterialPlan::new(&r, budget, ResourceReservations::default()).is_ok());
            let actual = evaluate(gpu, &worker, r);
            compare(&actual, &vec![value; actual.len()], 2e-5);
        }
        let diagnostics = worker.diagnostics();
        assert!(diagnostics.peak_bytes <= budget);
        assert_eq!(diagnostics.validation_errors, 0);
        worker.join().unwrap();
    });
}

#[test]
fn worker_coarse_removal_zero_weights_and_levels_are_explicit_and_preserve_signed_height() {
    with_gpu(|gpu| {
        let worker =
            GpuWorker::spawn(gpu.device.clone(), gpu.queue.clone(), GpuConfig::default()).unwrap();
        let e = Extent::new(31, 17).unwrap();
        let bytes = fixture(31, 17, "seeded");
        let image = reference::luminance(e, &bytes);
        let source = install(&worker, e, bytes);
        let b = Boundary::Repeat;
        let removed = evaluate(
            gpu,
            &worker,
            request(&source, 1, &[1.0], 3.0, true, b, None),
        );
        let expected = reference::reconstruct(&image, &[1.0], 0.0, b);
        compare(&removed, &expected.pixels, 2e-5);
        assert!(
            removed.iter().any(|v| *v < 0.0),
            "implicit clamping destroyed signed detail"
        );
        let zero = evaluate(
            gpu,
            &worker,
            request(&source, 2, &[0.0], 0.0, false, b, None),
        );
        compare(&zero, &vec![0.0; image.pixels.len()], 2e-5);
        let base = evaluate(
            gpu,
            &worker,
            request(&source, 3, &[0.0], 2.0, false, b, None),
        );
        compare(
            &base,
            &reference::reconstruct(&image, &[0.0], 2.0, b).pixels,
            2e-5,
        );
        let adjusted = evaluate(
            gpu,
            &worker,
            request(&source, 4, &[1.0], 3.0, true, b, Some((0.0, 0.5, 2.0))),
        );
        compare(
            &adjusted,
            &reference::levels(&expected, 0.0, 0.5, 2.0),
            2e-5,
        );
        assert!(adjusted.contains(&0.0));
        let weights = [0.25, -0.5, 1.5, 0.75];
        let weighted = evaluate(
            gpu,
            &worker,
            request(&source, 5, &weights, 4.0, false, b, None),
        );
        compare(
            &weighted,
            &reference::reconstruct(&image, &weights, 4.0, b).pixels,
            2e-5,
        );
        assert!(
            weighted.iter().any(|v| *v > 1.0),
            "unrequested normalization or clamp"
        );
        worker.join().unwrap();
    });
}

#[test]
fn worker_srgb_ramp_uses_rec709_after_transfer_ignores_alpha_and_never_reencodes_height() {
    with_gpu(|gpu| {
        let worker =
            GpuWorker::spawn(gpu.device.clone(), gpu.queue.clone(), GpuConfig::default()).unwrap();
        let e = Extent::new(16, 1).unwrap();
        let mut bytes: Vec<u8> = [0, 10, 11, 32, 64, 128, 192, 255]
            .into_iter()
            .flat_map(|v| [[v, v, v, 0], [v, v, v, 255]])
            .flatten()
            .collect();
        bytes[0..4].copy_from_slice(&[255, 0, 0, 0]);
        bytes[4..8].copy_from_slice(&[255, 0, 0, 255]);
        let expected = reference::luminance(e, &bytes);
        let source = install(&worker, e, bytes);
        let actual = evaluate(
            gpu,
            &worker,
            request(&source, 1, &[1.0], 1.0, false, Boundary::Repeat, None),
        );
        compare(&actual, &expected.pixels, 2e-5);
        for pair in actual.as_chunks::<2>().0 {
            assert!((pair[0] - pair[1]).abs() < 2e-6);
        }
        assert!(
            (actual[10] - 128.0 / 255.0).abs() > 0.1,
            "sRGB bytes treated as linear height"
        );
        assert!(
            (actual[0] - 0.2126).abs() < 2e-5,
            "red-only or average RGB is not Rec.709"
        );
        worker.join().unwrap();
    });
}

#[test]
fn worker_proxy_weights_remain_in_original_source_octaves() {
    with_gpu(|gpu| {
        let worker =
            GpuWorker::spawn(gpu.device.clone(), gpu.queue.clone(), GpuConfig::default()).unwrap();
        let e = Extent::new(2048, 16).unwrap();
        let bytes = fixture(e.width, e.height, "proxy");
        let image = reference::luminance(e, &bytes);
        let source = install(&worker, e, bytes);
        let mut proxy = request(&source, 1, &[0.0, 1.0], 1.0, false, Boundary::Repeat, None);
        proxy.quality = Quality::Proxy;
        proxy.extent = texelith_gpu::proxy_extent(e).unwrap();
        assert_eq!(proxy.extent, Extent::new(1024, 8).unwrap());
        let actual = evaluate(gpu, &worker, proxy);
        // The finest source octave is below the proxy's Nyquist limit. The
        // remaining all-one weights must reconstruct the area-reduced source.
        compare(&actual, &image.reduce().pixels, 2e-5);
        let full = evaluate(
            gpu,
            &worker,
            request(&source, 2, &[0.0, 1.0], 1.0, false, Boundary::Repeat, None),
        );
        compare(
            &full,
            &reference::reconstruct(&image, &[0.0, 1.0], 1.0, Boundary::Repeat).pixels,
            2e-5,
        );
        assert!(
            full.iter()
                .zip(image.pixels)
                .any(|(a, b)| (f64::from(*a) - b).abs() > 0.001)
        );
        worker.join().unwrap();
    });
}

#[test]
fn worker_repeated_content_is_bitwise_deterministic_and_uses_output_cache() {
    with_gpu(|gpu| {
        let worker =
            GpuWorker::spawn(gpu.device.clone(), gpu.queue.clone(), GpuConfig::default()).unwrap();
        let e = Extent::new(33, 19).unwrap();
        let source = install(&worker, e, fixture(e.width, e.height, "seeded"));
        let r = request(
            &source,
            1,
            &[0.5, 1.25, -0.25],
            0.8,
            false,
            Boundary::Repeat,
            None,
        );
        let plan = MaterialPlan::new(&r, GIB, ResourceReservations::default()).unwrap();
        let a = evaluate(gpu, &worker, r.clone());
        let before = worker.diagnostics();
        let mut later = r.clone();
        later.request_id = 2;
        later.snapshot = Arc::new(EvaluationSnapshot {
            generation: 2,
            graph: r.snapshot.graph.clone(),
        });
        assert_eq!(
            plan.output_keys(),
            MaterialPlan::new(&later, GIB, ResourceReservations::default())
                .unwrap()
                .output_keys()
        );
        let b = evaluate(gpu, &worker, later);
        let after = worker.diagnostics();
        assert_eq!(
            a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            b.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert!(after.cache_hits > before.cache_hits);
        assert_eq!(
            after.submitted, before.submitted,
            "warm output must avoid compute submission"
        );
        let changed = request(
            &source,
            3,
            &[0.6, 1.25, -0.25],
            0.8,
            false,
            Boundary::Repeat,
            None,
        );
        assert_ne!(
            plan.output_keys(),
            MaterialPlan::new(&changed, GIB, ResourceReservations::default())
                .unwrap()
                .output_keys()
        );
        let c = evaluate(gpu, &worker, changed);
        assert!(
            c.iter().zip(&a).any(|(a, b)| (a - b).abs() > 1e-5),
            "stale cache after slider edit"
        );
        assert!(
            worker.diagnostics().pyramid_cache_hits > after.pyramid_cache_hits,
            "a weight-only edit must reuse the completed Gaussian chain, not only ColorSource"
        );
        worker.join().unwrap();
    });
}

#[test]
fn material_plan_admits_height_under_one_gib_and_rejects_below_its_actual_peak() {
    let e = Extent::new(4096, 4096).unwrap();
    let source = SourceDescriptor {
        content_hash: [47; 32],
        extent: e,
        interpretation: texelith_core::InputInterpretation::SrgbPhoto,
    };
    let r = request(&source, 1, &[1.0], 1.0, false, Boundary::Repeat, None);
    let reservations = ResourceReservations {
        viewport_bytes: 144 * 1024 * 1024,
        readback_bytes: 0,
    };
    let plan =
        MaterialPlan::new(&r, GIB, reservations).expect("HeightFromAlbedo must be supported");
    assert!(plan.peak_bytes() > 256 * 1024 * 1024);
    assert!(plan.peak_bytes() <= GIB);
    assert!(matches!(
        MaterialPlan::new(&r, plan.peak_bytes() - 1, reservations),
        Err(texelith_gpu::MaterialError::BudgetExceeded { .. })
    ));
}

#[test]
fn worker_budget_rejection_has_no_partial_height_and_recovers_after_reservation_release() {
    with_gpu(|gpu| {
        let worker = GpuWorker::spawn(
            gpu.device.clone(),
            gpu.queue.clone(),
            GpuConfig {
                budget_bytes: 64 * 1024,
                max_cache_entries: 8,
            },
        )
        .unwrap();
        let e = Extent::new(16, 16).unwrap();
        let bytes = fixture(16, 16, "seeded");
        let expected = reference::luminance(e, &bytes);
        let source = install(&worker, e, bytes);
        worker
            .set_resource_reservations(ResourceReservations {
                viewport_bytes: 60 * 1024,
                readback_bytes: 0,
            })
            .unwrap();
        worker
            .request_material(request(
                &source,
                1,
                &[1.0],
                1.0,
                false,
                Boundary::Repeat,
                None,
            ))
            .unwrap();
        let event = wait(|| worker.try_event());
        assert!(
            matches!(event,WorkerEvent::BudgetExceeded { budget_bytes:65536,
            required_bytes } if required_bytes>65536),
            "{event:?}"
        );
        assert!(worker.try_latest_material().is_none());
        worker
            .set_resource_reservations(ResourceReservations::default())
            .unwrap();
        compare(
            &evaluate(
                gpu,
                &worker,
                request(&source, 2, &[1.0], 1.0, false, Boundary::Repeat, None),
            ),
            &expected.pixels,
            2e-5,
        );
        let d = worker.diagnostics();
        assert!(d.peak_bytes <= 65536);
        assert_eq!(d.validation_errors, 0);
        worker.join().unwrap();
    });
}

#[test]
fn height_cancellation_uses_latest_slot_and_never_publishes_obsolete_bundle() {
    use texelith_gpu::scheduler::{Completion, Scheduler};
    let source = SourceDescriptor {
        content_hash: [47; 32],
        extent: Extent::new(257, 193).unwrap(),
        interpretation: texelith_core::InputInterpretation::SrgbPhoto,
    };
    let mut scheduler = Scheduler::default();
    scheduler
        .request_material(request(
            &source,
            1,
            &[1.0],
            1.0,
            false,
            Boundary::Repeat,
            None,
        ))
        .unwrap();
    let first = scheduler.start_next().unwrap();
    assert!(scheduler.should_continue(first.id));
    for id in 2..=64 {
        scheduler
            .request_material(request(
                &source,
                id,
                &[0.5, 1.0],
                1.0,
                false,
                Boundary::Repeat,
                None,
            ))
            .unwrap();
        assert!(!scheduler.should_continue(first.id));
        assert!(scheduler.start_next().is_none());
        assert!(scheduler.try_latest_material().is_none());
    }
    assert_eq!(scheduler.complete(first.id), Completion::Discarded);
    assert!(scheduler.try_latest_material().is_none());
    let last = scheduler.start_next().unwrap();
    assert_eq!(scheduler.complete(last.id), Completion::Published);
    assert_eq!(scheduler.try_latest_material().unwrap().request_id, 64);
    assert_eq!(scheduler.diagnostics().peak_in_flight, 1);
    assert_eq!(scheduler.diagnostics().peak_pending, 1);
}

/// Exercise cancellation after the real worker has submitted ColorSource and
/// at least one height pass. Shared with the native height verification runner.
pub fn cancel_runtime(
    gpu: &Gpu,
    worker: &GpuWorker,
    source: &SourceDescriptor,
    serial: u64,
) -> texelith_gpu::WorkerDiagnostics {
    let before = worker.diagnostics();
    worker
        .request_material(request(
            source,
            serial,
            &[0.3],
            0.7,
            false,
            Boundary::Clamp,
            None,
        ))
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(30);
    while worker.diagnostics().submitted < before.submitted + 2 {
        assert!(
            Instant::now() < deadline,
            "height never reached GPU submission"
        );
        assert!(
            worker.try_latest_material().is_none(),
            "cold height finished before cancellation"
        );
        assert!(worker.try_event().is_none());
        std::thread::yield_now();
    }
    for id in serial + 1..=serial + 64 {
        worker
            .request_material(request(
                source,
                id,
                &[0.0],
                0.0,
                false,
                Boundary::Repeat,
                None,
            ))
            .unwrap();
    }
    let final_bundle = wait(|| {
        assert!(worker.try_event().is_none());
        worker.try_latest_material()
    });
    assert_eq!(final_bundle.request_id, serial + 64);
    assert_eq!(final_bundle.generation, serial + 64);
    compare(
        &gpu.read_scalar(final_bundle.outputs[&MapKind::Height].texture()),
        &vec![0.0; (source.extent.width * source.extent.height) as usize],
        2e-5,
    );
    let after = worker.diagnostics();
    assert!(after.cancelled_pass_batches > before.cancelled_pass_batches);
    assert!(after.cancellation_requests > before.cancellation_requests);
    assert_eq!(after.stale_publications, 0);
    assert_eq!(after.peak_in_flight, 1);
    assert_eq!(after.peak_pending, 1);
    assert_eq!(after.validation_errors, 0);
    after
}

#[test]
fn real_height_worker_cancels_submitted_passes_and_publishes_only_the_latest_generation() {
    with_gpu(|gpu| {
        let worker =
            GpuWorker::spawn(gpu.device.clone(), gpu.queue.clone(), GpuConfig::default()).unwrap();
        let extent = Extent::new(1024, 1024).unwrap();
        let source = install(&worker, extent, fixture(1024, 1024, "seeded"));
        cancel_runtime(gpu, &worker, &source, 1000);
        worker.join().unwrap();
    });
}

#[test]
fn height_content_keys_separate_all_weight_coarse_boundary_and_levels_controls() {
    let source = SourceDescriptor {
        content_hash: [47; 32],
        extent: Extent::new(33, 19).unwrap(),
        interpretation: texelith_core::InputInterpretation::SrgbPhoto,
    };
    let key = |r: MaterialRequest| {
        *MaterialPlan::new(&r, GIB, ResourceReservations::default())
            .unwrap()
            .output_keys()
            .get(&MapKind::Height)
            .unwrap()
    };
    let base = key(request(
        &source,
        1,
        &[1.0],
        1.0,
        false,
        Boundary::Repeat,
        None,
    ));
    for changed in [
        request(&source, 2, &[0.5], 1.0, false, Boundary::Repeat, None),
        request(&source, 3, &[1.0], 0.5, false, Boundary::Repeat, None),
        request(&source, 4, &[1.0], 1.0, true, Boundary::Repeat, None),
        request(&source, 5, &[1.0], 1.0, false, Boundary::Clamp, None),
        request(
            &source,
            6,
            &[1.0],
            1.0,
            false,
            Boundary::Repeat,
            Some((0.0, 1.0, 2.0)),
        ),
    ] {
        assert_ne!(base, key(changed));
    }
}
