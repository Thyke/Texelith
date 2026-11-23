use super::{behavior, gpu::with_gpu, reference};
use std::{collections::BTreeMap, sync::Arc};
use texelith_core::{
    Boundary, EvaluationSnapshot, Extent, InputInterpretation, MapKind, MaterialGraphDto,
    material_from_source,
};
use texelith_gpu::{
    GpuConfig, GpuWorker, JobId, MaterialPlan, MaterialRequest, Quality, ResourceReservations,
    TypedSourcePixels,
    seamless_offset::{OffsetKernels, OffsetTransform},
};

#[test]
fn default_4096_photo_graph_evaluates_height_at_1024_proxy_and_reuses_it() {
    with_gpu(|gpu| {
        let worker =
            GpuWorker::spawn(gpu.device.clone(), gpu.queue.clone(), GpuConfig::default()).unwrap();
        let source_extent = Extent::new(4096, 4096).unwrap();
        let extent = Extent::new(1024, 1024).unwrap();
        let code = |x: u32, y: u32| ((x * 13 + y * 7) % 256) as u8;
        // Four-by-four blocks give an independently known bilinear sample at
        // every proxy center, including each translated copy's center.
        let pixels: Vec<_> = (0..4096 * 4096)
            .flat_map(|i| {
                let c = code(i % 4096 / 4, i / 4096 / 4);
                [c, c, c, (i % 256) as u8]
            })
            .collect();
        let source = behavior::install(
            &worker,
            source_extent,
            TypedSourcePixels::Rgba8 {
                pixels: pixels.into(),
                interpretation: InputInterpretation::SrgbPhoto,
            },
        );
        let graph = material_from_source(source).unwrap();
        let height = MaterialGraphDto::from(graph.clone()).outputs[&MapKind::Height];
        let mut request = MaterialRequest {
            job_id: JobId(57),
            request_id: 1,
            snapshot: Arc::new(EvaluationSnapshot {
                generation: 1,
                graph,
            }),
            outputs: BTreeMap::from([(MapKind::Height, height)]),
            extent,
            quality: Quality::Proxy,
        };
        let plan = MaterialPlan::new(
            &request,
            GpuConfig::default().budget_bytes,
            ResourceReservations::default(),
        )
        .unwrap();
        let channel: Vec<_> = (0..1024 * 1024)
            .map(|i| reference::half(reference::linear(code(i % 1024, i / 1024))))
            .collect();
        let expected = reference::render(extent, &channel, 0.25, Boundary::Repeat);
        let mut previous = None;
        for serial in 1..=2 {
            request.request_id = serial;
            worker.request_material(request.clone()).unwrap();
            let result = behavior::wait(|| {
                if let Some(error) = worker.try_event() {
                    panic!("default photo proxy failed: {error:?}");
                }
                worker.try_latest_material()
            });
            assert_eq!(result.job_id, JobId(57));
            assert_eq!(result.request_id, serial);
            assert_eq!(result.generation, 1);
            assert_eq!(result.quality, Quality::Proxy);
            assert_eq!(result.extent, extent);
            assert_eq!(result.outputs.len(), 1);
            let texture = result.outputs[&MapKind::Height].texture();
            assert_eq!((texture.width(), texture.height()), (1024, 1024));
            assert_eq!(texture.format(), wgpu::TextureFormat::R32Float);
            let actual = gpu.read(texture);
            // The seamless intermediate stores linear RGB in binary16 before
            // default all-one height bands reconstruct its luminance.
            behavior::compare(actual.iter().map(|p| p[0]), &expected, 5e-4);
            assert!(actual.iter().flatten().all(|v| v.is_finite()));
            if let Some(previous) = previous {
                assert_eq!(actual, previous);
            }
            previous = Some(actual);
        }
        let stats = worker.diagnostics();
        assert!(stats.cache_hits > 0);
        assert!(stats.peak_bytes <= plan.peak_bytes());
        assert_eq!(stats.validation_errors, 0);
        worker.join().unwrap();
    });
}

#[test]
fn correlated_scalar_and_color_sample_source_extent_at_npot_and_singleton_proxy_centers() {
    with_gpu(|gpu| {
        let kernels = OffsetKernels::new(&gpu.device, gpu.queue.clone());
        for (source_extent, extent) in [
            (Extent::new(17, 9).unwrap(), Extent::new(7, 5).unwrap()),
            (Extent::new(1, 17).unwrap(), Extent::new(1, 5).unwrap()),
            (Extent::new(17, 1).unwrap(), Extent::new(5, 1).unwrap()),
        ] {
            let values: Vec<f32> = (0..source_extent.width * source_extent.height)
                .map(|i| [0.25, 0.5, 0.75, 1.0][(i * 7 % 4) as usize])
                .collect();
            let color: Vec<u16> = (0..values.len())
                .flat_map(|i| {
                    let v = [0x3400, 0x3800, 0x3a00, 0x3c00][i * 7 % 4];
                    [v, v, v, 0x3c00]
                })
                .collect();
            let values64: Vec<_> = values.iter().copied().map(f64::from).collect();
            let mut correlated = Vec::new();
            for (format, bytes) in [
                (wgpu::TextureFormat::R32Float, bytemuck::cast_slice(&values)),
                (
                    wgpu::TextureFormat::Rgba16Float,
                    bytemuck::cast_slice(&color),
                ),
            ] {
                let texture = |size: Extent| {
                    gpu.device.create_texture(&wgpu::TextureDescriptor {
                        label: Some("T-057 proxy fixture"),
                        size: wgpu::Extent3d {
                            width: size.width,
                            height: size.height,
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format,
                        usage: wgpu::TextureUsages::TEXTURE_BINDING
                            | wgpu::TextureUsages::STORAGE_BINDING
                            | wgpu::TextureUsages::COPY_DST
                            | wgpu::TextureUsages::COPY_SRC,
                        view_formats: &[],
                    })
                };
                let input = texture(source_extent);
                let output = texture(extent);
                gpu.queue.write_texture(
                    input.as_image_copy(),
                    bytes,
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(
                            source_extent.width
                                * if format == wgpu::TextureFormat::R32Float {
                                    4
                                } else {
                                    8
                                },
                        ),
                        rows_per_image: Some(source_extent.height),
                    },
                    input.size(),
                );
                let uniform = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("T-057 proxy uniform"),
                    size: 64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let pass = kernels
                    .prepare(
                        &gpu.device,
                        &input,
                        &output,
                        &uniform,
                        OffsetTransform::new(0.25).unwrap(),
                        Boundary::Repeat,
                    )
                    .unwrap();
                let mut results = Vec::new();
                for boundary in [Boundary::Repeat, Boundary::Clamp] {
                    for overlap in [0.1f32, 0.25, 0.5] {
                        pass.update(OffsetTransform::new(overlap).unwrap(), boundary);
                        let mut encoder = gpu.device.create_command_encoder(&Default::default());
                        pass.encode(&mut encoder);
                        gpu.queue.submit([encoder.finish()]);
                        let actual = gpu.read(&output);
                        let expected: Vec<_> = (0..extent.width * extent.height)
                            .map(|i| {
                                reference::at(
                                    source_extent,
                                    &values64,
                                    [
                                        (f64::from(i % extent.width) + 0.5)
                                            / f64::from(extent.width),
                                        (f64::from(i / extent.width) + 0.5)
                                            / f64::from(extent.height),
                                    ],
                                    f64::from(overlap),
                                    boundary,
                                )
                            })
                            .collect();
                        let tolerance = if format == wgpu::TextureFormat::R32Float {
                            2e-5
                        } else {
                            5e-4
                        };
                        for c in 0..if format == wgpu::TextureFormat::R32Float {
                            1
                        } else {
                            3
                        } {
                            behavior::compare(actual.iter().map(|p| p[c]), &expected, tolerance);
                        }
                        assert!(actual.iter().all(|p| p[3] == 1.0));
                        results.extend(actual.iter().map(|p| p[0]));
                    }
                }
                correlated.push(results);
            }
            behavior::compare(correlated[1].iter().copied(), &correlated[0], 5e-4);
        }
    });
}
