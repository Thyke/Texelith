use super::{
    gpu::{Gpu, with_gpu},
    oracle::compare,
};
use std::{
    sync::Arc,
    time::{Duration, Instant},
};
use texelith_core::{
    EvaluationSnapshot, Extent, Graph, InputInterpretation, MaterialGraphDto, MaterialNode,
    MaterialNodeParams, NodeId, NormalConvention, OutputPort, OutputRef, SourceDescriptor,
};
use texelith_gpu::{
    GpuConfig, GpuWorker, JobId, MaterialRequest, Pass, Quality, TypedSourcePixels,
    TypedSourceUpload, WorkerEvent,
};

fn linear(byte: u8) -> f64 {
    let value = f64::from(byte) / 255.0;
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

#[test]
fn signed_float_source_hashes_preserve_precision_and_canonicalize_only_signed_zero() {
    let extent = Extent::new(2, 1).unwrap();
    let hash = |values: [f32; 2]| {
        TypedSourcePixels::HeightF32(Arc::from(values))
            .descriptor(extent)
            .unwrap()
            .content_hash
    };
    assert_eq!(hash([-0.0, -0.25]), hash([0.0, -0.25]));
    assert_eq!(hash([-0.25, 2.0]), hash([-0.25, 2.0]));
    assert_ne!(hash([-0.25, 2.0]), hash([0.25, 2.0]));
    assert_ne!(
        hash([-0.25, 2.0]),
        hash([-0.25, f32::from_bits(2.0f32.to_bits() + 1)])
    );
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(
            TypedSourcePixels::HeightF32(Arc::from([bad, 0.0]))
                .descriptor(extent)
                .is_err()
        );
    }
    let rgba: Arc<[u8]> = Arc::from([0, 128, 255, 0, 255, 10, 11, 7]);
    for interpretation in [
        InputInterpretation::SrgbPhoto,
        InputInterpretation::LinearHeight,
    ] {
        assert_eq!(
            TypedSourcePixels::Rgba8 {
                pixels: rgba.clone(),
                interpretation
            }
            .descriptor(extent)
            .unwrap(),
            SourceDescriptor::from_rgba8(&rgba, extent, interpretation).unwrap()
        );
    }
    let normal = |convention| {
        TypedSourcePixels::NormalRgb8 {
            pixels: Arc::from([255, 64, 200, 20, 230, 128]),
            convention,
        }
        .descriptor(extent)
        .unwrap()
    };
    assert_ne!(
        normal(NormalConvention::DirectX).content_hash,
        normal(NormalConvention::OpenGl).content_hash
    );
}

fn convert(gpu: &Gpu, pass: Pass, bytes: &[u8], extent: Extent) -> wgpu::Texture {
    let input = gpu.upload(extent, wgpu::TextureFormat::Rgba8Unorm, bytes);
    let output = gpu.texture(extent, pass.output_format());
    let prepared = gpu.executor.prepare(pass, &input, &output, extent).unwrap();
    gpu.dispatch(&prepared);
    output
}

#[test]
fn gpu_source_preparation_decodes_photo_once_ignores_alpha_and_keeps_scalar_linear() {
    with_gpu(|gpu| {
        let codes = [
            [0, 10, 11, 0],
            [128, 64, 255, 1],
            [255, 128, 0, 255],
            [10, 11, 12, 99],
        ];
        let bytes: Vec<u8> = codes.into_iter().flatten().collect();
        let extent = Extent::new(4, 1).unwrap();
        let color = convert(gpu, Pass::ColorSource, &bytes, extent);
        let photo = convert(
            gpu,
            Pass::Source(InputInterpretation::SrgbPhoto),
            &bytes,
            extent,
        );
        let scalar = convert(
            gpu,
            Pass::Source(InputInterpretation::LinearHeight),
            &bytes,
            extent,
        );
        let actual = gpu.read_rgba(&color);
        for (pixel, code) in actual.iter().zip(codes) {
            for channel in 0..3 {
                assert!((pixel[channel] - linear(code[channel])).abs() <= 0.0005);
            }
            assert_eq!(pixel[3], 1.0);
        }
        let luminance: Vec<f64> = codes
            .iter()
            .map(|c| 0.2126 * linear(c[0]) + 0.7152 * linear(c[1]) + 0.0722 * linear(c[2]))
            .collect();
        compare(&gpu.read_scalar(&photo), &luminance, 2e-5);
        compare(
            &gpu.read_scalar(&scalar),
            &codes
                .iter()
                .map(|c| f64::from(c[0]) / 255.0)
                .collect::<Vec<_>>(),
            2e-5,
        );
        let mut opaque = bytes.clone();
        for pixel in opaque.as_chunks_mut::<4>().0 {
            pixel[3] = 255;
        }
        assert_eq!(
            gpu.read(&color),
            gpu.read(&convert(gpu, Pass::ColorSource, &opaque, extent))
        );
        assert_eq!(
            gpu.read(&photo),
            gpu.read(&convert(
                gpu,
                Pass::Source(InputInterpretation::SrgbPhoto),
                &opaque,
                extent
            ))
        );
        assert_eq!(
            gpu.read(&scalar),
            gpu.read(&convert(
                gpu,
                Pass::Source(InputInterpretation::LinearHeight),
                &opaque,
                extent
            ))
        );
    });
}

#[test]
fn gpu_imported_normals_are_signed_unit_vectors_and_preserve_edge_convention() {
    with_gpu(|gpu| {
        let codes = [[255, 64, 200, 0], [20, 230, 128, 7], [128, 128, 255, 255]];
        let bytes: Vec<u8> = codes.into_iter().flatten().collect();
        let extent = Extent::new(3, 1).unwrap();
        let mut outputs = Vec::new();
        for convention in [NormalConvention::OpenGl, NormalConvention::DirectX] {
            let output = convert(gpu, Pass::NormalSource(convention), &bytes, extent);
            for (actual, code) in gpu.read_rgba(&output).iter().zip(codes) {
                let signed = [code[0], code[1], code[2]].map(|v| f64::from(v) / 255.0 * 2.0 - 1.0);
                let length = signed.iter().map(|v| v * v).sum::<f64>().sqrt();
                for c in 0..3 {
                    assert!((actual[c] - signed[c] / length).abs() <= 0.002);
                }
                assert!(
                    (actual[..3].iter().map(|v| v * v).sum::<f64>().sqrt() - 1.0).abs() <= 0.002
                );
                assert_eq!(actual[3], 1.0);
            }
            outputs.push(gpu.read(&output));
        }
        // T-045 carries convention metadata. Canonicalization belongs to a
        // consuming node, so source preparation must not silently flip Y.
        assert_eq!(outputs[0], outputs[1]);
    });
}

fn wait<T>(mut next: impl FnMut() -> Option<T>) -> T {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        if let Some(value) = next() {
            return value;
        }
        assert!(Instant::now() < deadline, "source worker timed out");
        std::thread::yield_now();
    }
}

fn source_request(source: &SourceDescriptor, job: u64) -> MaterialRequest {
    let output = OutputRef {
        node: NodeId(1),
        port: OutputPort::Height,
    };
    let dto = MaterialGraphDto {
        schema_version: 2,
        graph: vec![MaterialNode {
            id: NodeId(1),
            inputs: vec![],
            params: MaterialNodeParams::HeightSource(source.clone()),
        }],
        outputs: [(output.port.map_kind(), output)].into(),
        selected_output: output,
    };
    MaterialRequest {
        job_id: JobId(job),
        request_id: 1,
        outputs: dto.outputs.clone(),
        extent: source.extent,
        quality: Quality::Full,
        snapshot: Arc::new(EvaluationSnapshot {
            generation: 1,
            graph: Graph::try_from(dto).unwrap(),
        }),
    }
}

#[test]
fn worker_prepares_u16_without_banding_and_f32_without_transfer_clipping_or_quantization() {
    with_gpu(|gpu| {
        let extent = Extent::new(5, 1).unwrap();
        let u16_values = [0, 32767, 32768, 32769, 65535];
        let float_values = [
            -2.0,
            -0.00001,
            0.50001,
            f32::from_bits(0.5f32.to_bits() + 1),
            2.0,
        ];
        let worker =
            GpuWorker::spawn(gpu.device.clone(), gpu.queue.clone(), GpuConfig::default()).unwrap();
        for (job, pixels, expected) in [
            (
                1,
                TypedSourcePixels::HeightU16(Arc::from(u16_values)),
                u16_values.map(|v| f64::from(v) / 65535.0),
            ),
            (
                2,
                TypedSourcePixels::HeightF32(Arc::from(float_values)),
                float_values.map(f64::from),
            ),
        ] {
            let descriptor = pixels.descriptor(extent).unwrap();
            worker
                .install_typed_source(TypedSourceUpload {
                    descriptor: descriptor.clone(),
                    pixels,
                })
                .unwrap();
            let ready = wait(|| worker.try_event());
            assert!(
                matches!(ready, WorkerEvent::SourceReady { content_hash } if content_hash == descriptor.content_hash),
                "{ready:?}"
            );
            worker
                .request_material(source_request(&descriptor, job))
                .unwrap();
            let bundle = wait(|| {
                if let Some(event) = worker.try_event() {
                    panic!("unexpected worker event: {event:?}");
                }
                worker.try_latest_material()
            });
            let texture = bundle.outputs.values().next().unwrap().texture();
            assert_eq!(texture.format(), wgpu::TextureFormat::R32Float);
            let actual = gpu.read_scalar(texture);
            compare(&actual, &expected, 1e-7);
            if job == 2 {
                assert_eq!(actual, float_values, "f32 source changed bits/value");
            } else {
                assert!(
                    actual[1] < actual[2] && actual[2] < actual[3],
                    "u16 steps banded"
                );
            }
        }
        worker.join().unwrap();
    });
}
