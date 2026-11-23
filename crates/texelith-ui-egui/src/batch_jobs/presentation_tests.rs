use super::*;
use crate::{
    batch_jobs::*, command_adapter::UiAction, controls::Controls, material_ui::MaterialIntegration,
};
use std::{sync::Arc, time::Instant};
use texelith_core::*;
use texelith_gpu::*;

fn device() -> (Arc<wgpu::Device>, Arc<wgpu::Queue>) {
    let backends = if cfg!(windows) {
        wgpu::Backends::DX12
    } else {
        wgpu::Backends::PRIMARY
    };
    let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    descriptor.backends = backends;
    let instance = wgpu::Instance::new(descriptor);
    let adapter = crate::material_ui::block_on(instance.enumerate_adapters(backends))
        .into_iter()
        .filter(|a| a.get_info().device_type != wgpu::DeviceType::Cpu)
        .min_by_key(|a| a.get_info().device_type != wgpu::DeviceType::DiscreteGpu)
        .expect("Native GPU required");
    eprintln!("T-087 4K regression: {:?}", adapter.get_info());
    let (d, q) = crate::material_ui::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_limits: wgpu::Limits {
            max_texture_dimension_2d: 4096,
            ..Default::default()
        },
        ..Default::default()
    }))
    .unwrap();
    (Arc::new(d), Arc::new(q))
}

fn directory(name: &str) -> std::path::PathBuf {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/evidence/T-087")
        .join(format!(
            "{name}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
    std::fs::create_dir_all(&root).unwrap();
    root
}

fn until(mut step: impl FnMut() -> bool) {
    let started = Instant::now();
    while !step() {
        assert!(
            started.elapsed() < Duration::from_secs(120),
            "worker timed out"
        );
        std::thread::sleep(Duration::from_millis(2));
    }
}

#[test]
fn four_k_import_slider_and_standalone_export_succeed_and_batch_waits() {
    if std::env::var("TEXELITH_CI_NO_GPU").as_deref() == Ok("1") {
        eprintln!(
            "SKIP GPU: no GPU adapter (TEXELITH_CI_NO_GPU=1); hardware assertions not run: {:?}",
            std::thread::current().name()
        );
        return;
    }
    let (device, queue) = device();
    let gpu =
        Arc::new(GpuWorker::spawn(device.clone(), queue.clone(), Default::default()).unwrap());
    let root = directory("four-k");
    let path = root.join("source.png");
    texelith_io::write_normal_png(&path, 4096, 4096, &[128, 64, 32].repeat(4096 * 4096), false)
        .unwrap();
    let mut controls = Controls::new_material();
    // This headless publication test has no 3D viewport render loop.
    controls.preview.split_view = false;
    controls.action(UiAction::Request(CommandRequest::SetMode {
        mode: ViewMode::TwoD,
    }));
    let mut runtime = MaterialIntegration::new(gpu.clone(), device.clone(), queue).unwrap();
    controls.open_image(path.clone());
    until(|| {
        runtime.tick(&mut controls);
        assert!(runtime.error.is_none(), "{}", runtime.status);
        runtime.publications > 0
    });
    let graph = controls.session().snapshot().unwrap().graph;
    assert_eq!(
        graph
            .material_output_extent(graph.material_outputs().unwrap()[&MapKind::Albedo])
            .unwrap(),
        Extent {
            width: 4096,
            height: 4096
        }
    );
    controls.action(UiAction::Request(CommandRequest::SelectMap {
        map: MapKind::Albedo,
    }));
    let published = runtime.publications;
    until(|| {
        runtime.tick(&mut controls);
        assert!(runtime.error.is_none(), "{}", runtime.status);
        runtime.publications > published
    });
    controls.action(UiAction::Request(CommandRequest::SelectMap {
        map: MapKind::Roughness,
    }));
    let node = controls
        .session()
        .snapshot()
        .unwrap()
        .graph
        .material_outputs()
        .unwrap()[&MapKind::Roughness]
        .node;
    let address = ParameterAddress {
        node,
        parameter: ParameterId::Curve,
    };
    controls.action(UiAction::DragBegin { address });
    controls.action(UiAction::Change {
        address,
        value: ParameterValue::Curve(vec![[0.0, 0.25], [1.0, 0.25]]),
    });
    controls.action(UiAction::DragRelease);
    let published = runtime.publications;
    until(|| {
        runtime.tick(&mut controls);
        assert!(runtime.error.is_none(), "{}", runtime.status);
        runtime.publications > published
    });
    assert_eq!(controls.session().history().undo_entries, 1);

    let mut exports = crate::export_jobs::ExportJobs::spawn(gpu.clone()).unwrap();
    controls.export_options_mut().destination = root.join("standalone");
    controls.request_material_export().unwrap();
    let encoding_contention = memory::reserve();
    exports.tick(&mut controls);
    assert!(exports.busy(), "{:?}", exports.status());
    until(|| {
        exports.tick(&mut controls);
        let status = exports.status().unwrap();
        assert!(!status.terminal(), "{status:?}");
        status.stage == ExportStage::Encoding && memory::background_waiting()
    });
    assert!(
        !root.join("standalone").exists(),
        "waiting must not publish a partial set"
    );
    drop(encoding_contention);
    until(|| {
        exports.tick(&mut controls);
        !exports.busy()
    });
    let status = exports.status().unwrap();
    assert_eq!(status.stage, ExportStage::Completed, "{status:?}");
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(status.destination.join("assignment.json")).unwrap())
            .unwrap();
    let mut rgb = 0;
    for output in manifest["outputs"].as_array().unwrap() {
        assert_eq!(output["extent"]["width"], 4096);
        assert_eq!(output["extent"]["height"], 4096);
        if output["channels"] == 3 {
            rgb += 1;
            let bytes =
                std::fs::read(status.destination.join(output["file"].as_str().unwrap())).unwrap();
            let decoded =
                texelith_io::decode_bytes(&bytes, texelith_io::ColorIntent::PhotoSrgb).unwrap();
            assert_eq!((decoded.width, decoded.height), (4096, 4096));
        }
    }
    assert!(
        rgb >= 2,
        "preset must exercise albedo and packed RGB outputs"
    );

    let input = root.join("batch.png");
    texelith_io::write_normal_png(&input, 32, 32, &[128, 64, 32].repeat(32 * 32), false).unwrap();
    let mut batch = BatchJobs::spawn(gpu.clone()).unwrap();
    let busy_memory = memory::reserve();
    controls
        .queue_batch(
            InputSelection::Files(vec![input]),
            root.join("batch"),
            root.join("job.json"),
            false,
        )
        .unwrap();
    until(|| {
        batch.tick(&mut controls, &mut exports);
        exports.tick(&mut controls);
        assert!(controls.batch_status().error.is_none());
        memory::background_waiting()
    });
    let started = Instant::now();
    while started.elapsed() < Duration::from_millis(100) {
        batch.tick(&mut controls, &mut exports);
        assert!(controls.batch_status().busy);
        assert!(
            controls.session().batch().is_none(),
            "preflight decode should wait for memory"
        );
        assert!(controls.batch_status().error.is_none());
        std::thread::sleep(Duration::from_millis(2));
    }
    drop(busy_memory);
    until(|| {
        batch.tick(&mut controls, &mut exports);
        exports.tick(&mut controls);
        !controls.batch_status().busy
    });
    assert_eq!(
        controls.session().batch().unwrap().items()[0].state,
        BatchState::Committed,
        "{:?}",
        controls.batch_status()
    );
    assert!(controls.batch_status().cpu_reserved_bytes <= CPU_BUDGET);
    drop(runtime);
    gpu.close();
    gpu.join().unwrap();
}

#[test]
fn tiled_samples_and_saves_match_numeric_oracle_including_tail_rows() {
    if std::env::var("TEXELITH_CI_NO_GPU").as_deref() == Ok("1") {
        eprintln!(
            "SKIP GPU: no GPU adapter (TEXELITH_CI_NO_GPU=1); hardware assertions not run: {:?}",
            std::thread::current().name()
        );
        return;
    }
    let (device, queue) = device();
    let root = directory("tiled-oracle");
    for (width, height) in [(1051, 131), (1, 193), (257, 1)] {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let values: Vec<_> = (0..width * height)
            .map(|i| [((i * 17) % 1021) as f32 / 1020.0, 0.0, 0.0, 1.0])
            .collect();
        let bytes: Vec<_> = values.iter().flat_map(|p| p[0].to_le_bytes()).collect();
        queue.write_texture(
            texture.as_image_copy(),
            &bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * 4),
                rows_per_image: Some(height),
            },
            texture.size(),
        );
        let (samples, nonconstant) = sample(&device, &queue, &texture).unwrap();
        assert!(nonconstant);
        let [w, h] = display_size(width, height);
        for y in 0..h {
            for x in 0..w {
                assert_eq!(
                    samples[(y * w + x) as usize],
                    values[(y * height / h * width + x * width / w) as usize]
                );
            }
        }
        for map in [
            MapKind::Height,
            MapKind::Roughness,
            MapKind::Albedo,
            MapKind::Normal,
        ] {
            let actual = root.join(format!("{width}-{height}-{map:?}.png"));
            let expected = actual.with_extension("expected.png");
            save_png(
                &device,
                &queue,
                &texture,
                &actual,
                map,
                Some(NormalConvention::DirectX),
                || true,
            )
            .unwrap();
            crate::material_ui::presenter::save_png(
                &expected,
                Extent { width, height },
                map,
                Some(NormalConvention::DirectX),
                &values,
                || true,
            )
            .unwrap();
            assert_eq!(
                std::fs::read(actual).unwrap(),
                std::fs::read(expected).unwrap()
            );
        }
        let mut invalid = bytes;
        // For the downsampled NPOT case, this last column is not displayed.
        // Validation must still inspect every source texel, including tail tiles.
        let last = invalid.len() - 4;
        invalid[last..].copy_from_slice(&f32::NAN.to_le_bytes());
        queue.write_texture(
            texture.as_image_copy(),
            &invalid,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * 4),
                rows_per_image: Some(height),
            },
            texture.size(),
        );
        assert!(sample(&device, &queue, &texture).is_err());
    }
}
