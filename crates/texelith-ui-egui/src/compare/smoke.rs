use super::*;
use crate::{command_adapter::UiAction, controls::Controls, startup::Config};
use std::{
    sync::Arc,
    time::{Duration, Instant},
};
use texelith_core::*;
use texelith_gpu::*;

pub fn run(config: &Config) -> Result<(), String> {
    let report = config.report.as_deref().ok_or("Missing report")?;
    let dir = report.parent().unwrap_or(std::path::Path::new("."));
    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
    desc.backends = config.backends()?;
    let instance = wgpu::Instance::new(desc);
    let adapter = crate::material_ui::block_on(instance.enumerate_adapters(config.backends()?))
        .into_iter()
        .find(|a| {
            config.adapter_name.as_ref().is_none_or(|name| {
                a.get_info()
                    .name
                    .to_lowercase()
                    .contains(&name.to_lowercase())
            })
        })
        .ok_or("Requested GPU unavailable")?;
    crate::startup::validate_adapter(&adapter)?;
    let info = adapter.get_info();
    let (device, queue) =
        crate::material_ui::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_limits: wgpu::Limits {
                max_texture_dimension_2d: 4096,
                ..Default::default()
            },
            ..Default::default()
        }))
        .map_err(|e| e.to_string())?;
    let device = Arc::new(device);
    let queue = Arc::new(queue);
    let gpu = Arc::new(
        GpuWorker::spawn(device.clone(), queue.clone(), Default::default())
            .map_err(|e| e.to_string())?,
    );
    let mut runtime =
        crate::material_ui::MaterialIntegration::new(gpu.clone(), device.clone(), queue.clone())?;
    let mut c = Controls::new_material();
    let mut checks = Vec::new();
    let mut camera_perf = serde_json::Value::Null;
    let result: Result<(), String> = (|| {
        let size = config.size;
        let pixels: Vec<_> = (0..size * size)
            .flat_map(|i| {
                let y = i / size;
                let x = i % size + (y / 64 % 2) * 64;
                if y % 64 < 5 || x % 128 < 5 {
                    [95, 87, 76]
                } else {
                    let grain = ((i * 13 + y * 7) % 17) as u8;
                    [155 + grain, 72 + grain, 42 + grain]
                }
            })
            .collect();
        let png = texelith_io::encoders::encode_png8(
            size,
            size,
            texelith_io::encoders::PngChannels::Rgb,
            texelith_io::encoders::PngColorIntent::Srgb,
            &pixels,
        )
        .map_err(|e| e.to_string())?;
        let fixture = dir.join("viewport-fixture.png");
        texelith_io::publish_png(&fixture, &png).map_err(|e| e.to_string())?;
        c.open_image(fixture);
        wait(&mut runtime, &mut c, &device, |r| r.publications > 0)?;
        dispatch(
            &mut c,
            CommandRequest::SetMode {
                mode: ViewMode::ThreeD,
            },
        );
        for mesh in [
            PreviewMesh::PreviewBall,
            PreviewMesh::RoundedCube,
            PreviewMesh::Plane,
            PreviewMesh::Cube,
            PreviewMesh::Sphere,
            PreviewMesh::Cylinder,
        ] {
            dispatch(&mut c, CommandRequest::SetMesh { mesh: mesh.clone() });
            dispatch(
                &mut c,
                CommandRequest::SetCamera {
                    camera: super::frame_camera(&mesh, 1024. / 768.),
                },
            );
            let frame = frame(&mut runtime, &mut c, &device)?.0;
            let bytes = readback(&device, &queue, &frame)?;
            require(
                bytes.as_chunks::<4>().0.iter().any(|p| p != &bytes[..4]),
                "Primitive preview was constant",
            )?;
            let png = texelith_io::encoders::encode_png8(
                frame.extent.width,
                frame.extent.height,
                texelith_io::encoders::PngChannels::Rgba,
                texelith_io::encoders::PngColorIntent::Srgb,
                &bytes,
            )
            .map_err(|e| e.to_string())?;
            texelith_io::publish_png(&dir.join(format!("{mesh:?}.png")), &png)
                .map_err(|e| e.to_string())?;
            checks.push(format!("{mesh:?}-command-real-render"));
        }
        let mut mesh_measurements = Vec::new();
        for mesh in [
            PreviewMesh::PreviewBall,
            PreviewMesh::RoundedCube,
            PreviewMesh::Sphere,
        ] {
            mesh_measurements.push(camera_performance(
                &mut runtime,
                &mut c,
                &device,
                &gpu,
                mesh,
            )?);
        }
        camera_perf = mesh_measurements[0].clone();
        camera_perf["mesh_cases"] = serde_json::json!(mesh_measurements);
        checks.push("1080p-camera-only-no-material-evaluation-idle-no-submission".into());
        let obj = dir.join("triangle.obj");
        std::fs::write(
            &obj,
            "v -1 -1 0\nv 1 -1 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 0.5 1\nf 1/1 2/2 3/3\n",
        )
        .map_err(|e| e.to_string())?;
        c.preview.pending = Some(assets::Load::Mesh(obj));
        wait(&mut runtime, &mut c, &device, |_| true)?;
        wait_controls(&mut runtime, &mut c, &device, |c| {
            matches!(c.session().view_state().mesh, PreviewMesh::Custom(_))
        })?;
        frame(&mut runtime, &mut c, &device)?;
        checks.push("custom-obj-bounded-io-command-render".into());
        let previous_mesh = c.session().view_state().mesh.clone();
        require(
            c.preview.imported_mesh.as_ref() == Some(&previous_mesh),
            "Imported mesh missing from selector",
        )?;
        dispatch(
            &mut c,
            CommandRequest::SetMesh {
                mesh: PreviewMesh::PreviewBall,
            },
        );
        frame(&mut runtime, &mut c, &device)?;
        dispatch(
            &mut c,
            CommandRequest::SetMesh {
                mesh: previous_mesh.clone(),
            },
        );
        frame(&mut runtime, &mut c, &device)?;
        checks.push("imported-mesh-reselection-after-built-in".into());
        let geometry: [f32; 15] = [
            -1., -1., 0., 1., -1., 0., 0., 1., 0., 0., 0., 1., 0., 0.5, 1.,
        ];
        let bin: Vec<u8> = geometry.into_iter().flat_map(f32::to_le_bytes).collect();
        std::fs::write(dir.join("triangle.bin"), bin).map_err(|e| e.to_string())?;
        let gltf = serde_json::json!({"asset":{"version":"2.0"},"buffers":[{"uri":"triangle.bin","byteLength":60}],"bufferViews":[{"buffer":0,"byteOffset":0,"byteLength":36},{"buffer":0,"byteOffset":36,"byteLength":24}],"accessors":[{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3","min":[-1,-1,0],"max":[1,1,0]},{"bufferView":1,"componentType":5126,"count":3,"type":"VEC2"}],"meshes":[{"primitives":[{"attributes":{"POSITION":0,"TEXCOORD_0":1},"mode":4}]}],"nodes":[{"mesh":0}],"scenes":[{"nodes":[0]}],"scene":0});
        let path = dir.join("triangle.gltf");
        std::fs::write(&path, serde_json::to_vec(&gltf).unwrap()).map_err(|e| e.to_string())?;
        c.preview.pending = Some(assets::Load::Mesh(path));
        wait_controls(&mut runtime, &mut c, &device, |c| {
            c.session().view_state().mesh != previous_mesh
        })?;
        frame(&mut runtime, &mut c, &device)?;
        checks.push("custom-gltf-local-buffer-io-command-render".into());

        let hdr = dir.join("constant.hdr");
        let mut hdr_bytes = b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 2 +X 4\n".to_vec();
        hdr_bytes.extend([128, 96, 64, 130].repeat(8));
        std::fs::write(&hdr, hdr_bytes).map_err(|e| e.to_string())?;
        c.preview.pending = Some(assets::Load::Hdr(hdr));
        wait_controls(&mut runtime, &mut c, &device, |c| {
            c.session().view_state().light.environment.is_some()
        })?;
        frame(&mut runtime, &mut c, &device)?;
        checks.push("hdr-bounded-io-light-command-render".into());
        dispatch(
            &mut c,
            CommandRequest::SetMesh {
                mesh: PreviewMesh::Plane,
            },
        );
        dispatch(
            &mut c,
            CommandRequest::SetCamera {
                camera: Camera {
                    yaw: 0.35,
                    pitch: 0.2,
                    ..Default::default()
                },
            },
        );
        dispatch(
            &mut c,
            CommandRequest::SetPom {
                pom: Pom {
                    enabled: true,
                    scale: 0.07,
                },
            },
        );
        for count in 1..=3 {
            dispatch(
                &mut c,
                CommandRequest::SetTiles {
                    tiles: Tiles {
                        count,
                        seed: 123,
                        random_rotation: true,
                    },
                },
            );
            frame(&mut runtime, &mut c, &device)?;
        }
        checks.push("pom-camera-1x-2x2-3x3-seeded-command-render".into());
        let snapshot = c.session().snapshot().unwrap();
        let node = snapshot
            .graph
            .material_nodes()
            .iter()
            .find(|n| matches!(n.params, MaterialNodeParams::NormalFromHeight { .. }))
            .ok_or("No normal producer")?
            .id;
        // Pin the fixture's asserted A value independently of photo defaults.
        dispatch(
            &mut c,
            CommandRequest::SetParameter {
                address: ParameterAddress {
                    node,
                    parameter: ParameterId::Strength,
                },
                value: ParameterValue::Float(1.),
                gesture: None,
            },
        );
        dispatch(&mut c, CommandRequest::CaptureA {});
        dispatch(
            &mut c,
            CommandRequest::SetParameter {
                address: ParameterAddress {
                    node,
                    parameter: ParameterId::Strength,
                },
                value: ParameterValue::Float(3.),
                gesture: None,
            },
        );
        let roughness = snapshot
            .graph
            .material_nodes()
            .iter()
            .find(|n| matches!(n.params, MaterialNodeParams::Roughness { .. }))
            .ok_or("No roughness producer")?
            .id;
        dispatch(
            &mut c,
            CommandRequest::SetParameter {
                address: ParameterAddress {
                    node: roughness,
                    parameter: ParameterId::Curve,
                },
                value: ParameterValue::Curve(vec![[0., 0.95], [1., 0.95]]),
                gesture: None,
            },
        );
        require(
            c.session()
                .captured_a()
                .unwrap()
                .graph
                .material_node(node)
                .unwrap()
                .params
                .parameter(ParameterId::Strength)
                == Ok(ParameterValue::Float(1.)),
            "A recipe changed with B edits",
        )?;
        dispatch(&mut c, CommandRequest::CaptureB {});
        c.preview.compare_enabled = true;
        let (a, b) = frame(&mut runtime, &mut c, &device)?;
        let b = b.ok_or("A/B pair missing")?;
        require(
            a.extent.width <= 1024 && b.extent.width <= 1024 && a.extent == b.extent,
            "A/B proxy bounds",
        )?;
        require(
            readback(&device, &queue, &a)? != readback(&device, &queue, &b)?,
            "Different A/B recipes rendered identically",
        )?;
        drop(a);
        drop(b);
        let compare_material_requests = runtime.viewport.material_requests;
        dispatch(
            &mut c,
            CommandRequest::SetCamera {
                camera: Camera {
                    yaw: -0.25,
                    ..Default::default()
                },
            },
        );
        let pair = frame(&mut runtime, &mut c, &device)?;
        require(
            pair.1.is_some(),
            "Camera edit did not regenerate both halves",
        )?;
        require(
            runtime.viewport.material_requests == compare_material_requests,
            "A/B camera motion re-evaluated material",
        )?;
        checks.push("captured-recipes-shared-camera-exposure-matching-proxy-pair".into());
        dispatch(
            &mut c,
            CommandRequest::SetMode {
                mode: ViewMode::TwoD,
            },
        );
        frame(&mut runtime, &mut c, &device)?;
        checks.push("2d-seeded-split-render".into());
        Ok(())
    })();
    let d = gpu.diagnostics();
    let passed = result.is_ok() && d.validation_errors == 0 && d.peak_bytes <= 1024 * 1024 * 1024;
    let value = serde_json::json!({"task":"T-069","scenario":"viewport","passed":passed,"adapter":info.name,"backend":format!("{:?}",info.backend),"checks":checks,"peak_bytes":d.peak_bytes,"validation_errors":d.validation_errors,"error":result.as_ref().err(),"camera_perf":camera_perf,"native_input":"deferred to orchestrator"});
    std::fs::write(report, serde_json::to_vec_pretty(&value).unwrap())
        .map_err(|e| e.to_string())?;
    gpu.close();
    gpu.join().map_err(|e| e.to_string())?;
    result?;
    require(passed, "Viewport validation or budget failed")
}
fn dispatch(c: &mut Controls, r: CommandRequest) {
    c.action(UiAction::Request(r));
}
fn require(ok: bool, message: &str) -> Result<(), String> {
    if ok { Ok(()) } else { Err(message.into()) }
}
fn wait(
    r: &mut crate::material_ui::MaterialIntegration,
    c: &mut Controls,
    d: &wgpu::Device,
    mut done: impl FnMut(&crate::material_ui::MaterialIntegration) -> bool,
) -> Result<(), String> {
    let start = Instant::now();
    loop {
        r.tick(c);
        d.poll(wgpu::PollType::Poll).map_err(|e| e.to_string())?;
        if let Some(e) = r.error.as_ref().or(r.viewport.error.as_ref()) {
            return Err(e.clone());
        }
        if done(r) {
            return Ok(());
        }
        if start.elapsed() > Duration::from_secs(90) {
            return Err(format!(
                "Viewport timeout: {} / {}",
                r.status, r.viewport.status
            ));
        }
        std::thread::sleep(Duration::from_millis(2));
    }
}
fn wait_controls(
    r: &mut crate::material_ui::MaterialIntegration,
    c: &mut Controls,
    d: &wgpu::Device,
    done: impl Fn(&Controls) -> bool,
) -> Result<(), String> {
    let start = Instant::now();
    loop {
        wait(r, c, d, |_| true)?;
        if done(c) {
            return Ok(());
        }
        if start.elapsed() > Duration::from_secs(30) {
            return Err("Asset command timed out".into());
        }
        std::thread::sleep(Duration::from_millis(2));
    }
}
fn frame(
    r: &mut crate::material_ui::MaterialIntegration,
    c: &mut Controls,
    d: &wgpu::Device,
) -> Result<(CompletedPreview, Option<CompletedPreview>), String> {
    let mut result = None;
    let start = Instant::now();
    while result.is_none() {
        wait(r, c, d, |_| true)?;
        result = r.viewport.take_ready();
        if start.elapsed() > Duration::from_secs(90) {
            return Err(format!("Frame timeout: {}", r.viewport.status));
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    Ok(result.unwrap())
}
pub(crate) fn readback(
    d: &wgpu::Device,
    q: &wgpu::Queue,
    f: &CompletedPreview,
) -> Result<Vec<u8>, String> {
    let stride = (f.extent.width * 4).div_ceil(256) * 256;
    let buffer = d.create_buffer(&wgpu::BufferDescriptor {
        label: Some("viewport diagnostic readback"),
        size: u64::from(stride) * u64::from(f.extent.height),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = d.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        f.lease.texture().as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(stride),
                rows_per_image: Some(f.extent.height),
            },
        },
        wgpu::Extent3d {
            width: f.extent.width,
            height: f.extent.height,
            depth_or_array_layers: 1,
        },
    );
    q.submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    buffer.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    d.poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| e.to_string())?;
    rx.recv()
        .map_err(|e| e.to_string())?
        .map_err(|e| e.to_string())?;
    let data = buffer
        .slice(..)
        .get_mapped_range()
        .map_err(|e| e.to_string())?;
    let mut bytes = Vec::new();
    for row in data.chunks_exact(stride as usize) {
        bytes.extend_from_slice(&row[..f.extent.width as usize * 4]);
    }
    drop(data);
    buffer.unmap();
    Ok(bytes)
}

fn camera_performance(
    r: &mut crate::material_ui::MaterialIntegration,
    c: &mut Controls,
    d: &wgpu::Device,
    gpu: &GpuWorker,
    mesh: PreviewMesh,
) -> Result<serde_json::Value, String> {
    r.viewport.set_extent(Extent::new(1920, 1080).unwrap());
    dispatch(c, CommandRequest::SetMesh { mesh: mesh.clone() });
    dispatch(
        c,
        CommandRequest::SetCamera {
            camera: super::frame_camera(&mesh, 1920. / 1080.),
        },
    );
    let mut samples = Vec::new();
    let mut materials = 0;
    for index in 0..106 {
        let camera = super::orbit(
            c.session().view_state().camera,
            [2.5, (index as f32 * 0.1).cos()],
            if index % 20 < 10 { 1. } else { -1. },
        );
        dispatch(c, CommandRequest::SetCamera { camera });
        let started = Instant::now();
        let rendered = frame(r, c, d)?.0;
        require(
            rendered.extent == Extent::new(1920, 1080).unwrap(),
            "Performance frame was not 1080p",
        )?;
        if index == 15 {
            materials = r.viewport.material_requests;
        }
        if index >= 16 {
            samples.push(started.elapsed().as_secs_f64() * 1000.);
        }
    }
    require(
        r.viewport.material_requests == materials,
        "Camera motion re-evaluated material",
    )?;
    // Let finite auxiliary thumbnail/readback work drain before measuring idle.
    for _ in 0..200 {
        wait(r, c, d, |_| true)?;
        std::thread::sleep(Duration::from_millis(2));
    }
    let idle_submitted = gpu.diagnostics().submitted;
    let idle_publications = r.viewport.publications;
    for _ in 0..30 {
        wait(r, c, d, |_| true)?;
        std::thread::sleep(Duration::from_millis(2));
    }
    require(
        gpu.diagnostics().submitted == idle_submitted
            && r.viewport.publications == idle_publications,
        "Idle viewport submitted work",
    )?;
    require(
        r.viewport_can_sleep(c),
        "Idle viewport still requests UI polling",
    )?;
    samples.sort_by(f64::total_cmp);
    let p95 = samples[(samples.len() * 95).div_ceil(100) - 1];
    // Native presentation/input cadence remains the orchestrator's window gate.
    Ok(
        serde_json::json!({"mesh":format!("{mesh:?}"),"extent":[1920,1080],"samples":samples.len(),"p95_ms":p95,
        "target_ms":16.7,"under_target":p95 < 16.7,"material_requests_during_camera_motion":0,
        "idle_submissions":0,"includes":"UI scheduling, worker, GPU completion; excludes native presentation"}),
    )
}
