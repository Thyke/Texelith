//! Headless real-command M1 integration gate; no foreground window required.
use super::{ImportKind, MAPS, runtime::MaterialIntegration};
use crate::{command_adapter::UiAction, controls::Controls, startup::Config};
use std::{
    sync::Arc,
    time::{Duration, Instant},
};
use texelith_core::*;
use texelith_gpu::*;

fn dispatch(c: &mut Controls, request: CommandRequest) {
    c.action(UiAction::Request(request));
}
fn wait(
    runtime: &mut MaterialIntegration,
    c: &mut Controls,
    device: &wgpu::Device,
    mut done: impl FnMut(&MaterialIntegration, &Controls) -> bool,
) -> Result<(), String> {
    let started = Instant::now();
    loop {
        runtime.tick(c);
        device
            .poll(wgpu::PollType::Poll)
            .map_err(|e| e.to_string())?;
        if let Some(e) = &runtime.error {
            return Err(e.clone());
        }
        if done(runtime, c) {
            return Ok(());
        }
        if started.elapsed() > Duration::from_secs(90) {
            return Err(format!(
                "maps diagnostic timed out: {} / {}",
                runtime.status, runtime.save_status
            ));
        }
        std::thread::sleep(Duration::from_millis(2));
    }
}
fn select(
    runtime: &mut MaterialIntegration,
    c: &mut Controls,
    device: &wgpu::Device,
    map: MapKind,
) -> Result<(), String> {
    let previous = runtime.publications;
    dispatch(c, CommandRequest::SelectMap { map });
    wait(runtime, c, device, |r, _| r.publications > previous)
}
fn edit(
    c: &mut Controls,
    predicate: impl Fn(&MaterialNodeParams) -> bool,
    parameter: ParameterId,
    value: ParameterValue,
) -> Result<(), String> {
    let snapshot = c.session().snapshot().ok_or("No material")?;
    let node = snapshot
        .graph
        .material_nodes()
        .iter()
        .find(|n| predicate(&n.params))
        .ok_or("Missing producer")?
        .id;
    c.action(UiAction::Change {
        address: ParameterAddress { node, parameter },
        value,
    });
    Ok(())
}

pub fn run(config: &Config) -> Result<(), String> {
    let report = config.report.as_deref().ok_or("missing report path")?;
    let parent = report
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(std::path::Path::new("."));
    std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    descriptor.backends = config.backends()?;
    let instance = wgpu::Instance::new(descriptor);
    let adapters = super::block_on(instance.enumerate_adapters(config.backends()?));
    let adapter = adapters
        .into_iter()
        .find(|a| {
            config.adapter_name.as_ref().is_none_or(|name| {
                a.get_info()
                    .name
                    .to_lowercase()
                    .contains(&name.to_lowercase())
            })
        })
        .ok_or("Requested native GPU adapter unavailable")?;
    crate::startup::validate_adapter(&adapter)?;
    let info = adapter.get_info();
    let (device, queue) = super::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("M1 maps diagnostic"),
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
    let mut runtime = MaterialIntegration::new(gpu.clone(), device.clone(), queue)?;
    let mut c = Controls::new_material();
    // Numeric-map diagnostics intentionally run without a 3D presentation loop.
    c.preview.split_view = false;
    let mut checks = Vec::new();
    let result = (|| {
        let size = config.size;
        let mut pixels = Vec::with_capacity(size as usize * size as usize * 3);
        for y in 0..size {
            for x in 0..size {
                let mortar = y % 64 < 4 || (x + (y / 64 % 2) * 48) % 96 < 4;
                let grain = (3.0 * ((x as f32 * 0.07).sin() + (y as f32 * 0.06).cos()) + 6.0) as u8;
                pixels.extend_from_slice(&if mortar {
                    [64 + grain, 62 + grain, 58 + grain]
                } else {
                    [148 + grain, 88 + grain, 56 + grain]
                });
            }
        }
        let fixture = parent.join("photo-fixture.png");
        let bytes = texelith_io::encoders::encode_png8(
            size,
            size,
            texelith_io::encoders::PngChannels::Rgb,
            texelith_io::encoders::PngColorIntent::Srgb,
            &pixels,
        )
        .map_err(|e| e.to_string())?;
        texelith_io::publish_png(&fixture, &bytes).map_err(|e| e.to_string())?;
        c.open_image(fixture);
        wait(&mut runtime, &mut c, &device, |r, c| {
            r.publications > 0 && c.session().snapshot().is_some()
        })?;
        checks.push("photo-import-command-effect-typed-upload".to_string());
        for map in MAPS {
            select(&mut runtime, &mut c, &device, map)?;
            if matches!(
                map,
                MapKind::Albedo | MapKind::Height | MapKind::Normal | MapKind::Roughness
            ) && !runtime.last_nonconstant
            {
                return Err(format!("{map:?} unexpectedly constant"));
            }
            checks.push(format!("{map:?}-real-gpu-preview"));
            let path = parent.join(format!("{map:?}.png"));
            let _ = std::fs::remove_file(&path);
            runtime.save_status.clear();
            c.save_png(path.clone(), map);
            wait(&mut runtime, &mut c, &device, |r, _| {
                r.save_status.starts_with("Saved ")
            })?;
            let bytes = std::fs::read(&path).map_err(|e| e.to_string())?;
            if u32::from_be_bytes(bytes[16..20].try_into().unwrap()) != size
                || u32::from_be_bytes(bytes[20..24].try_into().unwrap()) != size
            {
                return Err("Saved size mismatch".into());
            }
            let expected = if map == MapKind::Height {
                [16, 0]
            } else if matches!(map, MapKind::Normal | MapKind::Albedo) {
                [8, 2]
            } else {
                [8, 0]
            };
            if bytes[24..26] != expected {
                return Err(format!("{map:?} wrong PNG depth/channels"));
            }
            checks.push(format!("{map:?}-full-numeric-png-command-effect"));
        }
        edit(
            &mut c,
            |p| matches!(p, MaterialNodeParams::HeightFromAlbedo { .. }),
            ParameterId::CoarseWeight,
            ParameterValue::Float(0.8),
        )?;
        edit(
            &mut c,
            |p| matches!(p, MaterialNodeParams::NormalFromHeight { .. }),
            ParameterId::Filter,
            ParameterValue::Filter(NormalFilter::Scharr),
        )?;
        edit(
            &mut c,
            |p| matches!(p, MaterialNodeParams::HorizonAo { .. }),
            ParameterId::RadiusSourcePx,
            ParameterValue::Float(16.0),
        )?;
        edit(
            &mut c,
            |p| matches!(p, MaterialNodeParams::Curvature { .. }),
            ParameterId::EdgeScale,
            ParameterValue::Float(0.5),
        )?;
        edit(
            &mut c,
            |p| matches!(p, MaterialNodeParams::Roughness { .. }),
            ParameterId::Invert,
            ParameterValue::Bool(true),
        )?;
        edit(
            &mut c,
            |p| matches!(p, MaterialNodeParams::Metallic { .. }),
            ParameterId::Radius,
            ParameterValue::Float(0.5),
        )?;
        for map in MAPS {
            select(&mut runtime, &mut c, &device, map)?;
        }
        checks.push("all-producer-inspector-edits-evaluated".into());
        let normal = c
            .session()
            .snapshot()
            .unwrap()
            .graph
            .material_nodes()
            .iter()
            .find(|n| matches!(n.params, MaterialNodeParams::NormalFromHeight { .. }))
            .unwrap()
            .id;
        let address = ParameterAddress {
            node: normal,
            parameter: ParameterId::Strength,
        };
        c.action(UiAction::DragBegin { address });
        c.action(UiAction::Change {
            address,
            value: ParameterValue::Float(0.25),
        });
        select(&mut runtime, &mut c, &device, MapKind::Normal)?;
        c.action(UiAction::DragRelease);
        select(&mut runtime, &mut c, &device, MapKind::Normal)?;
        c.action(UiAction::UndoShortcut);
        select(&mut runtime, &mut c, &device, MapKind::Normal)?;
        c.action(UiAction::RedoShortcut);
        select(&mut runtime, &mut c, &device, MapKind::Normal)?;
        checks.push("proxy-release-undo-redo-real-evaluation".into());
        for seamless in [false, true] {
            dispatch(
                &mut c,
                CommandRequest::ConfigureMaterial {
                    seamless,
                    painted_mask: false,
                },
            );
            select(&mut runtime, &mut c, &device, MapKind::Albedo)?;
        }
        edit(
            &mut c,
            |p| matches!(p, MaterialNodeParams::Seamless { .. }),
            ParameterId::SeamlessMethod,
            ParameterValue::SeamlessMethod(SeamlessMethod::Splat {
                seed: 61,
                grid: 4,
                overlap: 0.25,
            }),
        )?;
        select(&mut runtime, &mut c, &device, MapKind::Albedo)?;
        checks.push("offset-original-periodic-splat-command-routes".into());
        for base in [
            SeamlessBase::OffsetBlend { overlap: 0.25 },
            SeamlessBase::Splat {
                seed: 61,
                grid: 4,
                overlap: 0.25,
            },
        ] {
            edit(
                &mut c,
                |p| matches!(p, MaterialNodeParams::Seamless { .. }),
                ParameterId::SeamlessMethod,
                ParameterValue::SeamlessMethod(SeamlessMethod::Histogram {
                    base: base.clone(),
                    strength: 1.0,
                }),
            )?;
            select(&mut runtime, &mut c, &device, MapKind::Albedo)?;
            if !runtime.last_nonconstant {
                return Err("Histogram Albedo unexpectedly constant".into());
            }
            checks.push(format!("Histogram-{base:?}-real-gpu-preview"));
        }
        dispatch(&mut c, CommandRequest::InspectSeams { enabled: true });
        select(&mut runtime, &mut c, &device, MapKind::Albedo)?;
        checks.push("gpu-seam-heatmap-and-finite-reduction".into());
        dispatch(&mut c, CommandRequest::InspectSeams { enabled: false });
        let mask = c
            .session()
            .snapshot()
            .unwrap()
            .graph
            .material_outputs()
            .unwrap()[&MapKind::Mask]
            .node;
        let stroke_id = StrokeId(61);
        dispatch(
            &mut c,
            CommandRequest::BeginStroke {
                node: mask,
                stroke_id,
                brush: Brush {
                    radius: 0.2,
                    hardness: 0.5,
                    opacity: 1.0,
                    mode: MaskMode::Replace,
                },
                sample: BrushSample {
                    uv: [0.5, 0.5],
                    pressure: 1.0,
                },
            },
        );
        dispatch(&mut c, CommandRequest::CommitStroke { stroke_id });
        dispatch(
            &mut c,
            CommandRequest::ConfigureMaterial {
                seamless: true,
                painted_mask: true,
            },
        );
        for map in [
            MapKind::Mask,
            MapKind::Roughness,
            MapKind::Metallic,
            MapKind::Curvature,
            MapKind::Cavity,
        ] {
            select(&mut runtime, &mut c, &device, map)?;
        }
        checks.push("paint-mask-optional-links-and-secondary-curvature-ports".into());
        let before = c.session().snapshot().unwrap();
        let history = c.session().history();
        c.open_image(parent.join("missing-input.png"));
        wait(&mut runtime, &mut c, &device, |r, _| {
            r.status.starts_with("Import failed")
        })?;
        if c.session().snapshot() != Some(before) || c.session().history() != history {
            return Err("Failed import changed material/history".into());
        }
        checks.push("failed-import-retains-material-and-history".into());
        for convention in [NormalConvention::OpenGl, NormalConvention::DirectX] {
            c.import_kind = ImportKind::Normal;
            c.normal_convention = convention;
            c.open_image(parent.join("Normal.png"));
            let previous = runtime.publications;
            wait(&mut runtime, &mut c, &device, |r, c| {
                r.publications > previous
                    && c.session().snapshot().is_some_and(|s| {
                        s.graph
                            .material_nodes()
                            .iter()
                            .any(|n| matches!(n.params, MaterialNodeParams::NormalSource { .. }))
                    })
            })?;
            if c.session().history().undo_entries != 0 {
                return Err("Imported convention polluted new-document history".into());
            }
            select(&mut runtime, &mut c, &device, MapKind::Edge)?;
            let snapshot = c.session().snapshot().unwrap();
            let height = snapshot.graph.material_outputs().unwrap()[&MapKind::Height];
            if !matches!(
                snapshot.graph.material_node(height.node).unwrap().params,
                MaterialNodeParams::HeightFromNormal { .. }
            ) || !super::supported(&snapshot, MapKind::Height)
            {
                return Err(
                    "Imported normal is missing its supported HeightFromNormal output".into(),
                );
            }
            select(&mut runtime, &mut c, &device, MapKind::Height)?;
            if !runtime.last_nonconstant {
                return Err("HeightFromNormal unexpectedly constant".into());
            }
            checks.push(format!(
                "{convention:?}-LinearNormal-HeightFromNormal-real-gpu-preview"
            ));
        }
        c.import_kind = ImportKind::Height;
        c.open_image(parent.join("Height.png"));
        let previous = runtime.publications;
        wait(&mut runtime, &mut c, &device, |r, _| {
            r.publications > previous
        })?;
        select(&mut runtime, &mut c, &device, MapKind::Height)?;
        checks.push("linear-png16-import-separate-height-branch".into());
        let exr = parent.join("height-fixture.exr");
        let values: Vec<f32> = (0..size * size)
            .map(|index| (index % size) as f32 / size as f32)
            .collect();
        let bytes =
            texelith_io::encoders::encode_exr32(size, size, &values).map_err(|e| e.to_string())?;
        std::fs::write(&exr, bytes).map_err(|e| e.to_string())?;
        c.open_image(exr);
        let previous = runtime.publications;
        wait(&mut runtime, &mut c, &device, |r, _| {
            r.publications > previous
        })?;
        select(&mut runtime, &mut c, &device, MapKind::Height)?;
        checks.push("linear-exr32-import-real-height-branch".into());

        if checks.len() < 20 {
            return Err("Zero/incomplete assertions".into());
        }
        Ok(())
    })();
    let diagnostics = gpu.diagnostics();
    let passed = result.is_ok() && diagnostics.validation_errors == 0;
    let value = serde_json::json!({"task":"T-122","scenario":"maps","passed":passed,"size":config.size,"adapter":{"name":info.name,"backend":format!("{:?}",info.backend)},"assertions":checks.len(),"checks":checks,"error":result.as_ref().err(),"runtime":runtime.report(),"validation_errors":diagnostics.validation_errors,"peak_bytes":diagnostics.peak_bytes,"exclusions":[],"native_input":"not measured; separate orchestrator visual/interaction gate"});
    std::fs::write(
        report,
        serde_json::to_vec_pretty(&value).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    drop(runtime);
    gpu.close();
    gpu.join().map_err(|e| e.to_string())?;
    if !passed {
        return Err(result
            .err()
            .unwrap_or_else(|| "GPU validation errors".into()));
    }
    Ok(())
}
