use crate::gpu_availability;

use super::*;
use crate::material_ui::MAPS;
use std::collections::BTreeMap;

type Maps = BTreeMap<MapKind, Vec<[f32; 4]>>;

fn brick(extent: Extent) -> TypedSourceUpload {
    let pixels = TypedSourcePixels::Rgba8 {
        pixels: (0..extent.width * extent.height)
            .flat_map(|i| {
                let (x, y) = (i % extent.width, i / extent.width);
                let row = y / 64;
                let u = (x + (row % 2) * 64) % 128;
                let v = y % 64;
                let grain = ((x.wrapping_mul(73856093) ^ y.wrapping_mul(19349663)) % 25) as u8;
                if u < 6 || v < 6 {
                    [52 + grain / 3, 49 + grain / 3, 45 + grain / 3, 255]
                } else {
                    let variation = ((x / 128 + row * 3) % 5) as u8 * 5;
                    [135 + variation + grain, 66 + grain, 43 + grain / 2, 255]
                }
            })
            .collect::<Vec<_>>()
            .into(),
        interpretation: InputInterpretation::SrgbPhoto,
    };
    TypedSourceUpload {
        descriptor: pixels.descriptor(extent).unwrap(),
        pixels,
    }
}

fn install(h: &Harness, upload: TypedSourceUpload) {
    h.runtime.gpu.install_typed_source(upload).unwrap();
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        if let Some(event) = h.runtime.gpu.try_event() {
            assert!(matches!(event, WorkerEvent::SourceReady { .. }));
            return;
        }
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(1));
    }
}

// Reconstruct the pre-T-143 factory without retaining a second production preset.
fn before(graph: Graph) -> Graph {
    let mut dto = MaterialGraphDto::from(graph);
    let remap = dto
        .graph
        .iter()
        .find(|n| matches!(n.params, MaterialNodeParams::NormalizeHeight {}))
        .unwrap();
    let (id, raw) = (remap.id, remap.inputs[0].source);
    dto.graph.retain(|n| n.id != id);
    for node in &mut dto.graph {
        for input in &mut node.inputs {
            if input.source.node == id {
                input.source = raw;
            }
        }
        match &mut node.params {
            MaterialNodeParams::HeightFromAlbedo { band_weights, .. } => *band_weights = vec![1.0],
            MaterialNodeParams::NormalFromHeight {
                strength,
                scale_weights,
                ..
            } => {
                *strength = 1.0;
                *scale_weights = vec![1.0];
            }
            MaterialNodeParams::Roughness { curve, base, .. } => {
                *curve = vec![[0.0, 0.0], [1.0, 1.0]];
                *base = 0.5;
            }
            MaterialNodeParams::Metallic { mask_mode, .. } => *mask_mode = MaskMode::Replace,
            _ => {}
        }
    }
    dto.outputs.insert(MapKind::Height, raw);
    Graph::try_from(dto).unwrap()
}

fn evaluate(h: &Harness, graph: Graph, serial: u64, quality: Quality) -> CompletedMaterial {
    let snapshot = EvaluationSnapshot {
        generation: serial,
        graph,
    };
    let mut request =
        super::super::super::request(snapshot.clone(), MapKind::Height, serial, quality).unwrap();
    request.outputs = MAPS
        .into_iter()
        .map(|map| (map, snapshot.graph.material_outputs().unwrap()[&map]))
        .collect();
    h.runtime.gpu.request_material(request).unwrap();
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        if let Some(event) = h.runtime.gpu.try_event() {
            panic!("{event:?}");
        }
        if let Some(bundle) = h.runtime.gpu.try_latest_material() {
            return bundle;
        }
        assert!(Instant::now() < deadline, "default maps timed out");
        std::thread::sleep(Duration::from_millis(1));
    }
}

fn read(h: &Harness, bundle: &CompletedMaterial) -> Maps {
    MAPS.into_iter()
        .map(|map| {
            (
                map,
                presenter::read_texture(&h.device, &h.queue, bundle.outputs[&map].texture())
                    .unwrap(),
            )
        })
        .collect()
}

fn range(pixels: &[[f32; 4]]) -> (f32, f32) {
    pixels
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), p| {
            (lo.min(p[0]), hi.max(p[0]))
        })
}

fn normal_detail(pixels: &[[f32; 4]], width: usize) -> f32 {
    pixels
        .windows(2)
        .enumerate()
        .filter(|(i, _)| (i + 1) % width != 0)
        .map(|(_, pair)| (0..3).map(|c| (pair[0][c] - pair[1][c]).abs()).sum::<f32>())
        .sum::<f32>()
        / pixels.len() as f32
}

fn sheet(h: &Harness, bundle: &CompletedMaterial, maps: &Maps, old: bool) -> Vec<u8> {
    let mut images = Vec::new();
    for map in MAPS {
        if old {
            let pixels = &maps[&map];
            let width = bundle.extent.width as usize;
            let height = bundle.extent.height as usize;
            images.push(egui::ColorImage::new(
                [128, 128],
                (0..128 * 128)
                    .map(|i| {
                        let x = ((2 * (i % 128) + 1) * width / 256).min(width - 1);
                        let y = ((2 * (i / 128) + 1) * height / 256).min(height - 1);
                        let p = pixels[y * width + x];
                        let rgb = match map {
                            MapKind::Albedo | MapKind::Normal => {
                                presenter::rgb(p, map, bundle.normal_convention)
                            }
                            _ => [(p[0].clamp(0.0, 1.0) * 255.0).round() as u8; 3],
                        };
                        egui::Color32::from_rgb(rgb[0], rgb[1], rgb[2])
                    })
                    .collect(),
            ));
        } else {
            let one = CompletedMaterial {
                job_id: bundle.job_id,
                request_id: bundle.request_id,
                generation: bundle.generation,
                extent: bundle.extent,
                quality: bundle.quality,
                normal_convention: bundle.normal_convention,
                outputs: [(map, bundle.outputs[&map].clone())].into(),
            };
            images.push(
                presenter::present_thumbnail(
                    &h.device,
                    &h.queue,
                    one,
                    map,
                    &AtomicU64::new(bundle.generation),
                )
                .unwrap()
                .unwrap(),
            );
        }
    }
    let rgb: Vec<_> = (0..128)
        .flat_map(|y| {
            images.iter().flat_map(move |image| {
                image.pixels[y * 128..(y + 1) * 128]
                    .iter()
                    .flat_map(|p| [p.r(), p.g(), p.b()])
            })
        })
        .collect();
    texelith_io::encoders::encode_png8(
        896,
        128,
        texelith_io::encoders::PngChannels::Rgb,
        texelith_io::encoders::PngColorIntent::Srgb,
        &rgb,
    )
    .unwrap()
}

#[test]
fn photo_defaults_show_brick_shapes_and_dielectric_maps() {
    if gpu_availability::skip_if_no_gpu() {
        return;
    }
    let _serial = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let h = Harness::new();
    let evidence = std::env::var_os("TEXELITH_T143_EVIDENCE").map(std::path::PathBuf::from);
    let extent = if evidence.is_some() {
        Extent::new(1024, 1024).unwrap()
    } else {
        Extent::new(257, 193).unwrap()
    };
    let upload = brick(extent);
    install(&h, upload.clone());
    let graph = demo_material_from_source(upload.descriptor.clone()).unwrap();
    let old = evaluate(&h, before(graph.clone()), 1, Quality::Full);
    let old_maps = read(&h, &old);
    let old_sheet = evidence.as_ref().map(|_| sheet(&h, &old, &old_maps, true));
    drop(old);
    let new = evaluate(&h, graph.clone(), 2, Quality::Full);
    let maps = read(&h, &new);
    let (lo, hi) = range(&maps[&MapKind::Height]);
    assert_eq!((lo, hi), (0.0, 1.0));
    assert!(maps[&MapKind::Metallic].iter().all(|p| p[0] == 0.0));
    let (lo, hi) = range(&maps[&MapKind::Roughness]);
    assert!(lo >= 0.45 && hi <= 0.8 && hi - lo > 0.03, "{lo}..{hi}");
    for p in &maps[&MapKind::Normal] {
        assert!(p.iter().all(|v| v.is_finite()));
        assert!((p[0] * p[0] + p[1] * p[1] + p[2] * p[2] - 1.0).abs() < 0.002);
    }
    assert!(
        maps[&MapKind::Normal].iter().any(|p| p[0].abs() > 0.05),
        "brick relief must remain visible"
    );
    let old_detail = normal_detail(&old_maps[&MapKind::Normal], extent.width as usize);
    let new_detail = normal_detail(&maps[&MapKind::Normal], extent.width as usize);
    assert!(
        new_detail < old_detail * 0.3,
        "normal grain: {new_detail} vs {old_detail}"
    );
    if let Some(dir) = evidence {
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("before.png"), old_sheet.unwrap()).unwrap();
        std::fs::write(dir.join("after.png"), sheet(&h, &new, &maps, false)).unwrap();
        let TypedSourcePixels::Rgba8 { pixels, .. } = &upload.pixels else {
            unreachable!()
        };
        std::fs::write(
            dir.join("synthetic-brick.png"),
            texelith_io::encoders::encode_png8(
                extent.width,
                extent.height,
                texelith_io::encoders::PngChannels::Rgba,
                texelith_io::encoders::PngColorIntent::Srgb,
                pixels,
            )
            .unwrap(),
        )
        .unwrap();
        let stats = |maps: &Maps| {
            MAPS.into_iter()
                .map(|map| {
                    let (lo, hi) = range(&maps[&map]);
                    (
                        format!("{map:?}"),
                        serde_json::json!({"red_min":lo,"red_max":hi}),
                    )
                })
                .collect::<BTreeMap<_, _>>()
        };
        std::fs::write(dir.join("ranges.json"), serde_json::to_vec_pretty(&serde_json::json!({"extent":[extent.width, extent.height],"before":stats(&old_maps),"after":stats(&maps),"normal_adjacent_difference":{"before":old_detail,"after":new_detail},"coverage":"Hardware GPU full evaluation and production thumbnail display conversion; no native window"})).unwrap()).unwrap();
    }
    drop(new);
    let proxy = evaluate(&h, graph, 3, Quality::Proxy);
    assert!(
        proxy
            .outputs
            .values()
            .all(|lease| lease.texture().width() == proxy.extent.width
                && lease.texture().height() == proxy.extent.height)
    );
}

#[test]
fn flat_photos_stay_neutral_and_metal_requires_an_explicit_edit() {
    if gpu_availability::skip_if_no_gpu() {
        return;
    }
    let _serial = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let h = Harness::new();
    for (index, rgb) in [[0, 0, 0], [128, 128, 128], [255, 255, 255], [200, 50, 30]]
        .into_iter()
        .enumerate()
    {
        let extent = Extent::new(17, 13).unwrap();
        let pixels = TypedSourcePixels::Rgba8 {
            pixels: [rgb[0], rgb[1], rgb[2], 255]
                .repeat((extent.width * extent.height) as usize)
                .into(),
            interpretation: InputInterpretation::SrgbPhoto,
        };
        let descriptor = pixels.descriptor(extent).unwrap();
        install(
            &h,
            TypedSourceUpload {
                descriptor: descriptor.clone(),
                pixels,
            },
        );
        let graph = demo_material_from_source(descriptor).unwrap();
        let maps = read(
            &h,
            &evaluate(&h, graph.clone(), index as u64 * 2 + 1, Quality::Full),
        );
        assert!(
            maps[&MapKind::Height].iter().all(|p| p[0] == 0.5),
            "flat {rgb:?}: {:?}",
            range(&maps[&MapKind::Height])
        );
        assert!(maps[&MapKind::Metallic].iter().all(|p| p[0] == 0.0));
        assert!(maps[&MapKind::Ao].iter().all(|p| p[0] == 1.0));
        assert!(
            maps[&MapKind::Normal]
                .iter()
                .all(|p| p[0] == 0.0 && p[1] == 0.0 && p[2] == 1.0)
        );
        let mut dto = MaterialGraphDto::from(graph);
        for node in &mut dto.graph {
            if let MaterialNodeParams::Metallic { mask_mode, .. } = &mut node.params {
                *mask_mode = MaskMode::Add;
            }
        }
        let maps = read(
            &h,
            &evaluate(
                &h,
                Graph::try_from(dto).unwrap(),
                index as u64 * 2 + 2,
                Quality::Full,
            ),
        );
        assert!(maps[&MapKind::Metallic].iter().all(|p| p[0] == 1.0));
    }
}

#[test]
fn scalar_display_transfer_uses_the_full_range_and_preserves_neutrals() {
    for map in [
        MapKind::Height,
        MapKind::Roughness,
        MapKind::Ao,
        MapKind::Metallic,
        MapKind::Edge,
        MapKind::Mask,
    ] {
        for (linear, display) in [(0.0, 0), (0.18, 118), (0.5, 188), (1.0, 255)] {
            assert_eq!(
                presenter::rgb([linear, 0.0, 0.0, 1.0], map, None),
                [display; 3]
            );
        }
    }
    assert_eq!(presenter::rgb([0.0; 4], MapKind::Normal, None), [128; 3]);
    let map = MapKind::Roughness;
    let pixels = [[0.6; 4], [0.7; 4], [0.8; 4]];
    let range = presenter::scalar_range(&pixels, map);
    assert_eq!(range, Some([0.6, 0.8]));
    assert_eq!(presenter::display_rgb(pixels[0], map, None, range), [0; 3]);
    assert_eq!(
        presenter::display_rgb(pixels[2], map, None, range),
        [255; 3]
    );
    for (value, display) in [(0.0, 0), (0.5, 188), (1.0, 255)] {
        let pixels = [[value; 4]; 4];
        let range = presenter::scalar_range(&pixels, map);
        assert_eq!(range, None);
        assert_eq!(
            presenter::display_rgb(pixels[0], map, None, range),
            [display; 3]
        );
    }
}
