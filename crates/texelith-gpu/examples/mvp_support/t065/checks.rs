use super::{oracle, support::*};
use texelith_core::{Extent, NormalConvention, PreviewMesh};
use texelith_gpu::{pbr::PbrSettings, viewport_shared::PreviewContent};

pub fn analytic(g: &Gpu) -> Result<()> {
    let worker = g.worker(BUDGET)?;
    let extent = Extent::new(129, 129).map_err(|e| e.to_string())?;
    let bundle = material(
        &worker,
        650,
        [128, 128, 255],
        NormalConvention::OpenGl,
        false,
        false,
    )?;
    let n = oracle::normal([128, 128, 255]);
    let color = [128, 96, 64].map(oracle::decode);
    let mut id = 0;
    let mut previous = None;
    for roughness in [None, Some(0.0), Some(0.25), Some(1.0)] {
        for metallic in [0.0, 1.0] {
            for exposure in [-2.0, 0.0, 2.0] {
                id += 1;
                let settings = PbrSettings {
                    roughness,
                    metallic: Some(metallic),
                    exposure,
                    ..Default::default()
                };
                let frame = render(
                    &worker,
                    request(bundle.clone(), id, extent, PreviewMesh::Plane, settings),
                )?;
                let bytes = g.read_display(frame.lease.texture())?;
                let expected = oracle::shade(
                    color,
                    f64::from(roughness.unwrap_or(0.45)),
                    metallic.into(),
                    n[2],
                    n[2],
                    n[2],
                    1.,
                )
                .map(|x| oracle::transfer(oracle::tone(x * 2f64.powf(exposure.into()))));
                let pixel = center(&bytes, extent);
                require(pixel[3] == 255, "nonopaque PBR fragment")?;
                for (got, want) in pixel[..3].iter().zip(expected) {
                    require(
                        got.abs_diff(want) <= 2,
                        &format!(
                            "analytic GGX/ACES/sRGB mismatch: r={roughness:?} m={metallic} ev={exposure}, {pixel:?} != {expected:?}"
                        ),
                    )?;
                }
                require(
                    bytes[..3]
                        .iter()
                        .all(|x| x.abs_diff(oracle::transfer(0.018)) <= 1),
                    "background changed with material exposure/tone mapping",
                )?;
                previous = Some((frame, bytes));
            }
        }
    }
    // Identical settings preserve the previous immutable output byte-for-byte.
    let (held, held_bytes) = previous.expect("analytic samples");
    let repeat = render(
        &worker,
        request(
            bundle.clone(),
            id + 1,
            extent,
            PreviewMesh::Plane,
            PbrSettings {
                roughness: Some(1.),
                metallic: Some(1.),
                exposure: 2.,
                ..Default::default()
            },
        ),
    )?;
    require(
        g.read_display(repeat.lease.texture())? == held_bytes,
        "nondeterministic lit frame",
    )?;
    let mut dark = request(
        bundle.clone(),
        id + 2,
        extent,
        PreviewMesh::Plane,
        PbrSettings::default(),
    );
    dark.view.light.intensity = 0.;
    let dark = render(&worker, dark)?;
    require(
        center(&g.read_display(dark.lease.texture())?, extent) == [0, 0, 0, 255],
        "zero direct radiance is not black",
    )?;
    // A normal tilted toward +Y must respond more strongly to a +Y light.
    // Equivalent DX data negates encoded Y; both paths must canonicalize once.
    let mut normal_pixels = Vec::new();
    for (job, normal, convention, ao) in [
        (651, [128, 204, 230], NormalConvention::OpenGl, false),
        (652, [128, 51, 230], NormalConvention::DirectX, false),
        (653, [128, 204, 230], NormalConvention::OpenGl, true),
    ] {
        let bundle = material(&worker, job, normal, convention, false, ao)?;
        let mut lit = request(
            bundle.clone(),
            1,
            extent,
            PreviewMesh::Plane,
            PbrSettings {
                roughness: Some(1.),
                ..Default::default()
            },
        );
        lit.view.light.direction = [0., 1., 1.];
        let frame = render(&worker, lit)?;
        normal_pixels.push(center(&g.read_display(frame.lease.texture())?, extent));
        if job == 651 {
            let mut away = request(
                bundle,
                2,
                extent,
                PreviewMesh::Plane,
                PbrSettings {
                    roughness: Some(1.),
                    ..Default::default()
                },
            );
            away.view.light.direction = [0., -1., 1.];
            let away = render(&worker, away)?;
            let away_pixel = center(&g.read_display(away.lease.texture())?, extent);
            require(
                away_pixel[0] + 20 < normal_pixels[0][0],
                &format!(
                    "signed +Y normal/TBN direction is reversed: toward={:?}, away={away_pixel:?}",
                    normal_pixels[0]
                ),
            )?;
        }
    }
    for p in &normal_pixels[1..] {
        require(
            p.iter()
                .zip(normal_pixels[0])
                .all(|(a, b)| a.abs_diff(b) <= 1),
            "DX/GL conversion or AO altered direct light",
        )?;
    }
    require(
        g.read_display(held.lease.texture())? == held_bytes,
        "replacement material overwrote a leased lit display",
    )?;
    diagnostics(&worker)?;
    worker.join().map_err(|e| e.to_string())
}
pub fn primitives_and_lifetime(
    g: &Gpu,
    capture_dir: Option<&std::path::Path>,
) -> Result<texelith_gpu::WorkerDiagnostics> {
    let worker = g.worker(BUDGET)?;
    let bundle = material(
        &worker,
        654,
        [128, 128, 255],
        NormalConvention::OpenGl,
        true,
        false,
    )?;
    let extent = Extent::new(384, 384).map_err(|e| e.to_string())?;
    let mut images = Vec::new();
    for (index, (name, mesh)) in [
        ("plane", PreviewMesh::Plane),
        ("cube", PreviewMesh::Cube),
        ("sphere", PreviewMesh::Sphere),
        ("cylinder", PreviewMesh::Cylinder),
    ]
    .into_iter()
    .enumerate()
    {
        let mut intent = request(
            bundle.clone(),
            index as u64 + 1,
            extent,
            mesh,
            PbrSettings::default(),
        );
        intent.view.camera.yaw = if name == "plane" { 0.0 } else { 0.5 };
        intent.view.camera.pitch = 0.28;
        intent.view.light.direction = [-0.5, 0.8, 1.0];
        intent.view.light.intensity = 3.0;
        let frame = render(&worker, intent)?;
        let bytes = g.read_display(frame.lease.texture())?;
        require(
            bytes.as_chunks::<4>().0.iter().all(|p| p[3] == 255),
            "nonopaque primitive/background",
        )?;
        let background = &bytes[..4];
        require(
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .filter(|p| *p != background)
                .count()
                > 5000,
            &format!("empty or wrongly culled {name}"),
        )?;
        require(
            images.iter().all(|old: &Vec<u8>| old != &bytes),
            "primitive selection did not change geometry",
        )?;
        if let Some(dir) = capture_dir {
            super::png::save(&dir.join(format!("{name}.png")), 384, 384, &bytes)?;
        }
        images.push(bytes);
        frame.lease.retire_after_submission();
    }
    let stable = render(
        &worker,
        request(
            bundle.clone(),
            5,
            extent,
            PreviewMesh::Sphere,
            PbrSettings::default(),
        ),
    )?;
    let stable_bytes = g.read_display(stable.lease.texture())?;
    // A stream of camera/mesh/resize intents must keep one in-flight authority.
    for id in 6..=69 {
        let e =
            Extent::new(71 + (id % 19) as u32, 53 + (id % 13) as u32).map_err(|e| e.to_string())?;
        let mut r = request(
            bundle.clone(),
            id,
            e,
            PreviewMesh::Cylinder,
            PbrSettings::default(),
        );
        r.view.camera.yaw = id as f32 * 0.02;
        r.view.camera.pitch = 1.5;
        r.view.light.direction = [f32::MAX, 0., f32::MAX];
        r.view.light.intensity = f32::MAX;
        worker.request_preview(r).map_err(|e| e.to_string())?;
    }
    let last = wait(|| {
        no_error(&worker)?;
        Ok(worker.try_latest_preview())
    })?;
    require(last.request_id == 69, "stale camera/mesh churn published")?;
    let invalid = PbrSettings {
        exposure: f32::NAN,
        ..Default::default()
    };
    require(
        worker
            .request_preview(request(
                bundle.clone(),
                70,
                extent,
                PreviewMesh::Plane,
                invalid,
            ))
            .is_err(),
        "nonfinite exposure accepted",
    )?;
    let mut diagnostic = request(
        bundle.clone(),
        70,
        Extent::new(257, 3).map_err(|e| e.to_string())?,
        PreviewMesh::Plane,
        PbrSettings::default(),
    );
    diagnostic.content = PreviewContent::Clear([0.0031308, 0.18, 0.5, 1.]);
    let ramp = render(&worker, diagnostic)?;
    let expected = [
        oracle::transfer(0.0031308),
        oracle::transfer(0.18),
        oracle::transfer(0.5),
        255,
    ];
    require(
        g.read_display(ramp.lease.texture())?
            .as_chunks::<4>()
            .0
            .iter()
            .all(|p| p.iter().zip(expected).all(|(a, b)| a.abs_diff(b) <= 1)),
        "exact display transfer/diagnostic compatibility",
    )?;
    require(
        g.read_display(stable.lease.texture())? == stable_bytes,
        "mesh/resize churn overwrote held frame",
    )?;
    if let Some(dir) = capture_dir {
        let size = Extent::new(128, 128).map_err(|e| e.to_string())?;
        let mut sheet = vec![0; 384 * 384 * 4];
        for row in 0..3 {
            for col in 0..3 {
                let roughness = [0.08, 0.45, 1.0][col];
                let settings = if row < 2 {
                    PbrSettings {
                        roughness: Some(roughness),
                        metallic: Some(row as f32),
                        ..Default::default()
                    }
                } else {
                    PbrSettings {
                        exposure: [-2., 0., 2.][col],
                        roughness: Some(0.45),
                        ..Default::default()
                    }
                };
                let mut intent = request(
                    bundle.clone(),
                    71 + (row * 3 + col) as u64,
                    size,
                    PreviewMesh::Sphere,
                    settings,
                );
                intent.view.light.direction = [-0.5, 0.8, 1.0];
                intent.view.light.intensity = 3.;
                let frame = render(&worker, intent)?;
                let bytes = g.read_display(frame.lease.texture())?;
                for y in 0..128 {
                    let dest = ((row * 128 + y) * 384 + col * 128) * 4;
                    sheet[dest..dest + 128 * 4]
                        .copy_from_slice(&bytes[y * 128 * 4..(y + 1) * 128 * 4]);
                }
            }
        }
        super::png::save(&dir.join("material-matrix.png"), 384, 384, &sheet)?;
    }
    let d = diagnostics(&worker)?;
    worker.join().map_err(|e| e.to_string())?;
    require(
        g.read_display(stable.lease.texture())? == stable_bytes,
        "shutdown invalidated lit display lease",
    )?;
    Ok(d)
}
pub fn budget(g: &Gpu) -> Result<()> {
    let worker = g.worker(512 * 1024)?;
    let bundle = material(
        &worker,
        655,
        [128, 128, 255],
        NormalConvention::OpenGl,
        false,
        false,
    )?;
    let small = Extent::new(17, 17).map_err(|e| e.to_string())?;
    let previous = render(
        &worker,
        request(
            bundle.clone(),
            1,
            small,
            PreviewMesh::Plane,
            PbrSettings::default(),
        ),
    )?;
    let bytes = g.read_display(previous.lease.texture())?;
    worker
        .request_preview(request(
            bundle,
            2,
            Extent::new(2048, 2048).map_err(|e| e.to_string())?,
            PreviewMesh::Sphere,
            PbrSettings::default(),
        ))
        .map_err(|e| e.to_string())?;
    let event = wait(|| Ok(worker.try_event()))?;
    require(
        matches!(event, texelith_gpu::WorkerEvent::BudgetExceeded { .. }),
        "PBR preflight did not reject oversized targets/mesh",
    )?;
    require(
        worker.diagnostics().peak_bytes <= 512 * 1024 && worker.try_latest_preview().is_none(),
        "budget rejection overallocated or published",
    )?;
    require(
        g.read_display(previous.lease.texture())? == bytes,
        "budget rejection invalidated old display",
    )?;
    worker.join().map_err(|e| e.to_string())
}
