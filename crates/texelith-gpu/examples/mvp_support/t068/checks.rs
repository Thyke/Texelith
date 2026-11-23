use super::{png, support::*};
use texelith_core::{Extent, NormalConvention, PreviewMesh};
use texelith_gpu::WorkerDiagnostics;
pub fn worker(g: &Gpu, captures: Option<&std::path::Path>) -> Result<WorkerDiagnostics> {
    let worker = g.worker(BUDGET)?;
    let extent = Extent::new(256, 256).map_err(|e| e.to_string())?;
    let flat = material(&worker, 680, Some(0.5), NormalConvention::OpenGl)?;
    let mut id = 0;
    for ibl in [false, true] {
        id += 1;
        let f = render(&worker, request(flat.clone(), id, extent, 0., ibl))?;
        let baseline = g.read_display(f.lease.texture())?;
        f.lease.retire_after_submission();
        id += 1;
        let f = render(&worker, request(flat.clone(), id, extent, 0.3, ibl))?;
        let displaced = g.read_display(f.lease.texture())?;
        f.lease.retire_after_submission();
        require(
            baseline == displaced,
            "constant-midlevel render identity (direct/IBL)",
        )?;
    }
    let relief = material(&worker, 681, None, NormalConvention::OpenGl)?;
    let mut capture = Vec::new();
    for scale in [0., 0.12] {
        id += 1;
        let f = render(&worker, request(relief.clone(), id, extent, scale, true))?;
        let pixels = g.read_display(f.lease.texture())?;
        f.lease.retire_after_submission();
        require(
            pixels.as_chunks::<4>().0.iter().all(|p| p[3] == 255),
            "nonopaque POM display",
        )?;
        capture.push(pixels);
    }
    let changed = capture[0]
        .as_chunks::<4>()
        .0
        .iter()
        .zip(capture[1].as_chunks::<4>().0.iter())
        .filter(|(a, b)| a != b)
        .count();
    require(
        changed > 1000,
        "nonconstant height must visibly displace material",
    )?;
    if let Some(dir) = captures {
        let mut montage = Vec::new();
        for y in 0..256 {
            for image in &capture {
                montage.extend_from_slice(&image[y * 1024..(y + 1) * 1024]);
            }
        }
        png::save(&dir.join("pom-off-on.png"), 512, 256, &montage)?;
    }
    let dx = material(&worker, 682, None, NormalConvention::DirectX)?;
    id += 1;
    let f = render(&worker, request(dx.clone(), id, extent, 0.12, true))?;
    let dx_pixels = g.read_display(f.lease.texture())?;
    f.lease.retire_after_submission();
    require(
        dx_pixels
            .iter()
            .zip(&capture[1])
            .all(|(a, b)| a.abs_diff(*b) <= 1),
        "POM convention must not flip view rays or normal twice",
    )?;
    let mut primitive_sheet = captures.map(|_| vec![0; 1024 * 256 * 4]);
    for (i, mesh) in [
        PreviewMesh::Plane,
        PreviewMesh::Cube,
        PreviewMesh::Sphere,
        PreviewMesh::Cylinder,
    ]
    .into_iter()
    .enumerate()
    {
        id += 1;
        let mut intent = request(dx.clone(), id, extent, f32::MAX, false);
        intent.view.mesh = mesh.clone();
        intent.view.camera.yaw = 1.52;
        let f = render(&worker, intent)?;
        let pixels = g.read_display(f.lease.texture())?;
        require(
            pixels.as_chunks::<4>().0.iter().all(|p| p[3] == 255),
            "grazing primitive display invalid",
        )?;
        if let Some(dir) = captures {
            png::save(&dir.join(format!("grazing-{i}.png")), 256, 256, &pixels)?;
        }
        f.lease.retire_after_submission();
        if let Some(sheet) = &mut primitive_sheet {
            id += 1;
            let mut intent = request(dx.clone(), id, extent, 0.12, true);
            intent.view.mesh = mesh;
            let f = render(&worker, intent)?;
            let pixels = g.read_display(f.lease.texture())?;
            for y in 0..256 {
                let start = (y * 1024 + i * 256) * 4;
                sheet[start..start + 1024].copy_from_slice(&pixels[y * 1024..(y + 1) * 1024]);
            }
            f.lease.retire_after_submission();
        }
    }
    if let (Some(dir), Some(sheet)) = (captures, primitive_sheet) {
        png::save(&dir.join("pom-primitives.png"), 1024, 256, &sheet)?;
    }
    id += 1;
    let mut invalid = request(dx.clone(), id, extent, -0.1, false);
    require(
        worker.request_preview(invalid.clone()).is_err(),
        "negative scale accepted",
    )?;
    invalid.view.pom.scale = f32::NAN;
    require(
        worker.request_preview(invalid).is_err(),
        "nonfinite scale accepted",
    )?;
    // Failing admission must remain a recoverable, explicit error.
    let tiny = g.worker(1_048_576)?;
    let tiny_material = material(&tiny, 684, None, NormalConvention::OpenGl)?;
    id += 1;
    let rejected = tiny.request_preview(request(
        tiny_material,
        id,
        Extent::new(1024, 1024).map_err(|e| e.to_string())?,
        0.2,
        false,
    ));
    if rejected.is_ok() {
        let e = wait(|| Ok(tiny.try_event()))?;
        require(
            matches!(e, texelith_gpu::WorkerEvent::BudgetExceeded { .. }),
            "tiny budget must fail admission",
        )?;
    }
    tiny.join().map_err(|e| e.to_string())?;
    let d = diagnostics(&worker)?;
    drop((flat, relief, dx));
    worker.join().map_err(|e| e.to_string())?;
    Ok(d)
}
