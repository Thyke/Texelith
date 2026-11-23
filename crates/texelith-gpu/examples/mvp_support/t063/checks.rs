use super::support::*;
use std::sync::Arc;
use texelith_core::{Extent, MapKind};
use texelith_gpu::{WorkerDiagnostics, WorkerEvent, viewport_shared::PreviewContent};

pub fn diagnostics(worker: &texelith_gpu::GpuWorker) -> Result<WorkerDiagnostics> {
    let d = worker.diagnostics();
    require(
        d.peak_in_flight <= 1 && d.peak_pending <= 1,
        "compute and preview must share one in-flight and one pending slot",
    )?;
    require(
        d.peak_bytes > 0 && d.peak_bytes <= BUDGET,
        "combined compute/view ledger exceeds 1 GiB",
    )?;
    require(
        d.stale_publications == 0 && d.validation_errors == 0,
        "stale publication or validation error",
    )?;
    Ok(d)
}

pub fn clear_and_ramp(g: &Gpu) -> Result<()> {
    let worker = g.worker(BUDGET)?;
    let bundle = material(&worker, 63, 7, 1)?;
    // The diagnostic ramp samples the material's linear height, not generated
    // display codes. At matching dimensions it must reproduce all 17 values.
    for (serial, width, height) in [(1, 17, 3), (2, 1, 17), (3, 17, 1), (4, 37, 23)] {
        let extent = Extent::new(width, height).map_err(|e| e.to_string())?;
        let color = [0.0031308, 0.18, 0.5, 1.0];
        let frame = render(
            &worker,
            request(
                bundle.clone(),
                serial * 2 - 1,
                extent,
                PreviewContent::Clear(color),
            ),
        )?;
        require(
            frame.extent == extent && frame.scale == 1.0,
            "small preview was resized",
        )?;
        require(
            frame.lease.texture().sample_count() == 1,
            "initial preview is single sample",
        )?;
        check_clear(&g.read_display(frame.lease.texture())?, color)?;
        let ramp = render(
            &worker,
            request(
                bundle.clone(),
                serial * 2,
                extent,
                PreviewContent::MaterialRamp,
            ),
        )?;
        check_ramp(&g.read_display(ramp.lease.texture())?, extent)?;
    }
    let extent = Extent::new(17, 3).map_err(|e| e.to_string())?;
    let first = render(
        &worker,
        request(bundle.clone(), 9, extent, PreviewContent::MaterialRamp),
    )?;
    let bytes = g.read_display(first.lease.texture())?;
    require(bytes.len() == 17 * 3 * 4, "wrong ramp readback length")?;
    for (i, pixel) in bytes.as_chunks::<4>().0.iter().enumerate() {
        let expected = srgb_code((i % 17) as f64 / 16.0);
        require(
            pixel[..3].iter().all(|c| c.abs_diff(expected) <= 1) && pixel[3] == 255,
            "material ramp must have exactly one sRGB encode and opaque alpha",
        )?;
    }
    let second = render(
        &worker,
        request(bundle, 10, extent, PreviewContent::MaterialRamp),
    )?;
    require(
        bytes == g.read_display(second.lease.texture())?,
        "identical input and view must produce identical display bytes",
    )?;
    diagnostics(&worker)?;
    worker.join().map_err(|e| e.to_string())
}

pub fn identity_and_churn(g: &Gpu) -> Result<WorkerDiagnostics> {
    let worker = g.worker(BUDGET)?;
    let bundle = material(&worker, 63, 10, 1)?;
    let extent = Extent::new(127, 93).map_err(|e| e.to_string())?;
    let color = [0.18, 0.5, 0.003, 1.0];
    let held = render(
        &worker,
        request(bundle.clone(), 1, extent, PreviewContent::Clear(color)),
    )?;
    let held_bytes = g.read_display(held.lease.texture())?;
    for (job, generation) in [(64, 10), (63, 9), (63, 11)] {
        let mut bad = request(bundle.clone(), 2, extent, PreviewContent::MaterialRamp);
        bad.job_id = texelith_gpu::JobId(job);
        bad.generation = generation;
        require(
            worker.request_preview(bad).is_err(),
            "reject mismatched material job/generation before dispatch",
        )?;
    }
    for id in 2..=257 {
        let extent =
            Extent::new(31 + (id % 19) as u32, 17 + (id % 13) as u32).map_err(|e| e.to_string())?;
        let mut preview = request(bundle.clone(), id, extent, PreviewContent::MaterialRamp);
        preview.view.camera.yaw = id as f32 * 0.01;
        worker.request_preview(preview).map_err(|e| e.to_string())?;
        let d = worker.diagnostics();
        require(
            d.pending <= 1 && d.in_flight <= 1,
            "mailbox grew during resize/camera burst",
        )?;
    }
    let latest = complete(&worker)?;
    require(
        latest.request_id == 257 && latest.generation == 10,
        "superseded preview escaped the completion slot",
    )?;
    require(
        latest.extent == Extent::new(31 + 257 % 19, 17 + 257 % 13).map_err(|e| e.to_string())?,
        "stale resize extent published",
    )?;
    require(
        g.read_display(held.lease.texture())? == held_bytes,
        "resize churn overwrote a still-leased display",
    )?;
    let stale = request(bundle.clone(), 256, extent, PreviewContent::MaterialRamp);
    require(
        worker.request_preview(stale).is_err(),
        "older request ID must not replace the latest view",
    )?;
    let d = diagnostics(&worker)?;
    require(
        d.coalesced > 0,
        "burst did not exercise latest-only coalescing",
    )?;
    drop(latest);
    // New documents may restart request/generation counters. Equal generation
    // and request numbers alone cannot identify a material from another job.
    let next = material(&worker, 64, 1, 1)?;
    require(
        worker
            .request_preview(request(bundle, 258, extent, PreviewContent::MaterialRamp))
            .is_err(),
        "old job material accepted after replacement",
    )?;
    let current = render(
        &worker,
        request(next, 1, extent, PreviewContent::MaterialRamp),
    )?;
    require(
        current.job_id == texelith_gpu::JobId(64) && current.generation == 1,
        "new job counters could not restart",
    )?;
    require(
        g.read_display(held.lease.texture())? == held_bytes,
        "material retirement changed the last valid image",
    )?;
    let d = diagnostics(&worker)?;
    worker.join().map_err(|e| e.to_string())?;
    require(
        g.read_display(held.lease.texture())? == held_bytes,
        "worker shutdown invalidated an external display lease",
    )?;
    held.lease.retire_after_submission();
    current.lease.retire_after_submission();
    g.finish()?;
    Ok(d)
}

pub fn lifetime_and_budget(g: &Gpu) -> Result<WorkerDiagnostics> {
    let worker = g.worker(BUDGET)?;
    let bundle = material(&worker, 63, 1, 1)?;
    require(
        bundle.outputs[&MapKind::Height].texture().format() == wgpu::TextureFormat::R32Float,
        "authoritative height precision changed",
    )?;
    require(
        bundle.outputs[&MapKind::Normal].texture().format() == wgpu::TextureFormat::Rgba16Float,
        "normal format changed",
    )?;
    let preview = request(
        bundle.clone(),
        1,
        Extent::new(4096, 2048).map_err(|e| e.to_string())?,
        PreviewContent::MaterialRamp,
    );
    // Transfer the last caller-owned bundle into the request. Successful render
    // and validation must not depend on retaining any caller material handle.
    drop(bundle);
    worker.request_preview(preview).map_err(|e| e.to_string())?;
    let frame = complete(&worker)?;
    require(
        frame.extent == Extent::new(2048, 1024).map_err(|e| e.to_string())?
            && (frame.scale - 0.5).abs() < 1e-6,
        "framebuffer cap must preserve aspect and report scale",
    )?;
    let d = diagnostics(&worker)?;
    let targets = 2048u64 * 1024 * (8 + 4 + 4);
    require(
        d.peak_bytes >= targets,
        "HDR/depth/display targets missing from application ledger",
    )?;
    worker.join().map_err(|e| e.to_string())?;

    let low = g.worker(128 * 1024)?;
    let bundle = material(&low, 63, 1, 1)?;
    let color = [0.18, 0.18, 0.18, 1.0];
    let old = render(
        &low,
        request(
            bundle.clone(),
            1,
            Extent::new(17, 3).map_err(|e| e.to_string())?,
            PreviewContent::Clear(color),
        ),
    )?;
    let old_bytes = g.read_display(old.lease.texture())?;
    // Valid intent is queued; admission fails asynchronously before allocation.
    low.request_preview(request(
        bundle,
        2,
        Extent::new(2048, 2048).map_err(|e| e.to_string())?,
        PreviewContent::MaterialRamp,
    ))
    .map_err(|e| e.to_string())?;
    let event = wait(|| Ok(low.try_event()))?;
    require(
        matches!(event, WorkerEvent::BudgetExceeded { budget_bytes, required_bytes } if budget_bytes == 128 * 1024 && required_bytes > budget_bytes),
        "oversized view must return bounded BudgetExceeded",
    )?;
    require(
        low.try_latest_preview().is_none(),
        "budget rejection published a partial preview",
    )?;
    require(
        low.diagnostics().peak_bytes <= 128 * 1024,
        "preflight exceeded configured budget",
    )?;
    require(
        g.read_display(old.lease.texture())? == old_bytes,
        "budget rejection discarded/overwrote previous valid display",
    )?;
    diagnostics(&low)?;
    low.join().map_err(|e| e.to_string())?;
    Ok(d)
}

pub fn compute_supersedes_preview(g: &Gpu) -> Result<()> {
    let worker = g.worker(BUDGET)?;
    let old = material(&worker, 63, 1, 1)?;
    let extent = Extent::new(2048, 1024).map_err(|e| e.to_string())?;
    worker
        .request_preview(request(
            old.clone(),
            1,
            extent,
            PreviewContent::MaterialRamp,
        ))
        .map_err(|e| e.to_string())?;
    let pixels = texelith_gpu::TypedSourcePixels::HeightF32(
        (0..51)
            .map(|i| (i % 17) as f32 / 16.0)
            .collect::<Vec<_>>()
            .into(),
    );
    let descriptor = pixels
        .descriptor(Extent::new(17, 3).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    worker
        .request_material(material_request(descriptor, 63, 2, 2)?)
        .map_err(|e| e.to_string())?;
    let next = wait(|| {
        no_error(&worker)?;
        Ok(worker.try_latest_material())
    })?;
    require(
        next.generation == 2 && next.outputs.len() == 2,
        "compute published an incoherent replacement",
    )?;
    require(
        worker.try_latest_preview().is_none(),
        "old preview survived newer material intent",
    )?;
    require(
        worker
            .request_preview(request(old, 2, extent, PreviewContent::MaterialRamp))
            .is_err(),
        "obsolete material lease accepted after newer compute",
    )?;
    let frame = render(
        &worker,
        request(
            Arc::new(next),
            3,
            Extent::new(37, 23).map_err(|e| e.to_string())?,
            PreviewContent::MaterialRamp,
        ),
    )?;
    require(
        frame.generation == 2,
        "preview used old material generation",
    )?;
    diagnostics(&worker)?;
    worker.join().map_err(|e| e.to_string())
}
