use super::{png, support::*};
use std::{sync::Arc, time::Instant};
use texelith_core::Extent;
use texelith_gpu::{WorkerDiagnostics, ibl::Environment};
pub fn worker(g: &Gpu, capture: Option<&std::path::Path>) -> Result<(WorkerDiagnostics, f64, f64)> {
    let worker = g.worker(BUDGET)?;
    let material = material(&worker)?;
    let environment = Arc::new(Environment::procedural());
    let extent = Extent {
        width: 160,
        height: 160,
    };
    let start = Instant::now();
    let frame = render(
        &worker,
        request(material.clone(), 1, extent, environment.clone(), 0.5, 0.0),
    )?;
    let cold_ms = start.elapsed().as_secs_f64() * 1000.0;
    let convolution_ms = worker.diagnostics().ibl_convolution_ns as f64 / 1e6;
    require(
        convolution_ms > 0.0 && convolution_ms < cold_ms,
        "missing isolated convolution timing",
    )?;
    frame.lease.retire_after_submission();
    let submissions = worker.diagnostics().submitted;
    let frame = render(
        &worker,
        request(
            material.clone(),
            2,
            extent,
            Arc::new(Environment::procedural()),
            0.5,
            0.0,
        ),
    )?;
    require(
        worker.diagnostics().submitted - submissions == 1,
        "content cache miss repeated convolution",
    )?;
    let bytes = g.read(frame.lease.texture(), 4)?;
    require(
        bytes.as_chunks::<4>().0.iter().any(|p| p[0] > 80),
        "environment did not illuminate material",
    )?;
    frame.lease.retire_after_submission();
    let mut matrix = vec![0u8; 800 * 320 * 4];
    for row in 0..2 {
        for col in 0..5 {
            let frame = render(
                &worker,
                request(
                    material.clone(),
                    3 + (row * 5 + col) as u64,
                    extent,
                    environment.clone(),
                    col as f32 / 4.0,
                    row as f32,
                ),
            )?;
            let bytes = g.read(frame.lease.texture(), 4)?;
            for y in 0..160 {
                let to = ((row * 160 + y) * 800 + col * 160) * 4;
                matrix[to..to + 640].copy_from_slice(&bytes[y * 640..(y + 1) * 640]);
            }
            frame.lease.retire_after_submission();
        }
    }
    if let Some(path) = capture {
        png::save(path, 800, 320, &matrix)?;
    }
    // Replace an in-progress environment after an observed face submission.
    let other = Arc::new(
        Environment::new(
            Extent {
                width: 1,
                height: 1,
            },
            &[[0.2, 0.4, 0.8]],
        )
        .map_err(|e| e.to_string())?,
    );
    let previous = worker.diagnostics().submitted;
    worker
        .request_preview(request(material.clone(), 30, extent, other, 0.2, 1.0))
        .map_err(|e| e.to_string())?;
    wait(|| {
        no_error(&worker)?;
        Ok((worker.diagnostics().submitted >= previous + 2).then_some(()))
    })?;
    require(
        worker.try_latest_preview().is_none(),
        "cold environment completed before cancellation fixture",
    )?;
    worker
        .request_preview(request(material, 31, extent, environment, 0.5, 0.0))
        .map_err(|e| e.to_string())?;
    let frame = complete(&worker, 31)?;
    frame.lease.retire_after_submission();
    g.finish()?;
    let d = worker.diagnostics();
    require(
        d.cancelled_pass_batches > 0,
        "face cancellation was not exercised",
    )?;
    require(
        d.validation_errors == 0 && d.stale_publications == 0,
        "validation/stale publication error",
    )?;
    require(
        d.peak_bytes <= BUDGET && d.peak_bytes > 4_892_672,
        "environment is absent from shared ledger",
    )?;
    require(
        d.peak_pending <= 1 && d.peak_in_flight <= 1,
        "unbounded worker",
    )?;
    worker.join().map_err(|e| e.to_string())?;
    let tiny = g.worker(1_000_000)?;
    let material = super::support::material(&tiny)?;
    tiny.request_preview(request(
        material,
        1,
        extent,
        Arc::new(Environment::procedural()),
        0.5,
        0.0,
    ))
    .map_err(|e| e.to_string())?;
    let event = wait(|| Ok(tiny.try_event()))?;
    require(
        format!("{event:?}").contains("Budget"),
        "IBL allocation was not rejected by shared budget",
    )?;
    require(
        tiny.try_latest_preview().is_none(),
        "overbudget IBL published",
    )?;
    tiny.join().map_err(|e| e.to_string())?;
    Ok((d, cold_ms, convolution_ms))
}
