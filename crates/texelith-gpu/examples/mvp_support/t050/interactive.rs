//! Completion-bound measurement of the real imported-normal material worker.
use super::{Result, gpu::Gpu, oracle, proxy_oracle, require};
use serde_json::{Value, json};
use std::{
    sync::Arc,
    time::{Duration, Instant},
};
use texelith_core::{
    EvaluationSnapshot, Extent, Graph, MapKind, MaterialNodeParams, NormalConvention,
    material_from_source,
};
use texelith_gpu::{
    GpuConfig, GpuWorker, JobId, MaterialPlan, MaterialRequest, Quality, ResourceReservations,
    TypedSourcePixels, TypedSourceUpload, WorkerEvent,
};

fn wait<T>(mut poll: impl FnMut() -> Result<Option<T>>) -> Result<T> {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        if let Some(value) = poll()? {
            return Ok(value);
        }
        require(Instant::now() < deadline, "interactive worker timeout")?;
        std::thread::yield_now();
    }
}

pub(super) fn measure(gpu: &Gpu) -> Result<Value> {
    let worker = GpuWorker::spawn(gpu.device.clone(), gpu.queue.clone(), GpuConfig::default())
        .map_err(|e| e.to_string())?;
    let result = run(gpu, &worker);
    worker.join().map_err(|e| e.to_string())?;
    result
}

fn run(gpu: &Gpu, worker: &GpuWorker) -> Result<Value> {
    let source_grid = oracle::Grid {
        w: 4096,
        h: 4096,
        repeat: true,
    };
    let proxy_grid = oracle::Grid {
        w: 1024,
        h: 1024,
        repeat: true,
    };
    let source_extent = Extent::new(4096, 4096).map_err(|e| e.to_string())?;
    let extent = Extent::new(1024, 1024).map_err(|e| e.to_string())?;
    let bytes = proxy_oracle::analytic_rgb(source_grid);
    let expected_normal = proxy_oracle::restrict(&bytes, source_grid, proxy_grid);
    let pixels = TypedSourcePixels::NormalRgb8 {
        pixels: bytes.into(),
        convention: NormalConvention::OpenGl,
    };
    let descriptor = pixels
        .descriptor(source_extent)
        .map_err(|e| e.to_string())?;
    let graph = material_from_source(descriptor.clone()).map_err(|e| e.to_string())?;
    let height = graph.material_outputs().unwrap()[&MapKind::Height];
    let normal = graph.material_outputs().unwrap()[&MapKind::Normal];
    worker
        .install_typed_source(TypedSourceUpload { descriptor, pixels })
        .map_err(|e| e.to_string())?;
    let ready = wait(|| Ok(worker.try_event()))?;
    require(
        matches!(ready, WorkerEvent::SourceReady { .. }),
        "normal source upload failed",
    )?;
    let mut rows = Vec::new();
    let mut final_bundle = None;
    let mut planned_peak = 0;
    let baseline = worker.diagnostics();
    eprintln!("Poisson interactive worker: 4K RGB8 source to 1K height, 10 warmups + 100 requests");
    for i in 0..110 {
        // Alternate a real solver control to invalidate the output key. On the
        // spectral path this leaves the numerical schedule unchanged.
        let mut dto = texelith_core::MaterialGraphDto::from(graph.clone());
        for node in &mut dto.graph {
            if let MaterialNodeParams::HeightFromNormal { pre_smooth, .. } = &mut node.params {
                *pre_smooth = 3 + i % 2;
            }
        }
        let request = MaterialRequest {
            job_id: JobId(50),
            request_id: u64::from(i + 1),
            snapshot: Arc::new(EvaluationSnapshot {
                generation: u64::from(i + 1),
                graph: Graph::try_from(dto).map_err(|e| e.to_string())?,
            }),
            outputs: [(MapKind::Height, height)].into(),
            extent,
            quality: Quality::Proxy,
        };
        planned_peak = MaterialPlan::new(&request, super::BUDGET, ResourceReservations::default())
            .map_err(|e| e.to_string())?
            .peak_bytes();
        drop(final_bundle.take());
        let start = Instant::now();
        worker
            .request_material(request)
            .map_err(|e| e.to_string())?;
        let bundle = wait(|| {
            require(worker.try_event().is_none(), "interactive worker error")?;
            Ok(worker.try_latest_material())
        })?;
        rows.push(vec![start.elapsed().as_secs_f64() * 1000.0]);
        require(
            bundle.extent == extent && bundle.request_id == u64::from(i + 1),
            "wrong proxy publication",
        )?;
        final_bundle = Some(bundle);
        if i % 25 == 24 {
            eprintln!(
                "Poisson interactive worker: {}/110 complete requests",
                i + 1
            );
        }
    }
    let diagnostics = worker.diagnostics();
    require(
        diagnostics.cache_hits == baseline.cache_hits
            && diagnostics.submitted >= baseline.submitted + 220,
        "interactive timing reused a cached output",
    )?;
    require(
        diagnostics.validation_errors == 0
            && diagnostics.stale_publications == 0
            && diagnostics.peak_bytes <= planned_peak,
        "interactive worker validation/publication/budget failure",
    )?;
    let height_values =
        gpu.read(final_bundle.as_ref().unwrap().outputs[&MapKind::Height].texture());
    drop(final_bundle);
    // Inspect the actual proxy normal separately, outside all timed solves.
    let request = MaterialRequest {
        job_id: JobId(50),
        request_id: 111,
        snapshot: Arc::new(EvaluationSnapshot {
            generation: 111,
            graph,
        }),
        outputs: [(MapKind::Normal, normal)].into(),
        extent,
        quality: Quality::Proxy,
    };
    worker
        .request_material(request)
        .map_err(|e| e.to_string())?;
    let bundle = wait(|| {
        require(
            worker.try_event().is_none(),
            "normal proxy readback request failed",
        )?;
        Ok(worker.try_latest_material())
    })?;
    let actual_normal = gpu.read_normal(bundle.outputs[&MapKind::Normal].texture());
    let normal_error = actual_normal
        .iter()
        .zip(&expected_normal)
        .flat_map(|(a, b)| a.iter().zip(b).map(|(a, b)| (a - b).abs()))
        .fold(0.0f64, f64::max);
    require(
        normal_error < 5e-4,
        "worker normal proxy disagrees with CPU area oracle",
    )?;
    let (gradient, _) = oracle::slopes(&actual_normal, false);
    let optimum = oracle::periodic_fft(proxy_grid, &proxy_grid.transpose(&gradient));
    let rms = oracle::rms(&height_values, &optimum);
    let gap = oracle::optimality_gap(proxy_grid, &height_values, &optimum);
    let relative_gap = gap / oracle::field_norm(&gradient);
    require(
        rms <= 1e-3 && relative_gap <= 1e-3 && oracle::mean(&height_values).abs() <= 2e-5,
        "worker proxy height does not recover least-squares optimum",
    )?;
    require(
        oracle::rms(&vec![0.0; proxy_grid.len()], &optimum) > 1e-3,
        "proxy fixture must reject zero height",
    )?;
    let timing = crate::contract::aggregate_timings(&rows, 10)?;
    Ok(
        json!({"kind":"completion_upper_bound","source_size":4096,"size":1024,
        "warmups":10,"measurements":timing.measurements,"p95_ms":timing.p95_ms,
        "peak_bytes":diagnostics.peak_bytes,"planned_peak_bytes":planned_peak,
        "cache_hits":diagnostics.cache_hits-baseline.cache_hits,"submissions":diagnostics.submitted-baseline.submitted,
        "validation_errors":diagnostics.validation_errors,"stale_publications":diagnostics.stale_publications,
        "rms":rms,"relative_optimality_gap":relative_gap,"normal_component_error":normal_error,
        "scope":"request through publication: source RGB8 decode/area restriction, fresh 1K solver, all submissions/readbacks; source upload and CPU oracle/readback excluded",
        "samples_ms":rows[10..].iter().map(|r|r[0]).collect::<Vec<_>>()}),
    )
}
