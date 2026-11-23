use std::{
    sync::Arc,
    time::{Duration, Instant},
};
use texelith_core::*;
use texelith_gpu::{viewport_shared::PreviewContent, *};
type Result<T> = std::result::Result<T, String>;
fn require(ok: bool, s: &str) -> Result<()> {
    if ok { Ok(()) } else { Err(s.into()) }
}
fn wait<T>(worker: &GpuWorker, mut next: impl FnMut() -> Option<T>) -> Result<T> {
    let start = Instant::now();
    loop {
        if let Some(value) = next() {
            return Ok(value);
        }
        if let Some(e) = worker.try_event() {
            return Err(format!("compare worker: {e:?}"));
        }
        if start.elapsed() > Duration::from_secs(60) {
            return Err("compare timeout".into());
        }
        std::thread::sleep(Duration::from_millis(1));
    }
}
fn material(
    worker: &GpuWorker,
    id: u64,
    extent: Extent,
    pixels: TypedSourcePixels,
    map: MapKind,
) -> Result<Arc<CompletedMaterial>> {
    let descriptor = pixels.descriptor(extent).map_err(|e| e.to_string())?;
    worker
        .install_typed_source(TypedSourceUpload {
            descriptor: descriptor.clone(),
            pixels,
        })
        .map_err(|e| e.to_string())?;
    let start = Instant::now();
    loop {
        if let Some(e) = worker.try_event() {
            require(
                matches!(e, WorkerEvent::SourceReady { .. }),
                "source upload failed",
            )?;
            break;
        }
        if start.elapsed() > Duration::from_secs(30) {
            return Err("upload timeout".into());
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    let graph = material_from_source(descriptor).map_err(|e| e.to_string())?;
    let outputs = graph
        .material_outputs()
        .unwrap()
        .iter()
        .filter(|(k, _)| **k == map)
        .map(|(k, v)| (*k, *v))
        .collect();
    worker
        .request_material(MaterialRequest {
            job_id: JobId(id),
            request_id: id,
            snapshot: Arc::new(EvaluationSnapshot {
                graph,
                generation: id,
            }),
            outputs,
            extent: proxy_extent(extent).map_err(|e| e.to_string())?,
            quality: Quality::Proxy,
        })
        .map_err(|e| e.to_string())?;
    Ok(Arc::new(wait(worker, || worker.try_latest_material())?))
}
fn request(bundle: Arc<CompletedMaterial>, id: u64, extent: Extent, seed: u32) -> PreviewRequest {
    let tiles = Tiles {
        count: 3,
        seed,
        random_rotation: true,
    };
    PreviewRequest {
        job_id: bundle.job_id,
        generation: bundle.generation,
        request_id: id,
        material: bundle,
        extent,
        view: ViewState {
            tiles,
            ..Default::default()
        },
        content: PreviewContent::Map {
            map: MapKind::Normal,
            tiles,
            convention: Some(NormalConvention::OpenGl),
        },
    }
}
fn render(worker: &GpuWorker, r: PreviewRequest) -> Result<CompletedPreview> {
    worker.request_preview(r).map_err(|e| e.to_string())?;
    wait(worker, || worker.try_latest_preview())
}
fn read(device: &wgpu::Device, queue: &wgpu::Queue, f: &CompletedPreview) -> Result<Vec<u8>> {
    let stride = (f.extent.width * 4).div_ceil(256) * 256;
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("T-069 oracle readback"),
        size: u64::from(stride) * u64::from(f.extent.height),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
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
    queue.submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    buffer.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| e.to_string())?;
    rx.recv()
        .map_err(|e| e.to_string())?
        .map_err(|e| e.to_string())?;
    let data = buffer
        .slice(..)
        .get_mapped_range()
        .map_err(|e| e.to_string())?;
    let mut result = Vec::new();
    for row in data.chunks_exact(stride as usize) {
        result.extend_from_slice(&row[..f.extent.width as usize * 4]);
    }
    drop(data);
    buffer.unmap();
    Ok(result)
}
pub(crate) fn run_registered(c: &crate::SuiteContext<'_>) -> Result<serde_json::Value> {
    let worker = GpuWorker::spawn(c.device.clone(), c.queue.clone(), Default::default())
        .map_err(|e| e.to_string())?;
    let bundle = material(
        &worker,
        1,
        Extent::new(17, 13).unwrap(),
        TypedSourcePixels::NormalRgb8 {
            pixels: [180, 150, 240].repeat(17 * 13).into(),
            convention: NormalConvention::OpenGl,
        },
        MapKind::Normal,
    )?;
    let size = proxy_extent(c.extent).map_err(|e| e.to_string())?;
    let a = render(&worker, request(bundle.clone(), 1, size, 97))?;
    let pixels = read(&c.device, &c.queue, &a)?;
    // Independent expected quarter-turn mapping at tile centers, not shader strings.
    let normal = [180., 150., 240.].map(|v| v / 255. * 2. - 1.);
    let len = normal.iter().map(|v| v * v).sum::<f32>().sqrt();
    let normal = normal.map(|v| v / len);
    for y in 0..3 {
        for x in 0..3 {
            let turn = compare::quarter_turn([x, y], 97);
            let [nx, ny, nz] = normal;
            let expected = match turn {
                0 => [nx, ny, nz],
                1 => [-ny, nx, nz],
                2 => [-nx, -ny, nz],
                _ => [ny, -nx, nz],
            }
            .map(|v| ((v * 0.5 + 0.5) * 255.).round() as i32);
            let px = ((2 * x + 1) * size.width / 6).min(size.width - 1);
            let py = ((2 * y + 1) * size.height / 6).min(size.height - 1);
            let at = ((py * size.width + px) * 4) as usize;
            require(
                pixels[at..at + 3]
                    .iter()
                    .zip(expected)
                    .all(|(v, e)| (i32::from(*v) - e).abs() <= 1),
                "GPU normal tangent xy rotation disagrees with oracle",
            )?;
        }
    }
    let repeat = render(&worker, request(bundle.clone(), 2, size, 97))?;
    require(
        read(&c.device, &c.queue, &repeat)? == pixels,
        "seeded repeat changed pixels",
    )?;
    drop(repeat);
    let changed = render(&worker, request(bundle.clone(), 3, size, 98))?;
    require(
        read(&c.device, &c.queue, &changed)? != pixels,
        "new seed did not change tile rotations",
    )?;
    drop(changed);
    // Supersede one half before publication; only the latest request can escape.
    worker
        .request_preview(request(bundle.clone(), 4, size, 99))
        .map_err(|e| e.to_string())?;
    worker
        .request_preview(request(bundle.clone(), 5, size, 100))
        .map_err(|e| e.to_string())?;
    let latest = wait(&worker, || worker.try_latest_preview())?;
    require(latest.request_id == 5, "stale comparison half published")?;
    drop(latest);
    drop(a);
    let mut lit = request(bundle.clone(), 6, size, 97);
    lit.view.mode = ViewMode::ThreeD;
    lit.view.camera.yaw = 0.3;
    lit.view.light.direction = [1., 1., 1.];
    lit.content = PreviewContent::Pbr(pbr::PbrSettings {
        exposure: 1.,
        ..Default::default()
    });
    let first = render(&worker, lit.clone())?;
    let image = read(&c.device, &c.queue, &first)?;
    lit.request_id = 7;
    let second = render(&worker, lit)?;
    require(
        image == read(&c.device, &c.queue, &second)?,
        "same camera/light/exposure pair differs",
    )?;
    drop(first);
    drop(second);
    drop(bundle);
    // NPOT white proves area weights normalize exactly and public albedo obeys proxy size.
    let source = Extent::new(2049, 1025).unwrap();
    let white = material(
        &worker,
        2,
        source,
        TypedSourcePixels::Rgba8 {
            pixels: [255, 255, 255, 255]
                .repeat(source.width as usize * source.height as usize)
                .into(),
            interpretation: InputInterpretation::SrgbPhoto,
        },
        MapKind::Albedo,
    )?;
    let texture = white.outputs[&MapKind::Albedo].texture();
    require(
        texture.width() == white.extent.width
            && texture.height() == white.extent.height
            && texture.width() <= 1024,
        "albedo bypassed declared proxy",
    )?;
    let mut r = request(white, 8, size, 0);
    r.content = PreviewContent::Map {
        map: MapKind::Albedo,
        tiles: Tiles::default(),
        convention: None,
    };
    let frame = render(&worker, r)?;
    require(
        read(&c.device, &c.queue, &frame)?.iter().all(|v| *v == 255),
        "NPOT white area proxy changed white",
    )?;
    drop(frame);
    let d = worker.diagnostics();
    worker.join().map_err(|e| e.to_string())?;
    require(
        d.validation_errors == 0 && d.peak_bytes <= 1_073_741_824,
        "compare validation/budget failure",
    )?;
    Ok(
        serde_json::json!({"suite":"compare","passed":true,"adapter":c.info.name,"backend":format!("{:?}",c.info.backend),"size":size.width,"checks":["gpu-tangent-xy-quarter-turn-oracle","seed-repeatability-and-change","matching-camera-light-exposure-pair","stale-half-rejection","npot-public-color-proxy-white","sequential-proxy-budget"],"peak_bytes":d.peak_bytes,"validation_errors":d.validation_errors,"peak_pending":d.peak_pending,"peak_in_flight":d.peak_in_flight}),
    )
}
