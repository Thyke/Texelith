//! Separate normal-only benchmark on the presentation device, after UI shutdown.
use crate::{display::SharedGpu, perf_metrics::NormalSample, perf_runtime::Shared};
use std::{
    sync::Arc,
    time::{Duration, Instant},
};

struct Wake(std::thread::Thread);
impl std::task::Wake for Wake {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }
}
fn block_on<F: std::future::Future>(future: F) -> F::Output {
    let waker = Arc::new(Wake(std::thread::current())).into();
    let mut context = std::task::Context::from_waker(&waker);
    let mut future = std::pin::pin!(future);
    loop {
        match future.as_mut().poll(&mut context) {
            std::task::Poll::Ready(v) => return v,
            std::task::Poll::Pending => std::thread::park_timeout(Duration::from_millis(1)),
        }
    }
}
fn wait(device: &wgpu::Device) -> Result<(), String> {
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(10)),
        })
        .map(|_| ())
        .map_err(|e| e.to_string())
}
pub(crate) fn measure(gpu: &SharedGpu, recorder: &Shared) -> Result<(), String> {
    let (device, queue) = (&gpu.device, &gpu.queue);
    let executor = block_on(texelith_gpu::GpuExecutor::new(
        device.clone(),
        queue.clone(),
    ))
    .map_err(|e| e.to_string())?;
    let extent = texelith_core::Extent {
        width: 4096,
        height: 4096,
    };
    let texture = |format| {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("T-020 normal-only reusable allocation"),
            size: wgpu::Extent3d {
                width: 4096,
                height: 4096,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        })
    };
    let height = texture(wgpu::TextureFormat::R32Float);
    let normal = texture(wgpu::TextureFormat::Rgba16Float);
    let mut pass = executor
        .prepare(
            texelith_gpu::Pass::Normal { strength: 1.0 },
            &height,
            &normal,
            extent,
        )
        .map_err(|e| e.to_string())?;
    let timestamp = device.features().contains(
        wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS,
    );
    let query = timestamp.then(|| {
        device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("normal dispatch timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count: 2,
        })
    });
    let resolve = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("normal timestamps resolve"),
        size: 256,
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("normal timestamps readback"),
        size: 16,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    {
        let mut r = recorder.lock().unwrap();
        // Query support without encoder brackets is missing evidence, not a
        // license to silently substitute completion latency on a capable GPU.
        r.trace.gpu_timestamps_available =
            device.features().contains(wgpu::Features::TIMESTAMP_QUERY);
        // Interactive pool is drained first. These allocations never overlap it.
        r.trace.counters.peak_app_bytes = r
            .trace
            .counters
            .peak_app_bytes
            .max(4096 * 4096 * 12 + pass.uniform_bytes() + 272);
    }
    for index in 0..110 {
        pass.update(texelith_gpu::Pass::Normal {
            strength: index as f32 / 11.0,
        })
        .map_err(|e| e.to_string())?;
        let mut encoder = device.create_command_encoder(&Default::default());
        if let Some(query) = &query {
            encoder.write_timestamp(query, 0);
        }
        pass.encode(&mut encoder);
        if let Some(query) = &query {
            encoder.write_timestamp(query, 1);
        }
        let command = encoder.finish();
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        let submitted = Instant::now();
        queue.submit([command]);
        queue.on_submitted_work_done(move || {
            let _ = tx.send(Instant::now());
        });
        wait(device)?;
        let completed = rx
            .recv_timeout(Duration::from_secs(10))
            .map_err(|e| e.to_string())?;
        let duration = if let Some(query) = &query {
            let mut encoder = device.create_command_encoder(&Default::default());
            encoder.resolve_query_set(query, 0..2, &resolve, 0);
            encoder.copy_buffer_to_buffer(&resolve, 0, &readback, 0, 16);
            queue.submit([encoder.finish()]);
            let (tx, rx) = std::sync::mpsc::sync_channel(1);
            readback.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            wait(device)?;
            rx.recv_timeout(Duration::from_secs(10))
                .map_err(|e| e.to_string())?
                .map_err(|e| e.to_string())?;
            let bytes = readback
                .slice(..)
                .get_mapped_range()
                .map_err(|e| e.to_string())?;
            let a = u64::from_le_bytes(bytes[..8].try_into().unwrap());
            let b = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
            drop(bytes);
            readback.unmap();
            let ticks = b
                .checked_sub(a)
                .filter(|n| *n > 0)
                .ok_or("invalid GPU timestamp pair")?;
            Some((ticks as f64 * queue.get_timestamp_period() as f64).ceil() as u64)
        } else {
            None
        };
        let mut r = recorder.lock().unwrap();
        let sample = NormalSample {
            extent: [4096, 4096],
            warmed: index >= 10,
            normal_only: true,
            submitted_ns: Some(r.ns(submitted)),
            completed_ns: Some(r.ns(completed)),
            gpu_duration_ns: duration,
        };
        r.trace.normal_dispatches.push(sample);
    }
    Ok(())
}
