//! Only this adapter anticipates a new API; there is no test-side GPU kernel.
use super::support::{BUDGET, Gpu};
use std::sync::{Arc, Mutex};
use texelith_gpu::{
    pool::{FenceId, ResourcePool},
    seam_heatmap::PreparedSeamHeatmap,
};

#[derive(Debug, PartialEq)]
pub struct ResultData {
    pub heatmap: Vec<f64>,
    pub boundary_rms: f64,
    pub interior_rms: f64,
    pub ratio: f64,
}

pub fn evaluate(gpu: &Gpu, input: &wgpu::Texture) -> ResultData {
    let heatmap = gpu.texture(
        input.width() as usize,
        input.height() as usize,
        wgpu::TextureFormat::R32Float,
    );
    let before = gpu.read_bytes(input);
    let pool = Arc::new(Mutex::new(ResourcePool::new(BUDGET)));
    let mut prepared = PreparedSeamHeatmap::new(&gpu.executor, input, &heatmap, pool.clone())
        .expect("public seam diagnostic must accept typed material output");
    assert!(
        !prepared.passes().is_empty(),
        "diagnostic must encode real work"
    );
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    for pass in prepared.passes() {
        pass.encode(&mut encoder);
    }
    prepared.mark_submitted(FenceId(1)).unwrap();
    gpu.queue.submit([encoder.finish()]);
    gpu.wait();
    let metrics = prepared
        .read_metrics()
        .expect("GPU boundary/interior reduction must be readable");
    prepared.complete(FenceId(1));
    let result = ResultData {
        heatmap: gpu.pixels(&heatmap).iter().map(|p| p[0]).collect(),
        boundary_rms: f64::from(metrics.boundary_rms),
        interior_rms: f64::from(metrics.interior_rms),
        ratio: f64::from(metrics.ratio),
    };
    assert_eq!(
        gpu.read_bytes(input),
        before,
        "diagnostic changed material/export pixels"
    );
    assert!(
        pool.lock().unwrap().diagnostics().peak_bytes > 0,
        "scratch/readback must be budgeted"
    );
    assert!(pool.lock().unwrap().diagnostics().peak_bytes <= BUDGET);
    drop(prepared);
    assert_eq!(
        pool.lock().unwrap().diagnostics().allocated_bytes,
        0,
        "completed diagnostic leaked scratch"
    );
    result
}

#[test]
fn extreme_finite_height_gradients_saturate_without_nonfinite_diagnostics() {
    super::support::with_gpu(|gpu| {
        let input = gpu.texture(17, 11, wgpu::TextureFormat::R32Float);
        let values: Vec<f32> = (0..17 * 11)
            .map(|i| if i % 2 == 0 { f32::MAX } else { -f32::MAX })
            .collect();
        gpu.queue.write_texture(
            input.as_image_copy(),
            bytemuck::cast_slice(&values),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(17 * 4),
                rows_per_image: Some(11),
            },
            input.size(),
        );
        let result = evaluate(gpu, &input);
        assert!(result.heatmap.iter().all(|v| v.is_finite() && *v >= 0.0));
        assert!(result.boundary_rms.is_finite() && result.boundary_rms > 0.0);
        assert!(result.interior_rms.is_finite() && result.interior_rms > 0.0);
        assert!(result.ratio.is_finite());
    });
}

#[test]
fn diagnostic_rejects_alias_extent_format_and_insufficient_budget_without_leaks() {
    super::support::with_gpu(|gpu| {
        let input = gpu.texture(32, 24, wgpu::TextureFormat::R32Float);
        let valid = gpu.texture(32, 24, wgpu::TextureFormat::R32Float);
        let mismatch = gpu.texture(31, 24, wgpu::TextureFormat::R32Float);
        let color = gpu.texture(32, 24, wgpu::TextureFormat::Rgba16Float);
        for output in [&input, &mismatch, &color] {
            let pool = Arc::new(Mutex::new(ResourcePool::new(BUDGET)));
            assert!(PreparedSeamHeatmap::new(&gpu.executor, &input, output, pool.clone()).is_err());
            assert_eq!(pool.lock().unwrap().diagnostics().allocated_bytes, 0);
        }
        let pool = Arc::new(Mutex::new(ResourcePool::new(0)));
        assert!(PreparedSeamHeatmap::new(&gpu.executor, &input, &valid, pool.clone()).is_err());
        assert_eq!(pool.lock().unwrap().diagnostics().allocated_bytes, 0);
        // A failed admission must not poison a subsequent diagnostic.
        let result = evaluate(gpu, &input);
        assert_eq!(
            (result.boundary_rms, result.interior_rms, result.ratio),
            (0.0, 0.0, 0.0)
        );
    });
}
