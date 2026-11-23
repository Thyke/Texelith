use super::{
    gpu::with_gpu,
    oracle::{Image, compare},
};
use std::sync::{Arc, Mutex};
use texelith_core::Extent;
use texelith_gpu::{
    PreparedMinMaxRemap,
    pool::{FenceId, ResourcePool},
};

#[test]
fn gpu_min_max_remap_is_explicit_finite_and_preserves_tiny_nonzero_ranges() {
    with_gpu(|gpu| {
        let adjacent = f32::from_bits(0.5f32.to_bits() + 1);
        for (pixels, expected) in [
            (vec![-2.0, 0.0, 6.0], vec![0.0, 0.25, 1.0]),
            (vec![0.5, adjacent], vec![0.0, 1.0]),
            (vec![-f32::MAX, 0.0, f32::MAX], vec![0.0, 0.5, 1.0]),
        ] {
            let image = Image::new(
                pixels.len() as u32,
                1,
                pixels.into_iter().map(f64::from).collect(),
            );
            let input = gpu.scalar(&image);
            let output = gpu.texture(image.extent, wgpu::TextureFormat::R32Float);
            let pool = Arc::new(Mutex::new(ResourcePool::new(1024 * 1024)));
            let mut remap =
                PreparedMinMaxRemap::new(&gpu.executor, &input, &output, 0.5, pool.clone())
                    .unwrap();
            assert!(!remap.passes().is_empty());
            assert!(pool.lock().unwrap().diagnostics().allocated_bytes > 0);
            let mut encoder = gpu.device.create_command_encoder(&Default::default());
            for pass in remap.passes() {
                pass.encode(&mut encoder);
            }
            remap.mark_submitted(FenceId(1)).unwrap();
            gpu.queue.submit([encoder.finish()]);
            gpu.wait();
            remap.complete(FenceId(1));
            compare(&gpu.read_scalar(&output), &expected, 2e-5);
            drop(remap);
            assert_eq!(pool.lock().unwrap().diagnostics().allocated_bytes, 0);
        }
    });
}

#[test]
fn gpu_constant_min_max_returns_configured_midlevel_on_npot_and_singleton_axes() {
    with_gpu(|gpu| {
        for (w, h, value, midlevel) in [
            (1, 1, -7.0, 0.5),
            (1, 17, 0.0, 0.25),
            (17, 1, 2.0, 0.75),
            (257, 193, 0.123, 0.5),
        ] {
            let image = Image::new(w, h, vec![value; (w * h) as usize]);
            let input = gpu.scalar(&image);
            let output = gpu.texture(image.extent, wgpu::TextureFormat::R32Float);
            let pool = Arc::new(Mutex::new(ResourcePool::new(4 * 1024 * 1024)));
            let mut remap =
                PreparedMinMaxRemap::new(&gpu.executor, &input, &output, midlevel, pool).unwrap();
            let mut encoder = gpu.device.create_command_encoder(&Default::default());
            for pass in remap.passes() {
                pass.encode(&mut encoder);
            }
            remap.mark_submitted(FenceId(1)).unwrap();
            gpu.queue.submit([encoder.finish()]);
            gpu.wait();
            remap.complete(FenceId(1));
            compare(
                &gpu.read_scalar(&output),
                &vec![f64::from(midlevel); (w * h) as usize],
                2e-5,
            );
        }
    });
}

#[test]
fn gpu_min_max_includes_npot_last_pixel_and_rows_in_global_extrema() {
    with_gpu(|gpu| {
        let mut pixels = vec![2.0; 257 * 193];
        pixels[0] = -2.0;
        *pixels.last_mut().unwrap() = 6.0;
        let image = Image::new(257, 193, pixels);
        let input = gpu.scalar(&image);
        let output = gpu.texture(image.extent, wgpu::TextureFormat::R32Float);
        let pool = Arc::new(Mutex::new(ResourcePool::new(4 * 1024 * 1024)));
        let mut remap =
            PreparedMinMaxRemap::new(&gpu.executor, &input, &output, 0.5, pool).unwrap();
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        for pass in remap.passes() {
            pass.encode(&mut encoder);
        }
        remap.mark_submitted(FenceId(1)).unwrap();
        gpu.queue.submit([encoder.finish()]);
        gpu.wait();
        remap.complete(FenceId(1));
        let expected: Vec<f64> = image.pixels.iter().map(|v| (v + 2.0) / 8.0).collect();
        compare(&gpu.read_scalar(&output), &expected, 2e-5);
    });
}

#[test]
fn remap_rejects_alias_bad_textures_invalid_midlevel_and_atomic_budget_failure() {
    with_gpu(|gpu| {
        let e = Extent::new(5, 3).unwrap();
        let input = gpu.texture(e, wgpu::TextureFormat::R32Float);
        let output = gpu.texture(e, wgpu::TextureFormat::R32Float);
        let wrong_extent = gpu.texture(Extent::new(3, 5).unwrap(), wgpu::TextureFormat::R32Float);
        let wrong_format = gpu.texture(e, wgpu::TextureFormat::Rgba16Float);
        let restricted = |usage| {
            gpu.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("T-047 restricted usage"),
                size: input.size(),
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R32Float,
                usage,
                view_formats: &[],
            })
        };
        let no_read = restricted(wgpu::TextureUsages::COPY_DST);
        let no_write = restricted(wgpu::TextureUsages::TEXTURE_BINDING);
        let pool = Arc::new(Mutex::new(ResourcePool::new(1024 * 1024)));
        for (read, write) in [
            (&input, &input),
            (&input, &wrong_extent),
            (&input, &wrong_format),
            (&wrong_format, &output),
            (&no_read, &output),
            (&input, &no_write),
        ] {
            assert!(
                PreparedMinMaxRemap::new(&gpu.executor, read, write, 0.5, pool.clone()).is_err()
            );
            assert_eq!(pool.lock().unwrap().diagnostics().allocated_bytes, 0);
        }
        for midlevel in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.1, 1.1] {
            assert!(
                PreparedMinMaxRemap::new(&gpu.executor, &input, &output, midlevel, pool.clone())
                    .is_err()
            );
            assert_eq!(pool.lock().unwrap().diagnostics().allocated_bytes, 0);
        }
        let tiny = Arc::new(Mutex::new(ResourcePool::new(1)));
        assert!(
            PreparedMinMaxRemap::new(&gpu.executor, &input, &output, 0.5, tiny.clone()).is_err()
        );
        assert_eq!(tiny.lock().unwrap().diagnostics().allocated_bytes, 0);
        let mut active =
            PreparedMinMaxRemap::new(&gpu.executor, &input, &output, 0.5, pool.clone()).unwrap();
        active.mark_submitted(FenceId(5)).unwrap();
        let charged = pool.lock().unwrap().diagnostics().allocated_bytes;
        drop(active);
        assert_eq!(pool.lock().unwrap().diagnostics().retired_bytes, charged);
        pool.lock().unwrap().complete(FenceId(4));
        assert_eq!(pool.lock().unwrap().diagnostics().retired_bytes, charged);
        pool.lock().unwrap().complete(FenceId(5));
        assert_eq!(pool.lock().unwrap().diagnostics().retired_bytes, 0);
    });
}
