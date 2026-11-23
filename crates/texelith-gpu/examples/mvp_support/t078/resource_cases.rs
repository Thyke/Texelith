use super::{fixtures::*, oracle::scalar};
use std::sync::{Arc, Mutex};
use texelith_core::*;
use texelith_gpu::{
    PreparedExportPacking,
    pool::{FenceId, ResourcePool},
};

pub(super) fn missing_duplicate_proxy_wrong_output_and_alias_sources_are_rejected_before_dispatch(
    gpu: &super::gpu::Gpu,
) {
    {
        let source = scalar(OutputPort::Height, 17, 13, &vec![0.25; 17 * 13]);
        let p = plan(
            extent(17, 13),
            vec![output(
                "height",
                ExportFormat::Png16,
                vec![map(OutputPort::Height, Component::R)],
            )],
        );
        let textures = [gpu.upload(&source)];
        let sources = [source];
        let bindings = gpu.bindings(&sources, &textures);
        let destination = gpu.texture(extent(17, 13), wgpu::TextureFormat::Rgba32Float);
        let pool = Arc::new(Mutex::new(ResourcePool::new(1024 * 1024)));
        assert!(
            PreparedExportPacking::new(&gpu.executor, &p, 0, &[], &destination, pool.clone())
                .is_err()
        );
        assert!(
            PreparedExportPacking::new(&gpu.executor, &p, 1, &bindings, &destination, pool.clone())
                .is_err()
        );
        let mut duplicate = gpu.bindings(&sources, &textures);
        duplicate.extend(gpu.bindings(&sources, &textures));
        assert!(
            PreparedExportPacking::new(
                &gpu.executor,
                &p,
                0,
                &duplicate,
                &destination,
                pool.clone()
            )
            .is_err()
        );
        let proxy = [gpu.texture(extent(8, 6), wgpu::TextureFormat::R32Float)];
        let proxy_binding = gpu.bindings(&sources, &proxy);
        assert!(
            PreparedExportPacking::new(
                &gpu.executor,
                &p,
                0,
                &proxy_binding,
                &destination,
                pool.clone()
            )
            .is_err()
        );
        let wrong_output = gpu.texture(extent(17, 13), wgpu::TextureFormat::Rgba16Float);
        assert!(
            PreparedExportPacking::new(
                &gpu.executor,
                &p,
                0,
                &bindings,
                &wrong_output,
                pool.clone()
            )
            .is_err()
        );
        let wrong_size = gpu.texture(extent(16, 13), wgpu::TextureFormat::Rgba32Float);
        assert!(
            PreparedExportPacking::new(&gpu.executor, &p, 0, &bindings, &wrong_size, pool.clone())
                .is_err()
        );
        let mut albedo = p.clone();
        albedo.outputs[0].channels[0] = map(OutputPort::Albedo, Component::R);
        let alias_sources = [super::gpu::vector(
            OutputPort::Albedo,
            17,
            13,
            vec![[0.0; 4]; 17 * 13],
        )];
        let alias_textures = [destination.clone()];
        let alias_bindings = gpu.bindings(&alias_sources, &alias_textures);
        assert!(
            PreparedExportPacking::new(
                &gpu.executor,
                &albedo,
                0,
                &alias_bindings,
                &destination,
                pool.clone()
            )
            .is_err()
        );
        assert_eq!(pool.lock().unwrap().diagnostics().allocated_bytes, 0);
    }
}

pub(super) fn zero_budget_fails_atomically_and_submitted_resources_retire_only_after_fence(
    gpu: &super::gpu::Gpu,
) {
    {
        let source = scalar(OutputPort::Height, 1, 1, &[0.25]);
        let p = plan(
            extent(1, 1),
            vec![output(
                "height",
                ExportFormat::Png16,
                vec![map(OutputPort::Height, Component::R)],
            )],
        );
        let textures = [gpu.upload(&source)];
        let bindings = gpu.bindings(&[source], &textures);
        let destination = gpu.texture(extent(1, 1), wgpu::TextureFormat::Rgba32Float);
        let denied = Arc::new(Mutex::new(ResourcePool::new(0)));
        assert!(
            PreparedExportPacking::new(
                &gpu.executor,
                &p,
                0,
                &bindings,
                &destination,
                denied.clone()
            )
            .is_err()
        );
        assert_eq!(denied.lock().unwrap().diagnostics().allocated_bytes, 0);
        let pool = Arc::new(Mutex::new(ResourcePool::new(1024 * 1024)));
        let mut prepared =
            PreparedExportPacking::new(&gpu.executor, &p, 0, &bindings, &destination, pool.clone())
                .unwrap();
        let allocated = pool.lock().unwrap().diagnostics().allocated_bytes;
        assert!(
            allocated >= 64,
            "common NodeUniform and task-specific storage must be charged"
        );
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        for pass in prepared.passes() {
            pass.encode(&mut encoder);
        }
        gpu.queue.submit([encoder.finish()]);
        prepared.mark_submitted(FenceId(78)).unwrap();
        drop(prepared);
        assert!(
            pool.lock().unwrap().diagnostics().allocated_bytes > 0,
            "dropped in-flight storage was freed early"
        );
        assert!(pool.lock().unwrap().diagnostics().retired_bytes > 0);
        gpu.wait();
        pool.lock().unwrap().complete(FenceId(78));
        assert_eq!(pool.lock().unwrap().diagnostics().allocated_bytes, 0);
        // Budget rejection must leave the executor usable.
        assert_eq!(
            gpu.run(&p, 0, &[scalar(OutputPort::Height, 1, 1, &[0.25])])
                .len(),
            1
        );
    }
}
