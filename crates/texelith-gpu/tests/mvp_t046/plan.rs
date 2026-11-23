use super::gpu::with_gpu;
use std::sync::{Arc, Mutex};
use texelith_core::{Boundary, Extent, InputInterpretation, SourceDescriptor};
use texelith_gpu::{
    PreparedPyramid,
    pool::{FenceId, ResourcePool},
};

#[test]
fn shared_chain_admits_atomically_and_keeps_cancelled_allocations_charged_until_fence() {
    with_gpu(|gpu| {
        let extent = Extent::new(5, 3).unwrap();
        let input = gpu.texture(extent, wgpu::TextureFormat::R32Float);
        let source = SourceDescriptor {
            content_hash: [7; 32],
            extent,
            interpretation: InputInterpretation::LinearHeight,
        };
        let tiny = Arc::new(Mutex::new(ResourcePool::new(64)));
        assert!(
            PreparedPyramid::new(
                &gpu.executor,
                &source,
                &input,
                Boundary::Repeat,
                tiny.clone()
            )
            .is_err()
        );
        assert_eq!(tiny.lock().unwrap().diagnostics().allocated_bytes, 0);
        let pool = Arc::new(Mutex::new(ResourcePool::new(4096)));
        let mut chain = PreparedPyramid::new(
            &gpu.executor,
            &source,
            &input,
            Boundary::Repeat,
            pool.clone(),
        )
        .unwrap();
        assert_eq!(chain.levels().len(), 4);
        assert_eq!(chain.passes().len(), 6);
        // Independent payload sum: shared coarse levels 24+8+4, scratch
        // (60+24+8), plus six 64-byte NodeUniforms and their upload staging.
        assert_eq!(chain.scratch_bytes(), 896);
        assert_eq!(pool.lock().unwrap().diagnostics().allocated_bytes, 896);
        assert!(!chain.is_ready());
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        chain.passes()[0].encode(&mut encoder);
        chain.mark_submitted(FenceId(1)).unwrap();
        gpu.queue.submit([encoder.finish()]);
        gpu.wait();
        chain.complete(FenceId(1), false);
        assert!(
            !chain.is_ready(),
            "partial/cancelled chains cannot be shared"
        );
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        for pass in chain.passes() {
            pass.encode(&mut encoder);
        }
        chain.mark_submitted(FenceId(2)).unwrap();
        gpu.queue.submit([encoder.finish()]);
        gpu.wait();
        chain.complete(FenceId(1), true);
        assert!(!chain.is_ready(), "an old fence cannot publish newer work");
        chain.complete(FenceId(2), true);
        assert!(chain.is_ready());
        chain.mark_submitted(FenceId(3)).unwrap();
        drop(chain);
        assert_eq!(pool.lock().unwrap().diagnostics().retired_bytes, 896);
        pool.lock().unwrap().complete(FenceId(3));
        assert_eq!(pool.lock().unwrap().diagnostics().retired_bytes, 0);
    });
}
