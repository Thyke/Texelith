use crate::gpu_availability;

use crate::{
    Axis, GpuExecutor, Pass, PreparedPyramid,
    pool::{AllocationDesc, FenceId, ResourcePool},
    resources::BindingKey,
};
use std::sync::{Arc, Mutex};
use texelith_core::{Boundary, Extent, InputInterpretation, SourceDescriptor};

#[test]
fn populated_worker_backing_is_reused_and_evicted_before_pyramid_allocation() {
    if gpu_availability::skip_if_no_gpu() {
        return;
    }
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    }))
    .expect("backing regression requires hardware");
    assert_ne!(adapter.get_info().device_type, wgpu::DeviceType::Cpu);
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_limits: wgpu::Limits {
            max_texture_dimension_2d: 4096,
            ..Default::default()
        },
        ..Default::default()
    }))
    .unwrap();
    let device = Arc::new(device);
    let queue = Arc::new(queue);
    let executor = pollster::block_on(GpuExecutor::new(device.clone(), queue.clone())).unwrap();
    let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let extent = Extent::new(5, 3).unwrap();
    let source = SourceDescriptor {
        content_hash: [9; 32],
        extent,
        interpretation: InputInterpretation::LinearHeight,
    };
    let texture = |extent| AllocationDesc::Texture {
        extent,
        format: wgpu::TextureFormat::R32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
    };
    // Source (60) + complete pyramid including staging (896), exactly.
    let pool = Arc::new(Mutex::new(ResourcePool::new(956)));
    let initial = pool
        .lock()
        .unwrap()
        .allocate_backed(
            &device,
            &[
                texture(extent),
                texture(extent),
                AllocationDesc::Buffer { bytes: 64 },
                texture(Extent::new(3, 2).unwrap()),
                AllocationDesc::Buffer { bytes: 128 },
            ],
        )
        .unwrap();
    let input = initial[0].1.texture().clone();
    let reused_coarse = initial[3].1.texture().clone();
    let coarse_id = initial[3].0;
    let scratch_id = initial[1].0;
    let uniform_id = initial[2].0;
    let evicted_id = initial[4].0;
    let operation = Pass::Blur {
        sigma_source_px: 1.0,
        axis: Axis::X,
    };
    let prepared = executor
        .prepare_pooled(
            operation,
            &input,
            initial[1].1.texture(),
            extent,
            (initial[2].1.buffer(), Some(initial[4].1.buffer())),
        )
        .unwrap();
    let key = BindingKey {
        input: initial[0].0,
        output: scratch_id,
        uniform: uniform_id,
        weights: Some(evicted_id),
        pipeline: operation.index(),
        source_width: 5,
        source_height: 3,
    };
    {
        let mut ledger = pool.lock().unwrap();
        ledger.backing.bindings.insert(key, prepared);
        // Model a worker cache after its submission's actual fence.
        for (id, _) in &initial {
            ledger.mark_submitted(*id, FenceId(1)).unwrap();
        }
        ledger.complete(FenceId(1));
        for (id, _) in &initial[1..] {
            ledger.release(*id).unwrap();
        }
    }
    // Flush the initialization writes before reusing their uniform buffers.
    queue.submit([]);
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    drop(initial);
    let mut chain =
        PreparedPyramid::new(&executor, &source, &input, Boundary::Repeat, pool.clone()).unwrap();
    assert_eq!(
        chain.levels()[1],
        reused_coarse,
        "ledger ID reuse must reuse the worker texture"
    );
    {
        let ledger = pool.lock().unwrap();
        assert!(
            ledger.contains(coarse_id)
                && ledger.contains(scratch_id)
                && ledger.contains(uniform_id)
        );
        assert!(!ledger.contains(evicted_id));
        assert!(!ledger.backing.resources.contains_key(&evicted_id));
        assert!(
            !ledger.backing.bindings.contains_key(&key),
            "bindings must not retain evicted payloads"
        );
        assert_eq!(ledger.backing.resources.len(), ledger.entries.len());
        assert_eq!(ledger.diagnostics().allocated_bytes, 956);
    }
    let mut encoder = device.create_command_encoder(&Default::default());
    for pass in chain.passes() {
        pass.encode(&mut encoder);
    }
    chain.mark_submitted(FenceId(2)).unwrap();
    queue.submit([encoder.finish()]);
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    chain.complete(FenceId(2), true);
    assert!(chain.is_ready());
    drop(chain);
    // A second chain must reuse the same physical allocations too.
    let chain =
        PreparedPyramid::new(&executor, &source, &input, Boundary::Clamp, pool.clone()).unwrap();
    assert_eq!(chain.levels()[1], reused_coarse);
    assert_eq!(pool.lock().unwrap().diagnostics().allocated_bytes, 956);
    drop(chain);
    queue.submit([]);
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    assert!(pollster::block_on(validation.pop()).is_none());
}
