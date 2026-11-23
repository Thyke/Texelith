use texelith_core::Extent;
use texelith_gpu::{
    pool::{AllocationDesc, FenceId, ResourcePool},
    viewport_resources::ViewportTargetPlan,
};

#[test]
fn framebuffer_targets_preserve_aspect_cap_and_exact_formats() {
    for (w, h, expected, scale) in [
        (1, 1, (1, 1), 1.0),
        (1, 4096, (1, 2048), 0.5),
        (4096, 1, (2048, 1), 0.5),
        (1920, 1080, (1920, 1080), 1.0),
        (4096, 2160, (2048, 1080), 0.5),
        (8192, 4096, (2048, 1024), 0.25),
        (257, 193, (257, 193), 1.0),
    ] {
        let plan = ViewportTargetPlan::new(w, h).unwrap();
        assert_eq!(plan.extent, Extent::new(expected.0, expected.1).unwrap());
        assert!((plan.scale - scale).abs() < 1e-6);
        let allocations = plan.allocations();
        assert_eq!(allocations.len(), 3, "HDR + depth + immutable display");
        for (allocation, format) in allocations.iter().zip([
            wgpu::TextureFormat::Rgba16Float,
            wgpu::TextureFormat::Depth32Float,
            wgpu::TextureFormat::Rgba8Unorm,
        ]) {
            let AllocationDesc::Texture {
                extent,
                format: actual,
                usage,
            } = *allocation
            else {
                panic!("target must be a texture")
            };
            assert_eq!(extent, plan.extent);
            assert_eq!(actual, format);
            if format == wgpu::TextureFormat::Depth32Float {
                assert!(usage.contains(wgpu::TextureUsages::RENDER_ATTACHMENT));
                assert!(!usage.contains(wgpu::TextureUsages::STORAGE_BINDING));
            } else {
                assert!(usage.contains(wgpu::TextureUsages::TEXTURE_BINDING));
            }
        }
        let sum: u64 = allocations.iter().map(|a| a.byte_len().unwrap()).sum();
        assert_eq!(sum, u64::from(expected.0) * u64::from(expected.1) * 16);
    }
    assert!(ViewportTargetPlan::new(0, 1).is_err());
    assert!(ViewportTargetPlan::new(1, 0).is_err());
    assert!(ViewportTargetPlan::new(0, 0).is_err());
    let largest = ViewportTargetPlan::new(u32::MAX, u32::MAX).unwrap();
    assert_eq!(largest.extent, Extent::new(2048, 2048).unwrap());
}

#[test]
fn target_admission_is_atomic_and_includes_depth_in_the_shared_ledger() {
    let plan = ViewportTargetPlan::new(37, 23).unwrap();
    let allocations = plan.allocations();
    let bytes = 37 * 23 * 16;
    let mut pool = ResourcePool::new(bytes - 1);
    assert!(pool.allocate_plan(&allocations).is_err());
    assert_eq!(
        pool.diagnostics().allocated_bytes,
        0,
        "failed plan allocated partial targets"
    );
    let mut pool = ResourcePool::new(bytes);
    assert_eq!(pool.allocate_plan(&allocations).unwrap().len(), 3);
    assert_eq!(pool.diagnostics().allocated_bytes, bytes);
    assert!(pool.allocate(AllocationDesc::Buffer { bytes: 1 }).is_err());
}

#[test]
fn externally_held_and_display_fenced_textures_cannot_be_reused() {
    let desc = AllocationDesc::Texture {
        extent: Extent::new(17, 3).unwrap(),
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
    };
    let bytes = desc.byte_len().unwrap();
    let mut pool = ResourcePool::new(bytes);
    let id = pool.allocate(desc).unwrap();
    pool.retain(id).unwrap();
    pool.mark_submitted(id, FenceId(2)).unwrap();
    pool.retire_ui(id, FenceId(9)).unwrap();
    pool.complete(FenceId(2));
    assert!(
        pool.allocate(desc).is_err(),
        "external clone still owns the texture"
    );
    pool.release(id).unwrap();
    pool.complete(FenceId(8));
    assert_eq!(pool.diagnostics().retired_bytes, bytes);
    assert!(
        pool.allocate(desc).is_err(),
        "compute completion is not the final display fence"
    );
    pool.complete(FenceId(9));
    assert_eq!(pool.diagnostics().retired_bytes, 0);
    assert_eq!(
        pool.allocate(desc).unwrap(),
        id,
        "reuse becomes legal only after display completion"
    );
}
