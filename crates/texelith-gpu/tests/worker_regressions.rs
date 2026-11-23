use texelith_gpu::pool::{AllocationDesc, FenceId, ResourcePool};

#[test]
fn pool_queue_staging_leaves_accounting_only_after_its_fence() {
    let mut pool = ResourcePool::new(64);
    let id = pool
        .allocate(AllocationDesc::Staging { bytes: 64 })
        .unwrap();
    pool.mark_submitted(id, FenceId(3)).unwrap();
    pool.release(id).unwrap();
    pool.complete(FenceId(2));
    assert_eq!(pool.diagnostics().allocated_bytes, 64);
    assert_eq!(pool.diagnostics().retired_bytes, 64);
    assert!(pool.allocate(AllocationDesc::Staging { bytes: 1 }).is_err());
    pool.complete(FenceId(3));
    assert_eq!(pool.diagnostics().allocated_bytes, 0);
    assert_eq!(pool.diagnostics().peak_bytes, 64);
    assert!(!pool.contains(id));

    let unused = pool
        .allocate(AllocationDesc::Staging { bytes: 64 })
        .unwrap();
    pool.release(unused).unwrap();
    assert_eq!(pool.diagnostics().allocated_bytes, 0);
}
