use super::material::{GIB, MIB, request, reservations};
use texelith_core::Extent;
use texelith_gpu::{
    MaterialError, MaterialPlan, ResourceReservations,
    pool::{AllocationDesc, FenceId, PoolError, ResourcePool},
};

#[test]
fn long_4k_chain_uses_liveness_instead_of_reserving_every_intermediate() {
    // 48 scalar intermediates alone would take 3 GiB if kept alive together.
    let r = request(1, 1, 1, 1, 4096, 48);
    let plan = MaterialPlan::new(&r, GIB, reservations()).unwrap();
    assert!(plan.peak_bytes() > 144 * MIB);
    assert!(
        plan.peak_bytes() <= GIB,
        "peak {} exceeds 1 GiB",
        plan.peak_bytes()
    );
    assert_eq!(plan.output_keys().len(), 2);
}

#[test]
fn reservations_consume_the_same_budget_and_low_budget_returns_bounded_error() {
    let r = request(1, 1, 1, 1, 4096, 3);
    let none = ResourceReservations {
        viewport_bytes: 0,
        readback_bytes: 0,
    };
    let base = MaterialPlan::new(&r, GIB, none).unwrap();
    let with_view = MaterialPlan::new(&r, GIB, reservations()).unwrap();
    assert_eq!(with_view.peak_bytes(), base.peak_bytes() + 144 * MIB);
    let export = ResourceReservations {
        viewport_bytes: 144 * MIB,
        readback_bytes: 128 * MIB,
    };
    let with_readback = MaterialPlan::new(&r, GIB, export).unwrap();
    assert!(with_readback.peak_bytes() >= with_view.peak_bytes());
    assert!(with_readback.peak_bytes() <= GIB);
    assert!(MaterialPlan::new(&r, with_view.peak_bytes(), reservations()).is_ok());
    assert!(matches!(
        MaterialPlan::new(&r, with_view.peak_bytes() - 1, reservations()),
        Err(MaterialError::BudgetExceeded { .. })
    ));
    assert!(matches!(
        MaterialPlan::new(&r, 1, reservations()),
        Err(MaterialError::BudgetExceeded { .. })
    ));
    let overflow = ResourceReservations {
        viewport_bytes: u64::MAX,
        readback_bytes: 128 * MIB,
    };
    assert!(MaterialPlan::new(&r, GIB, overflow).is_err());
}

fn bytes(n: u64) -> AllocationDesc {
    AllocationDesc::Buffer { bytes: n }
}

#[test]
fn solver_scratch_and_readback_cannot_overlap_at_peak_until_scratch_fence() {
    let mut pool = ResourcePool::new(GIB);
    let fixed = pool
        .allocate_plan(&[
            bytes(448 * MIB), // The current material bundle.
            bytes(144 * MIB), // Viewport targets and IBL.
            bytes(64 * MIB),  // Source.
            bytes(64 * MIB),  // Miscellaneous tracked allocations.
        ])
        .unwrap();
    assert_eq!(fixed.len(), 4);
    let scratch = pool.allocate(bytes(192 * MIB)).unwrap();
    assert_eq!(pool.diagnostics().allocated_bytes, 912 * MIB);
    pool.mark_submitted(scratch, FenceId(10)).unwrap();
    pool.release(scratch).unwrap();
    let readback = AllocationDesc::Readback {
        extent: Extent::new(4096, 4096).unwrap(),
        bytes_per_pixel: 8,
    };
    assert_eq!(readback.byte_len().unwrap(), 128 * MIB);
    for fence in [0, 9] {
        pool.complete(FenceId(fence));
        assert!(matches!(
            pool.allocate(readback),
            Err(PoolError::BudgetExceeded { .. })
        ));
        assert_eq!(pool.diagnostics().allocated_bytes, 912 * MIB);
        assert_eq!(pool.diagnostics().retired_bytes, 192 * MIB);
    }
    pool.complete(FenceId(10));
    let readback_id = pool.allocate(readback).unwrap();
    assert!(
        !pool.contains(scratch),
        "completed scratch must be evicted to admit readback"
    );
    assert!(pool.contains(readback_id));
    assert_eq!(pool.diagnostics().allocated_bytes, 848 * MIB);
    assert!(pool.diagnostics().peak_bytes <= GIB);
}

#[test]
fn old_material_draw_fence_retires_bundle_while_display_stays_pinned() {
    let mut pool = ResourcePool::new(GIB);
    let old = pool.allocate(bytes(448 * MIB)).unwrap();
    let display = pool.allocate(bytes(64 * MIB)).unwrap();
    let viewport = pool.allocate(bytes(144 * MIB)).unwrap();
    pool.retire_ui(old, FenceId(20)).unwrap();
    pool.complete(FenceId(19));
    assert!(matches!(
        pool.allocate(bytes(448 * MIB)),
        Err(PoolError::BudgetExceeded { .. })
    ));
    pool.complete(FenceId(20));
    let next = pool.allocate(bytes(448 * MIB)).unwrap();
    assert!(pool.contains(next));
    assert!(pool.contains(display));
    assert!(pool.contains(viewport));
    assert_eq!(pool.diagnostics().allocated_bytes, 656 * MIB);
    assert!(pool.diagnostics().peak_bytes <= GIB);
}

#[test]
fn failed_multiallocation_reservation_is_atomic_and_does_not_consume_capacity() {
    let mut pool = ResourcePool::new(100);
    let display = pool.allocate(bytes(40)).unwrap();
    let before = pool.diagnostics();
    assert!(pool.allocate_plan(&[bytes(30), bytes(31)]).is_err());
    assert_eq!(pool.diagnostics().allocated_bytes, before.allocated_bytes);
    assert_eq!(pool.diagnostics().peak_bytes, before.peak_bytes);
    assert!(pool.contains(display));
    let ids = pool.allocate_plan(&[bytes(30), bytes(30)]).unwrap();
    assert_eq!(ids.len(), 2);
    assert_ne!(ids[0], ids[1]);
    assert_eq!(pool.diagnostics().allocated_bytes, 100);
}

#[test]
fn unpinned_reusable_allocations_are_evicted_before_rejecting_reservation() {
    let mut pool = ResourcePool::new(100);
    let cached = pool.allocate(bytes(70)).unwrap();
    let display = pool.allocate(bytes(20)).unwrap();
    pool.release(cached).unwrap();
    let new = pool.allocate(bytes(80)).unwrap();
    assert!(!pool.contains(cached));
    assert!(pool.contains(display));
    assert!(pool.contains(new));
    assert_eq!(pool.diagnostics().allocated_bytes, 100);
    assert_eq!(pool.diagnostics().peak_bytes, 100);
}
