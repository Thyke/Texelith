//! CPU bookkeeping exercised with explicit fake queue fences, never frame delays.
use texelith_core::Extent;
use texelith_gpu::pool::{AllocationDesc, FenceId, PoolError, ResourcePool};
use wgpu::{TextureFormat as Format, TextureUsages as Usage};

const MIB: u64 = 1024 * 1024;

fn texture(width: u32, height: u32, format: Format) -> AllocationDesc {
    AllocationDesc::Texture {
        extent: Extent { width, height },
        format,
        usage: Usage::TEXTURE_BINDING | Usage::STORAGE_BINDING | Usage::COPY_SRC,
    }
}

fn scalar() -> AllocationDesc {
    texture(4, 4, Format::R32Float)
}

#[test]
fn pool_exact_texture_buffer_and_npot_padded_readback_bytes() {
    for (format, bytes) in [
        (Format::R32Float, 64 * MIB),
        (Format::Rgba16Float, 128 * MIB),
        (Format::Rgba8Unorm, 64 * MIB),
    ] {
        assert_eq!(texture(4096, 4096, format).byte_len().unwrap(), bytes);
    }
    let readback = AllocationDesc::Readback {
        extent: Extent {
            width: 65,
            height: 3,
        },
        bytes_per_pixel: 4,
    };
    assert_eq!(readback.byte_len().unwrap(), 512 * 3);
    let mut pool = ResourcePool::new(4096);
    let a = pool.allocate(texture(17, 11, Format::R32Float)).unwrap();
    let b = pool.allocate(readback).unwrap();
    let c = pool.allocate(AllocationDesc::Buffer { bytes: 64 }).unwrap();
    assert_eq!(
        pool.diagnostics().allocated_bytes,
        17 * 11 * 4 + 512 * 3 + 64
    );
    for id in [a, b, c] {
        pool.release(id).unwrap();
    }
    // Reusable resources still count against the hard budget.
    assert_eq!(pool.diagnostics().allocated_bytes, 2348);
}

#[test]
fn pool_invalid_sizes_and_arithmetic_overflow_never_allocate() {
    let mut pool = ResourcePool::new(u64::MAX);
    for desc in [
        texture(0, 4, Format::R32Float),
        texture(4, 0, Format::R32Float),
        AllocationDesc::Buffer { bytes: 0 },
        AllocationDesc::Readback {
            extent: Extent {
                width: u32::MAX,
                height: u32::MAX,
            },
            bytes_per_pixel: u32::MAX,
        },
    ] {
        assert!(pool.allocate(desc).is_err());
        assert_eq!(pool.diagnostics().allocated_bytes, 0);
        assert_eq!(pool.diagnostics().peak_bytes, 0);
    }
}

#[test]
fn pool_reuse_requires_exact_extent_format_and_usage() {
    let mut pool = ResourcePool::new(4096);
    let original = pool.allocate(scalar()).unwrap();
    pool.release(original).unwrap();
    assert_eq!(pool.allocate(scalar()).unwrap(), original);
    pool.release(original).unwrap();
    for desc in [
        texture(8, 2, Format::R32Float),
        texture(4, 4, Format::Rgba8Unorm),
        AllocationDesc::Texture {
            extent: Extent {
                width: 4,
                height: 4,
            },
            format: Format::R32Float,
            usage: Usage::COPY_SRC | Usage::COPY_DST,
        },
    ] {
        let other = pool.allocate(desc).unwrap();
        assert_ne!(other, original);
        pool.release(other).unwrap();
    }
    assert_eq!(pool.allocate(scalar()).unwrap(), original);
}

#[test]
fn pool_clone_leases_pin_until_last_owner_releases_without_double_counting() {
    let mut pool = ResourcePool::new(64);
    let id = pool.allocate(scalar()).unwrap();
    pool.retain(id).unwrap();
    pool.retain(id).unwrap();
    pool.release(id).unwrap();
    pool.release(id).unwrap();
    assert_eq!(pool.diagnostics().allocated_bytes, 64);
    assert!(matches!(
        pool.allocate(scalar()),
        Err(PoolError::BudgetExceeded { .. })
    ));
    pool.release(id).unwrap();
    assert_eq!(pool.allocate(scalar()).unwrap(), id);
    assert_eq!(pool.diagnostics().peak_bytes, 64);
}

#[test]
fn pool_in_flight_release_is_retired_and_cannot_reuse_before_fence() {
    let mut pool = ResourcePool::new(128);
    let first = pool.allocate(scalar()).unwrap();
    pool.mark_submitted(first, FenceId(10)).unwrap();
    pool.release(first).unwrap();
    assert_eq!(pool.diagnostics().retired_bytes, 64);
    let second = pool.allocate(scalar()).unwrap();
    assert_ne!(first, second);
    pool.complete(FenceId(9));
    assert_eq!(pool.diagnostics().retired_bytes, 64);
    assert!(matches!(
        pool.allocate(scalar()),
        Err(PoolError::BudgetExceeded { .. })
    ));
    pool.complete(FenceId(10));
    assert_eq!(pool.diagnostics().retired_bytes, 0);
    assert_eq!(pool.allocate(scalar()).unwrap(), first);
}

#[test]
fn pool_completion_does_not_unpin_display_or_cache_lease() {
    let mut pool = ResourcePool::new(64);
    let id = pool.allocate(scalar()).unwrap();
    pool.mark_submitted(id, FenceId(1)).unwrap();
    pool.complete(FenceId(1));
    assert!(matches!(
        pool.allocate(scalar()),
        Err(PoolError::BudgetExceeded { .. })
    ));
    assert_eq!(pool.diagnostics().allocated_bytes, 64);
    pool.release(id).unwrap();
    assert_eq!(pool.allocate(scalar()).unwrap(), id);
}

#[test]
fn pool_ui_retirement_waits_for_final_ui_submission_not_compute_or_frame_count() {
    let mut pool = ResourcePool::new(64);
    let display = pool.allocate(texture(4, 4, Format::Rgba8Unorm)).unwrap();
    pool.mark_submitted(display, FenceId(1)).unwrap();
    pool.complete(FenceId(1));
    pool.retire_ui(display, FenceId(8)).unwrap();
    for _ in 0..1000 {
        pool.complete(FenceId(1));
        assert!(pool.allocate(texture(4, 4, Format::Rgba8Unorm)).is_err());
    }
    assert_eq!(pool.diagnostics().retired_bytes, 64);
    pool.complete(FenceId(7));
    assert!(pool.allocate(texture(4, 4, Format::Rgba8Unorm)).is_err());
    pool.complete(FenceId(8));
    assert_eq!(
        pool.allocate(texture(4, 4, Format::Rgba8Unorm)).unwrap(),
        display
    );
}

#[test]
fn pool_latest_fence_wins_and_old_callbacks_do_not_retire_reused_resource() {
    let mut pool = ResourcePool::new(64);
    let id = pool.allocate(scalar()).unwrap();
    pool.mark_submitted(id, FenceId(2)).unwrap();
    pool.mark_submitted(id, FenceId(5)).unwrap();
    pool.release(id).unwrap();
    pool.complete(FenceId(2));
    assert!(pool.allocate(scalar()).is_err());
    pool.complete(FenceId(5));
    assert_eq!(pool.allocate(scalar()).unwrap(), id);
    pool.mark_submitted(id, FenceId(9)).unwrap();
    pool.release(id).unwrap();
    pool.complete(FenceId(2));
    pool.complete(FenceId(5));
    assert_eq!(pool.diagnostics().retired_bytes, 64);
    assert!(pool.allocate(scalar()).is_err());
    pool.complete(FenceId(9));
    assert_eq!(pool.allocate(scalar()).unwrap(), id);
}

#[test]
fn pool_lru_evicts_oldest_reusable_and_preserves_recent_and_pinned() {
    let mut pool = ResourcePool::new(192);
    let oldest = pool.allocate(scalar()).unwrap();
    let recent = pool.allocate(texture(8, 2, Format::R32Float)).unwrap();
    let pinned = pool.allocate(texture(2, 8, Format::R32Float)).unwrap();
    pool.release(oldest).unwrap();
    pool.release(recent).unwrap();
    assert_eq!(
        pool.allocate(texture(8, 2, Format::R32Float)).unwrap(),
        recent
    );
    pool.release(recent).unwrap();
    pool.allocate(texture(16, 1, Format::R32Float)).unwrap();
    assert!(!pool.contains(oldest));
    assert!(pool.contains(recent));
    assert!(pool.contains(pinned));
    assert_eq!(pool.diagnostics().allocated_bytes, 192);
    assert_eq!(pool.diagnostics().peak_bytes, 192);
}

#[test]
fn pool_budget_failure_is_structured_and_preserves_all_live_owners() {
    let mut pool = ResourcePool::new(32 * MIB);
    let pinned = pool
        .allocate(AllocationDesc::Buffer { bytes: 1024 })
        .unwrap();
    let before = pool.diagnostics();
    assert!(matches!(
        pool.allocate(texture(4096, 4096, Format::R32Float)),
        Err(PoolError::BudgetExceeded { .. })
    ));
    assert!(pool.contains(pinned));
    assert_eq!(pool.diagnostics().allocated_bytes, before.allocated_bytes);
    assert_eq!(pool.diagnostics().peak_bytes, before.peak_bytes);
}

#[test]
fn pool_whole_plan_admission_is_atomic_and_includes_buffers_and_retired_bytes() {
    let mut pool = ResourcePool::new(192);
    let retired = pool.allocate(scalar()).unwrap();
    pool.mark_submitted(retired, FenceId(4)).unwrap();
    pool.release(retired).unwrap();
    let plan = [scalar(), AllocationDesc::Buffer { bytes: 65 }];
    assert!(matches!(
        pool.allocate_plan(&plan),
        Err(PoolError::BudgetExceeded { .. })
    ));
    assert_eq!(pool.diagnostics().allocated_bytes, 64);
    assert_eq!(pool.diagnostics().peak_bytes, 64);
    assert_eq!(pool.diagnostics().retired_bytes, 64);
    pool.complete(FenceId(4));
    let admitted = pool.allocate_plan(&plan).unwrap();
    assert_eq!(admitted.len(), 2);
    assert_eq!(pool.diagnostics().allocated_bytes, 129);
    assert!(pool.diagnostics().peak_bytes <= 192);
}

#[test]
fn pool_equal_budget_is_allowed_but_one_byte_more_is_rejected() {
    let mut pool = ResourcePool::new(1024 * MIB);
    let id = pool
        .allocate(AllocationDesc::Buffer { bytes: 1024 * MIB })
        .unwrap();
    assert_eq!(pool.diagnostics().allocated_bytes, 1024 * MIB);
    assert!(pool.allocate(AllocationDesc::Buffer { bytes: 1 }).is_err());
    assert_eq!(pool.diagnostics().peak_bytes, 1024 * MIB);
    pool.release(id).unwrap();
}

#[test]
fn pool_double_release_cannot_underflow_or_make_in_flight_bytes_evictable() {
    let mut pool = ResourcePool::new(64);
    let id = pool.allocate(scalar()).unwrap();
    pool.mark_submitted(id, FenceId(6)).unwrap();
    pool.release(id).unwrap();
    assert!(pool.release(id).is_err());
    assert_eq!(pool.diagnostics().allocated_bytes, 64);
    assert_eq!(pool.diagnostics().retired_bytes, 64);
    assert!(pool.allocate(scalar()).is_err());
    pool.complete(FenceId(6));
    assert_eq!(pool.allocate(scalar()).unwrap(), id);
}
