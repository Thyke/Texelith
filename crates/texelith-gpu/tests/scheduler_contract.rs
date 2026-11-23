//! Production scheduler behavioral contracts.
mod worker_support;

use std::sync::Arc;
use texelith_core::NodeId;
use texelith_gpu::scheduler::{Completion, SaveEvent, Scheduler, SchedulerError, WorkKind};
use texelith_gpu::{Quality, RenderRequest};
use worker_support::{PROXY, SOURCE, snapshot};

fn request(id: u64, generation: u64, quality: Quality) -> RenderRequest {
    RenderRequest {
        request_id: id,
        snapshot: snapshot(generation, 7, 1.0),
        output: NodeId(4),
        extent: if quality == Quality::Full {
            SOURCE
        } else {
            PROXY
        },
        quality,
    }
}

#[test]
fn scheduler_ten_thousand_updates_keep_one_in_flight_and_one_latest_pending() {
    let mut scheduler = Scheduler::default();
    scheduler.request(request(1, 1, Quality::Proxy)).unwrap();
    let first = scheduler.start_next().unwrap();
    for id in 2..=10_000 {
        scheduler.request(request(id, id, Quality::Proxy)).unwrap();
        assert!(scheduler.start_next().is_none());
        let stats = scheduler.diagnostics();
        assert_eq!(stats.in_flight, 1);
        assert_eq!(stats.pending, 1);
    }
    scheduler
        .request(request(10_001, 10_000, Quality::Full))
        .unwrap();
    assert_eq!(scheduler.complete(first.id), Completion::Discarded);
    assert!(scheduler.try_latest().is_none());
    let last = scheduler.start_next().unwrap();
    assert_eq!(last.request.request_id, 10_001);
    assert_eq!(last.request.quality, Quality::Full);
    assert_eq!(scheduler.complete(last.id), Completion::Published);
    assert_eq!(scheduler.try_latest().unwrap().request_id, 10_001);
    assert!(scheduler.try_latest().is_none());
    let stats = scheduler.diagnostics();
    assert_eq!(stats.peak_in_flight, 1);
    assert_eq!(stats.peak_pending, 1);
    assert_eq!(stats.submitted, 2);
    assert_eq!(stats.coalesced, 9_999);
    assert_eq!(stats.stale_publications, 0);
}

#[test]
fn scheduler_unknown_duplicate_and_delayed_callbacks_cannot_complete_other_work() {
    let mut scheduler = Scheduler::default();
    scheduler.request(request(10, 4, Quality::Proxy)).unwrap();
    let old = scheduler.start_next().unwrap();
    assert_eq!(scheduler.complete(old.id + 100), Completion::Unknown);
    assert_eq!(scheduler.diagnostics().in_flight, 1);
    scheduler.request(request(11, 5, Quality::Full)).unwrap();
    assert_eq!(scheduler.complete(old.id), Completion::Discarded);
    let current = scheduler.start_next().unwrap();
    assert_ne!(old.id, current.id);
    assert_eq!(scheduler.complete(old.id), Completion::Unknown);
    assert_eq!(scheduler.diagnostics().in_flight, 1);
    assert_eq!(scheduler.complete(current.id), Completion::Published);
    assert_eq!(scheduler.complete(current.id), Completion::Unknown);
    assert_eq!(scheduler.try_latest().unwrap().request_id, 11);
}

#[test]
fn scheduler_same_generation_full_supersedes_submitted_proxy() {
    let mut scheduler = Scheduler::default();
    scheduler.request(request(1, 8, Quality::Proxy)).unwrap();
    let proxy = scheduler.start_next().unwrap();
    scheduler.request(request(2, 8, Quality::Full)).unwrap();
    assert_eq!(scheduler.complete(proxy.id), Completion::Discarded);
    assert!(scheduler.try_latest().is_none());
    let full = scheduler.start_next().unwrap();
    assert_eq!(scheduler.complete(full.id), Completion::Published);
    let published = scheduler.try_latest().unwrap();
    assert_eq!(published.quality, Quality::Full);
    assert_eq!(published.extent, SOURCE);
    assert_eq!(published.snapshot.generation, 8);
}

#[test]
fn scheduler_stale_request_or_generation_never_replaces_latest_intent() {
    for stale in [
        request(9, 10, Quality::Proxy),
        request(11, 9, Quality::Full),
    ] {
        let mut scheduler = Scheduler::default();
        scheduler.request(request(10, 10, Quality::Full)).unwrap();
        assert!(scheduler.request(stale).is_err());
        let current = scheduler.start_next().unwrap();
        assert_eq!(current.request.request_id, 10);
        assert_eq!(scheduler.complete(current.id), Completion::Published);
    }
}

#[test]
fn scheduler_request_id_must_increase_even_when_generation_increases() {
    let mut scheduler = Scheduler::default();
    scheduler.request(request(100, 1, Quality::Proxy)).unwrap();
    assert!(scheduler.request(request(100, 2, Quality::Full)).is_err());
    assert!(scheduler.request(request(99, 2, Quality::Full)).is_err());
    scheduler.request(request(101, 2, Quality::Full)).unwrap();
    assert_eq!(scheduler.start_next().unwrap().request.request_id, 101);
}

#[test]
fn scheduler_source_replacement_discards_old_work_and_retains_new_source() {
    let mut scheduler = Scheduler::default();
    scheduler.request(request(1, 1, Quality::Full)).unwrap();
    let old = scheduler.start_next().unwrap();
    let mut replacement = request(2, 2, Quality::Full);
    replacement.snapshot = snapshot(2, 99, 1.0);
    let expected = Arc::clone(&replacement.snapshot);
    scheduler.request(replacement).unwrap();
    assert_eq!(scheduler.complete(old.id), Completion::Discarded);
    let new = scheduler.start_next().unwrap();
    assert!(Arc::ptr_eq(&new.request.snapshot, &expected));
    assert_eq!(scheduler.complete(new.id), Completion::Published);
    assert!(Arc::ptr_eq(
        &scheduler.try_latest().unwrap().snapshot,
        &expected
    ));
}

#[test]
fn scheduler_output_switch_without_graph_edit_invalidates_submitted_output() {
    let mut scheduler = Scheduler::default();
    scheduler.request(request(1, 1, Quality::Full)).unwrap();
    let normal = scheduler.start_next().unwrap();
    let mut height = request(2, 1, Quality::Full);
    height.output = NodeId(3);
    scheduler.request(height).unwrap();
    assert_eq!(scheduler.complete(normal.id), Completion::Discarded);
    let next = scheduler.start_next().unwrap();
    assert_eq!(next.request.output, NodeId(3));
    assert_eq!(scheduler.complete(next.id), Completion::Published);
}

#[test]
fn scheduler_publication_mailbox_replaces_unconsumed_frame() {
    let mut scheduler = Scheduler::default();
    for id in 1..=3 {
        scheduler.request(request(id, id, Quality::Full)).unwrap();
        let work = scheduler.start_next().unwrap();
        assert_eq!(scheduler.complete(work.id), Completion::Published);
    }
    assert_eq!(scheduler.try_latest().unwrap().request_id, 3);
    assert!(scheduler.try_latest().is_none());
}

#[test]
fn scheduler_accepting_new_intent_removes_unconsumed_stale_completion() {
    let mut scheduler = Scheduler::default();
    scheduler.request(request(1, 1, Quality::Full)).unwrap();
    let work = scheduler.start_next().unwrap();
    scheduler.complete(work.id);
    scheduler.request(request(2, 2, Quality::Proxy)).unwrap();
    assert!(scheduler.try_latest().is_none());
}

#[test]
fn scheduler_close_without_completion_discards_pending_and_never_waits() {
    let mut scheduler = Scheduler::default();
    scheduler.request(request(1, 1, Quality::Full)).unwrap();
    let work = scheduler.start_next().unwrap();
    scheduler.request(request(2, 2, Quality::Full)).unwrap();
    // No completion has been delivered: close must return synchronously anyway.
    scheduler.close();
    scheduler.close();
    assert_eq!(scheduler.diagnostics().pending, 0);
    assert_eq!(scheduler.diagnostics().in_flight, 1);
    assert!(matches!(
        scheduler.request(request(3, 3, Quality::Full)),
        Err(SchedulerError::Closed)
    ));
    assert!(scheduler.start_next().is_none());
    assert_eq!(scheduler.complete(work.id), Completion::Discarded);
    assert_eq!(scheduler.diagnostics().in_flight, 0);
    assert!(scheduler.try_latest().is_none());
}

#[test]
fn scheduler_save_is_single_outstanding_and_interactive_work_has_priority() {
    let mut scheduler = Scheduler::default();
    scheduler.request(request(1, 1, Quality::Proxy)).unwrap();
    let save_id = scheduler
        .request_save(request(2, 1, Quality::Full))
        .unwrap();
    assert!(matches!(
        scheduler.request_save(request(3, 1, Quality::Full)),
        Err(SchedulerError::Busy)
    ));
    let interactive = scheduler.start_next().unwrap();
    assert_eq!(interactive.kind, WorkKind::Interactive);
    assert_eq!(scheduler.complete(interactive.id), Completion::Published);
    let save = scheduler.start_next().unwrap();
    assert_eq!(save.kind, WorkKind::Save);
    assert_eq!(save.request.extent, SOURCE);
    assert_eq!(save.request.quality, Quality::Full);
    // Busy remains true during readback, not just while the save is pending.
    assert!(matches!(
        scheduler.request_save(request(4, 1, Quality::Full)),
        Err(SchedulerError::Busy)
    ));
    assert_eq!(scheduler.complete(save.id), Completion::Saved);
    assert!(
        matches!(scheduler.try_save_event(), Some(SaveEvent::Completed { save_id: id, .. }) if id == save_id)
    );
    assert_eq!(scheduler.try_latest().unwrap().request_id, 1);
    assert!(scheduler.try_latest().is_none());
    scheduler
        .request_save(request(5, 1, Quality::Full))
        .unwrap();
}

#[test]
fn scheduler_save_rejects_proxy_and_partial_full_extent() {
    let mut scheduler = Scheduler::default();
    scheduler.request(request(1, 1, Quality::Proxy)).unwrap();
    assert!(
        scheduler
            .request_save(request(2, 1, Quality::Proxy))
            .is_err()
    );
    let mut undersized = request(3, 1, Quality::Full);
    undersized.extent = PROXY;
    assert!(scheduler.request_save(undersized).is_err());
    scheduler
        .request_save(request(4, 1, Quality::Full))
        .unwrap();
}

#[test]
fn scheduler_graph_or_source_change_cancels_save_before_or_after_submission() {
    for source_changed in [false, true] {
        for submitted in [false, true] {
            let mut scheduler = Scheduler::default();
            scheduler.request(request(1, 1, Quality::Full)).unwrap();
            let render = scheduler.start_next().unwrap();
            scheduler.complete(render.id);
            scheduler.try_latest();
            let save_id = scheduler
                .request_save(request(2, 1, Quality::Full))
                .unwrap();
            let active_save = submitted.then(|| scheduler.start_next().unwrap());
            let mut changed = request(3, 2, Quality::Proxy);
            changed.snapshot = snapshot(2, if source_changed { 8 } else { 7 }, 2.0);
            scheduler.request(changed).unwrap();
            if let Some(save) = active_save {
                assert_eq!(scheduler.complete(save.id), Completion::Discarded);
            }
            assert!(
                matches!(scheduler.try_save_event(), Some(SaveEvent::Cancelled { save_id: id }) if id == save_id)
            );
            assert!(scheduler.try_save_event().is_none());
            assert!(scheduler.try_latest().is_none());
            assert_eq!(scheduler.start_next().unwrap().kind, WorkKind::Interactive);
        }
    }
}

#[test]
fn scheduler_quality_change_in_same_graph_does_not_cancel_full_save() {
    let mut scheduler = Scheduler::default();
    scheduler.request(request(1, 1, Quality::Proxy)).unwrap();
    let render = scheduler.start_next().unwrap();
    scheduler.complete(render.id);
    scheduler.try_latest();
    let save_id = scheduler
        .request_save(request(2, 1, Quality::Full))
        .unwrap();
    let save = scheduler.start_next().unwrap();
    scheduler.request(request(3, 1, Quality::Full)).unwrap();
    assert_eq!(scheduler.complete(save.id), Completion::Saved);
    assert!(
        matches!(scheduler.try_save_event(), Some(SaveEvent::Completed { save_id: id, .. }) if id == save_id)
    );
    assert!(scheduler.try_latest().is_none());
}

#[test]
fn scheduler_save_completion_keeps_exact_requested_snapshot_output_and_extent() {
    let mut scheduler = Scheduler::default();
    scheduler.request(request(1, 5, Quality::Proxy)).unwrap();
    let render = scheduler.start_next().unwrap();
    scheduler.complete(render.id);
    scheduler.try_latest();
    let mut height_save = request(2, 5, Quality::Full);
    height_save.output = NodeId(3);
    let save_id = scheduler.request_save(height_save).unwrap();
    let save = scheduler.start_next().unwrap();
    assert_eq!(save.request.output, NodeId(3));
    assert_eq!(scheduler.complete(save.id), Completion::Saved);
    match scheduler.try_save_event().unwrap() {
        SaveEvent::Completed {
            save_id: id,
            request_id,
            generation,
            extent,
            output,
        } => {
            assert_eq!(id, save_id);
            assert_eq!(request_id, 2);
            assert_eq!(generation, 5);
            assert_eq!(extent, SOURCE);
            assert_eq!(output, NodeId(3));
        }
        other => panic!("expected exact full-map save, got {other:?}"),
    }
    assert!(scheduler.try_latest().is_none());
}

#[test]
fn scheduler_shutdown_explicitly_cancels_pending_or_in_flight_save() {
    for submitted in [false, true] {
        let mut scheduler = Scheduler::default();
        scheduler.request(request(1, 1, Quality::Proxy)).unwrap();
        let render = scheduler.start_next().unwrap();
        scheduler.complete(render.id);
        scheduler.try_latest();
        let save_id = scheduler
            .request_save(request(2, 1, Quality::Full))
            .unwrap();
        let active = submitted.then(|| scheduler.start_next().unwrap());
        scheduler.close();
        if let Some(save) = active {
            assert_eq!(scheduler.complete(save.id), Completion::Discarded);
        }
        assert!(
            matches!(scheduler.try_save_event(), Some(SaveEvent::Cancelled { save_id: id }) if id == save_id)
        );
        assert!(scheduler.try_save_event().is_none());
        assert!(scheduler.start_next().is_none());
        assert!(matches!(
            scheduler.request_save(request(3, 1, Quality::Full)),
            Err(SchedulerError::Closed)
        ));
    }
}

#[test]
fn scheduler_save_of_old_generation_is_rejected_without_consuming_save_slot() {
    let mut scheduler = Scheduler::default();
    scheduler.request(request(1, 5, Quality::Proxy)).unwrap();
    assert!(
        scheduler
            .request_save(request(2, 4, Quality::Full))
            .is_err()
    );
    scheduler
        .request_save(request(3, 5, Quality::Full))
        .unwrap();
}

#[test]
fn scheduler_device_loss_notifies_and_clears_pending_without_recovery() {
    let mut scheduler = Scheduler::default();
    scheduler.request(request(1, 1, Quality::Full)).unwrap();
    let work = scheduler.start_next().unwrap();
    let save_id = scheduler
        .request_save(request(2, 1, Quality::Full))
        .unwrap();
    scheduler.request(request(3, 1, Quality::Proxy)).unwrap();
    scheduler.fail(SchedulerError::DeviceLost);
    assert!(matches!(
        scheduler.try_error(),
        Some(SchedulerError::DeviceLost)
    ));
    assert!(scheduler.try_error().is_none());
    assert_eq!(scheduler.diagnostics().pending, 0);
    assert!(scheduler.request(request(4, 2, Quality::Full)).is_err());
    assert!(scheduler.start_next().is_none());
    assert_eq!(scheduler.complete(work.id), Completion::Discarded);
    assert!(scheduler.try_latest().is_none());
    assert!(
        matches!(scheduler.try_save_event(), Some(SaveEvent::Cancelled { save_id: id }) if id == save_id)
    );
}

#[test]
fn scheduler_unconsumed_save_result_keeps_readback_slot_busy() {
    let mut scheduler = Scheduler::default();
    scheduler.request(request(1, 1, Quality::Proxy)).unwrap();
    let render = scheduler.start_next().unwrap();
    scheduler.complete(render.id);
    scheduler.try_latest();
    scheduler
        .request_save(request(2, 1, Quality::Full))
        .unwrap();
    let save = scheduler.start_next().unwrap();
    assert_eq!(scheduler.complete(save.id), Completion::Saved);
    assert!(matches!(
        scheduler.request_save(request(3, 1, Quality::Full)),
        Err(SchedulerError::Busy)
    ));
    scheduler.try_save_event().unwrap();
    scheduler
        .request_save(request(4, 1, Quality::Full))
        .unwrap();
}
