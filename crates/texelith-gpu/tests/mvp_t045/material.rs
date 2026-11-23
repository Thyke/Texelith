use std::sync::Arc;
use texelith_core::{
    Boundary, EvaluationSnapshot, Extent, Graph, InputEdge, InputInterpretation, InputPort,
    MapKind, MaterialGraphDto, MaterialNode, MaterialNodeParams, MaterialOutputs, Node, NodeId,
    NodeParams, OutputPort, OutputRef, SourceDescriptor,
};
use texelith_gpu::{
    CompletedMaterial, GpuError, GpuWorker, JobId, MaterialPlan, MaterialRequest, Quality,
    ResourceReservations,
    scheduler::{Completion, Scheduler},
};

pub const MIB: u64 = 1024 * 1024;
pub const GIB: u64 = 1024 * MIB;

pub fn request(
    job: u64,
    id: u64,
    generation: u64,
    hash: u8,
    size: u32,
    levels: u64,
) -> MaterialRequest {
    let extent = Extent::new(size, size).unwrap();
    let mut nodes = vec![Node {
        id: NodeId(1),
        input: None,
        params: NodeParams::Source(SourceDescriptor {
            content_hash: [hash; 32],
            extent,
            interpretation: InputInterpretation::LinearHeight,
        }),
    }];
    for index in 0..levels {
        nodes.push(Node {
            id: NodeId(index + 2),
            input: Some(NodeId(index + 1)),
            params: NodeParams::Levels {
                black: 0.0,
                white: 1.0,
                gamma: 1.1,
            },
        });
    }
    nodes.push(Node {
        id: NodeId(levels + 2),
        input: Some(NodeId(levels + 1)),
        params: NodeParams::NormalFromHeight { strength: 2.0 },
    });
    let legacy = Graph::new(nodes, NodeId(levels + 2)).unwrap();
    let graph = Graph::try_from(MaterialGraphDto::from(legacy)).unwrap();
    let outputs = graph.material_outputs().unwrap().clone();
    MaterialRequest {
        job_id: JobId(job),
        request_id: id,
        snapshot: Arc::new(EvaluationSnapshot { generation, graph }),
        outputs,
        extent,
        quality: Quality::Full,
    }
}

pub fn reservations() -> ResourceReservations {
    ResourceReservations {
        viewport_bytes: 144 * MIB,
        readback_bytes: 0,
    }
}

#[test]
fn material_worker_api_preserves_legacy_entry_points_and_owns_complete_bundles() {
    let _: fn(&GpuWorker, MaterialRequest) -> Result<(), GpuError> = GpuWorker::request_material;
    let _: fn(&GpuWorker) -> Option<CompletedMaterial> = GpuWorker::try_latest_material;
    let _: fn(&GpuWorker, texelith_gpu::RenderRequest) -> Result<(), GpuError> = GpuWorker::request;
    fn send<T: Send>() {}
    send::<MaterialRequest>();
    send::<CompletedMaterial>();
    let _identity = |bundle: &CompletedMaterial| {
        (
            bundle.job_id,
            bundle.request_id,
            bundle.generation,
            bundle.extent,
            bundle.quality,
        )
    };
    let _leased_outputs = |bundle: &CompletedMaterial| {
        for (kind, lease) in &bundle.outputs {
            let _: &MapKind = kind;
            let _: &wgpu::Texture = lease.texture();
        }
    };
}

#[test]
fn equal_generations_and_reused_request_numbers_from_different_jobs_do_not_alias() {
    for next_source_hash in [1, 2] {
        let mut scheduler = Scheduler::default();
        scheduler
            .request_material(request(7, 1, 9, 1, 16, 2))
            .unwrap();
        let old = scheduler.start_next().unwrap();
        assert!(scheduler.should_continue(old.id));
        // Each document/batch job has its own counter, including the same generation.
        scheduler
            .request_material(request(8, 1, 9, next_source_hash, 16, 2))
            .unwrap();
        assert!(!scheduler.should_continue(old.id));
        assert!(scheduler.start_next().is_none());
        assert_eq!(scheduler.complete(old.id), Completion::Discarded);
        assert!(scheduler.try_latest_material().is_none());
        let current = scheduler.start_next().unwrap();
        assert_ne!(old.id, current.id);
        assert_eq!(scheduler.complete(old.id), Completion::Unknown);
        assert!(scheduler.should_continue(current.id));
        assert_eq!(scheduler.complete(current.id), Completion::Published);
        let bundle = scheduler.try_latest_material().unwrap();
        assert_eq!(bundle.job_id, JobId(8));
        assert_eq!(bundle.request_id, 1);
        assert_eq!(bundle.snapshot.generation, 9);
        assert_eq!(bundle.outputs.len(), 2);
        assert!(scheduler.try_latest_material().is_none());
    }
}

#[test]
fn generations_can_restart_for_a_new_job_but_stale_intent_within_a_job_is_rejected() {
    let mut scheduler = Scheduler::default();
    scheduler
        .request_material(request(1, 500, 90, 1, 16, 1))
        .unwrap();
    scheduler
        .request_material(request(2, 1, 1, 2, 16, 1))
        .unwrap();
    assert!(
        scheduler
            .request_material(request(2, 1, 2, 2, 16, 1))
            .is_err()
    );
    assert!(
        scheduler
            .request_material(request(2, 2, 0, 2, 16, 1))
            .is_err()
    );
    let work = scheduler.start_next().unwrap();
    assert_eq!(scheduler.complete(work.id), Completion::Published);
    assert_eq!(
        scheduler.try_latest_material().unwrap().snapshot.generation,
        1
    );
}

#[test]
fn output_switch_and_proxy_release_cancel_between_passes_without_partial_publication() {
    let mut scheduler = Scheduler::default();
    let mut proxy = request(1, 1, 4, 1, 2048, 2);
    proxy.quality = Quality::Proxy;
    proxy.extent = texelith_gpu::proxy_extent(proxy.extent).unwrap();
    scheduler.request_material(proxy).unwrap();
    let old = scheduler.start_next().unwrap();
    let mut full = request(1, 2, 4, 1, 2048, 2);
    full.outputs.remove(&MapKind::Normal);
    scheduler.request_material(full.clone()).unwrap();
    assert!(!scheduler.should_continue(old.id));
    assert!(scheduler.try_latest_material().is_none());
    assert_eq!(scheduler.complete(old.id), Completion::Discarded);
    let current = scheduler.start_next().unwrap();
    assert!(scheduler.try_latest_material().is_none());
    assert_eq!(scheduler.complete(current.id), Completion::Published);
    let published = scheduler.try_latest_material().unwrap();
    assert_eq!(published.outputs, full.outputs);
    assert_eq!(published.extent, full.extent);
    assert_eq!(published.quality, Quality::Full);
}

#[test]
fn material_updates_share_one_bounded_slot_and_close_never_publishes_cancelled_work() {
    let mut scheduler = Scheduler::default();
    scheduler
        .request_material(request(1, 1, 1, 1, 16, 1))
        .unwrap();
    let first = scheduler.start_next().unwrap();
    for id in 2..=1000 {
        scheduler
            .request_material(request(1, id, id, 1, 16, 1))
            .unwrap();
        assert_eq!(scheduler.diagnostics().in_flight, 1);
        assert_eq!(scheduler.diagnostics().pending, 1);
        assert!(scheduler.start_next().is_none());
    }
    assert_eq!(scheduler.complete(first.id), Completion::Discarded);
    let last = scheduler.start_next().unwrap();
    assert_eq!(scheduler.complete(last.id), Completion::Published);
    let bundle = scheduler.try_latest_material().unwrap();
    assert_eq!(bundle.request_id, 1000);
    assert_eq!(bundle.snapshot.generation, 1000);
    assert_eq!(bundle.outputs.len(), 2);
    assert_eq!(scheduler.diagnostics().peak_in_flight, 1);
    assert_eq!(scheduler.diagnostics().peak_pending, 1);
    scheduler
        .request_material(request(1, 1001, 1001, 1, 16, 1))
        .unwrap();
    let active = scheduler.start_next().unwrap();
    scheduler.close();
    assert!(!scheduler.should_continue(active.id));
    assert_eq!(scheduler.complete(active.id), Completion::Discarded);
    assert!(scheduler.try_latest_material().is_none());
    assert!(scheduler.start_next().is_none());
    assert!(
        scheduler
            .request_material(request(2, 1, 1, 1, 16, 1))
            .is_err()
    );
}

#[test]
fn invalid_output_sets_and_quality_extents_do_not_replace_valid_pending_material() {
    let valid = request(1, 1, 1, 1, 16, 1);
    let mut empty = valid.clone();
    empty.outputs.clear();
    let mut missing = valid.clone();
    missing.outputs.insert(
        MapKind::Height,
        OutputRef {
            node: NodeId(999),
            port: OutputPort::Height,
        },
    );
    let mut mismatched_kind = valid.clone();
    mismatched_kind
        .outputs
        .insert(MapKind::Height, valid.outputs[&MapKind::Normal]);
    let mut wrong_port = valid.clone();
    wrong_port.outputs.insert(
        MapKind::Height,
        OutputRef {
            node: NodeId(2),
            port: OutputPort::Normal,
        },
    );
    let mut wrong_extent = valid.clone();
    wrong_extent.extent = Extent::new(15, 16).unwrap();
    let mut wrong_proxy = valid.clone();
    wrong_proxy.quality = Quality::Proxy;
    wrong_proxy.extent = Extent::new(1, 1).unwrap();
    for mut invalid in [
        empty,
        missing,
        mismatched_kind,
        wrong_port,
        wrong_extent,
        wrong_proxy,
    ] {
        invalid.request_id = 2;
        let mut scheduler = Scheduler::default();
        scheduler.request_material(valid.clone()).unwrap();
        assert!(scheduler.request_material(invalid).is_err());
        let work = scheduler.start_next().unwrap();
        assert_eq!(scheduler.complete(work.id), Completion::Published);
        assert_eq!(scheduler.try_latest_material().unwrap().request_id, 1);
    }
}

#[test]
fn content_keys_ignore_jobs_requests_and_generations_but_separate_two_sources() {
    let first = request(1, 1, 7, 1, 16, 2);
    let same_content = request(900, 120, 7, 1, 16, 2);
    let later = request(1, 200, 999, 1, 16, 2);
    let other_source = request(2, 1, 7, 2, 16, 2);
    let plan = |r: &MaterialRequest| MaterialPlan::new(r, GIB, reservations()).unwrap();
    let original = plan(&first);
    assert_eq!(original.output_keys(), plan(&same_content).output_keys());
    assert_eq!(original.output_keys(), plan(&later).output_keys());
    for kind in [MapKind::Height, MapKind::Normal] {
        assert_ne!(
            original.output_keys()[&kind],
            plan(&other_source).output_keys()[&kind]
        );
    }
    let mut proxy = first;
    proxy.extent = Extent::new(8, 8).unwrap();
    proxy.quality = Quality::Proxy;
    // This extent is intentionally invalid: 16x16 sources already fit the proxy cap.
    assert!(MaterialPlan::new(&proxy, GIB, reservations()).is_err());
}

#[test]
fn horizon_ao_is_supported_when_requested() {
    let base = request(1, 1, 1, 1, 16, 1);
    let mut dto = MaterialGraphDto::from(base.snapshot.graph.clone());
    dto.graph.push(MaterialNode {
        id: NodeId(20),
        inputs: vec![InputEdge {
            port: InputPort::Height,
            source: base.outputs[&MapKind::Height],
        }],
        params: MaterialNodeParams::HorizonAo {
            directions: 8,
            steps: 8,
            radius_source_px: 32.0,
            bias: 0.01,
            boundary: Boundary::Repeat,
        },
    });
    let ao = OutputRef {
        node: NodeId(20),
        port: OutputPort::Ao,
    };
    dto.outputs.insert(MapKind::Ao, ao);
    let mut r = base;
    r.snapshot = Arc::new(EvaluationSnapshot {
        generation: 1,
        graph: Graph::try_from(dto).unwrap(),
    });
    assert!(MaterialPlan::new(&r, GIB, reservations()).is_ok());
    r.outputs.insert(MapKind::Ao, ao);
    assert!(MaterialPlan::new(&r, GIB, reservations()).is_ok());
}

#[test]
fn renumbered_material_nodes_keep_identical_output_content_keys() {
    let first = request(1, 1, 1, 1, 16, 2);
    let mut dto = MaterialGraphDto::from(first.snapshot.graph.clone());
    for node in &mut dto.graph {
        node.id.0 += 100;
        for input in &mut node.inputs {
            input.source.node.0 += 100;
        }
    }
    for output in dto.outputs.values_mut() {
        output.node.0 += 100;
    }
    dto.selected_output.node.0 += 100;
    let outputs: MaterialOutputs = dto.outputs.clone();
    let mut other = first.clone();
    other.snapshot = Arc::new(EvaluationSnapshot {
        generation: 9,
        graph: Graph::try_from(dto).unwrap(),
    });
    other.outputs = outputs;
    let plan = |r: &MaterialRequest| MaterialPlan::new(r, GIB, reservations()).unwrap();
    assert_eq!(plan(&first).output_keys(), plan(&other).output_keys());
}
