//! T-086 proposed queue API. Names beyond the ADR are a test/implementer contract.
use texelith_core::*;

#[path = "mvp_t086/regressions.rs"]
mod regressions;

fn source(hash: u8) -> SourceDescriptor {
    SourceDescriptor {
        content_hash: [hash; 32],
        extent: Extent {
            width: 16,
            height: 16,
        },
        interpretation: InputInterpretation::SrgbPhoto,
    }
}

fn fixture() -> (Session, ExportPlan) {
    let graph = material_from_source(source(1)).unwrap();
    let output = graph.material_outputs().unwrap()[&MapKind::Roughness];
    let plan = ExportPlan {
        source_extent: source(1).extent,
        size: SizePolicy::Keep,
        padding: Boundary::Repeat,
        displacement: DisplacementSettings::default(),
        naming: "{name}_{map}".into(),
        profile: EngineProfile {
            id: "custom".into(),
            revision: 1,
            engine_safe_srgb: false,
            normal_convention: NormalConvention::OpenGl,
        },
        outputs: vec![ExportOutput {
            name: "roughness".into(),
            format: ExportFormat::Png8,
            channels: vec![ChannelSpec {
                source: ChannelSource::Map(output, Component::R),
                invert: false,
                transfer: Transfer::Linear,
            }],
            normal_has_negative_z: false,
            quantization: None,
            import_hints: None,
        }],
    };
    let mut session = Session::from_snapshot(EvaluationSnapshot {
        generation: 10,
        graph,
    });
    session
        .dispatch(CommandRequest::SetExportSettings {
            settings: plan.clone(),
        })
        .unwrap();
    (session, plan)
}

fn item(name: &str) -> OpaqueLocator {
    OpaqueLocator(name.into())
}

// Mechanical adapters keep the original state assertions while supplying the
// identity now required by worker completion APIs.
fn transition(job: &mut BatchJob, index: usize, to: BatchState) -> Result<(), BatchError> {
    job.transition(job.work_id(index)?, to)
}

fn fail(job: &mut BatchJob, index: usize, error: String) -> Result<(), BatchError> {
    job.fail(job.work_id(index)?, error)
}

fn input(name: &str) -> BatchInput {
    BatchInput {
        source: item(name),
        source_digest: [name.as_bytes()[0]; 32],
        destinations: vec![item(&format!("out/{name}_roughness.png"))],
    }
}

// BatchJob::new captures the graph and settings, with input order defining stable
// item indices. Job and item IDs are distinct identities, independent of generation.
fn job(id: u64) -> (Session, BatchJob) {
    let (session, plan) = fixture();
    let capture = ExportCapture::new(
        &session.snapshot().unwrap(),
        session.document_revision(),
        &plan,
    )
    .unwrap();
    let job = BatchJob::new(
        id,
        capture,
        plan,
        vec![input("a"), input("b"), input("c")],
        false,
    )
    .unwrap();
    (session, job)
}

#[test]
fn substitution_is_immutable_and_preserves_graph_shape_and_parameters() {
    let (session, _) = fixture();
    let before = session.snapshot().unwrap();
    let replacement = source(9);
    let changed = instantiate_recipe(&before.graph, replacement.clone()).unwrap();
    assert_eq!(session.snapshot().unwrap(), before);
    assert_ne!(changed, before.graph);
    let old_nodes = before.graph.material_nodes();
    let new_nodes = changed.material_nodes();
    assert_eq!(old_nodes.len(), new_nodes.len());
    for (old, new) in old_nodes.iter().zip(new_nodes) {
        assert_eq!(old.id, new.id);
        assert_eq!(old.inputs, new.inputs);
        match (&old.params, &new.params) {
            (
                MaterialNodeParams::ColorSource(original),
                MaterialNodeParams::ColorSource(updated),
            ) => {
                assert_eq!(original, &source(1));
                assert_eq!(updated, &replacement);
            }
            _ => assert_eq!(old.params, new.params),
        }
    }
    let mut invalid = replacement;
    invalid.extent.width = 0;
    assert!(instantiate_recipe(&before.graph, invalid).is_err());
    assert_eq!(session.snapshot().unwrap(), before);
}

#[test]
fn state_machine_rejects_skips_and_terminal_reentry_transactionally() {
    let (_, mut job) = job(100);
    assert_eq!(job.items()[0].state, BatchState::Queued);
    let before = job.clone();
    assert!(matches!(
        transition(&mut job, 0, BatchState::Committed),
        Err(BatchError::InvalidTransition { .. })
    ));
    assert_eq!(job, before);
    for state in [
        BatchState::Loading,
        BatchState::Evaluating,
        BatchState::Encoding,
        BatchState::Committed,
    ] {
        transition(&mut job, 0, state).unwrap();
    }
    let before = job.clone();
    assert!(matches!(
        transition(&mut job, 0, BatchState::Loading),
        Err(BatchError::InvalidTransition { .. })
    ));
    assert_eq!(job, before);
    assert!(matches!(
        transition(&mut job, 99, BatchState::Loading),
        Err(BatchError::UnknownItem { .. })
    ));
    assert_eq!(job, before);
}

#[test]
fn one_active_item_cancellation_and_stop_on_error_are_enforced() {
    let (_, mut job) = job(101);
    transition(&mut job, 0, BatchState::Loading).unwrap();
    assert!(transition(&mut job, 1, BatchState::Loading).is_err());
    fail(&mut job, 0, "decode failed".into()).unwrap();
    assert_eq!(job.items()[0].state, BatchState::Failed);
    assert!(
        transition(&mut job, 1, BatchState::Loading).is_err(),
        "stop on error is default"
    );
    job.cancel().unwrap();
    assert_eq!(
        job.items().iter().map(|i| i.state).collect::<Vec<_>>(),
        vec![
            BatchState::Failed,
            BatchState::Cancelled,
            BatchState::Cancelled
        ]
    );
}

#[test]
fn cancellation_preserves_committed_and_retry_contains_failed_only() {
    let (session, mut job) = job(102);
    let visible = session.snapshot();
    let history = session.history();
    let revision = session.document_revision();
    for state in [
        BatchState::Loading,
        BatchState::Evaluating,
        BatchState::Encoding,
        BatchState::Committed,
    ] {
        transition(&mut job, 0, state).unwrap();
    }
    transition(&mut job, 1, BatchState::Loading).unwrap();
    fail(&mut job, 1, "decode failed".into()).unwrap();
    job.cancel().unwrap();
    let retry = job.retry_failed(103).unwrap();
    assert_ne!(job.id(), retry.id());
    assert_eq!(retry.items().len(), 1);
    assert_eq!(retry.items()[0].index, 1);
    assert_eq!(retry.items()[0].source, item("b"));
    assert_eq!(retry.items()[0].state, BatchState::Queued);
    assert_ne!(retry.items()[0].id, job.items()[1].id);
    assert_eq!(retry.recipe(), job.recipe());
    assert_eq!(retry.export_settings(), job.export_settings());
    assert_eq!(job.items()[0].state, BatchState::Committed);
    assert_eq!(session.snapshot(), visible);
    assert_eq!(session.history(), history);
    assert_eq!(session.document_revision(), revision);
    assert!(matches!(
        job.retry_failed(102),
        Err(BatchError::DuplicateJobId { .. })
    ));
}

#[test]
fn queue_commands_are_registered_and_match_dispatch() {
    let ids: Vec<_> = command_descriptors()
        .iter()
        .map(|d| d.id.as_str())
        .collect();
    for id in ["batch.start", "batch.cancel", "batch.retry_failed"] {
        assert_eq!(
            ids.iter().filter(|candidate| **candidate == id).count(),
            1,
            "{id}"
        );
        let request: CommandRequest =
            serde_json::from_value(serde_json::json!({"id": id, "payload": {}})).unwrap();
        assert!(
            Session::new().dispatch(request).is_err(),
            "{id} without a job/document must fail"
        );
    }
}

#[test]
fn explicit_continue_on_error_allows_next_item_and_cancel_invalidates_active() {
    let (session, plan) = fixture();
    let capture = ExportCapture::new(
        &session.snapshot().unwrap(),
        session.document_revision(),
        &plan,
    )
    .unwrap();
    let mut job = BatchJob::new(104, capture, plan, vec![input("a"), input("b")], true).unwrap();
    transition(&mut job, 0, BatchState::Loading).unwrap();
    fail(&mut job, 0, "bad source".into()).unwrap();
    transition(&mut job, 1, BatchState::Loading).unwrap();
    job.cancel().unwrap();
    assert_eq!(job.items()[1].state, BatchState::Cancelled);
    assert!(
        transition(&mut job, 1, BatchState::Evaluating).is_err(),
        "late completion must not revive cancelled work"
    );
}
