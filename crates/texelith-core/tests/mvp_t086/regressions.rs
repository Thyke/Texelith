use super::*;

fn continuing_job(id: u64) -> (Session, BatchJob) {
    let (session, base) = job(id);
    let job = BatchJob::new(
        id,
        base.recipe().clone(),
        base.export_settings().clone(),
        vec![input("a"), input("b"), input("c")],
        true,
    )
    .unwrap();
    (session, job)
}

fn enabled(session: &Session, id: CommandId) -> bool {
    session
        .availability()
        .iter()
        .find(|a| a.id == id)
        .unwrap()
        .enabled
}

fn finish(job: &mut BatchJob, index: usize) {
    for state in [
        BatchState::Loading,
        BatchState::Evaluating,
        BatchState::Encoding,
        BatchState::Committed,
    ] {
        transition(job, index, state).unwrap();
    }
}

#[test]
fn session_retry_cannot_replace_a_continuing_active_job() {
    let (mut session, job) = continuing_job(200);
    session.set_batch(job).unwrap();
    session.dispatch(CommandRequest::BatchStart {}).unwrap();
    let first = session.batch().unwrap().pending_work().unwrap();
    session.batch_fail(first, "bad input".into()).unwrap();
    assert!(enabled(&session, CommandId::BatchRetryFailed));
    session.dispatch(CommandRequest::BatchStart {}).unwrap();
    let before = session.batch().unwrap().clone();
    assert!(!enabled(&session, CommandId::BatchStart));
    assert!(!enabled(&session, CommandId::BatchRetryFailed));
    assert_eq!(
        session.dispatch(CommandRequest::BatchRetryFailed {}),
        Err(CommandError::Batch(BatchError::ActiveItem))
    );
    assert_eq!(session.batch(), Some(&before));
    let (_, replacement) = job_fixture(201);
    assert_eq!(session.set_batch(replacement), Err(BatchError::ActiveItem));
    assert_eq!(session.batch(), Some(&before));
    let work = before.pending_work().unwrap();
    session
        .batch_transition(work, BatchState::Evaluating)
        .unwrap();
    assert_eq!(
        session.batch().unwrap().items()[1].state,
        BatchState::Evaluating
    );
}

// Avoid shadowing the root fixture with the job local in some scenarios.
fn job_fixture(id: u64) -> (Session, BatchJob) {
    job(id)
}

#[test]
fn completions_validate_job_item_generation_and_request_before_mutating() {
    let (_, mut job) = continuing_job(210);
    transition(&mut job, 0, BatchState::Loading).unwrap();
    let loading = job.pending_work().unwrap();
    for stale in [
        BatchWorkId {
            item: BatchItemId {
                job: 209,
                ..loading.item
            },
            ..loading
        },
        BatchWorkId {
            item: BatchItemId {
                index: 1,
                ..loading.item
            },
            ..loading
        },
        BatchWorkId {
            generation: loading.generation + 1,
            ..loading
        },
        BatchWorkId {
            request: loading.request + 1,
            ..loading
        },
    ] {
        let before = job.clone();
        assert!(matches!(
            job.transition(stale, BatchState::Evaluating),
            Err(BatchError::StaleCompletion { .. })
        ));
        assert!(matches!(
            job.fail(stale, "stale failure".into()),
            Err(BatchError::StaleCompletion { .. })
        ));
        assert_eq!(job, before);
    }
    job.transition(loading, BatchState::Evaluating).unwrap();
    let before = job.clone();
    assert!(matches!(
        job.transition(loading, BatchState::Encoding),
        Err(BatchError::StaleCompletion { .. })
    ));
    assert!(matches!(
        job.fail(loading, "duplicate failure".into()),
        Err(BatchError::StaleCompletion { .. })
    ));
    assert_eq!(job, before);
    transition(&mut job, 0, BatchState::Encoding).unwrap();
    transition(&mut job, 0, BatchState::Committed).unwrap();
    transition(&mut job, 1, BatchState::Loading).unwrap();
    let before = job.clone();
    assert!(matches!(
        job.fail(loading, "late item zero failure".into()),
        Err(BatchError::StaleCompletion { .. })
    ));
    assert_eq!(job, before);
}

#[test]
fn retry_rejects_previous_job_results_even_when_index_and_generation_match() {
    let (_, mut old) = job(220);
    transition(&mut old, 0, BatchState::Loading).unwrap();
    let stale = old.pending_work().unwrap();
    old.fail(stale, "bad decode".into()).unwrap();
    let mut retry = old.retry_failed(221).unwrap();
    transition(&mut retry, 0, BatchState::Loading).unwrap();
    assert_eq!(retry.pending_work().unwrap().generation, stale.generation);
    assert_eq!(retry.pending_work().unwrap().request, stale.request);
    let before = retry.clone();
    assert!(matches!(
        retry.transition(stale, BatchState::Evaluating),
        Err(BatchError::StaleCompletion { .. })
    ));
    assert!(matches!(
        retry.fail(stale, "late failure".into()),
        Err(BatchError::StaleCompletion { .. })
    ));
    assert_eq!(retry, before);
}

#[test]
fn cancellation_keeps_work_owned_until_the_exact_retirement_arrives() {
    let (mut session, job) = continuing_job(230);
    session.set_batch(job).unwrap();
    session.dispatch(CommandRequest::BatchStart {}).unwrap();
    let failed = session.batch().unwrap().pending_work().unwrap();
    session.batch_fail(failed, "decode failure".into()).unwrap();
    session.dispatch(CommandRequest::BatchStart {}).unwrap();
    let pending = session.batch().unwrap().pending_work().unwrap();
    session.dispatch(CommandRequest::BatchCancel {}).unwrap();
    let before = session.batch().unwrap().clone();
    assert_eq!(before.pending_work(), Some(pending));
    assert!(!enabled(&session, CommandId::BatchRetryFailed));
    assert!(matches!(
        session.dispatch(CommandRequest::BatchRetryFailed {}),
        Err(CommandError::Batch(BatchError::ActiveItem))
    ));
    assert!(matches!(
        before.resume_incomplete(231),
        Err(BatchError::ActiveItem)
    ));
    assert!(
        session
            .batch_transition(pending, BatchState::Evaluating)
            .is_err()
    );
    assert!(session.batch_fail(pending, "late failure".into()).is_err());
    assert!(matches!(
        session.retire_cancelled_batch(failed),
        Err(BatchError::StaleCompletion { .. })
    ));
    assert_eq!(session.batch(), Some(&before));
    session.retire_cancelled_batch(pending).unwrap();
    assert!(enabled(&session, CommandId::BatchRetryFailed));
    session
        .dispatch(CommandRequest::BatchRetryFailed {})
        .unwrap();
    let retry = session.batch().unwrap();
    assert_eq!(retry.id(), 231);
    assert_eq!(retry.items().len(), 1);
    assert_eq!(retry.items()[0].index, 0);
    assert_eq!(retry.items()[0].state, BatchState::Queued);
    assert!(matches!(
        session.retire_cancelled_batch(pending),
        Err(BatchError::StaleCompletion { .. })
    ));
}

#[test]
fn session_queue_commands_leave_the_visible_document_and_captured_recipe_unchanged() {
    let (mut session, mut job) = job(240);
    finish(&mut job, 0);
    session.set_batch(job).unwrap();
    let captured = session.batch().unwrap().recipe().clone();
    let captured_settings = session.batch().unwrap().export_settings().clone();
    let mut edited = captured_settings.clone();
    edited.naming = "edited_{name}_{map}".into();
    session
        .dispatch(CommandRequest::SetExportSettings { settings: edited })
        .unwrap();
    let visible = session.snapshot();
    let history = session.history();
    let view = session.view_state().clone();
    let revision = session.document_revision();
    let request = session.request_id();
    let dirty = session.is_dirty();
    let export = session.export_settings().cloned();
    session.dispatch(CommandRequest::BatchStart {}).unwrap();
    let work = session.batch().unwrap().pending_work().unwrap();
    session.batch_fail(work, "bad second image".into()).unwrap();
    assert!(!enabled(&session, CommandId::BatchStart));
    let before = session.batch().cloned();
    assert_eq!(
        session.dispatch(CommandRequest::BatchStart {}),
        Err(CommandError::Batch(BatchError::StoppedOnError))
    );
    assert_eq!(session.batch().cloned(), before);
    session.dispatch(CommandRequest::BatchCancel {}).unwrap();
    session
        .dispatch(CommandRequest::BatchRetryFailed {})
        .unwrap();
    assert_eq!(
        session
            .batch()
            .unwrap()
            .items()
            .iter()
            .map(|i| i.index)
            .collect::<Vec<_>>(),
        vec![1]
    );
    assert_eq!(session.batch().unwrap().recipe(), &captured);
    assert_eq!(
        session.batch().unwrap().export_settings(),
        &captured_settings
    );
    assert_eq!(session.snapshot(), visible);
    assert_eq!(session.history(), history);
    assert_eq!(session.view_state(), &view);
    assert_eq!(session.document_revision(), revision);
    assert_eq!(session.request_id(), request);
    assert_eq!(session.is_dirty(), dirty);
    assert_eq!(session.export_settings().cloned(), export);
}

#[test]
fn session_never_reuses_a_previously_admitted_job_identity() {
    let (mut session, first) = job(250);
    session.set_batch(first).unwrap();
    let (_, same) = job(250);
    assert_eq!(
        session.set_batch(same),
        Err(BatchError::DuplicateJobId { id: 250 })
    );
    let (_, older) = job(249);
    assert_eq!(
        session.set_batch(older),
        Err(BatchError::DuplicateJobId { id: 249 })
    );
    let (_, later) = job(251);
    session.set_batch(later).unwrap();
    assert_eq!(session.batch().unwrap().id(), 251);
}

#[test]
fn preflight_rejects_collisions_across_items_outputs_and_inputs() {
    let (_, base) = job(260);
    let construct = |inputs| {
        BatchJob::new(
            261,
            base.recipe().clone(),
            base.export_settings().clone(),
            inputs,
            false,
        )
    };
    let mut a = input("a");
    let mut b = input("b");
    b.destinations = a.destinations.clone();
    assert!(matches!(
        construct(vec![a.clone(), b]),
        Err(BatchError::DuplicateDestination { .. })
    ));
    a.destinations.push(a.destinations[0].clone());
    assert!(matches!(
        construct(vec![a]),
        Err(BatchError::DuplicateDestination { .. })
    ));
    let mut a = input("a");
    a.destinations = vec![item("b")];
    assert!(matches!(
        construct(vec![a, input("b")]),
        Err(BatchError::DuplicateDestination { .. })
    ));
    let mut missing = input("a");
    missing.destinations.clear();
    assert!(matches!(
        construct(vec![missing]),
        Err(BatchError::InvalidCheckpoint(_))
    ));
    assert!(matches!(
        construct(vec![]),
        Err(BatchError::InvalidCheckpoint(_))
    ));
    assert!(matches!(
        construct(vec![input("a"), input("a")]),
        Err(BatchError::InvalidCheckpoint(_))
    ));
    assert!(construct(vec![input("a"), input("b")]).is_ok());
    let mut upper = input("b");
    upper.destinations[0] = item("OUT/A_ROUGHNESS.PNG");
    assert!(matches!(
        construct(vec![input("a"), upper]),
        Err(BatchError::DuplicateDestination { .. })
    ));
}

#[test]
fn a_fully_committed_checkpoint_has_nothing_to_retry_or_resume() {
    let (_, mut job) = job(265);
    for index in 0..3 {
        finish(&mut job, index);
    }
    let restored = BatchJob::restore(job.checkpoint()).unwrap();
    assert!(!restored.can_start());
    assert!(!restored.can_cancel());
    assert!(!restored.can_retry_failed());
    assert_eq!(restored.retry_failed(266), Err(BatchError::NoFailedItems));
    assert_eq!(
        restored.resume_incomplete(266),
        Err(BatchError::NoIncompleteItems)
    );
}

#[test]
fn legacy_source_substitution_preserves_parameters_and_validates_interpretation() {
    let original_source = SourceDescriptor {
        interpretation: InputInterpretation::LinearHeight,
        ..source(1)
    };
    let graph = Graph::new(
        vec![
            Node {
                id: NodeId(1),
                input: None,
                params: NodeParams::Source(original_source.clone()),
            },
            Node {
                id: NodeId(2),
                input: Some(NodeId(1)),
                params: NodeParams::Levels {
                    black: 0.1,
                    white: 0.8,
                    gamma: 1.7,
                },
            },
        ],
        NodeId(2),
    )
    .unwrap();
    let replacement = SourceDescriptor {
        content_hash: [5; 32],
        extent: Extent {
            width: 31,
            height: 17,
        },
        ..original_source.clone()
    };
    let changed = instantiate_recipe(&graph, replacement.clone()).unwrap();
    assert_eq!(
        changed.node(NodeId(1)).unwrap().params,
        NodeParams::Source(replacement)
    );
    assert_eq!(changed.node(NodeId(2)), graph.node(NodeId(2)));
    assert_eq!(changed.selected_output(), graph.selected_output());
    assert_eq!(
        graph.node(NodeId(1)).unwrap().params,
        NodeParams::Source(original_source)
    );
    assert!(
        instantiate_recipe(
            &graph,
            SourceDescriptor {
                interpretation: InputInterpretation::LinearNormal,
                ..source(7)
            }
        )
        .is_err()
    );
}

#[test]
fn mixed_resolution_substitution_preserves_painted_roughness_and_metallic_recipes() {
    let (session, _) = fixture();
    let visible = session.snapshot().unwrap();
    let strokes = vec![BrushStroke {
        id: StrokeId(7),
        brush: Brush {
            radius: 0.2,
            hardness: 0.6,
            opacity: 0.75,
            mode: MaskMode::Add,
        },
        points: vec![
            BrushSample {
                uv: [0.25, 0.5],
                pressure: 0.8,
            },
            BrushSample {
                uv: [0.275, 0.5],
                pressure: 1.0,
            },
        ],
    }];
    let mut dto = MaterialGraphDto::from(visible.graph.clone());
    for (index, kind) in [MapKind::Roughness, MapKind::Metallic]
        .into_iter()
        .enumerate()
    {
        let mask = OutputRef {
            node: NodeId(100 + index as u64),
            port: OutputPort::Mask,
        };
        let consumer = dto.outputs[&kind].node;
        dto.graph
            .iter_mut()
            .find(|node| node.id == consumer)
            .unwrap()
            .inputs
            .push(InputEdge {
                port: InputPort::Mask,
                source: mask,
            });
        dto.graph.push(MaterialNode {
            id: mask.node,
            inputs: vec![],
            params: MaterialNodeParams::PaintMask {
                boundary: texelith_core::Boundary::Repeat,
                extent: source(1).extent,
                stroke_digest: stroke_digest(&strokes).unwrap(),
                strokes: strokes.clone(),
            },
        });
    }
    let recipe = Graph::try_from(dto).unwrap();
    let original = recipe.clone();
    for extent in [
        Extent {
            width: 31,
            height: 17,
        },
        Extent {
            width: 8,
            height: 9,
        },
    ] {
        let replacement = SourceDescriptor {
            extent,
            ..source(9)
        };
        let changed = instantiate_recipe(&recipe, replacement.clone()).unwrap();
        assert_eq!(recipe, original);
        assert_eq!(session.snapshot().unwrap(), visible);
        assert_eq!(changed.material_outputs(), recipe.material_outputs());
        assert_eq!(changed.selected_output_ref(), recipe.selected_output_ref());
        assert_eq!(
            changed.material_nodes().len(),
            recipe.material_nodes().len()
        );
        for old in recipe.material_nodes() {
            let new = changed.material_node(old.id).unwrap();
            assert_eq!(new.inputs, old.inputs);
            let expected = match &old.params {
                MaterialNodeParams::ColorSource(_) => {
                    MaterialNodeParams::ColorSource(replacement.clone())
                }
                MaterialNodeParams::PaintMask {
                    stroke_digest,
                    strokes,
                    ..
                } => MaterialNodeParams::PaintMask {
                    boundary: texelith_core::Boundary::Repeat,
                    extent,
                    stroke_digest: *stroke_digest,
                    strokes: strokes.clone(),
                },
                params => params.clone(),
            };
            assert_eq!(new.params, expected);
        }
        for kind in [MapKind::Roughness, MapKind::Metallic] {
            let output = changed.material_outputs().unwrap()[&kind];
            assert_eq!(changed.material_output_extent(output).unwrap(), extent);
        }
    }
}

#[test]
fn checkpoint_round_trip_requires_explicit_resume_and_never_requeues_committed() {
    let (_, mut old) = continuing_job(270);
    finish(&mut old, 0);
    transition(&mut old, 1, BatchState::Loading).unwrap();
    transition(&mut old, 1, BatchState::Evaluating).unwrap();
    let checkpoint = old.checkpoint();
    let json = serde_json::to_value(&checkpoint).unwrap();
    assert!(json.get("generation").is_none());
    assert!(json["recipe"].get("generation").is_none());
    assert!(json.get("pending").is_none());
    let decoded: BatchCheckpoint = serde_json::from_value(json).unwrap();
    assert_eq!(decoded, checkpoint);
    let mut restored = BatchJob::restore(decoded).unwrap();
    assert_eq!(restored.pending_work(), None);
    assert!(!restored.can_start());
    assert_eq!(
        restored.items().iter().map(|i| i.state).collect::<Vec<_>>(),
        vec![
            BatchState::Committed,
            BatchState::Cancelled,
            BatchState::Cancelled
        ]
    );
    assert!(transition(&mut restored, 1, BatchState::Loading).is_err());
    assert_eq!(restored.retry_failed(271), Err(BatchError::NoFailedItems));
    let mut resumed = restored.resume_incomplete(271).unwrap();
    assert_eq!(
        resumed.items().iter().map(|i| i.index).collect::<Vec<_>>(),
        vec![1, 2]
    );
    assert_eq!(
        resumed.items()[0].source_digest,
        old.items()[1].source_digest
    );
    assert_eq!(resumed.items()[0].destinations, old.items()[1].destinations);
    assert!(
        resumed
            .items()
            .iter()
            .all(|i| i.id.job == 271 && i.state == BatchState::Queued)
    );
    assert_eq!(resumed.recipe().graph, old.recipe().graph);
    assert_eq!(resumed.export_settings(), old.export_settings());
    assert!(resumed.items().iter().all(|i| {
        !i.destinations
            .iter()
            .any(|d| old.items()[0].destinations.contains(d))
    }));
    // A sparse retry/resume checkpoint must retain original indices too.
    finish(&mut resumed, 1);
    let restored_again = BatchJob::restore(resumed.checkpoint()).unwrap();
    assert_eq!(
        restored_again
            .items()
            .iter()
            .map(|i| i.index)
            .collect::<Vec<_>>(),
        vec![1, 2]
    );
    assert_eq!(
        restored_again.resume_incomplete(272).unwrap().items()[0].index,
        2
    );
}

#[test]
fn recovery_preserves_failures_and_failed_only_retry_excludes_cancelled() {
    let (_, mut old) = continuing_job(280);
    finish(&mut old, 0);
    transition(&mut old, 1, BatchState::Loading).unwrap();
    fail(&mut old, 1, "source digest mismatch".into()).unwrap();
    old.cancel().unwrap();
    let restored = BatchJob::restore(old.checkpoint()).unwrap();
    assert_eq!(
        restored.items()[1].error.as_deref(),
        Some("source digest mismatch")
    );
    let retry = restored.retry_failed(281).unwrap();
    assert_eq!(retry.items().len(), 1);
    assert_eq!(retry.items()[0].index, 1);
    assert_eq!(retry.items()[0].error, None);
    assert_eq!(restored.resume_incomplete(281).unwrap().items().len(), 2);
}

#[test]
fn recovery_rejects_invalid_versions_indices_destinations_recipe_and_error_states() {
    let (_, base) = job(290);
    let checkpoint = base.checkpoint();
    let mut invalid = checkpoint.clone();
    invalid.schema_version += 1;
    assert!(matches!(
        BatchJob::restore(invalid),
        Err(BatchError::InvalidCheckpoint(_))
    ));
    for index in [0, 10_000] {
        let mut invalid = checkpoint.clone();
        invalid.items[1].index = index;
        assert!(matches!(
            BatchJob::restore(invalid),
            Err(BatchError::InvalidCheckpoint(_))
        ));
    }
    let mut invalid = checkpoint.clone();
    invalid.items.swap(0, 1);
    assert!(matches!(
        BatchJob::restore(invalid),
        Err(BatchError::InvalidCheckpoint(_))
    ));
    let mut invalid = checkpoint.clone();
    invalid.items[0].state = BatchState::Committed;
    invalid.items[1].state = BatchState::Failed;
    invalid.items[1].error = Some("failure".into());
    invalid.items[1].input.destinations = invalid.items[0].input.destinations.clone();
    assert!(matches!(
        BatchJob::restore(invalid),
        Err(BatchError::DuplicateDestination { .. })
    ));
    let mut invalid = checkpoint.clone();
    invalid.items[1].state = BatchState::Failed;
    assert!(matches!(
        BatchJob::restore(invalid),
        Err(BatchError::InvalidCheckpoint(_))
    ));
    let mut invalid = checkpoint.clone();
    invalid.items[1].error = Some("not a failed state".into());
    assert!(matches!(
        BatchJob::restore(invalid),
        Err(BatchError::InvalidCheckpoint(_))
    ));
    let mut invalid = checkpoint.clone();
    invalid.items[0].state = BatchState::Loading;
    invalid.items[1].state = BatchState::Encoding;
    assert!(matches!(
        BatchJob::restore(invalid),
        Err(BatchError::InvalidCheckpoint(_))
    ));
    let mut invalid = checkpoint.clone();
    invalid.recipe.export.source_extent.width += 1;
    assert!(matches!(
        BatchJob::restore(invalid),
        Err(BatchError::InvalidRecipe(_))
    ));
    let mut json = serde_json::to_value(checkpoint).unwrap();
    json["ignored_future_policy"] = true.into();
    assert!(serde_json::from_value::<BatchCheckpoint>(json).is_err());
}

#[test]
fn forged_captures_and_empty_failure_messages_are_rejected_transactionally() {
    let (_, mut base) = job(300);
    let mut capture = base.recipe().clone();
    capture.graph_hash[0] ^= 1;
    assert!(matches!(
        BatchJob::new(
            301,
            capture,
            base.export_settings().clone(),
            vec![input("a")],
            false
        ),
        Err(BatchError::InvalidCheckpoint(_))
    ));
    transition(&mut base, 0, BatchState::Loading).unwrap();
    let before = base.clone();
    let work = base.pending_work().unwrap();
    assert!(matches!(
        base.fail(work, String::new()),
        Err(BatchError::InvalidCheckpoint(_))
    ));
    assert_eq!(base, before);
    assert_eq!(base.retire_cancelled(work), Err(BatchError::ActiveItem));
    assert_eq!(base, before);
}
