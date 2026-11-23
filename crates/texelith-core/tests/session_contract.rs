//! T-014 tests-first contract, reusing all T-011 graph/command/error DTOs.
//!
//! Proposed new API (no implementation/stubs are supplied by this pass):
//! - `Session::new() -> Session` starts without a document.
//! - `Session::from_snapshot(EvaluationSnapshot) -> Session` constructs a session
//!   from an already validated graph and runtime generation, with empty history.
//!   This construction seam makes u64 exhaustion testable without 2^64 edits;
//!   it is not a dispatch command or project deserializer.
//! - `snapshot() -> Option<EvaluationSnapshot>` (None before the first import),
//!   `history() -> HistoryState`, `availability() -> Vec<CommandAvailability>`.
//! - `dispatch(CommandRequest)` and `complete_import(ImportToken, SourceDescriptor)`
//!   return `Result<CommandOutcome, CommandError>` as specified by ADR 0002.
//! - `fail_import(ImportToken) -> Result<CommandOutcome, CommandError>` consumes a
//!   failed decode token without changing the document/history/generation.
//!
//! No focus event enters core: frontend focus loss dispatches CommitGesture.
//! Outcome.snapshot is present for parameter evaluation changes and changed view
//! selection; the latter keeps the generation. Metadata/effect-only requests and
//! no-ops return None. No new dirty-set API is required: recursive cache keys are
//! checked on a branching DAG to verify downstream-only content invalidation.

use std::collections::BTreeMap;
use texelith_core::{
    CacheKey, CommandAvailability, CommandError, CommandId, CommandOutcome, CommandRequest,
    CoreError, Effect, EvaluationSnapshot, Extent, GestureId, Graph, HistoryState, ImportToken,
    InputInterpretation, MapKind, Node, NodeId, NodeKind, NodeParams, OpaqueLocator,
    PARAMETER_DESCRIPTORS, ParameterAddress, ParameterId, ParameterValue, Session,
    SourceDescriptor, command_descriptors, node_cache_key, validate,
};

fn source(byte: u8) -> SourceDescriptor {
    SourceDescriptor {
        content_hash: [byte; 32],
        extent: Extent {
            width: 13,
            height: 7,
        },
        interpretation: InputInterpretation::LinearHeight,
    }
}

fn loaded() -> Session {
    let mut session = Session::new();
    request_import(&mut session, 1);
    session.complete_import(ImportToken(1), source(1)).unwrap();
    session
}

fn snapshot(session: &Session) -> EvaluationSnapshot {
    session.snapshot().expect("loaded document")
}

fn address(session: &Session, parameter: ParameterId) -> ParameterAddress {
    let kind = parameter.descriptor().node_kind;
    let node = snapshot(session)
        .graph
        .nodes()
        .iter()
        .find(|node| node.params.kind() == kind)
        .unwrap()
        .id;
    ParameterAddress { node, parameter }
}

fn value(session: &Session, address: ParameterAddress) -> f32 {
    snapshot(session)
        .graph
        .node(address.node)
        .unwrap()
        .params
        .parameter(address.parameter)
        .unwrap()
}

fn output(session: &Session, map: MapKind) -> NodeId {
    let kind = match map {
        MapKind::Height => NodeKind::Blur,
        MapKind::Normal => NodeKind::NormalFromHeight,
        _ => panic!("legacy session fixture only has height and normal maps"),
    };
    snapshot(session)
        .graph
        .nodes()
        .iter()
        .find(|node| node.params.kind() == kind)
        .unwrap()
        .id
}

fn set(
    session: &mut Session,
    address: ParameterAddress,
    value: f32,
    gesture: Option<GestureId>,
) -> CommandOutcome {
    session
        .dispatch(CommandRequest::SetParameter {
            address,
            value: ParameterValue::Float(value),
            gesture,
        })
        .unwrap()
}

fn begin(session: &mut Session, address: ParameterAddress, gesture: GestureId) {
    let outcome = session
        .dispatch(CommandRequest::BeginGesture { gesture, address })
        .unwrap();
    metadata_only(session, &outcome);
    assert_eq!(outcome.history.active_gesture, Some(gesture));
}

fn request_import(session: &mut Session, token: u64) -> CommandOutcome {
    session
        .dispatch(CommandRequest::NewFromImage {
            token: ImportToken(token),
            source: OpaqueLocator(format!("opaque:source/{token}/ü")),
        })
        .unwrap()
}

fn enabled(session: &Session, id: CommandId) -> bool {
    let availability = session.availability();
    let matches: Vec<_> = availability.iter().filter(|entry| entry.id == id).collect();
    assert_eq!(matches.len(), 1, "one availability record for {id:?}");
    matches[0].enabled
}

fn metadata_only(session: &Session, outcome: &CommandOutcome) {
    assert!(outcome.effects.is_empty());
    assert!(outcome.snapshot.is_none());
    assert_eq!(outcome.history, session.history());
    assert_eq!(outcome.availability, session.availability());
}

#[derive(Debug, PartialEq)]
struct Observable {
    snapshot: Option<EvaluationSnapshot>,
    history: HistoryState,
    availability: Vec<CommandAvailability>,
}

fn observable(session: &Session) -> Observable {
    Observable {
        snapshot: session.snapshot(),
        history: session.history(),
        availability: session.availability(),
    }
}

fn rejected(session: &mut Session, request: CommandRequest, expected: CommandError) {
    let before = observable(session);
    assert_eq!(session.dispatch(request), Err(expected));
    assert_eq!(
        observable(session),
        before,
        "failed requests must be atomic"
    );
}

fn redo_branch(session: &mut Session, address: ParameterAddress) {
    set(session, address, 2.0, None);
    session.dispatch(CommandRequest::Undo).unwrap();
    assert_eq!(session.history().redo_entries, 1);
}

#[test]
fn registry_runtime_availability_covers_every_id_and_tracks_document_and_history() {
    let mut session = Session::new();
    assert!(session.snapshot().is_none());
    assert_eq!(session.history(), HistoryState::default());
    assert_eq!(session.availability().len(), command_descriptors().len());
    for descriptor in command_descriptors() {
        assert_eq!(
            enabled(&session, descriptor.id),
            matches!(
                descriptor.id,
                CommandId::NewFromImage | CommandId::OpenProject
            )
        );
    }
    request_import(&mut session, 10);
    let outcome = session
        .complete_import(ImportToken(10), source(10))
        .unwrap();
    assert_eq!(outcome.snapshot, session.snapshot());
    assert_eq!(outcome.history, session.history());
    assert_eq!(outcome.availability, session.availability());
    assert!(outcome.effects.is_empty());
    for id in [
        CommandId::SetParameter,
        CommandId::BeginGesture,
        CommandId::SelectOutput,
        CommandId::SavePng,
        CommandId::NewFromImage,
    ] {
        assert!(enabled(&session, id));
    }
    for id in [
        CommandId::Undo,
        CommandId::Redo,
        CommandId::CommitGesture,
        CommandId::CancelGesture,
    ] {
        assert!(!enabled(&session, id));
    }
    let address = address(&session, ParameterId::Gamma);
    begin(&mut session, address, GestureId(1));
    assert!(!enabled(&session, CommandId::BeginGesture));
    assert!(enabled(&session, CommandId::CommitGesture));
    assert!(enabled(&session, CommandId::CancelGesture));
    set(&mut session, address, 2.0, Some(GestureId(1)));
    assert!(
        enabled(&session, CommandId::Undo),
        "undo can commit then undo a live edit"
    );
    session.dispatch(CommandRequest::Undo).unwrap();
    assert!(!enabled(&session, CommandId::Undo));
    assert!(enabled(&session, CommandId::Redo));
}

#[test]
fn disabled_commands_without_a_document_are_atomic() {
    let mut session = Session::new();
    let address = ParameterAddress {
        node: NodeId(2),
        parameter: ParameterId::Gamma,
    };
    let requests = [
        CommandRequest::SetParameter {
            address,
            value: ParameterValue::Float(2.0),
            gesture: None,
        },
        CommandRequest::BeginGesture {
            address,
            gesture: GestureId(1),
        },
        CommandRequest::CommitGesture {
            gesture: GestureId(1),
        },
        CommandRequest::CancelGesture {
            gesture: GestureId(1),
        },
        CommandRequest::Undo,
        CommandRequest::Redo,
        CommandRequest::SelectOutput { output: NodeId(4) },
        CommandRequest::SavePng {
            destination: OpaqueLocator("destination".into()),
            map: MapKind::Height,
        },
    ];
    for request in requests {
        let before = observable(&session);
        assert!(session.dispatch(request).is_err());
        assert_eq!(observable(&session), before);
    }
}

#[test]
fn failed_first_import_keeps_the_session_empty_and_never_revives_superseded_tokens() {
    let mut session = Session::new();
    request_import(&mut session, 1);
    request_import(&mut session, 2);
    let before = observable(&session);
    let outcome = session.fail_import(ImportToken(2)).unwrap();
    metadata_only(&session, &outcome);
    assert_eq!(observable(&session), before);
    for token in [ImportToken(1), ImportToken(2), ImportToken(99)] {
        assert_eq!(
            session.complete_import(token, source(1)),
            Err(CommandError::StaleImport(token))
        );
        assert_eq!(
            session.fail_import(token),
            Err(CommandError::StaleImport(token))
        );
        assert_eq!(observable(&session), before);
    }
}

#[test]
fn parameter_validation_is_structured_atomic_and_does_not_consume_pending_import() {
    let mut session = loaded();
    let gamma = address(&session, ParameterId::Gamma);
    redo_branch(&mut session, gamma);
    request_import(&mut session, 80);
    let bad = [
        (
            ParameterAddress {
                node: NodeId(u64::MAX),
                ..gamma
            },
            2.0,
            CoreError::MissingNode(NodeId(u64::MAX)),
        ),
        (
            ParameterAddress {
                parameter: ParameterId::Strength,
                ..gamma
            },
            2.0,
            CoreError::WrongParameter {
                kind: NodeKind::Levels,
                parameter: ParameterId::Strength,
            },
        ),
        (
            gamma,
            f32::NAN,
            CoreError::NonFiniteParameter(ParameterId::Gamma),
        ),
        (
            gamma,
            f32::INFINITY,
            CoreError::NonFiniteParameter(ParameterId::Gamma),
        ),
        (
            gamma,
            f32::NEG_INFINITY,
            CoreError::NonFiniteParameter(ParameterId::Gamma),
        ),
        (
            gamma,
            0.09,
            CoreError::ParameterOutOfRange(ParameterId::Gamma),
        ),
        (
            gamma,
            10.1,
            CoreError::ParameterOutOfRange(ParameterId::Gamma),
        ),
        (
            ParameterAddress {
                parameter: ParameterId::Black,
                ..gamma
            },
            1.0,
            CoreError::InvalidLevelsInterval,
        ),
        (
            ParameterAddress {
                parameter: ParameterId::White,
                ..gamma
            },
            0.0000005,
            CoreError::InvalidLevelsInterval,
        ),
    ];
    for (address, value, error) in bad {
        rejected(
            &mut session,
            CommandRequest::SetParameter {
                address,
                value: ParameterValue::Float(value),
                gesture: None,
            },
            CommandError::Core(error),
        );
    }
    session
        .complete_import(ImportToken(80), source(80))
        .unwrap();
}

#[test]
fn each_parameter_descriptor_routes_to_its_node_and_undoes_independently() {
    for descriptor in PARAMETER_DESCRIPTORS {
        let mut session = loaded();
        let address = address(&session, descriptor.id);
        let before = snapshot(&session);
        assert_eq!(value(&session, address), descriptor.default);
        let changed = match descriptor.id {
            ParameterId::Black => 0.25,
            ParameterId::White => 0.75,
            ParameterId::Gamma | ParameterId::SigmaSourcePx | ParameterId::Strength => 2.0,
            _ => unreachable!("legacy descriptor list contains only scalar controls"),
        };
        let outcome = set(&mut session, address, changed, None);
        assert_eq!(outcome.snapshot, session.snapshot());
        assert!(outcome.effects.is_empty());
        assert_eq!(value(&session, address), changed);
        assert_eq!(outcome.history.undo_entries, 1);
        session.dispatch(CommandRequest::Undo).unwrap();
        assert_eq!(snapshot(&session).graph, before.graph);
        session.dispatch(CommandRequest::Redo).unwrap();
        assert_eq!(value(&session, address), changed);
    }
}

#[test]
fn import_request_emits_only_an_opaque_load_effect_and_failure_preserves_document() {
    let mut session = loaded();
    let gamma = address(&session, ParameterId::Gamma);
    redo_branch(&mut session, gamma);
    session
        .dispatch(CommandRequest::SelectOutput {
            output: output(&session, MapKind::Height),
        })
        .unwrap();
    let before = observable(&session);
    let outcome = request_import(&mut session, 20);
    assert_eq!(
        outcome.effects,
        vec![Effect::LoadImage {
            token: ImportToken(20),
            source: OpaqueLocator("opaque:source/20/ü".into()),
        }]
    );
    assert!(outcome.snapshot.is_none());
    assert_eq!(observable(&session), before);
    let failed = session.fail_import(ImportToken(20)).unwrap();
    metadata_only(&session, &failed);
    assert_eq!(observable(&session), before);
    assert_eq!(
        session.complete_import(ImportToken(20), source(20)),
        Err(CommandError::StaleImport(ImportToken(20)))
    );
    session.dispatch(CommandRequest::Redo).unwrap();
    assert_eq!(value(&session, gamma), 2.0);
}

#[test]
fn only_latest_import_wins_and_slider_generations_do_not_stale_it() {
    let mut session = loaded();
    let gamma = address(&session, ParameterId::Gamma);
    request_import(&mut session, 20);
    request_import(&mut session, 21);
    let before = observable(&session);
    assert_eq!(
        session.complete_import(ImportToken(20), source(20)),
        Err(CommandError::StaleImport(ImportToken(20)))
    );
    assert_eq!(
        session.fail_import(ImportToken(20)),
        Err(CommandError::StaleImport(ImportToken(20)))
    );
    assert_eq!(observable(&session), before);
    set(&mut session, gamma, 2.0, None);
    let previous_generation = snapshot(&session).generation;
    let imported = session
        .complete_import(ImportToken(21), source(21))
        .unwrap();
    assert_eq!(imported.snapshot, session.snapshot());
    assert_eq!(snapshot(&session).generation, previous_generation + 1);
    assert!(
        snapshot(&session)
            .graph
            .nodes()
            .iter()
            .any(|node| node.params == NodeParams::Source(source(21)))
    );
    assert_eq!(session.history(), HistoryState::default());
    let after = observable(&session);
    assert_eq!(
        session.complete_import(ImportToken(21), source(99)),
        Err(CommandError::StaleImport(ImportToken(21)))
    );
    assert_eq!(observable(&session), after);
}

#[test]
fn import_tokens_are_opaque_request_order_wins_even_when_numeric_value_decreases() {
    let mut session = loaded();
    request_import(&mut session, 99);
    request_import(&mut session, 2);
    assert_eq!(
        session.complete_import(ImportToken(99), source(99)),
        Err(CommandError::StaleImport(ImportToken(99)))
    );
    session.complete_import(ImportToken(2), source(2)).unwrap();
}

#[test]
fn invalid_import_completion_is_atomic_and_valid_completion_builds_default_chain() {
    let mut session = loaded();
    let gamma = address(&session, ParameterId::Gamma);
    set(&mut session, gamma, 2.0, None);
    begin(&mut session, gamma, GestureId(3));
    set(&mut session, gamma, 3.0, Some(GestureId(3)));
    request_import(&mut session, 30);
    for extent in [
        Extent {
            width: 0,
            height: 7,
        },
        Extent {
            width: 4097,
            height: 7,
        },
    ] {
        let before = observable(&session);
        let invalid = SourceDescriptor {
            extent,
            ..source(30)
        };
        assert_eq!(
            session.complete_import(ImportToken(30), invalid),
            Err(CommandError::Core(CoreError::InvalidExtent(extent)))
        );
        assert_eq!(observable(&session), before);
    }
    session
        .complete_import(ImportToken(30), source(30))
        .unwrap();
    let graph = snapshot(&session).graph;
    validate(&graph).unwrap();
    let ordered = graph.topological_nodes();
    assert_eq!(ordered.len(), 4);
    assert_eq!(ordered[0].params, NodeParams::Source(source(30)));
    for (index, kind) in [NodeKind::Levels, NodeKind::Blur, NodeKind::NormalFromHeight]
        .into_iter()
        .enumerate()
    {
        assert_eq!(ordered[index + 1].input, Some(ordered[index].id));
        assert_eq!(
            ordered[index + 1].params,
            NodeParams::default_for(kind, source(30))
        );
    }
    assert_eq!(session.history(), HistoryState::default());
}

#[test]
fn export_effect_captures_requested_map_output_and_generation_without_mutation() {
    let mut session = loaded();
    let gamma = address(&session, ParameterId::Gamma);
    redo_branch(&mut session, gamma);
    let selected = output(&session, MapKind::Normal);
    session
        .dispatch(CommandRequest::SelectOutput { output: selected })
        .unwrap();
    for map in [MapKind::Height, MapKind::Normal] {
        let before = observable(&session);
        let destination = OpaqueLocator("opaque:export/../ü:token".into());
        let outcome = session
            .dispatch(CommandRequest::SavePng {
                destination: destination.clone(),
                map,
            })
            .unwrap();
        assert_eq!(
            outcome.effects,
            vec![Effect::SavePng {
                destination,
                map,
                output: output(&session, map),
                generation: snapshot(&session).generation,
            }]
        );
        assert!(outcome.snapshot.is_none());
        assert_eq!(outcome.history, before.history);
        assert_eq!(outcome.availability, before.availability);
        assert_eq!(observable(&session), before);
    }
    begin(&mut session, gamma, GestureId(7));
    set(&mut session, gamma, 3.0, Some(GestureId(7)));
    let before = observable(&session);
    let outcome = session
        .dispatch(CommandRequest::SavePng {
            destination: OpaqueLocator("live-preview".into()),
            map: MapKind::Normal,
        })
        .unwrap();
    assert_eq!(
        outcome.effects,
        vec![Effect::SavePng {
            destination: OpaqueLocator("live-preview".into()),
            map: MapKind::Normal,
            output: output(&session, MapKind::Normal),
            generation: snapshot(&session).generation,
        }]
    );
    assert_eq!(
        observable(&session),
        before,
        "export must not commit an active drag"
    );
}

#[test]
fn invalid_view_selection_is_atomic() {
    let mut session = loaded();
    rejected(
        &mut session,
        CommandRequest::SelectOutput {
            output: NodeId(u64::MAX),
        },
        CommandError::Core(CoreError::MissingNode(NodeId(u64::MAX))),
    );
}

mod history {
    use super::*;

    #[test]
    fn thousand_drag_values_emit_snapshots_but_commit_one_net_parameter_diff() {
        let mut session = loaded();
        let address = address(&session, ParameterId::Gamma);
        let initial = snapshot(&session);
        let gesture = GestureId(42);
        begin(&mut session, address, gesture);
        for step in 1..=1000 {
            let outcome = set(
                &mut session,
                address,
                1.0 + step as f32 / 1000.0,
                Some(gesture),
            );
            assert_eq!(outcome.snapshot, session.snapshot());
            assert_eq!(snapshot(&session).generation, initial.generation + step);
            assert!(outcome.effects.is_empty());
            assert_eq!(outcome.history.undo_entries, 0);
            assert_eq!(outcome.history.active_gesture, Some(gesture));
        }
        let last = snapshot(&session);
        let outcome = session
            .dispatch(CommandRequest::CommitGesture { gesture })
            .unwrap();
        metadata_only(&session, &outcome);
        assert_eq!(snapshot(&session), last);
        assert_eq!(outcome.history.undo_entries, 1);
        assert_eq!(outcome.history.active_gesture, None);
        assert!(outcome.history.byte_count > 0);
        session.dispatch(CommandRequest::Undo).unwrap();
        assert_eq!(snapshot(&session).graph, initial.graph);
        session.dispatch(CommandRequest::Redo).unwrap();
        assert_eq!(snapshot(&session).graph, last.graph);
    }

    #[test]
    fn no_op_direct_empty_and_net_zero_gestures_preserve_redo() {
        let mut session = loaded();
        let address = address(&session, ParameterId::Gamma);
        redo_branch(&mut session, address);
        let before = observable(&session);
        let outcome = set(&mut session, address, 1.0, None);
        metadata_only(&session, &outcome);
        assert_eq!(observable(&session), before);
        for (token, change) in [(GestureId(1), false), (GestureId(2), true)] {
            begin(&mut session, address, token);
            if change {
                set(&mut session, address, 3.0, Some(token));
                set(&mut session, address, 1.0, Some(token));
            }
            let before_commit = snapshot(&session);
            session
                .dispatch(CommandRequest::CommitGesture { gesture: token })
                .unwrap();
            assert_eq!(snapshot(&session), before_commit);
            assert_eq!(session.history(), before.history);
        }
        session.dispatch(CommandRequest::Redo).unwrap();
        assert_eq!(value(&session, address), 2.0);
    }

    #[test]
    fn cancel_restores_original_with_one_generation_and_preserves_redo() {
        let mut session = loaded();
        let address = address(&session, ParameterId::Gamma);
        redo_branch(&mut session, address);
        let history = session.history();
        let gesture = GestureId(3);
        begin(&mut session, address, gesture);
        set(&mut session, address, 3.0, Some(gesture));
        let generation = snapshot(&session).generation;
        let outcome = session
            .dispatch(CommandRequest::CancelGesture { gesture })
            .unwrap();
        assert_eq!(outcome.snapshot, session.snapshot());
        assert_eq!(snapshot(&session).generation, generation + 1);
        assert_eq!(value(&session, address), 1.0);
        assert_eq!(session.history(), history);
        session.dispatch(CommandRequest::Redo).unwrap();
        assert_eq!(value(&session, address), 2.0);
    }

    #[test]
    fn cancel_empty_or_net_zero_gesture_needs_no_new_generation() {
        for change in [false, true] {
            let mut session = loaded();
            let address = address(&session, ParameterId::Gamma);
            redo_branch(&mut session, address);
            let history = session.history();
            begin(&mut session, address, GestureId(9));
            if change {
                set(&mut session, address, 3.0, Some(GestureId(9)));
                set(&mut session, address, 1.0, Some(GestureId(9)));
            }
            let before = snapshot(&session);
            let outcome = session
                .dispatch(CommandRequest::CancelGesture {
                    gesture: GestureId(9),
                })
                .unwrap();
            metadata_only(&session, &outcome);
            assert_eq!(snapshot(&session), before);
            assert_eq!(session.history(), history);
        }
    }

    #[test]
    fn new_commit_clears_redo_only_at_nonempty_commit() {
        for gesture in [None, Some(GestureId(1))] {
            let mut session = loaded();
            let address = address(&session, ParameterId::Gamma);
            redo_branch(&mut session, address);
            if let Some(gesture) = gesture {
                begin(&mut session, address, gesture);
            }
            set(&mut session, address, 3.0, gesture);
            if let Some(gesture) = gesture {
                assert_eq!(session.history().redo_entries, 1, "preview preserves redo");
                session
                    .dispatch(CommandRequest::CommitGesture { gesture })
                    .unwrap();
            }
            assert_eq!(session.history().redo_entries, 0);
            assert_eq!(session.history().undo_entries, 1);
            rejected(
                &mut session,
                CommandRequest::Redo,
                CommandError::Disabled(CommandId::Redo),
            );
        }
    }

    #[test]
    fn focus_loss_uses_commit_and_discrete_numeric_edits_remain_separate() {
        let mut session = loaded();
        let address = address(&session, ParameterId::Gamma);
        begin(&mut session, address, GestureId(5));
        set(&mut session, address, 2.0, Some(GestureId(5)));
        // The UI dispatches this same command on pointer release or focus loss.
        session
            .dispatch(CommandRequest::CommitGesture {
                gesture: GestureId(5),
            })
            .unwrap();
        set(&mut session, address, 3.0, None);
        set(&mut session, address, 4.0, None);
        assert_eq!(session.history().undo_entries, 3);
        for expected in [3.0, 2.0, 1.0] {
            session.dispatch(CommandRequest::Undo).unwrap();
            assert_eq!(value(&session, address), expected);
        }
        for expected in [2.0, 3.0, 4.0] {
            session.dispatch(CommandRequest::Redo).unwrap();
            assert_eq!(value(&session, address), expected);
        }
    }

    #[test]
    fn undo_during_a_drag_commits_then_undoes_the_drag() {
        let mut session = loaded();
        let address = address(&session, ParameterId::Gamma);
        set(&mut session, address, 2.0, None);
        begin(&mut session, address, GestureId(7));
        set(&mut session, address, 3.0, Some(GestureId(7)));
        session.dispatch(CommandRequest::Undo).unwrap();
        assert_eq!(value(&session, address), 2.0);
        assert_eq!(session.history().active_gesture, None);
        assert_eq!(session.history().undo_entries, 1);
        assert_eq!(session.history().redo_entries, 1);
        session.dispatch(CommandRequest::Undo).unwrap();
        assert_eq!(value(&session, address), 1.0);
        for expected in [2.0, 3.0] {
            session.dispatch(CommandRequest::Redo).unwrap();
            assert_eq!(value(&session, address), expected);
        }
    }

    #[test]
    fn redo_during_net_zero_drag_commits_it_then_replays_existing_redo() {
        let mut session = loaded();
        let address = address(&session, ParameterId::Gamma);
        redo_branch(&mut session, address);
        begin(&mut session, address, GestureId(8));
        set(&mut session, address, 3.0, Some(GestureId(8)));
        set(&mut session, address, 1.0, Some(GestureId(8)));
        session.dispatch(CommandRequest::Redo).unwrap();
        assert_eq!(value(&session, address), 2.0);
        assert_eq!(session.history().active_gesture, None);
        assert_eq!(session.history().undo_entries, 1);
        assert_eq!(session.history().redo_entries, 0);
    }

    #[test]
    fn redo_during_nonempty_drag_is_atomic_when_commit_would_clear_redo() {
        let mut session = loaded();
        let address = address(&session, ParameterId::Gamma);
        redo_branch(&mut session, address);
        begin(&mut session, address, GestureId(8));
        set(&mut session, address, 3.0, Some(GestureId(8)));
        // Commit would clear redo, so the whole redo request fails and rolls back.
        rejected(
            &mut session,
            CommandRequest::Redo,
            CommandError::Disabled(CommandId::Redo),
        );
        assert_eq!(session.history().active_gesture, Some(GestureId(8)));
        session
            .dispatch(CommandRequest::CancelGesture {
                gesture: GestureId(8),
            })
            .unwrap();
        session.dispatch(CommandRequest::Redo).unwrap();
        assert_eq!(value(&session, address), 2.0);
    }

    #[test]
    fn nested_mismatched_and_unscoped_gesture_requests_are_atomic() {
        let mut session = loaded();
        let address = address(&session, ParameterId::Gamma);
        let other = super::address(&session, ParameterId::Strength);
        begin(&mut session, address, GestureId(4));
        set(&mut session, address, 2.0, Some(GestureId(4)));
        for gesture in [GestureId(4), GestureId(5)] {
            rejected(
                &mut session,
                CommandRequest::BeginGesture { address, gesture },
                CommandError::GestureAlreadyActive,
            );
        }
        for request in [
            CommandRequest::CommitGesture {
                gesture: GestureId(5),
            },
            CommandRequest::CancelGesture {
                gesture: GestureId(5),
            },
            CommandRequest::SetParameter {
                address,
                value: ParameterValue::Float(3.0),
                gesture: Some(GestureId(5)),
            },
        ] {
            rejected(
                &mut session,
                request,
                CommandError::StaleGesture(GestureId(5)),
            );
        }
        for request in [
            CommandRequest::SetParameter {
                address: other,
                value: ParameterValue::Float(3.0),
                gesture: Some(GestureId(4)),
            },
            CommandRequest::SetParameter {
                address,
                value: ParameterValue::Float(3.0),
                gesture: None,
            },
            CommandRequest::SetParameter {
                address,
                value: ParameterValue::Float(f32::NAN),
                gesture: Some(GestureId(4)),
            },
        ] {
            let before = observable(&session);
            assert!(session.dispatch(request).is_err());
            assert_eq!(observable(&session), before);
        }
        session
            .dispatch(CommandRequest::CancelGesture {
                gesture: GestureId(4),
            })
            .unwrap();
        assert_eq!(value(&session, address), 1.0);
        for request in [
            CommandRequest::CommitGesture {
                gesture: GestureId(4),
            },
            CommandRequest::CancelGesture {
                gesture: GestureId(4),
            },
            CommandRequest::SetParameter {
                address,
                value: ParameterValue::Float(3.0),
                gesture: Some(GestureId(4)),
            },
        ] {
            rejected(
                &mut session,
                request,
                CommandError::StaleGesture(GestureId(4)),
            );
        }
    }

    #[test]
    fn invalid_begin_does_not_open_a_transaction() {
        let mut session = loaded();
        let address = ParameterAddress {
            node: NodeId(u64::MAX),
            parameter: ParameterId::Gamma,
        };
        rejected(
            &mut session,
            CommandRequest::BeginGesture {
                address,
                gesture: GestureId(1),
            },
            CommandError::Core(CoreError::MissingNode(NodeId(u64::MAX))),
        );
        let address = super::address(&session, ParameterId::Gamma);
        begin(&mut session, address, GestureId(1));
    }

    #[test]
    fn undo_of_empty_gesture_without_history_is_an_atomic_disabled_request() {
        let mut session = loaded();
        let address = address(&session, ParameterId::Gamma);
        begin(&mut session, address, GestureId(1));
        rejected(
            &mut session,
            CommandRequest::Undo,
            CommandError::Disabled(CommandId::Undo),
        );
        assert_eq!(session.history().active_gesture, Some(GestureId(1)));
        session
            .dispatch(CommandRequest::CommitGesture {
                gesture: GestureId(1),
            })
            .unwrap();
        assert_eq!(session.history(), HistoryState::default());
    }

    #[test]
    fn diagnostic_history_grows_without_a_small_count_cap_and_contains_no_image_storage() {
        let mut counts = Vec::new();
        for dimension in [1, 4096] {
            let mut session = Session::new();
            request_import(&mut session, 1);
            session
                .complete_import(
                    ImportToken(1),
                    SourceDescriptor {
                        extent: Extent {
                            width: dimension,
                            height: dimension,
                        },
                        ..source(1)
                    },
                )
                .unwrap();
            let address = address(&session, ParameterId::Gamma);
            for index in 0..4097 {
                set(
                    &mut session,
                    address,
                    if index % 2 == 0 { 2.0 } else { 3.0 },
                    None,
                );
            }
            let history = session.history();
            assert_eq!(history.undo_entries, 4097);
            assert!(
                history.byte_count >= history.undo_entries * 8,
                "at least the two f32 parameter values per entry"
            );
            let bytes = history.byte_count;
            for _ in 0..4097 {
                session.dispatch(CommandRequest::Undo).unwrap();
            }
            assert_eq!(value(&session, address), 1.0);
            assert_eq!(session.history().redo_entries, 4097);
            assert_eq!(
                session.history().byte_count,
                bytes,
                "undo moves retained diffs"
            );
            counts.push(bytes);
            request_import(&mut session, 2);
            session.complete_import(ImportToken(2), source(2)).unwrap();
            assert_eq!(session.history(), HistoryState::default());
        }
        assert_eq!(
            counts[0], counts[1],
            "history storage is independent of image dimensions"
        );
    }
}

mod generation {
    use super::*;

    fn keys(session: &Session) -> BTreeMap<NodeId, CacheKey> {
        let graph = snapshot(session).graph;
        let extent = graph
            .nodes()
            .iter()
            .find_map(|node| match node.params {
                NodeParams::Source(ref source) => Some(source.extent),
                _ => None,
            })
            .unwrap();
        let mut keys = BTreeMap::new();
        for node in graph.topological_nodes() {
            let input: Vec<_> = node.input.map(|id| keys[&id]).into_iter().collect();
            keys.insert(
                node.id,
                node_cache_key(node, &input, extent, extent).unwrap(),
            );
        }
        keys
    }

    #[test]
    fn monotonic_across_edits_undo_redo_cancel_and_identical_source_reimport() {
        let mut session = loaded();
        let address = address(&session, ParameterId::Gamma);
        let initial = snapshot(&session).generation;
        set(&mut session, address, 2.0, None);
        assert_eq!(snapshot(&session).generation, initial + 1);
        session.dispatch(CommandRequest::Undo).unwrap();
        assert_eq!(snapshot(&session).generation, initial + 2);
        session.dispatch(CommandRequest::Redo).unwrap();
        assert_eq!(snapshot(&session).generation, initial + 3);
        begin(&mut session, address, GestureId(1));
        set(&mut session, address, 3.0, Some(GestureId(1)));
        assert_eq!(snapshot(&session).generation, initial + 4);
        session
            .dispatch(CommandRequest::CancelGesture {
                gesture: GestureId(1),
            })
            .unwrap();
        assert_eq!(snapshot(&session).generation, initial + 5);
        request_import(&mut session, 2);
        session.complete_import(ImportToken(2), source(1)).unwrap();
        assert_eq!(snapshot(&session).generation, initial + 6);
        assert_eq!(session.history(), HistoryState::default());
    }

    #[test]
    fn no_ops_and_view_selection_preserve_generation_history_and_cache_identity() {
        let mut session = loaded();
        let address = address(&session, ParameterId::SigmaSourcePx);
        let initial = snapshot(&session);
        let initial_keys = keys(&session);
        let outcome = set(&mut session, address, -0.0, None);
        metadata_only(&session, &outcome);
        let history = session.history();
        for map in [MapKind::Height, MapKind::Normal] {
            let output = output(&session, map);
            let previous_output = snapshot(&session).graph.selected_output();
            let outcome = session
                .dispatch(CommandRequest::SelectOutput { output })
                .unwrap();
            assert_eq!(outcome.snapshot.is_some(), output != previous_output);
            if outcome.snapshot.is_some() {
                assert_eq!(outcome.snapshot, session.snapshot());
            }
            assert_eq!(snapshot(&session).graph.selected_output(), output);
            assert_eq!(snapshot(&session).generation, initial.generation);
            assert_eq!(session.history(), history);
            assert_eq!(keys(&session), initial_keys);
            let repeated = session
                .dispatch(CommandRequest::SelectOutput { output })
                .unwrap();
            metadata_only(&session, &repeated);
        }
    }

    #[test]
    fn downstream_only_invalidation_on_branching_graph_and_undo_reuses_content_keys() {
        let mut graph = snapshot(&loaded()).graph;
        let blur = graph
            .nodes()
            .iter()
            .find(|node| node.params.kind() == NodeKind::Blur)
            .unwrap()
            .id;
        let normal = graph
            .nodes()
            .iter()
            .find(|node| node.params.kind() == NodeKind::NormalFromHeight)
            .unwrap()
            .id;
        let branch = NodeId(100);
        let mut nodes = graph.nodes().to_vec();
        let source = nodes
            .iter()
            .find(|node| node.params.kind() == NodeKind::Source)
            .unwrap()
            .id;
        nodes.push(Node {
            id: branch,
            input: Some(source),
            params: NodeParams::Blur {
                sigma_source_px: 1.0,
            },
        });
        graph = Graph::new(nodes, normal).unwrap();
        let mut session = Session::from_snapshot(EvaluationSnapshot {
            generation: 40,
            graph,
        });
        let before = keys(&session);
        let address = ParameterAddress {
            node: blur,
            parameter: ParameterId::SigmaSourcePx,
        };
        set(&mut session, address, 2.0, None);
        let after = keys(&session);
        for (&id, &key) in &before {
            if id == blur || id == normal {
                assert_ne!(after[&id], key);
            } else {
                assert_eq!(after[&id], key, "unrelated ancestor/branch {id:?}");
            }
        }
        assert_eq!(snapshot(&session).generation, 41);
        session.dispatch(CommandRequest::Undo).unwrap();
        assert_eq!(keys(&session), before);
        assert_eq!(snapshot(&session).generation, 42);
        session.dispatch(CommandRequest::Redo).unwrap();
        assert_eq!(keys(&session), after);
        assert_eq!(snapshot(&session).generation, 43);
    }

    fn near_overflow() -> Session {
        let graph = snapshot(&loaded()).graph;
        Session::from_snapshot(EvaluationSnapshot {
            generation: u64::MAX - 1,
            graph,
        })
    }

    #[test]
    fn overflow_parameter_undo_and_import_errors_do_not_wrap_or_partially_mutate() {
        let mut session = near_overflow();
        let address = address(&session, ParameterId::Gamma);
        set(&mut session, address, 2.0, None);
        assert_eq!(snapshot(&session).generation, u64::MAX);
        rejected(
            &mut session,
            CommandRequest::SetParameter {
                address,
                value: ParameterValue::Float(3.0),
                gesture: None,
            },
            CommandError::GenerationOverflow,
        );
        rejected(
            &mut session,
            CommandRequest::Undo,
            CommandError::GenerationOverflow,
        );
        request_import(&mut session, 9);
        let before = observable(&session);
        assert_eq!(
            session.complete_import(ImportToken(9), source(9)),
            Err(CommandError::GenerationOverflow)
        );
        assert_eq!(observable(&session), before);
        assert_eq!(
            session.complete_import(ImportToken(9), source(9)),
            Err(CommandError::GenerationOverflow),
            "overflow must not consume the pending token"
        );
        let no_op = set(&mut session, address, 2.0, None);
        metadata_only(&session, &no_op);
        let normal = output(&session, MapKind::Normal);
        session
            .dispatch(CommandRequest::SelectOutput { output: normal })
            .unwrap();
        assert_eq!(snapshot(&session).generation, u64::MAX);
    }

    #[test]
    fn overflow_redo_and_cancel_preserve_history_and_live_transaction() {
        let mut seed = loaded();
        let address = address(&seed, ParameterId::Gamma);
        set(&mut seed, address, 2.0, None);
        // Leave enough space to create redo by undoing once.
        let mut redo = Session::from_snapshot(EvaluationSnapshot {
            generation: u64::MAX - 2,
            graph: snapshot(&seed).graph,
        });
        set(&mut redo, address, 3.0, None);
        redo.dispatch(CommandRequest::Undo).unwrap();
        assert_eq!(snapshot(&redo).generation, u64::MAX);
        rejected(
            &mut redo,
            CommandRequest::Redo,
            CommandError::GenerationOverflow,
        );
        let mut cancel = near_overflow();
        begin(&mut cancel, address, GestureId(1));
        set(&mut cancel, address, 3.0, Some(GestureId(1)));
        rejected(
            &mut cancel,
            CommandRequest::CancelGesture {
                gesture: GestureId(1),
            },
            CommandError::GenerationOverflow,
        );
        rejected(
            &mut cancel,
            CommandRequest::Undo,
            CommandError::GenerationOverflow,
        );
        assert_eq!(cancel.history().active_gesture, Some(GestureId(1)));
        let outcome = cancel
            .dispatch(CommandRequest::CommitGesture {
                gesture: GestureId(1),
            })
            .unwrap();
        metadata_only(&cancel, &outcome);
        assert_eq!(snapshot(&cancel).generation, u64::MAX);
        assert_eq!(cancel.history().undo_entries, 1);
    }
}
