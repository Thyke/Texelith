//! Tests-first T-018 contract. Proposed UI seam: `CommandAdapter::default()`;
//! `handle(UiAction) -> Vec<CommandRequest>` translates input without touching
//! a graph; `palette(&Session)` returns registry entries with `id` and `enabled`.
//! `Invoke` takes an origin and a typed command ID; parameter-requiring palette
//! actions return no fabricated request and focus their control in the UI.
//! The application dispatches every returned request through Session in order.

use texelith_core::{
    CommandId, CommandRequest, Extent, ImportToken, InputInterpretation, NodeId, OpaqueLocator,
    ParameterAddress, ParameterId, ParameterValue, Session, SourceDescriptor, command_descriptors,
};
use texelith_ui_egui::command_adapter::{ActionOrigin, CommandAdapter, UiAction};

fn loaded() -> Session {
    let mut session = Session::new();
    session
        .dispatch(CommandRequest::NewFromImage {
            token: ImportToken(1),
            source: OpaqueLocator("fixture".into()),
        })
        .unwrap();
    session
        .complete_import(
            ImportToken(1),
            SourceDescriptor {
                content_hash: [1; 32],
                extent: Extent {
                    width: 4,
                    height: 4,
                },
                interpretation: InputInterpretation::LinearHeight,
            },
        )
        .unwrap();
    session
}

fn address() -> ParameterAddress {
    ParameterAddress {
        node: NodeId(2),
        parameter: ParameterId::Gamma,
    }
}
fn value(session: &Session) -> f32 {
    session
        .snapshot()
        .unwrap()
        .graph
        .node(address().node)
        .unwrap()
        .params
        .parameter(address().parameter)
        .unwrap()
}
fn send(
    adapter: &mut CommandAdapter,
    session: &mut Session,
    action: UiAction,
) -> Vec<CommandRequest> {
    let requests = adapter.handle(action);
    for request in &requests {
        session.dispatch(request.clone()).unwrap();
    }
    requests
}
fn change(value: f32) -> UiAction {
    UiAction::Change {
        address: address(),
        value: ParameterValue::Float(value),
    }
}

#[test]
fn command_adapter_palette_gesture_end_uses_active_token() {
    for (id, cancel) in [
        (CommandId::CommitGesture, false),
        (CommandId::CancelGesture, true),
    ] {
        let (mut adapter, mut session) = (CommandAdapter::default(), loaded());
        let begin = send(
            &mut adapter,
            &mut session,
            UiAction::DragBegin { address: address() },
        );
        let CommandRequest::BeginGesture { gesture, .. } = begin[0] else {
            panic!("expected gesture")
        };
        send(&mut adapter, &mut session, change(2.0));
        let end = send(
            &mut adapter,
            &mut session,
            UiAction::Invoke {
                origin: ActionOrigin::Palette,
                id,
            },
        );
        assert_eq!(
            end,
            vec![if cancel {
                CommandRequest::CancelGesture { gesture }
            } else {
                CommandRequest::CommitGesture { gesture }
            }]
        );
        assert_eq!(value(&session), if cancel { 1.0 } else { 2.0 });
        assert_eq!(session.history().undo_entries, usize::from(!cancel));
    }
}

#[test]
fn command_adapter_drag_coalesces_and_dispatches_typed_requests() {
    let (mut adapter, mut session) = (CommandAdapter::default(), loaded());
    let begin = send(
        &mut adapter,
        &mut session,
        UiAction::DragBegin { address: address() },
    );
    let gesture = match begin.as_slice() {
        [
            CommandRequest::BeginGesture {
                gesture,
                address: a,
            },
        ] if *a == address() => *gesture,
        other => panic!("unexpected begin: {other:?}"),
    };
    for v in [2.0, 3.0] {
        assert_eq!(
            send(&mut adapter, &mut session, change(v)),
            vec![CommandRequest::SetParameter {
                address: address(),
                value: ParameterValue::Float(v),
                gesture: Some(gesture),
            }]
        );
    }
    assert_eq!(
        send(&mut adapter, &mut session, UiAction::DragRelease),
        vec![CommandRequest::CommitGesture { gesture }]
    );
    assert_eq!(value(&session), 3.0);
    assert_eq!(session.history().undo_entries, 1);
    send(&mut adapter, &mut session, UiAction::UndoShortcut);
    assert_eq!(value(&session), 1.0);
    send(&mut adapter, &mut session, UiAction::RedoShortcut);
    assert_eq!(value(&session), 3.0);
}

#[test]
fn command_adapter_keyboard_edits_are_individual_and_shortcuts_are_typed() {
    let (mut adapter, mut session) = (CommandAdapter::default(), loaded());
    for v in [2.0, 3.0] {
        assert_eq!(
            send(&mut adapter, &mut session, change(v)),
            vec![CommandRequest::SetParameter {
                address: address(),
                value: ParameterValue::Float(v),
                gesture: None,
            }]
        );
    }
    assert_eq!(session.history().undo_entries, 2);
    assert_eq!(
        send(&mut adapter, &mut session, UiAction::UndoShortcut),
        vec![CommandRequest::Undo]
    );
    assert_eq!(value(&session), 2.0);
    assert_eq!(
        send(&mut adapter, &mut session, UiAction::RedoShortcut),
        vec![CommandRequest::Redo]
    );
    assert_eq!(value(&session), 3.0);
}

#[test]
fn command_adapter_escape_cancels_and_focus_loss_commits() {
    let (mut adapter, mut session) = (CommandAdapter::default(), loaded());
    let begin = send(
        &mut adapter,
        &mut session,
        UiAction::DragBegin { address: address() },
    );
    let gesture = match begin[0] {
        CommandRequest::BeginGesture { gesture, .. } => gesture,
        _ => panic!(),
    };
    send(&mut adapter, &mut session, change(2.0));
    assert_eq!(
        send(&mut adapter, &mut session, UiAction::Escape),
        vec![CommandRequest::CancelGesture { gesture }]
    );
    assert_eq!(value(&session), 1.0);
    assert_eq!(session.history().undo_entries, 0);
    assert!(send(&mut adapter, &mut session, UiAction::DragRelease).is_empty());
    let begin = send(
        &mut adapter,
        &mut session,
        UiAction::DragBegin { address: address() },
    );
    let next = match begin[0] {
        CommandRequest::BeginGesture { gesture, .. } => gesture,
        _ => panic!(),
    };
    assert_ne!(gesture, next);
    send(&mut adapter, &mut session, change(3.0));
    assert_eq!(
        send(&mut adapter, &mut session, UiAction::FocusLost),
        vec![CommandRequest::CommitGesture { gesture: next }]
    );
    assert_eq!(session.history().undo_entries, 1);
    assert_eq!(value(&session), 3.0);
}

#[test]
fn command_adapter_palette_registry_availability_and_menu_equivalence() {
    let mut adapter = CommandAdapter::default();
    for session in [Session::new(), loaded()] {
        let entries = adapter.palette(&session);
        let availability = session.availability();
        assert_eq!(entries.len(), command_descriptors().len());
        for (entry, (descriptor, available)) in entries
            .iter()
            .zip(command_descriptors().iter().zip(&availability))
        {
            assert_eq!(entry.id, descriptor.id);
            assert_eq!(entry.enabled, available.enabled);
            assert_eq!(entry.id, available.id);
        }
    }
    for id in [CommandId::Undo, CommandId::Redo] {
        let menu = adapter.handle(UiAction::Invoke {
            origin: ActionOrigin::Menu,
            id,
        });
        let palette = adapter.handle(UiAction::Invoke {
            origin: ActionOrigin::Palette,
            id,
        });
        assert_eq!(menu, palette);
        assert_eq!(
            menu,
            vec![if id == CommandId::Undo {
                CommandRequest::Undo
            } else {
                CommandRequest::Redo
            }]
        );
    }
    for id in [
        CommandId::NewFromImage,
        CommandId::SetParameter,
        CommandId::SelectOutput,
        CommandId::SavePng,
    ] {
        assert!(
            adapter
                .handle(UiAction::Invoke {
                    origin: ActionOrigin::Palette,
                    id
                })
                .is_empty(),
            "{id:?} requires a payload"
        );
    }
}
