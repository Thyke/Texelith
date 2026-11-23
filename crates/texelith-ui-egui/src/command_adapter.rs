use texelith_core::{
    CommandId, CommandRequest, GestureId, ParameterAddress, ParameterValue, Session, StrokeId,
    command_descriptors,
};

#[derive(Clone, Copy, Debug)]
pub enum ActionOrigin {
    Menu,
    Palette,
}
#[derive(Clone, Debug)]
pub enum UiAction {
    DragBegin {
        address: ParameterAddress,
    },
    Change {
        address: ParameterAddress,
        value: ParameterValue,
    },
    DragRelease,
    Escape,
    FocusLost,
    UndoShortcut,
    RedoShortcut,
    Invoke {
        origin: ActionOrigin,
        id: CommandId,
    },
    Request(CommandRequest),
}
#[derive(Clone, Debug)]
pub struct PaletteEntry {
    pub id: CommandId,
    pub label: String,
    pub enabled: bool,
}
#[derive(Clone, Default)]
pub struct CommandAdapter {
    active: Option<(GestureId, ParameterAddress)>,
    stroke: Option<StrokeId>,
    next: u64,
}
impl CommandAdapter {
    pub(crate) fn active_address(&self) -> Option<ParameterAddress> {
        self.active.map(|(_, address)| address)
    }

    pub fn handle(&mut self, action: UiAction) -> Vec<CommandRequest> {
        match action {
            UiAction::DragBegin { address } if self.active.is_none() => {
                self.next = self.next.checked_add(1).expect("gesture IDs exhausted");
                let gesture = GestureId(self.next);
                self.active = Some((gesture, address));
                vec![CommandRequest::BeginGesture { gesture, address }]
            }
            UiAction::Change { address, value } => vec![CommandRequest::SetParameter {
                address,
                value,
                gesture: self.active.filter(|(_, a)| *a == address).map(|(g, _)| g),
            }],
            UiAction::DragRelease
            | UiAction::Invoke {
                id: CommandId::CommitGesture,
                ..
            } => self
                .active
                .take()
                .map(|(gesture, _)| CommandRequest::CommitGesture { gesture })
                .into_iter()
                .collect(),
            UiAction::FocusLost if self.stroke.is_some() => vec![CommandRequest::CommitStroke {
                stroke_id: self.stroke.take().unwrap(),
            }],
            UiAction::Escape if self.stroke.is_some() => vec![CommandRequest::CancelStroke {
                stroke_id: self.stroke.take().unwrap(),
            }],
            UiAction::FocusLost => self
                .active
                .take()
                .map(|(gesture, _)| CommandRequest::CommitGesture { gesture })
                .into_iter()
                .collect(),
            UiAction::Escape
            | UiAction::Invoke {
                id: CommandId::CancelGesture,
                ..
            } => self
                .active
                .take()
                .map(|(gesture, _)| CommandRequest::CancelGesture { gesture })
                .into_iter()
                .collect(),
            UiAction::UndoShortcut
            | UiAction::Invoke {
                id: CommandId::Undo,
                ..
            } => {
                self.active = None;
                vec![CommandRequest::Undo]
            }
            UiAction::RedoShortcut
            | UiAction::Invoke {
                id: CommandId::Redo,
                ..
            } => {
                self.active = None;
                vec![CommandRequest::Redo]
            }
            UiAction::Invoke {
                id: CommandId::RequestExport,
                ..
            } => vec![CommandRequest::RequestExport {}],
            UiAction::Request(request) => vec![request],
            UiAction::Invoke {
                id: CommandId::BatchStart,
                ..
            } => vec![CommandRequest::BatchStart {}],
            UiAction::Invoke {
                id: CommandId::BatchCancel,
                ..
            } => vec![CommandRequest::BatchCancel {}],
            UiAction::Invoke {
                id: CommandId::BatchRetryFailed,
                ..
            } => vec![CommandRequest::BatchRetryFailed {}],
            _ => vec![],
        }
    }
    pub fn sync(&mut self, session: &Session) {
        self.stroke = session.history().active_stroke;
        if session.history().active_gesture != self.active.map(|(id, _)| id) {
            self.active = None;
        }
    }
    pub fn palette(&self, session: &Session) -> Vec<PaletteEntry> {
        command_descriptors()
            .iter()
            .zip(session.availability())
            .map(|(d, a)| PaletteEntry {
                id: d.id,
                label: crate::input_locale::labels::command(d.id),
                enabled: a.enabled,
            })
            .collect()
    }
}
