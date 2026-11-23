use texelith_core::{
    CommandRequest, Effect, Extent, ImportToken, InputInterpretation, NodeId, ParameterAddress,
    ParameterId, ParameterValue, SourceDescriptor,
};
use texelith_gpu::Quality;
use texelith_ui_egui::{command_adapter::UiAction, controls::Controls, routing::RenderRouting};

fn source() -> SourceDescriptor {
    SourceDescriptor {
        content_hash: [7; 32],
        extent: Extent {
            width: 4096,
            height: 2048,
        },
        interpretation: InputInterpretation::LinearHeight,
    }
}
fn import_token(controls: &mut Controls) -> ImportToken {
    let effects = controls.take_effects();
    match &effects[0] {
        Effect::LoadImage { token, source } => {
            assert!(source.0.starts_with("native-"));
            assert!(
                !source.0.contains("private"),
                "native paths must stay outside core"
            );
            *token
        }
        other => panic!("unexpected {other:?}"),
    }
}
fn loaded() -> Controls {
    let mut controls = Controls::new(false);
    controls.open_image("private/input.png".into());
    let token = import_token(&mut controls);
    controls.complete_import(token, source()).unwrap();
    controls
}
fn address() -> ParameterAddress {
    ParameterAddress {
        node: NodeId(4),
        parameter: ParameterId::Strength,
    }
}
fn change(controls: &mut Controls, value: f32) {
    controls.action(UiAction::Change {
        address: address(),
        value: ParameterValue::Float(value),
    });
}

#[test]
fn release_and_output_selection_have_distinct_request_ids_at_the_same_generation() {
    let mut controls = loaded();
    let mut routing = RenderRouting::default();
    controls.action(UiAction::DragBegin { address: address() });
    change(&mut controls, 3.0);
    let snapshot = controls.session().snapshot().unwrap();
    let proxy = routing
        .issue(snapshot.clone(), NodeId(4), Quality::Proxy)
        .unwrap();
    controls.action(UiAction::DragRelease);
    let released = controls.session().snapshot().unwrap();
    assert_eq!(released.generation, snapshot.generation);
    let full = routing.issue(released, NodeId(4), Quality::Full).unwrap();
    assert!(full.request_id > proxy.request_id);
    assert_eq!(
        proxy.extent,
        Extent {
            width: 1024,
            height: 512
        }
    );
    assert_eq!(full.extent, source().extent);
    assert!(!routing.matches(
        proxy.request_id,
        proxy.snapshot.generation,
        proxy.output,
        proxy.extent,
        proxy.quality
    ));
    controls.action(UiAction::Request(CommandRequest::SelectOutput {
        output: NodeId(3),
    }));
    let height = routing
        .issue(
            controls.session().snapshot().unwrap(),
            NodeId(3),
            Quality::Full,
        )
        .unwrap();
    assert_eq!(height.snapshot.generation, full.snapshot.generation);
    assert!(height.request_id > full.request_id);
    assert!(!routing.matches(
        full.request_id,
        full.snapshot.generation,
        full.output,
        full.extent,
        full.quality
    ));
    assert!(routing.matches(
        height.request_id,
        height.snapshot.generation,
        height.output,
        height.extent,
        height.quality
    ));
    assert!(!routing.matches(
        height.request_id,
        height.snapshot.generation + 1,
        height.output,
        height.extent,
        height.quality
    ));
    assert!(!routing.matches(
        height.request_id,
        height.snapshot.generation,
        NodeId(4),
        height.extent,
        height.quality
    ));
    assert!(!routing.matches(
        height.request_id,
        height.snapshot.generation,
        height.output,
        proxy.extent,
        height.quality
    ));
    assert!(!routing.matches(
        height.request_id,
        height.snapshot.generation,
        height.output,
        height.extent,
        Quality::Proxy
    ));
}

#[test]
fn failed_and_superseded_imports_preserve_the_old_document_and_history() {
    let mut controls = loaded();
    change(&mut controls, 4.0);
    let before = controls.session().snapshot().unwrap();
    let history = controls.session().history();
    controls.open_image("private/old.png".into());
    let old = import_token(&mut controls);
    controls.open_image("private/new.png".into());
    let new = import_token(&mut controls);
    assert_ne!(old, new);
    assert!(controls.complete_import(old, source()).is_err());
    controls.fail_import(new);
    assert_eq!(controls.session().snapshot().unwrap(), before);
    assert_eq!(controls.session().history(), history);
    controls.open_image("private/replacement.png".into());
    let replacement = import_token(&mut controls);
    controls.complete_import(replacement, source()).unwrap();
    assert!(controls.session().snapshot().unwrap().generation > before.generation);
    assert_eq!(controls.session().history().undo_entries, 0);
}

#[test]
fn many_changes_commit_once_and_save_uses_an_opaque_locator() {
    let mut controls = loaded();
    controls.action(UiAction::DragBegin { address: address() });
    for step in 0..100 {
        change(&mut controls, step as f32 / 10.0);
    }
    controls.action(UiAction::DragRelease);
    assert_eq!(controls.session().history().undo_entries, 1);
    controls.action(UiAction::UndoShortcut);
    assert_eq!(
        controls
            .session()
            .snapshot()
            .unwrap()
            .graph
            .node(NodeId(4))
            .unwrap()
            .params
            .parameter(ParameterId::Strength),
        Ok(1.0)
    );
    controls.action(UiAction::RedoShortcut);
    controls.save_png("private/normal.png".into(), texelith_core::MapKind::Normal);
    match &controls.take_effects()[0] {
        Effect::SavePng {
            destination,
            output,
            generation,
            ..
        } => {
            assert!(destination.0.starts_with("native-"));
            assert_eq!(*output, NodeId(4));
            assert_eq!(
                *generation,
                controls.session().snapshot().unwrap().generation
            );
        }
        other => panic!("unexpected {other:?}"),
    }
}

#[derive(Debug)]
struct Dropped(std::path::PathBuf);
impl egui::DroppedFile for Dropped {
    fn path(&self) -> &std::path::Path {
        &self.0
    }
    fn bytes(&self) -> Result<Vec<u8>, String> {
        panic!("UI must never read dropped file bytes")
    }
}

#[test]
fn egui_drop_routes_replacement_without_reading_or_mutating_the_document() {
    let mut controls = loaded();
    change(&mut controls, 2.0);
    let before = controls.session().snapshot().unwrap();
    let history = controls.session().history();
    let ctx = egui::Context::default();
    let input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(900.0, 700.0),
        )),
        dropped_files: vec![std::sync::Arc::new(Dropped(
            "private/replacement.png".into(),
        ))],
        ..Default::default()
    };
    let mut output = ctx.run_ui(input, |ui| controls.show(ui));
    output.textures_delta.clear();
    let effects = controls.take_effects();
    assert_eq!(effects.len(), 1);
    assert!(
        matches!(&effects[0], Effect::LoadImage { source, .. } if source.0.starts_with("native-"))
    );
    assert_eq!(controls.session().snapshot().unwrap(), before);
    assert_eq!(controls.session().history(), history);
}
