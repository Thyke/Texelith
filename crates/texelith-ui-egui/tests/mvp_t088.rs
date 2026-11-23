use std::{
    path::PathBuf,
    time::{Duration, Instant},
};
use texelith_core::*;
use texelith_ui_egui::{command_adapter::UiAction, controls::Controls, project_batch::AssetChoice};

fn fixture(name: &str) -> (Controls, PathBuf) {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("evidence/T-088/tests")
        .join(format!("{name}-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let root = std::fs::canonicalize(root).unwrap();
    let source = root.join("source.png");
    texelith_io::write_normal_png(&source, 64, 64, &vec![128; 64 * 64 * 3], false).unwrap();
    let image = texelith_io::decode_path(&source, texelith_io::ColorIntent::PhotoSrgb).unwrap();
    let mut c = Controls::new_material();
    c.open_image(source);
    let Effect::LoadImage { token, .. } = c.take_effects().remove(0) else {
        panic!()
    };
    c.complete_import(
        token,
        SourceDescriptor::from_rgba8(
            &image.rgba8,
            Extent::new(64, 64).unwrap(),
            InputInterpretation::SrgbPhoto,
        )
        .unwrap(),
    )
    .unwrap();
    (c, root)
}
fn pump(c: &mut Controls) {
    let start = Instant::now();
    while c.project_busy() {
        c.poll_project_io();
        while let Some(upload) = c.take_project_upload() {
            c.ack_project_upload(upload.descriptor.content_hash);
        }
        assert!(
            start.elapsed() < Duration::from_secs(10),
            "project operation timed out"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
}
#[test]
fn save_reopen_exact_recipe_and_failed_open_preserves_document() {
    let (mut c, root) = fixture("roundtrip");
    c.action(UiAction::Request(CommandRequest::SetLocked {
        map: MapKind::Normal,
        locked: true,
    }));
    c.save_project_to(root.join("material.texelith.json"), true)
        .unwrap();
    pump(&mut c);
    assert!(!c.session().is_dirty(), "{}", c.project_feedback());
    let saved = c.session().snapshot().unwrap().graph;
    let mut opened = Controls::new_material();
    opened.open_project_path(root.join("material.texelith.json"));
    pump(&mut opened);
    assert_eq!(
        MaterialGraphDto::from(opened.session().snapshot().unwrap().graph),
        MaterialGraphDto::from(saved)
    );
    assert!(opened.session().view_state().maps[&MapKind::Normal].locked);
    let before = opened.session().snapshot().unwrap();
    opened.open_project_path(root.join("absent.texelith.json"));
    pump(&mut opened);
    assert_eq!(opened.session().snapshot().unwrap(), before);
    assert!(!opened.project_feedback().is_empty());
}
#[test]
fn save_snapshot_does_not_clean_later_edits_and_save_as_rebases_links() {
    let (mut c, root) = fixture("save-edit");
    c.save_project_to(root.join("first.texelith.json"), true)
        .unwrap();
    c.action(UiAction::Request(CommandRequest::SetLocked {
        map: MapKind::Height,
        locked: true,
    }));
    pump(&mut c);
    assert!(c.session().is_dirty());
    let child = root.join("nested");
    std::fs::create_dir_all(&child).unwrap();
    c.save_project_to(child.join("second.texelith.json"), true)
        .unwrap();
    pump(&mut c);
    assert!(!c.session().is_dirty());
    let dto = texelith_io::project_io::read_json(&child.join("second.texelith.json")).unwrap();
    assert_eq!(dto["assets"][0]["relative_path"], "../source.png");
}
#[test]
fn changed_and_missing_assets_require_explicit_choice_and_stale_open_cannot_publish() {
    let (mut c, root) = fixture("relink");
    let path = root.join("material.texelith.json");
    c.save_project_to(path.clone(), true).unwrap();
    pump(&mut c);
    texelith_io::write_normal_png(
        root.join("source.png"),
        64,
        64,
        &vec![200; 64 * 64 * 3],
        true,
    )
    .unwrap();
    let mut opened = Controls::new_material();
    opened.open_project_path(path.clone());
    pump(&mut opened);
    assert!(opened.session().snapshot().is_none());
    assert_eq!(opened.project_asset_issues().len(), 1);
    opened
        .resolve_project_asset(AssetChoice::UseChanged)
        .unwrap();
    pump(&mut opened);
    assert!(opened.session().snapshot().is_some());
    assert!(
        opened.session().is_dirty(),
        "accepting changed assets must require saving updated hashes"
    );
    opened.open_project_path(path);
    opened.cancel_project_open();
    pump(&mut opened);
    assert!(opened.project_asset_issues().is_empty());
}
#[test]
fn projects_workflow_and_unknown_smoke_names_are_validated() {
    for scenario in ["projects", "workflow"] {
        let args = [
            "--mvp-smoke",
            scenario,
            "--report",
            "target/evidence/T-088/test.json",
        ]
        .map(str::to_owned);
        assert!(texelith_ui_egui::startup::Config::parse(args).is_ok());
    }
    assert!(
        texelith_ui_egui::startup::Config::parse(["--mvp-smoke".into(), "unknown".into()]).is_err()
    );
}

#[test]
fn dirty_replacement_resumes_only_after_committed_current_revision() {
    use texelith_ui_egui::input_locale::DirtyChoice;
    let (mut c, root) = fixture("dirty-save");
    c.save_project_to(root.join("saved.texelith.json"), true)
        .unwrap();
    pump(&mut c);
    c.action(UiAction::Request(CommandRequest::SetLocked {
        map: MapKind::Height,
        locked: true,
    }));
    c.drop_paths(vec![root.join("replacement.png")]);
    c.resolve_dirty(DirtyChoice::Save);
    assert!(c.pending_replacement().is_some());
    pump(&mut c);
    assert!(c.pending_replacement().is_none());
    assert!(matches!(
        c.take_effects().as_slice(),
        [Effect::LoadImage { .. }]
    ));
}

#[test]
fn completing_an_old_save_does_not_attach_its_path_to_a_new_import() {
    let (mut c, root) = fixture("save-import-race");
    c.save_project_to(root.join("old.texelith.json"), true)
        .unwrap();
    c.open_image(root.join("new.png"));
    let Effect::LoadImage { token, .. } = c.take_effects().remove(0) else {
        panic!()
    };
    c.complete_import(
        token,
        SourceDescriptor {
            content_hash: [99; 32],
            extent: Extent::new(64, 64).unwrap(),
            interpretation: InputInterpretation::SrgbPhoto,
        },
    )
    .unwrap();
    pump(&mut c);
    assert!(c.project_path().is_none());
    assert!(c.session().is_dirty());
}

#[test]
fn restart_is_explicit_and_preserves_completed_items_and_captured_recipe() {
    use texelith_ui_egui::project_batch::{RestartChoice, restart_batch};
    let (mut c, root) = fixture("restart");
    c.save_project_to(root.join("material.texelith.json"), true)
        .unwrap();
    pump(&mut c);
    let snapshot = c.session().snapshot().unwrap();
    let settings = c.session().export_settings().unwrap().clone();
    let capture =
        ExportCapture::new(&snapshot, c.session().document_revision(), &settings).unwrap();
    let items = (0..2)
        .map(|i| BatchInput {
            source: OpaqueLocator(format!("source-{i}")),
            source_digest: [i as u8; 32],
            destinations: (0..settings.outputs.len())
                .map(|n| OpaqueLocator(format!("item-{i}-output-{n}")))
                .collect(),
        })
        .collect();
    let job = BatchJob::new(100, capture, settings, items, false).unwrap();
    let mut checkpoint = job.checkpoint();
    checkpoint.items[0].state = BatchState::Committed;
    checkpoint.items[1].state = BatchState::Evaluating;
    let recovered = BatchJob::restore(checkpoint).unwrap();
    assert!(!recovered.can_start());
    assert!(
        restart_batch(&recovered, RestartChoice::KeepStopped)
            .unwrap()
            .is_none()
    );
    let resumed = restart_batch(&recovered, RestartChoice::RetryIncomplete)
        .unwrap()
        .unwrap();
    assert_eq!(resumed.items().len(), 1);
    assert_eq!(resumed.items()[0].index, 1);
    assert_eq!(resumed.recipe(), recovered.recipe());
    assert!(resumed.can_start());
    assert_eq!(recovered.items()[0].state, BatchState::Committed);
}

#[test]
fn persisted_view_changes_have_their_own_save_revision() {
    let (mut c, root) = fixture("view-save");
    let path = root.join("view.texelith.json");
    c.save_project_to(path.clone(), true).unwrap();
    pump(&mut c);
    c.action(UiAction::Request(CommandRequest::SetSplit {
        fraction: 0.25,
    }));
    assert!(!c.session().is_dirty());
    assert!(c.session().has_unsaved_project());
    c.save_project_to(path.clone(), false).unwrap();
    c.action(UiAction::Request(CommandRequest::SetSplit {
        fraction: 0.75,
    }));
    pump(&mut c);
    assert!(c.session().has_unsaved_project());
    c.save_project_to(path, false).unwrap();
    pump(&mut c);
    assert!(!c.session().has_unsaved_project());
}

#[test]
fn escape_closes_save_as_before_cancelling_the_dirty_replacement() {
    use texelith_ui_egui::input_locale::{DirtyChoice, contract::key};
    let (mut c, root) = fixture("modal-escape");
    c.drop_paths(vec![root.join("replacement.png")]);
    c.resolve_dirty(DirtyChoice::Save);
    assert!(c.project_save_picker_open());
    assert!(c.pending_replacement().is_some());
    let ctx = egui::Context::default();
    let frame = |c: &mut Controls, events| {
        let mut output = ctx.run_ui(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(1280.0, 720.0),
                )),
                focused: true,
                events,
                ..Default::default()
            },
            |ui| c.show(ui),
        );
        output.textures_delta.clear();
    };
    frame(&mut c, vec![key(egui::Key::Escape, egui::Modifiers::NONE)]);
    assert!(!c.project_save_picker_open());
    assert!(c.pending_replacement().is_some());
    frame(
        &mut c,
        vec![egui::Event::Key {
            key: egui::Key::Escape,
            physical_key: None,
            pressed: false,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        }],
    );
    frame(&mut c, vec![key(egui::Key::Escape, egui::Modifiers::NONE)]);
    assert!(c.pending_replacement().is_none());
    assert!(c.session().is_dirty());
}
