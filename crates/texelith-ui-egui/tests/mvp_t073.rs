use egui::{Key, Modifiers};
use std::path::PathBuf;
use texelith_core::{
    CommandRequest, Effect, Extent, InputInterpretation, MapKind, SourceDescriptor,
};
use texelith_ui_egui::{
    command_adapter::UiAction,
    controls::Controls,
    input_locale::{DirtyChoice, DropIntent, InputAction, classify_drop, labels, shortcut},
    material_ui::MAPS,
};

#[path = "mvp_t073/ui.rs"]
mod ui;

fn material() -> Controls {
    let mut c = Controls::new_material();
    c.open_image("fixture.png".into());
    let Effect::LoadImage { token, .. } = c.take_effects().remove(0) else {
        panic!("load")
    };
    c.complete_import(
        token,
        SourceDescriptor {
            content_hash: [3; 32],
            extent: Extent {
                width: 64,
                height: 32,
            },
            interpretation: InputInterpretation::SrgbPhoto,
        },
    )
    .unwrap();
    c
}

#[test]
fn shortcut_requests_match_menu_and_palette_and_respect_editing() {
    for (key, map) in [
        Key::Num1,
        Key::Num2,
        Key::Num3,
        Key::Num4,
        Key::Num5,
        Key::Num6,
        Key::Num7,
    ]
    .into_iter()
    .zip(MAPS)
    {
        let expected = InputAction::Request(CommandRequest::SelectMap { map });
        assert_eq!(
            shortcut(key, Modifiers::NONE, false, false),
            Some(expected.clone())
        );
        assert_eq!(shortcut(key, Modifiers::NONE, true, false), None);
        assert_eq!(shortcut(key, Modifiers::NONE, false, true), None);
        assert_eq!(shortcut(key, Modifiers::CTRL, false, false), None);
        for action in [
            expected.clone(),
            expected.clone(),
            shortcut(key, Modifiers::NONE, false, false).unwrap(),
        ] {
            let mut c = material();
            let revision = c.session().document_revision();
            c.input_action(action);
            assert_eq!(
                c.session()
                    .snapshot()
                    .unwrap()
                    .graph
                    .selected_output_ref()
                    .port
                    .map_kind(),
                map
            );
            assert_eq!(c.session().document_revision(), revision);
            assert_eq!(c.session().history().undo_entries, 0);
        }
    }
    for modifiers in [
        Modifiers::CTRL,
        Modifiers::COMMAND,
        Modifiers {
            mac_cmd: true,
            command: true,
            ..Modifiers::NONE
        },
    ] {
        assert_eq!(
            shortcut(Key::K, modifiers, true, false),
            Some(InputAction::Palette)
        );
        assert_eq!(
            shortcut(Key::O, modifiers, false, false),
            Some(InputAction::OpenProject)
        );
        assert_eq!(
            shortcut(Key::S, modifiers, false, false),
            Some(InputAction::SaveProject { save_as: false })
        );
        assert_eq!(shortcut(Key::Z, modifiers, true, false), None);
        assert_eq!(
            shortcut(Key::Z, modifiers, false, false),
            Some(InputAction::Request(CommandRequest::Undo))
        );
        let shift = Modifiers {
            shift: true,
            ..modifiers
        };
        assert_eq!(
            shortcut(Key::Z, shift, false, false),
            Some(InputAction::Request(CommandRequest::Redo))
        );
        assert_eq!(
            shortcut(Key::S, shift, false, false),
            Some(InputAction::SaveProject { save_as: true })
        );
    }
}

#[test]
fn drops_are_typed_and_mixed_or_unsupported_lists_are_rejected_atomically() {
    for name in ["photo.PNG", "photo.jpg", "photo.JPEG"] {
        assert_eq!(
            classify_drop(&[name.into()]).unwrap(),
            DropIntent::Image(name.into())
        );
    }
    assert!(matches!(
        classify_drop(&["a.png".into(), "b.jpg".into()]),
        Ok(DropIntent::Batch(_))
    ));
    assert_eq!(
        classify_drop(&["my.TEXELITH.JSON".into()]).unwrap(),
        DropIntent::Project("my.TEXELITH.JSON".into())
    );
    for name in ["studio.hdr", "studio.exr"] {
        assert!(matches!(
            classify_drop(&[name.into()]),
            Ok(DropIntent::Environment(_))
        ));
    }
    for name in ["mesh.obj", "mesh.gltf", "mesh.glb"] {
        assert!(matches!(
            classify_drop(&[name.into()]),
            Ok(DropIntent::Mesh(_))
        ));
    }
    for paths in [
        vec![],
        vec!["a.txt".into()],
        vec!["a.png".into(), "b.obj".into()],
        vec!["a.texelith.json".into(), "b.png".into()],
    ] {
        assert!(classify_drop(&paths).is_err());
    }
}

#[test]
fn dirty_replacement_cancel_save_pending_and_discard_preserve_the_right_state() {
    let mut c = material();
    c.action(UiAction::Request(CommandRequest::SetLocked {
        map: MapKind::Normal,
        locked: true,
    }));
    assert!(c.session().is_dirty());
    let revision = c.session().document_revision();
    for choice in [DirtyChoice::Cancel, DirtyChoice::Save] {
        c.drop_paths(vec![PathBuf::from("replacement.png")]);
        assert!(c.pending_replacement().is_some());
        assert!(c.take_effects().is_empty());
        c.resolve_dirty(choice);
        assert_eq!(c.session().document_revision(), revision);
        assert!(c.session().is_dirty());
        assert!(c.take_effects().is_empty());
        if choice == DirtyChoice::Save {
            assert!(c.pending_replacement().is_some());
            assert!(c.project_save_picker_open());
            c.resolve_dirty(DirtyChoice::Cancel);
        }
    }
    c.drop_paths(vec!["replacement.png".into()]);
    c.resolve_dirty(DirtyChoice::Discard);
    assert!(c.pending_replacement().is_none());
    assert!(matches!(
        c.take_effects().as_slice(),
        [Effect::LoadImage { .. }]
    ));
    assert_eq!(
        c.session().document_revision(),
        revision,
        "dispatch does not replace before successful decode"
    );
    c.drop_paths(vec!["project.texelith.json".into()]);
    c.resolve_dirty(DirtyChoice::Cancel);
    assert!(c.take_effects().is_empty());
}

#[test]
fn pending_handlers_never_report_success_or_replace_document() {
    let mut c = material();
    let revision = c.session().document_revision();
    for (paths, key) in [
        (vec!["a.png", "b.jpg"], "batch-offer"),
        (vec!["studio.hdr"], "environment-pending"),
        (vec!["mesh.glb"], "mesh-pending"),
        (vec!["wrong.txt"], "drop-unsupported"),
    ] {
        c.drop_paths(paths.into_iter().map(PathBuf::from).collect());
        assert_eq!(c.input_feedback_key(), Some(key));
        assert!(c.take_effects().is_empty());
        assert_eq!(c.session().document_revision(), revision);
    }
    c.drop_paths(vec!["project.texelith.json".into()]);
    assert!(
        c.pending_replacement().is_some(),
        "new imports are unsaved documents"
    );
    c.resolve_dirty(DirtyChoice::Discard);
    let effects = c.take_effects();
    assert!(matches!(effects.as_slice(), [Effect::OpenProject { .. }]));
    assert_eq!(c.input_feedback_key(), Some("project-open-pending"));
    assert_eq!(c.session().document_revision(), revision);
}

#[test]
fn catalogs_cover_input_commands_and_existing_control_catalogs_with_english_fallback() {
    labels::validate_catalogs().unwrap();
    for key in labels::keys() {
        assert!(!labels::lookup("en-US", &key).is_empty(), "{key}");
        assert!(!labels::lookup("tr-TR", &key).is_empty(), "{key}");
    }
    assert_eq!(labels::lookup("fr-FR", "cancel"), "Cancel");
    assert_eq!(labels::lookup("tr-TR", "missing-key"), "missing-key");
    assert_eq!(labels::lookup("tr-TR", "cancel"), "İptal");
    assert_eq!(
        labels::with_fallback("cancel", "", "cancel = Cancel").unwrap(),
        "Cancel"
    );
    assert!(labels::check_coverage("cancel = Cancel", "").is_err());
    for descriptor in texelith_core::command_descriptors() {
        assert!(
            labels::keys().contains(&labels::command_key(descriptor.id)),
            "{}",
            descriptor.id.as_str()
        );
    }
}

#[test]
fn input_diagnostic_is_recognized_and_unknown_fails() {
    use texelith_ui_egui::startup::Config;
    assert!(
        Config::parse(["--mvp-smoke", "input", "--report", "out.json"].map(str::to_owned))
            .unwrap()
            .input_smoke
    );
    assert!(
        Config::parse(["--mvp-smoke", "unknown", "--report", "out.json"].map(str::to_owned))
            .is_err()
    );
}

#[test]
fn real_egui_events_cover_all_panels_and_focus_guards() {
    assert!(
        texelith_ui_egui::input_locale::contract::exercise()
            .unwrap()
            .len()
            >= 17
    );
}

#[test]
fn locale_and_snapshot_arguments_are_validated() {
    use texelith_ui_egui::startup::Config;
    assert_eq!(
        Config::parse(["--locale", "tr-TR"].map(str::to_owned))
            .unwrap()
            .locale,
        "tr-TR"
    );
    assert!(Config::parse(["--locale", "unknown"].map(str::to_owned)).is_err());
    assert!(Config::parse(["--input-snapshots"].map(str::to_owned)).is_err());
}
