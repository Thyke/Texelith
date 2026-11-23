use super::*;
use crate::{command_adapter::UiAction, controls::Controls};
use std::path::Path;

fn material(width: u32, height: u32) -> Controls {
    let mut c = Controls::new_material();
    c.open_image("folder-regression.png".into());
    let Effect::LoadImage { token, .. } = c.take_effects().remove(0) else {
        panic!("missing import effect")
    };
    c.complete_import(
        token,
        SourceDescriptor {
            content_hash: [151; 32],
            extent: Extent::new(width, height).unwrap(),
            interpretation: InputInterpretation::SrgbPhoto,
        },
    )
    .unwrap();
    c
}

fn folders() -> Vec<PathBuf> {
    let mut paths = vec![
        PathBuf::from("folder with spaces"),
        PathBuf::from("ç ğ ı ö ş ü İ"),
        PathBuf::from("existing-non-empty"),
        PathBuf::from(
            "OneDrive - Texture Studio/Documents/Materials/Client Projects/Stone surfaces/Final approved textures/Exports",
        ),
        PathBuf::from(format!("trailing{}", std::path::MAIN_SEPARATOR)),
    ];
    #[cfg(windows)]
    paths.push(PathBuf::from("D:\\"));
    #[cfg(not(windows))]
    paths.push(PathBuf::from("/"));
    paths
}

#[test]
fn picked_parent_never_enters_plan_and_always_receives_a_named_child() {
    let mut c = material(64, 32);
    let plan = c.packing_plan().unwrap();
    for base in folders() {
        c.select_export_folder(base.clone());
        assert_eq!(c.export_options.destination, base.join("Material"));
        assert_eq!(c.packing_plan().unwrap(), plan, "{}", base.display());
        c.request_material_export().unwrap();
        let exports = c.take_export_effects();
        let [(Effect::Export { plan: captured, .. }, options)] = exports.as_slice() else {
            panic!("missing export effect")
        };
        assert_eq!(captured, &plan);
        assert!(options.destination.is_absolute());
        assert_eq!(options.destination.file_name().unwrap(), "Material");
        assert!(!options.replace_existing);
    }
}

#[test]
fn odd_size_bc5_export_is_independent_of_folder() {
    let mut c = material(63, 33);
    for folder in folders() {
        c.select_export_folder(folder);
        c.request_material_export().unwrap();
        assert_eq!(c.take_export_effects().len(), 1);
    }
}

#[test]
fn default_engine_presets_accept_odd_sizes() {
    for (width, height) in [(1, 1), (3, 5), (1023, 1001), (4095, 4093)] {
        let mut c = material(width, height);
        for preset in ["unreal-v1", "unity-urp-v1", "unity-hdrp-v1", "godot-v1"] {
            c.select_export_preset(preset).unwrap();
            c.request_material_export().unwrap();
            assert_eq!(
                c.take_export_effects().len(),
                1,
                "{preset}: {width}x{height}"
            );
        }
    }
}

#[test]
fn invalid_fields_give_specific_errors_without_emitting_exports() {
    let mut c = material(64, 32);
    for (name, reason) in [
        ("bad:name", "invalid characters"),
        ("", "required"),
        ("CON", "reserved"),
        ("trailing.", "trailing"),
    ] {
        c.export_options.material_name = name.into();
        let error = c.request_material_export().unwrap_err();
        assert!(error.contains(reason), "{error}");
        assert!(c.take_effects().is_empty());
    }
    c.export_options.material_name = "Material".into();
    c.native_files.naming = Some("{unknown}_{map}".into());
    assert!(
        c.request_material_export()
            .unwrap_err()
            .contains("unknown token")
    );
    c.native_files.naming = None;
    c.export_options.destination = PathBuf::new();
    assert!(!c.request_material_export().unwrap_err().is_empty());
    c.export_options.destination = std::path::absolute(Path::new("/")).unwrap();
    assert!(
        c.request_material_export()
            .unwrap_err()
            .contains("named material folder")
    );
    assert!(c.take_effects().is_empty());
}

#[test]
fn invalid_plan_variants_report_the_actual_validation_rule() {
    let c = material(64, 32);
    let plan = c.packing_plan().unwrap();
    let mut invalid = plan.clone();
    invalid.outputs.clear();
    assert!(
        validate_export(&invalid)
            .unwrap_err()
            .to_string()
            .contains("at least one map")
    );
    invalid = plan.clone();
    invalid.size = SizePolicy::Pad {
        width: 32,
        height: 16,
    };
    assert!(
        validate_export(&invalid)
            .unwrap_err()
            .to_string()
            .contains("Padding size")
    );
    invalid = plan;
    invalid.naming = "../{name}".into();
    assert!(
        validate_export(&invalid)
            .unwrap_err()
            .to_string()
            .contains("Naming template")
    );
}

fn text_in(shape: &egui::epaint::Shape, needle: &str) -> bool {
    match shape {
        egui::epaint::Shape::Text(text) => text.galley.text().contains(needle),
        egui::epaint::Shape::Vec(shapes) => shapes.iter().any(|s| text_in(s, needle)),
        _ => false,
    }
}

#[test]
fn failed_export_renders_a_specific_message_beside_export_and_tooltip_detail() {
    let mut c = material(63, 33);
    c.export_options.material_name = "bad:name".into();
    c.confirm_material_export();
    let ctx = egui::Context::default();
    ctx.all_styles_mut(|style| {
        style.animation_time = 0.;
        style.interaction.tooltip_delay = 0.;
        style.interaction.show_tooltips_only_when_still = false;
    });
    let mut output = ctx.run_ui(
        egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(1280., 800.),
            )),
            ..Default::default()
        },
        |ui| c.show(ui),
    );
    output.textures_delta.clear();
    assert!(
        output
            .shapes
            .iter()
            .any(|s| text_in(&s.shape, "invalid characters"))
    );
    assert!(
        !output
            .shapes
            .iter()
            .any(|s| text_in(&s.shape, "invalid export plan"))
    );
    // Async failures retain the captured destination/output even after form edits.
    c.set_export_status(Some(JobStatus {
        job_id: 151,
        stage: ExportStage::Failed,
        output_name: "Normal".into(),
        completed_items: 0,
        total_items: 1,
        destination: PathBuf::from("captured-set"),
        error: Some("Normal: Folder is not writable\nDestination: captured-set\nAccess denied (os error 5)".into()),
        cancel_requested: false,
    }), false);
    assert!(c.native_files.export_detail.contains("captured-set"));
    assert!(c.native_files.export_detail.contains("os error 5"));
    let mut output = ctx.run_ui(egui::RawInput::default(), |ui| c.show(ui));
    output.textures_delta.clear();
    fn position(shape: &egui::epaint::Shape, needle: &str) -> Option<egui::Pos2> {
        match shape {
            egui::epaint::Shape::Text(text) if text.galley.text() == needle => {
                Some(text.pos + text.galley.size() * 0.5)
            }
            egui::epaint::Shape::Vec(shapes) => shapes.iter().find_map(|s| position(s, needle)),
            _ => None,
        }
    }
    let ready = crate::workspace::labels::text("ready");
    let status = output
        .shapes
        .iter()
        .find_map(|s| position(&s.shape, &ready))
        .unwrap();
    let mut tooltip_seen = false;
    for time in [1., 2., 3.] {
        let mut output = ctx.run_ui(
            egui::RawInput {
                time: Some(time),
                events: vec![egui::Event::PointerMoved(status)],
                ..Default::default()
            },
            |ui| c.show(ui),
        );
        output.textures_delta.clear();
        tooltip_seen |= output
            .shapes
            .iter()
            .any(|s| text_in(&s.shape, "os error 5"));
    }
    assert!(
        tooltip_seen,
        "status-bar hover must expose technical error detail"
    );
    // Dispatch still has no export work after a validation failure.
    c.action(UiAction::FocusLost);
    assert!(c.take_export_effects().is_empty());
}
