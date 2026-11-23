use std::path::PathBuf;
use texelith_ui_egui::batch_jobs::{InputSelection, enumerate};

fn directory(name: &str) -> PathBuf {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../target-evidence/T-087")
        .join(format!("{name}-{}", std::process::id()));
    std::fs::create_dir_all(&path).unwrap();
    path
}

#[test]
fn enumeration_is_sorted_deduplicated_and_recursion_is_explicit() {
    let root = directory("enumerate");
    std::fs::create_dir_all(root.join("nested")).unwrap();
    for name in ["z.png", "a.JPG", "ignored.txt", "nested/b.png"] {
        std::fs::write(root.join(name), b"fixture").unwrap();
    }
    let flat = enumerate(&InputSelection::Folder {
        path: root.clone(),
        recursive: false,
    })
    .unwrap();
    assert_eq!(
        flat.iter().map(|s| s.name.as_str()).collect::<Vec<_>>(),
        ["a.JPG", "z.png"]
    );
    let recursive = enumerate(&InputSelection::Folder {
        path: root.clone(),
        recursive: true,
    })
    .unwrap();
    assert_eq!(
        recursive
            .iter()
            .map(|s| s.name.as_str())
            .collect::<Vec<_>>(),
        ["a.JPG", "nested/b.png", "z.png"]
    );
    let files = enumerate(&InputSelection::Files(vec![
        root.join("z.png"),
        root.join("a.JPG"),
        root.join("./z.png"),
    ]))
    .unwrap();
    assert_eq!(files.len(), 2);
    assert_eq!(files[0].name, "a.JPG");
}

#[test]
fn memory_reservations_reject_overflow_and_preserve_interactive_headroom() {
    use texelith_ui_egui::batch_jobs::{CPU_BUDGET, INTERACTIVE_RESERVATION, export_reservation};
    assert_eq!(CPU_BUDGET, 512 * 1024 * 1024);
    assert_eq!(INTERACTIVE_RESERVATION, 256 * 1024 * 1024);
    assert!(export_reservation(1024, 1024, 16).unwrap() + INTERACTIVE_RESERVATION <= CPU_BUDGET);
    assert!(export_reservation(u32::MAX, u32::MAX, 16).is_err());
    assert!(export_reservation(4096, 4096, 16).is_err());
    assert!(export_reservation(4096, 4096, 3).unwrap() + INTERACTIVE_RESERVATION <= CPU_BUDGET);
}

#[test]
fn presentation_accounts_for_numeric_heatmap_readback_and_display_allocations() {
    use texelith_ui_egui::batch_jobs::{
        CPU_BUDGET, INTERACTIVE_RESERVATION, presentation_reservation,
    };
    let plain = presentation_reservation(1024, 1024, false, false).unwrap();
    let heat = presentation_reservation(1024, 1024, true, false).unwrap();
    assert!(plain >= 1024 * 1024 * (16 + 8));
    assert!(heat >= 1024 * 1024 * (16 + 16 + 4));
    assert!(heat > plain);
    assert!(heat + INTERACTIVE_RESERVATION <= CPU_BUDGET);
    assert!(presentation_reservation(4096, 4096, true, false).unwrap() <= 64 * 1024 * 1024);
    assert!(
        presentation_reservation(4096, 4096, false, true).unwrap() + INTERACTIVE_RESERVATION
            <= CPU_BUDGET
    );
    assert!(presentation_reservation(u32::MAX, u32::MAX, true, true).is_err());
    assert!(presentation_reservation(1024, 4096, true, false).unwrap() <= 64 * 1024 * 1024);
}

#[test]
fn batch_smoke_uses_real_routing_and_checks_cancel_retry_and_bounds() {
    let mut config = texelith_ui_egui::startup::Config::parse(
        [
            "--mvp-smoke",
            "batch",
            "--size",
            "32",
            "--report",
            "target-evidence/T-087/test.json",
        ]
        .map(str::to_owned),
    )
    .unwrap();
    config.report = Some(directory("gpu").join("test.json"));
    assert!(config.batch_smoke);
    let report = texelith_ui_egui::batch_jobs::smoke::exercise(&config).unwrap();
    for name in [
        "continue_on_error",
        "stop_on_error",
        "collisions",
        "cancel_retry",
        "immutable_recipe",
        "recovery",
        "mixed_extents",
        "source_residency",
        "manifest_collisions",
    ] {
        assert_eq!(report[name], true, "{name}");
    }
    assert!(report["peak_active"].as_u64().unwrap() <= 1);
    assert!(report["peak_prefetch"].as_u64().unwrap() <= 1);
    assert!(report["cpu_reserved_bytes"].as_u64().unwrap() <= 512 * 1024 * 1024);
    assert!(report["gpu_peak_bytes"].as_u64().unwrap() <= 1024 * 1024 * 1024);
    assert!(
        texelith_ui_egui::startup::Config::parse(
            ["--mvp-smoke", "unknown", "--report", "x"].map(str::to_owned)
        )
        .is_err()
    );
}
