//! Explicit hardware regression: exercises Session, worker mailboxes and the real window.
#[path = "../../texelith-gpu/tests/common/gpu_availability.rs"]
mod gpu_availability;

#[test]
#[ignore = "requires an interactive Windows desktop and RTX 5070 Ti / DX12"]
fn perf_metrics_completed_proxies_survive_the_next_input() {
    if gpu_availability::skip_if_no_gpu() {
        return;
    }
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis();
    let report = std::path::Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("evidence/T-020/handoff-regression-{stamp}.json"));
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_texelith-ui-egui"))
        .current_dir(&root)
        .args([
            "--perf-scenario",
            "normal-drag",
            "--size",
            "4096",
            "--duration-seconds",
            "10",
            "--backend",
            "dx12",
            "--adapter-name",
            "RTX 5070 Ti",
            "--report",
        ])
        .arg(&report)
        .output()
        .expect("launch real app");
    assert!(
        report.exists(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let data: serde_json::Value = serde_json::from_slice(&std::fs::read(&report).unwrap()).unwrap();
    assert_eq!(data["trace"]["completed"], true);
    assert_eq!(data["trace"]["dropped_samples"], 0);
    let metrics = &data["gate"]["metrics"];
    assert!(metrics["drag_frames"].as_u64().unwrap() >= 500);
    // Fast frames repeating the warm-up image must not pass this regression.
    assert!(
        metrics["fresh_publications"].as_u64().unwrap() >= 100,
        "completed proxy mailbox starved: {metrics}; report {}",
        report.display()
    );
    assert_eq!(data["trace"]["counters"]["stale_publications"], 0);
    assert_eq!(data["trace"]["counters"]["validation_errors"], 0);
    // Other performance thresholds remain enforced by the full three-run suite.
}
