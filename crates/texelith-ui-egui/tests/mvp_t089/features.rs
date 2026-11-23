use std::{path::Path, process::Command};

#[test]
fn linux_resolves_both_native_window_systems_with_only_approved_eframe_features() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let output = Command::new(env!("CARGO"))
        .current_dir(root)
        .args([
            "metadata",
            "--format-version",
            "1",
            "--locked",
            "--offline",
            "--filter-platform",
            "x86_64-unknown-linux-gnu",
        ])
        .output()
        .expect("run locked Cargo metadata for Linux");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let metadata: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    let packages = metadata["packages"].as_array().unwrap();
    let ui = packages
        .iter()
        .find(|p| p["name"] == "texelith-ui-egui")
        .unwrap();
    let eframe = ui["dependencies"]
        .as_array()
        .unwrap()
        .iter()
        .find(|d| d["name"] == "eframe")
        .unwrap();
    assert_eq!(eframe["req"], "=0.37.0");
    assert_eq!(eframe["uses_default_features"], false);
    let mut features: Vec<_> = eframe["features"]
        .as_array()
        .unwrap()
        .iter()
        .map(|f| f.as_str().unwrap())
        .collect();
    features.sort_unstable();
    assert_eq!(
        features,
        ["clipboard", "default_fonts", "wayland", "wgpu", "x11"]
    );

    let winit: Vec<_> = packages.iter().filter(|p| p["name"] == "winit").collect();
    assert_eq!(
        winit.len(),
        1,
        "native windowing must use one locked winit identity"
    );
    let node = metadata["resolve"]["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .find(|n| n["id"] == winit[0]["id"])
        .unwrap();
    let features = node["features"].as_array().unwrap();
    for required in ["x11", "wayland"] {
        assert!(
            features.iter().any(|f| f == required),
            "Linux native feature {required} was not forwarded to winit: {features:?}"
        );
    }
    for required in ["x11-dl", "wayland-client"] {
        let dependency = packages
            .iter()
            .find(|p| p["name"] == required)
            .unwrap_or_else(|| panic!("missing locked {required}"));
        assert!(
            metadata["resolve"]["nodes"]
                .as_array()
                .unwrap()
                .iter()
                .any(|n| n["id"] == dependency["id"]),
            "{required} absent from Linux resolution"
        );
    }
}
