//! Native bootstrap contracts; these tests do not create a window or GPU device.
use std::{path::PathBuf, process::Command, time::Duration};

use texelith_ui_egui::{
    measurement::{Lifecycle, SHUTDOWN_LIMIT, STARTUP_LIMIT, WindowSample},
    startup::Config,
};

#[path = "mvp_t089/features.rs"]
mod features;
#[path = "mvp_t089/native_args.rs"]
mod native_args;

fn parse(args: &[&str]) -> Result<Config, String> {
    Config::parse(args.iter().map(|arg| (*arg).to_owned()))
}

fn native_backend() -> (&'static str, wgpu::Backends) {
    if cfg!(target_os = "windows") {
        ("dx12", wgpu::Backends::DX12)
    } else if cfg!(target_os = "macos") {
        ("metal", wgpu::Backends::METAL)
    } else if cfg!(target_os = "linux") {
        ("vulkan", wgpu::Backends::VULKAN)
    } else {
        panic!("T-089 native coverage requires Windows, macOS or Linux")
    }
}

#[test]
fn default_config_selects_the_host_native_backend() {
    let config = Config::default();
    let (name, backend) = native_backend();
    assert_eq!(config.backend, name);
    assert_eq!(config.backends().unwrap(), backend);
    assert!(!config.render_smoke && !config.integration_smoke && !config.perf_scenario);
    assert!(config.input.is_none() && config.report.is_none());
}

#[test]
fn each_explicit_backend_is_exact_and_keeps_the_adapter_override() {
    for (name, expected) in [
        ("dx12", wgpu::Backends::DX12),
        ("metal", wgpu::Backends::METAL),
        ("vulkan", wgpu::Backends::VULKAN),
    ] {
        let config = parse(&["--backend", name, "--adapter-name", "Chosen GPU"]).unwrap();
        assert_eq!(config.backend, name);
        assert_eq!(config.backends().unwrap(), expected);
        assert_eq!(config.adapter_name.as_deref(), Some("Chosen GPU"));
        let options = config.native_options().unwrap();
        let egui_wgpu::WgpuSetup::CreateNew(setup) = options.wgpu_options.wgpu_setup else {
            panic!("native bootstrap must use the shared eframe device")
        };
        assert_eq!(setup.instance_descriptor.backends, expected);
        let error = setup.native_adapter_selector.unwrap()(&[], None).unwrap_err();
        assert!(error.contains("Chosen GPU"), "{error}");
    }
}

#[test]
fn unsupported_backend_and_blank_adapter_fail_before_window_creation() {
    for backend in ["", "gl", "glow", "cpu", "dx12,metal", "not-a-backend"] {
        let error = parse(&["--backend", backend]).unwrap_err();
        assert!(error.contains("backend"), "{backend:?}: {error}");
    }
    for args in [
        vec!["--backend"],
        vec!["--adapter-name"],
        vec!["--adapter-name", " \t "],
    ] {
        assert!(parse(&args).is_err(), "{args:?}");
    }
}

#[test]
fn native_options_keep_fifo_validation_and_surface_compatible_selection() {
    let options = Config::default().native_options().unwrap();
    assert!(matches!(options.renderer, eframe::Renderer::Wgpu));
    assert_eq!(
        options.wgpu_options.surface.present_mode,
        wgpu::PresentMode::Fifo
    );
    assert_eq!(
        options.wgpu_options.surface.desired_maximum_frame_latency,
        Some(2)
    );
    let egui_wgpu::WgpuSetup::CreateNew(setup) = options.wgpu_options.wgpu_setup else {
        panic!("bootstrap must let eframe create the shared device")
    };
    assert_eq!(setup.instance_descriptor.backends, native_backend().1);
    assert_eq!(
        setup.power_preference,
        wgpu::PowerPreference::HighPerformance
    );
    assert!(
        setup
            .instance_descriptor
            .flags
            .contains(wgpu::InstanceFlags::VALIDATION)
    );
    let error = setup.native_adapter_selector.unwrap()(&[], None).unwrap_err();
    assert!(
        error.contains("adapter"),
        "missing adapters must be an error: {error}"
    );
}

#[test]
fn paths_with_spaces_unicode_and_relative_components_are_not_rewritten() {
    let input = PathBuf::from("textures").join("çalışma taş 日本語.png");
    let normal = PathBuf::from("exports")
        .join(".. raw")
        .join("normal DX.png");
    let report = PathBuf::from("reports").join("ölçüm sonuç.json");
    let config = parse(&[
        "--backend",
        native_backend().0,
        "--input",
        input.to_str().unwrap(),
        "--integration-smoke",
        "--save-normal",
        normal.to_str().unwrap(),
        "--report",
        report.to_str().unwrap(),
    ])
    .unwrap();
    assert_eq!(config.input.as_ref(), Some(&input));
    assert_eq!(config.save_normal.as_ref(), Some(&normal));
    assert_eq!(config.report.as_ref(), Some(&report));
    assert!(config.integration_smoke);

    let snapshots = PathBuf::from("captures").join("arayüz 150%");
    let config = parse(&["--ui-snapshot", snapshots.to_str().unwrap()]).unwrap();
    assert_eq!(config.ui_snapshot, Some(snapshots));
}

// Child processes isolate environment overrides without mutating the test runner's
// environment (other tests may concurrently construct native options).
fn backend_probe(environment: Option<&str>, arguments: &[&str], expected: &str) {
    let output = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "backend_environment_child", "--nocapture"])
        .env_remove("WGPU_BACKEND")
        .envs(environment.map(|value| ("WGPU_BACKEND", value)))
        .env("TEXELITH_T089_BACKEND_PROBE", expected)
        .env(
            "TEXELITH_T089_PROBE_ARGS",
            serde_json::to_string(arguments).unwrap(),
        )
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn backend_environment_child() {
    let Ok(expected) = std::env::var("TEXELITH_T089_BACKEND_PROBE") else {
        assert_eq!(Config::default().backends().unwrap(), native_backend().1);
        return;
    };
    let args: Vec<String> =
        serde_json::from_str(&std::env::var("TEXELITH_T089_PROBE_ARGS").unwrap()).unwrap();
    let config = Config::parse(args).unwrap();
    assert_eq!(config.backend, expected);
    let expected_backend = match expected.as_str() {
        "dx12" => wgpu::Backends::DX12,
        "metal" => wgpu::Backends::METAL,
        "vulkan" => wgpu::Backends::VULKAN,
        _ => panic!("unknown probe expectation"),
    };
    let options = config.native_options().unwrap();
    let egui_wgpu::WgpuSetup::CreateNew(setup) = options.wgpu_options.wgpu_setup else {
        panic!("native setup required")
    };
    assert_eq!(setup.instance_descriptor.backends, expected_backend);
}

#[test]
fn absent_environment_uses_the_native_default() {
    backend_probe(None, &[], native_backend().0);
}

#[test]
fn wgpu_backend_environment_overrides_the_native_default() {
    for backend in ["dx12", "metal", "vulkan"] {
        backend_probe(Some(backend), &[], backend);
    }
}

#[test]
fn explicit_backend_takes_precedence_over_the_environment() {
    for (environment, explicit) in [("metal", "vulkan"), ("vulkan", "dx12"), ("dx12", "metal")] {
        backend_probe(Some(environment), &["--backend", explicit], explicit);
    }
}

#[test]
fn invalid_cli_exits_with_a_diagnostic_without_initializing_a_window() {
    let output = Command::new(env!("CARGO_BIN_EXE_texelith-ui-egui"))
        .args(["--backend", "not-a-backend"])
        .env_remove("WGPU_BACKEND")
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1));
    let error = String::from_utf8_lossy(&output.stderr);
    assert!(error.contains("unsupported backend"), "{error}");
    assert!(!error.contains("panicked"), "{error}");
}

#[cfg(any(unix, windows))]
#[test]
fn a_native_path_argument_does_not_panic_during_argument_decoding() {
    #[cfg(unix)]
    let path = {
        use std::os::unix::ffi::OsStringExt;
        std::ffi::OsString::from_vec(b"textures/native-\xff.png".to_vec())
    };
    #[cfg(windows)]
    let path = {
        use std::os::windows::ffi::OsStringExt;
        std::ffi::OsString::from_wide(&[0x74, 0x65, 0x78, 0xd800, 0x2e, 0x70, 0x6e, 0x67])
    };
    // The invalid backend guarantees termination before eframe or filesystem IO.
    // The earlier path must survive native argument decoding up to validation.
    let output = Command::new(env!("CARGO_BIN_EXE_texelith-ui-egui"))
        .arg("--input")
        .arg(path)
        .args(["--backend", "not-a-backend"])
        .env_remove("WGPU_BACKEND")
        .output()
        .unwrap();
    let error = String::from_utf8_lossy(&output.stderr);
    assert!(
        !error.contains("panicked"),
        "native path caused a panic: {error}"
    );
    assert_eq!(output.status.code(), Some(1), "{error}");
    assert!(error.contains("unsupported backend"), "{error}");
}

#[cfg(all(target_os = "windows", target_env = "msvc"))]
#[test]
fn windows_msvc_binary_has_no_unbundled_dynamic_crt_assumption() {
    assert!(
        std::hint::black_box(cfg!(target_feature = "crt-static")),
        "T-089 requires target-scoped static MSVC CRT, or a reviewed bundled-runtime contract"
    );
}

#[test]
fn platform_bootstrap_keeps_t120_usable_frame_and_shutdown_deadlines() {
    assert_eq!(STARTUP_LIMIT, Duration::from_secs(30));
    assert_eq!(SHUTDOWN_LIMIT, Duration::from_secs(15));
    let mut lifecycle = Lifecycle::default();
    lifecycle.phase("first_logic_tick", 1);
    lifecycle.phase("first_ui_tick", 2);
    lifecycle.phase("first_render_completed", 3);
    lifecycle.sample(WindowSample {
        at_ns: 4,
        window_found: true,
        foreground: Some(true),
        visible: Some(true),
        minimized: Some(false),
        occluded: Some(false),
        ..Default::default()
    });
    assert!(lifecycle.final_reasons().is_empty());
    assert_eq!(
        lifecycle.deadline(STARTUP_LIMIT, 10),
        Some(("startup deadline exceeded", false))
    );
    lifecycle.phase("first_usable_frame", 5);
    assert_eq!(lifecycle.deadline(STARTUP_LIMIT, 10), None);
    lifecycle.phase("shutdown_started", STARTUP_LIMIT.as_nanos() as u64);
    assert_eq!(
        lifecycle.deadline(STARTUP_LIMIT + SHUTDOWN_LIMIT, 10),
        Some(("shutdown deadline exceeded", true))
    );
}

#[test]
fn platform_bootstrap_does_not_turn_missing_observation_into_a_pass() {
    let mut lifecycle = Lifecycle::default();
    lifecycle.phase("first_render_completed", 1);
    lifecycle.sample(WindowSample {
        at_ns: 3_000_000_000,
        window_found: true,
        ..Default::default()
    });
    assert!(!lifecycle.final_reasons().is_empty());
    assert_eq!(
        lifecycle.deadline(Duration::from_secs(3), 10),
        Some(("measurement window is not presentable", false))
    );
}
