use texelith_ui_egui::startup::Config;

fn parse(args: &[&str]) -> Result<Config, String> {
    Config::parse(args.iter().map(|arg| (*arg).to_owned()))
}

#[test]
fn default_shell_uses_native_wgpu_without_a_smoke_exit() {
    let config = parse(&[]).unwrap();
    assert!(!config.render_smoke);
    #[cfg(target_os = "windows")]
    assert_eq!(config.backends().unwrap(), wgpu::Backends::DX12);
    #[cfg(target_os = "macos")]
    assert_eq!(config.backends().unwrap(), wgpu::Backends::METAL);
    #[cfg(target_os = "linux")]
    assert_eq!(config.backends().unwrap(), wgpu::Backends::VULKAN);
    let options = config.native_options().unwrap();
    assert!(matches!(options.renderer, eframe::Renderer::Wgpu));
    assert_eq!(
        options.wgpu_options.surface.present_mode,
        wgpu::PresentMode::Fifo
    );
    let egui_wgpu::WgpuSetup::CreateNew(setup) = options.wgpu_options.wgpu_setup else {
        panic!("must let eframe create the device")
    };
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
    assert!(setup.native_adapter_selector.unwrap()(&[], None).is_err());
}

#[test]
fn parses_the_lifetime_acceptance_command() {
    let config = parse(&[
        "--render-smoke",
        "--backend",
        "dx12",
        "--adapter-name",
        "RTX 5070 Ti",
        "--replace-texture-every-frame",
        "--frames",
        "600",
        "--report",
        "docs/evidence/T-013/lifetime-smoke.json",
    ])
    .unwrap();
    assert!(config.render_smoke && config.replace_texture_every_frame);
    assert_eq!(config.frames, 600);
    assert_eq!(config.adapter_name.as_deref(), Some("RTX 5070 Ti"));
    assert!(config.report.unwrap().ends_with("lifetime-smoke.json"));
}

#[test]
fn rejects_invalid_or_ambiguous_diagnostics() {
    for args in [
        vec!["--frames", "0"],
        vec!["--frames", "-1"],
        vec!["--frames"],
        vec!["--backend", "glow"],
        vec!["--adapter-name", ""],
        vec!["--report"],
        vec!["--replace-texture-every-frame"],
        vec!["--typo"],
    ] {
        assert!(parse(&args).is_err(), "{args:?}");
    }
}

#[test]
fn integration_smoke_requires_real_input_and_save_destination() {
    let config = parse(&[
        "--input",
        "input.png",
        "--integration-smoke",
        "--save-normal",
        "normal.png",
    ])
    .unwrap();
    assert!(config.integration_smoke);
    for args in [
        vec!["--integration-smoke"],
        vec!["--input"],
        vec!["--save-normal", "out.png"],
        vec!["--input", "in.png", "--render-smoke"],
        vec!["--preview-height"],
    ] {
        assert!(parse(&args).is_err(), "{args:?}");
    }
    assert!(
        parse(&[
            "--input",
            "in.png",
            "--preview-height",
            "--ui-snapshot",
            "captures"
        ])
        .unwrap()
        .preview_height
    );
}
