use std::{ffi::OsString, path::PathBuf, sync::Arc};

use crate::platform;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Config {
    pub studio_snapshots: bool,
    pub studio_snapshot_size: Option<[u32; 2]>,
    pub debug_layers: bool,
    pub projects_smoke: bool,
    pub workflow_smoke: bool,
    pub projects_snapshots: bool,
    pub mvp_smoke: bool,
    pub batch_smoke: bool,
    pub export_smoke: bool,
    pub packing_smoke: bool,
    pub packing_snapshots: bool,
    pub viewport: bool,
    pub viewport_mesh: String,
    pub simple_layout_smoke: bool,
    pub inspectors_smoke: bool,
    pub input_smoke: bool,
    pub input_snapshots: bool,
    pub locale: String,
    pub render_smoke: bool,
    pub command_smoke: bool,
    pub ui_snapshot: Option<PathBuf>,
    pub scale: String,
    pub backend: String,
    pub adapter_name: Option<String>,
    pub replace_texture_every_frame: bool,
    pub frames: u64,
    pub report: Option<PathBuf>,
    pub input: Option<PathBuf>,
    pub integration_smoke: bool,
    pub save_normal: Option<PathBuf>,
    pub preview_height: bool,
    pub perf_scenario: bool,
    pub size: u32,
    pub duration_seconds: u64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            studio_snapshots: false,
            studio_snapshot_size: None,
            debug_layers: false,
            projects_smoke: false,
            workflow_smoke: false,
            projects_snapshots: false,
            mvp_smoke: false,
            batch_smoke: false,
            export_smoke: false,
            packing_smoke: false,
            packing_snapshots: false,
            viewport: false,
            viewport_mesh: "preview-ball".into(),
            simple_layout_smoke: false,
            inspectors_smoke: false,
            input_smoke: false,
            input_snapshots: false,
            locale: "en-US".into(),
            render_smoke: false,
            command_smoke: false,
            ui_snapshot: None,
            scale: "1.0".into(),
            backend: platform::default_backend().into(),
            adapter_name: None,
            replace_texture_every_frame: false,
            frames: 180,
            report: None,
            input: None,
            integration_smoke: false,
            save_normal: None,
            preview_height: false,
            perf_scenario: false,
            size: 4096,
            duration_seconds: 10,
        }
    }
}

impl Config {
    pub fn parse(args: impl IntoIterator<Item = String>) -> Result<Self, String> {
        Self::parse_os(args.into_iter().map(OsString::from))
    }

    /// Parse native arguments without converting filesystem paths to Unicode.
    /// Explicit backend selection takes precedence over WGPU_BACKEND, then the OS default.
    pub fn parse_os(args: impl IntoIterator<Item = OsString>) -> Result<Self, String> {
        let mut config = Self::default();
        let mut explicit_backend = false;
        let mut args = args.into_iter();
        while let Some(arg) = args.next() {
            let arg = arg.to_str().ok_or("argument names must be Unicode text")?;
            match arg {
                "--locale" => {
                    config.locale = next_text(&mut args, "--locale")?;
                    if !["en-US", "tr-TR"].contains(&config.locale.as_str()) {
                        return Err("--locale must be en-US or tr-TR".into());
                    }
                }
                "--viewport" => config.viewport = true,
                "--viewport-mesh" => {
                    config.viewport_mesh = next_text(&mut args, "--viewport-mesh")?;
                    if ![
                        "preview-ball",
                        "rounded-cube",
                        "plane",
                        "cube",
                        "sphere",
                        "cylinder",
                    ]
                    .contains(&config.viewport_mesh.as_str())
                    {
                        return Err("unknown viewport mesh".into());
                    }
                    config.viewport = true;
                }
                "--input-snapshots" => config.input_snapshots = true,
                "--packing-snapshots" => config.packing_snapshots = true,
                "--projects-snapshots" => config.projects_snapshots = true,
                "--mvp-smoke" => {
                    let scenario = next_text(&mut args, "--mvp-smoke")?;
                    if scenario != "maps"
                        && scenario != "simple_layout"
                        && scenario != "inspectors"
                        && scenario != "input"
                        && scenario != "viewport"
                        && scenario != "export"
                        && scenario != "packing_ui"
                        && scenario != "batch"
                        && scenario != "projects"
                        && scenario != "workflow"
                    {
                        return Err("unknown --mvp-smoke scenario".into());
                    }
                    config.export_smoke = scenario == "export";
                    config.projects_smoke = scenario == "projects";
                    config.workflow_smoke = scenario == "workflow";
                    config.batch_smoke = scenario == "batch";
                    config.packing_smoke = scenario == "packing_ui";
                    config.viewport = scenario == "viewport";
                    config.simple_layout_smoke = scenario == "simple_layout";
                    config.inspectors_smoke = scenario == "inspectors";
                    config.input_smoke = scenario == "input";
                    config.mvp_smoke = true;
                }
                "--perf-scenario" => {
                    if next_text(&mut args, "--perf-scenario")? != "normal-drag" {
                        return Err("--perf-scenario requires normal-drag".into());
                    }
                    config.perf_scenario = true;
                }
                "--size" => {
                    config.size = next_text(&mut args, "--size")?
                        .parse()
                        .map_err(|_| "invalid --size")?
                }
                "--duration-seconds" => {
                    config.duration_seconds = next_text(&mut args, "--duration-seconds")?
                        .parse()
                        .map_err(|_| "duration must be an integer")?
                }
                "--render-smoke" => config.render_smoke = true,
                "--integration-smoke" => config.integration_smoke = true,
                "--preview-height" => config.preview_height = true,
                "--input" => config.input = Some(args.next().ok_or("missing --input path")?.into()),
                "--save-normal" => {
                    config.save_normal =
                        Some(args.next().ok_or("missing --save-normal path")?.into())
                }
                "--command-smoke" => config.command_smoke = true,
                "--ui-snapshot" => {
                    config.ui_snapshot =
                        Some(args.next().ok_or("missing --ui-snapshot directory")?.into())
                }
                "--studio-snapshots" => config.studio_snapshots = true,
                "--studio-snapshot-size" => {
                    config.studio_snapshot_size =
                        Some(
                            match next_text(&mut args, "--studio-snapshot-size")?.as_str() {
                                "1600x1000" => [1600, 1000],
                                "1280x800" => [1280, 800],
                                "960x600" => [960, 600],
                                _ => return Err(
                                    "--studio-snapshot-size must be 1600x1000, 1280x800 or 960x600"
                                        .into(),
                                ),
                            },
                        );
                }
                "--debug-layers" => config.debug_layers = true,
                "--scale" => config.scale = next_text(&mut args, "--scale")?,
                "--replace-texture-every-frame" => config.replace_texture_every_frame = true,
                "--backend" => {
                    config.backend = next_text(&mut args, "--backend")?;
                    explicit_backend = true;
                }
                "--adapter-name" => {
                    let name = next_text(&mut args, "--adapter-name")?;
                    if name.trim().is_empty() {
                        return Err("adapter name must not be empty".into());
                    }
                    config.adapter_name = Some(name);
                }
                "--frames" => {
                    config.frames = next_text(&mut args, "--frames")?
                        .parse()
                        .map_err(|_| "--frames must be a positive integer")?;
                    if config.frames == 0 {
                        return Err("--frames must be positive".into());
                    }
                }
                "--report" => {
                    config.report = Some(args.next().ok_or("missing --report value")?.into());
                }
                _ => return Err(format!("unknown argument: {arg}")),
            }
        }
        if !explicit_backend && let Some(backend) = platform::backend_from_environment()? {
            config.backend = backend;
        }
        config.backends()?;
        if config.studio_snapshots && config.ui_snapshot.is_none() {
            return Err("--studio-snapshots requires --ui-snapshot".into());
        }
        if config.studio_snapshot_size.is_some() && !config.studio_snapshots {
            return Err("--studio-snapshot-size requires --studio-snapshots".into());
        }
        if config.projects_snapshots && config.ui_snapshot.is_none() {
            return Err("--projects-snapshots requires --ui-snapshot".into());
        }
        if config.mvp_smoke {
            if config.render_smoke
                || config.command_smoke
                || config.integration_smoke
                || config.perf_scenario
                || config.ui_snapshot.is_some()
                || config.input.is_some()
            {
                return Err("--mvp-smoke cannot be combined with other modes".into());
            }
            if !(1..=4096).contains(&config.size) {
                return Err("--size must be in 1..=4096".into());
            }
            if config.report.is_none() {
                return Err("--mvp-smoke requires --report".into());
            }
        }
        if config.perf_scenario {
            if config.render_smoke
                || config.command_smoke
                || config.integration_smoke
                || config.ui_snapshot.is_some()
                || config.preview_height
            {
                return Err("--perf-scenario cannot be combined with other diagnostics".into());
            }
            if config.size != 4096 || !(10..=60).contains(&config.duration_seconds) {
                return Err("normal-drag requires --size 4096 and duration 10..=60 seconds".into());
            }
            if config.report.is_none() {
                config.report = Some("target/evidence/T-020/single-run.json".into());
            }
        }
        if config.preview_height && (config.input.is_none() || config.integration_smoke) {
            return Err(
                "--preview-height requires --input and cannot be used with --integration-smoke"
                    .into(),
            );
        }
        if config.integration_smoke && (config.input.is_none() || config.save_normal.is_none()) {
            return Err("--integration-smoke requires --input and --save-normal".into());
        }
        if config.integration_smoke
            && (config.render_smoke || config.command_smoke || config.ui_snapshot.is_some())
        {
            return Err(
                "--integration-smoke cannot be combined with other diagnostic modes".into(),
            );
        }
        if config.input.is_some() && (config.render_smoke || config.command_smoke) {
            return Err("--input cannot be combined with synthetic diagnostic modes".into());
        }
        if config.save_normal.is_some() && !config.integration_smoke {
            return Err("--save-normal requires --integration-smoke".into());
        }
        if !["1.0", "1.5"].contains(&config.scale.as_str()) {
            return Err("--scale must be 1.0 or 1.5".into());
        }
        if config.ui_snapshot.is_some() && config.render_smoke {
            return Err("--ui-snapshot cannot be combined with --render-smoke".into());
        }
        if config.input_snapshots && (config.ui_snapshot.is_none() || config.input.is_none()) {
            return Err("--input-snapshots requires --ui-snapshot and --input".into());
        }
        if config.packing_snapshots && (config.ui_snapshot.is_none() || config.input.is_none()) {
            return Err("--packing-snapshots requires --ui-snapshot and --input".into());
        }
        if config.replace_texture_every_frame && !config.render_smoke {
            return Err("--replace-texture-every-frame requires --render-smoke".into());
        }
        Ok(config)
    }

    pub fn backends(&self) -> Result<wgpu::Backends, String> {
        match self.backend.as_str() {
            "dx12" => Ok(wgpu::Backends::DX12),
            "vulkan" => Ok(wgpu::Backends::VULKAN),
            "metal" => Ok(wgpu::Backends::METAL),
            _ => Err(format!(
                "unsupported backend: {} (use dx12, vulkan, metal)",
                self.backend
            )),
        }
    }

    pub fn native_options(&self) -> Result<eframe::NativeOptions, String> {
        self.native_options_measured(None)
    }

    pub(crate) fn native_options_measured(
        &self,
        measurement: Option<crate::measurement::Shared>,
    ) -> Result<eframe::NativeOptions, String> {
        let mut setup = egui_wgpu::WgpuSetupCreateNew::without_display_handle();
        setup.instance_descriptor.backends = self.backends()?;
        setup
            .instance_descriptor
            .flags
            .insert(wgpu::InstanceFlags::VALIDATION);
        setup.power_preference = wgpu::PowerPreference::HighPerformance;
        let requested_name = self.adapter_name.clone();
        let selector_measurement = measurement.clone();
        setup.native_adapter_selector =
            Some(Arc::new(move |adapters, surface| {
                let mut candidates: Vec<_> = adapters
                    .iter()
                    .filter(|adapter| {
                        let info = adapter.get_info();
                        requested_name.as_ref().is_none_or(|name| {
                            info.name.to_lowercase().contains(&name.to_lowercase())
                        }) && surface.is_none_or(|surface| adapter.is_surface_supported(surface))
                    })
                    .collect();
                candidates.sort_by_key(|adapter| match adapter.get_info().device_type {
                    wgpu::DeviceType::DiscreteGpu => 0,
                    wgpu::DeviceType::IntegratedGpu => 1,
                    _ => 2,
                });
                let adapter = candidates.first().ok_or_else(|| format!(
                "no surface-compatible adapter matching {requested_name:?}; available: {:?}",
                adapters.iter().map(|a| a.get_info()).collect::<Vec<_>>()
            ))?;
                validate_adapter(adapter)?;
                if let Some(monitor) = &selector_measurement {
                    monitor.phase("adapter_selected");
                }
                Ok((*adapter).clone())
            }));
        let diagnostic = self.perf_scenario;
        setup.device_descriptor = Arc::new(move |adapter| {
            if let Some(monitor) = &measurement {
                monitor.phase("device_requested");
            }
            wgpu::DeviceDescriptor {
                label: Some("Texelith shared eframe device"),
                required_limits: wgpu::Limits {
                    max_texture_dimension_2d: 4096,
                    ..wgpu::Limits::default()
                },
                required_features: adapter.features()
                    & (wgpu::Features::TIMESTAMP_QUERY
                        | if diagnostic {
                            wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS
                        } else {
                            wgpu::Features::empty()
                        }),
                ..Default::default()
            }
        });
        let material_shell = !self.render_smoke
            && !self.command_smoke
            && !self.integration_smoke
            && !self.perf_scenario;
        Ok(eframe::NativeOptions {
            renderer: eframe::Renderer::Wgpu,
            viewport: egui::ViewportBuilder::default()
                .with_title("Texelith")
                .with_inner_size(if let Some([width, height]) = self.studio_snapshot_size {
                    [width as f32, height as f32]
                } else if self.studio_snapshots {
                    [1600.0, 1000.0]
                } else if material_shell {
                    [1280.0, 800.0]
                } else {
                    [1000.0, 720.0]
                })
                .with_min_inner_size(if material_shell && self.studio_snapshot_size.is_none() {
                    [960.0, 600.0]
                } else {
                    [560.0, 420.0]
                })
                .with_resizable(true),
            wgpu_options: egui_wgpu::WgpuConfiguration {
                wgpu_setup: setup.into(),
                surface: egui_wgpu::SurfaceConfig {
                    present_mode: wgpu::PresentMode::Fifo,
                    desired_maximum_frame_latency: Some(2),
                },
                ..Default::default()
            },
            centered: true,
            dithering: false,
            ..Default::default()
        })
    }
}

fn next_text(args: &mut impl Iterator<Item = OsString>, flag: &str) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("missing {flag} value"))?
        .into_string()
        .map_err(|_| format!("{flag} requires Unicode text"))
}

pub(crate) fn validate_adapter(adapter: &wgpu::Adapter) -> Result<(), String> {
    if adapter.get_info().device_type == wgpu::DeviceType::Cpu {
        return Err("software/CPU adapters are unsupported; a native GPU is required".into());
    }
    if adapter.limits().max_texture_dimension_2d < 4096 {
        return Err("adapter cannot support required 4096 texture dimensions".into());
    }
    for format in [
        wgpu::TextureFormat::R32Float,
        wgpu::TextureFormat::Rgba16Float,
    ] {
        let required = wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC;
        let support = adapter.get_texture_format_features(format);
        if !support.allowed_usages.contains(required)
            || !support
                .flags
                .contains(wgpu::TextureFormatFeatureFlags::STORAGE_WRITE_ONLY)
        {
            return Err(format!(
                "required storage/sample/copy format unsupported: {format:?}: {support:?}"
            ));
        }
    }
    let support = adapter.get_texture_format_features(wgpu::TextureFormat::Rgba8Unorm);
    if !support
        .allowed_usages
        .contains(wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST)
        || !support
            .flags
            .contains(wgpu::TextureFormatFeatureFlags::FILTERABLE)
    {
        return Err(format!(
            "required linear-filtered Rgba8Unorm display unsupported: {support:?}"
        ));
    }
    Ok(())
}
