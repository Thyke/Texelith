//! Headless numerical tests of both the independent ABI driver and production executor.
#[path = "support/harness.rs"]
mod harness;
#[path = "../tests/support/mod.rs"]
mod reference;

use harness::{Harness, Image, normals, scalar};
use reference::{
    abi::{BlurWeights, NodeUniform},
    oracle::*,
};
use serde_json::{Value, json};
use std::{path::PathBuf, sync::Arc, time::Instant};
use texelith_core::{Extent, InputInterpretation, SourceDescriptor};

struct Config {
    backend: String,
    adapter: String,
    suite: String,
    report: PathBuf,
}

impl Config {
    fn parse() -> Result<Self, String> {
        let mut config = Self {
            backend: "dx12".into(),
            adapter: "RTX 5070 Ti".into(),
            suite: "all".into(),
            report: "target/evidence/T-015/numerics.json".into(),
        };
        let mut args = std::env::args().skip(1);
        while let Some(key) = args.next() {
            let value = args
                .next()
                .ok_or_else(|| format!("missing value for {key}"))?;
            match key.as_str() {
                "--backend" => config.backend = value,
                "--adapter-name" => config.adapter = value,
                "--suite" => config.suite = value,
                "--report" => config.report = value.into(),
                _ => return Err(format!("unknown argument {key}")),
            }
        }
        if config.adapter.trim().is_empty() {
            return Err("adapter name cannot be empty".into());
        }
        if !["all", "4k"].contains(&config.suite.as_str()) {
            return Err("suite must be all or 4k".into());
        }
        config.backends()?;
        Ok(config)
    }

    fn backends(&self) -> Result<wgpu::Backends, String> {
        match self.backend.as_str() {
            "dx12" => Ok(wgpu::Backends::DX12),
            "vulkan" => Ok(wgpu::Backends::VULKAN),
            "metal" => Ok(wgpu::Backends::METAL),
            _ => Err("backend must be dx12, vulkan or metal".into()),
        }
    }
}

fn check(
    report: &mut Vec<Value>,
    name: &str,
    actual: &[f64],
    expected: &[f64],
    tolerance: f64,
) -> Result<(), String> {
    let result = compare(actual, expected, tolerance);
    report.push(json!({ "name": name, "passed": result.is_ok(), "tolerance": tolerance, "max_abs_error": result.as_ref().ok(), "error": result.as_ref().err() }));
    result.map(|_| ()).map_err(|e| format!("{name}: {e}"))
}

fn blur_pass(
    h: &mut Harness,
    input: &Image,
    source: Extent,
    sigma: f64,
    vertical: bool,
) -> Result<(Image, Vec<u8>), String> {
    let axis = if vertical {
        (input.extent.height, source.height)
    } else {
        (input.extent.width, source.width)
    };
    let sigma_axis = sigma * f64::from(axis.0) / f64::from(axis.1);
    let weights = gaussian(sigma_axis);
    let mut u = NodeUniform::new(input.extent, source);
    u.parameters[0] = sigma_axis as f32;
    u.flags = [u32::from(vertical), weights.len() as u32 / 2, 0, 0];
    h.run_blur(input, u, BlurWeights::new(&weights), sigma as f32, vertical)
}

fn check_normals(
    report: &mut Vec<Value>,
    name: &str,
    bytes: &[u8],
    expected: &[[f64; 4]],
) -> Result<(), String> {
    let actual = normals(bytes);
    for channel in 0..4 {
        check(
            report,
            &format!("{name}/channel_{channel}"),
            &actual.iter().map(|n| n[channel]).collect::<Vec<_>>(),
            &expected.iter().map(|n| n[channel]).collect::<Vec<_>>(),
            if channel == 3 { 0.0 } else { NORMAL_TOLERANCE },
        )?;
    }
    check(
        report,
        &format!("{name}/unit_length"),
        &actual
            .iter()
            .map(|n| n[..3].iter().map(|v| v * v).sum::<f64>().sqrt())
            .collect::<Vec<_>>(),
        &vec![1.0; actual.len()],
        LENGTH_TOLERANCE,
    )
}

fn check_display(
    report: &mut Vec<Value>,
    name: &str,
    bytes: &[u8],
    expected: &[[u8; 4]],
) -> Result<(), String> {
    for channel in 0..4 {
        check(
            report,
            &format!("{name}/channel_{channel}"),
            &bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|p| f64::from(p[channel]))
                .collect::<Vec<_>>(),
            &expected
                .iter()
                .map(|p| f64::from(p[channel]))
                .collect::<Vec<_>>(),
            if channel == 3 {
                0.0
            } else {
                DISPLAY_CODE_TOLERANCE
            },
        )?;
    }
    Ok(())
}

fn numerical_suite(h: &mut Harness, report: &mut Vec<Value>) -> Result<(), String> {
    for fixture in fixtures() {
        let input = h.upload_height(&fixture.height)?;
        let e = input.extent;
        for (black, white, gamma) in [
            (0.0, 1.0, 1.0),
            (0.25, 0.75, 2.0),
            (0.0, 1.0, 0.1),
            (0.0, 1.0, 10.0),
        ] {
            let mut u = NodeUniform::new(e, e);
            u.parameters = [black, white, gamma, 0.0];
            let (_, bytes) = h.run("levels", &input, u, None)?;
            let expected = levels(
                &fixture.height,
                f64::from(black),
                f64::from(white),
                f64::from(gamma),
            );
            check(
                report,
                &format!("{}/levels/{gamma}", fixture.name),
                &scalar(&bytes),
                &expected.pixels,
                HEIGHT_TOLERANCE,
            )?;
        }
        for sigma in [0.0, 0.25, 1.0, 5.0] {
            let (horizontal, xbytes) = blur_pass(h, &input, e, sigma, false)?;
            let (_, ybytes) = blur_pass(h, &horizontal, e, sigma, true)?;
            let xexpected = blur_axis(&fixture.height, sigma, false);
            let expected = blur(&fixture.height, e, sigma);
            check(
                report,
                &format!("{}/blur_x/{sigma}", fixture.name),
                &scalar(&xbytes),
                &xexpected.pixels,
                HEIGHT_TOLERANCE,
            )?;
            check(
                report,
                &format!("{}/blur_xy/{sigma}", fixture.name),
                &scalar(&ybytes),
                &expected.pixels,
                HEIGHT_TOLERANCE,
            )?;
            if fixture.name == "tileable_symmetric" {
                let score = wrapped_edge_score(&Height::new(e, scalar(&ybytes)));
                check(
                    report,
                    &format!("tileable_height_seam/{sigma}"),
                    &[score],
                    &[0.0],
                    HEIGHT_TOLERANCE,
                )?;
            }
        }
        for strength in [0.0, 1.0, 10.0] {
            let mut u = NodeUniform::new(e, e);
            u.parameters[0] = strength as f32;
            let (normal_image, bytes) = h.run("normal", &input, u, None)?;
            let expected = normal(&fixture.height, strength);
            check_normals(
                report,
                &format!("{}/normal/{strength}", fixture.name),
                &bytes,
                &expected,
            )?;
            let mut display_u = NodeUniform::new(e, e);
            display_u.flags[0] = 1;
            let (_, display) = h.run("display", &normal_image, display_u, None)?;
            check_display(
                report,
                &format!("{}/normal_display/{strength}", fixture.name),
                &display,
                &display_normal(&expected),
            )?;
            let (_, repeat) = h.run("normal", &input, u, None)?;
            let exact = bytes == repeat;
            report.push(json!({ "name": format!("{}/normal_repeat/{strength}", fixture.name), "passed": exact, "comparison": "exact output bytes" }));
            if !exact {
                return Err("normal output was nondeterministic".into());
            }
        }
        let (_, display) = h.run("display", &input, NodeUniform::new(e, e), None)?;
        check_display(
            report,
            &format!("{}/height_display", fixture.name),
            &display,
            &display_height(&fixture.height),
        )?;
        let (_, repeat) = h.run("display", &input, NodeUniform::new(e, e), None)?;
        let first = SourceDescriptor::from_rgba8(&display, e, InputInterpretation::LinearHeight)
            .map_err(|e| e.to_string())?;
        let second = SourceDescriptor::from_rgba8(&repeat, e, InputInterpretation::LinearHeight)
            .map_err(|e| e.to_string())?;
        report.push(json!({ "name": format!("{}/display_hash_repeat", fixture.name), "passed": first.content_hash == second.content_hash, "blake3": first.content_hash }));
        if display != repeat || first.content_hash != second.content_hash {
            return Err("display output/hash was nondeterministic".into());
        }
    }
    source_suite(h, report)?;
    proxy_suite(h, report)?;
    chain_suite(h, report)?;
    // Closely spaced heights expose an accidental R8/half-float intermediate.
    let dense = Height::generate(Extent::new(256, 1).unwrap(), |x, _| {
        0.4 + f64::from(x) * 1e-5
    });
    let input = h.upload_height(&dense)?;
    let mut u = NodeUniform::new(dense.extent, dense.extent);
    u.parameters = [0.0, 1.0, 1.0, 0.0];
    let (_, bytes) = h.run("levels", &input, u, None)?;
    let values = scalar(&bytes);
    check(
        report,
        "height_precision",
        &values,
        &dense.pixels,
        HEIGHT_TOLERANCE,
    )?;
    let increasing = values.windows(2).all(|p| p[1] > p[0]);
    report.push(json!({ "name": "height_no_banding", "passed": increasing }));
    if !increasing {
        return Err("height precision collapsed neighboring values".into());
    }
    Ok(())
}

fn source_suite(h: &mut Harness, report: &mut Vec<Value>) -> Result<(), String> {
    let e = Extent::new(9, 1).unwrap();
    let rgba = [
        0, 0, 0, 0, 255, 255, 255, 255, 128, 128, 128, 0, 255, 0, 0, 255, 0, 255, 0, 0, 10, 10, 10,
        255, 11, 11, 11, 0, 0, 0, 255, 255, 128, 64, 192, 0,
    ];
    for interpretation in [
        InputInterpretation::SrgbPhoto,
        InputInterpretation::LinearHeight,
    ] {
        let input = h.upload(e, wgpu::TextureFormat::Rgba8Unorm, &rgba)?;
        let mut u = NodeUniform::new(e, e);
        u.flags[0] = u32::from(interpretation == InputInterpretation::LinearHeight);
        let (prepared, bytes) = h.run("source", &input, u, None)?;
        let expected = prepare(&rgba, e, interpretation);
        check(
            report,
            &format!("source/{interpretation:?}"),
            &scalar(&bytes),
            &expected.pixels,
            HEIGHT_TOLERANCE,
        )?;
        let mut opaque = rgba;
        for p in opaque.as_chunks_mut::<4>().0 {
            p[3] = 255;
        }
        let opaque = h.upload(e, wgpu::TextureFormat::Rgba8Unorm, &opaque)?;
        let (_, obytes) = h.run("source", &opaque, u, None)?;
        check(
            report,
            &format!("source/ignore_alpha/{interpretation:?}"),
            &scalar(&obytes),
            &scalar(&bytes),
            0.0,
        )?;
        let out = Extent::new(3, 1).unwrap();
        let (_, down) = h.run("downsample", &prepared, NodeUniform::new(out, e), None)?;
        check(
            report,
            &format!("source/linear_area/{interpretation:?}"),
            &scalar(&down),
            &downsample(&expected, out).pixels,
            HEIGHT_TOLERANCE,
        )?;
    }
    for e in [
        Extent::new(5, 3).unwrap(),
        Extent::new(1, 5).unwrap(),
        Extent::new(5, 1).unwrap(),
    ] {
        let height = Height::generate(e, |x, y| f64::from(x + 2 * y) / 10.0);
        let out = Extent::new(e.width.div_ceil(3), e.height.div_ceil(3)).unwrap();
        let input = h.upload_height(&height)?;
        let (_, bytes) = h.run("downsample", &input, NodeUniform::new(out, e), None)?;
        check(
            report,
            &format!("area/{e:?}"),
            &scalar(&bytes),
            &downsample(&height, out).pixels,
            HEIGHT_TOLERANCE,
        )?;
    }
    Ok(())
}

fn proxy_suite(h: &mut Harness, report: &mut Vec<Value>) -> Result<(), String> {
    // Actual 1024 limit, generated in memory; analytic low frequency UV slopes.
    let e = Extent::new(2048, 32).unwrap();
    let full = Height::generate(e, |x, y| {
        0.1 + 0.25 * (f64::from(x) + 0.5) / f64::from(e.width)
            + 0.125 * (f64::from(y) + 0.5) / f64::from(e.height)
    });
    let input = h.upload_height(&full)?;
    let p = if h.production() {
        texelith_gpu::proxy_extent(e).map_err(|e| e.to_string())?
    } else {
        proxy_extent(e)
    };
    let (proxy, bytes) = h.run("downsample", &input, NodeUniform::new(p, e), None)?;
    let expected_proxy = downsample(&full, p);
    check(
        report,
        "proxy/area",
        &scalar(&bytes),
        &expected_proxy.pixels,
        HEIGHT_TOLERANCE,
    )?;
    let mut center_normals = Vec::new();
    for (name, image, expected) in [("full", &input, &full), ("proxy", &proxy, &expected_proxy)] {
        let (bx, _) = blur_pass(h, image, e, 1.0, false)?;
        let (by, bytes) = blur_pass(h, &bx, e, 1.0, true)?;
        let reference = blur(expected, e, 1.0);
        check(
            report,
            &format!("proxy/{name}/scaled_blur"),
            &scalar(&bytes),
            &reference.pixels,
            HEIGHT_TOLERANCE,
        )?;
        let mut u = NodeUniform::new(image.extent, e);
        u.parameters[0] = 3.0;
        let (_, bytes) = h.run("normal", &by, u, None)?;
        check_normals(
            report,
            &format!("proxy/{name}/normal"),
            &bytes,
            &normal(&reference, 3.0),
        )?;
        let center =
            (image.extent.height / 2 * image.extent.width + image.extent.width / 2) as usize;
        center_normals.push(normals(&bytes)[center]);
    }
    check(
        report,
        "proxy/uv_slope_matches_full",
        &center_normals[0],
        &center_normals[1],
        NORMAL_TOLERANCE,
    )
}

fn chain_suite(h: &mut Harness, report: &mut Vec<Value>) -> Result<(), String> {
    let source = Extent::new(34, 44).unwrap();
    let evaluation = Extent::new(17, 11).unwrap();
    let rgba: Vec<u8> = (0..source.height)
        .flat_map(|y| {
            (0..source.width).flat_map(move |x| {
                [
                    ((x * 7 + y * 3) % 256) as u8,
                    (x * 5 % 256) as u8,
                    (y * 3 % 256) as u8,
                    0,
                ]
            })
        })
        .collect();
    let input = h.upload(source, wgpu::TextureFormat::Rgba8Unorm, &rgba)?;
    let (prepared, _) = h.run("source", &input, NodeUniform::new(source, source), None)?;
    let reference_source = prepare(&rgba, source, InputInterpretation::SrgbPhoto);
    let (proxy, _) = h.run(
        "downsample",
        &prepared,
        NodeUniform::new(evaluation, source),
        None,
    )?;
    let reference_proxy = downsample(&reference_source, evaluation);
    let mut u = NodeUniform::new(evaluation, source);
    u.parameters = [0.1, 0.9, 1.25, 0.0];
    let (adjusted, _) = h.run("levels", &proxy, u, None)?;
    let reference_adjusted = levels(&reference_proxy, f64::from(0.1f32), f64::from(0.9f32), 1.25);
    // X sigma is 1, Y sigma is 0.5. Using one scale for both axes fails here.
    let (bx, xbytes) = blur_pass(h, &adjusted, source, 2.0, false)?;
    check(
        report,
        "chain/anisotropic_blur_x",
        &scalar(&xbytes),
        &blur_axis(&reference_adjusted, 1.0, false).pixels,
        HEIGHT_TOLERANCE,
    )?;
    let (by, ybytes) = blur_pass(h, &bx, source, 2.0, true)?;
    let reference_blurred = blur(&reference_adjusted, source, 2.0);
    check(
        report,
        "chain/anisotropic_blur_xy",
        &scalar(&ybytes),
        &reference_blurred.pixels,
        HEIGHT_TOLERANCE,
    )?;
    let mut u = NodeUniform::new(evaluation, source);
    u.parameters[0] = 2.0;
    let (normal_image, bytes) = h.run("normal", &by, u, None)?;
    let reference_normal = normal(&reference_blurred, 2.0);
    check_normals(report, "chain/normal", &bytes, &reference_normal)?;
    let mut u = NodeUniform::new(evaluation, source);
    u.flags[0] = 1;
    let (_, bytes) = h.run("display", &normal_image, u, None)?;
    check_display(
        report,
        "chain/display",
        &bytes,
        &display_normal(&reference_normal),
    )
}

fn four_k_suite(h: &mut Harness, report: &mut Vec<Value>) -> Result<(), String> {
    let started = Instant::now();
    let e = Extent::new(4096, 4096).unwrap();
    let rgba: Vec<u8> = (0..e.height)
        .flat_map(|y| {
            (0..e.width).flat_map(move |x| {
                let value = ((x / 16 + y / 16) % 256) as u8;
                [value, value.wrapping_add(17), 128, 255]
            })
        })
        .collect();
    let input = h.upload(e, wgpu::TextureFormat::Rgba8Unorm, &rgba)?;
    let mut u = NodeUniform::new(e, e);
    u.flags[0] = 1;
    let (height, _) = h.run("source", &input, u, None)?;
    let mut u = NodeUniform::new(e, e);
    u.parameters = [0.1, 0.9, 1.25, 0.0];
    let (adjusted, _) = h.run("levels", &height, u, None)?;
    let (bx, _) = blur_pass(h, &adjusted, e, 1.0, false)?;
    let (by, bytes) = blur_pass(h, &bx, e, 1.0, true)?;
    let values = scalar(&bytes);
    if values.len() != 4096 * 4096
        || values
            .iter()
            .any(|v| !v.is_finite() || !(0.0..=1.00002).contains(v))
    {
        return Err("invalid 4K scalar chain result".into());
    }
    let mut u = NodeUniform::new(e, e);
    u.parameters[0] = 2.0;
    let (normal_image, bytes) = h.run("normal", &by, u, None)?;
    let decoded = normals(&bytes);
    let lengths: Vec<_> = decoded
        .iter()
        .map(|n| n[..3].iter().map(|c| c * c).sum::<f64>().sqrt())
        .collect();
    check(
        report,
        "4k/normal_unit_length",
        &lengths,
        &vec![1.0; lengths.len()],
        LENGTH_TOLERANCE,
    )?;
    if decoded
        .iter()
        .any(|n| n.iter().any(|v| !v.is_finite()) || n[3] != 1.0)
    {
        return Err("invalid 4K normals".into());
    }
    let mut u = NodeUniform::new(e, e);
    u.flags[0] = 1;
    let (_, bytes) = h.run("display", &normal_image, u, None)?;
    if bytes.len() != 4096 * 4096 * 4 || bytes.as_chunks::<4>().0.iter().any(|p| p[3] != 255) {
        return Err("invalid 4K display conversion".into());
    }
    report.push(json!({ "name": "4k/full_chain", "passed": true, "extent": [4096,4096], "wall_ms_with_test_readbacks": started.elapsed().as_secs_f64() * 1000.0, "ui_performance_gate": false }));
    Ok(())
}

fn execute(config: &Config, report: &mut Value) -> Result<(), String> {
    let backends = config.backends()?;
    let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    descriptor.backends = backends;
    descriptor.flags.insert(wgpu::InstanceFlags::VALIDATION);
    let instance = wgpu::Instance::new(descriptor);
    let adapters = pollster::block_on(instance.enumerate_adapters(backends));
    report["available_adapters"] = json!(
        adapters
            .iter()
            .map(|a| format!("{:?}", a.get_info()))
            .collect::<Vec<_>>()
    );
    let adapter = adapters
        .into_iter()
        .find(|a| {
            let info = a.get_info();
            info.device_type != wgpu::DeviceType::Cpu
                && info
                    .name
                    .to_lowercase()
                    .contains(&config.adapter.to_lowercase())
        })
        .ok_or("requested native GPU adapter not found; this check fails rather than skips")?;
    let info = adapter.get_info();
    report["adapter"] = json!({ "name": info.name, "backend": format!("{:?}", info.backend), "driver": info.driver, "driver_info": info.driver_info, "vendor": info.vendor, "device": info.device, "limits": format!("{:?}", adapter.limits()), "timestamp_supported": adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY) });
    if adapter.limits().max_texture_dimension_2d < 4096 {
        return Err("adapter lacks 4096 dimension support".into());
    }
    let mut formats = Vec::new();
    for format in [
        wgpu::TextureFormat::R32Float,
        wgpu::TextureFormat::Rgba16Float,
        wgpu::TextureFormat::Rgba8Unorm,
    ] {
        let support = adapter.get_texture_format_features(format);
        let required = wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::COPY_DST;
        let supported = support.allowed_usages.contains(required)
            && support
                .flags
                .contains(wgpu::TextureFormatFeatureFlags::STORAGE_WRITE_ONLY);
        formats.push(json!({ "format": format!("{format:?}"), "features": format!("{support:?}"), "passed": supported }));
        report["formats"] = json!(formats);
        if !supported {
            return Err(format!("unsupported required format {format:?}"));
        }
    }
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("T-015 headless verifier only"),
        required_limits: wgpu::Limits {
            max_texture_dimension_2d: 4096,
            ..Default::default()
        },
        ..Default::default()
    }))
    .map_err(|e| e.to_string())?;
    let mut h = Harness::new(Arc::new(device), Arc::new(queue));
    let mut checks = Vec::new();
    let result = h.compile().and_then(|()| {
        for production in [false, true] {
            if production {
                h.enable_production()?;
                h.verify_rejections(&mut checks)?;
            }
            let first_check = checks.len();
            let result = if config.suite == "all" {
                numerical_suite(&mut h, &mut checks)
            } else {
                four_k_suite(&mut h, &mut checks)
            };
            for check in &mut checks[first_check..] {
                check["driver"] = json!(if production {
                    "production"
                } else {
                    "independent"
                });
            }
            result?;
        }
        report["production_executor_verified"] = json!(true);
        Ok(())
    });
    report["checks"] = json!(checks);
    report["gpu"] = h.report();
    result?;
    h.check_errors()?;
    if report["gpu"]["live_payload_bytes"].as_u64() != Some(0) {
        return Err("test resource accounting did not drain".into());
    }
    Ok(())
}

fn main() {
    let config = match Config::parse() {
        Ok(config) => config,
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(2);
        }
    };
    let mut report = json!({ "passed": false, "role": "implementer", "suite": config.suite, "requested_backend": config.backend, "requested_adapter": config.adapter, "profile": if cfg!(debug_assertions) { "debug" } else { "release" }, "worktree": env!("CARGO_MANIFEST_DIR"), "production_executor_verified": false, "scope": "independent shader ABI driver and production shared-device prepare/update/encode; same oracle assertions on both paths", "timing_kind": "CPU wall and submission-to-mapped upper bounds including copies; no GPU timestamps or UI gate", "tolerances": { "height_abs": HEIGHT_TOLERANCE, "normal_component_abs": NORMAL_TOLERANCE, "normal_length_abs": LENGTH_TOLERANCE, "display_code_abs": DISPLAY_CODE_TOLERANCE } });
    let outcome = execute(&config, &mut report);
    report["passed"] = json!(outcome.is_ok());
    if let Err(error) = &outcome {
        report["error"] = json!(error);
        eprintln!("FAIL: {error}");
    }
    let write = (|| -> Result<(), Box<dyn std::error::Error>> {
        if let Some(parent) = config.report.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&config.report, serde_json::to_vec_pretty(&report)?)?;
        Ok(())
    })();
    if let Err(error) = write {
        eprintln!("cannot write report: {error}");
        std::process::exit(1);
    }
    println!(
        "{}: {}",
        if outcome.is_ok() { "PASS" } else { "FAIL" },
        config.report.display()
    );
    if outcome.is_err() {
        std::process::exit(1);
    }
}
