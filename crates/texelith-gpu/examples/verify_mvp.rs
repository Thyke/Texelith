#[path = "mvp_support/contract.rs"]
mod contract;
#[allow(dead_code)]
#[path = "support/harness.rs"]
mod harness;
#[path = "mvp_support/pyramid.rs"]
mod pyramid;
#[path = "../tests/support/mod.rs"]
mod reference;
#[path = "mvp_support/resources.rs"]
mod resources;
// FAZ1_SUITE_MODULES: T-063 viewport target/lease behavior
#[path = "mvp_support/t063/mod.rs"]
mod viewport_resources;
// FAZ1_SUITE_MODULES: T-065 begin
#[path = "mvp_support/t065/mod.rs"]
mod pbr;
// FAZ1_SUITE_MODULES: T-065 end
// FAZ1_SUITE_MODULES: T-079 begin
#[path = "mvp_support/t079/mod.rs"]
mod readback;
// FAZ1_SUITE_MODULES: T-079 end
// FAZ1_SUITE_MODULES: T-067 begin
#[path = "mvp_support/t067/mod.rs"]
mod ibl;
// FAZ1_SUITE_MODULES: T-067 end
// FAZ1_SUITE_MODULES: T-081 begin
#[path = "mvp_support/t081/mod.rs"]
mod bc5;
// FAZ1_SUITE_MODULES: T-081 end
// FAZ1_SUITE_MODULES: T-068 begin
#[path = "mvp_support/t068/mod.rs"]
mod pom;
// FAZ1_SUITE_MODULES: T-068 end
// FAZ1_SUITE_MODULES: T-069
#[path = "mvp_support/t069/mod.rs"]
mod compare;
macro_rules! register_suites {
    ($( $visibility:vis $module:ident { $($export:ident),* } [$($state:ident, $runtime:literal)?] [$($suite:literal, $runner:literal, [$($validator:ident, $validation:literal)?])?]; )*) => {
        $($(#[path = $runner] mod $module;) ?)*
        const NODE_SUITES: &[(&str, SuiteRunner)] = &[
            $($(($suite, $module::run_registered),) ?)*
        ];
    };
}
texelith_gpu::gpu_nodes!(register_suites);
type SuiteRunner = fn(&SuiteContext<'_>) -> Result<serde_json::Value, String>;
struct SuiteContext<'a> {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    extent: Extent,
    info: &'a wgpu::AdapterInfo,
    adapter_name: &'a str,
    backend: &'a str,
}
use contract::*;
use serde_json::json;
use std::{
    sync::Arc,
    time::{Duration, Instant},
};
use texelith_core::{Extent, InputInterpretation, SourceDescriptor};
use texelith_gpu::{GpuExecutor, Pass};

fn run() -> Result<(), String> {
    let mut suite = String::new();
    let mut size = 0u32;
    let mut backend = String::new();
    let mut name = String::new();
    let mut report = String::new();
    let mut args = std::env::args().skip(1);
    while let Some(k) = args.next() {
        let v = args.next().ok_or("missing argument value")?;
        match k.as_str() {
            "--suite" => suite = v,
            "--size" => size = v.parse().map_err(|_| "invalid size")?,
            "--backend" => backend = v,
            "--adapter-name" => name = v,
            "--report" => report = v,
            _ => return Err(format!("unknown argument {k}")),
        }
    }
    let selected = select_suite(&suite)?;
    let extent = Extent::new(size, size).map_err(|e| e.to_string())?;
    if report.is_empty() {
        return Err("missing report path".into());
    }
    let backends = match backend.as_str() {
        "dx12" => wgpu::Backends::DX12,
        "vulkan" => wgpu::Backends::VULKAN,
        "metal" => wgpu::Backends::METAL,
        _ => return Err("invalid backend".into()),
    };
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
    desc.backends = backends;
    let instance = wgpu::Instance::new(desc);
    let adapters = pollster::block_on(instance.enumerate_adapters(backends));
    let infos: Vec<_> = adapters.iter().map(|a| a.get_info()).collect();
    let labels: Vec<_> = infos
        .iter()
        .map(|i| {
            (
                i.name.as_str(),
                match i.backend {
                    wgpu::Backend::Dx12 => "dx12",
                    wgpu::Backend::Vulkan => "vulkan",
                    wgpu::Backend::Metal => "metal",
                    _ => "other",
                },
            )
        })
        .collect();
    let index = select_adapter(&labels, &name, &backend)?;
    let adapter = &adapters[index];
    let info = &infos[index];
    if info.device_type == wgpu::DeviceType::Cpu {
        return Err("CPU adapter is not a native GPU".into());
    }
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_limits: wgpu::Limits {
            max_texture_dimension_2d: 4096,
            ..Default::default()
        },
        ..Default::default()
    }))
    .map_err(|e| e.to_string())?;
    let device = Arc::new(device);
    let queue = Arc::new(queue);
    let mut reports = Vec::new();
    for selected_suite in selected {
        let context = SuiteContext {
            device: device.clone(),
            queue: queue.clone(),
            extent,
            info,
            adapter_name: &name,
            backend: &backend,
        };
        let result = match selected_suite {
            // FAZ1_SUITE_RUNNERS: T-069
            "compare" => compare::run_registered(&context)?,
            // FAZ1_SUITE_RUNNERS: T-079 begin
            "readback" => readback::run_registered(&context)?,
            // FAZ1_SUITE_RUNNERS: T-079 end
            // FAZ1_SUITE_RUNNERS: T-081 begin
            "bc5" => bc5::run_registered(&context)?,
            // FAZ1_SUITE_RUNNERS: T-081 end
            // FAZ1_SUITE_RUNNERS: T-063 shared-device clear/ramp path
            "viewport_resources" => viewport_resources::run_registered(&context)?,
            // FAZ1_SUITE_RUNNERS: T-065 begin
            "pbr" => pbr::run_registered(&context)?,
            // FAZ1_SUITE_RUNNERS: T-065 end
            // FAZ1_SUITE_RUNNERS: T-067 begin
            "ibl" => ibl::run_registered(&context)?,
            // FAZ1_SUITE_RUNNERS: T-067 end
            // FAZ1_SUITE_RUNNERS: T-068 begin
            "pom" => pom::run_registered(&context)?,
            // FAZ1_SUITE_RUNNERS: T-068 end
            "resources" => resources::run(device.clone(), queue.clone(), extent, info)?,
            "pyramid" => pyramid::run(device.clone(), queue.clone(), extent, info)?,
            other => {
                let (_, runner) = NODE_SUITES
                    .iter()
                    .find(|(name, _)| *name == other)
                    .ok_or_else(|| format!("registered suite has no runner: {other}"))?;
                runner(&context)?
            }
        };
        reports.push(result);
    }

    let failure = reports
        .iter()
        .find(|r| r["passed"] != true)
        .map(|r| r["failure"].as_str().unwrap_or("suite failed").to_string());
    let result = if reports.len() == 1 {
        reports.remove(0)
    } else {
        json!({"passed":failure.is_none(),"suite":suite,"reports":reports})
    };
    let path = std::path::Path::new(&report);
    if let Some(p) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(p).map_err(|e| e.to_string())?;
    }
    std::fs::write(
        path,
        serde_json::to_vec_pretty(&result).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    if let Some(error) = failure {
        return Err(format!("{error}; report: {report}"));
    }
    println!("PASS: {report}");
    Ok(())
}
fn baseline(
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    extent: Extent,
    info: &wgpu::AdapterInfo,
) -> Result<serde_json::Value, String> {
    let size = extent.width;
    let mut h = harness::Harness::new(device.clone(), queue.clone());
    h.enable_production()?;
    let mut checks = Vec::new();
    for (w, hgt, kind) in [
        (3, 2, "flat"),
        (7, 5, "ramp"),
        (7, 5, "impulse"),
        (257, 193, "seeded"),
        (1, 13, "seeded"),
        (13, 1, "seeded"),
    ] {
        let e = Extent::new(w, hgt).map_err(|x| x.to_string())?;
        let data = match kind {
            "flat" => vec![0.5; (w * hgt) as usize],
            "ramp" => (0..w * hgt).map(|i| (i % w) as f32 / w as f32).collect(),
            "impulse" => (0..w * hgt)
                .map(|i| if i == (w * hgt) / 2 { 1.0 } else { 0.0 })
                .collect(),
            _ => fixture(w, hgt, 42)?,
        };
        let input = h.upload(
            e,
            wgpu::TextureFormat::R32Float,
            bytemuck::cast_slice(&data),
        )?;
        let mut u = reference::abi::NodeUniform::new(e, e);
        u.parameters[0] = 2.0;
        let (_, bytes) = h.run("normal", &input, u, None)?;
        let actual = harness::normals(&bytes);
        let expected = normal_from_height(w, hgt, &data, 2.0)?;
        if actual.len() != expected.len() {
            return Err("normal extent mismatch".into());
        }
        for (a, b) in actual.iter().zip(&expected) {
            for c in 0..3 {
                if !a[c].is_finite() || (a[c] - f64::from(b[c])).abs() > 0.002 {
                    return Err(format!("normal mismatch {w}x{hgt}"));
                }
            }
            let length = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
            if (length - 1.0).abs() > 0.002 {
                return Err("normal length mismatch".into());
            }
        }
        checks.push(json!({"name":format!("normal_{kind}_{w}x{hgt}"),"passed":true}));
    }
    h.check_errors()?;
    let executor = pollster::block_on(GpuExecutor::new(device.clone(), queue.clone()))
        .map_err(|e| e.to_string())?;
    let texture = |format| {
        device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        })
    };
    let input = texture(wgpu::TextureFormat::R32Float);
    let output = texture(wgpu::TextureFormat::Rgba16Float);
    let data = fixture(size, size, 42)?;
    queue.write_texture(
        input.as_image_copy(),
        bytemuck::cast_slice(&data),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(size * 4),
            rows_per_image: Some(size),
        },
        wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 1,
        },
    );
    let pass = executor
        .prepare(Pass::Normal { strength: 2.0 }, &input, &output, extent)
        .map_err(|e| e.to_string())?;
    let mut rows = Vec::new();
    let mut submissions = Vec::new();
    for _ in 0..110 {
        let mut encoder = device.create_command_encoder(&Default::default());
        pass.encode(&mut encoder);
        let now = Instant::now();
        queue.submit([encoder.finish()]);
        let submission = now.elapsed().as_secs_f64() * 1000.0;
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_secs(30)),
            })
            .map_err(|e| e.to_string())?;
        rows.push(vec![now.elapsed().as_secs_f64() * 1000.0]);
        submissions.push(submission);
    }
    let timing = aggregate_timings(&rows, 10)?;
    let submission_ms = submissions[10..].iter().sum::<f64>() / 100.0;
    let manifest = json!({"normal_strength":2.0,"boundary":"clamp","seed":42,"size":size});
    let bytes = serde_json::to_vec(&manifest).map_err(|e| e.to_string())?;
    let mut padded = bytes.clone();
    while padded.len() % 4 != 0 {
        padded.push(0);
    }
    let hash = SourceDescriptor::from_rgba8(
        &padded,
        Extent::new((padded.len() / 4) as u32, 1).map_err(|e| e.to_string())?,
        InputInterpretation::LinearHeight,
    )
    .map_err(|e| e.to_string())?
    .content_hash
    .iter()
    .map(|b| format!("{b:02x}"))
    .collect::<String>();
    let result = json!({"passed":true,"suite":"baseline","size":size,"adapter":{"name":info.name,"backend":format!("{:?}",info.backend)},"seed":42,"manifest":manifest,"manifest_hash":hash,"manifest_hash_encoding":"BLAKE3 of compact JSON padded with zero bytes to a multiple of four","checks":checks,"timing":{"kind":"completion_upper_bound","warmups":10,"measurements":timing.measurements,"p95_ms":timing.p95_ms,"submission_ms":submission_ms,"peak_bytes":u64::from(size)*u64::from(size)*12+pass.uniform_bytes(),"cancellation_errors":0,"cancellation_scope":"no cancellation requested; worker cancellation not exercised","scope":"normal pass; input and output texture payloads and uniform; excludes driver allocations"}});
    validate_completed_report(&result, h.check_errors())?;
    Ok(result)
}
fn main() {
    if let Err(e) = run() {
        eprintln!("FAIL: {e}");
        std::process::exit(1);
    }
}
