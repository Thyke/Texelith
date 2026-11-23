#[path = "../examples/mvp_support/t065/checks.rs"]
mod checks;
#[path = "../examples/mvp_support/contract.rs"]
mod contract;
#[path = "mvp_t065/formula.rs"]
mod formula;
#[path = "mvp_t065/mesh.rs"]
mod mesh;
#[path = "../examples/mvp_support/t065/oracle.rs"]
mod oracle;
#[path = "../examples/mvp_support/t065/png.rs"]
mod png;
#[path = "../examples/mvp_support/t065/report_contract.rs"]
mod report_contract;
#[path = "../examples/mvp_support/t065/support.rs"]
mod support;

#[test]
fn pbr_is_in_viewport_and_all_but_not_numeric_maps() {
    assert_eq!(contract::select_suite("pbr").unwrap(), ["pbr"]);
    for name in ["viewport", "all"] {
        assert_eq!(
            contract::select_suite(name)
                .unwrap()
                .iter()
                .filter(|s| **s == "pbr")
                .count(),
            1
        );
    }
    assert!(!contract::select_suite("maps").unwrap().contains(&"pbr"));
    assert!(contract::select_suite("").is_err());
    assert!(contract::select_suite("PBR").is_err());
}

#[test]
fn report_rejects_missing_checks_errors_budget_overruns_and_untimed_frames() {
    use serde_json::json;
    let valid = json!({"suite":"pbr","passed":true,"adapter":{"name":"schema fixture","backend":"Dx12"},"checks":report_contract::CHECKS.map(|name|json!({"name":name,"passed":true})),"pbr":{"shared_device":true,"budget_bytes":1_073_741_824u64,"peak_bytes":17_111_536,"validation_errors":0,"stale_publications":0,"peak_pending":1,"peak_in_flight":1,"primitives":["plane","cube","sphere","cylinder"]},"timing":{"scope":"worker_preview","kind":"completion_upper_bound","warmups":10,"measurements":100,"includes_readback":false,"limit_ms":20.,"p95_ms":1.5}});
    report_contract::validate(&valid).unwrap();
    for (section, field, value) in [
        ("pbr", "peak_bytes", json!(1_073_741_825u64)),
        ("pbr", "validation_errors", json!(1)),
        ("pbr", "stale_publications", json!(1)),
        ("pbr", "shared_device", json!(false)),
        ("timing", "measurements", json!(0)),
        ("timing", "p95_ms", json!(20.01)),
        ("pbr", "peak_in_flight", json!(2)),
    ] {
        let mut broken = valid.clone();
        broken[section][field] = value;
        assert!(report_contract::validate(&broken).is_err(), "{field}");
    }
    for name in report_contract::CHECKS {
        let mut broken = valid.clone();
        broken["checks"]
            .as_array_mut()
            .unwrap()
            .retain(|c| c["name"] != name);
        assert!(report_contract::validate(&broken).is_err());
    }
}
#[test]
fn shared_worker_renders_analytic_materials_normals_primitives_and_budget_rejections() {
    if std::env::var("TEXELITH_CI_NO_GPU").as_deref() == Ok("1") {
        eprintln!(
            "SKIP GPU: no GPU adapter (TEXELITH_CI_NO_GPU=1); hardware assertions not run: {:?}",
            std::thread::current().name()
        );
        return;
    }
    use std::sync::Arc;
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
    desc.backends = if cfg!(target_os = "windows") {
        wgpu::Backends::DX12
    } else {
        wgpu::Backends::PRIMARY
    };
    let instance = wgpu::Instance::new(desc);
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    }))
    .expect("T-065 GPU absence is a failed check");
    let info = adapter.get_info();
    eprintln!("T-065 {info:?}");
    assert_ne!(info.device_type, wgpu::DeviceType::Cpu);
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_limits: wgpu::Limits {
            max_texture_dimension_2d: 4096,
            ..Default::default()
        },
        ..Default::default()
    }))
    .unwrap();
    let g = support::Gpu {
        device: Arc::new(device),
        queue: Arc::new(queue),
    };
    let scope = g.device.push_error_scope(wgpu::ErrorFilter::Validation);
    checks::analytic(&g).unwrap();
    checks::primitives_and_lifetime(&g, None).unwrap();
    checks::budget(&g).unwrap();
    g.finish().unwrap();
    assert!(pollster::block_on(scope.pop()).is_none());
}
