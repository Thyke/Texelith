#[path = "common/gpu_availability.rs"]
mod gpu_availability;

#[path = "../examples/mvp_support/t067/checks.rs"]
mod checks;
#[path = "../examples/mvp_support/contract.rs"]
mod contract;
#[path = "../examples/mvp_support/t067/interpolation.rs"]
mod interpolation;
#[path = "../examples/mvp_support/t067/numerics.rs"]
mod numerics;
#[path = "../examples/mvp_support/t067/png.rs"]
mod png;
#[path = "../examples/mvp_support/t067/report_contract.rs"]
mod report_contract;
#[path = "../examples/mvp_support/t067/support.rs"]
mod support;
use texelith_core::Extent;
use texelith_gpu::ibl::*;
#[test]
fn validated_content_identity_bounds_and_fixed_work() {
    let e = Extent {
        width: 2,
        height: 1,
    };
    let a = Environment::new(e, &[[1.0, 2.0, 3.0]; 2]).unwrap();
    assert_eq!(
        a.key(),
        Environment::new(e, &[[1.0, 2.0, 3.0]; 2]).unwrap().key()
    );
    assert_ne!(
        a.key(),
        Environment::new(e, &[[1.0, 2.0, 3.1]; 2]).unwrap().key()
    );
    for value in [f32::NAN, f32::INFINITY, -0.1, 65505.0] {
        assert!(Environment::new(e, &[[value; 3]; 2]).is_err());
    }
    assert!(
        Environment::new(
            Extent {
                width: 2049,
                height: 1
            },
            &[]
        )
        .is_err()
    );
    assert!(Environment::new(e, &[[0.0; 3]]).is_err());
    assert_eq!(passes().count(), 61);
    assert_eq!(
        output_allocations()
            .iter()
            .map(|d| d.byte_len().unwrap())
            .sum::<u64>(),
        4_892_672
    );
    assert_eq!(roughness_lod(0.5), (3, 4, 0.5));
    assert_eq!(roughness_lod(1.0), (7, 7, 0.0));
}
#[test]
fn ibl_suite_membership() {
    assert_eq!(contract::select_suite("ibl").unwrap(), ["ibl"]);
    for name in ["viewport", "all"] {
        assert_eq!(
            contract::select_suite(name)
                .unwrap()
                .iter()
                .filter(|n| **n == "ibl")
                .count(),
            1
        );
    }
    assert!(!contract::select_suite("maps").unwrap().contains(&"ibl"));
}
#[test]
fn report_rejects_missing_behavior_budget_errors_and_hidden_cold_timing() {
    use serde_json::json;
    let valid = json!({"suite":"ibl","passed":true,"checks":report_contract::CHECKS.map(|name|json!({"name":name,"passed":true})),"ibl":{"shared_device":true,"validation_errors":0,"stale_publications":0,"peak_bytes":5_000_000,"cold_load_ms":250.0,"convolution_cold_ms":200.0},"timing":{"warmups":10,"measurements":100,"scope":"cached_worker_preview","p95_ms":2.0}});
    report_contract::validate(&valid).unwrap();
    for (section, key, value) in [
        ("ibl", "peak_bytes", json!(1_073_741_825u64)),
        ("ibl", "cold_load_ms", json!(0)),
        ("ibl", "validation_errors", json!(1)),
        ("ibl", "stale_publications", json!(1)),
        ("timing", "p95_ms", json!(20.0)),
        ("timing", "measurements", json!(0)),
        ("timing", "scope", json!("cold_and_warm")),
    ] {
        let mut bad = valid.clone();
        bad[section][key] = value;
        assert!(report_contract::validate(&bad).is_err());
    }
    for name in report_contract::CHECKS {
        let mut bad = valid.clone();
        bad["checks"]
            .as_array_mut()
            .unwrap()
            .retain(|c| c["name"] != name);
        assert!(report_contract::validate(&bad).is_err());
    }
}
#[test]
fn gpu_environment_energy_orientation_determinism_cache_cancellation_and_budget() {
    if gpu_availability::skip_if_no_gpu() {
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
    .expect("GPU absence is failure");
    assert_ne!(adapter.get_info().device_type, wgpu::DeviceType::Cpu);
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
    numerics::constant_and_direction(&g).unwrap();
    checks::worker(&g, None).unwrap();
    g.finish().unwrap();
    assert!(pollster::block_on(scope.pop()).is_none());
}
