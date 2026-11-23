//! Report schema fixtures are never hardware evidence.
use super::contract;
use serde_json::{Value, json};

fn report() -> Value {
    json!({
        "passed":true,"suite":"seam_heatmap","size":4096,"seed":60,
        "adapter":{"name":"RTX 5070 Ti","backend":"Dx12"},
        "manifest":{"boundary_band_px":4,"gradient":"forward_repeat","ratio_epsilon":1e-8},
        "manifest_hash":"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        "checks":[
            {"name":"cpu_oracle","passed":true},
            {"name":"seam_location","passed":true},
            {"name":"four_pixel_bands","passed":true},
            {"name":"flat_zero_denominator","passed":true},
            {"name":"periodic_and_injected_seam","passed":true},
            {"name":"typed_material_output","passed":true},
            {"name":"input_unchanged","passed":true},
            {"name":"npot_singleton","passed":true},
            {"name":"cancellation","passed":true},
            {"name":"budget","passed":true}
        ],
        "timing":{"scope":"full_node","kind":"completion_upper_bound","warmups":10,
            "measurements":100,"p95_ms":4.0,"submission_ms":0.1,
            "peak_bytes":167772160,"cancellation_errors":0},
        "maximum_quality":{"scope":"full_node","kind":"completion_upper_bound",
            "warmups":10,"measurements":100,"p95_ms":5.0}
    })
}

#[test]
fn report_requires_all_behavioral_checks_and_actual_full_node_default_and_maximum_timings() {
    let good = report();
    contract::validate_report(&good).expect("registered seam report must accept complete evidence");
    for index in 0..good["checks"].as_array().unwrap().len() {
        let mut missing = good.clone();
        missing["checks"].as_array_mut().unwrap().remove(index);
        assert!(
            contract::validate_report(&missing).is_err(),
            "accepted missing check {index}"
        );
    }
    for (path, value) in [
        ("/checks", json!([])),
        ("/checks/0/passed", json!(false)),
        ("/timing/scope", json!("encode_only")),
        ("/timing/kind", json!("cpu_encode_only")),
        ("/timing/warmups", json!(9)),
        ("/timing/measurements", json!(99)),
        ("/timing/p95_ms", json!(16.0)),
        ("/timing/p95_ms", Value::Null),
        ("/timing/peak_bytes", json!(0)),
        ("/timing/peak_bytes", json!(1_073_741_825_u64)),
        ("/timing/cancellation_errors", json!(1)),
        ("/manifest/boundary_band_px", json!(3)),
        ("/maximum_quality", Value::Null),
        ("/maximum_quality/scope", json!("encode_only")),
        ("/maximum_quality/kind", json!("cpu_encode_only")),
        ("/maximum_quality/warmups", json!(0)),
        ("/maximum_quality/measurements", json!(0)),
        ("/maximum_quality/p95_ms", Value::Null),
    ] {
        let mut bad = good.clone();
        *bad.pointer_mut(path).unwrap() = value;
        assert!(
            contract::validate_report(&bad).is_err(),
            "accepted invalid report at {path}"
        );
    }
    let mut worst = good;
    worst["maximum_quality"]["p95_ms"] = json!(40.0);
    assert!(
        contract::validate_report(&worst).is_ok(),
        "maximum-quality timing is reported separately from defaults"
    );
}
