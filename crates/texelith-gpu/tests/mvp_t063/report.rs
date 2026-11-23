use super::{
    report_contract::{CHECKS, validate},
    suite_contract,
};
use serde_json::{Value, json};

fn valid() -> Value {
    json!({
        "suite":"viewport_resources", "passed":true, "size":1024,
        "adapter":{"name":"test-only schema fixture", "backend":"Dx12"},
        "manifest":{}, "manifest_hash":"ab".repeat(32),
        "checks":CHECKS.iter().map(|name| json!({"name":name,"passed":true})).collect::<Vec<_>>(),
        "viewport_resources":{"budget_bytes":1_073_741_824u64,"peak_bytes":32_000_000,
            "target_bytes":16_777_216,"validation_errors":0,"stale_publications":0,
            "peak_pending":1,"peak_in_flight":1,"resize_requests":256,
            "coalesced_requests":200,"low_budget_rejections":1,"shared_device":true},
        "timing":{"kind":"completion_upper_bound","scope":"worker_preview","warmups":10,"measurements":100,"p95_ms":19.0}
    })
}
#[test]
fn reports_reject_missing_checks_errors_unbounded_work_and_over_budget() {
    validate(&valid()).unwrap();
    for key in ["validation_errors", "stale_publications"] {
        let mut r = valid();
        r["viewport_resources"][key] = json!(1);
        assert!(validate(&r).is_err(), "{key}");
    }
    for key in ["peak_in_flight", "peak_pending"] {
        for value in [0, 2] {
            let mut r = valid();
            r["viewport_resources"][key] = json!(value);
            assert!(validate(&r).is_err(), "{key}");
        }
    }
    for key in [
        "resize_requests",
        "coalesced_requests",
        "low_budget_rejections",
        "target_bytes",
        "peak_bytes",
    ] {
        let mut r = valid();
        r["viewport_resources"][key] = json!(0);
        assert!(validate(&r).is_err(), "{key}");
    }
    for name in CHECKS {
        let mut r = valid();
        r["checks"]
            .as_array_mut()
            .unwrap()
            .retain(|c| c["name"] != *name);
        assert!(validate(&r).is_err(), "missing {name}");
    }
    for (field, value) in [
        ("peak_bytes", json!(1_073_741_825u64)),
        ("budget_bytes", json!(1_073_741_825u64)),
        ("shared_device", json!(false)),
    ] {
        let mut r = valid();
        r["viewport_resources"][field] = value;
        assert!(validate(&r).is_err(), "{field}");
    }
    let mut r = valid();
    r["timing"]["p95_ms"] = json!(20.01);
    assert!(validate(&r).is_err());
    let mut r = valid();
    r["timing"]["measurements"] = json!(0);
    assert!(validate(&r).is_err());
    let mut r = valid();
    r["checks"] = json!([]);
    assert!(validate(&r).is_err());
}
#[test]
fn suite_is_registered_once_in_viewport_and_all_and_unknown_suites_still_fail() {
    assert_eq!(
        suite_contract::select_suite("viewport_resources").unwrap(),
        vec!["viewport_resources"]
    );
    for aggregate in ["viewport", "all"] {
        let selected = suite_contract::select_suite(aggregate).unwrap();
        assert_eq!(
            selected
                .iter()
                .filter(|s| **s == "viewport_resources")
                .count(),
            1
        );
        assert!(
            selected.contains(&"resources"),
            "retain existing viewport/resource coverage"
        );
    }
    assert!(
        !suite_contract::select_suite("maps")
            .unwrap()
            .contains(&"viewport_resources")
    );
    assert!(suite_contract::select_suite("").is_err());
    assert!(suite_contract::select_suite("viewport_resources_typo").is_err());
}
