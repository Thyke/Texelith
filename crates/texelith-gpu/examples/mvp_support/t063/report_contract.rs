//! Fail-closed evidence contract, independent of renderer implementation.
use serde_json::Value;
pub const CHECKS: &[&str] = &[
    "clear",
    "material_ramp",
    "determinism",
    "resize_churn",
    "matching_generation",
    "stale_preview_rejection",
    "display_lease_immutable",
    "display_fence",
    "combined_budget",
    "low_budget",
    "shared_device",
    "compute_preview_serialization",
];
pub fn validate(r: &Value) -> Result<(), String> {
    if r["suite"] != "viewport_resources" || r["passed"] != true {
        return Err("expected passing viewport_resources report".into());
    }
    if r["size"].as_u64().is_none_or(|v| !(1..=4096).contains(&v))
        || r["adapter"]["name"].as_str().is_none_or(str::is_empty)
        || r["adapter"]["backend"].as_str().is_none_or(str::is_empty)
        || r["manifest_hash"]
            .as_str()
            .is_none_or(|s| s.len() != 64 || !s.bytes().all(|b| b.is_ascii_hexdigit()))
        || !r["manifest"].is_object()
    {
        return Err("invalid viewport run identity".into());
    }
    let viewport = &r["viewport_resources"];
    let budget = viewport["budget_bytes"]
        .as_u64()
        .ok_or("missing shared budget")?;
    if budget == 0
        || budget > 1_073_741_824
        || viewport["peak_bytes"]
            .as_u64()
            .is_none_or(|v| v == 0 || v > budget)
        || viewport["target_bytes"]
            .as_u64()
            .is_none_or(|v| v == 0 || v > viewport["peak_bytes"].as_u64().unwrap_or(0))
    {
        return Err("invalid viewport resource accounting".into());
    }
    for name in ["validation_errors", "stale_publications"] {
        if viewport[name].as_u64() != Some(0) {
            return Err(format!("nonzero/missing {name}"));
        }
    }
    for name in ["peak_pending", "peak_in_flight"] {
        if viewport[name].as_u64() != Some(1) {
            return Err(format!("unbounded or unexercised {name}"));
        }
    }
    for name in [
        "resize_requests",
        "coalesced_requests",
        "low_budget_rejections",
    ] {
        if viewport[name].as_u64().is_none_or(|v| v == 0) {
            return Err(format!("unexercised {name}"));
        }
    }
    if viewport["shared_device"] != true {
        return Err("preview did not use the supplied device".into());
    }
    let checks = r["checks"].as_array().ok_or("missing viewport checks")?;
    for name in CHECKS {
        if checks
            .iter()
            .filter(|c| c["name"] == *name && c["passed"] == true)
            .count()
            != 1
        {
            return Err(format!(
                "missing, duplicate or failed viewport check {name}"
            ));
        }
    }
    if checks.iter().any(|c| c["passed"] != true) {
        return Err("failed check in passing report".into());
    }
    let timing = &r["timing"];
    if timing["kind"] != "completion_upper_bound"
        || timing["scope"] != "worker_preview"
        || timing["warmups"].as_u64() != Some(10)
        || timing["measurements"].as_u64() != Some(100)
        || timing["p95_ms"]
            .as_f64()
            .is_none_or(|v| !v.is_finite() || !(0.0..=20.0).contains(&v))
    {
        return Err("invalid viewport timing (p95 <=20 ms)".into());
    }
    Ok(())
}
