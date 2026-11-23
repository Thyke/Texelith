use serde_json::Value;
pub const CHECKS: [&str; 12] = [
    "analytic_ggx",
    "metallic_limits",
    "roughness_limits",
    "signed_normals",
    "dx_gl_once",
    "ao_excludes_direct",
    "exposure_aces",
    "single_exact_srgb",
    "all_primitives",
    "immutable_leases",
    "latest_only",
    "budget_rejection",
];
pub fn validate(r: &Value) -> Result<(), String> {
    let p = &r["pbr"];
    let t = &r["timing"];
    if r["suite"] != "pbr"
        || r["passed"] != true
        || r["adapter"]["name"].as_str().is_none_or(str::is_empty)
        || r["adapter"]["backend"].as_str().is_none_or(str::is_empty)
        || p["shared_device"] != true
        || p["budget_bytes"].as_u64() != Some(1_073_741_824)
        || p["peak_bytes"]
            .as_u64()
            .is_none_or(|v| v == 0 || v > 1_073_741_824)
        || p["validation_errors"].as_u64() != Some(0)
        || p["stale_publications"].as_u64() != Some(0)
        || p["peak_pending"].as_u64() != Some(1)
        || p["peak_in_flight"].as_u64() != Some(1)
        || t["scope"] != "worker_preview"
        || t["kind"] != "completion_upper_bound"
        || t["warmups"].as_u64() != Some(10)
        || t["measurements"].as_u64() != Some(100)
        || t["includes_readback"] != false
        || t["limit_ms"].as_f64() != Some(20.)
        || t["p95_ms"]
            .as_f64()
            .is_none_or(|v| !v.is_finite() || !(0.0..=20.0).contains(&v))
    {
        return Err("invalid PBR device/budget/publication/timing report".into());
    }
    for name in CHECKS {
        if !r["checks"].as_array().is_some_and(|checks| {
            checks
                .iter()
                .any(|c| c["name"] == name && c["passed"] == true)
        }) {
            return Err(format!("missing PBR check: {name}"));
        }
    }
    if p["primitives"] != serde_json::json!(["plane", "cube", "sphere", "cylinder"]) {
        return Err("missing primitive render coverage".into());
    }
    Ok(())
}
