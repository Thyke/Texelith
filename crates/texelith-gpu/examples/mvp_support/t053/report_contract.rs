use serde_json::Value;

pub const CHECKS: &[&str] = &[
    "knot_interpolation",
    "curve_endpoints",
    "sixteen_knots",
    "mask_zero_one_blend",
    "optional_mask_one",
    "invert_after_blend",
    "invert_twice_identity",
    "linear_rec709",
    "alpha_ignored",
    "no_srgb_encode",
    "bounded_finite",
    "constant",
    "ramp",
    "impulse",
    "npot_257x193",
    "singleton_axes",
    "determinism",
    "cache_invalidation",
    "cancellation",
    "low_budget",
];

pub fn validate_roughness_report(r: &Value) -> Result<(), String> {
    super::validate_report(r)?;
    if r["suite"] != "roughness" {
        return Err("expected roughness suite".into());
    }
    let stats = &r["roughness"];
    let budget = stats["budget_bytes"]
        .as_u64()
        .ok_or("missing roughness budget")?;
    if budget == 0
        || budget > 1_073_741_824
        || r["timing"]["peak_bytes"]
            .as_u64()
            .is_none_or(|n| n == 0 || n > budget)
        || r["timing"]["scope"] != "full_node"
        || stats["peak_in_flight"].as_u64() != Some(1)
    {
        return Err("invalid roughness budget or scope".into());
    }
    for field in ["cancellation_requests", "low_budget_rejections"] {
        if stats[field].as_u64().is_none_or(|n| n == 0) {
            return Err(format!("unexercised {field}"));
        }
    }
    for field in ["cancelled_publications", "validation_errors"] {
        if stats[field].as_u64() != Some(0) {
            return Err(format!("invalid {field}"));
        }
    }
    let maximum = &r["maximum_quality"];
    if !matches!(
        maximum["kind"].as_str(),
        Some("gpu_timestamp" | "completion_upper_bound")
    ) || maximum["warmups"].as_u64() != Some(10)
        || maximum["measurements"].as_u64() != Some(100)
        || maximum["knot_count"].as_u64() != Some(16)
        || maximum["scope"] != "full_node"
    {
        return Err("invalid roughness maximum quality".into());
    }
    for field in ["p95_ms", "submission_ms"] {
        if maximum[field]
            .as_f64()
            .is_none_or(|n| !n.is_finite() || n < 0.0)
        {
            return Err(format!("invalid maximum quality {field}"));
        }
    }
    let checks = r["checks"].as_array().ok_or("missing checks")?;
    for name in CHECKS {
        if !checks
            .iter()
            .any(|c| c["name"] == *name && c["passed"] == true)
        {
            return Err(format!("missing roughness oracle {name}"));
        }
    }
    Ok(())
}
