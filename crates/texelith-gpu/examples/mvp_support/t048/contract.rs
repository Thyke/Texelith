use serde_json::Value;
pub const CHECKS: &[&str] = &[
    "legacy_central",
    "sobel",
    "scharr",
    "multi_scale",
    "gradient_before_normalize",
    "uv_ramp_y_sign",
    "unit_length",
    "gl_dx_metadata",
    "no_double_flip",
    "proxy_scaling",
    "linear_height_precision",
    "constant",
    "ramp",
    "impulse",
    "npot_257x193",
    "singleton_axes",
    "repeat_clamp",
    "determinism",
    "cancellation",
    "low_budget",
];
pub fn validate(r: &Value) -> Result<(), String> {
    let n = &r["normal"];
    let budget = n["budget_bytes"].as_u64().ok_or("missing normal budget")?;
    if budget == 0
        || budget > 1_073_741_824
        || r["timing"]["peak_bytes"]
            .as_u64()
            .is_none_or(|v| v == 0 || v > budget)
    {
        return Err("invalid normal budget/peak".into());
    }
    for field in ["peak_in_flight", "peak_pending"] {
        if n[field].as_u64().is_none_or(|v| v > 1) {
            return Err(format!("invalid {field}"));
        }
    }
    for field in [
        "cancellation_requests",
        "cancelled_pass_batches",
        "low_budget_rejections",
    ] {
        if n[field].as_u64().is_none_or(|v| v == 0) {
            return Err(format!("unexercised {field}"));
        }
    }
    for field in ["cancelled_publications", "validation_errors"] {
        if n[field].as_u64() != Some(0) {
            return Err(format!("invalid {field}"));
        }
    }
    for key in ["timing", "maximum_quality"] {
        let t = &r[key];
        if !matches!(
            t["kind"].as_str(),
            Some("gpu_timestamp" | "completion_upper_bound")
        ) || t["warmups"] != 10
            || t["measurements"] != 100
            || t["scope"] != "full_node"
            || t["includes_pyramid"] != true
        {
            return Err("invalid full-node normal timing".into());
        }
        for field in ["p95_ms", "submission_ms"] {
            if t[field].as_f64().is_none_or(|v| !v.is_finite() || v < 0.0) {
                return Err(format!("invalid {field}"));
            }
        }
    }
    if r["maximum_quality"]["filter"] != "scharr" || r["maximum_quality"]["scales"] != 16 {
        return Err("invalid maximum quality configuration".into());
    }
    let checks = r["checks"].as_array().ok_or("normal checks")?;
    for name in CHECKS {
        if !checks
            .iter()
            .any(|c| c["name"] == *name && c["passed"] == true)
        {
            return Err(format!("missing normal check {name}"));
        }
    }
    Ok(())
}
