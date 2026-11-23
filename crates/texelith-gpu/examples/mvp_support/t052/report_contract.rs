use serde_json::Value;

pub const CHECKS: &[&str] = &[
    "flat_zero",
    "bump_positive",
    "bowl_negative",
    "output_separation",
    "convention_equivalence",
    "steep_finite",
    "cpu_oracle",
    "npot_257x193",
    "singleton_axes",
    "repeat_clamp",
    "determinism",
    "cancellation",
    "low_budget",
];

pub fn validate(report: &Value) -> Result<(), String> {
    for key in ["timing", "maximum_quality"] {
        let t = &report[key];
        if !matches!(
            t["kind"].as_str(),
            Some("gpu_timestamp" | "completion_upper_bound")
        ) || t["scope"] != "full_node"
            || t["warmups"] != 10
            || t["measurements"] != 100
        {
            return Err(format!("invalid curvature {key} coverage"));
        }
        for field in ["p95_ms", "submission_ms"] {
            if t[field].as_f64().is_none_or(|v| !v.is_finite() || v < 0.0) {
                return Err(format!("invalid curvature {key}/{field}"));
            }
        }
    }
    if report["timing"]["peak_bytes"]
        .as_u64()
        .is_none_or(|v| v == 0 || v > 1_073_741_824)
    {
        return Err("invalid curvature peak bytes".into());
    }
    let checks = report["checks"]
        .as_array()
        .ok_or("missing curvature checks")?;
    for name in CHECKS {
        if !checks
            .iter()
            .any(|c| c["name"] == *name && c["passed"] == true)
        {
            return Err(format!("missing curvature check {name}"));
        }
    }
    Ok(())
}
