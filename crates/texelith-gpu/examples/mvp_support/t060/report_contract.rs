use serde_json::Value;

pub const CHECKS: &[&str] = &[
    "cpu_oracle",
    "seam_location",
    "four_pixel_bands",
    "flat_zero_denominator",
    "periodic_and_injected_seam",
    "typed_material_output",
    "input_unchanged",
    "npot_singleton",
    "cancellation",
    "budget",
];

pub fn validate(report: &Value) -> Result<(), String> {
    if report["manifest"]["boundary_band_px"] != 4
        || report["manifest"]["gradient"] != "forward_repeat"
        || report["manifest"]["ratio_epsilon"] != 1e-8
    {
        return Err("invalid seam diagnostic manifest".into());
    }
    for key in ["timing", "maximum_quality"] {
        let t = &report[key];
        if t["scope"] != "full_node"
            || !matches!(
                t["kind"].as_str(),
                Some("completion_upper_bound" | "gpu_timestamp")
            )
            || t["warmups"] != 10
            || t["measurements"] != 100
            || t["p95_ms"]
                .as_f64()
                .is_none_or(|v| !v.is_finite() || v < 0.0)
        {
            return Err(format!("invalid seam diagnostic {key}"));
        }
    }
    if report["timing"]["peak_bytes"]
        .as_u64()
        .is_none_or(|v| v == 0 || v > 1_073_741_824)
    {
        return Err("invalid seam diagnostic memory coverage".into());
    }
    let checks = report["checks"].as_array().ok_or("missing seam checks")?;
    for name in CHECKS {
        if !checks
            .iter()
            .any(|c| c["name"] == *name && c["passed"] == true)
        {
            return Err(format!("missing seam diagnostic check {name}"));
        }
    }
    Ok(())
}
