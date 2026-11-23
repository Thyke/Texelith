//! User-approved T-050 gates (2026-11-11). The common 16 ms gate is applied
//! to real 1K interactive measurements; full-resolution timing has its own gate.
use serde_json::Value;

pub fn validate(
    report: &Value,
    validate_common: impl FnOnce(&Value) -> Result<(), String>,
) -> Result<(), String> {
    if report["suite"] != "poisson" || report["interactive_proxy"]["size"] != 1024 {
        return Err("missing Poisson 1K proxy measurements".into());
    }
    for key in ["timing", "maximum_quality", "interactive_proxy"] {
        let t = &report[key];
        if !matches!(
            t["kind"].as_str(),
            Some("gpu_timestamp" | "completion_upper_bound")
        ) || t["warmups"] != 10
            || t["measurements"] != 100
            || t["nonconverged_samples"] != 0
            || t["cancellation_errors"] != 0
            || t["scratch_peak_bytes"]
                .as_u64()
                .is_none_or(|n| n == 0 || n > 192 * 1024 * 1024)
            || t["peak_bytes"]
                .as_u64()
                .is_none_or(|n| n == 0 || n > 1024 * 1024 * 1024)
            || t["relative_optimality_gap"]
                .as_f64()
                .is_none_or(|v| !v.is_finite() || !(0.0..=1e-3).contains(&v))
            || t["rms"]
                .as_f64()
                .is_none_or(|v| !v.is_finite() || !(0.0..=1e-3).contains(&v))
        {
            return Err(format!("invalid Poisson coverage: {key}"));
        }
        for field in [
            "p95_ms",
            "submission_ms",
            "objective_residual_l2",
            "optimum_residual_l2",
            "gradient_l2",
        ] {
            if t[field].as_f64().is_none_or(|v| !v.is_finite() || v < 0.0) {
                return Err(format!("invalid {key}/{field}"));
            }
        }
        let tolerance = 1e-3 * t["gradient_l2"].as_f64().unwrap();
        if t["objective_residual_l2"].as_f64().unwrap()
            > t["optimum_residual_l2"].as_f64().unwrap().hypot(tolerance)
        {
            return Err(format!("objective exceeds optimum tolerance: {key}"));
        }
    }
    if report["timing"]["p95_ms"]
        .as_f64()
        .is_none_or(|v| v > 100.0)
    {
        return Err("Poisson full-resolution p95 must be <=100 ms".into());
    }
    if report["interactive_proxy"]["p95_ms"]
        .as_f64()
        .is_none_or(|v| v >= 16.0)
    {
        return Err("Poisson 1K proxy p95 must be <16 ms".into());
    }
    // Preserve the shared schema/error checks and its unchanged interactive
    // threshold. This validation view uses actual proxy data, never scaled or
    // fabricated timing, and is not the report written to disk.
    let mut interactive = report.clone();
    interactive["timing"] = report["interactive_proxy"].clone();
    validate_common(&interactive)
}
