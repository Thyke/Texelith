//! T-049 evidence contract. The suite runner must call this AND the shared
//! validate_completed_report with its actual validation-error collection.
use serde_json::Value;

pub const CHECKS: &[&str] = &[
    "adjoint",
    "constant_nullspace",
    "neumann_no_outward_flux",
    "periodic_sinusoids",
    "normal_gradient",
    "steep_normal_clamp_count",
    "normal_convention_once",
    "mean_zero",
    "weighted_jacobi",
    "restriction",
    "prolongation",
    "galerkin_257x193",
    "residual_l2",
    "ramp",
    "impulse",
    "singleton_axes",
    "determinism",
    "cancellation",
    "low_budget",
];

pub const OPERATORS: &[&str] = &[
    "normal_gradient",
    "clamp_count",
    "difference",
    "transpose",
    "laplacian",
    "mean",
    "project_mean",
    "jacobi",
    "restrict",
    "prolong",
    "coarse_operator",
    "residual",
    "l2_norm",
];

pub fn validate_poisson_report(r: &Value) -> Result<(), String> {
    if r["passed"] != true
        || r["suite"] != "poisson_operators"
        || r["size"].as_u64().is_none_or(|n| n == 0 || n > 4096)
        || r["seed"].as_u64().is_none()
        || !r["manifest"].is_object()
        || r["adapter"]["name"].as_str().is_none_or(str::is_empty)
        || r["adapter"]["backend"].as_str().is_none_or(str::is_empty)
    {
        return Err("invalid Poisson report identity".into());
    }
    let hash = r["manifest_hash"].as_str().ok_or("missing manifest hash")?;
    if hash.len() != 64 || !hash.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("invalid manifest hash".into());
    }
    let m = &r["manifest"];
    if m["boundary"] != "repeat"
        || m["omega"]
            .as_f64()
            .is_none_or(|v| (v - 2.0 / 3.0).abs() > 1e-7)
        || m["normal_z_floor"] != 0.05
        || m["slope_cap"] != 20.0
        || m["coarsening"] != "ceil_half_area_bilinear_galerkin"
        || m["algorithm_revision"].as_u64().is_none_or(|n| n == 0)
    {
        return Err("default operator manifest drift".into());
    }
    let checks = r["checks"].as_array().ok_or("missing assertions")?;
    if checks.is_empty()
        || checks
            .iter()
            .any(|c| c["passed"] != true || c["name"].as_str().is_none_or(str::is_empty))
    {
        return Err("failed or empty checks".into());
    }
    for name in CHECKS {
        if !checks
            .iter()
            .any(|c| c["name"] == *name && c["passed"] == true)
        {
            return Err(format!("missing oracle: {name}"));
        }
    }
    let coverage = &r["poisson_operators"];
    let budget = coverage["budget_bytes"].as_u64().ok_or("missing budget")?;
    let scratch = coverage["scratch_peak_bytes"]
        .as_u64()
        .ok_or("missing scratch peak")?;
    if budget == 0
        || budget > 1_073_741_824
        || scratch == 0
        || scratch > 201_326_592
        || coverage["peak_in_flight"] != 1
        || coverage["validation_errors"] != 0
        || coverage["cancelled_publications"] != 0
        || coverage["cancellation_requests"]
            .as_u64()
            .is_none_or(|v| v == 0)
        || coverage["low_budget_rejections"]
            .as_u64()
            .is_none_or(|v| v == 0)
    {
        return Err("unexercised or invalid cancellation/budget coverage".into());
    }
    for (field, limit) in [
        ("adjoint_relative_error", 2e-5),
        ("scalar_max_error", 2e-5),
        ("galerkin_max_error", 2e-5),
    ] {
        if coverage[field]
            .as_f64()
            .is_none_or(|v| !v.is_finite() || v < 0.0 || v > limit)
        {
            return Err(format!("invalid numerical error: {field}"));
        }
    }
    for key in ["timing", "maximum_quality"] {
        let t = &r[key];
        if !matches!(
            t["kind"].as_str(),
            Some("gpu_timestamp" | "completion_upper_bound")
        ) || t["warmups"] != 10
            || t["measurements"] != 100
            || t["scope"] != "full_operator_sweep"
            || t["includes_reductions"] != true
            || t["includes_coarse_operators"] != true
            || t["cancellation_errors"] != 0
        {
            return Err(format!("incomplete {key} sweep"));
        }
        for field in ["p95_ms", "submission_ms"] {
            if t[field].as_f64().is_none_or(|v| !v.is_finite() || v < 0.0) {
                return Err(format!("invalid {key}/{field}"));
            }
        }
        let peak = t["peak_bytes"].as_u64().ok_or("missing peak bytes")?;
        if peak == 0 || peak > budget || peak < scratch {
            return Err(format!("invalid {key} peak"));
        }
        let operators = t["operators"]
            .as_array()
            .ok_or("missing measured operators")?;
        for name in OPERATORS {
            if !operators.iter().any(|v| v == name) {
                return Err(format!("untimed operator: {name}"));
            }
        }
    }
    if r["timing"]["p95_ms"].as_f64().is_none_or(|v| v >= 16.0) {
        return Err("default 4K p95 must be below 16 ms".into());
    }
    if r["maximum_quality"]["settings"]
        .as_str()
        .is_none_or(str::is_empty)
    {
        return Err("maximum quality settings must be explicit".into());
    }
    Ok(())
}
