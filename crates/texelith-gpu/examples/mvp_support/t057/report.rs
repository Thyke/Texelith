//! Test/harness evidence validation, never a measurement runner or GPU fallback.
use serde_json::Value;
use texelith_core::{Extent, InputInterpretation, SourceDescriptor};

pub const REQUIRED_CHECKS: &[&str] = &[
    "constant",
    "ramp",
    "impulse",
    "npot_257x193",
    "singleton_axes",
    "cpu_oracle",
    "periodic_sampling",
    "coverage",
    "overlap_bounds",
    "linear_albedo",
    "height_precision",
    "correlated_layers",
    "no_edge_duplication",
    "seam_reduction",
    "structured_composition",
    "disable_lossless",
    "determinism",
    "cancellation",
    "low_budget",
    "cache_identity",
];

/// Same canonical padded-JSON/BLAKE3 boundary used by the baseline harness.
pub fn manifest_hash(manifest: &Value) -> Result<String, String> {
    let mut bytes = serde_json::to_vec(manifest).map_err(|e| e.to_string())?;
    while bytes.len() % 4 != 0 {
        bytes.push(0);
    }
    let extent = Extent::new((bytes.len() / 4) as u32, 1).map_err(|e| e.to_string())?;
    let hash = SourceDescriptor::from_rgba8(&bytes, extent, InputInterpretation::LinearHeight)
        .map_err(|e| e.to_string())?
        .content_hash;
    Ok(hash.iter().map(|b| format!("{b:02x}")).collect())
}

fn number(object: &Value, key: &str) -> Result<f64, String> {
    object[key]
        .as_f64()
        .filter(|v| v.is_finite() && *v >= 0.0)
        .ok_or_else(|| format!("missing/nonfinite/negative {key}"))
}

fn timing(t: &Value, gated: bool) -> Result<(), String> {
    if !matches!(
        t["kind"].as_str(),
        Some("gpu_timestamp" | "completion_upper_bound")
    ) || t["warmups"].as_u64() != Some(10)
        || t["measurements"].as_u64() != Some(100)
        || t["scope"] != "full_node"
        || t["includes_source"] != true
        || t["cache_hits_used_for_timing"].as_u64() != Some(0)
        || t["peak_bytes"]
            .as_u64()
            .is_none_or(|v| v == 0 || v > 1_073_741_824)
        || t["cancellation_errors"].as_u64() != Some(0)
    {
        return Err("incomplete full-node timing".into());
    }
    let p95 = number(t, "p95_ms")?;
    number(t, "submission_ms")?;
    if gated && p95 >= 16.0 {
        return Err("default full-node p95 must be <16 ms".into());
    }
    Ok(())
}

pub fn validate(r: &Value) -> Result<(), String> {
    if r["passed"] != true
        || r["suite"] != "seamless_offset"
        || r["size"].as_u64().is_none_or(|v| v == 0 || v > 4096)
        || r["seed"].as_u64() != Some(57)
        || r["adapter"]["name"].as_str().is_none_or(str::is_empty)
        || r["adapter"]["backend"].as_str().is_none_or(str::is_empty)
    {
        return Err("invalid seamless_offset report identity".into());
    }
    let manifest = &r["manifest"];
    if manifest["overlap"] != 0.25
        || manifest["boundary"] != "repeat"
        || manifest["epsilon"] != 1e-8
        || manifest["window"] != "cubic_smoothstep"
        || manifest["algorithm_revision"].as_u64() != Some(1)
        || r["manifest_hash"].as_str() != Some(manifest_hash(manifest)?.as_str())
    {
        return Err("defaults or manifest digest drifted".into());
    }
    let checks = r["checks"].as_array().ok_or("missing assertions")?;
    if checks
        .iter()
        .any(|c| c["passed"] != true || c["name"].as_str().is_none_or(str::is_empty))
    {
        return Err("failed/invalid check".into());
    }
    for name in REQUIRED_CHECKS {
        if checks.iter().filter(|c| c["name"] == *name).count() != 1 {
            return Err(format!("missing/duplicated assertion {name}"));
        }
    }
    timing(&r["timing"], true)?;
    timing(&r["maximum_quality"], false)?;
    if r["maximum_quality"]["overlap"] != 0.5 {
        return Err("maximum overlap not measured separately".into());
    }
    let resources = &r["resources"];
    let budget = resources["budget_bytes"]
        .as_u64()
        .filter(|v| *v > 0 && *v <= 1_073_741_824)
        .ok_or("invalid budget")?;
    for t in [&r["timing"], &r["maximum_quality"]] {
        if t["peak_bytes"].as_u64().is_none_or(|v| v > budget) {
            return Err("peak exceeds budget".into());
        }
    }
    if resources["peak_in_flight"].as_u64() != Some(1)
        || resources["peak_pending"].as_u64() != Some(1)
    {
        return Err("unbounded/unused scheduling".into());
    }
    for key in ["cancellation_requests", "low_budget_rejections"] {
        if resources[key].as_u64().is_none_or(|v| v == 0) {
            return Err(format!("unexercised {key}"));
        }
    }
    for key in ["cancelled_publications", "validation_errors"] {
        if resources[key].as_u64() != Some(0) {
            return Err(format!("invalid {key}"));
        }
    }
    let numeric = &r["numerical"];
    for key in ["constant_max_error", "period_max_error", "scalar_max_error"] {
        if number(numeric, key)? > 2e-5 {
            return Err(format!("{key} exceeds scalar bound"));
        }
    }
    if number(numeric, "color_max_error")? > 5e-4 || number(numeric, "minimum_coverage")? <= 1e-8 {
        return Err("color error or coverage bound failed".into());
    }
    let seams = r["seams"].as_array().ok_or("missing seam comparisons")?;
    for fixture in ["discontinuous_noise_256x256", "discontinuous_noise_257x193"] {
        let seam = seams
            .iter()
            .find(|s| s["fixture"] == fixture)
            .ok_or("missing fixed seam fixture")?;
        let before = number(seam, "source_boundary_rms")?;
        let interior = number(seam, "source_interior_rms")?;
        if before <= 0.01
            || interior <= 0.01
            || number(seam, "result_boundary_rms")? > before * 0.5
            || number(seam, "result_interior_rms")? < interior * 0.5
        {
            return Err("seam/detail empirical bounds failed".into());
        }
    }
    Ok(())
}
