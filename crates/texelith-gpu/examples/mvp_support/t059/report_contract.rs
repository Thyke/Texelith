//! Pure evidence validation. This neither performs GPU work nor invents measurements.
use serde_json::{Value, json};
use texelith_core::{Extent, InputInterpretation, SourceDescriptor};

pub const REQUIRED_CHECKS: &[&str] = &[
    "integer_histogram_cdf_lut",
    "scalar_oracle",
    "color_oracle",
    "variance_normalization",
    "constant",
    "near_black",
    "low_variance_fallback",
    "ramp",
    "impulse",
    "npot_257x193",
    "singleton_axes",
    "repeat_clamp",
    "strength_endpoints",
    "linear_luminance_chroma",
    "sdr_clipping",
    "height_precision",
    "both_bases",
    "cdf_noise_1024",
    "determinism",
    "source_lut_reuse",
    "source_lut_invalidation",
    "proxy_scaling",
    "periodic_coordinates",
    "seam_reduction",
    "cancellation",
    "low_budget",
];
pub const PASSES: &[&str] = &[
    "source",
    "histogram",
    "cdf",
    "lut",
    "gaussian_blend",
    "inverse",
];

pub fn default_manifest() -> Value {
    json!({"seed":59,"base":"offset","overlap":0.25,"boundary":"repeat","strength":1.0,
        "bins":1024,"cdf":"piecewise_uniform_bins","gaussian_lut_entries":8193,
        "gaussian_z_min":-8.0,"gaussian_z_step":0.001953125,"quantile_clip":0.000244140625,
        "epsilon":1e-8,"variance_floor":1e-6,"luminance_ratio_limit":16.0,"algorithm_revision":1})
}
pub fn manifest_hash(manifest: &Value) -> Result<String, String> {
    let mut bytes = serde_json::to_vec(manifest).map_err(|e| e.to_string())?;
    while !bytes.len().is_multiple_of(4) {
        bytes.push(0);
    }
    let e = Extent::new((bytes.len() / 4) as u32, 1).map_err(|e| e.to_string())?;
    let hash = SourceDescriptor::from_rgba8(&bytes, e, InputInterpretation::LinearHeight)
        .map_err(|e| e.to_string())?
        .content_hash;
    Ok(hash.iter().map(|b| format!("{b:02x}")).collect())
}
fn number(value: &Value, key: &str) -> Result<f64, String> {
    value[key]
        .as_f64()
        .filter(|v| v.is_finite() && *v >= 0.0)
        .ok_or_else(|| format!("invalid {key}"))
}
fn timing(t: &Value, gated: bool, budget: u64) -> Result<(), String> {
    if !matches!(
        t["kind"].as_str(),
        Some("gpu_timestamp" | "completion_upper_bound")
    ) || t["scope"] != "full_node"
        || t["warmups"] != 10
        || t["measurements"] != 100
        || t["peak_bytes"]
            .as_u64()
            .is_none_or(|v| v == 0 || v > budget)
        || t["cancellation_errors"] != 0
        || t["output_cache_hits_used_for_timing"] != 0
        || t["source_lut_cache_hits_used_for_timing"] != 0
        || t["cpu_pixel_conversions"] != 0
    {
        return Err("incomplete full-node measurement or hidden CPU/cache work".into());
    }
    for pass in PASSES {
        if t[format!("includes_{pass}")].as_bool() != Some(true) {
            return Err(format!("timing excludes {pass}"));
        }
    }
    if gated && number(t, "p95_ms")? >= 16.0 {
        return Err("default full-node p95 must be <16 ms".into());
    }
    number(t, "p95_ms")?;
    number(t, "submission_ms")?;
    Ok(())
}
pub fn validate(r: &Value) -> Result<(), String> {
    if r["passed"] != true
        || r["suite"] != "seamless_histogram"
        || r["size"] != 4096
        || r["seed"] != 59
        || r["adapter"]["name"].as_str().is_none_or(str::is_empty)
        || r["adapter"]["backend"].as_str().is_none_or(str::is_empty)
    {
        return Err("invalid histogram measurement identity".into());
    }
    if r["manifest"] != default_manifest()
        || r["manifest_hash"].as_str() != Some(manifest_hash(&r["manifest"])?.as_str())
    {
        return Err("histogram defaults or manifest digest drifted".into());
    }
    let checks = r["checks"].as_array().ok_or("missing checks")?;
    if checks
        .iter()
        .any(|c| c["passed"] != true || c["name"].as_str().is_none_or(str::is_empty))
    {
        return Err("failed/invalid check".into());
    }
    for name in REQUIRED_CHECKS {
        if checks.iter().filter(|c| c["name"] == *name).count() != 1 {
            return Err(format!("missing/duplicate check {name}"));
        }
    }
    let resources = &r["resources"];
    let budget = resources["budget_bytes"]
        .as_u64()
        .filter(|b| *b > 0 && *b <= 1_073_741_824)
        .ok_or("invalid budget")?;
    if resources["scratch_peak_bytes"]
        .as_u64()
        .is_none_or(|b| !(4096..=201_326_592).contains(&b) || b > budget)
        || resources["peak_in_flight"] != 1
        || resources["peak_pending"] != 1
    {
        return Err("invalid scratch/scheduling accounting".into());
    }
    for key in [
        "cancellation_requests",
        "low_budget_rejections",
        "source_lut_cache_hits",
        "source_lut_invalidations",
    ] {
        if resources[key].as_u64().is_none_or(|v| v == 0) {
            return Err(format!("unexercised {key}"));
        }
    }
    for key in ["cancelled_publications", "validation_errors"] {
        if resources[key].as_u64() != Some(0) {
            return Err(format!("invalid {key}"));
        }
    }
    timing(&r["timing"], true, budget)?;
    timing(&r["maximum_quality"], false, budget)?;
    if r["maximum_quality"]["base"] != "splat"
        || r["maximum_quality"]["grid"] != 8
        || r["maximum_quality"]["overlap"] != 0.5
        || r["maximum_quality"]["strength"] != 1.0
    {
        return Err(
            "maximum quality must separately measure grid 8 / overlap .5 histogram splats".into(),
        );
    }
    let n = &r["numerical"];
    for (key, bound) in [
        ("constant_max_error", 2e-5),
        ("period_max_error", 2e-5),
        ("scalar_max_error", 2e-5),
        ("color_max_error", 5e-4),
        ("height_precision_max_error", 3e-7),
    ] {
        if number(n, key)? > bound {
            return Err(format!("{key} exceeds bound"));
        }
    }
    if n["same_backend_identical"] != true
        || n["histogram_count_mismatch"] != 0
        || n["nonfinite_pixels"] != 0
        || number(n, "max_luminance_ratio")? > 16.0
        || number(n, "clipped_pixel_fraction")? > 1.0
    {
        return Err("determinism/integer histogram/finite/chroma bounds failed".into());
    }
    // Report color/clipping error separately; distribution guarantees apply to luminance.
    number(n, "color_mean_absolute_change")?;
    let cdf = r["cdf"]
        .as_array()
        .ok_or("missing measured CDF comparisons")?;
    for base in ["offset", "splat"] {
        let rows: Vec<_> = cdf.iter().filter(|c| c["base"] == base).collect();
        if rows.len() != 1 {
            return Err(format!("missing/duplicate {base} CDF"));
        }
        let c = rows[0];
        if c["fixture"] != "value_noise_seed59_1024"
            || c["size"] != 1024
            || c["bins"] != 1024
            || number(c, "max_distance")? > 0.05
        {
            return Err(format!("{base} CDF gate failed"));
        }
    }
    let seams = r["seams"].as_array().ok_or("missing seam comparisons")?;
    for base in ["offset", "splat"] {
        let s = seams
            .iter()
            .find(|s| s["base"] == base && s["fixture"] == "discontinuous_noise_257x193")
            .ok_or("missing fixed seam fixture")?;
        let before = number(s, "source_boundary_rms")?;
        let interior = number(s, "source_interior_rms")?;
        if before <= 0.01
            || interior <= 0.01
            || number(s, "result_boundary_rms")? > before * 0.5
            || number(s, "result_interior_rms")? < interior * 0.5
        {
            return Err("seam/detail gate failed".into());
        }
    }
    Ok(())
}
