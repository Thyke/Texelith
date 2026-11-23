use serde_json::Value;
pub const CHECKS: [&str; 9] = [
    "constant_energy",
    "cube_orientation_seams",
    "analytic_diffuse",
    "deterministic_prefilter",
    "brdf_energy",
    "roughness_mip_interpolation",
    "content_cache",
    "face_cancellation",
    "shared_budget",
];
pub fn validate(r: &Value) -> Result<(), String> {
    let bad = || Err("invalid IBL evidence".into());
    if r["suite"] != "ibl" || r["passed"] != true {
        return bad();
    }
    let Some(checks) = r["checks"].as_array() else {
        return bad();
    };
    if CHECKS.iter().any(|n| {
        !checks
            .iter()
            .any(|c| c["name"] == *n && c["passed"] == true)
    }) {
        return bad();
    }
    let ibl = &r["ibl"];
    let t = &r["timing"];
    if ibl["shared_device"] != true
        || ibl["validation_errors"] != 0
        || ibl["stale_publications"] != 0
        || ibl["peak_bytes"]
            .as_u64()
            .is_none_or(|v| !(4_892_672..=1_073_741_824).contains(&v))
        || ibl["cold_load_ms"]
            .as_f64()
            .is_none_or(|v| !v.is_finite() || v <= 0.0)
        || ibl["convolution_cold_ms"]
            .as_f64()
            .is_none_or(|v| !v.is_finite() || v <= 0.0)
        || t["warmups"] != 10
        || t["measurements"] != 100
        || t["scope"] != "cached_worker_preview"
        || t["p95_ms"]
            .as_f64()
            .is_none_or(|v| !v.is_finite() || !(0.0..20.0).contains(&v))
    {
        return bad();
    }
    Ok(())
}
