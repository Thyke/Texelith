#![allow(dead_code)]
use serde_json::Value;
type ReportValidator = fn(&Value) -> Result<(), String>;
macro_rules! registered_suite_names {
    ($( $visibility:vis $module:ident { $($export:ident),* } [$($state:ident, $runtime:literal)?] [$($suite:literal, $runner:literal, [$($validator:ident, $validation:literal)?])?]; )*) => {
        const NODE_SUITE_NAMES: &[&str] = &[$($($suite,) ?)*];
        $($($(#[path = $validation] pub(crate) mod $validator;)?)?)*
        const NODE_VALIDATORS: &[(&str, ReportValidator)] = &[
            $($($(($suite, $validator::validate),)?)?)*
        ];
    };
}
texelith_gpu::gpu_nodes!(registered_suite_names);
pub fn select_suite(name: &str) -> Result<Vec<&'static str>, String> {
    // FAZ1_SUITE_SELECTION: T-069
    if name == "compare" {
        return Ok(vec!["compare"]);
    }
    // FAZ1_SUITE_SELECTION: T-068 begin
    if name == "pom" {
        return Ok(vec!["pom"]);
    }
    // FAZ1_SUITE_SELECTION: T-068 end
    // FAZ1_SUITE_SELECTION: T-067 begin
    if name == "ibl" {
        return Ok(vec!["ibl"]);
    }
    // FAZ1_SUITE_SELECTION: T-067 end
    // FAZ1_SUITE_SELECTION: T-081 begin
    if name == "bc5" {
        return Ok(vec!["bc5"]);
    }
    // FAZ1_SUITE_SELECTION: T-081 end
    // FAZ1_SUITE_SELECTION: T-065 begin
    if name == "pbr" {
        return Ok(vec!["pbr"]);
    }
    // FAZ1_SUITE_SELECTION: T-065 end
    // FAZ1_SUITE_SELECTION: T-079 begin
    if name == "readback" {
        return Ok(vec!["readback"]);
    }
    // FAZ1_SUITE_SELECTION: T-079 end
    // FAZ1_SUITE_SELECTION: T-063 named suite and viewport aggregate
    if name == "viewport_resources" {
        return Ok(vec!["viewport_resources"]);
    }
    if name == "viewport" {
        return Ok(vec![
            "resources",
            // FAZ1_SUITE_SELECTION: T-069
            "compare",
            "viewport_resources",
            "pbr",
            "ibl", // T-067 appended
            // FAZ1_SUITE_SELECTION: T-068 begin
            "pom",
            // FAZ1_SUITE_SELECTION: T-068 end
        ]);
    }
    match name {
        "baseline" => Ok(vec!["normal"]),
        "resources" | "viewport" => Ok(vec!["resources"]),
        "pyramid" => Ok(vec!["pyramid"]),
        "maps" | "all" => {
            // Preserve the historical aggregate order and baseline aliases.
            let mut suites = vec!["normal", "resources", "pyramid"];
            // FAZ1_SUITE_SELECTION: T-079 begin
            suites.push("readback");
            // FAZ1_SUITE_SELECTION: T-079 end
            // FAZ1_SUITE_SELECTION: T-081 begin
            suites.push("bc5");
            // FAZ1_SUITE_SELECTION: T-081 end
            suites.extend(
                NODE_SUITE_NAMES
                    .iter()
                    .copied()
                    .filter(|name| *name != "normal"),
            );
            // FAZ1_SUITE_SELECTION: T-063 viewport belongs to all, not maps
            if name == "all" {
                // FAZ1_SUITE_SELECTION: T-069
                suites.push("compare");
                suites.push("viewport_resources");
                // FAZ1_SUITE_SELECTION: T-065 all includes lit inspection
                suites.push("pbr");
                suites.push("ibl"); // FAZ1_SUITE_SELECTION: T-067
                // FAZ1_SUITE_SELECTION: T-068 begin
                suites.push("pom");
                // FAZ1_SUITE_SELECTION: T-068 end
            }
            Ok(suites)
        }
        name => NODE_SUITE_NAMES
            .iter()
            .find(|registered| **registered == name)
            .map(|registered| vec![*registered])
            .ok_or_else(|| format!("unknown or empty suite: {name}")),
    }
}
// FAZ1_SUITE_SELECTION: T-053 begin
#[path = "t053/report_contract.rs"]
mod roughness_report;
#[allow(unused_imports)]
pub use roughness_report::validate_roughness_report;
// FAZ1_SUITE_SELECTION: T-053 end
pub fn select_adapter(
    adapters: &[(&str, &str)],
    name: &str,
    backend: &str,
) -> Result<usize, String> {
    if name.trim().is_empty() || backend.trim().is_empty() {
        return Err("empty adapter or backend".into());
    }
    adapters
        .iter()
        .position(|(n, b)| {
            n.to_lowercase().contains(&name.to_lowercase()) && b.eq_ignore_ascii_case(backend)
        })
        .ok_or_else(|| "requested adapter and backend not found".into())
}
pub fn fixture(w: u32, h: u32, seed: u64) -> Result<Vec<f32>, String> {
    if w == 0 || h == 0 || w > 4096 || h > 4096 {
        return Err("invalid fixture extent".into());
    }
    let mut state = seed;
    Ok((0..w as usize * h as usize)
        .map(|_| {
            state = state.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            ((z ^ (z >> 31)) >> 40) as f32 / 16777215.0
        })
        .collect())
}
pub fn normal_from_height(
    w: u32,
    h: u32,
    data: &[f32],
    strength: f32,
) -> Result<Vec<[f32; 3]>, String> {
    if w == 0
        || h == 0
        || w > 4096
        || h > 4096
        || data.len() != w as usize * h as usize
        || !strength.is_finite()
        || data.iter().any(|v| !v.is_finite())
    {
        return Err("invalid height fixture".into());
    }
    let get = |x: u32, y: u32| data[(y * w + x) as usize];
    let mut out = Vec::with_capacity(data.len());
    for y in 0..h {
        for x in 0..w {
            let dx = (get((x + 1).min(w - 1), y) - get(x.saturating_sub(1), y)) * w as f32 * 0.5;
            let dy = (get(x, (y + 1).min(h - 1)) - get(x, y.saturating_sub(1))) * h as f32 * 0.5;
            let a = -strength * dx;
            let b = strength * dy;
            let len = (a * a + b * b + 1.0).sqrt();
            out.push([a / len, b / len, 1.0 / len]);
        }
    }
    Ok(out)
}
pub fn compare_scalar(a: &[f32], b: &[f32], tolerance: f32) -> Result<(), String> {
    if a.is_empty()
        || a.len() != b.len()
        || !tolerance.is_finite()
        || tolerance < 0.0
        || a.iter()
            .zip(b)
            .any(|(x, y)| !x.is_finite() || !y.is_finite() || (x - y).abs() > tolerance)
    {
        Err("invalid or mismatching scalar values".into())
    } else {
        Ok(())
    }
}
pub struct Timing {
    pub measurements: usize,
    pub p95_ms: f64,
}
pub fn aggregate_timings(rows: &[Vec<f64>], warmups: usize) -> Result<Timing, String> {
    if rows.len() <= warmups
        || rows
            .iter()
            .any(|r| r.is_empty() || r.iter().any(|x| !x.is_finite() || *x < 0.0))
    {
        return Err("invalid timing samples".into());
    }
    let mut totals: Vec<f64> = rows[warmups..].iter().map(|r| r.iter().sum()).collect();
    if totals.iter().any(|x| !x.is_finite()) {
        return Err("timing overflow".into());
    }
    totals.sort_by(f64::total_cmp);
    let n = totals.len();
    Ok(Timing {
        measurements: n,
        p95_ms: totals[(n * 95).div_ceil(100) - 1],
    })
}
pub fn validate_report(r: &Value) -> Result<(), String> {
    let bad = || "invalid report".to_string();
    if r["passed"].as_bool() != Some(true)
        || select_suite(r["suite"].as_str().ok_or_else(bad)?).is_err()
        || !(1..=4096).contains(&r["size"].as_u64().ok_or_else(bad)?)
        || r["seed"].as_u64().is_none()
        || !r["manifest"].is_object()
    {
        return Err(bad());
    }
    let a = &r["adapter"];
    if a["name"].as_str().is_none_or(str::is_empty)
        || a["backend"].as_str().is_none_or(str::is_empty)
    {
        return Err(bad());
    }
    let hash = r["manifest_hash"].as_str().ok_or_else(bad)?;
    if hash.len() != 64 || !hash.bytes().all(|c| c.is_ascii_hexdigit()) {
        return Err(bad());
    }
    let checks = r["checks"].as_array().ok_or_else(bad)?;
    if checks.is_empty()
        || checks.iter().any(|c| {
            c["name"].as_str().is_none_or(str::is_empty) || c["passed"].as_bool() != Some(true)
        })
    {
        return Err(bad());
    }
    let t = &r["timing"];
    if !matches!(
        t["kind"].as_str(),
        Some("gpu_timestamp" | "completion_upper_bound")
    ) || t["warmups"].as_u64() != Some(10)
        || t["measurements"].as_u64() != Some(100)
        || t["peak_bytes"].as_u64().is_none_or(|n| n == 0)
        || t["cancellation_errors"].as_u64() != Some(0)
    {
        return Err(bad());
    }
    for key in ["p95_ms", "submission_ms"] {
        if t[key].as_f64().is_none_or(|v| !v.is_finite() || v < 0.0) {
            return Err(bad());
        }
    }
    if t["p95_ms"].as_f64().is_none_or(|v| v >= 16.0) {
        return Err("p95 must be below 16 ms".into());
    }
    if let Some((_, validate)) = NODE_VALIDATORS
        .iter()
        .find(|(suite, _)| r["suite"] == *suite)
    {
        validate(r)?;
    }
    Ok(())
}

pub fn validate_completed_report(r: &Value, gpu_errors: Result<(), String>) -> Result<(), String> {
    gpu_errors?;
    validate_report(r)
}

pub fn validate_resources_report(r: &Value) -> Result<(), String> {
    validate_report(r)?;
    let resources = &r["resources"];
    let budget = resources["budget_bytes"]
        .as_u64()
        .ok_or("missing resource budget")?;
    if budget == 0
        || budget > 1_073_741_824
        || r["timing"]["peak_bytes"]
            .as_u64()
            .is_none_or(|n| n == 0 || n > budget)
    {
        return Err("resource peak exceeds hard budget".into());
    }
    for key in ["peak_in_flight", "peak_pending"] {
        if resources[key].as_u64().is_none_or(|n| n > 1) {
            return Err(format!("invalid {key}"));
        }
    }
    for key in [
        "cancellation_requests",
        "viewport_reserved_bytes",
        "readback_reserved_bytes",
    ] {
        if resources[key].as_u64().is_none_or(|n| n == 0) {
            return Err(format!("missing exercised {key}"));
        }
    }
    for key in ["cancelled_publications", "validation_errors"] {
        if resources[key].as_u64() != Some(0) {
            return Err(format!("invalid {key}"));
        }
    }
    let checks = r["checks"]
        .as_array()
        .ok_or("missing resource assertions")?;
    for name in [
        "typed_sources",
        "job_identity",
        "coherent_outputs",
        "liveness",
        "cache_eviction",
        "lease_fence",
        "low_budget",
        "cancellation",
    ] {
        if !checks
            .iter()
            .any(|c| c["name"] == name && c["passed"] == true)
        {
            return Err(format!("missing resource check {name}"));
        }
    }
    Ok(())
}

pub fn validate_pyramid_report(r: &Value) -> Result<(), String> {
    validate_report(r)?;
    let levels = texelith_gpu::pyramid_extents(
        texelith_core::Extent::new(
            r["size"].as_u64().ok_or("missing size")? as u32,
            r["size"].as_u64().ok_or("missing size")? as u32,
        )
        .map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?
    .len() as u64;
    let p = &r["pyramid"];
    let budget = p["budget_bytes"].as_u64().ok_or("missing budget")?;
    if budget == 0
        || budget > 1_073_741_824
        || p["levels"].as_u64() != Some(levels)
        || p["scratch_peak_bytes"]
            .as_u64()
            .is_none_or(|n| n == 0 || n > 201_326_592)
        || r["timing"]["peak_bytes"]
            .as_u64()
            .is_none_or(|n| n == 0 || n > budget)
        || r["timing"]["scope"] != "full_node"
        || p["peak_in_flight"].as_u64().is_none_or(|n| n > 1)
    {
        return Err("invalid pyramid budget, extent or timing scope".into());
    }
    for field in [
        "cancellation_requests",
        "low_budget_rejections",
        "shared_cache_hits",
    ] {
        if p[field].as_u64().is_none_or(|n| n == 0) {
            return Err(format!("unexercised {field}"));
        }
    }
    for field in ["cancelled_publications", "validation_errors"] {
        if p[field].as_u64() != Some(0) {
            return Err(format!("invalid {field}"));
        }
    }
    let maximum = &r["maximum_quality"];
    if !matches!(
        maximum["kind"].as_str(),
        Some("gpu_timestamp" | "completion_upper_bound")
    ) || maximum["warmups"].as_u64() != Some(10)
        || maximum["measurements"].as_u64() != Some(100)
        || maximum["levels"].as_u64() != Some(levels)
        || maximum["scope"] != "full_node"
    {
        return Err("invalid maximum quality timing".into());
    }
    for key in ["p95_ms", "submission_ms"] {
        if maximum[key]
            .as_f64()
            .is_none_or(|n| !n.is_finite() || n < 0.0)
        {
            return Err(format!("invalid maximum quality {key}"));
        }
    }
    let checks = r["checks"].as_array().ok_or("missing checks")?;
    for name in [
        "source_transfer_alpha",
        "signed_source_hashes",
        "constant",
        "ramp",
        "impulse",
        "npot_257x193",
        "singleton_axes",
        "repeat_clamp",
        "manual_bilinear",
        "reconstruction",
        "determinism",
        "cancellation",
        "low_budget",
        "shared_cache",
    ] {
        if !checks
            .iter()
            .any(|c| c["name"] == name && c["passed"] == true)
        {
            return Err(format!("missing pyramid check {name}"));
        }
    }
    Ok(())
}

// FAZ1_SUITE_SELECTION: T-047 height report contract
pub fn validate_height_report(r: &Value) -> Result<(), String> {
    validate_report(r)?;
    if r["suite"] != "height" {
        return Err("expected height suite".into());
    }
    let height = &r["height"];
    let levels = ((r["size"].as_u64().ok_or("size")? as f64).log2().ceil() as u64) + 1;
    if height["levels"].as_u64() != Some(levels)
        || height["budget_bytes"]
            .as_u64()
            .is_none_or(|v| v == 0 || v > 1_073_741_824)
        || height["scratch_peak_bytes"]
            .as_u64()
            .is_none_or(|v| v == 0 || v > 201_326_592)
        || r["timing"]["peak_bytes"]
            .as_u64()
            .is_none_or(|v| v > height["budget_bytes"].as_u64().unwrap_or(0))
        || height["peak_in_flight"].as_u64() != Some(1)
        || height["cancellation_requests"]
            .as_u64()
            .is_none_or(|v| v == 0)
        || height["cancelled_publications"].as_u64() != Some(0)
        || height["validation_errors"].as_u64() != Some(0)
    {
        return Err("invalid height resource/cancellation coverage".into());
    }
    for key in [
        "low_budget_rejections",
        "shared_pyramid_cache_hits",
        "output_cache_hits",
    ] {
        if height[key].as_u64().is_none_or(|v| v == 0) {
            return Err(format!("missing height coverage: {key}"));
        }
    }
    for name in [
        "unit_reconstruction",
        "band_isolation",
        "coarse_removal",
        "explicit_levels",
        "flat_min_max",
        "min_max_remap",
        "proxy_scaling",
        "srgb_linear_ramp",
        "cache_reuse",
        "constant",
        "ramp",
        "impulse",
        "npot_257x193",
        "singleton_axes",
        "repeat_clamp",
        "determinism",
        "cancellation",
        "low_budget",
    ] {
        if !r["checks"]
            .as_array()
            .ok_or("checks")?
            .iter()
            .any(|c| c["name"] == name && c["passed"] == true)
        {
            return Err(format!("missing oracle: {name}"));
        }
    }
    for key in ["timing", "maximum_quality"] {
        let t = &r[key];
        if !matches!(
            t["kind"].as_str(),
            Some("gpu_timestamp" | "completion_upper_bound")
        ) || t["scope"] != "full_node"
            || t["includes_pyramid"] != true
            || t["warmups"].as_u64() != Some(10)
            || t["measurements"].as_u64() != Some(100)
        {
            return Err("height timing must include the full node".into());
        }
        for field in ["p95_ms", "submission_ms"] {
            if t[field].as_f64().is_none_or(|v| !v.is_finite() || v < 0.0) {
                return Err("invalid height timing".into());
            }
        }
    }
    if r["maximum_quality"]["levels"].as_u64() != Some(levels) {
        return Err("maximum quality levels".into());
    }
    Ok(())
}
