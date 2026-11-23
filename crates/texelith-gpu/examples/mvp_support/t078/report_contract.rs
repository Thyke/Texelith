//! Packing-specific evidence validation. This does not execute or simulate a GPU.
use serde_json::Value;
use texelith_core::{Extent, InputInterpretation, SourceDescriptor};

pub const REQUIRED_CHECKS: &[&str] = &[
    "cpu_oracle",
    "orm",
    "hdrp_mask_map",
    "gltf_metallic_roughness",
    "constants_invert",
    "srgb_once",
    "packed_linear",
    "mixed_transfer",
    "source_alpha_ignored",
    "normal_y_once",
    "normal_renormalization",
    "png16_precision",
    "signed_displacement",
    "ordered_dither",
    "keep_npot",
    "npot_257x193",
    "singleton_axes",
    "area_downsample",
    "bilinear_upscale",
    "pad_repeat_clamp_metadata",
    "bc5_multiple_of_four",
    "full_resolution_sources",
    "determinism",
    "cancellation",
    "low_budget",
    "fence_retirement",
];

pub fn manifest_hash(manifest: &Value) -> Result<String, String> {
    let mut bytes = serde_json::to_vec(manifest).map_err(|e| e.to_string())?;
    while !bytes.len().is_multiple_of(4) {
        bytes.push(0);
    }
    let extent = Extent::new((bytes.len() / 4) as u32, 1).map_err(|e| e.to_string())?;
    let descriptor =
        SourceDescriptor::from_rgba8(&bytes, extent, InputInterpretation::LinearHeight)
            .map_err(|e| e.to_string())?;
    Ok(descriptor
        .content_hash
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}

fn number(object: &Value, key: &str) -> Result<f64, String> {
    object[key]
        .as_f64()
        .filter(|v| v.is_finite() && *v >= 0.0)
        .ok_or_else(|| format!("missing/invalid {key}"))
}

fn timing(value: &Value, budget: u64, gated: bool) -> Result<(), String> {
    if !matches!(
        value["kind"].as_str(),
        Some("gpu_timestamp" | "completion_upper_bound")
    ) || value["warmups"].as_u64() != Some(10)
        || value["measurements"].as_u64() != Some(100)
        || value["scope"] != "full_node"
        || value["includes_source"] != true
        || value["cache_hits_used_for_timing"].as_u64() != Some(0)
        || value["cancellation_errors"].as_u64() != Some(0)
        || value["peak_bytes"]
            .as_u64()
            .is_none_or(|v| v == 0 || v > budget)
    {
        return Err("incomplete full-node timing".into());
    }
    let p95 = number(value, "p95_ms")?;
    number(value, "submission_ms")?;
    if gated && p95 >= 16.0 {
        return Err("default p95 must be <16 ms".into());
    }
    Ok(())
}

/// Node-only validator for ADR 0012. Shared validate_report runs separately.
pub fn validate(report: &Value) -> Result<(), String> {
    if report["passed"] != true
        || report["suite"] != "packing"
        || report["size"].as_u64().is_none_or(|v| v == 0 || v > 4096)
        || report["seed"].as_u64() != Some(78)
        || report["adapter"]["name"].as_str().is_none_or(str::is_empty)
        || report["adapter"]["backend"]
            .as_str()
            .is_none_or(str::is_empty)
    {
        return Err("invalid packing identity".into());
    }
    let manifest = &report["manifest"];
    if manifest["algorithm_revision"] != 1
        || manifest["default_output"] != "orm_rgb8"
        || manifest["default_size"] != "keep"
        || manifest["default_transfer"] != "linear"
        || manifest["default_dither"] != "none"
        || manifest["source_quality"] != "full"
        || report["manifest_hash"].as_str() != Some(manifest_hash(manifest)?.as_str())
    {
        return Err("packing defaults or digest drifted".into());
    }
    let checks = report["checks"]
        .as_array()
        .ok_or("missing packing assertions")?;
    if checks
        .iter()
        .any(|c| c["passed"] != true || c["name"].as_str().is_none_or(str::is_empty))
    {
        return Err("failed packing assertion".into());
    }
    for name in REQUIRED_CHECKS {
        if checks.iter().filter(|c| c["name"] == *name).count() != 1 {
            return Err(format!("missing/duplicate {name}"));
        }
    }
    let resources = &report["resources"];
    let budget = resources["budget_bytes"]
        .as_u64()
        .filter(|v| *v > 0 && *v <= 1_073_741_824)
        .ok_or("invalid budget")?;
    for key in ["peak_in_flight", "peak_pending"] {
        if resources[key].as_u64().is_none_or(|v| v > 1) {
            return Err(format!("invalid {key}"));
        }
    }
    for key in [
        "cancellation_requests",
        "cancelled_pass_batches",
        "low_budget_rejections",
        "fence_retirements",
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
    timing(&report["timing"], budget, true)?;
    timing(&report["maximum_quality"], budget, false)?;
    if report["maximum_quality"]["configuration"] != "rgba16_ordered4x4_resize" {
        return Err("maximum quality must be measured separately".into());
    }
    for (key, limit) in [
        ("scalar_max_error", 2e-5),
        ("normal_component_max_error", 2e-3),
        ("normal_length_max_error", 2e-3),
        ("integer_max_lsb_error", 1.0),
        ("displacement_max_error", 2e-5),
    ] {
        if number(&report["numerical"], key)? > limit {
            return Err(format!("excessive {key}"));
        }
    }
    Ok(())
}
