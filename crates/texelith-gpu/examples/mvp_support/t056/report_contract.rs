use serde_json::Value;
pub const CHECKS: &[&str] = &[
    "cpu_oracle",
    "stroke_order",
    "opacity_hardness_endpoints",
    "wrapped_edge",
    "clamp_edge",
    "undo_replay",
    "proxy_full_alignment",
    "stable_digest",
    "npot",
    "singleton_axes",
    "cancellation",
    "low_budget",
    "bounded_batches",
];
pub fn validate(r: &Value) -> Result<(), String> {
    let bad = || "invalid paint coverage/timing report".to_string();
    let budget = r["paint"]["budget_bytes"].as_u64().ok_or_else(bad)?;
    if budget == 0
        || budget > 1 << 30
        || r["timing"]["peak_bytes"]
            .as_u64()
            .is_none_or(|n| n == 0 || n > budget)
        || r["timing"]["scope"] != "full_node"
        || r["timing"]["cache_hits_used_for_timing"] != 0
    {
        return Err(bad());
    }
    let maximum = &r["maximum_quality"];
    if !matches!(
        maximum["kind"].as_str(),
        Some("completion_upper_bound" | "gpu_timestamp")
    ) || maximum["scope"] != "full_node"
        || maximum["warmups"] != 10
        || maximum["measurements"] != 100
        || maximum["p95_ms"]
            .as_f64()
            .is_none_or(|n| !n.is_finite() || n < 0.)
    {
        return Err(bad());
    }
    for name in ["peak_in_flight", "peak_pending"] {
        if r["paint"][name].as_u64().is_none_or(|n| n > 1) {
            return Err(bad());
        }
    }
    for name in [
        "cancellation_requests",
        "cancelled_pass_batches",
        "low_budget_rejections",
        "checkpoint_hits",
        "checkpoint_bytes",
    ] {
        if r["paint"][name]
            .as_u64()
            .is_none_or(|n| n == 0 || n > budget)
        {
            return Err(bad());
        }
    }
    for name in ["cancelled_publications", "validation_errors"] {
        if r["paint"][name] != 0 {
            return Err(bad());
        }
    }
    let batch = r["paint"]["max_segments_per_batch"]
        .as_u64()
        .ok_or_else(bad)?;
    if batch == 0
        || batch > 4096
        || r["paint"]["peak_uploaded_segments"]
            .as_u64()
            .is_none_or(|n| n == 0 || n > batch)
    {
        return Err(bad());
    }
    for name in ["replay_max_error", "scalar_max_error"] {
        if r["paint"][name]
            .as_f64()
            .is_none_or(|n| !n.is_finite() || !(0.0..=2e-5).contains(&n))
        {
            return Err(bad());
        }
    }
    let digest = r["manifest"]["stroke_digest"].as_str().ok_or_else(bad)?;
    if digest.len() != 64
        || !digest.bytes().all(|c| c.is_ascii_hexdigit())
        || r["manifest"]["point_count"]
            .as_u64()
            .is_none_or(|n| n == 0 || n > 100000)
    {
        return Err(bad());
    }
    let checks = r["checks"].as_array().ok_or_else(bad)?;
    for name in CHECKS {
        if !checks
            .iter()
            .any(|c| c["name"] == *name && c["passed"] == true)
        {
            return Err(format!("missing paint check {name}"));
        }
    }
    Ok(())
}
