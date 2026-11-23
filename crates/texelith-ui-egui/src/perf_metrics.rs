//! Deterministic ADR 0003 gate. All times are integer nanoseconds from one origin.
use serde::Serialize;
use std::{
    collections::{HashMap, HashSet},
    process::ExitCode,
};
use texelith_gpu::Quality;

#[derive(Clone, Debug, Default, Serialize)]
pub struct RunTrace {
    pub drag_start_ns: Option<u64>,
    pub drag_end_ns: Option<u64>,
    pub release_ns: Option<u64>,
    pub full_request_id: Option<u64>,
    pub completed: bool,
    pub dropped_samples: u64,
    pub cold_start_ns: Option<u64>,
    pub gpu_timestamps_available: bool,
    pub inputs: Vec<InputSample>,
    pub requests: Vec<RequestSample>,
    pub frames: Vec<FrameSample>,
    pub normal_dispatches: Vec<NormalSample>,
    pub counters: Counters,
}
#[derive(Clone, Debug, Serialize)]
pub struct InputSample {
    pub event_id: u64,
    pub scheduled_ns: Option<u64>,
    pub occurred_ns: Option<u64>,
    pub handled_ns: Option<u64>,
}
#[derive(Clone, Debug, Serialize)]
pub struct RequestSample {
    pub request_id: u64,
    pub generation: u64,
    pub event_id: u64,
    #[serde(serialize_with = "serialize_quality")]
    pub quality: Quality,
    pub extent: [u32; 2],
    pub requested_ns: Option<u64>,
    pub submitted_ns: Option<u64>,
    pub completed_ns: Option<u64>,
}
fn serialize_quality<S: serde::Serializer>(q: &Quality, s: S) -> Result<S::Ok, S::Error> {
    s.serialize_str(match q {
        Quality::Proxy => "Proxy",
        Quality::Full => "Full",
    })
}
#[derive(Clone, Debug, Serialize)]
pub struct FrameSample {
    pub update_start_ns: Option<u64>,
    pub update_end_ns: Option<u64>,
    pub rendered_ns: Option<u64>,
    pub request_id: Option<u64>,
}
#[derive(Clone, Debug, Serialize)]
pub struct NormalSample {
    pub extent: [u32; 2],
    pub warmed: bool,
    pub normal_only: bool,
    pub submitted_ns: Option<u64>,
    pub completed_ns: Option<u64>,
    pub gpu_duration_ns: Option<u64>,
}
#[derive(Clone, Debug, Default, Serialize)]
pub struct Counters {
    pub peak_in_flight: u64,
    pub peak_pending: u64,
    pub peak_app_bytes: u64,
    pub stale_publications: u64,
    pub validation_errors: u64,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum ColdStartStatus {
    Pass,
    Fail,
    NotMeasured,
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub enum NormalTiming {
    GpuTimestamp,
    #[default]
    CompletionUpperBound,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum Violation {
    Incomplete,
    InvalidTrace,
    UiUpdate,
    FrameP95,
    FrameP99,
    FrameMax,
    ProxyAge,
    PublicationGap,
    DragFrames,
    FreshPublications,
    ReleaseVisible,
    WarmNormalSamples,
    NormalP95,
    AppAllocation,
    PendingWork,
    InFlightWork,
    StalePublication,
    Validation,
}
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct Metrics {
    pub drag_frames: usize,
    pub fresh_publications: usize,
    pub ui_update_p95_ns: Option<u64>,
    pub frame_interval_p95_ns: Option<u64>,
    pub frame_interval_p99_ns: Option<u64>,
    pub frame_interval_max_ns: Option<u64>,
    pub proxy_age_p95_ns: Option<u64>,
    pub fresh_gap_max_ns: Option<u64>,
    pub release_to_visible_ns: Option<u64>,
    pub normal_p95_ns: Option<u64>,
    pub normal_completion_p95_ns: Option<u64>,
    pub normal_timing: NormalTiming,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct GateReport {
    pub metrics: Metrics,
    pub violations: Vec<Violation>,
    pub cold_start: ColdStartStatus,
}
impl GateReport {
    pub fn passed(&self) -> bool {
        self.violations.is_empty()
    }
    pub fn exit_code(&self) -> ExitCode {
        if self.passed() {
            ExitCode::SUCCESS
        } else {
            ExitCode::FAILURE
        }
    }
}
pub fn suite_exit_code(runs: &[GateReport], expected_runs: usize) -> ExitCode {
    if expected_runs > 0 && runs.len() == expected_runs && runs.iter().all(GateReport::passed) {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}
pub fn nearest_rank(samples: &[u64], percentile: u8) -> Option<u64> {
    if samples.is_empty() || !(1..=100).contains(&percentile) {
        return None;
    }
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    let rank = (u128::from(percentile) * sorted.len() as u128).div_ceil(100) as usize;
    Some(sorted[rank - 1])
}
fn violation(v: &mut Vec<Violation>, value: Violation) {
    if !v.contains(&value) {
        v.push(value);
    }
}
fn duration(a: Option<u64>, b: Option<u64>, v: &mut Vec<Violation>) -> Option<u64> {
    match (a, b) {
        (Some(a), Some(b)) if b >= a => Some(b - a),
        (Some(_), Some(_)) => {
            violation(v, Violation::InvalidTrace);
            None
        }
        _ => {
            violation(v, Violation::Incomplete);
            None
        }
    }
}
pub fn evaluate(t: &RunTrace) -> GateReport {
    use Violation::*;
    let mut v = Vec::new();
    let mut m = Metrics::default();
    if !t.completed || t.dropped_samples != 0 {
        violation(&mut v, Incomplete);
    }
    if duration(t.drag_start_ns, t.drag_end_ns, &mut v).is_none_or(|n| n < 10_000_000_000) {
        violation(&mut v, Incomplete);
    }
    duration(t.drag_end_ns, t.release_ns, &mut v);
    let mut inputs = HashMap::new();
    for i in &t.inputs {
        if inputs.insert(i.event_id, i).is_some() {
            violation(&mut v, InvalidTrace);
        }
        duration(i.scheduled_ns, i.occurred_ns, &mut v);
        duration(i.occurred_ns, i.handled_ns, &mut v);
    }
    let mut requests = HashMap::new();
    for r in &t.requests {
        if requests.insert(r.request_id, r).is_some() {
            violation(&mut v, InvalidTrace);
        }
        if let Some(i) = inputs.get(&r.event_id) {
            duration(i.handled_ns, r.requested_ns, &mut v);
        } else {
            violation(&mut v, InvalidTrace);
        }
        if r.submitted_ns.is_some() {
            duration(r.requested_ns, r.submitted_ns, &mut v);
        }
        if r.completed_ns.is_some() {
            duration(r.submitted_ns, r.completed_ns, &mut v);
        }
    }
    let mut updates = Vec::new();
    let mut intervals = Vec::new();
    let mut ages = Vec::new();
    let mut gaps = Vec::new();
    let mut last_active = None;
    let mut last_render = None;
    let mut last_fresh = t.drag_start_ns;
    let mut seen = HashSet::new();
    let mut last_identity: Option<(u64, u64)> = None;
    for f in &t.frames {
        let update = duration(f.update_start_ns, f.update_end_ns, &mut v);
        duration(f.update_end_ns, f.rendered_ns, &mut v);
        let Some(at) = f.rendered_ns else { continue };
        if last_render.is_some_and(|n| at <= n) {
            violation(&mut v, InvalidTrace);
        }
        last_render = Some(at);
        let active = t
            .drag_start_ns
            .zip(t.drag_end_ns)
            .is_some_and(|(a, b)| at >= a && at < b);
        if active {
            m.drag_frames += 1;
            if let Some(n) = update {
                updates.push(n);
            }
            if let Some(n) = last_active.and_then(|a| at.checked_sub(a)) {
                intervals.push(n);
            }
            last_active = Some(at);
        }
        let Some(id) = f.request_id else { continue };
        let Some(r) = requests.get(&id) else {
            violation(&mut v, InvalidTrace);
            continue;
        };
        duration(r.requested_ns, r.submitted_ns, &mut v);
        duration(r.submitted_ns, r.completed_ns, &mut v);
        duration(r.completed_ns, Some(at), &mut v);
        if let Some((old_id, old_generation)) = last_identity
            && (id < old_id || r.generation < old_generation)
        {
            violation(&mut v, StalePublication);
        }
        last_identity = Some((id, r.generation));
        let first = seen.insert(id);
        if active && first && r.quality == Quality::Proxy {
            m.fresh_publications += 1;
            if let Some(i) = inputs.get(&r.event_id)
                && let Some(n) = duration(i.occurred_ns, Some(at), &mut v)
            {
                ages.push(n);
            }
            if let Some(n) = duration(last_fresh, Some(at), &mut v) {
                gaps.push(n);
            }
            last_fresh = Some(at);
        }
        if Some(id) == t.full_request_id && m.release_to_visible_ns.is_none() {
            if r.quality != Quality::Full || r.extent != [4096, 4096] {
                violation(&mut v, InvalidTrace);
            }
            duration(t.release_ns, r.requested_ns, &mut v);
            m.release_to_visible_ns = duration(t.release_ns, Some(at), &mut v);
        }
    }
    if let Some(n) = duration(last_fresh, t.drag_end_ns, &mut v) {
        gaps.push(n);
    }
    if t.full_request_id.is_none() || m.release_to_visible_ns.is_none() {
        violation(&mut v, Incomplete);
    }
    m.ui_update_p95_ns = nearest_rank(&updates, 95);
    m.frame_interval_p95_ns = nearest_rank(&intervals, 95);
    m.frame_interval_p99_ns = nearest_rank(&intervals, 99);
    m.frame_interval_max_ns = intervals.iter().max().copied();
    m.proxy_age_p95_ns = nearest_rank(&ages, 95);
    m.fresh_gap_max_ns = gaps.iter().max().copied();
    let mut gpu = Vec::new();
    let mut completion = Vec::new();
    let warm: Vec<_> = t
        .normal_dispatches
        .iter()
        .filter(|s| s.warmed && s.normal_only && s.extent == [4096, 4096])
        .collect();
    if warm.len() < 100 {
        violation(&mut v, WarmNormalSamples);
    }
    for s in warm {
        if let Some(n) = duration(s.submitted_ns, s.completed_ns, &mut v) {
            completion.push(n);
        }
        if t.gpu_timestamps_available {
            if let Some(n) = s.gpu_duration_ns {
                gpu.push(n);
            } else {
                violation(&mut v, Incomplete);
            }
        }
    }
    m.normal_completion_p95_ns = nearest_rank(&completion, 95);
    m.normal_timing = if t.gpu_timestamps_available {
        NormalTiming::GpuTimestamp
    } else {
        NormalTiming::CompletionUpperBound
    };
    m.normal_p95_ns = if t.gpu_timestamps_available {
        nearest_rank(&gpu, 95)
    } else {
        m.normal_completion_p95_ns
    };
    for (value, bound, issue) in [
        (m.ui_update_p95_ns, 4_000_000, UiUpdate),
        (m.frame_interval_p95_ns, 20_000_000, FrameP95),
        (m.frame_interval_p99_ns, 33_400_000, FrameP99),
        (m.frame_interval_max_ns, 100_000_000, FrameMax),
        (m.proxy_age_p95_ns, 50_000_000, ProxyAge),
        (m.fresh_gap_max_ns, 100_000_000, PublicationGap),
        (m.release_to_visible_ns, 250_000_000, ReleaseVisible),
        (m.normal_p95_ns, 15_999_999, NormalP95),
    ] {
        match value {
            Some(n) if n > bound => violation(&mut v, issue),
            None => violation(&mut v, Incomplete),
            _ => {}
        }
    }
    for (failed, issue) in [
        (m.drag_frames < 500, DragFrames),
        (m.fresh_publications < 100, FreshPublications),
        (t.counters.peak_app_bytes > 1 << 30, AppAllocation),
        (t.counters.peak_pending > 1, PendingWork),
        (t.counters.peak_in_flight > 1, InFlightWork),
        (t.counters.stale_publications > 0, StalePublication),
        (t.counters.validation_errors > 0, Validation),
    ] {
        if failed {
            violation(&mut v, issue);
        }
    }
    GateReport {
        metrics: m,
        violations: v,
        cold_start: match t.cold_start_ns {
            None => ColdStartStatus::NotMeasured,
            Some(n) if n < 1_000_000_000 => ColdStartStatus::Pass,
            _ => ColdStartStatus::Fail,
        },
    }
}
