//! Synthetic contracts for the production ADR 0003 evaluator, before implementation.
//! Proposed API and timestamp semantics: docs/evidence/T-020/test-contract.md.

use std::process::ExitCode;

use texelith_gpu::Quality;
use texelith_ui_egui::{
    perf_metrics::{
        ColdStartStatus, Counters, FrameSample, GateReport, InputSample, NormalSample,
        NormalTiming, RequestSample, RunTrace, Violation, evaluate, nearest_rank, suite_exit_code,
    },
    startup::Config,
};

const MS: u64 = 1_000_000;
const START: u64 = 1_000 * MS;
const DURATION: u64 = 10_000 * MS;
const END: u64 = START + DURATION;
const GIB: u64 = 1 << 30;
type TraceMutation = fn(&mut RunTrace);

fn event_time(index: usize) -> u64 {
    START + u64::try_from(index).unwrap() * DURATION / 600
}

fn passing_trace() -> RunTrace {
    let mut trace = RunTrace {
        drag_start_ns: Some(START),
        drag_end_ns: Some(END),
        release_ns: Some(END),
        full_request_id: Some(601),
        completed: true,
        dropped_samples: 0,
        cold_start_ns: None,
        gpu_timestamps_available: true,
        inputs: Vec::new(),
        requests: Vec::new(),
        frames: Vec::new(),
        normal_dispatches: Vec::new(),
        counters: Counters {
            peak_in_flight: 1,
            peak_pending: 1,
            peak_app_bytes: GIB,
            stale_publications: 0,
            validation_errors: 0,
        },
    };
    for index in 0_usize..600 {
        let id = u64::try_from(index).unwrap() + 1;
        let at = event_time(index);
        trace.inputs.push(InputSample {
            event_id: id,
            scheduled_ns: Some(at),
            occurred_ns: Some(at),
            handled_ns: Some(at + MS),
        });
        trace.requests.push(RequestSample {
            request_id: id,
            generation: id,
            event_id: id,
            quality: Quality::Proxy,
            extent: [1024, 1024],
            requested_ns: Some(at + MS),
            submitted_ns: index.is_multiple_of(5).then_some(at + 2 * MS),
            completed_ns: index.is_multiple_of(5).then_some(at + 4 * MS),
        });
        trace.frames.push(FrameSample {
            update_start_ns: Some(at + 5 * MS),
            update_end_ns: Some(at + 6 * MS),
            rendered_ns: Some(at + 10 * MS),
            request_id: Some(id - u64::try_from(index % 5).unwrap()),
        });
    }
    trace.requests.push(RequestSample {
        request_id: 601,
        generation: 600,
        event_id: 600,
        quality: Quality::Full,
        extent: [4096, 4096],
        requested_ns: Some(END),
        submitted_ns: Some(END + 2 * MS),
        completed_ns: Some(END + 10 * MS),
    });
    trace.frames.push(FrameSample {
        update_start_ns: Some(END + 18 * MS),
        update_end_ns: Some(END + 19 * MS),
        rendered_ns: Some(END + 20 * MS),
        request_id: Some(601),
    });
    for index in 0..100_u64 {
        let at = END + (100 + index * 20) * MS;
        trace.normal_dispatches.push(NormalSample {
            extent: [4096, 4096],
            warmed: true,
            normal_only: true,
            submitted_ns: Some(at),
            completed_ns: Some(at + 3 * MS),
            gpu_duration_ns: Some(MS),
        });
    }
    trace
}

fn assert_violation(trace: &RunTrace, expected: Violation) -> GateReport {
    let report = evaluate(trace);
    assert!(!report.passed(), "unexpected PASS: {report:?}");
    assert!(
        report.violations.contains(&expected),
        "{expected:?}: {report:?}"
    );
    assert_eq!(report.exit_code(), ExitCode::FAILURE);
    report
}

fn assert_no_violation(trace: &RunTrace, unexpected: Violation) {
    let report = evaluate(trace);
    assert!(
        !report.violations.contains(&unexpected),
        "unexpected {unexpected:?}: {report:?}"
    );
}

// Assign a desired rendered cadence and move the associated synthetic event/work
// boundaries together. No real clock, sleeping, GPU or fixture image is involved.
fn set_intervals(trace: &mut RunTrace, intervals: &[u64]) {
    assert_eq!(intervals.len(), 599);
    let mut at = START;
    for index in 0_usize..600 {
        if index > 0 {
            at += intervals[index - 1];
        }
        let input = &mut trace.inputs[index];
        input.scheduled_ns = Some(at);
        input.occurred_ns = Some(at);
        input.handled_ns = Some(at + MS);
        let request = &mut trace.requests[index];
        request.requested_ns = Some(at + MS);
        if request.submitted_ns.is_some() {
            request.submitted_ns = Some(at + 2 * MS);
            request.completed_ns = Some(at + 4 * MS);
        }
        let frame = &mut trace.frames[index];
        frame.update_start_ns = Some(at + 5 * MS);
        frame.update_end_ns = Some(at + 6 * MS);
        frame.rendered_ns = Some(at + 10 * MS);
    }
    assert!(at + 10 * MS < END);
}

fn publish_every(trace: &mut RunTrace, stride: usize) {
    for index in 0_usize..600 {
        let at = trace.inputs[index].occurred_ns.unwrap();
        let fresh = index.is_multiple_of(stride);
        trace.requests[index].submitted_ns = fresh.then_some(at + 2 * MS);
        trace.requests[index].completed_ns = fresh.then_some(at + 4 * MS);
        trace.frames[index].request_id = Some(u64::try_from(index - index % stride).unwrap() + 1);
    }
}

fn delay_handling(trace: &mut RunTrace, count: usize, delay_ns: u64) {
    trace.drag_start_ns = Some(START - delay_ns);
    for input in &mut trace.inputs[..count] {
        input.scheduled_ns = input.scheduled_ns.map(|ns| ns - delay_ns);
        input.occurred_ns = input.occurred_ns.map(|ns| ns - delay_ns);
    }
}

#[test]
fn perf_metrics_percentile_is_nearest_rank_on_unsorted_samples() {
    let samples = [100, 1, 3, 2, 4];
    assert_eq!(nearest_rank(&samples, 50), Some(3));
    assert_eq!(nearest_rank(&samples, 95), Some(100));
    assert_eq!(nearest_rank(&samples, 99), Some(100));
    assert_eq!(nearest_rank(&samples, 100), Some(100));
    assert_eq!(nearest_rank(&samples, 1), Some(1));
    assert_eq!(samples, [100, 1, 3, 2, 4]);
    let hundred: Vec<u64> = (1..=100).rev().collect();
    assert_eq!(nearest_rank(&hundred, 95), Some(95));
    assert_eq!(nearest_rank(&hundred, 99), Some(99));
}

#[test]
fn perf_metrics_percentile_handles_empty_singleton_ties_and_invalid_percentiles() {
    assert_eq!(nearest_rank(&[], 95), None);
    assert_eq!(nearest_rank(&[7], 95), Some(7));
    assert_eq!(nearest_rank(&[0, 0, 0], 99), Some(0));
    assert_eq!(nearest_rank(&[1], 0), None);
    assert_eq!(nearest_rank(&[1], 101), None);
    assert_eq!(nearest_rank(&[u64::MAX, 0], 95), Some(u64::MAX));
}

#[test]
fn perf_metrics_complete_trace_passes_with_expected_metrics_and_success_exit() {
    let trace = passing_trace();
    let report = evaluate(&trace);
    assert!(report.passed(), "{report:?}");
    assert!(report.violations.is_empty());
    assert_eq!(report.exit_code(), ExitCode::SUCCESS);
    assert_eq!(report.metrics.drag_frames, 600);
    assert_eq!(report.metrics.fresh_publications, 120);
    assert_eq!(report.metrics.ui_update_p95_ns, Some(MS));
    assert_eq!(report.metrics.frame_interval_p95_ns, Some(16_666_667));
    assert_eq!(report.metrics.frame_interval_p99_ns, Some(16_666_667));
    assert_eq!(report.metrics.frame_interval_max_ns, Some(16_666_667));
    assert_eq!(report.metrics.proxy_age_p95_ns, Some(10 * MS));
    assert_eq!(report.metrics.fresh_gap_max_ns, Some(83_333_334));
    assert_eq!(report.metrics.release_to_visible_ns, Some(20 * MS));
    assert_eq!(report.metrics.normal_p95_ns, Some(MS));
    assert_eq!(report.metrics.normal_completion_p95_ns, Some(3 * MS));
    assert_eq!(report.metrics.normal_timing, NormalTiming::GpuTimestamp);
    assert_eq!(report.cold_start, ColdStartStatus::NotMeasured);
    assert_eq!(evaluate(&trace), report, "evaluation must be deterministic");
}

#[test]
fn perf_metrics_empty_and_incomplete_traces_cannot_pass() {
    let mut trace = passing_trace();
    trace.frames.clear();
    trace.inputs.clear();
    trace.requests.clear();
    trace.normal_dispatches.clear();
    let report = assert_violation(&trace, Violation::Incomplete);
    assert_eq!(report.metrics.ui_update_p95_ns, None);
    assert_eq!(report.metrics.frame_interval_p95_ns, None);
    assert_eq!(report.metrics.proxy_age_p95_ns, None);
    assert_eq!(report.metrics.normal_p95_ns, None);
    let mut trace = passing_trace();
    trace.completed = false;
    assert_violation(&trace, Violation::Incomplete);
    trace.completed = true;
    trace.dropped_samples = 1;
    assert_violation(&trace, Violation::Incomplete);
}

#[test]
fn perf_metrics_short_active_window_is_incomplete_even_with_enough_samples() {
    let mut trace = passing_trace();
    trace.drag_end_ns = Some(END - 1);
    assert_violation(&trace, Violation::Incomplete);
}

#[test]
fn perf_metrics_minimum_drag_frames_are_inclusive() {
    let mut trace = passing_trace();
    // Delete repeated frames only; keep all fresh publications and full visibility.
    let mut removed = 0;
    let mut index = 0_usize;
    trace.frames.retain(|_| {
        let repeated = index < 600 && !index.is_multiple_of(5);
        index += 1;
        if repeated && removed < 100 {
            removed += 1;
            false
        } else {
            true
        }
    });
    let report = evaluate(&trace);
    assert_eq!(report.metrics.drag_frames, 500);
    assert!(!report.violations.contains(&Violation::DragFrames));
    // The last drag frame is a repeat, not a fresh publication.
    trace.frames.remove(499);
    assert_violation(&trace, Violation::DragFrames);
}

#[test]
fn perf_metrics_fresh_publication_minimum_and_gap_equality_are_inclusive() {
    let mut trace = passing_trace();
    publish_every(&mut trace, 6);
    let report = evaluate(&trace);
    assert!(report.passed(), "{report:?}");
    assert_eq!(report.metrics.fresh_publications, 100);
    assert_eq!(report.metrics.fresh_gap_max_ns, Some(100 * MS));
    trace.frames[6].rendered_ns = trace.frames[6].rendered_ns.map(|ns| ns + 1);
    let report = assert_violation(&trace, Violation::PublicationGap);
    assert_eq!(report.metrics.fresh_gap_max_ns, Some(100 * MS + 1));
    trace.frames[6].rendered_ns = trace.frames[6].rendered_ns.map(|ns| ns - 1);
    trace.frames[594..600]
        .iter_mut()
        .for_each(|frame| frame.request_id = Some(589));
    assert_violation(&trace, Violation::FreshPublications);
}

#[test]
fn perf_metrics_frozen_but_fast_viewport_fails_freshness() {
    let mut trace = passing_trace();
    for frame in &mut trace.frames[..600] {
        frame.request_id = Some(1);
    }
    let report = assert_violation(&trace, Violation::FreshPublications);
    assert_eq!(report.metrics.fresh_publications, 1);
    assert_eq!(report.metrics.ui_update_p95_ns, Some(MS));
    assert_eq!(report.metrics.frame_interval_p95_ns, Some(16_666_667));
    assert!(report.violations.contains(&Violation::PublicationGap));
}

#[test]
fn perf_metrics_publication_age_uses_original_event_not_request_or_completion_time() {
    let mut trace = passing_trace();
    delay_handling(&mut trace, 600, 60 * MS);
    let report = assert_violation(&trace, Violation::ProxyAge);
    assert_eq!(report.metrics.proxy_age_p95_ns, Some(70 * MS));
}

#[test]
fn perf_metrics_age_is_first_render_using_texture_not_worker_completion() {
    let mut trace = passing_trace();
    delay_handling(&mut trace, 600, 41 * MS);
    // Completion age is 45 ms; visible age is 51 ms and must fail.
    let report = assert_violation(&trace, Violation::ProxyAge);
    assert_eq!(report.metrics.proxy_age_p95_ns, Some(51 * MS));
}

#[test]
fn perf_metrics_proxy_age_threshold_equality_and_one_ns_over() {
    let mut trace = passing_trace();
    delay_handling(&mut trace, 600, 40 * MS);
    assert_no_violation(&trace, Violation::ProxyAge);
    trace.drag_start_ns = trace.drag_start_ns.map(|ns| ns - 1);
    for input in &mut trace.inputs {
        input.scheduled_ns = input.scheduled_ns.map(|ns| ns - 1);
        input.occurred_ns = input.occurred_ns.map(|ns| ns - 1);
    }
    assert_violation(&trace, Violation::ProxyAge);
}

#[test]
fn perf_metrics_repeated_texture_frames_do_not_recompute_age_or_inflate_publications() {
    let trace = passing_trace();
    let report = evaluate(&trace);
    assert_eq!(report.metrics.fresh_publications, 120);
    assert_eq!(report.metrics.proxy_age_p95_ns, Some(10 * MS));
    assert!(report.passed(), "{report:?}");
}

#[test]
fn perf_metrics_coalesced_requests_are_not_zero_latency_publications() {
    let mut trace = passing_trace();
    // 480 requests were replaced before submission. Keeping their input/intent
    // records must not dilute the 120 real publication ages with zeroes.
    delay_handling(&mut trace, 40, 60 * MS);
    let report = assert_violation(&trace, Violation::ProxyAge);
    assert_eq!(report.metrics.fresh_publications, 120);
    assert_eq!(report.metrics.proxy_age_p95_ns, Some(70 * MS));
}

#[test]
fn perf_metrics_stale_counter_is_failure_even_if_timing_passes() {
    let mut trace = passing_trace();
    trace.counters.stale_publications = 1;
    assert_violation(&trace, Violation::StalePublication);
}

#[test]
fn perf_metrics_request_or_generation_regression_is_stale_even_if_counter_is_zero() {
    let mut trace = passing_trace();
    trace.frames[10].request_id = Some(1);
    assert_violation(&trace, Violation::StalePublication);
    let mut trace = passing_trace();
    trace.requests[10].generation = 1;
    assert_violation(&trace, Violation::StalePublication);
}

#[test]
fn perf_metrics_distinct_request_ids_at_same_generation_are_fresh() {
    let mut trace = passing_trace();
    for request in &mut trace.requests {
        request.generation = 1;
    }
    let report = evaluate(&trace);
    assert!(report.passed(), "{report:?}");
    assert_eq!(report.metrics.fresh_publications, 120);
    assert_eq!(report.metrics.release_to_visible_ns, Some(20 * MS));
}

#[test]
fn perf_metrics_internal_and_boundary_publication_gaps_are_enforced() {
    for range in [0..12, 240..252, 588..600] {
        let mut trace = passing_trace();
        let previous = (range.start > 0).then(|| trace.frames[range.start - 1].request_id.unwrap());
        for frame in &mut trace.frames[range] {
            frame.request_id = previous;
        }
        assert_violation(&trace, Violation::PublicationGap);
    }
}

#[test]
fn perf_metrics_ui_update_threshold_is_inclusive_and_does_not_drop_slow_samples() {
    let mut trace = passing_trace();
    for frame in &mut trace.frames[..600] {
        frame.update_end_ns = frame.update_start_ns.map(|ns| ns + 4 * MS);
    }
    assert!(evaluate(&trace).passed());
    for frame in &mut trace.frames[..31] {
        frame.update_end_ns = frame.update_end_ns.map(|ns| ns + 1);
    }
    let report = assert_violation(&trace, Violation::UiUpdate);
    assert_eq!(report.metrics.ui_update_p95_ns, Some(4 * MS + 1));
}

#[test]
fn perf_metrics_rendered_cadence_is_independent_of_fast_cpu_updates() {
    let mut trace = passing_trace();
    let mut intervals = vec![16 * MS; 599];
    for interval in intervals.iter_mut().take(31) {
        *interval = 21 * MS;
    }
    set_intervals(&mut trace, &intervals);
    let report = assert_violation(&trace, Violation::FrameP95);
    assert_eq!(report.metrics.ui_update_p95_ns, Some(MS));
    assert_eq!(report.metrics.frame_interval_p95_ns, Some(21 * MS));
}

#[test]
fn perf_metrics_frame_interval_thresholds_use_ns_and_inclusive_comparisons() {
    for (slow, count, violation) in [
        (20 * MS, 30, Violation::FrameP95),
        (33_400_000, 6, Violation::FrameP99),
        (100 * MS, 1, Violation::FrameMax),
    ] {
        let mut trace = passing_trace();
        let mut intervals = vec![16 * MS; 599];
        intervals[..count].fill(slow);
        set_intervals(&mut trace, &intervals);
        assert_no_violation(&trace, violation);
        intervals[..count].fill(slow + 1);
        set_intervals(&mut trace, &intervals);
        assert_violation(&trace, violation);
    }
}

#[test]
fn perf_metrics_single_long_frame_fails_max_even_when_percentiles_pass() {
    let mut trace = passing_trace();
    let mut intervals = vec![16 * MS; 599];
    intervals[300] = 101 * MS;
    set_intervals(&mut trace, &intervals);
    let report = assert_violation(&trace, Violation::FrameMax);
    assert!(!report.violations.contains(&Violation::FrameP95));
    assert!(!report.violations.contains(&Violation::FrameP99));
}

#[test]
fn perf_metrics_warmup_and_post_drag_frames_do_not_pollute_active_metrics() {
    let mut trace = passing_trace();
    for (start, end) in [(0, START - 1), (END, END + 1_000 * MS)] {
        trace.frames.push(FrameSample {
            update_start_ns: Some(start),
            update_end_ns: Some(end),
            rendered_ns: Some(end),
            request_id: None,
        });
    }
    trace.frames.sort_by_key(|frame| frame.rendered_ns);
    let report = evaluate(&trace);
    assert!(report.passed(), "{report:?}");
    assert_eq!(report.metrics.drag_frames, 600);
    assert_eq!(report.metrics.ui_update_p95_ns, Some(MS));
    assert_eq!(report.metrics.frame_interval_max_ns, Some(16_666_667));
}

#[test]
fn perf_metrics_full_latency_uses_release_to_first_render_not_completion_or_later_repeat() {
    let mut trace = passing_trace();
    let mut later = trace.frames[600].clone();
    later.update_start_ns = Some(END + 498 * MS);
    later.update_end_ns = Some(END + 499 * MS);
    later.rendered_ns = Some(END + 500 * MS);
    trace.frames.push(later);
    assert_eq!(
        evaluate(&trace).metrics.release_to_visible_ns,
        Some(20 * MS)
    );
    trace.frames[600].rendered_ns = Some(END + 250 * MS);
    assert!(evaluate(&trace).passed());
    trace.frames[600].rendered_ns = Some(END + 250 * MS + 1);
    let report = assert_violation(&trace, Violation::ReleaseVisible);
    assert_eq!(report.metrics.release_to_visible_ns, Some(250 * MS + 1));
}

#[test]
fn perf_metrics_completed_full_work_without_a_render_is_incomplete() {
    let mut trace = passing_trace();
    trace.frames.pop();
    let report = assert_violation(&trace, Violation::Incomplete);
    assert_eq!(report.metrics.release_to_visible_ns, None);
}

#[test]
fn perf_metrics_proxy_or_wrong_extent_cannot_satisfy_full_visibility() {
    let mut trace = passing_trace();
    trace.requests[600].quality = Quality::Proxy;
    assert_violation(&trace, Violation::InvalidTrace);
    let mut trace = passing_trace();
    trace.requests[600].extent = [1024, 1024];
    assert_violation(&trace, Violation::InvalidTrace);
}

#[test]
fn perf_metrics_missing_timestamps_never_become_zero_latency() {
    // Exercise every boundary used by a passing result, one missing field at a time.
    let cases: Vec<TraceMutation> = vec![
        |t| t.drag_start_ns = None,
        |t| t.drag_end_ns = None,
        |t| t.release_ns = None,
        |t| t.full_request_id = None,
        |t| t.inputs[0].scheduled_ns = None,
        |t| t.inputs[0].occurred_ns = None,
        |t| t.inputs[0].handled_ns = None,
        |t| t.requests[0].requested_ns = None,
        |t| t.requests[0].submitted_ns = None,
        |t| t.requests[0].completed_ns = None,
        |t| t.frames[0].update_start_ns = None,
        |t| t.frames[0].update_end_ns = None,
        |t| t.frames[0].rendered_ns = None,
        |t| t.requests[600].completed_ns = None,
        |t| t.normal_dispatches[0].submitted_ns = None,
        |t| t.normal_dispatches[0].completed_ns = None,
        |t| t.normal_dispatches[0].gpu_duration_ns = None,
    ];
    for (index, mutate) in cases.iter().enumerate() {
        let mut trace = passing_trace();
        mutate(&mut trace);
        let report = evaluate(&trace);
        assert!(
            !report.passed(),
            "missing timestamp case {index}: {report:?}"
        );
        assert!(
            report.violations.contains(&Violation::Incomplete),
            "case {index}: {report:?}"
        );
        assert_eq!(report.exit_code(), ExitCode::FAILURE);
    }
}

#[test]
fn perf_metrics_unknown_and_duplicate_identities_are_invalid() {
    let cases: Vec<TraceMutation> = vec![
        |t| t.frames[0].request_id = Some(999_999),
        |t| t.requests[0].event_id = 999_999,
        |t| t.inputs[1].event_id = t.inputs[0].event_id,
        |t| t.requests[1].request_id = t.requests[0].request_id,
    ];
    for mutate in cases {
        let mut trace = passing_trace();
        mutate(&mut trace);
        assert_violation(&trace, Violation::InvalidTrace);
    }
}

#[test]
fn perf_metrics_reversed_or_future_timestamps_are_invalid_not_saturating_subtractions() {
    let cases: Vec<TraceMutation> = vec![
        |t| t.drag_end_ns = Some(START - 1),
        |t| t.inputs[0].handled_ns = Some(START - 1),
        |t| t.requests[0].submitted_ns = Some(START),
        |t| t.requests[0].completed_ns = Some(START + MS),
        |t| t.frames[0].update_end_ns = Some(START),
        |t| t.frames[0].rendered_ns = Some(START + 3 * MS),
        |t| t.frames[1].rendered_ns = Some(START + 5 * MS),
        |t| t.frames[1].rendered_ns = t.frames[0].rendered_ns,
        |t| t.release_ns = Some(END + 30 * MS),
        |t| t.normal_dispatches[0].completed_ns = Some(0),
    ];
    for mutate in cases {
        let mut trace = passing_trace();
        mutate(&mut trace);
        assert_violation(&trace, Violation::InvalidTrace);
    }
}

#[test]
fn perf_metrics_normal_p95_is_strictly_less_than_16_ms() {
    let mut trace = passing_trace();
    for sample in &mut trace.normal_dispatches {
        sample.completed_ns = sample.submitted_ns.map(|ns| ns + 20 * MS);
        sample.gpu_duration_ns = Some(16 * MS - 1);
    }
    let report = evaluate(&trace);
    assert!(
        report.passed(),
        "GPU timestamps, not completion upper bound: {report:?}"
    );
    assert_eq!(report.metrics.normal_completion_p95_ns, Some(20 * MS));
    for sample in &mut trace.normal_dispatches {
        sample.gpu_duration_ns = Some(16 * MS);
    }
    assert_violation(&trace, Violation::NormalP95);
}

#[test]
fn perf_metrics_without_timestamps_requires_and_labels_conservative_completion_latency() {
    let mut trace = passing_trace();
    trace.gpu_timestamps_available = false;
    for sample in &mut trace.normal_dispatches {
        sample.gpu_duration_ns = None;
        sample.completed_ns = sample.submitted_ns.map(|ns| ns + 16 * MS - 1);
    }
    let report = evaluate(&trace);
    assert!(report.passed(), "{report:?}");
    assert_eq!(
        report.metrics.normal_timing,
        NormalTiming::CompletionUpperBound
    );
    assert_eq!(report.metrics.normal_p95_ns, Some(16 * MS - 1));
    for sample in &mut trace.normal_dispatches {
        sample.completed_ns = sample.submitted_ns.map(|ns| ns + 16 * MS);
    }
    assert_violation(&trace, Violation::NormalP95);
}

#[test]
fn perf_metrics_normal_benchmark_requires_100_warm_4k_normal_only_samples() {
    let cases: Vec<TraceMutation> = vec![
        |t| {
            t.normal_dispatches.pop();
        },
        |t| t.normal_dispatches[0].warmed = false,
        |t| t.normal_dispatches[0].normal_only = false,
        |t| t.normal_dispatches[0].extent = [1024, 1024],
    ];
    for mutate in cases {
        let mut trace = passing_trace();
        mutate(&mut trace);
        assert_violation(&trace, Violation::WarmNormalSamples);
    }
}

#[test]
fn perf_metrics_all_in_window_warm_dispatches_count_without_outlier_removal() {
    let mut trace = passing_trace();
    for sample in &mut trace.normal_dispatches[..6] {
        sample.gpu_duration_ns = Some(17 * MS);
        sample.completed_ns = sample.submitted_ns.map(|ns| ns + 18 * MS);
    }
    let report = assert_violation(&trace, Violation::NormalP95);
    assert_eq!(report.metrics.normal_p95_ns, Some(17 * MS));
}

#[test]
fn perf_metrics_cold_normal_warmup_does_not_pollute_the_warm_benchmark() {
    let mut trace = passing_trace();
    trace.normal_dispatches.insert(
        0,
        NormalSample {
            extent: [4096, 4096],
            warmed: false,
            normal_only: true,
            submitted_ns: Some(0),
            completed_ns: Some(500 * MS),
            gpu_duration_ns: Some(400 * MS),
        },
    );
    let report = evaluate(&trace);
    assert!(report.passed(), "{report:?}");
    assert_eq!(report.metrics.normal_p95_ns, Some(MS));
    assert_eq!(report.metrics.normal_completion_p95_ns, Some(3 * MS));
}

#[test]
fn perf_metrics_extra_slow_warm_dispatches_cannot_be_truncated_to_the_first_100() {
    let mut trace = passing_trace();
    for index in 0..6_u64 {
        let at = END + (3_000 + index * 20) * MS;
        trace.normal_dispatches.push(NormalSample {
            extent: [4096, 4096],
            warmed: true,
            normal_only: true,
            submitted_ns: Some(at),
            completed_ns: Some(at + 18 * MS),
            gpu_duration_ns: Some(17 * MS),
        });
    }
    let report = assert_violation(&trace, Violation::NormalP95);
    assert_eq!(report.metrics.normal_p95_ns, Some(17 * MS));
}

#[test]
fn perf_metrics_unavailable_gpu_timestamps_cannot_override_slow_completion_latency() {
    let mut trace = passing_trace();
    trace.gpu_timestamps_available = false;
    for sample in &mut trace.normal_dispatches {
        sample.completed_ns = sample.submitted_ns.map(|ns| ns + 20 * MS);
    }
    let report = assert_violation(&trace, Violation::NormalP95);
    assert_eq!(report.metrics.normal_p95_ns, Some(20 * MS));
    assert_eq!(
        report.metrics.normal_timing,
        NormalTiming::CompletionUpperBound
    );
}

#[test]
fn perf_metrics_memory_and_pending_work_bounds_and_validation_are_hard_failures() {
    let trace = passing_trace();
    assert!(
        evaluate(&trace).passed(),
        "one GiB and one pending/in-flight are legal"
    );
    let cases: Vec<(TraceMutation, Violation)> = vec![
        (
            |t| t.counters.peak_app_bytes = GIB + 1,
            Violation::AppAllocation,
        ),
        (|t| t.counters.peak_pending = 2, Violation::PendingWork),
        (|t| t.counters.peak_in_flight = 2, Violation::InFlightWork),
        (|t| t.counters.validation_errors = 1, Violation::Validation),
    ];
    for (mutate, expected) in cases {
        let mut trace = passing_trace();
        mutate(&mut trace);
        assert_violation(&trace, expected);
    }
}

#[test]
fn perf_metrics_three_individual_runs_must_pass_no_average_or_replacement() {
    let pass = evaluate(&passing_trace());
    assert_eq!(
        suite_exit_code(&[pass.clone(), pass.clone(), pass.clone()], 3),
        ExitCode::SUCCESS
    );
    assert_eq!(
        suite_exit_code(&[pass.clone(), pass.clone()], 3),
        ExitCode::FAILURE
    );
    assert_eq!(suite_exit_code(&[], 3), ExitCode::FAILURE);
    assert_eq!(suite_exit_code(&[], 0), ExitCode::FAILURE);
    let mut slow = passing_trace();
    slow.frames[600].rendered_ns = Some(END + 251 * MS);
    let fail = evaluate(&slow);
    for index in 0..3 {
        let mut runs = vec![pass.clone(); 3];
        runs[index] = fail.clone();
        assert_eq!(suite_exit_code(&runs, 3), ExitCode::FAILURE);
        assert!(!runs[index].passed());
    }
    assert_eq!(
        suite_exit_code(&[pass.clone(), pass.clone(), fail, pass], 3),
        ExitCode::FAILURE
    );
}

#[test]
fn perf_metrics_cold_start_is_strict_separate_and_missing_is_not_measured() {
    for (duration, status) in [
        (None, ColdStartStatus::NotMeasured),
        (Some(1_000 * MS - 1), ColdStartStatus::Pass),
        (Some(1_000 * MS), ColdStartStatus::Fail),
        (Some(2_000 * MS), ColdStartStatus::Fail),
    ] {
        let mut trace = passing_trace();
        trace.cold_start_ns = duration;
        let report = evaluate(&trace);
        assert_eq!(report.cold_start, status);
        assert!(
            report.passed(),
            "cold start is separate from slider: {report:?}"
        );
    }
}

fn parse(args: &[&str]) -> Result<Config, String> {
    Config::parse(args.iter().map(|arg| (*arg).to_owned()))
}

#[test]
fn perf_metrics_cli_accepts_the_independent_scenario_command() {
    let config = parse(&[
        "--perf-scenario",
        "normal-drag",
        "--size",
        "4096",
        "--duration-seconds",
        "10",
        "--backend",
        "dx12",
        "--adapter-name",
        "RTX 5070 Ti",
        "--report",
        "single-run.json",
    ])
    .expect("the documented scenario command must parse without an --input argument");
    assert_eq!(config.adapter_name.as_deref(), Some("RTX 5070 Ti"));
    assert_eq!(config.backend, "dx12");
    assert!(config.report.unwrap().ends_with("single-run.json"));
}

#[test]
fn perf_metrics_cli_rejects_ambiguous_or_invalid_scenarios() {
    for args in [
        vec!["--perf-scenario"],
        vec!["--perf-scenario", "typo"],
        vec!["--perf-scenario", "normal-drag", "--size", "0"],
        vec!["--perf-scenario", "normal-drag", "--duration-seconds", "0"],
        vec![
            "--perf-scenario",
            "normal-drag",
            "--duration-seconds",
            "NaN",
        ],
        vec!["--perf-scenario", "normal-drag", "--render-smoke"],
        vec!["--perf-scenario", "normal-drag", "--command-smoke"],
        vec![
            "--perf-scenario",
            "normal-drag",
            "--ui-snapshot",
            "captures",
        ],
        vec!["--perf-scenario", "normal-drag", "--integration-smoke"],
    ] {
        assert!(parse(&args).is_err(), "{args:?}");
    }
}
