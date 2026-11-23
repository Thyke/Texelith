use std::time::Duration;
use texelith_ui_egui::measurement::{Lifecycle, SHUTDOWN_LIMIT, STARTUP_LIMIT, WindowSample};

fn exposed(at_ns: u64) -> WindowSample {
    WindowSample {
        at_ns,
        window_found: true,
        foreground: Some(true),
        minimized: Some(false),
        visible: Some(true),
        occluded: Some(false),
        refresh_hz: Some(160),
        foreground_request_accepted: Some(true),
    }
}

#[test]
fn hidden_bootstrap_is_allowed_but_an_unpresented_run_is_invalid() {
    let mut lifecycle = Lifecycle::default();
    lifecycle.sample(WindowSample {
        visible: Some(false),
        ..exposed(100_000_000)
    });
    assert!(lifecycle.environment_reasons.is_empty());
    assert!(
        lifecycle
            .final_reasons()
            .iter()
            .any(|r| r.contains("never presented"))
    );
    lifecycle.phase("first_render_completed", 200_000_000);
    lifecycle.sample(exposed(300_000_000));
    assert!(lifecycle.final_reasons().is_empty());
}

#[test]
fn each_unpresentable_state_is_retained_after_recovery() {
    for bad in [
        WindowSample {
            minimized: Some(true),
            ..exposed(3_000_000_000)
        },
        WindowSample {
            visible: Some(false),
            ..exposed(3_000_000_000)
        },
        WindowSample {
            occluded: Some(true),
            ..exposed(3_000_000_000)
        },
        WindowSample {
            foreground: Some(false),
            ..exposed(3_000_000_000)
        },
    ] {
        let mut lifecycle = Lifecycle::default();
        lifecycle.phase("first_render_completed", 1);
        lifecycle.sample(bad);
        lifecycle.sample(exposed(4_000_000_000));
        assert_eq!(lifecycle.final_reasons().len(), 1);
        assert_eq!(
            lifecycle.deadline(Duration::from_secs(4), 10),
            Some(("measurement window is not presentable", false))
        );
    }
}

#[test]
fn shutdown_window_destruction_does_not_invalidate_a_valid_run() {
    let mut lifecycle = Lifecycle::default();
    lifecycle.phase("first_render_completed", 1);
    lifecycle.sample(exposed(3_000_000_000));
    lifecycle.phase("shutdown_started", 4_000_000_000);
    lifecycle.sample(WindowSample {
        visible: Some(false),
        foreground: Some(false),
        ..exposed(5_000_000_000)
    });
    assert!(lifecycle.final_reasons().is_empty());
}

#[test]
fn startup_and_shutdown_deadlines_do_not_need_ui_or_gpu_ticks() {
    let mut lifecycle = Lifecycle::default();
    assert_eq!(
        lifecycle.deadline(STARTUP_LIMIT - Duration::from_nanos(1), 10),
        None
    );
    assert_eq!(
        lifecycle.deadline(STARTUP_LIMIT, 10),
        Some(("startup deadline exceeded", false))
    );
    lifecycle.phase("shutdown_started", STARTUP_LIMIT.as_nanos() as u64);
    assert_eq!(
        lifecycle.deadline(STARTUP_LIMIT + SHUTDOWN_LIMIT - Duration::from_nanos(1), 10),
        None
    );
    assert_eq!(
        lifecycle.deadline(STARTUP_LIMIT + SHUTDOWN_LIMIT, 10),
        Some(("shutdown deadline exceeded", true))
    );
}

#[test]
fn a_presenting_but_slow_startup_is_an_application_timeout() {
    let mut lifecycle = Lifecycle::default();
    lifecycle.phase("first_render_completed", 1);
    lifecycle.sample(exposed(3_000_000_000));
    assert!(lifecycle.final_reasons().is_empty());
    assert_eq!(
        lifecycle.deadline(STARTUP_LIMIT, 10),
        Some(("startup deadline exceeded", false))
    );
    lifecycle.phase("first_usable_frame", 5);
    assert_eq!(lifecycle.deadline(STARTUP_LIMIT, 10), None);
    assert_eq!(
        lifecycle.deadline(Duration::from_secs(100), 10),
        Some(("scenario deadline exceeded", false))
    );
}

#[test]
fn phase_timestamps_keep_the_first_observation() {
    let mut lifecycle = Lifecycle::default();
    lifecycle.phase("first_logic_tick", 12);
    lifecycle.phase("first_logic_tick", 40);
    assert_eq!(lifecycle.phases_ns["first_logic_tick"], 12);
}
