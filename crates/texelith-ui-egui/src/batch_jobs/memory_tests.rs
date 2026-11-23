use super::*;
use std::time::Duration;

#[test]
fn reservations_charge_bytes_and_release_at_phase_end() {
    let budget = Arc::new(Budget::default());
    let first = budget.reserve(LIMIT / 2, false).unwrap();
    let second = budget.reserve(LIMIT / 2, true).unwrap();
    assert_eq!(budget.state.lock().unwrap().used, LIMIT);
    assert!(budget.reserve(LIMIT + 1, true).is_err());
    drop(first);
    drop(second);
    assert_eq!(budget.state.lock().unwrap().used, 0);
}

#[test]
fn interactive_waiter_precedes_next_background_phase() {
    let budget = Arc::new(Budget::default());
    let owner = budget.reserve(LIMIT, false).unwrap();
    let (send, receive) = std::sync::mpsc::channel();
    let foreground = budget.clone();
    let interactive = std::thread::spawn(move || {
        let _permit = foreground.reserve(LIMIT, true).unwrap();
        send.send(()).unwrap();
    });
    while budget.state.lock().unwrap().interactive_waiters == 0 {
        std::thread::yield_now();
    }
    let background = budget.clone();
    let next = std::thread::spawn(move || {
        let _permit = background.reserve(LIMIT, false).unwrap();
        receive.recv_timeout(Duration::from_secs(5)).unwrap();
    });
    drop(owner);
    interactive.join().unwrap();
    next.join().unwrap();
}

#[test]
fn four_k_codec_waits_for_capacity_and_can_cancel_admission() {
    let budget = Arc::new(Budget::default());
    let presentation = budget.reserve(PRESENTATION_BUDGET, true).unwrap();
    let bytes =
        super::super::export_phase_reservation(4096, 4096, 1, texelith_core::ExportFormat::Png16)
            .unwrap();
    assert!(bytes <= LIMIT && bytes + PRESENTATION_BUDGET > LIMIT);
    let (send, receive) = std::sync::mpsc::channel();
    let worker = budget.clone();
    let thread = std::thread::spawn(move || {
        let _encoding = worker.reserve(bytes, false).unwrap();
        send.send(()).unwrap();
    });
    assert!(receive.recv_timeout(Duration::from_millis(100)).is_err());
    drop(presentation);
    receive.recv_timeout(Duration::from_secs(5)).unwrap();
    thread.join().unwrap();
    assert!(budget.state.lock().unwrap().peak <= LIMIT);
    let owner = budget.reserve(LIMIT, true).unwrap();
    assert!(budget.reserve_while(bytes, false, || false).is_err());
    drop(owner);
}
