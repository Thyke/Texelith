use serde::Serialize;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

#[derive(Clone, Debug)]
pub enum Event {
    Accepted,
    Waiting(String),
    Readback(String),
    Encoding(String),
    Encoded(String),
    Published,
}

#[derive(Default, Debug)]
pub struct Trace {
    events: Vec<(u64, Event)>,
}

#[derive(Debug, Serialize)]
pub struct OutputTiming {
    pub name: String,
    pub evaluation_readback_ns: u64,
    pub cpu_encode_ns: u64,
}
#[derive(Debug, Serialize)]
pub struct Metric {
    pub request_to_disk_ns: u64,
    pub outputs: Vec<OutputTiming>,
}

impl Trace {
    pub fn record(&mut self, ns: u64, event: Event) -> Result<(), String> {
        if self
            .events
            .last()
            .is_some_and(|(last, e)| ns < *last || matches!(e, Event::Published))
        {
            return Err("Nonmonotonic or already published trace".into());
        }
        if self.events.is_empty() != matches!(event, Event::Accepted) {
            return Err("Trace requires exactly one acceptance event".into());
        }
        self.events.push((ns, event));
        Ok(())
    }

    pub fn summary(&self, completed: bool) -> Result<Metric, String> {
        if !completed {
            return Err("Export failed or cancelled".into());
        }
        let Some((start, Event::Accepted)) = self.events.first() else {
            return Err("Missing immutable job acceptance".into());
        };
        let Some((end, Event::Published)) = self.events.last() else {
            return Err("Missing final disk publication".into());
        };
        let mut outputs = Vec::new();
        let (events, remainder) = self.events[1..self.events.len() - 1].as_chunks::<4>();
        for chunk in events {
            let [
                (waiting, Event::Waiting(a)),
                (ready, Event::Readback(b)),
                (encoding, Event::Encoding(c)),
                (encoded, Event::Encoded(d)),
            ] = chunk
            else {
                return Err("Incomplete output phase sequence".into());
            };
            if a != b || b != c || c != d {
                return Err("Output phase identity mismatch".into());
            }
            outputs.push(OutputTiming {
                name: a.clone(),
                evaluation_readback_ns: ready - waiting,
                cpu_encode_ns: encoded - encoding,
            });
        }
        if outputs.is_empty() || !remainder.is_empty() {
            return Err("Incomplete output metrics".into());
        }
        Ok(Metric {
            request_to_disk_ns: end - start,
            outputs,
        })
    }
}

struct Active {
    clock: Instant,
    trace: Trace,
    error: Option<String>,
}
static ACTIVE: OnceLock<Mutex<Option<Active>>> = OnceLock::new();

pub(super) fn arm() -> Result<(), String> {
    let mut active = ACTIVE.get_or_init(Default::default).lock().unwrap();
    if active.is_some() {
        return Err("Performance trace already active".into());
    }
    *active = Some(Active {
        clock: Instant::now(),
        trace: Trace::default(),
        error: None,
    });
    Ok(())
}

/// Inert unless the diagnostic explicitly arms a trace. Called on the real job path.
pub(crate) fn mark(event: Event) {
    if let Some(active) = ACTIVE.get() {
        let mut guard = active.lock().unwrap();
        if let Some(active) = guard.as_mut() {
            let ns = active.clock.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64;
            if let Err(error) = active.trace.record(ns, event) {
                active.error = Some(error);
            }
        }
    }
}

pub(super) fn finish(completed: bool) -> Result<Metric, String> {
    let active = ACTIVE
        .get_or_init(Default::default)
        .lock()
        .unwrap()
        .take()
        .ok_or("No trace")?;
    if let Some(error) = active.error {
        return Err(error);
    }
    active.trace.summary(completed)
}
