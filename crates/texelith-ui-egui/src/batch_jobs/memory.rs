//! Shared phase scratch; retained sources, display images and the single export
//! plane use the other 256 MiB. Export evaluation/publication own no scratch.
use std::sync::{Arc, Condvar, LazyLock, Mutex};

const LIMIT: u64 = super::CPU_BUDGET - super::INTERACTIVE_RESERVATION;
/// Tiled 4K display/heatmap fits here independently of codec/decode estimates.
pub const PRESENTATION_BUDGET: u64 = 64 * 1024 * 1024;
#[derive(Default)]
struct State {
    used: u64,
    interactive_waiters: usize,
    background_waiters: usize,
    peak: u64,
}
#[derive(Default)]
struct Budget {
    state: Mutex<State>,
    available: Condvar,
}
static BUDGET: LazyLock<Arc<Budget>> = LazyLock::new(|| Arc::new(Budget::default()));
pub(crate) struct Permit {
    budget: Arc<Budget>,
    bytes: u64,
}
impl Budget {
    fn reserve(self: &Arc<Self>, bytes: u64, interactive: bool) -> Result<Permit, String> {
        self.reserve_while(bytes, interactive, || true)
    }
    fn reserve_while(
        self: &Arc<Self>,
        bytes: u64,
        interactive: bool,
        current: impl Fn() -> bool,
    ) -> Result<Permit, String> {
        if bytes > LIMIT {
            return Err("CPU scratch exceeds shared budget; use a smaller proxy".into());
        }
        let mut state = self.state.lock().unwrap();
        if interactive {
            state.interactive_waiters += 1;
        } else {
            state.background_waiters += 1;
        }
        while bytes > LIMIT - state.used || (!interactive && state.interactive_waiters != 0) {
            if !current() {
                if interactive {
                    state.interactive_waiters -= 1;
                } else {
                    state.background_waiters -= 1;
                }
                self.available.notify_all();
                return Err("CPU admission cancelled".into());
            }
            state = self
                .available
                .wait_timeout(state, std::time::Duration::from_millis(20))
                .unwrap()
                .0;
        }
        if interactive {
            state.interactive_waiters -= 1;
        } else {
            state.background_waiters -= 1;
        }
        state.used += bytes;
        state.peak = state.peak.max(state.used);
        Ok(Permit {
            budget: self.clone(),
            bytes,
        })
    }
}
/// Only workers wait; frame callbacks never acquire phase scratch.
pub(crate) fn reserve() -> Permit {
    BUDGET.reserve(LIMIT, true).unwrap()
}
pub(crate) fn reserve_background() -> Permit {
    BUDGET.reserve(LIMIT, false).unwrap()
}
pub(crate) fn reserve_encode(bytes: u64, current: impl Fn() -> bool) -> Result<Permit, String> {
    BUDGET.reserve_while(bytes, false, current)
}
pub(crate) fn peak_bytes() -> u64 {
    super::INTERACTIVE_RESERVATION + scratch_peak_bytes()
}
pub(crate) fn scratch_peak_bytes() -> u64 {
    BUDGET.state.lock().unwrap().peak
}
#[cfg(test)]
pub(crate) fn background_waiting() -> bool {
    BUDGET.state.lock().unwrap().background_waiters > 0
}

/// Tiled readback retains only display samples. Save converts directly to PNG
/// samples; full-resolution Vec<[f32; 4]> allocations are never made here.
pub fn presentation_reservation(
    width: u32,
    height: u32,
    heatmap: bool,
    save: bool,
) -> Result<u64, String> {
    if width == 0 || height == 0 || width > 4096 || height > 4096 {
        return Err("Presentation extent outside supported 1..4096 range".into());
    }
    let [display_width, display_height] = super::presentation::display_size(width, height);
    let display = u64::from(display_width) * u64::from(display_height);
    let checked = || -> Option<u64> {
        let numeric = display.checked_mul(16)?;
        let mapped = (u64::from(width) * 8)
            .div_ceil(256)
            .checked_mul(256)?
            .checked_mul(u64::from(height.min(super::presentation::TILE_ROWS)))?;
        let mut peak = numeric.checked_add(mapped)?;
        let retained = numeric.checked_mul(if heatmap { 2 } else { 1 })?;
        if heatmap {
            peak = peak.max(retained.checked_add(mapped)?);
        }
        // Keep the tile allowance through conversion and round to MiB so the
        // ledger also covers small driver mapping/collection overheads.
        peak = peak.max(
            retained
                .checked_add(display.checked_mul(7)?)?
                .checked_add(mapped)?,
        );
        if save {
            // Save owns the converted input too; it is not a retained export plane.
            let height_save = super::png_scratch(width, height, 2, 3).ok()?;
            let color_save = super::png_scratch(width, height, 3, 1).ok()?;
            // Save precedes display sampling; its scratch does not overlap the
            // sampled float vectors or the heatmap display conversion.
            peak = peak.max(height_save.max(color_save));
        }
        Some(peak.div_ceil(1024 * 1024) * (1024 * 1024))
    };
    let bytes = checked().ok_or("Presentation size overflow")?;
    if bytes > if save { LIMIT } else { PRESENTATION_BUDGET } {
        return Err("Presentation exceeds shared CPU budget; use a smaller proxy".into());
    }
    Ok(bytes)
}
impl Permit {
    /// Once encoding finishes, only the encoded Vec stays live during disk IO.
    pub(crate) fn retain(&mut self, bytes: u64) {
        assert!(bytes <= self.bytes);
        self.budget.state.lock().unwrap().used -= self.bytes - bytes;
        self.bytes = bytes;
        self.budget.available.notify_all();
    }
}
pub(crate) fn reserve_presentation(
    width: u32,
    height: u32,
    heatmap: bool,
    save: bool,
) -> Result<Permit, String> {
    BUDGET.reserve(
        presentation_reservation(width, height, heatmap, save)?,
        true,
    )
}
impl Drop for Permit {
    fn drop(&mut self) {
        self.budget.state.lock().unwrap().used -= self.bytes;
        self.budget.available.notify_all();
    }
}

#[cfg(test)]
#[path = "memory_tests.rs"]
mod tests;
