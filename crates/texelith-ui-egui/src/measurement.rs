//! Measurement-only lifecycle supervision, independent of UI and GPU progress.
use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

pub const STARTUP_LIMIT: Duration = Duration::from_secs(30);
pub const SHUTDOWN_LIMIT: Duration = Duration::from_secs(15);
const FOREGROUND_GRACE_NS: u64 = 2_000_000_000;

#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct WindowSample {
    pub at_ns: u64,
    pub window_found: bool,
    pub foreground: Option<bool>,
    pub minimized: Option<bool>,
    pub visible: Option<bool>,
    pub occluded: Option<bool>,
    pub refresh_hz: Option<u32>,
    pub foreground_request_accepted: Option<bool>,
}

#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct Lifecycle {
    pub phases_ns: BTreeMap<String, u64>,
    pub window_samples: Vec<WindowSample>,
    pub environment_reasons: BTreeSet<String>,
    pub stop_reason: Option<String>,
    pub forced_exit: bool,
    pub last_logic_tick_ns: Option<u64>,
}

impl Lifecycle {
    pub fn phase(&mut self, name: &str, at_ns: u64) {
        if !self.phases_ns.contains_key(name) {
            self.phases_ns.insert(name.into(), at_ns);
        }
    }

    pub fn sample(&mut self, sample: WindowSample) {
        // eframe intentionally starts hidden. Allow one initial foreground attempt
        // to settle; never forgive occlusion after this acquisition interval.
        if (sample.at_ns >= FOREGROUND_GRACE_NS
            || self.phases_ns.contains_key("first_usable_frame"))
            && !self.phases_ns.contains_key("shutdown_started")
        {
            for (invalid, reason) in [
                (sample.minimized == Some(true), "window minimized"),
                (sample.visible == Some(false), "window hidden"),
                (sample.occluded == Some(true), "window occluded"),
                (sample.foreground == Some(false), "window not foreground"),
                (!sample.window_found, "measurement window not found"),
                (
                    sample.window_found
                        && (sample.foreground.is_none() || sample.occluded.is_none()),
                    "window state observation unavailable",
                ),
            ] {
                if invalid {
                    self.environment_reasons.insert(reason.into());
                }
            }
        }
        // Total scenario deadline is at most 150 seconds; 100 ms sampling fits.
        if self.window_samples.len() < 2_000 {
            self.window_samples.push(sample);
        } else {
            self.environment_reasons
                .insert("window sample overflow".into());
        }
    }

    pub fn final_reasons(&self) -> BTreeSet<String> {
        let mut reasons = self.environment_reasons.clone();
        if !self.phases_ns.contains_key("first_render_completed") {
            reasons.insert("window never presented a completed frame".into());
        }
        if !self
            .window_samples
            .iter()
            .any(|s| s.window_found && s.foreground == Some(true) && s.occluded == Some(false))
        {
            reasons.insert("presentable foreground window never observed".into());
        }
        reasons
    }

    /// Returns the next stop request and whether the hard shutdown deadline passed.
    pub fn deadline(&self, now: Duration, duration_seconds: u64) -> Option<(&'static str, bool)> {
        let ns = now.as_nanos() as u64;
        if let Some(start) = self.phases_ns.get("shutdown_started") {
            return (ns.saturating_sub(*start) >= SHUTDOWN_LIMIT.as_nanos() as u64)
                .then_some(("shutdown deadline exceeded", true));
        }
        if !self.environment_reasons.is_empty() {
            return Some(("measurement window is not presentable", false));
        }
        if now >= STARTUP_LIMIT && !self.phases_ns.contains_key("first_usable_frame") {
            return Some(("startup deadline exceeded", false));
        }
        (now >= Duration::from_secs(duration_seconds + 90))
            .then_some(("scenario deadline exceeded", false))
    }
}

pub(crate) struct Monitor {
    origin: Instant,
    state: Mutex<Lifecycle>,
    wake_stop: AtomicBool,
    wake_thread: Mutex<Option<std::thread::JoinHandle<()>>>,
}
pub(crate) type Shared = Arc<Monitor>;

impl Monitor {
    pub fn phase(&self, name: &str) {
        self.state.lock().unwrap().phase(name, self.ns());
    }
    fn ns(&self) -> u64 {
        self.origin.elapsed().as_nanos() as u64
    }
    pub fn snapshot(&self) -> Lifecycle {
        self.state.lock().unwrap().clone()
    }
    pub fn logic_tick(&self) {
        let mut state = self.state.lock().unwrap();
        state.phase("first_logic_tick", self.ns());
        state.last_logic_tick_ns = Some(self.ns());
    }
    pub fn attach(self: &Arc<Self>, ctx: &egui::Context) -> Result<(), std::io::Error> {
        let monitor = Arc::downgrade(self);
        let ctx = ctx.clone();
        // egui methods take internal locks. Isolate wake requests from the hard
        // watchdog so even an event-loop/context lock stall cannot disable it.
        let thread = std::thread::Builder::new()
            .name("texelith-measure-wake".into())
            .spawn(move || {
                while let Some(monitor) = monitor.upgrade() {
                    if monitor.wake_stop.load(Ordering::Acquire) {
                        break;
                    }
                    if monitor.closing() {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                    ctx.request_repaint();
                    std::thread::sleep(Duration::from_millis(100));
                }
            })?;
        *self.wake_thread.lock().unwrap() = Some(thread);
        Ok(())
    }
    pub fn closing(&self) -> bool {
        self.state.lock().unwrap().stop_reason.is_some()
    }
}

pub(crate) struct Watchdog {
    pub monitor: Shared,
    stop: crossbeam_channel::Sender<()>,
    thread: Option<std::thread::JoinHandle<()>>,
    path: PathBuf,
}

fn write_snapshot(path: &Path, lifecycle: &Lifecycle) -> Result<(), String> {
    let reasons = lifecycle.final_reasons();
    let value = serde_json::json!({
        "schema": 1, "lifecycle": lifecycle, "present_mode": "Fifo",
        "environment_status": if reasons.is_empty() { "VALID" } else { "ENVIRONMENT_INVALID" },
        "environment_reasons": reasons,
        "startup_limit_seconds": STARTUP_LIMIT.as_secs(),
        "shutdown_limit_seconds": SHUTDOWN_LIMIT.as_secs(),
        "observation": "100 ms Win32 window samples; 3x3 client-area point coverage, visibility, minimization and DWM cloaking; not a compositor/scanout or GPU utilization measurement",
    });
    let temporary = path.with_extension("json.tmp");
    std::fs::write(
        &temporary,
        serde_json::to_vec_pretty(&value).map_err(|e| e.to_string())?,
    )
    .and_then(|()| std::fs::rename(temporary, path))
    .map_err(|e| e.to_string())
}

fn write_aborted_report(path: &Path, lifecycle: &Lifecycle) -> std::io::Result<()> {
    use std::io::Write;
    let reasons = lifecycle.final_reasons();
    let report = serde_json::json!({
        "schema": 2, "scenario": "normal-drag", "passed": false,
        "status": if reasons.is_empty() { "FAIL" } else { "ENVIRONMENT_INVALID" },
        "environment_reasons": reasons, "lifecycle": lifecycle,
        "error": lifecycle.stop_reason, "gate": null, "trace": null,
        "measurement": "aborted by independent watchdog; numeric samples unavailable; see lifecycle sidecar",
    });
    // Never overwrite an already recorded numeric gate, including a failure.
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    file.write_all(&serde_json::to_vec_pretty(&report)?)
}

impl Watchdog {
    pub fn start(report: &Path, duration_seconds: u64) -> Result<Self, String> {
        if let Some(parent) = report.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        let path = report.with_extension("lifecycle.json");
        let monitor = Arc::new(Monitor {
            origin: Instant::now(),
            state: Mutex::new(Lifecycle::default()),
            wake_stop: AtomicBool::new(false),
            wake_thread: Mutex::new(None),
        });
        monitor.phase("app_entry");
        write_snapshot(&path, &monitor.snapshot())?;
        let (stop, receiver) = crossbeam_channel::bounded(1);
        let worker = monitor.clone();
        let worker_path = path.clone();
        let worker_report = report.to_owned();
        let thread = std::thread::Builder::new()
            .name("texelith-measure-watchdog".into())
            .spawn(move || {
                let mut attempted = false;
                let mut last_write = Instant::now();
                loop {
                    if receiver.recv_timeout(Duration::from_millis(100))
                        != Err(crossbeam_channel::RecvTimeoutError::Timeout)
                    {
                        break;
                    }
                    let closing = worker
                        .state
                        .lock()
                        .unwrap()
                        .phases_ns
                        .contains_key("shutdown_started");
                    if !closing {
                        let mut sample = native::sample(&mut attempted);
                        sample.at_ns = worker.ns();
                        worker.state.lock().unwrap().sample(sample);
                    }
                    let decision = worker
                        .state
                        .lock()
                        .unwrap()
                        .deadline(worker.origin.elapsed(), duration_seconds);
                    if let Some((reason, force)) = decision {
                        {
                            let mut state = worker.state.lock().unwrap();
                            state.stop_reason.get_or_insert_with(|| reason.into());
                            state.phase("shutdown_started", worker.ns());
                            state.forced_exit = force;
                        }
                        if force {
                            worker.phase("watchdog_forced_exit");
                            let lifecycle = worker.snapshot();
                            let _ = write_snapshot(&worker_path, &lifecycle);
                            let _ = write_aborted_report(&worker_report, &lifecycle);
                            // A blocked driver/event-loop/join cannot be safely unwound.
                            // Measurement processes fail closed, without waiting on GPU locks.
                            std::process::exit(2);
                        }
                    }
                    if last_write.elapsed() >= Duration::from_secs(1) || decision.is_some() {
                        if let Err(error) = write_snapshot(&worker_path, &worker.snapshot()) {
                            eprintln!("Lifecycle report: {error}");
                        }
                        last_write = Instant::now();
                    }
                }
            })
            .map_err(|e| e.to_string())?;
        Ok(Self {
            monitor,
            stop,
            thread: Some(thread),
            path,
        })
    }

    pub fn finish(mut self) -> Result<(), String> {
        self.monitor.wake_stop.store(true, Ordering::Release);
        // Supervision remains active if the wake helper is stuck in an egui lock.
        if let Some(thread) = self.monitor.wake_thread.lock().unwrap().take() {
            thread
                .join()
                .map_err(|_| "measurement wake helper panicked")?;
        }
        let _ = self.stop.send(());
        if let Some(thread) = self.thread.take() {
            thread.join().map_err(|_| "measurement watchdog panicked")?;
        }
        self.monitor.phase("shutdown_completed");
        write_snapshot(&self.path, &self.monitor.snapshot())
    }
}

#[cfg(not(windows))]
mod native {
    pub fn sample(_: &mut bool) -> super::WindowSample {
        super::WindowSample::default() // unsupported observation cannot pass the suite
    }
}

#[cfg(windows)]
mod native {
    use super::WindowSample;
    use std::ffi::c_void;
    type Hwnd = *mut c_void;
    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct Point {
        x: i32,
        y: i32,
    }
    #[repr(C)]
    #[derive(Default)]
    struct Rect {
        left: i32,
        top: i32,
        right: i32,
        bottom: i32,
    }
    #[repr(C)]
    struct MonitorInfo {
        size: u32,
        monitor: Rect,
        work: Rect,
        flags: u32,
        device: [u16; 32],
    }
    #[repr(C, align(4))]
    struct DisplayMode([u8; 220]);
    #[link(name = "user32")]
    unsafe extern "system" {
        fn EnumWindows(callback: unsafe extern "system" fn(Hwnd, isize) -> i32, data: isize)
        -> i32;
        fn GetWindowThreadProcessId(window: Hwnd, pid: *mut u32) -> u32;
        fn GetWindow(window: Hwnd, command: u32) -> Hwnd;
        fn GetClassNameW(window: Hwnd, name: *mut u16, capacity: i32) -> i32;
        fn GetForegroundWindow() -> Hwnd;
        fn IsIconic(window: Hwnd) -> i32;
        fn IsWindowVisible(window: Hwnd) -> i32;
        fn GetClientRect(window: Hwnd, rect: *mut Rect) -> i32;
        fn ClientToScreen(window: Hwnd, point: *mut Point) -> i32;
        fn WindowFromPoint(point: Point) -> Hwnd;
        fn GetAncestor(window: Hwnd, flags: u32) -> Hwnd;
        fn ShowWindowAsync(window: Hwnd, command: i32) -> i32;
        fn SetForegroundWindow(window: Hwnd) -> i32;
        fn SetWindowPos(
            window: Hwnd,
            after: Hwnd,
            x: i32,
            y: i32,
            cx: i32,
            cy: i32,
            flags: u32,
        ) -> i32;
        fn MonitorFromWindow(window: Hwnd, flags: u32) -> Hwnd;
        fn GetMonitorInfoW(monitor: Hwnd, info: *mut MonitorInfo) -> i32;
        fn EnumDisplaySettingsW(device: *const u16, mode: u32, value: *mut u8) -> i32;
        fn GetDC(window: Hwnd) -> Hwnd;
        fn ReleaseDC(window: Hwnd, dc: Hwnd) -> i32;
    }
    #[link(name = "gdi32")]
    unsafe extern "system" {
        fn GetDeviceCaps(dc: Hwnd, index: i32) -> i32;
    }
    #[link(name = "dwmapi")]
    unsafe extern "system" {
        fn DwmGetWindowAttribute(window: Hwnd, attribute: u32, value: *mut u32, size: u32) -> i32;
    }
    unsafe extern "system" fn find(window: Hwnd, data: isize) -> i32 {
        let mut pid = 0;
        // EnumWindows only supplies live top-level handles; data points to the
        // stack-local output for the duration of this synchronous enumeration.
        unsafe {
            GetWindowThreadProcessId(window, &mut pid);
            if pid == std::process::id() && GetWindow(window, 4).is_null() {
                // Exclude console/driver helper windows owned by this process.
                // GetClassName does not send WM_GETTEXT to a potentially blocked UI.
                let mut class = [0_u16; 64];
                let count = GetClassNameW(window, class.as_mut_ptr(), class.len() as i32);
                if count > 0
                    && class[..count as usize]
                        .iter()
                        .copied()
                        .eq("Window Class".encode_utf16())
                {
                    *(data as *mut Hwnd) = window;
                    return 0;
                }
            }
        }
        1
    }
    pub fn sample(attempted: &mut bool) -> WindowSample {
        let mut window: Hwnd = std::ptr::null_mut();
        let mut sample = WindowSample::default();
        // All buffers match Win32 ABI sizes; no handles are retained or destroyed.
        unsafe {
            EnumWindows(find, (&mut window as *mut Hwnd) as isize);
            if window.is_null() {
                return sample;
            }
            sample.window_found = true;
            if !*attempted {
                *attempted = true;
                ShowWindowAsync(window, 1); // SW_SHOWNORMAL, including hidden bootstrap
                SetWindowPos(
                    window,
                    -1_isize as Hwnd,
                    0,
                    0,
                    0,
                    0,
                    0x4000 | 0x10 | 0x2 | 0x1,
                ); // async TOPMOST, no activation/move/resize
                sample.foreground_request_accepted = Some(SetForegroundWindow(window) != 0);
            }
            sample.foreground = Some(GetForegroundWindow() == window);
            sample.minimized = Some(IsIconic(window) != 0);
            sample.visible = Some(IsWindowVisible(window) != 0);
            let mut cloaked = 0;
            if DwmGetWindowAttribute(window, 14, &mut cloaked, 4) == 0 && cloaked != 0 {
                sample.occluded = Some(true);
            } else {
                let mut rect = Rect::default();
                if GetClientRect(window, &mut rect) != 0 && rect.right > 0 && rect.bottom > 0 {
                    let mut covered = false;
                    let mut known = true;
                    for y in [1, 3, 5] {
                        for x in [1, 3, 5] {
                            let mut point = Point {
                                x: rect.right * x / 6,
                                y: rect.bottom * y / 6,
                            };
                            if ClientToScreen(window, &mut point) == 0 {
                                known = false;
                                continue;
                            }
                            let top = WindowFromPoint(point);
                            if top.is_null() {
                                known = false;
                            } else if GetAncestor(top, 2) != window {
                                covered = true;
                            }
                        }
                    }
                    sample.occluded = if covered {
                        Some(true)
                    } else if known {
                        Some(false)
                    } else {
                        None
                    };
                }
            }
            let mut info = MonitorInfo {
                size: std::mem::size_of::<MonitorInfo>() as u32,
                monitor: Rect::default(),
                work: Rect::default(),
                flags: 0,
                device: [0; 32],
            };
            if GetMonitorInfoW(MonitorFromWindow(window, 2), &mut info) != 0 {
                let mut mode = DisplayMode([0_u8; 220]); // DEVMODEW: dmSize and dmDisplayFrequency
                mode.0[68..70].copy_from_slice(&220_u16.to_ne_bytes());
                if EnumDisplaySettingsW(info.device.as_ptr(), u32::MAX, mode.0.as_mut_ptr()) != 0 {
                    let hz = u32::from_ne_bytes(mode.0[184..188].try_into().unwrap());
                    sample.refresh_hz = (hz > 1).then_some(hz);
                }
            }
            if sample.refresh_hz.is_none() {
                let dc = GetDC(window);
                if !dc.is_null() {
                    let hz = GetDeviceCaps(dc, 116);
                    sample.refresh_hz = (hz > 1).then_some(hz as u32);
                    ReleaseDC(window, dc);
                }
            }
        }
        sample
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn watchdog_blocked_shutdown_child() {
        let Some(path) = std::env::var_os("TEXELITH_TEST_WATCHDOG_REPORT") else {
            return;
        };
        let watchdog = Watchdog::start(Path::new(&path), 10).unwrap();
        watchdog.monitor.phase("shutdown_started");
        // Model a driver/join that cannot cooperate with App::logic or close.
        std::thread::sleep(Duration::from_secs(60));
        panic!("watchdog did not enforce shutdown");
    }

    #[test]
    fn independent_watchdog_terminates_a_blocked_shutdown_and_preserves_evidence() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(format!(
            "../../docs/evidence/T-120/watchdog-test-{}.json",
            std::process::id()
        ));
        let mut child = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "measurement::tests::watchdog_blocked_shutdown_child",
            ])
            .env("TEXELITH_TEST_WATCHDOG_REPORT", &path)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .unwrap();
        let started = Instant::now();
        let status = loop {
            if let Some(status) = child.try_wait().unwrap() {
                break status;
            }
            if started.elapsed() > SHUTDOWN_LIMIT + Duration::from_secs(5) {
                child.kill().unwrap();
                child.wait().unwrap();
                panic!("blocked shutdown exceeded independent watchdog bound");
            }
            std::thread::sleep(Duration::from_millis(50));
        };
        assert_eq!(status.code(), Some(2));
        let report = path.with_extension("lifecycle.json");
        let value: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&report).unwrap()).unwrap();
        assert_eq!(value["environment_status"], "ENVIRONMENT_INVALID");
        assert_eq!(value["lifecycle"]["forced_exit"], true);
        let aborted: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(aborted["status"], "ENVIRONMENT_INVALID");
        assert_eq!(aborted["passed"], false);
        assert!(started.elapsed() < SHUTDOWN_LIMIT + Duration::from_secs(5));
        std::fs::remove_file(report).unwrap();
        std::fs::remove_file(path).unwrap();
    }
}
