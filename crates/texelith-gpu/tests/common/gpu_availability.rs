//! Shared hardware-test guard. Production adapter selection is unchanged.
use std::{
    future::Future,
    sync::{Arc, OnceLock},
    task::{Context, Poll, Wake, Waker},
};

/// Return before hardware setup when CI disables GPU checks or no native adapter exists.
/// Device creation and shader failures after a successful probe must still fail tests.
pub fn skip_if_no_gpu() -> bool {
    let disabled = ["TEXELITH_CI_NO_GPU", "TEXELITH_FORCE_NO_GPU"]
        .iter()
        .any(|key| std::env::var(key).as_deref() == Ok("1"));
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    if disabled || !AVAILABLE.get_or_init(probe) {
        eprintln!(
            "SKIPPED (no GPU adapter): {}",
            std::thread::current().name().unwrap_or("hardware test")
        );
        return true;
    }
    false
}

fn probe() -> bool {
    let backends = if cfg!(target_os = "windows") {
        wgpu::Backends::DX12
    } else if cfg!(target_os = "macos") {
        wgpu::Backends::METAL
    } else {
        wgpu::Backends::VULKAN
    };
    let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    descriptor.backends = backends;
    let instance = wgpu::Instance::new(descriptor);
    block_on(instance.enumerate_adapters(backends))
        .iter()
        .any(|adapter| adapter.get_info().device_type != wgpu::DeviceType::Cpu)
}

// UI tests do not depend on pollster; share this tiny executor across both crates.
fn block_on<T>(future: impl Future<Output = T>) -> T {
    struct ThreadWake(std::thread::Thread);
    impl Wake for ThreadWake {
        fn wake(self: Arc<Self>) {
            self.0.unpark();
        }
        fn wake_by_ref(self: &Arc<Self>) {
            self.0.unpark();
        }
    }
    let waker = Waker::from(Arc::new(ThreadWake(std::thread::current())));
    let mut context = Context::from_waker(&waker);
    let mut future = std::pin::pin!(future);
    loop {
        match future.as_mut().poll(&mut context) {
            Poll::Ready(value) => return value,
            Poll::Pending => std::thread::park(),
        }
    }
}
