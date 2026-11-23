#[path = "common/gpu_availability.rs"]
mod gpu_availability;

#[test]
fn process_overrides_skip_hardware_checks() {
    if std::env::var_os("TEXELITH_GPU_GUARD_CHILD").is_some() {
        assert!(gpu_availability::skip_if_no_gpu());
        return;
    }
    // Isolated processes avoid mutating the environment of concurrent tests.
    for variable in ["TEXELITH_CI_NO_GPU", "TEXELITH_FORCE_NO_GPU"] {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "process_overrides_skip_hardware_checks",
                "--nocapture",
            ])
            .env("TEXELITH_CI_NO_GPU", "0")
            .env("TEXELITH_FORCE_NO_GPU", "0")
            .env(variable, "1")
            .env("TEXELITH_GPU_GUARD_CHILD", "1")
            .output()
            .unwrap();
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(output.status.success(), "{variable}: {stderr}");
        assert!(
            stderr.contains("SKIPPED (no GPU adapter)"),
            "{variable}: {stderr}"
        );
    }
}
