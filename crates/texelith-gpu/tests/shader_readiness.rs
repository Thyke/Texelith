/// This red test is adapter-free. Numerical shader validation is in verify_nodes.
#[test]
fn required_adr_compute_passes_exist_before_gpu_verification_can_pass() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("shaders");
    let missing: Vec<_> = [
        "source.wgsl",
        "downsample.wgsl",
        "levels.wgsl",
        "blur.wgsl",
        "normal.wgsl",
        "display.wgsl",
    ]
    .into_iter()
    .filter(|name| std::fs::read_to_string(root.join(name)).map_or(true, |s| s.trim().is_empty()))
    .collect();
    assert!(
        missing.is_empty(),
        "T-015 implementation pending; missing compute passes: {missing:?}"
    );
}
