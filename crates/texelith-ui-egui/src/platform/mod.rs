//! Native startup policy. Backend overrides never enable a renderer fallback.

pub(crate) const fn default_backend() -> &'static str {
    if cfg!(target_os = "windows") {
        "dx12"
    } else if cfg!(target_os = "macos") {
        "metal"
    } else {
        "vulkan"
    }
}

pub(crate) fn backend_from_environment() -> Result<Option<String>, String> {
    std::env::var_os("WGPU_BACKEND")
        .map(|value| {
            value
                .into_string()
                .map_err(|_| "unsupported backend: WGPU_BACKEND must be Unicode text".into())
        })
        .transpose()
}
