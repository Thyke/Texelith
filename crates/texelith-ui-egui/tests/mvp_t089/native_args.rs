use std::ffi::OsString;

use texelith_ui_egui::startup::Config;

#[cfg(any(unix, windows))]
fn opaque_path() -> OsString {
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStringExt;
        OsString::from_vec(b"textures/native-\xff.png".to_vec())
    }
    #[cfg(windows)]
    {
        use std::os::windows::ffi::OsStringExt;
        OsString::from_wide(&[0x74, 0x65, 0x78, 0xd800, 0x2e, 0x70, 0x6e, 0x67])
    }
}

#[cfg(any(unix, windows))]
#[test]
fn native_parser_preserves_opaque_paths_for_every_filesystem_option() {
    let path = opaque_path();
    let config = Config::parse_os([
        "--input".into(),
        path.clone(),
        "--integration-smoke".into(),
        "--save-normal".into(),
        path.clone(),
        "--report".into(),
        path.clone(),
        "--backend".into(),
        "vulkan".into(),
    ])
    .unwrap();
    for parsed in [config.input, config.save_normal, config.report] {
        assert_eq!(parsed.unwrap().into_os_string(), path);
    }
    let config = Config::parse_os([
        "--ui-snapshot".into(),
        path.clone(),
        "--backend".into(),
        "vulkan".into(),
    ])
    .unwrap();
    assert_eq!(config.ui_snapshot.unwrap().into_os_string(), path);
}

#[cfg(any(unix, windows))]
#[test]
fn non_unicode_text_options_return_errors_without_panicking() {
    for flag in ["--backend", "--adapter-name", "--scale", "--frames"] {
        let error = Config::parse_os([flag.into(), opaque_path()]).unwrap_err();
        assert!(error.contains(flag), "{error}");
        assert!(error.contains("Unicode"), "{error}");
    }
    let error = Config::parse_os([opaque_path()]).unwrap_err();
    assert!(error.contains("argument names"), "{error}");
}
