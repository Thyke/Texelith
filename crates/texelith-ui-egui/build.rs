use std::{env, fs, path::PathBuf, process::Command};

fn main() {
    println!("cargo:rerun-if-changed=../../packaging/windows/texelith.ico");
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows")
        || env::var("CARGO_CFG_TARGET_ENV").as_deref() != Ok("msvc")
    {
        return;
    }
    let icon = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap())
        .join("../../packaging/windows/texelith.ico");
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let rc = out.join("texelith.rc");
    let res = out.join("texelith.res");
    let version = env::var("CARGO_PKG_VERSION").unwrap();
    let numbers = version.split('.').collect::<Vec<_>>().join(",");
    let source = format!(
        r#"1 ICON "{}"
1 VERSIONINFO
FILEVERSION {},0
PRODUCTVERSION {},0
FILEOS 0x40004
FILETYPE 1
BEGIN
  BLOCK "StringFileInfo"
  BEGIN
    BLOCK "040904b0"
    BEGIN
      VALUE "FileDescription", "Texelith\0"
      VALUE "FileVersion", "{}\0"
      VALUE "ProductName", "Texelith\0"
      VALUE "ProductVersion", "{}\0"
    END
  END
  BLOCK "VarFileInfo"
  BEGIN
    VALUE "Translation", 0x0409, 1200
  END
END
"#,
        icon.display().to_string().replace('\\', "\\\\"),
        numbers,
        numbers,
        version,
        version
    );
    fs::write(&rc, source).unwrap();
    // Cargo normally supplies the Windows SDK tool path through the MSVC environment.
    let sdk_bin = PathBuf::from(env::var_os("ProgramFiles(x86)").unwrap_or_default())
        .join("Windows Kits/10/bin");
    let rc_exe = fs::read_dir(sdk_bin)
        .ok()
        .into_iter()
        .flatten()
        .flatten()
        .map(|entry| entry.path().join("x64/rc.exe"))
        .filter(|path| path.is_file())
        .max()
        .unwrap_or_else(|| PathBuf::from("rc.exe"));
    let status = Command::new(rc_exe)
        .arg("/nologo")
        .arg(format!("/fo{}", res.display()))
        .arg(&rc)
        .status()
        .expect("Windows SDK rc.exe is required to build Texelith");
    assert!(status.success(), "Windows resource compilation failed");
    println!(
        "cargo:rustc-link-arg-bin=texelith-ui-egui={}",
        res.display()
    );
}
