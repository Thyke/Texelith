use std::{
    fs,
    io::{self, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};
use texelith_io::export_transactions::{
    CollisionPolicy, ExportFile, FailurePoint, NameContext, PublishOptions, expand_name, preflight,
    publish_set,
};

// Proposed T-077 public API: preflight takes fully expanded output filenames;
// publish_set invokes the bounded encoder once per file and the checkpoint after
// the named operation. An error at a checkpoint simulates an I/O failure/cancel.
static NEXT: AtomicU64 = AtomicU64::new(0);
struct Sandbox(PathBuf);
impl Sandbox {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "texelith-t077-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn path(&self) -> &Path {
        &self.0
    }
}
impl Drop for Sandbox {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn context<'a>(name: &'a str, map: &'a str, preset: &'a str, index: u32) -> NameContext<'a> {
    NameContext {
        name,
        map,
        preset,
        index,
    }
}
fn entries() -> Vec<ExportFile> {
    vec![
        ExportFile::new("T_rock_Albedo.png"),
        ExportFile::new("T_rock_Normal.png"),
    ]
}
fn encode(file: &ExportFile, out: &mut dyn Write) -> io::Result<()> {
    out.write_all(if file.name() == "T_rock_Albedo.png" {
        b"new albedo"
    } else {
        b"new normal"
    })
}
fn original_set(root: &Path) -> PathBuf {
    let dest = root.join("Rock");
    fs::create_dir(&dest).unwrap();
    fs::write(dest.join("T_rock_Albedo.png"), b"old albedo").unwrap();
    fs::write(dest.join("T_rock_Normal.png"), b"old normal").unwrap();
    dest
}
fn assert_old(dest: &Path) {
    assert_eq!(
        fs::read(dest.join("T_rock_Albedo.png")).unwrap(),
        b"old albedo"
    );
    assert_eq!(
        fs::read(dest.join("T_rock_Normal.png")).unwrap(),
        b"old normal"
    );
}
fn names(root: &Path) -> Vec<String> {
    let mut result: Vec<_> = fs::read_dir(root)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    result.sort();
    result
}

#[test]
fn expands_all_tokens_and_preserves_unicode_and_spaces() {
    let c = context("Kaya yüzeyi", "Normal", "unreal-v1", 7);
    assert_eq!(
        expand_name("T_{name}_{map}", &c).unwrap(),
        "T_Kaya yüzeyi_Normal"
    );
    assert_eq!(
        expand_name("{preset} {index} {name}", &c).unwrap(),
        "unreal-v1 0007 Kaya yüzeyi"
    );
    assert_eq!(
        expand_name("{index}", &context("x", "y", "z", 12345)).unwrap(),
        "12345"
    );
}

#[test]
fn rejects_nonportable_names_after_expansion_and_invalid_templates() {
    for invalid in [
        "",
        ".",
        "..",
        "../escape",
        "a/b",
        "a\\b",
        "C:drive",
        "a<z",
        "a\u{0001}",
        "name.",
        "name ",
        "CON",
        "con.txt",
        "NUL",
        "COM1",
        "lpt9.png",
    ] {
        assert!(
            expand_name("{name}", &context(invalid, "normal", "p", 0)).is_err(),
            "accepted {invalid:?}"
        );
    }
    for template in ["{unknown}", "{name", "name}", "{name}/x", "../{name}"] {
        assert!(
            expand_name(template, &context("rock", "normal", "p", 0)).is_err(),
            "accepted {template:?}"
        );
    }
}

#[test]
fn preflight_rejects_portable_collisions_and_duplicate_batch_names() {
    let root = Sandbox::new();
    assert!(preflight(root.path(), &["Rock.png", "rock.PNG"]).is_err());
    assert!(preflight(root.path(), &["same.png", "same.png"]).is_err());
    fs::write(root.path().join("Existing.PNG"), b"keep").unwrap();
    assert!(preflight(root.path(), &["existing.png"]).is_err());
    assert_eq!(fs::read(root.path().join("Existing.PNG")).unwrap(), b"keep");
    let a = expand_name("{name}_{index}", &context("same", "a", "p", 1)).unwrap();
    let b = expand_name("{name}_{index}", &context("same", "a", "p", 2)).unwrap();
    assert_ne!(a, b);
    assert!(preflight(root.path(), &[a.as_str(), b.as_str()]).is_ok());
}

#[test]
fn new_set_is_published_as_a_complete_set_and_default_never_overwrites() {
    let root = Sandbox::new();
    let dest = root.path().join("Rock");
    publish_set(&dest, &entries(), PublishOptions::default(), encode, |_| {
        Ok(())
    })
    .unwrap();
    assert_eq!(
        fs::read(dest.join("T_rock_Albedo.png")).unwrap(),
        b"new albedo"
    );
    assert_eq!(
        fs::read(dest.join("T_rock_Normal.png")).unwrap(),
        b"new normal"
    );
    assert_eq!(names(root.path()), ["Rock"]);
    assert!(
        publish_set(
            &dest,
            &entries(),
            PublishOptions::default(),
            encode,
            |_| Ok(())
        )
        .is_err()
    );
    assert_eq!(
        fs::read(dest.join("T_rock_Albedo.png")).unwrap(),
        b"new albedo"
    );
    assert_eq!(names(root.path()), ["Rock"]);
}

#[test]
fn explicit_replacement_publishes_all_new_files() {
    let root = Sandbox::new();
    let dest = original_set(root.path());
    let options = PublishOptions {
        collision: CollisionPolicy::Replace,
        ..Default::default()
    };
    publish_set(&dest, &entries(), options, encode, |_| Ok(())).unwrap();
    assert_eq!(
        fs::read(dest.join("T_rock_Albedo.png")).unwrap(),
        b"new albedo"
    );
    assert_eq!(
        fs::read(dest.join("T_rock_Normal.png")).unwrap(),
        b"new normal"
    );
    assert_eq!(names(root.path()), ["Rock"]);
}

#[test]
fn injected_failures_and_cancel_restore_old_set_and_only_remove_owned_staging() {
    for point in [
        FailurePoint::AfterEncode(0),
        FailurePoint::AfterFlush(0),
        FailurePoint::AfterEncode(1),
        FailurePoint::AfterFlush(1),
        FailurePoint::AfterBackupRename,
        FailurePoint::AfterPublishRename,
    ] {
        let root = Sandbox::new();
        let dest = original_set(root.path());
        fs::create_dir(root.path().join("unrelated-staging")).unwrap();
        fs::write(root.path().join("unrelated-staging/keep"), b"untouched").unwrap();
        let options = PublishOptions {
            collision: CollisionPolicy::Replace,
            ..Default::default()
        };
        let mut hit = false;
        let result = publish_set(&dest, &entries(), options, encode, |actual| {
            if actual == point {
                hit = true;
                Err(io::Error::new(
                    io::ErrorKind::Interrupted,
                    "injected cancel",
                ))
            } else {
                Ok(())
            }
        });
        assert!(hit, "checkpoint {point:?} was not reached");
        assert!(result.is_err(), "checkpoint {point:?} reported success");
        assert_old(&dest);
        assert_eq!(
            fs::read(root.path().join("unrelated-staging/keep")).unwrap(),
            b"untouched"
        );
        assert_eq!(names(root.path()), ["Rock", "unrelated-staging"]);
    }
}

#[test]
fn encoder_error_never_publishes_a_partial_new_set() {
    let root = Sandbox::new();
    let dest = root.path().join("Rock");
    let result = publish_set(
        &dest,
        &entries(),
        PublishOptions::default(),
        |_, out| {
            out.write_all(b"partial")?;
            Err(io::Error::other("encoder failed"))
        },
        |_| Ok(()),
    );
    assert!(result.is_err());
    assert!(!dest.exists());
    assert!(names(root.path()).is_empty());
}

#[path = "mvp_t077/regressions.rs"]
mod regressions;
