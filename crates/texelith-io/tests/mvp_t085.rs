//! Behavioral contract for T-085. The `project_io` names below are proposed
//! integration API; the implementer may adapt them while preserving assertions.
use serde_json::{Value, json};
use std::{
    fs, io,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};
use texelith_core::*;
use texelith_io::project_io::{
    AssetResolution, FailurePoint, ProjectIoError, RecoveryOutcome, SaveOutcome, canonical_json,
    parse_json, read_project, rebase_project, recover_json, recover_project, recover_with,
    recovery_paths, resolve_assets, save_json, save_project_with,
};

static NEXT: AtomicU64 = AtomicU64::new(0);
struct Dir(PathBuf);
impl Dir {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "texelith-t085-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Dir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn fixture() -> ProjectFile {
    let source = SourceDescriptor::from_rgba8(
        &[1, 2, 3, 255].repeat(16),
        Extent::new(4, 4).unwrap(),
        InputInterpretation::SrgbPhoto,
    )
    .unwrap();
    let graph = material_from_source(source.clone()).unwrap();
    let export = BuiltinExportPresets
        .find("unreal-v1", 1)
        .unwrap()
        .bind(&graph, &[])
        .unwrap();
    ProjectFile {
        format: "texelith.project".into(),
        schema_version: 1,
        graph: MaterialGraphDto::from(graph),
        assets: vec![AssetRef {
            id: 1,
            relative_path: "../textures/source.png".into(),
            decoded_hash: source.content_hash,
            interpretation: InputInterpretation::SrgbPhoto,
        }],
        export,
        view: ViewState::default().into(),
    }
}

#[test]
fn canonical_project_roundtrip_is_stable_and_validated() {
    let dto = fixture();
    let bytes = canonical_json(&dto).unwrap();
    assert!(bytes.ends_with(b"\n"));
    assert!(!bytes.contains(&b'\r'));
    assert!(
        std::str::from_utf8(&bytes)
            .unwrap()
            .contains("\n  \"format\"")
    );
    let loaded = read_project(&bytes).unwrap();
    assert_eq!(loaded.document(), &dto);
    assert_eq!(canonical_json(loaded.document()).unwrap(), bytes);
}

#[test]
fn bounded_json_rejects_duplicates_at_any_depth_and_oversize() {
    for input in [
        r#"{"x":1,"x":2}"#,
        r#"{"x":{"a":1,"\u0061":2}}"#,
        r#"[{"a":1,"a":2}]"#,
    ] {
        assert!(parse_json(input.as_bytes()).is_err(), "accepted {input}");
    }
    assert!(parse_json(&vec![b' '; 8 * 1024 * 1024 + 1]).is_err());
    let within = format!("{}0{}", "[".repeat(64), "]".repeat(64));
    let too_deep = format!("{}0{}", "[".repeat(65), "]".repeat(65));
    assert!(parse_json(within.as_bytes()).is_ok());
    assert!(parse_json(too_deep.as_bytes()).is_err());
}

#[test]
fn generic_json_helpers_are_usable_by_other_persistence_schemas() {
    let value = json!({"name":"preset", "nested":{"enabled":true}});
    let bytes = canonical_json(&value).unwrap();
    let decoded: Value = parse_json(&bytes).unwrap();
    assert_eq!(decoded, value);
    assert_eq!(canonical_json(&decoded).unwrap(), bytes);
}

#[test]
fn rejects_future_versions_unknown_fields_and_duplicate_project_fields() {
    let value = serde_json::to_value(fixture()).unwrap();
    let mut future = value.clone();
    future["schema_version"] = json!(2);
    assert!(read_project(&serde_json::to_vec(&future).unwrap()).is_err());
    let mut unknown = value;
    unknown["future_setting"] = json!(true);
    assert!(read_project(&serde_json::to_vec(&unknown).unwrap()).is_err());
    let encoded = String::from_utf8(canonical_json(&fixture()).unwrap()).unwrap();
    let duplicate = encoded.replacen(
        "\"schema_version\": 1",
        "\"schema_version\": 1, \"schema_version\": 1",
        1,
    );
    assert_ne!(duplicate, encoded);
    assert!(read_project(duplicate.as_bytes()).is_err());
}

#[test]
fn save_as_rebases_explicit_external_asset_without_mutating_original() {
    let dir = Dir::new();
    let old = dir.0.join("old/project.texelith.json");
    let new = dir.0.join("new/deeper/project.texelith.json");
    fs::create_dir_all(old.parent().unwrap()).unwrap();
    fs::create_dir_all(new.parent().unwrap()).unwrap();
    fs::create_dir_all(dir.0.join("textures")).unwrap();
    fs::write(dir.0.join("textures/source.png"), b"source").unwrap();
    let original = fixture();
    let rebased = rebase_project(&original, &old, &new).unwrap();
    assert_eq!(original.assets[0].relative_path, "../textures/source.png");
    assert_eq!(rebased.assets[0].relative_path, "../../textures/source.png");
    assert_eq!(
        fs::canonicalize(
            old.parent()
                .unwrap()
                .join(&original.assets[0].relative_path)
        )
        .unwrap(),
        fs::canonicalize(new.parent().unwrap().join(&rebased.assets[0].relative_path)).unwrap()
    );
}

#[test]
fn missing_and_changed_sources_are_reported_without_silent_acceptance() {
    let dir = Dir::new();
    fs::create_dir(dir.0.join("textures")).unwrap();
    fs::create_dir(dir.0.join("project")).unwrap();
    let project_path = dir.0.join("project/p.texelith.json");
    let dto = fixture();
    let missing = resolve_assets(&dto, &project_path).unwrap();
    assert!(matches!(&missing[0], AssetResolution::Missing { .. }));
    let asset = dir.0.join("textures/source.png");
    fs::write(
        &asset,
        texelith_io::encode_normal_png(4, 4, &[1, 2, 3].repeat(16)).unwrap(),
    )
    .unwrap();
    let matched = resolve_assets(&dto, &project_path).unwrap();
    assert!(matches!(&matched[0], AssetResolution::Matched { .. }));
    fs::write(
        &asset,
        texelith_io::encode_normal_png(4, 4, &[4, 5, 6].repeat(16)).unwrap(),
    )
    .unwrap();
    let changed = resolve_assets(&dto, &project_path).unwrap();
    assert!(matches!(&changed[0], AssetResolution::Changed { .. }));
    assert_eq!(dto.assets[0].decoded_hash, fixture().assets[0].decoded_hash);
}

#[test]
fn replacement_failure_restores_previous_project_and_failed_save_as_preserves_input() {
    let dir = Dir::new();
    let destination = dir.0.join("project.texelith.json");
    let old = b"previous valid contents";
    fs::write(&destination, old).unwrap();
    let dto = fixture();
    let before = dto.clone();
    let result = save_project_with(&destination, &dto, |point| {
        if point == FailurePoint::AfterBackupRename {
            Err(io::Error::other("injected Windows replacement failure"))
        } else {
            Ok(())
        }
    });
    assert!(result.is_err());
    assert_eq!(fs::read(&destination).unwrap(), old);
    assert_eq!(dto, before);
    let failed_save_as = dir.0.join("missing-parent/new.texelith.json");
    assert!(save_project_with(&failed_save_as, &dto, |_| Ok(())).is_err());
    assert!(!failed_save_as.exists());
    assert_eq!(dto, before);
    assert_eq!(fs::read(destination).unwrap(), old);
}

#[test]
fn interrupted_rollback_retains_recoverable_backup() {
    let dir = Dir::new();
    let path = dir.0.join("project.texelith.json");
    let old = canonical_json(&fixture()).unwrap();
    fs::write(&path, &old).unwrap();
    let mut changed = fixture();
    changed.view.split_fraction = 0.25;
    let result = save_project_with(&path, &changed, |point| match point {
        FailurePoint::AfterBackupRename | FailurePoint::BeforeRollbackRestore => {
            Err(io::Error::other("injected interruption"))
        }
        _ => Ok(()),
    });
    assert!(result.is_err());
    recover_project(&path).unwrap();
    assert_eq!(fs::read(&path).unwrap(), old);
    assert_eq!(
        read_project(&fs::read(&path).unwrap()).unwrap().document(),
        &fixture()
    );
    recover_project(&path).unwrap();
    assert_eq!(fs::read(&path).unwrap(), old);
}

#[test]
fn cleanup_failure_is_committed_and_recovery_keeps_the_new_project() {
    let dir = Dir::new();
    let path = dir.0.join("project.texelith.json");
    let old = canonical_json(&fixture()).unwrap();
    fs::write(&path, &old).unwrap();
    let mut changed = fixture();
    changed.view.split_fraction = 0.25;
    let new = canonical_json(&changed).unwrap();
    let result = save_project_with(&path, &changed, |point| {
        if point == FailurePoint::BeforeBackupCleanup {
            Err(io::Error::other("injected backup cleanup failure"))
        } else {
            Ok(())
        }
    })
    .unwrap();
    let SaveOutcome::CommittedWithCleanup { paths, .. } = result else {
        panic!("cleanup failure must report a committed save with a warning");
    };
    assert_eq!(fs::read(&path).unwrap(), new);
    assert_eq!(fs::read(&paths.backup).unwrap(), old);
    assert!(!paths.temporary.exists());
    assert_eq!(recover_project(&path).unwrap(), RecoveryOutcome::Unchanged);
    assert_eq!(fs::read(&path).unwrap(), new);
    assert!(!paths.backup.exists());
    assert_eq!(recover_project(&path).unwrap(), RecoveryOutcome::Unchanged);
    assert_eq!(fs::read(&path).unwrap(), new);
    assert!(matches!(
        save_project_with(&path, &fixture(), |_| Ok(())).unwrap(),
        SaveOutcome::Committed
    ));
    assert_eq!(fs::read(path).unwrap(), old);
}

#[test]
fn failed_post_publication_rolls_back_and_removes_staging() {
    let dir = Dir::new();
    let path = dir.0.join("project.texelith.json");
    let old = canonical_json(&fixture()).unwrap();
    fs::write(&path, &old).unwrap();
    let mut changed = fixture();
    changed.view.split_fraction = 0.25;
    let mut published = false;
    let result = save_project_with(&path, &changed, |point| {
        if point == FailurePoint::AfterPublishRename {
            published = true;
            assert_eq!(fs::read(&path).unwrap(), canonical_json(&changed).unwrap());
            Err(io::Error::other("injected post-publication failure"))
        } else {
            Ok(())
        }
    });
    assert!(published, "the test must reach actual file replacement");
    assert!(result.is_err());
    assert_eq!(fs::read(&path).unwrap(), old);
    let paths = recovery_paths(&path).unwrap();
    assert!(!paths.backup.exists());
    assert!(!paths.temporary.exists());
    assert!(!paths.backup_temporary.exists());
}

#[test]
fn first_save_failure_keeps_a_complete_recoverable_temp() {
    let dir = Dir::new();
    let path = dir.0.join("project.texelith.json");
    assert!(
        save_project_with(&path, &fixture(), |point| {
            if point == FailurePoint::AfterPublishRename {
                Err(io::Error::other("injected first-publication failure"))
            } else {
                Ok(())
            }
        })
        .is_err()
    );
    assert!(!path.exists());
    let paths = recovery_paths(&path).unwrap();
    let expected = canonical_json(&fixture()).unwrap();
    assert_eq!(fs::read(&paths.temporary).unwrap(), expected);
    assert_eq!(
        recover_project(&path).unwrap(),
        RecoveryOutcome::PromotedTemp
    );
    assert_eq!(fs::read(path).unwrap(), expected);
    assert!(!paths.temporary.exists());
}

#[test]
fn recovery_falls_back_when_main_is_missing_or_invalid() {
    for main in [None, Some(b"truncated".as_slice())] {
        let dir = Dir::new();
        let path = dir.0.join("project.texelith.json");
        let paths = recovery_paths(&path).unwrap();
        if let Some(bytes) = main {
            fs::write(&path, bytes).unwrap();
        }
        let old = canonical_json(&fixture()).unwrap();
        fs::write(&paths.backup, &old).unwrap();
        fs::write(&paths.temporary, b"incomplete temp").unwrap();
        assert_eq!(
            recover_project(&path).unwrap(),
            RecoveryOutcome::RestoredBackup
        );
        assert_eq!(fs::read(&path).unwrap(), old);
        assert!(!paths.backup.exists());
        assert!(!paths.temporary.exists());
        assert_eq!(recover_project(&path).unwrap(), RecoveryOutcome::Unchanged);
    }
}

#[test]
fn recovery_preserves_unsupported_main_and_all_recovery_entries() {
    for schema in [0, 2, u64::MAX] {
        for changed_shape in [false, true] {
            let dir = Dir::new();
            let path = dir.0.join("project.texelith.json");
            let paths = recovery_paths(&path).unwrap();
            let mut unsupported = serde_json::to_value(fixture()).unwrap();
            unsupported["schema_version"] = json!(schema);
            if changed_shape {
                unsupported["future_setting"] = json!({"preserve": true});
                unsupported["graph"] = json!({"new_graph_shape": true});
            }
            let main = serde_json::to_vec(&unsupported).unwrap();
            let backup = canonical_json(&fixture()).unwrap();
            let entries = [
                (&path, main.as_slice()),
                (&paths.backup, backup.as_slice()),
                (&paths.temporary, backup.as_slice()),
                (&paths.backup_temporary, b"partial backup".as_slice()),
            ];
            for (entry, bytes) in entries {
                fs::write(entry, bytes).unwrap();
            }
            assert!(matches!(
                recover_project(&path),
                Err(ProjectIoError::UnsupportedSchema {
                    found,
                    supported_min: 1,
                    supported_max: 1,
                }) if found == schema
            ));
            for (entry, bytes) in entries {
                assert_eq!(fs::read(entry).unwrap(), bytes);
            }
        }
    }
}

#[cfg(windows)]
#[test]
fn save_flushes_published_file_before_backup_cleanup_and_retains_backup_on_failure() {
    use std::os::windows::fs::OpenOptionsExt;

    let dir = Dir::new();
    let path = dir.0.join("project.texelith.json");
    let paths = recovery_paths(&path).unwrap();
    let old = canonical_json(&fixture()).unwrap();
    fs::write(&path, &old).unwrap();
    let mut changed = fixture();
    changed.view.split_fraction = 0.25;
    let new = canonical_json(&changed).unwrap();
    let mut deny_write = None;
    let mut reached_cleanup = false;
    let result = save_project_with(&path, &changed, |point| {
        if point == FailurePoint::AfterPublishRename {
            assert_eq!(fs::read(&path).unwrap(), new);
            // Deny the write handle required by FlushFileBuffers, after the real
            // ReplaceFileW call. A pre-replacement temp flush cannot pass this.
            deny_write = Some(
                fs::OpenOptions::new()
                    .read(true)
                    .share_mode(1) // FILE_SHARE_READ
                    .open(&path)
                    .unwrap(),
            );
        } else if point == FailurePoint::BeforeBackupCleanup {
            reached_cleanup = true;
        }
        Ok(())
    });
    assert!(matches!(result, Err(ProjectIoError::Io(_))));
    assert!(!reached_cleanup);
    assert!(deny_write.is_some());
    assert_eq!(fs::read(&path).unwrap(), new);
    assert_eq!(fs::read(&paths.backup).unwrap(), old);
    drop(deny_write);
    assert_eq!(recover_project(&path).unwrap(), RecoveryOutcome::Unchanged);
    assert_eq!(fs::read(&path).unwrap(), new);
    assert!(!paths.backup.exists());
}

#[cfg(windows)]
#[test]
fn recovery_flushes_valid_main_before_removing_recovery_entries() {
    use std::os::windows::fs::OpenOptionsExt;

    let dir = Dir::new();
    let path = dir.0.join("project.texelith.json");
    let paths = recovery_paths(&path).unwrap();
    let expected = canonical_json(&fixture()).unwrap();
    fs::write(&path, &expected).unwrap();
    fs::write(&paths.backup, &expected).unwrap();
    fs::write(&paths.temporary, b"partial temp").unwrap();
    let deny_write = fs::OpenOptions::new()
        .read(true)
        .share_mode(1)
        .open(&path)
        .unwrap();
    assert!(matches!(recover_project(&path), Err(ProjectIoError::Io(_))));
    assert_eq!(fs::read(&path).unwrap(), expected);
    assert_eq!(fs::read(&paths.backup).unwrap(), expected);
    assert_eq!(fs::read(&paths.temporary).unwrap(), b"partial temp");
    drop(deny_write);
    assert_eq!(recover_project(&path).unwrap(), RecoveryOutcome::Unchanged);
    assert!(!paths.backup.exists());
    assert!(!paths.temporary.exists());
}

#[test]
fn recovery_does_not_discard_valid_main_for_invalid_siblings() {
    let dir = Dir::new();
    let path = dir.0.join("project.texelith.json");
    let paths = recovery_paths(&path).unwrap();
    let expected = canonical_json(&fixture()).unwrap();
    fs::write(&path, &expected).unwrap();
    fs::write(&paths.backup, b"invalid backup").unwrap();
    fs::write(&paths.temporary, b"invalid temp").unwrap();
    fs::write(&paths.backup_temporary, b"partial backup").unwrap();
    assert_eq!(recover_project(&path).unwrap(), RecoveryOutcome::Unchanged);
    assert_eq!(fs::read(path).unwrap(), expected);
    assert!(!paths.backup.exists());
    assert!(!paths.temporary.exists());
    assert!(!paths.backup_temporary.exists());
}

#[test]
fn invalid_recovery_preserves_candidates_for_diagnosis() {
    let dir = Dir::new();
    let path = dir.0.join("project.texelith.json");
    let paths = recovery_paths(&path).unwrap();
    for entry in [
        &path,
        &paths.backup,
        &paths.temporary,
        &paths.backup_temporary,
    ] {
        fs::write(entry, b"invalid JSON").unwrap();
    }
    assert!(recover_project(&path).is_err());
    for entry in [
        &path,
        &paths.backup,
        &paths.temporary,
        &paths.backup_temporary,
    ] {
        assert_eq!(fs::read(entry).unwrap(), b"invalid JSON");
    }
}

#[test]
fn generic_recovery_applies_consumer_validation_and_can_promote_a_temp() {
    let dir = Dir::new();
    let path = dir.0.join("preset.json");
    let paths = recovery_paths(&path).unwrap();
    save_json(&path, &json!({"schema": "unsupported"})).unwrap();
    let expected = canonical_json(&json!({"schema": 1, "name": "preset"})).unwrap();
    fs::write(&paths.backup, b"{broken").unwrap();
    fs::write(&paths.temporary, &expected).unwrap();
    let validate = |bytes: &[u8]| {
        let value = parse_json(bytes)?;
        if value["schema"] == json!(1) {
            Ok(())
        } else {
            Err(ProjectIoError::Invalid("unsupported preset schema"))
        }
    };
    assert_eq!(
        recover_with(&path, validate).unwrap(),
        RecoveryOutcome::PromotedTemp
    );
    assert_eq!(fs::read(&path).unwrap(), expected);
    assert_eq!(recover_json(&path).unwrap(), RecoveryOutcome::Unchanged);
}

#[test]
fn save_as_rebases_saved_mesh_environment_and_missing_asset_references() {
    let dir = Dir::new();
    let old = dir.0.join("old/project.texelith.json");
    let new = dir.0.join("new/deeper/project.texelith.json");
    fs::create_dir_all(old.parent().unwrap()).unwrap();
    fs::create_dir_all(new.parent().unwrap()).unwrap();
    let mut dto = fixture();
    dto.view.mesh = PreviewMesh::Custom(OpaqueLocator("../meshes/custom.obj".into()));
    dto.view.light.environment = Some(OpaqueLocator("../lights/studio.exr".into()));
    let before = dto.clone();
    let rebased = rebase_project(&dto, &old, &new).unwrap();
    assert_eq!(dto, before);
    assert_eq!(rebased.assets[0].relative_path, "../../textures/source.png");
    assert_eq!(
        rebased.view.mesh,
        PreviewMesh::Custom(OpaqueLocator("../../meshes/custom.obj".into()))
    );
    assert_eq!(
        rebased.view.light.environment,
        Some(OpaqueLocator("../../lights/studio.exr".into()))
    );
}

fn assert_future_schema_preserved(value: Value, version: u64, supported: u32) {
    let dir = Dir::new();
    let path = dir.0.join("project.texelith.json");
    let paths = recovery_paths(&path).unwrap();
    let main = serde_json::to_vec(&value).unwrap();
    let backup = canonical_json(&fixture()).unwrap();
    let entries = [
        (&path, main.as_slice()),
        (&paths.backup, backup.as_slice()),
        (&paths.temporary, backup.as_slice()),
        (&paths.backup_temporary, b"partial backup".as_slice()),
    ];
    for (entry, bytes) in entries {
        fs::write(entry, bytes).unwrap();
    }
    for result in [
        read_project(&main).map(|_| ()),
        recover_project(&path).map(|_| ()),
    ] {
        assert!(
            matches!(
                result,
                Err(ProjectIoError::UnsupportedSchema {
                    found,
                    supported_min,
                    supported_max,
                }) if found == version && supported_min == supported && supported_max == supported
            ),
            "expected unsupported schema {version}, got {result:?}"
        );
    }
    for (entry, bytes) in entries {
        assert_eq!(fs::read(entry).unwrap(), bytes);
    }
}

#[test]
fn future_project_schema_precedes_invalid_content() {
    for version in [2, u64::MAX] {
        assert_future_schema_preserved(
            json!({"schema_version": version, "format": false, "graph": null}),
            version,
            1,
        );
    }
}

#[test]
fn future_graph_schema_precedes_invalid_project_and_graph_content() {
    for version in [3, u64::MAX] {
        for changed_shape in [false, true] {
            let mut value = serde_json::to_value(fixture()).unwrap();
            if changed_shape {
                value["graph"] = json!({"future_graph": true});
            }
            value["graph"]["schema_version"] = json!(version);
            value["assets"] = json!(false);
            value["unknown_project_field"] = json!(true);
            assert_future_schema_preserved(value, version, 2);
        }
    }
}

#[test]
fn future_nested_section_schema_precedes_invalid_content() {
    for pointer in [
        "/view/camera",
        "/assets/0",
        "/graph/graph/0/params",
        "/export",
    ] {
        for version in [2, u64::MAX] {
            let mut value = serde_json::to_value(fixture()).unwrap();
            *value.pointer_mut(pointer).unwrap() =
                json!({"schema_version": version, "future_shape": [null]});
            value["format"] = json!(false);
            assert_future_schema_preserved(value, version, 1);
        }
    }
    // Future sections may also appear inside new fields and arrays.
    let mut value = serde_json::to_value(fixture()).unwrap();
    value["future_sections"] = json!([{"schema_version": 2}]);
    assert_future_schema_preserved(value, 2, 1);
}

#[test]
fn supported_versions_with_invalid_content_allow_backup_recovery() {
    for pointer in ["", "/graph", "/view/camera", "/assets/0", "/export"] {
        let dir = Dir::new();
        let path = dir.0.join("project.texelith.json");
        let paths = recovery_paths(&path).unwrap();
        let mut value = serde_json::to_value(fixture()).unwrap();
        let section = value.pointer_mut(pointer).unwrap();
        section["schema_version"] = json!(if pointer == "/graph" { 2 } else { 1 });
        section["unknown_field"] = json!(true);
        let main = serde_json::to_vec(&value).unwrap();
        assert!(read_project(&main).is_err());
        let backup = canonical_json(&fixture()).unwrap();
        fs::write(&path, main).unwrap();
        fs::write(&paths.backup, &backup).unwrap();
        assert_eq!(
            recover_project(&path).unwrap(),
            RecoveryOutcome::RestoredBackup
        );
        assert_eq!(fs::read(&path).unwrap(), backup);
    }
}
