use super::*;
use texelith_io::export_transactions::{PublishOutcome, RecoveryRequired, RollbackFailure};

fn replacing() -> PublishOptions {
    PublishOptions {
        collision: CollisionPolicy::Replace,
        ..Default::default()
    }
}

#[test]
fn failed_rollback_renames_retain_journal_original_backup_and_both_errors() {
    for fail_retire in [false, true] {
        let root = Sandbox::new();
        let dest = original_set(root.path());
        let mut stage = None;
        let error = publish_set(&dest, &entries(), replacing(), encode, |point| {
            if point == FailurePoint::AfterFlush(0) {
                stage = Some(
                    fs::read_dir(root.path())?
                        .map(Result::unwrap)
                        .find(|e| {
                            e.file_name()
                                .to_string_lossy()
                                .starts_with(".texelith-stage-")
                        })
                        .unwrap()
                        .path(),
                );
            }
            if point
                == if fail_retire {
                    FailurePoint::AfterPublishRename
                } else {
                    FailurePoint::AfterBackupRename
                }
            {
                return Err(io::Error::other("original publication failure"));
            }
            if point
                == if fail_retire {
                    FailurePoint::BeforeRollbackRetire
                } else {
                    FailurePoint::BeforeRollbackRestore
                }
            {
                // Block rollback with a nonempty target directory. Retiring
                // fails in the OS; restoration rejects the intervening entry
                // before it can overwrite somebody else's directory.
                let target = if fail_retire {
                    stage.as_ref().unwrap()
                } else {
                    &dest
                };
                fs::create_dir(target)?;
                fs::write(target.join("obstruction"), b"keep")?;
            }
            Ok(())
        })
        .unwrap_err();
        let recovery = error
            .get_ref()
            .unwrap()
            .downcast_ref::<RollbackFailure>()
            .expect("rollback failure must report both failures and recovery paths");
        assert_eq!(
            recovery.publication_error.to_string(),
            "original publication failure"
        );
        assert!(!recovery.rollback_error.to_string().is_empty());
        assert!(error.to_string().contains("original publication failure"));
        assert!(error.to_string().contains("rollback failed"));
        assert_old(recovery.backup.as_ref().unwrap());
        let manifest = fs::read_to_string(&recovery.recovery_manifest).unwrap();
        assert!(manifest.contains("destination=Rock"));
        assert!(manifest.contains(&format!(
            "backup={}",
            recovery.backup.as_ref().unwrap().file_name().unwrap().to_string_lossy()
        )));
        assert!(!manifest.contains("state=committed"));
        if fail_retire {
            assert_eq!(
                recovery.recovery_manifest,
                dest.join(".texelith-transaction")
            );
        } else {
            assert_eq!(recovery.recovery_manifest.parent(), stage.as_deref());
        }
    }
}

#[test]
fn failed_publish_and_failed_restore_keep_recovery_metadata() {
    let root = Sandbox::new();
    let dest = original_set(root.path());
    let result = publish_set(&dest, &entries(), replacing(), encode, |point| {
        if point == FailurePoint::AfterBackupRename {
            fs::write(&dest, b"prevents both publication and restoration")?;
        }
        Ok(())
    });
    let error = result.unwrap_err();
    let recovery = error
        .get_ref()
        .unwrap()
        .downcast_ref::<RollbackFailure>()
        .unwrap();
    assert_old(recovery.backup.as_ref().unwrap());
    assert!(recovery.recovery_manifest.is_file());
    assert_eq!(
        fs::read(&dest).unwrap(),
        b"prevents both publication and restoration"
    );
}

#[test]
fn partial_backup_cleanup_is_committed_with_a_retained_cleanup_journal() {
    let root = Sandbox::new();
    let dest = original_set(root.path());
    let outcome = publish_set(&dest, &entries(), replacing(), encode, |point| {
        if point == FailurePoint::AfterBackupCleanup(0) {
            Err(io::Error::other(
                "cleanup interrupted after a real deletion",
            ))
        } else {
            Ok(())
        }
    })
    .expect("a cleanup failure must not report a committed export as failed");
    let PublishOutcome::CommittedWithCleanup {
        backup,
        recovery_manifest,
        error,
    } = outcome
    else {
        panic!("cleanup failure must be visible to the caller");
    };
    assert_eq!(
        error.to_string(),
        "cleanup interrupted after a real deletion"
    );
    assert_eq!(
        fs::read(dest.join("T_rock_Albedo.png")).unwrap(),
        b"new albedo"
    );
    assert_eq!(
        fs::read(dest.join("T_rock_Normal.png")).unwrap(),
        b"new normal"
    );
    assert_eq!(
        names(&backup).len(),
        1,
        "the old set was actually partially deleted"
    );
    assert_eq!(recovery_manifest, dest.join(".texelith-transaction"));
    let manifest = fs::read_to_string(recovery_manifest).unwrap();
    assert!(manifest.contains("state=committed\ncleanup=backup-pending"));
    assert!(manifest.contains(&format!(
        "backup={}",
        backup.file_name().unwrap().to_string_lossy()
    )));
    assert_eq!(names(root.path()).len(), 2);
}

#[test]
fn replacement_preflights_case_aliases_and_regular_files_before_encoding() {
    for regular_file in [false, true] {
        let root = Sandbox::new();
        let dest = root.path().join("Rock");
        if regular_file {
            fs::write(&dest, b"keep").unwrap();
        } else {
            fs::create_dir(root.path().join("ROCK")).unwrap();
        }
        let before = names(root.path());
        let mut encoded = 0;
        let result = publish_set(
            &dest,
            &entries(),
            replacing(),
            |_, _| {
                encoded += 1;
                Ok(())
            },
            |_| Ok(()),
        );
        assert!(result.is_err());
        assert_eq!(encoded, 0);
        assert_eq!(
            names(root.path()),
            before,
            "preflight must not create staging"
        );
        if regular_file {
            assert_eq!(fs::read(dest).unwrap(), b"keep");
        }
    }
}

#[test]
fn replacement_checks_all_siblings_even_when_exact_destination_exists() {
    let root = Sandbox::new();
    let dest = original_set(root.path());
    // NTFS can represent composed and decomposed Unicode siblings, including
    // on the Windows test host. These collide only after normalization.
    let alias = root.path().join("Ro\u{212a}");
    fs::create_dir(&alias).unwrap();
    let exact = root.path().join("RoK");
    if let Err(error) = fs::create_dir(&exact) {
        // Some filesystems cannot represent canonical aliases as siblings.
        assert_eq!(error.kind(), io::ErrorKind::AlreadyExists);
    }
    let before = names(root.path());
    assert!(preflight(root.path(), &["Ro\u{212a}", "RoK"]).is_err());
    let target = if before.iter().any(|n| n == "RoK") && before.iter().any(|n| n == "Ro\u{212a}") {
        exact
    } else {
        // Still exercise replacement with an alias differing from the actual
        // stored spelling when the filesystem itself already merges the pair.
        root.path().join(if before.iter().any(|n| n == "RoK") {
            "Ro\u{212a}"
        } else {
            "RoK"
        })
    };
    let mut encoded = false;
    let result = publish_set(
        &target,
        &entries(),
        replacing(),
        |_, _| {
            encoded = true;
            Ok(())
        },
        |_| Ok(()),
    );
    assert!(result.is_err());
    assert!(!encoded);
    assert_old(&dest);
    assert_eq!(names(root.path()), before);
}

#[test]
fn canonical_unicode_equivalents_collide_in_batch_disk_and_replacement_preflight() {
    for (a, b) in [
        ("caf\u{e9}", "cafe\u{301}"),
        ("\u{212b}", "A\u{30a}"),
        ("\u{ac01}", "\u{1100}\u{1161}\u{11a8}"),
        ("\u{1e0a}\u{323}", "D\u{323}\u{307}"),
    ] {
        let root = Sandbox::new();
        let (a, b) = (format!("{a}.png"), format!("{b}.png"));
        assert_eq!(
            preflight(root.path(), &[&a, &b]).unwrap_err().kind(),
            io::ErrorKind::AlreadyExists
        );
        let mut encoded = false;
        assert!(
            publish_set(
                &root.path().join("set"),
                &[ExportFile::new(&a), ExportFile::new(&b)],
                PublishOptions::default(),
                |_, _| {
                    encoded = true;
                    Ok(())
                },
                |_| Ok(())
            )
            .is_err()
        );
        assert!(!encoded);
        assert!(names(root.path()).is_empty());
        fs::create_dir(root.path().join(&a)).unwrap();
        let before = names(root.path());
        let alias = if before[0] == a { &b } else { &a };
        assert_eq!(
            preflight(root.path(), &[&b]).unwrap_err().kind(),
            io::ErrorKind::AlreadyExists
        );
        assert!(
            publish_set(
                &root.path().join(alias),
                &entries(),
                replacing(),
                |_, _| {
                    encoded = true;
                    Ok(())
                },
                |_| Ok(())
            )
            .is_err()
        );
        assert!(!encoded);
        assert_eq!(names(root.path()), before);
    }
}

#[test]
fn component_length_limits_are_checked_before_any_staging_or_encoder_write() {
    let root = Sandbox::new();
    for valid in [
        "a".repeat(255),
        format!("{}x", "é".repeat(127)),
        format!("{}abc", "😀".repeat(63)),
    ] {
        assert!(expand_name("{name}", &context(&valid, "", "", 0)).is_ok());
        assert!(preflight(root.path(), &[&valid]).is_ok());
    }
    for invalid in [
        "a".repeat(256),
        "é".repeat(128),
        "😀".repeat(64),
        format!("{}.png", "a".repeat(252)),
    ] {
        assert!(expand_name("{name}", &context(&invalid, "", "", 0)).is_err());
        assert!(preflight(root.path(), &[&invalid]).is_err());
        let mut encoded = false;
        for options in [PublishOptions::default(), replacing()] {
            assert!(
                publish_set(
                    &root.path().join("set"),
                    &[ExportFile::new(&invalid)],
                    options,
                    |_, _| {
                        encoded = true;
                        Ok(())
                    },
                    |_| Ok(())
                )
                .is_err()
            );
            assert!(
                publish_set(
                    &root.path().join(&invalid),
                    &entries(),
                    options,
                    |_, _| {
                        encoded = true;
                        Ok(())
                    },
                    |_| Ok(())
                )
                .is_err()
            );
        }
        assert!(!encoded);
        assert!(names(root.path()).is_empty());
    }
}

#[test]
fn windows_devices_include_superscripts_console_devices_and_space_dot_aliases() {
    let root = Sandbox::new();
    for invalid in [
        "COM¹",
        "com².png",
        "COM³ .png",
        "LPT¹",
        "lpt².png",
        "LPT³ .png",
        "CONIN$",
        "conin$.png",
        "CONOUT$",
        "conout$.png",
        "CLOCK$",
        "CON .png",
        "prn  .png",
        "NUL .png",
        "AUX .png",
        "COM1 .png",
        "LPT9 .png",
        "Rock.",
        "Rock ",
        "Rock. ",
    ] {
        assert!(
            expand_name("{name}", &context(invalid, "", "", 0)).is_err(),
            "accepted {invalid:?}"
        );
        assert!(
            preflight(root.path(), &[invalid]).is_err(),
            "accepted {invalid:?}"
        );
        let mut encoded = false;
        assert!(
            publish_set(
                &root.path().join("set"),
                &[ExportFile::new(invalid)],
                PublishOptions::default(),
                |_, _| {
                    encoded = true;
                    Ok(())
                },
                |_| Ok(())
            )
            .is_err()
        );
        assert!(!encoded);
        assert!(names(root.path()).is_empty());
    }
    for valid in [
        "COM0",
        "COM10",
        "LPT0",
        "CONSOLE.png",
        "T_CON.png",
        "COM⁴.png",
    ] {
        assert!(
            preflight(root.path(), &[valid]).is_ok(),
            "rejected {valid:?}"
        );
    }
}

#[test]
fn retry_after_failed_rollback_retirement_preserves_the_original_recovery_mapping() {
    let root = Sandbox::new();
    let dest = original_set(root.path());
    let error = publish_set(&dest, &entries(), replacing(), encode, |point| {
        if matches!(
            point,
            FailurePoint::AfterPublishRename | FailurePoint::BeforeRollbackRetire
        ) {
            Err(io::Error::other("injected publication/retirement failure"))
        } else {
            Ok(())
        }
    })
    .unwrap_err();
    let recovery = error
        .get_ref()
        .unwrap()
        .downcast_ref::<RollbackFailure>()
        .unwrap();
    let manifest = fs::read(&recovery.recovery_manifest).unwrap();
    let before = names(root.path());
    let mut encoded = false;
    let retry = publish_set(
        &dest,
        &entries(),
        replacing(),
        |_, _| {
            encoded = true;
            Ok(())
        },
        |_| Ok(()),
    );
    assert!(
        retry.is_err(),
        "retry must require recovery before replacement"
    );
    let error = retry.unwrap_err();
    let required = error
        .get_ref()
        .unwrap()
        .downcast_ref::<RecoveryRequired>()
        .unwrap();
    assert_eq!(required.recovery_manifest, recovery.recovery_manifest);
    assert!(
        error
            .to_string()
            .contains("resolve recovery or pending backup cleanup before replacing")
    );
    assert!(!encoded, "recovery preflight must precede encoding");
    assert_eq!(names(root.path()), before);
    assert_eq!(fs::read(&recovery.recovery_manifest).unwrap(), manifest);
    assert_old(recovery.backup.as_ref().unwrap());
}

#[test]
fn retry_after_partial_cleanup_preserves_the_pending_backup_and_journal() {
    let root = Sandbox::new();
    let dest = original_set(root.path());
    let outcome = publish_set(&dest, &entries(), replacing(), encode, |point| {
        if point == FailurePoint::AfterBackupCleanup(0) {
            Err(io::Error::other("injected cleanup failure"))
        } else {
            Ok(())
        }
    })
    .unwrap();
    let PublishOutcome::CommittedWithCleanup {
        backup,
        recovery_manifest,
        ..
    } = outcome
    else {
        panic!("expected pending cleanup");
    };
    let before = names(root.path());
    let backup_names = names(&backup);
    let backup_bytes: Vec<_> = backup_names
        .iter()
        .map(|name| fs::read(backup.join(name)).unwrap())
        .collect();
    let manifest = fs::read(&recovery_manifest).unwrap();
    let mut encoded = false;
    let retry = publish_set(
        &dest,
        &entries(),
        replacing(),
        |_, _| {
            encoded = true;
            Ok(())
        },
        |_| Ok(()),
    );
    assert!(
        retry.is_err(),
        "retry must require cleanup before replacement"
    );
    let error = retry.unwrap_err();
    let required = error
        .get_ref()
        .unwrap()
        .downcast_ref::<RecoveryRequired>()
        .unwrap();
    assert_eq!(required.recovery_manifest, recovery_manifest);
    assert!(!encoded);
    assert_eq!(names(root.path()), before);
    assert_eq!(fs::read(&recovery_manifest).unwrap(), manifest);
    assert_eq!(names(&backup), backup_names);
    for (name, bytes) in backup_names.iter().zip(backup_bytes) {
        assert_eq!(fs::read(backup.join(name)).unwrap(), bytes);
    }
    assert_eq!(
        fs::read(dest.join("T_rock_Albedo.png")).unwrap(),
        b"new albedo"
    );
    assert_eq!(
        fs::read(dest.join("T_rock_Normal.png")).unwrap(),
        b"new normal"
    );

    // Explicitly resolving this committed transaction's pending cleanup allows
    // a later replacement; the historical journal need not be discarded.
    fs::remove_dir_all(&backup).unwrap();
    assert!(matches!(
        publish_set(&dest, &entries(), replacing(), encode, |_| Ok(())).unwrap(),
        PublishOutcome::Committed
    ));
}

#[test]
fn successful_sets_can_be_replaced_repeatedly_after_cleanup() {
    let root = Sandbox::new();
    let dest = root.path().join("Rock");
    for _ in 0..3 {
        assert!(matches!(
            publish_set(&dest, &entries(), replacing(), encode, |_| Ok(())).unwrap(),
            PublishOutcome::Committed
        ));
        assert_eq!(names(root.path()), ["Rock"]);
    }
}

#[test]
fn replacement_rechecks_recovery_journal_before_moving_the_destination() {
    let root = Sandbox::new();
    let dest = original_set(root.path());
    let manifest_path = dest.join(".texelith-transaction");
    let journal = b"destination=Rock\nstate=rollback-failed\n";
    let error = publish_set(
        &dest,
        &entries(),
        replacing(),
        |file, out| {
            encode(file, out)?;
            fs::write(&manifest_path, journal)
        },
        |_| Ok(()),
    )
    .unwrap_err();
    let required = error
        .get_ref()
        .unwrap()
        .downcast_ref::<RecoveryRequired>()
        .unwrap();
    assert_eq!(required.recovery_manifest, manifest_path);
    assert_eq!(fs::read(manifest_path).unwrap(), journal);
    assert_old(&dest);
    assert_eq!(names(root.path()), ["Rock"]);
}
