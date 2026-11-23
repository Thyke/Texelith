use super::*;

#[test]
fn retry_and_resume_preserve_unrelated_manifest_and_recovery_sentinels() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../target-evidence/T-087")
        .join(format!(
            "manifest-regression-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
    std::fs::create_dir_all(&root).unwrap();
    let source = SourceDescriptor::from_rgba8(
        &[127; 4 * 32 * 32],
        Extent::new(32, 32).unwrap(),
        InputInterpretation::SrgbPhoto,
    )
    .unwrap();
    let graph = material_from_source(source).unwrap();
    let settings = crate::export_jobs::build_plan(
        &graph,
        &ExportOptions {
            preset: "blender-v1".into(),
            ..Default::default()
        },
    )
    .unwrap();
    let capture = ExportCapture::new(
        &EvaluationSnapshot {
            generation: 1,
            graph,
        },
        1,
        &settings,
    )
    .unwrap();
    let source = root.join("input.png");
    std::fs::write(&source, b"not decoded by manifest creation").unwrap();
    let source = std::fs::canonicalize(source).unwrap();
    let destination = root.join("output");
    let native = vec![NativeItem {
        index: 0,
        source: BatchSource {
            path: source.clone(),
            name: "input.png".into(),
        },
        destination: destination.clone(),
        material_name: "input".into(),
        decoded_hash: None,
    }];
    let input = BatchInput {
        source: OpaqueLocator(source.to_str().unwrap().replace('\\', "/")),
        source_digest: [0; 32],
        destinations: files::destinations(&settings, &destination, "input").unwrap(),
    };
    let mut job = BatchJob::new(1, capture, settings, vec![input], false).unwrap();
    let base = root.join("original.json");
    files::create_manifest(&base, &job, &native).unwrap();
    let original = std::fs::read(&base).unwrap();
    let work = job.work_id(0).unwrap();
    job.transition(work, BatchState::Loading).unwrap();
    let work = job.pending_work().unwrap();
    job.fail(work, "fixture decode failure".into()).unwrap();
    let retry = job.retry_failed(2).unwrap();
    let resume = job.resume_incomplete(3).unwrap();
    for candidate in [retry, resume] {
        let path = root.join(format!("original.job-{}.json", candidate.id()));
        let sentinel = b"unrelated durable job record";
        std::fs::write(&path, sentinel).unwrap();
        assert!(files::create_derived_manifest(&base, &candidate, &native).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), sentinel);
        assert_eq!(std::fs::read(&base).unwrap(), original);
        std::fs::remove_file(&path).unwrap();
        let recovery = texelith_io::project_io::recovery_paths(&path).unwrap();
        std::fs::write(&recovery.backup, sentinel).unwrap();
        assert!(files::create_derived_manifest(&base, &candidate, &native).is_err());
        assert_eq!(std::fs::read(&recovery.backup).unwrap(), sentinel);
        std::fs::remove_file(&recovery.backup).unwrap();
        let created = files::create_derived_manifest(&base, &candidate, &native).unwrap();
        assert_eq!(created, path);
        assert_eq!(
            texelith_io::project_io::read_json(&path).unwrap()["checkpoint"]["job"],
            candidate.id()
        );
        assert!(save_manifest(&path, &job, &native).is_err());
    }
}
