use super::*;

#[test]
fn a_manifest_created_after_preflight_cannot_be_replaced() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../target-evidence/T-087")
        .join(format!("publication-race-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let destination = root.join("job.json");
    let staged = root.join("staged.json");
    // Different process IDs keep repeated test runs independent.
    tx::preflight(&root, &["job.json"]).unwrap();
    std::fs::write(&staged, b"new complete record").unwrap();
    std::fs::write(&destination, b"concurrently created unrelated record").unwrap();
    assert!(install_manifest(&staged, &destination).is_err());
    assert_eq!(
        std::fs::read(&destination).unwrap(),
        b"concurrently created unrelated record"
    );
    assert_eq!(std::fs::read(&staged).unwrap(), b"new complete record");
}
