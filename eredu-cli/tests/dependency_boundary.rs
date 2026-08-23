use std::{path::Path, process::Command};

fn assert_facade_only_execution_dependency(manifest: &Path) {
    let output = Command::new(env!("CARGO"))
        .args([
            "tree",
            "--manifest-path",
            manifest.to_str().unwrap(),
            "--edges",
            "normal",
            "--depth",
            "1",
            "--prefix",
            "none",
        ])
        .output()
        .expect("cargo tree must run");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let tree = String::from_utf8(output.stdout).unwrap();
    assert!(tree.lines().any(|line| line.starts_with("eredu v")));
    for forbidden in [
        "eredu-backend-mlx ",
        "eredu-checkpoint ",
        "safemlx ",
        "safemlx-sys ",
    ] {
        assert!(
            !tree.lines().any(|line| line.starts_with(forbidden)),
            "application has direct implementation-crate dependency {forbidden:?} in:\n{tree}"
        );
    }
}

#[test]
fn application_targets_reach_the_selected_backend_only_through_eredu() {
    let cli = Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
    let ios = Path::new(env!("CARGO_MANIFEST_DIR")).join("../examples/eredu-ios/native/Cargo.toml");
    assert_facade_only_execution_dependency(&cli);
    assert_facade_only_execution_dependency(&ios);
}
