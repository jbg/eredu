use std::process::Command;

#[test]
fn no_default_features_dependency_graph_has_no_concrete_backend() {
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
    let output = Command::new(env!("CARGO"))
        .args([
            "tree",
            "--manifest-path",
            manifest.to_str().unwrap(),
            "--no-default-features",
            "--edges",
            "normal",
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
    for forbidden in ["eredu-backend-mlx ", "safemlx ", "safemlx-sys "] {
        assert!(
            !tree.lines().any(|line| line.starts_with(forbidden)),
            "forbidden dependency {forbidden:?} in:\n{tree}"
        );
    }
}
