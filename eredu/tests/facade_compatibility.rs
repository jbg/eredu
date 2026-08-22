use std::{path::Path, process::Command};

#[test]
fn retained_mlx_facade_paths_compile_for_a_downstream_crate() {
    let fixture =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/facade_compatibility/lib.rs");
    let project = tempfile::tempdir().expect("temporary downstream project");
    let source = project.path().join("src");
    std::fs::create_dir(&source).expect("downstream source directory");
    std::fs::copy(fixture, source.join("lib.rs")).expect("copy downstream fixture");

    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let workspace_root = Path::new(manifest_dir)
        .parent()
        .expect("eredu is a workspace member");
    std::fs::write(
        project.path().join("Cargo.toml"),
        format!(
            r#"[package]
name = "eredu-facade-compatibility"
version = "0.0.0"
edition = "2021"

[workspace]

[dependencies]
eredu = {{ path = {manifest_dir:?}, default-features = false, features = ["mlx"] }}
"#,
        ),
    )
    .expect("write downstream manifest");
    std::fs::copy(
        workspace_root.join("Cargo.lock"),
        project.path().join("Cargo.lock"),
    )
    .expect("seed downstream dependency resolution from the workspace lockfile");

    let output = Command::new(env!("CARGO"))
        .args(["check", "--manifest-path"])
        .arg(project.path().join("Cargo.toml"))
        .arg("--target-dir")
        .arg(workspace_root.join("target"))
        .arg("--offline")
        .output()
        .expect("cargo check must run");
    assert!(
        output.status.success(),
        "downstream compatibility fixture did not compile:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
}
