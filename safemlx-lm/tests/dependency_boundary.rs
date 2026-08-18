use std::process::Command;

fn rust_sources(root: &std::path::Path, output: &mut Vec<std::path::PathBuf>) {
    for entry in std::fs::read_dir(root).expect("source directory must be readable") {
        let path = entry.expect("source entry must be readable").path();
        if path.is_dir() {
            rust_sources(&path, output);
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            output.push(path);
        }
    }
}

#[test]
fn portable_facade_graph_has_no_accelerator_runtime() {
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
    for forbidden in ["safemlx ", "safemlx-sys "] {
        assert!(
            !tree.lines().any(|line| line.starts_with(forbidden)),
            "forbidden dependency {forbidden:?} in:\n{tree}"
        );
    }
}

#[test]
fn backend_implementation_does_not_import_the_facade() {
    let backend = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/backend");
    let mut sources = Vec::new();
    rust_sources(&backend, &mut sources);
    let offenders = sources
        .into_iter()
        .filter(|path| {
            std::fs::read_to_string(path)
                .expect("backend source must be readable")
                .contains("crate::api")
        })
        .collect::<Vec<_>>();
    assert!(
        offenders.is_empty(),
        "backend implementations must not depend upward on the facade: {offenders:#?}"
    );
}
