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

#[test]
fn facade_runtime_contains_only_backend_independent_orchestration() {
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let runtime = manifest.join("src/runtime");
    for removed in [
        "attention",
        "cache",
        "checkpoint",
        "distributed",
        "execution",
        "media",
        "residency",
    ] {
        assert!(
            !runtime.join(removed).exists(),
            "backend implementation directory leaked into facade runtime: {removed}"
        );
    }
    assert!(
        !runtime.join("generation/sampler.rs").exists(),
        "MLX sampling must be owned by backend::mlx"
    );

    let mut sources = Vec::new();
    rust_sources(&runtime, &mut sources);
    let offenders = sources
        .into_iter()
        .filter(|path| {
            let source = std::fs::read_to_string(path).expect("runtime source must be readable");
            source.contains("safemlx::") || source.contains("safemlx_sys::")
        })
        .collect::<Vec<_>>();
    assert!(
        offenders.is_empty(),
        "facade runtime must not name accelerator runtime types: {offenders:#?}"
    );
}

#[test]
fn crate_root_does_not_reexport_mlx_implementation_types() {
    let library = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/lib.rs");
    let source = std::fs::read_to_string(library).expect("crate root must be readable");
    assert!(source.contains("pub mod backend;"));
    assert!(!source.contains("pub use backend::mlx"));
    assert!(!source.contains("pub mod error;"));
}

#[test]
fn generic_loaded_model_source_does_not_import_backend_implementations() {
    let loaded = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/api/loaded.rs");
    let source = std::fs::read_to_string(loaded).expect("loaded-model source must be readable");
    for forbidden in [
        "safemlx::",
        "safemlx_sys::",
        "crate::backend::mlx",
        "crate::architectures",
        "cfg(feature = \"mlx\")",
    ] {
        assert!(
            !source.contains(forbidden),
            "generic LoadedModel orchestration contains backend implementation reference {forbidden:?}"
        );
    }
}

#[test]
fn portable_api_tests_do_not_depend_on_backend_implementations() {
    let api = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/api");
    let portable =
        std::fs::read_to_string(api.join("tests.rs")).expect("portable API tests must be readable");
    for forbidden in [
        "safemlx::",
        "safemlx_sys::",
        "crate::backend::mlx",
        "crate::architectures",
        "crate::nn",
    ] {
        assert!(
            !portable.contains(forbidden),
            "portable API tests contain backend implementation reference {forbidden:?}"
        );
    }
    assert!(
        api.join("tests/mlx.rs").is_file(),
        "MLX integration tests must have a dedicated module"
    );
}
