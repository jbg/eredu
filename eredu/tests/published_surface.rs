use std::process::Command;

#[test]
fn published_crates_do_not_expose_test_support_features() {
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
    let output = Command::new(env!("CARGO"))
        .args([
            "metadata",
            "--manifest-path",
            manifest.to_str().unwrap(),
            "--format-version",
            "1",
            "--no-deps",
        ])
        .output()
        .expect("cargo metadata must run");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let metadata: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    for name in ["eredu", "eredu-backend-mlx"] {
        let package = metadata["packages"]
            .as_array()
            .unwrap()
            .iter()
            .find(|package| package["name"] == name)
            .unwrap_or_else(|| panic!("missing {name} package metadata"));
        let features = package["features"].as_object().unwrap();
        for forbidden in ["test-support", "mlx-test-support"] {
            assert!(
                !features.contains_key(forbidden),
                "published package {name} exposes support-only feature {forbidden}"
            );
        }
    }
}
