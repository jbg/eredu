//! Writes the deterministic FP8 conformance input for independent reference generation.
#[allow(dead_code)]
#[path = "../src/tests/support/k2_horizon.rs"]
mod fixture;
fn main() -> anyhow::Result<()> {
    let destination = std::path::PathBuf::from(
        std::env::args_os()
            .nth(1)
            .ok_or_else(|| anyhow::anyhow!("usage: k2_fp8_fixture OUTPUT_DIRECTORY"))?,
    );
    let (_, encoded) = fixture::fp8();
    std::fs::create_dir_all(&destination)?;
    for name in ["config.json", "model.safetensors"] {
        std::fs::copy(encoded.path().join(name), destination.join(name))?;
    }
    Ok(())
}
