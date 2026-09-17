//! Public short-Host-lookahead parity through the existing process/mode driver.
use super::*;

fn source() -> Fixture {
    let fixture = fixture(false);
    let path = fixture.0.join("config.json");
    let mut config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    // Four nonzero physical units exceed both Host lookahead one and the
    // existing two-unit Device demand. This reaches the exact direct remainder.
    config["num_hidden_layers"] = 4.into();
    std::fs::write(path, serde_json::to_vec(&config).unwrap()).unwrap();
    let resolved = eredu_architectures::configuration::resolve_model_config(&config).unwrap();
    write_tensor_plan(&fixture.0, resolved.architecture.checkpoint());
    fixture
}

#[test]
#[ignore = "requires an accessible Metal device"]
fn native_managed_background_short_host_window_matches_ordinary_and_controlled() {
    // The existing runner uses fresh processes for every mode, five prompt
    // tokens (managed/controlled chunks 2/2/1), and four generated tokens with
    // cached state. It checks the public one-byte refusal, cancellation, output
    // parity and escaped output lifetime after model/source destruction.
    verify_modes_with_residency(
        "managed_plain::background::native_managed_background_short_host_window_matches_ordinary_and_controlled",
        "EREDU_PUBLIC_MANAGED_BACKGROUND_MODE",
        0.0,
        source,
        eredu_core::TextSamplingStrategy::Standard,
        Some(eredu_core::ResidencyPlan::DenseDiskStream {
            device_budget_bytes: 8 << 20,
            host_budget_bytes: 1 << 20,
            host_lookahead: 1,
            background_queue: 1,
        }),
    );
}
