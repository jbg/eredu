//! Actual selected NoState/KeyValue profiles must not need a fixed frontier.
use super::*;

#[test]
fn selected_lfm2_width_one_preserves_stateless_and_attention_only_profiles() {
    let base = heterogeneous_replicated_configs().remove(0);
    for (layers, access) in [
        (["conv", "conv"], Access::Stateless),
        (["conv", "full_attention"], Access::KeyValue),
        (["full_attention", "conv"], Access::KeyValue),
    ] {
        let mut config = base.clone();
        config["conv_L_cache"] = 1.into();
        config["layer_types"] = serde_json::json!(layers);
        for residency in residencies() {
            execute(&config, access, residency.clone(), Trial::Binding);
            for trial in [
                Trial::Split,
                Trial::Body(OutputDemand::StateOnly),
                Trial::Body(OutputDemand::LastPosition),
            ] {
                compare(&config, access, residency.clone(), trial);
            }
        }
    }
}
