//! The real selected shared mixer still owns recurrence with no history slot.
use super::super::qwen_hybrid_rows::{configuration, recurrent_parameter_pattern};
use super::*;

fn parameters(name: &str, shape: &[i32]) -> Option<NumericTensor> {
    let (base, step) = recurrent_parameter_pattern(name)?;
    let count: usize = shape.iter().map(|n| usize::try_from(*n).unwrap()).product();
    Some(NumericTensor::new(
        shape.to_vec(),
        (0..count).map(|i| base + step * (i % 7) as f32).collect(),
    ))
}

#[test]
fn selected_qwen_width_one_preserves_fixed_and_mixed_profiles_and_bound_rows() {
    for kind in ["qwen3_next", "qwen3_5_text"] {
        for (layers, access) in [
            (["linear_attention", "linear_attention"], Access::Fixed),
            (
                ["linear_attention", "full_attention"],
                Access::AttentionWithFixed,
            ),
            (
                ["full_attention", "linear_attention"],
                Access::AttentionWithFixed,
            ),
        ] {
            let mut config = configuration(kind, false, false);
            config["linear_conv_kernel_dim"] = 1.into();
            config["layer_types"] = serde_json::json!(layers);
            for residency in residencies() {
                execute_with_parameters(
                    &config,
                    access,
                    residency.clone(),
                    Trial::Binding,
                    parameters,
                );
                for trial in [
                    Trial::Split,
                    Trial::Body(OutputDemand::StateOnly),
                    Trial::Body(OutputDemand::LastPosition),
                ] {
                    compare_with_parameters(&config, access, residency.clone(), trial, parameters);
                }
            }
        }
    }
}

#[test]
fn selected_qwen_width_one_preserves_routed_and_mtp_cold_rejections() {
    use eredu_architectures::replicated_text::{
        safetensors_replicated_text_eligibility, ReplicatedTextIneligibility,
    };
    for kind in ["qwen3_next", "qwen3_5_text"] {
        let mut config = configuration(kind, true, false);
        config["linear_conv_kernel_dim"] = 1.into();
        let resolved = eredu_architectures::configuration::MODEL_CONFIGURATIONS
            .resolve_safetensors(&config)
            .unwrap();
        assert_eq!(
            safetensors_replicated_text_eligibility(
                resolved
                    .architecture_plan()
                    .safetensors_architecture()
                    .unwrap()
            ),
            Err(ReplicatedTextIneligibility::Routed)
        );
        let mut config = configuration(kind, false, false);
        config["linear_conv_kernel_dim"] = 1.into();
        config["mtp_num_hidden_layers"] = 1.into();
        let resolved = eredu_architectures::configuration::MODEL_CONFIGURATIONS
            .resolve_safetensors(&config)
            .unwrap();
        assert_eq!(
            safetensors_replicated_text_eligibility(
                resolved
                    .architecture_plan()
                    .safetensors_architecture()
                    .unwrap()
            ),
            Err(ReplicatedTextIneligibility::EmbeddedPrediction)
        );
    }
}
