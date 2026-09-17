//! Actual selected KDA/MLA owners, with the direct fixture's parameter values.
use super::super::kimi_linear_rows::{configuration, recurrent_parameter_pattern, Shape};
use super::*;

fn parameters(name: &str, shape: &[i32]) -> Option<NumericTensor> {
    let (base, step) = recurrent_parameter_pattern(name)?;
    let count: usize = shape.iter().map(|n| usize::try_from(*n).unwrap()).product();
    Some(NumericTensor::new(
        shape.to_vec(),
        (0..count).map(|i| base + step * (i % 7) as f32).collect(),
    ))
}

fn cases() -> Vec<(serde_json::Value, Access)> {
    let mut cases = Vec::new();
    for kernel in [1, 3] {
        cases.push((
            configuration(false, Shape::Fixed, None, kernel),
            Access::Fixed,
        ));
        for rank in [None, Some(3)] {
            cases.push((
                configuration(false, Shape::Mixed, rank, kernel),
                Access::CompressedAttentionWithFixed,
            ));
        }
    }
    // A pure MLA schedule has no convolution; do not multiply it by a dormant
    // kernel setting. Both direct and low-rank query projections execute.
    for rank in [None, Some(3)] {
        cases.push((
            configuration(false, Shape::Compressed, rank, 3),
            Access::CompressedAttention,
        ));
    }
    cases
}

#[test]
fn selected_kimi_rows_preserve_all_eighteen_hooks_and_complete_continued_state() {
    for (config, access) in cases() {
        for residency in residencies() {
            compare_with_parameters(&config, access, residency, Trial::Split, parameters);
        }
    }
}

#[test]
fn selected_kimi_binds_actual_prepared_sources_and_physical_readout() {
    for (config, access) in cases() {
        for residency in residencies() {
            execute_with_parameters(&config, access, residency, Trial::Binding, parameters);
        }
    }
}

#[test]
fn selected_kimi_body_rows_preserve_state_only_and_last_position_execution() {
    for (config, access) in cases() {
        for residency in residencies() {
            for demand in [OutputDemand::StateOnly, OutputDemand::LastPosition] {
                compare_with_parameters(
                    &config,
                    access,
                    residency.clone(),
                    Trial::Body(demand),
                    parameters,
                );
            }
        }
    }
}

#[test]
fn selected_kimi_preserves_actual_routed_and_mtp_eligibility_rejections() {
    use eredu_architectures::replicated_text::{
        safetensors_replicated_text_eligibility, ReplicatedTextIneligibility,
    };
    for shape in [Shape::Fixed, Shape::Mixed, Shape::Compressed] {
        for kernel in [1, 3] {
            let routed = configuration(true, shape, Some(3), kernel);
            let resolved = eredu_architectures::configuration::MODEL_CONFIGURATIONS
                .resolve_safetensors(&routed)
                .unwrap();
            assert_eq!(
                safetensors_replicated_text_eligibility(
                    resolved
                        .architecture_plan()
                        .safetensors_architecture()
                        .unwrap(),
                ),
                Err(ReplicatedTextIneligibility::Routed),
            );
            let mut mtp = configuration(false, shape, Some(3), kernel);
            mtp["num_nextn_predict_layers"] = 1.into();
            assert!(matches!(
                eredu_architectures::kimi_linear::model_args_from_config_value(&mtp),
                Err(eredu_architectures::kimi_linear::ConfigError::Invalid(message))
                    if message == "Kimi Linear MTP layers are not implemented",
            ));
        }
    }
}
