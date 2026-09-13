//! Cold physical-module ownership, independent of parallel storage placement.
use super::*;

pub(crate) fn parameter_residency(
    extension: &PredictionExtensionPlan,
    name: &str,
) -> Result<eredu_runtime::AuxiliaryModuleResidency, eredu_core::artifact::ArtifactError> {
    let indexed = |prefix: &str| {
        let rest = name.strip_prefix(prefix)?;
        let (index, field) = rest.split_once('.')?;
        Some((index.parse::<usize>().ok()?, field))
    };
    let unit = |index: usize, count: usize| {
        (index < count).then(|| (format!("prediction.unit.{index}"), false))
    };
    let shared = || ("prediction.shared".to_owned(), true);
    let owner = match extension.complete_architecture().model() {
        SafetensorsModelConfig::DeepSeekV3(args) => {
            let target = positive(args.num_hidden_layers, "target layer count")?;
            indexed("model.layers.")
                .and_then(|(index, _)| index.checked_sub(target))
                .and_then(|index| unit(index, extension.depth()))
        }
        SafetensorsModelConfig::DeepSeekV4(args) => indexed("mtp.").and_then(|(depth, field)| {
            let (stem, _) = field.split_once('.').unwrap_or((field, ""));
            let shared_input = depth == 0 && matches!(stem, "main_proj" | "main_norm");
            let shared_output = depth.checked_add(1) == Some(extension.depth())
                && matches!(
                    stem,
                    "norm"
                        | "hc_head_fn"
                        | "hc_head_base"
                        | "hc_head_scale"
                        | "markov_head"
                        | "confidence_head"
                );
            if args.dspark.is_some() && (shared_input || shared_output) {
                Some(shared())
            } else {
                unit(depth, extension.depth())
            }
        }),
        SafetensorsModelConfig::Inkling(_) => {
            if name.starts_with("model.mtp.chain_norm.") {
                Some(shared())
            } else {
                indexed("model.mtp.layers.").and_then(|(depth, _)| unit(depth, extension.depth()))
            }
        }
        SafetensorsModelConfig::QwenHybrid(_) => {
            if [
                "mtp.pre_fc_norm_hidden.",
                "mtp.pre_fc_norm_embedding.",
                "mtp.fc.",
                "mtp.norm.",
            ]
            .iter()
            .any(|prefix| name.starts_with(prefix))
            {
                Some(shared())
            } else {
                indexed("mtp.layers.").and_then(|(depth, _)| unit(depth, extension.depth()))
            }
        }
        SafetensorsModelConfig::NemotronH(args) => {
            let count = args
                .mtp_policies()
                .map_err(|error| invalid(error.to_string()))?
                .len();
            indexed("model.mtp.layers.").and_then(|(physical, _)| unit(physical, count))
        }
        _ => None,
    }
    .ok_or_else(|| {
        invalid(format!(
            "prediction parameter {name:?} has no physical module owner"
        ))
    })?;
    eredu_runtime::AuxiliaryModuleResidency::new(owner.0, owner.1)
        .map_err(|error| invalid(error.to_string()))
}
