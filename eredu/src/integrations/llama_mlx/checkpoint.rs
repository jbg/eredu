//! Architecture-owned checkpoint contracts for Llama and Mistral decoders.
//!
//! This module owns physical tensor names, geometry, bias policy, affine
//! SafeTensors companions, GGUF encodings, and the small RoPE catalog
//! exclusions accepted by the Llama-compatible loaders. Generic checkpoint
//! code only evaluates the resulting declarative constraints.

use std::collections::{BTreeSet, HashMap};

use eredu_checkpoint::schema::SafetensorsTensorConstraint;
use safemlx::ops::{GgufCheckpoint, GgufMetadataValue};
use serde_json::Value;

use super::model;
use crate::backend::mlx::runtime::checkpoint::store::SafetensorsWeightStore;
use eredu_checkpoint::store::WeightStore;
use eredu_checkpoint::validation;
use eredu_checkpoint::validation::{CheckpointIssue, CheckpointIssueKind, CheckpointValidation};

pub(crate) fn validate_safetensors(
    config: &Value,
    store: &SafetensorsWeightStore,
) -> CheckpointValidation {
    let args = match model::model_args_from_config_value(config) {
        Ok(args) => args,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    if args.num_hidden_layers as usize > store.keys().len() {
        return invalid_geometry(format!(
            "configured layer count {} exceeds the entire {}-tensor checkpoint catalog",
            args.num_hidden_layers,
            store.keys().len()
        ));
    }
    let plan = match eredu_architectures::llama::safetensors_plan(&args) {
        Ok(plan) => plan,
        Err(eredu_architectures::llama::SafetensorsPlanError::Geometry(error)) => {
            return invalid_geometry(error)
        }
        Err(eredu_architectures::llama::SafetensorsPlanError::Companion { name, detail }) => {
            return CheckpointValidation::Invalid(vec![CheckpointIssue {
                kind: CheckpointIssueKind::CompanionMismatch,
                detail,
                tensor_name: Some(name),
                tensor_type_code: None,
                metadata_key: Some("quantization_config.quant_method".into()),
            }]);
        }
    };
    let mut issues = validation_issues(validation::validate_safetensors_plan(store, &plan));
    let allowed = physical_keys(&plan.common_tensors);
    for key in store.keys() {
        if !allowed.contains(&key)
            && !key.starts_with("rope_freqs.")
            && !key.ends_with(".rotary_emb.inv_freq")
        {
            issues.push(unexpected(&key, "Llama SafeTensors"));
        }
    }
    CheckpointValidation::from_issues(issues)
}

fn physical_keys(tensors: &[SafetensorsTensorConstraint]) -> BTreeSet<String> {
    tensors
        .iter()
        .flat_map(|tensor| std::iter::once(&tensor.key).chain(&tensor.aliases))
        .cloned()
        .collect()
}

pub(crate) fn validate_gguf(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
) -> CheckpointValidation {
    if let Err(error) = checkpoint
        .catalog()
        .translated_outputs(eredu_architectures::llama::translate_gguf_weight_name)
    {
        return CheckpointValidation::Invalid(vec![CheckpointIssue {
            kind: CheckpointIssueKind::ConflictingLayout,
            detail: error.to_string(),
            tensor_name: None,
            tensor_type_code: None,
            metadata_key: None,
        }]);
    }
    let args = match model::model_args_from_gguf_catalog(checkpoint, metadata) {
        Ok(args) => args,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    if args.num_hidden_layers as usize > checkpoint.catalog().physical_tensor_count() {
        return invalid_geometry(format!(
            "configured layer count {} exceeds the entire {}-tensor GGUF catalog",
            args.num_hidden_layers,
            checkpoint.catalog().physical_tensor_count()
        ));
    }
    let plan = match eredu_architectures::llama::gguf_plan(&args) {
        Ok(plan) => plan,
        Err(error) => return invalid_geometry(error),
    };
    let mut issues = validation_issues(validation::validate_gguf_plan(checkpoint, &plan));
    let allowed = plan
        .common_tensors
        .iter()
        .flat_map(|tensor| std::iter::once(&tensor.key).chain(&tensor.aliases))
        .collect::<BTreeSet<_>>();
    for tensor in checkpoint.catalog().tensors() {
        let name = &tensor.descriptor().name;
        if !allowed.contains(name) && !name.starts_with("rope_freqs.") {
            issues.push(unexpected(name, "Llama GGUF"));
        }
    }
    CheckpointValidation::from_issues(issues)
}

fn validation_issues(validation: CheckpointValidation) -> Vec<CheckpointIssue> {
    match validation {
        CheckpointValidation::Exact => Vec::new(),
        CheckpointValidation::Invalid(issues) => issues,
        CheckpointValidation::Unverified(issue) => vec![issue],
    }
}

fn unexpected(name: &str, loader: &str) -> CheckpointIssue {
    CheckpointIssue {
        kind: CheckpointIssueKind::UnexpectedTensor,
        detail: format!("{loader} catalog contains unexpected tensor {name:?}"),
        tensor_name: Some(name.into()),
        tensor_type_code: None,
        metadata_key: None,
    }
}

fn invalid_geometry(detail: String) -> CheckpointValidation {
    CheckpointValidation::Invalid(vec![CheckpointIssue {
        kind: CheckpointIssueKind::InvalidGeometry,
        detail,
        tensor_name: None,
        tensor_type_code: None,
        metadata_key: None,
    }])
}
