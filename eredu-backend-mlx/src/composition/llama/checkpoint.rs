//! MLX checkpoint adaptation for the backend-neutral Llama architecture.
//!
//! The architecture crate owns tensor geometry, canonical plans, and name
//! translation. This module applies those contracts to concrete MLX
//! SafeTensors and GGUF sources during cold-path validation and loading.

use std::collections::HashMap;

use eredu_architectures::llama::ModelArgs;
use eredu_checkpoint::WeightQuantization;
use safemlx::ops::{GgufCheckpoint, GgufMetadataValue};
use safemlx::Stream;
use serde_json::Value;

use crate::backend::error::Error;
use crate::backend::runtime::checkpoint::load::{gguf_quantization_configs, GgufTensorNames};
use eredu_checkpoint::store::SafetensorsWeightStore;
use eredu_checkpoint::store::WeightStore;
use eredu_checkpoint::validation;
use eredu_checkpoint::validation::{CheckpointIssue, CheckpointIssueKind, CheckpointValidation};

pub fn validate_safetensors(
    config: &Value,
    store: &SafetensorsWeightStore,
) -> CheckpointValidation {
    let args = match eredu_architectures::llama::model_args_from_config_value(config) {
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
    validation::validate_safetensors_plan(store, &plan)
}

pub fn validate_gguf(
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
    let args = match model_args_from_gguf_catalog(checkpoint, metadata) {
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
    validation::validate_gguf_plan(checkpoint, &plan)
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

pub(crate) struct PreparedLlamaGguf {
    pub args: ModelArgs,
    pub eos_token_ids: Vec<u32>,
}

pub(crate) fn prepare_llama_gguf_checkpoint(
    source: &crate::composition::mlx::structural::AdmittedGguf,
    quantization: Option<WeightQuantization>,
    _weights_stream: &Stream,
) -> Result<PreparedLlamaGguf, Error> {
    if !matches!(
        source.architecture(),
        eredu_architectures::GgufArchitecture::Llama
            | eredu_architectures::GgufArchitecture::Mistral
    ) {
        return Err(Error::UnsupportedArchitecture(format!(
            "Llama GGUF loader received architecture {:?}",
            source.architecture()
        )));
    }
    let checkpoint = source.checkpoint();
    let metadata = source.metadata();
    let mut args = model_args_from_gguf_catalog(checkpoint, metadata)?;
    checkpoint
        .catalog()
        .translated_outputs(eredu_architectures::llama::translate_gguf_weight_name)
        .map_err(safemlx::error::IoError::from)?;
    let quantized_weight_configs = gguf_quantization_configs(
        checkpoint,
        eredu_architectures::llama::translate_gguf_weight_name,
    )?;
    if let Some(quantization) = quantization {
        args.quantized_weights = None;
        args.quantization = Some(quantization);
        args.quantized_weight_configs = None;
    } else {
        args.quantized_weights = Some(quantized_weight_configs.keys().cloned().collect());
        args.quantization = None;
        args.quantized_weight_configs = Some(quantized_weight_configs);
    }

    let eos_token_ids = crate::composition::mlx::gguf_eos_token_ids(metadata)?;
    Ok(PreparedLlamaGguf {
        args,
        eos_token_ids,
    })
}

struct NeutralGgufCatalog<'a, T: ?Sized>(&'a T);

impl<T: GgufTensorNames + ?Sized> eredu_architectures::llama::GgufTensorCatalog
    for NeutralGgufCatalog<'_, T>
{
    fn contains(&self, name: &str) -> bool {
        self.0.contains_gguf_tensor(name)
    }

    fn any(&self, predicate: &mut dyn FnMut(&str) -> bool) -> bool {
        self.0.any_gguf_tensor(predicate)
    }
}

/// Parses the GGUF arguments shared by structural preflight and loading.
pub fn model_args_from_gguf_catalog(
    arrays: &(impl GgufTensorNames + ?Sized),
    metadata: &HashMap<String, GgufMetadataValue>,
) -> Result<ModelArgs, Error> {
    eredu_architectures::llama::model_args_from_gguf_catalog(&NeutralGgufCatalog(arrays), metadata)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
}
