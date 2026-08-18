//! Architecture-owned checkpoint contracts for Llama and Mistral decoders.
//!
//! This module owns physical tensor names, geometry, bias policy, affine
//! SafeTensors companions, GGUF encodings, and the small RoPE catalog
//! exclusions accepted by the Llama-compatible loaders. Generic checkpoint
//! code only evaluates the resulting declarative constraints.

use std::collections::{BTreeSet, HashMap};
use std::fmt;

use safemlx::ops::{GgufCheckpoint, GgufMetadataValue};
use serde_json::Value;

use super::model::{self, ModelArgs};
use crate::backend::mlx::runtime::checkpoint::{
    contract::{CheckpointIssue, CheckpointIssueKind, CheckpointValidation},
    quantization::WeightQuantization,
    schema::{
        CatalogPolicy, GgufCheckpointPlan, GgufTensorConstraint, GgufTypeConstraint,
        SafetensorsCheckpointPlan, SafetensorsTensorConstraint, StoredDtypeConstraint,
        TensorOperation,
    },
    store::{SafetensorsWeightStore, StoredDtype, WeightStore},
    validation,
};

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
    let plan = match safetensors_plan(&args) {
        Ok(plan) => plan,
        Err(SafetensorsPlanError::Geometry(error)) => return invalid_geometry(error),
        Err(SafetensorsPlanError::Companion { name, detail }) => {
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

pub(crate) fn safetensors_plan(
    args: &ModelArgs,
) -> Result<SafetensorsCheckpointPlan, SafetensorsPlanError> {
    let hidden = dimension(args.hidden_size, "hidden size")?;
    let vocab = dimension(args.vocab_size, "vocabulary size")?;
    let intermediate = dimension(args.intermediate_size, "intermediate size")?;
    let query = checked_mul(
        dimension(args.num_attention_heads, "attention head count")?,
        dimension(args.head_dim, "attention head dimension")?,
        "query projection width",
    )?;
    let key_value = checked_mul(
        dimension(args.num_key_value_heads, "key/value head count")?,
        dimension(args.head_dim, "attention head dimension")?,
        "key/value projection width",
    )?;
    let mut tensors = Vec::new();
    add_safe(
        args,
        &mut tensors,
        "model.embed_tokens.weight",
        vec![vocab, hidden],
        TensorOperation::Matrix,
    )?;
    tensors.push(safe("model.norm.weight", vec![hidden]));
    if !args.tie_word_embeddings {
        add_safe(
            args,
            &mut tensors,
            "lm_head.weight",
            vec![vocab, hidden],
            TensorOperation::Matrix,
        )?;
    }
    for layer in 0..dimension(args.num_hidden_layers, "layer count")? {
        let block = format!("model.layers.{layer}");
        tensors.extend([
            safe(format!("{block}.input_layernorm.weight"), vec![hidden]),
            safe(
                format!("{block}.post_attention_layernorm.weight"),
                vec![hidden],
            ),
        ]);
        for (local, shape) in [
            ("self_attn.q_proj.weight", vec![query, hidden]),
            ("self_attn.k_proj.weight", vec![key_value, hidden]),
            ("self_attn.v_proj.weight", vec![key_value, hidden]),
            ("self_attn.o_proj.weight", vec![hidden, query]),
            ("mlp.gate_proj.weight", vec![intermediate, hidden]),
            ("mlp.up_proj.weight", vec![intermediate, hidden]),
            ("mlp.down_proj.weight", vec![hidden, intermediate]),
        ] {
            add_safe(
                args,
                &mut tensors,
                &format!("{block}.{local}"),
                shape,
                TensorOperation::Matrix,
            )?;
        }
        if args.attention_bias {
            for (local, size) in [
                ("q_proj.bias", query),
                ("k_proj.bias", key_value),
                ("v_proj.bias", key_value),
                ("o_proj.bias", hidden),
            ] {
                tensors.push(safe(format!("{block}.self_attn.{local}"), vec![size]));
            }
        }
        if args.mlp_bias {
            for (local, size) in [
                ("gate_proj.bias", intermediate),
                ("up_proj.bias", intermediate),
                ("down_proj.bias", hidden),
            ] {
                tensors.push(safe(format!("{block}.mlp.{local}"), vec![size]));
            }
        }
    }
    SafetensorsCheckpointPlan::new(
        "Llama SafeTensors",
        tensors,
        Vec::new(),
        CatalogPolicy::non_strict(),
    )
    .map_err(|error| SafetensorsPlanError::Geometry(error.to_string()))
}

fn add_safe(
    args: &ModelArgs,
    output: &mut Vec<SafetensorsTensorConstraint>,
    name: &str,
    shape: Vec<usize>,
    operation: TensorOperation,
) -> Result<(), SafetensorsPlanError> {
    let quantization = (operation == TensorOperation::Matrix)
        .then(|| args.affine_quantization_for(name))
        .flatten();
    output.extend(safe_matrix_constraints(name, shape, quantization)?);
    Ok(())
}

fn safe_matrix_constraints(
    name: &str,
    shape: Vec<usize>,
    quantization: Option<WeightQuantization>,
) -> Result<Vec<SafetensorsTensorConstraint>, SafetensorsPlanError> {
    let Some(quantization) = quantization else {
        return Ok(vec![safe(name, shape)]);
    };
    let input = *shape
        .last()
        .ok_or_else(|| format!("quantized Llama matrix {name:?} has scalar shape"))?;
    let bits = dimension(quantization.bits(), "quantization bit width")?;
    let group = dimension(quantization.group_size(), "quantization group size")?;
    let packed_bits = input
        .checked_mul(bits)
        .ok_or_else(|| SafetensorsPlanError::Companion {
            name: name.into(),
            detail: format!("quantized tensor {name:?} packing geometry overflows"),
        })?;
    if !input.is_multiple_of(group) || !input.is_multiple_of(32) || !packed_bits.is_multiple_of(32)
    {
        return Err(SafetensorsPlanError::Companion {
            name: name.into(),
            detail: format!(
                "quantized tensor {name:?} input dimension {input} is incompatible with group size {group} and {bits}-bit packing"
            ),
        });
    }
    let mut packed = shape.clone();
    *packed.last_mut().expect("matrix shape") = packed_bits / 32;
    let mut companion = shape;
    *companion.last_mut().expect("matrix shape") = input / group;
    let prefix = name.strip_suffix(".weight").unwrap_or(name);
    let dtype = || {
        StoredDtypeConstraint::OneOf(vec![
            StoredDtype::F16,
            StoredDtype::BF16,
            StoredDtype::F32,
            StoredDtype::U8,
        ])
    };
    let mut constraints = vec![SafetensorsTensorConstraint::required(
        name,
        packed,
        StoredDtypeConstraint::Exact(StoredDtype::U32),
    )
    .with_aliases([format!("{prefix}.inner.weight")])];
    constraints.push(
        SafetensorsTensorConstraint::required(
            format!("{prefix}.scales"),
            companion.clone(),
            dtype(),
        )
        .companion(),
    );
    if quantization.has_biases() {
        constraints.push(
            SafetensorsTensorConstraint::required(format!("{prefix}.biases"), companion, dtype())
                .companion(),
        );
    }
    Ok(constraints)
}

fn safe(key: impl Into<String>, shape: Vec<usize>) -> SafetensorsTensorConstraint {
    SafetensorsTensorConstraint::required(key, shape, StoredDtypeConstraint::Floating)
}

#[derive(Debug)]
pub(crate) enum SafetensorsPlanError {
    Geometry(String),
    Companion { name: String, detail: String },
}

impl fmt::Display for SafetensorsPlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Geometry(detail) | Self::Companion { detail, .. } => formatter.write_str(detail),
        }
    }
}

impl From<String> for SafetensorsPlanError {
    fn from(detail: String) -> Self {
        Self::Geometry(detail)
    }
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
        .translated_outputs(model::translate_gguf_weight_name)
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
    let plan = match gguf_plan(&args) {
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

pub(crate) fn gguf_plan(args: &ModelArgs) -> Result<GgufCheckpointPlan, String> {
    let hidden = dimension(args.hidden_size, "hidden size")?;
    let vocab = dimension(args.vocab_size, "vocabulary size")?;
    let intermediate = dimension(args.intermediate_size, "intermediate size")?;
    let query = checked_mul(
        dimension(args.num_attention_heads, "attention head count")?,
        dimension(args.head_dim, "attention head dimension")?,
        "query projection width",
    )?;
    let key_value = checked_mul(
        dimension(args.num_key_value_heads, "key/value head count")?,
        dimension(args.head_dim, "attention head dimension")?,
        "key/value projection width",
    )?;
    let mut tensors = vec![
        gguf(
            "token_embd.weight",
            vec![vocab, hidden],
            TensorOperation::Matrix,
        ),
        gguf("output_norm.weight", vec![hidden], TensorOperation::Vector),
    ];
    if !args.tie_word_embeddings {
        tensors.push(gguf(
            "output.weight",
            vec![vocab, hidden],
            TensorOperation::Matrix,
        ));
    }
    for layer in 0..dimension(args.num_hidden_layers, "layer count")? {
        let block = format!("blk.{layer}");
        for (local, shape, operation) in [
            ("attn_norm.weight", vec![hidden], TensorOperation::Vector),
            ("ffn_norm.weight", vec![hidden], TensorOperation::Vector),
            (
                "attn_q.weight",
                vec![query, hidden],
                TensorOperation::Matrix,
            ),
            (
                "attn_k.weight",
                vec![key_value, hidden],
                TensorOperation::Matrix,
            ),
            (
                "attn_v.weight",
                vec![key_value, hidden],
                TensorOperation::Matrix,
            ),
            (
                "attn_output.weight",
                vec![hidden, query],
                TensorOperation::Matrix,
            ),
            (
                "ffn_gate.weight",
                vec![intermediate, hidden],
                TensorOperation::Matrix,
            ),
            (
                "ffn_up.weight",
                vec![intermediate, hidden],
                TensorOperation::Matrix,
            ),
            (
                "ffn_down.weight",
                vec![hidden, intermediate],
                TensorOperation::Matrix,
            ),
        ] {
            tensors.push(gguf(format!("{block}.{local}"), shape, operation));
        }
        if args.attention_bias {
            for (local, size) in [
                ("attn_q.bias", query),
                ("attn_k.bias", key_value),
                ("attn_v.bias", key_value),
                ("attn_output.bias", hidden),
            ] {
                tensors.push(gguf(
                    format!("{block}.{local}"),
                    vec![size],
                    TensorOperation::Vector,
                ));
            }
        }
        if args.mlp_bias {
            for (local, size) in [
                ("ffn_gate.bias", intermediate),
                ("ffn_up.bias", intermediate),
                ("ffn_down.bias", hidden),
            ] {
                tensors.push(gguf(
                    format!("{block}.{local}"),
                    vec![size],
                    TensorOperation::Vector,
                ));
            }
        }
    }
    GgufCheckpointPlan::new(
        "Llama GGUF",
        tensors,
        Vec::new(),
        CatalogPolicy::non_strict(),
    )
    .map_err(|error| error.to_string())
}

fn gguf(
    key: impl Into<String>,
    shape: Vec<usize>,
    operation: TensorOperation,
) -> GgufTensorConstraint {
    GgufTensorConstraint::required(key, shape, GgufTypeConstraint::OperationClass(operation))
}

fn dimension(value: i32, name: &str) -> Result<usize, String> {
    usize::try_from(value)
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| format!("Llama {name} must be positive, got {value}"))
}

fn checked_mul(left: usize, right: usize, name: &str) -> Result<usize, String> {
    left.checked_mul(right)
        .ok_or_else(|| format!("Llama {name} geometry overflows"))
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

#[cfg(test)]
mod tests {
    use super::*;

    fn args() -> ModelArgs {
        model::model_args_from_config_value(&serde_json::json!({
            "model_type": "llama", "hidden_size": 32, "num_hidden_layers": 1,
            "intermediate_size": 64, "num_attention_heads": 4,
            "num_key_value_heads": 2, "rms_norm_eps": 1e-5, "vocab_size": 32,
            "max_position_embeddings": 128, "rope_theta": 10000.0,
            "attention_bias": true, "mlp_bias": true, "tie_word_embeddings": false
        }))
        .unwrap()
    }

    #[test]
    fn plans_own_tied_output_and_bias_geometry() {
        let mut args = args();
        let plan = safetensors_plan(&args).unwrap();
        let names = plan
            .common_tensors
            .iter()
            .map(|tensor| tensor.key.as_str())
            .collect::<BTreeSet<_>>();
        assert!(names.contains("lm_head.weight"));
        assert!(names.contains("model.layers.0.self_attn.o_proj.bias"));
        assert!(names.contains("model.layers.0.mlp.down_proj.bias"));

        args.tie_word_embeddings = true;
        let tied = safetensors_plan(&args).unwrap();
        assert!(!tied
            .common_tensors
            .iter()
            .any(|tensor| tensor.key == "lm_head.weight"));
    }
}
