//! Architecture-owned checkpoint contracts for GPT-OSS.

//!
//! GPT-OSS owns the tensor catalog and its native MXFP4 expert geometry here.
//! The generic checkpoint runtime only evaluates the resulting physical
//! constraints and never needs to know how GPT-OSS represents expert blocks.

use eredu_checkpoint::{StoredDtype, WeightQuantization};

use std::collections::HashMap;

use safemlx::ops::{GgufCheckpoint, GgufMetadataValue};
use serde_json::Value;

use super::model::{self, ModelArgs};
use crate::backend::mlx::runtime::checkpoint::store::{SafetensorsWeightStore, WeightStore};
use eredu_checkpoint::schema::{
    CatalogPolicy, GgufCheckpointPlan, GgufTensorConstraint, GgufTypeConstraint,
    SafetensorsCheckpointPlan, SafetensorsTensorConstraint, StoredDtypeConstraint, TensorOperation,
};
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
    let plan = match safetensors_plan(&args) {
        Ok(plan) => plan,
        Err(error) => return invalid_geometry(error),
    };
    validation::validate_safetensors_plan(store, &plan)
}

pub(crate) fn safetensors_plan(args: &ModelArgs) -> Result<SafetensorsCheckpointPlan, String> {
    let hidden = dimension(args.hidden_size, "hidden size")?;
    let vocab = dimension(args.vocab_size, "vocabulary size")?;
    let layers = dimension(args.num_hidden_layers, "layer count")?;
    let heads = dimension(args.num_attention_heads, "attention head count")?;
    let kv_heads = dimension(args.num_key_value_heads, "key/value head count")?;
    let head_dim = dimension(args.head_dim, "attention head width")?;
    let query = checked_mul(heads, head_dim, "query projection width")?;
    let key_value = checked_mul(kv_heads, head_dim, "key/value projection width")?;
    let experts = dimension(args.num_local_experts, "expert count")?;
    let intermediate = dimension(args.intermediate_size, "expert intermediate size")?;
    validate_mxfp4_geometry(hidden, intermediate)?;

    let mut tensors = Vec::new();
    add_safetensors_tensor(
        &mut tensors,
        "model.embed_tokens.weight",
        vec![vocab, hidden],
        TensorOperation::Matrix,
        args.quantization,
    )?;
    add_safetensors_tensor(
        &mut tensors,
        "model.norm.weight",
        vec![hidden],
        TensorOperation::Vector,
        args.quantization,
    )?;
    add_safetensors_tensor(
        &mut tensors,
        "lm_head.weight",
        vec![vocab, hidden],
        TensorOperation::Matrix,
        args.quantization,
    )?;

    for layer in 0..layers {
        let root = format!("model.layers.{layer}");
        for (name, shape, operation) in [
            (
                "input_layernorm.weight",
                vec![hidden],
                TensorOperation::Vector,
            ),
            (
                "post_attention_layernorm.weight",
                vec![hidden],
                TensorOperation::Vector,
            ),
            ("self_attn.sinks", vec![heads], TensorOperation::Vector),
            (
                "self_attn.q_proj.weight",
                vec![query, hidden],
                TensorOperation::Matrix,
            ),
            (
                "self_attn.q_proj.bias",
                vec![query],
                TensorOperation::Vector,
            ),
            (
                "self_attn.k_proj.weight",
                vec![key_value, hidden],
                TensorOperation::Matrix,
            ),
            (
                "self_attn.k_proj.bias",
                vec![key_value],
                TensorOperation::Vector,
            ),
            (
                "self_attn.v_proj.weight",
                vec![key_value, hidden],
                TensorOperation::Matrix,
            ),
            (
                "self_attn.v_proj.bias",
                vec![key_value],
                TensorOperation::Vector,
            ),
            (
                "self_attn.o_proj.weight",
                vec![hidden, query],
                TensorOperation::Matrix,
            ),
            (
                "self_attn.o_proj.bias",
                vec![hidden],
                TensorOperation::Vector,
            ),
            (
                "mlp.router.weight",
                vec![experts, hidden],
                TensorOperation::Dense,
            ),
            ("mlp.router.bias", vec![experts], TensorOperation::Vector),
        ] {
            add_safetensors_tensor(
                &mut tensors,
                &format!("{root}.{name}"),
                shape,
                operation,
                args.quantization,
            )?;
        }

        let expert_root = format!("{root}.mlp.experts");
        tensors.extend([
            SafetensorsTensorConstraint::required(
                format!("{expert_root}.gate_up_proj_blocks"),
                vec![
                    experts,
                    checked_mul(2, intermediate, "fused expert width")?,
                    hidden / 32,
                    16,
                ],
                StoredDtypeConstraint::Exact(StoredDtype::U8),
            ),
            SafetensorsTensorConstraint::required(
                format!("{expert_root}.gate_up_proj_scales"),
                vec![
                    experts,
                    checked_mul(2, intermediate, "fused expert width")?,
                    hidden / 32,
                ],
                StoredDtypeConstraint::Exact(StoredDtype::U8),
            )
            .companion(),
            SafetensorsTensorConstraint::required(
                format!("{expert_root}.gate_up_proj_bias"),
                vec![experts, checked_mul(2, intermediate, "fused expert width")?],
                StoredDtypeConstraint::Floating,
            )
            .companion(),
            SafetensorsTensorConstraint::required(
                format!("{expert_root}.down_proj_blocks"),
                vec![experts, hidden, intermediate / 32, 16],
                StoredDtypeConstraint::Exact(StoredDtype::U8),
            ),
            SafetensorsTensorConstraint::required(
                format!("{expert_root}.down_proj_scales"),
                vec![experts, hidden, intermediate / 32],
                StoredDtypeConstraint::Exact(StoredDtype::U8),
            )
            .companion(),
            SafetensorsTensorConstraint::required(
                format!("{expert_root}.down_proj_bias"),
                vec![experts, hidden],
                StoredDtypeConstraint::Floating,
            )
            .companion(),
        ]);
    }

    SafetensorsCheckpointPlan::new(
        "GPT-OSS SafeTensors",
        tensors,
        Vec::new(),
        CatalogPolicy::strict(),
    )
    .map_err(|error| error.to_string())
}

fn add_safetensors_tensor(
    output: &mut Vec<SafetensorsTensorConstraint>,
    name: &str,
    shape: Vec<usize>,
    operation: TensorOperation,
    quantization: Option<WeightQuantization>,
) -> Result<(), String> {
    if operation == TensorOperation::Matrix {
        if let Some(quantization) = quantization {
            output.extend(affine_constraints(name, shape, quantization)?);
            return Ok(());
        }
    }
    output.push(SafetensorsTensorConstraint::required(
        name,
        shape,
        StoredDtypeConstraint::Floating,
    ));
    Ok(())
}

fn affine_constraints(
    name: &str,
    shape: Vec<usize>,
    quantization: WeightQuantization,
) -> Result<Vec<SafetensorsTensorConstraint>, String> {
    let input = *shape
        .last()
        .ok_or_else(|| format!("GPT-OSS quantized tensor {name:?} has scalar shape"))?;
    let bits = quantization.bits() as usize;
    let group = quantization.group_size() as usize;
    let packed_bits = checked_mul(input, bits, "affine packing")?;
    if !input.is_multiple_of(group) || !input.is_multiple_of(32) || !packed_bits.is_multiple_of(32)
    {
        return Err(format!(
            "GPT-OSS quantized tensor {name:?} input dimension {input} is incompatible with group size {group} and {bits}-bit packing"
        ));
    }
    let mut packed = shape.clone();
    *packed.last_mut().expect("matrix shape") = packed_bits / 32;
    let mut companion_shape = shape;
    *companion_shape.last_mut().expect("matrix shape") = input / group;
    let prefix = name.strip_suffix(".weight").unwrap_or(name);
    let mut result = vec![SafetensorsTensorConstraint::required(
        name,
        packed,
        StoredDtypeConstraint::Exact(StoredDtype::U32),
    )
    .with_aliases([format!("{prefix}.inner.weight")])];
    let companion_dtype = || {
        StoredDtypeConstraint::OneOf(vec![
            StoredDtype::F16,
            StoredDtype::BF16,
            StoredDtype::F32,
            StoredDtype::U8,
        ])
    };
    result.push(
        SafetensorsTensorConstraint::required(
            format!("{prefix}.scales"),
            companion_shape.clone(),
            companion_dtype(),
        )
        .companion(),
    );
    if quantization.has_biases() {
        result.push(
            SafetensorsTensorConstraint::required(
                format!("{prefix}.biases"),
                companion_shape,
                companion_dtype(),
            )
            .companion(),
        );
    }
    Ok(result)
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
    let args = match model::args_from_gguf_catalog(checkpoint, metadata) {
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
    let mut issues = match validation::validate_gguf_plan(checkpoint, &plan) {
        CheckpointValidation::Exact => Vec::new(),
        CheckpointValidation::Invalid(issues) => issues,
        CheckpointValidation::Unverified(issue) => vec![issue],
    };
    issues.extend(validation::validate_matching_gguf_encodings(
        checkpoint,
        (0..args.num_hidden_layers as usize).map(|layer| {
            (
                format!("blk.{layer}.ffn_gate_exps.weight"),
                format!("blk.{layer}.ffn_up_exps.weight"),
            )
        }),
        "GPT-OSS",
    ));
    CheckpointValidation::from_issues(issues)
}

pub(crate) fn gguf_plan(args: &ModelArgs) -> Result<GgufCheckpointPlan, String> {
    let hidden = dimension(args.hidden_size, "hidden size")?;
    let vocab = dimension(args.vocab_size, "vocabulary size")?;
    let layers = dimension(args.num_hidden_layers, "layer count")?;
    let heads = dimension(args.num_attention_heads, "attention head count")?;
    let kv_heads = dimension(args.num_key_value_heads, "key/value head count")?;
    let head_dim = dimension(args.head_dim, "attention head width")?;
    let query = checked_mul(heads, head_dim, "query projection width")?;
    let key_value = checked_mul(kv_heads, head_dim, "key/value projection width")?;
    let experts = dimension(args.num_local_experts, "expert count")?;
    let intermediate = dimension(args.intermediate_size, "expert intermediate size")?;
    validate_mxfp4_geometry(hidden, intermediate)?;

    let mut tensors = vec![
        gguf(
            "token_embd.weight",
            vec![vocab, hidden],
            TensorOperation::Matrix,
        ),
        gguf("output_norm.weight", vec![hidden], TensorOperation::Vector),
        gguf(
            "output.weight",
            vec![vocab, hidden],
            TensorOperation::Matrix,
        ),
    ];
    for layer in 0..layers {
        let root = format!("blk.{layer}");
        tensors.extend([
            gguf(
                format!("{root}.attn_norm.weight"),
                vec![hidden],
                TensorOperation::Vector,
            ),
            gguf(
                format!("{root}.attn_post_norm.weight"),
                vec![hidden],
                TensorOperation::Vector,
            ),
            gguf(
                format!("{root}.attn_sinks.weight"),
                vec![heads],
                TensorOperation::Vector,
            )
            .with_aliases([format!("{root}.attn_sinks")]),
            gguf(
                format!("{root}.attn_q.weight"),
                vec![query, hidden],
                TensorOperation::Matrix,
            ),
            gguf(
                format!("{root}.attn_q.bias"),
                vec![query],
                TensorOperation::Vector,
            ),
            gguf(
                format!("{root}.attn_k.weight"),
                vec![key_value, hidden],
                TensorOperation::Matrix,
            ),
            gguf(
                format!("{root}.attn_k.bias"),
                vec![key_value],
                TensorOperation::Vector,
            ),
            gguf(
                format!("{root}.attn_v.weight"),
                vec![key_value, hidden],
                TensorOperation::Matrix,
            ),
            gguf(
                format!("{root}.attn_v.bias"),
                vec![key_value],
                TensorOperation::Vector,
            ),
            gguf(
                format!("{root}.attn_output.weight"),
                vec![hidden, query],
                TensorOperation::Matrix,
            ),
            gguf(
                format!("{root}.attn_output.bias"),
                vec![hidden],
                TensorOperation::Vector,
            ),
            gguf(
                format!("{root}.ffn_gate_inp.weight"),
                vec![experts, hidden],
                TensorOperation::Matrix,
            ),
            gguf(
                format!("{root}.ffn_gate_inp.bias"),
                vec![experts],
                TensorOperation::Vector,
            ),
            gguf(
                format!("{root}.ffn_gate_exps.weight"),
                vec![experts, intermediate, hidden],
                TensorOperation::MxFp4Matrix,
            ),
            gguf(
                format!("{root}.ffn_gate_exps.bias"),
                vec![experts, intermediate],
                TensorOperation::Vector,
            ),
            gguf(
                format!("{root}.ffn_up_exps.weight"),
                vec![experts, intermediate, hidden],
                TensorOperation::MxFp4Matrix,
            ),
            gguf(
                format!("{root}.ffn_up_exps.bias"),
                vec![experts, intermediate],
                TensorOperation::Vector,
            ),
            gguf(
                format!("{root}.ffn_down_exps.weight"),
                vec![experts, hidden, intermediate],
                TensorOperation::MxFp4Matrix,
            ),
            gguf(
                format!("{root}.ffn_down_exps.bias"),
                vec![experts, hidden],
                TensorOperation::Vector,
            ),
        ]);
    }
    GgufCheckpointPlan::new("GPT-OSS GGUF", tensors, Vec::new(), CatalogPolicy::strict())
        .map_err(|error| error.to_string())
}

fn gguf(
    key: impl Into<String>,
    shape: Vec<usize>,
    operation: TensorOperation,
) -> GgufTensorConstraint {
    GgufTensorConstraint::required(key, shape, GgufTypeConstraint::OperationClass(operation))
}

fn validate_mxfp4_geometry(hidden: usize, intermediate: usize) -> Result<(), String> {
    if !hidden.is_multiple_of(32) || !intermediate.is_multiple_of(32) {
        return Err(format!(
            "GPT-OSS MXFP4 dimensions must be divisible by 32, got hidden size {hidden} and intermediate size {intermediate}"
        ));
    }
    Ok(())
}

fn dimension(value: i32, name: &str) -> Result<usize, String> {
    usize::try_from(value)
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| format!("GPT-OSS {name} must be positive, got {value}"))
}

fn checked_mul(left: usize, right: usize, name: &str) -> Result<usize, String> {
    left.checked_mul(right)
        .ok_or_else(|| format!("GPT-OSS {name} geometry overflows"))
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
            "model_type": "gpt_oss",
            "hidden_size": 32,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": 32,
            "vocab_size": 32,
            "num_local_experts": 2,
            "num_experts_per_tok": 1,
            "rms_norm_eps": 1e-5,
            "sliding_window": 8,
            "max_position_embeddings": 128,
            "rope_theta": 150000,
            "rope_scaling": null,
            "layer_types": ["sliding_attention"],
            "quantization_config": {"quant_method": "mxfp4"}
        }))
        .unwrap()
    }

    #[test]
    fn plans_own_native_mxfp4_companions_and_sink_aliases() {
        let safe = safetensors_plan(&args()).unwrap();
        let scale = safe
            .common_tensors
            .iter()
            .find(|tensor| tensor.key.ends_with("gate_up_proj_scales"))
            .unwrap();
        assert_eq!(scale.shape, [2, 64, 1]);
        assert_eq!(scale.dtype, StoredDtypeConstraint::Exact(StoredDtype::U8));
        assert_eq!(scale.role, eredu_checkpoint::schema::TensorRole::Companion);

        let gguf = gguf_plan(&args()).unwrap();
        let sinks = gguf
            .common_tensors
            .iter()
            .find(|tensor| tensor.key == "blk.0.attn_sinks.weight")
            .unwrap();
        assert_eq!(sinks.aliases, ["blk.0.attn_sinks"]);
        assert!(gguf.common_tensors.iter().any(|tensor| {
            tensor.key == "blk.0.ffn_gate_exps.weight"
                && tensor.encoding
                    == GgufTypeConstraint::OperationClass(TensorOperation::MxFp4Matrix)
        }));
    }
}
