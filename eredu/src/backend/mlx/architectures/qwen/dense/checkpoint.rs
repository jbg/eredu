//! Architecture-owned checkpoint contracts for dense Qwen2/Qwen3 decoders.

use eredu_checkpoint::{StoredDtype, WeightQuantization};

use std::collections::HashMap;

use safemlx::ops::{GgufCheckpoint, GgufMetadataValue};
use serde_json::Value;

use super::{config_from_gguf_catalog, config_from_hf_value, DecoderConfig};
use crate::backend::mlx::runtime::checkpoint::store::SafetensorsWeightStore;
use eredu_checkpoint::schema::{
    AlternativeLayoutGroup, CatalogPolicy, GgufCheckpointPlan, GgufTensorConstraint,
    GgufTypeConstraint, LayoutVariant, SafetensorsCheckpointPlan, SafetensorsTensorConstraint,
    StoredDtypeConstraint, TensorOperation,
};
use eredu_checkpoint::store::WeightStore;
use eredu_checkpoint::validation;
use eredu_checkpoint::validation::{CheckpointIssue, CheckpointIssueKind, CheckpointValidation};

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub(crate) enum GgufVariant {
    Qwen2,
    Qwen3,
    Qwen3Moe,
}

impl GgufVariant {
    const fn label(self) -> &'static str {
        match self {
            Self::Qwen2 => "Qwen2",
            Self::Qwen3 | Self::Qwen3Moe => "Qwen3",
        }
    }

    const fn metadata_name(self) -> &'static str {
        match self {
            Self::Qwen2 => "qwen2",
            Self::Qwen3 => "qwen3",
            Self::Qwen3Moe => "qwen3moe",
        }
    }

    const fn is_moe(self) -> bool {
        matches!(self, Self::Qwen3Moe)
    }
}

pub(crate) fn validate_safetensors(
    config: &Value,
    store: &SafetensorsWeightStore,
) -> CheckpointValidation {
    let args = match config_from_hf_value(config) {
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

pub(crate) fn safetensors_plan(args: &DecoderConfig) -> Result<SafetensorsCheckpointPlan, String> {
    safetensors_plan_with_root(args, "model", true)
}

pub(crate) fn safetensors_plan_with_root(
    args: &DecoderConfig,
    root: &str,
    allow_derived_expert_layouts: bool,
) -> Result<SafetensorsCheckpointPlan, String> {
    let hidden = dimension(args.hidden_size, "hidden_size")?;
    let vocab = dimension(args.vocab_size, "vocab_size")?;
    let query = checked_mul(
        dimension(args.num_attention_heads, "num_attention_heads")?,
        dimension(args.head_dim, "head_dim")?,
        "query width",
    )?;
    let key_value = checked_mul(
        dimension(args.num_key_value_heads, "num_key_value_heads")?,
        dimension(args.head_dim, "head_dim")?,
        "key/value width",
    )?;
    let head = dimension(args.head_dim, "head_dim")?;
    let mut common = Vec::new();
    let mut groups = Vec::new();
    add_tensor(
        args,
        &mut common,
        format!("{root}.embed_tokens.weight"),
        vec![vocab, hidden],
        TensorOperation::Matrix,
        None,
    )?;
    add_tensor(
        args,
        &mut common,
        format!("{root}.norm.weight"),
        vec![hidden],
        TensorOperation::Vector,
        None,
    )?;
    if args.tie_word_embeddings {
        let tensors = matrix_constraints(
            "lm_head.weight",
            vec![vocab, hidden],
            args.weight_quantization_for("lm_head.weight"),
            Vec::new(),
        )?;
        groups.push(AlternativeLayoutGroup {
            id: "redundant tied output head".into(),
            required: false,
            variants: vec![LayoutVariant {
                id: "present".into(),
                discriminator_keys: tensors.iter().map(|tensor| tensor.key.clone()).collect(),
                tensors,
            }],
        });
    } else {
        add_tensor(
            args,
            &mut common,
            "lm_head.weight",
            vec![vocab, hidden],
            TensorOperation::Matrix,
            None,
        )?;
    }
    for layer in 0..dimension(args.num_hidden_layers, "num_hidden_layers")? {
        let block = format!("{root}.layers.{layer}");
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
            (
                "self_attn.q_proj.weight",
                vec![query, hidden],
                TensorOperation::Matrix,
            ),
            (
                "self_attn.k_proj.weight",
                vec![key_value, hidden],
                TensorOperation::Matrix,
            ),
            (
                "self_attn.v_proj.weight",
                vec![key_value, hidden],
                TensorOperation::Matrix,
            ),
            (
                "self_attn.o_proj.weight",
                vec![hidden, query],
                TensorOperation::Matrix,
            ),
        ] {
            add_tensor(
                args,
                &mut common,
                format!("{block}.{name}"),
                shape,
                operation,
                None,
            )?;
        }
        if args.qk_norm() {
            for name in ["q_norm.weight", "k_norm.weight"] {
                add_tensor(
                    args,
                    &mut common,
                    format!("{block}.self_attn.{name}"),
                    vec![head],
                    TensorOperation::Vector,
                    None,
                )?;
            }
        }
        if args.qkv_bias() {
            for (name, size) in [
                ("q_proj.bias", query),
                ("k_proj.bias", key_value),
                ("v_proj.bias", key_value),
            ] {
                add_tensor(
                    args,
                    &mut common,
                    format!("{block}.self_attn.{name}"),
                    vec![size],
                    TensorOperation::Vector,
                    None,
                )?;
            }
        }
        if args.is_moe() {
            let experts = dimension(args.num_experts, "num_experts")?;
            let intermediate = dimension(args.moe_intermediate_size, "moe_intermediate_size")?;
            add_tensor(
                args,
                &mut common,
                format!("{block}.mlp.gate.weight"),
                vec![experts, hidden],
                TensorOperation::Matrix,
                Some(None),
            )?;
            groups.push(expert_layout_group(
                args,
                &format!("{block}.mlp.experts"),
                experts,
                hidden,
                intermediate,
                allow_derived_expert_layouts,
            )?);
        } else {
            let intermediate = dimension(args.intermediate_size, "intermediate_size")?;
            for (projection, shape) in [
                ("gate_proj", vec![intermediate, hidden]),
                ("up_proj", vec![intermediate, hidden]),
                ("down_proj", vec![hidden, intermediate]),
            ] {
                add_tensor(
                    args,
                    &mut common,
                    format!("{block}.mlp.{projection}.weight"),
                    shape,
                    TensorOperation::Matrix,
                    None,
                )?;
            }
        }
    }
    SafetensorsCheckpointPlan::new(
        format!("dense {} SafeTensors", args.model_type),
        common,
        groups,
        CatalogPolicy::strict(),
    )
    .map_err(|error| error.to_string())
}

pub(crate) fn is_redundant_tied_output_head_key(args: &DecoderConfig, key: &str) -> bool {
    args.tie_word_embeddings
        && matches!(
            key,
            "lm_head.weight" | "lm_head.inner.weight" | "lm_head.scales" | "lm_head.biases"
        )
}

fn expert_layout_group(
    args: &DecoderConfig,
    prefix: &str,
    experts: usize,
    hidden: usize,
    intermediate: usize,
    allow_derived_layouts: bool,
) -> Result<AlternativeLayoutGroup<SafetensorsTensorConstraint>, String> {
    let gate_up_quantization = args.weight_quantization_for(&format!("{prefix}.gate_up_proj"));
    let down_quantization = args.weight_quantization_for(&format!("{prefix}.down_proj"));
    let packed_gate_up = format!("{prefix}.gate_up_proj");
    let packed_down = format!("{prefix}.down_proj");
    let packed = expert_variant(
        "packed",
        vec![packed_gate_up.clone()],
        [
            (
                packed_gate_up,
                vec![
                    experts,
                    checked_mul(2, intermediate, "packed expert width")?,
                    hidden,
                ],
                gate_up_quantization,
                Vec::new(),
            ),
            (
                packed_down.clone(),
                vec![experts, hidden, intermediate],
                down_quantization,
                Vec::new(),
            ),
        ],
    )?;
    let gate = format!("{prefix}.gate_proj");
    let up = format!("{prefix}.up_proj");
    let separate = expert_variant(
        "separate-packed",
        vec![gate.clone(), up.clone()],
        [
            (
                gate.clone(),
                vec![experts, intermediate, hidden],
                gate_up_quantization,
                Vec::new(),
            ),
            (
                up.clone(),
                vec![experts, intermediate, hidden],
                gate_up_quantization,
                Vec::new(),
            ),
            (
                packed_down,
                vec![experts, hidden, intermediate],
                down_quantization,
                Vec::new(),
            ),
        ],
    )?;
    let mut variants = vec![packed];
    if allow_derived_layouts {
        variants.push(separate);
    }
    if allow_derived_layouts && gate_up_quantization.is_none() && down_quantization.is_none() {
        let mut tensors = Vec::with_capacity(checked_mul(experts, 3, "expert tensor count")?);
        let mut discriminators = Vec::with_capacity(tensors.capacity());
        for expert in 0..experts {
            for (canonical, aliases, shape) in [
                (
                    format!("{prefix}.{expert}.gate_proj.weight"),
                    vec![format!("{prefix}.{expert}.w1.weight")],
                    vec![intermediate, hidden],
                ),
                (
                    format!("{prefix}.{expert}.up_proj.weight"),
                    vec![format!("{prefix}.{expert}.w3.weight")],
                    vec![intermediate, hidden],
                ),
                (
                    format!("{prefix}.{expert}.down_proj.weight"),
                    vec![format!("{prefix}.{expert}.w2.weight")],
                    vec![hidden, intermediate],
                ),
            ] {
                discriminators.push(canonical.clone());
                tensors.push((canonical, shape, None, aliases));
            }
        }
        variants.push(expert_variant("split", discriminators, tensors)?);
    }
    Ok(AlternativeLayoutGroup {
        id: format!("{prefix} storage"),
        required: true,
        variants,
    })
}

fn expert_variant(
    id: &str,
    discriminator_keys: Vec<String>,
    tensors: impl IntoIterator<Item = (String, Vec<usize>, Option<WeightQuantization>, Vec<String>)>,
) -> Result<LayoutVariant<SafetensorsTensorConstraint>, String> {
    let mut constraints = Vec::new();
    for (name, shape, quantization, aliases) in tensors {
        constraints.extend(matrix_constraints(&name, shape, quantization, aliases)?);
    }
    Ok(LayoutVariant {
        id: id.into(),
        tensors: constraints,
        discriminator_keys,
    })
}

fn add_tensor(
    args: &DecoderConfig,
    output: &mut Vec<SafetensorsTensorConstraint>,
    name: impl AsRef<str>,
    shape: Vec<usize>,
    operation: TensorOperation,
    quantization_override: Option<Option<WeightQuantization>>,
) -> Result<(), String> {
    let name = name.as_ref();
    let quantization = if operation == TensorOperation::Matrix {
        quantization_override.unwrap_or_else(|| args.weight_quantization_for(name))
    } else {
        None
    };
    output.extend(matrix_constraints(name, shape, quantization, Vec::new())?);
    Ok(())
}

fn matrix_constraints(
    name: &str,
    shape: Vec<usize>,
    quantization: Option<WeightQuantization>,
    aliases: Vec<String>,
) -> Result<Vec<SafetensorsTensorConstraint>, String> {
    let mut names = vec![name.to_string()];
    names.extend(aliases);
    if let Some(quantization) = quantization {
        affine_constraints(&names, shape, quantization)
    } else {
        Ok(vec![physical_constraint(
            &names,
            shape,
            StoredDtypeConstraint::Floating,
        )])
    }
}

fn affine_constraints(
    aliases: &[String],
    shape: Vec<usize>,
    quantization: WeightQuantization,
) -> Result<Vec<SafetensorsTensorConstraint>, String> {
    let input = *shape
        .last()
        .ok_or_else(|| "quantized matrix has empty shape".to_string())?;
    let bits = quantization.bits() as usize;
    let group = quantization.group_size() as usize;
    let packed_bits = checked_mul(input, bits, "quantized packing")?;
    if !input.is_multiple_of(group) || !input.is_multiple_of(32) || !packed_bits.is_multiple_of(32)
    {
        return Err(format!(
            "quantized input dimension {input} is incompatible with group size {group} and {bits}-bit packing"
        ));
    }
    let mut packed = shape.clone();
    *packed.last_mut().expect("non-empty shape") = packed_bits / 32;
    let weights = aliases
        .iter()
        .flat_map(|name| std::iter::once(name.clone()).chain(quantized_weight_alias(name)))
        .collect::<Vec<_>>();
    let mut companion_shape = shape;
    *companion_shape.last_mut().expect("non-empty shape") = input / group;
    let dtype = || {
        StoredDtypeConstraint::OneOf(vec![
            StoredDtype::F16,
            StoredDtype::BF16,
            StoredDtype::F32,
            StoredDtype::U8,
        ])
    };
    let scales = aliases
        .iter()
        .map(|name| format!("{}.scales", outer_prefix(name)))
        .collect::<Vec<_>>();
    let mut result = vec![
        physical_constraint(
            &weights,
            packed,
            StoredDtypeConstraint::Exact(StoredDtype::U32),
        ),
        physical_constraint(&scales, companion_shape.clone(), dtype()).companion(),
    ];
    if quantization.has_biases() {
        let biases = aliases
            .iter()
            .map(|name| format!("{}.biases", outer_prefix(name)))
            .collect::<Vec<_>>();
        result.push(physical_constraint(&biases, companion_shape, dtype()).companion());
    }
    Ok(result)
}

fn physical_constraint(
    names: &[String],
    shape: Vec<usize>,
    dtype: StoredDtypeConstraint,
) -> SafetensorsTensorConstraint {
    SafetensorsTensorConstraint::required(names[0].clone(), shape, dtype)
        .with_aliases(names[1..].iter().cloned())
}

fn quantized_weight_alias(name: &str) -> Option<String> {
    name.strip_suffix(".weight")
        .map(|prefix| format!("{prefix}.inner.weight"))
}

fn outer_prefix(name: &str) -> &str {
    name.strip_suffix(".weight").unwrap_or(name)
}

pub(crate) fn validate_gguf(
    variant: GgufVariant,
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
) -> CheckpointValidation {
    let is_moe = variant.is_moe();
    if let Err(error) = checkpoint
        .catalog()
        .translated_outputs(|name| super::translate_gguf_weight_name(name, is_moe))
    {
        return CheckpointValidation::Invalid(vec![CheckpointIssue {
            kind: CheckpointIssueKind::ConflictingLayout,
            detail: error.to_string(),
            tensor_name: None,
            tensor_type_code: None,
            metadata_key: None,
        }]);
    }
    let args = match config_from_gguf_catalog(checkpoint, metadata, variant.metadata_name(), is_moe)
    {
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
    let plan = match gguf_plan(&args, variant) {
        Ok(plan) => plan,
        Err(error) => return invalid_geometry(error),
    };
    let mut issues = validation_issues(validation::validate_gguf_plan(checkpoint, &plan));
    if is_moe {
        issues.extend(validation::validate_matching_gguf_encodings(
            checkpoint,
            (0..args.num_hidden_layers as usize).map(|layer| {
                (
                    format!("blk.{layer}.ffn_gate_exps.weight"),
                    format!("blk.{layer}.ffn_up_exps.weight"),
                )
            }),
            "Qwen3 MoE",
        ));
    }
    CheckpointValidation::from_issues(issues)
}

pub(crate) fn gguf_plan(
    args: &DecoderConfig,
    variant: GgufVariant,
) -> Result<GgufCheckpointPlan, String> {
    let hidden = dimension(args.hidden_size, "hidden_size")?;
    let vocab = dimension(args.vocab_size, "vocab_size")?;
    let head = dimension(args.head_dim, "head_dim")?;
    let query = checked_mul(
        dimension(args.num_attention_heads, "num_attention_heads")?,
        head,
        "query width",
    )?;
    let key_value = checked_mul(
        dimension(args.num_key_value_heads, "num_key_value_heads")?,
        head,
        "key/value width",
    )?;
    let mut tensors = vec![
        gguf_tensor(
            "token_embd.weight",
            vec![vocab, hidden],
            TensorOperation::Matrix,
        ),
        gguf_tensor("output_norm.weight", vec![hidden], TensorOperation::Vector),
    ];
    if !args.tie_word_embeddings {
        tensors.push(gguf_tensor(
            "output.weight",
            vec![vocab, hidden],
            TensorOperation::Matrix,
        ));
    }
    for layer in 0..dimension(args.num_hidden_layers, "num_hidden_layers")? {
        let block = format!("blk.{layer}");
        for (name, shape, operation) in [
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
        ] {
            tensors.push(gguf_tensor(format!("{block}.{name}"), shape, operation));
        }
        if args.qk_norm() {
            for name in ["attn_q_norm.weight", "attn_k_norm.weight"] {
                tensors.push(gguf_tensor(
                    format!("{block}.{name}"),
                    vec![head],
                    TensorOperation::Vector,
                ));
            }
        }
        if args.qkv_bias() {
            for (name, size) in [
                ("attn_q.bias", query),
                ("attn_k.bias", key_value),
                ("attn_v.bias", key_value),
            ] {
                tensors.push(gguf_tensor(
                    format!("{block}.{name}"),
                    vec![size],
                    TensorOperation::Vector,
                ));
            }
        }
        if args.is_moe() {
            let experts = dimension(args.num_experts, "num_experts")?;
            let intermediate = dimension(args.moe_intermediate_size, "moe_intermediate_size")?;
            for (name, shape) in [
                ("ffn_gate_inp.weight", vec![experts, hidden]),
                ("ffn_gate_exps.weight", vec![experts, intermediate, hidden]),
                ("ffn_up_exps.weight", vec![experts, intermediate, hidden]),
                ("ffn_down_exps.weight", vec![experts, hidden, intermediate]),
            ] {
                tensors.push(gguf_tensor(
                    format!("{block}.{name}"),
                    shape,
                    TensorOperation::Matrix,
                ));
            }
        } else {
            let intermediate = dimension(args.intermediate_size, "intermediate_size")?;
            for (name, shape) in [
                ("ffn_gate.weight", vec![intermediate, hidden]),
                ("ffn_up.weight", vec![intermediate, hidden]),
                ("ffn_down.weight", vec![hidden, intermediate]),
            ] {
                tensors.push(gguf_tensor(
                    format!("{block}.{name}"),
                    shape,
                    TensorOperation::Matrix,
                ));
            }
        }
    }
    GgufCheckpointPlan::new(
        format!("{} GGUF", variant.label()),
        tensors,
        Vec::new(),
        CatalogPolicy::non_strict(),
    )
    .map_err(|error| error.to_string())
}

fn gguf_tensor(
    key: impl Into<String>,
    shape: Vec<usize>,
    operation: TensorOperation,
) -> GgufTensorConstraint {
    GgufTensorConstraint::required(key, shape, GgufTypeConstraint::OperationClass(operation))
}

fn validation_issues(validation: CheckpointValidation) -> Vec<CheckpointIssue> {
    match validation {
        CheckpointValidation::Exact => Vec::new(),
        CheckpointValidation::Invalid(issues) => issues,
        CheckpointValidation::Unverified(issue) => vec![issue],
    }
}

fn dimension(value: i32, name: &str) -> Result<usize, String> {
    usize::try_from(value)
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| format!("dense Qwen {name} must be positive, got {value}"))
}

fn checked_mul(left: usize, right: usize, name: &str) -> Result<usize, String> {
    left.checked_mul(right)
        .ok_or_else(|| format!("dense Qwen {name} geometry overflows"))
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
