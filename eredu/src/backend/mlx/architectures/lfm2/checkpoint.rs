//! Architecture-owned checkpoint contracts for LFM2 and LFM2-MoE.

//!
//! LFM2 owns its hybrid operator schedule, short-convolution layouts, expert
//! storage alternatives, aliases, and quantization exclusions here. The
//! generic checkpoint runtime evaluates the resulting physical constraints.

use eredu_checkpoint::{StoredDtype, WeightQuantization};

use std::collections::{BTreeSet, HashMap};

use safemlx::ops::{GgufCheckpoint, GgufMetadataValue};
use serde_json::Value;

use super::model::{self, FeedForwardPolicy, ModelArgs, OperatorPolicy};
use crate::{
    backend::mlx::runtime::checkpoint::store::{SafetensorsWeightStore, WeightStore},
    AttentionPolicy,
};
use eredu_checkpoint::schema::{
    AlternativeLayoutGroup, CatalogPolicy, GgufCheckpointPlan, GgufTensorConstraint,
    GgufTypeConstraint, LayoutVariant, SafetensorsCheckpointPlan, SafetensorsTensorConstraint,
    StoredDtypeConstraint, TensorOperation,
};
use eredu_checkpoint::validation;
use eredu_checkpoint::validation::{CheckpointIssue, CheckpointIssueKind, CheckpointValidation};

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub(crate) enum GgufVariant {
    Dense,
    Moe,
}

impl GgufVariant {
    const fn is_moe(self) -> bool {
        matches!(self, Self::Moe)
    }

    const fn metadata_name(self) -> &'static str {
        match self {
            Self::Dense => "lfm2",
            Self::Moe => "lfm2moe",
        }
    }
}

pub(crate) fn validate_safetensors(
    config: &Value,
    store: &SafetensorsWeightStore,
    allow_derived_packed: bool,
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
    let plan = match safetensors_plan(&args, allow_derived_packed) {
        Ok(plan) => plan,
        Err(error) => return invalid_geometry(error),
    };
    let mut issues = validation_issues(validation::validate_safetensors_plan(store, &plan));
    validate_expert_prefix_catalogs(store, &args, &plan, &mut issues);
    finish(issues)
}

pub(crate) fn safetensors_plan(
    args: &ModelArgs,
    allow_derived_packed: bool,
) -> Result<SafetensorsCheckpointPlan, String> {
    let hidden = dimension(args.hidden_size, "hidden size")?;
    let vocab = dimension(args.vocab_size, "vocabulary size")?;
    let layers = dimension(args.num_hidden_layers, "layer count")?;
    let heads = dimension(args.num_attention_heads, "attention head count")?;
    let kv_heads = dimension(args.num_key_value_heads, "key/value head count")?;
    if !hidden.is_multiple_of(heads) {
        return Err(format!(
            "LFM2 hidden size {hidden} is not divisible by attention head count {heads}"
        ));
    }
    let head = hidden / heads;
    let key_value = checked_mul(kv_heads, head, "key/value projection width")?;
    let mut common = Vec::new();
    let mut groups = Vec::new();
    add_safe_matrix(
        args,
        &mut common,
        "model.embed_tokens.weight",
        vec![vocab, hidden],
        true,
    )?;
    common.push(safe("model.embedding_norm.weight", vec![hidden]));
    if !args.tie_word_embeddings {
        add_safe_matrix(
            args,
            &mut common,
            "lm_head.weight",
            vec![vocab, hidden],
            true,
        )?;
    }

    for layer in 0..layers {
        let root = format!("model.layers.{layer}");
        common.extend([
            safe(format!("{root}.operator_norm.weight"), vec![hidden]),
            safe(format!("{root}.ffn_norm.weight"), vec![hidden]),
        ]);
        let policy = args
            .layer_policy(layer)
            .ok_or_else(|| format!("LFM2 layer schedule has no policy for layer {layer}"))?;
        match policy.operator {
            OperatorPolicy::CausalConvolution => {
                let kernel = dimension(args.conv_l_cache, "short-convolution width")?;
                common.push(
                    safe(format!("{root}.conv.conv.weight"), vec![hidden, 1, kernel])
                        .with_alternate_shapes([vec![hidden, kernel, 1]]),
                );
                add_safe_matrix(
                    args,
                    &mut common,
                    &format!("{root}.conv.in_proj.weight"),
                    vec![
                        checked_mul(3, hidden, "short-convolution input width")?,
                        hidden,
                    ],
                    true,
                )?;
                add_safe_matrix(
                    args,
                    &mut common,
                    &format!("{root}.conv.out_proj.weight"),
                    vec![hidden, hidden],
                    true,
                )?;
                if args.conv_bias {
                    common.extend([
                        safe(format!("{root}.conv.conv.bias"), vec![hidden]),
                        safe(
                            format!("{root}.conv.in_proj.bias"),
                            vec![checked_mul(3, hidden, "short-convolution bias width")?],
                        ),
                        safe(format!("{root}.conv.out_proj.bias"), vec![hidden]),
                    ]);
                }
            }
            OperatorPolicy::SelfAttention(AttentionPolicy::Full) => {
                for (local, shape, matrix) in [
                    ("q_proj.weight", vec![hidden, hidden], true),
                    ("k_proj.weight", vec![key_value, hidden], true),
                    ("v_proj.weight", vec![key_value, hidden], true),
                    ("out_proj.weight", vec![hidden, hidden], true),
                    ("q_layernorm.weight", vec![head], false),
                    ("k_layernorm.weight", vec![head], false),
                ] {
                    let name = format!("{root}.self_attn.{local}");
                    if matrix {
                        add_safe_matrix(args, &mut common, &name, shape, true)?;
                    } else {
                        common.push(safe(name, shape));
                    }
                }
            }
            OperatorPolicy::SelfAttention(AttentionPolicy::Sliding { .. }) => {
                return Err("LFM2 structural admission does not support sliding attention".into());
            }
        }

        match policy.feed_forward {
            FeedForwardPolicy::Dense => {
                let intermediate =
                    dimension(args.dense_intermediate_size, "dense intermediate size")?;
                for (local, shape) in [
                    ("w1.weight", vec![intermediate, hidden]),
                    ("w2.weight", vec![hidden, intermediate]),
                    ("w3.weight", vec![intermediate, hidden]),
                ] {
                    add_safe_matrix(
                        args,
                        &mut common,
                        &format!("{root}.feed_forward.{local}"),
                        shape,
                        true,
                    )?;
                }
            }
            FeedForwardPolicy::SparseMoe => {
                add_safe_moe(
                    args,
                    layer,
                    &root,
                    allow_derived_packed,
                    &mut common,
                    &mut groups,
                )?;
            }
        }
    }

    SafetensorsCheckpointPlan::new(
        "LFM2 SafeTensors",
        common,
        groups,
        CatalogPolicy::non_strict(),
    )
    .map_err(|error| error.to_string())
}

fn add_safe_moe(
    args: &ModelArgs,
    layer: usize,
    root: &str,
    allow_derived_packed: bool,
    common: &mut Vec<SafetensorsTensorConstraint>,
    groups: &mut Vec<AlternativeLayoutGroup<SafetensorsTensorConstraint>>,
) -> Result<(), String> {
    let hidden = dimension(args.hidden_size, "hidden size")?;
    let experts = dimension(args.num_experts, "expert count")?;
    let intermediate = dimension(args.moe_intermediate_size, "MoE intermediate size")?;
    common.push(safe(
        format!("{root}.feed_forward.gate.weight"),
        vec![experts, hidden],
    ));
    if args.use_expert_bias {
        common.push(safe(
            format!("{root}.feed_forward.expert_bias"),
            vec![experts],
        ));
    }
    let prefix = format!("{root}.feed_forward.experts");
    let gate_up = format!("{prefix}.gate_up_proj");
    let gate = format!("{prefix}.gate_proj");
    let up = format!("{prefix}.up_proj");
    let down = format!("{prefix}.down_proj");
    let gate_up_quantization = args.weight_quantization_for(&gate_up);
    let down_quantization = args.weight_quantization_for(&down);
    let mut variants = vec![LayoutVariant {
        id: "fused packed".into(),
        tensors: [
            safe_matrix_constraints(
                &gate_up,
                vec![
                    experts,
                    checked_mul(2, intermediate, "fused expert width")?,
                    hidden,
                ],
                gate_up_quantization,
            )?,
            safe_matrix_constraints(
                &down,
                vec![experts, hidden, intermediate],
                down_quantization,
            )?,
        ]
        .concat(),
        discriminator_keys: vec![gate_up.clone()],
    }];
    if allow_derived_packed {
        variants.push(LayoutVariant {
            id: "separate packed".into(),
            tensors: [
                safe_matrix_constraints(
                    &gate,
                    vec![experts, intermediate, hidden],
                    gate_up_quantization,
                )?,
                safe_matrix_constraints(
                    &up,
                    vec![experts, intermediate, hidden],
                    gate_up_quantization,
                )?,
                safe_matrix_constraints(
                    &down,
                    vec![experts, hidden, intermediate],
                    down_quantization,
                )?,
            ]
            .concat(),
            discriminator_keys: vec![gate, up],
        });
    }
    if gate_up_quantization.is_none() && down_quantization.is_none() {
        let mut tensors = Vec::new();
        let mut discriminators = Vec::new();
        for expert in 0..experts {
            for (canonical, alias, shape) in [
                (
                    format!("{prefix}.{expert}.w1.weight"),
                    format!("{prefix}.{expert}.gate_proj.weight"),
                    vec![intermediate, hidden],
                ),
                (
                    format!("{prefix}.{expert}.w2.weight"),
                    format!("{prefix}.{expert}.down_proj.weight"),
                    vec![hidden, intermediate],
                ),
                (
                    format!("{prefix}.{expert}.w3.weight"),
                    format!("{prefix}.{expert}.up_proj.weight"),
                    vec![intermediate, hidden],
                ),
            ] {
                discriminators.push(canonical.clone());
                tensors.push(safe(canonical, shape).with_aliases([alias]));
            }
        }
        variants.push(LayoutVariant {
            id: "per-expert split".into(),
            tensors,
            discriminator_keys: discriminators,
        });
    }
    groups.push(AlternativeLayoutGroup {
        id: format!("LFM2 layer {layer} routed expert storage"),
        required: true,
        variants,
    });
    Ok(())
}

fn add_safe_matrix(
    args: &ModelArgs,
    output: &mut Vec<SafetensorsTensorConstraint>,
    name: &str,
    shape: Vec<usize>,
    quantizable: bool,
) -> Result<(), String> {
    let quantization = quantizable
        .then(|| args.weight_quantization_for(name))
        .flatten();
    output.extend(safe_matrix_constraints(name, shape, quantization)?);
    Ok(())
}

fn safe_matrix_constraints(
    name: &str,
    shape: Vec<usize>,
    quantization: Option<WeightQuantization>,
) -> Result<Vec<SafetensorsTensorConstraint>, String> {
    let Some(quantization) = quantization else {
        return Ok(vec![safe(name, shape)]);
    };
    let input = *shape
        .last()
        .ok_or_else(|| format!("LFM2 matrix {name:?} has scalar shape"))?;
    let bits = dimension(quantization.bits(), "quantization bit width")?;
    let group = dimension(quantization.group_size(), "quantization group size")?;
    let packed_bits = checked_mul(input, bits, "affine packing")?;
    if !input.is_multiple_of(group) || !input.is_multiple_of(32) || !packed_bits.is_multiple_of(32)
    {
        return Err(format!(
            "quantized tensor {name:?} input dimension {input} is incompatible with group size {group} and {bits}-bit packing"
        ));
    }
    let mut packed = shape.clone();
    *packed.last_mut().expect("matrix shape") = packed_bits / 32;
    let mut companion = shape;
    *companion.last_mut().expect("matrix shape") = input / group;
    let prefix = outer_prefix(name);
    let aliases = name
        .strip_suffix(".weight")
        .map(|prefix| vec![format!("{prefix}.inner.weight")])
        .unwrap_or_default();
    let companion_dtype = || {
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
    .with_aliases(aliases)];
    constraints.push(
        SafetensorsTensorConstraint::required(
            format!("{prefix}.scales"),
            companion.clone(),
            companion_dtype(),
        )
        .companion(),
    );
    if quantization.has_biases() {
        constraints.push(
            SafetensorsTensorConstraint::required(
                format!("{prefix}.biases"),
                companion,
                companion_dtype(),
            )
            .companion(),
        );
    }
    Ok(constraints)
}

fn safe(key: impl Into<String>, shape: Vec<usize>) -> SafetensorsTensorConstraint {
    SafetensorsTensorConstraint::required(key, shape, StoredDtypeConstraint::Floating)
}

fn outer_prefix(name: &str) -> &str {
    name.strip_suffix(".inner.weight")
        .or_else(|| name.strip_suffix(".weight"))
        .unwrap_or(name)
}

fn validate_expert_prefix_catalogs(
    store: &SafetensorsWeightStore,
    args: &ModelArgs,
    plan: &SafetensorsCheckpointPlan,
    issues: &mut Vec<CheckpointIssue>,
) {
    let allowed = plan
        .layout_groups
        .iter()
        .flat_map(|group| &group.variants)
        .flat_map(|variant| &variant.tensors)
        .flat_map(|tensor| std::iter::once(&tensor.key).chain(&tensor.aliases))
        .cloned()
        .collect::<BTreeSet<_>>();
    for (layer, policy) in args.layer_schedule.iter().enumerate() {
        if policy.feed_forward != FeedForwardPolicy::SparseMoe {
            continue;
        }
        let prefix = format!("model.layers.{layer}.feed_forward.experts.");
        for key in store
            .keys()
            .into_iter()
            .filter(|key| key.starts_with(&prefix) && !allowed.contains(key))
        {
            issues.push(CheckpointIssue {
                kind: CheckpointIssueKind::ConflictingLayout,
                detail: format!(
                    "expert catalog {:?} contains an unexpected, unsupported, or out-of-range tensor {key:?}",
                    prefix.trim_end_matches('.')
                ),
                tensor_name: Some(key),
                tensor_type_code: None,
                metadata_key: None,
            });
        }
    }
}

pub(crate) fn validate_gguf(
    variant: GgufVariant,
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
) -> CheckpointValidation {
    let is_moe = variant.is_moe();
    let translate = |name: &str| model::translate_gguf_weight_name(name, is_moe);
    if let Err(error) = checkpoint.catalog().translated_outputs(translate) {
        return CheckpointValidation::Invalid(vec![CheckpointIssue {
            kind: CheckpointIssueKind::ConflictingLayout,
            detail: error.to_string(),
            tensor_name: None,
            tensor_type_code: None,
            metadata_key: None,
        }]);
    }
    let args = match model::args_from_gguf_catalog(
        checkpoint,
        metadata,
        variant.metadata_name(),
        is_moe,
    ) {
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
    if is_moe {
        issues.extend(validation::validate_matching_gguf_encodings(
            checkpoint,
            args.layer_schedule
                .iter()
                .enumerate()
                .filter_map(|(layer, policy)| {
                    (policy.feed_forward == FeedForwardPolicy::SparseMoe).then_some((
                        format!("blk.{layer}.ffn_gate_exps.weight"),
                        format!("blk.{layer}.ffn_up_exps.weight"),
                    ))
                }),
            "LFM2 MoE",
        ));
    }
    finish(issues)
}

pub(crate) fn gguf_plan(args: &ModelArgs) -> Result<GgufCheckpointPlan, String> {
    let hidden = dimension(args.hidden_size, "hidden size")?;
    let vocab = dimension(args.vocab_size, "vocabulary size")?;
    let heads = dimension(args.num_attention_heads, "attention head count")?;
    let kv_heads = dimension(args.num_key_value_heads, "key/value head count")?;
    if !hidden.is_multiple_of(heads) {
        return Err(format!(
            "LFM2 hidden size {hidden} is not divisible by attention head count {heads}"
        ));
    }
    let head = hidden / heads;
    let key_value = checked_mul(kv_heads, head, "key/value projection width")?;
    let mut tensors = vec![
        gguf(
            "token_embd.weight",
            vec![vocab, hidden],
            TensorOperation::Matrix,
        ),
        gguf(
            "token_embd_norm.weight",
            vec![hidden],
            TensorOperation::Vector,
        ),
    ];
    if !args.tie_word_embeddings {
        tensors.push(gguf(
            "output.weight",
            vec![vocab, hidden],
            TensorOperation::Matrix,
        ));
    }
    for (layer, policy) in args.layer_schedule.iter().enumerate() {
        let block = format!("blk.{layer}");
        tensors.extend([
            gguf(
                format!("{block}.attn_norm.weight"),
                vec![hidden],
                TensorOperation::Vector,
            ),
            gguf(
                format!("{block}.ffn_norm.weight"),
                vec![hidden],
                TensorOperation::Vector,
            ),
        ]);
        match policy.operator {
            OperatorPolicy::CausalConvolution => {
                let kernel = dimension(args.conv_l_cache, "short-convolution width")?;
                tensors.push(gguf(
                    format!("{block}.shortconv.conv.weight"),
                    vec![hidden, kernel],
                    TensorOperation::Dense,
                ));
                tensors.extend([
                    gguf(
                        format!("{block}.shortconv.in_proj.weight"),
                        vec![
                            checked_mul(3, hidden, "short-convolution input width")?,
                            hidden,
                        ],
                        TensorOperation::Matrix,
                    ),
                    gguf(
                        format!("{block}.shortconv.out_proj.weight"),
                        vec![hidden, hidden],
                        TensorOperation::Matrix,
                    ),
                ]);
                if args.conv_bias {
                    tensors.extend([
                        gguf(
                            format!("{block}.shortconv.conv.bias"),
                            vec![hidden],
                            TensorOperation::Vector,
                        ),
                        gguf(
                            format!("{block}.shortconv.in_proj.bias"),
                            vec![checked_mul(3, hidden, "short-convolution bias width")?],
                            TensorOperation::Vector,
                        ),
                        gguf(
                            format!("{block}.shortconv.out_proj.bias"),
                            vec![hidden],
                            TensorOperation::Vector,
                        ),
                    ]);
                }
            }
            OperatorPolicy::SelfAttention(AttentionPolicy::Full) => {
                for (local, shape, operation) in [
                    (
                        "attn_q.weight",
                        vec![hidden, hidden],
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
                        vec![hidden, hidden],
                        TensorOperation::Matrix,
                    ),
                    ("attn_q_norm.weight", vec![head], TensorOperation::Vector),
                    ("attn_k_norm.weight", vec![head], TensorOperation::Vector),
                ] {
                    tensors.push(gguf(format!("{block}.{local}"), shape, operation));
                }
            }
            OperatorPolicy::SelfAttention(AttentionPolicy::Sliding { .. }) => {
                return Err("LFM2 structural admission does not support sliding attention".into());
            }
        }
        match policy.feed_forward {
            FeedForwardPolicy::Dense => {
                let intermediate =
                    dimension(args.dense_intermediate_size, "dense intermediate size")?;
                for (local, shape) in [
                    ("ffn_gate.weight", vec![intermediate, hidden]),
                    ("ffn_down.weight", vec![hidden, intermediate]),
                    ("ffn_up.weight", vec![intermediate, hidden]),
                ] {
                    tensors.push(gguf(
                        format!("{block}.{local}"),
                        shape,
                        TensorOperation::Matrix,
                    ));
                }
            }
            FeedForwardPolicy::SparseMoe => {
                let experts = dimension(args.num_experts, "expert count")?;
                let intermediate = dimension(args.moe_intermediate_size, "MoE intermediate size")?;
                tensors.extend([
                    gguf(
                        format!("{block}.ffn_gate_inp.weight"),
                        vec![experts, hidden],
                        TensorOperation::Matrix,
                    ),
                    gguf(
                        format!("{block}.ffn_gate_exps.weight"),
                        vec![experts, intermediate, hidden],
                        TensorOperation::Matrix,
                    ),
                    gguf(
                        format!("{block}.ffn_up_exps.weight"),
                        vec![experts, intermediate, hidden],
                        TensorOperation::Matrix,
                    ),
                    gguf(
                        format!("{block}.ffn_down_exps.weight"),
                        vec![experts, hidden, intermediate],
                        TensorOperation::Matrix,
                    ),
                ]);
                if args.use_expert_bias {
                    tensors.push(
                        gguf(
                            format!("{block}.ffn_exp_probs_b.bias"),
                            vec![experts],
                            TensorOperation::Vector,
                        )
                        .with_aliases([format!("{block}.exp_probs_b.bias")]),
                    );
                }
            }
        }
    }
    GgufCheckpointPlan::new(
        "LFM2 GGUF",
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
        .ok_or_else(|| format!("LFM2 {name} must be positive, got {value}"))
}

fn checked_mul(left: usize, right: usize, name: &str) -> Result<usize, String> {
    left.checked_mul(right)
        .ok_or_else(|| format!("LFM2 {name} geometry overflows"))
}

fn validation_issues(validation: CheckpointValidation) -> Vec<CheckpointIssue> {
    match validation {
        CheckpointValidation::Exact => Vec::new(),
        CheckpointValidation::Invalid(issues) => issues,
        CheckpointValidation::Unverified(issue) => vec![issue],
    }
}

fn finish(issues: Vec<CheckpointIssue>) -> CheckpointValidation {
    CheckpointValidation::from_issues(issues)
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
    use eredu_checkpoint::AffineQuantization;

    #[test]
    fn affine_constraints_exclude_runtime_shape_from_packed_storage() {
        let quantization = WeightQuantization::Affine(AffineQuantization::new(32, 4).unwrap());
        let constraints = safe_matrix_constraints(
            "model.layers.0.feed_forward.w1.weight",
            vec![8, 64],
            Some(quantization),
        )
        .unwrap();
        assert_eq!(constraints[0].shape, [8, 8]);
        assert_eq!(
            constraints[0].dtype,
            StoredDtypeConstraint::Exact(StoredDtype::U32)
        );
        assert_eq!(constraints[1].shape, [8, 2]);
    }

    #[test]
    fn plan_uses_each_feed_forward_policy_in_schedule_order() {
        let mut args = model::model_args_from_config_value(&serde_json::json!({
            "model_type": "lfm2_moe", "vocab_size": 32, "hidden_size": 16,
            "intermediate_size": 24, "num_hidden_layers": 3,
            "num_attention_heads": 4, "num_key_value_heads": 2,
            "max_position_embeddings": 64, "norm_eps": 1e-5,
            "layer_types": ["conv", "full_attention", "conv"],
            "conv_L_cache": 3, "block_auto_adjust_ff_dim": false,
            "moe_intermediate_size": 8, "num_dense_layers": 1,
            "num_experts": 2, "num_experts_per_tok": 1
        }))
        .unwrap();
        args.layer_schedule = crate::LayerSchedule::new(
            3,
            vec![
                super::super::model::LayerPolicy {
                    operator: OperatorPolicy::CausalConvolution,
                    feed_forward: FeedForwardPolicy::SparseMoe,
                },
                super::super::model::LayerPolicy {
                    operator: OperatorPolicy::SelfAttention(AttentionPolicy::Full),
                    feed_forward: FeedForwardPolicy::Dense,
                },
                super::super::model::LayerPolicy {
                    operator: OperatorPolicy::CausalConvolution,
                    feed_forward: FeedForwardPolicy::SparseMoe,
                },
            ],
        )
        .unwrap();
        let plan = safetensors_plan(&args, true).unwrap();
        let names = plan
            .common_tensors
            .iter()
            .map(|tensor| tensor.key.as_str())
            .chain(
                plan.layout_groups
                    .iter()
                    .flat_map(|group| &group.variants)
                    .flat_map(|variant| &variant.tensors)
                    .map(|tensor| tensor.key.as_str()),
            )
            .collect::<BTreeSet<_>>();
        assert!(names.contains("model.layers.0.feed_forward.gate.weight"));
        assert!(!names.contains("model.layers.0.feed_forward.w1.weight"));
        assert!(names.contains("model.layers.1.feed_forward.w1.weight"));
        assert!(!names.contains("model.layers.1.feed_forward.gate.weight"));
        assert!(names.contains("model.layers.2.feed_forward.gate.weight"));
    }
}
