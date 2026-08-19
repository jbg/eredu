//! Architecture-owned checkpoint contracts for Nemotron-H.

//!
//! Nemotron-H owns its hybrid layer schedule, public/runtime aliases, MTP
//! geometry, routed-expert storage alternatives, and GGUF name catalog here.
//! The generic checkpoint runtime only evaluates the resulting physical
//! constraints and remains unaware of recurrent, expert, and MTP semantics.

use eredu_checkpoint::{StoredDtype, WeightQuantization};

use std::collections::{BTreeSet, HashMap};

use safemlx::ops::{GgufCheckpoint, GgufMetadataValue};
use serde_json::Value;

use super::model::{self, LayerPolicy, ModelArgs};
use crate::backend::mlx::runtime::checkpoint::{
    contract::{CheckpointIssue, CheckpointIssueKind, CheckpointValidation},
    store::{SafetensorsWeightStore, WeightStore},
    validation,
};
use eredu_checkpoint::schema::{
    AlternativeLayoutGroup, CatalogPolicy, GgufCheckpointPlan, GgufTensorConstraint,
    GgufTypeConstraint, LayoutVariant, SafetensorsCheckpointPlan, SafetensorsTensorConstraint,
    StoredDtypeConstraint, TensorOperation,
};

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub(crate) enum GgufVariant {
    Dense,
    Moe,
}

impl GgufVariant {
    const fn metadata_name(self) -> &'static str {
        match self {
            Self::Dense => "nemotron_h",
            Self::Moe => "nemotron_h_moe",
        }
    }
}

pub(crate) fn validate_safetensors(
    config: &Value,
    store: &SafetensorsWeightStore,
) -> CheckpointValidation {
    let args = match model::model_args_from_config_value(config) {
        Ok(args) => args,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    let plan = match safetensors_plan(&args) {
        Ok(plan) => plan,
        Err(error) if error.starts_with("quantized tensor ") => {
            return CheckpointValidation::Invalid(vec![CheckpointIssue {
                kind: CheckpointIssueKind::CompanionMismatch,
                detail: error,
                tensor_name: None,
                tensor_type_code: None,
                metadata_key: Some("quantization_config.quant_method".into()),
            }]);
        }
        Err(error) => return invalid_geometry(error),
    };
    let mut issues = validation_issues(validation::validate_safetensors_plan(store, &plan));
    validate_expert_layout_conflicts(store, &args, &mut issues);
    validate_native_quantized_expert_layouts(store, &args, &mut issues);
    CheckpointValidation::from_issues(issues)
}

pub(crate) fn safetensors_plan(args: &ModelArgs) -> Result<SafetensorsCheckpointPlan, String> {
    let hidden = dimension(args.hidden_size, "hidden size")?;
    let vocab = dimension(args.vocab_size, "vocabulary size")?;
    let mut common = Vec::new();
    let mut groups = Vec::new();
    add_safe_alias(
        args,
        &mut common,
        "backbone.embeddings.weight",
        "model.embeddings.weight",
        vec![vocab, hidden],
    )?;
    add_safe_alias(
        args,
        &mut common,
        "backbone.norm_f.weight",
        "model.norm_f.weight",
        vec![hidden],
    )?;
    if !args.tie_word_embeddings {
        add_safe_matrix(args, &mut common, "lm_head.weight", vec![vocab, hidden])?;
    }

    for (layer, policy) in args.layer_schedule.iter().copied().enumerate() {
        let official = format!("backbone.layers.{layer}");
        let canonical = format!("model.layers.{layer}");
        add_safe_alias(
            args,
            &mut common,
            &format!("{official}.norm.weight"),
            &format!("{canonical}.norm.weight"),
            vec![hidden],
        )?;
        add_safe_layer(
            args,
            policy,
            &official,
            &canonical,
            &format!("layer {layer}"),
            &mut common,
            &mut groups,
        )?;
    }

    if args.num_nextn_predict_layers > 0 {
        let policies = args.mtp_policies().map_err(|error| error.to_string())?;
        let steps = dimension(args.num_nextn_predict_layers, "MTP prediction-step count")?;
        if !policies.len().is_multiple_of(steps) {
            return Err(format!(
                "Nemotron-H MTP policy count {} is not divisible by prediction-step count {steps}",
                policies.len()
            ));
        }
        let pattern_len = policies.len() / steps;
        if pattern_len == 0 {
            return Err("Nemotron-H MTP operator pattern cannot be empty".into());
        }
        for (layer, policy) in policies.iter().copied().enumerate() {
            let official = format!("mtp.layers.{layer}");
            let canonical = format!("model.mtp.layers.{layer}");
            add_safe_alias(
                args,
                &mut common,
                &format!("{official}.norm.weight"),
                &format!("{canonical}.norm.weight"),
                vec![hidden],
            )?;
            add_safe_mtp_layer(
                args,
                policy,
                &official,
                &canonical,
                &format!("MTP physical layer {layer}"),
                &mut common,
                &mut groups,
            )?;
        }
        for step in 0..steps {
            let start = checked_mul(step, pattern_len, "MTP physical layer offset")?;
            let end = start
                .checked_add(pattern_len - 1)
                .ok_or_else(|| "Nemotron-H MTP physical layer range overflows".to_string())?;
            for (layer, suffix, shape) in [
                (start, "enorm.weight", vec![hidden]),
                (start, "hnorm.weight", vec![hidden]),
                (
                    start,
                    "eh_proj.weight",
                    vec![
                        hidden,
                        checked_mul(hidden, 2, "MTP concatenated hidden width")?,
                    ],
                ),
                (end, "final_layernorm.weight", vec![hidden]),
            ] {
                add_safe_alias(
                    args,
                    &mut common,
                    &format!("mtp.layers.{layer}.{suffix}"),
                    &format!("model.mtp.layers.{layer}.{suffix}"),
                    shape,
                )?;
            }
        }
    }

    SafetensorsCheckpointPlan::new(
        "Nemotron-H SafeTensors",
        common,
        groups,
        CatalogPolicy::strict(),
    )
    .map_err(|error| error.to_string())
}

fn add_safe_layer(
    args: &ModelArgs,
    policy: LayerPolicy,
    official: &str,
    canonical: &str,
    label: &str,
    common: &mut Vec<SafetensorsTensorConstraint>,
    groups: &mut Vec<AlternativeLayoutGroup<SafetensorsTensorConstraint>>,
) -> Result<(), String> {
    match policy {
        LayerPolicy::Mamba => add_safe_mamba(args, official, canonical, common),
        LayerPolicy::SelfAttention(_) => {
            add_safe_attention(args, official, canonical, "attention", common)
        }
        LayerPolicy::DenseMlp => add_safe_dense(args, official, canonical, common),
        LayerPolicy::SparseMoe => {
            add_safe_moe(args, official, canonical, "moe", label, common, groups)
        }
    }
}

fn add_safe_mtp_layer(
    args: &ModelArgs,
    policy: LayerPolicy,
    official: &str,
    canonical: &str,
    label: &str,
    common: &mut Vec<SafetensorsTensorConstraint>,
    groups: &mut Vec<AlternativeLayoutGroup<SafetensorsTensorConstraint>>,
) -> Result<(), String> {
    match policy {
        LayerPolicy::SelfAttention(_) => {
            add_safe_attention(args, official, canonical, "mixer", common)
        }
        LayerPolicy::SparseMoe => {
            add_safe_moe(args, official, canonical, "mixer", label, common, groups)
        }
        _ => Err(format!(
            "Nemotron-H {label} uses unsupported policy {policy:?}"
        )),
    }
}

fn add_safe_mamba(
    args: &ModelArgs,
    official: &str,
    canonical: &str,
    output: &mut Vec<SafetensorsTensorConstraint>,
) -> Result<(), String> {
    let hidden = dimension(args.hidden_size, "hidden size")?;
    let heads = dimension(args.mamba_num_heads, "Mamba head count")?;
    let head = dimension(args.mamba_head_dim, "Mamba head width")?;
    let groups = dimension(args.n_groups, "Mamba state group count")?;
    let state = dimension(args.ssm_state_size, "Mamba state width")?;
    let kernel = dimension(args.conv_kernel, "Mamba convolution width")?;
    let intermediate = checked_mul(heads, head, "Mamba intermediate width")?;
    let grouped_state = checked_mul(groups, state, "Mamba grouped state width")?;
    let conv = checked_add(
        intermediate,
        checked_mul(2, grouped_state, "Mamba B/C state width")?,
        "Mamba convolution channels",
    )?;
    let projection = checked_add(
        checked_add(intermediate, conv, "Mamba projection content width")?,
        heads,
        "Mamba projection width",
    )?;
    for (source, target, shape) in [
        ("in_proj.weight", "in_proj.weight", vec![projection, hidden]),
        ("conv1d.weight", "conv1d.weight", vec![conv, 1, kernel]),
        ("dt_bias", "dt_bias", vec![heads]),
        ("A_log", "A_log", vec![heads]),
        ("D", "D", vec![heads]),
        ("norm.weight", "norm.weight", vec![intermediate]),
        (
            "out_proj.weight",
            "out_proj.weight",
            vec![hidden, intermediate],
        ),
    ] {
        add_safe_alias(
            args,
            output,
            &format!("{official}.mixer.{source}"),
            &format!("{canonical}.mamba.{target}"),
            shape,
        )?;
    }
    if args.use_conv_bias {
        add_safe_alias(
            args,
            output,
            &format!("{official}.mixer.conv1d.bias"),
            &format!("{canonical}.mamba.conv1d.bias"),
            vec![conv],
        )?;
    }
    if args.use_bias {
        for (projection_name, size) in [("in_proj", projection), ("out_proj", hidden)] {
            add_safe_alias(
                args,
                output,
                &format!("{official}.mixer.{projection_name}.bias"),
                &format!("{canonical}.mamba.{projection_name}.bias"),
                vec![size],
            )?;
        }
    }
    Ok(())
}

fn add_safe_attention(
    args: &ModelArgs,
    official: &str,
    canonical: &str,
    canonical_field: &str,
    output: &mut Vec<SafetensorsTensorConstraint>,
) -> Result<(), String> {
    let hidden = dimension(args.hidden_size, "hidden size")?;
    let head = dimension(args.head_dim, "attention head width")?;
    let query = checked_mul(
        dimension(args.num_attention_heads, "attention head count")?,
        head,
        "query projection width",
    )?;
    let key_value = checked_mul(
        dimension(args.num_key_value_heads, "key/value head count")?,
        head,
        "key/value projection width",
    )?;
    for (projection, outer, inner) in [
        ("q_proj", query, hidden),
        ("k_proj", key_value, hidden),
        ("v_proj", key_value, hidden),
        ("o_proj", hidden, query),
    ] {
        add_safe_alias(
            args,
            output,
            &format!("{official}.mixer.{projection}.weight"),
            &format!("{canonical}.{canonical_field}.{projection}.weight"),
            vec![outer, inner],
        )?;
        if args.attention_bias {
            add_safe_alias(
                args,
                output,
                &format!("{official}.mixer.{projection}.bias"),
                &format!("{canonical}.{canonical_field}.{projection}.bias"),
                vec![outer],
            )?;
        }
    }
    Ok(())
}

fn add_safe_dense(
    args: &ModelArgs,
    official: &str,
    canonical: &str,
    output: &mut Vec<SafetensorsTensorConstraint>,
) -> Result<(), String> {
    let hidden = dimension(args.hidden_size, "hidden size")?;
    let intermediate = dimension(args.intermediate_size, "dense intermediate width")?;
    for (projection, shape, bias) in [
        ("up_proj", vec![intermediate, hidden], intermediate),
        ("down_proj", vec![hidden, intermediate], hidden),
    ] {
        add_safe_alias(
            args,
            output,
            &format!("{official}.mixer.{projection}.weight"),
            &format!("{canonical}.mlp.{projection}.weight"),
            shape,
        )?;
        if args.mlp_bias {
            add_safe_alias(
                args,
                output,
                &format!("{official}.mixer.{projection}.bias"),
                &format!("{canonical}.mlp.{projection}.bias"),
                vec![bias],
            )?;
        }
    }
    Ok(())
}

fn add_safe_moe(
    args: &ModelArgs,
    official: &str,
    canonical: &str,
    canonical_field: &str,
    label: &str,
    common: &mut Vec<SafetensorsTensorConstraint>,
    groups: &mut Vec<AlternativeLayoutGroup<SafetensorsTensorConstraint>>,
) -> Result<(), String> {
    let hidden = dimension(args.hidden_size, "hidden size")?;
    let experts = dimension(args.n_routed_experts, "routed expert count")?;
    let intermediate = dimension(args.moe_intermediate_size, "expert intermediate width")?;
    let shared = dimension(
        args.moe_shared_expert_intermediate_size,
        "shared expert intermediate width",
    )?;
    for (suffix, shape) in [
        ("gate.weight", vec![experts, hidden]),
        ("gate.e_score_correction_bias", vec![experts]),
        ("shared_experts.up_proj.weight", vec![shared, hidden]),
        ("shared_experts.down_proj.weight", vec![hidden, shared]),
    ] {
        add_safe_alias(
            args,
            common,
            &format!("{official}.mixer.{suffix}"),
            &format!("{canonical}.{canonical_field}.{suffix}"),
            shape,
        )?;
    }
    if args.mlp_bias {
        for (projection, size) in [("up_proj", shared), ("down_proj", hidden)] {
            add_safe_alias(
                args,
                common,
                &format!("{official}.mixer.shared_experts.{projection}.bias"),
                &format!("{canonical}.{canonical_field}.shared_experts.{projection}.bias"),
                vec![size],
            )?;
        }
    }

    let official_experts = format!("{official}.mixer.experts");
    let canonical_experts = format!("{canonical}.{canonical_field}.experts");
    let mut packed = Vec::new();
    add_safe_alias(
        args,
        &mut packed,
        &format!("{official_experts}.up_proj"),
        &format!("{canonical_experts}.up_proj"),
        vec![experts, intermediate, hidden],
    )?;
    add_safe_alias(
        args,
        &mut packed,
        &format!("{official_experts}.down_proj"),
        &format!("{canonical_experts}.down_proj"),
        vec![experts, hidden, intermediate],
    )?;
    let mut split = Vec::new();
    for expert in 0..experts {
        add_safe_alias(
            args,
            &mut split,
            &format!("{official_experts}.{expert}.up_proj.weight"),
            &format!("{canonical_experts}.{expert}.up_proj.weight"),
            vec![intermediate, hidden],
        )?;
        add_safe_alias(
            args,
            &mut split,
            &format!("{official_experts}.{expert}.down_proj.weight"),
            &format!("{canonical_experts}.{expert}.down_proj.weight"),
            vec![hidden, intermediate],
        )?;
    }
    groups.push(AlternativeLayoutGroup {
        id: format!("Nemotron-H {label} routed experts"),
        required: true,
        variants: vec![
            LayoutVariant {
                id: "packed".into(),
                discriminator_keys: packed
                    .iter()
                    .filter(|tensor| tensor.role == eredu_checkpoint::schema::TensorRole::Tensor)
                    .map(|tensor| tensor.key.clone())
                    .collect(),
                tensors: packed,
            },
            LayoutVariant {
                id: "split".into(),
                discriminator_keys: split
                    .iter()
                    .filter(|tensor| tensor.role == eredu_checkpoint::schema::TensorRole::Tensor)
                    .map(|tensor| tensor.key.clone())
                    .collect(),
                tensors: split,
            },
        ],
    });
    Ok(())
}

fn add_safe_alias(
    args: &ModelArgs,
    output: &mut Vec<SafetensorsTensorConstraint>,
    official: &str,
    canonical: &str,
    shape: Vec<usize>,
) -> Result<(), String> {
    let quantizable = shape.len() >= 2
        && (canonical.ends_with(".weight")
            || canonical.ends_with(".experts.up_proj")
            || canonical.ends_with(".experts.down_proj"))
        && !canonical.contains(".conv1d.weight")
        && !canonical.ends_with(".moe.gate.weight");
    let quantization = quantizable
        .then(|| args.weight_quantization_for(canonical))
        .flatten();
    output.extend(safe_alias_constraints(
        official,
        canonical,
        shape,
        quantization,
    )?);
    Ok(())
}

fn add_safe_matrix(
    args: &ModelArgs,
    output: &mut Vec<SafetensorsTensorConstraint>,
    name: &str,
    shape: Vec<usize>,
) -> Result<(), String> {
    let quantization = args.weight_quantization_for(name);
    output.extend(safe_alias_constraints(name, name, shape, quantization)?);
    Ok(())
}

fn safe_alias_constraints(
    official: &str,
    canonical: &str,
    shape: Vec<usize>,
    quantization: Option<WeightQuantization>,
) -> Result<Vec<SafetensorsTensorConstraint>, String> {
    let mut names = vec![official.to_string(), canonical.to_string()];
    if quantization.is_some() {
        names.extend(names.clone().into_iter().filter_map(|name| {
            name.strip_suffix(".weight")
                .map(|prefix| format!("{prefix}.inner.weight"))
        }));
    }
    names.sort();
    names.dedup();
    let primary = official.to_string();
    let aliases = names
        .iter()
        .filter(|name| **name != primary)
        .cloned()
        .collect::<Vec<_>>();
    let Some(quantization) = quantization else {
        return Ok(vec![SafetensorsTensorConstraint::required(
            primary,
            shape,
            StoredDtypeConstraint::Floating,
        )
        .with_aliases(aliases)]);
    };
    let input = *shape
        .last()
        .ok_or_else(|| format!("Nemotron-H matrix {canonical:?} has scalar shape"))?;
    let bits = dimension(quantization.bits(), "quantization bit width")?;
    let group = dimension(quantization.group_size(), "quantization group size")?;
    let packed_bits = checked_mul(input, bits, "affine packing")?;
    if !input.is_multiple_of(group) || !input.is_multiple_of(32) || !packed_bits.is_multiple_of(32)
    {
        return Err(format!(
            "quantized tensor {canonical:?} input dimension {input} is incompatible with group size {group} and {bits}-bit packing"
        ));
    }
    let mut packed_shape = shape.clone();
    *packed_shape.last_mut().expect("matrix shape") = packed_bits / 32;
    let mut companion_shape = shape;
    *companion_shape.last_mut().expect("matrix shape") = input / group;
    let mut constraints = vec![SafetensorsTensorConstraint::required(
        primary,
        packed_shape,
        StoredDtypeConstraint::Exact(StoredDtype::U32),
    )
    .with_aliases(aliases)];
    let companion_dtype = || {
        StoredDtypeConstraint::OneOf(vec![
            StoredDtype::F16,
            StoredDtype::BF16,
            StoredDtype::F32,
            StoredDtype::U8,
        ])
    };
    let prefixes = names
        .iter()
        .map(|name| outer_prefix(name).to_string())
        .collect::<BTreeSet<_>>();
    for (suffix, required) in [("scales", true), ("biases", quantization.has_biases())] {
        if !required {
            continue;
        }
        let primary_companion = format!("{}.{}", outer_prefix(official), suffix);
        let aliases = prefixes
            .iter()
            .map(|prefix| format!("{prefix}.{suffix}"))
            .filter(|name| name != &primary_companion)
            .collect::<Vec<_>>();
        constraints.push(
            SafetensorsTensorConstraint::required(
                primary_companion,
                companion_shape.clone(),
                companion_dtype(),
            )
            .with_aliases(aliases)
            .companion(),
        );
    }
    Ok(constraints)
}

fn outer_prefix(name: &str) -> &str {
    name.strip_suffix(".inner.weight")
        .or_else(|| name.strip_suffix(".weight"))
        .unwrap_or(name)
}

fn validate_native_quantized_expert_layouts(
    store: &SafetensorsWeightStore,
    args: &ModelArgs,
    issues: &mut Vec<CheckpointIssue>,
) {
    if args.quantization.is_none() {
        return;
    }
    let keys = store.keys().into_iter().collect::<BTreeSet<_>>();
    for (label, official, canonical) in expert_layout_roots(args) {
        let split = (0..args.n_routed_experts as usize)
            .flat_map(|expert| {
                [
                    format!("{official}.{expert}.up_proj.weight"),
                    format!("{canonical}.{expert}.up_proj.weight"),
                    format!("{official}.{expert}.down_proj.weight"),
                    format!("{canonical}.{expert}.down_proj.weight"),
                ]
            })
            .find(|name| keys.contains(name));
        if let Some(name) = split {
            issues.push(CheckpointIssue {
                kind: CheckpointIssueKind::ConflictingLayout,
                detail: format!(
                    "checkpoint-native quantized Nemotron-H {label} requires packed routed expert banks"
                ),
                tensor_name: Some(name),
                tensor_type_code: None,
                metadata_key: Some("quantization".into()),
            });
        }
    }
}

fn validate_expert_layout_conflicts(
    store: &SafetensorsWeightStore,
    args: &ModelArgs,
    issues: &mut Vec<CheckpointIssue>,
) {
    let keys = store.keys().into_iter().collect::<BTreeSet<_>>();
    for (label, official, canonical) in expert_layout_roots(args) {
        let packed = [
            format!("{official}.up_proj"),
            format!("{canonical}.up_proj"),
            format!("{official}.down_proj"),
            format!("{canonical}.down_proj"),
        ];
        let split = (0..args.n_routed_experts as usize)
            .flat_map(|expert| {
                [
                    format!("{official}.{expert}.up_proj.weight"),
                    format!("{canonical}.{expert}.up_proj.weight"),
                    format!("{official}.{expert}.down_proj.weight"),
                    format!("{canonical}.{expert}.down_proj.weight"),
                ]
            })
            .find(|name| keys.contains(name));
        if packed.iter().any(|name| keys.contains(name)) {
            if let Some(name) = split {
                let group_id = format!("Nemotron-H {label} routed experts");
                issues.retain(|issue| {
                    !issue.detail.contains(&group_id)
                        && !issue.tensor_name.as_deref().is_some_and(|name| {
                            name.starts_with(&format!("{official}."))
                                || name.starts_with(&format!("{canonical}."))
                        })
                });
                issues.push(CheckpointIssue {
                    kind: CheckpointIssueKind::ConflictingLayout,
                    detail: format!(
                        "Nemotron-H {label} mixes packed and split routed expert tensors"
                    ),
                    tensor_name: Some(name),
                    tensor_type_code: None,
                    metadata_key: None,
                });
            }
        }
    }
}

fn expert_layout_roots(args: &ModelArgs) -> Vec<(String, String, String)> {
    let mut roots = args
        .layer_schedule
        .iter()
        .enumerate()
        .filter(|(_, policy)| **policy == LayerPolicy::SparseMoe)
        .map(|(layer, _)| {
            (
                format!("layer {layer}"),
                format!("backbone.layers.{layer}.mixer.experts"),
                format!("model.layers.{layer}.moe.experts"),
            )
        })
        .collect::<Vec<_>>();
    if let Ok(policies) = args.mtp_policies() {
        roots.extend(
            policies
                .iter()
                .enumerate()
                .filter(|(_, policy)| **policy == LayerPolicy::SparseMoe)
                .map(|(layer, _)| {
                    (
                        format!("MTP physical layer {layer}"),
                        format!("mtp.layers.{layer}.mixer.experts"),
                        format!("model.mtp.layers.{layer}.mixer.experts"),
                    )
                }),
        );
    }
    roots
}

pub(crate) fn validate_gguf(
    variant: GgufVariant,
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
    let args =
        match model::model_args_from_gguf_catalog(checkpoint, metadata, variant.metadata_name()) {
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
    validation::validate_gguf_plan(checkpoint, &plan)
}

pub(crate) fn gguf_plan(args: &ModelArgs) -> Result<GgufCheckpointPlan, String> {
    let hidden = dimension(args.hidden_size, "hidden size")?;
    let vocab = dimension(args.vocab_size, "vocabulary size")?;
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
    for (layer, policy) in args.layer_schedule.iter().copied().enumerate() {
        let block = format!("blk.{layer}");
        tensors.push(gguf(
            format!("{block}.attn_norm.weight"),
            vec![hidden],
            TensorOperation::Vector,
        ));
        match policy {
            LayerPolicy::Mamba => add_gguf_mamba(args, &block, &mut tensors)?,
            LayerPolicy::SelfAttention(_) => add_gguf_attention(args, &block, &mut tensors)?,
            LayerPolicy::DenseMlp => add_gguf_dense(args, &block, &mut tensors)?,
            LayerPolicy::SparseMoe => add_gguf_moe(args, &block, &mut tensors)?,
        }
    }
    let mut policy = CatalogPolicy::strict();
    policy.allowed_prefixes.push("rope_freqs.".into());
    GgufCheckpointPlan::new("Nemotron-H", tensors, Vec::new(), policy)
        .map_err(|error| error.to_string())
}

fn add_gguf_mamba(
    args: &ModelArgs,
    block: &str,
    output: &mut Vec<GgufTensorConstraint>,
) -> Result<(), String> {
    let hidden = dimension(args.hidden_size, "hidden size")?;
    let heads = dimension(args.mamba_num_heads, "Mamba head count")?;
    let intermediate = checked_mul(
        heads,
        dimension(args.mamba_head_dim, "Mamba head width")?,
        "Mamba intermediate width",
    )?;
    let conv = checked_add(
        intermediate,
        checked_mul(
            2,
            checked_mul(
                dimension(args.n_groups, "Mamba state group count")?,
                dimension(args.ssm_state_size, "Mamba state width")?,
                "Mamba grouped state width",
            )?,
            "Mamba B/C state width",
        )?,
        "Mamba convolution channels",
    )?;
    let projection = checked_add(
        checked_add(intermediate, conv, "Mamba projection content width")?,
        heads,
        "Mamba projection width",
    )?;
    output.extend([
        gguf(
            format!("{block}.ssm_in.weight"),
            vec![projection, hidden],
            TensorOperation::Matrix,
        ),
        gguf(
            format!("{block}.ssm_conv1d.weight"),
            vec![
                conv,
                dimension(args.conv_kernel, "Mamba convolution width")?,
            ],
            TensorOperation::Dense,
        ),
        gguf(
            format!("{block}.ssm_dt.bias"),
            vec![heads],
            TensorOperation::Vector,
        ),
        gguf(
            format!("{block}.ssm_a"),
            vec![heads],
            TensorOperation::Dense,
        ),
        gguf(
            format!("{block}.ssm_d"),
            vec![heads],
            TensorOperation::Vector,
        ),
        gguf(
            format!("{block}.ssm_norm.weight"),
            vec![intermediate],
            TensorOperation::Vector,
        ),
        gguf(
            format!("{block}.ssm_out.weight"),
            vec![hidden, intermediate],
            TensorOperation::Matrix,
        ),
    ]);
    if args.use_conv_bias {
        output.push(gguf(
            format!("{block}.ssm_conv1d.bias"),
            vec![conv],
            TensorOperation::Vector,
        ));
    }
    if args.use_bias {
        output.extend([
            gguf(
                format!("{block}.ssm_in.bias"),
                vec![projection],
                TensorOperation::Vector,
            ),
            gguf(
                format!("{block}.ssm_out.bias"),
                vec![hidden],
                TensorOperation::Vector,
            ),
        ]);
    }
    Ok(())
}

fn add_gguf_attention(
    args: &ModelArgs,
    block: &str,
    output: &mut Vec<GgufTensorConstraint>,
) -> Result<(), String> {
    let hidden = dimension(args.hidden_size, "hidden size")?;
    let head = dimension(args.head_dim, "attention head width")?;
    let query = checked_mul(
        dimension(args.num_attention_heads, "attention head count")?,
        head,
        "query projection width",
    )?;
    let key_value = checked_mul(
        dimension(args.num_key_value_heads, "key/value head count")?,
        head,
        "key/value projection width",
    )?;
    for (name, outer, inner) in [
        ("attn_q", query, hidden),
        ("attn_k", key_value, hidden),
        ("attn_v", key_value, hidden),
        ("attn_output", hidden, query),
    ] {
        output.push(gguf(
            format!("{block}.{name}.weight"),
            vec![outer, inner],
            TensorOperation::Matrix,
        ));
        if args.attention_bias {
            output.push(gguf(
                format!("{block}.{name}.bias"),
                vec![outer],
                TensorOperation::Vector,
            ));
        }
    }
    Ok(())
}

fn add_gguf_dense(
    args: &ModelArgs,
    block: &str,
    output: &mut Vec<GgufTensorConstraint>,
) -> Result<(), String> {
    let hidden = dimension(args.hidden_size, "hidden size")?;
    let intermediate = dimension(args.intermediate_size, "dense intermediate width")?;
    output.extend([
        gguf(
            format!("{block}.ffn_up.weight"),
            vec![intermediate, hidden],
            TensorOperation::Matrix,
        ),
        gguf(
            format!("{block}.ffn_down.weight"),
            vec![hidden, intermediate],
            TensorOperation::Matrix,
        ),
    ]);
    if args.mlp_bias {
        output.extend([
            gguf(
                format!("{block}.ffn_up.bias"),
                vec![intermediate],
                TensorOperation::Vector,
            ),
            gguf(
                format!("{block}.ffn_down.bias"),
                vec![hidden],
                TensorOperation::Vector,
            ),
        ]);
    }
    Ok(())
}

fn add_gguf_moe(
    args: &ModelArgs,
    block: &str,
    output: &mut Vec<GgufTensorConstraint>,
) -> Result<(), String> {
    let hidden = dimension(args.hidden_size, "hidden size")?;
    let experts = dimension(args.n_routed_experts, "routed expert count")?;
    let intermediate = dimension(args.moe_intermediate_size, "expert intermediate width")?;
    let shared = dimension(
        args.moe_shared_expert_intermediate_size,
        "shared expert intermediate width",
    )?;
    output.extend([
        gguf(
            format!("{block}.ffn_gate_inp.weight"),
            vec![experts, hidden],
            TensorOperation::Matrix,
        ),
        gguf(
            format!("{block}.exp_probs_b.bias"),
            vec![experts],
            TensorOperation::Vector,
        )
        .with_aliases([format!("{block}.ffn_exp_probs_b.bias")]),
        gguf(
            format!("{block}.ffn_up_shexp.weight"),
            vec![shared, hidden],
            TensorOperation::Matrix,
        ),
        gguf(
            format!("{block}.ffn_down_shexp.weight"),
            vec![hidden, shared],
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
    if args.mlp_bias {
        output.extend([
            gguf(
                format!("{block}.ffn_up_shexp.bias"),
                vec![shared],
                TensorOperation::Vector,
            ),
            gguf(
                format!("{block}.ffn_down_shexp.bias"),
                vec![hidden],
                TensorOperation::Vector,
            ),
        ]);
    }
    Ok(())
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
        .ok_or_else(|| format!("Nemotron-H {name} must be positive, got {value}"))
}

fn checked_add(left: usize, right: usize, name: &str) -> Result<usize, String> {
    left.checked_add(right)
        .ok_or_else(|| format!("Nemotron-H {name} geometry overflows"))
}

fn checked_mul(left: usize, right: usize, name: &str) -> Result<usize, String> {
    left.checked_mul(right)
        .ok_or_else(|| format!("Nemotron-H {name} geometry overflows"))
}

fn validation_issues(validation: CheckpointValidation) -> Vec<CheckpointIssue> {
    match validation {
        CheckpointValidation::Exact => Vec::new(),
        CheckpointValidation::Invalid(issues) => issues,
        CheckpointValidation::Unverified(issue) => vec![issue],
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

    #[test]
    fn expert_layouts_are_architecture_owned_alternatives() {
        let config = serde_json::json!({
            "model_type": "nemotron_h",
            "vocab_size": 32,
            "tie_word_embeddings": true,
            "hidden_size": 16,
            "intermediate_size": 24,
            "num_hidden_layers": 1,
            "hybrid_override_pattern": "E",
            "num_attention_heads": 2,
            "head_dim": 8,
            "num_key_value_heads": 1,
            "ssm_state_size": 4,
            "mamba_num_heads": 2,
            "n_groups": 1,
            "mamba_head_dim": 8,
            "conv_kernel": 4,
            "n_routed_experts": 2,
            "n_shared_experts": 1,
            "moe_intermediate_size": 12,
            "moe_shared_expert_intermediate_size": 12,
            "num_experts_per_tok": 1,
            "n_group": 1,
            "topk_group": 1
        });
        let args = model::model_args_from_config_value(&config).unwrap();
        let plan = safetensors_plan(&args).unwrap();
        assert_eq!(plan.layout_groups.len(), 1);
        assert_eq!(
            plan.layout_groups[0]
                .variants
                .iter()
                .map(|variant| variant.id.as_str())
                .collect::<Vec<_>>(),
            ["packed", "split"]
        );
    }
}
