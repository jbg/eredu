//! Architecture-owned checkpoint contracts for Qwen3.5 and Qwen3-Next.
//!
//! Normalized geometry, tensor names, aliases, physical layouts, and
//! quantization exclusions live here. The runtime checkpoint engine remains
//! architecture-neutral and only evaluates the resulting declarative plan.

use std::collections::BTreeSet;

use serde_json::Value;

use super::{qwen3_5 as qwen35, qwen3_next};
use crate::runtime::{
    attention::AttentionPolicy,
    checkpoint::{
        contract::{CheckpointIssue, CheckpointIssueKind, CheckpointValidation},
        quantization::WeightQuantization,
        schema::{
            AlternativeLayoutGroup, CatalogPolicy, LayoutVariant, SafetensorsCheckpointPlan,
            SafetensorsTensorConstraint, StoredDtypeConstraint, TensorOperation,
        },
        store::{SafetensorsWeightStore, StoredDtype, WeightStore},
        validation,
    },
};

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub(crate) enum SafetensorsVariant {
    Qwen3Next,
    Qwen35,
}

impl SafetensorsVariant {
    const fn label(self) -> &'static str {
        match self {
            Self::Qwen3Next => "Qwen3-Next",
            Self::Qwen35 => "Qwen3.5",
        }
    }

    const fn accepts_fused_linear_projection(self) -> bool {
        matches!(self, Self::Qwen3Next)
    }
}

pub(crate) fn validate_qwen3_next_safetensors(
    config: &Value,
    store: &SafetensorsWeightStore,
    expert_cache: bool,
) -> CheckpointValidation {
    let args = match qwen3_next::model_args_from_config_value(config) {
        Ok(args) => args,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    validate_safetensors(
        &args,
        None,
        store,
        expert_cache,
        SafetensorsVariant::Qwen3Next,
    )
}

pub(crate) fn validate_qwen35_safetensors(
    config: &Value,
    store: &SafetensorsWeightStore,
    expert_cache: bool,
) -> CheckpointValidation {
    let (args, _, _, vision) = match qwen35::model_config_from_value(config) {
        Ok(config) => config,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    validate_safetensors(
        &args,
        vision.as_ref(),
        store,
        expert_cache,
        SafetensorsVariant::Qwen35,
    )
}

fn validate_safetensors(
    args: &qwen35::ModelArgs,
    vision: Option<&qwen35::VisionConfig>,
    store: &SafetensorsWeightStore,
    expert_cache: bool,
    variant: SafetensorsVariant,
) -> CheckpointValidation {
    if !args.is_moe() && expert_cache {
        return invalid_geometry(format!(
            "sparse expert caching requires a {} MoE checkpoint",
            variant.label()
        ));
    }
    let mut issues = layout_issues(args, store, variant);
    let plan = match safetensors_plan(args, vision, variant) {
        Ok(plan) => plan,
        Err(error) => return invalid_geometry(error),
    };
    append_issues(
        validation::validate_safetensors_plan(store, &plan),
        &mut issues,
    );
    CheckpointValidation::from_issues(issues)
}

pub(crate) fn safetensors_plan(
    args: &qwen35::ModelArgs,
    vision: Option<&qwen35::VisionConfig>,
    variant: SafetensorsVariant,
) -> Result<SafetensorsCheckpointPlan, String> {
    let hidden = dimension(args.hidden_size, "hidden_size")?;
    let vocab = dimension(args.vocab_size, "vocab_size")?;
    let mut common = Vec::new();
    let mut groups = Vec::new();
    add_tensor(
        args,
        &mut common,
        "model.embed_tokens.weight",
        vec![vocab, hidden],
        TensorOperation::Matrix,
    )?;
    add_tensor(
        args,
        &mut common,
        "model.norm.weight",
        vec![hidden],
        TensorOperation::Vector,
    )?;
    if !args.tie_word_embeddings {
        add_tensor(
            args,
            &mut common,
            "lm_head.weight",
            vec![vocab, hidden],
            TensorOperation::Matrix,
        )?;
    }
    for layer in 0..dimension(args.num_hidden_layers, "num_hidden_layers")? {
        let policy = args.layer_schedule.get(layer).copied().ok_or_else(|| {
            format!("Qwen hybrid layer schedule is missing decoder layer {layer}")
        })?;
        add_block(
            args,
            &mut common,
            &mut groups,
            format!("model.layers.{layer}"),
            policy,
            variant,
        )?;
    }
    let mtp_layers = usize::try_from(args.mtp_num_hidden_layers)
        .map_err(|_| "Qwen hybrid MTP layer count must be non-negative".to_string())?;
    if mtp_layers > 0 {
        for (name, shape, operation) in [
            (
                "mtp.pre_fc_norm_hidden.weight",
                vec![hidden],
                TensorOperation::Vector,
            ),
            (
                "mtp.pre_fc_norm_embedding.weight",
                vec![hidden],
                TensorOperation::Vector,
            ),
            (
                "mtp.fc.weight",
                vec![hidden, checked_mul(2, hidden, "MTP input")?],
                TensorOperation::Matrix,
            ),
            ("mtp.norm.weight", vec![hidden], TensorOperation::Vector),
        ] {
            add_tensor(args, &mut common, name, shape, operation)?;
        }
        for layer in 0..mtp_layers {
            add_block(
                args,
                &mut common,
                &mut groups,
                format!("mtp.layers.{layer}"),
                qwen35::LayerPolicy::SelfAttention(AttentionPolicy::Full),
                variant,
            )?;
        }
    }
    if let Some(vision) = vision {
        add_vision(&mut common, vision, hidden)?;
    }
    let mut policy = CatalogPolicy::strict();
    policy.allowed_prefixes = vec![
        "visual.".into(),
        "vision_tower.".into(),
        "model.visual.".into(),
        "model.vision_tower.".into(),
    ];
    if !variant.accepts_fused_linear_projection() {
        allow_qwen35_fused_names(args, &mut policy)?;
    }
    SafetensorsCheckpointPlan::new(
        format!("{} SafeTensors", variant.label()),
        common,
        groups,
        policy,
    )
    .map_err(|error| error.to_string())
}

fn add_block(
    args: &qwen35::ModelArgs,
    common: &mut Vec<SafetensorsTensorConstraint>,
    groups: &mut Vec<AlternativeLayoutGroup<SafetensorsTensorConstraint>>,
    prefix: String,
    policy: qwen35::LayerPolicy,
    variant: SafetensorsVariant,
) -> Result<(), String> {
    let hidden = dimension(args.hidden_size, "hidden_size")?;
    for name in ["input_layernorm.weight", "post_attention_layernorm.weight"] {
        add_tensor(
            args,
            common,
            format!("{prefix}.{name}"),
            vec![hidden],
            TensorOperation::Vector,
        )?;
    }
    match policy {
        qwen35::LayerPolicy::SelfAttention(AttentionPolicy::Full) => {
            let head = dimension(args.head_dim, "head_dim")?;
            let query = checked_mul(
                dimension(args.num_attention_heads, "num_attention_heads")?,
                head,
                "query width",
            )?;
            let kv = checked_mul(
                dimension(args.num_key_value_heads, "num_key_value_heads")?,
                head,
                "key/value width",
            )?;
            for (name, shape, operation) in [
                (
                    "self_attn.q_proj.weight",
                    vec![checked_mul(2, query, "gated query width")?, hidden],
                    TensorOperation::Matrix,
                ),
                (
                    "self_attn.k_proj.weight",
                    vec![kv, hidden],
                    TensorOperation::Matrix,
                ),
                (
                    "self_attn.v_proj.weight",
                    vec![kv, hidden],
                    TensorOperation::Matrix,
                ),
                (
                    "self_attn.o_proj.weight",
                    vec![hidden, query],
                    TensorOperation::Matrix,
                ),
                (
                    "self_attn.q_norm.weight",
                    vec![head],
                    TensorOperation::Vector,
                ),
                (
                    "self_attn.k_norm.weight",
                    vec![head],
                    TensorOperation::Vector,
                ),
            ] {
                add_tensor(args, common, format!("{prefix}.{name}"), shape, operation)?;
            }
            if args.attention_bias {
                for (name, size) in [
                    ("q_proj", checked_mul(2, query, "query bias")?),
                    ("k_proj", kv),
                    ("v_proj", kv),
                    ("o_proj", hidden),
                ] {
                    add_tensor(
                        args,
                        common,
                        format!("{prefix}.self_attn.{name}.bias"),
                        vec![size],
                        TensorOperation::Vector,
                    )?;
                }
            }
        }
        qwen35::LayerPolicy::LinearAttention => {
            add_linear_attention(args, common, groups, &prefix, variant)?;
        }
        qwen35::LayerPolicy::SelfAttention(AttentionPolicy::Sliding { .. }) => {
            return Err("Qwen hybrid does not support sliding self-attention".into());
        }
    }
    if args.is_moe() {
        let experts = dimension(args.num_experts, "num_experts")?;
        let intermediate = dimension(args.moe_intermediate_size, "moe_intermediate_size")?;
        let shared = dimension(
            args.shared_expert_intermediate_size,
            "shared_expert_intermediate_size",
        )?;
        for (name, shape) in [
            ("mlp.gate.weight", vec![experts, hidden]),
            ("mlp.shared_expert.gate_proj.weight", vec![shared, hidden]),
            ("mlp.shared_expert.up_proj.weight", vec![shared, hidden]),
            ("mlp.shared_expert.down_proj.weight", vec![hidden, shared]),
            ("mlp.shared_expert_gate.weight", vec![1, hidden]),
        ] {
            add_tensor(
                args,
                common,
                format!("{prefix}.{name}"),
                shape,
                TensorOperation::Matrix,
            )?;
        }
        add_experts(
            args,
            groups,
            &format!("{prefix}.mlp.experts"),
            experts,
            hidden,
            intermediate,
        )?;
    } else {
        let intermediate = dimension(args.intermediate_size, "intermediate_size")?;
        for (name, shape) in [
            ("gate_proj", vec![intermediate, hidden]),
            ("up_proj", vec![intermediate, hidden]),
            ("down_proj", vec![hidden, intermediate]),
        ] {
            add_tensor(
                args,
                common,
                format!("{prefix}.mlp.{name}.weight"),
                shape,
                TensorOperation::Matrix,
            )?;
        }
    }
    Ok(())
}

fn add_linear_attention(
    args: &qwen35::ModelArgs,
    common: &mut Vec<SafetensorsTensorConstraint>,
    groups: &mut Vec<AlternativeLayoutGroup<SafetensorsTensorConstraint>>,
    block: &str,
    variant: SafetensorsVariant,
) -> Result<(), String> {
    let prefix = format!("{block}.linear_attn");
    let hidden = dimension(args.hidden_size, "hidden_size")?;
    let key = checked_mul(
        dimension(args.linear_num_key_heads, "linear_num_key_heads")?,
        dimension(args.linear_key_head_dim, "linear_key_head_dim")?,
        "linear key width",
    )?;
    let value = checked_mul(
        dimension(args.linear_num_value_heads, "linear_num_value_heads")?,
        dimension(args.linear_value_head_dim, "linear_value_head_dim")?,
        "linear value width",
    )?;
    let heads = dimension(args.linear_num_value_heads, "linear_num_value_heads")?;
    let mut variants = Vec::new();
    if variant.accepts_fused_linear_projection() {
        variants.push(layout_variant(
            args,
            "fused",
            [
                (
                    format!("{prefix}.in_proj_qkvz.weight"),
                    vec![
                        checked_add(
                            checked_mul(2, key, "fused key width")?,
                            checked_mul(2, value, "fused value width")?,
                            "fused QKVZ width",
                        )?,
                        hidden,
                    ],
                ),
                (
                    format!("{prefix}.in_proj_ba.weight"),
                    vec![checked_mul(2, heads, "fused BA width")?, hidden],
                ),
            ],
        )?);
    }
    variants.push(layout_variant(
        args,
        "split",
        [
            (
                format!("{prefix}.in_proj_qkv.weight"),
                vec![
                    checked_add(
                        checked_mul(2, key, "split query/key width")?,
                        value,
                        "split QKV width",
                    )?,
                    hidden,
                ],
            ),
            (format!("{prefix}.in_proj_z.weight"), vec![value, hidden]),
            (format!("{prefix}.in_proj_b.weight"), vec![heads, hidden]),
            (format!("{prefix}.in_proj_a.weight"), vec![heads, hidden]),
        ],
    )?);
    groups.push(AlternativeLayoutGroup {
        id: format!("{block} linear-attention inputs"),
        required: true,
        variants,
    });
    for (name, shape, operation) in [
        (
            "conv1d.weight",
            vec![
                checked_add(
                    checked_mul(2, key, "linear convolution key width")?,
                    value,
                    "linear convolution width",
                )?,
                1,
                dimension(args.linear_conv_kernel_dim, "linear_conv_kernel_dim")?,
            ],
            TensorOperation::Dense,
        ),
        ("dt_bias", vec![heads], TensorOperation::Vector),
        ("A_log", vec![heads], TensorOperation::Vector),
        (
            "norm.weight",
            vec![dimension(
                args.linear_value_head_dim,
                "linear_value_head_dim",
            )?],
            TensorOperation::Vector,
        ),
        (
            "out_proj.weight",
            vec![hidden, value],
            TensorOperation::Matrix,
        ),
    ] {
        add_tensor(args, common, format!("{prefix}.{name}"), shape, operation)?;
    }
    Ok(())
}

fn add_experts(
    args: &qwen35::ModelArgs,
    groups: &mut Vec<AlternativeLayoutGroup<SafetensorsTensorConstraint>>,
    prefix: &str,
    experts: usize,
    hidden: usize,
    intermediate: usize,
) -> Result<(), String> {
    let packed = layout_variant(
        args,
        "packed",
        [
            (
                format!("{prefix}.gate_up_proj"),
                vec![
                    experts,
                    checked_mul(2, intermediate, "packed expert width")?,
                    hidden,
                ],
            ),
            (
                format!("{prefix}.down_proj"),
                vec![experts, hidden, intermediate],
            ),
        ],
    )?;
    let mut split = Vec::with_capacity(checked_mul(experts, 3, "expert tensor count")?);
    for expert in 0..experts {
        split.extend([
            (
                format!("{prefix}.{expert}.gate_proj.weight"),
                vec![intermediate, hidden],
            ),
            (
                format!("{prefix}.{expert}.up_proj.weight"),
                vec![intermediate, hidden],
            ),
            (
                format!("{prefix}.{expert}.down_proj.weight"),
                vec![hidden, intermediate],
            ),
        ]);
    }
    groups.push(AlternativeLayoutGroup {
        id: format!("{prefix} storage"),
        required: true,
        variants: vec![packed, layout_variant(args, "split", split)?],
    });
    Ok(())
}

fn layout_variant(
    args: &qwen35::ModelArgs,
    id: &str,
    tensors: impl IntoIterator<Item = (String, Vec<usize>)>,
) -> Result<LayoutVariant<SafetensorsTensorConstraint>, String> {
    let tensors = tensors.into_iter().collect::<Vec<_>>();
    let discriminator_keys = tensors.iter().map(|(name, _)| name.clone()).collect();
    let mut constraints = Vec::new();
    for (name, shape) in tensors {
        constraints.extend(tensor_constraints(
            args,
            &name,
            shape,
            TensorOperation::Matrix,
        )?);
    }
    Ok(LayoutVariant {
        id: id.into(),
        tensors: constraints,
        discriminator_keys,
    })
}

fn add_tensor(
    args: &qwen35::ModelArgs,
    common: &mut Vec<SafetensorsTensorConstraint>,
    name: impl AsRef<str>,
    shape: Vec<usize>,
    operation: TensorOperation,
) -> Result<(), String> {
    common.extend(tensor_constraints(args, name.as_ref(), shape, operation)?);
    Ok(())
}

fn tensor_constraints(
    args: &qwen35::ModelArgs,
    canonical: &str,
    shape: Vec<usize>,
    operation: TensorOperation,
) -> Result<Vec<SafetensorsTensorConstraint>, String> {
    let aliases = qwen_aliases(canonical);
    match tensor_format(args, canonical, operation) {
        MatrixFormat::Dense => Ok(vec![physical_constraint(
            &aliases,
            shape,
            StoredDtypeConstraint::Floating,
        )]),
        MatrixFormat::Fp8 => {
            if shape.len() < 2 {
                return Err(format!(
                    "FP8 tensor {canonical:?} must have rank at least two"
                ));
            }
            let weight = physical_constraint(
                &aliases,
                shape.clone(),
                StoredDtypeConstraint::OneOf(vec![StoredDtype::F8E4M3, StoredDtype::U8]),
            );
            let mut scale_shape = shape;
            let rank = scale_shape.len();
            scale_shape[rank - 2] = scale_shape[rank - 2].div_ceil(128);
            scale_shape[rank - 1] = scale_shape[rank - 1].div_ceil(128);
            let scales = aliases
                .iter()
                .map(|name| fp8_scale_name(name))
                .collect::<Vec<_>>();
            Ok(vec![
                weight,
                physical_constraint(
                    &scales,
                    scale_shape,
                    StoredDtypeConstraint::OneOf(vec![
                        StoredDtype::F16,
                        StoredDtype::BF16,
                        StoredDtype::F32,
                        StoredDtype::U8,
                    ]),
                )
                .companion(),
            ])
        }
        MatrixFormat::Affine(quantization) => affine_constraints(&aliases, shape, quantization),
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

#[derive(Clone, Copy)]
enum MatrixFormat {
    Dense,
    Affine(WeightQuantization),
    Fp8,
}

fn tensor_format(args: &qwen35::ModelArgs, name: &str, operation: TensorOperation) -> MatrixFormat {
    if operation != TensorOperation::Matrix {
        return MatrixFormat::Dense;
    }
    let always_dense = name == "model.embed_tokens.weight"
        || name == "lm_head.weight"
        || name.ends_with(".mlp.gate.weight")
        || name.ends_with(".mlp.shared_expert_gate.weight")
        || (args.uses_fp8() && name.ends_with(".linear_attn.in_proj_ba.weight"));
    if always_dense && args.quantization_config.is_some() {
        MatrixFormat::Dense
    } else if name.ends_with(".mlp.gate.weight") || name.ends_with(".mlp.shared_expert_gate.weight")
    {
        MatrixFormat::Dense
    } else if let Some(quantization) = args.quantization {
        MatrixFormat::Affine(quantization)
    } else if args.uses_fp8() {
        MatrixFormat::Fp8
    } else {
        MatrixFormat::Dense
    }
}

fn qwen_aliases(canonical: &str) -> Vec<String> {
    let Some(rest) = canonical.strip_prefix("model.") else {
        return vec![canonical.into()];
    };
    vec![
        canonical.into(),
        format!("model.language_model.{rest}"),
        format!("language_model.{rest}"),
        format!("model.model.{rest}"),
    ]
}

fn add_vision(
    common: &mut Vec<SafetensorsTensorConstraint>,
    config: &qwen35::VisionConfig,
    text_hidden: usize,
) -> Result<(), String> {
    let hidden = dimension(config.hidden_size, "vision hidden_size")?;
    let intermediate = dimension(config.intermediate_size, "vision intermediate_size")?;
    let merge = dimension(config.spatial_merge_size, "vision spatial_merge_size")?;
    let merger = checked_mul(
        hidden,
        checked_mul(merge, merge, "vision merge")?,
        "vision merger",
    )?;
    let mut tensors = vec![
        (
            "pos_embed.weight".into(),
            vec![
                dimension(config.num_position_embeddings, "vision positions")?,
                hidden,
            ],
        ),
        (
            "patch_embed.proj.weight".into(),
            vec![
                hidden,
                dimension(config.in_channels, "vision channels")?,
                dimension(config.temporal_patch_size, "vision temporal patch")?,
                dimension(config.patch_size, "vision patch")?,
                dimension(config.patch_size, "vision patch")?,
            ],
        ),
        ("patch_embed.proj.bias".into(), vec![hidden]),
    ];
    for layer in 0..config.layer_count() {
        let root = format!("blocks.{layer}");
        tensors.extend([
            (format!("{root}.norm1.weight"), vec![hidden]),
            (format!("{root}.norm1.bias"), vec![hidden]),
            (
                format!("{root}.attn.qkv.weight"),
                vec![checked_mul(3, hidden, "vision QKV width")?, hidden],
            ),
            (
                format!("{root}.attn.qkv.bias"),
                vec![checked_mul(3, hidden, "vision QKV bias width")?],
            ),
            (format!("{root}.attn.proj.weight"), vec![hidden, hidden]),
            (format!("{root}.attn.proj.bias"), vec![hidden]),
            (format!("{root}.norm2.weight"), vec![hidden]),
            (format!("{root}.norm2.bias"), vec![hidden]),
            (
                format!("{root}.mlp.linear_fc1.weight"),
                vec![intermediate, hidden],
            ),
            (format!("{root}.mlp.linear_fc1.bias"), vec![intermediate]),
            (
                format!("{root}.mlp.linear_fc2.weight"),
                vec![hidden, intermediate],
            ),
            (format!("{root}.mlp.linear_fc2.bias"), vec![hidden]),
        ]);
    }
    tensors.extend([
        ("merger.norm.weight".into(), vec![hidden]),
        ("merger.norm.bias".into(), vec![hidden]),
        ("merger.linear_fc1.weight".into(), vec![merger, merger]),
        ("merger.linear_fc1.bias".into(), vec![merger]),
        ("merger.linear_fc2.weight".into(), vec![text_hidden, merger]),
        ("merger.linear_fc2.bias".into(), vec![text_hidden]),
    ]);
    for index in 0..config.deepstack_layer_count() {
        let root = format!("deepstack_merger_list.{index}");
        tensors.extend([
            (format!("{root}.norm.weight"), vec![merger]),
            (format!("{root}.norm.bias"), vec![merger]),
            (format!("{root}.linear_fc1.weight"), vec![merger, merger]),
            (format!("{root}.linear_fc1.bias"), vec![merger]),
            (
                format!("{root}.linear_fc2.weight"),
                vec![text_hidden, merger],
            ),
            (format!("{root}.linear_fc2.bias"), vec![text_hidden]),
        ]);
    }
    for (rest, shape) in tensors {
        let mut aliases = vision_aliases(&rest);
        for (canonical, released) in [("linear_fc1", "mlp.0"), ("linear_fc2", "mlp.2")] {
            if rest.contains(canonical) {
                aliases.extend(vision_aliases(&rest.replace(canonical, released)));
            }
        }
        aliases.sort();
        aliases.dedup();
        common.push(physical_constraint(
            &aliases,
            shape,
            StoredDtypeConstraint::Floating,
        ));
    }
    Ok(())
}

fn vision_aliases(rest: &str) -> Vec<String> {
    vec![
        format!("visual.{rest}"),
        format!("model.visual.{rest}"),
        format!("vision_tower.{rest}"),
        format!("model.vision_tower.{rest}"),
    ]
}

fn layout_issues(
    args: &qwen35::ModelArgs,
    store: &SafetensorsWeightStore,
    variant: SafetensorsVariant,
) -> Vec<CheckpointIssue> {
    let keys = store.keys().into_iter().collect::<BTreeSet<_>>();
    let mut issues = Vec::new();
    for layer in 0..args.num_hidden_layers.max(0) as usize {
        let prefix = format!("model.layers.{layer}");
        if args.layer_schedule.get(layer) == Some(&qwen35::LayerPolicy::LinearAttention) {
            check_linear_layout(&keys, &prefix, variant, &mut issues);
        }
        if args.is_moe() {
            check_expert_layout(
                args,
                &keys,
                &format!("{prefix}.mlp.experts"),
                variant,
                &mut issues,
            );
        }
    }
    if args.is_moe() {
        for layer in 0..args.mtp_num_hidden_layers.max(0) as usize {
            check_expert_layout(
                args,
                &keys,
                &format!("mtp.layers.{layer}.mlp.experts"),
                variant,
                &mut issues,
            );
        }
    }
    issues
}

fn check_linear_layout(
    keys: &BTreeSet<String>,
    block: &str,
    variant: SafetensorsVariant,
    issues: &mut Vec<CheckpointIssue>,
) {
    let fused = [
        format!("{block}.linear_attn.in_proj_qkvz.weight"),
        format!("{block}.linear_attn.in_proj_ba.weight"),
    ];
    let split = [
        format!("{block}.linear_attn.in_proj_qkv.weight"),
        format!("{block}.linear_attn.in_proj_z.weight"),
        format!("{block}.linear_attn.in_proj_b.weight"),
        format!("{block}.linear_attn.in_proj_a.weight"),
    ];
    let has_fused = fused.iter().any(|name| tensor_present(keys, name));
    let has_split = split.iter().any(|name| tensor_present(keys, name));
    if has_fused && has_split {
        let mut issue = conflict(format!(
            "{} linear-attention layer {block:?} mixes fused and split input projections",
            variant.label()
        ));
        issue.tensor_name = split
            .iter()
            .flat_map(|name| qwen_aliases(name))
            .find(|name| keys.contains(name));
        issues.push(issue);
    }
    if has_fused && !variant.accepts_fused_linear_projection() {
        let mut issue = conflict(format!(
            "{} SafeTensors requires split linear-attention input projections",
            variant.label()
        ));
        issue.tensor_name = fused
            .iter()
            .flat_map(|name| qwen_aliases(name))
            .find(|name| keys.contains(name));
        issues.push(issue);
    }
}

fn check_expert_layout(
    args: &qwen35::ModelArgs,
    keys: &BTreeSet<String>,
    prefix: &str,
    variant: SafetensorsVariant,
    issues: &mut Vec<CheckpointIssue>,
) {
    let packed = [
        format!("{prefix}.gate_up_proj"),
        format!("{prefix}.down_proj"),
    ];
    let split = (0..args.num_experts.max(0) as usize)
        .flat_map(|expert| {
            [
                format!("{prefix}.{expert}.gate_proj.weight"),
                format!("{prefix}.{expert}.up_proj.weight"),
                format!("{prefix}.{expert}.down_proj.weight"),
            ]
        })
        .collect::<Vec<_>>();
    let has_packed = packed.iter().any(|name| tensor_present(keys, name));
    let has_split = split.iter().any(|name| tensor_present(keys, name));
    if has_packed && has_split {
        issues.push(conflict(format!(
            "{} expert catalog {prefix:?} mixes packed and split tensors",
            variant.label()
        )));
    }
    if args.quantization.is_some() && has_split {
        let mut issue = conflict(format!(
            "{} checkpoint-native affine loader requires packed expert banks",
            variant.label()
        ));
        issue.metadata_key = Some("quantization".into());
        issues.push(issue);
    }
}

fn allow_qwen35_fused_names(
    args: &qwen35::ModelArgs,
    policy: &mut CatalogPolicy,
) -> Result<(), String> {
    let hidden = dimension(args.hidden_size, "hidden_size")?;
    let key = checked_mul(
        dimension(args.linear_num_key_heads, "linear_num_key_heads")?,
        dimension(args.linear_key_head_dim, "linear_key_head_dim")?,
        "linear key width",
    )?;
    let value = checked_mul(
        dimension(args.linear_num_value_heads, "linear_num_value_heads")?,
        dimension(args.linear_value_head_dim, "linear_value_head_dim")?,
        "linear value width",
    )?;
    let heads = dimension(args.linear_num_value_heads, "linear_num_value_heads")?;
    for layer in 0..dimension(args.num_hidden_layers, "num_hidden_layers")? {
        if args.layer_schedule.get(layer) != Some(&qwen35::LayerPolicy::LinearAttention) {
            continue;
        }
        for (name, shape) in [
            (
                format!("model.layers.{layer}.linear_attn.in_proj_qkvz.weight"),
                vec![
                    checked_add(
                        checked_mul(2, key, "fused key width")?,
                        checked_mul(2, value, "fused value width")?,
                        "fused QKVZ width",
                    )?,
                    hidden,
                ],
            ),
            (
                format!("model.layers.{layer}.linear_attn.in_proj_ba.weight"),
                vec![checked_mul(2, heads, "fused BA width")?, hidden],
            ),
        ] {
            for constraint in tensor_constraints(args, &name, shape, TensorOperation::Matrix)? {
                policy.explicitly_allowed_keys.insert(constraint.key);
                policy.explicitly_allowed_keys.extend(constraint.aliases);
            }
        }
    }
    Ok(())
}

fn tensor_present(keys: &BTreeSet<String>, canonical: &str) -> bool {
    qwen_aliases(canonical)
        .iter()
        .any(|name| keys.contains(name))
}

fn quantized_weight_alias(name: &str) -> Option<String> {
    name.strip_suffix(".weight")
        .map(|prefix| format!("{prefix}.inner.weight"))
}

fn outer_prefix(name: &str) -> &str {
    name.strip_suffix(".weight").unwrap_or(name)
}

fn fp8_scale_name(name: &str) -> String {
    name.strip_suffix(".weight").map_or_else(
        || format!("{name}_scale_inv"),
        |prefix| format!("{prefix}.weight_scale_inv"),
    )
}

fn dimension(value: i32, name: &str) -> Result<usize, String> {
    usize::try_from(value)
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| format!("Qwen hybrid {name} must be positive, got {value}"))
}

fn checked_mul(left: usize, right: usize, name: &str) -> Result<usize, String> {
    left.checked_mul(right)
        .ok_or_else(|| format!("Qwen hybrid {name} geometry overflows"))
}

fn checked_add(left: usize, right: usize, name: &str) -> Result<usize, String> {
    left.checked_add(right)
        .ok_or_else(|| format!("Qwen hybrid {name} geometry overflows"))
}

fn conflict(detail: String) -> CheckpointIssue {
    CheckpointIssue {
        kind: CheckpointIssueKind::ConflictingLayout,
        detail,
        tensor_name: None,
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

fn append_issues(validation: CheckpointValidation, issues: &mut Vec<CheckpointIssue>) {
    match validation {
        CheckpointValidation::Exact => {}
        CheckpointValidation::Invalid(mut nested) => issues.append(&mut nested),
        CheckpointValidation::Unverified(issue) => issues.push(issue),
    }
}
