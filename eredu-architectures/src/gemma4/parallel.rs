//! Semantic parameter placement for Gemma 4 components.

use eredu_nn::{AttentionStateSource, AttentionValueSource};
use eredu_runtime::{
    MemberSharding, ParallelPlanError, ParameterGroupSpec, ParameterMemberSpec, ParameterRole,
};

use super::{FeedForwardPolicy, ModelArgs};

/// Derives one Gemma 4 decoder block's rank-local construction geometry.
pub fn local_block_args(
    args: &ModelArgs,
    layer: usize,
    layout: &eredu_runtime::LocalModelLayout,
) -> Result<ModelArgs, ParallelPlanError> {
    let policy = args
        .layer_policy(layer)
        .ok_or_else(|| invalid(format!("Gemma 4 layer {layer} is out of range")))?;
    let root = format!("model.language_model.layers.{layer}");
    let tensor_at = |layer: usize, suffix: &str| {
        let root = format!("model.language_model.layers.{layer}");
        layout
            .tensor(&format!("{root}.{suffix}.weight"))
            .or_else(|| layout.tensor(&format!("{root}.{suffix}.inner.weight")))
            .or_else(|| layout.tensor(&format!("{root}.{suffix}")))
    };
    let tensor = |suffix: &str| {
        tensor_at(layer, suffix).ok_or_else(|| {
            ParallelPlanError::InvalidTensor(format!("missing local layout for {root}.{suffix}"))
        })
    };
    let query_width = local_axis(tensor("self_attn.q_proj")?, 0, "query width")?;
    let head_dim = i32::try_from(policy.head_dim.get()).map_err(|_| {
        ParallelPlanError::InvalidTensor("Gemma 4 head dimension exceeds i32".into())
    })?;
    if query_width % head_dim != 0 {
        return Err(ParallelPlanError::InvalidTensor(format!(
            "local Gemma 4 query width {query_width} splits head dimension {head_dim}"
        )));
    }
    let kv_tensor = if policy.key_value.owns_state() {
        Some(tensor("self_attn.k_proj")?)
    } else {
        (0..layer).rev().find_map(|owner| {
            args.layer_policy(owner)
                .filter(|candidate| {
                    candidate.attention == policy.attention && candidate.key_value.publishes_state()
                })
                .and_then(|_| tensor_at(owner, "self_attn.k_proj"))
        })
    }
    .ok_or_else(|| {
        ParallelPlanError::InvalidTensor(format!(
            "Gemma 4 shared-KV layer {layer} has no rank-local publisher"
        ))
    })?;
    let kv_width = local_axis(kv_tensor, 0, "key/value width")?;
    if kv_width % head_dim != 0 {
        return Err(ParallelPlanError::InvalidTensor(format!(
            "local Gemma 4 key/value width {kv_width} splits head dimension {head_dim}"
        )));
    }
    let dense_width = local_axis(tensor("mlp.gate_proj")?, 0, "dense intermediate width")?;
    let mut local = args.clone();
    local.num_attention_heads = query_width / head_dim;
    let mut layers = args.layer_schedule.iter().copied().collect::<Vec<_>>();
    layers[layer].num_key_value_heads =
        std::num::NonZeroU32::new(u32::try_from(kv_width / head_dim).map_err(|_| {
            ParallelPlanError::InvalidTensor("local Gemma 4 KV heads exceed u32".into())
        })?)
        .ok_or_else(|| {
            ParallelPlanError::InvalidTensor("local Gemma 4 KV heads are zero".into())
        })?;
    layers[layer].intermediate_size =
        std::num::NonZeroU32::new(u32::try_from(dense_width).map_err(|_| {
            ParallelPlanError::InvalidTensor("local Gemma 4 dense width exceeds u32".into())
        })?)
        .ok_or_else(|| {
            ParallelPlanError::InvalidTensor("local Gemma 4 dense width is zero".into())
        })?;
    local.layer_schedule = eredu_core::LayerSchedule::new(layers.len(), layers)
        .map_err(|error| invalid(error.to_string()))?;
    if policy.feed_forward == FeedForwardPolicy::DenseWithSparseMoe {
        let fused = local_axis(
            tensor("experts.switch_glu.gate_up_proj")?,
            1,
            "expert gate/up width",
        )?;
        if fused % 2 != 0 {
            return Err(ParallelPlanError::InvalidTensor(format!(
                "local Gemma 4 expert gate/up width {fused} is not even"
            )));
        }
        local.moe_intermediate_size = Some(fused / 2);
    }
    if args.hidden_size_per_layer_input > 0 {
        local.hidden_size_per_layer_input =
            local_axis(tensor("per_layer_input_gate")?, 0, "per-layer media width")?;
    }
    Ok(local)
}

fn local_axis(
    tensor: &eredu_runtime::LocalTensorLayout,
    axis: usize,
    label: &str,
) -> Result<i32, ParallelPlanError> {
    let value = *tensor.local_shape().get(axis).ok_or_else(|| {
        ParallelPlanError::InvalidTensor(format!("local Gemma 4 {label} tensor has no axis {axis}"))
    })?;
    i32::try_from(value)
        .map_err(|_| ParallelPlanError::InvalidTensor(format!("local Gemma 4 {label} exceeds i32")))
}

/// Declares pinned embeddings, final normalization, and vocabulary output.
pub fn static_parameter_groups(
    args: &ModelArgs,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    let hidden = dim(args.hidden_size)?;
    let vocab = dim(args.vocab_size)?;
    let mut groups = vec![group(
        "model.language_model.embed_tokens",
        ParameterRole::Vocabulary,
        [(
            "model.language_model.embed_tokens.weight".into(),
            vec![vocab, hidden],
            MemberSharding::Balanced { axis: 0 },
        )],
    )?];
    if args.hidden_size_per_layer_input > 0 {
        let per_layer = dim(args.hidden_size_per_layer_input)?;
        let combined = per_layer
            .checked_mul(args.num_hidden_layers())
            .ok_or_else(|| invalid("Gemma per-layer embedding width overflow"))?;
        groups.push(ParameterGroupSpec::partitioned(
            "model.language_model.per_layer_embedding_channels",
            ParameterRole::Channels,
            combined,
            [
                member(
                    "model.language_model.embed_tokens_per_layer.weight",
                    vec![
                        dim(args.vocab_size_per_layer_input.unwrap_or(args.vocab_size))?,
                        combined,
                    ],
                    MemberSharding::Partitioned { axis: 1 },
                ),
                member(
                    "model.language_model.per_layer_model_projection.weight",
                    vec![combined, hidden],
                    MemberSharding::Partitioned { axis: 0 },
                ),
            ],
        )?);
        groups.push(replicated(
            "model.language_model.per_layer_projection_norm",
            [(
                "model.language_model.per_layer_projection_norm.weight".into(),
                vec![per_layer],
            )],
        )?);
    }
    groups.push(replicated(
        "model.language_model.norm",
        [("model.language_model.norm.weight".into(), vec![hidden])],
    )?);
    if !args.tie_word_embeddings {
        groups.push(group(
            "lm_head",
            ParameterRole::Vocabulary,
            [(
                "lm_head.weight".into(),
                vec![vocab, hidden],
                MemberSharding::Balanced { axis: 0 },
            )],
        )?);
    }
    Ok(groups)
}

/// Declares one decoder layer's head, MLP, expert, media, and replicated groups.
pub fn layer_parameter_groups(
    args: &ModelArgs,
    layer: usize,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    let policy = args
        .layer_policy(layer)
        .ok_or_else(|| invalid(format!("Gemma 4 layer {layer} is out of range")))?;
    let root = format!("model.language_model.layers.{layer}");
    let attention = format!("{root}.self_attn");
    let hidden = dim(args.hidden_size)?;
    let head_dim = dim(policy.head_dim.get() as i32)?;
    let query_heads = dim(args.num_attention_heads)?;
    let kv_heads = dim(policy.num_key_value_heads.get() as i32)?;
    let query_width = query_heads
        .checked_mul(head_dim)
        .ok_or_else(|| invalid("Gemma query width overflow"))?;
    let kv_width = kv_heads
        .checked_mul(head_dim)
        .ok_or_else(|| invalid("Gemma KV width overflow"))?;
    let mut query_members = vec![
        member(
            format!("{attention}.q_proj.weight"),
            vec![query_width, hidden],
            MemberSharding::Partitioned { axis: 0 },
        ),
        member(
            format!("{attention}.o_proj.weight"),
            vec![hidden, query_width],
            MemberSharding::Partitioned { axis: 1 },
        ),
    ];
    if args.attention_bias {
        query_members.extend([
            member(
                format!("{attention}.q_proj.bias"),
                vec![query_width],
                MemberSharding::Partitioned { axis: 0 },
            ),
            member(
                format!("{attention}.o_proj.bias"),
                vec![hidden],
                MemberSharding::Replicated,
            ),
        ]);
    }
    let mut groups = vec![ParameterGroupSpec::partitioned(
        format!("{attention}.query_heads"),
        ParameterRole::AttentionHeads,
        query_heads,
        query_members,
    )?];
    if policy.key_value != AttentionStateSource::Shared {
        let mut kv_members = vec![member(
            format!("{attention}.k_proj.weight"),
            vec![kv_width, hidden],
            MemberSharding::Partitioned { axis: 0 },
        )];
        if policy.key_value.value() == Some(AttentionValueSource::Projected) {
            kv_members.push(member(
                format!("{attention}.v_proj.weight"),
                vec![kv_width, hidden],
                MemberSharding::Partitioned { axis: 0 },
            ));
        }
        if args.attention_bias {
            kv_members.push(member(
                format!("{attention}.k_proj.bias"),
                vec![kv_width],
                MemberSharding::Partitioned { axis: 0 },
            ));
            if policy.key_value.value() == Some(AttentionValueSource::Projected) {
                kv_members.push(member(
                    format!("{attention}.v_proj.bias"),
                    vec![kv_width],
                    MemberSharding::Partitioned { axis: 0 },
                ));
            }
        }
        groups.push(ParameterGroupSpec::partitioned(
            format!("{attention}.key_value_heads"),
            ParameterRole::AttentionHeads,
            kv_heads,
            kv_members,
        )?);
    }
    groups.push(replicated(
        format!("{attention}.head_norms"),
        std::iter::once((format!("{attention}.q_norm.weight"), vec![head_dim])).chain(
            (policy.key_value != AttentionStateSource::Shared)
                .then(|| (format!("{attention}.k_norm.weight"), vec![head_dim])),
        ),
    )?);
    let intermediate = dim(policy.intermediate_size.get() as i32)?;
    groups.push(ParameterGroupSpec::partitioned(
        format!("{root}.mlp.intermediate"),
        ParameterRole::FeedForwardIntermediate,
        intermediate,
        [
            member(
                format!("{root}.mlp.gate_proj.weight"),
                vec![intermediate, hidden],
                MemberSharding::Partitioned { axis: 0 },
            ),
            member(
                format!("{root}.mlp.up_proj.weight"),
                vec![intermediate, hidden],
                MemberSharding::Partitioned { axis: 0 },
            ),
            member(
                format!("{root}.mlp.down_proj.weight"),
                vec![hidden, intermediate],
                MemberSharding::Partitioned { axis: 1 },
            ),
        ],
    )?);
    if policy.feed_forward == FeedForwardPolicy::DenseWithSparseMoe {
        let experts = dim(args
            .num_experts
            .ok_or_else(|| invalid("Gemma sparse layer has no expert count"))?)?;
        let expert_width = dim(args
            .moe_intermediate_size
            .ok_or_else(|| invalid("Gemma sparse layer has no expert width"))?)?;
        let expert_root = format!("{root}.experts.switch_glu");
        groups.push(ParameterGroupSpec::partitioned(
            format!("{expert_root}.intermediate"),
            ParameterRole::ExpertIntermediate,
            expert_width,
            [
                member(
                    format!("{expert_root}.gate_up_proj"),
                    vec![experts, 2 * expert_width, hidden],
                    MemberSharding::PartitionedSegments {
                        axis: 1,
                        segments: vec![0..expert_width, expert_width..2 * expert_width],
                    },
                ),
                member(
                    format!("{expert_root}.down_proj"),
                    vec![experts, hidden, expert_width],
                    MemberSharding::Partitioned { axis: 2 },
                ),
            ],
        )?);
        groups.push(replicated(
            format!("{root}.router"),
            [
                (format!("{root}.router.proj.weight"), vec![experts, hidden]),
                (format!("{root}.router.scale"), vec![hidden]),
                (format!("{root}.router.per_expert_scale"), vec![experts]),
            ],
        )?);
    }
    if args.hidden_size_per_layer_input > 0 {
        let media = dim(args.hidden_size_per_layer_input)?;
        groups.push(ParameterGroupSpec::partitioned(
            format!("{root}.media_channels"),
            ParameterRole::Channels,
            media,
            [
                member(
                    format!("{root}.per_layer_input_gate.weight"),
                    vec![media, hidden],
                    MemberSharding::Partitioned { axis: 0 },
                ),
                member(
                    format!("{root}.per_layer_projection.weight"),
                    vec![hidden, media],
                    MemberSharding::Partitioned { axis: 1 },
                ),
            ],
        )?);
    }
    let mut replicated_members = vec![
        (format!("{root}.input_layernorm.weight"), vec![hidden]),
        (
            format!("{root}.post_attention_layernorm.weight"),
            vec![hidden],
        ),
        (
            format!("{root}.pre_feedforward_layernorm.weight"),
            vec![hidden],
        ),
        (
            format!("{root}.post_feedforward_layernorm.weight"),
            vec![hidden],
        ),
        (format!("{root}.layer_scalar"), vec![1]),
    ];
    if policy.feed_forward == FeedForwardPolicy::DenseWithSparseMoe {
        replicated_members.extend([
            (
                format!("{root}.post_feedforward_layernorm_1.weight"),
                vec![hidden],
            ),
            (
                format!("{root}.pre_feedforward_layernorm_2.weight"),
                vec![hidden],
            ),
            (
                format!("{root}.post_feedforward_layernorm_2.weight"),
                vec![hidden],
            ),
        ]);
    }
    if args.hidden_size_per_layer_input > 0 {
        replicated_members.push((
            format!("{root}.post_per_layer_input_norm.weight"),
            vec![hidden],
        ));
    }
    groups.push(replicated(
        format!("{root}.replicated"),
        replicated_members,
    )?);
    Ok(groups)
}

fn group(
    name: impl Into<String>,
    role: ParameterRole,
    members: impl IntoIterator<Item = (String, Vec<usize>, MemberSharding)>,
) -> Result<ParameterGroupSpec, ParallelPlanError> {
    ParameterGroupSpec::new(
        name,
        role,
        members
            .into_iter()
            .map(|(name, shape, sharding)| member(name, shape, sharding)),
    )
}

fn replicated(
    name: impl Into<String>,
    members: impl IntoIterator<Item = (String, Vec<usize>)>,
) -> Result<ParameterGroupSpec, ParallelPlanError> {
    group(
        name,
        ParameterRole::Replicated,
        members
            .into_iter()
            .map(|(name, shape)| (name, shape, MemberSharding::Replicated)),
    )
}

fn member(
    name: impl Into<String>,
    shape: Vec<usize>,
    sharding: MemberSharding,
) -> ParameterMemberSpec {
    ParameterMemberSpec::new(name, shape, sharding)
}

fn dim(value: i32) -> Result<usize, ParallelPlanError> {
    usize::try_from(value)
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| invalid(format!("invalid Gemma dimension {value}")))
}

fn invalid(message: impl Into<String>) -> ParallelPlanError {
    ParallelPlanError::InvalidGroup(message.into())
}

#[cfg(test)]
mod tests {
    use eredu_runtime::{LocalModelLayout, LocalTensorLayout, TensorPlacement};

    use super::*;

    fn args() -> ModelArgs {
        ModelArgs::from_hf_json(
            br#"{
              "model_type":"gemma4","hidden_size":16,"num_hidden_layers":2,
              "intermediate_size":32,"num_attention_heads":4,"num_key_value_heads":2,
              "head_dim":4,"rms_norm_eps":0.000001,"vocab_size":64,
              "max_position_embeddings":128,"layer_types":["full_attention","full_attention"],
              "num_kv_shared_layers":1,"enable_moe_block":true,"num_experts":4,
              "top_k_experts":2,"moe_intermediate_size":8
            }"#,
        )
        .unwrap()
    }

    #[test]
    fn shared_layer_omits_kv_group_and_sparse_group_is_semantic() {
        let owner = layer_parameter_groups(&args(), 0).unwrap();
        assert!(owner
            .iter()
            .any(|group| group.logical_name().ends_with("key_value_heads")));
        assert!(owner
            .iter()
            .any(|group| group.role() == ParameterRole::ExpertIntermediate));
        let shared = layer_parameter_groups(&args(), 1).unwrap();
        assert!(!shared
            .iter()
            .any(|group| group.logical_name().ends_with("key_value_heads")));
    }

    fn local_tensor(shape: Vec<usize>) -> LocalTensorLayout {
        LocalTensorLayout::new(
            "test",
            ParameterRole::AttentionHeads,
            shape.clone(),
            shape,
            TensorPlacement::Local,
            None,
            None,
            false,
        )
    }

    #[test]
    fn local_shared_kv_geometry_is_derived_from_its_publisher() {
        let args = args();
        let mut layout = LocalModelLayout::default();
        layout.insert(
            "model.language_model.layers.0.self_attn.k_proj.weight".into(),
            local_tensor(vec![4, 16]),
        );
        layout.insert(
            "model.language_model.layers.1.self_attn.q_proj.weight".into(),
            local_tensor(vec![8, 16]),
        );
        layout.insert(
            "model.language_model.layers.1.mlp.gate_proj.weight".into(),
            local_tensor(vec![16, 16]),
        );
        layout.insert(
            "model.language_model.layers.1.experts.switch_glu.gate_up_proj".into(),
            local_tensor(vec![4, 8, 16]),
        );
        let local = local_block_args(&args, 1, &layout).unwrap();
        let policy = local.layer_policy(1).unwrap();
        assert_eq!(local.num_attention_heads, 2);
        assert_eq!(policy.num_key_value_heads.get(), 1);
        assert_eq!(policy.intermediate_size.get(), 16);
        assert_eq!(local.moe_intermediate_size, Some(4));
    }
}
