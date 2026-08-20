//! Semantic parallel placement for DeepSeek target and prediction units.

use eredu_runtime::{
    MemberSharding, ParallelPlanError, ParameterGroupSpec, ParameterMemberSpec, ParameterRole,
};

use super::{LayerPolicy, V3Args, V4Args};

/// Declares pinned V3 embedding, normalization, and vocabulary placement.
pub fn v3_static_parameter_groups(
    args: &V3Args,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    Ok(vec![
        group(
            "model.embed_tokens",
            ParameterRole::Vocabulary,
            [(
                "model.embed_tokens.weight",
                vec![dim(args.vocab_size)?, dim(args.hidden_size)?],
                MemberSharding::Balanced { axis: 0 },
            )],
        )?,
        replicated(
            "model.norm",
            [("model.norm.weight", vec![dim(args.hidden_size)?])],
        )?,
        group(
            "lm_head",
            ParameterRole::Vocabulary,
            [(
                "lm_head.weight",
                vec![dim(args.vocab_size)?, dim(args.hidden_size)?],
                MemberSharding::Balanced { axis: 0 },
            )],
        )?,
    ])
}

/// Declares one V3 target or embedded-prediction unit's semantic groups.
pub fn v3_layer_parameter_groups(
    args: &V3Args,
    layer: usize,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    let target = usize::try_from(args.num_hidden_layers)
        .map_err(|_| invalid("V3 target layer count exceeds usize"))?;
    let total = usize::try_from(args.num_hidden_layers + args.num_nextn_predict_layers)
        .map_err(|_| invalid("V3 total layer count exceeds usize"))?;
    if layer >= total {
        return Err(invalid(format!(
            "V3 layer {layer} is outside {total} units"
        )));
    }
    let root = format!("model.layers.{layer}");
    let attention = format!("{root}.self_attn");
    let heads = dim(args.num_attention_heads)?;
    let query_width = dim(args.qk_nope_head_dim + args.qk_rope_head_dim)?;
    let kv_width = dim(args.qk_nope_head_dim + args.v_head_dim)?;
    let rank = dim(args.kv_lora_rank)?;
    let hidden = dim(args.hidden_size)?;
    let mut members = Vec::new();
    if let Some(query_rank) = args.q_lora_rank {
        members.push(member(
            format!("{attention}.q_b_proj.weight"),
            vec![heads * query_width, dim(query_rank)?],
            MemberSharding::Partitioned { axis: 0 },
        ));
    } else {
        members.push(member(
            format!("{attention}.q_proj.weight"),
            vec![heads * query_width, hidden],
            MemberSharding::Partitioned { axis: 0 },
        ));
    }
    members.extend([
        member(
            format!("{attention}.kv_b_proj.weight"),
            vec![heads * kv_width, rank],
            MemberSharding::Partitioned { axis: 0 },
        ),
        member(
            format!("{attention}.o_proj.weight"),
            vec![hidden, heads * dim(args.v_head_dim)?],
            MemberSharding::Partitioned { axis: 1 },
        ),
    ]);
    let mut groups = vec![ParameterGroupSpec::partitioned(
        format!("{attention}.heads"),
        ParameterRole::AttentionHeads,
        heads,
        members,
    )?];
    groups.push(replicated(
        format!("{root}.attention_rank"),
        [
            (
                format!("{attention}.kv_a_proj_with_mqa.weight"),
                vec![rank + dim(args.qk_rope_head_dim)?, hidden],
            ),
            (format!("{attention}.kv_a_layernorm.weight"), vec![rank]),
        ],
    )?);
    if let Some(query_rank) = args.q_lora_rank {
        groups.push(replicated(
            format!("{root}.query_rank"),
            [
                (
                    format!("{attention}.q_a_proj.weight"),
                    vec![dim(query_rank)?, hidden],
                ),
                (
                    format!("{attention}.q_a_layernorm.weight"),
                    vec![dim(query_rank)?],
                ),
            ],
        )?);
    }
    let sparse = layer >= target || args.layer_schedule.get(layer) == Some(&LayerPolicy::SparseMoe);
    groups.extend(if sparse {
        expert_groups_v3(args, layer)?
    } else {
        dense_v3_group(args, layer)?.into_iter().collect()
    });
    Ok(groups)
}

/// Declares pinned V4 vocabulary, target hyper-head, and final norm groups.
pub fn v4_static_parameter_groups(
    args: &V4Args,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    let hidden = dim(args.hidden_size)?;
    let streams = dim(args.hc_mult)?;
    Ok(vec![
        group(
            "embed",
            ParameterRole::Vocabulary,
            [(
                "embed.weight",
                vec![dim(args.vocab_size)?, hidden],
                MemberSharding::Balanced { axis: 0 },
            )],
        )?,
        replicated("norm", [("norm.weight", vec![hidden])])?,
        group(
            "head",
            ParameterRole::Vocabulary,
            [(
                "head.weight",
                vec![dim(args.vocab_size)?, hidden],
                MemberSharding::Balanced { axis: 0 },
            )],
        )?,
        replicated(
            "hyper_head",
            [
                ("hc_head_fn", vec![streams, streams * hidden]),
                ("hc_head_base", vec![streams]),
                ("hc_head_scale", vec![1]),
            ],
        )?,
    ])
}

/// Declares one V4 target, MTP, or DSpark block's low-rank, index-head,
/// hyper-stream, and expert placement.
pub fn v4_layer_parameter_groups(
    args: &V4Args,
    layer: usize,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    let target = usize::try_from(args.num_hidden_layers)
        .map_err(|_| invalid("V4 target layer count exceeds usize"))?;
    let total = usize::try_from(args.num_hidden_layers + args.num_nextn_predict_layers)
        .map_err(|_| invalid("V4 total layer count exceeds usize"))?;
    if layer >= total {
        return Err(invalid(format!(
            "V4 layer {layer} is outside {total} units"
        )));
    }
    let root = if layer < target {
        format!("layers.{layer}")
    } else {
        format!("mtp.{}", layer - target)
    };
    let heads = dim(args.num_attention_heads)?;
    let hidden = dim(args.hidden_size)?;
    let head = dim(args.head_dim)?;
    let groups_count = dim(args.o_groups)?;
    let rank = dim(args.o_lora_rank)?;
    let mut groups = vec![ParameterGroupSpec::partitioned(
        format!("{root}.attn.output_groups"),
        ParameterRole::AttentionHeads,
        groups_count,
        [
            member(
                format!("{root}.attn.wq_b.weight"),
                vec![heads * head, dim(args.q_lora_rank)?],
                MemberSharding::Partitioned { axis: 0 },
            ),
            member(
                format!("{root}.attn.wo_a.weight"),
                vec![groups_count * rank, heads * head / groups_count],
                MemberSharding::Partitioned { axis: 0 },
            ),
            member(
                format!("{root}.attn.wo_b.weight"),
                vec![hidden, groups_count * rank],
                MemberSharding::Partitioned { axis: 1 },
            ),
            member(
                format!("{root}.attn.attn_sink"),
                vec![heads],
                MemberSharding::Partitioned { axis: 0 },
            ),
        ],
    )?];
    let streams = dim(args.hc_mult)?;
    groups.push(replicated(
        format!("{root}.hyper_streams"),
        [
            (
                format!("{root}.hc_attn_fn"),
                vec![(2 + streams) * streams, streams * hidden],
            ),
            (
                format!("{root}.hc_attn_base"),
                vec![(2 + streams) * streams],
            ),
            (format!("{root}.hc_attn_scale"), vec![3]),
            (
                format!("{root}.hc_ffn_fn"),
                vec![(2 + streams) * streams, streams * hidden],
            ),
            (format!("{root}.hc_ffn_base"), vec![(2 + streams) * streams]),
            (format!("{root}.hc_ffn_scale"), vec![3]),
        ],
    )?);
    if layer < target
        && matches!(
            args.attention_policy(layer),
            Some(super::V4AttentionPolicy::Compressed { ratio: 4 })
        )
    {
        groups.push(replicated(
            format!("{root}.attn.index_heads"),
            [
                (
                    format!("{root}.attn.indexer.wq_b.weight"),
                    vec![
                        dim(args.index_n_heads)? * dim(args.index_head_dim)?,
                        dim(args.q_lora_rank)?,
                    ],
                ),
                (
                    format!("{root}.attn.indexer.weights_proj.weight"),
                    vec![dim(args.index_n_heads)?, hidden],
                ),
            ],
        )?);
    }
    groups.extend(expert_groups_v4(args, layer, &root)?);
    Ok(groups)
}

fn dense_v3_group(
    args: &V3Args,
    layer: usize,
) -> Result<Option<ParameterGroupSpec>, ParallelPlanError> {
    let root = format!("model.layers.{layer}.mlp");
    let width = dim(args.intermediate_size)?;
    let hidden = dim(args.hidden_size)?;
    Ok(Some(ParameterGroupSpec::partitioned(
        format!("{root}.intermediate"),
        ParameterRole::FeedForwardIntermediate,
        width,
        [
            member(
                format!("{root}.gate_proj.weight"),
                vec![width, hidden],
                MemberSharding::Partitioned { axis: 0 },
            ),
            member(
                format!("{root}.up_proj.weight"),
                vec![width, hidden],
                MemberSharding::Partitioned { axis: 0 },
            ),
            member(
                format!("{root}.down_proj.weight"),
                vec![hidden, width],
                MemberSharding::Partitioned { axis: 1 },
            ),
        ],
    )?))
}

fn expert_groups_v3(
    args: &V3Args,
    layer: usize,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    let mut groups = expert_groups(
        &format!("model.layers.{layer}.mlp"),
        dim(args.n_routed_experts)?,
        dim(args.moe_intermediate_size)?,
        dim(args.hidden_size)?,
        "experts.gate_up_proj",
        "experts.down_proj",
    )?;
    groups.push(shared_expert_group(
        &format!("model.layers.{layer}.mlp"),
        dim(args.moe_intermediate_size)? * dim(args.n_shared_experts)?,
        dim(args.hidden_size)?,
        "shared_experts.gate_proj.weight",
        "shared_experts.up_proj.weight",
        "shared_experts.down_proj.weight",
    )?);
    Ok(groups)
}

fn expert_groups_v4(
    args: &V4Args,
    _layer: usize,
    root: &str,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    let mut groups = expert_groups(
        &format!("{root}.ffn"),
        dim(args.n_routed_experts)?,
        dim(args.moe_intermediate_size)?,
        dim(args.hidden_size)?,
        "switch_mlp.gate_up_proj",
        "switch_mlp.down_proj",
    )?;
    groups.push(shared_expert_group(
        &format!("{root}.ffn"),
        dim(args.moe_intermediate_size)? * dim(args.n_shared_experts)?,
        dim(args.hidden_size)?,
        "shared_experts.w1.weight",
        "shared_experts.w3.weight",
        "shared_experts.w2.weight",
    )?);
    Ok(groups)
}

fn shared_expert_group(
    root: &str,
    width: usize,
    hidden: usize,
    gate: &str,
    up: &str,
    down: &str,
) -> Result<ParameterGroupSpec, ParallelPlanError> {
    ParameterGroupSpec::partitioned(
        format!("{root}.shared_expert_intermediate"),
        ParameterRole::ExpertIntermediate,
        width,
        [
            member(
                format!("{root}.{gate}"),
                vec![width, hidden],
                MemberSharding::Partitioned { axis: 0 },
            ),
            member(
                format!("{root}.{up}"),
                vec![width, hidden],
                MemberSharding::Partitioned { axis: 0 },
            ),
            member(
                format!("{root}.{down}"),
                vec![hidden, width],
                MemberSharding::Partitioned { axis: 1 },
            ),
        ],
    )
}

fn expert_groups(
    root: &str,
    experts: usize,
    width: usize,
    hidden: usize,
    gate_up: &str,
    down: &str,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    Ok(vec![ParameterGroupSpec::partitioned(
        format!("{root}.routed_expert_intermediate"),
        ParameterRole::ExpertIntermediate,
        width,
        [
            member(
                format!("{root}.{gate_up}"),
                vec![experts, 2 * width, hidden],
                MemberSharding::PartitionedSegments {
                    axis: 1,
                    segments: vec![0..width, width..2 * width],
                },
            ),
            member(
                format!("{root}.{down}"),
                vec![experts, hidden, width],
                MemberSharding::Partitioned { axis: 2 },
            ),
        ],
    )?])
}

fn replicated<N, I>(
    name: impl Into<String>,
    members: I,
) -> Result<ParameterGroupSpec, ParallelPlanError>
where
    N: Into<String>,
    I: IntoIterator<Item = (N, Vec<usize>)>,
{
    group(
        name,
        ParameterRole::Replicated,
        members
            .into_iter()
            .map(|(name, shape)| (name, shape, MemberSharding::Replicated)),
    )
}

fn group<N, I>(
    name: impl Into<String>,
    role: ParameterRole,
    members: I,
) -> Result<ParameterGroupSpec, ParallelPlanError>
where
    N: Into<String>,
    I: IntoIterator<Item = (N, Vec<usize>, MemberSharding)>,
{
    ParameterGroupSpec::new(
        name,
        role,
        members
            .into_iter()
            .map(|(name, shape, sharding)| member(name, shape, sharding)),
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
        .ok_or_else(|| invalid(format!("parallel dimension must be positive, got {value}")))
}

fn invalid(message: impl Into<String>) -> ParallelPlanError {
    ParallelPlanError::InvalidGroup(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deepseek::{parse_v3_config, parse_v4_config};

    #[test]
    fn plans_name_attention_experts_indexes_hyper_streams_and_draft_groups() {
        let v3 = parse_v3_config(&serde_json::json!({
            "hidden_size": 8, "intermediate_size": 16, "moe_intermediate_size": 8,
            "num_hidden_layers": 2, "num_attention_heads": 2, "vocab_size": 32,
            "max_position_embeddings": 64, "kv_lora_rank": 4, "qk_nope_head_dim": 2,
            "qk_rope_head_dim": 2, "v_head_dim": 2, "first_k_dense_replace": 1,
            "n_routed_experts": 4, "n_shared_experts": 1, "num_experts_per_tok": 2,
            "n_group": 2, "topk_group": 1, "num_nextn_predict_layers": 1,
            "tie_word_embeddings": false
        }))
        .unwrap();
        let v3_groups = v3_layer_parameter_groups(&v3, 2).unwrap();
        assert!(v3_groups
            .iter()
            .any(|group| group.role() == ParameterRole::AttentionHeads));
        assert!(v3_groups
            .iter()
            .any(|group| group.role() == ParameterRole::ExpertIntermediate));

        let v4 = parse_v4_config(&serde_json::json!({
            "hidden_size": 8, "moe_intermediate_size": 8, "num_hidden_layers": 3,
            "num_attention_heads": 2, "head_dim": 4, "qk_rope_head_dim": 2,
            "q_lora_rank": 4, "o_lora_rank": 2, "o_groups": 2, "vocab_size": 32,
            "max_position_embeddings": 64, "sliding_window": 8,
            "compress_ratios": [0, 4, 128, 0], "index_n_heads": 2,
            "index_head_dim": 4, "index_topk": 1, "hc_mult": 2,
            "hc_sinkhorn_iters": 2, "n_routed_experts": 4, "num_experts_per_tok": 2,
            "scoring_func": "sqrtsoftplus", "topk_method": "noaux_tc",
            "norm_topk_prob": true, "num_nextn_predict_layers": 1
        }))
        .unwrap();
        let target = v4_layer_parameter_groups(&v4, 1).unwrap();
        assert!(target
            .iter()
            .any(|group| group.logical_name().contains("index_heads")));
        assert!(target
            .iter()
            .any(|group| group.logical_name().contains("hyper_streams")));
        let draft = v4_layer_parameter_groups(&v4, 3).unwrap();
        assert!(draft
            .iter()
            .all(|group| group.logical_name().starts_with("mtp.0")));
    }
}
