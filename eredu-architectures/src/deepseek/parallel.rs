//! Semantic parallel placement for DeepSeek target and prediction units.

use eredu_checkpoint::LinearFormat;
use eredu_nn::VocabularyParallelRange;
use eredu_runtime::{
    ArchitectureParameterDescription, ExecutionGraph, ExecutionUnitLayout, LocalModelLayout,
    MemberSharding, OwnedParameterGroupSpec, ParallelPlanError, ParameterGroupOwner,
    ParameterGroupSpec, ParameterMemberSpec, ParameterRole, StateLayout, TensorPlacement,
};

use super::{
    v3, v3_architecture_fingerprint, v4, v4_architecture_fingerprint, LayerPolicy, V3Args, V4Args,
};

/// Planner-derived construction and mutable-state geometry for DeepSeek V3.
#[derive(Debug, Clone)]
pub struct V3LocalGeometry {
    args: V3Args,
    embedding_range: VocabularyParallelRange,
    output_range: VocabularyParallelRange,
    state_layout: StateLayout,
    architecture_fingerprint: String,
}

impl V3LocalGeometry {
    /// Returns the rank-local arguments used to build every target and MTP unit.
    pub const fn args(&self) -> &V3Args {
        &self.args
    }

    /// Returns input-embedding vocabulary ownership.
    pub const fn embedding_range(&self) -> &VocabularyParallelRange {
        &self.embedding_range
    }

    /// Returns output-head vocabulary ownership.
    pub const fn output_range(&self) -> &VocabularyParallelRange {
        &self.output_range
    }

    /// Returns the authoritative rank-local state layout.
    pub const fn state_layout(&self) -> &StateLayout {
        &self.state_layout
    }

    pub(super) fn validate_for(&self, args: &V3Args) -> Result<(), ParallelPlanError> {
        if self.architecture_fingerprint != v3_architecture_fingerprint(args) {
            return Err(invalid(
                "rank-local V3 geometry belongs to a different model configuration",
            ));
        }
        self.embedding_range
            .validate_global_rows(args.vocab_size)
            .map_err(|error| invalid(error.to_string()))?;
        self.output_range
            .validate_global_rows(args.vocab_size)
            .map_err(|error| invalid(error.to_string()))?;
        let expected = v3::state_layout(&self.args)
            .map_err(|error| invalid(format!("invalid local V3 state geometry: {error}")))?;
        if expected != self.state_layout {
            return Err(invalid(
                "rank-local V3 state layout drifted from unit geometry",
            ));
        }
        Ok(())
    }
}

/// Planner-derived construction and mutable-state geometry for DeepSeek V4.
#[derive(Debug, Clone)]
pub struct V4LocalGeometry {
    args: V4Args,
    embedding_range: VocabularyParallelRange,
    output_range: VocabularyParallelRange,
    state_layout: StateLayout,
    architecture_fingerprint: String,
}

impl V4LocalGeometry {
    /// Returns the rank-local arguments used to build target and draft units.
    pub const fn args(&self) -> &V4Args {
        &self.args
    }

    /// Returns input-embedding vocabulary ownership.
    pub const fn embedding_range(&self) -> &VocabularyParallelRange {
        &self.embedding_range
    }

    /// Returns output-head vocabulary ownership.
    pub const fn output_range(&self) -> &VocabularyParallelRange {
        &self.output_range
    }

    /// Returns the authoritative rank-local state layout.
    pub const fn state_layout(&self) -> &StateLayout {
        &self.state_layout
    }

    pub(super) fn validate_for(&self, args: &V4Args) -> Result<(), ParallelPlanError> {
        if self.architecture_fingerprint != v4_architecture_fingerprint(args) {
            return Err(invalid(
                "rank-local V4 geometry belongs to a different model configuration",
            ));
        }
        self.embedding_range
            .validate_global_rows(args.vocab_size)
            .map_err(|error| invalid(error.to_string()))?;
        self.output_range
            .validate_global_rows(args.vocab_size)
            .map_err(|error| invalid(error.to_string()))?;
        let expected = v4::state_layout(&self.args)
            .map_err(|error| invalid(format!("invalid local V4 state geometry: {error}")))?;
        if expected != self.state_layout {
            return Err(invalid(
                "rank-local V4 state layout drifted from unit geometry",
            ));
        }
        Ok(())
    }
}

fn local_axis(
    layout: &LocalModelLayout,
    target: &str,
    axis: usize,
    family: &str,
) -> Result<usize, ParallelPlanError> {
    let tensor = layout.tensor(target).ok_or_else(|| {
        invalid(format!(
            "missing local {family} tensor placement for {target}"
        ))
    })?;
    let local = tensor
        .local_shape()
        .get(axis)
        .copied()
        .ok_or_else(|| invalid(format!("{family} tensor {target} has no local axis {axis}")))?;
    let global = tensor.global_shape().get(axis).copied().ok_or_else(|| {
        invalid(format!(
            "{family} tensor {target} has no global axis {axis}"
        ))
    })?;
    if local == 0 || local > global {
        return Err(invalid(format!(
            "{family} tensor {target} has invalid local width {local} of {global}"
        )));
    }
    Ok(local)
}

fn vocabulary_range(
    layout: &LocalModelLayout,
    logical_name: &str,
    global_vocabulary: usize,
    family: &str,
) -> Result<VocabularyParallelRange, ParallelPlanError> {
    let mut selected = None;
    for (target, tensor) in layout
        .tensors()
        .filter(|(_, tensor)| tensor.logical_name() == logical_name)
    {
        if tensor.global_shape().first().copied() != Some(global_vocabulary) {
            return Err(invalid(format!(
                "{family} vocabulary member {target} has global shape {:?}, expected {global_vocabulary} rows",
                tensor.global_shape()
            )));
        }
        let range = match tensor.placement() {
            TensorPlacement::Range {
                axis: 0,
                start,
                end,
            } => *start..*end,
            TensorPlacement::Replicated => 0..global_vocabulary,
            placement => {
                return Err(invalid(format!(
                    "{family} vocabulary member {target} has non-row placement {placement:?}"
                )))
            }
        };
        if selected.as_ref().is_some_and(|current| current != &range) {
            return Err(invalid(format!(
                "{family} vocabulary group {logical_name} has inconsistent companion ranges"
            )));
        }
        selected = Some(range);
    }
    let local = selected.ok_or_else(|| {
        invalid(format!(
            "missing local {family} vocabulary layout for {logical_name}"
        ))
    })?;
    let range = VocabularyParallelRange {
        global_vocabulary,
        local,
    };
    range
        .validate()
        .map_err(|error| invalid(error.to_string()))?;
    Ok(range)
}

fn same_local_axis(
    layout: &LocalModelLayout,
    targets: impl IntoIterator<Item = String>,
    axis: usize,
    family: &str,
    label: &str,
) -> Result<usize, ParallelPlanError> {
    let mut selected = None;
    for target in targets {
        let width = local_axis(layout, &target, axis, family)?;
        if selected.is_some_and(|current| current != width) {
            return Err(invalid(format!(
                "{family} {label} placement differs between execution units"
            )));
        }
        selected = Some(width);
    }
    selected.ok_or_else(|| invalid(format!("{family} has no {label} placement")))
}

/// Derives complete V3 rank-local geometry exclusively from a resolved plan.
pub fn v3_local_geometry(
    args: &V3Args,
    layout: &LocalModelLayout,
) -> Result<V3LocalGeometry, ParallelPlanError> {
    args.validate()
        .map_err(|error| invalid(error.to_string()))?;
    let target = usize::try_from(args.num_hidden_layers)
        .map_err(|_| invalid("V3 target layer count exceeds usize"))?;
    let total = usize::try_from(args.num_hidden_layers + args.num_nextn_predict_layers)
        .map_err(|_| invalid("V3 total layer count exceeds usize"))?;
    let head_width = usize::try_from(args.qk_nope_head_dim + args.qk_rope_head_dim)
        .map_err(|_| invalid("V3 query head width exceeds usize"))?;
    let query_targets = (0..total).map(|layer| {
        let root = format!("model.layers.{layer}.self_attn");
        if args.q_lora_rank.is_some() {
            format!("{root}.q_b_proj.weight")
        } else {
            format!("{root}.q_proj.weight")
        }
    });
    let query_width = same_local_axis(layout, query_targets, 0, "V3", "attention heads")?;
    if query_width % head_width != 0 {
        return Err(invalid(
            "local V3 query width does not contain complete heads",
        ));
    }
    let mut local = args.clone();
    local.num_attention_heads = i32::try_from(query_width / head_width)
        .map_err(|_| invalid("local V3 attention head count exceeds i32"))?;

    let dense_layers = (0..target)
        .filter(|layer| args.layer_schedule.get(*layer) != Some(&LayerPolicy::SparseMoe));
    if let Some(first) = dense_layers.clone().next() {
        let width = same_local_axis(
            layout,
            dense_layers.map(|layer| format!("model.layers.{layer}.mlp.gate_proj.weight")),
            0,
            "V3",
            "dense intermediate",
        )?;
        local.intermediate_size =
            i32::try_from(width).map_err(|_| invalid("local V3 dense width exceeds i32"))?;
        let _ = first;
    }
    let sparse_layers = (0..total).filter(|layer| {
        *layer >= target || args.layer_schedule.get(*layer) == Some(&LayerPolicy::SparseMoe)
    });
    if sparse_layers.clone().next().is_some() {
        let fused = same_local_axis(
            layout,
            sparse_layers.map(|layer| format!("model.layers.{layer}.mlp.experts.gate_up_proj")),
            1,
            "V3",
            "expert intermediate",
        )?;
        if fused % 2 != 0 {
            return Err(invalid("local V3 packed expert width is not even"));
        }
        local.moe_intermediate_size =
            i32::try_from(fused / 2).map_err(|_| invalid("local V3 expert width exceeds i32"))?;
    }
    local
        .validate()
        .map_err(|error| invalid(error.to_string()))?;
    let global_vocabulary = usize::try_from(args.vocab_size)
        .map_err(|_| invalid("V3 vocabulary size exceeds usize"))?;
    let geometry = V3LocalGeometry {
        state_layout: v3::state_layout(&local)
            .map_err(|error| invalid(format!("invalid local V3 state layout: {error}")))?,
        embedding_range: vocabulary_range(layout, "model.embed_tokens", global_vocabulary, "V3")?,
        output_range: vocabulary_range(layout, "lm_head", global_vocabulary, "V3")?,
        architecture_fingerprint: v3_architecture_fingerprint(args),
        args: local,
    };
    geometry.validate_for(args)?;
    Ok(geometry)
}

/// Derives complete V4 rank-local geometry exclusively from a resolved plan.
pub fn v4_local_geometry(
    args: &V4Args,
    layout: &LocalModelLayout,
) -> Result<V4LocalGeometry, ParallelPlanError> {
    args.validate()
        .map_err(|error| invalid(error.to_string()))?;
    let total = usize::try_from(args.num_hidden_layers + args.num_nextn_predict_layers)
        .map_err(|_| invalid("V4 total layer count exceeds usize"))?;
    let roots = (0..total).map(|layer| {
        if layer < args.num_hidden_layers as usize {
            format!("layers.{layer}")
        } else {
            format!("mtp.{}", layer - args.num_hidden_layers as usize)
        }
    });
    let query_width = same_local_axis(
        layout,
        roots.clone().map(|root| format!("{root}.attn.wq_b.weight")),
        0,
        "V4",
        "attention heads",
    )?;
    let head_dim = usize::try_from(args.head_dim)
        .map_err(|_| invalid("V4 attention head width exceeds usize"))?;
    if query_width % head_dim != 0 {
        return Err(invalid(
            "local V4 query width does not contain complete heads",
        ));
    }
    let output_width = same_local_axis(
        layout,
        roots.clone().map(|root| format!("{root}.attn.wo_a.weight")),
        0,
        "V4",
        "attention output groups",
    )?;
    let output_rank =
        usize::try_from(args.o_lora_rank).map_err(|_| invalid("V4 output rank exceeds usize"))?;
    if output_width % output_rank != 0 {
        return Err(invalid(
            "local V4 output projection does not contain complete groups",
        ));
    }
    let fused = same_local_axis(
        layout,
        roots.map(|root| format!("{root}.ffn.switch_mlp.gate_up_proj")),
        1,
        "V4",
        "expert intermediate",
    )?;
    if fused % 2 != 0 {
        return Err(invalid("local V4 packed expert width is not even"));
    }
    let mut local = args.clone();
    local.num_attention_heads = i32::try_from(query_width / head_dim)
        .map_err(|_| invalid("local V4 attention head count exceeds i32"))?;
    local.o_groups = i32::try_from(output_width / output_rank)
        .map_err(|_| invalid("local V4 output group count exceeds i32"))?;
    local.moe_intermediate_size =
        i32::try_from(fused / 2).map_err(|_| invalid("local V4 expert width exceeds i32"))?;
    local
        .validate()
        .map_err(|error| invalid(error.to_string()))?;
    let global_vocabulary = usize::try_from(args.vocab_size)
        .map_err(|_| invalid("V4 vocabulary size exceeds usize"))?;
    let geometry = V4LocalGeometry {
        state_layout: v4::state_layout(&local)
            .map_err(|error| invalid(format!("invalid local V4 state layout: {error}")))?,
        embedding_range: vocabulary_range(layout, "embed", global_vocabulary, "V4")?,
        output_range: vocabulary_range(layout, "head", global_vocabulary, "V4")?,
        architecture_fingerprint: v4_architecture_fingerprint(args),
        args: local,
    };
    geometry.validate_for(args)?;
    Ok(geometry)
}

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
    groups.push(replicated(
        format!("{root}.norms"),
        [
            (format!("{root}.input_layernorm.weight"), vec![hidden]),
            (
                format!("{root}.post_attention_layernorm.weight"),
                vec![hidden],
            ),
        ],
    )?);
    let sparse = layer >= target || args.layer_schedule.get(layer) == Some(&LayerPolicy::SparseMoe);
    groups.extend(if sparse {
        expert_groups_v3(args, layer)?
    } else {
        dense_v3_group(args, layer)?.into_iter().collect()
    });
    expand_linear_formats(groups, |name| args.linear_format_for(name))
}

/// Declares pinned V4 vocabulary, target hyper-head, and final norm groups.
pub fn v4_static_parameter_groups(
    args: &V4Args,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    let hidden = dim(args.hidden_size)?;
    let streams = dim(args.hc_mult)?;
    let mut groups = vec![
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
    ];
    if let Some(config) = &args.dspark {
        let last = usize::try_from(args.num_nextn_predict_layers)
            .map_err(|_| invalid("V4 DSpark depth exceeds usize"))?
            .checked_sub(1)
            .ok_or_else(|| invalid("V4 DSpark requires at least one draft layer"))?;
        let captures = config.target_layer_ids.len();
        let markov = dim(config.markov_rank)?;
        groups.push(replicated(
            "dspark",
            [
                (
                    "mtp.0.main_proj.weight".to_string(),
                    vec![hidden, hidden * captures],
                ),
                ("mtp.0.main_norm.weight".to_string(), vec![hidden]),
                (format!("mtp.{last}.norm.weight"), vec![hidden]),
                (
                    format!("mtp.{last}.hc_head_fn"),
                    vec![streams, streams * hidden],
                ),
                (format!("mtp.{last}.hc_head_base"), vec![streams]),
                (format!("mtp.{last}.hc_head_scale"), vec![1]),
                (
                    format!("mtp.{last}.markov_head.markov_w1.weight"),
                    vec![dim(args.vocab_size)?, markov],
                ),
                (
                    format!("mtp.{last}.markov_head.markov_w2.weight"),
                    vec![dim(args.vocab_size)?, markov],
                ),
                (
                    format!("mtp.{last}.confidence_head.proj.weight"),
                    vec![1, hidden + markov],
                ),
            ],
        )?);
    }
    expand_linear_formats(groups, |name| args.linear_format_for(name))
}

/// Describes every V3 target/prediction parameter with explicit graph ownership.
pub fn v3_parameter_description(
    args: &V3Args,
) -> Result<ArchitectureParameterDescription, ParallelPlanError> {
    deepseek_parameter_description(
        usize::try_from(args.num_hidden_layers)
            .map_err(|_| invalid("V3 target layer count exceeds usize"))?,
        usize::try_from(args.num_nextn_predict_layers)
            .map_err(|_| invalid("V3 prediction count exceeds usize"))?,
        v3_static_parameter_groups(args)?,
        ["embedding", "norm", "output"],
        |flat| v3_layer_parameter_groups(args, flat),
    )
}

/// Describes every V4 target/prediction parameter with explicit graph ownership.
pub fn v4_parameter_description(
    args: &V4Args,
) -> Result<ArchitectureParameterDescription, ParallelPlanError> {
    let mut static_roles = vec!["embedding", "norm", "output", "hyper_head"];
    if args.dspark.is_some() {
        static_roles.push("mtp");
    }
    deepseek_parameter_description(
        usize::try_from(args.num_hidden_layers)
            .map_err(|_| invalid("V4 target layer count exceeds usize"))?,
        usize::try_from(args.num_nextn_predict_layers)
            .map_err(|_| invalid("V4 prediction count exceeds usize"))?,
        v4_static_parameter_groups(args)?,
        static_roles,
        |flat| v4_layer_parameter_groups(args, flat),
    )
}

fn deepseek_parameter_description(
    targets: usize,
    predictions: usize,
    static_groups: Vec<ParameterGroupSpec>,
    static_roles: impl IntoIterator<Item = &'static str>,
    mut unit_groups: impl FnMut(usize) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError>,
) -> Result<ArchitectureParameterDescription, ParallelPlanError> {
    let graph = ExecutionGraph::chain(
        std::iter::once("target".to_owned())
            .chain((0..predictions).map(|depth| format!("mtp.{depth}"))),
    )
    .map_err(|error| invalid(error.to_string()))?;
    let layout = ExecutionUnitLayout::new(
        &graph,
        std::iter::once(targets).chain(std::iter::repeat_n(1, predictions)),
    )
    .map_err(|error| invalid(error.to_string()))?;
    let mut expected = static_groups.clone();
    let mut owned = static_groups
        .into_iter()
        .zip(static_roles)
        .map(|(group, role)| {
            let owner = if role == "embedding" {
                ParameterGroupOwner::static_any_of(["embedding", "mtp"])
            } else {
                ParameterGroupOwner::static_role(role)
            };
            OwnedParameterGroupSpec::new(owner, group)
        })
        .collect::<Vec<_>>();
    for flat in 0..targets + predictions {
        let (group_index, global_unit) = if flat < targets {
            (0, flat)
        } else {
            (flat - targets + 1, 0)
        };
        let groups = unit_groups(flat)?;
        expected.extend(groups.iter().cloned());
        let owner = ParameterGroupOwner::execution_unit(
            layout
                .group_id(group_index)
                .expect("DeepSeek layout group")
                .clone(),
            global_unit,
        );
        owned.extend(
            groups
                .into_iter()
                .map(|group| OwnedParameterGroupSpec::new(owner.clone(), group)),
        );
    }
    ArchitectureParameterDescription::new(&graph, &layout, expected, owned)
        .map_err(|error| invalid(error.to_string()))
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
    expand_linear_formats(groups, |name| args.linear_format_for(name))
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
    let root = format!("model.layers.{layer}.mlp");
    let experts = dim(args.n_routed_experts)?;
    let hidden = dim(args.hidden_size)?;
    let mut groups = expert_groups(
        &root,
        experts,
        dim(args.moe_intermediate_size)?,
        hidden,
        "experts.gate_up_proj",
        "experts.down_proj",
    )?;
    groups.push(replicated(
        format!("{root}.router"),
        [
            (format!("{root}.gate.weight"), vec![experts, hidden]),
            (
                format!("{root}.gate.e_score_correction_bias"),
                vec![experts],
            ),
        ],
    )?);
    groups.push(shared_expert_group(
        &root,
        dim(args.moe_intermediate_size)? * dim(args.n_shared_experts)?,
        hidden,
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

fn remap_segments(
    sharding: &MemberSharding,
    axis: usize,
    divisor: usize,
    name: &str,
) -> Result<MemberSharding, ParallelPlanError> {
    let remap = |segments: &[std::ops::Range<usize>]| {
        segments
            .iter()
            .map(|segment| {
                if !segment.start.is_multiple_of(divisor)
                    || !segment.end.is_multiple_of(divisor)
                {
                    return Err(invalid(format!(
                        "packed DeepSeek companion {name} segment {segment:?} is not aligned to {divisor}"
                    )));
                }
                Ok(segment.start / divisor..segment.end / divisor)
            })
            .collect::<Result<Vec<_>, _>>()
    };
    match sharding {
        MemberSharding::PartitionedSegments {
            axis: selected,
            segments,
        } if *selected == axis => Ok(MemberSharding::PartitionedSegments {
            axis: *selected,
            segments: remap(segments)?,
        }),
        MemberSharding::Segmented {
            axis: selected,
            segments,
        } if *selected == axis => Ok(MemberSharding::Segmented {
            axis: *selected,
            segments: remap(segments)?,
        }),
        other => Ok(other.clone()),
    }
}

fn expand_linear_member(
    source: &ParameterMemberSpec,
    format: LinearFormat,
) -> Result<Vec<ParameterMemberSpec>, ParallelPlanError> {
    let name = source.target();
    let shape = source.global_shape();
    let expert_bank = name.ends_with(".gate_up_proj") || name.ends_with(".down_proj");
    if (!name.ends_with(".weight") && !expert_bank)
        || shape.len() < 2
        || format == LinearFormat::Dense
    {
        return Ok(vec![source.clone()]);
    }
    let row_axis = shape.len() - 2;
    let column_axis = shape.len() - 1;
    let prefix = name.strip_suffix(".weight").unwrap_or(name);
    match format {
        LinearFormat::Dense => unreachable!(),
        LinearFormat::E4M3BlockFp8(fp8) => {
            fp8.validate().map_err(|error| invalid(error.to_string()))?;
            let rows = usize::try_from(fp8.block_rows)
                .map_err(|_| invalid(format!("invalid block rows for {name}")))?;
            let columns = usize::try_from(fp8.block_columns)
                .map_err(|_| invalid(format!("invalid block columns for {name}")))?;
            let mut scale_shape = shape.to_vec();
            scale_shape[row_axis] = scale_shape[row_axis].div_ceil(rows);
            scale_shape[column_axis] = scale_shape[column_axis].div_ceil(columns);
            let scale_sharding = remap_segments(source.sharding(), row_axis, rows, name)
                .and_then(|value| remap_segments(&value, column_axis, columns, name))?;
            Ok(vec![
                source.clone(),
                member(
                    if expert_bank {
                        format!("{prefix}_scales")
                    } else {
                        format!("{prefix}.weight_scale_inv")
                    },
                    scale_shape,
                    scale_sharding,
                ),
            ])
        }
        LinearFormat::GgufIQuant { ggml_type, .. } => {
            let (block_values, block_bytes) = ggml_type
                .block_and_bytes()
                .map_err(|error| invalid(error.to_string()))?;
            let block_values = usize::try_from(block_values)
                .map_err(|_| invalid(format!("GGUF block width for {name} exceeds usize")))?;
            let block_bytes = usize::try_from(block_bytes)
                .map_err(|_| invalid(format!("GGUF block bytes for {name} exceeds usize")))?;
            let input = shape[column_axis];
            if !input.is_multiple_of(block_values) {
                return Err(invalid(format!(
                    "GGUF DeepSeek matrix {name} input {input} is not aligned to block {block_values}"
                )));
            }
            let mut packed = shape.to_vec();
            packed[column_axis] = input / block_values * block_bytes;
            Ok(vec![member(
                name,
                packed,
                remap_segments(source.sharding(), column_axis, block_values, name)?,
            )])
        }
        LinearFormat::Affine(_) | LinearFormat::MxFp4 => {
            let quantization = format
                .weight_quantization()
                .expect("packed affine format has quantization");
            let bits = usize::try_from(quantization.bits())
                .map_err(|_| invalid(format!("packed bit width for {name} exceeds usize")))?;
            let group = usize::try_from(quantization.group_size())
                .map_err(|_| invalid(format!("packed group width for {name} exceeds usize")))?;
            let input = shape[column_axis];
            let packed_bits = input
                .checked_mul(bits)
                .ok_or_else(|| invalid(format!("packed DeepSeek matrix {name} overflows")))?;
            if group == 0 || !input.is_multiple_of(group) || !packed_bits.is_multiple_of(32) {
                return Err(invalid(format!(
                    "packed DeepSeek matrix {name} input {input} is incompatible with group {group} and {bits} bits"
                )));
            }
            let pack = 32 / bits;
            let mut packed = shape.to_vec();
            packed[column_axis] = packed_bits / 32;
            let mut companion = shape.to_vec();
            companion[column_axis] = input / group;
            let mut members = vec![member(
                name,
                packed,
                remap_segments(source.sharding(), column_axis, pack, name)?,
            )];
            let companion_sharding = remap_segments(source.sharding(), column_axis, group, name)?;
            members.push(member(
                if expert_bank {
                    format!("{prefix}_scales")
                } else {
                    format!("{prefix}.scales")
                },
                companion.clone(),
                companion_sharding.clone(),
            ));
            if quantization.has_biases() {
                members.push(member(
                    if expert_bank {
                        format!("{prefix}_biases")
                    } else {
                        format!("{prefix}.biases")
                    },
                    companion,
                    companion_sharding,
                ));
            }
            Ok(members)
        }
    }
}

fn gcd(mut left: usize, mut right: usize) -> usize {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

fn expand_linear_formats(
    groups: Vec<ParameterGroupSpec>,
    format: impl Fn(&str) -> LinearFormat,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    groups
        .into_iter()
        .map(|group| {
            let mut members = Vec::new();
            for source in group.members() {
                members.extend(expand_linear_member(source, format(source.target()))?);
            }
            match group.partition_units() {
                Some(mut units) => {
                    for member in &members {
                        match member.sharding() {
                            MemberSharding::Partitioned { axis } => {
                                units = gcd(units, member.global_shape()[*axis]);
                            }
                            MemberSharding::PartitionedSegments { segments, .. }
                            | MemberSharding::Segmented { segments, .. } => {
                                for segment in segments {
                                    units = gcd(units, segment.len());
                                }
                            }
                            _ => {}
                        }
                    }
                    ParameterGroupSpec::partitioned(
                        group.logical_name(),
                        group.role(),
                        units,
                        members,
                    )
                }
                None => ParameterGroupSpec::new(group.logical_name(), group.role(), members),
            }
        })
        .collect()
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
    use eredu_checkpoint::{BlockFp8Format, BlockFp8ScaleEncoding};
    use eredu_runtime::LocalTensorLayout;

    fn insert(
        layout: &mut LocalModelLayout,
        target: impl Into<String>,
        logical: impl Into<String>,
        global: Vec<usize>,
        local: Vec<usize>,
        placement: TensorPlacement,
    ) {
        layout.insert(
            target.into(),
            LocalTensorLayout::new(
                logical,
                ParameterRole::Replicated,
                global,
                local,
                placement,
                None,
                None,
                false,
            ),
        );
    }

    fn vocab(
        layout: &mut LocalModelLayout,
        target: &str,
        logical: &str,
        global: usize,
        local: std::ops::Range<usize>,
    ) {
        insert(
            layout,
            target,
            logical,
            vec![global, 8],
            vec![local.len(), 8],
            TensorPlacement::Range {
                axis: 0,
                start: local.start,
                end: local.end,
            },
        );
    }

    fn v3_args() -> V3Args {
        parse_v3_config(&serde_json::json!({
            "hidden_size": 8, "intermediate_size": 16, "moe_intermediate_size": 8,
            "num_hidden_layers": 2, "num_attention_heads": 2, "vocab_size": 31,
            "max_position_embeddings": 64, "kv_lora_rank": 4, "qk_nope_head_dim": 2,
            "qk_rope_head_dim": 2, "v_head_dim": 2, "first_k_dense_replace": 1,
            "n_routed_experts": 4, "n_shared_experts": 1, "num_experts_per_tok": 2,
            "n_group": 2, "topk_group": 1, "num_nextn_predict_layers": 1,
            "tie_word_embeddings": false
        }))
        .unwrap()
    }

    fn v4_args() -> V4Args {
        parse_v4_config(&serde_json::json!({
            "hidden_size": 8, "moe_intermediate_size": 8, "num_hidden_layers": 3,
            "num_attention_heads": 2, "head_dim": 4, "qk_rope_head_dim": 2,
            "q_lora_rank": 4, "o_lora_rank": 2, "o_groups": 2, "vocab_size": 31,
            "max_position_embeddings": 64, "sliding_window": 8,
            "compress_ratios": [0, 4, 128, 0], "index_n_heads": 2,
            "index_head_dim": 4, "index_topk": 1, "hc_mult": 2,
            "hc_sinkhorn_iters": 2, "n_routed_experts": 4, "num_experts_per_tok": 2,
            "scoring_func": "sqrtsoftplus", "topk_method": "noaux_tc",
            "norm_topk_prob": true, "num_nextn_predict_layers": 1
        }))
        .unwrap()
    }

    #[test]
    fn v3_fp8_plan_keeps_inverse_scales_atomic_with_weight_shards() {
        let mut args = v3_args();
        args.hidden_size = 128;
        args.intermediate_size = 128;
        args.moe_intermediate_size = 128;
        args.linear_format = LinearFormat::E4M3BlockFp8(
            BlockFp8Format::new(128, 128, BlockFp8ScaleEncoding::Ue8m0).unwrap(),
        );
        let groups = v3_layer_parameter_groups(&args, 1).unwrap();
        assert!(groups
            .iter()
            .flat_map(ParameterGroupSpec::members)
            .any(|member| {
                member
                    .target()
                    .ends_with("self_attn.q_proj.weight_scale_inv")
            }));
        let expert = groups
            .iter()
            .find(|group| group.logical_name().ends_with("routed_expert_intermediate"))
            .unwrap();
        let names = expert
            .members()
            .iter()
            .map(ParameterMemberSpec::target)
            .collect::<Vec<_>>();
        assert!(names
            .iter()
            .any(|name| name.ends_with("experts.gate_up_proj")));
        assert!(names
            .iter()
            .any(|name| name.ends_with("experts.gate_up_proj_scales")));
        assert!(names
            .iter()
            .any(|name| name.ends_with("experts.down_proj_scales")));
        assert_eq!(expert.partition_units(), Some(1));
    }

    #[test]
    fn plans_name_attention_experts_indexes_hyper_streams_and_draft_groups() {
        let v3 = v3_args();
        let v3_description = v3_parameter_description(&v3).unwrap();
        assert_eq!(v3_description.unit_layout().group_range(0), Some(0..2));
        assert_eq!(v3_description.unit_layout().group_range(1), Some(2..3));
        let v3_groups = v3_layer_parameter_groups(&v3, 2).unwrap();
        assert!(v3_groups
            .iter()
            .any(|group| group.role() == ParameterRole::AttentionHeads));
        assert!(v3_groups
            .iter()
            .any(|group| group.role() == ParameterRole::ExpertIntermediate));

        let v4 = v4_args();
        let v4_description = v4_parameter_description(&v4).unwrap();
        assert_eq!(v4_description.unit_layout().group_range(0), Some(0..3));
        assert_eq!(v4_description.unit_layout().group_range(1), Some(3..4));
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

    #[test]
    fn v3_description_owns_each_block_norm_with_its_execution_unit() {
        let description = v3_parameter_description(&v3_args()).unwrap();
        let norms = description
            .groups()
            .iter()
            .filter(|owned| owned.group().logical_name().ends_with(".norms"))
            .map(|owned| {
                (
                    owned.group().logical_name(),
                    owned.owner(),
                    owned
                        .group()
                        .members()
                        .iter()
                        .map(ParameterMemberSpec::target)
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<_>>();

        assert_eq!(norms.len(), 3);
        for (layer, (logical_name, owner, members)) in norms.into_iter().enumerate() {
            assert_eq!(logical_name, format!("model.layers.{layer}.norms"));
            let (group, global_unit) = match owner {
                ParameterGroupOwner::ExecutionUnit { group, global_unit } => {
                    (group.as_str(), *global_unit)
                }
                owner => panic!("unexpected V3 norm owner {owner:?}"),
            };
            if layer < 2 {
                assert_eq!((group, global_unit), ("target", layer));
            } else {
                assert_eq!((group, global_unit), ("mtp.0", 0));
            }
            assert_eq!(
                members,
                [
                    format!("model.layers.{layer}.input_layernorm.weight"),
                    format!("model.layers.{layer}.post_attention_layernorm.weight"),
                ]
            );
        }
    }

    #[test]
    fn v3_description_owns_each_sparse_router_with_its_execution_unit() {
        let description = v3_parameter_description(&v3_args()).unwrap();
        let routers = description
            .groups()
            .iter()
            .filter(|owned| owned.group().logical_name().ends_with(".router"))
            .collect::<Vec<_>>();

        assert_eq!(routers.len(), 2);
        for (layer, owned, expected_group, expected_unit) in
            [(1, routers[0], "target", 1), (2, routers[1], "mtp.0", 0)]
        {
            assert_eq!(
                owned.group().logical_name(),
                format!("model.layers.{layer}.mlp.router")
            );
            let (group, global_unit) = match owned.owner() {
                ParameterGroupOwner::ExecutionUnit { group, global_unit } => {
                    (group.as_str(), *global_unit)
                }
                owner => panic!("unexpected V3 router owner {owner:?}"),
            };
            assert_eq!((group, global_unit), (expected_group, expected_unit));
            assert_eq!(
                owned
                    .group()
                    .members()
                    .iter()
                    .map(ParameterMemberSpec::target)
                    .collect::<Vec<_>>(),
                [
                    format!("model.layers.{layer}.mlp.gate.weight"),
                    format!("model.layers.{layer}.mlp.gate.e_score_correction_bias"),
                ]
            );
        }
    }

    #[test]
    fn v4_static_modules_have_distinct_architecture_owned_roles() {
        let description = v4_parameter_description(&v4_args()).unwrap();
        let roles = description
            .groups()
            .iter()
            .filter_map(|group| match group.owner() {
                ParameterGroupOwner::StaticRole(role) => {
                    Some((group.group().logical_name(), role.as_str()))
                }
                ParameterGroupOwner::StaticAnyOf(roles) => Some((
                    group.group().logical_name(),
                    roles.first().expect("shared static owner").as_str(),
                )),
                ParameterGroupOwner::ExecutionUnit { .. } => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(
            roles,
            [
                ("embed", "embedding"),
                ("norm", "norm"),
                ("head", "output"),
                ("hyper_head", "hyper_head"),
            ]
        );
    }

    #[test]
    fn v4_boundary_plan_owns_flattened_transport_width() {
        let mut args = v4_args();
        let boundary = v4::TargetBoundarySchema::from_args(&args).unwrap();
        assert_eq!(boundary.activation_hidden_size(), 16);

        args.hidden_size = i32::MAX;
        args.hc_mult = 2;
        assert!(v4::TargetBoundarySchema::from_args(&args).is_err());
    }

    #[test]
    fn v3_geometry_uses_one_plan_for_units_vocabulary_and_state() {
        let args = v3_args();
        let mut layout = LocalModelLayout::default();
        vocab(
            &mut layout,
            "model.embed_tokens.weight",
            "model.embed_tokens",
            31,
            0..16,
        );
        vocab(&mut layout, "lm_head.weight", "lm_head", 31, 0..16);
        for layer in 0..3 {
            insert(
                &mut layout,
                format!("model.layers.{layer}.self_attn.q_proj.weight"),
                format!("model.layers.{layer}.self_attn.heads"),
                vec![8, 8],
                vec![4, 8],
                TensorPlacement::Range {
                    axis: 0,
                    start: 0,
                    end: 4,
                },
            );
        }
        insert(
            &mut layout,
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.intermediate",
            vec![16, 8],
            vec![8, 8],
            TensorPlacement::Range {
                axis: 0,
                start: 0,
                end: 8,
            },
        );
        for layer in 1..3 {
            insert(
                &mut layout,
                format!("model.layers.{layer}.mlp.experts.gate_up_proj"),
                format!("model.layers.{layer}.mlp.routed_expert_intermediate"),
                vec![4, 16, 8],
                vec![4, 8, 8],
                TensorPlacement::Range {
                    axis: 1,
                    start: 0,
                    end: 8,
                },
            );
        }
        let geometry = v3_local_geometry(&args, &layout).unwrap();
        assert_eq!(geometry.args().num_attention_heads, 1);
        assert_eq!(geometry.args().intermediate_size, 8);
        assert_eq!(geometry.args().moe_intermediate_size, 4);
        assert_eq!(geometry.embedding_range().local, 0..16);
        assert_eq!(
            geometry.state_layout(),
            &v3::state_layout(geometry.args()).unwrap()
        );
    }

    #[test]
    fn v4_geometry_rejects_unit_drift_before_construction() {
        let args = v4_args();
        let mut layout = LocalModelLayout::default();
        vocab(&mut layout, "embed.weight", "embed", 31, 16..31);
        vocab(&mut layout, "head.weight", "head", 31, 16..31);
        for layer in 0..4 {
            let root = if layer < 3 {
                format!("layers.{layer}")
            } else {
                "mtp.0".into()
            };
            let query = if layer == 3 { 8 } else { 4 };
            insert(
                &mut layout,
                format!("{root}.attn.wq_b.weight"),
                format!("{root}.attn.output_groups"),
                vec![8, 4],
                vec![query, 4],
                TensorPlacement::Range {
                    axis: 0,
                    start: 0,
                    end: query,
                },
            );
            insert(
                &mut layout,
                format!("{root}.attn.wo_a.weight"),
                format!("{root}.attn.output_groups"),
                vec![4, 4],
                vec![2, 4],
                TensorPlacement::Range {
                    axis: 0,
                    start: 0,
                    end: 2,
                },
            );
            insert(
                &mut layout,
                format!("{root}.ffn.switch_mlp.experts.gate_up_proj"),
                format!("{root}.ffn.routed_expert_intermediate"),
                vec![4, 16, 8],
                vec![4, 8, 8],
                TensorPlacement::Range {
                    axis: 1,
                    start: 0,
                    end: 8,
                },
            );
        }
        assert!(matches!(
            v4_local_geometry(&args, &layout),
            Err(ParallelPlanError::InvalidGroup(message))
                if message.contains("differs between execution units")
        ));
    }
}
