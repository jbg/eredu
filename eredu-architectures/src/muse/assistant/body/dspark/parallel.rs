//! DSpark body parameter ownership and TP-local neural geometry.

use super::DsparkBodyDefinition;
use eredu_core::ParallelRankTopology;
use eredu_nn::Error;
use eredu_runtime::{
    derive_partitioned_local_layout, expand_linear_format_parameter_groups,
    ArchitectureParameterDescription, ExecutionGraph, ExecutionGroupId, MemberSharding,
    OwnedParameterGroupSpec, ParameterGroupOwner, ParameterGroupSpec, ParameterMemberSpec,
    ParameterRole,
};

impl DsparkBodyDefinition {
    pub(in crate::muse::assistant) fn partitions_heads_evenly(&self, parts: usize) -> bool {
        self.args.num_attention_heads as usize % parts == 0
            && self.args.num_key_value_heads as usize % parts == 0
    }

    /// Declares context, decoder and final-norm parameters at their actual unit
    /// boundaries, including all selected physical encoding companions.
    pub fn parameter_description(&self) -> Result<ArchitectureParameterDescription, Error> {
        let a = &self.args;
        let hidden = a.hidden_size as usize;
        let head = a.head_dim as usize;
        let queries = a.num_attention_heads as usize;
        let keys = a.num_key_value_heads as usize;
        let intermediate = a.intermediate_size as usize;
        let replicated = |name: &str, members: Vec<(&str, Vec<usize>)>| {
            ParameterGroupSpec::new(
                name,
                ParameterRole::Replicated,
                members.into_iter().map(|(name, shape)| {
                    ParameterMemberSpec::new(name, shape, MemberSharding::Replicated)
                }),
            )
            .map_err(Error::backend_source)
        };
        let mut owned = Vec::new();
        let mut append = |group_id: &str, unit: usize, groups: Vec<ParameterGroupSpec>| {
            let groups = expand_linear_format_parameter_groups(groups, |member| {
                crate::linear_format::standard_parallel_linear_format(
                    member,
                    a.linear_format_for(member.target()),
                )
            })
            .map_err(Error::backend_source)?;
            let id = ExecutionGroupId::new(group_id).map_err(Error::backend_source)?;
            owned.extend(groups.into_iter().map(|group| {
                OwnedParameterGroupSpec::new(
                    ParameterGroupOwner::execution_unit(id.clone(), unit),
                    group,
                )
            }));
            Ok::<_, Error>(())
        };
        append(
            "assistant.context",
            0,
            vec![replicated(
                "assistant.context",
                vec![
                    (
                        "fc.weight",
                        vec![hidden, hidden * self.geometry.captures as usize],
                    ),
                    ("hidden_norm.weight", vec![hidden]),
                ],
            )?],
        )?;
        for layer in 0..self.geometry.layers {
            let root = format!("layers.{layer}");
            let attention = format!("{root}.self_attn");
            let feed_forward = format!("{root}.feed_forward");
            let member = |name: String, shape, axis| {
                ParameterMemberSpec::new(name, shape, MemberSharding::Partitioned { axis })
            };
            let groups = vec![
                ParameterGroupSpec::partitioned(
                    format!("{attention}.query_heads"),
                    ParameterRole::AttentionHeads,
                    queries,
                    [
                        member(
                            format!("{attention}.q_proj.weight"),
                            vec![queries * head, hidden],
                            0,
                        ),
                        member(
                            format!("{attention}.o_gate_proj.weight"),
                            vec![queries * head, hidden],
                            0,
                        ),
                        member(
                            format!("{attention}.o_proj.weight"),
                            vec![hidden, queries * head],
                            1,
                        ),
                    ],
                )
                .map_err(Error::backend_source)?,
                ParameterGroupSpec::partitioned(
                    format!("{attention}.key_value_heads"),
                    ParameterRole::AttentionHeads,
                    keys,
                    [
                        member(
                            format!("{attention}.k_proj.weight"),
                            vec![keys * head, hidden],
                            0,
                        ),
                        member(
                            format!("{attention}.v_proj.weight"),
                            vec![keys * head, hidden],
                            0,
                        ),
                    ],
                )
                .map_err(Error::backend_source)?,
                ParameterGroupSpec::partitioned(
                    format!("{feed_forward}.intermediate"),
                    ParameterRole::FeedForwardIntermediate,
                    intermediate,
                    [
                        member(
                            format!("{feed_forward}.gate_proj.weight"),
                            vec![intermediate, hidden],
                            0,
                        ),
                        member(
                            format!("{feed_forward}.up_proj.weight"),
                            vec![intermediate, hidden],
                            0,
                        ),
                        member(
                            format!("{feed_forward}.down_proj.weight"),
                            vec![hidden, intermediate],
                            1,
                        ),
                    ],
                )
                .map_err(Error::backend_source)?,
                ParameterGroupSpec::new(
                    format!("{root}.normalization_and_residual"),
                    ParameterRole::Replicated,
                    [
                        "self_attn.pre_qkv_proj_norm.weight",
                        "pre_ffn_norm.weight",
                        "post_ffn_norm.weight",
                        "add_attn.gate",
                        "add_ffn.gate",
                    ]
                    .into_iter()
                    .map(|suffix| {
                        ParameterMemberSpec::new(
                            format!("{root}.{suffix}"),
                            vec![hidden],
                            MemberSharding::Replicated,
                        )
                    }),
                )
                .map_err(Error::backend_source)?,
            ];
            append("assistant.decoder", layer, groups)?;
        }
        append(
            "assistant.final_norm",
            0,
            vec![replicated(
                "assistant.final_norm",
                vec![("norm.weight", vec![hidden])],
            )?],
        )?;
        let graph = ExecutionGraph::chain([
            "assistant.context",
            "assistant.decoder",
            "assistant.final_norm",
        ])
        .map_err(Error::backend_source)?;
        ArchitectureParameterDescription::new(
            &graph,
            self.geometry.unit_layout(),
            owned.iter().map(|group| group.group().clone()),
            owned.clone(),
        )
        .map_err(Error::backend_source)
    }

    /// Selects exact TP placement and derives local head, K/V and dense widths.
    /// This body primitive does not place assistant heads or pipeline stages.
    pub fn partition_tensor_parallel(
        &self,
        rank: ParallelRankTopology,
    ) -> Result<(Self, ArchitectureParameterDescription), Error> {
        if rank.pipeline_parallel_size() != 1
            || rank.expert_parallel_size() != 1
            || rank.data_parallel_size() != 1
        {
            return Err(Error::backend(
                "DSpark body TP construction requires singleton pipeline, expert and data axes",
            ));
        }
        let parts = rank.tensor_parallel_size();
        if self.args.num_attention_heads as usize % parts != 0
            || self.args.num_key_value_heads as usize % parts != 0
        {
            return Err(Error::backend(
                "DSpark body TP construction requires evenly partitioned query and key/value heads",
            ));
        }
        let description = self.parameter_description()?;
        let layout = derive_partitioned_local_layout(&description, rank).map_err(Error::backend)?;
        let width = |name: &str| {
            layout
                .tensor(name)
                .and_then(|tensor| tensor.local_shape().first())
                .and_then(|width| i32::try_from(*width).ok())
                .filter(|width| *width > 0)
                .ok_or_else(|| Error::backend(format!("invalid DSpark local width for {name}")))
        };
        let mut local = self.clone();
        local.blocks = (0..self.geometry.layers)
            .map(|layer| {
                let mut args = self.args.clone();
                let root = format!("layers.{layer}");
                let queries = width(&format!("{root}.self_attn.q_proj.weight"))?;
                let keys = width(&format!("{root}.self_attn.k_proj.weight"))?;
                if queries % args.head_dim != 0 || keys % args.head_dim != 0 {
                    return Err(Error::backend(
                        "DSpark local projection splits an attention head",
                    ));
                }
                args.num_attention_heads = queries / args.head_dim;
                args.num_key_value_heads = keys / args.head_dim;
                args.intermediate_size = width(&format!("{root}.feed_forward.gate_proj.weight"))?;
                args.validate().map_err(Error::backend)?;
                Ok(args)
            })
            .collect::<Result<_, Error>>()?;
        let keys = local.blocks[0].num_key_value_heads;
        if local
            .blocks
            .iter()
            .any(|args| args.num_key_value_heads != keys)
        {
            return Err(Error::backend(
                "DSpark body context requires uniform local key/value heads",
            ));
        }
        local.geometry.key_value_heads = keys;
        let description = description
            .with_partition_layout(rank, layout)
            .map_err(Error::backend_source)?;
        Ok((local, description))
    }
}
