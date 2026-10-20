//! Backend-free canonical parameter topology for target and prediction units.

use std::collections::BTreeMap;

use eredu_nn::LinearFormatSpec;
use eredu_runtime::{
    ArchitectureParameterDescription, ExecutionUnitLayout, MemberSharding as S,
    OwnedParameterGroupSpec, ParallelPlanError, ParameterGroupOwner, ParameterGroupSpec,
    ParameterMemberSpec, ParameterRole,
};

use super::{LayerPolicy, ModelArgs};

fn invalid(error: impl std::fmt::Display) -> ParallelPlanError {
    ParallelPlanError::InvalidGroup(error.to_string())
}

fn checked_width(name: &str, width: usize) -> Result<usize, ParallelPlanError> {
    i32::try_from(width).map_err(|_| invalid(format!("Nemotron-H {name} width exceeds i32")))?;
    Ok(width)
}

fn product(name: &str, left: usize, right: usize) -> Result<usize, ParallelPlanError> {
    checked_width(
        name,
        left.checked_mul(right)
            .ok_or_else(|| invalid(format!("Nemotron-H {name} width overflowed usize")))?,
    )
}

fn sum(name: &str, left: usize, right: usize) -> Result<usize, ParallelPlanError> {
    checked_width(
        name,
        left.checked_add(right)
            .ok_or_else(|| invalid(format!("Nemotron-H {name} width overflowed usize")))?,
    )
}

struct Declaration<'a> {
    args: &'a ModelArgs,
    formats: BTreeMap<String, LinearFormatSpec>,
}

impl Declaration<'_> {
    fn linear(
        &mut self,
        prefix: &str,
        output: usize,
        input: usize,
        bias: bool,
        weight_sharding: S,
        bias_sharding: S,
    ) -> Result<Vec<ParameterMemberSpec>, ParallelPlanError> {
        let name = format!("{prefix}.weight");
        self.formats.insert(
            name.clone(),
            crate::linear_format::standard_linear_format(
                &name,
                self.args.weight_quantization_for(&name).into(),
            )
            .map_err(invalid)?,
        );
        let mut members = vec![ParameterMemberSpec::new(
            name,
            vec![output, input],
            weight_sharding,
        )];
        if bias {
            members.push(ParameterMemberSpec::new(
                format!("{prefix}.bias"),
                vec![output],
                bias_sharding,
            ));
        }
        Ok(members)
    }

    fn norm(&self, name: &str, parameter: &str) -> Result<ParameterGroupSpec, ParallelPlanError> {
        ParameterGroupSpec::new(
            name,
            ParameterRole::Replicated,
            [ParameterMemberSpec::new(
                parameter,
                vec![self.args.hidden_size as usize],
                S::Replicated,
            )],
        )
    }

    fn dense(
        &mut self,
        prefix: &str,
        logical_name: &str,
        width: usize,
        role: ParameterRole,
    ) -> Result<ParameterGroupSpec, ParallelPlanError> {
        let hidden = self.args.hidden_size as usize;
        let mut members = self.linear(
            &format!("{prefix}.up_proj"),
            width,
            hidden,
            self.args.mlp_bias,
            S::Partitioned { axis: 0 },
            S::Partitioned { axis: 0 },
        )?;
        members.extend(self.linear(
            &format!("{prefix}.down_proj"),
            hidden,
            width,
            self.args.mlp_bias,
            S::Partitioned { axis: 1 },
            S::Replicated,
        )?);
        ParameterGroupSpec::partitioned(logical_name, role, width, members)
    }

    fn block(
        &mut self,
        root: &str,
        policy: LayerPolicy,
        identity: usize,
        prediction: bool,
    ) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
        let args = self.args;
        let hidden = args.hidden_size as usize;
        let logical = if prediction {
            format!("{root}.mixer")
        } else {
            root.into()
        };
        let mut groups =
            vec![self.norm(&format!("{logical}.norm"), &format!("{root}.norm.weight"))?];
        match policy {
            LayerPolicy::Mamba => {
                let prefix = format!("{root}.mamba");
                let heads = args.mamba_num_heads as usize;
                let intermediate =
                    product("Mamba intermediate", heads, args.mamba_head_dim as usize)?;
                let grouped = product(
                    "Mamba state",
                    args.n_groups as usize,
                    args.ssm_state_size as usize,
                )?;
                let conv = sum(
                    "Mamba convolution",
                    intermediate,
                    product("Mamba state projections", 2, grouped)?,
                )?;
                let projection = sum(
                    "Mamba input projection",
                    sum("Mamba input projection", intermediate, conv)?,
                    heads,
                )?;
                let segments = vec![
                    0..intermediate,
                    intermediate..2 * intermediate,
                    2 * intermediate..2 * intermediate + grouped,
                    2 * intermediate + grouped..2 * intermediate + 2 * grouped,
                    2 * intermediate + 2 * grouped..projection,
                ];
                let mut members = self.linear(
                    &format!("{prefix}.in_proj"),
                    projection,
                    hidden,
                    args.use_bias,
                    S::PartitionedSegments {
                        axis: 0,
                        segments: segments.clone(),
                    },
                    S::PartitionedSegments { axis: 0, segments },
                )?;
                let sharding = S::PartitionedSegments {
                    axis: 0,
                    segments: vec![
                        0..intermediate,
                        intermediate..intermediate + grouped,
                        intermediate + grouped..conv,
                    ],
                };
                members.push(ParameterMemberSpec::new(
                    format!("{prefix}.conv1d.weight"),
                    vec![conv, 1, args.conv_kernel as usize],
                    sharding.clone(),
                ));
                if args.use_conv_bias {
                    members.push(ParameterMemberSpec::new(
                        format!("{prefix}.conv1d.bias"),
                        vec![conv],
                        sharding,
                    ));
                }
                for (field, width) in [
                    ("dt_bias", heads),
                    ("A_log", heads),
                    ("D", heads),
                    ("norm.weight", intermediate),
                ] {
                    members.push(ParameterMemberSpec::new(
                        format!("{prefix}.{field}"),
                        vec![width],
                        S::Partitioned { axis: 0 },
                    ));
                }
                members.extend(self.linear(
                    &format!("{prefix}.out_proj"),
                    hidden,
                    intermediate,
                    args.use_bias,
                    S::Partitioned { axis: 1 },
                    S::Replicated,
                )?);
                groups.push(ParameterGroupSpec::partitioned(
                    format!("{logical}.mamba.heads"),
                    ParameterRole::Channels,
                    args.n_groups as usize,
                    members,
                )?);
            }
            LayerPolicy::SelfAttention(_) => {
                let prefix = if prediction {
                    format!("{root}.mixer")
                } else {
                    format!("{root}.attention")
                };
                let query = product(
                    "attention query",
                    args.num_attention_heads as usize,
                    args.head_dim as usize,
                )?;
                let kv = product(
                    "attention KV",
                    args.num_key_value_heads as usize,
                    args.head_dim as usize,
                )?;
                let mut members = Vec::new();
                for (field, width) in [("q_proj", query), ("k_proj", kv), ("v_proj", kv)] {
                    members.extend(self.linear(
                        &format!("{prefix}.{field}"),
                        width,
                        hidden,
                        args.attention_bias,
                        S::Partitioned { axis: 0 },
                        S::Partitioned { axis: 0 },
                    )?);
                }
                members.extend(self.linear(
                    &format!("{prefix}.o_proj"),
                    hidden,
                    query,
                    args.attention_bias,
                    S::Partitioned { axis: 1 },
                    S::Replicated,
                )?);
                groups.push(ParameterGroupSpec::partitioned(
                    format!("{logical}.attention.heads"),
                    ParameterRole::AttentionHeads,
                    args.num_key_value_heads as usize,
                    members,
                )?);
            }
            LayerPolicy::DenseMlp => groups.push(self.dense(
                &format!("{root}.mlp"),
                &format!("{root}.mlp.intermediate"),
                args.intermediate_size as usize,
                ParameterRole::FeedForwardIntermediate,
            )?),
            LayerPolicy::SparseMoe => {
                let prefix = if prediction {
                    format!("{root}.mixer")
                } else {
                    format!("{root}.moe")
                };
                let selector = super::mlp::selector_spec(args, &prefix).map_err(invalid)?;
                self.formats.insert(
                    selector.weight().id.as_str().into(),
                    selector.format().clone(),
                );
                groups.push(eredu_runtime::top_k_group_selector_logical_group(
                    format!("{logical}.moe.gate"),
                    &selector,
                )?);
                let bank = super::expert_bank_spec(args, identity).map_err(invalid)?;
                for projection in [bank.up(), bank.down()] {
                    self.formats.insert(
                        projection.weight().id.as_str().into(),
                        projection.format().clone(),
                    );
                }
                groups.push(eredu_runtime::packed_relu2_logical_group(
                    format!("{logical}.moe.experts.intermediate"),
                    &bank,
                    args.moe_intermediate_size as usize,
                )?);
                groups.push(self.dense(
                    &format!("{prefix}.shared_experts"),
                    &format!("{logical}.moe.shared_experts.intermediate"),
                    args.moe_shared_expert_intermediate_size as usize,
                    ParameterRole::ExpertIntermediate,
                )?);
            }
        }
        Ok(groups)
    }
}

/// Declares complete canonical target and patterned prediction topology without native resources.
pub fn parameter_description(
    args: &ModelArgs,
) -> Result<ArchitectureParameterDescription, ParallelPlanError> {
    args.validate().map_err(invalid)?;
    let layers = args.num_hidden_layers as usize;
    let steps = args.num_nextn_predict_layers as usize;
    let prediction = args.mtp_policies().map_err(invalid)?;
    let pattern = if steps == 0 {
        0
    } else {
        prediction.len() / steps
    };
    let graph = crate::decoder::SequentialPredictionGroups::new_pattern(
        "model.layers",
        layers,
        "model.mtp.layers",
        steps,
        pattern,
    )
    .map_err(invalid)?
    .execution_graph()
    .map_err(invalid)?;
    let layout = ExecutionUnitLayout::new(
        &graph,
        std::iter::once(layers).chain(std::iter::repeat_n(pattern, steps)),
    )
    .map_err(invalid)?;
    let mut declaration = Declaration {
        args,
        formats: BTreeMap::new(),
    };
    let mut roles = vec!["embedding", "mtp"];
    if args.tie_word_embeddings {
        roles.push("output");
    }
    let mut owned = vec![
        OwnedParameterGroupSpec::new(
            ParameterGroupOwner::static_any_of(roles),
            ParameterGroupSpec::new(
                "model.embeddings",
                ParameterRole::Vocabulary,
                declaration.linear(
                    "model.embeddings",
                    args.vocab_size as usize,
                    args.hidden_size as usize,
                    false,
                    S::Balanced { axis: 0 },
                    S::Replicated,
                )?,
            )?,
        ),
        OwnedParameterGroupSpec::new(
            ParameterGroupOwner::static_role("norm"),
            declaration.norm("model.norm_f", "model.norm_f.weight")?,
        ),
    ];
    if !args.tie_word_embeddings {
        owned.push(OwnedParameterGroupSpec::new(
            ParameterGroupOwner::static_role("output"),
            ParameterGroupSpec::new(
                "lm_head",
                ParameterRole::Vocabulary,
                declaration.linear(
                    "lm_head",
                    args.vocab_size as usize,
                    args.hidden_size as usize,
                    false,
                    S::Balanced { axis: 0 },
                    S::Replicated,
                )?,
            )?,
        ));
    }
    for (group, count) in std::iter::once(layers)
        .chain(std::iter::repeat_n(pattern, steps))
        .enumerate()
    {
        for index in 0..count {
            let owner = ParameterGroupOwner::execution_unit(
                layout.group_id(group).expect("declared group").clone(),
                index,
            );
            let groups = if group == 0 {
                declaration.block(
                    &format!("model.layers.{index}"),
                    *args
                        .layer_schedule
                        .get(index)
                        .expect("declared target unit"),
                    index,
                    false,
                )?
            } else {
                let physical = (group - 1) * pattern + index;
                let root = format!("model.mtp.layers.{physical}");
                let mut groups = Vec::new();
                if index == 0 {
                    for field in ["enorm", "hnorm"] {
                        groups.push(declaration.norm(
                            &format!("{root}.{field}"),
                            &format!("{root}.{field}.weight"),
                        )?);
                    }
                    let input = product("prediction fusion", args.hidden_size as usize, 2)?;
                    groups.push(ParameterGroupSpec::new(
                        format!("{root}.eh_proj"),
                        ParameterRole::Replicated,
                        declaration.linear(
                            &format!("{root}.eh_proj"),
                            args.hidden_size as usize,
                            input,
                            false,
                            S::Replicated,
                            S::Replicated,
                        )?,
                    )?);
                }
                groups.extend(declaration.block(
                    &root,
                    prediction[physical],
                    layers + physical,
                    true,
                )?);
                if index + 1 == pattern {
                    groups.push(declaration.norm(
                        &format!("{root}.final_layernorm"),
                        &format!("{root}.final_layernorm.weight"),
                    )?);
                }
                groups
            };
            owned.extend(
                groups
                    .into_iter()
                    .map(|group| OwnedParameterGroupSpec::new(owner.clone(), group)),
            );
        }
    }
    let mut expanded = Vec::with_capacity(owned.len());
    for group in owned {
        let groups = eredu_runtime::expand_linear_format_parameter_groups(
            vec![group.group().clone()],
            |member| Ok(declaration.formats.get(member.target()).cloned()),
        )?;
        expanded.extend(
            groups
                .into_iter()
                .map(|expanded| OwnedParameterGroupSpec::new(group.owner().clone(), expanded)),
        );
    }
    ArchitectureParameterDescription::new(
        &graph,
        &layout,
        expanded
            .iter()
            .map(|g| g.group().clone())
            .collect::<Vec<_>>(),
        expanded,
    )
    .map_err(invalid)
}
