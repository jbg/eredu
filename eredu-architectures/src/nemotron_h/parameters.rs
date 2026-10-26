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
    let mut owned = super::static_spec(args, super::FinalNormalization::Native)
        .pinned_parameter_groups("model.embeddings", "model.norm_f", "lm_head")?
        .into_iter()
        .map(|(role, group)| {
            let owner = if role == "embedding" {
                ParameterGroupOwner::static_any_of(roles.clone())
            } else {
                ParameterGroupOwner::static_role(role)
            };
            OwnedParameterGroupSpec::new(owner, group)
        })
        .collect::<Vec<_>>();
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
    let expanded = eredu_runtime::expand_owned_linear_format_parameter_groups(owned, |member| {
        Ok(declaration.formats.get(member.target()).cloned())
    })?;
    ArchitectureParameterDescription::from_owned(&graph, &layout, expanded).map_err(invalid)
}

/// Validates parameter, invocation, transport and state views as one retained definition.
pub(crate) fn component_definition(
    parameters: ArchitectureParameterDescription,
    state: eredu_runtime::StateLayout,
) -> Result<eredu_runtime::ComponentDefinition, eredu_nn::Error> {
    let layout = parameters.unit_layout();
    let target = layout.group_range(0).expect("declared target group").len();
    let units = component_units(&parameters);
    let prediction = prediction_groups(&parameters);
    let transport = (0..parameters.graph().groups().len())
        .map(group_transport)
        .collect();
    let partition = crate::transport::pipeline_with_output_state(0, target, &state);
    eredu_runtime::ComponentDefinition::new(
        "nemotron-h",
        parameters,
        units,
        transport,
        crate::decoder::TARGET_EXECUTION_GROUP,
        prediction,
        state,
        partition,
    )
    .map_err(eredu_nn::Error::from)
}

fn component_units(
    parameters: &ArchitectureParameterDescription,
) -> Vec<eredu_runtime::ComponentUnit> {
    let layout = parameters.unit_layout();
    let target = layout.group_range(0).expect("declared target group").len();
    (0..layout.len())
        .map(|ordinal| {
            let address = layout.address(ordinal).expect("declared unit");
            let path = if address.group() == 0 {
                format!("model.layers.{}", address.index())
            } else {
                format!("model.mtp.layers.{}", ordinal - target)
            };
            eredu_runtime::ComponentUnit {
                path,
                state: ordinal..ordinal + 1,
            }
        })
        .collect()
}
fn prediction_groups(parameters: &ArchitectureParameterDescription) -> Vec<String> {
    let layout = parameters.unit_layout();
    let groups = parameters.graph().groups().len();
    (1..groups)
        .map(|group| {
            layout
                .group_id(group)
                .expect("declared prediction group")
                .as_str()
                .to_owned()
        })
        .collect()
}
fn group_transport(group: usize) -> eredu_runtime::ArchitectureGroupTransport {
    if group == 0 {
        return crate::transport::decoder();
    }
    let mut transport = crate::transport::prediction();
    if group == 1 {
        transport.first_owner_static_roles.push("mtp".into());
    }
    transport
}
fn invalid_definition(
    definition: &eredu_runtime::ComponentDefinition,
    resource: impl Into<String>,
) -> eredu_nn::Error {
    eredu_runtime::ComponentDefinitionError::Invalid {
        component: definition.id().into(),
        resource: resource.into(),
    }
    .into()
}
fn validate_identity(
    definition: &eredu_runtime::ComponentDefinition,
) -> Result<(), eredu_nn::Error> {
    if definition.id() != "nemotron-h" {
        return Err(invalid_definition(
            definition,
            "Nemotron-H component identity",
        ));
    }
    Ok(())
}
/// Projects exact executable formats while retaining logical consumers and observation seams.
pub fn project_parameters(
    args: &ModelArgs,
    definition: &std::sync::Arc<eredu_runtime::ComponentDefinition>,
) -> Result<std::sync::Arc<eredu_runtime::ComponentDefinition>, eredu_nn::Error> {
    validate_identity(definition)?;
    let parameters = parameter_description(args).map_err(eredu_nn::Error::backend_source)?;
    definition
        .project_parameters(parameters)
        .map_err(eredu_nn::Error::from)
}
pub(super) fn select_definition(
    args: &ModelArgs,
    definition: std::sync::Arc<eredu_runtime::ComponentDefinition>,
    state: &eredu_runtime::StateLayout,
    vocabulary: (
        eredu_nn::VocabularyParallelRange,
        Option<eredu_nn::VocabularyParallelRange>,
    ),
    units: impl IntoIterator<Item = (usize, super::LayerGeometry)>,
) -> Result<std::sync::Arc<eredu_runtime::ComponentDefinition>, eredu_nn::Error> {
    validate_identity(&definition)?;
    definition.validate_parameter_topology(
        &parameter_description(args).map_err(eredu_nn::Error::backend_source)?,
    )?;
    let global = super::state_layout(args).map_err(eredu_nn::Error::backend_source)?;
    if definition.state_layout() != &global && definition.state_layout() != state {
        return Err(invalid_definition(
            &definition,
            "Nemotron-H retained state differs from global or selected geometry",
        ));
    }
    if let Some((_, layout)) = definition.parameters().partition_layout() {
        let retained =
            super::local_geometry(args, layout).map_err(eredu_nn::Error::backend_source)?;
        if retained.state_layout() != state
            || retained.embedding_range() != &vocabulary.0
            || retained.output_range() != vocabulary.1.as_ref()
        {
            return Err(invalid_definition(
                &definition,
                "Nemotron-H retained state or vocabulary placement differs from selected geometry",
            ));
        }
        for (index, geometry) in units {
            let actual = if index < retained.target_units().len() {
                retained.target_unit(index)
            } else {
                retained.prediction_unit(index - retained.target_units().len())
            };
            if actual != Some(&geometry) {
                return Err(invalid_definition(
                    &definition,
                    format!(
                        "Nemotron-H retained unit {index} placement differs from selected geometry"
                    ),
                ));
            }
        }
    }
    let target = definition
        .parameters()
        .unit_layout()
        .group_range(0)
        .expect("declared target group")
        .len();
    definition.validate_execution_contract(
        crate::decoder::TARGET_EXECUTION_GROUP,
        &prediction_groups(definition.parameters()),
        &component_units(definition.parameters()),
        &(0..definition.parameters().graph().groups().len())
            .map(group_transport)
            .collect::<Vec<_>>(),
        &crate::transport::pipeline_with_output_state(0, target, definition.state_layout()),
        |_, _, boundaries| !boundaries.is_kernel_owned(),
    )?;
    if definition.state_layout() == state {
        Ok(definition)
    } else {
        definition
            .as_ref()
            .clone()
            .with_state_layout(state.clone())
            .map(std::sync::Arc::new)
            .map_err(eredu_nn::Error::from)
    }
}
