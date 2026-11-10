//! Backend-free canonical parameter topology for target and prediction units.

use eredu_nn::{LinearSpec, LogicalParameterRole};
use eredu_runtime::{
    ArchitectureParameterDescription, ExecutionUnitLayout, MemberSharding as S,
    OwnedParameterGroupSpec, ParallelPlanError, ParameterGroupOwner, ParameterGroupSpec,
    ParameterRole,
};

use super::ModelArgs;

fn invalid(error: impl std::fmt::Display) -> ParallelPlanError {
    ParallelPlanError::InvalidGroup(error.to_string())
}

struct Declaration;

impl Declaration {
    fn projections(
        &self,
        name: &str,
        role: ParameterRole,
        units: usize,
        specs: &(impl eredu_nn::DeclareLogicalParameters + ?Sized),
        row: &LinearSpec,
    ) -> Result<ParameterGroupSpec, ParallelPlanError> {
        eredu_runtime::logical_parameter_group(name, role, Some(units), specs, |d| {
            if d.parameter.id == row.weight.id {
                S::Partitioned { axis: 1 }
            } else if row
                .bias
                .as_ref()
                .is_some_and(|bias| bias.id == d.parameter.id)
            {
                S::Replicated
            } else {
                S::Partitioned { axis: 0 }
            }
        })
    }

    fn block(
        &self,
        root: &str,
        block: &super::declarations::TargetBlockParameters,
        prediction: bool,
    ) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
        let logical = if prediction {
            format!("{root}.mixer")
        } else {
            root.into()
        };
        let mut groups = vec![eredu_runtime::logical_parameter_group(
            format!("{logical}.norm"),
            ParameterRole::Replicated,
            None,
            &block.norm,
            |_| S::Replicated,
        )?];
        use super::declarations::TargetOperatorParameters;
        match &block.operator {
            TargetOperatorParameters::Mamba(spec) => {
                groups.push(eredu_runtime::logical_parameter_group(
                    format!("{logical}.mamba.heads"),
                    ParameterRole::Channels,
                    Some(spec.geometry.groups() as usize),
                    spec,
                    |declaration| {
                        let id = &declaration.parameter.id;
                        if id == &spec.in_proj.weight.id
                            || spec
                                .in_proj
                                .bias
                                .as_ref()
                                .is_some_and(|bias| &bias.id == id)
                        {
                            S::PartitionedSegments {
                                axis: 0,
                                segments: spec.input_segments(),
                            }
                        } else if id == &spec.convolution.weight.id
                            || spec
                                .convolution
                                .bias
                                .as_ref()
                                .is_some_and(|bias| &bias.id == id)
                        {
                            S::PartitionedSegments {
                                axis: 0,
                                segments: spec.convolution_segments(),
                            }
                        } else if id == &spec.out_proj.weight.id {
                            S::Partitioned { axis: 1 }
                        } else if spec
                            .out_proj
                            .bias
                            .as_ref()
                            .is_some_and(|bias| &bias.id == id)
                        {
                            S::Replicated
                        } else {
                            S::Partitioned { axis: 0 }
                        }
                    },
                )?);
            }
            TargetOperatorParameters::Attention(spec) => {
                groups.push(self.projections(
                    &format!("{logical}.attention.heads"),
                    ParameterRole::AttentionHeads,
                    spec.projection.geometry.key_value_heads() as usize,
                    spec,
                    &spec.projection.output,
                )?);
            }
            TargetOperatorParameters::Dense(spec) => groups.push(self.projections(
                &format!("{logical}.mlp.intermediate"),
                ParameterRole::FeedForwardIntermediate,
                spec.network.up.output as usize,
                spec,
                &spec.network.down,
            )?),
            TargetOperatorParameters::Sparse(spec) => {
                let selector = &spec.selector;
                let bank = &spec.experts;
                groups.push(eredu_runtime::logical_parameter_group(
                    format!("{logical}.moe.gate"),
                    ParameterRole::Replicated,
                    None,
                    selector,
                    |_| S::Replicated,
                )?);
                groups.push(eredu_runtime::logical_parameter_group(
                    format!("{logical}.moe.experts.intermediate"),
                    ParameterRole::ExpertIntermediate,
                    Some(bank.intermediate_dimensions() as usize),
                    bank,
                    |d| {
                        if d.parameter.id == bank.up().weight().id
                            || bank
                                .up()
                                .bias()
                                .is_some_and(|bias| bias.id == d.parameter.id)
                        {
                            S::Partitioned { axis: 1 }
                        } else if d.role == LogicalParameterRole::LinearWeight {
                            S::Partitioned { axis: 2 }
                        } else {
                            S::Replicated
                        }
                    },
                )?);
                groups.push(self.projections(
                    &format!("{logical}.moe.shared_experts.intermediate"),
                    ParameterRole::ExpertIntermediate,
                    spec.shared.network.up.output as usize,
                    &spec.shared,
                    &spec.shared.network.down,
                )?);
            }
        }
        Ok(groups)
    }
}

/// Declares the invocation graph, physical unit paths and flat-address projection once.
pub(super) fn execution_schedule(
    args: &ModelArgs,
) -> Result<
    (
        crate::decoder::SequentialPredictionGroups,
        eredu_runtime::ExecutionGraph,
        ExecutionUnitLayout,
    ),
    eredu_nn::Error,
> {
    args.validate().map_err(eredu_nn::Error::backend_source)?;
    let layers = usize::try_from(args.num_hidden_layers).map_err(eredu_nn::Error::backend)?;
    let steps = usize::try_from(args.num_nextn_predict_layers).map_err(eredu_nn::Error::backend)?;
    let prediction = args
        .mtp_policies()
        .map_err(eredu_nn::Error::backend_source)?;
    let pattern = if steps == 0 {
        0
    } else {
        prediction
            .len()
            .checked_div(steps)
            .filter(|pattern| *pattern > 0)
            .ok_or_else(|| {
                eredu_nn::Error::backend("Nemotron-H MTP operator pattern cannot be empty")
            })?
    };
    let schedule = crate::decoder::SequentialPredictionGroups::new_pattern(
        "model.layers",
        layers,
        "model.mtp.layers",
        steps,
        pattern,
    )?;
    let graph = schedule.execution_graph()?;
    let layout = ExecutionUnitLayout::new(
        &graph,
        (0..=steps)
            .map(|group| schedule.unit_count(group))
            .collect::<Result<Vec<_>, _>>()?,
    )
    .map_err(eredu_nn::Error::backend_source)?;
    Ok((schedule, graph, layout))
}

pub(super) fn parameter_description_with_scopes(
    args: &ModelArgs,
    statics: &crate::decoder::StaticModuleConstruction,
    blocks: &[super::declarations::TargetBlockParameters],
    predictions: &[super::mtp::PredictionUnitParameters],
) -> Result<ArchitectureParameterDescription, ParallelPlanError> {
    let (schedule, graph, layout) = execution_schedule(args).map_err(invalid)?;
    let declaration = Declaration;
    let mut roles = vec!["embedding", "mtp"];
    if args.tie_word_embeddings {
        roles.push("output");
    }
    let mut owned = statics
        .pinned_parameter_groups(
            crate::decoder::ParameterEncoding::Stored,
            "model.embeddings",
            "model.norm_f",
            "lm_head",
        )?
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
    for group in 0..layout.group_count() {
        let count = schedule.unit_count(group).map_err(invalid)?;
        for index in 0..count {
            let owner = ParameterGroupOwner::execution_unit(
                layout.group_id(group).expect("declared group").clone(),
                index,
            );
            let root = schedule.unit_path(group, index).map_err(invalid)?;

            let groups = if group == 0 {
                declaration.block(
                    &root,
                    blocks
                        .get(index)
                        .ok_or_else(|| invalid("target scope omits a block"))?,
                    false,
                )?
            } else {
                let physical = layout
                    .ordinal(group, index)
                    .and_then(|ordinal| ordinal.checked_sub(args.num_hidden_layers as usize))
                    .ok_or_else(|| invalid("unresolved prediction invocation"))?;
                let parameters = predictions
                    .get(physical)
                    .ok_or_else(|| invalid("prediction scope omits an invocation"))?;
                let replicated = |spec: &dyn eredu_nn::DeclareLogicalParameters| {
                    let declarations = spec.logical_parameters().map_err(invalid)?;
                    let name = declarations
                        .first()
                        .ok_or_else(|| invalid("empty prediction boundary"))?
                        .parameter
                        .id
                        .as_str()
                        .strip_suffix(".weight")
                        .ok_or_else(|| invalid("prediction boundary has no weight identity"))?;
                    eredu_runtime::logical_parameter_group(
                        name,
                        ParameterRole::Replicated,
                        None,
                        spec,
                        |_| S::Replicated,
                    )
                };
                let mut groups = Vec::new();
                for norm in parameters
                    .embedding_norm
                    .iter()
                    .chain(parameters.hidden_norm.iter())
                {
                    groups.push(replicated(norm)?);
                }
                if let Some(fusion) = &parameters.fusion {
                    groups.push(replicated(fusion)?);
                }
                groups.extend(declaration.block(&root, &parameters.block, true)?);
                if let Some(norm) = &parameters.final_norm {
                    groups.push(replicated(norm)?);
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
    ArchitectureParameterDescription::from_owned(&graph, &layout, owned).map_err(invalid)
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
pub(super) fn select_definition(
    declaration: &super::Declaration,
    definition: std::sync::Arc<eredu_runtime::ComponentDefinition>,
    state: &eredu_runtime::StateLayout,
    vocabulary: (
        eredu_nn::VocabularyParallelRange,
        Option<eredu_nn::VocabularyParallelRange>,
    ),
    units: impl IntoIterator<Item = (usize, super::LayerGeometry)>,
) -> Result<std::sync::Arc<eredu_runtime::ComponentDefinition>, eredu_nn::Error> {
    validate_identity(&definition)?;
    definition.validate_parameter_topology(declaration.definition().parameters())?;
    let global = declaration.definition().state_layout();
    if definition.state_layout() != global && definition.state_layout() != state {
        return Err(invalid_definition(
            &definition,
            "Nemotron-H retained state differs from global or selected geometry",
        ));
    }
    if let Some((_, layout)) = definition.parameters().partition_layout() {
        let retained =
            super::local_geometry(declaration, layout).map_err(eredu_nn::Error::backend_source)?;
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
    definition
        .select_state_layout(state)
        .map_err(eredu_nn::Error::from)
}
