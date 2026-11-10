//! Canonical hybrid target and prediction parameter declarations without native resources.

use eredu_nn::{
    DeclareLogicalParameters, GatedProductGroupLayout, LinearFormatSpec, LogicalParameterRole,
};
use eredu_runtime::{
    ArchitectureParameterDescription, ExecutionUnitLayout, MemberSharding as S,
    OwnedParameterGroupSpec, ParallelPlanError, ParameterGroupOwner, ParameterGroupSpec,
    ParameterRole,
};

use super::{HybridConfig, HybridLayerPolicy};

fn invalid(error: impl std::fmt::Display) -> ParallelPlanError {
    ParallelPlanError::InvalidGroup(error.to_string())
}

/// Owned comparison projection of construction identities, geometry and encodings.
#[derive(Debug, PartialEq)]
struct ConstructionParameterGeometry {
    parameter: eredu_nn::ParameterSpec,
    shape: Vec<usize>,
    role: LogicalParameterRole,
    format: Option<LinearFormatSpec>,
}
#[derive(Debug, PartialEq)]
pub(super) struct UnitParameterGeometry {
    groups: Vec<ParameterGroupSpec>,
    parameters: Vec<ConstructionParameterGeometry>,
}
struct GroupProjection<'a> {
    schedule: &'a crate::decoder::SequentialPredictionGroups,
    encoding: crate::decoder::ParameterEncoding,
    geometry: Option<&'a mut Vec<ConstructionParameterGeometry>>,
}
impl GroupProjection<'_> {
    fn group(
        &mut self,
        name: impl Into<String>,
        role: ParameterRole,
        units: Option<usize>,
        spec: &(impl DeclareLogicalParameters + ?Sized),
        sharding: impl FnMut(&eredu_nn::LogicalParameterDeclaration<'_>) -> S,
    ) -> Result<ParameterGroupSpec, ParallelPlanError> {
        if let Some(geometry) = &mut self.geometry {
            geometry.extend(
                spec.logical_parameters()
                    .map_err(invalid)?
                    .into_iter()
                    .map(|d| ConstructionParameterGeometry {
                        parameter: d.parameter.clone(),
                        shape: d.shape,
                        role: d.role,
                        format: d.linear_format.cloned(),
                    }),
            );
        }
        match self.encoding {
            crate::decoder::ParameterEncoding::Stored => {
                eredu_runtime::logical_parameter_group(name, role, units, spec, sharding)
            }
            crate::decoder::ParameterEncoding::Logical => {
                let members = eredu_runtime::logical_parameter_members(spec, sharding)?;
                match units {
                    Some(units) => ParameterGroupSpec::partitioned(name, role, units, members),
                    None => ParameterGroupSpec::new(name, role, members),
                }
            }
        }
    }
    fn mlp(
        &mut self,
        spec: &eredu_nn::SplitGatedProductSpec,
        logical: &str,
        role: ParameterRole,
    ) -> Result<ParameterGroupSpec, ParallelPlanError> {
        let down = &spec.down;
        self.group(logical, role, Some(spec.gate.output as usize), spec, |d| {
            if d.parameter.id == down.weight.id {
                S::Partitioned { axis: 1 }
            } else if d.role == LogicalParameterRole::LinearWeight {
                S::Partitioned { axis: 0 }
            } else {
                S::Replicated
            }
        })
    }

    fn unit(
        &mut self,
        config: &HybridConfig,
        group: usize,
        index: usize,
    ) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
        let path = self.schedule.unit_path(group, index).map_err(invalid)?;
        let policy = if group == 0 {
            config
                .layer_schedule
                .get(index)
                .copied()
                .ok_or_else(|| invalid(format!("Qwen hybrid has no target unit {index}")))?
        } else {
            HybridLayerPolicy::SelfAttention(eredu_core::attention::AttentionPolicy::Full)
        };
        self.block(config, &path, policy)
    }

    fn block(
        &mut self,
        config: &HybridConfig,
        root: &str,
        policy: HybridLayerPolicy,
    ) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
        let scope = super::BlockParameters::at(config, root, policy).map_err(invalid)?;
        let mut groups = vec![
            self.group(
                format!("{root}.input_norm"),
                ParameterRole::Replicated,
                None,
                &scope.norms[0],
                |_| S::Replicated,
            )?,
            self.group(
                format!("{root}.post_attention_norm"),
                ParameterRole::Replicated,
                None,
                &scope.norms[1],
                |_| S::Replicated,
            )?,
        ];
        match &scope.mixer {
            super::specs::MixerParameters::Recurrent(spec) => {
                let segments = S::PartitionedSegments {
                    axis: 0,
                    segments: spec.geometry.qkv_ranges().map_err(invalid)?.to_vec(),
                };
                groups.push(self.group(
                    format!("{root}.linear_attn.heads"),
                    ParameterRole::Channels,
                    Some(spec.geometry.key_heads() as usize),
                    spec,
                    |declaration| {
                        let id = &declaration.parameter.id;
                        let sharding = if id == &spec.input_qkv.weight.id
                            || id == &spec.convolution.weight.id
                        {
                            segments.clone()
                        } else if id == &spec.output.weight.id {
                            S::Partitioned { axis: 1 }
                        } else if id == &spec.normalization_weight.id
                            || spec.output.bias.as_ref().is_some_and(|bias| &bias.id == id)
                        {
                            S::Replicated
                        } else {
                            S::Partitioned { axis: 0 }
                        };
                        sharding
                    },
                )?);
            }
            super::specs::MixerParameters::Attention(spec) => {
                let output = &spec.projection.output;
                groups.push(self.group(
                    format!("{root}.attention.heads"),
                    ParameterRole::AttentionHeads,
                    Some(spec.projection.geometry.key_value_heads() as usize),
                    spec,
                    |d| {
                        if d.parameter.id == output.weight.id {
                            S::Partitioned { axis: 1 }
                        } else if d.role == LogicalParameterRole::Normalization
                            || output
                                .bias
                                .as_ref()
                                .is_some_and(|bias| bias.id == d.parameter.id)
                        {
                            S::Replicated
                        } else {
                            S::Partitioned { axis: 0 }
                        }
                    },
                )?);
            }
        }
        if let super::specs::FeedForwardParameters::Routed(spec) = &scope.feed_forward {
            groups.push(self.group(
                format!("{root}.mlp.router"),
                ParameterRole::Replicated,
                None,
                &spec.routed.selector,
                |_| S::Replicated,
            )?);
            let GatedProductGroupLayout::Packed { gate_up, .. } = spec.routed.experts.layout()
            else {
                unreachable!("hybrid declares packed experts")
            };
            let intermediate = spec.routed.experts.intermediate_dimensions() as usize;
            groups.push(self.group(
                format!("{root}.mlp.experts.intermediate"),
                ParameterRole::ExpertIntermediate,
                Some(intermediate),
                &spec.routed.experts,
                |d| {
                    if d.parameter.id == gate_up.weight().id
                        || gate_up.bias().is_some_and(|bias| bias.id == d.parameter.id)
                    {
                        S::PartitionedSegments {
                            axis: 1,
                            segments: vec![0..intermediate, intermediate..2 * intermediate],
                        }
                    } else if d.role == LogicalParameterRole::LinearWeight {
                        S::Partitioned { axis: 2 }
                    } else {
                        S::Replicated
                    }
                },
            )?);
            groups.push(self.mlp(
                &spec.shared,
                &format!("{root}.mlp.shared.intermediate"),
                ParameterRole::ExpertIntermediate,
            )?);
            groups.push(self.group(
                format!("{root}.mlp.shared_gate"),
                ParameterRole::Replicated,
                None,
                &spec.shared_gate,
                |_| S::Replicated,
            )?);
        } else if let super::specs::FeedForwardParameters::Dense(spec) = &scope.feed_forward {
            groups.push(self.mlp(
                spec,
                &format!("{root}.mlp.intermediate"),
                ParameterRole::FeedForwardIntermediate,
            )?);
        }
        Ok(groups)
    }
}

/// Declares target and prediction invocation identities independently of storage and state.
pub(super) fn execution_schedule(
    config: &HybridConfig,
) -> Result<
    (
        crate::decoder::SequentialPredictionGroups,
        eredu_runtime::ExecutionGraph,
        ExecutionUnitLayout,
    ),
    eredu_nn::Error,
> {
    let targets = usize::try_from(config.num_hidden_layers).map_err(eredu_nn::Error::backend)?;
    let steps = usize::try_from(config.mtp_num_hidden_layers).map_err(eredu_nn::Error::backend)?;
    let schedule = crate::decoder::SequentialPredictionGroups::new_pattern(
        "model.layers",
        targets,
        "mtp.layers",
        steps,
        1,
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

/// Declares global checkpoint geometry and canonical ownership before native materialization.
pub fn parameter_description(
    config: &HybridConfig,
    encoding: crate::decoder::ParameterEncoding,
) -> Result<ArchitectureParameterDescription, ParallelPlanError> {
    config.validate().map_err(invalid)?;
    let (schedule, graph, layout) = execution_schedule(config).map_err(invalid)?;
    let steps = layout.group_count() - 1;
    let mut declaration = GroupProjection {
        schedule: &schedule,
        encoding,
        geometry: None,
    };
    let statics = config.statics();
    let mut roles = vec!["embedding", "mtp"];
    if config.tie_word_embeddings {
        roles.push("output");
    }
    let pinned =
        statics.pinned_parameter_groups(encoding, "model.embed_tokens", "model.norm", "lm_head")?;
    let mut owned = pinned
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
    if steps > 0 {
        let consumers = (1..layout.group_count())
            .map(|group| (layout.group_id(group).expect("declared group").clone(), 0))
            .collect::<Vec<_>>();
        let owner = ParameterGroupOwner::static_unit_consumers("mtp", consumers);
        let shared = config
            .prediction_shared()
            .ok_or_else(|| invalid("undeclared hybrid prediction shared scope"))?;
        for norm in shared.normalizations() {
            let parameters = norm.logical_parameters().map_err(invalid)?;
            let name = parameters[0]
                .parameter
                .id
                .as_str()
                .strip_suffix(".weight")
                .expect("prediction normalization weight");
            owned.push(OwnedParameterGroupSpec::new(
                owner.clone(),
                declaration.group(name, ParameterRole::Replicated, None, norm, |_| {
                    S::Replicated
                })?,
            ));
        }
        let fusion = &shared.fusion;
        let name = fusion
            .weight
            .id
            .as_str()
            .strip_suffix(".weight")
            .expect("prediction fusion weight");
        owned.push(OwnedParameterGroupSpec::new(
            owner,
            declaration.group(name, ParameterRole::Replicated, None, fusion, |_| {
                S::Replicated
            })?,
        ));
    }
    for group in 0..layout.group_count() {
        let count = schedule.unit_count(group).map_err(invalid)?;
        for index in 0..count {
            let owner = ParameterGroupOwner::execution_unit(
                layout.group_id(group).expect("declared group").clone(),
                index,
            );
            owned.extend(
                declaration
                    .unit(config, group, index)?
                    .into_iter()
                    .map(|group| OwnedParameterGroupSpec::new(owner.clone(), group)),
            );
        }
    }
    ArchitectureParameterDescription::from_owned(&graph, &layout, owned).map_err(invalid)
}

/// Declares one target or prediction unit from normalized global geometry.
pub fn unit_parallel_parameter_groups(
    config: &HybridConfig,
    group: usize,
    index: usize,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    config.validate().map_err(invalid)?;
    let (schedule, _, _) = execution_schedule(config).map_err(invalid)?;
    GroupProjection {
        schedule: &schedule,
        encoding: crate::decoder::ParameterEncoding::Stored,
        geometry: None,
    }
    .unit(config, group, index)
}

/// Compares retained logical construction and encoding facts without expanding
/// rank-local shapes into a new checkpoint-global sharding plan.
pub(super) fn unit_parameter_geometry(
    config: &HybridConfig,
    group: usize,
    index: usize,
) -> Result<UnitParameterGeometry, ParallelPlanError> {
    let (schedule, _, _) = execution_schedule(config).map_err(invalid)?;
    let mut parameters = Vec::new();
    let groups = GroupProjection {
        schedule: &schedule,
        encoding: crate::decoder::ParameterEncoding::Logical,
        geometry: Some(&mut parameters),
    }
    .unit(config, group, index)?;
    Ok(UnitParameterGeometry { groups, parameters })
}

/// Validates target and prediction invocation, storage, transport and state projections.
/// Parameter geometry remains checkpoint-global when mutable state is rank-local.
pub fn component_definition(
    config: &HybridConfig,
    geometry: Option<&super::LocalGeometry>,
) -> Result<eredu_runtime::ComponentDefinition, eredu_nn::Error> {
    let parameters = parameter_description(config, crate::decoder::ParameterEncoding::Stored)
        .map_err(eredu_nn::Error::backend)?;
    let state = component_state(config, geometry)?;
    let units = component_units(config, &parameters)?;
    let prediction = prediction_groups(&parameters);
    let transport = (0..parameters.graph().groups().len())
        .map(group_transport)
        .collect();
    let partition = component_state_partition(&parameters, &state);
    eredu_runtime::ComponentDefinition::new(
        "qwen-hybrid",
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

fn component_state(
    config: &HybridConfig,
    geometry: Option<&super::LocalGeometry>,
) -> Result<eredu_runtime::StateLayout, eredu_nn::Error> {
    match geometry {
        Some(geometry) => {
            geometry
                .validate_for(config)
                .map_err(eredu_nn::Error::backend)?;
            Ok(geometry.state_layout().clone())
        }
        None => super::state_layout(config).map_err(eredu_nn::Error::backend),
    }
}
fn component_units(
    config: &HybridConfig,
    parameters: &ArchitectureParameterDescription,
) -> Result<Vec<eredu_runtime::ComponentUnit>, eredu_nn::Error> {
    let (schedule, _, _) = execution_schedule(config)?;
    let layout = parameters.unit_layout();
    (0..layout.len())
        .map(|ordinal| {
            let address = layout.address(ordinal).expect("declared invocation");
            Ok(eredu_runtime::ComponentUnit {
                path: schedule.unit_path(address.group(), address.index())?,
                state: ordinal..ordinal + 1,
            })
        })
        .collect()
}

/// Ordered prediction groups of the text declaration; the target group is first.
pub(super) fn prediction_groups(parameters: &ArchitectureParameterDescription) -> Vec<String> {
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
fn component_state_partition(
    parameters: &ArchitectureParameterDescription,
    state: &eredu_runtime::StateLayout,
) -> eredu_runtime::ArchitectureStatePartitionPlan {
    let targets = parameters
        .unit_layout()
        .group_range(0)
        .expect("declared target group")
        .len();
    crate::transport::pipeline_with_output_state(0, targets, state)
}

/// Projects executable formats while retaining logical consumers, state and observation seams.
pub fn project_parameters(
    config: &HybridConfig,
    definition: &std::sync::Arc<eredu_runtime::ComponentDefinition>,
) -> Result<std::sync::Arc<eredu_runtime::ComponentDefinition>, eredu_nn::Error> {
    validate_text_identity(definition)?;
    let parameters = parameter_description(config, crate::decoder::ParameterEncoding::Stored)
        .map_err(eredu_nn::Error::backend)?;
    definition
        .project_parameters(parameters)
        .map_err(eredu_nn::Error::from)
}
fn validate_text_identity(
    definition: &eredu_runtime::ComponentDefinition,
) -> Result<(), eredu_nn::Error> {
    if definition.id() != "qwen-hybrid" {
        return Err(invalid_definition(
            definition,
            "Qwen hybrid component identity",
        ));
    }
    Ok(())
}
pub(crate) fn select_definition(
    config: &HybridConfig,
    definition: std::sync::Arc<eredu_runtime::ComponentDefinition>,
    geometry: Option<&super::LocalGeometry>,
) -> Result<std::sync::Arc<eredu_runtime::ComponentDefinition>, eredu_nn::Error> {
    validate_text_identity(&definition)?;
    definition.validate_parameter_topology(
        &parameter_description(config, crate::decoder::ParameterEncoding::Stored)
            .map_err(eredu_nn::Error::backend)?,
    )?;
    let state = component_state(config, geometry)?;
    let global = component_state(config, None)?;
    if definition.state_layout() != &global && definition.state_layout() != &state {
        return Err(invalid_definition(
            &definition,
            "Qwen hybrid retained state differs from global or selected geometry",
        ));
    }
    if let Some((_, layout)) = definition.parameters().partition_layout() {
        let retained =
            super::local_geometry(config, layout).map_err(eredu_nn::Error::backend_source)?;
        let full = eredu_nn::VocabularyParallelRange {
            global_vocabulary: config.vocab_size as usize,
            local: 0..config.vocab_size as usize,
        };
        let output = if config.tie_word_embeddings {
            None
        } else {
            Some(geometry.and_then(|g| g.output_range()).unwrap_or(&full))
        };
        if retained.state_layout() != &state
            || retained.embedding_range()
                != geometry.map_or(&full, super::LocalGeometry::embedding_range)
            || retained.output_range() != output
            || (0..config.num_hidden_layers as usize).any(|i| {
                !same_local_dimensions(
                    retained.target(i).expect("declared target"),
                    geometry.and_then(|g| g.target(i)).unwrap_or(config),
                )
            })
            || (0..config.mtp_num_hidden_layers as usize).any(|i| {
                !same_local_dimensions(
                    retained.prediction(i).expect("declared prediction"),
                    geometry.and_then(|g| g.prediction(i)).unwrap_or(config),
                )
            })
        {
            return Err(invalid_definition(
                &definition,
                "Qwen hybrid retained placement differs from selected geometry",
            ));
        }
    }
    let parameters = definition.parameters();
    definition.validate_execution_contract(
        crate::decoder::TARGET_EXECUTION_GROUP,
        &prediction_groups(parameters),
        &component_units(config, parameters)?,
        &(0..parameters.graph().groups().len())
            .map(group_transport)
            .collect::<Vec<_>>(),
        &component_state_partition(parameters, definition.state_layout()),
        |_, _, boundaries| !boundaries.is_kernel_owned(),
    )?;
    definition
        .select_state_layout(&state)
        .map_err(eredu_nn::Error::from)
}

pub(crate) fn conditional_execution_graph(
    parsed: &super::ParsedHybridConfig,
) -> Result<eredu_runtime::ExecutionGraph, eredu_nn::Error> {
    use eredu_runtime::ExecutionGroupSpec;
    parsed.text.validate().map_err(eredu_nn::Error::backend)?;
    parsed
        .vision
        .as_ref()
        .ok_or_else(|| eredu_nn::Error::backend("conditional Qwen requires vision config"))?
        .validate_for(crate::qwen::vision::VisionMode::WindowScheduled)
        .map_err(eredu_nn::Error::backend)?;
    if parsed
        .vision
        .as_ref()
        .is_some_and(|v| v.root() != "model.visual")
    {
        return Err(eredu_nn::Error::backend(
            "conditional vision namespace differs from component",
        ));
    }
    let (_, text, _) = execution_schedule(&parsed.text)?;
    let mut groups = vec![
        ExecutionGroupSpec::root(super::VISION_EXECUTION_GROUP),
        ExecutionGroupSpec::with_dependencies(
            text.groups()[0].id(),
            [super::VISION_EXECUTION_GROUP],
        ),
    ];
    groups.extend(text.groups().iter().skip(1).cloned());
    eredu_runtime::ExecutionGraph::new(groups, text.groups()[text.output()].id())
        .map_err(eredu_nn::Error::backend_source)
}

/// Declares canonical vision, target and prediction ownership without native modules.
pub fn conditional_parameter_description(
    parsed: &super::ParsedHybridConfig,
) -> Result<ArchitectureParameterDescription, eredu_nn::Error> {
    let graph = conditional_execution_graph(parsed)?;
    let vision = parsed.vision.as_ref().expect("validated vision config");
    let text = parameter_description(&parsed.text, crate::decoder::ParameterEncoding::Stored)
        .map_err(eredu_nn::Error::backend)?;
    let counts = std::iter::once(vision.layer_count()).chain(
        (0..text.unit_layout().group_count()).map(|group| {
            text.unit_layout()
                .group_range(group)
                .expect("declared text group")
                .len()
        }),
    );
    let layout = ExecutionUnitLayout::new(&graph, counts).map_err(eredu_nn::Error::backend)?;
    let mut owned = text.groups().to_vec();
    let group = eredu_runtime::ExecutionGroupId::new(super::VISION_EXECUTION_GROUP)
        .map_err(eredu_nn::Error::backend)?;
    owned.extend(
        crate::qwen::vision::owned_static_parallel_parameter_groups(
            vision,
            group.clone(),
            "vision",
        )
        .map_err(eredu_nn::Error::backend)?,
    );
    for layer in 0..vision.layer_count() {
        owned.extend(
            crate::qwen::vision::block_parallel_parameter_groups(vision, layer)
                .map_err(eredu_nn::Error::backend)?
                .into_iter()
                .map(|group_spec| {
                    OwnedParameterGroupSpec::new(
                        ParameterGroupOwner::execution_unit(group.clone(), layer),
                        group_spec,
                    )
                }),
        );
    }
    ArchitectureParameterDescription::from_owned(&graph, &layout, owned)
        .map_err(eredu_nn::Error::backend)
}

/// Validates conditional invocation, transport and text/MTP state ownership before materialization.
pub fn conditional_component_definition(
    parsed: &super::ParsedHybridConfig,
    geometry: Option<&super::ConditionalLocalGeometry>,
) -> Result<eredu_runtime::ComponentDefinition, eredu_nn::Error> {
    let parameters = conditional_parameter_description(parsed)?;
    let state = conditional_state(parsed, geometry)?;
    let units = conditional_units(parsed, &parameters)?;
    let prediction = conditional_prediction_groups(&parameters);
    let transport = (0..parameters.graph().groups().len())
        .map(conditional_group_transport)
        .collect();
    let partition = conditional_state_partition(&parameters, &state);
    eredu_runtime::ComponentDefinition::new(
        "qwen-hybrid-conditional",
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

fn conditional_state(
    parsed: &super::ParsedHybridConfig,
    geometry: Option<&super::ConditionalLocalGeometry>,
) -> Result<eredu_runtime::StateLayout, eredu_nn::Error> {
    match geometry {
        Some(geometry) => {
            geometry
                .validate_for(parsed)
                .map_err(eredu_nn::Error::backend)?;
            Ok(geometry.state_layout().clone())
        }
        None => super::state_layout(&parsed.text).map_err(eredu_nn::Error::backend),
    }
}

fn conditional_units(
    parsed: &super::ParsedHybridConfig,
    parameters: &ArchitectureParameterDescription,
) -> Result<Vec<eredu_runtime::ComponentUnit>, eredu_nn::Error> {
    let (schedule, _, _) = execution_schedule(&parsed.text)?;
    let layout = parameters.unit_layout();
    let vision_units = layout.group_range(0).expect("declared vision group").len();
    (0..layout.len())
        .map(|ordinal| {
            let address = layout.address(ordinal).expect("declared invocation");
            let (path, state) = if address.group() == 0 {
                (format!("model.visual.blocks.{}", address.index()), 0..0)
            } else {
                let slot = ordinal - vision_units;
                (
                    schedule.unit_path(address.group() - 1, address.index())?,
                    slot..slot + 1,
                )
            };
            Ok(eredu_runtime::ComponentUnit { path, state })
        })
        .collect()
}

fn conditional_prediction_groups(parameters: &ArchitectureParameterDescription) -> Vec<String> {
    let layout = parameters.unit_layout();
    let groups = parameters.graph().groups().len();
    (2..groups)
        .map(|group| {
            layout
                .group_id(group)
                .expect("declared prediction group")
                .as_str()
                .to_owned()
        })
        .collect()
}

fn conditional_group_transport(group: usize) -> eredu_runtime::ArchitectureGroupTransport {
    match group {
        0 => eredu_runtime::ArchitectureGroupTransport {
            placement: eredu_runtime::ArchitectureGroupPlacement::Pipeline,
            kind: eredu_runtime::ArchitectureGroupKind::VisionEncoder,
            first_owner_static_roles: vec!["vision".into()],
            last_owner_static_roles: Vec::new(),
            merge_destination: eredu_runtime::ArchitectureMergeDestination::FirstPipelineOwner,
            parallel_subgroup: Some(eredu_runtime::ArchitectureParallelSubgroup::TensorSharded),
            request_optional: true,
        },
        1 => crate::transport::decoder(),
        group => {
            let mut transport = crate::transport::prediction();
            if group == 2 {
                transport.first_owner_static_roles.push("mtp".into());
            }
            transport
        }
    }
}
fn conditional_state_partition(
    parameters: &ArchitectureParameterDescription,
    state: &eredu_runtime::StateLayout,
) -> eredu_runtime::ArchitectureStatePartitionPlan {
    let targets = parameters
        .unit_layout()
        .group_range(1)
        .expect("declared target group")
        .len();
    crate::transport::pipeline_with_output_state(1, targets, state)
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
fn validate_conditional_identity(
    definition: &eredu_runtime::ComponentDefinition,
) -> Result<(), eredu_nn::Error> {
    if definition.id() != "qwen-hybrid-conditional" {
        return Err(invalid_definition(
            definition,
            "conditional Qwen component identity",
        ));
    }
    Ok(())
}

/// Projects exact conditional formats while retaining invocation, storage and state identities.
pub fn project_conditional_parameters(
    parsed: &super::ParsedHybridConfig,
    definition: &std::sync::Arc<eredu_runtime::ComponentDefinition>,
) -> Result<std::sync::Arc<eredu_runtime::ComponentDefinition>, eredu_nn::Error> {
    validate_conditional_identity(definition)?;
    let parameters = conditional_parameter_description(parsed)?;
    definition
        .project_parameters(parameters)
        .map_err(eredu_nn::Error::from)
}

fn same_local_dimensions(a: &super::HybridConfig, b: &super::HybridConfig) -> bool {
    a.num_attention_heads == b.num_attention_heads
        && a.num_key_value_heads == b.num_key_value_heads
        && a.linear_num_key_heads == b.linear_num_key_heads
        && a.linear_num_value_heads == b.linear_num_value_heads
        && a.intermediate_size == b.intermediate_size
        && a.moe_intermediate_size == b.moe_intermediate_size
        && a.shared_expert_intermediate_size == b.shared_expert_intermediate_size
}

pub(crate) fn select_conditional_definition(
    parsed: &super::ParsedHybridConfig,
    definition: std::sync::Arc<eredu_runtime::ComponentDefinition>,
    geometry: Option<&super::ConditionalLocalGeometry>,
) -> Result<std::sync::Arc<eredu_runtime::ComponentDefinition>, eredu_nn::Error> {
    validate_conditional_identity(&definition)?;
    definition.validate_parameter_topology(&conditional_parameter_description(parsed)?)?;
    let state = conditional_state(parsed, geometry)?;
    let global = conditional_state(parsed, None)?;
    if definition.state_layout() != &global && definition.state_layout() != &state {
        return Err(invalid_definition(
            &definition,
            "conditional Qwen retained state differs from global or selected geometry",
        ));
    }
    if let Some((_, layout)) = definition.parameters().partition_layout() {
        let retained = super::conditional_local_geometry(parsed, layout)
            .map_err(eredu_nn::Error::backend_source)?;
        let vision = parsed
            .vision
            .as_ref()
            .expect("validated conditional vision");
        let full = eredu_nn::VocabularyParallelRange {
            global_vocabulary: parsed.text.vocab_size as usize,
            local: 0..parsed.text.vocab_size as usize,
        };
        let output = if parsed.text.tie_word_embeddings {
            None
        } else {
            Some(
                geometry
                    .and_then(|g| g.text().output_range())
                    .unwrap_or(&full),
            )
        };
        let unsharded = crate::qwen::vision::VisionStaticSelection::all(vision.clone())
            .and_then(|s| s.unsharded_merger_intermediates())
            .map_err(eredu_nn::Error::backend_source)?;
        if retained.state_layout() != &state
            || retained.text().embedding_range()
                != geometry.map_or(&full, |g| g.text().embedding_range())
            || retained.text().output_range() != output
            || retained.merger_widths()
                != geometry.map_or(
                    unsharded.as_slice(),
                    super::ConditionalLocalGeometry::merger_widths,
                )
            || (0..vision.layer_count()).any(|i| {
                retained.vision_block(i)
                    != geometry.map_or(Some((vision.num_heads, vision.intermediate_size)), |g| {
                        g.vision_block(i)
                    })
            })
            || (0..parsed.text.num_hidden_layers as usize).any(|i| {
                !same_local_dimensions(
                    retained.text().target(i).expect("declared local target"),
                    geometry
                        .and_then(|g| g.text().target(i))
                        .unwrap_or(&parsed.text),
                )
            })
            || (0..parsed.text.mtp_num_hidden_layers as usize).any(|i| {
                !same_local_dimensions(
                    retained
                        .text()
                        .prediction(i)
                        .expect("declared local prediction"),
                    geometry
                        .and_then(|g| g.text().prediction(i))
                        .unwrap_or(&parsed.text),
                )
            })
        {
            return Err(invalid_definition(
                &definition,
                "conditional Qwen retained placement differs from selected geometry",
            ));
        }
    }
    let parameters = definition.parameters();
    definition.validate_execution_contract(
        crate::decoder::TARGET_EXECUTION_GROUP,
        &conditional_prediction_groups(parameters),
        &conditional_units(parsed, parameters)?,
        &(0..parameters.graph().groups().len())
            .map(conditional_group_transport)
            .collect::<Vec<_>>(),
        &conditional_state_partition(parameters, definition.state_layout()),
        |_, _, boundaries| !boundaries.is_kernel_owned(),
    )?;
    definition
        .select_state_layout(&state)
        .map_err(eredu_nn::Error::from)
}
