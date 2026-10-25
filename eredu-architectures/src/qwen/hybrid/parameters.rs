//! Canonical hybrid target and prediction parameter declarations without native resources.

use std::collections::BTreeMap;

use eredu_nn::{GatedProductGroupLayout, LinearFormatSpec, LinearSpec};
use eredu_runtime::{
    ArchitectureParameterDescription, ExecutionUnitLayout, MemberSharding as S,
    OwnedParameterGroupSpec, ParallelPlanError, ParameterGroupOwner, ParameterGroupSpec,
    ParameterMemberSpec, ParameterRole,
};

use super::{HybridConfig, HybridLayerPolicy};

fn invalid(error: impl std::fmt::Display) -> ParallelPlanError {
    ParallelPlanError::InvalidGroup(error.to_string())
}

#[derive(Default)]
struct Declaration {
    formats: BTreeMap<String, LinearFormatSpec>,
}

impl Declaration {
    fn linear(&mut self, spec: LinearSpec, weight: S, bias: S) -> Vec<ParameterMemberSpec> {
        self.formats
            .insert(spec.weight.id.as_str().into(), spec.format);
        let mut members = vec![ParameterMemberSpec::new(
            spec.weight.id.as_str(),
            vec![spec.output as usize, spec.input as usize],
            weight,
        )];
        if let Some(parameter) = spec.bias {
            members.push(ParameterMemberSpec::new(
                parameter.id.as_str(),
                vec![spec.output as usize],
                bias,
            ));
        }
        members
    }

    fn norm(
        &self,
        name: &str,
        parameter: &str,
        width: i32,
    ) -> Result<ParameterGroupSpec, ParallelPlanError> {
        ParameterGroupSpec::new(
            name,
            ParameterRole::Replicated,
            [ParameterMemberSpec::new(
                parameter,
                vec![width as usize],
                S::Replicated,
            )],
        )
    }

    fn mlp(
        &mut self,
        config: &HybridConfig,
        prefix: &str,
        logical: &str,
        width: i32,
        role: ParameterRole,
    ) -> Result<ParameterGroupSpec, ParallelPlanError> {
        let mut members = Vec::new();
        for (index, spec) in super::block::mlp_projection_specs(config, prefix, width)
            .map_err(invalid)?
            .into_iter()
            .enumerate()
        {
            members.extend(self.linear(
                spec,
                S::Partitioned {
                    axis: if index == 2 { 1 } else { 0 },
                },
                S::Replicated,
            ));
        }
        ParameterGroupSpec::partitioned(logical, role, width as usize, members)
    }

    fn unit(
        &mut self,
        config: &HybridConfig,
        group: usize,
        index: usize,
    ) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
        if group == 0 {
            let policy = config
                .layer_schedule
                .get(index)
                .copied()
                .ok_or_else(|| invalid(format!("Qwen hybrid has no target unit {index}")))?;
            self.block(config, &format!("model.layers.{index}"), policy, index)
        } else if index == 0 && group <= config.mtp_num_hidden_layers as usize {
            self.block(
                config,
                &format!("mtp.layers.{}", group - 1),
                HybridLayerPolicy::SelfAttention(eredu_core::attention::AttentionPolicy::Full),
                config.num_hidden_layers as usize + group - 1,
            )
        } else {
            Err(invalid(format!(
                "Qwen hybrid has no unit ({group}, {index})"
            )))
        }
    }

    fn block(
        &mut self,
        config: &HybridConfig,
        root: &str,
        policy: HybridLayerPolicy,
        layer: usize,
    ) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
        let mut groups = vec![
            self.norm(
                &format!("{root}.input_norm"),
                &format!("{root}.input_layernorm.weight"),
                config.hidden_size,
            )?,
            self.norm(
                &format!("{root}.post_attention_norm"),
                &format!("{root}.post_attention_layernorm.weight"),
                config.hidden_size,
            )?,
        ];
        match policy {
            HybridLayerPolicy::LinearAttention => {
                let spec = super::recurrent_spec(config, layer).map_err(invalid)?;
                let key = spec.key_heads as usize * spec.key_head_dim as usize;
                let value = spec.value_heads as usize * spec.value_head_dim as usize;
                let segments = S::PartitionedSegments {
                    axis: 0,
                    segments: vec![0..key, key..2 * key, 2 * key..2 * key + value],
                };
                let mut members = self.linear(spec.input_qkv, segments.clone(), S::Replicated);
                for projection in [spec.input_gate, spec.input_beta, spec.input_decay] {
                    members.extend(self.linear(
                        projection,
                        S::Partitioned { axis: 0 },
                        S::Replicated,
                    ));
                }
                members.push(ParameterMemberSpec::new(
                    spec.convolution.weight.id.as_str(),
                    vec![
                        spec.convolution.channels as usize,
                        1,
                        spec.convolution.kernel_size as usize,
                    ],
                    segments,
                ));
                for parameter in [spec.decay_bias, spec.transition_log] {
                    members.push(ParameterMemberSpec::new(
                        parameter.id.as_str(),
                        vec![spec.value_heads as usize],
                        S::Partitioned { axis: 0 },
                    ));
                }
                members.push(ParameterMemberSpec::new(
                    spec.normalization_weight.id.as_str(),
                    vec![spec.value_head_dim as usize],
                    S::Replicated,
                ));
                members.extend(self.linear(spec.output, S::Partitioned { axis: 1 }, S::Replicated));
                groups.push(ParameterGroupSpec::partitioned(
                    format!("{root}.linear_attn.heads"),
                    ParameterRole::Channels,
                    spec.key_heads as usize,
                    members,
                )?);
            }
            HybridLayerPolicy::SelfAttention(_) => {
                let mut members = Vec::new();
                for (index, spec) in super::block::attention_projection_specs(config, root)
                    .map_err(invalid)?
                    .into_iter()
                    .enumerate()
                {
                    members.extend(self.linear(
                        spec,
                        S::Partitioned {
                            axis: if index == 3 { 1 } else { 0 },
                        },
                        if index == 3 {
                            S::Replicated
                        } else {
                            S::Partitioned { axis: 0 }
                        },
                    ));
                }
                for name in ["q_norm", "k_norm"] {
                    members.push(ParameterMemberSpec::new(
                        format!("{root}.self_attn.{name}.weight"),
                        vec![config.head_dim as usize],
                        S::Replicated,
                    ));
                }
                groups.push(ParameterGroupSpec::partitioned(
                    format!("{root}.attention.heads"),
                    ParameterRole::AttentionHeads,
                    config.num_key_value_heads as usize,
                    members,
                )?);
            }
        }
        if config.is_moe() {
            let spec =
                super::block::shared_routed_spec(config, layer, root, None).map_err(invalid)?;
            self.formats.insert(
                spec.router.weight().id.as_str().into(),
                spec.router.format().clone(),
            );
            groups.push(eredu_runtime::top_k_group_selector_logical_group(
                format!("{root}.mlp.router"),
                &spec.router,
            )?);
            let GatedProductGroupLayout::Packed { gate_up, down } = spec.experts.layout() else {
                unreachable!("hybrid declares packed experts")
            };
            for projection in [gate_up, down] {
                self.formats.insert(
                    projection.weight().id.as_str().into(),
                    projection.format().clone(),
                );
            }
            groups.push(eredu_runtime::packed_gated_product_logical_group(
                format!("{root}.mlp.experts.intermediate"),
                &spec.experts,
                config.moe_intermediate_size as usize,
            )?);
            groups.push(self.mlp(
                config,
                &format!("{root}.mlp.shared_expert"),
                &format!("{root}.mlp.shared.intermediate"),
                config.shared_expert_intermediate_size,
                ParameterRole::ExpertIntermediate,
            )?);
            let members = self.linear(spec.shared_gate, S::Replicated, S::Replicated);
            groups.push(ParameterGroupSpec::new(
                format!("{root}.mlp.shared_gate"),
                ParameterRole::Replicated,
                members,
            )?);
        } else {
            groups.push(self.mlp(
                config,
                &format!("{root}.mlp"),
                &format!("{root}.mlp.intermediate"),
                config.intermediate_size,
                ParameterRole::FeedForwardIntermediate,
            )?);
        }
        Ok(groups)
    }
}

/// Declares global checkpoint geometry and canonical ownership before native materialization.
pub fn parameter_description(
    config: &HybridConfig,
) -> Result<ArchitectureParameterDescription, ParallelPlanError> {
    config.validate().map_err(invalid)?;
    let targets = config.num_hidden_layers as usize;
    let steps = config.mtp_num_hidden_layers as usize;
    let graph = crate::decoder::SequentialPredictionGroups::new_pattern(
        "model.layers",
        targets,
        "mtp.layers",
        steps,
        1,
    )
    .map_err(invalid)?
    .execution_graph()
    .map_err(invalid)?;
    let layout = ExecutionUnitLayout::new(
        &graph,
        std::iter::once(targets).chain(std::iter::repeat_n(1, steps)),
    )
    .map_err(invalid)?;
    let mut declaration = Declaration::default();
    let statics = super::model::static_module_spec(config);
    let mut roles = vec!["embedding", "mtp"];
    if config.tie_word_embeddings {
        roles.push("output");
    }
    let embedding_format = crate::linear_format::standard_linear_format(
        &statics.embedding_weight,
        statics.embedding_quantization.into(),
    )
    .map_err(invalid)?;
    declaration
        .formats
        .insert(statics.embedding_weight.clone(), embedding_format);
    let mut owned = vec![
        OwnedParameterGroupSpec::new(
            ParameterGroupOwner::static_any_of(roles),
            ParameterGroupSpec::new(
                "model.embed_tokens",
                ParameterRole::Vocabulary,
                [ParameterMemberSpec::new(
                    statics.embedding_weight,
                    vec![statics.vocabulary as usize, statics.hidden_size as usize],
                    S::Balanced { axis: 0 },
                )],
            )?,
        ),
        OwnedParameterGroupSpec::new(
            ParameterGroupOwner::static_role("norm"),
            declaration.norm(
                "model.norm",
                &statics.normalization_weight,
                statics.hidden_size,
            )?,
        ),
    ];
    if !config.tie_word_embeddings {
        let members = declaration.linear(
            super::block::linear_spec(
                config,
                "lm_head",
                config.hidden_size,
                config.vocab_size,
                false,
            )
            .map_err(invalid)?,
            S::Balanced { axis: 0 },
            S::Replicated,
        );
        owned.push(OwnedParameterGroupSpec::new(
            ParameterGroupOwner::static_role("output"),
            ParameterGroupSpec::new("lm_head", ParameterRole::Vocabulary, members)?,
        ));
    }
    if steps > 0 {
        let consumers = (0..steps)
            .map(|depth| {
                eredu_runtime::ExecutionGroupId::new(format!("mtp.{depth}")).map(|group| (group, 0))
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(invalid)?;
        let owner = ParameterGroupOwner::static_unit_consumers("mtp", consumers);
        for name in [
            "mtp.pre_fc_norm_hidden",
            "mtp.pre_fc_norm_embedding",
            "mtp.norm",
        ] {
            owned.push(OwnedParameterGroupSpec::new(
                owner.clone(),
                declaration.norm(name, &format!("{name}.weight"), config.hidden_size)?,
            ));
        }
        let members = declaration.linear(
            super::mtp::fusion_spec(config).map_err(invalid)?,
            S::Replicated,
            S::Replicated,
        );
        owned.push(OwnedParameterGroupSpec::new(
            owner,
            ParameterGroupSpec::new("mtp.fc", ParameterRole::Replicated, members)?,
        ));
    }
    for (group, count) in std::iter::once(targets)
        .chain(std::iter::repeat_n(1, steps))
        .enumerate()
    {
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
            .map(|group| group.group().clone())
            .collect::<Vec<_>>(),
        expanded,
    )
    .map_err(invalid)
}

/// Declares one target or prediction unit from normalized global geometry.
pub fn unit_parallel_parameter_groups(
    config: &HybridConfig,
    group: usize,
    index: usize,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    config.validate().map_err(invalid)?;
    let (groups, formats) = unit_parameter_geometry(config, group, index)?;
    eredu_runtime::expand_linear_format_parameter_groups(groups, |member| {
        Ok(formats.get(member.target()).cloned())
    })
}

/// Logical matrices and formats from the actual rank-local construction specs.
/// Encoding/placement expansion remains checkpoint-global; local geometry is not
/// reinterpreted as a new global sharding plan.
pub(super) fn unit_parameter_geometry(
    config: &HybridConfig,
    group: usize,
    index: usize,
) -> Result<(Vec<ParameterGroupSpec>, BTreeMap<String, LinearFormatSpec>), ParallelPlanError> {
    let mut declaration = Declaration::default();
    let groups = declaration.unit(config, group, index)?;
    Ok((groups, declaration.formats))
}

/// Validates target and prediction invocation, storage, transport and state projections.
/// Parameter geometry remains checkpoint-global when mutable state is rank-local.
pub fn component_definition(
    config: &HybridConfig,
    geometry: Option<&super::LocalGeometry>,
) -> Result<eredu_runtime::ComponentDefinition, eredu_nn::Error> {
    let parameters = parameter_description(config).map_err(eredu_nn::Error::backend)?;
    let state = component_state(config, geometry)?;
    let units = component_units(&parameters);
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
    parameters: &ArchitectureParameterDescription,
) -> Vec<eredu_runtime::ComponentUnit> {
    let layout = parameters.unit_layout();
    (0..layout.len())
        .map(|ordinal| {
            let address = layout.address(ordinal).expect("declared invocation");
            eredu_runtime::ComponentUnit {
                path: if address.group() == 0 {
                    format!("model.layers.{}", address.index())
                } else {
                    format!("mtp.layers.{}", address.group() - 1)
                },
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
    let parameters = parameter_description(config).map_err(eredu_nn::Error::backend)?;
    if definition.parameters() == &parameters {
        return Ok(definition.clone());
    }
    let uses = definition.parameter_uses().clone();
    definition
        .as_ref()
        .clone()
        .with_parameters(parameters, uses)
        .map(std::sync::Arc::new)
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
        &parameter_description(config).map_err(eredu_nn::Error::backend)?,
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
        &component_units(parameters),
        &(0..parameters.graph().groups().len())
            .map(group_transport)
            .collect::<Vec<_>>(),
        &component_state_partition(parameters, definition.state_layout()),
        |_, _, boundaries| !boundaries.is_kernel_owned(),
    )?;
    if definition.state_layout() == &state {
        Ok(definition)
    } else {
        definition
            .as_ref()
            .clone()
            .with_state_layout(state)
            .map(std::sync::Arc::new)
            .map_err(eredu_nn::Error::from)
    }
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
    let mut groups = vec![
        ExecutionGroupSpec::root(super::VISION_EXECUTION_GROUP),
        ExecutionGroupSpec::with_dependencies(
            crate::decoder::TARGET_EXECUTION_GROUP,
            [super::VISION_EXECUTION_GROUP],
        ),
    ];
    let mut output = crate::decoder::TARGET_EXECUTION_GROUP.to_owned();
    for depth in 0..parsed.text.mtp_num_hidden_layers {
        let id = format!("mtp.{depth}");
        groups.push(ExecutionGroupSpec::with_dependencies(id.clone(), [output]));
        output = id;
    }
    eredu_runtime::ExecutionGraph::new(groups, output).map_err(eredu_nn::Error::backend)
}

/// Declares canonical vision, target and prediction ownership without native modules.
pub fn conditional_parameter_description(
    parsed: &super::ParsedHybridConfig,
) -> Result<ArchitectureParameterDescription, eredu_nn::Error> {
    let graph = conditional_execution_graph(parsed)?;
    let vision = parsed.vision.as_ref().expect("validated vision config");
    let text = parameter_description(&parsed.text).map_err(eredu_nn::Error::backend)?;
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
            "model.visual",
            group.clone(),
            "vision",
        )
        .map_err(eredu_nn::Error::backend)?,
    );
    for layer in 0..vision.layer_count() {
        owned.extend(
            crate::qwen::vision::block_parallel_parameter_groups(vision, "model.visual", layer)
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
    ArchitectureParameterDescription::new(
        &graph,
        &layout,
        owned
            .iter()
            .map(|group| group.group().clone())
            .collect::<Vec<_>>(),
        owned,
    )
    .map_err(eredu_nn::Error::backend)
}

/// Validates conditional invocation, transport and text/MTP state ownership before materialization.
pub fn conditional_component_definition(
    parsed: &super::ParsedHybridConfig,
    geometry: Option<&super::ConditionalLocalGeometry>,
) -> Result<eredu_runtime::ComponentDefinition, eredu_nn::Error> {
    let parameters = conditional_parameter_description(parsed)?;
    let state = conditional_state(parsed, geometry)?;
    let units = conditional_units(&parameters);
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
    parameters: &ArchitectureParameterDescription,
) -> Vec<eredu_runtime::ComponentUnit> {
    let layout = parameters.unit_layout();
    let vision_units = layout.group_range(0).expect("declared vision group").len();
    (0..layout.len())
        .map(|ordinal| {
            let address = layout.address(ordinal).expect("declared invocation");
            let (path, state) = match address.group() {
                0 => (format!("model.visual.blocks.{}", address.index()), 0..0),
                group => {
                    let slot = ordinal - vision_units;
                    let path = if group == 1 {
                        format!("model.layers.{}", address.index())
                    } else {
                        format!("mtp.layers.{}", group - 2)
                    };
                    (path, slot..slot + 1)
                }
            };
            eredu_runtime::ComponentUnit { path, state }
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
    if definition.parameters() == &parameters {
        return Ok(definition.clone());
    }
    let uses = definition.parameter_uses().clone();
    definition
        .as_ref()
        .clone()
        .with_parameters(parameters, uses)
        .map(std::sync::Arc::new)
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
        let unsharded =
            crate::qwen::vision::VisionStaticSelection::all(vision.clone(), "model.visual")
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
        &conditional_units(parameters),
        &(0..parameters.graph().groups().len())
            .map(conditional_group_transport)
            .collect::<Vec<_>>(),
        &conditional_state_partition(parameters, definition.state_layout()),
        |_, _, boundaries| !boundaries.is_kernel_owned(),
    )?;
    if definition.state_layout() == &state {
        Ok(definition)
    } else {
        definition
            .as_ref()
            .clone()
            .with_state_layout(state)
            .map(std::sync::Arc::new)
            .map_err(eredu_nn::Error::from)
    }
}
