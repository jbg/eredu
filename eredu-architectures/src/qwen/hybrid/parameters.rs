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
    let mut declaration = Declaration::default();
    let groups = declaration.unit(config, group, index)?;
    eredu_runtime::expand_linear_format_parameter_groups(groups, |member| {
        Ok(declaration.formats.get(member.target()).cloned())
    })
}

/// Validates target and prediction invocation, storage, transport and state projections.
/// Parameter geometry remains checkpoint-global when mutable state is rank-local.
pub fn component_definition(
    config: &HybridConfig,
    geometry: Option<&super::LocalGeometry>,
) -> Result<eredu_runtime::ComponentDefinition, eredu_nn::Error> {
    let parameters = parameter_description(config).map_err(eredu_nn::Error::backend)?;
    let state = match geometry {
        Some(geometry) => {
            geometry
                .validate_for(config)
                .map_err(eredu_nn::Error::backend)?;
            geometry.state_layout().clone()
        }
        None => super::state_layout(config).map_err(eredu_nn::Error::backend)?,
    };
    let layout = parameters.unit_layout();
    let targets = layout.group_range(0).expect("declared target group").len();
    let units = (0..layout.len())
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
        .collect();
    let groups = parameters.graph().groups().len();
    let prediction = (1..groups)
        .map(|group| {
            layout
                .group_id(group)
                .expect("declared prediction group")
                .as_str()
                .to_owned()
        })
        .collect();
    let transport = (0..groups)
        .map(|group| {
            if group == 0 {
                return crate::transport::decoder();
            }
            let mut transport = crate::transport::prediction();
            if group == 1 {
                transport.first_owner_static_roles.push("mtp".into());
            }
            transport
        })
        .collect();
    let partition = crate::transport::pipeline_with_output_state(0, targets, &state);
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
