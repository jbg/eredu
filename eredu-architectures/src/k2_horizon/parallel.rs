//! K2 head, routed-bank, and shared-expert placement in the common decoder.
use super::ModelArgs;
use eredu_runtime::{MemberSharding, ParallelPlanError, ParameterGroupSpec, ParameterRole};

/// Complete physical parameter topology derived before any backend allocation.
pub fn parameter_description(
    declaration: &super::Declaration,
) -> Result<eredu_runtime::ArchitectureParameterDescription, ParallelPlanError> {
    use eredu_nn::DeclareLogicalParameters;
    let args = declaration.args();
    let blocks = declaration
        .blocks()
        .map_err(|e| ParallelPlanError::InvalidGroup(e.to_string()))?;
    let common = declaration
        .common_projection()
        .map_err(|e| ParallelPlanError::InvalidGroup(e.to_string()))?;
    common.parameter_description_with_groups(args, |layer| {
        let Some(routed) = blocks[layer].routed.as_ref() else {
            return Ok(None);
        };
        let intermediate = routed.parameters.experts.intermediate_dimensions() as usize;
        let prefix = format!("model.layers.{layer}");
        fn group(
            name: impl Into<String>,
            role: ParameterRole,
            units: Option<usize>,
            spec: &dyn DeclareLogicalParameters,
            sharding: impl FnMut(&eredu_nn::LogicalParameterDeclaration<'_>) -> MemberSharding,
        ) -> Result<ParameterGroupSpec, ParallelPlanError> {
            let mut declarations = spec
                .logical_parameters()
                .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
            declarations.sort_by(|a, b| a.parameter.id.as_str().cmp(b.parameter.id.as_str()));
            eredu_runtime::logical_parameter_group(name, role, units, &declarations, sharding)
        }
        let mut groups = vec![group(
            format!("{prefix}.mlp.gate"),
            ParameterRole::Replicated,
            None,
            &routed.parameters.selector,
            |_| MemberSharding::Replicated,
        )?];
        let experts = format!("{prefix}.mlp.experts");
        let eredu_nn::GatedProductGroupLayout::Packed { gate_up, down } =
            routed.parameters.experts.layout()
        else {
            return Err(ParallelPlanError::InvalidGroup(
                "K2 placement requires packed expert projections".into(),
            ));
        };
        let output_format = down.format().encoding();
        let bank = group(
            format!("{experts}.intermediate"),
            ParameterRole::ExpertIntermediate,
            Some(crate::linear_format::input_partition_units(
                &experts,
                intermediate,
                1,
                output_format,
            )?),
            &routed.parameters.experts,
            |declaration| {
                if declaration.parameter.id == gate_up.weight().id {
                    MemberSharding::PartitionedSegments {
                        axis: 1,
                        segments: vec![0..intermediate, intermediate..2 * intermediate],
                    }
                } else {
                    MemberSharding::Partitioned { axis: 2 }
                }
            },
        )?;
        groups.push(crate::linear_format::dense_ffn_partition_tail(
            bank,
            intermediate,
            output_format,
            |name| args.linear_format_for(name),
        )?);
        if let Some(shared_spec) = &routed.shared {
            let shared = format!("{prefix}.mlp.shared_experts");
            let output_format = shared_spec.down.format.encoding();
            let shared_group = group(
                format!("{shared}.intermediate"),
                ParameterRole::FeedForwardIntermediate,
                Some(crate::linear_format::input_partition_units(
                    &shared,
                    shared_spec.gate.output as usize,
                    1,
                    output_format,
                )?),
                shared_spec,
                |declaration| MemberSharding::Partitioned {
                    axis: usize::from(declaration.parameter.id == shared_spec.down.weight.id),
                },
            )?;
            groups.push(crate::linear_format::dense_ffn_partition_tail(
                shared_group,
                shared_spec.gate.output as usize,
                output_format,
                |name| args.linear_format_for(name),
            )?);
        }
        if let Some(values) = &blocks[layer].values {
            groups.push(group(
                format!("{prefix}.self_attn.v_router"),
                ParameterRole::Replicated,
                None,
                &values.router,
                |_| MemberSharding::Replicated,
            )?);
            groups.push(group(
                format!("{prefix}.self_attn.v_experts.output"),
                ParameterRole::ExpertOutput,
                Some((values.experts.output_dimensions() / args.head_dim) as usize),
                &values.experts,
                |_| MemberSharding::Partitioned { axis: 1 },
            )?);
        }
        Ok(Some(groups))
    })
}

impl crate::decoder::PartitionedConfig for ModelArgs {
    fn restore_global_geometry(&mut self, global: &Self) -> Result<(), eredu_nn::Error> {
        self.retained_scopes = global.retained_scopes.clone();
        self.fields.num_attention_heads = global.num_attention_heads;
        self.fields.num_key_value_heads = global.num_key_value_heads;
        self.fields.intermediate_size = global.intermediate_size;
        self.fields.num_experts = global.num_experts;
        self.fields.moe_intermediate_size = global.moe_intermediate_size;
        self.local_value_experts = global.local_value_experts;
        self.local_shared_intermediate_size = global.local_shared_intermediate_size;
        self.value_output_start = global.value_output_start;
        Ok(())
    }

    fn routed_bank_order(&self, layer: usize) -> Vec<eredu_runtime::RoutedBankId> {
        let mut order = Vec::new();
        if self.is_mova_layer(layer) {
            order.push(super::ExpertBank::AttentionValue.id());
        }
        if self.is_sparse_layer(layer) {
            order.push(super::ExpertBank::FeedForward.id());
        }
        order
    }

    fn routed_bank_tensor_reductions(
        &self,
        layer: usize,
        bank: eredu_runtime::RoutedBankId,
    ) -> Result<(usize, usize), eredu_nn::Error> {
        if bank == super::ExpertBank::AttentionValue.id() && self.is_mova_layer(layer) {
            // Value projections are complete output shards; only the subsequent
            // attention output projection is summed across tensor ranks.
            Ok((0, 1))
        } else if bank == super::ExpertBank::FeedForward.id() && self.is_sparse_layer(layer) {
            // MoE without value routing still reduces ordinary attention first.
            Ok((
                usize::from(!self.is_mova_layer(layer)),
                1 + usize::from(self.num_shared_experts > 0),
            ))
        } else {
            Err(eredu_nn::Error::backend(
                "K2 block does not invoke the selected bank",
            ))
        }
    }

    fn set_local_geometry(
        &mut self,
        query_heads: i32,
        key_value_heads: i32,
        intermediate: i32,
    ) -> Result<(), eredu_nn::Error> {
        if query_heads <= 0
            || key_value_heads <= 0
            || intermediate <= 0
            || query_heads % key_value_heads != 0
        {
            return Err(eredu_nn::Error::backend(
                "local K2 geometry must contain complete positive GQA groups",
            ));
        }
        self.retained_scopes = None;
        self.fields.num_attention_heads = query_heads;
        self.fields.num_key_value_heads = key_value_heads;
        self.fields.intermediate_size = intermediate;
        Ok(())
    }

    fn local_block_config(
        &self,
        layer: usize,
        layout: &eredu_runtime::LocalModelLayout,
    ) -> Result<Self, eredu_nn::Error> {
        let declaration = super::Declaration::new(self.clone())?;
        local_block_args(self, &declaration.block(layer)?, layer, layout)
            .map_err(eredu_nn::Error::backend)
    }

    fn partition_parameter_description(
        &self,
    ) -> Result<eredu_runtime::ArchitectureParameterDescription, eredu_runtime::ParallelPlanError>
    {
        parameter_description(
            &super::Declaration::new(self.clone())
                .map_err(|e| ParallelPlanError::InvalidGroup(e.to_string()))?,
        )
    }
}

/// Derives local head widths and each bank's cardinality from the retained layout.
pub fn local_block_args(
    args: &ModelArgs,
    declaration: &super::specs::BlockSpec,
    layer: usize,
    layout: &eredu_runtime::LocalModelLayout,
) -> Result<ModelArgs, ParallelPlanError> {
    let prefix = format!("model.layers.{layer}");
    let tensor = |suffix: &str| {
        layout.tensor(&format!("{prefix}.{suffix}")).ok_or_else(|| {
            ParallelPlanError::InvalidTensor(format!("missing local K2 tensor {prefix}.{suffix}"))
        })
    };
    let query = tensor("self_attn.q_proj.weight")?;
    let key = tensor("self_attn.k_proj.weight")?;
    let mut local = args.clone();
    local.retained_scopes = None;
    let q = *query
        .local_shape()
        .first()
        .ok_or_else(|| ParallelPlanError::InvalidTensor("scalar Q projection".into()))?
        as i32;
    let k = *key
        .local_shape()
        .first()
        .ok_or_else(|| ParallelPlanError::InvalidTensor("scalar K projection".into()))?
        as i32;
    if q % args.head_dim != 0 || k % args.head_dim != 0 {
        return Err(ParallelPlanError::InvalidTensor(
            "local projection splits a K2 head".into(),
        ));
    }
    local.fields.num_attention_heads = q / args.head_dim;
    local.fields.num_key_value_heads = k / args.head_dim;
    if let Some(routed) = &declaration.routed {
        let source = routed
            .parameters
            .experts
            .packed_parameters()
            .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
        let experts = layout
            .tensor(source.gate_up().parameter.id.as_str())
            .ok_or_else(|| {
                ParallelPlanError::InvalidTensor("missing declared K2 expert projection".into())
            })?;
        local.fields.num_experts = experts.local_shape()[0] as i32;
        local.fields.moe_intermediate_size = (experts.local_shape()[1] / 2) as i32;
        if let Some(shared) = &routed.shared {
            let shared = layout
                .tensor(shared.down.weight.id.as_str())
                .ok_or_else(|| {
                    ParallelPlanError::InvalidTensor("missing declared K2 shared projection".into())
                })?;
            local.local_shared_intermediate_size = Some(
                i32::try_from(shared.local_shape()[1])
                    .map_err(|error| ParallelPlanError::InvalidTensor(error.to_string()))?,
            );
        }
    } else {
        local.fields.intermediate_size = tensor("mlp.gate_proj.weight")?.local_shape()[0] as i32;
    }
    if let Some(declared) = &declaration.values {
        let values = layout
            .tensor(declared.experts.projection().weight().id.as_str())
            .ok_or_else(|| {
                ParallelPlanError::InvalidTensor("missing declared K2 value projection".into())
            })?;
        local.local_value_experts = Some(values.local_shape()[0] as i32);
        // Output ownership follows the same retained contiguous KV-head partition.
        let dimension = values.local_shape()[1] as i32;
        if dimension != k {
            return Err(ParallelPlanError::InvalidTensor(
                "value bank and key head ownership differ".into(),
            ));
        }
        local.value_output_start = 0;
        for placement in std::iter::once(values.placement()).chain(values.additional_placements()) {
            match placement {
                eredu_runtime::TensorPlacement::Range { axis: 1, start, .. } => {
                    local.value_output_start = *start as i32
                }
                eredu_runtime::TensorPlacement::Shard { axis: 1, index, .. } => {
                    local.value_output_start = *index as i32 * dimension
                }
                eredu_runtime::TensorPlacement::Indices { axis: 1, .. } => {
                    return Err(ParallelPlanError::InvalidTensor(
                        "K2 value output rows must be contiguous complete heads".into(),
                    ));
                }
                _ => {}
            }
        }
    }
    Ok(local)
}

/// Independently fixes each bank's global ownership and rank-local equation.
/// A value expert retains the full hidden input and owns only its KV-head rows.
pub fn expert_realization_plans(
    declaration: &super::Declaration,
    topology: eredu_core::ParallelRankTopology,
    layout: Option<&eredu_runtime::LocalModelLayout>,
) -> Result<
    std::collections::BTreeMap<eredu_runtime::RoutedBankId, crate::routed_text::RoutedGroupedPlan>,
    eredu_nn::Error,
> {
    let args = declaration.args();
    let blocks = declaration.blocks()?;
    use super::ExpertBank;
    use eredu_nn::Error;
    use std::collections::BTreeMap;
    if topology.tensor_parallel_size() > 1 && layout.is_none() {
        return Err(Error::backend(
            "tensor-parallel K2 expert planning requires its retained layout",
        ));
    }
    let owner = eredu_runtime::ExecutionGroupId::new(crate::decoder::TEXT_DECODER_EXECUTION_GROUP)
        .map_err(Error::backend)?;
    let mut plans = BTreeMap::new();
    for bank in [ExpertBank::FeedForward, ExpertBank::AttentionValue] {
        let count = |block: &super::specs::BlockSpec| match bank {
            ExpertBank::FeedForward => block
                .routed
                .as_ref()
                .map(|routed| routed.parameters.experts.group_count()),
            ExpertBank::AttentionValue => block
                .values
                .as_ref()
                .map(|values| values.experts.group_count()),
        };
        let Some(global_count) = blocks.iter().find_map(count) else {
            continue;
        };
        if blocks
            .iter()
            .filter_map(count)
            .any(|count| count != global_count)
        {
            return Err(Error::backend(
                "K2 bank declarations have incompatible expert counts",
            ));
        }
        let local_count = eredu_core::balanced_contiguous_range(
            global_count as usize,
            topology.expert_parallel_size(),
            topology.expert_parallel_rank(),
            false,
        )
        .map_err(Error::backend)?
        .len() as i32;
        let mut gated = BTreeMap::new();
        let mut linear = BTreeMap::new();
        for (layer, block) in blocks.iter().enumerate() {
            if count(block).is_none() {
                continue;
            }
            let mut local = layout
                .map(|layout| local_block_args(args, block, layer, layout))
                .transpose()
                .map_err(Error::backend)?
                .unwrap_or_else(|| args.clone());
            local.retained_scopes = None;
            match bank {
                ExpertBank::FeedForward => local.fields.num_experts = local_count,
                ExpertBank::AttentionValue => local.local_value_experts = Some(local_count),
            }
            let selected = block.clone().with_local_geometry(&local)?;
            match bank {
                ExpertBank::FeedForward => {
                    gated.insert(
                        (owner.clone(), layer),
                        selected
                            .routed
                            .expect("declared feed-forward bank")
                            .parameters
                            .experts,
                    );
                }
                ExpertBank::AttentionValue => {
                    linear.insert(
                        (owner.clone(), layer),
                        selected.values.expect("declared value bank").experts,
                    );
                }
            }
        }
        let plan = match bank {
            ExpertBank::FeedForward => crate::routed_text::RoutedGroupedPlan::Gated(
                crate::ExpertRealizationPlan::balanced(global_count as usize, topology, gated)
                    .map_err(Error::backend)?,
            ),
            ExpertBank::AttentionValue => crate::routed_text::RoutedGroupedPlan::Linear(
                crate::ExpertRealizationPlan::balanced(global_count as usize, topology, linear)
                    .map_err(Error::backend)?,
            ),
        };
        plans.insert(bank.id(), plan);
    }
    Ok(plans)
}

/// Localizes both independent expert banks while retaining global routers.
pub fn partition_local_routed_geometry(
    declaration: &super::Declaration,
    layout: &eredu_runtime::LocalModelLayout,
    owned_units: std::ops::Range<usize>,
    topology: eredu_core::ParallelRankTopology,
    plans: &std::collections::BTreeMap<
        eredu_runtime::RoutedBankId,
        crate::routed_text::RoutedGroupedPlan,
    >,
) -> Result<crate::decoder::PartitionLocalGeometry<ModelArgs>, ParallelPlanError> {
    let args = declaration.args();
    if expert_realization_plans(declaration, topology, Some(layout))
        .map_err(|e| ParallelPlanError::InvalidGroup(e.to_string()))?
        != *plans
    {
        return Err(ParallelPlanError::InvalidGroup(
            "K2 bank ownership or local projection geometry differs from selection".into(),
        ));
    }
    crate::decoder::partition_local_geometry_with(
        args,
        declaration.common(),
        parameter_description(declaration)?.invocations(),
        layout,
        owned_units,
        |args, layer, layout| {
            let mut local = local_block_args(args, &declaration.block(layer)?, layer, layout)
                .map_err(eredu_nn::Error::backend)?;
            if args.is_sparse_layer(layer) {
                local.fields.num_experts = plans[&super::ExpertBank::FeedForward.id()]
                    .local_global_group_indices()
                    .len() as i32;
            }
            if args.is_mova_layer(layer) {
                local.local_value_experts = Some(
                    plans[&super::ExpertBank::AttentionValue.id()]
                        .local_global_group_indices()
                        .len() as i32,
                );
            }
            Ok(local)
        },
    )
}
