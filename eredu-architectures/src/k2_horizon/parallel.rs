//! K2 head, routed-bank, and shared-expert placement in the common decoder.
use super::{
    model::{FeedForward, TransformerBlock},
    ModelArgs,
};
use eredu_nn::GroupedNeuralBackend;
use eredu_runtime::{
    aligned_partition_units, module_parameter_group, partitioned_module_parameter_group,
    MemberSharding, ParallelPlanError, ParameterGroupSpec, ParameterRole,
};

fn executable_shapes(
    args: &ModelArgs,
) -> Result<std::collections::BTreeMap<String, Vec<usize>>, ParallelPlanError> {
    let mut shapes = super::parameter_shapes(args, true)
        .map_err(ParallelPlanError::InvalidTensor)?
        .into_iter()
        .collect::<std::collections::BTreeMap<_, _>>();
    for layer in 0..args.num_hidden_layers as usize {
        if args.is_sparse_layer(layer) {
            let root = format!("model.layers.{layer}.mlp.experts");
            let mut gate = shapes.remove(&format!("{root}.gate_proj.weight")).unwrap();
            shapes.remove(&format!("{root}.up_proj.weight"));
            gate[1] *= 2;
            shapes.insert(format!("{root}.gate_up_proj"), gate);
            let down = shapes.remove(&format!("{root}.down_proj.weight")).unwrap();
            shapes.insert(format!("{root}.down_proj"), down);
        }
    }
    Ok(shapes)
}

/// Complete physical parameter topology derived before any backend allocation.
pub fn parameter_description(
    args: &ModelArgs,
) -> Result<eredu_runtime::ArchitectureParameterDescription, ParallelPlanError> {
    let shapes = executable_shapes(args)?;
    crate::decoder::parameter_description(
        args,
        |layer| !args.is_mova_layer(layer),
        |layer| {
            if !args.is_sparse_layer(layer) {
                return Ok(None);
            }
            let prefix = format!("model.layers.{layer}");
            let members = |root: &str, sharding: &dyn Fn(&str, &[usize]) -> MemberSharding| {
                shapes
                    .iter()
                    .filter(|(name, _)| name.starts_with(&format!("{root}.")))
                    .map(|(name, shape)| {
                        eredu_runtime::ParameterMemberSpec::new(
                            name,
                            shape.clone(),
                            sharding(name, shape),
                        )
                    })
                    .collect::<Vec<_>>()
            };
            let mut groups = Vec::new();
            let router = format!("{prefix}.mlp.gate");
            groups.push(ParameterGroupSpec::new(
                &router,
                ParameterRole::Replicated,
                members(&router, &|_, _| MemberSharding::Replicated),
            )?);
            let experts = format!("{prefix}.mlp.experts");
            let alignment = crate::linear_format::input_partition_alignment(
                args.linear_format_for(&format!("{experts}.down_proj")),
            ) as usize;
            groups.push(ParameterGroupSpec::partitioned(
                format!("{experts}.intermediate"),
                ParameterRole::ExpertIntermediate,
                aligned_partition_units(
                    &experts,
                    args.moe_intermediate_size as usize,
                    1,
                    alignment,
                )?,
                members(&experts, &|name, shape| {
                    if name.ends_with("gate_up_proj") {
                        MemberSharding::PartitionedSegments {
                            axis: 1,
                            segments: vec![0..shape[1] / 2, shape[1] / 2..shape[1]],
                        }
                    } else {
                        MemberSharding::Partitioned { axis: 2 }
                    }
                }),
            )?);
            if args.num_shared_experts > 0 {
                let shared = format!("{prefix}.mlp.shared_experts");
                let alignment = crate::linear_format::input_partition_alignment(
                    args.linear_format_for(&format!("{shared}.down_proj.weight")),
                ) as usize;
                groups.push(ParameterGroupSpec::partitioned(
                    format!("{shared}.intermediate"),
                    ParameterRole::FeedForwardIntermediate,
                    aligned_partition_units(
                        &shared,
                        (args.moe_intermediate_size * args.num_shared_experts) as usize,
                        1,
                        alignment,
                    )?,
                    members(&shared, &|name, _| MemberSharding::Partitioned {
                        axis: usize::from(name.ends_with("down_proj.weight")),
                    }),
                )?);
            }
            if args.is_mova_layer(layer) {
                let router = format!("{prefix}.self_attn.v_router");
                groups.push(ParameterGroupSpec::new(
                    &router,
                    ParameterRole::Replicated,
                    members(&router, &|_, _| MemberSharding::Replicated),
                )?);
                let values = format!("{prefix}.self_attn.v_experts");
                groups.push(ParameterGroupSpec::partitioned(
                    format!("{values}.output"),
                    ParameterRole::ExpertOutput,
                    args.num_key_value_heads as usize,
                    members(&values, &|_, _| MemberSharding::Partitioned { axis: 1 }),
                )?);
            }
            Ok(Some(groups))
        },
    )
}

impl crate::decoder::PartitionedConfig for ModelArgs {
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
        local_block_args(self, layer, layout).map_err(eredu_nn::Error::backend)
    }

    fn validate_partition_parameters(
        &self,
        parameters: &eredu_runtime::ArchitectureParameterDescription,
    ) -> Result<(), eredu_nn::Error> {
        crate::decoder::validate_partitioned_decoder_description(self, parameters)?;
        let expected = executable_shapes(self).map_err(eredu_nn::Error::backend)?;
        let logical = ParameterGroupSpec::new(
            "k2.expected_parameters",
            ParameterRole::Replicated,
            expected.into_iter().map(|(name, shape)| {
                eredu_runtime::ParameterMemberSpec::new(name, shape, MemberSharding::Replicated)
            }),
        )
        .map_err(eredu_nn::Error::backend)?;
        let expanded =
            eredu_runtime::expand_linear_format_parameter_groups(vec![logical], |member| {
                crate::linear_format::standard_parallel_linear_format(
                    member,
                    self.linear_format_for(member.target()),
                )
            })
            .map_err(eredu_nn::Error::backend)?;
        let expected = expanded
            .iter()
            .flat_map(|group| group.members())
            .map(|member| (member.target().to_owned(), member.global_shape().to_vec()))
            .collect::<std::collections::BTreeMap<_, _>>();
        let actual = parameters
            .groups()
            .iter()
            .flat_map(|owned| owned.group().members())
            .map(|member| (member.target().to_owned(), member.global_shape().to_vec()))
            .collect::<std::collections::BTreeMap<_, _>>();
        if actual != expected {
            let differences = expected
                .iter()
                .filter(|(name, shape)| actual.get(*name) != Some(*shape))
                .take(6)
                .map(|(name, shape)| {
                    format!("{name}: expected {shape:?}, actual {:?}", actual.get(name))
                })
                .collect::<Vec<_>>();
            return Err(eredu_nn::Error::backend(format!(
                "partition parameters differ from normalized K2 bank geometry: {}",
                differences.join("; ")
            )));
        }
        Ok(())
    }
}

/// Describes block parameters without moving routing or layer equations into a backend.
pub fn block_parameter_groups<B: GroupedNeuralBackend>(
    block: &TransformerBlock<B>,
    args: &ModelArgs,
    layer: usize,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    let mut groups = crate::decoder::block_common_parallel_parameter_groups(block, args, layer)?;
    let prefix = format!("model.layers.{layer}");
    if let Some(values) = &block.mlp.values {
        groups.push(module_parameter_group::<B::Tensor, _>(
            format!("{prefix}.self_attn.v_router"),
            ParameterRole::Replicated,
            &values.router,
            |_, _| Ok(MemberSharding::Replicated),
        )?);
        groups.push(partitioned_module_parameter_group::<B::Tensor, _>(
            format!("{prefix}.self_attn.v_experts.output"),
            ParameterRole::ExpertOutput,
            args.num_key_value_heads as usize,
            &values.experts,
            |_, shape| {
                if shape.len() < 2 {
                    return Err(ParallelPlanError::InvalidTensor(
                        "value-bank parameter rank is below two".into(),
                    ));
                }
                Ok(MemberSharding::Partitioned { axis: 1 })
            },
        )?);
    }
    match &block.mlp.feed_forward {
        FeedForward::Dense(mlp) => groups.push(crate::decoder::dense_mlp_parallel_parameter_group(
            mlp, args, layer,
        )?),
        FeedForward::Routed {
            router,
            experts,
            shared,
        } => {
            groups.push(module_parameter_group::<B::Tensor, _>(
                format!("{prefix}.mlp.gate"),
                ParameterRole::Replicated,
                router,
                |_, _| Ok(MemberSharding::Replicated),
            )?);
            let width = args.moe_intermediate_size as usize;
            let alignment = crate::linear_format::input_partition_alignment(
                args.linear_format_for(&format!("{prefix}.mlp.experts.down_proj")),
            ) as usize;
            let units = aligned_partition_units(&prefix, width, 1, alignment)?;
            groups.push(partitioned_module_parameter_group::<B::Tensor, _>(
                format!("{prefix}.mlp.experts.intermediate"),
                ParameterRole::ExpertIntermediate,
                units,
                experts,
                |metadata, shape| {
                    let name = metadata.id.as_str();
                    if name.contains("gate_up_proj") {
                        let rows = shape.get(1).copied().ok_or_else(|| {
                            ParallelPlanError::InvalidTensor(format!(
                                "fused expert parameter {name} has no output axis"
                            ))
                        })?;
                        if rows == 0 || rows % 2 != 0 {
                            return Err(ParallelPlanError::InvalidTensor(format!(
                                "fused expert parameter {name} must contain two equal row segments"
                            )));
                        }
                        // Scale rows follow the same two projections, in their
                        // physical block coordinates rather than output neurons.
                        Ok(MemberSharding::PartitionedSegments {
                            axis: 1,
                            segments: vec![0..rows / 2, rows / 2..rows],
                        })
                    } else if name.contains("down_proj") {
                        Ok(MemberSharding::Partitioned { axis: 2 })
                    } else {
                        Err(ParallelPlanError::InvalidTensor(format!(
                            "unexpected expert parameter {name}"
                        )))
                    }
                },
            )?);
            if let Some(shared) = shared {
                let width = (args.moe_intermediate_size * args.num_shared_experts) as usize;
                let alignment =
                    crate::linear_format::input_partition_alignment(args.linear_format_for(
                        &format!("{prefix}.mlp.shared_experts.down_proj.weight"),
                    )) as usize;
                let units = aligned_partition_units(&prefix, width, 1, alignment)?;
                groups.push(partitioned_module_parameter_group::<B::Tensor, _>(
                    format!("{prefix}.mlp.shared_experts.intermediate"),
                    ParameterRole::FeedForwardIntermediate,
                    units,
                    shared,
                    |metadata, _| {
                        Ok(MemberSharding::Partitioned {
                            axis: if metadata.id.as_str().contains("down_proj") {
                                1
                            } else {
                                0
                            },
                        })
                    },
                )?);
            }
        }
    }
    Ok(groups)
}

/// Derives local head widths and each bank's cardinality from the retained layout.
pub fn local_block_args(
    args: &ModelArgs,
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
    if args.is_sparse_layer(layer) {
        let experts = tensor("mlp.experts.gate_up_proj")?;
        local.fields.num_experts = experts.local_shape()[0] as i32;
        local.fields.moe_intermediate_size = (experts.local_shape()[1] / 2) as i32;
    } else {
        local.fields.intermediate_size = tensor("mlp.gate_proj.weight")?.local_shape()[0] as i32;
    }
    if args.is_mova_layer(layer) {
        let values = tensor("self_attn.v_experts.weight")?;
        local.fields.mova_num_experts = values.local_shape()[0] as i32;
        // Output ownership follows the same balanced contiguous KV-head partition.
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
                    ))
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
    args: &ModelArgs,
    topology: eredu_core::ParallelRankTopology,
    layout: Option<&eredu_runtime::LocalModelLayout>,
) -> Result<
    std::collections::BTreeMap<eredu_runtime::RoutedBankId, crate::routed_text::RoutedGroupedPlan>,
    eredu_nn::Error,
> {
    use super::ExpertBank;
    use eredu_nn::Error;
    use std::collections::BTreeMap;
    if topology.tensor_parallel_size() > 1 && layout.is_none() {
        return Err(Error::backend(
            "tensor-parallel K2 expert planning requires its retained layout",
        ));
    }
    let owner = eredu_runtime::ExecutionGroupId::new("text_decoder").map_err(Error::backend)?;
    let mut plans = BTreeMap::new();
    for bank in [ExpertBank::FeedForward, ExpertBank::AttentionValue] {
        let global_count = match bank {
            ExpertBank::FeedForward => args.num_experts,
            ExpertBank::AttentionValue => args.mova_num_experts,
        };
        if global_count == 0 || !args.is_moe() {
            continue;
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
        for layer in 0..args.num_hidden_layers as usize {
            if !args.is_sparse_layer(layer) {
                continue;
            }
            let local = layout
                .map(|layout| local_block_args(args, layer, layout))
                .transpose()
                .map_err(Error::backend)?
                .unwrap_or_else(|| args.clone());
            match bank {
                ExpertBank::FeedForward => {
                    gated.insert(
                        (owner.clone(), layer),
                        super::feed_forward_expert_spec(args, layer)?
                            .with_group_geometry(local_count, local.moe_intermediate_size)?,
                    );
                }
                ExpertBank::AttentionValue => {
                    linear.insert(
                        (owner.clone(), layer),
                        super::value_expert_spec(args, layer)?
                            .partition_output(local.value_output_range())?
                            .with_group_count(local_count)?,
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
    args: &ModelArgs,
    layout: &eredu_runtime::LocalModelLayout,
    owned_units: std::ops::Range<usize>,
    topology: eredu_core::ParallelRankTopology,
    plans: &std::collections::BTreeMap<
        eredu_runtime::RoutedBankId,
        crate::routed_text::RoutedGroupedPlan,
    >,
) -> Result<crate::decoder::PartitionLocalGeometry<ModelArgs>, ParallelPlanError> {
    if expert_realization_plans(args, topology, Some(layout))
        .map_err(|e| ParallelPlanError::InvalidGroup(e.to_string()))?
        != *plans
    {
        return Err(ParallelPlanError::InvalidGroup(
            "K2 bank ownership or local projection geometry differs from selection".into(),
        ));
    }
    crate::decoder::partition_local_geometry_with(
        args,
        layout,
        owned_units,
        |args, layer, layout| {
            let mut local =
                local_block_args(args, layer, layout).map_err(eredu_nn::Error::backend)?;
            if args.is_sparse_layer(layer) {
                local.fields.num_experts = plans[&super::ExpertBank::FeedForward.id()]
                    .local_global_group_indices()
                    .len() as i32;
            }
            if args.is_mova_layer(layer) {
                local.fields.mova_num_experts = plans[&super::ExpertBank::AttentionValue.id()]
                    .local_global_group_indices()
                    .len() as i32;
            }
            Ok(local)
        },
    )
}
