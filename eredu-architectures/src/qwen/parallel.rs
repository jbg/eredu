//! Semantic tensor-parallel placement for Qwen decoder blocks.

use eredu_runtime::ParallelPlanError;

use super::ModelArgs;

/// Complete parameter topology derived from neutral router and expert specifications.
pub fn parameter_description(
    declaration: &super::Declaration,
    encoding: crate::decoder::ParameterEncoding,
) -> Result<eredu_runtime::ArchitectureParameterDescription, ParallelPlanError> {
    let args = declaration.args();
    if !args.is_moe() {
        return declaration.common().parameter_description(args, encoding);
    }
    crate::decoder::routed_parameter_description(args, encoding, declaration.common(), |layer| {
        let root = format!("{}.layers.{layer}.mlp", args.parameter_root);
        Ok(crate::decoder::RoutedBlockParameters {
            selector_name: format!("{root}.gate"),
            bank_name: format!("{root}.experts.intermediate"),
            parameters: declaration.routed(layer)?.clone(),
        })
    })
}

/// Derives rank-local Qwen construction geometry from a semantic placement plan.
pub fn local_block_args(
    args: &ModelArgs,
    bank: Option<&eredu_nn::GroupedGatedProductSpec>,
    layer: usize,
    layout: &eredu_runtime::LocalModelLayout,
) -> Result<ModelArgs, ParallelPlanError> {
    let prefix = format!("{}.layers.{layer}", args.parameter_root);
    let tensor = |suffix: &str| {
        layout
            .tensor(&format!("{prefix}.{suffix}.weight"))
            .or_else(|| layout.tensor(&format!("{prefix}.{suffix}")))
            .ok_or_else(|| {
                ParallelPlanError::InvalidTensor(format!(
                    "missing local layout for {prefix}.{suffix}"
                ))
            })
    };
    let query = tensor("self_attn.q_proj")?;
    let key = tensor("self_attn.k_proj")?;
    let query_width = i32::try_from(*query.local_shape().first().ok_or_else(|| {
        ParallelPlanError::InvalidTensor("local Qwen query projection is scalar".into())
    })?)
    .map_err(|_| ParallelPlanError::InvalidTensor("local Qwen query width exceeds i32".into()))?;
    let key_width = i32::try_from(*key.local_shape().first().ok_or_else(|| {
        ParallelPlanError::InvalidTensor("local Qwen key projection is scalar".into())
    })?)
    .map_err(|_| ParallelPlanError::InvalidTensor("local Qwen key width exceeds i32".into()))?;
    if query_width <= 0
        || key_width <= 0
        || query_width % args.head_dim != 0
        || key_width % args.head_dim != 0
    {
        return Err(ParallelPlanError::InvalidTensor(format!(
            "local Qwen attention widths q={query_width}, k={key_width} split head dimension {}",
            args.head_dim
        )));
    }
    let mut local = args.clone();
    local.num_attention_heads = query_width / args.head_dim;
    local.num_key_value_heads = key_width / args.head_dim;
    if args.is_moe() {
        let bank = bank
            .ok_or_else(|| ParallelPlanError::InvalidGroup("missing declared Qwen bank".into()))?;
        let declared = bank
            .packed_parameters()
            .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
        let gate_up = layout
            .tensor(declared.gate_up().parameter.id.as_str())
            .ok_or_else(|| {
                ParallelPlanError::InvalidTensor(format!(
                    "missing local layout for {}",
                    declared.gate_up().parameter.id
                ))
            })?;
        let fused_width = *gate_up.local_shape().get(1).ok_or_else(|| {
            ParallelPlanError::InvalidTensor(
                "local Qwen expert gate/up tensor has rank below 2".into(),
            )
        })?;
        if !fused_width.is_multiple_of(2) {
            return Err(ParallelPlanError::InvalidTensor(format!(
                "local Qwen expert gate/up width {fused_width} is not even"
            )));
        }
        local.moe_intermediate_size = i32::try_from(fused_width / 2).map_err(|_| {
            ParallelPlanError::InvalidTensor("local Qwen expert width exceeds i32".into())
        })?;
        if local.moe_intermediate_size <= 0 {
            return Err(ParallelPlanError::InvalidTensor(
                "local Qwen expert width must be positive".into(),
            ));
        }
    } else {
        let gate = tensor("mlp.gate_proj")?;
        local.intermediate_size = i32::try_from(*gate.local_shape().first().ok_or_else(|| {
            ParallelPlanError::InvalidTensor("local Qwen gate projection is scalar".into())
        })?)
        .map_err(|_| ParallelPlanError::InvalidTensor("local Qwen MLP width exceeds i32".into()))?;
        if local.intermediate_size <= 0 {
            return Err(ParallelPlanError::InvalidTensor(
                "local Qwen MLP width must be positive".into(),
            ));
        }
    }
    Ok(local)
}

/// Complete rank-local geometry for the canonical neutral Qwen text model.
pub type LocalGeometry = crate::decoder::LocalGeometry<ModelArgs>;

/// Derives Qwen unit, vocabulary, and state geometry from one typed plan.
pub fn local_geometry(
    declaration: &super::Declaration,
    layout: &eredu_runtime::LocalModelLayout,
) -> Result<LocalGeometry, ParallelPlanError> {
    crate::decoder::local_geometry(
        declaration.args(),
        declaration.common(),
        layout,
        |args, layer, layout| {
            let bank = if args.is_moe() {
                Some(
                    declaration
                        .expert_bank(layer)
                        .map_err(|e| ParallelPlanError::InvalidGroup(e.to_string()))?,
                )
            } else {
                None
            };
            local_block_args(args, bank, layer, layout)
        },
    )
}

/// Derives exact PP×TP×EP-local geometry from the selected routed realization.
pub fn partition_local_routed_geometry(
    declaration: &super::Declaration,
    layout: &eredu_runtime::LocalModelLayout,
    owned_units: std::ops::Range<usize>,
    topology: eredu_core::ParallelRankTopology,
    realization: &crate::ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>,
) -> Result<crate::decoder::PartitionLocalGeometry<ModelArgs>, ParallelPlanError> {
    let args = declaration.args();
    if !args.is_moe() {
        return Err(ParallelPlanError::InvalidGroup(
            "partition-local routed Qwen geometry requires Qwen3-MoE".into(),
        ));
    }
    validate_routed_realization(declaration, layout, topology, realization)?;
    crate::decoder::partition_local_geometry_with(
        args,
        declaration.common(),
        parameter_description(declaration, crate::decoder::ParameterEncoding::Stored)?
            .invocations(),
        layout,
        owned_units,
        |args, layer, layout| {
            let mut local =
                local_block_args(args, Some(declaration.expert_bank(layer)?), layer, layout)
                    .map_err(|error| eredu_nn::Error::backend(error.to_string()))?;
            let local_experts = i32::try_from(realization.local_global_group_indices().len())
                .map_err(eredu_nn::Error::backend)?;
            local.num_experts = local_experts;
            Ok(local)
        },
    )
}

fn validate_routed_realization(
    declaration: &super::Declaration,
    layout: &eredu_runtime::LocalModelLayout,
    topology: eredu_core::ParallelRankTopology,
    realization: &crate::ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>,
) -> Result<(), ParallelPlanError> {
    let args = declaration.args();
    let banks = declaration
        .expert_banks()
        .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
    let global_experts = banks
        .values()
        .next()
        .ok_or_else(|| ParallelPlanError::InvalidGroup("routed Qwen has no declared banks".into()))?
        .group_count() as usize;
    let expected_local = eredu_core::balanced_contiguous_range(
        global_experts,
        topology.expert_parallel_size(),
        topology.expert_parallel_rank(),
        false,
    )
    .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?
    .collect::<Vec<_>>();
    if realization.global_expert_count() != global_experts
        || realization.expert_parallel_size() != topology.expert_parallel_size()
        || realization.expert_parallel_rank() != topology.expert_parallel_rank()
        || realization.local_global_group_indices() != expected_local
    {
        return Err(ParallelPlanError::InvalidGroup(
            "selected routed Qwen expert ownership differs from the Cartesian rank".into(),
        ));
    }
    if realization.unit_specs().len() != banks.len() {
        return Err(ParallelPlanError::InvalidGroup(
            "selected routed Qwen realization unit schedule drifted".into(),
        ));
    }
    let local_count = i32::try_from(expected_local.len())
        .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
    for (layer, bank) in banks {
        let local = local_block_args(
            args,
            Some(
                declaration
                    .expert_bank(layer)
                    .map_err(|e| ParallelPlanError::InvalidGroup(e.to_string()))?,
            ),
            layer,
            layout,
        )?;
        let expected = bank
            .with_group_geometry(local_count, local.moe_intermediate_size)
            .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
        if realization.unit_spec(crate::decoder::TEXT_DECODER_EXECUTION_GROUP, layer)
            != Some(&expected)
        {
            return Err(ParallelPlanError::InvalidGroup(format!(
                "selected routed Qwen bank declaration drifted at unit {layer}"
            )));
        }
    }
    Ok(())
}
