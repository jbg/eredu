//! Semantic TP/PP/EP placement for neutral GPT-OSS blocks.

use eredu_runtime::ParallelPlanError;

use super::config::ModelArgs;

/// Complete parameter topology derived from neutral router and expert specifications.
pub fn parameter_description(
    declaration: &super::Declaration,
    encoding: crate::decoder::ParameterEncoding,
) -> Result<eredu_runtime::ArchitectureParameterDescription, ParallelPlanError> {
    let args = declaration.args();
    crate::decoder::routed_parameter_description(args, encoding, declaration.common(), |layer| {
        let root = format!("{}.layers.{layer}.mlp", args.parameter_root);
        let spec = declaration.projections().parameters(layer)?.clone();
        Ok(crate::decoder::RoutedBlockParameters {
            selector_name: format!("{root}.router"),
            bank_name: format!("{root}.experts.intermediate"),
            parameters: spec,
        })
    })
}

/// Derives TP-local attention and expert geometry from semantic placement.
pub fn local_block_args(
    args: &ModelArgs,
    bank: &eredu_nn::GroupedGatedProductSpec,
    layer: usize,
    layout: &eredu_runtime::LocalModelLayout,
) -> Result<ModelArgs, ParallelPlanError> {
    let prefix = format!("{}.layers.{layer}", args.parameter_root);
    let tensor = |identity: &str| {
        layout.tensor(identity).ok_or_else(|| {
            ParallelPlanError::InvalidTensor(format!("missing local layout for {identity}"))
        })
    };
    let query = tensor(&format!("{prefix}.self_attn.q_proj.weight"))?;
    let key = tensor(&format!("{prefix}.self_attn.k_proj.weight"))?;
    let query_width = *query.local_shape().first().ok_or_else(|| {
        ParallelPlanError::InvalidTensor("local GPT-OSS query projection is scalar".into())
    })?;
    let key_width = *key.local_shape().first().ok_or_else(|| {
        ParallelPlanError::InvalidTensor("local GPT-OSS key projection is scalar".into())
    })?;
    let head_dim = usize::try_from(args.head_dim)
        .map_err(|_| ParallelPlanError::InvalidTensor("GPT-OSS head width exceeds usize".into()))?;
    if head_dim == 0
        || query_width == 0
        || key_width == 0
        || query_width % head_dim != 0
        || key_width % head_dim != 0
    {
        return Err(ParallelPlanError::InvalidTensor(format!(
            "local GPT-OSS attention widths q={query_width}, k={key_width} split head width {head_dim}"
        )));
    }

    let packed = bank
        .packed_parameters()
        .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
    let gate_up = tensor(packed.gate_up().parameter.id.as_str())?;
    let gate_up_shape = gate_up.local_shape();
    if gate_up_shape.len() < 2
        || gate_up_shape[0] == 0
        || gate_up_shape[1] == 0
        || gate_up_shape[1] % 2 != 0
    {
        return Err(ParallelPlanError::InvalidTensor(format!(
            "local GPT-OSS gate/up shape {gate_up_shape:?} is not [experts, 2*intermediate, ...]"
        )));
    }
    let mut local = args.clone();
    local.num_attention_heads = i32::try_from(query_width / head_dim).map_err(|_| {
        ParallelPlanError::InvalidTensor("local GPT-OSS query-head count exceeds i32".into())
    })?;
    local.num_key_value_heads = i32::try_from(key_width / head_dim).map_err(|_| {
        ParallelPlanError::InvalidTensor("local GPT-OSS key/value-head count exceeds i32".into())
    })?;
    local.num_local_experts = i32::try_from(gate_up_shape[0]).map_err(|_| {
        ParallelPlanError::InvalidTensor("local GPT-OSS expert count exceeds i32".into())
    })?;
    local.intermediate_size = i32::try_from(gate_up_shape[1] / 2).map_err(|_| {
        ParallelPlanError::InvalidTensor("local GPT-OSS expert width exceeds i32".into())
    })?;
    Ok(local)
}

/// Complete rank-local geometry for the canonical neutral GPT-OSS model.
pub type LocalGeometry = crate::decoder::LocalGeometry<ModelArgs>;

/// Derives GPT-OSS unit, vocabulary, and state geometry from one typed plan.
pub fn local_geometry(
    declaration: &super::Declaration,
    layout: &eredu_runtime::LocalModelLayout,
) -> Result<LocalGeometry, ParallelPlanError> {
    crate::decoder::local_geometry(
        declaration.args(),
        declaration.common(),
        layout,
        |_, layer, layout| {
            local_block_args(
                declaration.args(),
                declaration
                    .expert_bank(layer)
                    .map_err(|e| ParallelPlanError::InvalidGroup(e.to_string()))?,
                layer,
                layout,
            )
        },
    )
}

/// Derives exact PP×TP×EP-local GPT-OSS geometry from the selected realization.
pub fn partition_local_routed_geometry(
    declaration: &super::Declaration,
    layout: &eredu_runtime::LocalModelLayout,
    owned_units: std::ops::Range<usize>,
    topology: eredu_core::ParallelRankTopology,
    realization: &crate::ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>,
) -> Result<crate::decoder::PartitionLocalGeometry<ModelArgs>, ParallelPlanError> {
    let args = declaration.args();
    validate_routed_realization(declaration, layout, topology, realization)?;
    crate::decoder::partition_local_geometry_with(
        args,
        declaration.common(),
        parameter_description(declaration, crate::decoder::ParameterEncoding::Stored)?
            .invocations(),
        layout,
        owned_units,
        |_, layer, layout| {
            let mut local = local_block_args(
                declaration.args(),
                declaration.expert_bank(layer)?,
                layer,
                layout,
            )
            .map_err(|error| eredu_nn::Error::backend(error.to_string()))?;
            let local_experts = i32::try_from(realization.local_global_group_indices().len())
                .map_err(eredu_nn::Error::backend)?;
            local.num_local_experts = local_experts;
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
    let banks = declaration
        .expert_banks()
        .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
    let global_experts = banks
        .values()
        .next()
        .ok_or_else(|| {
            ParallelPlanError::InvalidGroup("GPT-OSS has no declared expert banks".into())
        })?
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
            "selected GPT-OSS expert ownership differs from the Cartesian rank".into(),
        ));
    }
    if realization.unit_specs().len() != banks.len() {
        return Err(ParallelPlanError::InvalidGroup(
            "selected GPT-OSS realization unit schedule drifted".into(),
        ));
    }
    let local_count = i32::try_from(expected_local.len()).map_err(|_| {
        ParallelPlanError::InvalidGroup("GPT-OSS local expert count exceeds i32".into())
    })?;
    for (layer, bank) in banks {
        let local = local_block_args(
            declaration.args(),
            declaration
                .expert_bank(layer)
                .map_err(|e| ParallelPlanError::InvalidGroup(e.to_string()))?,
            layer,
            layout,
        )?;
        let expected = bank
            .with_group_geometry(local_count, local.intermediate_size)
            .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
        if realization.unit_spec(crate::decoder::TEXT_DECODER_EXECUTION_GROUP, layer)
            != Some(&expected)
        {
            return Err(ParallelPlanError::InvalidGroup(format!(
                "selected GPT-OSS bank declaration drifted at layer {layer}"
            )));
        }
    }
    Ok(())
}
