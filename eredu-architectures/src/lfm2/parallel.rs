//! Semantic tensor-parallel placement for LFM2 physical blocks.

use eredu_nn::{GroupedNeuralBackend, VocabularyParallelRange};
use eredu_runtime::{
    aligned_partition_units, module_parameter_group, partitioned_module_parameter_group,
    ArchitectureParameterDescription, ExecutionGraph, ExecutionUnitLayout, LocalModelLayout,
    MemberSharding, OwnedParameterGroupSpec, ParallelPlanError, ParameterGroupOwner,
    ParameterGroupSpec, ParameterMemberSpec, ParameterRole, StateLayout, TensorPlacement,
};
use std::ops::Range;

use crate::decoder::StaticModules;

use super::{
    prompt_cache_architecture_fingerprint, state_layout_with_geometry, Block, BlockGeometry,
    DenseSwiGlu, FeedForward, LayerCacheGeometry, ModelArgs, OperatorPolicy, TokenMixer,
};

/// Complete planner-derived construction and state geometry for one LFM2 rank.
///
/// Fields remain private so block widths, vocabulary ownership, and mutable
/// state can only be realized together from the same typed local layout.
#[derive(Debug, Clone)]
pub struct LocalGeometry {
    blocks: Vec<BlockGeometry>,
    embedding_range: VocabularyParallelRange,
    output_range: Option<VocabularyParallelRange>,
    state_layout: StateLayout,
    architecture_fingerprint: String,
    global_block_geometry: BlockGeometry,
    tied_head: bool,
}

/// Backend-neutral geometry retained by one genuinely pipeline-local LFM2 model.
///
/// Block geometry is stored only for the architecture-global units owned by
/// this pipeline partition. The complete state layout remains available so
/// [`eredu_runtime::ArchitecturePartition`] can derive the exact local slice
/// while preserving its architecture-global offset.
#[derive(Debug, Clone)]
pub struct PartitionLocalGeometry {
    owned_units: Range<usize>,
    blocks: Vec<BlockGeometry>,
    embedding_range: VocabularyParallelRange,
    output_range: Option<VocabularyParallelRange>,
    complete_state_layout: StateLayout,
    architecture_fingerprint: String,
    tied_head: bool,
    static_roles: Vec<String>,
    boundary_schema: eredu_runtime::NoAuxiliaryBoundarySchema,
    expert_realization: Option<crate::ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>>,
    expert_intermediate_range: Option<Range<usize>>,
    expert_banks: Vec<PartitionExpertBankOwnership>,
}

/// One compact routed LFM2 bank at the PP×EP×TP ownership intersection.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct PartitionExpertBankOwnership {
    global_unit: usize,
    global_expert: usize,
    owner_local_expert: usize,
    intermediate_range: Range<usize>,
}

impl PartitionExpertBankOwnership {
    /// Returns the architecture-global physical unit containing this bank.
    pub const fn global_unit(&self) -> usize {
        self.global_unit
    }
    /// Returns the architecture-global expert identity used as the bank key.
    pub const fn global_expert(&self) -> usize {
        self.global_expert
    }
    /// Returns the compact expert ordinal within this EP owner's bank.
    pub const fn owner_local_expert(&self) -> usize {
        self.owner_local_expert
    }
    /// Returns the TP-owned routed-intermediate interval.
    pub fn intermediate_range(&self) -> Range<usize> {
        self.intermediate_range.clone()
    }
    /// Returns the stable architecture-global parameter-bank key.
    pub const fn bank_key(&self) -> eredu_runtime::ParameterBankKey {
        eredu_runtime::ParameterBankKey::new(self.global_unit, self.global_expert)
    }
}

impl PartitionLocalGeometry {
    /// Architecture-global execution-unit range physically owned here.
    pub fn owned_units(&self) -> Range<usize> {
        self.owned_units.clone()
    }

    /// Returns local geometry for one owned architecture-global block.
    pub fn block(&self, global_unit: usize) -> Option<&BlockGeometry> {
        self.owned_units
            .contains(&global_unit)
            .then(|| &self.blocks[global_unit - self.owned_units.start])
    }

    /// Number of block configurations physically retained by this partition.
    pub fn local_unit_count(&self) -> usize {
        self.blocks.len()
    }

    /// Input-embedding vocabulary ownership for this tensor coordinate.
    pub const fn embedding_range(&self) -> &VocabularyParallelRange {
        &self.embedding_range
    }

    /// Untied output-head vocabulary ownership for this tensor coordinate.
    pub const fn output_range(&self) -> Option<&VocabularyParallelRange> {
        self.output_range.as_ref()
    }

    /// Complete TP-local state geometry before pipeline slicing.
    pub const fn complete_state_layout(&self) -> &StateLayout {
        &self.complete_state_layout
    }

    /// Static module roles placed by this pipeline-local range.
    pub fn static_roles(&self) -> &[String] {
        &self.static_roles
    }

    /// Exact no-auxiliary hidden-state boundary for this family.
    pub const fn boundary_schema(&self) -> &eredu_runtime::NoAuxiliaryBoundarySchema {
        &self.boundary_schema
    }

    /// Returns the PP-local state slice in local ordinal order.
    pub fn local_state_layout(&self) -> Result<StateLayout, ParallelPlanError> {
        self.complete_state_layout
            .slice(self.owned_units.clone())
            .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))
    }

    /// Architecture-global ordinal represented by local state ordinal zero.
    pub const fn state_global_offset(&self) -> usize {
        self.owned_units.start
    }

    /// Validates selected static ownership and role-exact boundary before construction.
    pub fn validate_partition_contract(
        &self,
        ownership: &eredu_runtime::PartitionOwnership,
        boundary: &eredu_runtime::NoAuxiliaryBoundarySchema,
    ) -> Result<(), ParallelPlanError> {
        let owns_input = self.owned_units.start == 0;
        let owns_output = self.static_roles.iter().any(|role| role == "output");
        if ownership.owns_input() != owns_input
            || ownership.owns_output() != owns_output
            || ownership.static_roles() != self.static_roles
            || boundary != &self.boundary_schema
        {
            return Err(ParallelPlanError::InvalidGroup(
                "LFM2 partition ownership or boundary drifted from local geometry".into(),
            ));
        }
        Ok(())
    }

    /// Selected expert authority for routed partitions.
    pub const fn expert_realization(
        &self,
    ) -> Option<&crate::ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>> {
        self.expert_realization.as_ref()
    }

    /// Compact routed banks physically owned by this PP×EP×TP rank.
    pub fn expert_banks(&self) -> &[PartitionExpertBankOwnership] {
        &self.expert_banks
    }

    /// Exact checkpoint-global TP slice used by every local routed bank.
    pub fn expert_intermediate_range(&self) -> Option<Range<usize>> {
        self.expert_intermediate_range.clone()
    }

    pub(super) fn validate_for(&self, args: &ModelArgs) -> Result<(), ParallelPlanError> {
        args.validate()
            .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
        if args.has_sparse_moe_layers() != self.expert_realization.is_some() {
            return Err(ParallelPlanError::InvalidGroup(
                "partition-local LFM2 routed authority differs from its schedule".into(),
            ));
        }
        let count = usize::try_from(args.num_hidden_layers).map_err(|_| {
            ParallelPlanError::InvalidGroup("LFM2 layer count exceeds usize".into())
        })?;
        if self.owned_units.is_empty()
            || self.owned_units.end > count
            || self.blocks.len() != self.owned_units.len()
            || self.architecture_fingerprint != prompt_cache_architecture_fingerprint(args)
            || self.tied_head != args.tie_word_embeddings
            || self.static_roles != expected_static_roles(&self.owned_units, count)
            || self.boundary_schema
                != eredu_runtime::NoAuxiliaryBoundarySchema::new(args.hidden_size)
        {
            return Err(ParallelPlanError::InvalidGroup(
                "partition-local LFM2 geometry belongs to a different model or range".into(),
            ));
        }
        self.embedding_range
            .validate_global_rows(args.vocab_size)
            .map_err(|error| ParallelPlanError::InvalidTensor(error.to_string()))?;
        match (args.tie_word_embeddings, &self.output_range) {
            (true, None) => {}
            (false, Some(range)) => range
                .validate_global_rows(args.vocab_size)
                .map_err(|error| ParallelPlanError::InvalidTensor(error.to_string()))?,
            _ => {
                return Err(ParallelPlanError::InvalidGroup(
                    "partition-local LFM2 output vocabulary ownership is inconsistent".into(),
                ));
            }
        }
        if self.complete_state_layout.len() != count {
            return Err(ParallelPlanError::InvalidGroup(
                "partition-local LFM2 complete state length is invalid".into(),
            ));
        }
        for global in self.owned_units.clone() {
            let block = self.block(global).expect("validated owned LFM2 block");
            let policy = self.complete_state_layout.layer(global).ok_or_else(|| {
                ParallelPlanError::InvalidGroup("missing LFM2 complete state layer".into())
            })?;
            let operator = args
                .layer_policy(global)
                .ok_or_else(|| {
                    ParallelPlanError::InvalidGroup(format!("LFM2 has no layer {global}"))
                })?
                .operator;
            let matches = match (operator, policy) {
                (
                    OperatorPolicy::SelfAttention(_),
                    eredu_core::cache::LayerCachePolicy::KeyValue {
                        num_key_value_heads,
                        ..
                    },
                ) => i32::try_from(num_key_value_heads.get())
                    .is_ok_and(|heads| heads == block.key_value_heads),
                (
                    OperatorPolicy::CausalConvolution,
                    eredu_core::cache::LayerCachePolicy::FixedState { tensors },
                ) => tensors.first().is_some_and(|tensor| {
                    matches!(
                        tensor.shape.get(2),
                        Some(eredu_core::cache::StateTensorDimension::Fixed(width))
                            if i32::try_from(width.get()).is_ok_and(|width| width == block.convolution_channels)
                    )
                }),
                (
                    OperatorPolicy::CausalConvolution,
                    eredu_core::cache::LayerCachePolicy::NoState,
                ) => args.conv_l_cache == 1,
                _ => false,
            };
            if !matches {
                return Err(ParallelPlanError::InvalidGroup(format!(
                    "owned LFM2 block {global} differs from complete TP-local state geometry"
                )));
            }
        }
        Ok(())
    }
}

impl LocalGeometry {
    /// Returns local block geometry in physical execution order.
    pub fn blocks(&self) -> &[BlockGeometry] {
        &self.blocks
    }

    /// Returns one validated rank-local block geometry.
    pub fn block(&self, layer: usize) -> Option<&BlockGeometry> {
        self.blocks.get(layer)
    }

    /// Returns input-embedding vocabulary ownership.
    pub const fn embedding_range(&self) -> &VocabularyParallelRange {
        &self.embedding_range
    }

    /// Returns untied output-head vocabulary ownership.
    pub const fn output_range(&self) -> Option<&VocabularyParallelRange> {
        self.output_range.as_ref()
    }

    /// Returns the authoritative state layout derived from local blocks.
    pub const fn state_layout(&self) -> &StateLayout {
        &self.state_layout
    }

    pub(super) fn validate_for(&self, args: &ModelArgs) -> Result<(), ParallelPlanError> {
        let expected_layers = usize::try_from(args.num_hidden_layers).map_err(|_| {
            ParallelPlanError::InvalidGroup("LFM2 layer count exceeds usize".into())
        })?;
        if self.blocks.len() != expected_layers
            || self.architecture_fingerprint != prompt_cache_architecture_fingerprint(args)
            || self.global_block_geometry != BlockGeometry::replicated(args)
            || self.tied_head != args.tie_word_embeddings
        {
            return Err(ParallelPlanError::InvalidGroup(
                "rank-local LFM2 geometry belongs to a different model configuration".into(),
            ));
        }
        self.embedding_range
            .validate_global_rows(args.vocab_size)
            .map_err(|error| ParallelPlanError::InvalidTensor(error.to_string()))?;
        match (args.tie_word_embeddings, &self.output_range) {
            (true, None) => {}
            (false, Some(range)) => range
                .validate_global_rows(args.vocab_size)
                .map_err(|error| ParallelPlanError::InvalidTensor(error.to_string()))?,
            (true, Some(_)) => {
                return Err(ParallelPlanError::InvalidGroup(
                    "tied LFM2 output unexpectedly owns a separate vocabulary range".into(),
                ));
            }
            (false, None) => {
                return Err(ParallelPlanError::InvalidGroup(
                    "untied LFM2 output is missing vocabulary ownership".into(),
                ));
            }
        }
        let expected = state_layout_with_geometry(args, &state_geometry(args, &self.blocks))
            .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
        if expected != self.state_layout {
            return Err(ParallelPlanError::InvalidGroup(
                "rank-local LFM2 state layout drifted from block geometry".into(),
            ));
        }
        Ok(())
    }
}

fn local_width(
    layout: &LocalModelLayout,
    name: &str,
    axis: usize,
) -> Result<i32, ParallelPlanError> {
    let tensor = layout.tensor(name).ok_or_else(|| {
        ParallelPlanError::InvalidTensor(format!("missing local LFM2 layout for {name}"))
    })?;
    let local = *tensor.local_shape().get(axis).ok_or_else(|| {
        ParallelPlanError::InvalidTensor(format!("LFM2 tensor {name} has no axis {axis}"))
    })?;
    let global = *tensor.global_shape().get(axis).ok_or_else(|| {
        ParallelPlanError::InvalidTensor(format!("LFM2 tensor {name} has no global axis {axis}"))
    })?;
    if local == 0 || local > global {
        return Err(ParallelPlanError::InvalidTensor(format!(
            "LFM2 tensor {name} has invalid local width {local} of global width {global}"
        )));
    }
    i32::try_from(local)
        .map_err(|_| ParallelPlanError::InvalidTensor(format!("LFM2 width for {name} exceeds i32")))
}

/// Derives one block's local widths exclusively from the resolved placement.
pub fn local_block_geometry(
    args: &ModelArgs,
    layer: usize,
    layout: &LocalModelLayout,
) -> Result<BlockGeometry, ParallelPlanError> {
    let root = format!("model.layers.{layer}");
    let mut geometry = BlockGeometry::replicated(args);
    match args
        .layer_policy(layer)
        .ok_or_else(|| ParallelPlanError::InvalidGroup(format!("LFM2 has no layer {layer}")))?
        .operator
    {
        OperatorPolicy::SelfAttention(_) => {
            let head = args.hidden_size / args.num_attention_heads;
            let query = local_width(layout, &format!("{root}.self_attn.q_proj.weight"), 0)?;
            let key = local_width(layout, &format!("{root}.self_attn.k_proj.weight"), 0)?;
            if query % head != 0 || key % head != 0 {
                return Err(ParallelPlanError::InvalidTensor(format!(
                    "local LFM2 attention widths q={query} k={key} split head dimension {head}"
                )));
            }
            geometry.query_heads = query / head;
            geometry.key_value_heads = key / head;
        }
        OperatorPolicy::CausalConvolution => {
            geometry.convolution_channels =
                local_width(layout, &format!("{root}.conv.conv.weight"), 0)?;
        }
    }
    match args.layer_policy(layer).unwrap().feed_forward {
        super::FeedForwardPolicy::Dense => {
            geometry.dense_intermediate =
                local_width(layout, &format!("{root}.feed_forward.w1.weight"), 0)?;
        }
        super::FeedForwardPolicy::SparseMoe => {
            let fused = local_width(
                layout,
                &format!("{root}.feed_forward.experts.gate_up_proj"),
                1,
            )?;
            if fused % 2 != 0 {
                return Err(ParallelPlanError::InvalidTensor(format!(
                    "local LFM2 packed expert width {fused} is not even"
                )));
            }
            geometry.expert_intermediate = fused / 2;
        }
    }
    Ok(geometry)
}

/// Returns rank-local mutable-state geometry for every scheduled block.
pub fn local_state_geometry(
    args: &ModelArgs,
    layout: &LocalModelLayout,
) -> Result<Vec<LayerCacheGeometry>, ParallelPlanError> {
    let blocks = (0..args.num_hidden_layers as usize)
        .map(|layer| local_block_geometry(args, layer, layout))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(state_geometry(args, &blocks))
}

/// Derives complete TP-local heterogeneous state before pipeline slicing.
///
/// State follows attention key/value heads or convolution output channels;
/// feed-forward routing does not alter either state-bearing width.
pub fn partitioned_state_layout(
    args: &ModelArgs,
    tensor_rank: usize,
    tensor_size: usize,
) -> Result<StateLayout, ParallelPlanError> {
    args.validate()
        .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
    if tensor_size == 0 || tensor_rank >= tensor_size {
        return Err(ParallelPlanError::InvalidGroup(format!(
            "invalid LFM2 tensor coordinate {tensor_rank}/{tensor_size}"
        )));
    }
    let select = |value: i32, role: &str| {
        let global = usize::try_from(value)
            .map_err(|_| ParallelPlanError::InvalidGroup(format!("LFM2 {role} is not positive")))?;
        let local = eredu_core::balanced_contiguous_range(global, tensor_size, tensor_rank, false)
            .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?
            .len();
        i32::try_from(local)
            .map_err(|_| ParallelPlanError::InvalidGroup(format!("LFM2 local {role} exceeds i32")))
    };
    let key_value_heads = select(args.num_key_value_heads, "key/value head count")?;
    let convolution_channels = select(args.hidden_size, "convolution channel count")?;
    let geometry = args
        .layer_schedule
        .iter()
        .map(|policy| match policy.operator {
            OperatorPolicy::CausalConvolution => LayerCacheGeometry {
                kv_heads: None,
                convolution_channels: Some(convolution_channels),
            },
            OperatorPolicy::SelfAttention(_) => LayerCacheGeometry {
                kv_heads: Some(key_value_heads),
                convolution_channels: None,
            },
        })
        .collect::<Vec<_>>();
    state_layout_with_geometry(args, &geometry)
        .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))
}

fn state_geometry(args: &ModelArgs, blocks: &[BlockGeometry]) -> Vec<LayerCacheGeometry> {
    args.layer_schedule
        .iter()
        .zip(blocks)
        .map(|(policy, geometry)| match policy.operator {
            OperatorPolicy::SelfAttention(_) => LayerCacheGeometry {
                kv_heads: Some(geometry.key_value_heads),
                convolution_channels: None,
            },
            OperatorPolicy::CausalConvolution => LayerCacheGeometry {
                kv_heads: None,
                convolution_channels: Some(geometry.convolution_channels),
            },
        })
        .collect()
}

fn vocabulary_range(
    layout: &LocalModelLayout,
    logical_name: &str,
    global_vocabulary: usize,
) -> Result<VocabularyParallelRange, ParallelPlanError> {
    let mut selected = None;
    let mut found = false;
    for (target, tensor) in layout
        .tensors()
        .filter(|(_, tensor)| tensor.logical_name() == logical_name)
    {
        found = true;
        if tensor.global_shape().first().copied() != Some(global_vocabulary) {
            return Err(ParallelPlanError::InvalidTensor(format!(
                "LFM2 vocabulary member {target} has global shape {:?}, expected {global_vocabulary} rows",
                tensor.global_shape()
            )));
        }
        let range = match tensor.placement() {
            TensorPlacement::Range {
                axis: 0,
                start,
                end,
            } => *start..*end,
            TensorPlacement::Replicated => 0..global_vocabulary,
            placement => {
                return Err(ParallelPlanError::InvalidTensor(format!(
                    "LFM2 vocabulary member {target} has non-row placement {placement:?}"
                )));
            }
        };
        if selected.as_ref().is_some_and(|current| current != &range) {
            return Err(ParallelPlanError::InvalidTensor(format!(
                "LFM2 vocabulary group {logical_name} has inconsistent companion selections"
            )));
        }
        selected = Some(range);
    }
    if !found {
        return Err(ParallelPlanError::InvalidTensor(format!(
            "missing local LFM2 vocabulary layout for {logical_name}"
        )));
    }
    let range = VocabularyParallelRange {
        global_vocabulary,
        local: selected.expect("a found LFM2 vocabulary member supplies a selection"),
    };
    range
        .validate()
        .map_err(|error| ParallelPlanError::InvalidTensor(error.to_string()))?;
    Ok(range)
}

/// Derives complete rank-local LFM2 construction geometry from one typed plan.
pub fn local_geometry(
    args: &ModelArgs,
    layout: &LocalModelLayout,
) -> Result<LocalGeometry, ParallelPlanError> {
    args.validate()
        .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
    let layers = usize::try_from(args.num_hidden_layers)
        .map_err(|_| ParallelPlanError::InvalidGroup("LFM2 layer count exceeds usize".into()))?;
    let blocks = (0..layers)
        .map(|layer| local_block_geometry(args, layer, layout))
        .collect::<Result<Vec<_>, _>>()?;
    let state_layout = state_layout_with_geometry(args, &state_geometry(args, &blocks))
        .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
    let global_vocabulary = usize::try_from(args.vocab_size).map_err(|_| {
        ParallelPlanError::InvalidGroup("LFM2 vocabulary size exceeds usize".into())
    })?;
    let embedding_range = vocabulary_range(layout, "model.embed_tokens", global_vocabulary)?;
    let output_range = if args.tie_word_embeddings {
        None
    } else {
        Some(vocabulary_range(layout, "lm_head", global_vocabulary)?)
    };
    let geometry = LocalGeometry {
        blocks,
        embedding_range,
        output_range,
        state_layout,
        architecture_fingerprint: prompt_cache_architecture_fingerprint(args),
        global_block_geometry: BlockGeometry::replicated(args),
        tied_head: args.tie_word_embeddings,
    };
    geometry.validate_for(args)?;
    Ok(geometry)
}

/// Derives exact TP-local and PP-local construction geometry for dense LFM2.
pub fn partition_local_geometry(
    args: &ModelArgs,
    layout: &LocalModelLayout,
    owned_units: Range<usize>,
) -> Result<PartitionLocalGeometry, ParallelPlanError> {
    args.validate()
        .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
    if args.has_sparse_moe_layers() {
        return Err(ParallelPlanError::InvalidGroup(
            "partitioned resident LFM2 does not accept routed feed-forward layers".into(),
        ));
    }
    let count = usize::try_from(args.num_hidden_layers)
        .map_err(|_| ParallelPlanError::InvalidGroup("LFM2 layer count exceeds usize".into()))?;
    if owned_units.is_empty() || owned_units.end > count {
        return Err(ParallelPlanError::InvalidGroup(format!(
            "LFM2 local unit range {owned_units:?} is outside {count} layers"
        )));
    }
    let complete = local_geometry(args, layout)?;
    let blocks = owned_units
        .clone()
        .map(|unit| {
            complete.block(unit).copied().ok_or_else(|| {
                ParallelPlanError::InvalidGroup(format!("LFM2 has no local block {unit}"))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let geometry = PartitionLocalGeometry {
        owned_units: owned_units.clone(),
        blocks,
        embedding_range: complete.embedding_range.clone(),
        output_range: complete.output_range.clone(),
        complete_state_layout: complete.state_layout.clone(),
        architecture_fingerprint: complete.architecture_fingerprint.clone(),
        tied_head: complete.tied_head,
        static_roles: expected_static_roles(&owned_units, count),
        boundary_schema: eredu_runtime::NoAuxiliaryBoundarySchema::new(args.hidden_size),
        expert_realization: None,
        expert_intermediate_range: None,
        expert_banks: Vec::new(),
    };
    geometry.validate_for(args)?;
    Ok(geometry)
}

/// Derives prediction-free routed LFM2 geometry from the selected expert plan.
pub fn partition_local_routed_geometry(
    args: &ModelArgs,
    layout: &LocalModelLayout,
    owned_units: Range<usize>,
    topology: eredu_core::ParallelRankTopology,
    realization: &crate::ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>,
) -> Result<PartitionLocalGeometry, ParallelPlanError> {
    if !args.has_sparse_moe_layers() {
        return Err(ParallelPlanError::InvalidGroup(
            "routed LFM2 requires sparse units".into(),
        ));
    }
    let count = usize::try_from(args.num_hidden_layers)
        .map_err(|_| ParallelPlanError::InvalidGroup("LFM2 layer count exceeds usize".into()))?;
    let expected_units = eredu_core::balanced_contiguous_range(
        count,
        topology.pipeline_parallel_size(),
        topology.pipeline_parallel_rank(),
        false,
    )
    .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
    if owned_units.is_empty() || owned_units.end > count || owned_units != expected_units {
        return Err(ParallelPlanError::InvalidGroup(
            "routed LFM2 PP ownership differs from its Cartesian rank".into(),
        ));
    }
    let complete = local_geometry(args, layout)?;
    let global_experts = usize::try_from(args.num_experts)
        .map_err(|_| ParallelPlanError::InvalidGroup("LFM2 expert count exceeds usize".into()))?;
    let local_experts = validate_realization(topology, global_experts, realization)?;
    let owner_local_count = i32::try_from(local_experts.len()).map_err(|_| {
        ParallelPlanError::InvalidGroup("LFM2 local expert count exceeds i32".into())
    })?;
    let mut intermediate_range = None;
    let mut expected_sparse_units = 0;
    for (unit, policy) in args.layer_schedule.iter().enumerate() {
        if policy.feed_forward != super::FeedForwardPolicy::SparseMoe {
            continue;
        }
        expected_sparse_units += 1;
        let block = complete.block(unit).ok_or_else(|| {
            ParallelPlanError::InvalidGroup("missing local LFM2 sparse block".into())
        })?;
        let expected = super::moe::expert_bank_spec(args, unit)
            .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?
            .with_group_geometry(owner_local_count, block.expert_intermediate)
            .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
        if realization.unit_spec(crate::decoder::TARGET_EXECUTION_GROUP, unit) != Some(&expected) {
            return Err(ParallelPlanError::InvalidGroup(format!(
                "LFM2 expert plan drifted at unit {unit}"
            )));
        }
        let target = format!("model.layers.{unit}.feed_forward.experts.gate_up_proj");
        let range = exact_logical_range(layout, &target, args.moe_intermediate_size)?;
        if intermediate_range
            .as_ref()
            .is_some_and(|current| current != &range)
        {
            return Err(ParallelPlanError::InvalidTensor(
                "LFM2 expert TP ranges differ between units".into(),
            ));
        }
        intermediate_range = Some(range);
    }
    if realization.unit_specs().len() != expected_sparse_units {
        return Err(ParallelPlanError::InvalidGroup(
            "LFM2 expert unit schedule drifted".into(),
        ));
    }
    let intermediate_range = intermediate_range
        .ok_or_else(|| ParallelPlanError::InvalidGroup("LFM2 has no routed TP range".into()))?;
    let blocks = owned_units
        .clone()
        .map(|unit| {
            complete
                .block(unit)
                .copied()
                .ok_or_else(|| ParallelPlanError::InvalidGroup("missing owned LFM2 block".into()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let expert_banks = owned_units
        .clone()
        .filter(|unit| {
            args.layer_policy(*unit)
                .is_some_and(|policy| policy.feed_forward == super::FeedForwardPolicy::SparseMoe)
        })
        .flat_map(|global_unit| {
            local_experts.iter().copied().enumerate().map({
                let range = intermediate_range.clone();
                move |(owner_local_expert, global_expert)| PartitionExpertBankOwnership {
                    global_unit,
                    global_expert,
                    owner_local_expert,
                    intermediate_range: range.clone(),
                }
            })
        })
        .collect();
    let geometry = PartitionLocalGeometry {
        owned_units: owned_units.clone(),
        blocks,
        embedding_range: complete.embedding_range.clone(),
        output_range: complete.output_range.clone(),
        complete_state_layout: complete.state_layout.clone(),
        architecture_fingerprint: complete.architecture_fingerprint.clone(),
        tied_head: complete.tied_head,
        static_roles: expected_static_roles(&owned_units, count),
        boundary_schema: eredu_runtime::NoAuxiliaryBoundarySchema::new(args.hidden_size),
        expert_realization: Some(realization.clone()),
        expert_intermediate_range: Some(intermediate_range),
        expert_banks,
    };
    geometry.validate_for(args)?;
    Ok(geometry)
}

fn expected_static_roles(owned: &Range<usize>, count: usize) -> Vec<String> {
    let mut roles = Vec::new();
    if owned.start == 0 {
        roles.push("embedding".into());
    }
    if owned.end == count {
        roles.extend(["norm".into(), "output".into()]);
    }
    roles
}

fn exact_logical_range(
    layout: &LocalModelLayout,
    target: &str,
    global: i32,
) -> Result<Range<usize>, ParallelPlanError> {
    let global = usize::try_from(global)
        .map_err(|_| ParallelPlanError::InvalidTensor("expert width exceeds usize".into()))?;
    let tensor = layout
        .tensor(target)
        .ok_or_else(|| ParallelPlanError::InvalidTensor(format!("missing {target}")))?;
    if tensor.logical_units() != Some(global) {
        return Err(ParallelPlanError::InvalidTensor(format!(
            "{target} has wrong semantic width"
        )));
    }
    tensor
        .logical_range()
        .cloned()
        .filter(|range| !range.is_empty() && range.end <= global)
        .ok_or_else(|| ParallelPlanError::InvalidTensor(format!("{target} has no exact TP range")))
}

fn validate_realization(
    topology: eredu_core::ParallelRankTopology,
    global: usize,
    realization: &crate::ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>,
) -> Result<Vec<usize>, ParallelPlanError> {
    let local = eredu_core::balanced_contiguous_range(
        global,
        topology.expert_parallel_size(),
        topology.expert_parallel_rank(),
        false,
    )
    .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?
    .collect::<Vec<_>>();
    let expected_group = topology
        .subgroup(eredu_core::ParallelAxis::Expert)
        .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
    let selected_group = realization
        .collective_group(eredu_core::CollectiveGroupId::new(0))
        .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
    if realization.global_expert_count() != global
        || realization.expert_parallel_size() != topology.expert_parallel_size()
        || realization.expert_parallel_rank() != topology.expert_parallel_rank()
        || realization.local_global_group_indices() != local
        || selected_group.members() != expected_group.global_ranks()
        || selected_group.local_rank() != expected_group.rank()
    {
        return Err(ParallelPlanError::InvalidGroup(
            "LFM2 expert owner differs from Cartesian rank".into(),
        ));
    }
    Ok(local)
}

/// Describes dense LFM2 parameter topology without constructing backend modules.
pub fn dense_parameter_description(
    args: &ModelArgs,
) -> Result<ArchitectureParameterDescription, ParallelPlanError> {
    args.validate()
        .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
    if args.has_sparse_moe_layers() {
        return Err(ParallelPlanError::InvalidGroup(
            "dense LFM2 parameter topology does not accept routed layers".into(),
        ));
    }
    let layers = usize::try_from(args.num_hidden_layers)
        .map_err(|_| ParallelPlanError::InvalidGroup("LFM2 layer count exceeds usize".into()))?;
    let graph = ExecutionGraph::new(
        vec![eredu_runtime::ExecutionGroupSpec::root(
            crate::decoder::TARGET_EXECUTION_GROUP,
        )],
        crate::decoder::TARGET_EXECUTION_GROUP,
    )
    .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
    let layout = ExecutionUnitLayout::new(&graph, [layers])
        .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
    let mut groups = vec![ParameterGroupSpec::new(
        "model.embed_tokens",
        ParameterRole::Vocabulary,
        [ParameterMemberSpec::new(
            "model.embed_tokens.weight",
            vec![args.vocab_size as usize, args.hidden_size as usize],
            MemberSharding::Balanced { axis: 0 },
        )],
    )?];
    groups.push(ParameterGroupSpec::new(
        "model.embedding_norm",
        ParameterRole::Replicated,
        [ParameterMemberSpec::new(
            "model.embedding_norm.weight",
            vec![args.hidden_size as usize],
            MemberSharding::Replicated,
        )],
    )?);
    if !args.tie_word_embeddings {
        groups.push(ParameterGroupSpec::new(
            "lm_head",
            ParameterRole::Vocabulary,
            [ParameterMemberSpec::new(
                "lm_head.weight",
                vec![args.vocab_size as usize, args.hidden_size as usize],
                MemberSharding::Balanced { axis: 0 },
            )],
        )?);
    }
    let static_count = groups.len();
    let head_dim = usize::try_from(args.hidden_size / args.num_attention_heads)
        .map_err(|_| ParallelPlanError::InvalidGroup("LFM2 head width exceeds usize".into()))?;
    let hidden = usize::try_from(args.hidden_size)
        .map_err(|_| ParallelPlanError::InvalidGroup("LFM2 hidden width exceeds usize".into()))?;
    let kv_heads = usize::try_from(args.num_key_value_heads)
        .map_err(|_| ParallelPlanError::InvalidGroup("LFM2 KV heads exceed usize".into()))?;
    let intermediate = usize::try_from(args.dense_intermediate_size).map_err(|_| {
        ParallelPlanError::InvalidGroup("LFM2 intermediate width exceeds usize".into())
    })?;
    for layer in 0..layers {
        let root = format!("model.layers.{layer}");
        match args
            .layer_policy(layer)
            .expect("validated LFM2 schedule")
            .operator
        {
            OperatorPolicy::SelfAttention(_) => groups.push(ParameterGroupSpec::partitioned(
                format!("{root}.self_attn.heads"),
                ParameterRole::AttentionHeads,
                kv_heads,
                [
                    ParameterMemberSpec::new(
                        format!("{root}.self_attn.q_proj.weight"),
                        vec![hidden, hidden],
                        MemberSharding::Partitioned { axis: 0 },
                    ),
                    ParameterMemberSpec::new(
                        format!("{root}.self_attn.k_proj.weight"),
                        vec![kv_heads * head_dim, hidden],
                        MemberSharding::Partitioned { axis: 0 },
                    ),
                    ParameterMemberSpec::new(
                        format!("{root}.self_attn.v_proj.weight"),
                        vec![kv_heads * head_dim, hidden],
                        MemberSharding::Partitioned { axis: 0 },
                    ),
                    ParameterMemberSpec::new(
                        format!("{root}.self_attn.out_proj.weight"),
                        vec![hidden, hidden],
                        MemberSharding::Partitioned { axis: 1 },
                    ),
                    ParameterMemberSpec::new(
                        format!("{root}.self_attn.q_layernorm.weight"),
                        vec![head_dim],
                        MemberSharding::Replicated,
                    ),
                    ParameterMemberSpec::new(
                        format!("{root}.self_attn.k_layernorm.weight"),
                        vec![head_dim],
                        MemberSharding::Replicated,
                    ),
                ],
            )?),
            OperatorPolicy::CausalConvolution => {
                let segments = vec![0..hidden, hidden..2 * hidden, 2 * hidden..3 * hidden];
                let mut members = vec![
                    ParameterMemberSpec::new(
                        format!("{root}.conv.in_proj.weight"),
                        vec![3 * hidden, hidden],
                        MemberSharding::PartitionedSegments {
                            axis: 0,
                            segments: segments.clone(),
                        },
                    ),
                    ParameterMemberSpec::new(
                        format!("{root}.conv.conv.weight"),
                        vec![hidden, 1, args.conv_l_cache as usize],
                        MemberSharding::Partitioned { axis: 0 },
                    ),
                    ParameterMemberSpec::new(
                        format!("{root}.conv.out_proj.weight"),
                        vec![hidden, hidden],
                        MemberSharding::Partitioned { axis: 1 },
                    ),
                ];
                if args.conv_bias {
                    members.extend([
                        ParameterMemberSpec::new(
                            format!("{root}.conv.in_proj.bias"),
                            vec![3 * hidden],
                            MemberSharding::PartitionedSegments {
                                axis: 0,
                                segments: segments.clone(),
                            },
                        ),
                        ParameterMemberSpec::new(
                            format!("{root}.conv.out_proj.bias"),
                            vec![hidden],
                            MemberSharding::Replicated,
                        ),
                        ParameterMemberSpec::new(
                            format!("{root}.conv.conv.bias"),
                            vec![hidden],
                            MemberSharding::Partitioned { axis: 0 },
                        ),
                    ]);
                }
                groups.push(ParameterGroupSpec::partitioned(
                    format!("{root}.conv.channels"),
                    ParameterRole::Channels,
                    hidden,
                    members,
                )?);
            }
        }
        for name in ["operator_norm", "ffn_norm"] {
            groups.push(ParameterGroupSpec::new(
                format!("{root}.{name}"),
                ParameterRole::Replicated,
                [ParameterMemberSpec::new(
                    format!("{root}.{name}.weight"),
                    vec![hidden],
                    MemberSharding::Replicated,
                )],
            )?);
        }
        let alignment = args
            .weight_quantization_for(&format!("{root}.feed_forward.w2.weight"))
            .map_or(Ok(1), |quantization| {
                usize::try_from(quantization.group_size()).map_err(|_| {
                    ParallelPlanError::InvalidGroup(
                        "LFM2 dense quantization group exceeds usize".into(),
                    )
                })
            })?;
        groups.push(ParameterGroupSpec::partitioned(
            format!("{root}.feed_forward.intermediate"),
            ParameterRole::FeedForwardIntermediate,
            aligned_partition_units(&root, intermediate, 1, alignment)?,
            [
                ParameterMemberSpec::new(
                    format!("{root}.feed_forward.w1.weight"),
                    vec![intermediate, hidden],
                    MemberSharding::Partitioned { axis: 0 },
                ),
                ParameterMemberSpec::new(
                    format!("{root}.feed_forward.w3.weight"),
                    vec![intermediate, hidden],
                    MemberSharding::Partitioned { axis: 0 },
                ),
                ParameterMemberSpec::new(
                    format!("{root}.feed_forward.w2.weight"),
                    vec![hidden, intermediate],
                    MemberSharding::Partitioned { axis: 1 },
                ),
            ],
        )?);
    }
    let groups = eredu_runtime::expand_linear_format_parameter_groups(groups, |member| {
        let name = member.target();
        let linear = name.ends_with(".weight")
            && (name.contains("_proj.weight")
                || name.contains("feed_forward.w")
                || name == "lm_head.weight");
        linear
            .then(|| {
                crate::linear_format::standard_linear_format(
                    name,
                    args.weight_quantization_for(name).into(),
                )
                .map_err(|error| ParallelPlanError::InvalidTensor(error.to_string()))
            })
            .transpose()
    })?;
    let mut owned = Vec::with_capacity(groups.len());
    for (index, group) in groups.iter().cloned().enumerate() {
        let owner = if index < static_count {
            match index {
                0 if args.tie_word_embeddings => {
                    ParameterGroupOwner::static_any_of(["embedding", "output"])
                }
                0 => ParameterGroupOwner::static_role("embedding"),
                1 => ParameterGroupOwner::static_role("norm"),
                _ => ParameterGroupOwner::static_role("output"),
            }
        } else {
            let name = group.logical_name();
            let layer = name
                .strip_prefix("model.layers.")
                .and_then(|suffix| suffix.split('.').next())
                .and_then(|layer| layer.parse::<usize>().ok())
                .ok_or_else(|| {
                    ParallelPlanError::InvalidGroup(format!(
                        "LFM2 group {name:?} has no layer owner"
                    ))
                })?;
            ParameterGroupOwner::execution_unit(
                layout.group_id(0).expect("LFM2 group").clone(),
                layer,
            )
        };
        owned.push(OwnedParameterGroupSpec::new(owner, group));
    }
    ArchitectureParameterDescription::new(&graph, &layout, groups, owned)
        .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))
}

/// Declares vocabulary and replicated final-normalization groups.
pub fn static_parallel_parameter_groups<
    B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
>(
    modules: &StaticModules<B>,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    let mut groups = vec![
        module_parameter_group::<B::Tensor, _>(
            "model.embed_tokens",
            ParameterRole::Vocabulary,
            &modules.embeddings,
            |_, shape| {
                if shape.is_empty() {
                    Err(ParallelPlanError::InvalidTensor(
                        "LFM2 embedding parameter is scalar".into(),
                    ))
                } else {
                    Ok(MemberSharding::Balanced { axis: 0 })
                }
            },
        )?,
        module_parameter_group::<B::Tensor, _>(
            "model.embedding_norm",
            ParameterRole::Replicated,
            &modules.norm,
            |_, _| Ok(MemberSharding::Replicated),
        )?,
    ];
    if let Some(head) = &modules.lm_head {
        groups.push(module_parameter_group::<B::Tensor, _>(
            "lm_head",
            ParameterRole::Vocabulary,
            head,
            |_, shape| {
                if shape.is_empty() {
                    Err(ParallelPlanError::InvalidTensor(
                        "LFM2 output parameter is scalar".into(),
                    ))
                } else {
                    Ok(MemberSharding::Balanced { axis: 0 })
                }
            },
        )?);
    }
    Ok(groups)
}

/// Declares semantic groups for one scheduled LFM2 block.
pub fn layer_parallel_parameter_groups<
    B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
>(
    block: &Block<B>,
    args: &ModelArgs,
    layer: usize,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    let root = format!("model.layers.{layer}");
    let mut groups = Vec::new();
    match &block.mixer {
        TokenMixer::Attention(attention) => {
            let kv_heads = usize::try_from(args.num_key_value_heads).map_err(|_| {
                ParallelPlanError::InvalidGroup("LFM2 KV head count exceeds usize".into())
            })?;
            groups.push(partitioned_module_parameter_group::<B::Tensor, _>(
                format!("{root}.self_attn.heads"),
                ParameterRole::AttentionHeads,
                kv_heads,
                attention,
                |metadata, shape| {
                    let name = metadata.id.as_str();
                    if name.ends_with("q_proj.weight")
                        || name.ends_with("k_proj.weight")
                        || name.ends_with("v_proj.weight")
                    {
                        Ok(MemberSharding::Partitioned { axis: 0 })
                    } else if name.ends_with("out_proj.weight") && shape.len() >= 2 {
                        Ok(MemberSharding::Partitioned { axis: 1 })
                    } else {
                        Ok(MemberSharding::Replicated)
                    }
                },
            )?);
        }
        TokenMixer::ShortConvolution(convolution) => {
            let channels = usize::try_from(args.hidden_size).map_err(|_| {
                ParallelPlanError::InvalidGroup("LFM2 convolution width exceeds usize".into())
            })?;
            let segments = vec![
                0..channels,
                channels..2 * channels,
                2 * channels..3 * channels,
            ];
            groups.push(partitioned_module_parameter_group::<B::Tensor, _>(
                format!("{root}.conv.channels"),
                ParameterRole::Channels,
                channels,
                convolution,
                |metadata, shape| {
                    let name = metadata.id.as_str();
                    if name.contains("in_proj") {
                        Ok(MemberSharding::PartitionedSegments {
                            axis: 0,
                            segments: segments.clone(),
                        })
                    } else if name.ends_with("conv.weight") || name.ends_with("conv.bias") {
                        Ok(MemberSharding::Partitioned { axis: 0 })
                    } else if name.ends_with("out_proj.weight") && shape.len() >= 2 {
                        Ok(MemberSharding::Partitioned { axis: 1 })
                    } else {
                        Ok(MemberSharding::Replicated)
                    }
                },
            )?);
        }
    }
    for (name, norm) in [
        ("operator_norm", &block.operator_norm),
        ("ffn_norm", &block.feed_forward_norm),
    ] {
        groups.push(module_parameter_group::<B::Tensor, _>(
            format!("{root}.{name}"),
            ParameterRole::Replicated,
            norm,
            |_, _| Ok(MemberSharding::Replicated),
        )?);
    }
    match &block.feed_forward {
        FeedForward::Dense(DenseSwiGlu { gate, down, up }) => {
            let width = usize::try_from(args.dense_intermediate_size).map_err(|_| {
                ParallelPlanError::InvalidGroup("LFM2 dense width exceeds usize".into())
            })?;
            let alignment = args
                .weight_quantization_for(&format!("{root}.feed_forward.w2.weight"))
                .map_or(Ok(1), |quantization| {
                    usize::try_from(quantization.group_size()).map_err(|_| {
                        ParallelPlanError::InvalidGroup(
                            "LFM2 dense quantization group exceeds usize".into(),
                        )
                    })
                })?;
            groups.push(eredu_runtime::partitioned_projection_group::<
                B::Tensor,
                B::Linear,
            >(
                format!("{root}.feed_forward.intermediate"),
                ParameterRole::FeedForwardIntermediate,
                &[
                    (gate, eredu_runtime::ProjectionSharding::Column),
                    (up, eredu_runtime::ProjectionSharding::Column),
                    (down, eredu_runtime::ProjectionSharding::Row),
                ],
                aligned_partition_units(&root, width, 1, alignment)?,
            )?);
        }
        FeedForward::Routed(moe) => {
            groups.push(module_parameter_group::<B::Tensor, _>(
                format!("{root}.feed_forward.gate"),
                ParameterRole::Replicated,
                &moe.router,
                |_, _| Ok(MemberSharding::Replicated),
            )?);
            let width = usize::try_from(args.moe_intermediate_size).map_err(|_| {
                ParallelPlanError::InvalidGroup("LFM2 expert width exceeds usize".into())
            })?;
            let segments = vec![0..width, width..2 * width];
            groups.push(partitioned_module_parameter_group::<B::Tensor, _>(
                format!("{root}.feed_forward.experts.intermediate"),
                ParameterRole::ExpertIntermediate,
                width,
                &moe.experts,
                |metadata, _| {
                    if metadata.id.as_str().contains("gate_up_proj") {
                        Ok(MemberSharding::PartitionedSegments {
                            axis: 1,
                            segments: segments.clone(),
                        })
                    } else {
                        Ok(MemberSharding::Partitioned { axis: 2 })
                    }
                },
            )?);
        }
    }
    Ok(groups)
}

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_runtime::{LocalTensorLayout, ParameterRole};

    fn args() -> ModelArgs {
        crate::lfm2::model_args_from_config_value(&serde_json::json!({
            "model_type":"lfm2_moe","vocab_size":16,"hidden_size":12,
            "intermediate_size":18,"num_hidden_layers":3,
            "num_attention_heads":3,"num_key_value_heads":3,
            "max_position_embeddings":64,"layer_types":["conv","full_attention","conv"],
            "conv_L_cache":3,"block_auto_adjust_ff_dim":false,
            "num_dense_layers":1,"moe_intermediate_size":8,
            "num_experts":2,"num_experts_per_tok":1,"tie_word_embeddings":false
        }))
        .unwrap()
    }

    #[allow(clippy::too_many_arguments)]
    fn insert(
        layout: &mut LocalModelLayout,
        target: &str,
        logical: &str,
        global: Vec<usize>,
        local: Vec<usize>,
        placement: TensorPlacement,
    ) {
        layout.insert(
            target.into(),
            LocalTensorLayout::new(
                logical,
                ParameterRole::Replicated,
                global,
                local,
                placement,
                None,
                None,
                false,
            ),
        );
    }

    fn range(axis: usize, end: usize) -> TensorPlacement {
        TensorPlacement::Range {
            axis,
            start: 0,
            end,
        }
    }

    fn valid_layout() -> LocalModelLayout {
        let mut layout = LocalModelLayout::default();
        insert(
            &mut layout,
            "model.embed_tokens.weight",
            "model.embed_tokens",
            vec![16, 12],
            vec![8, 12],
            range(0, 8),
        );
        insert(
            &mut layout,
            "lm_head.weight",
            "lm_head",
            vec![16, 12],
            vec![8, 12],
            range(0, 8),
        );
        insert(
            &mut layout,
            "model.layers.0.conv.conv.weight",
            "model.layers.0.conv.channels",
            vec![12, 1, 3],
            vec![8, 1, 3],
            range(0, 8),
        );
        insert(
            &mut layout,
            "model.layers.0.feed_forward.w1.weight",
            "model.layers.0.feed_forward.intermediate",
            vec![18, 12],
            vec![9, 12],
            range(0, 9),
        );
        insert(
            &mut layout,
            "model.layers.1.self_attn.q_proj.weight",
            "model.layers.1.self_attn.heads",
            vec![12, 12],
            vec![8, 12],
            range(0, 8),
        );
        insert(
            &mut layout,
            "model.layers.1.self_attn.k_proj.weight",
            "model.layers.1.self_attn.heads",
            vec![12, 12],
            vec![8, 12],
            range(0, 8),
        );
        insert(
            &mut layout,
            "model.layers.1.feed_forward.experts.gate_up_proj",
            "model.layers.1.feed_forward.experts.intermediate",
            vec![2, 16, 12],
            vec![2, 10, 12],
            range(1, 10),
        );
        insert(
            &mut layout,
            "model.layers.2.conv.conv.weight",
            "model.layers.2.conv.channels",
            vec![12, 1, 3],
            vec![8, 1, 3],
            range(0, 8),
        );
        insert(
            &mut layout,
            "model.layers.2.feed_forward.experts.gate_up_proj",
            "model.layers.2.feed_forward.experts.intermediate",
            vec![2, 16, 12],
            vec![2, 10, 12],
            range(1, 10),
        );
        layout
    }

    fn routed_layout() -> LocalModelLayout {
        let mut layout = valid_layout();
        for unit in [1, 2] {
            let target = format!("model.layers.{unit}.feed_forward.experts.gate_up_proj");
            insert_logical(
                &mut layout,
                &target,
                &format!("model.layers.{unit}.feed_forward.experts.intermediate"),
                vec![2, 16, 12],
                vec![2, 10, 12],
                range(1, 10),
                8,
                0..5,
            );
        }
        layout
    }

    #[allow(clippy::too_many_arguments)]
    fn insert_logical(
        layout: &mut LocalModelLayout,
        target: &str,
        logical: &str,
        global: Vec<usize>,
        local: Vec<usize>,
        placement: TensorPlacement,
        logical_units: usize,
        logical_range: Range<usize>,
    ) {
        layout.insert(
            target.into(),
            LocalTensorLayout::new(
                logical,
                ParameterRole::ExpertIntermediate,
                global,
                local,
                placement,
                Some(logical_units),
                Some(logical_range),
                false,
            ),
        );
    }

    fn realization(
        topology: eredu_core::ParallelRankTopology,
    ) -> crate::ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec> {
        let args = args();
        let local_count = i32::try_from(
            eredu_core::balanced_contiguous_range(
                2,
                topology.expert_parallel_size(),
                topology.expert_parallel_rank(),
                false,
            )
            .unwrap()
            .len(),
        )
        .unwrap();
        let group = eredu_runtime::ExecutionGroupId::new("target").unwrap();
        let specs = [1, 2]
            .into_iter()
            .map(|unit| {
                (
                    (group.clone(), unit),
                    crate::lfm2::moe::expert_bank_spec(&args, unit)
                        .unwrap()
                        .with_group_geometry(local_count, 5)
                        .unwrap(),
                )
            })
            .collect();
        crate::ExpertRealizationPlan::balanced(2, topology, specs).unwrap()
    }

    #[test]
    fn local_geometry_owns_blocks_vocabulary_and_state_together() {
        let args = args();
        let geometry = local_geometry(&args, &valid_layout()).unwrap();
        assert_eq!(geometry.blocks().len(), 3);
        assert_eq!(geometry.block(0).unwrap().convolution_channels, 8);
        assert_eq!(geometry.block(0).unwrap().dense_intermediate, 9);
        assert_eq!(geometry.block(1).unwrap().query_heads, 2);
        assert_eq!(geometry.block(1).unwrap().key_value_heads, 2);
        assert_eq!(geometry.block(1).unwrap().expert_intermediate, 5);
        assert_eq!(geometry.block(2).unwrap().convolution_channels, 8);
        assert_eq!(geometry.embedding_range().local, 0..8);
        assert_eq!(geometry.output_range().unwrap().local, 0..8);
        assert_ne!(
            geometry.state_layout(),
            &crate::lfm2::state_layout(&args).unwrap()
        );
        geometry.validate_for(&args).unwrap();
    }

    #[test]
    fn local_geometry_rejects_incomplete_heads_and_vocabulary_drift() {
        let args = args();
        let mut layout = valid_layout();
        insert(
            &mut layout,
            "model.layers.1.self_attn.q_proj.weight",
            "model.layers.1.self_attn.heads",
            vec![12, 12],
            vec![7, 12],
            range(0, 7),
        );
        assert!(local_geometry(&args, &layout)
            .unwrap_err()
            .to_string()
            .contains("split head dimension"));

        let mut layout = valid_layout();
        insert(
            &mut layout,
            "model.embed_tokens.scales",
            "model.embed_tokens",
            vec![16, 1],
            vec![8, 1],
            TensorPlacement::Range {
                axis: 0,
                start: 8,
                end: 16,
            },
        );
        assert!(local_geometry(&args, &layout)
            .unwrap_err()
            .to_string()
            .contains("inconsistent companion selections"));
    }

    #[test]
    fn tied_geometry_uses_embedding_ownership_for_output() {
        let mut args = args();
        args.tie_word_embeddings = true;
        let geometry = local_geometry(&args, &valid_layout()).unwrap();
        assert!(geometry.output_range().is_none());
        geometry.validate_for(&args).unwrap();
    }

    #[test]
    fn routed_partition_binds_exact_pp_ep_tp_bank_ownership() {
        let shape = eredu_core::ParallelTopology::new(2, 2, 2, 1).unwrap();
        let topology = eredu_core::ParallelRankTopology::new(shape, 1).unwrap();
        let plan = realization(topology);
        let geometry =
            partition_local_routed_geometry(&args(), &routed_layout(), 0..2, topology, &plan)
                .unwrap();

        assert_eq!(geometry.owned_units(), 0..2);
        assert_eq!(geometry.static_roles(), ["embedding"]);
        assert_eq!(
            geometry.boundary_schema(),
            &eredu_runtime::NoAuxiliaryBoundarySchema::new(12)
        );
        assert_eq!(geometry.expert_intermediate_range(), Some(0..5));
        assert_eq!(geometry.expert_banks().len(), 1);
        assert_eq!(geometry.state_global_offset(), 0);
        assert_eq!(geometry.local_state_layout().unwrap().len(), 2);
        let ownership = eredu_runtime::PartitionOwnership::new(true, false, ["embedding"]).unwrap();
        geometry
            .validate_partition_contract(&ownership, geometry.boundary_schema())
            .unwrap();
        assert!(geometry
            .validate_partition_contract(
                &ownership,
                &eredu_runtime::NoAuxiliaryBoundarySchema::new(13)
            )
            .is_err());
        let bank = &geometry.expert_banks()[0];
        assert_eq!(bank.global_unit(), 1);
        assert_eq!(bank.global_expert(), 1);
        assert_eq!(bank.owner_local_expert(), 0);
        assert_eq!(bank.bank_key(), eredu_runtime::ParameterBankKey::new(1, 1));
        assert_eq!(geometry.complete_state_layout().len(), 3);

        let wrong_topology = eredu_core::ParallelRankTopology::new(shape, 0).unwrap();
        let wrong = realization(wrong_topology);
        assert!(
            partition_local_routed_geometry(&args(), &routed_layout(), 0..2, topology, &wrong)
                .is_err()
        );
        assert!(
            partition_local_routed_geometry(&args(), &routed_layout(), 1..3, topology, &plan)
                .is_err()
        );
    }
}
