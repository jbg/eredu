//! Semantic tensor-parallel placement for Kimi physical blocks.

use eredu_nn::{BlockwiseAttentionBackend, GroupedNeuralBackend, VocabularyParallelRange};
use eredu_runtime::{
    aligned_partition_units, module_parameter_group, partitioned_module_parameter_group,
    LocalModelLayout, MemberSharding, ParallelPlanError, ParameterGroupSpec, ParameterRole,
    StateLayout, TensorPlacement,
};
use std::ops::Range;

use crate::decoder::StaticModules;

use super::{
    prompt_cache_architecture_fingerprint, state_layout_with_geometry, AttentionKind, Block,
    BlockGeometry, DenseSwiGlu, FeedForward, FeedForwardPolicy, LayerCacheGeometry, ModelArgs,
    TokenMixer,
};

/// Complete planner-derived construction and state geometry for one Kimi rank.
///
/// Fields are deliberately private so unit widths, vocabulary ownership, and
/// mutable-state geometry can only be produced together from one typed local
/// layout.
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

/// Exact TP-local and PP-local geometry for one dense Kimi partition.
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
    shared_expert_intermediate_range: Option<Range<usize>>,
    expert_banks: Vec<PartitionExpertBankOwnership>,
}

/// One compact routed Kimi bank at the PP×EP×TP ownership intersection.
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
    /// Architecture-global block range physically owned by this partition.
    pub fn owned_units(&self) -> Range<usize> {
        self.owned_units.clone()
    }

    /// Resolves an owned global block index to its local construction geometry.
    pub fn block(&self, global_unit: usize) -> Option<&BlockGeometry> {
        self.owned_units
            .contains(&global_unit)
            .then(|| &self.blocks[global_unit - self.owned_units.start])
    }

    /// Number of block configurations physically retained here.
    pub fn local_unit_count(&self) -> usize {
        self.blocks.len()
    }

    /// Tensor-coordinate vocabulary ownership for input embeddings.
    pub const fn embedding_range(&self) -> &VocabularyParallelRange {
        &self.embedding_range
    }

    /// Tensor-coordinate vocabulary ownership for an untied output head.
    pub const fn output_range(&self) -> Option<&VocabularyParallelRange> {
        self.output_range.as_ref()
    }

    /// Complete TP-local heterogeneous state before PP slicing.
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
                "Kimi partition ownership or boundary drifted from local geometry".into(),
            ));
        }
        Ok(())
    }

    /// Borrows the immutable routed-expert realization retained by this geometry.
    pub const fn expert_realization(
        &self,
    ) -> Option<&crate::ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>> {
        self.expert_realization.as_ref()
    }
    /// Returns this TP rank's routed-intermediate interval when routed.
    pub fn expert_intermediate_range(&self) -> Option<Range<usize>> {
        self.expert_intermediate_range.clone()
    }
    /// Returns this TP rank's replicated-over-EP shared-expert interval.
    pub fn shared_expert_intermediate_range(&self) -> Option<Range<usize>> {
        self.shared_expert_intermediate_range.clone()
    }
    /// Borrows the exact PP×EP×TP compact expert-bank ownership.
    pub fn expert_banks(&self) -> &[PartitionExpertBankOwnership] {
        &self.expert_banks
    }

    pub(super) fn validate_for(&self, args: &ModelArgs) -> Result<(), ParallelPlanError> {
        args.validate()
            .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
        if args.has_sparse_moe_layers() != self.expert_realization.is_some() {
            return Err(ParallelPlanError::InvalidGroup(
                "partition-local Kimi routed authority differs from its schedule".into(),
            ));
        }
        let count = usize::try_from(args.num_hidden_layers).map_err(|_| {
            ParallelPlanError::InvalidGroup("Kimi layer count exceeds usize".into())
        })?;
        if self.owned_units.is_empty()
            || self.owned_units.end > count
            || self.blocks.len() != self.owned_units.len()
            || self.complete_state_layout.len() != count
            || self.architecture_fingerprint != prompt_cache_architecture_fingerprint(args)
            || self.tied_head != args.tie_word_embeddings
            || self.static_roles != expected_static_roles(&self.owned_units, count)
            || self.boundary_schema
                != eredu_runtime::NoAuxiliaryBoundarySchema::new(args.hidden_size)
        {
            return Err(ParallelPlanError::InvalidGroup(
                "partition-local Kimi geometry belongs to a different model or range".into(),
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
                    "partition-local Kimi output vocabulary ownership is inconsistent".into(),
                ))
            }
        }
        for global in self.owned_units.clone() {
            let block = self.block(global).expect("validated owned Kimi block");
            let policy = args.layer_policy(global).ok_or_else(|| {
                ParallelPlanError::InvalidGroup(format!("Kimi has no layer {global}"))
            })?;
            let state = self.complete_state_layout.layer(global).ok_or_else(|| {
                ParallelPlanError::InvalidGroup("missing Kimi complete state layer".into())
            })?;
            let matches = match (policy.attention, state) {
                (
                    AttentionKind::Kda,
                    eredu_core::cache::LayerCachePolicy::FixedState { tensors },
                ) => {
                    let width = block.kda_heads.checked_mul(args.kda_config.head_dim);
                    tensors.len() == 4
                        && width.is_some_and(|width| {
                            tensors[..3].iter().all(|tensor| {
                                tensor.dtype == eredu_core::cache::StateTensorDtype::Floating
                                    && matches!(
                                        tensor.shape.get(2),
                                        Some(eredu_core::cache::StateTensorDimension::Fixed(value))
                                            if i32::try_from(value.get()) == Ok(width)
                                    )
                            })
                        })
                        && tensors.last().is_some_and(|tensor| {
                            tensor.dtype == eredu_core::cache::StateTensorDtype::Float32
                                && matches!(
                                    tensor.shape.get(1),
                                    Some(eredu_core::cache::StateTensorDimension::Fixed(value))
                                        if i32::try_from(value.get()) == Ok(block.kda_heads)
                                )
                        })
                }
                (
                    AttentionKind::Mla,
                    eredu_core::cache::LayerCachePolicy::CompressedLatentRotary {
                        latent_dim,
                        rotary_dim,
                        ..
                    },
                ) => {
                    i32::try_from(latent_dim.get()) == Ok(args.kv_lora_rank)
                        && i32::try_from(rotary_dim.get()) == Ok(args.qk_rope_head_dim)
                }
                _ => false,
            };
            if !matches {
                return Err(ParallelPlanError::InvalidGroup(format!(
                    "owned Kimi block {global} differs from complete TP-local state geometry"
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

    /// Returns the authoritative state layout derived from these blocks.
    pub const fn state_layout(&self) -> &StateLayout {
        &self.state_layout
    }

    pub(super) fn validate_for(&self, args: &ModelArgs) -> Result<(), ParallelPlanError> {
        let expected_layers = usize::try_from(args.num_hidden_layers).map_err(|_| {
            ParallelPlanError::InvalidGroup("Kimi layer count exceeds usize".into())
        })?;
        if self.blocks.len() != expected_layers
            || self.architecture_fingerprint != prompt_cache_architecture_fingerprint(args)
            || self.global_block_geometry != BlockGeometry::replicated(args)
            || self.tied_head != args.tie_word_embeddings
        {
            return Err(ParallelPlanError::InvalidGroup(
                "rank-local Kimi geometry belongs to a different model configuration".into(),
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
                    "tied Kimi output unexpectedly owns a separate vocabulary range".into(),
                ))
            }
            (false, None) => {
                return Err(ParallelPlanError::InvalidGroup(
                    "untied Kimi output is missing vocabulary ownership".into(),
                ))
            }
        }
        let expected = state_layout_with_geometry(args, &state_geometry(args, &self.blocks))
            .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
        if expected != self.state_layout {
            return Err(ParallelPlanError::InvalidGroup(
                "rank-local Kimi state layout drifted from block geometry".into(),
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
        ParallelPlanError::InvalidTensor(format!("missing local Kimi layout for {name}"))
    })?;
    let local = *tensor.local_shape().get(axis).ok_or_else(|| {
        ParallelPlanError::InvalidTensor(format!("Kimi tensor {name} has no axis {axis}"))
    })?;
    let global = *tensor.global_shape().get(axis).ok_or_else(|| {
        ParallelPlanError::InvalidTensor(format!("Kimi tensor {name} has no global axis {axis}"))
    })?;
    if local == 0 || local > global {
        return Err(ParallelPlanError::InvalidTensor(format!(
            "Kimi tensor {name} has invalid local width {local} of global width {global}"
        )));
    }
    i32::try_from(local)
        .map_err(|_| ParallelPlanError::InvalidTensor(format!("Kimi width for {name} exceeds i32")))
}

/// Derives one block's local widths exclusively from resolved placement.
pub fn local_block_geometry(
    args: &ModelArgs,
    layer: usize,
    layout: &LocalModelLayout,
) -> Result<BlockGeometry, ParallelPlanError> {
    let root = format!("model.layers.{layer}");
    let policy = args
        .layer_policy(layer)
        .ok_or_else(|| ParallelPlanError::InvalidGroup(format!("Kimi has no layer {layer}")))?;
    let mut geometry = BlockGeometry::replicated(args);
    match policy.attention {
        AttentionKind::Kda => {
            let width = local_width(layout, &format!("{root}.self_attn.q_proj.weight"), 0)?;
            if width % args.kda_config.head_dim != 0 {
                return Err(ParallelPlanError::InvalidTensor(
                    "local KDA width does not contain complete heads".into(),
                ));
            }
            geometry.kda_heads = width / args.kda_config.head_dim;
        }
        AttentionKind::Mla => {
            let query_name = if args.q_lora_rank.is_some() {
                format!("{root}.self_attn.q_b_proj.weight")
            } else {
                format!("{root}.self_attn.q_proj.weight")
            };
            let width = local_width(layout, &query_name, 0)?;
            let head = args.qk_nope_head_dim + args.qk_rope_head_dim;
            if width % head != 0 {
                return Err(ParallelPlanError::InvalidTensor(
                    "local MLA width does not contain complete heads".into(),
                ));
            }
            geometry.mla_heads = width / head;
        }
    }
    match policy.feed_forward {
        FeedForwardPolicy::Dense => {
            geometry.dense_intermediate =
                local_width(layout, &format!("{root}.mlp.gate_proj.weight"), 0)?;
        }
        FeedForwardPolicy::SparseMoe => {
            let fused = local_width(layout, &format!("{root}.mlp.experts.gate_up_proj"), 1)?;
            if fused % 2 != 0 {
                return Err(ParallelPlanError::InvalidTensor(
                    "local Kimi packed expert width is not even".into(),
                ));
            }
            geometry.routed_intermediate = fused / 2;
            geometry.shared_intermediate = local_width(
                layout,
                &format!("{root}.mlp.shared_experts.gate_proj.weight"),
                0,
            )?;
        }
    }
    Ok(geometry)
}

/// Returns rank-local KDA state geometry for every physical layer.
pub fn local_state_geometry(
    args: &ModelArgs,
    layout: &LocalModelLayout,
) -> Result<Vec<LayerCacheGeometry>, ParallelPlanError> {
    let blocks = (0..args.num_hidden_layers as usize)
        .map(|layer| local_block_geometry(args, layer, layout))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(state_geometry(args, &blocks))
}

/// Derives the complete TP-local heterogeneous state before pipeline slicing.
///
/// KDA recurrent and convolutional state follows the locally owned attention
/// heads, while MLA compressed latent/rotary state is head-independent.  This
/// projection is used during admission, before any backend module exists.
pub fn partitioned_state_layout(
    args: &ModelArgs,
    tensor_rank: usize,
    tensor_size: usize,
) -> Result<StateLayout, ParallelPlanError> {
    args.validate()
        .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
    if tensor_size == 0 || tensor_rank >= tensor_size {
        return Err(ParallelPlanError::InvalidGroup(format!(
            "invalid Kimi tensor coordinate {tensor_rank}/{tensor_size}"
        )));
    }
    let kda_heads = usize::try_from(args.kda_config.num_heads)
        .map_err(|_| ParallelPlanError::InvalidGroup("Kimi KDA head count exceeds usize".into()))?;
    let local_kda_heads =
        eredu_core::balanced_contiguous_range(kda_heads, tensor_size, tensor_rank, false)
            .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?
            .len();
    let local_kda_heads = i32::try_from(local_kda_heads).map_err(|_| {
        ParallelPlanError::InvalidGroup("local Kimi KDA head count exceeds i32".into())
    })?;
    let geometry = args
        .layer_schedule
        .iter()
        .map(|policy| LayerCacheGeometry {
            kda_heads: (policy.attention == AttentionKind::Kda).then_some(local_kda_heads),
        })
        .collect::<Vec<_>>();
    state_layout_with_geometry(args, &geometry)
        .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))
}

fn state_geometry(args: &ModelArgs, blocks: &[BlockGeometry]) -> Vec<LayerCacheGeometry> {
    args.layer_schedule
        .iter()
        .zip(blocks)
        .map(|(policy, block)| LayerCacheGeometry {
            kda_heads: (policy.attention == AttentionKind::Kda).then_some(block.kda_heads),
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
                "Kimi vocabulary member {target} has global shape {:?}, expected {global_vocabulary} rows",
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
                    "Kimi vocabulary member {target} has non-row placement {placement:?}"
                )))
            }
        };
        if selected.as_ref().is_some_and(|current| current != &range) {
            return Err(ParallelPlanError::InvalidTensor(format!(
                "Kimi vocabulary group {logical_name} has inconsistent companion selections"
            )));
        }
        selected = Some(range);
    }
    if !found {
        return Err(ParallelPlanError::InvalidTensor(format!(
            "missing local Kimi vocabulary layout for {logical_name}"
        )));
    }
    let range = VocabularyParallelRange {
        global_vocabulary,
        local: selected.expect("a found Kimi vocabulary member supplies a selection"),
    };
    range
        .validate()
        .map_err(|error| ParallelPlanError::InvalidTensor(error.to_string()))?;
    Ok(range)
}

/// Derives complete rank-local Kimi construction geometry from one typed plan.
pub fn local_geometry(
    args: &ModelArgs,
    layout: &LocalModelLayout,
) -> Result<LocalGeometry, ParallelPlanError> {
    args.validate()
        .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
    let layers = usize::try_from(args.num_hidden_layers)
        .map_err(|_| ParallelPlanError::InvalidGroup("Kimi layer count exceeds usize".into()))?;
    let blocks = (0..layers)
        .map(|layer| local_block_geometry(args, layer, layout))
        .collect::<Result<Vec<_>, _>>()?;
    let state_layout = state_layout_with_geometry(args, &state_geometry(args, &blocks))
        .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
    let global_vocabulary = usize::try_from(args.vocab_size).map_err(|_| {
        ParallelPlanError::InvalidGroup("Kimi vocabulary size exceeds usize".into())
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

/// Derives dense Kimi construction geometry for one exact TP/PP partition.
pub fn partition_local_geometry(
    args: &ModelArgs,
    layout: &LocalModelLayout,
    owned_units: Range<usize>,
) -> Result<PartitionLocalGeometry, ParallelPlanError> {
    if args.has_sparse_moe_layers() {
        return Err(ParallelPlanError::InvalidGroup(
            "partitioned resident Kimi does not accept routed feed-forward layers".into(),
        ));
    }
    let count = usize::try_from(args.num_hidden_layers)
        .map_err(|_| ParallelPlanError::InvalidGroup("Kimi layer count exceeds usize".into()))?;
    if owned_units.is_empty() || owned_units.end > count {
        return Err(ParallelPlanError::InvalidGroup(format!(
            "Kimi local unit range {owned_units:?} is outside {count} layers"
        )));
    }
    let complete = local_geometry(args, layout)?;
    let blocks = owned_units
        .clone()
        .map(|unit| {
            complete.block(unit).copied().ok_or_else(|| {
                ParallelPlanError::InvalidGroup(format!("Kimi has no local block {unit}"))
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
        shared_expert_intermediate_range: None,
        expert_banks: Vec::new(),
    };
    geometry.validate_for(args)?;
    Ok(geometry)
}

/// Derives prediction-free routed Kimi geometry from the selected expert plan.
pub fn partition_local_routed_geometry(
    args: &ModelArgs,
    layout: &LocalModelLayout,
    owned_units: Range<usize>,
    topology: eredu_core::ParallelRankTopology,
    realization: &crate::ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>,
) -> Result<PartitionLocalGeometry, ParallelPlanError> {
    if args.num_nextn_predict_layers != 0 {
        return Err(ParallelPlanError::InvalidGroup(
            "partition-local Kimi rejects embedded prediction".into(),
        ));
    }
    if !args.has_sparse_moe_layers() {
        return Err(ParallelPlanError::InvalidGroup(
            "routed Kimi requires sparse units".into(),
        ));
    }
    let count = usize::try_from(args.num_hidden_layers)
        .map_err(|_| ParallelPlanError::InvalidGroup("Kimi layer count exceeds usize".into()))?;
    let expected_units = eredu_core::balanced_contiguous_range(
        count,
        topology.pipeline_parallel_size(),
        topology.pipeline_parallel_rank(),
        false,
    )
    .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
    if owned_units.is_empty() || owned_units.end > count || owned_units != expected_units {
        return Err(ParallelPlanError::InvalidGroup(
            "routed Kimi PP ownership differs from its Cartesian rank".into(),
        ));
    }
    let complete = local_geometry(args, layout)?;
    let global_experts = usize::try_from(args.num_experts)
        .map_err(|_| ParallelPlanError::InvalidGroup("Kimi expert count exceeds usize".into()))?;
    let local_experts = validate_realization(topology, global_experts, realization)?;
    let local_count = i32::try_from(local_experts.len()).map_err(|_| {
        ParallelPlanError::InvalidGroup("Kimi local expert count exceeds i32".into())
    })?;
    let mut intermediate_range = None;
    let mut shared_range = None;
    let mut expected_sparse_units = 0;
    for unit in 0..count {
        if args.layer_policy(unit).map(|policy| policy.feed_forward)
            != Some(FeedForwardPolicy::SparseMoe)
        {
            continue;
        }
        expected_sparse_units += 1;
        let block = complete.block(unit).ok_or_else(|| {
            ParallelPlanError::InvalidGroup("missing local Kimi sparse block".into())
        })?;
        let expected = super::moe::expert_bank_spec(args, unit)
            .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?
            .with_group_geometry(local_count, block.routed_intermediate)
            .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
        if realization.unit_spec(crate::decoder::TARGET_EXECUTION_GROUP, unit) != Some(&expected) {
            return Err(ParallelPlanError::InvalidGroup(format!(
                "Kimi expert plan drifted at unit {unit}"
            )));
        }
        let target = format!("model.layers.{unit}.mlp.experts.gate_up_proj");
        let range = exact_logical_range(layout, &target, args.moe_intermediate_size)?;
        if intermediate_range
            .as_ref()
            .is_some_and(|current| current != &range)
        {
            return Err(ParallelPlanError::InvalidTensor(
                "Kimi expert TP ranges differ".into(),
            ));
        }
        intermediate_range = Some(range);
        let shared = exact_logical_range(
            layout,
            &format!("model.layers.{unit}.mlp.shared_experts.gate_proj.weight"),
            args.moe_intermediate_size
                .checked_mul(args.num_shared_experts)
                .ok_or_else(|| {
                    ParallelPlanError::InvalidTensor(
                        "Kimi shared-expert width overflowed i32".into(),
                    )
                })?,
        )?;
        if i32::try_from(shared.len()) != Ok(block.shared_intermediate) {
            return Err(ParallelPlanError::InvalidTensor(
                "Kimi shared-expert TP interval differs from local block width".into(),
            ));
        }
        if shared_range
            .as_ref()
            .is_some_and(|current| current != &shared)
        {
            return Err(ParallelPlanError::InvalidTensor(
                "Kimi shared-expert TP ranges differ between units".into(),
            ));
        }
        shared_range = Some(shared);
    }
    if realization.unit_specs().len() != expected_sparse_units {
        return Err(ParallelPlanError::InvalidGroup(
            "Kimi expert unit schedule drifted".into(),
        ));
    }
    let intermediate_range = intermediate_range
        .ok_or_else(|| ParallelPlanError::InvalidGroup("Kimi has no routed TP range".into()))?;
    let shared_range = shared_range.ok_or_else(|| {
        ParallelPlanError::InvalidGroup("Kimi has no shared-expert TP range".into())
    })?;
    let blocks = owned_units
        .clone()
        .map(|unit| {
            complete
                .block(unit)
                .copied()
                .ok_or_else(|| ParallelPlanError::InvalidGroup("missing owned Kimi block".into()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let expert_banks = owned_units
        .clone()
        .filter(|unit| {
            args.layer_policy(*unit)
                .is_some_and(|policy| policy.feed_forward == FeedForwardPolicy::SparseMoe)
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
        shared_expert_intermediate_range: Some(shared_range),
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
            "Kimi expert owner differs from Cartesian rank".into(),
        ));
    }
    Ok(local)
}

/// Declares vocabulary and replicated final-normalization groups.
pub fn static_parallel_parameter_groups<B>(
    modules: &StaticModules<B>,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError>
where
    B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend + BlockwiseAttentionBackend,
{
    let mut groups = vec![
        module_parameter_group::<B::Tensor, _>(
            "model.embed_tokens",
            ParameterRole::Vocabulary,
            &modules.embeddings,
            |_, shape| {
                (!shape.is_empty())
                    .then_some(MemberSharding::Balanced { axis: 0 })
                    .ok_or_else(|| {
                        ParallelPlanError::InvalidTensor("Kimi embedding is scalar".into())
                    })
            },
        )?,
        module_parameter_group::<B::Tensor, _>(
            "model.norm",
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
                (!shape.is_empty())
                    .then_some(MemberSharding::Balanced { axis: 0 })
                    .ok_or_else(|| ParallelPlanError::InvalidTensor("Kimi output is scalar".into()))
            },
        )?);
    }
    Ok(groups)
}

/// Declares semantic groups for one scheduled Kimi block.
pub fn layer_parallel_parameter_groups<B>(
    block: &Block<B>,
    args: &ModelArgs,
    layer: usize,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError>
where
    B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend + BlockwiseAttentionBackend,
{
    let root = format!("model.layers.{layer}");
    let mut groups = Vec::new();
    let (units, role) = match &block.mixer {
        TokenMixer::Kda(_) => (args.kda_config.num_heads, ParameterRole::AttentionHeads),
        TokenMixer::Mla(_) => (args.num_attention_heads, ParameterRole::AttentionHeads),
    };
    groups.push(partitioned_module_parameter_group::<B::Tensor, _>(
        format!("{root}.self_attn.heads"),
        role,
        usize::try_from(units)
            .map_err(|_| ParallelPlanError::InvalidGroup("Kimi head count exceeds usize".into()))?,
        &block.mixer,
        |metadata, shape| {
            let name = metadata.id.as_str();
            if name.ends_with("q_proj.weight")
                || name.ends_with("k_proj.weight")
                || name.ends_with("v_proj.weight")
                || name.ends_with("q_b_proj.weight")
                || name.ends_with("kv_b_proj.weight")
                || name.ends_with("k_b_proj.weight")
                || name.ends_with("v_b_proj.weight")
                || name.ends_with("f_b_proj.weight")
                || name.ends_with("b_proj.weight")
                || name.ends_with("g_b_proj.weight")
                || name.ends_with("dt_bias")
                || name.contains("conv1d.weight")
            {
                Ok(MemberSharding::Partitioned { axis: 0 })
            } else if name.ends_with("A_log") && shape.len() >= 3 {
                Ok(MemberSharding::Partitioned { axis: 2 })
            } else if name.ends_with("o_proj.weight") && shape.len() >= 2 {
                Ok(MemberSharding::Partitioned { axis: 1 })
            } else {
                Ok(MemberSharding::Replicated)
            }
        },
    )?);
    for (name, norm) in [
        ("input_layernorm", &block.input_norm),
        ("post_attention_layernorm", &block.post_attention_norm),
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
            let width = usize::try_from(args.intermediate_size).map_err(|_| {
                ParallelPlanError::InvalidGroup("Kimi dense width exceeds usize".into())
            })?;
            groups.push(eredu_runtime::partitioned_projection_group::<
                B::Tensor,
                B::Linear,
            >(
                format!("{root}.mlp.intermediate"),
                ParameterRole::FeedForwardIntermediate,
                &[
                    (gate, eredu_runtime::ProjectionSharding::Column),
                    (up, eredu_runtime::ProjectionSharding::Column),
                    (down, eredu_runtime::ProjectionSharding::Row),
                ],
                aligned_partition_units(&root, width, 1, 1)?,
            )?);
        }
        FeedForward::Sparse(moe) => {
            groups.push(module_parameter_group::<B::Tensor, _>(
                format!("{root}.mlp.gate"),
                ParameterRole::Replicated,
                &moe.router,
                |_, _| Ok(MemberSharding::Replicated),
            )?);
            let width = usize::try_from(args.moe_intermediate_size).map_err(|_| {
                ParallelPlanError::InvalidGroup("Kimi expert width exceeds usize".into())
            })?;
            let segments = vec![0..width, width..2 * width];
            groups.push(partitioned_module_parameter_group::<B::Tensor, _>(
                format!("{root}.mlp.experts.intermediate"),
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
            groups.push(eredu_runtime::partitioned_projection_group::<
                B::Tensor,
                B::Linear,
            >(
                format!("{root}.mlp.shared_experts.intermediate"),
                ParameterRole::FeedForwardIntermediate,
                &[
                    (&moe.shared.gate, eredu_runtime::ProjectionSharding::Column),
                    (&moe.shared.up, eredu_runtime::ProjectionSharding::Column),
                    (&moe.shared.down, eredu_runtime::ProjectionSharding::Row),
                ],
                aligned_partition_units(&root, width, 1, 1)?,
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
        crate::kimi_linear::model_args_from_config_value(&serde_json::json!({
            "model_type":"kimi_linear","vocab_size":16,"hidden_size":12,"num_hidden_layers":2,
            "num_attention_heads":3,"num_key_value_heads":3,"intermediate_size":17,"head_dim":4,
            "model_max_length":64,"linear_attn_config":{"kda_layers":[1],"full_attn_layers":[2],"num_heads":3,"head_dim":4,"short_conv_kernel_size":3},
            "num_experts":2,"moe_intermediate_size":9,"kv_lora_rank":6,"qk_nope_head_dim":4,"qk_rope_head_dim":2,"v_head_dim":4,
            "mla_use_nope":true,"num_experts_per_token":1,"num_shared_experts":1,"routed_scaling_factor":1.0,
            "first_k_dense_replace":1,"num_expert_group":1,"topk_group":1
        }))
        .unwrap()
    }

    fn dense_args() -> ModelArgs {
        crate::kimi_linear::model_args_from_config_value(&serde_json::json!({
            "model_type":"kimi_linear","vocab_size":16,"hidden_size":12,"num_hidden_layers":2,
            "num_attention_heads":3,"num_key_value_heads":3,"intermediate_size":17,"head_dim":4,
            "model_max_length":64,"linear_attn_config":{"kda_layers":[1],"full_attn_layers":[2],"num_heads":3,"head_dim":4,"short_conv_kernel_size":3},
            "num_experts":2,"moe_intermediate_size":9,"kv_lora_rank":6,"qk_nope_head_dim":4,"qk_rope_head_dim":2,"v_head_dim":4,
            "mla_use_nope":true,"num_experts_per_token":1,"num_shared_experts":1,"routed_scaling_factor":1.0,
            "first_k_dense_replace":2,"num_expert_group":1,"topk_group":1
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

    fn valid_layout() -> LocalModelLayout {
        let mut layout = LocalModelLayout::default();
        insert(
            &mut layout,
            "model.embed_tokens.weight",
            "model.embed_tokens",
            vec![16, 12],
            vec![8, 12],
            TensorPlacement::Range {
                axis: 0,
                start: 0,
                end: 8,
            },
        );
        insert(
            &mut layout,
            "lm_head.weight",
            "lm_head",
            vec![16, 12],
            vec![8, 12],
            TensorPlacement::Range {
                axis: 0,
                start: 0,
                end: 8,
            },
        );
        insert(
            &mut layout,
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.heads",
            vec![12, 12],
            vec![8, 12],
            TensorPlacement::Range {
                axis: 0,
                start: 0,
                end: 8,
            },
        );
        insert(
            &mut layout,
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.intermediate",
            vec![17, 12],
            vec![9, 12],
            TensorPlacement::Range {
                axis: 0,
                start: 0,
                end: 9,
            },
        );
        insert(
            &mut layout,
            "model.layers.1.self_attn.q_proj.weight",
            "model.layers.1.self_attn.heads",
            vec![18, 12],
            vec![12, 12],
            TensorPlacement::Range {
                axis: 0,
                start: 0,
                end: 12,
            },
        );
        insert(
            &mut layout,
            "model.layers.1.mlp.experts.gate_up_proj",
            "model.layers.1.mlp.experts.intermediate",
            vec![2, 18, 12],
            vec![2, 10, 12],
            TensorPlacement::Range {
                axis: 1,
                start: 0,
                end: 10,
            },
        );
        insert(
            &mut layout,
            "model.layers.1.mlp.shared_experts.gate_proj.weight",
            "model.layers.1.mlp.shared_experts.intermediate",
            vec![9, 12],
            vec![5, 12],
            TensorPlacement::Range {
                axis: 0,
                start: 0,
                end: 5,
            },
        );
        layout
    }

    fn valid_dense_layout() -> LocalModelLayout {
        let mut layout = valid_layout();
        insert(
            &mut layout,
            "model.layers.1.mlp.gate_proj.weight",
            "model.layers.1.mlp.intermediate",
            vec![17, 12],
            vec![9, 12],
            TensorPlacement::Range {
                axis: 0,
                start: 0,
                end: 9,
            },
        );
        layout
    }

    fn routed_layout() -> LocalModelLayout {
        let mut layout = valid_layout();
        insert_logical(
            &mut layout,
            "model.layers.1.mlp.experts.gate_up_proj",
            "model.layers.1.mlp.experts.intermediate",
            vec![2, 18, 12],
            vec![2, 10, 12],
            TensorPlacement::Range {
                axis: 1,
                start: 0,
                end: 10,
            },
            9,
            0..5,
        );
        insert_logical(
            &mut layout,
            "model.layers.1.mlp.shared_experts.gate_proj.weight",
            "model.layers.1.mlp.shared_experts.intermediate",
            vec![9, 12],
            vec![5, 12],
            TensorPlacement::Range {
                axis: 0,
                start: 0,
                end: 5,
            },
            9,
            0..5,
        );
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
        let specs = [((group, 1), {
            crate::kimi_linear::moe::expert_bank_spec(&args, 1)
                .unwrap()
                .with_group_geometry(local_count, 5)
                .unwrap()
        })]
        .into_iter()
        .collect();
        crate::ExpertRealizationPlan::balanced(2, topology, specs).unwrap()
    }

    #[test]
    fn local_geometry_owns_blocks_vocabulary_and_state_together() {
        let args = args();
        let geometry = local_geometry(&args, &valid_layout()).unwrap();
        assert_eq!(geometry.blocks().len(), 2);
        assert_eq!(geometry.block(0).unwrap().kda_heads, 2);
        assert_eq!(geometry.block(0).unwrap().dense_intermediate, 9);
        assert_eq!(geometry.block(1).unwrap().mla_heads, 2);
        assert_eq!(geometry.block(1).unwrap().routed_intermediate, 5);
        assert_eq!(geometry.block(1).unwrap().shared_intermediate, 5);
        assert_eq!(geometry.embedding_range().local, 0..8);
        assert_eq!(geometry.output_range().unwrap().local, 0..8);
        assert_ne!(
            geometry.state_layout(),
            &crate::kimi_linear::state_layout(&args).unwrap()
        );
        geometry.validate_for(&args).unwrap();
    }

    #[test]
    fn partition_geometry_retains_only_owned_dense_units_and_exact_mixed_state() {
        let args = dense_args();
        let layout = valid_dense_layout();
        let geometry = partition_local_geometry(&args, &layout, 1..2).unwrap();
        assert_eq!(geometry.owned_units(), 1..2);
        assert_eq!(geometry.local_unit_count(), 1);
        assert!(geometry.block(0).is_none());
        assert_eq!(geometry.block(1).unwrap().mla_heads, 2);
        assert_eq!(geometry.complete_state_layout().len(), 2);
        assert!(matches!(
            geometry.complete_state_layout().layer(0),
            Some(eredu_core::cache::LayerCachePolicy::FixedState { tensors })
                if tensors.len() == 4
                    && tensors.last().unwrap().dtype
                        == eredu_core::cache::StateTensorDtype::Float32
        ));
        assert!(matches!(
            geometry.complete_state_layout().layer(1),
            Some(eredu_core::cache::LayerCachePolicy::CompressedLatentRotary {
                latent_dim,
                rotary_dim,
                ..
            }) if latent_dim.get() == 6 && rotary_dim.get() == 2
        ));
        assert!(partition_local_geometry(&args, &layout, 0..0).is_err());
        assert!(partition_local_geometry(&args, &layout, 1..3).is_err());
        assert!(partition_local_geometry(&self::args(), &layout, 0..1).is_err());

        let mut malformed = layout;
        insert(
            &mut malformed,
            "model.layers.1.self_attn.q_proj.weight",
            "model.layers.1.self_attn.heads",
            vec![18, 12],
            vec![11, 12],
            TensorPlacement::Range {
                axis: 0,
                start: 0,
                end: 11,
            },
        );
        assert!(partition_local_geometry(&args, &malformed, 1..2).is_err());
    }

    #[test]
    fn routed_partition_binds_sparse_stage_and_rejects_prediction_or_plan_drift() {
        let shape = eredu_core::ParallelTopology::new(2, 2, 2, 1).unwrap();
        let topology = eredu_core::ParallelRankTopology::new(shape, 5).unwrap();
        let plan = realization(topology);
        let geometry =
            partition_local_routed_geometry(&args(), &routed_layout(), 1..2, topology, &plan)
                .unwrap();

        assert_eq!(geometry.owned_units(), 1..2);
        assert_eq!(geometry.static_roles(), ["norm", "output"]);
        assert_eq!(
            geometry.boundary_schema(),
            &eredu_runtime::NoAuxiliaryBoundarySchema::new(12)
        );
        assert_eq!(geometry.expert_intermediate_range(), Some(0..5));
        assert_eq!(geometry.shared_expert_intermediate_range(), Some(0..5));
        assert_eq!(geometry.expert_banks().len(), 1);
        assert_eq!(geometry.state_global_offset(), 1);
        assert_eq!(geometry.local_state_layout().unwrap().len(), 1);
        let ownership =
            eredu_runtime::PartitionOwnership::new(false, true, ["norm", "output"]).unwrap();
        geometry
            .validate_partition_contract(&ownership, geometry.boundary_schema())
            .unwrap();
        assert!(geometry
            .validate_partition_contract(
                &eredu_runtime::PartitionOwnership::new(true, false, ["embedding"]).unwrap(),
                geometry.boundary_schema()
            )
            .is_err());
        let bank = &geometry.expert_banks()[0];
        assert_eq!((bank.global_unit(), bank.global_expert()), (1, 1));
        assert_eq!(bank.owner_local_expert(), 0);
        assert!(matches!(
            geometry.complete_state_layout().layer(0),
            Some(eredu_core::cache::LayerCachePolicy::FixedState { .. })
        ));
        assert!(matches!(
            geometry.complete_state_layout().layer(1),
            Some(eredu_core::cache::LayerCachePolicy::CompressedLatentRotary { .. })
        ));

        let mut predicted = args();
        predicted.num_nextn_predict_layers = 1;
        assert!(partition_local_routed_geometry(
            &predicted,
            &routed_layout(),
            1..2,
            topology,
            &plan
        )
        .is_err());
        let wrong_topology = eredu_core::ParallelRankTopology::new(shape, 4).unwrap();
        assert!(partition_local_routed_geometry(
            &args(),
            &routed_layout(),
            1..2,
            topology,
            &realization(wrong_topology)
        )
        .is_err());
    }

    #[test]
    fn local_geometry_rejects_incomplete_heads_and_vocabulary_drift() {
        let args = args();
        let mut layout = valid_layout();
        insert(
            &mut layout,
            "model.layers.1.self_attn.q_proj.weight",
            "model.layers.1.self_attn.heads",
            vec![18, 12],
            vec![11, 12],
            TensorPlacement::Range {
                axis: 0,
                start: 0,
                end: 11,
            },
        );
        assert!(local_geometry(&args, &layout)
            .unwrap_err()
            .to_string()
            .contains("complete heads"));

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
}
