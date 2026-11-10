//! Semantic tensor-parallel placement for Kimi physical blocks.

use eredu_nn::{LogicalParameterRole, VocabularyParallelRange};
use eredu_runtime::{
    LocalModelLayout, MemberSharding, ParallelPlanError, ParameterRole, StateLayout,
};
use std::ops::Range;

use super::{prompt_cache_architecture_fingerprint, BlockGeometry, Declaration, ModelArgs};

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
    definition: std::sync::Arc<eredu_runtime::ComponentDefinition>,
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
    definition: std::sync::Arc<eredu_runtime::ComponentDefinition>,
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
        eredu_runtime::ParameterBankKey::new(0, self.global_unit, self.global_expert)
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

    pub(super) fn validate_for(&self, declaration: &Declaration) -> Result<(), ParallelPlanError> {
        let args = declaration.args();
        if !declaration.expert_banks().is_empty() != self.expert_realization.is_some() {
            return Err(ParallelPlanError::InvalidGroup(
                "partition-local Kimi routed authority differs from its schedule".into(),
            ));
        }
        let count = declaration.blocks.len();
        if self.owned_units.is_empty()
            || self.owned_units.end > count
            || self.blocks.len() != self.owned_units.len()
            || self.complete_state_layout.len() != count
            || self.architecture_fingerprint != prompt_cache_architecture_fingerprint(args)
            || self.definition.parameters() != declaration.definition().parameters()
            || self.tied_head != declaration.statics.head_spec().is_none()
            || self.static_roles != expected_static_roles(&self.owned_units, count)
            || self.boundary_schema
                != eredu_runtime::NoAuxiliaryBoundarySchema::new(
                    declaration.statics.embedding_spec().dimensions,
                )
        {
            return Err(ParallelPlanError::InvalidGroup(
                "partition-local Kimi geometry belongs to a different model or range".into(),
            ));
        }
        self.embedding_range
            .validate_global_rows(declaration.statics.embedding_spec().vocabulary)
            .map_err(|error| ParallelPlanError::InvalidTensor(error.to_string()))?;
        match (
            declaration.statics.head_spec().is_none(),
            &self.output_range,
        ) {
            (true, None) => {}
            (false, Some(range)) => range
                .validate_global_rows(declaration.statics.embedding_spec().vocabulary)
                .map_err(|error| ParallelPlanError::InvalidTensor(error.to_string()))?,
            _ => {
                return Err(ParallelPlanError::InvalidGroup(
                    "partition-local Kimi output vocabulary ownership is inconsistent".into(),
                ));
            }
        }
        for global in self.owned_units.clone() {
            let block = self.block(global).expect("validated owned Kimi block");
            let state = self.complete_state_layout.layer(global).ok_or_else(|| {
                ParallelPlanError::InvalidGroup("missing Kimi complete state layer".into())
            })?;
            let expected = declaration.blocks[global]
                .clone()
                .local(*block, None)
                .and_then(|p| p.mixer.state_policy())
                .map_err(invalid)?;
            let matches = expected == *state;
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

    pub(super) fn validate_for(&self, declaration: &Declaration) -> Result<(), ParallelPlanError> {
        let args = declaration.args();
        let expected_layers = declaration.blocks.len();
        if self.blocks.len() != expected_layers
            || self.architecture_fingerprint != prompt_cache_architecture_fingerprint(args)
            || self.definition.parameters() != declaration.definition().parameters()
            || self.tied_head != declaration.statics.head_spec().is_none()
        {
            return Err(ParallelPlanError::InvalidGroup(
                "rank-local Kimi geometry belongs to a different model configuration".into(),
            ));
        }
        self.embedding_range
            .validate_global_rows(declaration.statics.embedding_spec().vocabulary)
            .map_err(|error| ParallelPlanError::InvalidTensor(error.to_string()))?;
        match (
            declaration.statics.head_spec().is_none(),
            &self.output_range,
        ) {
            (true, None) => {}
            (false, Some(range)) => range
                .validate_global_rows(declaration.statics.embedding_spec().vocabulary)
                .map_err(|error| ParallelPlanError::InvalidTensor(error.to_string()))?,
            (true, Some(_)) => {
                return Err(ParallelPlanError::InvalidGroup(
                    "tied Kimi output unexpectedly owns a separate vocabulary range".into(),
                ));
            }
            (false, None) => {
                return Err(ParallelPlanError::InvalidGroup(
                    "untied Kimi output is missing vocabulary ownership".into(),
                ));
            }
        }
        let expected = declaration
            .state_layout_with_geometry(&self.blocks)
            .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
        if expected != self.state_layout {
            return Err(ParallelPlanError::InvalidGroup(
                "rank-local Kimi state layout drifted from block geometry".into(),
            ));
        }
        Ok(())
    }
}

fn invalid(error: impl std::fmt::Display) -> ParallelPlanError {
    ParallelPlanError::InvalidGroup(error.to_string())
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
    declaration: &Declaration,
    layer: usize,
    layout: &LocalModelLayout,
) -> Result<BlockGeometry, ParallelPlanError> {
    use super::declarations::{FeedForwardParameters as F, MixerParameters as M};
    let scope = declaration
        .blocks
        .get(layer)
        .ok_or_else(|| ParallelPlanError::InvalidGroup(format!("Kimi has no layer {layer}")))?;
    let mut geometry = scope.geometry(declaration.args());
    match &scope.mixer {
        M::Kda(p) => {
            let width = local_width(layout, p.q.weight.id.as_str(), 0)?;
            let head = p.geometry.key_dimensions();
            if width % head != 0 {
                return Err(ParallelPlanError::InvalidTensor(
                    "local KDA width does not contain complete heads".into(),
                ));
            }
            geometry.kda_heads = width / head;
        }
        M::Mla(p) => {
            let query = match &p.query {
                super::specs::QuerySpec::Direct(p) => p,
                super::specs::QuerySpec::LowRank(p) => &p.second,
            };
            let width = local_width(layout, query.weight.id.as_str(), 0)?;
            let head = p.geometry.query_head_dimensions();
            if width % head != 0 {
                return Err(ParallelPlanError::InvalidTensor(
                    "local MLA width does not contain complete heads".into(),
                ));
            }
            geometry.mla_heads = width / head;
        }
    }
    match &scope.feed_forward {
        F::Dense(p) => {
            geometry.dense_intermediate = local_width(layout, p.gate.weight.id.as_str(), 0)?
        }
        F::Sparse(p) => {
            let source = p.routed.experts.packed_parameters().map_err(invalid)?;
            let fused = local_width(layout, source.gate_up().parameter.id.as_str(), 1)?;
            if fused % 2 != 0 {
                return Err(ParallelPlanError::InvalidTensor(
                    "local Kimi packed expert width is not even".into(),
                ));
            }
            geometry.routed_intermediate = fused / 2;
            geometry.shared_intermediate =
                local_width(layout, p.shared.gate.weight.id.as_str(), 0)?;
        }
    }
    scope.clone().local(geometry, None).map_err(invalid)?;
    Ok(geometry)
}

/// Derives the complete TP-local heterogeneous state before pipeline slicing.
///
/// KDA recurrent and convolutional state follows the locally owned attention
/// heads, while MLA compressed latent/rotary state is head-independent.  This
/// projection is used during admission, before any backend module exists.
pub fn partitioned_state_layout(
    declaration: &Declaration,
    tensor_rank: usize,
    tensor_size: usize,
) -> Result<StateLayout, ParallelPlanError> {
    if tensor_size == 0 || tensor_rank >= tensor_size {
        return Err(ParallelPlanError::InvalidGroup(format!(
            "invalid Kimi tensor coordinate {tensor_rank}/{tensor_size}"
        )));
    }
    let geometry = declaration
        .blocks
        .iter()
        .map(|scope| {
            let mut geometry = scope.geometry(declaration.args());
            if let super::declarations::MixerParameters::Kda(p) = &scope.mixer {
                geometry.kda_heads = i32::try_from(
                    eredu_core::balanced_contiguous_range(
                        p.geometry.value_heads() as usize,
                        tensor_size,
                        tensor_rank,
                        false,
                    )
                    .map_err(invalid)?
                    .len(),
                )
                .map_err(invalid)?;
            }
            Ok(geometry)
        })
        .collect::<Result<Vec<_>, ParallelPlanError>>()?;
    declaration
        .state_layout_with_geometry(&geometry)
        .map_err(invalid)
}

/// Derives complete rank-local Kimi construction geometry from one typed plan.
pub fn local_geometry(
    declaration: &Declaration,
    layout: &LocalModelLayout,
) -> Result<LocalGeometry, ParallelPlanError> {
    let args = declaration.args();
    let layers = declaration.blocks.len();
    let blocks = (0..layers)
        .map(|layer| local_block_geometry(declaration, layer, layout))
        .collect::<Result<Vec<_>, _>>()?;
    let state_layout = declaration
        .state_layout_with_geometry(&blocks)
        .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
    let (embedding_range, output_range) = declaration.statics.vocabulary_ranges(
        layout,
        "model.embed_tokens",
        "model.norm",
        "lm_head",
    )?;
    let geometry = LocalGeometry {
        blocks,
        embedding_range,
        output_range,
        state_layout,
        architecture_fingerprint: prompt_cache_architecture_fingerprint(args),
        definition: std::sync::Arc::clone(declaration.definition()),
        tied_head: declaration.statics.head_spec().is_none(),
    };
    geometry.validate_for(declaration)?;
    Ok(geometry)
}

/// Derives dense Kimi construction geometry for one exact TP/PP partition.
pub fn partition_local_geometry(
    declaration: &Declaration,
    layout: &LocalModelLayout,
    owned_units: Range<usize>,
) -> Result<PartitionLocalGeometry, ParallelPlanError> {
    if !declaration.expert_banks().is_empty() {
        return Err(ParallelPlanError::InvalidGroup(
            "partitioned resident Kimi does not accept routed feed-forward layers".into(),
        ));
    }
    let count = declaration.blocks.len();
    if owned_units.is_empty() || owned_units.end > count {
        return Err(ParallelPlanError::InvalidGroup(format!(
            "Kimi local unit range {owned_units:?} is outside {count} layers"
        )));
    }
    let complete = local_geometry(declaration, layout)?;
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
        definition: std::sync::Arc::clone(&complete.definition),
        tied_head: complete.tied_head,
        static_roles: expected_static_roles(&owned_units, count),
        boundary_schema: eredu_runtime::NoAuxiliaryBoundarySchema::new(
            declaration.statics.embedding_spec().dimensions,
        ),
        expert_realization: None,
        expert_intermediate_range: None,
        shared_expert_intermediate_range: None,
        expert_banks: Vec::new(),
    };
    geometry.validate_for(declaration)?;
    Ok(geometry)
}

/// Derives prediction-free routed Kimi geometry from the selected expert plan.
pub fn partition_local_routed_geometry(
    declaration: &Declaration,
    layout: &LocalModelLayout,
    owned_units: Range<usize>,
    topology: eredu_core::ParallelRankTopology,
    realization: &crate::ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>,
) -> Result<PartitionLocalGeometry, ParallelPlanError> {
    let args = declaration.args();
    if args.num_nextn_predict_layers != 0 {
        return Err(ParallelPlanError::InvalidGroup(
            "partition-local Kimi rejects embedded prediction".into(),
        ));
    }
    if declaration.expert_banks().is_empty() {
        return Err(ParallelPlanError::InvalidGroup(
            "routed Kimi requires sparse units".into(),
        ));
    }
    let count = declaration.blocks.len();
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
    let complete = local_geometry(declaration, layout)?;
    let banks = declaration.expert_banks();
    let global_experts = banks
        .values()
        .next()
        .ok_or_else(|| {
            ParallelPlanError::InvalidGroup("sparse schedule has no declared expert bank".into())
        })?
        .group_count() as usize;
    let local_experts = validate_realization(topology, global_experts, realization)?;
    let local_count = i32::try_from(local_experts.len()).map_err(|_| {
        ParallelPlanError::InvalidGroup("Kimi local expert count exceeds i32".into())
    })?;
    let mut intermediate_range = None;
    let mut shared_range = None;
    for (&unit, declared) in &banks {
        let block = complete.block(unit).ok_or_else(|| {
            ParallelPlanError::InvalidGroup("missing local Kimi sparse block".into())
        })?;
        let expected = declared
            .clone()
            .with_group_geometry(local_count, block.routed_intermediate)
            .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
        if realization.unit_spec(crate::decoder::TARGET_EXECUTION_GROUP, unit) != Some(&expected) {
            return Err(ParallelPlanError::InvalidGroup(format!(
                "Kimi expert plan drifted at unit {unit}"
            )));
        }
        let source = declared
            .packed_parameters()
            .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
        let range = exact_logical_range(
            layout,
            source.gate_up().parameter.id.as_str(),
            declared.intermediate_dimensions(),
        )?;
        if intermediate_range
            .as_ref()
            .is_some_and(|current| current != &range)
        {
            return Err(ParallelPlanError::InvalidTensor(
                "Kimi expert TP ranges differ".into(),
            ));
        }
        intermediate_range = Some(range);
        let super::declarations::FeedForwardParameters::Sparse(parameters) =
            &declaration.blocks[unit].feed_forward
        else {
            return Err(ParallelPlanError::InvalidGroup(
                "Kimi shared scope is unresolved".into(),
            ));
        };
        let shared_specs = &parameters.shared;
        let shared_gate = &shared_specs.gate;
        let shared =
            exact_logical_range(layout, shared_gate.weight.id.as_str(), shared_gate.output)?;
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
    if realization.unit_specs().len() != banks.len() {
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
        .filter(|unit| banks.contains_key(unit))
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
        definition: std::sync::Arc::clone(&complete.definition),
        tied_head: complete.tied_head,
        static_roles: expected_static_roles(&owned_units, count),
        boundary_schema: eredu_runtime::NoAuxiliaryBoundarySchema::new(
            declaration.statics.embedding_spec().dimensions,
        ),
        expert_realization: Some(realization.clone()),
        expert_intermediate_range: Some(intermediate_range),
        shared_expert_intermediate_range: Some(shared_range),
        expert_banks,
    };
    geometry.validate_for(declaration)?;
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
    tensor.expanded_logical_range(global)
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

pub(super) fn parameter_description_from_scopes(
    args: &ModelArgs,
    statics: &crate::decoder::StaticModuleConstruction,
    blocks: &[super::declarations::BlockParameters],
) -> Result<eredu_runtime::ArchitectureParameterDescription, ParallelPlanError> {
    use eredu_runtime::{
        ArchitectureParameterDescription, ExecutionGraph, ExecutionUnitLayout,
        OwnedParameterGroupSpec, ParameterGroupOwner,
    };
    args.validate()
        .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
    let layers = blocks.len();
    let graph = ExecutionGraph::chain([crate::decoder::TARGET_EXECUTION_GROUP])
        .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
    let layout = ExecutionUnitLayout::new(&graph, [layers])
        .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
    let owner_group = layout.group_id(0).expect("Kimi target group").clone();
    let static_groups = statics.pinned_parameter_groups(
        crate::decoder::ParameterEncoding::Stored,
        "model.embed_tokens",
        "model.norm",
        "lm_head",
    )?;
    let mut owned = static_groups
        .into_iter()
        .map(|(role, group)| {
            let owner = if role == "embedding" && args.tie_word_embeddings {
                ParameterGroupOwner::static_any_of(["embedding", "output"])
            } else {
                ParameterGroupOwner::static_role(role)
            };
            OwnedParameterGroupSpec::new(owner, group)
        })
        .collect::<Vec<_>>();
    for (layer, scope) in blocks.iter().enumerate() {
        let root = format!("model.layers.{layer}");
        let attention = format!("{root}.self_attn");
        let attention_group = match &scope.mixer {
            super::declarations::MixerParameters::Kda(spec) => {
                eredu_runtime::logical_parameter_group(
                    format!("{attention}.heads"),
                    ParameterRole::AttentionHeads,
                    Some(spec.geometry.value_heads() as usize),
                    spec,
                    |declaration| {
                        let id = &declaration.parameter.id;
                        if id == &spec.output.weight.id {
                            MemberSharding::Partitioned { axis: 1 }
                        } else if id == &spec.a_log.id {
                            MemberSharding::Partitioned { axis: 2 }
                        } else if id == &spec.f_a.weight.id
                            || id == &spec.g_a.weight.id
                            || declaration.role == LogicalParameterRole::Normalization
                        {
                            MemberSharding::Replicated
                        } else {
                            MemberSharding::Partitioned { axis: 0 }
                        }
                    },
                )?
            }
            super::declarations::MixerParameters::Mla(spec) => {
                eredu_runtime::logical_parameter_group(
                    format!("{attention}.heads"),
                    ParameterRole::AttentionHeads,
                    Some(spec.geometry.heads() as usize),
                    spec,
                    |declaration| {
                        let id = &declaration.parameter.id;
                        let first = match &spec.query {
                            super::specs::QuerySpec::LowRank(query) => {
                                query.first.as_ref().map(|p| &p.weight.id)
                            }
                            _ => None,
                        };
                        if id == &spec.output.weight.id {
                            MemberSharding::Partitioned { axis: 1 }
                        } else if id == &spec.kv_a.weight.id
                            || Some(id) == first
                            || declaration.role == LogicalParameterRole::Normalization
                        {
                            MemberSharding::Replicated
                        } else {
                            MemberSharding::Partitioned { axis: 0 }
                        }
                    },
                )?
            }
        };
        let mut groups = vec![attention_group];
        for (field, spec) in [
            ("input_layernorm", &scope.input_norm),
            ("post_attention_layernorm", &scope.post_attention_norm),
        ] {
            groups.push(eredu_runtime::logical_parameter_group(
                format!("{root}.{field}"),
                ParameterRole::Replicated,
                None,
                spec,
                |_| MemberSharding::Replicated,
            )?);
        }
        let sparse = match &scope.feed_forward {
            super::declarations::FeedForwardParameters::Sparse(spec) => Some(spec),
            _ => None,
        };
        if let Some(parameters) = &sparse {
            groups.push(eredu_runtime::logical_parameter_group(
                format!("{root}.mlp.gate"),
                ParameterRole::Replicated,
                None,
                &parameters.routed.selector,
                |_| MemberSharding::Replicated,
            )?);
            let bank = &parameters.routed.experts;
            let eredu_nn::GatedProductGroupLayout::Packed { gate_up, .. } = bank.layout() else {
                return Err(ParallelPlanError::InvalidGroup(
                    "Kimi requires packed experts".into(),
                ));
            };
            let width = bank.intermediate_dimensions() as usize;
            groups.push(eredu_runtime::logical_parameter_group(
                format!("{root}.mlp.experts.intermediate"),
                ParameterRole::ExpertIntermediate,
                Some(width),
                bank,
                |p| {
                    if p.parameter.id == gate_up.weight().id
                        || gate_up.bias().is_some_and(|bias| p.parameter.id == bias.id)
                    {
                        MemberSharding::PartitionedSegments {
                            axis: 1,
                            segments: vec![0..width, width..2 * width],
                        }
                    } else if p.role == LogicalParameterRole::LinearWeight {
                        MemberSharding::Partitioned { axis: 2 }
                    } else {
                        MemberSharding::Replicated
                    }
                },
            )?);
        }
        let (prefix, specs) = match &scope.feed_forward {
            super::declarations::FeedForwardParameters::Dense(spec) => {
                (format!("{root}.mlp"), spec)
            }
            super::declarations::FeedForwardParameters::Sparse(spec) => {
                (format!("{root}.mlp.shared_experts"), &spec.shared)
            }
        };
        groups.push(eredu_runtime::logical_parameter_group(
            format!("{prefix}.intermediate"),
            ParameterRole::FeedForwardIntermediate,
            Some(specs.down.input as usize),
            specs,
            |p| {
                if p.parameter.id == specs.down.weight.id {
                    MemberSharding::Partitioned { axis: 1 }
                } else {
                    MemberSharding::Partitioned { axis: 0 }
                }
            },
        )?);
        owned.extend(groups.into_iter().map(|group| {
            OwnedParameterGroupSpec::new(
                ParameterGroupOwner::execution_unit(owner_group.clone(), layer),
                group,
            )
        }));
    }
    ArchitectureParameterDescription::from_owned(&graph, &layout, owned)
        .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))
}
