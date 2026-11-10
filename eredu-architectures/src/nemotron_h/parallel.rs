//! Semantic placement and rank-local geometry for Nemotron-H physical units.

use eredu_nn::VocabularyParallelRange;
use eredu_runtime::{LocalModelLayout, ParallelPlanError, StateLayout};
use std::ops::Range;

use super::{prompt_cache_architecture_fingerprint, LayerGeometry};

/// Complete planner-derived construction and state geometry for one Nemotron-H rank.
#[derive(Debug, Clone)]
pub struct LocalGeometry {
    target: Vec<LayerGeometry>,
    prediction: Vec<LayerGeometry>,
    embedding_range: VocabularyParallelRange,
    output_range: Option<VocabularyParallelRange>,
    state_layout: StateLayout,
    architecture_fingerprint: String,
    parameters: std::sync::Arc<eredu_runtime::ArchitectureParameterDescription>,
    tied_head: bool,
}

/// Exact TP-local and PP-local geometry for one dense target partition.
#[derive(Debug, Clone)]
pub struct PartitionLocalGeometry {
    owned_units: Range<usize>,
    units: Vec<LayerGeometry>,
    embedding_range: VocabularyParallelRange,
    output_range: Option<VocabularyParallelRange>,
    complete_state_layout: StateLayout,
    architecture_fingerprint: String,
    parameters: std::sync::Arc<eredu_runtime::ArchitectureParameterDescription>,
    tied_head: bool,
    static_roles: Vec<String>,
    boundary_schema: super::TargetBoundarySchema,
    expert_realization: Option<crate::ExpertRealizationPlan<eredu_nn::GroupedRelu2Spec>>,
    routed_expert_intermediate_range: Option<Range<usize>>,
    shared_expert_intermediate_range: Option<Range<usize>>,
    expert_banks: Vec<PartitionExpertBankOwnership>,
}

/// One compact routed Nemotron-H bank at the PP×EP×TP ownership intersection.
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
    /// Architecture-global target-unit range physically owned here.
    pub fn owned_units(&self) -> Range<usize> {
        self.owned_units.clone()
    }

    /// Resolves an owned global unit to its local construction geometry.
    pub fn unit(&self, global_unit: usize) -> Option<&LayerGeometry> {
        self.owned_units
            .contains(&global_unit)
            .then(|| &self.units[global_unit - self.owned_units.start])
    }

    /// Number of unit configurations physically retained here.
    pub fn local_unit_count(&self) -> usize {
        self.units.len()
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

    /// Exact role-tagged target boundary retained for inter-stage transfer.
    pub const fn boundary_schema(&self) -> &super::TargetBoundarySchema {
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
        boundary: &super::TargetBoundarySchema,
    ) -> Result<(), ParallelPlanError> {
        let owns_input = self.owned_units.start == 0;
        let owns_output = self.static_roles.iter().any(|role| role == "output");
        if ownership.owns_input() != owns_input
            || ownership.owns_output() != owns_output
            || ownership.static_roles() != self.static_roles
            || boundary != &self.boundary_schema
        {
            return Err(ParallelPlanError::InvalidGroup(
                "Nemotron-H partition ownership or boundary drifted from local geometry".into(),
            ));
        }
        Ok(())
    }

    /// Borrows the immutable routed-expert realization retained by this geometry.
    pub const fn expert_realization(
        &self,
    ) -> Option<&crate::ExpertRealizationPlan<eredu_nn::GroupedRelu2Spec>> {
        self.expert_realization.as_ref()
    }

    /// Returns this TP rank's routed-expert intermediate interval.
    pub fn routed_expert_intermediate_range(&self) -> Option<Range<usize>> {
        self.routed_expert_intermediate_range.clone()
    }

    /// Returns this TP rank's replicated-over-EP shared-expert interval.
    pub fn shared_expert_intermediate_range(&self) -> Option<Range<usize>> {
        self.shared_expert_intermediate_range.clone()
    }

    /// Borrows the exact PP×EP×TP compact routed-bank ownership.
    pub fn expert_banks(&self) -> &[PartitionExpertBankOwnership] {
        &self.expert_banks
    }

    pub(super) fn validate_for(
        &self,
        declaration: &super::Declaration,
    ) -> Result<(), ParallelPlanError> {
        let args = declaration.args();
        if args.has_sparse_moe_layers() != self.expert_realization.is_some() {
            return Err(ParallelPlanError::InvalidGroup(
                "partition-local Nemotron-H routed authority differs from its schedule".into(),
            ));
        }
        if args.num_nextn_predict_layers != 0 {
            return Err(ParallelPlanError::InvalidGroup(
                "partitioned resident Nemotron-H does not accept embedded prediction".into(),
            ));
        }
        let count = usize::try_from(args.num_hidden_layers).map_err(|_| {
            ParallelPlanError::InvalidGroup("Nemotron-H layer count exceeds usize".into())
        })?;
        if self.owned_units.is_empty()
            || self.owned_units.end > count
            || self.units.len() != self.owned_units.len()
            || self.complete_state_layout.len() != count
            || self.architecture_fingerprint != prompt_cache_architecture_fingerprint(args)
            || self.parameters.as_ref() != declaration.definition().parameters()
            || self.tied_head != args.tie_word_embeddings
            || self.static_roles != expected_static_roles(&self.owned_units, count)
            || self.boundary_schema != super::TargetBoundarySchema::from_declaration(declaration)
        {
            return Err(ParallelPlanError::InvalidGroup(
                "partition-local Nemotron-H geometry belongs to a different model or range".into(),
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
                    "partition-local Nemotron-H output vocabulary ownership is inconsistent".into(),
                ));
            }
        }
        for global in self.owned_units.clone() {
            let geometry = *self.unit(global).expect("validated owned Nemotron-H unit");
            let policy = args.layer_schedule.get(global).copied().ok_or_else(|| {
                ParallelPlanError::InvalidGroup(format!("Nemotron-H has no unit {global}"))
            })?;
            let state = self.complete_state_layout.layer(global).ok_or_else(|| {
                ParallelPlanError::InvalidGroup("missing complete Nemotron-H state layer".into())
            })?;
            let scope = declaration.block(global).ok_or_else(|| {
                ParallelPlanError::InvalidGroup("owned invocation has no declared scope".into())
            })?;
            let expected = match (&scope.operator, geometry) {
                (
                    super::declarations::TargetOperatorParameters::Dense(_),
                    LayerGeometry::DenseMlp { .. },
                )
                | (
                    super::declarations::TargetOperatorParameters::Sparse(_),
                    LayerGeometry::SparseMoe { .. },
                ) => scope.state_policy(policy),
                _ => scope
                    .clone()
                    .local(geometry, None)
                    .and_then(|scope| scope.state_policy(policy)),
            }
            .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
            let matches = *state == expected;
            if !matches {
                return Err(ParallelPlanError::InvalidGroup(format!(
                    "owned Nemotron-H unit {global} differs from complete TP-local state geometry"
                )));
            }
        }
        Ok(())
    }
}

impl LocalGeometry {
    /// Returns target-unit geometry in physical execution order.
    pub fn target_units(&self) -> &[LayerGeometry] {
        &self.target
    }

    /// Returns one target unit's rank-local geometry.
    pub fn target_unit(&self, index: usize) -> Option<&LayerGeometry> {
        self.target.get(index)
    }

    /// Returns appended MTP geometry in global physical order.
    pub fn prediction_units(&self) -> &[LayerGeometry] {
        &self.prediction
    }

    /// Returns one MTP unit's rank-local geometry by physical index.
    pub fn prediction_unit(&self, physical: usize) -> Option<&LayerGeometry> {
        self.prediction.get(physical)
    }

    /// Returns input-embedding vocabulary ownership.
    pub const fn embedding_range(&self) -> &VocabularyParallelRange {
        &self.embedding_range
    }

    /// Returns untied output-head vocabulary ownership.
    pub const fn output_range(&self) -> Option<&VocabularyParallelRange> {
        self.output_range.as_ref()
    }

    /// Returns the authoritative heterogeneous target-plus-MTP state layout.
    pub const fn state_layout(&self) -> &StateLayout {
        &self.state_layout
    }

    pub(super) fn validate_for(
        &self,
        declaration: &super::Declaration,
    ) -> Result<(), ParallelPlanError> {
        let args = declaration.args();
        let parameters = declaration.definition().parameters();
        let layout = parameters.unit_layout();
        let target_count = layout.group_range(0).expect("declared target group").len();
        if self.parameters.as_ref() != parameters
            || self.target.len() != target_count
            || self.prediction.len() != layout.len() - target_count
            || self.architecture_fingerprint != prompt_cache_architecture_fingerprint(args)
            || self.tied_head != args.tie_word_embeddings
        {
            return Err(ParallelPlanError::InvalidGroup(
                "rank-local Nemotron-H geometry belongs to a different model configuration".into(),
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
                    "tied Nemotron-H output unexpectedly owns a separate vocabulary range".into(),
                ));
            }
            (false, None) => {
                return Err(ParallelPlanError::InvalidGroup(
                    "untied Nemotron-H output is missing vocabulary ownership".into(),
                ));
            }
        }
        let geometry = self
            .target
            .iter()
            .chain(&self.prediction)
            .copied()
            .collect::<Vec<_>>();
        let expected = declaration
            .state_layout_for(&geometry)
            .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
        if expected != self.state_layout {
            return Err(ParallelPlanError::InvalidGroup(
                "rank-local Nemotron-H state layout drifted from unit geometry".into(),
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
        ParallelPlanError::InvalidTensor(format!("missing local Nemotron-H layout for {name}"))
    })?;
    let local = *tensor.local_shape().get(axis).ok_or_else(|| {
        ParallelPlanError::InvalidTensor(format!("Nemotron-H tensor {name} has no axis {axis}"))
    })?;
    let global = *tensor.global_shape().get(axis).ok_or_else(|| {
        ParallelPlanError::InvalidTensor(format!(
            "Nemotron-H tensor {name} has no global axis {axis}"
        ))
    })?;
    if local == 0 || local > global {
        return Err(ParallelPlanError::InvalidTensor(format!(
            "Nemotron-H tensor {name} has invalid local width {local} of global width {global}"
        )));
    }
    i32::try_from(local).map_err(|_| {
        ParallelPlanError::InvalidTensor(format!("Nemotron-H width for {name} exceeds i32"))
    })
}

fn scope_local_geometry(
    scope: &super::declarations::TargetBlockParameters,
    layout: &LocalModelLayout,
) -> Result<LayerGeometry, ParallelPlanError> {
    use super::declarations::TargetOperatorParameters as P;
    Ok(match &scope.operator {
        P::Mamba(spec) => {
            let heads = local_width(layout, spec.dt_bias.id.as_str(), 0)?;
            let per_group = spec.geometry.heads() / spec.geometry.groups();
            if heads % per_group != 0 {
                return Err(ParallelPlanError::InvalidTensor(
                    "local Mamba heads do not contain complete state groups".into(),
                ));
            }
            LayerGeometry::Mamba {
                heads,
                groups: heads / per_group,
            }
        }
        P::Attention(spec) => {
            let query = local_width(layout, spec.projection.query.weight.id.as_str(), 0)?;
            let key = local_width(layout, spec.projection.key.weight.id.as_str(), 0)?;
            let qdim = spec.projection.geometry.key_dimensions();
            if query % qdim != 0 || key % qdim != 0 {
                return Err(ParallelPlanError::InvalidTensor(
                    "local attention widths do not contain complete heads".into(),
                ));
            }
            LayerGeometry::Attention {
                query_heads: query / qdim,
                kv_heads: key / qdim,
            }
        }
        P::Dense(spec) => LayerGeometry::DenseMlp {
            intermediate: local_width(layout, spec.network.up.weight.id.as_str(), 0)?,
        },
        P::Sparse(spec) => LayerGeometry::SparseMoe {
            routed: local_width(layout, spec.experts.up().weight().id.as_str(), 1)?,
            shared: local_width(layout, spec.shared.network.up.weight.id.as_str(), 0)?,
        },
    })
}

/// Derives one declared target block's geometry from selected parameter placement.
pub fn local_block_geometry(
    declaration: &super::Declaration,
    layer: usize,
    layout: &LocalModelLayout,
) -> Result<LayerGeometry, ParallelPlanError> {
    scope_local_geometry(
        declaration.blocks.get(layer).ok_or_else(|| {
            ParallelPlanError::InvalidGroup("target unit is outside its declaration".into())
        })?,
        layout,
    )
}

/// Resolves target and independent prediction invocations from their declared slots.
pub fn local_state_geometry(
    declaration: &super::Declaration,
    layout: &LocalModelLayout,
) -> Result<Vec<LayerGeometry>, ParallelPlanError> {
    declaration
        .blocks
        .iter()
        .chain(declaration.predictions.iter().map(|unit| &unit.block))
        .map(|scope| scope_local_geometry(scope, layout))
        .collect()
}
pub(super) fn global_target_geometry(declaration: &super::Declaration) -> Vec<LayerGeometry> {
    declaration
        .blocks
        .iter()
        .map(|block| block.geometry())
        .collect()
}
pub(super) fn global_prediction_geometry(declaration: &super::Declaration) -> Vec<LayerGeometry> {
    declaration
        .predictions
        .iter()
        .map(|unit| unit.block.geometry())
        .collect()
}

/// Derives complete rank-local target, MTP, vocabulary, and state geometry.
pub fn local_geometry(
    declaration: &super::Declaration,
    layout: &LocalModelLayout,
) -> Result<LocalGeometry, ParallelPlanError> {
    let args = declaration.args();
    args.validate()
        .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
    let geometry = local_state_geometry(declaration, layout)?;
    let target_count = usize::try_from(args.num_hidden_layers).map_err(|_| {
        ParallelPlanError::InvalidGroup("Nemotron-H target layer count exceeds usize".into())
    })?;
    let (target, prediction) = geometry.split_at(target_count);
    let state_layout = declaration
        .state_layout_for(&geometry)
        .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
    let (embedding_range, output_range) = declaration.statics.vocabulary_ranges(
        layout,
        "model.embeddings",
        "model.norm_f",
        "lm_head",
    )?;
    let local = LocalGeometry {
        target: target.to_vec(),
        prediction: prediction.to_vec(),
        embedding_range,
        output_range,
        state_layout,
        architecture_fingerprint: prompt_cache_architecture_fingerprint(args),
        parameters: std::sync::Arc::new(declaration.definition().parameters().clone()),
        tied_head: args.tie_word_embeddings,
    };
    local.validate_for(declaration)?;
    Ok(local)
}

/// Derives dense target construction geometry for one exact TP/PP partition.
pub fn partition_local_geometry(
    declaration: &super::Declaration,
    layout: &LocalModelLayout,
    owned_units: Range<usize>,
) -> Result<PartitionLocalGeometry, ParallelPlanError> {
    let args = declaration.args();
    if args.has_sparse_moe_layers() {
        return Err(ParallelPlanError::InvalidGroup(
            "partitioned resident Nemotron-H does not accept routed units".into(),
        ));
    }
    if args.num_nextn_predict_layers != 0 {
        return Err(ParallelPlanError::InvalidGroup(
            "partitioned resident Nemotron-H does not accept embedded prediction".into(),
        ));
    }
    let count = usize::try_from(args.num_hidden_layers).map_err(|_| {
        ParallelPlanError::InvalidGroup("Nemotron-H target layer count exceeds usize".into())
    })?;
    if owned_units.is_empty() || owned_units.end > count {
        return Err(ParallelPlanError::InvalidGroup(format!(
            "Nemotron-H local unit range {owned_units:?} is outside {count} target layers"
        )));
    }
    let complete = local_geometry(declaration, layout)?;
    let units = owned_units
        .clone()
        .map(|unit| {
            complete.target_unit(unit).copied().ok_or_else(|| {
                ParallelPlanError::InvalidGroup(format!(
                    "Nemotron-H has no local target unit {unit}"
                ))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let geometry = PartitionLocalGeometry {
        owned_units: owned_units.clone(),
        units,
        embedding_range: complete.embedding_range.clone(),
        output_range: complete.output_range.clone(),
        complete_state_layout: complete.state_layout.clone(),
        architecture_fingerprint: complete.architecture_fingerprint.clone(),
        parameters: std::sync::Arc::clone(&complete.parameters),
        tied_head: complete.tied_head,
        static_roles: expected_static_roles(&owned_units, count),
        boundary_schema: super::TargetBoundarySchema::from_declaration(declaration),
        expert_realization: None,
        routed_expert_intermediate_range: None,
        shared_expert_intermediate_range: None,
        expert_banks: Vec::new(),
    };
    geometry.validate_for(declaration)?;
    Ok(geometry)
}

/// Derives prediction-free routed Nemotron-H geometry from the selected expert plan.
pub fn partition_local_routed_geometry(
    declaration: &super::Declaration,
    layout: &LocalModelLayout,
    owned_units: Range<usize>,
    topology: eredu_core::ParallelRankTopology,
    realization: &crate::ExpertRealizationPlan<eredu_nn::GroupedRelu2Spec>,
) -> Result<PartitionLocalGeometry, ParallelPlanError> {
    let args = declaration.args();
    if args.num_nextn_predict_layers != 0 {
        return Err(ParallelPlanError::InvalidGroup(
            "partition-local Nemotron-H rejects embedded prediction".into(),
        ));
    }
    if !args.has_sparse_moe_layers() {
        return Err(ParallelPlanError::InvalidGroup(
            "routed Nemotron-H requires sparse units".into(),
        ));
    }
    let count = usize::try_from(args.num_hidden_layers).map_err(|_| {
        ParallelPlanError::InvalidGroup("Nemotron-H target layer count exceeds usize".into())
    })?;
    let expected_units = eredu_core::balanced_contiguous_range(
        count,
        topology.pipeline_parallel_size(),
        topology.pipeline_parallel_rank(),
        false,
    )
    .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
    if owned_units.is_empty() || owned_units.end > count || owned_units != expected_units {
        return Err(ParallelPlanError::InvalidGroup(
            "routed Nemotron-H PP ownership differs from its Cartesian rank".into(),
        ));
    }

    let complete = local_geometry(declaration, layout)?;
    let banks = declaration.expert_banks();
    let global_experts = banks
        .values()
        .next()
        .ok_or_else(|| {
            ParallelPlanError::InvalidGroup(
                "Nemotron-H sparse schedule has no declared bank".into(),
            )
        })?
        .group_count() as usize;
    let local_experts = validate_realization(topology, global_experts, realization)?;
    let local_count = i32::try_from(local_experts.len()).map_err(|_| {
        ParallelPlanError::InvalidGroup("Nemotron-H local expert count exceeds i32".into())
    })?;
    let mut routed_range = None;
    let mut shared_range = None;
    for (&unit, bank) in &banks {
        let geometry = complete.target_unit(unit).copied().ok_or_else(|| {
            ParallelPlanError::InvalidGroup("missing local Nemotron-H sparse unit".into())
        })?;
        let LayerGeometry::SparseMoe { routed, shared } = geometry else {
            return Err(ParallelPlanError::InvalidGroup(
                "Nemotron-H sparse schedule has non-sparse local geometry".into(),
            ));
        };
        let expected = (**bank)
            .clone()
            .with_group_geometry(local_count, routed)
            .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
        if realization.unit_spec("target", unit) != Some(&expected) {
            return Err(ParallelPlanError::InvalidGroup(format!(
                "Nemotron-H expert plan drifted at unit {unit}"
            )));
        }
        let current_routed = exact_logical_range(
            layout,
            bank.up().weight().id.as_str(),
            bank.intermediate_dimensions(),
        )?;
        let Some(super::declarations::TargetOperatorParameters::Sparse(declared)) =
            declaration.block(unit).map(|scope| &scope.operator)
        else {
            return Err(ParallelPlanError::InvalidGroup(
                "retained sparse invocation has no scope".into(),
            ));
        };
        let shared_up = &declared.shared.network.up;
        let current_shared =
            exact_logical_range(layout, shared_up.weight.id.as_str(), shared_up.output)?;
        merge_exact_range(&mut routed_range, current_routed, "routed expert")?;
        merge_exact_range(&mut shared_range, current_shared, "shared expert")?;
        if i32::try_from(
            routed_range
                .as_ref()
                .expect("routed range was just inserted")
                .len(),
        ) != Ok(routed)
            || i32::try_from(
                shared_range
                    .as_ref()
                    .expect("shared range was just inserted")
                    .len(),
            ) != Ok(shared)
        {
            return Err(ParallelPlanError::InvalidTensor(
                "Nemotron-H block widths differ from exact TP intervals".into(),
            ));
        }
    }
    if realization.unit_specs().len() != banks.len() {
        return Err(ParallelPlanError::InvalidGroup(
            "Nemotron-H expert unit schedule drifted".into(),
        ));
    }
    let routed_range = routed_range.ok_or_else(|| {
        ParallelPlanError::InvalidGroup("Nemotron-H has no routed TP range".into())
    })?;
    let shared_range = shared_range.ok_or_else(|| {
        ParallelPlanError::InvalidGroup("Nemotron-H has no shared-expert TP range".into())
    })?;
    let units = owned_units
        .clone()
        .map(|unit| {
            complete.target_unit(unit).copied().ok_or_else(|| {
                ParallelPlanError::InvalidGroup("missing owned Nemotron-H unit".into())
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let expert_banks = owned_units
        .clone()
        .filter(|unit| banks.contains_key(unit))
        .flat_map(|global_unit| {
            local_experts.iter().copied().enumerate().map({
                let range = routed_range.clone();
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
        units,
        embedding_range: complete.embedding_range.clone(),
        output_range: complete.output_range.clone(),
        complete_state_layout: complete.state_layout.clone(),
        architecture_fingerprint: complete.architecture_fingerprint.clone(),
        parameters: std::sync::Arc::clone(&complete.parameters),
        tied_head: complete.tied_head,
        static_roles: expected_static_roles(&owned_units, count),
        boundary_schema: super::TargetBoundarySchema::from_declaration(declaration),
        expert_realization: Some(realization.clone()),
        routed_expert_intermediate_range: Some(routed_range),
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

fn merge_exact_range(
    selected: &mut Option<Range<usize>>,
    current: Range<usize>,
    role: &str,
) -> Result<(), ParallelPlanError> {
    if selected.as_ref().is_some_and(|range| range != &current) {
        return Err(ParallelPlanError::InvalidTensor(format!(
            "Nemotron-H {role} TP ranges differ between units"
        )));
    }
    *selected = Some(current);
    Ok(())
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
    realization: &crate::ExpertRealizationPlan<eredu_nn::GroupedRelu2Spec>,
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
            "Nemotron-H expert owner differs from Cartesian rank".into(),
        ));
    }
    Ok(local)
}

/// Derives exact TP-local target state before pipeline ownership is sliced.
pub fn partitioned_state_layout(
    declaration: &super::Declaration,
    tensor_rank: usize,
    tensor_size: usize,
) -> Result<StateLayout, ParallelPlanError> {
    let args = declaration.args();
    args.validate()
        .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
    if args.num_nextn_predict_layers != 0 {
        return Err(ParallelPlanError::InvalidGroup(
            "partitioned Nemotron-H state requires a prediction-free target schedule".into(),
        ));
    }
    if tensor_size == 0 || tensor_rank >= tensor_size {
        return Err(ParallelPlanError::InvalidGroup(
            "Nemotron-H tensor rank is outside its topology".into(),
        ));
    }
    let select = |value: i32, role: &str| {
        let global = usize::try_from(value).map_err(|_| {
            ParallelPlanError::InvalidGroup(format!("Nemotron-H {role} is not positive"))
        })?;
        let local = eredu_core::balanced_contiguous_range(global, tensor_size, tensor_rank, false)
            .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?
            .len();
        i32::try_from(local).map_err(|_| {
            ParallelPlanError::InvalidGroup(format!("Nemotron-H local {role} count exceeds i32"))
        })
    };
    let geometry = global_target_geometry(declaration)
        .into_iter()
        .map(|geometry| {
            Ok(match geometry {
                LayerGeometry::Mamba { heads, groups } => {
                    let local = select(groups, "Mamba groups")?;
                    LayerGeometry::Mamba {
                        heads: local * (heads / groups),
                        groups: local,
                    }
                }
                LayerGeometry::Attention {
                    query_heads,
                    kv_heads,
                } => LayerGeometry::Attention {
                    query_heads: select(query_heads, "query heads")?,
                    kv_heads: select(kv_heads, "key/value heads")?,
                },
                LayerGeometry::DenseMlp { intermediate } => LayerGeometry::DenseMlp {
                    intermediate: select(intermediate, "MLP width")?,
                },
                geometry @ LayerGeometry::SparseMoe { .. } => geometry,
            })
        })
        .collect::<Result<Vec<_>, ParallelPlanError>>()?;
    declaration
        .state_layout_for(&geometry)
        .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))
}
