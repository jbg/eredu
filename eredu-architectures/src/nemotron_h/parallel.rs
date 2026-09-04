//! Semantic placement and rank-local geometry for Nemotron-H physical units.

use eredu_nn::{GroupedNeuralBackend, VocabularyParallelRange};
use eredu_runtime::{
    aligned_partition_units, module_parameter_group, partitioned_module_parameter_group,
    partitioned_projection_group, LocalModelLayout, MemberSharding, ParallelPlanError,
    ParameterGroupSpec, ParameterRole, ProjectionSharding, StateLayout, TensorPlacement,
};
use std::ops::Range;

use super::{
    prompt_cache_architecture_fingerprint, state_layout_with_geometry, Block, DenseMlp,
    LayerGeometry, LayerPolicy, ModelArgs, Operator, PredictionUnit, Unit,
};
use crate::decoder::StaticModules;

/// Complete planner-derived construction and state geometry for one Nemotron-H rank.
#[derive(Debug, Clone)]
pub struct LocalGeometry {
    target: Vec<LayerGeometry>,
    prediction: Vec<LayerGeometry>,
    embedding_range: VocabularyParallelRange,
    output_range: Option<VocabularyParallelRange>,
    state_layout: StateLayout,
    architecture_fingerprint: String,
    global_target: Vec<LayerGeometry>,
    global_prediction: Vec<LayerGeometry>,
    prediction_steps: usize,
    prediction_pattern: usize,
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
        eredu_runtime::ParameterBankKey::new(self.global_unit, self.global_expert)
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

    pub(super) fn validate_for(&self, args: &ModelArgs) -> Result<(), ParallelPlanError> {
        args.validate()
            .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
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
            || self.tied_head != args.tie_word_embeddings
            || self.static_roles != expected_static_roles(&self.owned_units, count)
            || self.boundary_schema != super::TargetBoundarySchema::from_args(args)
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
                ))
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
            let matches = match (policy, geometry, state) {
                (
                    LayerPolicy::Mamba,
                    LayerGeometry::Mamba { heads, groups },
                    eredu_core::cache::LayerCachePolicy::FixedState { tensors },
                ) => {
                    let history = args.conv_kernel - 1;
                    let convolution_width =
                        heads.checked_mul(args.mamba_head_dim).and_then(|width| {
                            groups
                                .checked_mul(args.ssm_state_size)
                                .and_then(|state| state.checked_mul(2))
                                .and_then(|state| width.checked_add(state))
                        });
                    let convolution_count = usize::from(history > 0);
                    tensors.len() == convolution_count + 1
                        && (history == 0
                            || tensors.first().is_some_and(|tensor| {
                                tensor.role
                                    == eredu_core::cache::StateTensorRole::Convolution { slot: 0 }
                                    && tensor.dtype == eredu_core::cache::StateTensorDtype::Floating
                                    && fixed_shape(
                                        &tensor.shape,
                                        &[None, Some(history), convolution_width],
                                    )
                            }))
                        && tensors.last().is_some_and(|tensor| {
                            tensor.role == eredu_core::cache::StateTensorRole::Recurrent
                                && tensor.dtype == eredu_core::cache::StateTensorDtype::Float32
                                && fixed_shape(
                                    &tensor.shape,
                                    &[
                                        None,
                                        Some(heads),
                                        Some(args.mamba_head_dim),
                                        Some(args.ssm_state_size),
                                    ],
                                )
                        })
                }
                (
                    LayerPolicy::SelfAttention(expected_attention),
                    LayerGeometry::Attention { kv_heads, .. },
                    eredu_core::cache::LayerCachePolicy::KeyValue {
                        attention,
                        num_key_value_heads,
                        head_dim,
                    },
                ) => {
                    *attention == expected_attention
                        && i32::try_from(num_key_value_heads.get()) == Ok(kv_heads)
                        && i32::try_from(head_dim.get()) == Ok(args.head_dim)
                }
                (
                    LayerPolicy::DenseMlp,
                    LayerGeometry::DenseMlp { .. },
                    eredu_core::cache::LayerCachePolicy::NoState,
                ) => true,
                (
                    LayerPolicy::SparseMoe,
                    LayerGeometry::SparseMoe { .. },
                    eredu_core::cache::LayerCachePolicy::NoState,
                ) => true,
                _ => false,
            };
            if !matches {
                return Err(ParallelPlanError::InvalidGroup(format!(
                    "owned Nemotron-H unit {global} differs from complete TP-local state geometry"
                )));
            }
        }
        Ok(())
    }
}

fn fixed_shape(
    actual: &[eredu_core::cache::StateTensorDimension],
    expected: &[Option<i32>],
) -> bool {
    actual.len() == expected.len()
        && actual
            .iter()
            .zip(expected)
            .all(|(actual, expected)| match (actual, expected) {
                (eredu_core::cache::StateTensorDimension::Batch, None) => true,
                (eredu_core::cache::StateTensorDimension::Fixed(value), Some(expected)) => {
                    i32::try_from(value.get()) == Ok(*expected)
                }
                _ => false,
            })
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

    pub(super) fn validate_for(&self, args: &ModelArgs) -> Result<(), ParallelPlanError> {
        args.validate()
            .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
        let expected_target = global_target_geometry(args)?;
        let expected_prediction = global_prediction_geometry(args)?;
        let prediction_steps = usize::try_from(args.num_nextn_predict_layers).map_err(|_| {
            ParallelPlanError::InvalidGroup("Nemotron-H prediction-step count exceeds usize".into())
        })?;
        let prediction_pattern = if prediction_steps == 0 {
            0
        } else {
            if expected_prediction.len() % prediction_steps != 0 {
                return Err(ParallelPlanError::InvalidGroup(
                    "Nemotron-H MTP geometry does not divide into prediction steps".into(),
                ));
            }
            expected_prediction
                .len()
                .checked_div(prediction_steps)
                .filter(|pattern| *pattern > 0)
                .ok_or_else(|| {
                    ParallelPlanError::InvalidGroup(
                        "Nemotron-H MTP geometry has an empty prediction pattern".into(),
                    )
                })?
        };
        if self.target.len() != expected_target.len()
            || self.prediction.len() != expected_prediction.len()
            || self.global_target != expected_target
            || self.global_prediction != expected_prediction
            || self.prediction_steps != prediction_steps
            || self.prediction_pattern != prediction_pattern
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
                ))
            }
            (false, None) => {
                return Err(ParallelPlanError::InvalidGroup(
                    "untied Nemotron-H output is missing vocabulary ownership".into(),
                ))
            }
        }
        let geometry = self
            .target
            .iter()
            .chain(&self.prediction)
            .copied()
            .collect::<Vec<_>>();
        let expected = state_layout_with_geometry(args, &geometry)
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

/// Derives one target block's local geometry from the resolved parameter layout.
pub fn local_block_geometry(
    args: &ModelArgs,
    layer: usize,
    layout: &LocalModelLayout,
) -> Result<LayerGeometry, ParallelPlanError> {
    let root = format!("model.layers.{layer}");
    match args.layer_schedule.get(layer).copied().ok_or_else(|| {
        ParallelPlanError::InvalidGroup(format!("Nemotron-H has no layer {layer}"))
    })? {
        LayerPolicy::Mamba => {
            let heads = local_width(layout, &format!("{root}.mamba.dt_bias"), 0)?;
            if args.n_groups <= 0 || args.mamba_num_heads % args.n_groups != 0 {
                return Err(ParallelPlanError::InvalidGroup(
                    "Nemotron-H Mamba heads do not divide into state groups".into(),
                ));
            }
            let heads_per_group = args.mamba_num_heads / args.n_groups;
            if heads_per_group <= 0 || heads % heads_per_group != 0 {
                return Err(ParallelPlanError::InvalidTensor(
                    "local Mamba heads do not contain complete state groups".into(),
                ));
            }
            Ok(LayerGeometry::Mamba {
                heads,
                groups: heads / heads_per_group,
            })
        }
        LayerPolicy::SelfAttention(_) => {
            let query = local_width(layout, &format!("{root}.attention.q_proj.weight"), 0)?;
            let key = local_width(layout, &format!("{root}.attention.k_proj.weight"), 0)?;
            if args.head_dim <= 0 || query % args.head_dim != 0 || key % args.head_dim != 0 {
                return Err(ParallelPlanError::InvalidTensor(
                    "local attention widths do not contain complete heads".into(),
                ));
            }
            Ok(LayerGeometry::Attention {
                query_heads: query / args.head_dim,
                kv_heads: key / args.head_dim,
            })
        }
        LayerPolicy::DenseMlp => Ok(LayerGeometry::DenseMlp {
            intermediate: local_width(layout, &format!("{root}.mlp.up_proj.weight"), 0)?,
        }),
        LayerPolicy::SparseMoe => Ok(LayerGeometry::SparseMoe {
            routed: local_width(layout, &format!("{root}.moe.experts.up_proj"), 1)?,
            shared: local_width(
                layout,
                &format!("{root}.moe.shared_experts.up_proj.weight"),
                0,
            )?,
        }),
    }
}

/// Returns resolved state geometry for target and appended MTP units.
pub fn local_state_geometry(
    args: &ModelArgs,
    layout: &LocalModelLayout,
) -> Result<Vec<LayerGeometry>, ParallelPlanError> {
    let mut geometry = (0..args.num_hidden_layers as usize)
        .map(|layer| local_block_geometry(args, layer, layout))
        .collect::<Result<Vec<_>, _>>()?;
    let target = args.num_hidden_layers as usize;
    for (physical, policy) in args
        .mtp_policies()
        .map_err(|e| ParallelPlanError::InvalidGroup(e.to_string()))?
        .into_iter()
        .enumerate()
    {
        let root = format!("model.mtp.layers.{physical}.mixer");
        geometry.push(match policy {
            LayerPolicy::SelfAttention(_) => {
                let query = local_width(layout, &format!("{root}.q_proj.weight"), 0)?;
                let key = local_width(layout, &format!("{root}.k_proj.weight"), 0)?;
                if args.head_dim <= 0
                    || query % args.head_dim != 0
                    || key % args.head_dim != 0
                {
                    return Err(ParallelPlanError::InvalidTensor(format!(
                        "local Nemotron-H MTP attention widths at physical layer {physical} do not contain complete heads"
                    )));
                }
                LayerGeometry::Attention {
                    query_heads: query / args.head_dim,
                    kv_heads: key / args.head_dim,
                }
            }
            LayerPolicy::SparseMoe => LayerGeometry::SparseMoe {
                routed: local_width(layout, &format!("{root}.experts.up_proj"), 1)?,
                shared: local_width(layout, &format!("{root}.shared_experts.up_proj.weight"), 0)?,
            },
            _ => {
                return Err(ParallelPlanError::InvalidGroup(format!(
                    "unsupported MTP policy at global state layer {}",
                    target + physical
                )))
            }
        });
    }
    Ok(geometry)
}

fn global_target_geometry(args: &ModelArgs) -> Result<Vec<LayerGeometry>, ParallelPlanError> {
    Ok(args
        .layer_schedule
        .iter()
        .map(|policy| match policy {
            LayerPolicy::Mamba => LayerGeometry::Mamba {
                heads: args.mamba_num_heads,
                groups: args.n_groups,
            },
            LayerPolicy::SelfAttention(_) => LayerGeometry::Attention {
                query_heads: args.num_attention_heads,
                kv_heads: args.num_key_value_heads,
            },
            LayerPolicy::DenseMlp => LayerGeometry::DenseMlp {
                intermediate: args.intermediate_size,
            },
            LayerPolicy::SparseMoe => LayerGeometry::SparseMoe {
                routed: args.moe_intermediate_size,
                shared: args.moe_shared_expert_intermediate_size,
            },
        })
        .collect())
}

fn global_prediction_geometry(args: &ModelArgs) -> Result<Vec<LayerGeometry>, ParallelPlanError> {
    Ok(args
        .mtp_policies()
        .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?
        .into_iter()
        .map(|policy| match policy {
            LayerPolicy::SelfAttention(_) => LayerGeometry::Attention {
                query_heads: args.num_attention_heads,
                kv_heads: args.num_key_value_heads,
            },
            LayerPolicy::SparseMoe => LayerGeometry::SparseMoe {
                routed: args.moe_intermediate_size,
                shared: args.moe_shared_expert_intermediate_size,
            },
            _ => unreachable!("validated Nemotron-H MTP schedules contain attention and MoE"),
        })
        .collect())
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
                "Nemotron-H vocabulary member {target} has global shape {:?}, expected {global_vocabulary} rows",
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
                    "Nemotron-H vocabulary member {target} has non-row placement {placement:?}"
                )))
            }
        };
        if selected.as_ref().is_some_and(|current| current != &range) {
            return Err(ParallelPlanError::InvalidTensor(format!(
                "Nemotron-H vocabulary group {logical_name} has inconsistent companion selections"
            )));
        }
        selected = Some(range);
    }
    if !found {
        return Err(ParallelPlanError::InvalidTensor(format!(
            "missing local Nemotron-H vocabulary layout for {logical_name}"
        )));
    }
    let range = VocabularyParallelRange {
        global_vocabulary,
        local: selected.expect("a found vocabulary member supplies a selection"),
    };
    range
        .validate()
        .map_err(|error| ParallelPlanError::InvalidTensor(error.to_string()))?;
    Ok(range)
}

/// Derives complete rank-local target, MTP, vocabulary, and state geometry.
pub fn local_geometry(
    args: &ModelArgs,
    layout: &LocalModelLayout,
) -> Result<LocalGeometry, ParallelPlanError> {
    args.validate()
        .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
    let geometry = local_state_geometry(args, layout)?;
    let target_count = usize::try_from(args.num_hidden_layers).map_err(|_| {
        ParallelPlanError::InvalidGroup("Nemotron-H target layer count exceeds usize".into())
    })?;
    let (target, prediction) = geometry.split_at(target_count);
    let state_layout = state_layout_with_geometry(args, &geometry)
        .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
    let vocabulary = usize::try_from(args.vocab_size).map_err(|_| {
        ParallelPlanError::InvalidGroup("Nemotron-H vocabulary exceeds usize".into())
    })?;
    let embedding_range = vocabulary_range(layout, "model.embeddings", vocabulary)?;
    let output_range = if args.tie_word_embeddings {
        None
    } else {
        Some(vocabulary_range(layout, "lm_head", vocabulary)?)
    };
    let prediction_steps = usize::try_from(args.num_nextn_predict_layers).map_err(|_| {
        ParallelPlanError::InvalidGroup("Nemotron-H prediction-step count exceeds usize".into())
    })?;
    let prediction_pattern = if prediction_steps == 0 {
        0
    } else {
        if prediction.len() % prediction_steps != 0 {
            return Err(ParallelPlanError::InvalidGroup(
                "Nemotron-H MTP geometry does not divide into prediction steps".into(),
            ));
        }
        prediction
            .len()
            .checked_div(prediction_steps)
            .filter(|n| *n > 0)
            .ok_or_else(|| {
                ParallelPlanError::InvalidGroup(
                    "Nemotron-H MTP geometry has an empty prediction pattern".into(),
                )
            })?
    };
    let local = LocalGeometry {
        target: target.to_vec(),
        prediction: prediction.to_vec(),
        embedding_range,
        output_range,
        state_layout,
        architecture_fingerprint: prompt_cache_architecture_fingerprint(args),
        global_target: global_target_geometry(args)?,
        global_prediction: global_prediction_geometry(args)?,
        prediction_steps,
        prediction_pattern,
        tied_head: args.tie_word_embeddings,
    };
    local.validate_for(args)?;
    Ok(local)
}

/// Derives dense target construction geometry for one exact TP/PP partition.
pub fn partition_local_geometry(
    args: &ModelArgs,
    layout: &LocalModelLayout,
    owned_units: Range<usize>,
) -> Result<PartitionLocalGeometry, ParallelPlanError> {
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
    let complete = local_geometry(args, layout)?;
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
        tied_head: complete.tied_head,
        static_roles: expected_static_roles(&owned_units, count),
        boundary_schema: super::TargetBoundarySchema::from_args(args),
        expert_realization: None,
        routed_expert_intermediate_range: None,
        shared_expert_intermediate_range: None,
        expert_banks: Vec::new(),
    };
    geometry.validate_for(args)?;
    Ok(geometry)
}

/// Derives prediction-free routed Nemotron-H geometry from the selected expert plan.
pub fn partition_local_routed_geometry(
    args: &ModelArgs,
    layout: &LocalModelLayout,
    owned_units: Range<usize>,
    topology: eredu_core::ParallelRankTopology,
    realization: &crate::ExpertRealizationPlan<eredu_nn::GroupedRelu2Spec>,
) -> Result<PartitionLocalGeometry, ParallelPlanError> {
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

    let complete = local_geometry(args, layout)?;
    let global_experts = usize::try_from(args.n_routed_experts).map_err(|_| {
        ParallelPlanError::InvalidGroup("Nemotron-H expert count exceeds usize".into())
    })?;
    let local_experts = validate_realization(topology, global_experts, realization)?;
    let local_count = i32::try_from(local_experts.len()).map_err(|_| {
        ParallelPlanError::InvalidGroup("Nemotron-H local expert count exceeds i32".into())
    })?;
    let mut routed_range = None;
    let mut shared_range = None;
    let mut expected_sparse_units = 0;
    for (unit, policy) in args.layer_schedule.iter().copied().enumerate() {
        if policy != LayerPolicy::SparseMoe {
            continue;
        }
        expected_sparse_units += 1;
        let geometry = complete.target_unit(unit).copied().ok_or_else(|| {
            ParallelPlanError::InvalidGroup("missing local Nemotron-H sparse unit".into())
        })?;
        let LayerGeometry::SparseMoe { routed, shared } = geometry else {
            return Err(ParallelPlanError::InvalidGroup(
                "Nemotron-H sparse schedule has non-sparse local geometry".into(),
            ));
        };
        let expected = super::localized_expert_bank_spec(args, unit, local_count, routed)
            .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
        if realization.unit_spec("target", unit) != Some(&expected) {
            return Err(ParallelPlanError::InvalidGroup(format!(
                "Nemotron-H expert plan drifted at unit {unit}"
            )));
        }
        let current_routed = exact_logical_range(
            layout,
            &format!("model.layers.{unit}.moe.experts.up_proj"),
            args.moe_intermediate_size,
        )?;
        let current_shared = exact_logical_range(
            layout,
            &format!("model.layers.{unit}.moe.shared_experts.up_proj.weight"),
            args.moe_shared_expert_intermediate_size,
        )?;
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
    if realization.unit_specs().len() != expected_sparse_units {
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
        .filter(|unit| args.layer_schedule.get(*unit) == Some(&LayerPolicy::SparseMoe))
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
        tied_head: complete.tied_head,
        static_roles: expected_static_roles(&owned_units, count),
        boundary_schema: super::TargetBoundarySchema::from_args(args),
        expert_realization: Some(realization.clone()),
        routed_expert_intermediate_range: Some(routed_range),
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
    args: &ModelArgs,
    tensor_rank: usize,
    tensor_size: usize,
) -> Result<StateLayout, ParallelPlanError> {
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
    let geometry = args
        .layer_schedule
        .iter()
        .map(|policy| match policy {
            LayerPolicy::Mamba => {
                if args.n_groups <= 0 || args.mamba_num_heads % args.n_groups != 0 {
                    return Err(ParallelPlanError::InvalidGroup(
                        "Nemotron-H Mamba heads do not divide into state groups".into(),
                    ));
                }
                let groups = select(args.n_groups, "Mamba groups")?;
                Ok(LayerGeometry::Mamba {
                    heads: groups * (args.mamba_num_heads / args.n_groups),
                    groups,
                })
            }
            LayerPolicy::SelfAttention(_) => Ok(LayerGeometry::Attention {
                query_heads: select(args.num_attention_heads, "query heads")?,
                kv_heads: select(args.num_key_value_heads, "key/value heads")?,
            }),
            LayerPolicy::DenseMlp => Ok(LayerGeometry::DenseMlp {
                intermediate: select(args.intermediate_size, "MLP width")?,
            }),
            LayerPolicy::SparseMoe => Ok(LayerGeometry::SparseMoe {
                routed: args.moe_intermediate_size,
                shared: args.moe_shared_expert_intermediate_size,
            }),
        })
        .collect::<Result<Vec<_>, ParallelPlanError>>()?;
    state_layout_with_geometry(args, &geometry)
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
            "model.embeddings",
            ParameterRole::Vocabulary,
            &modules.embeddings,
            |_, shape| {
                if shape.is_empty() {
                    Err(ParallelPlanError::InvalidTensor(
                        "Nemotron-H embedding is scalar".into(),
                    ))
                } else {
                    Ok(MemberSharding::Balanced { axis: 0 })
                }
            },
        )?,
        module_parameter_group::<B::Tensor, _>(
            "model.norm_f",
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
                        "Nemotron-H output is scalar".into(),
                    ))
                } else {
                    Ok(MemberSharding::Balanced { axis: 0 })
                }
            },
        )?);
    }
    Ok(groups)
}

fn dense_groups<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend>(
    root: &str,
    mlp: &DenseMlp<B>,
    width: i32,
    role: ParameterRole,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    Ok(vec![partitioned_projection_group::<B::Tensor, B::Linear>(
        format!("{root}.intermediate"),
        role,
        &[
            (&mlp.up_proj, ProjectionSharding::Column),
            (&mlp.down_proj, ProjectionSharding::Row),
        ],
        aligned_partition_units(
            root,
            usize::try_from(width).map_err(|_| {
                ParallelPlanError::InvalidGroup(
                    "Nemotron-H intermediate width exceeds usize".into(),
                )
            })?,
            1,
            1,
        )?,
    )?])
}

/// Declares semantic groups for one target physical block.
pub fn layer_parallel_parameter_groups<
    B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
>(
    block: &Block<B>,
    args: &ModelArgs,
    layer: usize,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    let root = format!("model.layers.{layer}");
    block_parallel_parameter_groups(block, args, &root)
}

fn block_parallel_parameter_groups<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend>(
    block: &Block<B>,
    args: &ModelArgs,
    root: &str,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    let mut groups = vec![module_parameter_group::<B::Tensor, _>(
        format!("{root}.norm"),
        ParameterRole::Replicated,
        &block.norm,
        |_, _| Ok(MemberSharding::Replicated),
    )?];
    match &block.operator {
        Operator::Mamba(mamba) => {
            let heads = usize::try_from(args.mamba_num_heads)
                .map_err(|_| ParallelPlanError::InvalidGroup("Mamba heads exceed usize".into()))?;
            let intermediate = usize::try_from(args.mamba_num_heads * args.mamba_head_dim)
                .map_err(|_| ParallelPlanError::InvalidGroup("Mamba width exceeds usize".into()))?;
            let grouped = usize::try_from(args.n_groups * args.ssm_state_size).map_err(|_| {
                ParallelPlanError::InvalidGroup("Mamba state width exceeds usize".into())
            })?;
            let segments = vec![
                0..intermediate,
                intermediate..2 * intermediate,
                2 * intermediate..2 * intermediate + grouped,
                2 * intermediate + grouped..2 * intermediate + 2 * grouped,
                2 * intermediate + 2 * grouped..2 * intermediate + 2 * grouped + heads,
            ];
            groups.push(partitioned_module_parameter_group::<B::Tensor, _>(
                format!("{root}.mamba.heads"),
                ParameterRole::Channels,
                usize::try_from(args.n_groups).map_err(|_| {
                    ParallelPlanError::InvalidGroup("Mamba group count exceeds usize".into())
                })?,
                mamba,
                |metadata, shape| {
                    let name = metadata
                        .linear_companion_of
                        .as_ref()
                        .unwrap_or(&metadata.id)
                        .as_str();
                    if name.ends_with("in_proj.weight") || name.ends_with("in_proj.bias") {
                        Ok(MemberSharding::PartitionedSegments {
                            axis: 0,
                            segments: segments.clone(),
                        })
                    } else if name.ends_with("conv1d.weight")
                        || name.ends_with("conv1d.bias")
                        || name.ends_with("dt_bias")
                        || name.ends_with("A_log")
                        || name.ends_with("D")
                        || name.ends_with("norm.weight")
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
        Operator::Attention(attention) => {
            groups.push(partitioned_module_parameter_group::<B::Tensor, _>(
                format!("{root}.attention.heads"),
                ParameterRole::AttentionHeads,
                usize::try_from(args.num_key_value_heads).map_err(|_| {
                    ParallelPlanError::InvalidGroup("attention head count exceeds usize".into())
                })?,
                attention,
                |metadata, shape| {
                    let name = metadata
                        .linear_companion_of
                        .as_ref()
                        .unwrap_or(&metadata.id)
                        .as_str();
                    if name.ends_with("q_proj.weight")
                        || name.ends_with("k_proj.weight")
                        || name.ends_with("v_proj.weight")
                        || name.ends_with("q_proj.bias")
                        || name.ends_with("k_proj.bias")
                        || name.ends_with("v_proj.bias")
                    {
                        Ok(MemberSharding::Partitioned { axis: 0 })
                    } else if name.ends_with("o_proj.weight") && shape.len() >= 2 {
                        Ok(MemberSharding::Partitioned { axis: 1 })
                    } else {
                        Ok(MemberSharding::Replicated)
                    }
                },
            )?)
        }
        Operator::Dense(mlp) => groups.extend(dense_groups(
            &format!("{root}.mlp"),
            mlp,
            args.intermediate_size,
            ParameterRole::FeedForwardIntermediate,
        )?),
        Operator::Sparse(moe) => {
            groups.push(module_parameter_group::<B::Tensor, _>(
                format!("{root}.moe.gate"),
                ParameterRole::Replicated,
                &moe.gate,
                |_, _| Ok(MemberSharding::Replicated),
            )?);
            groups.push(partitioned_module_parameter_group::<B::Tensor, _>(
                format!("{root}.moe.experts.intermediate"),
                ParameterRole::ExpertIntermediate,
                usize::try_from(args.moe_intermediate_size).map_err(|_| {
                    ParallelPlanError::InvalidGroup("expert width exceeds usize".into())
                })?,
                &moe.experts,
                |metadata, _| {
                    let name = metadata
                        .linear_companion_of
                        .as_ref()
                        .unwrap_or(&metadata.id)
                        .as_str();
                    if name.contains("up_proj") {
                        Ok(MemberSharding::Partitioned { axis: 1 })
                    } else {
                        Ok(MemberSharding::Partitioned { axis: 2 })
                    }
                },
            )?);
            groups.extend(dense_groups(
                &format!("{root}.moe.shared_experts"),
                &moe.shared_experts,
                args.moe_shared_expert_intermediate_size,
                ParameterRole::ExpertIntermediate,
            )?);
        }
    }
    Ok(groups)
}

/// Declares semantic placement for one target or appended prediction unit.
pub fn unit_parallel_parameter_groups<
    B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
>(
    unit: &Unit<B>,
    args: &ModelArgs,
    flat: usize,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    let target = usize::try_from(args.num_hidden_layers)
        .map_err(|_| ParallelPlanError::InvalidGroup("layer count exceeds usize".into()))?;
    match unit {
        Unit::Target(block) if flat < target => layer_parallel_parameter_groups(block, args, flat),
        Unit::Prediction(prediction) if flat >= target => {
            prediction_parallel_parameter_groups(prediction, args, flat - target)
        }
        _ => Err(ParallelPlanError::InvalidGroup(format!(
            "Nemotron-H unit kind does not match flat position {flat}"
        ))),
    }
}

fn prediction_parallel_parameter_groups<
    B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
>(
    unit: &PredictionUnit<B>,
    args: &ModelArgs,
    physical: usize,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    let root = format!("model.mtp.layers.{physical}");
    let mut groups = Vec::new();
    if let Some(norm) = &unit.embedding_norm {
        groups.push(module_parameter_group::<B::Tensor, _>(
            format!("{root}.enorm"),
            ParameterRole::Replicated,
            norm,
            |_, _| Ok(MemberSharding::Replicated),
        )?);
    }
    if let Some(norm) = &unit.hidden_norm {
        groups.push(module_parameter_group::<B::Tensor, _>(
            format!("{root}.hnorm"),
            ParameterRole::Replicated,
            norm,
            |_, _| Ok(MemberSharding::Replicated),
        )?);
    }
    if let Some(fusion) = &unit.fusion {
        groups.push(module_parameter_group::<B::Tensor, _>(
            format!("{root}.eh_proj"),
            ParameterRole::Replicated,
            fusion,
            |_, _| Ok(MemberSharding::Replicated),
        )?);
    }
    groups.extend(block_parallel_parameter_groups(
        &unit.block,
        args,
        &format!("{root}.mixer"),
    )?);
    if let Some(norm) = &unit.final_norm {
        groups.push(module_parameter_group::<B::Tensor, _>(
            format!("{root}.final_layernorm"),
            ParameterRole::Replicated,
            norm,
            |_, _| Ok(MemberSharding::Replicated),
        )?);
    }
    Ok(groups)
}

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_runtime::{LocalTensorLayout, ParameterRole};

    fn args() -> ModelArgs {
        crate::nemotron_h::model_args_from_config_value(&serde_json::json!({
            "model_type":"nemotron_h", "vocab_size":32, "hidden_size":16,
            "intermediate_size":24, "num_hidden_layers":4,
            "hybrid_override_pattern":"M*-E", "num_attention_heads":4,
            "num_key_value_heads":2, "head_dim":4, "mamba_num_heads":4,
            "n_groups":2, "mamba_head_dim":4, "ssm_state_size":3,
            "conv_kernel":3, "n_routed_experts":4, "n_shared_experts":1,
            "moe_intermediate_size":8, "moe_shared_expert_intermediate_size":8,
            "num_experts_per_tok":2, "n_group":2, "topk_group":1,
            "num_nextn_predict_layers":1, "mtp_hybrid_override_pattern":"*E",
            "tie_word_embeddings":false
        }))
        .unwrap()
    }

    fn dense_args() -> ModelArgs {
        crate::nemotron_h::model_args_from_config_value(&serde_json::json!({
            "model_type":"nemotron_h", "vocab_size":32, "hidden_size":16,
            "intermediate_size":24, "num_hidden_layers":4,
            "hybrid_override_pattern":"M*-*", "num_attention_heads":4,
            "num_key_value_heads":2, "head_dim":4, "mamba_num_heads":4,
            "n_groups":2, "mamba_head_dim":4, "ssm_state_size":3,
            "conv_kernel":3, "sliding_window":2, "n_routed_experts":4,
            "n_shared_experts":1, "moe_intermediate_size":8,
            "moe_shared_expert_intermediate_size":8, "num_experts_per_tok":2,
            "n_group":2, "topk_group":1, "num_nextn_predict_layers":0,
            "tie_word_embeddings":false
        }))
        .unwrap()
    }

    fn routed_args() -> ModelArgs {
        crate::nemotron_h::model_args_from_config_value(&serde_json::json!({
            "model_type":"nemotron_h", "vocab_size":32, "hidden_size":16,
            "intermediate_size":24, "num_hidden_layers":4,
            "hybrid_override_pattern":"M*-E", "num_attention_heads":4,
            "num_key_value_heads":2, "head_dim":4, "mamba_num_heads":4,
            "n_groups":2, "mamba_head_dim":4, "ssm_state_size":3,
            "conv_kernel":3, "sliding_window":2, "n_routed_experts":4,
            "n_shared_experts":1, "moe_intermediate_size":8,
            "moe_shared_expert_intermediate_size":8, "num_experts_per_tok":2,
            "n_group":2, "topk_group":1, "num_nextn_predict_layers":0,
            "tie_word_embeddings":false
        }))
        .unwrap()
    }

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
            "model.embeddings.weight",
            "model.embeddings",
            vec![32, 16],
            vec![16, 16],
            range(0, 16),
        );
        insert(
            &mut layout,
            "lm_head.weight",
            "lm_head",
            vec![32, 16],
            vec![16, 16],
            range(0, 16),
        );
        insert(
            &mut layout,
            "model.layers.0.mamba.dt_bias",
            "model.layers.0.mamba.heads",
            vec![4],
            vec![2],
            range(0, 2),
        );
        insert(
            &mut layout,
            "model.layers.1.attention.q_proj.weight",
            "model.layers.1.attention.heads",
            vec![16, 16],
            vec![8, 16],
            range(0, 8),
        );
        insert(
            &mut layout,
            "model.layers.1.attention.k_proj.weight",
            "model.layers.1.attention.heads",
            vec![8, 16],
            vec![4, 16],
            range(0, 4),
        );
        insert(
            &mut layout,
            "model.layers.2.mlp.up_proj.weight",
            "model.layers.2.mlp.intermediate",
            vec![24, 16],
            vec![12, 16],
            range(0, 12),
        );
        insert(
            &mut layout,
            "model.layers.3.moe.experts.up_proj",
            "model.layers.3.moe.experts.intermediate",
            vec![4, 8, 16],
            vec![4, 4, 16],
            range(1, 4),
        );
        insert(
            &mut layout,
            "model.layers.3.moe.shared_experts.up_proj.weight",
            "model.layers.3.moe.shared_experts.intermediate",
            vec![8, 16],
            vec![4, 16],
            range(0, 4),
        );
        insert(
            &mut layout,
            "model.mtp.layers.0.mixer.q_proj.weight",
            "model.mtp.layers.0.mixer.attention.heads",
            vec![16, 16],
            vec![8, 16],
            range(0, 8),
        );
        insert(
            &mut layout,
            "model.mtp.layers.0.mixer.k_proj.weight",
            "model.mtp.layers.0.mixer.attention.heads",
            vec![8, 16],
            vec![4, 16],
            range(0, 4),
        );
        insert(
            &mut layout,
            "model.mtp.layers.1.mixer.experts.up_proj",
            "model.mtp.layers.1.mixer.experts.intermediate",
            vec![4, 8, 16],
            vec![4, 4, 16],
            range(1, 4),
        );
        insert(
            &mut layout,
            "model.mtp.layers.1.mixer.shared_experts.up_proj.weight",
            "model.mtp.layers.1.mixer.shared_experts.intermediate",
            vec![8, 16],
            vec![4, 16],
            range(0, 4),
        );
        layout
    }

    fn valid_dense_layout() -> LocalModelLayout {
        let mut layout = valid_layout();
        insert(
            &mut layout,
            "model.layers.3.attention.q_proj.weight",
            "model.layers.3.attention.heads",
            vec![16, 16],
            vec![8, 16],
            range(0, 8),
        );
        insert(
            &mut layout,
            "model.layers.3.attention.k_proj.weight",
            "model.layers.3.attention.heads",
            vec![8, 16],
            vec![4, 16],
            range(0, 4),
        );
        layout
    }

    fn routed_layout() -> LocalModelLayout {
        let mut layout = valid_layout();
        insert_logical(
            &mut layout,
            "model.layers.3.moe.experts.up_proj",
            "model.layers.3.moe.experts.intermediate",
            vec![4, 8, 16],
            vec![4, 4, 16],
            range(1, 4),
            8,
            0..4,
        );
        insert_logical(
            &mut layout,
            "model.layers.3.moe.shared_experts.up_proj.weight",
            "model.layers.3.moe.shared_experts.intermediate",
            vec![8, 16],
            vec![4, 16],
            range(0, 4),
            8,
            0..4,
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
    ) -> crate::ExpertRealizationPlan<eredu_nn::GroupedRelu2Spec> {
        let args = routed_args();
        let local_count = i32::try_from(
            eredu_core::balanced_contiguous_range(
                4,
                topology.expert_parallel_size(),
                topology.expert_parallel_rank(),
                false,
            )
            .unwrap()
            .len(),
        )
        .unwrap();
        let group = eredu_runtime::ExecutionGroupId::new("target").unwrap();
        let specs = [(
            (group, 3),
            crate::nemotron_h::localized_expert_bank_spec(&args, 3, local_count, 4).unwrap(),
        )]
        .into_iter()
        .collect();
        crate::ExpertRealizationPlan::balanced(4, topology, specs).unwrap()
    }

    #[test]
    fn local_geometry_owns_target_mtp_vocabulary_and_state_together() {
        let args = args();
        let geometry = local_geometry(&args, &valid_layout()).unwrap();
        assert_eq!(geometry.target_units().len(), 4);
        assert_eq!(geometry.prediction_units().len(), 2);
        assert_eq!(
            geometry.target_unit(0),
            Some(&LayerGeometry::Mamba {
                heads: 2,
                groups: 1
            })
        );
        assert_eq!(
            geometry.prediction_unit(0),
            Some(&LayerGeometry::Attention {
                query_heads: 2,
                kv_heads: 1
            })
        );
        assert_eq!(geometry.embedding_range().local, 0..16);
        assert_eq!(geometry.output_range().unwrap().local, 0..16);
        assert_ne!(
            geometry.state_layout(),
            &crate::nemotron_h::state_layout(&args).unwrap()
        );
        geometry.validate_for(&args).unwrap();
    }

    #[test]
    fn partition_geometry_retains_only_owned_units_and_exact_mixed_state() {
        let args = dense_args();
        let layout = valid_dense_layout();
        let geometry = partition_local_geometry(&args, &layout, 2..4).unwrap();
        assert_eq!(geometry.owned_units(), 2..4);
        assert_eq!(geometry.local_unit_count(), 2);
        assert!(geometry.unit(1).is_none());
        assert_eq!(
            geometry.unit(2),
            Some(&LayerGeometry::DenseMlp { intermediate: 12 })
        );
        assert_eq!(
            geometry.unit(3),
            Some(&LayerGeometry::Attention {
                query_heads: 2,
                kv_heads: 1
            })
        );
        assert!(matches!(
            geometry.complete_state_layout().layer(0),
            Some(eredu_core::cache::LayerCachePolicy::FixedState { tensors })
                if tensors.len() == 2
                    && tensors[0].role
                        == eredu_core::cache::StateTensorRole::Convolution { slot: 0 }
                    && tensors[1].role == eredu_core::cache::StateTensorRole::Recurrent
                    && tensors[1].dtype == eredu_core::cache::StateTensorDtype::Float32
        ));
        assert!(matches!(
            geometry.complete_state_layout().layer(1),
            Some(eredu_core::cache::LayerCachePolicy::KeyValue {
                attention: eredu_core::AttentionPolicy::Sliding { window },
                num_key_value_heads,
                head_dim,
            }) if window.get() == 2 && num_key_value_heads.get() == 1 && head_dim.get() == 4
        ));
        assert!(matches!(
            geometry.complete_state_layout().layer(2),
            Some(eredu_core::cache::LayerCachePolicy::NoState)
        ));
        assert!(partition_local_geometry(&args, &layout, 0..0).is_err());
        assert!(partition_local_geometry(&args, &layout, 3..5).is_err());
        assert!(partition_local_geometry(&self::args(), &layout, 0..1).is_err());
        let mut predicted = args.clone();
        predicted.num_nextn_predict_layers = 1;
        predicted.mtp_hybrid_override_pattern = Some("*".into());
        assert!(partition_local_geometry(&predicted, &layout, 0..1).is_err());

        let mut malformed = layout;
        insert(
            &mut malformed,
            "model.layers.3.attention.q_proj.weight",
            "model.layers.3.attention.heads",
            vec![16, 16],
            vec![7, 16],
            range(0, 7),
        );
        assert!(partition_local_geometry(&args, &malformed, 2..4).is_err());
    }

    #[test]
    fn routed_partition_retains_hybrid_state_and_exact_shared_and_compact_banks() {
        let shape = eredu_core::ParallelTopology::new(2, 2, 2, 1).unwrap();
        let topology = eredu_core::ParallelRankTopology::new(shape, 5).unwrap();
        let plan = realization(topology);
        let geometry = partition_local_routed_geometry(
            &routed_args(),
            &routed_layout(),
            2..4,
            topology,
            &plan,
        )
        .unwrap();

        assert_eq!(geometry.owned_units(), 2..4);
        assert_eq!(geometry.static_roles(), ["norm", "output"]);
        assert_eq!(
            geometry.boundary_schema(),
            &crate::nemotron_h::TargetBoundarySchema::from_args(&routed_args())
        );
        assert_eq!(geometry.routed_expert_intermediate_range(), Some(0..4));
        assert_eq!(geometry.shared_expert_intermediate_range(), Some(0..4));
        assert_eq!(geometry.expert_banks().len(), 2);
        assert_eq!(geometry.state_global_offset(), 2);
        assert_eq!(geometry.local_state_layout().unwrap().len(), 2);
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
        assert_eq!(
            geometry
                .expert_banks()
                .iter()
                .map(|bank| (
                    bank.global_unit(),
                    bank.global_expert(),
                    bank.owner_local_expert()
                ))
                .collect::<Vec<_>>(),
            vec![(3, 2, 0), (3, 3, 1)]
        );
        assert!(matches!(
            geometry.complete_state_layout().layer(0),
            Some(eredu_core::cache::LayerCachePolicy::FixedState { .. })
        ));
        assert!(matches!(
            geometry.complete_state_layout().layer(1),
            Some(eredu_core::cache::LayerCachePolicy::KeyValue { .. })
        ));
        assert_eq!(geometry.complete_state_layout().len(), 4);

        let mut predicted = routed_args();
        predicted.num_nextn_predict_layers = 1;
        assert!(partition_local_routed_geometry(
            &predicted,
            &routed_layout(),
            2..4,
            topology,
            &plan
        )
        .is_err());
        let wrong_topology = eredu_core::ParallelRankTopology::new(shape, 4).unwrap();
        assert!(partition_local_routed_geometry(
            &routed_args(),
            &routed_layout(),
            2..4,
            topology,
            &realization(wrong_topology)
        )
        .is_err());
    }

    #[test]
    fn preadmission_partitioned_state_is_exactly_tp_local_and_rejects_topology_drift() {
        let args = dense_args();
        let layout = partitioned_state_layout(&args, 1, 2).unwrap();
        assert!(matches!(
            layout.layer(0),
            Some(eredu_core::cache::LayerCachePolicy::FixedState { tensors })
                if tensors.len() == 2
                    && matches!(tensors[0].shape.as_slice(), [
                        eredu_core::cache::StateTensorDimension::Batch,
                        eredu_core::cache::StateTensorDimension::Fixed(history),
                        eredu_core::cache::StateTensorDimension::Fixed(width),
                    ] if history.get() == 2 && width.get() == 14)
                    && matches!(tensors[1].shape.as_slice(), [
                        eredu_core::cache::StateTensorDimension::Batch,
                        eredu_core::cache::StateTensorDimension::Fixed(heads),
                        eredu_core::cache::StateTensorDimension::Fixed(head_dim),
                        eredu_core::cache::StateTensorDimension::Fixed(state),
                    ] if heads.get() == 2 && head_dim.get() == 4 && state.get() == 3)
        ));
        assert!(matches!(
            layout.layer(1),
            Some(eredu_core::cache::LayerCachePolicy::KeyValue {
                num_key_value_heads,
                head_dim,
                ..
            }) if num_key_value_heads.get() == 1 && head_dim.get() == 4
        ));
        assert!(partitioned_state_layout(&args, 2, 2).is_err());
        let mut drifted = args;
        drifted.mamba_num_heads = 6;
        drifted.n_groups = 3;
        assert!(partitioned_state_layout(&drifted, 0, 2).is_ok());
        assert!(partitioned_state_layout(&drifted, 1, 2).is_ok());
    }

    #[test]
    fn local_geometry_rejects_incomplete_or_zero_mtp_heads() {
        let args = args();
        let mut layout = valid_layout();
        insert(
            &mut layout,
            "model.mtp.layers.0.mixer.q_proj.weight",
            "model.mtp.layers.0.mixer.attention.heads",
            vec![16, 16],
            vec![7, 16],
            range(0, 7),
        );
        assert!(local_geometry(&args, &layout)
            .unwrap_err()
            .to_string()
            .contains("complete heads"));

        let mut layout = valid_layout();
        insert(
            &mut layout,
            "model.mtp.layers.0.mixer.k_proj.weight",
            "model.mtp.layers.0.mixer.attention.heads",
            vec![8, 16],
            vec![0, 16],
            range(0, 0),
        );
        assert!(local_geometry(&args, &layout)
            .unwrap_err()
            .to_string()
            .contains("invalid local width 0"));
    }

    #[test]
    fn local_geometry_rejects_vocabulary_companion_drift() {
        let args = args();
        let mut layout = valid_layout();
        insert(
            &mut layout,
            "model.embeddings.scales",
            "model.embeddings",
            vec![32, 1],
            vec![16, 1],
            TensorPlacement::Range {
                axis: 0,
                start: 16,
                end: 32,
            },
        );
        assert!(local_geometry(&args, &layout)
            .unwrap_err()
            .to_string()
            .contains("inconsistent companion selections"));
    }

    #[test]
    fn local_geometry_validation_preserves_prediction_grouping() {
        let args = args();
        let mut geometry = local_geometry(&args, &valid_layout()).unwrap();
        geometry.prediction_steps = 2;
        geometry.prediction_pattern = 1;
        assert!(geometry
            .validate_for(&args)
            .unwrap_err()
            .to_string()
            .contains("different model configuration"));
    }
}
