//! Semantic parallel placement for DeepSeek target and prediction units.

use std::ops::Range;

use eredu_nn::{DeclareLogicalParameters, VocabularyParallelRange};
use eredu_runtime::{
    ArchitectureParameterDescription, ArchitecturePartition, ExecutionGraph, ExecutionUnitLayout,
    LocalModelLayout, MemberSharding, OwnedParameterGroupSpec, ParallelPlanError,
    ParameterGroupOwner, ParameterGroupSpec, ParameterRole, PartitionOwnership, StateLayout,
    TensorPlacement,
};

use super::{
    v3, v3_architecture_fingerprint, v4, v4_architecture_fingerprint, LayerPolicy, V3Args, V4Args,
};

/// Planner-derived construction and mutable-state geometry for DeepSeek V3.
#[derive(Debug, Clone)]
pub struct V3LocalGeometry {
    args: V3Args,
    embedding_range: VocabularyParallelRange,
    output_range: VocabularyParallelRange,
    state_layout: StateLayout,
    routed_expert_intermediate_range: Option<Range<usize>>,
    shared_expert_intermediate_range: Option<Range<usize>>,
    parameters: std::sync::Arc<ArchitectureParameterDescription>,
    architecture_fingerprint: String,
}

impl V3LocalGeometry {
    /// Returns the rank-local arguments used to build every target and MTP unit.
    pub const fn args(&self) -> &V3Args {
        &self.args
    }

    /// Returns input-embedding vocabulary ownership.
    pub const fn embedding_range(&self) -> &VocabularyParallelRange {
        &self.embedding_range
    }

    /// Returns output-head vocabulary ownership.
    pub const fn output_range(&self) -> &VocabularyParallelRange {
        &self.output_range
    }

    /// Returns the authoritative rank-local state layout.
    pub const fn state_layout(&self) -> &StateLayout {
        &self.state_layout
    }

    /// Returns the exact checkpoint-global TP channel range for every routed bank.
    pub fn routed_expert_intermediate_range(&self) -> Option<Range<usize>> {
        self.routed_expert_intermediate_range.clone()
    }

    /// Returns the exact checkpoint-global TP channel range for shared experts.
    pub fn shared_expert_intermediate_range(&self) -> Option<Range<usize>> {
        self.shared_expert_intermediate_range.clone()
    }

    pub(super) fn validate_for(&self, args: &V3Args) -> Result<(), ParallelPlanError> {
        let parameters = v3_parameters(args)?;
        if self.parameters.as_ref() != &parameters
            || self.architecture_fingerprint != v3_architecture_fingerprint(args)
        {
            return Err(invalid(
                "rank-local V3 geometry belongs to a different model configuration",
            ));
        }
        self.embedding_range
            .validate_global_rows(args.vocab_size)
            .map_err(|error| invalid(error.to_string()))?;
        self.output_range
            .validate_global_rows(args.vocab_size)
            .map_err(|error| invalid(error.to_string()))?;
        let expected = v3::state_layout(&self.args)
            .map_err(|error| invalid(format!("invalid local V3 state geometry: {error}")))?;
        if expected != self.state_layout {
            return Err(invalid(
                "rank-local V3 state layout drifted from unit geometry",
            ));
        }
        let local_expert = usize::try_from(self.args.moe_intermediate_size)
            .map_err(|_| invalid("local V3 expert width exceeds usize"))?;
        let local_shared = local_expert
            .checked_mul(
                usize::try_from(args.n_shared_experts)
                    .map_err(|_| invalid("V3 shared expert count exceeds usize"))?,
            )
            .ok_or_else(|| invalid("local V3 shared expert width overflowed usize"))?;
        match (
            &self.routed_expert_intermediate_range,
            &self.shared_expert_intermediate_range,
        ) {
            (Some(routed), Some(shared))
                if routed.len() == local_expert && shared.len() == local_shared => {}
            (None, None) => {}
            _ => {
                return Err(invalid(
                    "rank-local V3 routed/shared expert channel ranges drifted",
                ));
            }
        }
        Ok(())
    }
}

/// Planner-derived construction and mutable-state geometry for DeepSeek V4.
#[derive(Debug, Clone)]
pub struct V4LocalGeometry {
    args: V4Args,
    embedding_range: VocabularyParallelRange,
    output_range: VocabularyParallelRange,
    state_layout: StateLayout,
    routed_expert_intermediate_range: Option<Range<usize>>,
    shared_expert_intermediate_range: Option<Range<usize>>,
    parameters: std::sync::Arc<ArchitectureParameterDescription>,
    architecture_fingerprint: String,
}

impl V4LocalGeometry {
    /// Returns the rank-local arguments used to build target and draft units.
    pub const fn args(&self) -> &V4Args {
        &self.args
    }

    /// Returns input-embedding vocabulary ownership.
    pub const fn embedding_range(&self) -> &VocabularyParallelRange {
        &self.embedding_range
    }

    /// Returns output-head vocabulary ownership.
    pub const fn output_range(&self) -> &VocabularyParallelRange {
        &self.output_range
    }

    /// Returns the authoritative rank-local state layout.
    pub const fn state_layout(&self) -> &StateLayout {
        &self.state_layout
    }

    /// Returns the exact checkpoint-global TP channel range for every routed bank.
    pub fn routed_expert_intermediate_range(&self) -> Option<Range<usize>> {
        self.routed_expert_intermediate_range.clone()
    }

    /// Returns the exact checkpoint-global TP channel range for shared experts.
    pub fn shared_expert_intermediate_range(&self) -> Option<Range<usize>> {
        self.shared_expert_intermediate_range.clone()
    }

    pub(super) fn validate_for(&self, args: &V4Args) -> Result<(), ParallelPlanError> {
        let parameters = v4_parameters(args)?;
        if self.parameters.as_ref() != &parameters
            || self.architecture_fingerprint != v4_architecture_fingerprint(args)
        {
            return Err(invalid(
                "rank-local V4 geometry belongs to a different model configuration",
            ));
        }
        self.embedding_range
            .validate_global_rows(args.vocab_size)
            .map_err(|error| invalid(error.to_string()))?;
        self.output_range
            .validate_global_rows(args.vocab_size)
            .map_err(|error| invalid(error.to_string()))?;
        let expected = v4::state_layout(&self.args)
            .map_err(|error| invalid(format!("invalid local V4 state geometry: {error}")))?;
        if expected != self.state_layout {
            return Err(invalid(
                "rank-local V4 state layout drifted from unit geometry",
            ));
        }
        let local_expert = usize::try_from(self.args.moe_intermediate_size)
            .map_err(|_| invalid("local V4 expert width exceeds usize"))?;
        let local_shared = local_expert
            .checked_mul(
                usize::try_from(args.n_shared_experts)
                    .map_err(|_| invalid("V4 shared expert count exceeds usize"))?,
            )
            .ok_or_else(|| invalid("local V4 shared expert width overflowed usize"))?;
        match (
            &self.routed_expert_intermediate_range,
            &self.shared_expert_intermediate_range,
        ) {
            (Some(routed), Some(shared))
                if routed.len() == local_expert && shared.len() == local_shared => {}
            (None, None) => {}
            _ => {
                return Err(invalid(
                    "rank-local V4 routed/shared expert channel ranges drifted",
                ));
            }
        }
        Ok(())
    }
}

fn local_axis(
    layout: &LocalModelLayout,
    target: &str,
    axis: usize,
    family: &str,
) -> Result<usize, ParallelPlanError> {
    let tensor = layout.tensor(target).ok_or_else(|| {
        invalid(format!(
            "missing local {family} tensor placement for {target}"
        ))
    })?;
    let local = tensor
        .local_shape()
        .get(axis)
        .copied()
        .ok_or_else(|| invalid(format!("{family} tensor {target} has no local axis {axis}")))?;
    let global = tensor.global_shape().get(axis).copied().ok_or_else(|| {
        invalid(format!(
            "{family} tensor {target} has no global axis {axis}"
        ))
    })?;
    if local == 0 || local > global {
        return Err(invalid(format!(
            "{family} tensor {target} has invalid local width {local} of {global}"
        )));
    }
    Ok(local)
}

fn vocabulary_range(
    layout: &LocalModelLayout,
    logical_name: &str,
    weight: &str,
    global_vocabulary: usize,
    family: &str,
) -> Result<VocabularyParallelRange, ParallelPlanError> {
    // Vocabulary coordinates belong to the weight rows. FP8 scale rows index
    // blocks of vocabulary entries; their physical ranges are compiled by the
    // format-aware parameter planner and are not token coordinates.
    let tensor = layout
        .tensor(weight)
        .ok_or_else(|| invalid(format!("missing local {family} vocabulary weight {weight}")))?;
    if tensor.logical_name() != logical_name
        || tensor.global_shape().first().copied() != Some(global_vocabulary)
    {
        return Err(invalid(format!(
            "{family} vocabulary weight {weight} has group {:?} and shape {:?}, expected {logical_name} with {global_vocabulary} rows",
            tensor.logical_name(),
            tensor.global_shape()
        )));
    }
    let local = match tensor.placement() {
        TensorPlacement::Range {
            axis: 0,
            start,
            end,
        } => *start..*end,
        TensorPlacement::Replicated => 0..global_vocabulary,
        placement => {
            return Err(invalid(format!(
                "{family} vocabulary weight {weight} has non-row placement {placement:?}"
            )));
        }
    };
    let range = VocabularyParallelRange {
        global_vocabulary,
        local,
    };
    range
        .validate()
        .map_err(|error| invalid(error.to_string()))?;
    Ok(range)
}

fn same_local_axis(
    layout: &LocalModelLayout,
    targets: impl IntoIterator<Item = String>,
    axis: usize,
    family: &str,
    label: &str,
) -> Result<usize, ParallelPlanError> {
    let mut selected = None;
    for target in targets {
        let width = local_axis(layout, &target, axis, family)?;
        if selected.is_some_and(|current| current != width) {
            return Err(invalid(format!(
                "{family} {label} placement differs between execution units"
            )));
        }
        selected = Some(width);
    }
    selected.ok_or_else(|| invalid(format!("{family} has no {label} placement")))
}

fn same_logical_range(
    layout: &LocalModelLayout,
    targets: impl IntoIterator<Item = String>,
    global_units: usize,
    family: &str,
    label: &str,
) -> Result<Option<Range<usize>>, ParallelPlanError> {
    let mut selected: Option<Range<usize>> = None;
    let mut omitted = false;
    for target in targets {
        let Some(tensor) = layout.tensor(&target) else {
            omitted = true;
            continue;
        };
        if tensor.logical_units().is_none() && tensor.logical_range().is_none() {
            omitted = true;
            continue;
        }
        let range = tensor
            .expanded_logical_range(global_units)
            .map_err(|error| invalid(format!("{family} {label} tensor {target}: {error}")))?;
        if selected.as_ref().is_some_and(|current| current != &range) {
            return Err(invalid(format!(
                "{family} {label} placement differs between execution units"
            )));
        }
        selected = Some(range);
    }
    if omitted && selected.is_some() {
        return Err(invalid(format!(
            "{family} {label} placement mixes exact and shape-only semantic ranges"
        )));
    }
    Ok(selected)
}

/// Derives complete V3 rank-local geometry exclusively from a resolved plan.
pub fn v3_local_geometry(
    args: &V3Args,
    layout: &LocalModelLayout,
) -> Result<V3LocalGeometry, ParallelPlanError> {
    args.validate()
        .map_err(|error| invalid(error.to_string()))?;
    let target = usize::try_from(args.num_hidden_layers)
        .map_err(|_| invalid("V3 target layer count exceeds usize"))?;
    let total = usize::try_from(args.num_hidden_layers + args.num_nextn_predict_layers)
        .map_err(|_| invalid("V3 total layer count exceeds usize"))?;
    let attention =
        super::v3_specs::attention_spec(args, "model.layers.0.self_attn").map_err(nn_invalid)?;
    let head_width = attention.geometry.query_head_dimensions() as usize;
    let query_targets = (0..total)
        .map(|layer| {
            let declaration =
                super::v3_specs::attention_spec(args, &format!("model.layers.{layer}.self_attn"))
                    .map_err(nn_invalid)?;
            let query = match declaration.query {
                super::v3_specs::QuerySpec::Direct(query) => query,
                super::v3_specs::QuerySpec::LowRank(query) => query.second,
            };
            Ok(query.weight.id.as_str().to_owned())
        })
        .collect::<Result<Vec<_>, ParallelPlanError>>()?;
    let query_width = same_local_axis(layout, query_targets, 0, "V3", "attention heads")?;
    if query_width % head_width != 0 {
        return Err(invalid(
            "local V3 query width does not contain complete heads",
        ));
    }
    let mut dense = args.intermediate_size;
    let mut routed_width = args.moe_intermediate_size;
    let heads = i32::try_from(query_width / head_width)
        .map_err(|_| invalid("local V3 attention head count exceeds i32"))?;

    let dense_layers = (0..target)
        .filter(|layer| args.layer_schedule.get(*layer) != Some(&LayerPolicy::SparseMoe));
    if let Some(first) = dense_layers.clone().next() {
        let width = same_local_axis(
            layout,
            dense_layers.map(|layer| format!("model.layers.{layer}.mlp.gate_proj.weight")),
            0,
            "V3",
            "dense intermediate",
        )?;
        dense = i32::try_from(width).map_err(|_| invalid("local V3 dense width exceeds i32"))?;
        let _ = first;
    }
    let sparse_layers = (0..total)
        .filter(|layer| {
            *layer >= target || args.layer_schedule.get(*layer) == Some(&LayerPolicy::SparseMoe)
        })
        .collect::<Vec<_>>();
    let (routed_expert_intermediate_range, shared_expert_intermediate_range) =
        if sparse_layers.is_empty() {
            (None, None)
        } else {
            let fused = same_local_axis(
                layout,
                sparse_layers
                    .iter()
                    .map(|layer| format!("model.layers.{layer}.mlp.experts.gate_up_proj")),
                1,
                "V3",
                "expert intermediate",
            )?;
            if fused % 2 != 0 {
                return Err(invalid("local V3 packed expert width is not even"));
            }
            routed_width = i32::try_from(fused / 2)
                .map_err(|_| invalid("local V3 expert width exceeds i32"))?;
            let global_routed = dim(args.moe_intermediate_size)?;
            let routed = same_logical_range(
                layout,
                sparse_layers
                    .iter()
                    .map(|layer| format!("model.layers.{layer}.mlp.experts.gate_up_proj")),
                global_routed,
                "V3",
                "routed expert intermediate",
            )?;
            let global_shared = global_routed
                .checked_mul(dim(args.n_shared_experts)?)
                .ok_or_else(|| invalid("V3 shared expert width overflowed usize"))?;
            let shared = same_logical_range(
                layout,
                sparse_layers.iter().map(|layer| {
                    format!("model.layers.{layer}.mlp.shared_experts.gate_proj.weight")
                }),
                global_shared,
                "V3",
                "shared expert intermediate",
            )?;
            (routed, shared)
        };
    let local = args
        .with_local_geometry(heads, dense, routed_width)
        .map_err(|e| invalid(e.to_string()))?;
    local
        .validate()
        .map_err(|error| invalid(error.to_string()))?;
    let global_vocabulary = usize::try_from(args.vocab_size)
        .map_err(|_| invalid("V3 vocabulary size exceeds usize"))?;
    let geometry = V3LocalGeometry {
        state_layout: v3::state_layout(&local)
            .map_err(|error| invalid(format!("invalid local V3 state layout: {error}")))?,
        embedding_range: vocabulary_range(
            layout,
            "model.embed_tokens",
            "model.embed_tokens.weight",
            global_vocabulary,
            "V3",
        )?,
        output_range: vocabulary_range(
            layout,
            "lm_head",
            "lm_head.weight",
            global_vocabulary,
            "V3",
        )?,
        routed_expert_intermediate_range,
        shared_expert_intermediate_range,
        parameters: std::sync::Arc::new(v3_parameters(args)?),
        architecture_fingerprint: v3_architecture_fingerprint(args),
        args: local,
    };
    geometry.validate_for(args)?;
    Ok(geometry)
}

/// Derives complete V4 rank-local geometry exclusively from a resolved plan.
pub fn v4_local_geometry(
    args: &V4Args,
    layout: &LocalModelLayout,
) -> Result<V4LocalGeometry, ParallelPlanError> {
    args.validate()
        .map_err(|error| invalid(error.to_string()))?;
    let total = usize::try_from(args.num_hidden_layers + args.num_nextn_predict_layers)
        .map_err(|_| invalid("V4 total layer count exceeds usize"))?;
    let roots = (0..total)
        .map(|layer| {
            if layer < args.num_hidden_layers as usize {
                format!("layers.{layer}")
            } else {
                format!("mtp.{}", layer - args.num_hidden_layers as usize)
            }
        })
        .collect::<Vec<_>>();
    let attention = roots
        .iter()
        .enumerate()
        .map(|(layer, root)| {
            super::v4_specs::attention_spec(args, layer, &format!("{root}.attn"))
                .map_err(nn_invalid)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let routed = roots
        .iter()
        .enumerate()
        .map(|(layer, root)| {
            super::v4_specs::moe_spec(args, layer, &format!("{root}.ffn"), None).map_err(nn_invalid)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let query_width = same_local_axis(
        layout,
        attention
            .iter()
            .map(|spec| spec.query.second.weight.id.as_str().to_owned()),
        0,
        "V4",
        "attention heads",
    )?;
    let head_dim = usize::try_from(attention[0].geometry.key_dimensions())
        .map_err(|_| invalid("V4 attention head width exceeds usize"))?;
    if query_width % head_dim != 0 {
        return Err(invalid(
            "local V4 query width does not contain complete heads",
        ));
    }
    let output_width = same_local_axis(
        layout,
        attention
            .iter()
            .map(|spec| spec.output_a.weight.id.as_str().to_owned()),
        0,
        "V4",
        "attention output groups",
    )?;
    let output_rank = usize::try_from(attention[0].output_rank())
        .map_err(|_| invalid("V4 output rank exceeds usize"))?;
    if output_width % output_rank != 0 {
        return Err(invalid(
            "local V4 output projection does not contain complete groups",
        ));
    }
    let fused = same_local_axis(
        layout,
        routed.iter().map(|spec| {
            let eredu_nn::GatedProductGroupLayout::Packed { gate_up, .. } =
                spec.routed.parameters.experts.layout()
            else {
                unreachable!("V4 declares packed experts")
            };
            gate_up.weight().id.as_str().to_owned()
        }),
        1,
        "V4",
        "expert intermediate",
    )?;
    if fused % 2 != 0 {
        return Err(invalid("local V4 packed expert width is not even"));
    }
    let heads = i32::try_from(query_width / head_dim)
        .map_err(|_| invalid("local V4 attention head count exceeds i32"))?;
    let groups = i32::try_from(output_width / output_rank)
        .map_err(|_| invalid("local V4 output group count exceeds i32"))?;
    let routed_width =
        i32::try_from(fused / 2).map_err(|_| invalid("local V4 expert width exceeds i32"))?;
    let global_routed = dim(routed[0]
        .routed
        .parameters
        .experts
        .intermediate_dimensions())?;
    let routed_expert_intermediate_range = same_logical_range(
        layout,
        routed.iter().map(|spec| {
            let eredu_nn::GatedProductGroupLayout::Packed { gate_up, .. } =
                spec.routed.parameters.experts.layout()
            else {
                unreachable!("V4 declares packed experts")
            };
            gate_up.weight().id.as_str().to_owned()
        }),
        global_routed,
        "V4",
        "routed expert intermediate",
    )?;
    let global_shared = global_routed
        .checked_mul(dim(args.n_shared_experts)?)
        .ok_or_else(|| invalid("V4 shared expert width overflowed usize"))?;
    let shared_expert_intermediate_range = same_logical_range(
        layout,
        roots
            .iter()
            .map(|root| format!("{root}.ffn.shared_experts.w1.weight")),
        global_shared,
        "V4",
        "shared expert intermediate",
    )?;
    let local = args
        .with_local_geometry(heads, groups, routed_width)
        .map_err(|e| invalid(e.to_string()))?;
    local
        .validate()
        .map_err(|error| invalid(error.to_string()))?;
    let global_vocabulary = usize::try_from(args.vocab_size)
        .map_err(|_| invalid("V4 vocabulary size exceeds usize"))?;
    let geometry = V4LocalGeometry {
        state_layout: v4::state_layout(&local)
            .map_err(|error| invalid(format!("invalid local V4 state layout: {error}")))?,
        embedding_range: vocabulary_range(
            layout,
            "embed",
            "embed.weight",
            global_vocabulary,
            "V4",
        )?,
        output_range: vocabulary_range(layout, "head", "head.weight", global_vocabulary, "V4")?,
        routed_expert_intermediate_range,
        shared_expert_intermediate_range,
        parameters: std::sync::Arc::new(v4_parameters(args)?),
        architecture_fingerprint: v4_architecture_fingerprint(args),
        args: local,
    };
    geometry.validate_for(args)?;
    Ok(geometry)
}

/// One compact routed bank owned by a DeepSeek PP×EP rank.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct PartitionExpertBankOwnership {
    global_unit: usize,
    global_expert: usize,
    owner_local_expert: usize,
    intermediate_range: Range<usize>,
}

impl PartitionExpertBankOwnership {
    /// Architecture-global target unit containing this bank.
    pub const fn global_unit(&self) -> usize {
        self.global_unit
    }

    /// Checkpoint-global expert identity used as the addressable bank key.
    pub const fn global_expert(&self) -> usize {
        self.global_expert
    }

    /// Compact owner-local execution identity used inside the realized bank.
    pub const fn owner_local_expert(&self) -> usize {
        self.owner_local_expert
    }

    /// Exact checkpoint-global TP intermediate-channel slice.
    pub fn intermediate_range(&self) -> Range<usize> {
        self.intermediate_range.clone()
    }

    /// Stable addressable cache identity; the local ordinal is never used here.
    pub const fn bank_key(&self) -> eredu_runtime::ParameterBankKey {
        eredu_runtime::ParameterBankKey::new(0, self.global_unit, self.global_expert)
    }
}

/// Exact prediction-free V3 geometry owned by one TP×PP×EP rank.
#[derive(Debug, Clone)]
pub struct V3PartitionLocalGeometry {
    target_units: Range<usize>,
    local: V3LocalGeometry,
    static_roles: Vec<String>,
    expert_realization: crate::ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>,
    expert_banks: Vec<PartitionExpertBankOwnership>,
}

/// Exact prediction-free V4 geometry owned by one TP×PP×EP rank.
#[derive(Debug, Clone)]
pub struct V4PartitionLocalGeometry {
    target_units: Range<usize>,
    local: V4LocalGeometry,
    static_roles: Vec<String>,
    expert_realization: crate::ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>,
    expert_banks: Vec<PartitionExpertBankOwnership>,
}

macro_rules! partition_local_geometry_accessors {
    ($ty:ty, $local:ty) => {
        impl $ty {
            /// Architecture-global target units owned by this pipeline rank.
            pub fn target_units(&self) -> Range<usize> {
                self.target_units.clone()
            }

            /// Complete planner-derived TP-local geometry before PP slicing.
            pub const fn local_geometry(&self) -> &$local {
                &self.local
            }

            /// Exact local compressed/pooling state slice for the owned target units.
            pub fn local_state_layout(&self) -> Result<StateLayout, ParallelPlanError> {
                self.local
                    .state_layout()
                    .slice(self.target_units.clone())
                    .map_err(|error| invalid(error.to_string()))
            }

            /// Static roles placed on this input/output owner.
            pub fn static_roles(&self) -> &[String] {
                &self.static_roles
            }

            /// Immutable selected routed-expert authority.
            pub const fn expert_realization(
                &self,
            ) -> &crate::ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec> {
                &self.expert_realization
            }

            /// Compact banks at the PP-layer ∩ EP-expert ∩ TP-channel intersection.
            pub fn expert_banks(&self) -> &[PartitionExpertBankOwnership] {
                &self.expert_banks
            }
        }
    };
}

partition_local_geometry_accessors!(V3PartitionLocalGeometry, V3LocalGeometry);
partition_local_geometry_accessors!(V4PartitionLocalGeometry, V4LocalGeometry);

/// Validated architecture-owned V3 partition handoff.
#[derive(Debug, Clone)]
pub struct V3PartitionLocalFoundation {
    geometry: V3PartitionLocalGeometry,
    resident_parameter_targets: Vec<String>,
    routed_parameter_targets: Vec<String>,
}

/// Validated architecture-owned V4 partition handoff.
#[derive(Debug, Clone)]
pub struct V4PartitionLocalFoundation {
    geometry: V4PartitionLocalGeometry,
    resident_parameter_targets: Vec<String>,
    routed_parameter_targets: Vec<String>,
}

macro_rules! partition_local_foundation_accessors {
    ($ty:ty, $geometry:ty) => {
        impl $ty {
            /// Exact family-local construction and state geometry.
            pub const fn geometry(&self) -> &$geometry {
                &self.geometry
            }

            /// Selected static, attention, router, shared-expert, and other resident targets.
            pub fn resident_parameter_targets(&self) -> &[String] {
                &self.resident_parameter_targets
            }

            /// Compact addressable expert recipes for packed source-bank targets.
            pub fn routed_parameter_targets(&self) -> &[String] {
                &self.routed_parameter_targets
            }
        }
    };
}

partition_local_foundation_accessors!(V3PartitionLocalFoundation, V3PartitionLocalGeometry);
partition_local_foundation_accessors!(V4PartitionLocalFoundation, V4PartitionLocalGeometry);

fn require_prediction_free_v3(args: &V3Args) -> Result<(), ParallelPlanError> {
    if args.num_nextn_predict_layers != 0 {
        return Err(invalid(
            "partition-local DeepSeek-V3 does not admit embedded prediction units",
        ));
    }
    if !args.has_sparse_moe_layers() {
        return Err(invalid(
            "partition-local DeepSeek-V3 routed execution requires sparse target units",
        ));
    }
    Ok(())
}

fn require_prediction_free_v4(args: &V4Args) -> Result<(), ParallelPlanError> {
    if args.num_nextn_predict_layers != 0 || args.dspark.is_some() {
        return Err(invalid(
            "partition-local DeepSeek-V4 does not admit embedded prediction units",
        ));
    }
    Ok(())
}

fn target_range(
    group_ranges: impl IntoIterator<Item = (impl AsRef<str>, Range<usize>)>,
    target_count: usize,
) -> Result<Range<usize>, ParallelPlanError> {
    let mut target = None;
    for (group, range) in group_ranges {
        if group.as_ref() != crate::decoder::TARGET_EXECUTION_GROUP {
            return Err(invalid(format!(
                "prediction-free DeepSeek partition contains unsupported group {:?}",
                group.as_ref()
            )));
        }
        if range.is_empty() || range.end > target_count || target.replace(range).is_some() {
            return Err(invalid("invalid or duplicate DeepSeek target range"));
        }
    }
    target.ok_or_else(|| invalid("DeepSeek partition has no target range"))
}

fn validate_expert_topology(
    global_experts: usize,
    topology: eredu_core::ParallelRankTopology,
    realization: &crate::ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>,
) -> Result<Vec<usize>, ParallelPlanError> {
    let expected = eredu_core::balanced_contiguous_range(
        global_experts,
        topology.expert_parallel_size(),
        topology.expert_parallel_rank(),
        false,
    )
    .map_err(|error| invalid(error.to_string()))?
    .collect::<Vec<_>>();
    let expected_collective = topology
        .subgroup(eredu_core::ParallelAxis::Expert)
        .map_err(|error| invalid(error.to_string()))?;
    let selected_collective = realization
        .collective_group(eredu_core::CollectiveGroupId::new(0))
        .map_err(|error| invalid(error.to_string()))?;
    if realization.global_expert_count() != global_experts
        || realization.expert_parallel_size() != topology.expert_parallel_size()
        || realization.expert_parallel_rank() != topology.expert_parallel_rank()
        || realization.local_global_group_indices() != expected
        || selected_collective.members() != expected_collective.global_ranks()
        || selected_collective.local_rank() != expected_collective.rank()
    {
        return Err(invalid(
            "selected DeepSeek expert ownership differs from the Cartesian rank",
        ));
    }
    Ok(expected)
}

fn compact_banks(
    target_units: Range<usize>,
    routed_units: impl Fn(usize) -> bool,
    global_experts: &[usize],
    intermediate_range: Range<usize>,
) -> Vec<PartitionExpertBankOwnership> {
    target_units
        .filter(|unit| routed_units(*unit))
        .flat_map(|global_unit| {
            global_experts.iter().copied().enumerate().map({
                let intermediate_range = intermediate_range.clone();
                move |(owner_local_expert, global_expert)| PartitionExpertBankOwnership {
                    global_unit,
                    global_expert,
                    owner_local_expert,
                    intermediate_range: intermediate_range.clone(),
                }
            })
        })
        .collect()
}

fn expected_static_roles(
    target_units: &Range<usize>,
    target_count: usize,
    ownership: &PartitionOwnership,
    last_roles: &[&str],
) -> Result<Vec<String>, ParallelPlanError> {
    if ownership.owns_input() != (target_units.start == 0)
        || ownership.owns_output() != (target_units.end == target_count)
    {
        return Err(invalid(
            "DeepSeek input/output ownership differs from the target range",
        ));
    }
    let mut roles = Vec::new();
    if ownership.owns_input() {
        roles.push("embedding".to_owned());
    }
    if ownership.owns_output() {
        roles.extend(last_roles.iter().map(|role| (*role).to_owned()));
    }
    if ownership.static_roles() != roles {
        return Err(invalid(format!(
            "DeepSeek static roles {:?} differ from {roles:?}",
            ownership.static_roles()
        )));
    }
    Ok(roles)
}

/// Derives exact prediction-free V3 TP×PP×EP geometry from routed-expert authority.
pub fn v3_partition_local_geometry(
    args: &V3Args,
    layout: &LocalModelLayout,
    group_ranges: impl IntoIterator<Item = (impl AsRef<str>, Range<usize>)>,
    ownership: &PartitionOwnership,
    topology: eredu_core::ParallelRankTopology,
    realization: &crate::ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>,
) -> Result<V3PartitionLocalGeometry, ParallelPlanError> {
    require_prediction_free_v3(args)?;
    let target_count = usize::try_from(args.num_hidden_layers)
        .map_err(|_| invalid("V3 target count exceeds usize"))?;
    let target_units = target_range(group_ranges, target_count)?;
    let local = v3_local_geometry(args, layout)?;
    let banks = v3::expert_bank_specs(args).map_err(nn_invalid)?;
    let global_experts = banks
        .values()
        .next()
        .ok_or_else(|| invalid("V3 routed target has no declared expert banks"))?
        .group_count() as usize;
    let global_experts = validate_expert_topology(global_experts, topology, realization)?;
    if realization.unit_specs().len() != banks.len() {
        return Err(invalid("selected V3 expert unit schedule drifted"));
    }
    let local_expert_count = i32::try_from(global_experts.len())
        .map_err(|_| invalid("V3 local expert count exceeds i32"))?;
    for (unit, bank) in &banks {
        let expected = bank
            .clone()
            .with_group_geometry(local_expert_count, local.args().moe_intermediate_size)
            .map_err(|error| invalid(error.to_string()))?;
        if realization.unit_spec(crate::decoder::TARGET_EXECUTION_GROUP, *unit) != Some(&expected) {
            return Err(invalid(format!(
                "selected V3 expert bank geometry drifted at target unit {unit}"
            )));
        }
    }
    let intermediate_range = local
        .routed_expert_intermediate_range()
        .ok_or_else(|| invalid("V3 routed target has no TP expert range"))?;
    let static_roles =
        expected_static_roles(&target_units, target_count, ownership, &["norm", "output"])?;
    let expert_banks = compact_banks(
        target_units.clone(),
        |unit| banks.contains_key(&unit),
        &global_experts,
        intermediate_range,
    );
    Ok(V3PartitionLocalGeometry {
        target_units,
        local,
        static_roles,
        expert_realization: realization.clone(),
        expert_banks,
    })
}

/// Derives exact prediction-free V4 TP×PP×EP geometry from routed-expert authority.
pub fn v4_partition_local_geometry(
    args: &V4Args,
    layout: &LocalModelLayout,
    group_ranges: impl IntoIterator<Item = (impl AsRef<str>, Range<usize>)>,
    ownership: &PartitionOwnership,
    topology: eredu_core::ParallelRankTopology,
    realization: &crate::ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>,
) -> Result<V4PartitionLocalGeometry, ParallelPlanError> {
    require_prediction_free_v4(args)?;
    let target_count = usize::try_from(args.num_hidden_layers)
        .map_err(|_| invalid("V4 target count exceeds usize"))?;
    let target_units = target_range(group_ranges, target_count)?;
    let local = v4_local_geometry(args, layout)?;
    let banks = v4::expert_bank_specs(args).map_err(nn_invalid)?;
    let global_experts = banks
        .values()
        .next()
        .ok_or_else(|| invalid("V4 routed target has no declared expert banks"))?
        .group_count() as usize;
    let global_experts = validate_expert_topology(global_experts, topology, realization)?;
    if realization.unit_specs().len() != banks.len() {
        return Err(invalid("selected V4 expert unit schedule drifted"));
    }
    let local_expert_count = i32::try_from(global_experts.len())
        .map_err(|_| invalid("V4 local expert count exceeds i32"))?;
    for (&unit, bank) in &banks {
        let expected = bank
            .clone()
            .with_group_geometry(local_expert_count, local.args().moe_intermediate_size)
            .map_err(|error| invalid(error.to_string()))?;
        if realization.unit_spec(crate::decoder::TARGET_EXECUTION_GROUP, unit) != Some(&expected) {
            return Err(invalid(format!(
                "selected V4 expert bank geometry drifted at target unit {unit}"
            )));
        }
    }
    let intermediate_range = local
        .routed_expert_intermediate_range()
        .ok_or_else(|| invalid("V4 routed target has no exact TP expert range"))?;
    let static_roles = expected_static_roles(
        &target_units,
        target_count,
        ownership,
        &["norm", "output", "hyper_head"],
    )?;
    let expert_banks = compact_banks(
        target_units.clone(),
        |_| true,
        &global_experts,
        intermediate_range,
    );
    Ok(V4PartitionLocalGeometry {
        target_units,
        local,
        static_roles,
        expert_realization: realization.clone(),
        expert_banks,
    })
}

fn split_parameter_targets(
    partition: &ArchitecturePartition<impl Sized, impl Sized>,
    routed: impl Fn(&str) -> bool,
) -> (Vec<String>, Vec<String>) {
    let mut resident = Vec::new();
    let mut routed_targets = Vec::new();
    for target in partition
        .parameter_bindings()
        .iter()
        .flat_map(|binding| binding.members())
        .map(|member| member.target())
    {
        if routed(target) {
            routed_targets.push(target.to_owned());
        } else {
            resident.push(target.to_owned());
        }
    }
    (resident, routed_targets)
}

impl V3PartitionLocalFoundation {
    /// Validates V3 groups, compressed state, boundary roles, and compact-bank selection.
    pub fn from_partition(
        args: &V3Args,
        partition: &ArchitecturePartition<V3PartitionLocalGeometry, v3::TargetBoundarySchema>,
    ) -> Result<Self, ParallelPlanError> {
        require_prediction_free_v3(args)?;
        let geometry = partition.local_geometry();
        geometry.local.validate_for(args)?;
        let target_count = usize::try_from(args.num_hidden_layers)
            .map_err(|_| invalid("V3 target count exceeds usize"))?;
        if geometry.target_units.is_empty() || geometry.target_units.end > target_count {
            return Err(invalid("V3 partition-local target range drifted"));
        }
        let expected_groups = [(
            crate::decoder::TARGET_EXECUTION_GROUP,
            geometry.target_units(),
        )];
        validate_partition_common(
            partition,
            &expected_groups,
            &geometry.local_state_layout()?,
            v3::TargetBoundarySchema::from_args(args),
        )?;
        let (resident_parameter_targets, routed_parameter_targets) =
            split_parameter_targets(partition, |target| {
                target.contains(".mlp.experts.gate_up_proj")
                    || target.contains(".mlp.experts.down_proj")
            });
        for unit in geometry
            .target_units()
            .filter(|unit| args.layer_schedule.get(*unit) == Some(&LayerPolicy::SparseMoe))
        {
            let root = format!("model.layers.{unit}.mlp");
            if !routed_parameter_targets
                .iter()
                .any(|target| target.starts_with(&root))
                || !resident_parameter_targets
                    .iter()
                    .any(|target| target.starts_with(&format!("{root}.shared_experts")))
                || !resident_parameter_targets
                    .iter()
                    .any(|target| target.starts_with(&format!("{root}.gate")))
            {
                return Err(invalid(format!(
                    "V3 target unit {unit} does not own routed, shared, and router parameters"
                )));
            }
        }
        Ok(Self {
            geometry: geometry.clone(),
            resident_parameter_targets,
            routed_parameter_targets,
        })
    }
}

impl V4PartitionLocalFoundation {
    /// Validates V4 groups, pooled/compressed state, boundary roles, and compact banks.
    pub fn from_partition(
        args: &V4Args,
        partition: &ArchitecturePartition<V4PartitionLocalGeometry, v4::TargetBoundarySchema>,
    ) -> Result<Self, ParallelPlanError> {
        require_prediction_free_v4(args)?;
        let geometry = partition.local_geometry();
        geometry.local.validate_for(args)?;
        let target_count = usize::try_from(args.num_hidden_layers)
            .map_err(|_| invalid("V4 target count exceeds usize"))?;
        if geometry.target_units.is_empty() || geometry.target_units.end > target_count {
            return Err(invalid("V4 partition-local target range drifted"));
        }
        let expected_groups = [(
            crate::decoder::TARGET_EXECUTION_GROUP,
            geometry.target_units(),
        )];
        let boundary = v4::TargetBoundarySchema::from_args(args)
            .map_err(|error| invalid(error.to_string()))?;
        validate_partition_common(
            partition,
            &expected_groups,
            &geometry.local_state_layout()?,
            boundary,
        )?;
        let (resident_parameter_targets, routed_parameter_targets) =
            split_parameter_targets(partition, |target| {
                target.contains(".ffn.switch_mlp.gate_up_proj")
                    || target.contains(".ffn.switch_mlp.down_proj")
            });
        for unit in geometry.target_units() {
            let root = format!("layers.{unit}.ffn");
            if !routed_parameter_targets
                .iter()
                .any(|target| target.starts_with(&root))
                || !resident_parameter_targets
                    .iter()
                    .any(|target| target.starts_with(&format!("{root}.shared_experts")))
                || !resident_parameter_targets
                    .iter()
                    .any(|target| target.starts_with(&format!("{root}.gate")))
            {
                return Err(invalid(format!(
                    "V4 target unit {unit} does not own routed, shared, and router parameters"
                )));
            }
        }
        Ok(Self {
            geometry: geometry.clone(),
            resident_parameter_targets,
            routed_parameter_targets,
        })
    }
}

fn validate_partition_common<G, A>(
    partition: &ArchitecturePartition<G, A>,
    expected_groups: &[(&str, Range<usize>)],
    expected_state: &StateLayout,
    expected_boundary: A,
) -> Result<(), ParallelPlanError>
where
    A: PartialEq,
{
    let actual_groups = partition
        .groups()
        .iter()
        .map(|group| (group.group().as_str(), group.global_units()))
        .collect::<Vec<_>>();
    if actual_groups != expected_groups {
        return Err(invalid(
            "DeepSeek partition groups differ from family-local geometry",
        ));
    }
    let state = partition
        .state()
        .ok_or_else(|| invalid("DeepSeek target partition has no selected state"))?;
    if state.global_layer_offset() != expected_groups[0].1.start || state.layout() != expected_state
    {
        return Err(invalid(
            "DeepSeek partition state differs from its global target range",
        ));
    }
    if partition.boundary_schema() != &expected_boundary {
        return Err(invalid(
            "DeepSeek partition boundary differs from the role-exact target schema",
        ));
    }
    Ok(())
}

/// Declares pinned V3 embedding, normalization, and vocabulary placement.
pub fn v3_static_parameter_groups(
    args: &V3Args,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    Ok(v3_pinned_parameter_groups(args)?
        .into_iter()
        .map(|(_, group)| group)
        .collect())
}

fn v3_pinned_parameter_groups(
    args: &V3Args,
) -> Result<Vec<(&'static str, ParameterGroupSpec)>, ParallelPlanError> {
    args.parameters().statics.pinned_parameter_groups(
        crate::decoder::ParameterEncoding::Stored,
        "model.embed_tokens",
        "model.norm",
        "lm_head",
    )
}

/// Declares one V3 target or embedded-prediction unit's semantic groups.
pub fn v3_layer_parameter_groups(
    args: &V3Args,
    layer: usize,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    let target = usize::try_from(args.num_hidden_layers)
        .map_err(|_| invalid("V3 target layer count exceeds usize"))?;
    let total = usize::try_from(args.num_hidden_layers + args.num_nextn_predict_layers)
        .map_err(|_| invalid("V3 total layer count exceeds usize"))?;
    if layer >= total {
        return Err(invalid(format!(
            "V3 layer {layer} is outside {total} units"
        )));
    }
    let root = format!("model.layers.{layer}");
    let attention = format!("{root}.self_attn");
    let spec = super::v3_specs::attention_spec(args, &attention).map_err(nn_invalid)?;
    let query = match &spec.query {
        super::v3_specs::QuerySpec::Direct(projection) => projection,
        super::v3_specs::QuerySpec::LowRank(query) => &query.second,
    };
    let mut heads = Vec::new();
    for projection in [query, &spec.kv_b, &spec.output] {
        heads.extend(projection.logical_parameters().map_err(nn_invalid)?);
    }
    let mut groups = vec![eredu_runtime::logical_parameter_group(
        format!("{attention}.heads"),
        ParameterRole::AttentionHeads,
        Some(spec.geometry.heads() as usize),
        &heads,
        |p| {
            if p.parameter.id == spec.output.weight.id {
                MemberSharding::Partitioned { axis: 1 }
            } else {
                MemberSharding::Partitioned { axis: 0 }
            }
        },
    )?];
    let mut rank = spec.kv_a.logical_parameters().map_err(nn_invalid)?;
    rank.extend(spec.kv_norm.logical_parameters().map_err(nn_invalid)?);
    groups.push(replicated_group(format!("{root}.attention_rank"), &rank)?);
    if let super::v3_specs::QuerySpec::LowRank(query) = &spec.query {
        let mut rank = Vec::new();
        if let Some(first) = &query.first {
            rank.extend(first.logical_parameters().map_err(nn_invalid)?);
        }
        rank.extend(
            query
                .normalization
                .logical_parameters()
                .map_err(nn_invalid)?,
        );
        groups.push(replicated_group(format!("{root}.query_rank"), &rank)?);
    }
    let norms = ["input_layernorm", "post_attention_layernorm"]
        .into_iter()
        .map(|field| {
            super::v3_specs::normalization_spec(
                args,
                &format!("{root}.{field}.weight"),
                args.hidden_size,
            )
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(nn_invalid)?;
    groups.push(replicated_group(format!("{root}.norms"), norms.as_slice())?);
    if layer >= target {
        let prediction = super::v3_specs::prediction_spec(args, &root).map_err(nn_invalid)?;
        groups.push(replicated_group(format!("{root}.prediction"), &prediction)?);
    }
    let sparse = layer >= target || args.layer_schedule.get(layer) == Some(&LayerPolicy::SparseMoe);
    if sparse {
        let declaration = v3::routed_spec(args, layer, None).map_err(nn_invalid)?;
        groups.push(packed_expert_group(
            format!("{root}.mlp.routed_expert_intermediate"),
            &declaration.parameters.experts,
        )?);
        groups.push(replicated_group(
            format!("{root}.mlp.router"),
            &declaration.parameters.selector,
        )?);
        groups.push(gated_projection_group(
            format!("{root}.mlp.shared_expert_intermediate"),
            ParameterRole::ExpertIntermediate,
            &declaration.shared,
        )?);
    } else {
        let dense =
            super::v3_specs::dense_parameters(args, &format!("{root}.mlp")).map_err(nn_invalid)?;
        groups.push(gated_projection_group(
            format!("{root}.mlp.intermediate"),
            ParameterRole::FeedForwardIntermediate,
            &dense,
        )?);
    }
    Ok(groups)
}

fn nn_invalid(error: eredu_nn::Error) -> ParallelPlanError {
    invalid(error.to_string())
}

fn replicated_group(
    name: impl Into<String>,
    spec: &(impl DeclareLogicalParameters + ?Sized),
) -> Result<ParameterGroupSpec, ParallelPlanError> {
    eredu_runtime::logical_parameter_group(name, ParameterRole::Replicated, None, spec, |_| {
        MemberSharding::Replicated
    })
}
fn gated_projection_group(
    name: String,
    role: ParameterRole,
    parameters: &eredu_nn::SplitGatedProductSpec,
) -> Result<ParameterGroupSpec, ParallelPlanError> {
    eredu_runtime::logical_parameter_group(
        name,
        role,
        Some(parameters.down.input as usize),
        parameters,
        |p| {
            if p.parameter.id == parameters.down.weight.id {
                MemberSharding::Partitioned { axis: 1 }
            } else {
                MemberSharding::Partitioned { axis: 0 }
            }
        },
    )
}
fn packed_expert_group(
    name: String,
    bank: &eredu_nn::GroupedGatedProductSpec,
) -> Result<ParameterGroupSpec, ParallelPlanError> {
    let eredu_nn::GatedProductGroupLayout::Packed { gate_up, .. } = bank.layout() else {
        return Err(invalid("DeepSeek placement requires packed experts"));
    };
    let width = bank.intermediate_dimensions() as usize;
    eredu_runtime::logical_parameter_group(
        name,
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
            } else if p.role == eredu_nn::LogicalParameterRole::LinearWeight {
                MemberSharding::Partitioned { axis: 2 }
            } else {
                MemberSharding::Replicated
            }
        },
    )
}

/// Declares pinned V4 vocabulary, target hyper-head, and final norm groups.
pub fn v4_static_parameter_groups(
    args: &V4Args,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    Ok(v4_pinned_parameter_groups(args)?
        .into_iter()
        .map(|(_, group)| group)
        .collect())
}

fn v4_pinned_parameter_groups(
    args: &V4Args,
) -> Result<Vec<(&'static str, ParameterGroupSpec)>, ParallelPlanError> {
    let mut groups = args.parameters().statics.pinned_parameter_groups(
        crate::decoder::ParameterEncoding::Stored,
        "embed",
        "norm",
        "head",
    )?;
    groups.push((
        "hyper_head",
        replicated_group(
            "hyper_head",
            &super::v4_specs::hyper_head_spec(args, "hc_head").map_err(nn_invalid)?,
        )?,
    ));
    if args.dspark.is_some() {
        groups.push((
            "mtp",
            replicated_group(
                "dspark",
                &super::v4_specs::dspark_spec(args).map_err(nn_invalid)?,
            )?,
        ));
    }
    Ok(groups)
}

/// Describes every V3 target/prediction parameter with explicit graph ownership.
pub fn v3_parameter_description(
    args: &V3Args,
) -> Result<ArchitectureParameterDescription, eredu_nn::Error> {
    args.validate().map_err(eredu_nn::Error::backend_source)?;
    super::v3::state_layout(args)?;
    v3_parameters(args).map_err(eredu_nn::Error::backend_source)
}

pub(super) fn v3_parameters(
    args: &V3Args,
) -> Result<ArchitectureParameterDescription, ParallelPlanError> {
    deepseek_parameter_description(
        usize::try_from(args.num_hidden_layers)
            .map_err(|_| invalid("V3 target layer count exceeds usize"))?,
        usize::try_from(args.num_nextn_predict_layers)
            .map_err(|_| invalid("V3 prediction count exceeds usize"))?,
        v3_pinned_parameter_groups(args)?,
        |flat| v3_layer_parameter_groups(args, flat),
    )
}

/// Describes every V4 target/prediction parameter with explicit graph ownership.
pub fn v4_parameter_description(
    args: &V4Args,
) -> Result<ArchitectureParameterDescription, eredu_nn::Error> {
    args.validate().map_err(eredu_nn::Error::backend_source)?;
    v4_parameters(args).map_err(eredu_nn::Error::backend_source)
}

pub(super) fn v4_parameters(
    args: &V4Args,
) -> Result<ArchitectureParameterDescription, ParallelPlanError> {
    deepseek_parameter_description(
        usize::try_from(args.num_hidden_layers)
            .map_err(|_| invalid("V4 target layer count exceeds usize"))?,
        usize::try_from(args.num_nextn_predict_layers)
            .map_err(|_| invalid("V4 prediction count exceeds usize"))?,
        v4_pinned_parameter_groups(args)?,
        |flat| v4_layer_parameter_groups(args, flat),
    )
}

/// Projects the shared target/prediction invocation layout independently of parameter namespaces.
pub(super) fn execution_layout(
    targets: usize,
    predictions: usize,
) -> Result<(ExecutionGraph, ExecutionUnitLayout), ParallelPlanError> {
    let graph = ExecutionGraph::chain(
        std::iter::once(crate::decoder::TARGET_EXECUTION_GROUP.to_owned())
            .chain((0..predictions).map(|depth| format!("mtp.{depth}"))),
    )
    .map_err(|error| invalid(error.to_string()))?;
    let layout = ExecutionUnitLayout::new(
        &graph,
        std::iter::once(targets).chain(std::iter::repeat_n(1, predictions)),
    )
    .map_err(|error| invalid(error.to_string()))?;
    Ok((graph, layout))
}

fn deepseek_parameter_description(
    targets: usize,
    predictions: usize,
    static_groups: Vec<(&'static str, ParameterGroupSpec)>,
    mut unit_groups: impl FnMut(usize) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError>,
) -> Result<ArchitectureParameterDescription, ParallelPlanError> {
    let (graph, layout) = execution_layout(targets, predictions)?;
    let mut owned = static_groups
        .into_iter()
        .map(|(role, group)| {
            let owner = if role == "embedding" {
                ParameterGroupOwner::static_any_of(["embedding", "mtp"])
            } else {
                ParameterGroupOwner::static_role(role)
            };
            OwnedParameterGroupSpec::new(owner, group)
        })
        .collect::<Vec<_>>();
    for flat in 0..layout.len() {
        let address = layout.address(flat).expect("declared invocation");
        let groups = unit_groups(flat)?;
        let owner = ParameterGroupOwner::execution_unit(
            layout
                .group_id(address.group())
                .expect("declared group")
                .clone(),
            address.index(),
        );
        owned.extend(
            groups
                .into_iter()
                .map(|group| OwnedParameterGroupSpec::new(owner.clone(), group)),
        );
    }
    ArchitectureParameterDescription::from_owned(&graph, &layout, owned)
        .map_err(|error| invalid(error.to_string()))
}

/// Declares one V4 target, MTP, or DSpark block's low-rank, index-head,
/// hyper-stream, and expert placement.
pub fn v4_layer_parameter_groups(
    args: &V4Args,
    layer: usize,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    let target = usize::try_from(args.num_hidden_layers)
        .map_err(|_| invalid("V4 target layer count exceeds usize"))?;
    let total = usize::try_from(args.num_hidden_layers + args.num_nextn_predict_layers)
        .map_err(|_| invalid("V4 total layer count exceeds usize"))?;
    if layer >= total {
        return Err(invalid(format!(
            "V4 layer {layer} is outside {total} units"
        )));
    }
    let root = if layer < target {
        format!("layers.{layer}")
    } else {
        format!("mtp.{}", layer - target)
    };
    let attention = super::v4_specs::attention_spec(args, layer, &format!("{root}.attn"))
        .map_err(nn_invalid)?;
    let mut output = Vec::new();
    for projection in [
        &attention.query.second,
        &attention.output_a,
        &attention.output_b,
    ] {
        output.extend(projection.logical_parameters().map_err(nn_invalid)?);
    }
    let declarations = attention.logical_parameters().map_err(nn_invalid)?;
    let sink = declarations
        .iter()
        .find(|p| p.role == eredu_nn::LogicalParameterRole::AttentionSink)
        .expect("V4 attention declares its sink");
    output.push(sink.clone());
    let mut groups = vec![eredu_runtime::logical_parameter_group(
        format!("{root}.attn.output_groups"),
        ParameterRole::AttentionHeads,
        Some(dim(attention.groups)?),
        &output,
        |p| {
            if p.parameter.id == attention.output_b.weight.id {
                MemberSharding::Partitioned { axis: 1 }
            } else {
                MemberSharding::Partitioned { axis: 0 }
            }
        },
    )?];
    let mut shared = Vec::new();
    if let Some(first) = &attention.query.first {
        shared.extend(first.logical_parameters().map_err(nn_invalid)?);
    }
    shared.extend(
        attention
            .query
            .normalization
            .logical_parameters()
            .map_err(nn_invalid)?,
    );
    shared.extend(attention.kv.logical_parameters().map_err(nn_invalid)?);
    shared.extend(attention.kv_norm.logical_parameters().map_err(nn_invalid)?);
    let norms = super::v4_specs::block_normalizations(args, &root).map_err(nn_invalid)?;
    for norm in &norms {
        shared.extend(norm.logical_parameters().map_err(nn_invalid)?);
    }
    groups.push(replicated_group(
        format!("{root}.attention_shared"),
        &shared,
    )?);
    let hyper = super::v4_specs::hyper_block_spec(args, &root).map_err(nn_invalid)?;
    groups.push(replicated_group(format!("{root}.hyper_streams"), &hyper)?);
    if let Some(indexer) = &attention.indexer {
        let mut parameters = indexer.query.logical_parameters().map_err(nn_invalid)?;
        parameters.extend(indexer.weights.logical_parameters().map_err(nn_invalid)?);
        groups.push(replicated_group(
            format!("{root}.attn.index_heads"),
            &parameters,
        )?);
    }
    if let Some(compressor) = &attention.compressor {
        groups.push(replicated_group(
            format!("{root}.attn.compressor"),
            compressor,
        )?);
    }
    if let Some(indexer) = &attention.indexer {
        groups.push(replicated_group(
            format!("{root}.attn.indexer.compressor"),
            &indexer.compressor,
        )?);
    }
    if layer >= target && args.dspark.is_none() {
        let prediction = super::v4_specs::prediction_spec(args, &root).map_err(nn_invalid)?;
        groups.push(replicated_group(format!("{root}.prediction"), &prediction)?);
    }
    let moe =
        super::v4_specs::moe_spec(args, layer, &format!("{root}.ffn"), None).map_err(nn_invalid)?;
    groups.push(packed_expert_group(
        format!("{root}.ffn.routed_expert_intermediate"),
        &moe.routed.parameters.experts,
    )?);
    groups.push(gated_projection_group(
        format!("{root}.ffn.shared_expert_intermediate"),
        ParameterRole::ExpertIntermediate,
        &moe.routed.shared,
    )?);
    let mut router = moe
        .routed
        .parameters
        .selector
        .logical_parameters()
        .map_err(nn_invalid)?;
    if let Some(table) = &moe.token_table {
        router.extend(table.logical_parameters().map_err(nn_invalid)?);
    }
    groups.push(replicated_group(format!("{root}.ffn.router"), &router)?);
    Ok(groups)
}

fn dim(value: i32) -> Result<usize, ParallelPlanError> {
    usize::try_from(value)
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| invalid(format!("parallel dimension must be positive, got {value}")))
}

fn invalid(message: impl Into<String>) -> ParallelPlanError {
    ParallelPlanError::InvalidGroup(message.into())
}
