//! Semantic tensor-parallel parameter plans and rank-local Moshi geometry.

use std::{collections::BTreeMap, ops::Range};

use crate::decoder;
use eredu_checkpoint::LinearFormat;
use eredu_core::{checkpoint::TensorDtype, CollectiveGroupId, ParallelRankTopology};
use eredu_nn::{DeclareLogicalParameters, NeuralBackend};
use eredu_runtime::{
    aligned_partition_units, ArchitectureParameterDescription, ArchitecturePartition,
    CommunicationCompletionPolicy, CommunicationGroupRequirements, CommunicationManifest,
    CommunicationOperation, CommunicationOperationRequirement, CommunicationTensorLimits,
    ExecutionGraph, ExecutionUnitLayout, LayerRuntimeState, LayeredPartitionDriver,
    LocalModelLayout, MemberSharding, NoAuxiliaryBoundarySchema, OwnedParameterGroupSpec,
    ParallelPlanError, ParameterGroupOwner, ParameterGroupSpec, ParameterRole, PartitionOwnership,
    PartitionedExecutionPlan, PipelineActivationDtype, PipelineWireContract, StateLayout,
    StateSegmentLifetime, StateSegmentSpec, TopologyCommunicationPlan,
};

use super::{
    MoshiConfig, MoshiTransformerConfig, Unit, DEPTH_STATE_SEGMENT, TEMPORAL_STATE_SEGMENT,
};

/// Exact collective cardinality for one tensor-parallel realtime traversal.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct MoshiCollectiveCount {
    /// Row-parallel reductions in canonical execution order.
    pub all_sum: usize,
    /// Uneven vocabulary gathers in canonical text-then-depth order.
    pub all_gather: usize,
}

/// Exact architecture declaration for one matrix and its physical encoding.
///
/// This preserves the semantic dense shape separately from the selected
/// executable representation so generic materializers never have to recover
/// format or companion policy from a parameter name.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct MoshiMatrixContract {
    target: String,
    logical_shape: Vec<usize>,
    physical_shape: Vec<usize>,
    format: LinearFormat,
    packed_axis: Option<usize>,
    scale: Option<String>,
    affine_bias: Option<String>,
}

impl MoshiMatrixContract {
    /// Canonical primary parameter target.
    pub fn target(&self) -> &str {
        &self.target
    }

    /// Architecture-semantic dense matrix shape.
    pub fn logical_shape(&self) -> &[usize] {
        &self.logical_shape
    }

    /// Physical executable weight shape before rank-local selection.
    pub fn physical_shape(&self) -> &[usize] {
        &self.physical_shape
    }

    /// Exact executable encoding selected by the architecture.
    pub const fn format(&self) -> LinearFormat {
        self.format
    }

    /// Architecture-declared packing axis, when encoded.
    pub const fn packed_axis(&self) -> Option<usize> {
        self.packed_axis
    }

    /// Exact scale companion target, when present.
    pub fn scale(&self) -> Option<&str> {
        self.scale.as_deref()
    }

    /// Exact affine-bias companion target, when present.
    pub fn affine_bias(&self) -> Option<&str> {
        self.affine_bias.as_deref()
    }
}

/// Complete Moshi parameter topology plus exact primary-matrix formats.
#[derive(Debug, Clone)]
pub struct MoshiParameterContract {
    config: MoshiConfig,
    definition: std::sync::Arc<eredu_runtime::ComponentDefinition>,
    matrices: BTreeMap<String, MoshiMatrixContract>,
}

impl MoshiParameterContract {
    pub(super) fn declarations(&self) -> &std::sync::Arc<super::declarations::Declarations> {
        self.config.declarations()
    }
    /// Normalized configuration paired with these exact parameter scopes.
    pub fn configuration(&self) -> &MoshiConfig {
        &self.config
    }
    pub(super) fn retained_definition(&self) -> std::sync::Arc<eredu_runtime::ComponentDefinition> {
        self.definition.clone()
    }
    pub(super) fn with_quantization(
        &self,
        quantization: Option<eredu_checkpoint::WeightQuantization>,
    ) -> Result<Self, ParallelPlanError> {
        if quantization == self.config.native_quantization() {
            return Ok(self.clone());
        }
        let config = self
            .config
            .with_native_quantization(quantization)
            .map_err(|e| ParallelPlanError::InvalidGroup(e.to_string()))?;
        parameter_contract(&config)
    }

    /// Complete neutral ownership and sharding description.
    pub fn description(&self) -> &ArchitectureParameterDescription {
        self.definition.parameters()
    }

    /// Validated invocation, parameter, state and transport contract.
    pub fn definition(&self) -> &eredu_runtime::ComponentDefinition {
        &self.definition
    }

    /// Transfers the retained source-free definition to component construction.
    pub fn into_definition(self) -> std::sync::Arc<eredu_runtime::ComponentDefinition> {
        self.definition
    }

    /// Exact primary matrix contracts keyed by canonical target.
    pub const fn matrices(&self) -> &BTreeMap<String, MoshiMatrixContract> {
        &self.matrices
    }

    /// Consumes this contract into its complete parameter description.
    pub fn into_description(self) -> ArchitectureParameterDescription {
        std::sync::Arc::unwrap_or_clone(self.definition).into_parameters()
    }
}

/// Architecture-selected rank-local Moshi tensor-parallel handoff.
///
/// The concrete backend receives exact placement, family geometry, and opaque
/// communication identities. It never has to infer Moshi sharding or inspect a
/// semantic parallel axis while constructing the local model.
#[derive(Debug, Clone)]
pub struct MoshiParallelSelection {
    definition: std::sync::Arc<eredu_runtime::ComponentDefinition>,
    topology: eredu_core::ParallelTopology,
    layout: LocalModelLayout,
    geometry: LocalGeometry,
    communication: CommunicationManifest,
    tensor_group: CollectiveGroupId,
    execution_plan: PartitionedExecutionPlan,
    execution_identity: String,
}

impl MoshiParallelSelection {
    /// Validated contract with exact selected local state geometry.
    pub fn definition(&self) -> &eredu_runtime::ComponentDefinition {
        &self.definition
    }

    pub(super) fn retained_definition(&self) -> std::sync::Arc<eredu_runtime::ComponentDefinition> {
        self.definition.clone()
    }

    /// Selected pure tensor-parallel topology.
    pub const fn topology(&self) -> eredu_core::ParallelTopology {
        self.topology
    }

    /// Exact architecture-authored physical placement for this rank.
    pub const fn layout(&self) -> &LocalModelLayout {
        &self.layout
    }

    /// Exact architecture-owned rank-local construction and state geometry.
    pub const fn geometry(&self) -> &LocalGeometry {
        &self.geometry
    }

    /// Opaque communication groups and operation contracts for this rank.
    pub const fn communication(&self) -> &CommunicationManifest {
        &self.communication
    }

    /// Opaque group used by rank-local tensor-parallel operators.
    pub const fn tensor_group(&self) -> CollectiveGroupId {
        self.tensor_group
    }

    /// Fully local neutral traversal plan for this pure-TP rank.
    pub const fn execution_plan(&self) -> &PartitionedExecutionPlan {
        &self.execution_plan
    }

    /// Stable architecture identity including the selected rank-local shard.
    pub fn execution_identity(&self) -> &str {
        &self.execution_identity
    }

    /// Decomposes the selected handoff for backend materialization.
    pub fn into_parts(
        self,
    ) -> (
        LocalModelLayout,
        LocalGeometry,
        CommunicationManifest,
        CollectiveGroupId,
        PartitionedExecutionPlan,
    ) {
        (
            self.layout,
            self.geometry,
            self.communication,
            self.tensor_group,
            self.execution_plan,
        )
    }
}

/// Selects the Moshi tensor-parallel extension around its neutral
/// layered model.
///
/// This is deliberately limited to pure TP. It moves
/// placement, local geometry, and group projection into architecture code
/// without expanding the excluded realtime lifecycle.
pub fn select_parallel_execution<'a>(
    config: &MoshiConfig,
    contract: &MoshiParameterContract,
    rank: ParallelRankTopology,
    maximum_batch_size: i32,
    maximum_sequence_length: i32,
    activation_dtype: PipelineActivationDtype,
    completion: CommunicationCompletionPolicy,
    aliases: impl IntoIterator<Item = (&'a str, &'a str)>,
) -> Result<MoshiParallelSelection, ParallelPlanError> {
    let definition = contract.definition.clone();
    let parameters = definition.parameters();
    let topology = rank.topology();
    if topology.data() != 1
        || topology.tensor() <= 1
        || topology.pipeline() != 1
        || topology.expert() != 1
    {
        return Err(ParallelPlanError::InvalidGroup(format!(
            "Moshi realtime admits pure tensor parallelism only, got TP/PP/EP/DP={}/{}/{}/{}",
            topology.tensor(),
            topology.pipeline(),
            topology.expert(),
            topology.data()
        )));
    }
    let batch = usize::try_from(maximum_batch_size)
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| {
            ParallelPlanError::InvalidGroup(
                "Moshi maximum tensor-parallel batch size must be positive".into(),
            )
        })?;
    let sequence = usize::try_from(maximum_sequence_length)
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| {
            ParallelPlanError::InvalidGroup(
                "Moshi maximum tensor-parallel sequence length must be positive".into(),
            )
        })?;
    let layout = eredu_runtime::derive_partitioned_local_layout(parameters, rank)
        .map_err(ParallelPlanError::InvalidTensor)?;
    if layout.is_empty() {
        return Err(ParallelPlanError::InvalidTensor(
            "Moshi declared no tensor-parallel parameters".into(),
        ));
    }
    let geometry = local_geometry(config, contract, &layout, aliases)?;

    let tokens = batch.checked_mul(sequence).ok_or_else(|| {
        ParallelPlanError::InvalidGroup("Moshi invocation element bound overflowed".into())
    })?;
    let temporal_hidden = usize::try_from(config.temporal().hidden_size()).map_err(|_| {
        ParallelPlanError::InvalidGroup("Moshi temporal hidden size exceeds usize".into())
    })?;
    let depth_hidden = usize::try_from(config.depth_template().hidden_size()).map_err(|_| {
        ParallelPlanError::InvalidGroup("Moshi depth hidden size exceeds usize".into())
    })?;
    let hidden_elements = tokens
        .checked_mul(temporal_hidden.max(depth_hidden))
        .ok_or_else(|| {
            ParallelPlanError::InvalidGroup("Moshi collective hidden bound overflowed".into())
        })?;
    let global_vocabulary = usize::try_from(
        config
            .text_vocabulary_size()
            .max(config.audio_vocabulary_size()),
    )
    .map_err(|_| ParallelPlanError::InvalidGroup("Moshi vocabulary exceeds usize".into()))?;
    let local_vocabulary = std::iter::once("text_linear.weight".to_owned())
        .chain(
            (0..config.frame_schedule().depth_audio_codebooks())
                .map(|slice| format!("depformer.slices.{slice}.linear_out.weight")),
        )
        .map(|target| {
            geometry
                .vocabulary_range(&target)
                .map(Range::len)
                .ok_or_else(|| {
                    ParallelPlanError::InvalidTensor(format!(
                        "Moshi local output vocabulary is missing {target}"
                    ))
                })
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .max()
        .ok_or_else(|| {
            ParallelPlanError::InvalidTensor("Moshi local output vocabulary is empty".into())
        })?;
    let local_vocabulary_bound = global_vocabulary.div_ceil(topology.tensor());
    if local_vocabulary > local_vocabulary_bound {
        return Err(ParallelPlanError::InvalidTensor(
            "Moshi local vocabulary exceeds the common tensor-group bound".into(),
        ));
    }
    let local_logit_elements = tokens.checked_mul(local_vocabulary_bound).ok_or_else(|| {
        ParallelPlanError::InvalidGroup("Moshi local logit bound overflowed".into())
    })?;
    let complete_logit_elements = tokens.checked_mul(global_vocabulary).ok_or_else(|| {
        ParallelPlanError::InvalidGroup("Moshi complete logit bound overflowed".into())
    })?;
    let collective_dtypes = [TensorDtype::F32, TensorDtype::F16, TensorDtype::Bf16];
    let reduce = CommunicationOperationRequirement::tensors(
        CommunicationOperation::AllReduceSum,
        collective_dtypes.clone(),
        CommunicationTensorLimits::new(1, 3, hidden_elements, None)
            .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?,
        true,
    )
    .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
    let gather = CommunicationOperationRequirement::tensors(
        CommunicationOperation::AllGatherUneven,
        collective_dtypes,
        CommunicationTensorLimits::new(1, 3, local_logit_elements, None)
            .and_then(|limits| limits.with_output_tensor_elements(complete_logit_elements))
            .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?,
        true,
    )
    .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
    let requirements = CommunicationGroupRequirements::new([reduce, gather])
        .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
    let plan = TopologyCommunicationPlan::new()
        .with_completion_policy(completion)
        .with_tensor_groups(requirements);
    let tensor_group = plan
        .tensor_group_id(topology, rank)
        .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?
        .ok_or_else(|| {
            ParallelPlanError::InvalidGroup("Moshi tensor group projection is missing".into())
        })?;
    let communication = eredu_runtime::project_communication_manifest(topology, rank, &plan)
        .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
    let temporal_units = definition
        .unit_count(0)
        .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
    let depth_units = definition
        .unit_count(1)
        .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
    let complete_state = definition.state_layout();
    let state_plan = definition.state_partition_plan();
    let ownership = PartitionOwnership::new(true, true, ["embedding", "norm", "output"])
        .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
    let partition = ArchitecturePartition::from_description(
        parameters,
        [
            ("temporal_transformer", 0..temporal_units),
            ("depth_codebook_slices", 0..depth_units),
        ],
        ownership,
        complete_state,
        state_plan,
        geometry.clone(),
        NoAuxiliaryBoundarySchema::new(config.temporal().hidden_size()),
    )
    .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
    let temporal_driver = LayeredPartitionDriver::new(&partition, 0, 0..temporal_units)
        .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
    let depth_driver =
        LayeredPartitionDriver::new_with_state_ownership(&partition, 1, 0..depth_units, false)
            .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
    let transport = (0..parameters.graph().groups().len())
        .map(|group| {
            definition
                .group_transport(group)
                .map(|transport| (transport.kind, transport.request_optional))
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
    let execution_plan = PartitionedExecutionPlan::new(
        parameters.graph().clone(),
        transport,
        vec![Some(temporal_driver), Some(depth_driver)],
        Vec::new(),
        None,
        None,
        PipelineWireContract::new(activation_dtype),
    )
    .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
    let execution_identity = format!(
        "{};tp-rank={}/{}",
        config.architecture_fingerprint(),
        rank.tensor_parallel_rank(),
        rank.tensor_parallel_size()
    );
    let definition = definition
        .select_state_layout(geometry.state_layout())
        .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
    Ok(MoshiParallelSelection {
        definition,
        topology,
        layout,
        geometry,
        communication,
        tensor_group,
        execution_plan,
        execution_identity,
    })
}

/// Describes the complete Moshi parameter topology from normalized
/// architecture configuration without constructing backend modules.
///
/// Parallel admission uses this description before opening a checkpoint or
/// allocating any concrete-backend object. The ordinary layered model uses
/// the same declaration, preventing construction-time metadata from becoming
/// a second semantic planning path.
pub fn parameter_description(
    config: &MoshiConfig,
) -> Result<ArchitectureParameterDescription, ParallelPlanError> {
    Ok(parameter_contract(config)?.into_description())
}

/// Describes complete Moshi ownership together with exact executable formats.
pub fn parameter_contract(
    config: &MoshiConfig,
) -> Result<MoshiParameterContract, ParallelPlanError> {
    let declarations = config.declarations();
    let dimension = |label: &str, value: i32| {
        usize::try_from(value)
            .map_err(|_| ParallelPlanError::InvalidTensor(format!("Moshi {label} exceeds usize")))
    };
    let temporal_layers = dimension(
        "temporal layer count",
        config.temporal().num_hidden_layers(),
    )?;
    let depth_slices = config.frame_schedule().depth_audio_codebooks();
    let graph = ExecutionGraph::chain(["temporal_transformer", "depth_codebook_slices"])
        .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
    let layout = ExecutionUnitLayout::new(&graph, [temporal_layers, depth_slices])
        .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
    let temporal_group = layout
        .group_id(0)
        .expect("Moshi layout contains the temporal group")
        .clone();
    let depth_group = layout
        .group_id(1)
        .expect("Moshi layout contains the depth group")
        .clone();
    let mut owned = Vec::new();

    for table in &declarations.statics.embeddings {
        let spec = &table.embedding;
        owned.push(OwnedParameterGroupSpec::new(
            ParameterGroupOwner::static_role("embedding"),
            declared_group(
                spec.weight.id.as_str().trim_end_matches(".weight"),
                ParameterRole::Vocabulary,
                spec,
                MemberSharding::Balanced { axis: 0 },
            )?,
        ));
    }
    owned.push(OwnedParameterGroupSpec::new(
        ParameterGroupOwner::static_role("norm"),
        declared_group(
            "out_norm",
            ParameterRole::Replicated,
            &declarations.statics.output_norm,
            MemberSharding::Replicated,
        )?,
    ));
    owned.push(OwnedParameterGroupSpec::new(
        ParameterGroupOwner::static_role("output"),
        declared_group(
            "text_linear",
            ParameterRole::Vocabulary,
            &declarations.statics.text_output,
            MemberSharding::Balanced { axis: 0 },
        )?,
    ));
    for (layer, spec) in declarations.temporal.iter().enumerate() {
        for group in declared_block_groups(spec)? {
            owned.push(OwnedParameterGroupSpec::new(
                ParameterGroupOwner::execution_unit(temporal_group.clone(), layer),
                group,
            ));
        }
    }
    for (slice, spec) in declarations.depth.iter().enumerate() {
        let owner = ParameterGroupOwner::execution_unit(depth_group.clone(), slice);
        let edge = &spec.edge;
        let mut groups = vec![
            declared_group(
                edge.embedding
                    .weight
                    .id
                    .as_str()
                    .trim_end_matches(".weight"),
                ParameterRole::Vocabulary,
                &edge.embedding,
                MemberSharding::Balanced { axis: 0 },
            )?,
            declared_group(
                edge.input.weight.id.as_str().trim_end_matches(".weight"),
                ParameterRole::Replicated,
                &edge.input,
                MemberSharding::Replicated,
            )?,
            declared_group(
                edge.output.weight.id.as_str().trim_end_matches(".weight"),
                ParameterRole::Vocabulary,
                &edge.output,
                MemberSharding::Balanced { axis: 0 },
            )?,
        ];
        for block in &spec.blocks {
            groups.extend(declared_block_groups(block)?);
        }
        owned.extend(
            groups
                .into_iter()
                .map(|group| OwnedParameterGroupSpec::new(owner.clone(), group)),
        );
    }

    let description = ArchitectureParameterDescription::from_owned(&graph, &layout, owned)
        .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
    let physical = description
        .groups()
        .iter()
        .flat_map(|owned| owned.group().members())
        .map(|member| (member.target(), member))
        .collect::<BTreeMap<_, _>>();
    let mut matrices = BTreeMap::new();
    for declaration in declarations.logical_parameters().map_err(nn_invalid)? {
        let Some(format) = declaration.linear_format else {
            continue;
        };
        let target = declaration.parameter.id.as_str();
        let primary = physical.get(target).ok_or_else(|| {
            ParallelPlanError::InvalidTensor(format!(
                "Moshi matrix {target:?} has no physical parameter placement"
            ))
        })?;
        let contract = MoshiMatrixContract {
            target: target.to_owned(),
            packed_axis: (format.encoding() != LinearFormat::Dense)
                .then_some(declaration.shape.len() - 1),
            logical_shape: declaration.shape,
            physical_shape: primary.global_shape().to_vec(),
            format: format.encoding(),
            scale: format
                .scale()
                .map(|parameter| parameter.id.as_str().to_owned()),
            affine_bias: format
                .affine_bias()
                .map(|parameter| parameter.id.as_str().to_owned()),
        };
        if matrices.insert(target.to_owned(), contract).is_some() {
            return Err(ParallelPlanError::InvalidTensor(format!(
                "Moshi parameter contract repeats matrix {target:?}"
            )));
        }
    }
    let definition = std::sync::Arc::new(
        super::model::definition_with_parameters(config, description)
            .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?,
    );
    Ok(MoshiParameterContract {
        config: config.clone(),
        definition,
        matrices,
    })
}
fn nn_invalid(error: eredu_nn::Error) -> ParallelPlanError {
    ParallelPlanError::InvalidGroup(error.to_string())
}
fn declared_group(
    name: impl Into<String>,
    role: ParameterRole,
    spec: &impl eredu_nn::DeclareLogicalParameters,
    sharding: MemberSharding,
) -> Result<ParameterGroupSpec, ParallelPlanError> {
    eredu_runtime::logical_parameter_group(name, role, None, spec, |_| sharding.clone())
}
fn declared_block_groups(
    spec: &super::declarations::BlockConstruction,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    let root = &spec.root;
    let alignment =
        crate::linear_format::input_partition_alignment(spec.attention.output.format.encoding())
            as usize;
    let attention_units = aligned_partition_units(
        &format!("{root}.self_attn"),
        spec.attention.geometry.query_heads() as usize,
        spec.attention.geometry.key_dimensions() as usize,
        alignment,
    )?;
    let gating_units = aligned_partition_units(
        &format!("{root}.gating"),
        spec.gating.down.input as usize,
        1,
        alignment,
    )?;
    Ok(vec![
        eredu_runtime::logical_parameter_group(
            format!("{root}.self_attn.projections"),
            ParameterRole::AttentionHeads,
            Some(attention_units),
            &spec.attention,
            |parameter| {
                if parameter.parameter.id == spec.fused_attention().weight.id {
                    MemberSharding::PartitionedSegments {
                        axis: 0,
                        segments: spec.qkv_segments(),
                    }
                } else {
                    MemberSharding::Partitioned { axis: 1 }
                }
            },
        )?,
        declared_group(
            format!("{root}.norm1"),
            ParameterRole::Replicated,
            &spec.norms[0],
            MemberSharding::Replicated,
        )?,
        declared_group(
            format!("{root}.norm2"),
            ParameterRole::Replicated,
            &spec.norms[1],
            MemberSharding::Replicated,
        )?,
        eredu_runtime::logical_parameter_group(
            format!("{root}.gating.projections"),
            ParameterRole::FeedForwardIntermediate,
            Some(gating_units),
            &spec.gating,
            |parameter| {
                if parameter.parameter.id == spec.gating.gate_up.weight.id {
                    MemberSharding::PartitionedSegments {
                        axis: 0,
                        segments: spec.gated_segments(),
                    }
                } else {
                    MemberSharding::Partitioned { axis: 1 }
                }
            },
        )?,
    ])
}

/// Derives the backend-independent collective oracle for one traversal.
///
/// `executed_depth_slices` permits fully forced tails to state their shorter
/// traversal explicitly while rejecting impossible cardinalities.
pub fn collective_count(
    config: &MoshiConfig,
    executed_depth_slices: usize,
) -> Result<MoshiCollectiveCount, ParallelPlanError> {
    let depth_slices = config.frame_schedule().depth_audio_codebooks();
    if executed_depth_slices > depth_slices {
        return Err(ParallelPlanError::InvalidGroup(format!(
            "executed Moshi depth slices {executed_depth_slices} exceed {depth_slices}"
        )));
    }
    let temporal_layers = usize::try_from(config.temporal().num_hidden_layers()).map_err(|_| {
        ParallelPlanError::InvalidGroup("Moshi temporal layer count exceeds usize".into())
    })?;
    let depth_layers =
        usize::try_from(config.depth_template().num_hidden_layers()).map_err(|_| {
            ParallelPlanError::InvalidGroup("Moshi depth layer count exceeds usize".into())
        })?;
    collective_count_components(
        config.frame_schedule().total_audio_codebooks(),
        temporal_layers,
        executed_depth_slices,
        depth_layers,
    )
}

fn collective_count_components(
    audio_embedding_tables: usize,
    temporal_layers: usize,
    executed_depth_slices: usize,
    depth_layers: usize,
) -> Result<MoshiCollectiveCount, ParallelPlanError> {
    let overflow = |operation: &str| {
        ParallelPlanError::InvalidGroup(format!(
            "Moshi collective count overflowed while {operation}"
        ))
    };
    let all_gather = executed_depth_slices
        .checked_add(1)
        .ok_or_else(|| overflow("including the text vocabulary gather"))?;
    let embedding_sums = audio_embedding_tables
        .checked_add(1)
        .ok_or_else(|| overflow("including the text embedding"))?;
    let temporal_sums = temporal_layers
        .checked_mul(2)
        .ok_or_else(|| overflow("counting temporal row projections"))?;
    let depth_block_sums = depth_layers
        .checked_mul(2)
        .and_then(|count| count.checked_add(1))
        .ok_or_else(|| overflow("counting one depth slice"))?;
    let depth_sums = executed_depth_slices
        .checked_mul(depth_block_sums)
        .ok_or_else(|| overflow("counting executed depth slices"))?;
    let all_sum = embedding_sums
        .checked_add(temporal_sums)
        .and_then(|count| count.checked_add(depth_sums))
        .ok_or_else(|| overflow("combining all-sum phases"))?;
    Ok(MoshiCollectiveCount {
        all_sum,
        all_gather,
    })
}

/// Rank-local widths used to construct one shared decoder block.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct LocalTransformerGeometry {
    attention_heads: i32,
    gated_hidden_size: i32,
}

impl LocalTransformerGeometry {
    /// Rank-local complete attention heads.
    pub const fn attention_heads(&self) -> i32 {
        self.attention_heads
    }

    /// Rank-local width of each fused SwiGLU component.
    pub const fn gated_hidden_size(&self) -> i32 {
        self.gated_hidden_size
    }
}

/// Complete rank-local construction and state geometry for one Moshi rank.
#[derive(Debug, Clone)]
pub struct LocalGeometry {
    source_declarations: std::sync::Arc<super::declarations::Declarations>,
    temporal: Vec<LocalTransformerGeometry>,
    depth: Vec<Vec<LocalTransformerGeometry>>,
    vocabulary_ranges: BTreeMap<String, Range<usize>>,
    state_layout: StateLayout,
    architecture_fingerprint: String,
}

impl LocalGeometry {
    /// Local temporal block geometry in execution order.
    pub fn temporal(&self) -> &[LocalTransformerGeometry] {
        &self.temporal
    }

    /// Local depth transformer geometry by slice then layer.
    pub fn depth(&self) -> &[Vec<LocalTransformerGeometry>] {
        &self.depth
    }

    /// Exact vocabulary range owned by one embedding or output parameter.
    pub fn vocabulary_range(&self, target: &str) -> Option<&Range<usize>> {
        self.vocabulary_ranges.get(target)
    }

    /// Persistent-temporal plus reusable frame-local depth cache geometry.
    pub fn state_layout(&self) -> &StateLayout {
        &self.state_layout
    }

    /// Validates that every local width, vocabulary range, and state slot was
    /// derived from this exact normalized Moshi configuration.
    pub fn validate_for(&self, config: &MoshiConfig) -> Result<(), ParallelPlanError> {
        let temporal_layers =
            usize::try_from(config.temporal().num_hidden_layers()).map_err(|_| {
                ParallelPlanError::InvalidGroup("Moshi temporal layer count exceeds usize".into())
            })?;
        let depth_slices = config.frame_schedule().depth_audio_codebooks();
        let depth_layers =
            usize::try_from(config.depth_template().num_hidden_layers()).map_err(|_| {
                ParallelPlanError::InvalidGroup("Moshi depth layer count exceeds usize".into())
            })?;
        if self.architecture_fingerprint != config.architecture_fingerprint()
            || self.temporal.len() != temporal_layers
            || self.depth.len() != depth_slices
            || self.depth.iter().any(|slice| slice.len() != depth_layers)
        {
            return Err(ParallelPlanError::InvalidGroup(
                "rank-local Moshi geometry belongs to a different normalized configuration".into(),
            ));
        }

        let expected_vocabularies = vocabulary_global_rows(&self.source_declarations)?;
        if self.vocabulary_ranges.len() != expected_vocabularies.len() {
            return Err(ParallelPlanError::InvalidTensor(
                "rank-local Moshi vocabulary ownership is incomplete".into(),
            ));
        }
        for (target, global_rows) in expected_vocabularies {
            let range = self.vocabulary_ranges.get(&target).ok_or_else(|| {
                ParallelPlanError::InvalidTensor(format!(
                    "rank-local Moshi geometry is missing vocabulary range for {target}"
                ))
            })?;
            if range.start >= range.end || range.end > global_rows {
                return Err(ParallelPlanError::InvalidTensor(format!(
                    "rank-local Moshi vocabulary range {range:?} for {target} exceeds {global_rows} rows"
                )));
            }
        }

        let depth = self.depth.first().ok_or_else(|| {
            ParallelPlanError::InvalidGroup("rank-local Moshi geometry has no depth slices".into())
        })?;
        if self.depth.iter().any(|slice| slice != depth) {
            return Err(ParallelPlanError::InvalidGroup(
                "rank-local Moshi depth slices have inconsistent geometry".into(),
            ));
        }
        let expected_state =
            local_state_layout(config, &self.temporal, depth, &self.source_declarations)?;
        if expected_state != self.state_layout {
            return Err(ParallelPlanError::InvalidGroup(
                "rank-local Moshi state layout drifted from transformer geometry".into(),
            ));
        }
        Ok(())
    }

    /// Builds one rank-local temporal decoder configuration.
    pub fn temporal_config(
        &self,
        global: &MoshiTransformerConfig,
        layer: usize,
    ) -> Result<MoshiTransformerConfig, ParallelPlanError> {
        local_config(global, self.temporal.get(layer), "temporal", layer)
    }

    /// Builds one rank-local depth decoder configuration.
    pub fn depth_config(
        &self,
        global: &MoshiTransformerConfig,
        slice: usize,
        layer: usize,
    ) -> Result<MoshiTransformerConfig, ParallelPlanError> {
        local_config(
            global,
            self.depth.get(slice).and_then(|layers| layers.get(layer)),
            "depth",
            layer,
        )
    }

    /// Builds one rank-local execution unit using the canonical Moshi unit
    /// types and shared decoder block implementation.
    pub fn build_unit<B: NeuralBackend + eredu_nn::DistributedNeuralBackend>(
        &self,
        parameters: &MoshiParameterContract,
        group: usize,
        index: usize,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<Unit<B>, eredu_nn::Error> {
        let config = parameters.configuration();
        match group {
            0 => {
                let local = self
                    .temporal_config(config.temporal(), index)
                    .map_err(eredu_nn::Error::backend)?;
                super::block::build(
                    &local,
                    index,
                    parameters
                        .declarations()
                        .temporal
                        .get(index)
                        .ok_or_else(|| {
                            eredu_nn::Error::backend("missing temporal parameter owner")
                        })?,
                    context,
                )
                .map(Unit::Temporal)
            }
            1 => super::DepthSlice::new_parallel(parameters, index, self, context).map(Unit::Depth),
            _ => Err(eredu_nn::Error::backend(format!(
                "Moshi execution group {group} is outside 0..2"
            ))),
        }
    }
}

fn local_config(
    global: &MoshiTransformerConfig,
    geometry: Option<&LocalTransformerGeometry>,
    stack: &str,
    layer: usize,
) -> Result<MoshiTransformerConfig, ParallelPlanError> {
    let geometry = geometry.ok_or_else(|| {
        ParallelPlanError::InvalidGroup(format!(
            "missing rank-local Moshi {stack} geometry for layer {layer}"
        ))
    })?;
    global
        .with_parallel_geometry(geometry.attention_heads, geometry.gated_hidden_size)
        .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))
}

/// Executes one rank-local temporal block while preserving the canonical
/// Moshi mask and state-slot semantics.
#[allow(clippy::too_many_arguments)]
pub fn forward_temporal_block_parallel<B, S>(
    block: &mut crate::decoder::TransformerBlock<B>,
    index: usize,
    hidden: &B::Tensor,
    state: &mut S,
    forward: &mut super::ForwardContext<B::Tensor>,
    parallel: &B::ParallelContext,
    context: &<B::Tensor as eredu_nn::Tensor>::Context,
) -> Result<B::Tensor, eredu_nn::Error>
where
    B: NeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: eredu_nn::AttentionCache<B::Tensor>,
{
    super::block::forward_parallel(
        block,
        index,
        hidden,
        forward.temporal_mask(),
        forward.allow_sliding_prefill(),
        state,
        parallel,
        context,
    )
}

/// Derives and validates all rank-local Moshi widths from a resolved placement.
///
/// `contract` supplies canonical vocabulary groups and exact physical companions.
/// `aliases` maps logical aliases to canonical owners. Every alias present in
/// the model layout must have the same local shape and placement as its owner.
pub fn local_geometry<'a>(
    config: &MoshiConfig,
    contract: &MoshiParameterContract,
    layout: &eredu_runtime::LocalModelLayout,
    aliases: impl IntoIterator<Item = (&'a str, &'a str)>,
) -> Result<LocalGeometry, ParallelPlanError> {
    if config.architecture_fingerprint() != contract.configuration().architecture_fingerprint() {
        return Err(ParallelPlanError::InvalidGroup(
            "Moshi placement configuration differs from retained declarations".into(),
        ));
    }
    validate_alias_layouts(layout, aliases)?;

    let temporal_layers = usize::try_from(config.temporal().num_hidden_layers()).map_err(|_| {
        ParallelPlanError::InvalidGroup("Moshi temporal layer count exceeds usize".into())
    })?;
    let temporal = (0..temporal_layers)
        .map(|layer| local_block_geometry(config.temporal(), layer, layout, contract))
        .collect::<Result<Vec<_>, _>>()?;

    let depth = (0..config.frame_schedule().depth_audio_codebooks())
        .map(|slice| {
            let transformer = config
                .depth_transformer(slice)
                .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
            (0..transformer.num_hidden_layers() as usize)
                .map(|layer| local_block_geometry(&transformer, layer, layout, contract))
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;

    let mut vocabulary_ranges = BTreeMap::new();
    for owned in contract
        .description()
        .groups()
        .iter()
        .filter(|owned| owned.group().role() == ParameterRole::Vocabulary)
    {
        let group = owned.group();
        let matrix = group
            .members()
            .iter()
            .find_map(|member| contract.matrices().get(member.target()))
            .ok_or_else(|| {
                ParallelPlanError::InvalidGroup(format!(
                    "Moshi vocabulary group {:?} has no declared matrix",
                    group.logical_name()
                ))
            })?;
        let rows = matrix.logical_shape()[0];
        let range = eredu_runtime::balanced_parameter_group_range(group, layout, 0, rows)?;
        vocabulary_ranges.insert(matrix.target().to_owned(), range);
    }

    let depth_geometry = depth
        .first()
        .ok_or_else(|| ParallelPlanError::InvalidGroup("Moshi has no depth slices".into()))?;
    if depth.iter().any(|slice| {
        slice
            .iter()
            .map(LocalTransformerGeometry::attention_heads)
            .ne(depth_geometry
                .iter()
                .map(LocalTransformerGeometry::attention_heads))
    }) {
        return Err(ParallelPlanError::InvalidGroup(
            "Moshi depth slices resolved different rank-local cache geometry".into(),
        ));
    }
    let state_layout =
        local_state_layout(config, &temporal, depth_geometry, contract.declarations())?;
    let geometry = LocalGeometry {
        temporal,
        depth,
        source_declarations: contract.declarations().clone(),
        vocabulary_ranges,
        state_layout,
        architecture_fingerprint: config.architecture_fingerprint().to_owned(),
    };
    geometry.validate_for(config)?;
    Ok(geometry)
}

fn vocabulary_global_rows(
    declarations: &super::declarations::Declarations,
) -> Result<BTreeMap<String, usize>, ParallelPlanError> {
    let construction = &declarations.statics;
    let mut rows = BTreeMap::new();
    let mut add = |weight: &eredu_nn::ParameterSpec, vocabulary| -> Result<(), ParallelPlanError> {
        let vocabulary = usize::try_from(vocabulary).map_err(|_| {
            ParallelPlanError::InvalidGroup("Moshi vocabulary exceeds usize".into())
        })?;
        rows.insert(weight.id.as_str().to_owned(), vocabulary);
        Ok(())
    };
    for spec in &construction.embeddings {
        add(&spec.embedding.weight, spec.embedding.vocabulary)?;
    }
    add(
        &construction.text_output.weight,
        construction.text_output.output,
    )?;
    for slice in &declarations.depth {
        let construction = &slice.edge;
        add(
            &construction.embedding.weight,
            construction.embedding.vocabulary,
        )?;
        add(&construction.output.weight, construction.output.output)?;
    }
    Ok(rows)
}

fn local_block_geometry(
    config: &MoshiTransformerConfig,
    layer: usize,
    layout: &eredu_runtime::LocalModelLayout,
    contract: &MoshiParameterContract,
) -> Result<LocalTransformerGeometry, ParallelPlanError> {
    if layer >= config.num_hidden_layers() as usize {
        return Err(ParallelPlanError::InvalidGroup(format!(
            "Moshi layer {layer} is outside {}",
            config.num_hidden_layers()
        )));
    }
    let root = format!(
        "{}.layers.{layer}",
        crate::decoder::Config::parameter_root(config)
    );
    let spec = contract
        .declarations()
        .temporal
        .iter()
        .chain(
            contract
                .declarations()
                .depth
                .iter()
                .flat_map(|slice| slice.blocks.iter()),
        )
        .find(|spec| spec.root == root)
        .ok_or_else(|| {
            ParallelPlanError::InvalidGroup(format!("Moshi parameter owner {root} is missing"))
        })?;
    let root = &spec.root;
    let logical_axis = |projection: &eredu_nn::LinearSpec,
                        tensor: &eredu_runtime::LocalTensorLayout,
                        axis: usize,
                        label|
     -> Result<usize, ParallelPlanError> {
        let target = projection.weight.id.as_str();
        let matrix = contract.matrices().get(target).ok_or_else(|| {
            ParallelPlanError::InvalidTensor(format!("Moshi matrix contract is missing {target}"))
        })?;
        if matrix.logical_shape() != [projection.output as usize, projection.input as usize] {
            return Err(ParallelPlanError::InvalidTensor(format!(
                "Moshi matrix {target} declaration differs from retained geometry"
            )));
        }
        let physical = matrix
            .physical_shape()
            .get(axis)
            .copied()
            .filter(|&n| n > 0)
            .ok_or_else(|| {
                ParallelPlanError::InvalidTensor(format!(
                    "Moshi matrix {target} has no physical axis {axis}"
                ))
            })?;
        let logical = matrix.logical_shape()[axis];
        let scaled = local_axis(tensor, axis, label)?
            .checked_mul(logical)
            .ok_or_else(|| {
                ParallelPlanError::InvalidTensor(format!(
                    "Moshi matrix {target} logical width overflowed"
                ))
            })?;
        if !scaled.is_multiple_of(physical) {
            return Err(ParallelPlanError::InvalidTensor(format!(
                "Moshi matrix {target} local storage splits a logical element"
            )));
        }
        Ok(scaled / physical)
    };
    let attention = required_tensor(layout, spec.fused_attention().weight.id.as_str())?;
    let attention_width = logical_axis(spec.fused_attention(), attention, 0, "fused QKV width")?;
    if !attention_width.is_multiple_of(3) {
        return Err(ParallelPlanError::InvalidTensor(format!(
            "Moshi {root} local fused QKV width {attention_width} is not divisible by three"
        )));
    }
    let query_width = attention_width / 3;
    let head = usize::try_from(spec.attention.geometry.key_dimensions()).map_err(|_| {
        ParallelPlanError::InvalidGroup("Moshi head dimension exceeds usize".into())
    })?;
    if query_width == 0 || !query_width.is_multiple_of(head) {
        return Err(ParallelPlanError::InvalidTensor(format!(
            "Moshi {root} local query width {query_width} splits head dimension {head}"
        )));
    }
    let output = required_tensor(layout, spec.attention.output.weight.id.as_str())?;
    if logical_axis(
        &spec.attention.output,
        output,
        1,
        "attention output input width",
    )? != query_width
    {
        return Err(ParallelPlanError::InvalidTensor(format!(
            "Moshi {root} attention row projection does not consume its local head partition"
        )));
    }

    let gating = required_tensor(layout, spec.gating.gate_up.weight.id.as_str())?;
    let fused_gating = logical_axis(&spec.gating.gate_up, gating, 0, "fused gating width")?;
    if !fused_gating.is_multiple_of(2) {
        return Err(ParallelPlanError::InvalidTensor(format!(
            "Moshi {root} local fused gating width {fused_gating} is not even"
        )));
    }
    let gated_hidden_size = fused_gating / 2;
    let down = required_tensor(layout, spec.gating.down.weight.id.as_str())?;
    if logical_axis(&spec.gating.down, down, 1, "gating output input width")? != gated_hidden_size {
        return Err(ParallelPlanError::InvalidTensor(format!(
            "Moshi {root} gating row projection does not consume its local fused partition"
        )));
    }

    validate_companion_group(layout, attention)?;
    validate_companion_group(layout, gating)?;
    Ok(LocalTransformerGeometry {
        attention_heads: i32::try_from(query_width / head).map_err(|_| {
            ParallelPlanError::InvalidTensor("Moshi local head count exceeds i32".into())
        })?,
        gated_hidden_size: i32::try_from(gated_hidden_size).map_err(|_| {
            ParallelPlanError::InvalidTensor("Moshi local gated width exceeds i32".into())
        })?,
    })
}
fn required_tensor<'a>(
    layout: &'a eredu_runtime::LocalModelLayout,
    target: &str,
) -> Result<&'a eredu_runtime::LocalTensorLayout, ParallelPlanError> {
    layout.tensor(target).ok_or_else(|| {
        ParallelPlanError::InvalidTensor(format!("missing rank-local Moshi layout for {target}"))
    })
}

fn local_axis(
    tensor: &eredu_runtime::LocalTensorLayout,
    axis: usize,
    label: &str,
) -> Result<usize, ParallelPlanError> {
    tensor.local_shape().get(axis).copied().ok_or_else(|| {
        ParallelPlanError::InvalidTensor(format!(
            "Moshi {label} tensor has no axis {axis}: {:?}",
            tensor.local_shape()
        ))
    })
}

fn validate_companion_group(
    layout: &eredu_runtime::LocalModelLayout,
    representative: &eredu_runtime::LocalTensorLayout,
) -> Result<(), ParallelPlanError> {
    let expected = representative.logical_range();
    for (target, member) in layout
        .tensors()
        .filter(|(_, member)| member.logical_name() == representative.logical_name())
    {
        if member.logical_range() != expected {
            return Err(ParallelPlanError::InvalidTensor(format!(
                "Moshi quantization companion {target} does not share logical range {expected:?}"
            )));
        }
    }
    Ok(())
}

fn validate_alias_layouts<'a>(
    layout: &eredu_runtime::LocalModelLayout,
    aliases: impl IntoIterator<Item = (&'a str, &'a str)>,
) -> Result<(), ParallelPlanError> {
    for (alias, owner) in aliases {
        let alias_layout = required_tensor(layout, alias)?;
        let owner_layout = required_tensor(layout, owner)?;
        if alias_layout.global_shape() != owner_layout.global_shape()
            || alias_layout.local_shape() != owner_layout.local_shape()
            || alias_layout.placement() != owner_layout.placement()
        {
            return Err(ParallelPlanError::InvalidTensor(format!(
                "Moshi alias {alias} does not share owner {owner} placement"
            )));
        }
    }
    Ok(())
}

fn local_state_layout(
    config: &MoshiConfig,
    temporal_geometry: &[LocalTransformerGeometry],
    depth_geometry: &[LocalTransformerGeometry],
    declarations: &super::declarations::Declarations,
) -> Result<StateLayout, ParallelPlanError> {
    let geometry = |source: &[super::declarations::BlockConstruction],
                    blocks: &[LocalTransformerGeometry]| {
        if source.len() != blocks.len() {
            return Err(ParallelPlanError::InvalidGroup(
                "Moshi local block count differs from declared owners".into(),
            ));
        }
        source
            .iter()
            .zip(blocks)
            .map(|(spec, block)| {
                spec.localized(block.attention_heads, block.gated_hidden_size)
                    .map(|local| local.attention.geometry)
                    .map_err(nn_invalid)
            })
            .collect::<Result<Vec<_>, _>>()
    };
    let temporal = decoder::cache_layout_with_attention_geometry(
        config.temporal(),
        geometry(&declarations.temporal, temporal_geometry)?,
    )
    .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
    let depth = decoder::cache_layout_with_attention_geometry(
        config.depth_template(),
        geometry(
            &declarations
                .depth
                .first()
                .ok_or_else(|| ParallelPlanError::InvalidGroup("Moshi has no depth owners".into()))?
                .blocks,
            depth_geometry,
        )?,
    )
    .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
    let temporal_count = temporal.len();
    let depth_count = depth.len();
    let policies = temporal
        .iter()
        .cloned()
        .chain(depth.iter().cloned())
        .collect::<Vec<_>>();
    let schedule = eredu_core::LayerSchedule::new(policies.len(), policies)
        .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
    StateLayout::segmented(
        schedule,
        [
            StateSegmentSpec::new(
                TEMPORAL_STATE_SEGMENT,
                0..temporal_count,
                StateSegmentLifetime::Persistent,
                0,
            )
            .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?,
            StateSegmentSpec::new(
                DEPTH_STATE_SEGMENT,
                temporal_count..temporal_count + depth_count,
                StateSegmentLifetime::FrameLocal,
                0,
            )
            .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?,
        ],
    )
    .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))
}
