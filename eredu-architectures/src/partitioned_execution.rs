//! Architecture-owned admission and typed handoff for partitioned execution.

mod deepseek_dense;
mod diffusion;
mod nemotron_labs_diffusion;
pub(crate) mod qwen4_exp;
mod qwen_hybrid;

use eredu_core::communication::{CommunicationOperation, PipelineActivationDtype};
use std::{
    collections::{BTreeMap, BTreeSet},
    marker::PhantomData,
    ops::Range,
};

use eredu_core::communication::{
    BoundaryDimensionContract, BoundaryRoleContract, CommunicationCapabilities,
    CommunicationCompletionPolicy, CommunicationGroupRequirements, CommunicationManifest,
    CommunicationOperationRequirement, CommunicationRouteDescriptor, CommunicationRouteId,
    CommunicationTensorLimits, RoleExactBoundaryContract,
};
use eredu_core::{
    checkpoint::TensorDtype, ArtifactInspection, ParallelAxis, ParallelRankTopology,
    ParallelTopology,
};
use eredu_runtime::{
    derive_partitioned_local_layout, derive_partitioned_transform_source_layout,
    validate_transform_parameter_space, ArchitectureGroupKind, ArchitectureGroupPlacement,
    ArchitecturePartition, BoundaryWireSchema, PartitionOwnership, PartitionState,
    PipelinePartitionArchitecture, ResolvedBoundaryWireSchema, TopologyCommunicationPlan,
};

pub use eredu_runtime::executable_binding::partition_executors::*;

use crate::{
    preparation::ArchitectureCapabilities,
    processor_plan::ArtifactArchitecturePlan,
    replicated_text::{
        CompositeTextRequirements, ReplicatedTextExecutionClass, SelectedCompositeTextRealization,
    },
    RoutedTextRequirements, SelectedRoutedTextRealization,
};

/// Declares the exact payload-free mechanism required to propagate a local
/// shared-session phase failure to every partition rank.
///
/// Backend composition may add this requirement to the architecture-selected
/// world/session group only when it can realize
/// [`eredu_core::communication::FailureAgreementBackend`]. Barrier-only backends
/// remain explicitly unsupported rather than treating arrival as a status.
pub const fn partitioned_session_failure_agreement_requirement() -> CommunicationOperationRequirement
{
    CommunicationOperationRequirement::failure_agreement(true)
}

/// Selects an opt-in session-group contract with explicit failure agreement.
///
/// Pipeline callers may include their exact output-publication requirement;
/// tensor-only callers pass `None`. Production partitioned sessions select
/// this contract instead of treating a barrier as failure agreement.
pub fn partitioned_session_failure_agreement_group(
    publication: Option<CommunicationOperationRequirement>,
) -> Result<CommunicationGroupRequirements, eredu_core::communication::CommunicationManifestError> {
    let mut operations = Vec::with_capacity(usize::from(publication.is_some()) + 1);
    if let Some(publication) = publication {
        if publication.operation() != CommunicationOperation::Broadcast {
            return Err(
                eredu_core::communication::CommunicationManifestError::InvalidOperationLimits,
            );
        }
        operations.push(publication);
    }
    operations.push(partitioned_session_failure_agreement_requirement());
    CommunicationGroupRequirements::new(operations)
}

pub use eredu_runtime::partitioned_execution::{
    DirectPartitionExecutor, DirectResidentPartitionPass,
};

/// Architecture-owned preconstruction admission facts for one rank.
///
/// This value intentionally does not claim to be a selected partition: exact
/// family-local construction geometry and physical parameter bindings are
/// supplied only by the later typed architecture binding.
#[derive(Debug, Clone, PartialEq)]
pub struct PartitionedAdmission<R> {
    execution: R,
    rank: eredu_runtime::PipelineRankProjection,
    activation_dtype: PipelineActivationDtype,
    boundary: ResolvedBoundaryWireSchema,
    boundary_routes: Vec<SelectedPartitionBoundaryRoute>,
    communication: CommunicationManifest,
    session_group: Option<eredu_core::CollectiveGroupId>,
    tensor_group: Option<eredu_core::CollectiveGroupId>,
    expert_group: Option<eredu_core::CollectiveGroupId>,
}

/// Admitted direct-text rank facts.
pub type DirectPartitionedAdmission =
    PartitionedAdmission<eredu_runtime::ReplicatedTextRequirements>;
/// Admitted routed-text rank facts.
pub type RoutedPartitionedAdmission = PartitionedAdmission<RoutedTextRequirements>;
/// Admitted composite rank facts.
pub type CompositePartitionedAdmission = PartitionedAdmission<CompositeTextRequirements>;

impl<R> PartitionedAdmission<R> {
    /// Previously derived execution-class requirements, preserved without reinterpretation.
    pub const fn execution(&self) -> &R {
        &self.execution
    }

    /// Exact topology and local rank coordinates.
    pub const fn topology(&self) -> ParallelRankTopology {
        self.rank.topology()
    }

    /// Rank-local group/unit ownership in canonical architecture order.
    pub fn groups(&self) -> &[eredu_runtime::PipelineGroupOwnership] {
        self.rank.groups()
    }

    /// Input, output, and static-role ownership for this rank.
    pub const fn ownership(&self) -> &PartitionOwnership {
        self.rank.ownership()
    }

    /// Authoritative final-output world rank in this data-parallel cohort.
    /// Selected from the output group's semantic merge ownership before binding.
    pub const fn publication_owner(&self) -> usize {
        self.rank.publication_owner()
    }

    /// Exact rank-local mutable-state slice, if this rank owns state.
    pub const fn state(&self) -> Option<&PartitionState> {
        self.rank.state()
    }

    /// Canonical logical parameter targets owned by this rank before TP/EP lowering.
    pub fn logical_parameter_targets(&self) -> &[String] {
        self.rank.parameter_targets()
    }

    /// Selected floating dtype for activation-valued boundary tensors.
    pub const fn activation_dtype(&self) -> PipelineActivationDtype {
        self.activation_dtype
    }

    /// Exact architecture boundary resolved at the admitted invocation limits.
    pub const fn boundary(&self) -> &ResolvedBoundaryWireSchema {
        &self.boundary
    }

    /// Selected semantic routes in the same canonical order as the manifest.
    pub fn boundary_routes(&self) -> &[SelectedPartitionBoundaryRoute] {
        &self.boundary_routes
    }

    /// Complete opaque communication manifest projected for this rank.
    pub const fn communication(&self) -> &CommunicationManifest {
        &self.communication
    }

    /// Exact opaque world/session group selected for publication and commit.
    pub const fn session_group(&self) -> Option<eredu_core::CollectiveGroupId> {
        self.session_group
    }

    /// Exact opaque tensor group selected for this rank.
    pub const fn tensor_group(&self) -> Option<eredu_core::CollectiveGroupId> {
        self.tensor_group
    }

    /// Exact opaque expert group selected for this rank.
    pub const fn expert_group(&self) -> Option<eredu_core::CollectiveGroupId> {
        self.expert_group
    }

    /// Exact cache-placement topology derived from the admitted Cartesian rank.
    pub fn prompt_cache_topology(&self) -> Result<eredu_core::cache::PromptCacheTopology, String> {
        let rank = self.rank.topology();
        let topology = rank.topology();
        eredu_core::cache::PromptCacheTopology::new(
            (topology.pipeline() > 1)
                .then_some((topology.pipeline(), rank.pipeline_parallel_rank())),
            (topology.tensor() > 1).then_some((topology.tensor(), rank.tensor_parallel_rank())),
            (topology.expert() > 1).then_some((topology.expert(), rank.expert_parallel_rank())),
            true,
        )
        .map_err(|error| error.to_string())
    }
}

/// Caller invocation bounds resolved before architecture or communication construction.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct PartitionedSelectionRequest {
    topology: ParallelTopology,
    global_rank: usize,
    maximum_batch_size: i32,
    maximum_sequence_length: i32,
    activation_dtype: PipelineActivationDtype,
    completion_policy: Option<CommunicationCompletionPolicy>,
}

impl PartitionedSelectionRequest {
    /// Validates positive invocation bounds and one exact world rank.
    pub fn new(
        topology: ParallelTopology,
        global_rank: usize,
        maximum_batch_size: i32,
        maximum_sequence_length: i32,
        activation_dtype: PipelineActivationDtype,
    ) -> Result<Self, String> {
        ParallelRankTopology::new(topology, global_rank).map_err(|error| error.to_string())?;
        if maximum_batch_size <= 0 || maximum_sequence_length <= 0 {
            return Err("partitioned invocation limits must be positive".into());
        }
        Ok(Self {
            topology,
            global_rank,
            maximum_batch_size,
            maximum_sequence_length,
            activation_dtype,
            completion_policy: None,
        })
    }

    /// Selects the bounded native-completion policy for this distributed session.
    pub const fn with_completion_policy(mut self, policy: CommunicationCompletionPolicy) -> Self {
        self.completion_policy = Some(policy);
        self
    }

    /// Requested Cartesian topology.
    pub const fn topology(self) -> ParallelTopology {
        self.topology
    }

    /// Requested world rank.
    pub const fn global_rank(self) -> usize {
        self.global_rank
    }

    /// Maximum admitted batch size.
    pub const fn maximum_batch_size(self) -> i32 {
        self.maximum_batch_size
    }

    /// Maximum admitted sequence length.
    pub const fn maximum_sequence_length(self) -> i32 {
        self.maximum_sequence_length
    }

    /// Selected pipeline activation dtype.
    pub const fn activation_dtype(self) -> PipelineActivationDtype {
        self.activation_dtype
    }

    /// Selected bounded native-completion policy, if the caller supplied one.
    pub const fn completion_policy(self) -> Option<CommunicationCompletionPolicy> {
        self.completion_policy
    }
}

/// Architecture-owned dispatch over the three prediction-free partitioned execution classes.
pub trait PartitionedAdmissionDispatcher: Sized {
    /// Common dispatch result.
    type Output;
    /// Dispatch failure.
    type Error;

    /// Receives admitted direct-text facts.
    fn direct(self, requirements: DirectPartitionedAdmission) -> Result<Self::Output, Self::Error>;
    /// Receives admitted routed-text facts.
    fn routed(self, requirements: RoutedPartitionedAdmission) -> Result<Self::Output, Self::Error>;
    /// Receives admitted composite facts.
    fn composite(
        self,
        requirements: CompositePartitionedAdmission,
    ) -> Result<Self::Output, Self::Error>;
}

/// Failure while deriving or dispatching one partitioned execution request.
#[derive(Debug, thiserror::Error)]
pub enum PartitionedAdmissionError<E> {
    /// Architecture, artifact, topology, or rank admission failed.
    #[error("partitioned execution is unsupported: {0}")]
    Unsupported(String),
    /// The class-specific recipient rejected the admitted requirements.
    #[error("partitioned requirements dispatch failed: {0}")]
    Dispatch(E),
}

/// Projects an already retained routed family contract without inspecting artifacts again.
pub(crate) fn routed_partitioned_admission_from_requirements(
    execution: RoutedTextRequirements,
    request: PartitionedSelectionRequest,
    boundary_schema: BoundaryWireSchema,
    state_layout: &eredu_runtime::StateLayout,
    prediction_capture_limits: Option<(usize, usize)>,
    additional_tensor_reduction: Option<&CommunicationOperationRequirement>,
) -> Result<RoutedPartitionedAdmission, String> {
    let rank = ParallelRankTopology::new(request.topology(), request.global_rank())
        .map_err(|error| error.to_string())?;
    let boundary = boundary_schema
        .resolve(
            request.maximum_batch_size(),
            request.maximum_sequence_length(),
        )
        .map_err(|error| error.to_string())?;
    let completion = request.completion_policy().ok_or_else(|| {
        "partitioned execution requires a communication completion policy".to_owned()
    })?;
    let (communication, routes, session, tensor, expert) = communication_manifest(
        execution.text(),
        None,
        rank,
        request.activation_dtype(),
        &boundary,
        &boundary_schema,
        Some(execution.routes_per_token()),
        prediction_capture_limits,
        completion,
        additional_tensor_reduction,
    )?;
    partitioned_admission(
        execution,
        rank,
        request.activation_dtype(),
        boundary,
        routes,
        communication,
        session,
        tensor,
        expert,
        Some(state_layout),
        |value| value.text(),
    )
}

/// Projects retained composite geometry without rebuilding a family configuration.
pub(crate) fn composite_partitioned_admission_from_requirements(
    execution: CompositeTextRequirements,
    request: PartitionedSelectionRequest,
    boundary_schema: BoundaryWireSchema,
    state_layout: &eredu_runtime::StateLayout,
    prediction_capture_limits: Option<(usize, usize)>,
    additional_tensor_reduction: Option<&CommunicationOperationRequirement>,
) -> Result<CompositePartitionedAdmission, String> {
    let rank = ParallelRankTopology::new(request.topology(), request.global_rank())
        .map_err(|error| error.to_string())?;
    let boundary = boundary_schema
        .resolve(
            request.maximum_batch_size(),
            request.maximum_sequence_length(),
        )
        .map_err(|error| error.to_string())?;
    let completion = request.completion_policy().ok_or_else(|| {
        "partitioned execution requires a communication completion policy".to_owned()
    })?;
    let (communication, routes, session, tensor, expert) = communication_manifest(
        execution.execution(),
        Some(&execution),
        rank,
        request.activation_dtype(),
        &boundary,
        &boundary_schema,
        execution
            .routed_execution()
            .map(RoutedTextRequirements::routes_per_token),
        prediction_capture_limits,
        completion,
        additional_tensor_reduction,
    )?;
    partitioned_admission(
        execution,
        rank,
        request.activation_dtype(),
        boundary,
        routes,
        communication,
        session,
        tensor,
        expert,
        Some(state_layout),
        |value| value.execution(),
    )
}

/// Derives prediction-free admission facts and dispatches one semantic execution class.
pub fn dispatch_partitioned_admission<D>(
    inspection: &ArtifactInspection<ArtifactArchitecturePlan>,
    request: PartitionedSelectionRequest,
    dispatcher: D,
) -> Result<D::Output, PartitionedAdmissionError<D::Error>>
where
    D: PartitionedAdmissionDispatcher,
{
    let completion_policy = request.completion_policy().ok_or_else(|| {
        PartitionedAdmissionError::Unsupported(
            "partitioned execution requires an explicit bounded communication completion policy"
                .into(),
        )
    })?;
    let topology = request.topology();
    let rank = ParallelRankTopology::new(topology, request.global_rank())
        .map_err(|error| PartitionedAdmissionError::Unsupported(error.to_string()))?;
    let capabilities =
        architecture_capabilities(inspection).map_err(PartitionedAdmissionError::Unsupported)?;
    let class = crate::replicated_text::replicated_text_execution_class(inspection)
        .map_err(|error| PartitionedAdmissionError::Unsupported(error.to_string()))?;
    let routed = matches!(&class, ReplicatedTextExecutionClass::Routed(_))
        || matches!(
            &class,
            ReplicatedTextExecutionClass::Composite(execution)
                if execution.routed_execution().is_some()
        );
    validate_topology(topology, capabilities, routed)
        .map_err(PartitionedAdmissionError::Unsupported)?;
    let boundary_schema = crate::replicated_text::partitioned_boundary_schema(inspection, rank)
        .map_err(|error| PartitionedAdmissionError::Unsupported(error.to_string()))?;
    let boundary = boundary_schema
        .resolve(
            request.maximum_batch_size(),
            request.maximum_sequence_length(),
        )
        .map_err(|error| PartitionedAdmissionError::Unsupported(error.to_string()))?;
    let prediction_capture_limits = inspection
        .architecture_plan()
        .prediction_extension()
        .map(|extension| {
            extension.target_capture_limits(
                request.maximum_batch_size(),
                request.maximum_sequence_length(),
            )
        })
        .transpose()
        .map_err(|error| PartitionedAdmissionError::Unsupported(error.to_string()))?;

    match class {
        ReplicatedTextExecutionClass::Replicated(execution) => {
            let state_layout = direct_partitioned_state_layout(inspection, rank)
                .map_err(PartitionedAdmissionError::Unsupported)?
                .unwrap_or_else(|| execution.state_layout().clone());
            let (communication, boundary_routes, session_group, tensor_group, expert_group) =
                communication_manifest(
                    &execution,
                    None,
                    rank,
                    request.activation_dtype(),
                    &boundary,
                    &boundary_schema,
                    None,
                    prediction_capture_limits,
                    completion_policy,
                    None,
                )
                .map_err(PartitionedAdmissionError::Unsupported)?;
            let requirements = partitioned_admission(
                execution,
                rank,
                request.activation_dtype(),
                boundary,
                boundary_routes,
                communication,
                session_group,
                tensor_group,
                expert_group,
                Some(&state_layout),
                |value| value,
            )
            .map_err(PartitionedAdmissionError::Unsupported)?;
            dispatcher
                .direct(requirements)
                .map_err(PartitionedAdmissionError::Dispatch)
        }
        ReplicatedTextExecutionClass::Routed(execution) => {
            let state_layout = direct_partitioned_state_layout(inspection, rank)
                .map_err(PartitionedAdmissionError::Unsupported)?
                .unwrap_or_else(|| execution.text().state_layout().clone());
            let (communication, boundary_routes, session_group, tensor_group, expert_group) =
                communication_manifest(
                    execution.text(),
                    None,
                    rank,
                    request.activation_dtype(),
                    &boundary,
                    &boundary_schema,
                    Some(execution.routes_per_token()),
                    prediction_capture_limits,
                    completion_policy,
                    None,
                )
                .map_err(PartitionedAdmissionError::Unsupported)?;
            let requirements = partitioned_admission(
                execution,
                rank,
                request.activation_dtype(),
                boundary,
                boundary_routes,
                communication,
                session_group,
                tensor_group,
                expert_group,
                Some(&state_layout),
                |value| value.text(),
            )
            .map_err(PartitionedAdmissionError::Unsupported)?;
            dispatcher
                .routed(requirements)
                .map_err(PartitionedAdmissionError::Dispatch)
        }
        ReplicatedTextExecutionClass::Composite(execution) => {
            let state_layout = composite_partitioned_state_layout(inspection, rank)
                .map_err(PartitionedAdmissionError::Unsupported)?;
            let execution =
                match crate::replicated_text::composite_config(inspection.architecture_plan())
                    .map_err(|error| PartitionedAdmissionError::Unsupported(error.to_string()))?
                {
                    Some(crate::replicated_text::CompositeConfig::Gemma4(args)) => {
                        let units = args
                            .text
                            .pipeline_layer_ranges(rank.pipeline_parallel_size())
                            .map_err(|error| {
                                PartitionedAdmissionError::Unsupported(error.to_string())
                            })?[rank.pipeline_parallel_rank()]
                        .clone();
                        let declared = crate::gemma4::pipeline::partition_state_layout(
                            &args.text,
                            execution.state_layout(),
                            units,
                        )
                        .map_err(|error| {
                            PartitionedAdmissionError::Unsupported(error.to_string())
                        })?;
                        execution.with_state_layout(declared).map_err(|error| {
                            PartitionedAdmissionError::Unsupported(error.to_string())
                        })?
                    }
                    _ => execution,
                };
            let (communication, boundary_routes, session_group, tensor_group, expert_group) =
                communication_manifest(
                    execution.execution(),
                    Some(&execution),
                    rank,
                    request.activation_dtype(),
                    &boundary,
                    &boundary_schema,
                    execution
                        .routed_execution()
                        .map(RoutedTextRequirements::routes_per_token),
                    prediction_capture_limits,
                    completion_policy,
                    None,
                )
                .map_err(PartitionedAdmissionError::Unsupported)?;
            let requirements = partitioned_admission(
                execution,
                rank,
                request.activation_dtype(),
                boundary,
                boundary_routes,
                communication,
                session_group,
                tensor_group,
                expert_group,
                Some(&state_layout),
                |value| value.execution(),
            )
            .map_err(PartitionedAdmissionError::Unsupported)?;
            dispatcher
                .composite(requirements)
                .map_err(PartitionedAdmissionError::Dispatch)
        }
    }
}

fn local_partition_head_count(global: i32, rank: ParallelRankTopology) -> Result<i32, String> {
    let global = usize::try_from(global)
        .map_err(|_| "partitioned state head count is not positive".to_owned())?;
    let local = eredu_core::balanced_contiguous_range(
        global,
        rank.tensor_parallel_size(),
        rank.tensor_parallel_rank(),
        false,
    )
    .map_err(|error| error.to_string())?
    .len();
    i32::try_from(local).map_err(|_| "partitioned state head count exceeds i32".to_owned())
}

/// Derives the complete TP-local composite state before PP ownership slices it.
///
/// This is deliberately configuration-only: admission fixes state geometry
/// before a backend module, state allocation, or architecture partition exists.
fn composite_partitioned_state_layout(
    inspection: &ArtifactInspection<ArtifactArchitecturePlan>,
    rank: ParallelRankTopology,
) -> Result<eredu_runtime::StateLayout, String> {
    let config = crate::replicated_text::composite_config(inspection.architecture_plan())
        .map_err(|error| error.to_string())?
        .ok_or_else(|| "composite artifact has no normalized family configuration".to_owned())?;
    match config {
        crate::replicated_text::CompositeConfig::NemotronLabsDiffusion(args) => {
            let parameters =
                crate::nemotron_labs_diffusion::parallel::composite_parameter_description(args)
                    .map_err(|error| error.to_string())?;
            let layout = derive_partitioned_local_layout(&parameters, rank)?;
            let geometry = crate::decoder::partition_local_geometry(
                args,
                args.text_declaration().as_ref(),
                parameters.invocations(),
                &layout,
                0..crate::decoder::Config::num_hidden_layers(args) as usize,
            )
            .map_err(|error| error.to_string())?;
            Ok(geometry.complete_state_layout().clone())
        }

        crate::replicated_text::CompositeConfig::Diffusion(args) => {
            crate::diffusion_gemma::parallel::tensor_state_layout(args, rank)
                .map_err(|error| error.to_string())
        }
        crate::replicated_text::CompositeConfig::Gemma4(args) => {
            let base = crate::gemma4::graph::tensor_state_layout(args, rank)
                .map_err(|error| error.to_string())?;
            let units = args
                .text
                .pipeline_layer_ranges(rank.pipeline_parallel_size())
                .map_err(|error| error.to_string())?[rank.pipeline_parallel_rank()]
            .clone();
            crate::gemma4::pipeline::partition_state_layout(&args.text, &base, units)
                .map_err(|error| error.to_string())
        }
        crate::replicated_text::CompositeConfig::Muse(args) => {
            let parameters = crate::muse::component_definition(args, None)
                .map_err(|error| error.to_string())?
                .parameters()
                .clone();
            let layout = eredu_runtime::derive_partitioned_local_layout(&parameters, rank)?;
            let local = crate::muse::local_decoder_config(args, 0, &layout)
                .map_err(|error| error.to_string())?;
            crate::muse::state_layout(&local).map_err(|error| error.to_string())
        }
        crate::replicated_text::CompositeConfig::QwenVl(args) => {
            let heads = local_partition_head_count(args.text.num_key_value_heads, rank)?;
            let layers = usize::try_from(args.text.num_hidden_layers)
                .map_err(|_| "Qwen3-VL layer count exceeds usize".to_owned())?;
            let geometry = (0..layers)
                .map(|layer| {
                    args.text_declaration()
                        .common()
                        .block_parameters(layer)
                        .attention
                        .geometry
                        .partition_key_value_heads(0..heads as usize)
                        .map_err(|error| error.to_string())
                })
                .collect::<Result<Vec<_>, _>>()?;
            let text = crate::decoder::cache_layout_with_attention_geometry(&args.text, geometry)
                .map_err(|error| error.to_string())?;
            let text = eredu_runtime::StateLayout::new(text).map_err(|error| error.to_string())?;
            crate::qwen::vl::state_layout_with_text_state(args.text_declaration(), &text)
                .map_err(|error| error.to_string())
        }
        crate::replicated_text::CompositeConfig::QwenHybrid(args) => {
            qwen_hybrid_partitioned_state_layout(&args.text, rank)
        }
        crate::replicated_text::CompositeConfig::Inkling(args) => {
            let mut local = args.text_config.clone();
            for windowed in [false, true] {
                let Some((layer, _)) = args
                    .text_config
                    .layer_schedule
                    .iter()
                    .enumerate()
                    .find(|(_, policy)| policy.attention.window().is_some() == windowed)
                else {
                    continue;
                };
                let spec = args.common(layer).map_err(|error| error.to_string())?;
                let heads = local_partition_head_count(
                    spec.attention.projection.geometry.key_value_heads(),
                    rank,
                )?;
                let geometry = spec
                    .attention
                    .projection
                    .geometry
                    .partition_key_value_heads(0..heads as usize)
                    .map_err(|error| error.to_string())?;
                if windowed {
                    local.swa_num_attention_heads = Some(geometry.query_heads());
                    local.swa_num_key_value_heads = Some(geometry.key_value_heads());
                } else {
                    local.num_attention_heads = geometry.query_heads();
                    local.num_key_value_heads = geometry.key_value_heads();
                }
            }
            let target = crate::inkling::parallel_state_layout(
                args,
                &vec![local; args.text_config.layer_schedule.len()],
            )
            .map_err(|error| error.to_string())?;
            crate::inkling::composite_state_layout(&target, None).map_err(|error| error.to_string())
        }
    }
}

fn qwen_hybrid_partitioned_state_layout(
    text: &crate::qwen::hybrid::HybridConfig,
    rank: eredu_core::ParallelRankTopology,
) -> Result<eredu_runtime::StateLayout, String> {
    let attention_heads = local_partition_head_count(text.num_key_value_heads, rank)?;
    let selected = text
        .attention_head_geometry()
        .and_then(|geometry| geometry.partition_key_value_heads(0..attention_heads as usize))
        .map_err(|error| error.to_string())?;
    let recurrent = text
        .recurrent_head_geometry()
        .map_err(|error| error.to_string())?;
    let key_heads = local_partition_head_count(recurrent.key_heads(), rank)?;
    let value_heads = recurrent
        .value_heads()
        .checked_mul(key_heads)
        .and_then(|value| value.checked_div(recurrent.key_heads()))
        .ok_or_else(|| {
            "Qwen hybrid recurrent state heads do not partition proportionally".to_owned()
        })?;
    if value_heads.checked_mul(recurrent.key_heads())
        != recurrent.value_heads().checked_mul(key_heads)
    {
        return Err(
            "Qwen hybrid recurrent value heads split the selected key-head partition".into(),
        );
    }
    let mut local = text.policy().clone();
    local.num_attention_heads = selected.query_heads();
    local.num_key_value_heads = selected.key_value_heads();
    local.linear_num_key_heads = key_heads;
    local.linear_num_value_heads = value_heads;
    let local = text.with_local_policy(local).map_err(|e| e.to_string())?;
    let predictions = usize::try_from(text.mtp_num_hidden_layers)
        .map_err(|_| "Qwen hybrid MTP depth is negative".to_owned())?;
    crate::qwen::hybrid::state_layout_with_block_configs(
        text,
        std::iter::repeat_n(&local, text.layer_schedule.len()),
        std::iter::repeat_n(text, predictions),
    )
    .map_err(|error| error.to_string())
}

fn direct_partitioned_state_layout(
    inspection: &ArtifactInspection<ArtifactArchitecturePlan>,
    rank: ParallelRankTopology,
) -> Result<Option<eredu_runtime::StateLayout>, String> {
    if let Some(config) = crate::routed_text::diffusion_args(inspection) {
        let parameters = crate::diffusion_gemma::parallel::parameter_description(config)
            .map_err(|error| error.to_string())?;
        let layout = derive_partitioned_local_layout(&parameters, rank)?;
        return crate::diffusion_gemma::parallel::local_geometry(
            config,
            &layout,
            0..config.text.num_hidden_layers(),
        )
        .map(|geometry| geometry.state_layout().cloned())
        .map_err(|error| error.to_string());
    }
    match (
        inspection
            .architecture_plan()
            .safetensors_architecture()
            .map(|plan| plan.model()),
        inspection
            .architecture_plan()
            .gguf_plan()
            .map(|plan| plan.model()),
    ) {
        (Some(crate::configuration::SafetensorsModelConfig::K2Horizon(args)), None)
        | (None, Some(crate::configuration::GgufModelConfig::K2Horizon(args)))
            if args.is_moe() =>
        {
            routed_decoder_partitioned_state_layout(
                args,
                rank,
                inspection
                    .architecture_plan()
                    .k2_declaration(args)?
                    .common(),
            )
            .map(Some)
        }
        (Some(crate::configuration::SafetensorsModelConfig::K2Horizon(args)), None)
        | (None, Some(crate::configuration::GgufModelConfig::K2Horizon(args)))
            if !args.is_moe() =>
        {
            dense_decoder_partitioned_state_layout(
                args,
                rank,
                inspection
                    .architecture_plan()
                    .k2_declaration(args)?
                    .common(),
            )
            .map(Some)
        }
        (Some(crate::configuration::SafetensorsModelConfig::Nanbeige(args)), None)
        | (None, Some(crate::configuration::GgufModelConfig::Nanbeige(args))) => {
            dense_decoder_partitioned_state_layout(
                args,
                rank,
                inspection
                    .architecture_plan()
                    .dense_declaration()
                    .ok_or("missing admitted dense scopes")?
                    .as_ref(),
            )
            .map(Some)
        }
        (Some(crate::configuration::SafetensorsModelConfig::Gemma2(args)), None)
        | (None, Some(crate::configuration::GgufModelConfig::Gemma2(args))) => {
            dense_decoder_partitioned_state_layout(
                args,
                rank,
                inspection
                    .architecture_plan()
                    .dense_declaration()
                    .ok_or("missing admitted dense scopes")?
                    .as_ref(),
            )
            .map(Some)
        }
        (Some(crate::configuration::SafetensorsModelConfig::NemotronLabsDiffusion(args)), None) => {
            dense_decoder_partitioned_state_layout(args, rank, args.text_declaration().as_ref())
                .map(Some)
        }
        (None, Some(crate::configuration::GgufModelConfig::NemotronLabsDiffusion(plan))) => {
            dense_decoder_partitioned_state_layout(
                plan.text_config(),
                rank,
                plan.text_config().text_declaration().as_ref(),
            )
            .map(Some)
        }
        (Some(crate::configuration::SafetensorsModelConfig::Llama(args)), None)
        | (None, Some(crate::configuration::GgufModelConfig::Llama(args))) => {
            dense_decoder_partitioned_state_layout(
                args,
                rank,
                inspection
                    .architecture_plan()
                    .dense_declaration()
                    .ok_or("missing admitted dense scopes")?
                    .as_ref(),
            )
            .map(Some)
        }
        (Some(crate::configuration::SafetensorsModelConfig::Qwen(args)), None)
        | (None, Some(crate::configuration::GgufModelConfig::Qwen(args)))
            if args.is_moe() =>
        {
            routed_decoder_partitioned_state_layout(
                args,
                rank,
                inspection
                    .architecture_plan()
                    .qwen_declaration(args)?
                    .common(),
            )
            .map(Some)
        }
        (Some(crate::configuration::SafetensorsModelConfig::Qwen(args)), None)
        | (None, Some(crate::configuration::GgufModelConfig::Qwen(args))) => {
            dense_decoder_partitioned_state_layout(
                args,
                rank,
                inspection
                    .architecture_plan()
                    .qwen_declaration(args)?
                    .common(),
            )
            .map(Some)
        }
        (Some(crate::configuration::SafetensorsModelConfig::GptOss(args)), None)
        | (None, Some(crate::configuration::GgufModelConfig::GptOss(args))) => {
            routed_decoder_partitioned_state_layout(
                args,
                rank,
                inspection
                    .architecture_plan()
                    .gpt_oss_declaration(args)?
                    .common(),
            )
            .map(Some)
        }
        (Some(crate::configuration::SafetensorsModelConfig::DeepSeekV3(args)), None)
        | (None, Some(crate::configuration::GgufModelConfig::DeepSeekV3(args))) => {
            let parameters = crate::deepseek::parallel::v3_parameter_description(args)
                .map_err(|error| error.to_string())?;
            let layout = derive_partitioned_local_layout(&parameters, rank)?;
            crate::deepseek::parallel::v3_local_geometry(args, &layout)
                .map(|geometry| Some(geometry.state_layout().clone()))
                .map_err(|error| error.to_string())
        }
        (Some(crate::configuration::SafetensorsModelConfig::DeepSeekV4(args)), None) => {
            let parameters = crate::deepseek::parallel::v4_parameter_description(args)
                .map_err(|error| error.to_string())?;
            let layout = derive_partitioned_local_layout(&parameters, rank)?;
            crate::deepseek::parallel::v4_local_geometry(args, &layout)
                .map(|geometry| Some(geometry.state_layout().clone()))
                .map_err(|error| error.to_string())
        }
        (Some(crate::configuration::SafetensorsModelConfig::QwenHybrid(args)), None)
        | (None, Some(crate::configuration::GgufModelConfig::QwenHybrid(args)))
            if args.vision.is_none() && args.text.mtp_num_hidden_layers == 0 =>
        {
            qwen_hybrid_partitioned_state_layout(&args.text, rank).map(Some)
        }
        (Some(crate::configuration::SafetensorsModelConfig::Lfm2(args)), None)
        | (None, Some(crate::configuration::GgufModelConfig::Lfm2(args))) => {
            let declaration = inspection.architecture_plan().lfm2_declaration(args)?;
            crate::lfm2::partitioned_state_layout(
                &declaration,
                rank.tensor_parallel_rank(),
                rank.tensor_parallel_size(),
            )
            .map(Some)
            .map_err(|error| error.to_string())
        }
        (Some(crate::configuration::SafetensorsModelConfig::Lfm3(args)), None)
        | (None, Some(crate::configuration::GgufModelConfig::Lfm3(args))) => {
            let declaration = inspection.architecture_plan().lfm3_declaration(args)?;
            crate::lfm3::partitioned_state_layout(
                &declaration,
                rank.tensor_parallel_rank(),
                rank.tensor_parallel_size(),
            )
            .map(Some)
            .map_err(|error| error.to_string())
        }
        (Some(crate::configuration::SafetensorsModelConfig::KimiLinear(args)), None)
        | (None, Some(crate::configuration::GgufModelConfig::KimiLinear(args))) => {
            let declaration = inspection.architecture_plan().kimi_declaration(args)?;
            crate::kimi_linear::partitioned_state_layout(
                &declaration,
                rank.tensor_parallel_rank(),
                rank.tensor_parallel_size(),
            )
            .map(Some)
            .map_err(|error| error.to_string())
        }
        (Some(crate::configuration::SafetensorsModelConfig::NemotronH(args)), None)
        | (None, Some(crate::configuration::GgufModelConfig::NemotronH(args)))
            if args.num_nextn_predict_layers == 0 =>
        {
            crate::nemotron_h::partitioned_state_layout(
                inspection
                    .architecture_plan()
                    .nemotron_declaration(args)?
                    .as_ref(),
                rank.tensor_parallel_rank(),
                rank.tensor_parallel_size(),
            )
            .map(Some)
            .map_err(|error| error.to_string())
        }
        _ => Ok(None),
    }
}

fn routed_decoder_partitioned_state_layout<C>(
    args: &C,
    rank: ParallelRankTopology,
    common: &crate::decoder::DecoderDeclaration,
) -> Result<eredu_runtime::StateLayout, String>
where
    C: crate::decoder::Config,
{
    let layers = usize::try_from(args.num_hidden_layers())
        .map_err(|_| "routed decoder layer count exceeds usize".to_owned())?;
    let geometry = (0..layers)
        .map(|layer| {
            let owner = args.block_parameter_owner(layer);
            let parameters = &common.block_parameters(owner).attention;
            let range = crate::decoder::attention_partition::AttentionPartition::from_parameters(
                &parameters,
                args.attention_value_format(owner),
            )
            .and_then(|partition| {
                partition.head_range(rank.tensor_parallel_size(), rank.tensor_parallel_rank())
            })
            .map_err(|error| error.to_string())?;
            parameters
                .geometry
                .partition_key_value_heads(range)
                .map_err(|error| error.to_string())
        })
        .collect::<Result<Vec<_>, String>>()?;
    let cache = crate::decoder::cache_layout_with_attention_geometry(args, geometry)
        .map_err(|error| error.to_string())?;
    eredu_runtime::StateLayout::new(cache).map_err(|error| error.to_string())
}

fn dense_decoder_partitioned_state_layout<C>(
    args: &C,
    rank: ParallelRankTopology,
    declaration: &crate::decoder::DecoderDeclaration,
) -> Result<eredu_runtime::StateLayout, String>
where
    C: crate::decoder::PartitionedConfig,
{
    let parameters = declaration
        .parameter_description(args, crate::decoder::ParameterEncoding::Stored)
        .map_err(|error| error.to_string())?;
    let layout = derive_partitioned_local_layout(&parameters, rank)?;
    let count = parameters.unit_layout().len();
    crate::decoder::partition_local_geometry(
        args,
        declaration,
        parameters.invocations(),
        &layout,
        0..count,
    )
    .map(|geometry| geometry.complete_state_layout().clone())
    .map_err(|error| error.to_string())
}

fn architecture_capabilities(
    inspection: &ArtifactInspection<ArtifactArchitecturePlan>,
) -> Result<ArchitectureCapabilities, String> {
    match (
        inspection.architecture_plan().safetensors_architecture(),
        inspection.architecture_plan().gguf_plan(),
    ) {
        (Some(plan), None) => crate::preparation::prepared_safetensors_capabilities(plan)
            .map_err(|error| error.to_string()),
        (None, Some(plan)) => Ok(crate::preparation::prepared_gguf_capabilities(plan)),
        _ => Err("artifact has no unique normalized architecture plan".into()),
    }
}

fn validate_topology(
    topology: ParallelTopology,
    capabilities: ArchitectureCapabilities,
    routed: bool,
) -> Result<(), String> {
    if topology.data() > 1 {
        return Err("data-parallel execution is not supported".into());
    }
    if topology.is_replicated() {
        return Err(
            "partitioned execution requires an active tensor, pipeline, or expert axis".into(),
        );
    }
    if topology.expert() > 1 && !routed {
        return Err(
            "expert-parallel execution awaits an exact architecture exchange contract".into(),
        );
    }
    if capabilities
        .embedded_draft_layers()
        .is_some_and(|layers| layers > 0)
    {
        return Err("embedded prediction is active".into());
    }
    let supported = capabilities.parallel_plan();
    for (axis, admitted) in [
        (ParallelAxis::Tensor, supported.tensor_parallel()),
        (ParallelAxis::Pipeline, supported.pipeline_parallel()),
        (ParallelAxis::Expert, supported.expert_parallel()),
    ] {
        if topology.is_axis_active(axis) && !admitted {
            return Err(format!(
                "architecture does not admit active {axis:?} parallelism"
            ));
        }
    }
    Ok(())
}

fn communication_manifest(
    execution: &eredu_runtime::ReplicatedTextRequirements,
    composite: Option<&crate::replicated_text::CompositeTextRequirements>,
    rank: ParallelRankTopology,
    activation_dtype: PipelineActivationDtype,
    boundary: &ResolvedBoundaryWireSchema,
    symbolic_boundary: &BoundaryWireSchema,
    routes_per_token: Option<usize>,
    prediction_capture_limits: Option<(usize, usize)>,
    completion_policy: CommunicationCompletionPolicy,
    additional_tensor_reduction: Option<&CommunicationOperationRequirement>,
) -> Result<
    (
        CommunicationManifest,
        Vec<SelectedPartitionBoundaryRoute>,
        Option<eredu_core::CollectiveGroupId>,
        Option<eredu_core::CollectiveGroupId>,
        Option<eredu_core::CollectiveGroupId>,
    ),
    String,
> {
    let topology = rank.topology();
    validate_pipeline_decoder_coverage(execution, topology)?;
    if let Some(geometry) = execution.block_refinement() {
        let maximum_sequence = boundary
            .primary()
            .shape()
            .get(1)
            .copied()
            .ok_or_else(|| "partitioned refinement has no sequence bound".to_owned())?;
        if maximum_sequence
            < i32::try_from(geometry.canvas_length.get())
                .map_err(|_| "canvas length exceeds partition wire geometry")?
        {
            return Err(
                "partitioned invocation limits must cover the complete admitted refinement canvas"
                    .into(),
            );
        }
    }
    let activation_dtype = activation_tensor_dtype(activation_dtype)?;
    // Wire encoding does not establish projection, embedding or readout
    // arithmetic. Retain the actual selected floating parameter/recipe widths,
    // together with F32 promotion, for native collectives over those values.
    let mut collective_dtypes = vec![activation_dtype.clone(), TensorDtype::F32];
    if let Some(dtype) = execution.floating_state_source() {
        if matches!(
            dtype,
            TensorDtype::F16 | TensorDtype::Bf16 | TensorDtype::F32
        ) {
            collective_dtypes.push(dtype.clone());
        }
    }
    for parameter in execution
        .parameters()
        .iter()
        .chain(execution.auxiliary_parameters())
    {
        let dtype = execution
            .derived_recipe_outputs()
            .get(parameter.name())
            .map(|m| m.dtype.clone())
            .or_else(|| {
                parameter
                    .source_encoding()
                    .and_then(|encoding| encoding.scalar_dtype())
                    .map(Into::into)
            });
        for dtype in
            crate::linear_format::encoded_floating_compute_precisions(parameter.native_executable())
                .iter()
                .cloned()
                .chain(dtype)
        {
            match dtype {
                eredu_checkpoint::recipe::RecipeDtype::F16 => {
                    collective_dtypes.push(TensorDtype::F16)
                }
                eredu_checkpoint::recipe::RecipeDtype::BF16 => {
                    collective_dtypes.push(TensorDtype::Bf16)
                }
                _ => {}
            }
        }
    }
    collective_dtypes = collective_dtypes
        .into_iter()
        .fold(Vec::new(), |mut dtypes, dtype| {
            if !dtypes.contains(&dtype) {
                dtypes.push(dtype);
            }
            dtypes
        });
    let primary_elements = tensor_elements(boundary.primary().shape())?;
    // Media TP reductions retain native encoder positions and width, which
    // may exceed both the projected decoder sequence and the decoder width.
    let mut reduction_elements = primary_elements;
    if let Some(composite) = composite {
        for (group, transport) in execution.group_transports().iter().enumerate() {
            if transport.parallel_subgroup
                != Some(eredu_runtime::ArchitectureParallelSubgroup::TensorSharded)
            {
                continue;
            }
            let units = execution
                .execution_units()
                .group_range(group)
                .ok_or("tensor-sharded group has no unit geometry")?
                .len();
            for unit in 0..units {
                if let Some((schema, sequence)) = composite.partition_boundaries().resolve(
                    group,
                    group,
                    unit,
                    units,
                    boundary.primary().shape()[1],
                )? {
                    let resolved = schema
                        .resolve(boundary.primary().shape()[0], sequence)
                        .map_err(|error| error.to_string())?;
                    reduction_elements =
                        reduction_elements.max(tensor_elements(resolved.primary().shape())?);
                }
            }
        }
    }

    let mut plan = TopologyCommunicationPlan::new().with_completion_policy(completion_policy);

    if topology.pipeline() > 1 || topology.expert() > 1 {
        let (_, complete_logit_elements) = output_tensor_elements(execution, topology, boundary)?;
        let publication_rank = prediction_capture_limits.map_or(3, |(rank, _)| rank.max(3));
        // Decoder captures can be wider than the vocabulary output. Both use
        // these invocation bounds; media-only reductions do not enlarge them.
        let publication_elements = prediction_capture_limits.map_or(
            complete_logit_elements.max(primary_elements),
            |(_, elements)| elements.max(complete_logit_elements).max(primary_elements),
        );
        let publication_dtypes = collective_dtypes.clone();
        let publication = CommunicationOperationRequirement::tensors(
            CommunicationOperation::Broadcast,
            publication_dtypes,
            CommunicationTensorLimits::new(1, publication_rank, publication_elements, None)
                .map_err(|error| error.to_string())?,
            true,
        )
        .map_err(|error| error.to_string())?;
        let mut requirements = vec![
            publication,
            partitioned_session_failure_agreement_requirement(),
        ];
        if topology.expert() > 1 && additional_tensor_reduction.is_some() {
            requirements.push(
                CommunicationOperationRequirement::tensors(
                    CommunicationOperation::AllReduceSum,
                    vec![TensorDtype::I32],
                    CommunicationTensorLimits::new(1, 1, 1, None)
                        .map_err(|error| error.to_string())?,
                    true,
                )
                .map_err(|error| error.to_string())?,
            );
        }
        plan = plan.with_session_group(
            CommunicationGroupRequirements::new(requirements).map_err(|error| error.to_string())?,
        );
    }

    if topology.tensor() > 1 || routes_per_token.is_some() {
        let (local_logit_elements, complete_logit_elements) =
            output_tensor_elements(execution, topology, boundary)?;
        let additional_limits = additional_tensor_reduction
            .map(|requirement| {
                if requirement.operation() != CommunicationOperation::AllReduceSum {
                    return Err("additional tensor operation must be a sum reduction".to_owned());
                }
                requirement
                    .limits()
                    .ok_or_else(|| "tensor reduction has no limits".to_owned())
            })
            .transpose()?;
        let mut reduction_dtypes = collective_dtypes.clone();
        if let Some(requirement) = additional_tensor_reduction {
            for dtype in requirement.dtypes() {
                if !reduction_dtypes.contains(dtype) {
                    reduction_dtypes.push(dtype.clone());
                }
            }
        }
        let reduce = CommunicationOperationRequirement::tensors(
            CommunicationOperation::AllReduceSum,
            reduction_dtypes,
            CommunicationTensorLimits::new(
                additional_limits.map_or(1, |limits| limits.max_tensors().max(1)),
                additional_limits.map_or(boundary.primary().shape().len(), |limits| {
                    limits
                        .max_tensor_rank()
                        .max(boundary.primary().shape().len())
                }),
                additional_limits.map_or(reduction_elements, |limits| {
                    limits.max_tensor_elements().max(reduction_elements)
                }),
                None,
            )
            .and_then(|limits| {
                limits.with_output_tensor_elements(
                    additional_limits.map_or(reduction_elements, |extra| {
                        extra.max_output_tensor_elements().max(reduction_elements)
                    }),
                )
            })
            .map_err(|error| error.to_string())?,
            true,
        )
        .map_err(|error| error.to_string())?;
        let gather = CommunicationOperationRequirement::tensors(
            CommunicationOperation::AllGatherUneven,
            collective_dtypes.clone(),
            CommunicationTensorLimits::new(1, 3, local_logit_elements, None)
                .and_then(|limits| limits.with_output_tensor_elements(complete_logit_elements))
                .map_err(|error| error.to_string())?,
            true,
        )
        .map_err(|error| error.to_string())?;
        // Hooks reached only by this pipeline stage must agree local failure
        // before the next tensor collective. Selecting this payload-free
        // mechanism allocates no observation tensors on the ordinary path.
        let mut requirements = vec![
            reduce,
            gather,
            partitioned_session_failure_agreement_requirement(),
        ];
        if topology.pipeline() == 1 {
            let publication_rank = prediction_capture_limits.map_or(3, |(rank, _)| rank.max(3));
            let publication_elements = prediction_capture_limits.map_or(
                complete_logit_elements.max(primary_elements),
                |(_, elements)| elements.max(complete_logit_elements).max(primary_elements),
            );
            let publication_dtypes = collective_dtypes.clone();
            let publication = CommunicationOperationRequirement::tensors(
                CommunicationOperation::Broadcast,
                publication_dtypes,
                CommunicationTensorLimits::new(1, publication_rank, publication_elements, None)
                    .map_err(|error| error.to_string())?,
                true,
            )
            .map_err(|error| error.to_string())?;
            requirements.push(publication);
        }
        plan = plan.with_tensor_groups(
            CommunicationGroupRequirements::new(requirements).map_err(|error| error.to_string())?,
        );
    }

    let expert_requirements = if topology.expert() > 1 {
        let routes = routes_per_token
            .ok_or_else(|| "expert-parallel topology has no routed cardinality".to_owned())?;
        let packed_rows =
            primary_elements
                .checked_div(*boundary.primary().shape().last().ok_or_else(|| {
                    "partitioned primary boundary has no hidden dimension".to_owned()
                })? as usize)
                .and_then(|rows| rows.checked_mul(routes))
                .ok_or_else(|| "expert route geometry overflowed".to_owned())?;
        let exchange_elements = packed_rows
            .checked_mul(*boundary.primary().shape().last().unwrap() as usize)
            .ok_or_else(|| "expert exchange element geometry overflowed".to_owned())?;
        let counts = CommunicationOperationRequirement::tensors(
            CommunicationOperation::AllGatherEven,
            [TensorDtype::I32],
            CommunicationTensorLimits::new(1, 1, topology.expert(), None)
                .and_then(|limits| {
                    limits.with_output_tensor_elements(topology.expert() * topology.expert())
                })
                .map_err(|error| error.to_string())?,
            true,
        )
        .map_err(|error| error.to_string())?;
        // Routed expert values preserve native compute precision independently
        // of pipeline wire encoding, just like tensor-parallel contributions.
        let mut exchange_dtypes = collective_dtypes.clone();
        exchange_dtypes.push(TensorDtype::I32);
        // An owner can receive every rank's routes and submit that complete
        // fan-in again when returning expert results to their source ranks.
        let owner_exchange_elements = exchange_elements
            .max(packed_rows)
            .checked_mul(topology.expert())
            .ok_or_else(|| "expert exchange fan-in geometry overflowed".to_owned())?;
        let exchange = CommunicationOperationRequirement::tensors(
            CommunicationOperation::VariableAllToAll,
            exchange_dtypes,
            CommunicationTensorLimits::new(1, 2, owner_exchange_elements, Some(packed_rows))
                .and_then(|limits| limits.with_output_tensor_elements(owner_exchange_elements))
                .map_err(|error| error.to_string())?,
            true,
        )
        .map_err(|error| error.to_string())?;
        let requirements = CommunicationGroupRequirements::new([
            counts,
            exchange,
            partitioned_session_failure_agreement_requirement(),
        ])
        .map_err(|error| error.to_string())?;
        plan = plan.with_expert_groups(requirements.clone());
        Some(requirements)
    } else {
        None
    };

    // Component callbacks execute on every tensor/expert replica in this stage.
    // An expert-axis group or a tensor-axis group alone cannot certify them all.
    if topology.pipeline() > 1 && topology.expert() > 1 {
        plan = plan.with_pipeline_stage_groups(
            CommunicationGroupRequirements::new([
                partitioned_session_failure_agreement_requirement(),
            ])
            .map_err(|error| error.to_string())?,
        );
    }
    let projected = eredu_runtime::project_communication_manifest(topology, rank, &plan)
        .map_err(|error| error.to_string())?;
    let (routes, selected_boundary_routes) = pipeline_routes(
        execution,
        composite,
        topology,
        activation_dtype,
        boundary,
        symbolic_boundary,
    )?;
    // Row providers retain a real data-group context even when TP is singleton:
    // their status group may include EP peers or inactive pipeline stages.
    let tensor_group = if topology.tensor() > 1
        || (topology.expert() > 1 && additional_tensor_reduction.is_some())
    {
        plan.tensor_group_id(topology, rank)
            .map_err(|error| error.to_string())?
    } else {
        None
    };
    let session_group = plan.session_group_id().or_else(|| {
        (topology.pipeline() == 1 && topology.tensor() > 1)
            .then_some(tensor_group)
            .flatten()
    });
    let expert_group = expert_requirements.as_ref().and_then(|requirements| {
        projected
            .groups()
            .iter()
            .find(|group| {
                group.requirements() == requirements
                    && group.local_index().is_some()
                    && group.members().len() == topology.expert()
            })
            .map(eredu_core::communication::CommunicationGroupDescriptor::id)
    });
    let manifest = CommunicationManifest::new(
        topology.world_size(),
        rank.global_rank(),
        projected.groups().to_vec(),
        routes,
    )
    .map_err(|error| error.to_string())?
    .with_completion_policy(completion_policy);
    Ok((
        manifest,
        selected_boundary_routes,
        session_group,
        tensor_group,
        expert_group,
    ))
}

fn output_tensor_elements(
    execution: &eredu_runtime::ReplicatedTextRequirements,
    topology: ParallelTopology,
    boundary: &ResolvedBoundaryWireSchema,
) -> Result<(usize, usize), String> {
    let embedding = execution
        .parameters()
        .iter()
        .find(|parameter| {
            parameter.role() == eredu_runtime::ReplicatedTextParameterRole::Embedding
                && parameter.logical_shape().len() == 2
        })
        .ok_or_else(|| {
            "partitioned text requirements have no token embedding geometry".to_owned()
        })?;
    let vocabulary = embedding.logical_shape()[0];
    let local_vocabulary = vocabulary
        .checked_add(topology.tensor() - 1)
        .and_then(|value| value.checked_div(topology.tensor()))
        .ok_or_else(|| "tensor-parallel vocabulary geometry overflowed".to_owned())?;
    let token_rows =
        boundary.primary().shape()[..2]
            .iter()
            .try_fold(1usize, |rows, dimension| {
                rows.checked_mul(
                    usize::try_from(*dimension).map_err(|_| {
                        "partitioned invocation geometry is not positive".to_owned()
                    })?,
                )
                .ok_or_else(|| "partitioned token row count overflowed".to_owned())
            })?;
    let local = token_rows
        .checked_mul(local_vocabulary)
        .ok_or_else(|| "tensor-parallel logit geometry overflowed".to_owned())?;
    let complete = token_rows
        .checked_mul(vocabulary)
        .ok_or_else(|| "complete tensor-parallel logit geometry overflowed".to_owned())?;
    Ok((local, complete))
}

fn validate_pipeline_decoder_coverage(
    execution: &eredu_runtime::ReplicatedTextRequirements,
    topology: ParallelTopology,
) -> Result<(), String> {
    if topology.pipeline() <= 1 {
        return Ok(());
    }
    for (group, transport) in execution.group_transports().iter().enumerate() {
        if transport.kind != ArchitectureGroupKind::Decoder
            || transport.placement != ArchitectureGroupPlacement::Pipeline
        {
            continue;
        }
        let units = execution
            .execution_units()
            .group_range(group)
            .ok_or_else(|| format!("decoder execution group {group} has no unit geometry"))?
            .len();
        if units < topology.pipeline() {
            return Err(format!(
                "decoder execution group {group} has {units} units for {} pipeline stages",
                topology.pipeline()
            ));
        }
    }
    Ok(())
}

fn pipeline_routes(
    execution: &eredu_runtime::ReplicatedTextRequirements,
    composite: Option<&crate::replicated_text::CompositeTextRequirements>,
    topology: ParallelTopology,
    activation_dtype: eredu_core::checkpoint::TensorDtype,
    boundary: &ResolvedBoundaryWireSchema,
    symbolic_boundary: &BoundaryWireSchema,
) -> Result<
    (
        Vec<CommunicationRouteDescriptor>,
        Vec<SelectedPartitionBoundaryRoute>,
    ),
    String,
> {
    if topology.pipeline() == 1 {
        return Ok((Vec::new(), Vec::new()));
    }
    let route_contract = |symbolic_boundary: &BoundaryWireSchema,
                          boundary: &ResolvedBoundaryWireSchema|
     -> Result<
        (CommunicationOperationRequirement, RoleExactBoundaryContract),
        String,
    > {
        let max_tensors = 1usize
            .checked_add(boundary.auxiliary().len())
            .ok_or_else(|| "pipeline boundary tensor count overflowed".to_owned())?;
        let max_rank = boundary
            .tensors()
            .map(|tensor| tensor.shape().len())
            .max()
            .unwrap_or(0);
        let max_elements = boundary
            .tensors()
            .map(|tensor| tensor_elements(tensor.shape()))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .max()
            .ok_or_else(|| "pipeline boundary is empty".to_owned())?;
        let mut dtypes = Vec::new();
        for tensor in boundary.tensors() {
            let dtype = match tensor.dtype() {
                eredu_core::communication::BoundaryTensorDtype::Activation => {
                    activation_dtype.clone()
                }
                eredu_core::communication::BoundaryTensorDtype::Uint32 => {
                    eredu_core::checkpoint::TensorDtype::U32
                }
                eredu_core::communication::BoundaryTensorDtype::Float32 => {
                    eredu_core::checkpoint::TensorDtype::F32
                }
                eredu_core::communication::BoundaryTensorDtype::Int32 => {
                    eredu_core::checkpoint::TensorDtype::I32
                }
                _ => return Err("pipeline boundary uses an unsupported scalar dtype".into()),
            };
            if !dtypes.contains(&dtype) {
                dtypes.push(dtype);
            }
        }
        let requirement = CommunicationOperationRequirement::tensors(
            CommunicationOperation::SendReceive,
            dtypes,
            CommunicationTensorLimits::new(max_tensors, max_rank, max_elements, None)
                .map_err(|error| error.to_string())?,
            true,
        )
        .map_err(|error| error.to_string())?;
        let boundary_contract = RoleExactBoundaryContract::new(
            boundary.identity(),
            symbolic_boundary
                .tensors()
                .zip(boundary.tensors())
                .map(|(symbolic, resolved)| {
                    if symbolic.role() != resolved.role() || symbolic.dtype() != resolved.dtype() {
                        return Err(
                            "resolved pipeline boundary drifted from its symbolic schema".into(),
                        );
                    }
                    let dtype = match resolved.dtype() {
                        eredu_core::communication::BoundaryTensorDtype::Activation => {
                            activation_dtype.clone()
                        }
                        eredu_core::communication::BoundaryTensorDtype::Uint32 => TensorDtype::U32,
                        eredu_core::communication::BoundaryTensorDtype::Int32 => TensorDtype::I32,
                        eredu_core::communication::BoundaryTensorDtype::Float32 => TensorDtype::F32,
                        _ => {
                            return Err("pipeline boundary uses an unsupported scalar dtype".into());
                        }
                    };
                    let shape = symbolic
                        .shape()
                        .iter()
                        .zip(resolved.shape())
                        .map(
                            |(dimension, maximum)| -> Result<BoundaryDimensionContract, String> {
                                let maximum = usize::try_from(*maximum).map_err(|_| {
                                    "pipeline boundary dimension is not representable".to_owned()
                                })?;
                                Ok(match dimension {
                                    eredu_runtime::BoundaryTensorDimension::Batch
                                    | eredu_runtime::BoundaryTensorDimension::Sequence => {
                                        BoundaryDimensionContract::Variable { maximum }
                                    }
                                    eredu_runtime::BoundaryTensorDimension::Fixed(value) => {
                                        let value = usize::try_from(*value).map_err(|_| {
                                            "fixed pipeline boundary dimension is not representable"
                                                .to_owned()
                                        })?;
                                        if value != maximum {
                                            return Err(
                                            "resolved fixed pipeline boundary dimension drifted"
                                                .to_owned(),
                                        );
                                        }
                                        BoundaryDimensionContract::Fixed(value)
                                    }
                                    _ => {
                                        return Err(
                                            "unsupported symbolic boundary dimension".to_owned()
                                        );
                                    }
                                })
                            },
                        )
                        .collect::<Result<Vec<_>, _>>()?;
                    BoundaryRoleContract::symbolic(resolved.role(), dtype, shape)
                        .map_err(|error| error.to_string())
                })
                .collect::<Result<Vec<_>, String>>()?,
        )
        .map_err(|error| error.to_string())?;
        Ok((requirement, boundary_contract))
    };

    let pp_size = topology.pipeline();
    let placement = eredu_runtime::PipelinePlacementPlan::new(
        execution.execution_graph(),
        execution.execution_units(),
        execution.group_transports(),
        pp_size,
    )
    .map_err(|error| error.to_string())?;
    let mut routes = Vec::new();
    let mut selected = Vec::new();
    for semantic in placement.routes() {
        let decoder_continuation = semantic.source_group == semantic.destination_group
            && execution.group_transports()[semantic.source_group].kind
                == ArchitectureGroupKind::Decoder;
        let group_continuation =
            semantic.source_group == semantic.destination_group && !decoder_continuation;
        let (route_symbolic, route_boundary) = if decoder_continuation {
            (symbolic_boundary.clone(), boundary.clone())
        } else {
            // Optional-root edges use their architecture-owned learned-context
            // schema when present. They never borrow decoder cache/position roles.
            let shape = boundary.primary().shape();
            let declared = composite
                .map(|requirements| {
                    requirements.partition_boundaries().resolve(
                        semantic.source_group,
                        semantic.destination_group,
                        semantic.source_pipeline,
                        pp_size,
                        shape[1],
                    )
                })
                .transpose()?
                .flatten();
            let (symbolic, maximum_sequence) = match declared {
                Some(declared) => declared,
                None if group_continuation => {
                    return Err(
                        "composite pipeline continuation has no architecture-authored boundary"
                            .into(),
                    );
                }
                None => (
                    BoundaryWireSchema::new(
                        "composite-group-activation-v1",
                        symbolic_boundary.primary().clone(),
                        [],
                    )
                    .map_err(|error| error.to_string())?,
                    shape[1],
                ),
            };
            let resolved = symbolic
                .resolve(shape[0], maximum_sequence)
                .map_err(|error| error.to_string())?;
            (symbolic, resolved)
        };
        let (requirement, boundary_contract) = route_contract(&route_symbolic, &route_boundary)?;
        for source in 0..topology.world_size() {
            let coordinates = topology
                .coordinates(source)
                .map_err(|error| error.to_string())?;
            if coordinates.pipeline() != semantic.source_pipeline {
                continue;
            }
            let destination = topology
                .rank_for(coordinates.with_pipeline(semantic.destination_pipeline))
                .map_err(|error| error.to_string())?;
            let order = routes.len();
            let route_id = CommunicationRouteId::new(order as u64);
            routes.push(
                CommunicationRouteDescriptor::new(
                    route_id,
                    order,
                    source,
                    destination,
                    requirement.clone(),
                )
                .and_then(|route| route.with_boundary_contract(boundary_contract.clone()))
                .map(|route| {
                    if topology.pipeline() > 2 && topology.tensor() > 1 {
                        route.with_world_broadcast()
                    } else {
                        route
                    }
                })
                .map_err(|error| error.to_string())?,
            );
            selected.push(SelectedPartitionBoundaryRoute::new(
                eredu_runtime::PartitionBoundaryRoute {
                    source_group: semantic.source_group,
                    destination_group: semantic.destination_group,
                    source_rank: source,
                    destination_rank: destination,
                    route: route_id,
                },
                route_boundary.clone(),
                semantic.source_unit_end,
            ));
        }
    }
    Ok((routes, selected))
}

fn tensor_elements(shape: &[i32]) -> Result<usize, String> {
    shape.iter().try_fold(1usize, |elements, dimension| {
        let dimension = usize::try_from(*dimension)
            .map_err(|_| "communication tensor dimension is not positive".to_owned())?;
        elements
            .checked_mul(dimension)
            .ok_or_else(|| "communication tensor element count overflowed".to_owned())
    })
}

fn activation_tensor_dtype(
    dtype: PipelineActivationDtype,
) -> Result<eredu_core::checkpoint::TensorDtype, String> {
    Ok(match dtype {
        PipelineActivationDtype::Float16 => eredu_core::checkpoint::TensorDtype::F16,
        PipelineActivationDtype::Bfloat16 => eredu_core::checkpoint::TensorDtype::Bf16,
        PipelineActivationDtype::Float32 => eredu_core::checkpoint::TensorDtype::F32,
        _ => return Err("unsupported pipeline activation dtype".into()),
    })
}

fn partitioned_admission<R, F>(
    execution: R,
    topology: ParallelRankTopology,
    activation_dtype: PipelineActivationDtype,
    boundary: ResolvedBoundaryWireSchema,
    boundary_routes: Vec<SelectedPartitionBoundaryRoute>,
    communication: CommunicationManifest,
    session_group: Option<eredu_core::CollectiveGroupId>,
    tensor_group: Option<eredu_core::CollectiveGroupId>,
    expert_group: Option<eredu_core::CollectiveGroupId>,
    state_layout: Option<&eredu_runtime::StateLayout>,
    text: F,
) -> Result<PartitionedAdmission<R>, String>
where
    F: FnOnce(&R) -> &eredu_runtime::ReplicatedTextRequirements,
{
    let text = text(&execution);
    let rank = eredu_runtime::PipelineRankProjection::new(text, state_layout, topology, true)
        .map_err(|error| error.to_string())?;
    Ok(PartitionedAdmission {
        execution,
        rank,
        activation_dtype,
        boundary,
        boundary_routes,
        communication,
        session_group,
        tensor_group,
        expert_group,
    })
}

/// Capability-checked admission retaining one already selected base realization.
///
/// This admission excludes family-local construction geometry and physical
/// parameter bindings, which belong to the selected partitioned realization.
#[derive(Debug, Clone)]
pub struct SelectedPartitionedAdmission<R, Q> {
    requirements: PartitionedAdmission<Q>,
    base: R,
    materialization_tasks: Vec<eredu_runtime::ReplicatedTextMaterializationTask>,
}

impl<R, Q> SelectedPartitionedAdmission<R, Q> {
    /// Partitioned facts admitted before architecture construction.
    pub const fn requirements(&self) -> &PartitionedAdmission<Q> {
        &self.requirements
    }

    /// Previously selected realization retained without re-selection.
    pub const fn base(&self) -> &R {
        &self.base
    }

    /// Exact physical tasks selected before backend resources exist.
    ///
    /// Family topology later attaches atomic output companions and projects this
    /// immutable selection onto the already admitted rank ownership.
    pub fn materialization_tasks(&self) -> &[eredu_runtime::ReplicatedTextMaterializationTask] {
        &self.materialization_tasks
    }

    /// Consumes the immutable proof without dropping or re-selecting physical work.
    pub fn into_parts(
        self,
    ) -> (
        R,
        PartitionedAdmission<Q>,
        Vec<eredu_runtime::ReplicatedTextMaterializationTask>,
    ) {
        (self.base, self.requirements, self.materialization_tasks)
    }
}

/// Complete fail-closed partitioned-admission diagnostic.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
#[error("partitioned admission is unsupported: {issues}", issues = .issues.join("; "))]
pub struct PartitionedAdmissionSelectionError {
    issues: Vec<String>,
}

impl PartitionedAdmissionSelectionError {
    /// Every mismatch or unavailable communication mechanism in stable order.
    pub fn issues(&self) -> &[String] {
        &self.issues
    }
}

/// Admits a direct partition around one authoritative replicated realization.
pub fn select_direct_partitioned_admission(
    requirements: DirectPartitionedAdmission,
    selected: eredu_runtime::SelectedReplicatedTextRealization,
    communication: &CommunicationCapabilities,
) -> Result<
    SelectedPartitionedAdmission<
        eredu_runtime::SelectedReplicatedTextRealization,
        eredu_runtime::ReplicatedTextRequirements,
    >,
    PartitionedAdmissionSelectionError,
> {
    let materialization_tasks = preselect_materialization_tasks(&requirements, &selected)?;
    select_partitioned(
        requirements,
        selected,
        materialization_tasks,
        communication,
        |requirements, selected| selected.requirements() == requirements,
    )
}

/// Admits a routed partition around one authoritative routed realization.
pub fn select_routed_partitioned_admission(
    requirements: RoutedPartitionedAdmission,
    selected: SelectedRoutedTextRealization,
    communication: &CommunicationCapabilities,
) -> Result<
    SelectedPartitionedAdmission<SelectedRoutedTextRealization, RoutedTextRequirements>,
    PartitionedAdmissionSelectionError,
> {
    let materialization_tasks = preselect_materialization_tasks(&requirements, selected.text())?;
    select_partitioned(
        requirements,
        selected,
        materialization_tasks,
        communication,
        |requirements, selected| {
            crate::routed_text::validate_selected_routed_handoff(requirements, selected).is_ok()
        },
    )
}

/// Admits a composite partition around one authoritative composite realization.
pub fn select_composite_partitioned_admission(
    requirements: CompositePartitionedAdmission,
    selected: SelectedCompositeTextRealization,
    communication: &CommunicationCapabilities,
) -> Result<
    SelectedPartitionedAdmission<SelectedCompositeTextRealization, CompositeTextRequirements>,
    PartitionedAdmissionSelectionError,
> {
    let materialization_tasks =
        preselect_materialization_tasks(&requirements, selected.execution())?;
    select_partitioned(
        requirements,
        selected,
        materialization_tasks,
        communication,
        |requirements, selected| {
            let decoder_matches = match (requirements.routed_execution(), selected) {
                (None, SelectedCompositeTextRealization::Direct(_)) => true,
                (Some(expected), SelectedCompositeTextRealization::Routed { execution, .. }) => {
                    crate::routed_text::validate_selected_routed_handoff(expected, execution)
                        .is_ok()
                }
                _ => false,
            };
            decoder_matches
                && selected.execution().requirements() == requirements.execution()
                && selected.processor().requirements() == requirements.processor_execution()
        },
    )
}

fn select_partitioned<R, Q>(
    requirements: PartitionedAdmission<Q>,
    selected: R,
    materialization_tasks: Vec<eredu_runtime::ReplicatedTextMaterializationTask>,
    communication: &CommunicationCapabilities,
    agrees: impl FnOnce(&Q, &R) -> bool,
) -> Result<SelectedPartitionedAdmission<R, Q>, PartitionedAdmissionSelectionError>
where
    Q: TextRequirements,
{
    let mut issues = Vec::new();
    if !agrees(&requirements.execution, &selected) {
        issues.push("selected base realization does not match partitioned requirements".into());
    }
    if let Err(error) = communication.validate_manifest(&requirements.communication) {
        issues.push(error.to_string());
    }
    if let Err(error) = validate_selected_boundary_routes(&requirements) {
        issues.push(error);
    }
    if !issues.is_empty() {
        return Err(PartitionedAdmissionSelectionError { issues });
    }
    Ok(SelectedPartitionedAdmission {
        requirements,
        base: selected,
        materialization_tasks,
    })
}

fn preselect_materialization_tasks<Q>(
    requirements: &PartitionedAdmission<Q>,
    selected: &eredu_runtime::SelectedReplicatedTextRealization,
) -> Result<Vec<eredu_runtime::ReplicatedTextMaterializationTask>, PartitionedAdmissionSelectionError>
where
    Q: TextRequirements,
{
    let tasks =
        eredu_runtime::replicated_text_materialization_tasks(selected).map_err(|error| {
            PartitionedAdmissionSelectionError {
                issues: vec![error.to_string()],
            }
        })?;
    let targets = requirements
        .logical_parameter_targets()
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>();
    let mut claimed = BTreeSet::new();
    for task in &tasks {
        let primary_matches = std::iter::once(task.name())
            .chain(task.aliases().iter().map(String::as_str))
            .filter(|name| targets.contains(*name))
            .map(str::to_owned)
            .collect::<BTreeSet<_>>();
        if primary_matches.len() > 1 {
            return Err(PartitionedAdmissionSelectionError {
                issues: vec![format!(
                    "rank-local task {:?} ambiguously matches targets {primary_matches:?}",
                    task.name()
                )],
            });
        }
        let companion_matches = task
            .output_companions()
            .iter()
            .filter(|companion| targets.contains(companion.name()))
            .map(|companion| companion.name().to_owned())
            .collect::<BTreeSet<_>>();
        if primary_matches.is_empty() && !companion_matches.is_empty() {
            return Err(PartitionedAdmissionSelectionError {
                issues: vec![format!(
                    "rank-local task {:?} would select companions {companion_matches:?} without their atomic primary",
                    task.name(),
                )],
            });
        }
        for target in primary_matches.into_iter().chain(companion_matches) {
            if !claimed.insert(target.clone()) {
                return Err(PartitionedAdmissionSelectionError {
                    issues: vec![format!(
                        "rank-local parameter target {target:?} is claimed more than once"
                    )],
                });
            }
        }
    }
    let missing_required = targets
        .difference(&claimed)
        .filter(|target| {
            selected
                .requirements()
                .parameters()
                .iter()
                .find(|parameter| {
                    parameter.name() == target.as_str()
                        || parameter.aliases().iter().any(|alias| alias == *target)
                })
                .is_none_or(|parameter| {
                    parameter.presence().has_physical_source()
                        || matches!(
                            parameter.presence(),
                            eredu_runtime::ReplicatedTextParameterPresence::Derived { .. }
                        )
                })
        })
        .collect::<Vec<_>>();
    if !missing_required.is_empty() {
        return Err(PartitionedAdmissionSelectionError {
            issues: vec![format!(
                "rank-local exact task coverage differs from parameter ownership: missing={:?}",
                missing_required
            )],
        });
    }
    Ok(tasks)
}

pub(crate) fn validate_selected_boundary_routes<R>(
    requirements: &PartitionedAdmission<R>,
) -> Result<(), String>
where
    R: TextRequirements,
{
    let descriptors = requirements.communication.routes();
    if requirements.boundary_routes.len() != descriptors.len() {
        return Err("selected boundary-route cardinality differs from the manifest".into());
    }
    let graph = requirements.execution.text().execution_graph();
    let mut ids = std::collections::BTreeSet::new();
    for (selected, descriptor) in requirements.boundary_routes.iter().zip(descriptors) {
        let route = selected.route();
        if !ids.insert(route.route) {
            return Err("selected boundary routes contain a duplicate opaque identity".into());
        }
        if route.route != descriptor.id()
            || route.source_rank != descriptor.source()
            || route.destination_rank != descriptor.destination()
        {
            return Err("selected semantic boundary endpoints differ from the manifest".into());
        }
        if route.source_group >= graph.groups().len()
            || route.destination_group >= graph.groups().len()
            || (route.source_group != route.destination_group
                && !graph
                    .dependencies(route.destination_group)
                    .is_some_and(|dependencies| dependencies.contains(&route.source_group)))
        {
            return Err("selected semantic boundary route is not an execution-graph edge".into());
        }
        let contract = descriptor.boundary_contract().ok_or_else(|| {
            "selected boundary route has no role-exact manifest contract".to_owned()
        })?;
        let schema = selected.schema();
        if contract.schema() != schema.identity()
            || contract.roles().len() != 1 + schema.auxiliary().len()
            || contract
                .roles()
                .iter()
                .zip(schema.tensors())
                .any(|(role, spec)| role.role() != spec.role())
        {
            return Err(
                "selected boundary schema/cardinality differs from its manifest contract".into(),
            );
        }
    }
    Ok(())
}

/// Postconstruction binding of an admission to a validated concrete partition.
#[derive(Debug, Clone)]
pub struct BoundPartitionedAdmission<R, Q, G, W> {
    selected: SelectedPartitionedAdmission<R, Q>,
    partition: ArchitecturePartition<G, W>,
}

impl<R, Q, G, W> BoundPartitionedAdmission<R, Q, G, W> {
    /// Previously selected text/routed/composite realization.
    pub const fn base(&self) -> &R {
        &self.selected.base
    }

    /// Exact topology and local semantic coordinates.
    pub const fn topology(&self) -> ParallelRankTopology {
        self.selected.requirements.topology()
    }

    /// Selected opaque communication manifest.
    pub const fn communication(&self) -> &CommunicationManifest {
        &self.selected.requirements.communication
    }

    /// Exact opaque world/session group selected for publication and commit.
    pub const fn session_group(&self) -> Option<eredu_core::CollectiveGroupId> {
        self.selected.requirements.session_group
    }

    /// Exact opaque tensor group selected for this rank.
    pub const fn tensor_group(&self) -> Option<eredu_core::CollectiveGroupId> {
        self.selected.requirements.tensor_group
    }

    /// Exact opaque expert group selected for this rank.
    pub const fn expert_group(&self) -> Option<eredu_core::CollectiveGroupId> {
        self.selected.requirements.expert_group
    }

    /// Exact cache-placement topology derived before backend construction.
    pub fn prompt_cache_topology(&self) -> Result<eredu_core::cache::PromptCacheTopology, String> {
        self.selected.requirements.prompt_cache_topology()
    }

    /// Exact activation dtype admitted for this partition's communication wire.
    pub const fn activation_dtype(&self) -> PipelineActivationDtype {
        self.selected.requirements.activation_dtype
    }

    /// Selected semantic routes and their role-exact schemas.
    pub fn boundary_routes(&self) -> &[SelectedPartitionBoundaryRoute] {
        &self.selected.requirements.boundary_routes
    }

    /// Resolves one opaque route to the exact architecture-selected schema.
    pub fn boundary_schema_for_route(
        &self,
        route: CommunicationRouteId,
    ) -> Option<&ResolvedBoundaryWireSchema> {
        self.boundary_routes()
            .iter()
            .find(|selected| selected.route().route == route)
            .map(SelectedPartitionBoundaryRoute::schema)
    }

    /// Builds the exact pure-tensor driver and shared publication plan.
    pub fn direct_execution_plan(&self) -> Result<eredu_runtime::PartitionedExecutionPlan, String> {
        eredu_runtime::partitioned_execution::direct_execution_plan(
            &self.partition,
            self.topology(),
            self.session_group(),
            self.tensor_group(),
            self.selected.requirements.publication_owner(),
            self.activation_dtype(),
        )
    }

    /// Builds the exact routed TP/EP driver and shared publication plan.
    pub fn routed_execution_plan(&self) -> Result<eredu_runtime::PartitionedExecutionPlan, String> {
        let topology = self.topology();
        if topology.pipeline_parallel_size() != 1
            || (topology.tensor_parallel_size() == 1 && topology.expert_parallel_size() == 1)
        {
            return Err(
                "routed partition plan requires active tensor or expert parallelism".into(),
            );
        }
        let [owned] = self.partition.groups() else {
            return Err("routed partition must own exactly one group".into());
        };
        if !self.partition.ownership().owns_input() || !self.partition.ownership().owns_output() {
            return Err("routed partition must own input and output".into());
        }
        let group = owned.group_index();
        let driver = eredu_runtime::LayeredPartitionDriver::new(
            &self.partition,
            group,
            owned.global_units(),
        )
        .map_err(|error| error.to_string())?;
        let session = self.session_group().ok_or_else(|| {
            "routed partition admission has no selected publication group".to_owned()
        })?;
        let owner_rank = self.selected.requirements.publication_owner();
        let graph = self.partition.graph().clone();
        let mut contracts = vec![(ArchitectureGroupKind::Decoder, false); graph.groups().len()];
        contracts[group] = (ArchitectureGroupKind::Decoder, false);
        let mut drivers = vec![None; graph.groups().len()];
        drivers[group] = Some(driver);
        eredu_runtime::PartitionedExecutionPlan::new(
            graph,
            contracts,
            drivers,
            Vec::new(),
            Some(eredu_runtime::PartitionOutputPublication {
                group: session,
                owner_rank,
            }),
            Some(session),
            eredu_runtime::PipelineWireContract::new(self.activation_dtype()),
        )
        .map_err(|error| error.to_string())
    }

    /// Builds the exact pipeline driver selected by this admission.
    ///
    /// The plan uses only preserved topology, opaque group identities, route
    /// descriptors, and the validated partition. Backend composition therefore
    /// cannot reinterpret pipeline ownership or invent publication authority.
    pub fn pipeline_execution_plan(
        &self,
    ) -> Result<eredu_runtime::PartitionedExecutionPlan, String> {
        let topology = self.topology();
        if topology.pipeline_parallel_size() <= 1 {
            return Err("pipeline partition plan requires active pipeline parallelism".into());
        }
        let [owned] = self.partition.groups() else {
            return Err("pipeline partition must own exactly one group".into());
        };
        let group = owned.group_index();
        let driver = eredu_runtime::LayeredPartitionDriver::new(
            &self.partition,
            group,
            owned.global_units(),
        )
        .map_err(|error| error.to_string())?;
        let routes = self
            .boundary_routes()
            .iter()
            .map(|selected| selected.route().clone())
            .collect::<Vec<_>>();
        let session = self.session_group().ok_or_else(|| {
            "pipeline partition admission has no selected session publication group".to_owned()
        })?;
        let owner_rank = self.selected.requirements.publication_owner();
        let publication = eredu_runtime::PartitionOutputPublication {
            group: session,
            owner_rank,
        };
        let graph = self.partition.graph().clone();
        let mut contracts = vec![(ArchitectureGroupKind::Decoder, false); graph.groups().len()];
        contracts[group] = (ArchitectureGroupKind::Decoder, false);
        let mut drivers = vec![None; graph.groups().len()];
        drivers[group] = Some(driver);
        eredu_runtime::PartitionedExecutionPlan::new(
            graph,
            contracts,
            drivers,
            routes,
            Some(publication),
            Some(session),
            eredu_runtime::PipelineWireContract::new(self.activation_dtype()),
        )
        .map_err(|error| error.to_string())
    }

    /// Exact validated architecture partition.
    pub const fn partition(&self) -> &ArchitecturePartition<G, W> {
        &self.partition
    }

    /// Consumes the proof into its authoritative base selection and partition.
    pub fn into_parts(self) -> (R, ArchitecturePartition<G, W>, CommunicationManifest) {
        (
            self.selected.base,
            self.partition,
            self.selected.requirements.communication,
        )
    }
}

impl<R, G, W> BoundPartitionedAdmission<R, CompositeTextRequirements, G, W> {
    /// Builds one exact multi-group composite execution plan.
    pub fn composite_execution_plan(
        &self,
    ) -> Result<eredu_runtime::PartitionedExecutionPlan, String> {
        let topology = self.topology();
        if topology.topology().is_replicated() {
            return Err("composite partition plan requires an active parallel axis".into());
        }
        let graph = self.partition.graph().clone();
        let transports = self
            .selected
            .requirements
            .execution
            .text()
            .group_transports();
        if transports.len() != graph.groups().len() {
            return Err("composite group transports differ from its execution graph".into());
        }
        let contracts = transports
            .iter()
            .map(|transport| (transport.kind, transport.request_optional))
            .collect::<Vec<_>>();
        let mut drivers: Vec<Option<eredu_runtime::LayeredPartitionDriver>> =
            (0..graph.groups().len()).map(|_| None).collect();
        for owned in self.partition.groups() {
            let group = owned.group_index();
            let state_range = match self
                .selected
                .requirements
                .execution
                .text()
                .state_partition_plan()
            {
                Some(plan) => plan
                    .group_state_range(self.partition.unit_layout(), group, owned.global_units())
                    .map_err(|error| error.to_string())?,
                None => (transports[group].kind == ArchitectureGroupKind::Decoder)
                    .then(|| owned.global_units()),
            };
            drivers[group] = Some(
                eredu_runtime::LayeredPartitionDriver::new_with_state_range(
                    &self.partition,
                    group,
                    owned.global_units(),
                    state_range,
                )
                .map_err(|error| error.to_string())?,
            );
        }
        let routes = self
            .boundary_routes()
            .iter()
            .map(|selected| selected.route().clone())
            .collect::<Vec<_>>();
        let session = self.session_group().ok_or_else(|| {
            "composite partition admission has no selected publication group".to_owned()
        })?;
        let owner_rank = self.selected.requirements.publication_owner();
        eredu_runtime::PartitionedExecutionPlan::new(
            graph,
            contracts,
            drivers,
            routes,
            Some(eredu_runtime::PartitionOutputPublication {
                group: session,
                owner_rank,
            }),
            Some(session),
            eredu_runtime::PipelineWireContract::new(self.activation_dtype()),
        )
        .map_err(|error| error.to_string())
    }
}

/// Typed architecture paired with a postconstruction partition binding.
pub struct PreparedPartitionedAdmission<A, R, Q, G, W> {
    architecture: A,
    selected: BoundPartitionedAdmission<R, Q, G, W>,
}

impl<A, R, Q, G, W> PreparedPartitionedAdmission<A, R, Q, G, W> {
    /// Concrete statically dispatched architecture.
    pub const fn architecture(&self) -> &A {
        &self.architecture
    }

    /// Validated partition binding and preserved base realization.
    pub const fn selected(&self) -> &BoundPartitionedAdmission<R, Q, G, W> {
        &self.selected
    }

    /// Consumes the typed handoff.
    pub fn into_parts(self) -> (A, BoundPartitionedAdmission<R, Q, G, W>) {
        (self.architecture, self.selected)
    }
}

/// Failure while a typed architecture is paired with an admitted partition.
#[derive(Debug, thiserror::Error)]
pub enum PartitionedDispatchError<E> {
    /// The concrete architecture or partition disagreed with admission.
    #[error("invalid selected architecture partition: {0}")]
    Architecture(String),
    /// The generic partition constructor rejected the typed handoff.
    #[error("partitioned architecture binding failed: {0}")]
    Visitor(E),
}

pub(crate) fn prepare_partitioned<B, S, A, R, Q, G, W>(
    architecture: A,
    selected: SelectedPartitionedAdmission<R, Q>,
    partition: ArchitecturePartition<G, W>,
) -> Result<PreparedPartitionedAdmission<A, R, Q, G, W>, String>
where
    B: eredu_nn::NeuralBackend,
    S: eredu_runtime::RuntimeState<B>,
    A: eredu_runtime::LayeredArchitecture<B, S>,
    A::Error: std::fmt::Display,
    Q: TextRequirements,
{
    selected.requirements.validate_binding(&partition)?;
    partition
        .validate_architecture::<B, S, A>(&architecture)
        .map_err(|error| error.to_string())?;
    Ok(PreparedPartitionedAdmission {
        architecture,
        selected: BoundPartitionedAdmission {
            selected,
            partition,
        },
    })
}

impl<R> PartitionedAdmission<R> {
    pub(crate) fn ownership_view(
        &self,
    ) -> (
        &eredu_runtime::ReplicatedTextRequirements,
        &eredu_runtime::PipelineRankProjection,
    )
    where
        R: TextRequirements,
    {
        (self.execution.text(), &self.rank)
    }

    pub(crate) fn construct_partition<G, W>(
        &self,
        parameters: &eredu_runtime::ArchitectureParameterDescription,
        complete_state: &eredu_runtime::StateLayout,
        state_plan: &eredu_runtime::ArchitectureStatePartitionPlan,
        local_geometry: G,
        boundary_schema: W,
    ) -> Result<ArchitecturePartition<G, W>, eredu_runtime::PipelineRankBindingError>
    where
        R: TextRequirements,
        W: eredu_runtime::ArchitectureBoundary,
    {
        self.rank.construct_partition(
            self.execution.text(),
            parameters,
            complete_state,
            state_plan,
            local_geometry,
            boundary_schema,
        )
    }

    pub(crate) fn validate_binding<G, W>(
        &self,
        partition: &ArchitecturePartition<G, W>,
    ) -> Result<(), String>
    where
        R: TextRequirements,
    {
        self.rank
            .validate_partition(self.execution.text(), partition)
            .map_err(|error| error.to_string())
    }
}

pub(crate) trait TextRequirements {
    fn text(&self) -> &eredu_runtime::ReplicatedTextRequirements;
}

fn invalid(message: &str) -> eredu_core::parameters::ParameterError {
    eredu_core::parameters::ParameterError::Invalid(message.into())
}

/// Prediction modules retain their own TP placement and are physically replicated
/// over the other execution axes. Use the same single-group compiler as preparation,
/// while preserving the global rank in public coordinate ownership.
pub(crate) fn prediction_parameter_layout_for_rank(
    prepared: &crate::prediction_extension::PreparedPredictionPlacement,
    tasks: &[eredu_runtime::ReplicatedTextMaterializationTask],
    parameter: &str,
    global_rank: usize,
    reservation: &mut (impl eredu_core::capture::CaptureReservation + ?Sized),
) -> Result<eredu_runtime::ParameterPartitionLayout, eredu_core::parameters::ParameterError> {
    use eredu_core::parameters::ParameterError;
    let topology = ParallelRankTopology::new(prepared.topology().topology(), global_rank)
        .map_err(|error| invalid(&error.to_string()))?;
    let parameters = prepared.parameters().ok_or_else(|| {
        ParameterError::Unsupported(
            "prepared prediction has no parameter placement compiler".into(),
        )
    })?;
    let resolved = eredu_runtime::resolve_parameter_member(tasks, parameters, parameter)?;
    let tensor = eredu_core::ParallelTopology::new(topology.tensor_parallel_size(), 1, 1, 1)
        .map_err(|error| invalid(&error.to_string()))?;
    let placement = ParallelRankTopology::new(tensor, topology.tensor_parallel_rank())
        .map_err(|error| invalid(&error.to_string()))?;
    // Reuse the prepared local layout and preserve architecture-authored peer
    // layouts. Ordinary peer placement needs only this atomic parameter group;
    // deriving the entire model for every catalog member makes queries quadratic.
    let layout = if prepared.qwen4_spec().is_some()
        || topology.tensor_parallel_rank() == prepared.topology().tensor_parallel_rank()
    {
        prepared
            .layout_for_rank(global_rank)
            .map_err(ParameterError::Unsupported)?
    } else {
        None
    };
    eredu_runtime::derive_parameter_member_layout(
        resolved,
        topology,
        placement,
        true,
        layout.as_deref(),
        reservation,
    )
}

/// One architecture-authored boundary at its actual execution-unit ordinal.
/// The point is replicated across the unit's TP/EP participants, not other PP stages.
pub(crate) struct PartitionedUnitObservation {
    pub group: eredu_runtime::ExecutionGroupId,
    pub unit: usize,
    pub path: String,
}

/// Projects another rank from the same retained global requirements. This uses
/// exactly the ownership compiler used by admission, without state allocation,
/// source resolution, communication realization or execution-class reselection.
pub(crate) fn selected_component_layout_for_rank(
    ownership: (
        &eredu_runtime::ReplicatedTextRequirements,
        &eredu_runtime::PipelineRankProjection,
    ),
    descriptor: &eredu_core::ArchitectureDescriptor,
    parameters: &eredu_runtime::ArchitectureParameterDescription,
    global_rank: usize,
    routed: Option<&crate::SelectedRoutedTextRealization>,
    authored: Option<&eredu_runtime::LocalModelLayout>,
    unit_observations: &[PartitionedUnitObservation],
) -> Result<
    crate::component_partition::ComponentPartitionLayout,
    crate::component_partition::ComponentPartitionError,
> {
    use crate::component_partition::{ComponentPartitionError, ComponentPartitionLayout};
    let (execution, admitted_rank) = ownership;
    let topology = ParallelRankTopology::new(admitted_rank.topology().topology(), global_rank)
        .map_err(|error| ComponentPartitionError::ParameterLayout(error.to_string()))?;
    let derived;
    let layout = if let Some(layout) = authored {
        layout
    } else {
        derived = derive_partitioned_local_layout(parameters, topology)
            .map_err(ComponentPartitionError::ParameterLayout)?;
        &derived
    };
    let projected_rank;
    let rank = if topology == admitted_rank.topology() {
        admitted_rank
    } else {
        projected_rank =
            eredu_runtime::PipelineRankProjection::new(execution, None, topology, false)
                .map_err(|error| ComponentPartitionError::ParameterLayout(error.to_string()))?;
        &projected_rank
    };
    let mut result = ComponentPartitionLayout::from_ownership(
        descriptor,
        parameters,
        layout,
        topology,
        rank.ownership(),
        rank.groups(),
        rank.publication_owner(),
        routed,
        execution.component_definition(),
    )?;
    for point in unit_observations {
        let declared = parameters
            .graph()
            .group_index(point.group.as_str())
            .and_then(|group| parameters.unit_layout().ordinal(group, point.unit));
        if declared.is_none() {
            return Err(ComponentPartitionError::InvalidPlacement(
                point.path.clone(),
            ));
        }
        let local = rank.owns_unit(&point.group, point.unit);
        result.register_unit_observation(descriptor, &point.path, local)?;
    }
    Ok(result)
}

impl TextRequirements for eredu_runtime::ReplicatedTextRequirements {
    fn text(&self) -> &eredu_runtime::ReplicatedTextRequirements {
        self
    }
}

impl TextRequirements for RoutedTextRequirements {
    fn text(&self) -> &eredu_runtime::ReplicatedTextRequirements {
        self.text()
    }
}

impl TextRequirements for CompositeTextRequirements {
    fn text(&self) -> &eredu_runtime::ReplicatedTextRequirements {
        self.execution()
    }
}

pub use eredu_runtime::executable_binding::dense_partition::{
    DenseDecoderPartitionedDispatchError, PartitionedArchitectureVisitor,
    PartitionedPredictionTargetVisitor, PreparedPartitionedArchitecture,
};

fn prepare_dense_partition_binding<B, A, G, W>(
    prepared: PreparedPartitionedAdmission<
        A,
        eredu_runtime::SelectedReplicatedTextRealization,
        eredu_runtime::ReplicatedTextRequirements,
        G,
        W,
    >,
    source_architecture: Option<(A, eredu_runtime::LocalModelLayout)>,
    layout: eredu_runtime::LocalModelLayout,
    tasks: Vec<eredu_runtime::ReplicatedTextMaterializationTask>,
    capability_estimate: crate::capability::CapabilityEstimate,
    effective_model_type: String,
) -> PreparedPartitionedArchitecture<B, A, G, W>
where
    B: eredu_nn::NeuralBackend,
    A: eredu_runtime::ArchitectureParameters<B>,
    A::DefinitionError: std::fmt::Display,
{
    let pipeline = prepared.selected().topology().pipeline_parallel_size() > 1;
    let session_facts = crate::prepared_execution::partition_facts::dense_session_facts::<B, _, _, _>(
        &prepared,
        &capability_estimate,
        &effective_model_type,
        pipeline,
    );
    let (architecture, selected) = prepared.into_parts();
    let (base, partition, communication) = selected.into_parts();
    PreparedPartitionedArchitecture::from_admission(
        architecture,
        base,
        partition,
        communication,
        source_architecture,
        layout,
        tasks,
        session_facts,
        pipeline,
    )
}

/// Exact generic dense-decoder partition constructed from a prior admission.
pub type PreparedDenseDecoderPartitionedArchitecture<B, C, P> = PreparedPartitionedArchitecture<
    B,
    crate::decoder::PartitionedLayeredModel<B, C, P>,
    crate::decoder::PartitionLocalGeometry<C>,
    eredu_runtime::NoAuxiliaryBoundarySchema,
>;

/// Exact dense LFM2 partition constructed from a prior admission.
pub type PreparedLfm2PartitionedArchitecture<B> = PreparedPartitionedArchitecture<
    B,
    crate::lfm2::PartitionedLayeredModel<B>,
    crate::lfm2::PartitionLocalGeometry,
    eredu_runtime::NoAuxiliaryBoundarySchema,
>;

fn visit_dense_decoder_partitioned_architecture_internal<B, S, V>(
    inspection: &ArtifactInspection<ArtifactArchitecturePlan>,
    selected: SelectedPartitionedAdmission<
        eredu_runtime::SelectedReplicatedTextRealization,
        eredu_runtime::ReplicatedTextRequirements,
    >,
    store: eredu_checkpoint::store::SharedCheckpointSource,
    context: &<B::Tensor as eredu_nn::Tensor>::Context,
    visitor: V,
) -> Result<V::Output, DenseDecoderPartitionedDispatchError<V::Error>>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::BlockwiseAttentionBackend,
    S: eredu_runtime::LayerRuntimeState<B>,
    S::LayerState: eredu_nn::ReadOnlyAttentionCache<B::Tensor>
        + eredu_nn::CompressedAttentionCache<B::Tensor>
        + eredu_runtime::RuntimeStateComponents<B>,
    V: PartitionedArchitectureVisitor<B, S>,
{
    let expected = crate::replicated_text::replicated_text_requirements(inspection)
        .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    if &expected != selected.requirements().execution() {
        return Err(DenseDecoderPartitionedDispatchError::Architecture(
            "partitioned admission belongs to a different architecture or artifact".into(),
        ));
    }
    eredu_runtime::validate_checkpoint_handoff(
        &expected,
        store.as_ref(),
        eredu_runtime::SourceHandoffScope::Primary,
    )
    .map_err(|handoff| DenseDecoderPartitionedDispatchError::Architecture(handoff.to_string()))?;
    match (
        inspection
            .architecture_plan()
            .safetensors_architecture()
            .map(|plan| plan.model()),
        inspection
            .architecture_plan()
            .gguf_plan()
            .map(|plan| plan.model()),
    ) {
        (Some(crate::configuration::SafetensorsModelConfig::NemotronLabsDiffusion(args)), None) => {
            nemotron_labs_diffusion::prepare::<B, S, V>(args, selected, store, context, visitor)
        }
        (None, Some(crate::configuration::GgufModelConfig::NemotronLabsDiffusion(plan))) => {
            nemotron_labs_diffusion::prepare::<B, S, V>(
                plan.text_config(),
                selected,
                store,
                context,
                visitor,
            )
        }
        (Some(crate::configuration::SafetensorsModelConfig::DiffusionGemma(args)), None)
        | (None, Some(crate::configuration::GgufModelConfig::DiffusionGemma(args))) => {
            diffusion::prepare_dense::<B, S, V>(args, selected, store, context, visitor)
        }
        (Some(crate::configuration::SafetensorsModelConfig::DeepSeekV3(args)), None)
        | (None, Some(crate::configuration::GgufModelConfig::DeepSeekV3(args))) => {
            deepseek_dense::prepare::<B, S, V>(args, selected, store, context, visitor)
        }
        (Some(crate::configuration::SafetensorsModelConfig::Nanbeige(args)), None)
        | (None, Some(crate::configuration::GgufModelConfig::Nanbeige(args))) => {
            let source_args = selected
                .base()
                .uses_weight_transform()
                .then(|| crate::replicated_text::source_nanbeige_args(args, selected.base()))
                .transpose()
                .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
            let selected_args =
                crate::replicated_text::selected_nanbeige_args(args, selected.base())
                    .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
            let admitted = inspection
                .architecture_plan()
                .dense_declaration()
                .ok_or_else(|| {
                    DenseDecoderPartitionedDispatchError::Architecture(
                        "missing admitted dense scopes".into(),
                    )
                })?;
            let dense_scopes = crate::replicated_text::DecoderConstructionScopes {
                source: std::sync::Arc::clone(admitted),
                selected: std::sync::Arc::new(
                    crate::nanbeige::project_declaration_formats(admitted, args, &selected_args)
                        .map_err(|e| {
                            DenseDecoderPartitionedDispatchError::Architecture(e.to_string())
                        })?,
                ),
            };
            let estimate = crate::capability::nanbeige(&selected_args, &dense_scopes.selected)
                .map_err(|e| DenseDecoderPartitionedDispatchError::Architecture(e.to_string()))?;
            prepare_dense_decoder_partition::<B, S, _, crate::decoder::DenseBlockFactory, V>(
                "nanbeige".into(),
                selected_args,
                source_args,
                dense_scopes,
                selected,
                store,
                context,
                visitor,
                estimate,
            )
        }
        (Some(crate::configuration::SafetensorsModelConfig::Gemma2(args)), None)
        | (None, Some(crate::configuration::GgufModelConfig::Gemma2(args))) => {
            let source_args = selected
                .base()
                .uses_weight_transform()
                .then(|| crate::replicated_text::source_gemma2_args(args, selected.base()))
                .transpose()
                .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
            let selected_args = crate::replicated_text::selected_gemma2_args(args, selected.base())
                .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;

            let admitted = inspection
                .architecture_plan()
                .dense_declaration()
                .ok_or_else(|| {
                    DenseDecoderPartitionedDispatchError::Architecture(
                        "missing admitted dense scopes".into(),
                    )
                })?;
            let dense_scopes = crate::replicated_text::DecoderConstructionScopes {
                selected: std::sync::Arc::new(
                    crate::gemma2::project_declaration_formats(admitted, args, &selected_args)
                        .map_err(|error| {
                            DenseDecoderPartitionedDispatchError::Architecture(error.to_string())
                        })?,
                ),
                source: std::sync::Arc::new(
                    crate::gemma2::project_declaration_formats(
                        admitted,
                        args,
                        source_args.as_ref().unwrap_or(args),
                    )
                    .map_err(|error| {
                        DenseDecoderPartitionedDispatchError::Architecture(error.to_string())
                    })?,
                ),
            };
            let capability_estimate =
                crate::capability::gemma2(&selected_args, &dense_scopes.selected).map_err(
                    |error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()),
                )?;
            prepare_dense_decoder_partition::<B, S, _, crate::decoder::DenseBlockFactory, V>(
                "gemma2".to_owned(),
                selected_args,
                source_args,
                dense_scopes,
                selected,
                store,
                context,
                visitor,
                capability_estimate,
            )
        }
        (Some(crate::configuration::SafetensorsModelConfig::Llama(args)), None)
        | (None, Some(crate::configuration::GgufModelConfig::Llama(args))) => {
            let source_args = selected
                .base()
                .uses_weight_transform()
                .then(|| crate::replicated_text::source_llama_args(args, selected.base()))
                .transpose()
                .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
            let selected_args = crate::replicated_text::selected_llama_args(args, selected.base())
                .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;

            let admitted = inspection
                .architecture_plan()
                .dense_declaration()
                .ok_or_else(|| {
                    DenseDecoderPartitionedDispatchError::Architecture(
                        "missing admitted dense scopes".into(),
                    )
                })?;
            let dense_scopes = crate::replicated_text::DecoderConstructionScopes {
                selected: std::sync::Arc::new(
                    crate::llama::project_declaration_formats(admitted, args, &selected_args)
                        .map_err(|error| {
                            DenseDecoderPartitionedDispatchError::Architecture(error.to_string())
                        })?,
                ),
                source: std::sync::Arc::new(
                    crate::llama::project_declaration_formats(
                        admitted,
                        args,
                        source_args.as_ref().unwrap_or(args),
                    )
                    .map_err(|error| {
                        DenseDecoderPartitionedDispatchError::Architecture(error.to_string())
                    })?,
                ),
            };
            let capability_estimate =
                crate::capability::llama(&selected_args, &dense_scopes.selected).map_err(
                    |error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()),
                )?;
            prepare_dense_decoder_partition::<B, S, _, crate::decoder::DenseBlockFactory, V>(
                selected_args.model_type.clone(),
                selected_args,
                source_args,
                dense_scopes,
                selected,
                store,
                context,
                visitor,
                capability_estimate,
            )
        }
        (Some(crate::configuration::SafetensorsModelConfig::Qwen(args)), None)
        | (None, Some(crate::configuration::GgufModelConfig::Qwen(args)))
            if !args.is_moe() =>
        {
            let source_args = selected
                .base()
                .uses_weight_transform()
                .then(|| crate::replicated_text::source_qwen_args(args, selected.base()))
                .transpose()
                .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
            let selected_args = crate::replicated_text::selected_qwen_args(args, selected.base())
                .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
            let source_declaration = inspection
                .architecture_plan()
                .qwen_declaration(source_args.as_ref().unwrap_or(args))
                .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
            let declaration = inspection
                .architecture_plan()
                .qwen_declaration(&selected_args)
                .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
            let dense_scopes = crate::replicated_text::DecoderConstructionScopes {
                source: source_declaration.common().clone(),
                selected: declaration.common().clone(),
            };
            let capability_estimate = crate::capability::qwen(&declaration)
                .map_err(|e| DenseDecoderPartitionedDispatchError::Architecture(e.to_string()))?;
            prepare_dense_decoder_partition::<B, S, _, crate::qwen::QwenBlockFactory, V>(
                selected_args.model_type.clone(),
                selected_args,
                source_args,
                dense_scopes,
                selected,
                store,
                context,
                visitor,
                capability_estimate,
            )
        }
        (Some(crate::configuration::SafetensorsModelConfig::K2Horizon(args)), None)
        | (None, Some(crate::configuration::GgufModelConfig::K2Horizon(args)))
            if !args.is_moe() =>
        {
            let source_args = selected
                .base()
                .uses_weight_transform()
                .then(|| crate::replicated_text::source_k2_horizon_args(args, selected.base()))
                .transpose()
                .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
            let selected_args =
                crate::replicated_text::selected_k2_horizon_args(args, selected.base())
                    .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
            let source_declaration = inspection
                .architecture_plan()
                .k2_declaration(source_args.as_ref().unwrap_or(args))
                .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
            let declaration = inspection
                .architecture_plan()
                .k2_declaration(&selected_args)
                .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
            let dense_scopes = crate::replicated_text::DecoderConstructionScopes {
                source: source_declaration.common().clone(),
                selected: declaration.common().clone(),
            };
            let capability = crate::capability::k2_horizon(&declaration)
                .map_err(|e| DenseDecoderPartitionedDispatchError::Architecture(e.to_string()))?;
            prepare_dense_decoder_partition::<B, S, _, crate::k2_horizon::DenseBlockFactory, V>(
                "k2_horizon".to_owned(),
                selected_args,
                source_args,
                dense_scopes,
                selected,
                store,
                context,
                visitor,
                capability,
            )
        }
        _ => Err(DenseDecoderPartitionedDispatchError::Architecture(
            "partitioned architecture is not a supported dense decoder".into(),
        )),
    }
}

/// Constructs and visits the exact architecture-owned neutral partition.
///
/// This is the exhaustive production dispatch seam. Unsupported families never
/// reach a backend visitor, and the selected admission is consumed exactly once.
pub fn visit_resident_partitioned_architecture<B, S, V>(
    inspection: &ArtifactInspection<ArtifactArchitecturePlan>,
    selected: SelectedPartitionedAdmission<
        eredu_runtime::SelectedReplicatedTextRealization,
        eredu_runtime::ReplicatedTextRequirements,
    >,
    store: eredu_checkpoint::store::SharedCheckpointSource,
    context: &<B::Tensor as eredu_nn::Tensor>::Context,
    visitor: V,
) -> Result<V::Output, DenseDecoderPartitionedDispatchError<V::Error>>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::BlockwiseAttentionBackend,
    S: eredu_runtime::LayerRuntimeState<B>,
    S::LayerState: eredu_nn::ReadOnlyAttentionCache<B::Tensor>
        + eredu_nn::CompressedAttentionCache<B::Tensor>
        + eredu_runtime::RuntimeStateComponents<B>,
    V: PartitionedArchitectureVisitor<B, S>,
{
    let safetensors = inspection
        .architecture_plan()
        .safetensors_architecture()
        .map(|plan| plan.model());
    let gguf = inspection
        .architecture_plan()
        .gguf_plan()
        .map(|plan| plan.model());
    if let Some(args) = match (safetensors, gguf) {
        (Some(crate::configuration::SafetensorsModelConfig::QwenHybrid(args)), None)
        | (None, Some(crate::configuration::GgufModelConfig::QwenHybrid(args)))
            if args.vision.is_none()
                && !args.text.is_moe()
                && args.text.mtp_num_hidden_layers == 0 =>
        {
            Some(&args.text)
        }
        _ => None,
    } {
        let expected =
            crate::replicated_text::replicated_text_requirements(inspection).map_err(|error| {
                DenseDecoderPartitionedDispatchError::Architecture(error.to_string())
            })?;
        if &expected != selected.requirements().execution() {
            return Err(DenseDecoderPartitionedDispatchError::Architecture(
                "partitioned admission belongs to a different architecture or artifact".into(),
            ));
        }
        eredu_runtime::validate_checkpoint_handoff(
            &expected,
            store.as_ref(),
            eredu_runtime::SourceHandoffScope::Primary,
        )
        .map_err(|handoff| {
            DenseDecoderPartitionedDispatchError::Architecture(handoff.to_string())
        })?;
        return qwen_hybrid::prepare::<B, S, V>(args, selected, store, context, visitor);
    }
    if let Some(args) = match (safetensors, gguf) {
        (Some(crate::configuration::SafetensorsModelConfig::NemotronH(args)), None)
        | (None, Some(crate::configuration::GgufModelConfig::NemotronH(args))) => Some(args),
        _ => None,
    } {
        let expected =
            crate::replicated_text::replicated_text_requirements(inspection).map_err(|error| {
                DenseDecoderPartitionedDispatchError::Architecture(error.to_string())
            })?;
        if &expected != selected.requirements().execution() {
            return Err(DenseDecoderPartitionedDispatchError::Architecture(
                "partitioned admission belongs to a different architecture or artifact".into(),
            ));
        }
        eredu_runtime::validate_checkpoint_handoff(
            &expected,
            store.as_ref(),
            eredu_runtime::SourceHandoffScope::Primary,
        )
        .map_err(|handoff| {
            DenseDecoderPartitionedDispatchError::Architecture(handoff.to_string())
        })?;
        let selected_args = crate::replicated_text::selected_nemotron_h_args(args, selected.base())
            .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
        let source_args = selected
            .base()
            .uses_weight_transform()
            .then(|| args.clone());
        let capability_estimate = crate::capability::nemotron_h(
            inspection
                .architecture_plan()
                .nemotron_declaration(&selected_args)
                .map_err(DenseDecoderPartitionedDispatchError::Architecture)?
                .as_ref(),
        )
        .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
        return prepare_nemotron_h_partition::<B, S, _>(
            inspection.architecture_plan(),
            selected_args.model_type.clone(),
            selected_args,
            source_args,
            selected,
            store,
            context,
            OrdinaryNemotronPartitionVisitor(visitor),
            capability_estimate,
        );
    }
    if let Some(args) = match (safetensors, gguf) {
        (Some(crate::configuration::SafetensorsModelConfig::KimiLinear(args)), None)
        | (None, Some(crate::configuration::GgufModelConfig::KimiLinear(args))) => Some(args),
        _ => None,
    } {
        let expected =
            crate::replicated_text::replicated_text_requirements(inspection).map_err(|error| {
                DenseDecoderPartitionedDispatchError::Architecture(error.to_string())
            })?;
        if &expected != selected.requirements().execution() {
            return Err(DenseDecoderPartitionedDispatchError::Architecture(
                "partitioned admission belongs to a different architecture or artifact".into(),
            ));
        }
        eredu_runtime::validate_checkpoint_handoff(
            &expected,
            store.as_ref(),
            eredu_runtime::SourceHandoffScope::Primary,
        )
        .map_err(|handoff| {
            DenseDecoderPartitionedDispatchError::Architecture(handoff.to_string())
        })?;
        let selected_args =
            crate::replicated_text::selected_kimi_linear_args(args, selected.base())
                .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
        let source_args = selected
            .base()
            .uses_weight_transform()
            .then(|| args.clone());
        let capability_estimate = crate::capability::kimi_linear(
            inspection
                .architecture_plan()
                .kimi_declaration(&selected_args)
                .map_err(DenseDecoderPartitionedDispatchError::Architecture)?
                .as_ref(),
        )
        .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
        return prepare_kimi_linear_partition::<B, S, V>(
            inspection.architecture_plan(),
            selected_args.model_type.clone(),
            selected_args,
            source_args,
            selected,
            store,
            context,
            visitor,
            capability_estimate,
        );
    }
    if matches!(
        (safetensors, gguf),
        (
            Some(crate::configuration::SafetensorsModelConfig::Lfm3(_)),
            None
        ) | (None, Some(crate::configuration::GgufModelConfig::Lfm3(_)))
    ) {
        let expected =
            crate::replicated_text::replicated_text_requirements(inspection).map_err(|error| {
                DenseDecoderPartitionedDispatchError::Architecture(error.to_string())
            })?;
        if &expected != selected.requirements().execution() {
            return Err(DenseDecoderPartitionedDispatchError::Architecture(
                "partitioned admission belongs to a different architecture or artifact".into(),
            ));
        }
        eredu_runtime::validate_checkpoint_handoff(
            &expected,
            store.as_ref(),
            eredu_runtime::SourceHandoffScope::Primary,
        )
        .map_err(|handoff| {
            DenseDecoderPartitionedDispatchError::Architecture(handoff.to_string())
        })?;
        let args = match (safetensors, gguf) {
            (Some(crate::configuration::SafetensorsModelConfig::Lfm3(args)), None)
            | (None, Some(crate::configuration::GgufModelConfig::Lfm3(args))) => args,
            _ => unreachable!("LFM3 partitioned dispatch was classified above"),
        };
        let selected_args = crate::replicated_text::selected_lfm3_args(args, selected.base())
            .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
        let source_args = selected
            .base()
            .uses_weight_transform()
            .then(|| args.clone());
        let capability_estimate = crate::capability::lfm3(
            inspection
                .architecture_plan()
                .lfm3_declaration(&selected_args)
                .map_err(DenseDecoderPartitionedDispatchError::Architecture)?
                .as_ref(),
        )
        .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
        return prepare_lfm3_partition::<B, S, V>(
            inspection.architecture_plan(),
            selected_args.model_type.clone(),
            selected_args,
            source_args,
            selected,
            store,
            context,
            visitor,
            capability_estimate,
        );
    }
    let is_lfm2 = match (safetensors, gguf) {
        (Some(crate::configuration::SafetensorsModelConfig::Lfm2(args)), None)
        | (None, Some(crate::configuration::GgufModelConfig::Lfm2(args))) => {
            !args.has_sparse_moe_layers()
        }
        _ => false,
    };
    if !is_lfm2 {
        return visit_dense_decoder_partitioned_architecture_internal::<B, S, V>(
            inspection, selected, store, context, visitor,
        );
    }
    let expected = crate::replicated_text::replicated_text_requirements(inspection)
        .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    if &expected != selected.requirements().execution() {
        return Err(DenseDecoderPartitionedDispatchError::Architecture(
            "partitioned admission belongs to a different architecture or artifact".into(),
        ));
    }
    eredu_runtime::validate_checkpoint_handoff(
        &expected,
        store.as_ref(),
        eredu_runtime::SourceHandoffScope::Primary,
    )
    .map_err(|handoff| DenseDecoderPartitionedDispatchError::Architecture(handoff.to_string()))?;
    let args = match (safetensors, gguf) {
        (Some(crate::configuration::SafetensorsModelConfig::Lfm2(args)), None)
        | (None, Some(crate::configuration::GgufModelConfig::Lfm2(args))) => args,
        _ => unreachable!("LFM2 partitioned dispatch was classified above"),
    };
    let selected_args = crate::replicated_text::selected_lfm2_args(args, selected.base())
        .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
    let source_args = selected
        .base()
        .uses_weight_transform()
        .then(|| args.clone());
    let capability_estimate = crate::capability::lfm2(
        inspection
            .architecture_plan()
            .lfm2_declaration(&selected_args)
            .map_err(DenseDecoderPartitionedDispatchError::Architecture)?
            .as_ref(),
    )
    .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    prepare_lfm2_partition::<B, S, V>(
        inspection.architecture_plan(),
        selected_args.model_type.clone(),
        selected_args,
        source_args,
        selected,
        store,
        context,
        visitor,
        capability_estimate,
    )
}

/// Constructs a resident Nemotron-H prediction target and pairs it before backend erasure.
pub fn visit_resident_partitioned_prediction_target_architecture<B, S, M, V>(
    inspection: &ArtifactInspection<ArtifactArchitecturePlan>,
    selected: SelectedPartitionedAdmission<
        eredu_runtime::SelectedReplicatedTextRealization,
        eredu_runtime::ReplicatedTextRequirements,
    >,
    extension: crate::prediction_extension::MaterializedPredictionExtension<B, M>,
    store: eredu_checkpoint::store::SharedCheckpointSource,
    context: &<B::Tensor as eredu_nn::Tensor>::Context,
    visitor: V,
) -> Result<V::Output, DenseDecoderPartitionedDispatchError<V::Error>>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::HyperNeuralBackend,
    S: eredu_runtime::LayerRuntimeState<B>,
    S::LayerState: eredu_nn::AttentionCache<B::Tensor>
        + eredu_nn::CompressedAttentionCache<B::Tensor>
        + eredu_runtime::RuntimeStateComponents<B>,
    M: crate::prediction_extension::PredictionExtensionMaterializer<B>,
    V: PartitionedPredictionTargetVisitor<B, S, M>,
{
    if let Some(args) = match (
        inspection
            .architecture_plan()
            .safetensors_architecture()
            .map(|plan| plan.model()),
        inspection
            .architecture_plan()
            .gguf_plan()
            .map(|plan| plan.model()),
    ) {
        (Some(crate::configuration::SafetensorsModelConfig::QwenHybrid(args)), None)
        | (None, Some(crate::configuration::GgufModelConfig::QwenHybrid(args)))
            if args.vision.is_none() =>
        {
            Some(&args.text)
        }
        _ => None,
    } {
        let expected =
            crate::replicated_text::replicated_text_requirements(inspection).map_err(|error| {
                DenseDecoderPartitionedDispatchError::Architecture(error.to_string())
            })?;
        if &expected != selected.requirements().execution() {
            return Err(DenseDecoderPartitionedDispatchError::Architecture(
                "Qwen prediction target admission belongs to another artifact".into(),
            ));
        }
        eredu_runtime::validate_checkpoint_handoff(
            &expected,
            store.as_ref(),
            eredu_runtime::SourceHandoffScope::Primary,
        )
        .map_err(|handoff| {
            DenseDecoderPartitionedDispatchError::Architecture(handoff.to_string())
        })?;
        return qwen_hybrid::prepare_prediction::<B, S, M, V>(
            args, selected, extension, store, context, visitor,
        );
    }

    let extension = <crate::nemotron_h::PartitionedLayeredModel<B> as crate::prediction_extension::MaterializedPredictionTarget<B>>::pair_prediction_extension(extension)
        .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    let args = match (
        inspection
            .architecture_plan()
            .safetensors_architecture()
            .map(|plan| plan.model()),
        inspection
            .architecture_plan()
            .gguf_plan()
            .map(|plan| plan.model()),
    ) {
        (Some(crate::configuration::SafetensorsModelConfig::NemotronH(args)), None)
        | (None, Some(crate::configuration::GgufModelConfig::NemotronH(args))) => args,
        _ => {
            return Err(DenseDecoderPartitionedDispatchError::Architecture(
                "resident prediction target requires Nemotron-H".into(),
            ));
        }
    };
    let expected = crate::replicated_text::replicated_text_requirements(inspection)
        .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    if &expected != selected.requirements().execution() {
        return Err(DenseDecoderPartitionedDispatchError::Architecture(
            "partitioned admission belongs to a different architecture or artifact".into(),
        ));
    }
    eredu_runtime::validate_checkpoint_handoff(
        &expected,
        store.as_ref(),
        eredu_runtime::SourceHandoffScope::Primary,
    )
    .map_err(|handoff| DenseDecoderPartitionedDispatchError::Architecture(handoff.to_string()))?;
    let selected_args = crate::replicated_text::selected_nemotron_h_args(args, selected.base())
        .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
    let source_args = selected
        .base()
        .uses_weight_transform()
        .then(|| args.clone());
    let capability_estimate = crate::capability::nemotron_h(
        inspection
            .architecture_plan()
            .nemotron_declaration(&selected_args)
            .map_err(DenseDecoderPartitionedDispatchError::Architecture)?
            .as_ref(),
    )
    .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    prepare_nemotron_h_partition::<B, S, _>(
        inspection.architecture_plan(),
        selected_args.model_type.clone(),
        selected_args,
        source_args,
        selected,
        store,
        context,
        PredictionNemotronPartitionVisitor::<B, M, _> {
            extension,
            visitor,
            marker: PhantomData,
        },
        capability_estimate,
    )
}

trait NemotronPartitionVisitor<B, S>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: eredu_runtime::LayerRuntimeState<B>,
    S::LayerState: eredu_nn::AttentionCache<B::Tensor> + eredu_runtime::RuntimeStateComponents<B>,
{
    type Output;
    type Error;

    fn visit(
        self,
        prepared: PreparedPartitionedArchitecture<
            B,
            crate::nemotron_h::PartitionedLayeredModel<B>,
            crate::nemotron_h::PartitionLocalGeometry,
            crate::nemotron_h::TargetBoundarySchema,
        >,
        store: eredu_checkpoint::store::SharedCheckpointSource,
    ) -> Result<Self::Output, Self::Error>;
}

struct OrdinaryNemotronPartitionVisitor<V>(V);

impl<B, S, V> NemotronPartitionVisitor<B, S> for OrdinaryNemotronPartitionVisitor<V>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: eredu_runtime::LayerRuntimeState<B>,
    S::LayerState: eredu_nn::AttentionCache<B::Tensor> + eredu_runtime::RuntimeStateComponents<B>,
    V: PartitionedArchitectureVisitor<B, S>,
{
    type Output = V::Output;
    type Error = V::Error;

    fn visit(
        self,
        prepared: PreparedPartitionedArchitecture<
            B,
            crate::nemotron_h::PartitionedLayeredModel<B>,
            crate::nemotron_h::PartitionLocalGeometry,
            crate::nemotron_h::TargetBoundarySchema,
        >,
        store: eredu_checkpoint::store::SharedCheckpointSource,
    ) -> Result<Self::Output, Self::Error> {
        self.0.visit(prepared, store)
    }
}

struct PredictionNemotronPartitionVisitor<B, M, V>
where
    B: eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::HyperNeuralBackend,
    M: crate::prediction_extension::PredictionExtensionMaterializer<B>,
{
    extension: <crate::nemotron_h::PartitionedLayeredModel<B> as eredu_runtime::speculative::PredictionTarget<B>>::Extension<M>,
    visitor: V,
    marker: PhantomData<fn() -> (B, M)>,
}

impl<B, S, M, V> NemotronPartitionVisitor<B, S> for PredictionNemotronPartitionVisitor<B, M, V>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::HyperNeuralBackend,
    S: eredu_runtime::LayerRuntimeState<B>,
    S::LayerState: eredu_nn::AttentionCache<B::Tensor> + eredu_runtime::RuntimeStateComponents<B>,
    M: crate::prediction_extension::PredictionExtensionMaterializer<B>,
    V: PartitionedPredictionTargetVisitor<B, S, M>,
{
    type Output = V::Output;
    type Error = V::Error;

    fn visit(
        self,
        prepared: PreparedPartitionedArchitecture<
            B,
            crate::nemotron_h::PartitionedLayeredModel<B>,
            crate::nemotron_h::PartitionLocalGeometry,
            crate::nemotron_h::TargetBoundarySchema,
        >,
        store: eredu_checkpoint::store::SharedCheckpointSource,
    ) -> Result<Self::Output, Self::Error> {
        self.visitor.visit(prepared, self.extension, store)
    }
}

#[allow(clippy::too_many_arguments)]
fn prepare_nemotron_h_partition<B, S, V>(
    architecture: &ArtifactArchitecturePlan,
    effective_model_type: String,
    selected_args: crate::nemotron_h::ModelArgs,
    source_args: Option<crate::nemotron_h::ModelArgs>,
    selected: SelectedPartitionedAdmission<
        eredu_runtime::SelectedReplicatedTextRealization,
        eredu_runtime::ReplicatedTextRequirements,
    >,
    store: eredu_checkpoint::store::SharedCheckpointSource,
    context: &<B::Tensor as eredu_nn::Tensor>::Context,
    visitor: V,
    capability_estimate: crate::capability::CapabilityEstimate,
) -> Result<V::Output, DenseDecoderPartitionedDispatchError<V::Error>>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::BlockwiseAttentionBackend,
    S: eredu_runtime::LayerRuntimeState<B>,
    S::LayerState: eredu_nn::AttentionCache<B::Tensor>
        + eredu_nn::CompressedAttentionCache<B::Tensor>
        + eredu_runtime::RuntimeStateComponents<B>,
    V: NemotronPartitionVisitor<B, S>,
{
    if selected_args.has_sparse_moe_layers() || selected_args.num_nextn_predict_layers != 0 {
        return Err(DenseDecoderPartitionedDispatchError::Architecture(
            "resident Nemotron-H requires a dense prediction-free target schedule".into(),
        ));
    }
    let retained = selected
        .base()
        .requirements()
        .shared_component_definition()
        .ok_or_else(|| {
            DenseDecoderPartitionedDispatchError::Architecture(
                "Nemotron-H selection has no retained definition".into(),
            )
        })?;
    let definition = crate::replicated_text::executable_component(selected.base())
        .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
    let parameters = definition.parameters().clone();
    let rank = selected.requirements().topology();
    let layout = derive_partitioned_local_layout(&parameters, rank)
        .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
    let [owned] = selected.requirements().groups() else {
        return Err(DenseDecoderPartitionedDispatchError::Architecture(
            "dense Nemotron-H must own exactly one execution group".into(),
        ));
    };
    if owned.group().as_str() != crate::decoder::TARGET_EXECUTION_GROUP {
        return Err(DenseDecoderPartitionedDispatchError::Architecture(
            "Nemotron-H admission names a different execution group".into(),
        ));
    }
    let declaration = architecture
        .nemotron_declaration(&selected_args)
        .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
    let geometry =
        crate::nemotron_h::partition_local_geometry(&declaration, &layout, owned.units()).map_err(
            |error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()),
        )?;
    let complete_state = geometry.complete_state_layout().clone();
    let state_plan = crate::transport::pipeline_state(0, &complete_state);
    let partition = selected
        .requirements()
        .construct_partition(
            &parameters,
            &complete_state,
            &state_plan,
            geometry,
            crate::nemotron_h::TargetBoundarySchema::from_declaration(&declaration),
        )
        .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    let tasks = eredu_runtime::partition_selected_replicated_text_materialization_tasks(
        selected.materialization_tasks(),
        &parameters,
        &partition,
    )
    .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    let source_architecture = source_args
        .map(|source_args| {
            let source_declaration = architecture
                .nemotron_declaration(&source_args)
                .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
            let source_definition = retained
                .project_parameters(source_declaration.definition().parameters().clone())
                .map_err(|error| {
                    DenseDecoderPartitionedDispatchError::Architecture(error.to_string())
                })?;
            let source_parameters = source_definition.parameters().clone();
            let source_layout =
                derive_partitioned_transform_source_layout(&source_parameters, &parameters, rank)
                    .map_err(|error| {
                    DenseDecoderPartitionedDispatchError::Architecture(error.to_string())
                })?;
            let source_geometry = crate::nemotron_h::partition_local_geometry(
                &source_declaration,
                &source_layout,
                owned.units(),
            )
            .map_err(|error| {
                DenseDecoderPartitionedDispatchError::Architecture(error.to_string())
            })?;
            if source_geometry.complete_state_layout() != &complete_state {
                return Err(DenseDecoderPartitionedDispatchError::Architecture(
                    "Nemotron-H transform source changed rank-local state geometry".into(),
                ));
            }
            let source_partition = selected
                .requirements()
                .construct_partition(
                    &source_parameters,
                    &complete_state,
                    &state_plan,
                    source_geometry,
                    crate::nemotron_h::TargetBoundarySchema::from_declaration(&source_declaration),
                )
                .map_err(|error| {
                    DenseDecoderPartitionedDispatchError::Architecture(error.to_string())
                })?;
            let source_architecture = crate::nemotron_h::NemotronKernel::<B>::from_partition(
                source_declaration,
                source_definition,
                &source_partition,
                context,
            )
            .map_err(|error| {
                DenseDecoderPartitionedDispatchError::Architecture(error.to_string())
            })?;
            Ok((source_architecture, source_layout))
        })
        .transpose()?;
    let architecture = crate::nemotron_h::NemotronKernel::<B>::from_partition(
        declaration,
        definition,
        &partition,
        context,
    )
    .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    let prepared = prepare_partitioned::<B, S, _, _, _, _, _>(architecture, selected, partition)
        .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
    visitor
        .visit(
            prepare_dense_partition_binding(
                prepared,
                source_architecture,
                layout,
                tasks,
                capability_estimate,
                effective_model_type,
            ),
            store,
        )
        .map_err(DenseDecoderPartitionedDispatchError::Visitor)
}

#[allow(clippy::too_many_arguments)]
fn prepare_kimi_linear_partition<B, S, V>(
    architecture: &ArtifactArchitecturePlan,
    effective_model_type: String,
    selected_args: crate::kimi_linear::ModelArgs,
    source_args: Option<crate::kimi_linear::ModelArgs>,
    selected: SelectedPartitionedAdmission<
        eredu_runtime::SelectedReplicatedTextRealization,
        eredu_runtime::ReplicatedTextRequirements,
    >,
    store: eredu_checkpoint::store::SharedCheckpointSource,
    context: &<B::Tensor as eredu_nn::Tensor>::Context,
    visitor: V,
    capability_estimate: crate::capability::CapabilityEstimate,
) -> Result<V::Output, DenseDecoderPartitionedDispatchError<V::Error>>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::BlockwiseAttentionBackend,
    S: eredu_runtime::LayerRuntimeState<B>,
    S::LayerState: eredu_nn::AttentionCache<B::Tensor>
        + eredu_nn::CompressedAttentionCache<B::Tensor>
        + eredu_runtime::RuntimeStateComponents<B>,
    V: PartitionedArchitectureVisitor<B, S>,
{
    if selected_args.has_sparse_moe_layers() || selected_args.num_nextn_predict_layers != 0 {
        return Err(DenseDecoderPartitionedDispatchError::Architecture(
            "resident Kimi Linear requires a dense prediction-free schedule".into(),
        ));
    }
    let retained = selected
        .base()
        .requirements()
        .shared_component_definition()
        .ok_or_else(|| {
            DenseDecoderPartitionedDispatchError::Architecture(
                "Kimi Linear selection has no retained definition".into(),
            )
        })?;
    let definition = crate::replicated_text::executable_component(selected.base())
        .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
    let parameters = definition.parameters().clone();
    let rank = selected.requirements().topology();
    let layout = derive_partitioned_local_layout(&parameters, rank)
        .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
    let [owned] = selected.requirements().groups() else {
        return Err(DenseDecoderPartitionedDispatchError::Architecture(
            "dense Kimi Linear must own exactly one execution group".into(),
        ));
    };
    if owned.group().as_str() != crate::decoder::TARGET_EXECUTION_GROUP {
        return Err(DenseDecoderPartitionedDispatchError::Architecture(
            "Kimi Linear admission names a different execution group".into(),
        ));
    }
    let declaration = architecture
        .kimi_declaration(&selected_args)
        .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
    let geometry =
        crate::kimi_linear::partition_local_geometry(&declaration, &layout, owned.units())
            .map_err(|error| {
                DenseDecoderPartitionedDispatchError::Architecture(error.to_string())
            })?;
    let complete_state = geometry.complete_state_layout().clone();
    let state_plan = crate::transport::pipeline_state(0, &complete_state);
    let partition = selected
        .requirements()
        .construct_partition(
            &parameters,
            &complete_state,
            &state_plan,
            geometry,
            eredu_runtime::NoAuxiliaryBoundarySchema::new(selected_args.hidden_size),
        )
        .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    let tasks = eredu_runtime::partition_selected_replicated_text_materialization_tasks(
        selected.materialization_tasks(),
        &parameters,
        &partition,
    )
    .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    let source_architecture = source_args
        .map(|source_args| {
            let source_args = crate::kimi_linear::with_checkpoint_formats(
                &source_args,
                selected.base().requirements().native_quantized_formats(),
            )
            .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
            let source_declaration = architecture
                .kimi_declaration(&source_args)
                .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
            let source_definition = retained
                .project_parameters(source_declaration.definition().parameters().clone())
                .map_err(|error| {
                    DenseDecoderPartitionedDispatchError::Architecture(error.to_string())
                })?;
            let source_parameters = source_definition.parameters().clone();
            let source_layout =
                derive_partitioned_transform_source_layout(&source_parameters, &parameters, rank)
                    .map_err(|error| {
                    DenseDecoderPartitionedDispatchError::Architecture(error.to_string())
                })?;
            let source_geometry = crate::kimi_linear::partition_local_geometry(
                &source_declaration,
                &source_layout,
                owned.units(),
            )
            .map_err(|error| {
                DenseDecoderPartitionedDispatchError::Architecture(error.to_string())
            })?;
            if source_geometry.complete_state_layout() != &complete_state {
                return Err(DenseDecoderPartitionedDispatchError::Architecture(
                    "Kimi Linear transform source changed rank-local state geometry".into(),
                ));
            }
            let source_partition = selected
                .requirements()
                .construct_partition(
                    &source_parameters,
                    &complete_state,
                    &state_plan,
                    source_geometry,
                    eredu_runtime::NoAuxiliaryBoundarySchema::new(source_args.hidden_size),
                )
                .map_err(|error| {
                    DenseDecoderPartitionedDispatchError::Architecture(error.to_string())
                })?;
            let source_architecture = crate::kimi_linear::KimiKernel::<B>::from_partition(
                source_declaration,
                source_definition,
                &source_partition,
                context,
            )
            .map_err(|error| {
                DenseDecoderPartitionedDispatchError::Architecture(error.to_string())
            })?;
            Ok((source_architecture, source_layout))
        })
        .transpose()?;
    let architecture = crate::kimi_linear::KimiKernel::<B>::from_partition(
        declaration,
        definition,
        &partition,
        context,
    )
    .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    let prepared = prepare_partitioned::<B, S, _, _, _, _, _>(architecture, selected, partition)
        .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
    visitor
        .visit(
            prepare_dense_partition_binding(
                prepared,
                source_architecture,
                layout,
                tasks,
                capability_estimate,
                effective_model_type,
            ),
            store,
        )
        .map_err(DenseDecoderPartitionedDispatchError::Visitor)
}

#[allow(clippy::too_many_arguments)]
fn prepare_lfm2_partition<B, S, V>(
    architecture: &ArtifactArchitecturePlan,
    effective_model_type: String,
    selected_args: crate::lfm2::ModelArgs,
    source_args: Option<crate::lfm2::ModelArgs>,
    selected: SelectedPartitionedAdmission<
        eredu_runtime::SelectedReplicatedTextRealization,
        eredu_runtime::ReplicatedTextRequirements,
    >,
    store: eredu_checkpoint::store::SharedCheckpointSource,
    context: &<B::Tensor as eredu_nn::Tensor>::Context,
    visitor: V,
    capability_estimate: crate::capability::CapabilityEstimate,
) -> Result<V::Output, DenseDecoderPartitionedDispatchError<V::Error>>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: eredu_runtime::LayerRuntimeState<B>,
    S::LayerState: eredu_nn::AttentionCache<B::Tensor> + eredu_runtime::RuntimeStateComponents<B>,
    V: PartitionedArchitectureVisitor<B, S>,
{
    if selected_args.has_sparse_moe_layers() {
        return Err(DenseDecoderPartitionedDispatchError::Architecture(
            "dense LFM2 parameter topology does not accept routed layers".into(),
        ));
    }
    let retained = selected
        .base()
        .requirements()
        .shared_component_definition()
        .ok_or_else(|| {
            DenseDecoderPartitionedDispatchError::Architecture(
                "LFM2 selection has no retained definition".into(),
            )
        })?;
    let definition = crate::replicated_text::executable_component(selected.base())
        .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
    let parameters = definition.parameters().clone();
    let rank = selected.requirements().topology();
    let layout = derive_partitioned_local_layout(&parameters, rank)
        .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
    let [owned] = selected.requirements().groups() else {
        return Err(DenseDecoderPartitionedDispatchError::Architecture(
            "dense LFM2 must own exactly one execution group".into(),
        ));
    };
    if owned.group().as_str() != crate::decoder::TARGET_EXECUTION_GROUP {
        return Err(DenseDecoderPartitionedDispatchError::Architecture(
            "LFM2 admission names a different execution group".into(),
        ));
    }
    let declaration = architecture
        .lfm2_declaration(&selected_args)
        .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
    let geometry = crate::lfm2::partition_local_geometry(&declaration, &layout, owned.units())
        .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    let complete_state = geometry.complete_state_layout().clone();
    let state_plan = crate::transport::pipeline_state(0, &complete_state);
    let partition = selected
        .requirements()
        .construct_partition(
            &parameters,
            &complete_state,
            &state_plan,
            geometry,
            eredu_runtime::NoAuxiliaryBoundarySchema::new(selected_args.hidden_size),
        )
        .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    let tasks = eredu_runtime::partition_selected_replicated_text_materialization_tasks(
        selected.materialization_tasks(),
        &parameters,
        &partition,
    )
    .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    let source_architecture = source_args
        .map(|source_args| {
            let source_args = crate::lfm2::with_checkpoint_formats(
                &source_args,
                selected.base().requirements().native_quantized_formats(),
            )
            .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
            let source_declaration = architecture
                .lfm2_declaration(&source_args)
                .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
            let source_definition = retained
                .project_parameters(source_declaration.definition().parameters().clone())
                .map_err(|error| {
                    DenseDecoderPartitionedDispatchError::Architecture(error.to_string())
                })?;
            let source_parameters = source_definition.parameters().clone();
            let source_layout =
                derive_partitioned_transform_source_layout(&source_parameters, &parameters, rank)
                    .map_err(|error| {
                    DenseDecoderPartitionedDispatchError::Architecture(error.to_string())
                })?;
            let source_geometry = crate::lfm2::partition_local_geometry(
                &source_declaration,
                &source_layout,
                owned.units(),
            )
            .map_err(|error| {
                DenseDecoderPartitionedDispatchError::Architecture(error.to_string())
            })?;
            if source_geometry.complete_state_layout() != &complete_state {
                return Err(DenseDecoderPartitionedDispatchError::Architecture(
                    "LFM2 transform source changed rank-local state geometry".into(),
                ));
            }
            let source_partition = selected
                .requirements()
                .construct_partition(
                    &source_parameters,
                    &complete_state,
                    &state_plan,
                    source_geometry,
                    eredu_runtime::NoAuxiliaryBoundarySchema::new(source_args.hidden_size),
                )
                .map_err(|error| {
                    DenseDecoderPartitionedDispatchError::Architecture(error.to_string())
                })?;
            let source_architecture = crate::lfm2::Lfm2Kernel::<B>::from_partition(
                source_declaration,
                source_definition,
                &source_partition,
                context,
            )
            .map_err(|error| {
                DenseDecoderPartitionedDispatchError::Architecture(error.to_string())
            })?;
            Ok((source_architecture, source_layout))
        })
        .transpose()?;
    let architecture = crate::lfm2::Lfm2Kernel::<B>::from_partition(
        architecture
            .lfm2_declaration(&selected_args)
            .map_err(DenseDecoderPartitionedDispatchError::Architecture)?,
        definition,
        &partition,
        context,
    )
    .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    let prepared = prepare_partitioned::<B, S, _, _, _, _, _>(architecture, selected, partition)
        .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
    visitor
        .visit(
            prepare_dense_partition_binding(
                prepared,
                source_architecture,
                layout,
                tasks,
                capability_estimate,
                effective_model_type,
            ),
            store,
        )
        .map_err(DenseDecoderPartitionedDispatchError::Visitor)
}
#[allow(clippy::too_many_arguments)]
fn prepare_lfm3_partition<B, S, V>(
    architecture: &ArtifactArchitecturePlan,
    effective_model_type: String,
    selected_args: crate::lfm3::ModelArgs,
    source_args: Option<crate::lfm3::ModelArgs>,
    selected: SelectedPartitionedAdmission<
        eredu_runtime::SelectedReplicatedTextRealization,
        eredu_runtime::ReplicatedTextRequirements,
    >,
    store: eredu_checkpoint::store::SharedCheckpointSource,
    context: &<B::Tensor as eredu_nn::Tensor>::Context,
    visitor: V,
    capability_estimate: crate::capability::CapabilityEstimate,
) -> Result<V::Output, DenseDecoderPartitionedDispatchError<V::Error>>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: eredu_runtime::LayerRuntimeState<B>,
    S::LayerState: eredu_nn::AttentionCache<B::Tensor> + eredu_runtime::RuntimeStateComponents<B>,
    V: PartitionedArchitectureVisitor<B, S>,
{
    let retained = selected
        .base()
        .requirements()
        .shared_component_definition()
        .ok_or_else(|| {
            DenseDecoderPartitionedDispatchError::Architecture(
                "LFM3 selection has no retained definition".into(),
            )
        })?;
    let definition = crate::replicated_text::executable_component(selected.base())
        .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
    let parameters = definition.parameters().clone();
    let rank = selected.requirements().topology();
    let layout = derive_partitioned_local_layout(&parameters, rank)
        .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
    let [owned] = selected.requirements().groups() else {
        return Err(DenseDecoderPartitionedDispatchError::Architecture(
            "dense LFM3 must own exactly one execution group".into(),
        ));
    };
    if owned.group().as_str() != crate::decoder::TARGET_EXECUTION_GROUP {
        return Err(DenseDecoderPartitionedDispatchError::Architecture(
            "LFM3 admission names a different execution group".into(),
        ));
    }
    let declaration = architecture
        .lfm3_declaration(&selected_args)
        .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
    let geometry = crate::lfm3::partition_local_geometry(&declaration, &layout, owned.units())
        .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    let complete_state = geometry.complete_state_layout().clone();
    let state_plan = crate::transport::pipeline_state(0, &complete_state);
    let partition = selected
        .requirements()
        .construct_partition(
            &parameters,
            &complete_state,
            &state_plan,
            geometry,
            eredu_runtime::NoAuxiliaryBoundarySchema::new(selected_args.hidden_size),
        )
        .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    let tasks = eredu_runtime::partition_selected_replicated_text_materialization_tasks(
        selected.materialization_tasks(),
        &parameters,
        &partition,
    )
    .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    let source_architecture = source_args
        .map(|source_args| {
            let source_args = crate::lfm3::with_checkpoint_formats(
                &source_args,
                selected.base().requirements().native_quantized_formats(),
            )
            .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
            let source_declaration = architecture
                .lfm3_declaration(&source_args)
                .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
            let source_definition = retained
                .project_parameters(source_declaration.definition().parameters().clone())
                .map_err(|error| {
                    DenseDecoderPartitionedDispatchError::Architecture(error.to_string())
                })?;
            let source_parameters = source_definition.parameters().clone();
            let source_layout =
                derive_partitioned_transform_source_layout(&source_parameters, &parameters, rank)
                    .map_err(|error| {
                    DenseDecoderPartitionedDispatchError::Architecture(error.to_string())
                })?;
            let source_geometry = crate::lfm3::partition_local_geometry(
                &source_declaration,
                &source_layout,
                owned.units(),
            )
            .map_err(|error| {
                DenseDecoderPartitionedDispatchError::Architecture(error.to_string())
            })?;
            if source_geometry.complete_state_layout() != &complete_state {
                return Err(DenseDecoderPartitionedDispatchError::Architecture(
                    "LFM3 transform source changed rank-local state geometry".into(),
                ));
            }
            let source_partition = selected
                .requirements()
                .construct_partition(
                    &source_parameters,
                    &complete_state,
                    &state_plan,
                    source_geometry,
                    eredu_runtime::NoAuxiliaryBoundarySchema::new(source_args.hidden_size),
                )
                .map_err(|error| {
                    DenseDecoderPartitionedDispatchError::Architecture(error.to_string())
                })?;
            let source_architecture = crate::lfm3::Lfm3Kernel::<B>::from_partition(
                source_declaration,
                source_definition,
                &source_partition,
                context,
            )
            .map_err(|error| {
                DenseDecoderPartitionedDispatchError::Architecture(error.to_string())
            })?;
            Ok((source_architecture, source_layout))
        })
        .transpose()?;
    let architecture = crate::lfm3::Lfm3Kernel::<B>::from_partition(
        architecture
            .lfm3_declaration(&selected_args)
            .map_err(DenseDecoderPartitionedDispatchError::Architecture)?,
        definition,
        &partition,
        context,
    )
    .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    let prepared = prepare_partitioned::<B, S, _, _, _, _, _>(architecture, selected, partition)
        .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
    visitor
        .visit(
            prepare_dense_partition_binding(
                prepared,
                source_architecture,
                layout,
                tasks,
                capability_estimate,
                effective_model_type,
            ),
            store,
        )
        .map_err(DenseDecoderPartitionedDispatchError::Visitor)
}

fn prepare_dense_decoder_partition<B, S, C, P, V>(
    effective_model_type: String,
    selected_args: C,
    source_args: Option<C>,
    dense_scopes: crate::replicated_text::DecoderConstructionScopes,
    selected: SelectedPartitionedAdmission<
        eredu_runtime::SelectedReplicatedTextRealization,
        eredu_runtime::ReplicatedTextRequirements,
    >,
    store: eredu_checkpoint::store::SharedCheckpointSource,
    context: &<B::Tensor as eredu_nn::Tensor>::Context,
    visitor: V,
    capability_estimate: crate::capability::CapabilityEstimate,
) -> Result<V::Output, DenseDecoderPartitionedDispatchError<V::Error>>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: eredu_runtime::LayerRuntimeState<B>,
    S::LayerState: eredu_nn::AttentionCache<B::Tensor> + eredu_runtime::RuntimeStateComponents<B>,
    C: crate::decoder::PartitionedConfig,
    P: crate::decoder::BlockFactory<B, C>,
    P::FeedForward: crate::decoder::TensorParallelProjectionOperator<B>,
    V: PartitionedArchitectureVisitor<B, S>,
{
    let definition = crate::replicated_text::executable_component(selected.base())
        .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
    let parameters = definition.parameters();
    let rank = selected.requirements().topology();
    let layout = derive_partitioned_local_layout(parameters, rank)
        .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
    let [owned] = selected.requirements().groups() else {
        return Err(DenseDecoderPartitionedDispatchError::Architecture(
            "dense decoder must own exactly one execution group".into(),
        ));
    };
    if owned.group().as_str() != crate::decoder::TEXT_DECODER_EXECUTION_GROUP {
        return Err(DenseDecoderPartitionedDispatchError::Architecture(
            "dense-decoder admission names a different execution group".into(),
        ));
    }
    let geometry = crate::decoder::partition_local_geometry(
        &selected_args,
        dense_scopes.selected.as_ref(),
        parameters.invocations(),
        &layout,
        owned.units(),
    )
    .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    let complete_state = geometry.complete_state_layout().clone();
    let state_plan = definition.state_partition_plan();
    let partition = selected
        .requirements()
        .construct_partition(
            parameters,
            &complete_state,
            state_plan,
            geometry,
            eredu_runtime::NoAuxiliaryBoundarySchema::new(selected_args.hidden_size()),
        )
        .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    let tasks = eredu_runtime::partition_selected_replicated_text_materialization_tasks(
        selected.materialization_tasks(),
        parameters,
        &partition,
    )
    .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    let source_architecture = source_args
        .map(|source_args| {
            let source_definition = selected
                .base()
                .requirements()
                .shared_component_definition()
                .ok_or_else(|| {
                    DenseDecoderPartitionedDispatchError::Architecture(
                        "decoder selection has no retained source definition".into(),
                    )
                })?;
            let source_parameters = source_definition.parameters();
            let source_layout =
                derive_partitioned_transform_source_layout(source_parameters, parameters, rank)
                    .map_err(|error| {
                        DenseDecoderPartitionedDispatchError::Architecture(error.to_string())
                    })?;
            let source_geometry = crate::decoder::partition_local_geometry(
                &source_args,
                dense_scopes.source.as_ref(),
                source_parameters.invocations(),
                &source_layout,
                owned.units(),
            )
            .map_err(|error| {
                DenseDecoderPartitionedDispatchError::Architecture(error.to_string())
            })?;
            if source_geometry.complete_state_layout() != &complete_state {
                return Err(DenseDecoderPartitionedDispatchError::Architecture(
                    "source and target transform state geometry differs before local construction"
                        .into(),
                ));
            }
            let source_partition = selected
                .requirements()
                .construct_partition(
                    source_parameters,
                    &source_geometry.complete_state_layout().clone(),
                    state_plan,
                    source_geometry,
                    eredu_runtime::NoAuxiliaryBoundarySchema::new(source_args.hidden_size()),
                )
                .map_err(|error| {
                    DenseDecoderPartitionedDispatchError::Architecture(error.to_string())
                })?;
            partition
                .validate_transform_parameter_owners(
                    &source_partition,
                    tasks.iter().map(|task| task.name()),
                )
                .map_err(|error| {
                    DenseDecoderPartitionedDispatchError::Architecture(error.to_string())
                })?;
            let source_architecture = crate::decoder::DecoderKernel::<B, C, P>::from_partition(
                source_args,
                source_definition,
                dense_scopes.source.as_ref(),
                None,
                &source_partition,
                context,
            )
            .map_err(|error| {
                DenseDecoderPartitionedDispatchError::Architecture(error.to_string())
            })?;
            Ok((source_architecture, source_layout))
        })
        .transpose()?;
    let architecture = crate::decoder::DecoderKernel::<B, C, P>::from_partition(
        selected_args,
        definition,
        dense_scopes.selected.as_ref(),
        None,
        &partition,
        context,
    )
    .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    let prepared = prepare_partitioned::<B, S, _, _, _, _, _>(architecture, selected, partition)
        .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
    visitor
        .visit(
            prepare_dense_partition_binding(
                prepared,
                source_architecture,
                layout,
                tasks,
                capability_estimate,
                effective_model_type,
            ),
            store,
        )
        .map_err(DenseDecoderPartitionedDispatchError::Visitor)
}

/// Returns whether the normalized architecture has an exact partition constructor.
pub fn is_supported_dense_decoder_partition(plan: &ArtifactArchitecturePlan) -> bool {
    match (
        plan.safetensors_architecture().map(|plan| plan.model()),
        plan.gguf_plan().map(|plan| plan.model()),
    ) {
        (Some(crate::configuration::SafetensorsModelConfig::NemotronLabsDiffusion(args)), None) => {
            args.vision.is_none()
        }
        (None, Some(crate::configuration::GgufModelConfig::NemotronLabsDiffusion(_))) => true,

        (Some(crate::configuration::SafetensorsModelConfig::Gemma2(_)), None)
        | (None, Some(crate::configuration::GgufModelConfig::Gemma2(_)))
        | (Some(crate::configuration::SafetensorsModelConfig::Llama(_)), None)
        | (None, Some(crate::configuration::GgufModelConfig::Llama(_)))
        | (Some(crate::configuration::SafetensorsModelConfig::Nanbeige(_)), None)
        | (None, Some(crate::configuration::GgufModelConfig::Nanbeige(_))) => true,
        (Some(crate::configuration::SafetensorsModelConfig::K2Horizon(args)), None)
        | (None, Some(crate::configuration::GgufModelConfig::K2Horizon(args))) => !args.is_moe(),
        (Some(crate::configuration::SafetensorsModelConfig::Qwen(args)), None)
        | (None, Some(crate::configuration::GgufModelConfig::Qwen(args))) => !args.is_moe(),
        (Some(crate::configuration::SafetensorsModelConfig::Lfm2(args)), None)
        | (None, Some(crate::configuration::GgufModelConfig::Lfm2(args))) => {
            !args.has_sparse_moe_layers()
        }
        (Some(crate::configuration::SafetensorsModelConfig::KimiLinear(args)), None)
        | (None, Some(crate::configuration::GgufModelConfig::KimiLinear(args))) => {
            !args.has_sparse_moe_layers() && args.num_nextn_predict_layers == 0
        }
        (Some(crate::configuration::SafetensorsModelConfig::NemotronH(args)), None)
        | (None, Some(crate::configuration::GgufModelConfig::NemotronH(args))) => {
            !args.has_sparse_moe_layers() && args.num_nextn_predict_layers == 0
        }
        (Some(crate::configuration::SafetensorsModelConfig::DeepSeekV3(args)), None)
        | (None, Some(crate::configuration::GgufModelConfig::DeepSeekV3(args))) => {
            !args.has_sparse_moe_layers() && args.num_nextn_predict_layers == 0
        }
        (Some(crate::configuration::SafetensorsModelConfig::QwenHybrid(args)), None)
        | (None, Some(crate::configuration::GgufModelConfig::QwenHybrid(args))) => {
            args.vision.is_none() && !args.text.is_moe() && args.text.mtp_num_hidden_layers == 0
        }
        (Some(crate::configuration::SafetensorsModelConfig::DiffusionGemma(args)), None)
        | (None, Some(crate::configuration::GgufModelConfig::DiffusionGemma(args))) => {
            args.vision.is_none() && !args.has_sparse_moe_layers()
        }
        (Some(crate::configuration::SafetensorsModelConfig::Lfm3(_)), None)
        | (None, Some(crate::configuration::GgufModelConfig::Lfm3(_))) => true,
        _ => false,
    }
}

/// Returns whether the normalized architecture has an exact routed partition constructor.
pub fn is_supported_routed_decoder_partition(plan: &ArtifactArchitecturePlan) -> bool {
    match (
        plan.safetensors_architecture().map(|plan| plan.model()),
        plan.gguf_plan().map(|plan| plan.model()),
    ) {
        (Some(crate::configuration::SafetensorsModelConfig::K2Horizon(args)), None)
        | (None, Some(crate::configuration::GgufModelConfig::K2Horizon(args))) => args.is_moe(),
        (Some(crate::configuration::SafetensorsModelConfig::Qwen(args)), None)
        | (None, Some(crate::configuration::GgufModelConfig::Qwen(args))) => args.is_moe(),
        (Some(crate::configuration::SafetensorsModelConfig::GptOss(_)), None)
        | (None, Some(crate::configuration::GgufModelConfig::GptOss(_))) => true,
        (Some(crate::configuration::SafetensorsModelConfig::Lfm2(args)), None)
        | (None, Some(crate::configuration::GgufModelConfig::Lfm2(args))) => {
            args.has_sparse_moe_layers()
        }
        (Some(crate::configuration::SafetensorsModelConfig::KimiLinear(args)), None)
        | (None, Some(crate::configuration::GgufModelConfig::KimiLinear(args))) => {
            args.has_sparse_moe_layers() && args.num_nextn_predict_layers == 0
        }
        (Some(crate::configuration::SafetensorsModelConfig::DeepSeekV3(args)), None)
        | (None, Some(crate::configuration::GgufModelConfig::DeepSeekV3(args))) => {
            args.has_sparse_moe_layers() && args.num_nextn_predict_layers == 0
        }
        (Some(crate::configuration::SafetensorsModelConfig::NemotronH(args)), None)
        | (None, Some(crate::configuration::GgufModelConfig::NemotronH(args))) => {
            args.has_sparse_moe_layers() && args.num_nextn_predict_layers == 0
        }
        (Some(crate::configuration::SafetensorsModelConfig::QwenHybrid(args)), None)
        | (None, Some(crate::configuration::GgufModelConfig::QwenHybrid(args))) => {
            args.vision.is_none() && args.text.is_moe() && args.text.mtp_num_hidden_layers == 0
        }
        (Some(crate::configuration::SafetensorsModelConfig::DeepSeekV4(args)), None) => {
            args.num_nextn_predict_layers == 0 && plan.prediction_extension().is_none()
        }
        (Some(crate::configuration::SafetensorsModelConfig::DiffusionGemma(args)), None)
        | (None, Some(crate::configuration::GgufModelConfig::DiffusionGemma(args))) => {
            args.vision.is_none() && args.has_sparse_moe_layers()
        }
        _ => false,
    }
}

/// Returns whether the artifact is a DeepSeek-V4 ordinary target model.
///
/// A projected target may retain an exact additive prediction-extension
/// contract.  That contract does not change the target equations or routed
/// placement selected here.
pub fn is_supported_deepseek_v4_routed_partition(plan: &ArtifactArchitecturePlan) -> bool {
    matches!(
        plan.safetensors_architecture().map(|plan| plan.model()),
        Some(crate::configuration::SafetensorsModelConfig::DeepSeekV4(args))
            if args.num_nextn_predict_layers == 0
    )
}

/// Classifies an admitted routed partition before any payload is opened.
pub fn routed_decoder_partitioned_production_supported(
    inspection: &ArtifactInspection<ArtifactArchitecturePlan>,
    selected: &SelectedPartitionedAdmission<SelectedRoutedTextRealization, RoutedTextRequirements>,
) -> bool {
    if !is_supported_routed_decoder_partition(inspection.architecture_plan()) {
        return false;
    }
    let topology = selected.requirements().topology().topology();
    let residency_supported = routed_partitioned_residency_supported(
        selected.base().text().residency(),
        selected.base().bank_residency(),
    );
    topology.data() == 1
        && (topology.tensor() > 1 || topology.pipeline() > 1 || topology.expert() > 1)
        && residency_supported
}

/// Returns whether one admitted routed target has a complete neutral
/// production constructor, including architectures with distinct state types.
pub fn routed_partitioned_production_supported(
    inspection: &ArtifactInspection<ArtifactArchitecturePlan>,
    selected: &SelectedPartitionedAdmission<SelectedRoutedTextRealization, RoutedTextRequirements>,
) -> bool {
    routed_decoder_partitioned_production_supported(inspection, selected)
        || deepseek_v4_routed_partitioned_production_supported(inspection, selected)
}

/// Dispatches an admitted routed target to its architecture-owned equation and
/// state-class constructor without exposing that classification to a backend.
pub fn dispatch_routed_partitioned_production<T, R>(
    inspection: &ArtifactInspection<ArtifactArchitecturePlan>,
    selected: SelectedPartitionedAdmission<SelectedRoutedTextRealization, RoutedTextRequirements>,
    mechanisms: T,
    gated: impl FnOnce(
        T,
        &ArtifactInspection<ArtifactArchitecturePlan>,
        SelectedPartitionedAdmission<SelectedRoutedTextRealization, RoutedTextRequirements>,
    ) -> R,
    relu2: impl FnOnce(
        T,
        &ArtifactInspection<ArtifactArchitecturePlan>,
        SelectedPartitionedAdmission<SelectedRoutedTextRealization, RoutedTextRequirements>,
    ) -> R,
    pooling: impl FnOnce(
        T,
        &ArtifactInspection<ArtifactArchitecturePlan>,
        SelectedPartitionedAdmission<SelectedRoutedTextRealization, RoutedTextRequirements>,
    ) -> R,
) -> R {
    if deepseek_v4_routed_partitioned_production_supported(inspection, &selected) {
        pooling(mechanisms, inspection, selected)
    } else if selected
        .base()
        .bank(eredu_runtime::RoutedBankId::new(0))
        .expect("validated family bank")
        .plan()
        .relu2()
        .is_some()
    {
        relu2(mechanisms, inspection, selected)
    } else {
        gated(mechanisms, inspection, selected)
    }
}

fn routed_partitioned_residency_supported(
    ordinary: eredu_runtime::LayerWeightResidency,
    bank: eredu_runtime::ParameterBankResidency,
) -> bool {
    match bank {
        eredu_runtime::ParameterBankResidency::WithLayer
        | eredu_runtime::ParameterBankResidency::IndependentCache(_) => matches!(
            ordinary,
            eredu_runtime::LayerWeightResidency::FullyResident
                | eredu_runtime::LayerWeightResidency::LayerwiseHost(_)
                | eredu_runtime::LayerWeightResidency::DenseDiskStream(_)
        ),
        _ => false,
    }
}

/// Classifies an admitted DeepSeek-V4 target partition before payload access.
pub fn deepseek_v4_routed_partitioned_production_supported(
    inspection: &ArtifactInspection<ArtifactArchitecturePlan>,
    selected: &SelectedPartitionedAdmission<SelectedRoutedTextRealization, RoutedTextRequirements>,
) -> bool {
    if !is_supported_deepseek_v4_routed_partition(inspection.architecture_plan())
        || inspection.format() != eredu_core::ArtifactFormat::SafeTensors
    {
        return false;
    }
    let topology = selected.requirements().topology().topology();
    topology.data() == 1
        && (topology.tensor() > 1 || topology.pipeline() > 1 || topology.expert() > 1)
        && routed_partitioned_residency_supported(
            selected.base().text().residency(),
            selected.base().bank_residency(),
        )
}

/// Constructs an exact prediction-free gated-product partition and dispatches it.
pub fn visit_routed_partitioned_production<B, S, V>(
    inspection: &ArtifactInspection<ArtifactArchitecturePlan>,
    selected: SelectedPartitionedAdmission<SelectedRoutedTextRealization, RoutedTextRequirements>,
    store: eredu_checkpoint::store::SharedCheckpointSource,
    context: &<B::Tensor as eredu_nn::Tensor>::Context,
    visitor: V,
) -> Result<V::Output, DenseDecoderPartitionedDispatchError<V::Error>>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::BlockwiseAttentionBackend,
    S: eredu_runtime::LayerRuntimeState<B>,
    S::LayerState: eredu_nn::ReadOnlyAttentionCache<B::Tensor>
        + eredu_nn::CompressedAttentionCache<B::Tensor>
        + eredu_runtime::RuntimeStateComponents<B>,
    V: RoutedPartitionedProductionVisitor<B, S, crate::SelectedRoutedBank>,
{
    let expected = crate::routed_text::routed_text_requirements(inspection)
        .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    if &expected != selected.requirements().execution() {
        return Err(DenseDecoderPartitionedDispatchError::Architecture(
            "routed partition admission belongs to a different architecture or artifact".into(),
        ));
    }
    eredu_runtime::validate_checkpoint_handoff(
        expected.text(),
        store.as_ref(),
        eredu_runtime::SourceHandoffScope::Primary,
    )
    .map_err(|handoff| DenseDecoderPartitionedDispatchError::Architecture(handoff.to_string()))?;
    if let Some(config) = crate::routed_text::diffusion_args(inspection) {
        return diffusion::prepare_routed::<B, S, _>(
            config,
            selected,
            store,
            context,
            OrdinaryFamilyRoutedPartitionVisitor(visitor),
        );
    }
    match (
        inspection
            .architecture_plan()
            .safetensors_architecture()
            .map(|plan| plan.model()),
        inspection
            .architecture_plan()
            .gguf_plan()
            .map(|plan| plan.model()),
    ) {
        (Some(crate::configuration::SafetensorsModelConfig::QwenHybrid(args)), None)
        | (None, Some(crate::configuration::GgufModelConfig::QwenHybrid(args)))
            if args.vision.is_none()
                && args.text.is_moe()
                && args.text.mtp_num_hidden_layers == 0 =>
        {
            qwen_hybrid::prepare_routed::<B, S, _>(
                &args.text,
                selected,
                store,
                context,
                OrdinaryFamilyRoutedPartitionVisitor(visitor),
            )
        }
        (Some(crate::configuration::SafetensorsModelConfig::K2Horizon(args)), None)
        | (None, Some(crate::configuration::GgufModelConfig::K2Horizon(args)))
            if args.is_moe() =>
        {
            let args =
                crate::replicated_text::selected_k2_horizon_args(args, selected.base().text())
                    .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
            let declaration = inspection
                .architecture_plan()
                .k2_declaration(&args)
                .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
            let capability = crate::capability::k2_horizon(&declaration)
                .map_err(|e| DenseDecoderPartitionedDispatchError::Architecture(e.to_string()))?;
            prepare_routed_decoder_partition::<B, S, _, crate::k2_horizon::BlockFactory, V>(
                args.model_type.clone(),
                declaration.common().clone(),
                declaration.projections().cloned(),
                args,
                selected,
                store,
                context,
                visitor,
                capability,
            )
        }
        (Some(crate::configuration::SafetensorsModelConfig::Qwen(args)), None)
        | (None, Some(crate::configuration::GgufModelConfig::Qwen(args)))
            if args.is_moe() =>
        {
            let selected_args =
                crate::replicated_text::selected_qwen_args(args, selected.base().text())
                    .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
            let declaration = inspection
                .architecture_plan()
                .qwen_declaration(&selected_args)
                .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
            let capability = crate::capability::qwen(&declaration).map_err(|error| {
                DenseDecoderPartitionedDispatchError::Architecture(error.to_string())
            })?;
            prepare_routed_decoder_partition::<B, S, _, crate::qwen::RoutedQwenBlockFactory, V>(
                selected_args.model_type.clone(),
                declaration.common().clone(),
                declaration.projections().cloned(),
                selected_args,
                selected,
                store,
                context,
                visitor,
                capability,
            )
        }
        (Some(crate::configuration::SafetensorsModelConfig::GptOss(args)), None)
        | (None, Some(crate::configuration::GgufModelConfig::GptOss(args))) => {
            let selected_args =
                crate::replicated_text::selected_gpt_oss_args(args, selected.base().text())
                    .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
            let declaration = inspection
                .architecture_plan()
                .gpt_oss_declaration(&selected_args)
                .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
            let capability = crate::capability::gpt_oss(&declaration).map_err(|error| {
                DenseDecoderPartitionedDispatchError::Architecture(error.to_string())
            })?;
            prepare_routed_decoder_partition::<B, S, _, crate::gpt_oss::GptOssBlockFactory, V>(
                selected_args.model_type.clone(),
                declaration.common().clone(),
                Some(declaration.projections().clone()),
                selected_args,
                selected,
                store,
                context,
                visitor,
                capability,
            )
        }
        (Some(crate::configuration::SafetensorsModelConfig::Lfm2(args)), None)
        | (None, Some(crate::configuration::GgufModelConfig::Lfm2(args)))
            if args.has_sparse_moe_layers() =>
        {
            let selected_args =
                crate::replicated_text::selected_lfm2_args(args, selected.base().text())
                    .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
            let capability = crate::capability::lfm2(
                inspection
                    .architecture_plan()
                    .lfm2_declaration(&selected_args)
                    .map_err(DenseDecoderPartitionedDispatchError::Architecture)?
                    .as_ref(),
            )
            .map_err(|error| {
                DenseDecoderPartitionedDispatchError::Architecture(error.to_string())
            })?;
            prepare_lfm2_routed_partition::<B, S, V>(
                inspection.architecture_plan(),
                selected_args.model_type.clone(),
                selected_args,
                args.clone(),
                selected,
                store,
                context,
                visitor,
                capability,
            )
        }
        (Some(crate::configuration::SafetensorsModelConfig::KimiLinear(args)), None)
        | (None, Some(crate::configuration::GgufModelConfig::KimiLinear(args)))
            if args.has_sparse_moe_layers() && args.num_nextn_predict_layers == 0 =>
        {
            let selected_args =
                crate::replicated_text::selected_kimi_linear_args(args, selected.base().text())
                    .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
            let capability = crate::capability::kimi_linear(
                inspection
                    .architecture_plan()
                    .kimi_declaration(&selected_args)
                    .map_err(DenseDecoderPartitionedDispatchError::Architecture)?
                    .as_ref(),
            )
            .map_err(|error| {
                DenseDecoderPartitionedDispatchError::Architecture(error.to_string())
            })?;
            prepare_kimi_linear_routed_partition::<B, S, V>(
                inspection.architecture_plan(),
                selected_args.model_type.clone(),
                selected_args,
                args.clone(),
                selected,
                store,
                context,
                visitor,
                capability,
            )
        }
        (Some(crate::configuration::SafetensorsModelConfig::DeepSeekV3(args)), None)
        | (None, Some(crate::configuration::GgufModelConfig::DeepSeekV3(args)))
            if args.has_sparse_moe_layers() && args.num_nextn_predict_layers == 0 =>
        {
            let selected_args =
                crate::replicated_text::selected_deepseek_v3_args(args, selected.base().text())
                    .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
            let capability = crate::capability::deepseek_v3(&selected_args).map_err(|error| {
                DenseDecoderPartitionedDispatchError::Architecture(error.to_string())
            })?;
            prepare_deepseek_v3_routed_partition::<B, S, _>(
                selected_args.model_type.clone(),
                selected_args,
                selected,
                store,
                context,
                OrdinaryFamilyRoutedPartitionVisitor(visitor),
                capability,
            )
        }
        _ => Err(DenseDecoderPartitionedDispatchError::Architecture(
            "routed partition architecture has no gated-product partition constructor".into(),
        )),
    }
}

/// Constructs and visits one DeepSeek-V4 ordinary target partition.
pub fn visit_pooling_routed_partitioned_production<B, S, V>(
    inspection: &ArtifactInspection<ArtifactArchitecturePlan>,
    selected: SelectedPartitionedAdmission<SelectedRoutedTextRealization, RoutedTextRequirements>,
    store: eredu_checkpoint::store::SharedCheckpointSource,
    context: &<B::Tensor as eredu_nn::Tensor>::Context,
    visitor: V,
) -> Result<V::Output, DenseDecoderPartitionedDispatchError<V::Error>>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::HyperNeuralBackend,
    S: eredu_runtime::LayerRuntimeState<B>,
    S::LayerState: eredu_nn::PoolingAttentionCache<B::Tensor>,
    V: RoutedPartitionedProductionVisitor<B, S, crate::SelectedRoutedBank>,
{
    let expected = crate::routed_text::routed_text_requirements(inspection)
        .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    if &expected != selected.requirements().execution() {
        return Err(DenseDecoderPartitionedDispatchError::Architecture(
            "DeepSeek-V4 routed admission belongs to a different artifact".into(),
        ));
    }
    eredu_runtime::validate_checkpoint_handoff(
        expected.text(),
        store.as_ref(),
        eredu_runtime::SourceHandoffScope::Primary,
    )
    .map_err(|handoff| DenseDecoderPartitionedDispatchError::Architecture(handoff.to_string()))?;
    let Some(crate::configuration::SafetensorsModelConfig::DeepSeekV4(args)) = inspection
        .architecture_plan()
        .safetensors_architecture()
        .map(|plan| plan.model())
    else {
        return Err(DenseDecoderPartitionedDispatchError::Architecture(
            "DeepSeek-V4 routed production requires indexed SafeTensors".into(),
        ));
    };
    if args.num_nextn_predict_layers != 0 {
        return Err(DenseDecoderPartitionedDispatchError::Architecture(
            "DeepSeek-V4 routed target still contains prediction units".into(),
        ));
    }
    let selected_args =
        crate::replicated_text::selected_deepseek_v4_args(args, selected.base().text())
            .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
    let capability = crate::capability::deepseek_v4(&selected_args)
        .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    prepare_deepseek_v4_routed_partition::<B, S, _>(
        selected_args.model_type.clone(),
        selected_args,
        selected,
        store,
        context,
        OrdinaryFamilyRoutedPartitionVisitor(visitor),
        capability,
    )
}

/// Constructs a DeepSeek-V4 routed prediction target and pairs it before backend erasure.
pub fn visit_pooling_routed_partitioned_prediction_target_production<B, S, M, V>(
    inspection: &ArtifactInspection<ArtifactArchitecturePlan>,
    selected: SelectedPartitionedAdmission<SelectedRoutedTextRealization, RoutedTextRequirements>,
    extension: crate::prediction_extension::MaterializedPredictionExtension<B, M>,
    store: eredu_checkpoint::store::SharedCheckpointSource,
    context: &<B::Tensor as eredu_nn::Tensor>::Context,
    visitor: V,
) -> Result<V::Output, DenseDecoderPartitionedDispatchError<V::Error>>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::HyperNeuralBackend,
    S: eredu_runtime::LayerRuntimeState<B>,
    S::LayerState: eredu_nn::PoolingAttentionCache<B::Tensor>,
    M: crate::prediction_extension::PredictionExtensionMaterializer<B>,
    V: RoutedPartitionedPredictionTargetProductionVisitor<B, S, M, crate::SelectedRoutedBank>,
{
    let extension = <crate::deepseek::v4::Model<B> as crate::prediction_extension::MaterializedPredictionTarget<B>>::pair_prediction_extension(extension)
        .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    let expected = crate::routed_text::routed_text_requirements(inspection)
        .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    if &expected != selected.requirements().execution() {
        return Err(DenseDecoderPartitionedDispatchError::Architecture(
            "DeepSeek-V4 routed admission belongs to a different artifact".into(),
        ));
    }
    eredu_runtime::validate_checkpoint_handoff(
        expected.text(),
        store.as_ref(),
        eredu_runtime::SourceHandoffScope::Primary,
    )
    .map_err(|handoff| DenseDecoderPartitionedDispatchError::Architecture(handoff.to_string()))?;
    let Some(crate::configuration::SafetensorsModelConfig::DeepSeekV4(args)) = inspection
        .architecture_plan()
        .safetensors_architecture()
        .map(|plan| plan.model())
    else {
        return Err(DenseDecoderPartitionedDispatchError::Architecture(
            "DeepSeek-V4 routed production requires indexed SafeTensors".into(),
        ));
    };
    let selected_args =
        crate::replicated_text::selected_deepseek_v4_args(args, selected.base().text())
            .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
    let capability = crate::capability::deepseek_v4(&selected_args)
        .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    prepare_deepseek_v4_routed_partition::<B, S, _>(
        selected_args.model_type.clone(),
        selected_args,
        selected,
        store,
        context,
        PredictionFamilyRoutedPartitionVisitor::<B, crate::deepseek::v4::Model<B>, M, _> {
            extension,
            visitor,
            marker: PhantomData,
        },
        capability,
    )
}

/// Constructs a DeepSeek-V3 routed prediction target and pairs it before backend erasure.
pub fn visit_routed_partitioned_prediction_target_production<B, S, M, V>(
    inspection: &ArtifactInspection<ArtifactArchitecturePlan>,
    selected: SelectedPartitionedAdmission<SelectedRoutedTextRealization, RoutedTextRequirements>,
    extension: crate::prediction_extension::MaterializedPredictionExtension<B, M>,
    store: eredu_checkpoint::store::SharedCheckpointSource,
    context: &<B::Tensor as eredu_nn::Tensor>::Context,
    visitor: V,
) -> Result<V::Output, DenseDecoderPartitionedDispatchError<V::Error>>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::HyperNeuralBackend,
    S: eredu_runtime::LayerRuntimeState<B>,
    S::LayerState: eredu_nn::AttentionCache<B::Tensor>
        + eredu_nn::CompressedAttentionCache<B::Tensor>
        + eredu_runtime::RuntimeStateComponents<B>,
    M: crate::prediction_extension::PredictionExtensionMaterializer<B>,
    V: RoutedPartitionedPredictionTargetProductionVisitor<B, S, M, crate::SelectedRoutedBank>,
{
    if let Some(args) = match (
        inspection
            .architecture_plan()
            .safetensors_architecture()
            .map(|plan| plan.model()),
        inspection
            .architecture_plan()
            .gguf_plan()
            .map(|plan| plan.model()),
    ) {
        (Some(crate::configuration::SafetensorsModelConfig::QwenHybrid(args)), None)
        | (None, Some(crate::configuration::GgufModelConfig::QwenHybrid(args)))
            if args.vision.is_none() =>
        {
            Some(&args.text)
        }
        _ => None,
    } {
        let expected =
            crate::routed_text::routed_text_requirements(inspection).map_err(|error| {
                DenseDecoderPartitionedDispatchError::Architecture(error.to_string())
            })?;
        if &expected != selected.requirements().execution() {
            return Err(DenseDecoderPartitionedDispatchError::Architecture(
                "Qwen prediction target admission belongs to another artifact".into(),
            ));
        }
        eredu_runtime::validate_checkpoint_handoff(
            expected.text(),
            store.as_ref(),
            eredu_runtime::SourceHandoffScope::Primary,
        )
        .map_err(|handoff| {
            DenseDecoderPartitionedDispatchError::Architecture(handoff.to_string())
        })?;
        let extension = <crate::qwen::hybrid::LayeredModel<B> as crate::prediction_extension::MaterializedPredictionTarget<B>>::pair_prediction_extension(extension)
            .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
        return qwen_hybrid::prepare_routed::<B, S, _>(
            args,
            selected,
            store,
            context,
            PredictionFamilyRoutedPartitionVisitor::<B, crate::qwen::hybrid::LayeredModel<B>, M, _> {
                extension,
                visitor,
                marker: PhantomData,
            },
        );
    }

    let extension = <crate::deepseek::v3::Model<B> as crate::prediction_extension::MaterializedPredictionTarget<B>>::pair_prediction_extension(extension)
        .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    let expected = crate::routed_text::routed_text_requirements(inspection)
        .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    if &expected != selected.requirements().execution() {
        return Err(DenseDecoderPartitionedDispatchError::Architecture(
            "DeepSeek-V3 routed admission belongs to a different artifact".into(),
        ));
    }
    eredu_runtime::validate_checkpoint_handoff(
        expected.text(),
        store.as_ref(),
        eredu_runtime::SourceHandoffScope::Primary,
    )
    .map_err(|handoff| DenseDecoderPartitionedDispatchError::Architecture(handoff.to_string()))?;
    let args = match (
        inspection
            .architecture_plan()
            .safetensors_architecture()
            .map(|plan| plan.model()),
        inspection
            .architecture_plan()
            .gguf_plan()
            .map(|plan| plan.model()),
    ) {
        (Some(crate::configuration::SafetensorsModelConfig::DeepSeekV3(args)), None)
        | (None, Some(crate::configuration::GgufModelConfig::DeepSeekV3(args))) => args,
        _ => {
            return Err(DenseDecoderPartitionedDispatchError::Architecture(
                "routed prediction target requires DeepSeek-V3".into(),
            ));
        }
    };
    let selected_args =
        crate::replicated_text::selected_deepseek_v3_args(args, selected.base().text())
            .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
    let capability = crate::capability::deepseek_v3(&selected_args)
        .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    prepare_deepseek_v3_routed_partition::<B, S, _>(
        selected_args.model_type.clone(),
        selected_args,
        selected,
        store,
        context,
        PredictionFamilyRoutedPartitionVisitor::<B, crate::deepseek::v3::Model<B>, M, _> {
            extension,
            visitor,
            marker: PhantomData,
        },
        capability,
    )
}

/// Constructs and visits one exact prediction-free Nemotron-H ReLU-squared partition.
///
/// This equation-specific entry remains separate from the gated-product public
/// route until a backend implements the same family-blind visitor for both
/// grouped equations.
pub fn visit_relu2_routed_partitioned_production<B, S, V>(
    inspection: &ArtifactInspection<ArtifactArchitecturePlan>,
    selected: SelectedPartitionedAdmission<SelectedRoutedTextRealization, RoutedTextRequirements>,
    store: eredu_checkpoint::store::SharedCheckpointSource,
    context: &<B::Tensor as eredu_nn::Tensor>::Context,
    visitor: V,
) -> Result<V::Output, DenseDecoderPartitionedDispatchError<V::Error>>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: eredu_runtime::LayerRuntimeState<B>,
    S::LayerState: eredu_nn::AttentionCache<B::Tensor> + eredu_runtime::RuntimeStateComponents<B>,
    V: RoutedPartitionedProductionVisitor<B, S, crate::SelectedRoutedBank>,
{
    let expected = crate::routed_text::routed_text_requirements(inspection)
        .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    if &expected != selected.requirements().execution() {
        return Err(DenseDecoderPartitionedDispatchError::Architecture(
            "ReLU-squared routed admission belongs to a different architecture or artifact".into(),
        ));
    }
    eredu_runtime::validate_checkpoint_handoff(
        expected.text(),
        store.as_ref(),
        eredu_runtime::SourceHandoffScope::Primary,
    )
    .map_err(|handoff| DenseDecoderPartitionedDispatchError::Architecture(handoff.to_string()))?;
    let args = match (
        inspection
            .architecture_plan()
            .safetensors_architecture()
            .map(|plan| plan.model()),
        inspection
            .architecture_plan()
            .gguf_plan()
            .map(|plan| plan.model()),
    ) {
        (Some(crate::configuration::SafetensorsModelConfig::NemotronH(args)), None)
        | (None, Some(crate::configuration::GgufModelConfig::NemotronH(args))) => args,
        _ => {
            return Err(DenseDecoderPartitionedDispatchError::Architecture(
                "ReLU-squared partition production requires Nemotron-H".into(),
            ));
        }
    };
    if args.num_nextn_predict_layers != 0 || !args.has_sparse_moe_layers() {
        return Err(DenseDecoderPartitionedDispatchError::Architecture(
            "ReLU-squared partition production requires prediction-free sparse Nemotron-H".into(),
        ));
    }
    if selected
        .base()
        .bank(eredu_runtime::RoutedBankId::new(0))
        .expect("validated family bank")
        .plan()
        .relu2()
        .is_none()
    {
        return Err(DenseDecoderPartitionedDispatchError::Architecture(
            "Nemotron-H selected a non-ReLU-squared expert equation".into(),
        ));
    }
    let selected_args =
        crate::replicated_text::selected_nemotron_h_args(args, selected.base().text())
            .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
    let capability_estimate = crate::capability::nemotron_h(
        inspection
            .architecture_plan()
            .nemotron_declaration(&selected_args)
            .map_err(DenseDecoderPartitionedDispatchError::Architecture)?
            .as_ref(),
    )
    .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    prepare_nemotron_h_routed_partition::<B, S, _>(
        inspection.architecture_plan(),
        selected_args.model_type.clone(),
        selected_args,
        args.clone(),
        selected,
        store,
        context,
        OrdinaryFamilyRoutedPartitionVisitor(visitor),
        capability_estimate,
    )
}

/// Constructs a routed ReLU-squared prediction target and pairs it before backend erasure.
pub fn visit_relu2_routed_partitioned_prediction_target_production<B, S, M, V>(
    inspection: &ArtifactInspection<ArtifactArchitecturePlan>,
    selected: SelectedPartitionedAdmission<SelectedRoutedTextRealization, RoutedTextRequirements>,
    extension: crate::prediction_extension::MaterializedPredictionExtension<B, M>,
    store: eredu_checkpoint::store::SharedCheckpointSource,
    context: &<B::Tensor as eredu_nn::Tensor>::Context,
    visitor: V,
) -> Result<V::Output, DenseDecoderPartitionedDispatchError<V::Error>>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::HyperNeuralBackend,
    S: eredu_runtime::LayerRuntimeState<B>,
    S::LayerState: eredu_nn::AttentionCache<B::Tensor> + eredu_runtime::RuntimeStateComponents<B>,
    M: crate::prediction_extension::PredictionExtensionMaterializer<B>,
    V: RoutedPartitionedPredictionTargetProductionVisitor<B, S, M, crate::SelectedRoutedBank>,
{
    let extension = <crate::nemotron_h::PartitionedLayeredModel<B> as crate::prediction_extension::MaterializedPredictionTarget<B>>::pair_prediction_extension(extension)
        .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    let expected = crate::routed_text::routed_text_requirements(inspection)
        .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    if &expected != selected.requirements().execution() {
        return Err(DenseDecoderPartitionedDispatchError::Architecture(
            "ReLU-squared routed prediction admission belongs to a different artifact".into(),
        ));
    }
    eredu_runtime::validate_checkpoint_handoff(
        expected.text(),
        store.as_ref(),
        eredu_runtime::SourceHandoffScope::Primary,
    )
    .map_err(|handoff| DenseDecoderPartitionedDispatchError::Architecture(handoff.to_string()))?;
    let args = match (
        inspection
            .architecture_plan()
            .safetensors_architecture()
            .map(|plan| plan.model()),
        inspection
            .architecture_plan()
            .gguf_plan()
            .map(|plan| plan.model()),
    ) {
        (Some(crate::configuration::SafetensorsModelConfig::NemotronH(args)), None)
        | (None, Some(crate::configuration::GgufModelConfig::NemotronH(args))) => args,
        _ => {
            return Err(DenseDecoderPartitionedDispatchError::Architecture(
                "ReLU-squared routed prediction requires Nemotron-H".into(),
            ));
        }
    };
    // The retained target inspection excludes prediction units; the paired
    // extension above owns their depth, parameters and mutable state.
    if !args.has_sparse_moe_layers() {
        return Err(DenseDecoderPartitionedDispatchError::Architecture(
            "ReLU-squared routed prediction requires a sparse Nemotron-H target".into(),
        ));
    }
    let selected_args =
        crate::replicated_text::selected_nemotron_h_args(args, selected.base().text())
            .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
    let capability_estimate = crate::capability::nemotron_h(
        inspection
            .architecture_plan()
            .nemotron_declaration(&selected_args)
            .map_err(DenseDecoderPartitionedDispatchError::Architecture)?
            .as_ref(),
    )
    .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    prepare_nemotron_h_routed_partition::<B, S, _>(
        inspection.architecture_plan(),
        selected_args.model_type.clone(),
        selected_args,
        args.clone(),
        selected,
        store,
        context,
        PredictionFamilyRoutedPartitionVisitor::<
            B,
            crate::nemotron_h::PartitionedLayeredModel<B>,
            M,
            _,
        > {
            extension,
            visitor,
            marker: PhantomData,
        },
        capability_estimate,
    )
}

#[allow(clippy::too_many_arguments)]
fn prepare_routed_decoder_partition<B, S, C, P, V>(
    effective_model_type: String,
    common_declaration: std::sync::Arc<crate::decoder::DecoderDeclaration>,
    projection_declaration: Option<crate::decoder::ProjectionDeclarations<P::Parameters>>,
    selected_args: C,
    selected: SelectedPartitionedAdmission<SelectedRoutedTextRealization, RoutedTextRequirements>,
    store: eredu_checkpoint::store::SharedCheckpointSource,
    context: &<B::Tensor as eredu_nn::Tensor>::Context,
    visitor: V,
    capability_estimate: crate::capability::CapabilityEstimate,
) -> Result<V::Output, DenseDecoderPartitionedDispatchError<V::Error>>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: eredu_runtime::LayerRuntimeState<B>,
    S::LayerState: eredu_nn::AttentionCache<B::Tensor> + eredu_runtime::RuntimeStateComponents<B>,
    C: RoutedPartitionedConfig<P::Parameters>,
    P: crate::decoder::BlockFactory<B, C>,
    P::FeedForward: crate::decoder::TensorParallelRoutedProjectionOperator<B>,
    V: RoutedPartitionedProductionVisitor<B, S, crate::SelectedRoutedBank>,
{
    let definition = crate::replicated_text::executable_component(selected.base().text())
        .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
    let parameters = definition.parameters();
    let rank = selected.requirements().topology();
    let layout = derive_partitioned_local_layout(parameters, rank)
        .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
    let [owned] = selected.requirements().groups() else {
        return Err(DenseDecoderPartitionedDispatchError::Architecture(
            "routed decoder must own exactly one execution group".into(),
        ));
    };
    if owned.group().as_str() != crate::decoder::TEXT_DECODER_EXECUTION_GROUP {
        return Err(DenseDecoderPartitionedDispatchError::Architecture(
            "routed decoder admission names a different execution group".into(),
        ));
    }
    let plans = selected_args
        .partition_expert_plans(
            Some(&common_declaration),
            projection_declaration.as_ref(),
            &layout,
            rank,
        )
        .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    if plans.keys().ne(selected.base().banks().keys()) {
        return Err(DenseDecoderPartitionedDispatchError::Architecture(
            "partition bank identities differ from admission".into(),
        ));
    }
    let geometry = selected_args
        .routed_partition_geometry(
            Some(&common_declaration),
            projection_declaration.as_ref(),
            &layout,
            owned.units(),
            rank,
            &plans,
        )
        .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    let complete_state = geometry.complete_state_layout().clone();
    let state_plan = definition.state_partition_plan();
    let partition = selected
        .requirements()
        .construct_partition(
            parameters,
            &complete_state,
            state_plan,
            geometry,
            eredu_runtime::NoAuxiliaryBoundarySchema::new(selected_args.hidden_size()),
        )
        .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    let mut tasks = eredu_runtime::partition_selected_replicated_text_materialization_tasks(
        selected.materialization_tasks(),
        parameters,
        &partition,
    )
    .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    let local_units = owned.units().collect::<BTreeSet<_>>();
    let mut banks = BTreeMap::new();
    for (id, plan) in plans {
        let selected_bank = &selected.base().banks()[&id];
        let mut members = crate::routed_text::project_addressable_members_with_tasks(
            selected_bank.catalog(),
            selected.base().text(),
            &tasks,
        )
        .map_err(|e| DenseDecoderPartitionedDispatchError::Architecture(e.to_string()))?;
        let local_experts = plan.local_global_group_indices();
        members.retain(|member| {
            local_units.contains(&member.key().unit())
                && local_experts.contains(&member.key().member())
        });
        let members = members
            .into_iter()
            .map(|member| member.with_owner_rank(rank.global_rank()))
            .collect();
        banks.insert(
            id,
            selected_bank
                .with_partition_geometry(plan, selected_bank.catalog().clone(), members, &layout)
                .map_err(|error| {
                    DenseDecoderPartitionedDispatchError::Architecture(error.to_string())
                })?,
        );
    }
    let bank_residency = selected.base().bank_residency();
    if matches!(
        bank_residency,
        eredu_runtime::ParameterBankResidency::IndependentCache(_)
    ) {
        let targets = banks
            .values()
            .flat_map(|bank| bank.catalog().logical_targets())
            .collect::<BTreeSet<_>>();
        tasks.retain(|task| !targets.contains(task.name()));
    }
    let architecture = crate::decoder::DecoderKernel::<B, C, P>::from_partition(
        selected_args,
        definition,
        common_declaration.as_ref(),
        projection_declaration.as_ref(),
        &partition,
        context,
    )
    .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    let prepared = prepare_partitioned::<B, S, _, _, _, _, _>(architecture, selected, partition)
        .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
    let execution = prepare_routed_execution::<B, S, _, _, _>(&prepared, &banks)
        .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
    visitor
        .visit(
            AdmittedRoutedPartition {
                prepared,
                source_architecture: None,
                layout,
                provider_layout: None,
                tasks,
                capability_estimate,
                effective_model_type,
                bank_residency,
                banks,
                rows: None,
                execution,
                backend: PhantomData,
            }
            .into_binding(),
            store,
        )
        .map_err(DenseDecoderPartitionedDispatchError::Visitor)
}

#[allow(clippy::too_many_arguments)]
fn prepare_lfm2_routed_partition<B, S, V>(
    architecture: &ArtifactArchitecturePlan,
    effective_model_type: String,
    selected_args: crate::lfm2::ModelArgs,
    source_args: crate::lfm2::ModelArgs,
    selected: SelectedPartitionedAdmission<SelectedRoutedTextRealization, RoutedTextRequirements>,
    store: eredu_checkpoint::store::SharedCheckpointSource,
    context: &<B::Tensor as eredu_nn::Tensor>::Context,
    visitor: V,
    capability_estimate: crate::capability::CapabilityEstimate,
) -> Result<V::Output, DenseDecoderPartitionedDispatchError<V::Error>>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: eredu_runtime::LayerRuntimeState<B>,
    S::LayerState: eredu_nn::AttentionCache<B::Tensor> + eredu_runtime::RuntimeStateComponents<B>,
    V: RoutedPartitionedProductionVisitor<B, S, crate::SelectedRoutedBank>,
{
    let retained = selected
        .base()
        .text()
        .requirements()
        .shared_component_definition()
        .ok_or_else(|| {
            DenseDecoderPartitionedDispatchError::Architecture(
                "LFM2 selection has no retained definition".into(),
            )
        })?;
    let definition = crate::replicated_text::executable_component(selected.base().text())
        .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
    let parameters = definition.parameters().clone();
    let rank = selected.requirements().topology();
    let layout = derive_partitioned_local_layout(&parameters, rank)
        .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
    let [owned] = selected.requirements().groups() else {
        return Err(DenseDecoderPartitionedDispatchError::Architecture(
            "routed LFM2 must own exactly one execution group".into(),
        ));
    };
    if owned.group().as_str() != crate::decoder::TARGET_EXECUTION_GROUP {
        return Err(DenseDecoderPartitionedDispatchError::Architecture(
            "routed LFM2 admission names a different execution group".into(),
        ));
    }
    let declaration = architecture
        .lfm2_declaration(&selected_args)
        .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
    let plan_geometry = crate::lfm2::local_geometry(&declaration, &layout)
        .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    let plan = crate::lfm2::expert_realization_plan(&declaration, Some(&plan_geometry), rank)
        .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?
        .ok_or_else(|| {
            DenseDecoderPartitionedDispatchError::Architecture(
                "routed LFM2 has no selected expert realization".into(),
            )
        })?;
    let geometry = crate::lfm2::partition_local_routed_geometry(
        &declaration,
        &layout,
        owned.units(),
        rank,
        &plan,
    )
    .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    let complete_state = geometry.complete_state_layout().clone();
    let state_plan = crate::transport::pipeline_state(0, &complete_state);
    let partition = selected
        .requirements()
        .construct_partition(
            &parameters,
            &complete_state,
            &state_plan,
            geometry,
            eredu_runtime::NoAuxiliaryBoundarySchema::new(selected_args.hidden_size),
        )
        .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    let source_architecture = if selected.base().text().uses_weight_transform() {
        let error = |error: String| DenseDecoderPartitionedDispatchError::Architecture(error);
        let source_args = crate::lfm2::with_checkpoint_formats(
            &source_args,
            selected
                .base()
                .text()
                .requirements()
                .native_quantized_formats(),
        )
        .map_err(error)?;
        let source_declaration = architecture
            .lfm2_declaration(&source_args)
            .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
        let source_definition = retained
            .project_parameters(source_declaration.definition().parameters().clone())
            .map_err(|e| error(e.to_string()))?;
        let source_parameters = source_definition.parameters().clone();
        let source_layout =
            derive_partitioned_transform_source_layout(&source_parameters, &parameters, rank)
                .map_err(|failure| error(failure.to_string()))?;
        let source_local = crate::lfm2::local_geometry(&source_declaration, &source_layout)
            .map_err(|e| error(e.to_string()))?;
        if source_local.state_layout() != &complete_state {
            return Err(error(
                "routed LFM2 transform source changed local state geometry".into(),
            ));
        }
        let source_plan =
            crate::lfm2::expert_realization_plan(&source_declaration, Some(&source_local), rank)
                .map_err(|e| error(e.to_string()))?
                .ok_or_else(|| {
                    error("routed LFM2 transform source has no expert realization".into())
                })?;
        let source_geometry = crate::lfm2::partition_local_routed_geometry(
            &source_declaration,
            &source_layout,
            owned.units(),
            rank,
            &source_plan,
        )
        .map_err(|e| error(e.to_string()))?;
        let source_partition = selected
            .requirements()
            .construct_partition(
                &source_parameters,
                &complete_state,
                &state_plan,
                source_geometry,
                eredu_runtime::NoAuxiliaryBoundarySchema::new(source_args.hidden_size),
            )
            .map_err(|e| error(e.to_string()))?;
        let source = crate::lfm2::Lfm2Kernel::<B>::from_partition(
            source_declaration,
            source_definition,
            &source_partition,
            context,
        )
        .map_err(|e| error(e.to_string()))?;
        Some((source, source_layout))
    } else {
        None
    };
    prepare_family_routed_partition::<B, S, _, _, _, _>(
        crate::lfm2::Lfm2Kernel::<B>::from_partition(declaration, definition, &partition, context)
            .map_err(|error| {
                DenseDecoderPartitionedDispatchError::Architecture(error.to_string())
            })?,
        source_architecture,
        selected,
        partition,
        parameters,
        layout,
        None,
        plan,
        store,
        OrdinaryFamilyRoutedPartitionVisitor(visitor),
        capability_estimate,
        effective_model_type,
    )
}

#[allow(clippy::too_many_arguments)]
fn prepare_kimi_linear_routed_partition<B, S, V>(
    architecture: &ArtifactArchitecturePlan,
    effective_model_type: String,
    selected_args: crate::kimi_linear::ModelArgs,
    source_args: crate::kimi_linear::ModelArgs,
    selected: SelectedPartitionedAdmission<SelectedRoutedTextRealization, RoutedTextRequirements>,
    store: eredu_checkpoint::store::SharedCheckpointSource,
    context: &<B::Tensor as eredu_nn::Tensor>::Context,
    visitor: V,
    capability_estimate: crate::capability::CapabilityEstimate,
) -> Result<V::Output, DenseDecoderPartitionedDispatchError<V::Error>>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::BlockwiseAttentionBackend,
    S: eredu_runtime::LayerRuntimeState<B>,
    S::LayerState:
        eredu_nn::CompressedAttentionCache<B::Tensor> + eredu_runtime::RuntimeStateComponents<B>,
    V: RoutedPartitionedProductionVisitor<B, S, crate::SelectedRoutedBank>,
{
    let retained = selected
        .base()
        .text()
        .requirements()
        .shared_component_definition()
        .ok_or_else(|| {
            DenseDecoderPartitionedDispatchError::Architecture(
                "Kimi Linear selection has no retained definition".into(),
            )
        })?;
    let definition = crate::replicated_text::executable_component(selected.base().text())
        .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
    let parameters = definition.parameters().clone();
    let rank = selected.requirements().topology();
    let layout = derive_partitioned_local_layout(&parameters, rank)
        .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
    let [owned] = selected.requirements().groups() else {
        return Err(DenseDecoderPartitionedDispatchError::Architecture(
            "routed Kimi Linear must own exactly one execution group".into(),
        ));
    };
    if owned.group().as_str() != crate::decoder::TARGET_EXECUTION_GROUP {
        return Err(DenseDecoderPartitionedDispatchError::Architecture(
            "routed Kimi Linear admission names a different execution group".into(),
        ));
    }
    let declaration = architecture
        .kimi_declaration(&selected_args)
        .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
    let plan_geometry = crate::kimi_linear::local_geometry(&declaration, &layout)
        .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    let plan =
        crate::kimi_linear::expert_realization_plan(&declaration, Some(&plan_geometry), rank)
            .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?
            .ok_or_else(|| {
                DenseDecoderPartitionedDispatchError::Architecture(
                    "routed Kimi Linear has no selected expert realization".into(),
                )
            })?;
    let geometry = crate::kimi_linear::partition_local_routed_geometry(
        &declaration,
        &layout,
        owned.units(),
        rank,
        &plan,
    )
    .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    let complete_state = geometry.complete_state_layout().clone();
    let state_plan = crate::transport::pipeline_state(0, &complete_state);
    let partition = selected
        .requirements()
        .construct_partition(
            &parameters,
            &complete_state,
            &state_plan,
            geometry,
            eredu_runtime::NoAuxiliaryBoundarySchema::new(selected_args.hidden_size),
        )
        .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    let source_architecture = if selected.base().text().uses_weight_transform() {
        let error = |error: String| DenseDecoderPartitionedDispatchError::Architecture(error);
        let source_args = crate::kimi_linear::with_checkpoint_formats(
            &source_args,
            selected
                .base()
                .text()
                .requirements()
                .native_quantized_formats(),
        )
        .map_err(error)?;
        let source_declaration = architecture
            .kimi_declaration(&source_args)
            .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
        let source_definition = retained
            .project_parameters(source_declaration.definition().parameters().clone())
            .map_err(|e| error(e.to_string()))?;
        let source_parameters = source_definition.parameters().clone();
        let source_layout =
            derive_partitioned_transform_source_layout(&source_parameters, &parameters, rank)
                .map_err(|failure| error(failure.to_string()))?;
        let source_local = crate::kimi_linear::local_geometry(&source_declaration, &source_layout)
            .map_err(|e| error(e.to_string()))?;
        if source_local.state_layout() != &complete_state {
            return Err(error(
                "routed Kimi Linear transform source changed local state geometry".into(),
            ));
        }
        let source_plan = crate::kimi_linear::expert_realization_plan(
            &source_declaration,
            Some(&source_local),
            rank,
        )
        .map_err(|e| error(e.to_string()))?
        .ok_or_else(|| {
            error("routed Kimi Linear transform source has no expert realization".into())
        })?;
        let source_geometry = crate::kimi_linear::partition_local_routed_geometry(
            &source_declaration,
            &source_layout,
            owned.units(),
            rank,
            &source_plan,
        )
        .map_err(|e| error(e.to_string()))?;
        let source_partition = selected
            .requirements()
            .construct_partition(
                &source_parameters,
                &complete_state,
                &state_plan,
                source_geometry,
                eredu_runtime::NoAuxiliaryBoundarySchema::new(source_args.hidden_size),
            )
            .map_err(|e| error(e.to_string()))?;
        let source = crate::kimi_linear::KimiKernel::<B>::from_partition(
            source_declaration,
            source_definition,
            &source_partition,
            context,
        )
        .map_err(|e| error(e.to_string()))?;
        Some((source, source_layout))
    } else {
        None
    };
    prepare_family_routed_partition::<B, S, _, _, _, _>(
        crate::kimi_linear::KimiKernel::<B>::from_partition(
            declaration,
            definition,
            &partition,
            context,
        )
        .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?,
        source_architecture,
        selected,
        partition,
        parameters,
        layout,
        None,
        plan,
        store,
        OrdinaryFamilyRoutedPartitionVisitor(visitor),
        capability_estimate,
        effective_model_type,
    )
}

#[allow(clippy::too_many_arguments)]
fn prepare_nemotron_h_routed_partition<B, S, V>(
    architecture: &ArtifactArchitecturePlan,
    effective_model_type: String,
    selected_args: crate::nemotron_h::ModelArgs,
    source_args: crate::nemotron_h::ModelArgs,
    selected: SelectedPartitionedAdmission<SelectedRoutedTextRealization, RoutedTextRequirements>,
    store: eredu_checkpoint::store::SharedCheckpointSource,
    context: &<B::Tensor as eredu_nn::Tensor>::Context,
    visitor: V,
    capability_estimate: crate::capability::CapabilityEstimate,
) -> Result<V::Output, DenseDecoderPartitionedDispatchError<V::Error>>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: eredu_runtime::LayerRuntimeState<B>,
    S::LayerState: eredu_nn::AttentionCache<B::Tensor> + eredu_runtime::RuntimeStateComponents<B>,
    V: FamilyRoutedPartitionVisitor<B, S, crate::nemotron_h::PartitionedLayeredModel<B>>,
{
    let retained = selected
        .base()
        .text()
        .requirements()
        .shared_component_definition()
        .ok_or_else(|| {
            DenseDecoderPartitionedDispatchError::Architecture(
                "Nemotron-H selection has no retained definition".into(),
            )
        })?;
    let definition = crate::replicated_text::executable_component(selected.base().text())
        .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
    let parameters = definition.parameters().clone();
    let rank = selected.requirements().topology();
    let layout = derive_partitioned_local_layout(&parameters, rank)
        .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
    let [owned] = selected.requirements().groups() else {
        return Err(DenseDecoderPartitionedDispatchError::Architecture(
            "routed Nemotron-H must own exactly one execution group".into(),
        ));
    };
    if owned.group().as_str() != crate::decoder::TARGET_EXECUTION_GROUP {
        return Err(DenseDecoderPartitionedDispatchError::Architecture(
            "routed Nemotron-H admission names a different execution group".into(),
        ));
    }
    let declaration = architecture
        .nemotron_declaration(&selected_args)
        .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
    let plan_geometry = crate::nemotron_h::local_geometry(&declaration, &layout)
        .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    let plan = crate::nemotron_h::expert_realization_plan(&declaration, Some(&plan_geometry), rank)
        .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?
        .ok_or_else(|| {
            DenseDecoderPartitionedDispatchError::Architecture(
                "routed Nemotron-H has no selected expert realization".into(),
            )
        })?;
    let geometry = crate::nemotron_h::partition_local_routed_geometry(
        &declaration,
        &layout,
        owned.units(),
        rank,
        &plan,
    )
    .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    let complete_state = geometry.complete_state_layout().clone();
    let state_plan = crate::transport::pipeline_state(0, &complete_state);
    let partition = selected
        .requirements()
        .construct_partition(
            &parameters,
            &complete_state,
            &state_plan,
            geometry,
            crate::nemotron_h::TargetBoundarySchema::from_declaration(&declaration),
        )
        .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    let source_architecture = if selected.base().text().uses_weight_transform() {
        let error = |error: String| DenseDecoderPartitionedDispatchError::Architecture(error);
        let source_declaration = architecture
            .nemotron_declaration(&source_args)
            .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
        let source_definition = retained
            .project_parameters(source_declaration.definition().parameters().clone())
            .map_err(|e| error(e.to_string()))?;
        let source_parameters = source_definition.parameters().clone();
        let source_layout =
            derive_partitioned_transform_source_layout(&source_parameters, &parameters, rank)
                .map_err(|failure| error(failure.to_string()))?;
        let source_local = crate::nemotron_h::local_geometry(&source_declaration, &source_layout)
            .map_err(|e| error(e.to_string()))?;
        if source_local.state_layout() != &complete_state {
            return Err(error(
                "routed Nemotron transform source changed local state geometry".into(),
            ));
        }
        let source_plan = crate::nemotron_h::expert_realization_plan(
            &source_declaration,
            Some(&source_local),
            rank,
        )
        .map_err(|e| error(e.to_string()))?
        .ok_or_else(|| {
            error("routed Nemotron transform source has no expert realization".into())
        })?;
        let source_geometry = crate::nemotron_h::partition_local_routed_geometry(
            &source_declaration,
            &source_layout,
            owned.units(),
            rank,
            &source_plan,
        )
        .map_err(|e| error(e.to_string()))?;
        let source_partition = selected
            .requirements()
            .construct_partition(
                &source_parameters,
                &complete_state,
                &state_plan,
                source_geometry,
                crate::nemotron_h::TargetBoundarySchema::from_declaration(&source_declaration),
            )
            .map_err(|e| error(e.to_string()))?;
        let source = crate::nemotron_h::NemotronKernel::<B>::from_partition(
            source_declaration,
            source_definition,
            &source_partition,
            context,
        )
        .map_err(|e| error(e.to_string()))?;
        Some((source, source_layout))
    } else {
        None
    };
    prepare_family_routed_partition::<B, S, _, _, _, _>(
        crate::nemotron_h::NemotronKernel::<B>::from_partition(
            declaration,
            definition,
            &partition,
            context,
        )
        .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?,
        source_architecture,
        selected,
        partition,
        parameters,
        layout,
        None,
        plan,
        store,
        visitor,
        capability_estimate,
        effective_model_type,
    )
}

#[allow(clippy::too_many_arguments)]
fn prepare_deepseek_v3_routed_partition<B, S, V>(
    effective_model_type: String,
    selected_args: crate::deepseek::V3Args,
    selected: SelectedPartitionedAdmission<SelectedRoutedTextRealization, RoutedTextRequirements>,
    store: eredu_checkpoint::store::SharedCheckpointSource,
    context: &<B::Tensor as eredu_nn::Tensor>::Context,
    visitor: V,
    capability_estimate: crate::capability::CapabilityEstimate,
) -> Result<V::Output, DenseDecoderPartitionedDispatchError<V::Error>>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::BlockwiseAttentionBackend,
    S: eredu_runtime::LayerRuntimeState<B>,
    S::LayerState:
        eredu_nn::CompressedAttentionCache<B::Tensor> + eredu_runtime::RuntimeStateComponents<B>,
    V: FamilyRoutedPartitionVisitor<B, S, crate::deepseek::v3::Model<B>>,
{
    if selected_args.num_nextn_predict_layers != 0 || !selected_args.has_sparse_moe_layers() {
        return Err(DenseDecoderPartitionedDispatchError::Architecture(
            "DeepSeek-V3 routed partition requires sparse target units without prediction".into(),
        ));
    }
    if selected
        .base()
        .bank(eredu_runtime::RoutedBankId::new(0))
        .expect("validated family bank")
        .plan()
        .gated()
        .is_none()
    {
        return Err(DenseDecoderPartitionedDispatchError::Architecture(
            "DeepSeek-V3 selected a non-gated expert equation".into(),
        ));
    }
    let retained = selected
        .base()
        .text()
        .requirements()
        .shared_component_definition()
        .ok_or_else(|| {
            DenseDecoderPartitionedDispatchError::Architecture(
                "V3 selection has no retained definition".into(),
            )
        })?;
    let definition = crate::replicated_text::executable_component(selected.base().text())
        .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
    let parameters = definition.parameters().clone();
    let rank = selected.requirements().topology();
    let layout = derive_partitioned_local_layout(&parameters, rank)
        .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
    let [owned] = selected.requirements().groups() else {
        return Err(DenseDecoderPartitionedDispatchError::Architecture(
            "prediction-free DeepSeek-V3 must own exactly one target group".into(),
        ));
    };
    if owned.group().as_str() != crate::decoder::TARGET_EXECUTION_GROUP {
        return Err(DenseDecoderPartitionedDispatchError::Architecture(
            "DeepSeek-V3 admission names a different execution group".into(),
        ));
    }
    let local = crate::deepseek::parallel::v3_local_geometry(&selected_args, &layout)
        .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    let plan = crate::deepseek::v3_expert_realization_plan(&selected_args, Some(&local), rank)
        .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?
        .ok_or_else(|| {
            DenseDecoderPartitionedDispatchError::Architecture(
                "V3 partition has no routed units".into(),
            )
        })?;
    let geometry = crate::deepseek::v3_partition_local_geometry(
        &selected_args,
        &layout,
        [(owned.group().as_str(), owned.units())],
        selected.requirements().ownership(),
        rank,
        &plan,
    )
    .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    let complete_state = local.state_layout().clone();
    let state_plan = crate::transport::pipeline_state(0, &complete_state);
    let partition = selected
        .requirements()
        .construct_partition(
            &parameters,
            &complete_state,
            &state_plan,
            geometry,
            crate::deepseek::v3::TargetBoundarySchema::from_args(&selected_args),
        )
        .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    crate::deepseek::V3PartitionLocalFoundation::from_partition(&selected_args, &partition)
        .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    let source_architecture = if selected.base().text().uses_weight_transform() {
        let error = |error: String| DenseDecoderPartitionedDispatchError::Architecture(error);
        let source_args =
            crate::replicated_text::source_deepseek_v3_args(&selected_args, selected.base().text())
                .map_err(error)?;
        let source_definition = crate::deepseek::v3::project_parameters(&source_args, &retained)
            .map_err(|e| error(e.to_string()))?;
        let source_parameters = source_definition.parameters().clone();
        let source_layout =
            derive_partitioned_transform_source_layout(&source_parameters, &parameters, rank)
                .map_err(|failure| error(failure.to_string()))?;
        let source_local =
            crate::deepseek::parallel::v3_local_geometry(&source_args, &source_layout)
                .map_err(|e| error(e.to_string()))?;
        if source_local.state_layout() != &complete_state {
            return Err(error(
                "routed V3 transform source changed local state geometry".into(),
            ));
        }
        let source_plan =
            crate::deepseek::v3_expert_realization_plan(&source_args, Some(&source_local), rank)
                .map_err(|e| error(e.to_string()))?
                .ok_or_else(|| error("V3 source partition has no routed units".into()))?;
        let source_geometry = crate::deepseek::v3_partition_local_geometry(
            &source_args,
            &source_layout,
            [(owned.group().as_str(), owned.units())],
            selected.requirements().ownership(),
            rank,
            &source_plan,
        )
        .map_err(|e| error(e.to_string()))?;
        let source_partition = selected
            .requirements()
            .construct_partition(
                &source_parameters,
                &complete_state,
                &state_plan,
                source_geometry,
                crate::deepseek::v3::TargetBoundarySchema::from_args(&source_args),
            )
            .map_err(|e| error(e.to_string()))?;
        crate::deepseek::V3PartitionLocalFoundation::from_partition(
            &source_args,
            &source_partition,
        )
        .map_err(|e| error(e.to_string()))?;
        let mut source = crate::deepseek::v3::V3Kernel::<B>::from_partition_definition(
            source_args,
            source_definition,
            source_local,
            &source_partition,
            context,
        )
        .map_err(|e| error(e.to_string()))?;
        source.kernel_mut().install_expert_realization(source_plan);
        Some((source, source_layout))
    } else {
        None
    };
    let mut architecture = crate::deepseek::v3::V3Kernel::<B>::from_partition_definition(
        selected_args,
        definition,
        local,
        &partition,
        context,
    )
    .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    architecture
        .kernel_mut()
        .install_expert_realization(plan.clone());
    prepare_family_routed_partition::<B, S, _, _, _, _>(
        architecture,
        source_architecture,
        selected,
        partition,
        parameters,
        layout,
        None,
        plan,
        store,
        visitor,
        capability_estimate,
        effective_model_type,
    )
}

#[allow(clippy::too_many_arguments)]
fn prepare_deepseek_v4_routed_partition<B, S, V>(
    effective_model_type: String,
    selected_args: crate::deepseek::V4Args,
    selected: SelectedPartitionedAdmission<SelectedRoutedTextRealization, RoutedTextRequirements>,
    store: eredu_checkpoint::store::SharedCheckpointSource,
    context: &<B::Tensor as eredu_nn::Tensor>::Context,
    visitor: V,
    capability_estimate: crate::capability::CapabilityEstimate,
) -> Result<V::Output, DenseDecoderPartitionedDispatchError<V::Error>>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::HyperNeuralBackend,
    S: eredu_runtime::LayerRuntimeState<B>,
    S::LayerState: eredu_nn::PoolingAttentionCache<B::Tensor>,
    V: FamilyRoutedPartitionVisitor<B, S, crate::deepseek::v4::Model<B>>,
{
    if selected_args.num_nextn_predict_layers != 0 || selected_args.dspark.is_some() {
        return Err(DenseDecoderPartitionedDispatchError::Architecture(
            "DeepSeek-V4 routed partition requires sparse target units without prediction".into(),
        ));
    }
    if selected
        .base()
        .bank(eredu_runtime::RoutedBankId::new(0))
        .expect("validated family bank")
        .plan()
        .gated()
        .is_none()
    {
        return Err(DenseDecoderPartitionedDispatchError::Architecture(
            "DeepSeek-V4 selected a non-gated expert equation".into(),
        ));
    }
    let retained = selected
        .base()
        .text()
        .requirements()
        .shared_component_definition()
        .ok_or_else(|| {
            DenseDecoderPartitionedDispatchError::Architecture(
                "V4 selection has no retained definition".into(),
            )
        })?;
    let definition = crate::replicated_text::executable_component(selected.base().text())
        .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
    let parameters = definition.parameters().clone();
    let rank = selected.requirements().topology();
    let layout = derive_partitioned_local_layout(&parameters, rank)
        .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
    let [owned] = selected.requirements().groups() else {
        return Err(DenseDecoderPartitionedDispatchError::Architecture(
            "prediction-free DeepSeek-V4 must own exactly one target group".into(),
        ));
    };
    if owned.group().as_str() != crate::decoder::TARGET_EXECUTION_GROUP {
        return Err(DenseDecoderPartitionedDispatchError::Architecture(
            "DeepSeek-V4 admission names a different execution group".into(),
        ));
    }
    let local = crate::deepseek::parallel::v4_local_geometry(&selected_args, &layout)
        .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    let plan = crate::deepseek::v4_expert_realization_plan(&selected_args, Some(&local), rank)
        .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    let geometry = crate::deepseek::v4_partition_local_geometry(
        &selected_args,
        &layout,
        [(owned.group().as_str(), owned.units())],
        selected.requirements().ownership(),
        rank,
        &plan,
    )
    .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    let complete_state = local.state_layout().clone();
    let state_plan = crate::transport::pipeline_state(0, &complete_state);
    let boundary = crate::deepseek::v4::TargetBoundarySchema::from_args(&selected_args)
        .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    let partition = selected
        .requirements()
        .construct_partition(
            &parameters,
            &complete_state,
            &state_plan,
            geometry,
            boundary,
        )
        .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    crate::deepseek::V4PartitionLocalFoundation::from_partition(&selected_args, &partition)
        .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    let source_architecture = if selected.base().text().uses_weight_transform() {
        let error = |error: String| DenseDecoderPartitionedDispatchError::Architecture(error);
        let source_args =
            crate::replicated_text::source_deepseek_v4_args(&selected_args, selected.base().text())
                .map_err(error)?;
        let source_definition = crate::deepseek::v4::project_parameters(&source_args, &retained)
            .map_err(|e| error(e.to_string()))?;
        let source_parameters = source_definition.parameters().clone();
        let source_layout =
            derive_partitioned_transform_source_layout(&source_parameters, &parameters, rank)
                .map_err(|failure| error(failure.to_string()))?;
        let source_local =
            crate::deepseek::parallel::v4_local_geometry(&source_args, &source_layout)
                .map_err(|e| error(e.to_string()))?;
        if source_local.state_layout() != &complete_state {
            return Err(error(
                "routed V4 transform source changed local state geometry".into(),
            ));
        }
        let source_plan =
            crate::deepseek::v4_expert_realization_plan(&source_args, Some(&source_local), rank)
                .map_err(|e| error(e.to_string()))?;
        let source_geometry = crate::deepseek::v4_partition_local_geometry(
            &source_args,
            &source_layout,
            [(owned.group().as_str(), owned.units())],
            selected.requirements().ownership(),
            rank,
            &source_plan,
        )
        .map_err(|e| error(e.to_string()))?;
        let source_partition = selected
            .requirements()
            .construct_partition(
                &source_parameters,
                &complete_state,
                &state_plan,
                source_geometry,
                crate::deepseek::v4::TargetBoundarySchema::from_args(&source_args)
                    .map_err(|e| error(e.to_string()))?,
            )
            .map_err(|e| error(e.to_string()))?;
        crate::deepseek::V4PartitionLocalFoundation::from_partition(
            &source_args,
            &source_partition,
        )
        .map_err(|e| error(e.to_string()))?;
        let mut source = crate::deepseek::v4::V4Kernel::<B>::from_partition_definition(
            source_args,
            source_definition,
            source_local,
            &source_partition,
            context,
        )
        .map_err(|e| error(e.to_string()))?;
        source.kernel_mut().install_expert_realization(source_plan);
        Some((source, source_layout))
    } else {
        None
    };
    let mut architecture = crate::deepseek::v4::V4Kernel::<B>::from_partition_definition(
        selected_args,
        definition,
        local,
        &partition,
        context,
    )
    .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    architecture
        .kernel_mut()
        .install_expert_realization(plan.clone());
    prepare_family_routed_partition::<B, S, _, _, _, _>(
        architecture,
        source_architecture,
        selected,
        partition,
        parameters,
        layout,
        None,
        plan,
        store,
        visitor,
        capability_estimate,
        effective_model_type,
    )
}

#[allow(clippy::too_many_arguments)]
fn prepare_family_routed_partition<B, S, A, G, E, V>(
    architecture: A,
    source_architecture: Option<(A, eredu_runtime::LocalModelLayout)>,
    selected: SelectedPartitionedAdmission<SelectedRoutedTextRealization, RoutedTextRequirements>,
    partition: ArchitecturePartition<G, A::Boundary>,
    parameters: eredu_runtime::ArchitectureParameterDescription,
    layout: eredu_runtime::LocalModelLayout,
    rows: Option<eredu_runtime::SelectedRowLookups>,
    plan: crate::ExpertRealizationPlan<E>,
    store: eredu_checkpoint::store::SharedCheckpointSource,
    visitor: V,
    capability_estimate: crate::capability::CapabilityEstimate,
    effective_model_type: String,
) -> Result<V::Output, DenseDecoderPartitionedDispatchError<V::Error>>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: eredu_runtime::LayerRuntimeState<B>,
    A: PipelinePartitionArchitecture<B, S>
        + eredu_runtime::ReplicatedTextArchitecture<B, S, Error = eredu_nn::Error>
        + eredu_runtime::ParallelRoutedLayeredArchitecture<B, S>
        + 'static,
    A::StaticModules: Clone,
    G: 'static,
    E: RoutedCollectiveSpec + crate::routed_text::RoutedGroupedSpec,
    V: FamilyRoutedPartitionVisitor<B, S, A>,
{
    selected
        .requirements()
        .validate_binding(&partition)
        .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
    let mut tasks = eredu_runtime::partition_selected_replicated_text_materialization_tasks(
        selected.materialization_tasks(),
        &parameters,
        &partition,
    )
    .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    let mut addressable_members = crate::routed_text::project_addressable_members_with_tasks(
        selected
            .base()
            .bank(eredu_runtime::RoutedBankId::new(0))
            .expect("validated family bank")
            .catalog(),
        selected.base().text(),
        &tasks,
    )
    .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    let local_units = partition
        .units()
        .map(|address| address.index())
        .collect::<BTreeSet<_>>();
    let local_experts = plan
        .local_global_group_indices()
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    addressable_members.retain(|member| {
        local_units.contains(&member.key().unit()) && local_experts.contains(&member.key().member())
    });
    let global_rank = selected.requirements().topology().global_rank();
    addressable_members = addressable_members
        .into_iter()
        .map(|member| member.with_owner_rank(global_rank))
        .collect();
    let bank_residency = selected.base().bank_residency();
    if matches!(
        bank_residency,
        eredu_runtime::ParameterBankResidency::IndependentCache(_)
    ) {
        let targets = selected
            .base()
            .bank(eredu_runtime::RoutedBankId::new(0))
            .expect("validated family bank")
            .catalog()
            .units()
            .iter()
            .filter(|unit| {
                unit.distribution() == crate::ExpertResidencyDistribution::ExpertParallel
            })
            .flat_map(crate::ExpertResidencyUnit::parameters)
            .map(crate::ExpertParameterRecipe::logical_target)
            .collect::<BTreeSet<_>>();
        tasks.retain(|task| !targets.contains(task.name()));
    }
    let catalog = selected
        .base()
        .bank(eredu_runtime::RoutedBankId::new(0))
        .expect("validated family bank")
        .catalog()
        .clone();
    let prepared = prepare_partitioned::<B, S, _, _, _, _, _>(architecture, selected, partition)
        .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
    let banks = BTreeMap::from([(
        eredu_runtime::RoutedBankId::new(0),
        prepared
            .selected()
            .base()
            .bank(eredu_runtime::RoutedBankId::new(0))
            .expect("validated family bank")
            .with_partition_geometry(
                E::into_routed_grouped_plan(plan),
                catalog,
                addressable_members,
                &layout,
            )
            .map_err(|error| {
                DenseDecoderPartitionedDispatchError::Architecture(error.to_string())
            })?,
    )]);
    let execution = prepare_routed_execution::<B, S, _, _, _>(&prepared, &banks)
        .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
    visitor
        .visit(
            AdmittedRoutedPartition {
                prepared,
                source_architecture,
                layout,
                provider_layout: None,
                tasks,
                capability_estimate,
                effective_model_type,
                bank_residency,
                banks,
                rows,
                execution,
                backend: PhantomData,
            },
            store,
        )
        .map_err(DenseDecoderPartitionedDispatchError::Visitor)
}

trait RoutedPartitionedConfig<S>: crate::decoder::PartitionedConfig {
    fn partition_expert_plans(
        &self,
        common: Option<&std::sync::Arc<crate::decoder::DecoderDeclaration>>,
        declarations: Option<&crate::decoder::ProjectionDeclarations<S>>,
        layout: &eredu_runtime::LocalModelLayout,
        topology: ParallelRankTopology,
    ) -> Result<
        BTreeMap<eredu_runtime::RoutedBankId, crate::routed_text::RoutedGroupedPlan>,
        eredu_nn::Error,
    >;

    fn routed_partition_geometry(
        &self,
        common: Option<&std::sync::Arc<crate::decoder::DecoderDeclaration>>,
        declarations: Option<&crate::decoder::ProjectionDeclarations<S>>,
        layout: &eredu_runtime::LocalModelLayout,
        owned_units: Range<usize>,
        topology: ParallelRankTopology,
        realizations: &BTreeMap<eredu_runtime::RoutedBankId, crate::routed_text::RoutedGroupedPlan>,
    ) -> Result<crate::decoder::PartitionLocalGeometry<Self>, eredu_runtime::ParallelPlanError>;
}

impl RoutedPartitionedConfig<crate::qwen::FeedForwardParameters> for crate::qwen::ModelArgs {
    fn partition_expert_plans(
        &self,
        common: Option<&std::sync::Arc<crate::decoder::DecoderDeclaration>>,
        declarations: Option<
            &crate::decoder::ProjectionDeclarations<crate::qwen::FeedForwardParameters>,
        >,
        layout: &eredu_runtime::LocalModelLayout,
        topology: ParallelRankTopology,
    ) -> Result<
        BTreeMap<eredu_runtime::RoutedBankId, crate::routed_text::RoutedGroupedPlan>,
        eredu_nn::Error,
    > {
        let declaration = crate::qwen::Declaration::from_scopes(
            self.clone(),
            common
                .ok_or_else(|| eredu_nn::Error::backend("missing Qwen common scopes"))?
                .clone(),
            declarations.cloned(),
        )?;
        let geometry = crate::qwen::local_geometry(&declaration, layout)
            .map_err(eredu_nn::Error::backend_source)?;
        let plan =
            crate::qwen::expert_realization_plan(&declaration, Some(&geometry), topology)?
                .ok_or_else(|| eredu_nn::Error::backend("partitioned Qwen has no routed bank"))?;
        Ok(BTreeMap::from([(
            eredu_runtime::RoutedBankId::new(0),
            plan.into(),
        )]))
    }

    fn routed_partition_geometry(
        &self,
        common: Option<&std::sync::Arc<crate::decoder::DecoderDeclaration>>,
        declarations: Option<
            &crate::decoder::ProjectionDeclarations<crate::qwen::FeedForwardParameters>,
        >,
        layout: &eredu_runtime::LocalModelLayout,
        owned_units: Range<usize>,
        topology: ParallelRankTopology,
        realizations: &BTreeMap<eredu_runtime::RoutedBankId, crate::routed_text::RoutedGroupedPlan>,
    ) -> Result<crate::decoder::PartitionLocalGeometry<Self>, eredu_runtime::ParallelPlanError>
    {
        let declaration = crate::qwen::Declaration::from_scopes(
            self.clone(),
            common
                .ok_or_else(|| {
                    eredu_runtime::ParallelPlanError::InvalidGroup(
                        "missing Qwen common scopes".into(),
                    )
                })?
                .clone(),
            declarations.cloned(),
        )
        .map_err(|e| eredu_runtime::ParallelPlanError::InvalidGroup(e.to_string()))?;
        crate::qwen::partition_local_routed_geometry(
            &declaration,
            layout,
            owned_units,
            topology,
            realizations
                .get(&eredu_runtime::RoutedBankId::new(0))
                .and_then(|plan| plan.gated())
                .ok_or_else(|| {
                    eredu_runtime::ParallelPlanError::InvalidGroup("missing gated bank".into())
                })?,
        )
    }
}

impl RoutedPartitionedConfig<eredu_nn::RoutedGatedProductSpec> for crate::gpt_oss::ModelArgs {
    fn partition_expert_plans(
        &self,
        common: Option<&std::sync::Arc<crate::decoder::DecoderDeclaration>>,
        declarations: Option<
            &crate::decoder::ProjectionDeclarations<eredu_nn::RoutedGatedProductSpec>,
        >,
        layout: &eredu_runtime::LocalModelLayout,
        topology: ParallelRankTopology,
    ) -> Result<
        BTreeMap<eredu_runtime::RoutedBankId, crate::routed_text::RoutedGroupedPlan>,
        eredu_nn::Error,
    > {
        let declaration = crate::gpt_oss::Declaration::from_scopes(
            self.clone(),
            common
                .ok_or_else(|| eredu_nn::Error::backend("missing admitted GPT-OSS common scopes"))?
                .clone(),
            declarations
                .ok_or_else(|| eredu_nn::Error::backend("missing admitted GPT-OSS projections"))?
                .clone(),
        )?;
        let geometry = crate::gpt_oss::local_geometry(&declaration, layout)
            .map_err(eredu_nn::Error::backend_source)?;
        crate::gpt_oss::expert_realization_plan(&declaration, Some(&geometry), topology)
            .map(|plan| BTreeMap::from([(eredu_runtime::RoutedBankId::new(0), plan.into())]))
    }

    fn routed_partition_geometry(
        &self,
        common: Option<&std::sync::Arc<crate::decoder::DecoderDeclaration>>,
        declarations: Option<
            &crate::decoder::ProjectionDeclarations<eredu_nn::RoutedGatedProductSpec>,
        >,
        layout: &eredu_runtime::LocalModelLayout,
        owned_units: Range<usize>,
        topology: ParallelRankTopology,
        realizations: &BTreeMap<eredu_runtime::RoutedBankId, crate::routed_text::RoutedGroupedPlan>,
    ) -> Result<crate::decoder::PartitionLocalGeometry<Self>, eredu_runtime::ParallelPlanError>
    {
        let declaration = crate::gpt_oss::Declaration::from_scopes(
            self.clone(),
            common
                .ok_or_else(|| {
                    eredu_runtime::ParallelPlanError::InvalidGroup(
                        "missing admitted GPT-OSS common scopes".into(),
                    )
                })?
                .clone(),
            declarations
                .ok_or_else(|| {
                    eredu_runtime::ParallelPlanError::InvalidGroup(
                        "missing admitted GPT-OSS projections".into(),
                    )
                })?
                .clone(),
        )
        .map_err(|e| eredu_runtime::ParallelPlanError::InvalidGroup(e.to_string()))?;
        crate::gpt_oss::partition_local_routed_geometry(
            &declaration,
            layout,
            owned_units,
            topology,
            realizations
                .get(&eredu_runtime::RoutedBankId::new(0))
                .and_then(|plan| plan.gated())
                .ok_or_else(|| {
                    eredu_runtime::ParallelPlanError::InvalidGroup("missing gated bank".into())
                })?,
        )
    }
}

impl RoutedPartitionedConfig<crate::k2_horizon::ProjectionParameters>
    for crate::k2_horizon::ModelArgs
{
    fn partition_expert_plans(
        &self,
        common: Option<&std::sync::Arc<crate::decoder::DecoderDeclaration>>,
        declarations: Option<
            &crate::decoder::ProjectionDeclarations<crate::k2_horizon::ProjectionParameters>,
        >,
        layout: &eredu_runtime::LocalModelLayout,
        topology: ParallelRankTopology,
    ) -> Result<
        BTreeMap<eredu_runtime::RoutedBankId, crate::routed_text::RoutedGroupedPlan>,
        eredu_nn::Error,
    > {
        let declaration = crate::k2_horizon::Declaration::from_scopes(
            self.clone(),
            common
                .ok_or_else(|| eredu_nn::Error::backend("missing K2 common scopes"))?
                .clone(),
            declarations.cloned(),
        )?;
        crate::k2_horizon::expert_realization_plans(&declaration, topology, Some(layout))
    }
    fn routed_partition_geometry(
        &self,
        common: Option<&std::sync::Arc<crate::decoder::DecoderDeclaration>>,
        declarations: Option<
            &crate::decoder::ProjectionDeclarations<crate::k2_horizon::ProjectionParameters>,
        >,
        layout: &eredu_runtime::LocalModelLayout,
        owned: Range<usize>,
        topology: ParallelRankTopology,
        plans: &BTreeMap<eredu_runtime::RoutedBankId, crate::routed_text::RoutedGroupedPlan>,
    ) -> Result<crate::decoder::PartitionLocalGeometry<Self>, eredu_runtime::ParallelPlanError>
    {
        let declaration = crate::k2_horizon::Declaration::from_scopes(
            self.clone(),
            common
                .ok_or_else(|| {
                    eredu_runtime::ParallelPlanError::InvalidGroup(
                        "missing K2 common scopes".into(),
                    )
                })?
                .clone(),
            declarations.cloned(),
        )
        .map_err(|e| eredu_runtime::ParallelPlanError::InvalidGroup(e.to_string()))?;
        crate::k2_horizon::partition_local_routed_geometry(
            &declaration,
            layout,
            owned,
            topology,
            plans,
        )
    }
}

/// Immutable production route selected for an admitted dense-decoder partition.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum DenseDecoderPartitionedProductionRoute {
    /// Selected-residency execution through the neutral runtime.
    NeutralPartitioned,
    /// The admission is not implemented by this production class.
    Unsupported(DenseDecoderPartitionedUnsupportedReason),
}

/// Architecture-owned reason why a partition cannot enter neutral execution.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum DenseDecoderPartitionedUnsupportedReason {
    /// The normalized family is routed, hybrid, multimodal, or otherwise unsupported.
    UnsupportedArchitecture,
    /// The artifact has no neutral partitioned materializer for this architecture.
    UnsupportedArtifact,
    /// The topology is not a supported TP/PP dense partition.
    UnsupportedTopology,
    /// Weight residency is bounded or otherwise nonresident.
    NonResident,
    /// At least one parameter requires a load-time transform.
    TransformedParameters,
}

#[derive(Clone, Copy)]
struct DenseDecoderProductionSupport {
    unindexed_safetensors: bool,
    gguf: bool,
    bounded_residency: bool,
    transforms: bool,
}

impl DenseDecoderProductionSupport {
    const INDEXED_RESIDENT_DIRECT: Self = Self {
        unindexed_safetensors: false,
        gguf: false,
        bounded_residency: false,
        transforms: false,
    };

    const HETEROGENEOUS_DENSE: Self = Self {
        unindexed_safetensors: true,
        gguf: true,
        bounded_residency: true,
        transforms: true,
    };

    const COMPLETE_DENSE: Self = Self {
        unindexed_safetensors: true,
        gguf: true,
        bounded_residency: true,
        transforms: true,
    };
}

fn classify_dense_decoder_production_route(
    support: DenseDecoderProductionSupport,
    format: eredu_core::ArtifactFormat,
    indexed_safetensors: bool,
    topology: eredu_core::ParallelTopology,
    residency: &eredu_runtime::LayerWeightResidency,
    has_transform: bool,
) -> DenseDecoderPartitionedProductionRoute {
    use DenseDecoderPartitionedProductionRoute as Route;
    use DenseDecoderPartitionedUnsupportedReason as Reason;

    let supported_artifact = match format {
        eredu_core::ArtifactFormat::SafeTensors => {
            indexed_safetensors || support.unindexed_safetensors
        }
        eredu_core::ArtifactFormat::Gguf => support.gguf,
        _ => false,
    };
    if !supported_artifact {
        return Route::Unsupported(Reason::UnsupportedArtifact);
    }
    if (topology.tensor() == 1 && topology.pipeline() == 1) || topology.expert() != 1 {
        return Route::Unsupported(Reason::UnsupportedTopology);
    }
    match residency {
        eredu_runtime::LayerWeightResidency::FullyResident => {}
        eredu_runtime::LayerWeightResidency::LayerwiseHost(_)
        | eredu_runtime::LayerWeightResidency::DenseDiskStream(_)
            if support.bounded_residency => {}
        _ => return Route::Unsupported(Reason::NonResident),
    }
    if has_transform && !support.transforms {
        return Route::Unsupported(Reason::TransformedParameters);
    }
    Route::NeutralPartitioned
}

/// Classifies one already-selected direct admission exactly once, before payload opening.
pub fn dense_decoder_partitioned_production_route(
    inspection: &ArtifactInspection<ArtifactArchitecturePlan>,
    selected: &SelectedPartitionedAdmission<
        eredu_runtime::SelectedReplicatedTextRealization,
        eredu_runtime::ReplicatedTextRequirements,
    >,
) -> DenseDecoderPartitionedProductionRoute {
    use DenseDecoderPartitionedProductionRoute as Route;
    use DenseDecoderPartitionedUnsupportedReason as Reason;

    if !is_supported_dense_decoder_partition(inspection.architecture_plan()) {
        return Route::Unsupported(Reason::UnsupportedArchitecture);
    }
    let models = (
        inspection
            .architecture_plan()
            .safetensors_architecture()
            .map(|plan| plan.model()),
        inspection
            .architecture_plan()
            .gguf_plan()
            .map(|plan| plan.model()),
    );
    let llama_compatible = matches!(
        models,
        (
            Some(crate::configuration::SafetensorsModelConfig::NemotronLabsDiffusion(_)),
            None
        ) | (
            None,
            Some(crate::configuration::GgufModelConfig::NemotronLabsDiffusion(_))
        ) | (
            Some(crate::configuration::SafetensorsModelConfig::Gemma2(_)),
            None
        ) | (None, Some(crate::configuration::GgufModelConfig::Gemma2(_)))
            | (
                Some(crate::configuration::SafetensorsModelConfig::Llama(_)),
                None
            )
            | (None, Some(crate::configuration::GgufModelConfig::Llama(_)))
            | (
                Some(crate::configuration::SafetensorsModelConfig::Nanbeige(_)),
                None
            )
            | (
                None,
                Some(crate::configuration::GgufModelConfig::Nanbeige(_))
            )
    );
    let qwen_compatible = match models {
        (Some(crate::configuration::SafetensorsModelConfig::Qwen(args)), None)
        | (None, Some(crate::configuration::GgufModelConfig::Qwen(args))) => !args.is_moe(),
        (Some(crate::configuration::SafetensorsModelConfig::K2Horizon(args)), None)
        | (None, Some(crate::configuration::GgufModelConfig::K2Horizon(args))) => !args.is_moe(),
        _ => false,
    };
    let heterogeneous_dense = matches!(
        models,
        (Some(crate::configuration::SafetensorsModelConfig::Lfm2(args)), None)
            if !args.has_sparse_moe_layers()
    ) || matches!(
        models,
        (None, Some(crate::configuration::GgufModelConfig::Lfm2(args)))
            if !args.has_sparse_moe_layers()
    ) || matches!(
        models,
        (
            Some(crate::configuration::SafetensorsModelConfig::Lfm3(_)),
            None
        ) | (None, Some(crate::configuration::GgufModelConfig::Lfm3(_)))
    ) || matches!(
        models,
        (Some(crate::configuration::SafetensorsModelConfig::KimiLinear(args)), None)
            if !args.has_sparse_moe_layers() && args.num_nextn_predict_layers == 0
    ) || matches!(
        models,
        (None, Some(crate::configuration::GgufModelConfig::KimiLinear(args)))
            if !args.has_sparse_moe_layers() && args.num_nextn_predict_layers == 0
    ) || matches!(
        models,
        (Some(crate::configuration::SafetensorsModelConfig::NemotronH(args)), None)
            if !args.has_sparse_moe_layers() && args.num_nextn_predict_layers == 0
    ) || matches!(
        models,
        (None, Some(crate::configuration::GgufModelConfig::NemotronH(args)))
            if !args.has_sparse_moe_layers() && args.num_nextn_predict_layers == 0
    ) || matches!(
        models,
        (Some(crate::configuration::SafetensorsModelConfig::DeepSeekV3(args)), None)
            | (None, Some(crate::configuration::GgufModelConfig::DeepSeekV3(args)))
            if !args.has_sparse_moe_layers() && args.num_nextn_predict_layers == 0
    ) || matches!(
        models,
        (Some(crate::configuration::SafetensorsModelConfig::QwenHybrid(args)), None)
            | (None, Some(crate::configuration::GgufModelConfig::QwenHybrid(args)))
            if args.vision.is_none() && !args.text.is_moe() && args.text.mtp_num_hidden_layers == 0
    );
    let support = if llama_compatible || qwen_compatible {
        DenseDecoderProductionSupport::COMPLETE_DENSE
    } else if heterogeneous_dense {
        DenseDecoderProductionSupport::HETEROGENEOUS_DENSE
    } else {
        DenseDecoderProductionSupport::INDEXED_RESIDENT_DIRECT
    };
    classify_dense_decoder_production_route(
        support,
        inspection.format(),
        inspection
            .safetensors_shards()
            .and_then(eredu_checkpoint::safetensors::SafetensorsShards::tensor_locations)
            .is_some(),
        selected.requirements().topology().topology(),
        &selected.base().residency(),
        selected.base().parameters().iter().any(|parameter| {
            !matches!(
                parameter.lowering(),
                eredu_runtime::WeightLoweringKind::Direct
                    | eredu_runtime::WeightLoweringKind::Derived
            )
        }),
    )
}

/// Backend-generic visitor for an exact direct partition.
pub trait DirectPartitionedArchitectureVisitor<B, S, G, W>
where
    B: eredu_nn::NeuralBackend,
    S: eredu_runtime::RuntimeState<B>,
{
    /// Completed neutral construction output.
    type Output;
    /// Mechanism binding failure.
    type Error;

    /// Receives one statically known direct architecture and exact partition proof.
    fn visit<A>(
        self,
        prepared: PreparedPartitionedAdmission<
            A,
            eredu_runtime::SelectedReplicatedTextRealization,
            eredu_runtime::ReplicatedTextRequirements,
            G,
            W,
        >,
    ) -> Result<Self::Output, Self::Error>
    where
        A: eredu_runtime::PartitionedLayeredArchitecture<B, S, Boundary = W>,
        A::Error: std::fmt::Display;
}

/// Pairs and dispatches one direct architecture without re-selecting its realization.
pub fn visit_direct_partitioned_architecture<B, S, A, G, W, V>(
    architecture: A,
    selected: SelectedPartitionedAdmission<
        eredu_runtime::SelectedReplicatedTextRealization,
        eredu_runtime::ReplicatedTextRequirements,
    >,
    partition: ArchitecturePartition<G, W>,
    visitor: V,
) -> Result<V::Output, PartitionedDispatchError<V::Error>>
where
    B: eredu_nn::NeuralBackend,
    S: eredu_runtime::RuntimeState<B>,
    A: eredu_runtime::PartitionedLayeredArchitecture<B, S, Boundary = W>,
    A::Error: std::fmt::Display,
    V: DirectPartitionedArchitectureVisitor<B, S, G, W>,
{
    let prepared = prepare_partitioned::<B, S, _, _, _, _, _>(architecture, selected, partition)
        .map_err(PartitionedDispatchError::Architecture)?;
    visitor
        .visit(prepared)
        .map_err(PartitionedDispatchError::Visitor)
}

/// Family-blind backend visitor for an exact direct partitioned prediction target.
pub trait DirectPartitionedPredictionTargetVisitor<B, S, M, G, W>
where
    B: eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::GroupedNeuralBackend
        + eredu_nn::HyperNeuralBackend,
    S: eredu_runtime::RuntimeState<B>,
    M: crate::prediction_extension::PredictionExtensionMaterializer<B>,
{
    /// Completed neutral construction output.
    type Output;
    /// Mechanism binding failure.
    type Error;

    /// Receives a direct partition only after exact extension pairing.
    fn visit<A>(
        self,
        prepared: PreparedPartitionedAdmission<
            A,
            eredu_runtime::SelectedReplicatedTextRealization,
            eredu_runtime::ReplicatedTextRequirements,
            G,
            W,
        >,
        extension: <A as eredu_runtime::speculative::PredictionTarget<B>>::Extension<M>,
    ) -> Result<Self::Output, Self::Error>
    where
        A: eredu_runtime::PartitionedLayeredArchitecture<B, S, Boundary = W>
            + crate::prediction_extension::MaterializedPredictionTarget<B>,
        A::Error: std::fmt::Display;
}

/// Pairs and dispatches one direct partitioned prediction target.
pub fn visit_direct_partitioned_prediction_target_architecture<B, S, M, A, G, W, V>(
    architecture: A,
    extension: crate::prediction_extension::MaterializedPredictionExtension<B, M>,
    selected: SelectedPartitionedAdmission<
        eredu_runtime::SelectedReplicatedTextRealization,
        eredu_runtime::ReplicatedTextRequirements,
    >,
    partition: ArchitecturePartition<G, W>,
    visitor: V,
) -> Result<V::Output, PartitionedDispatchError<V::Error>>
where
    B: eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::GroupedNeuralBackend
        + eredu_nn::HyperNeuralBackend,
    S: eredu_runtime::RuntimeState<B>,
    M: crate::prediction_extension::PredictionExtensionMaterializer<B>,
    A: eredu_runtime::PartitionedLayeredArchitecture<B, S, Boundary = W>
        + crate::prediction_extension::MaterializedPredictionTarget<B>,
    A::Error: std::fmt::Display,
    V: DirectPartitionedPredictionTargetVisitor<B, S, M, G, W>,
{
    let prepared = prepare_partitioned::<B, S, _, _, _, _, _>(architecture, selected, partition)
        .map_err(PartitionedDispatchError::Architecture)?;
    let extension = A::pair_prediction_extension(extension)
        .map_err(|error| PartitionedDispatchError::Architecture(error.to_string()))?;
    visitor
        .visit(prepared, extension)
        .map_err(PartitionedDispatchError::Visitor)
}

/// Backend-generic visitor for an exact routed partition.
pub trait RoutedPartitionedArchitectureVisitor<B, S, G, W>
where
    B: eredu_nn::GroupedNeuralBackend,
    S: eredu_runtime::RuntimeState<B>,
{
    /// Completed neutral construction output.
    type Output;
    /// Mechanism binding failure.
    type Error;

    /// Receives one statically known routed architecture and exact partition proof.
    fn visit<A>(
        self,
        prepared: PreparedPartitionedAdmission<
            A,
            SelectedRoutedTextRealization,
            RoutedTextRequirements,
            G,
            W,
        >,
    ) -> Result<Self::Output, Self::Error>
    where
        A: eredu_runtime::PartitionedLayeredArchitecture<B, S, Boundary = W>
            + eredu_runtime::ParallelRoutedLayeredArchitecture<B, S>,
        A::Error: std::fmt::Display;
}

/// Exact routed decoder partition constructed from a selected admission.
pub(crate) struct AdmittedRoutedPartition<B, A, G, W>
where
    B: eredu_nn::GroupedNeuralBackend,
{
    prepared: PreparedPartitionedAdmission<
        A,
        SelectedRoutedTextRealization,
        RoutedTextRequirements,
        G,
        W,
    >,
    source_architecture: Option<(A, eredu_runtime::LocalModelLayout)>,
    layout: eredu_runtime::LocalModelLayout,
    provider_layout: Option<eredu_runtime::LocalModelLayout>,
    tasks: Vec<eredu_runtime::ReplicatedTextMaterializationTask>,
    capability_estimate: crate::capability::CapabilityEstimate,
    effective_model_type: String,
    bank_residency: eredu_runtime::ParameterBankResidency,
    banks: BTreeMap<eredu_runtime::RoutedBankId, crate::routed_text::SelectedRoutedBank>,
    rows: Option<eredu_runtime::SelectedRowLookups>,
    execution: PreparedRoutedExecutionHandoff,
    backend: PhantomData<fn() -> B>,
}

fn prepare_routed_execution<B, S, A, G, W>(
    prepared: &PreparedPartitionedAdmission<
        A,
        SelectedRoutedTextRealization,
        RoutedTextRequirements,
        G,
        W,
    >,
    banks: &BTreeMap<eredu_runtime::RoutedBankId, crate::routed_text::SelectedRoutedBank>,
) -> Result<PreparedRoutedExecutionHandoff, String>
where
    B: eredu_nn::GroupedNeuralBackend,
    S: eredu_runtime::RuntimeState<B>,
    A: PipelinePartitionArchitecture<B, S>,
    A::Error: std::fmt::Display,
{
    let selected = prepared.selected();
    let topology = selected.topology();
    let pipeline = topology.pipeline_parallel_size() > 1;
    let execution_plan = if pipeline {
        selected.pipeline_execution_plan()?
    } else {
        selected.routed_execution_plan()?
    };
    let sampling_group = execution_plan
        .publication()
        .ok_or_else(|| "routed partition has no publication group".to_owned())?
        .group;
    let communication_tensor_group = selected.tensor_group();
    let active_tensor_group = (topology.tensor_parallel_size() > 1)
        .then_some(communication_tensor_group)
        .flatten();
    let expert_group = selected.expert_group();
    let realizations = banks
        .iter()
        .map(|(id, bank)| (*id, bank.plan().clone()))
        .collect();
    let strategy = if pipeline {
        let collective_waves = expert_group
            .map(|_| {
                let unit_count = selected.partition().unit_layout().len();
                let hidden_width = selected
                    .boundary_routes()
                    .first()
                    .and_then(|route| route.schema().primary().shape().last())
                    .copied()
                    .and_then(|width| usize::try_from(width).ok())
                    .ok_or_else(|| {
                        "routed pipeline has no selected hidden wire width".to_owned()
                    })?;
                let output_width =
                    usize::try_from(prepared.architecture().partition_output_width())
                        .map_err(|_| "routed output width is negative".to_owned())?;
                let mut blocks = Vec::with_capacity(unit_count);
                for unit in 0..unit_count {
                    let actual = banks
                        .iter()
                        .filter(|(_, bank)| bank.plan().has_unit(bank.owner_group().as_str(), unit))
                        .map(|(id, _)| *id)
                        .collect::<BTreeSet<_>>();
                    if actual.is_empty() {
                        let sums = prepared
                            .architecture()
                            .partition_routed_tensor_reductions(unit, false)
                            .map_err(|e| e.to_string())?;
                        blocks.push(vec![RoutedExpertUnitWave::ordinary(
                            unit,
                            hidden_width,
                            sums,
                        )]);
                        continue;
                    }
                    let order = prepared.architecture().partition_routed_bank_order(unit);
                    if order.iter().copied().collect::<BTreeSet<_>>() != actual
                        || order.len() != actual.len()
                    {
                        return Err(format!(
                            "unit {unit} collective order differs from its bank invocations"
                        ));
                    }
                    blocks.push(
                        order
                            .into_iter()
                            .map(|id| {
                                let bank = &banks[&id];
                                let sums = prepared
                                    .architecture()
                                    .partition_routed_bank_tensor_reductions(unit, id)
                                    .map_err(|e| e.to_string())?;
                                RoutedExpertUnitWave::routed(
                                    id,
                                    unit,
                                    bank.owner_group(),
                                    bank.plan(),
                                    topology.tensor_parallel_size(),
                                    sums,
                                )
                            })
                            .collect::<Result<Vec<_>, String>>()?,
                    );
                }
                let waves = RoutedExpertCollectiveWaveSchedule::from_block_waves(
                    blocks,
                    topology.tensor_parallel_size(),
                    topology.tensor_parallel_rank(),
                    topology.pipeline_parallel_size(),
                    output_width,
                )?;
                Ok::<_, String>(
                    waves.with_vocabulary_sharding(
                        prepared
                            .architecture()
                            .partition_tensor_vocabulary_sharded(),
                    ),
                )
            })
            .transpose()?;
        PreparedRoutedUnitStrategy::Pipeline {
            realizations,
            expert_group,
            tensor_group: active_tensor_group,
            collective_waves,
        }
    } else {
        PreparedRoutedUnitStrategy::Local {
            realizations,
            expert_group,
            tensor_group: active_tensor_group,
        }
    };
    PreparedRoutedExecutionHandoff::new(
        execution_plan,
        communication_tensor_group,
        sampling_group,
        selected.activation_dtype(),
        strategy,
    )
}

impl<B, A, G, W> AdmittedRoutedPartition<B, A, G, W>
where
    B: eredu_nn::GroupedNeuralBackend,
{
    /// Immutable execution recipe already selected by architecture admission.
    pub const fn execution_handoff(&self) -> &PreparedRoutedExecutionHandoff {
        &self.execution
    }

    fn into_binding(self) -> eredu_runtime::executable_binding::routed_partition_binding::PreparedRoutedPartitionedArchitecture<B, A, G, W, crate::SelectedRoutedBank>
    where
        A: eredu_runtime::ArchitectureParameters<B>,
        A::DefinitionError: std::fmt::Display,
{
        let facts = self.session_facts();
        let unowned_sources = self.unowned_expert_checkpoint_sources();
        let tensor_rank = self.prepared.selected().topology().tensor_parallel_rank();
        let expert_exchange = self.prepared.selected().expert_group().is_some();
        let Self {
            prepared,
            source_architecture,
            layout,
            provider_layout,
            tasks,
            bank_residency,
            banks,
            rows,
            execution,
            ..
        } = self;
        let excluded = if matches!(
            bank_residency,
            eredu_runtime::ParameterBankResidency::IndependentCache(_)
        ) {
            banks
                .values()
                .flat_map(|bank| bank.catalog().units())
                .filter(|unit| {
                    unit.distribution() == crate::ExpertResidencyDistribution::ExpertParallel
                })
                .flat_map(crate::ExpertResidencyUnit::parameters)
                .map(|parameter| parameter.logical_target().to_owned())
                .collect()
        } else {
            BTreeSet::new()
        };

        let (architecture, selected) = prepared.into_parts();
        let (routed, partition, communication) = selected.into_parts();
        eredu_runtime::executable_binding::routed_partition_binding::PreparedRoutedPartitionedArchitecture::from_admission(
            architecture, routed.text().clone(), partition, communication, source_architecture,
            layout, provider_layout, tasks, facts, tensor_rank,
            eredu_runtime::executable_binding::routed_partition::PreparedPartitionBanks::new(bank_residency, banks, expert_exchange),
            rows, unowned_sources, excluded, execution,
        )
    }

    /// Returns the typed architecture and its exact admitted partition.
    pub const fn prepared(
        &self,
    ) -> &PreparedPartitionedAdmission<A, SelectedRoutedTextRealization, RoutedTextRequirements, G, W>
    {
        &self.prepared
    }

    /// Returns the architecture-derived capability estimate.
    pub const fn capability_estimate(&self) -> &crate::capability::CapabilityEstimate {
        &self.capability_estimate
    }

    /// Returns the normalized model type.
    pub fn effective_model_type(&self) -> &str {
        &self.effective_model_type
    }

    /// Every independently identified rank-local bank and its exact sources.
    pub fn banks(
        &self,
    ) -> &BTreeMap<eredu_runtime::RoutedBankId, crate::routed_text::SelectedRoutedBank> {
        &self.banks
    }
}

pub use eredu_runtime::executable_binding::routed_partition_binding::{
    RoutedPartitionedPredictionTargetProductionVisitor, RoutedPartitionedProductionVisitor,
};

trait FamilyRoutedPartitionVisitor<B, S, A>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: eredu_runtime::RuntimeState<B>,
    A: PipelinePartitionArchitecture<B, S>
        + eredu_runtime::ReplicatedTextArchitecture<B, S, Error = eredu_nn::Error>
        + eredu_runtime::ParallelRoutedLayeredArchitecture<B, S>
        + 'static,
    A::StaticModules: Clone,
{
    type Output;
    type Error;

    fn visit<G>(
        self,
        prepared: AdmittedRoutedPartition<B, A, G, A::Boundary>,
        store: eredu_checkpoint::store::SharedCheckpointSource,
    ) -> Result<Self::Output, Self::Error>
    where
        G: 'static;
}

struct OrdinaryFamilyRoutedPartitionVisitor<V>(V);

impl<B, S, A, V> FamilyRoutedPartitionVisitor<B, S, A> for OrdinaryFamilyRoutedPartitionVisitor<V>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: eredu_runtime::RuntimeState<B>,
    A: PipelinePartitionArchitecture<B, S>
        + eredu_runtime::ReplicatedTextArchitecture<B, S, Error = eredu_nn::Error>
        + eredu_runtime::ParallelRoutedLayeredArchitecture<B, S>
        + 'static,
    A::StaticModules: Clone,
    V: RoutedPartitionedProductionVisitor<B, S, crate::SelectedRoutedBank>,
{
    type Output = V::Output;
    type Error = V::Error;

    fn visit<G>(
        self,
        prepared: AdmittedRoutedPartition<B, A, G, A::Boundary>,
        store: eredu_checkpoint::store::SharedCheckpointSource,
    ) -> Result<Self::Output, Self::Error>
    where
        G: 'static,
    {
        self.0.visit(prepared.into_binding(), store)
    }
}

struct PredictionFamilyRoutedPartitionVisitor<B, A, M, V>
where
    B: eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::GroupedNeuralBackend
        + eredu_nn::HyperNeuralBackend,
    A: crate::prediction_extension::MaterializedPredictionTarget<B>,
    M: crate::prediction_extension::PredictionExtensionMaterializer<B>,
{
    extension: A::Extension<M>,
    visitor: V,
    marker: PhantomData<fn() -> B>,
}

impl<B, S, A, M, V> FamilyRoutedPartitionVisitor<B, S, A>
    for PredictionFamilyRoutedPartitionVisitor<B, A, M, V>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::HyperNeuralBackend,
    S: eredu_runtime::RuntimeState<B>,
    A: PipelinePartitionArchitecture<B, S>
        + eredu_runtime::ReplicatedTextArchitecture<B, S, Error = eredu_nn::Error>
        + eredu_runtime::ParallelRoutedLayeredArchitecture<B, S>
        + crate::prediction_extension::MaterializedPredictionTarget<B>
        + 'static,
    A::StaticModules: Clone,
    M: crate::prediction_extension::PredictionExtensionMaterializer<B>,
    V: RoutedPartitionedPredictionTargetProductionVisitor<B, S, M, crate::SelectedRoutedBank>,
{
    type Output = V::Output;
    type Error = V::Error;

    fn visit<G>(
        self,
        prepared: AdmittedRoutedPartition<B, A, G, A::Boundary>,
        store: eredu_checkpoint::store::SharedCheckpointSource,
    ) -> Result<Self::Output, Self::Error>
    where
        G: 'static,
    {
        self.visitor
            .visit(prepared.into_binding(), self.extension, store)
    }
}

/// Pairs and dispatches one routed architecture without re-selecting its realization.
pub fn visit_routed_partitioned_architecture<B, S, A, G, W, V>(
    architecture: A,
    selected: SelectedPartitionedAdmission<SelectedRoutedTextRealization, RoutedTextRequirements>,
    partition: ArchitecturePartition<G, W>,
    visitor: V,
) -> Result<V::Output, PartitionedDispatchError<V::Error>>
where
    B: eredu_nn::GroupedNeuralBackend,
    S: eredu_runtime::RuntimeState<B>,
    A: eredu_runtime::PartitionedLayeredArchitecture<B, S, Boundary = W>
        + eredu_runtime::ParallelRoutedLayeredArchitecture<B, S>,
    A::Error: std::fmt::Display,
    V: RoutedPartitionedArchitectureVisitor<B, S, G, W>,
{
    let prepared = prepare_partitioned::<B, S, _, _, _, _, _>(architecture, selected, partition)
        .map_err(PartitionedDispatchError::Architecture)?;
    visitor
        .visit(prepared)
        .map_err(PartitionedDispatchError::Visitor)
}

/// Family-blind backend visitor for an exact routed partitioned prediction target.
pub trait RoutedPartitionedPredictionTargetVisitor<B, S, M, G, W>
where
    B: eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::GroupedNeuralBackend
        + eredu_nn::HyperNeuralBackend,
    S: eredu_runtime::RuntimeState<B>,
    M: crate::prediction_extension::PredictionExtensionMaterializer<B>,
{
    /// Completed neutral construction output.
    type Output;
    /// Mechanism binding failure.
    type Error;

    /// Receives a routed partition only after exact extension pairing.
    fn visit<A>(
        self,
        prepared: PreparedPartitionedAdmission<
            A,
            SelectedRoutedTextRealization,
            RoutedTextRequirements,
            G,
            W,
        >,
        extension: <A as eredu_runtime::speculative::PredictionTarget<B>>::Extension<M>,
    ) -> Result<Self::Output, Self::Error>
    where
        A: eredu_runtime::PartitionedLayeredArchitecture<B, S, Boundary = W>
            + eredu_runtime::ParallelRoutedLayeredArchitecture<B, S>
            + crate::prediction_extension::MaterializedPredictionTarget<B>,
        A::Error: std::fmt::Display;
}

/// Pairs and dispatches one routed partitioned prediction target.
pub fn visit_routed_partitioned_prediction_target_architecture<B, S, M, A, G, W, V>(
    architecture: A,
    extension: crate::prediction_extension::MaterializedPredictionExtension<B, M>,
    selected: SelectedPartitionedAdmission<SelectedRoutedTextRealization, RoutedTextRequirements>,
    partition: ArchitecturePartition<G, W>,
    visitor: V,
) -> Result<V::Output, PartitionedDispatchError<V::Error>>
where
    B: eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::GroupedNeuralBackend
        + eredu_nn::HyperNeuralBackend,
    S: eredu_runtime::RuntimeState<B>,
    M: crate::prediction_extension::PredictionExtensionMaterializer<B>,
    A: eredu_runtime::PartitionedLayeredArchitecture<B, S, Boundary = W>
        + eredu_runtime::ParallelRoutedLayeredArchitecture<B, S>
        + crate::prediction_extension::MaterializedPredictionTarget<B>,
    A::Error: std::fmt::Display,
    V: RoutedPartitionedPredictionTargetVisitor<B, S, M, G, W>,
{
    let prepared = prepare_partitioned::<B, S, _, _, _, _, _>(architecture, selected, partition)
        .map_err(PartitionedDispatchError::Architecture)?;
    let extension = A::pair_prediction_extension(extension)
        .map_err(|error| PartitionedDispatchError::Architecture(error.to_string()))?;
    visitor
        .visit(prepared, extension)
        .map_err(PartitionedDispatchError::Visitor)
}

/// Backend-generic visitor for an exact composite partition.
pub trait CompositePartitionedArchitectureVisitor<B, S, G, W>
where
    B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: eredu_runtime::RuntimeState<B>,
    W: eredu_runtime::ArchitectureBoundary,
{
    /// Completed neutral construction output.
    type Output;
    /// Mechanism binding failure.
    type Error;

    /// Receives one statically known composite architecture and exact partition proof.
    fn visit<A>(
        self,
        prepared: PreparedPartitionedAdmission<
            A,
            SelectedCompositeTextRealization,
            CompositeTextRequirements,
            G,
            W,
        >,
    ) -> Result<Self::Output, Self::Error>
    where
        A: crate::composite_execution::ParallelCompositeArchitecture<B, S, Error = eredu_nn::Error>
            + eredu_runtime::PartitionedLayeredArchitecture<B, S, Boundary = W>
            + eredu_runtime::ParallelRoutedLayeredArchitecture<B, S>
            + 'static,
        A::Error: std::fmt::Display;
}

/// Pairs and dispatches one composite architecture without re-selecting its realization.
pub fn visit_composite_partitioned_architecture<B, S, A, G, W, V>(
    architecture: A,
    selected: SelectedPartitionedAdmission<
        SelectedCompositeTextRealization,
        CompositeTextRequirements,
    >,
    partition: ArchitecturePartition<G, W>,
    visitor: V,
) -> Result<V::Output, PartitionedDispatchError<V::Error>>
where
    B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: eredu_runtime::RuntimeState<B>,
    A: crate::composite_execution::ParallelCompositeArchitecture<B, S, Error = eredu_nn::Error>
        + eredu_runtime::PartitionedLayeredArchitecture<B, S, Boundary = W>
        + eredu_runtime::ParallelRoutedLayeredArchitecture<B, S>
        + 'static,
    A::Error: std::fmt::Display,
    W: eredu_runtime::ArchitectureBoundary,
    V: CompositePartitionedArchitectureVisitor<B, S, G, W>,
{
    let prepared = prepare_partitioned::<B, S, _, _, _, _, _>(architecture, selected, partition)
        .map_err(PartitionedDispatchError::Architecture)?;
    visitor
        .visit(prepared)
        .map_err(PartitionedDispatchError::Visitor)
}

/// Family-blind backend visitor for an exact composite partitioned prediction target.
pub trait CompositePartitionedPredictionTargetVisitor<B, S, M, G, W>
where
    B: eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::GroupedNeuralBackend
        + eredu_nn::HyperNeuralBackend,
    S: eredu_runtime::RuntimeState<B>,
    M: crate::prediction_extension::PredictionExtensionMaterializer<B>,
    W: eredu_runtime::ArchitectureBoundary,
{
    /// Completed neutral construction output.
    type Output;
    /// Mechanism binding failure.
    type Error;

    /// Receives a composite partition only after exact extension pairing.
    fn visit<A>(
        self,
        prepared: PreparedPartitionedAdmission<
            A,
            SelectedCompositeTextRealization,
            CompositeTextRequirements,
            G,
            W,
        >,
        extension: <A as eredu_runtime::speculative::PredictionTarget<B>>::Extension<M>,
    ) -> Result<Self::Output, Self::Error>
    where
        A: crate::composite_execution::ParallelCompositeArchitecture<B, S, Error = eredu_nn::Error>
            + eredu_runtime::PartitionedLayeredArchitecture<B, S, Boundary = W>
            + eredu_runtime::ParallelRoutedLayeredArchitecture<B, S>
            + crate::prediction_extension::MaterializedPredictionTarget<B>
            + 'static,
        A::Error: std::fmt::Display;
}

/// Pairs and dispatches one composite partitioned prediction target.
pub fn visit_composite_partitioned_prediction_target_architecture<B, S, M, A, G, W, V>(
    architecture: A,
    extension: crate::prediction_extension::MaterializedPredictionExtension<B, M>,
    selected: SelectedPartitionedAdmission<
        SelectedCompositeTextRealization,
        CompositeTextRequirements,
    >,
    partition: ArchitecturePartition<G, W>,
    visitor: V,
) -> Result<V::Output, PartitionedDispatchError<V::Error>>
where
    B: eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::GroupedNeuralBackend
        + eredu_nn::HyperNeuralBackend,
    S: eredu_runtime::RuntimeState<B>,
    M: crate::prediction_extension::PredictionExtensionMaterializer<B>,
    A: crate::composite_execution::ParallelCompositeArchitecture<B, S, Error = eredu_nn::Error>
        + eredu_runtime::PartitionedLayeredArchitecture<B, S, Boundary = W>
        + eredu_runtime::ParallelRoutedLayeredArchitecture<B, S>
        + crate::prediction_extension::MaterializedPredictionTarget<B>
        + 'static,
    A::Error: std::fmt::Display,
    W: eredu_runtime::ArchitectureBoundary,
    V: CompositePartitionedPredictionTargetVisitor<B, S, M, G, W>,
{
    let prepared = prepare_partitioned::<B, S, _, _, _, _, _>(architecture, selected, partition)
        .map_err(PartitionedDispatchError::Architecture)?;
    let extension = A::pair_prediction_extension(extension)
        .map_err(|error| PartitionedDispatchError::Architecture(error.to_string()))?;
    visitor
        .visit(prepared, extension)
        .map_err(PartitionedDispatchError::Visitor)
}
