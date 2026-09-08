//! Normalized portable policy for cold model preparation.

use std::num::NonZeroUsize;

use eredu_checkpoint::{AffineQuantization, WeightQuantization};
use eredu_core::{
    CompletionCancellationMode, DraftingPlan, ParallelRankTopology, PreparationPolicy,
    QuantizationRequest, ResidencyRequest, SessionCapabilities,
};

use crate::{
    CacheResidencyPolicy, CommunicationCompletionPolicy, LayerWeightResidency,
    PipelineWireContract, WeightResidency,
};

/// Portable drafting intent resolved before checkpoint payload selection.
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
pub enum DraftingLoadRequest {
    /// Preserve direct-load behavior by installing an admitted embedded extension.
    #[default]
    ArchitectureDefault,
    /// Materialize only the ordinary target.
    Disabled,
    /// Require and install an embedded extension with this proposal bound.
    Embedded {
        /// Maximum number of draft tokens proposed per target step.
        max_draft_tokens: NonZeroUsize,
    },
    /// Prepare only the ordinary target for later external-assistant pairing.
    ExternalTarget,
}

impl DraftingLoadRequest {
    /// Creates an embedded drafting request with a positive proposal capacity.
    pub fn embedded(max_draft_tokens: usize) -> Result<Self, NormalizedLoadRequestError> {
        let max_draft_tokens = NonZeroUsize::new(max_draft_tokens)
            .ok_or(NormalizedLoadRequestError::ZeroEmbeddedDraftCapacity)?;
        Ok(Self::Embedded { max_draft_tokens })
    }

    /// Returns the selected embedded proposal capacity, when requested explicitly.
    pub const fn embedded_capacity(self) -> Option<NonZeroUsize> {
        match self {
            Self::Embedded { max_draft_tokens } => Some(max_draft_tokens),
            Self::ArchitectureDefault | Self::Disabled | Self::ExternalTarget => None,
        }
    }
}

/// Portable topology, wire, and invocation geometry selected as one atomic request.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct ParallelLoadRequest {
    rank: ParallelRankTopology,
    wire: PipelineWireContract,
    maximum_batch_size: i32,
    maximum_sequence_length: i32,
    completion: CommunicationCompletionPolicy,
}

impl ParallelLoadRequest {
    /// Binds a rank, activation wire, and maximum invocation geometry.
    pub fn new(
        rank: ParallelRankTopology,
        wire: PipelineWireContract,
        maximum_batch_size: i32,
        maximum_sequence_length: i32,
        completion: CommunicationCompletionPolicy,
    ) -> Result<Self, NormalizedLoadRequestError> {
        if rank.is_replicated() {
            return Err(NormalizedLoadRequestError::ReplicatedParallelTopology);
        }
        if maximum_batch_size <= 0 || maximum_sequence_length <= 0 {
            return Err(NormalizedLoadRequestError::InvalidInvocationLimits {
                maximum_batch_size,
                maximum_sequence_length,
            });
        }
        Ok(Self {
            rank,
            wire,
            maximum_batch_size,
            maximum_sequence_length,
            completion,
        })
    }

    /// Returns the selected portable rank.
    pub const fn rank(self) -> ParallelRankTopology {
        self.rank
    }

    /// Returns the exact pipeline activation wire contract.
    pub const fn wire(self) -> PipelineWireContract {
        self.wire
    }

    /// Returns the maximum admitted batch and sequence geometry.
    pub const fn invocation_limits(self) -> (i32, i32) {
        (self.maximum_batch_size, self.maximum_sequence_length)
    }

    /// Returns the bounded completion policy selected for this parallel request.
    pub const fn completion(self) -> CommunicationCompletionPolicy {
        self.completion
    }

    const fn with_completion(mut self, completion: CommunicationCompletionPolicy) -> Self {
        self.completion = completion;
        self
    }
}

/// Failure while validating or lowering a normalized cold-load request.
#[derive(Debug, Clone, thiserror::Error)]
#[non_exhaustive]
pub enum NormalizedLoadRequestError {
    /// A prepared source must have a positive reader-cache bound.
    #[error("source reader-cache limit must be positive")]
    ZeroCachedShards,
    /// Parallel invocation limits must both be positive.
    #[error(
        "partitioned invocation limits must be positive, got batch {maximum_batch_size} and sequence {maximum_sequence_length}"
    )]
    InvalidInvocationLimits {
        /// Maximum admitted batch size.
        maximum_batch_size: i32,
        /// Maximum admitted sequence length.
        maximum_sequence_length: i32,
    },
    /// A replicated topology was incorrectly attached as parallel execution.
    #[error("explicit parallel execution requires a non-replicated topology")]
    ReplicatedParallelTopology,
    /// A local completion setting was attached before a complete parallel request.
    #[error("parallel execution cannot replace an existing local completion policy")]
    ConflictingCompletionPolicy,
    /// An ordinary model request retained a communication-only policy.
    #[error("a communication completion policy requires a parallel topology")]
    OrphanedModelCompletion,
    /// Embedded drafting requires positive capacity.
    #[error("embedded draft capacity must be positive")]
    ZeroEmbeddedDraftCapacity,
    /// The portable planner supplied a drafting mode this runtime does not understand.
    #[error("unsupported speculative drafting plan")]
    UnsupportedDraftingPlan,
    /// The default realtime completion contract could not be constructed.
    #[error("{0}")]
    Completion(String),
    /// A quantization request could not be represented by the checkpoint contract.
    #[error("{0}")]
    Quantization(String),
}

/// One normalized backend-neutral request for cold model preparation.
///
/// Parallel rank, wire, and invocation fields are represented atomically. A concrete backend
/// may pair this value with a separate native device or resource token, but must not add native
/// identity to this request.
#[derive(Debug, Clone, Default, Eq, PartialEq)]
pub struct NormalizedLoadRequest {
    quantization: Option<QuantizationRequest>,
    max_cached_shards: Option<NonZeroUsize>,
    parallel: Option<ParallelLoadRequest>,
    communication_completion: Option<CommunicationCompletionPolicy>,
    weight_residency: WeightResidency,
    state_residency: CacheResidencyPolicy,
    required_session_capabilities: SessionCapabilities,
    prompt_cache_persistence: bool,
    drafting: DraftingLoadRequest,
}

/// A normalized model request whose cross-field invariants were checked once.
#[derive(Debug, Clone, Copy)]
pub struct ValidatedModelLoadRequest<'a> {
    request: &'a NormalizedLoadRequest,
    policy: PreparationPolicy,
}

impl<'a> ValidatedModelLoadRequest<'a> {
    /// Exact portable preparation policy projected during validation.
    pub const fn preparation_policy(self) -> PreparationPolicy {
        self.policy
    }

    /// Original normalized request covered by this validation proof.
    pub const fn request(self) -> &'a NormalizedLoadRequest {
        self.request
    }
}

impl NormalizedLoadRequest {
    /// Requires persisted prompt-prefix import/export independently of state residency.
    pub const fn with_prompt_cache_persistence(mut self, required: bool) -> Self {
        self.prompt_cache_persistence = required;
        self
    }

    /// Returns explicit persisted prompt-prefix import/export intent.
    pub const fn prompt_cache_persistence(&self) -> bool {
        self.prompt_cache_persistence
    }

    /// Selects the exact reader-cache limit, including fully resident execution.
    pub const fn with_max_cached_shards(mut self, maximum: NonZeroUsize) -> Self {
        self.max_cached_shards = Some(maximum);
        self
    }

    /// Returns the source reader-cache bound retained by cold selection.
    pub const fn max_cached_shards(&self) -> usize {
        match self.max_cached_shards {
            Some(maximum) => maximum.get(),
            None => self.weight_residency.max_cached_shards(),
        }
    }

    /// Creates a request that quantizes eligible dense weights on load.
    pub fn with_quantization(quantization: QuantizationRequest) -> Self {
        Self {
            quantization: Some(quantization),
            ..Self::default()
        }
    }

    /// Attaches one complete portable parallel-execution request.
    pub fn with_parallel_execution(
        mut self,
        parallel: ParallelLoadRequest,
    ) -> Result<Self, NormalizedLoadRequestError> {
        if self.communication_completion.is_some() {
            return Err(NormalizedLoadRequestError::ConflictingCompletionPolicy);
        }
        self.parallel = Some(parallel);
        Ok(self)
    }

    /// Selects the bounded completion policy used by communication or local realtime work.
    pub const fn with_communication_completion_policy(
        mut self,
        policy: CommunicationCompletionPolicy,
    ) -> Self {
        self.set_communication_completion_policy(policy);
        self
    }

    /// Replaces the bounded completion policy without changing another request field.
    pub const fn set_communication_completion_policy(
        &mut self,
        policy: CommunicationCompletionPolicy,
    ) {
        match self.parallel {
            Some(parallel) => self.parallel = Some(parallel.with_completion(policy)),
            None => self.communication_completion = Some(policy),
        }
    }

    /// Selects fully resident or bounded checkpoint-weight execution.
    pub fn with_weight_residency(mut self, residency: WeightResidency) -> Self {
        self.weight_residency = residency;
        self
    }

    /// Selects mutable-state residency and paging controls.
    pub fn with_state_residency(mut self, residency: CacheResidencyPolicy) -> Self {
        self.state_residency = residency;
        self
    }

    /// Requires capabilities from the exact inspected and realized session.
    pub const fn with_required_session_capabilities(
        mut self,
        capabilities: SessionCapabilities,
    ) -> Self {
        self.required_session_capabilities = capabilities;
        self
    }

    /// Selects portable drafting intent.
    pub const fn with_drafting(mut self, drafting: DraftingLoadRequest) -> Self {
        self.drafting = drafting;
        self
    }

    /// Applies a portable execution plan's drafting mode before payload selection.
    pub fn with_drafting_plan(
        self,
        plan: &DraftingPlan,
    ) -> Result<Self, NormalizedLoadRequestError> {
        let drafting = match plan {
            DraftingPlan::Disabled => DraftingLoadRequest::Disabled,
            DraftingPlan::Embedded {
                max_draft_tokens, ..
            } => DraftingLoadRequest::embedded(*max_draft_tokens)?,
            DraftingPlan::External { .. } => DraftingLoadRequest::ExternalTarget,
            _ => return Err(NormalizedLoadRequestError::UnsupportedDraftingPlan),
        };
        Ok(self.with_drafting(drafting))
    }

    /// Returns the requested dense-weight transformation.
    pub const fn quantization(&self) -> Option<QuantizationRequest> {
        self.quantization
    }

    /// Returns the complete portable parallel request.
    pub const fn parallel_execution(&self) -> Option<ParallelLoadRequest> {
        self.parallel
    }

    /// Returns the selected portable rank, when parallel execution was attached.
    pub const fn parallel_topology(&self) -> Option<ParallelRankTopology> {
        match self.parallel {
            Some(parallel) => Some(parallel.rank()),
            None => None,
        }
    }

    /// Returns the activation wire contract for parallel execution.
    pub const fn pipeline_wire_contract(&self) -> Option<PipelineWireContract> {
        match self.parallel {
            Some(parallel) => Some(parallel.wire()),
            None => None,
        }
    }

    /// Reports whether a portable parallel execution request is attached.
    pub const fn has_parallel_execution(&self) -> bool {
        self.parallel.is_some()
    }

    /// Returns invocation limits already validated by parallel construction.
    pub fn partitioned_invocation_limits(&self) -> Option<(i32, i32)> {
        self.parallel.map(ParallelLoadRequest::invocation_limits)
    }

    /// Returns bounded communication completion for model preparation.
    pub fn communication_completion_policy(
        &self,
    ) -> Result<Option<CommunicationCompletionPolicy>, NormalizedLoadRequestError> {
        match (self.parallel, self.communication_completion) {
            (Some(parallel), None) => Ok(Some(parallel.completion())),
            (None, None) => Ok(None),
            (None, Some(_)) => Err(NormalizedLoadRequestError::OrphanedModelCompletion),
            (Some(_), Some(_)) => Err(NormalizedLoadRequestError::ConflictingCompletionPolicy),
        }
    }

    /// Returns completion policy for realtime work, including local async evaluation.
    pub fn realtime_completion_policy(
        &self,
    ) -> Result<CommunicationCompletionPolicy, NormalizedLoadRequestError> {
        if let Some(parallel) = self.parallel {
            return Ok(parallel.completion());
        }
        self.communication_completion.map_or_else(
            || {
                CommunicationCompletionPolicy::new(
                    std::time::Duration::from_secs(30),
                    CompletionCancellationMode::QuarantineUntilComplete,
                )
                .map_err(|error| NormalizedLoadRequestError::Completion(error.to_string()))
            },
            Ok,
        )
    }

    /// Returns selected immutable-weight residency.
    pub const fn weight_residency(&self) -> WeightResidency {
        self.weight_residency
    }

    /// Returns selected mutable-state residency.
    pub const fn state_residency(&self) -> &CacheResidencyPolicy {
        &self.state_residency
    }

    /// Returns capabilities required from the realized session.
    pub const fn required_session_capabilities(&self) -> SessionCapabilities {
        self.required_session_capabilities
    }

    /// Returns the selected pre-payload drafting intent.
    pub const fn drafting(&self) -> DraftingLoadRequest {
        self.drafting
    }

    /// Converts the requested transformation to the checkpoint lowering vocabulary.
    pub fn weight_quantization(
        &self,
    ) -> Result<Option<WeightQuantization>, NormalizedLoadRequestError> {
        self.quantization
            .map(|request| match request {
                QuantizationRequest::Affine { group_size, bits } => {
                    let group_size = i32::try_from(group_size).map_err(|_| {
                        NormalizedLoadRequestError::Quantization(format!(
                            "group_size must fit in i32, got {group_size}"
                        ))
                    })?;
                    AffineQuantization::new(group_size, i32::from(bits))
                        .map(WeightQuantization::Affine)
                        .map_err(|error| {
                            NormalizedLoadRequestError::Quantization(error.to_string())
                        })
                }
                QuantizationRequest::MxFp4 => Ok(WeightQuantization::MxFp4),
                _ => Err(NormalizedLoadRequestError::Quantization(
                    "unknown load-time transformation request".into(),
                )),
            })
            .transpose()
    }

    /// Validates the complete ordinary model-preparation request.
    pub fn validate_model_preparation(
        &self,
    ) -> Result<ValidatedModelLoadRequest<'_>, NormalizedLoadRequestError> {
        if self.max_cached_shards() == 0 {
            return Err(NormalizedLoadRequestError::ZeroCachedShards);
        }
        self.weight_quantization()?;
        self.communication_completion_policy()?;
        Ok(ValidatedModelLoadRequest {
            request: self,
            policy: self.project_preparation_policy(),
        })
    }

    /// Converts this normalized request into core's portable preparation policy.
    pub fn preparation_policy(&self) -> Result<PreparationPolicy, NormalizedLoadRequestError> {
        self.validate_model_preparation()
            .map(ValidatedModelLoadRequest::preparation_policy)
    }

    fn project_preparation_policy(&self) -> PreparationPolicy {
        let residency = if self.weight_residency.parameter_bank_cache().is_some() {
            ResidencyRequest::AddressableParameterBanks
        } else {
            match self.weight_residency.layers() {
                LayerWeightResidency::FullyResident => ResidencyRequest::FullyResident,
                LayerWeightResidency::LayerwiseHost(_) => ResidencyRequest::LayerwiseHost,
                LayerWeightResidency::DenseDiskStream(_) => ResidencyRequest::DenseDiskStream,
            }
        };
        let mut policy = PreparationPolicy::new(self.quantization, residency)
            .with_required_session_capabilities(self.required_session_capabilities);
        if let Some(topology) = self.parallel_topology() {
            policy = policy.with_topology(topology.topology());
        }
        policy
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_core::{ParallelTopology, QuantizationRequest};

    fn completion() -> CommunicationCompletionPolicy {
        CommunicationCompletionPolicy::new(
            std::time::Duration::from_secs(1),
            CompletionCancellationMode::QuarantineUntilComplete,
        )
        .unwrap()
    }

    #[test]
    fn parallel_policy_is_atomic_and_exact() {
        let rank =
            ParallelRankTopology::new(ParallelTopology::new(2, 1, 1, 1).unwrap(), 1).unwrap();
        let parallel = ParallelLoadRequest::new(
            rank,
            PipelineWireContract::new(crate::PipelineActivationDtype::Float32),
            2,
            128,
            completion(),
        )
        .unwrap();
        let request = NormalizedLoadRequest::with_quantization(QuantizationRequest::MxFp4)
            .with_parallel_execution(parallel)
            .unwrap()
            .with_required_session_capabilities(SessionCapabilities::new(true, false, true));

        request.validate_model_preparation().unwrap();
        assert_eq!(request.parallel_execution(), Some(parallel));
        assert_eq!(request.parallel_topology(), Some(rank));
        assert_eq!(request.partitioned_invocation_limits(), Some((2, 128)));
        assert_eq!(
            request.communication_completion_policy().unwrap(),
            Some(completion())
        );
        assert_eq!(
            request.preparation_policy().unwrap().topology(),
            Some(rank.topology())
        );
    }

    #[test]
    fn invalid_parallel_geometry_fails_at_parallel_construction() {
        let rank =
            ParallelRankTopology::new(ParallelTopology::new(2, 1, 1, 1).unwrap(), 0).unwrap();
        let request = ParallelLoadRequest::new(
            rank,
            PipelineWireContract::new(crate::PipelineActivationDtype::Float32),
            0,
            128,
            completion(),
        );
        assert!(matches!(
            request,
            Err(NormalizedLoadRequestError::InvalidInvocationLimits { .. })
        ));

        let request = ParallelLoadRequest::new(
            rank,
            PipelineWireContract::new(crate::PipelineActivationDtype::Float32),
            1,
            -1,
            completion(),
        );
        assert!(matches!(
            request,
            Err(NormalizedLoadRequestError::InvalidInvocationLimits { .. })
        ));
    }

    #[test]
    fn replicated_topology_is_rejected_at_parallel_construction() {
        let rank =
            ParallelRankTopology::new(ParallelTopology::new(1, 1, 1, 1).unwrap(), 0).unwrap();
        assert!(matches!(
            ParallelLoadRequest::new(
                rank,
                PipelineWireContract::new(crate::PipelineActivationDtype::Float32),
                1,
                128,
                completion(),
            ),
            Err(NormalizedLoadRequestError::ReplicatedParallelTopology)
        ));
    }

    #[test]
    fn local_completion_cannot_be_reinterpreted_as_parallel_completion() {
        let rank =
            ParallelRankTopology::new(ParallelTopology::new(2, 1, 1, 1).unwrap(), 0).unwrap();
        let parallel = ParallelLoadRequest::new(
            rank,
            PipelineWireContract::new(crate::PipelineActivationDtype::Float32),
            1,
            128,
            completion(),
        )
        .unwrap();
        let request =
            NormalizedLoadRequest::default().with_communication_completion_policy(completion());
        assert!(matches!(
            request.with_parallel_execution(parallel),
            Err(NormalizedLoadRequestError::ConflictingCompletionPolicy)
        ));
    }

    #[test]
    fn embedded_drafting_capacity_is_positive_by_construction() {
        assert!(matches!(
            DraftingLoadRequest::embedded(0),
            Err(NormalizedLoadRequestError::ZeroEmbeddedDraftCapacity)
        ));
        assert_eq!(
            DraftingLoadRequest::embedded(4)
                .unwrap()
                .embedded_capacity()
                .unwrap()
                .get(),
            4
        );
    }

    #[test]
    fn local_realtime_completion_does_not_become_valid_model_communication() {
        let request =
            NormalizedLoadRequest::default().with_communication_completion_policy(completion());
        assert_eq!(request.realtime_completion_policy().unwrap(), completion());
        assert!(matches!(
            request.validate_model_preparation(),
            Err(NormalizedLoadRequestError::OrphanedModelCompletion)
        ));
    }
}
