//! MLX adapter over the normalized portable preparation request.

use eredu_core::QuantizationRequest;

use crate::backend::error::Error;
use eredu_runtime::{
    NormalizedLoadRequest, NormalizedLoadRequestError, ParallelLoadRequest, PipelineWireContract,
    WeightResidency,
};

/// Caller request translated into an authoritative realization before materialization.
#[derive(Debug, Clone, Default, Eq, PartialEq)]
pub struct MlxLoadRequest {
    /// Complete backend-neutral policy supplied to architecture selection.
    normalized: NormalizedLoadRequest,
    /// Optional native device paired with the normalized portable rank.
    parallel_device: Option<crate::backend::DeviceAssignment>,
}

impl MlxLoadRequest {
    /// Wraps one complete portable request without adding a native target.
    pub fn from_normalized(normalized: NormalizedLoadRequest) -> Self {
        Self {
            normalized,
            parallel_device: None,
        }
    }

    /// Returns the singular portable policy wrapped by this native adapter.
    pub const fn normalized(&self) -> &NormalizedLoadRequest {
        &self.normalized
    }

    #[cfg(test)]
    pub(crate) fn test_communication_completion_policy(
    ) -> eredu_runtime::CommunicationCompletionPolicy {
        eredu_runtime::CommunicationCompletionPolicy::new(
            std::time::Duration::from_secs(30),
            eredu_core::CompletionCancellationMode::QuarantineUntilComplete,
        )
        .expect("test completion policy is positive")
    }

    /// Creates load options that quantize eligible dense weights on load.
    pub fn with_quantization(quantization: QuantizationRequest) -> Self {
        Self {
            normalized: NormalizedLoadRequest::with_quantization(quantization),
            parallel_device: None,
        }
    }

    /// Adds a validated MLX parallel topology and its activation wire contract.
    pub(crate) fn with_parallel_topology(
        mut self,
        topology: eredu_core::ParallelRankTopology,
        device: crate::backend::DeviceAssignment,
        pipeline_wire: PipelineWireContract,
        maximum_batch_size: i32,
        maximum_sequence_length: i32,
        completion_policy: eredu_runtime::CommunicationCompletionPolicy,
    ) -> Result<Self, Error> {
        let parallel = ParallelLoadRequest::new(
            topology,
            pipeline_wire,
            maximum_batch_size,
            maximum_sequence_length,
            completion_policy,
        )
        .map_err(normalized_request_error)?;
        self.normalized = self
            .normalized
            .with_parallel_execution(parallel)
            .map_err(normalized_request_error)?;
        self.parallel_device = Some(device);
        Ok(self)
    }

    /// Creates load options for a validated MLX parallel topology and
    /// activation wire contract.
    pub(crate) fn with_parallel(
        topology: eredu_core::ParallelRankTopology,
        device: crate::backend::DeviceAssignment,
        pipeline_wire: PipelineWireContract,
        maximum_batch_size: i32,
        maximum_sequence_length: i32,
        completion_policy: eredu_runtime::CommunicationCompletionPolicy,
    ) -> Result<Self, Error> {
        Self::default().with_parallel_topology(
            topology,
            device,
            pipeline_wire,
            maximum_batch_size,
            maximum_sequence_length,
            completion_policy,
        )
    }

    /// Selects the bounded completion policy for every native communication
    /// operation in the distributed session.
    pub const fn with_communication_completion_policy(
        mut self,
        policy: eredu_runtime::CommunicationCompletionPolicy,
    ) -> Self {
        self.normalized.set_communication_completion_policy(policy);
        self
    }

    /// Selects fully resident or bounded layer execution for checkpoint weights.
    pub fn with_weight_residency(mut self, residency: WeightResidency) -> Self {
        self.normalized = self.normalized.with_weight_residency(residency);
        self
    }

    /// Selects the exact mutable-state residency and paging controls.
    pub fn with_state_residency(mut self, residency: eredu_runtime::CacheResidencyPolicy) -> Self {
        self.normalized = self.normalized.with_state_residency(residency);
        self
    }

    /// Requires capabilities from the exact inspected and realized session.
    pub fn with_required_session_capabilities(
        mut self,
        capabilities: eredu_core::SessionCapabilities,
    ) -> Self {
        self.normalized = self
            .normalized
            .with_required_session_capabilities(capabilities);
        self
    }

    /// Applies the execution plan's drafting mode before target payload selection.
    pub(crate) fn with_drafting_plan(
        mut self,
        plan: &eredu_core::DraftingPlan,
    ) -> Result<Self, Error> {
        self.normalized = self
            .normalized
            .with_drafting_plan(plan)
            .map_err(normalized_request_error)?;
        Ok(self)
    }

    /// Returns the selected distributed topology, if any.
    pub(crate) const fn parallel_topology(&self) -> Option<eredu_core::ParallelRankTopology> {
        self.normalized.parallel_topology()
    }

    pub(crate) fn parallel_rank_context(
        &self,
    ) -> Result<Option<crate::backend::MlxRankContext>, Error> {
        self.checked_normalized().map(|(_, rank)| rank)
    }

    /// Returns the normalized request atomically paired with its validated MLX rank token.
    pub(crate) fn checked_normalized(
        &self,
    ) -> Result<
        (
            &NormalizedLoadRequest,
            Option<crate::backend::MlxRankContext>,
        ),
        Error,
    > {
        match (self.normalized.parallel_topology(), self.parallel_device) {
            (Some(topology), Some(device)) => crate::backend::MlxRankContext::new(
                topology.world_size(),
                topology.global_rank(),
                device,
            )
            .map(|rank| (&self.normalized, Some(rank))),
            (None, None) => Ok((&self.normalized, None)),
            (Some(_), None) => Err(Error::Parallel(
                "parallel execution has no MLX device assignment".into(),
            )),
            (None, Some(_)) => Err(Error::Parallel(
                "MLX device assignment requires a parallel topology".into(),
            )),
        }
    }

    /// Returns the activation wire contract paired with the distributed
    /// topology, if any.
    pub const fn pipeline_wire_contract(&self) -> Option<PipelineWireContract> {
        self.normalized.pipeline_wire_contract()
    }

    /// Reports whether composition attached a native parallel execution plan.
    pub const fn has_parallel_execution(&self) -> bool {
        self.normalized.has_parallel_execution()
    }

    /// Returns the requested dense-weight transformation, if any.
    pub const fn quantization(&self) -> Option<QuantizationRequest> {
        self.normalized.quantization()
    }

    /// Returns the selected immutable-weight residency policy.
    pub const fn weight_residency(&self) -> WeightResidency {
        self.normalized.weight_residency()
    }

    /// Returns the selected mutable-state residency policy.
    pub const fn state_residency(&self) -> &eredu_runtime::CacheResidencyPolicy {
        self.normalized.state_residency()
    }

    /// Returns the capabilities required from the realized session.
    pub const fn required_session_capabilities(&self) -> eredu_core::SessionCapabilities {
        self.normalized.required_session_capabilities()
    }

    /// Converts these MLX load options into the portable preparation policy.
    pub fn preparation_policy(&self) -> Result<eredu_core::PreparationPolicy, Error> {
        self.checked_normalized()?
            .0
            .preparation_policy()
            .map_err(normalized_request_error)
    }
}

fn normalized_request_error(error: NormalizedLoadRequestError) -> Error {
    let message = error.to_string();
    match error {
        NormalizedLoadRequestError::Quantization(message) => Error::Quantization(message),
        NormalizedLoadRequestError::ZeroEmbeddedDraftCapacity
        | NormalizedLoadRequestError::UnsupportedDraftingPlan => Error::AutomaticPlanning(message),
        _ => Error::Parallel(message),
    }
}

#[cfg(test)]
mod tests;
