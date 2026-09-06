//! MLX adapter over the normalized portable preparation request.

use crate::backend::error::Error;
use eredu_runtime::{
    NormalizedLoadRequest, NormalizedLoadRequestError, ParallelLoadRequest, PipelineWireContract,
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

    /// Adds a validated MLX parallel topology and its activation wire contract.
    pub fn with_parallel_topology(
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
