//! Source-funded metadata and failure destinations for the shared publisher.
use super::*;
use eredu_nn::workspace::{WorkspaceMetadataFunding, WorkspaceMetadataFundingError};
use std::mem::{size_of, size_of_val};

#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
struct Failure<E: std::error::Error + 'static> {
    #[source]
    cause: E,
    funding: WorkspaceMetadataFunding,
}
pub(super) struct Controls<E> {
    funding: WorkspaceMetadataFunding,
    marker: PhantomData<fn() -> E>,
}
impl<E: std::error::Error + Send + Sync + 'static> Controls<E> {
    pub(super) fn prepare<T, C>(
        funding: &WorkspaceMetadataFunding,
    ) -> Result<Self, PartitionExecutionError> {
        let overflow = || {
            PartitionExecutionError::PublicationMetadata(WorkspaceMetadataFundingError::Overflow)
        };
        let frames = [
            size_of::<Self>(),
            size_of::<Option<Self>>(),
            size_of::<Result<Option<Self>, PartitionExecutionError>>(),
            size_of::<(T, PartitionOutputPublication, DistributedExecutionPhase)>(),
            size_of::<eredu_core::Submission<T, C>>(),
            size_of::<Result<eredu_core::Submission<T, C>, E>>(),
            size_of::<BoundedSubmissionOutcome<T>>(),
            size_of::<Result<BoundedSubmissionOutcome<T>, E>>(),
            size_of::<Result<T, PartitionExecutionError>>(),
            size_of::<Failure<E>>(),
            size_of::<eredu_core::BackendFailure>(),
            size_of::<CommunicationPoison>(),
            size_of::<std::sync::MutexGuard<'_, Option<CommunicationPoison>>>(),
            size_of::<(
                CommunicationOperation,
                DistributedExecutionPhase,
                Option<CommunicationRouteId>,
            )>(),
            size_of::<(TensorDtype, usize, Option<usize>)>(),
            size_of::<Option<(TensorDtype, usize, Option<usize>)>>(),
            CommunicationManifest::group_operation_control_bytes().ok_or_else(overflow)?,
            CommunicationOperationRequirement::tensor_metadata_control_bytes()
                .ok_or_else(overflow)?,
            eredu_core::BackendFailure::source_retention_peak_bytes::<Failure<E>>()
                .ok_or_else(overflow)?,
        ];
        funding
            .reserve_metadata(
                frames
                    .into_iter()
                    .try_fold(size_of_val(&frames), usize::checked_add)
                    .ok_or_else(overflow)?,
            )
            .map_err(PartitionExecutionError::PublicationMetadata)?;
        Ok(Self {
            funding: funding.clone(),
            marker: PhantomData,
        })
    }
    pub(super) fn validate<B: NeuralBackend, I: CommunicationTensorMetadata<B>>(
        &self,
        inspector: &I,
        value: &B::Tensor,
        requirement: &CommunicationOperationRequirement,
        completed: bool,
    ) -> Result<(), PartitionExecutionError> {
        let (dtype, rank, elements) = inspector
            .fixed_metadata_with_funding(value, &self.funding)
            .map_err(PartitionExecutionError::PublicationMetadata)?
            .ok_or(PartitionExecutionError::PreparedTensorMetadataUnavailable)?;
        requirement
            .validate_tensor_metadata(&dtype, rank, elements, completed)
            .map_err(PartitionExecutionError::PreparedTensor)
    }
    pub(super) fn failure(
        &self,
        authority: &PartitionCommunicationAuthority,
        cause: E,
        phase: DistributedExecutionPhase,
        completion: bool,
    ) -> PartitionExecutionError {
        // Same first-poison/disposition semantics as the ordinary authority.
        authority.mark_poisoned(CommunicationPoison {
            operation: CommunicationOperation::Broadcast,
            phase,
            route: None,
            cancellation: authority
                .policy
                .expect("selected publication completion")
                .cancellation(),
        });
        PartitionExecutionError::PreparedCommunication {
            operation: CommunicationOperation::Broadcast,
            phase,
            completion,
            source: eredu_core::BackendFailure::from_error(Failure {
                cause,
                funding: self.funding.clone(),
            }),
        }
    }
}

#[cfg(test)]
mod tests;
