//! Shared source-funded tensor validation and communication failure destinations.
use super::*;
use eredu_nn::workspace::{HostMetadataFunding, HostMetadataFundingError};
use std::mem::{size_of, size_of_val};

#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
struct Failure<E: std::error::Error + 'static> {
    #[source]
    cause: E,
    funding: HostMetadataFunding,
}
#[derive(Debug, thiserror::Error)]
pub(super) enum WaveCause {
    #[error(transparent)]
    Native(eredu_core::BackendFailure),
    #[error(transparent)]
    Contract(PartitionExecutionError),
}
pub(super) struct Controls<E> {
    funding: HostMetadataFunding,
    operation: CommunicationOperation,
    marker: PhantomData<fn() -> E>,
}
impl<E: std::error::Error + Send + Sync + 'static> Controls<E> {
    pub(super) fn prepare<T>(
        funding: &HostMetadataFunding, operation: CommunicationOperation, caller_controls: usize,
    ) -> Result<Self, PartitionExecutionError> {
        let overflow = || {
            PartitionExecutionError::PublicationMetadata(HostMetadataFundingError::Overflow)
        };
        let frames = [
            size_of::<Self>(),
            size_of::<Option<Self>>(),
            size_of::<Result<Option<Self>, PartitionExecutionError>>(),
            caller_controls,
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
            operation,
            marker: PhantomData,
        })
    }
    pub(super) fn same_account(&self, funding: &HostMetadataFunding) -> bool {
        self.funding.same_account(funding)
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
            operation: self.operation,
            phase,
            route: None,
            cancellation: authority
                .policy
                .expect("selected publication completion")
                .cancellation(),
        });
        PartitionExecutionError::PreparedCommunication {
            operation: self.operation,
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
