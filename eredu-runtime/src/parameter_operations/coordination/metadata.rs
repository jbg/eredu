//! Owned protocol buffers and closed error representations share the transport's payer.
use super::*;
use eredu_core::{HostMetadataFunding, HostMetadataFundingError, Submission};
use std::mem::{size_of, size_of_val};

fn reserve(funding: Option<&HostMetadataFunding>, parts: &[usize]) -> Result<(), ParameterError> {
    let bytes = parts
        .iter()
        .copied()
        .try_fold(size_of_val(parts), usize::checked_add)
        .ok_or(HostMetadataFundingError::Overflow)?;
    if let Some(funding) = funding {
        funding.reserve_metadata(bytes)?;
    }
    Ok(())
}
pub(super) fn reserve_operation<T: ParameterOperationTransport, P, O>(
    transport: &T,
    callbacks: &[usize],
) -> Result<(), ParameterError> {
    let callback_bytes = callbacks
        .iter()
        .copied()
        .try_fold(size_of_val(callbacks), usize::checked_add)
        .ok_or(HostMetadataFundingError::Overflow)?;
    reserve(
        transport.metadata_funding(),
        &[
            callback_bytes,
            size_of::<Operation<'_, T>>(),
            size_of::<P>(),
            size_of::<O>(),
            size_of::<Result<P, ParameterError>>(),
            size_of::<Result<O, ParameterError>>(),
            size_of::<ParameterError>() * 3,
            size_of::<Error>() * 3,
            size_of::<eredu_core::parameters::ParameterCoordinationFailure>(),
            size_of::<[u32; PARAMETER_CONTROL_WORDS]>(),
            size_of::<Sha256>(),
            size_of::<ParameterOperationBinding>(),
            size_of::<ParameterOperationKind>(),
            size_of::<[u8; 32]>() * 2,
            size_of::<CaptureUsage>() * 3,
            size_of::<ParameterCoordinationUsage>(),
            size_of::<BoundedCompletionWait>(),
            size_of::<Submission<T::GatherOutput, T::Completion>>(),
            size_of::<BoundedSubmissionOutcome<T::GatherOutput>>(),
            size_of::<Vec<u32>>() * 4,
            size_of::<Vec<Vec<u32>>>(),
            size_of::<std::slice::ChunksExact<'_, u32>>(),
            size_of::<Option<usize>>() * 2,
            size_of::<MutexGuard<'_, State>>(),
            size_of::<TryLockError<MutexGuard<'_, State>>>(),
            HostMetadataFunding::reservation_control_bytes(),
        ],
    )
}
pub(super) fn vector<T>(
    funding: Option<&HostMetadataFunding>,
    count: usize,
) -> Result<Vec<T>, ParameterError> {
    let bytes = count
        .checked_mul(size_of::<T>())
        .ok_or(HostMetadataFundingError::Overflow)?;
    reserve(
        funding,
        &[
            bytes,
            size_of::<Vec<T>>(),
            size_of::<Result<Vec<T>, ParameterError>>(),
            size_of::<std::collections::TryReserveError>(),
            size_of::<usize>(),
            HostMetadataFunding::reservation_control_bytes(),
        ],
    )?;
    let mut result = Vec::new();
    result
        .try_reserve_exact(count)
        .map_err(|_| HostMetadataFundingError::Unavailable)?;
    Ok(result)
}
pub(super) fn retain(
    mut error: ParameterError,
    funding: Option<&HostMetadataFunding>,
) -> ParameterError {
    if let (ParameterError::Coordination(cause), Some(funding)) = (&mut error, funding) {
        cause.retain_prepaid_metadata(funding);
    }
    error
}
