//! Exact actual shared-driver host destinations and request identity.
use super::*;
use eredu_core::{HostPreparationAuthority, SpeculativeBuffer, SpeculativeRequestIdentity};
use eredu_nn::workspace::{HostMetadataFunding, HostMetadataFundingError};
use std::mem::{size_of, size_of_val};

pub(in crate::composition::mlx::speculative) fn buffer_bytes<T>(capacity: usize) -> Option<usize> {
    let frames = [
        size_of::<Result<SpeculativeBuffer<T>, Error>>(),
        size_of::<(usize, SpeculativeExecutionStreams<'static>)>(),
        size_of::<
            Option<(
                &crate::composition::mlx::speculative::OriginalSpeculativeNumericalSources,
                &crate::backend::OriginalCopyEnvironment<'static>,
            )>,
        >(),
    ];
    SpeculativeBuffer::<T>::retained_control_bytes(capacity)?
        .checked_add(HostPreparationAuthority::retention_bytes::<
            HostMetadataFunding,
        >()?)?
        .checked_add(size_of_val(&frames))
        .and_then(|n| frames.into_iter().try_fold(n, usize::checked_add))
}
pub(in crate::composition::mlx::speculative) fn buffer<T>(
    capacity: usize,
    context: SpeculativeExecutionStreams<'_>,
) -> Result<SpeculativeBuffer<T>, Error> {
    let Some((sources, environment)) = context.original_numerical() else {
        return Ok(SpeculativeBuffer::with_capacity(capacity));
    };
    sources.validate_environment(environment)?;
    let bytes = buffer_bytes::<T>(capacity).ok_or(Error::WorkspacePlanning(
        HostMetadataFundingError::Overflow,
    ))?;
    sources
        .metadata_funding()
        .reserve_metadata(bytes)
        .map_err(Error::WorkspacePlanning)?;
    let authority = HostPreparationAuthority::retain(sources.metadata_funding().clone());
    SpeculativeBuffer::try_new_retained(capacity, authority)
        .map_err(|cause| sources.retain_startup_error(cause))
}
pub(in crate::composition::mlx::speculative) fn identity(
    context: SpeculativeExecutionStreams<'_>,
) -> Result<SpeculativeRequestIdentity, Error> {
    let Some((sources, environment)) = context.original_numerical() else {
        return Ok(SpeculativeRequestIdentity::new());
    };
    sources.validate_environment(environment)?;
    let parts = [
        SpeculativeRequestIdentity::retained_control_bytes(),
        HostPreparationAuthority::retention_bytes::<HostMetadataFunding>(),
        Some(size_of::<Result<SpeculativeRequestIdentity, Error>>()),
        Some(size_of::<SpeculativeExecutionStreams<'static>>()),
        Some(size_of::<
            Option<(
                &crate::composition::mlx::speculative::OriginalSpeculativeNumericalSources,
                &crate::backend::OriginalCopyEnvironment<'static>,
            )>,
        >()),
    ];
    let bytes = parts
        .into_iter()
        .try_fold(size_of_val(&parts), |n, p| n.checked_add(p?))
        .ok_or(Error::WorkspacePlanning(
            HostMetadataFundingError::Overflow,
        ))?;
    sources
        .metadata_funding()
        .reserve_metadata(bytes)
        .map_err(Error::WorkspacePlanning)?;
    Ok(SpeculativeRequestIdentity::with_authority(
        HostPreparationAuthority::retain(sources.metadata_funding().clone()),
    ))
}

/// The caller supplies the exact concrete payload/shell query. This producer
/// adds only its own authority and fixed argument/error transports, then debits
/// the same source-pair account before constructing that authority.
pub(in crate::composition::mlx::speculative) fn metadata(
    bytes: Option<usize>, context: SpeculativeExecutionStreams<'_>,
) -> Result<HostPreparationAuthority, Error> {
    let Some((sources, environment)) = context.original_numerical() else {
        return Ok(HostPreparationAuthority::unmanaged());
    };
    sources.validate_environment(environment)?;
    let parts = [
        bytes,
        HostPreparationAuthority::retention_bytes::<HostMetadataFunding>(),
        Some(size_of::<Result<HostPreparationAuthority, Error>>()),
        Some(size_of::<(Option<usize>, SpeculativeExecutionStreams<'static>)>()),
        Some(size_of::<Option<(&crate::composition::mlx::speculative::OriginalSpeculativeNumericalSources,
            &crate::backend::OriginalCopyEnvironment<'static>)>>()),
    ];
    let bytes = parts.into_iter().try_fold(size_of_val(&parts), |n, p| n.checked_add(p?))
        .ok_or(Error::WorkspacePlanning(HostMetadataFundingError::Unavailable))?;
    sources.metadata_funding().reserve_metadata(bytes).map_err(Error::WorkspacePlanning)?;
    Ok(HostPreparationAuthority::retain(sources.metadata_funding().clone()))
}
