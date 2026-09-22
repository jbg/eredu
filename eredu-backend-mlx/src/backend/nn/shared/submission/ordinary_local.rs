//! The existing local callback borrows its actual Work's indexed channel.
use super::*;
use crate::backend::error::Error;
use crate::backend::runtime::residency::parameter_bank::OrdinaryIndexedLocalSource;
use eredu_nn::workspace::WorkspaceExpertRegionView;
use safemlx::error::Exception;
use std::mem::{size_of, size_of_val};

#[derive(Debug, thiserror::Error)]
#[error("ordinary local expert source differs from the admitted receive rows")]
struct LocalSourceFailure {
    _host: eredu_core::HostPreparationAuthority,
}

/// Fixed local entry transports and its retained typed refusal. Bank-channel
/// and occurrence guards are separately quoted by the indexed request owner.
/// The architecture's callback and ordinary routing vectors remain its source.
pub(crate) fn ordinary_expert_local_control_bytes() -> Option<usize> {
    let parts = [
        size_of::<Option<OrdinaryExecutionOwner>>(),
        size_of::<Option<&OrdinaryIndexedLocalSource>>(),
        size_of::<WorkspaceExpertRegionView<'static>>(),
        size_of::<(&MlxTensor, &MlxTensor, &MlxTensor, &[i32])>(),
        size_of::<(usize, Option<usize>, i32)>(),
        size_of::<Result<(), Error>>(),
        Exception::retained_source_control_bytes::<LocalSourceFailure>()?,
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}

pub(super) fn run<P, E, F>(
    source: WorkspaceExpertRegionView<'_>,
    bank: &mut P,
    input: &MlxTensor,
    scores: &MlxTensor,
    coefficients: &MlxTensor,
    indices: &[i32],
    run: F,
) -> Result<Result<eredu_runtime::RoutedExpertTensorParallelOutput<MlxTensor>, E>, Error>
where
    P: eredu_nn::Parameterized<MlxTensor>,
    F: FnOnce(&mut P) -> Result<eredu_runtime::RoutedExpertTensorParallelOutput<MlxTensor>, E>,
{
    if let Some(observer) = safemlx::OriginalScopeObserver::try_current()? {
        return Err(observer.domain_error().into());
    }
    let Some(owner) = current_ordinary_execution_owner()? else {
        return Ok(run(bank));
    };
    let Some(addressable) = source.addressable else {
        return Ok(run(bank));
    };
    let invalid = || {
        Error::from(Exception::from_retained_source(LocalSourceFailure {
            _host: owner.host().clone(),
        }))
    };
    let indexed = owner.indexed_local().ok_or_else(invalid)?;
    indexed.reserve_callback(ordinary_expert_local_control_bytes().ok_or(
        Error::WorkspacePlanning(eredu_nn::workspace::HostMetadataFundingError::Overflow),
    )?)?;
    let rows = indices.len();
    let native_rows = i32::try_from(rows).map_err(|_| invalid())?;
    if source
        .maximum_received_rows()
        .is_none_or(|maximum| rows > maximum)
        || input.as_array().shape().first().copied() != Some(native_rows)
        || scores.as_array().shape() != [native_rows, 1]
        || coefficients.as_array().shape() != [native_rows, 1]
    {
        return Err(invalid());
    }
    let local = indexed.enter_local(addressable, rows)?;
    let result = run(bank);
    local.finish(result.is_ok())?;
    Ok(result)
}
