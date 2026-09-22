//! Synchronous cache evaluation under the actual admitted ordinary Work.
use crate::backend::nn::{
    shared::{OrdinaryExecutionOwner, current_ordinary_execution_owner},
    workspace::{OrdinaryCallControls, OrdinaryNativeControls},
};
use eredu_core::HostPreparationAuthority;
use safemlx::{Array, error::Exception, ops::OrdinaryRecipeCall};
use std::mem::{size_of, size_of_val};

#[derive(Debug, thiserror::Error)]
#[error("cache evaluation failed: {cause}")]
struct Failure {
    #[source]
    cause: Exception,
    _host: HostPreparationAuthority,
}

/// Uses the synchronous worker already selected by ordinary sealing and scans.
/// Lookup authenticates the current physical observer; the host loan adds no
/// submission authority. The enclosing Work remains responsible for recovery.
pub(crate) fn evaluate_cache_arrays<const N: usize>(arrays: [&Array; N]) -> Result<(), Exception> {
    let owner = current_ordinary_execution_owner()?;
    safemlx::transforms::eval(arrays).map_err(|cause| match owner {
        Some(owner) => Exception::from_retained_source(Failure {
            cause,
            _host: owner.host().clone(),
        }),
        None => cause,
    })
}

/// Actual synchronous ArrayVector caller and its retained error transport.
/// Manager catalogs, block metadata, native Eval and recovery are separate
/// sources; this query does not qualify any of those populations.
pub(crate) fn ordinary_cache_evaluation_call_controls(
    roots: usize,
) -> Option<OrdinaryCallControls> {
    let call = OrdinaryRecipeCall::Evaluate { inputs: roots }.control_bytes()?;
    let frames = [
        size_of::<Option<OrdinaryExecutionOwner>>(),
        size_of::<Result<Option<OrdinaryExecutionOwner>, Exception>>(),
        size_of::<Failure>(),
        size_of::<HostPreparationAuthority>(),
        size_of::<Result<(), Exception>>(),
        Exception::retained_source_control_bytes::<Failure>()?,
    ];
    let bytes = frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)?;
    let mut observed = OrdinaryNativeControls::default();
    if let Some(source) = call.observed_controls() {
        observed.include(source)?;
    }
    Some(OrdinaryCallControls {
        metadata_bytes: u64::try_from(call.metadata_bytes())
            .ok()?
            .checked_add(u64::try_from(bytes).ok()?)?,
        observed,
    })
}
