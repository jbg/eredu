//! Typed source-funded error handoff for the shared Embedded executor.
use super::*;
use eredu_nn::workspace::{
    WorkspaceMetadataError, HostMetadataFunding, HostMetadataFundingError,
};
use std::mem::{size_of, size_of_val};

#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
struct NeuralCause<E: std::error::Error + Send + Sync + 'static> {
    #[source]
    cause: E,
    // The closed NN source deallocates its shell before this payload. Its
    // concrete cause is destroyed before the last planning account alias.
    _funding: HostMetadataFunding,
}
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
struct Diagnostic(String);
#[derive(Debug, thiserror::Error)]
#[error("prediction observation failed")]
struct ObservationMarker;

fn transport_controls<E>(
    funding: &HostMetadataFunding,
) -> Result<(), HostMetadataFundingError> {
    let controls = [
        size_of::<E>(),
        size_of::<Option<E>>(),
        size_of::<&mut dyn std::any::Any>(),
        size_of::<SpeculativeExecutionStreams<'_>>(),
        size_of::<Error>(),
        size_of::<Result<eredu_core::BackendFailure, Error>>(),
        size_of::<HostMetadataFunding>(),
        size_of::<HostMetadataFundingError>(),
        size_of::<Result<(), HostMetadataFundingError>>(),
        size_of::<Option<usize>>(),
    ];
    let bytes = controls
        .into_iter()
        .try_fold(size_of_val(&controls), usize::checked_add)
        .ok_or(HostMetadataFundingError::Overflow)?;
    funding.reserve_metadata(bytes)
}

// Retains the same concrete E through the existing shared diagnostic producer.
// No E or operation result is stored in this small pre-operation receipt.
fn prepare_session_funding<E: std::error::Error + Send + Sync + 'static>(
    funding: &HostMetadataFunding,
    additional_controls: Option<usize>,
) -> Result<crate::composition::mlx::model::PreparedPlanningError<E>, HostMetadataFundingError> {
    use crate::composition::mlx::model::PreparedPlanningError;
    let additional_controls = additional_controls.ok_or(HostMetadataFundingError::Overflow)?;
    if additional_controls != 0 {
        funding.reserve_metadata(additional_controls)?;
    }
    transport_controls::<E>(funding)?;
    funding.reserve_metadata(size_of::<(
        Option<PreparedPlanningError<E>>,
        Result<Option<PreparedPlanningError<E>>, Error>,
        &HostMetadataFunding,
    )>())?;
    PreparedPlanningError::prepare(funding)
}

pub(super) fn prepare_session_cause<E: std::error::Error + Send + Sync + 'static>(
    context: SpeculativeExecutionStreams<'_>,
    additional_controls: Option<usize>,
) -> Result<impl FnOnce(E) -> Error, Error> {
    let prepared = match context.original_numerical() {
        Some((sources, _)) => Some(prepare_session_funding::<E>(sources.metadata_funding(), additional_controls)
            .map_err(Error::WorkspacePlanning)?),
        None => None,
    };
    let convert = move |cause: E| match prepared {
        Some(prepared) => prepared.retain(cause),
        None => ordinary_session_cause(cause),
    };
    // The callback owns only its receipt. It never wraps the model operation,
    // its success value or mutable state, so those construction frames stay out
    // of this transport. The same native operation control census sees its size.
    if let Some((sources, _)) = context.original_numerical() {
        sources.metadata_funding().reserve_metadata(
            conversion_control_bytes(&convert)
                .ok_or(Error::WorkspacePlanning(HostMetadataFundingError::Overflow))?,
        ).map_err(Error::WorkspacePlanning)?;
    }
    Ok(convert)
}

fn conversion_control_bytes<F>(_: &F) -> Option<usize> {
    size_of::<F>().checked_add(size_of::<Result<F, Error>>())
        .and_then(|bytes| bytes.checked_add(size_of::<(&F, Option<usize>)>()))
}
fn ordinary_session_cause(cause: impl std::fmt::Display) -> Error {
    Error::Exception(Exception::custom(cause.to_string()))
}

pub(super) fn session_cause<E: std::error::Error + Send + Sync + 'static>(
    cause: E,
    context: SpeculativeExecutionStreams<'_>,
) -> Error {
    let Some((sources, _)) = context.original_numerical() else {
        return ordinary_session_cause(cause);
    };
    if let Err(refusal) = transport_controls::<E>(sources.metadata_funding()) {
        drop(cause);
        return Error::WorkspacePlanning(refusal);
    }
    // Moving a known existing envelope requires no Box or diagnostic String.
    // The temporary Option permits a safe owned downcast without heap erasure.
    let mut cause = Some(cause);
    let erased = &mut cause as &mut dyn std::any::Any;
    if let Some(cause) = erased.downcast_mut::<Option<Error>>() {
        return sources.retain_error(cause.take().expect("owned native cause"));
    }
    if let Some(cause) = erased.downcast_mut::<Option<eredu_core::BackendFailure>>() {
        return Error::StorageSource(cause.take().expect("owned retained cause"));
    }
    if let Some(cause) = erased.downcast_mut::<Option<Exception>>() {
        return Error::Exception(cause.take().expect("owned native exception"));
    }
    // This existing worker pays only its concrete error shell. If funding
    // refuses, it retires the cause safely and returns exact inline funding.
    sources.retain_startup_error(cause.take().expect("owned session cause"))
}

pub(super) fn session_arguments(
    arguments: std::fmt::Arguments<'_>,
    context: SpeculativeExecutionStreams<'_>,
) -> Error {
    let Some((sources, _)) = context.original_numerical() else {
        return Error::Exception(Exception::custom(arguments.to_string()));
    };
    if let Err(refusal) =
        transport_controls::<(std::fmt::Arguments<'_>, Diagnostic)>(sources.metadata_funding())
    {
        return Error::WorkspacePlanning(refusal);
    }
    // Only architecture-owned deterministic argument lists enter this hook.
    // The shared count/write worker reserves the actual UTF-8 destination.
    match sources.metadata_funding().metadata_string(arguments) {
        Ok(message) => sources.retain_startup_error(Diagnostic(message)),
        Err(cause) => sources.retain_startup_error(cause),
    }
}

fn neural_cause_funded<E: std::error::Error + Send + Sync + 'static>(
    cause: E,
    funding: &HostMetadataFunding,
) -> eredu_nn::Error {
    if let Err(refusal) = transport_controls::<NeuralCause<E>>(funding) {
        drop(cause);
        return WorkspaceMetadataError::from(refusal).into();
    }
    funding.metadata_source(NeuralCause {
        cause,
        _funding: funding.clone(),
    })
}
pub(super) fn neural_cause<E: std::error::Error + Send + Sync + 'static>(
    cause: E,
    context: SpeculativeExecutionStreams<'_>,
) -> eredu_nn::Error {
    match context.original_numerical() {
        Some((sources, _)) => neural_cause_funded(cause, sources.metadata_funding()),
        None => eredu_nn::Error::backend_retained_source(cause),
    }
}
pub(super) fn neural_observer(
    cause: &Error,
    context: SpeculativeExecutionStreams<'_>,
) -> eredu_nn::Error {
    match context.original_numerical() {
        // ObserverErrorBridge owns the actual cause until resolve. This marker
        // has its own paid transport; it does not format or duplicate that cause.
        Some((sources, _)) => neural_cause_funded(ObservationMarker, sources.metadata_funding()),
        None => eredu_nn::Error::backend(cause.to_string()),
    }
}

#[cfg(test)]
mod tests;

use eredu_nn::workspace::WorkspaceMetadataAllocation;
