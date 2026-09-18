//! Paid original transports for the existing borrowed observation adapter.
use super::*;
use safemlx::OriginalScopeObserver;
use std::mem::{size_of, size_of_val};

#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
struct OriginalCause<E: std::error::Error + 'static> {
    #[source]
    cause: E,
    // The fixed native carrier retains the accepted host/native domain. Its
    // error construction neither submits work nor marks that domain complete.
    _custody: Exception,
}
fn carrier() -> Option<Exception> {
    match OriginalScopeObserver::try_current() {
        Ok(Some(scope)) => Some(scope.invalid_input_error()),
        Ok(None) => None,
        Err(cause) => Some(cause),
    }
}
pub(super) fn source<E: std::error::Error + Send + Sync + 'static>(cause: E) -> ComputeError {
    match carrier() {
        Some(custody) => ComputeError::backend_retained_source(OriginalCause {
            cause,
            _custody: custody,
        }),
        None => ComputeError::backend_retained_source(cause),
    }
}
pub(super) fn callback(cause: ComputeError) -> ComputeError {
    match carrier() {
        Some(custody) => ComputeError::backend_retained_source(OriginalCause {
            cause,
            _custody: custody,
        }),
        None => cause,
    }
}
pub(super) fn signal(_: &ComputeError) -> Exception {
    carrier().unwrap_or_else(|| Exception::from_source(eredu_nn::GeneratedTensorRetentionSignal))
}
pub(super) fn native(cause: Exception) -> ComputeError {
    source(RetainedInputFailure(cause))
}

pub(super) type Factory<'a> = eredu_nn::MappedGeneratedTensorFactory<
    'a,
    Array,
    MlxTensor,
    Exception,
    ComputeError,
    for<'b> fn(&'b Array) -> &'b MlxTensor,
    fn(Array) -> MlxTensor,
    fn(Exception) -> ComputeError,
    fn(&ComputeError) -> Exception,
>;

pub(super) fn control_bytes() -> Option<usize> {
    let controls = [
        ComputeError::retained_source_construction_bytes::<OriginalCause<ComputeError>>()?,
        ComputeError::retained_source_construction_bytes::<OriginalCause<Exception>>()?,
        ComputeError::retained_source_construction_bytes::<
            OriginalCause<RetainedInputFailure<Exception>>,
        >()?,
        OriginalScopeObserver::control_bytes()?,
        size_of::<NativeInputObserver<'_>>(),
        size_of::<Option<NativeInputObserver<'_>>>(),
        size_of::<Factory<'_>>(),
        size_of::<Option<ComputeError>>(),
        size_of::<Result<(), ComputeError>>(),
        size_of::<Result<(), Exception>>(),
        size_of::<Result<MlxTensor, ComputeError>>(),
        size_of::<Option<Exception>>(),
        size_of::<Exception>(),
        size_of::<&mut dyn FnMut(&Array) -> Result<(), Exception>>(),
        size_of::<&mut dyn FnMut(&MlxTensor) -> Result<(), ComputeError>>(),
        size_of::<&mut dyn FnMut() -> Result<Array, Exception>>(),
        size_of::<&mut dyn FnMut() -> Result<MlxTensor, ComputeError>>(),
    ];
    controls
        .into_iter()
        .try_fold(size_of_val(&controls), usize::checked_add)
}

/// Fixed transports shared by borrowed grouped callbacks. Their shape buffers
/// are separate prepared destinations owned by the selected observer directory.
pub(super) fn grouped_callback_control_bytes() -> Option<usize> {
    let controls = [
        ComputeError::retained_source_construction_bytes::<OriginalCause<ComputeError>>()?,
        ComputeError::retained_source_construction_bytes::<
            OriginalCause<RetainedInputFailure<Exception>>,
        >()?,
        OriginalScopeObserver::control_bytes()?,
        size_of::<Option<Exception>>(),
        size_of::<Exception>(),
        size_of::<Result<Option<OriginalScopeObserver>, Exception>>(),
    ];
    controls
        .into_iter()
        .try_fold(size_of_val(&controls), usize::checked_add)
}
