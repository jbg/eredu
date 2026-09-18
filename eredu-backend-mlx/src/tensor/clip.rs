//! The tensor-bound branch of native clip: Maximum followed by Minimum.
use super::{Error, MlxTensor};
use safemlx::{Array, OriginalScopeObserver, Stream, error::Exception};
use std::mem::{size_of, size_of_val};

fn error(cause: Exception) -> Error {
    match OriginalScopeObserver::try_current() {
        Ok(None) => Error::backend_retained_source(cause),
        _ => Error::backend_retained_source(cause),
    }
}
pub(super) fn run(
    input: &MlxTensor,
    minimum: &MlxTensor,
    maximum: &MlxTensor,
    stream: &Stream,
) -> Result<MlxTensor, Error> {
    // mlx::core::clip executes these exact two workers in this order. Both
    // bounds are already tensors, so no optional/scalar bound wrapper is needed.
    // In particular, reversed bounds preserve Maximum-then-Minimum semantics.
    let lower =
        safemlx::ops::maximum(input.as_array(), minimum.as_array(), stream).map_err(error)?;
    safemlx::ops::minimum(&lower, maximum.as_array(), stream)
        .map(MlxTensor::from_array)
        .map_err(error)
}
pub(crate) fn control_bytes() -> Option<usize> {
    let sizes = [
        size_of::<(&MlxTensor, &MlxTensor, &MlxTensor, &Stream)>(), // trait adapter
        size_of::<(&MlxTensor, &MlxTensor, &MlxTensor, &Stream)>(), // shared worker
        size_of::<[Array; 2]>(),
        size_of::<[Result<Array, Exception>; 2]>(),
        size_of::<Result<Array, Error>>(),
        size_of::<Result<MlxTensor, Exception>>(),
        size_of::<[Result<MlxTensor, Error>; 2]>(),
        size_of::<[(&Array, &Array, &Stream); 2]>(), // borrowed binary wrappers
        size_of::<Result<Option<OriginalScopeObserver>, Exception>>(),
        OriginalScopeObserver::control_bytes()?,
        Error::retained_source_construction_bytes::<Exception>()?,
    ];
    sizes
        .into_iter()
        .try_fold(size_of_val(&sizes), usize::checked_add)
}
