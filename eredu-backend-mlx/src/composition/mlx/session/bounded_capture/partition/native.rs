//! The same receiver readiness worker under an actual retained native owner.
use crate::backend::array_copy::{
    CaptureCompletion, CaptureTensorNativeError, PreparedCaptureTensor,
};
use eredu_core::checkpoint::TensorDtype;
use safemlx::{Array, Stream};
use std::{
    cell::RefCell,
    mem::{size_of, size_of_val},
};

/// An adapter to the enclosing operation's existing recovery collector.
/// It cannot select another stream, completion scope, or allocation account.
pub(in crate::composition::mlx) trait SourceRetainer {
    type Error;
    fn retain(&self, source: &Array) -> Result<(), Self::Error>;
    fn native_error(cause: CaptureTensorNativeError) -> Self::Error;
}
pub(in crate::composition::mlx) fn validate_source(
    source: &Array,
    shape: &[usize],
) -> Result<TensorDtype, CaptureTensorNativeError> {
    PreparedCaptureTensor::validate_borrowed_shape(source, shape).map(|dtype| match dtype {
        safemlx::Dtype::Float32 => TensorDtype::F32,
        safemlx::Dtype::Float16 => TensorDtype::F16,
        safemlx::Dtype::Bfloat16 => TensorDtype::Bf16,
        _ => unreachable!("validated floating source"),
    })
}
pub(in crate::composition::mlx) fn complete_source<R: SourceRetainer>(
    retainer: &R,
    source: &Array,
    stream: &Stream,
    roots: &RefCell<Vec<Array>>,
    completion: CaptureCompletion<'_>,
) -> Result<(), R::Error> {
    PreparedCaptureTensor::validate_stream(stream).map_err(R::native_error)?;
    completion
        .reserve_roots(roots, 1)
        .map_err(R::native_error)?;
    retainer.retain(source)?;
    drop(completion.settle(source, stream).map_err(R::native_error)?);
    Ok(())
}
pub(in crate::composition::mlx) fn control_bytes<R: SourceRetainer>() -> Option<usize> {
    let fields = [
        size_of::<R>(),
        size_of::<(
            &R,
            &Array,
            &Stream,
            &RefCell<Vec<Array>>,
            CaptureCompletion<'_>,
        )>(),
        size_of::<(&Array, &[usize])>(),
        size_of::<Result<TensorDtype, CaptureTensorNativeError>>(),
        size_of::<Result<(), R::Error>>(),
        size_of::<Result<(), CaptureTensorNativeError>>(),
        size_of::<Result<safemlx::EvaluatedArray<'_>, CaptureTensorNativeError>>(),
        safemlx::OperationEvent::nested_completion_control_bytes::<1>()?,
        safemlx::OriginalScopeObserver::control_bytes()?,
    ];
    fields
        .into_iter()
        .try_fold(size_of_val(&fields), usize::checked_add)
}

pub(in crate::composition::mlx) struct ModelSourceRetainer<'a> {
    pub roots: &'a RefCell<Vec<Array>>,
    pub completion: CaptureCompletion<'a>,
}
impl SourceRetainer for ModelSourceRetainer<'_> {
    type Error = CaptureTensorNativeError;
    fn retain(&self, source: &Array) -> Result<(), Self::Error> {
        let retained = self.completion.clone_array(source)?;
        let mut roots = self
            .roots
            .try_borrow_mut()
            .map_err(|_| CaptureTensorNativeError::CollectorBusy)?;
        if roots.len() == roots.capacity() {
            return Err(eredu_runtime::working_memory::WorkingMemoryError::UnknownBound.into());
        }
        roots.push(retained);
        Ok(())
    }
    fn native_error(cause: CaptureTensorNativeError) -> Self::Error {
        cause
    }
}
/// The actual model adapter's caller, retained clone and error transports.
pub(in crate::composition::mlx) fn model_control_bytes() -> Option<usize> {
    let fields = [
        control_bytes::<ModelSourceRetainer<'_>>()?,
        size_of::<(&ModelSourceRetainer<'_>, &Array)>(),
        size_of::<Array>(),
        size_of::<Result<Array, CaptureTensorNativeError>>(),
        size_of::<std::cell::RefMut<'_, Vec<Array>>>(),
        size_of::<std::cell::BorrowMutError>(),
        safemlx::PreparedArrayClone::control_bytes()?,
        Array::inspection_clone_handle_bytes(),
    ];
    fields
        .into_iter()
        .try_fold(size_of_val(&fields), usize::checked_add)
}
