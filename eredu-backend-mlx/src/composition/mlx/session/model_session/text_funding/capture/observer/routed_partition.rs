//! Read-only original invocation and five-source validation before native copying.
use super::*;
use crate::backend::array_copy::{CaptureTensorNativeError, CompletedPartitionRoutedCaptureSource};
use eredu_core::capture::{
    PartitionRoutedUnitCaptureLayout, PartitionRoutedUnitCaptureRequest,
    PartitionRoutedUnitCaptureSource,
};

fn failure(
    backend: &NativeScheduledCapture<'_>,
    cause: CaptureTensorNativeError,
) -> FundedCaptureError<Error> {
    FundedCaptureError::Backend(backend.work.capture_error(Error::Other(Box::new(cause))))
}
fn dtype(value: &Array) -> Result<TensorDtype, CaptureTensorNativeError> {
    match value.dtype() {
        safemlx::Dtype::Float32 => Ok(TensorDtype::F32),
        safemlx::Dtype::Float16 => Ok(TensorDtype::F16),
        safemlx::Dtype::Bfloat16 => Ok(TensorDtype::Bf16),
        other => Err(CaptureTensorNativeError::UnsupportedDtype(other)),
    }
}
pub(super) fn validate_invocation(
    backend: &NativeScheduledCapture<'_>,
    input: &eredu_runtime::RoutedUnitInvocation<'_, Array>,
    layout: &PartitionRoutedUnitCaptureLayout<'_>,
) -> Result<(u64, TensorDtype), FundedCaptureError<Error>> {
    let shape = input.input.shape();
    let rows = PartitionRoutedUnitCaptureLayout::input_rows(shape)
        .map_err(|_| failure(backend, CaptureTensorNativeError::ShapeMismatch))?;
    layout
        .validate_input_invocation(
            rows,
            input.unit_coordinates,
            input.origins.map(|origins| origins.capture_coordinates()),
        )
        .map_err(FundedCaptureError::Admission)?;
    Ok((
        rows,
        dtype(input.input).map_err(|cause| failure(backend, cause))?,
    ))
}
pub(super) fn validate_source(
    backend: &NativeScheduledCapture<'_>,
    source: &PartitionRoutedUnitCaptureSource<'_, Array>,
    request: &PartitionRoutedUnitCaptureRequest<'_>,
    invocation_source_tokens: u64,
    actual_native_rows: u64,
) -> Result<TensorDtype, FundedCaptureError<Error>> {
    validate_batch(
        backend,
        source,
        &PartitionRoutedUnitCaptureLayout {
            geometry: request.geometry,
            source_tokens: invocation_source_tokens,
            ownership: request.ownership,
        },
        actual_native_rows,
    )
    .map(|(dtype, _)| dtype)
}

pub(super) fn validate_batch(
    backend: &NativeScheduledCapture<'_>,
    source: &PartitionRoutedUnitCaptureSource<'_, Array>,
    layout: &PartitionRoutedUnitCaptureLayout<'_>,
    actual_native_rows: u64,
) -> Result<(TensorDtype, u64), FundedCaptureError<Error>> {
    let value = &source.source;
    let shapes = [
        value.values.shape(),
        value.token_indices.shape(),
        value.selection_indices.shape(),
        value.coefficients.shape(),
        value.source_groups.shape(),
    ];
    let dtypes = [
        value.values.dtype(),
        value.token_indices.dtype(),
        value.selection_indices.dtype(),
        value.coefficients.dtype(),
        value.source_groups.dtype(),
    ];
    let (_, tokens) = CompletedPartitionRoutedCaptureSource::validate_source_layouts(
        shapes,
        Some(dtypes),
        value.token_offset,
        crate::backend::array_copy::PartitionRoutedCaptureLayout {
            geometry: layout.geometry,
            source_tokens: layout.source_tokens,
            ownership: layout.ownership,
            origins: source.origins,
            units: source.unit_coordinates,
        },
    )
    .map_err(|cause| failure(backend, cause))?;
    if u64::try_from(shapes[4][0]).ok() != Some(actual_native_rows) {
        return Err(failure(backend, CaptureTensorNativeError::SourceChanged));
    }
    Ok((
        dtype(value.values).map_err(|cause| failure(backend, cause))?,
        tokens,
    ))
}

pub(super) fn control_bytes() -> Option<usize> {
    use std::mem::{size_of, size_of_val};
    let parts = [
        size_of::<(
            &NativeScheduledCapture<'_>,
            &eredu_runtime::RoutedUnitInvocation<'_, Array>,
            &PartitionRoutedUnitCaptureLayout<'_>,
        )>(),
        size_of::<(
            &NativeScheduledCapture<'_>,
            &PartitionRoutedUnitCaptureSource<'_, Array>,
            &PartitionRoutedUnitCaptureRequest<'_>,
            u64,
            u64,
        )>(),
        size_of::<[&[i32]; 5]>(),
        size_of::<&[i32]>(),
        size_of::<std::slice::Iter<'_, i32>>(),
        size_of::<Result<u64, eredu_core::capture::RoutedUnitValidationError>>(),
        size_of::<u64>(),
        size_of::<[safemlx::Dtype; 5]>(),
        size_of::<TensorDtype>() * 2,
        size_of::<u64>() * 2,
        size_of::<Result<(u64, TensorDtype), FundedCaptureError<Error>>>(),
        size_of::<Result<TensorDtype, FundedCaptureError<Error>>>(),
        size_of::<Result<TensorDtype, CaptureTensorNativeError>>(),
        size_of::<CaptureTensorNativeError>(),
        size_of::<Box<CaptureTensorNativeError>>(),
        size_of::<Result<(), CaptureError>>(),
        size_of::<Result<[u64; 3], CaptureError>>(),
        size_of::<PartitionRoutedUnitCaptureLayout<'_>>() * 2,
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}
