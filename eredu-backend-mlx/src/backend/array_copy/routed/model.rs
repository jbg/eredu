//! The existing scalar source decoder under the original model occurrence.
use super::*;
use crate::backend::array_copy::{CaptureCompletion, PreparedCaptureTensor};
use eredu_runtime::working_memory::{CaptureRoutedBatchWriter, OriginalSpeculativeBudgetCustody};
use std::{
    cell::RefCell,
    mem::{size_of, size_of_val},
};

pub(crate) fn execute(
    source: &RoutedUnitCaptureSource<'_, Array>,
    writer: CaptureRoutedBatchWriter<'_, '_>,
    stream: &safemlx::Stream,
    roots: &RefCell<Vec<Array>>,
    custody: &OriginalSpeculativeBudgetCustody,
    observer: &safemlx::OriginalScopeObserver,
) -> Result<(), Error> {
    writer.validate_model_custody(custody)?;
    let bank = writer.geometry().bank();
    let tokens = writer.geometry().source_shape()[0] as u64;
    CompletedRoutedCaptureSource::validate_borrowed(source, bank, tokens)?;
    PreparedCaptureTensor::validate_stream(stream)?;
    let completion = CaptureCompletion::Original(observer);
    let inputs = [
        source.values,
        source.token_indices,
        source.selection_indices,
        source.coefficients,
        source.source_groups,
    ];
    // The same model Q retains every source graph before the first completion.
    // No separate scheduled reservation or source registration is manufactured.
    let completed = completion.settle_retained(inputs, stream, roots)?;
    let source = CompletedRoutedCaptureSource::new(source, completed, bank, tokens)?;
    source.copy_routed(RoutedCaptureTransfer::Model(writer.prepare_model(custody)?))
}

pub(crate) fn control_bytes() -> Option<usize> {
    let frames = [
        CompletedRoutedCaptureSource::control_bytes()?,
        CaptureCompletion::retained_settlement_control_bytes::<5>()?,
        safemlx::Stream::device_type_control_bytes()?,
        size_of::<(
            &RoutedUnitCaptureSource<'_, Array>,
            CaptureRoutedBatchWriter<'_, '_>,
            &safemlx::Stream,
            &RefCell<Vec<Array>>,
            &OriginalSpeculativeBudgetCustody,
            &safemlx::OriginalScopeObserver,
        )>(),
        size_of::<CaptureRoutedModelTransfer<'_, '_, '_>>(),
        size_of::<
            Result<
                CaptureRoutedModelTransfer<'_, '_, '_>,
                eredu_runtime::working_memory::WorkingMemoryError,
            >,
        >(),
        size_of::<Result<(), Error>>(),
        size_of::<(RoutedUnitGeometry, u64, CaptureCompletion<'_>)>(),
    ];
    frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)
}
