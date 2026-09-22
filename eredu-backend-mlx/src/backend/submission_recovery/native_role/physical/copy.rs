//! A completed independent parameter root using the shared numerical worker.
use super::*;
use crate::backend::{
    OriginalCopyEnvironment,
    array_copy::IsolatedArrayCopy,
    nn::workspace::{ExistingArrayProjection, ResidentExecutionMechanisms},
};
use eredu_nn::{Tensor, workspace::WorkspaceContext};
use std::mem::{size_of, size_of_val};

pub(crate) fn copy_array(
    source: &safemlx::Array,
    mechanism: ResidentExecutionMechanisms,
    context: &WorkspaceContext,
    environment: &OriginalCopyEnvironment<'_>,
    execution: &eredu_runtime::working_memory::InferenceExecutionIdentity,
    stream: &safemlx::Stream,
) -> Result<CompletedNumerical<safemlx::Array>, Error> {
    let mut projection = ExistingArrayProjection::with_source_count(context, 1)
        .map_err(|cause| Error::Neural(context.metadata_source(cause)))?;
    let source_value = projection.project(source).map_err(Error::Neural)?;
    if !projection.is_complete() {
        return Err(Error::OriginalSourceContract {
            stage: "independent parameter copy completed backing",
            cause: WorkingMemoryError::UnknownBound,
        });
    }
    context.begin_span();
    // IsolatedArrayCopy completes this exact frontier before its eager copy.
    let contiguous = source_value.contiguous(context).map_err(Error::Neural)?;
    context
        .complete_values(&[&contiguous])
        .map_err(Error::Neural)?;
    let output = contiguous.deep_copy(context).map_err(Error::Neural)?;
    context.complete_values(&[&output]).map_err(Error::Neural)?;
    let report = context.finish_report(&[output]).map_err(Error::Neural)?;
    let overflow = || Error::PrefillControl(WorkingMemoryError::Overflow);
    let frames = [
        size_of::<(
            &safemlx::Array,
            ResidentExecutionMechanisms,
            &WorkspaceContext,
            &OriginalCopyEnvironment<'_>,
            &eredu_runtime::working_memory::InferenceExecutionIdentity,
            &safemlx::Stream,
        )>(),
        size_of::<IsolatedArrayCopy<'_>>(),
        size_of::<safemlx::Array>(),
        size_of::<Result<safemlx::Array, Error>>(),
        crate::backend::runtime::cache::completed_borrow_control_bytes().ok_or_else(overflow)?,
    ];
    let controls = frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)
        .ok_or_else(overflow)?;
    execute_numerical(
        &report,
        &[source],
        1,
        mechanism,
        context,
        environment,
        execution,
        controls,
        || {
            let value = IsolatedArrayCopy::new(source).copy(stream)?;
            crate::backend::runtime::cache::complete_and_borrow(&value, stream)?;
            Ok(value)
        },
    )
}
