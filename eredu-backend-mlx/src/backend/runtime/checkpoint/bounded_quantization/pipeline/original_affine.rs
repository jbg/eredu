//! CPU tile construction using the selected original owner's admitted storage.
use super::*;
use crate::backend::runtime::checkpoint::store::CheckpointMaterializationError;
use eredu_runtime::working_memory::WorkingMemoryError;
use safemlx::{CpuAffineQuantizeSubmissionLayout, DeviceType, OperationEvent};

/// The caller supplies a completed source and funds the quoted physical,
/// Graph and Record capacities, source custody, runtime, streams and owner.
/// This entry authenticates that owner and checks the quote against the actual
/// source before consuming constructor storage. It creates no new admission.
pub(crate) fn submit_original_affine_tile(
    mut prepared: WeightMaterialization,
    quantization: eredu_checkpoint::AffineQuantization,
    target: &BoundedQuantizationTarget,
    stream: &Stream,
    layout: CpuAffineQuantizeSubmissionLayout,
) -> Result<WeightMaterialization, Error> {
    let observer = prepared.original_observer()?.clone();
    if prepared.inputs().len() != 1 || !prepared.outputs().is_empty() {
        return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
    }
    if stream.device_type()? != DeviceType::Cpu {
        return Err(Error::PrefillControl(WorkingMemoryError::UnknownBound));
    }
    let input = &prepared.inputs()[0];
    let companion = match target.affine_companion_dtype {
        RecipeDtype::F16 => Dtype::Float16,
        RecipeDtype::BF16 => Dtype::Bfloat16,
        RecipeDtype::F32 => Dtype::Float32,
        _ => return Err(Error::PrefillControl(WorkingMemoryError::UnknownBound)),
    };
    let (columns, leading) = input
        .shape()
        .split_last()
        .ok_or(Error::PrefillControl(WorkingMemoryError::UnknownBound))?;
    let rows = leading
        .iter()
        .try_fold(1usize, |rows, &dimension| {
            rows.checked_mul(usize::try_from(dimension).ok()?)
        })
        .ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))?;
    let columns = usize::try_from(*columns)
        .map_err(|_| Error::PrefillControl(WorkingMemoryError::UnknownBound))?;
    let actual = OperationEvent::cpu_affine_quantize_submission_layout(
        input.dtype(),
        companion,
        input.ndim(),
        rows,
        columns,
        quantization.group_size,
        quantization.bits,
    )
    .ok_or(Error::PrefillControl(WorkingMemoryError::UnknownBound))?;
    if actual != layout {
        return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
    }
    OperationEvent::validate_traversal_leaf(input, &observer)
        .map_err(CheckpointMaterializationError::OriginalNative)?;
    prepare_quantized_outputs_with(
        &mut prepared,
        quantization.into(),
        target,
        stream,
        Some((layout, &observer)),
        |array, dtype, stream| array.as_dtype(dtype, stream),
    )?;
    Ok(prepared.submit_prepared_outputs_with_traversal(stream, &layout.traversal())?)
}
