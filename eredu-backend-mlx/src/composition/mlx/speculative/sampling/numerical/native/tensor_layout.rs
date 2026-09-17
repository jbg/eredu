//! Borrow exact completed floating input layout for composed CPU equations.
use super::*;
use eredu_nn::workspace::{WorkspaceFloatingType, WorkspaceRepresentation};

pub(super) fn completed_floating(
    value: &OriginalNumericalValue,
    context: &WorkspaceContext,
) -> Result<WorkspaceTensor, Error> {
    let parts = [
        Array::descriptor_control_bytes().ok_or_else(model::overflow)?,
        size_of::<(&OriginalNumericalValue, &WorkspaceContext)>(),
        size_of::<WorkspaceRepresentation>(),
        size_of::<WorkspaceFloatingType>(),
        size_of::<safemlx::Dtype>(),
        size_of::<&Array>(),
        size_of::<Option<bool>>(),
        size_of::<bool>() * 2,
        size_of::<Option<&i32>>(),
        size_of::<Option<&i64>>(),
        size_of::<std::slice::Iter<i64>>(),
        size_of::<Result<WorkspaceTensor, Error>>(),
    ];
    context.charge_metadata(
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
            .ok_or_else(model::overflow)?,
    ).map_err(eredu_nn::Error::from)?;
    // The caller authenticated request, stream and closed source before this
    // descriptive loan. Neither this inspection nor a stride fact creates one.
    let array = &value.value().array;
    let descriptor = array.try_descriptor()?;
    let dtype = match descriptor.facts().dtype() {
        safemlx::Dtype::Float32 => WorkspaceFloatingType::Float32,
        safemlx::Dtype::Bfloat16 => WorkspaceFloatingType::Bfloat16,
        _ => {
            return Err(Error::PrefillControl(
                eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
            ))
        }
    };
    if descriptor.facts().allocation().is_none()
        || array.signed_strides().iter().any(|&stride| stride < 0)
    {
        return Err(Error::PrefillControl(
            eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
        ));
    }
    let row = descriptor.row_contiguous().ok_or(Error::PrefillControl(
        eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
    ))?;
    let last = descriptor.shape().last().is_none_or(|&n| n == 1)
        || array.signed_strides().last() == Some(&1);
    WorkspaceTensor::existing(
        context
            .layout(descriptor.shape(), WorkspaceDtype::Float32)?
            .with_representation(Some(
                WorkspaceRepresentation::new(dtype, row).with_last_axis_contiguous(last),
            )),
        context,
    )
    .map_err(Error::from)
}
