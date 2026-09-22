//! Native joint selection with shared coefficients for selected and always-on groups.
use crate::backend::nn::layers as nn;
use crate::MlxTensor;
use eredu_nn::{Error as ComputeError, JointGroupSelection, JointGroupSelectionInput};
use safemlx::{
    error::Exception,
    ops::{
        argpartition_axis, concatenate_axis,
        indexing::{take_along_axis, TryIndexOp},
        matmul, sigmoid, softmax_axis,
    },
    Array, Stream,
};
fn compute<T>(value: Result<T, Exception>) -> Result<T, ComputeError> {
    value.map_err(ComputeError::backend_retained_source)
}
pub(crate) fn joint_selection(
    input: JointGroupSelectionInput<'_, MlxTensor>,
    context: &Stream,
) -> Result<JointGroupSelection<MlxTensor>, ComputeError> {
    input.validate()?;
    let hidden_width = input.hidden().as_array().dim(-1);
    let flat = compute(
        input
            .hidden()
            .as_array()
            .reshape(&[-1, hidden_width], context),
    )?;
    let logits = compute(matmul(
        &flat,
        &compute(input.weight().as_array().transpose(context))?,
        context,
    ))?;
    let primary = compute(logits.try_index_device((.., ..input.selectable_groups()), context))?;
    let always_on = compute(logits.try_index_device((.., input.selectable_groups()..), context))?;
    let choice = compute(sigmoid(&primary, context))?;
    let choice = compute(choice.add(input.correction_bias().as_array(), context))?;
    let primary_indices = compute(argpartition_axis(choice, -input.top_k(), -1, context))?;
    let primary_indices =
        compute(primary_indices.try_index_device((.., -input.top_k()..), context))?;
    let selected_logits = compute(take_along_axis(&primary, &primary_indices, -1, context))?;
    let all_logits = compute(concatenate_axis(&[selected_logits, always_on], -1, context))?;
    let coefficients = compute(nn::log_sigmoid(all_logits, context))?;
    let coefficients = compute(softmax_axis(coefficients, -1, true, context))?;
    let coefficients = compute(
        coefficients.multiply(
            Array::try_from_f32(input.coefficient_scale())
                .map_err(ComputeError::backend_retained_source)?,
            context,
        ),
    )?;
    let coefficients = compute(coefficients.multiply(input.global_scale().as_array(), context))?;
    let primary_coefficients =
        compute(coefficients.try_index_device((.., ..input.top_k()), context))?;
    let always_on_coefficients =
        compute(coefficients.try_index_device((.., input.top_k()..), context))?;
    Ok(JointGroupSelection::new(
        MlxTensor::from_array(primary_indices),
        MlxTensor::from_array(primary_coefficients),
        MlxTensor::from_array(always_on_coefficients),
    ))
}

/// Every returned native wrapper in the selected equation, including both
/// log-sigmoid negatives and its integer zero, is independently retained.
pub(crate) fn joint_selection_control_bytes() -> Option<usize> {
    joint_selection_ordinary_frame_bytes()?
        .checked_add(std::mem::size_of::<[usize; 2]>())?
        .checked_add(safemlx::ops::concatenate_axis_control_bytes()?)?
        .checked_add(safemlx::OriginalScopeObserver::control_bytes()?)
}

/// Fixed Rust frames and inline arguments of the shared joint equation. Safe
/// operator wrappers, native graph controls and completion remain separate.
pub(crate) fn joint_selection_ordinary_frame_bytes() -> Option<usize> {
    use std::mem::{size_of, size_of_val};
    let frames = [
        size_of::<JointGroupSelectionInput<'_, MlxTensor>>() * 2,
        size_of::<JointGroupSelection<MlxTensor>>(),
        size_of::<Result<JointGroupSelection<MlxTensor>, ComputeError>>(),
        size_of::<(&JointGroupSelectionInput<'_, MlxTensor>, &Stream)>(),
        size_of::<[Array; 2]>(),
        size_of::<[i32; 2]>(),
        size_of::<[&Array; 4]>(),
        size_of::<[&Stream; 4]>(),
        size_of::<Result<(), ComputeError>>(),
        size_of::<Array>() * 21,
        size_of::<Result<Array, Exception>>() * 21,
        size_of::<Result<Array, ComputeError>>() * 17,
    ];
    frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)
}
