//! The selected route-order reduction and rank-order gather share one worker.
use super::*;
use super::packed::PackedOperations;
use std::alloc::Layout;

pub(crate) fn result_controls<O: PackedOperations>(count: usize) -> Option<usize> {
    let frames = [size_of::<(&O, Vec<(usize, O::Value)>)>(),
        size_of::<std::vec::IntoIter<(usize, O::Value)>>(), size_of::<Vec<O::Value>>(),
        size_of::<Result<O::Value, O::Error>>(), size_of::<Option<(usize, O::Value)>>(),
        size_of::<[O::Value; 2]>(), size_of::<[usize; 4]>(),
        size_of::<std::ops::Range<usize>>(), Layout::array::<O::Value>(count).ok()?.size()];
    frames.into_iter().try_fold(size_of_val(&frames), usize::checked_add)
}

/// Keep the ordinary left fold in selected route order, including floating
/// rounding and NaN behavior. Input handles move through the fold.
pub(crate) fn sum<O: PackedOperations>(ops: &O, values: Vec<(usize, O::Value)>)
    -> Result<O::Value, O::Error> {
    ops.charge(result_controls::<O>(values.len()).ok_or_else(|| ops.invalid())?)?;
    let mut values = values.into_iter();
    let (_, mut reduced) = values.next().ok_or_else(|| ops.invalid())?;
    for (_, value) in values { reduced = ops.add(&reduced, &value)?; }
    Ok(reduced)
}

/// Gather follows the actual local rank order. Validated unique source ranks
/// form a permutation, so cycle placement needs no recursive sort or index bank.
pub(crate) fn stacked<O: PackedOperations>(ops: &O, mut values: Vec<(usize, O::Value)>)
    -> Result<O::Value, O::Error> {
    ops.charge(result_controls::<O>(values.len()).ok_or_else(|| ops.invalid())?)?;
    if values.is_empty() { return Err(ops.invalid()); }
    if values.iter().any(|(source, _)| *source >= values.len()) { return Err(ops.invalid()); }
    for index in 0..values.len() {
        while values[index].0 != index {
            let destination = values[index].0;
            // An already placed destination proves a duplicate source rank.
            if values[destination].0 == destination { return Err(ops.invalid()); }
            values.swap(index, destination);
        }
    }
    let mut ordered = ops.value_buffer(values.len())?;
    for (_, value) in values { ordered.push(value); }
    ops.stack_members(&ordered)
}

pub(crate) fn gather<O: PackedOperations>(ops: &O, values: Vec<(usize, O::Value)>,
    input_shape: &[i32]) -> Result<O::Value, O::Error> {
    ops.charge(size_of::<(&O, Vec<(usize, O::Value)>, &[i32], usize, O::Value,
        Result<O::Value, O::Error>)>())?;
    let count = values.len();
    let result = stacked(ops, values)?;
    // Ordinary all_gather leaves a scalar input's stacked vector unchanged.
    if input_shape.is_empty() { return Ok(result); }
    super::packed::flatten(ops, &result, input_shape, count)
}
