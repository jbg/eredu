//! Shared leading-block permutation for ordinary and counted variable exchange.
use super::*;
use super::packed::PackedOperations;
use std::alloc::Layout;

pub(crate) trait BlockOperations: PackedOperations {
    fn interval(&self, input: &Self::Value, start: i32, end: i32) -> Result<Self::Value, Self::Error>;
    fn alias(&self, input: &Self::Value) -> Self::Value;
    fn concatenate(&self, values: &[Self::Value]) -> Result<Self::Value, Self::Error>;
}
pub(crate) fn controls<O: BlockOperations, I>(rank: usize, peers: usize) -> Option<usize> {
    let frames = [size_of::<(&O, &O::Value, &[usize], I)>(), size_of::<Vec<i32>>(),
        size_of::<Vec<O::Value>>(), size_of::<Vec<&O::Value>>(),
        size_of::<[O::Value; 2]>(), size_of::<Result<O::Value, O::Error>>(),
        size_of::<[usize; 4]>(), size_of::<[i32; 3]>(),
        Layout::array::<i32>(peers.checked_add(rank)?).ok()?.size(),
        Layout::array::<O::Value>(peers).ok()?.size(),
        Layout::array::<&O::Value>(peers).ok()?.size(),
        crate::tensor::narrow::control_bytes(rank)?.checked_mul(peers)?,
        safemlx::Array::inspection_clone_handle_bytes(),
    ];
    frames.into_iter().try_fold(size_of_val(&frames), usize::checked_add)
}
pub(crate) fn concatenate<O: BlockOperations, I: Iterator<Item = usize>>(
    ops: &O, input: &O::Value, counts: &[usize], order: I,
) -> Result<O::Value, O::Error> {
    ops.charge(controls::<O, I>(ops.shape(input).len(), counts.len()).ok_or_else(|| ops.invalid())?)?;
    let mut offsets = ops.shape_buffer(counts.len())?;
    let mut offset = 0i32;
    for &count in counts {
        offsets.push(offset);
        offset = offset.checked_add(i32::try_from(count).map_err(|_| ops.invalid())?)
            .ok_or_else(|| ops.invalid())?;
    }
    if ops.shape(input).first().copied() != Some(offset) { return Err(ops.invalid()); }
    let mut blocks = ops.value_buffer(counts.len())?;
    for logical in order {
        let count = *counts.get(logical).ok_or_else(|| ops.invalid())?;
        if count == 0 { continue; }
        // The plan supplies at most one block per member; this prevents an
        // unpriced Vec growth even if an internal caller violates that contract.
        if blocks.len() == counts.len() { return Err(ops.invalid()); }
        let end = offsets[logical].checked_add(i32::try_from(count).map_err(|_| ops.invalid())?)
            .ok_or_else(|| ops.invalid())?;
        blocks.push(ops.interval(input, offsets[logical], end)?);
    }
    match blocks.as_slice() {
        [] => {
            let mut shape = ops.shape_buffer(ops.shape(input).len())?;
            shape.extend_from_slice(ops.shape(input));
            *shape.first_mut().ok_or_else(|| ops.invalid())? = 0;
            ops.zeros(input, &shape)
        }
        [block] => Ok(ops.alias(block)),
        blocks => ops.concatenate(blocks),
    }
}
impl BlockOperations for Native<'_> {
    fn interval(&self, input: &Array, start: i32, end: i32) -> Result<Array, Self::Error> {
        use ref_cast::RefCast;
        crate::MlxTensor::ref_cast(input).narrow_axis(0, start, end, self.0)
            .map(crate::MlxTensor::into_array).map_err(safemlx::error::Exception::from_source)
    }
    fn alias(&self, input: &Array) -> Array { input.clone() }
    fn concatenate(&self, values: &[Array]) -> Result<Array, Self::Error> {
        let mut borrowed = Vec::with_capacity(values.len());
        borrowed.extend(values.iter());
        safemlx::ops::concatenate_axis(&borrowed, 0, self.0)
    }
}
impl BlockOperations for Workspace<'_> {
    fn interval(&self, input: &WorkspaceTensor, start: i32, end: i32) -> Result<WorkspaceTensor, Self::Error> {
        input.narrow_axis(0, start, end, self.0)
    }
    fn alias(&self, input: &WorkspaceTensor) -> WorkspaceTensor { input.clone() }
    fn concatenate(&self, values: &[WorkspaceTensor]) -> Result<WorkspaceTensor, Self::Error> {
        WorkspaceTensor::concatenate(values, 0, self.0)
    }
}


/// Same final leading-axis concatenation for ordinary local routes and their
/// paid native/cold transform role. No input alias is cloned by this adapter.
pub(crate) fn join_controls<O: BlockOperations>(inputs: usize) -> Option<usize> {
    let frames = [size_of::<(&O, &[O::Value])>(), size_of::<Vec<&O::Value>>(),
        size_of::<Result<O::Value, O::Error>>(),
        Layout::array::<&O::Value>(inputs).ok()?.size()];
    frames.into_iter().try_fold(size_of_val(&frames), usize::checked_add)
}
pub(crate) fn join<O: BlockOperations>(ops: &O, values: &[O::Value]) -> Result<O::Value, O::Error> {
    ops.charge(join_controls::<O>(values.len()).ok_or_else(|| ops.invalid())?)?;
    if values.is_empty() { return Err(ops.invalid()); }
    ops.concatenate(values)
}
