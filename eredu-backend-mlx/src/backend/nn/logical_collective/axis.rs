//! One axis permutation plan for ordinary and source-funded variable transport.
use super::*;
use super::packed::PackedOperations;
#[derive(Clone, Copy)]
pub(crate) struct AxisPlan { rank: usize, axis: usize }
impl AxisPlan {
    pub(crate) fn new(rank: usize, axis: usize) -> Option<Self> {
        (rank > 0 && axis < rank && i32::try_from(rank).is_ok()).then_some(Self { rank, axis })
    }
    pub(crate) fn axes(&self, inverse: bool) -> impl ExactSizeIterator<Item = i32> + '_ {
        (0..self.rank).map(move |position| {
            (if inverse {
                if position == self.axis { 0 } else if position < self.axis { position + 1 } else { position }
            } else if position == 0 { self.axis }
            else if position <= self.axis { position - 1 } else { position }) as i32
        })
    }
}
pub(crate) trait AxisOperations: PackedOperations {
    fn transpose(&self, value: &Self::Value, axes: &[i32]) -> Result<Self::Value, Self::Error>;
}
pub(crate) fn controls<O: AxisOperations>(rank: usize) -> Option<usize> {
    let frames = [size_of::<AxisPlan>(), size_of::<bool>(), size_of::<(&O, &O::Value, AxisPlan)>(),
        size_of::<Vec<i32>>(), size_of::<Result<O::Value, O::Error>>(),
        std::alloc::Layout::array::<i32>(rank).ok()?.size()];
    frames.into_iter().try_fold(size_of_val(&frames), usize::checked_add)
}
pub(crate) fn transpose<O: AxisOperations>(ops: &O, value: &O::Value, plan: AxisPlan, inverse: bool)
    -> Result<O::Value, O::Error> {
    ops.charge(controls::<O>(plan.rank).ok_or_else(|| ops.invalid())?)?;
    if ops.shape(value).len() != plan.rank { return Err(ops.invalid()); }
    let mut axes = ops.shape_buffer(plan.rank)?; axes.extend(plan.axes(inverse));
    ops.transpose(value, &axes)
}
impl AxisOperations for Native<'_> {
    fn transpose(&self, value: &Array, axes: &[i32]) -> Result<Array, Self::Error> {
        value.transpose_axes(axes, self.0)
    }
}
impl AxisOperations for Workspace<'_> {
    fn transpose(&self, value: &WorkspaceTensor, axes: &[i32]) -> Result<WorkspaceTensor, Self::Error> {
        value.transpose_axes(axes, self.0)
    }
}
