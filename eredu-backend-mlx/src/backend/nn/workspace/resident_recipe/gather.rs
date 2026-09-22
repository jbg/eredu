//! The shared checked take-axis worker and its native strided Gather.
use super::*;

pub(super) fn lowering(operation: WorkspaceOperationView<'_>) -> Option<Lowering> {
    let WorkspaceOperationKindView::Gather { axis } = operation.kind else {
        return None;
    };
    if operation.inputs.len() != 2 || operation.outputs.len() != 1 {
        return None;
    }
    let source = operation.inputs.get(0)?;
    let indices = operation.inputs.get(1)?;
    let output = operation.outputs.get(0)?;
    let extent = *source.shape().get(axis)?;
    let count = indices.elements().ok()?;
    if !matches!(
        indices.dtype(),
        WorkspaceDtype::Int32 | WorkspaceDtype::Uint32 | WorkspaceDtype::Uint8
    ) || (extent <= 0 && count != 0)
        || source.dtype() != output.dtype()
        || !output.shape().iter().eq(source.shape()[..axis]
            .iter()
            .chain(indices.shape())
            .chain(&source.shape()[axis + 1..]))
    {
        return None;
    }
    // ops.cpp::take: the single-index gather's astype candidate and Gather,
    // optional axis permutation, then Squeeze. One index needs no broadcast.
    let transpose = usize::from(axis != 0);
    let mut value = Lowering::plain(3 + transpose, 4 + transpose, 0);
    if count != 0 {
        // validate_take_indices: ge/lt/and (3 * 5 nodes / 6 edges),
        // not (cast + unary), any (Reduce + Squeeze), zeros_like (two
        // Broadcasts + cast + Full), where (three casts/broadcasts + Select).
        // The actual eager seeds are the two comparison endpoints and zero.
        value.primitives += 3 * 5 + 2 + 2 + 4 + 7;
        value.edges += 3 * 6 + 2 + 2 + 4 + 9;
        value.seeds = 3;
        value.validations = 1;
        // Reduce can compact its input and allocate a second-pass accumulator.
        value.maximum_births = value.primitives + value.seeds + 2;
    }
    // Gather first concatenates index shape and source slice shape; the
    // indexed singleton is only removed afterward. This is also the worker
    // rank used to derive graph and native dispatch storage.
    value.intermediate_rank = source.shape().len().checked_add(indices.shape().len())?;
    Some(value)
}
