//! Gather and dense embedding storage, including the validation graph retained
//! by the native submission. Metal reads strided source rows directly: neither
//! operation copies or widens the complete source table.

use super::facts::{self, add, buffer_capacity, mul, Emitter, FactResult, Output};
use super::{
    reduction::{capacity_fixed as capacity, sum_cost_fixed as sum_cost},
    *,
};
use eredu_checkpoint::LinearFormat;
use eredu_nn::EmbeddingLookupPolicy;

pub(super) fn operation_bound(
    operation: &WorkspaceOperation,
    allocation: NativeAllocationFacts,
) -> Result<Option<WorkspaceOperationBound>, Error> {
    facts::ordinary(|sink| emit(operation.as_view(), allocation, sink))
}

pub(super) fn emit(
    operation: WorkspaceOperationView<'_>,
    allocation: NativeAllocationFacts,
    sink: &mut Emitter<'_>,
) -> FactResult<Option<WorkspaceOperationFacts>> {
    match &operation.kind {
        WorkspaceOperationKindView::Gather { axis } => {
            gather(operation, *axis, allocation, sink).map(Some)
        }
        WorkspaceOperationKindView::Embedding(format, policy)
            if format.encoding() == LinearFormat::Dense =>
        {
            format.validate_fixed()?;
            policy.validate_fixed()?;
            embedding(operation, *policy, allocation, sink).map(Some)
        }
        _ => Ok(None),
    }
}

fn boolean(allocation: NativeAllocationFacts, elements: u64) -> FactResult<u64> {
    buffer_capacity(allocation, elements.max(1))
}

fn gather(
    operation: WorkspaceOperationView<'_>,
    axis: usize,
    allocation: NativeAllocationFacts,
    sink: &mut Emitter<'_>,
) -> FactResult<WorkspaceOperationFacts> {
    if operation.inputs.len() != 2 || operation.outputs.len() != 1 {
        return Err(MlxWorkspaceFactError::descriptor(
            "invalid Metal gather descriptor",
        ));
    }
    let source = operation.inputs.get(0).expect("checked two inputs");
    let indices = operation.inputs.get(1).expect("checked two inputs");
    let output = operation.outputs.get(0).expect("checked one output");
    let extent = source
        .shape()
        .get(axis)
        .copied()
        .ok_or_else(|| MlxWorkspaceFactError::descriptor("invalid Metal gather axis"))?;
    let count = indices.elements()?;
    if !matches!(
        indices.dtype(),
        WorkspaceDtype::Int32 | WorkspaceDtype::Uint32 | WorkspaceDtype::Uint8
    ) || extent == 0 && count != 0
    {
        return Err(MlxWorkspaceFactError::descriptor(
            "invalid Metal gather index domain",
        ));
    }
    let expected = source.shape()[..axis]
        .iter()
        .chain(indices.shape())
        .chain(&source.shape()[axis + 1..]);
    if !output.shape().iter().eq(expected) || output.dtype() != source.dtype() {
        return Err(MlxWorkspaceFactError::descriptor(
            "Metal gather result geometry differs",
        ));
    }
    let scratch = if count == 0 {
        0
    } else {
        // Comparisons with I32 bounds can widen unsigned indices to I64.
        // Count both comparison casts and their scalar construction/casts,
        // four boolean intermediates, and the retained any() reduction.
        let width = if indices.dtype() == WorkspaceDtype::Uint32 {
            8
        } else {
            4
        };
        let comparison = buffer_capacity(allocation, mul(count, width)?)?;
        let scalar = buffer_capacity(allocation, width)?;
        let validation = add(
            add(mul(2, comparison)?, mul(4, scalar)?)?,
            add(
                mul(4, boolean(allocation, count)?)?,
                sum_cost(allocation, count, 1, count)?,
            )?,
        )?;
        // Safe indices: zeros_like and where. Native source values are never
        // converted; gather's final transpose and squeeze are shared views.
        add(
            validation,
            add(
                mul(2, capacity(allocation, count)?)?,
                mul(2, capacity(allocation, 1)?)?,
            )?,
        )?
    };
    sink.output(Output::Allocate(buffer_capacity(
        allocation,
        output.bytes()?.max(4),
    )?))?;
    sink.finish(scratch, format_args!("vendored MLX Metal gather: direct strided source reads, index validation and safe-index buffers retained through completion, possible I64 comparison promotion for U32 indices, one output allocation and aliasing axis permutation; page={} with bounded oversized reuse; active tensor buffers only", allocation.page_size()))
}

fn embedding(
    operation: WorkspaceOperationView<'_>,
    policy: EmbeddingLookupPolicy,
    allocation: NativeAllocationFacts,
    sink: &mut Emitter<'_>,
) -> FactResult<WorkspaceOperationFacts> {
    if operation.inputs.len() != 2 || operation.outputs.len() != 1 {
        return Err(MlxWorkspaceFactError::descriptor(
            "invalid Metal dense embedding descriptor",
        ));
    }
    let ids = operation.inputs.get(0).expect("checked two inputs");
    let weight = operation.inputs.get(1).expect("checked two inputs");
    let output = operation.outputs.get(0).expect("checked one output");
    if !matches!(ids.dtype(), WorkspaceDtype::Int32 | WorkspaceDtype::Uint32)
        || weight.dtype() != WorkspaceDtype::Float32
        || weight.shape().len() != 2
        || weight.shape().iter().any(|n| *n <= 0)
    {
        return Err(MlxWorkspaceFactError::descriptor(
            "invalid Metal dense embedding inputs",
        ));
    }
    let shape = ids
        .shape()
        .iter()
        .copied()
        .chain(std::iter::once(weight.shape()[1]));
    if !output.shape().iter().copied().eq(shape) || output.dtype() != WorkspaceDtype::Float32 {
        return Err(MlxWorkspaceFactError::descriptor(
            "Metal dense embedding output geometry differs",
        ));
    }
    let result = capacity(allocation, output.elements()?)?;
    let scratch =
        embedding_validation_cost_fixed(ids.elements()?, output.elements()?, policy, allocation)?;
    sink.output(Output::Allocate(result))?;
    sink.finish(scratch, format_args!("MLX dense embedding: I32 normalization, retained token-domain reduction, safe-index masking, direct strided table gather and optional zero-sentinel output; no full-table copy/cast; page={} with bounded oversized reuse; active tensor buffers only", allocation.page_size()))
}

/// Shared token validation and optional sentinel-zeroing cost. Physical row
/// gathering/dequantization is charged by the selected embedding mechanism.
pub(super) fn embedding_validation_cost(
    count: u64,
    output_elements: u64,
    policy: EmbeddingLookupPolicy,
    allocation: NativeAllocationFacts,
) -> Result<u64, Error> {
    embedding_validation_cost_fixed(count, output_elements, policy, allocation)
        .map_err(MlxWorkspaceFactError::ordinary)
}

pub(super) fn embedding_validation_cost_fixed(
    count: u64,
    output_elements: u64,
    policy: EmbeddingLookupPolicy,
    allocation: NativeAllocationFacts,
) -> FactResult<u64> {
    let index = capacity(allocation, count)?;
    let mask = boolean(allocation, count)?;
    let scalar = capacity(allocation, 1)?;
    let result = capacity(allocation, output_elements)?;
    // IDs normalize to I32 before comparisons. Domain validation creates
    // ge/lt/and/not and a retained any() result. Safe lookup repeats ge/lt/and
    // and builds zeros_like/where. Charge all outputs and possible ID casts.
    let mut scratch = add(
        add(mul(3, index)?, mul(7, mask)?)?,
        add(mul(6, scalar)?, sum_cost(allocation, count, 1, count)?)?,
    )?;
    if matches!(policy, EmbeddingLookupPolicy::ZeroSentinel(_)) {
        // Validation eq/or plus the output sentinel mask. The final where
        // includes zero embeddings, its result, and scalar initialization.
        scratch = add(
            scratch,
            add(add(mul(3, mask)?, mul(4, scalar)?)?, mul(2, result)?)?,
        )?;
    }
    Ok(scratch)
}

#[cfg(test)]
mod tests;
