//! Native forward-reduction storage. The vendored Metal implementation has
//! one optional contiguous input copy and one optional two-pass accumulator.
//! Threadgroup-local storage and encoded host arguments are not tensor buffers.

use super::facts::{self, Aliases, Emitter, FactResult, Output};
use super::*;

pub(super) fn operation_bound(
    operation: &WorkspaceOperation,
    allocation: MetalAllocationFacts,
) -> Result<Option<WorkspaceOperationBound>, Error> {
    facts::ordinary(|sink| emit(operation.as_view(), allocation, sink))
}

pub(super) fn emit(
    operation: WorkspaceOperationView<'_>,
    allocation: MetalAllocationFacts,
    sink: &mut Emitter<'_>,
) -> FactResult<Option<WorkspaceOperationFacts>> {
    use facts::{add, mul};
    let WorkspaceOperationKindView::Reduction(
        name @ ("sum" | "sum_all" | "mean" | "min" | "max"),
        axis,
        keep,
    ) = operation.kind
    else {
        return Ok(None);
    };
    if operation.inputs.len() != 1 || operation.outputs.len() != 1 {
        return Err(MlxWorkspaceFactError::descriptor(
            "invalid Metal reduction descriptor",
        ));
    }
    let input = operation.inputs.get(0).expect("checked one input");
    let output = operation.outputs.get(0).expect("checked one output");
    if matches!(name, "min" | "max")
        && (input.dtype() != WorkspaceDtype::Float32 || input.elements()? == 0)
    {
        return Err(MlxWorkspaceFactError::descriptor(
            "invalid floating extremum reduction",
        ));
    }
    let all = name == "sum_all";
    if all
        && (axis != 0 || keep || input.elements()? == 0 || input.dtype() != WorkspaceDtype::Float32)
    {
        return Err(MlxWorkspaceFactError::descriptor(
            "invalid full F32 reduction descriptor",
        ));
    }
    let axis = usize::try_from(axis)
        .map_err(|_| MlxWorkspaceFactError::descriptor("invalid Metal reduction axis"))?;
    let extent = if all {
        input.elements()?
    } else {
        input
            .shape()
            .get(axis)
            .copied()
            .ok_or_else(|| MlxWorkspaceFactError::descriptor("invalid Metal reduction axis"))?
            as u64
    };
    let expected = input
        .shape()
        .iter()
        .enumerate()
        .filter_map(|(ordinal, &extent)| {
            if all {
                None
            } else if ordinal == axis {
                if keep { Some(1) } else { None }
            } else {
                Some(extent)
            }
        });
    let dtype = if name == "mean" {
        WorkspaceDtype::Float32
    } else {
        match input.dtype() {
            WorkspaceDtype::Bool => WorkspaceDtype::Int32,
            WorkspaceDtype::Uint8 => WorkspaceDtype::Uint32,
            dtype => dtype,
        }
    };
    if !output.shape().iter().copied().eq(expected) || output.dtype() != dtype {
        return Err(MlxWorkspaceFactError::descriptor(
            "Metal reduction result geometry or dtype differs",
        ));
    }
    let output_bytes = capacity_fixed(allocation, output.elements()?)?;
    let mut total = sum_cost_fixed(allocation, input.elements()?, output.elements()?, extent)?;
    if name == "mean" {
        // MLX mean is sum / number_of_elements. Count the scalar and both
        // possible division input casts plus the division result.
        total = add(
            total,
            add(mul(3, output_bytes)?, capacity_fixed(allocation, 1)?)?,
        )?;
    }
    sink.output(Output::AllocateOrAliasInputs {
        bytes: output_bytes,
        inputs: Aliases::Slice(&[0]),
    })?;
    sink.finish(total - output_bytes, format_args!("vendored MLX Metal {name}: one possible contiguous input copy, at most one partial accumulator (4096 scalars or 128 per result), output and mean division/scalar when applicable; page={} with bounded oversized reuse; active tensor buffers only", allocation.page_size())).map(Some)
}

/// Capacity of an at-most-F32 tensor, including a possible scalar backing for
/// an empty broadcast and Metal reduction's minimum four-byte output buffer.
pub(super) fn capacity(allocation: MetalAllocationFacts, elements: u64) -> Result<u64, Error> {
    capacity_fixed(allocation, elements).map_err(MlxWorkspaceFactError::ordinary)
}

pub(super) fn capacity_fixed(allocation: MetalAllocationFacts, elements: u64) -> FactResult<u64> {
    facts::buffer_capacity(allocation, facts::mul(elements.max(1), 4)?)
}

/// All new buffers for one sum, including its result. This also bounds the
/// same Reduce worker for a floating minimum/maximum, or the accumulation part of a mean or norm. Callers price subsequent arithmetic.
pub(super) fn sum_cost(
    allocation: MetalAllocationFacts,
    input_elements: u64,
    output_elements: u64,
    reduction_extent: u64,
) -> Result<u64, Error> {
    sum_cost_fixed(
        allocation,
        input_elements,
        output_elements,
        reduction_extent,
    )
    .map_err(MlxWorkspaceFactError::ordinary)
}

pub(super) fn sum_cost_fixed(
    allocation: MetalAllocationFacts,
    input_elements: u64,
    output_elements: u64,
    reduction_extent: u64,
) -> FactResult<u64> {
    use facts::{add, mul};
    let output = capacity_fixed(allocation, output_elements)?;
    if input_elements == 0 || reduction_extent <= 1 {
        return Ok(output);
    }
    // Retain the native dispatch thresholds: short reductions never allocate
    // a partial array. The bound considers every stride-dependent branch that
    // the geometry permits, without assuming contiguous input.
    let long_column = if reduction_extent >= 1024 {
        mul(
            if reduction_extent >= 32768 { 128 } else { 32 },
            output_elements,
        )?
    } else {
        0
    };
    let two_pass_column = if reduction_extent > 256 && output_elements / 32 < 1024 {
        mul(32, output_elements)?
    } else {
        0
    };
    // REDUCE_N_READS is four. An all-reduce above 4096 input elements uses
    // 128 partial scalars up to 2^26 input bytes, then 4096. Actual input
    // elements are at most four bytes; smaller dtypes cannot need more rows.
    let all_reduce = if output_elements == 1 && input_elements > 4096 {
        if mul(input_elements, 4)? > (1 << 26) {
            4096
        } else {
            128
        }
    } else {
        0
    };
    let partial_elements = long_column.max(two_pass_column).max(all_reduce);
    // Each branch performs its second pass directly; it does not allocate a
    // second accumulator. GeneralReduce can additionally copy the input once.
    let partial = if partial_elements == 0 {
        0
    } else {
        capacity_fixed(allocation, partial_elements)?
    };
    add(
        output,
        add(capacity_fixed(allocation, input_elements)?, partial)?,
    )
}

#[cfg(test)]
mod tests;
