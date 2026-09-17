//! Final-axis Metal softmax and the general-axis native equation. Precise
//! final-axis accumulation is in registers; it does not widen a whole tensor.

use super::facts::{self, add, mul, Aliases, Emitter, FactResult, Output};
use super::{
    reduction::{capacity_fixed as capacity, sum_cost_fixed as sum_cost},
    *,
};

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
    let WorkspaceOperationKindView::Reduction("softmax", axis, _) = operation.kind else {
        return Ok(None);
    };
    if operation.inputs.len() != 1 || operation.outputs.len() != 1 {
        return Err(MlxWorkspaceFactError::descriptor(
            "invalid Metal softmax descriptor",
        ));
    }
    let input = operation.inputs.get(0).expect("checked one input");
    let output = operation.outputs.get(0).expect("checked one output");
    // Integer softmax has axis-dependent conversion behavior in MLX. It needs
    // a separate declared output contract before it can authorize a full trace.
    if input.dtype() != WorkspaceDtype::Float32 {
        return Ok(None);
    }
    let axis = usize::try_from(axis)
        .map_err(|_| MlxWorkspaceFactError::descriptor("invalid Metal softmax axis"))?;
    let extent = input
        .shape()
        .get(axis)
        .copied()
        .ok_or_else(|| MlxWorkspaceFactError::descriptor("invalid Metal softmax axis"))?
        as u64;
    let count = input.elements()?;
    let last = axis + 1 == input.shape().len();
    if output.shape() != input.shape() || output.dtype() != WorkspaceDtype::Float32 {
        return Err(MlxWorkspaceFactError::descriptor(
            "Metal softmax result geometry or dtype differs",
        ));
    }
    if count == 0 {
        sink.output(Output::AliasInput(0))?;
        return sink
            .finish(
                0,
                format_args!(
                    "vendored MLX empty softmax returns its input without a new operation"
                ),
            )
            .map(Some);
    }
    let full = capacity(allocation, count)?;
    let total = if last {
        // Contiguous input gets a result or donation. Strided input gets one
        // contiguous copy, reused as the result. Precise accumulation does not
        // change the floating input/output dtype or allocate an F32 cast.
        full
    } else {
        let rows = count / extent;
        // Precise input cast; max; subtraction; exponential; sum; division;
        // restoration of the input dtype. Both reductions use the same native
        // partial-array geometry; price them independently through completion.
        add(
            add(mul(8, full)?, mul(2, capacity(allocation, rows)?)?)?,
            mul(2, sum_cost(allocation, count, rows, extent)?)?,
        )?
    };
    sink.output(Output::AllocateOrAliasInputs {
        bytes: full,
        inputs: Aliases::Slice(&[0]),
    })?;
    sink.finish(total - full, format_args!("vendored MLX Metal floating softmax: one final-axis result/contiguous buffer, or complete general-axis max/subtract/exp/sum/divide/cast equation and both reduction workspaces; page={} with bounded oversized reuse; active tensor buffers only", allocation.page_size())).map(Some)
}

#[cfg(test)]
mod tests;
