//! Selected native routing mechanisms, retaining full index backing and shared
//! coefficient allocations across distinct logical output views.

use super::facts::{self, add, mul, Emitter, FactResult, HostEmitter, Output};
use super::{
    matrix::matmul_cost_fixed as matmul_cost, reduction::capacity_fixed as capacity,
    sampling::sort_fixed as sort, *,
};

mod selector;
pub(super) use selector::with_projection as with_selector_projection;

fn invalid() -> MlxWorkspaceFactError {
    MlxWorkspaceFactError::descriptor("invalid Metal routing workspace descriptor")
}
pub(super) fn operation_bound(
    op: &WorkspaceOperation,
    a: MetalAllocationFacts,
) -> Result<Option<WorkspaceOperationBound>, Error> {
    if matches!(op.kind, WorkspaceOperationKind::GroupSelection { .. }) {
        return selector::bounds(op, a).map(|bound| bound.map(|(tensor, _)| tensor));
    }
    facts::ordinary(|sink| emit(op.as_view(), a, sink))
}

#[derive(Clone, Copy)]
pub(super) struct JointGeometry {
    pub rows: u64,
    pub rows_i32: i32,
    pub dimensions: i32,
    pub groups: i32,
    pub selectable: i32,
    pub selected: i32,
    pub coefficients: i32,
}
pub(super) fn joint_geometry(op: WorkspaceOperationView<'_>) -> FactResult<Option<JointGeometry>> {
    let WorkspaceOperationKindView::JointGroupSelection(spec) = op.kind else { return Ok(None); };
    if op.inputs.len() != 4 || op.outputs.len() != 3 {
        return Err(invalid());
    }
    let (p, shared, k) = (
        spec.selectable_groups(),
        spec.always_on_groups(),
        spec.top_k(),
    );
    let groups = p.checked_add(shared).ok_or_else(invalid)?;
    let coefficients = k.checked_add(shared).ok_or_else(invalid)?;
    let hidden = op.inputs.get(0).unwrap().shape();
    let d = *hidden.last().ok_or_else(invalid)?;
    if hidden.len() < 2
        || d <= 0
        || op.inputs.get(1).unwrap().shape() != [groups, d]
        || op.inputs.get(2).unwrap().shape() != [p]
        || op.inputs.get(3).unwrap().shape() != [1]
    {
        return Err(invalid());
    }
    let rows = op.inputs.get(0).unwrap().elements()? / d as u64;
    let rows_i32 = i32::try_from(rows).map_err(|_| invalid())?;
    for (i, (width, dtype)) in [
        (k, WorkspaceDtype::Uint32),
        (k, WorkspaceDtype::Float32),
        (shared, WorkspaceDtype::Float32),
    ]
    .into_iter()
    .enumerate()
    {
        if op.outputs.get(i).unwrap().shape() != [rows_i32, width]
            || op.outputs.get(i).unwrap().dtype() != dtype
        {
            return Err(invalid());
        }
    }
    if op
        .inputs
        .iter()
        .any(|l| l.dtype() != WorkspaceDtype::Float32)
    {
        return Ok(None);
    }
    Ok(Some(JointGeometry {rows, rows_i32, dimensions:d, groups,
        selectable:p, selected:k, coefficients}))
}

pub(super) fn emit(
    op: WorkspaceOperationView<'_>,
    a: MetalAllocationFacts,
    sink: &mut Emitter<'_>,
) -> FactResult<Option<WorkspaceOperationFacts>> {
    if matches!(op.kind, WorkspaceOperationKindView::GroupSelection { .. }) {
        return Ok(selector::emit(op, a, sink)?.map(|(tensor, _)| tensor));
    }
    let Some(g) = joint_geometry(op)? else { return Ok(None); };
    let JointGeometry { rows, rows_i32, dimensions: d, groups,
        selectable: p, selected: k, coefficients } = g;
    let primary = mul(rows, p as u64)?;
    let selected = mul(rows, k as u64)?;
    let joined = mul(rows, coefficients as u64)?;
    let primary_bytes = capacity(a, primary)?;
    let coefficient_bytes = capacity(a, joined)?;
    let scalar = capacity(a, 1)?;
    // Hidden flattening can copy; matmul covers promotions and stride-dependent
    // copies of the original dense weight without a per-token weight replica.
    let mut total = add(
        capacity(a, op.inputs.get(0).unwrap().elements()?)?,
        matmul_cost(&[rows_i32, d], &[d, groups], a)?,
    )?;
    // Native sigmoid and correction addition, with possible dtype promotion
    // of each participating operand. The original logits stay unbiased.
    total = add(total, add(mul(4, primary_bytes)?, capacity(a, p as u64)?)?)?;
    total = add(total, sort(a, primary, rows, p as u64)?)?;
    total = add(total, capacity(a, selected)?)?; // direct GatherAxis of raw logits
                                                 // Concatenate; negative/logaddexp/negative (including possible promotion);
                                                 // precise last-axis softmax; fixed and learned scale products/promotions.
    total = add(total, add(mul(10, coefficient_bytes)?, mul(5, scalar)?)?)?;
    let retained = add(primary_bytes, coefficient_bytes)?;
    let scratch = total.checked_sub(retained).ok_or_else(invalid)?;
    sink.output(Output::Allocate(primary_bytes))?;
    sink.output(Output::Allocate(coefficient_bytes))?;
    sink.output(Output::AliasOutput(1))?;
    sink.finish(scratch, format_args!("MLX Metal joint group selection: dense product, corrected sigmoid ranking, full native partition index backing, direct unbiased logit gather and shared log-sigmoid/softmax coefficient allocation; casts and child buffers retained through completion; page={} with bounded oversized reuse; no disjoint host numerical payload",a.page_size())).map(Some)
}

pub(super) fn selector_host_bound(
    op: &WorkspaceOperation,
    a: MetalAllocationFacts,
) -> Result<Option<WorkspaceHostBound>, Error> {
    Ok(selector::bounds(op, a)?.map(|(_, bytes)| WorkspaceHostBound {
        bytes,
        assumptions: "selected top-k routing: maximum simultaneously live Rust partition-ID, correction, keep-mask and validation-count payloads; native CPU tie partition writes shared allocator storage in place; request control arrays are borrowed and separately retained by admission; command/shape metadata excluded".into(),
    }))
}

pub(super) fn emit_selector_host(
    op: WorkspaceOperationView<'_>,
    a: MetalAllocationFacts,
    sink: &mut HostEmitter<'_>,
) -> FactResult<Option<WorkspaceHostFacts>> {
    let mut tensor = Emitter::count();
    let Some((_, bytes)) = selector::emit(op, a, &mut tensor)? else {
        return Ok(None);
    };
    sink.finish(bytes, format_args!("selected top-k routing: maximum simultaneously live Rust partition-ID, correction, keep-mask and validation-count payloads; native CPU tie partition writes shared allocator storage in place; request control arrays are borrowed and separately retained by admission; command/shape metadata excluded")).map(Some)
}

pub(super) fn ordinary_selector_error(
    op: &WorkspaceOperation,
    error: MlxWorkspaceFactError,
) -> Error {
    selector::ordinary_error(op, error)
}

#[cfg(test)]
mod tests;
