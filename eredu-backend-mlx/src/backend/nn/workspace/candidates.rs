//! Concrete Metal buffers for the shared raw candidate extraction program.
use super::facts::{self, add, buffer_capacity, Emitter, FactResult, Output};
use super::reduction::{capacity_fixed as capacity, sum_cost_fixed as sum_cost};
use super::*;
pub(super) fn operation_bound(
    op: &WorkspaceOperation,
    a: NativeAllocationFacts,
) -> Result<Option<WorkspaceOperationBound>, Error> {
    facts::ordinary(|sink| emit(op.as_view(), a, sink))
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Geometry { pub(super) vocabulary: u32, pub(super) count: u32, pub(super) rows: i32 }

/// Same exact worker input/output descriptor for both native mechanism quotes.
pub(super) fn geometry(op: WorkspaceOperationView<'_>) -> FactResult<Option<Geometry>> {
    let WorkspaceOperationKindView::CandidateExtraction { vocabulary, count } = op.kind else {
        return Ok(None);
    };
    let invalid =
        || MlxWorkspaceFactError::descriptor("invalid raw candidate extraction descriptor");
    let Some([input]) = op.inputs.array() else {
        return Err(invalid());
    };
    let Some([ids, scores]) = op.outputs.array() else {
        return Err(invalid());
    };
    if vocabulary == 0
        || vocabulary > i32::MAX as u32
        || count == 0
        || count > vocabulary
        || input.dtype() != WorkspaceDtype::Float32
        || input.shape().len() != 3
        || input.shape()[0] != 1
        || input.shape()[1] <= 0
        || input.shape()[2] != vocabulary as i32
        || ids.shape() != [count as i32]
        || ids.dtype() != WorkspaceDtype::Uint32
        || scores.shape() != [count as i32]
        || scores.dtype() != WorkspaceDtype::Float32
    {
        return Err(invalid());
    }
    Ok(Some(Geometry { vocabulary, count, rows: input.shape()[1] }))
}

pub(super) fn emit(
    op: WorkspaceOperationView<'_>,
    a: NativeAllocationFacts,
    sink: &mut Emitter<'_>,
) -> FactResult<Option<WorkspaceOperationFacts>> {
    let Some(Geometry { vocabulary, count, .. }) = geometry(op)? else { return Ok(None); };
    let width = u64::from(vocabulary);
    // Direct row view; possible half->F32 cast. MLX isfinite expands into
    // two scalar infinities, isposinf/isneginf, their OR, isnan, a second
    // OR and NOT. Account each actual output even if a backend can fuse it.
    // U32 mask, sum and actual F32 argsort/merge storage follow.
    let row_cast = capacity(a, width)?;
    let positive_infinity = capacity(a, 1)?;
    let negative_infinity = capacity(a, 1)?;
    let is_positive_infinity = buffer_capacity(a, width)?;
    let is_negative_infinity = buffer_capacity(a, width)?;
    let is_infinite = buffer_capacity(a, width)?;
    let is_nan = buffer_capacity(a, width)?;
    let is_nonfinite = buffer_capacity(a, width)?;
    let is_finite = buffer_capacity(a, width)?;
    let mask = capacity(a, width)?;
    let sum = sum_cost(a, width, 1, width)?;
    let sorted = super::sampling::sort_fixed(a, width, 1, width)?;
    let scratch = [
        row_cast,
        positive_infinity,
        negative_infinity,
        is_positive_infinity,
        is_negative_infinity,
        is_infinite,
        is_nan,
        is_nonfinite,
        is_finite,
        mask,
        sum,
        sorted,
    ]
    .into_iter()
    .try_fold(0, add)?;
    // IDs contiguous may alias the full sort buffer (already retained above) or
    // allocate K. Native take uses trusted ArgSort IDs and reads strided source
    // directly; no high-level embedding validation program is substituted.
    let selected = capacity(a, u64::from(count))?;
    sink.output(Output::Allocate(selected))?;
    sink.output(Output::Allocate(selected))?;
    sink.finish(scratch, format_args!("selected MLX Metal terminal-row raw candidate extraction: actual cast, two scalar infinities, six Bool finite-test outputs, U32 mask/sum, full U32 sort and >2048 merge ping-pong/partitions, optional K-ID contiguity and K-F32 gather; full sort backing retained even when K slice aliases it. Source backing, host candidate destination and graph/encoder/control storage are separate. CPU stable_sort scratch is not covered")).map(Some)
}
