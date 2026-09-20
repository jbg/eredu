//! Allocation bounds for the selected local/pooled native mechanisms. Two-term
//! einsums lower to batch_tensordot, with input and output reshape copies around
//! ordinary matmul. No bound scales a gather by the complete broadcast source.

use super::facts::{self, add, buffer_capacity, mul, Aliases, Emitter, FactResult, Output};
use super::{
    matrix::einsum_matmul_cost_fixed as einsum_cost,
    reduction::{capacity_fixed as capacity, sum_cost_fixed as sum_cost},
    *,
};
use crate::backend::nn::attention::pooled_mask_shapes_fixed;

fn invalid() -> MlxWorkspaceFactError {
    MlxWorkspaceFactError::descriptor("invalid Metal pooled workspace descriptor")
}
fn product(shape: &[i32]) -> FactResult<u64> {
    shape
        .iter()
        .try_fold(1, |n, &d| mul(n, u64::try_from(d).map_err(|_| invalid())?))
}
fn width_sum(a: i32, b: i32) -> FactResult<i32> {
    a.checked_add(b).ok_or_else(invalid)
}
fn positive(scale: f32) -> FactResult<()> {
    if scale.is_finite() && scale > 0. {
        Ok(())
    } else {
        Err(invalid())
    }
}
fn broadcast(shape: &[i32], target: &[i32]) -> FactResult<()> {
    if shape.len() > target.len()
        || shape
            .iter()
            .rev()
            .zip(target.iter().rev())
            .any(|(a, b)| a != b && *a != 1)
    {
        return Err(invalid());
    }
    Ok(())
}
fn floating(layouts: WorkspaceLayoutList<'_>) -> bool {
    layouts.iter().all(|x| x.dtype() == WorkspaceDtype::Float32)
}
fn index(layout: WorkspaceLayoutView<'_>) -> bool {
    matches!(
        layout.dtype(),
        WorkspaceDtype::Int32 | WorkspaceDtype::Uint32
    )
}
fn query(op: WorkspaceOperationView<'_>, inputs: usize) -> FactResult<[i32; 4]> {
    if op.inputs.len() != inputs || op.outputs.len() != 1 {
        return Err(invalid());
    }
    let q: [i32; 4] = op
        .inputs
        .get(0)
        .unwrap()
        .shape()
        .try_into()
        .map_err(|_| invalid())?;
    if q.iter().any(|n| *n <= 0) {
        return Err(invalid());
    }
    Ok(q)
}
fn output(op: WorkspaceOperationView<'_>, shape: &[i32], dtype: WorkspaceDtype) -> FactResult<()> {
    if op.outputs.get(0).unwrap().shape() != shape || op.outputs.get(0).unwrap().dtype() != dtype {
        return Err(invalid());
    }
    Ok(())
}
fn mask_cost(
    mask: Option<WorkspaceLayoutView<'_>>,
    target: &[i32],
    a: NativeAllocationFacts,
) -> FactResult<Option<u64>> {
    let Some(mask) = mask else {
        return Ok(Some(0));
    };
    broadcast(mask.shape(), target)?;
    if !matches!(mask.dtype(), WorkspaceDtype::Bool | WorkspaceDtype::Float32) {
        return Ok(None);
    }
    // Where/add result and possible score promotion, mask conversion, and
    // finite-min scalar construction/cast. Broadcast masks remain views.
    Ok(Some(add(
        add(
            mul(2, capacity(a, product(target)?)?)?,
            capacity(a, mask.elements()?)?,
        )?,
        mul(2, capacity(a, 1)?)?,
    )?))
}
fn finish(
    total: u64,
    retained: u64,
    a: NativeAllocationFacts,
    detail: &str,
    alias_query: bool,
    sink: &mut Emitter<'_>,
) -> FactResult<WorkspaceOperationFacts> {
    sink.output(if alias_query {
        Output::AllocateOrAliasInputs {
            bytes: retained,
            inputs: Aliases::Slice(&[0]),
        }
    } else {
        Output::Allocate(retained)
    })?;
    sink.finish(total.checked_sub(retained).ok_or_else(invalid)?, format_args!("selected MLX Metal pooled mechanism: {detail}; all child buffers retained through completion, including casts and reshape copies; page={} with bounded oversized reuse; active native payload only", a.page_size()))
}

pub(super) fn operation_bound(
    op: &WorkspaceOperation,
    a: NativeAllocationFacts,
    blocks: Option<u32>,
) -> Result<Option<WorkspaceOperationBound>, Error> {
    facts::ordinary(|sink| emit(op.as_view(), a, blocks, sink))
}

pub(super) fn emit(
    op: WorkspaceOperationView<'_>,
    a: NativeAllocationFacts,
    blocks: Option<u32>,
    sink: &mut Emitter<'_>,
) -> FactResult<Option<WorkspaceOperationFacts>> {
    use WorkspaceOperationKindView as K;
    match op.kind {
        K::PooledAttention {
            scale,
            local_mask,
            pooled_mask,
            sinks,
        } => pooled(op, a, blocks, scale, local_mask, pooled_mask, sinks, sink),
        K::IndexedAttention {
            scale,
            local_mask,
            pooled_mask,
            sinks,
        } => indexed(op, a, scale, local_mask, pooled_mask, sinks, sink),
        K::PooledPositions { .. } => positions(op, a, sink),
        K::GatherPooledMask => gather(op, a, sink).map(Some),
        _ => Ok(None),
    }
}

fn pooled(
    op: WorkspaceOperationView<'_>,
    a: NativeAllocationFacts,
    blocks: Option<u32>,
    scale: f32,
    lm: bool,
    pm: bool,
    sinks: bool,
    sink: &mut Emitter<'_>,
) -> FactResult<Option<WorkspaceOperationFacts>> {
    positive(scale)?;
    let q = query(
        op,
        3 + usize::from(lm) + usize::from(pm) + usize::from(sinks),
    )?;
    let [b, h, n, d] = q;
    let l = op.inputs.get(1).unwrap().shape();
    let p = op.inputs.get(2).unwrap().shape();
    if l.len() != 3 || p.len() != 3 || l[0] != b || p[0] != b || l[2] != d || p[2] != d {
        return Err(invalid());
    }
    let keys = width_sum(l[1], p[1])?;
    if keys <= 0 {
        return Err(invalid());
    }
    output(op, &q, WorkspaceDtype::Float32)?;
    if !floating(op.inputs.slice(0..3).unwrap()) {
        return Ok(None);
    }
    let local = lm.then(|| op.inputs.get(3).unwrap());
    let pooled = pm.then(|| op.inputs.get(3 + usize::from(lm)).unwrap());
    let shapes = pooled_mask_shapes_fixed(
        &q,
        l[1],
        p[1],
        local.map(|x| x.shape()),
        pooled.map(|x| x.shape()),
    )
    .map_err(MlxWorkspaceFactError::descriptor)?;
    // Concatenation may promote each floating source and always materializes
    // the combined K=V bank. Expand-dims are aliasing views.
    let bank_shape = [b, 1, keys, d];
    let bank = WorkspaceLayoutView::new(&bank_shape, WorkspaceDtype::Float32)?;
    let mut total = add(
        add(
            capacity(a, op.inputs.get(1).unwrap().elements()?)?,
            capacity(a, op.inputs.get(2).unwrap().elements()?)?,
        )?,
        capacity(a, bank.elements()?)?,
    )?;
    let mut inputs = [op.inputs.get(0).unwrap(), bank, bank, bank, bank];
    let mut input_count = 3;
    let mask_shape;
    if let Some((ls, ps)) = shapes {
        if local
            .into_iter()
            .chain(pooled)
            .any(|x| !matches!(x.dtype(), WorkspaceDtype::Bool | WorkspaceDtype::Float32))
        {
            return Ok(None);
        }
        let dtype = if local
            .into_iter()
            .chain(pooled)
            .any(|x| x.dtype() == WorkspaceDtype::Float32)
        {
            WorkspaceDtype::Float32
        } else {
            WorkspaceDtype::Bool
        };
        mask_shape = [ls[0], ls[1], ls[2], keys];
        let mask = WorkspaceLayoutView::new(&mask_shape, dtype)?;
        // Each source can be cast after broadcasting, plus missing true scalar
        // operands and the joined mask. At-most-F32 capacities cover all cases.
        total = add(
            total,
            add(
                mul(
                    2,
                    add(capacity(a, product(&ls)?)?, capacity(a, product(&ps)?)?)?,
                )?,
                add(capacity(a, mask.elements()?)?, mul(4, capacity(a, 1)?)?)?,
            )?,
        )?;
        inputs[input_count] = mask;
        input_count += 1;
    }
    if sinks {
        inputs[input_count] = op.inputs.last().unwrap();
        input_count += 1;
    }
    let child = WorkspaceOperationView {
        kind: WorkspaceOperationKindView::Attention {
            causal: false,
            window: None,
            sinks,
            softcap: false,
            arithmetic: eredu_nn::AttentionArithmetic::Fused,
        },
        inputs: WorkspaceLayoutList::Views(&inputs[..input_count]),
        outputs: op.outputs,
    };
    let mut child_output = Emitter::count();
    let Some(bound) = attention::emit(child, a, blocks, &mut child_output)? else {
        return Ok(None);
    };
    let (retained, alias) = match child_output.first_output() {
        Some(WorkspaceOutputEffect::Allocate(bytes)) => (bytes, false),
        Some(WorkspaceOutputEffect::AllocateOrAliasInputs { bytes, .. }) => (bytes, true),
        _ => return Err(invalid()),
    };
    total = add(total, add(retained, bound.scratch_bytes)?)?;
    let _ = (h, n);
    finish(total, retained, a, "promoted local/pooled bank concatenation, broadcast-normalized masks, and the retained fused/fallback SDPA mechanism", alias, sink).map(Some)
}

pub(super) fn indexed_geometry(
    op: WorkspaceOperationView<'_>,
) -> FactResult<crate::backend::nn::attention::indexed::Geometry> {
    let WorkspaceOperationKindView::IndexedAttention {
        scale,
        local_mask,
        pooled_mask,
        sinks,
    } = op.kind
    else {
        return Err(invalid());
    };
    positive(scale)?;
    query(
        op,
        6 + usize::from(local_mask) + usize::from(pooled_mask) + usize::from(sinks),
    )?;
    let sources = [
        op.inputs.get(0).unwrap().shape(),
        op.inputs.get(1).unwrap().shape(),
        op.inputs.get(2).unwrap().shape(),
        op.inputs.get(3).unwrap().shape(),
        op.inputs.get(4).unwrap().shape(),
        op.inputs.get(5).unwrap().shape(),
    ];
    let g = crate::backend::nn::attention::indexed::Geometry::new(
        sources,
        scale,
        sinks.then(|| op.inputs.last().unwrap().shape()),
    )
    .map_err(|_| invalid())?;
    if g.shape.local < 0 || g.shape.value_dimensions <= 0 || !index(op.inputs.get(5).unwrap()) {
        return Err(invalid());
    }
    output(op, &g.output(), WorkspaceDtype::Float32)?;
    Ok(g)
}

fn indexed(
    op: WorkspaceOperationView<'_>,
    a: NativeAllocationFacts,
    _scale: f32,
    lm: bool,
    pm: bool,
    sinks: bool,
    sink: &mut Emitter<'_>,
) -> FactResult<Option<WorkspaceOperationFacts>> {
    let g = indexed_geometry(op)?;
    let shape = g.shape;
    let (b, h, q, d, l, v, s) = (
        shape.batch,
        shape.heads,
        shape.queries,
        shape.key_dimensions,
        shape.local,
        shape.value_dimensions,
        shape.selected,
    );
    if !floating(op.inputs.slice(0..5).unwrap()) {
        return Ok(None);
    }
    let local_mask = lm.then(|| op.inputs.get(6).unwrap());
    let pooled_mask = pm.then(|| op.inputs.get(6 + usize::from(lm)).unwrap());
    let Some(local_mask) = mask_cost(local_mask, &[b, h, q, l], a)? else {
        return Ok(None);
    };
    let Some(pooled_mask) = mask_cost(pooled_mask, &[b, h, q, s], a)? else {
        return Ok(None);
    };
    let r = |shape: &[i32]| capacity(a, product(shape)?);
    // GatherAxis reads the broadcast sources and indices with their strides;
    // it allocates only the selected keys and values, not B*Q*P source copies.
    let mut total = add(r(&[b, q, s, d])?, r(&[b, q, s, v])?)?;
    // Query scaling: result, possible input/scalar promotion, scalar itself.
    total = add(
        total,
        add(mul(2, r(&[b, h, q, d])?)?, mul(2, capacity(a, 1)?)?)?,
    )?;
    let rows = g.rows;
    total = add(
        total,
        einsum_cost(&[b, rows, d], &[b, d, l], &[b, h, q, l], a)?,
    )?;
    total = add(
        total,
        einsum_cost(&[b, q, h, d], &[b, q, d, s], &[b, h, q, s], a)?,
    )?;
    total = add(total, add(local_mask, pooled_mask)?)?;
    let joined = g.joined;
    let joined_count = product(&[b, h, q, joined])?;
    // Concatenate may cast both score sources. Final-axis precise softmax
    // allocates one output/contiguous buffer, without a full F32 widening.
    total = add(
        total,
        add(
            add(r(&[b, h, q, l])?, r(&[b, h, q, s])?)?,
            mul(2, capacity(a, joined_count)?)?,
        )?,
    )?;
    if sinks {
        let sink = op.inputs.last().unwrap();
        if sink.shape() != [h] {
            return Err(invalid());
        }
        if sink.dtype() != WorkspaceDtype::Float32 {
            return Ok(None);
        }
        total = add(total, add(capacity(a, h as u64)?, r(&[b, h, q, 1])?)?)?;
    }
    total = add(
        total,
        einsum_cost(&[b, rows, l], &[b, l, v], &[b, h, q, v], a)?,
    )?;
    total = add(
        total,
        einsum_cost(&[b, q, h, s], &[b, q, s, v], &[b, h, q, v], a)?,
    )?;
    let retained = r(&[b, h, q, v])?;
    total = add(total, mul(3, retained)?)?; // add and possible context promotions
    finish(total,retained,a,"two direct selected gathers; four exact batched einsum contractions; masked local/selected scores, shared sink softmax and context addition",false, sink).map(Some)
}

pub(super) fn positions_geometry(
    op: WorkspaceOperationView<'_>,
) -> FactResult<(
    crate::backend::nn::attention::pooled_positions::Geometry,
    bool,
)> {
    let WorkspaceOperationKindView::PooledPositions {
        top_k,
        scale,
        head_scale,
        masked,
    } = op.kind
    else {
        return Err(invalid());
    };
    positive(scale)?;
    positive(head_scale)?;
    if op.inputs.len() != 3 + usize::from(masked) || op.outputs.len() != 1 {
        return Err(invalid());
    }
    let g = crate::backend::nn::attention::pooled_positions::Geometry::new(
        op.inputs.get(0).unwrap().shape(),
        op.inputs.get(1).unwrap().shape(),
        op.inputs.get(2).unwrap().shape(),
        top_k,
        scale,
        head_scale,
    )
    .map_err(|_| invalid())?;
    output(op, &g.output(), WorkspaceDtype::Uint32)?;
    if masked {
        broadcast(
            op.inputs.get(3).unwrap().shape(),
            &[g.batch, g.queries, g.pooled],
        )?;
    }
    Ok((g, masked))
}
fn positions(
    op: WorkspaceOperationView<'_>,
    a: NativeAllocationFacts,
    sink: &mut Emitter<'_>,
) -> FactResult<Option<WorkspaceOperationFacts>> {
    let (g, masked) = positions_geometry(op)?;
    let (b, h, q, d, p) = (g.batch, g.heads, g.queries, g.dimensions, g.pooled);
    let w = op.inputs.get(2).unwrap().shape();
    if !floating(op.inputs.slice(0..3).unwrap())
        || masked && op.inputs.get(3).unwrap().dtype() != WorkspaceDtype::Bool
    {
        return Ok(None);
    }
    if p == 0 {
        let retained = capacity(a, 0)?;
        return finish(
            mul(3, retained)?,
            retained,
            a,
            "empty pooled selection uses a zero-length U32 fill with scalar backing",
            false,
            sink,
        )
        .map(Some);
    }
    let rows = product(&[b, q])?;
    let scores = product(&[b, h, q, p])?;
    let reduced = product(&[b, q, p])?;
    // Explicit F32 input casts precede the einsum; its internal matmul casts
    // are conservatively retained too. Weight scaling is per token/head.
    let mut total = add(
        capacity(a, product(&[b, h, q, d])?)?,
        capacity(a, product(&[b, p, d])?)?,
    )?;
    total = add(
        total,
        einsum_cost(&[b, g.flattened_queries, d], &[b, d, p], &[b, h, q, p], a)?,
    )?;
    // Maximum, score scaling and weighted product each have an output;
    // their F32 scalar operands cannot widen the already-F32 scores.
    total = add(
        total,
        add(mul(3, capacity(a, scores)?)?, mul(3, capacity(a, 1)?)?)?,
    )?;
    total = add(total, mul(2, capacity(a, product(w)?)?)?)?;
    total = add(total, sum_cost(a, scores, reduced, h as u64)?)?;
    if masked {
        total = add(total, add(capacity(a, reduced)?, capacity(a, 1)?)?)?;
    }
    total = add(total, sampling::sort_fixed(a, reduced, rows, p as u64)?)?;
    // The returned top-k slice retains the entire U32 argpartition output.
    // Charge that allocation as retained, not only its logical selected tail.
    finish(total,capacity(a,reduced)?,a,"F32 einsum, nonnegative scoring, head-weight reduction, optional mask and native merge-sort partition; selected tail retains the full pooled-width index buffer",false, sink).map(Some)
}

pub(super) fn gather_geometry(
    op: WorkspaceOperationView<'_>,
) -> FactResult<crate::backend::nn::attention::gather_mask::Geometry> {
    if op.inputs.len() != 2 || op.outputs.len() != 1 {
        return Err(invalid());
    }
    let mask = op.inputs.get(0).unwrap();
    let indices = op.inputs.get(1).unwrap();
    let m = mask.shape();
    let i = indices.shape();
    let geometry = crate::backend::nn::attention::gather_mask::Geometry::new(m, i)
        .map_err(|_| invalid())?;
    if m[0] != i[1]
        || !index(indices)
        || m[1] == 0 && indices.elements()? != 0
    {
        return Err(invalid());
    }
    output(op, &geometry.output, mask.dtype())?;
    Ok(geometry)
}
fn gather(
    op: WorkspaceOperationView<'_>,
    a: NativeAllocationFacts,
    sink: &mut Emitter<'_>,
) -> FactResult<WorkspaceOperationFacts> {
    gather_geometry(op)?;
    let retained = buffer_capacity(a, op.outputs.get(0).unwrap().bytes()?.max(4))?;
    finish(retained,retained,a,"broadcast/expand views and direct strided GatherAxis output; no source-table or index replication",false, sink)
}

#[cfg(test)]
mod tests;
