//! Inference attention buffers for the selected Metal equations and kernels.
//! Shapes select fused vector/full attention or the ordinary product equation.
//! All possible casts/copies remain charged through the enclosing completion.

use super::{
    matrix::matmul_cost_fixed as matmul_cost,
    reduction::{capacity_fixed as capacity, sum_cost_fixed as sum_cost},
    *,
};
use crate::backend::nn::attention::{input_score_query_step, INPUT_SCORE_ROW_BUDGET};
use eredu_nn::{operation_geometry::SlidingAttentionGeometry, AttentionArithmetic};

use super::facts::{self, add, mul, Aliases, Emitter, FactResult, Output};

pub(super) mod blockwise;

#[derive(Clone, Copy)]
struct Geometry {
    b: i32,
    h: i32,
    kv: i32,
    q: i32,
    k: i32,
    d: i32,
    v: i32,
}
impl Geometry {
    fn rows(self) -> FactResult<u64> {
        mul(mul(self.b as u64, self.h as u64)?, self.q as u64)
    }
    fn scores(self) -> FactResult<u64> {
        mul(self.rows()?, self.k as u64)
    }
    fn output(self) -> FactResult<u64> {
        mul(self.rows()?, self.v as u64)
    }
    fn query(self) -> FactResult<u64> {
        mul(self.rows()?, self.d as u64)
    }
    fn keys(self, expanded: bool) -> FactResult<u64> {
        mul(
            mul(
                mul(
                    self.b as u64,
                    if expanded { self.h } else { self.kv } as u64,
                )?,
                self.k as u64,
            )?,
            self.d as u64,
        )
    }
    fn values(self, expanded: bool) -> FactResult<u64> {
        mul(
            mul(
                mul(
                    self.b as u64,
                    if expanded { self.h } else { self.kv } as u64,
                )?,
                self.k as u64,
            )?,
            self.v as u64,
        )
    }
    fn fused(self, mask: Mask) -> bool {
        let vector = self.q <= 8
            && self.q <= self.k
            && i64::from(self.q) * i64::from(self.h / self.kv) <= 32
            && ((self.d == self.v && matches!(self.d, 64 | 96 | 128 | 256))
                || self.d == 192 && self.v == 128);
        let full = self.q > 8
            && self.d == self.v
            && matches!(self.d, 64 | 80 | 128)
            && (!matches!(mask, Mask::Causal) || self.q <= self.k);
        vector || full
    }
}
#[derive(Clone, Copy)]
enum Mask {
    None,
    Causal,
    Boolean,
    Additive(u64),
}
impl Mask {
    fn array(self) -> bool {
        matches!(self, Self::Boolean | Self::Additive(_))
    }
    fn boolean(self) -> bool {
        matches!(self, Self::Boolean | Self::Causal)
    }
}

pub(super) fn operation_bound(
    operation: &WorkspaceOperation,
    allocation: MetalAllocationFacts,
    blocks: Option<u32>,
) -> Result<Option<WorkspaceOperationBound>, Error> {
    facts::ordinary(|sink| emit(operation.as_view(), allocation, blocks, sink))
}

pub(super) fn emit(
    operation: WorkspaceOperationView<'_>,
    allocation: MetalAllocationFacts,
    blocks: Option<u32>,
    sink: &mut Emitter<'_>,
) -> FactResult<Option<WorkspaceOperationFacts>> {
    if matches!(operation.kind, WorkspaceOperationKindView::BlockwiseAttention { .. }) {
        return blockwise::emit(operation, allocation, sink);
    }
    let WorkspaceOperationKindView::Attention {
        causal,
        window,
        sinks,
        softcap,
        arithmetic,
    } = operation.kind
    else {
        return Ok(None);
    };
    let base = 3 + usize::from(sinks);
    if !(base..=base + 1).contains(&operation.inputs.len()) || operation.outputs.len() != 1 {
        return Err(MlxWorkspaceFactError::descriptor(
            "invalid Metal attention descriptor",
        ));
    }
    let inputs = operation.inputs;
    let q = inputs.get(0).unwrap().shape();
    let k = inputs.get(1).unwrap().shape();
    let v = inputs.get(2).unwrap().shape();
    if q.len() != 4
        || k.len() != 4
        || v.len() != 4
        || q.iter().chain(k).chain(v).any(|n| *n <= 0)
        || q[0] != k[0]
        || k[..3] != v[..3]
        || q[3] != k[3]
        || q[1] % k[1] != 0
    {
        return Err(MlxWorkspaceFactError::descriptor(
            "invalid Metal attention geometry",
        ));
    }
    if inputs
        .slice(0..3)
        .unwrap()
        .iter()
        .any(|x| x.dtype() != WorkspaceDtype::Float32)
        || sinks && inputs.last().unwrap().dtype() != WorkspaceDtype::Float32
    {
        return Ok(None);
    }
    let g = Geometry {
        b: q[0],
        h: q[1],
        kv: k[1],
        q: q[2],
        k: k[2],
        d: q[3],
        v: v[3],
    };
    if sinks && inputs.last().unwrap().shape() != [g.h] {
        return Err(MlxWorkspaceFactError::descriptor(
            "invalid Metal attention sink shape",
        ));
    }
    let mut mask = if causal { Mask::Causal } else { Mask::None };
    if inputs.len() > base {
        let layout = inputs.get(3).unwrap();
        let target = [g.b, g.h, g.q, g.k];
        if layout.shape().len() > 4
            || layout
                .shape()
                .iter()
                .rev()
                .zip(target.iter().rev())
                .any(|(x, y)| x != y && *x != 1)
        {
            return Err(MlxWorkspaceFactError::descriptor(
                "Metal attention mask does not broadcast to scores",
            ));
        }
        mask = match layout.dtype() {
            WorkspaceDtype::Bool => Mask::Boolean,
            WorkspaceDtype::Float32 => Mask::Additive(layout.elements()?),
            _ => return Ok(None),
        };
    }
    let output_storage;
    let output_shape: &[i32] = if let Some((window, offset)) = window {
        SlidingAttentionGeometry::new_fixed(g.q, g.k, window, offset)?;
        output_storage = [
            g.b,
            g.q,
            g.h.checked_mul(g.v)
                .ok_or(MlxWorkspaceFactError::descriptor(
                    "Metal attention joined heads overflow",
                ))?,
            0,
        ];
        &output_storage[..3]
    } else {
        output_storage = [g.b, g.h, g.q, g.v];
        &output_storage
    };
    if operation.outputs.first().unwrap().shape() != output_shape
        || operation.outputs.first().unwrap().dtype() != WorkspaceDtype::Float32
    {
        return Err(MlxWorkspaceFactError::descriptor(
            "Metal attention output geometry or dtype differs",
        ));
    }
    let output = capacity(allocation, g.output()?)?;
    let total = if let Some((window, offset)) = window {
        let origin = SlidingAttentionGeometry::new_fixed(g.q, g.k, window, offset)?.key_origin();
        if arithmetic == AttentionArithmetic::Fused && !softcap && offset == 0 && g.q <= window {
            // Native causal SDPA, head permutation, then possible reshape copy.
            add(
                fused_cost(g, Mask::Causal, sinks, allocation, blocks)?,
                output,
            )?
        } else {
            let mut total = 0;
            let mut start = 0;
            while start < g.q {
                let end = start + super::super::attention::SLIDING_QUERY_TILE.min(g.q - start);
                let abs = offset + start;
                let first = (abs - (window - 1)).max(origin);
                let tile = Geometry {
                    q: end - start,
                    k: offset + end - first,
                    ..g
                };
                total = add(
                    total,
                    causal_mask_cost(tile.q, abs - first, window - 1, allocation)?,
                )?;
                let Some(cost) = equation_cost(
                    tile,
                    Mask::Boolean,
                    sinks,
                    softcap,
                    arithmetic,
                    allocation,
                    blocks,
                )?
                else {
                    return Ok(None);
                };
                total = add(total, cost)?;
                start = end;
            }
            // Same-dtype chunk concatenation followed by head-joining reshape.
            add(total, mul(2, output)?)?
        }
    } else {
        let Some(cost) = equation_cost(g, mask, sinks, softcap, arithmetic, allocation, blocks)?
        else {
            return Ok(None);
        };
        cost
    };
    sink.output(
        if arithmetic == AttentionArithmetic::Fused
            && !softcap
            && g.d == g.v
            && (window.is_some() || g.fused(mask) && g.q <= 8)
        {
            Output::AllocateOrAliasInputs {
                bytes: output,
                inputs: Aliases::Slice(&[0]),
            }
        } else {
            Output::Allocate(output)
        },
    )?;
    sink.finish(total-output, format_args!("vendored MLX Metal forward inference attention; exact fused eligibility or explicit product/softmax equation; all compatible floating dtypes through F32, casts, layout copies and grouped-head replication; retained SDPA blocks={blocks:?}, device-default union when unset; sliding query tiles=256; input-score rows above 8192 use completed 256-key blocks with retained normalization/state/query outputs; other intermediates charged through the enclosing completion; page={} with bounded oversized reuse; active tensor buffers only, excluding cache/heap/driver/JIT and host allocations",allocation.page_size())).map(Some)
}

fn causal_mask_cost(
    q: i32,
    offset: i32,
    distance: i32,
    allocation: MetalAllocationFacts,
) -> FactResult<u64> {
    let shape = [
        q,
        offset
            .checked_add(q)
            .ok_or(MlxWorkspaceFactError::descriptor(
                "attention mask endpoint overflows",
            ))?,
    ];
    let output = WorkspaceLayoutView::new(&shape, WorkspaceDtype::Bool)?;
    let operation = WorkspaceOperationView {
        kind: WorkspaceOperationKindView::CausalMask(
            eredu_nn::operation_geometry::CausalMaskGeometry::new_fixed(q, offset, Some(distance))?,
        ),
        inputs: WorkspaceLayoutList::Views(&[]),
        outputs: WorkspaceLayoutList::Views(std::slice::from_ref(&output)),
    };
    let mut child = Emitter::count();
    let bound = basic::emit(operation, allocation, &mut child)?.ok_or(
        MlxWorkspaceFactError::descriptor("causal mask workspace unavailable"),
    )?;
    let result = match child.first_output() {
        Some(WorkspaceOutputEffect::Allocate(n))
        | Some(WorkspaceOutputEffect::AllocateOrAliasInputs { bytes: n, .. }) => n,
        _ => {
            return Err(MlxWorkspaceFactError::descriptor(
                "unexpected causal mask storage",
            ));
        }
    };
    add(result, bound.scratch_bytes)
}

fn equation_cost(
    g: Geometry,
    mask: Mask,
    sinks: bool,
    cap: bool,
    arithmetic: AttentionArithmetic,
    a: MetalAllocationFacts,
    blocks: Option<u32>,
) -> FactResult<Option<u64>> {
    if arithmetic == AttentionArithmetic::Fused && !cap {
        return fused_cost(g, mask, sinks, a, blocks).map(Some);
    }
    if arithmetic == AttentionArithmetic::InputScores
        && mul(g.q as u64, g.k as u64)? > INPUT_SCORE_ROW_BUDGET as u64
    {
        if g.k > INPUT_SCORE_ROW_BUDGET {
            return blockwise::input_score_cost(g, mask, sinks, cap, a).map(Some);
        }
        let step = input_score_query_step(g.k);
        let mut start = 0;
        let mut total = 0;
        while start < g.q {
            let q = step.min(g.q - start);
            let tile = Geometry { q, ..g };
            let tile_mask = match mask {
                Mask::Boolean => Mask::Boolean,
                Mask::Additive(_) => Mask::Additive(tile.scores()?),
                other => other,
            };
            total = add(total, explicit_cost(tile, tile_mask, sinks, cap, true, a)?)?;
            start += q;
        }
        return add(total, capacity(a, g.output()?)?).map(Some);
    }
    explicit_cost(
        g,
        mask,
        sinks,
        cap,
        arithmetic == AttentionArithmetic::InputScores,
        a,
    )
    .map(Some)
}

fn fused_cost(
    g: Geometry,
    mask: Mask,
    sinks: bool,
    a: MetalAllocationFacts,
    blocks: Option<u32>,
) -> FactResult<u64> {
    let r = |n| capacity(a, n);
    // Native front-end Q/K/V promotion, optional wrapper/native mask casts,
    // and sink promotion all precede the kernel or fallback equation.
    let mut total = add(
        add(r(g.query()?)?, r(g.keys(false)?)?)?,
        r(g.values(false)?)?,
    )?;
    if let Mask::Additive(n) = mask {
        total = add(total, mul(2, r(n)?)?)?;
    }
    if sinks {
        total = add(total, r(g.h as u64)?)?;
    }
    if !g.fused(mask) {
        return add(total, fallback_cost(g, mask, sinks, a)?);
    }
    total = add(
        total,
        add(
            add(r(g.query()?)?, r(g.keys(false)?)?)?,
            r(g.values(false)?)?,
        )?,
    )?;
    total = add(total, r(g.output()?)?)?;
    if mask.array() {
        total = add(total, r(g.scores()?)?)?;
    }
    if sinks {
        total = add(total, r(g.h as u64)?)?;
    }
    if g.q <= 8 && g.k >= 1024 {
        let blocks = u64::from(blocks.unwrap_or_else(|| default_blocks(g)));
        let rows = mul(g.rows()?, blocks)?;
        total = add(total, add(r(mul(rows, g.v as u64)?)?, mul(2, r(rows)?)?)?)?;
    }
    Ok(total)
}
fn default_blocks(g: Geometry) -> u32 {
    let n = g.k;
    let simds = i64::from(g.q) * i64::from(g.h / g.kv);
    let small = if n > 1024 && simds > 4 {
        match n {
            ..=8192 => 128,
            8193..=32768 => 256,
            32769..=65536 => 512,
            _ => 1024,
        }
    } else {
        64
    };
    let large = if simds <= 2 && n > 8192 {
        256
    } else if simds >= 6 && n >= 65536 {
        1024
    } else if simds >= 6 && n >= 16384 {
        512
    } else {
        128
    };
    small.max(large)
}
fn fallback_cost(g: Geometry, mask: Mask, sinks: bool, a: MetalAllocationFacts) -> FactResult<u64> {
    let r = |n| capacity(a, n);
    let mut total = add(mul(3, r(g.query()?)?)?, r(1)?)?;
    let grouped = g.h != g.kv;
    let shapes;
    let rank = if grouped {
        total = add(total, r(g.query()?)?)?;
        shapes = [
            [g.b, g.kv, g.h / g.kv, g.q, g.d],
            [g.b, g.kv, 1, g.d, g.k],
            [g.b, g.kv, 1, g.k, g.v],
            [g.b, g.kv, g.h / g.kv, g.q, g.k],
        ];
        5
    } else {
        shapes = [
            [g.b, g.h, g.q, g.d, 0],
            [g.b, g.h, g.d, g.k, 0],
            [g.b, g.h, g.k, g.v, 0],
            [g.b, g.h, g.q, g.k, 0],
        ];
        4
    };
    let [q, k, v, prob] = shapes.each_ref().map(|shape| &shape[..rank]);
    total = add(total, matmul_cost(q, k, a)?)?;
    let score = r(g.scores()?)?;
    if matches!(mask, Mask::Causal) {
        total = add(
            total,
            add(
                add(r(g.q as u64)?, r(g.k as u64)?)?,
                add(r(mul(g.q as u64, g.k as u64)?)?, mul(2, r(1)?)?)?,
            )?,
        )?;
    } else if grouped && mask.array() {
        total = add(total, score)?;
    }
    if !matches!(mask, Mask::None) {
        total = add(total, mul(if mask.boolean() { 4 } else { 3 }, score)?)?;
        if mask.boolean() {
            total = add(total, r(1)?)?;
        }
    }
    let width = add(g.k as u64, u64::from(sinks))?;
    let soft = r(mul(g.rows()?, width)?)?;
    if sinks {
        total = add(
            total,
            add(add(score, r(g.rows()?)?)?, add(soft, r(g.h as u64)?)?)?,
        )?;
    }
    total = add(total, soft)?;
    total = add(total, matmul_cost(prob, v, a)?)?;
    if grouped {
        total = add(total, r(g.output()?)?)?;
    }
    Ok(total)
}

fn explicit_cost(
    g: Geometry,
    mask: Mask,
    sinks: bool,
    cap: bool,
    input_scores: bool,
    a: MetalAllocationFacts,
) -> FactResult<u64> {
    let r = |n| capacity(a, n);
    let q = [g.b, g.h, g.q, g.d];
    let kt = [g.b, g.h, g.d, g.k];
    let p = [g.b, g.h, g.q, g.k];
    let v = [g.b, g.h, g.k, g.v];
    // Original reshape, grouped-head expansion reshape, and key/query casts.
    let mut total = add(
        add(r(g.keys(false)?)?, mul(2, r(g.keys(true)?)?)?)?,
        add(
            add(r(g.values(false)?)?, r(g.values(true)?)?)?,
            r(g.query()?)?,
        )?,
    )?;
    total = add(total, product_cost(&q, &kt, input_scores, false, a)?)?;
    let score = r(g.scores()?)?;
    total = add(total, add(mul(5, score)?, r(1)?)?)?;
    if cap {
        total = add(total, add(mul(9, score)?, mul(2, r(1)?)?)?)?;
    }
    if mask.boolean() {
        total = add(total, add(mul(4, score)?, r(1)?)?)?;
    } else if mask.array() {
        total = add(total, mul(5, score)?)?;
    }
    let soft = r(mul(g.rows()?, add(g.k as u64, u64::from(sinks))?)?)?;
    if sinks {
        total = add(
            total,
            add(
                mul(2, r(g.h as u64)?)?,
                add(add(score, r(g.rows()?)?)?, soft)?,
            )?,
        )?;
    }
    // Explicit F32 score cast and one custom-softmax input copy plus result.
    total = add(total, mul(3, soft)?)?;
    total = add(total, score)?; // restore probability input dtype
    total = add(total, product_cost(&p, &v, true, true, a)?)?;
    Ok(total)
}

fn product_cost(
    left: &[i32; 4],
    right: &[i32; 4],
    bf16: bool,
    columns: bool,
    a: MetalAllocationFacts,
) -> FactResult<u64> {
    let plain = matmul_cost(left, right, a)?;
    if !bf16 {
        return Ok(plain);
    }
    let elements = |shape: &[i32]| shape.iter().try_fold(1, |n, d| mul(n, *d as u64));
    let batches = usize::try_from(left[0])?
        .checked_mul(usize::try_from(left[1])?)
        .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
    let input = safemlx::RepeatedI32InputPlan::new(batches, usize::try_from(left[2])?)
        .ok_or_else(|| MlxWorkspaceFactError::descriptor("matrix group input extent overflow"))?;
    let rows = u64::try_from(input.elements())?;
    let l = capacity(a, elements(left)?)?;
    let r = capacity(a, elements(right)?)?;
    // One I32 source, filled directly in its final buffer. The remaining `ids`
    // terms below are actual cast/validation/copy buffers, not host staging.
    let ids = capacity(a, rows)?;
    // A BF16 attempt constructs both reshapes and group IDs before the QK
    // custom kernel checks width divisibility. Price the fallback attempt too.
    let fallback = add(plain, add(add(l, r)?, ids)?)?;
    if !columns && left[3] % 32 != 0 {
        return Ok(fallback);
    }
    let output = capacity(a, mul(rows, right[3] as u64)?)?;
    let custom = add(
        add(mul(2, add(add(l, r)?, ids)?)?, mul(3, ids)?)?,
        add(
            sum_cost(a, rows, 1, rows)?,
            add(mul(2, output)?, mul(2, capacity(a, 1)?)?)?,
        )?,
    )?;
    // Original grouped projection also retains its deferred domain predicate
    // and masks invalid IDs before the custom kernel. That safe-index branch
    // has more buffers than the ordinary immediate all() assertion above.
    // Reuse the same validation producer as grouped/embedding facts; its
    // scratch excludes the input source, which remains exactly one `ids`.
    let validation = super::indexing::embedding_validation_cost_fixed(
        rows,
        0,
        eredu_nn::EmbeddingLookupPolicy::Strict,
        a,
    )?;
    let original = add(
        add(mul(2, add(l, r)?)?, mul(3, ids)?)?,
        add(validation, mul(2, output)?)?,
    )?;
    Ok(fallback.max(custom).max(original))
}

#[cfg(test)]
mod tests;
