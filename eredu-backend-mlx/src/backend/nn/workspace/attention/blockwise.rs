//! Two-pass input-score attention. Each accumulator step waits for native
//! ownership retirement, so its workspace is a maximum across key blocks.
//! Finished query outputs remain live until the final concatenation.

use super::*;
use crate::backend::nn::attention::INPUT_SCORE_KEY_TILE;

pub(super) fn input_score_cost(
    geometry: Geometry,
    mask: Mask,
    sinks: bool,
    cap: bool,
    allocation: MetalAllocationFacts,
) -> FactResult<u64> {
    // This native path uses one query at a time when a complete key row is
    // larger than the input-score budget. The caller validates positive shapes.
    let query = Geometry { q: 1, ..geometry };
    let full = Geometry {
        k: INPUT_SCORE_KEY_TILE.min(geometry.k),
        ..query
    };
    let mut step = block_cost(full, mask, sinks, cap, allocation)?;
    let tail = geometry.k % INPUT_SCORE_KEY_TILE;
    if tail != 0 {
        step = step.max(block_cost(
            Geometry { k: tail, ..query },
            mask,
            sinks,
            cap,
            allocation,
        )?);
    }
    let output = capacity(allocation, query.output()?)?;
    // One query cast, the previous normalization pair and value accumulator
    // coexist with the next step's newly allocated buffers. Previously finished
    // queries may retain both their F32 accumulator and output dtype conversion.
    let during_blocks = add(
        add(
            capacity(allocation, query.query()?)?,
            add(mul(2, capacity(allocation, query.rows()?)?)?, output)?,
        )?,
        add(step, mul(mul(2, geometry.q as u64 - 1)?, output)?)?,
    )?;
    // The final casts/concatenation run after all block scopes have settled.
    let during_concat = add(
        mul(mul(2, geometry.q as u64)?, output)?,
        capacity(allocation, geometry.output()?)?,
    )?;
    Ok(during_blocks.max(during_concat))
}

fn block_cost(
    g: Geometry,
    mask: Mask,
    sinks: bool,
    cap: bool,
    a: MetalAllocationFacts,
) -> FactResult<u64> {
    let (common, normalization, values) = block_costs(g, mask, sinks, cap, a, true, None, false)?;
    add(common, normalization.max(values))
}

fn block_costs(
    g: Geometry,
    mask: Mask,
    sinks: bool,
    cap: bool,
    a: MetalAllocationFacts,
    input_scores: bool,
    absolute: Option<eredu_nn::operation_geometry::AbsoluteAttentionMaskGeometry>,
    bias: bool,
) -> FactResult<(u64, u64, u64)> {
    let r = |n| capacity(a, n);
    let scores = r(g.scores()?)?;
    let rows = r(g.rows()?)?;
    let output = r(g.output()?)?;
    let scalar = r(1)?;
    // Native grouped-head reshape/replication and F32 key/value casts. Include
    // the value transform even in the normalization pass where it can stay lazy.
    let mut common = add(r(g.keys(true)?)?, r(g.values(true)?)?)?;
    if g.h != g.kv {
        common = add(
            common,
            add(
                add(r(g.keys(false)?)?, r(g.keys(true)?)?)?,
                add(r(g.values(false)?)?, r(g.values(true)?)?)?,
            )?,
        )?;
    }
    common = add(
        common,
        matmul_cost(&[g.b, g.h, g.q, g.d], &[g.b, g.h, g.d, g.k], a)?,
    )?;
    // Score output rounding, F32 restoration, scaling and final input rounding.
    common = add(
        common,
        if input_scores {
            add(mul(6, scores)?, scalar)?
        } else {
            add(r(g.query()?)?, scalar)?
        },
    )?;
    if cap {
        common = add(common, add(mul(8, scores)?, mul(2, scalar)?)?)?;
    }
    // Actual source-qualified page mask, or existing contiguous all-true source.
    common = add(
        common,
        match absolute {
            Some(geometry) => stages::mask_bytes(geometry, a)?,
            None => add(mul(2, r(mul(g.q as u64, g.k as u64)?)?)?, scalar)?,
        },
    )?;
    if bias {
        common = add(common, mul(3, scores)?)?;
    }

    let mask_cost = match mask {
        Mask::None => add(mul(4, scores)?, scalar)?,    // where
        Mask::Boolean => add(mul(7, scores)?, scalar)?, // logical-and + where
        Mask::Additive(_) => add(mul(16, scores)?, mul(2, scalar)?)?, // -inf comparison, not, and, cast, add, where
        Mask::Causal => {
            return Err(MlxWorkspaceFactError::descriptor(
                "input-score blocks require the explicit causal mask",
            ));
        }
    };
    common = add(common, add(mask_cost, scores)?)?; // F32 masked scores

    // Normalization pass: max, exp(score-max), masked weights, sum, zero value
    // accumulator and the online recurrence. Scalar/binary/unary counts include
    // possible casts; the reductions use the existing native partial-array fact.
    let normalization = add(
        add(
            mul(2, sum_cost(a, g.scores()?, g.rows()?, g.k as u64)?)?,
            mul(9, scores)?,
        )?,
        add(add(mul(11, output)?, scalar)?, mul(22, rows)?)?,
    )?;
    let normalization = if !input_scores {
        // Fused normalization computes weighted values instead of a zero
        // accumulator. Retain the same shared merge populations.
        add(
            normalization,
            matmul_cost(&[g.b, g.h, g.q, g.k], &[g.b, g.h, g.k, g.v], a)?,
        )?
        .checked_sub(output)
        .and_then(|n| n.checked_sub(scalar))
        .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?
    } else {
        normalization
    };
    let normalization = if sinks {
        add(normalization, mul(2, r(g.h as u64)?)?)?
    } else {
        normalization
    };
    // Value pass: safe denominator (comparison + where), normalized rounded
    // probabilities, F32 probability cast, ordinary product and running sum.
    let values = add(
        add(add(mul(7, rows)?, mul(2, scalar)?)?, mul(14, scores)?)?,
        add(
            matmul_cost(&[g.b, g.h, g.q, g.k], &[g.b, g.h, g.k, g.v], a)?,
            mul(3, output)?,
        )?,
    )?;
    Ok((common, normalization, values))
}

pub(in crate::backend::nn::workspace) mod descriptor;
mod stages;
pub(super) use stages::emit;
