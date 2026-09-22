//! Two-pass input-score attention. Each accumulator step waits for native
//! ownership retirement, so its workspace is a maximum across key blocks.
//! Finished query outputs remain live until the final concatenation.

use super::*;
use crate::backend::nn::attention::INPUT_SCORE_KEY_TILE;

/// Complete phase alternatives before the enclosing operation selects a peak.
/// Each row records total bytes including retained output, default bytes and births.
pub(super) struct InputScoreBranches {
    rows: [(u64, u64, usize); 5],
    length: usize,
}
impl InputScoreBranches {
    pub(super) fn alternatives(&self) -> &[(u64, u64, usize)] {
        &self.rows[..self.length]
    }
}

pub(super) fn input_score_branches(
    geometry: Geometry,
    mask: Mask,
    sinks: bool,
    cap: bool,
    allocation: NativeAllocationFacts,
) -> FactResult<InputScoreBranches> {
    // This native path uses one query at a time when a complete key row is
    // larger than the input-score budget. The caller validates positive shapes.
    let query = Geometry { q: 1, ..geometry };
    let output = capacity(allocation, query.output()?)?;
    // One query cast, the previous normalization pair and value accumulator
    // coexist with the next step's newly allocated buffers. Previously finished
    // queries may retain both their F32 accumulator and output dtype conversion.
    // Their completed step graphs no longer retain the eager scalar/page inputs.
    let retained = add(
        add(
            capacity(allocation, query.query()?)?,
            add(mul(2, capacity(allocation, query.rows()?)?)?, output)?,
        )?,
        mul(mul(2, geometry.q as u64 - 1)?, output)?,
    )?;
    let mut result = InputScoreBranches {
        rows: [(0, 0, 0); 5],
        length: 0,
    };
    let full = INPUT_SCORE_KEY_TILE.min(geometry.k);
    let tail = geometry.k % INPUT_SCORE_KEY_TILE;
    for keys in [Some(full), (tail != 0 && tail != full).then_some(tail)]
        .into_iter()
        .flatten()
    {
        let g = Geometry { k: keys, ..query };
        let (common, normalization, values) =
            block_costs(g, mask, sinks, cap, allocation, true, None, false)?;
        for (value_pass, phase) in [(false, normalization), (true, values)] {
            let (default_bytes, default_births) =
                block_defaults(g, mask, cap, allocation, true, None, value_pass)?;
            result.rows[result.length] = (
                add(retained, add(common, phase)?)?,
                default_bytes,
                default_births,
            );
            result.length += 1;
        }
    }
    // The final casts/concatenation run after all block scopes have settled.
    result.rows[result.length] = (
        add(
            mul(mul(2, geometry.q as u64)?, output)?,
            capacity(allocation, geometry.output()?)?,
        )?,
        0,
        0,
    );
    result.length += 1;
    Ok(result)
}

/// Eager default allocations in the same selected native recurrence step.
/// These are already included in `block_costs`; casts and graph outputs remain
/// in that step's execution allocation population.
fn block_defaults(
    g: Geometry,
    mask: Mask,
    cap: bool,
    a: NativeAllocationFacts,
    input_scores: bool,
    absolute: Option<eredu_nn::operation_geometry::AbsoluteAttentionMaskGeometry>,
    value_pass: bool,
) -> FactResult<(u64, usize)> {
    // Scale, finite-min where, optional cap/reciprocal and native isneginf seed.
    let mut scalars = 2 + 2 * usize::from(cap) + usize::from(matches!(mask, Mask::Additive(_)));
    scalars += if value_pass {
        2
    } else {
        usize::from(input_scores)
    };
    let (mask_bytes, mask_births) = match absolute {
        Some(geometry) => {
            // The aranges are stream-produced; only window/prefix operands are
            // eager scalar constructors in the actual integer-mask worker.
            let seeds = usize::from(geometry.window().is_some())
                + usize::from(geometry.window().is_some() && geometry.prefix() > 0);
            (mul(capacity(a, 1)?, seeds as u64)?, seeds)
        }
        None if a.original_storage => (capacity(a, 1)?, 1),
        None => (
            // The ordinary noncausal worker copies the actual Bool page; the
            // original worker above constructs full(true) on the active stream.
            facts::buffer_capacity(a, mul(g.q as u64, g.k as u64)?)?,
            1,
        ),
    };
    Ok((
        add(mul(capacity(a, 1)?, scalars as u64)?, mask_bytes)?,
        scalars
            .checked_add(mask_births)
            .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?,
    ))
}

fn block_costs(
    g: Geometry,
    mask: Mask,
    sinks: bool,
    cap: bool,
    a: NativeAllocationFacts,
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
