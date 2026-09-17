//! Same two-pass recurrence and per-key-block completion as the native worker.
//! Counts sum attempted constructors. Physical payload remains the shared emitter's
//! maximum live block plus retained query outputs; no all-block lazy strategy.
use super::*;
use crate::backend::nn::attention::INPUT_SCORE_KEY_TILE;

fn append(plan: &mut Plan, p: usize, e: usize, seeds: usize) -> Option<()> {
    plan.append(Lowering::plain(p, e, seeds), 0)
}
fn common(g: Geometry<'_>) -> Option<Plan> {
    let mut plan = Plan::new();
    // K/V fixed block views and optional head replication (three views each).
    if g.fixed_views {
        plan.views(2)?;
    }
    if g.q[1] == g.k[1] {
        plan.lowering.backend_shells = 2; // actual keys.clone(), values.clone()
    } else {
        plan.views(6)?;
    }
    plan.views(2)?; // F32 K/V casts
    // QK product, swapped key, output/F32/scale/output rounding.
    plan.append(Lowering::product(8, 9, 0, 1), 0)?;
    plan.views(if g.arithmetic == AttentionArithmetic::InputScores {
        4
    } else {
        1
    })?; // swap + actual score-rounding casts
    append(&mut plan, 5, 6, 1)?; // scale multiply and actual scalar source
    if g.softcap {
        append(&mut plan, 12, 14, 2)?; // two multiplies and tanh's cast/unary
    }
    if g.bias {
        plan.views(1)?;
        append(&mut plan, 5, 6, 0)?;
    }
    if let Some(geometry) = g.absolute {
        // Two exact coordinate aranges/reshapes and the causal comparison.
        append(&mut plan, 2 + 2 + 5, 2 + 6, 0)?;
        if geometry.window().is_some() {
            // Earliest-key subtraction, visibility compare and final AND.
            append(&mut plan, 3 * 5, 3 * 6, 1)?;
            if geometry.prefix() > 0 {
                append(&mut plan, 2 * 5, 2 * 6, 1)?;
            }
        }
    } else {
        // Contiguous worker's existing all-true source.
        append(&mut plan, 3, 3, 1)?;
    }
    if let Some(dtype) = g.mask {
        plan.views(1)?; // fixed last-axis mask slice
        if dtype == WorkspaceDtype::Bool {
            append(&mut plan, 5, 6, 0)?; // allowed AND caller mask
        } else {
            append(&mut plan, 5, 6, 1)?; // isneginf equality/source
            append(&mut plan, 2, 2, 0)?; // logical_not cast/unary
            append(&mut plan, 5, 6, 0)?; // allowed AND finite
            plan.views(1)?; // caller mask cast
            append(&mut plan, 5, 6, 0)?; // additive bias
        }
    }
    append(&mut plan, 7, 9, 1)?; // where(mask, scores, F32::MIN)
    plan.views(1)?; // masked scores to F32
    plan.lowering.intermediate_rank = 5;
    Some(plan)
}
fn normalize(g: Geometry<'_>, first: bool) -> Option<Plan> {
    let mut plan = common(g)?;
    plan.append(reduction_lowering(2, 2, 0, 2), 0)?; // row max
    append(&mut plan, 5, 6, 0)?; // score minus max
    append(&mut plan, 2, 2, 0)?; // exp
    plan.views(1)?; // mask cast
    append(&mut plan, 5, 6, 0)?; // apply mask
    plan.append(reduction_lowering(2, 2, 0, 2), 0)?; // row sum
    if g.arithmetic == AttentionArithmetic::InputScores {
        append(&mut plan, 3, 3, 1)?; // zero normalization-pass value accumulator
    } else {
        plan.append(Lowering::product(8, 9, 0, 1), 0)?; // fused weighted value product
    }
    if !first {
        append(&mut plan, 5, 6, 0)?; // new maximum
        append(&mut plan, 14, 16, 0)?; // two subtract/exp scales
        append(&mut plan, 30, 36, 0)?; // sum and accumulator: two mul + add each
    } else if g.sink.is_some() {
        plan.views(3)?; // sink cast, reshape, broadcast
        append(&mut plan, 5, 6, 0)?; // maximum
        append(&mut plan, 14, 16, 0)?; // sink and block subtract/exp
        append(&mut plan, 15, 18, 0)?; // sum mul/add, accumulator mul
    }
    plan.lowering.nested_completions = 1;
    plan.controls = block_controls()?;
    Some(plan)
}
fn values(g: Geometry<'_>, first: bool) -> Option<Plan> {
    let mut plan = common(g)?;
    append(&mut plan, 12, 15, 2)?; // denominator comparison + where(1)
    append(&mut plan, 5, 6, 0)?; // subtract global maximum
    append(&mut plan, 2, 2, 0)?; // exp
    plan.views(1)?; // effective-mask cast
    append(&mut plan, 10, 12, 0)?; // multiply mask and divide denominator
    plan.views(2)?; // probability output rounding, F32 product input
    plan.append(Lowering::product(8, 9, 0, 1), 0)?;
    if !first {
        append(&mut plan, 5, 6, 0)?;
    }
    plan.lowering.nested_completions = 1;
    plan.controls = block_controls()?;
    Some(plan)
}
pub(super) fn plan(g: Geometry<'_>) -> Option<Plan> {
    if g.arithmetic != AttentionArithmetic::InputScores || g.k[2] <= INPUT_SCORE_ROW_BUDGET {
        return None;
    }
    let blocks = usize::try_from(g.k[2])
        .ok()?
        .div_ceil(INPUT_SCORE_KEY_TILE as usize);
    let queries = usize::try_from(g.q[2]).ok()?;
    let mut query = Plan::new();
    // Query slice, optional sliced mask, accumulator's F32 query and mask view.
    query.views(2 + 2 * usize::from(g.mask.is_some()))?;
    query.lowering.backend_shells = usize::from(g.sink.is_some());
    for pass in [false, true] {
        let first = if pass {
            values(g, true)?
        } else {
            normalize(g, true)?
        };
        query.append(first.lowering, first.controls)?;
        let rest = if pass {
            values(g, false)?
        } else {
            normalize(g, false)?
        }
        .repeat(blocks.checked_sub(1)?)?;
        query.append(rest.lowering, rest.controls)?;
    }
    query.views(1)?; // final output-dtype cast after the value pass
    let all = query.repeat(queries)?;
    let mut plan = Plan::new();
    plan.views(usize::from(g.mask.is_some()))?; // outer mask broadcast
    plan.append(all.lowering, all.controls)?;
    plan.bank_and_concat(queries)?;
    plan.controls = plan.controls.checked_add(frame_controls())?;
    Some(plan)
}
fn block_controls() -> Option<usize> {
    use crate::backend::runtime::cache::kv::{
        BlockwiseAttentionAccumulator, KeyValueAttentionBlock,
    };
    use std::mem::size_of;
    [
        size_of::<BlockwiseAttentionAccumulator>(),
        size_of::<KeyValueAttentionBlock>(),
        // Five common K/V/score/mask handles, three normalization handles,
        // eight merge/sink handles, four value-pass handles and eight native
        // argument temporaries from the two-pass shared block worker. These
        // alternative branches are summed, without relying on stack reuse.
        size_of::<[Array; 5 + 3 + 8 + 4 + 8]>(),
        size_of::<[Option<Array>; 4]>(),
        size_of::<[i32; 20]>(),
        size_of::<[i64; 6]>(),
        size_of::<[f32; 3]>(),
        size_of::<[&Array; 3]>(),
        size_of::<Result<(), safemlx::error::Exception>>(),
        // Completed-recurrence wrapper surrounds the unchanged exact native
        // settlement worker; both argument/result frames coexist.
        size_of::<(&mut BlockwiseAttentionAccumulator, &KeyValueAttentionBlock, Option<&Array>, &safemlx::Stream)>(),
        size_of::<Result<(), safemlx::error::Exception>>(),
        size_of::<bool>(),
        size_of::<Option<safemlx::OriginalScopeObserver>>(),
        safemlx::OperationEvent::nested_completion_control_bytes::<3>()?,
    ]
    .into_iter()
    .try_fold(0usize, usize::checked_add)
}

mod stages;
pub(super) use stages::stage_plan;
