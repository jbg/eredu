//! Actual rotary application plus still-lazy constructor ancestors.
use super::*;

// FrequencyScaledRope::new: one Arange, fifteen binary workers (two casts,
// two broadcasts, one result each), one reciprocal (cast + result), and two
// where workers (three casts, three broadcasts, result). Twelve eager scalar
// descriptors are distinct. InputProducts adds the retained reciprocal.
fn frequency_ancestors(
    algorithm: eredu_nn::RotaryAlgorithm,
    inverse: bool,
) -> (usize, usize, usize, usize) {
    use eredu_nn::RotaryAlgorithm as A;
    match algorithm {
        A::Llama3 { .. } => (
            1 + 15 * 5 + 2 + 2 * 7 + 2 * usize::from(inverse),
            15 * 6 + 2 + 2 * 9 + 2 * usize::from(inverse),
            12,
            0,
        ),
        // The same constructor uploads one denominator leaf, then keeps its
        // reciprocal lazy. Include that first-use eager descriptor as well as
        // the external-leaf traversal alternative.
        A::Proportional { .. } if inverse => (2, 2, 1, 1),
        _ => (0, 0, 0, 1),
    }
}

pub(super) fn lowering(operation: WorkspaceOperationView<'_>) -> Option<Lowering> {
    use WorkspaceOperationKindView as K;
    use eredu_nn::{RotaryAlgorithm as A, RotaryArithmetic};
    let shape = operation.inputs.get(0)?.shape();
    let mut hidden_leaves = 0;
    let mut prefix = (0usize, 0usize, 0usize);
    let mut flattened = false;
    match operation.kind {
        K::TensorRotary(..) | K::RotaryFrequencies(..) => {}
        K::Rotary(_, None) => {
            if operation.inputs.len() != 3 || operation.outputs.len() != 1
                || shape.len() != 4
            {
                return None;
            }
            crate::backend::nn::attention::apply_rotary_embeddings_control_bytes()?;
            // Shared apply_rotary_embeddings: two possible batch expansions,
            // two dtype casts, four static index calls (Slice + Reshape),
            // negative-half Multiply and two further products plus Add
            // (four binary workers), then two-input concatenate (3/4).
            let mut value = Lowering::plain(2 + 2 + 4 * 2 + 4 * 5 + 3,
                2 + 2 + 4 * 2 + 4 * 6 + 4, 1);
            value.intermediate_rank = 4;
            return Some(value);
        }
        K::Rotary(spec, Some(_)) => {
            if spec.arithmetic == RotaryArithmetic::InputProducts {
                let (p, e, seeds, leaves) = frequency_ancestors(spec.algorithm, true);
                let mut value = Lowering::plain(74 + p, 84 + e, 1 + seeds);
                value.hidden_leaves = leaves;
                value.intermediate_rank = 4;
                return Some(value);
            }
            match spec.algorithm {
                A::Default | A::Linear { .. } => {}
                A::Proportional { .. } => {
                    flattened = true;
                    hidden_leaves = 1;
                    prefix = (2, 2, 0);
                }
                A::Yarn { .. } => {
                    flattened = true;
                    hidden_leaves = 1;
                    prefix = (7, 8, 1);
                }
                A::Llama3 { .. } => {
                    let (p, e, seeds, leaves) = frequency_ancestors(spec.algorithm, false);
                    if !spec.traditional {
                        // Same explicit position/trigonometric/product family
                        // as InputProducts. Its additional casts and pair-view
                        // alternatives conservatively cover native wavelength
                        // products without replacing their actual equation.
                        let mut value = Lowering::plain(74 + p, 84 + e, 1 + seeds);
                        value.hidden_leaves = leaves;
                        value.intermediate_rank = 4;
                        return Some(value);
                    }
                    flattened = true;
                    prefix = (p + 2, e + 2, seeds);
                    hidden_leaves = leaves;
                }
            }
        }
        // Caller-provided cosine/sine and multi-axis variants have independent
        // shared lowering recipes; neither is inferred from fused RoPE.
        _ => return None,
    }
    let batches = if flattened {
        shape
            .get(..shape.len().checked_sub(2)?)?
            .iter()
            .try_fold(1usize, |n, x| n.checked_mul(usize::try_from(*x).ok()?))?
    } else if shape.len() > 2 {
        usize::try_from(shape[0]).ok()?
    } else {
        1
    };
    // safemlx::fast::rope iterates each batch, creates a Slice when B>1,
    // then concatenates B results. Every fused call has an eager offset and
    // up to two casts. Concat casts each input and has B input edges.
    let sliced = usize::from(batches != 1);
    let p = batches
        .checked_mul(3usize.checked_add(2 * sliced)?)?
        .checked_add(if batches == 1 {
            0
        } else {
            batches.checked_add(1)?
        })?
        .checked_add(prefix.0)?;
    let e = batches
        .checked_mul(5usize.checked_add(2 * sliced)?)?
        .checked_add(if batches == 1 {
            0
        } else {
            batches.checked_mul(2)?
        })?
        .checked_add(prefix.1)?;
    let seeds = batches.checked_add(prefix.2)?;
    p.checked_add(seeds)?;
    let mut value = Lowering::plain(p, e, seeds);
    value.hidden_leaves = hidden_leaves;
    // ArrayVector geometric growth is bounded by twice actual input arity.
    value.maximum_operands = batches.max(4);
    value.intermediate_rank = 3;
    Some(value)
}

pub(super) fn control_bytes(operation: WorkspaceOperationView<'_>) -> Option<usize> {
    use eredu_nn::{RotaryAlgorithm as A, RotaryArithmetic};
    if matches!(operation.kind, WorkspaceOperationKindView::Rotary(_, None)) {
        return crate::backend::nn::attention::apply_rotary_embeddings_control_bytes();
    }
    let WorkspaceOperationKindView::Rotary(spec, Some(_)) = operation.kind else {
        return None;
    };
    let shape = operation.inputs.get(0)?.shape();
    let explicit = spec.arithmetic == RotaryArithmetic::InputProducts
        || (matches!(spec.algorithm, A::Llama3 { .. }) && !spec.traditional);
    let flattened = !matches!(spec.algorithm, A::Default | A::Linear { .. });
    if !explicit && !flattened && shape.len() > 4 {
        return None;
    }
    let batches = if explicit {
        1
    } else if flattened {
        shape
            .get(..shape.len().checked_sub(2)?)?
            .iter()
            .try_fold(1usize, |n, v| n.checked_mul(usize::try_from(*v).ok()?))?
    } else if shape.len() > 2 {
        usize::try_from(shape[0]).ok()?
    } else {
        1
    };
    crate::backend::nn::rope::rotary_control_bytes(explicit, batches)
}
