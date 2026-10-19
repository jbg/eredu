//! Visibility over the keys actually presented by a scheduled cache.
use super::*;

pub(super) fn attention_mask<B: GroupedNeuralBackend>(
    explicit: Option<&B::Tensor>,
    sequence: i32,
    keys: i32,
    window: Option<i32>,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<Option<B::Tensor>, Error> {
    let past = keys
        .checked_sub(sequence)
        .filter(|&past| past >= 0)
        .ok_or_else(|| Error::backend("Muse attention keys omit submitted query positions"))?;
    let explicit = explicit
        .map(|mask| {
            let width = *mask
                .shape()
                .last()
                .ok_or_else(|| Error::backend("Muse attention mask has no key axis"))?;
            if width == 1 || width == keys {
                return Ok(mask.clone());
            }
            if width < keys {
                return Err(Error::backend("Muse attention mask omits retained keys"));
            }
            let mut indexes = vec![eredu_nn::Index::Full; mask.shape().len()];
            *indexes.last_mut().unwrap() = eredu_nn::Index::Range(width - keys, width);
            mask.index(&indexes, context)
        })
        .transpose()?;
    if (sequence == 1 && window.is_none_or(|window| keys <= window))
        || (explicit.is_some() && window.is_none())
    {
        return Ok(explicit);
    }
    let causal = B::causal_mask(sequence, past, window.map(|window| window - 1), context)?;
    let Some(explicit) = explicit else {
        return Ok(Some(causal));
    };
    let additive = |mask: B::Tensor| match mask.element_type() {
        Some(eredu_nn::TensorElementType::Bool) => {
            let zero = B::Tensor::full_f32(0.0, &[], context)?;
            let excluded = B::Tensor::full_f32(f32::NEG_INFINITY, &[], context)?;
            B::Tensor::where_condition(&mask, &zero, &excluded, context)
        }
        Some(
            eredu_nn::TensorElementType::F16
            | eredu_nn::TensorElementType::Bf16
            | eredu_nn::TensorElementType::F32
            | eredu_nn::TensorElementType::F64,
        ) => Ok(mask),
        _ => Err(Error::backend(
            "Muse attention mask must declare boolean or floating storage",
        )),
    };
    Ok(Some(additive(explicit)?.add(&additive(causal)?, context)?))
}
