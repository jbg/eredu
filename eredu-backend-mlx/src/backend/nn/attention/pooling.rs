//! Shape-only normalization shared by pooled execution and its cold facts.

pub(in crate::backend::nn) fn pooled_mask_shapes(
    query: &[i32],
    local: i32,
    pooled: i32,
    local_mask: Option<&[i32]>,
    pooled_mask: Option<&[i32]>,
) -> Result<Option<([i32; 4], [i32; 4])>, eredu_nn::Error> {
    pooled_mask_shapes_fixed(query, local, pooled, local_mask, pooled_mask)
        .map_err(eredu_nn::Error::backend)
}

/// Same checked normalization with its closed static cause and no allocation.
pub(in crate::backend::nn) fn pooled_mask_shapes_fixed(
    query: &[i32],
    local: i32,
    pooled: i32,
    local_mask: Option<&[i32]>,
    pooled_mask: Option<&[i32]>,
) -> Result<Option<([i32; 4], [i32; 4])>, &'static str> {
    let invalid = || "invalid pooled attention mask geometry";
    if query.len() != 4 || local < 0 || pooled < 0 {
        return Err(invalid());
    }
    if local_mask.is_none() && pooled_mask.is_none() {
        return Ok(None);
    }
    let mut leading = [1; 3];
    for (mask, width) in [(local_mask, local), (pooled_mask, pooled)] {
        let Some(mask) = mask else { continue };
        if mask.len() > 4 {
            return Err(invalid());
        }
        let mut shape = [1; 4];
        shape[4 - mask.len()..].copy_from_slice(mask);
        for axis in 0..4 {
            let target = if axis == 3 { width } else { query[axis] };
            if shape[axis] != 1 && shape[axis] != target {
                return Err(invalid());
            }
            if axis < 3 && shape[axis] != 1 {
                leading[axis] = shape[axis];
            }
        }
    }
    let [b, h, q] = leading;
    Ok(Some(([b, h, q, local], [b, h, q, pooled])))
}

/// Fixed outer worker frames around the existing native SDPA construction.
pub(in crate::backend::nn) fn pooled_attention_control_bytes(masked: bool) -> Option<usize> {
    use std::mem::{size_of, size_of_val};
    use crate::MlxTensor;
    use safemlx::{Array, Stream, error::Exception};
    let base = [
        size_of::<eredu_nn::PooledAttentionInput<'_, MlxTensor>>(),
        size_of::<&Stream>(),
        size_of::<[&[i32]; 3]>(), // borrowed query/local/pooled shapes
        size_of::<[i32; 2]>(), // local and pooled token counts
        size_of::<Option<([i32; 4], [i32; 4])>>(),
        size_of::<Result<Option<([i32; 4], [i32; 4])>, eredu_nn::Error>>(),
        size_of::<[Array; 2]>(), // expanded banks transferred into concat
        size_of::<Array>(), // joined K=V
        size_of::<Option<Array>>(), // normalized mask
        size_of::<Result<Option<Array>, Exception>>(),
        size_of::<Result<Array, Exception>>(),
        size_of::<Result<MlxTensor, eredu_nn::Error>>(),
    ];
    let bytes = base.into_iter().try_fold(size_of_val(&base), usize::checked_add)?;
    if !masked { return Some(bytes); }
    let normalization = [
        size_of::<bool>(),
        size_of::<[Option<&MlxTensor>; 2]>(),
        size_of::<(&Stream, &bool)>(), // normalize closure captures
        size_of::<(&eredu_nn::PooledAttentionInput<'_, MlxTensor>, &Array, &Stream)>(), // map closure
        size_of::<([i32; 4], [i32; 4])>(), // map argument
        size_of::<(Option<&MlxTensor>, &[i32])>(), // one sequential call
        size_of::<Array>(), // value before broadcast
        size_of::<[Array; 2]>(), // live concat source destination
        size_of::<Array>(), // joined mask before dtype promotion
        size_of::<Result<Array, Exception>>(),
        size_of::<Option<safemlx::fast::ScaledDotProductAttentionMask<'_>>>(),
    ];
    normalization.into_iter().try_fold(bytes.checked_add(size_of_val(&normalization))?, usize::checked_add)
}
