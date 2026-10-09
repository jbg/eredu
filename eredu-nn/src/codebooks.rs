//! Ordered feedback lookup in a stack of independently indexed codebooks.

use crate::{Error, Index, Tensor, TensorElementType};

/// Sums one vector from each `[depth, vocabulary, width]` F32 table in increasing
/// depth order, starting from zero. I32 codes have shape `[batch, time, depth]`.
/// Signed indices retain the ordinary gather wrap-from-end semantics. Only the
/// vocabulary-sized sentinel contributes zero; all other invalid codes
/// fail through the tensor's strict gather contract. Backends may fuse this
/// mechanism, preserving the ordered F32 additions and strict index domain.
pub fn ordered_codebook_lookup_sum<T: Tensor>(
    table: &T,
    codes: &T,
    context: &T::Context,
) -> Result<T, Error> {
    validate(table, codes)?;
    let mut sum = T::full_f32(0.0, &[codes.dim(0), codes.dim(1), table.dim(2)], context)?;
    let zero = T::full_f32(0.0, &[], context)?;
    let zero_index = T::full_i32(0, &[], context)?;
    for depth in 0..table.dim(0) {
        let means = table.index(&[Index::At(depth)], context)?;
        let ids = codes.index(&[Index::Full, Index::Full, Index::At(depth)], context)?;
        let pad = ids.equal_i32(table.dim(1), context)?;
        let safe_ids = T::where_condition(&pad, &zero_index, &ids, context)?;
        let selected = means.take_axis(&safe_ids, 0, context)?;
        let selected =
            T::where_condition(&pad.expand_dims(-1, context)?, &zero, &selected, context)?;
        sum = sum.add(&selected, context)?;
    }
    Ok(sum)
}

/// Checks tensor-free geometry and scalar representation before dispatch.
pub fn validate<T: Tensor>(table: &T, codes: &T) -> Result<(), Error> {
    if table.shape().len() != 3
        || table.shape().iter().any(|&n| n <= 0)
        || table.element_type() != Some(TensorElementType::F32)
        || codes.shape().len() != 3
        || codes.shape().iter().any(|&n| n <= 0)
        || codes.dim(2) != table.dim(0)
        || codes.element_type() != Some(TensorElementType::I32)
    {
        return Err(Error::backend(
            "ordered codebook lookup requires F32 [depth, vocabulary, width] and I32 [batch, time, depth]",
        ));
    }
    Ok(())
}
