//! Bounded accumulation of weighted rows from encoded embedding tables.

use crate::{Error, Index, Tensor, TensorElementType};

/// Uses the table's lookup arithmetic for coefficients, accumulating bounded
/// vocabulary blocks in FP32 before narrowing the completed embedding.
pub fn weighted_embedding_rows<T: Tensor>(
    coefficients: &T,
    vocabulary: i32,
    context: &T::Context,
    mut lookup: impl FnMut(&T) -> Result<T, Error>,
    mut settle: impl FnMut(&T) -> Result<(), Error>,
) -> Result<T, Error> {
    if vocabulary <= 0 || coefficients.shape().last() != Some(&vocabulary) {
        return Err(Error::backend(
            "weighted embedding requires a complete vocabulary axis",
        ));
    }
    let flat = coefficients.reshape(&[-1, vocabulary], context)?;
    let mut output: Option<T> = None;
    let mut element = None;
    for start in (0..vocabulary).step_by(512) {
        let end = start.saturating_add(512).min(vocabulary);
        let ids = T::from_i32_slice(&(start..end).collect::<Vec<_>>(), &[end - start], context)?;
        let rows = lookup(&ids)?;
        let dtype = rows
            .element_type()
            .ok_or_else(|| Error::backend("unknown embedding row dtype"))?;
        let weights = flat
            .index(&[Index::Full, Index::Range(start, end)], context)?
            .cast_float(dtype, context)?
            .cast_float(TensorElementType::F32, context)?;
        let contribution = T::matmul(
            &weights,
            &rows.cast_float(TensorElementType::F32, context)?,
            context,
        )?;
        let sum = match output {
            Some(previous) => previous.add(&contribution, context)?,
            None => contribution,
        };
        settle(&sum)?;
        output = Some(sum);
        element = Some(dtype);
    }
    let output = output.expect("positive vocabulary");
    let mut shape = coefficients.shape().to_vec();
    *shape.last_mut().expect("validated vocabulary axis") = output.dim(1);
    output
        .cast_float(element.expect("positive vocabulary"), context)?
        .reshape(&shape, context)
}
