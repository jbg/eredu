//! Checked tensor views on the sequence axis of batch-first values.

use std::ops::Range;

use crate::{Error, Index, Tensor};

/// Returns the sequence extent of a batch-first tensor.
pub fn length<T: Tensor>(value: &T) -> Result<usize, Error> {
    value
        .shape()
        .get(1)
        .and_then(|&extent| usize::try_from(extent).ok())
        .ok_or_else(|| Error::backend("expected a nonnegative batch-first sequence axis"))
}

/// Returns a view over an exact sequence interval.
pub fn span<T: Tensor>(value: &T, range: Range<usize>, context: &T::Context) -> Result<T, Error> {
    if range.start > range.end || range.end > length(value)? {
        return Err(Error::backend("sequence interval is out of bounds"));
    }
    value.index(
        &[
            Index::Full,
            Index::Range(range.start as i32, range.end as i32),
        ],
        context,
    )
}

/// Selects a single in-bounds sequence row with the requested rank.
pub fn row<T: Tensor>(
    value: &T,
    row: usize,
    keep_axis: bool,
    context: &T::Context,
) -> Result<T, Error> {
    if row >= length(value)? {
        return Err(Error::backend("sequence row is out of bounds"));
    }
    if keep_axis {
        span(value, row..row + 1, context)
    } else {
        value.index(&[Index::Full, Index::At(row as i32)], context)
    }
}
