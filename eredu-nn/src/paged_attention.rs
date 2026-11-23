//! Portable page visibility and online softmax for a bounded attention scan.

use std::ops::Range;

use crate::{AttentionArithmetic, Error, Tensor};

/// Input-rounded probabilities need the global normalization before their
/// per-page value products can be computed.
pub const fn requires_value_pass(arithmetic: AttentionArithmetic) -> bool {
    matches!(arithmetic, AttentionArithmetic::InputScores)
}

/// Intersects a physical page with the permanent prefix and retained suffix.
pub fn retained_page_ranges(
    page: Range<i64>,
    prefix_end: i64,
    suffix_start: i64,
) -> impl Iterator<Item = Range<i64>> {
    [
        page.start..page.end.min(prefix_end).min(suffix_start),
        page.start.max(suffix_start)..page.end,
    ]
    .into_iter()
    .filter(|range| !range.is_empty())
}

/// Converts absolute page positions to checked local mask columns.
pub fn mask_columns(page: Range<i64>, origin: i64, width: i32) -> Result<Range<i32>, Error> {
    let local = |position: i64| {
        position
            .checked_sub(origin)
            .and_then(|value| i32::try_from(value).ok())
            .filter(|value| (0..=width).contains(value))
            .ok_or_else(|| Error::backend("attention mask does not cover the selected page"))
    };
    let start = local(page.start)?;
    let end = local(page.end)?;
    if start >= end {
        return Err(Error::backend("invalid attention mask page range"));
    }
    Ok(start..end)
}

/// Builds one row-major visibility mask in absolute query and key coordinates.
pub fn page_visibility(
    query_start: i64,
    query_len: i32,
    page: Range<i64>,
    sliding_window: Option<i32>,
    prefix_tokens: i64,
) -> Result<Vec<bool>, Error> {
    let queries = usize::try_from(query_len).ok();
    let keys = page
        .end
        .checked_sub(page.start)
        .and_then(|n| usize::try_from(n).ok());
    let elements = queries
        .zip(keys)
        .and_then(|(queries, keys)| queries.checked_mul(keys))
        .ok_or_else(|| Error::backend("attention page mask geometry overflow"))?;
    let query_end = query_start
        .checked_add(i64::from(query_len))
        .ok_or_else(|| Error::backend("attention query positions overflow"))?;
    if sliding_window.is_some_and(|window| window <= 0) {
        return Err(Error::backend("attention window must be positive"));
    }
    let mut output = Vec::new();
    output
        .try_reserve_exact(elements)
        .map_err(Error::backend_source)?;
    for query in query_start..query_end {
        for key in page.clone() {
            output.push(
                key <= query
                    && sliding_window.is_none_or(|window| {
                        key < prefix_tokens || key >= query.saturating_sub(i64::from(window - 1))
                    }),
            );
        }
    }
    Ok(output)
}

/// One online-softmax partial with a common exponent origin for both sums.
pub struct SoftmaxPartial<T> {
    /// Row-wise common exponent origin.
    pub maximum: T,
    /// Sum of the shifted exponential weights.
    pub mass: T,
    /// Sum of the shifted weighted values.
    pub weighted_values: T,
}

/// Merges two partials using native maximum and exponential operations.
/// Both partials use the arithmetic selected for the same attention request.
pub fn merge_softmax<T: Tensor>(
    retained: &SoftmaxPartial<T>,
    page: &SoftmaxPartial<T>,
    context: &T::Context,
    maximum: impl FnOnce(&T, &T, &T::Context) -> Result<T, Error>,
    exp: impl Fn(T, &T::Context) -> Result<T, Error>,
) -> Result<SoftmaxPartial<T>, Error> {
    let maximum = maximum(&retained.maximum, &page.maximum, context)?;
    let old_scale = exp(retained.maximum.subtract(&maximum, context)?, context)?;
    let page_scale = exp(page.maximum.subtract(&maximum, context)?, context)?;
    let combine = |old: &T, next: &T| {
        old.multiply(&old_scale, context)?
            .add(&next.multiply(&page_scale, context)?, context)
    };
    Ok(SoftmaxPartial {
        maximum,
        mass: combine(&retained.mass, &page.mass)?,
        weighted_values: combine(&retained.weighted_values, &page.weighted_values)?,
    })
}
