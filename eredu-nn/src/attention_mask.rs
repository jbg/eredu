//! Portable attention-mask geometry and additive masks.

use std::ops::Range;

use crate::{operation_geometry::CausalMaskGeometry, Error, Tensor};

/// Materializes a causal mask with bidirectional spans inside the current query sequence.
///
/// Spans are ordered, disjoint, nonempty local ranges. The cache offset
/// shifts their key positions. The optional past limit still applies inside
/// each span and to the cached prefix.
pub fn causal_mask_with_bidirectional_spans<T: Tensor>(
    sequence: i32,
    offset: i32,
    max_past: Option<i32>,
    spans: &[Range<i32>],
    context: &T::Context,
) -> Result<T, Error> {
    let geometry = CausalMaskGeometry::new(sequence, offset, max_past)?;
    let mut previous_end = 0;
    for span in spans {
        if span.start < previous_end || span.end <= span.start || span.end > geometry.sequence() {
            return Err(Error::backend("invalid or overlapping attention spans"));
        }
        previous_end = span.end;
    }
    let keys = geometry.keys() as usize;
    let elements = (geometry.sequence() as usize)
        .checked_mul(keys)
        .ok_or_else(|| Error::backend("attention mask size overflows"))?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(elements)
        .map_err(Error::backend_source)?;
    values.resize(elements, f32::NEG_INFINITY);

    let mut spans = spans.iter().peekable();
    for (query, row) in values.chunks_exact_mut(keys.max(1)).enumerate() {
        let query = query as i32;
        while spans.peek().is_some_and(|span| span.end <= query) {
            spans.next();
        }
        let end = spans
            .peek()
            .filter(|span| span.start <= query)
            .map_or(query + 1, |span| span.end)
            + geometry.offset();
        let position = geometry.offset() + query;
        let start = max_past.map_or(0, |distance| (position - distance).max(0));
        row[start as usize..end as usize].fill(0.0);
    }
    T::from_f32_slice(&values, &[geometry.sequence(), geometry.keys()], context)
}

/// Checked causal visibility over the compact sliding-window key span.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LocalWindowVisibility {
    causal: CausalMaskGeometry,
    first_key: i32,
    max_past: i32,
}

impl LocalWindowVisibility {
    /// A window of one includes only the current token.
    pub fn new(query_tokens: i32, offset: i32, window: i32) -> Result<Self, Error> {
        if window <= 0 {
            return Err(Error::backend("local visibility window must be positive"));
        }
        let max_past = window - 1;
        let causal = CausalMaskGeometry::new(query_tokens, offset, Some(max_past))?;
        let first_key = (offset - max_past).max(0);
        check_visibility_extent(query_tokens, causal.keys() - first_key)?;
        Ok(Self {
            causal,
            first_key,
            max_past,
        })
    }

    /// Absolute query positions.
    pub fn queries(self) -> Range<i32> {
        self.causal.offset()..self.causal.keys()
    }

    /// Absolute key positions, cropped to the first query's window.
    pub fn keys(self) -> Range<i32> {
        self.first_key..self.causal.keys()
    }

    /// Inclusive backward distance from each query.
    pub const fn max_past(self) -> i32 {
        self.max_past
    }

    /// Row and column counts of the boolean mask.
    pub const fn shape(self) -> [i32; 2] {
        [self.causal.sequence(), self.causal.keys() - self.first_key]
    }

    /// Whether the query row observes the key column in the cropped span.
    pub fn allows(self, row: i32, column: i32) -> bool {
        (0..self.shape()[1]).contains(&column) && self.causal.allows(row, self.first_key + column)
    }
}

/// Checked causal visibility over complete pooled source windows.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PooledWindowVisibility {
    causal: CausalMaskGeometry,
    pooled_tokens: i32,
    ratio: i32,
}

impl PooledWindowVisibility {
    /// A key is visible once all of its source tokens precede or equal the query.
    pub fn new(
        query_tokens: i32,
        offset: i32,
        pooled_tokens: i32,
        ratio: i32,
    ) -> Result<Self, Error> {
        if ratio <= 0 {
            return Err(Error::backend("pooled visibility ratio must be positive"));
        }
        let causal = CausalMaskGeometry::new(query_tokens, offset, None)?;
        check_visibility_extent(query_tokens, pooled_tokens)?;
        Ok(Self {
            causal,
            pooled_tokens,
            ratio,
        })
    }

    /// Absolute query positions.
    pub fn queries(self) -> Range<i32> {
        self.causal.offset()..self.causal.keys()
    }

    /// Source tokens in one pooled window.
    pub const fn ratio(self) -> i32 {
        self.ratio
    }

    /// Row and column counts of the boolean mask.
    pub const fn shape(self) -> [i32; 2] {
        [self.causal.sequence(), self.pooled_tokens]
    }

    /// Whether every pooled key is visible to the first query, and therefore to all queries.
    pub fn all_keys_visible(self) -> bool {
        self.pooled_tokens <= (self.causal.offset() + 1) / self.ratio
    }

    /// Whether the query row can observe the complete pooled window represented by the column.
    pub fn allows(self, row: i32, column: i32) -> bool {
        (0..self.causal.sequence()).contains(&row)
            && (0..self.pooled_tokens).contains(&column)
            && column < (self.causal.offset() + row + 1) / self.ratio
    }
}

fn check_visibility_extent(queries: i32, keys: i32) -> Result<(), Error> {
    if queries <= 0 || keys <= 0 {
        return Err(Error::backend(
            "visibility requires positive query and key counts",
        ));
    }
    if (queries as usize)
        .checked_mul(keys as usize)
        .is_none_or(|extent| extent > isize::MAX as usize)
    {
        return Err(Error::backend(
            "visibility extent exceeds addressable storage",
        ));
    }
    Ok(())
}
