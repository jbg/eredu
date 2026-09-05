//! Pooling-window accumulation for compressed attention streams.

use super::*;

/// Append-only compressed-token cache with an incomplete pooling window.
///
/// Values and gate logits are accumulated until `ratio` source tokens are
/// available. Complete windows are returned to the architecture's pooling
/// operator, while pooled outputs grow independently at the compressed token
/// rate. This representation is shared by compressed-attention and indexer
/// streams and preserves partial decode state across calls.
#[derive(Debug, Clone)]
pub struct PoolingCache {
    ratio: i32,
    pending_values: Option<Array>,
    pending_gates: Option<Array>,
    pooled: Option<Array>,
    overlap_values: Option<Array>,
    overlap_gates: Option<Array>,
    processed_tokens: i32,
}

/// Exact materialized state of an append-only pooling stream.
#[derive(Debug, Clone, Default)]
pub struct PoolingCacheState {
    pub pending_values: Option<Array>,
    pub pending_gates: Option<Array>,
    pub pooled: Option<Array>,
    pub overlap_values: Option<Array>,
    pub overlap_gates: Option<Array>,
}

/// Complete source windows emitted by [`PoolingCache::accumulate_windows`].
pub struct PoolingWindows {
    /// Source values shaped `[batch, complete_source_tokens, width]`.
    pub values: Array,
    /// Gate logits with the same source-token extent.
    pub gates: Array,
    /// Absolute source-token position of the first returned value.
    pub base_position: i32,
}

impl PoolingCache {
    /// Number of source tokens represented by one pooled token.
    pub const fn ratio(&self) -> i32 {
        self.ratio
    }

    /// Creates an empty pooling stream.
    pub fn new(ratio: i32) -> Result<Self, Exception> {
        if ratio <= 0 {
            return Err(Exception::custom("pooling ratio must be positive"));
        }
        Ok(Self {
            ratio,
            pending_values: None,
            pending_gates: None,
            pooled: None,
            overlap_values: None,
            overlap_gates: None,
            processed_tokens: 0,
        })
    }

    pub fn deep_clone_state(&self) -> Result<Self, Exception> {
        let clone = |array: &Option<Array>| {
            array
                .as_ref()
                .map(|array| array.clone().deep_clone())
                .transpose()
        };
        Ok(Self {
            ratio: self.ratio,
            pending_values: clone(&self.pending_values)?,
            pending_gates: clone(&self.pending_gates)?,
            pooled: clone(&self.pooled)?,
            overlap_values: clone(&self.overlap_values)?,
            overlap_gates: clone(&self.overlap_gates)?,
            processed_tokens: self.processed_tokens,
        })
    }

    /// Source tokens represented by complete and incomplete windows.
    pub const fn processed_tokens(&self) -> i32 {
        self.processed_tokens
    }

    /// Number of compressed tokens currently retained.
    pub fn pooled_tokens(&self) -> i32 {
        self.pooled.as_ref().map_or(0, |pooled| pooled.dim(1))
    }

    /// Returns every array whose lifetime is part of this cache state.
    pub fn arrays(&self) -> impl Iterator<Item = &Array> {
        self.pending_values
            .iter()
            .chain(self.pending_gates.iter())
            .chain(self.pooled.iter())
            .chain(self.overlap_values.iter())
            .chain(self.overlap_gates.iter())
    }

    /// Borrows the exact array components used by prompt-cache persistence.
    pub fn state_arrays(&self) -> [Option<&Array>; 5] {
        [
            self.pending_values.as_ref(),
            self.pending_gates.as_ref(),
            self.pooled.as_ref(),
            self.overlap_values.as_ref(),
            self.overlap_gates.as_ref(),
        ]
    }

    /// Restores an exactly validated pooling frontier.
    pub fn restore_state(
        &mut self,
        state: PoolingCacheState,
        processed_tokens: i32,
    ) -> Result<(), Exception> {
        if processed_tokens < 0 {
            return Err(Exception::custom(
                "restored pooling source offset must be non-negative",
            ));
        }
        if state.pending_values.is_some() != state.pending_gates.is_some()
            || state.overlap_values.is_some() != state.overlap_gates.is_some()
        {
            return Err(Exception::custom(
                "restored pooling value/gate components are incomplete",
            ));
        }
        let pending = processed_tokens % self.ratio;
        if state
            .pending_values
            .as_ref()
            .map_or(0, |array| array.dim(1))
            != pending
            || state.pooled.as_ref().map_or(0, |array| array.dim(1))
                != processed_tokens / self.ratio
        {
            return Err(Exception::custom(
                "restored pooling state does not match its source-token frontier",
            ));
        }
        self.pending_values = state.pending_values;
        self.pending_gates = state.pending_gates;
        self.pooled = state.pooled;
        self.overlap_values = state.overlap_values;
        self.overlap_gates = state.overlap_gates;
        self.processed_tokens = processed_tokens;
        Ok(())
    }

    /// Replaces the overlap carried between adjacent complete windows and
    /// returns the previous pair. This is used by stride-one compressed
    /// streams whose logical window spans the previous and current groups.
    pub fn replace_overlap(
        &mut self,
        values: Array,
        gates: Array,
    ) -> (Option<Array>, Option<Array>) {
        let previous_values = self.overlap_values.replace(values);
        let previous_gates = self.overlap_gates.replace(gates);
        (previous_values, previous_gates)
    }

    /// Adds source values and returns only complete pooling windows.
    pub fn accumulate_windows(
        &mut self,
        values: Array,
        gates: Array,
        absolute_offset: i32,
        stream: &Stream,
    ) -> Result<PoolingWindows, Exception> {
        if values.ndim() != 3
            || gates.ndim() != 3
            || values.dim(0) != gates.dim(0)
            || values.dim(1) != gates.dim(1)
        {
            return Err(Exception::custom(
                "pooling values and gates must be rank-3 with matching batch/token dimensions",
            ));
        }
        if absolute_offset != self.processed_tokens {
            return Err(Exception::custom(format!(
                "pooling cache expected source offset {}, got {absolute_offset}",
                self.processed_tokens
            )));
        }
        let previous_pending = self.pending_values.as_ref().map_or(0, |value| value.dim(1));
        let values = match self.pending_values.take() {
            Some(previous) => concatenate_axis(&[previous, values], 1, stream)?,
            None => values,
        };
        let gates = match self.pending_gates.take() {
            Some(previous) => concatenate_axis(&[previous, gates], 1, stream)?,
            None => gates,
        };
        let total = values.dim(1);
        let usable = total / self.ratio * self.ratio;
        self.processed_tokens = self
            .processed_tokens
            .checked_add(total - previous_pending)
            .ok_or_else(|| Exception::custom("pooling source offset overflowed"))?;
        let ready_values = values.try_index_device((.., ..usable, ..), stream)?;
        let ready_gates = gates.try_index_device((.., ..usable, ..), stream)?;
        if usable < total {
            self.pending_values = Some(values.try_index_device((.., usable.., ..), stream)?);
            self.pending_gates = Some(gates.try_index_device((.., usable.., ..), stream)?);
        }
        Ok(PoolingWindows {
            values: ready_values,
            gates: ready_gates,
            base_position: absolute_offset - previous_pending,
        })
    }

    /// Appends newly compressed values and returns the complete compressed history.
    pub fn update_and_fetch(
        &mut self,
        new_pooled: Array,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        if new_pooled.ndim() != 3 {
            return Err(Exception::custom(
                "pooled cache values must have shape [batch, tokens, width]",
            ));
        }
        let empty_shape = [new_pooled.dim(0), 0, new_pooled.dim(-1)];
        let dtype = new_pooled.dtype();
        if new_pooled.dim(1) > 0 {
            self.pooled = Some(match self.pooled.take() {
                Some(previous) => concatenate_axis(&[previous, new_pooled], 1, stream)?,
                None => new_pooled,
            });
        }
        match &self.pooled {
            Some(pooled) => Ok(pooled.clone()),
            None => zeros_dtype(&empty_shape, dtype, stream),
        }
    }

    /// Builds the causal validity mask for compressed positions.
    ///
    /// Query `offset + j` may consume pooled token `i` only after its complete
    /// source window exists: `i < (offset + j + 1) / ratio`.
    pub fn make_mask(
        &self,
        query_tokens: i32,
        offset: i32,
        stream: &Stream,
    ) -> Result<Option<Array>, Exception> {
        let pooled_tokens = self.pooled_tokens();
        if pooled_tokens == 0 || query_tokens == 1 {
            return Ok(None);
        }
        let pooled = Array::arange::<i32, i32>(Some(0), pooled_tokens, None, stream)?;
        let visible =
            Array::arange::<i32, i32>(Some(offset + 1), offset + query_tokens + 1, None, stream)?
                .floor_divide(Array::from_int(self.ratio), stream)?
                .reshape(&[query_tokens, 1], stream)?;
        Ok(Some(
            pooled
                .reshape(&[1, pooled_tokens], stream)?
                .lt(visible, stream)?,
        ))
    }

    /// Clears all complete and partial compressed state.
    pub fn clear(&mut self) {
        self.pending_values = None;
        self.pending_gates = None;
        self.pooled = None;
        self.overlap_values = None;
        self.overlap_gates = None;
        self.processed_tokens = 0;
    }
}
