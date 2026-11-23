//! Pooling-window accumulation for compressed attention streams.

use eredu_core::cache::mechanisms::{LiveCacheBackend, PoolingCacheStorage};
use eredu_nn::PoolingWindows;

type Storage<M> = <M as LiveCacheBackend>::Tensor;
type Queue<M> = <M as LiveCacheBackend>::Queue;

use crate::state::pooling::{invalid, PoolingStateError};

/// Append-only compressed-token cache with an incomplete pooling window.
///
/// Values and gate logits are accumulated until `ratio` source tokens are
/// available. Complete windows are returned to the architecture's pooling
/// operator, while pooled outputs grow independently at the compressed token
/// rate. This representation is shared by compressed-attention and indexer
/// streams and preserves partial decode state across calls.
#[derive(Debug, Clone)]
pub struct PoolingCache<M: PoolingCacheStorage> {
    ratio: i32,
    pending_values: Option<Storage<M>>,
    pending_gates: Option<Storage<M>>,
    pooled: Option<Storage<M>>,
    overlap_values: Option<Storage<M>>,
    overlap_gates: Option<Storage<M>>,
    processed_tokens: i32,
}

/// Exact materialized state of an append-only pooling stream.
#[derive(Debug, Clone, Default)]
pub struct PoolingCacheState<T> {
    /// Source values in the incomplete window.
    pub pending_values: Option<T>,
    /// Source gates in the incomplete window.
    pub pending_gates: Option<T>,
    /// Committed pooled history.
    pub pooled: Option<T>,
    /// Values shared with the next overlapping window.
    pub overlap_values: Option<T>,
    /// Gates shared with the next overlapping window.
    pub overlap_gates: Option<T>,
}

impl<M: PoolingCacheStorage> PoolingCache<M> {
    /// Number of source tokens represented by one pooled token.
    pub const fn ratio(&self) -> i32 {
        self.ratio
    }

    /// Creates an empty pooling stream.
    pub fn new(ratio: i32) -> Result<Self, PoolingStateError> {
        if ratio <= 0 {
            return Err(invalid("pooling ratio must be positive"));
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

    /// Copies every logical element of partial, pooled and overlap views into
    /// independent compact storage. Transaction graph clones are insufficient
    /// for durable snapshots of strided views.
    pub fn isolated_snapshot(&self, stream: &Queue<M>) -> Result<Self, PoolingStateError> {
        let copy = |array: &Option<Storage<M>>| {
            array
                .as_ref()
                .map(|array| M::copy_tensor(array, stream))
                .transpose()
        };
        Ok(Self {
            ratio: self.ratio,
            pending_values: copy(&self.pending_values)?,
            pending_gates: copy(&self.pending_gates)?,
            pooled: copy(&self.pooled)?,
            overlap_values: copy(&self.overlap_values)?,
            overlap_gates: copy(&self.overlap_gates)?,
            processed_tokens: self.processed_tokens,
        })
    }

    /// Source tokens represented by complete and incomplete windows.
    pub const fn processed_tokens(&self) -> i32 {
        self.processed_tokens
    }

    /// Number of compressed tokens currently retained.
    pub fn pooled_tokens(&self) -> i32 {
        self.pooled
            .as_ref()
            .map_or(0, |pooled| M::shape(pooled).get(1).copied().unwrap_or(-1))
    }

    /// Returns every array whose lifetime is part of this cache state.
    pub fn arrays(&self) -> impl Iterator<Item = &Storage<M>> {
        self.pending_values
            .iter()
            .chain(self.pending_gates.iter())
            .chain(self.pooled.iter())
            .chain(self.overlap_values.iter())
            .chain(self.overlap_gates.iter())
    }

    /// Borrows the exact array components used by prompt-cache persistence.
    pub fn state_arrays(&self) -> [Option<&Storage<M>>; 5] {
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
        state: PoolingCacheState<Storage<M>>,
        processed_tokens: i32,
    ) -> Result<(), PoolingStateError> {
        if processed_tokens < 0 {
            return Err(invalid(
                "restored pooling source offset must be non-negative",
            ));
        }
        if state.pending_values.is_some() != state.pending_gates.is_some()
            || state.overlap_values.is_some() != state.overlap_gates.is_some()
        {
            return Err(invalid(
                "restored pooling value/gate components are incomplete",
            ));
        }
        let pending = processed_tokens % self.ratio;
        if state
            .pending_values
            .as_ref()
            .map_or(0, |array| M::shape(array).get(1).copied().unwrap_or(-1))
            != pending
            || state
                .pooled
                .as_ref()
                .map_or(0, |array| M::shape(array).get(1).copied().unwrap_or(-1))
                != processed_tokens / self.ratio
        {
            return Err(invalid(
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
        values: Storage<M>,
        gates: Storage<M>,
    ) -> (Option<Storage<M>>, Option<Storage<M>>) {
        let previous_values = self.overlap_values.replace(values);
        let previous_gates = self.overlap_gates.replace(gates);
        (previous_values, previous_gates)
    }

    /// Adds source values and returns only complete pooling windows.
    pub fn accumulate_windows(
        &mut self,
        values: Storage<M>,
        gates: Storage<M>,
        absolute_offset: i32,
        stream: &Queue<M>,
    ) -> Result<PoolingWindows<Storage<M>>, PoolingStateError> {
        let values_shape = M::shape(&values);
        let gates_shape = M::shape(&gates);
        if values_shape.len() != 3
            || gates_shape.len() != 3
            || values_shape[..2] != gates_shape[..2]
        {
            return Err(invalid(
                "pooling values and gates must be rank-3 with matching batch/token dimensions",
            ));
        }
        if absolute_offset != self.processed_tokens {
            return Err(invalid(format!(
                "pooling cache expected source offset {}, got {absolute_offset}",
                self.processed_tokens
            )));
        }
        let previous_pending = self
            .pending_values
            .as_ref()
            .map_or(0, |value| M::shape(value).get(1).copied().unwrap_or(-1));
        let values = match self.pending_values.as_ref() {
            Some(previous) => M::concatenate_tokens(&[previous, &values], 1, stream)?,
            None => values,
        };
        let gates = match self.pending_gates.as_ref() {
            Some(previous) => M::concatenate_tokens(&[previous, &gates], 1, stream)?,
            None => gates,
        };
        let total = M::shape(&values)[1];
        let usable = total / self.ratio * self.ratio;
        let processed_tokens = self
            .processed_tokens
            .checked_add(total - previous_pending)
            .ok_or_else(|| invalid("pooling source offset overflowed"))?;
        let ready_values = M::slice_tokens(&values, 1, 0..usable, stream)?;
        let ready_gates = M::slice_tokens(&gates, 1, 0..usable, stream)?;
        let (pending_values, pending_gates) = if usable < total {
            (
                Some(M::slice_tokens(&values, 1, usable..total, stream)?),
                Some(M::slice_tokens(&gates, 1, usable..total, stream)?),
            )
        } else {
            (None, None)
        };
        self.pending_values = pending_values;
        self.pending_gates = pending_gates;
        self.processed_tokens = processed_tokens;
        Ok(PoolingWindows {
            values: ready_values,
            gates: ready_gates,
            base_position: absolute_offset - previous_pending,
        })
    }

    /// Appends newly compressed values and returns the complete compressed history.
    pub fn update_and_fetch(
        &mut self,
        new_pooled: Storage<M>,
        stream: &Queue<M>,
    ) -> Result<Storage<M>, PoolingStateError> {
        let shape = M::shape(&new_pooled);
        if shape.len() != 3 {
            return Err(invalid(
                "pooled cache values must have shape [batch, tokens, width]",
            ));
        }
        if shape[1] == 0 {
            return match &self.pooled {
                Some(pooled) => Ok(pooled.clone()),
                None => Ok(M::zeroed_cache_tensor(shape, &new_pooled, stream)?),
            };
        }
        let pooled = match &self.pooled {
            Some(previous) => M::concatenate_tokens(&[previous, &new_pooled], 1, stream)?,
            None => new_pooled,
        };
        self.pooled = Some(pooled.clone());
        Ok(pooled)
    }

    /// Builds the causal validity mask for compressed positions.
    ///
    /// Query `offset + j` may consume pooled token `i` only after its complete
    /// source window exists: `i < (offset + j + 1) / ratio`.
    pub fn make_mask(
        &self,
        query_tokens: i32,
        offset: i32,
        stream: &Queue<M>,
    ) -> Result<Option<Storage<M>>, PoolingStateError> {
        let pooled_tokens = self.pooled_tokens();
        if pooled_tokens == 0 {
            return Ok(None);
        }
        let geometry = eredu_nn::attention_mask::PooledWindowVisibility::new(
            query_tokens,
            offset,
            pooled_tokens,
            self.ratio,
        )?;
        if query_tokens == 1 && geometry.all_keys_visible() {
            return Ok(None);
        }
        Ok(Some(M::pooled_visibility(geometry, stream)?))
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
