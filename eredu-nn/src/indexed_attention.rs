//! Selected-position attention independent of source storage and model family.
use crate::{AttentionArithmetic, Error, Tensor};

/// An optional bounded, unindexed source sharing the selected source's softmax.
#[derive(Debug)]
pub struct LocalAttentionInput<'a, T> {
    /// Keys shaped `[batch, kv_heads, local_tokens, key_dimensions]`.
    pub keys: &'a T,
    /// Values shaped `[batch, kv_heads, local_tokens, value_dimensions]`.
    pub values: &'a T,
    /// Boolean eligibility or additive scores broadcastable to `[B, H, Q, local_tokens]`.
    pub mask: Option<&'a T>,
}

/// Attention to explicit original or compressed token positions, optionally
/// sharing softmax with a bounded local source and zero-valued learned sinks.
///
/// Source K/V have grouped-query geometry. The caller owns selection and causal
/// eligibility. Only selected source rows may be gathered; no whole-history
/// gather or query-by-history score/mask is part of this contract. A cache-aware
/// implementation reads its retained history instead of the latest append in
/// `keys` and `values`. Each selected slot participates once in the softmax.
#[derive(Debug)]
pub struct IndexedAttentionInput<'a, T> {
    /// Queries shaped `[batch, query_heads, query_tokens, key_dimensions]`.
    pub queries: &'a T,
    /// Resident source keys, or the latest append for a blockwise cache.
    /// Shape is `[batch, kv_heads, source_tokens, key_dimensions]`.
    pub keys: &'a T,
    /// Corresponding values shaped `[batch, kv_heads, source_tokens, value_dimensions]`.
    pub values: &'a T,
    /// Absolute position of the first resident source row. A cache-aware
    /// implementation instead resolves positions against its retained blocks.
    pub key_position_offset: i32,
    /// Integer absolute positions shaped `[batch, query_tokens, selected]`.
    /// `-1` is an invalid slot and never dereferenced. Repeated positions retain
    /// their requested order and multiplicity.
    pub selected_positions: &'a T,
    /// Optional boolean validity with exactly the selected-position shape.
    /// False slots are never dereferenced, regardless of their position value.
    pub validity: Option<&'a T>,
    /// Optional boolean eligibility or additive scores broadcastable to
    /// `[batch, query_heads, query_tokens, selected]`.
    pub mask: Option<&'a T>,
    /// Optional bounded unindexed source, using the same K/V head geometry.
    pub local: Option<LocalAttentionInput<'a, T>>,
    /// Positive finite score scale.
    pub scale: f32,
    /// Rounding boundaries shared with ordinary attention.
    pub arithmetic: AttentionArithmetic,
    /// Optional learned per-query-head sink logits with zero values.
    pub sinks: Option<&'a T>,
}

fn broadcast_mask<T: Tensor>(mask: Option<&T>, target: [i32; 4]) -> Result<(), Error> {
    if let Some(mask) = mask {
        let shape = mask.shape();
        if shape.len() > 4
            || shape
                .iter()
                .rev()
                .zip(target.iter().rev())
                .any(|(a, b)| *a != 1 && a != b)
        {
            return Err(Error::backend(format!(
                "indexed-attention mask {shape:?} cannot broadcast to {target:?}"
            )));
        }
    }
    Ok(())
}

impl<T: Tensor> IndexedAttentionInput<'_, T> {
    /// Checks ranks, grouped head geometry, masks and arithmetic policy without
    /// evaluating a tensor. Implementations additionally check position values.
    pub fn validate(&self) -> Result<(), Error> {
        let q = self.queries.shape();
        let k = self.keys.shape();
        let v = self.values.shape();
        let selected = self.selected_positions.shape();
        if q.len() != 4
            || k.len() != 4
            || v.len() != 4
            || selected.len() != 3
            || q.iter().any(|d| *d <= 0)
            || k.iter()
                .enumerate()
                .chain(v.iter().enumerate())
                .any(|(axis, d)| if axis == 2 { *d < 0 } else { *d <= 0 })
            || q[0] != k[0]
            || q[0] != v[0]
            || k[1] != v[1]
            || q[1] % k[1] != 0
            || k[2] != v[2]
            || q[3] != k[3]
            || selected[0] != q[0]
            || selected[1] != q[2]
            || selected[2] < 0
            || self.key_position_offset < 0
        {
            return Err(Error::backend("invalid grouped indexed-attention geometry"));
        }
        if !self.scale.is_finite() || self.scale <= 0.0 {
            return Err(Error::backend(
                "indexed-attention scale must be finite and positive",
            ));
        }
        if self.validity.is_some_and(|mask| mask.shape() != selected) {
            return Err(Error::backend(
                "indexed-attention validity must match selected positions",
            ));
        }
        broadcast_mask(self.mask, [q[0], q[1], q[2], selected[2]])?;
        if let Some(local) = &self.local {
            let lk = local.keys.shape();
            let lv = local.values.shape();
            if lk.len() != 4
                || lv.len() != 4
                || lk[..2] != k[..2]
                || lv[..2] != k[..2]
                || lk[2] < 0
                || lk[2] != lv[2]
                || lk[3] != k[3]
                || lv[3] != v[3]
            {
                return Err(Error::backend(
                    "indexed-attention local source geometry mismatch",
                ));
            }
            broadcast_mask(local.mask, [q[0], q[1], q[2], lk[2]])?;
        }
        if self.sinks.is_some_and(|sinks| sinks.shape() != [q[1]]) {
            return Err(Error::backend(
                "indexed-attention sinks must have one logit per query head",
            ));
        }
        Ok(())
    }
}

impl<T> Copy for LocalAttentionInput<'_, T> {}
impl<T> Clone for LocalAttentionInput<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for IndexedAttentionInput<'_, T> {}
impl<T> Clone for IndexedAttentionInput<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}
