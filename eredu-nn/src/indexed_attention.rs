//! Allocation-free semantic validation shared by indexed-attention realizations.

/// Fixed reason that an indexed-attention input does not satisfy its contract.
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum IndexedAttentionValidationError {
    /// Ranks, corresponding dimensions, or the selected pooled extent differ.
    #[error("invalid indexed-attention geometry")]
    Geometry,
    /// The query scale is not finite and positive.
    #[error("indexed-attention scale must be finite and positive")]
    Scale,
    /// Learned sinks do not have one value per query head.
    #[error("indexed-attention sinks must have one value per head")]
    Sinks,
}

/// Validated logical dimensions of the existing indexed-attention mechanism.
/// This carries no allocation, device, source, completion, or submission grant.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IndexedAttentionGeometry {
    /// Batch width.
    pub batch: i32,
    /// Query heads.
    pub heads: i32,
    /// Query tokens.
    pub queries: i32,
    /// Key/query vector width.
    pub key_dimensions: i32,
    /// Value/output vector width.
    pub value_dimensions: i32,
    /// Local key/value tokens; this may be zero.
    pub local: i32,
    /// Available pooled key/value tokens.
    pub pooled: i32,
    /// Selected pooled positions for each query.
    pub selected: i32,
}
impl IndexedAttentionGeometry {
    /// Checks query, local K/V, pooled K/V and selected-position shapes in that
    /// order, followed by the query scale and optional learned sink shape.
    pub fn new(
        shapes: [&[i32]; 6],
        scale: f32,
        sinks: Option<&[i32]>,
    ) -> Result<Self, IndexedAttentionValidationError> {
        let [q, lk, lv, pk, pv, indices] = shapes;
        if q.len() != 4
            || lk.len() != 3
            || lv.len() != 3
            || pk.len() != 3
            || pv.len() != 3
            || indices.len() != 3
            || q[0] != lk[0]
            || q[0] != lv[0]
            || q[0] != pk[0]
            || q[0] != pv[0]
            || q[0] != indices[0]
            || q[2] != indices[1]
            || q[3] != lk[2]
            || q[3] != pk[2]
            || lk[1] != lv[1]
            || pk[1] != pv[1]
            || lv[2] != pv[2]
            || indices[2] <= 0
            || pk[1] <= 0
        {
            return Err(IndexedAttentionValidationError::Geometry);
        }
        if !scale.is_finite() || scale <= 0.0 {
            return Err(IndexedAttentionValidationError::Scale);
        }
        if sinks.is_some_and(|s| s != [q[1]]) {
            return Err(IndexedAttentionValidationError::Sinks);
        }
        Ok(Self {
            batch: q[0],
            heads: q[1],
            queries: q[2],
            key_dimensions: q[3],
            value_dimensions: lv[2],
            local: lk[1],
            pooled: pk[1],
            selected: indices[2],
        })
    }
}
