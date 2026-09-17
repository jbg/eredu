//! Exact page-local causal/window/prefix coordinates for integer mask workers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("absolute attention mask has unrepresentable or invalid coordinates")]
pub struct AbsoluteAttentionMaskError;

/// A page-local mask; source and execution authority are not implied by geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AbsoluteAttentionMaskGeometry {
    query_start: i32,
    query_end: i32,
    key_start: i32,
    key_end: i32,
    window: Option<i32>,
    prefix: i32,
}
impl AbsoluteAttentionMaskGeometry {
    pub fn new(
        query_start: i64,
        queries: i32,
        key_start: i64,
        key_end: i64,
        window: Option<i32>,
        prefix: i64,
    ) -> Result<Self, AbsoluteAttentionMaskError> {
        let invalid = AbsoluteAttentionMaskError;
        if query_start < 0
            || queries <= 0
            || key_start < 0
            || key_end <= key_start
            || prefix < 0
            || window.is_some_and(|n| n <= 0)
        {
            return Err(invalid);
        }
        let query_start = i32::try_from(query_start).map_err(|_| invalid)?;
        let query_end = query_start.checked_add(queries).ok_or(invalid)?;
        let key_start = i32::try_from(key_start).map_err(|_| invalid)?;
        let key_end = i32::try_from(key_end).map_err(|_| invalid)?;
        // Prefix comparisons beyond the last representable key are identical
        // to the endpoint comparison; no floating-point coordinate is used.
        let prefix = prefix.min(i64::from(i32::MAX)) as i32;
        Ok(Self {
            query_start,
            query_end,
            key_start,
            key_end,
            window,
            prefix,
        })
    }
    pub fn query_start(self) -> i32 {
        self.query_start
    }
    pub fn query_end(self) -> i32 {
        self.query_end
    }
    pub fn queries(self) -> i32 {
        self.query_end - self.query_start
    }
    pub fn key_start(self) -> i32 {
        self.key_start
    }
    pub fn key_end(self) -> i32 {
        self.key_end
    }
    pub fn keys(self) -> i32 {
        self.key_end - self.key_start
    }
    pub fn window(self) -> Option<i32> {
        self.window
    }
    pub fn prefix(self) -> i32 {
        self.prefix
    }
    pub fn allows(self, query: i32, key: i32) -> bool {
        if query < 0 || query >= self.queries() || key < 0 || key >= self.keys() {
            return false;
        }
        let q = self.query_start + query;
        let k = self.key_start + key;
        k <= q && (k < self.prefix || self.window.is_none_or(|n| q - k < n))
    }
}
