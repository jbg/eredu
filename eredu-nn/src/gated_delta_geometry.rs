//! Checked projection and state axes for recurrent gated-delta declarations.

use crate::Error;
use std::ops::Range;

/// Key heads expand to independent value-head recurrent matrices.
///
/// These are parameter axes, not a scan equation, state owner, gate policy or
/// backend capability. Fused Q/K/V widths are checked only when requested;
/// separate-projection families need only represent their individual widths.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GatedDeltaHeadGeometry {
    key_heads: i32,
    value_heads: i32,
    key_dimensions: i32,
    value_dimensions: i32,
}
impl GatedDeltaHeadGeometry {
    /// Checks positive axes, integral head expansion and individual affine widths.
    pub fn new(
        key_heads: i32,
        value_heads: i32,
        key_dimensions: i32,
        value_dimensions: i32,
    ) -> Result<Self, Error> {
        if key_heads <= 0
            || value_heads <= 0
            || key_dimensions <= 0
            || value_dimensions <= 0
            || value_heads % key_heads != 0
            || key_heads.checked_mul(key_dimensions).is_none()
            || value_heads.checked_mul(value_dimensions).is_none()
        {
            return Err(Error::backend("invalid gated-delta head geometry"));
        }
        Ok(Self {
            key_heads,
            value_heads,
            key_dimensions,
            value_dimensions,
        })
    }
    /// Query/key heads before expansion.
    pub const fn key_heads(self) -> i32 {
        self.key_heads
    }
    /// Independent value heads and recurrent matrices.
    pub const fn value_heads(self) -> i32 {
        self.value_heads
    }
    /// Rows in each recurrent matrix.
    pub const fn key_dimensions(self) -> i32 {
        self.key_dimensions
    }
    /// Columns in each recurrent matrix.
    pub const fn value_dimensions(self) -> i32 {
        self.value_dimensions
    }
    /// Complete independent query or key projection rows.
    pub const fn key_width(self) -> i32 {
        self.key_heads * self.key_dimensions
    }
    /// Complete value projection rows and output-projection input columns.
    pub const fn value_width(self) -> i32 {
        self.value_heads * self.value_dimensions
    }
    /// Checked fused query, key and value projection rows.
    pub fn qkv_width(self) -> Result<i32, Error> {
        self.key_width()
            .checked_mul(2)
            .and_then(|key| key.checked_add(self.value_width()))
            .ok_or_else(|| Error::backend("gated-delta QKV width overflowed"))
    }
    /// Component-major fused row ranges, shared by placement and storage extensions.
    pub fn qkv_ranges(self) -> Result<[Range<usize>; 3], Error> {
        let width = self.qkv_width()? as usize;
        let key = self.key_width() as usize;
        Ok([0..key, key..2 * key, 2 * key..width])
    }
}
