//! Checked logical projection axes for multi-head latent attention.

use crate::{AttentionHeadGeometry, Error};

/// Head expansion and head-independent latent ingress geometry.
///
/// Query heads contain non-positional and positional features. Reconstruction
/// emits non-positional keys and values, either interleaved per head or through
/// separate affines. Packed widths are checked when requested by a fused layout.
/// Architectures retain rotary, NoPE, cache and score equations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LatentAttentionGeometry {
    heads: AttentionHeadGeometry,
    non_positional_dimensions: i32,
    latent_dimensions: i32,
}
impl LatentAttentionGeometry {
    /// Checks shared axes, latent ingress and independent projected widths before allocation.
    pub fn new(
        heads: i32,
        non_positional_dimensions: i32,
        positional_dimensions: i32,
        value_dimensions: i32,
        latent_dimensions: i32,
    ) -> Result<Self, Error> {
        let invalid = || Error::backend("invalid latent attention projection geometry");
        if non_positional_dimensions <= 0 || positional_dimensions < 0 || latent_dimensions <= 0 {
            return Err(invalid());
        }
        let query = non_positional_dimensions
            .checked_add(positional_dimensions)
            .ok_or_else(invalid)?;
        latent_dimensions
            .checked_add(positional_dimensions)
            .ok_or_else(invalid)?;
        Ok(Self {
            heads: AttentionHeadGeometry::new(heads, heads, query, value_dimensions)?,
            non_positional_dimensions,
            latent_dimensions,
        })
    }
    /// Expanded query/key/value heads, independent of retained latent state.
    pub const fn heads(self) -> i32 {
        self.heads.query_heads()
    }
    /// Non-positional key features reconstructed for each head.
    pub const fn non_positional_dimensions(self) -> i32 {
        self.non_positional_dimensions
    }
    /// Shared positional features appended to each query head.
    pub const fn positional_dimensions(self) -> i32 {
        self.heads.key_dimensions() - self.non_positional_dimensions
    }
    /// Value features per expanded head.
    pub const fn value_dimensions(self) -> i32 {
        self.heads.value_dimensions()
    }
    /// Head-independent normalized latent features.
    pub const fn latent_dimensions(self) -> i32 {
        self.latent_dimensions
    }
    /// Non-positional plus positional features in each query head.
    pub const fn query_head_dimensions(self) -> i32 {
        self.heads.key_dimensions()
    }
    /// Interleaved non-positional key and value features per reconstructed head.
    pub fn reconstruction_head_dimensions(self) -> Result<i32, Error> {
        self.non_positional_dimensions
            .checked_add(self.value_dimensions())
            .ok_or_else(|| Error::backend("latent attention reconstruction head width overflow"))
    }
    /// Complete query projection rows.
    pub const fn query_width(self) -> i32 {
        self.heads.query_width()
    }
    /// Complete independent non-positional key projection rows.
    pub const fn key_width(self) -> i32 {
        self.heads() * self.non_positional_dimensions
    }
    /// Concatenated value-head width entering the output projection.
    pub const fn value_width(self) -> i32 {
        self.heads.attended_width()
    }
    /// Complete fused non-positional key/value reconstruction rows.
    pub fn reconstruction_width(self) -> Result<i32, Error> {
        self.heads()
            .checked_mul(self.reconstruction_head_dimensions()?)
            .ok_or_else(|| Error::backend("latent attention reconstruction width overflow"))
    }
    /// Latent plus shared positional rows emitted by the ingress affine.
    pub const fn ingress_width(self) -> i32 {
        self.latent_dimensions + self.positional_dimensions()
    }
}
