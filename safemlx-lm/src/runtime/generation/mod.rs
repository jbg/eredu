//! MLX token sampling and committed-token streaming.

/// Token sampling policies.
#[cfg(feature = "mlx")]
pub mod sampler;
/// Protocol-independent semantic streaming.
pub mod streaming;
