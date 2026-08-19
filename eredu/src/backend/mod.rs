//! Execution-backend implementations.
//!
//! This namespace is always available. Concrete implementations are exposed
//! only when their corresponding crate feature is enabled.

/// MLX implementation of the backend-neutral language-model contract.
#[cfg(feature = "mlx")]
pub mod mlx;
