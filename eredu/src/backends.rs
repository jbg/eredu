//! Feature-gated native factories for artifact inspection and planned loading.

#[cfg(feature = "candle")]
pub use eredu_backend_candle::CandleBackendFactory;
#[cfg(feature = "mlx")]
pub use eredu_backend_mlx::MlxBackendFactory;
