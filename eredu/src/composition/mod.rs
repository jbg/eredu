//! Cold-path architecture/backend composition selected by public loaders.

#[cfg(feature = "mlx")]
pub(crate) mod llama_checkpoint;
#[cfg(feature = "mlx")]
pub(crate) mod llama_mlx;
