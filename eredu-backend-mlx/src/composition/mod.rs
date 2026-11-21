//! Cold-path architecture/backend composition selected by public loaders.

pub(crate) use crate::backend::nn::shared::MlxNeuralBackend;

mod observer_mapping;

pub mod mlx;
