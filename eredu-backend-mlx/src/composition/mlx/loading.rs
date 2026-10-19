//! MLX checkpoint materialization after backend-neutral planning.

use std::sync::Arc;

use eredu_architectures::{
    prepared_sources::{prepare_model_sources, PreparedModelSources},
    processor_plan::ArtifactArchitecturePlan,
};
use safemlx::Stream;

#[cfg(any(feature = "image", feature = "audio"))]
use crate::composition::mlx::ModelProcessor;

use crate::{
    backend::error::Error,
    backend::MlxModel,
    composition::{mlx::Executable, MlxNeuralBackend},
    MlxLoadRequest,
};

mod materialization;
mod selection;

pub(crate) use materialization::materialize_model_plan;
pub(crate) use selection::{select_preparation, MlxPreparationMechanisms};
pub use selection::{MlxModelConfig, MlxSelectedPreparation};
