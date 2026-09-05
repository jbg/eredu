//! MLX checkpoint materialization after backend-neutral planning.

use std::sync::Arc;

use eredu_architectures::{
    prepared_sources::{prepare_model_sources, PreparedModelSourceGraph, PreparedModelSources},
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
#[cfg(test)]
pub(super) use materialization::{bind_replicated_text, prepared_safetensors_architecture};
#[cfg(test)]
pub(crate) use selection::prepare_selected_sources;
#[cfg(test)]
pub(crate) use selection::select_preparation_with_grouped_capabilities;
pub(crate) use selection::{select_preparation, MlxPreparationMechanisms};
pub use selection::{MlxModelConfig, MlxSelectedPreparation};

#[cfg(test)]
use materialization::{inspected_floating_state_dtype_bytes, mlx_floating_state_dtype_bytes};
#[cfg(test)]
use selection::select_preparation_with_mechanism_capabilities;

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod floating_state_dtype_tests;
