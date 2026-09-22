//! MLX checkpoint materialization after backend-neutral planning.

use std::sync::Arc;

use eredu_architectures::{
    prepared_sources::{PreparedModelSources, prepare_model_sources},
    processor_plan::ArtifactArchitecturePlan,
};
use safemlx::Stream;

#[cfg(any(feature = "image", feature = "audio"))]
use crate::composition::mlx::ModelProcessor;

use crate::{
    MlxLoadRequest,
    backend::MlxModel,
    backend::error::Error,
    composition::{MlxNeuralBackend, mlx::Executable},
};

mod materialization;
mod construction_sources;
pub(crate) use construction_sources::PreparedNativeConstructionSources;
mod addressable;
pub(crate) use addressable::prepare_addressable_source;
mod selection;

#[cfg(test)]
pub(super) use materialization::{bind_replicated_text, prepared_safetensors_architecture};
pub(crate) use materialization::{
    materialize_model_plan, materialize_model_plan_with_construction_sources,
};
#[cfg(test)]
pub(crate) use selection::prepare_selected_sources;
#[cfg(test)]
pub(crate) use selection::select_preparation_with_grouped_capabilities;
pub use selection::{MlxModelConfig, MlxSelectedPreparation};
pub(crate) use selection::{MlxPreparationMechanisms, select_preparation};

#[cfg(test)]
use materialization::{inspected_floating_state_dtype_bytes, mlx_floating_state_dtype_bytes};
#[cfg(test)]
use selection::select_preparation_with_mechanism_capabilities;

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod floating_state_dtype_tests;
