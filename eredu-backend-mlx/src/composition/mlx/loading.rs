//! MLX checkpoint materialization after backend-neutral planning.

use safemlx::Stream;

use crate::{backend::error::Error, composition::MlxNeuralBackend};

mod materialization;
mod selection;

pub use selection::MlxModelPlacement;
pub(crate) use selection::{direct_rank, MlxPreparationMechanisms};
