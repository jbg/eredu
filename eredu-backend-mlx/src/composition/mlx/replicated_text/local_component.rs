//! MLX storage and completion mechanisms for local layerwise components.

use crate::backend::runtime::cache::MlxCacheBackend;
use eredu_core::backend::LocalComponentStorage;
use eredu_core::BackendFailure;

use crate::{backend::nn::shared::MlxNeuralBackend, MlxTensor};

impl LocalComponentStorage for MlxNeuralBackend {
    type CacheStorage = MlxCacheBackend;

    fn settle_local_values<'a>(
        values: impl Iterator<Item = &'a MlxTensor>,
    ) -> Result<(), BackendFailure> {
        safemlx::transforms::eval(values.map(MlxTensor::as_array))
            .map_err(BackendFailure::from_error)
    }
}
