//! Portable multimodal request preparation for the MLX session.

use safemlx_lm_core::{
    ModelRuntime, MultimodalPreparationBackend, MultimodalPreparationFailure,
    TokenizedMultimodalRequest,
};

use super::{MlxBackend, MlxModelInput};
use crate::{backend::mlx::error::Error, backend::mlx::runtime::media::ProcessorPreparationError};

impl MultimodalPreparationBackend for MlxBackend<'_> {
    fn prepare_multimodal_input<E>(
        runtime: &ModelRuntime<Self>,
        request: &TokenizedMultimodalRequest,
        encode_backend_text: &mut dyn FnMut(&str) -> Result<Vec<u32>, E>,
    ) -> Result<Self::Prompt, MultimodalPreparationFailure<Self::Error, E>>
    where
        E: std::backend::mlx::error::Error + Send + Sync + 'static,
    {
        let processor = runtime.session().processor().ok_or_else(|| {
            MultimodalPreparationFailure::Backend(Error::Processor(format!(
                "MLX model session {} has no multimodal processor",
                runtime.session().model_type()
            )))
        })?;
        let prepared = processor
            .prepare_portable_input(request, encode_backend_text)
            .map_err(|error| match error {
                ProcessorPreparationError::Backend(error) => {
                    MultimodalPreparationFailure::Backend(error)
                }
                ProcessorPreparationError::Text(error) => MultimodalPreparationFailure::Text(error),
            })?;
        Ok(MlxModelInput::from_prepared(&prepared))
    }
}
