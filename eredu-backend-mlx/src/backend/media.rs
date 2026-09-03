//! Portable multimodal request preparation for the MLX session.

use eredu_core::{
    ModelRuntime, MultimodalPreparationBackend, MultimodalPreparationFailure,
    TokenizedMultimodalRequest,
};

use super::MlxBackend;
use crate::composition::mlx::MlxModelInput;
use crate::{backend::error::Error, backend::runtime::media::ProcessorPreparationError};

impl MlxBackend<'_> {
    /// Prepares raw multimodal input while exposing processor output for inspection or replacement.
    pub fn prepare_multimodal_input_with_observer<E>(
        runtime: &ModelRuntime<Self>,
        request: &TokenizedMultimodalRequest,
        encode_backend_text: &mut dyn FnMut(&str) -> Result<Vec<u32>, E>,
        observer: &mut dyn eredu_runtime::ActivationObserver<
            safemlx::Array,
            safemlx::error::Exception,
        >,
    ) -> Result<MlxModelInput, MultimodalPreparationFailure<Error, E>>
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        let processor = runtime.session().processor().ok_or_else(|| {
            MultimodalPreparationFailure::Backend(Error::Processor(format!(
                "MLX model session {} has no multimodal processor",
                runtime.session().effective_model_type()
            )))
        })?;
        let prepared = processor
            .prepare_portable_input_with_observer(request, encode_backend_text, observer)
            .map_err(|error| match error {
                ProcessorPreparationError::Backend(error) => {
                    MultimodalPreparationFailure::Backend(error)
                }
                ProcessorPreparationError::Text(error) => MultimodalPreparationFailure::Text(error),
            })?;
        Ok(MlxModelInput::from_prepared(&prepared))
    }
}

impl MultimodalPreparationBackend for MlxBackend<'_> {
    fn prepare_multimodal_input<E>(
        runtime: &ModelRuntime<Self>,
        request: &TokenizedMultimodalRequest,
        encode_backend_text: &mut dyn FnMut(&str) -> Result<Vec<u32>, E>,
    ) -> Result<Self::Prompt, MultimodalPreparationFailure<Self::Error, E>>
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        let processor = runtime.session().processor().ok_or_else(|| {
            MultimodalPreparationFailure::Backend(Error::Processor(format!(
                "MLX model session {} has no multimodal processor",
                runtime.session().effective_model_type()
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
