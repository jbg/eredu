//! Backend-neutral tokenizer and media-preparation facade.

use eredu_core::{
    MediaBinding, MediaRequestError, MultimodalPreparationBackend, MultimodalPreparationFailure,
    MultimodalRequest,
};

use super::{LoadedModel, TextModelError};
use crate::runtime::chat::PreparedChat;

/// Failure while preparing portable media for a selected backend session.
#[derive(Debug, thiserror::Error)]
pub enum MultimodalPreparationError<E: std::error::Error + Send + Sync + 'static> {
    /// Decoded media or rendered-placeholder composition was invalid.
    #[error(transparent)]
    Request(#[from] MediaRequestError),
    /// The attached tokenizer failed to encode a text segment.
    #[error(transparent)]
    Text(#[from] TextModelError),
    /// The selected backend could not preprocess the request.
    #[error("selected backend failed multimodal preparation: {0}")]
    Backend(#[source] E),
}

impl<B: MultimodalPreparationBackend> LoadedModel<B> {
    /// Tokenizes text segments and asks this model's backend to prepare media.
    pub fn prepare_multimodal_input(
        &self,
        request: &MultimodalRequest,
    ) -> Result<B::Prompt, MultimodalPreparationError<B::Error>> {
        let tokenized = request.tokenize(|text| self.encode(text, false))?;
        B::prepare_multimodal_input(&self.runtime, &tokenized, &mut |text| {
            self.encode(text, false)
        })
        .map_err(|error| match error {
            MultimodalPreparationFailure::Backend(error) => {
                MultimodalPreparationError::Backend(error)
            }
            MultimodalPreparationFailure::Text(error) => MultimodalPreparationError::Text(error),
        })
    }

    /// Binds decoded media to a rendered chat and prepares its backend prompt.
    pub fn prepare_chat_multimodal_input(
        &self,
        prepared_chat: &PreparedChat,
        bindings: &[MediaBinding],
    ) -> Result<B::Prompt, MultimodalPreparationError<B::Error>> {
        let request = MultimodalRequest::from_chat(prepared_chat.rendered_prompt(), bindings)?;
        self.prepare_multimodal_input(&request)
    }
}
