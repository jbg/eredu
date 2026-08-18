//! MLX model loading, architecture dispatch, and generation extensions.
//!
//! Use [`crate::api::LoadedModel`] when you want to load a model directory
//! together with its tokenizer and chat template. Use
//! [`crate::load_model`] and [`crate::api::load_tokenizer`] when you
//! want to manage those pieces separately.
//! Ordinary generation is available for every `TextGenerationBackend`;
//! prepared-chat speculative generation is available through the
//! [`PreparedChatSpeculativeBackend`] capability on the same `LoadedModel<B>`.

use super::portable::LoadedModel;
use super::request::{
    PreparedChatMtpBatchOutput, PreparedChatMtpBatchRequest, PreparedChatMtpGenerationOutput,
    PreparedChatMtpGenerationRequest, PreparedChatSpeculativeBackend,
};
use crate::backend::mlx::speculative::MlxDrafter;
use crate::core::generation::SemanticEvent;
use crate::core::MtpCapability;
use crate::error::Error;

impl PreparedChatSpeculativeBackend for crate::backend::mlx::MlxBackend<'static> {
    type Drafter = MlxDrafter;
    type SpeculativeError = Error;

    fn mtp_capability(model: &LoadedModel<Self>) -> MtpCapability {
        model.mlx_mtp_capability()
    }

    fn execute_prepared_chat_mtp<'a, F>(
        model: &mut LoadedModel<Self>,
        request: PreparedChatMtpGenerationRequest<'a, Self, Self::Drafter, F>,
    ) -> Result<PreparedChatMtpGenerationOutput, Error>
    where
        F: FnMut(SemanticEvent),
    {
        model.execute_prepared_chat_mtp_mlx(request)
    }

    fn execute_prepared_chat_mtp_batch<'a>(
        model: &mut LoadedModel<Self>,
        request: PreparedChatMtpBatchRequest<'a, Self, Self::Drafter>,
    ) -> Result<PreparedChatMtpBatchOutput, Error> {
        model.execute_prepared_chat_mtp_batch_mlx(request)
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
