//! Private producer composition. Public inference remains on existing entry
//! points until HF, outer input/events and native inventory obligations close.
use super::*;
use crate::api::{request::BackendGenerationTokenSource, ConstraintError};
use crate::runtime::{
    chat::constraints::ConstraintController, generation::streaming::RetainedConsumerCursor,
};
use eredu_core::{
    BackendFailure, GenerationDecoderError, GenerationSequenceRequest, TokenIdsInputPlan,
    TokenInputRejection,
};
use eredu_runtime::working_memory::LoadedGenerationDecoderInput;

type SourceError<B> = eredu_core::ControlledTextGenerationError<
    <B as eredu_core::BackendProvider>::Error,
    ConstraintError,
>;
pub(super) type InputCursor<B> =
    RetainedConsumerCursor<SourceError<B>, GenerationDecoderError, false, true>;
impl<B: TextGenerationBackend> LoadedModel<B> {
    /// Validates actual caller IDs by borrowed HF lookup (added-first spelling,
    /// sparse IDs). No tokenizer, source or destination copy is performed here.
    fn original_token_plan<'i>(
        &self,
        ids: &'i [u32],
    ) -> Result<TokenIdsInputPlan<'i>, BackendFailure> {
        let plan =
            TokenIdsInputPlan::new(ids).map_err(TokenInputRejection::into_backend_failure)?;
        let vocab = self.tokenizer.decode_vocabulary();
        if ids.iter().any(|&id| vocab.id_to_token(id).is_none()) {
            return Err(TokenInputRejection::InvalidToken.into_backend_failure());
        }
        Ok(plan)
    }
    /// The existing fixed validity controller, source adapter and retained cursor
    /// compose one explicit plain request. The caller selects its original stop
    /// header beforehand; this method does not compile or override source policy.
    /// The returned pair permits ordinary looping or explicit single advancement
    /// through the same cursor and the same Delivery/first-finish_step readiness.
    pub(crate) fn start_original_plain_tokens<'a>(
        &'a mut self,
        ids: &[u32],
        config: TextGenerationConfig,
        decoder: &LoadedGenerationDecoderInput,
    ) -> Result<(BackendGenerationTokenSource<'a, B>, InputCursor<B>), BackendFailure> {
        let plan = self.original_token_plan(ids)?;
        let maximum = config
            .sampling()
            .max_new_tokens
            .ok_or_else(|| TokenInputRejection::Unsupported.into_backend_failure())?;
        let consumer = InputCursor::<B>::layout()
            .ok_or_else(|| TokenInputRejection::Overflow.into_backend_failure())?;
        let controller = ConstraintController::text(self.token_validity.clone());
        let request = GenerationSequenceRequest::new(maximum, &self.eos_token_ids)
            .with_decoder(decoder)
            .with_consumer(&consumer);
        let mut generator = eredu_core::ControlledTextGeneration::from_token_ids_with_sequence(
            &mut self.runtime,
            plan,
            config,
            controller,
            None,
            request,
        )
        .map_err(|error| match error {
            eredu_core::ControlledTextGenerationError::Preparation(error) => error,
            error => BackendFailure::from_error(error),
        })?;
        let sequence = generator
            .take_prepared_sequence()
            .expect("genuine original input requires sequence provider");
        let cursor = InputCursor::<B>::from_sequence(sequence)
            .unwrap_or_else(|_| unreachable!("core checked exact original consumer/mode"));
        Ok((
            BackendGenerationTokenSource {
                generator,
                on_token: None,
                delivery_failure: None,
                generation_started: std::time::Instant::now(),
                time_to_first_token: None,
            },
            cursor,
        ))
    }
}
pub(crate) mod chat;
pub(crate) mod plain;
#[cfg(test)]
mod tests;
