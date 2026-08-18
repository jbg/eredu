//! Facade composition of portable capability policy and MLX observations.

use safemlx::Stream;
use safemlx_lm_core::{
    apply_admission_policy, AdmissionRequest, AdmissionResult, AvailableMemory, CapabilityError,
    InputTokenCount, ModelCapabilities, RuntimeStateEstimate, StaticMemoryReport,
};

use super::LoadedModel;
use crate::{
    backend::mlx::{capability, MlxBackend},
    runtime::{chat::PreparedChat, media::PreparedModelInput},
};

impl LoadedModel<MlxBackend<'static>> {
    /// Returns architecture-independent capabilities derived from validated loaded state.
    pub fn capabilities(&self) -> Result<ModelCapabilities, CapabilityError> {
        capability::model_capabilities(self.model())
    }

    /// Counts an ordinary encoded prompt exactly.
    pub fn count_token_ids(&self, token_ids: &[u32]) -> Result<InputTokenCount, CapabilityError> {
        let tokens =
            u64::try_from(token_ids.len()).map_err(|_| CapabilityError::ArithmeticOverflow {
                operation: "token-id length",
            })?;
        Ok(InputTokenCount::text(tokens))
    }

    /// Tokenizes and counts a rendered text prompt exactly.
    pub fn count_text(
        &self,
        text: &str,
        add_special_tokens: bool,
    ) -> Result<InputTokenCount, CapabilityError> {
        let ids = self
            .encode(text, add_special_tokens)
            .map_err(|error| CapabilityError::Observation(error.to_string()))?;
        self.count_token_ids(&ids)
    }

    /// Counts the exact rendered prompt stored in a prepared chat.
    pub fn count_prepared_chat(
        &self,
        chat: &PreparedChat,
    ) -> Result<InputTokenCount, CapabilityError> {
        self.count_text(chat.rendered_prompt(), false)
    }

    /// Counts text IDs and actual model positions in processor-prepared multimodal input.
    pub fn count_prepared_input(
        &self,
        prepared: &PreparedModelInput,
        stream: &Stream,
    ) -> Result<InputTokenCount, CapabilityError> {
        capability::count_prepared_input(self.model(), prepared, stream)
    }

    /// Estimates persistent request state and prepared-media execution workspace.
    pub fn estimate_runtime_state(
        &self,
        input: InputTokenCount,
        max_output_tokens: u64,
        batch_size: u64,
    ) -> Result<RuntimeStateEstimate, CapabilityError> {
        capability::model_runtime_state(self.model(), input, max_output_tokens, batch_size)
    }

    /// Reports logical checkpoint/residency accounting and MLX allocator observations.
    pub fn static_memory(&self) -> Result<StaticMemoryReport, CapabilityError> {
        capability::static_model_memory(self.model())
    }

    /// Applies portable context and memory policy without allocating a model cache.
    pub fn admit(
        &self,
        request: AdmissionRequest,
        available: Option<&AvailableMemory>,
    ) -> Result<AdmissionResult, CapabilityError> {
        let capabilities = self.capabilities()?;
        let state = self.estimate_runtime_state(
            request.input,
            request.max_output_tokens,
            request.batch_size,
        )?;
        apply_admission_policy(&capabilities, request, state, available)
    }
}
