//! Facade composition of portable capability policy and backend observations.

use eredu_core::{
    apply_admission_policy, AdmissionRequest, AdmissionResult, AvailableMemory, CapabilityError,
    InputTokenCount, ModelCapabilities, ModelCapabilityBackend, RuntimeStateEstimate,
    StaticMemoryReport,
};

use super::LoadedModel;
use crate::runtime::chat::PreparedChat;

impl<B: ModelCapabilityBackend> LoadedModel<B> {
    /// Returns backend-independent capabilities derived from validated loaded state.
    pub fn capabilities(&self) -> Result<ModelCapabilities, CapabilityError> {
        B::model_capabilities(&self.runtime)
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

    /// Counts text IDs and actual model positions in a backend-prepared input.
    pub fn count_prepared_input(
        &self,
        input: &B::Prompt,
    ) -> Result<InputTokenCount, CapabilityError> {
        B::count_prepared_input(&self.runtime, input)
    }

    /// Estimates persistent request state and prepared-media execution workspace.
    pub fn estimate_runtime_state(
        &self,
        input: InputTokenCount,
        max_output_tokens: u64,
        batch_size: u64,
    ) -> Result<RuntimeStateEstimate, CapabilityError> {
        B::estimate_runtime_state(&self.runtime, input, max_output_tokens, batch_size)
    }

    /// Reports logical checkpoint/residency accounting and backend allocator observations.
    pub fn static_memory(&self) -> Result<StaticMemoryReport, CapabilityError> {
        B::static_memory(&self.runtime)
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

impl<B: ModelCapabilityBackend> LoadedModel<B> {
    /// Reads this execution's conversion-retention budgets without evaluating,
    /// settling, or changing request state. Embedded owners share these budgets.
    pub fn parameter_conversion_retention(
        &self,
    ) -> Result<
        eredu_core::Observed<Vec<eredu_core::residency::ParameterConversionRetentionReport>>,
        eredu_core::BackendFailure,
    > {
        B::parameter_conversion_retention(&self.runtime)
    }
}

impl<B: ModelCapabilityBackend, D: eredu_core::residency::ParameterConversionRetentionObserver>
    super::PlannedModel<B, D>
{
    /// Reports every loaded participant's retention budget. External drafting
    /// has an independent allowance; embedded owners are included in the target.
    /// Queries never settle, evaluate, reserve, or advance execution.
    pub fn parameter_conversion_retention(
        &self,
    ) -> Result<eredu_core::residency::ExecutionConversionRetentionReport, eredu_core::BackendFailure>
    {
        Ok(eredu_core::residency::ExecutionConversionRetentionReport {
            target: self.model().parameter_conversion_retention()?,
            external_drafter: match self.drafting() {
                eredu_core::RealizedDrafting::External(draft) => {
                    Some(draft.parameter_conversion_retention()?)
                }
                _ => None,
            },
        })
    }
}

impl<B: ModelCapabilityBackend, D: eredu_core::residency::ParameterConversionRetentionTrimmer>
    super::PlannedModel<B, D>
{
    /// Releases optional conversions for the target and separately loaded drafter.
    /// Each participant establishes its ordinary settled boundary; no request
    /// state, sampling state or allocator-cache policy is changed.
    pub fn trim_parameter_conversions(
        &mut self,
    ) -> Result<
        eredu_core::residency::ExecutionConversionRetentionTrimReport,
        eredu_core::residency::ParameterConversionTrimError,
    > {
        let (model, drafting) = self.parts_mut();
        let target = model.trim_parameter_conversions()?;
        let external_drafter = match drafting {
            eredu_core::RealizedDrafting::External(draft) => {
                Some(draft.trim_parameter_conversions()?)
            }
            _ => None,
        };
        Ok(
            eredu_core::residency::ExecutionConversionRetentionTrimReport {
                target,
                external_drafter,
            },
        )
    }
}
