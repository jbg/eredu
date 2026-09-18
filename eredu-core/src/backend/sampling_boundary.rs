//! Closed sampling updates at completed, drained ordinary boundaries.
use super::*;
use crate::execution_control::{
    SamplingOverride, SamplingOverrideError, SamplingStateFacts, TextSamplingControlBackend,
};

/// Permits one prospective sampler update through the backend's admitted worker.
/// It exposes neither mutable model state nor controller policy. The backend
/// must acquire any replacement sampling allowance before changing live state.
pub struct TextSamplingBoundary<'a, B: TextGenerationBackend> {
    pub(super) runtime: &'a mut ModelRuntime<B>,
    pub(super) state: &'a mut B::TextGenerationState,
    pub(super) context: &'a TextStepContext,
}

impl<B: TextSamplingControlBackend> TextSamplingBoundary<'_, B> {
    /// Reads current sampler compatibility without advancing state or randomness.
    pub fn facts(&self) -> SamplingStateFacts {
        B::sampling_control_facts(self.state)
    }

    /// Validates and installs a future change without revising the admitted
    /// model/controller source. Failure preserves the previous logical sampler.
    pub fn apply(
        &mut self,
        request: SamplingOverride,
    ) -> Result<SamplingStateFacts, SamplingOverrideError<B::Error>> {
        B::apply_sampling_override(self.runtime, self.state, Some(self.context), request)
    }
}

impl<B: TextSamplingControlBackend, C: TokenFilterController> ControlledTextGeneration<'_, B, C> {
    /// Requires exact completion and capture delivery before allowing a sampler
    /// update. The guard cannot submit a prediction or replace other run policy.
    pub fn sampling_boundary(
        &mut self,
    ) -> Result<TextSamplingBoundary<'_, B>, TextContinuationError<B::Error, C::Error>> {
        self.snapshot_source()?;
        Ok(TextSamplingBoundary {
            runtime: self.runtime,
            state: &mut self.inner.backend_state,
            context: &self.inner.step_context,
        })
    }
}
