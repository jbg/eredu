//! Source-qualified independent lanes for the existing fair speculative driver.
use super::*;
use eredu_core::{SpeculativeGenerationBatchOutput, SpeculativeSchedulerOptions};
use std::num::NonZeroUsize;

/// One original plain prompt with independent sampling, stopping and callback.
/// Application-owned callback payloads and the input string stay caller-owned.
pub struct ManagedPlainTextSpeculativeBatchLane<'a, F> {
    /// Literal prompt and its independent generation, seed and stopping policy.
    pub text: ManagedPlainTextRequest<'a>,
    /// Maximum assistant proposals verified for this lane in one target block.
    pub max_draft_tokens: NonZeroUsize,
    /// Cooperative cancellation affects only this lane.
    pub cancellation: GenerationCancellationToken,
    /// Receives only this lane's committed semantic events.
    pub on_event: F,
}
/// Independent original lanes sharing one selected assistant and fair scheduler.
/// Each lane receives its own prepared sources and nonrefunding request accounts.
pub struct ManagedPlainTextSpeculativeBatchRequest<'a, D, F> {
    /// Actual embedded or independently loaded assistant selection.
    pub drafting: SpeculativeDraft<'a, D>,
    /// Application-provided request list in stable output order.
    pub lanes: Vec<ManagedPlainTextSpeculativeBatchLane<'a, F>>,
    /// Existing bounded fairness and optimistic-lookahead controls.
    pub scheduler: SpeculativeSchedulerOptions,
}

impl<B: OriginalTokenizerBackend + SpeculativeGenerationBackend> LoadedModel<B> {
    /// Generates independent original plain lanes through the shared speculative
    /// scheduler. Every lane supplies a managed capacity and uses the same loaded
    /// tokenizer source. All host admission precedes prompt preparation and the
    /// actual backend dispatch; a failed lane prevents the entire batch starting.
    pub fn generate_managed_plain_text_speculative_batch<'a, F: FnMut(SemanticEvent) + 'a>(
        &mut self,
        source: &ManagedPlainTextSource,
        request: ManagedPlainTextSpeculativeBatchRequest<'a, B::Drafter, F>,
    ) -> Result<SpeculativeGenerationBatchOutput, ManagedPlainTextSpeculativeError> {
        let mut funding = None;
        self.run_managed_plain_speculative_batch(source, request, &mut funding)
            .map_err(|cause| ManagedPlainTextSpeculativeError { cause, funding })
    }
    fn run_managed_plain_speculative_batch<'a, F: FnMut(SemanticEvent) + 'a>(
        &mut self,
        source: &ManagedPlainTextSource,
        request: ManagedPlainTextSpeculativeBatchRequest<'a, B::Drafter, F>,
        retained: &mut Option<HostMetadataFunding>,
    ) -> Result<SpeculativeGenerationBatchOutput, Cause> {
        let ManagedPlainTextSpeculativeBatchRequest {
            drafting,
            lanes,
            scheduler,
        } = request;
        let validation = self.validate_speculative_settings(
            source,
            lanes.iter().map(|lane| lane.text.settings.clone()),
        );
        let request = self.prepare_speculative_batch(
            drafting,
            lanes.into_iter(),
            validation,
            scheduler,
            retained,
            |lane, retained| {
                self.prepare_managed_speculative_host(
                    source,
                    lane.text,
                    lane.max_draft_tokens,
                    lane.cancellation,
                    lane.on_event,
                    retained,
                )
            },
        )?;
        self.execute_speculative_batch(request, scheduler, retained)
    }
}
