//! Source-qualified independent lanes for the existing fair speculative driver.
use super::*;
use eredu_core::{SpeculativeGenerationBatchOutput, SpeculativeGenerationBatchRequest,
    SpeculativeGenerationLane, SpeculativeSchedulerOptions};
use eredu_core::run_preparation::TextPreparationStage as Stage;
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
        &mut self, source: &ManagedPlainTextSource,
        request: ManagedPlainTextSpeculativeBatchRequest<'a, B::Drafter, F>,
    ) -> Result<SpeculativeGenerationBatchOutput, ManagedPlainTextSpeculativeError> {
        let mut funding = None;
        self.run_managed_plain_speculative_batch(source, request, &mut funding)
            .map_err(|cause| ManagedPlainTextSpeculativeError { cause, funding })
    }
    fn run_managed_plain_speculative_batch<'a, F: FnMut(SemanticEvent) + 'a>(
        &mut self, source: &ManagedPlainTextSource,
        request: ManagedPlainTextSpeculativeBatchRequest<'a, B::Drafter, F>,
        retained: &mut Option<WorkspaceMetadataFunding>,
    ) -> Result<SpeculativeGenerationBatchOutput, Cause> {
        let ManagedPlainTextSpeculativeBatchRequest { drafting, lanes, scheduler } = request;
        let count = lanes.len();
        // One host vote for all lanes, matching ordinary batch preparation. An
        // invalid later policy cannot leave a peer entering an earlier prompt.
        let local = (|| {
            scheduler.validate()?;
            if !source.original().matches_configuration(&self.tokenizer) {
                return Err(TokenInputRejection::IdentityMismatch.into());
            }
            for lane in &lanes {
                lane.text.settings.inference.managed_memory_capacity_bytes
                    .ok_or(TokenInputRejection::Unsupported)?;
                self.resolve_text_generation_settings(lane.text.settings)?;
            }
            let mut input = lanes.into_iter();
            let Some(first) = input.next() else { return Ok(None); };
            let first = self.prepare_managed_speculative_host(source, first.text,
                first.max_draft_tokens, first.cancellation, first.on_event, retained)?;
            let funding = first.funding();
            let parts = [
                size_of::<ManagedPlainTextSpeculativeBatchRequest<'a, B::Drafter, F>>(),
                size_of::<ManagedPlainTextSpeculativeBatchLane<'a, F>>(),
                size_of::<std::vec::IntoIter<ManagedPlainTextSpeculativeBatchLane<'a, F>>>(),
                size_of::<Option<preparation::PreparedLane<'a>>>(),
                size_of::<Result<Option<eredu_core::SpeculativeBuffer<preparation::PreparedLane<'a>>>, Cause>>(),
                size_of::<eredu_runtime::RunSpeculativeGeneration>(),
                size_of::<SpeculativeGenerationBatchRequest<'a, B, B::Drafter, PreparedChatSpeculativeConstraint>>(),
                size_of::<Result<SpeculativeGenerationBatchOutput, Cause>>(),
                size_of::<Result<SpeculativeGenerationBatchOutput, ManagedPlainTextSpeculativeError>>(),
                size_of::<(SpeculativeDraft<'a, B::Drafter>, SpeculativeSchedulerOptions, usize)>(),
            ];
            reserve(funding, sum(&parts))?;
            let mut prepared = preparation::buffer(count, funding)?;
            prepared.try_push(first)?;
            for lane in input {
                prepared.try_push(self.prepare_managed_speculative_host(source, lane.text,
                    lane.max_draft_tokens, lane.cancellation, lane.on_event, retained)?)?;
            }
            Ok(Some(prepared))
        })();
        let prepared = self.runtime.finish_text_preparation(Stage::Request, local, Cause::Backend)?;
        let Some(prepared) = prepared else {
            // Zero lanes create no destination allocation or request authority.
            return Ok(SpeculativeGenerationBatchOutput::new(Vec::new(), Default::default()));
        };
        let funding = retained.as_ref().expect("first prepared lane retained batch custody");
        let local = (|| {
            let mut lanes = preparation::buffer::<SpeculativeGenerationLane<'a, B, PreparedChatSpeculativeConstraint>>(count, funding)?;
            for lane in prepared { lanes.try_push(lane.finish(self)?)?; }
            Ok(lanes)
        })();
        let lanes = self.runtime.finish_text_preparation(Stage::Prompt, local, Cause::Backend)?;
        let request = SpeculativeGenerationBatchRequest::new(drafting, lanes, self.tokenizer_fingerprint);
        B::with_speculative_execution(&mut self.runtime, request,
            eredu_runtime::RunSpeculativeGeneration::new(scheduler))
            .map_err(|cause| Cause::Backend(B::into_backend_failure(cause)))
    }
}
