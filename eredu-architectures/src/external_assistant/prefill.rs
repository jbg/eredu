//! One architecture consumer inside the existing shared prefill span.
use super::*;

/// Receives real selected target captures before the shared guard completes.
/// Implementations settle every capture-derived root before returning, retain
/// only their semantic seed, and never infer completion from a frontier number.
pub trait ExternalPrefillReceiver<T, E> {
    /// A default request with a whole-invocation observer must retain one full
    /// span. This is a callback geometry constraint, never budget authority.
    fn requires_whole_prompt(&self) -> bool {
        false
    }
    /// Selects source-derived span geometry before the shared request is made.
    /// Strict admission must price the same full span or reject it; it must not
    /// replace this constraint with a smaller span on allocation failure.
    fn prefill_chunk_positions(
        &self,
        requested: Option<std::num::NonZeroU64>,
        input_positions: Option<u64>,
        token_only: bool,
    ) -> Option<std::num::NonZeroU64> {
        if requested.is_none() && (self.requires_whole_prompt() || !token_only) {
            input_positions.and_then(std::num::NonZeroU64::new)
        } else {
            requested
        }
    }
    /// Consumes this particular completed target source. Ordinary producers
    /// retain their existing callback; prepared consumers retain the witness.
    fn consume_with_evidence(&mut self, chunk: &eredu_runtime::prefill::PrefillChunk,
        frontier: u64, scores: &mut Option<T>,
        capture: crate::composite_execution::ExternalPredictionTargetCapture<T>,
        _evidence: Option<crate::speculative_execution::PreparedEmbeddedEvidence>,
    ) -> Result<(), E> { self.consume(chunk, frontier, scores, capture) }
    /// Preserves the original installed observer's requested readout.
    fn output_demand(&self) -> eredu_core::OutputDemand;
    /// Validates compatibility before target execution; no collective work.
    fn prepare(&mut self, chunk: &eredu_runtime::prefill::PrefillChunk) -> Result<(), E>;
    /// Consumes exact architecture capture at the freshly inspected installed
    /// frontier. Scores retain the physical demand; shared final indexing remains
    /// outside this callback and under the same request's final guard.
    fn consume(
        &mut self,
        chunk: &eredu_runtime::prefill::PrefillChunk,
        frontier: u64,
        scores: &mut Option<T>,
        capture: crate::composite_execution::ExternalPredictionTargetCapture<T>,
    ) -> Result<(), E>;
}

pub(crate) struct CaptureIdentity<'a> {
    pub(super) selected: &'a eredu_runtime::SelectedSpeculativeRealization,
    pub(super) input: &'a SpeculativeIdentity,
}
impl CaptureIdentity<'_> {
    pub(crate) fn paths(&self) -> impl Iterator<Item = &str> {
        self.selected
            .requirements()
            .capture()
            .entries()
            .iter()
            .map(|entry| entry.path().as_str())
    }
    pub(crate) fn validate_values<'a,A,M,I>(&self,frontier:u64,values:impl FnMut()->I,
        context:M::Context<'_>)->Result<(),M::Error>
    where A:ExternalAssistantArchitecture,M:ExternalAssistantExecutionMechanisms<A>,
        M::Tensor:'a,I:Iterator<Item=&'a M::Tensor> {
        self.selected.validate_capture_values(&self.selected.lane_identity_ref(self.input,frontier),
            frontier,values,|value,entry|M::capture_shape_matches(value,entry,context),
            |error|M::capture_contract_error(error,context))
    }
}
