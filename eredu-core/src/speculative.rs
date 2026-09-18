//! High-level contracts and orchestration for speculative execution backends.

use crate::generation::{SemanticState, SemanticStateOwner};

/// Allocation-free byte-trigger matching shared by semantic controllers.
pub mod byte_trigger;
mod forbidden_controller;
pub use forbidden_controller::{
    ForbiddenControllerDecision, ForbiddenControllerError, ForbiddenControllerInputs,
    ForbiddenControllerMutation, ForbiddenControllerSource, PreparedForbiddenControllerIdentity,
    PreparedForbiddenInputCopy, packed_controller_token_bytes,
};

mod grammar_controller;
pub use grammar_controller::{NoPreparedGrammar, PreparedGrammarIdentity, PreparedGrammarController, PreparedGrammarInstallCause, PreparedGrammarInstallError, PreparedGrammarSource};

mod plain_controller;
pub use plain_controller::{
    PlainControllerError, PlainControllerHistory, PlainControllerSource,
    PreparedPlainControllerIdentity,
};

mod frontdoor;
pub use frontdoor::{
    SpeculativeConfiguration, SpeculativeConfigurationError, SpeculativeEventCallback,
};

mod configuration;
use configuration::RequestConfiguration;

mod geometry;
mod prefill;
pub use geometry::SpeculativeRequestGeometry;
pub use prefill::{PredictionPrefillAlignment, PredictionPrefillSeed};

mod identity;
pub use identity::SpeculativeRequestIdentity;

mod sequence;
pub use sequence::{
    SpeculativeSequence, SpeculativeSequenceAllocationError, SpeculativeSequenceRef,
    SpeculativeTokenIds, SpeculativeTokenIdsIntoIter,
};

mod buffer;
pub use buffer::{SpeculativeBuffer, SpeculativeBufferAllocationError, SpeculativeBufferIntoIter};

#[cfg(test)]
use crate::{GenerationSequence, backend::BoundedCompletionOutcome};
use crate::{
    backend::{
        BoundedCompletion, BoundedCompletionWait, Completion, CompletionCancellationMode,
        ModelRuntime, SpeculativeTokenFilterController, Submission, TextGenerationBackend,
        TextGenerationConfig,
    },
    generation::{
        FinishReason, GenerationCancellationToken, GenerationError,
        SpeculativeCancellationDisposition, SpeculativeConfig, SpeculativeRequestId,
        SpeculativeRequestLifecycle, SpeculativeRequestStatus, SpeculativeRound,
        SpeculativeSchedulerOptions, TokenTerminalSignals,
    },
};
use serde::{Deserialize, Serialize};
use std::{
    sync::Arc,
    time::{Duration, Instant},
};

mod values;
pub use values::{SpeculativeValues, SpeculativeValuesIntoIter};

mod control;
pub use control::*;
mod activation;
pub use activation::*;
mod coordination;
pub use coordination::*;
mod rollback;
pub use rollback::SpeculativeRollbackFailure;

/// Draft-model source selected for one speculative-generation request.
#[non_exhaustive]
pub enum SpeculativeDraft<'a, D> {
    /// Separately prepared assistant owned by the selected backend.
    External(&'a mut D),
    /// Draft heads embedded in the selected target model.
    Embedded,
}

/// Shared terminal generation output with speculative execution statistics.
pub type SpeculativeGenerationOutput =
    crate::generation::GenerationOutput<SpeculativeStats, SpeculativeTokenIds>;

/// Completed speculative requests plus aggregate fair-scheduler telemetry.
pub struct SpeculativeGenerationBatchOutput {
    /// Per-request results in submission order.
    requests: SpeculativeBuffer<SpeculativeGenerationOutput>,
    /// Aggregate scheduler telemetry.
    scheduler: SpeculativeSchedulerStats,
}

impl SpeculativeGenerationBatchOutput {
    /// Creates a completed batch in stable submission order.
    pub fn new(
        requests: impl Into<SpeculativeBuffer<SpeculativeGenerationOutput>>,
        scheduler: SpeculativeSchedulerStats,
    ) -> Self {
        Self {
            requests: requests.into(),
            scheduler,
        }
    }
    /// Per-request results in submission order.
    pub fn requests(&self) -> &[SpeculativeGenerationOutput] {
        &self.requests
    }
    /// Consumes the batch and returns its request results.
    pub fn into_requests(self) -> SpeculativeBuffer<SpeculativeGenerationOutput> {
        self.requests
    }
    /// Aggregate scheduler telemetry.
    pub const fn scheduler(&self) -> &SpeculativeSchedulerStats {
        &self.scheduler
    }
    /// Appends a result while adapting another backend-neutral execution path.
    pub fn push_request(
        &mut self,
        request: SpeculativeGenerationOutput,
    ) -> Result<(), GenerationError> {
        self.requests.try_push(request)
    }
    /// Clears adapted request results while retaining scheduler telemetry.
    pub fn clear_requests(&mut self) {
        self.requests.clear();
    }
}

/// One independently executable lane in a speculative batch.
pub struct SpeculativeGenerationLane<'a, B, C>
where
    B: TextGenerationBackend,
    C: SpeculativeTokenFilterController,
{
    /// Backend-owned prompt prepared by the selected session backend.
    prompt: Option<B::Prompt>,
    /// Fully resolved portable sampling configuration and random seed.
    generation: Option<TextGenerationConfig>,
    /// Resolved token budget, proposal width, temperature, and EOS ids.
    config: Option<SpeculativeConfiguration>,
    /// Portable canonical grammar state.
    constraint: Option<C>,
    /// Transactional decoded semantic parser state.
    semantic: Option<SemanticStateOwner>,
    /// Cooperative cancellation owned by this lane.
    cancellation: Option<GenerationCancellationToken>,
    /// Called synchronously for canonical events from this lane.
    on_event: Option<SpeculativeEventCallback<'a>>,
}

impl<'a, B, C> SpeculativeGenerationLane<'a, B, C>
where
    B: TextGenerationBackend,
    C: SpeculativeTokenFilterController,
{
    /// Creates one independently executable speculative lane.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        prompt: B::Prompt,
        generation: TextGenerationConfig,
        config: impl Into<SpeculativeConfiguration>,
        constraint: C,
        semantic: impl Into<SemanticStateOwner>,
        cancellation: GenerationCancellationToken,
        on_event: impl Into<SpeculativeEventCallback<'a>>,
    ) -> Self {
        Self {
            prompt: Some(prompt),
            generation: Some(generation),
            config: Some(config.into()),
            constraint: Some(constraint),
            semantic: Some(semantic.into()),
            cancellation: Some(cancellation),
            on_event: Some(on_event.into()),
        }
    }
    /// Takes the backend-owned prompt exactly once.
    pub fn take_prompt(&mut self) -> B::Prompt {
        self.prompt.take().expect("lane prompt already taken")
    }
    /// Borrows the backend-owned prompt before preparation consumes it.
    pub fn prompt(&self) -> &B::Prompt {
        self.prompt.as_ref().expect("lane prompt already taken")
    }
    /// Takes the resolved generation controls exactly once.
    pub fn take_generation(&mut self) -> TextGenerationConfig {
        self.generation
            .take()
            .expect("lane generation already taken")
    }
    /// Borrows resolved generation controls.
    pub fn generation(&self) -> &TextGenerationConfig {
        self.generation
            .as_ref()
            .expect("lane generation already taken")
    }
    /// Takes the speculative controls exactly once.
    pub fn take_config(&mut self) -> SpeculativeConfiguration {
        self.config.take().expect("lane config already taken")
    }
    /// Borrows speculative controls.
    pub fn config(&self) -> &SpeculativeConfig {
        self.config.as_ref().expect("lane config already taken")
    }
    /// Borrows the closed initial configuration for source authentication.
    pub fn configuration(&self) -> &SpeculativeConfiguration {
        self.config.as_ref().expect("lane config already taken")
    }
    /// Borrows the closed event callback for source authentication.
    pub fn event_callback(&self) -> &SpeculativeEventCallback<'a> {
        self.on_event
            .as_ref()
            .expect("lane event callback already taken")
    }
    /// Takes the grammar controller exactly once.
    pub fn take_constraint(&mut self) -> C {
        self.constraint
            .take()
            .expect("lane constraint already taken")
    }
    /// Borrows the closed semantic source without permitting replacement.
    pub fn semantic(&self) -> &SemanticStateOwner {
        self.semantic
            .as_ref()
            .expect("lane semantic state already taken")
    }
    /// Takes semantic state exactly once.
    pub fn take_semantic(&mut self) -> SemanticStateOwner {
        self.semantic
            .take()
            .expect("lane semantic state already taken")
    }
    /// Takes cancellation state exactly once.
    pub fn take_cancellation(&mut self) -> GenerationCancellationToken {
        self.cancellation
            .take()
            .expect("lane cancellation already taken")
    }
    /// Takes the event callback exactly once.
    pub fn take_on_event(&mut self) -> SpeculativeEventCallback<'a> {
        self.on_event
            .take()
            .expect("lane event callback already taken")
    }
}

/// Backend-preparation input for one or more speculative lanes.
pub struct SpeculativeGenerationBatchRequest<'a, B, D, C>
where
    B: TextGenerationBackend,
    C: SpeculativeTokenFilterController,
{
    /// Embedded or separately prepared draft-model selection.
    drafting: Option<SpeculativeDraft<'a, D>>,
    /// Independently prepared speculative lanes.
    lanes: Option<SpeculativeBuffer<SpeculativeGenerationLane<'a, B, C>>>,
    /// Target tokenizer vocabulary identity used for drafter compatibility.
    tokenizer_fingerprint: [u8; 32],
}

impl<'a, B, D, C> SpeculativeGenerationBatchRequest<'a, B, D, C>
where
    B: TextGenerationBackend,
    C: SpeculativeTokenFilterController,
{
    /// Creates one validated backend-preparation request.
    pub fn new(
        drafting: SpeculativeDraft<'a, D>,
        lanes: impl Into<SpeculativeBuffer<SpeculativeGenerationLane<'a, B, C>>>,
        tokenizer_fingerprint: [u8; 32],
    ) -> Self {
        Self {
            drafting: Some(drafting),
            lanes: Some(lanes.into()),
            tokenizer_fingerprint,
        }
    }
    /// Target tokenizer vocabulary identity.
    pub const fn tokenizer_fingerprint(&self) -> [u8; 32] {
        self.tokenizer_fingerprint
    }
    /// Takes draft selection exactly once.
    pub fn take_drafting(&mut self) -> SpeculativeDraft<'a, D> {
        self.drafting.take().expect("draft selection already taken")
    }
    /// Takes prepared lanes exactly once.
    pub fn take_lanes(&mut self) -> SpeculativeBuffer<SpeculativeGenerationLane<'a, B, C>> {
        self.lanes.take().expect("speculative lanes already taken")
    }
}

/// Optional speculative model-session capability.
///
/// Implementations prepare native executors, caches, sampling state, and
/// execution placement, then expose them to the caller-provided neutral
/// visitor. The backend must not drive request lifecycles or fair scheduling.
/// A backend is selected for the complete model session; requests cannot mix
/// runtime implementations.
pub trait SpeculativeGenerationBackend: TextGenerationBackend {
    /// Backend-owned separately prepared draft model.
    type Drafter;

    /// Reports fail-closed speculative support for the selected model session.
    fn speculative_capability(runtime: &ModelRuntime<Self>) -> SpeculativeCapability;

    /// Whether this selected target/draft path accepts an explicit prefill chunk
    /// policy. This cold query must not allocate or enter native work. Returning
    /// true requires the shared scheduler through ordinary and controlled
    /// execution; it does not imply managed-memory or observation support.
    fn supports_speculative_prefill_chunking(
        _runtime: &ModelRuntime<Self>,
        _drafting: &SpeculativeDraft<'_, Self::Drafter>,
    ) -> bool {
        false
    }

    /// Exact internal activation/edit support for the selected speculative path.
    /// Ordinary sampler-logit capture has a separate admission contract.
    fn speculative_activation_discovery(
        _runtime: &ModelRuntime<Self>,
    ) -> Result<SpeculativeActivationDiscovery, crate::capture::CaptureError> {
        Err(crate::capture::CaptureError::Unsupported(
            "backend has no scoped speculative activation admission".into(),
        ))
    }

    /// Rechecks immutable internal authority against the actual loaded session.
    fn validate_speculative_activations(
        runtime: &ModelRuntime<Self>,
        plan: &AdmittedSpeculativeActivations,
    ) -> Result<(), crate::capture::CaptureError> {
        plan.validate(&Self::speculative_activation_discovery(runtime)?)
    }

    /// Genuine mutable hooks with explicit speculative attribution.
    fn speculative_intervention_discovery(
        _runtime: &ModelRuntime<Self>,
    ) -> Result<crate::intervention::InterventionDiscovery, crate::capture::CaptureError> {
        Err(crate::capture::CaptureError::Unsupported(
            "backend has no speculative tensor interventions".into(),
        ))
    }
    /// Validates edit geometry, exact loaded capabilities and evidence budgets.
    fn validate_speculative_interventions(
        _runtime: &ModelRuntime<Self>,
        _capture: &crate::capture::AdmittedCapturePlan,
        _plan: &crate::intervention::AdmittedInterventionPlan,
    ) -> Result<(), crate::capture::CaptureError> {
        Err(crate::capture::CaptureError::Unsupported(
            "backend has no speculative tensor interventions".into(),
        ))
    }

    /// Validates speculative observation support without allocating or executing.
    /// Ordinary capture support does not imply attribution for speculative phases.
    fn validate_speculative_capture(
        _runtime: &ModelRuntime<Self>,
        plan: &crate::capture::AdmittedCapturePlan,
    ) -> Result<(), crate::capture::CaptureError> {
        if plan.is_empty() {
            Ok(())
        } else {
            Err(crate::capture::CaptureError::Unsupported(
                "backend has no bounded speculative capture".into(),
            ))
        }
    }

    /// Prepares native execution resources and lends them to neutral orchestration.
    fn with_speculative_execution<C, V>(
        runtime: &mut ModelRuntime<Self>,
        request: SpeculativeGenerationBatchRequest<'_, Self, Self::Drafter, C>,
        visitor: V,
    ) -> Result<SpeculativeGenerationBatchOutput, Self::Error>
    where
        C: SpeculativeTokenFilterController,
        V: SpeculativeGenerationVisitor;
}

/// Relationship between target and assistant execution placements.
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum SpeculativeExecutionTopology {
    /// Target and assistant operations share one ordered execution queue.
    #[default]
    Single,
    /// Distinct queues share one device and can use ordered handoffs.
    SameDeviceSplit,
    /// Target and assistant use different devices and require transfers.
    CrossDeviceSplit,
}

impl std::fmt::Display for SpeculativeExecutionTopology {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::Single => "single",
            Self::SameDeviceSplit => "same-device-split",
            Self::CrossDeviceSplit => "cross-device-split",
        })
    }
}

/// How a model exposes speculative draft-token weights.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum SpeculativeDraftSource {
    /// Drafting weights live in a separately prepared model.
    Separate,
    /// Drafting weights are embedded in the selected target model.
    Embedded,
}

/// Fail-closed speculative-decoding capability of a prepared model session.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum SpeculativeCapability {
    /// The model does not advertise executable draft weights.
    Unavailable,
    /// The target declares an interface for the stated draft source.
    ///
    /// Further preparation is still required before execution is ready. For
    /// an external assistant this includes tokenizer and architecture
    /// compatibility proof, assistant construction, and target/assistant
    /// pairing.
    Declared {
        /// Location of the drafting weights.
        draft_source: SpeculativeDraftSource,
    },
    /// Speculative execution is already available with the stated draft source.
    Ready {
        /// Location of the drafting weights.
        draft_source: SpeculativeDraftSource,
    },
    /// Draft weights exist, but this backend cannot execute them.
    Unsupported {
        /// Location of the drafting weights.
        draft_source: SpeculativeDraftSource,
        /// Stable architecture identity reported by the backend.
        architecture: String,
    },
}

impl SpeculativeCapability {
    /// Returns the declared draft source, including a source that is not ready yet.
    pub const fn draft_source(&self) -> Option<SpeculativeDraftSource> {
        match self {
            Self::Declared { draft_source }
            | Self::Ready { draft_source }
            | Self::Unsupported { draft_source, .. } => Some(*draft_source),
            Self::Unavailable => None,
        }
    }

    /// Returns whether planning may attempt to realize the requested source.
    ///
    /// This accepts both a declaration awaiting preparation and an already
    /// ready realization. It rejects a source that is absent or unsupported.
    pub fn admits_source(&self, requested: SpeculativeDraftSource) -> bool {
        matches!(
            self,
            Self::Declared { draft_source } | Self::Ready { draft_source }
                if *draft_source == requested
        )
    }

    /// Returns whether the requested source is executable in this session now.
    pub fn is_ready_for(&self, requested: SpeculativeDraftSource) -> bool {
        matches!(
            self,
            Self::Ready { draft_source } if *draft_source == requested
        )
    }
}

/// Scalar telemetry shared by ordinary and retained speculative statistics.
/// Copying these counters allocates nothing and carries no execution authority.
#[derive(Debug, Clone, Copy, Default)]
pub struct SpeculativeStatsCounters {
    /// Relationship between the request's target and draft execution placements.
    execution_topology: SpeculativeExecutionTopology,
    /// Target tokens evaluated during prefill and verification.
    target_tokens: usize,
    /// Assistant tokens proposed.
    draft_tokens: usize,
    /// Assistant tokens accepted by target verification.
    accepted_tokens: usize,
    /// Number of target verification rounds.
    rounds: usize,
    /// Tokens emitted, including a terminal EOS token when one is produced.
    emitted_tokens: usize,
    /// Tokens drafted on an optimistic continuation.
    optimistic_draft_tokens: usize,
    /// Optimistic continuation blocks drafted.
    optimistic_draft_blocks: usize,
    /// Optimistically drafted tokens promoted after full acceptance.
    reused_optimistic_tokens: usize,
    /// Optimistic continuation blocks promoted after full acceptance.
    reused_optimistic_blocks: usize,
    /// First optimistic tokens consumed by matching target bonuses.
    consumed_optimistic_tokens: usize,
    /// Optimistically drafted tokens discarded.
    discarded_optimistic_tokens: usize,
    /// Optimistic continuation blocks discarded.
    discarded_optimistic_blocks: usize,
    /// Target bonus tokens emitted while an optimistic branch existed.
    optimistic_target_bonus_tokens: usize,
    /// Non-terminal target bonuses matching the first optimistic token.
    optimistic_bonus_matches: usize,
    /// Non-terminal target bonuses differing from the first optimistic token.
    optimistic_bonus_mismatches: usize,
    /// Whether deterministic cost accounting disabled further optimistic branches.
    adaptive_lookahead_disabled: bool,
    /// Host wall time spent producing optional same-request branches.
    optimistic_draft_time: Duration,
    /// Host wall time retained target verification remained in flight.
    verification_in_flight_time: Duration,
    /// Whether architecture component timings were collected.
    component_timings_collected: bool,
    /// Device execution time spent encoding committed target context.
    draft_context_time: Duration,
    /// Device execution time spent executing assistant proposal blocks.
    draft_assistant_time: Duration,
    /// Device execution time spent projecting proposal states to logits.
    draft_head_time: Duration,
    /// Device execution time spent executing target verification passes.
    target_verification_time: Duration,
    /// Scheduler operations performed for this request.
    scheduler_turns: usize,
    /// Draft turns performed while another request had target work in flight.
    cross_request_draft_opportunities: usize,
    /// Wall-clock generation duration.
    elapsed: Duration,
    /// Time from lane submission to its first committed token.
    submission_to_first_token: Option<Duration>,
}

/// Statistics with the actual immutable-history custody of this branch.
/// Managed history copies are fallible; scalar observations need no allocation.
#[derive(Debug, Default)]
pub struct SpeculativeStats {
    counters: SpeculativeStatsCounters,
    accept_lens: SpeculativeBuffer<usize>,
}
impl SpeculativeStats {
    /// Accepted proposal count per completed round.
    pub fn accept_lens(&self) -> &[usize] {
        &self.accept_lens
    }
    /// Stack-only telemetry comparison without copying the history allocation.
    pub const fn counters(&self) -> SpeculativeStatsCounters {
        self.counters
    }
    /// The same concrete provider query used for the next actual copy.
    pub fn copy_storage_bytes<E: SpeculativeExecutor>(&self, executor: &E) -> Option<u64> {
        u64::try_from(executor.driver_buffer_bytes::<usize>(self.accept_lens.len())?).ok()
    }
    /// Copies the current history and, for a verification transaction, one
    /// next-round slot through the actual driver destination. Snapshot copies
    /// request only their existing prefix. Scalar state moves unchanged.
    pub fn copy_for_driver<E: SpeculativeExecutor>(
        &self,
        executor: &E,
        append_round: bool,
        context: E::Context<'_>,
    ) -> Result<Self, SpeculativeDriverError<E::Error>> {
        let capacity = driver_capacity(self.accept_lens.len(), usize::from(append_round))?;
        let mut accept_lens = executor.driver_buffer(capacity, context)?;
        accept_lens
            .try_extend(self.accept_lens.iter().copied())
            .map_err(SpeculativeDriverError::Generation)?;
        Ok(Self {
            counters: self.counters,
            accept_lens,
        })
    }
}

impl SpeculativeStats {
    /// Host time from lane submission to first commitment, before publication.
    /// No draft proposal or visible-text event starts this metric. Preparation
    /// and time queued before submission are included in the terminal output's
    /// [`crate::GenerationTiming`] instead.
    pub const fn submission_to_first_token(&self) -> Option<Duration> {
        self.counters.submission_to_first_token
    }

    /// Selected target/draft placement relationship.
    pub const fn execution_topology(&self) -> SpeculativeExecutionTopology {
        self.counters.execution_topology
    }
    /// Target tokens evaluated.
    pub const fn target_tokens(&self) -> usize {
        self.counters.target_tokens
    }
    /// Assistant tokens proposed.
    pub const fn draft_tokens(&self) -> usize {
        self.counters.draft_tokens
    }
    /// Assistant tokens accepted.
    pub const fn accepted_tokens(&self) -> usize {
        self.counters.accepted_tokens
    }
    /// Target verification rounds.
    pub const fn rounds(&self) -> usize {
        self.counters.rounds
    }
    /// Emitted token count.
    pub const fn emitted_tokens(&self) -> usize {
        self.counters.emitted_tokens
    }
    /// Optimistically drafted token count.
    pub const fn optimistic_draft_tokens(&self) -> usize {
        self.counters.optimistic_draft_tokens
    }
    /// Optimistic block count.
    pub const fn optimistic_draft_blocks(&self) -> usize {
        self.counters.optimistic_draft_blocks
    }
    /// Reused optimistic token count.
    pub const fn reused_optimistic_tokens(&self) -> usize {
        self.counters.reused_optimistic_tokens
    }
    /// Reused optimistic block count.
    pub const fn reused_optimistic_blocks(&self) -> usize {
        self.counters.reused_optimistic_blocks
    }
    /// Optimistic tokens consumed by target bonuses.
    pub const fn consumed_optimistic_tokens(&self) -> usize {
        self.counters.consumed_optimistic_tokens
    }
    /// Discarded optimistic token count.
    pub const fn discarded_optimistic_tokens(&self) -> usize {
        self.counters.discarded_optimistic_tokens
    }
    /// Discarded optimistic block count.
    pub const fn discarded_optimistic_blocks(&self) -> usize {
        self.counters.discarded_optimistic_blocks
    }
    /// Target bonuses emitted while an optimistic branch existed.
    pub const fn optimistic_target_bonus_tokens(&self) -> usize {
        self.counters.optimistic_target_bonus_tokens
    }
    /// Matching optimistic bonus count.
    pub const fn optimistic_bonus_matches(&self) -> usize {
        self.counters.optimistic_bonus_matches
    }
    /// Mismatching optimistic bonus count.
    pub const fn optimistic_bonus_mismatches(&self) -> usize {
        self.counters.optimistic_bonus_mismatches
    }
    /// Whether adaptive lookahead is disabled.
    pub const fn adaptive_lookahead_disabled(&self) -> bool {
        self.counters.adaptive_lookahead_disabled
    }
    /// Time spent drafting optimistic branches.
    pub const fn optimistic_draft_time(&self) -> Duration {
        self.counters.optimistic_draft_time
    }
    /// Time retained verification remained in flight.
    pub const fn verification_in_flight_time(&self) -> Duration {
        self.counters.verification_in_flight_time
    }
    /// Whether component timings were collected.
    pub const fn component_timings_collected(&self) -> bool {
        self.counters.component_timings_collected
    }
    /// Draft-context device time.
    pub const fn draft_context_time(&self) -> Duration {
        self.counters.draft_context_time
    }
    /// Draft-assistant device time.
    pub const fn draft_assistant_time(&self) -> Duration {
        self.counters.draft_assistant_time
    }
    /// Draft-head device time.
    pub const fn draft_head_time(&self) -> Duration {
        self.counters.draft_head_time
    }
    /// Target-verification device time.
    pub const fn target_verification_time(&self) -> Duration {
        self.counters.target_verification_time
    }
    /// Scheduler turns for this request.
    pub const fn scheduler_turns(&self) -> usize {
        self.counters.scheduler_turns
    }
    /// Draft turns performed beside other in-flight target work.
    pub const fn cross_request_draft_opportunities(&self) -> usize {
        self.counters.cross_request_draft_opportunities
    }
    /// Wall-clock generation duration.
    pub const fn elapsed(&self) -> Duration {
        self.counters.elapsed
    }

    /// Adds backend-measured component timings without exposing mutable fields.
    pub fn add_component_timings(
        &mut self,
        draft_context: Duration,
        draft_assistant: Duration,
        draft_head: Duration,
        target_verification: Duration,
    ) {
        self.counters.draft_context_time += draft_context;
        self.counters.draft_assistant_time += draft_assistant;
        self.counters.draft_head_time += draft_head;
        self.counters.target_verification_time += target_verification;
        self.counters.component_timings_collected = true;
    }

    /// Adds completed scheduler rounds to portable telemetry.
    pub fn add_scheduler_rounds(&mut self, rounds: usize) {
        self.counters.rounds += rounds;
    }

    /// Records aggregate optimistic work used by adaptive-lookahead policy.
    pub fn record_optimistic_accounting(
        &mut self,
        drafted_blocks: usize,
        reused_tokens: usize,
        discarded_tokens: usize,
    ) {
        self.counters.optimistic_draft_blocks += drafted_blocks;
        self.counters.reused_optimistic_tokens += reused_tokens;
        self.counters.discarded_optimistic_tokens += discarded_tokens;
    }

    /// Clears the cached adaptive-lookahead decision before policy re-evaluation.
    pub fn reset_adaptive_lookahead_decision(&mut self) {
        self.counters.adaptive_lookahead_disabled = false;
    }

    /// Fraction of proposed tokens accepted by the target.
    pub fn accept_rate(&self) -> f64 {
        if self.counters.draft_tokens == 0 {
            0.0
        } else {
            self.counters.accepted_tokens as f64 / self.counters.draft_tokens as f64
        }
    }

    /// Re-evaluates whether optional lookahead remains profitable.
    pub fn update_adaptive_lookahead(&mut self, options: SpeculativeSchedulerOptions) {
        if !options.adaptive_lookahead
            || self.counters.adaptive_lookahead_disabled
            || self.counters.optimistic_draft_blocks < options.adaptive_lookahead_min_blocks
        {
            return;
        }
        self.counters.adaptive_lookahead_disabled = self.counters.reused_optimistic_tokens == 0
            || self.counters.reused_optimistic_tokens < self.counters.discarded_optimistic_tokens;
    }
}

impl SpeculativeStatsCounters {
    /// Completed verification rounds.
    pub const fn rounds(&self) -> usize {
        self.rounds
    }
    /// Accepted proposal tokens.
    pub const fn accepted_tokens(&self) -> usize {
        self.accepted_tokens
    }
    /// Reused optimistic tokens.
    pub const fn reused_optimistic_tokens(&self) -> usize {
        self.reused_optimistic_tokens
    }
    /// Optimistic tokens consumed as a bonus.
    pub const fn consumed_optimistic_tokens(&self) -> usize {
        self.consumed_optimistic_tokens
    }
    /// Discarded optimistic tokens.
    pub const fn discarded_optimistic_tokens(&self) -> usize {
        self.discarded_optimistic_tokens
    }
}

/// Aggregate bounded-scheduler telemetry.
#[derive(Debug, Clone, Default)]
pub struct SpeculativeSchedulerStats {
    /// Relationship between scheduler target and draft placements.
    execution_topology: SpeculativeExecutionTopology,
    /// Total scheduler operations.
    turns: usize,
    /// Draft turns performed while another request was being verified.
    cross_request_draft_opportunities: usize,
    /// Maximum simultaneously retained target verification transactions.
    peak_in_flight_verifications: usize,
    /// Maximum simultaneously retained optimistic draft branches.
    peak_optimistic_branches: usize,
}

impl SpeculativeSchedulerStats {
    /// Selected target/draft placement relationship.
    pub const fn execution_topology(&self) -> SpeculativeExecutionTopology {
        self.execution_topology
    }
    /// Scheduler turn count.
    pub const fn turns(&self) -> usize {
        self.turns
    }
    /// Draft turns performed beside other in-flight target work.
    pub const fn cross_request_draft_opportunities(&self) -> usize {
        self.cross_request_draft_opportunities
    }
    /// Peak retained target verifications.
    pub const fn peak_in_flight_verifications(&self) -> usize {
        self.peak_in_flight_verifications
    }
    /// Peak retained optimistic branches.
    pub const fn peak_optimistic_branches(&self) -> usize {
        self.peak_optimistic_branches
    }
}

/// Backend telemetry that can contribute to portable speculative statistics.
///
/// Implementations translate backend-specific measurements into the stable
/// semantic counters and durations owned by [`SpeculativeStats`].
pub trait SpeculativeTelemetry: Default {
    /// Records one completed backend observation.
    fn record(self, stats: &mut SpeculativeStats);
}

impl SpeculativeTelemetry for () {
    fn record(self, _stats: &mut SpeculativeStats) {}
}

/// Result of synchronous speculative prefill at a safe cache boundary.
/// Cancellation supplies no logits or seed and must precede sampling. A backend
/// returns it only after its native work and distributed cancellation agree.
#[derive(Debug)]
pub enum SpeculativePrefillOutcome<T> {
    /// Prefill completed and supplies its actual selected result.
    Complete(T),
    /// Prefill stopped at a safely completed, collectively agreed boundary.
    Cancelled {
        /// Target input positions already evaluated, retained across rollback.
        /// A mechanism for one model reports that model's completed positions;
        /// a composed target/draft executor reports target work only.
        evaluated_tokens: usize,
    },
}
impl<T> SpeculativePrefillOutcome<T> {
    /// Transforms a completed payload without manufacturing one for cancellation.
    pub fn map<U>(self, map: impl FnOnce(T) -> U) -> SpeculativePrefillOutcome<U> {
        match self {
            Self::Complete(value) => SpeculativePrefillOutcome::Complete(map(value)),
            Self::Cancelled { evaluated_tokens } => {
                SpeculativePrefillOutcome::Cancelled { evaluated_tokens }
            }
        }
    }
}

/// Backend-owned first-token output and assistant seed state.
#[derive(Debug)]
pub struct SpeculativePrefill<State, Logits> {
    /// Opaque logits used by the selected backend sampler.
    logits: Logits,
    /// Backend state from which the first proposal round begins.
    state: State,
    /// Number of prompt tokens evaluated by the target.
    evaluated_tokens: usize,
}

impl<State, Logits> SpeculativePrefill<State, Logits> {
    /// Creates a backend-owned prefill result.
    pub const fn new(logits: Logits, state: State, evaluated_tokens: usize) -> Self {
        Self {
            logits,
            state,
            evaluated_tokens,
        }
    }

    /// Decomposes the prefill into logits, proposal seed state, and evaluated-token count.
    pub fn into_parts(self) -> (Logits, State, usize) {
        (self.logits, self.state, self.evaluated_tokens)
    }
}

/// Result of committing one exact target verification transaction.
#[derive(Debug)]
pub struct SpeculativeCommit<State> {
    /// Assistant seed state matching the committed target cache.
    state: State,
    /// Target tokens replayed while restoring the exact retained prefix.
    replayed_tokens: usize,
}

impl<State> SpeculativeCommit<State> {
    /// Creates an exact target-commit result.
    pub const fn new(state: State, replayed_tokens: usize) -> Self {
        Self {
            state,
            replayed_tokens,
        }
    }

    /// Decomposes the commit into its exact proposal seed and replay count.
    pub fn into_parts(self) -> (State, usize) {
        (self.state, self.replayed_tokens)
    }
}

/// Whole-session speculative execution contract.
///
/// Tensor values, execution queues, caches, model state, logits, native
/// completions, and errors remain opaque associated types. The contract models
/// only high-level prefill, proposal, verification, and exact commit actions;
/// it deliberately does not define primitive tensor operations.
pub trait SpeculativeExecutor {
    /// Backend-owned model input accepted by prefill submission.
    type Input;
    /// Complete backend-owned target cache.
    type Cache;
    /// Target state used to seed one proposal round.
    type TargetState;
    /// Private, discardable assistant state.
    type DraftState;

    /// Copies the pending proposal state for independent optimistic advancement.
    /// Preparation may fail; the existing pending branch remains owned and unchanged.
    /// Backends with retained state must use their actual admitted copy mechanism.
    fn copy_draft_state<'a>(
        &self,
        state: &Self::DraftState,
        context: Self::Context<'a>,
    ) -> Result<Self::DraftState, Self::Error>
    where
        Self: 'a;
    /// Exact target-cache checkpoint marker.
    type CacheCheckpoint;
    /// Retained target verification output.
    type Verification;
    /// Opaque logits consumed by the backend's sampling adapter.
    type Logits;
    /// Backend execution assignment for one operation.
    type Context<'a>: Copy;
    /// Exact completion for submitted verification work.
    type Completion: BoundedCompletion<Error = Self::Error>;
    /// Optional backend-specific component telemetry.
    type Telemetry: SpeculativeTelemetry;
    /// Structured backend error.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Selects the already retained execution assignment for one stable table ID.
    /// The outer context owns batch coordination and shared table storage. This
    /// projection must preserve that lifetime, allocate nothing, and return the
    /// same request/source assignment each time. It grants no new admission.
    /// Backends with a single assignment preserve ordinary behavior by default.
    fn request_context<'a>(
        &self,
        _request: SpeculativeRequestId,
        context: Self::Context<'a>,
    ) -> Result<Self::Context<'a>, Self::Error>
    where
        Self: 'a,
    {
        Ok(context)
    }

    /// Creates one host destination before the shared driver starts its work.
    /// Managed implementations construct fixed storage with retained original
    /// custody; ordinary implementations preserve the allocating Vec behavior.
    fn driver_buffer<T>(
        &self,
        capacity: usize,
        _context: Self::Context<'_>,
    ) -> Result<SpeculativeBuffer<T>, Self::Error> {
        Ok(SpeculativeBuffer::with_capacity(capacity))
    }

    /// Admits one actual non-vector host metadata constructor before its birth.
    /// `None` is unknown storage: managed implementations must refuse it. The
    /// ordinary default grants no managed evidence. The caller retains this
    /// authority through the concrete payload, shared shell and failure owners.
    fn driver_host_metadata(
        &self,
        _bytes: Option<usize>,
        _context: Self::Context<'_>,
    ) -> Result<crate::HostPreparationAuthority, Self::Error> {
        Ok(crate::HostPreparationAuthority::unmanaged())
    }

    /// Requested storage for the same driver destination constructor.
    fn driver_buffer_bytes<T>(&self, capacity: usize) -> Option<usize> {
        SpeculativeBuffer::<T>::retained_control_bytes(capacity)
    }
    /// Creates the exact request/snapshot identity before any prefill work.
    fn driver_identity(
        &self,
        _context: Self::Context<'_>,
    ) -> Result<SpeculativeRequestIdentity, Self::Error> {
        Ok(SpeculativeRequestIdentity::new())
    }

    /// Copies canonical sequence state before transactional work. Ordinary
    /// defaults never relabel or allocate a retained provider without its owner.
    fn copy_sequence(
        &self,
        source: SpeculativeSequenceRef<'_>,
        _context: Self::Context<'_>,
    ) -> Result<SpeculativeSequence, SpeculativeDriverError<Self::Error>> {
        source
            .copy_ordinary()
            .map_err(SpeculativeDriverError::Preparation)
    }
    /// Provider copy storage within a complete controlled snapshot estimate.
    /// This is descriptive; the actual constructor still requires fresh admission.
    fn sequence_copy_bytes(&self, source: &SpeculativeSequence) -> Option<u64> {
        match source {
            SpeculativeSequence::Ordinary(_) => source.snapshot_storage_bytes(),
            SpeculativeSequence::Retained(_) => None,
        }
    }

    /// Transfers an already retained neutral failure without allocating a new source.
    /// Success must preserve its existing kind, operation and source identity;
    /// it must not infer origin or reclassify an arbitrary error. Return
    /// `Err(error)` with the identical unmodified owner when no retained
    /// representation is available. The default preserves ordinary conversion.
    /// This hook does not allocate storage, grant admission, or settle work.
    fn take_retained_failure(error: Self::Error) -> Result<crate::BackendFailure, Self::Error> {
        Err(error)
    }

    /// Agrees a host preparation result before any participant enters the next
    /// speculative phase. Distributed executors use their retained session
    /// transport and cumulative ledger; local executors resolve their own status.
    fn agree_text_preparation<'a>(
        &mut self,
        _stage: crate::run_preparation::TextPreparationStage,
        status: crate::run_preparation::TextPreparationStatus,
        _context: Self::Context<'a>,
    ) -> Result<crate::run_preparation::TextPreparationOutcome, crate::BackendFailure> {
        use crate::run_preparation::{TextPreparationOutcome as O, TextPreparationStatus as S};
        Ok(match status {
            S::Ready => O::Ready,
            S::Cancelled => O::Cancelled,
            S::Failed => O::Rejected { rank: 0 },
        })
    }

    /// Resolves per-request cancellation and asynchronous completion facts before
    /// the portable scheduler selects an action. Distributed implementations must
    /// preserve request order and lifecycle identity, require completion and
    /// optimistic eligibility on every participant, and propagate any cancellation
    /// or expired deadline. No native model action may occur during coordination.
    fn coordinate_speculative_step<'a>(
        &mut self,
        local: Vec<SpeculativeScheduleState>,
        _context: Self::Context<'a>,
    ) -> Result<Vec<SpeculativeScheduleState>, crate::BackendFailure> {
        Ok(local)
    }

    /// Coordinates the same facts while preserving their actual host owner.
    /// Legacy adapters receive only ordinary storage; unsupported retained
    /// storage refuses before any scheduling action or ordinary conversion.
    fn coordinate_speculative_buffer<'a>(
        &mut self,
        local: SpeculativeBuffer<SpeculativeScheduleState>,
        context: Self::Context<'a>,
    ) -> Result<SpeculativeBuffer<SpeculativeScheduleState>, crate::BackendFailure> {
        match local.try_into_ordinary() {
            Ok(local) => self
                .coordinate_speculative_step(local, context)
                .map(Into::into),
            Err(_) => Err(crate::HostMetadataFundingError::Unavailable.into()),
        }
    }

    /// Whether internal activation collection needs scheduler provenance.
    /// Disabled instrumentation does not hash or retain generated prefixes.
    fn requires_activation_origin(&self) -> bool {
        false
    }

    /// Installs the origin of the next synchronous executor operation. `None`
    /// clears it on success, failure, or unwind. This must not perform native
    /// work; individual forward invocations retain their own bounded evidence.
    fn set_activation_origin(&mut self, _origin: Option<SpeculativeActivationOrigin>) {}

    /// Moves one previously charged internal activation record to its consumer.
    fn take_activation_capture(&mut self) -> Option<SpeculativeActivationCapture> {
        None
    }

    /// Drains a retained portable capture failure across the native error domain.
    fn take_activation_error(&mut self) -> Option<SpeculativeControlError> {
        None
    }

    /// Mandatory loaded-authority revalidation before internal edit replacement.
    /// Original executors may use their retained declaration instead of rebuilding
    /// admission DTOs; ordinary executors keep the same discovery validation.
    fn validate_activation_readmission(
        &self,
        plan: &AdmittedSpeculativeActivations,
        discovery: Option<&SpeculativeActivationDiscovery>,
    ) -> Result<(), SpeculativeControlError> {
        plan.validate(discovery.ok_or(SpeculativeControlError::Unsupported(
            "loaded execution has no internal activation discovery",
        ))?)
        .map_err(Into::into)
    }

    /// Replaces prospective internal edits at a drained canonical boundary.
    /// The loaded controller must validate authority against current discovery.
    fn readmit_activation_interventions(
        &mut self,
        _plan: AdmittedSpeculativeActivations,
    ) -> Result<(), SpeculativeControlError> {
        Err(SpeculativeControlError::Unsupported(
            "executor cannot replace internal interventions",
        ))
    }

    /// Installs internal capture authority for this borrowed executor scope.
    /// Implementations restore prior observer ownership when the scope ends;
    /// the next ordinary run must not inherit this plan or a fresh allowance.
    fn configure_activation_capture<'a>(
        &mut self,
        plan: AdmittedSpeculativeActivations,
        _request: SpeculativeRequestId,
        _context: Self::Context<'a>,
    ) -> Result<(), SpeculativeControlError> {
        if plan.is_empty() {
            Ok(())
        } else {
            Err(SpeculativeControlError::Unsupported(
                "executor has no scoped speculative activation collector",
            ))
        }
    }

    /// Complete bound for a durable canonical cache and assistant seed snapshot.
    /// Transaction rollback markers alone are insufficient for reusable snapshots.
    fn control_snapshot_estimate(
        &self,
        _cache: &Self::Cache,
        _state: &Self::TargetState,
    ) -> Option<crate::execution_control::SnapshotEstimate> {
        None
    }

    /// Copies an immutable, reusable checkpoint with independently restorable state.
    #[allow(clippy::type_complexity)]
    fn control_snapshot<'a>(
        &self,
        _cache: &Self::Cache,
        _state: &Self::TargetState,
        _context: Self::Context<'a>,
    ) -> Result<Option<(Self::CacheCheckpoint, Self::TargetState)>, SpeculativeControlError> {
        Ok(None)
    }

    /// Prepares fresh execution occurrences for an authenticated canonical
    /// snapshot before copying or replacing any mutable state. The lifecycle is
    /// explicit: a terminal snapshot must not regain work merely from its length.
    /// Existing copy, observation and occurrence spending is never refunded.
    fn prepare_control_continuation<'a>(
        &mut self,
        _committed: usize,
        _status: SpeculativeRequestStatus,
        _context: Self::Context<'a>,
    ) -> Result<(), SpeculativeControlError> {
        Ok(())
    }

    /// Atomically restores cache and returns an isolated assistant seed. The saved
    /// checkpoint remains reusable. Must settle copying before returning.
    fn restore_control_snapshot<'a>(
        &mut self,
        _cache: &mut Self::Cache,
        _checkpoint: &Self::CacheCheckpoint,
        _state: &Self::TargetState,
        _context: Self::Context<'a>,
    ) -> Result<Option<Self::TargetState>, SpeculativeControlError> {
        Ok(None)
    }

    /// Maximum proposals supported in one verification transaction.
    fn max_proposals(&self) -> usize {
        usize::MAX
    }

    /// Enables optional component telemetry.
    fn set_telemetry_enabled(&mut self, _enabled: bool) {}

    /// Whether optional component telemetry is available.
    fn supports_telemetry(&self) -> bool {
        false
    }

    /// Resolves and drains assistant telemetry since the previous call.
    fn take_telemetry(&mut self) -> Result<Self::Telemetry, Self::Error> {
        Ok(Self::Telemetry::default())
    }

    /// Resolves telemetry retained by one target verification output.
    fn take_verification_telemetry(
        &mut self,
        _output: &mut Self::Verification,
    ) -> Result<Self::Telemetry, Self::Error> {
        Ok(Self::Telemetry::default())
    }

    /// Whether cloned assistant state can be promoted after an exact bonus match.
    fn supports_exact_optimistic_promotion(&self) -> bool {
        false
    }

    /// Prefills the target and returns first-token logits plus assistant seed state.
    fn prefill<'context>(
        &mut self,
        input: Self::Input,
        cache: &mut Self::Cache,
        context: Self::Context<'context>,
    ) -> Result<SpeculativePrefill<Self::TargetState, Self::Logits>, Self::Error>;

    /// Cancellable prefill, with no sampling or output publication on cancellation.
    /// Successful return must permit the existing cache rollback protocol.
    ///
    /// Compatibility executors still enter their complete prefill operation:
    /// local cancellation cannot skip a collective-bearing call. The shared
    /// registration path agrees cancellation after this return, before sampling.
    /// Scheduled implementations override this to stop at agreed settled spans.
    fn prefill_cancellable<'context>(
        &mut self,
        input: Self::Input,
        cache: &mut Self::Cache,
        _cancellation: &GenerationCancellationToken,
        context: Self::Context<'context>,
    ) -> Result<
        SpeculativePrefillOutcome<SpeculativePrefill<Self::TargetState, Self::Logits>>,
        Self::Error,
    > {
        self.prefill(input, cache, context)
            .map(SpeculativePrefillOutcome::Complete)
    }

    /// Starts one private proposal round sized to the available output budget.
    fn begin_proposal<'a>(
        &mut self,
        state: &Self::TargetState,
        last_token: u32,
        proposal_capacity: usize,
        context: Self::Context<'a>,
    ) -> Result<Self::DraftState, Self::Error>;

    /// Produces opaque next-token logits and advances private assistant state.
    fn proposal_logits<'a>(
        &mut self,
        state: &mut Self::DraftState,
        last_token: u32,
        context: Self::Context<'a>,
    ) -> Result<Self::Logits, Self::Error>;

    /// Captures the exact cache boundary before target verification.
    fn checkpoint(&self, cache: &Self::Cache) -> Result<Self::CacheCheckpoint, Self::Error>;

    /// Captures the same boundary with the actual request's host-construction
    /// authority. Ordinary executors keep their existing checkpoint worker.
    fn checkpoint_with_context<'a>(
        &self,
        cache: &Self::Cache,
        _context: Self::Context<'a>,
    ) -> Result<Self::CacheCheckpoint, Self::Error> {
        self.checkpoint(cache)
    }

    /// Restores the exact cache boundary after an aborted verification transaction.
    fn restore_checkpoint<'a>(
        &mut self,
        cache: &mut Self::Cache,
        checkpoint: &Self::CacheCheckpoint,
        context: Self::Context<'a>,
    ) -> Result<(), Self::Error>;

    /// Submits verification of the last committed token and proposal block.
    ///
    /// Implementations materialize token tensors internally and return an exact
    /// completion retaining every resource required by the submission.
    fn submit_verification<'a>(
        &mut self,
        input_tokens: &[u32],
        cache: &mut Self::Cache,
        context: Self::Context<'a>,
    ) -> Result<Submission<Self::Verification, Self::Completion>, Self::Error>;

    /// Selects one prediction row from retained verification output.
    fn verification_logits<'a>(
        &self,
        output: &Self::Verification,
        index: usize,
        context: Self::Context<'a>,
    ) -> Result<Self::Logits, Self::Error>;

    /// Commits exactly the requested verified inputs and restores matching seed state.
    fn commit_verification<'a>(
        &mut self,
        output: Self::Verification,
        draft_state: Self::DraftState,
        cache: &mut Self::Cache,
        checkpoint: &Self::CacheCheckpoint,
        verified_inputs: usize,
        context: Self::Context<'a>,
    ) -> Result<SpeculativeCommit<Self::TargetState>, Self::Error>;
}

/// Target decision for one assistant proposal.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
#[non_exhaustive]
pub enum ProposalDecision {
    /// Retain the assistant proposal.
    Accept,
    /// Reject it and commit this target replacement.
    Reject(u32),
}

/// Logical model side on which an opaque sampling operation executes.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
#[non_exhaustive]
pub enum SamplingPlacement {
    /// Canonical target-model execution.
    Target,
    /// Tentative assistant-model execution.
    Draft,
}

/// Absolute committed-output coordinate for position-stable draft randomness.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct SpeculativeDraftRandomPosition(usize);

impl SpeculativeDraftRandomPosition {
    /// Creates an absolute draft-randomness coordinate.
    pub const fn new(position: usize) -> Self {
        Self(position)
    }

    /// Returns the zero-based absolute output position.
    pub const fn get(self) -> usize {
        self.0
    }
}

/// Backend-owned random streams for canonical and position-stable sampling.
#[derive(Debug, Clone)]
pub struct SpeculativeRandomness<R, D> {
    /// Sequential target randomness.
    target: Option<R>,
    /// Position-addressable assistant randomness.
    draft: Option<D>,
}

impl<R, D> SpeculativeRandomness<R, D> {
    /// Creates independent target and draft random streams.
    pub const fn new(target: Option<R>, draft: Option<D>) -> Self {
        Self { target, draft }
    }
}

/// One backend-prepared lane lent to neutral speculative orchestration.
///
/// The lane contains opaque execution values but no scheduler or lifecycle
/// policy. Its cache borrow remains valid only for the visitor invocation.
pub struct PreparedSpeculativeLane<'a, E, S, C, P>
where
    E: SpeculativeExecutor,
    S: SpeculativeSampling<Logits = E::Logits, Error = E::Error>,
    C: SpeculativeConstraint,
    P: SpeculativePublisher<C>,
{
    /// Backend-owned request cache.
    cache: Option<&'a mut E::Cache>,
    /// Backend-owned prepared model input.
    input: Option<E::Input>,
    /// Validated speculative generation controls.
    config: Option<SpeculativeConfiguration>,
    /// Canonical sampling, constraint, publication, and cancellation state.
    runtime: Option<SpeculativeOutputRuntime<S, C, P>>,
    /// Independent target and draft random streams.
    randomness: Option<SpeculativeRandomness<S::RandomState, S::DraftRandomness>>,
}

impl<'a, E, S, C, P> PreparedSpeculativeLane<'a, E, S, C, P>
where
    E: SpeculativeExecutor,
    S: SpeculativeSampling<Logits = E::Logits, Error = E::Error>,
    C: SpeculativeConstraint,
    P: SpeculativePublisher<C>,
{
    /// Creates one backend-prepared lane for neutral orchestration.
    pub fn new(
        cache: &'a mut E::Cache,
        input: E::Input,
        config: impl Into<SpeculativeConfiguration>,
        runtime: SpeculativeOutputRuntime<S, C, P>,
        randomness: SpeculativeRandomness<S::RandomState, S::DraftRandomness>,
    ) -> Self {
        Self {
            cache: Some(cache),
            input: Some(input),
            config: Some(config.into()),
            runtime: Some(runtime),
            randomness: Some(randomness),
        }
    }
    /// Takes the backend cache borrow exactly once.
    pub fn take_cache(&mut self) -> &'a mut E::Cache {
        self.cache.take().expect("prepared cache already taken")
    }
    /// Takes model input exactly once.
    pub fn take_input(&mut self) -> E::Input {
        self.input.take().expect("prepared input already taken")
    }
    /// Takes speculative controls exactly once.
    pub fn take_config(&mut self) -> SpeculativeConfiguration {
        self.config.take().expect("prepared config already taken")
    }
    /// Borrows the actually selected finite output allowance before installation.
    pub fn config(&self) -> &SpeculativeConfig {
        self.config.as_ref().expect("prepared config already taken")
    }
    /// Borrows canonical output controls before submitting a prepared lane.
    pub fn runtime_mut(&mut self) -> &mut SpeculativeOutputRuntime<S, C, P> {
        self.runtime.as_mut().expect("lane runtime already taken")
    }

    /// Takes portable output state exactly once.
    pub fn take_runtime(&mut self) -> SpeculativeOutputRuntime<S, C, P> {
        self.runtime.take().expect("prepared runtime already taken")
    }
    /// Takes target/draft randomness exactly once.
    pub fn take_randomness(&mut self) -> SpeculativeRandomness<S::RandomState, S::DraftRandomness> {
        self.randomness
            .take()
            .expect("prepared randomness already taken")
    }
}

/// Facade/runtime-owned driver for backend-prepared speculative execution.
///
/// The generic method lets a backend lend any concrete executor realization
/// without erasing native tensor, cache, completion, or sampling types. The
/// visitor owns request registration, fair action selection, completion
/// driving, terminal validation, and public output construction.
pub trait SpeculativeGenerationVisitor {
    /// Actual retained scheduler policy used by this visitor. Cold admission
    /// consumes these options before preparing native sources or lane state.
    /// A custom lifecycle with no fixed policy leaves its bound unknown.
    fn scheduler_options(&self) -> Option<SpeculativeSchedulerOptions> {
        None
    }

    /// Drives one prepared set of lanes through the neutral lifecycle.
    #[allow(clippy::too_many_arguments)]
    fn run<'a, E, S, C, P>(
        self,
        executor: &'a mut E,
        lanes: impl Into<SpeculativeBuffer<PreparedSpeculativeLane<'a, E, S, C, P>>>,
        topology: SpeculativeExecutionTopology,
        optimistic_execution_available: bool,
        component_timings_collected: bool,
        context: E::Context<'a>,
    ) -> Result<SpeculativeGenerationBatchOutput, SpeculativeDriverError<E::Error>>
    where
        E: SpeculativeExecutor + 'a,
        S: SpeculativeSampling<Logits = E::Logits, Error = E::Error, Context<'a> = E::Context<'a>>
            + 'a,
        C: SpeculativeConstraint,
        P: SpeculativePublisher<C>;
}

/// Backend mechanisms used by core-owned speculative sampling policy.
///
/// Implementations keep logits, distributions, and random state opaque while
/// exposing only the probability and RNG operations needed by the portable
/// accept-or-replace algorithm. Core owns when those mechanisms run, which
/// token is selected, and when tentative sampler state is promoted.
pub trait SpeculativeSampling: Clone {
    /// Revalidate borrowed admitted edits against a retained loaded declaration
    /// when no ordinary discovery DTO is installed. This grants no execution.
    fn validate_control_interventions_prepared<'a>(
        &self,
        plans: &[SpeculativeInterventionPlan],
        _context: Self::Context<'a>,
    ) -> Result<(), SpeculativeControlError>
    where
        Self: 'a,
    {
        if plans.is_empty() {
            Ok(())
        } else {
            Err(SpeculativeControlError::Unsupported(
                "loaded execution has no speculative intervention discovery",
            ))
        }
    }
    /// Context-bearing installation boundary for a freshly paid immutable edit
    /// source. Ordinary samplers preserve their existing replacement semantics;
    /// original samplers must construct their own source/destination custody.
    fn control_intervene_prepared<'a>(
        &mut self,
        plans: Vec<SpeculativeInterventionPlan>,
        _context: Self::Context<'a>,
    ) -> Result<(), SpeculativeControlError>
    where
        Self: 'a,
    {
        self.control_intervene(plans)
    }
    /// Installs immutable role-specific plans, preserving the shared capture ledger.
    fn control_intervene(
        &mut self,
        plans: Vec<SpeculativeInterventionPlan>,
    ) -> Result<(), SpeculativeControlError> {
        if plans.is_empty() {
            Ok(())
        } else {
            Err(SpeculativeControlError::Unsupported(
                "sampler has no speculative tensor interventions",
            ))
        }
    }
    /// Whether prospective temperature changes are supported and require a positive value.
    fn control_requires_positive_temperature(&self) -> Option<bool> {
        None
    }
    /// Creates an explicit seed without drawing from the current streams.
    fn control_seed<'a>(
        _seed: u64,
        _context: Self::Context<'a>,
    ) -> Result<Self::Seed, SpeculativeControlError>
    where
        Self: 'a,
    {
        Err(SpeculativeControlError::Unsupported(
            "sampler has no explicit reseeding",
        ))
    }
    /// Stages a single canonical decision; tentative clones retain its absolute position.
    fn control_force_next(
        &mut self,
        _token: u32,
        _vocabulary: usize,
        _position: usize,
    ) -> Result<(), SpeculativeControlError> {
        Err(SpeculativeControlError::Unsupported(
            "sampler has no token forcing",
        ))
    }
    /// Clears an uncommitted choice without advancing state.
    fn control_clear_forced(&mut self) -> bool {
        false
    }
    /// Clears a choice with explicit preparation failure. The default preserves
    /// the existing infallible ordinary callback; retained implementations may
    /// prepare an isolated replacement before changing their source owner.
    fn try_control_clear_forced(&mut self) -> Result<bool, SpeculativeControlError> {
        Ok(self.control_clear_forced())
    }
    /// Canonical choice waiting for target commitment.
    fn control_pending_forced(&self) -> Option<u32> {
        None
    }
    /// Raw model logits.
    type Logits;
    /// Processed distribution retained for verification.
    type Distribution;
    /// Caller-provided randomness seed.
    type Seed;
    /// Sequential random state.
    type RandomState: Clone;
    /// Position-addressable assistant random state.
    type DraftRandomness: Clone;
    /// Backend-native root from which core allocates canonical substreams.
    type RandomnessRoot;
    /// Backend execution assignment.
    type Context<'a>: Copy
    where
        Self: 'a;
    /// Structured backend error.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Transfers an already retained neutral failure without allocating a new source.
    /// Success must preserve its existing kind, operation and source identity;
    /// it must not infer origin or reclassify an arbitrary error. Return
    /// `Err(error)` with the identical unmodified owner when no retained
    /// representation is available. The default preserves ordinary conversion.
    /// This hook does not allocate storage, grant admission, or settle work.
    fn take_retained_failure(error: Self::Error) -> Result<crate::BackendFailure, Self::Error> {
        Err(error)
    }

    /// Installs explicitly admitted raw-logit capture before any generation. Clones
    /// must share non-rewindable capture accounting, never duplicate its allowance.
    fn enable_control_capture(
        &mut self,
        plan: crate::capture::AdmittedCapturePlan,
    ) -> Result<(), crate::capture::CaptureError> {
        if plan.plan().selections.is_empty() {
            Ok(())
        } else {
            Err(crate::capture::CaptureError::Unsupported(
                "speculative raw-logit capture is unavailable for this sampler".into(),
            ))
        }
    }

    /// Context-bearing preparation of the same raw-logit observer. Original
    /// implementations construct their source/destinations before execution;
    /// ordinary implementations retain the existing installation behavior.
    fn enable_control_capture_prepared<'a>(
        &mut self,
        plan: crate::capture::AdmittedCapturePlan,
        _context: Self::Context<'a>,
    ) -> Result<(), SpeculativeControlError>
    where
        Self: 'a,
    {
        self.enable_control_capture(plan).map_err(Into::into)
    }

    /// Drains completed captures from the last scheduler action, including
    /// tentative target rows and failed draft proposals. No native values escape.
    /// Original producers return their freshly admitted buffer and shared frame
    /// owners; ordinary Vec conversion remains ordinary and gains no custody.
    fn take_control_captures(&mut self) -> SpeculativeBuffer<SpeculativePredictionCapture> {
        SpeculativeBuffer::default()
    }

    /// Complete conservative bytes for cloning this sampler and both RNG streams.
    /// Return unknown unless clones isolate mutable native state.
    fn control_snapshot_bytes(
        &self,
        _target: Option<&Self::RandomState>,
        _draft: Option<&Self::DraftRandomness>,
    ) -> Option<u64> {
        None
    }

    /// Exact host allocations and controls made by `copy_control_snapshot` for
    /// these actual source values. This is separate from logical snapshot size;
    /// it grants no native copy/completion authority. Unknown state stays unknown.
    fn control_snapshot_metadata_bytes(
        &self,
        _target: Option<&Self::RandomState>,
        _draft: Option<&Self::DraftRandomness>,
    ) -> Option<usize> {
        None
    }

    /// Copies the sampler and RNG state after the actual executor pays the query
    /// above. A qualified implementation must preserve capture/copy ledgers and
    /// retain the supplied host custody in every independently escaping payload,
    /// clone and allocating failure. Retaining it only in an enclosing snapshot
    /// is insufficient. Native work needs its own admitted copy mechanism.
    /// The ordinary default preserves Clone; retained unknown state is rejected
    /// before any clone is invoked.
    #[allow(clippy::type_complexity)]
    fn copy_control_snapshot(
        &self,
        target: Option<&Self::RandomState>,
        draft: Option<&Self::DraftRandomness>,
        host: crate::HostPreparationAuthority,
    ) -> Result<
        (
            Self,
            Option<Self::RandomState>,
            Option<Self::DraftRandomness>,
        ),
        SpeculativeControlError,
    > {
        if !host.is_unmanaged() {
            return Err(SpeculativeControlError::Unsupported(
                "sampler has no paid isolated snapshot constructor",
            ));
        }
        Ok((self.clone(), target.cloned(), draft.cloned()))
    }

    /// Whether cloned sampler state is safe for optimistic promotion.
    fn supports_exact_optimistic_promotion(&self) -> bool {
        false
    }

    /// Whether the canonical grammar accepts its current prefix.
    fn grammar_is_complete(&mut self) -> Result<bool, Self::Error> {
        Ok(false)
    }

    /// Whether a tentative token history completes the grammar.
    fn prefix_is_complete(&self, _history: &[u32]) -> Result<bool, Self::Error> {
        Ok(false)
    }

    /// Creates one opaque native root from the caller-provided seed.
    fn randomness_root<'a>(
        seed: Option<Self::Seed>,
        context: Self::Context<'a>,
    ) -> Result<Self::RandomnessRoot, Self::Error>
    where
        Self: 'a;

    /// Splits the next canonical target substream from a native root.
    fn target_randomness_from_root<'a>(
        root: &mut Self::RandomnessRoot,
        context: Self::Context<'a>,
    ) -> Result<Self::RandomState, Self::Error>
    where
        Self: 'a;

    /// Splits the next position-addressable draft root from a native root.
    fn draft_randomness_from_root<'a>(
        root: &mut Self::RandomnessRoot,
        context: Self::Context<'a>,
    ) -> Result<Self::DraftRandomness, Self::Error>
    where
        Self: 'a;

    /// Allocates target then draft randomness in the canonical neutral order.
    fn initialize_randomness<'a>(
        seed: Option<Self::Seed>,
        temperature: f32,
        context: Self::Context<'a>,
    ) -> Result<SpeculativeRandomness<Self::RandomState, Self::DraftRandomness>, Self::Error>
    where
        Self: 'a,
    {
        if temperature == 0.0 {
            return Ok(SpeculativeRandomness::new(None, None));
        }
        let mut root = Self::randomness_root(seed, context)?;
        let target = Self::target_randomness_from_root(&mut root, context)?;
        let draft = Self::draft_randomness_from_root(&mut root, context)?;
        Ok(SpeculativeRandomness::new(Some(target), Some(draft)))
    }

    /// Derives assistant randomness for one absolute output position.
    fn draft_randomness_at<'a>(
        root: &Self::DraftRandomness,
        position: SpeculativeDraftRandomPosition,
        context: Self::Context<'a>,
    ) -> Result<Self::RandomState, Self::Error>
    where
        Self: 'a;

    /// Processes raw logits against one logical history.
    fn process_logits<'a>(
        &mut self,
        logits: &Self::Logits,
        temperature: f32,
        history: &[u32],
        placement: SamplingPlacement,
        context: Self::Context<'a>,
    ) -> Result<Self::Distribution, Self::Error>
    where
        Self: 'a;

    /// Samples one token from a processed distribution.
    fn sample<'a>(
        &self,
        distribution: &Self::Distribution,
        temperature: f32,
        randomness: Option<&mut Self::RandomState>,
        placement: SamplingPlacement,
        context: Self::Context<'a>,
    ) -> Result<u32, Self::Error>
    where
        Self: 'a;

    /// Returns the normalized probability assigned to one token.
    fn probability_at<'a>(
        &self,
        distribution: &Self::Distribution,
        token: u32,
        placement: SamplingPlacement,
        context: Self::Context<'a>,
    ) -> Result<f32, Self::Error>
    where
        Self: 'a;

    /// Draws one value from the half-open unit interval.
    fn sample_unit_interval<'a>(
        &self,
        randomness: Option<&mut Self::RandomState>,
        context: Self::Context<'a>,
    ) -> Result<f32, Self::Error>
    where
        Self: 'a;

    /// Computes the normalized positive probability difference `left-right`.
    ///
    /// `None` means that the positive difference has no usable mass. Core
    /// owns the fallback to the target distribution in that case.
    fn positive_probability_difference<'a>(
        &self,
        left: &Self::Distribution,
        right: &Self::Distribution,
        placement: SamplingPlacement,
        context: Self::Context<'a>,
    ) -> Result<Option<Self::Distribution>, Self::Error>
    where
        Self: 'a;

    /// Applies a mechanism-specific sampler update for a core-selected token.
    fn update_sampler_state<'a>(
        &mut self,
        distribution: &Self::Distribution,
        token: u32,
        placement: SamplingPlacement,
        context: Self::Context<'a>,
    ) -> Result<(), Self::Error>
    where
        Self: 'a;

    /// Makes retained assistant distributions available to target resolution.
    fn prepare_verification<'a>(
        &self,
        _distributions: &mut [&mut Self::Distribution],
        _temperature: f32,
        _context: Self::Context<'a>,
    ) -> Result<(), Self::Error>
    where
        Self: 'a,
    {
        Ok(())
    }
}

/// Computes the portable speculative acceptance probability.
///
/// A proposal with no draft probability is accepted because the assistant
/// could not have sampled it from the represented distribution. Otherwise
/// acceptance is capped at one exactly as required by speculative decoding.
pub fn speculative_acceptance_probability(target_probability: f32, draft_probability: f32) -> f32 {
    if draft_probability <= 0.0 {
        1.0
    } else {
        (target_probability / draft_probability).min(1.0)
    }
}

/// Applies the core-owned accept-or-replace policy to one proposal.
pub fn decide_speculative_proposal<'a, S>(
    sampler: &S,
    target: &S::Distribution,
    draft: &S::Distribution,
    proposed: u32,
    temperature: f32,
    randomness: Option<&mut S::RandomState>,
    context: S::Context<'a>,
) -> Result<ProposalDecision, S::Error>
where
    S: SpeculativeSampling + 'a,
{
    let mut randomness = randomness;
    if temperature == 0.0 {
        let chosen = sampler.sample(
            target,
            temperature,
            None,
            SamplingPlacement::Target,
            context,
        )?;
        return Ok(if chosen == proposed {
            ProposalDecision::Accept
        } else {
            ProposalDecision::Reject(chosen)
        });
    }

    let target_probability =
        sampler.probability_at(target, proposed, SamplingPlacement::Target, context)?;
    let draft_probability =
        sampler.probability_at(draft, proposed, SamplingPlacement::Target, context)?;
    let acceptance = speculative_acceptance_probability(target_probability, draft_probability);
    if sampler.sample_unit_interval(randomness.as_deref_mut(), context)? <= acceptance {
        return Ok(ProposalDecision::Accept);
    }

    let residual = sampler.positive_probability_difference(
        target,
        draft,
        SamplingPlacement::Target,
        context,
    )?;
    let replacement = sampler.sample(
        residual.as_ref().unwrap_or(target),
        temperature,
        randomness,
        SamplingPlacement::Target,
        context,
    )?;
    Ok(ProposalDecision::Reject(replacement))
}

/// One sampled assistant proposal and its retained distribution.
#[derive(Debug)]
pub struct SpeculativeProposal<D> {
    /// Proposed token id.
    token: u32,
    /// Backend-owned processed assistant distribution.
    distribution: D,
}

impl<D> SpeculativeProposal<D> {
    /// Creates one retained assistant proposal.
    pub const fn new(token: u32, distribution: D) -> Self {
        Self {
            token,
            distribution,
        }
    }
    /// Proposed token id.
    pub const fn token(&self) -> u32 {
        self.token
    }
    /// Retained assistant distribution.
    pub const fn distribution(&self) -> &D {
        &self.distribution
    }
}

/// Backend-owned assistant state paired with a portable proposal sequence.
pub struct SpeculativeDraftBlock<S, D> {
    /// Assistant state after producing every proposal.
    state: S,
    /// Ordered proposed tokens and opaque distributions.
    proposals: SpeculativeBuffer<SpeculativeProposal<D>>,
}

impl<S, D> SpeculativeDraftBlock<S, D> {
    /// Creates one ordered assistant proposal block.
    pub fn new(state: S, proposals: impl Into<SpeculativeBuffer<SpeculativeProposal<D>>>) -> Self {
        Self {
            state,
            proposals: proposals.into(),
        }
    }
    /// Assistant state after every proposal.
    pub const fn state(&self) -> &S {
        &self.state
    }
    /// Ordered proposals retained by this block.
    pub fn proposals(&self) -> &[SpeculativeProposal<D>] {
        &self.proposals
    }
}

/// Tentative continuation drafted against an assumed canonical prefix.
pub struct SpeculativeOptimisticBranch<S, D> {
    /// Backend-owned tentative draft block.
    block: SpeculativeDraftBlock<S, D>,
    /// Prefix against which the block was produced.
    assumed_prefix: SpeculativeBuffer<u32>,
}

impl<S, D> SpeculativeOptimisticBranch<S, D> {
    /// Creates one tentative continuation tied to an assumed prefix.
    pub fn new(
        block: SpeculativeDraftBlock<S, D>,
        assumed_prefix: impl Into<SpeculativeBuffer<u32>>,
    ) -> Self {
        Self {
            block,
            assumed_prefix: assumed_prefix.into(),
        }
    }
}

/// Optimistic state retained after a committed target transaction.
#[non_exhaustive]
pub enum SpeculativeContinuation<S, D> {
    /// No reusable proposal block remains.
    None,
    /// A matching branch may seed the next canonical round.
    Promoted(SpeculativeDraftBlock<S, D>),
}

impl<S, D> SpeculativeContinuation<S, D> {
    /// Returns the promoted block, when one exists.
    pub fn into_block(self) -> Option<SpeculativeDraftBlock<S, D>> {
        match self {
            Self::None => None,
            Self::Promoted(block) => Some(block),
        }
    }
}

/// Exact target verification resources retained through resolution.
///
/// Completion is declared first so it is dropped before the output and every
/// resource reachable from it. Its destructor must preserve exact-completion
/// safety when the scheduler itself is abandoned.
pub struct PendingSpeculativeVerification<E, D>
where
    E: SpeculativeExecutor,
{
    completion: E::Completion,
    verification: E::Verification,
    checkpoint: E::CacheCheckpoint,
    block: SpeculativeDraftBlock<E::DraftState, D>,
    optimistic: Option<SpeculativeOptimisticBranch<E::DraftState, D>>,
    submitted: Instant,
    submitted_tokens: usize,
    // Prepared before submission; retained through completion and restoration.
    rollback_failure: rollback::PreparedRollback<SpeculativeDriverError<E::Error>, E::Error>,
}

impl<E, D> PendingSpeculativeVerification<E, D>
where
    E: SpeculativeExecutor,
{
    /// Observes exact completion without consuming or waiting on retained resources.
    pub fn is_complete(&self) -> Result<bool, E::Error> {
        self.completion.is_complete()
    }

    /// Canonical block being verified.
    pub const fn block(&self) -> &SpeculativeDraftBlock<E::DraftState, D> {
        &self.block
    }

    /// Whether one optimistic continuation is retained.
    pub const fn has_optimistic_branch(&self) -> bool {
        self.optimistic.is_some()
    }

    /// Installs exactly one tentative optimistic branch.
    pub fn set_optimistic_branch(
        &mut self,
        branch: SpeculativeOptimisticBranch<E::DraftState, D>,
    ) -> Result<(), GenerationError> {
        if self.optimistic.is_some() {
            return Err(GenerationError::OptimisticBranchAlreadyPresent);
        }
        self.optimistic = Some(branch);
        Ok(())
    }

    /// Number of target tokens submitted for verification.
    pub const fn submitted_tokens(&self) -> usize {
        self.submitted_tokens
    }

    /// Time elapsed since target submission.
    pub fn elapsed(&self) -> Duration {
        self.submitted.elapsed()
    }
}

/// Structured failure in backend-independent speculative output handling.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum SpeculativeOutputError {
    /// Paid shared diagnostic retaining the actual portable producer source,
    /// partial output/input and its funding. Aliases allocate no new source.
    #[error("{0}")]
    Retained(#[from] crate::SharedBackendFailure),
    /// Actual fixed host-account refusal before a semantic allocation.
    #[error("{0}")]
    HostFunding(#[from] crate::HostMetadataFundingError),
    /// Original decoder/stop transition failed without allocating a diagnostic.
    #[error("{0}")]
    Decoder(#[from] crate::GenerationDecoderError),
    /// A qualified semantic destination or source refused this operation.
    #[error("speculative semantic storage: {0}")]
    Storage(&'static str),
    /// Transactional semantic parsing, decoding, or stop matching failed.
    #[error("speculative semantic state failed during {operation}: {message}")]
    Semantic {
        /// Logical semantic operation.
        operation: String,
        /// Portable diagnostic detail.
        message: String,
    },
    /// A committed-token callback rejected publication.
    #[error("speculative output publication failed: {message}")]
    Publication {
        /// Portable diagnostic detail.
        message: String,
    },
}

/// Coarse production lifecycle boundaries for speculative work.
///
/// These boundaries complement typed activation observation. They make the
/// ordering of admission, native work, completion, cache persistence, and
/// publication observable without exposing backend tensors or queues.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum SpeculativeLifecycleStage {
    /// Neutral request admission, before compatibility selection.
    Admission,
    /// Compatibility selection succeeded, before native construction.
    Compatibility,
    /// Prepared input is about to be consumed.
    Input,
    /// Backend execution or construction is about to begin.
    Execution,
    /// A cross-device transfer is about to begin.
    Transfer,
    /// An exact completion is about to be inspected or waited.
    Completion,
    /// Completed model output is about to enter portable resolution.
    Observation,
    /// Backend cache state is about to be made canonical.
    CachePersistence,
    /// Committed output is about to be published.
    Publication,
    /// Cancellation state is about to be published.
    Cancellation,
}

/// Explicit, production-carried observer for coarse speculative lifecycle work.
///
/// Returning an error prevents the associated boundary from starting. The
/// observer is shared because one prepared realization can serve several
/// scheduler lanes concurrently.
pub trait SpeculativeLifecycleObserver: Send + Sync {
    /// Observes one lifecycle boundary before its associated work.
    fn observe(&self, stage: SpeculativeLifecycleStage) -> Result<(), SpeculativeOutputError>;
}

impl<F> SpeculativeLifecycleObserver for F
where
    F: Fn(SpeculativeLifecycleStage) -> Result<(), SpeculativeOutputError> + Send + Sync,
{
    fn observe(&self, stage: SpeculativeLifecycleStage) -> Result<(), SpeculativeOutputError> {
        self(stage)
    }
}

impl SpeculativeOutputError {
    /// Creates a semantic-state failure with operation context.
    pub fn semantic(operation: impl Into<String>, message: impl Into<String>) -> Self {
        Self::Semantic {
            operation: operation.into(),
            message: message.into(),
        }
    }

    /// Creates a committed-output publication failure.
    pub fn publication(message: impl Into<String>) -> Self {
        Self::Publication {
            message: message.into(),
        }
    }
}

/// Transactional semantic state paired with committed token sequencing.
pub trait SpeculativeConstraint: Sized {
    /// Complete conservative bytes retained by a fork, or unknown.
    fn control_snapshot_bytes(&self) -> Option<u64> {
        None
    }

    /// Exact host allocations and controls made by the qualified control fork.
    /// A logical snapshot estimate alone does not qualify a callback constructor.
    fn control_snapshot_metadata_bytes(&self) -> Option<usize> {
        None
    }

    /// Forks a controlled snapshot after the actual executor pays its host query.
    /// Qualified implementations retain custody in independently escaping state,
    /// aliases and allocating failures, without refunding observation budgets.
    /// Unknown caller state keeps its ordinary behavior only without custody.
    fn fork_control_snapshot(
        &self,
        host: crate::HostPreparationAuthority,
    ) -> Result<Self, SpeculativeControlError> {
        if !host.is_unmanaged() {
            return Err(SpeculativeControlError::Unsupported(
                "constraint has no paid isolated snapshot constructor",
            ));
        }
        self.fork().map_err(SpeculativeControlError::Output)
    }

    /// Forks state for tentative verification.
    fn fork(&self) -> Result<Self, SpeculativeOutputError>;
    /// Stages one token and reports a matched stop condition.
    fn push_token(&mut self, token: u32) -> Result<bool, SpeculativeOutputError>;
    /// Stages terminal output.
    fn finish(&mut self, reason: FinishReason) -> Result<(), SpeculativeOutputError>;
}

/// Backend adapter that publishes committed output and terminal cancellation.
///
/// The adapter may own callbacks and decoded semantic-event buffers, but core
/// decides when publication is legal relative to exact cache commit.
pub trait SpeculativePublisher<C> {
    /// Publishes tokens and staged semantic output after cache commit.
    ///
    /// Returns `true` when cancellation won this boundary. A successful call
    /// is one atomic publication boundary; implementations must stage any
    /// already-pending cancellation before exposing tokens or semantic events.
    fn publish_committed(
        &mut self,
        constraint: &mut C,
        tokens: &[u32],
        cancellation: &GenerationCancellationToken,
        sequence_finished: bool,
    ) -> Result<bool, SpeculativeOutputError>;

    /// Publishes the cancellation terminal state.
    fn publish_cancelled(&mut self, constraint: &mut C) -> Result<(), SpeculativeOutputError>;
}

/// Optional transactional semantic state shared by plain and structured speculative decoding.
pub struct SpeculativeSemanticConstraint {
    state: Option<SemanticStateOwner>,
}

impl SpeculativeSemanticConstraint {
    /// Creates an unconstrained output state for token-only generation.
    pub const fn plain() -> Self {
        Self { state: None }
    }

    /// Creates a transactional structured-output state.
    pub fn semantic(state: impl Into<SemanticStateOwner>) -> Self {
        Self {
            state: Some(state.into()),
        }
    }
}

impl SpeculativeConstraint for SpeculativeSemanticConstraint {
    fn control_snapshot_bytes(&self) -> Option<u64> {
        match &self.state {
            Some(state) => state.state().control_snapshot_bytes(),
            None => Some(0),
        }
    }

    fn control_snapshot_metadata_bytes(&self) -> Option<usize> {
        let outer = std::mem::size_of::<Self>()
            .checked_add(std::mem::size_of::<Result<Self, SpeculativeControlError>>())?
            .checked_add(std::mem::size_of::<crate::HostPreparationAuthority>())?;
        match &self.state {
            Some(state) => outer.checked_add(state.state().control_snapshot_metadata_bytes()?),
            None => Some(outer),
        }
    }
    fn fork_control_snapshot(
        &self,
        host: crate::HostPreparationAuthority,
    ) -> Result<Self, SpeculativeControlError> {
        if host.is_unmanaged() {
            return self.fork().map_err(SpeculativeControlError::Output);
        }
        Ok(Self {
            state: self
                .state
                .as_ref()
                .map(|state| state.fork_prepared(host))
                .transpose()
                .map_err(SpeculativeControlError::Output)?,
        })
    }

    fn fork(&self) -> Result<Self, SpeculativeOutputError> {
        Ok(Self {
            state: self.state.as_ref().map(|state| state.fork()).transpose()?,
        })
    }

    fn push_token(&mut self, token: u32) -> Result<bool, SpeculativeOutputError> {
        self.state
            .as_mut()
            .map(|state| state.state_mut().push_token(token))
            .transpose()
            .map(|matched| matched.unwrap_or(false))
    }

    fn finish(&mut self, reason: FinishReason) -> Result<(), SpeculativeOutputError> {
        if let Some(state) = &mut self.state {
            state.state_mut().finish(reason)?;
        }
        Ok(())
    }
}

/// Fallible atomic committed-token batch callback.
type SpeculativeTokenCallback<'a> = dyn FnMut(&[u32]) -> Result<(), SpeculativeOutputError> + 'a;

/// Core-owned committed-token and semantic-event publication adapter.
pub struct SpeculativeCallbackPublisher<'a> {
    on_tokens: Option<Box<SpeculativeTokenCallback<'a>>>,
    on_event: Option<SpeculativeEventCallback<'a>>,
    host: crate::HostPreparationAuthority,
}

impl<'a> SpeculativeCallbackPublisher<'a> {
    /// Publishes committed token ids without decoded semantic events.
    /// The callback is one atomic publication attempt: returning an error must
    /// leave every token in the supplied batch unpublished.
    pub fn tokens(
        on_tokens: impl FnMut(&[u32]) -> Result<(), SpeculativeOutputError> + 'a,
    ) -> Self {
        Self {
            on_tokens: Some(Box::new(on_tokens)),
            on_event: None,
            host: crate::HostPreparationAuthority::unmanaged(),
        }
    }

    /// Publishes transactional semantic events and ignores raw token callbacks.
    pub fn semantic(on_event: impl FnMut(crate::generation::SemanticEvent) + 'a) -> Self {
        Self::semantic_boxed(Box::new(on_event))
    }
    /// Moves an existing caller-owned callback; no second callback Box is born.
    pub fn semantic_boxed(on_event: Box<dyn FnMut(crate::generation::SemanticEvent) + 'a>) -> Self {
        Self::semantic_prepared_boxed(on_event, crate::HostPreparationAuthority::unmanaged())
    }
    /// Retains paid publisher controls. The supplied callback is caller-owned;
    /// this neither adopts its earlier allocation nor qualifies callback work.
    pub fn semantic_prepared_boxed(
        on_event: Box<dyn FnMut(crate::generation::SemanticEvent) + 'a>,
        host: crate::HostPreparationAuthority,
    ) -> Self {
        Self::semantic_prepared_callback(on_event.into(), host)
    }
    /// Moves the closed callback without extracting or reboxing its allocation.
    pub fn semantic_callback(on_event: SpeculativeEventCallback<'a>) -> Self {
        Self::semantic_prepared_callback(on_event, crate::HostPreparationAuthority::unmanaged())
    }
    /// Retains the separately paid publisher controls around a closed callback.
    pub fn semantic_prepared_callback(
        on_event: SpeculativeEventCallback<'a>,
        host: crate::HostPreparationAuthority,
    ) -> Self {
        Self {
            on_tokens: None,
            on_event: Some(on_event),
            host,
        }
    }
    /// Fixed handoff and callback-invocation controls, excluding caller payload.
    pub fn prepared_control_bytes() -> Option<usize> {
        let parts = [
            std::mem::size_of::<Self>(),
            std::mem::size_of::<crate::HostPreparationAuthority>(),
            std::mem::size_of::<Result<bool, SpeculativeOutputError>>(),
            std::mem::size_of::<Result<(), SpeculativeOutputError>>(),
            std::mem::size_of::<crate::generation::SemanticEvent>(),
        ];
        parts
            .into_iter()
            .try_fold(std::mem::size_of_val(&parts), usize::checked_add)
    }
}

impl SpeculativePublisher<SpeculativeSemanticConstraint> for SpeculativeCallbackPublisher<'_> {
    fn publish_committed(
        &mut self,
        constraint: &mut SpeculativeSemanticConstraint,
        tokens: &[u32],
        cancellation: &GenerationCancellationToken,
        sequence_finished: bool,
    ) -> Result<bool, SpeculativeOutputError> {
        let cancellation_won = cancellation.is_cancelled() && !sequence_finished;
        if cancellation_won {
            if let Some(state) = &mut constraint.state {
                state.state_mut().cancel()?;
            }
        }
        if let Some(on_tokens) = &mut self.on_tokens {
            on_tokens(tokens)?;
        }
        if let (Some(state), Some(on_event)) = (&mut constraint.state, &mut self.on_event) {
            state.state_mut().publish_events(on_event.callback_mut());
        }
        let cancellation_after_callbacks =
            !cancellation_won && cancellation.is_cancelled() && !sequence_finished;
        if cancellation_after_callbacks {
            if let (Some(state), Some(on_event)) = (&mut constraint.state, &mut self.on_event) {
                state.state_mut().cancel()?;
                state.state_mut().publish_events(on_event.callback_mut());
            }
        }
        Ok(cancellation_won || cancellation_after_callbacks)
    }

    fn publish_cancelled(
        &mut self,
        constraint: &mut SpeculativeSemanticConstraint,
    ) -> Result<(), SpeculativeOutputError> {
        if let (Some(state), Some(on_event)) = (&mut constraint.state, &mut self.on_event) {
            state.state_mut().cancel()?;
            state.state_mut().publish_events(on_event.callback_mut());
        }
        Ok(())
    }
}

/// Canonical speculative sampler, sequence, constraint, and output sink.
pub struct SpeculativeOutputRuntime<S, C, P> {
    sampler: S,
    sequence: SpeculativeSequence,
    constraint: C,
    publisher: P,
    cancellation: GenerationCancellationToken,
    lifecycle_observer: Option<Arc<dyn SpeculativeLifecycleObserver>>,
}

impl<S, C, P> SpeculativeOutputRuntime<S, C, P>
where
    S: SpeculativeSampling,
    C: SpeculativeConstraint,
    P: SpeculativePublisher<C>,
{
    /// Creates one canonical output runtime.
    pub fn new(
        sampler: S,
        sequence: impl Into<SpeculativeSequence>,
        constraint: C,
        publisher: P,
        cancellation: GenerationCancellationToken,
    ) -> Self {
        Self {
            sampler,
            sequence: sequence.into(),
            constraint,
            publisher,
            cancellation,
            lifecycle_observer: None,
        }
    }

    /// Installs explicit production lifecycle observation for this lane.
    pub fn with_lifecycle_observer(
        mut self,
        observer: Arc<dyn SpeculativeLifecycleObserver>,
    ) -> Self {
        self.lifecycle_observer = Some(observer);
        self
    }

    /// Observes one lifecycle boundary before its associated work.
    pub fn observe_lifecycle(
        &self,
        stage: SpeculativeLifecycleStage,
    ) -> Result<(), SpeculativeOutputError> {
        self.lifecycle_observer
            .as_ref()
            .map_or(Ok(()), |observer| observer.observe(stage))
    }

    /// Canonical sampling state.
    pub const fn sampler(&self) -> &S {
        &self.sampler
    }

    /// Mutable canonical sampling state.
    pub const fn sampler_mut(&mut self) -> &mut S {
        &mut self.sampler
    }

    /// Canonical committed sequence.
    pub const fn sequence(&self) -> &SpeculativeSequence {
        &self.sequence
    }

    /// Mutable canonical committed sequence.
    pub const fn sequence_mut(&mut self) -> &mut SpeculativeSequence {
        &mut self.sequence
    }

    /// Transactional semantic constraint.
    pub const fn constraint(&self) -> &C {
        &self.constraint
    }

    /// Mutable transactional semantic constraint.
    pub const fn constraint_mut(&mut self) -> &mut C {
        &mut self.constraint
    }

    /// Cooperative cancellation token.
    pub const fn cancellation(&self) -> &GenerationCancellationToken {
        &self.cancellation
    }

    /// Applies cancellation and publishes its terminal semantic state.
    pub fn cancel(&mut self) -> Result<(), SpeculativeOutputError> {
        if self.sequence.is_finished() {
            return Ok(());
        }
        let mut constraint = self.constraint.fork()?;
        self.observe_lifecycle(SpeculativeLifecycleStage::Cancellation)?;
        self.publisher.publish_cancelled(&mut constraint)?;
        self.constraint = constraint;
        self.sequence.cancel();
        Ok(())
    }

    /// Installs logical state only after its matching backend boundary committed.
    pub fn install_committed_state(
        &mut self,
        sampler: S,
        constraint: C,
        sequence: SpeculativeSequence,
    ) {
        self.sampler = sampler;
        self.constraint = constraint;
        self.sequence = sequence;
    }

    /// Publishes tokens only after their backend cache transaction committed.
    pub fn publish_committed(&mut self, tokens: &[u32]) -> Result<bool, SpeculativeOutputError> {
        self.observe_lifecycle(SpeculativeLifecycleStage::Publication)?;
        let cancellation_won = self.publisher.publish_committed(
            &mut self.constraint,
            tokens,
            &self.cancellation,
            self.sequence.is_finished(),
        )? || (self.cancellation.is_cancelled()
            && !self.sequence.is_finished());
        if cancellation_won {
            self.sequence.cancel();
        }
        Ok(cancellation_won)
    }

    fn publish_candidate(
        &mut self,
        constraint: &mut C,
        sequence: &mut SpeculativeSequence,
        tokens: &[u32],
    ) -> Result<bool, SpeculativeOutputError> {
        self.observe_lifecycle(SpeculativeLifecycleStage::Publication)?;
        let cancellation_won = self.publisher.publish_committed(
            constraint,
            tokens,
            &self.cancellation,
            sequence.is_finished(),
        )? || (self.cancellation.is_cancelled() && !sequence.is_finished());
        if cancellation_won {
            sequence.cancel();
        }
        Ok(cancellation_won)
    }

    fn cancel_candidate(
        &mut self,
        constraint: &mut C,
        sequence: &mut SpeculativeSequence,
    ) -> Result<(), SpeculativeOutputError> {
        if !sequence.is_finished() {
            self.observe_lifecycle(SpeculativeLifecycleStage::Cancellation)?;
        }
        if sequence.cancel() {
            self.publisher.publish_cancelled(constraint)?;
        }
        Ok(())
    }

    /// Consumes the runtime into its backend-owned parts.
    pub(crate) fn into_parts(self) -> (S, SpeculativeSequence, C, P) {
        (self.sampler, self.sequence, self.constraint, self.publisher)
    }
}

/// Error returned by portable proposal and verification drivers.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum SpeculativeDriverError<E: std::error::Error + 'static> {
    /// Backend execution or sampling failed.
    #[error("{0}")]
    Backend(#[from] E),
    /// Verification submission failed and its checkpoint could not be restored.
    /// Both concrete causes retain their independent resource custody. This
    /// outcome never certifies rollback, completion or reusable cache state.
    #[error(
        "verification submission failed: {submission}; checkpoint restoration also failed: {rollback}"
    )]
    SubmissionRollback {
        /// Original submission failure, preserved as the primary source.
        #[source]
        submission: E,
        /// Failed cleanup; retained independently of the submission source.
        rollback: E,
    },
    /// An operation failed and checkpoint restoration also failed. Both owned
    /// causes survive through a diagnostic prepared before native submission.
    #[error(transparent)]
    Rollback(crate::BackendFailure),
    /// Peer preparation rejection or failure of its retained transport.
    #[error(transparent)]
    Preparation(crate::BackendFailure),
    /// An executor returned incompatible coordinated scheduler facts.
    #[error("invalid speculative scheduler coordination: {0}")]
    Coordination(&'static str),
    /// Transactional semantic output or committed publication failed.
    #[error(transparent)]
    Output(SpeculativeOutputError),
    /// Portable lifecycle validation failed.
    #[error(transparent)]
    Generation(GenerationError),
    /// Exact verification exceeded its deadline after entering a safe disposition.
    #[error("speculative verification completion deadline exceeded ({cancellation:?})")]
    CompletionDeadline {
        /// Cancellation or retained-orphan mechanism actually applied.
        cancellation: CompletionCancellationMode,
    },
    /// The backend completion cannot honor the selected timeout disposition.
    #[error("speculative completion does not support {cancellation:?}")]
    UnsupportedCompletionCancellation {
        /// Disposition rejected before any speculative submission.
        cancellation: CompletionCancellationMode,
    },
}

/// Resolved speculative transaction ready for backend cache commit.
pub struct ResolvedSpeculativeRound<S, C, R> {
    /// Tentatively advanced sampler state.
    sampler: S,
    /// Tentatively advanced semantic state.
    constraint: C,
    /// Tentatively advanced canonical sequence.
    sequence: SpeculativeSequence,
    /// Tentatively advanced target randomness.
    target_randomness: Option<R>,
    /// Number of accepted proposals.
    accepted_proposals: usize,
    /// Tokens visible after cache commit.
    committed_tokens: SpeculativeBuffer<u32>,
    /// Exact verification inputs retained by cache commit.
    verified_inputs: usize,
    /// Target bonus token, when full acceptance produced one.
    bonus_token: Option<u32>,
    /// Terminal reason after this round.
    finish_reason: Option<FinishReason>,
}

fn driver_capacity<E: std::error::Error + 'static>(
    a: usize,
    b: usize,
) -> Result<usize, SpeculativeDriverError<E>> {
    a.checked_add(b).ok_or_else(|| {
        SpeculativeDriverError::Preparation(crate::HostMetadataFundingError::Overflow.into())
    })
}

/// Generates one assistant proposal block through opaque backend operations.
#[allow(clippy::too_many_arguments)]
pub fn propose_block<'a, E, S>(
    executor: &mut E,
    sampler: &S,
    state: &mut E::DraftState,
    first_previous: u32,
    count: usize,
    base_history: &[u32],
    temperature: f32,
    eos_token_ids: &[u32],
    draft_randomness: Option<&S::DraftRandomness>,
    context: E::Context<'a>,
) -> Result<SpeculativeBuffer<SpeculativeProposal<S::Distribution>>, SpeculativeDriverError<E::Error>>
where
    E: SpeculativeExecutor + 'a,
    S: SpeculativeSampling<Logits = E::Logits, Error = E::Error, Context<'a> = E::Context<'a>> + 'a,
{
    propose_block_at(
        executor,
        sampler,
        state,
        first_previous,
        count,
        base_history,
        temperature,
        eos_token_ids,
        draft_randomness,
        context,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn propose_block_at<'a, E, S>(
    executor: &mut E,
    sampler: &S,
    state: &mut E::DraftState,
    first_previous: u32,
    count: usize,
    base_history: &[u32],
    temperature: f32,
    eos_token_ids: &[u32],
    draft_randomness: Option<&S::DraftRandomness>,
    context: E::Context<'a>,
    origin: Option<SpeculativeActivationOrigin>,
) -> Result<SpeculativeBuffer<SpeculativeProposal<S::Distribution>>, SpeculativeDriverError<E::Error>>
where
    E: SpeculativeExecutor + 'a,
    S: SpeculativeSampling<Logits = E::Logits, Error = E::Error, Context<'a> = E::Context<'a>> + 'a,
{
    let mut branch_sampler = sampler.clone();
    let mut history =
        executor.driver_buffer(driver_capacity(base_history.len(), count)?, context)?;
    history
        .try_extend(base_history.iter().copied())
        .map_err(SpeculativeDriverError::Generation)?;
    let mut proposals =
        executor.driver_buffer::<SpeculativeProposal<S::Distribution>>(count, context)?;
    for offset in 0..count {
        let local = (|| -> Result<bool, SpeculativeDriverError<E::Error>> {
            let previous = proposals
                .last()
                .map_or(first_previous, |proposal| proposal.token);
            let raw = with_activation_origin(
                executor,
                origin.map(|o| o.with_prefix(&history)),
                |executor| executor.proposal_logits(state, previous, context),
            )?;
            let distribution = branch_sampler.process_logits(
                &raw,
                temperature,
                &history,
                SamplingPlacement::Draft,
                context,
            )?;
            let mut position_state = draft_randomness
                .map(|root| {
                    S::draft_randomness_at(
                        root,
                        SpeculativeDraftRandomPosition::new(base_history.len() + offset),
                        context,
                    )
                })
                .transpose()?;
            let token = branch_sampler.sample(
                &distribution,
                temperature,
                position_state.as_mut(),
                SamplingPlacement::Draft,
                context,
            )?;
            proposals
                .try_push(SpeculativeProposal {
                    token,
                    distribution,
                })
                .map_err(SpeculativeDriverError::Generation)?;
            history
                .try_push(token)
                .map_err(SpeculativeDriverError::Generation)?;
            Ok(eos_token_ids.contains(&token) || branch_sampler.prefix_is_complete(&history)?)
        })();
        if coordination::ready_result(executor, local, context)? {
            break;
        }
    }
    Ok(proposals)
}

/// Resolves one target verification transaction without backend-specific math.
#[allow(clippy::too_many_arguments)]
pub fn resolve_round<'a, 'sequence, E, S, C>(
    executor: &E,
    verification: &E::Verification,
    proposals: impl Into<SpeculativeBuffer<SpeculativeProposal<S::Distribution>>>,
    sampler: &S,
    sequence: impl Into<SpeculativeSequenceRef<'sequence>>,
    constraint: &C,
    target_randomness: Option<&S::RandomState>,
    temperature: f32,
    context: E::Context<'a>,
) -> Result<ResolvedSpeculativeRound<S, C, S::RandomState>, SpeculativeDriverError<E::Error>>
where
    E: SpeculativeExecutor + 'a,
    S: SpeculativeSampling<Logits = E::Logits, Error = E::Error, Context<'a> = E::Context<'a>> + 'a,
    C: SpeculativeConstraint,
{
    let mut proposals = proposals.into();
    let mut draft_distributions = executor.driver_buffer(proposals.len(), context)?;
    draft_distributions
        .try_extend(
            proposals
                .iter_mut()
                .map(|proposal| &mut proposal.distribution),
        )
        .map_err(SpeculativeDriverError::Generation)?;
    sampler.prepare_verification(&mut draft_distributions, temperature, context)?;
    drop(draft_distributions);
    let proposal_count = proposals.len();
    let mut sampler = sampler.clone();
    let mut sequence = executor.copy_sequence(sequence.into(), context)?;
    let mut constraint = constraint.fork().map_err(SpeculativeDriverError::Output)?;
    let mut target_randomness = target_randomness.cloned();
    let mut history = executor.driver_buffer(
        driver_capacity(sequence.tokens().len(), proposal_count)?,
        context,
    )?;
    history
        .try_extend(sequence.tokens().iter().copied())
        .map_err(SpeculativeDriverError::Generation)?;
    let round_tokens = executor.driver_buffer(driver_capacity(proposal_count, 1)?, context)?;
    let mut round = SpeculativeRound::with_buffer(proposal_count, round_tokens)
        .map_err(SpeculativeDriverError::Generation)?;
    let mut finish_reason = None;

    for (index, proposal) in proposals.iter().enumerate() {
        let raw = executor.verification_logits(verification, index, context)?;
        let target = sampler.process_logits(
            &raw,
            temperature,
            &history,
            SamplingPlacement::Target,
            context,
        )?;
        match decide_speculative_proposal(
            &sampler,
            &target,
            &proposal.distribution,
            proposal.token,
            temperature,
            target_randomness.as_mut(),
            context,
        )? {
            ProposalDecision::Accept => {
                sampler.update_sampler_state(
                    &target,
                    proposal.token,
                    SamplingPlacement::Target,
                    context,
                )?;
                history
                    .try_push(proposal.token)
                    .map_err(SpeculativeDriverError::Generation)?;
                finish_reason = commit_terminal_token(
                    &mut sequence,
                    &mut sampler,
                    &mut constraint,
                    proposal.token,
                )?;
                round
                    .accept(proposal.token, finish_reason.is_some())
                    .map_err(SpeculativeDriverError::Generation)?;
                if finish_reason.is_some() {
                    break;
                }
            }
            ProposalDecision::Reject(replacement) => {
                sampler.update_sampler_state(
                    &target,
                    replacement,
                    SamplingPlacement::Target,
                    context,
                )?;
                finish_reason = commit_terminal_token(
                    &mut sequence,
                    &mut sampler,
                    &mut constraint,
                    replacement,
                )?;
                round
                    .reject_with(replacement, finish_reason.is_some())
                    .map_err(SpeculativeDriverError::Generation)?;
                break;
            }
        }
    }

    let mut bonus_token = None;
    if round.is_full_acceptance() && !sequence.is_finished() {
        let raw = executor.verification_logits(verification, proposal_count, context)?;
        let target = sampler.process_logits(
            &raw,
            temperature,
            &history,
            SamplingPlacement::Target,
            context,
        )?;
        let chosen = sampler.sample(
            &target,
            temperature,
            target_randomness.as_mut(),
            SamplingPlacement::Target,
            context,
        )?;
        sampler.update_sampler_state(&target, chosen, SamplingPlacement::Target, context)?;
        finish_reason =
            commit_terminal_token(&mut sequence, &mut sampler, &mut constraint, chosen)?;
        round
            .bonus(chosen, finish_reason.is_some())
            .map_err(SpeculativeDriverError::Generation)?;
        bonus_token = Some(chosen);
    }
    let plan = round
        .commit_plan()
        .map_err(SpeculativeDriverError::Generation)?;
    let accepted_proposals = plan.accepted_proposals;
    let verified_inputs = plan.verified_inputs;
    let committed_tokens = round.into_storage();
    Ok(ResolvedSpeculativeRound {
        sampler,
        constraint,
        sequence,
        target_randomness,
        accepted_proposals,
        committed_tokens,
        verified_inputs,
        bonus_token,
        finish_reason,
    })
}

/// Submits one exact target verification and takes ownership of its resources.
pub fn submit_verification_transaction<'a, E, D>(
    executor: &mut E,
    cache: &mut E::Cache,
    last_committed_token: u32,
    block: SpeculativeDraftBlock<E::DraftState, D>,
    context: E::Context<'a>,
) -> Result<PendingSpeculativeVerification<E, D>, SpeculativeDriverError<E::Error>>
where
    E: SpeculativeExecutor + 'a,
{
    if block.proposals.is_empty() {
        return Err(SpeculativeDriverError::Generation(
            GenerationError::EmptyProposalBlock,
        ));
    }
    let rollback_failure =
        rollback::PreparedRollback::<SpeculativeDriverError<E::Error>, E::Error>::prepare(
            executor, context,
        )?;
    let mut input_tokens =
        executor.driver_buffer(driver_capacity(block.proposals.len(), 1)?, context)?;
    input_tokens
        .try_push(last_committed_token)
        .map_err(SpeculativeDriverError::Generation)?;
    input_tokens
        .try_extend(block.proposals.iter().map(|proposal| proposal.token))
        .map_err(SpeculativeDriverError::Generation)?;
    let checkpoint = executor
        .checkpoint_with_context(cache, context)
        .map_err(SpeculativeDriverError::Backend);
    let checkpoint = coordination::ready_result(executor, checkpoint, context)?;
    let submission = match executor.submit_verification(&input_tokens, cache, context) {
        Ok(submission) => submission,
        Err(submission) => {
            // Restoration runs while the original failure and its native
            // prefix remain owned. Retain both causes if it cannot settle.
            return Err(
                match executor.restore_checkpoint(cache, &checkpoint, context) {
                    Ok(()) => SpeculativeDriverError::Backend(submission),
                    Err(rollback) => SpeculativeDriverError::SubmissionRollback {
                        submission,
                        rollback,
                    },
                },
            );
        }
    };
    Ok(PendingSpeculativeVerification {
        completion: submission.completion,
        verification: submission.output,
        checkpoint,
        block,
        optimistic: None,
        submitted: Instant::now(),
        submitted_tokens: input_tokens.len(),
        rollback_failure,
    })
}

/// Request state selected after committed output publication.
#[non_exhaustive]
pub enum SpeculativePublicationStatus<S, D> {
    /// Continue from canonical target state and an optional promoted block.
    Continue(SpeculativeContinuation<S, D>),
    /// Generation reached a normal terminal condition.
    Completed,
    /// Cancellation won at or after the exact commit boundary.
    Cancelled,
}

/// Backend and portable state after exact commit and legal publication.
pub struct PublishedSpeculativeVerification<TargetState, DraftState, Distribution, RandomState, T> {
    /// Target state matching the committed backend cache.
    target_state: TargetState,
    /// Canonical target randomness after resolution.
    target_randomness: Option<RandomState>,
    /// Updated portable request telemetry.
    stats: SpeculativeStats,
    /// Backend component telemetry observed at exact completion.
    telemetry: T,
    /// Request continuation selected after publication.
    status: SpeculativePublicationStatus<DraftState, Distribution>,
}

/// Publication result produced after a speculative verification commits.
pub type PublishedSpeculativeResult<E, S> = Result<
    PublishedSpeculativeVerification<
        <E as SpeculativeExecutor>::TargetState,
        <E as SpeculativeExecutor>::DraftState,
        <S as SpeculativeSampling>::Distribution,
        <S as SpeculativeSampling>::RandomState,
        <E as SpeculativeExecutor>::Telemetry,
    >,
    SpeculativeDriverError<<E as SpeculativeExecutor>::Error>,
>;

/// Waits, resolves, commits, and only then publishes one verification.
///
/// Portable sampler, sequence, constraint, telemetry, and optimistic state are
/// advanced transactionally. A backend cache-commit failure leaves the
/// canonical output runtime unchanged and publishes nothing.
#[allow(clippy::too_many_arguments)]
pub fn resolve_commit_and_publish<'a, E, S, C, P>(
    executor: &mut E,
    cache: &mut E::Cache,
    pending: PendingSpeculativeVerification<E, S::Distribution>,
    runtime: &mut SpeculativeOutputRuntime<S, C, P>,
    target_randomness: Option<&S::RandomState>,
    temperature: f32,
    mut stats: SpeculativeStats,
    options: SpeculativeSchedulerOptions,
    context: E::Context<'a>,
) -> PublishedSpeculativeResult<E, S>
where
    E: SpeculativeExecutor + 'a,
    S: SpeculativeSampling<Logits = E::Logits, Error = E::Error, Context<'a> = E::Context<'a>> + 'a,
    C: SpeculativeConstraint,
    P: SpeculativePublisher<C>,
{
    let PendingSpeculativeVerification {
        completion,
        verification,
        checkpoint,
        block,
        optimistic,
        submitted,
        submitted_tokens: _,
        rollback_failure,
    } = pending;
    // Keep unconsumed native payloads outside phase closures so a local or
    // peer preparation failure restores the checkpoint before releasing them.
    let mut completion = Some(completion);
    let mut verification = Some(verification);
    let mut block = Some(block);
    let mut optimistic = optimistic;
    let stage = crate::run_preparation::TextPreparationStage::Delivery;
    let result = (|| {
        let prepared = (|| {
            let policy = options.completion_wait();
            let observed = if policy.is_ok() {
                runtime.observe_lifecycle(SpeculativeLifecycleStage::Completion)
            } else {
                Ok(())
            };
            coordination::settle_completion::<E>(&mut completion, policy, observed)?;
            let telemetry = executor.take_verification_telemetry(
                verification.as_mut().expect("pending verification"),
            )?;
            stats.counters.verification_in_flight_time += submitted.elapsed();
            runtime
                .observe_lifecycle(SpeculativeLifecycleStage::Observation)
                .map_err(SpeculativeDriverError::Output)?;
            let proposals = &block.as_ref().expect("pending draft").proposals;
            let mut canonical_proposal_prefix = executor.driver_buffer(
                driver_capacity(runtime.sequence().tokens().len(), proposals.len())?,
                context,
            )?;
            canonical_proposal_prefix
                .try_extend(runtime.sequence().tokens().iter().copied())
                .map_err(SpeculativeDriverError::Generation)?;
            canonical_proposal_prefix
                .try_extend(proposals.iter().map(|proposal| proposal.token))
                .map_err(SpeculativeDriverError::Generation)?;
            let resolved = resolve_round::<E, S, C>(
                executor,
                verification.as_ref().expect("pending verification"),
                std::mem::take(&mut block.as_mut().expect("pending draft").proposals),
                runtime.sampler(),
                runtime.sequence(),
                runtime.constraint(),
                target_randomness,
                temperature,
                context,
            )?;
            let continuation = resolve_optimistic_branch(
                optimistic.take(),
                &canonical_proposal_prefix,
                resolved.bonus_token,
                resolved.finish_reason.is_some(),
                &mut stats,
            )
            .map_err(SpeculativeDriverError::Generation)?;
            stats.counters.accepted_tokens += resolved.accepted_proposals;
            stats
                .accept_lens
                .try_push(resolved.accepted_proposals)
                .map_err(SpeculativeDriverError::Generation)?;
            stats.counters.rounds += 1;
            runtime
                .observe_lifecycle(SpeculativeLifecycleStage::CachePersistence)
                .map_err(SpeculativeDriverError::Output)?;
            Ok((resolved, continuation, telemetry))
        })();
        let (mut resolved, mut continuation, telemetry) =
            crate::run_preparation::finish_preparation(
                stage,
                prepared,
                |status| executor.agree_text_preparation(stage, status, context),
                SpeculativeDriverError::Preparation,
            )?;
        let terminal = resolved.finish_reason;
        let committed_tokens = resolved.committed_tokens;
        let commit = executor
            .commit_verification(
                verification.take().expect("pending verification"),
                block.take().expect("pending draft").state,
                cache,
                &checkpoint,
                resolved.verified_inputs,
                context,
            )
            .map_err(SpeculativeDriverError::Backend);
        let commit = crate::run_preparation::finish_preparation(
            stage,
            commit,
            |status| executor.agree_text_preparation(stage, status, context),
            SpeculativeDriverError::Preparation,
        )?;
        stats.counters.target_tokens += commit.replayed_tokens;
        stats.counters.emitted_tokens += committed_tokens.len();
        let target_randomness = resolved.target_randomness;
        let cancelled = coordination::publish_candidate(
            executor,
            runtime,
            &mut resolved.constraint,
            &mut resolved.sequence,
            &committed_tokens,
            context,
        )?;
        runtime.install_committed_state(resolved.sampler, resolved.constraint, resolved.sequence);
        let status = if cancelled {
            discard_continuation(&mut stats, continuation);
            SpeculativePublicationStatus::Cancelled
        } else if terminal.is_some() {
            discard_continuation(&mut stats, continuation);
            SpeculativePublicationStatus::Completed
        } else {
            stats.update_adaptive_lookahead(options);
            SpeculativePublicationStatus::Continue(std::mem::replace(
                &mut continuation,
                SpeculativeContinuation::None,
            ))
        };
        Ok(PublishedSpeculativeVerification {
            target_state: commit.state,
            target_randomness,
            stats,
            telemetry,
            status,
        })
    })();
    match result {
        Ok(value) => Ok(value),
        Err(operation) => {
            Err(rollback_failure.restore(operation, executor, cache, &checkpoint, context))
        }
    }
}

/// Resolves an exact retained verification solely to reach a safe cancellation boundary.
#[allow(clippy::too_many_arguments)]
pub fn cancel_pending_verification<'a, E, S, C, P>(
    executor: &mut E,
    cache: &mut E::Cache,
    pending: PendingSpeculativeVerification<E, S::Distribution>,
    runtime: &mut SpeculativeOutputRuntime<S, C, P>,
    mut stats: SpeculativeStats,
    completion_wait: BoundedCompletionWait,
    context: E::Context<'a>,
) -> Result<(SpeculativeStats, E::Telemetry), SpeculativeDriverError<E::Error>>
where
    E: SpeculativeExecutor + 'a,
    S: SpeculativeSampling<Logits = E::Logits, Error = E::Error, Context<'a> = E::Context<'a>> + 'a,
    C: SpeculativeConstraint,
    P: SpeculativePublisher<C>,
{
    let PendingSpeculativeVerification {
        completion,
        verification,
        checkpoint,
        block,
        optimistic,
        submitted,
        submitted_tokens: _,
        rollback_failure,
    } = pending;
    let mut completion = Some(completion);
    let mut verification = Some(verification);
    let mut block = Some(block);
    let mut optimistic = optimistic;
    let stage = crate::run_preparation::TextPreparationStage::Delivery;
    let result = (|| {
        let prepared = (|| {
            let observed = runtime.observe_lifecycle(SpeculativeLifecycleStage::Completion);
            coordination::settle_completion::<E>(&mut completion, Ok(completion_wait), observed)?;
            let telemetry = executor.take_verification_telemetry(
                verification.as_mut().expect("pending verification"),
            )?;
            stats.counters.verification_in_flight_time += submitted.elapsed();
            runtime
                .observe_lifecycle(SpeculativeLifecycleStage::Observation)
                .map_err(SpeculativeDriverError::Output)?;
            discard_branch(&mut stats, optimistic.take());
            runtime
                .observe_lifecycle(SpeculativeLifecycleStage::CachePersistence)
                .map_err(SpeculativeDriverError::Output)?;
            let constraint = runtime
                .constraint()
                .fork()
                .map_err(SpeculativeDriverError::Output)?;
            Ok((
                constraint,
                executor.copy_sequence(runtime.sequence().into(), context)?,
                telemetry,
            ))
        })();
        let (mut constraint, mut sequence, telemetry) = crate::run_preparation::finish_preparation(
            stage,
            prepared,
            |status| executor.agree_text_preparation(stage, status, context),
            SpeculativeDriverError::Preparation,
        )?;
        let commit = executor
            .commit_verification(
                verification.take().expect("pending verification"),
                block.take().expect("pending draft").state,
                cache,
                &checkpoint,
                1,
                context,
            )
            .map_err(SpeculativeDriverError::Backend);
        let commit = crate::run_preparation::finish_preparation(
            stage,
            commit,
            |status| executor.agree_text_preparation(stage, status, context),
            SpeculativeDriverError::Preparation,
        )?;
        stats.counters.target_tokens += commit.replayed_tokens;
        let cancelled = runtime
            .cancel_candidate(&mut constraint, &mut sequence)
            .map_err(SpeculativeDriverError::Output);
        crate::run_preparation::finish_preparation(
            stage,
            cancelled,
            |status| executor.agree_text_preparation(stage, status, context),
            SpeculativeDriverError::Preparation,
        )?;
        runtime.install_committed_state(runtime.sampler().clone(), constraint, sequence);
        Ok((stats, telemetry))
    })();
    match result {
        Ok(value) => Ok(value),
        Err(operation) => {
            Err(rollback_failure.restore(operation, executor, cache, &checkpoint, context))
        }
    }
}

/// Resolves, promotes, or discards one optimistic branch and updates telemetry.
pub fn resolve_optimistic_branch<S, D>(
    branch: Option<SpeculativeOptimisticBranch<S, D>>,
    canonical_prefix: &[u32],
    bonus: Option<u32>,
    terminal: bool,
    stats: &mut SpeculativeStats,
) -> Result<SpeculativeContinuation<S, D>, GenerationError> {
    let Some(branch) = branch else {
        return Ok(SpeculativeContinuation::None);
    };
    let Some(bonus) = bonus else {
        discard_branch(stats, Some(branch));
        return Ok(SpeculativeContinuation::None);
    };
    let decision = crate::generation::resolve_optimistic_reuse_parts(
        &branch.assumed_prefix,
        canonical_prefix,
        branch
            .block
            .proposals
            .first()
            .map(|proposal| proposal.token),
        branch.block.proposals.len(),
        bonus,
        terminal,
    )?;
    stats.counters.optimistic_target_bonus_tokens += 1;
    if decision == crate::generation::OptimisticReuseDecision::DiscardTerminal {
        discard_branch(stats, Some(branch));
        return Ok(SpeculativeContinuation::None);
    }
    let drafted = branch.block.proposals.len();
    let SpeculativeDraftBlock {
        state,
        mut proposals,
    } = branch.block;
    let _matched_or_discarded = proposals
        .remove_first()
        .expect("validated optimistic branch is non-empty");
    Ok(match decision {
        crate::generation::OptimisticReuseDecision::DiscardMismatch => {
            stats.counters.optimistic_bonus_mismatches += 1;
            stats.counters.discarded_optimistic_tokens += drafted;
            stats.counters.discarded_optimistic_blocks += 1;
            SpeculativeContinuation::None
        }
        crate::generation::OptimisticReuseDecision::MatchedConsumed => {
            stats.counters.optimistic_bonus_matches += 1;
            stats.counters.consumed_optimistic_tokens += 1;
            SpeculativeContinuation::None
        }
        crate::generation::OptimisticReuseDecision::MatchedRetained => {
            stats.counters.optimistic_bonus_matches += 1;
            stats.counters.consumed_optimistic_tokens += 1;
            stats.counters.draft_tokens += proposals.len();
            stats.counters.reused_optimistic_tokens += proposals.len();
            stats.counters.reused_optimistic_blocks += 1;
            SpeculativeContinuation::Promoted(SpeculativeDraftBlock { state, proposals })
        }
        crate::generation::OptimisticReuseDecision::DiscardTerminal => {
            unreachable!("terminal decision handled before branch destruction")
        }
    })
}

fn discard_branch<S, D>(
    stats: &mut SpeculativeStats,
    branch: Option<SpeculativeOptimisticBranch<S, D>>,
) {
    if let Some(branch) = branch {
        stats.counters.discarded_optimistic_tokens += branch.block.proposals.len();
        stats.counters.discarded_optimistic_blocks += 1;
    }
}

fn discard_continuation<S, D>(
    stats: &mut SpeculativeStats,
    continuation: SpeculativeContinuation<S, D>,
) {
    if let SpeculativeContinuation::Promoted(block) = continuation {
        stats.counters.discarded_optimistic_tokens += block.proposals.len();
        stats.counters.discarded_optimistic_blocks += 1;
        stats.counters.draft_tokens = stats
            .counters
            .draft_tokens
            .saturating_sub(block.proposals.len());
        stats.counters.reused_optimistic_tokens = stats
            .counters
            .reused_optimistic_tokens
            .saturating_sub(block.proposals.len());
        stats.counters.reused_optimistic_blocks =
            stats.counters.reused_optimistic_blocks.saturating_sub(1);
    }
}

fn commit_terminal_token<S, C>(
    sequence: &mut SpeculativeSequence,
    sampler: &mut S,
    constraint: &mut C,
    token: u32,
) -> Result<Option<FinishReason>, SpeculativeDriverError<S::Error>>
where
    S: SpeculativeSampling,
    C: SpeculativeConstraint,
{
    let stop_matched = constraint
        .push_token(token)
        .map_err(SpeculativeDriverError::Output)?;
    let grammar_complete = if stop_matched {
        false
    } else {
        sampler.grammar_is_complete()?
    };
    let reason = sequence
        .commit(
            token,
            TokenTerminalSignals {
                stop_sequence: stop_matched,
                grammar_complete,
            },
        )
        .map_err(SpeculativeDriverError::Generation)?
        .finish_reason;
    if let Some(reason) = reason {
        constraint
            .finish(reason)
            .map_err(SpeculativeDriverError::Output)?;
    }
    Ok(reason)
}

/// One backend-neutral speculative request with opaque execution resources.
///
/// The request owns every resource slot whose presence is constrained by the
/// lifecycle: target state, canonical draft block, exact in-flight
/// verification, randomness, output state, and cache access. Backends choose
/// the concrete associated types but cannot maintain a parallel request state.
pub struct SpeculativeRequest<'cache, E, S, C, P>
where
    E: SpeculativeExecutor,
    S: SpeculativeSampling<Logits = E::Logits, Error = E::Error>,
    C: SpeculativeConstraint,
    P: SpeculativePublisher<C>,
{
    control_identity: SpeculativeRequestIdentity,
    id: SpeculativeRequestId,
    cache: &'cache mut E::Cache,
    config: RequestConfiguration,
    runtime: SpeculativeOutputRuntime<S, C, P>,
    target_randomness: Option<S::RandomState>,
    draft_randomness: Option<S::DraftRandomness>,
    stats: SpeculativeStats,
    started: Instant,
    target_state: Option<E::TargetState>,
    block: Option<SpeculativeDraftBlock<E::DraftState, S::Distribution>>,
    pending: Option<PendingSpeculativeVerification<E, S::Distribution>>,
    lifecycle: SpeculativeRequestLifecycle,
}

impl<'cache, E, S, C, P> SpeculativeRequest<'cache, E, S, C, P>
where
    E: SpeculativeExecutor,
    S: SpeculativeSampling<Logits = E::Logits, Error = E::Error>,
    C: SpeculativeConstraint,
    P: SpeculativePublisher<C>,
{
    fn activation_origin(
        &self,
        executor: &E,
        optimistic: bool,
    ) -> Option<SpeculativeActivationOrigin> {
        executor.requires_activation_origin().then(|| {
            SpeculativeActivationOrigin::new(self.id, self.runtime.sequence().tokens(), optimistic)
        })
    }

    /// Stable insertion-order identity.
    pub const fn id(&self) -> SpeculativeRequestId {
        self.id
    }

    /// Current validated lifecycle status.
    pub const fn status(&self) -> SpeculativeRequestStatus {
        self.lifecycle.status()
    }

    /// Portable request statistics.
    pub const fn stats(&self) -> &SpeculativeStats {
        &self.stats
    }

    /// Canonical committed token sequence.
    pub const fn sequence(&self) -> &SpeculativeSequence {
        self.runtime.sequence()
    }

    /// Canonical sampler state.
    pub const fn sampler(&self) -> &S {
        self.runtime.sampler()
    }

    /// Canonical proposal block awaiting submission, when present.
    pub const fn block(&self) -> Option<&SpeculativeDraftBlock<E::DraftState, S::Distribution>> {
        self.block.as_ref()
    }

    /// Whether an exact target verification remains retained.
    pub const fn has_pending_verification(&self) -> bool {
        self.pending.is_some()
    }

    fn transition(
        &mut self,
        next: SpeculativeRequestStatus,
    ) -> Result<(), SpeculativeDriverError<E::Error>> {
        self.lifecycle
            .transition(next)
            .map_err(SpeculativeDriverError::Generation)
    }

    fn request_cancellation(&mut self) -> Result<(), SpeculativeDriverError<E::Error>> {
        let lifecycle = self.lifecycle.clone();
        match self
            .lifecycle
            .request_cancellation(self.pending.is_some())
            .map_err(SpeculativeDriverError::Generation)?
        {
            SpeculativeCancellationDisposition::AlreadyTerminal
            | SpeculativeCancellationDisposition::Deferred => {}
            SpeculativeCancellationDisposition::CancelNow => {
                if let Err(error) = self.runtime.cancel() {
                    self.lifecycle = lifecycle;
                    return Err(SpeculativeDriverError::Output(error));
                }
                self.block = None;
                self.stats.counters.elapsed = self.started.elapsed();
            }
        }
        Ok(())
    }

    fn candidate<'context>(
        &self,
        executor: &E,
        optimistic_execution_available: bool,
        completion_wait: BoundedCompletionWait,
        context: E::Context<'context>,
    ) -> Result<SpeculativeCandidate, SpeculativeDriverError<E::Error>>
    where
        E: 'context,
        S: SpeculativeSampling<
                Logits = E::Logits,
                Error = E::Error,
                Context<'context> = E::Context<'context>,
            > + 'context,
    {
        let (verification_complete, verification_deadline_expired) =
            if let Some(pending) = self.pending.as_ref() {
                (
                    pending.is_complete()?,
                    pending.submitted.elapsed() >= completion_wait.timeout(),
                )
            } else {
                (false, false)
            };
        let optimistic_eligible = if self.lifecycle.status()
            != SpeculativeRequestStatus::TargetVerificationInFlight
            || !optimistic_execution_available
            || self.runtime.cancellation().is_cancelled()
            || self.lifecycle.cancellation_pending()
        {
            false
        } else {
            let pending = self
                .pending
                .as_ref()
                .expect("in-flight request retains its verification transaction");
            let block = pending.block();
            let assumed_len = driver_capacity(
                self.runtime.sequence().tokens().len(),
                block.proposals.len(),
            )?;
            let mut assumed_prefix = executor.driver_buffer(assumed_len, context)?;
            assumed_prefix
                .try_extend(self.runtime.sequence().tokens().iter().copied())
                .map_err(SpeculativeDriverError::Generation)?;
            assumed_prefix
                .try_extend(block.proposals.iter().map(|proposal| proposal.token))
                .map_err(SpeculativeDriverError::Generation)?;
            executor.supports_exact_optimistic_promotion()
                && self.runtime.sampler().supports_exact_optimistic_promotion()
                && !self.stats.counters.adaptive_lookahead_disabled
                && !block.proposals.is_empty()
                && !self.runtime.sampler().prefix_is_complete(&assumed_prefix)?
                && !block
                    .proposals
                    .last()
                    .is_some_and(|proposal| self.config.eos_token_ids.contains(&proposal.token))
                && self.config.max_tokens.saturating_sub(assumed_len) > 1
        };
        Ok(SpeculativeCandidate {
            status: self.lifecycle.status(),
            optimistic_eligible,
            verification_complete,
            verification_deadline_expired,
        })
    }

    fn draft_committed<'context>(
        &mut self,
        executor: &mut E,
        context: E::Context<'context>,
    ) -> Result<bool, SpeculativeDriverError<E::Error>>
    where
        E: 'context,
        S: SpeculativeSampling<
                Logits = E::Logits,
                Error = E::Error,
                Context<'context> = E::Context<'context>,
            > + 'context,
    {
        let observed = self
            .runtime
            .observe_lifecycle(SpeculativeLifecycleStage::Execution)
            .map_err(SpeculativeDriverError::Output);
        coordination::ready_result(executor, observed, context)?;
        let target_count = self
            .config
            .geometry(executor.max_proposals())
            .proposal_count(self.runtime.sequence().tokens().len());
        if target_count == 0 {
            self.transition(SpeculativeRequestStatus::Completed)?;
            self.stats.counters.elapsed = self.started.elapsed();
            return Ok(false);
        }

        let origin = self.activation_origin(executor, false);
        let mut block = if let Some(block) = self.block.take() {
            block
        } else {
            let last = *self
                .runtime
                .sequence()
                .tokens()
                .last()
                .expect("prefill emitted a token");
            let target_state = self
                .target_state
                .as_ref()
                .expect("ready request has target state");
            let state = with_activation_origin(executor, origin, |executor| {
                executor.begin_proposal(target_state, last, target_count, context)
            })
            .map_err(SpeculativeDriverError::Backend);
            SpeculativeDraftBlock {
                state: coordination::ready_result(executor, state, context)?,
                proposals: SpeculativeBuffer::default(),
            }
        };
        if block.proposals.len() > target_count {
            return Err(SpeculativeDriverError::Generation(
                GenerationError::ProposalCapacityExceeded {
                    proposed: block.proposals.len(),
                    capacity: target_count,
                },
            ));
        }
        let additional = if block
            .proposals
            .last()
            .is_some_and(|proposal| self.config.eos_token_ids.contains(&proposal.token))
        {
            0
        } else {
            target_count - block.proposals.len()
        };
        if additional > 0 {
            // A fresh block can use the canonical slice directly and keep the
            // returned proposal destination. Only an actual promoted prefix
            // needs concatenation storage before the next native proposal.
            let history = if block.proposals.is_empty() {
                None
            } else {
                let mut history = executor.driver_buffer(
                    driver_capacity(
                        self.runtime.sequence().tokens().len(),
                        block.proposals.len(),
                    )?,
                    context,
                )?;
                history
                    .try_extend(self.runtime.sequence().tokens().iter().copied())
                    .map_err(SpeculativeDriverError::Generation)?;
                history
                    .try_extend(block.proposals.iter().map(|proposal| proposal.token))
                    .map_err(SpeculativeDriverError::Generation)?;
                Some(history)
            };
            let combined = if block.proposals.is_empty() {
                None
            } else {
                Some(executor.driver_buffer(target_count, context)?)
            };
            let previous = block.proposals.last().map_or_else(
                || {
                    *self
                        .runtime
                        .sequence()
                        .tokens()
                        .last()
                        .expect("prefill emitted a token")
                },
                |proposal| proposal.token,
            );
            let proposals = propose_block_at(
                executor,
                self.runtime.sampler(),
                &mut block.state,
                previous,
                additional,
                history
                    .as_deref()
                    .unwrap_or(self.runtime.sequence().tokens()),
                self.config.temperature,
                &self.config.eos_token_ids,
                self.draft_randomness.as_ref(),
                context,
                origin,
            )?;
            self.stats.counters.draft_tokens += proposals.len();
            block.proposals = if let Some(mut combined) = combined {
                combined
                    .try_extend(block.proposals)
                    .map_err(SpeculativeDriverError::Generation)?;
                combined
                    .try_extend(proposals)
                    .map_err(SpeculativeDriverError::Generation)?;
                combined
            } else {
                proposals
            };
        }
        executor.take_telemetry()?.record(&mut self.stats);
        self.block = Some(block);
        self.transition(SpeculativeRequestStatus::ReadyToSubmitVerification)?;
        Ok(additional > 0)
    }

    fn submit_verification<'context>(
        &mut self,
        executor: &mut E,
        context: E::Context<'context>,
    ) -> Result<(), SpeculativeDriverError<E::Error>>
    where
        E: 'context,
        S: SpeculativeSampling<
                Logits = E::Logits,
                Error = E::Error,
                Context<'context> = E::Context<'context>,
            > + 'context,
    {
        let observed = self
            .runtime
            .observe_lifecycle(SpeculativeLifecycleStage::Execution)
            .map_err(SpeculativeDriverError::Output);
        coordination::ready_result(executor, observed, context)?;
        let block = self
            .block
            .take()
            .expect("verification-ready request has a draft block");
        let last = *self
            .runtime
            .sequence()
            .tokens()
            .last()
            .expect("prefill emitted a token");
        let origin = self.activation_origin(executor, false);
        let pending = with_activation_origin(executor, origin, |executor| {
            submit_verification_transaction(executor, self.cache, last, block, context)
        })?;
        self.stats.counters.target_tokens += pending.submitted_tokens();
        self.pending = Some(pending);
        self.transition(SpeculativeRequestStatus::TargetVerificationInFlight)
    }

    fn draft_optimistic<'context>(
        &mut self,
        executor: &mut E,
        context: E::Context<'context>,
    ) -> Result<(), SpeculativeDriverError<E::Error>>
    where
        E: 'context,
        S: SpeculativeSampling<
                Logits = E::Logits,
                Error = E::Error,
                Context<'context> = E::Context<'context>,
            > + 'context,
    {
        let observed = self
            .runtime
            .observe_lifecycle(SpeculativeLifecycleStage::Execution)
            .map_err(SpeculativeDriverError::Output);
        coordination::ready_result(executor, observed, context)?;
        let started = Instant::now();
        self.transition(SpeculativeRequestStatus::OptimisticDraftRunning)?;
        let origin = self.activation_origin(executor, true);
        let pending = self
            .pending
            .as_mut()
            .expect("optimistic request has an in-flight verification");
        let block = pending.block();
        let assumed_len = driver_capacity(
            self.runtime.sequence().tokens().len(),
            block.proposals.len(),
        )?;
        let count = self
            .config
            .geometry(executor.max_proposals())
            .proposal_count(assumed_len);
        let copied = executor
            .copy_draft_state(&block.state, context)
            .map_err(SpeculativeDriverError::Backend);
        let mut state = coordination::ready_result(executor, copied, context)?;
        let last = block
            .proposals
            .last()
            .expect("optimistic block has an assumed token")
            .token;
        let mut history = executor.driver_buffer(assumed_len, context)?;
        history
            .try_extend(self.runtime.sequence().tokens().iter().copied())
            .map_err(SpeculativeDriverError::Generation)?;
        history
            .try_extend(block.proposals.iter().map(|proposal| proposal.token))
            .map_err(SpeculativeDriverError::Generation)?;
        let proposals = propose_block_at(
            executor,
            self.runtime.sampler(),
            &mut state,
            last,
            count,
            &history,
            self.config.temperature,
            &self.config.eos_token_ids,
            self.draft_randomness.as_ref(),
            context,
            origin,
        )?;
        self.stats.counters.optimistic_draft_tokens += proposals.len();
        self.stats.counters.optimistic_draft_blocks += 1;
        self.stats.counters.optimistic_draft_time += started.elapsed();
        pending
            .set_optimistic_branch(SpeculativeOptimisticBranch {
                block: SpeculativeDraftBlock { state, proposals },
                assumed_prefix: history,
            })
            .map_err(SpeculativeDriverError::Generation)?;
        self.transition(SpeculativeRequestStatus::OptimisticDraftReady)
    }

    fn resolve_verification<'context>(
        &mut self,
        executor: &mut E,
        options: SpeculativeSchedulerOptions,
        context: E::Context<'context>,
    ) -> Result<(), SpeculativeDriverError<E::Error>>
    where
        E: 'context,
        S: SpeculativeSampling<
                Logits = E::Logits,
                Error = E::Error,
                Context<'context> = E::Context<'context>,
            > + 'context,
    {
        let prepared_stats = self.stats.copy_for_driver(executor, true, context);
        let stats = coordination::ready_result(executor, prepared_stats, context)?;
        let origin = self.activation_origin(executor, false);
        self.transition(SpeculativeRequestStatus::VerificationResolution)?;
        let pending = self
            .pending
            .take()
            .expect("resolving request has an in-flight verification");
        if self.lifecycle.cancellation_pending() || self.runtime.cancellation().is_cancelled() {
            let (mut stats, telemetry) = with_activation_origin(executor, origin, |executor| {
                cancel_pending_verification(
                    executor,
                    self.cache,
                    pending,
                    &mut self.runtime,
                    stats,
                    options
                        .completion_wait()
                        .map_err(SpeculativeDriverError::Generation)?,
                    context,
                )
            })?;
            telemetry.record(&mut stats);
            self.stats = stats;
            self.transition(SpeculativeRequestStatus::Cancelled)?;
            self.stats.counters.elapsed = self.started.elapsed();
            return Ok(());
        }
        let mut published = with_activation_origin(executor, origin, |executor| {
            resolve_commit_and_publish(
                executor,
                self.cache,
                pending,
                &mut self.runtime,
                self.target_randomness.as_ref(),
                self.config.temperature,
                stats,
                options,
                context,
            )
        })?;
        published.telemetry.record(&mut published.stats);
        self.target_state = Some(published.target_state);
        self.target_randomness = published.target_randomness;
        self.stats = published.stats;
        match published.status {
            SpeculativePublicationStatus::Continue(continuation) => {
                self.block = continuation.into_block();
                self.transition(SpeculativeRequestStatus::ReadyToDraft)?;
            }
            SpeculativePublicationStatus::Completed => {
                self.transition(SpeculativeRequestStatus::Completed)?;
                self.stats.counters.elapsed = self.started.elapsed();
            }
            SpeculativePublicationStatus::Cancelled => {
                self.transition(SpeculativeRequestStatus::Cancelled)?;
                self.stats.counters.elapsed = self.started.elapsed();
            }
        }
        Ok(())
    }
}

/// One completed request returned in stable submission order.
pub struct CompletedSpeculativeRequest<S> {
    /// Stable request identity.
    id: SpeculativeRequestId,
    /// Canonical generated token sequence.
    token_ids: SpeculativeTokenIds,
    /// Portable request telemetry.
    stats: SpeculativeStats,
    /// Final backend sampling state.
    sampler: S,
    /// Terminal reason selected by the canonical sequence.
    finish_reason: Option<FinishReason>,
    /// Terminal lifecycle status.
    status: SpeculativeRequestStatus,
}

impl<S> CompletedSpeculativeRequest<S> {
    /// Stable request identity.
    pub const fn id(&self) -> SpeculativeRequestId {
        self.id
    }
    /// Canonical emitted token ids.
    pub fn token_ids(&self) -> &[u32] {
        &self.token_ids
    }
    /// Portable request telemetry.
    pub const fn stats(&self) -> &SpeculativeStats {
        &self.stats
    }
    /// Final sampling state.
    pub const fn sampler(&self) -> &S {
        &self.sampler
    }
    /// Terminal reason, when completed normally.
    pub const fn finish_reason(&self) -> Option<FinishReason> {
        self.finish_reason
    }
    /// Terminal request status.
    pub const fn status(&self) -> SpeculativeRequestStatus {
        self.status
    }
    /// Consumes the request into a named handoff artifact.
    pub fn into_artifact(self) -> CompletedSpeculativeRequestArtifact<S> {
        CompletedSpeculativeRequestArtifact {
            id: self.id,
            token_ids: self.token_ids,
            stats: self.stats,
            sampler: self.sampler,
            finish_reason: self.finish_reason,
            status: self.status,
        }
    }
}

/// Named consuming artifact for adapting one completed speculative request.
pub struct CompletedSpeculativeRequestArtifact<S> {
    id: SpeculativeRequestId,
    token_ids: SpeculativeTokenIds,
    stats: SpeculativeStats,
    sampler: S,
    finish_reason: Option<FinishReason>,
    status: SpeculativeRequestStatus,
}

impl<S> CompletedSpeculativeRequestArtifact<S> {
    /// Stable request identity.
    pub const fn id(&self) -> SpeculativeRequestId {
        self.id
    }
    /// Takes canonical token ids.
    pub fn take_token_ids(&mut self) -> SpeculativeTokenIds {
        std::mem::take(&mut self.token_ids)
    }
    /// Takes request telemetry.
    pub fn take_stats(&mut self) -> SpeculativeStats {
        std::mem::take(&mut self.stats)
    }
    /// Consumes the artifact into its final sampler.
    pub fn into_sampler(self) -> S {
        self.sampler
    }
    /// Terminal finish reason.
    pub const fn finish_reason(&self) -> Option<FinishReason> {
        self.finish_reason
    }
    /// Terminal lifecycle status.
    pub const fn status(&self) -> SpeculativeRequestStatus {
        self.status
    }
}

/// Completed request table and aggregate fair-scheduler telemetry.
pub struct CompletedSpeculativeSchedule<S> {
    /// Requests in stable submission order.
    requests: SpeculativeBuffer<CompletedSpeculativeRequest<S>>,
    /// Aggregate scheduler telemetry.
    scheduler: SpeculativeSchedulerStats,
}

impl<S> CompletedSpeculativeSchedule<S> {
    /// Consumes the schedule into request results.
    pub fn into_requests(self) -> SpeculativeBuffer<CompletedSpeculativeRequest<S>> {
        self.requests
    }
    /// Takes completed requests while retaining access to scheduler telemetry.
    pub fn take_requests(&mut self) -> SpeculativeBuffer<CompletedSpeculativeRequest<S>> {
        std::mem::take(&mut self.requests)
    }
    /// Takes aggregate scheduler telemetry.
    pub fn take_scheduler(&mut self) -> SpeculativeSchedulerStats {
        std::mem::take(&mut self.scheduler)
    }
    /// Aggregate scheduler telemetry.
    pub const fn scheduler(&self) -> &SpeculativeSchedulerStats {
        &self.scheduler
    }
}

/// Canonical table and action coordinator for speculative requests.
pub struct SpeculativeRequestTable<'cache, E, S, C, P>
where
    E: SpeculativeExecutor,
    S: SpeculativeSampling<Logits = E::Logits, Error = E::Error>,
    C: SpeculativeConstraint,
    P: SpeculativePublisher<C>,
{
    schedule: SpeculativeSchedule,
    requests: SpeculativeBuffer<SpeculativeRequest<'cache, E, S, C, P>>,
    completed: SpeculativeBuffer<CompletedSpeculativeRequest<S>>,
    stats: SpeculativeSchedulerStats,
}

impl<'cache, E, S, C, P> SpeculativeRequestTable<'cache, E, S, C, P>
where
    E: SpeculativeExecutor,
    S: SpeculativeSampling<Logits = E::Logits, Error = E::Error>,
    C: SpeculativeConstraint,
    P: SpeculativePublisher<C>,
{
    /// Creates an empty validated request table.
    pub fn new(
        options: SpeculativeSchedulerOptions,
        topology: SpeculativeExecutionTopology,
    ) -> Result<Self, GenerationError> {
        Ok(Self {
            schedule: SpeculativeSchedule::new(options)?,
            requests: SpeculativeBuffer::default(),
            completed: SpeculativeBuffer::default(),
            stats: SpeculativeSchedulerStats {
                execution_topology: topology,
                ..SpeculativeSchedulerStats::default()
            },
        })
    }

    /// Reserves both actual request and terminal row destinations before any
    /// lane begins. Growth constructs a new paid buffer before moving old rows;
    /// existing Vec allocations are never relabelled with fresh custody.
    pub fn reserve_requests(
        &mut self,
        count: usize,
        executor: &E,
        context: E::Context<'_>,
    ) -> Result<(), SpeculativeDriverError<E::Error>> {
        if count <= self.requests.capacity() && count <= self.completed.capacity() {
            return Ok(());
        }
        let mut requests = executor.driver_buffer(count, context)?;
        let completed = executor.driver_buffer(count, context)?;
        requests
            .try_extend(std::mem::take(&mut self.requests))
            .map_err(SpeculativeDriverError::Generation)?;
        self.requests = requests;
        self.completed = completed;
        Ok(())
    }

    /// Returns one request by stable identity.
    pub fn request(
        &self,
        id: SpeculativeRequestId,
    ) -> Option<&SpeculativeRequest<'cache, E, S, C, P>> {
        self.requests.get(id.index())
    }

    /// Returns one request's current status.
    pub fn status(&self, id: SpeculativeRequestId) -> Option<SpeculativeRequestStatus> {
        self.request(id).map(SpeculativeRequest::status)
    }

    /// Whether every request is terminal.
    pub fn is_finished(&self) -> bool {
        self.requests
            .iter()
            .all(|request| request.lifecycle.is_terminal())
    }

    /// Validated scheduler options.
    pub const fn options(&self) -> SpeculativeSchedulerOptions {
        self.schedule.options()
    }

    /// Prefills and inserts one request, or records its pre-existing terminal state.
    #[allow(clippy::too_many_arguments)]
    pub fn submit<'context>(
        &mut self,
        executor: &mut E,
        cache: &'cache mut E::Cache,
        input: E::Input,
        config: impl Into<SpeculativeConfiguration>,
        mut runtime: SpeculativeOutputRuntime<S, C, P>,
        randomness: SpeculativeRandomness<S::RandomState, S::DraftRandomness>,
        component_timings_collected: bool,
        context: E::Context<'context>,
    ) -> Result<SpeculativeRequestId, SpeculativeDriverError<E::Error>>
    where
        E: 'context,
        S: SpeculativeSampling<
                Logits = E::Logits,
                Error = E::Error,
                Context<'context> = E::Context<'context>,
            > + 'context,
    {
        let config = config.into();
        let stage = crate::run_preparation::TextPreparationStage::Delivery;
        let prepared = (|| {
            config
                .validate()
                .map_err(SpeculativeDriverError::Generation)?;
            if executor.max_proposals() == 0 {
                return Err(SpeculativeDriverError::Generation(
                    GenerationError::NoBackendDraftCapacity,
                ));
            }
            Ok((!runtime.cancellation().is_cancelled()).then_some(()))
        })();
        if crate::run_preparation::finish_preparation_cancellable(
            stage,
            prepared,
            |status| executor.agree_text_preparation(stage, status, context),
            SpeculativeDriverError::Preparation,
        )?
        .is_none()
        {
            runtime.cancellation().cancel();
        }
        let started = Instant::now();
        let needed = driver_capacity(self.requests.len(), 1)?;
        let capacity = if needed > self.requests.capacity() {
            driver_capacity(self.requests.capacity(), self.requests.capacity())?.max(needed)
        } else {
            needed
        };
        let storage = self.reserve_requests(capacity, executor, context);
        crate::run_preparation::finish_preparation(
            stage,
            storage,
            |status| executor.agree_text_preparation(stage, status, context),
            SpeculativeDriverError::Preparation,
        )?;
        let id = SpeculativeRequestId::new(self.requests.len());
        let selected = executor
            .request_context(id, context)
            .map_err(SpeculativeDriverError::Backend);
        let context = crate::run_preparation::finish_preparation(
            stage,
            selected,
            |status| executor.agree_text_preparation(stage, status, context),
            SpeculativeDriverError::Preparation,
        )?;
        let history = executor
            .driver_host_metadata(Self::publication_control_bytes(), context)
            .map_err(SpeculativeDriverError::Backend)
            .and_then(|funding| {
                executor
                    .driver_buffer(0, context)
                    .map(|history| (funding, history))
                    .map_err(SpeculativeDriverError::Backend)
            });
        let history = history.and_then(|(funding, history)| {
            executor
                .driver_identity(context)
                .map(|identity| (funding, history, identity))
                .map_err(SpeculativeDriverError::Backend)
        });
        let history = history.and_then(|(funding, history, identity)| {
            RequestConfiguration::prepare(config, executor, context)
                .map(|config| (funding, history, identity, config))
        });
        // This is the selected request's ordinary coordinated preparation. The
        // final row-assembly controls stay owned through its consuming call;
        // the fixed destination already owns the row after that call returns.
        let (_publication_funding, accept_lens, control_identity, config) =
            crate::run_preparation::finish_preparation(
                stage,
                history,
                |status| executor.agree_text_preparation(stage, status, context),
                SpeculativeDriverError::Preparation,
            )?;
        let mut stats = SpeculativeStats {
            counters: SpeculativeStatsCounters {
                execution_topology: self.stats.execution_topology,
                component_timings_collected,
                ..SpeculativeStatsCounters::default()
            },
            accept_lens,
        };
        let (mut target_randomness, draft_randomness) = (randomness.target, randomness.draft);
        let (target_state, lifecycle) = if runtime.cancellation().is_cancelled() {
            let cancelled = runtime.cancel().map_err(SpeculativeDriverError::Output);
            crate::run_preparation::finish_preparation(
                stage,
                cancelled,
                |status| executor.agree_text_preparation(stage, status, context),
                SpeculativeDriverError::Preparation,
            )?;
            stats.counters.elapsed = started.elapsed();
            (None, SpeculativeRequestLifecycle::cancelled())
        } else if runtime.sequence().is_finished() {
            stats.counters.elapsed = started.elapsed();
            (None, SpeculativeRequestLifecycle::completed())
        } else {
            'prefill_registration: {
                let observed = runtime
                    .observe_lifecycle(SpeculativeLifecycleStage::Input)
                    .and_then(|()| runtime.observe_lifecycle(SpeculativeLifecycleStage::Execution))
                    .map_err(SpeculativeDriverError::Output);
                crate::run_preparation::finish_preparation(
                    stage,
                    observed,
                    |status| executor.agree_text_preparation(stage, status, context),
                    SpeculativeDriverError::Preparation,
                )?;
                let rollback_failure = rollback::PreparedRollback::<
                    SpeculativeDriverError<E::Error>,
                    E::Error,
                >::prepare(executor, context)
                .map_err(SpeculativeDriverError::Backend);
                let rollback_failure = crate::run_preparation::finish_preparation(
                    stage,
                    rollback_failure,
                    |status| executor.agree_text_preparation(stage, status, context),
                    SpeculativeDriverError::Preparation,
                )?;
                let checkpoint = executor
                    .checkpoint_with_context(cache, context)
                    .map_err(SpeculativeDriverError::Backend);
                let checkpoint = crate::run_preparation::finish_preparation(
                    stage,
                    checkpoint,
                    |status| executor.agree_text_preparation(stage, status, context),
                    SpeculativeDriverError::Preparation,
                )?;
                let attempt = (|| {
                    let origin = executor
                        .requires_activation_origin()
                        .then(|| SpeculativeActivationOrigin::new(id, &[], false));
                    let cancellation = runtime.cancellation().clone();
                    let prefill = with_activation_origin(executor, origin, |executor| {
                        executor.prefill_cancellable(input, cache, &cancellation, context)
                    })
                    .map_err(SpeculativeDriverError::Backend)
                    .map(|outcome| match outcome {
                        SpeculativePrefillOutcome::Complete(value) => {
                            stats.counters.target_tokens = value.evaluated_tokens;
                            (!cancellation.is_cancelled()).then_some(value)
                        }
                        SpeculativePrefillOutcome::Cancelled { evaluated_tokens } => {
                            stats.counters.target_tokens = evaluated_tokens;
                            None
                        }
                    });
                    // Every participant resolves this boundary before any sampler
                    // clone/update or first-token publication can happen.
                    let Some(prefill) = crate::run_preparation::finish_preparation_cancellable(
                        stage,
                        prefill,
                        |status| executor.agree_text_preparation(stage, status, context),
                        SpeculativeDriverError::Preparation,
                    )?
                    else {
                        return Ok(None);
                    };
                    let computed = (|| {
                        let mut sampler = runtime.sampler().clone();
                        let mut constraint = runtime
                            .constraint()
                            .fork()
                            .map_err(SpeculativeDriverError::Output)?;
                        let mut sequence =
                            executor.copy_sequence(runtime.sequence().into(), context)?;
                        let mut target_randomness = target_randomness.clone();
                        let first_logits = sampler.process_logits(
                            &prefill.logits,
                            config.temperature,
                            &[],
                            SamplingPlacement::Target,
                            context,
                        )?;
                        let first = sampler.sample(
                            &first_logits,
                            config.temperature,
                            target_randomness.as_mut(),
                            SamplingPlacement::Target,
                            context,
                        )?;
                        sampler.update_sampler_state(
                            &first_logits,
                            first,
                            SamplingPlacement::Target,
                            context,
                        )?;
                        let reason = commit_terminal_token(
                            &mut sequence,
                            &mut sampler,
                            &mut constraint,
                            first,
                        )?;
                        let submission_to_first_token = started.elapsed();
                        Ok((
                            prefill,
                            sampler,
                            constraint,
                            sequence,
                            target_randomness,
                            reason,
                            first,
                            submission_to_first_token,
                        ))
                    })();
                    let (
                        prefill,
                        sampler,
                        mut constraint,
                        mut sequence,
                        target_randomness,
                        reason,
                        first,
                        submission_to_first_token,
                    ) = crate::run_preparation::finish_preparation(
                        stage,
                        computed,
                        |status| executor.agree_text_preparation(stage, status, context),
                        SpeculativeDriverError::Preparation,
                    )?;
                    let cancelled = coordination::publish_candidate(
                        executor,
                        &mut runtime,
                        &mut constraint,
                        &mut sequence,
                        &[first],
                        context,
                    )?;
                    runtime.install_committed_state(sampler, constraint, sequence);
                    Ok::<_, SpeculativeDriverError<E::Error>>(Some((
                        prefill.evaluated_tokens,
                        prefill.state,
                        target_randomness,
                        reason,
                        cancelled,
                        submission_to_first_token,
                    )))
                })();
                let (
                    evaluated_tokens,
                    target_state,
                    committed_randomness,
                    reason,
                    cancelled,
                    submission_to_first_token,
                ) = match attempt {
                    Ok(Some(result)) => result,
                    Ok(None) => {
                        let restored = executor
                            .restore_checkpoint(cache, &checkpoint, context)
                            .map_err(SpeculativeDriverError::Backend);
                        crate::run_preparation::finish_preparation(
                            stage,
                            restored,
                            |status| executor.agree_text_preparation(stage, status, context),
                            SpeculativeDriverError::Preparation,
                        )?;
                        runtime.cancellation().cancel();
                        let cancelled = runtime.cancel().map_err(SpeculativeDriverError::Output);
                        crate::run_preparation::finish_preparation(
                            stage,
                            cancelled,
                            |status| executor.agree_text_preparation(stage, status, context),
                            SpeculativeDriverError::Preparation,
                        )?;
                        stats.counters.elapsed = started.elapsed();
                        break 'prefill_registration (
                            None,
                            SpeculativeRequestLifecycle::cancelled(),
                        );
                    }
                    Err(operation) => {
                        return Err(rollback_failure.restore(
                            operation,
                            executor,
                            cache,
                            &checkpoint,
                            context,
                        ));
                    }
                };
                stats.counters.submission_to_first_token = Some(submission_to_first_token);
                stats.counters.target_tokens = evaluated_tokens;
                stats.counters.scheduler_turns = 1;
                stats.counters.emitted_tokens = 1;
                let lifecycle = if cancelled {
                    stats.counters.elapsed = started.elapsed();
                    SpeculativeRequestLifecycle::cancelled()
                } else if reason.is_some() {
                    stats.counters.elapsed = started.elapsed();
                    SpeculativeRequestLifecycle::completed()
                } else {
                    let mut lifecycle = SpeculativeRequestLifecycle::new();
                    lifecycle
                        .transition(SpeculativeRequestStatus::ReadyToDraft)
                        .map_err(SpeculativeDriverError::Generation)?;
                    lifecycle
                };
                self.stats.turns += 1;
                // Both terminal and prefilled requests publish through the
                // same paid destination below. Keep first-token RNG progress;
                // cancelled/failed attempts above still restore their checkpoint.
                target_randomness = committed_randomness;
                (Some(target_state), lifecycle)
            }
        };
        // No row assembly temporary needs to overlap the cold prefill quote.
        self.publish_submitted(
            control_identity,
            id,
            cache,
            config,
            runtime,
            target_randomness,
            draft_randomness,
            stats,
            started,
            target_state,
            lifecycle,
        )
    }

    fn publication_control_bytes() -> Option<usize> {
        let parts = [
            std::mem::size_of::<(
                &mut Self,
                SpeculativeRequestIdentity,
                SpeculativeRequestId,
                &'cache mut E::Cache,
                RequestConfiguration,
                SpeculativeOutputRuntime<S, C, P>,
                Option<S::RandomState>,
                Option<S::DraftRandomness>,
                SpeculativeStats,
                Instant,
                Option<E::TargetState>,
                SpeculativeRequestLifecycle,
            )>(),
            std::mem::size_of::<SpeculativeRequest<'cache, E, S, C, P>>(),
            std::mem::size_of::<Result<(), GenerationError>>(),
            std::mem::size_of::<Result<SpeculativeRequestId, SpeculativeDriverError<E::Error>>>(),
            std::mem::size_of::<crate::HostPreparationAuthority>(),
        ];
        parts
            .into_iter()
            .try_fold(std::mem::size_of_val(&parts), usize::checked_add)
    }

    #[inline(never)]
    #[allow(clippy::too_many_arguments)]
    fn publish_submitted(
        &mut self,
        control_identity: SpeculativeRequestIdentity,
        id: SpeculativeRequestId,
        cache: &'cache mut E::Cache,
        config: RequestConfiguration,
        runtime: SpeculativeOutputRuntime<S, C, P>,
        target_randomness: Option<S::RandomState>,
        draft_randomness: Option<S::DraftRandomness>,
        stats: SpeculativeStats,
        started: Instant,
        target_state: Option<E::TargetState>,
        lifecycle: SpeculativeRequestLifecycle,
    ) -> Result<SpeculativeRequestId, SpeculativeDriverError<E::Error>> {
        self.requests
            .try_push(SpeculativeRequest {
                control_identity,
                id,
                cache,
                config,
                runtime,
                target_randomness,
                draft_randomness,
                stats,
                started,
                target_state,
                block: None,
                pending: None,
                lifecycle,
            })
            .map_err(SpeculativeDriverError::Generation)?;
        Ok(id)
    }

    /// Requests cancellation without releasing an exact in-flight transaction.
    pub fn cancel(
        &mut self,
        id: SpeculativeRequestId,
    ) -> Result<(), SpeculativeDriverError<E::Error>> {
        let request = self.requests.get_mut(id.index()).ok_or_else(|| {
            SpeculativeDriverError::Generation(GenerationError::UnknownSpeculativeRequest {
                index: id.index(),
            })
        })?;
        request.request_cancellation()
    }

    /// Records cancellation without invoking a publisher or changing the
    /// lifecycle. The next coordinated action resolves this request with peers.
    pub fn signal_cancellation(
        &mut self,
        id: SpeculativeRequestId,
    ) -> Result<(), SpeculativeDriverError<E::Error>> {
        let request = self.requests.get_mut(id.index()).ok_or_else(|| {
            SpeculativeDriverError::Generation(GenerationError::UnknownSpeculativeRequest {
                index: id.index(),
            })
        })?;
        request.runtime.cancellation().cancel();
        Ok(())
    }

    /// Applies one fairly selected request action.
    pub fn step<'context>(
        &mut self,
        executor: &mut E,
        optimistic_execution_available: bool,
        context: E::Context<'context>,
    ) -> Result<bool, SpeculativeDriverError<E::Error>>
    where
        E: 'context,
        S: SpeculativeSampling<
                Logits = E::Logits,
                Error = E::Error,
                Context<'context> = E::Context<'context>,
            > + 'context,
    {
        let stage = crate::run_preparation::TextPreparationStage::Delivery;
        let local = (|| {
            let mut local = executor.driver_buffer(self.requests.len(), context)?;
            for request in &self.requests {
                let request_context = executor.request_context(request.id, context)?;
                let candidate = request.candidate(
                    executor,
                    optimistic_execution_available,
                    self.schedule
                        .options()
                        .completion_wait()
                        .expect("speculative schedule retains validated completion options"),
                    request_context,
                )?;
                local
                    .try_push(SpeculativeScheduleState {
                        request: request.id,
                        status: candidate.status,
                        cancellation_requested: request.runtime.cancellation().is_cancelled()
                            || request.lifecycle.cancellation_pending(),
                        verification_complete: candidate.verification_complete,
                        verification_deadline_expired: candidate.verification_deadline_expired,
                        optimistic_eligible: candidate.optimistic_eligible,
                    })
                    .map_err(SpeculativeDriverError::Generation)?;
            }
            Ok::<_, SpeculativeDriverError<E::Error>>(local)
        })();
        let local = crate::run_preparation::finish_preparation(
            stage,
            local,
            |status| executor.agree_text_preparation(stage, status, context),
            SpeculativeDriverError::Preparation,
        )?;
        let coordinated = executor
            .coordinate_speculative_buffer(local, context)
            .map_err(SpeculativeDriverError::Preparation)
            .and_then(|states| {
                if states.len() != self.requests.len()
                    || states.iter().zip(&self.requests).any(|(state, request)| {
                        state.request != request.id || state.status != request.lifecycle.status()
                    })
                {
                    Err(SpeculativeDriverError::Coordination(
                        "request identity or lifecycle",
                    ))
                } else {
                    Ok(states)
                }
            });
        let coordinated = crate::run_preparation::finish_preparation(
            stage,
            coordinated,
            |status| executor.agree_text_preparation(stage, status, context),
            SpeculativeDriverError::Preparation,
        )?;
        let cancelled = (|| {
            for (state, request) in coordinated.iter().zip(&mut self.requests) {
                if state.cancellation_requested && !request.lifecycle.is_terminal() {
                    request.runtime.cancellation().cancel();
                    request.request_cancellation()?;
                }
            }
            Ok(())
        })();
        crate::run_preparation::finish_preparation(
            stage,
            cancelled,
            |status| executor.agree_text_preparation(stage, status, context),
            SpeculativeDriverError::Preparation,
        )?;
        if self.is_finished() {
            return Ok(false);
        }
        let candidates = (|| {
            let mut candidates = executor.driver_buffer(self.requests.len(), context)?;
            for (state, request) in coordinated.into_iter().zip(&self.requests) {
                candidates
                    .try_push(SpeculativeCandidate {
                        status: request.lifecycle.status(),
                        optimistic_eligible: state.optimistic_eligible
                            && !state.cancellation_requested,
                        verification_complete: state.verification_complete,
                        verification_deadline_expired: state.verification_deadline_expired,
                    })
                    .map_err(SpeculativeDriverError::Generation)?;
            }
            Ok(candidates)
        })();
        let candidates = coordination::ready_result(executor, candidates, context)?;
        let Some(action) = self
            .schedule
            .next_action(&candidates)
            .map_err(SpeculativeDriverError::Generation)?
        else {
            return Ok(false);
        };
        let index = match action {
            SpeculativeAction::SubmitVerification(index)
            | SpeculativeAction::DraftOptimistic(index)
            | SpeculativeAction::PollVerification(index)
            | SpeculativeAction::ResolveVerification(index)
            | SpeculativeAction::DraftCommitted { index, .. } => index,
        };
        let selected = executor
            .request_context(self.requests[index].id, context)
            .map_err(SpeculativeDriverError::Backend);
        let request_context = crate::run_preparation::finish_preparation(
            stage,
            selected,
            |status| executor.agree_text_preparation(stage, status, context),
            SpeculativeDriverError::Preparation,
        )?;
        self.stats.turns += 1;
        self.requests[index].stats.counters.scheduler_turns += 1;
        let completed = (|| {
            match action {
                SpeculativeAction::SubmitVerification(index) => {
                    self.requests[index].submit_verification(executor, request_context)?;
                    let in_flight = self
                        .requests
                        .iter()
                        .filter(|request| request.pending.is_some())
                        .count();
                    self.stats.peak_in_flight_verifications =
                        self.stats.peak_in_flight_verifications.max(in_flight);
                }
                SpeculativeAction::DraftCommitted {
                    index,
                    cross_request,
                } => {
                    let drafted =
                        self.requests[index].draft_committed(executor, request_context)?;
                    if cross_request && drafted {
                        self.requests[index]
                            .stats
                            .counters
                            .cross_request_draft_opportunities += 1;
                        self.stats.cross_request_draft_opportunities += 1;
                    }
                }
                SpeculativeAction::DraftOptimistic(index) => {
                    self.requests[index].draft_optimistic(executor, request_context)?;
                    let optimistic =
                        self.requests
                            .iter()
                            .filter(|request| {
                                request.pending.as_ref().is_some_and(
                                    PendingSpeculativeVerification::has_optimistic_branch,
                                )
                            })
                            .count();
                    self.stats.peak_optimistic_branches =
                        self.stats.peak_optimistic_branches.max(optimistic);
                }
                SpeculativeAction::PollVerification(_) => {}
                SpeculativeAction::ResolveVerification(index) => {
                    self.requests[index].resolve_verification(
                        executor,
                        self.schedule.options(),
                        request_context,
                    )?;
                }
            }
            Ok(true)
        })();
        crate::run_preparation::finish_preparation(
            stage,
            completed,
            |status| executor.agree_text_preparation(stage, status, context),
            SpeculativeDriverError::Preparation,
        )
    }

    /// Drives every request to a terminal state.
    pub fn run<'context>(
        &mut self,
        executor: &mut E,
        optimistic_execution_available: bool,
        context: E::Context<'context>,
    ) -> Result<(), SpeculativeDriverError<E::Error>>
    where
        E: 'context,
        S: SpeculativeSampling<
                Logits = E::Logits,
                Error = E::Error,
                Context<'context> = E::Context<'context>,
            > + 'context,
    {
        while self.step(executor, optimistic_execution_available, context)? {}
        Ok(())
    }

    /// Consumes a terminal table and returns outputs in stable submission order.
    pub fn finish(
        self,
    ) -> Result<CompletedSpeculativeSchedule<S>, SpeculativeDriverError<E::Error>> {
        if !self.is_finished() {
            return Err(SpeculativeDriverError::Generation(
                GenerationError::ActiveSpeculativeRequests,
            ));
        }
        let mut completed = self.completed;
        for request in self.requests {
            let (sampler, sequence, _, _) = request.runtime.into_parts();
            completed
                .try_push(CompletedSpeculativeRequest {
                    id: request.id,
                    finish_reason: sequence.finish_reason(),
                    token_ids: sequence.into_token_ids(),
                    stats: request.stats,
                    sampler,
                    status: request.lifecycle.status(),
                })
                .map_err(SpeculativeDriverError::Generation)?;
        }
        Ok(CompletedSpeculativeSchedule {
            requests: completed,
            scheduler: self.stats,
        })
    }
}

/// Portable candidate snapshot used by fair speculative action selection.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct SpeculativeCandidate {
    /// Current validated request status.
    status: SpeculativeRequestStatus,
    /// Whether this request may start exact optimistic work now.
    optimistic_eligible: bool,
    /// Whether retained verification reached exact completion without blocking.
    verification_complete: bool,
    /// Whether an incomplete verification reached its selected bounded deadline.
    verification_deadline_expired: bool,
}

/// One backend action selected by the portable fair scheduler.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
#[non_exhaustive]
pub enum SpeculativeAction {
    /// Submit a prepared proposal block.
    SubmitVerification(usize),
    /// Draft canonical proposals; the flag records cross-request overlap.
    DraftCommitted {
        /// Selected request index.
        index: usize,
        /// Whether target work from another request is in flight.
        cross_request: bool,
    },
    /// Draft against an unresolved optimistic prefix.
    DraftOptimistic(usize),
    /// Nonblocking observation retained an incomplete verification.
    PollVerification(usize),
    /// Resolve one exact verification completion.
    ResolveVerification(usize),
}

/// Backend-neutral fair action selector for speculative requests.
pub struct SpeculativeSchedule {
    options: SpeculativeSchedulerOptions,
    cursor: usize,
}

impl SpeculativeSchedule {
    /// Creates a validated schedule.
    pub fn new(options: SpeculativeSchedulerOptions) -> Result<Self, GenerationError> {
        Ok(Self {
            options: options.validate()?,
            cursor: 0,
        })
    }

    /// Validated scheduler options.
    pub const fn options(&self) -> SpeculativeSchedulerOptions {
        self.options
    }

    /// Selects the next fair action, or `None` when every request is terminal.
    pub fn next_action(
        &mut self,
        candidates: &[SpeculativeCandidate],
    ) -> Result<Option<SpeculativeAction>, GenerationError> {
        if candidates.iter().all(|candidate| {
            matches!(
                candidate.status,
                SpeculativeRequestStatus::Completed | SpeculativeRequestStatus::Cancelled
            )
        }) {
            return Ok(None);
        }
        let in_flight = candidates
            .iter()
            .filter(|candidate| {
                matches!(
                    candidate.status,
                    SpeculativeRequestStatus::TargetVerificationInFlight
                        | SpeculativeRequestStatus::OptimisticDraftRunning
                        | SpeculativeRequestStatus::OptimisticDraftReady
                        | SpeculativeRequestStatus::VerificationResolution
                )
            })
            .count();
        let optimistic = candidates
            .iter()
            .filter(|candidate| candidate.status == SpeculativeRequestStatus::OptimisticDraftReady)
            .count();

        if in_flight < self.options.max_in_flight_verifications {
            if let Some(index) = self.select(candidates, |candidate| {
                candidate.status == SpeculativeRequestStatus::ReadyToSubmitVerification
            }) {
                return Ok(Some(SpeculativeAction::SubmitVerification(index)));
            }
        }
        if in_flight > 0 {
            if optimistic < self.options.max_optimistic_branches
                && self.options.lookahead_blocks > 0
            {
                if let Some(index) = self.select(candidates, |candidate| {
                    candidate.status == SpeculativeRequestStatus::TargetVerificationInFlight
                        && candidate.optimistic_eligible
                }) {
                    return Ok(Some(SpeculativeAction::DraftOptimistic(index)));
                }
            }
            if let Some(index) = self.select(candidates, |candidate| {
                candidate.status == SpeculativeRequestStatus::ReadyToDraft
            }) {
                return Ok(Some(SpeculativeAction::DraftCommitted {
                    index,
                    cross_request: true,
                }));
            }
            if let Some(index) = self.select(candidates, |candidate| {
                matches!(
                    candidate.status,
                    SpeculativeRequestStatus::TargetVerificationInFlight
                        | SpeculativeRequestStatus::OptimisticDraftReady
                ) && (candidate.verification_complete || candidate.verification_deadline_expired)
            }) {
                return Ok(Some(SpeculativeAction::ResolveVerification(index)));
            }
            if let Some(index) = self.select(candidates, |candidate| {
                matches!(
                    candidate.status,
                    SpeculativeRequestStatus::TargetVerificationInFlight
                        | SpeculativeRequestStatus::OptimisticDraftReady
                ) && !candidate.verification_complete
                    && !candidate.verification_deadline_expired
            }) {
                return Ok(Some(SpeculativeAction::PollVerification(index)));
            }
        } else if let Some(index) = self.select(candidates, |candidate| {
            candidate.status == SpeculativeRequestStatus::ReadyToDraft
        }) {
            return Ok(Some(SpeculativeAction::DraftCommitted {
                index,
                cross_request: false,
            }));
        }
        Err(GenerationError::StalledSpeculativeSchedule)
    }

    fn select(
        &mut self,
        candidates: &[SpeculativeCandidate],
        predicate: impl Fn(&SpeculativeCandidate) -> bool,
    ) -> Option<usize> {
        for offset in 0..candidates.len() {
            let index = (self.cursor + offset) % candidates.len();
            if predicate(&candidates[index]) {
                self.cursor = (index + 1) % candidates.len();
                return Some(index);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    mod origin;
    use super::*;
    use std::{
        cell::{Cell, RefCell},
        convert::Infallible,
        fmt,
        rc::Rc,
        sync::{
            Arc, Mutex,
            atomic::{AtomicUsize, Ordering},
        },
    };

    type TransactionTrace = Rc<RefCell<Vec<&'static str>>>;

    #[test]
    fn speculative_telemetry_preserves_exact_statistics_and_durations() {
        let stats = SpeculativeStats {
            counters: SpeculativeStatsCounters {
                execution_topology: SpeculativeExecutionTopology::CrossDeviceSplit,
                target_tokens: 31,
                draft_tokens: 8,
                accepted_tokens: 5,
                rounds: 2,
                emitted_tokens: 7,
                optimistic_draft_tokens: 9,
                reused_optimistic_tokens: 4,
                discarded_optimistic_tokens: 5,
                adaptive_lookahead_disabled: true,
                optimistic_draft_time: Duration::from_millis(125),
                verification_in_flight_time: Duration::from_millis(375),
                ..SpeculativeStatsCounters::default()
            },
            accept_lens: vec![2, 3].into(),
        };
        assert_eq!(
            crate::speculative_decoding_telemetry(&stats),
            crate::SpeculativeDecodingTelemetry {
                execution_topology: "cross-device-split".into(),
                target_tokens: 31,
                draft_tokens: 8,
                accepted_tokens: 5,
                accept_rate: 0.625,
                rounds: 2,
                accept_lens: vec![2, 3],
                emitted_tokens: 7,
                optimistic_draft_tokens: 9,
                reused_optimistic_tokens: 4,
                discarded_optimistic_tokens: 5,
                adaptive_lookahead_disabled: true,
                optimistic_draft_seconds: 0.125,
                verification_in_flight_seconds: 0.375,
            }
        );
        assert_eq!(
            crate::speculative_decoding_telemetry(&SpeculativeStats::default()).accept_rate,
            0.0
        );
    }

    #[test]
    fn speculative_capability_schema_round_trips_without_backend_identity() {
        let capability = SpeculativeCapability::Unsupported {
            draft_source: SpeculativeDraftSource::Embedded,
            architecture: "future_decoder".into(),
        };
        let json = serde_json::to_string(&capability).unwrap();
        assert_eq!(
            serde_json::from_str::<SpeculativeCapability>(&json).unwrap(),
            capability
        );
        assert!(!json.contains("mlx"));
    }

    #[test]
    fn declared_capability_admits_preparation_without_claiming_execution_readiness() {
        let capability = SpeculativeCapability::Declared {
            draft_source: SpeculativeDraftSource::Separate,
        };

        assert_eq!(
            capability.draft_source(),
            Some(SpeculativeDraftSource::Separate)
        );
        assert!(capability.admits_source(SpeculativeDraftSource::Separate));
        assert!(!capability.admits_source(SpeculativeDraftSource::Embedded));
        assert!(!capability.is_ready_for(SpeculativeDraftSource::Separate));
        assert_eq!(
            serde_json::from_str::<SpeculativeCapability>(
                &serde_json::to_string(&capability).unwrap()
            )
            .unwrap(),
            capability
        );
    }

    #[test]
    fn only_ready_capability_claims_immediate_execution() {
        let ready = SpeculativeCapability::Ready {
            draft_source: SpeculativeDraftSource::Embedded,
        };
        let unsupported = SpeculativeCapability::Unsupported {
            draft_source: SpeculativeDraftSource::Embedded,
            architecture: "example".into(),
        };

        assert!(ready.admits_source(SpeculativeDraftSource::Embedded));
        assert!(ready.is_ready_for(SpeculativeDraftSource::Embedded));
        assert!(!unsupported.admits_source(SpeculativeDraftSource::Embedded));
        assert!(!unsupported.is_ready_for(SpeculativeDraftSource::Embedded));
        assert!(
            !SpeculativeCapability::Unavailable.admits_source(SpeculativeDraftSource::Embedded)
        );
    }

    #[derive(Debug, Clone, Default)]
    struct Done {
        trace: Option<TransactionTrace>,
    }

    impl Completion for Done {
        type Error = Infallible;

        fn is_complete(&self) -> Result<bool, Self::Error> {
            Ok(true)
        }

        fn wait(&self) -> Result<(), Self::Error> {
            if let Some(trace) = &self.trace {
                trace.borrow_mut().push("wait");
            }
            Ok(())
        }
    }

    impl BoundedCompletion for Done {
        fn wait_bounded(
            self,
            _policy: BoundedCompletionWait,
        ) -> Result<BoundedCompletionOutcome, Self::Error> {
            self.wait()?;
            Ok(BoundedCompletionOutcome::Completed)
        }
    }

    #[derive(Clone, Default)]
    struct PortableSemanticState {
        events: Vec<crate::generation::SemanticEvent>,
    }

    impl SemanticState for PortableSemanticState {
        fn fork_owned(&self) -> Result<SemanticStateOwner, SpeculativeOutputError> {
            let mut fork = self.clone();
            fork.events.clear();
            Ok(Box::new(fork).into())
        }

        fn push_token(&mut self, token: u32) -> Result<bool, SpeculativeOutputError> {
            self.events
                .push(crate::generation::SemanticEvent::TextDelta(
                    token.to_string().into(),
                ));
            Ok(false)
        }

        fn finish(&mut self, reason: FinishReason) -> Result<(), SpeculativeOutputError> {
            self.events
                .push(crate::generation::SemanticEvent::Finished { reason });
            Ok(())
        }

        fn cancel(&mut self) -> Result<(), SpeculativeOutputError> {
            self.finish(FinishReason::Cancelled)
        }

        fn take_events(&mut self) -> SpeculativeBuffer<crate::generation::SemanticEvent> {
            std::mem::take(&mut self.events).into()
        }
    }

    #[test]
    fn core_semantic_publisher_commits_and_cancels_without_backend_errors() {
        let published = Rc::new(RefCell::new(Vec::new()));
        let mut constraint =
            SpeculativeSemanticConstraint::semantic(Box::new(PortableSemanticState::default()));
        constraint.push_token(7).unwrap();
        constraint.finish(FinishReason::MaxTokens).unwrap();
        {
            let published = Rc::clone(&published);
            let mut publisher = SpeculativeCallbackPublisher::semantic(move |event| {
                published.borrow_mut().push(event)
            });
            assert!(
                !publisher
                    .publish_committed(
                        &mut constraint,
                        &[7],
                        &GenerationCancellationToken::new(),
                        true,
                    )
                    .unwrap()
            );
        }
        assert_eq!(
            *published.borrow(),
            vec![
                crate::generation::SemanticEvent::TextDelta("7".into()),
                crate::generation::SemanticEvent::Finished {
                    reason: FinishReason::MaxTokens,
                },
            ]
        );

        let cancelled = Rc::new(RefCell::new(Vec::new()));
        let mut constraint =
            SpeculativeSemanticConstraint::semantic(Box::new(PortableSemanticState::default()));
        {
            let cancelled = Rc::clone(&cancelled);
            let mut publisher = SpeculativeCallbackPublisher::semantic(move |event| {
                cancelled.borrow_mut().push(event)
            });
            publisher.publish_cancelled(&mut constraint).unwrap();
        }
        assert_eq!(
            *cancelled.borrow(),
            vec![crate::generation::SemanticEvent::Finished {
                reason: FinishReason::Cancelled,
            }]
        );

        let mut constraint = SpeculativeSemanticConstraint::plain();
        let mut publisher = SpeculativeCallbackPublisher::tokens(|_| {
            Err(SpeculativeOutputError::publication("consumer closed"))
        });
        assert_eq!(
            publisher
                .publish_committed(
                    &mut constraint,
                    &[11],
                    &GenerationCancellationToken::new(),
                    false,
                )
                .unwrap_err(),
            SpeculativeOutputError::publication("consumer closed")
        );
    }

    #[derive(Default)]
    struct MockExecutor {
        host: Option<crate::HostPreparationAuthority>,
        trace: Option<TransactionTrace>,
        full_acceptance: bool,
        cancel_during_prefill: Option<GenerationCancellationToken>,
        peer_cancels_prefill: bool,
        capture_origins: bool,
        origin: Option<SpeculativeActivationOrigin>,
        activations: Vec<(&'static str, Option<SpeculativeActivationOrigin>)>,
    }

    struct MockVerification {
        tokens: Vec<u32>,
        logits: Vec<Vec<f32>>,
    }

    impl SpeculativeExecutor for MockExecutor {
        type Input = Vec<u32>;
        type Cache = Vec<u32>;
        type TargetState = usize;
        type DraftState = Vec<u32>;

        fn copy_draft_state<'a>(
            &self,
            state: &Self::DraftState,
            _context: Self::Context<'a>,
        ) -> Result<Self::DraftState, Self::Error>
        where
            Self: 'a,
        {
            Ok(state.clone())
        }
        type CacheCheckpoint = usize;
        type Verification = MockVerification;
        type Logits = Vec<f32>;
        type Context<'a> = ();
        type Completion = Done;
        type Telemetry = ();
        type Error = Infallible;

        fn driver_buffer<T>(
            &self,
            capacity: usize,
            _: (),
        ) -> Result<SpeculativeBuffer<T>, Infallible> {
            Ok(match &self.host {
                Some(host) => SpeculativeBuffer::try_new_retained(capacity, host.clone()).unwrap(),
                None => SpeculativeBuffer::with_capacity(capacity),
            })
        }
        fn driver_identity(&self, _: ()) -> Result<SpeculativeRequestIdentity, Infallible> {
            Ok(match &self.host {
                Some(host) => SpeculativeRequestIdentity::with_authority(host.clone()),
                None => SpeculativeRequestIdentity::new(),
            })
        }
        fn driver_host_metadata(
            &self,
            _: Option<usize>,
            _: (),
        ) -> Result<crate::HostPreparationAuthority, Infallible> {
            Ok(self
                .host
                .clone()
                .unwrap_or_else(crate::HostPreparationAuthority::unmanaged))
        }
        fn coordinate_speculative_buffer<'a>(
            &mut self,
            local: SpeculativeBuffer<SpeculativeScheduleState>,
            _: (),
        ) -> Result<SpeculativeBuffer<SpeculativeScheduleState>, crate::BackendFailure> {
            Ok(local) // this fixture's same existing single-participant coordinator
        }

        fn agree_text_preparation<'a>(
            &mut self,
            _: crate::run_preparation::TextPreparationStage,
            status: crate::run_preparation::TextPreparationStatus,
            _: Self::Context<'a>,
        ) -> Result<crate::run_preparation::TextPreparationOutcome, crate::BackendFailure> {
            use crate::run_preparation::{TextPreparationOutcome as O, TextPreparationStatus as S};
            if self.peer_cancels_prefill
                && self
                    .activations
                    .last()
                    .is_some_and(|(phase, _)| *phase == "prefill")
            {
                self.peer_cancels_prefill = false;
                return Ok(O::Cancelled);
            }
            Ok(match status {
                S::Ready => O::Ready,
                S::Cancelled => O::Cancelled,
                S::Failed => O::Rejected { rank: 0 },
            })
        }

        fn requires_activation_origin(&self) -> bool {
            self.capture_origins
        }

        fn set_activation_origin(&mut self, origin: Option<SpeculativeActivationOrigin>) {
            self.origin = origin;
        }

        fn supports_exact_optimistic_promotion(&self) -> bool {
            true
        }

        fn prefill<'context>(
            &mut self,
            input: Self::Input,
            cache: &mut Self::Cache,
            _: Self::Context<'context>,
        ) -> Result<SpeculativePrefill<Self::TargetState, Self::Logits>, Self::Error> {
            self.activations.push(("prefill", self.origin));
            cache.extend_from_slice(&input);
            if let Some(token) = &self.cancel_during_prefill {
                token.cancel();
            }
            Ok(SpeculativePrefill {
                logits: vec![0.0, 1.0],
                state: cache.len(),
                evaluated_tokens: input.len(),
            })
        }

        fn begin_proposal<'a>(
            &mut self,
            _: &Self::TargetState,
            last_token: u32,
            _: usize,
            _: Self::Context<'a>,
        ) -> Result<Self::DraftState, Self::Error> {
            self.activations.push(("begin", self.origin));
            Ok(vec![last_token])
        }

        fn proposal_logits<'a>(
            &mut self,
            state: &mut Self::DraftState,
            last_token: u32,
            _: Self::Context<'a>,
        ) -> Result<Self::Logits, Self::Error> {
            self.activations.push(("proposal", self.origin));
            state.push(last_token + 1);
            Ok(vec![0.0, 1.0])
        }

        fn checkpoint(&self, cache: &Self::Cache) -> Result<Self::CacheCheckpoint, Self::Error> {
            Ok(cache.len())
        }

        fn restore_checkpoint<'a>(
            &mut self,
            cache: &mut Self::Cache,
            checkpoint: &Self::CacheCheckpoint,
            _: Self::Context<'a>,
        ) -> Result<(), Self::Error> {
            cache.truncate(*checkpoint);
            Ok(())
        }

        fn submit_verification<'a>(
            &mut self,
            input_tokens: &[u32],
            cache: &mut Self::Cache,
            _: Self::Context<'a>,
        ) -> Result<Submission<Self::Verification, Self::Completion>, Self::Error> {
            self.activations.push(("verify", self.origin));
            cache.extend_from_slice(input_tokens);
            let logits = if self.full_acceptance {
                vec![vec![0.0, 1.0], vec![0.0, 1.0], vec![0.0, 1.0]]
            } else {
                vec![vec![0.0, 1.0], vec![1.0, 0.0], vec![0.0, 1.0]]
            };
            Ok(Submission {
                output: MockVerification {
                    tokens: input_tokens.to_vec(),
                    logits,
                },
                completion: Done {
                    trace: self.trace.clone(),
                },
            })
        }

        fn verification_logits<'a>(
            &self,
            output: &Self::Verification,
            index: usize,
            _: Self::Context<'a>,
        ) -> Result<Self::Logits, Self::Error> {
            Ok(output.logits[index].clone())
        }

        fn commit_verification<'a>(
            &mut self,
            output: Self::Verification,
            draft_state: Self::DraftState,
            cache: &mut Self::Cache,
            checkpoint: &Self::CacheCheckpoint,
            verified_inputs: usize,
            _: Self::Context<'a>,
        ) -> Result<SpeculativeCommit<Self::TargetState>, Self::Error> {
            assert!(!output.tokens.is_empty());
            if let Some(trace) = &self.trace {
                trace.borrow_mut().push("commit");
            }
            self.activations.push(("replay", self.origin));
            cache.truncate(*checkpoint + verified_inputs);
            Ok(SpeculativeCommit {
                state: draft_state.len(),
                replayed_tokens: 0,
            })
        }
    }

    #[test]
    fn mock_executor_prefill_propose_verify_and_commit_without_a_tensor_runtime() {
        let mut executor = MockExecutor::default();
        let mut cache = Vec::new();
        let prefill = executor.prefill(vec![4, 5], &mut cache, ()).unwrap();
        let mut draft = executor.begin_proposal(&prefill.state, 5, 2, ()).unwrap();
        assert_eq!(
            executor.proposal_logits(&mut draft, 5, ()).unwrap(),
            [0.0, 1.0]
        );
        let checkpoint = executor.checkpoint(&cache).unwrap();
        let submission = executor
            .submit_verification(&[5, 6], &mut cache, ())
            .unwrap();
        submission.completion.wait().unwrap();
        let commit = executor
            .commit_verification(submission.output, draft, &mut cache, &checkpoint, 1, ())
            .unwrap();
        assert_eq!(cache, [4, 5, 5]);
        assert_eq!(commit.replayed_tokens, 0);
    }

    #[test]
    fn execution_topology_is_a_portable_schema() {
        let topology = SpeculativeExecutionTopology::CrossDeviceSplit;
        let encoded = serde_json::to_string(&topology).unwrap();
        assert_eq!(encoded, "\"cross_device_split\"");
        assert_eq!(
            serde_json::from_str::<SpeculativeExecutionTopology>(&encoded).unwrap(),
            topology
        );
    }

    #[derive(Clone)]
    struct MockSampling {
        committed: Vec<u32>,
        trace: Option<TransactionTrace>,
        unit_draw: f32,
        draft_prefix_limit: Option<usize>,
    }

    impl Default for MockSampling {
        fn default() -> Self {
            Self {
                committed: Vec::new(),
                trace: None,
                unit_draw: 0.5,
                draft_prefix_limit: None,
            }
        }
    }

    impl MockSampling {
        fn record(&self, operation: &'static str) {
            if let Some(trace) = &self.trace {
                trace.borrow_mut().push(operation);
            }
        }
    }

    impl SpeculativeSampling for MockSampling {
        type Logits = Vec<f32>;
        type Distribution = Vec<f32>;
        type Seed = ();
        type RandomState = usize;
        type DraftRandomness = usize;
        type RandomnessRoot = usize;
        type Context<'a> = ();
        type Error = Infallible;

        fn supports_exact_optimistic_promotion(&self) -> bool {
            true
        }

        fn prefix_is_complete(&self, history: &[u32]) -> Result<bool, Self::Error> {
            Ok(self
                .draft_prefix_limit
                .is_some_and(|limit| history.len() >= limit))
        }

        fn randomness_root<'a>(
            _: Option<Self::Seed>,
            _: Self::Context<'a>,
        ) -> Result<Self::RandomnessRoot, Self::Error>
        where
            Self: 'a,
        {
            Ok(0)
        }

        fn target_randomness_from_root<'a>(
            root: &mut Self::RandomnessRoot,
            _: Self::Context<'a>,
        ) -> Result<Self::RandomState, Self::Error>
        where
            Self: 'a,
        {
            let target = *root;
            *root += 1;
            Ok(target)
        }

        fn draft_randomness_from_root<'a>(
            root: &mut Self::RandomnessRoot,
            _: Self::Context<'a>,
        ) -> Result<Self::DraftRandomness, Self::Error>
        where
            Self: 'a,
        {
            let draft = *root;
            *root += 1;
            Ok(draft)
        }

        fn draft_randomness_at<'a>(
            root: &Self::DraftRandomness,
            position: SpeculativeDraftRandomPosition,
            _: Self::Context<'a>,
        ) -> Result<Self::RandomState, Self::Error>
        where
            Self: 'a,
        {
            Ok(root + position.get())
        }

        fn process_logits<'a>(
            &mut self,
            logits: &Self::Logits,
            _: f32,
            _: &[u32],
            _: SamplingPlacement,
            _: Self::Context<'a>,
        ) -> Result<Self::Distribution, Self::Error>
        where
            Self: 'a,
        {
            Ok(logits.clone())
        }

        fn sample<'a>(
            &self,
            distribution: &Self::Distribution,
            _: f32,
            randomness: Option<&mut Self::RandomState>,
            _: SamplingPlacement,
            _: Self::Context<'a>,
        ) -> Result<u32, Self::Error>
        where
            Self: 'a,
        {
            self.record("sample");
            if let Some(randomness) = randomness {
                *randomness += 1;
            }
            Ok(argmax(distribution))
        }

        fn probability_at<'a>(
            &self,
            distribution: &Self::Distribution,
            token: u32,
            _: SamplingPlacement,
            _: Self::Context<'a>,
        ) -> Result<f32, Self::Error>
        where
            Self: 'a,
        {
            self.record("probability");
            let maximum = distribution
                .iter()
                .copied()
                .max_by(f32::total_cmp)
                .unwrap_or(0.0);
            let normalizer = distribution
                .iter()
                .map(|value| (value - maximum).exp())
                .sum::<f32>();
            Ok((distribution[token as usize] - maximum).exp() / normalizer)
        }

        fn sample_unit_interval<'a>(
            &self,
            randomness: Option<&mut Self::RandomState>,
            _: Self::Context<'a>,
        ) -> Result<f32, Self::Error>
        where
            Self: 'a,
        {
            self.record("uniform");
            if let Some(randomness) = randomness {
                *randomness += 1;
            }
            Ok(self.unit_draw)
        }

        fn positive_probability_difference<'a>(
            &self,
            left: &Self::Distribution,
            right: &Self::Distribution,
            _: SamplingPlacement,
            _: Self::Context<'a>,
        ) -> Result<Option<Self::Distribution>, Self::Error>
        where
            Self: 'a,
        {
            self.record("difference");
            let probabilities = |distribution: &[f32]| {
                let maximum = distribution
                    .iter()
                    .copied()
                    .max_by(f32::total_cmp)
                    .unwrap_or(0.0);
                let values = distribution
                    .iter()
                    .map(|value| (value - maximum).exp())
                    .collect::<Vec<_>>();
                let normalizer = values.iter().sum::<f32>();
                values
                    .into_iter()
                    .map(|value| value / normalizer)
                    .collect::<Vec<_>>()
            };
            let left = probabilities(left);
            let right = probabilities(right);
            let difference = left
                .iter()
                .zip(right)
                .map(|(left, right)| (left - right).max(0.0))
                .collect::<Vec<_>>();
            Ok(difference
                .iter()
                .any(|value| *value > f32::EPSILON)
                .then_some(difference))
        }

        fn update_sampler_state<'a>(
            &mut self,
            _: &Self::Distribution,
            token: u32,
            _: SamplingPlacement,
            _: Self::Context<'a>,
        ) -> Result<(), Self::Error>
        where
            Self: 'a,
        {
            self.record("update");
            self.committed.push(token);
            Ok(())
        }
    }

    fn argmax(values: &[f32]) -> u32 {
        values
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.total_cmp(right))
            .map(|(index, _)| index as u32)
            .unwrap()
    }

    #[derive(Default)]
    struct MockConstraint {
        tokens: Vec<u32>,
        finished: Option<FinishReason>,
    }

    impl SpeculativeConstraint for MockConstraint {
        fn fork(&self) -> Result<Self, SpeculativeOutputError> {
            Ok(Self {
                tokens: self.tokens.clone(),
                finished: self.finished,
            })
        }

        fn push_token(&mut self, token: u32) -> Result<bool, SpeculativeOutputError> {
            self.tokens.push(token);
            Ok(false)
        }

        fn finish(&mut self, reason: FinishReason) -> Result<(), SpeculativeOutputError> {
            self.finished = Some(reason);
            Ok(())
        }
    }

    #[derive(Default)]
    struct MockPublisher {
        tokens: Vec<u32>,
        cancelled: bool,
        trace: Option<TransactionTrace>,
    }

    impl SpeculativePublisher<MockConstraint> for MockPublisher {
        fn publish_committed(
            &mut self,
            _: &mut MockConstraint,
            tokens: &[u32],
            _: &GenerationCancellationToken,
            _: bool,
        ) -> Result<bool, SpeculativeOutputError> {
            if let Some(trace) = &self.trace {
                trace.borrow_mut().push("publish");
            }
            self.tokens.extend_from_slice(tokens);
            Ok(false)
        }

        fn publish_cancelled(
            &mut self,
            _: &mut MockConstraint,
        ) -> Result<(), SpeculativeOutputError> {
            if let Some(trace) = &self.trace {
                trace.borrow_mut().push("cancel");
            }
            self.cancelled = true;
            Ok(())
        }
    }

    fn mock_output_runtime(
        cancellation: GenerationCancellationToken,
        trace: Option<TransactionTrace>,
    ) -> SpeculativeOutputRuntime<MockSampling, MockConstraint, MockPublisher> {
        let mut sequence = GenerationSequence::new(8, []);
        sequence.commit(5, TokenTerminalSignals::default()).unwrap();
        SpeculativeOutputRuntime::new(
            MockSampling::default(),
            sequence,
            MockConstraint::default(),
            MockPublisher {
                trace,
                ..MockPublisher::default()
            },
            cancellation,
        )
    }

    fn empty_mock_runtime(
        max_tokens: usize,
        cancellation: GenerationCancellationToken,
    ) -> SpeculativeOutputRuntime<MockSampling, MockConstraint, MockPublisher> {
        SpeculativeOutputRuntime::new(
            MockSampling::default(),
            GenerationSequence::new(max_tokens, []),
            MockConstraint::default(),
            MockPublisher::default(),
            cancellation,
        )
    }

    #[test]
    fn snapshot_configuration_copies_before_retention_and_rejects_unknown_callbacks() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        #[derive(Debug)]
        struct Custody(Arc<AtomicUsize>);
        impl Drop for Custody {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }
        struct UnknownConstraint(std::rc::Rc<std::cell::Cell<usize>>);
        impl SpeculativeConstraint for UnknownConstraint {
            fn control_snapshot_bytes(&self) -> Option<u64> {
                Some(0)
            }
            fn fork(&self) -> Result<Self, SpeculativeOutputError> {
                self.0.set(self.0.get() + 1);
                Ok(Self(self.0.clone()))
            }
            fn push_token(&mut self, _: u32) -> Result<bool, SpeculativeOutputError> {
                Ok(false)
            }
            fn finish(&mut self, _: FinishReason) -> Result<(), SpeculativeOutputError> {
                Ok(())
            }
        }
        let source = SpeculativeConfig {
            max_tokens: 17,
            max_draft_tokens: 4,
            temperature: 0.0,
            eos_token_ids: vec![3, 11],
        };
        let ordinary_pointer = source.eos_token_ids.as_ptr();
        let expected_geometry = SpeculativeRequestGeometry::new(&source, 2);
        let ordinary = RequestConfiguration::prepare(source, &MockExecutor::default(), ()).unwrap();
        assert_eq!(ordinary.eos_token_ids.as_ptr(), ordinary_pointer);
        assert_eq!(ordinary.geometry(2), expected_geometry);

        let retired = Arc::new(AtomicUsize::new(0));
        let host = crate::HostPreparationAuthority::retain(Custody(retired.clone()));
        let executor = MockExecutor {
            host: Some(host.clone()),
            ..MockExecutor::default()
        };
        let source = SpeculativeConfig {
            max_tokens: 17,
            max_draft_tokens: 4,
            temperature: 0.0,
            eos_token_ids: vec![3, 11],
        };
        let input_pointer = source.eos_token_ids.as_ptr();
        let config = RequestConfiguration::prepare(source, &executor, ()).unwrap();
        assert_ne!(config.eos_token_ids.as_ptr(), input_pointer);
        assert_eq!(
            config.eos_token_ids.as_slice(),
            ordinary.eos_token_ids.as_slice()
        );
        assert_eq!(config.geometry(2), expected_geometry);
        let snapshot = config.clone();
        assert_eq!(
            snapshot.eos_token_ids.as_ptr(),
            config.eos_token_ids.as_ptr()
        );

        let forks = std::rc::Rc::new(std::cell::Cell::new(0));
        let unknown = UnknownConstraint(forks.clone());
        assert_eq!(unknown.control_snapshot_bytes(), Some(0));
        assert_eq!(unknown.control_snapshot_metadata_bytes(), None);
        assert!(matches!(
            unknown.fork_control_snapshot(host.clone()),
            Err(SpeculativeControlError::Unsupported(_))
        ));
        assert_eq!(forks.get(), 0);
        unknown
            .fork_control_snapshot(crate::HostPreparationAuthority::unmanaged())
            .unwrap();
        assert_eq!(forks.get(), 1);
        let sampler = MockSampling {
            committed: vec![2, 7],
            ..MockSampling::default()
        };
        assert!(matches!(
            sampler.copy_control_snapshot(Some(&5), Some(&9), host.clone()),
            Err(SpeculativeControlError::Unsupported(_))
        ));
        let (copied, target, draft) = sampler
            .copy_control_snapshot(
                Some(&5),
                Some(&9),
                crate::HostPreparationAuthority::unmanaged(),
            )
            .unwrap();
        assert_eq!(copied.committed, sampler.committed);
        assert_eq!((target, draft), (Some(5), Some(9)));
        let plain = SpeculativeSemanticConstraint::plain();
        assert!(plain.control_snapshot_metadata_bytes().is_some());
        plain.fork_control_snapshot(host.clone()).unwrap();

        drop(config);
        drop(executor);
        drop(host);
        assert_eq!(retired.load(Ordering::SeqCst), 0);
        assert_eq!(snapshot.eos_token_ids.as_slice(), &[3, 11]);
        drop(snapshot);
        assert_eq!(retired.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn controlled_record_values_keep_wire_arrays_and_shared_custody() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        #[derive(Debug)]
        struct Custody(Arc<AtomicUsize>);
        impl Drop for Custody {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }
        #[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
        #[serde(transparent)]
        struct Row(u32);
        impl Clone for Row {
            fn clone(&self) -> Self {
                panic!("retained record clone must not clone its payload")
            }
        }
        let retired = Arc::new(AtomicUsize::new(0));
        let host = crate::HostPreparationAuthority::retain(Custody(retired.clone()));
        let executor = MockExecutor {
            host: Some(host.clone()),
            ..MockExecutor::default()
        };
        let values = SpeculativeValues::collect_with_metadata(
            [Row(2), Row(5)].into_iter(),
            &executor,
            (),
            0,
        )
        .unwrap();
        let alias = values.clone();
        assert_eq!(serde_json::to_string(&alias).unwrap(), "[2,5]");
        let decoded: SpeculativeValues<Row> = serde_json::from_str("[2,5]").unwrap();
        assert_eq!(alias, decoded);
        let tokens =
            SpeculativeValues::collect_with_metadata([4, 7, 9].into_iter(), &executor, (), 0)
                .unwrap();
        let mut iter = tokens.into_iter();
        assert_eq!(iter.next(), Some(4));
        assert_eq!(iter.next_back(), Some(9));
        drop(values);
        drop(executor);
        drop(host);
        assert_eq!(retired.load(Ordering::SeqCst), 0);
        assert_eq!(serde_json::to_string(&alias).unwrap(), "[2,5]");
        drop(alias);
        assert_eq!(retired.load(Ordering::SeqCst), 0);
        assert_eq!(iter.next(), Some(7));
        assert_eq!(iter.next(), None);
        drop(iter);
        assert_eq!(retired.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn fixed_request_and_stats_destinations_match_scheduler_and_keep_terminal_custody() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        #[derive(Debug)]
        struct Custody(Arc<AtomicUsize>);
        impl Drop for Custody {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }
        fn run(
            host: Option<crate::HostPreparationAuthority>,
        ) -> CompletedSpeculativeSchedule<MockSampling> {
            let mut executor = MockExecutor {
                host,
                full_acceptance: true,
                ..MockExecutor::default()
            };
            let mut cache = Vec::new();
            let mut table = SpeculativeRequestTable::new(
                SpeculativeSchedulerOptions::default().with_lookahead(false),
                SpeculativeExecutionTopology::Single,
            )
            .unwrap();
            table.reserve_requests(1, &executor, ()).unwrap();
            table
                .submit(
                    &mut executor,
                    &mut cache,
                    vec![4],
                    SpeculativeConfig {
                        max_tokens: 5,
                        max_draft_tokens: 2,
                        temperature: 0.7,
                        eos_token_ids: Vec::new(),
                    },
                    empty_mock_runtime(5, GenerationCancellationToken::new()),
                    SpeculativeRandomness {
                        target: Some(0),
                        draft: Some(0),
                    },
                    false,
                    (),
                )
                .unwrap();
            table.run(&mut executor, false, ()).unwrap();
            let completed = table.finish().unwrap();
            drop(executor);
            completed
        }
        let mut ordinary = run(None).take_requests();
        let retired = Arc::new(AtomicUsize::new(0));
        let mut retained = run(Some(crate::HostPreparationAuthority::retain(Custody(
            retired.clone(),
        ))))
        .take_requests();
        let ordinary_request = ordinary.pop().unwrap();
        let retained_request = retained.pop().unwrap();
        drop(ordinary);
        drop(retained);
        let ordinary = ordinary_request;
        let retained = retained_request;
        assert_eq!(retained.token_ids(), ordinary.token_ids());
        assert_eq!(
            retained.stats().accept_lens(),
            ordinary.stats().accept_lens()
        );
        assert_eq!(retained.stats().rounds(), ordinary.stats().rounds());
        let mut artifact = retained.into_artifact();
        let stats = artifact.take_stats();
        drop(artifact);
        assert_eq!(retired.load(Ordering::SeqCst), 0);
        // The returned history owns its constructor custody after both the
        // request table and completed-row allocation have retired.
        drop(stats);
        assert_eq!(retired.load(Ordering::SeqCst), 1);
    }

    #[derive(Default)]
    struct LifecycleTrace {
        stages: Mutex<Vec<SpeculativeLifecycleStage>>,
        fail: Option<SpeculativeLifecycleStage>,
    }

    impl LifecycleTrace {
        fn failing(stage: SpeculativeLifecycleStage) -> Self {
            Self {
                stages: Mutex::default(),
                fail: Some(stage),
            }
        }

        fn stages(&self) -> Vec<SpeculativeLifecycleStage> {
            self.stages.lock().unwrap().clone()
        }
    }

    impl SpeculativeLifecycleObserver for LifecycleTrace {
        fn observe(&self, stage: SpeculativeLifecycleStage) -> Result<(), SpeculativeOutputError> {
            self.stages.lock().unwrap().push(stage);
            if self.fail == Some(stage) {
                Err(SpeculativeOutputError::semantic(
                    "lifecycle observation",
                    format!("injected {stage:?} failure"),
                ))
            } else {
                Ok(())
            }
        }
    }

    #[test]
    fn request_table_consumes_explicit_production_lifecycle_observation() {
        let observer = Arc::new(LifecycleTrace::default());
        let mut executor = MockExecutor::default();
        let mut cache = Vec::new();
        let config = SpeculativeConfig {
            max_tokens: 3,
            max_draft_tokens: 2,
            temperature: 0.7,
            eos_token_ids: Vec::new(),
        };
        let runtime = empty_mock_runtime(config.max_tokens, GenerationCancellationToken::new())
            .with_lifecycle_observer(observer.clone());
        let mut table = SpeculativeRequestTable::new(
            SpeculativeSchedulerOptions::default().with_lookahead(false),
            SpeculativeExecutionTopology::Single,
        )
        .unwrap();
        table
            .submit(
                &mut executor,
                &mut cache,
                vec![4],
                config,
                runtime,
                SpeculativeRandomness {
                    target: Some(0),
                    draft: Some(0),
                },
                false,
                (),
            )
            .unwrap();
        table.run(&mut executor, false, ()).unwrap();
        table.finish().unwrap();

        let stages = observer.stages();
        assert_eq!(stages[0], SpeculativeLifecycleStage::Input);
        assert_eq!(stages[1], SpeculativeLifecycleStage::Execution);
        assert_eq!(stages[2], SpeculativeLifecycleStage::Publication);
        assert_eq!(
            stages
                .iter()
                .filter(|stage| **stage == SpeculativeLifecycleStage::Observation)
                .count(),
            1
        );
        let observation = stages
            .iter()
            .position(|stage| *stage == SpeculativeLifecycleStage::Observation)
            .unwrap();
        let persistence = stages
            .iter()
            .position(|stage| *stage == SpeculativeLifecycleStage::CachePersistence)
            .unwrap();
        let final_publication = stages
            .iter()
            .rposition(|stage| *stage == SpeculativeLifecycleStage::Publication)
            .unwrap();
        assert!(observation < persistence);
        assert!(persistence < final_publication);
    }

    #[test]
    fn cancelled_prefill_rolls_back_before_sampling_in_run_and_step_registration() {
        for stepped in [false, true] {
            for peer in [false, true] {
                let cancel = GenerationCancellationToken::new();
                let trace = TransactionTrace::default();
                let mut executor = MockExecutor {
                    cancel_during_prefill: (!peer).then(|| cancel.clone()),
                    peer_cancels_prefill: peer,
                    ..Default::default()
                };
                // The compatibility executor mutates its cache before returning;
                // cancellation becomes visible only at the shared post-prefill vote.
                let mut cache = vec![9];
                let runtime = SpeculativeOutputRuntime::new(
                    MockSampling {
                        trace: Some(trace.clone()),
                        ..Default::default()
                    },
                    GenerationSequence::new(7, []),
                    MockConstraint::default(),
                    MockPublisher {
                        trace: Some(trace.clone()),
                        ..Default::default()
                    },
                    cancel.clone(),
                );
                let mut table = SpeculativeRequestTable::new(
                    SpeculativeSchedulerOptions::default(),
                    SpeculativeExecutionTopology::Single,
                )
                .unwrap();
                let id = table
                    .submit(
                        &mut executor,
                        &mut cache,
                        vec![1, 3, 2, 4, 5],
                        SpeculativeConfig {
                            max_tokens: 7,
                            max_draft_tokens: 2,
                            temperature: 0.7,
                            eos_token_ids: vec![],
                        },
                        runtime,
                        SpeculativeRandomness {
                            target: Some(0),
                            draft: Some(0),
                        },
                        false,
                        (),
                    )
                    .unwrap();
                assert_eq!(table.status(id), Some(SpeculativeRequestStatus::Cancelled));
                if stepped {
                    assert!(!table.step(&mut executor, false, ()).unwrap());
                } else {
                    table.run(&mut executor, false, ()).unwrap();
                }
                let mut completed = table.finish().unwrap().into_requests();
                let completed = completed.pop().unwrap();
                assert_eq!(completed.stats.target_tokens(), 5);
                assert_eq!(completed.stats.emitted_tokens(), 0);
                assert_eq!(completed.stats.draft_tokens(), 0);
                assert!(completed.token_ids.is_empty());
                assert!(completed.sampler.committed.is_empty());
                assert_eq!(completed.finish_reason, Some(FinishReason::Cancelled));
                assert!(cancel.is_cancelled());
                assert_eq!(cache, [9]);
                assert_eq!(*trace.borrow(), ["cancel"]);
                assert_eq!(
                    executor
                        .activations
                        .iter()
                        .map(|(name, _)| *name)
                        .collect::<Vec<_>>(),
                    ["prefill"]
                );
            }
        }
    }

    #[test]
    fn input_and_execution_observer_failures_prevent_prefill_and_publication() {
        for failure in [
            SpeculativeLifecycleStage::Input,
            SpeculativeLifecycleStage::Execution,
        ] {
            let observer = Arc::new(LifecycleTrace::failing(failure));
            let publication = TransactionTrace::default();
            let runtime = SpeculativeOutputRuntime::new(
                MockSampling::default(),
                GenerationSequence::new(3, []),
                MockConstraint::default(),
                MockPublisher {
                    trace: Some(publication.clone()),
                    ..MockPublisher::default()
                },
                GenerationCancellationToken::new(),
            )
            .with_lifecycle_observer(observer.clone());
            let mut executor = MockExecutor::default();
            let mut cache = Vec::new();
            let mut table = SpeculativeRequestTable::new(
                SpeculativeSchedulerOptions::default().with_lookahead(false),
                SpeculativeExecutionTopology::Single,
            )
            .unwrap();
            let error = table
                .submit(
                    &mut executor,
                    &mut cache,
                    vec![4],
                    SpeculativeConfig {
                        max_tokens: 3,
                        max_draft_tokens: 2,
                        temperature: 0.7,
                        eos_token_ids: Vec::new(),
                    },
                    runtime,
                    SpeculativeRandomness {
                        target: Some(0),
                        draft: Some(0),
                    },
                    false,
                    (),
                )
                .unwrap_err();
            assert!(matches!(error, SpeculativeDriverError::Output(_)));
            assert!(cache.is_empty());
            assert!(publication.borrow().is_empty());
            assert_eq!(observer.stages().last(), Some(&failure));
        }
    }

    #[test]
    fn deterministic_proposal_policy_uses_only_target_selection() {
        let trace = TransactionTrace::default();
        let sampler = MockSampling {
            trace: Some(trace.clone()),
            ..MockSampling::default()
        };
        let mut randomness = 7;
        let decision = decide_speculative_proposal(
            &sampler,
            &vec![2.0, 0.0],
            &vec![0.0, 2.0],
            1,
            0.0,
            Some(&mut randomness),
            (),
        )
        .unwrap();

        assert_eq!(decision, ProposalDecision::Reject(0));
        assert_eq!(randomness, 7);
        assert_eq!(*trace.borrow(), ["sample"]);
    }

    #[test]
    fn neutral_randomness_assigns_target_then_position_stable_draft() {
        let randomness = MockSampling::initialize_randomness(Some(()), 0.7, ()).unwrap();
        assert_eq!(randomness.target, Some(0));
        assert_eq!(randomness.draft, Some(1));
        assert_eq!(
            MockSampling::draft_randomness_at(
                randomness.draft.as_ref().unwrap(),
                SpeculativeDraftRandomPosition::new(4),
                (),
            )
            .unwrap(),
            5
        );

        let deterministic = MockSampling::initialize_randomness(None, 0.0, ()).unwrap();
        assert_eq!(deterministic.target, None);
        assert_eq!(deterministic.draft, None);
    }

    #[test]
    fn stochastic_proposal_policy_causally_selects_acceptance_or_residual() {
        assert_eq!(speculative_acceptance_probability(0.25, 0.5), 0.5);
        assert_eq!(speculative_acceptance_probability(0.25, 0.0), 1.0);

        let accepted_trace = TransactionTrace::default();
        let accepted_sampler = MockSampling {
            trace: Some(accepted_trace.clone()),
            unit_draw: 0.9,
            ..MockSampling::default()
        };
        let mut accepted_randomness = 0;
        let accepted = decide_speculative_proposal(
            &accepted_sampler,
            &vec![0.0, 1.0],
            &vec![0.0, 1.0],
            1,
            0.7,
            Some(&mut accepted_randomness),
            (),
        )
        .unwrap();
        assert_eq!(accepted, ProposalDecision::Accept);
        assert_eq!(accepted_randomness, 1);
        assert_eq!(
            *accepted_trace.borrow(),
            ["probability", "probability", "uniform"]
        );

        let rejected_trace = TransactionTrace::default();
        let rejected_sampler = MockSampling {
            trace: Some(rejected_trace.clone()),
            unit_draw: 0.5,
            ..MockSampling::default()
        };
        let mut rejected_randomness = 0;
        let rejected = decide_speculative_proposal(
            &rejected_sampler,
            &vec![0.0, 2.0],
            &vec![2.0, 0.0],
            0,
            0.7,
            Some(&mut rejected_randomness),
            (),
        )
        .unwrap();
        assert_eq!(rejected, ProposalDecision::Reject(1));
        assert_eq!(rejected_randomness, 2);
        assert_eq!(
            *rejected_trace.borrow(),
            [
                "probability",
                "probability",
                "uniform",
                "difference",
                "sample"
            ]
        );
    }

    #[test]
    fn portable_driver_proposes_and_resolves_acceptance_and_replacement() {
        let mut executor = MockExecutor::default();
        let sampler = MockSampling::default();
        let mut draft = executor.begin_proposal(&2, 5, 2, ()).unwrap();
        let proposals = propose_block(
            &mut executor,
            &sampler,
            &mut draft,
            5,
            2,
            &[5],
            0.7,
            &[],
            Some(&0),
            (),
        )
        .unwrap();
        assert_eq!(
            proposals
                .iter()
                .map(|proposal| proposal.token)
                .collect::<Vec<_>>(),
            [1, 1]
        );

        let mut cache = vec![4, 5];
        let verification = executor
            .submit_verification(&[5, 1, 1], &mut cache, ())
            .unwrap()
            .output;
        let mut sequence = GenerationSequence::new(8, []);
        sequence.commit(5, TokenTerminalSignals::default()).unwrap();
        let canonical_randomness = 0;
        let resolved = resolve_round::<MockExecutor, MockSampling, MockConstraint>(
            &executor,
            &verification,
            proposals,
            &sampler,
            &sequence,
            &MockConstraint::default(),
            Some(&canonical_randomness),
            0.7,
            (),
        )
        .unwrap();
        assert_eq!(resolved.accepted_proposals, 1);
        assert_eq!(resolved.committed_tokens, [1, 0]);
        assert_eq!(resolved.verified_inputs, 2);
        assert_eq!(resolved.sampler.committed, [1, 0]);
        assert_eq!(resolved.sequence.tokens(), [5, 1, 0]);
        assert_eq!(resolved.constraint.tokens, [1, 0]);
        assert_eq!(resolved.target_randomness, Some(3));
        assert_eq!(resolved.finish_reason, None);
        assert!(sampler.committed.is_empty());
        assert_eq!(canonical_randomness, 0);
    }

    #[test]
    fn portable_schedule_is_fair_and_respects_retained_capacity() {
        let mut schedule =
            SpeculativeSchedule::new(SpeculativeSchedulerOptions::default()).unwrap();
        let ready = SpeculativeCandidate {
            status: SpeculativeRequestStatus::ReadyToSubmitVerification,
            optimistic_eligible: false,
            verification_complete: false,
            verification_deadline_expired: false,
        };
        assert_eq!(
            schedule.next_action(&[ready, ready]).unwrap(),
            Some(SpeculativeAction::SubmitVerification(0))
        );
        assert_eq!(
            schedule.next_action(&[ready, ready]).unwrap(),
            Some(SpeculativeAction::SubmitVerification(1))
        );

        let in_flight = SpeculativeCandidate {
            status: SpeculativeRequestStatus::TargetVerificationInFlight,
            optimistic_eligible: false,
            verification_complete: true,
            verification_deadline_expired: false,
        };
        let draft = SpeculativeCandidate {
            status: SpeculativeRequestStatus::ReadyToDraft,
            optimistic_eligible: false,
            verification_complete: false,
            verification_deadline_expired: false,
        };
        assert_eq!(
            schedule.next_action(&[in_flight, ready, draft]).unwrap(),
            Some(SpeculativeAction::DraftCommitted {
                index: 2,
                cross_request: true,
            })
        );
    }

    #[test]
    fn request_table_owns_actions_resources_fairness_and_deferred_cancellation() {
        let mut executor = MockExecutor::default();
        let mut first_cache = Vec::new();
        let mut second_cache = Vec::new();
        let options = SpeculativeSchedulerOptions::default().with_lookahead(false);
        let mut table =
            SpeculativeRequestTable::new(options, SpeculativeExecutionTopology::Single).unwrap();
        let config = SpeculativeConfig {
            max_tokens: 3,
            max_draft_tokens: 2,
            temperature: 0.7,
            eos_token_ids: Vec::new(),
        };
        let first_cancellation = GenerationCancellationToken::new();
        let first = table
            .submit(
                &mut executor,
                &mut first_cache,
                vec![4],
                config.clone(),
                empty_mock_runtime(config.max_tokens, first_cancellation.clone()),
                SpeculativeRandomness {
                    target: Some(0),
                    draft: Some(0),
                },
                false,
                (),
            )
            .unwrap();
        let second = table
            .submit(
                &mut executor,
                &mut second_cache,
                vec![8],
                config.clone(),
                empty_mock_runtime(config.max_tokens, GenerationCancellationToken::new()),
                SpeculativeRandomness {
                    target: Some(0),
                    draft: Some(10),
                },
                false,
                (),
            )
            .unwrap();

        assert_eq!(
            table.status(first),
            Some(SpeculativeRequestStatus::ReadyToDraft)
        );
        assert_eq!(
            table.status(second),
            Some(SpeculativeRequestStatus::ReadyToDraft)
        );
        table.step(&mut executor, false, ()).unwrap();
        table.step(&mut executor, false, ()).unwrap();
        assert!(table.request(first).unwrap().has_pending_verification());
        first_cancellation.cancel();
        table.run(&mut executor, false, ()).unwrap();

        let output = table.finish().unwrap();
        assert_eq!(output.requests.len(), 2);
        assert_eq!(output.requests[0].id, first);
        assert_eq!(
            output.requests[0].status,
            SpeculativeRequestStatus::Cancelled
        );
        assert_eq!(output.requests[0].token_ids, [1]);
        assert_eq!(output.requests[1].id, second);
        assert_eq!(
            output.requests[1].status,
            SpeculativeRequestStatus::Completed
        );
        assert_eq!(output.requests[1].token_ids, [1, 1, 0]);
        assert!(output.scheduler.cross_request_draft_opportunities > 0);
        assert_eq!(first_cache, [4, 1]);
        assert_eq!(second_cache, [8, 1, 1]);
    }

    #[test]
    fn request_table_applies_optimistic_actions_without_backend_scheduler_state() {
        let mut executor = MockExecutor::default();
        let mut cache = Vec::new();
        let config = SpeculativeConfig {
            max_tokens: 5,
            max_draft_tokens: 2,
            temperature: 0.7,
            eos_token_ids: Vec::new(),
        };
        let mut table = SpeculativeRequestTable::new(
            SpeculativeSchedulerOptions::default(),
            SpeculativeExecutionTopology::SameDeviceSplit,
        )
        .unwrap();
        let id = table
            .submit(
                &mut executor,
                &mut cache,
                vec![4],
                config.clone(),
                empty_mock_runtime(config.max_tokens, GenerationCancellationToken::new()),
                SpeculativeRandomness {
                    target: Some(0),
                    draft: Some(0),
                },
                false,
                (),
            )
            .unwrap();

        table.step(&mut executor, true, ()).unwrap();
        table.step(&mut executor, true, ()).unwrap();
        table.step(&mut executor, true, ()).unwrap();
        assert_eq!(
            table.status(id),
            Some(SpeculativeRequestStatus::OptimisticDraftReady)
        );
        table.run(&mut executor, true, ()).unwrap();
        let output = table.finish().unwrap();
        assert_eq!(
            output.requests[0].status,
            SpeculativeRequestStatus::Completed
        );
        assert!(output.requests[0].stats.counters.optimistic_draft_blocks > 0);
        assert!(
            output.requests[0]
                .stats
                .counters
                .discarded_optimistic_blocks
                > 0
        );
        assert_eq!(output.scheduler.peak_optimistic_branches, 1);
    }

    #[test]
    fn request_table_promotes_only_the_exact_matching_optimistic_suffix() {
        let mut executor = MockExecutor {
            full_acceptance: true,
            ..MockExecutor::default()
        };
        let mut cache = Vec::new();
        let config = SpeculativeConfig {
            max_tokens: 8,
            max_draft_tokens: 2,
            temperature: 0.7,
            eos_token_ids: Vec::new(),
        };
        let mut table = SpeculativeRequestTable::new(
            SpeculativeSchedulerOptions::default(),
            SpeculativeExecutionTopology::SameDeviceSplit,
        )
        .unwrap();
        let id = table
            .submit(
                &mut executor,
                &mut cache,
                vec![4],
                config.clone(),
                empty_mock_runtime(config.max_tokens, GenerationCancellationToken::new()),
                SpeculativeRandomness {
                    target: Some(0),
                    draft: Some(0),
                },
                false,
                (),
            )
            .unwrap();

        table.step(&mut executor, true, ()).unwrap();
        table.step(&mut executor, true, ()).unwrap();
        table.step(&mut executor, true, ()).unwrap();
        assert_eq!(
            table.status(id),
            Some(SpeculativeRequestStatus::OptimisticDraftReady)
        );
        table.step(&mut executor, true, ()).unwrap();

        let request = table.request(id).unwrap();
        assert_eq!(request.status(), SpeculativeRequestStatus::ReadyToDraft);
        assert_eq!(
            request
                .block()
                .unwrap()
                .proposals()
                .iter()
                .map(|proposal| proposal.token())
                .collect::<Vec<_>>(),
            [1]
        );
        assert_eq!(request.stats().optimistic_bonus_matches(), 1);
        assert_eq!(request.stats().consumed_optimistic_tokens(), 1);
        assert_eq!(request.stats().reused_optimistic_tokens(), 1);
        assert_eq!(request.stats().reused_optimistic_blocks(), 1);
        assert_eq!(request.stats().discarded_optimistic_tokens(), 0);
    }

    #[test]
    fn request_table_draft_generation_stops_at_the_sampler_grammar_boundary() {
        let mut executor = MockExecutor::default();
        let mut cache = Vec::new();
        let config = SpeculativeConfig {
            max_tokens: 8,
            max_draft_tokens: 4,
            temperature: 0.7,
            eos_token_ids: Vec::new(),
        };
        let runtime = SpeculativeOutputRuntime::new(
            MockSampling {
                draft_prefix_limit: Some(2),
                ..MockSampling::default()
            },
            GenerationSequence::new(config.max_tokens, []),
            MockConstraint::default(),
            MockPublisher::default(),
            GenerationCancellationToken::new(),
        );
        let mut table = SpeculativeRequestTable::new(
            SpeculativeSchedulerOptions::default().with_lookahead(false),
            SpeculativeExecutionTopology::Single,
        )
        .unwrap();
        let id = table
            .submit(
                &mut executor,
                &mut cache,
                vec![4],
                config,
                runtime,
                SpeculativeRandomness {
                    target: Some(0),
                    draft: Some(0),
                },
                false,
                (),
            )
            .unwrap();

        table.step(&mut executor, false, ()).unwrap();

        let request = table.request(id).unwrap();
        assert_eq!(
            request.status(),
            SpeculativeRequestStatus::ReadyToSubmitVerification
        );
        assert_eq!(request.block().unwrap().proposals().len(), 1);
        assert_eq!(request.stats().draft_tokens(), 1);
    }

    #[test]
    fn coordinator_commits_before_publication_and_discards_mismatched_lookahead() {
        let trace = TransactionTrace::default();
        let mut executor = MockExecutor {
            trace: Some(trace.clone()),
            ..MockExecutor::default()
        };
        let mut cache = vec![4, 5];
        let block = SpeculativeDraftBlock {
            state: vec![5, 1, 1],
            proposals: vec![
                SpeculativeProposal {
                    token: 1,
                    distribution: vec![0.0, 1.0],
                },
                SpeculativeProposal {
                    token: 1,
                    distribution: vec![0.0, 1.0],
                },
            ]
            .into(),
        };
        let mut pending =
            submit_verification_transaction(&mut executor, &mut cache, 5, block, ()).unwrap();
        pending
            .set_optimistic_branch(SpeculativeOptimisticBranch {
                block: SpeculativeDraftBlock {
                    state: vec![5, 1, 1, 2],
                    proposals: vec![SpeculativeProposal {
                        token: 2,
                        distribution: vec![0.0, 0.0, 1.0],
                    }]
                    .into(),
                },
                assumed_prefix: vec![5, 1, 1].into(),
            })
            .unwrap();
        let mut runtime =
            mock_output_runtime(GenerationCancellationToken::new(), Some(trace.clone()));
        let published = resolve_commit_and_publish(
            &mut executor,
            &mut cache,
            pending,
            &mut runtime,
            Some(&0),
            0.7,
            SpeculativeStats::default(),
            SpeculativeSchedulerOptions::default(),
            (),
        )
        .unwrap();

        assert!(matches!(
            published.status,
            SpeculativePublicationStatus::Continue(SpeculativeContinuation::None)
        ));
        assert_eq!(published.stats.counters.accepted_tokens, 1);
        assert_eq!(published.stats.counters.discarded_optimistic_tokens, 1);
        assert_eq!(cache, [4, 5, 5, 1]);
        let (_, sequence, constraint, publisher) = runtime.into_parts();
        assert_eq!(sequence.tokens(), [5, 1, 0]);
        assert_eq!(constraint.tokens, [1, 0]);
        assert_eq!(publisher.tokens, [1, 0]);
        assert!(!publisher.cancelled);
        assert_eq!(*trace.borrow(), ["wait", "commit", "publish"]);
    }

    #[test]
    fn coordinator_cancels_only_after_retained_verification_is_safe() {
        let trace = TransactionTrace::default();
        let mut executor = MockExecutor {
            trace: Some(trace.clone()),
            ..MockExecutor::default()
        };
        let mut cache = vec![4, 5];
        let block = SpeculativeDraftBlock {
            state: vec![5, 1],
            proposals: vec![SpeculativeProposal {
                token: 1,
                distribution: vec![0.0, 1.0],
            }]
            .into(),
        };
        let mut pending =
            submit_verification_transaction(&mut executor, &mut cache, 5, block, ()).unwrap();
        pending
            .set_optimistic_branch(SpeculativeOptimisticBranch {
                block: SpeculativeDraftBlock {
                    state: vec![5, 1, 2],
                    proposals: vec![SpeculativeProposal {
                        token: 2,
                        distribution: vec![0.0, 0.0, 1.0],
                    }]
                    .into(),
                },
                assumed_prefix: vec![5, 1].into(),
            })
            .unwrap();
        let cancellation = GenerationCancellationToken::new();
        cancellation.cancel();
        let mut runtime = mock_output_runtime(cancellation, Some(trace.clone()));
        let (stats, ()) = cancel_pending_verification(
            &mut executor,
            &mut cache,
            pending,
            &mut runtime,
            SpeculativeStats::default(),
            SpeculativeSchedulerOptions::default()
                .completion_wait()
                .unwrap(),
            (),
        )
        .unwrap();

        assert_eq!(stats.counters.discarded_optimistic_tokens, 1);
        assert_eq!(cache, [4, 5, 5]);
        let (_, sequence, _, publisher) = runtime.into_parts();
        assert_eq!(sequence.finish_reason(), Some(FinishReason::Cancelled));
        assert!(publisher.tokens.is_empty());
        assert!(publisher.cancelled);
        assert_eq!(*trace.borrow(), ["wait", "commit", "cancel"]);
    }

    type FailureTrace = Rc<RefCell<Vec<&'static str>>>;

    #[derive(Debug, Clone, Copy, Eq, PartialEq)]
    enum TransactionFailure {
        RequestSource,
        Submission,
        Completion,
        Commit,
        Restore,
    }

    impl fmt::Display for TransactionFailure {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str(match self {
                Self::RequestSource => "request source unavailable",
                Self::Submission => "submission failed",
                Self::Completion => "completion failed",
                Self::Commit => "commit failed",
                Self::Restore => "restore failed",
            })
        }
    }

    impl std::error::Error for TransactionFailure {}

    struct DropProbe {
        event: &'static str,
        drops: Rc<Cell<usize>>,
        trace: FailureTrace,
    }

    impl Drop for DropProbe {
        fn drop(&mut self) {
            self.drops.set(self.drops.get() + 1);
            self.trace.borrow_mut().push(self.event);
        }
    }

    struct DelayedCompletion {
        ready: Rc<Cell<bool>>,
        ready_after_polls: Option<usize>,
        polls: Rc<Cell<usize>>,
        fail: bool,
        publication_attempts: Rc<Cell<usize>>,
        publications_at_submission: usize,
        trace: FailureTrace,
        _probe: DropProbe,
    }

    impl Completion for DelayedCompletion {
        type Error = TransactionFailure;

        fn is_complete(&self) -> Result<bool, Self::Error> {
            let polls = self.polls.get() + 1;
            self.polls.set(polls);
            if self
                .ready_after_polls
                .is_some_and(|ready_after| polls >= ready_after)
            {
                self.ready.set(true);
            }
            Ok(self.ready.get())
        }

        fn wait(&self) -> Result<(), Self::Error> {
            assert_eq!(
                self.publication_attempts.get(),
                self.publications_at_submission,
                "verification completion must precede any later publication"
            );
            self.trace.borrow_mut().push("wait");
            self.ready.set(true);
            if self.fail {
                Err(TransactionFailure::Completion)
            } else {
                Ok(())
            }
        }
    }

    impl BoundedCompletion for DelayedCompletion {
        fn wait_bounded(
            self,
            _policy: BoundedCompletionWait,
        ) -> Result<BoundedCompletionOutcome, Self::Error> {
            self.wait()?;
            Ok(BoundedCompletionOutcome::Completed)
        }
    }

    struct TransactionVerification {
        logits: Vec<Vec<f32>>,
        ready: Rc<Cell<bool>>,
        _probe: DropProbe,
    }

    struct TransactionDraftState {
        values: Vec<u32>,
        _probe: Option<DropProbe>,
    }

    impl Clone for TransactionDraftState {
        fn clone(&self) -> Self {
            Self {
                values: self.values.clone(),
                _probe: None,
            }
        }
    }

    #[derive(Clone)]
    struct TransactionCheckpoint {
        target: Vec<u32>,
        draft: Vec<u32>,
    }

    struct TransactionCache {
        target: Vec<u32>,
        draft: Vec<u32>,
        fail_restore: bool,
        trace: FailureTrace,
    }

    struct TransactionExecutor {
        rejected_request: Option<SpeculativeRequestId>,
        fail_submission: bool,
        fail_completion: bool,
        fail_commit: bool,
        replayed_tokens: usize,
        ready_after_polls: Option<usize>,
        ready: Rc<Cell<bool>>,
        completion_polls: Rc<Cell<usize>>,
        publication_attempts: Rc<Cell<usize>>,
        completion_drops: Rc<Cell<usize>>,
        verification_drops: Rc<Cell<usize>>,
        committed_draft: Rc<RefCell<Vec<u32>>>,
        trace: FailureTrace,
    }

    impl TransactionExecutor {
        fn new(trace: FailureTrace, publication_attempts: Rc<Cell<usize>>) -> Self {
            Self {
                rejected_request: None,
                fail_submission: false,
                fail_completion: false,
                fail_commit: false,
                replayed_tokens: 0,
                ready_after_polls: Some(1),
                ready: Rc::new(Cell::new(false)),
                completion_polls: Rc::new(Cell::new(0)),
                publication_attempts,
                completion_drops: Rc::new(Cell::new(0)),
                verification_drops: Rc::new(Cell::new(0)),
                committed_draft: Rc::new(RefCell::new(Vec::new())),
                trace,
            }
        }
    }

    impl SpeculativeExecutor for TransactionExecutor {
        type Input = Vec<u32>;
        type Cache = TransactionCache;
        type TargetState = (Vec<u32>, Vec<u32>);
        type DraftState = TransactionDraftState;

        fn copy_draft_state<'a>(
            &self,
            state: &Self::DraftState,
            _context: Self::Context<'a>,
        ) -> Result<Self::DraftState, Self::Error>
        where
            Self: 'a,
        {
            Ok(state.clone())
        }
        type CacheCheckpoint = TransactionCheckpoint;
        type Verification = TransactionVerification;
        type Logits = Vec<f32>;
        type Context<'a> = ();
        type Completion = DelayedCompletion;
        type Telemetry = ();
        type Error = TransactionFailure;

        fn request_context<'a>(
            &self,
            request: SpeculativeRequestId,
            context: (),
        ) -> Result<(), TransactionFailure>
        where
            Self: 'a,
        {
            if self.rejected_request == Some(request) {
                Err(TransactionFailure::RequestSource)
            } else {
                Ok(context)
            }
        }

        fn prefill<'a>(
            &mut self,
            input: Self::Input,
            cache: &mut Self::Cache,
            _: Self::Context<'a>,
        ) -> Result<SpeculativePrefill<Self::TargetState, Self::Logits>, Self::Error> {
            cache.target.extend(input);
            Ok(SpeculativePrefill::new(
                vec![0.0, 1.0],
                (cache.target.clone(), cache.draft.clone()),
                1,
            ))
        }

        fn begin_proposal<'a>(
            &mut self,
            _: &Self::TargetState,
            last_token: u32,
            _: usize,
            _: Self::Context<'a>,
        ) -> Result<Self::DraftState, Self::Error> {
            Ok(TransactionDraftState {
                values: vec![last_token],
                _probe: None,
            })
        }

        fn proposal_logits<'a>(
            &mut self,
            state: &mut Self::DraftState,
            last_token: u32,
            _: Self::Context<'a>,
        ) -> Result<Self::Logits, Self::Error> {
            state.values.push(last_token);
            Ok(vec![0.0, 1.0])
        }

        fn checkpoint(&self, cache: &Self::Cache) -> Result<Self::CacheCheckpoint, Self::Error> {
            Ok(TransactionCheckpoint {
                target: cache.target.clone(),
                draft: cache.draft.clone(),
            })
        }

        fn restore_checkpoint<'a>(
            &mut self,
            cache: &mut Self::Cache,
            checkpoint: &Self::CacheCheckpoint,
            _: Self::Context<'a>,
        ) -> Result<(), Self::Error> {
            cache.trace.borrow_mut().push("restore");
            if cache.fail_restore {
                return Err(TransactionFailure::Restore);
            }
            cache.target.clone_from(&checkpoint.target);
            cache.draft.clone_from(&checkpoint.draft);
            Ok(())
        }

        fn submit_verification<'a>(
            &mut self,
            input_tokens: &[u32],
            cache: &mut Self::Cache,
            _: Self::Context<'a>,
        ) -> Result<Submission<Self::Verification, Self::Completion>, Self::Error> {
            self.ready.set(false);
            self.completion_polls.set(0);
            cache.target.extend_from_slice(input_tokens);
            self.trace.borrow_mut().push("submit");
            if self.fail_submission {
                return Err(TransactionFailure::Submission);
            }
            Ok(Submission {
                output: TransactionVerification {
                    logits: vec![vec![0.0, 1.0], vec![1.0, 0.0], vec![0.0, 1.0]],
                    ready: self.ready.clone(),
                    _probe: DropProbe {
                        event: "drop_verification",
                        drops: self.verification_drops.clone(),
                        trace: self.trace.clone(),
                    },
                },
                completion: DelayedCompletion {
                    ready: self.ready.clone(),
                    ready_after_polls: self.ready_after_polls,
                    polls: self.completion_polls.clone(),
                    fail: self.fail_completion,
                    publications_at_submission: self.publication_attempts.get(),
                    publication_attempts: self.publication_attempts.clone(),
                    trace: self.trace.clone(),
                    _probe: DropProbe {
                        event: "drop_completion",
                        drops: self.completion_drops.clone(),
                        trace: self.trace.clone(),
                    },
                },
            })
        }

        fn verification_logits<'a>(
            &self,
            output: &Self::Verification,
            index: usize,
            _: Self::Context<'a>,
        ) -> Result<Self::Logits, Self::Error> {
            assert!(
                output.ready.get(),
                "verification read before completion wait"
            );
            Ok(output.logits[index].clone())
        }

        fn commit_verification<'a>(
            &mut self,
            output: Self::Verification,
            state: Self::DraftState,
            cache: &mut Self::Cache,
            checkpoint: &Self::CacheCheckpoint,
            verified_inputs: usize,
            _: Self::Context<'a>,
        ) -> Result<SpeculativeCommit<Self::TargetState>, Self::Error> {
            assert!(output.ready.get(), "commit before completion wait");
            self.trace.borrow_mut().push("commit");
            self.committed_draft.borrow_mut().clone_from(&state.values);
            if self.fail_commit {
                return Err(TransactionFailure::Commit);
            }
            cache
                .target
                .truncate(checkpoint.target.len() + verified_inputs);
            cache.draft.clone_from(&state.values);
            Ok(SpeculativeCommit::new(
                (cache.target.clone(), cache.draft.clone()),
                self.replayed_tokens,
            ))
        }
    }

    #[derive(Clone, Default)]
    struct TransactionSampling {
        committed: Vec<u32>,
    }

    impl SpeculativeSampling for TransactionSampling {
        type Logits = Vec<f32>;
        type Distribution = Vec<f32>;
        type Seed = ();
        type RandomState = usize;
        type DraftRandomness = usize;
        type RandomnessRoot = usize;
        type Context<'a> = ();
        type Error = TransactionFailure;

        fn randomness_root<'a>(_: Option<Self::Seed>, _: ()) -> Result<usize, Self::Error>
        where
            Self: 'a,
        {
            Ok(0)
        }

        fn target_randomness_from_root<'a>(root: &mut usize, _: ()) -> Result<usize, Self::Error>
        where
            Self: 'a,
        {
            Ok(*root)
        }

        fn draft_randomness_from_root<'a>(root: &mut usize, _: ()) -> Result<usize, Self::Error>
        where
            Self: 'a,
        {
            Ok(*root)
        }

        fn draft_randomness_at<'a>(
            root: &usize,
            position: SpeculativeDraftRandomPosition,
            _: (),
        ) -> Result<usize, Self::Error>
        where
            Self: 'a,
        {
            Ok(*root + position.get())
        }

        fn process_logits<'a>(
            &mut self,
            logits: &Vec<f32>,
            _: f32,
            _: &[u32],
            _: SamplingPlacement,
            _: (),
        ) -> Result<Vec<f32>, Self::Error>
        where
            Self: 'a,
        {
            Ok(logits.clone())
        }

        fn sample<'a>(
            &self,
            distribution: &Vec<f32>,
            _: f32,
            _: Option<&mut usize>,
            _: SamplingPlacement,
            _: (),
        ) -> Result<u32, Self::Error>
        where
            Self: 'a,
        {
            Ok(argmax(distribution))
        }

        fn probability_at<'a>(
            &self,
            distribution: &Vec<f32>,
            token: u32,
            _: SamplingPlacement,
            _: (),
        ) -> Result<f32, Self::Error>
        where
            Self: 'a,
        {
            Ok(if argmax(distribution) == token {
                1.0
            } else {
                0.0
            })
        }

        fn sample_unit_interval<'a>(&self, _: Option<&mut usize>, _: ()) -> Result<f32, Self::Error>
        where
            Self: 'a,
        {
            Ok(0.5)
        }

        fn positive_probability_difference<'a>(
            &self,
            left: &Vec<f32>,
            _: &Vec<f32>,
            _: SamplingPlacement,
            _: (),
        ) -> Result<Option<Vec<f32>>, Self::Error>
        where
            Self: 'a,
        {
            Ok(Some(left.clone()))
        }

        fn update_sampler_state<'a>(
            &mut self,
            _: &Vec<f32>,
            token: u32,
            _: SamplingPlacement,
            _: (),
        ) -> Result<(), Self::Error>
        where
            Self: 'a,
        {
            self.committed.push(token);
            Ok(())
        }
    }

    struct ObservedPublisher {
        tokens: Vec<u32>,
        cancelled: bool,
        fail_committed: bool,
        fail_cancelled: bool,
        attempts: Rc<Cell<usize>>,
        trace: FailureTrace,
    }

    impl SpeculativePublisher<MockConstraint> for ObservedPublisher {
        fn publish_committed(
            &mut self,
            _: &mut MockConstraint,
            tokens: &[u32],
            _: &GenerationCancellationToken,
            _: bool,
        ) -> Result<bool, SpeculativeOutputError> {
            self.attempts.set(self.attempts.get() + 1);
            self.trace.borrow_mut().push("publish");
            if self.fail_committed {
                return Err(SpeculativeOutputError::publication("injected failure"));
            }
            self.tokens.extend_from_slice(tokens);
            Ok(false)
        }

        fn publish_cancelled(
            &mut self,
            _: &mut MockConstraint,
        ) -> Result<(), SpeculativeOutputError> {
            self.attempts.set(self.attempts.get() + 1);
            self.trace.borrow_mut().push("cancel");
            if self.fail_cancelled {
                return Err(SpeculativeOutputError::publication(
                    "injected cancellation failure",
                ));
            }
            self.cancelled = true;
            Ok(())
        }
    }

    fn transaction_cache(trace: FailureTrace) -> TransactionCache {
        TransactionCache {
            target: vec![4, 5],
            draft: vec![4, 5],
            fail_restore: false,
            trace,
        }
    }

    fn transaction_block(
        trace: FailureTrace,
        draft_drops: Rc<Cell<usize>>,
    ) -> SpeculativeDraftBlock<TransactionDraftState, Vec<f32>> {
        SpeculativeDraftBlock::new(
            TransactionDraftState {
                values: vec![5, 1, 1],
                _probe: Some(DropProbe {
                    event: "drop_draft",
                    drops: draft_drops,
                    trace,
                }),
            },
            vec![
                SpeculativeProposal::new(1, vec![0.0, 1.0]),
                SpeculativeProposal::new(1, vec![0.0, 1.0]),
            ],
        )
    }

    fn transaction_runtime(
        trace: FailureTrace,
        attempts: Rc<Cell<usize>>,
        fail_committed: bool,
        cancellation: GenerationCancellationToken,
    ) -> SpeculativeOutputRuntime<TransactionSampling, MockConstraint, ObservedPublisher> {
        let mut sequence = GenerationSequence::new(8, []);
        sequence.commit(5, TokenTerminalSignals::default()).unwrap();
        SpeculativeOutputRuntime::new(
            TransactionSampling::default(),
            sequence,
            MockConstraint::default(),
            ObservedPublisher {
                tokens: Vec::new(),
                cancelled: false,
                fail_committed,
                fail_cancelled: false,
                attempts,
                trace,
            },
            cancellation,
        )
    }

    fn empty_transaction_runtime(
        max_tokens: usize,
        trace: FailureTrace,
        attempts: Rc<Cell<usize>>,
    ) -> SpeculativeOutputRuntime<TransactionSampling, MockConstraint, ObservedPublisher> {
        SpeculativeOutputRuntime::new(
            TransactionSampling::default(),
            GenerationSequence::new(max_tokens, []),
            MockConstraint::default(),
            ObservedPublisher {
                tokens: Vec::new(),
                cancelled: false,
                fail_committed: false,
                fail_cancelled: false,
                attempts,
                trace,
            },
            GenerationCancellationToken::new(),
        )
    }

    #[test]
    fn request_context_refusal_preserves_lanes_before_prefill_and_scheduling() {
        for reject_before_prefill in [true, false] {
            let trace = FailureTrace::default();
            let attempts = Rc::new(Cell::new(0));
            let mut executor = TransactionExecutor::new(trace.clone(), attempts.clone());
            let mut first = transaction_cache(trace.clone());
            let mut second = transaction_cache(trace.clone());
            let untouched_second = (second.target.clone(), second.draft.clone());
            let mut table = SpeculativeRequestTable::new(
                SpeculativeSchedulerOptions::default().with_lookahead(false),
                SpeculativeExecutionTopology::Single,
            )
            .unwrap();
            let config = || SpeculativeConfig {
                max_tokens: 4,
                max_draft_tokens: 2,
                temperature: 0.7,
                eos_token_ids: Vec::new(),
            };
            let first_id = table
                .submit(
                    &mut executor,
                    &mut first,
                    vec![7, 9],
                    config(),
                    empty_transaction_runtime(4, trace.clone(), attempts.clone()),
                    SpeculativeRandomness::new(Some(0), Some(0)),
                    false,
                    (),
                )
                .unwrap();
            let second_id = SpeculativeRequestId::new(1);
            if reject_before_prefill {
                executor.rejected_request = Some(second_id);
            }
            let attempts_before = attempts.get();
            let first_tokens = table
                .request(first_id)
                .unwrap()
                .sequence()
                .tokens()
                .to_vec();
            let second_result = table.submit(
                &mut executor,
                &mut second,
                vec![13],
                config(),
                empty_transaction_runtime(4, trace.clone(), attempts.clone()),
                SpeculativeRandomness::new(Some(5), Some(8)),
                false,
                (),
            );
            if reject_before_prefill {
                assert!(matches!(
                    second_result,
                    Err(SpeculativeDriverError::Backend(
                        TransactionFailure::RequestSource
                    ))
                ));
                assert!(table.request(second_id).is_none());
                assert_eq!(attempts.get(), attempts_before);
                assert_eq!(
                    table.request(first_id).unwrap().sequence().tokens(),
                    first_tokens
                );
                drop(table);
                assert_eq!((second.target, second.draft), untouched_second);
            } else {
                assert_eq!(second_result.unwrap(), second_id);
                let second_tokens = table
                    .request(second_id)
                    .unwrap()
                    .sequence()
                    .tokens()
                    .to_vec();
                let attempts_before = attempts.get();
                let turns_before = table.stats.turns;
                executor.rejected_request = Some(second_id);
                assert!(matches!(
                    table.step(&mut executor, false, ()),
                    Err(SpeculativeDriverError::Backend(
                        TransactionFailure::RequestSource
                    ))
                ));
                assert_eq!(table.stats.turns, turns_before);
                assert_eq!(attempts.get(), attempts_before);
                assert_eq!(
                    table.request(first_id).unwrap().sequence().tokens(),
                    first_tokens
                );
                assert_eq!(
                    table.request(second_id).unwrap().sequence().tokens(),
                    second_tokens
                );
                assert!(!table.request(first_id).unwrap().has_pending_verification());
                assert!(!table.request(second_id).unwrap().has_pending_verification());
            }
        }
    }

    #[test]
    fn verification_submission_preserves_failure_when_rollback_also_fails() {
        for fail_restore in [false, true] {
            let trace = FailureTrace::default();
            let attempts = Rc::new(Cell::new(0));
            let mut executor = TransactionExecutor::new(trace.clone(), attempts.clone());
            executor.fail_submission = true;
            let mut cache = transaction_cache(trace.clone());
            cache.fail_restore = fail_restore;
            let old_target = cache.target.clone();
            let old_draft = cache.draft.clone();
            let draft_drops = Rc::new(Cell::new(0));
            let block = SpeculativeDraftBlock {
                state: TransactionDraftState {
                    values: vec![5],
                    _probe: Some(DropProbe {
                        event: "drop_failed_draft",
                        drops: draft_drops.clone(),
                        trace: trace.clone(),
                    }),
                },
                proposals: vec![SpeculativeProposal {
                    token: 1,
                    distribution: (),
                }]
                .into(),
            };
            let error =
                match submit_verification_transaction(&mut executor, &mut cache, 5, block, ()) {
                    Err(error) => error,
                    Ok(_) => panic!("submission refusal cannot return a pending verification"),
                };
            if fail_restore {
                assert!(matches!(
                    &error,
                    SpeculativeDriverError::SubmissionRollback {
                        submission: TransactionFailure::Submission,
                        rollback: TransactionFailure::Restore,
                    }
                ));
                assert_eq!(
                    std::error::Error::source(&error)
                        .and_then(|cause| cause.downcast_ref::<TransactionFailure>()),
                    Some(&TransactionFailure::Submission),
                );
                assert_ne!(
                    cache.target, old_target,
                    "failed rollback must not claim restoration"
                );
            } else {
                assert!(matches!(
                    error,
                    SpeculativeDriverError::Backend(TransactionFailure::Submission)
                ));
                assert_eq!(cache.target, old_target);
            }
            assert_eq!(cache.draft, old_draft);
            assert_eq!(attempts.get(), 0);
            assert_eq!(executor.completion_drops.get(), 0);
            assert_eq!(draft_drops.get(), 1);
            assert_eq!(
                &*trace.borrow(),
                &["submit", "restore", "drop_failed_draft"]
            );
        }
    }

    #[test]
    fn observation_and_cache_persistence_failures_restore_before_publication() {
        for failure in [
            SpeculativeLifecycleStage::Completion,
            SpeculativeLifecycleStage::Observation,
            SpeculativeLifecycleStage::CachePersistence,
        ] {
            let observer = Arc::new(LifecycleTrace::failing(failure));
            let trace = FailureTrace::default();
            let attempts = Rc::new(Cell::new(0));
            let mut executor = TransactionExecutor::new(trace.clone(), attempts.clone());
            let mut cache = transaction_cache(trace.clone());
            let config = SpeculativeConfig {
                max_tokens: 3,
                max_draft_tokens: 2,
                temperature: 0.7,
                eos_token_ids: Vec::new(),
            };
            let runtime =
                empty_transaction_runtime(config.max_tokens, trace.clone(), attempts.clone())
                    .with_lifecycle_observer(observer.clone());
            let mut table = SpeculativeRequestTable::new(
                SpeculativeSchedulerOptions::default().with_lookahead(false),
                SpeculativeExecutionTopology::Single,
            )
            .unwrap();
            let id = table
                .submit(
                    &mut executor,
                    &mut cache,
                    vec![4],
                    config,
                    runtime,
                    SpeculativeRandomness {
                        target: Some(0),
                        draft: Some(0),
                    },
                    false,
                    (),
                )
                .unwrap();
            table.step(&mut executor, false, ()).unwrap();
            table.step(&mut executor, false, ()).unwrap();
            assert!(table.request(id).unwrap().has_pending_verification());

            let error = table.step(&mut executor, false, ()).unwrap_err();
            assert!(matches!(error, SpeculativeDriverError::Output(_)));
            assert_eq!(attempts.get(), 1, "failed boundary published output");
            assert_eq!(cache.target, [4, 5, 4]);
            assert_eq!(cache.draft, [4, 5]);
            assert!(!trace.borrow().contains(&"commit"));
            assert!(trace.borrow().contains(&"restore"));
            if failure == SpeculativeLifecycleStage::Completion {
                let trace = trace.borrow();
                let completion_drop = trace
                    .iter()
                    .position(|event| *event == "drop_completion")
                    .expect("completion failure must dispose retained work");
                let restore = trace
                    .iter()
                    .position(|event| *event == "restore")
                    .expect("completion failure must restore the checkpoint");
                assert!(completion_drop < restore);
            }
            assert_eq!(
                observer
                    .stages()
                    .iter()
                    .filter(|stage| **stage == failure)
                    .count(),
                1
            );
        }
    }

    #[test]
    fn cancellation_observer_failure_prevents_terminal_mutation_and_publication() {
        let observer = Arc::new(LifecycleTrace::failing(
            SpeculativeLifecycleStage::Cancellation,
        ));
        let trace = FailureTrace::default();
        let attempts = Rc::new(Cell::new(0));
        let mut executor = TransactionExecutor::new(trace.clone(), attempts.clone());
        let mut cache = transaction_cache(trace.clone());
        let cancellation = GenerationCancellationToken::new();
        cancellation.cancel();
        let runtime = transaction_runtime(trace, attempts.clone(), false, cancellation)
            .with_lifecycle_observer(observer.clone());
        let mut table = SpeculativeRequestTable::new(
            SpeculativeSchedulerOptions::default().with_lookahead(false),
            SpeculativeExecutionTopology::Single,
        )
        .unwrap();
        let error = table
            .submit(
                &mut executor,
                &mut cache,
                vec![4],
                SpeculativeConfig {
                    max_tokens: 3,
                    max_draft_tokens: 2,
                    temperature: 0.7,
                    eos_token_ids: Vec::new(),
                },
                runtime,
                SpeculativeRandomness {
                    target: Some(0),
                    draft: Some(0),
                },
                false,
                (),
            )
            .unwrap_err();
        assert!(matches!(error, SpeculativeDriverError::Output(_)));
        assert_eq!(attempts.get(), 0);
        assert_eq!(cache.target, [4, 5]);
        assert_eq!(cache.draft, [4, 5]);
        assert_eq!(observer.stages(), [SpeculativeLifecycleStage::Cancellation]);
    }

    #[test]
    fn prefill_publication_failure_restores_cache_and_logical_state() {
        let trace = FailureTrace::default();
        let attempts = Rc::new(Cell::new(0));
        let mut executor = TransactionExecutor::new(trace.clone(), attempts.clone());
        let mut cache = transaction_cache(trace.clone());
        let mut runtime = empty_transaction_runtime(3, trace.clone(), attempts.clone());
        runtime.publisher.fail_committed = true;
        let mut table = SpeculativeRequestTable::new(
            SpeculativeSchedulerOptions::default().with_lookahead(false),
            SpeculativeExecutionTopology::Single,
        )
        .unwrap();
        let error = table
            .submit(
                &mut executor,
                &mut cache,
                vec![4],
                SpeculativeConfig {
                    max_tokens: 3,
                    max_draft_tokens: 2,
                    temperature: 0.7,
                    eos_token_ids: Vec::new(),
                },
                runtime,
                SpeculativeRandomness {
                    target: Some(0),
                    draft: Some(0),
                },
                false,
                (),
            )
            .unwrap_err();
        assert!(matches!(error, SpeculativeDriverError::Output(_)));
        assert_eq!(table.requests.len(), 0);
        drop(table);
        assert_eq!(cache.target, [4, 5]);
        assert_eq!(cache.draft, [4, 5]);
        assert!(trace.borrow().contains(&"restore"));
    }

    #[test]
    fn pending_cancellation_publication_failure_restores_cache_without_cancelling_sequence() {
        let trace = FailureTrace::default();
        let attempts = Rc::new(Cell::new(0));
        let mut executor = TransactionExecutor::new(trace.clone(), attempts.clone());
        let mut cache = transaction_cache(trace.clone());
        let mut table = SpeculativeRequestTable::new(
            SpeculativeSchedulerOptions::default().with_lookahead(false),
            SpeculativeExecutionTopology::Single,
        )
        .unwrap();
        let id = table
            .submit(
                &mut executor,
                &mut cache,
                vec![4],
                SpeculativeConfig {
                    max_tokens: 3,
                    max_draft_tokens: 2,
                    temperature: 0.7,
                    eos_token_ids: Vec::new(),
                },
                empty_transaction_runtime(3, trace.clone(), attempts),
                SpeculativeRandomness {
                    target: Some(0),
                    draft: Some(0),
                },
                false,
                (),
            )
            .unwrap();
        table.step(&mut executor, false, ()).unwrap();
        table.step(&mut executor, false, ()).unwrap();
        table.requests[id.index()].runtime.publisher.fail_cancelled = true;
        table.cancel(id).unwrap();
        let error = table.step(&mut executor, false, ()).unwrap_err();
        assert!(matches!(error, SpeculativeDriverError::Output(_)));
        assert!(!table.requests[id.index()].runtime.sequence().is_finished());
        drop(table);
        assert_eq!(cache.target, [4, 5, 4]);
        assert!(trace.borrow().contains(&"restore"));
    }

    #[test]
    fn direct_cancellation_publication_failure_restores_lifecycle_and_sequence() {
        let trace = FailureTrace::default();
        let attempts = Rc::new(Cell::new(0));
        let mut executor = TransactionExecutor::new(trace.clone(), attempts.clone());
        let mut cache = transaction_cache(trace.clone());
        let mut table = SpeculativeRequestTable::new(
            SpeculativeSchedulerOptions::default().with_lookahead(false),
            SpeculativeExecutionTopology::Single,
        )
        .unwrap();
        let id = table
            .submit(
                &mut executor,
                &mut cache,
                vec![4],
                SpeculativeConfig {
                    max_tokens: 3,
                    max_draft_tokens: 2,
                    temperature: 0.7,
                    eos_token_ids: Vec::new(),
                },
                empty_transaction_runtime(3, trace, attempts),
                SpeculativeRandomness {
                    target: Some(0),
                    draft: Some(0),
                },
                false,
                (),
            )
            .unwrap();
        table.step(&mut executor, false, ()).unwrap();
        assert!(table.requests[id.index()].block().is_some());
        assert!(!table.requests[id.index()].has_pending_verification());
        let status = table.requests[id.index()].status();
        let tokens = table.requests[id.index()].sequence().tokens().to_vec();
        let finish_reason = table.requests[id.index()].sequence().finish_reason();
        table.requests[id.index()].runtime.publisher.fail_cancelled = true;

        let error = table.cancel(id).unwrap_err();

        assert!(matches!(error, SpeculativeDriverError::Output(_)));
        assert_eq!(table.requests[id.index()].status(), status);
        assert_eq!(table.requests[id.index()].sequence().tokens(), tokens);
        assert_eq!(
            table.requests[id.index()].sequence().finish_reason(),
            finish_reason
        );
        assert!(table.requests[id.index()].block().is_some());
        assert!(!table.requests[id.index()].runtime.publisher.cancelled);
    }

    #[test]
    fn publication_observer_failure_occurs_after_commit_but_before_publisher() {
        let publication_boundaries = Arc::new(AtomicUsize::new(0));
        let observer: Arc<dyn SpeculativeLifecycleObserver> = Arc::new({
            let publication_boundaries = Arc::clone(&publication_boundaries);
            move |stage| {
                if stage == SpeculativeLifecycleStage::Publication
                    && publication_boundaries.fetch_add(1, Ordering::SeqCst) == 1
                {
                    Err(SpeculativeOutputError::publication(
                        "injected lifecycle publication failure",
                    ))
                } else {
                    Ok(())
                }
            }
        });
        let trace = FailureTrace::default();
        let attempts = Rc::new(Cell::new(0));
        let mut executor = TransactionExecutor::new(trace.clone(), attempts.clone());
        let mut cache = transaction_cache(trace.clone());
        let config = SpeculativeConfig {
            max_tokens: 3,
            max_draft_tokens: 2,
            temperature: 0.7,
            eos_token_ids: Vec::new(),
        };
        let runtime = empty_transaction_runtime(config.max_tokens, trace.clone(), attempts.clone())
            .with_lifecycle_observer(observer);
        let mut table = SpeculativeRequestTable::new(
            SpeculativeSchedulerOptions::default().with_lookahead(false),
            SpeculativeExecutionTopology::Single,
        )
        .unwrap();
        table
            .submit(
                &mut executor,
                &mut cache,
                vec![4],
                config,
                runtime,
                SpeculativeRandomness {
                    target: Some(0),
                    draft: Some(0),
                },
                false,
                (),
            )
            .unwrap();
        table.step(&mut executor, false, ()).unwrap();
        table.step(&mut executor, false, ()).unwrap();
        let error = table.step(&mut executor, false, ()).unwrap_err();
        assert!(matches!(error, SpeculativeDriverError::Output(_)));
        assert_eq!(publication_boundaries.load(Ordering::SeqCst), 2);
        assert_eq!(attempts.get(), 1, "verification output reached publisher");
        assert!(trace.borrow().contains(&"commit"));
        assert_eq!(
            trace
                .borrow()
                .iter()
                .filter(|event| **event == "publish")
                .count(),
            1
        );
    }

    #[test]
    fn request_table_accounts_nonzero_cache_replay_after_delayed_exact_completion() {
        let trace = FailureTrace::default();
        let attempts = Rc::new(Cell::new(0));
        let mut executor = TransactionExecutor::new(trace.clone(), attempts.clone());
        executor.replayed_tokens = 4;
        let mut cache = transaction_cache(trace.clone());
        let config = SpeculativeConfig {
            max_tokens: 3,
            max_draft_tokens: 2,
            temperature: 0.7,
            eos_token_ids: Vec::new(),
        };
        let mut table = SpeculativeRequestTable::new(
            SpeculativeSchedulerOptions::default().with_lookahead(false),
            SpeculativeExecutionTopology::Single,
        )
        .unwrap();
        let id = table
            .submit(
                &mut executor,
                &mut cache,
                vec![4],
                config.clone(),
                empty_transaction_runtime(config.max_tokens, trace.clone(), attempts.clone()),
                SpeculativeRandomness {
                    target: Some(0),
                    draft: Some(0),
                },
                false,
                (),
            )
            .unwrap();
        assert_eq!(attempts.get(), 1);

        table.run(&mut executor, false, ()).unwrap();
        let output = table.finish().unwrap();
        assert_eq!(output.requests[0].id(), id);
        assert_eq!(output.requests[0].stats().target_tokens(), 8);
        assert_eq!(output.requests[0].stats().emitted_tokens(), 3);
        assert_eq!(attempts.get(), 2);
        assert_eq!(cache.target, [4, 5, 4, 1, 1]);
        assert_eq!(cache.draft, [1, 1, 1]);
        let trace = trace.borrow();
        assert!(
            trace.iter().position(|event| *event == "wait").unwrap()
                < trace.iter().position(|event| *event == "commit").unwrap()
        );
        assert_eq!(trace.iter().filter(|event| **event == "publish").count(), 2);
    }

    #[test]
    fn request_table_completion_failure_restores_cache_before_any_new_publication() {
        let trace = FailureTrace::default();
        let attempts = Rc::new(Cell::new(0));
        let mut executor = TransactionExecutor::new(trace.clone(), attempts.clone());
        executor.fail_completion = true;
        let mut cache = transaction_cache(trace.clone());
        let config = SpeculativeConfig {
            max_tokens: 3,
            max_draft_tokens: 2,
            temperature: 0.7,
            eos_token_ids: Vec::new(),
        };
        let mut table = SpeculativeRequestTable::new(
            SpeculativeSchedulerOptions::default().with_lookahead(false),
            SpeculativeExecutionTopology::Single,
        )
        .unwrap();
        let id = table
            .submit(
                &mut executor,
                &mut cache,
                vec![4],
                config.clone(),
                empty_transaction_runtime(config.max_tokens, trace.clone(), attempts.clone()),
                SpeculativeRandomness {
                    target: Some(0),
                    draft: Some(0),
                },
                false,
                (),
            )
            .unwrap();
        assert_eq!(attempts.get(), 1, "only prefill is published");
        table.step(&mut executor, false, ()).unwrap();
        table.step(&mut executor, false, ()).unwrap();
        assert!(table.request(id).unwrap().has_pending_verification());

        let error = table.step(&mut executor, false, ()).unwrap_err();
        assert_eq!(error.to_string(), "completion failed");
        assert_eq!(attempts.get(), 1, "failed verification publishes nothing");
        drop(table);
        assert_eq!(cache.target, [4, 5, 4]);
        assert_eq!(cache.draft, [4, 5]);
        let trace = trace.borrow();
        assert_eq!(trace.iter().filter(|event| **event == "publish").count(), 1);
        assert!(trace.contains(&"restore"));
    }

    #[test]
    fn request_table_never_resolving_completion_retains_capacity_without_wait_or_publication() {
        let trace = FailureTrace::default();
        let attempts = Rc::new(Cell::new(0));
        let mut executor = TransactionExecutor::new(trace.clone(), attempts.clone());
        executor.ready_after_polls = None;
        let mut first_cache = transaction_cache(trace.clone());
        let mut second_cache = transaction_cache(trace.clone());
        let config = SpeculativeConfig {
            max_tokens: 4,
            max_draft_tokens: 2,
            temperature: 0.7,
            eos_token_ids: Vec::new(),
        };
        let mut table = SpeculativeRequestTable::new(
            SpeculativeSchedulerOptions::default().with_lookahead(false),
            SpeculativeExecutionTopology::Single,
        )
        .unwrap();
        let first = table
            .submit(
                &mut executor,
                &mut first_cache,
                vec![4],
                config.clone(),
                empty_transaction_runtime(config.max_tokens, trace.clone(), attempts.clone()),
                SpeculativeRandomness {
                    target: Some(0),
                    draft: Some(0),
                },
                false,
                (),
            )
            .unwrap();
        table.step(&mut executor, false, ()).unwrap();
        table.step(&mut executor, false, ()).unwrap();
        assert!(table.request(first).unwrap().has_pending_verification());
        table.cancel(first).unwrap();

        let second = table
            .submit(
                &mut executor,
                &mut second_cache,
                vec![8],
                config.clone(),
                empty_transaction_runtime(config.max_tokens, trace.clone(), attempts.clone()),
                SpeculativeRandomness {
                    target: Some(10),
                    draft: Some(10),
                },
                false,
                (),
            )
            .unwrap();
        for _ in 0..4 {
            assert!(table.step(&mut executor, false, ()).unwrap());
        }

        assert_eq!(
            table.status(first),
            Some(SpeculativeRequestStatus::TargetVerificationInFlight)
        );
        assert!(table.request(first).unwrap().has_pending_verification());
        assert_eq!(
            table.status(second),
            Some(SpeculativeRequestStatus::ReadyToSubmitVerification)
        );
        assert!(!table.request(second).unwrap().has_pending_verification());
        assert!(executor.completion_polls.get() >= 4);
        assert_eq!(executor.completion_drops.get(), 0);
        assert_eq!(executor.verification_drops.get(), 0);
        assert_eq!(attempts.get(), 2, "only the two prefills are published");
        {
            let trace = trace.borrow();
            assert!(!trace.contains(&"wait"));
            assert!(!trace.contains(&"commit"));
            assert!(!trace.contains(&"restore"));
            assert!(!trace.contains(&"cancel"));
            assert_eq!(trace.iter().filter(|event| **event == "publish").count(), 2);
        }

        drop(table);
        assert_eq!(first_cache.target, [4, 5, 4, 1, 1, 1]);
        assert_eq!(first_cache.draft, [4, 5]);
        assert_eq!(second_cache.target, [4, 5, 8]);
        assert_eq!(second_cache.draft, [4, 5]);
    }

    #[test]
    fn delayed_completion_retains_every_resource_and_waits_before_commit_or_publication() {
        let trace = FailureTrace::default();
        let attempts = Rc::new(Cell::new(0));
        let draft_drops = Rc::new(Cell::new(0));
        let mut executor = TransactionExecutor::new(trace.clone(), attempts.clone());
        let mut cache = transaction_cache(trace.clone());
        let pending = submit_verification_transaction(
            &mut executor,
            &mut cache,
            5,
            transaction_block(trace.clone(), draft_drops.clone()),
            (),
        )
        .unwrap();

        assert!(!executor.ready.get());
        assert_eq!(executor.completion_drops.get(), 0);
        assert_eq!(executor.verification_drops.get(), 0);
        assert_eq!(draft_drops.get(), 0);
        assert_eq!(attempts.get(), 0);
        assert_eq!(cache.target, [4, 5, 5, 1, 1]);

        let mut runtime = transaction_runtime(
            trace.clone(),
            attempts.clone(),
            false,
            GenerationCancellationToken::new(),
        );
        resolve_commit_and_publish(
            &mut executor,
            &mut cache,
            pending,
            &mut runtime,
            Some(&0),
            0.7,
            SpeculativeStats::default(),
            SpeculativeSchedulerOptions::default(),
            (),
        )
        .unwrap();

        assert!(executor.ready.get());
        assert_eq!(executor.completion_drops.get(), 1);
        assert_eq!(executor.verification_drops.get(), 1);
        assert_eq!(draft_drops.get(), 1);
        assert_eq!(executor.committed_draft.borrow().as_slice(), [5, 1, 1]);
        assert_eq!(cache.target, [4, 5, 5, 1]);
        assert_eq!(cache.draft, [5, 1, 1]);
        assert_eq!(
            trace.borrow().as_slice(),
            [
                "submit",
                "wait",
                "commit",
                "drop_draft",
                "drop_verification",
                "publish",
                "drop_completion"
            ]
        );
    }

    #[test]
    fn completion_failure_drops_resources_and_restores_every_checkpoint_without_publication() {
        let trace = FailureTrace::default();
        let attempts = Rc::new(Cell::new(0));
        let draft_drops = Rc::new(Cell::new(0));
        let mut executor = TransactionExecutor::new(trace.clone(), attempts.clone());
        executor.fail_completion = true;
        let mut cache = transaction_cache(trace.clone());
        let pending = submit_verification_transaction(
            &mut executor,
            &mut cache,
            5,
            transaction_block(trace.clone(), draft_drops.clone()),
            (),
        )
        .unwrap();
        let mut runtime = transaction_runtime(
            trace.clone(),
            attempts.clone(),
            false,
            GenerationCancellationToken::new(),
        );

        let error = resolve_commit_and_publish(
            &mut executor,
            &mut cache,
            pending,
            &mut runtime,
            Some(&0),
            0.7,
            SpeculativeStats::default(),
            SpeculativeSchedulerOptions::default(),
            (),
        )
        .err()
        .unwrap();

        assert_eq!(error.to_string(), "completion failed");
        assert_eq!(cache.target, [4, 5]);
        assert_eq!(cache.draft, [4, 5]);
        assert_eq!(attempts.get(), 0);
        assert_eq!(executor.completion_drops.get(), 1);
        assert_eq!(executor.verification_drops.get(), 1);
        assert_eq!(draft_drops.get(), 1);
        assert!(executor.committed_draft.borrow().is_empty());
        let (sampler, sequence, constraint, publisher) = runtime.into_parts();
        assert!(sampler.committed.is_empty());
        assert_eq!(sequence.tokens(), [5]);
        assert!(constraint.tokens.is_empty());
        assert!(publisher.tokens.is_empty());
        assert_eq!(
            trace.borrow().as_slice(),
            [
                "submit",
                "wait",
                "drop_completion",
                "restore",
                "drop_draft",
                "drop_verification"
            ]
        );
    }

    #[test]
    fn invalid_direct_completion_policy_disposes_work_before_restore() {
        let trace = FailureTrace::default();
        let attempts = Rc::new(Cell::new(0));
        let draft_drops = Rc::new(Cell::new(0));
        let mut executor = TransactionExecutor::new(trace.clone(), attempts.clone());
        executor.ready_after_polls = None;
        let mut cache = transaction_cache(trace.clone());
        let pending = submit_verification_transaction(
            &mut executor,
            &mut cache,
            5,
            transaction_block(trace.clone(), draft_drops),
            (),
        )
        .unwrap();
        let mut runtime = transaction_runtime(
            trace.clone(),
            attempts,
            false,
            GenerationCancellationToken::new(),
        );
        let options = SpeculativeSchedulerOptions {
            completion_timeout_milliseconds: 0,
            ..SpeculativeSchedulerOptions::default()
        };

        let error = resolve_commit_and_publish(
            &mut executor,
            &mut cache,
            pending,
            &mut runtime,
            Some(&0),
            0.7,
            SpeculativeStats::default(),
            options,
            (),
        )
        .err()
        .expect("invalid completion policy must fail");

        assert!(matches!(
            error,
            SpeculativeDriverError::Generation(GenerationError::ZeroSpeculativeCompletionTimeout)
        ));
        let trace = trace.borrow();
        let drop = trace
            .iter()
            .position(|event| *event == "drop_completion")
            .unwrap();
        let restore = trace.iter().position(|event| *event == "restore").unwrap();
        assert!(drop < restore);
    }

    #[test]
    fn publisher_failure_restores_backend_and_keeps_logical_state_uncommitted() {
        let trace = FailureTrace::default();
        let attempts = Rc::new(Cell::new(0));
        let mut executor = TransactionExecutor::new(trace.clone(), attempts.clone());
        let mut cache = transaction_cache(trace.clone());
        let pending = submit_verification_transaction(
            &mut executor,
            &mut cache,
            5,
            transaction_block(trace.clone(), Rc::new(Cell::new(0))),
            (),
        )
        .unwrap();
        let mut runtime = transaction_runtime(
            trace.clone(),
            attempts.clone(),
            true,
            GenerationCancellationToken::new(),
        );

        let error = resolve_commit_and_publish(
            &mut executor,
            &mut cache,
            pending,
            &mut runtime,
            Some(&0),
            0.7,
            SpeculativeStats::default(),
            SpeculativeSchedulerOptions::default(),
            (),
        )
        .err()
        .unwrap();

        assert!(matches!(error, SpeculativeDriverError::Output(_)));
        assert_eq!(cache.target, [4, 5]);
        assert_eq!(cache.draft, [4, 5]);
        assert_eq!(executor.committed_draft.borrow().as_slice(), [5, 1, 1]);
        let (sampler, sequence, constraint, publisher) = runtime.into_parts();
        assert!(sampler.committed.is_empty());
        assert_eq!(sequence.tokens(), [5]);
        assert!(constraint.tokens.is_empty());
        assert!(publisher.tokens.is_empty());
        assert_eq!(attempts.get(), 1);
        assert!(trace.borrow().contains(&"restore"));
    }

    #[test]
    fn commit_failure_restores_target_and_draft_checkpoints_without_promotion() {
        let trace = FailureTrace::default();
        let attempts = Rc::new(Cell::new(0));
        let mut executor = TransactionExecutor::new(trace.clone(), attempts.clone());
        executor.fail_commit = true;
        let mut cache = transaction_cache(trace.clone());
        let pending = submit_verification_transaction(
            &mut executor,
            &mut cache,
            5,
            transaction_block(trace.clone(), Rc::new(Cell::new(0))),
            (),
        )
        .unwrap();
        let mut runtime = transaction_runtime(
            trace.clone(),
            attempts.clone(),
            false,
            GenerationCancellationToken::new(),
        );

        let error = resolve_commit_and_publish(
            &mut executor,
            &mut cache,
            pending,
            &mut runtime,
            Some(&0),
            0.7,
            SpeculativeStats::default(),
            SpeculativeSchedulerOptions::default(),
            (),
        )
        .err()
        .unwrap();

        assert_eq!(error.to_string(), "commit failed");
        assert_eq!(cache.target, [4, 5]);
        assert_eq!(cache.draft, [4, 5]);
        assert_eq!(executor.committed_draft.borrow().as_slice(), [5, 1, 1]);
        assert_eq!(attempts.get(), 0);
        let (sampler, sequence, constraint, publisher) = runtime.into_parts();
        assert!(sampler.committed.is_empty());
        assert_eq!(sequence.tokens(), [5]);
        assert!(constraint.tokens.is_empty());
        assert!(publisher.tokens.is_empty());
        let trace = trace.borrow();
        assert!(
            trace.iter().position(|event| *event == "commit").unwrap()
                < trace.iter().position(|event| *event == "restore").unwrap()
        );
    }

    #[test]
    fn pending_cancellation_waits_commits_safe_prefix_and_discards_draft_state() {
        let trace = FailureTrace::default();
        let attempts = Rc::new(Cell::new(0));
        let draft_drops = Rc::new(Cell::new(0));
        let mut executor = TransactionExecutor::new(trace.clone(), attempts.clone());
        let mut cache = transaction_cache(trace.clone());
        let pending = submit_verification_transaction(
            &mut executor,
            &mut cache,
            5,
            transaction_block(trace.clone(), draft_drops.clone()),
            (),
        )
        .unwrap();
        let cancellation = GenerationCancellationToken::new();
        cancellation.cancel();
        let mut runtime = transaction_runtime(trace.clone(), attempts.clone(), false, cancellation);

        cancel_pending_verification(
            &mut executor,
            &mut cache,
            pending,
            &mut runtime,
            SpeculativeStats::default(),
            SpeculativeSchedulerOptions::default()
                .completion_wait()
                .unwrap(),
            (),
        )
        .unwrap();

        assert_eq!(cache.target, [4, 5, 5]);
        assert_eq!(cache.draft, [5, 1, 1]);
        assert_eq!(executor.committed_draft.borrow().as_slice(), [5, 1, 1]);
        assert_eq!(draft_drops.get(), 1);
        assert_eq!(executor.completion_drops.get(), 1);
        assert_eq!(executor.verification_drops.get(), 1);
        let (sampler, sequence, constraint, publisher) = runtime.into_parts();
        assert!(sampler.committed.is_empty());
        assert_eq!(sequence.tokens(), [5]);
        assert_eq!(sequence.finish_reason(), Some(FinishReason::Cancelled));
        assert!(constraint.tokens.is_empty());
        assert!(publisher.tokens.is_empty());
        assert!(publisher.cancelled);
        assert_eq!(attempts.get(), 1);
        let trace = trace.borrow();
        assert!(
            trace.iter().position(|event| *event == "wait").unwrap()
                < trace.iter().position(|event| *event == "commit").unwrap()
        );
        assert!(
            trace.iter().position(|event| *event == "commit").unwrap()
                < trace.iter().position(|event| *event == "cancel").unwrap()
        );
    }

    #[test]
    fn restore_failure_is_an_explicit_indeterminate_backend_error_without_publication() {
        let trace = FailureTrace::default();
        let attempts = Rc::new(Cell::new(0));
        let mut executor = TransactionExecutor::new(trace.clone(), attempts.clone());
        executor.fail_completion = true;
        let mut cache = transaction_cache(trace.clone());
        cache.fail_restore = true;
        let pending = submit_verification_transaction(
            &mut executor,
            &mut cache,
            5,
            transaction_block(trace.clone(), Rc::new(Cell::new(0))),
            (),
        )
        .unwrap();
        let mut runtime = transaction_runtime(
            trace.clone(),
            attempts.clone(),
            false,
            GenerationCancellationToken::new(),
        );

        let error = resolve_commit_and_publish(
            &mut executor,
            &mut cache,
            pending,
            &mut runtime,
            Some(&0),
            0.7,
            SpeculativeStats::default(),
            SpeculativeSchedulerOptions::default(),
            (),
        )
        .err()
        .unwrap();

        let SpeculativeDriverError::Rollback(retained) = &error else {
            panic!("both failed operation and rollback must be retained: {error}");
        };
        let pair = std::error::Error::source(retained)
            .unwrap()
            .source()
            .unwrap()
            .downcast_ref::<SpeculativeRollbackFailure<
                SpeculativeDriverError<TransactionFailure>,
                TransactionFailure,
            >>()
            .unwrap();
        assert_eq!(pair.rollback, TransactionFailure::Restore);
        assert!(matches!(
            pair.operation,
            SpeculativeDriverError::Backend(TransactionFailure::Completion)
        ));
        assert!(error.to_string().contains("completion failed"));
        assert!(error.to_string().contains("restore failed"));
        assert_eq!(cache.target, [4, 5, 5, 1, 1]);
        assert_eq!(cache.draft, [4, 5]);
        assert_eq!(attempts.get(), 0);
        let (sampler, sequence, constraint, publisher) = runtime.into_parts();
        assert!(sampler.committed.is_empty());
        assert_eq!(sequence.tokens(), [5]);
        assert!(constraint.tokens.is_empty());
        assert!(publisher.tokens.is_empty());
        assert_eq!(
            trace.borrow().as_slice()[..4],
            ["submit", "wait", "drop_completion", "restore"]
        );
    }
}
