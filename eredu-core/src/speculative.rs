//! High-level contracts and orchestration for speculative execution backends.

#[cfg(test)]
use crate::backend::BoundedCompletionOutcome;
use crate::{
    backend::{
        BoundedCompletion, BoundedCompletionWait, Completion, CompletionCancellationMode,
        ModelRuntime, SpeculativeTokenFilterController, Submission, TextGenerationBackend,
        TextGenerationConfig,
    },
    generation::{
        FinishReason, GenerationCancellationToken, GenerationError, GenerationRequestConfig,
        GenerationSequence, SemanticEvent, SpeculativeCancellationDisposition,
        SpeculativeRequestId, SpeculativeRequestLifecycle, SpeculativeRequestStatus,
        SpeculativeSchedulerOptions, TokenTerminalSignals,
    },
};
use serde::{Deserialize, Serialize};
use std::{
    sync::Arc,
    time::{Duration, Instant},
};

pub mod erasure;
pub use erasure::{
    DynGenerationCache, DynGenerationCheckpoint, DynGenerationDraftState, DynGenerationExecutor,
    DynGenerationTargetState, DynGenerationVerification, GenerationExecutorCacheFactory,
    GenerationExecutorTypes,
};

mod control;
pub use control::*;
mod activation;
pub use activation::*;
mod coordination;
pub use coordination::*;

/// Draft-model source selected for one speculative-generation request.
#[non_exhaustive]
pub enum SpeculativeDraft<'a, D> {
    /// Loaded target with its shared-backbone modes, without an external assistant.
    /// Targets without a retained shared draft execute causal or ordinary refinement only.
    Target,
    /// Masked drafting and causal verification over the retained shared target.
    SharedBackbone,
    /// Separately prepared assistant owned by the selected backend.
    External(&'a mut D),
    /// Draft heads embedded in the selected target model.
    Embedded,
}

impl<D> SpeculativeDraft<'_, D> {
    /// Lends the same source without transferring its assistant borrow.
    pub fn reborrow(&mut self) -> SpeculativeDraft<'_, D> {
        match self {
            Self::Target => SpeculativeDraft::Target,
            Self::SharedBackbone => SpeculativeDraft::SharedBackbone,
            Self::Embedded => SpeculativeDraft::Embedded,
            Self::External(draft) => SpeculativeDraft::External(draft),
        }
    }
}

/// Shared terminal generation output with speculative execution statistics.
pub type SpeculativeGenerationOutput = crate::generation::GenerationOutput<SpeculativeStats>;

/// Completed speculative requests plus aggregate fair-scheduler telemetry.
pub struct SpeculativeGenerationBatchOutput {
    /// Per-request results in submission order.
    requests: Vec<SpeculativeGenerationOutput>,
    /// Aggregate scheduler telemetry.
    scheduler: SpeculativeSchedulerStats,
}

impl SpeculativeGenerationBatchOutput {
    /// Creates a completed batch in stable submission order.
    pub fn new(
        requests: Vec<SpeculativeGenerationOutput>,
        scheduler: SpeculativeSchedulerStats,
    ) -> Self {
        Self {
            requests,
            scheduler,
        }
    }
    /// Per-request results in submission order.
    pub fn requests(&self) -> &[SpeculativeGenerationOutput] {
        &self.requests
    }
    /// Consumes the batch and returns its request results.
    pub fn into_requests(self) -> Vec<SpeculativeGenerationOutput> {
        self.requests
    }
    /// Aggregate scheduler telemetry.
    pub const fn scheduler(&self) -> &SpeculativeSchedulerStats {
        &self.scheduler
    }
    /// Appends a result while adapting another backend-neutral execution path.
    pub fn push_request(&mut self, request: SpeculativeGenerationOutput) {
        self.requests.push(request);
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
    config: Option<GenerationRequestConfig>,
    /// Portable canonical grammar state.
    constraint: Option<C>,
    /// Transactional decoded semantic parser state.
    semantic: Option<Box<dyn SpeculativeSemanticState>>,
    /// Cooperative cancellation owned by this lane.
    cancellation: Option<GenerationCancellationToken>,
    /// Called synchronously for canonical events from this lane.
    on_event: Option<Box<dyn FnMut(SemanticEvent) + 'a>>,
    control_handle: Option<crate::execution_control::GenerationControlHandle>,
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
        config: GenerationRequestConfig,
        constraint: C,
        semantic: Box<dyn SpeculativeSemanticState>,
        cancellation: GenerationCancellationToken,
        on_event: Box<dyn FnMut(SemanticEvent) + 'a>,
    ) -> Self {
        Self {
            prompt: Some(prompt),
            generation: Some(generation),
            config: Some(config),
            constraint: Some(constraint),
            semantic: Some(semantic),
            cancellation: Some(cancellation),
            on_event: Some(on_event),
            control_handle: None,
        }
    }
    /// Retains optional mode commands for this independently progressing lane.
    pub fn with_control_handle(
        mut self,
        handle: Option<crate::execution_control::GenerationControlHandle>,
    ) -> Self {
        self.control_handle = handle;
        self
    }
    /// Transfers command ownership into prepared generation without rebinding it.
    pub fn take_control_handle(
        &mut self,
    ) -> Option<crate::execution_control::GenerationControlHandle> {
        self.control_handle.take()
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
    pub fn take_config(&mut self) -> GenerationRequestConfig {
        self.config.take().expect("lane config already taken")
    }
    /// Borrows speculative controls.
    pub fn config(&self) -> &GenerationRequestConfig {
        self.config.as_ref().expect("lane config already taken")
    }
    /// Takes the grammar controller exactly once.
    pub fn take_constraint(&mut self) -> C {
        self.constraint
            .take()
            .expect("lane constraint already taken")
    }
    /// Takes semantic state exactly once.
    pub fn take_semantic(&mut self) -> Box<dyn SpeculativeSemanticState> {
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
    pub fn take_on_event(&mut self) -> Box<dyn FnMut(SemanticEvent) + 'a> {
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
    lanes: Option<Vec<SpeculativeGenerationLane<'a, B, C>>>,
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
        lanes: Vec<SpeculativeGenerationLane<'a, B, C>>,
        tokenizer_fingerprint: [u8; 32],
    ) -> Self {
        Self {
            drafting: Some(drafting),
            lanes: Some(lanes),
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
    pub fn take_lanes(&mut self) -> Vec<SpeculativeGenerationLane<'a, B, C>> {
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
    /// Attention patterns over one shared parameter catalog, with optional draft LoRA.
    SharedBackbone,
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
    /// Causal likelihood-ratio verification cannot consume this target's scores.
    IncompatibleScores {
        /// Actual semantics of the selected target distribution.
        score_kind: crate::refinement::PredictionScoreKind,
    },
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
            Self::Unavailable | Self::IncompatibleScores { .. } => None,
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

/// Statistics collected from one speculative sequence.
#[derive(Debug, Clone, Default)]
pub struct SpeculativeStats {
    /// Relationship between the request's target and draft execution placements.
    execution_topology: SpeculativeExecutionTopology,
    /// Target tokens evaluated during prefill and verification.
    target_tokens: usize,
    /// Complete canvas positions evaluated by ordinary refinement.
    refinement_tokens: usize,
    /// Ordinary numerical refinement invocations.
    refinement_steps: usize,
    /// Assistant tokens proposed.
    draft_tokens: usize,
    /// Assistant tokens accepted by target verification.
    accepted_tokens: usize,
    /// Number of target verification rounds.
    rounds: usize,
    /// Accepted proposal count for each round.
    accept_lens: Vec<usize>,
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

impl SpeculativeStats {
    /// Host time from lane submission to first commitment, before publication.
    /// No draft proposal or visible-text event starts this metric. Preparation
    /// and time queued before submission are included in the terminal output's
    /// [`crate::GenerationTiming`] instead.
    pub const fn submission_to_first_token(&self) -> Option<Duration> {
        self.submission_to_first_token
    }

    /// Selected target/draft placement relationship.
    pub const fn execution_topology(&self) -> SpeculativeExecutionTopology {
        self.execution_topology
    }
    /// Target tokens evaluated.
    pub const fn target_tokens(&self) -> usize {
        self.target_tokens
    }
    /// Complete canvas positions evaluated by ordinary refinement.
    pub const fn refinement_tokens(&self) -> usize {
        self.refinement_tokens
    }
    /// Ordinary numerical refinement invocations.
    pub const fn refinement_steps(&self) -> usize {
        self.refinement_steps
    }
    /// Assistant tokens proposed.
    pub const fn draft_tokens(&self) -> usize {
        self.draft_tokens
    }
    /// Assistant tokens accepted.
    pub const fn accepted_tokens(&self) -> usize {
        self.accepted_tokens
    }
    /// Target verification rounds.
    pub const fn rounds(&self) -> usize {
        self.rounds
    }
    /// Accepted proposal count per round.
    pub fn accept_lens(&self) -> &[usize] {
        &self.accept_lens
    }
    /// Emitted token count.
    pub const fn emitted_tokens(&self) -> usize {
        self.emitted_tokens
    }
    /// Optimistically drafted token count.
    pub const fn optimistic_draft_tokens(&self) -> usize {
        self.optimistic_draft_tokens
    }
    /// Optimistic block count.
    pub const fn optimistic_draft_blocks(&self) -> usize {
        self.optimistic_draft_blocks
    }
    /// Reused optimistic token count.
    pub const fn reused_optimistic_tokens(&self) -> usize {
        self.reused_optimistic_tokens
    }
    /// Reused optimistic block count.
    pub const fn reused_optimistic_blocks(&self) -> usize {
        self.reused_optimistic_blocks
    }
    /// Optimistic tokens consumed by target bonuses.
    pub const fn consumed_optimistic_tokens(&self) -> usize {
        self.consumed_optimistic_tokens
    }
    /// Discarded optimistic token count.
    pub const fn discarded_optimistic_tokens(&self) -> usize {
        self.discarded_optimistic_tokens
    }
    /// Discarded optimistic block count.
    pub const fn discarded_optimistic_blocks(&self) -> usize {
        self.discarded_optimistic_blocks
    }
    /// Target bonuses emitted while an optimistic branch existed.
    pub const fn optimistic_target_bonus_tokens(&self) -> usize {
        self.optimistic_target_bonus_tokens
    }
    /// Matching optimistic bonus count.
    pub const fn optimistic_bonus_matches(&self) -> usize {
        self.optimistic_bonus_matches
    }
    /// Mismatching optimistic bonus count.
    pub const fn optimistic_bonus_mismatches(&self) -> usize {
        self.optimistic_bonus_mismatches
    }
    /// Whether adaptive lookahead is disabled.
    pub const fn adaptive_lookahead_disabled(&self) -> bool {
        self.adaptive_lookahead_disabled
    }
    /// Time spent drafting optimistic branches.
    pub const fn optimistic_draft_time(&self) -> Duration {
        self.optimistic_draft_time
    }
    /// Time retained verification remained in flight.
    pub const fn verification_in_flight_time(&self) -> Duration {
        self.verification_in_flight_time
    }
    /// Whether component timings were collected.
    pub const fn component_timings_collected(&self) -> bool {
        self.component_timings_collected
    }
    /// Draft-context device time.
    pub const fn draft_context_time(&self) -> Duration {
        self.draft_context_time
    }
    /// Draft-assistant device time.
    pub const fn draft_assistant_time(&self) -> Duration {
        self.draft_assistant_time
    }
    /// Draft-head device time.
    pub const fn draft_head_time(&self) -> Duration {
        self.draft_head_time
    }
    /// Target-verification device time.
    pub const fn target_verification_time(&self) -> Duration {
        self.target_verification_time
    }
    /// Scheduler turns for this request.
    pub const fn scheduler_turns(&self) -> usize {
        self.scheduler_turns
    }
    /// Draft turns performed beside other in-flight target work.
    pub const fn cross_request_draft_opportunities(&self) -> usize {
        self.cross_request_draft_opportunities
    }
    /// Wall-clock generation duration.
    pub const fn elapsed(&self) -> Duration {
        self.elapsed
    }

    /// Adds backend-measured component timings without exposing mutable fields.
    pub fn add_component_timings(
        &mut self,
        draft_context: Duration,
        draft_assistant: Duration,
        draft_head: Duration,
        target_verification: Duration,
    ) {
        self.draft_context_time += draft_context;
        self.draft_assistant_time += draft_assistant;
        self.draft_head_time += draft_head;
        self.target_verification_time += target_verification;
        self.component_timings_collected = true;
    }

    /// Adds completed scheduler rounds to portable telemetry.
    pub fn add_scheduler_rounds(&mut self, rounds: usize) {
        self.rounds += rounds;
    }

    /// Records aggregate optimistic work used by adaptive-lookahead policy.
    pub fn record_optimistic_accounting(
        &mut self,
        drafted_blocks: usize,
        reused_tokens: usize,
        discarded_tokens: usize,
    ) {
        self.optimistic_draft_blocks += drafted_blocks;
        self.reused_optimistic_tokens += reused_tokens;
        self.discarded_optimistic_tokens += discarded_tokens;
    }

    /// Clears the cached adaptive-lookahead decision before policy re-evaluation.
    pub fn reset_adaptive_lookahead_decision(&mut self) {
        self.adaptive_lookahead_disabled = false;
    }

    /// Fraction of proposed tokens accepted by the target.
    pub fn accept_rate(&self) -> f64 {
        if self.draft_tokens == 0 {
            0.0
        } else {
            self.accepted_tokens as f64 / self.draft_tokens as f64
        }
    }

    /// Re-evaluates whether optional lookahead remains profitable.
    pub fn update_adaptive_lookahead(&mut self, options: SpeculativeSchedulerOptions) {
        if !options.adaptive_lookahead
            || self.adaptive_lookahead_disabled
            || self.optimistic_draft_blocks < options.adaptive_lookahead_min_blocks
        {
            return;
        }
        self.adaptive_lookahead_disabled = self.reused_optimistic_tokens == 0
            || self.reused_optimistic_tokens < self.discarded_optimistic_tokens;
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

/// One proposal-source advancement, separate from target verification scores.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpeculativeProposalStep<L> {
    /// A causal assistant distribution is processed and sampled by the shared driver.
    CausalLogits(L),
    /// A masked assistant's selected token has no causal proposal probability.
    /// The shared driver preserves it and uses sampled-target matching.
    TargetMatchingToken(u32),
}

/// Verification evidence supplied by an admitted draft source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpeculativeVerificationPolicy {
    /// The proposal process supplies known conditional sampling distributions.
    LikelihoodRatio,
    /// Selected masked tokens are compared with sequential processed target draws.
    TargetMatching,
}

/// Settled complete-canvas selections with explicit invocation accounting.
/// Position rows are never causal draft likelihoods or conditional target scores.
pub struct RefinementBlock<L> {
    /// Selected identities and their retained full position scores.
    pub positions: Vec<(u32, L)>,
    /// Complete model positions evaluated, including repeated refinement passes.
    pub evaluated_positions: usize,
    /// Numerical refinement invocations completed by this block.
    pub iterations: u32,
}

/// One settled numerical refinement advancement. Progress does not commit output.
pub enum RefinementProgress<L> {
    /// More numerical invocations remain; the complete canvas stays request-local.
    Pending {
        /// Admitted canvas width, including positions beyond the output allowance.
        width: usize,
        /// Positions evaluated by this advancement only.
        evaluated_positions: usize,
        /// Numerical invocations completed by this advancement only.
        iterations: u32,
    },
    /// Selected identities and position scores, ready for later finalization.
    Complete(RefinementBlock<L>),
}

/// Whole-session speculative execution contract.
///
/// Tensor values, execution queues, caches, model state, logits, native
/// completions, and errors remain opaque associated types. The contract models
/// only high-level prefill, proposal, verification, and exact commit actions;
/// it deliberately does not define primitive tensor operations.
pub trait SpeculativeExecutor {
    /// Completed native decisions use this source protocol instead of exposing
    /// logits or fabricating verification probability evidence.
    fn uses_completed_decisions(&self) -> bool {
        false
    }

    /// Installs a prepared completed-decision source without executing a prompt.
    fn prepare_decision_source<'a>(
        &mut self,
        _input: Self::Input,
        _cache: &mut Self::Cache,
        _context: Self::Context<'a>,
    ) -> Result<Option<Self::TargetState>, Self::Error>
    where
        Self: 'a,
    {
        Ok(None)
    }

    /// Advances one context, numerical, or sampled operation. A decision retains
    /// exact native output until commitment; progress never advances constraints.
    fn advance_decision_source<'a>(
        &mut self,
        _cache: &mut Self::Cache,
        _state: &mut Self::TargetState,
        _remaining: usize,
        _context: Self::Context<'a>,
    ) -> Result<Option<CompletedDecisionProgress>, Self::Error>
    where
        Self: 'a,
    {
        Ok(None)
    }

    /// Completes one nonfinal context chunk without requesting a sampling decision.
    fn advance_decision_context<'a>(
        &mut self,
        _cache: &mut Self::Cache,
        _context: Self::Context<'a>,
    ) -> Result<Option<crate::backend::TextProgress>, Self::Error>
    where
        Self: 'a,
    {
        Ok(None)
    }

    /// Exact commitment allowance for a completed source, including unlimited runs.
    fn completed_decision_allowance(&self, _cache: &Self::Cache) -> Option<Option<usize>> {
        None
    }

    /// Settles native sampling before a decision becomes authoritative.
    fn commit_decision_source<'a>(
        &mut self,
        _cache: &mut Self::Cache,
        _token: u32,
        _committed_position: u64,
        _context: Self::Context<'a>,
    ) -> Result<(), Self::Error>
    where
        Self: 'a,
    {
        Ok(())
    }

    /// Settles a completed-decision block and its committed suffix. None selects
    /// the ordinary score-source handoff; false retains the queued command.
    fn settle_decision_source<'a>(
        &mut self,
        _cache: &mut Self::Cache,
        _previous: crate::decoding::DecodingMode,
        _requested: crate::decoding::DecodingMode,
        _context: Self::Context<'a>,
    ) -> Result<Option<bool>, Self::Error>
    where
        Self: 'a,
    {
        Ok(None)
    }
    /// Exact verification contract; masked confidence selection supplies no causal q.
    fn verification_policy(&self) -> SpeculativeVerificationPolicy {
        SpeculativeVerificationPolicy::LikelihoodRatio
    }
    /// Backend-owned model input accepted by prefill submission.
    type Input;
    /// Complete backend-owned target cache.
    type Cache;
    /// Target state used to seed one proposal round.
    type TargetState;
    /// Private, discardable assistant state.
    type DraftState: Clone;
    /// Exact target-cache checkpoint marker.
    type CacheCheckpoint;
    /// Retained target verification output.
    type Verification;
    /// Opaque logits consumed by the backend's sampling adapter.
    type Logits: Clone;
    /// Backend execution assignment for one operation.
    type Context<'a>: Copy
    where
        Self: 'a;
    /// Exact completion for submitted verification work.
    type Completion: BoundedCompletion<Error = Self::Error>;
    /// Optional backend-specific component telemetry.
    type Telemetry: SpeculativeTelemetry;
    /// Structured backend error.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Mode mechanisms retained by this executor, independent of artifacts.
    fn decoding_capabilities(&self) -> crate::decoding::DecodingCapabilities {
        let native =
            Self::Completion::supports_cancellation(CompletionCancellationMode::NativeCancel);
        let quarantine = Self::Completion::supports_cancellation(
            CompletionCancellationMode::QuarantineUntilComplete,
        );
        crate::decoding::DecodingCapabilities {
            autoregressive: true,
            speculation: self.max_proposals() != 0,
            mode_changes: true,
            ..Default::default()
        }
        .with_completion_support(native, quarantine)
    }

    /// Refines a complete admitted canvas over the settled causal prefix.
    /// Returned rows retain selected identities and position-score semantics.
    /// `committed` pins a constraint-corrected prefix when its dependent suffix
    /// needs refinement again. Implementations retain request-local randomness
    /// in `state`; neither canonical cache nor target sampling may advance.
    fn refine_block<'a>(
        &mut self,
        _state: &mut Self::TargetState,
        _next_logits: &Self::Logits,
        _committed: &[u32],
        _policy: Option<crate::refinement::RefinementPolicy>,
        _temperature: f32,
        _context: Self::Context<'a>,
    ) -> Result<Option<RefinementProgress<Self::Logits>>, Self::Error>
    where
        Self: 'a,
    {
        Ok(None)
    }

    /// Agrees source-free commands before scheduling native work.
    fn coordinate_decoding_control<'a>(
        &mut self,
        local: crate::decoding::DecodingCoordination,
        _context: Self::Context<'a>,
    ) -> Result<crate::decoding::DecodingCoordination, crate::BackendFailure>
    where
        Self: 'a,
    {
        Ok(local)
    }

    /// Complete storage retained by an immutable next-token row at a handoff.
    fn logits_snapshot_estimate(
        &self,
        _logits: &Self::Logits,
    ) -> Option<crate::execution_control::SnapshotEstimate> {
        None
    }

    /// Installs request-local draft options without executing or rebinding parameters.
    fn configure_decoding_mode(
        &mut self,
        _state: &mut Self::TargetState,
        _mode: crate::decoding::DecodingMode,
        _temperature: f32,
    ) -> Result<(), GenerationError> {
        Ok(())
    }

    /// Agrees a host preparation result before any participant enters the next
    /// speculative phase. Distributed executors use their retained session
    /// transport and cumulative ledger; local executors resolve their own status.
    fn agree_text_preparation<'a>(
        &mut self,
        _stage: crate::run_preparation::TextPreparationStage,
        status: crate::run_preparation::TextPreparationStatus,
        _context: Self::Context<'a>,
    ) -> Result<crate::run_preparation::TextPreparationOutcome, crate::BackendFailure>
    where
        Self: 'a,
    {
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
    ) -> Result<Vec<SpeculativeScheduleState>, crate::BackendFailure>
    where
        Self: 'a,
    {
        Ok(local)
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
    ) -> Result<(), SpeculativeControlError>
    where
        Self: 'a,
    {
        if plan.is_empty() {
            Ok(())
        } else {
            Err(SpeculativeControlError::Unsupported(
                "executor has no scoped speculative activation collector",
            ))
        }
    }

    /// Releases optional conversions; caller must establish a settled boundary.
    fn trim_parameter_conversions(
        &mut self,
    ) -> Result<
        crate::residency::ExecutionConversionRetentionTrimReport,
        crate::residency::ParameterConversionTrimError,
    > {
        Err(crate::residency::ParameterConversionTrimError::Unsupported)
    }

    /// Reads load-selected retention budgets without polling, settling, or touching
    /// request state. This is valid even while verification is pending.
    fn parameter_conversion_retention(
        &self,
    ) -> Result<Option<crate::residency::ExecutionConversionRetentionReport>, crate::BackendFailure>
    {
        Ok(None)
    }

    /// Pure observation of settled model and prediction cache frontiers, capacity
    /// and residency. No polling, synchronization, copying or budget consumption.
    /// Executors without complete settled observations remain unsupported by default.
    fn continuation_memory_observation(
        &self,
        _cache: &Self::Cache,
        _state: &Self::TargetState,
        _additional_positions: u64,
    ) -> Result<Option<SpeculativeContinuationObservation>, crate::BackendFailure> {
        Ok(None)
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
    ) -> Result<Option<(Self::CacheCheckpoint, Self::TargetState)>, SpeculativeControlError>
    where
        Self: 'a,
    {
        Ok(None)
    }

    /// Atomically restores cache and returns an isolated assistant seed. The saved
    /// checkpoint remains reusable. Must settle copying before returning.
    fn restore_control_snapshot<'a>(
        &mut self,
        _cache: &mut Self::Cache,
        _checkpoint: &Self::CacheCheckpoint,
        _state: &Self::TargetState,
        _context: Self::Context<'a>,
    ) -> Result<Option<Self::TargetState>, SpeculativeControlError>
    where
        Self: 'a,
    {
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
    ) -> Result<SpeculativePrefill<Self::TargetState, Self::Logits>, Self::Error>
    where
        Self: 'context;

    /// Starts one private source state sized to the available output budget.
    /// Zero capacity preserves source alignment for a causal target step without
    /// executing drafting or advancing proposal randomness.
    fn begin_proposal<'a>(
        &mut self,
        state: &Self::TargetState,
        last_token: u32,
        proposal_capacity: usize,
        context: Self::Context<'a>,
    ) -> Result<Self::DraftState, Self::Error>
    where
        Self: 'a;

    /// Advances private assistant state and returns explicit proposal evidence.
    /// Causal assistants return logits; masked assistants return already selected
    /// tokens whose verification samples the causal target distribution.
    fn proposal_step<'a>(
        &mut self,
        state: &mut Self::DraftState,
        last_token: u32,
        context: Self::Context<'a>,
    ) -> Result<SpeculativeProposalStep<Self::Logits>, Self::Error>
    where
        Self: 'a;

    /// Captures the exact cache boundary before target verification.
    fn checkpoint(&self, cache: &Self::Cache) -> Result<Self::CacheCheckpoint, Self::Error>;

    /// Restores the exact cache boundary after an aborted verification transaction.
    fn restore_checkpoint<'a>(
        &mut self,
        cache: &mut Self::Cache,
        checkpoint: &Self::CacheCheckpoint,
        context: Self::Context<'a>,
    ) -> Result<(), Self::Error>
    where
        Self: 'a;

    /// Submits verification of the last committed token and proposal block.
    ///
    /// Implementations materialize token tensors internally and return an exact
    /// completion retaining every resource required by the submission.
    fn submit_verification<'a>(
        &mut self,
        input_tokens: &[u32],
        cache: &mut Self::Cache,
        context: Self::Context<'a>,
    ) -> Result<Submission<Self::Verification, Self::Completion>, Self::Error>
    where
        Self: 'a;

    /// Selects one prediction row from retained verification output.
    fn verification_logits<'a>(
        &self,
        output: &Self::Verification,
        index: usize,
        context: Self::Context<'a>,
    ) -> Result<Self::Logits, Self::Error>
    where
        Self: 'a;

    /// Commits exactly the requested verified inputs and restores matching seed state.
    fn commit_verification<'a>(
        &mut self,
        output: Self::Verification,
        draft_state: Self::DraftState,
        cache: &mut Self::Cache,
        checkpoint: &Self::CacheCheckpoint,
        verified_inputs: usize,
        context: Self::Context<'a>,
    ) -> Result<SpeculativeCommit<Self::TargetState>, Self::Error>
    where
        Self: 'a;
}

/// One source-owned physical operation, without invented logits or probabilities.
pub enum CompletedDecisionProgress {
    /// Settled context or numerical work with no authoritative token.
    Progress(crate::backend::TextProgress),
    /// An exact native decision awaiting the shared commitment boundary.
    Token {
        /// Canonical vocabulary identity.
        token_id: u32,
        /// Numerical meaning declared by the executed invocation.
        score_kind: crate::refinement::PredictionScoreKind,
    },
    /// Source termination without another decision.
    Finished,
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
    /// Selected masked canvases own a separate proposal stream and do not use
    /// the conditional drafter's position-addressable sampling keys.
    pub fn with_verification_policy(mut self, policy: SpeculativeVerificationPolicy) -> Self {
        if policy == SpeculativeVerificationPolicy::TargetMatching {
            self.draft = None;
        }
        self
    }
}

/// Exclusive canonical cache storage for a generation lane. Owned caches support
/// persistent drivers; borrowed caches preserve an enclosing execution scope's
/// final state. Neither form clones cache values or parameter owners.
pub enum GenerationCache<'a, T> {
    /// The enclosing scope owns the independently mutable cache.
    Borrowed(Box<dyn std::ops::DerefMut<Target = T> + 'a>),
    /// The request owns its independently mutable cache.
    Owned(T),
}
impl<'a, T> From<&'a mut T> for GenerationCache<'a, T> {
    fn from(cache: &'a mut T) -> Self {
        Self::Borrowed(Box::new(cache))
    }
}
impl<T> std::ops::Deref for GenerationCache<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        match self {
            Self::Borrowed(cache) => cache.as_ref(),
            Self::Owned(cache) => cache,
        }
    }
}
impl<T> std::ops::DerefMut for GenerationCache<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        match self {
            Self::Borrowed(cache) => cache.as_mut(),
            Self::Owned(cache) => cache,
        }
    }
}

/// One backend-prepared lane lent to neutral speculative orchestration.
///
/// Opaque execution values carry no scheduler or lifecycle policy. The lane
/// transfers owned cache storage or an exclusive borrow valid for the scope.
pub struct PreparedSpeculativeLane<'a, E, S, C, P>
where
    E: SpeculativeExecutor,
    S: SpeculativeSampling<Logits = E::Logits, Error = E::Error>,
    C: SpeculativeConstraint,
    P: SpeculativePublisher<C>,
{
    /// Backend-owned request cache.
    cache: Option<GenerationCache<'a, E::Cache>>,
    /// Backend-owned prepared model input.
    input: Option<E::Input>,
    /// Validated speculative generation controls.
    config: Option<GenerationRequestConfig>,
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
        cache: impl Into<GenerationCache<'a, E::Cache>>,
        input: E::Input,
        config: GenerationRequestConfig,
        runtime: SpeculativeOutputRuntime<S, C, P>,
        randomness: SpeculativeRandomness<S::RandomState, S::DraftRandomness>,
    ) -> Self {
        Self {
            cache: Some(cache.into()),
            input: Some(input),
            config: Some(config),
            runtime: Some(runtime),
            randomness: Some(randomness),
        }
    }
    /// Takes exclusive owned or borrowed cache storage exactly once.
    pub fn take_cache(&mut self) -> GenerationCache<'a, E::Cache> {
        self.cache.take().expect("prepared cache already taken")
    }
    /// Takes model input exactly once.
    pub fn take_input(&mut self) -> E::Input {
        self.input.take().expect("prepared input already taken")
    }
    /// Takes speculative controls exactly once.
    pub fn take_config(&mut self) -> GenerationRequestConfig {
        self.config.take().expect("prepared config already taken")
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
    /// Drives one prepared set of lanes through the neutral lifecycle.
    #[allow(clippy::too_many_arguments)]
    fn run<'a, E, S, C, P>(
        self,
        executor: &'a mut E,
        lanes: Vec<PreparedSpeculativeLane<'a, E, S, C, P>>,
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

    /// Drains completed captures from the last scheduler action, including
    /// tentative target rows and failed draft proposals. No native values escape.
    fn take_control_captures(&mut self) -> Vec<SpeculativePredictionCapture> {
        Vec::new()
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

    /// Total current and future sampler/RNG storage for a continuation horizon.
    /// Unknown custom samplers must leave the upper allowance unavailable.
    fn continuation_storage_bytes(
        &self,
        _target: Option<&Self::RandomState>,
        _draft: Option<&Self::DraftRandomness>,
        _additional_tokens: u64,
    ) -> Option<u64> {
        None
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

    /// Restricts position scores against the evolving commitment controller.
    /// Preserves an admissible selected identity without causal penalties,
    /// truncation, temperature scaling or another random draw. A mask identity
    /// selects masked-token preservation; absence selects entropy-policy argmax.
    /// Replacing a masked selection requires dependent-suffix refinement.
    fn finalize_refinement_position<'a>(
        &mut self,
        _logits: &Self::Logits,
        _selected: u32,
        _mask_token_id: Option<u32>,
        _history: &[u32],
        _context: Self::Context<'a>,
    ) -> Result<Option<(u32, Self::Distribution)>, Self::Error>
    where
        Self: 'a,
    {
        Ok(None)
    }

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
    evidence: &SpeculativeProposalEvidence<S::Distribution>,
    proposed: u32,
    temperature: f32,
    randomness: Option<&mut S::RandomState>,
    context: S::Context<'a>,
) -> Result<ProposalDecision, S::Error>
where
    S: SpeculativeSampling + 'a,
{
    let mut randomness = randomness;
    if temperature == 0.0 || matches!(evidence, SpeculativeProposalEvidence::TargetMatching) {
        let chosen = sampler.sample(
            target,
            temperature,
            if temperature == 0.0 { None } else { randomness },
            SamplingPlacement::Target,
            context,
        )?;
        return Ok(if chosen == proposed {
            ProposalDecision::Accept
        } else {
            ProposalDecision::Reject(chosen)
        });
    }

    let SpeculativeProposalEvidence::CausalDistribution(draft) = evidence else {
        unreachable!("target matching is resolved by target sampling")
    };
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

/// Verification evidence for a proposal. Position scores are never interpreted
/// as causal draft probabilities.
#[derive(Debug, Clone)]
pub enum SpeculativeProposalEvidence<D> {
    /// Processed causal assistant distribution used for likelihood-ratio verification.
    CausalDistribution(D),
    /// Sample the processed target and accept only a matching token.
    TargetMatching,
}

/// One proposed token and the evidence defining its verification policy.
#[derive(Debug)]
pub struct SpeculativeProposal<D> {
    token: u32,
    evidence: SpeculativeProposalEvidence<D>,
}

impl<D> SpeculativeProposal<D> {
    /// Creates a causal assistant proposal for likelihood-ratio verification.
    pub const fn causal(token: u32, distribution: D) -> Self {
        Self {
            token,
            evidence: SpeculativeProposalEvidence::CausalDistribution(distribution),
        }
    }
    /// Creates a masked proposal without inventing a causal assistant distribution.
    pub const fn target_matching(token: u32) -> Self {
        Self {
            token,
            evidence: SpeculativeProposalEvidence::TargetMatching,
        }
    }
    /// Proposed token id.
    pub const fn token(&self) -> u32 {
        self.token
    }
    /// Evidence consumed by target verification.
    pub const fn evidence(&self) -> &SpeculativeProposalEvidence<D> {
        &self.evidence
    }
    /// Causal assistant distribution, absent for masked proposals.
    pub const fn distribution(&self) -> Option<&D> {
        match &self.evidence {
            SpeculativeProposalEvidence::CausalDistribution(distribution) => Some(distribution),
            SpeculativeProposalEvidence::TargetMatching => None,
        }
    }
}

/// Backend-owned assistant state paired with a portable proposal sequence.
pub struct SpeculativeDraftBlock<S, D> {
    /// Assistant state after producing every proposal.
    state: S,
    /// Ordered proposed tokens and opaque distributions.
    proposals: Vec<SpeculativeProposal<D>>,
}

impl<S, D> SpeculativeDraftBlock<S, D> {
    /// Creates one ordered assistant proposal block.
    pub fn new(state: S, proposals: Vec<SpeculativeProposal<D>>) -> Self {
        Self { state, proposals }
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
    assumed_prefix: Vec<u32>,
}

impl<S, D> SpeculativeOptimisticBranch<S, D> {
    /// Creates one tentative continuation tied to an assumed prefix.
    pub fn new(block: SpeculativeDraftBlock<S, D>, assumed_prefix: Vec<u32>) -> Self {
        Self {
            block,
            assumed_prefix,
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

    /// Total retained storage through a further token horizon, including the
    /// installed state. Observation must not fork, advance or drain that state.
    fn continuation_storage_bytes(&self, _additional_tokens: u64) -> Option<u64> {
        None
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

/// Object-safe forkable semantic state used by speculative transactions.
///
/// This interface owns decoded semantic events and never exposes a backend
/// tensor, stream, completion, or error type.
pub trait SpeculativeSemanticState {
    /// Complete conservative bytes retained by a semantic fork, or unknown.
    fn control_snapshot_bytes(&self) -> Option<u64> {
        None
    }

    /// Total current and future semantic storage, including events staged until
    /// commitment. Unknown custom parser or decoder growth remains unavailable.
    fn continuation_storage_bytes(&self, _additional_tokens: u64) -> Option<u64> {
        None
    }

    /// Forks the exact committed prefix for tentative verification.
    fn fork_box(&self) -> Result<Box<dyn SpeculativeSemanticState>, SpeculativeOutputError>;
    /// Stages one token and reports whether a stop sequence matched.
    fn push_token(&mut self, token: u32) -> Result<bool, SpeculativeOutputError>;
    /// Stages normal terminal output.
    fn finish(&mut self, reason: FinishReason) -> Result<(), SpeculativeOutputError>;
    /// Stages cancellation output.
    fn cancel(&mut self) -> Result<(), SpeculativeOutputError>;
    /// Drains events authorized by the next exact commit boundary.
    fn take_events(&mut self) -> Vec<crate::generation::SemanticEvent>;
}

/// Optional transactional semantic state shared by plain and structured speculative decoding.
pub struct SpeculativeSemanticConstraint {
    state: Option<Box<dyn SpeculativeSemanticState>>,
}

impl SpeculativeSemanticConstraint {
    /// Creates an unconstrained output state for token-only generation.
    pub const fn plain() -> Self {
        Self { state: None }
    }

    /// Creates a transactional structured-output state.
    pub fn semantic(state: Box<dyn SpeculativeSemanticState>) -> Self {
        Self { state: Some(state) }
    }
}

impl SpeculativeConstraint for SpeculativeSemanticConstraint {
    fn continuation_storage_bytes(&self, additional_tokens: u64) -> Option<u64> {
        match &self.state {
            Some(state) => state.continuation_storage_bytes(additional_tokens),
            None => Some(0),
        }
    }

    fn control_snapshot_bytes(&self) -> Option<u64> {
        match &self.state {
            Some(state) => state.control_snapshot_bytes(),
            None => Some(0),
        }
    }

    fn fork(&self) -> Result<Self, SpeculativeOutputError> {
        Ok(Self {
            state: self
                .state
                .as_ref()
                .map(|state| state.fork_box())
                .transpose()?,
        })
    }

    fn push_token(&mut self, token: u32) -> Result<bool, SpeculativeOutputError> {
        self.state
            .as_mut()
            .map(|state| state.push_token(token))
            .transpose()
            .map(|matched| matched.unwrap_or(false))
    }

    fn finish(&mut self, reason: FinishReason) -> Result<(), SpeculativeOutputError> {
        if let Some(state) = &mut self.state {
            state.finish(reason)?;
        }
        Ok(())
    }
}

/// Fallible atomic committed-token batch callback.
type SpeculativeTokenCallback<'a> = dyn FnMut(&[u32]) -> Result<(), SpeculativeOutputError> + 'a;

/// Core-owned committed-token and semantic-event publication adapter.
pub struct SpeculativeCallbackPublisher<'a> {
    on_tokens: Box<SpeculativeTokenCallback<'a>>,
    on_event: Option<Box<dyn FnMut(crate::generation::SemanticEvent) + 'a>>,
}

impl<'a> SpeculativeCallbackPublisher<'a> {
    /// Publishes committed token ids without decoded semantic events.
    /// The callback is one atomic publication attempt: returning an error must
    /// leave every token in the supplied batch unpublished.
    pub fn tokens(
        on_tokens: impl FnMut(&[u32]) -> Result<(), SpeculativeOutputError> + 'a,
    ) -> Self {
        Self {
            on_tokens: Box::new(on_tokens),
            on_event: None,
        }
    }

    /// Publishes transactional semantic events and ignores raw token callbacks.
    pub fn semantic(on_event: impl FnMut(crate::generation::SemanticEvent) + 'a) -> Self {
        Self {
            on_tokens: Box::new(|_| Ok(())),
            on_event: Some(Box::new(on_event)),
        }
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
                state.cancel()?;
            }
        }
        (self.on_tokens)(tokens)?;
        if let (Some(state), Some(on_event)) = (&mut constraint.state, &mut self.on_event) {
            for event in state.take_events() {
                on_event(event);
            }
        }
        let cancellation_after_callbacks =
            !cancellation_won && cancellation.is_cancelled() && !sequence_finished;
        if cancellation_after_callbacks {
            if let (Some(state), Some(on_event)) = (&mut constraint.state, &mut self.on_event) {
                state.cancel()?;
                for event in state.take_events() {
                    on_event(event);
                }
            }
        }
        Ok(cancellation_won || cancellation_after_callbacks)
    }

    fn publish_cancelled(
        &mut self,
        constraint: &mut SpeculativeSemanticConstraint,
    ) -> Result<(), SpeculativeOutputError> {
        if let (Some(state), Some(on_event)) = (&mut constraint.state, &mut self.on_event) {
            state.cancel()?;
            for event in state.take_events() {
                on_event(event);
            }
        }
        Ok(())
    }
}

/// Canonical speculative sampler, sequence, constraint, and output sink.
pub struct SpeculativeOutputRuntime<S, C, P> {
    sampler: S,
    sequence: GenerationSequence,
    constraint: C,
    publisher: P,
    cancellation: GenerationCancellationToken,
    lifecycle_observer: Option<Arc<dyn SpeculativeLifecycleObserver>>,
    control_handle: Option<crate::execution_control::GenerationControlHandle>,
    decoding: Option<crate::decoding::DecodingControl>,
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
        sequence: GenerationSequence,
        constraint: C,
        publisher: P,
        cancellation: GenerationCancellationToken,
    ) -> Self {
        Self {
            sampler,
            sequence,
            constraint,
            publisher,
            cancellation,
            lifecycle_observer: None,
            control_handle: None,
            decoding: None,
        }
    }

    /// Shares commands with uninterrupted advancement before prefill starts.
    pub fn with_control_handle(
        mut self,
        handle: Option<crate::execution_control::GenerationControlHandle>,
    ) -> Self {
        self.control_handle = handle;
        self
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
    pub const fn sequence(&self) -> &GenerationSequence {
        &self.sequence
    }

    /// Mutable canonical committed sequence.
    pub const fn sequence_mut(&mut self) -> &mut GenerationSequence {
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
        let mut sequence = self.sequence.clone();
        self.cancel_candidate(&mut constraint, &mut sequence)?;
        self.constraint = constraint;
        self.sequence = sequence;
        Ok(())
    }

    /// Installs logical state only after its matching backend boundary committed.
    pub fn install_committed_state(
        &mut self,
        sampler: S,
        constraint: C,
        sequence: GenerationSequence,
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
        sequence: &mut GenerationSequence,
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
        sequence: &mut GenerationSequence,
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
    pub(crate) fn into_parts(self) -> (S, GenerationSequence, C, P) {
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
    sequence: GenerationSequence,
    /// Tentatively advanced target randomness.
    target_randomness: Option<R>,
    /// Number of accepted proposals.
    accepted_proposals: usize,
    /// Tokens visible after cache commit.
    committed_tokens: Vec<u32>,
    /// Exact verification inputs retained by cache commit.
    verified_inputs: usize,
    /// Target bonus token, when full acceptance produced one.
    bonus_token: Option<u32>,
    /// Terminal reason after this round.
    finish_reason: Option<FinishReason>,
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
) -> Result<Vec<SpeculativeProposal<S::Distribution>>, SpeculativeDriverError<E::Error>>
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
) -> Result<Vec<SpeculativeProposal<S::Distribution>>, SpeculativeDriverError<E::Error>>
where
    E: SpeculativeExecutor + 'a,
    S: SpeculativeSampling<Logits = E::Logits, Error = E::Error, Context<'a> = E::Context<'a>> + 'a,
{
    let mut branch_sampler = sampler.clone();
    let mut history = Vec::with_capacity(base_history.len() + count);
    history.extend_from_slice(base_history);
    let mut proposals: Vec<SpeculativeProposal<S::Distribution>> = Vec::with_capacity(count);
    for offset in 0..count {
        let local = (|| -> Result<bool, SpeculativeDriverError<E::Error>> {
            let previous = proposals
                .last()
                .map_or(first_previous, |proposal| proposal.token);
            let raw = with_activation_origin(
                executor,
                origin.map(|o| o.with_prefix(&history)),
                |executor| executor.proposal_step(state, previous, context),
            )?;
            let proposal = match raw {
                SpeculativeProposalStep::TargetMatchingToken(token) => {
                    SpeculativeProposal::target_matching(token)
                }
                SpeculativeProposalStep::CausalLogits(raw) => {
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
                    SpeculativeProposal::causal(token, distribution)
                }
            };
            let token = proposal.token;
            proposals.push(proposal);
            history.push(token);
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
pub fn resolve_round<'a, E, S, C>(
    executor: &E,
    verification: &E::Verification,
    mut proposals: Vec<SpeculativeProposal<S::Distribution>>,
    sampler: &S,
    sequence: &GenerationSequence,
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
    let mut draft_distributions = proposals
        .iter_mut()
        .filter_map(|proposal| match &mut proposal.evidence {
            SpeculativeProposalEvidence::CausalDistribution(distribution) => Some(distribution),
            SpeculativeProposalEvidence::TargetMatching => None,
        })
        .collect::<Vec<_>>();
    if !draft_distributions.is_empty() {
        sampler.prepare_verification(&mut draft_distributions, temperature, context)?;
    }
    let proposal_count = proposals.len();
    let mut sampler = sampler.clone();
    let mut sequence = sequence.clone();
    let mut constraint = constraint.fork().map_err(SpeculativeDriverError::Output)?;
    let mut target_randomness = target_randomness.cloned();
    let mut history = sequence.tokens().to_vec();
    let mut round = crate::generation::VerificationCursor::new(proposal_count)
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
            &proposal.evidence,
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
                history.push(proposal.token);
                finish_reason = commit_terminal_token(
                    &mut sequence,
                    &mut sampler,
                    &mut constraint,
                    proposal.token,
                )?;
                round
                    .stage(
                        crate::generation::VerificationDecision::Accept(proposal.token),
                        finish_reason.is_some(),
                    )
                    .map_err(SpeculativeDriverError::Generation)?;
                round
                    .commit(proposal.token)
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
                    .stage(
                        crate::generation::VerificationDecision::Replacement(replacement),
                        finish_reason.is_some(),
                    )
                    .map_err(SpeculativeDriverError::Generation)?;
                round
                    .commit(replacement)
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
            .stage(
                crate::generation::VerificationDecision::Bonus(chosen),
                finish_reason.is_some(),
            )
            .map_err(SpeculativeDriverError::Generation)?;
        round
            .commit(chosen)
            .map_err(SpeculativeDriverError::Generation)?;
        bonus_token = Some(chosen);
    }
    let plan = round
        .commit_plan()
        .map_err(SpeculativeDriverError::Generation)?;
    Ok(ResolvedSpeculativeRound {
        sampler,
        constraint,
        sequence,
        target_randomness,
        accepted_proposals: plan.accepted_proposals,
        committed_tokens: plan.committed_tokens.to_vec(),
        verified_inputs: plan.verified_inputs,
        bonus_token,
        finish_reason,
    })
}

/// Submits one exact target verification and takes ownership of its resources.
/// An empty proposal list appends the pending committed token and exposes one
/// causal target row, using the same completion and rollback contract.
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
    let mut input_tokens = Vec::with_capacity(block.proposals.len() + 1);
    input_tokens.push(last_committed_token);
    input_tokens.extend(block.proposals.iter().map(|proposal| proposal.token));
    let checkpoint = executor
        .checkpoint(cache)
        .map_err(SpeculativeDriverError::Backend);
    let checkpoint = coordination::ready_result(executor, checkpoint, context)?;
    let submission = match executor.submit_verification(&input_tokens, cache, context) {
        Ok(submission) => submission,
        Err(error) => {
            executor.restore_checkpoint(cache, &checkpoint, context)?;
            return Err(error.into());
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
    })
}

/// Settled causal state and next-token scores after appending a committed suffix.
/// Sampling, publication and proposal randomness do not advance at this boundary.
pub struct SettledCommittedSuffix<S, L, T> {
    /// Source seed aligned with the completed canonical append.
    pub state: S,
    /// Causal next-token row retained for the next generation action.
    pub logits: L,
    /// Additional input evaluations needed to retain the accepted prefix.
    pub replayed_tokens: usize,
    /// Component observations collected after exact completion.
    pub telemetry: T,
}

/// Appends a nonempty committed suffix through the executor's zero-proposal
/// transaction. On failure the checkpoint restores the canonical frontier; an
/// executor whose native disposition prevents restoration returns that failure.
#[allow(clippy::type_complexity)]
pub fn settle_committed_suffix<'a, E: SpeculativeExecutor + 'a>(
    executor: &mut E,
    cache: &mut E::Cache,
    state: &E::TargetState,
    tokens: &[u32],
    wait: BoundedCompletionWait,
    mut observe: impl FnMut(SpeculativeLifecycleStage) -> Result<(), SpeculativeOutputError>,
    context: E::Context<'a>,
) -> Result<
    SettledCommittedSuffix<E::TargetState, E::Logits, E::Telemetry>,
    SpeculativeDriverError<E::Error>,
> {
    let first = coordination::ready_result(
        executor,
        tokens
            .first()
            .copied()
            .ok_or(SpeculativeDriverError::Generation(
                GenerationError::EmptyCommittedSuffix,
            )),
        context,
    )?;
    let admitted = if E::Completion::supports_cancellation(wait.cancellation()) {
        Ok(())
    } else {
        Err(SpeculativeDriverError::UnsupportedCompletionCancellation {
            cancellation: wait.cancellation(),
        })
    };
    coordination::ready_result(executor, admitted, context)?;
    let checkpoint = executor
        .checkpoint(cache)
        .map_err(SpeculativeDriverError::Backend);
    let checkpoint = coordination::ready_result(executor, checkpoint, context)?;
    let mut retained_output = None;
    let mut completion = None;
    let result = (|| {
        let observed =
            observe(SpeculativeLifecycleStage::Execution).map_err(SpeculativeDriverError::Output);
        coordination::ready_result(executor, observed, context)?;
        let draft = executor
            .begin_proposal(state, first, 0, context)
            .map_err(SpeculativeDriverError::Backend);
        let draft = coordination::ready_result(executor, draft, context)?;
        let submitted = executor
            .submit_verification(tokens, cache, context)
            .map_err(SpeculativeDriverError::Backend);
        let submission = coordination::ready_result(executor, submitted, context)?;
        completion = Some(submission.completion);
        retained_output = Some(submission.output);
        coordination::settle_completion::<E>(
            &mut completion,
            Ok(wait),
            observe(SpeculativeLifecycleStage::Completion),
        )?;
        let computed = (|| {
            observe(SpeculativeLifecycleStage::Observation)
                .map_err(SpeculativeDriverError::Output)?;
            let output = retained_output
                .as_mut()
                .expect("suffix verification is retained");
            let telemetry = executor.take_verification_telemetry(output)?;
            let logits = executor.verification_logits(output, tokens.len() - 1, context)?;
            Ok((telemetry, logits))
        })();
        let (telemetry, logits) = coordination::ready_result(executor, computed, context)?;
        let observed = observe(SpeculativeLifecycleStage::CachePersistence)
            .map_err(SpeculativeDriverError::Output);
        coordination::ready_result(executor, observed, context)?;
        let committed = executor
            .commit_verification(
                retained_output
                    .take()
                    .expect("suffix verification is retained"),
                draft,
                cache,
                &checkpoint,
                tokens.len(),
                context,
            )
            .map_err(SpeculativeDriverError::Backend);
        let committed = coordination::ready_result(executor, committed, context)?;
        Ok(SettledCommittedSuffix {
            state: committed.state,
            logits,
            replayed_tokens: committed.replayed_tokens,
            telemetry,
        })
    })();
    if result.is_err() {
        executor.restore_checkpoint(cache, &checkpoint, context)?;
    }
    // Retain successful completion and unconsumed outputs through publication or rollback.
    drop(completion);
    drop(retained_output);
    result
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
    /// Published suffix that has not yet entered causal KV.
    pending_committed_tokens: Vec<u32>,
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
            stats.verification_in_flight_time += submitted.elapsed();
            runtime
                .observe_lifecycle(SpeculativeLifecycleStage::Observation)
                .map_err(SpeculativeDriverError::Output)?;
            let mut canonical_proposal_prefix = runtime.sequence().tokens().to_vec();
            canonical_proposal_prefix.extend(
                block
                    .as_ref()
                    .expect("pending draft")
                    .proposals
                    .iter()
                    .map(|proposal| proposal.token),
            );
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
            stats.accepted_tokens += resolved.accepted_proposals;
            stats.accept_lens.push(resolved.accepted_proposals);
            stats.rounds += 1;
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
        let pending_committed_tokens = if committed_tokens.len() + 1 > resolved.verified_inputs {
            committed_tokens.last().copied().into_iter().collect()
        } else {
            Vec::new()
        };
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
        stats.target_tokens += commit.replayed_tokens;
        stats.emitted_tokens += committed_tokens.len();
        let target_randomness = resolved.target_randomness;
        if terminal.is_some() {
            if let Some(control) = &runtime.decoding {
                control.finish();
            }
        }
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
            pending_committed_tokens,
        })
    })();
    if result.is_err() {
        // A failed restore is an explicit indeterminate backend error. It does
        // not permit publication or reuse of the original native transaction.
        executor.restore_checkpoint(cache, &checkpoint, context)?;
    }
    result
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
            stats.verification_in_flight_time += submitted.elapsed();
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
            Ok((constraint, runtime.sequence().clone(), telemetry))
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
        stats.target_tokens += commit.replayed_tokens;
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
    if result.is_err() {
        executor.restore_checkpoint(cache, &checkpoint, context)?;
    }
    result
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
    let optimistic_tokens = branch
        .block
        .proposals
        .iter()
        .map(|proposal| proposal.token)
        .collect::<Vec<_>>();
    let decision = crate::generation::resolve_optimistic_reuse(
        &branch.assumed_prefix,
        canonical_prefix,
        &optimistic_tokens,
        bonus,
        terminal,
    )?;
    stats.optimistic_target_bonus_tokens += 1;
    if decision == crate::generation::OptimisticReuseDecision::DiscardTerminal {
        discard_branch(stats, Some(branch));
        return Ok(SpeculativeContinuation::None);
    }
    let drafted = branch.block.proposals.len();
    let SpeculativeDraftBlock { state, proposals } = branch.block;
    let mut proposals = proposals.into_iter();
    let _matched_or_discarded = proposals
        .next()
        .expect("validated optimistic branch is non-empty");
    Ok(match decision {
        crate::generation::OptimisticReuseDecision::DiscardMismatch => {
            stats.optimistic_bonus_mismatches += 1;
            stats.discarded_optimistic_tokens += drafted;
            stats.discarded_optimistic_blocks += 1;
            SpeculativeContinuation::None
        }
        crate::generation::OptimisticReuseDecision::MatchedConsumed => {
            stats.optimistic_bonus_matches += 1;
            stats.consumed_optimistic_tokens += 1;
            SpeculativeContinuation::None
        }
        crate::generation::OptimisticReuseDecision::MatchedRetained => {
            stats.optimistic_bonus_matches += 1;
            stats.consumed_optimistic_tokens += 1;
            let proposals = proposals.collect::<Vec<_>>();
            stats.draft_tokens += proposals.len();
            stats.reused_optimistic_tokens += proposals.len();
            stats.reused_optimistic_blocks += 1;
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
        stats.discarded_optimistic_tokens += branch.block.proposals.len();
        stats.discarded_optimistic_blocks += 1;
    }
}

fn discard_continuation<S, D>(
    stats: &mut SpeculativeStats,
    continuation: SpeculativeContinuation<S, D>,
) {
    if let SpeculativeContinuation::Promoted(block) = continuation {
        stats.discarded_optimistic_tokens += block.proposals.len();
        stats.discarded_optimistic_blocks += 1;
        stats.draft_tokens = stats.draft_tokens.saturating_sub(block.proposals.len());
        stats.reused_optimistic_tokens = stats
            .reused_optimistic_tokens
            .saturating_sub(block.proposals.len());
        stats.reused_optimistic_blocks = stats.reused_optimistic_blocks.saturating_sub(1);
    }
}

fn commit_terminal_token<S, C>(
    sequence: &mut GenerationSequence,
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

/// Mutable canvas commitment cursor; selected rows remain tentative until
/// individually published. Constraint replacement discards only future rows.
#[derive(Clone)]
struct RefinementContinuation<L> {
    positions: Option<Vec<(u32, L)>>,
    committed: Vec<u32>,
    width: usize,
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
    control_identity: Arc<()>,
    id: SpeculativeRequestId,
    cache: GenerationCache<'cache, E::Cache>,
    config: GenerationRequestConfig,
    runtime: SpeculativeOutputRuntime<S, C, P>,
    target_randomness: Option<S::RandomState>,
    draft_randomness: Option<S::DraftRandomness>,
    stats: SpeculativeStats,
    started: Instant,
    target_state: Option<E::TargetState>,
    pending_committed_tokens: Vec<u32>,
    next_logits: Option<E::Logits>,
    refinement: Option<RefinementContinuation<E::Logits>>,
    decoding: crate::decoding::DecodingControl,
    block: Option<SpeculativeDraftBlock<E::DraftState, S::Distribution>>,
    pending: Option<PendingSpeculativeVerification<E, S::Distribution>>,
    lifecycle: SpeculativeRequestLifecycle,
    completed_decision: Option<CompletedDecisionProgress>,
}

impl<'cache, E, S, C, P> SpeculativeRequest<'cache, E, S, C, P>
where
    E: SpeculativeExecutor,
    S: SpeculativeSampling<Logits = E::Logits, Error = E::Error>,
    C: SpeculativeConstraint,
    P: SpeculativePublisher<C>,
{
    pub(crate) fn completed_source_cache(&self) -> &E::Cache {
        &self.cache
    }

    pub(crate) fn completed_source_parts(
        &mut self,
    ) -> (&mut E::Cache, &mut SpeculativeOutputRuntime<S, C, P>) {
        (&mut self.cache, &mut self.runtime)
    }

    pub(crate) fn take_completed_decision(&mut self) -> Option<CompletedDecisionProgress> {
        self.completed_decision.take()
    }

    pub(crate) fn completed_source_control(&self) -> &crate::decoding::DecodingControl {
        &self.decoding
    }

    pub(crate) fn replace_completed_cancellation(&mut self, token: GenerationCancellationToken) {
        self.runtime.cancellation = token;
    }

    pub(crate) fn restore_completed_sequence(&mut self, sequence: GenerationSequence) {
        self.config.max_tokens = sequence.max_tokens();
        self.runtime.sequence = sequence;
        self.completed_decision = None;
        self.lifecycle = if self.runtime.sequence.is_finished() || self.decoding.state().terminal {
            SpeculativeRequestLifecycle::completed()
        } else {
            let mut lifecycle = SpeculativeRequestLifecycle::new();
            lifecycle
                .transition(SpeculativeRequestStatus::ReadyToDraft)
                .expect("prepared source becomes ready");
            lifecycle
        };
    }

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

    /// Monotone request age for separating active work from controlled pauses.
    pub fn submission_age(&self) -> Duration {
        self.started.elapsed()
    }

    /// Portable request statistics.
    pub const fn stats(&self) -> &SpeculativeStats {
        &self.stats
    }

    /// Canonical committed token sequence.
    pub const fn sequence(&self) -> &GenerationSequence {
        self.runtime.sequence()
    }

    /// Authoritative suffix still awaiting one causal append.
    pub fn pending_committed_tokens(&self) -> &[u32] {
        &self.pending_committed_tokens
    }

    /// Whether a partially committed canvas still owns tentative positions.
    pub fn has_tentative_refinement(&self) -> bool {
        self.refinement.is_some()
    }

    /// Active mode, queued command and most recent settled acknowledgement.
    pub fn decoding_state(&self) -> crate::decoding::DecodingState {
        self.decoding.state()
    }

    /// Queues one validated transition over the already retained source.
    pub fn request_decoding_mode(
        &self,
        mode: crate::decoding::DecodingMode,
    ) -> Result<(), crate::decoding::DecodingModeError> {
        if self.runtime.cancellation().is_cancelled() || self.lifecycle.is_terminal() {
            return Err(crate::decoding::DecodingModeError::Terminal);
        }
        self.decoding.request_decoding_mode(mode)
    }

    /// Shares the same mailbox with cross-thread uninterrupted control.
    pub fn bind_control_handle(
        &self,
        handle: &crate::execution_control::GenerationControlHandle,
    ) -> Result<(), crate::decoding::DecodingModeError> {
        handle.bind_decoding(self.decoding.clone(), Some(self.runtime.cancellation()))
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
            .map_err(SpeculativeDriverError::Generation)?;
        if self.lifecycle.is_terminal() {
            self.decoding.finish();
        }
        Ok(())
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
                self.refinement = None;
                self.stats.elapsed = self.started.elapsed();
            }
        }
        self.decoding.finish();
        Ok(())
    }

    fn candidate<'context>(
        &self,
        executor: &E,
        optimistic_execution_available: bool,
        completion_wait: BoundedCompletionWait,
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
            || self.decoding.state().pending.is_some()
            || !matches!(self.config.decoding_mode, crate::decoding::DecodingMode::SelfSpeculation(options) if options.request().scheduler.lookahead_blocks > 0)
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
            let assumed_len = self.runtime.sequence().tokens().len() + block.proposals.len();
            let mut assumed_prefix = Vec::with_capacity(assumed_len);
            assumed_prefix.extend_from_slice(self.runtime.sequence().tokens());
            assumed_prefix.extend(block.proposals.iter().map(|proposal| proposal.token));
            executor.supports_exact_optimistic_promotion()
                && self.runtime.sampler().supports_exact_optimistic_promotion()
                && !self.stats.adaptive_lookahead_disabled
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

    fn settle_requested_mode<'context>(
        &mut self,
        executor: &mut E,
        ceiling: SpeculativeSchedulerOptions,
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
        let Some(requested) = self.decoding.state().pending else {
            return Ok(false);
        };
        if self.lifecycle.status() != SpeculativeRequestStatus::ReadyToDraft
            || self.pending.is_some()
            || self.refinement.is_some()
        {
            return Ok(false);
        }
        if executor.uses_completed_decisions() {
            if executor
                .settle_decision_source(
                    &mut self.cache,
                    self.config.decoding_mode,
                    requested,
                    context,
                )
                .map_err(SpeculativeDriverError::Backend)?
                != Some(true)
            {
                return Ok(false);
            }
            if self
                .decoding
                .acknowledge(requested, self.sequence().tokens().len() as u64, || {
                    self.runtime.cancellation().is_cancelled()
                })
                .is_none()
            {
                self.request_cancellation()?;
                return Ok(false);
            }
            self.config.decoding_mode = requested;
            return Ok(true);
        }
        if !self.pending_committed_tokens.is_empty() {
            let origin = self.activation_origin(executor, false);
            let settled = with_activation_origin(executor, origin, |executor| {
                settle_committed_suffix(
                    executor,
                    &mut self.cache,
                    self.target_state.as_ref().expect("ready target seed"),
                    &self.pending_committed_tokens,
                    self.config
                        .scheduler_options(ceiling)
                        .completion_wait()
                        .map_err(SpeculativeDriverError::Generation)?,
                    |stage| self.runtime.observe_lifecycle(stage),
                    context,
                )
            })?;
            settled.telemetry.record(&mut self.stats);
            self.stats.target_tokens +=
                self.pending_committed_tokens.len() + settled.replayed_tokens;
            self.target_state = Some(settled.state);
            self.pending_committed_tokens.clear();
            self.next_logits = Some(settled.logits);
        }
        if let Some(block) = self.block.take() {
            discard_continuation(&mut self.stats, SpeculativeContinuation::Promoted(block));
        }
        let configured = executor
            .configure_decoding_mode(
                self.target_state.as_mut().expect("ready target seed"),
                requested,
                self.config.temperature,
            )
            .map_err(SpeculativeDriverError::Generation);
        coordination::ready_result(executor, configured, context)?;
        if self
            .decoding
            .acknowledge(requested, self.sequence().tokens().len() as u64, || {
                self.runtime.cancellation().is_cancelled()
            })
            .is_none()
        {
            self.request_cancellation()?;
            return Ok(false);
        }
        self.config.decoding_mode = requested;
        Ok(true)
    }

    fn commit_retained_next_scores<'context>(
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
        let computed = (|| {
            self.runtime
                .observe_lifecycle(SpeculativeLifecycleStage::Execution)
                .map_err(SpeculativeDriverError::Output)?;
            let mut sampler = self.runtime.sampler().clone();
            let mut constraint = self
                .runtime
                .constraint()
                .fork()
                .map_err(SpeculativeDriverError::Output)?;
            let mut sequence = self.sequence().clone();
            let mut randomness = self.target_randomness.clone();
            let distribution = sampler.process_logits(
                self.next_logits.as_ref().expect("retained causal row"),
                self.config.temperature,
                sequence.tokens(),
                SamplingPlacement::Target,
                context,
            )?;
            let token = sampler.sample(
                &distribution,
                self.config.temperature,
                randomness.as_mut(),
                SamplingPlacement::Target,
                context,
            )?;
            sampler.update_sampler_state(
                &distribution,
                token,
                SamplingPlacement::Target,
                context,
            )?;
            let terminal =
                commit_terminal_token(&mut sequence, &mut sampler, &mut constraint, token)?;
            Ok((sampler, constraint, sequence, randomness, token, terminal))
        })();
        let (sampler, mut constraint, mut sequence, randomness, token, terminal) =
            coordination::ready_result(executor, computed, context)?;
        self.stats
            .submission_to_first_token
            .get_or_insert_with(|| self.started.elapsed());
        if terminal.is_some() {
            self.decoding.finish();
        }
        let cancelled = coordination::publish_candidate(
            executor,
            &mut self.runtime,
            &mut constraint,
            &mut sequence,
            &[token],
            context,
        )?;
        self.runtime
            .install_committed_state(sampler, constraint, sequence);
        self.target_randomness = randomness;
        self.pending_committed_tokens.push(token);
        self.next_logits = None;
        self.stats.emitted_tokens += 1;
        if cancelled {
            self.transition(SpeculativeRequestStatus::Cancelled)?;
        } else if terminal.is_some() {
            self.transition(SpeculativeRequestStatus::Completed)?;
        }
        if self.lifecycle.is_terminal() {
            self.stats.elapsed = self.started.elapsed();
        }
        Ok(())
    }

    fn coordinate_refinement_cancellation<'a>(
        &mut self,
        executor: &mut E,
        context: E::Context<'a>,
    ) -> Result<bool, SpeculativeDriverError<E::Error>> {
        let cancelled = self
            .decoding
            .coordinate(
                self.runtime.cancellation().is_cancelled(),
                self.sequence().tokens().len() as u64,
                Some(
                    self.config
                        .max_tokens
                        .saturating_sub(self.sequence().tokens().len()),
                ),
                |local| executor.coordinate_decoding_control(local, context),
            )
            .map_err(|error| match error {
                crate::decoding::DecodingControlError::Agreement(error) => {
                    SpeculativeDriverError::Preparation(error)
                }
                crate::decoding::DecodingControlError::Mode(error) => {
                    SpeculativeDriverError::Generation(GenerationError::DecodingMode(error))
                }
            })?;
        if cancelled {
            self.runtime.cancellation().cancel();
            self.request_cancellation()?;
        }
        Ok(cancelled)
    }

    fn refine_committed<'context>(
        &mut self,
        executor: &mut E,
        policy: Option<crate::refinement::RefinementPolicy>,
        ceiling: SpeculativeSchedulerOptions,
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
        if self.refinement.is_none() && !self.pending_committed_tokens.is_empty() {
            let origin = self.activation_origin(executor, false);
            let settled = with_activation_origin(executor, origin, |executor| {
                settle_committed_suffix(
                    executor,
                    &mut self.cache,
                    self.target_state.as_ref().expect("ready target seed"),
                    &self.pending_committed_tokens,
                    self.config
                        .scheduler_options(ceiling)
                        .completion_wait()
                        .map_err(SpeculativeDriverError::Generation)?,
                    |stage| self.runtime.observe_lifecycle(stage),
                    context,
                )
            })?;
            settled.telemetry.record(&mut self.stats);
            self.stats.target_tokens +=
                self.pending_committed_tokens.len() + settled.replayed_tokens;
            self.target_state = Some(settled.state);
            self.pending_committed_tokens.clear();
            self.next_logits = Some(settled.logits);
        }
        let unavailable = || {
            SpeculativeDriverError::Generation(GenerationError::DecodingMode(
                crate::decoding::DecodingModeError::Unsupported("refinement commitment policy"),
            ))
        };
        if self
            .refinement
            .as_ref()
            .is_none_or(|canvas| canvas.positions.is_none())
        {
            let observed = self
                .runtime
                .observe_lifecycle(SpeculativeLifecycleStage::Execution)
                .map_err(SpeculativeDriverError::Output);
            coordination::ready_result(executor, observed, context)?;
            let pinned = self
                .refinement
                .as_ref()
                .map_or(&[][..], |canvas| canvas.committed.as_slice());
            let origin = self.activation_origin(executor, false);
            let refined = with_activation_origin(executor, origin, |executor| {
                executor.refine_block(
                    self.target_state.as_mut().expect("ready target seed"),
                    self.next_logits.as_ref().expect("settled causal row"),
                    pinned,
                    policy,
                    self.config.temperature,
                    context,
                )
            })
            .map_err(SpeculativeDriverError::Backend);
            let refined =
                coordination::ready_result(executor, refined, context)?.ok_or_else(unavailable)?;
            let refined = match refined {
                RefinementProgress::Pending {
                    width,
                    evaluated_positions,
                    iterations,
                } => {
                    if width <= pinned.len()
                        || self
                            .refinement
                            .as_ref()
                            .is_some_and(|canvas| canvas.width != width)
                    {
                        return Err(SpeculativeDriverError::Coordination(
                            "refinement changed its admitted canvas geometry",
                        ));
                    }
                    self.stats.refinement_tokens += evaluated_positions;
                    self.stats.refinement_steps += iterations as usize;
                    if self.refinement.is_none() {
                        self.refinement = Some(RefinementContinuation {
                            positions: None,
                            committed: Vec::new(),
                            width,
                        });
                    }
                    self.coordinate_refinement_cancellation(executor, context)?;
                    return Ok(());
                }
                RefinementProgress::Complete(refined) => refined,
            };
            self.stats.refinement_tokens += refined.evaluated_positions;
            self.stats.refinement_steps += refined.iterations as usize;
            let rows = refined.positions;
            if rows.len() <= pinned.len()
                || rows
                    .iter()
                    .zip(pinned)
                    .any(|((token, _), pinned)| token != pinned)
                || self
                    .refinement
                    .as_ref()
                    .is_some_and(|canvas| canvas.width != rows.len())
            {
                return Err(SpeculativeDriverError::Coordination(
                    "refinement changed its committed prefix or admitted geometry",
                ));
            }
            if self.coordinate_refinement_cancellation(executor, context)? {
                return Ok(());
            }
            match &mut self.refinement {
                Some(canvas) => canvas.positions = Some(rows),
                slot @ None => {
                    *slot = Some(RefinementContinuation {
                        width: rows.len(),
                        positions: Some(rows),
                        committed: Vec::new(),
                    })
                }
            }
            return Ok(());
        }
        let canvas = self.refinement.as_ref().expect("settled tentative canvas");
        let (selected, scores) =
            &canvas.positions.as_ref().expect("settled position rows")[canvas.committed.len()];
        let selected = *selected;
        let mut sampler = self.runtime.sampler().clone();
        let mut constraint = self
            .runtime
            .constraint()
            .fork()
            .map_err(SpeculativeDriverError::Output)?;
        let mut sequence = self.sequence().clone();
        let finalized = (|| {
            let (token, distribution) = sampler
                .finalize_refinement_position(
                    scores,
                    selected,
                    executor.decoding_capabilities().mask_token_id,
                    sequence.tokens(),
                    context,
                )?
                .ok_or_else(unavailable)?;
            sampler.update_sampler_state(
                &distribution,
                token,
                SamplingPlacement::Target,
                context,
            )?;
            let terminal =
                commit_terminal_token(&mut sequence, &mut sampler, &mut constraint, token)?;
            Ok((token, terminal))
        })();
        let (token, terminal) = coordination::ready_result(executor, finalized, context)?;
        if self.coordinate_refinement_cancellation(executor, context)? {
            return Ok(());
        }
        if terminal.is_some() {
            self.decoding.finish();
        }
        self.stats
            .submission_to_first_token
            .get_or_insert_with(|| self.started.elapsed());
        let cancelled = coordination::publish_candidate(
            executor,
            &mut self.runtime,
            &mut constraint,
            &mut sequence,
            &[token],
            context,
        )?;
        self.runtime
            .install_committed_state(sampler, constraint, sequence);
        self.stats.emitted_tokens += 1;
        self.pending_committed_tokens.push(token);
        let canvas = self.refinement.as_mut().expect("settled tentative canvas");
        canvas.committed.push(token);
        if canvas.committed.len() == canvas.width || terminal.is_some() || cancelled {
            self.refinement = None;
            self.next_logits = None;
        } else if executor.decoding_capabilities().masked_unmasking && token != selected {
            // A replacement becomes authoritative before any dependent token
            // can leave tentative state. The next action refines that suffix.
            canvas.positions = None;
        }
        if cancelled {
            self.transition(SpeculativeRequestStatus::Cancelled)?;
        } else if terminal.is_some() {
            self.transition(SpeculativeRequestStatus::Completed)?;
        }
        if self.lifecycle.is_terminal() {
            self.stats.elapsed = self.started.elapsed();
        }
        Ok(())
    }

    fn draft_committed<'context>(
        &mut self,
        executor: &mut E,
        ceiling: SpeculativeSchedulerOptions,
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
        if executor.uses_completed_decisions() {
            self.runtime
                .observe_lifecycle(SpeculativeLifecycleStage::Execution)
                .map_err(SpeculativeDriverError::Output)?;
            let decision = executor
                .advance_decision_source(
                    &mut self.cache,
                    self.target_state
                        .as_mut()
                        .expect("prepared decision source"),
                    self.runtime.sequence().remaining(),
                    context,
                )
                .map_err(SpeculativeDriverError::Backend)?
                .ok_or(SpeculativeDriverError::Coordination(
                    "completed decision source advancement",
                ))?;
            if self.runtime.cancellation().is_cancelled() {
                self.request_cancellation()?;
                return Ok(false);
            }
            match decision {
                CompletedDecisionProgress::Token {
                    token_id,
                    score_kind,
                } => {
                    let committed_position = self.sequence().tokens().len() as u64 + 1;
                    executor
                        .commit_decision_source(
                            &mut self.cache,
                            token_id,
                            committed_position,
                            context,
                        )
                        .map_err(SpeculativeDriverError::Backend)?;
                    let terminal = commit_terminal_token(
                        &mut self.runtime.sequence,
                        &mut self.runtime.sampler,
                        &mut self.runtime.constraint,
                        token_id,
                    )?;
                    let cancelled = self
                        .runtime
                        .publish_committed(&[token_id])
                        .map_err(SpeculativeDriverError::Output)?;
                    self.stats
                        .submission_to_first_token
                        .get_or_insert_with(|| self.started.elapsed());
                    self.stats.emitted_tokens += 1;
                    self.completed_decision = Some(CompletedDecisionProgress::Token {
                        token_id,
                        score_kind,
                    });
                    if cancelled {
                        self.transition(SpeculativeRequestStatus::Cancelled)?;
                    } else if terminal.is_some() {
                        self.transition(SpeculativeRequestStatus::Completed)?;
                    }
                }
                CompletedDecisionProgress::Progress(progress) => {
                    self.completed_decision = Some(CompletedDecisionProgress::Progress(progress));
                }
                CompletedDecisionProgress::Finished => {
                    self.transition(SpeculativeRequestStatus::Completed)?
                }
            }
            if self.lifecycle.is_terminal() {
                self.decoding.finish();
                self.stats.elapsed = self.started.elapsed();
            }
            return Ok(false);
        }
        if let crate::decoding::DecodingMode::Diffusion { policy } = self.config.decoding_mode {
            self.refine_committed(executor, policy, ceiling, context)?;
            return Ok(false);
        }
        if self.next_logits.is_some() {
            self.commit_retained_next_scores(executor, context)?;
            return Ok(false);
        }
        let configured = executor
            .configure_decoding_mode(
                self.target_state.as_mut().expect("ready target seed"),
                self.config.decoding_mode,
                self.config.temperature,
            )
            .map_err(SpeculativeDriverError::Generation);
        coordination::ready_result(executor, configured, context)?;
        let observed = self
            .runtime
            .observe_lifecycle(SpeculativeLifecycleStage::Execution)
            .map_err(SpeculativeDriverError::Output);
        coordination::ready_result(executor, observed, context)?;
        let target_count = self
            .config
            .proposal_capacity()
            .min(executor.max_proposals())
            .min(
                self.config
                    .max_tokens
                    .saturating_sub(self.runtime.sequence().tokens().len()),
            );
        if self.config.max_tokens <= self.runtime.sequence().tokens().len() {
            self.transition(SpeculativeRequestStatus::Completed)?;
            self.stats.elapsed = self.started.elapsed();
            return Ok(false);
        }

        let origin = self.activation_origin(executor, false);
        let mut block = if let Some(block) = self.block.take() {
            block
        } else {
            let last = *self
                .pending_committed_tokens
                .last()
                .expect("ready request retains its uncached committed token");
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
                proposals: Vec::new(),
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
            let mut history =
                Vec::with_capacity(self.runtime.sequence().tokens().len() + block.proposals.len());
            history.extend_from_slice(self.runtime.sequence().tokens());
            history.extend(block.proposals.iter().map(|proposal| proposal.token));
            let previous = block.proposals.last().map_or_else(
                || {
                    *self
                        .pending_committed_tokens
                        .last()
                        .expect("ready request retains its uncached committed token")
                },
                |proposal| proposal.token,
            );
            let proposals = propose_block_at(
                executor,
                self.runtime.sampler(),
                &mut block.state,
                previous,
                additional,
                &history,
                self.config.temperature,
                &self.config.eos_token_ids,
                self.draft_randomness.as_ref(),
                context,
                origin,
            )?;
            self.stats.draft_tokens += proposals.len();
            block.proposals.extend(proposals);
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
            .pending_committed_tokens
            .last()
            .expect("verification retains its uncached committed token");
        let origin = self.activation_origin(executor, false);
        let pending = with_activation_origin(executor, origin, |executor| {
            submit_verification_transaction(executor, &mut self.cache, last, block, context)
        })?;
        self.stats.target_tokens += pending.submitted_tokens();
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
        let assumed_len = self.runtime.sequence().tokens().len() + block.proposals.len();
        let count = self
            .config
            .proposal_capacity()
            .min(executor.max_proposals())
            .min(self.config.max_tokens.saturating_sub(assumed_len));
        let mut state = block.state.clone();
        let last = block
            .proposals
            .last()
            .expect("optimistic block has an assumed token")
            .token;
        let mut history = Vec::with_capacity(assumed_len);
        history.extend_from_slice(self.runtime.sequence().tokens());
        history.extend(block.proposals.iter().map(|proposal| proposal.token));
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
        self.stats.optimistic_draft_tokens += proposals.len();
        self.stats.optimistic_draft_blocks += 1;
        self.stats.optimistic_draft_time += started.elapsed();
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
                    &mut self.cache,
                    pending,
                    &mut self.runtime,
                    self.stats.clone(),
                    options
                        .completion_wait()
                        .map_err(SpeculativeDriverError::Generation)?,
                    context,
                )
            })?;
            telemetry.record(&mut stats);
            self.stats = stats;
            self.transition(SpeculativeRequestStatus::Cancelled)?;
            self.stats.elapsed = self.started.elapsed();
            return Ok(());
        }
        let mut published = with_activation_origin(executor, origin, |executor| {
            resolve_commit_and_publish(
                executor,
                &mut self.cache,
                pending,
                &mut self.runtime,
                self.target_randomness.as_ref(),
                self.config.temperature,
                self.stats.clone(),
                options,
                context,
            )
        })?;
        published.telemetry.record(&mut published.stats);
        self.target_state = Some(published.target_state);
        self.pending_committed_tokens = published.pending_committed_tokens;
        self.target_randomness = published.target_randomness;
        self.stats = published.stats;
        match published.status {
            SpeculativePublicationStatus::Continue(continuation) => {
                self.block = continuation.into_block();
                self.transition(SpeculativeRequestStatus::ReadyToDraft)?;
            }
            SpeculativePublicationStatus::Completed => {
                self.transition(SpeculativeRequestStatus::Completed)?;
                self.stats.elapsed = self.started.elapsed();
            }
            SpeculativePublicationStatus::Cancelled => {
                self.transition(SpeculativeRequestStatus::Cancelled)?;
                self.stats.elapsed = self.started.elapsed();
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
    token_ids: Vec<u32>,
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
    token_ids: Vec<u32>,
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
    pub fn take_token_ids(&mut self) -> Vec<u32> {
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
    requests: Vec<CompletedSpeculativeRequest<S>>,
    /// Aggregate scheduler telemetry.
    scheduler: SpeculativeSchedulerStats,
}

impl<S> CompletedSpeculativeSchedule<S> {
    /// Consumes the schedule into request results.
    pub fn into_requests(self) -> Vec<CompletedSpeculativeRequest<S>> {
        self.requests
    }
    /// Takes completed requests while retaining access to scheduler telemetry.
    pub fn take_requests(&mut self) -> Vec<CompletedSpeculativeRequest<S>> {
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

/// Canonical table and action coordinator for causal and speculative requests.
pub struct GenerationRequestTable<'cache, E, S, C, P>
where
    E: SpeculativeExecutor,
    S: SpeculativeSampling<Logits = E::Logits, Error = E::Error>,
    C: SpeculativeConstraint,
    P: SpeculativePublisher<C>,
{
    schedule: SpeculativeSchedule,
    speculation_limits: Option<crate::decoding::SpeculationLimits>,
    requests: Vec<SpeculativeRequest<'cache, E, S, C, P>>,
    stats: SpeculativeSchedulerStats,
}

impl<E, S, C, P> Drop for GenerationRequestTable<'_, E, S, C, P>
where
    E: SpeculativeExecutor,
    S: SpeculativeSampling<Logits = E::Logits, Error = E::Error>,
    C: SpeculativeConstraint,
    P: SpeculativePublisher<C>,
{
    fn drop(&mut self) {
        for request in &self.requests {
            request.decoding.finish();
        }
    }
}

impl<'cache, E, S, C, P> GenerationRequestTable<'cache, E, S, C, P>
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
            speculation_limits: None,
            requests: Vec::new(),
            stats: SpeculativeSchedulerStats {
                execution_topology: topology,
                ..SpeculativeSchedulerStats::default()
            },
        })
    }

    /// Retains the selected source policy before registering any lanes.
    pub fn with_speculation_limits(
        mut self,
        limits: Option<crate::decoding::SpeculationLimits>,
    ) -> Self {
        assert!(
            self.requests.is_empty(),
            "speculation limits are set before submitting requests"
        );
        self.speculation_limits = limits;
        self
    }

    pub(crate) fn fork_completed_source(
        &self,
        id: SpeculativeRequestId,
        cache: E::Cache,
        state: E::TargetState,
        runtime: SpeculativeOutputRuntime<S, C, P>,
        decoding: crate::decoding::DecodingControl,
    ) -> Self {
        let parent = self.request(id).expect("retained source");
        let mut config = parent.config.clone();
        config.max_tokens = runtime.sequence.max_tokens();
        config.decoding_mode = decoding.state().active;
        let mut lifecycle = SpeculativeRequestLifecycle::new();
        lifecycle
            .transition(SpeculativeRequestStatus::ReadyToDraft)
            .expect("prepared source becomes ready");
        if runtime.sequence.is_finished() || decoding.state().terminal {
            lifecycle = SpeculativeRequestLifecycle::completed();
        }
        Self {
            schedule: SpeculativeSchedule::new(self.options()).expect("retained scheduler options"),
            speculation_limits: self.speculation_limits,
            stats: self.stats.clone(),
            requests: vec![SpeculativeRequest {
                control_identity: Arc::new(()),
                id: SpeculativeRequestId::new(0),
                cache: GenerationCache::Owned(cache),
                config,
                runtime,
                target_randomness: None,
                draft_randomness: None,
                stats: parent.stats.clone(),
                started: parent.started,
                target_state: Some(state),
                pending_committed_tokens: Vec::new(),
                next_logits: None,
                refinement: None,
                decoding,
                block: None,
                pending: None,
                lifecycle,
                completed_decision: None,
            }],
        }
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
        cache: impl Into<GenerationCache<'cache, E::Cache>>,
        input: E::Input,
        config: GenerationRequestConfig,
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
        let mut cache_slot = cache.into();
        let cache = &mut *cache_slot;
        let stage = crate::run_preparation::TextPreparationStage::Delivery;
        let mut decoding = None;
        let prepared = (|| {
            config
                .validate()
                .map_err(SpeculativeDriverError::Generation)?;
            if config.proposal_capacity() > 0 && executor.max_proposals() == 0 {
                return Err(SpeculativeDriverError::Generation(
                    GenerationError::NoBackendDraftCapacity,
                ));
            }
            let completion = config
                .scheduler_options(self.schedule.options())
                .completion_wait()
                .map_err(SpeculativeDriverError::Generation)?;
            if !E::Completion::supports_cancellation(completion.cancellation()) {
                return Err(SpeculativeDriverError::UnsupportedCompletionCancellation {
                    cancellation: completion.cancellation(),
                });
            }
            let mut capabilities = executor.decoding_capabilities();
            capabilities.speculation_limits = self.speculation_limits;
            if let Some(limits) = self.speculation_limits {
                capabilities.max_draft_tokens = Some(
                    capabilities
                        .max_draft_tokens
                        .map_or(limits.max_draft_tokens, |native| {
                            native.min(limits.max_draft_tokens)
                        }),
                );
            }
            decoding = Some(crate::decoding::DecodingRegistration::new(
                crate::decoding::DecodingControl::new(
                    capabilities,
                    config.decoding_mode,
                    runtime.sampler().control_requires_positive_temperature() == Some(true),
                )
                .map_err(|error| {
                    SpeculativeDriverError::Generation(GenerationError::DecodingMode(error))
                })?,
            ));
            if let Some(handle) = &runtime.control_handle {
                handle
                    .bind_decoding(
                        decoding
                            .as_ref()
                            .expect("registered owner")
                            .control()
                            .clone(),
                        Some(runtime.cancellation()),
                    )
                    .map_err(|error| {
                        SpeculativeDriverError::Generation(GenerationError::DecodingMode(error))
                    })?;
            }
            runtime.decoding = Some(
                decoding
                    .as_ref()
                    .expect("registered owner")
                    .control()
                    .clone(),
            );
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
        let decoding = decoding.expect("validated preparation retains its command owner");
        let id = SpeculativeRequestId::new(self.requests.len());
        let started = Instant::now();
        let mut stats = SpeculativeStats {
            execution_topology: self.stats.execution_topology,
            component_timings_collected,
            ..SpeculativeStats::default()
        };
        let (target_randomness, draft_randomness) = (randomness.target, randomness.draft);
        let (target_state, lifecycle) = if runtime.cancellation().is_cancelled() {
            let cancelled = runtime.cancel().map_err(SpeculativeDriverError::Output);
            crate::run_preparation::finish_preparation(
                stage,
                cancelled,
                |status| executor.agree_text_preparation(stage, status, context),
                SpeculativeDriverError::Preparation,
            )?;
            stats.elapsed = started.elapsed();
            (None, SpeculativeRequestLifecycle::cancelled())
        } else if runtime.sequence().is_finished() {
            stats.elapsed = started.elapsed();
            (None, SpeculativeRequestLifecycle::completed())
        } else {
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
            if executor.uses_completed_decisions() {
                let state = executor
                    .prepare_decision_source(input, cache, context)
                    .map_err(SpeculativeDriverError::Backend)?
                    .ok_or(SpeculativeDriverError::Coordination(
                        "completed decision source preparation",
                    ))?;
                self.requests.push(SpeculativeRequest {
                    completed_decision: None,
                    control_identity: Arc::new(()),
                    id,
                    cache: cache_slot,
                    config,
                    runtime,
                    target_randomness,
                    draft_randomness,
                    stats,
                    started,
                    target_state: Some(state),
                    pending_committed_tokens: Vec::new(),
                    next_logits: None,
                    refinement: None,
                    decoding: decoding.install(),
                    block: None,
                    pending: None,
                    lifecycle: {
                        let mut lifecycle = SpeculativeRequestLifecycle::new();
                        lifecycle
                            .transition(SpeculativeRequestStatus::ReadyToDraft)
                            .map_err(SpeculativeDriverError::Generation)?;
                        lifecycle
                    },
                });
                return Ok(id);
            }
            let checkpoint = executor
                .checkpoint(cache)
                .map_err(SpeculativeDriverError::Backend);
            let checkpoint = crate::run_preparation::finish_preparation(
                stage,
                checkpoint,
                |status| executor.agree_text_preparation(stage, status, context),
                SpeculativeDriverError::Preparation,
            )?;
            if matches!(
                config.decoding_mode,
                crate::decoding::DecodingMode::Diffusion { .. }
            ) {
                let origin = executor
                    .requires_activation_origin()
                    .then(|| SpeculativeActivationOrigin::new(id, &[], false));
                let prefilling = with_activation_origin(executor, origin, |executor| {
                    executor.prefill(input, cache, context)
                })
                .map_err(SpeculativeDriverError::Backend);
                let prefill = match coordination::ready_result(executor, prefilling, context) {
                    Ok(prefill) => prefill,
                    Err(error) => {
                        executor.restore_checkpoint(cache, &checkpoint, context)?;
                        return Err(error);
                    }
                };
                stats.target_tokens = prefill.evaluated_tokens;
                let mut lifecycle = SpeculativeRequestLifecycle::new();
                lifecycle
                    .transition(SpeculativeRequestStatus::ReadyToDraft)
                    .map_err(SpeculativeDriverError::Generation)?;
                self.requests.push(SpeculativeRequest {
                    completed_decision: None,
                    control_identity: Arc::new(()),
                    id,
                    cache: cache_slot,
                    config,
                    runtime,
                    target_randomness,
                    draft_randomness,
                    stats,
                    started,
                    target_state: Some(prefill.state),
                    pending_committed_tokens: Vec::new(),
                    next_logits: Some(prefill.logits),
                    refinement: None,
                    decoding: decoding.install(),
                    block: None,
                    pending: None,
                    lifecycle,
                });
                return Ok(id);
            }
            let attempt = (|| {
                let computed = (|| {
                    let origin = executor
                        .requires_activation_origin()
                        .then(|| SpeculativeActivationOrigin::new(id, &[], false));
                    let prefill = with_activation_origin(executor, origin, |executor| {
                        executor.prefill(input, cache, context)
                    })?;
                    let mut sampler = runtime.sampler().clone();
                    let mut constraint = runtime
                        .constraint()
                        .fork()
                        .map_err(SpeculativeDriverError::Output)?;
                    let mut sequence = runtime.sequence().clone();
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
                    let reason =
                        commit_terminal_token(&mut sequence, &mut sampler, &mut constraint, first)?;
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
                if reason.is_some() {
                    decoding.control().finish();
                }
                let cancelled = coordination::publish_candidate(
                    executor,
                    &mut runtime,
                    &mut constraint,
                    &mut sequence,
                    &[first],
                    context,
                )?;
                runtime.install_committed_state(sampler, constraint, sequence);
                Ok::<_, SpeculativeDriverError<E::Error>>((
                    prefill.evaluated_tokens,
                    prefill.state,
                    target_randomness,
                    reason,
                    cancelled,
                    submission_to_first_token,
                ))
            })();
            let (
                evaluated_tokens,
                target_state,
                target_randomness,
                reason,
                cancelled,
                submission_to_first_token,
            ) = match attempt {
                Ok(result) => result,
                Err(error) => {
                    executor.restore_checkpoint(cache, &checkpoint, context)?;
                    return Err(error);
                }
            };
            stats.submission_to_first_token = Some(submission_to_first_token);
            stats.target_tokens = evaluated_tokens;
            stats.scheduler_turns = 1;
            stats.emitted_tokens = 1;
            let lifecycle = if cancelled {
                stats.elapsed = started.elapsed();
                SpeculativeRequestLifecycle::cancelled()
            } else if reason.is_some() {
                stats.elapsed = started.elapsed();
                SpeculativeRequestLifecycle::completed()
            } else {
                let mut lifecycle = SpeculativeRequestLifecycle::new();
                lifecycle
                    .transition(SpeculativeRequestStatus::ReadyToDraft)
                    .map_err(SpeculativeDriverError::Generation)?;
                lifecycle
            };
            self.stats.turns += 1;
            if lifecycle.is_terminal() {
                decoding.control().finish();
            }
            self.requests.push(SpeculativeRequest {
                completed_decision: None,
                control_identity: Arc::new(()),
                id,
                cache: cache_slot,
                config,
                pending_committed_tokens: runtime
                    .sequence()
                    .tokens()
                    .last()
                    .copied()
                    .into_iter()
                    .collect(),
                next_logits: None,
                refinement: None,
                decoding: decoding.install(),
                runtime,
                target_randomness,
                draft_randomness,
                stats,
                started,
                target_state: Some(target_state),
                block: None,
                pending: None,
                lifecycle,
            });
            return Ok(id);
        };
        if lifecycle.is_terminal() {
            decoding.control().finish();
        }
        self.requests.push(SpeculativeRequest {
            completed_decision: None,
            control_identity: Arc::new(()),
            id,
            cache: cache_slot,
            config,
            pending_committed_tokens: Vec::new(),
            next_logits: None,
            refinement: None,
            decoding: decoding.install(),
            runtime,
            target_randomness,
            draft_randomness,
            stats,
            started,
            target_state,
            block: None,
            pending: None,
            lifecycle,
        });
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

    pub(crate) fn advance_completed_context<'context>(
        &mut self,
        executor: &mut E,
        id: SpeculativeRequestId,
        context: E::Context<'context>,
    ) -> Result<bool, SpeculativeDriverError<E::Error>>
    where
        E: 'context,
    {
        let request = self.request_mut(id).expect("retained completed source");
        if request.lifecycle.is_terminal() {
            return Ok(false);
        }
        let cancelled = request
            .decoding
            .coordinate(
                request.runtime.cancellation().is_cancelled(),
                request.sequence().tokens().len() as u64,
                executor
                    .completed_decision_allowance(&request.cache)
                    .flatten(),
                |local| executor.coordinate_decoding_control(local, context),
            )
            .map_err(|error| match error {
                crate::decoding::DecodingControlError::Agreement(error) => {
                    SpeculativeDriverError::Preparation(error)
                }
                crate::decoding::DecodingControlError::Mode(error) => {
                    SpeculativeDriverError::Generation(GenerationError::DecodingMode(error))
                }
            })?;
        if cancelled {
            request.runtime.cancellation().cancel();
            request.request_cancellation()?;
            return Ok(false);
        }
        request
            .runtime
            .observe_lifecycle(SpeculativeLifecycleStage::Execution)
            .map_err(SpeculativeDriverError::Output)?;
        let progress = executor
            .advance_decision_context(&mut request.cache, context)
            .map_err(SpeculativeDriverError::Backend)?;
        if progress.is_some() {
            request.stats.scheduler_turns += 1;
            self.stats.turns += 1;
        }
        Ok(progress.is_some())
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
        for request in &mut self.requests {
            if request.lifecycle.is_terminal() {
                continue;
            }
            let cancelled = request
                .decoding
                .coordinate(
                    request.runtime.cancellation().is_cancelled()
                        || request.lifecycle.cancellation_pending(),
                    request.sequence().tokens().len() as u64,
                    executor
                        .completed_decision_allowance(&request.cache)
                        .unwrap_or_else(|| {
                            Some(
                                request
                                    .config
                                    .max_tokens
                                    .saturating_sub(request.sequence().tokens().len()),
                            )
                        }),
                    |local| executor.coordinate_decoding_control(local, context),
                )
                .map_err(|error| match error {
                    crate::decoding::DecodingControlError::Agreement(error) => {
                        SpeculativeDriverError::Preparation(error)
                    }
                    crate::decoding::DecodingControlError::Mode(error) => {
                        SpeculativeDriverError::Generation(GenerationError::DecodingMode(error))
                    }
                })?;
            if cancelled {
                request.runtime.cancellation().cancel();
                request.request_cancellation()?;
            }
            let changed = request.settle_requested_mode(executor, self.schedule.options(), context);
            match changed {
                Ok(true) => {
                    self.stats.turns += 1;
                    request.stats.scheduler_turns += 1;
                    return Ok(true);
                }
                Ok(false) => {}
                Err(error) => {
                    request.decoding.finish();
                    return Err(error);
                }
            }
        }
        let local = self
            .requests
            .iter()
            .map(|request| {
                let candidate = request.candidate(
                    executor,
                    optimistic_execution_available,
                    request
                        .config
                        .scheduler_options(self.schedule.options())
                        .completion_wait()
                        .expect("speculative schedule retains validated completion options"),
                )?;
                Ok(SpeculativeScheduleState {
                    request: request.id,
                    status: candidate.status,
                    cancellation_requested: request.runtime.cancellation().is_cancelled()
                        || request.lifecycle.cancellation_pending(),
                    verification_complete: candidate.verification_complete,
                    verification_deadline_expired: candidate.verification_deadline_expired,
                    optimistic_eligible: candidate.optimistic_eligible,
                })
            })
            .collect::<Result<Vec<_>, SpeculativeDriverError<E::Error>>>();
        let local = crate::run_preparation::finish_preparation(
            stage,
            local,
            |status| executor.agree_text_preparation(stage, status, context),
            SpeculativeDriverError::Preparation,
        )?;
        let coordinated = executor
            .coordinate_speculative_step(local, context)
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
        let candidates = coordinated
            .into_iter()
            .zip(&self.requests)
            .map(|(state, request)| SpeculativeCandidate {
                status: request.lifecycle.status(),
                optimistic_eligible: state.optimistic_eligible && !state.cancellation_requested,
                verification_complete: state.verification_complete,
                verification_deadline_expired: state.verification_deadline_expired,
            })
            .collect::<Vec<_>>();
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
        self.stats.turns += 1;
        self.requests[index].stats.scheduler_turns += 1;
        let completed = (|| {
            match action {
                SpeculativeAction::SubmitVerification(index) => {
                    self.requests[index].submit_verification(executor, context)?;
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
                    let drafted = self.requests[index].draft_committed(
                        executor,
                        self.schedule.options(),
                        context,
                    )?;
                    if cross_request && drafted {
                        self.requests[index].stats.cross_request_draft_opportunities += 1;
                        self.stats.cross_request_draft_opportunities += 1;
                    }
                }
                SpeculativeAction::DraftOptimistic(index) => {
                    self.requests[index].draft_optimistic(executor, context)?;
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
                    let options = self.requests[index]
                        .config
                        .scheduler_options(self.schedule.options());
                    self.requests[index].resolve_verification(executor, options, context)?;
                }
            }
            Ok(true)
        })();
        let result = crate::run_preparation::finish_preparation(
            stage,
            completed,
            |status| executor.agree_text_preparation(stage, status, context),
            SpeculativeDriverError::Preparation,
        );
        if result.is_err() {
            self.requests[index].decoding.finish();
        }
        result
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
        mut self,
    ) -> Result<CompletedSpeculativeSchedule<S>, SpeculativeDriverError<E::Error>> {
        if !self.is_finished() {
            return Err(SpeculativeDriverError::Generation(
                GenerationError::ActiveSpeculativeRequests,
            ));
        }
        Ok(CompletedSpeculativeSchedule {
            requests: std::mem::take(&mut self.requests)
                .into_iter()
                .map(|request| {
                    let (sampler, sequence, _, _) = request.runtime.into_parts();
                    CompletedSpeculativeRequest {
                        id: request.id,
                        finish_reason: sequence.finish_reason(),
                        token_ids: sequence.into_tokens(),
                        stats: request.stats,
                        sampler,
                        status: request.lifecycle.status(),
                    }
                })
                .collect(),
            scheduler: std::mem::take(&mut self.stats),
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
            atomic::{AtomicUsize, Ordering},
            Arc, Mutex,
        },
    };

    type TransactionTrace = Rc<RefCell<Vec<&'static str>>>;

    #[test]
    fn speculative_telemetry_preserves_exact_statistics_and_durations() {
        let stats = SpeculativeStats {
            execution_topology: SpeculativeExecutionTopology::CrossDeviceSplit,
            target_tokens: 31,
            draft_tokens: 8,
            accepted_tokens: 5,
            rounds: 2,
            accept_lens: vec![2, 3],
            emitted_tokens: 7,
            optimistic_draft_tokens: 9,
            reused_optimistic_tokens: 4,
            discarded_optimistic_tokens: 5,
            adaptive_lookahead_disabled: true,
            optimistic_draft_time: Duration::from_millis(125),
            verification_in_flight_time: Duration::from_millis(375),
            ..SpeculativeStats::default()
        };
        assert_eq!(
            crate::speculative_decoding_telemetry(&stats),
            crate::SpeculativeDecodingTelemetry {
                execution_topology: "cross-device-split".into(),
                target_tokens: 31,
                refinement_tokens: 0,
                refinement_steps: 0,
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
        assert!(!SpeculativeCapability::Unavailable.admits_source(SpeculativeDraftSource::Embedded));
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

    impl SpeculativeSemanticState for PortableSemanticState {
        fn fork_box(&self) -> Result<Box<dyn SpeculativeSemanticState>, SpeculativeOutputError> {
            let mut fork = self.clone();
            fork.events.clear();
            Ok(Box::new(fork))
        }

        fn push_token(&mut self, token: u32) -> Result<bool, SpeculativeOutputError> {
            self.events
                .push(crate::generation::SemanticEvent::TextDelta(
                    token.to_string(),
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

        fn take_events(&mut self) -> Vec<crate::generation::SemanticEvent> {
            std::mem::take(&mut self.events)
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
            assert!(!publisher
                .publish_committed(
                    &mut constraint,
                    &[7],
                    &GenerationCancellationToken::new(),
                    true,
                )
                .unwrap());
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
        trace: Option<TransactionTrace>,
        full_acceptance: bool,
        masked_draft: bool,
        capture_origins: bool,
        origin: Option<SpeculativeActivationOrigin>,
        activations: Vec<(&'static str, Option<SpeculativeActivationOrigin>)>,
        draft_capacity: Option<usize>,
        refinement: Option<Vec<u32>>,
        refined_prefixes: Vec<Vec<u32>>,
    }

    struct MockGenerationTypes;
    impl GenerationExecutorTypes for MockGenerationTypes {
        type Input = Vec<u32>;
        type Logits = Vec<f32>;
        type Context<'a> = ();
        type Completion = Done;
        type Telemetry = ();
        type Error = Infallible;
        fn erased_type_mismatch(value: &'static str) -> Infallible {
            panic!("mismatched fixture value: {value}")
        }
    }
    impl GenerationExecutorCacheFactory<MockGenerationTypes> for MockExecutor {
        fn new_cache<'a>(&mut self, _: ()) -> Result<Vec<u32>, Infallible>
        where
            Self: 'a,
        {
            Ok(Vec::new())
        }
        fn bind_context<'a>(
            _: <MockGenerationTypes as GenerationExecutorTypes>::Context<'a>,
        ) -> Self::Context<'a>
        where
            Self: 'a,
        {
        }
    }

    struct MockVerification {
        tokens: Vec<u32>,
        logits: Vec<Vec<f32>>,
    }

    impl SpeculativeExecutor for MockExecutor {
        fn decoding_capabilities(&self) -> crate::decoding::DecodingCapabilities {
            crate::decoding::DecodingCapabilities {
                autoregressive: true,
                speculation: self.max_proposals() != 0,
                mode_changes: true,
                diffusion: self.refinement.is_some(),
                masked_unmasking: self.refinement.is_some(),
                mask_token_id: self.refinement.as_ref().map(|_| 0),
                ..Default::default()
            }
        }
        fn refine_block<'a>(
            &mut self,
            _: &mut usize,
            _: &Vec<f32>,
            committed: &[u32],
            _: Option<crate::refinement::RefinementPolicy>,
            _: f32,
            _: (),
        ) -> Result<Option<RefinementProgress<Vec<f32>>>, Infallible>
        where
            Self: 'a,
        {
            self.refined_prefixes.push(committed.to_vec());
            Ok(self.refinement.as_ref().map(|tokens| {
                RefinementProgress::Complete(RefinementBlock {
                    positions: tokens
                        .iter()
                        .enumerate()
                        .map(|(position, token)| {
                            (
                                committed.get(position).copied().unwrap_or(*token),
                                vec![10.0, 1.0, 2.0],
                            )
                        })
                        .collect(),
                    evaluated_positions: tokens.len() * 2,
                    iterations: 2,
                })
            }))
        }
        type Input = Vec<u32>;
        type Cache = Vec<u32>;
        type TargetState = usize;
        type DraftState = Vec<u32>;
        type CacheCheckpoint = usize;
        type Verification = MockVerification;
        type Logits = Vec<f32>;

        fn max_proposals(&self) -> usize {
            self.draft_capacity.unwrap_or(usize::MAX)
        }
        type Context<'a>
            = ()
        where
            Self: 'a;
        type Completion = Done;
        type Telemetry = ();
        type Error = Infallible;

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
        ) -> Result<SpeculativePrefill<Self::TargetState, Self::Logits>, Self::Error>
        where
            Self: 'context,
        {
            self.activations.push(("prefill", self.origin));
            cache.extend_from_slice(&input);
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
        ) -> Result<Self::DraftState, Self::Error>
        where
            Self: 'a,
        {
            self.activations.push(("begin", self.origin));
            Ok(vec![last_token])
        }

        fn proposal_step<'a>(
            &mut self,
            state: &mut Self::DraftState,
            last_token: u32,
            _: Self::Context<'a>,
        ) -> Result<SpeculativeProposalStep<Self::Logits>, Self::Error>
        where
            Self: 'a,
        {
            self.activations.push(("proposal", self.origin));
            state.push(last_token + 1);
            Ok(if self.masked_draft {
                SpeculativeProposalStep::TargetMatchingToken(1)
            } else {
                SpeculativeProposalStep::CausalLogits(vec![0.0, 1.0])
            })
        }

        fn checkpoint(&self, cache: &Self::Cache) -> Result<Self::CacheCheckpoint, Self::Error> {
            Ok(cache.len())
        }

        fn restore_checkpoint<'a>(
            &mut self,
            cache: &mut Self::Cache,
            checkpoint: &Self::CacheCheckpoint,
            _: Self::Context<'a>,
        ) -> Result<(), Self::Error>
        where
            Self: 'a,
        {
            cache.truncate(*checkpoint);
            Ok(())
        }

        fn submit_verification<'a>(
            &mut self,
            input_tokens: &[u32],
            cache: &mut Self::Cache,
            _: Self::Context<'a>,
        ) -> Result<Submission<Self::Verification, Self::Completion>, Self::Error>
        where
            Self: 'a,
        {
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
        ) -> Result<Self::Logits, Self::Error>
        where
            Self: 'a,
        {
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
        ) -> Result<SpeculativeCommit<Self::TargetState>, Self::Error>
        where
            Self: 'a,
        {
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

    #[derive(Clone)]
    struct MockSampling {
        committed: Vec<u32>,
        trace: Option<TransactionTrace>,
        unit_draw: f32,
        categorical: bool,
        draft_prefix_limit: Option<usize>,
    }

    impl Default for MockSampling {
        fn default() -> Self {
            Self {
                committed: Vec::new(),
                trace: None,
                unit_draw: 0.5,
                categorical: false,
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
        fn finalize_refinement_position<'a>(
            &mut self,
            scores: &Vec<f32>,
            selected: u32,
            mask: Option<u32>,
            _: &[u32],
            _: (),
        ) -> Result<Option<(u32, Vec<f32>)>, Infallible>
        where
            Self: 'a,
        {
            let mut scores = scores.clone();
            if let Some(mask) = mask {
                scores[mask as usize] = f32::NEG_INFINITY;
            }
            let token = if mask.is_some() && scores[selected as usize].is_finite() {
                selected
            } else {
                argmax(&scores)
            };
            Ok(Some((token, scores)))
        }
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
            let mut choice = argmax(distribution);
            if let Some(randomness) = randomness {
                if self.categorical {
                    // An enumerable set of 100 equally likely target draws.
                    let draw = (*randomness % 100) as f64 / 100.0 + 0.005;
                    let total: f64 = distribution.iter().map(|x| f64::from(*x).exp()).sum();
                    let mut cumulative = 0.0;
                    for (token, score) in distribution.iter().enumerate() {
                        cumulative += f64::from(*score).exp() / total;
                        if draw < cumulative {
                            choice = token as u32;
                            break;
                        }
                    }
                }
                *randomness += 1;
            }
            Ok(choice)
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
    fn request_table_runs_causal_steps_without_a_drafter() {
        let mut executor = MockExecutor {
            draft_capacity: Some(0),
            ..Default::default()
        };
        let mut cache = Vec::new();
        let mut table = GenerationRequestTable::new(
            SpeculativeSchedulerOptions::default(),
            SpeculativeExecutionTopology::Single,
        )
        .unwrap();
        let id = table
            .submit(
                &mut executor,
                &mut cache,
                vec![4],
                GenerationRequestConfig {
                    max_tokens: 4,
                    ..Default::default()
                },
                empty_mock_runtime(4, GenerationCancellationToken::new()),
                SpeculativeRandomness {
                    target: None,
                    draft: None,
                },
                false,
                (),
            )
            .unwrap();
        table.run(&mut executor, true, ()).unwrap();
        let request = table.request(id).unwrap();
        assert_eq!(request.sequence().tokens(), [1, 1, 1, 1]);
        assert_eq!(request.pending_committed_tokens(), &[1]);
        assert_eq!(request.runtime.publisher.tokens, [1, 1, 1, 1]);
        assert_eq!(request.stats().draft_tokens, 0);
        assert_eq!(request.stats().optimistic_draft_tokens, 0);
        assert_eq!(request.status(), SpeculativeRequestStatus::Completed);
        table.finish().unwrap();
        assert_eq!(
            cache,
            [4, 1, 1, 1],
            "each prior committed token enters the causal cache once"
        );
        assert_eq!(
            executor
                .activations
                .iter()
                .filter(|(phase, _)| *phase == "verify")
                .count(),
            3
        );
        assert!(!executor
            .activations
            .iter()
            .any(|(phase, _)| *phase == "proposal"));
    }

    #[test]
    fn suffix_settlement_retains_next_scores_without_sampling_or_publication() {
        let trace = TransactionTrace::default();
        let mut executor = MockExecutor {
            trace: Some(trace.clone()),
            ..Default::default()
        };
        let mut cache = vec![4, 5];
        let settled = settle_committed_suffix(
            &mut executor,
            &mut cache,
            &2,
            &[1],
            SpeculativeSchedulerOptions::default()
                .completion_wait()
                .unwrap(),
            |_| Ok(()),
            (),
        )
        .unwrap();
        assert_eq!(cache, [4, 5, 1]);
        assert_eq!(settled.logits, [0.0, 1.0]);
        assert_eq!(settled.replayed_tokens, 0);
        assert_eq!(*trace.borrow(), ["wait", "commit"]);
        assert!(!executor
            .activations
            .iter()
            .any(|(phase, _)| *phase == "proposal"));
    }

    #[test]
    fn failed_prefill_fences_a_bound_mode_handle_before_any_execution() {
        let handle = crate::execution_control::GenerationControlHandle::default();
        let runtime = empty_mock_runtime(3, GenerationCancellationToken::new())
            .with_control_handle(Some(handle.clone()))
            .with_lifecycle_observer(Arc::new(LifecycleTrace::failing(
                SpeculativeLifecycleStage::Execution,
            )));
        let mut executor = MockExecutor::default();
        let mut cache = Vec::new();
        let mut table = GenerationRequestTable::new(
            SpeculativeSchedulerOptions::default(),
            SpeculativeExecutionTopology::Single,
        )
        .unwrap();
        assert!(matches!(
            table.submit(
                &mut executor,
                &mut cache,
                vec![4],
                GenerationRequestConfig::default(),
                runtime,
                SpeculativeRandomness {
                    target: None,
                    draft: None
                },
                false,
                (),
            ),
            Err(SpeculativeDriverError::Output(_))
        ));
        assert!(executor.activations.is_empty());
        assert!(handle.decoding_state().unwrap().terminal);
        assert_eq!(
            handle.request_decoding_mode(crate::decoding::DecodingMode::Autoregressive),
            Err(crate::decoding::DecodingModeError::Terminal)
        );
    }

    #[test]
    fn terminal_causal_output_closes_commands_before_its_delivery_callback() {
        for budget in [1, 3] {
            let handle = crate::execution_control::GenerationControlHandle::default();
            let mut observed = Vec::new();
            let mut executor = MockExecutor::default();
            let mut cache = Vec::new();
            let runtime = SpeculativeOutputRuntime::new(
                MockSampling::default(),
                GenerationSequence::new(budget, []),
                SpeculativeSemanticConstraint::plain(),
                SpeculativeCallbackPublisher::tokens(|tokens| {
                    observed.push((tokens.to_vec(), handle.decoding_state().unwrap().terminal));
                    Ok(())
                }),
                GenerationCancellationToken::new(),
            )
            .with_control_handle(Some(handle.clone()));
            let mut table = GenerationRequestTable::new(
                SpeculativeSchedulerOptions::default(),
                SpeculativeExecutionTopology::Single,
            )
            .unwrap();
            table
                .submit(
                    &mut executor,
                    &mut cache,
                    vec![4],
                    GenerationRequestConfig {
                        max_tokens: budget,
                        ..Default::default()
                    },
                    runtime,
                    SpeculativeRandomness {
                        target: None,
                        draft: None,
                    },
                    false,
                    (),
                )
                .unwrap();
            table.run(&mut executor, true, ()).unwrap();
            table.finish().unwrap();
            assert_eq!(observed.len(), budget);
            assert!(observed.last().unwrap().1);
            assert!(observed[..budget - 1].iter().all(|(_, terminal)| !terminal));
        }
    }

    #[test]
    fn request_table_mode_handoffs_settle_suffix_and_preserve_the_retained_target_row() {
        use crate::decoding::{DecodingMode, DecodingModeError, SpeculationOptions};
        let speculation = DecodingMode::SelfSpeculation(
            SpeculationOptions::default()
                .with_proposal_capacity(std::num::NonZeroUsize::new(2).unwrap()),
        );
        let mut executor = MockExecutor {
            full_acceptance: true,
            ..Default::default()
        };
        let mut cache = Vec::new();
        let mut table = GenerationRequestTable::new(
            SpeculativeSchedulerOptions::default(),
            SpeculativeExecutionTopology::Single,
        )
        .unwrap();
        let id = table
            .submit(
                &mut executor,
                &mut cache,
                vec![4],
                GenerationRequestConfig {
                    max_tokens: 9,
                    decoding_mode: speculation,
                    ..Default::default()
                },
                empty_mock_runtime(9, GenerationCancellationToken::new()),
                SpeculativeRandomness {
                    target: None,
                    draft: None,
                },
                false,
                (),
            )
            .unwrap();
        table.step(&mut executor, true, ()).unwrap();
        table.step(&mut executor, true, ()).unwrap();
        let request = table.request(id).unwrap();
        assert!(request.has_pending_verification());
        assert!(matches!(
            request.request_decoding_mode(DecodingMode::Diffusion { policy: None }),
            Err(DecodingModeError::Unsupported(_))
        ));
        request
            .request_decoding_mode(DecodingMode::Autoregressive)
            .unwrap();
        assert_eq!(
            request.request_decoding_mode(speculation),
            Err(DecodingModeError::ModeChangePending)
        );
        table.step(&mut executor, true, ()).unwrap();
        assert_eq!(
            table.request(id).unwrap().decoding_state().active,
            speculation
        );
        assert_eq!(table.request(id).unwrap().sequence().tokens().len(), 4);
        table.step(&mut executor, true, ()).unwrap();
        let request = table.request(id).unwrap();
        assert_eq!(
            request
                .decoding_state()
                .acknowledgement
                .unwrap()
                .committed_position,
            4
        );
        assert_eq!(
            request.decoding_state().active,
            DecodingMode::Autoregressive
        );
        assert!(request.pending_committed_tokens().is_empty());
        assert_eq!(*request.cache, [4, 1, 1, 1, 1]);
        let evaluations = request.stats().target_tokens;
        table.step(&mut executor, true, ()).unwrap();
        let request = table.request(id).unwrap();
        assert_eq!(request.sequence().tokens().len(), 5);
        assert_eq!(
            request.stats().target_tokens,
            evaluations,
            "retained target scores require no duplicate append"
        );
        assert_eq!(*request.cache, [4, 1, 1, 1, 1]);
        request.request_decoding_mode(speculation).unwrap();
        table.step(&mut executor, true, ()).unwrap();
        assert_eq!(
            table
                .request(id)
                .unwrap()
                .decoding_state()
                .acknowledgement
                .unwrap()
                .committed_position,
            5
        );
        table.run(&mut executor, true, ()).unwrap();
        let request = table.request(id).unwrap();
        assert_eq!(request.sequence().tokens(), [1; 9]);
        assert_eq!(*request.cache, [4, 1, 1, 1, 1, 1, 1, 1, 1]);
        assert_eq!(
            request.request_decoding_mode(DecodingMode::Autoregressive),
            Err(DecodingModeError::Terminal)
        );
    }

    #[test]
    fn request_mode_disables_lookahead_within_a_permissive_scheduler() {
        let options = crate::decoding::SpeculationOptions::try_from(
            crate::decoding::SpeculationOptionsRequest {
                scheduler: SpeculativeSchedulerOptions::default().with_lookahead(false),
                ..Default::default()
            },
        )
        .unwrap();
        let mut executor = MockExecutor {
            full_acceptance: true,
            ..Default::default()
        };
        let mut cache = Vec::new();
        let mut table = GenerationRequestTable::new(
            SpeculativeSchedulerOptions::default(),
            SpeculativeExecutionTopology::Single,
        )
        .unwrap();
        let id = table
            .submit(
                &mut executor,
                &mut cache,
                vec![4],
                GenerationRequestConfig {
                    max_tokens: 7,
                    decoding_mode: crate::decoding::DecodingMode::SelfSpeculation(
                        options.with_proposal_capacity(std::num::NonZeroUsize::new(2).unwrap()),
                    ),
                    ..Default::default()
                },
                empty_mock_runtime(7, GenerationCancellationToken::new()),
                SpeculativeRandomness {
                    target: None,
                    draft: None,
                },
                false,
                (),
            )
            .unwrap();
        table.run(&mut executor, true, ()).unwrap();
        let request = table.request(id).unwrap();
        assert_eq!(request.sequence().tokens(), [1, 1, 1, 1, 1, 1, 1]);
        assert!(request.stats().draft_tokens > 0);
        assert_eq!(request.stats().optimistic_draft_tokens, 0);
        assert_eq!(request.stats().reused_optimistic_tokens, 0);
    }

    #[test]
    fn diffusion_table_preserves_selections_refines_replacements_and_settles_the_suffix() {
        let mut executor = MockExecutor {
            refinement: Some(vec![1, 0, 1]),
            ..Default::default()
        };
        let mut table =
            GenerationRequestTable::new(Default::default(), SpeculativeExecutionTopology::Single)
                .unwrap();
        let id = table
            .submit(
                &mut executor,
                GenerationCache::Owned(Vec::new()),
                vec![4],
                GenerationRequestConfig {
                    max_tokens: 8,
                    decoding_mode: crate::decoding::DecodingMode::Diffusion { policy: None },
                    ..Default::default()
                },
                empty_mock_runtime(8, GenerationCancellationToken::new()),
                SpeculativeRandomness::new(Some(19), Some(23)),
                false,
                (),
            )
            .unwrap();
        assert!(table.request(id).unwrap().sequence().tokens().is_empty());
        while table.request(id).unwrap().sequence().tokens().len() < 3 {
            table.step(&mut executor, false, ()).unwrap();
        }
        let request = table.request(id).unwrap();
        // Selected one survives a higher-scoring two. The unresolved mask is
        // replaced by two; its dependent suffix is numerically refined again.
        assert_eq!(request.sequence().tokens(), [1, 2, 1]);
        assert_eq!(executor.refined_prefixes, [vec![], vec![1, 2]]);
        assert_eq!(&**request.cache, &[4]);
        assert_eq!(request.target_randomness, Some(19));
        assert_eq!(request.stats().refinement_tokens(), 12);
        assert_eq!(request.stats().refinement_steps(), 4);
        request
            .request_decoding_mode(crate::decoding::DecodingMode::Autoregressive)
            .unwrap();
        table.step(&mut executor, false, ()).unwrap();
        let request = table.request(id).unwrap();
        assert_eq!(&**request.cache, &[4, 1, 2, 1]);
        assert!(request.pending_committed_tokens().is_empty());
        assert_eq!(
            request
                .decoding_state()
                .acknowledgement
                .unwrap()
                .committed_position,
            3
        );
    }

    #[test]
    fn erased_generation_drives_diffusion_and_causal_handoff_with_owned_state() {
        let mut source = MockExecutor {
            refinement: Some(vec![1, 0, 1]),
            ..Default::default()
        };
        {
            let mut executor = DynGenerationExecutor::<MockGenerationTypes>::new(&mut source);
            let cache = executor.new_cache(()).unwrap();
            let mut table = GenerationRequestTable::new(
                Default::default(),
                SpeculativeExecutionTopology::Single,
            )
            .unwrap();
            let id = table
                .submit(
                    &mut executor,
                    GenerationCache::Owned(cache),
                    vec![4],
                    GenerationRequestConfig {
                        max_tokens: 8,
                        decoding_mode: crate::decoding::DecodingMode::Diffusion { policy: None },
                        ..Default::default()
                    },
                    empty_mock_runtime(8, GenerationCancellationToken::new()),
                    SpeculativeRandomness::new(Some(19), Some(23)),
                    false,
                    (),
                )
                .unwrap();
            while table.request(id).unwrap().sequence().tokens().len() < 3 {
                table.step(&mut executor, false, ()).unwrap();
            }
            assert_eq!(table.request(id).unwrap().sequence().tokens(), [1, 2, 1]);
            table
                .request(id)
                .unwrap()
                .request_decoding_mode(crate::decoding::DecodingMode::Autoregressive)
                .unwrap();
            table.step(&mut executor, false, ()).unwrap();
            assert_eq!(
                table
                    .request(id)
                    .unwrap()
                    .decoding_state()
                    .acknowledgement
                    .unwrap()
                    .committed_position,
                3
            );
            assert!(table
                .request(id)
                .unwrap()
                .pending_committed_tokens()
                .is_empty());
            table.step(&mut executor, false, ()).unwrap();
            assert_eq!(table.request(id).unwrap().sequence().tokens(), [1, 2, 1, 1]);
            assert!(table
                .request(id)
                .unwrap()
                .stats()
                .submission_to_first_token()
                .is_some());
        }
        assert_eq!(source.refined_prefixes, [vec![], vec![1, 2]]);
    }

    #[test]
    fn diffusion_table_keeps_full_geometry_for_budget_and_eos_inside_blocks() {
        for (limit, eos, expected) in [
            (1, vec![], vec![2]),
            (2, vec![], vec![2, 1]),
            (8, vec![1], vec![2, 1]),
        ] {
            let mut executor = MockExecutor {
                refinement: Some(vec![2, 1, 2]),
                ..Default::default()
            };
            let mut cache = Vec::new();
            let mut table = GenerationRequestTable::new(
                Default::default(),
                SpeculativeExecutionTopology::Single,
            )
            .unwrap();
            let output = SpeculativeOutputRuntime::new(
                MockSampling::default(),
                GenerationSequence::new(limit, eos.clone()),
                SpeculativeSemanticConstraint::plain(),
                SpeculativeCallbackPublisher::tokens(|_| Ok(())),
                Default::default(),
            );
            let id = table
                .submit(
                    &mut executor,
                    &mut cache,
                    vec![4],
                    GenerationRequestConfig {
                        max_tokens: limit,
                        eos_token_ids: eos,
                        decoding_mode: crate::decoding::DecodingMode::Diffusion { policy: None },
                        ..Default::default()
                    },
                    output,
                    SpeculativeRandomness::new(Some(19), Some(23)),
                    false,
                    (),
                )
                .unwrap();
            while !table.is_finished() {
                table.step(&mut executor, false, ()).unwrap();
            }
            let request = table.request(id).unwrap();
            assert_eq!(request.sequence().tokens(), expected);
            assert_eq!(request.pending_committed_tokens(), expected);
            assert_eq!(&**request.cache, &[4]);
            assert_eq!(request.stats().refinement_tokens(), 6);
            assert_eq!(request.target_randomness, Some(19));
            assert_eq!(request.status(), SpeculativeRequestStatus::Completed);
            assert_eq!(
                request.request_decoding_mode(crate::decoding::DecodingMode::Autoregressive),
                Err(crate::decoding::DecodingModeError::Terminal)
            );
        }
    }

    #[test]
    fn diffusion_table_cancellation_and_publication_failure_leave_tentative_output_uncommitted() {
        struct CancelAtRefinement {
            cancellation: GenerationCancellationToken,
            executions: AtomicUsize,
        }
        impl SpeculativeLifecycleObserver for CancelAtRefinement {
            fn observe(
                &self,
                stage: SpeculativeLifecycleStage,
            ) -> Result<(), SpeculativeOutputError> {
                if stage == SpeculativeLifecycleStage::Execution
                    && self.executions.fetch_add(1, Ordering::Relaxed) != 0
                {
                    self.cancellation.cancel();
                }
                Ok(())
            }
        }
        for cancellation_case in [true, false] {
            let cancellation = GenerationCancellationToken::new();
            let observer: Arc<dyn SpeculativeLifecycleObserver> = if cancellation_case {
                Arc::new(CancelAtRefinement {
                    cancellation: cancellation.clone(),
                    executions: AtomicUsize::new(0),
                })
            } else {
                Arc::new(LifecycleTrace::failing(
                    SpeculativeLifecycleStage::Publication,
                ))
            };
            let mut executor = MockExecutor {
                refinement: Some(vec![2, 1, 2]),
                ..Default::default()
            };
            let mut cache = Vec::new();
            let mut table = GenerationRequestTable::new(
                Default::default(),
                SpeculativeExecutionTopology::Single,
            )
            .unwrap();
            let output = empty_mock_runtime(1, cancellation).with_lifecycle_observer(observer);
            let id = table
                .submit(
                    &mut executor,
                    &mut cache,
                    vec![4],
                    GenerationRequestConfig {
                        max_tokens: 1,
                        decoding_mode: crate::decoding::DecodingMode::Diffusion { policy: None },
                        ..Default::default()
                    },
                    output,
                    SpeculativeRandomness::new(Some(19), Some(23)),
                    false,
                    (),
                )
                .unwrap();
            let numerical = table.step(&mut executor, false, ());
            assert!(
                numerical.is_ok(),
                "numerical completion precedes publication"
            );
            let result = if cancellation_case {
                numerical
            } else {
                table.step(&mut executor, false, ())
            };
            assert_eq!(result.is_ok(), cancellation_case);
            let request = table.request(id).unwrap();
            assert!(request.sequence().tokens().is_empty());
            assert!(request.pending_committed_tokens().is_empty());
            assert_eq!(&**request.cache, &[4]);
            assert_eq!(request.target_randomness, Some(19));
            assert_eq!(
                request.request_decoding_mode(crate::decoding::DecodingMode::Autoregressive),
                Err(crate::decoding::DecodingModeError::Terminal)
            );
            if cancellation_case {
                assert_eq!(request.status(), SpeculativeRequestStatus::Cancelled);
            }
        }
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
        let config = GenerationRequestConfig {
            max_tokens: 3,
            decoding_mode: crate::decoding::DecodingMode::SelfSpeculation(
                crate::decoding::SpeculationOptions::default().with_proposal_capacity(
                    std::num::NonZeroUsize::new(2).expect("positive proposal capacity"),
                ),
            ),
            temperature: 0.7,
            eos_token_ids: Vec::new(),
        };
        let runtime = empty_mock_runtime(config.max_tokens, GenerationCancellationToken::new())
            .with_lifecycle_observer(observer.clone());
        let mut table = GenerationRequestTable::new(
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
            let mut table = GenerationRequestTable::new(
                SpeculativeSchedulerOptions::default().with_lookahead(false),
                SpeculativeExecutionTopology::Single,
            )
            .unwrap();
            let error = table
                .submit(
                    &mut executor,
                    &mut cache,
                    vec![4],
                    GenerationRequestConfig {
                        max_tokens: 3,
                        decoding_mode: crate::decoding::DecodingMode::SelfSpeculation(
                            crate::decoding::SpeculationOptions::default().with_proposal_capacity(
                                std::num::NonZeroUsize::new(2).expect("positive proposal capacity"),
                            ),
                        ),
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
            drop(table);
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
            &SpeculativeProposalEvidence::CausalDistribution(vec![0.0, 2.0]),
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
    fn target_matching_preserves_an_enumerable_stochastic_target_distribution() {
        let target = vec![0.25_f32.ln(), 0.75_f32.ln()];
        // Even a constant, arbitrarily biased proposal leaves target output unchanged.
        for proposed in [0, 1] {
            let trace = TransactionTrace::default();
            let sampler = MockSampling {
                categorical: true,
                trace: Some(trace.clone()),
                ..Default::default()
            };
            let mut counts = [0; 2];
            let mut accepts = 0;
            for draw in 0..100 {
                let mut random = draw;
                let decision = decide_speculative_proposal(
                    &sampler,
                    &target,
                    &SpeculativeProposalEvidence::TargetMatching,
                    proposed,
                    0.7,
                    Some(&mut random),
                    (),
                )
                .unwrap();
                let token = match decision {
                    ProposalDecision::Accept => {
                        accepts += 1;
                        proposed
                    }
                    ProposalDecision::Reject(token) => token,
                };
                counts[token as usize] += 1;
                assert_eq!(random, draw + 1);
            }
            assert_eq!(counts, [25, 75]);
            assert_eq!(accepts, counts[proposed as usize]);
            assert_eq!(*trace.borrow(), vec!["sample"; 100]);
        }
    }

    #[test]
    fn masked_proposals_use_transactional_target_commit_and_suffix_rollback() {
        for full_acceptance in [false, true] {
            let trace = TransactionTrace::default();
            let mut executor = MockExecutor {
                trace: Some(trace.clone()),
                full_acceptance,
                masked_draft: true,
                ..Default::default()
            };
            let mut cache = vec![4, 5];
            let sampler = MockSampling {
                trace: Some(trace.clone()),
                ..Default::default()
            };
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
                    .map(SpeculativeProposal::token)
                    .collect::<Vec<_>>(),
                [1, 1]
            );
            assert!(proposals.iter().all(|p| p.distribution().is_none()));
            assert!(
                trace.borrow().is_empty(),
                "masked tokens enter verification without causal draft sampling"
            );
            let block = SpeculativeDraftBlock::new(draft, proposals);
            let pending =
                submit_verification_transaction(&mut executor, &mut cache, 5, block, ()).unwrap();
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
            let (sampler, sequence, constraint, publisher) = runtime.into_parts();
            let expected = if full_acceptance {
                vec![5, 1, 1, 1]
            } else {
                vec![5, 1, 0]
            };
            assert_eq!(sequence.tokens(), expected);
            assert_eq!(sampler.committed, expected[1..]);
            assert_eq!(constraint.tokens, expected[1..]);
            assert_eq!(publisher.tokens, expected[1..]);
            assert_eq!(
                cache,
                if full_acceptance {
                    vec![4, 5, 5, 1, 1]
                } else {
                    vec![4, 5, 5, 1]
                }
            );
            assert_eq!(
                published.stats.accepted_tokens,
                if full_acceptance { 2 } else { 1 }
            );
            assert_eq!(*trace.borrow(), ["wait", "commit", "publish"]);
        }
    }

    #[test]
    fn accepted_eos_is_cached_and_leaves_no_committed_suffix() {
        let mut executor = MockExecutor {
            full_acceptance: true,
            masked_draft: true,
            ..Default::default()
        };
        let mut cache = vec![4, 5];
        let mut draft = executor.begin_proposal(&2, 5, 2, ()).unwrap();
        let proposals = propose_block(
            &mut executor,
            &MockSampling::default(),
            &mut draft,
            5,
            2,
            &[5],
            0.0,
            &[1],
            None,
            (),
        )
        .unwrap();
        let pending = submit_verification_transaction(
            &mut executor,
            &mut cache,
            5,
            SpeculativeDraftBlock::new(draft, proposals),
            (),
        )
        .unwrap();
        let mut sequence = GenerationSequence::new(8, [1]);
        sequence.commit(5, TokenTerminalSignals::default()).unwrap();
        let mut runtime = SpeculativeOutputRuntime::new(
            MockSampling::default(),
            sequence,
            MockConstraint::default(),
            MockPublisher::default(),
            GenerationCancellationToken::new(),
        );
        let published = resolve_commit_and_publish(
            &mut executor,
            &mut cache,
            pending,
            &mut runtime,
            None,
            0.0,
            SpeculativeStats::default(),
            SpeculativeSchedulerOptions::default(),
            (),
        )
        .unwrap();
        assert_eq!(runtime.sequence().tokens(), [5, 1]);
        assert_eq!(cache, [4, 5, 5, 1]);
        assert!(published.pending_committed_tokens.is_empty());
        assert!(matches!(
            published.status,
            SpeculativePublicationStatus::Completed
        ));
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
            &SpeculativeProposalEvidence::CausalDistribution(vec![0.0, 1.0]),
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
            &SpeculativeProposalEvidence::CausalDistribution(vec![2.0, 0.0]),
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
            GenerationRequestTable::new(options, SpeculativeExecutionTopology::Single).unwrap();
        let config = GenerationRequestConfig {
            max_tokens: 3,
            decoding_mode: crate::decoding::DecodingMode::SelfSpeculation(
                crate::decoding::SpeculationOptions::default().with_proposal_capacity(
                    std::num::NonZeroUsize::new(2).expect("positive proposal capacity"),
                ),
            ),
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
        let config = GenerationRequestConfig {
            max_tokens: 5,
            decoding_mode: crate::decoding::DecodingMode::SelfSpeculation(
                crate::decoding::SpeculationOptions::default().with_proposal_capacity(
                    std::num::NonZeroUsize::new(2).expect("positive proposal capacity"),
                ),
            ),
            temperature: 0.7,
            eos_token_ids: Vec::new(),
        };
        let mut table = GenerationRequestTable::new(
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
        assert!(output.requests[0].stats.optimistic_draft_blocks > 0);
        assert!(output.requests[0].stats.discarded_optimistic_blocks > 0);
        assert_eq!(output.scheduler.peak_optimistic_branches, 1);
    }

    #[test]
    fn request_table_promotes_only_the_exact_matching_optimistic_suffix() {
        let mut executor = MockExecutor {
            full_acceptance: true,
            ..MockExecutor::default()
        };
        let mut cache = Vec::new();
        let config = GenerationRequestConfig {
            max_tokens: 8,
            decoding_mode: crate::decoding::DecodingMode::SelfSpeculation(
                crate::decoding::SpeculationOptions::default().with_proposal_capacity(
                    std::num::NonZeroUsize::new(2).expect("positive proposal capacity"),
                ),
            ),
            temperature: 0.7,
            eos_token_ids: Vec::new(),
        };
        let mut table = GenerationRequestTable::new(
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
        assert_eq!(request.stats().optimistic_bonus_matches, 1);
        assert_eq!(request.stats().consumed_optimistic_tokens, 1);
        assert_eq!(request.stats().reused_optimistic_tokens, 1);
        assert_eq!(request.stats().reused_optimistic_blocks, 1);
        assert_eq!(request.stats().discarded_optimistic_tokens, 0);
    }

    #[test]
    fn request_table_draft_generation_stops_at_the_sampler_grammar_boundary() {
        let mut executor = MockExecutor::default();
        let mut cache = Vec::new();
        let config = GenerationRequestConfig {
            max_tokens: 8,
            decoding_mode: crate::decoding::DecodingMode::SelfSpeculation(
                crate::decoding::SpeculationOptions::default().with_proposal_capacity(
                    std::num::NonZeroUsize::new(4).expect("positive proposal capacity"),
                ),
            ),
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
        let mut table = GenerationRequestTable::new(
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
        assert_eq!(request.stats().draft_tokens, 1);
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
                    evidence: SpeculativeProposalEvidence::CausalDistribution(vec![0.0, 1.0]),
                },
                SpeculativeProposal {
                    token: 1,
                    evidence: SpeculativeProposalEvidence::CausalDistribution(vec![0.0, 1.0]),
                },
            ],
        };
        let mut pending =
            submit_verification_transaction(&mut executor, &mut cache, 5, block, ()).unwrap();
        pending
            .set_optimistic_branch(SpeculativeOptimisticBranch {
                block: SpeculativeDraftBlock {
                    state: vec![5, 1, 1, 2],
                    proposals: vec![SpeculativeProposal {
                        token: 2,
                        evidence: SpeculativeProposalEvidence::CausalDistribution(vec![
                            0.0, 0.0, 1.0,
                        ]),
                    }],
                },
                assumed_prefix: vec![5, 1, 1],
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
        assert_eq!(published.stats.accepted_tokens, 1);
        assert_eq!(published.stats.discarded_optimistic_tokens, 1);
        assert_eq!(cache, [4, 5, 5, 1]);
        let (_, sequence, constraint, publisher) = runtime.into_parts();
        assert_eq!(sequence.tokens(), [5, 1, 0]);
        assert_eq!(constraint.tokens, [1, 0]);
        assert_eq!(publisher.tokens, [1, 0]);
        assert!(!publisher.cancelled);
        assert_eq!(*trace.borrow(), ["wait", "commit", "publish"]);
    }

    #[test]
    fn causal_transaction_appends_one_input_and_samples_without_draft_evidence() {
        for masked_draft in [false, true] {
            let trace = TransactionTrace::default();
            let mut executor = MockExecutor {
                masked_draft,
                trace: Some(trace.clone()),
                ..MockExecutor::default()
            };
            let mut cache = vec![4, 5];
            let state = executor.begin_proposal(&0, 5, 0, ()).unwrap();
            let pending = submit_verification_transaction(
                &mut executor,
                &mut cache,
                5,
                SpeculativeDraftBlock {
                    state,
                    proposals: Vec::new(),
                },
                (),
            )
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
            assert_eq!(
                cache,
                [4, 5, 5],
                "only the pending input enters canonical K/V"
            );
            assert_eq!(runtime.sequence().tokens(), [5, 1]);
            assert_eq!(runtime.sampler().committed, [1]);
            assert_eq!(published.stats.accepted_tokens, 0);
            assert_eq!(published.stats.emitted_tokens, 1);
            assert!(!executor
                .activations
                .iter()
                .any(|(phase, _)| *phase == "proposal"));
            assert_eq!(*trace.borrow(), ["wait", "commit", "publish"]);
        }
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
                evidence: SpeculativeProposalEvidence::CausalDistribution(vec![0.0, 1.0]),
            }],
        };
        let mut pending =
            submit_verification_transaction(&mut executor, &mut cache, 5, block, ()).unwrap();
        pending
            .set_optimistic_branch(SpeculativeOptimisticBranch {
                block: SpeculativeDraftBlock {
                    state: vec![5, 1, 2],
                    proposals: vec![SpeculativeProposal {
                        token: 2,
                        evidence: SpeculativeProposalEvidence::CausalDistribution(vec![
                            0.0, 0.0, 1.0,
                        ]),
                    }],
                },
                assumed_prefix: vec![5, 1],
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

        assert_eq!(stats.discarded_optimistic_tokens, 1);
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
        Completion,
        Commit,
        Restore,
    }

    impl fmt::Display for TransactionFailure {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str(match self {
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
        type CacheCheckpoint = TransactionCheckpoint;
        type Verification = TransactionVerification;
        type Logits = Vec<f32>;
        type Context<'a>
            = ()
        where
            Self: 'a;
        type Completion = DelayedCompletion;
        type Telemetry = ();
        type Error = TransactionFailure;

        fn prefill<'a>(
            &mut self,
            input: Self::Input,
            cache: &mut Self::Cache,
            _: Self::Context<'a>,
        ) -> Result<SpeculativePrefill<Self::TargetState, Self::Logits>, Self::Error>
        where
            Self: 'a,
        {
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
        ) -> Result<Self::DraftState, Self::Error>
        where
            Self: 'a,
        {
            Ok(TransactionDraftState {
                values: vec![last_token],
                _probe: None,
            })
        }

        fn proposal_step<'a>(
            &mut self,
            state: &mut Self::DraftState,
            last_token: u32,
            _: Self::Context<'a>,
        ) -> Result<SpeculativeProposalStep<Self::Logits>, Self::Error>
        where
            Self: 'a,
        {
            state.values.push(last_token);
            Ok(SpeculativeProposalStep::CausalLogits(vec![0.0, 1.0]))
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
        ) -> Result<(), Self::Error>
        where
            Self: 'a,
        {
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
        ) -> Result<Submission<Self::Verification, Self::Completion>, Self::Error>
        where
            Self: 'a,
        {
            self.ready.set(false);
            self.completion_polls.set(0);
            cache.target.extend_from_slice(input_tokens);
            self.trace.borrow_mut().push("submit");
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
        ) -> Result<Self::Logits, Self::Error>
        where
            Self: 'a,
        {
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
        ) -> Result<SpeculativeCommit<Self::TargetState>, Self::Error>
        where
            Self: 'a,
        {
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
                SpeculativeProposal::causal(1, vec![0.0, 1.0]),
                SpeculativeProposal::causal(1, vec![0.0, 1.0]),
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
            let config = GenerationRequestConfig {
                max_tokens: 3,
                decoding_mode: crate::decoding::DecodingMode::SelfSpeculation(
                    crate::decoding::SpeculationOptions::default().with_proposal_capacity(
                        std::num::NonZeroUsize::new(2).expect("positive proposal capacity"),
                    ),
                ),
                temperature: 0.7,
                eos_token_ids: Vec::new(),
            };
            let runtime =
                empty_transaction_runtime(config.max_tokens, trace.clone(), attempts.clone())
                    .with_lifecycle_observer(observer.clone());
            let mut table = GenerationRequestTable::new(
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
            drop(table);
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
        let mut table = GenerationRequestTable::new(
            SpeculativeSchedulerOptions::default().with_lookahead(false),
            SpeculativeExecutionTopology::Single,
        )
        .unwrap();
        let error = table
            .submit(
                &mut executor,
                &mut cache,
                vec![4],
                GenerationRequestConfig {
                    max_tokens: 3,
                    decoding_mode: crate::decoding::DecodingMode::SelfSpeculation(
                        crate::decoding::SpeculationOptions::default().with_proposal_capacity(
                            std::num::NonZeroUsize::new(2).expect("positive proposal capacity"),
                        ),
                    ),
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
        drop(table);
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
        let mut table = GenerationRequestTable::new(
            SpeculativeSchedulerOptions::default().with_lookahead(false),
            SpeculativeExecutionTopology::Single,
        )
        .unwrap();
        let error = table
            .submit(
                &mut executor,
                &mut cache,
                vec![4],
                GenerationRequestConfig {
                    max_tokens: 3,
                    decoding_mode: crate::decoding::DecodingMode::SelfSpeculation(
                        crate::decoding::SpeculationOptions::default().with_proposal_capacity(
                            std::num::NonZeroUsize::new(2).expect("positive proposal capacity"),
                        ),
                    ),
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
        let mut table = GenerationRequestTable::new(
            SpeculativeSchedulerOptions::default().with_lookahead(false),
            SpeculativeExecutionTopology::Single,
        )
        .unwrap();
        let id = table
            .submit(
                &mut executor,
                &mut cache,
                vec![4],
                GenerationRequestConfig {
                    max_tokens: 3,
                    decoding_mode: crate::decoding::DecodingMode::SelfSpeculation(
                        crate::decoding::SpeculationOptions::default().with_proposal_capacity(
                            std::num::NonZeroUsize::new(2).expect("positive proposal capacity"),
                        ),
                    ),
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
        let mut table = GenerationRequestTable::new(
            SpeculativeSchedulerOptions::default().with_lookahead(false),
            SpeculativeExecutionTopology::Single,
        )
        .unwrap();
        let id = table
            .submit(
                &mut executor,
                &mut cache,
                vec![4],
                GenerationRequestConfig {
                    max_tokens: 3,
                    decoding_mode: crate::decoding::DecodingMode::SelfSpeculation(
                        crate::decoding::SpeculationOptions::default().with_proposal_capacity(
                            std::num::NonZeroUsize::new(2).expect("positive proposal capacity"),
                        ),
                    ),
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
        let config = GenerationRequestConfig {
            max_tokens: 3,
            decoding_mode: crate::decoding::DecodingMode::SelfSpeculation(
                crate::decoding::SpeculationOptions::default().with_proposal_capacity(
                    std::num::NonZeroUsize::new(2).expect("positive proposal capacity"),
                ),
            ),
            temperature: 0.7,
            eos_token_ids: Vec::new(),
        };
        let runtime = empty_transaction_runtime(config.max_tokens, trace.clone(), attempts.clone())
            .with_lifecycle_observer(observer);
        let mut table = GenerationRequestTable::new(
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
        let config = GenerationRequestConfig {
            max_tokens: 3,
            decoding_mode: crate::decoding::DecodingMode::SelfSpeculation(
                crate::decoding::SpeculationOptions::default().with_proposal_capacity(
                    std::num::NonZeroUsize::new(2).expect("positive proposal capacity"),
                ),
            ),
            temperature: 0.7,
            eos_token_ids: Vec::new(),
        };
        let mut table = GenerationRequestTable::new(
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
        assert_eq!(output.requests[0].stats().target_tokens, 8);
        assert_eq!(output.requests[0].stats().emitted_tokens, 3);
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
        let config = GenerationRequestConfig {
            max_tokens: 3,
            decoding_mode: crate::decoding::DecodingMode::SelfSpeculation(
                crate::decoding::SpeculationOptions::default().with_proposal_capacity(
                    std::num::NonZeroUsize::new(2).expect("positive proposal capacity"),
                ),
            ),
            temperature: 0.7,
            eos_token_ids: Vec::new(),
        };
        let mut table = GenerationRequestTable::new(
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
        let config = GenerationRequestConfig {
            max_tokens: 4,
            decoding_mode: crate::decoding::DecodingMode::SelfSpeculation(
                crate::decoding::SpeculationOptions::default().with_proposal_capacity(
                    std::num::NonZeroUsize::new(2).expect("positive proposal capacity"),
                ),
            ),
            temperature: 0.7,
            eos_token_ids: Vec::new(),
        };
        let mut table = GenerationRequestTable::new(
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

        assert_eq!(error.to_string(), "restore failed");
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
