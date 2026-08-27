//! Portable generation configuration, lifecycle, and speculative bookkeeping.

use serde::{Deserialize, Serialize};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

/// Why generation reached a terminal state.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    /// A checkpoint end-of-sequence token was committed.
    Eos,
    /// A caller or protocol stop sequence matched.
    StopSequence,
    /// The committed generation grammar reached an accepting state.
    GrammarComplete,
    /// The configured output-token budget was exhausted.
    MaxTokens,
    /// The caller cooperatively cancelled generation.
    Cancelled,
}

/// Cheap thread-safe cooperative cancellation observed between submissions.
#[derive(Debug, Clone, Default)]
pub struct GenerationCancellationToken {
    cancelled: Arc<AtomicBool>,
}

impl GenerationCancellationToken {
    /// Creates an active token.
    pub fn new() -> Self {
        Self::default()
    }

    /// Permanently requests cancellation for every clone.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    /// Returns whether cancellation has been requested.
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }
}

/// Terminal signals observed while committing one token.
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
pub struct TokenTerminalSignals {
    /// Decoded output matched a stop sequence.
    pub stop_sequence: bool,
    /// The committed grammar is complete.
    pub grammar_complete: bool,
}

/// Result of one canonical token commit.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct TokenCommit {
    /// Token id committed to the logical output.
    pub token_id: u32,
    /// Zero-based generated-token position.
    pub position: usize,
    /// Terminal reason selected after this commit, if any.
    pub finish_reason: Option<FinishReason>,
}

/// Canonical committed-token sequence and terminal-condition precedence.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct GenerationSequence {
    max_tokens: usize,
    eos_token_ids: Vec<u32>,
    tokens: Vec<u32>,
    finish_reason: Option<FinishReason>,
}

impl GenerationSequence {
    /// Creates an empty output sequence with a fixed token budget.
    pub fn new(max_tokens: usize, eos_token_ids: impl IntoIterator<Item = u32>) -> Self {
        let mut eos_token_ids = eos_token_ids.into_iter().collect::<Vec<_>>();
        eos_token_ids.sort_unstable();
        eos_token_ids.dedup();
        Self {
            max_tokens,
            eos_token_ids,
            tokens: Vec::with_capacity(max_tokens),
            finish_reason: (max_tokens == 0).then_some(FinishReason::MaxTokens),
        }
    }

    /// Commits one token and applies stop, grammar, EOS, and budget precedence.
    pub fn commit(
        &mut self,
        token_id: u32,
        signals: TokenTerminalSignals,
    ) -> Result<TokenCommit, GenerationError> {
        if self.finish_reason.is_some() {
            return Err(GenerationError::AlreadyFinished);
        }
        let position = self.tokens.len();
        self.tokens.push(token_id);
        let finish_reason = signals
            .stop_sequence
            .then_some(FinishReason::StopSequence)
            .or_else(|| {
                signals
                    .grammar_complete
                    .then_some(FinishReason::GrammarComplete)
            })
            .or_else(|| {
                self.eos_token_ids
                    .binary_search(&token_id)
                    .is_ok()
                    .then_some(FinishReason::Eos)
            })
            .or_else(|| (self.tokens.len() == self.max_tokens).then_some(FinishReason::MaxTokens));
        self.finish_reason = finish_reason;
        Ok(TokenCommit {
            token_id,
            position,
            finish_reason,
        })
    }

    /// Applies cancellation if the sequence has not already terminated.
    pub fn cancel(&mut self) -> bool {
        if self.tokens.is_empty() && self.finish_reason == Some(FinishReason::MaxTokens) {
            self.finish_reason = Some(FinishReason::Cancelled);
            true
        } else if self.finish_reason.is_some() {
            false
        } else {
            self.finish_reason = Some(FinishReason::Cancelled);
            true
        }
    }

    /// Observes a cooperative token at a submission boundary.
    pub fn observe_cancellation(&mut self, cancellation: &GenerationCancellationToken) -> bool {
        cancellation.is_cancelled() && self.cancel()
    }

    /// Committed tokenizer ids in canonical order.
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    /// Consumes the state into committed tokenizer ids.
    pub fn into_tokens(self) -> Vec<u32> {
        self.tokens
    }

    /// Number of remaining token slots.
    pub fn remaining(&self) -> usize {
        self.max_tokens.saturating_sub(self.tokens.len())
    }

    /// Configured output-token budget.
    pub const fn max_tokens(&self) -> usize {
        self.max_tokens
    }

    /// Selected terminal reason.
    pub const fn finish_reason(&self) -> Option<FinishReason> {
        self.finish_reason
    }

    /// Whether no further token may be committed.
    pub const fn is_finished(&self) -> bool {
        self.finish_reason.is_some()
    }
}

/// Kind of non-proposal token emitted by speculative verification.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum SpeculativeTail {
    /// A rejected proposal was replaced from the target distribution.
    Replacement,
    /// Every proposal was accepted and the target emitted its bonus token.
    Bonus,
}

/// Canonical bookkeeping for one proposal/verification transaction.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct SpeculativeRound {
    proposal_count: usize,
    accepted: usize,
    committed_tokens: Vec<u32>,
    tail: Option<SpeculativeTail>,
    terminal: bool,
}

impl SpeculativeRound {
    /// Starts a verification round for a non-empty proposal block.
    pub fn new(proposal_count: usize) -> Result<Self, GenerationError> {
        if proposal_count == 0 {
            return Err(GenerationError::EmptyProposalBlock);
        }
        Ok(Self {
            proposal_count,
            accepted: 0,
            committed_tokens: Vec::with_capacity(proposal_count + 1),
            tail: None,
            terminal: false,
        })
    }

    /// Records the next accepted proposal token.
    pub fn accept(&mut self, token: u32, terminal: bool) -> Result<(), GenerationError> {
        if self.tail.is_some() || self.accepted == self.proposal_count || self.terminal {
            return Err(GenerationError::InvalidSpeculativeTransition);
        }
        self.accepted += 1;
        self.committed_tokens.push(token);
        self.terminal = terminal;
        Ok(())
    }

    /// Records the target replacement for the first rejected proposal.
    pub fn reject_with(&mut self, token: u32, terminal: bool) -> Result<(), GenerationError> {
        if self.tail.is_some() || self.accepted == self.proposal_count || self.terminal {
            return Err(GenerationError::InvalidSpeculativeTransition);
        }
        self.tail = Some(SpeculativeTail::Replacement);
        self.committed_tokens.push(token);
        self.terminal = terminal;
        Ok(())
    }

    /// Records the target bonus after complete proposal acceptance.
    pub fn bonus(&mut self, token: u32, terminal: bool) -> Result<(), GenerationError> {
        if self.tail.is_some() || self.accepted != self.proposal_count || self.terminal {
            return Err(GenerationError::InvalidSpeculativeTransition);
        }
        self.tail = Some(SpeculativeTail::Bonus);
        self.committed_tokens.push(token);
        self.terminal = terminal;
        Ok(())
    }

    /// Whether every proposal was accepted.
    pub const fn is_full_acceptance(&self) -> bool {
        self.accepted == self.proposal_count
    }

    /// Produces the exact cache-retention and output-publication plan.
    pub fn commit_plan(&self) -> Result<SpeculativeCommitPlan<'_>, GenerationError> {
        if !self.terminal && self.tail.is_none() {
            return Err(GenerationError::IncompleteSpeculativeRound);
        }
        Ok(SpeculativeCommitPlan {
            accepted_proposals: self.accepted,
            committed_tokens: &self.committed_tokens,
            verified_inputs: if self.tail.is_some() {
                1 + self.accepted
            } else {
                self.accepted
            },
            full_acceptance: self.accepted == self.proposal_count,
            tail: self.tail,
            terminal: self.terminal,
        })
    }
}

/// Borrowed resolution plan for a speculative transaction.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct SpeculativeCommitPlan<'a> {
    /// Number of accepted assistant proposals.
    pub accepted_proposals: usize,
    /// Tokens that become visible after the backend cache commit succeeds.
    pub committed_tokens: &'a [u32],
    /// Verification inputs retained in the target cache.
    pub verified_inputs: usize,
    /// Whether every assistant proposal was accepted.
    pub full_acceptance: bool,
    /// Optional replacement or target bonus.
    pub tail: Option<SpeculativeTail>,
    /// Whether the last committed token terminated generation.
    pub terminal: bool,
}

/// Reuse decision for a proposal block drafted against an assumed prefix.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum OptimisticReuseDecision {
    /// Terminal output makes the complete branch unusable.
    DiscardTerminal,
    /// The target bonus differed from the optimistic first token.
    DiscardMismatch,
    /// The matching first token consumed the complete optimistic block.
    MatchedConsumed,
    /// The first optimistic token matched and later proposals remain reusable.
    MatchedRetained,
}

/// Resolves optimistic proposal reuse without inspecting backend state.
pub fn resolve_optimistic_reuse(
    assumed_prefix: &[u32],
    canonical_prefix: &[u32],
    optimistic_tokens: &[u32],
    bonus: u32,
    terminal: bool,
) -> Result<OptimisticReuseDecision, GenerationError> {
    if assumed_prefix != canonical_prefix {
        return Err(GenerationError::OptimisticPrefixDiverged);
    }
    let first = optimistic_tokens
        .first()
        .ok_or(GenerationError::EmptyOptimisticBranch)?;
    if terminal {
        return Ok(OptimisticReuseDecision::DiscardTerminal);
    }
    if *first != bonus {
        return Ok(OptimisticReuseDecision::DiscardMismatch);
    }
    Ok(if optimistic_tokens.len() == 1 {
        OptimisticReuseDecision::MatchedConsumed
    } else {
        OptimisticReuseDecision::MatchedRetained
    })
}

/// Options shared by speculative multi-token backends.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MtpConfig {
    /// Maximum number of output tokens, including terminal tokens.
    pub max_tokens: usize,
    /// Maximum assistant proposals per verification round.
    pub max_draft_tokens: usize,
    /// Sampling temperature. Zero selects greedy verification.
    pub temperature: f32,
    /// Token ids that terminate a sequence.
    pub eos_token_ids: Vec<u32>,
}

impl Default for MtpConfig {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            max_draft_tokens: 4,
            temperature: 0.0,
            eos_token_ids: Vec::new(),
        }
    }
}

impl MtpConfig {
    /// Validates backend-independent speculative settings.
    pub fn validate(&self) -> Result<(), GenerationError> {
        if self.max_draft_tokens == 0 {
            return Err(GenerationError::ZeroDraftTokens);
        }
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            return Err(GenerationError::InvalidTemperature(self.temperature));
        }
        Ok(())
    }
}

/// Bounded fair-scheduler settings for speculative requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MtpSchedulerOptions {
    /// Maximum retained target verification transactions.
    pub max_in_flight_verifications: usize,
    /// Maximum retained optimistic branches.
    pub max_optimistic_branches: usize,
    /// Proposal blocks drafted ahead per request; currently zero or one.
    pub lookahead_blocks: usize,
    /// Disable lookahead adaptively when discarded work dominates reuse.
    pub adaptive_lookahead: bool,
    /// Resolved branches required before adaptive disabling.
    pub adaptive_lookahead_min_blocks: usize,
}

impl Default for MtpSchedulerOptions {
    fn default() -> Self {
        Self {
            max_in_flight_verifications: 1,
            max_optimistic_branches: 1,
            lookahead_blocks: 1,
            adaptive_lookahead: true,
            adaptive_lookahead_min_blocks: 4,
        }
    }
}

impl MtpSchedulerOptions {
    /// Enables or disables same-request optimistic lookahead.
    pub fn with_lookahead(mut self, enabled: bool) -> Self {
        self.lookahead_blocks = usize::from(enabled);
        if enabled {
            self.max_optimistic_branches = self.max_optimistic_branches.max(1);
        }
        self
    }

    /// Validates scheduler capacity and lookahead invariants.
    pub fn validate(self) -> Result<Self, GenerationError> {
        if self.max_in_flight_verifications == 0 {
            return Err(GenerationError::ZeroInFlightVerifications);
        }
        if self.lookahead_blocks > 1 {
            return Err(GenerationError::TooManyLookaheadBlocks);
        }
        if self.lookahead_blocks > 0 && self.max_optimistic_branches == 0 {
            return Err(GenerationError::LookaheadWithoutBranchCapacity);
        }
        if self.lookahead_blocks > 0
            && self.adaptive_lookahead
            && self.adaptive_lookahead_min_blocks == 0
        {
            return Err(GenerationError::ZeroAdaptiveLookaheadWindow);
        }
        Ok(self)
    }
}

/// Stable speculative scheduler request identifier.
#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct MtpRequestId(usize);

impl MtpRequestId {
    /// Creates an identifier from its stable scheduler insertion index.
    pub const fn new(index: usize) -> Self {
        Self(index)
    }

    /// Returns the scheduler insertion index.
    pub const fn index(self) -> usize {
        self.0
    }
}

/// Explicit speculative request/round state.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MtpRequestPhase {
    /// Target prompt prefill and first-token sampling.
    Prefill,
    /// Committed target state is ready to seed proposals.
    ReadyToDraft,
    /// A proposal block is ready for target submission.
    ReadyToSubmitVerification,
    /// Target verification is submitted and unresolved.
    TargetVerificationInFlight,
    /// Same-request continuation is being drafted optimistically.
    OptimisticDraftInProgress,
    /// Verification is in flight and its optimistic branch is ready.
    OptimisticDraftReady,
    /// Target results are being accepted/rejected and committed.
    VerificationResolution,
    /// The request reached a normal terminal condition.
    Completed,
    /// The request was cancelled.
    Cancelled,
}

/// Result of requesting cancellation at a speculative submission boundary.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum MtpCancellationDisposition {
    /// The request was already terminal.
    AlreadyTerminal,
    /// No backend submission is retained, so cancellation completed now.
    CancelNow,
    /// A backend submission must reach its exact safe boundary first.
    Deferred,
}

/// Validated backend-neutral lifecycle of one speculative request.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct MtpRequestLifecycle {
    phase: MtpRequestPhase,
    cancellation_pending: bool,
}

impl Default for MtpRequestLifecycle {
    fn default() -> Self {
        Self::new()
    }
}

impl MtpRequestLifecycle {
    /// Starts a request at prompt prefill.
    pub const fn new() -> Self {
        Self {
            phase: MtpRequestPhase::Prefill,
            cancellation_pending: false,
        }
    }

    /// Creates a request completed before backend submission.
    pub const fn completed() -> Self {
        Self {
            phase: MtpRequestPhase::Completed,
            cancellation_pending: false,
        }
    }

    /// Creates a request cancelled before backend submission.
    pub const fn cancelled() -> Self {
        Self {
            phase: MtpRequestPhase::Cancelled,
            cancellation_pending: false,
        }
    }

    /// Current lifecycle phase.
    pub const fn phase(&self) -> MtpRequestPhase {
        self.phase
    }

    /// Whether cancellation must be applied after an exact backend boundary.
    pub const fn cancellation_pending(&self) -> bool {
        self.cancellation_pending
    }

    /// Whether the request can own no further submissions.
    pub const fn is_terminal(&self) -> bool {
        matches!(
            self.phase,
            MtpRequestPhase::Completed | MtpRequestPhase::Cancelled
        )
    }

    /// Requests cancellation without discarding an in-flight backend transaction.
    pub fn request_cancellation(
        &mut self,
        submission_retained: bool,
    ) -> Result<MtpCancellationDisposition, GenerationError> {
        if self.is_terminal() {
            return Ok(MtpCancellationDisposition::AlreadyTerminal);
        }
        if submission_retained {
            self.cancellation_pending = true;
            Ok(MtpCancellationDisposition::Deferred)
        } else {
            self.transition(MtpRequestPhase::Cancelled)?;
            Ok(MtpCancellationDisposition::CancelNow)
        }
    }

    /// Applies one legal lifecycle transition.
    pub fn transition(&mut self, next: MtpRequestPhase) -> Result<(), GenerationError> {
        let allowed = matches!(
            (self.phase, next),
            (MtpRequestPhase::Prefill, MtpRequestPhase::ReadyToDraft)
                | (MtpRequestPhase::Prefill, MtpRequestPhase::Completed)
                | (MtpRequestPhase::Prefill, MtpRequestPhase::Cancelled)
                | (
                    MtpRequestPhase::ReadyToDraft,
                    MtpRequestPhase::ReadyToSubmitVerification
                )
                | (MtpRequestPhase::ReadyToDraft, MtpRequestPhase::Completed)
                | (MtpRequestPhase::ReadyToDraft, MtpRequestPhase::Cancelled)
                | (
                    MtpRequestPhase::ReadyToSubmitVerification,
                    MtpRequestPhase::TargetVerificationInFlight
                )
                | (
                    MtpRequestPhase::ReadyToSubmitVerification,
                    MtpRequestPhase::Cancelled
                )
                | (
                    MtpRequestPhase::TargetVerificationInFlight,
                    MtpRequestPhase::OptimisticDraftInProgress
                )
                | (
                    MtpRequestPhase::TargetVerificationInFlight,
                    MtpRequestPhase::VerificationResolution
                )
                | (
                    MtpRequestPhase::OptimisticDraftInProgress,
                    MtpRequestPhase::OptimisticDraftReady
                )
                | (
                    MtpRequestPhase::OptimisticDraftReady,
                    MtpRequestPhase::VerificationResolution
                )
                | (
                    MtpRequestPhase::VerificationResolution,
                    MtpRequestPhase::ReadyToDraft
                )
                | (
                    MtpRequestPhase::VerificationResolution,
                    MtpRequestPhase::Completed
                )
                | (
                    MtpRequestPhase::VerificationResolution,
                    MtpRequestPhase::Cancelled
                )
        );
        if !allowed {
            return Err(GenerationError::InvalidMtpPhaseTransition {
                from: self.phase,
                to: next,
            });
        }
        self.phase = next;
        if self.is_terminal() {
            self.cancellation_pending = false;
        }
        Ok(())
    }
}

/// Sampling values declared by a checkpoint generation configuration.
#[derive(Debug, Clone, Default, PartialEq, Deserialize, Serialize)]
pub struct CheckpointGenerationConfig {
    /// Whether stochastic sampling is recommended.
    #[serde(default)]
    pub do_sample: Option<bool>,
    /// Recommended temperature.
    #[serde(default)]
    pub temperature: Option<f32>,
    /// Recommended top-k cutoff.
    #[serde(default)]
    pub top_k: Option<i32>,
    /// Recommended nucleus probability.
    #[serde(default)]
    pub top_p: Option<f32>,
    /// Recommended minimum-probability filter.
    #[serde(default)]
    pub min_p: Option<f32>,
    /// Recommended multiplicative repetition penalty.
    #[serde(default)]
    pub repetition_penalty: Option<f32>,
    /// Recommended generated-history window for repetition penalties.
    #[serde(default)]
    pub repeat_last_n: Option<i32>,
    /// Recommended additive frequency penalty.
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    /// Recommended additive presence penalty.
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    /// Recommended maximum number of new tokens.
    #[serde(default)]
    pub max_new_tokens: Option<usize>,
}

/// Per-request overrides layered over checkpoint generation settings.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct GenerationConfigOverrides {
    /// Overrides stochastic-versus-greedy selection.
    pub do_sample: Option<bool>,
    /// Overrides temperature.
    pub temperature: Option<f32>,
    /// Overrides top-k filtering.
    pub top_k: Option<i32>,
    /// Overrides top-p filtering.
    pub top_p: Option<f32>,
    /// Overrides min-p filtering.
    pub min_p: Option<f32>,
    /// Overrides the multiplicative repetition penalty.
    pub repetition_penalty: Option<f32>,
    /// Overrides the generated-history window used by repetition penalties.
    pub repeat_last_n: Option<i32>,
    /// Overrides the additive frequency penalty.
    pub frequency_penalty: Option<f32>,
    /// Overrides the additive presence penalty.
    pub presence_penalty: Option<f32>,
    /// Overrides the output-token budget.
    pub max_new_tokens: Option<usize>,
}

/// Fully resolved and validated generation settings.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ResolvedGenerationConfig {
    /// Whether sampling is stochastic.
    pub do_sample: bool,
    /// Effective temperature.
    pub temperature: f32,
    /// Effective top-k cutoff.
    pub top_k: i32,
    /// Effective top-p probability.
    pub top_p: f32,
    /// Effective min-p probability.
    pub min_p: f32,
    /// Effective multiplicative repetition penalty.
    pub repetition_penalty: f32,
    /// Effective generated-history window for repetition penalties.
    pub repeat_last_n: i32,
    /// Effective additive frequency penalty.
    pub frequency_penalty: f32,
    /// Effective additive presence penalty.
    pub presence_penalty: f32,
    /// Effective token budget, when declared.
    pub max_new_tokens: Option<usize>,
}

/// Resolves checkpoint and request sampling settings without a tensor runtime.
pub fn resolve_generation_config(
    checkpoint: Option<&CheckpointGenerationConfig>,
    overrides: GenerationConfigOverrides,
) -> Result<ResolvedGenerationConfig, GenerationError> {
    let checkpoint_present = checkpoint.is_some();
    let checkpoint = checkpoint.cloned().unwrap_or_default();
    let (do_sample, temperature) = if let Some(do_sample) = overrides.do_sample {
        if do_sample {
            (
                true,
                overrides
                    .temperature
                    .or(checkpoint.temperature)
                    .unwrap_or(1.0),
            )
        } else {
            (false, 0.0)
        }
    } else if let Some(temperature) = overrides.temperature {
        (temperature > 0.0, temperature)
    } else if checkpoint.do_sample.unwrap_or(false) {
        (true, checkpoint.temperature.unwrap_or(1.0))
    } else {
        (false, 0.0)
    };
    let resolved = ResolvedGenerationConfig {
        do_sample,
        temperature,
        top_k: overrides
            .top_k
            .or(checkpoint.top_k)
            .unwrap_or(if checkpoint_present { 50 } else { 40 }),
        top_p: overrides
            .top_p
            .or(checkpoint.top_p)
            .unwrap_or(if checkpoint_present { 1.0 } else { 0.95 }),
        min_p: overrides
            .min_p
            .or(checkpoint.min_p)
            .unwrap_or(if checkpoint_present { 0.0 } else { 0.05 }),
        repetition_penalty: overrides
            .repetition_penalty
            .or(checkpoint.repetition_penalty)
            .unwrap_or(1.0),
        repeat_last_n: overrides
            .repeat_last_n
            .or(checkpoint.repeat_last_n)
            .unwrap_or(64),
        frequency_penalty: overrides
            .frequency_penalty
            .or(checkpoint.frequency_penalty)
            .unwrap_or(0.0),
        presence_penalty: overrides
            .presence_penalty
            .or(checkpoint.presence_penalty)
            .unwrap_or(0.0),
        max_new_tokens: overrides.max_new_tokens.or(checkpoint.max_new_tokens),
    };
    if !resolved.temperature.is_finite() || resolved.temperature < 0.0 {
        return Err(GenerationError::InvalidTemperature(resolved.temperature));
    }
    if resolved.do_sample && resolved.temperature == 0.0 {
        return Err(GenerationError::StochasticZeroTemperature);
    }
    if resolved.top_k < 0 {
        return Err(GenerationError::InvalidTopK(resolved.top_k));
    }
    if !resolved.top_p.is_finite() || !(0.0..=1.0).contains(&resolved.top_p) {
        return Err(GenerationError::InvalidTopP(resolved.top_p));
    }
    if !resolved.min_p.is_finite() || !(0.0..=1.0).contains(&resolved.min_p) {
        return Err(GenerationError::InvalidMinP(resolved.min_p));
    }
    if !resolved.repetition_penalty.is_finite() || resolved.repetition_penalty <= 0.0 {
        return Err(GenerationError::InvalidRepetitionPenalty(
            resolved.repetition_penalty,
        ));
    }
    if !resolved.frequency_penalty.is_finite() {
        return Err(GenerationError::InvalidFrequencyPenalty(
            resolved.frequency_penalty,
        ));
    }
    if !resolved.presence_penalty.is_finite() {
        return Err(GenerationError::InvalidPresencePenalty(
            resolved.presence_penalty,
        ));
    }
    if resolved.max_new_tokens == Some(0) {
        return Err(GenerationError::ZeroTokenBudget);
    }
    Ok(resolved)
}

/// Semantic event emitted by generation orchestration.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub enum SemanticEvent {
    /// Incremental reasoning content.
    ReasoningDelta(String),
    /// Incremental user-visible text.
    TextDelta(String),
    /// A structured tool call began.
    ToolCallStart {
        /// Zero-based tool-call position in the assistant turn.
        index: usize,
        /// Stable tool-call identifier.
        id: String,
        /// Tool name selected by the model.
        name: String,
    },
    /// Incremental structured tool arguments.
    ToolArgumentsDelta {
        /// Zero-based tool-call position in the assistant turn.
        index: usize,
        /// A fragment of the tool call's JSON arguments.
        json_fragment: String,
    },
    /// A structured tool call ended.
    ToolCallEnd,
    /// Generation terminated.
    Finished {
        /// The condition that ended the stream.
        reason: FinishReason,
    },
}

/// Invalid backend-neutral generation configuration or state transition.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum GenerationError {
    /// A token was committed after termination.
    #[error("generation has already finished")]
    AlreadyFinished,
    /// A proposal transaction cannot be empty.
    #[error("speculative verification requires at least one proposal")]
    EmptyProposalBlock,
    /// Proposal acceptance/rejection order was invalid.
    #[error("invalid speculative verification state transition")]
    InvalidSpeculativeTransition,
    /// No terminal outcome or replacement/bonus tail resolved the round.
    #[error("speculative verification round is not resolved")]
    IncompleteSpeculativeRound,
    /// Optimistic state was forked from a different prefix.
    #[error("optimistic proposal prefix diverged from the canonical committed prefix")]
    OptimisticPrefixDiverged,
    /// An optimistic branch contained no proposals.
    #[error("optimistic proposal branch is empty")]
    EmptyOptimisticBranch,
    /// A second optimistic branch was installed for one target transaction.
    #[error("speculative verification already retains an optimistic branch")]
    OptimisticBranchAlreadyPresent,
    /// A speculative request identifier does not belong to this table.
    #[error("unknown MTP request id {index}")]
    UnknownMtpRequest {
        /// Requested stable insertion index.
        index: usize,
    },
    /// A promoted proposal block no longer fits the canonical request budget.
    #[error("promoted MTP block has {proposed} proposals but canonical capacity is {capacity}")]
    ProposalCapacityExceeded {
        /// Proposals retained by the promoted block.
        proposed: usize,
        /// Current canonical proposal capacity.
        capacity: usize,
    },
    /// A speculative request table was consumed before reaching terminal state.
    #[error("cannot finish an MTP scheduler with active requests")]
    ActiveSpeculativeRequests,
    /// A terminal speculative request did not retain its canonical finish reason.
    #[error("completed MTP request {index} has no finish reason")]
    MissingMtpFinishReason {
        /// Stable request insertion index.
        index: usize,
    },
    /// No assistant proposal may be generated per round.
    #[error("MTP max_draft_tokens must be positive")]
    ZeroDraftTokens,
    /// The selected backend cannot submit an assistant proposal.
    #[error("MTP backend does not permit any draft tokens")]
    NoBackendDraftCapacity,
    /// Temperature is NaN, infinite, or negative.
    #[error("temperature must be finite and non-negative, got {0}")]
    InvalidTemperature(f32),
    /// Stochastic sampling cannot use zero temperature.
    #[error("do_sample=true requires a temperature greater than zero")]
    StochasticZeroTemperature,
    /// Mirostat V2 target surprise is non-finite or non-positive.
    #[error("Mirostat V2 tau must be finite and positive, got {0}")]
    InvalidMirostatTau(f32),
    /// Mirostat V2 adaptation rate is non-finite or non-positive.
    #[error("Mirostat V2 eta must be finite and positive, got {0}")]
    InvalidMirostatEta(f32),
    /// Top-k is negative.
    #[error("top_k must be non-negative, got {0}")]
    InvalidTopK(i32),
    /// Top-p lies outside zero through one.
    #[error("top_p must be between zero and one, got {0}")]
    InvalidTopP(f32),
    /// Min-p lies outside zero through one.
    #[error("min_p must be between zero and one, got {0}")]
    InvalidMinP(f32),
    /// Repetition penalty is non-finite or non-positive.
    #[error("repetition_penalty must be finite and positive, got {0}")]
    InvalidRepetitionPenalty(f32),
    /// Frequency penalty is non-finite.
    #[error("frequency_penalty must be finite, got {0}")]
    InvalidFrequencyPenalty(f32),
    /// Presence penalty is non-finite.
    #[error("presence_penalty must be finite, got {0}")]
    InvalidPresencePenalty(f32),
    /// An explicit token budget was zero.
    #[error("max_new_tokens must be positive when supplied")]
    ZeroTokenBudget,
    /// Scheduler cannot retain any target transaction.
    #[error("MTP max_in_flight_verifications must be positive")]
    ZeroInFlightVerifications,
    /// Current scheduler supports no more than one lookahead block.
    #[error("MTP scheduler currently supports at most one lookahead block")]
    TooManyLookaheadBlocks,
    /// Lookahead was enabled with no branch capacity.
    #[error("MTP lookahead requires at least one optimistic branch slot")]
    LookaheadWithoutBranchCapacity,
    /// Adaptive lookahead needs a non-zero observation window.
    #[error("MTP adaptive_lookahead_min_blocks must be positive")]
    ZeroAdaptiveLookaheadWindow,
    /// Active requests expose no legal scheduler action.
    #[error("MTP scheduler reached a non-terminal state with no eligible operation")]
    StalledSpeculativeSchedule,
    /// The requested speculative lifecycle edge is invalid.
    #[error("invalid MTP request phase transition from {from:?} to {to:?}")]
    InvalidMtpPhaseTransition {
        /// Current phase.
        from: MtpRequestPhase,
        /// Requested phase.
        to: MtpRequestPhase,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terminal_precedence_and_cancellation_are_canonical() {
        let mut sequence = GenerationSequence::new(2, [7]);
        let first = sequence.commit(1, TokenTerminalSignals::default()).unwrap();
        assert_eq!(first.finish_reason, None);
        let second = sequence
            .commit(
                7,
                TokenTerminalSignals {
                    stop_sequence: true,
                    grammar_complete: true,
                },
            )
            .unwrap();
        assert_eq!(second.finish_reason, Some(FinishReason::StopSequence));
        assert!(!sequence.cancel());

        let token = GenerationCancellationToken::new();
        let mut active = GenerationSequence::new(3, []);
        token.cancel();
        assert!(active.observe_cancellation(&token));
        assert_eq!(active.finish_reason(), Some(FinishReason::Cancelled));
    }

    #[test]
    fn speculative_commit_plan_preserves_trailing_token_cache_semantics() {
        let mut rejected = SpeculativeRound::new(3).unwrap();
        rejected.accept(10, false).unwrap();
        rejected.reject_with(99, false).unwrap();
        let plan = rejected.commit_plan().unwrap();
        assert_eq!(plan.accepted_proposals, 1);
        assert_eq!(plan.committed_tokens, &[10, 99]);
        assert_eq!(plan.verified_inputs, 2);
        assert_eq!(plan.tail, Some(SpeculativeTail::Replacement));

        let mut terminal_accept = SpeculativeRound::new(2).unwrap();
        terminal_accept.accept(10, false).unwrap();
        terminal_accept.accept(11, true).unwrap();
        let plan = terminal_accept.commit_plan().unwrap();
        assert!(plan.full_acceptance);
        assert_eq!(plan.verified_inputs, 2);
        assert_eq!(plan.tail, None);

        let mut bonus = SpeculativeRound::new(2).unwrap();
        bonus.accept(10, false).unwrap();
        bonus.accept(11, false).unwrap();
        bonus.bonus(12, false).unwrap();
        assert_eq!(bonus.commit_plan().unwrap().verified_inputs, 3);

        let mut incomplete = SpeculativeRound::new(2).unwrap();
        incomplete.accept(10, false).unwrap();
        assert!(matches!(
            incomplete.commit_plan(),
            Err(GenerationError::IncompleteSpeculativeRound)
        ));
    }

    #[test]
    fn optimistic_reuse_is_pure_and_fail_closed() {
        assert_eq!(
            resolve_optimistic_reuse(&[1], &[1], &[2, 3], 2, false).unwrap(),
            OptimisticReuseDecision::MatchedRetained
        );
        assert_eq!(
            resolve_optimistic_reuse(&[1], &[1], &[2], 2, false).unwrap(),
            OptimisticReuseDecision::MatchedConsumed
        );
        assert_eq!(
            resolve_optimistic_reuse(&[1], &[1], &[2], 9, false).unwrap(),
            OptimisticReuseDecision::DiscardMismatch
        );
        assert!(matches!(
            resolve_optimistic_reuse(&[1], &[9], &[2], 2, false),
            Err(GenerationError::OptimisticPrefixDiverged)
        ));
    }

    #[test]
    fn sampler_and_scheduler_configuration_validate_without_a_backend() {
        let checkpoint = CheckpointGenerationConfig {
            do_sample: Some(true),
            temperature: Some(0.8),
            top_k: Some(64),
            repetition_penalty: Some(1.1),
            ..CheckpointGenerationConfig::default()
        };
        let resolved =
            resolve_generation_config(Some(&checkpoint), GenerationConfigOverrides::default())
                .unwrap();
        assert!(resolved.do_sample);
        assert_eq!(resolved.top_k, 64);
        assert_eq!(resolved.repetition_penalty, 1.1);
        assert_eq!(resolved.repeat_last_n, 64);
        assert!(matches!(
            resolve_generation_config(
                None,
                GenerationConfigOverrides {
                    frequency_penalty: Some(f32::NAN),
                    ..GenerationConfigOverrides::default()
                }
            ),
            Err(GenerationError::InvalidFrequencyPenalty(value)) if value.is_nan()
        ));
        assert!(MtpConfig::default().validate().is_ok());
        assert!(MtpSchedulerOptions::default().validate().is_ok());
        assert!(matches!(
            MtpSchedulerOptions {
                max_in_flight_verifications: 0,
                ..MtpSchedulerOptions::default()
            }
            .validate(),
            Err(GenerationError::ZeroInFlightVerifications)
        ));

        let config_json = serde_json::to_string(&resolved).unwrap();
        assert_eq!(
            serde_json::from_str::<ResolvedGenerationConfig>(&config_json).unwrap(),
            resolved
        );
        let options = MtpSchedulerOptions::default();
        let options_json = serde_json::to_string(&options).unwrap();
        assert_eq!(
            serde_json::from_str::<MtpSchedulerOptions>(&options_json).unwrap(),
            options
        );
    }

    #[test]
    fn semantic_events_round_trip_without_a_backend() {
        let event = SemanticEvent::ToolCallStart {
            index: 2,
            id: "call_2".into(),
            name: "lookup".into(),
        };
        let json = serde_json::to_string(&event).unwrap();
        assert_eq!(serde_json::from_str::<SemanticEvent>(&json).unwrap(), event);

        let mut zero_budget = GenerationSequence::new(0, []);
        assert_eq!(zero_budget.finish_reason(), Some(FinishReason::MaxTokens));
        assert!(zero_budget.cancel());
        assert_eq!(zero_budget.finish_reason(), Some(FinishReason::Cancelled));
    }

    #[test]
    fn speculative_request_lifecycle_defers_cancellation_exactly() {
        let mut lifecycle = MtpRequestLifecycle::new();
        lifecycle.transition(MtpRequestPhase::ReadyToDraft).unwrap();
        lifecycle
            .transition(MtpRequestPhase::ReadyToSubmitVerification)
            .unwrap();
        lifecycle
            .transition(MtpRequestPhase::TargetVerificationInFlight)
            .unwrap();
        assert_eq!(
            lifecycle.request_cancellation(true).unwrap(),
            MtpCancellationDisposition::Deferred
        );
        assert!(lifecycle.cancellation_pending());
        lifecycle
            .transition(MtpRequestPhase::VerificationResolution)
            .unwrap();
        lifecycle.transition(MtpRequestPhase::Cancelled).unwrap();
        assert!(lifecycle.is_terminal());
        assert!(!lifecycle.cancellation_pending());
        assert!(matches!(
            lifecycle.transition(MtpRequestPhase::ReadyToDraft),
            Err(GenerationError::InvalidMtpPhaseTransition { .. })
        ));
    }
}
