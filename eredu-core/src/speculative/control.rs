//! Inspection and exact canonical-boundary snapshots of the shared driver.

use super::*;
use crate::{execution_control::SnapshotEstimate, BackendFailure};

/// Portable controlled-speculation failure. Native causes remain error sources.
#[derive(Debug, thiserror::Error)]
pub enum SpeculativeControlError {
    /// Mode admission against the retained source or logical lifecycle failed.
    #[error(transparent)]
    DecodingMode(#[from] crate::decoding::DecodingModeError),
    /// The selected mechanism cannot implement this operation exactly.
    #[error("controlled speculation is unsupported: {0}")]
    Unsupported(&'static str),
    /// A verification or proposal transaction is still retained.
    #[error("speculative control requires a canonical boundary without pending proposals")]
    NotQuiescent,
    /// An invalidated session cannot be advanced or restored.
    #[error("controlled speculative session has failed or been cancelled")]
    Failed,
    /// A snapshot does not belong to this run.
    #[error("snapshot does not belong to this controlled speculative run")]
    IncompatibleSnapshot,
    /// An inactive branch slot belongs to another scope or has been released.
    #[error("branch does not belong to this controlled speculative session")]
    IncompatibleBranch,
    /// A prospective control request violates its portable policy.
    #[error("invalid speculative control: {0}")]
    Invalid(&'static str),
    /// Forced ID is outside the canonical tokenizer domain.
    #[error("forced token {0} is outside the canonical vocabulary")]
    InvalidToken(u32),
    /// The current grammar forbids this canonical token.
    #[error("forced token {0} conflicts with the active constraints")]
    ForbiddenToken(u32),
    /// An uncommitted choice must be cleared before replacement.
    #[error("a forced token is already pending")]
    PendingToken,
    /// Capture configuration or bounded collection failed.
    #[error(transparent)]
    Capture(#[from] crate::capture::CaptureError),
    /// Bounded state or delivery admission failed.
    #[error(transparent)]
    Control(#[from] crate::execution_control::ExecutionControlError),
    /// Semantic snapshot preparation failed.
    #[error(transparent)]
    Output(#[from] SpeculativeOutputError),
    /// Native execution or driver failure, retaining its original source.
    #[error(transparent)]
    Backend(BackendFailure),
}

impl SpeculativeControlError {
    /// Translates a native implementation failure without exposing its type.
    pub fn backend(error: impl std::error::Error + Send + Sync + 'static) -> Self {
        Self::Backend(BackendFailure::from_error(error))
    }
}

/// Host-only tentative canvas state, distinct from committed output and drafts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RefinementProposalView {
    /// Absolute generated position of the next commitment.
    pub position: usize,
    /// Finalized positions in the active complete canvas.
    pub finalized_positions: usize,
    /// Complete admitted invocation width, even with a shortened output budget.
    pub canvas_positions: usize,
    /// Selected dependent suffix, absent during numerical refinement or after
    /// a constraint invalidates it.
    pub token_ids: Option<Vec<u32>>,
    /// Meaning of the numerical position rows retained by this canvas.
    pub score_kind: crate::refinement::PredictionScoreKind,
}

/// Host-only view of a proposal block. These tokens are never committed output.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpeculativeProposalView {
    /// Absolute generated-token position of the first proposal.
    pub position: usize,
    /// Proposed tokenizer ids in draft order, including unverified suffixes.
    pub token_ids: Vec<u32>,
    /// Full assumed generated prefix for optimistic work, otherwise absent.
    pub assumed_prefix: Option<Vec<u32>>,
}

/// Read-only native storage facts for one independently executable model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeculativeModelMemoryObservation {
    /// Fresh conversions retained by this participant at the observed boundary.
    /// Absent historical attribution cannot reuse startup observations.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameter_conversions: Option<Vec<crate::residency::ResidentParameterConversion>>,
    /// Current canonical cache frontier, observed from the actual state owner.
    pub current_positions: u64,
    /// Current retained state and backing-capacity allowance.
    pub current_state_bytes: Option<u64>,
    /// State allowance across the requested horizon, including interior capacity peaks.
    pub peak_state_bytes: Option<u64>,
    /// Current parameter backing, including cached parameter conversions.
    pub parameters: crate::StaticMemoryReport,
    /// Point-in-time host capacity observation.
    pub available: crate::AvailableMemory,
    /// Current allocator cache policy for this physical allocator.
    pub allocator_cache_limit: crate::Observed<u64>,
}

/// Native prediction-state facts in ordinary local layer order.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeculativePredictionMemoryObservation {
    /// Actual installed frontiers, without assuming they equal the target frontier.
    pub layer_positions: Vec<u64>,
    /// Installed native state and backing-capacity allowance.
    pub current_state_bytes: Option<u64>,
    /// Native capacity envelope through the requested additional positions.
    pub peak_state_bytes: Option<u64>,
}

/// Installed embedded state and separately retained target features.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddedContinuationObservation {
    /// Canonical prediction state, not the separately retained assistant seed.
    pub prediction: SpeculativePredictionMemoryObservation,
    /// Native capture copy/descriptor allowance. A sliced view may retain larger
    /// backing; runtime composition also includes the selected full-prefix feature
    /// envelope. This observation alone is not a distinct resident-byte counter.
    pub retained_feature_bytes: Option<u64>,
}

/// Read-only shared parameter and allocator facts from the native residency owner.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeculativeParameterMemoryObservation {
    /// Shared target and embedded parameter residency, charged once.
    pub parameters: crate::StaticMemoryReport,
    /// Point-in-time available capacity.
    pub available: crate::AvailableMemory,
    /// Current allocator-cache policy.
    pub allocator_cache_limit: crate::Observed<u64>,
    /// Current conversions with authoritative retaining bindings.
    pub parameter_conversions: Option<Vec<crate::residency::ResidentParameterConversion>>,
}

/// Native observations for a settled speculative transaction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeculativeContinuationObservation {
    /// Installed target state and residency.
    pub target: SpeculativeModelMemoryObservation,
    /// Installed independent draft state and residency.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub draft: Option<SpeculativeModelMemoryObservation>,
    /// Installed embedded owner, mutually exclusive with an independent draft.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedded: Option<EmbeddedContinuationObservation>,
    /// Shared parameter conversion owners observed at this settled boundary.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameter_conversions: Option<Vec<crate::residency::ResidentParameterConversion>>,
    /// Durable assistant seed retained alongside the installed caches.
    pub seed_bytes: Option<u64>,
}

/// Complete saved canonical execution and output state; callbacks are not copied.
/// Creation requires explicit durable native checkpoints and known copy costs.
pub struct SpeculativeControlSnapshot<E: SpeculativeExecutor, S: SpeculativeSampling, C> {
    identity: Arc<()>,
    cache: E::CacheCheckpoint,
    target: E::TargetState,
    sampler: S,
    constraint: C,
    sequence: GenerationSequence,
    target_randomness: Option<S::RandomState>,
    draft_randomness: Option<S::DraftRandomness>,
    lifecycle: SpeculativeRequestLifecycle,
    stats: SpeculativeStats,
    config: GenerationRequestConfig,
    pending_committed_tokens: Vec<u32>,
    next_logits: Option<E::Logits>,
    refinement: Option<RefinementContinuation<E::Logits>>,
    decoding: crate::decoding::DecodingState,
}

impl<E: SpeculativeExecutor, S: SpeculativeSampling, C> SpeculativeControlSnapshot<E, S, C> {
    /// Canonical prefix retained by this immutable state.
    pub fn token_ids(&self) -> &[u32] {
        self.sequence.tokens()
    }
    /// Lifecycle retained with the prefix.
    pub fn status(&self) -> SpeculativeRequestStatus {
        self.lifecycle.status()
    }
    /// Logical attention policy retained with the canonical cache and suffix.
    pub const fn decoding_state(&self) -> crate::decoding::DecodingState {
        self.decoding
    }
}

impl<E, S, C, P> SpeculativeRequest<'_, E, S, C, P>
where
    E: SpeculativeExecutor,
    S: SpeculativeSampling<Logits = E::Logits, Error = E::Error>,
    C: SpeculativeConstraint,
    P: SpeculativePublisher<C>,
{
    /// Mutable sampler access for draining shared control observations.
    pub fn sampler_mut(&mut self) -> &mut S {
        self.runtime.sampler_mut()
    }

    /// Shared portable facts for prospective sampling admission.
    pub fn control_sampling_facts(&self) -> Option<(f32, bool, bool)> {
        Some((
            self.config.temperature,
            self.sampler().control_requires_positive_temperature()?,
            self.target_randomness.is_some() && self.draft_randomness.is_some(),
        ))
    }

    /// Installs a previously validated prospective temperature and optional seed.
    pub fn control_override_sampling<'a>(
        &mut self,
        temperature: f32,
        reseed: Option<u64>,
        context: S::Context<'a>,
    ) -> Result<(), SpeculativeControlError>
    where
        S: 'a,
    {
        self.validate_control_edit()?;
        if let Some(seed) = reseed {
            let seed = S::control_seed(seed, context)?;
            // A seed supplied while greedy is retained for a later stochastic switch.
            let randomness = S::initialize_randomness(Some(seed), 1.0, context)
                .map_err(SpeculativeControlError::backend)?;
            self.target_randomness = randomness.target;
            self.draft_randomness = randomness.draft;
        }
        self.config.temperature = temperature;
        Ok(())
    }

    /// Validates a choice before any model work, using the canonical constraint owner.
    pub fn control_force_next(
        &mut self,
        token: u32,
        vocabulary: usize,
    ) -> Result<(), SpeculativeControlError> {
        self.validate_control_edit()?;
        let position = self.sequence().tokens().len();
        self.sampler_mut()
            .control_force_next(token, vocabulary, position)
    }

    /// Observes the partially finalized canvas without reading native values.
    pub fn refinement_proposals(&self) -> Option<RefinementProposalView> {
        self.refinement.as_ref().map(|canvas| RefinementProposalView {
            position: self.sequence().tokens().len(), finalized_positions: canvas.committed.len(),
            canvas_positions: canvas.width,
            token_ids: canvas.positions.as_ref().map(|rows| rows.iter().skip(canvas.committed.len()).map(|(token, _)| *token).collect()),
            score_kind: crate::refinement::PredictionScoreKind::RefinementPosition,
        })
    }

    /// Canonical proposals, including the block retained during verification.
    pub fn proposals(&self) -> Option<SpeculativeProposalView> {
        self.block
            .as_ref()
            .or_else(|| self.pending.as_ref().map(|p| &p.block))
            .map(|block| SpeculativeProposalView {
                position: self.sequence().tokens().len(),
                token_ids: block.proposals.iter().map(|p| p.token).collect(),
                assumed_prefix: None,
            })
    }

    /// Optimistic proposals tied to their exact assumed prefix.
    pub fn optimistic_proposals(&self) -> Option<SpeculativeProposalView> {
        self.pending
            .as_ref()?
            .optimistic
            .as_ref()
            .map(|branch| SpeculativeProposalView {
                position: branch.assumed_prefix.len(),
                token_ids: branch.block.proposals.iter().map(|p| p.token).collect(),
                assumed_prefix: Some(branch.assumed_prefix.clone()),
            })
    }

    /// Whether no submitted causal verification or speculative draft blocks remain.
    /// Settled tentative refinement rows remain snapshotable between commitments.
    pub fn is_control_boundary(&self) -> bool {
        self.pending.is_none() && self.block.is_none() && self.target_state.is_some()
    }

    fn check_control_boundary(&self) -> Result<(), SpeculativeControlError> {
        if self.runtime.cancellation().is_cancelled()
            || self.status() == SpeculativeRequestStatus::Cancelled
        {
            return Err(SpeculativeControlError::Failed);
        }
        if !self.is_control_boundary() {
            return Err(SpeculativeControlError::NotQuiescent);
        }
        Ok(())
    }

    /// Requires a healthy, nonterminal canonical boundary for a prospective edit.
    pub fn validate_control_edit(&self) -> Result<(), SpeculativeControlError> {
        self.check_control_boundary()?;
        if self.lifecycle.is_terminal() {
            return Err(SpeculativeControlError::Invalid("generation is terminal"));
        }
        Ok(())
    }

    /// Identifies the state owner preventing complete snapshot admission.
    pub fn control_snapshot_unavailable_reason(&self, executor: &E) -> Option<&'static str> {
        if self.runtime.cancellation().is_cancelled()
            || self.status() == SpeculativeRequestStatus::Cancelled
        {
            return Some("cancelled sessions cannot be restored");
        }
        if !self.is_control_boundary() {
            return Some("no canonical boundary without retained speculative work");
        }
        let Some(target) = self.target_state.as_ref() else {
            return Some("prefill has not committed");
        };
        if executor
            .control_snapshot_estimate(&self.cache, target)
            .is_none()
        {
            return Some(
                "native cache or assistant seed has no complete isolated snapshot estimate",
            );
        }
        if self
            .runtime
            .sampler()
            .control_snapshot_bytes(
                self.target_randomness.as_ref(),
                self.draft_randomness.as_ref(),
            )
            .is_none()
        {
            return Some(
                "sampler, grammar or RNG state has no complete isolated snapshot estimate",
            );
        }
        if self.runtime.constraint().control_snapshot_bytes().is_none() {
            return Some("semantic decoder or parser has no complete snapshot estimate");
        }
        if self.control_snapshot_estimate(executor).is_none() {
            return Some("complete snapshot size overflowed");
        }
        None
    }

    /// Observes settled speculative state without copying, submitting or polling work.
    pub fn continuation_memory_observation(
        &self,
        executor: &E,
        additional_positions: u64,
    ) -> Result<Option<SpeculativeContinuationObservation>, SpeculativeControlError> {
        self.validate_control_edit()?;
        executor
            .continuation_memory_observation(
                &self.cache,
                self.target_state
                    .as_ref()
                    .expect("checked canonical boundary"),
                additional_positions,
            )
            .map_err(SpeculativeControlError::Backend)
    }

    /// Configured proposal ceiling, independent of adaptive lookahead decisions.
    pub fn forecast_max_draft_tokens(&self) -> u64 {
        self.config.proposal_capacity() as u64
    }

    /// Cache growth required by an admitted handoff, even without further output.
    /// A previously settled score row needs no second append.
    pub fn forecast_mode_settlement_positions(&self) -> u64 {
        if self.decoding.state().pending.is_some() {
            self.refinement.as_ref().map_or(self.pending_committed_tokens.len(), |canvas| canvas.width) as u64
        } else { 0 }
    }

    /// Effective per-request scheduler controls under shared resource ceilings.
    pub fn forecast_scheduler_options(
        &self,
        ceiling: crate::generation::SpeculativeSchedulerOptions,
    ) -> crate::generation::SpeculativeSchedulerOptions {
        self.config.scheduler_options(ceiling)
    }

    /// Active and admitted pending policies over one retained execution.
    pub fn forecast_decoding_policies(&self, ceiling: crate::generation::SpeculativeSchedulerOptions)
        -> Vec<(crate::decoding::DecodingMode, crate::generation::SpeculativeSchedulerOptions)> {
        let state = self.decoding.state();
        std::iter::once(state.active).chain(state.pending.filter(|mode| *mode != state.active))
            .map(|mode| {
                let mut config = self.config.clone();
                config.decoding_mode = mode;
                (mode, config.scheduler_options(ceiling))
            }).collect()
    }

    /// Immutable next-target row retained independently of the source's seed.
    pub fn retained_next_scores_estimate(&self, executor: &E) -> Option<SnapshotEstimate> {
        match &self.next_logits {
            Some(logits) => executor.logits_snapshot_estimate(logits),
            None => Some(SnapshotEstimate { retained_bytes: 0, copy_bytes: 0 }),
        }
    }

    /// Retained host/sampling/semantic state, including both random streams.
    /// This is an allowance, never resident credit.
    pub fn continuation_host_bytes(&self) -> Option<u64> {
        self.runtime
            .sampler()
            .control_snapshot_bytes(
                self.target_randomness.as_ref(),
                self.draft_randomness.as_ref(),
            )?
            .checked_add(self.runtime.constraint().control_snapshot_bytes()?)?
            .checked_add(self.sequence().snapshot_storage_bytes()?)?
            .checked_add((self.config.eos_token_ids.capacity() as u64).checked_mul(4)?)?
            .checked_add((self.pending_committed_tokens.capacity() as u64).checked_mul(4)?)?
            .checked_add(
                (self.stats.accept_lens().len() as u64)
                    .checked_mul(std::mem::size_of::<usize>() as u64)?,
            )?
            .checked_add(std::mem::size_of::<Self>() as u64)
    }

    /// Total future host retention, preserving unknown custom sampling/semantic costs.
    pub fn continuation_host_peak_bytes(&self, additional: u64) -> Option<u64> {
        self.runtime
            .sampler()
            .continuation_storage_bytes(
                self.target_randomness.as_ref(),
                self.draft_randomness.as_ref(),
                additional,
            )?
            .checked_add(
                self.runtime
                    .constraint()
                    .continuation_storage_bytes(additional)?,
            )?
            .checked_add(self.sequence().snapshot_storage_bytes()?)?
            .checked_add(additional.checked_mul(32)?)?
            .checked_add((self.config.eos_token_ids.capacity() as u64).checked_mul(4)?)?
            .checked_add((self.pending_committed_tokens.capacity() as u64).checked_mul(4)?)?
            .checked_add((self.stats.accept_lens().len() as u64).checked_mul(16)?)?
            .checked_add(std::mem::size_of::<Self>() as u64)
    }

    /// Complete conservative retention and copy bound for tentative position rows
    /// and their partially finalized commitment cursor.
    pub fn retained_refinement_estimate(&self, executor: &E) -> Option<SnapshotEstimate> {
        let Some(canvas) = &self.refinement else { return Some(SnapshotEstimate { retained_bytes: 0, copy_bytes: 0 }); };
        let host = (std::mem::size_of::<RefinementContinuation<E::Logits>>() as u64)
            .checked_add((canvas.committed.len() as u64).checked_mul(4)?)?
            .checked_add((canvas.positions.as_ref().map_or(0, Vec::len) as u64)
                .checked_mul(std::mem::size_of::<(u32, E::Logits)>() as u64)?)?;
        let mut estimate = SnapshotEstimate { retained_bytes: host, copy_bytes: host };
        for (_, logits) in canvas.positions.iter().flatten() {
            let row = executor.logits_snapshot_estimate(logits)?;
            estimate.retained_bytes = estimate.retained_bytes.checked_add(row.retained_bytes)?;
            estimate.copy_bytes = estimate.copy_bytes.checked_add(row.copy_bytes)?;
        }
        Some(estimate)
    }

    /// Complete conservative copy bound; unknown components fail closed.
    pub fn control_snapshot_estimate(&self, executor: &E) -> Option<SnapshotEstimate> {
        if !self.is_control_boundary() {
            return None;
        }
        let native = executor.control_snapshot_estimate(&self.cache, self.target_state.as_ref()?)?;
        let host = self
            .runtime
            .sampler()
            .control_snapshot_bytes(
                self.target_randomness.as_ref(),
                self.draft_randomness.as_ref(),
            )?
            .checked_add(self.runtime.constraint().control_snapshot_bytes()?)?
            .checked_add(self.sequence().snapshot_storage_bytes()?)?
            .checked_add((self.config.eos_token_ids.len() as u64).checked_mul(4)?)?
            .checked_add((self.pending_committed_tokens.len() as u64).checked_mul(4)?)?
            .checked_add(
                (self.stats.accept_lens().len() as u64)
                    .checked_mul(std::mem::size_of::<usize>() as u64)?,
            )?
            .checked_add(std::mem::size_of::<SpeculativeControlSnapshot<E, S, C>>() as u64)?;
        let next = match self.next_logits.as_ref() {
            Some(logits) => executor.logits_snapshot_estimate(logits)?,
            None => SnapshotEstimate { retained_bytes: 0, copy_bytes: 0 },
        };
        let refinement = self.retained_refinement_estimate(executor)?;
        Some(SnapshotEstimate {
            retained_bytes: native.retained_bytes.checked_add(host)?.checked_add(next.retained_bytes)?.checked_add(refinement.retained_bytes)?,
            copy_bytes: native.copy_bytes.checked_add(host)?.checked_add(next.copy_bytes)?.checked_add(refinement.copy_bytes)?,
        })
    }

    /// Copies complete state after the caller reserves the declared bound.
    pub fn control_snapshot<'a>(
        &self,
        executor: &E,
        context: E::Context<'a>,
    ) -> Result<SpeculativeControlSnapshot<E, S, C>, SpeculativeControlError> {
        self.check_control_boundary()?;
        let constraint = self.runtime.constraint().fork()?;
        let (cache, target) = executor
            .control_snapshot(
                &self.cache,
                self.target_state.as_ref().expect("checked boundary"),
                context,
            )?
            .ok_or(SpeculativeControlError::Unsupported(
                "executor has no durable speculative checkpoint",
            ))?;
        Ok(SpeculativeControlSnapshot {
            identity: Arc::clone(&self.control_identity),
            cache,
            target,
            constraint,
            sampler: self.runtime.sampler().clone(),
            sequence: self.sequence().clone(),
            target_randomness: self.target_randomness.clone(),
            draft_randomness: self.draft_randomness.clone(),
            lifecycle: self.lifecycle.clone(),
            stats: self.stats.clone(),
            config: self.config.clone(),
            pending_committed_tokens: self.pending_committed_tokens.clone(),
            next_logits: self.next_logits.clone(),
            refinement: self.refinement.clone(),
            decoding: self.decoding.state(),
        })
    }

    /// Command identity of the active logical continuation.
    pub fn decoding_owner(&self) -> crate::decoding::GenerationCommandOwner {
        crate::decoding::GenerationCommandOwner {
            identity: Arc::clone(&self.control_identity), control: self.decoding.clone(),
        }
    }

    /// Gives an inactive fork an independent mailbox initialized from its snapshot.
    pub fn fork_decoding_owner(&self, snapshot: &SpeculativeControlSnapshot<E, S, C>)
        -> Result<crate::decoding::GenerationCommandOwner, SpeculativeControlError> {
        if !Arc::ptr_eq(&self.control_identity, &snapshot.identity) {
            return Err(SpeculativeControlError::IncompatibleSnapshot);
        }
        Ok(crate::decoding::GenerationCommandOwner {
            identity: Arc::clone(&self.control_identity), control: self.decoding.fork(snapshot.decoding),
        })
    }

    /// Restores snapshot commands along with matching native and output state.
    pub fn restore_control_snapshot<'a>(
        &mut self, executor: &mut E, snapshot: &SpeculativeControlSnapshot<E, S, C>, context: E::Context<'a>,
    ) -> Result<(), SpeculativeControlError> where E: 'a {
        self.install_snapshot_state(executor, snapshot, context)?;
        self.decoding.restore(snapshot.decoding);
        Ok(())
    }

    /// Exchanges active branch ownership while retaining its independently queued commands.
    pub fn activate_control_snapshot<'a>(&mut self, executor: &mut E,
        snapshot: &SpeculativeControlSnapshot<E, S, C>, owner: &crate::decoding::GenerationCommandOwner,
        context: E::Context<'a>,
    ) -> Result<(), SpeculativeControlError> where E: 'a {
        if !Arc::ptr_eq(&self.control_identity, &owner.identity) || owner.state().active != snapshot.decoding.active {
            return Err(SpeculativeControlError::IncompatibleBranch);
        }
        self.install_snapshot_state(executor, snapshot, context)?;
        self.decoding = owner.control.clone();
        self.runtime.decoding = Some(self.decoding.clone());
        Ok(())
    }

    /// Native replacement is atomic; a failed owner cannot resume partial state.
    fn install_snapshot_state<'a>(
        &mut self,
        executor: &mut E,
        snapshot: &SpeculativeControlSnapshot<E, S, C>,
        context: E::Context<'a>,
    ) -> Result<(), SpeculativeControlError>
    where
        E: 'a,
    {
        self.check_control_boundary()?;
        if !Arc::ptr_eq(&self.control_identity, &snapshot.identity) {
            return Err(SpeculativeControlError::IncompatibleSnapshot);
        }
        let constraint = snapshot.constraint.fork()?;
        let sampler = snapshot.sampler.clone();
        let sequence = snapshot.sequence.clone();
        let target_randomness = snapshot.target_randomness.clone();
        let draft_randomness = snapshot.draft_randomness.clone();
        let target = executor
            .restore_control_snapshot(&mut self.cache, &snapshot.cache, &snapshot.target, context)?
            .ok_or(SpeculativeControlError::Unsupported(
                "executor has no durable speculative restore",
            ))?;
        self.runtime
            .install_committed_state(sampler, constraint, sequence);
        self.target_state = Some(target);
        self.target_randomness = target_randomness;
        self.draft_randomness = draft_randomness;
        self.lifecycle = snapshot.lifecycle.clone();
        self.stats = snapshot.stats.clone();
        self.config = snapshot.config.clone();
        self.pending_committed_tokens.clone_from(&snapshot.pending_committed_tokens);
        self.next_logits = snapshot.next_logits.clone();
        self.refinement = snapshot.refinement.clone();
        Ok(())
    }
}

impl<'cache, E, S, C, P> GenerationRequestTable<'cache, E, S, C, P>
where
    E: SpeculativeExecutor,
    S: SpeculativeSampling<Logits = E::Logits, Error = E::Error>,
    C: SpeculativeConstraint,
    P: SpeculativePublisher<C>,
{
    /// Mutable canonical request for controlled state restoration.
    pub fn request_mut(
        &mut self,
        id: SpeculativeRequestId,
    ) -> Option<&mut SpeculativeRequest<'cache, E, S, C, P>> {
        self.requests.get_mut(id.index())
    }
}

/// Raw prediction capture from sampling input, before filtering or random draws.
/// Target rows can belong to an uncommitted verification; use the step's
/// verification dispositions to determine which predictions became canonical.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeculativePredictionCapture {
    /// Target verifier or draft model.
    pub role: SpeculativeCaptureRole,
    /// Absolute generated-token prediction position (zero is target prefill).
    pub position: u64,
    /// Invocation semantics distinguish position scores from causal conditionals.
    #[serde(default)]
    pub score_kind: crate::refinement::PredictionScoreKind,
    /// Existing admitted bounded host capture records.
    pub capture: crate::capture::CapturedStep,
}

/// Model responsible for one raw prediction observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpeculativeCaptureRole {
    /// Ordinary target prefill or verification row.
    Target,
    /// Canonical or optimistic draft proposal.
    Draft,
}

/// Actual forward computation producing an internal speculative activation.
/// These phases are distinct from committed prediction positions and from
/// sequence-row coordinates within a captured tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "phase", rename_all = "snake_case")]
pub enum SpeculativeActivationPhase {
    /// Initial ordinary-target prompt evaluation.
    TargetPrefill,
    /// Prediction-local state seeded from target prompt hidden values.
    PredictionPrefill,
    /// One tentative sequential prediction invocation.
    Proposal {
        /// Zero-based depth within the current proposal block.
        depth: usize,
    },
    /// A tentative fused prediction block.
    FusedProposal,
    /// One adapter-disabled complete-canvas refinement forward.
    Refinement {
        /// Zero-based numerical refinement iteration.
        iteration: u32,
    },
    /// One draft-only complete-canvas refinement forward over the shared target.
    MaskedDraft {
        /// Zero-based numerical refinement iteration.
        iteration: u32,
    },
    /// Ordinary-target evaluation of a tentative proposal block.
    Verification,
    /// Prediction-local replay of retained verified inputs.
    PredictionReplay,
    /// Ordinary-target replay after rejecting a suffix of a verification block.
    TargetReplay,
}

/// Scheduler provenance for one speculative executor operation. Tensor rows
/// remain separate physical coordinates; this is not proof of token commitment.
/// A run's delivery identity distinguishes restores and branches with the same
/// prefix. No token history is retained in this fixed-size record.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpeculativeActivationOrigin {
    /// Stable request within the owning scheduler.
    pub request: SpeculativeRequestId,
    /// Number of canonical generated tokens before this operation.
    pub committed_tokens: usize,
    /// Generated-token schedule coordinate: the next proposal position, the
    /// start of a verification/replay block, or zero for initial prefill. It
    /// does not identify every physical row of an internal activation.
    pub prediction: usize,
    /// SHA-256 of the exact generated prefix assumed by this operation, using
    /// the domain `eredu.speculative.activation-prefix.v1` and little-endian u32
    /// token IDs. The prefix length is `prediction`.
    pub prefix_digest: [u8; 32],
    /// Whether this work assumed a still-unverified previous proposal block.
    /// Canonical proposals are also tentative, even when this is false.
    pub optimistic: bool,
}

/// Bounded internal evidence from an actual speculative forward. A successful
/// forward is still tentative until the generation driver commits its tokens.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeculativeActivationCapture {
    /// Exact loaded activation-plan identity when installed through immutable
    /// public admission. Low-level observer fixtures may have no such authority.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub admission_identity: Option<String>,
    /// Monotonic collector invocation identity; restore never rewinds it.
    pub invocation: u64,
    /// Scheduler request and assumed generated prefix.
    pub origin: SpeculativeActivationOrigin,
    /// Architecture computation that produced this evidence.
    pub phase: SpeculativeActivationPhase,
    /// Whether this forward and its evidence finished successfully. False
    /// retains failure evidence without presenting it as committed output.
    pub completed: bool,
    /// Existing value/effective-value records, exact physical shape and shared
    /// cumulative resource usage. Its transaction outcome describes the forward
    /// only; even `Committed` does not imply speculative token acceptance.
    pub captures: crate::capture::CapturedStep,
}

impl SpeculativeActivationOrigin {
    pub(super) fn new(request: SpeculativeRequestId, committed: &[u32], optimistic: bool) -> Self {
        Self {
            request,
            committed_tokens: committed.len(),
            prediction: committed.len(),
            prefix_digest: Self::digest(committed),
            optimistic,
        }
    }

    pub(super) fn with_prefix(self, prefix: &[u32]) -> Self {
        Self {
            prediction: prefix.len(),
            prefix_digest: Self::digest(prefix),
            ..self
        }
    }

    fn digest(prefix: &[u32]) -> [u8; 32] {
        use sha2::{Digest, Sha256};
        let mut digest = Sha256::new();
        digest.update(b"eredu.speculative.activation-prefix.v1");
        for token in prefix {
            digest.update(token.to_le_bytes());
        }
        digest.finalize().into()
    }
}

/// Bounds an operation's provenance lifetime, including backend errors and
/// unwinding. Completion ownership remains with the ordinary submission driver.
pub(super) fn with_activation_origin<E: SpeculativeExecutor, R>(
    executor: &mut E,
    origin: Option<SpeculativeActivationOrigin>,
    operation: impl FnOnce(&mut E) -> R,
) -> R {
    struct Guard<'a, E: SpeculativeExecutor>(&'a mut E);
    impl<E: SpeculativeExecutor> Drop for Guard<'_, E> {
        fn drop(&mut self) {
            self.0.set_activation_origin(None);
        }
    }
    let guard = Guard(executor);
    guard.0.set_activation_origin(origin);
    operation(guard.0)
}

/// Prospective tensor edits admitted for one model's speculative prediction rows.
#[derive(Debug, Clone)]
pub struct SpeculativeInterventionPlan {
    /// Target or draft logits; roles never share an implicit edit.
    pub role: SpeculativeCaptureRole,
    /// Existing bounded intervention contract, including evidence and scheduling.
    pub plan: crate::intervention::AdmittedInterventionPlan,
}
