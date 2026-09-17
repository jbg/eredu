//! Inspection and exact canonical-boundary snapshots of the shared driver.

use super::*;
use crate::{BackendFailure, execution_control::SnapshotEstimate};

mod capture_delivery;
mod reductions_delivery;
pub use reductions_delivery::{
    PreparedSpeculativePrefillReductions, SharedSpeculativePrefillReductions,
    SpeculativePrefillReductionsDelivery,
};
mod snapshot_metadata;

/// Portable controlled-speculation failure. Native causes remain error sources.
#[derive(Debug, thiserror::Error)]
pub enum SpeculativeControlError {
    /// The selected mechanism cannot implement this operation exactly.
    #[error("controlled speculation is unsupported: {0}")]
    Unsupported(&'static str),
    /// A verification or proposal transaction is still retained.
    #[error("speculative snapshots require a canonical boundary without pending proposals")]
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

    /// Gives the actual mechanism one opportunity to transfer its retained source.
    /// Success moves that owner directly, without formatting or allocating. A
    /// refused transfer preserves ordinary conversion of the unchanged error.
    pub fn backend_with_retained<E: std::error::Error + Send + Sync + 'static>(
        error: E,
        take: impl FnOnce(E) -> Result<BackendFailure, E>,
    ) -> Self {
        match take(error) {
            Ok(retained) => Self::Backend(retained),
            Err(error) => Self::backend(error),
        }
    }

    /// Transfers only an explicitly retained backend branch. All other branches,
    /// and backend errors refused by the hook, retain the original driver wrapper
    /// and ordinary source chain. No controller or cancellation branch is flattened.
    pub fn driver_with_retained<E: std::error::Error + Send + Sync + 'static>(
        error: SpeculativeDriverError<E>,
        take: impl FnOnce(E) -> Result<BackendFailure, E>,
    ) -> Self {
        match error {
            SpeculativeDriverError::Backend(error) => match take(error) {
                Ok(retained) => Self::Backend(retained),
                Err(error) => Self::backend(SpeculativeDriverError::Backend(error)),
            },
            error => Self::backend(error),
        }
    }
}

/// Host-only view of a proposal block. These tokens are never committed output.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpeculativeProposalView {
    /// Absolute generated-token position of the first proposal.
    pub position: usize,
    /// Proposed tokenizer ids in draft order, including unverified suffixes.
    pub token_ids: SpeculativeValues<u32>,
    /// Full assumed generated prefix for optimistic work, otherwise absent.
    pub assumed_prefix: Option<SpeculativeValues<u32>>,
}

struct ProposalSource<'a,D> {
    position:usize,
    proposals:&'a [SpeculativeProposal<D>],
    assumed_prefix:Option<&'a [u32]>,
}
impl<D> ProposalSource<'_,D> {
    fn ordinary(self)->SpeculativeProposalView {
        SpeculativeProposalView {
            position:self.position,
            token_ids:self.proposals.iter().map(|p|p.token).collect::<Vec<_>>().into(),
            assumed_prefix:self.assumed_prefix.map(|p|p.to_vec().into()),
        }
    }
    fn with_metadata<E: SpeculativeExecutor>(
        self,
        executor: &E,
        context: E::Context<'_>,
    ) -> Result<SpeculativeProposalView, SpeculativeDriverError<E::Error>> {
        let token_ids = SpeculativeValues::collect_with_metadata(
            self.proposals.iter().map(|p| p.token),
            executor,
            context,
            std::mem::size_of::<SpeculativeProposalView>() + std::mem::size_of::<Self>(),
        )?;
        let assumed_prefix = self
            .assumed_prefix
            .map(|p| {
                SpeculativeValues::collect_with_metadata(p.iter().copied(), executor, context, 0)
            })
            .transpose()?;
        Ok(SpeculativeProposalView {
            position: self.position,
            token_ids,
            assumed_prefix,
        })
    }
}

/// Complete saved canonical execution and output state; callbacks are not copied.
/// Creation requires explicit durable native checkpoints and known copy costs.
pub struct SpeculativeControlSnapshot<E: SpeculativeExecutor, S: SpeculativeSampling, C> {
    identity: SpeculativeRequestIdentity,
    cache: E::CacheCheckpoint,
    target: E::TargetState,
    sampler: S,
    constraint: C,
    sequence: SpeculativeSequence,
    target_randomness: Option<S::RandomState>,
    draft_randomness: Option<S::DraftRandomness>,
    lifecycle: SpeculativeRequestLifecycle,
    stats: SpeculativeStats,
    config: RequestConfiguration,
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
            let randomness =
                S::initialize_randomness(Some(seed), 1.0, context).map_err(|error| {
                    SpeculativeControlError::backend_with_retained(error, S::take_retained_failure)
                })?;
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

    fn proposal_source(&self,optimistic:bool)->Option<ProposalSource<'_,S::Distribution>> {
        if optimistic {
            let branch = self.pending.as_ref()?.optimistic.as_ref()?;
            Some(ProposalSource {
                position: branch.assumed_prefix.len(),
                proposals: &branch.block.proposals,
                assumed_prefix: Some(&branch.assumed_prefix),
            })
        } else {
            let block = self
                .block
                .as_ref()
                .or_else(|| self.pending.as_ref().map(|p| &p.block))?;
            Some(ProposalSource {
                position: self.sequence().tokens().len(),
                proposals: &block.proposals,
                assumed_prefix: None,
            })
        }
    }
    /// Canonical proposals, including the block retained during verification.
    /// Context-free inspection uses ordinary caller-owned storage.
    pub fn proposals(&self)->Option<SpeculativeProposalView> {
        self.proposal_source(false).map(ProposalSource::ordinary)
    }
    /// Optimistic proposals tied to their exact assumed prefix.
    pub fn optimistic_proposals(&self)->Option<SpeculativeProposalView> {
        self.proposal_source(true).map(ProposalSource::ordinary)
    }
    /// The same canonical view through the actual driver metadata destination.
    pub fn proposals_with_metadata(
        &self,
        executor: &E,
        context: E::Context<'_>,
    ) -> Result<Option<SpeculativeProposalView>, SpeculativeDriverError<E::Error>> {
        self.proposal_source(false)
            .map(|p| p.with_metadata(executor, context))
            .transpose()
    }
    /// The same optimistic view through the actual driver metadata destination.
    pub fn optimistic_proposals_with_metadata(
        &self,
        executor: &E,
        context: E::Context<'_>,
    ) -> Result<Option<SpeculativeProposalView>, SpeculativeDriverError<E::Error>> {
        self.proposal_source(true)
            .map(|p| p.with_metadata(executor, context))
            .transpose()
    }

    /// Whether no tentative work remains at this canonical boundary.
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
            .control_snapshot_estimate(self.cache, target)
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

    /// Complete conservative copy bound; unknown components fail closed.
    pub fn control_snapshot_estimate(&self, executor: &E) -> Option<SnapshotEstimate> {
        if !self.is_control_boundary() {
            return None;
        }
        let native = executor.control_snapshot_estimate(self.cache, self.target_state.as_ref()?)?;
        let host = self
            .runtime
            .sampler()
            .control_snapshot_bytes(
                self.target_randomness.as_ref(),
                self.draft_randomness.as_ref(),
            )?
            .checked_add(self.runtime.constraint().control_snapshot_bytes()?)?
            .checked_add(executor.sequence_copy_bytes(self.sequence())?)?
            .checked_add((self.config.eos_token_ids.len() as u64).checked_mul(4)?)?
            .checked_add(self.stats.copy_storage_bytes(executor)?)?
            .checked_add(std::mem::size_of::<SpeculativeControlSnapshot<E, S, C>>() as u64)?;
        Some(SnapshotEstimate {
            retained_bytes: native.retained_bytes.checked_add(host)?,
            copy_bytes: native.copy_bytes.checked_add(host)?,
        })
    }

    /// Copies complete state after the caller reserves the declared bound.
    pub fn control_snapshot<'a>(
        &self,
        executor: &E,
        context: E::Context<'a>,
    ) -> Result<SpeculativeControlSnapshot<E, S, C>, SpeculativeControlError> {
        self.check_control_boundary()?;
        let sampler_host = snapshot_metadata::sampler(
            self.runtime.sampler(),
            self.target_randomness.as_ref(),
            self.draft_randomness.as_ref(),
            executor,
            context,
        )?;
        let constraint_host =
            snapshot_metadata::constraint(self.runtime.constraint(), executor, context)?;
        let constraint = self
            .runtime
            .constraint()
            .fork_control_snapshot(constraint_host)?;
        let sequence = executor
            .copy_sequence(self.sequence().into(), context)
            .map_err(|cause| {
                SpeculativeControlError::driver_with_retained(cause, E::take_retained_failure)
            })?;
        let stats = self
            .stats
            .copy_for_driver(executor, false, context)
            .map_err(|cause| {
                SpeculativeControlError::driver_with_retained(cause, E::take_retained_failure)
            })?;
        let (cache, target) = executor
            .control_snapshot(
                self.cache,
                self.target_state.as_ref().expect("checked boundary"),
                context,
            )?
            .ok_or(SpeculativeControlError::Unsupported(
                "executor has no durable speculative checkpoint",
            ))?;
        let (sampler, target_randomness, draft_randomness) =
            self.runtime.sampler().copy_control_snapshot(
                self.target_randomness.as_ref(),
                self.draft_randomness.as_ref(),
                sampler_host,
            )?;
        Ok(SpeculativeControlSnapshot {
            identity: self.control_identity.clone(),
            cache,
            target,
            constraint,
            sampler,
            sequence,
            target_randomness,
            draft_randomness,
            lifecycle: self.lifecycle.clone(),
            stats,
            config: self.config.clone(),
        })
    }

    /// Restores at a canonical boundary. Native replacement must be atomic;
    /// failure invalidates the caller's session instead of resuming partial state.
    pub fn restore_control_snapshot<'a>(
        &mut self,
        executor: &mut E,
        snapshot: &SpeculativeControlSnapshot<E, S, C>,
        context: E::Context<'a>,
    ) -> Result<(), SpeculativeControlError>
    where
        E: 'a,
    {
        self.check_control_boundary()?;
        if !self.control_identity.same(&snapshot.identity) {
            return Err(SpeculativeControlError::IncompatibleSnapshot);
        }
        executor.prepare_control_continuation(
            snapshot.sequence.tokens().len(),
            snapshot.status(),
            context,
        )?;
        let sampler_host = snapshot_metadata::sampler(
            &snapshot.sampler,
            snapshot.target_randomness.as_ref(),
            snapshot.draft_randomness.as_ref(),
            executor,
            context,
        )?;
        let constraint_host =
            snapshot_metadata::constraint(&snapshot.constraint, executor, context)?;
        let constraint = snapshot.constraint.fork_control_snapshot(constraint_host)?;
        let (sampler, target_randomness, draft_randomness) =
            snapshot.sampler.copy_control_snapshot(
                snapshot.target_randomness.as_ref(),
                snapshot.draft_randomness.as_ref(),
                sampler_host,
            )?;
        let sequence = executor
            .copy_sequence((&snapshot.sequence).into(), context)
            .map_err(|cause| {
                SpeculativeControlError::driver_with_retained(cause, E::take_retained_failure)
            })?;
        let stats = snapshot
            .stats
            .copy_for_driver(executor, false, context)
            .map_err(|cause| {
                SpeculativeControlError::driver_with_retained(cause, E::take_retained_failure)
            })?;
        let target = executor
            .restore_control_snapshot(self.cache, &snapshot.cache, &snapshot.target, context)?
            .ok_or(SpeculativeControlError::Unsupported(
                "executor has no durable speculative restore",
            ))?;
        self.runtime
            .install_committed_state(sampler, constraint, sequence);
        self.target_state = Some(target);
        self.target_randomness = target_randomness;
        self.draft_randomness = draft_randomness;
        self.lifecycle = snapshot.lifecycle.clone();
        self.stats = stats;
        self.config = snapshot.config.clone();
        Ok(())
    }
}

impl<'cache, E, S, C, P> SpeculativeRequestTable<'cache, E, S, C, P>
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
#[derive(Debug, Serialize, Deserialize)]
pub struct SpeculativePredictionCapture {
    /// Target verifier or draft model.
    pub role: SpeculativeCaptureRole,
    /// Absolute generated-token prediction position (zero is target prefill).
    pub position: u64,
    /// Existing capture records with their actual legacy or shared ownership.
    /// Shared frames retain their source account through escaped records and
    /// snapshot aliases; this carrier establishes no new completion or budget.
    #[serde(with = "capture_delivery")]
    pub capture: crate::capture::CapturedStepDelivery,
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

/// Physical source coordinates of one initial-prefill activation invocation.
/// These fields describe completed/attempted work, never generated commitment
/// or allocation authority. Hidden/token starts are prompt-relative; position
/// is the actual target-cache start of the enclosing span.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpeculativePrefillSpan {
    /// Complete original input row count, independent of the current chunk.
    pub prompt_tokens: u64,
    /// Inclusive original prompt row of the enclosing target span.
    pub input_start: u64,
    /// Exclusive original prompt row of the enclosing target span.
    pub input_end: u64,
    /// Actual target cache position before this span.
    pub position: u64,
    /// Original prompt coordinate of the first supplied hidden row.
    pub hidden_start: u64,
    /// Original prompt coordinate of the first supplied token row.
    pub token_start: u64,
    /// Actual physical row count for this phase.
    pub sequence: u64,
    /// Actual auxiliary cache frontier before seed execution.
    pub seed_start: u64,
}
impl SpeculativePrefillSpan {
    /// Checks phase-specific row alignment without reading tensor contents.
    pub fn validate(self, phase: SpeculativeActivationPhase, sequence: usize) -> bool {
        let Some(width) = self.input_end.checked_sub(self.input_start) else {
            return false;
        };
        if self.input_end > self.prompt_tokens
            || width == 0
            || self.sequence == 0
            || u64::try_from(sequence).ok() != Some(self.sequence)
            || self.position.checked_add(width).is_none()
        {
            return false;
        }
        match phase {
            SpeculativeActivationPhase::TargetPrefill => {
                self.sequence == width
                    && self.hidden_start == self.input_start
                    && self.token_start == self.input_start
            }
            SpeculativeActivationPhase::PredictionPrefill => {
                self.hidden_start
                    .checked_add(self.sequence)
                    .is_some_and(|end| end <= self.input_end)
                    && self.token_start.checked_add(self.sequence) == Some(self.input_end)
                    && (self.token_start == self.hidden_start
                        || self.hidden_start.checked_add(1) == Some(self.token_start))
            }
            _ => false,
        }
    }
}

/// Bounded internal evidence from an actual speculative forward. A successful
/// forward is still tentative until the generation driver commits its tokens.
#[derive(Debug, Serialize, Deserialize)]
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
    /// Exact physical span for chunked initial prefill. Legacy whole invocations
    /// omit it; generated-prefix origin retains its independent meaning.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefill_span: Option<SpeculativePrefillSpan>,
    /// Whether this forward and its evidence finished successfully. False
    /// retains failure evidence without presenting it as committed output.
    pub completed: bool,
    /// Existing value/effective-value records, exact physical shape and shared
    /// cumulative resource usage. Its transaction outcome describes the forward
    /// only; even `Committed` does not imply speculative token acceptance.
    /// Shared frames preserve their original source and host custody. Access the
    /// immutable record with CapturedStepDelivery::as_step.
    #[serde(with = "capture_delivery")]
    pub captures: crate::capture::CapturedStepDelivery,
    /// Logical aggregates (statistics or Preview) across a split prefill, carried by its
    /// last real physical envelope. Physical invocation attribution is unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefill_reductions: Option<SpeculativePrefillReductionsDelivery>,
}

/// Selected architecture geometry, not completion or allocation authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpeculativePrefillReductionGeometry {
    /// Original target prompt rows.
    pub target_sequence: u64,
    /// Actual selected auxiliary seed rows (possibly zero).
    pub prediction_sequence: u64,
}

/// Logical aggregate evidence kept distinct from each physical record.
/// The historical reduction name also includes a selected global Preview prefix.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeculativePrefillReduction {
    /// Named target or prediction prefill computation.
    pub phase: SpeculativeActivationPhase,
    /// Original admission ordinal, not a replacement selector.
    pub selection_index: usize,
    /// Expected full logical sequence extent.
    pub logical_sequence: u64,
    /// Successfully covered physical rows, including no-overlap windows.
    pub covered_sequence: u64,
    /// First successful physical contributor, if any.
    pub first_invocation: Option<u64>,
    /// Last successful physical contributor, if any; IDs may interleave phases.
    pub last_invocation: Option<u64>,
    /// Number of successfully observed physical windows.
    pub windows: u64,
    /// Global disposition, independent of earlier successful physical windows.
    pub status: SpeculativePrefillReductionStatus,
    /// Global source/selected geometry and aggregate. This never substitutes
    /// for the physical tensor shape in the enclosing invocation record.
    pub record: crate::capture::CaptureRecord,
}

/// A logical aggregate is provisional until the whole prefill has settled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpeculativePrefillReductionStatus {
    /// Internal provisional state, never a successfully completed global result.
    Pending,
    /// Complete coverage and successful enclosing prefill.
    Complete,
    /// Enclosing prefill was cancelled or failed after this lane's work.
    Aborted,
    /// The logical selection was explicitly skipped, including a quota limit.
    Skipped,
    /// A required hook or its checked aggregation failed.
    Failed,
}

/// One admitted intervention aggregated across physical prefill windows.
/// Values remain bounded Preview/Summary evidence; no full activation is retained.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeculativePrefillInterventionReduction {
    /// Actual target or shifted prediction-seed computation.
    pub phase: SpeculativeActivationPhase,
    /// Original intervention admission ordinal.
    pub operation_index: usize,
    /// Complete logical source extent for this phase.
    pub logical_sequence: u64,
    /// Successfully committed contiguous physical rows.
    pub covered_sequence: u64,
    /// First successful physical contributor, if any.
    pub first_invocation: Option<u64>,
    /// Last successful physical contributor, if any.
    pub last_invocation: Option<u64>,
    /// Number of committed physical windows.
    pub windows: u64,
    /// Whole-operation disposition, independent of physical prefix success.
    pub status: SpeculativePrefillReductionStatus,
    /// Original operation identity, aggregate evidence and sparse progress.
    pub record: crate::intervention::InterventionRecord,
}

/// One fresh prefill operation; restore cannot rewind its invocation identity.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeculativePrefillReductions {
    /// First accepted physical invocation of this operation.
    pub logical_invocation: u64,
    /// Same exact scheduler origin as every contributing physical invocation.
    pub origin: SpeculativeActivationOrigin,
    /// Original selected target and seed rows; not native-resource counts.
    pub records: Vec<SpeculativePrefillReduction>,
    /// Separately attributed intervention evidence and routing progress.
    /// Existing capture selection ordinals retain their original meaning.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub interventions: Vec<SpeculativePrefillInterventionReduction>,
    /// Additional nonrefundable host/encoding reservations, excluding the
    /// physical transformation charges already recorded on each invocation.
    pub charged: crate::capture::CaptureUsage,
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

#[cfg(test)]
mod retained_error_tests;
