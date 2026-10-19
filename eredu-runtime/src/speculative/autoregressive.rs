//! Ordinary causal execution and speculation between independent language models.
//!
//! Models retain their ordinary equations, residency and state geometry. This
//! mechanism owns only proposal isolation, verification and accepted-prefix replay.
use super::{causal_verification, CausalVerification};
use eredu_core::{
    execution_control::SnapshotEstimate, BoundedCompletion, SpeculativeCommit, SpeculativeExecutor,
    SpeculativePrefill, Submission,
};

mod refinement;
pub use refinement::{
    NoTargetRefinement, SharedTargetRefinement, SharedTargetRefinementState, TargetRefinement,
};
mod causal;
pub use causal::{CausalExecutor, CausalSeed};

/// Attribution of a model invocation within an independent-draft transaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AutoregressivePass {
    /// Canonical target prompt.
    TargetPrefill,
    /// Canonical draft prompt.
    DraftPrefill,
    /// Tentative draft advancement.
    Proposal,
    /// Tentative target verification.
    Verification,
    /// Accepted target prefix replay following a rejection.
    TargetCommit,
    /// Canonical draft advancement to the accepted target frontier.
    DraftCommit,
}

/// Ordinary model and state operations required by independent-model drafting.
/// Checkpoints are immutable and cloning one must preserve branch isolation.
/// Methods retaining native work must retain all inputs through exact completion.
pub trait AutoregressiveMechanisms {
    /// Common asynchronous scheduling facts for the retained target topology.
    fn coordinate_speculative_step(
        local: Vec<eredu_core::SpeculativeScheduleState>,
        _context: Self::Context<'_>,
    ) -> Result<Vec<eredu_core::SpeculativeScheduleState>, eredu_core::BackendFailure> {
        Ok(local)
    }

    /// Agrees host readiness before the next source invocation.
    fn agree_text_preparation(
        stage: eredu_core::run_preparation::TextPreparationStage,
        status: eredu_core::run_preparation::TextPreparationStatus,
        _context: Self::Context<'_>,
    ) -> Result<eredu_core::run_preparation::TextPreparationOutcome, eredu_core::BackendFailure>
    {
        let _ = stage;
        use eredu_core::run_preparation::{
            TextPreparationOutcome as O, TextPreparationStatus as S,
        };
        Ok(match status {
            S::Ready => O::Ready,
            S::Cancelled => O::Cancelled,
            S::Failed => O::Rejected { rank: 0 },
        })
    }
    /// Agrees per-request controls through the retained execution transport.
    fn coordinate_decoding_control(
        local: eredu_core::decoding::DecodingCoordination,
        _context: Self::Context<'_>,
    ) -> Result<eredu_core::decoding::DecodingCoordination, eredu_core::BackendFailure> {
        Ok(local)
    }
    /// An ordinary executable; no model-family dispatch occurs in this mechanism.
    type Model: ?Sized;
    /// Prepared input reusable by both tokenizer-compatible models.
    type Input;
    /// Complete ordinary mutable state.
    type State;
    /// Immutable reusable state checkpoint.
    type Checkpoint: Clone;
    /// Complete sequence logits.
    type Output;
    /// One sampling distribution.
    type Logits: Clone;
    /// Selected native placement.
    type Context<'a>: Copy;
    /// Exact verification completion.
    type Completion: BoundedCompletion<Error = Self::Error>;
    /// Optional backend timing and resource measurements.
    type Telemetry: eredu_core::SpeculativeTelemetry;
    /// Native mechanism failure.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Retention/copy allowance for a handoff's immutable causal score row.
    fn logits_snapshot_estimate(_logits: &Self::Logits) -> Option<SnapshotEstimate> {
        None
    }

    /// Realizes one empty ordinary state using the model's retained selection.
    fn empty(
        model: &mut Self::Model,
        pass: AutoregressivePass,
        context: Self::Context<'_>,
    ) -> Result<Self::State, Self::Error>;
    /// Executes a prepared prompt and returns logits and evaluated token count.
    fn prefill(
        model: &mut Self::Model,
        input: &Self::Input,
        state: &mut Self::State,
        pass: AutoregressivePass,
        context: Self::Context<'_>,
    ) -> Result<(Self::Output, usize), Self::Error>;
    /// Executes supplied token IDs with the ordinary decoder.
    fn decode(
        model: &mut Self::Model,
        tokens: &[u32],
        state: &mut Self::State,
        pass: AutoregressivePass,
        context: Self::Context<'_>,
    ) -> Result<Self::Output, Self::Error>;
    /// Captures complete state without permitting mutation through the checkpoint.
    fn checkpoint(state: &Self::State) -> Result<Self::Checkpoint, Self::Error>;
    /// Creates isolated state, settling any copies before publication.
    fn restore(
        saved: &Self::Checkpoint,
        context: Self::Context<'_>,
    ) -> Result<Self::State, Self::Error>;
    /// Retains the verified prefix in place when the state has an exact truncation
    /// mechanism. `false` leaves state unchanged and selects checkpoint replay.
    fn retain_verified_prefix(
        _state: &mut Self::State,
        _saved: &Self::Checkpoint,
        _accepted_inputs: usize,
        _context: Self::Context<'_>,
    ) -> Result<bool, Self::Error> {
        Ok(false)
    }
    /// Reports a complete logical bound for a durable checkpoint copy.
    fn estimate(saved: &Self::Checkpoint) -> Option<SnapshotEstimate>;
    /// Retained immutable seed including native backing capacity; no checkpoint
    /// is created. Unknown capacity must remain unknown in continuation outlooks.
    fn checkpoint_retained_bytes(_saved: &Self::Checkpoint) -> Option<u64> {
        None
    }
    /// Estimates installed state without allocating a checkpoint or issuing copies.
    fn estimate_state(state: &Self::State) -> Option<SnapshotEstimate>;
    /// Releases optional conversions; caller must establish a settled boundary.
    fn trim_parameter_conversions(
        _model: &mut Self::Model,
    ) -> Result<
        Vec<eredu_core::residency::ParameterConversionRetentionTrimReport>,
        eredu_core::residency::ParameterConversionTrimError,
    > {
        Err(eredu_core::residency::ParameterConversionTrimError::Unsupported)
    }

    /// Observes one model-scoped budget without touching state or native work.
    fn parameter_conversion_retention(
        _model: &Self::Model,
    ) -> Result<
        eredu_core::Observed<Vec<eredu_core::residency::ParameterConversionRetentionReport>>,
        eredu_core::BackendFailure,
    > {
        Ok(eredu_core::residency::unreported_parameter_conversion_retention())
    }

    /// Observes actual cache geometry, capacity and parameter residency without work.
    fn memory_observation(
        _model: &Self::Model,
        _state: &Self::State,
        _additional_positions: u64,
    ) -> Result<
        Option<eredu_core::speculative::SpeculativeModelMemoryObservation>,
        eredu_core::BackendFailure,
    > {
        Ok(None)
    }
    /// Selects one sequence position for the shared sampler.
    fn logits(
        output: &Self::Output,
        position: usize,
        context: Self::Context<'_>,
    ) -> Result<Self::Logits, Self::Error>;
    /// Submits exact completion retaining verification and all mutable state use.
    fn completion(
        output: &Self::Output,
        context: Self::Context<'_>,
    ) -> Result<Self::Completion, Self::Error>;
    /// Constructs a typed native implementation error for invalid transactions.
    fn invalid(message: &'static str) -> Self::Error;
}

/// Both ordinary caches at the canonical frontier.
pub struct AutoregressiveCache<S> {
    /// Target verification cache.
    pub target: S,
    /// Draft cache advanced only after canonical commitment.
    pub draft: S,
}

/// Immutable joint rollback boundary.
pub struct AutoregressiveCheckpoint<C> {
    target: C,
    draft: C,
}

/// Private draft branch. Cloning never shares mutable cache storage.
#[derive(Clone)]
pub struct AutoregressiveProposal<C, R> {
    saved: C,
    refinement: R,
}

/// Prepared prompt and its independent target refinement state.
pub struct AutoregressiveInput<I, R> {
    input: I,
    refinement: R,
}
impl<I, R> AutoregressiveInput<I, R> {
    /// Moves prepared inputs and numerical authority into one request.
    pub fn new(input: I, refinement: R) -> Self {
        Self { input, refinement }
    }
}

/// Causal target prefix, aligned assistant prefix and request-local refinement.
#[derive(Clone)]
pub struct AutoregressiveSeed<C, R> {
    target: Option<C>,
    draft: C,
    refinement: R,
}

/// Independent drafting through the shared ordinary and controlled drivers.
pub struct AutoregressiveExecutor<'a, M: AutoregressiveMechanisms, R: TargetRefinement<M>> {
    target: &'a mut M::Model,
    draft: &'a mut M::Model,
    capacity: std::num::NonZeroUsize,
    refinement: R,
}

impl<'a, M: AutoregressiveMechanisms, R: TargetRefinement<M>> AutoregressiveExecutor<'a, M, R> {
    /// Binds ordinary executables after tokenizer and placement admission.
    pub fn new(
        target: &'a mut M::Model,
        draft: &'a mut M::Model,
        capacity: std::num::NonZeroUsize,
        refinement: R,
    ) -> Self {
        Self {
            target,
            draft,
            capacity,
            refinement,
        }
    }
    /// Realizes an isolated pair of ordinary lane caches.
    pub fn new_cache(
        &mut self,
        context: M::Context<'_>,
    ) -> Result<AutoregressiveCache<M::State>, M::Error> {
        Ok(AutoregressiveCache {
            target: M::empty(self.target, AutoregressivePass::TargetPrefill, context)?,
            draft: M::empty(self.draft, AutoregressivePass::DraftPrefill, context)?,
        })
    }
}

impl<M: AutoregressiveMechanisms, R: TargetRefinement<M>> SpeculativeExecutor
    for AutoregressiveExecutor<'_, M, R>
{
    fn decoding_capabilities(&self) -> eredu_core::decoding::DecodingCapabilities {
        let ordinary = self.refinement.capabilities(self.target);
        let mut capabilities =
            eredu_core::decoding::DecodingCapabilities::causal_drafting(self.capacity.get());
        capabilities.autoregressive &= ordinary.autoregressive;
        capabilities.speculation &= ordinary.autoregressive;
        capabilities.diffusion = ordinary.diffusion;
        capabilities.entropy_feedback = ordinary.entropy_feedback;
        capabilities.masked_unmasking = ordinary.masked_unmasking;
        capabilities.mask_token_id = ordinary.mask_token_id;
        capabilities.with_completion_support(
            M::Completion::supports_cancellation(
                eredu_core::CompletionCancellationMode::NativeCancel,
            ),
            M::Completion::supports_cancellation(
                eredu_core::CompletionCancellationMode::QuarantineUntilComplete,
            ),
        )
    }
    fn refine_block<'a>(
        &mut self,
        state: &mut Self::TargetState,
        logits: &Self::Logits,
        committed: &[u32],
        policy: Option<eredu_core::refinement::RefinementPolicy>,
        temperature: f32,
        context: Self::Context<'a>,
    ) -> Result<Option<eredu_core::speculative::RefinementProgress<Self::Logits>>, Self::Error>
    where
        Self: 'a,
    {
        let Some(prefix) = &state.target else {
            return Ok(None);
        };
        self.refinement.advance(
            self.target,
            prefix,
            &mut state.refinement,
            logits,
            committed,
            policy,
            temperature,
            context,
        )
    }
    fn logits_snapshot_estimate(&self, logits: &Self::Logits) -> Option<SnapshotEstimate> {
        M::logits_snapshot_estimate(logits)
    }
    fn coordinate_speculative_step<'context>(
        &mut self,
        local: Vec<eredu_core::SpeculativeScheduleState>,
        context: Self::Context<'context>,
    ) -> Result<Vec<eredu_core::SpeculativeScheduleState>, eredu_core::BackendFailure>
    where
        Self: 'context,
    {
        M::coordinate_speculative_step(local, context)
    }
    fn agree_text_preparation<'context>(
        &mut self,
        stage: eredu_core::run_preparation::TextPreparationStage,
        status: eredu_core::run_preparation::TextPreparationStatus,
        context: Self::Context<'context>,
    ) -> Result<eredu_core::run_preparation::TextPreparationOutcome, eredu_core::BackendFailure>
    where
        Self: 'context,
    {
        M::agree_text_preparation(stage, status, context)
    }
    fn coordinate_decoding_control<'context>(
        &mut self,
        local: eredu_core::decoding::DecodingCoordination,
        context: Self::Context<'context>,
    ) -> Result<eredu_core::decoding::DecodingCoordination, eredu_core::BackendFailure>
    where
        Self: 'context,
    {
        M::coordinate_decoding_control(local, context)
    }
    type Input = AutoregressiveInput<M::Input, R::State>;
    type Cache = AutoregressiveCache<M::State>;
    type TargetState = AutoregressiveSeed<M::Checkpoint, R::State>;
    type DraftState = AutoregressiveProposal<M::Checkpoint, R::State>;
    type CacheCheckpoint = AutoregressiveCheckpoint<M::Checkpoint>;
    type Verification = CausalVerification<M::Output>;
    type Logits = M::Logits;
    type Context<'a>
        = M::Context<'a>
    where
        Self: 'a;
    type Completion = M::Completion;
    type Telemetry = M::Telemetry;
    type Error = M::Error;

    fn max_proposals(&self) -> usize {
        self.capacity.get()
    }

    fn supports_exact_optimistic_promotion(&self) -> bool {
        true
    }

    fn prefill<'context>(
        &mut self,
        input: Self::Input,
        cache: &mut Self::Cache,
        context: Self::Context<'context>,
    ) -> Result<SpeculativePrefill<Self::TargetState, Self::Logits>, Self::Error>
    where
        Self: 'context,
    {
        let (output, count) = M::prefill(
            self.target,
            &input.input,
            &mut cache.target,
            AutoregressivePass::TargetPrefill,
            context,
        )?;
        if count == 0 {
            return Err(M::invalid(
                "independent drafting requires a nonempty prompt",
            ));
        }
        let (_, draft_count) = M::prefill(
            self.draft,
            &input.input,
            &mut cache.draft,
            AutoregressivePass::DraftPrefill,
            context,
        )?;
        if draft_count != count {
            return Err(M::invalid("draft and target prompt lengths differ"));
        }
        Ok(SpeculativePrefill::new(
            M::logits(&output, count - 1, context)?,
            AutoregressiveSeed {
                target: if self.refinement.capabilities(self.target).diffusion {
                    Some(M::checkpoint(&cache.target)?)
                } else {
                    None
                },
                draft: M::checkpoint(&cache.draft)?,
                refinement: input.refinement,
            },
            count,
        ))
    }

    fn begin_proposal<'context>(
        &mut self,
        state: &Self::TargetState,
        _: u32,
        capacity: usize,
        _: Self::Context<'context>,
    ) -> Result<Self::DraftState, Self::Error>
    where
        Self: 'context,
    {
        if capacity > self.capacity.get() {
            return Err(M::invalid("invalid independent draft proposal capacity"));
        }
        Ok(AutoregressiveProposal {
            saved: state.draft.clone(),
            refinement: state.refinement.clone(),
        })
    }

    fn proposal_step<'context>(
        &mut self,
        proposal: &mut Self::DraftState,
        last_token: u32,
        context: Self::Context<'context>,
    ) -> Result<eredu_core::speculative::SpeculativeProposalStep<Self::Logits>, Self::Error>
    where
        Self: 'context,
    {
        let mut state = M::restore(&proposal.saved, context)?;
        let output = M::decode(
            self.draft,
            &[last_token],
            &mut state,
            AutoregressivePass::Proposal,
            context,
        )?;
        proposal.saved = M::checkpoint(&state)?;
        M::logits(&output, 0, context)
            .map(eredu_core::speculative::SpeculativeProposalStep::CausalLogits)
    }

    fn checkpoint(&self, cache: &Self::Cache) -> Result<Self::CacheCheckpoint, Self::Error> {
        Ok(AutoregressiveCheckpoint {
            target: M::checkpoint(&cache.target)?,
            draft: M::checkpoint(&cache.draft)?,
        })
    }

    fn restore_checkpoint<'context>(
        &mut self,
        cache: &mut Self::Cache,
        saved: &Self::CacheCheckpoint,
        context: Self::Context<'context>,
    ) -> Result<(), Self::Error>
    where
        Self: 'context,
    {
        // Prepare both copies before replacing either installed state.
        let target = M::restore(&saved.target, context)?;
        let draft = M::restore(&saved.draft, context)?;
        *cache = AutoregressiveCache { target, draft };
        Ok(())
    }

    fn submit_verification<'context>(
        &mut self,
        tokens: &[u32],
        cache: &mut Self::Cache,
        context: Self::Context<'context>,
    ) -> Result<Submission<Self::Verification, Self::Completion>, Self::Error>
    where
        Self: 'context,
    {
        if tokens.is_empty() || tokens.len() > self.capacity.get().saturating_add(1) {
            return Err(M::invalid("invalid independent draft verification width"));
        }
        causal_verification::submit::<M>(self.target, tokens, &mut cache.target, context)
    }

    fn verification_logits<'context>(
        &self,
        output: &Self::Verification,
        index: usize,
        context: Self::Context<'context>,
    ) -> Result<Self::Logits, Self::Error>
    where
        Self: 'context,
    {
        causal_verification::logits::<M>(output, index, context)
    }

    fn commit_verification<'context>(
        &mut self,
        output: Self::Verification,
        proposal: Self::DraftState,
        cache: &mut Self::Cache,
        saved: &Self::CacheCheckpoint,
        verified: usize,
        context: Self::Context<'context>,
    ) -> Result<SpeculativeCommit<Self::TargetState>, Self::Error>
    where
        Self: 'context,
    {
        if verified == 0 || verified > output.inputs.len() {
            return Err(M::invalid("invalid independent verification commitment"));
        }
        let tokens = &output.inputs[..verified];
        let replayed = if verified < output.inputs.len() {
            let mut state = M::restore(&saved.target, context)?;
            M::decode(
                self.target,
                tokens,
                &mut state,
                AutoregressivePass::TargetCommit,
                context,
            )?;
            cache.target = state;
            verified
        } else {
            0
        };
        let mut draft = M::restore(&saved.draft, context)?;
        M::decode(
            self.draft,
            tokens,
            &mut draft,
            AutoregressivePass::DraftCommit,
            context,
        )?;
        let seed = M::checkpoint(&draft)?;
        cache.draft = draft;
        Ok(SpeculativeCommit::new(
            AutoregressiveSeed {
                target: if self.refinement.capabilities(self.target).diffusion {
                    Some(M::checkpoint(&cache.target)?)
                } else {
                    None
                },
                draft: seed,
                refinement: proposal.refinement,
            },
            replayed,
        ))
    }

    /// Releases optional conversions; caller must establish a settled boundary.
    fn trim_parameter_conversions(
        &mut self,
    ) -> Result<
        eredu_core::residency::ExecutionConversionRetentionTrimReport,
        eredu_core::residency::ParameterConversionTrimError,
    > {
        Ok(
            eredu_core::residency::ExecutionConversionRetentionTrimReport {
                target: M::trim_parameter_conversions(self.target)?,
                external_drafter: Some(M::trim_parameter_conversions(self.draft)?),
            },
        )
    }

    fn parameter_conversion_retention(
        &self,
    ) -> Result<
        Option<eredu_core::residency::ExecutionConversionRetentionReport>,
        eredu_core::BackendFailure,
    > {
        Ok(Some(
            eredu_core::residency::ExecutionConversionRetentionReport {
                target: M::parameter_conversion_retention(self.target)?,
                external_drafter: Some(M::parameter_conversion_retention(self.draft)?),
            },
        ))
    }

    fn continuation_memory_observation(
        &self,
        cache: &Self::Cache,
        state: &Self::TargetState,
        additional_positions: u64,
    ) -> Result<
        Option<eredu_core::speculative::SpeculativeContinuationObservation>,
        eredu_core::BackendFailure,
    > {
        let Some(target) = M::memory_observation(self.target, &cache.target, additional_positions)?
        else {
            return Ok(None);
        };
        let Some(draft) = M::memory_observation(self.draft, &cache.draft, additional_positions)?
        else {
            return Ok(None);
        };
        Ok(Some(
            eredu_core::speculative::SpeculativeContinuationObservation {
                target,
                draft: Some(draft),
                embedded: None,
                parameter_conversions: None,
                seed_bytes: M::checkpoint_retained_bytes(&state.draft)
                    .and_then(|bytes| {
                        bytes.checked_add(
                            state
                                .target
                                .as_ref()
                                .map_or(Some(0), M::checkpoint_retained_bytes)?,
                        )
                    })
                    .and_then(|bytes| {
                        bytes.checked_add(
                            self.refinement.estimate(&state.refinement)?.retained_bytes,
                        )
                    }),
            },
        ))
    }

    fn control_snapshot_estimate(
        &self,
        cache: &Self::Cache,
        state: &Self::TargetState,
    ) -> Option<SnapshotEstimate> {
        [
            M::estimate_state(&cache.target)?,
            M::estimate_state(&cache.draft)?,
            M::estimate(&state.draft)?,
            state.target.as_ref().map_or(
                Some(SnapshotEstimate {
                    retained_bytes: 0,
                    copy_bytes: 0,
                }),
                M::estimate,
            )?,
            self.refinement.estimate(&state.refinement)?,
        ]
        .into_iter()
        .try_fold(
            SnapshotEstimate {
                retained_bytes: 0,
                copy_bytes: 0,
            },
            |sum, item| {
                Some(SnapshotEstimate {
                    retained_bytes: sum.retained_bytes.checked_add(item.retained_bytes)?,
                    copy_bytes: sum.copy_bytes.checked_add(item.copy_bytes)?,
                })
            },
        )
    }

    fn control_snapshot<'context>(
        &self,
        cache: &Self::Cache,
        state: &Self::TargetState,
        _: Self::Context<'context>,
    ) -> Result<
        Option<(Self::CacheCheckpoint, Self::TargetState)>,
        eredu_core::speculative::SpeculativeControlError,
    >
    where
        Self: 'context,
    {
        Ok(Some((
            self.checkpoint(cache)
                .map_err(eredu_core::speculative::SpeculativeControlError::backend)?,
            state.clone(),
        )))
    }

    fn restore_control_snapshot<'context>(
        &mut self,
        cache: &mut Self::Cache,
        saved: &Self::CacheCheckpoint,
        state: &Self::TargetState,
        context: Self::Context<'context>,
    ) -> Result<Option<Self::TargetState>, eredu_core::speculative::SpeculativeControlError>
    where
        Self: 'context,
    {
        self.restore_checkpoint(cache, saved, context)
            .map_err(eredu_core::speculative::SpeculativeControlError::backend)?;
        Ok(Some(state.clone()))
    }
}
