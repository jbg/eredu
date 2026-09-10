//! Speculative transactions between two independently executable language models.
//!
//! Models retain their ordinary equations, residency and state geometry. This
//! mechanism owns only proposal isolation, verification and accepted-prefix replay.
use eredu_core::{
    execution_control::SnapshotEstimate, BoundedCompletion, SpeculativeCommit, SpeculativeExecutor,
    SpeculativePrefill, Submission,
};

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
    type Logits;
    /// Selected native placement.
    type Context<'a>: Copy;
    /// Exact verification completion.
    type Completion: BoundedCompletion<Error = Self::Error>;
    /// Optional backend timing and resource measurements.
    type Telemetry: eredu_core::SpeculativeTelemetry;
    /// Native mechanism failure.
    type Error: std::error::Error + Send + Sync + 'static;

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
    /// Reports a complete logical bound for a durable checkpoint copy.
    fn estimate(saved: &Self::Checkpoint) -> Option<SnapshotEstimate>;
    /// Estimates installed state without allocating a checkpoint or issuing copies.
    fn estimate_state(state: &Self::State) -> Option<SnapshotEstimate>;
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
pub struct AutoregressiveProposal<C> {
    saved: C,
}

/// Retained target outputs and their exact verification inputs.
pub struct AutoregressiveVerification<O> {
    output: O,
    tokens: Vec<u32>,
}

/// Independent drafting through the shared ordinary and controlled drivers.
pub struct AutoregressiveExecutor<'a, M: AutoregressiveMechanisms> {
    target: &'a mut M::Model,
    draft: &'a mut M::Model,
    capacity: std::num::NonZeroUsize,
}

impl<'a, M: AutoregressiveMechanisms> AutoregressiveExecutor<'a, M> {
    /// Binds ordinary executables after tokenizer and placement admission.
    pub fn new(
        target: &'a mut M::Model,
        draft: &'a mut M::Model,
        capacity: std::num::NonZeroUsize,
    ) -> Self {
        Self {
            target,
            draft,
            capacity,
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

impl<M: AutoregressiveMechanisms> SpeculativeExecutor for AutoregressiveExecutor<'_, M> {
    type Input = M::Input;
    type Cache = AutoregressiveCache<M::State>;
    type TargetState = M::Checkpoint;
    type DraftState = AutoregressiveProposal<M::Checkpoint>;
    type CacheCheckpoint = AutoregressiveCheckpoint<M::Checkpoint>;
    type Verification = AutoregressiveVerification<M::Output>;
    type Logits = M::Logits;
    type Context<'a> = M::Context<'a>;
    type Completion = M::Completion;
    type Telemetry = M::Telemetry;
    type Error = M::Error;

    fn max_proposals(&self) -> usize {
        self.capacity.get()
    }

    fn supports_exact_optimistic_promotion(&self) -> bool {
        true
    }

    fn prefill(
        &mut self,
        input: Self::Input,
        cache: &mut Self::Cache,
        context: Self::Context<'_>,
    ) -> Result<SpeculativePrefill<Self::TargetState, Self::Logits>, Self::Error> {
        let (output, count) = M::prefill(
            self.target,
            &input,
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
            &input,
            &mut cache.draft,
            AutoregressivePass::DraftPrefill,
            context,
        )?;
        if draft_count != count {
            return Err(M::invalid("draft and target prompt lengths differ"));
        }
        Ok(SpeculativePrefill::new(
            M::logits(&output, count - 1, context)?,
            M::checkpoint(&cache.draft)?,
            count,
        ))
    }

    fn begin_proposal(
        &mut self,
        state: &Self::TargetState,
        _: u32,
        capacity: usize,
        _: Self::Context<'_>,
    ) -> Result<Self::DraftState, Self::Error> {
        if capacity == 0 || capacity > self.capacity.get() {
            return Err(M::invalid("invalid independent draft proposal capacity"));
        }
        Ok(AutoregressiveProposal {
            saved: state.clone(),
        })
    }

    fn proposal_logits(
        &mut self,
        proposal: &mut Self::DraftState,
        last_token: u32,
        context: Self::Context<'_>,
    ) -> Result<Self::Logits, Self::Error> {
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
    }

    fn checkpoint(&self, cache: &Self::Cache) -> Result<Self::CacheCheckpoint, Self::Error> {
        Ok(AutoregressiveCheckpoint {
            target: M::checkpoint(&cache.target)?,
            draft: M::checkpoint(&cache.draft)?,
        })
    }

    fn restore_checkpoint(
        &mut self,
        cache: &mut Self::Cache,
        saved: &Self::CacheCheckpoint,
        context: Self::Context<'_>,
    ) -> Result<(), Self::Error> {
        // Prepare both copies before replacing either installed state.
        let target = M::restore(&saved.target, context)?;
        let draft = M::restore(&saved.draft, context)?;
        *cache = AutoregressiveCache { target, draft };
        Ok(())
    }

    fn submit_verification(
        &mut self,
        tokens: &[u32],
        cache: &mut Self::Cache,
        context: Self::Context<'_>,
    ) -> Result<Submission<Self::Verification, Self::Completion>, Self::Error> {
        if tokens.is_empty() || tokens.len() > self.capacity.get().saturating_add(1) {
            return Err(M::invalid("invalid independent draft verification width"));
        }
        let output = M::decode(
            self.target,
            tokens,
            &mut cache.target,
            AutoregressivePass::Verification,
            context,
        )?;
        let completion = M::completion(&output, context)?;
        Ok(Submission {
            output: AutoregressiveVerification {
                output,
                tokens: tokens.to_vec(),
            },
            completion,
        })
    }

    fn verification_logits(
        &self,
        output: &Self::Verification,
        index: usize,
        context: Self::Context<'_>,
    ) -> Result<Self::Logits, Self::Error> {
        if index >= output.tokens.len() {
            return Err(M::invalid("independent verification index is out of range"));
        }
        M::logits(&output.output, index, context)
    }

    fn commit_verification(
        &mut self,
        output: Self::Verification,
        _: Self::DraftState,
        cache: &mut Self::Cache,
        saved: &Self::CacheCheckpoint,
        verified: usize,
        context: Self::Context<'_>,
    ) -> Result<SpeculativeCommit<Self::TargetState>, Self::Error> {
        if verified == 0 || verified > output.tokens.len() {
            return Err(M::invalid("invalid independent verification commitment"));
        }
        let tokens = &output.tokens[..verified];
        let replayed = if verified < output.tokens.len() {
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
        Ok(SpeculativeCommit::new(seed, replayed))
    }

    fn control_snapshot_estimate(
        &self,
        cache: &Self::Cache,
        state: &Self::TargetState,
    ) -> Option<SnapshotEstimate> {
        [
            M::estimate_state(&cache.target)?,
            M::estimate_state(&cache.draft)?,
            M::estimate(state)?,
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

    fn control_snapshot(
        &self,
        cache: &Self::Cache,
        state: &Self::TargetState,
        _: Self::Context<'_>,
    ) -> Result<Option<(Self::CacheCheckpoint, Self::TargetState)>, Self::Error> {
        Ok(Some((self.checkpoint(cache)?, state.clone())))
    }

    fn restore_control_snapshot(
        &mut self,
        cache: &mut Self::Cache,
        saved: &Self::CacheCheckpoint,
        state: &Self::TargetState,
        context: Self::Context<'_>,
    ) -> Result<Option<Self::TargetState>, Self::Error> {
        self.restore_checkpoint(cache, saved, context)?;
        Ok(Some(state.clone()))
    }
}
