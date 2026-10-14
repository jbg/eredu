//! Incremental target matching at the ordinary driver's commitment boundaries.
use super::*;
use eredu_core::{
    BoundedCompletionOutcome, BoundedCompletionWait, CompletionCancellationMode,
    SpeculativeProposalStep,
};

/// Failure before a portable verification cursor advances.
#[derive(Debug, thiserror::Error)]
pub enum SharedBackboneTransactionError<E: std::error::Error + 'static> {
    /// Native draft, verification, completion or accepted-prefix storage failed.
    #[error("shared verification failed: {0}")]
    Backend(#[source] E),
    /// A masked source supplied probabilities from a causal draft distribution.
    #[error("masked proposal supplied causal draft evidence")]
    CausalDraftEvidence,
    /// Verification has not safely completed within the admitted allowance.
    #[error("shared verification deadline exceeded ({cancellation:?})")]
    CompletionDeadline {
        /// Safe native disposition selected by the admitted completion policy.
        cancellation: CompletionCancellationMode,
    },
    /// The requested deadline disposition cannot be honored before submission.
    #[error("shared verification does not support {cancellation:?}")]
    UnsupportedCompletionCancellation {
        /// Disposition rejected without native work or proposal draws.
        cancellation: CompletionCancellationMode,
    },
    /// Neutral verification ordering or frozen-choice validation failed.
    #[error(transparent)]
    Policy(#[from] eredu_core::GenerationError),
}

/// Immutable native evidence and a request-local accepted-prefix cursor.
/// Target sampling remains with the evolving ordinary sampler and controller.
pub struct SharedBackboneTransaction<M: SharedBackboneMechanisms> {
    baseline: M::Checkpoint,
    verified: M::Checkpoint,
    proposal: SharedBackboneProposal<M::DraftOptions, M::ProposalRandomness>,
    verification: CausalVerification<M::Output>,
    tokens: Vec<u32>,
    cursor: eredu_core::generation::VerificationCursor,
    capacity: std::num::NonZeroUsize,
}

impl<M: SharedBackboneMechanisms> Clone for SharedBackboneTransaction<M>
where
    M::Output: Clone,
{
    fn clone(&self) -> Self {
        Self {
            baseline: self.baseline.clone(),
            verified: self.verified.clone(),
            proposal: self.proposal.clone(),
            verification: self.verification.clone(),
            tokens: self.tokens.clone(),
            cursor: self.cursor.clone(),
            capacity: self.capacity,
        }
    }
}

impl<M: SharedBackboneMechanisms> SharedBackboneTransaction<M> {
    /// Submits one masked proposal and retains only exactly completed verification.
    pub fn submit<'context>(
        executor: &mut SharedBackboneExecutor<'context, M>,
        cache: &mut M::State,
        last: u32,
        options: M::DraftOptions,
        randomness: M::ProposalRandomness,
        wait: BoundedCompletionWait,
        context: M::Context<'context>,
    ) -> Result<Self, SharedBackboneTransactionError<M::Error>>  where M: 'context {
        use SharedBackboneTransactionError::Backend;
        if !M::Completion::supports_cancellation(wait.cancellation()) {
            return Err(
                SharedBackboneTransactionError::UnsupportedCompletionCancellation {
                    cancellation: wait.cancellation(),
                },
            );
        }
        let capacity = executor.capacity;
        let baseline = executor.checkpoint(cache).map_err(Backend)?;
        let mut retained_output = None;
        let result = (|| {
            let seed =
                executor.resume_seed(baseline.clone(), options, randomness.clone(), randomness);
            let mut proposal = executor
                .begin_proposal(&seed, last, capacity.get(), context)
                .map_err(Backend)?;
            let mut tokens = vec![last];
            for _ in 0..capacity.get() {
                let SpeculativeProposalStep::TargetMatchingToken(token) = executor
                    .proposal_step(&mut proposal, *tokens.last().unwrap(), context)
                    .map_err(Backend)?
                else {
                    return Err(SharedBackboneTransactionError::CausalDraftEvidence);
                };
                tokens.push(token);
            }
            let submitted = executor
                .submit_verification(&tokens, cache, context)
                .map_err(Backend)?;
            retained_output = Some(submitted.output);
            match submitted.completion.wait_bounded(wait).map_err(Backend)? {
                BoundedCompletionOutcome::Completed => {}
                BoundedCompletionOutcome::DeadlineExceeded { cancellation } => {
                    return Err(SharedBackboneTransactionError::CompletionDeadline {
                        cancellation,
                    });
                }
            }
            let verified = executor.checkpoint(cache).map_err(Backend)?;
            Ok(Self {
                baseline: baseline.clone(),
                verified,
                proposal,
                verification: retained_output
                    .take()
                    .expect("completed verification is retained"),
                tokens,
                cursor: eredu_core::generation::VerificationCursor::new(capacity.get())?,
                capacity,
            })
        })();
        if result.is_err() {
            executor
                .restore_checkpoint(cache, &baseline, context)
                .map_err(Backend)?;
        }
        result
    }
    /// Fixed proposal allowance retained independently of the model's parameters.
    pub const fn capacity(&self) -> std::num::NonZeroUsize {
        self.capacity
    }
    /// Checkpoint before this verification transaction.
    pub fn baseline(&self) -> &M::Checkpoint {
        &self.baseline
    }
    /// Completed verification checkpoint, including the tentative suffix.
    pub fn verified(&self) -> &M::Checkpoint {
        &self.verified
    }
    /// Causal outputs retained until accepted-prefix commitment.
    pub fn verification(&self) -> &CausalVerification<M::Output> {
        &self.verification
    }
    /// Spent proposal randomness, preserved even if verification rejects a token.
    pub fn proposal_randomness(&self) -> &M::ProposalRandomness {
        self.proposal.randomness()
    }
    /// Tentative proposals, excluding the preceding committed token.
    pub fn proposals(&self) -> &[u32] {
        &self.tokens[1..]
    }
    /// Conservative host storage for the tokens and cursor, excluding native evidence.
    pub fn host_retained_bytes(&self) -> Option<u64> {
        (self.tokens.capacity() as u64)
            .checked_add(self.proposal.tokens.capacity() as u64)?
            .checked_add(self.verification.inputs.capacity() as u64)?
            .checked_mul(4)?
            .checked_add(self.cursor.heap_bytes()?)?
            .checked_add(std::mem::size_of::<Self>() as u64)
    }
    /// Causal row for the next evolving target decision.
    pub fn logits<'context>(
        &self,
        executor: &SharedBackboneExecutor<'context, M>,
        context: M::Context<'context>,
    ) -> Result<M::Logits, SharedBackboneTransactionError<M::Error>>  where M: 'context {
        let index = self.cursor.position()?;
        executor
            .verification_logits(&self.verification, index, context)
            .map_err(SharedBackboneTransactionError::Backend)
    }
    /// Retains a target decision without advancing the accepted prefix.
    pub fn stage_target_token(
        &mut self,
        token: u32,
    ) -> Result<(), SharedBackboneTransactionError<M::Error>> {
        use eredu_core::generation::VerificationDecision;
        let index = self.cursor.position()?;
        let decision = match self.tokens.get(index + 1) {
            Some(proposal) if *proposal == token => VerificationDecision::Accept(token),
            Some(_) => VerificationDecision::Replacement(token),
            None => VerificationDecision::Bonus(token),
        };
        self.cursor.stage(decision, false)?;
        Ok(())
    }
    /// Commits a matching proposal or the first replacement/full-acceptance bonus.
    /// Returns whether the cursor can continue and the accepted native cache.
    /// A continuing cursor's committed token is already in that cache.
    pub fn commit_target_token<'context>(
        &mut self,
        executor: &mut SharedBackboneExecutor<'context, M>,
        token: u32,
        context: M::Context<'context>,
    ) -> Result<(bool, M::State), SharedBackboneTransactionError<M::Error>>
    where
        M::Output: Clone,
        M: 'context {
        use SharedBackboneTransactionError::Backend;
        let boundary = self.cursor.boundary(token)?;
        let mut cache = M::restore(&self.verified, context).map_err(Backend)?;
        executor
            .commit_verification(
                self.verification.clone(),
                self.proposal.clone(),
                &mut cache,
                &self.baseline,
                boundary.verified_inputs,
                context,
            )
            .map_err(Backend)?;
        self.cursor.commit(token)?;
        Ok((boundary.continues, cache))
    }
}
