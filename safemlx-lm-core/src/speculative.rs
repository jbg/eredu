//! High-level contracts and orchestration for speculative execution backends.

use crate::{
    backend::{Completion, Submission},
    generation::{
        FinishReason, GenerationError, GenerationSequence, MtpRequestPhase, MtpSchedulerOptions,
        SpeculativeRound, TokenTerminalSignals,
    },
};
use serde::{Deserialize, Serialize};

/// Relationship between target and assistant execution placements.
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
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

/// Backend-owned first-token output and assistant seed state.
#[derive(Debug)]
pub struct SpeculativePrefill<State, Logits> {
    /// Opaque logits used by the selected backend sampler.
    pub logits: Logits,
    /// Backend state from which the first proposal round begins.
    pub state: State,
    /// Number of prompt tokens evaluated by the target.
    pub evaluated_tokens: usize,
}

/// Result of committing one exact target verification transaction.
#[derive(Debug)]
pub struct SpeculativeCommit<State> {
    /// Assistant seed state matching the committed target cache.
    pub state: State,
    /// Target tokens replayed while restoring the exact retained prefix.
    pub replayed_tokens: usize,
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
    type DraftState: Clone;
    /// Exact target-cache checkpoint marker.
    type CacheCheckpoint;
    /// Retained target verification output.
    type Verification;
    /// Opaque logits consumed by the backend's sampling adapter.
    type Logits;
    /// Backend execution assignment for one operation.
    type Context<'a>: Copy
    where
        Self: 'a;
    /// Exact completion for submitted verification work.
    type Completion: Completion<Error = Self::Error>;
    /// Optional backend-specific component telemetry.
    type Telemetry: Default;
    /// Structured backend error.
    type Error: std::error::Error + Send + Sync + 'static;

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
    fn checkpoint(cache: &Self::Cache) -> Self::CacheCheckpoint;

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
        checkpoint: Self::CacheCheckpoint,
        verified_inputs: usize,
        context: Self::Context<'a>,
    ) -> Result<SpeculativeCommit<Self::TargetState>, Self::Error>;
}

/// Target decision for one assistant proposal.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum ProposalDecision {
    /// Retain the assistant proposal.
    Accept,
    /// Reject it and commit this target replacement.
    Reject(u32),
}

/// Logical model side on which an opaque sampling operation executes.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum SamplingPlacement {
    /// Canonical target-model execution.
    Target,
    /// Tentative assistant-model execution.
    Draft,
}

/// Backend-owned random streams for canonical and position-stable sampling.
#[derive(Debug, Clone)]
pub struct SpeculativeRandomness<R, D> {
    /// Sequential target randomness.
    pub target: Option<R>,
    /// Position-addressable assistant randomness.
    pub draft: Option<D>,
}

/// High-level sampling contract used by speculative orchestration.
///
/// Backends implement complete semantic operations over opaque logits and
/// distributions. Core never requests softmax, indexing, random kernels, or
/// another primitive tensor operation.
pub trait SpeculativeSampling: Clone {
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
    /// Backend execution assignment.
    type Context<'a>: Copy
    where
        Self: 'a;
    /// Structured backend error.
    type Error: std::error::Error + Send + Sync + 'static;

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

    /// Splits caller randomness into canonical and position-stable streams.
    fn initialize_randomness<'a>(
        seed: Option<Self::Seed>,
        temperature: f32,
        context: Self::Context<'a>,
    ) -> Result<SpeculativeRandomness<Self::RandomState, Self::DraftRandomness>, Self::Error>
    where
        Self: 'a;

    /// Derives assistant randomness for one absolute output position.
    fn draft_randomness_at<'a>(
        root: &Self::DraftRandomness,
        position: usize,
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

    /// Makes the exact accept-or-replacement decision for one proposal.
    fn decide_proposal<'a>(
        &self,
        target: &Self::Distribution,
        draft: &Self::Distribution,
        proposed: u32,
        temperature: f32,
        randomness: Option<&mut Self::RandomState>,
        context: Self::Context<'a>,
    ) -> Result<ProposalDecision, Self::Error>
    where
        Self: 'a;

    /// Commits a token only after target acceptance or replacement.
    fn commit_token<'a>(
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

/// One sampled assistant proposal and its retained distribution.
#[derive(Debug)]
pub struct SpeculativeProposal<D> {
    /// Proposed token id.
    pub token: u32,
    /// Backend-owned processed assistant distribution.
    pub distribution: D,
}

/// Transactional semantic state paired with committed token sequencing.
pub trait SpeculativeConstraint: Sized {
    /// Structured backend error.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Forks state for tentative verification.
    fn fork(&self) -> Result<Self, Self::Error>;
    /// Stages one token and reports a matched stop condition.
    fn push_token(&mut self, token: u32) -> Result<bool, Self::Error>;
    /// Stages terminal output.
    fn finish(&mut self, reason: FinishReason) -> Result<(), Self::Error>;
}

/// Error returned by portable proposal and verification drivers.
#[derive(Debug, thiserror::Error)]
pub enum SpeculativeDriverError<E: std::error::Error + 'static> {
    /// Backend execution or sampling failed.
    #[error(transparent)]
    Backend(#[from] E),
    /// Portable lifecycle validation failed.
    #[error(transparent)]
    Generation(GenerationError),
}

/// Resolved speculative transaction ready for backend cache commit.
pub struct ResolvedSpeculativeRound<S, C, R> {
    /// Tentatively advanced sampler state.
    pub sampler: S,
    /// Tentatively advanced semantic state.
    pub constraint: C,
    /// Tentatively advanced canonical sequence.
    pub sequence: GenerationSequence,
    /// Tentatively advanced target randomness.
    pub target_randomness: Option<R>,
    /// Number of accepted proposals.
    pub accepted_proposals: usize,
    /// Tokens visible after cache commit.
    pub committed_tokens: Vec<u32>,
    /// Exact verification inputs retained by cache commit.
    pub verified_inputs: usize,
    /// Target bonus token, when full acceptance produced one.
    pub bonus_token: Option<u32>,
    /// Terminal reason after this round.
    pub finish_reason: Option<FinishReason>,
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
    let mut branch_sampler = sampler.clone();
    let mut history = Vec::with_capacity(base_history.len() + count);
    history.extend_from_slice(base_history);
    let mut proposals: Vec<SpeculativeProposal<S::Distribution>> = Vec::with_capacity(count);
    for offset in 0..count {
        let previous = proposals
            .last()
            .map_or(first_previous, |proposal| proposal.token);
        let raw = executor.proposal_logits(state, previous, context)?;
        let distribution = branch_sampler.process_logits(
            &raw,
            temperature,
            &history,
            SamplingPlacement::Draft,
            context,
        )?;
        let mut position_state = draft_randomness
            .map(|root| S::draft_randomness_at(root, base_history.len() + offset, context))
            .transpose()?;
        let token = branch_sampler.sample(
            &distribution,
            temperature,
            position_state.as_mut(),
            SamplingPlacement::Draft,
            context,
        )?;
        proposals.push(SpeculativeProposal {
            token,
            distribution,
        });
        history.push(token);
        if eos_token_ids.contains(&token) || branch_sampler.prefix_is_complete(&history)? {
            break;
        }
    }
    Ok(proposals)
}

/// Resolves one target verification transaction without backend-specific math.
#[allow(clippy::too_many_arguments)]
pub fn resolve_round<'a, E, S, C>(
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
    C: SpeculativeConstraint<Error = E::Error>,
{
    let mut draft_distributions = proposals
        .iter_mut()
        .map(|proposal| &mut proposal.distribution)
        .collect::<Vec<_>>();
    sampler.prepare_verification(&mut draft_distributions, temperature, context)?;
    let proposal_count = proposals.len();
    let mut sampler = sampler.clone();
    let mut sequence = sequence.clone();
    let mut constraint = constraint.fork()?;
    let mut target_randomness = target_randomness.cloned();
    let mut history = sequence.tokens().to_vec();
    let mut round =
        SpeculativeRound::new(proposal_count).map_err(SpeculativeDriverError::Generation)?;
    let mut finish_reason = None;

    for (index, proposal) in proposals.iter().enumerate() {
        let raw = E::verification_logits(verification, index, context)?;
        let target = sampler.process_logits(
            &raw,
            temperature,
            &history,
            SamplingPlacement::Target,
            context,
        )?;
        match sampler.decide_proposal(
            &target,
            &proposal.distribution,
            proposal.token,
            temperature,
            target_randomness.as_mut(),
            context,
        )? {
            ProposalDecision::Accept => {
                sampler.commit_token(
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
                    .accept(proposal.token, finish_reason.is_some())
                    .map_err(SpeculativeDriverError::Generation)?;
                if finish_reason.is_some() {
                    break;
                }
            }
            ProposalDecision::Reject(replacement) => {
                sampler.commit_token(&target, replacement, SamplingPlacement::Target, context)?;
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
        let raw = E::verification_logits(verification, proposal_count, context)?;
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
        sampler.commit_token(&target, chosen, SamplingPlacement::Target, context)?;
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

fn commit_terminal_token<S, C>(
    sequence: &mut GenerationSequence,
    sampler: &mut S,
    constraint: &mut C,
    token: u32,
) -> Result<Option<FinishReason>, SpeculativeDriverError<S::Error>>
where
    S: SpeculativeSampling,
    C: SpeculativeConstraint<Error = S::Error>,
{
    let stop_matched = constraint.push_token(token)?;
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
        constraint.finish(reason)?;
    }
    Ok(reason)
}

/// Portable candidate snapshot used by fair speculative action selection.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct SpeculativeCandidate {
    /// Current validated request phase.
    pub phase: MtpRequestPhase,
    /// Whether this request may start exact optimistic work now.
    pub optimistic_eligible: bool,
}

/// One backend action selected by the portable fair scheduler.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
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
    /// Resolve one exact verification completion.
    ResolveVerification(usize),
}

/// Backend-neutral fair action selector for speculative requests.
pub struct SpeculativeSchedule {
    options: MtpSchedulerOptions,
    cursor: usize,
}

impl SpeculativeSchedule {
    /// Creates a validated schedule.
    pub fn new(options: MtpSchedulerOptions) -> Result<Self, GenerationError> {
        Ok(Self {
            options: options.validate()?,
            cursor: 0,
        })
    }

    /// Validated scheduler options.
    pub const fn options(&self) -> MtpSchedulerOptions {
        self.options
    }

    /// Selects the next fair action, or `None` when every request is terminal.
    pub fn next_action(
        &mut self,
        candidates: &[SpeculativeCandidate],
    ) -> Result<Option<SpeculativeAction>, GenerationError> {
        if candidates.iter().all(|candidate| {
            matches!(
                candidate.phase,
                MtpRequestPhase::Completed | MtpRequestPhase::Cancelled
            )
        }) {
            return Ok(None);
        }
        let in_flight = candidates
            .iter()
            .filter(|candidate| {
                matches!(
                    candidate.phase,
                    MtpRequestPhase::TargetVerificationInFlight
                        | MtpRequestPhase::OptimisticDraftInProgress
                        | MtpRequestPhase::OptimisticDraftReady
                        | MtpRequestPhase::VerificationResolution
                )
            })
            .count();
        let optimistic = candidates
            .iter()
            .filter(|candidate| candidate.phase == MtpRequestPhase::OptimisticDraftReady)
            .count();

        if in_flight < self.options.max_in_flight_verifications {
            if let Some(index) = self.select(candidates, |candidate| {
                candidate.phase == MtpRequestPhase::ReadyToSubmitVerification
            }) {
                return Ok(Some(SpeculativeAction::SubmitVerification(index)));
            }
        }
        if in_flight > 0 {
            if optimistic < self.options.max_optimistic_branches
                && self.options.lookahead_blocks > 0
            {
                if let Some(index) = self.select(candidates, |candidate| {
                    candidate.phase == MtpRequestPhase::TargetVerificationInFlight
                        && candidate.optimistic_eligible
                }) {
                    return Ok(Some(SpeculativeAction::DraftOptimistic(index)));
                }
            }
            if let Some(index) = self.select(candidates, |candidate| {
                candidate.phase == MtpRequestPhase::ReadyToDraft
            }) {
                return Ok(Some(SpeculativeAction::DraftCommitted {
                    index,
                    cross_request: true,
                }));
            }
            if let Some(index) = self.select(candidates, |candidate| {
                matches!(
                    candidate.phase,
                    MtpRequestPhase::TargetVerificationInFlight
                        | MtpRequestPhase::OptimisticDraftReady
                )
            }) {
                return Ok(Some(SpeculativeAction::ResolveVerification(index)));
            }
        } else if let Some(index) = self.select(candidates, |candidate| {
            candidate.phase == MtpRequestPhase::ReadyToDraft
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
    use super::*;
    use std::convert::Infallible;

    #[derive(Debug, Clone, Copy)]
    struct Done;

    impl Completion for Done {
        type Error = Infallible;

        fn is_complete(&self) -> Result<bool, Self::Error> {
            Ok(true)
        }

        fn wait(&self) -> Result<(), Self::Error> {
            Ok(())
        }
    }

    struct MockExecutor;

    struct MockVerification {
        tokens: Vec<u32>,
        logits: Vec<Vec<f32>>,
    }

    impl SpeculativeExecutor for MockExecutor {
        type Input = Vec<u32>;
        type Cache = Vec<u32>;
        type TargetState = usize;
        type DraftState = Vec<u32>;
        type CacheCheckpoint = usize;
        type Verification = MockVerification;
        type Logits = Vec<f32>;
        type Context<'a> = ();
        type Completion = Done;
        type Telemetry = ();
        type Error = Infallible;

        fn prefill<'context>(
            &mut self,
            input: Self::Input,
            cache: &mut Self::Cache,
            _: Self::Context<'context>,
        ) -> Result<SpeculativePrefill<Self::TargetState, Self::Logits>, Self::Error>
        where
            Self: 'context,
        {
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
        ) -> Result<Self::DraftState, Self::Error> {
            Ok(vec![last_token])
        }

        fn proposal_logits<'a>(
            &mut self,
            state: &mut Self::DraftState,
            last_token: u32,
            _: Self::Context<'a>,
        ) -> Result<Self::Logits, Self::Error> {
            state.push(last_token + 1);
            Ok(vec![0.0, 1.0])
        }

        fn checkpoint(cache: &Self::Cache) -> Self::CacheCheckpoint {
            cache.len()
        }

        fn submit_verification<'a>(
            &mut self,
            input_tokens: &[u32],
            cache: &mut Self::Cache,
            _: Self::Context<'a>,
        ) -> Result<Submission<Self::Verification, Self::Completion>, Self::Error> {
            cache.extend_from_slice(input_tokens);
            Ok(Submission {
                output: MockVerification {
                    tokens: input_tokens.to_vec(),
                    logits: vec![vec![0.0, 1.0], vec![1.0, 0.0], vec![0.0, 1.0]],
                },
                completion: Done,
            })
        }

        fn verification_logits<'a>(
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
            checkpoint: Self::CacheCheckpoint,
            verified_inputs: usize,
            _: Self::Context<'a>,
        ) -> Result<SpeculativeCommit<Self::TargetState>, Self::Error> {
            assert!(!output.tokens.is_empty());
            cache.truncate(checkpoint + verified_inputs);
            Ok(SpeculativeCommit {
                state: draft_state.len(),
                replayed_tokens: 0,
            })
        }
    }

    #[test]
    fn mock_executor_prefill_propose_verify_and_commit_without_a_tensor_runtime() {
        let mut executor = MockExecutor;
        let mut cache = Vec::new();
        let prefill = executor.prefill(vec![4, 5], &mut cache, ()).unwrap();
        let mut draft = executor.begin_proposal(&prefill.state, 5, 2, ()).unwrap();
        assert_eq!(
            executor.proposal_logits(&mut draft, 5, ()).unwrap(),
            [0.0, 1.0]
        );
        let checkpoint = MockExecutor::checkpoint(&cache);
        let submission = executor
            .submit_verification(&[5, 6], &mut cache, ())
            .unwrap();
        submission.completion.wait().unwrap();
        let commit = executor
            .commit_verification(submission.output, draft, &mut cache, checkpoint, 1, ())
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

    #[derive(Clone, Default)]
    struct MockSampling {
        committed: Vec<u32>,
    }

    impl SpeculativeSampling for MockSampling {
        type Logits = Vec<f32>;
        type Distribution = Vec<f32>;
        type Seed = ();
        type RandomState = usize;
        type DraftRandomness = usize;
        type Context<'a> = ();
        type Error = Infallible;

        fn initialize_randomness<'a>(
            _: Option<Self::Seed>,
            _: f32,
            _: Self::Context<'a>,
        ) -> Result<SpeculativeRandomness<Self::RandomState, Self::DraftRandomness>, Self::Error>
        where
            Self: 'a,
        {
            Ok(SpeculativeRandomness {
                target: Some(0),
                draft: Some(0),
            })
        }

        fn draft_randomness_at<'a>(
            root: &Self::DraftRandomness,
            position: usize,
            _: Self::Context<'a>,
        ) -> Result<Self::RandomState, Self::Error>
        where
            Self: 'a,
        {
            Ok(root + position)
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
            if let Some(randomness) = randomness {
                *randomness += 1;
            }
            Ok(argmax(distribution))
        }

        fn decide_proposal<'a>(
            &self,
            target: &Self::Distribution,
            _: &Self::Distribution,
            proposed: u32,
            _: f32,
            randomness: Option<&mut Self::RandomState>,
            _: Self::Context<'a>,
        ) -> Result<ProposalDecision, Self::Error>
        where
            Self: 'a,
        {
            if let Some(randomness) = randomness {
                *randomness += 1;
            }
            let target = argmax(target);
            Ok(if target == proposed {
                ProposalDecision::Accept
            } else {
                ProposalDecision::Reject(target)
            })
        }

        fn commit_token<'a>(
            &mut self,
            _: &Self::Distribution,
            token: u32,
            _: SamplingPlacement,
            _: Self::Context<'a>,
        ) -> Result<(), Self::Error>
        where
            Self: 'a,
        {
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
        type Error = Infallible;

        fn fork(&self) -> Result<Self, Self::Error> {
            Ok(Self {
                tokens: self.tokens.clone(),
                finished: self.finished,
            })
        }

        fn push_token(&mut self, token: u32) -> Result<bool, Self::Error> {
            self.tokens.push(token);
            Ok(false)
        }

        fn finish(&mut self, reason: FinishReason) -> Result<(), Self::Error> {
            self.finished = Some(reason);
            Ok(())
        }
    }

    #[test]
    fn portable_driver_proposes_and_resolves_acceptance_and_replacement() {
        let mut executor = MockExecutor;
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
        let resolved = resolve_round::<MockExecutor, MockSampling, MockConstraint>(
            &verification,
            proposals,
            &sampler,
            &sequence,
            &MockConstraint::default(),
            Some(&0),
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
        assert_eq!(resolved.target_randomness, Some(2));
        assert_eq!(resolved.finish_reason, None);
    }

    #[test]
    fn portable_schedule_is_fair_and_respects_retained_capacity() {
        let mut schedule = SpeculativeSchedule::new(MtpSchedulerOptions::default()).unwrap();
        let ready = SpeculativeCandidate {
            phase: MtpRequestPhase::ReadyToSubmitVerification,
            optimistic_eligible: false,
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
            phase: MtpRequestPhase::TargetVerificationInFlight,
            optimistic_eligible: false,
        };
        let draft = SpeculativeCandidate {
            phase: MtpRequestPhase::ReadyToDraft,
            optimistic_eligible: false,
        };
        assert_eq!(
            schedule.next_action(&[in_flight, ready, draft]).unwrap(),
            Some(SpeculativeAction::DraftCommitted {
                index: 2,
                cross_request: true,
            })
        );
    }
}
