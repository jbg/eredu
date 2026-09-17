//! Backend-neutral causal-model and token-sampling contracts.

use crate::execution_control::TokenChoiceController;
use eredu_core::{
    SpeculativeTokenFilterController, TokenFilter, TokenFilterController,
    generation::ResolvedGenerationConfig,
};
use eredu_nn::Tensor;

mod prepared_choice;
pub use prepared_choice::{
    PreparedCategoricalPolicy, PreparedGreedyError, PreparedGreedyPolicy,
    SpeculativeCategoricalProgram, SpeculativeGreedyProgram,
};
mod prepared_logits;
pub use prepared_logits::{
    LogitProgramError, PreparedLogitPolicy, PreparedLogitPolicyError, SpeculativeLogitProgram,
};
mod prepared_grammar;
pub use prepared_grammar::{PreparedGrammarSampler, PreparedGrammarLogits, PreparedGrammarSamplerCause, PreparedGrammarSamplerError};
mod prepared_controller;
pub use prepared_controller::{
    PreparedControlledChoiceError, PreparedControlledLogits, PreparedControllerChoice,
    PreparedControllerError, PreparedSpeculativeController,
};
mod adaptive_commit;
pub use adaptive_commit::{PreparedAdaptiveCommit, PreparedAdaptiveCommitError};
mod speculative_copy;
pub use speculative_copy::PreparedSpeculativeSamplerCopy;
mod history;
pub(crate) use history::next_history_capacity;
#[cfg(test)]
pub(crate) use history::payload_copy_count;
mod text_sampler;
pub use text_sampler::{
    ConfiguredTextSampler, SamplerCopyError, SamplerCopyPlan, SamplerProjectionError,
    SamplerWorkspaceProjection,
};

/// Monomorphized causal model used by generation sessions.
pub trait CausalModel<S> {
    /// Backend-native tensor handle containing logits and decode token ids.
    type Tensor: Tensor;
    /// Borrowed, tokenizer/media-prepared prefill input.
    type Input<'a>: Copy;
    /// Concrete model or backend failure.
    type Error;

    /// Computes initial logits and updates mutable state.
    fn prefill_input_logits(
        &mut self,
        input: Self::Input<'_>,
        state: &mut S,
        context: &<Self::Tensor as Tensor>::Context,
    ) -> Result<Self::Tensor, Self::Error>;

    /// Computes logits for decode tokens using existing mutable state.
    fn decode_logits(
        &mut self,
        input_tokens: &Self::Tensor,
        state: &mut S,
        context: &<Self::Tensor as Tensor>::Context,
    ) -> Result<Self::Tensor, Self::Error>;

    /// Adjusts prefill logits before backend-native sampling.
    fn adjust_prefill_logits(
        &mut self,
        logits: Self::Tensor,
        _state: &mut S,
        _context: &<Self::Tensor as Tensor>::Context,
    ) -> Result<Self::Tensor, Self::Error> {
        Ok(logits)
    }
}

/// Portable cardinality of one zero-based token-id domain.
///
/// Backends validate native token tensors without copying their values to the
/// host. Architectures select the exact domain for each prediction boundary.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct TokenDomain {
    cardinality: usize,
}

impl TokenDomain {
    /// Creates the token IDs `0..cardinality`.
    pub const fn new(cardinality: usize) -> Self {
        Self { cardinality }
    }

    /// Number of valid zero-based token IDs.
    pub const fn cardinality(self) -> usize {
        self.cardinality
    }
}

/// Backend primitives required by generic token-sampling policies.
///
/// The runtime owns ordering, history, adaptive state, and constraint rollback.
/// Implementations operate directly on native logits and random state without
/// copying values through a neutral tensor representation.
pub trait SamplingBackend {
    /// Backend-native logits tensor.
    type Logits: Clone;
    /// Backend-native sampled-token tensor.
    type Token: Clone;
    /// Backend-native random-key stream.
    type RandomState;
    /// Execution context, such as a stream.
    type Context: ?Sized;
    /// Backend failure.
    type Error;

    /// Creates a backend error for a portable policy or constraint failure.
    fn error(message: String) -> Self::Error;

    /// Copies an actual token descriptor under its retained host account.
    fn clone_token_with_host_source(_value:&Self::Token,_funding:&eredu_core::HostMetadataFunding,
        _context:&Self::Context)->Result<Self::Token,eredu_core::BackendFailure> {
        Err(eredu_core::HostMetadataFundingError::Unavailable.into())
    }
    /// Copies an actual logits descriptor under its retained host account.
    fn clone_logits_with_host_source(_value:&Self::Logits,_funding:&eredu_core::HostMetadataFunding,
        _context:&Self::Context)->Result<Self::Logits,eredu_core::BackendFailure> {
        Err(eredu_core::HostMetadataFundingError::Unavailable.into())
    }

    /// Copies the actual random owner under an admitted host source.
    /// A backend must qualify its native child copy separately.
    fn clone_random_with_host_source(
        _value:&Self::RandomState,_funding:&eredu_core::HostMetadataFunding,
        _context:&Self::Context,
     )->Result<Self::RandomState,eredu_core::BackendFailure> {
        Err(eredu_core::HostMetadataFundingError::Unavailable.into())
    }

    /// Validates a native token tensor against one architecture-selected domain.
    ///
    /// The returned token must retain a backend-native dependency on the range
    /// check so lazy backends cannot commit an unchecked forced token.
    fn validate_token(
        token: &Self::Token,
        domain: TokenDomain,
        context: &Self::Context,
    ) -> Result<Self::Token, Self::Error>;

    /// Scales logits by inverse temperature, preserving the native tensor.
    fn scale_temperature(
        logits: &Self::Logits,
        temperature: f32,
        context: &Self::Context,
    ) -> Result<Self::Logits, Self::Error>;

    /// Applies repetition, frequency, and presence penalties.
    fn apply_penalties(
        logits: &Self::Logits,
        history: &[u32],
        penalties: PenaltyConfig,
        context: &Self::Context,
    ) -> Result<Self::Logits, Self::Error>;

    /// Masks all but the highest `top_k` logits. Non-positive values disable it.
    fn apply_top_k(
        logits: Self::Logits,
        top_k: i32,
        context: &Self::Context,
    ) -> Result<Self::Logits, Self::Error>;

    /// Applies nucleus filtering while retaining canonical vocabulary order.
    fn apply_top_p(
        logits: Self::Logits,
        top_p: f32,
        context: &Self::Context,
    ) -> Result<Self::Logits, Self::Error>;

    /// Applies minimum-relative-probability filtering.
    fn apply_min_p(
        logits: Self::Logits,
        min_p: f32,
        context: &Self::Context,
    ) -> Result<Self::Logits, Self::Error>;

    /// Masks tokens rejected by a portable vocabulary filter.
    /// Explicit masks are closed sets: wider logits must mask the missing IDs,
    /// while narrower logits use the executable prefix. Reject an empty
    /// intersection before sampling; [`TokenFilter::allowed_mask_for`] implements
    /// this shared domain policy without backend tensors.
    fn apply_token_filter(
        logits: &Self::Logits,
        filter: &TokenFilter,
        context: &Self::Context,
    ) -> Result<Self::Logits, Self::Error>;

    /// Applies Mirostat's surprise cutoff after penalties and temperature.
    fn apply_mirostat(
        logits: &Self::Logits,
        history: &[u32],
        penalties: PenaltyConfig,
        temperature: f32,
        mu: f32,
        context: &Self::Context,
    ) -> Result<Self::Logits, Self::Error>;

    /// Selects from raw logits, applying temperature for stochastic sampling.
    fn sample_raw(
        logits: &Self::Logits,
        temperature: f32,
        random: Option<&mut Self::RandomState>,
        context: &Self::Context,
    ) -> Result<Self::Token, Self::Error>;

    /// Selects from logits already scaled by the policy.
    fn sample_processed(
        logits: &Self::Logits,
        temperature: f32,
        random: Option<&mut Self::RandomState>,
        context: &Self::Context,
    ) -> Result<Self::Token, Self::Error>;

    /// Materializes only the selected scalar token identifier.
    fn token_id(token: &Self::Token, context: &Self::Context) -> Result<u32, Self::Error>;

    /// Materializes one committed token probability from processed logits.
    fn token_probability(
        logits: &Self::Logits,
        token: u32,
        context: &Self::Context,
    ) -> Result<f32, Self::Error>;
}

/// Backend-neutral repetition/frequency/presence controls.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PenaltyConfig {
    /// Repetition multiplier; `1.0` disables it.
    pub repeat_penalty: f32,
    /// Number of recent tokens considered; negative means all.
    pub repeat_last_n: i32,
    /// Per-occurrence logit penalty.
    pub frequency_penalty: f32,
    /// One-time penalty for every present token.
    pub presence_penalty: f32,
}

impl PenaltyConfig {
    /// Returns whether every penalty is disabled.
    pub fn is_identity(self) -> bool {
        self.repeat_penalty == 1.0 && self.frequency_penalty == 0.0 && self.presence_penalty == 0.0
    }
}

impl Default for PenaltyConfig {
    fn default() -> Self {
        Self {
            repeat_penalty: 1.0,
            repeat_last_n: 64,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
        }
    }
}

/// Sampling policy suitable for lossless speculative decoding.
pub trait SpeculativeSampler<B: SamplingBackend> {
    /// Actual associated dynamic grammar state, or the uninhabited fixed type.
    /// This alone supplies neither a source nor numerical authority.
    type PreparedGrammar: eredu_core::speculative::PreparedGrammarController;

    /// Exact typed grammar, policy and provisional-copy worker. The native
    /// consumer must authenticate its source and supply paid operation funding.
    fn prepared_grammar_controller(&self) -> Option<PreparedGrammarSampler<'_, Self, Self::PreparedGrammar>>
    where Self: Sized { None }

    /// Source-bound fixed-controller decision and paid provisional transaction.
    /// This is separate from immutable grammar-free host copying. Unknown
    /// callbacks and controllers never enter through that simpler projection.
    fn prepared_controller(&self) -> Option<PreparedSpeculativeController<'_, Self>>
    where
        Self: Sized,
    {
        None
    }

    /// Closed copy of an actual known speculative policy. Unknown
    /// callbacks return None; ordinary Clone remains independent of this hook.
    fn prepared_host_copy(&self) -> Option<PreparedSpeculativeSamplerCopy<'_, Self>>
    where
        Self: Sized,
    {
        None
    }

    /// Source-bound provisional adaptive commit. Unknown mutable callbacks
    /// provide no witness and remain outside admitted execution.
    fn prepared_adaptive_commit(&self) -> Option<PreparedAdaptiveCommit<'_, Self>>
    where
        Self: Sized,
    {
        None
    }

    /// Closed projection of this policy's exact categorical worker, independently
    /// of processing, random-key provenance, and commit behavior.
    fn prepared_categorical_policy(&self) -> Option<PreparedCategoricalPolicy<'_>> {
        None
    }

    /// Independent closed projection of greedy choice and unchanged commit.
    /// A logit-processing projection never certifies either callback.
    fn prepared_greedy_policy(&self) -> Option<PreparedGreedyPolicy<'_>> {
        None
    }

    /// Immutable projection of a known deterministic worker. Unknown callbacks
    /// return None; this never certifies or invokes their `process_logits` body.
    fn prepared_logit_policy(&self) -> Option<PreparedLogitPolicy<'_>> {
        None
    }

    /// Whether prospective sampling is supported, and whether zero temperature is forbidden.
    fn control_requires_positive_temperature(&self) -> Option<bool> {
        None
    }
    /// Stages a canonical one-token restriction through the ordinary constraint policy.
    fn control_force_next(
        &mut self,
        _token: u32,
        _domain: TokenDomain,
        _position: usize,
    ) -> Result<(), eredu_core::speculative::SpeculativeControlError> {
        Err(
            eredu_core::speculative::SpeculativeControlError::Unsupported(
                "sampler has no token forcing",
            ),
        )
    }
    /// Removes an uncommitted forced choice.
    fn control_clear_forced(&mut self) -> bool {
        false
    }
    /// Current pending canonical choice.
    fn control_pending_forced(&self) -> Option<u32> {
        None
    }
    /// Complete bytes retained by a clone with isolated mutable sampling state.
    fn control_snapshot_bytes(&self) -> Option<u64> {
        None
    }

    /// Whether loaded checkpoint defaults should wrap this policy.
    fn uses_checkpoint_defaults(&self) -> bool {
        false
    }

    /// Whether optimistic draft work is an exact discardable fork.
    fn supports_exact_optimistic_promotion(&self) -> bool {
        false
    }

    /// Whether the committed generation grammar is complete.
    fn grammar_is_complete(&mut self) -> Result<bool, B::Error> {
        Ok(false)
    }

    /// Whether an uncommitted logical prefix completes the grammar.
    fn prefix_is_complete(&self, _history: &[u32]) -> Result<bool, B::Error> {
        Ok(false)
    }

    /// Applies penalties, filters, and temperature.
    fn process_logits(
        &mut self,
        logits: &B::Logits,
        temperature: f32,
        history: &[u32],
        context: &B::Context,
    ) -> Result<B::Logits, B::Error>;

    /// Captures/intervenes on raw logits with the exact pre-forcing domain, then
    /// runs ordinary processing. Unknown policies explicitly supply no domain.
    /// The callback must run once, before filtering, and must not retain logits.
    fn process_logits_with_capture(
        &mut self,
        logits: &B::Logits,
        temperature: f32,
        history: &[u32],
        context: &B::Context,
        capture: impl FnOnce(
            &B::Logits,
            Option<eredu_core::capture::CaptureTokenDomain<'_>>,
        ) -> Result<B::Logits, B::Error>,
    ) -> Result<B::Logits, B::Error>
    where
        Self: Sized,
    {
        let effective = capture(logits, None)?;
        self.process_logits(&effective, temperature, history, context)
    }

    /// Selects from already processed logits.
    fn sample_processed(
        &self,
        logits: &B::Logits,
        temperature: f32,
        random: Option<&mut B::RandomState>,
        context: &B::Context,
    ) -> Result<B::Token, B::Error> {
        B::sample_processed(logits, temperature, random, context)
    }

    /// Commits a token selected from a processed target distribution.
    fn commit_token(
        &mut self,
        _processed_logits: &B::Logits,
        _token: u32,
        _context: &B::Context,
    ) -> Result<(), B::Error> {
        Ok(())
    }
}

/// Strategy for choosing a token from model logits.
pub trait Sampler<B: SamplingBackend> {
    /// Copies exact sampler storage before using the ordinary decision worker.
    fn clone_with_host_source(&self,_funding:&eredu_core::HostMetadataFunding)
        ->Result<Self,eredu_core::HostMetadataFundingError> where Self:Sized {
        Err(eredu_core::HostMetadataFundingError::Unavailable)
    }

    /// Reserves this policy's actual host mutation before one ordinary sample.
    fn reserve_sample_with_host_source(&self,_funding:&eredu_core::HostMetadataFunding)
        ->Result<(),eredu_core::HostMetadataFundingError> {
        Err(eredu_core::HostMetadataFundingError::Unavailable)
    }


    /// Whether loaded checkpoint defaults should wrap this policy.
    fn uses_checkpoint_defaults(&self) -> bool {
        false
    }

    /// Selects one token from raw model logits.
    fn sample(
        &mut self,
        logits: &B::Logits,
        temperature: f32,
        random: Option<&mut B::RandomState>,
        context: &B::Context,
    ) -> Result<B::Token, B::Error>;
}

/// Grammar-aware wrapper around a backend-neutral sampling policy.
pub struct ConstrainedSampler<S, C> {
    policy: S,
    controller: TokenChoiceController<C>,
}

fn commit_components<P, C, E>(
    policy: &mut P,
    controller: &mut C,
    token: u32,
    commit_policy: impl FnOnce(&mut P, u32) -> Result<(), E>,
    commit_controller: impl FnOnce(&mut C, u32) -> Result<(), E>,
) -> Result<(), E> {
    commit_policy(policy, token)?;
    commit_controller(controller, token)
}

struct ConstraintCheckpoint<S, C> {
    policy: S,
    controller: TokenChoiceController<C>,
}

impl<S: Clone, C: Clone> Clone for ConstrainedSampler<S, C> {
    fn clone(&self) -> Self {
        Self {
            policy: self.policy.clone(),
            controller: self.controller.clone(),
        }
    }
}

impl<S, C: TokenFilterController> ConstrainedSampler<S, C> {
    /// Wraps a policy with a portable canonical constraint controller.
    pub fn new(policy: S, controller: C) -> Self {
        Self {
            policy,
            controller: TokenChoiceController::new(controller, TokenDomain::new(0)),
        }
    }

    /// Returns the wrapped policy.
    pub const fn policy(&self) -> &S {
        &self.policy
    }

    /// Returns the portable constraint controller.
    pub fn controller(&self) -> &C {
        self.controller.inner()
    }

    /// Returns the portable constraint controller mutably.
    pub fn controller_mut(&mut self) -> &mut C {
        self.controller.inner_mut()
    }
}

impl<S: Clone, C: Clone> ConstrainedSampler<S, C> {
    fn checkpoint(&self) -> ConstraintCheckpoint<S, C> {
        ConstraintCheckpoint {
            policy: self.policy.clone(),
            controller: self.controller.clone(),
        }
    }
}

impl<B, S, C> SpeculativeSampler<B> for ConstrainedSampler<S, C>
where
    B: SamplingBackend,
    S: SpeculativeSampler<B> + Clone,
    C: SpeculativeTokenFilterController,
{
    type PreparedGrammar = C::PreparedGrammar;
    fn prepared_grammar_controller(&self) -> Option<PreparedGrammarSampler<'_, Self, Self::PreparedGrammar>> {
        prepared_grammar::constrained::<B, S, C>(self)
    }
    fn prepared_controller(&self) -> Option<PreparedSpeculativeController<'_, Self>> {
        prepared_controller::constrained::<B, S, C>(self)
    }
    fn prepared_adaptive_commit(&self) -> Option<PreparedAdaptiveCommit<'_, Self>> {
        adaptive_commit::constrained::<B, S, C>(self)
    }
    fn control_requires_positive_temperature(&self) -> Option<bool> {
        self.policy.control_requires_positive_temperature()
    }
    fn control_force_next(
        &mut self,
        token: u32,
        domain: TokenDomain,
        position: usize,
    ) -> Result<(), eredu_core::speculative::SpeculativeControlError> {
        use crate::execution_control::TokenChoiceError;
        use eredu_core::speculative::SpeculativeControlError;
        self.controller
            .force_at(token, domain, position)
            .map_err(|error| match error {
                TokenChoiceError::InvalidToken(token) => {
                    SpeculativeControlError::InvalidToken(token)
                }
                TokenChoiceError::Forbidden(token) => {
                    SpeculativeControlError::ForbiddenToken(token)
                }
                TokenChoiceError::AlreadyPending => SpeculativeControlError::PendingToken,
                TokenChoiceError::Constraint(error) => SpeculativeControlError::backend(error),
                TokenChoiceError::UnexpectedCommit { .. } => {
                    SpeculativeControlError::Invalid("inconsistent forced choice")
                }
            })
    }
    fn control_clear_forced(&mut self) -> bool {
        self.controller.clear_forced()
    }
    fn control_pending_forced(&self) -> Option<u32> {
        self.controller.pending_forced()
    }
    fn control_snapshot_bytes(&self) -> Option<u64> {
        self.policy
            .control_snapshot_bytes()?
            .checked_add(self.controller.control_snapshot_bytes()?)?
            .checked_add(std::mem::size_of::<Self>() as u64)
    }

    fn supports_exact_optimistic_promotion(&self) -> bool {
        self.policy.supports_exact_optimistic_promotion()
    }

    fn grammar_is_complete(&mut self) -> Result<bool, B::Error> {
        self.controller
            .is_complete()
            .map_err(|error| B::error(error.to_string()))
    }

    fn prefix_is_complete(&self, history: &[u32]) -> Result<bool, B::Error> {
        self.controller
            .prefix_is_complete(history)
            .map_err(|error| B::error(error.to_string()))
    }

    fn process_logits(
        &mut self,
        logits: &B::Logits,
        temperature: f32,
        history: &[u32],
        context: &B::Context,
    ) -> Result<B::Logits, B::Error> {
        let filter = self
            .controller
            .filter_at(history)
            .map_err(|error| B::error(error.to_string()))?;
        let masked = B::apply_token_filter(logits, &filter, context)?;
        self.policy
            .process_logits(&masked, temperature, history, context)
    }

    fn process_logits_with_capture(
        &mut self,
        logits: &B::Logits,
        temperature: f32,
        history: &[u32],
        context: &B::Context,
        capture: impl FnOnce(
            &B::Logits,
            Option<eredu_core::capture::CaptureTokenDomain<'_>>,
        ) -> Result<B::Logits, B::Error>,
    ) -> Result<B::Logits, B::Error> {
        let decision = self
            .controller
            .decision_at(history)
            .map_err(|error| B::error(error.to_string()))?;
        let effective = capture(logits, decision.capture_domain())?;
        let masked = B::apply_token_filter(&effective, decision.filter(), context)?;
        self.policy
            .process_logits(&masked, temperature, history, context)
    }

    fn sample_processed(
        &self,
        logits: &B::Logits,
        temperature: f32,
        random: Option<&mut B::RandomState>,
        context: &B::Context,
    ) -> Result<B::Token, B::Error> {
        self.policy
            .sample_processed(logits, temperature, random, context)
    }

    fn commit_token(
        &mut self,
        processed_logits: &B::Logits,
        token: u32,
        context: &B::Context,
    ) -> Result<(), B::Error> {
        let checkpoint = self.checkpoint();
        if let Err(error) = commit_components(
            &mut self.policy,
            &mut self.controller,
            token,
            |policy, token| policy.commit_token(processed_logits, token, context),
            |controller, token| {
                controller
                    .commit_token(token)
                    .map_err(|error| B::error(error.to_string()))
            },
        ) {
            self.policy = checkpoint.policy;
            self.controller = checkpoint.controller;
            return Err(error);
        }
        Ok(())
    }
}

impl<B, S, C> Sampler<B> for ConstrainedSampler<S, C>
where
    B: SamplingBackend,
    S: Sampler<B> + Clone,
    C: TokenFilterController + Clone,
{
    fn sample(
        &mut self,
        logits: &B::Logits,
        temperature: f32,
        random: Option<&mut B::RandomState>,
        context: &B::Context,
    ) -> Result<B::Token, B::Error> {
        let checkpoint = self.checkpoint();
        let filter = self
            .controller
            .current_filter()
            .map_err(|error| B::error(error.to_string()))?;
        let masked = B::apply_token_filter(logits, &filter, context)?;
        let token = self.policy.sample(&masked, temperature, random, context)?;
        let token_id = B::token_id(&token, context)?;
        if let Err(error) = self.controller.commit_token(token_id) {
            self.policy = checkpoint.policy;
            self.controller = checkpoint.controller;
            return Err(B::error(error.to_string()));
        }
        Ok(token)
    }
}

/// Stateless greedy/categorical sampler.
#[derive(Debug, Clone, Copy)]
pub struct DefaultSampler;

impl<B: SamplingBackend> SpeculativeSampler<B> for DefaultSampler {
    type PreparedGrammar = eredu_core::speculative::NoPreparedGrammar;
    fn prepared_host_copy(&self) -> Option<PreparedSpeculativeSamplerCopy<'_, Self>> {
        Some(speculative_copy::default(self))
    }
    fn prepared_categorical_policy(&self) -> Option<PreparedCategoricalPolicy<'_>> {
        Some(PreparedCategoricalPolicy::default(self))
    }
    fn prepared_greedy_policy(&self) -> Option<PreparedGreedyPolicy<'_>> {
        Some(PreparedGreedyPolicy::default(self))
    }
    fn prepared_logit_policy(&self) -> Option<PreparedLogitPolicy<'_>> {
        Some(PreparedLogitPolicy::default(self))
    }
    fn control_requires_positive_temperature(&self) -> Option<bool> {
        Some(false)
    }
    fn uses_checkpoint_defaults(&self) -> bool {
        true
    }

    fn supports_exact_optimistic_promotion(&self) -> bool {
        true
    }

    fn process_logits(
        &mut self,
        logits: &B::Logits,
        temperature: f32,
        _history: &[u32],
        context: &B::Context,
    ) -> Result<B::Logits, B::Error> {
        prepared_logits::default_process::<B>(logits, temperature, context)
    }
}

impl<B: SamplingBackend> Sampler<B> for DefaultSampler {
    fn uses_checkpoint_defaults(&self) -> bool {
        true
    }

    fn sample(
        &mut self,
        logits: &B::Logits,
        temperature: f32,
        random: Option<&mut B::RandomState>,
        context: &B::Context,
    ) -> Result<B::Token, B::Error> {
        B::sample_raw(logits, temperature, random, context)
    }
}

/// Configurable backend-neutral text sampler.
#[derive(Debug, Clone)]
pub struct GenerationSampler {
    /// Keep only the `top_k` highest-logit tokens when positive.
    pub top_k: i32,
    /// Nucleus probability mass.
    pub top_p: f32,
    /// Minimum probability relative to the most probable token.
    pub min_p: f32,
    /// Repetition multiplier.
    pub repeat_penalty: f32,
    /// Number of recent tokens considered by penalties.
    pub repeat_last_n: i32,
    /// Per-occurrence penalty.
    pub frequency_penalty: f32,
    /// One-time presence penalty.
    pub presence_penalty: f32,
    generated_tokens: history::TokenHistory,
}

impl Default for GenerationSampler {
    fn default() -> Self {
        Self {
            top_k: 40,
            top_p: 0.95,
            min_p: 0.05,
            repeat_penalty: 1.0,
            repeat_last_n: 64,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            generated_tokens: history::TokenHistory::default(),
        }
    }
}

impl GenerationSampler {
    /// Creates a sampler with default controls.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a sampler from a resolved portable generation configuration.
    pub fn from_resolved(config: ResolvedGenerationConfig) -> Self {
        Self::new()
            .top_k(config.top_k)
            .top_p(config.top_p)
            .min_p(config.min_p)
            .penalties(
                config.repetition_penalty,
                config.repeat_last_n,
                config.frequency_penalty,
                config.presence_penalty,
            )
    }

    /// Seeds accepted-token history.
    pub fn with_generated_tokens(mut self, tokens: impl IntoIterator<Item = u32>) -> Self {
        self.generated_tokens = history::TokenHistory::from_tokens(tokens);
        self
    }

    /// Sets top-k filtering.
    pub fn top_k(mut self, value: i32) -> Self {
        self.top_k = value;
        self
    }

    /// Sets nucleus filtering.
    pub fn top_p(mut self, value: f32) -> Self {
        self.top_p = value;
        self
    }

    /// Sets minimum-relative-probability filtering.
    pub fn min_p(mut self, value: f32) -> Self {
        self.min_p = value;
        self
    }

    /// Sets repetition, frequency, and presence penalties.
    pub fn penalties(
        mut self,
        repeat_penalty: f32,
        repeat_last_n: i32,
        frequency_penalty: f32,
        presence_penalty: f32,
    ) -> Self {
        self.repeat_penalty = repeat_penalty;
        self.repeat_last_n = repeat_last_n;
        self.frequency_penalty = frequency_penalty;
        self.presence_penalty = presence_penalty;
        self
    }

    /// Returns accepted-token history.
    pub fn generated_tokens(&self) -> &[u32] {
        self.generated_tokens.as_slice()
    }

    /// Replaces accepted-token history.
    pub fn set_generated_tokens(&mut self, tokens: impl IntoIterator<Item = u32>) {
        self.generated_tokens = history::TokenHistory::from_tokens(tokens);
    }

    /// Records a token accepted outside this sampler.
    pub fn accept_token(&mut self, token: u32) {
        self.generated_tokens.push(token);
    }

    /// Clears accepted-token history.
    pub fn clear_generated_tokens(&mut self) {
        self.generated_tokens.clear();
    }

    /// Returns the portable penalty controls.
    pub const fn penalty_config(&self) -> PenaltyConfig {
        PenaltyConfig {
            repeat_penalty: self.repeat_penalty,
            repeat_last_n: self.repeat_last_n,
            frequency_penalty: self.frequency_penalty,
            presence_penalty: self.presence_penalty,
        }
    }

    fn process_for<B: SamplingBackend>(
        &self,
        logits: &B::Logits,
        history: &[u32],
        context: &B::Context,
    ) -> Result<B::Logits, B::Error> {
        self.process_with::<B>(logits, context, |logits, penalties, context| {
            B::apply_penalties(logits, history, penalties, context)
        })
    }

    // The numerical sampler and extent-only workspace projection share filter
    // ordering. Only the closed penalty primitive differs in its history input.
    pub(crate) fn process_with<B: SamplingBackend>(
        &self,
        logits: &B::Logits,
        context: &B::Context,
        penalties: impl FnOnce(&B::Logits, PenaltyConfig, &B::Context) -> Result<B::Logits, B::Error>,
    ) -> Result<B::Logits, B::Error> {
        prepared_logits::Standard::from_source(self).process::<B>(logits, context, penalties)
    }
}

impl<B: SamplingBackend> SpeculativeSampler<B> for GenerationSampler {
    type PreparedGrammar = eredu_core::speculative::NoPreparedGrammar;
    fn prepared_host_copy(&self) -> Option<PreparedSpeculativeSamplerCopy<'_, Self>> {
        speculative_copy::standard(self)
    }
    fn prepared_categorical_policy(&self) -> Option<PreparedCategoricalPolicy<'_>> {
        Some(PreparedCategoricalPolicy::standard(self))
    }
    fn prepared_greedy_policy(&self) -> Option<PreparedGreedyPolicy<'_>> {
        Some(PreparedGreedyPolicy::standard(self))
    }
    fn prepared_logit_policy(&self) -> Option<PreparedLogitPolicy<'_>> {
        Some(PreparedLogitPolicy::standard(self))
    }
    fn control_requires_positive_temperature(&self) -> Option<bool> {
        Some(false)
    }
    fn supports_exact_optimistic_promotion(&self) -> bool {
        true
    }

    fn process_logits(
        &mut self,
        logits: &B::Logits,
        temperature: f32,
        history: &[u32],
        context: &B::Context,
    ) -> Result<B::Logits, B::Error> {
        let logits = self.process_for::<B>(logits, history, context)?;
        prepared_logits::finish_standard::<B>(logits, temperature, context)
    }
}

impl GenerationSampler {
    /// Exact one-sample history growth used by the paid ordinary sampler.
    pub fn host_sample_bytes(&self)->Option<usize> {
        let backing=usize::try_from(self.generated_tokens.push_payload_bytes()?).ok()?;
        let parts=[std::mem::size_of::<Box<[u32]>>(),std::mem::size_of::<Vec<u32>>(),
            std::mem::size_of::<(usize,u32)>(),eredu_core::HostMetadataFunding::reservation_control_bytes()];
        parts.into_iter().try_fold(std::mem::size_of_val(&parts).checked_add(backing)?,usize::checked_add)
    }
    /// Source-derived exact history-copy and sampler-shell population used by
    /// the shared paid clone hook. This query creates no history destination.
    pub fn host_clone_bytes(&self)->Option<usize> {
        let source=self.generated_tokens.prepare_copy()?;
        let parts=[std::mem::size_of::<Self>()*2,
            std::mem::size_of::<history::TokenHistoryCopy<'_>>(),
            std::mem::size_of::<Result<Self,eredu_core::HostMetadataFundingError>>(),
            eredu_core::HostMetadataFunding::reservation_control_bytes()];
        parts.into_iter().try_fold(std::mem::size_of_val(&parts)
            .checked_add(usize::try_from(source.bytes()).ok()?)?,usize::checked_add)
    }
}

impl<B: SamplingBackend> Sampler<B> for GenerationSampler {
    fn clone_with_host_source(&self,funding:&eredu_core::HostMetadataFunding)
        ->Result<Self,eredu_core::HostMetadataFundingError> {
        let source=self.generated_tokens.prepare_copy().ok_or(eredu_core::HostMetadataFundingError::Overflow)?;
        let bytes=self.host_clone_bytes().ok_or(eredu_core::HostMetadataFundingError::Overflow)?;
        funding.reserve_metadata(bytes)?;
        Ok(self.copy_with_history(source.copy()))
    }

    fn reserve_sample_with_host_source(&self,funding:&eredu_core::HostMetadataFunding)
        ->Result<(),eredu_core::HostMetadataFundingError> {
        funding.reserve_metadata(self.host_sample_bytes()
            .ok_or(eredu_core::HostMetadataFundingError::Overflow)?)
    }

    fn sample(
        &mut self,
        logits: &B::Logits,
        temperature: f32,
        random: Option<&mut B::RandomState>,
        context: &B::Context,
    ) -> Result<B::Token, B::Error> {
        let logits = self.process_for::<B>(logits, self.generated_tokens.as_slice(), context)?;
        let token = B::sample_raw(&logits, temperature, random, context)?;
        self.generated_tokens.push(B::token_id(&token, context)?);
        Ok(token)
    }
}

/// Adaptive Mirostat V2 policy with backend-neutral state.
#[derive(Debug, Clone)]
pub struct MirostatV2Sampler {
    tau: f32,
    eta: f32,
    mu: f32,
    penalties: GenerationSampler,
}

impl Default for MirostatV2Sampler {
    fn default() -> Self {
        Self {
            tau: 5.0,
            eta: 0.1,
            mu: 10.0,
            penalties: GenerationSampler::new().top_k(0).top_p(1.0).min_p(0.0),
        }
    }
}

impl MirostatV2Sampler {
    /// Creates a sampler targeting `tau` bits of surprise.
    pub fn new(tau: f32, eta: f32) -> Result<Self, SamplingConfigurationError> {
        validate_positive_finite("Mirostat V2 tau", tau)?;
        validate_positive_finite("Mirostat V2 eta", eta)?;
        Ok(Self {
            tau,
            eta,
            mu: 2.0 * tau,
            penalties: GenerationSampler::new().top_k(0).top_p(1.0).min_p(0.0),
        })
    }

    /// Sets penalties applied before adaptive truncation.
    pub fn penalties(
        mut self,
        repeat_penalty: f32,
        repeat_last_n: i32,
        frequency_penalty: f32,
        presence_penalty: f32,
    ) -> Self {
        self.penalties = self.penalties.penalties(
            repeat_penalty,
            repeat_last_n,
            frequency_penalty,
            presence_penalty,
        );
        self
    }

    /// Target surprise in bits.
    pub const fn tau(&self) -> f32 {
        self.tau
    }

    /// Adaptation rate.
    pub const fn eta(&self) -> f32 {
        self.eta
    }

    /// Current adaptive surprise limit.
    pub const fn mu(&self) -> f32 {
        self.mu
    }

    /// Accepted-token history.
    pub fn generated_tokens(&self) -> &[u32] {
        self.penalties.generated_tokens()
    }

    /// Records an externally accepted token and its normalized probability.
    pub fn accept_token(
        &mut self,
        token: u32,
        probability: f32,
    ) -> Result<(), SamplingConfigurationError> {
        self.accept_token_fixed(token, probability)
            .map_err(|cause| SamplingConfigurationError::Invalid(cause.to_string()))
    }

    /// Resets adaptive state and history.
    pub fn reset(&mut self) {
        self.mu = 2.0 * self.tau;
        self.penalties.clear_generated_tokens();
    }

    pub(crate) fn next_mu(
        &self,
        mu: f32,
        probability: f32,
    ) -> Result<f32, SamplingConfigurationError> {
        self.next_mu_fixed(mu, probability)
            .map_err(|cause| SamplingConfigurationError::Invalid(cause.to_string()))
    }

    fn next_mu_fixed(&self, mu: f32, probability: f32) -> Result<f32, PreparedAdaptiveCommitError> {
        adaptive_commit::validate_probability(probability)?;
        Ok(mu - self.eta * (-probability.log2() - self.tau))
    }

    fn accept_token_fixed(
        &mut self,
        token: u32,
        probability: f32,
    ) -> Result<(), PreparedAdaptiveCommitError> {
        self.mu = self.next_mu_fixed(self.mu, probability)?;
        self.penalties.accept_token(token);
        Ok(())
    }

    fn process_for<B: SamplingBackend>(
        &self,
        logits: &B::Logits,
        temperature: f32,
        history: &[u32],
        context: &B::Context,
    ) -> Result<B::Logits, B::Error> {
        self.process_with::<B>(
            logits,
            temperature,
            self.mu,
            context,
            |logits, penalties, temperature, mu, context| {
                B::apply_mirostat(logits, history, penalties, temperature, mu, context)
            },
        )
    }

    pub(crate) fn process_with<B: SamplingBackend>(
        &self,
        logits: &B::Logits,
        temperature: f32,
        mu: f32,
        context: &B::Context,
        cutoff: impl FnOnce(
            &B::Logits,
            PenaltyConfig,
            f32,
            f32,
            &B::Context,
        ) -> Result<B::Logits, B::Error>,
    ) -> Result<B::Logits, B::Error> {
        prepared_logits::adaptive_process::<B>(
            logits,
            self.penalties.penalty_config(),
            temperature,
            mu,
            context,
            cutoff,
        )
    }

    fn commit_for<B: SamplingBackend>(
        &mut self,
        logits: &B::Logits,
        token: u32,
        context: &B::Context,
    ) -> Result<(), B::Error> {
        let probability = B::token_probability(logits, token, context)?;
        self.accept_token(token, probability)
            .map_err(|error| B::error(error.to_string()))
    }
}

impl<B: SamplingBackend> Sampler<B> for MirostatV2Sampler {
    fn sample(
        &mut self,
        logits: &B::Logits,
        temperature: f32,
        random: Option<&mut B::RandomState>,
        context: &B::Context,
    ) -> Result<B::Token, B::Error> {
        let processed = self.process_for::<B>(
            logits,
            temperature,
            self.penalties.generated_tokens(),
            context,
        )?;
        let token = B::sample_processed(&processed, temperature, random, context)?;
        self.commit_for::<B>(&processed, B::token_id(&token, context)?, context)?;
        Ok(token)
    }
}

impl<B: SamplingBackend> SpeculativeSampler<B> for MirostatV2Sampler {
    type PreparedGrammar = eredu_core::speculative::NoPreparedGrammar;
    fn prepared_host_copy(&self) -> Option<PreparedSpeculativeSamplerCopy<'_, Self>> {
        speculative_copy::adaptive(self)
    }
    fn prepared_adaptive_commit(&self) -> Option<PreparedAdaptiveCommit<'_, Self>> {
        adaptive_commit::adaptive(self)
    }
    fn control_snapshot_bytes(&self) -> Option<u64> {
        u64::try_from(std::mem::size_of::<Self>())
            .ok()?
            .checked_add(self.penalties.generated_tokens.prepare_copy()?.bytes())
    }
    fn prepared_categorical_policy(&self) -> Option<PreparedCategoricalPolicy<'_>> {
        Some(PreparedCategoricalPolicy::adaptive(self))
    }
    fn prepared_logit_policy(&self) -> Option<PreparedLogitPolicy<'_>> {
        Some(PreparedLogitPolicy::adaptive(self))
    }
    fn control_requires_positive_temperature(&self) -> Option<bool> {
        Some(true)
    }
    fn process_logits(
        &mut self,
        logits: &B::Logits,
        temperature: f32,
        history: &[u32],
        context: &B::Context,
    ) -> Result<B::Logits, B::Error> {
        self.process_for::<B>(logits, temperature, history, context)
    }

    fn commit_token(
        &mut self,
        processed_logits: &B::Logits,
        token: u32,
        context: &B::Context,
    ) -> Result<(), B::Error> {
        self.commit_for::<B>(processed_logits, token, context)
    }
}

/// Invalid backend-neutral sampling configuration.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
pub enum SamplingConfigurationError {
    /// A numeric or probability control is invalid.
    #[error("{0}")]
    Invalid(String),
}

fn validate_positive_finite(name: &str, value: f32) -> Result<(), SamplingConfigurationError> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(SamplingConfigurationError::Invalid(format!(
            "{name} must be finite and greater than zero"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::{GenerationSampler, MirostatV2Sampler};

    #[test]
    fn generation_history_is_backend_neutral() {
        let mut sampler = GenerationSampler::new().with_generated_tokens([1, 2]);
        sampler.accept_token(3);
        assert_eq!(sampler.generated_tokens(), &[1, 2, 3]);
        sampler.set_generated_tokens([5, 8]);
        assert_eq!(sampler.generated_tokens(), &[5, 8]);
        sampler.clear_generated_tokens();
        assert!(sampler.generated_tokens().is_empty());
    }

    #[test]
    fn mirostat_state_is_backend_neutral() {
        let mut sampler = MirostatV2Sampler::default();
        sampler.accept_token(42, 2.0f32.powi(-7)).unwrap();
        assert!((sampler.mu() - 9.8).abs() < 1e-6);
        assert_eq!(sampler.generated_tokens(), &[42]);
        sampler.reset();
        assert_eq!(sampler.mu(), 10.0);
        assert!(sampler.generated_tokens().is_empty());
    }
}

mod token_mask;
pub use token_mask::{TokenMaskError, TokenMaskPlan};

mod synchronization;
pub use synchronization::{
    SamplingSynchronizationDriver, SamplingSynchronizationError, SamplingSynchronizationPlan,
    SynchronizedSampling, synchronize_sampling,
};
