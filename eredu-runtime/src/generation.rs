//! Backend-neutral causal-model and token-sampling contracts.

use eredu_core::{
    generation::ResolvedGenerationConfig, SpeculativeTokenFilterController, TokenFilter,
    TokenFilterController,
};
use eredu_nn::Tensor;

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
    controller: C,
}

struct ConstraintCheckpoint<S, C> {
    policy: S,
    controller: C,
}

impl<S: Clone, C: Clone> Clone for ConstrainedSampler<S, C> {
    fn clone(&self) -> Self {
        Self {
            policy: self.policy.clone(),
            controller: self.controller.clone(),
        }
    }
}

impl<S, C> ConstrainedSampler<S, C> {
    /// Wraps a policy with a portable canonical constraint controller.
    pub fn new(policy: S, controller: C) -> Self {
        Self { policy, controller }
    }

    /// Returns the wrapped policy.
    pub const fn policy(&self) -> &S {
        &self.policy
    }

    /// Returns the portable constraint controller.
    pub const fn controller(&self) -> &C {
        &self.controller
    }

    /// Returns the portable constraint controller mutably.
    pub fn controller_mut(&mut self) -> &mut C {
        &mut self.controller
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
        if let Err(error) = self
            .policy
            .commit_token(processed_logits, token, context)
            .and_then(|()| {
                self.controller
                    .commit_token(token)
                    .map_err(|error| B::error(error.to_string()))
            })
        {
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
        if temperature == 0.0 {
            Ok(logits.clone())
        } else {
            B::scale_temperature(logits, temperature, context)
        }
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
    generated_tokens: Vec<u32>,
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
            generated_tokens: Vec::new(),
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
        self.generated_tokens = tokens.into_iter().collect();
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
        &self.generated_tokens
    }

    /// Replaces accepted-token history.
    pub fn set_generated_tokens(&mut self, tokens: impl IntoIterator<Item = u32>) {
        self.generated_tokens = tokens.into_iter().collect();
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
        let logits = B::apply_penalties(logits, history, self.penalty_config(), context)?;
        let logits = B::apply_top_k(logits, self.top_k, context)?;
        let logits = B::apply_top_p(logits, self.top_p, context)?;
        B::apply_min_p(logits, self.min_p, context)
    }
}

impl<B: SamplingBackend> SpeculativeSampler<B> for GenerationSampler {
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
        if temperature == 0.0 {
            Ok(logits)
        } else {
            B::scale_temperature(&logits, temperature, context)
        }
    }
}

impl<B: SamplingBackend> Sampler<B> for GenerationSampler {
    fn sample(
        &mut self,
        logits: &B::Logits,
        temperature: f32,
        random: Option<&mut B::RandomState>,
        context: &B::Context,
    ) -> Result<B::Token, B::Error> {
        let logits = self.process_for::<B>(logits, &self.generated_tokens, context)?;
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
        if !probability.is_finite() || probability <= 0.0 || probability > 1.0 {
            return Err(SamplingConfigurationError::Invalid(
                "accepted Mirostat V2 token probability must be finite and in (0, 1]".into(),
            ));
        }
        self.update_mu(-probability.log2());
        self.penalties.accept_token(token);
        Ok(())
    }

    /// Resets adaptive state and history.
    pub fn reset(&mut self) {
        self.mu = 2.0 * self.tau;
        self.penalties.clear_generated_tokens();
    }

    fn update_mu(&mut self, observed_surprise: f32) {
        self.mu -= self.eta * (observed_surprise - self.tau);
    }

    fn process_for<B: SamplingBackend>(
        &self,
        logits: &B::Logits,
        temperature: f32,
        history: &[u32],
        context: &B::Context,
    ) -> Result<B::Logits, B::Error> {
        if !temperature.is_finite() || temperature <= 0.0 {
            return Err(B::error(
                "Mirostat V2 requires a finite temperature greater than zero".into(),
            ));
        }
        B::apply_mirostat(
            logits,
            history,
            self.penalties.penalty_config(),
            temperature,
            self.mu,
            context,
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
