//! Exact numerical operations for portable token-sampling policies.

/// Portable host sampling arithmetic.
pub mod host;
/// Explicit seeded host random streams.
pub mod host_random;
/// Tensor adapter for host sampling.
pub mod host_tensor;

use crate::TokenFilter;

/// Native allocation of independent, seed-addressable random storage.
pub trait SeededRandomKey: crate::BackendProvider {
    /// The backend's retained random state.
    type Key;

    /// Materializes one key from the exact supplied seed.
    fn seeded_random(seed: u64) -> Result<Self::Key, Self::Error>;
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
/// Implementations operate on the score row and random stream supplied by the
/// selected sampling route. A device implementation retains native storage,
/// while a host implementation consumes an explicitly acquired host row.
pub trait SamplingBackend {
    /// Describes sampling storage without tensors, random draws or state changes.
    /// Backends without implementation facts preserve explicit incomplete coverage.
    fn mechanism_memory(
        invocation: &eredu_nn::mechanism_memory::MechanismInvocation,
    ) -> Result<eredu_nn::mechanism_memory::MechanismMemoryContract, eredu_nn::Error> {
        eredu_nn::mechanism_memory::MechanismMemoryContract::unknown(invocation)
    }

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

    /// Materializes a fixed group of selected scalar IDs. Backends may evaluate
    /// and transfer the group together; order and scalar-token errors are retained.
    fn token_ids<const N: usize>(
        tokens: [&Self::Token; N],
        context: &Self::Context,
    ) -> Result<[u32; N], Self::Error> {
        let mut ids = [0; N];
        for (id, token) in ids.iter_mut().zip(tokens) {
            *id = Self::token_id(token, context)?;
        }
        Ok(ids)
    }

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

/// Filtering before Gumbel-Max categorical selection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GumbelTruncation {
    /// Retain every candidate.
    None,
    /// Retain candidates at least as large as the kth logit, including ties.
    TopK(usize),
    /// Sort ascending and remove cumulative probability `<= 1 - p`, keeping
    /// at least the largest logit. The threshold is formed in F64 before its
    /// F32 numerical realization, matching the declared probability value.
    TopP(f64),
}
impl GumbelTruncation {
    /// Validates the portable categorical filtering policy without a draw.
    pub fn validate(self) -> Result<(), DenseSamplingPolicyError> {
        match self {
            Self::TopK(0) => Err(DenseSamplingPolicyError::InvalidTruncation),
            Self::TopP(p) if !p.is_finite() || !(0.0..=1.0).contains(&p) => {
                Err(DenseSamplingPolicyError::InvalidTruncation)
            }
            _ => Ok(()),
        }
    }
}

/// A draw or filtering request rejected before random state advances.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum DenseSamplingPolicyError {
    /// Dense random output must have positive dimensions and an addressable F32 allocation.
    #[error("dense random shape must have positive dimensions and an addressable F32 extent")]
    InvalidShape,
    /// Top-k must be positive and top-p a finite probability in `[0, 1]`.
    #[error("invalid Gumbel categorical truncation")]
    InvalidTruncation,
    /// The Gumbel logarithm stabilizer must be finite and strictly between zero and one.
    #[error("Gumbel epsilon must lie strictly between zero and one")]
    InvalidEpsilon,
}

/// Validates output geometry and returns its scalar count, without allocation.
/// An empty shape denotes one scalar; zero-sized axes are rejected.
pub fn validate_dense_draw_shape(shape: &[i32]) -> Result<usize, DenseSamplingPolicyError> {
    shape.iter().try_fold(1usize, |count, &dimension| {
        let dimension = usize::try_from(dimension)
            .ok()
            .filter(|&n| n > 0)
            .ok_or(DenseSamplingPolicyError::InvalidShape)?;
        count
            .checked_mul(dimension)
            .filter(|&n| n <= isize::MAX as usize / std::mem::size_of::<f32>())
            .ok_or(DenseSamplingPolicyError::InvalidShape)
    })
}

/// Validates the Gumbel logarithm stabilizer before consuming randomness.
pub fn validate_gumbel_epsilon(epsilon: f32) -> Result<(), DenseSamplingPolicyError> {
    if epsilon.is_finite() && epsilon > 0.0 && epsilon < 1.0 {
        Ok(())
    } else {
        Err(DenseSamplingPolicyError::InvalidEpsilon)
    }
}

/// Optional numerical capability sharing the ordinary sampler's random state.
///
/// `Logits` is the backend's floating tensor handle and also represents these
/// dense F32 payloads. This extension introduces neither a scheduler nor a second
/// RNG. All random primitives use explicit retained keys; supplied-draw selection
/// consumes no randomness. Backends validate shape and policy before a draw.
pub trait DenseSamplingBackend: SamplingBackend {
    /// Draws independent F32 uniforms in `[0,1)` with the requested shape.
    fn uniform_f32(
        shape: &[i32],
        random: &mut Self::RandomState,
        context: &Self::Context,
    ) -> Result<Self::Logits, Self::Error>;
    /// Draws independent F32 standard normals with the requested shape.
    fn standard_normal_f32(
        shape: &[i32],
        random: &mut Self::RandomState,
        context: &Self::Context,
    ) -> Result<Self::Logits, Self::Error>;
    /// Side-effect-free shape facts used to request one uniform per logit.
    fn dense_shape(value: &Self::Logits) -> &[i32];
    /// Selects the maximum of filtered F32 log-probabilities plus
    /// `-ln(-ln(uniform + epsilon) + epsilon)` along the last axis.
    ///
    /// Uniform values must be in `[0,1)` and match the logits' shape. Returns
    /// one native token per row; does not publish it or update sampler history.
    fn gumbel_max(
        logits: &Self::Logits,
        uniform: &Self::Logits,
        truncation: GumbelTruncation,
        epsilon: f32,
        context: &Self::Context,
    ) -> Result<Self::Token, Self::Error>;
}
