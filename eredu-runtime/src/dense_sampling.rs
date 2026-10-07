//! Dense random draws and supplied-draw categorical sampling over retained session randomness.

use crate::SamplingBackend;

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

/// Failure in transition-local dense sampling, preserving the backend source.
#[derive(Debug, thiserror::Error)]
pub enum DenseSamplingError<E> {
    /// The requested dense distribution or geometry is invalid.
    #[error(transparent)]
    Policy(#[from] DenseSamplingPolicyError),
    /// The session has no retained random stream; an explicit seed is required.
    #[error("dense sampling requires a retained random stream")]
    MissingRandomState,
    /// A native draw or numerical operation failed.
    #[error("dense sampling backend failed: {0}")]
    Backend(#[source] E),
}

/// Validates and proposes one uniform draw on an independently cloned stream.
/// The owning unpublished transition retains the candidate only after success.
pub(crate) fn draw_uniform<B: DenseSamplingBackend>(
    random: &mut Option<B::RandomState>,
    shape: &[i32],
    context: &B::Context,
) -> Result<B::Logits, DenseSamplingError<B::Error>>
where
    B::RandomState: Clone,
{
    draw::<B>(random, shape, context, B::uniform_f32)
}

/// Validates and proposes one standard-normal draw using the same transaction.
pub(crate) fn draw_standard_normal<B: DenseSamplingBackend>(
    random: &mut Option<B::RandomState>,
    shape: &[i32],
    context: &B::Context,
) -> Result<B::Logits, DenseSamplingError<B::Error>>
where
    B::RandomState: Clone,
{
    draw::<B>(random, shape, context, B::standard_normal_f32)
}

fn draw<B: DenseSamplingBackend>(
    random: &mut Option<B::RandomState>,
    shape: &[i32],
    context: &B::Context,
    operation: impl FnOnce(&[i32], &mut B::RandomState, &B::Context) -> Result<B::Logits, B::Error>,
) -> Result<B::Logits, DenseSamplingError<B::Error>>
where
    B::RandomState: Clone,
{
    validate_dense_draw_shape(shape)?;
    let mut candidate = random
        .clone()
        .ok_or(DenseSamplingError::MissingRandomState)?;
    let output = operation(shape, &mut candidate, context).map_err(DenseSamplingError::Backend)?;
    *random = Some(candidate);
    Ok(output)
}
