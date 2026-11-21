//! Dense random draws and supplied-draw categorical sampling over retained session randomness.

use eredu_core::sampling::{
    validate_dense_draw_shape, DenseSamplingBackend, DenseSamplingPolicyError,
};

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
