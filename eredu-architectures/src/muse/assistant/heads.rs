//! Completion-safe individual assistant-head operations over ordinary unit policies.

use eredu_nn::{Error, NeuralBackend, Tensor};
use eredu_runtime::{
    ExecutionUnitLayout, LayerwiseAcquireError, LayerwisePolicy, LayerwisePolicyForward,
};

/// Every weight-dependent result and its inputs cross completion before a head
/// lease is released. The policy owns restoration on errors and unwinding.
pub(super) fn execute<B, U, P, F>(
    policy: &mut P,
    layout: &ExecutionUnitLayout,
    ordinal: usize,
    input: &B::Tensor,
    inputs: &[&B::Tensor],
    build: impl FnOnce(&<B::Tensor as Tensor>::Context) -> Result<U, Error>,
    operation: F,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<B::Tensor, Error>
where
    B: NeuralBackend,
    P: LayerwisePolicy<B, U>,
    P::Error: std::error::Error + Send + Sync + 'static,
    F: FnOnce(&mut U) -> Result<B::Tensor, Error>,
{
    let address = layout
        .address(ordinal)
        .ok_or_else(|| Error::backend("invalid assistant head unit ordinal"))?;
    let mut forward = LayerwisePolicyForward::<B, U, P>::begin(policy, input, context)
        .map_err(Error::backend_source)?;
    let unit = forward
        .acquire(ordinal, address, build)
        .map_err(|error| match error {
            LayerwiseAcquireError::Architecture(error) => error,
            LayerwiseAcquireError::Policy(error) => Error::backend_source(error),
        })?;
    let output = operation(unit)?;
    forward
        .complete(
            &output,
            std::iter::empty(),
            std::iter::once(input).chain(inputs.iter().copied()),
        )
        .map_err(Error::backend_source)?;
    forward.finish(&output).map_err(Error::backend_source)?;
    Ok(output)
}
