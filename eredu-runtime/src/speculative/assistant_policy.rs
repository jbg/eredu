//! Parameter ownership for a complete assistant module or its independent units.
use crate::{ExecutionUnitAddress, LayerwiseAcquireError, LayerwisePolicy};
use eredu_core::BackendFailure;
use eredu_nn::{NeuralBackend, Tensor};

/// An assistant retains a unit policy only when it has separately acquired units.
pub struct AssistantParameterPolicy<P> {
    pub(super) units: Option<P>,
}

impl<P> AssistantParameterPolicy<P> {
    /// Borrows the independently acquired unit policy when present.
    pub fn units(&self) -> Option<&P> {
        self.units.as_ref()
    }
}

fn missing_units<E: From<BackendFailure>>() -> E {
    BackendFailure::from_error(std::io::Error::other(
        "resident assistant module has no separate execution units",
    ))
    .into()
}

impl<B, U, P> LayerwisePolicy<B, U> for AssistantParameterPolicy<P>
where
    B: NeuralBackend,
    P: LayerwisePolicy<B, U, Error: From<BackendFailure>>,
{
    type Lease = P::Lease;
    type Error = P::Error;
    fn permits_deferred_serial_groups(&self) -> bool {
        self.units
            .as_ref()
            .is_some_and(LayerwisePolicy::permits_deferred_serial_groups)
    }
    fn begin(
        &mut self,
        initial: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(), P::Error> {
        self.units
            .as_mut()
            .ok_or_else(missing_units::<P::Error>)?
            .begin(initial, context)
    }
    fn acquire<E, F>(
        &mut self,
        ordinal: usize,
        address: ExecutionUnitAddress,
        build: F,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Lease, LayerwiseAcquireError<E, P::Error>>
    where
        F: FnOnce(&<B::Tensor as Tensor>::Context) -> Result<U, E>,
    {
        self.units
            .as_mut()
            .ok_or_else(missing_units::<P::Error>)
            .map_err(LayerwiseAcquireError::Policy)?
            .acquire(ordinal, address, build, context)
    }
    fn complete<'a, S, C>(
        &mut self,
        ordinal: usize,
        address: ExecutionUnitAddress,
        lease: Self::Lease,
        output: &'a B::Tensor,
        state: S,
        inputs: C,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(), P::Error>
    where
        S: Iterator<Item = &'a B::Tensor>,
        C: Iterator<Item = &'a B::Tensor>,
        B::Tensor: 'a,
    {
        self.units
            .as_mut()
            .ok_or_else(missing_units::<P::Error>)?
            .complete(ordinal, address, lease, output, state, inputs, context)
    }
    fn abort(
        &mut self,
        active: Option<(usize, ExecutionUnitAddress, Self::Lease)>,
        context: &<B::Tensor as Tensor>::Context,
    ) {
        if let Some(policy) = &mut self.units {
            policy.abort(active, context);
        } else {
            debug_assert!(active.is_none());
        }
    }
    fn finish(
        &mut self,
        output: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(), P::Error> {
        self.units
            .as_mut()
            .ok_or_else(missing_units::<P::Error>)?
            .finish(output, context)
    }
}
