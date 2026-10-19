//! Native parameter-policy ownership for architecture-selected assistant runtimes.
use crate::{
    backend::{
        error::Error,
        nn::shared::MlxNeuralBackend,
        runtime::execution::generic::{MlxSelectedLayerwisePolicy, MlxSelectedUnitLease},
    },
    MlxTensor,
};
use eredu_nn::Parameterized;
use eredu_runtime::{ExecutionUnitAddress, LayerwiseAcquireError, LayerwisePolicy};
use safemlx::Stream;

pub(crate) struct MlxAssistantParameterPolicy<U: 'static>(
    pub(crate) Option<MlxSelectedLayerwisePolicy<U, ()>>,
);
impl<U: 'static> MlxAssistantParameterPolicy<U> {
    fn policy(&mut self) -> Result<&mut MlxSelectedLayerwisePolicy<U, ()>, Error> {
        self.0.as_mut().ok_or_else(|| {
            Error::Parallel("resident module construction has no separate unit policy".into())
        })
    }
}
impl<U: Parameterized<MlxTensor> + 'static> LayerwisePolicy<MlxNeuralBackend, U>
    for MlxAssistantParameterPolicy<U>
{
    type Lease = MlxSelectedUnitLease<U>;
    type Error = Error;
    fn permits_deferred_serial_groups(&self) -> bool {
        self.0
            .as_ref()
            .is_some_and(LayerwisePolicy::permits_deferred_serial_groups)
    }
    fn begin(&mut self, initial: &MlxTensor, context: &Stream) -> Result<(), Error> {
        self.policy()?.begin(initial, context)
    }
    fn acquire<E, F>(
        &mut self,
        ordinal: usize,
        address: ExecutionUnitAddress,
        build: F,
        context: &Stream,
    ) -> Result<Self::Lease, LayerwiseAcquireError<E, Error>>
    where
        F: FnOnce(&Stream) -> Result<U, E>,
    {
        self.policy()
            .map_err(LayerwiseAcquireError::Policy)?
            .acquire(ordinal, address, build, context)
    }
    fn complete<'a, S, C>(
        &mut self,
        ordinal: usize,
        address: ExecutionUnitAddress,
        lease: Self::Lease,
        output: &'a MlxTensor,
        state: S,
        inputs: C,
        context: &Stream,
    ) -> Result<(), Error>
    where
        S: Iterator<Item = &'a MlxTensor>,
        C: Iterator<Item = &'a MlxTensor>,
        MlxTensor: 'a,
    {
        self.policy()?
            .complete(ordinal, address, lease, output, state, inputs, context)
    }
    fn abort(
        &mut self,
        active: Option<(usize, ExecutionUnitAddress, Self::Lease)>,
        context: &Stream,
    ) {
        if let Some(policy) = &mut self.0 {
            policy.abort(active, context);
        } else {
            debug_assert!(active.is_none());
        }
    }
    fn finish(&mut self, output: &MlxTensor, context: &Stream) -> Result<(), Error> {
        self.policy()?.finish(output, context)
    }
}
