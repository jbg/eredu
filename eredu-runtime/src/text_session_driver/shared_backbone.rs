//! Shared-backbone transactions over the installed executable and its native lease.

use super::*;
use crate::speculative::autoregressive::{CausalExecutor, CausalSeed, NoTargetRefinement};
use crate::speculative::shared_backbone::{
    SharedBackboneDraftOptions, SharedBackboneExecutor, SharedBackboneMechanisms,
    SharedBackboneTransaction,
};
use eredu_core::speculative::{
    settle_committed_suffix, SpeculativeCacheFactory, SpeculativeDriverError,
};
use eredu_core::BoundedCompletionWait;
use std::num::NonZeroUsize;

impl<R: 'static, N> TextSessionDriver<R, N> {
    /// Replays authoritative tokens into the installed causal cache and retains
    /// the score row needed to resume generation after a mode transition.
    pub fn append_committed_suffix<B, M>(
        &mut self,
        backend: &B,
        tokens: &[u32],
        wait: BoundedCompletionWait,
        context: M::Context<'_>,
    ) -> Result<M::Logits, B::Error>
    where
        B: BackendProvider,
        N: TextSessionMechanisms<B, Resources = R>,
        R: DerefMut<Target = TextSessionResourcesFor<B, N>>,
        M: SharedBackboneMechanisms<Model = N::Executable> + 'static,
        B::Error: From<M::Error> + From<SpeculativeDriverError<M::Error>>,
    {
        let invocation = self.begin::<B>(backend, false)?;
        let context = M::reborrow_context(context);
        let model = Rc::get_mut(invocation.root)
            .expect("exclusive resource root")
            .model
            .as_mut();
        let result = {
            let mut executor =
                CausalExecutor::<M, NoTargetRefinement>::new(model, NoTargetRefinement);
            executor
                .create_cache(&(), context)
                .map_err(B::Error::from)
                .and_then(|mut cache| {
                    let settled = settle_committed_suffix(
                        &mut executor,
                        &mut cache,
                        &CausalSeed::new(None, ()),
                        tokens,
                        wait,
                        None,
                        context,
                    )
                    .map_err(B::Error::from)?;
                    Ok((settled.logits, cache))
                })
        };
        let result = result.and_then(|(logits, mut cache)| {
            M::publish_committed(model, &mut cache).map_err(B::Error::from)?;
            Ok(logits)
        });
        invocation.settle(result)
    }

    /// Verifies one admitted masked proposal against the installed causal decoder.
    pub fn propose_shared<B, M>(
        &mut self,
        backend: &B,
        capacity: NonZeroUsize,
        last: u32,
        options: SharedBackboneDraftOptions,
        randomness: M::ProposalRandomness,
        wait: BoundedCompletionWait,
        context: M::Context<'_>,
    ) -> Result<SharedBackboneTransaction<M>, B::Error>
    where
        B: BackendProvider,
        N: TextSessionMechanisms<B, Resources = R>,
        R: DerefMut<Target = TextSessionResourcesFor<B, N>>,
        M: SharedBackboneMechanisms<Model = N::Executable> + 'static,
        B::Error: From<M::Error> + From<eredu_core::BackendFailure>,
    {
        let invocation = self.begin::<B>(backend, false)?;
        let context = M::reborrow_context(context);
        let model = Rc::get_mut(invocation.root)
            .expect("exclusive resource root")
            .model
            .as_mut();
        let mut executor = SharedBackboneExecutor::<M>::new(model, capacity);
        let result = executor
            .create_cache(&(), context)
            .map_err(B::Error::from)
            .and_then(|mut cache| {
                SharedBackboneTransaction::submit(
                    &mut executor,
                    &mut cache,
                    last,
                    options,
                    randomness,
                    wait,
                    context,
                )
                .map_err(|error| B::Error::from(eredu_core::BackendFailure::from_error(error)))
            });
        invocation.settle(result)
    }

    /// Reads the next causal score row from the retained verification.
    pub fn shared_target_scores<B, M>(
        &mut self,
        backend: &B,
        transaction: &SharedBackboneTransaction<M>,
        context: M::Context<'_>,
    ) -> Result<M::Logits, B::Error>
    where
        B: BackendProvider,
        N: TextSessionMechanisms<B, Resources = R>,
        R: DerefMut<Target = TextSessionResourcesFor<B, N>>,
        M: SharedBackboneMechanisms<Model = N::Executable> + 'static,
        B::Error: From<eredu_core::BackendFailure>,
    {
        let invocation = self.begin::<B>(backend, false)?;
        let context = M::reborrow_context(context);
        let model = Rc::get_mut(invocation.root)
            .expect("exclusive resource root")
            .model
            .as_mut();
        let executor = SharedBackboneExecutor::<M>::new(model, transaction.capacity());
        let result = transaction
            .logits(&executor, context)
            .map_err(|error| B::Error::from(eredu_core::BackendFailure::from_error(error)));
        invocation.settle(result)
    }

    /// Installs the accepted cache and settles its native publication.
    pub fn commit_shared_target<B, M>(
        &mut self,
        backend: &B,
        transaction: &mut SharedBackboneTransaction<M>,
        token: u32,
        context: M::Context<'_>,
    ) -> Result<bool, B::Error>
    where
        B: BackendProvider,
        N: TextSessionMechanisms<B, Resources = R>,
        R: DerefMut<Target = TextSessionResourcesFor<B, N>>,
        M: SharedBackboneMechanisms<Model = N::Executable> + 'static,
        M::Output: Clone,
        B::Error: From<M::Error> + From<eredu_core::BackendFailure>,
    {
        let invocation = self.begin::<B>(backend, false)?;
        let context = M::reborrow_context(context);
        let model = Rc::get_mut(invocation.root)
            .expect("exclusive resource root")
            .model
            .as_mut();
        let mut executor = SharedBackboneExecutor::<M>::new(model, transaction.capacity());
        let result = transaction.commit_target_token(&mut executor, token, context);
        drop(executor);
        let result = result
            .map_err(|error| B::Error::from(eredu_core::BackendFailure::from_error(error)))
            .and_then(|(matching, mut cache)| {
                M::publish_committed(model, &mut cache)
                    .map(|()| matching)
                    .map_err(B::Error::from)
            });
        invocation.settle(result)
    }
}
