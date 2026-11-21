//! Portable ownership and type erasure of a routed prediction target.

use std::{any::Any, marker::PhantomData};

use eredu_core::{
    cache::mechanisms::CacheTensorStorage, execution_control::SnapshotEstimate, BackendFailure,
    BackendFailureKind,
};
use eredu_nn::NeuralBackend;

use crate::{executable_binding::text_invocation::TextPredictionLane, state::PortableTextState};

trait ErasedPredictionTarget<C: CacheTensorStorage>: Any {
    fn roots(&self) -> Vec<&C::Tensor>;
    fn checkpoint(&self) -> Result<Box<dyn ErasedPredictionTarget<C>>, BackendFailure>;
    fn estimate(&self) -> Option<SnapshotEstimate>;
    fn growth(&self, additional: u64) -> Option<u64>;
    fn copy(&self, queue: &C::Queue) -> Result<Box<dyn ErasedPredictionTarget<C>>, BackendFailure>;
    fn retain_prefix(&mut self, length: i32, queue: &C::Queue) -> Result<bool, BackendFailure>;
    fn restore(
        &mut self,
        checkpoint: &dyn ErasedPredictionTarget<C>,
        queue: &C::Queue,
    ) -> Result<(), BackendFailure>;
    fn offset(&self) -> i32;
}

struct TargetState<B, S> {
    state: S,
    backend: PhantomData<fn() -> B>,
}

impl<B, C, S> ErasedPredictionTarget<C> for TargetState<B, S>
where
    B: NeuralBackend<Tensor = C::Value> + 'static,
    C: CacheTensorStorage,
    S: PortableTextState<B, C> + 'static,
{
    fn roots(&self) -> Vec<&C::Tensor> {
        self.state.retained_arrays()
    }

    fn checkpoint(&self) -> Result<Box<dyn ErasedPredictionTarget<C>>, BackendFailure> {
        let state = self.state.deep_checkpoint()?;
        Ok(Box::new(Self {
            state,
            backend: PhantomData,
        }))
    }

    fn estimate(&self) -> Option<SnapshotEstimate> {
        self.state.isolated_snapshot_estimate()
    }

    fn growth(&self, additional: u64) -> Option<u64> {
        self.state.isolated_snapshot_growth(additional)
    }

    fn copy(&self, queue: &C::Queue) -> Result<Box<dyn ErasedPredictionTarget<C>>, BackendFailure> {
        let state = self.state.isolated_snapshot(queue)?;
        Ok(Box::new(Self {
            state,
            backend: PhantomData,
        }))
    }

    fn retain_prefix(&mut self, length: i32, queue: &C::Queue) -> Result<bool, BackendFailure> {
        self.state.retain_causal_prefix(length, queue)
    }

    fn restore(
        &mut self,
        checkpoint: &dyn ErasedPredictionTarget<C>,
        queue: &C::Queue,
    ) -> Result<(), BackendFailure> {
        let checkpoint = (checkpoint as &dyn Any)
            .downcast_ref::<Self>()
            .ok_or_else(|| invalid_state("prediction target checkpoint type changed"))?;
        self.state.restore_checkpoint(&checkpoint.state, queue)
    }

    fn offset(&self) -> i32 {
        self.state.offset()
    }
}

fn invalid_state(detail: &'static str) -> BackendFailure {
    BackendFailure::new(
        BackendFailureKind::InvalidSession,
        std::io::Error::other(detail),
    )
}

/// Ordinary target state retained independently of the bound route's concrete type.
pub struct PortablePredictionTargetState<B, C: CacheTensorStorage, E> {
    state: Option<Box<dyn ErasedPredictionTarget<C>>>,
    types: PhantomData<fn() -> (B, E)>,
}

impl<B, C: CacheTensorStorage, E> PortablePredictionTargetState<B, C, E> {
    fn new(state: Box<dyn ErasedPredictionTarget<C>>) -> Self {
        Self {
            state: Some(state),
            types: PhantomData,
        }
    }

    fn state(&self) -> Result<&dyn ErasedPredictionTarget<C>, BackendFailure> {
        self.state
            .as_deref()
            .ok_or_else(|| invalid_state("prediction target state is active"))
    }

    /// Checkpoints the target after settling its roots and the checkpoint's roots.
    pub fn checkpoint_view(&self) -> Result<Self, BackendFailure> {
        let current = self.state()?;
        C::settle_tensors(&current.roots())?;
        let checkpoint = current.checkpoint()?;
        C::settle_tensors(&checkpoint.roots())?;
        Ok(Self::new(checkpoint))
    }

    /// Captures a rollback checkpoint of the current target state.
    pub fn deep_clone(&self) -> Result<Self, BackendFailure> {
        self.state()?.checkpoint().map(Self::new)
    }

    /// Copies the complete target state for independently advancing control sessions.
    pub fn control_copy(&self, queue: &C::Queue) -> Result<Self, BackendFailure> {
        self.state()?.copy(queue).map(Self::new)
    }

    /// Returns the complete fixed and live-cache snapshot bound, when available.
    pub fn control_estimate(&self) -> Option<SnapshotEstimate> {
        self.state().ok()?.estimate()
    }

    /// Returns additional retained storage through the admitted input span.
    pub fn control_growth(&self, additional: u64) -> Option<u64> {
        self.state().ok()?.growth(additional)
    }

    /// Returns the current causal frontier and logical memory bounds.
    pub fn memory_bounds(
        &self,
        additional: u64,
    ) -> Result<(u64, Option<u64>, Option<u64>), BackendFailure> {
        let retained = self
            .control_estimate()
            .map(|estimate| estimate.retained_bytes);
        Ok((
            self.generation()?,
            retained.and_then(|bytes| bytes.checked_add(self.control_growth(0)?)),
            retained.and_then(|bytes| bytes.checked_add(self.control_growth(additional)?)),
        ))
    }

    /// Restores all target state from a type-compatible rollback checkpoint.
    pub fn restore(&mut self, checkpoint: &Self, queue: &C::Queue) -> Result<(), BackendFailure> {
        self.state
            .as_deref_mut()
            .ok_or_else(|| invalid_state("prediction target state is active"))?
            .restore(checkpoint.state()?, queue)
    }

    /// Retains the accepted prefix where the state's transaction supports truncation.
    pub fn retain_causal_prefix(
        &mut self,
        length: i32,
        queue: &C::Queue,
    ) -> Result<bool, BackendFailure> {
        self.state
            .as_deref_mut()
            .ok_or_else(|| invalid_state("prediction target state is active"))?
            .retain_prefix(length, queue)
    }

    /// Returns the exact committed target position.
    pub fn generation(&self) -> Result<u64, BackendFailure> {
        u64::try_from(self.state()?.offset()).map_err(BackendFailure::from_error)
    }
}

impl<B, C, E, S> TextPredictionLane<S> for PortablePredictionTargetState<B, C, E>
where
    B: NeuralBackend<Tensor = C::Value> + 'static,
    C: CacheTensorStorage,
    E: From<BackendFailure>,
    S: PortableTextState<B, C> + 'static,
{
    type Error = E;

    fn from_state(state: S) -> Self {
        Self::new(Box::new(TargetState::<B, S> {
            state,
            backend: PhantomData,
        }))
    }

    fn take_state(&mut self) -> Result<S, E> {
        if !(self.state()? as &dyn Any).is::<TargetState<B, S>>() {
            return Err(invalid_state("prediction target state type differs").into());
        }
        let state = self.state.take().expect("target is installed");
        let typed = (state as Box<dyn Any>)
            .downcast::<TargetState<B, S>>()
            .expect("target state type was checked");
        Ok(typed.state)
    }

    fn restore_state(&mut self, state: S) {
        *self = Self::from_state(state);
    }
}

/// State lifecycle available to typed prediction strategies.
pub trait PredictionTargetLifecycle<S, Q: ?Sized> {
    /// Captures the rollback state prior to a tentative invocation.
    fn checkpoint(state: &S) -> Result<S, BackendFailure>;
    /// Restores state after rejection.
    fn restore(state: &mut S, checkpoint: &S, context: &Q) -> Result<(), BackendFailure>;
    /// Returns the exact target position.
    fn position(state: &S) -> Result<u64, BackendFailure>;
    /// Returns logical copy bounds for the target.
    fn estimate(state: &S) -> Option<SnapshotEstimate>;
    /// Returns the current and forecast mutable state allowances.
    fn memory_bounds(state: &S, additional: u64) -> (Option<u64>, Option<u64>);
    /// Creates a settled, independently writable state if the profile supports it.
    fn snapshot(state: &S, context: &Q) -> Result<Option<S>, BackendFailure>;
}

impl<B, C, E, S> PredictionTargetLifecycle<S, C::Queue> for PortablePredictionTargetState<B, C, E>
where
    B: NeuralBackend<Tensor = C::Value>,
    C: CacheTensorStorage,
    S: PortableTextState<B, C>,
{
    fn checkpoint(state: &S) -> Result<S, BackendFailure> {
        state.deep_checkpoint()
    }

    fn restore(state: &mut S, checkpoint: &S, context: &C::Queue) -> Result<(), BackendFailure> {
        state.restore_checkpoint(checkpoint, context)
    }

    fn position(state: &S) -> Result<u64, BackendFailure> {
        u64::try_from(state.offset()).map_err(BackendFailure::from_error)
    }

    fn estimate(state: &S) -> Option<SnapshotEstimate> {
        state.isolated_snapshot_estimate()
    }

    fn memory_bounds(state: &S, additional: u64) -> (Option<u64>, Option<u64>) {
        let retained = state
            .isolated_snapshot_estimate()
            .map(|estimate| estimate.retained_bytes);
        (
            retained.and_then(|bytes| bytes.checked_add(state.isolated_snapshot_growth(0)?)),
            retained
                .and_then(|bytes| bytes.checked_add(state.isolated_snapshot_growth(additional)?)),
        )
    }

    fn snapshot(state: &S, context: &C::Queue) -> Result<Option<S>, BackendFailure> {
        if !state.supports_isolated_snapshot() {
            return Ok(None);
        }
        C::settle_tensors(&state.retained_arrays())?;
        let copy = state.isolated_snapshot(context)?;
        C::settle_tensors(&copy.retained_arrays())?;
        Ok(Some(copy))
    }
}
