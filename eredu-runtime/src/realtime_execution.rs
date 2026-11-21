//! Portable ownership of installed realtime components and invocations.

use eredu_core::realtime_mechanisms::{
    RealtimeCompletionCreationError, RealtimeQueue, RealtimeSubmissionOwner,
};
use std::{
    cell::{Cell, RefCell, RefMut},
    rc::Rc,
};

use eredu_checkpoint::store::SharedCheckpointSource;
use eredu_core::{BackendFailure, Completion};
use eredu_nn::{RetainedTensorRoots, Tensor};

use crate::{
    GenerationSampler, LayerwiseModelMetadata, PreparedRealtimeFrameExecutor,
    SequentialDecisionDriver, UnitResidencyTelemetry,
};
use eredu_core::sampling::SamplingBackend;

struct Execution<T, Q> {
    components: RefCell<T>,
    queue: Q,
    poisoned: Cell<bool>,
    active: Cell<bool>,
}

/// Installed components and one exclusive invocation gate shared by siblings.
pub struct SharedRealtimeExecution<T: 'static, Q: RealtimeQueue> {
    inner: Option<Rc<Q::Retired<Execution<T, Q>>>>,
}

impl<T: 'static, Q: RealtimeQueue> SharedRealtimeExecution<T, Q> {
    /// Installs bound components on their exact queue.
    pub fn new(components: T, queue: Q) -> Self {
        Self {
            inner: Some(Rc::new(Q::retire(Execution {
                components: RefCell::new(components),
                queue,
                poisoned: Cell::new(false),
                active: Cell::new(false),
            }))),
        }
    }

    fn inner(&self) -> &Rc<Q::Retired<Execution<T, Q>>> {
        self.inner.as_ref().expect("live realtime execution")
    }

    /// Retains the installed components and invocation gate for a sibling.
    pub fn share(&self) -> Self {
        Self {
            inner: Some(Rc::clone(self.inner())),
        }
    }

    /// Exact queue associated with the installed components.
    pub fn queue(&self) -> &Q {
        &self.inner().queue
    }

    fn components(&self) -> &RefCell<T> {
        &self.inner().components
    }

    /// Whether unresolved work temporarily owns the invocation gate.
    pub fn is_busy(&self) -> bool {
        Q::reclaim();
        !self.inner().poisoned.get() && self.inner().active.get()
    }

    /// Rejects use after terminal native failure.
    pub fn require_healthy(&self) -> Result<(), BackendFailure> {
        if self.inner().poisoned.get() {
            Err(failure("realtime execution is poisoned"))
        } else {
            Ok(())
        }
    }

    /// Validates availability without acquiring authority or starting work.
    pub fn require_available(&self) -> Result<(), BackendFailure> {
        Q::reclaim();
        self.require_healthy()?;
        if self.inner().active.get() {
            Err(failure(
                "realtime execution retains an outstanding invocation",
            ))
        } else {
            Ok(())
        }
    }

    /// Enters a native scope only after the portable caller has admitted work.
    pub fn begin<R: 'static>(
        &self,
        retention: R,
        initial: &[Q::Tensor],
    ) -> Result<Q::Scope, BackendFailure> {
        self.require_available()?;
        self.inner().active.set(true);
        self.queue().begin(
            Lease::<T, Q, R> {
                execution: Rc::clone(self.inner()),
                _retention: retention,
            },
            initial,
        )
    }

    pub(crate) fn open_components(&self) -> Result<RefMut<'_, T>, BackendFailure> {
        self.require_available()?;
        Ok(self.components().borrow_mut())
    }
}

impl<T: 'static, Q: RealtimeQueue> Drop for SharedRealtimeExecution<T, Q> {
    fn drop(&mut self) {
        drop(self.inner.take());
        Q::release_retired();
    }
}

struct Lease<T: 'static, Q: RealtimeQueue, R> {
    execution: Rc<Q::Retired<Execution<T, Q>>>,
    _retention: R,
}

impl<T: 'static, Q: RealtimeQueue, R> RealtimeSubmissionOwner for Lease<T, Q, R> {
    fn poison(&self) {
        self.execution.poisoned.set(true);
    }
}

impl<T: 'static, Q: RealtimeQueue, R> Drop for Lease<T, Q, R> {
    fn drop(&mut self) {
        self.execution.active.set(false);
    }
}

fn failure(message: &'static str) -> BackendFailure {
    BackendFailure::from_error(std::io::Error::other(message))
}

impl<T: 'static, Q: RealtimeQueue> SharedRealtimeExecution<T, Q> {
    /// Runs admitted startup work and settles its roots before returning.
    pub fn prepare_startup<O: RetainedTensorRoots<Q::Tensor>>(
        &self,
        operation: impl FnOnce(&mut T, &<Q::Tensor as Tensor>::Context) -> Result<O, BackendFailure>,
    ) -> Result<O, BackendFailure> {
        let mut components = self.open_components()?;
        let scope = self.begin((), &[])?;
        let output = operation(&mut components, self.queue().context())?;
        let mut roots = Vec::new();
        output.visit_retained_tensors(&mut |tensor| roots.push(tensor.clone()));
        let completion = Q::finish(scope, roots).map_err(|error| match error {
            RealtimeCompletionCreationError::BeforeSubmission(error)
            | RealtimeCompletionCreationError::AfterSubmission { error, .. } => error,
        })?;
        completion.wait().map_err(BackendFailure::from_error)?;
        drop(completion);
        drop(components);
        Q::finish_startup()?;
        Ok(output)
    }
}

type Roots<T> = Box<dyn RetainedTensorRoots<T>>;
trait RealtimeExecutable<B: SamplingBackend, S> {
    fn residency_report(&self) -> Result<crate::ResidencyReport, BackendFailure>;
    fn dense_stream_report(&self) -> Result<Option<crate::DenseDiskStreamReport>, BackendFailure>;
    fn execution_group_reports(
        &self,
    ) -> Result<Vec<crate::ResidentLayerGroupReport>, BackendFailure>;
    fn execute(
        &mut self,
        state: &mut S,
        temporal: &[B::Token],
        driver: &mut SequentialDecisionDriver<B, GenerationSampler>,
        context: &B::Context,
    ) -> Result<Roots<B::Logits>, BackendFailure>;
}

impl<B, S, E, F> RealtimeExecutable<B, S> for E
where
    B: SamplingBackend,
    E: PreparedRealtimeFrameExecutor<B, GenerationSampler, S> + UnitResidencyTelemetry<Error = F>,
    <E as PreparedRealtimeFrameExecutor<B, GenerationSampler, S>>::Error:
        std::error::Error + Send + Sync + 'static,
    E::Retained: RetainedTensorRoots<B::Logits> + 'static,
    F: std::error::Error + Send + Sync + 'static,
{
    fn residency_report(&self) -> Result<crate::ResidencyReport, BackendFailure> {
        UnitResidencyTelemetry::residency_report(self).map_err(BackendFailure::from_error)
    }

    fn dense_stream_report(&self) -> Result<Option<crate::DenseDiskStreamReport>, BackendFailure> {
        UnitResidencyTelemetry::dense_stream_report(self).map_err(BackendFailure::from_error)
    }

    fn execution_group_reports(
        &self,
    ) -> Result<Vec<crate::ResidentLayerGroupReport>, BackendFailure> {
        UnitResidencyTelemetry::execution_group_reports(self).map_err(BackendFailure::from_error)
    }

    fn execute(
        &mut self,
        state: &mut S,
        temporal: &[B::Token],
        driver: &mut SequentialDecisionDriver<B, GenerationSampler>,
        context: &B::Context,
    ) -> Result<Roots<B::Logits>, BackendFailure> {
        PreparedRealtimeFrameExecutor::execute(self, state, temporal, driver, context)
            .map(|roots| Box::new(roots) as Roots<_>)
            .map_err(BackendFailure::from_error)
    }
}

struct InstalledPayload<B: SamplingBackend, S> {
    executor: Box<dyn RealtimeExecutable<B, S>>,
    _source: SharedCheckpointSource,
}

/// Erased portable categorical execution and the native queue it retains.
pub struct InstalledRealtimeExecution<B: SamplingBackend + 'static, S: 'static, Q: RealtimeQueue> {
    metadata: LayerwiseModelMetadata,
    owner: SharedRealtimeExecution<InstalledPayload<B, S>, Q>,
}

impl<B, S, Q> InstalledRealtimeExecution<B, S, Q>
where
    B: SamplingBackend<
            Token = Q::Tensor,
            Logits = Q::Tensor,
            Context = <Q::Tensor as Tensor>::Context,
        > + 'static,
    S: 'static,
    Q: RealtimeQueue,
{
    /// Erases the prepared architecture executor after portable binding.
    pub fn bind<E, F>(
        executor: E,
        metadata: LayerwiseModelMetadata,
        source: SharedCheckpointSource,
        queue: Q,
    ) -> Self
    where
        E: PreparedRealtimeFrameExecutor<B, GenerationSampler, S>
            + UnitResidencyTelemetry<Error = F>
            + 'static,
        <E as PreparedRealtimeFrameExecutor<B, GenerationSampler, S>>::Error:
            std::error::Error + Send + Sync + 'static,
        E::Retained: RetainedTensorRoots<B::Logits> + 'static,
        F: std::error::Error + Send + Sync + 'static,
    {
        Self {
            metadata,
            owner: SharedRealtimeExecution::new(
                InstalledPayload {
                    executor: Box::new(executor),
                    _source: source,
                },
                queue,
            ),
        }
    }

    /// Parameter topology and residency metadata.
    pub fn metadata(&self) -> &LayerwiseModelMetadata {
        &self.metadata
    }
    /// Exact native queue and retained resources.
    pub fn queue(&self) -> &Q {
        self.owner.queue()
    }
    /// Rejects a previously poisoned execution.
    pub fn require_healthy(&self) -> Result<(), BackendFailure> {
        self.owner.require_healthy()
    }
    /// Opens the native submission selected by the portable driver.
    pub(crate) fn begin(&self) -> Result<Q::Scope, BackendFailure> {
        self.owner.begin((), &[])
    }

    /// Logical residency and transfer telemetry.
    pub fn residency_report(&self) -> Result<crate::ResidencyReport, BackendFailure> {
        self.require_healthy()?;
        self.owner.components().borrow().executor.residency_report()
    }

    /// Disk-stream counters for an installed bounded execution.
    pub fn dense_stream_report(
        &self,
    ) -> Result<Option<crate::DenseDiskStreamReport>, BackendFailure> {
        self.require_healthy()?;
        self.owner
            .components()
            .borrow()
            .executor
            .dense_stream_report()
    }

    /// Per-group residency for the installed execution.
    pub fn execution_group_reports(
        &self,
    ) -> Result<Vec<crate::ResidentLayerGroupReport>, BackendFailure> {
        self.require_healthy()?;
        self.owner
            .components()
            .borrow()
            .executor
            .execution_group_reports()
    }
}

impl<B, S, Q, Branch> PreparedRealtimeFrameExecutor<B, GenerationSampler, Branch>
    for &InstalledRealtimeExecution<B, S, Q>
where
    B: SamplingBackend + 'static,
    Q: RealtimeQueue,
    S: 'static,
    Branch: std::ops::DerefMut<Target = S>,
{
    type Error = BackendFailure;
    type Retained = Roots<B::Logits>;
    fn execute(
        &mut self,
        state: &mut Branch,
        temporal: &[B::Token],
        driver: &mut SequentialDecisionDriver<B, GenerationSampler>,
        context: &B::Context,
    ) -> Result<Self::Retained, Self::Error> {
        self.owner
            .components()
            .borrow_mut()
            .executor
            .execute(state, temporal, driver, context)
    }
}
