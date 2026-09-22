//! Recovery owns the actual embedded equation roots and parallel source.
use super::*;
use crate::backend::{
    runtime::distributed::{
        Group,
        topology::original_source::{
            control::{
                CaptureSourceOwner, OriginalParallelControlOwner,
                OriginalParallelControlProjection, SpeculativeModelRole,
            },
            parallel::OriginalParallelInvocation,
        },
    },
    submission_recovery::prefill::{InvocationRootSource, TransientRootsProjection},
};
use eredu_nn::workspace::WorkspaceMetadataAllocation;
use eredu_runtime::working_memory::{
    InferenceExecutionIdentity, OriginalHostSourceBank, OriginalHostSourceCustody,
};
use std::rc::{Rc, Weak};

pub(crate) struct EmbeddedRootSource {
    pub(super) roots: OriginalSpeculativeRoots,
    pub(super) observer: OriginalScopeObserver,
    pub(super) completion: ResidentCompletionRecipe,
    pub(super) graph: RefCell<Option<safemlx::PreparedResidentGraph>>,
    pub(super) started: Cell<bool>,
    pub(super) completed: Cell<bool>,
    construction_closed: Cell<bool>,
    closed: Cell<bool>,
    parallel: Option<OriginalParallelInvocation>,
    control: Option<OriginalParallelControlOwner>,
    role: SpeculativeModelRole,
    stream: safemlx::StreamCopyPlan<()>,
    funding: HostMetadataFunding,
}
pub(super) struct EmbeddedRootOwner(Option<Rc<EmbeddedRootSource>>);
impl Clone for EmbeddedRootOwner {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}
impl Drop for EmbeddedRootOwner {
    fn drop(&mut self) {
        if let Some(value) = self.0.take() {
            drop(Rc::into_inner(value));
        }
    }
}
impl EmbeddedRootOwner {
    pub(super) fn inner(&self) -> &Rc<EmbeddedRootSource> {
        self.0.as_ref().expect("live embedded root owner")
    }
    pub(super) fn new(
        roots: OriginalSpeculativeRoots,
        observer: OriginalScopeObserver,
        completion: ResidentCompletionRecipe,
        parallel: Option<OriginalParallelInvocation>,
        control: Option<OriginalParallelControlOwner>,
        role: SpeculativeModelRole,
        stream: safemlx::StreamCopyPlan<()>,
        funding: HostMetadataFunding,
    ) -> Self {
        Self(Some(Rc::new(EmbeddedRootSource {
            roots,
            observer,
            completion,
            parallel,
            control,
            role,
            stream,
            funding,
            graph: RefCell::new(None),
            started: Cell::new(false),
            completed: Cell::new(false),
            construction_closed: Cell::new(false),
            closed: Cell::new(false),
        })))
    }
}
impl EmbeddedRootSource {
    pub(super) fn authenticate(&self, require_started: bool) -> Result<(), Error> {
        if let Some(cause) = self.observer.retained_failure() {
            return Err(cause.into());
        }
        if self.closed.get()
            || self.completed.get()
            || self.observer.status().blocked()
            || (require_started && !self.started.get())
            || !self
                .observer
                .same_scope(&OriginalScopeObserver::require_current()?)
        {
            return Err(Error::PrefillScopeUnavailable);
        }
        Ok(())
    }
    fn authenticate_stream(&self, stream: &Stream) -> Result<(), Error> {
        self.funding
            .reserve_metadata(
                self.stream
                    .source_comparison_control_bytes()
                    .ok_or(Error::PrefillScopeUnavailable)?,
            )
            .map_err(Error::WorkspacePlanning)?;
        self.authenticate(true)?;
        if !self.stream.matches_source(stream) {
            return Err(Error::PrefillScopeUnavailable);
        }
        Ok(())
    }
    pub(super) fn close(&self) {
        self.closed.set(true);
        drop(self.graph.borrow_mut().take());
    }
    pub(super) fn finish_graph(&self) -> Result<(), Error> {
        let graph = self
            .graph
            .try_borrow_mut()
            .map_err(|_| Error::PrefillScopeReentrant)?
            .take();
        if graph.is_none() && !self.construction_closed.get() {
            return Err(Error::PrefillScopeUnavailable);
        }
        drop(graph);
        Ok(())
    }
    pub(super) fn bind_initialized(
        &self,
        bank: Option<OriginalHostSourceBank>,
    ) -> Result<(), Error> {
        if let Some(bank) = bank {
            self.authenticate(false)?;
            if self.started.get() {
                return Err(Error::PrefillScopeUnavailable);
            }
            self.control
                .as_ref()
                .ok_or(Error::PrefillScopeUnavailable)?
                .retained()
                .with_token_sources(Some(bank))?;
        }
        Ok(())
    }
}
impl InvocationRootSource for EmbeddedRootSource {
    fn metadata_funding(&self) -> &HostMetadataFunding {
        &self.funding
    }
    fn source_custody(&self) -> Option<OriginalHostSourceCustody> {
        Some(self.role.budget_custody().into())
    }
    fn scope_observer(&self) -> Result<OriginalScopeObserver, Error> {
        self.authenticate(false)?;
        Ok(self.observer.clone())
    }
    fn observer(&self) -> Result<OriginalScopeObserver, Error> {
        self.authenticate(true)?;
        Ok(self.observer.clone())
    }
    fn completion_recipe(&self) -> Result<ResidentCompletionRecipe, Error> {
        self.authenticate(true)?;
        Ok(self.completion)
    }
    fn append(&self, value: &Array) -> Result<(), Error> {
        self.authenticate(true)?;
        self.roots
            .0
            .try_borrow_mut()
            .map_err(|_| Error::PrefillScopeReentrant)?
            .append(value)
            .map_err(|cause| Error::Neural(self.funding.metadata_source(cause)))
    }
    fn retire_completed(&self, value: &Array) -> Result<(), Error> {
        self.authenticate(true)?;
        self.roots
            .0
            .try_borrow_mut()
            .map_err(|_| Error::PrefillScopeReentrant)?
            .retire_completed_current(value)
            .map_err(Error::PrefillRoots)
    }
    fn close_construction(&self) -> Result<(), Error> {
        self.authenticate(true)?;
        if self.construction_closed.get() {
            return Err(Error::PrefillScopeUnavailable);
        }
        let graph = self
            .graph
            .try_borrow_mut()
            .map_err(|_| Error::PrefillScopeReentrant)?
            .take()
            .ok_or(Error::PrefillScopeUnavailable)?;
        self.construction_closed.set(true);
        drop(graph);
        Ok(())
    }
}
/// A weak lexical loan; the accepted equation Recovery retains every source.
#[derive(Clone)]
pub(crate) struct EmbeddedParallelInvocation {
    source: Weak<EmbeddedRootSource>,
    funding: HostMetadataFunding,
    custody: OriginalSpeculativeBudgetCustody,
}
impl std::fmt::Debug for EmbeddedParallelInvocation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EmbeddedParallelInvocation")
            .finish_non_exhaustive()
    }
}
impl EmbeddedParallelInvocation {
    pub(super) fn new(source: &Rc<EmbeddedRootSource>) -> Self {
        Self {
            source: Rc::downgrade(source),
            funding: source.funding.clone(),
            custody: source.role.budget_custody(),
        }
    }
    fn owner(&self) -> Result<EmbeddedRootOwner, Error> {
        self.source
            .upgrade()
            .map(|source| EmbeddedRootOwner(Some(source)))
            .ok_or(Error::PrefillScopeUnavailable)
    }
    pub(crate) fn metadata_funding(&self) -> &HostMetadataFunding {
        &self.funding
    }
    pub(crate) fn validate_execution(
        &self,
        identity: &InferenceExecutionIdentity,
    ) -> Result<(), Error> {
        let owner = self.owner()?;
        owner.inner().authenticate(false)?;
        owner
            .inner()
            .role
            .validate_execution(identity)
            .map_err(Error::PrefillControl)
    }
    pub(crate) fn with_parallel<T, E, F>(
        &self,
        stream: &Stream,
        foreign: Option<&OriginalParallelControlProjection>,
        run: F,
    ) -> Result<Result<T, E>, Error>
    where
        F: FnOnce(Option<(&mut Group, &HostMetadataFunding)>) -> Result<T, E>,
    {
        let owner = self.owner()?;
        let source = owner.inner();
        source.authenticate_stream(stream)?;
        if foreign.is_some() {
            return Err(Error::PrefillScopeUnavailable);
        }
        let Some(parallel) = &source.parallel else {
            if source.control.is_some() {
                return Err(Error::PrefillScopeUnavailable);
            }
            return Ok(run(None));
        };
        let control = source
            .control
            .as_ref()
            .ok_or(Error::PrefillScopeUnavailable)?;
        let (installation, projection) = control.install()?;
        let result = (|| {
            parallel.prepare_lending::<F, Result<T, E>>()?;
            let neural = parallel.has_neural_context();
            parallel.with_model_context(
                &source.observer,
                stream,
                Some(&projection),
                TransientRootsProjection::from_invocation(source),
                |group, funding| run(neural.then_some((group, funding))),
            )
        })();
        installation.close();
        let success = matches!(&result, Ok(Ok(_)));
        let finished = control.finish_model(success);
        if success {
            finished?;
        }
        result
    }
    pub(crate) fn with_parallel_publication<T, E, F>(
        &self,
        stream: &Stream,
        run: F,
    ) -> Result<Result<T, E>, Error>
    where
        F: FnOnce(Option<(&Group, &HostMetadataFunding)>) -> Result<T, E>,
    {
        let owner = self.owner()?;
        let source = owner.inner();
        source.authenticate_stream(stream)?;
        match &source.parallel {
            Some(parallel) => {
                parallel.prepare_lending::<F, Result<T, E>>()?;
                parallel.with_publication_context(
                    &source.observer,
                    stream,
                    TransientRootsProjection::from_invocation(source),
                    run,
                )
            }
            None => Ok(run(None)),
        }
    }
}
pub(super) fn control_bytes() -> Option<usize> {
    let parts = [
        size_of::<EmbeddedRootOwner>(),
        size_of::<Option<EmbeddedRootOwner>>(),
        size_of::<EmbeddedParallelInvocation>(),
        size_of::<EmbeddedRootSource>(),
        rc_bytes::<EmbeddedRootSource>()?,
        size_of::<SpeculativeModelRole>(),
        size_of::<Option<OriginalParallelControlOwner>>(),
        size_of::<Result<OriginalParallelControlOwner, Error>>(),
        size_of::<Option<OriginalParallelInvocation>>(),
        size_of::<Result<OriginalParallelInvocation, Error>>(),
        size_of::<Result<(), Error>>(),
        size_of::<OriginalHostSourceCustody>(),
        TransientRootsProjection::control_bytes()?,
        OriginalScopeObserver::control_bytes()?,
        safemlx::StreamCopyPlan::<()>::capture_control_bytes().ok()?,
        size_of::<safemlx::StreamCopyPlan<()>>(),
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}
