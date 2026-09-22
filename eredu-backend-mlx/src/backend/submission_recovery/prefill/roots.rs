//! Closed completion payload, borrowed through a separately retained projection.
use crate::backend::error::Error;
use eredu_runtime::working_memory::{
    OriginalHostSourceCustody, OriginalPrefillRootCustody, OriginalPrefillRootProjectionCustody,
};
use safemlx::{
    Array, PrefillRoots, PrefillRootsRuntime, PreparedPrefillFailure, SubmissionGraphQuota,
    SubmissionScope,
};
use std::{
    alloc::Layout,
    cell::{Cell, OnceCell, RefCell},
    mem::size_of,
    rc::{Rc, Weak},
};

struct Payload {
    host_construction: RefCell<Option<safemlx::PreparedResidentGraph>>,
    value: RefCell<Option<PrefillRoots>>,
    initialized: Cell<bool>,
    // Original custody now lives in the one preallocated native failure owner.
    original: bool,
    traversal: Cell<Option<crate::backend::nn::workspace::ResidentCompletionRecipe>>,
    // Exact immutable per-span source remains inside the existing Recovery Q.
    parallel: OnceCell<crate::backend::runtime::distributed::topology::original_source::parallel::OriginalParallelInvocation>,
    addressable: OnceCell<crate::backend::submission_recovery::addressable::AddressableExecutionRow>,
}
/// Only Recovery owns a long-lived strong handle. Every temporary strong exit
/// is closed and consuming, including upgrades on fixed refusal or unwinding.
pub(super) struct RootsOwner(Option<Rc<Payload>>);
impl Drop for RootsOwner {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            drop(Rc::into_inner(owner));
        }
    }
}
/// The weak control survives payload retirement only with its independently
/// priced custody. Fields drop weak-header first, original token last.
enum ProjectionCustody {
    Prefill(OriginalPrefillRootProjectionCustody),
    Model(eredu_runtime::working_memory::OriginalTextControlGuard),
}
pub(crate) struct RootsProjection {
    value: Weak<Payload>,
    _custody: Option<ProjectionCustody>,
}
impl std::fmt::Debug for RootsProjection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RootsProjection").finish_non_exhaustive()
    }
}
/// Private bank-owned weak handoff. It never keeps a completed/failed root
/// payload alive; the containing original prefill bank retains its paid header.
#[derive(Clone)]
pub(super) struct CaptureProjection(Weak<Payload>);
impl CaptureProjection {
    fn retained_observer(&self) -> Result<safemlx::OriginalScopeObserver, Error> {
        let owner = RootsOwner(Some(
            self.0.upgrade().ok_or(Error::PrefillScopeUnavailable)?,
        ));
        let payload = owner.0.as_ref().expect("closed upgraded root payload");
        if !payload.original {
            return Err(Error::PrefillScopeUnavailable);
        }
        // End the root-bank loan before native observer authentication. The
        // already-created bank supplies identity; TLS supplies only equality.
        let observer = payload
            .host_construction
            .try_borrow()
            .map_err(|_| Error::PrefillScopeReentrant)?
            .as_ref()
            .map(|bank| bank.observer().clone())
            .ok_or(Error::PrefillScopeUnavailable)?;
        // A retained failed payload remains owned for recovery, but cannot
        // authorize another capture. This query performs no progress or wait.
        if let Some(cause) = observer.retained_failure() {
            return Err(cause.into());
        }
        if observer.status().blocked() {
            return Err(Error::PrefillScopeUnavailable);
        }
        Ok(observer)
    }
    pub(super) fn observer(&self) -> Result<safemlx::OriginalScopeObserver, Error> {
        let observer = self.retained_observer()?;
        let current = safemlx::OriginalScopeObserver::require_current()?;
        if !observer.same_scope(&current) {
            return Err(Error::PrefillControl(
                eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
            ));
        }
        Ok(observer)
    }
    pub(super) fn capture_observer(&self) -> Result<safemlx::OriginalScopeObserver, Error> {
        let parent = self.retained_observer()?;
        crate::backend::nn::tensor::TokenValidationScope::capture_observer_for(&parent)
            .map_err(Into::into)
    }
}
/// Native completion sources lend their actual root collector without creating
/// a different execution role. Composition retains the concrete source and its
/// recovery owner; this interface only authenticates and borrows that owner.
pub(crate) trait InvocationRootSource {
    fn metadata_funding(&self) -> &eredu_nn::workspace::HostMetadataFunding;
    /// Accounting identity of this actual invocation, when it can lend a paged
    /// source. An unqualified completion projection supplies no such identity.
    fn source_custody(&self) -> Option<OriginalHostSourceCustody> {
        None
    }
    /// Source installation can precede graph construction. Implementations
    /// still authenticate the actual current scope and reject failed owners.
    fn scope_observer(&self) -> Result<safemlx::OriginalScopeObserver, Error> {
        self.observer()
    }
    fn observer(&self) -> Result<safemlx::OriginalScopeObserver, Error>;
    fn completion_recipe(
        &self,
    ) -> Result<crate::backend::nn::workspace::ResidentCompletionRecipe, Error>;
    fn append(&self, value: &Array) -> Result<(), Error>;
    fn retire_completed(&self, value: &Array) -> Result<(), Error>;
    fn close_construction(&self) -> Result<(), Error>;
}
#[derive(Clone)]
enum RootSource {
    Prefill(CaptureProjection),
    Invocation {
        source: Weak<dyn InvocationRootSource>,
        _funding: eredu_nn::workspace::HostMetadataFunding,
        _custody: Option<OriginalHostSourceCustody>,
    },
}
enum RootLoan {
    Prefill(RootsOwner),
    Invocation(Rc<dyn InvocationRootSource>),
}
impl RootLoan {
    fn observer(&self) -> Result<safemlx::OriginalScopeObserver, Error> {
        match self {
            Self::Prefill(owner) => owner.capture_projection().observer(),
            Self::Invocation(source) => source.observer(),
        }
    }
    fn recipe(&self) -> Result<crate::backend::nn::workspace::ResidentCompletionRecipe, Error> {
        match self {
            Self::Prefill(owner) => owner
                .0
                .as_ref()
                .expect("closed root owner")
                .traversal
                .get()
                .ok_or(Error::PrefillScopeUnavailable),
            Self::Invocation(source) => source.completion_recipe(),
        }
    }
    fn append(&self, value: &Array) -> Result<(), Error> {
        match self {
            Self::Prefill(owner) => owner
                .0
                .as_ref()
                .expect("closed root owner")
                .value
                .try_borrow_mut()
                .map_err(|_| Error::PrefillScopeReentrant)?
                .as_mut()
                .ok_or(Error::PrefillScopeUnavailable)?
                .append(value)
                .map_err(|cause| Error::PrefillRoots(cause.into())),
            Self::Invocation(source) => source.append(value),
        }
    }
    fn retire_completed(&self, value: &Array) -> Result<(), Error> {
        match self {
            Self::Prefill(owner) => owner
                .0
                .as_ref()
                .expect("closed root owner")
                .value
                .try_borrow_mut()
                .map_err(|_| Error::PrefillScopeReentrant)?
                .as_mut()
                .ok_or(Error::PrefillScopeUnavailable)?
                .retire_completed_current(value)
                .map_err(Error::PrefillRoots),
            Self::Invocation(source) => source.retire_completed(value),
        }
    }
    fn close_construction(&self) -> Result<(), Error> {
        match self {
            Self::Prefill(owner) => {
                let construction = owner
                    .0
                    .as_ref()
                    .expect("closed root owner")
                    .host_construction
                    .try_borrow_mut()
                    .map_err(|_| Error::PrefillScopeReentrant)?
                    .take();
                drop(construction);
                Ok(())
            }
            Self::Invocation(source) => source.close_construction(),
        }
    }
}
/// Weak append-only view of the actual model completion owner. The projection
/// cannot create a root bank, reserve capacity, or grant execution authority.
#[derive(Clone)]
pub(crate) struct TransientRootsProjection(RootSource);
impl TransientRootsProjection {
    pub(crate) fn from_invocation<T: InvocationRootSource + 'static>(source: &Rc<T>) -> Self {
        let erased: Rc<dyn InvocationRootSource> = source.clone();
        Self(RootSource::Invocation {
            source: Rc::downgrade(&erased),
            _funding: source.metadata_funding().clone(),
            _custody: source.source_custody(),
        })
    }
    fn owner(&self) -> Result<RootLoan, Error> {
        match &self.0 {
            RootSource::Prefill(source) => Ok(RootLoan::Prefill(RootsOwner(Some(
                source.0.upgrade().ok_or(Error::PrefillScopeUnavailable)?,
            )))),
            RootSource::Invocation { source, .. } => source
                .upgrade()
                .map(RootLoan::Invocation)
                .ok_or(Error::PrefillScopeUnavailable),
        }
    }
    pub(crate) fn validate_boundary_leaf(&self, value: &Array) -> Result<(), Error> {
        let observer = self.owner()?.observer()?;
        safemlx::OperationEvent::validate_traversal_leaf(value, &observer)?;
        Ok(())
    }
    pub(crate) fn boundary_traversal(
        &self,
        roots: usize,
        funding: &eredu_nn::workspace::HostMetadataFunding,
    ) -> Result<
        (
            safemlx::OriginalScopeObserver,
            safemlx::OperationEvalTraversalLayout,
        ),
        Error,
    > {
        let owner = self.owner()?;
        let observer = owner.observer()?;
        let recipe = owner.recipe()?;
        let query = recipe
            .traversal
            .query_control_bytes()
            .and_then(|n| n.checked_mul(2))
            .ok_or(Error::PrefillScopeUnavailable)?;
        funding
            .reserve_metadata(query)
            .map_err(Error::WorkspacePlanning)?;
        let admitted = recipe
            .nested_traversal()
            .ok_or(Error::PrefillScopeUnavailable)?;
        if roots == 0 || roots > admitted.roots() {
            return Err(Error::PrefillScopeUnavailable);
        }
        safemlx::OperationEvent::validate_nested_completion(roots)?;
        let mut limits = admitted.limits();
        limits.roots = roots;
        let exact = safemlx::OperationEvent::eval_traversal_layout(limits)
            .ok_or(Error::PrefillScopeUnavailable)?;
        Ok((observer, exact))
    }
    pub(crate) fn boundary_traversal_control_bytes() -> Option<usize> {
        let frames = [
            size_of::<(&Self, usize, &eredu_nn::workspace::HostMetadataFunding)>(),
            size_of::<RootLoan>(),
            size_of::<crate::backend::nn::workspace::ResidentCompletionRecipe>(),
            size_of::<safemlx::OperationEvalTraversalLayout>(),
            size_of::<safemlx::OperationEvalTraversalLimits>(),
            size_of::<
                Result<
                    (
                        safemlx::OriginalScopeObserver,
                        safemlx::OperationEvalTraversalLayout,
                    ),
                    Error,
                >,
            >(),
            safemlx::OperationEvent::nested_completion_control_bytes::<0>()?,
        ];
        frames
            .into_iter()
            .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
    }
    pub(crate) fn authenticate_scope(
        &self,
        scope: &SubmissionScope,
    ) -> Result<safemlx::OriginalScopeObserver, Error> {
        let observer = self.owner()?.observer()?;
        if !observer.belongs_to(scope) {
            return Err(Error::PrefillControl(
                eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
            ));
        }
        Ok(observer)
    }
    /// Authenticate the actual invocation account as well as its native scope.
    /// A weak projection or an equal plan alone cannot lend a paged source.
    pub(crate) fn authenticate_source(
        &self,
        scope: &SubmissionScope,
        custody: &OriginalHostSourceCustody,
    ) -> Result<safemlx::OriginalScopeObserver, Error> {
        let owner = self.owner()?;
        let RootLoan::Invocation(source) = &owner else {
            return Err(Error::PrefillControl(
                eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
            ));
        };
        if !source
            .source_custody()
            .is_some_and(|actual| actual.same_source(custody))
        {
            return Err(Error::PrefillControl(
                eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
            ));
        }
        let observer = source.scope_observer()?;
        if !observer.belongs_to(scope) {
            return Err(Error::PrefillControl(
                eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
            ));
        }
        Ok(observer)
    }
    pub(crate) fn authenticate_source_control_bytes() -> Option<usize> {
        let frames = [
            size_of::<(&Self, &SubmissionScope, &OriginalHostSourceCustody)>(),
            size_of::<RootLoan>(),
            size_of::<Result<RootLoan, Error>>(),
            size_of::<OriginalHostSourceCustody>(),
            size_of::<Option<OriginalHostSourceCustody>>(),
            size_of::<safemlx::OriginalScopeObserver>(),
            size_of::<Result<safemlx::OriginalScopeObserver, Error>>(),
        ];
        frames
            .into_iter()
            .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
    }
    pub(crate) fn append(&self, value: &Array) -> Result<(), Error> {
        let owner = self.owner()?;
        owner.observer()?;
        owner.append(value)
    }
    pub(crate) fn retire_completed(&self, value: &Array) -> Result<(), Error> {
        let owner = self.owner()?;
        owner.observer()?;
        owner.retire_completed(value)
    }
    pub(crate) fn publication_settlement_control_bytes() -> Option<usize> {
        let frames = [
            size_of::<(&Self, &Array, &safemlx::Stream)>(),
            size_of::<RootLoan>(),
            size_of::<Result<(), Error>>(),
            size_of::<Option<safemlx::PreparedResidentGraph>>(),
            size_of::<std::cell::RefMut<'_, Option<safemlx::PreparedResidentGraph>>>(),
            safemlx::OperationEvent::nested_completion_control_bytes::<1>()?,
            safemlx::OperationEvent::traversal_leaf_control_bytes()?,
            size_of::<safemlx::OriginalScopeObserver>(),
        ];
        frames
            .into_iter()
            .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
    }
    /// Settle the intermediate publication leaf with this owner's admitted
    /// nested worker. The outer root collector remains a one-use completion.
    pub(crate) fn settle_publication(
        &self,
        value: &Array,
        stream: &safemlx::Stream,
    ) -> Result<(), Error> {
        let owner = self.owner()?;
        let observer = owner.observer()?;
        safemlx::OperationEvent::complete_nested([value], stream)?;
        safemlx::OperationEvent::validate_traversal_leaf(value, &observer)?;
        owner.close_construction()
    }
    pub(crate) fn control_bytes() -> Option<usize> {
        let frames = [
            size_of::<Self>(),
            size_of::<RootLoan>(),
            size_of::<Option<RootLoan>>(),
            size_of::<(&Self, &Array)>(),
            size_of::<safemlx::OriginalScopeObserver>(),
            size_of::<std::cell::RefMut<'_, Option<PrefillRoots>>>(),
            size_of::<std::cell::RefMut<'_, PrefillRoots>>(),
            size_of::<Result<(), Error>>(),
            size_of::<(&Self, &SubmissionScope)>(),
            size_of::<Result<safemlx::OriginalScopeObserver, Error>>(),
            size_of::<Rc<dyn InvocationRootSource>>(),
            size_of::<Weak<dyn InvocationRootSource>>(),
            size_of::<Result<RootLoan, Error>>(),
        ];
        frames
            .into_iter()
            .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
    }
}

impl RootsOwner {
    pub(super) fn transient_projection(&self) -> Result<TransientRootsProjection, Error> {
        let projection = self.capture_projection();
        projection.observer()?;
        Ok(TransientRootsProjection(RootSource::Prefill(projection)))
    }
    pub(super) fn capture_projection(&self) -> CaptureProjection {
        CaptureProjection(Rc::downgrade(self.0.as_ref().expect("closed root payload")))
    }
    pub(super) fn ordinary() -> (Self, RootsProjection) {
        let owner = Rc::new(Payload {
            host_construction: RefCell::new(None),
            value: RefCell::new(None),
            initialized: Cell::new(false),
            original: false,
            traversal: Cell::new(None),
            parallel: OnceCell::new(),
            addressable: OnceCell::new(),
        });
        let projection = RootsProjection {
            value: Rc::downgrade(&owner),
            _custody: None,
        };
        (Self(Some(owner)), projection)
    }
    pub(super) fn new(
        runtime: &PrefillRootsRuntime,
        capacity: usize,
        graph: Option<&SubmissionGraphQuota>,
        custody: Option<OriginalPrefillRootCustody>,
        projection: Option<OriginalPrefillRootProjectionCustody>,
    ) -> Result<(Self, RootsProjection), Error> {
        let original = custody.is_some();
        let value = match custody {
            Some(custody) => {
                let graph = graph.ok_or(Error::PrefillScopeUnavailable)?;
                let prepared = PreparedPrefillFailure::try_new(custody).map_err(|error| {
                    let (cause, owner) = error.into_parts();
                    // Fixed cause carries no payload; the same owner covers all
                    // constructor locals and the node deallocates before owner.
                    drop(owner);
                    Error::PrefillRoots(cause.into())
                })?;
                let failure = prepared.try_allocate().map_err(|error| {
                    let (cause, pending) = error.into_parts();
                    drop(pending);
                    Error::PrefillRoots(cause.into())
                })?;
                PrefillRoots::new_retained(runtime, capacity, graph, &failure)
                    .map_err(|e| Error::PrefillRoots(e.into()))?
            }
            None => PrefillRoots::new(runtime, capacity, graph)
                .map_err(|e| Error::PrefillRoots(e.into()))?,
        };
        let owner = Rc::new(Payload {
            host_construction: RefCell::new(None),
            value: RefCell::new(Some(value)),
            initialized: Cell::new(true),
            original,
            traversal: Cell::new(None),
            parallel: OnceCell::new(),
            addressable: OnceCell::new(),
        });
        let view = RootsProjection {
            value: Rc::downgrade(&owner),
            _custody: projection.map(ProjectionCustody::Prefill),
        };
        Ok((Self(Some(owner)), view))
    }
    pub(super) fn with_model_failure(
        runtime: &PrefillRootsRuntime,
        capacity: usize,
        graph: &SubmissionGraphQuota,
        failure: &safemlx::RetainedPrefillFailure,
        controls: eredu_runtime::working_memory::OriginalTextControlGuard,
    ) -> Result<(Self, RootsProjection), Error> {
        let value = PrefillRoots::new_retained(runtime, capacity, graph, failure)
            .map_err(|e| Error::PrefillRoots(e.into()))?;
        let owner = Rc::new(Payload {
            host_construction: RefCell::new(None),
            value: RefCell::new(Some(value)),
            initialized: Cell::new(true),
            original: true,
            traversal: Cell::new(None),
            parallel: OnceCell::new(),
            addressable: OnceCell::new(),
        });
        let projection = RootsProjection {
            value: Rc::downgrade(&owner),
            _custody: Some(ProjectionCustody::Model(controls)),
        };
        Ok((Self(Some(owner)), projection))
    }
    pub(super) fn with_parallel(
        self,
        parallel:Option<crate::backend::runtime::distributed::topology::original_source::parallel::OriginalParallelInvocation>,
    ) -> Result<Self, Error> {
        if let Some(parallel) = parallel {
            let payload = self.0.as_ref().expect("closed root payload");
            if !payload.original || payload.host_construction.borrow().is_some() {
                return Err(Error::PrefillScopeUnavailable);
            }
            payload
                .parallel
                .set(parallel)
                .map_err(|_| Error::PrefillScopeUnavailable)?;
        }
        Ok(self)
    }
    pub(super) fn with_addressable(
        self,
        row: Option<crate::backend::submission_recovery::addressable::AddressableExecutionRow>,
    ) -> Result<Self, Error> {
        if let Some(row) = row {
            let payload = self.0.as_ref().expect("closed root payload");
            if !payload.original || payload.host_construction.borrow().is_some() {
                return Err(Error::PrefillScopeUnavailable);
            }
            payload
                .addressable
                .set(row)
                .map_err(|_| Error::PrefillScopeUnavailable)?;
        }
        Ok(self)
    }
    pub(super) fn with_traversal(
        mut self,
        traversal: Option<crate::backend::nn::workspace::ResidentCompletionRecipe>,
    ) -> Result<Self, Error> {
        let payload = self.0.as_mut().expect("closed root payload");
        // The native root owner is still cold; its only projection is weak.
        // No caller can mutate this recipe after scope installation.
        payload.traversal.set(traversal);
        Ok(self)
    }
    pub(super) fn begin_host(&self, scope: &SubmissionScope) -> Result<(), Error> {
        let payload = self.0.as_ref().expect("closed root payload");
        let Some(recipe) = payload.traversal.get() else {
            return Ok(());
        };
        let observer = safemlx::OriginalScopeObserver::require_current()?;
        if !observer.belongs_to(scope) || payload.host_construction.borrow().is_some() {
            return Err(Error::PrefillScopeUnavailable);
        }
        let mut prepared =
            safemlx::OperationEvent::prepare_resident_graph(recipe.graph, &observer)?;
        if recipe.nested_completions != 0 {
            let traversal = recipe
                .nested_traversal()
                .ok_or(Error::PrefillScopeUnavailable)?;
            prepared.configure_nested_completions(&traversal, recipe.nested_completions)?;
        }
        // No source/RefCell loan is held during the native allocation path.
        *payload.host_construction.borrow_mut() = Some(prepared);
        Ok(())
    }
    pub(super) fn bind_scope(&self, scope: &SubmissionScope) -> Result<(), Error> {
        let payload = self.0.as_ref().expect("closed root payload");
        let mut value = payload
            .value
            .try_borrow_mut()
            .map_err(|_| Error::PrefillScopeReentrant)?;
        value
            .as_mut()
            .ok_or(Error::PrefillScopeUnavailable)?
            .bind_scope(scope)
            .map_err(|e| Error::PrefillRoots(e.into()))
    }
}
impl RootsProjection {
    /// Activate this exact request row around the shared ordinary forward.
    /// A lexical request-channel installation cannot outlive its root owner.
    pub(crate) fn with_addressable<T, E, F>(&self, run: F) -> Result<Result<T, E>, Error>
    where
        F: FnOnce() -> Result<Result<T, E>, Error>,
    {
        let owner = self.owner()?;
        let payload = owner.0.as_ref().expect("closed upgraded root payload");
        match payload.addressable.get() {
            Some(row) => {
                owner.capture_projection().observer()?;
                let mut run = Some(run);
                let mut result = None;
                row.during(&mut || {
                    let value = run.take().ok_or(Error::PrefillScopeUnavailable)?()?;
                    let succeeded = value.is_ok();
                    result = Some(value);
                    Ok(succeeded)
                })?;
                result.ok_or(Error::PrefillScopeUnavailable)
            }
            None => run(),
        }
    }
    /// The shared forward lends this weak original root view beside its control
    /// context. This never changes the optional neural parallel context.
    pub(crate) fn transient_roots(&self) -> Result<TransientRootsProjection, Error> {
        self.owner()?.transient_projection()
    }
    /// Loans the source of this exact active root owner after releasing every
    /// RefCell guard. Only existing Recovery owns the long-lived strong payload.
    pub(crate) fn with_parallel<T,F>(&self,run:F)->Result<T,Error>
    where F:FnOnce(Option<(&crate::backend::runtime::distributed::topology::original_source::parallel::OriginalParallelInvocation,&safemlx::OriginalScopeObserver)>)->Result<T,Error>{
        let owner = self.owner()?;
        let payload = owner.0.as_ref().expect("closed upgraded payload");
        let Some(parallel) = payload.parallel.get() else {
            return run(None);
        };
        parallel.prepare_lending::<F, T>()?;
        let observer = owner.capture_projection().observer()?;
        run(Some((parallel, &observer)))
    }

    /// Reborrow this exact original invocation for the selected publication.
    /// The extra weak root view stays inside its Q-owned invocation.
    pub(crate) fn with_parallel_publication<T, E, F>(
        &self,
        stream: &safemlx::Stream,
        run: F,
    ) -> Result<Result<T, E>, Error>
    where
        F: FnOnce(
            Option<(
                &crate::backend::runtime::distributed::Group,
                &eredu_nn::workspace::HostMetadataFunding,
            )>,
        ) -> Result<T, E>,
    {
        let owner = self.owner()?;
        let payload = owner.0.as_ref().expect("closed publication root owner");
        let Some(parallel) = payload.parallel.get() else {
            return Ok(run(None));
        };
        parallel.prepare_lending::<F, Result<T, E>>()?;
        let observer = owner.capture_projection().observer()?;
        parallel.with_publication_context(&observer, stream, owner.transient_projection()?, run)
    }

    /// Only the request-wide ModelExecution guard can retain another weak view.
    /// Named prefill projection custody remains move-only.
    pub(super) fn clone_model(&self) -> Result<Self, Error> {
        let Some(ProjectionCustody::Model(controls)) = &self._custody else {
            return Err(Error::PrefillScopeUnavailable);
        };
        Ok(Self {
            value: self.value.clone(),
            _custody: Some(ProjectionCustody::Model(controls.clone())),
        })
    }
    /// Only the explicit ordinary empty owner may allocate buffers at completion.
    /// Original owners were populated before source work and never enter this path.
    pub(crate) fn prepare_ordinary(
        &self,
        runtime: &PrefillRootsRuntime,
        capacity: usize,
    ) -> Result<(), Error> {
        let owner = self.owner()?;
        let payload = owner.0.as_ref().expect("closed root payload");
        if payload.original {
            return Ok(());
        }
        if payload.initialized.replace(true) {
            return Err(Error::PrefillScopeUnavailable);
        }
        let value = PrefillRoots::new(runtime, capacity, None)
            .map_err(|e| Error::PrefillRoots(e.into()))?;
        let previous = payload.value.replace(Some(value));
        debug_assert!(previous.is_none());
        drop(previous);
        Ok(())
    }
    fn owner(&self) -> Result<RootsOwner, Error> {
        self.value
            .upgrade()
            .map(|value| RootsOwner(Some(value)))
            .ok_or(Error::PrefillScopeUnavailable)
    }
    /// Only bounded append runs under this loan. It invokes no external callback,
    /// runtime entry, allocation, error formatting or native submission.
    pub(crate) fn append(&self, value: &Array) -> Result<(), Error> {
        let owner = self.owner()?;
        let payload = owner.0.as_ref().expect("closed live root payload");
        let mut loan = payload
            .value
            .try_borrow_mut()
            .map_err(|_| Error::PrefillScopeReentrant)?;
        loan.as_mut()
            .ok_or(Error::PrefillScopeReentrant)?
            .append(value)
            .map_err(|e| Error::PrefillRoots(e.into()))
    }
    pub(crate) fn append_token_validations(&self) -> Result<(), Error> {
        let owner = self.owner()?;
        let payload = owner.0.as_ref().expect("closed live root payload");
        let mut loan = payload
            .value
            .try_borrow_mut()
            .map_err(|_| Error::PrefillScopeReentrant)?;
        crate::backend::nn::tensor::append_active_token_validation_roots(
            loan.as_mut().ok_or(Error::PrefillScopeReentrant)?,
        )
        .map_err(|e| Error::PrefillRoots(e.into()))
    }
    /// Native submission runs with no RefCell/source/state/TLS loan held. The
    /// lexical carrier restores the same partially populated owner before any
    /// error/unwind can release this temporary strong handle.
    pub(crate) fn complete(&self) -> Result<(), Error> {
        self.complete_with_stream(None)
    }
    pub(crate) fn has_prepared_completion(&self) -> Result<bool, Error> {
        let owner = self.owner()?;
        Ok(owner
            .0
            .as_ref()
            .expect("live root payload")
            .traversal
            .get()
            .is_some())
    }
    pub(crate) fn complete_on_stream(&self, stream: &safemlx::Stream) -> Result<(), Error> {
        self.complete_with_stream(Some(stream))
    }
    fn complete_with_stream(&self, selected: Option<&safemlx::Stream>) -> Result<(), Error> {
        complete_roots_owner(self.owner()?, selected)
    }
}
fn complete_roots_owner(
    owner: RootsOwner,
    selected: Option<&safemlx::Stream>,
) -> Result<(), Error> {
    let payload = owner.0.as_ref().expect("closed live root payload");
    if selected.is_some() && !payload.original {
        return Err(Error::PrefillScopeUnavailable);
    }
    if payload.original {
        crate::backend::nn::workspace::ProjectedPagedSources::validate_current_append_completion()?;
    }
    let construction = payload
        .host_construction
        .try_borrow_mut()
        .map_err(|_| Error::PrefillScopeReentrant)?
        .take();
    // Host construction ends before Eval reserves its own Graph prologues.
    // Consumed descriptor/primitive blocks remain with their real births.
    drop(construction);
    let value = payload
        .value
        .try_borrow_mut()
        .map_err(|_| Error::PrefillScopeReentrant)?
        .take()
        .ok_or(Error::PrefillScopeReentrant)?;
    let mut active = Active {
        value: Some(value),
        destination: &payload.value,
    };
    let roots = active.value.as_mut().expect("lexical native root owner");
    if payload.original {
        if let Some(stream) = selected {
            if let Some(layout) = payload.traversal.get() {
                roots.complete_current_scope_on_stream_prepared(stream, &layout.traversal)?;
            } else {
                roots.complete_current_scope_on_stream(stream)?;
            }
        } else {
            if payload.traversal.get().is_some() {
                return Err(Error::PrefillScopeUnavailable);
            }
            roots.complete_current_scope()?;
        }
    } else {
        roots.submit()?;
        roots.wait()?;
        roots.validate()?;
    }
    #[cfg(test)]
    super::test_trace::record(if selected.is_some() {
        super::test_trace::Event::ModelCompleted { roots: roots.len() }
    } else {
        super::test_trace::Event::Completed {
            roots: roots.len(),
            original: payload.original,
        }
    });
    // This is root validation, not all-Scope terminal callback evidence.
    // Recovery retains its strong owner until its independent probe settles.
    Ok(())
}
struct Active<'a> {
    value: Option<PrefillRoots>,
    destination: &'a RefCell<Option<PrefillRoots>>,
}
impl Drop for Active<'_> {
    fn drop(&mut self) {
        // No outstanding loan survives a public/native call. Reentrant complete
        // sees None and returns; it cannot populate or borrow this destination.
        let previous = self.destination.replace(self.value.take());
        debug_assert!(previous.is_none());
        drop(previous);
    }
}
/// Exact concrete Rc layout from the pinned toolchain, plus actual constructor,
/// weak-upgrade, short-loan and unwind carrier representations. Graph/native
/// owner extents are supplied by PrefillRoots::layout separately.
pub(super) fn control_bytes() -> Option<u64> {
    let header = Layout::new::<[Cell<usize>; 2]>()
        .align_to(2)
        .ok()?
        .pad_to_align();
    let allocation = header
        .extend(Layout::new::<Payload>())
        .ok()?
        .0
        .pad_to_align()
        .size();
    let failure = PreparedPrefillFailure::<OriginalPrefillRootCustody>::layout()
        .ok()?
        .total_bytes()?;
    let total = [
        size_of::<Payload>(),
        size_of::<Payload>(),
        size_of::<RootsOwner>(),
        size_of::<RootsProjection>(),
        size_of::<CaptureProjection>(),
        size_of::<Option<CaptureProjection>>(),
        size_of::<Result<safemlx::OriginalScopeObserver, Error>>(),
        size_of::<safemlx::OriginalScopeObserver>(),
        size_of::<safemlx::OriginalScopeObserver>(),
        size_of::<std::cell::Ref<'static, Option<safemlx::PreparedResidentGraph>>>(),
        size_of::<Option<safemlx::error::Exception>>(),
        safemlx::OriginalScopeObserver::control_bytes()?,
        size_of::<(RootsOwner, RootsProjection)>(),
        size_of::<Result<(RootsOwner, RootsProjection), Error>>(),
        size_of::<Rc<Payload>>(),
        size_of::<Weak<Payload>>(),
        size_of::<Option<Rc<Payload>>>(),
        size_of::<std::mem::ManuallyDrop<Rc<Payload>>>(),
        size_of::<Result<Payload, Rc<Payload>>>(),
        size_of::<Option<Payload>>(),
        size_of::<Result<RootsOwner, Error>>(),
        size_of::<Active<'static>>(),
        size_of::<Option<PrefillRoots>>(),
        size_of::<std::cell::RefMut<'static, Option<PrefillRoots>>>(),
        size_of::<Result<(), Error>>(),
    ]
    .into_iter()
    .try_fold(allocation.checked_add(failure)?, usize::checked_add)?;
    u64::try_from(total).ok()
}

#[cfg(test)]
mod tests;
