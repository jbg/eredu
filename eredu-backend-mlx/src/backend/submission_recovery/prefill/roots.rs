//! Closed completion payload, borrowed through a separately retained projection.
use crate::backend::error::Error;
use eredu_runtime::working_memory::{
    OriginalPrefillRootCustody, OriginalPrefillRootProjectionCustody,
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
    pub(super) fn observer(&self)->Result<safemlx::OriginalScopeObserver,Error>{
        let observer=self.retained_observer()?;
        let current = safemlx::OriginalScopeObserver::require_current()?;
        if !observer.same_scope(&current) {
            return Err(Error::PrefillControl(
                eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
            ));
        }
        Ok(observer)
    }
    pub(super) fn capture_observer(&self)->Result<safemlx::OriginalScopeObserver,Error>{
        let parent=self.retained_observer()?;
        crate::backend::nn::tensor::TokenValidationScope::capture_observer_for(&parent).map_err(Into::into)
    }

}
/// Weak append-only view of this actual original model completion owner.
/// It cannot create or submit a root owner and carries no replacement capacity.
#[derive(Clone)]
pub(crate) struct TransientRootsProjection(CaptureProjection);
impl TransientRootsProjection {
    /// Detach only a leaf whose existing original event is already complete.
    /// No graph is evaluated and no nested attempt is reserved by this handoff.
    pub(crate) fn validate_boundary_leaf(&self,value:&Array)->Result<(),Error>{
        let observer=self.0.observer()?;
        safemlx::OperationEvent::validate_traversal_leaf(value,&observer)?;
        Ok(())
    }

    /// Borrow the actual enclosing model recipe, preserving its admitted DAG
    /// and nested root capacity. An unquoted boundary remains a typed refusal.
    pub(crate) fn boundary_traversal(&self, roots:usize, funding:&eredu_nn::workspace::WorkspaceMetadataFunding)
        ->Result<(safemlx::OriginalScopeObserver,safemlx::OperationEvalTraversalLayout),Error>{
        let observer=self.0.observer()?;
        let owner=RootsOwner(Some(self.0.0.upgrade().ok_or(Error::PrefillScopeUnavailable)?));
        let recipe=owner.0.as_ref().expect("closed boundary root owner").traversal.get()
            .ok_or(Error::PrefillScopeUnavailable)?;
        let query=recipe.traversal.query_control_bytes().and_then(|n|n.checked_mul(2))
            .ok_or(Error::PrefillScopeUnavailable)?;
        funding.reserve_metadata(query).map_err(Error::WorkspacePlanning)?;
        let admitted=recipe.nested_traversal().ok_or(Error::PrefillScopeUnavailable)?;
        if roots==0 || roots>admitted.roots(){return Err(Error::PrefillScopeUnavailable);}
        // This read-only check authenticates the live native bank and remaining
        // attempt. It neither creates a bank nor refunds a consumed frontier.
        safemlx::OperationEvent::validate_nested_completion(roots)?;
        let mut limits=admitted.limits();limits.roots=roots;
        let exact=safemlx::OperationEvent::eval_traversal_layout(limits)
            .ok_or(Error::PrefillScopeUnavailable)?;
        Ok((observer,exact))
    }
    pub(crate) fn boundary_traversal_control_bytes()->Option<usize>{
        let frames=[size_of::<(&Self,usize,&eredu_nn::workspace::WorkspaceMetadataFunding)>(),size_of::<RootsOwner>(),
            size_of::<crate::backend::nn::workspace::ResidentCompletionRecipe>(),
            size_of::<safemlx::OperationEvalTraversalLayout>(),
            size_of::<safemlx::OperationEvalTraversalLimits>(),
            size_of::<Result<(safemlx::OriginalScopeObserver,safemlx::OperationEvalTraversalLayout),Error>>(),
            safemlx::OperationEvent::nested_completion_control_bytes::<0>()?];
        frames.into_iter().try_fold(std::mem::size_of_val(&frames),usize::checked_add)
    }
    /// Authenticate the existing original model owner, including its retained
    /// failure and current scope. This weak projection cannot create a role.
    pub(crate) fn authenticate_scope(
        &self,
        scope: &SubmissionScope,
    ) -> Result<safemlx::OriginalScopeObserver, Error> {
        let observer = self.0.observer()?;
        if !observer.belongs_to(scope) {
            return Err(Error::PrefillControl(
                eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
            ));
        }
        Ok(observer)
    }
    pub(crate) fn append(&self, value: &Array) -> Result<(), Error> {
        self.0.observer()?;
        let owner = RootsOwner(Some(
            self.0.0.upgrade().ok_or(Error::PrefillScopeUnavailable)?,
        ));
        let payload = owner.0.as_ref().expect("closed transient root payload");
        let mut roots = payload
            .value
            .try_borrow_mut()
            .map_err(|_| Error::PrefillScopeReentrant)?;
        roots
            .as_mut()
            .ok_or(Error::PrefillScopeUnavailable)?
            .append(value)
            .map_err(|cause| Error::PrefillRoots(cause.into()))
    }
    pub(crate) fn publication_settlement_control_bytes()->Option<usize> {
        let frames=[std::mem::size_of::<(&Self,&Array,&safemlx::Stream)>(),
            std::mem::size_of::<RootsOwner>(),std::mem::size_of::<Result<(),Error>>(),
            std::mem::size_of::<Option<safemlx::PreparedResidentGraph>>(),
            std::mem::size_of::<std::cell::RefMut<'_,Option<safemlx::PreparedResidentGraph>>>(),
            safemlx::OperationEvent::nested_completion_control_bytes::<1>()?,
            safemlx::OperationEvent::traversal_leaf_control_bytes()?,
            std::mem::size_of::<safemlx::OriginalScopeObserver>()];
        frames.into_iter().try_fold(std::mem::size_of_val(&frames),usize::checked_add)
    }
    /// The publication contribution needs an intermediate settled leaf. Use
    /// the already quoted nested worker while the real resident bank is live;
    /// preserve the outer one-shot root collector for shared session completion.
    pub(crate) fn settle_publication(&self,value:&Array,stream:&safemlx::Stream)->Result<(),Error> {
        let observer=self.0.observer()?;
        let owner=RootsOwner(Some(self.0.0.upgrade().ok_or(Error::PrefillScopeUnavailable)?));
        let payload=owner.0.as_ref().expect("closed publication root owner");
        // The fixed root borrow stays local to the synchronous native worker.
        // On failure that worker restores its bank and Q retains all recovery
        // state. No collector attempt or native source is reset/refunded.
        safemlx::OperationEvent::complete_nested([value],stream)?;
        // Waiting retires the completion record, but its root may still carry
        // the completed event. The existing exact-scope validator publishes
        // `available` and detaches that event without evaluating a new graph.
        // Do this while the model owner is still live, before the following
        // read-only distributed source requires a detached settled leaf.
        safemlx::OperationEvent::validate_traversal_leaf(value,&observer)?;
        let construction=payload.host_construction.try_borrow_mut()
            .map_err(|_|Error::PrefillScopeReentrant)?.take();
        // Model/contribution constructors are now finished. The accepted CPU
        // publication owns its separate exact source bank; final completion
        // will consume the still-filling outer PrefillRoots exactly once.
        drop(construction);
        Ok(())
    }
    pub(crate) fn control_bytes() -> Option<usize> {
        use std::mem::size_of;
        let frames = [
            size_of::<Self>(),
            size_of::<RootsOwner>(),
            size_of::<Option<RootsOwner>>(),
            size_of::<(&Self, &Array)>(),
            size_of::<safemlx::OriginalScopeObserver>(),
            size_of::<std::cell::RefMut<'_, Option<PrefillRoots>>>(),
            size_of::<Result<(), Error>>(),
            size_of::<(&Self, &SubmissionScope)>(),
            size_of::<Result<safemlx::OriginalScopeObserver, Error>>(),
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
        Ok(TransientRootsProjection(projection))
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
            payload.addressable.set(row).map_err(|_| Error::PrefillScopeUnavailable)?;
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
    where F: FnOnce() -> Result<Result<T, E>, Error> {
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
    pub(crate) fn transient_roots(&self)->Result<TransientRootsProjection,Error>{
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
    pub(crate) fn with_parallel_publication<T,E,F>(&self,stream:&safemlx::Stream,run:F)->Result<Result<T,E>,Error>
    where F:FnOnce(Option<(&crate::backend::runtime::distributed::Group,&eredu_nn::workspace::WorkspaceMetadataFunding)>)->Result<T,E> {
        let owner=self.owner()?;
        let payload=owner.0.as_ref().expect("closed publication root owner");
        let Some(parallel)=payload.parallel.get() else{return Ok(run(None))};
        parallel.prepare_lending::<F,Result<T,E>>()?;
        let observer=owner.capture_projection().observer()?;
        parallel.with_publication_context(&observer,stream,owner.transient_projection()?,run)
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
        complete_roots_owner(self.owner()?,selected)
    }
}
fn complete_roots_owner(owner:RootsOwner, selected:Option<&safemlx::Stream>)->Result<(),Error> {
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
