//! Decode reservation and roots borrow the already configured ModelExecution.
//! No child Scope, carrier replacement, or early outer-retirement claim exists.
use super::{memory, Error, RootsOwner, RootsProjection};
use eredu_runtime::working_memory::{
    InferenceExecutionIdentity, InferenceRequest, OriginalTextControlGuard, TextPrefillScopeFacts,
    WorkingMemoryError,
};
use safemlx::{
    OriginalScopeObserver, PrefillRoots, PrefillRootsRuntime, RetainedPrefillFailure,
    SubmissionGraphQuota, SubmissionScope,
};
use std::{
    alloc::Layout,
    cell::{Cell, OnceCell},
    mem::size_of,
    rc::{Rc, Weak},
};

pub(crate) struct ModelExecutionPreparation {
    request: InferenceRequest,
    capacity: usize,
    runtime: PrefillRootsRuntime,
    controls: OriginalTextControlGuard,
    traversal: Option<crate::backend::nn::workspace::ResidentCompletionRecipe>,
    parallel: Option<crate::backend::runtime::distributed::topology::original_source::parallel::OriginalParallelInvocation>,
    addressable: Option<crate::backend::submission_recovery::addressable::AddressableExecutionRow>,
}
impl ModelExecutionPreparation {
    pub(crate) fn new(
        request: &InferenceRequest,
        facts: TextPrefillScopeFacts,
        runtime: PrefillRootsRuntime,
        controls: OriginalTextControlGuard,
    ) -> Result<Self, Error> {
        if request.geometry() != facts.plan().geometry() {
            return Err(memory(WorkingMemoryError::IdentityMismatch));
        }
        controls
            .validate_reservation(request.memory_reservation())
            .map_err(memory)?;
        Ok(Self {
            request: request.clone(),
            capacity: usize::try_from(facts.root_capacity())
                .map_err(|_| memory(WorkingMemoryError::Overflow))?,
            runtime,
            controls,
            traversal: None,
            parallel: None,
            addressable: None,
        })
    }
    pub(crate) fn with_addressable_source(
        mut self,
        recipe: Option<&crate::backend::nn::workspace::ResidentNativeRecipe>,
        owner: Option<&crate::backend::submission_recovery::addressable::AddressableRequestOwner>,
        step: &eredu_runtime::working_memory::InferenceTextStep,
    ) -> Result<Self, Error> {
        self.request
            .validate_same_request(step.request())
            .map_err(memory)?;
        if let Some(owner) = owner {
            owner.validate_request(step.request())?;
        }
        self.addressable = recipe
            .map(|recipe| recipe.addressable_for_step(step))
            .transpose()?
            .flatten()
            .map(|(row, invocation)| {
                owner
                    .ok_or(Error::PrefillScopeUnavailable)?
                    .row(row, 0, invocation)
            })
            .transpose()?;
        Ok(self)
    }
    pub(crate) fn with_resident_recipe(
        mut self,
        recipe: Option<&crate::backend::nn::workspace::ResidentNativeRecipe>,
        step: &eredu_runtime::working_memory::InferenceTextStep,
    ) -> Result<Self, Error> {
        self.request
            .validate_same_request(step.request())
            .map_err(memory)?;
        self.traversal = recipe
            .map(|recipe| recipe.completion_for_step(step))
            .transpose()?;
        self.parallel = recipe
            .map(|recipe| recipe.parallel_for_step(step))
            .transpose()?
            .flatten();
        Ok(self)
    }
}
struct ModelExecution {
    roots: RootsOwner,
    projection: RootsProjection,
    observer: OnceCell<OriginalScopeObserver>,
    checked_out: Cell<bool>,
    completed: Cell<bool>,
    request: InferenceRequest,
}
/// Actual prepared root owner precedes the custody covering its final Rc allocation.
pub(crate) struct ModelExecutionOwner {
    value: Option<Rc<ModelExecution>>,
    controls: OriginalTextControlGuard,
}
impl Drop for ModelExecutionOwner {
    fn drop(&mut self) {
        if let Some(value) = self.value.take() {
            drop(Rc::into_inner(value));
        }
    }
}
#[derive(Clone)]
pub(crate) struct ModelExecutionProjection {
    value: Weak<ModelExecution>,
    controls: OriginalTextControlGuard,
}
impl std::fmt::Debug for ModelExecutionProjection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelExecutionProjection")
            .finish_non_exhaustive()
    }
}
pub(crate) struct ModelReservationGuard(ModelExecutionOwner);
impl ModelExecutionOwner {
    pub(crate) fn prepare(
        preparation: ModelExecutionPreparation,
        graph: &SubmissionGraphQuota,
        failure: &RetainedPrefillFailure,
    ) -> Result<Self, Error> {
        let (roots, projection) = RootsOwner::with_model_failure(
            &preparation.runtime,
            preparation.capacity,
            graph,
            failure,
            preparation.controls.clone(),
        )?;
        let roots = roots
            .with_traversal(preparation.traversal)?
            .with_parallel(preparation.parallel)?
            .with_addressable(preparation.addressable)?;
        Ok(Self {
            value: Some(Rc::new(ModelExecution {
                roots,
                projection,
                observer: OnceCell::new(),
                checked_out: Cell::new(false),
                completed: Cell::new(false),
                request: preparation.request,
            })),
            controls: preparation.controls,
        })
    }
    fn value(&self) -> &ModelExecution {
        self.value.as_ref().expect("closed model root owner")
    }
    pub(crate) fn bind_scope(&self, scope: &SubmissionScope) -> Result<(), Error> {
        self.value().roots.bind_scope(scope)
    }
    pub(crate) fn transient_roots(&self) -> Result<super::roots::TransientRootsProjection, Error> {
        self.require_current()?;
        self.value().roots.transient_projection()
    }
    pub(crate) fn begin_host(&self, scope: &SubmissionScope) -> Result<(), Error> {
        self.require_current()?;
        self.value().roots.begin_host(scope)
    }
    pub(crate) fn observe_scope(&self, scope: &SubmissionScope) -> Result<(), Error> {
        let observer = OriginalScopeObserver::require_current()?;
        if !observer.belongs_to(scope) {
            return Err(Error::PrefillScopeUnavailable);
        }
        self.value()
            .observer
            .set(observer)
            .map_err(|_| Error::PrefillScopeUnavailable)
    }
    /// Clone only this already authenticated role before its owner is released.
    /// It carries no root and cannot be used to submit another operation.
    pub(crate) fn retirement_observer(&self) -> Result<OriginalScopeObserver, Error> {
        self.value()
            .observer
            .get()
            .cloned()
            .ok_or(Error::PrefillScopeUnavailable)
    }
    /// Capture borrows this already bound model role and its finite nested bank.
    pub(crate) fn capture_observer(&self) -> Result<OriginalScopeObserver, Error> {
        let parent = self.retirement_observer()?;
        crate::backend::nn::tensor::TokenValidationScope::capture_observer_for(&parent)
            .map_err(Into::into)
    }
    pub(crate) fn projection(&self) -> ModelExecutionProjection {
        ModelExecutionProjection {
            value: Rc::downgrade(self.value.as_ref().expect("closed model root owner")),
            controls: self.controls.clone(),
        }
    }
    fn require_current(&self) -> Result<(), Error> {
        let actual = OriginalScopeObserver::require_current()?;
        if !self
            .value()
            .observer
            .get()
            .ok_or(Error::PrefillScopeUnavailable)?
            .same_scope(&actual)
        {
            return Err(Error::PrefillScopeUnavailable);
        }
        Ok(())
    }
}
impl ModelExecutionProjection {
    fn owner(&self) -> Result<ModelExecutionOwner, Error> {
        Ok(ModelExecutionOwner {
            value: Some(self.value.upgrade().ok_or(Error::PrefillScopeUnavailable)?),
            controls: self.controls.clone(),
        })
    }
    pub(crate) fn is_live(&self) -> bool {
        self.value.strong_count() != 0
    }
    pub(crate) fn validate_execution(
        &self,
        execution: &InferenceExecutionIdentity,
    ) -> Result<(), Error> {
        let owner = self.owner()?;
        owner
            .value()
            .request
            .validate(execution, owner.value().request.geometry())
            .map_err(memory)
    }
    pub(crate) fn begin(
        &self,
        request: &InferenceRequest,
    ) -> Result<(ModelReservationGuard, RootsProjection), Error> {
        let owner = self.owner()?;
        owner
            .value()
            .request
            .validate_same_request(request)
            .map_err(memory)?;
        owner.require_current()?;
        let roots = owner.value().projection.clone_model()?;
        if owner.value().checked_out.replace(true) {
            return Err(Error::PrefillScopeUnavailable);
        }
        Ok((ModelReservationGuard(owner), roots))
    }
    pub(crate) fn mark_completed(&self) -> Result<(), Error> {
        let owner = self.owner()?;
        owner.require_current()?;
        if !owner.value().checked_out.get() || owner.value().completed.replace(true) {
            return Err(Error::PrefillScopeUnavailable);
        }
        Ok(())
    }
}
impl ModelReservationGuard {
    /// Only acknowledges this inner transaction's validated roots. Publication,
    /// later consumers, Scope sealing, and neutral release stay with outer Recovery.
    pub(crate) fn finish(self) -> Result<(), Error> {
        self.0.require_current()?;
        if !self.0.value().completed.get() {
            return Err(Error::PrefillScopeUnavailable);
        }
        Ok(())
    }
}
/// Per actual decode: same collector's native/fixed host storage plus concrete
/// owner/projection/guard allocations and transport. Graph/Record producer fit
/// remains a separate required fact; no platform/queue bound is inferred here.
pub(crate) fn control_bytes(capacity: usize) -> Option<u64> {
    let root_layout = PrefillRoots::layout(capacity).ok()?;
    let header = Layout::new::<[Cell<usize>; 2]>()
        .align_to(2)
        .ok()?
        .pad_to_align();
    let allocation = header
        .extend(Layout::new::<ModelExecution>())
        .ok()?
        .0
        .pad_to_align()
        .size();
    let fixed = [
        usize::try_from(super::super::retirement::control_bytes()?).ok()?,
        allocation,
        size_of::<ModelExecution>(),
        size_of::<ModelExecutionPreparation>(),
        size_of::<ModelExecutionOwner>(),
        size_of::<ModelExecutionProjection>(),
        size_of::<ModelReservationGuard>(),
        crate::backend::nn::tensor::TokenValidationRole::control_bytes()?,
        size_of::<super::PrefillControlProjection>(),
        size_of::<Option<super::PrefillControlProjection>>(),
        size_of::<super::ReservationGuard>(),
        size_of::<Option<super::ReservationGuard>>(),
        size_of::<Result<super::ReservationGuard, Error>>(),
        size_of::<Result<Option<ModelExecutionPreparation>, Error>>(),
        size_of::<Option<ModelExecutionOwner>>(),
        size_of::<std::cell::Ref<'static, Option<ModelExecutionOwner>>>(),
        size_of::<
            Result<std::cell::Ref<'static, Option<ModelExecutionOwner>>, std::cell::BorrowError>,
        >(),
        size_of::<Result<ModelExecutionOwner, Error>>(),
        size_of::<Result<(ModelReservationGuard, RootsProjection), Error>>(),
        size_of::<Result<(), Error>>(),
        size_of::<Option<ModelExecutionPreparation>>(),
        OriginalScopeObserver::control_bytes()?,
        root_layout.host_bytes()?,
    ]
    .into_iter()
    .try_fold(0usize, usize::checked_add)?;
    u64::try_from(fixed)
        .ok()?
        .checked_add(super::roots::control_bytes()?)
}
