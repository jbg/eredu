//! Exact invocation-local weak binding; only Recovery retains native roots.
use super::*;
use eredu_runtime::working_memory::{OriginalEmbeddedSpeculativeRole, OriginalExternalSpeculativeRole};
use safemlx::StreamCopyPlan;
use std::{
    cell::Cell,
    rc::{Rc, Weak},
};

enum CompletionRole {
    Embedded(OriginalEmbeddedSpeculativeRole),
    External(OriginalExternalSpeculativeRole),
}
impl CompletionRole {
    fn budget_custody(&self) -> OriginalSpeculativeBudgetCustody {
        match self {
            Self::Embedded(role) => role.budget_custody(),
            Self::External(role) => role.budget_custody(),
        }
    }
}

struct Payload {
    roots: RefCell<Option<NestedRootCompletion>>,
    observer: RefCell<Option<OriginalScopeObserver>>,
    active: Cell<bool>,
    issued: Cell<bool>,
    stream: StreamCopyPlan<HostMetadataFunding>,
    role: CompletionRole,
    funding: HostMetadataFunding,
}
/// Native arrays and event stay in Q until its existing Recovery settles.
pub(crate) struct NestedCompletionOwner(Option<Rc<Payload>>);
impl Drop for NestedCompletionOwner {
    fn drop(&mut self) {
        if let Some(value) = self.0.take() {
            drop(Rc::into_inner(value));
        }
    }
}
/// An inactive weak slot grants nothing. Its own H alias covers the weak header
/// even if the session retains the slot after the native Q has retired.
#[derive(Clone)]
pub(crate) struct NestedCompletionProjection {
    value: Weak<Payload>,
    funding: HostMetadataFunding,
}
pub(crate) struct NestedCompletionActivation(NestedCompletionProjection);
impl Drop for NestedCompletionActivation {
    fn drop(&mut self) {
        if let Some(value) = self.0.value.upgrade() {
            value.active.set(false);
            // Do not release a partially accepted completion/event here.
            drop(Rc::into_inner(value));
        }
    }
}
impl NestedCompletionOwner {
    pub(crate) fn control_bytes(
        roots: usize,
        validations: usize,
        stream: &StreamCopyPlan<HostMetadataFunding>,
    ) -> Option<usize> {
        let parts = [
            size_of::<Self>(),
            std::alloc::Layout::new::<[usize; 2]>()
                .extend(std::alloc::Layout::new::<Payload>())
                .ok()?
                .0
                .pad_to_align()
                .size(),
            size_of::<NestedCompletionProjection>(),
            size_of::<NestedCompletionActivation>(),
            size_of::<(NestedCompletionProjection, NestedCompletionActivation)>(),
            size_of::<Result<(NestedCompletionProjection, NestedCompletionActivation), Error>>(),
            size_of::<(
                &Self,
                &OriginalEmbeddedSpeculativeRole,
                &OriginalScopeObserver,
                &Stream,
            )>(),
            size_of::<(
                &NestedCompletionProjection,
                &Stream,
                &mut dyn FnMut(&MlxTensor),
            )>(),
            size_of::<Option<NestedCompletionProjection>>(),
            size_of::<Result<NestedCompletionActivation, Error>>(),
            size_of::<Option<Rc<Payload>>>(),
            size_of::<Result<Self, Error>>(),
            size_of::<StreamCopyPlan<HostMetadataFunding>>(),
            size_of::<RefCell<Option<NestedCompletionProjection>>>(),
            size_of::<std::cell::Ref<'_, Option<NestedCompletionProjection>>>(),
            size_of::<std::cell::RefMut<'_, Option<NestedCompletionProjection>>>(),
            size_of::<Option<OriginalScopeObserver>>(),
            size_of::<Active<'_>>(),
            size_of::<CompletionRole>(),
            size_of::<(&OriginalExternalSpeculativeRole, &CompletionRole)>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)?
            .checked_add(NestedRootCompletion::control_bytes(roots, validations)?)?
            .checked_add(stream.control_bytes()?)?
            .checked_add(stream.source_comparison_control_bytes()?.checked_mul(2)?)
    }
    pub(crate) fn prepare(
        roots: usize,
        validations: usize,
        role: &OriginalEmbeddedSpeculativeRole,
        funding: &HostMetadataFunding,
        stream: StreamCopyPlan<HostMetadataFunding>,
    ) -> Result<Self, Error> {
        Self::prepare_role(roots, validations, || CompletionRole::Embedded(role.clone()), funding, stream)
    }
    pub(crate) fn prepare_external(
        roots: usize, validations: usize, role: &OriginalExternalSpeculativeRole,
        funding: &HostMetadataFunding, stream: StreamCopyPlan<HostMetadataFunding>,
    ) -> Result<Self, Error> {
        Self::prepare_role(roots, validations, || CompletionRole::External(role.clone()), funding, stream)
    }
    fn prepare_role(
        roots: usize, validations: usize, role: impl FnOnce() -> CompletionRole,
        funding: &HostMetadataFunding, stream: StreamCopyPlan<HostMetadataFunding>,
    ) -> Result<Self, Error> {
        funding
            .reserve_metadata(
                Self::control_bytes(roots, validations, &stream)
                    .ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))?,
            )
            .map_err(Error::WorkspacePlanning)?;
        let role = role();
        let roots =
            NestedRootCompletion::prepare(roots, validations, role.budget_custody(), funding)?;
        Ok(Self(Some(Rc::new(Payload {
            roots: RefCell::new(Some(roots)),
            observer: RefCell::new(None),
            active: Cell::new(false),
            issued: Cell::new(false),
            stream,
            role,
            funding: funding.clone(),
        }))))
    }
    pub(crate) fn activate(
        &self,
        role: &OriginalEmbeddedSpeculativeRole,
        observer: &OriginalScopeObserver,
        stream: &Stream,
    ) -> Result<(NestedCompletionProjection, NestedCompletionActivation), Error> {
        self.activate_matching(observer, stream, |stored| {
            matches!(stored, CompletionRole::Embedded(actual) if actual.same_role(role))
        })
    }
    pub(crate) fn activate_external(
        &self, role: &OriginalExternalSpeculativeRole,
        observer: &OriginalScopeObserver, stream: &Stream,
    ) -> Result<(NestedCompletionProjection, NestedCompletionActivation), Error> {
        self.activate_matching(observer, stream, |stored| {
            matches!(stored, CompletionRole::External(actual) if actual.same_role(role))
        })
    }
    fn activate_matching(
        &self, observer: &OriginalScopeObserver, stream: &Stream,
        matches_role: impl FnOnce(&CompletionRole) -> bool,
    ) -> Result<(NestedCompletionProjection, NestedCompletionActivation), Error> {
        let value = self.0.as_ref().ok_or(Error::PrefillScopeUnavailable)?;
        if !matches_role(&value.role)
            || !value.stream.matches_source(stream)
            || !observer.same_scope(&OriginalScopeObserver::require_current()?)
            || value.issued.replace(true)
        {
            return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
        }
        *value
            .observer
            .try_borrow_mut()
            .map_err(|_| Error::PrefillScopeReentrant)? = Some(observer.clone());
        value.active.set(true);
        let projection = NestedCompletionProjection {
            value: Rc::downgrade(value),
            funding: value.funding.clone(),
        };
        Ok((projection.clone(), NestedCompletionActivation(projection)))
    }
    pub(crate) fn validate_complete(&self) -> Result<(), Error> {
        self.0
            .as_ref()
            .ok_or(Error::PrefillScopeUnavailable)?
            .roots
            .try_borrow()
            .map_err(|_| Error::PrefillScopeReentrant)?
            .as_ref()
            .ok_or(Error::PrefillScopeUnavailable)?
            .validate_complete()
    }
}
struct Active<'a> {
    value: Option<NestedRootCompletion>,
    destination: &'a RefCell<Option<NestedRootCompletion>>,
}
impl Drop for Active<'_> {
    fn drop(&mut self) {
        *self.destination.borrow_mut() = self.value.take();
    }
}
impl NestedCompletionProjection {
    pub(crate) fn is_active(&self) -> bool {
        let value = self.value.upgrade();
        let active = value.as_ref().is_some_and(|value| value.active.get());
        if let Some(value) = value {
            drop(Rc::into_inner(value));
        }
        active
    }
    pub(crate) fn complete(
        &self,
        stream: &Stream,
        visit: impl FnOnce(&mut dyn FnMut(&MlxTensor)) -> Result<(), Error>,
    ) -> Result<(), Error> {
        let owner = NestedCompletionOwner(Some(
            self.value.upgrade().ok_or(Error::PrefillScopeUnavailable)?,
        ));
        let payload = owner.0.as_ref().expect("closed completion owner");
        if !payload.active.get() || !payload.stream.matches_source(stream) {
            return Err(Error::PrefillScopeUnavailable);
        }
        let observer = payload
            .observer
            .try_borrow()
            .map_err(|_| Error::PrefillScopeReentrant)?
            .as_ref()
            .ok_or(Error::PrefillScopeUnavailable)?
            .clone();
        if !observer.same_scope(&OriginalScopeObserver::require_current()?) {
            return Err(Error::PrefillScopeUnavailable);
        }
        let value = payload
            .roots
            .try_borrow_mut()
            .map_err(|_| Error::PrefillScopeReentrant)?
            .take()
            .ok_or(Error::PrefillScopeReentrant)?;
        let mut active = Active {
            value: Some(value),
            destination: &payload.roots,
        };
        // State visits and temporary source loans end inside the worker before
        // submission. No RefCell loan is held over native submit/wait.
        active
            .value
            .as_mut()
            .expect("loaned completion worker")
            .complete(visit, &observer, stream)
    }
}
