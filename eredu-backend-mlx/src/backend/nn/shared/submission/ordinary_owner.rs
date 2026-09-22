//! Host custody selected by the actual current ordinary physical source.
use super::*;
use crate::backend::nn::workspace::OrdinaryPagedWork;
use crate::backend::runtime::residency::parameter_bank::OrdinaryIndexedLocalSource;
use eredu_core::HostPreparationAuthority;
use safemlx::{ScopedPhysicalBackingObserver, error::Exception};
use std::{
    alloc::Layout,
    mem::{size_of, size_of_val},
};

/// An immutable loan from one admitted Work. Neither field grants permission
/// to execute; the native scope and enclosing request retain that authority.
#[derive(Clone)]
pub(crate) struct OrdinaryExecutionOwner {
    host: HostPreparationAuthority,
    observer: ScopedPhysicalBackingObserver,
    indexed: Option<OrdinaryIndexedLocalSource>,
    paged: Option<OrdinaryPagedWork>,
    checkpoint:
        Option<crate::backend::runtime::cache::state::ordinary_checkpoint::OrdinaryCheckpointWork>,
    metadata_funding: Option<eredu_core::HostMetadataFunding>,
}
impl std::fmt::Debug for OrdinaryExecutionOwner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OrdinaryExecutionOwner")
            .field("indexed", &self.indexed.is_some())
            .finish_non_exhaustive()
    }
}
impl OrdinaryExecutionOwner {
    pub(crate) fn new(
        host: HostPreparationAuthority,
        observer: ScopedPhysicalBackingObserver,
    ) -> Self {
        Self {
            host,
            observer,
            indexed: None,
            paged: None,
            checkpoint: None,
            metadata_funding: None,
        }
    }
    pub(crate) fn with_checkpoint(
        mut self,
        source: Option<
            crate::backend::runtime::cache::state::ordinary_checkpoint::OrdinaryCheckpointWork,
        >,
        funding: eredu_core::HostMetadataFunding,
    ) -> Self {
        self.checkpoint = source;
        self.metadata_funding = Some(funding);
        self
    }
    pub(crate) fn checkpoint(
        &self,
    ) -> Option<&crate::backend::runtime::cache::state::ordinary_checkpoint::OrdinaryCheckpointWork>
    {
        self.checkpoint.as_ref()
    }
    pub(crate) fn metadata_funding(&self) -> Option<&eredu_core::HostMetadataFunding> {
        self.metadata_funding.as_ref()
    }
    pub(crate) fn paged(&self) -> Option<&OrdinaryPagedWork> {
        self.paged.as_ref()
    }
    pub(crate) fn indexed_local(&self) -> Option<&OrdinaryIndexedLocalSource> {
        self.indexed.as_ref()
    }
    pub(crate) fn host(&self) -> &HostPreparationAuthority {
        &self.host
    }
    pub(crate) fn observer(&self) -> &ScopedPhysicalBackingObserver {
        &self.observer
    }
}
struct Node {
    owner: OrdinaryExecutionOwner,
    indexed: RefCell<Option<OrdinaryIndexedLocalSource>>,
    paged: RefCell<Option<OrdinaryPagedWork>>,
    next: RefCell<Option<Reference>>,
}
struct Reference(Option<Rc<Node>>);
impl Reference {
    fn same(&self, other: &Self) -> bool {
        Rc::ptr_eq(self.0.as_ref().unwrap(), other.0.as_ref().unwrap())
    }
}
impl Clone for Reference {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}
impl std::ops::Deref for Reference {
    type Target = Node;
    fn deref(&self) -> &Node {
        self.0.as_deref().unwrap()
    }
}
impl Drop for Reference {
    fn drop(&mut self) {
        if let Some(shared) = self.0.take() {
            if let Some(value) = Rc::into_inner(shared) {
                drop(value);
            }
        }
    }
}
#[derive(Debug, thiserror::Error)]
pub(crate) enum OrdinaryExecutionSourceFailure {
    #[error("ordinary execution source registry unavailable")]
    Registry,
    #[error("current ordinary physical source has no admitted execution owner")]
    MissingSource,
    #[error("current ordinary physical source has duplicate execution owners")]
    DuplicateSource,
}
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
struct RetainedSourceFailure {
    #[source]
    cause: OrdinaryExecutionSourceFailure,
    _host: HostPreparationAuthority,
}
impl Node {
    fn failure(&self, cause: OrdinaryExecutionSourceFailure) -> Exception {
        Exception::from_retained_source(RetainedSourceFailure {
            cause,
            _host: self.owner.host.clone(),
        })
    }
}
thread_local! {
    static OWNERS: RefCell<Option<Reference>> = const { RefCell::new(None) };
}

/// One prepaid registration, retained by its actual Work through completion.
/// Registration order has no role in source selection. A sealed older Work may
/// remain registered while later requests run under different native scopes.
pub(crate) struct OrdinaryExecutionRegistration {
    node: Option<Reference>,
}
impl OrdinaryExecutionRegistration {
    /// One final Rc allocation, the inline registry baseline, and the named
    /// construction/lookup/unlink transports. No capacity-dependent collection
    /// or allocation occurs during lookup, clone, or retirement.
    pub(crate) fn control_bytes() -> Option<usize> {
        let (layout, _) = Layout::new::<[usize; 2]>()
            .extend(Layout::new::<Node>())
            .ok()?;
        let parts = [
            layout.pad_to_align().size(),
            Exception::retained_source_control_bytes::<OrdinaryExecutionSourceFailure>()?,
            Exception::retained_source_control_bytes::<RetainedSourceFailure>()?,
            size_of::<RetainedSourceFailure>(),
            size_of::<(&Node, OrdinaryExecutionSourceFailure)>(),
            size_of::<Node>(),
            size_of::<Self>(),
            size_of::<OrdinaryExecutionOwner>(),
            size_of::<Option<OrdinaryExecutionOwner>>(),
            size_of::<OrdinaryPagedWork>(),
            size_of::<Option<OrdinaryPagedWork>>(),
            size_of::<std::cell::Ref<'static, Option<OrdinaryPagedWork>>>(),
            size_of::<std::cell::RefMut<'static, Option<OrdinaryPagedWork>>>(),
            size_of::<(&Self, OrdinaryPagedWork)>(),
            size_of::<OrdinaryIndexedLocalSource>(),
            size_of::<Option<OrdinaryIndexedLocalSource>>(),
            size_of::<std::cell::Ref<'static, Option<OrdinaryIndexedLocalSource>>>(),
            size_of::<std::cell::RefMut<'static, Option<OrdinaryIndexedLocalSource>>>(),
            size_of::<(&Self, OrdinaryIndexedLocalSource)>(),
            size_of::<RefCell<Option<Reference>>>(),
            size_of::<Reference>(),
            size_of::<Option<Reference>>(),
            size_of::<Option<Reference>>(),
            size_of::<std::cell::Ref<'static, Option<Reference>>>(),
            size_of::<std::cell::RefMut<'static, Option<Reference>>>(),
            size_of::<std::cell::BorrowError>(),
            size_of::<std::cell::BorrowMutError>(),
            size_of::<std::thread::AccessError>(),
            size_of::<Result<Self, Exception>>(),
            size_of::<Result<Option<OrdinaryExecutionOwner>, Exception>>(),
            size_of::<(&ScopedPhysicalBackingObserver, bool, [usize; 3])>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
    /// Bind once from this Work's actual indexed installation before execution.
    /// The loan shares its existing owner and allocates no additional registry.
    pub(crate) fn bind_indexed(&self, source: OrdinaryIndexedLocalSource) -> Result<(), Exception> {
        let node = self
            .node
            .as_ref()
            .ok_or_else(|| Exception::from_source(OrdinaryExecutionSourceFailure::MissingSource))?;
        let mut slot = node
            .indexed
            .try_borrow_mut()
            .map_err(|_| node.failure(OrdinaryExecutionSourceFailure::Registry))?;
        if slot.is_some() {
            return Err(node.failure(OrdinaryExecutionSourceFailure::DuplicateSource));
        }
        *slot = Some(source);
        Ok(())
    }
    /// Attach only this Work's exact immutable paged source before execution.
    pub(crate) fn bind_paged(&self, source: OrdinaryPagedWork) -> Result<(), Exception> {
        let node = self
            .node
            .as_ref()
            .ok_or_else(|| Exception::from_source(OrdinaryExecutionSourceFailure::MissingSource))?;
        let mut slot = node
            .paged
            .try_borrow_mut()
            .map_err(|_| node.failure(OrdinaryExecutionSourceFailure::Registry))?;
        if slot.is_some() {
            return Err(node.failure(OrdinaryExecutionSourceFailure::DuplicateSource));
        }
        *slot = Some(source);
        Ok(())
    }
    /// The caller has already paid control_bytes on this owner's host custody.
    /// Allocate the final node before publishing any registry reference.
    pub(crate) fn new(owner: OrdinaryExecutionOwner) -> Result<Self, Exception> {
        let node = Reference(Some(Rc::new(Node {
            owner,
            indexed: RefCell::new(None),
            paged: RefCell::new(None),
            next: RefCell::new(None),
        })));
        let published = OWNERS.try_with(|head| {
            let mut head = head.try_borrow_mut().map_err(|_| ())?;
            *node.next.borrow_mut() = head.take();
            *head = Some(node.clone());
            Ok::<_, ()>(())
        });
        if !matches!(published, Ok(Ok(()))) {
            return Err(node.failure(OrdinaryExecutionSourceFailure::Registry));
        }
        Ok(Self { node: Some(node) })
    }
}
impl Drop for OrdinaryExecutionRegistration {
    fn drop(&mut self) {
        let Some(owned) = self.node.take() else {
            return;
        };
        // owned keeps the removed payload alive until every RefCell loan ends.
        // The Reference destructor deallocates the Rc before retiring custody.
        let unlinked = OWNERS.try_with(|head| {
            let Ok(mut head) = head.try_borrow_mut() else {
                return false;
            };
            if head.as_ref().is_some_and(|node| node.same(&owned)) {
                *head = owned.next.borrow_mut().take();
                return true;
            }
            let mut previous = head.clone();
            while let Some(node) = previous {
                let next = node.next.borrow().clone();
                if next.as_ref().is_some_and(|next| next.same(&owned)) {
                    *node.next.borrow_mut() = owned.next.borrow_mut().take();
                    return true;
                }
                previous = next;
            }
            true
        });
        if !matches!(unlinked, Ok(true)) {
            // Teardown or an inconsistent loan cannot prove that publication
            // ended. Preserve the complete prepaid chain instead of revoking it.
            std::mem::forget(owned);
        }
    }
}

/// Select only an owner whose exact physical observer is inherited by the
/// current native scope/worker. A live unrelated registration is never a loan.
pub(crate) fn current_ordinary_execution_owner() -> Result<Option<OrdinaryExecutionOwner>, Exception>
{
    OWNERS
        .try_with(|head| {
            let head = head
                .try_borrow()
                .map_err(|_| Exception::from_source(OrdinaryExecutionSourceFailure::Registry))?;
            let mut at = head.clone();
            let mut found = None;
            while let Some(node) = at {
                if node.owner.observer.is_current() {
                    if found.is_some() {
                        return Err(node.failure(OrdinaryExecutionSourceFailure::DuplicateSource));
                    }
                    let mut owner = node.owner.clone();
                    owner.indexed = node
                        .indexed
                        .try_borrow()
                        .map_err(|_| node.failure(OrdinaryExecutionSourceFailure::Registry))?
                        .clone();
                    owner.paged = node
                        .paged
                        .try_borrow()
                        .map_err(|_| node.failure(OrdinaryExecutionSourceFailure::Registry))?
                        .clone();
                    found = Some(owner);
                }
                at = node.next.borrow().clone();
            }
            if found.is_none() && ScopedPhysicalBackingObserver::has_current() {
                return Err(Exception::from_source(
                    OrdinaryExecutionSourceFailure::MissingSource,
                ));
            }
            Ok(found)
        })
        .map_err(|_| Exception::from_source(OrdinaryExecutionSourceFailure::Registry))?
}

pub(super) fn begin<T: Retention>(
    retention: T,
    owner: Option<&OrdinaryExecutionOwner>,
) -> Result<Recovery<T>, Exception> {
    let Some(owner) = owner else {
        return Recovery::begin(retention);
    };
    let prepared =
        crate::backend::submission_recovery::PreparedRecovery::new(retention, owner.host.clone())
            .map_err(|error| {
            // No Scope was begun; the same preparation returns both owners.
            // Their normal drop cannot discard submitted native activity.
            Exception::from_source(error.cause)
        })?;
    let mut recovery = prepared.try_begin().map_err(|error| {
        // try_begin returns this never-started preparation unchanged. Its
        // node has no probe and native rejected before owner transfer.
        Exception::from_source(error.cause)
    })?;
    recovery.configure_scope(|scope| scope.bind_physical_observer(&owner.observer))?;
    Ok(recovery)
}
