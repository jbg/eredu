//! Finite destination constructors for the same independently admitted copy.
use super::*;
use crate::backend::runtime::residency::storage::{
    CopyPublicationLayout, PendingCopyPublication, retain_copy_publication_failure,
};
use eredu_core::HostPreparationAuthority;
use eredu_runtime::working_memory::{WorkspaceCopyCustody, WorkspaceCopyRetention};
use std::{alloc::Layout, mem::size_of};

pub(in crate::composition::mlx::session::model_session) struct SnapshotPublicationPlan {
    native_roots: usize,
    rows: usize,
    publication: CopyPublicationLayout,
    bytes: usize,
}
impl SnapshotPublicationPlan {
    pub(in crate::composition::mlx::session::model_session) fn new(
        native_roots: usize,
        child_metadata: usize,
    ) -> Result<Self, WorkingMemoryError> {
        Self::with_host(native_roots, 0, child_metadata)
    }
    pub(in crate::composition::mlx::session::model_session) fn with_host(
        native_roots: usize,
        host_rows: usize,
        child_metadata: usize,
    ) -> Result<Self, WorkingMemoryError> {
        let rows = native_roots
            .checked_add(host_rows)
            .ok_or(WorkingMemoryError::Overflow)?
            .checked_add(child_metadata)
            .ok_or(WorkingMemoryError::Overflow)?;
        let publication =
            CopyPublicationLayout::with_host(native_roots, host_rows, child_metadata)?;
        let bytes = [
            publication.control_bytes(),
            array::<Array>(native_roots)?,
            array::<safemlx::PreparedArrayClone>(native_roots)?,
            array::<RetainedStorage>(1)?,
            array::<RetainedStoragePublication>(1)?,
            native_roots
                .checked_mul(Array::inspection_clone_handle_bytes())
                .ok_or(WorkingMemoryError::Overflow)?,
            native_roots
                .checked_mul(
                    safemlx::PreparedArrayClone::control_bytes()
                        .ok_or(WorkingMemoryError::UnknownBound)?,
                )
                .ok_or(WorkingMemoryError::Overflow)?,
            RetainedStorage::original_collector_control_bytes(rows)
                .ok_or(WorkingMemoryError::UnknownBound)?,
            size_of::<Self>(),
            size_of::<Result<Self, WorkingMemoryError>>(),
            size_of::<Result<FundedWorkOwner, Error>>(),
            size_of::<Result<RetainedStorage, Error>>(),
            size_of::<UnsubmittedScope>(),
            size_of::<PendingLoan<'static>>(),
            size_of::<Result<PendingLoan<'static>, Error>>(),
            size_of::<std::cell::RefMut<'static, Option<PendingCopyPublication>>>(),
            size_of::<std::cell::RefMut<'static, Vec<Array>>>(),
            size_of::<std::cell::RefMut<'static, Vec<safemlx::PreparedArrayClone>>>(),
            size_of::<std::cell::RefMut<'static, Vec<RetainedStorage>>>(),
            size_of::<std::cell::RefMut<'static, Vec<RetainedStoragePublication>>>(),
            size_of::<std::collections::TryReserveError>(),
            size_of::<Option<safemlx::OriginalBufferBudget>>(),
            size_of::<WorkspaceCopyRetention>(),
            size_of::<&crate::backend::array_copy::PreparedSavedHostCopy>(),
            size_of::<Result<(), Error>>(),
            size_of::<HostPreparationAuthority>(),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
        .ok_or(WorkingMemoryError::Overflow)?;
        // FundedWork/Rc/submission/capture common controls already belong to
        // RootCollectorPlan's B contribution. Only incremental constructors
        // and finite destinations enter this H component.
        Ok(Self {
            native_roots,
            rows,
            publication,
            bytes,
        })
    }
    pub(in crate::composition::mlx::session::model_session) fn control_bytes(&self) -> usize {
        self.bytes
    }

    /// Called before any native scope or submission begins. Failure cancels only
    /// this untouched scope; later copy/publication failures keep normal recovery.
    pub(in crate::composition::mlx::session::model_session) fn construct(
        self,
        scope: WorkingMemoryFundingScope,
        copy: &WorkspaceCopyCustody,
        host: &HostPreparationAuthority,
        budget: Option<&safemlx::OriginalBufferBudget>,
    ) -> Result<FundedWorkOwner, Error> {
        let mut untouched = UnsubmittedScope(Some(scope));
        let result = (|| {
            if self.native_roots != 0 && budget.is_none() {
                return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
            }
            let pending = self
                .publication
                .construct(untouched.get(), copy, host, budget)?;
            let pool = untouched.get().pool().clone();
            let work = FundedWork::allocate(
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                Some(SnapshotWork {
                    pending: RefCell::new(Some(pending)),
                    _copy: copy.retention(),
                    _host: host.clone(),
                }),
            );
            collectors::reserve(&mut work.roots.borrow_mut(), self.native_roots)?;
            collectors::reserve(&mut work.root_clones.borrow_mut(), self.native_roots)?;
            collectors::reserve(&mut work.publications.borrow_mut(), 1)?;
            collectors::reserve(&mut work.inventories.borrow_mut(), 1)?;
            for _ in 0..self.native_roots {
                let clone = safemlx::PreparedArrayClone::try_prepare_for_inspection().map_err(|cause| {
                    Error::from(crate::backend::runtime::residency::manager::ResidencyError::OriginalClone(cause))
                })?;
                work.root_clones.borrow_mut().push(clone);
            }
            let inventory = RetainedStorage::prepare_snapshot_publication(self.rows, &pool, host)?;
            work.inventories.borrow_mut().push(inventory);
            work.collectors_prepared.set(true);
            *work.scope.borrow_mut() = untouched.0.take();
            Ok(work)
        })();
        // All failed prefix owners above retire while H and the untouched
        // account are still live. Preserve the original error, even if account
        // cancellation itself refuses and retains its conservative charge.
        result.map_err(|cause| retain_copy_publication_failure(cause, host))
    }
}
fn array<T>(count: usize) -> Result<usize, WorkingMemoryError> {
    Ok(Layout::array::<T>(count)
        .map_err(|_| WorkingMemoryError::Overflow)?
        .size())
}
struct UnsubmittedScope(Option<WorkingMemoryFundingScope>);
impl UnsubmittedScope {
    fn get(&self) -> &WorkingMemoryFundingScope {
        self.0.as_ref().expect("unsubmitted copy scope")
    }
}
impl Drop for UnsubmittedScope {
    fn drop(&mut self) {
        if let Some(scope) = self.0.take() {
            let _ = scope.certify();
        }
    }
}

// Last in FundedWork: all vectors, roots, registration attempts and native
// publication prefixes retire before the copy account and host control owner.
pub(super) struct SnapshotWork {
    pending: RefCell<Option<PendingCopyPublication>>,
    _copy: WorkspaceCopyRetention,
    _host: HostPreparationAuthority,
}
impl SnapshotWork {
    pub(super) fn bind_host_copy(
        &self,
        copy: &crate::backend::array_copy::PreparedSavedHostCopy,
    ) -> Result<(), Error> {
        let mut pending = PendingLoan::take(&self.pending)?;
        pending
            .value
            .as_mut()
            .expect("owned publication")
            .bind_host_copy(copy)
    }
    pub(super) fn publish(
        &self,
        storage: RetainedStorage,
        scope: &WorkingMemoryFundingScope,
    ) -> Result<RetainedStoragePublication, Error> {
        let mut pending = PendingLoan::take(&self.pending)?;
        pending
            .value
            .as_mut()
            .expect("owned publication")
            .publish(storage, scope)
    }
}
struct PendingLoan<'a> {
    slot: &'a RefCell<Option<PendingCopyPublication>>,
    value: Option<PendingCopyPublication>,
}
impl<'a> PendingLoan<'a> {
    fn take(slot: &'a RefCell<Option<PendingCopyPublication>>) -> Result<Self, Error> {
        let value = slot
            .try_borrow_mut()
            .map_err(|_| Error::PrefillScopeReentrant)?
            .take()
            .ok_or(Error::PrefillControl(
                WorkingMemoryError::PreparationAlreadyStarted,
            ))?;
        Ok(Self {
            slot,
            value: Some(value),
        })
    }
}
impl Drop for PendingLoan<'_> {
    fn drop(&mut self) {
        let mut slot = self.slot.borrow_mut();
        assert!(slot.is_none(), "exclusive snapshot publication slot");
        *slot = self.value.take();
    }
}
