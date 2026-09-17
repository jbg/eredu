//! Shared canonical manager/page/tail copy, independent of the outer slot type.
use super::*;

/// Borrowed physical source and exact prospective independent manager layout.
/// The caller separately retains every outer/fixed table source and destination.
pub(in crate::backend::runtime::cache::state) struct PreparedPagedStorageCopy<'a> {
    snapshot: PagedSnapshotSource<'a>,
    layout: PagedArrayCopyLayout,
    controls: usize,
}
pub(in crate::backend::runtime::cache::state) struct PagedWork {
    pub(super) pending: PreparedPagedArrayCopy,
    pub(super) context: WorkspaceContext,
    pub(super) host: HostPreparationAuthority,
}
impl<'a> PreparedPagedStorageCopy<'a> {
    pub(in crate::backend::runtime::cache::state) fn inspect(
        snapshot: PagedSnapshotSource<'a>,
    ) -> Result<Self, PagedKvPreparationError> {
        let layout = snapshot
            .manager()
            .inspect_paged_array_copy()
            .map_err(SnapshotProjectionCause::Paged)?;
        let mut controls = layout.control_bytes();
        snapshot.visit_pagers(&mut |cache| -> Result<(), PagedKvPreparationError> {
            cache.original_tail_operands().map_err(ResidentKvPreparationError::Memory)?;
            controls = controls.checked_add(PagedKeyValueCache::copy_tail_program_control_bytes()
                .ok_or(SnapshotProjectionCause::Overflow)?)
                .and_then(|n| n.checked_add(crate::backend::runtime::cache::residency::PreparedIndependentCacheManager::copy_tail_control_bytes()?))
                .and_then(|n| n.checked_add(crate::backend::runtime::cache::residency::PreparedIndependentCacheManager::copy_tail_loan_control_bytes()?))
                .ok_or(SnapshotProjectionCause::Overflow)?;
            Ok(())
        })?;
        let frames = [
            size_of::<Self>(),
            size_of::<PagedWork>(),
            size_of::<Result<Self, PagedKvPreparationError>>(),
            size_of::<Result<PagedWork, Error>>(),
            size_of::<PagedSnapshotSource<'_>>(),
            size_of::<PagedArrayCopyLayout>(),
            size_of::<
                Result<
                    PagedArrayCopyLayout,
                    crate::backend::runtime::cache::residency::CacheSourceError,
                >,
            >(),
            size_of::<(&Self, &HostPreparationAuthority, usize)>(),
            size_of::<(
                &mut PagedWork,
                &Stream,
                &RefCell<Vec<Array>>,
                &mut dyn FnMut(&Array) -> Result<(), Error>,
                &mut dyn FnMut(&PreparedSavedHostCopy) -> Result<(), Error>,
                bool,
            )>(),
            size_of::<(
                &mut PagedWork,
                &PagedKeyValueCache,
                &Stream,
                &RefCell<Vec<Array>>,
                &mut dyn FnMut(&Array) -> Result<(), Error>,
            )>(),
            size_of::<Result<crate::backend::runtime::cache::kv::PagedKeyValueCache, Error>>(),
            size_of::<(
                &mut PagedWork,
                &mut PreparedOriginalCopy,
                &OriginalCopyEnvironment<'_>,
            )>(),
            size_of::<Result<(), Error>>(),
            size_of::<(&mut usize, &PagedKeyValueCache)>(),
            size_of::<Result<(), WorkingMemoryError>>(),
            size_of::<Result<usize, WorkingMemoryError>>(),
            size_of::<&mut dyn FnMut(&PagedKeyValueCache) -> Result<(), PagedKvPreparationError>>(),
        ];
        controls = frames
            .into_iter()
            .try_fold(
                controls
                    .checked_add(std::mem::size_of_val(&frames))
                    .ok_or(SnapshotProjectionCause::Overflow)?,
                usize::checked_add,
            )
            .ok_or(SnapshotProjectionCause::Overflow)?;
        Ok(Self {
            snapshot,
            layout,
            controls,
        })
    }
    pub(in crate::backend::runtime::cache::state) fn snapshot(&self) -> &PagedSnapshotSource<'a> {
        &self.snapshot
    }
    pub(in crate::backend::runtime::cache::state) fn logical_snapshot_manager_bytes(
        &self,
    ) -> Option<u64> {
        u64::try_from(self.layout.control_bytes())
            .ok()?
            .checked_add(self.layout.retained_file_bytes())
    }
    pub(in crate::backend::runtime::cache::state) fn host_preparation_bytes(
        &self,
        caller_controls: usize,
    ) -> Option<usize> {
        context::control_bytes(self.controls.checked_add(caller_controls)?)
    }
    pub(in crate::backend::runtime::cache::state) fn prepare_work(
        &self,
        host: &HostPreparationAuthority,
        caller_controls: usize,
    ) -> Result<PagedWork, Error> {
        let result = (|| {
            let manager = self.snapshot.manager();
            let current = manager.inspect_paged_array_copy().map_err(|cause| {
                Error::StorageSource(eredu_core::BackendFailure::from_error(cause))
            })?;
            if current != self.layout {
                return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
            }
            let controls = self
                .controls
                .checked_add(caller_controls)
                .ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))?;
            let context = context::prepare(controls, host)?;
            let pending = manager.prepare_paged_array_copy(&self.layout, &context)?;
            Ok(PagedWork {
                pending,
                context,
                host: host.clone(),
            })
        })();
        result.map_err(|cause| context::failure(cause, host))
    }
}
impl PagedWork {
    pub(in crate::backend::runtime::cache::state) fn context(&self) -> &WorkspaceContext {
        &self.context
    }
    pub(in crate::backend::runtime::cache::state) fn host(&self) -> &HostPreparationAuthority {
        &self.host
    }
    /// Copy the canonical population exactly once before copying any per-layer tail.
    pub(in crate::backend::runtime::cache::state) fn copy_pages(
        &mut self,
        stream: &Stream,
        roots: &RefCell<Vec<Array>>,
        observe: &mut dyn FnMut(&Array) -> Result<(), Error>,
        observe_host: &mut dyn FnMut(&PreparedSavedHostCopy) -> Result<(), Error>,
        retain_host: bool,
    ) -> Result<(), Error> {
        if retain_host {
            self.pending
                .copy_retained_for_resume(stream, roots, observe)
        } else {
            self.pending
                .copy_retained_with_host(stream, roots, observe, observe_host)
        }
    }
    /// The original tail worker retains native/Host failures under this same manager.
    pub(in crate::backend::runtime::cache::state) fn copy_tail(
        &mut self,
        cache: &PagedKeyValueCache,
        stream: &Stream,
        roots: &RefCell<Vec<Array>>,
        observe: &mut dyn FnMut(&Array) -> Result<(), Error>,
    ) -> Result<PagedKeyValueCache, Error> {
        cache.copy_retained_tail_with(
            self.pending.destination(),
            stream,
            roots,
            &self.context,
            observe,
        )
    }
    pub(in crate::backend::runtime::cache::state) fn finish(&mut self) -> Result<(), Error> {
        self.pending.finish(&self.host)
    }
    pub(in crate::backend::runtime::cache::state) fn retain_failure(self, cause: Error) -> Error {
        self.pending.retain_failure(cause)
    }
    pub(in crate::backend::runtime::cache::state) fn prepare_host_destinations(
        &mut self,
        copy: &mut PreparedOriginalCopy,
        environment: &OriginalCopyEnvironment<'_>,
    ) -> Result<(), Error> {
        self.pending.prepare_host_destinations(copy, environment)
    }
}
