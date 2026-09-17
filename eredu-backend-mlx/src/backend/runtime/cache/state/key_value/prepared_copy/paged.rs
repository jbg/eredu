//! Exact live/saved paged slots using the shared canonical and numerical workers.
use super::*;
use crate::backend::{
    OriginalCopyEnvironment,
    array_copy::{PreparedOriginalCopy, PreparedSavedHostCopy},
    error::Error,
    runtime::cache::{
        kv::PagedKeyValueCache,
        residency::{PagedArrayCopyLayout, PreparedPagedArrayCopy},
        state::{SnapshotProjectionCause, key_value::snapshot_source::PagedSnapshotSource},
    },
};
use eredu_core::HostPreparationAuthority;
use eredu_nn::workspace::WorkspaceContext;
use std::mem::size_of;
mod context;
mod dense;
mod storage;
mod workspace;
pub(crate) use dense::InitializedPagedDenseCopy;
pub(in crate::backend::runtime::cache::state) use storage::{PagedWork, PreparedPagedStorageCopy};

#[derive(Debug, thiserror::Error)]
pub(crate) enum PagedKvPreparationError {
    #[error(transparent)]
    Source(#[from] SnapshotProjectionCause),
    #[error(transparent)]
    Slots(#[from] ResidentKvPreparationError),
}
/// Immutable selected source and cold host construction. No copy/run authority.
pub(crate) struct PreparedPagedKvCopy<'a> {
    source: KvCopySource<'a>,
    storage: PreparedPagedStorageCopy<'a>,
    controls: usize,
}
pub(crate) struct PreparedPagedKvHostCopy<'a> {
    slots: RegisteredDecoderHostCopy<'a, MlxKeyValueLayerState, StorageIdentity>,
    work: PagedWork,
}
pub(crate) struct InitializedPagedKvCopy {
    slots: InitializedDecoderSlots<MlxKeyValueLayerState>,
    work: PagedWork,
}
/// The actual funded saved table. Its paged variant is private and cannot be
/// interpreted as resident storage or directly installed as runnable state.
pub(crate) struct SavedPagedKvCopy {
    inner: SavedResidentKvCopy,
}
impl<'a> PreparedPagedKvCopy<'a> {
    pub(crate) fn prepare_live(
        source: &'a MlxKeyValueState,
    ) -> Result<Option<Self>, PagedKvPreparationError> {
        if source.paged_transaction_branch
            && source
                .layers
                .slots()
                .iter()
                .any(|layer| matches!(layer, MlxKeyValueLayerState::Paged(_)))
        {
            return Err(SnapshotProjectionCause::ChangedAt("live paged transaction branch").into());
        }
        Self::prepare(KvCopySource::Live(source))
    }
    fn prepare(source: KvCopySource<'a>) -> Result<Option<Self>, PagedKvPreparationError> {
        let Some(snapshot) = PagedSnapshotSource::prepare(source.paged_snapshot_state())? else {
            return Ok(None);
        };
        let storage = PreparedPagedStorageCopy::inspect(snapshot)?;
        let mut controls = 0usize;
        let frames = [
            size_of::<Self>(),
            size_of::<PagedWork>(),
            dense::control_bytes().ok_or(SnapshotProjectionCause::Overflow)?,
            size_of::<&mut dyn FnMut(MlxKeyValueLayerState) -> Result<(), Error>>(),
            size_of::<&mut dyn FnMut(&Array) -> Result<(), Error>>(),
            size_of::<&mut dyn FnMut(&PreparedSavedHostCopy) -> Result<(), Error>>(),
            size_of::<bool>(),
            size_of::<(
                &mut PagedWork,
                &PreparedPagedKvCopy<'_>,
                &Stream,
                &RefCell<Vec<Array>>,
                &mut dyn FnMut(&Array) -> Result<(), Error>,
                &mut dyn FnMut(MlxKeyValueLayerState) -> Result<(), Error>,
            )>(),
            size_of::<(
                &mut InitializedPagedKvCopy,
                &mut PreparedOriginalCopy,
                &OriginalCopyEnvironment<'_>,
            )>(),
            size_of::<PreparedPagedKvHostCopy<'_>>(),
            size_of::<InitializedPagedKvCopy>(),
            size_of::<SavedPagedKvCopy>(),
            size_of::<Result<SavedPagedKvCopy, Error>>(),
            size_of::<Option<InitializedDecoderSlots<MlxKeyValueLayerState>>>(),
            size_of::<(&Self, &mut PagedWork, &Stream, &RefCell<Vec<Array>>)>(),
            size_of::<std::ops::Range<usize>>(),
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
        source.slot_initialization_fixed()?;
        Ok(Some(Self {
            source,
            storage,
            controls,
        }))
    }
    pub(crate) fn snapshot_source(&self) -> &PagedSnapshotSource<'a> {
        self.storage.snapshot()
    }
    /// Logical snapshot policy includes the actual independent manager's
    /// source-derived construction/retained-control bound and immutable file
    /// bytes. File aliases perform no numerical copy; the shared snapshot policy
    /// conservatively includes them in its cumulative materialization allowance.
    /// The actual physical Disk reservation stays with the shared file, never Q.
    pub(crate) fn logical_snapshot_manager_bytes(&self) -> Option<u64> {
        self.storage.logical_snapshot_manager_bytes()
    }
    pub(crate) fn shared_layout(&self) -> &'a SharedStateLayout {
        self.source.shared_layout()
    }
    pub(crate) fn global_layer_start(&self) -> usize {
        self.source.global_layer_start()
    }
    pub(crate) fn registered_source_tables(&self) -> usize {
        self.source.registered_source_tables()
    }
    pub(crate) fn host_copy_initialization_peak_bytes(
        &self,
    ) -> Result<u64, ResidentKvPreparationError> {
        self.source.host_copy_initialization_peak_bytes()
    }
    pub(crate) fn host_copy_preparation_bytes(
        &self,
    ) -> Result<usize, eredu_runtime::working_memory::DecoderHostPreparationError> {
        use eredu_runtime::working_memory::DecoderHostPreparationError as E;
        self.source
            .host_copy_preparation_bytes()?
            .checked_add(
                self.storage
                    .host_preparation_bytes(self.controls)
                    .ok_or(E::Overflow)?,
            )
            .ok_or(E::Overflow)
    }
    pub(crate) fn host_copy(
        &self,
        pool: &WorkingMemoryPool,
        host: &HostPreparationAuthority,
    ) -> Result<PreparedPagedKvHostCopy<'a>, Error> {
        let result = (|| {
            // The same shared source worker checks the exact descriptor before
            // any independent manager construction.
            let slots = self
                .source
                .host_copy_with_preparation(pool, Some(host))
                .map_err(|cause| {
                    Error::StorageSource(eredu_core::BackendFailure::from_error(cause))
                })?;
            let work = self.storage.prepare_work(host, self.controls)?;
            Ok(PreparedPagedKvHostCopy { slots, work })
        })();
        result.map_err(|cause| context::failure(cause, host))
    }

    pub(crate) fn copy_retained(
        self,
        initialized: InitializedPagedKvCopy,
        stream: &Stream,
        roots: &RefCell<Vec<Array>>,
    ) -> Result<SavedPagedKvCopy, Error> {
        self.copy_retained_with(initialized, stream, roots, &mut |_| Ok(()))
    }
    pub(crate) fn copy_retained_with(
        self,
        initialized: InitializedPagedKvCopy,
        stream: &Stream,
        roots: &RefCell<Vec<Array>>,
        observe: &mut dyn FnMut(&Array) -> Result<(), Error>,
    ) -> Result<SavedPagedKvCopy, Error> {
        self.copy_retained_with_host(initialized, stream, roots, observe, &mut |_| Ok(()))
    }
    pub(crate) fn copy_retained_with_host(
        self,
        initialized: InitializedPagedKvCopy,
        stream: &Stream,
        roots: &RefCell<Vec<Array>>,
        observe: &mut dyn FnMut(&Array) -> Result<(), Error>,
        observe_host: &mut dyn FnMut(&PreparedSavedHostCopy) -> Result<(), Error>,
    ) -> Result<SavedPagedKvCopy, Error> {
        let InitializedPagedKvCopy { slots, mut work } = initialized;
        // The failed native prefix remains in work until it becomes an owning
        // error. Its pool reservation cannot retire before enclosing recovery.
        let mut slots = Some(slots);
        let result = (|| {
            let destination = slots.as_mut().expect("one admitted table");
            destination
                .validate_source(
                    &self
                        .source
                        .slot_initialization_fixed()
                        .map_err(|cause| Error::Neural(work.context.metadata_source(cause)))?,
                )
                .map_err(Error::PrefillControl)?;
            work.copy_layers(&self, stream, roots, observe, observe_host, &mut |copied| {
                destination
                    .push(copied)
                    .map_err(|_| Error::PrefillControl(WorkingMemoryError::IdentityMismatch))
            })?;
            Ok::<_, Error>(())
        })();
        if let Err(cause) = result {
            // Source pins and complete/partial manager outlive partial cells.
            drop(slots);
            return Err(work.pending.retain_failure(cause));
        }
        let layers = match slots.take().expect("one admitted table").finish() {
            Ok(layers) => layers,
            Err(failure) => {
                drop(failure);
                return Err(work
                    .pending
                    .retain_failure(Error::PrefillControl(WorkingMemoryError::IdentityMismatch)));
            }
        };
        Ok(SavedPagedKvCopy {
            inner: SavedResidentKvCopy {
                layers,
                layout: self.shared_layout().clone(),
                global_layer_start: self.global_layer_start(),
            },
        })
    }
}
impl<'a> PreparedPagedKvHostCopy<'a> {
    pub(crate) fn initialization_peak_bytes(&self) -> u64 {
        self.slots.initialization_peak_bytes()
    }
    pub(crate) fn admit(
        self,
        pool: &WorkingMemoryPool,
        sampling: eredu_runtime::working_memory::RegisteredSamplingCopy<'a, StorageIdentity>,
        complete: eredu_runtime::working_memory::WorkingMemoryStorage<StorageIdentity>,
        limits: eredu_runtime::working_memory::WorkspaceCopyLimits,
    ) -> Result<
        (
            eredu_runtime::working_memory::FundedSamplerCopy,
            InitializedPagedKvCopy,
            eredu_runtime::working_memory::AdmittedWorkspaceCopy,
        ),
        Error,
    > {
        let Self { slots, work } = self;
        let joined = sampling
            .with_decoder_slots(slots, complete)
            .map_err(|cause| {
                context::failure(
                    Error::StorageSource(eredu_core::BackendFailure::from_error(cause)),
                    &work.host,
                )
            })?;
        let (sampler, slots, native) =
            pool.copy_text_components(joined, limits).map_err(|cause| {
                context::failure(
                    Error::StorageSource(eredu_core::BackendFailure::from_error(cause)),
                    &work.host,
                )
            })?;
        Ok((sampler, InitializedPagedKvCopy { slots, work }, native))
    }
}
impl SavedPagedKvCopy {
    pub(crate) fn prepare_copy(&self) -> Result<PreparedPagedKvCopy<'_>, PagedKvPreparationError> {
        PreparedPagedKvCopy::prepare(KvCopySource::Saved(&self.inner))?
            .ok_or_else(|| SnapshotProjectionCause::ChangedAt("saved paged owner has no paged layers").into())
    }
    pub(crate) fn shared_layout(&self) -> &SharedStateLayout {
        self.inner.shared_layout()
    }
    pub(crate) fn global_layer_start(&self) -> usize {
        self.inner.global_layer_start()
    }
    pub(crate) fn retained_slot_bytes(&self) -> u64 {
        self.inner.retained_slot_bytes()
    }
    pub(crate) fn protected_slot_bytes(&self) -> u64 {
        self.inner.protected_slot_bytes()
    }
}

impl PagedWork {
    // One canonical page/tail traversal for saved and installable dense tables.
    // The caller retains partial cells; Pending retains source pins and manager.
    fn copy_layers(
        &mut self,
        source: &PreparedPagedKvCopy<'_>,
        stream: &Stream,
        roots: &RefCell<Vec<Array>>,
        observe: &mut dyn FnMut(&Array) -> Result<(), Error>,
        observe_host: &mut dyn FnMut(&PreparedSavedHostCopy) -> Result<(), Error>,
        push: &mut dyn FnMut(MlxKeyValueLayerState) -> Result<(), Error>,
    ) -> Result<(), Error> {
        self.copy_layers_mode(source, stream, roots, observe, observe_host, push, false)
    }
    fn copy_layers_for_resume(
        &mut self,
        source: &PreparedPagedKvCopy<'_>,
        stream: &Stream,
        roots: &RefCell<Vec<Array>>,
        observe: &mut dyn FnMut(&Array) -> Result<(), Error>,
        push: &mut dyn FnMut(MlxKeyValueLayerState) -> Result<(), Error>,
    ) -> Result<(), Error> {
        self.copy_layers_mode(source, stream, roots, observe, &mut |_| Ok(()), push, true)
    }
    fn copy_layers_mode(
        &mut self,
        source: &PreparedPagedKvCopy<'_>,
        stream: &Stream,
        roots: &RefCell<Vec<Array>>,
        observe: &mut dyn FnMut(&Array) -> Result<(), Error>,
        observe_host: &mut dyn FnMut(&PreparedSavedHostCopy) -> Result<(), Error>,
        push: &mut dyn FnMut(MlxKeyValueLayerState) -> Result<(), Error>,
        retain_host: bool,
    ) -> Result<(), Error> {
        self.copy_pages(stream, roots, observe, observe_host, retain_host)?;
        for index in 0..source.source.len() {
            let layer = source
                .source
                .layer(index)
                .ok_or(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))?;
            let copied = match layer {
                MlxKeyValueLayerState::Stateless => MlxKeyValueLayerState::Stateless,
                MlxKeyValueLayerState::Paged(cache) => {
                    MlxKeyValueLayerState::Paged(self.copy_tail(cache, stream, roots, observe)?)
                }
                MlxKeyValueLayerState::Device(_) => {
                    return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
                }
            };
            push(copied)?;
        }
        self.finish()
    }
}

impl InitializedPagedKvCopy {
    pub(crate) fn prepare_host_destinations(
        &mut self,
        copy: &mut PreparedOriginalCopy,
        environment: &OriginalCopyEnvironment<'_>,
    ) -> Result<(), Error> {
        self.work.prepare_host_destinations(copy, environment)
    }
}
