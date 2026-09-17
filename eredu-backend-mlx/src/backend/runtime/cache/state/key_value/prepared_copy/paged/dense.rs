//! Same canonical page/tail copies into the original prompt's dense host table.
use super::super::dense::PreparedDenseResidentKvState;
use super::*;
use eredu_runtime::working_memory::{
    InferencePreparationStage, InitializedDenseDecoderSlots, RegisteredDenseDecoderInitialization,
    WorkingMemoryFundingRun, WorkingMemoryFundingScope, WorkingMemoryStorage,
};

pub(crate) struct InitializedPagedDenseCopy<'a> {
    slots: InitializedDenseDecoderSlots<
        'a,
        MlxKeyValueLayerState,
        MlxKeyValueLayerState,
        StorageIdentity,
    >,
    work: PagedWork,
}
impl<'a> PreparedPagedKvCopy<'a> {
    pub(crate) fn dense_initialization_peak_bytes_fixed(
        &self,
    ) -> Result<u64, crate::backend::runtime::cache::state::ResidentDecoderPreparationError> {
        Ok(self
            .source
            .slot_initialization_fixed()?
            .for_dense_destination::<MlxKeyValueLayerState>()?
            .initialization_peak_bytes())
    }
    pub(crate) fn dense_host_preparation_bytes(
        &self,
    ) -> Result<usize, eredu_runtime::working_memory::DecoderHostPreparationError> {
        use eredu_runtime::working_memory::DecoderHostPreparationError as E;
        let nested =
            eredu_runtime::HostMetadataKey::maximum_clone_storage_bytes().ok_or(E::UnknownBound)?;
        let slots = RegisteredDenseDecoderInitialization::<
            MlxKeyValueLayerState,
            MlxKeyValueLayerState,
            StorageIdentity,
        >::preparation_control_bytes(
            matches!(self.source, KvCopySource::Live(_)), nested
        )?;
        slots
            .checked_add(self.storage.host_preparation_bytes(self.controls).ok_or(E::Overflow)?)
            .ok_or(E::Overflow)
    }
    pub(crate) fn construct_dense(
        &self,
        stage: InferencePreparationStage,
        funding: &WorkingMemoryFundingRun,
        complete: WorkingMemoryStorage<StorageIdentity>,
        pool: &WorkingMemoryPool,
        host: &HostPreparationAuthority,
    ) -> Result<(InitializedPagedDenseCopy<'a>, WorkingMemoryFundingScope), Error> {
        let PreparedPagedKvHostCopy { slots, work } = self.host_copy(pool, host)?;
        let result = (|| {
            let slots = slots.for_dense_destination().map_err(|cause| {
                Error::StorageSource(eredu_core::BackendFailure::from_error(cause))
            })?;
            stage
                .construct_dense_decoder(slots, funding, complete)
                .map_err(|cause| {
                    Error::StorageSource(eredu_core::BackendFailure::from_error(cause))
                })
        })()
        .map_err(|cause| context::failure(cause, &work.host))?;
        Ok((
            InitializedPagedDenseCopy {
                slots: result.0,
                work,
            },
            result.1,
        ))
    }
    pub(crate) fn copy_dense_retained_with(
        self,
        initialized: InitializedPagedDenseCopy<'a>,
        stream: &Stream,
        roots: &RefCell<Vec<Array>>,
        observe: &mut dyn FnMut(&Array) -> Result<(), Error>,
    ) -> Result<PreparedDenseResidentKvState<'a>, Error> {
        let InitializedPagedDenseCopy { slots, mut work } = initialized;
        let mut slots = Some(slots);
        let result = (|| {
            let destination = slots.as_mut().expect("one admitted dense table");
            let source = self
                .source
                .slot_initialization_fixed()
                .map_err(|cause| Error::Neural(work.context.metadata_source(cause)))?
                .for_dense_destination::<MlxKeyValueLayerState>()
                .map_err(|cause| Error::Neural(work.context.metadata_source(cause)))?;
            destination
                .validate_source(&source)
                .map_err(Error::PrefillControl)?;
            work.copy_layers_for_resume(&self, stream, roots, observe, &mut |copied| {
                destination
                    .push(copied)
                    .map_err(|_| Error::PrefillControl(WorkingMemoryError::IdentityMismatch))
            })
        })();
        if let Err(cause) = result {
            drop(slots);
            return Err(work.pending.retain_failure(cause));
        }
        let layers = match slots.take().expect("one admitted dense table").finish() {
            Ok(layers) => layers,
            Err(failure) => {
                drop(failure);
                return Err(work
                    .pending
                    .retain_failure(Error::PrefillControl(WorkingMemoryError::IdentityMismatch)));
            }
        };
        Ok(PreparedDenseResidentKvState::from_paged_parts(
            layers,
            self.shared_layout().clone(),
            self.global_layer_start(),
        ))
    }
}
pub(super) fn control_bytes() -> Option<usize> {
    let parts = [
        size_of::<InitializedPagedDenseCopy<'_>>(),
        size_of::<
            Option<
                InitializedDenseDecoderSlots<
                    '_,
                    MlxKeyValueLayerState,
                    MlxKeyValueLayerState,
                    StorageIdentity,
                >,
            >,
        >(),
        size_of::<
            RegisteredDenseDecoderInitialization<
                '_,
                MlxKeyValueLayerState,
                MlxKeyValueLayerState,
                StorageIdentity,
            >,
        >(),
        size_of::<Result<(InitializedPagedDenseCopy<'_>, WorkingMemoryFundingScope), Error>>(),
        size_of::<PreparedDenseResidentKvState<'_>>(),
        size_of::<Result<PreparedDenseResidentKvState<'_>, Error>>(),
        size_of::<(
            &mut PagedWork,
            &PreparedPagedKvCopy<'_>,
            &Stream,
            &RefCell<Vec<Array>>,
            &mut dyn FnMut(&Array) -> Result<(), Error>,
            &mut dyn FnMut(MlxKeyValueLayerState) -> Result<(), Error>,
        )>(),
    ];
    parts
        .into_iter()
        .try_fold(std::mem::size_of_val(&parts), usize::checked_add)
}
