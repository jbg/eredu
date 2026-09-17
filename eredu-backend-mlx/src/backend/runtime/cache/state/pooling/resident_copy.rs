//! Resident Pooling decoder copying into closed, non-runnable funded slots.

use super::{MlxPoolingAttentionCache, MlxPoolingAttentionState};
use crate::backend::runtime::{cache::kv::LiveKeyValueCache, residency::storage::StorageIdentity};
use eredu_runtime::{
    HostSlotInitialization, HostSlotInitializationError, RuntimeState, SharedStateLayout,
    working_memory::{
        DecoderCopyAdmissionError, FundedDecoderSlots, InitializedDecoderSlots,
        RegisteredDecoderHostCopy, WorkingMemoryError, WorkingMemoryPool,
    },
};
use safemlx::{Array, Stream, error::Exception};
use std::cell::RefCell;

#[derive(Debug, thiserror::Error)]
pub(crate) enum ResidentPoolingCopyError {
    #[error("resident pooling decoder copy requires a separate paged manager-copy program")]
    PagedStorage,
    #[error("pooling table copy requires an actual layer table")]
    AbsentTable,
    #[error("resident pooling slot initialization: {0}")]
    Initialization(#[from] HostSlotInitializationError),
    #[error("resident pooling source or destination: {0}")]
    Memory(#[from] WorkingMemoryError),
    #[error("resident pooling host copy admission: {0}")]
    Admission(#[from] DecoderCopyAdmissionError),
    #[error("resident pooling numerical copy: {0}")]
    Native(#[from] Exception),
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum ResidentPoolingPreparationError {
    #[error("resident pooling decoder copy requires a separate paged manager-copy program")]
    PagedStorage,
    #[error("pooling table copy requires an actual layer table")]
    AbsentTable,
    #[error(transparent)]
    Initialization(#[from] HostSlotInitializationError),
    #[error(transparent)]
    Memory(#[from] WorkingMemoryError),
    #[error(transparent)]
    Local(#[from] super::PagedPoolingCopy),
}
impl From<ResidentPoolingPreparationError> for ResidentPoolingCopyError {
    fn from(cause: ResidentPoolingPreparationError) -> Self {
        match cause {
            ResidentPoolingPreparationError::PagedStorage => Self::PagedStorage,
            ResidentPoolingPreparationError::AbsentTable => Self::AbsentTable,
            ResidentPoolingPreparationError::Initialization(cause) => Self::Initialization(cause),
            ResidentPoolingPreparationError::Memory(cause) => Self::Memory(cause),
            ResidentPoolingPreparationError::Local(cause) => {
                Self::Native(Exception::from_source(cause))
            }
        }
    }
}

mod copied_workspace;
mod dense;
pub(crate) use copied_workspace::ProjectedDenseResidentPoolingCopy;
pub(crate) use dense::{
    DenseResidentPoolingPublishError, PreparedDenseResidentPoolingState,
    PublishedDenseResidentPoolingState,
};

enum Source<'a> {
    Live(&'a MlxPoolingAttentionState),
    Saved(&'a SavedResidentPoolingCopy),
}

/// Actual layer access remains borrowed through preparation and copying. No
/// source clones, table allocation, native query or housekeeping occurs here.
pub(crate) struct PreparedResidentPoolingCopy<'a> {
    source: Source<'a>,
    layout: &'a SharedStateLayout,
}
impl<'a> PreparedResidentPoolingCopy<'a> {
    pub(crate) fn prepare(
        source: &'a MlxPoolingAttentionState,
    ) -> Result<Self, ResidentPoolingCopyError> {
        Self::prepare_fixed(source).map_err(Into::into)
    }
    fn prepare_saved(
        source: &'a SavedResidentPoolingCopy,
    ) -> Result<Self, ResidentPoolingCopyError> {
        Self::prepare_saved_fixed(source).map_err(Into::into)
    }
    fn validate_resident(&self) -> Result<(), ResidentPoolingCopyError> {
        self.validate_resident_fixed().map_err(Into::into)
    }
    fn slot_initialization(
        &self,
    ) -> Result<HostSlotInitialization<'a, MlxPoolingAttentionCache>, ResidentPoolingCopyError>
    {
        self.slot_initialization_fixed().map_err(Into::into)
    }
    pub(crate) fn prepare_fixed(
        source: &'a MlxPoolingAttentionState,
    ) -> Result<Self, ResidentPoolingPreparationError> {
        // This leaf only handles actual tables. The closed resident dispatcher
        // selects its separate no-table source branch for stateless DeviceState
        // and preserves complete-source pins without manufacturing a table.
        let slots = source
            .prepare_layer_copy_slots()?
            .ok_or(ResidentPoolingPreparationError::AbsentTable)?;
        let layout = source
            .shared_layout()
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        if slots.len() != layout.layout().len() {
            return Err(WorkingMemoryError::IdentityMismatch.into());
        }
        let plan = Self {
            source: Source::Live(source),
            layout,
        };
        plan.validate_resident_fixed()?;
        Ok(plan)
    }
    pub(crate) fn prepare_saved_fixed(
        source: &'a SavedResidentPoolingCopy,
    ) -> Result<Self, ResidentPoolingPreparationError> {
        let plan = Self {
            source: Source::Saved(source),
            layout: &source.layout,
        };
        let _ = plan.slot_initialization_fixed()?;
        plan.validate_resident_fixed()?;
        Ok(plan)
    }
    fn len(&self) -> usize {
        match self.source {
            Source::Live(source) => source.as_ref().len(),
            Source::Saved(source) => source.layers.len(),
        }
    }
    fn layer(&self, index: usize) -> Option<&'a MlxPoolingAttentionCache> {
        match self.source {
            Source::Live(source) => source.as_ref().get(index),
            Source::Saved(source) => source.layers.get(index),
        }
    }
    fn validate_resident_fixed(&self) -> Result<(), ResidentPoolingPreparationError> {
        if self.len() != self.layout.layout().len() {
            return Err(WorkingMemoryError::IdentityMismatch.into());
        }
        for index in 0..self.len() {
            let layer = self
                .layer(index)
                .ok_or(WorkingMemoryError::IdentityMismatch)?;
            if layer.residency_manager().is_some() {
                return Err(ResidentPoolingPreparationError::PagedStorage);
            }
            let _ = layer.prepare_isolated_copy_fixed()?;
        }
        Ok(())
    }
    fn slot_initialization_fixed(
        &self,
    ) -> Result<HostSlotInitialization<'a, MlxPoolingAttentionCache>, ResidentPoolingPreparationError>
    {
        Ok(match self.source {
            Source::Live(source) => source
                .prepare_layer_copy_slots()?
                .ok_or(ResidentPoolingPreparationError::AbsentTable)?,
            Source::Saved(source) => source.layers.prepare_copy_slots()?,
        })
    }
    pub(crate) fn shared_layout(&self) -> &'a SharedStateLayout {
        self.layout
    }

    pub(crate) fn visit_operands(&self, visitor: &mut dyn FnMut(&'a Array)) {
        for index in 0..self.len() {
            self.layer(index)
                .expect("validated actual layer count")
                .prepare_isolated_copy()
                .expect("validated resident representation")
                .visit_operands(visitor);
        }
    }

    /// All actual held numerical descriptors. This separate traversal is for
    /// source custody/final publication, not a destination-work estimate.
    pub(crate) fn visit_retained_arrays(&self, visitor: &mut dyn FnMut(&'a Array)) {
        for index in 0..self.len() {
            let layer = self.layer(index).expect("validated actual layer count");
            match layer.local() {
                LiveKeyValueCache::Resident(local) => {
                    for array in local.arrays() {
                        visitor(array);
                    }
                }
                LiveKeyValueCache::Paged(_) => unreachable!("validated resident source"),
            }
            match layer {
                MlxPoolingAttentionCache::Local(_) => {}
                MlxPoolingAttentionCache::Compressed { pool, .. } => {
                    for array in pool.arrays() {
                        visitor(array);
                    }
                }
                MlxPoolingAttentionCache::Sparse {
                    pool, index_pool, ..
                } => {
                    for array in pool.arrays().chain(index_pool.arrays()) {
                        visitor(array);
                    }
                }
            }
        }
    }

    /// Same source branch as host_copy; empty registered tables still count.
    pub(crate) fn registered_source_tables(&self) -> usize {
        usize::from(matches!(self.source, Source::Live(_)))
    }

    pub(crate) fn host_copy_initialization_peak_bytes(
        &self,
    ) -> Result<u64, ResidentPoolingPreparationError> {
        Ok(self
            .slot_initialization_fixed()?
            .initialization_peak_bytes())
    }

    pub(crate) fn host_copy_preparation_bytes(
        &self,
    ) -> Result<usize, eredu_runtime::working_memory::DecoderHostPreparationError> {
        RegisteredDecoderHostCopy::<MlxPoolingAttentionCache, StorageIdentity>::preparation_control_bytes(
            matches!(self.source, Source::Live(_)),
        )
    }

    pub(crate) fn host_copy(
        &self,
        pool: &WorkingMemoryPool,
    ) -> Result<
        RegisteredDecoderHostCopy<'a, MlxPoolingAttentionCache, StorageIdentity>,
        ResidentPoolingCopyError,
    > {
        self.host_copy_with_preparation(pool, None)
    }

    pub(crate) fn host_copy_with_preparation(
        &self,
        pool: &WorkingMemoryPool,
        preparation: Option<&eredu_core::HostPreparationAuthority>,
    ) -> Result<
        RegisteredDecoderHostCopy<'a, MlxPoolingAttentionCache, StorageIdentity>,
        ResidentPoolingCopyError,
    > {
        Ok(match self.source {
            Source::Live(_) => {
                let plan = self.slot_initialization()?;
                let key = StorageIdentity::HostMetadata(
                    plan.source_metadata().identity().registry_key().clone(),
                );
                RegisteredDecoderHostCopy::bind_with_preparation(pool, plan, key, preparation)?
            }
            Source::Saved(source) => source.layers.prepare_copy()?,
        })
    }

    fn copy_layer_retained(
        &self,
        index: usize,
        stream: &Stream,
        roots: &RefCell<Vec<Array>>,
    ) -> Result<MlxPoolingAttentionCache, ResidentPoolingCopyError> {
        Ok(self
            .layer(index)
            .ok_or(WorkingMemoryError::IdentityMismatch)?
            .prepare_isolated_copy()?
            .copy_retained(stream, roots)?)
    }

    pub(crate) fn copy_retained(
        self,
        mut destination: InitializedDecoderSlots<MlxPoolingAttentionCache>,
        stream: &Stream,
        roots: &RefCell<Vec<Array>>,
    ) -> Result<SavedResidentPoolingCopy, ResidentPoolingCopyError> {
        destination.validate_source(&self.slot_initialization()?)?;
        for index in 0..self.len() {
            let copied = self.copy_layer_retained(index, stream, roots)?;
            destination
                .push(copied)
                .map_err(|_| WorkingMemoryError::IdentityMismatch)?;
            #[cfg(test)]
            tests::after_layer(index)?;
        }
        let layers = destination
            .finish()
            .map_err(|_| WorkingMemoryError::IdentityMismatch)?;
        Ok(SavedResidentPoolingCopy {
            layers,
            layout: self.layout.clone(),
        })
    }
}

/// Frozen local layer values and closed slot custody. It deliberately has no
/// DeviceState, InferenceRetention, manager, global-index guess or mutable API.
pub(crate) struct SavedResidentPoolingCopy {
    layers: FundedDecoderSlots<MlxPoolingAttentionCache>,
    layout: SharedStateLayout,
}
impl SavedResidentPoolingCopy {
    pub(crate) fn prepare_copy(
        &self,
    ) -> Result<PreparedResidentPoolingCopy<'_>, ResidentPoolingCopyError> {
        PreparedResidentPoolingCopy::prepare_saved(self)
    }
    pub(crate) fn shared_layout(&self) -> &SharedStateLayout {
        &self.layout
    }
    pub(crate) fn retained_slot_bytes(&self) -> u64 {
        self.layers.retained_bytes()
    }
    pub(crate) fn protected_slot_bytes(&self) -> u64 {
        self.layers.protected_bytes()
    }
}

#[cfg(test)]
mod tests;
