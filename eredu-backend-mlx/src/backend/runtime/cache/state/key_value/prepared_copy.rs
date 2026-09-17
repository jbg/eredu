//! Borrowed resident KV copying into closed, non-runnable saved slot ownership.

use super::{MlxKeyValueLayerState, MlxKeyValueState};
use crate::backend::runtime::residency::storage::StorageIdentity;
use eredu_runtime::{
    HostSlotInitialization, HostSlotInitializationError, SharedStateLayout,
    working_memory::{
        DecoderCopyAdmissionError, FundedDecoderSlots, InitializedDecoderSlots,
        RegisteredDecoderHostCopy, WorkingMemoryError, WorkingMemoryPool,
    },
};
use safemlx::{Array, Stream, error::Exception};
use std::cell::RefCell;

mod dense;
mod paged;
mod workspace;
pub(in crate::backend::runtime::cache::state) use paged::{PagedWork, PreparedPagedStorageCopy};
pub(crate) use dense::{
    DenseResidentKvPublishError, PreparedDenseResidentKvState, PublishedDenseResidentKvState,
};
pub(crate) use paged::{
    InitializedPagedDenseCopy, InitializedPagedKvCopy, PagedKvPreparationError,
    PreparedPagedKvCopy, PreparedPagedKvHostCopy, SavedPagedKvCopy,
};
pub(crate) use workspace::ProjectedDenseResidentKvCopy;

#[derive(Debug, thiserror::Error)]
pub(crate) enum ResidentKvCopyError {
    #[error("resident decoder copy has no closed paged-storage program")]
    PagedStorage,
    #[error("resident decoder slot initialization: {0}")]
    Initialization(#[from] HostSlotInitializationError),
    #[error("resident decoder copy source or destination: {0}")]
    Memory(#[from] WorkingMemoryError),
    #[error("resident decoder host copy admission: {0}")]
    Admission(#[from] DecoderCopyAdmissionError),
    #[error("resident decoder numerical copy: {0}")]
    Native(#[from] Exception),
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum ResidentKvPreparationError {
    #[error("resident decoder copy has no closed paged-storage program")]
    PagedStorage,
    #[error(transparent)]
    Initialization(#[from] HostSlotInitializationError),
    #[error(transparent)]
    Memory(#[from] WorkingMemoryError),
}
impl From<ResidentKvPreparationError> for ResidentKvCopyError {
    fn from(cause: ResidentKvPreparationError) -> Self {
        match cause {
            ResidentKvPreparationError::PagedStorage => Self::PagedStorage,
            ResidentKvPreparationError::Initialization(cause) => Self::Initialization(cause),
            ResidentKvPreparationError::Memory(cause) => Self::Memory(cause),
        }
    }
}

#[derive(Clone, Copy)]
pub(super) enum KvCopySource<'a> {
    Live(&'a MlxKeyValueState),
    Saved(&'a SavedResidentKvCopy),
}

/// A source-only plan. No array/table clone, collector allocation, native query,
/// completion or registration occurs during preparation/operand visitation.
/// Callers must retain settled, exclusively accessed source values and provide
/// admitted nested numerical work plus recovery for every copied array.
pub(crate) struct PreparedResidentKvCopy<'a> {
    source: KvCopySource<'a>,
}

impl<'a> PreparedResidentKvCopy<'a> {
    pub(crate) fn prepare(source: &'a MlxKeyValueState) -> Result<Self, ResidentKvCopyError> {
        Self::prepare_fixed(source).map_err(Into::into)
    }
    pub(crate) fn prepare_saved(
        source: &'a SavedResidentKvCopy,
    ) -> Result<Self, ResidentKvCopyError> {
        Self::prepare_saved_fixed(source).map_err(Into::into)
    }
    fn validate_resident(&self) -> Result<(), ResidentKvCopyError> {
        self.validate_resident_fixed().map_err(Into::into)
    }
    fn slot_initialization(
        &self,
    ) -> Result<HostSlotInitialization<'a, MlxKeyValueLayerState>, ResidentKvCopyError> {
        self.slot_initialization_fixed().map_err(Into::into)
    }
    pub(crate) fn prepare_fixed(
        source: &'a MlxKeyValueState,
    ) -> Result<Self, ResidentKvPreparationError> {
        let plan = Self {
            source: KvCopySource::Live(source),
        };
        plan.validate_resident_fixed()?;
        // Validate concrete fixed-slot geometry while the actual table is held.
        let _ = plan.slot_initialization_fixed()?;
        Ok(plan)
    }

    pub(crate) fn prepare_saved_fixed(
        source: &'a SavedResidentKvCopy,
    ) -> Result<Self, ResidentKvPreparationError> {
        let plan = Self {
            source: KvCopySource::Saved(source),
        };
        plan.validate_resident_fixed()?;
        let _ = plan.slot_initialization_fixed()?;
        Ok(plan)
    }

    fn len(&self) -> usize {
        self.source.len()
    }

    fn layer(&self, index: usize) -> Option<&'a MlxKeyValueLayerState> {
        self.source.layer(index)
    }

    fn validate_resident_fixed(&self) -> Result<(), ResidentKvPreparationError> {
        for index in 0..self.len() {
            match self.layer(index) {
                Some(MlxKeyValueLayerState::Stateless | MlxKeyValueLayerState::Device(_)) => {}
                Some(MlxKeyValueLayerState::Paged(_)) => {
                    return Err(ResidentKvPreparationError::PagedStorage);
                }
                None => return Err(WorkingMemoryError::IdentityMismatch.into()),
            }
        }
        Ok(())
    }

    fn slot_initialization_fixed(
        &self,
    ) -> Result<HostSlotInitialization<'a, MlxKeyValueLayerState>, ResidentKvPreparationError> {
        self.source.slot_initialization_fixed()
    }

    pub(crate) fn shared_layout(&self) -> &'a SharedStateLayout {
        self.source.shared_layout()
    }

    pub(crate) fn global_layer_start(&self) -> usize {
        self.source.global_layer_start()
    }

    /// Every stored operand, in layer then key/value order. Same-backing source
    /// aliases remain separate operands and produce separate destinations.
    pub(crate) fn visit_operands(&self, visitor: &mut dyn FnMut(&'a Array)) {
        for index in 0..self.len() {
            if let Some(MlxKeyValueLayerState::Device(cache)) = self.layer(index) {
                cache.prepare_isolated_copy().visit_operands(visitor);
            }
        }
    }

    /// Same source branch as host_copy; empty registered tables still count.
    pub(crate) fn registered_source_tables(&self) -> usize {
        self.source.registered_source_tables()
    }

    /// Bind existing host custody without copying or registering source payload.
    /// The enclosing complete-source inventory must also pin shared-layout and
    /// numerical origins; this host association does not prove their health.
    pub(crate) fn host_copy_initialization_peak_bytes(
        &self,
    ) -> Result<u64, ResidentKvPreparationError> {
        self.source.host_copy_initialization_peak_bytes()
    }

    pub(crate) fn host_copy_preparation_bytes(
        &self,
    ) -> Result<usize, eredu_runtime::working_memory::DecoderHostPreparationError> {
        self.source.host_copy_preparation_bytes()
    }

    pub(crate) fn host_copy(
        &self,
        pool: &WorkingMemoryPool,
    ) -> Result<
        RegisteredDecoderHostCopy<'a, MlxKeyValueLayerState, StorageIdentity>,
        ResidentKvCopyError,
    > {
        self.host_copy_with_preparation(pool, None)
    }

    pub(crate) fn host_copy_with_preparation(
        &self,
        pool: &WorkingMemoryPool,
        preparation: Option<&eredu_core::HostPreparationAuthority>,
    ) -> Result<
        RegisteredDecoderHostCopy<'a, MlxKeyValueLayerState, StorageIdentity>,
        ResidentKvCopyError,
    > {
        self.source.host_copy_with_preparation(pool, preparation)
    }

    // Both saved optional cells and installable dense cells consume this exact
    // per-layer program. Source aliases remain distinct copy operands; Concat
    // retains its key/value ordering and every scalar/padding control.
    fn copy_layer_retained(
        &self,
        index: usize,
        stream: &Stream,
        roots: &RefCell<Vec<Array>>,
    ) -> Result<MlxKeyValueLayerState, ResidentKvCopyError> {
        copy_resident_kv_layer_retained(
            self.layer(index)
                .ok_or(WorkingMemoryError::IdentityMismatch)?,
            stream,
            roots,
        )
    }

    /// Fill the actual admitted fixed cells. Partial native intermediates and
    /// outputs enter the caller's collector before later fallible work. This
    /// caller supplies the collector; existing array workers may reserve more
    /// capacity in it. Price/retain that collector and the source owner separately.
    /// This method neither settles nor publishes native work.
    pub(crate) fn copy_retained(
        self,
        mut destination: InitializedDecoderSlots<MlxKeyValueLayerState>,
        stream: &Stream,
        roots: &RefCell<Vec<Array>>,
    ) -> Result<SavedResidentKvCopy, ResidentKvCopyError> {
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
        Ok(SavedResidentKvCopy {
            layers,
            layout: self.shared_layout().clone(),
            global_layer_start: self.global_layer_start(),
        })
    }
}

/// Immutable saved values with actual funded slot payload. This is not a
/// MlxKeyValueState: no inference retention, transaction flag, run grant, mutable
/// state access, allocating Clone or conversion into runnable state is exposed.
/// Shared layout keeps its existing custody; only slot payload is copied here.
pub(crate) struct SavedResidentKvCopy {
    layers: FundedDecoderSlots<MlxKeyValueLayerState>,
    layout: SharedStateLayout,
    global_layer_start: usize,
}

impl SavedResidentKvCopy {
    pub(crate) fn prepare_copy(&self) -> Result<PreparedResidentKvCopy<'_>, ResidentKvCopyError> {
        PreparedResidentKvCopy::prepare_saved(self)
    }

    pub(crate) fn shared_layout(&self) -> &SharedStateLayout {
        &self.layout
    }

    pub(crate) fn global_layer_start(&self) -> usize {
        self.global_layer_start
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

// Shared numerical leaf for both actual KV and KV-only Hybrid layer tables.
pub(in crate::backend::runtime::cache::state) fn copy_resident_kv_layer_retained(
    layer: &MlxKeyValueLayerState,
    stream: &Stream,
    roots: &RefCell<Vec<Array>>,
) -> Result<MlxKeyValueLayerState, ResidentKvCopyError> {
    Ok(match layer {
        MlxKeyValueLayerState::Stateless => MlxKeyValueLayerState::Stateless,
        MlxKeyValueLayerState::Device(cache) => MlxKeyValueLayerState::Device(
            cache.prepare_isolated_copy().copy_retained(stream, roots)?,
        ),
        MlxKeyValueLayerState::Paged(_) => return Err(ResidentKvCopyError::PagedStorage),
    })
}

impl<'a> KvCopySource<'a> {
    pub(super) fn len(&self) -> usize {
        match *self {
            KvCopySource::Live(source) => source.layers.len(),
            KvCopySource::Saved(source) => source.layers.len(),
        }
    }
    pub(super) fn layer(&self, index: usize) -> Option<&'a MlxKeyValueLayerState> {
        match *self {
            KvCopySource::Live(source) => source.layers.slots().get(index),
            KvCopySource::Saved(source) => source.layers.get(index),
        }
    }
    pub(super) fn slot_initialization_fixed(
        &self,
    ) -> Result<HostSlotInitialization<'a, MlxKeyValueLayerState>, ResidentKvPreparationError> {
        Ok(match *self {
            KvCopySource::Live(source) => source.layers.prepare_copy_slots()?,
            KvCopySource::Saved(source) => source.layers.prepare_copy_slots()?,
        })
    }
    pub(crate) fn shared_layout(&self) -> &'a SharedStateLayout {
        match *self {
            KvCopySource::Live(source) => &source.layout,
            KvCopySource::Saved(source) => &source.layout,
        }
    }
    pub(crate) fn global_layer_start(&self) -> usize {
        match *self {
            KvCopySource::Live(source) => source.global_layer_start,
            KvCopySource::Saved(source) => source.global_layer_start,
        }
    }
    pub(crate) fn registered_source_tables(&self) -> usize {
        usize::from(matches!(*self, KvCopySource::Live(_)))
    }
    pub(crate) fn host_copy_initialization_peak_bytes(
        &self,
    ) -> Result<u64, ResidentKvPreparationError> {
        Ok(self
            .slot_initialization_fixed()?
            .initialization_peak_bytes())
    }
    pub(crate) fn host_copy_preparation_bytes(
        &self,
    ) -> Result<usize, eredu_runtime::working_memory::DecoderHostPreparationError> {
        RegisteredDecoderHostCopy::<MlxKeyValueLayerState, StorageIdentity>::preparation_control_bytes(
            matches!(*self, KvCopySource::Live(_)),
        )
    }
    pub(crate) fn host_copy_with_preparation(
        &self,
        pool: &WorkingMemoryPool,
        preparation: Option<&eredu_core::HostPreparationAuthority>,
    ) -> Result<
        RegisteredDecoderHostCopy<'a, MlxKeyValueLayerState, StorageIdentity>,
        ResidentKvCopyError,
    > {
        Ok(match *self {
            KvCopySource::Live(source) => RegisteredDecoderHostCopy::bind_with_preparation(
                pool,
                source.layers.prepare_copy_slots()?,
                StorageIdentity::HostMetadata(
                    source.layers.metadata().identity().registry_key().clone(),
                ),
                preparation,
            )?,
            KvCopySource::Saved(source) => source.layers.prepare_copy()?,
        })
    }
}
