//! Closed native decoder-copy dispatch, distinct from runnable model state.

use super::{
    MlxHybridState, MlxKeyValueLayerState, MlxKeyValueState,
    MlxPoolingAttentionCache, MlxPoolingAttentionState, PreparedResidentKvCopy,
    ResidentKvCopyError, SavedResidentKvCopy,
    hybrid::{
        InitializedHybridGroupCopy, PreparedHybridGroupHostCopy, PreparedHybridGroupedCopy,
        SavedHybridGroupedCopy,
    },
    key_value::{
        InitializedPagedKvCopy, PagedKvPreparationError, PreparedPagedKvCopy,
        PreparedPagedKvHostCopy, SavedPagedKvCopy,
    },
    pooling::{PreparedResidentPoolingCopy, ResidentPoolingCopyError, SavedResidentPoolingCopy},
};
use crate::backend::{error::Error, runtime::residency::storage::StorageIdentity};
use eredu_runtime::{
    SharedStateLayout,
    working_memory::{
        AdmittedWorkspaceCopy, FundedSamplerCopy, InitializedDecoderSlots,
        RegisteredDecoderHostCopy, RegisteredSamplingCopy, WorkingMemoryError, WorkingMemoryPool,
        WorkingMemoryStorage, WorkspaceCopyLimits,
    },
};
use safemlx::{Array, Stream};
use std::{cell::RefCell, fmt};

mod dense;
mod projection_control;
mod stateless;
pub(crate) use dense::{PreparedResidentDenseCopy, PublishedResidentDecoderState};
use stateless::{PreparedStatelessPoolingCopy, SavedStatelessPoolingCopy};

fn kv_error(error: ResidentKvCopyError) -> Error {
    match error {
        ResidentKvCopyError::PagedStorage => unknown(),
        error => Error::Other(Box::new(error)),
    }
}
fn pooling_error(error: ResidentPoolingCopyError) -> Error {
    match error {
        ResidentPoolingCopyError::PagedStorage | ResidentPoolingCopyError::AbsentTable => unknown(),
        error => Error::Other(Box::new(error)),
    }
}
fn unknown() -> Error {
    Error::Other(Box::new(WorkingMemoryError::UnknownBound))
}
fn mismatch() -> Error {
    Error::Other(Box::new(WorkingMemoryError::IdentityMismatch))
}

/// Source inspection causes contain no allocated diagnostic or native work.
#[derive(Debug, thiserror::Error)]
pub(crate) enum ResidentDecoderPreparationError {
    #[error(transparent)]
    Paged(#[from] PagedKvPreparationError),
    #[error(transparent)]
    Memory(#[from] WorkingMemoryError),
    #[error(transparent)]
    Initialization(#[from] eredu_runtime::HostSlotInitializationError),
    #[error(transparent)]
    KeyValue(#[from] super::key_value::ResidentKvPreparationError),
    #[error(transparent)]
    Pooling(#[from] super::pooling::ResidentPoolingPreparationError),
    #[error(transparent)]
    Compressed(#[from] crate::backend::runtime::cache::kv::InvalidResidentCopy),
    #[error(transparent)]
    Boundary(#[from] eredu_runtime::replicated_session::RuntimeInspectionBoundary),
    #[error("{0}")]
    Policy(&'static str),
}
impl ResidentDecoderPreparationError {
    pub(crate) fn into_error(self) -> Error {
        match self {
            Self::Paged(cause) => Error::Other(Box::new(cause)),
            Self::KeyValue(cause) => kv_error(cause.into()),
            Self::Pooling(cause) => pooling_error(cause.into()),
            Self::Compressed(cause) => safemlx::error::Exception::from_source(cause).into(),
            Self::Memory(cause) => Error::Other(Box::new(cause)),
            Self::Initialization(cause) => Error::Other(Box::new(cause)),
            Self::Boundary(cause) => Error::Other(Box::new(cause)),
            Self::Policy(reason) => Error::ArchitectureModel(reason.into()),
        }
    }
}

/// An exact borrowed native representation. Private variants prevent supplying
/// a slot type selected independently from its real source. This remains a
/// component plan, not whole-snapshot or runnable-state authority.
pub(crate) struct PreparedResidentDecoderCopy<'a> {
    storage: PreparedStorage<'a>,
}
enum PreparedStorage<'a> {
    Paged(PreparedPagedKvCopy<'a>),
    KeyValue(PreparedResidentKvCopy<'a>),
    HybridGrouped(PreparedHybridGroupedCopy<'a>),
    Pooling(PreparedResidentPoolingCopy<'a>),
    StatelessPooling(PreparedStatelessPoolingCopy<'a>),
}

impl<'a> PreparedResidentDecoderCopy<'a> {
    pub(crate) fn is_paged(&self) -> bool {
        matches!(&self.storage, PreparedStorage::Paged(_))
            || matches!(&self.storage,PreparedStorage::HybridGrouped(plan) if plan.is_paged())
    }
    fn paged_source(&self) -> Option<&(dyn SnapshotArraySources + '_)> {
        match &self.storage {
            PreparedStorage::Paged(plan) => Some(plan.snapshot_source()),
            PreparedStorage::HybridGrouped(plan) if plan.is_paged() => Some(plan),
            _ => None,
        }
    }
    /// Consumes the exact borrowed representation for its closed dense worker.
    /// No table is fabricated or converted when the source is not ordinary KV.
    pub(crate) fn into_dense_key_value(self) -> Option<PreparedResidentKvCopy<'a>> {
        match self.storage {
            PreparedStorage::KeyValue(plan) => Some(plan),
            PreparedStorage::Paged(_)
            | PreparedStorage::HybridGrouped(_)
            | PreparedStorage::Pooling(_)
            | PreparedStorage::StatelessPooling(_) => None,
        }
    }

    /// Borrows the selected dense KV representation. Other decoder mechanisms
    /// need their own closed live-destination projection and remain separate.
    pub(crate) fn dense_key_value(&self) -> Option<&PreparedResidentKvCopy<'a>> {
        match &self.storage {
            PreparedStorage::KeyValue(plan) => Some(plan),
            PreparedStorage::Paged(_)
            | PreparedStorage::HybridGrouped(_)
            | PreparedStorage::Pooling(_)
            | PreparedStorage::StatelessPooling(_) => None,
        }
    }

    /// Visits actual registered child tables; frozen grouped children instead
    /// have authenticated private holds. Never fabricates funded table entries.
    pub(crate) fn visit_registered_child_metadata(
        &self,
        visitor: &mut dyn FnMut(&eredu_runtime::HostSlotMetadata) -> Result<(), Error>,
    ) -> Result<(), Error> {
        match &self.storage {
            PreparedStorage::HybridGrouped(plan) => plan.visit_registered_child_metadata(visitor),
            _ => Ok(()),
        }
    }

    pub(crate) fn visit_registered_child_metadata_borrowed(
        &self,
        visitor: &mut dyn FnMut(&eredu_runtime::HostSlotMetadata),
    ) {
        match &self.storage {
            PreparedStorage::HybridGrouped(plan) => {
                plan.visit_registered_child_metadata_borrowed(visitor)
            }
            _ => {}
        }
    }

    pub(crate) fn into_dense(self) -> Result<PreparedResidentDenseCopy<'a>, Error> {
        self.into_dense_fixed()
            .map_err(ResidentDecoderPreparationError::into_error)
    }
    pub(crate) fn into_dense_fixed(
        self,
    ) -> Result<PreparedResidentDenseCopy<'a>, ResidentDecoderPreparationError> {
        match self.storage {
            PreparedStorage::KeyValue(plan) => Ok(PreparedResidentDenseCopy::KeyValue(plan)),
            PreparedStorage::HybridGrouped(plan) => {
                Ok(PreparedResidentDenseCopy::HybridGrouped(plan))
            }
            PreparedStorage::Pooling(plan) => Ok(PreparedResidentDenseCopy::Pooling(plan)),
            PreparedStorage::Paged(plan) => Ok(PreparedResidentDenseCopy::Paged(plan)),
            PreparedStorage::StatelessPooling(_) => Err(WorkingMemoryError::UnknownBound.into()),
        }
    }

    pub(crate) fn hybrid(source: &'a MlxHybridState) -> Result<Self, Error> {
        Self::hybrid_fixed(source).map_err(ResidentDecoderPreparationError::into_error)
    }

    pub(crate) fn key_value(source: &'a MlxKeyValueState) -> Result<Self, Error> {
        Self::key_value_fixed(source).map_err(ResidentDecoderPreparationError::into_error)
    }

    pub(crate) fn pooling(source: &'a MlxPoolingAttentionState) -> Result<Self, Error> {
        let storage = if source
            .prepare_layer_copy_slots()
            .map_err(|error| Error::Other(Box::new(error)))?
            .is_some()
        {
            PreparedStorage::Pooling(
                PreparedResidentPoolingCopy::prepare(source).map_err(pooling_error)?,
            )
        } else {
            PreparedStorage::StatelessPooling(PreparedStatelessPoolingCopy::prepare(source)?)
        };
        Ok(Self { storage })
    }

    pub(crate) fn hybrid_fixed(
        source: &'a MlxHybridState,
    ) -> Result<Self, ResidentDecoderPreparationError> {
        // Every Hybrid layer owns a fixed child table, including a zero-slot
        // table. The checked source must retain that topology: after a saved
        // copy the dense group worker publishes each actual child attachment
        // under the same prompt claim before releasing the outer live state.
        // Testing only nonzero role payloads would choose an outer-only copy
        // and leave newly constructed empty child identities unregistered.
        Ok(Self {
            storage: PreparedStorage::HybridGrouped(
                PreparedHybridGroupedCopy::prepare_fixed(source)?,
            ),
        })
    }

    pub(crate) fn key_value_fixed(
        source: &'a MlxKeyValueState,
    ) -> Result<Self, ResidentDecoderPreparationError> {
        Ok(Self {
            storage: match PreparedPagedKvCopy::prepare_live(source)? {
                Some(plan) => PreparedStorage::Paged(plan),
                None => PreparedStorage::KeyValue(PreparedResidentKvCopy::prepare_fixed(source)?),
            },
        })
    }

    pub(crate) fn pooling_fixed(
        source: &'a MlxPoolingAttentionState,
    ) -> Result<Self, ResidentDecoderPreparationError> {
        let storage = if source.prepare_layer_copy_slots()?.is_some() {
            PreparedStorage::Pooling(PreparedResidentPoolingCopy::prepare_fixed(source)?)
        } else {
            PreparedStorage::StatelessPooling(PreparedStatelessPoolingCopy::prepare_fixed(source)?)
        };
        Ok(Self { storage })
    }

    /// Side-effect-free wrapper population from the same Registered/Funded
    /// source branches that the host-copy worker selects, including empty tables.
    pub(crate) fn prepared_source_pin_control_bytes(&self) -> Result<usize, WorkingMemoryError> {
        let registered = match &self.storage {
            PreparedStorage::Paged(plan) => plan.registered_source_tables(),
            PreparedStorage::KeyValue(plan) => plan.registered_source_tables(),
            PreparedStorage::HybridGrouped(plan) => plan.registered_source_tables()?,
            PreparedStorage::Pooling(plan) => plan.registered_source_tables(),
            PreparedStorage::StatelessPooling(_) => 0,
        };
        let prepared = registered
            .checked_add(1)
            .ok_or(WorkingMemoryError::Overflow)?;
        // Operand binding owns no original-source carrier. Every finite table
        // pin and the complete-source pin retain their preparation carrier.
        WorkingMemoryStorage::<StorageIdentity>::copy_source_wrapper_bytes(1, prepared)
    }

    pub(crate) fn host_copy_initialization_peak_bytes(
        &self,
    ) -> Result<u64, ResidentDecoderPreparationError> {
        Ok(match &self.storage {
            PreparedStorage::Paged(plan) => plan.host_copy_initialization_peak_bytes()?,
            PreparedStorage::KeyValue(plan) => plan.host_copy_initialization_peak_bytes()?,
            PreparedStorage::HybridGrouped(plan) => plan.host_copy_initialization_peak_bytes()?,
            PreparedStorage::Pooling(plan) => plan.host_copy_initialization_peak_bytes()?,
            PreparedStorage::StatelessPooling(_) => 0,
        })
    }

    pub(crate) fn host_copy_preparation_bytes(
        &self,
    ) -> Result<usize, eredu_runtime::working_memory::DecoderHostPreparationError> {
        let tables = match &self.storage {
            PreparedStorage::Paged(plan) => plan.host_copy_preparation_bytes()?,
            PreparedStorage::KeyValue(plan) => plan.host_copy_preparation_bytes()?,
            PreparedStorage::HybridGrouped(plan) => plan.host_copy_preparation_bytes()?,
            PreparedStorage::Pooling(plan) => plan.host_copy_preparation_bytes()?,
            PreparedStorage::StatelessPooling(_) => 0,
        };
        let account = match &self.storage {
            PreparedStorage::StatelessPooling(_) => {
                eredu_runtime::working_memory::WorkspaceCopyAccountLayout::sampling()
            }
            PreparedStorage::HybridGrouped(plan) => plan.copy_account_layout(),
            _ => eredu_runtime::working_memory::WorkspaceCopyAccountLayout::decoder_table(),
        }
        .map_err(|cause| match cause {
            WorkingMemoryError::Overflow => {
                eredu_runtime::working_memory::DecoderHostPreparationError::Overflow
            }
            _ => eredu_runtime::working_memory::DecoderHostPreparationError::UnknownBound,
        })?;
        [
            tables,
            account.requested_bytes(),
            std::mem::size_of::<PreparedResidentDecoderHostCopy<'_>>(),
            std::mem::size_of::<Result<PreparedResidentDecoderHostCopy<'_>, Error>>(),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
        .ok_or(eredu_runtime::working_memory::DecoderHostPreparationError::Overflow)
    }

    pub(crate) fn host_copy_prepared(
        &self,
        pool: &WorkingMemoryPool,
        authority: &eredu_core::HostPreparationAuthority,
    ) -> Result<PreparedResidentDecoderHostCopy<'a>, Error> {
        self.host_copy_with_preparation(pool, Some(authority))
    }

    pub(crate) fn host_copy(
        &self,
        pool: &WorkingMemoryPool,
    ) -> Result<PreparedResidentDecoderHostCopy<'a>, Error> {
        self.host_copy_with_preparation(pool, None)
    }

    pub(crate) fn host_copy_with_preparation(
        &self,
        pool: &WorkingMemoryPool,
        preparation: Option<&eredu_core::HostPreparationAuthority>,
    ) -> Result<PreparedResidentDecoderHostCopy<'a>, Error> {
        Ok(PreparedResidentDecoderHostCopy {
            storage: match &self.storage {
                PreparedStorage::Paged(plan) => {
                    HostStorage::Paged(plan.host_copy(pool, preparation.ok_or_else(unknown)?)?)
                }
                PreparedStorage::KeyValue(plan) => HostStorage::KeyValue(
                    plan.host_copy_with_preparation(pool, preparation)
                        .map_err(kv_error)?,
                ),
                PreparedStorage::HybridGrouped(plan) => {
                    HostStorage::HybridGrouped(plan.host_copy_with_preparation(pool, preparation)?)
                }
                PreparedStorage::Pooling(plan) => HostStorage::Pooling(
                    plan.host_copy_with_preparation(pool, preparation)
                        .map_err(pooling_error)?,
                ),
                PreparedStorage::StatelessPooling(plan) => HostStorage::StatelessPooling(*plan),
            },
        })
    }

    pub(crate) fn visit_operands(
        &self,
        visitor: &mut dyn FnMut(&Array),
    ) -> Result<(), SnapshotProjectionCause> {
        match &self.storage {
            PreparedStorage::Paged(plan) => {
                return plan.snapshot_source().visit_arrays(&mut |array| {
                    visitor(array);
                    Ok(())
                });
            }
            PreparedStorage::KeyValue(plan) => plan.visit_operands(visitor),
            PreparedStorage::HybridGrouped(plan) => {
                if plan.is_paged(){return plan.visit_paged_arrays(visitor);}
                plan.visit_operands(visitor)
            },
            PreparedStorage::Pooling(plan) => plan.visit_operands(visitor),
            PreparedStorage::StatelessPooling(_) => {}
        }
        Ok(())
    }

    /// Source custody is distinct from destination operands. KV and resident
    /// Pooling currently copy every held slot; future compact representations
    /// may retain source stores that are not requested destination operands.
    pub(crate) fn visit_retained_arrays(
        &self,
        visitor: &mut dyn FnMut(&Array),
    ) -> Result<(), SnapshotProjectionCause> {
        match &self.storage {
            PreparedStorage::Paged(plan) => {
                return plan.snapshot_source().visit_arrays(&mut |array| {
                    visitor(array);
                    Ok(())
                });
            }
            PreparedStorage::KeyValue(plan) => plan.visit_operands(visitor),
            PreparedStorage::HybridGrouped(plan) => {
                if plan.is_paged(){return plan.visit_paged_arrays(visitor);}
                plan.visit_retained_arrays(visitor)
            },
            PreparedStorage::Pooling(plan) => plan.visit_retained_arrays(visitor),
            PreparedStorage::StatelessPooling(_) => {}
        }
        Ok(())
    }

    pub(crate) fn shared_layout(&self) -> Option<&'a SharedStateLayout> {
        match &self.storage {
            PreparedStorage::Paged(plan) => Some(plan.shared_layout()),
            PreparedStorage::KeyValue(plan) => Some(plan.shared_layout()),
            PreparedStorage::HybridGrouped(plan) => Some(plan.shared_layout()),
            PreparedStorage::Pooling(plan) => Some(plan.shared_layout()),
            PreparedStorage::StatelessPooling(plan) => plan.shared_layout(),
        }
    }

    /// Pooling DeviceState stores only its local layout, not a global index.
    /// Its original executable provenance remains with the enclosing pair.
    pub(crate) fn global_layer_start(&self) -> Option<usize> {
        match &self.storage {
            PreparedStorage::Paged(plan) => Some(plan.global_layer_start()),
            PreparedStorage::KeyValue(plan) => Some(plan.global_layer_start()),
            PreparedStorage::HybridGrouped(plan) => Some(plan.global_layer_start()),
            PreparedStorage::Pooling(_) | PreparedStorage::StatelessPooling(_) => None,
        }
    }

    /// Uses admitted concrete slots and existing numerical recovery. Mismatched
    /// representation rejects before native work; each leaf checks exact source
    /// identity/geometry before filling. No source manager or run grant escapes.
    pub(crate) fn copy_retained(
        self,
        slots: InitializedResidentDecoderCopy<'a>,
        stream: &Stream,
        roots: &RefCell<Vec<Array>>,
    ) -> Result<SavedResidentDecoderCopy, Error> {
        self.copy_retained_observed(slots, stream, roots, &mut |_| Ok(()))
    }
    pub(crate) fn copy_retained_observed(
        self,
        slots: InitializedResidentDecoderCopy<'a>,
        stream: &Stream,
        roots: &RefCell<Vec<Array>>,
        observe: &mut dyn FnMut(&Array) -> Result<(), Error>,
    ) -> Result<SavedResidentDecoderCopy, Error> {
        self.copy_retained_with_host(slots, stream, roots, observe, &mut |_| Ok(()))
    }
    pub(crate) fn copy_retained_with_host(
        self,
        slots: InitializedResidentDecoderCopy<'a>,
        stream: &Stream,
        roots: &RefCell<Vec<Array>>,
        observe: &mut dyn FnMut(&Array) -> Result<(), Error>,
        observe_host: &mut dyn FnMut(
            &crate::backend::array_copy::PreparedSavedHostCopy,
        ) -> Result<(), Error>,
    ) -> Result<SavedResidentDecoderCopy, Error> {
        let storage = match (self.storage, slots.storage) {
            (PreparedStorage::Paged(plan), InitializedStorage::Paged(slots)) => {
                SavedStorage::Paged(plan.copy_retained_with_host(
                    slots,
                    stream,
                    roots,
                    observe,
                    observe_host,
                )?)
            }
            (PreparedStorage::KeyValue(plan), InitializedStorage::KeyValue(slots)) => {
                SavedStorage::KeyValue(plan.copy_retained(slots, stream, roots).map_err(kv_error)?)
            }
            (PreparedStorage::HybridGrouped(plan), InitializedStorage::HybridGrouped(slots)) => {
                SavedStorage::HybridGrouped(plan.copy_retained_with(slots, stream, roots, observe, observe_host)?)
            }
            (PreparedStorage::Pooling(plan), InitializedStorage::Pooling(slots)) => {
                SavedStorage::Pooling(
                    plan.copy_retained(slots, stream, roots)
                        .map_err(pooling_error)?,
                )
            }
            (
                PreparedStorage::StatelessPooling(plan),
                InitializedStorage::StatelessPooling(source),
            ) => SavedStorage::StatelessPooling(plan.copy(source)?),
            _ => return Err(mismatch()),
        };
        Ok(SavedResidentDecoderCopy { storage })
    }
}

impl fmt::Debug for PreparedResidentDecoderCopy<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreparedResidentDecoderCopy")
            .field("global_layer_start", &self.global_layer_start())
            .finish_non_exhaustive()
    }
}

/// Only the concrete slot type varies here. All accounting, source-origin
/// validation and host initialization stay in the one neutral runtime path.
pub(crate) struct PreparedResidentDecoderHostCopy<'a> {
    storage: HostStorage<'a>,
}
enum HostStorage<'a> {
    Paged(PreparedPagedKvHostCopy<'a>),
    KeyValue(RegisteredDecoderHostCopy<'a, MlxKeyValueLayerState, StorageIdentity>),
    HybridGrouped(PreparedHybridGroupHostCopy<'a>),
    Pooling(RegisteredDecoderHostCopy<'a, MlxPoolingAttentionCache, StorageIdentity>),
    StatelessPooling(PreparedStatelessPoolingCopy<'a>),
}
impl<'a> PreparedResidentDecoderHostCopy<'a> {
    pub(crate) fn initialization_peak_bytes(&self) -> u64 {
        match &self.storage {
            HostStorage::Paged(plan) => plan.initialization_peak_bytes(),
            HostStorage::KeyValue(plan) => plan.initialization_peak_bytes(),
            HostStorage::HybridGrouped(plan) => plan.initialization_peak_bytes(),
            HostStorage::Pooling(plan) => plan.initialization_peak_bytes(),
            HostStorage::StatelessPooling(_) => 0,
        }
    }

    pub(crate) fn admit(
        self,
        pool: &WorkingMemoryPool,
        sampling: RegisteredSamplingCopy<'a, StorageIdentity>,
        complete_source: WorkingMemoryStorage<StorageIdentity>,
        limits: WorkspaceCopyLimits,
    ) -> Result<
        (
            FundedSamplerCopy,
            InitializedResidentDecoderCopy<'a>,
            AdmittedWorkspaceCopy,
        ),
        Error,
    > {
        let (sampler, storage, native) = match self.storage {
            HostStorage::Paged(plan) => {
                let (sampler, slots, native) =
                    plan.admit(pool, sampling, complete_source, limits)?;
                (sampler, InitializedStorage::Paged(slots), native)
            }
            HostStorage::KeyValue(plan) => {
                let joined = sampling
                    .with_decoder_slots(plan, complete_source)
                    .map_err(|error| Error::Other(Box::new(error)))?;
                let (sampler, slots, native) = pool
                    .copy_text_components(joined, limits)
                    .map_err(|error| Error::Other(Box::new(error)))?;
                (sampler, InitializedStorage::KeyValue(slots), native)
            }
            HostStorage::HybridGrouped(plan) => {
                let (sampler, slots, native) =
                    plan.admit(pool, sampling, complete_source, limits)?;
                (sampler, InitializedStorage::HybridGrouped(slots), native)
            }
            HostStorage::Pooling(plan) => {
                let joined = sampling
                    .with_decoder_slots(plan, complete_source)
                    .map_err(|error| Error::Other(Box::new(error)))?;
                let (sampler, slots, native) = pool
                    .copy_text_components(joined, limits)
                    .map_err(|error| Error::Other(Box::new(error)))?;
                (sampler, InitializedStorage::Pooling(slots), native)
            }
            HostStorage::StatelessPooling(source) => {
                let joined = sampling.with_complete_source(complete_source);
                let (sampler, native) = pool
                    .copy_sampling_components_with_source(joined, limits)
                    .map_err(|error| Error::Other(Box::new(error)))?;
                (
                    sampler,
                    InitializedStorage::StatelessPooling(source),
                    native,
                )
            }
        };
        Ok((sampler, InitializedResidentDecoderCopy { storage }, native))
    }
}

/// Move-only typed destination. Neither native code nor callers can fabricate
/// these variants or unwrap an admitted table into mutable runnable state.
pub(crate) struct InitializedResidentDecoderCopy<'a> {
    storage: InitializedStorage<'a>,
}
impl InitializedResidentDecoderCopy<'_> {
    pub(crate) fn prepare_host_destinations(
        &mut self,
        copy: &mut crate::backend::array_copy::PreparedOriginalCopy,
        environment: &crate::backend::OriginalCopyEnvironment<'_>,
    ) -> Result<(), Error> {
        match &mut self.storage {
            InitializedStorage::Paged(slots) => slots.prepare_host_destinations(copy, environment),
            InitializedStorage::HybridGrouped(slots) => slots.prepare_host_destinations(copy, environment),
            _ => Ok(()),
        }
    }
}
enum InitializedStorage<'a> {
    Paged(InitializedPagedKvCopy),
    KeyValue(InitializedDecoderSlots<MlxKeyValueLayerState>),
    HybridGrouped(InitializedHybridGroupCopy),
    Pooling(InitializedDecoderSlots<MlxPoolingAttentionCache>),
    StatelessPooling(PreparedStatelessPoolingCopy<'a>),
}

/// Actual frozen values and their slot custody. No mutable state, old inference
/// grant, allocating Clone or owning native-array extraction is exposed.
pub(crate) struct SavedResidentDecoderCopy {
    storage: SavedStorage,
}
enum SavedStorage {
    Paged(SavedPagedKvCopy),
    KeyValue(SavedResidentKvCopy),
    HybridGrouped(SavedHybridGroupedCopy),
    Pooling(SavedResidentPoolingCopy),
    StatelessPooling(SavedStatelessPoolingCopy),
}

impl SavedResidentDecoderCopy {
    pub(crate) fn visit_registered_child_metadata(
        &self,
        visitor: &mut dyn FnMut(&eredu_runtime::HostSlotMetadata) -> Result<(), Error>,
    ) -> Result<(), Error> {
        self.prepare_copy()?
            .visit_registered_child_metadata(visitor)
    }

    pub(crate) fn prepare_copy(&self) -> Result<PreparedResidentDecoderCopy<'_>, Error> {
        Ok(PreparedResidentDecoderCopy {
            storage: match &self.storage {
                SavedStorage::Paged(saved) => PreparedStorage::Paged(
                    saved
                        .prepare_copy()
                        .map_err(|cause| Error::Other(Box::new(cause)))?,
                ),
                SavedStorage::KeyValue(saved) => {
                    PreparedStorage::KeyValue(saved.prepare_copy().map_err(kv_error)?)
                }
                SavedStorage::HybridGrouped(saved) => {
                    PreparedStorage::HybridGrouped(saved.prepare_copy()?)
                }
                SavedStorage::Pooling(saved) => {
                    PreparedStorage::Pooling(saved.prepare_copy().map_err(pooling_error)?)
                }
                SavedStorage::StatelessPooling(saved) => {
                    PreparedStorage::StatelessPooling(saved.prepare_copy())
                }
            },
        })
    }
    pub(crate) fn prepare_copy_fixed(
        &self,
    ) -> Result<PreparedResidentDecoderCopy<'_>, ResidentDecoderPreparationError> {
        Ok(PreparedResidentDecoderCopy {
            storage: match &self.storage {
                SavedStorage::Paged(saved) => PreparedStorage::Paged(saved.prepare_copy()?),
                SavedStorage::KeyValue(saved) => {
                    PreparedStorage::KeyValue(PreparedResidentKvCopy::prepare_saved_fixed(saved)?)
                }
                SavedStorage::HybridGrouped(saved) => {
                    PreparedStorage::HybridGrouped(saved.prepare_copy_fixed()?)
                }
                SavedStorage::Pooling(saved) => PreparedStorage::Pooling(
                    PreparedResidentPoolingCopy::prepare_saved_fixed(saved)?,
                ),
                SavedStorage::StatelessPooling(saved) => {
                    PreparedStorage::StatelessPooling(saved.prepare_copy())
                }
            },
        })
    }
    pub(crate) fn shared_layout(&self) -> Option<&SharedStateLayout> {
        match &self.storage {
            SavedStorage::Paged(saved) => Some(saved.shared_layout()),
            SavedStorage::KeyValue(saved) => Some(saved.shared_layout()),
            SavedStorage::HybridGrouped(saved) => Some(saved.shared_layout()),
            SavedStorage::Pooling(saved) => Some(saved.shared_layout()),
            SavedStorage::StatelessPooling(saved) => saved.shared_layout(),
        }
    }
    pub(crate) fn global_layer_start(&self) -> Option<usize> {
        match &self.storage {
            SavedStorage::Paged(saved) => Some(saved.global_layer_start()),
            SavedStorage::KeyValue(saved) => Some(saved.global_layer_start()),
            SavedStorage::HybridGrouped(saved) => Some(saved.global_layer_start()),
            SavedStorage::Pooling(_) | SavedStorage::StatelessPooling(_) => None,
        }
    }
    pub(crate) fn retained_slot_bytes(&self) -> u64 {
        match &self.storage {
            SavedStorage::Paged(saved) => saved.retained_slot_bytes(),
            SavedStorage::KeyValue(saved) => saved.retained_slot_bytes(),
            SavedStorage::HybridGrouped(saved) => saved.retained_slot_bytes(),
            SavedStorage::Pooling(saved) => saved.retained_slot_bytes(),
            SavedStorage::StatelessPooling(_) => 0,
        }
    }
    pub(crate) fn protected_slot_bytes(&self) -> u64 {
        match &self.storage {
            SavedStorage::Paged(saved) => saved.protected_slot_bytes(),
            SavedStorage::KeyValue(saved) => saved.protected_slot_bytes(),
            SavedStorage::HybridGrouped(saved) => saved.protected_slot_bytes(),
            SavedStorage::Pooling(saved) => saved.protected_slot_bytes(),
            SavedStorage::StatelessPooling(_) => 0,
        }
    }
}
impl fmt::Debug for SavedResidentDecoderCopy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SavedResidentDecoderCopy")
            .field("global_layer_start", &self.global_layer_start())
            .field("retained_slot_bytes", &self.retained_slot_bytes())
            .finish_non_exhaustive()
    }
}

pub(super) mod host_copy;

pub(crate) use host_copy::OriginalResidentState;

mod source_projection;
pub(crate) use source_projection::{
    SnapshotArraySources, SnapshotOperand, SnapshotProjectionCause, SnapshotProjectionPlan,
};

mod original;
pub(crate) use original::{
    CompletedResidentSource, OriginalResidentSourceBinding, bind_completed_resident_source_priors,
    bind_completed_resident_sources, bind_unselected_completed_resident_source_priors,
    copy as copy_original_resident_state, copy_with_source as copy_completed_resident_state,
};
