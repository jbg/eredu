//! MLX materialization and transfer execution for immutable weight units.
//!
//! A [`crate::backend::runtime::residency::manager::ResidencyManager`] moves caller-defined groups of
//! checkpoint selections from an [`eredu_checkpoint::store::CheckpointSource`] into
//! immutable typed host-transfer buffers or execution-stream arrays. The
//! manager accounts for logical host and device copies independently, even on
//! unified-memory systems.
//! Missing units can be reserved and submitted as one batch. Caller-owned
//! [`crate::backend::runtime::residency::manager::ResidentTransfer`] values retain
//! source leases until MLX reports exact completion of the submitted
//! transfer.

use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{Arc, Condvar, Mutex, MutexGuard},
    time::Instant,
};

use safemlx::{
    Array, DeviceType, Event, HostTransferBuffer, HostTransferPolicy, ImmutableHostTransferBuffer,
    Stream, host_transfer_capacity_upper_bound, transforms::async_eval_with_event,
};

use crate::{
    backend::nn::shared::MlxNeuralBackend,
    backend::residency::sample_allocator_memory,
    backend::runtime::checkpoint::recipe::WeightRecipeError,
    backend::runtime::checkpoint::store::{
        CheckpointMaterializationError, MlxParameterMaterializationContext,
        PendingWeightMaterialization, WeightMaterialization,
    },
};
use eredu_core::residency::{
    EvictedResidencyCopy, MemoryTier, OffloadPlan, OffloadReport, OffloadUnitId, PrefetchOutcome,
    ResidencyLedgerError, TransferDirection, UnitResidencyReport,
};

use eredu_runtime::ResidencyReport;
use eredu_runtime::residency::{
    OffloadUnit, ResidencyController, ResidencyControllerError, ResidencyLease,
    ResidencyLeaseOwner, ResidencyLeaseStorage, ResidencyWindowError, ResidencyWindowManager,
    WeightBinding,
};
mod background;
pub(crate) use background::{
    BackgroundHostReadFailure, BackgroundHostReadOwner, BackgroundSourceAttempt,
    PreparedBackgroundHostReads, PreparedBackgroundHostWindow, PreparedHostProtection,
    PreparedHostPublication,
};
mod eviction;
mod host_acquisition;
pub use host_acquisition::HostAcquisitionFailure;
pub(crate) use host_acquisition::OrdinaryMaterializedRequirements;
mod parameter_source;
pub(crate) use parameter_source::{ResidentParameterSource, ResidentParameterSourceError};
mod construction;
pub use construction::OriginalManagerError;
pub(crate) use construction::{
    ForegroundDiskDescriptors, ForegroundDiskReadError, ForegroundDiskReadLayout,
    ForegroundDiskReadPlan, ForegroundDiskSourceError, ForegroundMaterializationPopulation,
    OriginalManagerPlan, PreparedForegroundDiskIo, PreparedForegroundDiskRead,
    ReadForegroundDiskBatch, prepare_foreground_disk_descriptors,
};
mod owner;
mod rows;
pub(crate) use owner::ManagerCustody;
use owner::ManagerOwner;
pub use owner::{ManagerWeak, ResidentHostOwner, RetainedHostBuffer};

/// A resident unit that prevents eviction of one tier until it is dropped.
pub type ResidentUnitLease = ResidencyLease<ResidentLeaseStorage, ManagerInner, ManagerWeak>;

/// Host or device storage retained by a weight-residency lease.
pub enum ResidentLeaseStorage {
    /// Immutable host-transfer buffers.
    Host(ResidentHostOwner),
    /// Materialized device arrays.
    Device(ResidentArraysOwner),
}

/// Prospective controls for binding completed source identity to real cached
/// cells. The source window bounds both calls and cells before native execution.
pub(crate) struct PreparedCompletedSourceRetention {
    leases: usize,
    cells: usize,
}
impl PreparedCompletedSourceRetention {
    pub(crate) fn prepare(
        leases: usize,
        cells: usize,
        context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<Self, crate::backend::Error> {
        use crate::backend::submission_recovery::native_role::physical::CompletedNumericalSource;
        fn iterator_bytes<T>(_: impl FnOnce(&'static NamedArrays) -> T) -> usize {
            std::mem::size_of::<T>()
        }
        let per_cell = safemlx::SharedOriginalBufferInspection::inspection_control_bytes()
            .and_then(|n| {
                n.checked_add(std::mem::size_of::<(
                    CompletedNumericalSource,
                    Result<(), CompletedNumericalSource>,
                )>())
            });
        let fixed = std::mem::size_of::<(
            Self,
            &ResidentLeaseStorage,
            &CompletedNumericalSource,
            &eredu_nn::workspace::WorkspaceContext,
            super::storage::RetainedStorageRef<'_>,
            Result<(), safemlx::OriginalBufferCause>,
        )>();
        let bytes = per_cell
            .and_then(|n| n.checked_mul(cells))
            .and_then(|n| {
                fixed
                    .checked_add(iterator_bytes(NamedArrays::retained_values))?
                    .checked_mul(leases.checked_add(1)?)
                    .and_then(|fixed| n.checked_add(fixed))
            })
            .ok_or_else(|| {
                crate::backend::Error::Neural(
                    eredu_nn::workspace::WorkspaceMetadataError::Overflow.into(),
                )
            })?;
        context
            .charge_metadata(bytes)
            .map_err(|cause| crate::backend::Error::Neural(cause.into()))?;
        Ok(Self { leases, cells })
    }
    pub(crate) fn retain(
        &mut self,
        storage: &ResidentLeaseStorage,
        source: &crate::backend::submission_recovery::native_role::physical::CompletedNumericalSource,
        context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<(), crate::backend::Error> {
        let failure = || {
            crate::backend::Error::PrefillControl(
                eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
            )
        };
        self.leases = self.leases.checked_sub(1).ok_or_else(failure)?;
        let ResidentLeaseStorage::Device(owner) = storage else {
            return Ok(());
        };
        for value in owner.arrays.retained_values() {
            if let super::storage::RetainedStorageRef::CanonicalArray(cell) = value {
                self.cells = self.cells.checked_sub(1).ok_or_else(failure)?;
                cell.retain_completed_numerical_source(source)
                    .map_err(|cause| {
                        crate::backend::Error::Neural(context.metadata_source(cause))
                    })?;
            }
        }
        Ok(())
    }
}

impl ResidentLeaseStorage {
    pub(crate) fn completed_numerical_sources(
        &self,
    ) -> impl Iterator<Item = &crate::backend::submission_recovery::native_role::physical::CompletedNumericalSource>{
        let arrays = match self {
            Self::Device(owner) => Some(owner.arrays.retained_values()),
            Self::Host(_) => None,
        };
        arrays
            .into_iter()
            .flatten()
            .filter_map(|value| match value {
                super::storage::RetainedStorageRef::CanonicalArray(cell) => {
                    cell.completed_numerical_source()
                }
                _ => None,
            })
    }
    pub(crate) fn completed_numerical_source_control_bytes() -> Option<usize> {
        fn iterator_bytes<T>(_: impl FnOnce(&'static ResidentLeaseStorage) -> T) -> usize {
            std::mem::size_of::<T>()
        }
        iterator_bytes(Self::completed_numerical_sources).checked_add(std::mem::size_of::<(
            &Self,
            crate::backend::submission_recovery::native_role::physical::CompletedNumericalSource,
        )>())
    }
    /// Borrow only completed receipts carried by this exact acquired storage.
    pub(crate) fn host_receipts(
        &self,
    ) -> impl Iterator<Item = super::storage::RetainedAllocationReceipt<'_>> {
        let host = match self {
            Self::Host(owner) => Some(owner.buffers.values()),
            _ => None,
        };
        let device = match self {
            Self::Device(owner) => Some(owner.arrays.host_sources()),
            _ => None,
        };
        host.into_iter()
            .flatten()
            .chain(device.into_iter().flatten())
            .filter_map(RetainedHostBuffer::attachment_receipt)
    }
    pub(crate) fn host_receipt_control_bytes() -> Option<usize> {
        fn iterator_bytes<T>(_: impl FnOnce(&'static ResidentLeaseStorage) -> T) -> usize {
            std::mem::size_of::<T>()
        }
        iterator_bytes(Self::host_receipts)
            .checked_add(std::mem::size_of::<&Self>())?
            .checked_add(RetainedHostBuffer::attachment_receipt_control_bytes()?)
    }
}

/// Borrowed binding names from the actual host map or device destination.
/// The iterator itself uses no separate heap allocation.
pub struct ResidentBindingNames<'a> {
    inner: ResidentBindingNamesInner<'a>,
}
enum ResidentBindingNamesInner<'a> {
    Host(rows::Keys<'a, String, RetainedHostBuffer>),
    Device(named_arrays::NamedIter<'a>),
}
impl<'a> Iterator for ResidentBindingNames<'a> {
    type Item = &'a str;
    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.inner {
            ResidentBindingNamesInner::Host(names) => names.next().map(String::as_str),
            ResidentBindingNamesInner::Device(values) => values.next().map(|(name, _)| name),
        }
    }
}

impl ResidencyLeaseStorage for ResidentLeaseStorage {
    type DeviceValue = Array;
    type HostValue = ImmutableHostTransferBuffer;
    type Error = ResidencyError;
    type BindingNames<'a> = ResidentBindingNames<'a>;

    fn device_value<'a>(
        &'a self,
        id: &OffloadUnitId,
        name: &str,
    ) -> Result<&'a Self::DeviceValue, Self::Error> {
        match self {
            ResidentLeaseStorage::Device(arrays) => {
                arrays
                    .arrays
                    .get(name)
                    .ok_or_else(|| ResidencyError::UnknownBinding {
                        id: id.clone(),
                        name: name.to_string(),
                    })
            }
            ResidentLeaseStorage::Host(_) => Err(ResidencyError::HostBindingIsNotArray {
                id: id.clone(),
                name: name.to_string(),
            }),
        }
    }

    fn host_value<'a>(
        &'a self,
        id: &OffloadUnitId,
        name: &str,
    ) -> Result<&'a Self::HostValue, Self::Error> {
        match self {
            ResidentLeaseStorage::Host(buffers) => {
                buffers.buffers.get(name).map(AsRef::as_ref).ok_or_else(|| {
                    ResidencyError::UnknownBinding {
                        id: id.clone(),
                        name: name.to_string(),
                    }
                })
            }
            ResidentLeaseStorage::Device(_) => Err(ResidencyError::DeviceBindingIsNotHostBuffer {
                id: id.clone(),
                name: name.to_string(),
            }),
        }
    }

    fn binding_names(&self) -> Self::BindingNames<'_> {
        ResidentBindingNames {
            inner: match self {
                ResidentLeaseStorage::Host(buffers) => {
                    ResidentBindingNamesInner::Host(buffers.buffers.keys())
                }
                ResidentLeaseStorage::Device(arrays) => {
                    ResidentBindingNamesInner::Device(arrays.arrays.iter())
                }
            },
        }
    }
}

/// Structured failures from residency validation and state transitions.
#[derive(Debug, thiserror::Error)]
pub enum ResidencyError {
    /// The ordinary request refused acquisition-control storage before mutation.
    #[error("ordinary residency host funding: {0}")]
    HostMetadataFunding(#[source] eredu_nn::workspace::HostMetadataFundingError),
    /// The actual ordinary host constructor has no qualified control layout.
    #[error("ordinary host transfer control source is unavailable")]
    OrdinaryHostControlSource,
    /// The paid ordinary Host destination could not allocate its named rows.
    #[error("ordinary Host destination allocation: {0}")]
    HostDestinationReserve(#[source] std::collections::TryReserveError),
    /// Ordinary acquisition preserves its exact host payer through the cause.
    #[error(transparent)]
    HostAcquisition(#[from] HostAcquisitionFailure),
    /// Allocation-free native metadata refusal during a cold source census.
    #[error("cold retained array inspection: {0}")]
    OriginalArrayInspection(#[source] safemlx::ArrayMetadataError),
    /// Allocation-free immutable host metadata refusal during a cold census.
    #[error("cold retained host inspection: {0}")]
    OriginalHostInspection(#[source] safemlx::HostTransferMetadataError),
    #[error("original retained inventory: {0}")]
    /// Original retained-storage collection failed its admitted contract.
    OriginalInventory(#[source] eredu_runtime::working_memory::WorkingMemoryError),
    /// Immutable original source construction or finite native alias refusal.
    #[error("original immutable host source: {0}")]
    OriginalHostInput(#[source] safemlx::PreparedInputCause),
    #[error("prepared retained descriptor: {0}")]
    /// A prepared native handle could not retain the inspected array.
    OriginalClone(#[source] safemlx::PreparedArrayCloneCause),

    /// The retained canonical owner closure or caller destination was invalid.
    #[error("invalid residency operation closure: {0:?}")]
    OperationClosure(eredu_runtime::residency::ResidencyClosureError),
    /// The manager control state is held by another operation; no work started.
    #[error("original residency manager is busy")]
    OriginalManagerBusy,
    /// An earlier transfer must settle at its actual owner before acquisition.
    #[error("original residency acquisition has an unresolved predecessor transfer")]
    OriginalPendingTransfer,
    /// A source-bound final named destination rejected activation or publication.
    #[error(transparent)]
    OriginalNamedDestination(#[from] NamedArrayError),

    /// Prepared neutral admission failed; exact source and final destinations
    /// remain owned until this error is destroyed under its original custody.
    #[error(transparent)]
    OriginalAdmission(#[from] PreparedAdmissionFailure),

    /// Fixed neutral explicit-eviction refusal; the consuming bank retains the
    /// exact source ID and original operation custody with this error.
    #[error("original residency eviction from {tier:?}: {cause}")]
    OriginalEviction {
        /// Same neutral policy/pin/accounting refusal as ordinary eviction.
        #[source]
        cause: eredu_core::residency::ResidencyEvictionError,
        /// Actual requested materialized tier.
        tier: MemoryTier,
    },

    /// Fixed original native observation/submission transport. This preserves
    /// the actual native cause without allocating a residency ID or message.
    #[error("original residency operation: {0}")]
    OriginalNative(#[source] safemlx::error::Exception),
    /// Exact post-payload registered-role retirement refusal.
    #[error("original residency retirement: {0}")]
    OriginalRetirement(
        #[from] crate::backend::runtime::execution::generic::RegisteredScopeRetirementCause,
    ),
    /// Fixed cache-origin refusal. It does not certify streams, sources or rows.
    #[error("original converted-cache owner: {0}")]
    OriginalCache(#[source] eredu_runtime::working_memory::WorkingMemoryError),
    /// Exact prepaid materialization stream wrapper refusal.
    #[error("original materialization stream: {0}")]
    OriginalStreams(
        #[source] crate::backend::runtime::checkpoint::store::PreparedMaterializationStreamError,
    ),
    /// The explicit original-operation role does not match its retained owner.
    #[error("original residency operation domain mismatch")]
    OriginalOperationDomain,
    /// A consuming original finish retained its node in recovery. The caller
    /// cannot query the now-absent local handle as if retirement had succeeded.
    #[error("original residency operation retention transferred to recovery")]
    OriginalOperationRetirementTransferred,
    /// An exactly prepared operation family has no unused slot left.
    #[error("original {family} operation storage exhausted after {prepared} slots")]
    OriginalOperationCapacity {
        /// Concrete prepriced operation family.
        family: &'static str,
        /// Number of original slots prepared for that family.
        prepared: usize,
    },

    /// An active admitted disk operation rejected a route or backing change.
    #[error("admitted direct disk route rejected: {source}")]
    AdmittedDiskRoute {
        /// Original typed source, including memory and checkpoint failures.
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    /// A complete owner binding set is unsupported by the MLX parameter backend.
    #[error("MLX residency binding preflight failed: {0}")]
    BindingPreflight(String),
    /// A backend-neutral binding or offload-unit declaration was invalid.
    #[error(transparent)]
    Declaration(#[from] eredu_runtime::residency::ResidencyDeclarationError),
    /// Backend-neutral ownership or capacity transition failed.
    #[error(transparent)]
    Ledger(#[from] ResidencyLedgerError),
    /// Backend-neutral plan and declaration validation failed.
    #[error(transparent)]
    Controller(#[from] ResidencyControllerError),
    /// Backend-neutral binding selection rewrite failed.
    #[error(transparent)]
    BindingSelection(#[from] eredu_runtime::residency::WeightBindingSelectionError),
    /// Backend-neutral ordered-window validation or accounting failed.
    #[error(transparent)]
    Window(#[from] ResidencyWindowError),
    /// Binding sizes did not sum to the plan's unit size.
    #[error(
        "residency unit {id} defines {actual_bytes} bytes but its plan reserves {planned_bytes}"
    )]
    UnitByteMismatch {
        /// Unit identifier.
        id: OffloadUnitId,
        /// Bytes reserved by the plan.
        planned_bytes: u64,
        /// Sum of binding sizes.
        actual_bytes: u64,
    },
    /// A binding's selected checkpoint size contradicted its definition.
    #[error(
        "binding {binding:?} in unit {id} selects {actual_bytes} bytes but declares {expected_bytes}"
    )]
    BindingByteMismatch {
        /// Unit identifier.
        id: OffloadUnitId,
        /// Binding name.
        binding: String,
        /// Declared size.
        expected_bytes: u64,
        /// Store-validated size.
        actual_bytes: u64,
    },
    /// Exact source catalog or retained read authentication failed.
    #[error("original source recipe: {0}")]
    OriginalSourceRecipe(#[from] WeightRecipeError),
    /// A read through retained disk descriptors failed during ordinary execution.
    /// The error preserves its detached source owner until the cause is released.
    #[error("detached disk read: {0}")]
    DetachedDiskRead(#[source] eredu_core::BackendFailure),
    /// A transformed disk source requires its exact active ordinary execution owner.
    #[error("ordinary materialized disk source has no matching execution owner")]
    OrdinaryMaterializationSource,
    /// The shared prepared-leaf equation retains its original failure custody.
    #[error("materialized disk read: {0}")]
    MaterializedDiskRead(#[source] eredu_core::BackendFailure),
    /// Exact foreground read/copy refusal, retaining its admitted source prefix.
    #[error("original foreground disk materialization: {0}")]
    OriginalForegroundDiskMaterialization(
        #[from] materialization::ForegroundDiskMaterializationError,
    ),
    /// Exact failed host publication and its source-owned preparation prefix.
    #[error("{0}")]
    OriginalHostPublication(#[source] Box<background::BackgroundHostPublicationFailure>),
    /// Failed exact background window, retaining its source/control owner.
    #[error("{0}")]
    OriginalHostWindow(#[source] Box<background::BackgroundHostWindowFailure>),
    /// A source-bound background forward failed before its successful join.
    #[error("{0}")]
    OriginalBackgroundCoordinator(#[source] Box<super::dense_stream::BackgroundCoordinatorFailure>),
    /// A derived-weight recipe was invalid or could not be materialized.
    #[error("derived-weight recipe for binding {binding:?} failed: {source}")]
    Recipe {
        /// Local binding name.
        binding: String,
        /// Recipe failure.
        #[source]
        source: WeightRecipeError,
    },
    /// The configured source stream was not a CPU stream.
    #[error("the residency source stream must target the CPU")]
    InvalidSourceStream,
    /// A binding lookup failed on a valid resident unit.
    #[error("residency unit {id} has no binding named {name:?}")]
    UnknownBinding {
        /// Unit identifier.
        id: OffloadUnitId,
        /// Unknown local name.
        name: String,
    },
    /// A caller requested an executable array from typed host storage.
    #[error(
        "host-resident binding {name:?} in unit {id} is a typed transfer buffer, not an MLX array"
    )]
    HostBindingIsNotArray {
        /// Unit identifier.
        id: OffloadUnitId,
        /// Requested binding name.
        name: String,
    },
    /// A caller requested typed host storage from a device-resident copy.
    #[error(
        "device-resident binding {name:?} in unit {id} is an MLX array, not a host-transfer buffer"
    )]
    DeviceBindingIsNotHostBuffer {
        /// Unit identifier.
        id: OffloadUnitId,
        /// Requested binding name.
        name: String,
    },
    /// A backend allocated beyond its advertised pre-allocation capacity bound.
    #[error(
        "host-transfer allocation for residency unit {id} used {actual_bytes} bytes, exceeding reserved upper bound {reserved_bytes}"
    )]
    HostCapacityBoundExceeded {
        /// Unit identifier.
        id: OffloadUnitId,
        /// Capacity reserved before materialization.
        reserved_bytes: u64,
        /// Exact allocated capacity.
        actual_bytes: u64,
    },
    /// Checked byte or recency arithmetic overflowed.
    #[error("residency arithmetic overflow: {context}")]
    ArithmeticOverflow {
        /// Calculation that overflowed.
        context: &'static str,
    },
    /// Backend-neutral checkpoint inspection or lease acquisition failed.
    #[error(transparent)]
    CheckpointStore(#[from] eredu_checkpoint::store::StoreError),
    /// MLX checkpoint materialization failed.
    #[error(transparent)]
    CheckpointMaterialization(#[from] CheckpointMaterializationError),
    /// An MLX copy or evaluation failed.
    #[error("MLX {operation} failed for residency unit {id}: {source}")]
    Mlx {
        /// Unit identifier.
        id: OffloadUnitId,
        /// Failed operation.
        operation: &'static str,
        /// MLX exception.
        #[source]
        source: safemlx::error::Exception,
    },
    /// Serialized manager state was poisoned by a prior panic.
    #[error("residency manager state is poisoned")]
    StatePoisoned,
}

/// Serialized, shareable manager for immutable checkpoint weight residency.
#[derive(Clone)]
pub struct ResidencyManager {
    inner: ManagerOwner,
}

/// Immutable source identity is retained per physical unit. Transformed modules
/// may use the same source key with different geometry or values.
pub(super) enum ResidencySources {
    Ordinary {
        primary: eredu_checkpoint::store::RetainedCheckpointSource,
        units: rows::Rows<OffloadUnitId, eredu_checkpoint::store::RetainedCheckpointSource>,
    },
    Original(construction::OriginalHostSources),
}
impl ResidencySources {
    fn foreground(&self) -> Option<&ForegroundDiskDescriptors> {
        match self {
            Self::Original(source) => source.foreground(),
            Self::Ordinary { .. } => None,
        }
    }
    fn source(&self, unit: &OffloadUnitId) -> &dyn eredu_checkpoint::store::CheckpointSource {
        match self {
            Self::Ordinary { primary, units } => units.get(unit).unwrap_or(primary).as_ref(),
            Self::Original(source) => source.catalog(unit),
        }
    }
    fn retained(
        &self,
        unit: &OffloadUnitId,
    ) -> Option<&eredu_checkpoint::store::RetainedCheckpointSource> {
        match self {
            Self::Ordinary { primary, units } => Some(units.get(unit).unwrap_or(primary)),
            Self::Original(_) => None,
        }
    }
    fn ordinary_sources(
        &self,
    ) -> impl Iterator<Item = &eredu_checkpoint::store::RetainedCheckpointSource> {
        let (primary, units) = match self {
            Self::Ordinary { primary, units } => (Some(primary), Some(units)),
            Self::Original(_) => (None, None),
        };
        primary
            .into_iter()
            .chain(units.into_iter().flat_map(|units| units.values()))
    }
    fn prepared_host(&self, unit: &OffloadUnitId) -> Option<&ResidentHostOwner> {
        match self {
            Self::Original(source) => source.host(unit),
            Self::Ordinary { .. } => None,
        }
    }
}

fn preflight_residency_owner_bindings(
    sources: &ResidencySources,
    control: &ResidencyController,
) -> Result<(), ResidencyError> {
    for unit in control.units() {
        // The neutral controller already validated aliases across the complete
        // unit graph. Only physical owners select recipes against this unit's
        // exact source; a cross-unit alias is not a second source binding.
        let owners = unit
            .bindings()
            .iter()
            .filter(|binding| !binding.is_alias())
            .cloned()
            .collect::<Vec<_>>();
        eredu_runtime::preflight_bindings::<MlxNeuralBackend>(sources.source(unit.id()), &owners)
            .map_err(|error| ResidencyError::BindingPreflight(error.to_string()))?;
    }
    Ok(())
}

impl ResidencyManager {
    #[cfg(test)]
    pub(crate) fn test_evict_completed_parameter_alias(
        &self,
    ) -> Result<Option<CanonicalArrayOwner>, ResidencyError> {
        let selected = {
            let state = self.inner.state.lock().unwrap();
            state.storage.iter().find_map(|(id, unit)| {
                unit.device
                    .as_ref()?
                    .arrays
                    .retained_values()
                    .find_map(|value| match value {
                        super::storage::RetainedStorageRef::CanonicalArray(cell)
                            if cell.completed_numerical_source().is_some()
                                || cell.test_host_source().is_some_and(|host| {
                                    host.attachment_receipt().is_some()
                                }) =>
                        {
                            Some((id.clone(), cell.clone()))
                        }
                        _ => None,
                    })
            })
        };
        let Some((id, alias)) = selected else {
            return Ok(None);
        };
        assert!(self.evict(&id, MemoryTier::Device)?);
        Ok(Some(alias))
    }
    pub(crate) fn original_host_recipe(
        &self,
        unit: &OffloadUnitId,
        binding: &str,
    ) -> Option<&eredu_checkpoint::recipe::RecipeMetadata> {
        match &self.inner.sources {
            ResidencySources::Original(source) => source.output(unit, binding),
            _ => None,
        }
    }
    pub(crate) fn original_checkpoint_source(
        &self,
    ) -> Option<&eredu_checkpoint::store::RetainedCheckpointSource> {
        match &self.inner.sources {
            ResidencySources::Original(source) => Some(source.policy()),
            _ => None,
        }
    }
    /// Exact source-only disk owner. Request read slots and native operation
    /// admission must still be supplied before materializing any disk unit.
    pub(crate) fn original_foreground_disk_descriptors(
        &self,
    ) -> Option<&construction::ForegroundDiskDescriptors> {
        match &self.inner.sources {
            ResidencySources::Original(source) => source.foreground(),
            _ => None,
        }
    }
    pub(crate) fn original_dense_controller(
        &self,
    ) -> Option<crate::backend::runtime::execution::layerwise::PreparedDenseController> {
        self.inner.dense_controller.get().cloned()
    }
    pub(crate) fn original_dense_controller_control_bytes(&self) -> Option<usize> {
        self.inner.dense_controller.get()?;
        let controls = [
            crate::backend::runtime::execution::layerwise::PreparedDenseController::control_bytes(
            )?,
            std::mem::size_of::<std::sync::MutexGuard<'_, ManagerState>>(),
            std::mem::size_of::<std::sync::TryLockError<std::sync::MutexGuard<'_, ManagerState>>>(),
            std::mem::size_of::<Result<(), ResidencyError>>(),
            std::mem::size_of::<[&ManagerWeak; 1]>(),
        ];
        controls
            .into_iter()
            .try_fold(std::mem::size_of_val(&controls), usize::checked_add)
    }
    pub(crate) fn with_original_dense_ledger<T>(
        &self,
        source: &ManagerWeak,
        operation: impl FnOnce(&eredu_core::residency::ResidencyLedger) -> T,
    ) -> Result<T, ResidencyError> {
        if source.as_ptr() != self.inner.as_ptr()
            || self.inner.source_custody().is_none()
            || self
                .inner
                .failed_transfer
                .load(std::sync::atomic::Ordering::Acquire)
        {
            return Err(ResidencyError::OriginalOperationDomain);
        }
        let state = self
            .inner
            .state
            .try_lock()
            .map_err(|_| ResidencyError::OriginalManagerBusy)?;
        Ok(operation(state.control.ledger()))
    }
    /// Clone only this constructor's raw source custody. No allocation or grant.
    pub(crate) fn original_source_custody(&self) -> Option<ManagerCustody> {
        self.inner.source_custody()
    }

    /// Ordinary worker preparation over the actual retained unit declarations,
    /// including canonical owners. This borrows no recovery/native operation.
    pub(crate) fn prefetch_unit_domain(&self) -> Result<Vec<OffloadUnitId>, ResidencyError> {
        let state = self
            .inner
            .state
            .lock()
            .map_err(|_| ResidencyError::StatePoisoned)?;
        Ok(state
            .control
            .units()
            .map(|unit| unit.id().clone())
            .collect())
    }

    /// Retains this manager's current physical arrays, host buffers and original
    /// or transformed source stores. This cold query performs no recovery reap,
    /// completion poll, payload read, eviction or native materialization.
    ///
    /// In-flight or failed transfers make the bound unknown because they can own
    /// resources outside the manager's storage map. This snapshot does not pin
    /// ledger entries or price future streaming, and it must be composed with
    /// module parameters, state and other owners before request admission.
    pub fn retained_storage(&self) -> Result<super::storage::RetainedStorage, ResidencyError> {
        let mut storage = crate::backend::runtime::residency::storage::RetainedStorage::default();
        self.collect_retained_storage(&mut storage)?;
        Ok(storage)
    }

    /// Fills caller-owned storage without allocating an intermediate inventory.
    pub fn collect_retained_storage(
        &self,
        storage: &mut crate::backend::runtime::residency::storage::RetainedStorage,
    ) -> Result<(), ResidencyError> {
        for source in self.inner.sources.ordinary_sources() {
            storage.include_checkpoint_source(source.as_ref())?;
        }
        if let ResidencySources::Original(source) = &self.inner.sources {
            for host in source.hosts() {
                for buffer in host.buffers.values() {
                    storage.include_retained_host(buffer.clone())?;
                }
            }
        }
        // `self.lock()` reaps native recovery. Admission inspection must not
        // establish completion as a side effect, so take only the state mutex.
        let state = self
            .inner
            .state
            .lock()
            .map_err(|_| ResidencyError::StatePoisoned)?;
        if self
            .inner
            .failed_transfer
            .load(std::sync::atomic::Ordering::Acquire)
        {
            storage.mark_incomplete();
        }
        for (id, retained) in &state.storage {
            for tier in [MemoryTier::Host, MemoryTier::Device] {
                if state
                    .control
                    .ledger()
                    .copy_status(id, tier)?
                    .is_some_and(|copy| copy.in_flight().is_some())
                {
                    storage.mark_incomplete();
                }
            }
            if let Some(host) = &retained.host {
                for buffer in host.buffers.values() {
                    storage.include_retained_host(buffer.clone())?;
                }
            }
            if let Some(device) = &retained.device {
                // The owning collector needs the same positive host witness as
                // the borrowed visitor. These source owners belong to device
                // cells, not the Host tier, and can back zero-copy array aliases.
                for buffer in device.arrays.host_sources() {
                    storage.include_retained_host(buffer.clone())?;
                }
                for array in device.arrays.retained_values() {
                    match array {
                        super::storage::RetainedStorageRef::CanonicalArray(cell) => {
                            storage.include_canonical_array(cell)?
                        }
                        super::storage::RetainedStorageRef::Array(array) => {
                            storage.include_array(array)?
                        }
                        _ => unreachable!("array iterator"),
                    }
                }
            }
        }
        Ok(())
    }

    /// Returns the MLX stream index used for device residency transfers.
    pub fn device_stream_index(&self) -> Result<i32, ResidencyError> {
        self.lock()?
            .device_stream
            .get_index()
            .map_err(|source| ResidencyError::Mlx {
                id: internal_id(),
                operation: "device residency stream index",
                source,
            })
    }

    /// Validates plan/unit identity, binding sizes, selections, and streams.
    ///
    /// Construction does not create MLX arrays. Call [`Self::initialize`] to
    /// materialize units assigned to host or device by the plan.
    pub fn new<S>(
        store: Arc<S>,
        plan: OffloadPlan,
        units: impl IntoIterator<Item = OffloadUnit>,
        source_stream: Stream,
        device_stream: Stream,
    ) -> Result<Self, ResidencyError>
    where
        S: eredu_checkpoint::store::CheckpointSource + 'static,
    {
        let store: eredu_checkpoint::store::RetainedCheckpointSource = store.into();
        Self::new_shared(store, plan, units, source_stream, device_stream)
    }

    /// Creates a manager from an already type-erased checkpoint store.
    pub fn new_shared(
        store: impl Into<eredu_checkpoint::store::RetainedCheckpointSource>,
        plan: OffloadPlan,
        units: impl IntoIterator<Item = OffloadUnit>,
        source_stream: Stream,
        device_stream: Stream,
    ) -> Result<Self, ResidencyError> {
        Self::new_retained_sources(
            store.into(),
            BTreeMap::new(),
            plan,
            units,
            source_stream,
            device_stream,
        )
    }

    /// Creates one shared reservation ledger over exact per-unit source stores.
    /// Units absent from `unit_sources` use the primary source. Sources are
    /// retained as admitted; no checkpoint is reopened or resolved here.
    pub fn new_shared_sources(
        store: eredu_checkpoint::store::SharedCheckpointSource,
        unit_sources: BTreeMap<OffloadUnitId, eredu_checkpoint::store::SharedCheckpointSource>,
        plan: OffloadPlan,
        units: impl IntoIterator<Item = OffloadUnit>,
        source_stream: Stream,
        device_stream: Stream,
    ) -> Result<Self, ResidencyError> {
        Self::new_retained_sources(
            store.into(),
            unit_sources
                .into_iter()
                .map(|(id, source)| (id, source.into()))
                .collect(),
            plan,
            units,
            source_stream,
            device_stream,
        )
    }

    /// Same manager driver retaining opaque source roots without outgoing Arc.
    pub fn new_retained_sources(
        store: eredu_checkpoint::store::RetainedCheckpointSource,
        unit_sources: BTreeMap<OffloadUnitId, eredu_checkpoint::store::RetainedCheckpointSource>,
        plan: OffloadPlan,
        units: impl IntoIterator<Item = OffloadUnit>,
        source_stream: Stream,
        device_stream: Stream,
    ) -> Result<Self, ResidencyError> {
        Self::new_shared_sources_impl(
            store,
            unit_sources,
            plan,
            units,
            source_stream,
            device_stream,
            None,
        )
    }

    /// Same factory with an actually admitted shared cache. Only this fixed
    /// cache owner is covered; manager/catalog/stream producers stay separate.
    /// The ordinary factory never silently adopts or replaces its cache.
    pub(crate) fn new_shared_sources_with_cache(
        store: impl Into<eredu_checkpoint::store::RetainedCheckpointSource>,
        unit_sources: BTreeMap<OffloadUnitId, eredu_checkpoint::store::RetainedCheckpointSource>,
        plan: OffloadPlan,
        units: impl IntoIterator<Item = OffloadUnit>,
        source_stream: Stream,
        device_stream: Stream,
        cache: crate::backend::runtime::checkpoint::store::CacheHandle,
        pool: &eredu_runtime::working_memory::MemoryLedger,
    ) -> Result<Self, ResidencyError> {
        Self::new_shared_sources_impl(
            store.into(),
            unit_sources,
            plan,
            units,
            source_stream,
            device_stream,
            Some((cache, pool)),
        )
    }

    fn new_shared_sources_impl(
        store: eredu_checkpoint::store::RetainedCheckpointSource,
        unit_sources: BTreeMap<OffloadUnitId, eredu_checkpoint::store::RetainedCheckpointSource>,
        plan: OffloadPlan,
        units: impl IntoIterator<Item = OffloadUnit>,
        source_stream: Stream,
        device_stream: Stream,
        cache: Option<(
            crate::backend::runtime::checkpoint::store::CacheHandle,
            &eredu_runtime::working_memory::MemoryLedger,
        )>,
    ) -> Result<Self, ResidencyError> {
        let sources = ResidencySources::Ordinary {
            primary: store,
            units: unit_sources.into_iter().collect(),
        };
        let units = units.into_iter().collect::<Vec<_>>();
        let control = ResidencyController::new_with_catalogs(|id| sources.source(id), plan, units)?;
        for id in match &sources {
            ResidencySources::Ordinary { units, .. } => units.keys(),
            _ => unreachable!("ordinary constructor"),
        } {
            if control.unit(id).is_none() {
                return Err(
                    ResidencyControllerError::UnexpectedUnitDefinition { id: id.clone() }.into(),
                );
            }
        }
        preflight_residency_owner_bindings(&sources, &control)?;

        let source_device = source_stream
            .get_device()
            .map_err(|source| ResidencyError::Mlx {
                id: internal_id(),
                operation: "source stream inspection",
                source,
            })?;
        if source_device
            .get_type()
            .map_err(|source| ResidencyError::Mlx {
                id: internal_id(),
                operation: "source device inspection",
                source,
            })?
            != DeviceType::Cpu
        {
            return Err(ResidencyError::InvalidSourceStream);
        }

        // Preserve every existing source/plan/stream validation above. Check
        // supplied origin before any new manager/context owner below is built.
        if let Some((cache, pool)) = &cache {
            cache
                .validate_pool(pool)
                .map_err(ResidencyError::OriginalCache)?;
        }
        let prepared_streams = match &cache {
            Some((_, pool)) => Some(crate::backend::runtime::checkpoint::store::PreparedMaterializationStreams::prepare(pool, &source_stream, &device_stream)
                .map_err(ResidencyError::OriginalStreams)?),
            None => None,
        };
        let alias_owner_pins = rows::AliasPins::new(control.units().map(OffloadUnit::id));
        let storage = control
            .units()
            .map(|unit| (unit.id().clone(), UnitStorage::default()))
            .collect();
        let custody = owner::ManagerCustody::default();
        let failed_transfer = owner::FailureFlag::new(custody.clone());
        Ok(Self {
            inner: ManagerOwner::new(ManagerInner {
                parameter_exclusions: None,
                parameter_constructors: None,
                sources,
                host_workspace: std::sync::OnceLock::new(),
                dense_controller: std::sync::OnceLock::new(),
                original_operation_source:std::sync::OnceLock::new(),
                background_operation_source: std::sync::OnceLock::new(),
                supplementary_source:std::sync::OnceLock::new(),
                failed_transfer: failed_transfer.clone(),
                state: Mutex::new(ManagerState {
                    failed_transfer,
                    control,
                    storage,
                    alias_owner_pins,
                    admitted_disk_route: std::sync::Weak::new(),
                    admitted_disk_window: BTreeSet::new(),
                    materialization: match cache {
                        Some((cache, pool)) => crate::backend::runtime::checkpoint::store::ManagerMaterializationContext::prepared(
                            prepared_streams.expect("prepared exact stream pair"), cache, pool,
                        ).map_err(ResidencyError::OriginalCache)?,
                        None => crate::backend::runtime::checkpoint::store::ManagerMaterializationContext::ordinary(
                            MlxParameterMaterializationContext::new(&source_stream, &device_stream)
                        ),
                    },
                    source_stream: owner::ManagerStream::Ordinary(source_stream),
                    device_stream: owner::ManagerStream::Ordinary(device_stream),
                }),
                changed: Condvar::new(),
            }, custody),
        })
    }

    /// One selected-plan lookup or installation check, including the private
    /// state loan. The two calls are sequential; no shared cache is allocated.
    pub(crate) fn gguf_cache_identity_control_bytes() -> Option<usize> {
        use crate::backend::runtime::checkpoint::store::{
            CacheHandle, ManagerMaterializationContext, MaterializationView,
        };
        let controls = [
            size_of::<&Self>(),
            size_of::<&CacheHandle>(),
            size_of::<&ManagerMaterializationContext>(),
            size_of::<MaterializationView<'static>>(),
            size_of::<std::sync::TryLockResult<MutexGuard<'static, ManagerState>>>(),
            size_of::<std::sync::TryLockError<MutexGuard<'static, ManagerState>>>(),
            size_of::<MutexGuard<'static, ManagerState>>(),
            size_of::<Result<MutexGuard<'static, ManagerState>, ResidencyError>>(),
            size_of::<Result<CacheHandle, ResidencyError>>(),
            size_of::<Result<(), ResidencyError>>(),
            size_of::<ResidencyError>(),
            size_of::<bool>(),
        ];
        controls
            .into_iter()
            .try_fold(CacheHandle::identity_control_bytes()?, usize::checked_add)?
            .checked_add(std::mem::size_of_val(&controls))
    }

    /// Retain only the actual cache owner. No stream clone, source callback or
    /// native operation occurs while the manager state is borrowed.
    pub(crate) fn gguf_cache_handle(
        &self,
    ) -> Result<crate::backend::runtime::checkpoint::store::CacheHandle, ResidencyError> {
        let state = self.inner.state.try_lock().map_err(|error| match error {
            std::sync::TryLockError::WouldBlock => ResidencyError::OriginalManagerBusy,
            std::sync::TryLockError::Poisoned(_) => ResidencyError::StatePoisoned,
        })?;
        Ok(state.materialization.cache_handle())
    }

    pub(crate) fn validate_gguf_cache(
        &self,
        expected: &crate::backend::runtime::checkpoint::store::CacheHandle,
    ) -> Result<(), ResidencyError> {
        let state = self.inner.state.try_lock().map_err(|error| match error {
            std::sync::TryLockError::WouldBlock => ResidencyError::OriginalManagerBusy,
            std::sync::TryLockError::Poisoned(_) => ResidencyError::StatePoisoned,
        })?;
        if state.materialization.matches_cache(expected) {
            Ok(())
        } else {
            Err(ResidencyError::OriginalOperationDomain)
        }
    }

    /// Materializes all planned host and device units in identifier order.
    ///
    /// Disk units remain array-free. A failure never publishes a partial unit;
    /// units completed earlier remain resident and fully accounted, allowing a
    /// caller to inspect the report and retry initialization.
    pub fn initialize(&self) -> Result<(), ResidencyError> {
        let mut state = self.lock()?;
        if state.control.ledger_mut().initialized() {
            return Ok(());
        }
        let assignments = state
            .control
            .ledger_mut()
            .plan()
            .units()
            .iter()
            .map(|unit| (unit.id().clone(), unit.tier()))
            .collect::<Vec<_>>();
        for (id, tier) in assignments {
            if tier != MemoryTier::Disk {
                ensure_resident(&mut state, &self.inner.sources, &id, tier, true)?;
            }
        }
        state.control.ledger_mut().mark_initialized();
        Ok(())
    }

    /// Synchronously prepares one host or device copy and records hit/miss telemetry.
    ///
    /// This provides caller-directed lookahead but does not overlap transfer
    /// with computation.
    pub fn prefetch(
        &self,
        id: &OffloadUnitId,
        tier: MemoryTier,
    ) -> Result<PrefetchOutcome, ResidencyError> {
        validate_target(tier, "prefetch")?;
        let mut state = self.lock()?;
        disk_workspace::validate_disk_access(&state, std::slice::from_ref(id), tier)?;
        loop {
            state.control.ledger_mut().require_initialized()?;
            let copy = state.control.ledger_mut().copy_status(id, tier)?;
            if !copy.is_some_and(|copy| copy.in_flight().is_some()) {
                break;
            }
            state = self.wait_for_transfer(state)?;
        }
        prefetch_locked(&mut state, &self.inner.sources, id, tier)
    }

    /// Ensures residency and returns an RAII lease protecting the requested copy.
    pub fn acquire(
        &self,
        id: &OffloadUnitId,
        tier: MemoryTier,
    ) -> Result<ResidentUnitLease, ResidencyError> {
        self.acquire_with_demand(id, tier, 1)
    }

    /// Ensures residency and records weighted demand for eviction policy.
    ///
    /// `demand` may be larger than one when duplicate entry requests
    /// share a single acquisition. Frequency counters saturate on overflow.
    pub fn acquire_with_demand(
        &self,
        id: &OffloadUnitId,
        tier: MemoryTier,
        demand: u64,
    ) -> Result<ResidentUnitLease, ResidencyError> {
        self.acquire_many_with_demand(&[(id.clone(), demand)], tier)?
            .pop()
            .ok_or(ResidencyError::StatePoisoned)
    }

    /// Acquires a deterministic entry set with one batched residency transition.
    ///
    /// Missing copies reserve capacity before any materialization starts. All
    /// requested units are protected from eviction, all lazy outputs are
    /// evaluated together, and leases are published only after the batch is
    /// complete.
    pub fn acquire_many_with_demand(
        &self,
        requests: &[(OffloadUnitId, u64)],
        tier: MemoryTier,
    ) -> Result<Vec<ResidentUnitLease>, ResidencyError> {
        self.acquire_many_with_mode(requests, tier, false)
            .map(|(leases, _)| leases.into_ordinary())
    }

    /// Submits one residency batch and returns its owning completion lease.
    ///
    /// Missing copies are submitted to MLX for asynchronous evaluation, but
    /// this method does not block the host for their completion. Call
    /// [`ResidentTransfer::order_after`] before evaluating work on another
    /// compatible stream. The transfer guard owns every source dependency
    /// until it is synchronized or dropped.
    pub fn acquire_many_with_transfer(
        &self,
        requests: &[(OffloadUnitId, u64)],
        tier: MemoryTier,
    ) -> Result<ResidentTransfer, ResidencyError> {
        let (leases, submitted) = self.acquire_many_with_mode(requests, tier, true)?;
        let transfer = match submitted {
            None => ResidentTransfer::immediate(leases.into_ordinary(), tier),
            Some(submitted) => ResidentTransfer::submitted(leases, submitted),
        };
        Ok(transfer)
    }

    pub(crate) fn acquire_many_with_original_transfer(
        &self,
        requests: &[(OffloadUnitId, u64)],
        tier: MemoryTier,
        slots: &mut OriginalResidencySlots<'_>,
        observer: &safemlx::OriginalScopeObserver,
    ) -> Result<ResidentTransfer, ResidencyError> {
        // Host publication requires the exact admitted worker and one-use
        // destination. Other original acquisitions remain Device promotions.
        if tier != MemoryTier::Device
            && !(tier == MemoryTier::Host && slots.background_host.is_some())
        {
            return Err(ResidencyError::OriginalOperationDomain);
        }
        transfer::validate_original_observer(observer)?;
        // Reserve the root acquisition's warm-hit owner before any pin/source
        // work. Missing batches additionally consume their own transfer slots.
        let mut immediate = slots.transfers.checkout().map_err(|cause| {
            ResidencyError::OriginalOperationCapacity {
                family: "resident warm acquisition",
                prepared: cause.prepared,
            }
        })?;
        let prepared_leases = immediate.take_lease_collection(&self.inner, requests, tier)?;
        let (leases, submitted) = self.acquire_many_with_operations(
            requests,
            tier,
            true,
            Some((slots, observer)),
            Some(prepared_leases),
            None,
        )?;
        Ok(match submitted {
            None => ResidentTransfer::immediate_original(leases, tier, observer, immediate),
            Some(submitted) => ResidentTransfer::submitted(leases, submitted),
        })
    }

    fn acquire_many_with_mode(
        &self,
        requests: &[(OffloadUnitId, u64)],
        tier: MemoryTier,
        return_transfer: bool,
    ) -> Result<
        (
            transfer::ResidentLeaseCollection,
            Option<SubmittedResidentTransfer>,
        ),
        ResidencyError,
    > {
        self.acquire_many_with_operations(requests, tier, return_transfer, None, None, None)
    }

    fn acquire_many_with_operations(
        &self,
        requests: &[(OffloadUnitId, u64)],
        tier: MemoryTier,
        return_transfer: bool,
        mut original: Option<(
            &mut OriginalResidencySlots<'_>,
            &safemlx::OriginalScopeObserver,
        )>,
        prepared_leases: Option<transfer::PreparedLeaseCollection>,
        mut host: Option<&mut host_acquisition::PreparedHostAcquisition>,
    ) -> Result<
        (
            transfer::ResidentLeaseCollection,
            Option<SubmittedResidentTransfer>,
        ),
        ResidencyError,
    > {
        if (original.is_some() && host.is_some())
            || (original.is_some() || host.is_some()) != prepared_leases.is_some()
        {
            return Err(ResidencyError::OriginalOperationDomain);
        }
        if let Some(leases) = &prepared_leases {
            leases.validate(&self.inner, requests, tier)?;
        }
        if let Some((_, observer)) = &original {
            let current = safemlx::OriginalScopeObserver::require_current()
                .map_err(ResidencyError::OriginalNative)?;
            if !current.same_scope(observer) {
                return Err(ResidencyError::OriginalOperationDomain);
            }
        }
        validate_target(tier, "acquire")?;
        let mut state = match &original {
            Some((_, observer)) => self.lock_original(observer)?,
            None => self.lock()?,
        };
        // The original window's final lease IDs serve this immutable request
        // phase too. Their loan ends before LeaseBuilder moves the same buffer
        // into its consuming iterator. Ordinary collection behavior is retained.
        let ordinary_ids = prepared_leases.is_none().then(|| {
            requests
                .iter()
                .map(|(id, _)| id.clone())
                .collect::<Vec<_>>()
        });
        let ids = match &prepared_leases {
            Some(prepared) => prepared.requested_ids()?,
            None => ordinary_ids.as_deref().expect("ordinary request IDs"),
        };
        disk_workspace::validate_disk_access(&state, ids, tier)?;
        let controller = match original.as_mut() {
            Some((slots, _)) => Some(
                PreparedControllerAttempt::take(
                    slots.controller,
                    &self.inner,
                    state.control.ledger(),
                    ids.len(),
                )?
                .validate(state.control.ledger(), ids, tier)?,
            ),
            None => {
                if let Some(host) = host.as_mut() {
                    Some(
                        PreparedControllerAttempt::take(
                            &mut host.controller,
                            &self.inner,
                            state.control.ledger(),
                            ids.len(),
                        )?
                        .validate(state.control.ledger(), ids, tier)?,
                    )
                } else {
                    state.control.ledger_mut().validate_batch(ids, tier)?;
                    None
                }
            }
        };
        loop {
            state.control.ledger_mut().require_initialized()?;
            for (id, _) in requests {
                state.control.ledger_mut().spec(id)?;
            }
            let waiting = requests.iter().any(|(id, _)| {
                state
                    .control
                    .ledger_mut()
                    .copy_status(id, tier)
                    .ok()
                    .flatten()
                    .is_some_and(|copy| copy.in_flight().is_some())
            });
            if !waiting {
                break;
            }
            if original.is_some() {
                // The selected caller drains its actual preceding owner. This
                // method cannot globally reap or adopt another transfer here.
                return Err(ResidencyError::OriginalPendingTransfer);
            }
            state = self.wait_for_transfer(state)?;
        }
        let missing = requests
            .iter()
            .filter(|(id, _)| {
                !state
                    .control
                    .ledger_mut()
                    .is_resident(id, tier)
                    .unwrap_or(false)
            })
            .count();
        let started = Instant::now();
        let residency = transfer::ensure_many_resident_with_operations(
            &mut state,
            &self.inner.sources,
            ids,
            tier,
            return_transfer,
            false,
            Some(&self.inner),
            original,
            controller,
            host,
        );
        if missing > 0 {
            state
                .control
                .ledger_mut()
                .record_prefetch_stall(started.elapsed());
        }
        let (_, submitted) = residency?;
        if let Some(submitted) = &submitted {
            submitted.attach_owner(self.inner.downgrade());
        }
        let mut leases = transfer::LeaseBuilder::new(prepared_leases);
        let acquired = (|| -> Result<(), ResidencyError> {
            for (id, demand) in requests {
                let unit = state.storage.get(id).ok_or(ResidencyError::StatePoisoned)?;
                let storage = match tier {
                    MemoryTier::Host => ResidentLeaseStorage::Host(Clone::clone(
                        unit.host.as_ref().ok_or(ResidencyError::StatePoisoned)?,
                    )),
                    MemoryTier::Device => ResidentLeaseStorage::Device(
                        unit.device
                            .as_ref()
                            .ok_or(ResidencyError::StatePoisoned)?
                            .clone(),
                    ),
                    MemoryTier::Disk => unreachable!("validated above"),
                };
                leases.acquire(
                    state.control.ledger_mut(),
                    id,
                    tier,
                    *demand,
                    storage,
                    self.inner.downgrade(),
                )?;
            }
            Ok(())
        })();
        if let Err(error) = acquired {
            if let Some(submitted) = &submitted {
                submitted.retain_partial_leases(leases.finish());
            }
            return Err(error);
        }
        Ok((leases.finish(), submitted))
    }

    /// Returns whether a logical copy currently resides in a memory tier.
    pub fn is_resident(
        &self,
        id: &OffloadUnitId,
        tier: MemoryTier,
    ) -> Result<bool, ResidencyError> {
        validate_target(tier, "is_resident")?;
        let state = self.lock()?;
        Ok(state.control.ledger().is_resident(id, tier)?)
    }

    /// Fills caller-owned scalar rows from one coherent logical cache snapshot.
    /// The identifiers borrow the retained catalog; no diagnostic inventory or
    /// source description is allocated while the state lock is held.
    pub(crate) fn fill_resident_capacities(
        &self,
        rows: &mut [(&OffloadUnitId, Option<u64>, Option<u64>)],
    ) -> Result<(), ResidencyError> {
        let state = self.lock()?;
        let ledger = state.control.ledger();
        for (id, host, device) in rows {
            *host = ledger
                .copy_status(id, MemoryTier::Host)?
                .map(|copy| copy.bytes());
            *device = ledger
                .copy_status(id, MemoryTier::Device)?
                .map(|copy| copy.bytes());
        }
        Ok(())
    }

    /// Replaces the protected window and synchronously prepares bounded lookahead.
    ///
    /// `active` units are protected from automatic eviction. At most the first
    /// configured number of distinct `upcoming` units are prefetched, in caller
    /// order. Repeated and overlapping windows are deterministic.
    pub fn prepare_window(
        &self,
        active: &[OffloadUnitId],
        upcoming: &[OffloadUnitId],
        tier: MemoryTier,
    ) -> Result<Vec<(OffloadUnitId, PrefetchOutcome)>, ResidencyError> {
        self.prepare_group_window("default", active, upcoming, tier)
    }

    /// Replaces one named group's protected window and prepares bounded lookahead.
    ///
    /// Protection owned by other groups remains active. This permits independent
    /// text, vision, audio, temporal, and depth stack scheduling.
    pub fn prepare_group_window(
        &self,
        group: &str,
        active: &[OffloadUnitId],
        upcoming: &[OffloadUnitId],
        tier: MemoryTier,
    ) -> Result<Vec<(OffloadUnitId, PrefetchOutcome)>, ResidencyError> {
        validate_target(tier, "prepare_group_window")?;
        let mut state = self.lock()?;
        disk_workspace::validate_disk_access(
            &state,
            &active.iter().chain(upcoming).cloned().collect::<Vec<_>>(),
            tier,
        )?;
        loop {
            state.control.ledger_mut().require_initialized()?;
            for id in active.iter().chain(upcoming) {
                state.control.ledger_mut().spec(id)?;
            }
            let waiting = active.iter().chain(upcoming).any(|id| {
                state
                    .control
                    .ledger_mut()
                    .copy_status(id, tier)
                    .ok()
                    .flatten()
                    .is_some_and(|copy| copy.in_flight().is_some())
            });
            if !waiting {
                break;
            }
            state = self.wait_for_transfer(state)?;
        }
        let selected = state
            .control
            .commit_group_window(group, active, upcoming, tier)?;
        selected
            .into_iter()
            .map(|id| {
                prefetch_locked(&mut state, &self.inner.sources, &id, tier)
                    .map(|outcome| (id, outcome))
            })
            .collect()
    }

    /// Replaces one named protected window without materializing its units.
    ///
    /// This is used by schedulers that submit materialization through a
    /// separate bounded service. Protection owned by other named windows is
    /// preserved.
    pub fn protect_group_window(
        &self,
        group: &str,
        active: &[OffloadUnitId],
        tier: MemoryTier,
    ) -> Result<(), ResidencyError> {
        let mut state = self.lock()?;
        validate_target(tier, "protect_group_window")?;
        state.control.protect_group_window(group, active, tier)?;
        Ok(())
    }

    /// Drops completed device copies outside the selected unit window, keeping
    /// the exact persistent alias owners of the admitted source. Callers first
    /// settle preceding consumers and replace their group-window protection.
    pub(crate) fn trim_device_units(
        &self,
        units: &[OffloadUnitId],
        active: &[OffloadUnitId],
    ) -> Result<(), ResidencyError> {
        if let Some(persistent) = self.original_foreground_persistent_units() {
            for id in units {
                if !active.contains(id) && !persistent.clone().any(|owner| owner == id) {
                    self.evict(id, MemoryTier::Device)?;
                }
            }
        } else {
            let persistent = self.admitted_disk_persistent_units();
            for id in units {
                if !active.contains(id) && !persistent.contains(id) {
                    // Ordinary aliases retain the canonical owner's explicit
                    // logical pin independently of an admitted disk receipt.
                    // Other live leases still pass through evict's refusal.
                    let canonical_alias = self
                        .lock()?
                        .alias_owner_pins
                        .contains(id, MemoryTier::Device);
                    if !canonical_alias {
                        self.evict(id, MemoryTier::Device)?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Explicitly evicts one host or device copy.
    ///
    /// Evicting an absent copy is an idempotent success returning `false`.
    pub fn evict(&self, id: &OffloadUnitId, tier: MemoryTier) -> Result<bool, ResidencyError> {
        validate_target(tier, "evict")?;
        let mut state = self.lock()?;
        let Some(evicted) = state.control.ledger_mut().evict(id, tier)? else {
            return Ok(false);
        };
        if !state
            .storage
            .get_mut(id)
            .is_some_and(|unit| unit.remove_storage(tier))
        {
            return Err(ResidencyError::StatePoisoned);
        }
        debug_assert_eq!(evicted.id, *id);
        Ok(true)
    }

    /// Samples optional MLX allocator and process metrics on explicit request.
    pub fn sample_memory(
        &self,
        include_mlx: bool,
        include_process: bool,
    ) -> Result<(), ResidencyError> {
        let mut state = self.lock()?;
        if include_mlx {
            let metrics = sample_allocator_memory().map_err(|source| ResidencyError::Mlx {
                id: internal_id(),
                operation: "allocator memory sampling",
                source,
            })?;
            state.control.ledger_mut().record_allocator_memory(metrics);
        }
        if include_process {
            state.control.ledger_mut().sample_process_metrics();
        }
        Ok(())
    }

    /// Borrows the actual detached payload counter also exposed by the
    /// catalog diagnostics returned by `report`.
    #[cfg(test)]
    pub(crate) fn detached_physical_read_bytes(&self, source: usize) -> Option<u64> {
        match &self.inner.sources {
            ResidencySources::Original(owner) => owner.detached_physical_read_bytes(source),
            ResidencySources::Ordinary { .. } => None,
        }
    }

    /// Returns an immutable point-in-time residency and storage report.
    pub fn report(&self) -> Result<ResidencyReport, ResidencyError> {
        let (initialized, offload, units, active_window) = self.telemetry_snapshot()?;
        let (primary, unit_sources) = match &self.inner.sources {
            ResidencySources::Ordinary { primary, units } => (
                primary.source_diagnostics()?,
                units
                    .iter()
                    .map(|(id, source)| {
                        source.source_diagnostics().map(|value| (id.clone(), value))
                    })
                    .collect::<Result<_, _>>()?,
            ),
            ResidencySources::Original(source) => (
                source.policy().source_diagnostics()?,
                units
                    .iter()
                    .map(|unit| {
                        source
                            .catalog(unit.id())
                            .source_diagnostics()
                            .map(|value| (unit.id().clone(), value))
                    })
                    .collect::<Result<_, _>>()?,
            ),
        };
        Ok(
            ResidencyReport::new(initialized, offload, units, active_window, primary)
                .with_unit_sources(unit_sources),
        )
    }
    /// Returns initialized state, aggregate telemetry, unit reports, and active window.
    pub fn telemetry_snapshot(
        &self,
    ) -> Result<
        (
            bool,
            OffloadReport,
            Vec<UnitResidencyReport>,
            Vec<OffloadUnitId>,
        ),
        ResidencyError,
    > {
        let state = self.lock()?;
        let active = state.control.ledger().active_window();
        let units = state.control.ledger().unit_reports();
        Ok((
            state.control.ledger().initialized(),
            state.control.ledger().telemetry(),
            units,
            active.into_iter().collect(),
        ))
    }

    fn lock_original(
        &self,
        observer: &safemlx::OriginalScopeObserver,
    ) -> Result<MutexGuard<'_, ManagerState>, ResidencyError> {
        // Explicit original entry performs no global recovery/ordinary reaping.
        if self
            .inner
            .failed_transfer
            .load(std::sync::atomic::Ordering::Acquire)
        {
            let cause = observer.retained_failure().unwrap_or_else(|| {
                observer
                    .observation_error(safemlx::ScopedSubmissionProgress::Unobservable)
                    .expect("fixed original refusal")
            });
            return Err(ResidencyError::OriginalNative(cause));
        }
        match self.inner.state.try_lock() {
            Ok(state) => Ok(state),
            Err(std::sync::TryLockError::WouldBlock) => Err(ResidencyError::OriginalManagerBusy),
            Err(std::sync::TryLockError::Poisoned(_)) => Err(ResidencyError::StatePoisoned),
        }
    }

    fn lock(&self) -> Result<MutexGuard<'_, ManagerState>, ResidencyError> {
        crate::backend::submission_recovery::reap();
        crate::backend::ordinary_retirement::reclaim();
        if self
            .inner
            .failed_transfer
            .load(std::sync::atomic::Ordering::Acquire)
        {
            return Err(ResidencyError::Mlx {
                id: internal_id(),
                operation: "resident transfer admission",
                source: safemlx::error::Exception::custom(
                    "residency manager is poisoned by a failed or unobservable native transfer",
                ),
            });
        }
        self.inner
            .state
            .lock()
            .map_err(|_| ResidencyError::StatePoisoned)
    }

    fn wait_for_transfer<'a>(
        &'a self,
        state: MutexGuard<'a, ManagerState>,
    ) -> Result<MutexGuard<'a, ManagerState>, ResidencyError> {
        let (state, _) = self
            .inner
            .changed
            .wait_timeout(state, std::time::Duration::from_millis(25))
            .map_err(|_| ResidencyError::StatePoisoned)?;
        drop(state);
        // Recovery may stage manager-lock owners; never reclaim under this mutex.
        self.lock()
    }
}

impl ResidencyWindowManager for ResidencyManager {
    type Error = ResidencyError;

    fn prepare_window(
        &self,
        active: &[OffloadUnitId],
        upcoming: &[OffloadUnitId],
        tier: MemoryTier,
    ) -> Result<Vec<(OffloadUnitId, PrefetchOutcome)>, Self::Error> {
        ResidencyManager::prepare_window(self, active, upcoming, tier)
    }

    fn prepare_group_window(
        &self,
        group: &str,
        active: &[OffloadUnitId],
        upcoming: &[OffloadUnitId],
        tier: MemoryTier,
    ) -> Result<Vec<(OffloadUnitId, PrefetchOutcome)>, Self::Error> {
        ResidencyManager::prepare_group_window(self, group, active, upcoming, tier)
    }

    fn evict(&self, id: &OffloadUnitId, tier: MemoryTier) -> Result<bool, Self::Error> {
        ResidencyManager::evict(self, id, tier)
    }

    fn unit_reports(&self) -> Result<Vec<UnitResidencyReport>, Self::Error> {
        self.telemetry_snapshot().map(|(_, _, units, _)| units)
    }
}

mod borrowed_storage;
mod capacity;
pub(crate) use capacity::PreparedWeightOwnerSlotBounds;

mod transfer;
use transfer::*;
pub use transfer::{
    ManagerInner, ResidentArrays, ResidentHostBuffers, ResidentTransfer, ResidentTransferResources,
};

mod named_arrays;
pub(crate) use named_arrays::CanonicalArrayOwner;
use named_arrays::NamedArrays;
pub(crate) use named_arrays::{
    NameCatalogOwner, NamePreparationError, NamedPreparationSource, NamedStorageLayout,
};
pub use named_arrays::{NamedArrayError, ResidentArraysOwner};
mod materialization;
pub use materialization::host_capacity_upper_bound_for_bindings;
pub(crate) use materialization::original_host_copy_control_bytes;
use materialization::*;

mod host_workspace;
pub(crate) use host_workspace::{
    HostCopyIdentity, HostCopySourcePins, HostCopyWorkspace, HostCopyWorkspaceError,
};
mod disk_workspace;
pub(crate) use disk_workspace::{
    DiskCopyWorkspace, DiskCopyWorkspaceError, DiskRouteGuard, DiskRouteReceipt,
    PreparedDiskReadPlans,
};

#[cfg(test)]
#[path = "manager/tests.rs"]
mod tests;

pub(crate) use materialization::PreparedHostMaterialization;
pub(crate) use transfer::{
    PreparedResidentTransfer, PreparedTransferDestinationCause, PreparedTransferObservation,
    TransferPayloadShape,
};

mod operation_population;
pub(crate) use operation_population::MaterializationPopulation;

mod control_custody;
mod controller_attempt;
use control_custody::ResidencyControlCustody;
pub use controller_attempt::PreparedAdmissionFailure;
pub(crate) use controller_attempt::{
    ControllerPreparationCause, ControllerPreparationError, PreparedControllerAttempt,
};

impl ResidencyError {
    pub(crate) fn capacity(&self) -> Option<eredu_core::residency::ResidencyCapacityRef<'_>> {
        match self {
            Self::Ledger(cause) => cause.capacity(),
            Self::OriginalAdmission(cause) => cause.capacity(),
            _ => None,
        }
    }
}

mod closure_ids;
pub(crate) use closure_ids::{
    ClosurePreparationCause, ClosurePreparationError, PreparedClosureIds,
};

mod operation_slots;
pub(crate) use operation_slots::{
    ForegroundDiskPopulation, ForegroundDiskSourceCapacity, ForegroundDiskSourceSeries,
    ForegroundDiskSubsetCeiling, ForegroundDiskWindowPlan, PreparedForegroundDiskSlots,
};
pub(crate) use operation_slots::{
    OriginalHostPublicationSlots, OriginalMaterializedLoan, OriginalResidencySlots,
};

#[cfg(test)]
pub(crate) use tests::exercise_original_capacity_retry;

#[cfg(test)]
pub(crate) use named_arrays::tests::NamedArraysFixture;

mod operation_source;
pub(crate) use operation_source::ForegroundDiskIdentity;
pub(crate) use operation_source::{
    OperationSourceFailure, OperationWindows, OriginalResidencySource, SelectedResidencySource,
    SupplementaryResidencySource, WindowPopulation,
};

#[cfg(test)]
pub(crate) use tests::LeaseReturnFixture;

pub(crate) mod acquisition_destinations;

/// Consume a declared immutable source alias for initial parameter binding.
/// Ordinary stores have no source slot; repeated original operation binding
/// uses its separate accepted request Graph shell population.
pub(crate) fn clone_original_source_value(
    lease: &ResidentUnitLease,
    name: &str,
) -> Result<Option<Array>, ResidencyError> {
    let ResidentLeaseStorage::Device(owner) = lease.storage() else {
        return Ok(None);
    };
    let NamedArrays::Source(values) = &owner.arrays else {
        return Ok(None);
    };
    let value = values
        .get(name)
        .ok_or(ResidencyError::OriginalOperationDomain)?;
    value
        .source
        .try_prepared_source_array()
        .map(Some)
        .map_err(ResidencyError::OriginalHostInput)
}
