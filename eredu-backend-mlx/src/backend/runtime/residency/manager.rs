//! MLX materialization and transfer execution for immutable weight units.
//!
//! Materializes exact checkpoint selections as immutable host-transfer buffers
//! or execution-stream arrays. The portable residency driver accounts for
//! logical host and device copies separately. MLX submits the admitted batches
//! and retains their native resources through exact completion.

use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
};

use safemlx::{
    host_transfer_capacity_upper_bound, transforms::async_eval_with_event, Array, DeviceType,
    Event, HostTransferBuffer, HostTransferPolicy, ImmutableHostTransferBuffer, Stream,
};

use crate::{
    backend::nn::shared::MlxNeuralBackend,
    backend::residency::sample_allocator_memory,
    backend::runtime::checkpoint::recipe::{MlxWeightRecipeExt, WeightRecipeError},
    backend::runtime::checkpoint::store::{
        CheckpointMaterializationError, MlxParameterMaterializationContext,
        PendingWeightMaterialization, WeightMaterialization,
    },
};
use eredu_core::residency::{
    MemoryTier, OffloadUnitId, ResidencyLease, ResidencyLeaseStorage, ResidencyLedgerError,
    TransferDirection,
};

use eredu_core::residency::parameter_binding::WeightBinding;
use eredu_core::residency::ResidencyObservationBindings;

/// A resident unit that prevents eviction of one tier until it is dropped.
pub type ResidentUnitLease = ResidencyLease<ResidentLeaseStorage>;

/// Host or device storage retained by a weight-residency lease.
pub enum ResidentLeaseStorage {
    /// Immutable host-transfer buffers.
    Host(Arc<ResidentHostBuffers>),
    /// Materialized device arrays.
    Device(Arc<ResidentArrays>),
}

impl ResidencyLeaseStorage for ResidentLeaseStorage {
    type DeviceValue = Array;
    type HostValue = ImmutableHostTransferBuffer;
    type Error = ResidencyError;
    type BindingNames<'a> = Box<dyn Iterator<Item = &'a str> + 'a>;

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
            ResidentLeaseStorage::Host(buffers) => buffers
                .buffers
                .get(name)
                .map(|allocation| &allocation.buffer)
                .ok_or_else(|| ResidencyError::UnknownBinding {
                    id: id.clone(),
                    name: name.to_string(),
                }),
            ResidentLeaseStorage::Device(_) => Err(ResidencyError::DeviceBindingIsNotHostBuffer {
                id: id.clone(),
                name: name.to_string(),
            }),
        }
    }

    fn binding_names(&self) -> Self::BindingNames<'_> {
        match self {
            ResidentLeaseStorage::Host(buffers) => {
                Box::new(buffers.buffers.keys().map(String::as_str))
            }
            ResidentLeaseStorage::Device(arrays) => {
                Box::new(arrays.arrays.keys().map(String::as_str))
            }
        }
    }
}

/// Structured failures from residency validation and state transitions.
#[derive(Debug, thiserror::Error)]
pub enum ResidencyError {
    /// Conversion registration or selection conflicts with its shared authority.
    #[error(transparent)]
    ConversionRetention(#[from] eredu_core::residency::conversion_claim::ConversionRetentionError),
    /// A backend resource observation violates its neutral accounting contract.
    #[error(transparent)]
    ResourceObservation(#[from] eredu_core::resources::ResourceDescriptionError),
    /// A backend-neutral binding or offload-unit declaration was invalid.
    #[error(transparent)]
    Declaration(#[from] eredu_core::residency::parameter_binding::ResidencyDeclarationError),
    /// Backend-neutral ownership or capacity transition failed.
    #[error("{0}")]
    Ledger(#[from] ResidencyLedgerError),
    /// Backend-neutral binding selection rewrite failed.
    #[error(transparent)]
    BindingSelection(#[from] eredu_core::residency::parameter_binding::WeightBindingSelectionError),
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
    #[error("binding {binding:?} in unit {id} selects {actual_bytes} bytes but declares {expected_bytes}")]
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
    /// Failure reported by the owner of the residency operation.
    #[error("{0}")]
    Operation(#[from] eredu_core::BackendFailure),
    /// Native storage no longer matches its admitted slots.
    #[error("residency manager state is poisoned")]
    StatePoisoned,
}

/// Native resources shared by this residency installation.
#[doc(hidden)]
#[derive(Clone)]
pub struct MlxResidencyResources {
    failed_transfer: Arc<AtomicBool>,
    resource_identity: eredu_core::resources::ResourceIdentity,
}

impl NativeResidencyStorage {
    // MLX's ungrouped learned RMS output uses result_type(input, gain). A
    // rank-one F32 gain therefore guarantees F32 for supported floating inputs.
    // This reports representation, not whether a family uses an array as a gain;
    // that association and all downstream dtype preservation belong to topology.
    fn f32_rms_normalization_gains(
        &self,
        bindings: &ResidencyObservationBindings<'_>,
    ) -> BTreeSet<String> {
        let mut gains = BTreeSet::new();
        let mut conflicts = BTreeSet::new();
        for (unit_id, storage) in &self.storage {
            let Some(unit) = &storage.device else {
                continue;
            };
            for (name, array) in &unit.arrays {
                let logical = bindings
                    .logical_target(unit_id, name)
                    .unwrap_or(name)
                    .to_owned();
                if array.ndim() == 1 && array.dtype() == safemlx::Dtype::Float32 {
                    gains.insert(logical);
                } else {
                    conflicts.insert(logical);
                }
            }
        }
        gains.retain(|name| !conflicts.contains(name));
        gains
    }

    fn projection_storage_observations(
        &self,
        bindings: &ResidencyObservationBindings<'_>,
    ) -> Result<
        BTreeMap<String, eredu_core::residency::projection::ProjectionStorageFacts>,
        ResidencyError,
    > {
        let mut facts = BTreeMap::new();
        if !bindings.projection_storage_eligible || !self.conversion_retention_registered {
            return Ok(facts);
        }
        let mut conflicts = BTreeSet::new();
        for (unit_id, storage) in &self.storage {
            let Some(unit) = &storage.device else {
                continue;
            };
            for (name, array) in &unit.arrays {
                let logical = bindings
                    .logical_target(unit_id, name)
                    .unwrap_or(name)
                    .to_owned();
                let Some(fact) =
                    crate::backend::nn::mixed_projection::storage_facts(array, &self.device_stream)
                        .map_err(|source| ResidencyError::Mlx {
                            id: unit_id.clone(),
                            operation: "projection storage facts",
                            source,
                        })?
                else {
                    conflicts.insert(logical);
                    continue;
                };
                if facts.get(&logical).is_some_and(|prior| prior != &fact) {
                    conflicts.insert(logical.clone());
                }
                facts.insert(logical, fact);
            }
        }
        for name in conflicts {
            facts.remove(&name);
        }
        Ok(facts)
    }

    /// Observe existing cache entries without evaluating or creating conversions.
    fn parameter_conversion_observations(
        &self,
        bindings: &ResidencyObservationBindings<'_>,
    ) -> Vec<eredu_core::residency::ResidentParameterConversion> {
        use eredu_core::residency::{
            ResidentParameterConversion, ResidentParameterConversionBinding,
        };
        use eredu_core::{
            resources::{
                ResourceAllocation, ResourceByteBounds, ResourceExtent, ResourceRole, ResourceSize,
                ResourceUse,
            },
            Observed,
        };

        let mut observations = BTreeMap::new();
        let retention_group = bindings.retention_group.cloned();
        for (unit_id, storage) in &self.storage {
            let Some(unit) = storage.device.as_ref() else {
                continue;
            };
            let cache = unit
                .parameter_conversions
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            let owner = cache.owner_identity();
            let conversions = cache.resident_conversions();
            for (name, array) in &unit.arrays {
                let Some(conversion) = conversions.get(&array.graph_identity()) else {
                    continue;
                };
                let observation = observations.entry(conversion.allocation.clone())
                    .or_insert_with(|| ResidentParameterConversion {
                        allocation: ResourceAllocation {
                            identity: conversion.allocation.clone(),
                            uses: Vec::new(),
                            placement: Observed::Unavailable {
                                reason: "native conversion cache does not identify a physical capacity pool".into(),
                            },
                            size: ResourceSize::Fixed { extent: ResourceExtent {
                                payload: ResourceByteBounds::exact(conversion.payload_bytes),
                                capacity: ResourceByteBounds::unknown(conversion.payload_bytes,
                                    "MLX does not expose cached conversion backing capacity"),
                            } },
                        },
                        bindings: Vec::new(),
                    });
                let usage = ResourceUse {
                    owner: owner.clone(),
                    role: ResourceRole::Parameters,
                };
                if !observation.allocation.uses.contains(&usage) {
                    observation.allocation.uses.push(usage);
                }
                let logical_target = bindings.logical_target(unit_id, name).map(str::to_owned);
                observation
                    .bindings
                    .push(ResidentParameterConversionBinding {
                        retention_group: retention_group.clone(),
                        owner: owner.clone(),
                        unit: unit_id.clone(),
                        name: name.clone(),
                        logical_target,
                    });
            }
        }
        observations.into_values().collect()
    }

    /// Observes currently cached host allocations under one storage lock. No
    /// payload reads, evaluation, completion polling or native allocation occur.
    /// Eviction removes cache ownership, not independently retained transfers.
    fn host_storage_resources(
        &self,
        identity: eredu_core::resources::ResourceIdentity,
    ) -> Result<eredu_core::resources::ResourceDescription, ResidencyError> {
        host_resources::describe(
            identity,
            self.storage.values().filter_map(|s| s.host.as_deref()),
            None,
        )
    }
}

impl eredu_core::residency::ResidencyStorageMechanisms for MlxResidencyResources {
    type Context = (Stream, Stream);
    type Storage = NativeResidencyStorage;
    type Error = ResidencyError;
    type Transfer = ResidentTransfer;

    fn register_conversion_resources(
        storage: &mut NativeResidencyStorage,
        budget: &eredu_core::residency::conversion_claim::ConversionRetentionBudget,
    ) -> Result<(), ResidencyError> {
        storage.conversion_retention_registered = true;
        for unit in storage
            .storage
            .values()
            .filter_map(|unit| unit.device.as_ref())
        {
            unit.parameter_conversions
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .register(unit.arrays.values(), budget)?;
        }
        Ok(())
    }

    fn trim_conversion_resources(
        budget: &eredu_core::residency::conversion_claim::ConversionRetentionBudget,
    ) -> Result<
        eredu_core::residency::ParameterConversionRetentionTrimReport,
        eredu_core::residency::conversion_claim::ConversionRetentionError,
    > {
        crate::backend::nn::parameter_conversion::trim_budget(budget)
    }

    fn open_storage(
        (source_stream, device_stream): Self::Context,
    ) -> Result<(NativeResidencyStorage, Self), ResidencyError> {
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

        let failed_transfer = Arc::new(AtomicBool::new(false));
        let storage = NativeResidencyStorage {
            failed_transfer: Arc::clone(&failed_transfer),
            conversion_retention_registered: false,
            storage: BTreeMap::new(),
            materialization: MlxParameterMaterializationContext::new(
                &source_stream,
                &device_stream,
            ),
            source_stream,
            device_stream,
        };
        Ok((
            storage,
            Self {
                failed_transfer,
                resource_identity: host_resources::identity("mlx.residency_manager"),
            },
        ))
    }

    fn progress(&self) -> Result<(), ResidencyError> {
        crate::backend::submission_recovery::reap();
        crate::backend::ordinary_retirement::reclaim();
        if self.failed_transfer.load(Ordering::Acquire) {
            return Err(ResidencyError::Mlx {
                id: internal_id(),
                operation: "resident transfer admission",
                source: safemlx::error::Exception::custom(
                    "residency manager is poisoned by a failed or unobservable native transfer",
                ),
            });
        }
        Ok(())
    }

    fn transfer(
        leases: Vec<ResidentUnitLease>,
        submitted: Option<SubmittedResidentTransfer>,
    ) -> ResidentTransfer {
        match submitted {
            Some(submitted) => ResidentTransfer::submitted(leases, submitted),
            None => ResidentTransfer::immediate(leases),
        }
    }
}

impl eredu_core::residency::observations::DenseResidencyMechanisms for MlxResidencyResources {
    /// Returns the MLX stream index used for device residency transfers.
    fn transfer_stream_index(storage: &NativeResidencyStorage) -> Result<i32, ResidencyError> {
        storage
            .device_stream
            .get_index()
            .map_err(|source| ResidencyError::Mlx {
                id: internal_id(),
                operation: "device residency stream index",
                source,
            })
    }

    fn settle_transfer(transfer: &mut ResidentTransfer) -> Result<(), ResidencyError> {
        transfer.synchronize()
    }

    fn allocator_memory(
        &self,
    ) -> Result<eredu_core::residency::AllocatorMemoryMetrics, ResidencyError> {
        sample_allocator_memory().map_err(|source| ResidencyError::Mlx {
            id: internal_id(),
            operation: "allocator memory sampling",
            source,
        })
    }

    fn observe_residency(
        &self,
        storage: &NativeResidencyStorage,
        bindings: ResidencyObservationBindings<'_>,
    ) -> Result<eredu_core::residency::observations::NativeResidencyObservations, ResidencyError>
    {
        Ok(
            eredu_core::residency::observations::NativeResidencyObservations {
                host_storage: storage.host_storage_resources(self.resource_identity.clone())?,
                conversions: storage.parameter_conversion_observations(&bindings),
                projection_storage: storage.projection_storage_observations(&bindings)?,
                f32_normalization_gains: storage.f32_rms_normalization_gains(&bindings),
            },
        )
    }

    fn observe_detached(
        &self,
        limits: eredu_core::resources::detached::DetachedResourceLimits,
    ) -> Result<eredu_core::resources::detached::DetachedResourceObservation, ResidencyError> {
        Ok(crate::backend::submission_recovery::describe_resources(
            limits,
        )?)
    }
}

mod transfer;
use transfer::*;
pub use transfer::{
    ResidentArrays, ResidentHostBuffers, ResidentTransfer, ResidentTransferResources,
};

mod materialization;
pub use materialization::host_capacity_upper_bound_for_bindings;
use materialization::*;

mod host_resources;
use host_resources::ResidentHostAllocation;
