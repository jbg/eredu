//! Ordinary acquisition destinations paid by the enclosing request's host account.
//! Source geometry describes the existing controller worker; it grants no native
//! observer, transfer completion, or model materialization authority.

use super::*;
#[path = "host_acquisition/transfer_source.rs"]
mod transfer_source;
use eredu_nn::workspace::{HostMetadataFunding, HostMetadataFundingError};
use eredu_runtime::residency::ResidencyClosureSlot;
use eredu_runtime::working_memory::MemoryLedger;
use std::{
    alloc::Layout,
    collections::TryReserveError,
    mem::{size_of, size_of_val},
};

/// Additional allocations from the actual transformed foreground producers.
/// `native == None` is an authenticated empty population, not an unknown.
pub(crate) struct OrdinaryMaterializedRequirements {
    pub(crate) native: Option<eredu_core::DomainMemoryRequirements>,
    pub(crate) metadata: usize,
    _funding: HostMetadataFunding,
}

impl ResidencyManager {
    /// Scratch for inspecting this manager's immutable canonical closure.
    /// Absence means this manager owns no foreground materialization producer.
    pub(crate) fn ordinary_materialization_scratch_len(
        &self,
        funding: &HostMetadataFunding,
    ) -> Result<Option<usize>, ResidencyError> {
        funding
            .reserve_metadata(size_of::<(
                &Self,
                &HostMetadataFunding,
                Option<usize>,
                Result<Option<usize>, ResidencyError>,
                std::sync::MutexGuard<'_, ManagerState>,
                std::sync::TryLockResult<std::sync::MutexGuard<'_, ManagerState>>,
            )>())
            .map_err(ResidencyError::HostMetadataFunding)?;
        if self.inner.sources.foreground().is_none() {
            return Ok(None);
        }
        let state = self.inner.state.try_lock().map_err(|cause| match cause {
            std::sync::TryLockError::WouldBlock => ResidencyError::OriginalManagerBusy,
            std::sync::TryLockError::Poisoned(_) => ResidencyError::StatePoisoned,
        })?;
        Ok(Some(state.control.ledger().plan().units().len()))
    }

    /// Sum the source requirements of the actual closure. Each selected recipe
    /// may execute once in this attempt; no static source pin is charged again.
    pub(crate) fn ordinary_foreground_materialization_requirements(
        &self,
        roots: &[OffloadUnitId],
        scratch: &mut [ResidencyClosureSlot],
        pool: &MemoryLedger,
        runtime: &safemlx::PreparedInputRuntime,
        allocation: crate::backend::nn::workspace::NativeAllocationFacts,
        funding: &HostMetadataFunding,
    ) -> Result<Option<OrdinaryMaterializedRequirements>, ResidencyError> {
        self.inner
            .validate_pool(pool)
            .map_err(ResidencyError::OriginalCache)?;
        let Some(source) = self.inner.sources.foreground() else {
            return Ok(None);
        };
        funding
            .reserve_metadata(size_of::<(
                OrdinaryMaterializedRequirements,
                Option<OrdinaryMaterializedRequirements>,
                Result<Option<OrdinaryMaterializedRequirements>, ResidencyError>,
                &Self,
                &[OffloadUnitId],
                &mut [ResidencyClosureSlot],
                &MemoryLedger,
                &safemlx::PreparedInputRuntime,
                crate::backend::nn::workspace::NativeAllocationFacts,
                &HostMetadataFunding,
            )>())
            .map_err(ResidencyError::HostMetadataFunding)?;
        let state = self.inner.state.try_lock().map_err(|cause| match cause {
            std::sync::TryLockError::WouldBlock => ResidencyError::OriginalManagerBusy,
            std::sync::TryLockError::Poisoned(_) => ResidencyError::StatePoisoned,
        })?;
        let closure = state
            .control
            .operation_closure(roots, scratch)
            .map_err(ResidencyError::OperationClosure)?;
        let reports = eredu_runtime::working_memory::WorkspaceReportMetadata::with_funding(funding);
        let mut result = OrdinaryMaterializedRequirements {
            native: None,
            metadata: 0,
            _funding: funding.clone(),
        };
        for unit in closure.units() {
            let Some(quote) = source.ordinary_materialized_quote(
                unit.id(),
                pool,
                runtime,
                allocation,
                state.materialization.view().source_stream(),
                funding,
            )?
            else {
                continue;
            };
            result.native = Some(match result.native.take() {
                None => quote.requirements,
                Some(previous) => reports
                    .combine_domain_requirements(&previous, &quote.requirements, true)
                    .map_err(|cause| {
                        ResidencyError::OriginalSourceRecipe(WeightRecipeError::Workspace(
                            reports.error(cause),
                        ))
                    })?,
            });
            result.metadata = result.metadata.checked_add(quote.metadata).ok_or(
                ResidencyError::HostMetadataFunding(HostMetadataFundingError::Overflow),
            )?;
        }
        Ok(Some(result))
    }
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum HostAcquisitionCause {
    #[error("ordinary residency destination geometry overflowed")]
    Overflow,
    #[error("ordinary residency destination funding: {0}")]
    Funding(#[source] HostMetadataFundingError),
    #[error("ordinary residency destination allocation: {0}")]
    Allocation(#[source] TryReserveError),
    #[error("ordinary residency source: {0}")]
    Source(#[source] OperationSourceFailure),
    #[error("ordinary residency controller destination: {0}")]
    Controller(#[source] controller_attempt::ControllerPreparationCause),
    #[error("ordinary residency closure destination: {0}")]
    Closure(#[source] closure_ids::ClosurePreparationCause),
    #[error("ordinary residency lease destination: {0}")]
    Lease(#[source] NamedArrayError),
    #[error("ordinary residency named destination: {0}")]
    Named(#[source] named_arrays::NamedPreparationSource),
    #[error("ordinary residency recovery scope: {0}")]
    Scope(#[source] safemlx::SubmissionScopeOwnerCause),
    #[error("ordinary residency acquisition: {0}")]
    Acquisition(#[source] ResidencyError),
}

/// The error allocation and any retained dynamic source retire before its payer.
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub struct HostAcquisitionFailure {
    #[source]
    cause: Box<HostAcquisitionCause>,
    funding: HostMetadataFunding,
}

impl HostAcquisitionFailure {
    pub(super) fn new(cause: HostAcquisitionCause, funding: &HostMetadataFunding) -> Self {
        Self {
            cause: Box::new(cause),
            funding: funding.clone(),
        }
    }

    pub(super) fn control_bytes() -> Option<usize> {
        let frames = [
            Layout::new::<HostAcquisitionCause>().size(),
            size_of::<Self>(),
            size_of::<HostAcquisitionCause>(),
            size_of::<Box<HostAcquisitionCause>>(),
            size_of::<Result<(), Self>>(),
            HostMetadataFunding::reservation_control_bytes(),
        ];
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
    }
}

/// Prepared host controls remain separate from the original-operation slots.
/// Both modes feed the same controller, closure and final lease destinations.
pub(super) struct PreparedHostAcquisition {
    pub(super) population: WindowPopulation,
    pub(super) scratch: Vec<ResidencyClosureSlot>,
    pub(super) controller: Option<PreparedControllerAttempt>,
    pub(super) closure: Option<closure_ids::PreparedClosureIds>,
    pub(super) leases: Option<transfer::PreparedLeaseCollection>,
    pub(super) transfer: Option<transfer::PreparedHostTransfer>,
    pub(super) funding: HostMetadataFunding,
}

impl PreparedHostAcquisition {
    /// These are only the shared acquisition-control destinations. Transfer
    /// recovery and materialization storage have their own producer census.
    pub(super) fn destination_bytes(
        population: WindowPopulation,
        tier: MemoryTier,
    ) -> Option<usize> {
        let shape = TransferPayloadShape::window(population)?;
        let controls = [
            controller_attempt::PreparedControllerAttempt::control_bytes(
                population.controller_units,
                population.requested,
                population.units,
                population.maximum_id_bytes,
            )?,
            closure_ids::PreparedClosureIds::control_bytes(
                population.units,
                population.unit_id_bytes,
            )?,
            transfer::PreparedLeaseCollection::control_bytes(shape)?,
            ResidentHostOwner::storage_bytes()
                .ok()?
                .checked_mul(u64::try_from(population.units).ok()?)?,
        ];
        let bytes = controls.into_iter().try_fold(0usize, |n, value| {
            n.checked_add(usize::try_from(value).ok()?)
        })?;
        if tier == MemoryTier::Host {
            bytes
                .checked_add(
                    Layout::array::<(OffloadUnitId, ResidentHostBuffers, u64, u64, u64)>(
                        population.units,
                    )
                    .ok()?
                    .size(),
                )?
                .checked_add(population.unit_id_bytes)
        } else {
            Some(bytes)
        }
    }

    pub(super) fn source_bytes(units: usize, requests: usize, id_bytes: usize) -> Option<usize> {
        let frames = [
            Layout::array::<ResidencyClosureSlot>(units).ok()?.size(),
            Layout::array::<OffloadUnitId>(requests).ok()?.size(),
            id_bytes,
            size_of::<Self>(),
            size_of::<Vec<OffloadUnitId>>(),
            size_of::<Result<Self, HostAcquisitionFailure>>(),
            size_of::<(
                &ResidencyManager,
                &[(OffloadUnitId, u64)],
                MemoryTier,
                &HostMetadataFunding,
            )>(),
            size_of::<std::sync::MutexGuard<'static, ManagerState>>(),
            size_of::<Result<std::sync::MutexGuard<'static, ManagerState>, ResidencyError>>(),
            size_of::<WindowPopulation>(),
            HostAcquisitionFailure::control_bytes()?,
        ];
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
    }

    /// The caller pays the source constructor before entering this method.
    /// It then debits the actual selected destinations before allocating them.
    pub(super) fn prepare_paid(
        manager: &ResidencyManager,
        requests: &[(OffloadUnitId, u64)],
        tier: MemoryTier,
        units: usize,
        funding: &HostMetadataFunding,
    ) -> Result<Self, HostAcquisitionFailure> {
        let fail = |cause| HostAcquisitionFailure::new(cause, funding);
        let mut roots = Vec::new();
        roots
            .try_reserve_exact(requests.len())
            .map_err(|e| fail(HostAcquisitionCause::Allocation(e)))?;
        for (id, _) in requests {
            roots.push(id.clone());
        }
        let mut ready = Self {
            population: WindowPopulation::default(),
            scratch: Vec::new(),
            controller: None,
            closure: None,
            leases: None,
            transfer: None,
            funding: funding.clone(),
        };
        ready
            .scratch
            .try_reserve_exact(units)
            .map_err(|e| fail(HostAcquisitionCause::Allocation(e)))?;
        ready.scratch.resize(units, ResidencyClosureSlot::default());
        ready.population = manager
            .selected_root_population(&roots, &mut ready.scratch)
            .map_err(|e| fail(HostAcquisitionCause::Source(e)))?;
        if ready.population.controller_units != units {
            return Err(fail(HostAcquisitionCause::Source(
                OperationSourceFailure::Layout,
            )));
        }
        let bytes = Self::destination_bytes(ready.population, tier)
            .ok_or_else(|| fail(HostAcquisitionCause::Overflow))?;
        funding
            .reserve_metadata(bytes)
            .map_err(|e| fail(HostAcquisitionCause::Funding(e)))?;
        ready.controller = Some(
            manager
                .prepare_controller_with_custody(ready.population, funding.clone().into())
                .map_err(|error| fail(HostAcquisitionCause::Controller(error.cause)))?,
        );
        ready.closure = Some(
            manager
                .prepare_closure_ids_with_custody(
                    &roots,
                    &mut ready.scratch,
                    ready.population.units,
                    ready.population.unit_id_bytes,
                    funding.clone().into(),
                )
                .map_err(|error| fail(HostAcquisitionCause::Closure(error.cause)))?,
        );
        let shape = TransferPayloadShape::window(ready.population)
            .ok_or_else(|| fail(HostAcquisitionCause::Overflow))?;
        ready.leases = Some(
            transfer::PreparedLeaseCollection::try_new_with_custody(
                manager,
                &roots,
                shape,
                tier,
                funding.clone().into(),
            )
            .map_err(|error| {
                fail(match error.cause {
                    transfer::LeasePreparationCause::Source(cause) => {
                        HostAcquisitionCause::Lease(cause)
                    }
                    transfer::LeasePreparationCause::Reserve(cause) => {
                        HostAcquisitionCause::Allocation(cause)
                    }
                })
            })?,
        );
        if tier == MemoryTier::Device {
            ready.transfer = Some(transfer::PreparedHostTransfer::prepare(
                manager,
                &roots,
                &mut ready.scratch,
                ready.population,
                funding,
            )?);
        }
        Ok(ready)
    }
}

impl ResidencyManager {
    /// New named rows for a Host tier publication over exact retained backups.
    /// Physical buffers already belong to the source and are not recharged.
    pub(crate) fn ordinary_retained_host_control_bytes(
        &self,
        roots: &[OffloadUnitId],
        scratch: &mut [ResidencyClosureSlot],
    ) -> Result<Option<usize>, ResidencyError> {
        if self.inner.sources.foreground().is_some() {
            return Ok(None);
        }
        let state = self.inner.state.try_lock().map_err(|cause| match cause {
            std::sync::TryLockError::WouldBlock => ResidencyError::OriginalManagerBusy,
            std::sync::TryLockError::Poisoned(_) => ResidencyError::StatePoisoned,
        })?;
        let closure = state
            .control
            .operation_closure(roots, scratch)
            .map_err(ResidencyError::OperationClosure)?;
        let result = closure.units().try_fold(Some(0usize), |total, unit| {
            let Some(total) = total else {
                return Ok(None);
            };
            if self.inner.sources.prepared_host(unit.id()).is_none() {
                return Ok(None);
            }
            transfer::retained_host_source_control_bytes(unit)
                .and_then(|bytes| total.checked_add(bytes))
                .map(Some)
                .ok_or(ResidencyError::ArithmeticOverflow {
                    context: "ordinary retained Host controls",
                })
        });
        result
    }

    /// Simultaneous owned foreground destinations. Their already-retained file
    /// source and any existing resident copies are not charged here.
    pub(crate) fn ordinary_foreground_backing_capacity(
        &self,
        roots: &[OffloadUnitId],
        scratch: &mut [ResidencyClosureSlot],
        runtime: &safemlx::PreparedInputRuntime,
    ) -> Result<Option<u64>, ResidencyError> {
        let Some(source) = self.inner.sources.foreground() else {
            return Ok(None);
        };
        let state = self.inner.state.try_lock().map_err(|cause| match cause {
            std::sync::TryLockError::WouldBlock => ResidencyError::OriginalManagerBusy,
            std::sync::TryLockError::Poisoned(_) => ResidencyError::StatePoisoned,
        })?;
        let closure = state
            .control
            .operation_closure(roots, scratch)
            .map_err(ResidencyError::OperationClosure)?;
        let result = closure
            .units()
            .try_fold(0u64, |total, unit| {
                total
                    .checked_add(source.ordinary_backing_capacity(unit.id(), runtime)?)
                    .ok_or(ResidencyError::ArithmeticOverflow {
                        context: "ordinary foreground backing sum",
                    })
            })
            .map(Some);
        result
    }

    /// Per-binding ordinary allocator capacity for the exact canonical copy
    /// worker. Aliases reuse those destinations and add no fresh backing.
    pub(crate) fn ordinary_copy_destination_capacity(
        &self,
        roots: &[OffloadUnitId],
        scratch: &mut [ResidencyClosureSlot],
        allocation: crate::backend::nn::workspace::NativeAllocationFacts,
    ) -> Result<u64, ResidencyError> {
        let state = self.inner.state.try_lock().map_err(|cause| match cause {
            std::sync::TryLockError::WouldBlock => ResidencyError::OriginalManagerBusy,
            std::sync::TryLockError::Poisoned(_) => ResidencyError::StatePoisoned,
        })?;
        let closure = state
            .control
            .operation_closure(roots, scratch)
            .map_err(ResidencyError::OperationClosure)?;
        let result = closure.units().try_fold(0u64, |total, unit| {
            unit.bindings()
                .iter()
                .filter(|binding| !binding.is_alias())
                .try_fold(total, |sum, binding| {
                    let bytes = allocation
                        .fixed_buffer_capacity(binding.expected_bytes())
                        .map_err(|_| ResidencyError::ArithmeticOverflow {
                            context: "ordinary copy backing capacity",
                        })?;
                    sum.checked_add(bytes)
                        .ok_or(ResidencyError::ArithmeticOverflow {
                            context: "ordinary copy backing sum",
                        })
                })
        });
        result
    }

    /// Independently observed native host controls of the same retained reads.
    /// Physical backing, native copies and dispatch remain separate facts.
    pub(crate) fn ordinary_foreground_observed_control_bytes(
        &self,
        roots: &[OffloadUnitId],
        scratch: &mut [ResidencyClosureSlot],
        runtime: &safemlx::PreparedInputRuntime,
    ) -> Result<Option<usize>, ResidencyError> {
        let Some(source) = self.inner.sources.foreground() else {
            return Ok(None);
        };
        let state = self.inner.state.try_lock().map_err(|cause| match cause {
            std::sync::TryLockError::WouldBlock => ResidencyError::OriginalManagerBusy,
            std::sync::TryLockError::Poisoned(_) => ResidencyError::StatePoisoned,
        })?;
        let closure = state
            .control
            .operation_closure(roots, scratch)
            .map_err(ResidencyError::OperationClosure)?;
        let result = closure
            .units()
            .try_fold(0usize, |total, unit| {
                total
                    .checked_add(source.ordinary_observed_control_bytes(unit.id(), runtime)?)
                    .ok_or(ResidencyError::HostMetadataFunding(
                        HostMetadataFundingError::Overflow,
                    ))
            })
            .map(Some);
        result
    }

    /// Host allocations of the retained foreground reader for the actual
    /// canonical closure. A nonforeground source returns `None`; its recipe
    /// worker needs its own facts. Native buffers, copies and completion are
    /// separate contributors even when this reader is qualified.
    pub(crate) fn ordinary_foreground_read_control_bytes(
        &self,
        roots: &[OffloadUnitId],
        scratch: &mut [ResidencyClosureSlot],
    ) -> Result<Option<usize>, ResidencyError> {
        let Some(source) = self.inner.sources.foreground() else {
            return Ok(None);
        };
        let state = self.inner.state.try_lock().map_err(|cause| match cause {
            std::sync::TryLockError::WouldBlock => ResidencyError::OriginalManagerBusy,
            std::sync::TryLockError::Poisoned(_) => ResidencyError::StatePoisoned,
        })?;
        let closure = state
            .control
            .operation_closure(roots, scratch)
            .map_err(ResidencyError::OperationClosure)?;
        let result = closure
            .units()
            .try_fold(0usize, |total, unit| {
                total
                    .checked_add(source.ordinary_read_control_bytes(unit.id())?)
                    .ok_or(ResidencyError::HostMetadataFunding(
                        HostMetadataFundingError::Overflow,
                    ))
            })
            .map(Some);
        result
    }

    /// Descriptive constructor requests from the actual canonical closure.
    /// Covers source scratch, controller/closure/lease destinations, the warm
    /// application, and Device named/publication/recovery storage. Recipe/read
    /// preparation, numerical copy and completion remain separate contributors.
    /// The caller owns the root slice and topology scratch; no grant is issued.
    pub(crate) fn ordinary_acquisition_control_bytes(
        &self,
        roots: &[OffloadUnitId],
        scratch: &mut [ResidencyClosureSlot],
        tier: MemoryTier,
    ) -> Result<usize, OperationSourceFailure> {
        if !matches!(tier, MemoryTier::Host | MemoryTier::Device) {
            return Err(OperationSourceFailure::Layout);
        }
        let population = self.selected_root_population(roots, scratch)?;
        let amount = PreparedHostAcquisition::source_bytes(
            population.controller_units,
            population.requested,
            population.requested_id_bytes,
        )
        .and_then(|n| {
            n.checked_add(PreparedHostAcquisition::destination_bytes(
                population, tier,
            )?)
        })
        .and_then(|n| n.checked_add(ResidentTransfer::host_immediate_control_bytes()?))
        .and_then(|n| match tier {
            MemoryTier::Device => {
                n.checked_add(transfer::PreparedHostTransfer::control_bytes(population)?)
            }
            MemoryTier::Host => Some(n),
            MemoryTier::Disk => None,
        });
        amount.ok_or(OperationSourceFailure::Overflow)
    }

    /// The same C copy/retention/completion wrappers prepaid by the ordinary
    /// Device transfer constructor. These are separate from shared controller
    /// destinations and from physically observed native graph controls.
    pub(crate) fn ordinary_transfer_wrapper_control_bytes(
        population: WindowPopulation,
    ) -> Option<usize> {
        transfer::PreparedHostTransfer::native_wrapper_control_bytes(population)?
            .checked_add(ResidentTransfer::host_consumer_control_bytes()?)
    }

    /// Uses the ordinary transfer worker with source-bound, paid controller and
    /// lease storage. Transfer/materialization contributions remain independent
    /// requirements; this entry point alone does not certify their completeness.
    pub(crate) fn acquire_many_with_host_transfer(
        &self,
        requests: &[(OffloadUnitId, u64)],
        tier: MemoryTier,
        funding: &HostMetadataFunding,
    ) -> Result<ResidentTransfer, ResidencyError> {
        let units = self.lock()?.control.units().len();
        let overflow = || ResidencyError::HostMetadataFunding(HostMetadataFundingError::Overflow);
        let id_bytes = requests
            .iter()
            .try_fold(0usize, |n, (id, _)| n.checked_add(id.as_str().len()))
            .ok_or_else(overflow)?;
        let bytes = PreparedHostAcquisition::source_bytes(units, requests.len(), id_bytes)
            .and_then(|n| n.checked_add(ResidentTransfer::host_immediate_control_bytes()?))
            .ok_or_else(overflow)?;
        funding
            .reserve_metadata(bytes)
            .map_err(ResidencyError::HostMetadataFunding)?;
        let mut ready =
            PreparedHostAcquisition::prepare_paid(self, requests, tier, units, funding)?;
        let leases = ready.leases.take().expect("unissued paid lease collection");
        let result = self.acquire_many_with_operations(
            requests,
            tier,
            true,
            None,
            Some(leases),
            Some(&mut ready),
        );
        let (leases, submitted) = result.map_err(|cause| {
            HostAcquisitionFailure::new(HostAcquisitionCause::Acquisition(cause), funding)
        })?;
        Ok(match submitted {
            Some(submitted) => ResidentTransfer::submitted(leases, submitted),
            None => ResidentTransfer::immediate_host_paid(leases, tier, self),
        })
    }
}
