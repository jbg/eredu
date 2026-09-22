//! Ordinary acquisition facts from the retained selected-member source.
use super::*;
use crate::backend::nn::workspace::OrdinaryNativeControls;
use crate::backend::runtime::residency::manager::{SupplementaryResidencySource, WindowPopulation};
use eredu_nn::workspace::WorkspaceMetadataAllocation;
use eredu_runtime::residency::ResidencyClosureSlot;

/// Source-specific contributors which have not supplied a finite ordinary census.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OrdinaryResidencyMissing {
    /// The source can invoke retained recipes instead of foreground reads.
    RecipeMaterialization,
    /// Ordinary graph construction, evaluation and dispatch for the host copy.
    TransferExecution,
    /// A selected staging or destination allocator has no established placement.
    DestinationPlacement,
}

/// One separately placed source category; simultaneous rows add by domain.
#[derive(Clone, Copy, Debug)]
pub(crate) struct OrdinaryResidencyPhysical {
    pub(crate) bytes: u64,
    pub(crate) placement: &'static eredu_core::MemoryPlacement,
    /// Maximum fresh backing owners in this selected chunk window.
    pub(crate) allocations: usize,
}

/// Descriptive maximum member-window facts. Observed native controls and the
/// request's host metadata are disjoint; neither grants execution authority.
pub(crate) struct OrdinaryIndexedResidencyFacts {
    binding: IndexedBankSource,
    source: SupplementaryResidencySource,
    first: AddressableChunkCensus,
    revision: u64,
    window: WindowPopulation,
    manager: usize,
    reader: Option<usize>,
    observed_host: Option<usize>,
    copy_wrappers: usize,
    transfer: Option<OrdinaryNativeControls>,
    host_staging: Option<OrdinaryResidencyPhysical>,
    destination: Option<OrdinaryResidencyPhysical>,
    source_pins: Option<u64>,
    funding: HostMetadataFunding,
}
impl OrdinaryIndexedResidencyFacts {
    pub(crate) fn first(&self) -> AddressableChunkCensus {
        self.first
    }
    pub(crate) fn parameter_revision(&self) -> u64 {
        self.revision
    }
    pub(crate) fn manager_control_bytes(&self) -> usize {
        self.manager
    }
    pub(crate) fn foreground_read_control_bytes(&self) -> Option<usize> {
        self.reader
    }
    pub(crate) fn observed_host_control_bytes(&self) -> Option<usize> {
        self.observed_host
    }
    pub(crate) fn copy_wrapper_control_bytes(&self) -> usize {
        self.copy_wrappers
    }
    /// Observed graph/Eval/dispatch/wait allocations, with their actual source
    /// population. The metadata payer and host-buffer constructor are separate.
    pub(crate) fn transfer_native_controls(&self) -> Option<OrdinaryNativeControls> {
        self.transfer
    }
    pub(crate) fn host_staging(&self) -> Option<OrdinaryResidencyPhysical> {
        self.host_staging
    }
    pub(crate) fn destination(&self) -> Option<OrdinaryResidencyPhysical> {
        self.destination
    }
    /// One installation per invocation; do not multiply by selected chunks.
    pub(crate) fn source_pin_control_bytes(&self) -> Option<u64> {
        self.source_pins
    }
    pub(crate) fn prepare_source_pins(
        &self,
        funding: &HostMetadataFunding,
    ) -> Result<Option<crate::backend::runtime::residency::manager::HostCopySourcePins>, Error>
    {
        let fail = |cause| failed(cause, &self.binding.bank, funding, None);
        self.binding
            .bank
            .with_workspace_source(funding, |bank| {
                if !bank.same_source(&self.binding.bank)
                    || bank.parameter_revision() != self.revision
                {
                    return Err(fail(Cause::Identity));
                }
                match self.source.host() {
                    Some(host) => host
                        .pin_retained_sources_with_host(bank.manager(), funding)
                        .map(Some)
                        .map_err(|e| fail(Cause::Host(e))),
                    None => Ok(None),
                }
            })
            .map_err(|e| fail(Cause::Bank(e)))?
    }
    /// Includes only the request metadata payer. Physical-observer controls are
    /// reported independently and must not be added to this payer a second time.
    pub(crate) fn host_control_bytes(&self) -> Option<usize> {
        self.transfer?;
        self.manager
            .checked_add(self.reader?)?
            .checked_add(self.copy_wrappers)
    }
    pub(crate) fn missing(&self) -> impl Iterator<Item = OrdinaryResidencyMissing> {
        [
            self.reader
                .is_none()
                .then_some(OrdinaryResidencyMissing::RecipeMaterialization),
            self.transfer
                .is_none()
                .then_some(OrdinaryResidencyMissing::TransferExecution),
            (self.destination.is_none() || self.host_staging.is_none())
                .then_some(OrdinaryResidencyMissing::DestinationPlacement),
        ]
        .into_iter()
        .flatten()
    }
    pub(crate) fn validate(
        &self,
        binding: &IndexedBankSource,
        first: AddressableChunkCensus,
    ) -> Result<(), Error> {
        let fail = |cause| failed(cause, &self.binding.bank, &self.funding, None);
        if first != self.first
            || (!Arc::ptr_eq(&binding.bank.inner, &self.binding.bank.inner)
                || binding.bank.scope != self.binding.bank.scope)
        {
            return Err(fail(Cause::Identity));
        }
        binding
            .bank
            .with_workspace_source(&self.funding, |bank| {
                if !bank.same_source(&self.binding.bank)
                    || bank.parameter_revision() != self.revision
                {
                    return Err(fail(Cause::Identity));
                }
                bank.manager()
                    .validate_supplementary_source(&self.source)
                    .map_err(|cause| fail(Cause::Source(cause)))
            })
            .map_err(|cause| fail(Cause::Bank(cause)))?
    }
}
impl IndexedBankSource {
    /// The sum of any selected k closures is bounded by both the complete
    /// eligible closure and k times its largest singleton. This uses the actual
    /// shared worker's control query; no original-operation constructor is priced.
    pub(crate) fn inspect_ordinary_residency(
        &self,
        first: AddressableChunkCensus,
        runtime: &PreparedInputRuntime,
        allocation: crate::backend::nn::workspace::NativeAllocationFacts,
        funding: &HostMetadataFunding,
    ) -> Result<OrdinaryIndexedResidencyFacts, Error> {
        let fail = |cause| failed(cause, &self.bank, funding, None);
        let frames = [
            size_of::<OrdinaryIndexedResidencyFacts>(),
            size_of::<Result<OrdinaryIndexedResidencyFacts, Error>>(),
            size_of::<(
                &Self,
                AddressableChunkCensus,
                &PreparedInputRuntime,
                crate::backend::nn::workspace::NativeAllocationFacts,
                &HostMetadataFunding,
            )>(),
            size_of::<(
                ResidencyManager,
                SupplementaryResidencySource,
                Vec<OffloadUnitId>,
                u64,
            )>(),
            size_of::<Vec<ResidencyClosureSlot>>(),
            size_of::<[usize; 6]>(),
            size_of::<[Option<usize>; 4]>(),
            size_of::<WindowPopulation>(),
            size_of::<Option<(safemlx::DeviceType, i32)>>(),
            size_of::<Option<&eredu_core::MemoryPlacement>>(),
            ResidencyManager::ordinary_transfer_inspection_control_bytes()
                .ok_or_else(|| fail(Cause::Overflow))?,
            AddressableChunkPlan::control_bytes(),
            eredu_nn::Error::retained_source_construction_bytes::<Failure>()
                .ok_or_else(|| fail(Cause::Overflow))?,
        ];
        funding
            .reserve_metadata(
                frames
                    .into_iter()
                    .try_fold(size_of_val(&frames), usize::checked_add)
                    .ok_or_else(|| fail(Cause::Overflow))?,
            )
            .map_err(|e| fail(Cause::Funding(e)))?;
        if first.index() != 0 {
            return Err(fail(Cause::Identity));
        }
        let (manager, source, eligible, revision) = self
            .bank
            .with_workspace_source(funding, |bank| {
                let (members, maximum) = bank
                    .unit_population(first.bank(), first.unit())
                    .ok_or_else(|| fail(Cause::Geometry))?;
                let expected = AddressableChunkPlan::new(
                    first.total_rows(),
                    first.routes(),
                    members,
                    first.access(),
                    Some(maximum),
                    self.options.prefill_compact_bank_target_bytes(),
                )
                .map_err(|_| fail(Cause::Geometry))?;
                if expected != first.plan() || !bank.same_source(&self.bank) {
                    return Err(fail(Cause::Identity));
                }
                let source = bank
                    .manager()
                    .supplementary_residency_source()
                    .ok_or_else(|| fail(Cause::Identity))?;
                let mut eligible = funding
                    .metadata_vec(members)
                    .map_err(|e| fail(Cause::Consumer(Error::Neural(e))))?;
                let ids = bank
                    .unit_members(first.bank(), first.unit())
                    .try_fold(0usize, |n, (key, _)| n.checked_add(key.unit_id_length()))
                    .ok_or_else(|| fail(Cause::Overflow))?;
                funding
                    .reserve_metadata(ids)
                    .map_err(|e| fail(Cause::Funding(e)))?;
                for (key, _) in bank.unit_members(first.bank(), first.unit()) {
                    eligible.push(key.unit_id());
                }
                if eligible.len() != members {
                    return Err(fail(Cause::Identity));
                }
                Ok::<_, Error>((
                    bank.manager().clone(),
                    source.clone(),
                    eligible,
                    bank.parameter_revision(),
                ))
            })
            .map_err(|e| fail(Cause::Bank(e)))??;
        let units = source.source().controller_units;
        let mut scratch = funding
            .metadata_vec(units)
            .map_err(|e| fail(Cause::Consumer(Error::Neural(e))))?;
        scratch.resize_with(units, ResidencyClosureSlot::default);
        let count = first.maximum_members();
        let window = manager
            .selected_operation_ceiling(&source, &eligible, count, &mut scratch)
            .map_err(|e| fail(Cause::Source(e)))?;
        let acquire =
            |ids: &[OffloadUnitId], scratch: &mut [ResidencyClosureSlot]| -> Result<usize, Error> {
                let host = manager
                    .ordinary_acquisition_control_bytes(ids, scratch, MemoryTier::Host)
                    .map_err(|e| fail(Cause::Source(e)))?;
                let device = manager
                    .ordinary_acquisition_control_bytes(ids, scratch, MemoryTier::Device)
                    .map_err(|e| fail(Cause::Source(e)))?;
                host.checked_add(device)
                    .ok_or_else(|| fail(Cause::Overflow))
            };
        let manager_full = acquire(&eligible, &mut scratch)?;
        let source_pins = source
            .host()
            .map(|host| {
                host.retained_pin_control_bytes()
                    .ok_or_else(|| fail(Cause::Overflow))
            })
            .transpose()?;
        if source_pins.is_some() {
            manager
                .validate_supplementary_host(&source)
                .map_err(|e| fail(Cause::Host(e)))?;
        }
        let reader_full = manager
            .ordinary_foreground_read_control_bytes(&eligible, &mut scratch)
            .map_err(|e| fail(Cause::Residency(e)))?
            .or(manager
                .ordinary_retained_host_control_bytes(&eligible, &mut scratch)
                .map_err(|e| fail(Cause::Residency(e)))?);
        let observed_full = manager
            .ordinary_foreground_observed_control_bytes(&eligible, &mut scratch, runtime)
            .map_err(|e| fail(Cause::Residency(e)))?;
        let destination_full = manager
            .ordinary_copy_destination_capacity(&eligible, &mut scratch, allocation)
            .map_err(|e| fail(Cause::Residency(e)))?;
        let staging_full = manager
            .ordinary_foreground_backing_capacity(&eligible, &mut scratch, runtime)
            .map_err(|e| fail(Cause::Residency(e)))?;
        let (mut manager_max, mut reader_max, mut observed_max) = (0usize, 0usize, 0usize);
        let (mut staging_max, mut destination_max) = (0u64, 0u64);
        for id in &eligible {
            let ids = std::slice::from_ref(id);
            manager_max = manager_max.max(acquire(ids, &mut scratch)?);
            destination_max = destination_max.max(
                manager
                    .ordinary_copy_destination_capacity(ids, &mut scratch, allocation)
                    .map_err(|e| fail(Cause::Residency(e)))?,
            );
            if let Some(value) = manager
                .ordinary_foreground_backing_capacity(ids, &mut scratch, runtime)
                .map_err(|e| fail(Cause::Residency(e)))?
            {
                staging_max = staging_max.max(value);
            }

            if let Some(value) = manager
                .ordinary_foreground_read_control_bytes(ids, &mut scratch)
                .map_err(|e| fail(Cause::Residency(e)))?
                .or(manager
                    .ordinary_retained_host_control_bytes(ids, &mut scratch)
                    .map_err(|e| fail(Cause::Residency(e)))?)
            {
                reader_max = reader_max.max(value);
            }
            if let Some(value) = manager
                .ordinary_foreground_observed_control_bytes(ids, &mut scratch, runtime)
                .map_err(|e| fail(Cause::Residency(e)))?
            {
                observed_max = observed_max.max(value);
            }
        }
        let bound = |full: usize, one: usize| -> Result<usize, Error> {
            Ok(full.min(
                one.checked_mul(count)
                    .ok_or_else(|| fail(Cause::Overflow))?,
            ))
        };
        // Host and Device acquisition may each reach the reader if the actual
        // cache evicts between them. Both attempts retain their own metadata.
        let observed_full = observed_full.or(source_pins.map(|_| 0));
        let reader = reader_full
            .map(|full| {
                bound(full, reader_max)
                    .and_then(|n| n.checked_mul(2).ok_or_else(|| fail(Cause::Overflow)))
            })
            .transpose()?;
        let observed_host = observed_full
            .map(|full| {
                bound(full, observed_max)
                    .and_then(|n| n.checked_mul(2).ok_or_else(|| fail(Cause::Overflow)))
            })
            .transpose()?;
        let copy_wrappers = ResidencyManager::ordinary_transfer_wrapper_control_bytes(window)
            .ok_or_else(|| fail(Cause::Overflow))?;
        let transfer = manager
            .ordinary_transfer_controls(&eligible, &mut scratch, window)
            .map_err(|e| fail(Cause::Residency(e)))?;
        let bound_physical = |full: u64, one: u64| -> Result<u64, Error> {
            Ok(full.min(
                one.checked_mul(u64::try_from(count).map_err(|_| fail(Cause::Overflow))?)
                    .ok_or_else(|| fail(Cause::Overflow))?,
            ))
        };
        let host_staging = staging_full
            .or(source_pins.map(|_| 0))
            .zip(crate::backend::managed_memory::cold_placement_fact(
                runtime.allocation_placement(),
            ))
            .map(|(full, placement)| {
                let bytes = bound_physical(full, staging_max)?
                    .checked_mul(2)
                    .ok_or_else(|| fail(Cause::Overflow))?;
                Ok::<_, Error>(OrdinaryResidencyPhysical {
                    bytes,
                    placement,
                    allocations: if staging_full.is_some() {
                        window
                            .physical_bindings
                            .checked_mul(2)
                            .ok_or_else(|| fail(Cause::Overflow))?
                    } else {
                        0
                    },
                })
            })
            .transpose()?;
        let destination_device = source
            .host()
            .map(|host| {
                (
                    host.destination_device_type(),
                    host.destination_device_index(),
                )
            })
            .or_else(|| {
                source.foreground().map(|disk| {
                    (
                        disk.destination_device_type(),
                        disk.destination_device_index(),
                    )
                })
            });
        let destination = destination_device
            .and_then(|(device, index)| {
                crate::backend::managed_memory::cold_copy_placement(device, index)
            })
            .map(|placement| {
                bound_physical(destination_full, destination_max).map(|bytes| {
                    OrdinaryResidencyPhysical {
                        bytes,
                        placement,
                        allocations: window.physical_bindings,
                    }
                })
            })
            .transpose()?;
        Ok(OrdinaryIndexedResidencyFacts {
            binding: self.clone(),
            source,
            first,
            revision,
            window,
            manager: bound(manager_full, manager_max)?,
            reader,
            observed_host,
            copy_wrappers,
            transfer,
            host_staging,
            destination,
            source_pins,
            funding: funding.clone(),
        })
    }
}
