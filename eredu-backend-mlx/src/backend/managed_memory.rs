//! Process-local accounting for Eredu-managed MLX host and device storage.
//!
//! One ledger coordinates host memory and every registered local accelerator.
//! Native hardware facts establish physical sharing; backing identities establish
//! storage sharing. Unlimited limits preserve ownership and admission checks.

use eredu_core::{
    DomainMemoryRequirements, MemoryDeviceId, MemoryDomainDescription, MemoryLimits,
    MemoryLocation, MemoryPlacement, MemoryTopology,
};
use std::sync::{Arc, Mutex, OnceLock};

pub(crate) mod bf16_projection_kernel;
pub(crate) mod input_allocator;
pub(crate) mod kernel_family;
pub(crate) mod metal_device;
mod ordinary_attachment;
mod publication_clone;
pub(crate) use publication_clone::PreparedPublicationClone;
mod physical_backing;
pub(crate) use ordinary_attachment::OrdinaryArrayAttachment;
pub(crate) use physical_backing::{
    ordinary_root_metadata_bytes, prepare_fresh_scoped_observer, prepare_scoped_observer,
    prepare_temporary_scoped_observer, scoped_observer_bytes,
};
pub(crate) mod pointwise_kernel;
pub(crate) mod recurrent_kernel;
pub(crate) mod router;
pub(crate) mod row_kernels;
pub(crate) mod scheduler;

use eredu_runtime::working_memory::{MemoryLedger, WorkingMemoryUnquotedLease};

use super::{error::Error, submission_recovery};

// This static owns its fixed accounting allocation for process lifetime. It
// contains no native context, runtime Rc or request-specific storage authority.
struct ProcessLedger {
    pool: MemoryLedger,
    default_placement: Arc<MemoryPlacement>,
    default_native_placement: safemlx::AllocationPlacement,
    managed_placement: Option<Arc<MemoryPlacement>>,
    gpu_placement: Option<Arc<MemoryPlacement>>,
    device_placements: Vec<(u32, Arc<MemoryPlacement>)>,
    // Missing fixed-owner qualification excludes inference under finite and
    // unlimited limits. This lease survives for the coordinator's lifetime.
    _unqualified: Option<WorkingMemoryUnquotedLease>,
}

impl ProcessLedger {
    fn new(
        facts: safemlx::RuntimeStaticBaseline,
        declarations: Option<&eredu_core::MemoryLimitDeclarations>,
    ) -> Result<Self, Error> {
        use eredu_runtime::working_memory::WorkingMemoryError;
        let topology = memory_topology()?;
        let default_native_placement = safemlx::default_allocation_placement()?;
        let default_placement =
            Arc::new(resolve_placement(default_native_placement, &topology).map_err(domain_error)?);
        let gpu_native_placement = safemlx::gpu_allocation_placement()?;
        let gpu_placement = if gpu_native_placement == safemlx::AllocationPlacement::Unknown {
            None
        } else {
            Some(Arc::new(
                resolve_placement(gpu_native_placement, &topology).map_err(domain_error)?,
            ))
        };
        let gpu_descriptor_bytes = gpu_placement
            .as_ref()
            .map(|placement| {
                placement.backing_bytes().and_then(|bytes| {
                    bytes
                        .checked_add(
                            (std::mem::size_of::<MemoryPlacement>()
                                + 2 * std::mem::size_of::<usize>())
                                as u64,
                        )
                        .ok_or(eredu_core::MemoryDomainError::Overflow)
                })
            })
            .transpose()
            .map_err(domain_error)?
            .unwrap_or(0);
        let device_count = topology
            .domains()
            .try_fold(0usize, |total, (_, description)| {
                let count = description
                    .locations
                    .iter()
                    .filter(|location| matches!(location, MemoryLocation::Device(_)))
                    .count();
                total.checked_add(count).ok_or(WorkingMemoryError::Overflow)
            })
            .map_err(|error| Error::Other(Box::new(error)))?;
        let mut device_placements = Vec::with_capacity(device_count);
        for (domain, description) in topology.domains() {
            for location in &description.locations {
                if let MemoryLocation::Device(device) = location {
                    device_placements.push((
                        device.ordinal,
                        Arc::new(MemoryPlacement::fixed(&topology, domain).map_err(domain_error)?),
                    ));
                }
            }
        }
        // The allocator facts certify whether the linked mechanism can produce
        // CUDA managed backings; registered ordinals alone do not establish it.
        let managed_count = match (default_native_placement, gpu_native_placement) {
            (safemlx::AllocationPlacement::CudaManaged { device_count }, _) => Some(device_count),
            (_, safemlx::AllocationPlacement::CudaAllocatorCandidates { device_count }) => {
                Some(device_count)
            }
            _ => None,
        };
        let managed_placement = managed_count
            .map(|device_count| {
                resolve_placement(
                    safemlx::AllocationPlacement::CudaManaged { device_count },
                    &topology,
                )
                .map(Arc::new)
                .map_err(domain_error)
            })
            .transpose()?;
        let managed_descriptor_bytes = managed_placement
            .as_ref()
            .map(|placement| {
                placement.backing_bytes().and_then(|bytes| {
                    bytes
                        .checked_add(
                            (std::mem::size_of::<MemoryPlacement>()
                                + 2 * std::mem::size_of::<usize>())
                                as u64,
                        )
                        .ok_or(eredu_core::MemoryDomainError::Overflow)
                })
            })
            .transpose()
            .map_err(domain_error)?
            .unwrap_or(0);
        let descriptor_bytes = default_placement
            .backing_bytes()
            .map_err(domain_error)?
            .checked_add(
                (std::mem::size_of::<MemoryPlacement>() + 2 * std::mem::size_of::<usize>()) as u64,
            )
            .ok_or(WorkingMemoryError::Overflow)
            .map_err(|error| Error::Other(Box::new(error)))?;
        let known = facts
            .known_static_storage_bytes()
            .and_then(|bytes| {
                bytes.checked_add(
                    std::mem::size_of::<OnceLock<Self>>()
                        + std::mem::size_of::<Mutex<()>>()
                        + std::mem::size_of_val(&PHYSICAL_OBSERVATION),
                )
            })
            .and_then(|bytes| {
                bytes.checked_add(std::mem::size_of::<
                    OnceLock<Result<Arc<MemoryTopology>, Arc<Error>>>,
                >())
            })
            .and_then(|bytes| bytes.checked_add(input_allocator::static_storage_bytes()))
            .and_then(|bytes| bytes.checked_add(metal_device::static_storage_bytes()))
            .and_then(|bytes| bytes.checked_add(scheduler::static_storage_bytes()))
            .and_then(|bytes| bytes.checked_add(router::static_storage_bytes()))
            .and_then(|bytes| bytes.checked_add(pointwise_kernel::static_storage_bytes()))
            .and_then(|bytes| bytes.checked_add(row_kernels::static_storage_bytes()))
            .and_then(|bytes| bytes.checked_add(recurrent_kernel::static_storage_bytes()))
            .and_then(|bytes| bytes.checked_add(bf16_projection_kernel::static_storage_bytes()))
            .and_then(|bytes| {
                bytes.checked_add(crate::backend::nn::fp8::kernel::static_storage_bytes())
            })
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or(WorkingMemoryError::Overflow)
            .map_err(|error| Error::Other(Box::new(error)))?;
        let device_descriptor_bytes = device_placements
            .capacity()
            .checked_mul(std::mem::size_of::<(u32, Arc<MemoryPlacement>)>())
            .and_then(|bytes| {
                device_placements
                    .len()
                    .checked_mul(
                        std::mem::size_of::<MemoryPlacement>() + 2 * std::mem::size_of::<usize>(),
                    )
                    .and_then(|descriptors| bytes.checked_add(descriptors))
            })
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or(WorkingMemoryError::Overflow)
            .map_err(|error| Error::Other(Box::new(error)))?;
        let mutex_backing =
            eredu_runtime::working_memory::OriginalHostMetadataCustody::initialized_mutex_bytes()
                .map_err(|error| Error::Other(Box::new(error)))?;
        let known = known
            .checked_add(mutex_backing)
            .and_then(|bytes| bytes.checked_add(device_descriptor_bytes))
            .and_then(|bytes| bytes.checked_add(descriptor_bytes))
            .and_then(|bytes| bytes.checked_add(managed_descriptor_bytes))
            .and_then(|bytes| bytes.checked_add(gpu_descriptor_bytes))
            .ok_or(WorkingMemoryError::Overflow)
            .map_err(|error| Error::Other(Box::new(error)))?;
        let limits = declarations
            .map(|limits| limits.resolve(&topology))
            .transpose()
            .map_err(domain_error)?
            .unwrap_or_else(|| MemoryLimits::unlimited(&topology));
        let mut baseline = DomainMemoryRequirements::zero(&topology);
        baseline
            .add_allocation(
                known,
                &MemoryPlacement::fixed(&topology, topology.host_domain()).map_err(domain_error)?,
            )
            .map_err(domain_error)?;
        let pool = MemoryLedger::new(topology, limits, baseline)
            .map_err(|error| Error::Other(Box::new(error)))?;
        let unqualified = if facts.fixed_storage_bytes().is_none() {
            Some(
                pool.acquire_unquoted()
                    .map_err(|error| Error::Other(Box::new(error)))?,
            )
        } else {
            None
        };
        Ok(Self {
            pool,
            default_placement,
            default_native_placement,
            managed_placement,
            gpu_placement,
            device_placements,
            _unqualified: unqualified,
        })
    }
}

fn domain_error(error: eredu_core::MemoryDomainError) -> Error {
    Error::Other(Box::new(error))
}

/// Backend hardware declarations, captured once by the process ledger.
fn physical_topology() -> Result<MemoryTopology, Error> {
    let devices = safemlx::physical_memory_topology()?;
    let mut host = MemoryDomainDescription {
        name: "host".into(),
        locations: vec![MemoryLocation::Host],
    };
    let mut separate = Vec::new();
    for device in devices {
        let location = MemoryLocation::Device(MemoryDeviceId {
            backend: "mlx",
            ordinal: device.ordinal,
        });
        if device.shares_host_memory {
            host.locations.push(location);
        } else {
            separate.push(MemoryDomainDescription {
                name: format!("mlx-gpu-{}", device.ordinal),
                locations: vec![location],
            });
        }
    }
    let mut domains = Vec::with_capacity(
        separate
            .len()
            .checked_add(1)
            .ok_or(eredu_runtime::working_memory::WorkingMemoryError::Overflow)
            .map_err(|error| Error::Other(Box::new(error)))?,
    );
    domains.push(host);
    domains.extend(separate);
    MemoryTopology::new(domains).map_err(domain_error)
}

fn resolve_placement(
    placement: safemlx::AllocationPlacement,
    topology: &MemoryTopology,
) -> Result<MemoryPlacement, eredu_core::MemoryDomainError> {
    let device = |ordinal| {
        MemoryLocation::Device(MemoryDeviceId {
            backend: "mlx",
            ordinal,
        })
    };
    match placement {
        safemlx::AllocationPlacement::Host => MemoryPlacement::fixed(topology, topology.host_domain()),
        safemlx::AllocationPlacement::Device { ordinal } => MemoryPlacement::fixed(topology, topology.domain_for(device(ordinal))?),
        safemlx::AllocationPlacement::CudaManaged { device_count } => MemoryPlacement::possible_locations(
            topology, std::iter::once(MemoryLocation::Host).chain((0..device_count).map(device)),
            "MLX CUDA allocator: full backing capacity in host memory and every registered GPU eligible for managed migration".into()),
        safemlx::AllocationPlacement::CudaAllocatorCandidates { device_count } => MemoryPlacement::possible_locations(
            topology, std::iter::once(MemoryLocation::Host).chain((0..device_count).map(device)),
            "MLX CUDA GPU allocator alternatives: fixed target allocation, independent managed or pinned small allocations, and same-placement cache reuse; full capacity in every registered candidate domain".into()),
        safemlx::AllocationPlacement::Unknown => Err(eredu_core::MemoryDomainError::InvalidPlacement),
    }
}

#[derive(Debug)]
struct TopologyInitializationFailure(Arc<Error>);
impl std::fmt::Display for TopologyInitializationFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.0, f)
    }
}
impl std::error::Error for TopologyInitializationFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&*self.0)
    }
}

/// One fallible process coordinator; initialization errors retain native sources.
static TOPOLOGY: OnceLock<Result<Arc<MemoryTopology>, Arc<Error>>> = OnceLock::new();
static LEDGER: OnceLock<ProcessLedger> = OnceLock::new();
static LEDGER_INIT: Mutex<()> = Mutex::new(());
static PHYSICAL_OBSERVATION: OnceLock<Result<(), Arc<safemlx::error::Exception>>> = OnceLock::new();

fn configured_ledger(
    declarations: Option<&eredu_core::MemoryLimitDeclarations>,
) -> Result<MemoryLedger, Error> {
    let topology = memory_topology()?;
    let requested = declarations
        .map(|declarations| declarations.resolve(&topology))
        .transpose()
        .map_err(domain_error)?;
    if LEDGER.get().is_none() {
        let _initialization = LEDGER_INIT.lock().map_err(|_| {
            Error::Other(Box::new(
                eredu_runtime::working_memory::WorkingMemoryError::Poisoned,
            ))
        })?;
        if LEDGER.get().is_none() {
            let initialized = ProcessLedger::new(safemlx::runtime_static_baseline(), declarations)?;
            // All fallible native/layout/capacity work completes before publishing
            // the immutable coordinator. Ordinary rejection leaves it uninstalled.
            if LEDGER.set(initialized).is_err() {
                unreachable!("serialized ledger initialization")
            }
        }
    }
    let domain = LEDGER.get().expect("initialized ledger");
    if requested
        .as_ref()
        .is_some_and(|limits| limits != domain.pool.configured_limits())
    {
        return Err(Error::InvalidOperation(
            "physical memory limits differ from the initialized process ledger",
        ));
    }
    let observed = PHYSICAL_OBSERVATION.get_or_init(|| {
        safemlx::observe_physical_backings(&physical_backing::PHYSICAL_ROOTS).map_err(Arc::new)
    });
    if let Err(error) = observed {
        return Err(Error::Other(Box::new(PhysicalObservationFailure(
            error.clone(),
        ))));
    }
    Ok(domain.pool.clone())
}
#[derive(Debug)]
struct PhysicalObservationFailure(Arc<safemlx::error::Exception>);
impl std::fmt::Display for PhysicalObservationFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}
impl std::error::Error for PhysicalObservationFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&*self.0)
    }
}

/// Configure physical limits before loading or allocating native resources.
/// Repeated equal configuration preserves the same process ledger identity.
/// An initialized ledger's limits cannot be replaced by a later caller.
pub fn configure_memory_limits(
    declarations: &eredu_core::MemoryLimitDeclarations,
) -> Result<(), Error> {
    configured_ledger(Some(declarations)).map(|_| ())
}

pub(crate) fn try_ledger() -> Result<MemoryLedger, Error> {
    configured_ledger(None)
}

pub(crate) fn ledger() -> MemoryLedger {
    try_ledger().expect("native physical memory topology and fixed accounting layout are available")
}

/// Coherent physical-domain charges and limits for the process coordinator.
/// Placement allowances are conservative accounting, not residency telemetry.
pub fn memory_snapshot() -> Result<eredu_runtime::working_memory::MemoryLedgerSnapshot, Error> {
    try_ledger()?.snapshot().map_err(Error::text_admission)
}

/// Physical identifiers and sharing of the actual process coordinator.
pub fn memory_topology() -> Result<Arc<MemoryTopology>, Error> {
    match TOPOLOGY.get_or_init(|| physical_topology().map(Arc::new).map_err(Arc::new)) {
        Ok(topology) => Ok(Arc::clone(topology)),
        Err(error) => Err(Error::Other(Box::new(TopologyInitializationFailure(
            Arc::clone(error),
        )))),
    }
}

#[derive(Debug)]
struct Owner {
    pool: MemoryLedger,
    // A distinct attachment namespace preserves original unquoted authority
    // even if the same immutable source is later inventoried in the pool.
    host_source_accounting_id: eredu_core::SharedStorageAccountingId,
    // Allocation authority is immutable. A residency transition can drop its
    // own handle, but cannot revoke exclusion from surviving descendants.
    _unquoted: WorkingMemoryUnquotedLease,
    // Last: the owning Arc and accounting identity retire before this hold.
    _host: eredu_core::HostPreparationAuthority,
}

// The source owns this accounting token only; no payload back-edge is retained.
struct NativeMetadataOwner {
    _native: NativeMemoryOwner,
    _host: eredu_core::HostPreparationAuthority,
}
impl eredu_core::SharedStorageRetirement for NativeMetadataOwner {
    fn retire(self: Arc<Self>) {
        drop(Arc::into_inner(self));
    }
}

/// Shared ownership of unquoted managed work and its surviving payloads.
///
/// Acquire before allocating; retain through native recovery and descendants.
/// This owner holds no native resources and certifies no byte coverage. Its
/// presence excludes quoted request admission until the last clone retires.
/// Merely registering an inventory does not promote or clear this owner.
#[derive(Debug)]
pub(crate) struct NativeMemoryOwner(Option<Arc<Owner>>);
impl Clone for NativeMemoryOwner {
    fn clone(&self) -> Self {
        Self(Some(self.shared().clone()))
    }
}
impl Drop for NativeMemoryOwner {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            drop(Arc::into_inner(owner));
        }
    }
}
fn metadata_memory_error(
    cause: eredu_core::HostMetadataFundingError,
) -> eredu_runtime::working_memory::WorkingMemoryError {
    use eredu_core::HostMetadataFundingError as H;
    use eredu_runtime::working_memory::WorkingMemoryError as W;
    match cause {
        H::Domain(cause) => W::Domain(cause),
        H::DomainAllowance {
            domain,
            required,
            available,
        } => W::DomainAllowanceExceeded {
            domain,
            required_bytes: required,
            available_bytes: available,
        },
        H::Overflow => W::Overflow,
        H::Poisoned => W::Poisoned,
        cause => {
            W::MetadataConstruction(eredu_nn::workspace::WorkspaceMetadataError::Funding(cause))
        }
    }
}

impl NativeMemoryOwner {
    pub(crate) fn acquire(pool: &MemoryLedger) -> Result<Self, Error> {
        Self::acquire_typed(pool).map_err(Error::PrefillControl)
    }

    /// Same real ordinary lease, with allocation-free policy rejection for a
    /// caller that has not yet acquired any construction/error custody.
    pub(crate) fn acquire_typed(
        pool: &MemoryLedger,
    ) -> Result<Self, eredu_runtime::working_memory::WorkingMemoryError> {
        let unquoted = pool.acquire_unquoted()?;
        let controls = Self::retained_control_bytes()?;
        let metadata = pool
            .prepare_storage_metadata()
            .map_err(metadata_memory_error)?;
        let host = metadata
            .prepare_host_owner(controls)
            .map_err(metadata_memory_error)?;
        Ok(Self(Some(Arc::new(Owner {
            pool: pool.clone(),
            host_source_accounting_id: Default::default(),
            _unquoted: unquoted,
            _host: host,
        }))))
    }

    fn retained_control_bytes() -> Result<usize, eredu_runtime::working_memory::WorkingMemoryError>
    {
        use eredu_runtime::working_memory::{OriginalHostMetadataCustody, WorkingMemoryError};
        OriginalHostMetadataCustody::shared_storage_bytes(std::alloc::Layout::new::<Owner>())?
            .checked_add(OriginalHostMetadataCustody::shared_storage_bytes(
                eredu_core::SharedStorageAccountingId::shared_payload_layout(),
            )?)
            .and_then(|n| {
                n.checked_add(std::mem::size_of::<(
                    Self,
                    Option<Owner>,
                    Result<Self, WorkingMemoryError>,
                )>() as u64)
            })
            .and_then(|n| usize::try_from(n).ok())
            .ok_or(WorkingMemoryError::Overflow)
    }

    /// Same constructor controls used by acquisition, including the retained
    /// neutral exclusion lease and its host-only preparation account.
    pub(crate) fn preparation_bytes()
    -> Result<u64, eredu_runtime::working_memory::WorkingMemoryError> {
        use eredu_runtime::working_memory::{StorageMetadataFunding, WorkingMemoryError};
        let owner = StorageMetadataFunding::host_owner_bytes(Self::retained_control_bytes()?)
            .map_err(metadata_memory_error)?;
        MemoryLedger::unquoted_owner_control_bytes()?
            .checked_add(MemoryLedger::storage_metadata_control_bytes()?)
            .and_then(|n| n.checked_add(u64::try_from(owner).ok()?))
            .ok_or(WorkingMemoryError::Overflow)
    }

    fn shared(&self) -> &Arc<Owner> {
        self.0.as_ref().expect("live native accounting owner")
    }

    pub(crate) fn pool(&self) -> &MemoryLedger {
        &self.shared().pool
    }

    pub(crate) fn same_authority(&self, other: &Self) -> bool {
        Arc::ptr_eq(self.shared(), other.shared())
    }

    /// Retains the neutral exclusion authority with portable state. This never
    /// clears or promotes the original lease, even when native owners retire.
    pub(crate) fn unquoted_lease(&self) -> Result<WorkingMemoryUnquotedLease, Error> {
        Ok(self.shared()._unquoted.clone())
    }

    /// Retains authority on an already materialized backing without evaluating.
    /// Lazy producers must complete inside their existing recovery scope first.
    pub(crate) fn retain_array(&self, array: &safemlx::Array) -> Result<(), Error> {
        if array
            .allocation_info()?
            .is_some_and(|info| info.bytes() == 0)
        {
            return Ok(());
        }
        OrdinaryArrayAttachment::<Self>::prepare(self.pool(), 0)?.attach(array, self.clone())
    }

    /// Preserves the original unquoted construction authority on every alias
    /// of immutable host metadata. This is exclusion, not a finite byte grant;
    /// later registration never promotes or revokes that original authority.
    pub(crate) fn retain_metadata(
        &self,
        metadata: &eredu_runtime::SharedHostMetadata,
    ) -> Result<(), Error> {
        if let eredu_runtime::SharedHostMetadata::Input(input) = metadata {
            if let Some(result) = input.original_residence(self.pool()) {
                // This exact closed B source already owns its payload/control.
                // Preserve the source's authority; never attach an ordinary
                // owner or manufacture a second finite source charge.
                return result.map_err(Error::PrefillControl);
            }
        }
        metadata
            .try_attach_owned_prepared(&self.shared().host_source_accounting_id, |layout| {
                use eredu_core::SharedStorageOwner;
                use eredu_runtime::working_memory::{
                    OriginalHostMetadataCustody, WorkingMemoryError,
                };
                let owner_bytes =
                    OriginalHostMetadataCustody::shared_storage_bytes(std::alloc::Layout::new::<
                        NativeMetadataOwner,
                    >())?;
                let controls = [
                    std::mem::size_of::<NativeMetadataOwner>(),
                    std::mem::size_of::<SharedStorageOwner<NativeMetadataOwner>>(),
                    std::mem::size_of::<
                        Result<SharedStorageOwner<NativeMetadataOwner>, WorkingMemoryError>,
                    >(),
                    std::mem::size_of::<(eredu_core::SharedStorageAttachmentLayout, &Self, usize)>(
                    ),
                    std::mem::size_of::<Option<NativeMetadataOwner>>(),
                ];
                let bytes = usize::try_from(owner_bytes)
                    .ok()
                    .and_then(|n| n.checked_add(layout.requested_bytes()))
                    .and_then(|n| n.checked_add(std::mem::size_of_val(&controls)))
                    .ok_or(WorkingMemoryError::Overflow)?;
                let bytes = controls
                    .into_iter()
                    .try_fold(bytes, usize::checked_add)
                    .ok_or(WorkingMemoryError::Overflow)?;
                let funding = self
                    .pool()
                    .prepare_storage_metadata()
                    .map_err(metadata_memory_error)?;
                let host = funding
                    .prepare_host_owner(bytes)
                    .map_err(metadata_memory_error)?;
                Ok::<_, WorkingMemoryError>(SharedStorageOwner::new(NativeMetadataOwner {
                    _native: self.clone(),
                    _host: host,
                }))
            })
            .map(|_| ())
            .map_err(Error::HostMetadataAttachment)
    }
}

impl submission_recovery::Retention for NativeMemoryOwner {
    fn observe(&self, _: submission_recovery::Status) {}
}

/// Distinct allocation authorities retained by restored or submitted state.
/// Clones share leases; merging the same authority never creates a new lease.
#[derive(Clone, Debug, Default)]
pub(crate) struct NativeMemoryRetention(Vec<NativeMemoryOwner>);

impl NativeMemoryRetention {
    /// Borrowed exact authorities for a separate finite exchange constructor.
    /// This grants neither storage nor another ordinary/native account.
    pub(crate) fn owners(&self) -> &[NativeMemoryOwner] {
        &self.0
    }

    /// Accepts the caller's already constructed finite, deduplicated rows.
    pub(crate) fn from_prepared_owners(owners: Vec<NativeMemoryOwner>) -> Self {
        Self(owners)
    }

    pub(crate) fn from_owner(owner: &NativeMemoryOwner) -> Self {
        Self(vec![owner.clone()])
    }

    pub(crate) fn singleton_metadata_bytes() -> u64 {
        std::mem::size_of::<NativeMemoryOwner>() as u64
    }

    /// Builds a copied authority list with room for one independent operation.
    /// Reserve its final logical size up front instead of growing a cloned Vec.
    pub(crate) fn with_added_owner(&self, owner: &NativeMemoryOwner) -> Self {
        let mut copied = Self(Vec::with_capacity(self.0.len() + 1));
        copied.0.extend(self.0.iter().cloned());
        copied.retain(owner);
        copied
    }

    pub(crate) fn metadata_bytes_with_added_owner(&self) -> Option<u64> {
        u64::try_from(self.0.len())
            .ok()?
            .checked_add(1)?
            .checked_mul(Self::singleton_metadata_bytes())
    }

    pub(crate) fn retain(&mut self, owner: &NativeMemoryOwner) {
        if !self.0.iter().any(|existing| existing.same_authority(owner)) {
            self.0.push(owner.clone());
        }
    }

    pub(crate) fn extend_from(&mut self, other: &Self) {
        for owner in &other.0 {
            self.retain(owner);
        }
    }

    pub(crate) fn covers_pool(&self, pool: &MemoryLedger) -> bool {
        self.0.iter().any(|owner| owner.pool().same_ledger(pool))
    }

    /// Whether these local handles all share the given immutable authority.
    /// Empty retention satisfies this check; it cannot conceal another owner.
    pub(crate) fn retains_only(&self, owner: &NativeMemoryOwner) -> bool {
        self.0.iter().all(|retained| retained.same_authority(owner))
    }

    pub(crate) fn logical_metadata_bytes(&self) -> Option<u64> {
        u64::try_from(self.0.capacity())
            .ok()?
            .checked_mul(u64::try_from(std::mem::size_of::<NativeMemoryOwner>()).ok()?)
    }

    pub(crate) fn retain_array(&self, array: &safemlx::Array) -> Result<(), Error> {
        if self.0.is_empty()
            || array
                .allocation_info()?
                .is_some_and(|info| info.bytes() == 0)
        {
            return Ok(());
        }
        let bytes = self
            .0
            .len()
            .checked_mul(std::mem::size_of::<NativeMemoryOwner>())
            .ok_or(Error::PrefillControl(
                eredu_runtime::working_memory::WorkingMemoryError::Overflow,
            ))?;
        let prepared = OrdinaryArrayAttachment::<Self>::prepare(self.0[0].pool(), bytes)?;
        prepared.attach(array, self.clone())
    }
}

impl submission_recovery::Retention for NativeMemoryRetention {
    fn observe(&self, _: submission_recovery::Status) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_core::{
        Admission, EstimationCompleteness, ExecutionWorkspaceEstimate, InferenceGeometry,
        InputTokenCount, LayerSchedule, OutputDemand, StateMemoryLayout, WorkspaceBound,
        cache::LayerCachePolicy,
    };
    use eredu_runtime::working_memory::{InferenceExecutionIdentity, WorkingMemoryError};
    use std::{cell::Cell, rc::Rc};

    fn zero_admission(pool: &MemoryLedger) -> Admission {
        let geometry = InferenceGeometry {
            batch_size: 1,
            cached_positions: 0,
            input_positions: 1,
            max_output_tokens: 0,
            prefill_chunk_positions: 1,
            output: OutputDemand::StateOnly,
        };
        let layout = StateMemoryLayout::new(
            LayerSchedule::new(1, vec![LayerCachePolicy::NoState]).unwrap(),
            vec![0],
            1,
            1,
            EstimationCompleteness::Complete,
        )
        .unwrap();
        let zero = || WorkspaceBound::bounded(0, "stateless fixture without managed payload");
        let mut state = eredu_core::estimate_runtime_state(
            &layout,
            InputTokenCount::text(1),
            0,
            1,
            std::num::NonZeroU8::new(4).unwrap(),
        )
        .unwrap()
        .with_execution_workspace(ExecutionWorkspaceEstimate {
            geometry,
            activations: zero(),
            attention: zero(),
            vocabulary: zero(),
            state_update: zero(),
            materialization: zero(),
            retained: zero(),
            physical_domains: Some(eredu_core::DomainExecutionWorkspaceEstimate {
                geometry,
                activations: DomainMemoryRequirements::zero(pool.topology()),
                attention: DomainMemoryRequirements::zero(pool.topology()),
                vocabulary: DomainMemoryRequirements::zero(pool.topology()),
                state_update: DomainMemoryRequirements::zero(pool.topology()),
                materialization: DomainMemoryRequirements::zero(pool.topology()),
                retained: DomainMemoryRequirements::zero(pool.topology()),
            }),
        })
        .unwrap();
        state.physical_domains = Some(eredu_core::DomainRuntimeStateEstimate {
            geometry,
            decoder_state: DomainMemoryRequirements::zero(pool.topology()),
            media_embeddings: DomainMemoryRequirements::zero(pool.topology()),
            media_workspace: DomainMemoryRequirements::zero(pool.topology()),
        });
        Admission {
            memory_limits: eredu_core::MemoryLimitDeclarations::unlimited(),
            additional_headroom: eredu_core::MemoryHeadroomDeclarations::none(),
            requested_positions: 1,
            state,
            incremental_required_bytes: Some(0),
        }
    }

    fn host_charge(pool: &MemoryLedger) -> u64 {
        pool.snapshot()
            .unwrap()
            .domains
            .iter()
            .find(|domain| domain.domain == pool.topology().host_domain())
            .unwrap()
            .current_charge_bytes
    }
    fn test_pool(existing: u64) -> MemoryLedger {
        let topology = Arc::new(
            MemoryTopology::new(vec![MemoryDomainDescription {
                name: "host".into(),
                locations: vec![MemoryLocation::Host],
            }])
            .unwrap(),
        );
        let mut baseline = DomainMemoryRequirements::zero(&topology);
        baseline
            .add_allocation(
                existing,
                &MemoryPlacement::fixed(&topology, topology.host_domain()).unwrap(),
            )
            .unwrap();
        MemoryLedger::new(
            Arc::clone(&topology),
            MemoryLimits::unlimited(&topology),
            baseline,
        )
        .unwrap()
    }
    fn fixed_baseline_required() -> bool {
        std::env::var_os("EREDU_REQUIRE_STATIC_BASELINE_QUALIFICATION").is_some()
    }

    #[test]
    fn managed_candidates_collapse_physical_sharing_without_losing_the_basis() {
        let gpu = |ordinal| {
            MemoryLocation::Device(MemoryDeviceId {
                backend: "mlx",
                ordinal,
            })
        };
        let topology = MemoryTopology::new(vec![
            MemoryDomainDescription {
                name: "shared".into(),
                locations: vec![MemoryLocation::Host, gpu(0)],
            },
            MemoryDomainDescription {
                name: "second".into(),
                locations: vec![gpu(1)],
            },
        ])
        .unwrap();
        let placement = resolve_placement(
            safemlx::AllocationPlacement::CudaManaged { device_count: 2 },
            &topology,
        )
        .unwrap();
        assert_eq!(placement.domains().len(), 2);
        let mut requirements = DomainMemoryRequirements::zero(&topology);
        requirements.add_allocation(64, &placement).unwrap();
        for (_, charge) in requirements.iter() {
            assert_eq!(charge.accounted_bytes, 0);
            assert_eq!(charge.placement_allowance_bytes, 64);
        }
        assert_eq!(requirements.placement_allowances().len(), 1);
        assert!(
            resolve_placement(
                safemlx::AllocationPlacement::CudaManaged { device_count: 3 },
                &topology
            )
            .is_err()
        );
        assert!(resolve_placement(safemlx::AllocationPlacement::Unknown, &topology).is_err());
    }

    #[test]
    fn fixed_native_baseline_is_charged_once_at_exact_and_one_short_capacity() {
        let facts = safemlx::runtime_static_baseline();
        let domain = ProcessLedger::new(facts, None).unwrap();
        let bytes = host_charge(&domain.pool);
        assert!(bytes > 0);
        if fixed_baseline_required() {
            assert!(domain._unqualified.is_none());
        }
        if domain._unqualified.is_some() {
            assert!(matches!(
                domain.pool.reserve(
                    &InferenceExecutionIdentity::default(),
                    &zero_admission(&domain.pool)
                ),
                Err(WorkingMemoryError::UnknownBound)
            ));
            return;
        }
        let same = domain.pool.clone();
        assert!(same.same_ledger(&domain.pool));
        let mut admission = zero_admission(&domain.pool);
        // A nonzero application request, separate from the fixed baseline.
        admission.incremental_required_bytes = Some(17);
        admission
            .state
            .execution_workspace
            .as_mut()
            .unwrap()
            .physical_domains
            .as_mut()
            .unwrap()
            .retained
            .add_allocation(17, domain.pool.host_placement())
            .unwrap();
        admission
            .state
            .execution_workspace
            .as_mut()
            .unwrap()
            .retained = WorkspaceBound::bounded(17, "fixture retained producer storage");
        let required = same
            .reservation_requirements(&admission, None)
            .unwrap()
            .get(same.topology().host_domain())
            .unwrap()
            .total()
            .unwrap();
        let reservation = same
            .reserve_with_capacity(
                &InferenceExecutionIdentity::default(),
                &admission,
                MemoryLimits::resolve(
                    same.topology(),
                    [(
                        same.topology().host_domain(),
                        eredu_core::MemoryLimit::Finite(bytes + required),
                    )],
                )
                .unwrap(),
            )
            .unwrap();
        assert_eq!(host_charge(&domain.pool), bytes + required);
        drop(reservation);
        assert!(matches!(
            same.reserve_with_capacity(
                &InferenceExecutionIdentity::default(),
                &admission,
                MemoryLimits::resolve(same.topology(), [(same.topology().host_domain(), eredu_core::MemoryLimit::Finite(bytes + required - 1))]).unwrap()
            ),
            Err(WorkingMemoryError::Domain(eredu_core::MemoryDomainError::BudgetExceeded { requested_bytes, .. })) if requested_bytes == required
        ));
        drop(same);
        assert_eq!(host_charge(&domain.pool), bytes);
    }

    #[test]
    fn fixed_native_baseline_concurrent_requests_share_one_charge() {
        use std::sync::Barrier;
        let slot = OnceLock::<ProcessLedger>::new();
        let starts = Barrier::new(4);
        let all_reserved = Barrier::new(4);
        let all_observed = Barrier::new(4);
        std::thread::scope(|threads| {
            let handles = (0..4)
                .map(|_| {
                    threads.spawn(|| {
                        starts.wait();
                        let domain = slot.get_or_init(|| {
                            ProcessLedger::new(safemlx::runtime_static_baseline(), None).unwrap()
                        });
                        // Existing fixed bytes are independent of each live request.
                        // Every contender uses the same actual constructor and domain.
                        let baseline = host_charge(&domain.pool);
                        let mut admission = zero_admission(&domain.pool);
                        admission.incremental_required_bytes = Some(17);
                        admission
                            .state
                            .execution_workspace
                            .as_mut()
                            .unwrap()
                            .physical_domains
                            .as_mut()
                            .unwrap()
                            .retained
                            .add_allocation(17, domain.pool.host_placement())
                            .unwrap();
                        admission
                            .state
                            .execution_workspace
                            .as_mut()
                            .unwrap()
                            .retained =
                            WorkspaceBound::bounded(17, "concurrent fixture retained storage");
                        let required = domain
                            .pool
                            .reservation_requirements(&admission, None)
                            .unwrap()
                            .get(domain.pool.topology().host_domain())
                            .unwrap()
                            .total()
                            .unwrap();
                        let qualified = domain._unqualified.is_none();
                        // Synchronize BEFORE anyone reserves, so each observed subtotal
                        // is the same fixed baseline, not a sibling's live reservation.
                        starts.wait();
                        let reservation = domain.pool.reserve_with_capacity(
                            &InferenceExecutionIdentity::default(),
                            &admission,
                            MemoryLimits::resolve(
                                domain.pool.topology(),
                                [(
                                    domain.pool.topology().host_domain(),
                                    eredu_core::MemoryLimit::Finite(baseline + 4 * required),
                                )],
                            )
                            .unwrap(),
                        );
                        all_reserved.wait();
                        let concurrent_bytes =
                            Ok::<_, WorkingMemoryError>(host_charge(&domain.pool));
                        // Keep actual Result owners until all siblings observed
                        // the full charge; a refusal must not hang the barriers.
                        let alias = domain.pool.clone();
                        all_observed.wait();
                        if fixed_baseline_required() {
                            assert!(qualified);
                        }
                        assert_eq!(
                            concurrent_bytes.unwrap(),
                            baseline + if qualified { 4 * required } else { 0 }
                        );
                        assert!(alias.same_ledger(&domain.pool));
                        if qualified {
                            drop(reservation.unwrap());
                        } else {
                            assert!(matches!(reservation, Err(WorkingMemoryError::UnknownBound)));
                        }
                        (alias, baseline)
                    })
                })
                .collect::<Vec<_>>();
            let mut results = handles.into_iter().map(|thread| thread.join().unwrap());
            let (first, bytes) = results.next().unwrap();
            for (next, next_bytes) in results {
                assert!(first.same_ledger(&next));
                assert_eq!(next_bytes, bytes);
            }
            assert_eq!(host_charge(&first), bytes);
        });
    }

    #[test]
    fn unknown_fixed_native_baseline_keeps_ordinary_authority_and_strict_refusal() {
        let mut facts = safemlx::runtime_static_baseline();
        facts.constant_registry = false; // explicit unsupported producer fixture
        let domain = ProcessLedger::new(facts, None).unwrap();
        let bytes = host_charge(&domain.pool);
        assert!(bytes > 0);
        assert!(domain._unqualified.is_some());
        assert_eq!(domain.pool.unquoted_owner_count().unwrap(), 1);
        let ordinary = NativeMemoryOwner::acquire_typed(&domain.pool).unwrap();
        assert_eq!(domain.pool.unquoted_owner_count().unwrap(), 2);
        drop(ordinary);
        let alias = domain.pool.clone();
        assert!(matches!(
            alias.reserve(
                &InferenceExecutionIdentity::default(),
                &zero_admission(&domain.pool)
            ),
            Err(WorkingMemoryError::UnknownBound)
        ));
        assert_eq!(alias.unquoted_owner_count().unwrap(), 1);
        assert_eq!(host_charge(&alias), bytes);
    }

    #[test]
    fn metadata_reattachment_reuses_paid_node_and_alias_retains_exclusion() {
        // The fixture owns source payload construction. The native attachment
        // pays its separate node/owner controls and preserves exclusion.
        let pool = test_pool(0);
        let baseline = host_charge(&pool);
        let source =
            eredu_runtime::SharedHostMetadata::Layout(eredu_runtime::SharedStateLayout::new(
                eredu_runtime::StateLayout::new(
                    LayerSchedule::new(1, vec![LayerCachePolicy::NoState]).unwrap(),
                )
                .unwrap(),
            ));
        let alias = source.clone();
        let owner = NativeMemoryOwner::acquire(&pool).unwrap();
        let owner_charge = host_charge(&pool);
        owner.retain_metadata(&source).unwrap();
        let attached = pool.snapshot().unwrap();
        assert!(host_charge(&pool) > owner_charge);
        owner.retain_metadata(&alias).unwrap();
        assert_eq!(pool.snapshot().unwrap(), attached);
        drop(owner);
        drop(source);
        assert_eq!(pool.unquoted_owner_count().unwrap(), 1);
        assert_eq!(pool.snapshot().unwrap(), attached);
        drop(alias);
        assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
        assert_eq!(host_charge(&pool), baseline);
    }

    #[test]
    fn clones_share_one_unquoted_owner_across_threads() {
        let pool = test_pool(13);
        let baseline = host_charge(&pool);
        let owner = NativeMemoryOwner::acquire(&pool).unwrap();
        let charged = host_charge(&pool);
        assert_eq!(
            charged - baseline,
            NativeMemoryOwner::preparation_bytes().unwrap()
        );
        let descendant = owner.clone();
        assert_eq!(host_charge(owner.pool()), charged);
        assert_eq!(pool.unquoted_owner_count().unwrap(), 1);
        drop(owner);
        assert_eq!(pool.unquoted_owner_count().unwrap(), 1);
        std::thread::spawn(move || {
            assert_eq!(descendant.pool().unquoted_owner_count().unwrap(), 1);
            drop(descendant);
        })
        .join()
        .unwrap();
        assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
        assert_eq!(host_charge(&pool), baseline);
    }

    #[test]
    fn rejected_acquisition_preserves_zero_byte_reservation_and_error_source() {
        let pool = test_pool(0);
        let baseline = host_charge(&pool);
        let required = pool
            .reservation_requirements(&zero_admission(&pool), None)
            .unwrap()
            .get(pool.topology().host_domain())
            .unwrap()
            .total()
            .unwrap();
        let reservation = pool
            .reserve(
                &InferenceExecutionIdentity::default(),
                &zero_admission(&pool),
            )
            .unwrap();
        assert!(matches!(
            NativeMemoryOwner::acquire(&pool),
            Err(Error::PrefillControl(
                WorkingMemoryError::ReservedWorkActive
            ))
        ));
        assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
        assert_eq!(host_charge(&pool), baseline + required);
        drop(reservation);
        let owner = NativeMemoryOwner::acquire(&pool).unwrap();
        assert_eq!(pool.unquoted_owner_count().unwrap(), 1);
        drop(owner);
        assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
    }

    struct Pending(Rc<Cell<bool>>);
    impl submission_recovery::Probe for Pending {
        fn seal(&mut self) {}
        fn progress(&self) -> submission_recovery::Status {
            submission_recovery::Status {
                settled: self.0.get(),
                failed: true,
                blocked: !self.0.get(),
            }
        }
    }

    #[test]
    fn pending_failure_and_unwind_retain_owner_until_independent_settlement() {
        for unwind in [false, true] {
            let pool = test_pool(0);
            let baseline = host_charge(&pool);
            let settled = Rc::new(Cell::new(false));
            let owner = NativeMemoryOwner::acquire(&pool).unwrap();
            let recovery = submission_recovery::Recovery::with_probe(
                owner.clone(),
                Pending(Rc::clone(&settled)),
            );
            drop(owner);
            if unwind {
                let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let _recovery = recovery;
                    panic!("fixture operation unwind");
                }));
                assert!(panic.is_err());
            } else {
                drop(recovery);
            }
            submission_recovery::reap();
            assert_eq!(pool.unquoted_owner_count().unwrap(), 1);
            settled.set(true);
            submission_recovery::wait_for_retirement(|| pool.unquoted_owner_count().unwrap() == 0);
            assert_eq!(host_charge(&pool), baseline);
        }
    }
}

pub(crate) mod gpu_stream;

/// Borrow only previously established native facts; cold tracing creates no
/// device, stream, allocator or execution authority.
pub(crate) fn cold_topology() -> Option<&'static MemoryTopology> {
    Some(LEDGER.get()?.pool.topology())
}
pub(crate) fn cold_default_placement() -> Option<&'static MemoryPlacement> {
    Some(&LEDGER.get()?.default_placement)
}
/// The guarded allocator's established physical strategy, independent of the
/// ordinary allocator's managed-memory candidate envelope.
pub(crate) fn cold_original_allocator_placement() -> Option<&'static MemoryPlacement> {
    cold_placement_fact(safemlx::PreparedInputRuntime::established_allocation_placement()?)
}
pub(crate) fn cold_allocation_placement(
    info: &safemlx::AllocationInfo,
) -> Option<&'static MemoryPlacement> {
    cold_placement_fact(info.placement())
}
/// Translate a previously established native allocator fact without creating
/// an allocator, stream or placement owner.
pub(crate) fn cold_placement_fact(
    placement: safemlx::AllocationPlacement,
) -> Option<&'static MemoryPlacement> {
    let domain = LEDGER.get()?;
    match placement {
        safemlx::AllocationPlacement::Host => Some(domain.pool.host_placement()),
        placement if placement == domain.default_native_placement => {
            Some(&domain.default_placement)
        }
        safemlx::AllocationPlacement::CudaManaged { device_count }
            if usize::try_from(device_count).ok() == Some(domain.device_placements.len()) =>
        {
            domain.managed_placement.as_deref()
        }
        safemlx::AllocationPlacement::Device { ordinal } => domain
            .device_placements
            .iter()
            .find(|(device, _)| *device == ordinal)
            .map(|(_, placement)| placement.as_ref()),
        _ => None,
    }
}
pub(crate) fn allocation_placement_handle(
    info: &safemlx::AllocationInfo,
    pool: &MemoryLedger,
) -> Result<Arc<MemoryPlacement>, eredu_core::MemoryDomainError> {
    placement_fact_handle(info.placement(), pool)
}
pub(crate) fn placement_fact_handle(
    placement: safemlx::AllocationPlacement,
    pool: &MemoryLedger,
) -> Result<Arc<MemoryPlacement>, eredu_core::MemoryDomainError> {
    if placement == safemlx::AllocationPlacement::Host {
        return Ok(pool.host_placement_handle());
    }
    let domain = LEDGER
        .get()
        .ok_or(eredu_core::MemoryDomainError::InvalidPlacement)?;
    pool.topology().slot(domain.pool.topology().host_domain())?;
    match placement {
        placement if placement == domain.default_native_placement => {
            Ok(Arc::clone(&domain.default_placement))
        }
        safemlx::AllocationPlacement::CudaManaged { device_count }
            if usize::try_from(device_count).ok() == Some(domain.device_placements.len()) =>
        {
            domain
                .managed_placement
                .as_ref()
                .map(Arc::clone)
                .ok_or(eredu_core::MemoryDomainError::InvalidPlacement)
        }
        safemlx::AllocationPlacement::Device { ordinal } => domain
            .device_placements
            .iter()
            .find(|(device, _)| *device == ordinal)
            .map(|(_, placement)| Arc::clone(placement))
            .ok_or(eredu_core::MemoryDomainError::InvalidPlacement),
        _ => Err(eredu_core::MemoryDomainError::InvalidPlacement),
    }
}

pub(crate) fn cold_shared_allocator_placement() -> Option<&'static MemoryPlacement> {
    let domain = LEDGER.get()?;
    (domain.default_native_placement == safemlx::AllocationPlacement::Host)
        .then_some(&domain.default_placement)
}

/// Prospective placement from the linked GPU allocator's complete alternatives.
pub(crate) fn cold_gpu_allocator_placement() -> Option<&'static MemoryPlacement> {
    LEDGER.get()?.gpu_placement.as_deref()
}
pub(crate) fn cold_copy_placement(
    device: safemlx::DeviceType,
    ordinal: i32,
) -> Option<&'static MemoryPlacement> {
    let domain = LEDGER.get()?;
    match device {
        safemlx::DeviceType::Cpu if ordinal == 0 => Some(&domain.default_placement),
        safemlx::DeviceType::Gpu
            if domain
                .device_placements
                .iter()
                .any(|(registered, _)| i32::try_from(*registered).ok() == Some(ordinal)) =>
        {
            domain.gpu_placement.as_deref()
        }
        _ => None,
    }
}

#[cfg(test)]
pub(crate) fn cold_default_placement_handle() -> Option<Arc<MemoryPlacement>> {
    Some(Arc::clone(&LEDGER.get()?.default_placement))
}
