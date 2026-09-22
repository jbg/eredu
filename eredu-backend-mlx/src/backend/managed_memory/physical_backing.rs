//! Ordinary allocator roots retain canonical accounting across native cache reuse.
use super::*;
use crate::backend::runtime::residency::storage::StorageIdentity;
use eredu_runtime::working_memory::{
    PendingStorageAllocation, StorageAllocation, StoragePublicationLayout,
    WorkingMemoryAllocationFunding, WorkingMemoryError, WorkingMemoryFundingScope,
};

#[derive(Debug)]
struct PendingRoot(PendingStorageAllocation<StorageIdentity>);
impl safemlx::PhysicalBackingPublication for PendingRoot {
    fn publish(&mut self) -> safemlx::error::Result<()> {
        self.0
            .publish()
            .map_err(safemlx::error::Exception::from_source)
    }
}
pub(super) struct PhysicalRoots;
pub(super) static PHYSICAL_ROOTS: PhysicalRoots = PhysicalRoots;

impl safemlx::PhysicalBackingObserver for PhysicalRoots {
    fn admit(
        &self,
        facts: safemlx::AllocationInfo,
    ) -> safemlx::error::Result<safemlx::PhysicalBackingCustody> {
        type Owner = PendingRoot;
        let domain = LEDGER.get().ok_or_else(|| {
            safemlx::error::Exception::custom("physical ledger is not initialized")
        })?;
        let pool = &domain.pool;
        let placement = allocation_placement_handle(&facts, pool)
            .map_err(safemlx::error::Exception::from_source)?;
        let bytes = u64::try_from(facts.bytes()).map_err(safemlx::error::Exception::from_source)?;
        let controls = u64::try_from(facts.host_control_bytes())
            .map_err(safemlx::error::Exception::from_source)?;
        let owner_bytes = u64::try_from(safemlx::PhysicalBackingCustody::control_bytes::<Owner>())
            .map_err(safemlx::error::Exception::from_source)?;
        // Metadata preparation is its own paid boundary. The subsequent single
        // publication transaction admits every physical domain before native
        // controls or backing are allocated. Failure allocates no native root.
        let prepared = StoragePublicationLayout::<StorageIdentity>::pending(2)
            .and_then(|layout| layout.with_additional_host_metadata(owner_bytes))
            .and_then(|layout| layout.fund(pool))
            .map_err(safemlx::error::Exception::from_source)?;
        let owner = prepared
            .reserve_storage([
                (
                    StorageIdentity::Native(facts.identity()),
                    StorageAllocation::new(bytes, placement),
                ),
                (
                    StorageIdentity::NativeControl(facts.identity()),
                    StorageAllocation::new(controls, pool.host_placement_handle()),
                ),
            ])
            .map_err(safemlx::error::Exception::from_source)?;
        Ok(safemlx::PhysicalBackingCustody::new_pending(PendingRoot(
            owner,
        )))
    }
}

/// Persistent bookkeeping for one new backing, from the actual pending worker.
/// Imported roots already retain this owner and expose only native control rows.
pub(crate) fn ordinary_root_metadata_bytes() -> Result<u64, WorkingMemoryError> {
    let owner = u64::try_from(safemlx::PhysicalBackingCustody::control_bytes::<PendingRoot>())
        .map_err(|_| WorkingMemoryError::Overflow)?;
    let layout = StoragePublicationLayout::<StorageIdentity>::pending(2)?
        .with_additional_host_metadata(owner)?;
    layout
        .requested_bytes()
        .checked_add(MemoryLedger::storage_metadata_control_bytes()?)
        .ok_or(WorkingMemoryError::Overflow)
}
struct AssignedRoots {
    funding: WorkingMemoryAllocationFunding,
    // Native Scope/Task aliases may outlive the publishing Work. Keep the
    // observer's actual constructor grant until its final shared block retires.
    _host: eredu_core::HostPreparationAuthority,
}
impl safemlx::PhysicalBackingObserver for AssignedRoots {
    fn admit(
        &self,
        facts: safemlx::AllocationInfo,
    ) -> safemlx::error::Result<safemlx::PhysicalBackingCustody> {
        let pool = self.funding.pool();
        let placement = allocation_placement_handle(&facts, pool)
            .map_err(safemlx::error::Exception::from_source)?;
        let owner_bytes =
            u64::try_from(safemlx::PhysicalBackingCustody::control_bytes::<PendingRoot>())
                .map_err(safemlx::error::Exception::from_source)?;
        let funding = self
            .funding
            .prepare_storage_metadata()
            .map_err(safemlx::error::Exception::from_source)?;
        let prepared = StoragePublicationLayout::<StorageIdentity>::pending(2)
            .and_then(|layout| layout.with_additional_host_metadata(owner_bytes))
            .and_then(|layout| layout.prepare(pool, &funding))
            .map_err(safemlx::error::Exception::from_source)?;
        let owner = prepared
            .reserve_storage_from(
                &self.funding,
                [
                    (
                        StorageIdentity::Native(facts.identity()),
                        StorageAllocation::new(
                            u64::try_from(facts.bytes())
                                .map_err(safemlx::error::Exception::from_source)?,
                            placement,
                        ),
                    ),
                    (
                        StorageIdentity::NativeControl(facts.identity()),
                        StorageAllocation::new(
                            u64::try_from(facts.host_control_bytes())
                                .map_err(safemlx::error::Exception::from_source)?,
                            pool.host_placement_handle(),
                        ),
                    ),
                ],
            )
            .map_err(safemlx::error::Exception::from_source)?;
        Ok(safemlx::PhysicalBackingCustody::new_pending(PendingRoot(
            owner,
        )))
    }
}
pub(crate) fn scoped_observer_bytes() -> Result<u64, WorkingMemoryError> {
    eredu_runtime::working_memory::OriginalHostMetadataCustody::shared_storage_bytes(
        std::alloc::Layout::new::<AssignedRoots>(),
    )?
    .checked_add(
        u64::try_from(safemlx::ScopedPhysicalBackingObserver::control_bytes()
            .checked_add(safemlx::ScopedPhysicalBackingObserver::retirement_control_bytes::<AssignedRoots>())
            .ok_or(WorkingMemoryError::Overflow)?)
            .map_err(|_| WorkingMemoryError::Overflow)?,
    )
    .ok_or(WorkingMemoryError::Overflow)
}
/// The caller funds `scoped_observer_bytes` and retains that constructor hold
/// through this owner and every native scope/Task alias before dropping it.
pub(crate) fn prepare_scoped_observer(
    scope: &mut WorkingMemoryFundingScope,
    host: eredu_core::HostPreparationAuthority,
) -> Result<safemlx::ScopedPhysicalBackingObserver, crate::backend::error::Error> {
    prepare_observer(scope, host, BackingPolicy::Cached)
}
/// Temporary constructor roots retain their exact payer through their final
/// alias, then retire rather than extending that custody through native cache.
pub(crate) fn prepare_temporary_scoped_observer(
    scope: &mut WorkingMemoryFundingScope,
    host: eredu_core::HostPreparationAuthority,
) -> Result<safemlx::ScopedPhysicalBackingObserver, crate::backend::error::Error> {
    prepare_observer(scope, host, BackingPolicy::Temporary)
}
/// Bounded ordinary request work creates independent roots and retires them
/// after their final physical alias. It never borrows or relabels older cached
/// roots, whose own ledger charges remain live until actual physical eviction.
pub(crate) fn prepare_fresh_scoped_observer(
    scope: &mut WorkingMemoryFundingScope,
    host: eredu_core::HostPreparationAuthority,
) -> Result<safemlx::ScopedPhysicalBackingObserver, crate::backend::error::Error> {
    prepare_observer(scope, host, BackingPolicy::Fresh)
}
enum BackingPolicy {
    Cached,
    Temporary,
    Fresh,
}
fn prepare_observer(
    scope: &mut WorkingMemoryFundingScope,
    host: eredu_core::HostPreparationAuthority,
    policy: BackingPolicy,
) -> Result<safemlx::ScopedPhysicalBackingObserver, crate::backend::error::Error> {
    let funding = scope
        .allocation_funding()
        .map_err(crate::backend::error::Error::PrefillControl)?;
    let observer = Arc::new(AssignedRoots {
        funding,
        _host: host,
    });
    match policy {
        BackingPolicy::Cached => safemlx::ScopedPhysicalBackingObserver::new(observer),
        BackingPolicy::Temporary => safemlx::ScopedPhysicalBackingObserver::new_uncached(observer),
        BackingPolicy::Fresh => safemlx::ScopedPhysicalBackingObserver::new_fresh(observer),
    }
    .map_err(Into::into)
}
