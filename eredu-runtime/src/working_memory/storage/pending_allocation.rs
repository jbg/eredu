//! Allocation-only allowance, prepared before native work and published afterward.
use super::*;
use eredu_core::{HostPreparationAuthority, MemoryPlacement};

#[derive(Debug)]
struct PublicationRow {
    locator: EntryLocator,
    bytes: u64,
    placement: Arc<MemoryPlacement>,
}
/// Move-only physical allocation allowance. It grants no inference or submission
/// authority, and its identities remain unavailable until backing is published.
#[derive(Debug)]
pub struct PendingStorageAllocation<K: Ord + Send + 'static> {
    rows: Vec<PublicationRow>,
    published: bool,
    // Its paid preparation owns the rows above until their buffers retire.
    storage: WorkingMemoryStorage<K>,
}
pub(super) fn control_bytes(population: usize) -> Result<u64, WorkingMemoryError> {
    cold_metadata::vector_bytes::<PublicationRow>(population)
}
pub(super) fn reserve<K: Clone + Ord + Send + 'static>(
    pool: &MemoryLedger,
    unique: Vec<(K, StorageAllocation)>,
    host: &HostPreparationAuthority,
    funding: Option<&super::super::WorkingMemoryAllocationFunding>,
) -> Result<PendingStorageAllocation<K>, WorkingMemoryError> {
    let rows = super::super::qualified_storage::vector(unique.len(), true)?;
    let bytes = unique
        .iter()
        .try_fold(0u64, |n, (_, a)| n.checked_add(a.capacity_bytes()));
    let mut storage = WorkingMemoryStorage::pending_domains(
        unique.iter().map(|(key, _)| key.clone()).collect(),
        bytes,
    );
    Arc::get_mut(&mut storage.0).unwrap().preparation = Some(host.clone());
    pool.commit_physical_inventory_mode(
        unique,
        funding.map(physical::PhysicalFunding::Allocation),
        usize::from(funding.is_some()),
        Vec::new(),
        host,
        true,
    )?;
    storage.activate(pool.clone(), funding.map(|source| source.account()));
    Ok(PendingStorageAllocation {
        storage,
        rows,
        published: false,
    })
}
impl<K: Ord + Send + 'static> PendingStorageAllocation<K> {
    /// Converts the accepted allowance into canonical backing after the producer
    /// establishes allocation success. Capacity and historical peak do not change.
    /// All identity checks precede the allocation-free commit under one lock.
    pub fn publish(&mut self) -> Result<(), WorkingMemoryError> {
        if self.published {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let registration = &self.storage.0;
        if registration.keys.is_empty() {
            self.published = true;
            return Ok(());
        }
        let pool = registration
            .pool
            .as_ref()
            .expect("accepted pending storage");
        let mut usage = pool
            .0
            .usage
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        let registry = usage
            .storage
            .get(&TypeId::of::<K>())
            .and_then(|value| value.downcast_ref::<Registry<K>>())
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        self.rows.clear();
        for key in &registration.keys {
            let (locator, entry) = registry
                .locate(key)
                .ok_or(WorkingMemoryError::IdentityMismatch)?;
            if !entry.pending_allocation
                || entry.owners != 1
                || entry.funding != registration.funding
                || entry.prepaid.is_some()
            {
                return Err(WorkingMemoryError::IdentityMismatch);
            }
            // Capacity was prepared before the original reservation transaction.
            assert!(self.rows.len() < self.rows.capacity());
            self.rows.push(PublicationRow {
                locator,
                bytes: entry.bytes,
                placement: Arc::clone(&entry.placement),
            });
        }
        for (slot, (domain, _)) in pool.topology().domains().enumerate() {
            let bytes = self
                .rows
                .iter()
                .filter(|row| row.placement.domains().contains(&domain))
                .try_fold(0u64, |sum, row| sum.checked_add(row.bytes))
                .ok_or(WorkingMemoryError::Overflow)?;
            usage.domains[slot]
                .reserved
                .checked_sub(bytes)
                .ok_or(WorkingMemoryError::Poisoned)?;
            usage.domains[slot]
                .registered
                .checked_add(bytes)
                .ok_or(WorkingMemoryError::Overflow)?;
        }
        // No lookup by a provider key, allocation, callback or destruction follows.
        for (slot, (domain, _)) in pool.topology().domains().enumerate() {
            let bytes = self
                .rows
                .iter()
                .filter(|row| row.placement.domains().contains(&domain))
                .fold(0u64, |sum, row| sum + row.bytes);
            usage.domains[slot].reserved -= bytes;
            usage.domains[slot].registered += bytes;
        }
        let registry = usage
            .storage
            .get_mut(&TypeId::of::<K>())
            .unwrap()
            .downcast_mut::<Registry<K>>()
            .unwrap();
        for row in &self.rows {
            registry.at_mut(row.locator).pending_allocation = false;
        }
        self.published = true;
        drop(usage);
        self.rows.clear();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::working_memory::InferenceExecutionIdentity;
    use eredu_core::{
        DomainMemoryRequirements, MemoryDeviceId, MemoryDomainDescription, MemoryDomainError,
        MemoryLimit, MemoryLimits, MemoryLocation, MemoryTopology,
    };

    fn pool(last_limit: MemoryLimit) -> MemoryLedger {
        let topology = Arc::new(
            MemoryTopology::new(vec![
                MemoryDomainDescription {
                    name: "host".into(),
                    locations: vec![MemoryLocation::Host],
                },
                MemoryDomainDescription {
                    name: "device".into(),
                    locations: vec![MemoryLocation::Device(MemoryDeviceId {
                        backend: "fixture",
                        ordinal: 0,
                    })],
                },
            ])
            .unwrap(),
        );
        let limits = MemoryLimits::resolve(
            &topology,
            topology.domains().enumerate().map(|(index, (id, _))| {
                (
                    id,
                    if index == 0 {
                        MemoryLimit::Unlimited
                    } else {
                        last_limit
                    },
                )
            }),
        )
        .unwrap();
        let baseline = DomainMemoryRequirements::zero(&topology);
        MemoryLedger::new(topology, limits, baseline).unwrap()
    }
    fn allocations(pool: &MemoryLedger, bytes: u64) -> [(u32, StorageAllocation); 2] {
        let device = pool.topology().domains().nth(1).unwrap().0;
        [
            (1, StorageAllocation::new(16, pool.host_placement_handle())),
            (
                2,
                StorageAllocation::new(
                    bytes,
                    Arc::new(MemoryPlacement::fixed(pool.topology(), device).unwrap()),
                ),
            ),
        ]
    }
    #[test]
    fn native_success_converts_reserved_to_registered_without_changing_charge_or_peak() {
        for limit in [MemoryLimit::Finite(64), MemoryLimit::Unlimited] {
            let pool = pool(limit);
            let baseline = pool.snapshot().unwrap();
            let funding = pool.prepare_storage_metadata().unwrap();
            let prepared = StoragePublicationLayout::pending(2)
                .unwrap()
                .prepare(&pool, &funding)
                .unwrap();
            let pin = StoragePublicationLayout::new(1)
                .unwrap()
                .prepare(&pool, &funding)
                .unwrap();
            let mut pending = prepared.reserve_storage(allocations(&pool, 64)).unwrap();
            let reserved = pool.snapshot().unwrap();
            assert_eq!(reserved.domains[1].outstanding_reservation_bytes, 64);
            assert_eq!(reserved.domains[1].registered_storage_bytes, 0);
            assert!(matches!(
                pin.pin_registered_storage([(2u32, 64)]),
                Err(WorkingMemoryError::IdentityMismatch)
            ));
            pending.publish().unwrap();
            let published = pool.snapshot().unwrap();
            for (before, after) in reserved.domains.iter().zip(&published.domains) {
                assert_eq!(before.current_charge_bytes, after.current_charge_bytes);
                assert_eq!(before.historical_peak_bytes, after.historical_peak_bytes);
            }
            assert_eq!(published.domains[1].outstanding_reservation_bytes, 0);
            assert_eq!(published.domains[1].registered_storage_bytes, 64);
            assert_eq!(
                pending.publish().unwrap_err(),
                WorkingMemoryError::IdentityMismatch
            );
            drop((pending, funding));
            for (before, after) in baseline
                .domains
                .iter()
                .zip(pool.snapshot().unwrap().domains)
            {
                assert_eq!(before.current_charge_bytes, after.current_charge_bytes);
            }
        }
    }
    #[test]
    fn native_allocation_failure_refunds_only_the_unpublished_grant() {
        let pool = pool(MemoryLimit::Finite(64));
        let funding = pool.prepare_storage_metadata().unwrap();
        let prepared = StoragePublicationLayout::pending(2)
            .unwrap()
            .prepare(&pool, &funding)
            .unwrap();
        let before = pool.snapshot().unwrap();
        let pending = prepared.reserve_storage(allocations(&pool, 64)).unwrap();
        drop(pending);
        let after = pool.snapshot().unwrap();
        for (before, after) in before.domains.iter().zip(&after.domains) {
            assert_eq!(before.current_charge_bytes, after.current_charge_bytes);
            assert_eq!(
                before.registered_storage_bytes,
                after.registered_storage_bytes
            );
            assert_eq!(
                before.outstanding_reservation_bytes,
                after.outstanding_reservation_bytes
            );
        }
        assert_eq!(after.domains[1].historical_peak_bytes, 64);
    }
    #[test]
    fn final_domain_refusal_preserves_all_domains_after_metadata_preparation() {
        let pool = pool(MemoryLimit::Finite(63));
        let funding = pool.prepare_storage_metadata().unwrap();
        let prepared = StoragePublicationLayout::pending(2)
            .unwrap()
            .prepare(&pool, &funding)
            .unwrap();
        let before = pool.snapshot().unwrap();
        assert!(matches!(
            prepared.reserve_storage(allocations(&pool, 64)),
            Err(WorkingMemoryError::Domain(
                MemoryDomainError::BudgetExceeded { .. }
            ))
        ));
        assert_eq!(before, pool.snapshot().unwrap());
    }
    fn assigned_admission(
        pool: &MemoryLedger,
        placement: &MemoryPlacement,
        preparations: usize,
    ) -> eredu_core::Admission {
        let publication = StoragePublicationLayout::<u32>::pending(1)
            .unwrap()
            .requested_bytes();
        let host = WorkingMemoryFundingScope::allocation_funding_control_bytes()
            .unwrap()
            .checked_add(MemoryLedger::storage_metadata_control_bytes().unwrap())
            .unwrap()
            .checked_add(
                publication
                    .checked_mul(preparations.try_into().unwrap())
                    .unwrap(),
            )
            .unwrap();
        let mut admission = crate::working_memory::memory_fixture::host_admission(pool, host);
        let workspace = admission.state.execution_workspace.as_mut().unwrap();
        let requirements = &mut workspace.physical_domains.as_mut().unwrap().activations;
        requirements.add_allocation(64, placement).unwrap();
        let total = requirements
            .iter()
            .map(|(_, charge)| charge.total().unwrap())
            .sum();
        workspace.activations =
            eredu_core::WorkspaceBound::bounded(total, "actual assigned fixture population");
        admission.incremental_required_bytes = Some(total);
        admission
    }

    #[test]
    fn assigned_birth_publication_and_failed_allocation_preserve_the_reserved_charge() {
        for limit in [MemoryLimit::Finite(64), MemoryLimit::Unlimited] {
            for candidate in [false, true] {
                let pool = pool(limit);
                let baseline = pool.snapshot().unwrap();
                let device = pool.topology().domains().nth(1).unwrap().0;
                let placement = Arc::new(if candidate {
                    MemoryPlacement::possible(
                        pool.topology(),
                        vec![pool.topology().host_domain(), device],
                        "actual finite managed fixture candidates".into(),
                    )
                    .unwrap()
                } else {
                    MemoryPlacement::fixed(pool.topology(), device).unwrap()
                });
                let admission = assigned_admission(&pool, &placement, 3);
                let reservation = pool
                    .reserve(&InferenceExecutionIdentity::default(), &admission)
                    .unwrap();
                let (reservation, run) = reservation.into_funding().unwrap();
                let mut scope = run.scope().unwrap();
                let funding = scope.allocation_funding().unwrap();
                fn thread_safe<T: Send + Sync>() {}
                thread_safe::<crate::working_memory::WorkingMemoryAllocationFunding>();
                let metadata = funding.prepare_storage_metadata().unwrap();
                let prepared = StoragePublicationLayout::<u32>::pending(1)
                    .unwrap()
                    .prepare(&pool, &metadata)
                    .unwrap();
                let retry = StoragePublicationLayout::<u32>::pending(1)
                    .unwrap()
                    .prepare(&pool, &metadata)
                    .unwrap();
                let stale = StoragePublicationLayout::<u32>::pending(1)
                    .unwrap()
                    .prepare(&pool, &metadata)
                    .unwrap();
                let before = pool.snapshot().unwrap();
                let pending = prepared
                    .reserve_storage_from(
                        &funding,
                        [(1, StorageAllocation::new(64, placement.clone()))],
                    )
                    .unwrap();
                assert_eq!(
                    before,
                    pool.snapshot().unwrap(),
                    "birth consumes assigned capacity, with no second charge"
                );
                assert!(pool.registered_allocation(&1u32).unwrap().is_none());
                drop(pending);
                assert_eq!(
                    before,
                    pool.snapshot().unwrap(),
                    "failed native allocation restores its original allowance"
                );
                let mut pending = retry
                    .reserve_storage_from(
                        &funding,
                        [(2, StorageAllocation::new(64, placement.clone()))],
                    )
                    .unwrap();
                pending.publish().unwrap();
                let published = pool.snapshot().unwrap();
                for (a, b) in before.domains.iter().zip(&published.domains) {
                    assert_eq!(a.current_charge_bytes, b.current_charge_bytes);
                    assert_eq!(a.historical_peak_bytes, b.historical_peak_bytes);
                    assert_eq!(
                        a.estimated_placement_allowance_bytes,
                        b.estimated_placement_allowance_bytes
                    );
                }
                assert_eq!(published.domains[1].registered_storage_bytes, 64);
                assert_eq!(published.domains[1].outstanding_reservation_bytes, 0);
                scope.certify().unwrap();
                let before_stale = pool.snapshot().unwrap();
                assert!(matches!(
                    stale.reserve_storage_from(
                        &funding,
                        [(3, StorageAllocation::new(0, placement))]
                    ),
                    Err(WorkingMemoryError::ExecutionFenced)
                ));
                assert_eq!(before_stale, pool.snapshot().unwrap());
                assert!(funding.prepare_storage_metadata().is_err());
                drop((funding, metadata, run, reservation));
                assert_eq!(pool.snapshot().unwrap().domains[1].current_charge_bytes, 64);
                drop(pending);
                let final_state = pool.snapshot().unwrap();
                assert_eq!(final_state.funding_accounts, 0);
                for (a, b) in baseline.domains.iter().zip(final_state.domains) {
                    assert_eq!(a.current_charge_bytes, b.current_charge_bytes);
                }
            }
        }
    }

    #[test]
    fn assigned_final_domain_rejection_and_foreign_scope_leave_every_balance_unchanged() {
        let pool = pool(MemoryLimit::Finite(64));
        let device = pool.topology().domains().nth(1).unwrap().0;
        let placement = Arc::new(MemoryPlacement::fixed(pool.topology(), device).unwrap());
        let reservation = pool
            .reserve(
                &InferenceExecutionIdentity::default(),
                &assigned_admission(&pool, &placement, 2),
            )
            .unwrap();
        let (reservation, run) = reservation.into_funding().unwrap();
        let mut scope = run.scope().unwrap();
        let funding = scope.allocation_funding().unwrap();
        let metadata = funding.prepare_storage_metadata().unwrap();
        let prepared = StoragePublicationLayout::<u32>::pending(1)
            .unwrap()
            .prepare(&pool, &metadata)
            .unwrap();
        let before = pool.snapshot().unwrap();
        assert!(matches!(
            prepared.reserve_storage_from(
                &funding,
                [(1, StorageAllocation::new(65, placement.clone()))]
            ),
            Err(WorkingMemoryError::DomainAllowanceExceeded {
                required_bytes: 65,
                available_bytes: 64,
                ..
            })
        ));
        assert_eq!(before, pool.snapshot().unwrap());
        let foreign = super::tests::pool(MemoryLimit::Unlimited);
        let prepared = StoragePublicationLayout::<u32>::pending(1)
            .unwrap()
            .fund(&foreign)
            .unwrap();
        let before = pool.snapshot().unwrap();
        assert!(matches!(
            prepared.reserve_storage_from(&funding, [(1, StorageAllocation::new(64, placement))]),
            Err(WorkingMemoryError::IdentityMismatch)
        ));
        assert_eq!(before, pool.snapshot().unwrap());
        scope.certify().unwrap();
        drop((funding, metadata, run, reservation));
        assert_eq!(pool.snapshot().unwrap().funding_accounts, 0);
    }

    #[test]
    fn unused_native_allocation_handle_retires_without_storage_publication() {
        let pool = pool(MemoryLimit::Finite(64));
        let baseline = pool.snapshot().unwrap();
        let placement =
            MemoryPlacement::fixed(pool.topology(), pool.topology().domains().nth(1).unwrap().0)
                .unwrap();
        let reservation = pool
            .reserve(
                &InferenceExecutionIdentity::default(),
                &assigned_admission(&pool, &placement, 1),
            )
            .unwrap();
        let (reservation, run) = reservation.into_funding().unwrap();
        let mut scope = run.scope().unwrap();
        let before = pool.snapshot().unwrap();
        let handle = scope.allocation_funding().unwrap();
        let alias = handle.clone();
        let after = pool.snapshot().unwrap();
        let controls = WorkingMemoryFundingScope::allocation_funding_control_bytes().unwrap();
        assert_eq!(
            after.domains[0].registered_storage_bytes - before.domains[0].registered_storage_bytes,
            controls
        );
        assert_eq!(
            after.domains[0].current_charge_bytes,
            before.domains[0].current_charge_bytes
        );
        scope.certify().unwrap();
        drop((run, reservation, handle));
        assert_eq!(pool.snapshot().unwrap().funding_accounts, 1);
        assert!(alias.prepare_storage_metadata().is_err());
        drop(alias);
        let retired = pool.snapshot().unwrap();
        assert_eq!(retired.funding_accounts, 0);
        assert_eq!(retired.reservations, 0);
        for (old, new) in baseline.domains.iter().zip(retired.domains) {
            assert_eq!(old.current_charge_bytes, new.current_charge_bytes);
        }
    }

    #[test]
    fn one_scope_cannot_borrow_another_live_scopes_allocation_authority() {
        let pool = pool(MemoryLimit::Finite(64));
        let device = pool.topology().domains().nth(1).unwrap().0;
        let placement = Arc::new(MemoryPlacement::fixed(pool.topology(), device).unwrap());
        let reservation = pool
            .reserve(
                &InferenceExecutionIdentity::default(),
                &assigned_admission(&pool, &placement, 1),
            )
            .unwrap();
        let (reservation, run) = reservation.into_funding().unwrap();
        let mut first = run.scope().unwrap();
        let second = run.scope().unwrap();
        let funding = first.allocation_funding().unwrap();
        let metadata = funding.prepare_storage_metadata().unwrap();
        let prepared = StoragePublicationLayout::<u32>::pending(1)
            .unwrap()
            .prepare(&pool, &metadata)
            .unwrap();
        first.certify().unwrap();
        let before = pool.snapshot().unwrap();
        assert!(matches!(
            prepared.reserve_storage_from(&funding, [(1, StorageAllocation::new(64, placement))]),
            Err(WorkingMemoryError::ExecutionFenced)
        ));
        assert_eq!(before, pool.snapshot().unwrap());
        second.certify().unwrap();
        drop((funding, metadata, run, reservation));
        assert_eq!(pool.snapshot().unwrap().funding_accounts, 0);
    }

    #[test]
    fn assigned_fixed_birth_restores_candidate_allowance_or_retires_after_scope_completion() {
        let pool = pool(MemoryLimit::Finite(64));
        let baseline = pool.snapshot().unwrap();
        let device = pool.topology().domains().nth(1).unwrap().0;
        let possible = MemoryPlacement::possible(
            pool.topology(),
            vec![pool.topology().host_domain(), device],
            "allocator alternatives".into(),
        )
        .unwrap();
        let fixed = Arc::new(MemoryPlacement::fixed(pool.topology(), device).unwrap());
        let reservation = pool
            .reserve(
                &InferenceExecutionIdentity::default(),
                &assigned_admission(&pool, &possible, 2),
            )
            .unwrap();
        let (reservation, run) = reservation.into_funding().unwrap();
        let mut scope = run.scope().unwrap();
        let funding = scope.allocation_funding().unwrap();
        let metadata = funding.prepare_storage_metadata().unwrap();
        let first = StoragePublicationLayout::<u32>::pending(1)
            .unwrap()
            .prepare(&pool, &metadata)
            .unwrap();
        let second = StoragePublicationLayout::<u32>::pending(1)
            .unwrap()
            .prepare(&pool, &metadata)
            .unwrap();
        let before = pool.snapshot().unwrap();
        let pending = first
            .reserve_storage_from(&funding, [(1, StorageAllocation::new(64, fixed.clone()))])
            .unwrap();
        let allocated = pool.snapshot().unwrap();
        assert_eq!(
            allocated.domains[1].current_charge_bytes,
            before.domains[1].current_charge_bytes
        );
        assert_eq!(allocated.domains[1].estimated_placement_allowance_bytes, 0);
        drop(pending);
        assert_eq!(before, pool.snapshot().unwrap());
        let pending = second
            .reserve_storage_from(&funding, [(2, StorageAllocation::new(64, fixed))])
            .unwrap();
        scope.certify().unwrap();
        drop((funding, metadata, run, reservation));
        let closing = pool.snapshot().unwrap();
        assert_eq!(closing.domains[1].outstanding_reservation_bytes, 64);
        assert_eq!(closing.domains[1].registered_storage_bytes, 0);
        drop(pending);
        let retired = pool.snapshot().unwrap();
        assert_eq!(retired.funding_accounts, 0);
        for (a, b) in baseline.domains.iter().zip(retired.domains) {
            assert_eq!(a.current_charge_bytes, b.current_charge_bytes);
        }
    }

    #[test]
    fn concurrent_native_births_compete_for_one_assigned_account_without_new_scopes() {
        let pool = pool(MemoryLimit::Finite(64));
        let device = pool.topology().domains().nth(1).unwrap().0;
        let placement = Arc::new(MemoryPlacement::fixed(pool.topology(), device).unwrap());
        let reservation = pool
            .reserve(
                &InferenceExecutionIdentity::default(),
                &assigned_admission(&pool, &placement, 2),
            )
            .unwrap();
        let (reservation, run) = reservation.into_funding().unwrap();
        let mut scope = run.scope().unwrap();
        let funding = scope.allocation_funding().unwrap();
        let metadata = funding.prepare_storage_metadata().unwrap();
        let a = StoragePublicationLayout::<u32>::pending(1)
            .unwrap()
            .prepare(&pool, &metadata)
            .unwrap();
        let b = StoragePublicationLayout::<u32>::pending(1)
            .unwrap()
            .prepare(&pool, &metadata)
            .unwrap();
        let before = pool.snapshot().unwrap();
        let barrier = Arc::new(std::sync::Barrier::new(2));
        let workers: Vec<_> = [a, b]
            .into_iter()
            .enumerate()
            .map(|(i, prepared)| {
                let source = funding.clone();
                let placement = placement.clone();
                let barrier = barrier.clone();
                std::thread::spawn(move || {
                    barrier.wait();
                    let result = prepared.reserve_storage_from(
                        &source,
                        [(
                            u32::try_from(i).unwrap(),
                            StorageAllocation::new(64, placement),
                        )],
                    );
                    // A successful pending allocation stays live through both attempts.
                    barrier.wait();
                    result
                })
            })
            .collect();
        let results: Vec<_> = workers
            .into_iter()
            .map(|worker| worker.join().unwrap())
            .collect();
        assert_eq!(results.iter().filter(|result| result.is_ok()).count(), 1);
        assert!(results.iter().any(|result| matches!(
            result,
            Err(WorkingMemoryError::DomainAllowanceExceeded {
                required_bytes: 64,
                available_bytes: 0,
                ..
            })
        )));
        assert_eq!(before, pool.snapshot().unwrap());
        drop(results);
        assert_eq!(before, pool.snapshot().unwrap());
        scope.certify().unwrap();
        drop((funding, metadata, run, reservation));
        assert_eq!(pool.snapshot().unwrap().funding_accounts, 0);
    }

    #[test]
    fn abandoned_native_scope_fences_its_allocation_handle_and_preserves_quarantine() {
        let pool = pool(MemoryLimit::Finite(64));
        let device = pool.topology().domains().nth(1).unwrap().0;
        let placement = Arc::new(MemoryPlacement::fixed(pool.topology(), device).unwrap());
        let reservation = pool
            .reserve(
                &InferenceExecutionIdentity::default(),
                &assigned_admission(&pool, &placement, 1),
            )
            .unwrap();
        let (reservation, run) = reservation.into_funding().unwrap();
        let mut scope = run.scope().unwrap();
        let funding = scope.allocation_funding().unwrap();
        let metadata = funding.prepare_storage_metadata().unwrap();
        let prepared = StoragePublicationLayout::<u32>::pending(1)
            .unwrap()
            .prepare(&pool, &metadata)
            .unwrap();
        drop(scope);
        let before = pool.snapshot().unwrap();
        assert!(matches!(
            prepared.reserve_storage_from(&funding, [(1, StorageAllocation::new(64, placement))]),
            Err(WorkingMemoryError::ExecutionFenced)
        ));
        assert_eq!(before, pool.snapshot().unwrap());
        drop((funding, metadata, run, reservation));
        assert_eq!(
            pool.snapshot().unwrap().domains[1].outstanding_reservation_bytes,
            64
        );
        assert!(pool.acquire_unquoted().is_err());
    }
}
