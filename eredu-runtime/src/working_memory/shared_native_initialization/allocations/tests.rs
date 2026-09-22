use super::*;
use crate::working_memory::{
    PendingStorageAllocation, StorageAllocation, StoragePublicationLayout,
};
use eredu_core::{
    MemoryDomainDescription, MemoryLimit, MemoryLimits, MemoryLocation, MemoryPlacement,
    MemoryTopology,
};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

#[derive(Debug)]
struct Constructor {
    pool: MemoryLedger,
    temporary: DomainMemoryRequirements,
    placement: Arc<MemoryPlacement>,
    calls: Arc<AtomicUsize>,
    abandon: bool,
    retain_native: bool,
}
#[derive(Debug)]
struct Output {
    allocation: Option<PendingStorageAllocation<u32>>,
    custody: SharedNativeInitializationCustody,
}
impl SharedNativeInitializer for Constructor {
    type Output = Output;
    type Error = WorkingMemoryError;
    fn required_storage_bytes(&self) -> Result<usize, WorkingMemoryError> {
        Ok(size_of::<Output>())
    }
    fn temporary_allocation_requirements(&self) -> Option<&DomainMemoryRequirements> {
        Some(&self.temporary)
    }
    fn initialize(
        self,
        mut custody: SharedNativeInitializationCustody,
    ) -> Result<Output, WorkingMemoryError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        custody.validate_pool(&self.pool)?;
        let mut scope = custody.take_allocation_scope()?;
        assert!(matches!(
            custody.take_allocation_scope(),
            Err(WorkingMemoryError::AlreadyStarted)
        ));
        if self.abandon {
            drop(scope);
            return Ok(Output {
                allocation: None,
                custody,
            });
        }
        let funding = scope.allocation_funding()?;
        let metadata = funding.prepare_storage_metadata().map_err(funding_error)?;
        let prepared =
            StoragePublicationLayout::<u32>::pending(1)?.prepare(&self.pool, &metadata)?;
        let before = self.pool.snapshot()?;
        let mut allocation = prepared
            .reserve_storage_from(&funding, [(41, StorageAllocation::new(64, self.placement))])?;
        assert_eq!(self.pool.snapshot()?, before);
        allocation.publish()?;
        let published = self.pool.snapshot()?;
        for (before, after) in before.domains.iter().zip(&published.domains) {
            assert_eq!(before.current_charge_bytes, after.current_charge_bytes);
            assert_eq!(before.historical_peak_bytes, after.historical_peak_bytes);
        }
        scope.certify()?;
        assert!(funding.prepare_storage_metadata().is_err());
        Ok(Output {
            allocation: self.retain_native.then_some(allocation),
            custody,
        })
    }
}
fn topology(unified: bool) -> Arc<MemoryTopology> {
    let device = MemoryLocation::Device(eredu_core::MemoryDeviceId {
        backend: "constructor-test",
        ordinal: 0,
    });
    Arc::new(
        MemoryTopology::new(if unified {
            vec![MemoryDomainDescription {
                name: "shared".into(),
                locations: vec![MemoryLocation::Host, device],
            }]
        } else {
            vec![
                MemoryDomainDescription {
                    name: "host".into(),
                    locations: vec![MemoryLocation::Host],
                },
                MemoryDomainDescription {
                    name: "device".into(),
                    locations: vec![device],
                },
            ]
        })
        .unwrap(),
    )
}
fn constructor(pool: &MemoryLedger) -> Constructor {
    let domain = pool.topology().domains().last().unwrap().0;
    let placement = Arc::new(MemoryPlacement::fixed(pool.topology(), domain).unwrap());
    let mut temporary = DomainMemoryRequirements::zero(pool.topology());
    let host = WorkingMemoryFundingScope::allocation_funding_control_bytes()
        .unwrap()
        .checked_add(MemoryLedger::storage_metadata_control_bytes().unwrap())
        .unwrap()
        .checked_add(
            StoragePublicationLayout::<u32>::pending(1)
                .unwrap()
                .requested_bytes(),
        )
        .unwrap();
    temporary
        .add_allocation(host, &pool.host_placement_handle())
        .unwrap();
    temporary.add_allocation(64, &placement).unwrap();
    Constructor {
        pool: pool.clone(),
        temporary,
        placement,
        calls: Arc::new(AtomicUsize::new(0)),
        abandon: false,
        retain_native: true,
    }
}
fn ledger(topology: Arc<MemoryTopology>, finite: bool, short: bool) -> MemoryLedger {
    let empty = DomainMemoryRequirements::zero(&topology);
    let probe = MemoryLedger::new(
        topology.clone(),
        MemoryLimits::unlimited(&topology),
        empty.clone(),
    )
    .unwrap();
    let requirements = probe
        .shared_native_initialization_requirements(&constructor(&probe))
        .unwrap();
    let baseline = probe.snapshot().unwrap();
    let last = topology.domains().last().unwrap().0;
    let limits = MemoryLimits::resolve(
        &topology,
        topology.domains().map(|(domain, _)| {
            let initial = baseline
                .domains
                .iter()
                .find(|row| row.domain == domain)
                .unwrap()
                .current_charge_bytes;
            let amount = initial
                .checked_add(requirements.get(domain).unwrap().total().unwrap())
                .unwrap();
            (
                domain,
                if finite {
                    MemoryLimit::Finite(amount - u64::from(short && domain == last))
                } else {
                    MemoryLimit::Unlimited
                },
            )
        }),
    )
    .unwrap();
    MemoryLedger::new(topology, limits, empty).unwrap()
}
#[test]
fn shared_constructor_allocations_use_one_domain_admission_and_retire_independently() {
    for unified in [true, false] {
        for finite in [true, false] {
            let pool = ledger(topology(unified), finite, false);
            let initial = pool.snapshot().unwrap();
            let initialized = pool.initialize_shared_native(constructor(&pool)).unwrap();
            assert!(initialized.output().allocation.is_some());
            assert!(matches!(
                pool.acquire_unquoted(),
                Err(WorkingMemoryError::ReservedWorkActive)
            ));
            let account_alias = initialized.output().custody.0.share();
            drop(initialized);
            let after_allocation = pool.snapshot().unwrap();
            assert_eq!(after_allocation.funding_accounts, 1);
            assert_eq!(after_allocation.reservations, 0);
            drop(pool.acquire_unquoted().unwrap());
            assert!(pool.registered_allocation(&41u32).unwrap().is_none());
            for domain in &after_allocation.domains {
                assert_eq!(
                    domain.registered_storage_bytes,
                    domain.registry_metadata_bytes
                );
            }
            assert!(
                after_allocation.domains[0].current_charge_bytes
                    > initial.domains[0].current_charge_bytes
            );
            drop(account_alias);
            let retired = pool.snapshot().unwrap();
            assert_eq!(retired.funding_accounts, 0);
            assert_eq!(retired.reservations, 0);
            for (before, after) in initial.domains.iter().zip(&retired.domains) {
                assert_eq!(before.current_charge_bytes, after.current_charge_bytes);
            }
        }
    }
}

#[test]
fn completed_shared_constructor_keeps_fixed_charge_without_request_exclusion() {
    for unified in [true, false] {
        for finite in [true, false] {
            let pool = ledger(topology(unified), finite, false);
            let initial = pool.snapshot().unwrap();
            let identifier = pool.0.usage.lock().unwrap().next_funding;
            let mut plan = constructor(&pool);
            plan.retain_native = false;
            let initialized = pool.initialize_shared_native(plan).unwrap();
            let alias = initialized.output().custody.0.share();
            assert!(initialized.output().allocation.is_none());
            let completed = pool.snapshot().unwrap();
            assert_eq!(pool.0.usage.lock().unwrap().next_funding, identifier + 1);
            assert_eq!(completed.reservations, 0);
            assert_eq!(completed.funding_accounts, 1);
            assert!(
                completed.domains[0].current_charge_bytes > initial.domains[0].current_charge_bytes
            );
            if !unified {
                assert_eq!(
                    completed.domains[1].current_charge_bytes,
                    initial.domains[1].current_charge_bytes
                );
            }
            let unquoted = pool.acquire_unquoted().unwrap();
            // A new native lease does not consume, refund or replace the fixed
            // account, and its original physical ceiling still applies.
            assert_eq!(pool.snapshot().unwrap().funding_accounts, 1);
            assert_eq!(pool.0.usage.lock().unwrap().next_funding, identifier + 1);
            if finite {
                let metadata = pool.prepare_storage_metadata().unwrap();
                let before = pool.snapshot().unwrap();
                assert!(metadata.prepare_host_owner(usize::MAX / 2).is_err());
                assert_eq!(pool.snapshot().unwrap(), before);
                drop(metadata);
            }
            drop(unquoted);
            drop(initialized);
            for (before, after) in completed
                .domains
                .iter()
                .zip(&pool.snapshot().unwrap().domains)
            {
                assert_eq!(before.current_charge_bytes, after.current_charge_bytes);
                assert_eq!(
                    before.registered_storage_bytes,
                    after.registered_storage_bytes
                );
            }
            drop(alias);
            let retired = pool.snapshot().unwrap();
            assert_eq!(retired.reservations, 0);
            assert_eq!(retired.funding_accounts, 0);
            for (before, after) in initial.domains.iter().zip(&retired.domains) {
                assert_eq!(before.current_charge_bytes, after.current_charge_bytes);
            }
        }
    }
}

#[test]
fn escaped_allocation_cannot_replace_retired_constructor_fixed_identity() {
    let pool = ledger(topology(false), false, false);
    let initial = pool.snapshot().unwrap();
    let plan = constructor(&pool);
    let fixed = MemoryLedger::shared_native_initialization_required_bytes(&plan).unwrap();
    let (account, mut scope) = pool
        .admit_shared_native_allocations(fixed, &plan.temporary)
        .unwrap();
    let before_duplicate = pool.snapshot().unwrap();
    let identifier = pool.0.usage.lock().unwrap().next_funding;
    assert!(StorageMetadataFunding::from_shared_constructor_account(&pool, scope.id).is_err());
    assert_eq!(pool.snapshot().unwrap(), before_duplicate);
    assert_eq!(pool.0.usage.lock().unwrap().next_funding, identifier);
    let funding = scope.allocation_funding().unwrap();
    let metadata = funding.prepare_storage_metadata().unwrap();
    let prepared = StoragePublicationLayout::<u32>::pending(1)
        .unwrap()
        .prepare(&pool, &metadata)
        .unwrap();
    let mut allocation = prepared
        .reserve_storage_from(&funding, [(43, StorageAllocation::new(64, plan.placement))])
        .unwrap();
    allocation.publish().unwrap();
    scope.certify().unwrap();
    drop((metadata, funding));
    drop(account);
    assert_eq!(pool.snapshot().unwrap().reservations, 1);
    assert!(matches!(
        pool.acquire_unquoted(),
        Err(WorkingMemoryError::ReservedWorkActive)
    ));
    drop(allocation);
    let retired = pool.snapshot().unwrap();
    assert_eq!(retired.reservations, 0);
    assert_eq!(retired.funding_accounts, 0);
    for (before, after) in initial.domains.iter().zip(&retired.domains) {
        assert_eq!(before.current_charge_bytes, after.current_charge_bytes);
    }
}
#[test]
fn shared_constructor_final_domain_refusal_is_atomic_and_foreign_requirements_reject() {
    for unified in [true, false] {
        let pool = ledger(topology(unified), true, true);
        let plan = constructor(&pool);
        let calls = plan.calls.clone();
        let before = pool.snapshot().unwrap();
        let identifier = pool.0.usage.lock().unwrap().next_funding;
        let error = pool.initialize_shared_native(plan).unwrap_err();
        assert!(matches!(
            error.accounting_failure(),
            Some(WorkingMemoryError::Domain(
                eredu_core::MemoryDomainError::BudgetExceeded { .. }
            ))
        ));
        assert_eq!(pool.snapshot().unwrap(), before);
        assert_eq!(pool.0.usage.lock().unwrap().next_funding, identifier);
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }
    let pool = ledger(topology(false), false, false);
    let foreign = ledger(topology(false), false, false);
    let mut plan = constructor(&pool);
    plan.temporary = constructor(&foreign).temporary;
    let before = pool.snapshot().unwrap();
    assert!(pool.initialize_shared_native(plan).is_err());
    assert_eq!(pool.snapshot().unwrap(), before);
}
#[test]
fn shared_constructor_abandoned_scope_keeps_quarantine_under_unlimited_limits() {
    let pool = ledger(topology(false), false, false);
    let mut plan = constructor(&pool);
    plan.abandon = true;
    let initial = pool.snapshot().unwrap();
    drop(pool.initialize_shared_native(plan).unwrap());
    let abandoned = pool.snapshot().unwrap();
    assert_eq!(abandoned.funding_accounts, 1);
    assert!(abandoned.domains[1].current_charge_bytes > initial.domains[1].current_charge_bytes);
    assert!(pool.acquire_unquoted().is_err());
}

#[test]
fn shared_constructor_unquoted_exclusion_and_unlimited_overflow_precede_native_work() {
    let pool = ledger(topology(false), false, false);
    let unquoted = pool.acquire_unquoted().unwrap();
    let plan = constructor(&pool);
    let calls = plan.calls.clone();
    let before = pool.snapshot().unwrap();
    let error = pool.initialize_shared_native(plan).unwrap_err();
    assert!(matches!(
        error.accounting_failure(),
        Some(WorkingMemoryError::UnknownBound)
    ));
    assert_eq!(pool.snapshot().unwrap(), before);
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    drop((error, unquoted));

    let topology = topology(false);
    let placement =
        MemoryPlacement::fixed(&topology, topology.domains().last().unwrap().0).unwrap();
    let mut baseline = DomainMemoryRequirements::zero(&topology);
    baseline.add_allocation(1, &placement).unwrap();
    let pool = MemoryLedger::new(
        topology.clone(),
        MemoryLimits::unlimited(&topology),
        baseline,
    )
    .unwrap();
    let mut plan = constructor(&pool);
    plan.temporary = DomainMemoryRequirements::zero(&topology);
    plan.temporary.add_allocation(u64::MAX, &placement).unwrap();
    let calls = plan.calls.clone();
    let before = pool.snapshot().unwrap();
    let identifier = pool.0.usage.lock().unwrap().next_funding;
    let error = pool.initialize_shared_native(plan).unwrap_err();
    assert!(matches!(
        error.accounting_failure(),
        Some(WorkingMemoryError::Domain(
            eredu_core::MemoryDomainError::Overflow
        ))
    ));
    assert_eq!(pool.snapshot().unwrap(), before);
    assert_eq!(pool.0.usage.lock().unwrap().next_funding, identifier);
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}
