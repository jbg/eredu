use super::*;
use eredu_core::cache::CacheRepresentation;
use eredu_nn::workspace::{
    WorkspaceMechanisms, WorkspaceMetadataAccount, WorkspaceMetadataFunding,
    WorkspaceMetadataFundingError, WorkspaceOperation, WorkspaceOperationBound,
};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};

#[derive(Debug, Default)]
struct AccountState {
    remaining: Mutex<usize>,
    retired: AtomicBool,
}
#[derive(Debug)]
struct Account(Arc<AccountState>);
impl WorkspaceMetadataAccount for Account {
    fn reserve_metadata(&self, bytes: usize) -> Result<(), WorkspaceMetadataFundingError> {
        let mut remaining = self.0.remaining.lock().unwrap();
        *remaining =
            remaining
                .checked_sub(bytes)
                .ok_or(WorkspaceMetadataFundingError::Capacity {
                    required: bytes as u64,
                    available: *remaining as u64,
                })?;
        Ok(())
    }
}
impl Drop for Account {
    fn drop(&mut self) {
        self.0.retired.store(true, Ordering::SeqCst);
    }
}
#[derive(Debug)]
struct NoEquations;
impl WorkspaceMechanisms for NoEquations {
    fn operation_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, Error> {
        panic!("catalog storage runs no equations")
    }
}
fn context() -> (WorkspaceContext, Arc<AccountState>) {
    let state = Arc::new(AccountState {
        remaining: Mutex::new(usize::MAX),
        retired: AtomicBool::new(false),
    });
    let funding = WorkspaceMetadataFunding::new(Account(state.clone())).unwrap();
    (
        WorkspaceContext::new_with_metadata_funding(NoEquations, funding).unwrap(),
        state,
    )
}
fn id(start: i64) -> CacheBlockId {
    CacheBlockId {
        session_id: 9,
        global_layer: 7,
        representation: CacheRepresentation::KeyValue,
        start,
        end: start + 2,
        rank: None,
    }
}

#[test]
fn prepared_catalogs_preserve_canonical_leases_atomic_refusal_and_retirement() {
    let (ctx, account) = context();
    let mut table = CacheRecordTable::new();
    table.insert(3usize, 30usize);
    let destination = PreparedCacheTable::prepare(2, &ctx).unwrap();
    drop(table.install(destination).unwrap());
    *account.remaining.lock().unwrap() = 0;
    assert_eq!(table.insert_prepared(1, 10), Ok(None));
    assert_eq!(table.insert_prepared(3, 31), Ok(Some(30)));
    assert_eq!(
        table.insert_prepared(5, 50),
        Err((CacheTableCapacityError::Exhausted, 5, 50))
    );
    assert_eq!(
        table.iter().map(|(k, v)| (*k, *v)).collect::<Vec<_>>(),
        [(1, 10), (3, 31)]
    );
    drop(ctx);
    assert!(!account.retired.load(Ordering::SeqCst));
    let (next, next_account) = context();
    let old = table
        .install(PreparedCacheTable::prepare(3, &next).unwrap())
        .unwrap();
    assert!(old.is_empty());
    assert!(!account.retired.load(Ordering::SeqCst));
    drop(old); // A backend can place this after its manager guard.
    assert!(account.retired.load(Ordering::SeqCst));
    drop(next);
    assert!(!next_account.retired.load(Ordering::SeqCst));
    drop(table);
    assert!(next_account.retired.load(Ordering::SeqCst));

    let (ctx, account) = context();
    let mut lifecycle = CacheBlockLifecycle::new();
    lifecycle.insert(id(0), true).unwrap();
    lifecycle.acquire(&id(0)).unwrap();
    lifecycle.set_tail(7, MutableCacheTail { bytes: 16, end: 3 });
    let clock = lifecycle.access_clock;
    let rejected = lifecycle
        .install_storage(PreparedCacheLifecycle::prepare(1, 0, &ctx).unwrap())
        .unwrap_err();
    assert_eq!(lifecycle.lease_count(&id(0)).unwrap(), 1);
    assert_eq!(lifecycle.access_clock, clock);
    drop(rejected);
    drop(
        lifecycle
            .install_storage(PreparedCacheLifecycle::prepare(2, 1, &ctx).unwrap())
            .unwrap(),
    );
    *account.remaining.lock().unwrap() = 0;
    assert!(lifecycle.is_protected_prefix(&id(0)).unwrap());
    assert_eq!(lifecycle.lease_count(&id(0)).unwrap(), 1);
    lifecycle.insert_prepared(id(2), false).unwrap();
    let clock = lifecycle.access_clock;
    assert!(matches!(
        lifecycle.insert_prepared(id(4), false),
        Err(CacheLifecycleError::Capacity(
            CacheTableCapacityError::Exhausted
        ))
    ));
    assert_eq!(lifecycle.access_clock, clock);
    assert!(matches!(
        lifecycle.replace_prepared(&[(id(0), 0)], None, 7, MutableCacheTail::default()),
        Err(CacheLifecycleError::UnexpectedLeaseCount { .. })
    ));
    assert!(matches!(
        lifecycle.replace_prepared(
            &[(id(2), 0)],
            Some((id(4), false)),
            8,
            MutableCacheTail::default()
        ),
        Err(CacheLifecycleError::Capacity(
            CacheTableCapacityError::Exhausted
        ))
    ));
    assert_eq!(lifecycle.access_clock, clock);
    assert_eq!(lifecycle.lease_count(&id(2)).unwrap(), 0);
    assert_eq!(
        lifecycle.tail(7),
        Some(MutableCacheTail { bytes: 16, end: 3 })
    );
    lifecycle.release(&id(0)).unwrap();
    lifecycle
        .replace_prepared(
            &[(id(2), 0)],
            Some((id(4), false)),
            7,
            MutableCacheTail { bytes: 8, end: 6 },
        )
        .unwrap();
    assert_eq!(lifecycle.lease_count(&id(4)).unwrap(), 0);
    lifecycle.clear().unwrap();
    lifecycle.insert_prepared(id(6), false).unwrap();
    lifecycle
        .set_tail_prepared(7, MutableCacheTail { bytes: 4, end: 9 })
        .unwrap();
    drop(ctx);
    assert!(!account.retired.load(Ordering::SeqCst));
    drop(lifecycle);
    assert!(account.retired.load(Ordering::SeqCst));
}

impl eredu_nn::workspace::WorkspaceFactMechanisms for NoEquations {
    type Error = std::convert::Infallible;
    fn operation_facts(
        &self,
        _: eredu_nn::workspace::WorkspaceOperationView<'_>,
    ) -> Result<Option<eredu_nn::workspace::WorkspaceOperationFacts>, Self::Error> {
        panic!("catalog preparation cannot quote equations")
    }
    fn write_operation_facts(
        &self,
        _: eredu_nn::workspace::WorkspaceOperationView<'_>,
        _: eredu_nn::workspace::WorkspaceEffectDestination<'_>,
    ) -> Result<Option<eredu_nn::workspace::WorkspaceOperationFacts>, Self::Error> {
        panic!("catalog preparation cannot emit equations")
    }
    fn host_facts(
        &self,
        _: eredu_nn::workspace::WorkspaceOperationView<'_>,
    ) -> Result<Option<eredu_nn::workspace::WorkspaceHostFacts>, Self::Error> {
        panic!("catalog preparation cannot quote host equations")
    }
    fn write_host_facts(
        &self,
        _: eredu_nn::workspace::WorkspaceOperationView<'_>,
        _: eredu_nn::workspace::WorkspaceHostDestination<'_>,
    ) -> Result<Option<eredu_nn::workspace::WorkspaceHostFacts>, Self::Error> {
        panic!("catalog preparation cannot emit host equations")
    }
}

#[test]
fn prepared_occupancy_moves_only_admitted_bytes_and_retains_partial_copy_custody() {
    use crate::cache::{CachePoolError, CachePoolLimits, CachePoolUsage, CacheResidencyPool};
    let (ctx, account) = context();
    let pool = CacheResidencyPool::new(CachePoolLimits::new(40, 0, 1, 0).unwrap());
    let source = pool.register_manager(7).unwrap();
    let destination = pool
        .prepare_manager_registration(&ctx)
        .unwrap()
        .register(8)
        .unwrap();
    let usage = |device_bytes| CachePoolUsage {
        device_bytes,
        ..CachePoolUsage::default()
    };
    pool.update_manager(7, usage(8)).unwrap();
    let refused = pool
        .prepare_reservation(&ctx)
        .unwrap()
        .reserve(usage(33))
        .unwrap_err();
    assert!(matches!(
        refused.cause(),
        CachePoolError::BudgetExceeded {
            required: 41,
            budget: 40,
            ..
        }
    ));
    assert_eq!(pool.report().unwrap().current_device_bytes, 8);
    let mut reserved = pool
        .prepare_reservation(&ctx)
        .unwrap()
        .reserve(usage(12))
        .unwrap();
    assert_eq!(pool.report().unwrap().current_device_bytes, 20);
    let foreign_pool = CacheResidencyPool::new(pool.limits());
    let foreign = foreign_pool.register_manager(8).unwrap();
    assert_eq!(
        reserved.publish_to_manager(&foreign, usage(6)),
        Err(CachePoolError::ForeignMembership)
    );
    reserved.publish_to_manager(&destination, usage(6)).unwrap();
    assert_eq!(pool.report().unwrap().current_device_bytes, 20);
    assert!(reserved.publish_to_manager(&destination, usage(5)).is_err());
    assert!(
        reserved
            .publish_to_manager(&destination, usage(13))
            .is_err()
    );
    assert_eq!(pool.report().unwrap().current_device_bytes, 20);
    // Partial numerical failure discards only the unconsumed reservation. The
    // exact membership continues to own the six already-published bytes.
    drop(reserved);
    assert_eq!(pool.report().unwrap().current_device_bytes, 14);
    drop(source);
    assert_eq!(pool.report().unwrap().current_device_bytes, 6);
    drop((ctx, pool, foreign, foreign_pool));
    assert!(!account.retired.load(Ordering::SeqCst));
    drop(destination);
    // The refused preparation owns the same pool/storage and its paid H.
    assert!(!account.retired.load(Ordering::SeqCst));
    drop(refused);
    assert!(account.retired.load(Ordering::SeqCst));
}

#[test]
fn empty_pool_tables_retire_copy_funding_after_unlock_and_preserve_live_rows() {
    use crate::cache::{CachePoolLimits, CachePoolUsage, CacheResidencyPool};
    #[derive(Debug)]
    struct PoolAccount {
        account: Account,
        pool: CacheResidencyPool,
        unlocked: Arc<AtomicBool>,
    }
    impl WorkspaceMetadataAccount for PoolAccount {
        fn reserve_metadata(&self, bytes: usize) -> Result<(), WorkspaceMetadataFundingError> {
            self.account.reserve_metadata(bytes)
        }
    }
    impl Drop for PoolAccount {
        fn drop(&mut self) {
            self.unlocked
                .store(self.pool.state.try_lock().is_ok(), Ordering::SeqCst);
        }
    }
    for membership_first in [false, true] {
        let pool = CacheResidencyPool::new(CachePoolLimits::new(40, 0, 1, 0).unwrap());
        let account = Arc::new(AccountState {
            remaining: Mutex::new(usize::MAX),
            retired: AtomicBool::new(false),
        });
        let unlocked = Arc::new(AtomicBool::new(false));
        let funding = WorkspaceMetadataFunding::new(PoolAccount {
            account: Account(account.clone()),
            pool: pool.clone(),
            unlocked: unlocked.clone(),
        })
        .unwrap();
        let ctx = WorkspaceContext::new_with_metadata_funding(NoEquations, funding).unwrap();
        let first = pool
            .prepare_manager_registration(&ctx)
            .unwrap()
            .register(1)
            .unwrap();
        let second = pool
            .prepare_manager_registration(&ctx)
            .unwrap()
            .register(2)
            .unwrap();
        let usage = |device_bytes| CachePoolUsage {
            device_bytes,
            ..CachePoolUsage::default()
        };
        let held = pool
            .prepare_reservation(&ctx)
            .unwrap()
            .reserve(usage(8))
            .unwrap();
        let newer = pool
            .prepare_reservation(&ctx)
            .unwrap()
            .reserve(usage(12))
            .unwrap();
        drop((ctx, second, newer));
        assert!(!account.retired.load(Ordering::SeqCst));
        assert_eq!(pool.report().unwrap().managers, 1);
        assert_eq!(held.remaining_usage().unwrap(), usage(8));
        assert_eq!(pool.report().unwrap().current_device_bytes, 8);
        if membership_first {
            drop(first);
            assert!(!account.retired.load(Ordering::SeqCst));
            assert_eq!(pool.report().unwrap().current_device_bytes, 8);
            drop(held);
        } else {
            drop(held);
            assert!(!account.retired.load(Ordering::SeqCst));
            assert_eq!(pool.report().unwrap().managers, 1);
            drop(first);
        }
        assert!(account.retired.load(Ordering::SeqCst));
        assert!(unlocked.load(Ordering::SeqCst));
        let report = pool.report().unwrap();
        assert_eq!(report.managers, 0);
        assert_eq!(report.current_device_bytes, 0);
        assert_eq!(report.peak_device_bytes, 20);
        // The same externally retained process pool remains usable.
        let (next, next_account) = context();
        let member = pool
            .prepare_manager_registration(&next)
            .unwrap()
            .register(3)
            .unwrap();
        drop(next);
        assert!(!next_account.retired.load(Ordering::SeqCst));
        drop(member);
        assert!(next_account.retired.load(Ordering::SeqCst));
    }
}
