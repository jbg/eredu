use super::*;
use crate::working_memory::{WorkingMemoryReservation, WorkingMemoryStorage};
use eredu_core::{
    cache::LayerCachePolicy, Admission, EstimationCompleteness, ExecutionWorkspaceEstimate,
    InferenceGeometry, InputTokenCount, LayerSchedule, OutputDemand, StateMemoryLayout,
    WorkspaceBound,
};
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Barrier,
};

// Explicit portable stateless workspace; this fixture makes no native quote claim.
fn admission(bytes: u64) -> Admission {
    let geometry = InferenceGeometry {
        batch_size: 1,
        cached_positions: 0,
        input_positions: 1,
        max_output_tokens: 1,
        prefill_chunk_positions: 1,
        output: OutputDemand::LastPosition,
    };
    let layout = StateMemoryLayout::new(
        LayerSchedule::new(1, vec![LayerCachePolicy::NoState]).unwrap(),
        vec![0],
        1,
        1,
        EstimationCompleteness::Complete,
    )
    .unwrap();
    let state = eredu_core::estimate_runtime_state(
        &layout,
        InputTokenCount::text(1),
        1,
        1,
        std::num::NonZeroU8::new(4).unwrap(),
    )
    .unwrap();
    let bound = |n| WorkspaceBound::bounded(n, "portable capacity handoff fixture");
    let state = state
        .with_execution_workspace(ExecutionWorkspaceEstimate {
            geometry,
            activations: bound(bytes),
            attention: bound(0),
            vocabulary: bound(0),
            state_update: bound(0),
            materialization: bound(0),
            retained: bound(0),
        })
        .unwrap();
    Admission {
        requested_positions: 2,
        state,
        incremental_required_bytes: bytes,
        available_memory_bytes: None,
    }
}

fn reserve(
    pool: &WorkingMemoryPool,
    execution: &InferenceExecutionIdentity,
    bytes: u64,
    capacity: Option<u64>,
    handoffs: &[WorkingMemoryCapacityHandoff],
) -> Result<WorkingMemoryReservation, WorkingMemoryError> {
    pool.reserve_limited_with_minimum_and_handoffs(
        execution,
        &admission(bytes),
        capacity,
        None,
        handoffs,
    )
}

fn completed(
    pool: &WorkingMemoryPool,
    execution: &InferenceExecutionIdentity,
    bytes: u64,
    capacity: u64,
    stored: u64,
) -> (
    WorkingMemoryReservation,
    WorkingMemoryCapacityHandoff,
    BTreeMap<u32, WorkingMemoryStorage<u32>>,
) {
    let (metadata, mut run) = reserve(pool, execution, bytes, Some(capacity), &[])
        .unwrap()
        .into_funding()
        .unwrap();
    let token = run.take_capacity_handoff().unwrap();
    let scope = run.scope().unwrap();
    let storage = scope.adopt_storage_individually([(1u32, stored)]).unwrap();
    scope.certify().unwrap();
    run.close().unwrap();
    (metadata, token, storage)
}

#[derive(Debug, PartialEq, Eq)]
struct Snapshot {
    counters: (u64, usize, usize, u64, u64, u64),
    capacities: BTreeMap<u64, usize>,
    accounts: Vec<(u64, u64, usize, usize, usize, bool, bool, bool, Option<u64>)>,
}
fn snapshot(pool: &WorkingMemoryPool) -> Snapshot {
    let usage = pool.0.usage.lock().unwrap_or_else(|p| p.into_inner());
    Snapshot {
        counters: (
            usage.reserved,
            usage.reservations,
            usage.unquoted_owners,
            usage.registered,
            usage.peak,
            usage.next_funding,
        ),
        capacities: usage.funding.capacity_counts(),
        accounts: usage
            .funding
            .iter()
            .map(|(id, s)| {
                (
                    *id,
                    s.remaining,
                    s.allocations,
                    s.registrations,
                    s.scopes,
                    s.run_open,
                    s.quarantined,
                    s.metadata_live,
                    s.capacity,
                )
            })
            .collect(),
    }
}

#[test]
fn exact_capacity_commits_once_and_escaped_predecessor_keeps_adopted_ceiling() {
    let pool = WorkingMemoryPool::new(500, 0).unwrap();
    let execution = InferenceExecutionIdentity::default();
    let (metadata, token, storage) = completed(&pool, &execution, 100, 100, 60);
    let original = metadata.admission().clone();
    let before = snapshot(&pool);
    assert!(matches!(
        reserve(
            &pool,
            &execution,
            80,
            Some(139),
            std::slice::from_ref(&token)
        ),
        Err(WorkingMemoryError::BudgetExceeded { .. })
    ));
    assert_eq!(snapshot(&pool), before);
    let successor = reserve(
        &pool,
        &execution,
        80,
        Some(140),
        std::slice::from_ref(&token),
    )
    .unwrap();
    assert_eq!(pool.used_bytes().unwrap(), 140);
    assert_eq!(pool.effective_capacity().unwrap(), 140);
    assert_eq!(metadata.0.capacity, Some(100));
    assert_eq!(
        metadata.admission().incremental_required_bytes,
        original.incremental_required_bytes
    );
    assert_eq!(snapshot(&pool).capacities, BTreeMap::from([(140, 2)]));
    drop(successor);
    assert_eq!(pool.used_bytes().unwrap(), 60);
    assert_eq!(pool.effective_capacity().unwrap(), 140);
    assert!(matches!(
        reserve(&pool, &execution, 81, Some(500), &[]),
        Err(WorkingMemoryError::BudgetExceeded { .. })
    ));
    drop(metadata);
    assert!(!token.is_retired().unwrap());
    assert_eq!(pool.effective_capacity().unwrap(), 140);
    drop(storage);
    assert!(token.is_retired().unwrap());
    assert_eq!(pool.used_bytes().unwrap(), 0);
    assert_eq!(pool.effective_capacity().unwrap(), 500);
}

#[test]
fn successive_handoffs_preserve_physical_origins_and_alias_descendant_ceilings() {
    let pool = WorkingMemoryPool::new(500, 0).unwrap();
    let execution = InferenceExecutionIdentity::default();
    let (a, token_a, storage_a) = completed(&pool, &execution, 100, 100, 40);
    let (b, mut run_b) = reserve(
        &pool,
        &execution,
        20,
        Some(140),
        std::slice::from_ref(&token_a),
    )
    .unwrap()
    .into_funding()
    .unwrap();
    let token_b = run_b.take_capacity_handoff().unwrap();
    let scope = run_b.scope().unwrap();
    let mut storage_b = scope
        .adopt_storage_individually([(1u32, 40), (2, 20)])
        .unwrap();
    scope.certify().unwrap();
    run_b.close().unwrap();
    drop((a, storage_a));
    let tokens = [token_a, token_b];
    assert_eq!(pool.used_bytes().unwrap(), 60);
    let c = reserve(&pool, &execution, 10, Some(180), &tokens).unwrap();
    assert_eq!(snapshot(&pool).capacities, BTreeMap::from([(180, 3)]));
    drop((c, b));
    drop(storage_b.remove(&2));
    assert_eq!(pool.used_bytes().unwrap(), 40);
    assert_eq!(pool.effective_capacity().unwrap(), 180);
    assert!(tokens.iter().all(|t| !t.is_retired().unwrap()));
    drop(storage_b);
    assert_eq!(pool.used_bytes().unwrap(), 0);
    assert_eq!(pool.effective_capacity().unwrap(), 500);
    assert!(tokens.iter().all(|t| t.is_retired().unwrap()));
}

#[test]
fn zero_byte_and_existing_only_aliases_retain_the_delegated_policy() {
    let pool = WorkingMemoryPool::new(500, 0).unwrap();
    let execution = InferenceExecutionIdentity::default();
    let baseline = pool.register_storage([(0u32, 40)]).unwrap();
    let (metadata, mut run) = reserve(&pool, &execution, 0, Some(80), &[])
        .unwrap()
        .into_funding()
        .unwrap();
    let token = run.take_capacity_handoff().unwrap();
    let scope = run.scope().unwrap();
    let aliases = scope
        .adopt_storage_individually([(0u32, 40), (1, 0)])
        .unwrap();
    scope.certify().unwrap();
    run.close().unwrap();
    drop(metadata);
    let successor = reserve(
        &pool,
        &execution,
        0,
        Some(100),
        std::slice::from_ref(&token),
    )
    .unwrap();
    drop(successor);
    assert_eq!(pool.used_bytes().unwrap(), 40);
    assert_eq!(pool.effective_capacity().unwrap(), 100);
    assert!(matches!(
        pool.acquire_unquoted(),
        Err(WorkingMemoryError::ReservedWorkActive)
    ));
    drop(aliases);
    assert!(token.is_retired().unwrap());
    assert_eq!(pool.effective_capacity().unwrap(), 500);
    drop(pool.acquire_unquoted().unwrap());
    drop(baseline);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn active_scoped_and_quarantined_accounts_cannot_raise_but_equal_policy_is_unchanged() {
    for phase in 0..3 {
        let pool = WorkingMemoryPool::new(500, 0).unwrap();
        let execution = InferenceExecutionIdentity::default();
        let (metadata, mut run) = reserve(&pool, &execution, 40, Some(100), &[])
            .unwrap()
            .into_funding()
            .unwrap();
        let token = run.take_capacity_handoff().unwrap();
        let scope = run.scope().unwrap();
        let storage = scope.adopt_storage_individually([(1u32, 20)]).unwrap();
        let mut run = Some(run);
        let mut scope = Some(scope);
        if phase >= 1 {
            run.take().unwrap().close().unwrap();
        }
        if phase == 2 {
            drop(scope.take());
        }
        let before = snapshot(&pool);
        assert!(matches!(
            reserve(
                &pool,
                &execution,
                0,
                Some(200),
                std::slice::from_ref(&token)
            ),
            Err(WorkingMemoryError::ExecutionFenced)
        ));
        assert_eq!(snapshot(&pool), before);
        for unchanged in [100, 90] {
            let same = reserve(
                &pool,
                &execution,
                0,
                Some(unchanged),
                std::slice::from_ref(&token),
            )
            .unwrap();
            assert_eq!(
                pool.0.usage.lock().unwrap().funding[&token.id].capacity,
                Some(100)
            );
            drop(same);
        }
        if let Some(scope) = scope {
            scope.certify().unwrap();
        }
        drop(run);
        if phase != 2 {
            drop(
                reserve(
                    &pool,
                    &execution,
                    0,
                    Some(200),
                    std::slice::from_ref(&token),
                )
                .unwrap(),
            );
            assert_eq!(pool.effective_capacity().unwrap(), 200);
        }
        drop((metadata, storage));
        if phase == 2 {
            assert_eq!(pool.used_bytes().unwrap(), 40);
            assert!(!token.is_retired().unwrap());
        } else {
            assert!(token.is_retired().unwrap());
        }
    }
}

#[test]
fn unbounded_predecessor_is_not_changed_or_required_to_close() {
    let pool = WorkingMemoryPool::new(500, 0).unwrap();
    let execution = InferenceExecutionIdentity::default();
    let (metadata, mut run) = reserve(&pool, &execution, 40, None, &[])
        .unwrap()
        .into_funding()
        .unwrap();
    let token = run.take_capacity_handoff().unwrap();
    let successor = reserve(
        &pool,
        &execution,
        20,
        Some(100),
        std::slice::from_ref(&token),
    )
    .unwrap();
    assert_eq!(
        pool.0.usage.lock().unwrap().funding[&token.id].capacity,
        None
    );
    drop(successor);
    assert_eq!(pool.effective_capacity().unwrap(), 500);
    drop((metadata, run));
}

#[test]
fn unrelated_equal_ceilings_and_fixed_pool_capacity_are_preserved() {
    let pool = WorkingMemoryPool::new(500, 0).unwrap();
    let execution = InferenceExecutionIdentity::default();
    let (metadata, token, storage) = completed(&pool, &execution, 100, 100, 40);
    let unrelated = reserve(
        &pool,
        &InferenceExecutionIdentity::default(),
        0,
        Some(100),
        &[],
    )
    .unwrap();
    let before = snapshot(&pool);
    assert!(matches!(
        reserve(
            &pool,
            &execution,
            100,
            Some(200),
            std::slice::from_ref(&token)
        ),
        Err(WorkingMemoryError::BudgetExceeded { .. })
    ));
    assert_eq!(snapshot(&pool), before);
    let successor = reserve(
        &pool,
        &execution,
        50,
        Some(200),
        std::slice::from_ref(&token),
    )
    .unwrap();
    assert_eq!(pool.effective_capacity().unwrap(), 100);
    assert_eq!(
        snapshot(&pool).capacities,
        BTreeMap::from([(100, 1), (200, 2)])
    );
    drop((successor, unrelated));
    assert_eq!(pool.effective_capacity().unwrap(), 200);
    drop((metadata, storage));

    let pool = WorkingMemoryPool::new(120, 0).unwrap();
    let (_metadata, token, _storage) = completed(&pool, &execution, 100, 100, 40);
    let before = snapshot(&pool);
    assert!(matches!(
        reserve(
            &pool,
            &execution,
            81,
            Some(200),
            std::slice::from_ref(&token)
        ),
        Err(WorkingMemoryError::BudgetExceeded { .. })
    ));
    assert_eq!(snapshot(&pool), before);
}

#[test]
fn identities_duplicates_and_retired_tokens_cannot_authorize_other_accounts() {
    let pool = WorkingMemoryPool::new(500, 0).unwrap();
    let execution = InferenceExecutionIdentity::default();
    let (metadata, mut run) = reserve(&pool, &execution, 40, Some(100), &[])
        .unwrap()
        .into_funding()
        .unwrap();
    let token = run.take_capacity_handoff().unwrap();
    assert!(matches!(
        run.take_capacity_handoff(),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    run.close().unwrap();
    let before = snapshot(&pool);
    let foreign = WorkingMemoryPool::new(500, 0).unwrap();
    assert!(matches!(
        reserve(
            &foreign,
            &execution,
            0,
            Some(200),
            std::slice::from_ref(&token)
        ),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    assert!(matches!(
        reserve(
            &pool,
            &InferenceExecutionIdentity::default(),
            0,
            Some(200),
            std::slice::from_ref(&token)
        ),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    // Private test construction exercises duplicate detection; public tokens cannot clone.
    let duplicate = WorkingMemoryCapacityHandoff {
        pool: token.pool.clone(),
        execution: token.execution.clone(),
        id: token.id,
    };
    let tokens = [token, duplicate];
    assert!(matches!(
        reserve(&pool, &execution, 0, Some(200), &tokens),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    assert_eq!(snapshot(&pool), before);
    drop(metadata);
    assert!(tokens[0].is_retired().unwrap());
    drop(reserve(&pool, &execution, 1, Some(200), &tokens[..1]).unwrap());
    let weak = Arc::downgrade(&pool.0);
    drop(pool);
    assert!(weak.upgrade().is_none());
    assert!(tokens[0].is_retired().unwrap());
}

#[test]
fn reservation_counter_overflow_does_not_partially_raise() {
    let pool = WorkingMemoryPool::new(500, 0).unwrap();
    let execution = InferenceExecutionIdentity::default();
    let (_metadata, token, _storage) = completed(&pool, &execution, 100, 100, 40);
    // Ceiling ownership is now represented by the actual fixed account nodes;
    // there is no separately forgeable map multiplicity. Exhausting the one
    // account count must still reject before committing any predecessor raise.
    let original = {
        let mut u = pool.0.usage.lock().unwrap();
        let n = u.reservations;
        u.reservations = usize::MAX;
        n
    };
    let before = snapshot(&pool);
    assert!(matches!(
        reserve(
            &pool,
            &execution,
            0,
            Some(200),
            std::slice::from_ref(&token)
        ),
        Err(WorkingMemoryError::Overflow)
    ));
    assert_eq!(snapshot(&pool), before);
    pool.0.usage.lock().unwrap().reservations = original;
}

#[test]
fn handoff_does_not_bypass_unquoted_exclusion_or_poison() {
    let pool = WorkingMemoryPool::new(500, 0).unwrap();
    let execution = InferenceExecutionIdentity::default();
    let (metadata, token, storage) = completed(&pool, &execution, 100, 100, 40);
    drop((metadata, storage));
    let unquoted = pool.acquire_unquoted().unwrap();
    let before = snapshot(&pool);
    assert!(matches!(
        reserve(
            &pool,
            &execution,
            0,
            Some(200),
            std::slice::from_ref(&token)
        ),
        Err(WorkingMemoryError::UnknownBound)
    ));
    assert_eq!(snapshot(&pool), before);
    drop(unquoted);
    let (_metadata, token, _storage) = completed(&pool, &execution, 100, 100, 40);
    let before = snapshot(&pool);
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _guard = pool.0.usage.lock().unwrap();
        panic!("poison handoff ledger");
    }))
    .is_err());
    assert!(matches!(
        token.is_retired(),
        Err(WorkingMemoryError::Poisoned)
    ));
    assert!(matches!(
        reserve(
            &pool,
            &execution,
            0,
            Some(200),
            std::slice::from_ref(&token)
        ),
        Err(WorkingMemoryError::Poisoned)
    ));
    assert_eq!(snapshot(&pool), before);
}

#[test]
fn certification_and_handoff_race_is_one_atomic_eligibility_decision() {
    for _ in 0..8 {
        let pool = WorkingMemoryPool::new(500, 0).unwrap();
        let execution = InferenceExecutionIdentity::default();
        let (metadata, mut run) = reserve(&pool, &execution, 100, Some(100), &[])
            .unwrap()
            .into_funding()
            .unwrap();
        let token = run.take_capacity_handoff().unwrap();
        let scope = run.scope().unwrap();
        let storage = scope.adopt_storage_individually([(1u32, 40)]).unwrap();
        run.close().unwrap();
        let barrier = Barrier::new(2);
        let result = std::thread::scope(|threads| {
            let certify = threads.spawn(|| {
                barrier.wait();
                scope.certify().unwrap();
            });
            barrier.wait();
            let result = reserve(
                &pool,
                &execution,
                60,
                Some(120),
                std::slice::from_ref(&token),
            );
            certify.join().unwrap();
            result
        });
        let successor = match result {
            Ok(value) => value,
            Err(WorkingMemoryError::ExecutionFenced) => {
                assert_eq!(pool.effective_capacity().unwrap(), 100);
                reserve(
                    &pool,
                    &execution,
                    60,
                    Some(120),
                    std::slice::from_ref(&token),
                )
                .unwrap()
            }
            Err(error) => panic!("unexpected race result: {error}"),
        };
        assert_eq!(pool.used_bytes().unwrap(), 100);
        assert_eq!(pool.effective_capacity().unwrap(), 120);
        drop((successor, metadata, storage));
        assert!(token.is_retired().unwrap());
    }
}

#[test]
fn concurrent_successors_cannot_oversubscribe_the_same_completed_account() {
    let pool = WorkingMemoryPool::new(500, 0).unwrap();
    let execution = InferenceExecutionIdentity::default();
    let (_metadata, token, _storage) = completed(&pool, &execution, 80, 80, 40);
    let barrier = Barrier::new(2);
    let results = std::thread::scope(|threads| {
        let attempt = || {
            barrier.wait();
            reserve(
                &pool,
                &execution,
                60,
                Some(100),
                std::slice::from_ref(&token),
            )
        };
        let first = threads.spawn(attempt);
        let second = threads.spawn(attempt);
        [first.join().unwrap(), second.join().unwrap()]
    });
    assert_eq!(results.iter().filter(|r| r.is_ok()).count(), 1);
    assert_eq!(
        results
            .iter()
            .filter(|r| matches!(r, Err(WorkingMemoryError::BudgetExceeded { .. })))
            .count(),
        1
    );
    assert_eq!(pool.used_bytes().unwrap(), 100);
    assert_eq!(pool.effective_capacity().unwrap(), 100);
    assert_eq!(snapshot(&pool).capacities, BTreeMap::from([(100, 2)]));
}

struct Probe {
    pool: WorkingMemoryPool,
    armed: AtomicBool,
    drops: AtomicUsize,
}
struct Key {
    id: u32,
    probe: Arc<Probe>,
}
impl Clone for Key {
    fn clone(&self) -> Self {
        assert!(self.probe.pool.0.usage.try_lock().is_ok());
        Self {
            id: self.id,
            probe: self.probe.clone(),
        }
    }
}
impl PartialEq for Key {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}
impl Eq for Key {}
impl PartialOrd for Key {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Key {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.id.cmp(&other.id)
    }
}
impl Drop for Key {
    fn drop(&mut self) {
        if self.probe.armed.load(Ordering::SeqCst) {
            assert!(self.probe.pool.0.usage.try_lock().is_ok());
            assert_eq!(self.probe.pool.used_bytes().unwrap(), 40);
            assert_eq!(self.probe.pool.effective_capacity().unwrap(), 200);
            self.probe.drops.fetch_add(1, Ordering::SeqCst);
        }
    }
}

#[test]
fn escaped_key_destructors_run_unlocked_before_adopted_charge_and_ceiling_retire() {
    let pool = WorkingMemoryPool::new(500, 0).unwrap();
    let execution = InferenceExecutionIdentity::default();
    let (metadata, mut run) = reserve(&pool, &execution, 100, Some(100), &[])
        .unwrap()
        .into_funding()
        .unwrap();
    let token = run.take_capacity_handoff().unwrap();
    let scope = run.scope().unwrap();
    let probe = Arc::new(Probe {
        pool: pool.clone(),
        armed: AtomicBool::new(false),
        drops: AtomicUsize::new(0),
    });
    let storage = scope
        .adopt_storage_individually([(
            Key {
                id: 1,
                probe: probe.clone(),
            },
            40,
        )])
        .unwrap()
        .into_values()
        .collect::<Vec<_>>();
    scope.certify().unwrap();
    run.close().unwrap();
    drop(metadata);
    probe.armed.store(true, Ordering::SeqCst);
    drop(
        reserve(
            &pool,
            &execution,
            0,
            Some(200),
            std::slice::from_ref(&token),
        )
        .unwrap(),
    );
    assert_eq!(probe.drops.load(Ordering::SeqCst), 0);
    drop(storage);
    assert!(probe.drops.load(Ordering::SeqCst) > 0);
    assert_eq!(pool.used_bytes().unwrap(), 0);
    assert_eq!(pool.effective_capacity().unwrap(), 500);
    assert!(token.is_retired().unwrap());
}
