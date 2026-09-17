use super::*;
use crate::working_memory::WorkingMemoryStorage;

const UNCOPIED_BYTES: u64 = 37;
const COMPLETE_BYTES: u64 = ARRAY_SOURCE_BYTES + UNCOPIED_BYTES;

fn complete(pool: &WorkingMemoryPool) -> WorkingMemoryStorage<u32> {
    pool.register_storage([(1u32, ARRAY_SOURCE_BYTES), (9, UNCOPIED_BYTES)])
        .unwrap()
}

fn joined<'a>(
    pool: &WorkingMemoryPool,
    sampler: BorrowedFundedSampler<'a>,
    complete: &WorkingMemoryStorage<u32>,
) -> RegisteredSamplingCopyWithSource<'a, u32> {
    joint(pool, sampler).with_complete_source(complete.clone())
}

fn snapshot(pool: &WorkingMemoryPool) -> ((u64, u64, u64), usize, usize) {
    let accounts = pool.0.usage.lock().unwrap().funding.len();
    (usage(pool), accounts, attempts())
}

#[test]
fn complete_source_exact_and_one_short_keep_one_account_and_unchanged_demand() {
    for adaptive in [false, true] {
        let pool = WorkingMemoryPool::new(8192, 0).unwrap();
        let complete = complete(&pool);
        let (mut sampler, preparation, run) = source(&pool, 8192, 8, adaptive);
        grow(&mut sampler, &[3, 11, 7, 19, 5]);
        let required = joint(&pool, sampler.borrow_funded()).required_bytes();
        assert_eq!(
            joined(&pool, sampler.borrow_funded(), &complete).required_bytes(),
            required
        );
        let before = snapshot(&pool);
        let safety = 7;
        let total = required + safety;
        let mut limits = WorkspaceCopyLimits::new((before.0).0 + total);
        limits.safety_reserve_bytes = safety;
        limits.application_memory_budget_bytes = Some(total - 1);
        assert!(matches!(
            pool.copy_sampling_components_with_source(
                joined(&pool, sampler.borrow_funded(), &complete), limits
            ),
            Err(SamplingCopyAdmissionError::ApplicationBudgetExceeded {
                required_bytes, budget_bytes,
            }) if required_bytes == total && budget_bytes == total - 1
        ));
        assert_eq!(snapshot(&pool), before);
        limits.application_memory_budget_bytes = Some(total);
        limits.capacity_bytes -= 1;
        assert!(matches!(
            pool.copy_sampling_components_with_source(
                joined(&pool, sampler.borrow_funded(), &complete), limits
            ),
            Err(SamplingCopyAdmissionError::Memory(WorkingMemoryError::BudgetExceeded {
                required_bytes, available_bytes,
            })) if required_bytes == total && available_bytes == total - 1
        ));
        assert_eq!(snapshot(&pool), before);
        limits.capacity_bytes += 1;
        let result = pool
            .copy_sampling_components_with_source(
                joined(&pool, sampler.borrow_funded(), &complete),
                limits,
            )
            .unwrap();
        assert_eq!(snapshot(&pool).1, before.1 + 1);
        assert_eq!(attempts(), before.2 + 1);
        assert_eq!(result.1.bytes(), total);
        assert_eq!(result.0.bytes(), required - COPY_BYTES);
        assert_eq!(history(result.0.as_sampler()), &[3, 11, 7, 19, 5]);
        assert_ne!(
            history(result.0.as_sampler()).as_ptr(),
            history(sampler.as_sampler()).as_ptr()
        );
        assert_eq!(pool.used_bytes().unwrap(), limits.capacity_bytes);
        drop((sampler, preparation, run, complete));
        assert_eq!(pool.used_bytes().unwrap(), COMPLETE_BYTES + total);
        settle(result);
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}

#[test]
fn foreign_complete_inventory_including_empty_inventory_rejects_before_copy() {
    let pool = WorkingMemoryPool::new(8192, 0).unwrap();
    let other = WorkingMemoryPool::new(8192, 0).unwrap();
    let _actual = complete(&pool);
    let (sampler, _preparation, _run) = source(&pool, 8192, 8, false);
    for foreign in [
        complete(&other),
        other.register_storage([] as [(u32, u64); 0]).unwrap(),
    ] {
        let before = (snapshot(&pool), usage(&other));
        let copy = joint(&pool, sampler.borrow_funded()).with_complete_source(foreign.clone());
        assert!(matches!(
            pool.copy_sampling_components_with_source(copy, WorkspaceCopyLimits::new(8192)),
            Err(SamplingCopyAdmissionError::Memory(
                WorkingMemoryError::IdentityMismatch
            ))
        ));
        assert_eq!((snapshot(&pool), usage(&other)), before);
    }
}

#[test]
fn newly_quarantined_uncopied_source_rejects_in_the_same_admission() {
    let pool = WorkingMemoryPool::new(8192, 0).unwrap();
    let _operand = pool.register_storage([(1u32, ARRAY_SOURCE_BYTES)]).unwrap();
    let (sampler, _preparation, _run) = source(&pool, 8192, 8, false);
    let origin = pool
        .admit_workspace_copy(
            copy_plan(&pool, 1, ARRAY_SOURCE_BYTES, 1),
            WorkspaceCopyLimits::new(8192),
        )
        .unwrap();
    let (origin, scope) = origin.into_parts();
    let _uncopied = scope
        .adopt_storage_individually([(9u32, UNCOPIED_BYTES)])
        .unwrap();
    let complete = pool
        .pin_registered_storage([(1u32, ARRAY_SOURCE_BYTES), (9, UNCOPIED_BYTES)])
        .unwrap();
    let copy = joined(&pool, sampler.borrow_funded(), &complete);
    // Only the uncopied root's origin becomes unhealthy after plan construction.
    // The sampler and the operand remain independently valid.
    drop(scope);
    let before = snapshot(&pool);
    assert!(matches!(
        pool.copy_sampling_components_with_source(copy, WorkspaceCopyLimits::new(8192)),
        Err(SamplingCopyAdmissionError::Memory(
            WorkingMemoryError::ExecutionFenced
        ))
    ));
    assert_eq!(snapshot(&pool), before);
    // The existing route still validates only its own sources, as before.
    settle(
        pool.copy_sampling_components(
            joint(&pool, sampler.borrow_funded()),
            WorkspaceCopyLimits::new(8192),
        )
        .unwrap(),
    );
    drop(origin);
}

#[test]
fn native_scope_retains_uncopied_roots_after_host_and_destination_retire() {
    let pool = WorkingMemoryPool::new(8192, 0).unwrap();
    let complete = complete(&pool);
    let (sampler, preparation, run) = source(&pool, 8192, 8, false);
    let (copied, native) = pool
        .copy_sampling_components_with_source(
            joined(&pool, sampler.borrow_funded(), &complete),
            WorkspaceCopyLimits::new(8192),
        )
        .unwrap();
    let bytes = native.bytes();
    let (custody, scope) = native.into_parts();
    drop((sampler, preparation, run, complete, copied, custody));
    assert_eq!(pool.used_bytes().unwrap(), COMPLETE_BYTES + bytes);
    drop(
        pool.pin_registered_storage([(1u32, ARRAY_SOURCE_BYTES), (9, UNCOPIED_BYTES)])
            .unwrap(),
    );
    assert!(matches!(
        pool.acquire_unquoted(),
        Err(WorkingMemoryError::ReservedWorkActive)
    ));
    // Only certification of the native scope releases both source inventories.
    scope.certify().unwrap();
    assert_eq!(pool.used_bytes().unwrap(), 0);
    assert!(matches!(
        pool.pin_registered_storage([(9u32, UNCOPIED_BYTES)]),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    drop(pool.acquire_unquoted().unwrap());
}

#[test]
fn native_quarantine_retains_complete_bundle_after_all_other_owners_retire() {
    let pool = WorkingMemoryPool::new(8192, 0).unwrap();
    let complete = complete(&pool);
    let (sampler, preparation, run) = source(&pool, 8192, 8, false);
    let (copied, native) = pool
        .copy_sampling_components_with_source(
            joined(&pool, sampler.borrow_funded(), &complete),
            WorkspaceCopyLimits::new(8192),
        )
        .unwrap();
    let bytes = native.bytes();
    let (custody, scope) = native.into_parts();
    let outputs = scope.adopt_storage_individually([(2u32, 16)]).unwrap();
    drop((
        sampler,
        preparation,
        run,
        complete,
        copied,
        custody,
        outputs,
    ));
    drop(scope);
    assert_eq!(pool.used_bytes().unwrap(), COMPLETE_BYTES + bytes);
    drop(
        pool.pin_registered_storage([(1u32, ARRAY_SOURCE_BYTES), (9, UNCOPIED_BYTES)])
            .unwrap(),
    );
    assert_eq!(pool.used_bytes().unwrap(), COMPLETE_BYTES + bytes);
    assert!(matches!(
        pool.acquire_unquoted(),
        Err(WorkingMemoryError::ReservedWorkActive)
    ));
}

#[test]
fn copy_unwind_quarantines_uncopied_sources_without_a_returned_host_owner() {
    let pool = WorkingMemoryPool::new(8192, 0).unwrap();
    let complete = complete(&pool);
    let (sampler, preparation, run) = source(&pool, 8192, 8, false);
    let copy = joined(&pool, sampler.borrow_funded(), &complete);
    let bytes = copy.required_bytes();
    FAIL_COPY.with(|flag| flag.set(true));
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        pool.copy_sampling_components_with_source(copy, WorkspaceCopyLimits::new(8192))
            .unwrap()
    }));
    assert!(result.is_err());
    drop((sampler, preparation, run, complete));
    assert_eq!(pool.used_bytes().unwrap(), COMPLETE_BYTES + bytes);
    drop(
        pool.pin_registered_storage([(9u32, UNCOPIED_BYTES)])
            .unwrap(),
    );
    assert!(matches!(
        pool.acquire_unquoted(),
        Err(WorkingMemoryError::ReservedWorkActive)
    ));
}

#[test]
fn host_hold_still_blocks_native_adoption_with_overlapping_source_pins() {
    let pool = WorkingMemoryPool::new(8192, 0).unwrap();
    let complete = complete(&pool);
    let (mut sampler, preparation, run) = source(&pool, 8192, 8, false);
    grow(&mut sampler, &[3, 11, 7, 19, 5]);
    let (copied, native) = pool
        .copy_sampling_components_with_source(
            joined(&pool, sampler.borrow_funded(), &complete),
            WorkspaceCopyLimits::new(8192),
        )
        .unwrap();
    let host = copied.bytes();
    let (custody, scope) = native.into_parts();
    let aliases = scope
        .adopt_storage_individually([(1u32, ARRAY_SOURCE_BYTES), (9, UNCOPIED_BYTES)])
        .unwrap();
    let output = scope
        .adopt_storage_individually([(2u32, COPY_BYTES)])
        .unwrap();
    let before = usage(&pool);
    assert!(matches!(
        scope.adopt_storage_individually([(3u32, 1)]),
        Err(WorkingMemoryError::BudgetExceeded {
            required_bytes: 1,
            available_bytes: 0
        })
    ));
    assert_eq!(usage(&pool), before);
    drop((sampler, preparation, run, complete, aliases));
    scope.certify().unwrap();
    assert_eq!(pool.used_bytes().unwrap(), host + COPY_BYTES);
    drop(custody);
    assert_eq!(pool.used_bytes().unwrap(), host + COPY_BYTES);
    drop(copied);
    assert_eq!(pool.used_bytes().unwrap(), COPY_BYTES);
    drop(output);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn genuine_empty_operand_program_retains_layout_without_a_decoder_table() {
    let pool = WorkingMemoryPool::new(8192, 0).unwrap();
    // The only additional root represents immutable source layout metadata;
    // there is no table registration or fabricated zero-capacity slot owner.
    let layout = pool.register_storage([(9u32, UNCOPIED_BYTES)]).unwrap();
    let (mut sampler, preparation, run) = source(&pool, 8192, 8, true);
    grow(&mut sampler, &[3, 11, 7]);
    let context = WorkspaceContext::new(Facts::default());
    let source = RegisteredWorkspaceStorage::bind(
        &pool,
        &context,
        [] as [(u32, WorkspaceExistingStorage); 0],
    )
    .unwrap();
    let program =
        WorkspaceIsolatedCopyPlan::prepare(&context, source.borrowed_storage(), &[]).unwrap();
    assert_eq!(program.incremental_bytes(), Some(0));
    let arrays = RegisteredWorkspaceCopy::bind(program, source).unwrap();
    let copy = RegisteredSamplingCopy::prepare(sampler.borrow_funded(), arrays)
        .unwrap()
        .with_complete_source(layout.clone());
    let host = sampler
        .as_sampler()
        .prepare_copy()
        .unwrap()
        .retained_bytes();
    assert_eq!(copy.required_bytes(), host);
    let before = snapshot(&pool);
    let (copied, native) = pool
        .copy_sampling_components_with_source(copy, WorkspaceCopyLimits::new((before.0).0 + host))
        .unwrap();
    assert_eq!(snapshot(&pool).1, before.1 + 1);
    assert_eq!((copied.bytes(), native.bytes()), (host, host));
    assert_eq!(history(copied.as_sampler()), &[3, 11, 7]);
    let (custody, scope) = native.into_parts();
    assert!(matches!(
        scope.adopt_storage_individually([(2u32, 1)]),
        Err(WorkingMemoryError::BudgetExceeded {
            required_bytes: 1,
            available_bytes: 0
        })
    ));
    drop((sampler, preparation, run, layout, copied, custody));
    assert_eq!(pool.used_bytes().unwrap(), UNCOPIED_BYTES + host);
    scope.certify().unwrap();
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
