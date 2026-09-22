use super::*;
use crate::working_memory::WorkingMemoryStorage;

const UNCOPIED_BYTES: u64 = 37;
const COMPLETE_BYTES: u64 = ARRAY_SOURCE_BYTES + UNCOPIED_BYTES;

fn complete(pool: &MemoryLedger) -> WorkingMemoryStorage<u32> {
    pool.register_host_storage([(1u32, ARRAY_SOURCE_BYTES), (9, UNCOPIED_BYTES)])
        .unwrap()
}

fn joined<'a>(
    pool: &MemoryLedger,
    sampler: BorrowedFundedSampler<'a>,
    complete: &WorkingMemoryStorage<u32>,
) -> RegisteredSamplingCopyWithSource<'a, u32> {
    joint(pool, sampler).with_complete_source(complete.clone())
}

fn snapshot(pool: &MemoryLedger) -> ((u64, u64, u64), usize, usize) {
    let accounts = pool.0.usage.lock().unwrap().funding.len();
    (usage(pool), accounts, attempts())
}

#[test]
fn complete_source_exact_and_one_short_keep_one_account_and_unchanged_demand() {
    for adaptive in [false, true] {
        let pool = crate::working_memory::memory_fixture::host_ledger((1 << 20), 0).unwrap();
        let complete = complete(&pool);
        let (mut sampler, preparation, run) = source(&pool, (1 << 20), 8, adaptive);
        grow(&mut sampler, &[3, 11, 7, 19, 5]);
        let required = joint(&pool, sampler.borrow_funded())
            .required_bytes()
            .unwrap();
        assert_eq!(
            joined(&pool, sampler.borrow_funded(), &complete)
                .required_bytes()
                .unwrap(),
            required
        );
        let rejected = joined(&pool, sampler.borrow_funded(), &complete);
        let full = pool
            .sampling_copy_with_source_requirements(&rejected, &WorkspaceCopyLimits::default())
            .unwrap()
            .get(pool.topology().host_domain())
            .unwrap()
            .total()
            .unwrap();
        let initial = current(&pool);
        let before = snapshot(&pool);
        let safety = 7;
        let total = full + safety;
        let mut limits = WorkspaceCopyLimits::new(
            crate::working_memory::memory_fixture::host_limits(initial + total),
        );
        limits.additional_headroom =
            eredu_core::MemoryHeadroomDeclarations::new([("host".into(), safety)]);
        limits.memory_limits =
            crate::working_memory::memory_fixture::host_limits(initial + total - 1);
        assert!(matches!(
            pool.copy_sampling_components_with_source(
                rejected, limits.clone()),
            Err(SamplingCopyAdmissionError::Memory(WorkingMemoryError::Domain(eredu_core::MemoryDomainError::BudgetExceeded { requested_bytes: required_bytes, limit_bytes, existing_bytes, .. }))) if required_bytes == total && limit_bytes - existing_bytes == total - 1
        ));
        assert_eq!(snapshot(&pool), before);
        limits.memory_limits = crate::working_memory::memory_fixture::host_limits(initial + total);
        let result = pool
            .copy_sampling_components_with_source(
                joined(&pool, sampler.borrow_funded(), &complete),
                limits.clone(),
            )
            .unwrap();
        assert_eq!(snapshot(&pool).1, before.1 + 1);
        assert_eq!(attempts(), before.2 + 1);
        assert_eq!(
            result
                .1
                .requirements()
                .get(pool.topology().host_domain())
                .unwrap()
                .total()
                .unwrap(),
            total
        );
        assert_eq!(result.0.bytes(), required - COPY_BYTES);
        assert_eq!(history(result.0.as_sampler()), &[3, 11, 7, 19, 5]);
        assert_ne!(
            history(result.0.as_sampler()).as_ptr(),
            history(sampler.as_sampler()).as_ptr()
        );
        assert_eq!(
            pool.payload_used_bytes().unwrap(),
            (before.0).0 + required + safety
        );
        drop((sampler, preparation, run, complete));
        assert_eq!(
            pool.payload_used_bytes().unwrap(),
            COMPLETE_BYTES + required + safety
        );
        settle(result);
        assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    }
}

#[test]
fn foreign_complete_inventory_including_empty_inventory_rejects_before_copy() {
    let pool = crate::working_memory::memory_fixture::host_ledger((1 << 20), 0).unwrap();
    let other = crate::working_memory::memory_fixture::host_ledger((1 << 20), 0).unwrap();
    let _actual = complete(&pool);
    let (sampler, _preparation, _run) = source(&pool, (1 << 20), 8, false);
    for foreign in [
        complete(&other),
        other.register_host_storage([] as [(u32, u64); 0]).unwrap(),
    ] {
        let before = (snapshot(&pool), usage(&other));
        let copy = joint(&pool, sampler.borrow_funded()).with_complete_source(foreign.clone());
        assert!(matches!(
            pool.copy_sampling_components_with_source(
                copy,
                WorkspaceCopyLimits::new(crate::working_memory::memory_fixture::host_limits(
                    (1 << 20)
                ))
            ),
            Err(SamplingCopyAdmissionError::Memory(
                WorkingMemoryError::IdentityMismatch
            ))
        ));
        assert_eq!((snapshot(&pool), usage(&other)), before);
    }
}

#[test]
fn newly_quarantined_uncopied_source_rejects_in_the_same_admission() {
    let pool = crate::working_memory::memory_fixture::host_ledger((1 << 20), 0).unwrap();
    let _operand = pool
        .register_host_storage([(1u32, ARRAY_SOURCE_BYTES)])
        .unwrap();
    let (sampler, _preparation, _run) = source(&pool, (1 << 20), 8, false);
    let origin = pool
        .admit_workspace_copy(
            copy_plan(&pool, 1, ARRAY_SOURCE_BYTES, 1),
            WorkspaceCopyLimits::new(crate::working_memory::memory_fixture::host_limits(
                (1 << 20),
            )),
        )
        .unwrap();
    let (origin, scope) = origin.into_parts();
    let _uncopied = scope
        .publish_host_storage_fixture([(9u32, UNCOPIED_BYTES)])
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
        pool.copy_sampling_components_with_source(
            copy,
            WorkspaceCopyLimits::new(crate::working_memory::memory_fixture::host_limits(
                (1 << 20)
            ))
        ),
        Err(SamplingCopyAdmissionError::Memory(
            WorkingMemoryError::ExecutionFenced
        ))
    ));
    assert_eq!(snapshot(&pool), before);
    // The existing route still validates only its own sources, as before.
    settle(
        pool.copy_sampling_components(
            joint(&pool, sampler.borrow_funded()),
            WorkspaceCopyLimits::new(crate::working_memory::memory_fixture::host_limits(
                (1 << 20),
            )),
        )
        .unwrap(),
    );
    drop(origin);
}

#[test]
fn native_scope_retains_uncopied_roots_after_host_and_destination_retire() {
    let pool = crate::working_memory::memory_fixture::host_ledger((1 << 20), 0).unwrap();
    let complete = complete(&pool);
    let (sampler, preparation, run) = source(&pool, (1 << 20), 8, false);
    let (copied, native) = pool
        .copy_sampling_components_with_source(
            joined(&pool, sampler.borrow_funded(), &complete),
            WorkspaceCopyLimits::new(crate::working_memory::memory_fixture::host_limits(
                (1 << 20),
            )),
        )
        .unwrap();
    let bytes = copied.bytes() + COPY_BYTES;
    let (custody, scope) = native.into_parts();
    drop((sampler, preparation, run, complete, copied, custody));
    assert_eq!(pool.payload_used_bytes().unwrap(), COMPLETE_BYTES + bytes);
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
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    assert!(matches!(
        pool.pin_registered_storage([(9u32, UNCOPIED_BYTES)]),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    crate::working_memory::memory_fixture::assert_unquoted_idle(&pool);
}

#[test]
fn native_quarantine_retains_complete_bundle_after_all_other_owners_retire() {
    let pool = crate::working_memory::memory_fixture::host_ledger((1 << 20), 0).unwrap();
    let complete = complete(&pool);
    let (sampler, preparation, run) = source(&pool, (1 << 20), 8, false);
    let (copied, native) = pool
        .copy_sampling_components_with_source(
            joined(&pool, sampler.borrow_funded(), &complete),
            WorkspaceCopyLimits::new(crate::working_memory::memory_fixture::host_limits(
                (1 << 20),
            )),
        )
        .unwrap();
    let bytes = copied.bytes() + COPY_BYTES;
    let (custody, scope) = native.into_parts();
    let outputs = scope.publish_host_storage_fixture([(2u32, 16)]).unwrap();
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
    assert_eq!(pool.payload_used_bytes().unwrap(), COMPLETE_BYTES + bytes);
    drop(
        pool.pin_registered_storage([(1u32, ARRAY_SOURCE_BYTES), (9, UNCOPIED_BYTES)])
            .unwrap(),
    );
    assert_eq!(pool.payload_used_bytes().unwrap(), COMPLETE_BYTES + bytes);
    assert!(matches!(
        pool.acquire_unquoted(),
        Err(WorkingMemoryError::ReservedWorkActive)
    ));
}

#[test]
fn copy_unwind_quarantines_uncopied_sources_without_a_returned_host_owner() {
    let pool = crate::working_memory::memory_fixture::host_ledger((1 << 20), 0).unwrap();
    let complete = complete(&pool);
    let (sampler, preparation, run) = source(&pool, (1 << 20), 8, false);
    let copy = joined(&pool, sampler.borrow_funded(), &complete);
    let bytes = copy.required_bytes().unwrap();
    FAIL_COPY.with(|flag| flag.set(true));
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        pool.copy_sampling_components_with_source(
            copy,
            WorkspaceCopyLimits::new(crate::working_memory::memory_fixture::host_limits(
                (1 << 20),
            )),
        )
        .unwrap()
    }));
    assert!(result.is_err());
    drop((sampler, preparation, run, complete));
    assert_eq!(pool.payload_used_bytes().unwrap(), COMPLETE_BYTES + bytes);
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
    let pool = crate::working_memory::memory_fixture::host_ledger((1 << 20), 0).unwrap();
    let complete = complete(&pool);
    let (mut sampler, preparation, run) = source(&pool, (1 << 20), 8, false);
    grow(&mut sampler, &[3, 11, 7, 19, 5]);
    let (copied, native) = pool
        .copy_sampling_components_with_source(
            joined(&pool, sampler.borrow_funded(), &complete),
            WorkspaceCopyLimits::new(crate::working_memory::memory_fixture::host_limits(
                (1 << 20),
            )),
        )
        .unwrap();
    let host = copied.bytes();
    let (custody, scope) = native.into_parts();
    let aliases = scope
        .publish_host_storage_fixture([(1u32, ARRAY_SOURCE_BYTES), (9, UNCOPIED_BYTES)])
        .unwrap();
    let output = scope
        .publish_host_storage_fixture([(2u32, COPY_BYTES)])
        .unwrap();
    let rejected = crate::working_memory::StoragePublicationLayout::<u32>::new(1)
        .unwrap()
        .fund(&pool)
        .unwrap();
    let before = usage(&pool);
    assert!(matches!(
        rejected.adopt_storage_individually(
            &scope,
            [(
                3u32,
                crate::working_memory::StorageAllocation::new(1, pool.host_placement_handle())
            )]
        ),
        Err(WorkingMemoryError::DomainAllowanceExceeded {
            required_bytes: 1,
            available_bytes: 0,
            ..
        })
    ));
    assert_eq!(usage(&pool), before);
    drop((sampler, preparation, run, complete, aliases));
    scope.certify().unwrap();
    assert_eq!(pool.payload_used_bytes().unwrap(), host + COPY_BYTES);
    drop(custody);
    assert_eq!(pool.payload_used_bytes().unwrap(), host + COPY_BYTES);
    drop(copied);
    assert_eq!(pool.payload_used_bytes().unwrap(), COPY_BYTES);
    drop(output);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[test]
fn genuine_empty_operand_program_retains_layout_without_a_decoder_table() {
    let pool = crate::working_memory::memory_fixture::host_ledger((1 << 20), 0).unwrap();
    // The only additional root represents immutable source layout metadata;
    // there is no table registration or fabricated zero-capacity slot owner.
    let layout = pool
        .register_host_storage([(9u32, UNCOPIED_BYTES)])
        .unwrap();
    let (mut sampler, preparation, run) = source(&pool, (1 << 20), 8, true);
    grow(&mut sampler, &[3, 11, 7]);
    let context = WorkspaceContext::new(Facts::default().with_pool(&pool));
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
    assert_eq!(copy.required_bytes().unwrap(), host);
    let rejected = crate::working_memory::StoragePublicationLayout::<u32>::new(1)
        .unwrap()
        .fund(&pool)
        .unwrap();
    let required = pool
        .sampling_copy_with_source_requirements(
            &copy,
            &WorkspaceCopyLimits::new(Default::default()),
        )
        .unwrap()
        .get(pool.topology().host_domain())
        .unwrap()
        .total()
        .unwrap();
    let capacity = current(&pool) + required;
    let before = snapshot(&pool);
    let (copied, native) = pool
        .copy_sampling_components_with_source(
            copy,
            WorkspaceCopyLimits::new(crate::working_memory::memory_fixture::host_limits(capacity)),
        )
        .unwrap();
    assert_eq!(snapshot(&pool).1, before.1 + 1);
    assert_eq!(
        (
            copied.bytes(),
            native
                .requirements()
                .get(pool.topology().host_domain())
                .unwrap()
                .total()
                .unwrap()
        ),
        (host, required)
    );
    assert_eq!(history(copied.as_sampler()), &[3, 11, 7]);
    let (custody, scope) = native.into_parts();
    assert!(matches!(
        rejected.adopt_storage_individually(
            &scope,
            [(
                2u32,
                crate::working_memory::StorageAllocation::new(1, pool.host_placement_handle())
            )]
        ),
        Err(WorkingMemoryError::DomainAllowanceExceeded {
            required_bytes: 1,
            available_bytes: 0,
            ..
        })
    ));
    drop((sampler, preparation, run, layout, copied, custody));
    assert_eq!(pool.payload_used_bytes().unwrap(), UNCOPIED_BYTES + host);
    scope.certify().unwrap();
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}
