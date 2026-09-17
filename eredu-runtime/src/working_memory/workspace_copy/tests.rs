use super::*;
use eredu_nn::workspace::{
    WorkspaceContext, WorkspaceDtype, WorkspaceExistingStorage, WorkspaceHostBound,
    WorkspaceLayout, WorkspaceMechanisms, WorkspaceOperation, WorkspaceOperationBound,
    WorkspaceOperationKind, WorkspaceOutputStorage, WorkspaceTensor,
};

#[derive(Debug, Default)]
struct Facts {
    missing_tensor: bool,
    missing_host: bool,
}

impl WorkspaceMechanisms for Facts {
    fn operation_bound(
        &self,
        operation: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, eredu_nn::Error> {
        if self.missing_tensor {
            return Ok(None);
        }
        let [layout] = operation.outputs.as_slice() else {
            panic!("closed copy has one output");
        };
        let capacity = layout.bytes()?.div_ceil(16) * 16;
        let effect = match operation.kind {
            WorkspaceOperationKind::Contiguous => WorkspaceOutputStorage::AllocateOrAliasInputs {
                bytes: capacity,
                inputs: vec![0],
            },
            WorkspaceOperationKind::DeepCopy => WorkspaceOutputStorage::Allocate(capacity),
            _ => panic!("only the closed copy program is allowed"),
        };
        Ok(Some(WorkspaceOperationBound {
            outputs: vec![effect],
            scratch_bytes: 3,
            assumptions: "fixture: sixteen-byte padded result and three scratch bytes".into(),
        }))
    }

    fn host_workspace_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceHostBound>, eredu_nn::Error> {
        Ok((!self.missing_host).then(|| WorkspaceHostBound {
            bytes: 5,
            assumptions: "fixture: five disjoint staging bytes per operation".into(),
        }))
    }
}

const COPY_BYTES: u64 = 48; // two padded outputs, two scratch and two host bounds
const SOURCE_BYTES: u64 = 64;

fn plan_parts(
    pool: &WorkingMemoryPool,
    key: u32,
    source_bytes: u64,
    copies: usize,
    facts: Facts,
) -> (WorkspaceIsolatedCopyPlan, RegisteredWorkspaceStorage<u32>) {
    let context = WorkspaceContext::new(facts);
    let root = WorkspaceExistingStorage::new(Some(source_bytes), &context);
    let source = RegisteredWorkspaceStorage::bind(pool, &context, [(key, root.clone())]).unwrap();
    let sources = (0..copies)
        .map(|_| {
            WorkspaceTensor::existing_with_storage(
                WorkspaceLayout::new(&[2], WorkspaceDtype::Uint32).unwrap(),
                &root,
                &context,
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let plan =
        WorkspaceIsolatedCopyPlan::prepare(&context, source.borrowed_storage(), &sources).unwrap();
    (plan, source)
}

fn copy_plan(
    pool: &WorkingMemoryPool,
    key: u32,
    source_bytes: u64,
    copies: usize,
) -> RegisteredWorkspaceCopy<u32> {
    let (plan, source) = plan_parts(pool, key, source_bytes, copies, Facts::default());
    assert_eq!(plan.incremental_bytes(), Some(COPY_BYTES * copies as u64));
    RegisteredWorkspaceCopy::bind(plan, source).unwrap()
}

fn usage(pool: &WorkingMemoryPool) -> (u64, u64, u64) {
    (
        pool.used_bytes().unwrap(),
        pool.peak_bytes().unwrap(),
        pool.effective_capacity().unwrap(),
    )
}

fn settle_unused(copy: AdmittedWorkspaceCopy) {
    let (custody, scope) = copy.into_parts();
    scope.certify().unwrap();
    drop(custody);
}

#[test]
fn exact_capacity_counts_source_plus_closed_program_and_preserves_diagnostics() {
    let pool = WorkingMemoryPool::new(4096, 7).unwrap();
    let source = pool.register_storage([(1u32, SOURCE_BYTES)]).unwrap();
    let plan = copy_plan(&pool, 1, SOURCE_BYTES, 1);
    let mut edited = plan.report().clone();
    edited.total_bytes = Some(0);
    edited.residual.as_mut().unwrap().total_bytes = Some(0);
    assert_eq!(plan.report().total_bytes, Some(COPY_BYTES));
    let capacity = 7 + SOURCE_BYTES + COPY_BYTES;
    let before = usage(&pool);
    assert!(matches!(
        pool.admit_workspace_copy(plan, WorkspaceCopyLimits::new(capacity - 1)),
        Err(WorkspaceCopyAdmissionError::Memory(WorkingMemoryError::BudgetExceeded {
            required_bytes: COPY_BYTES,
            available_bytes
        })) if available_bytes == COPY_BYTES - 1
    ));
    assert_eq!(usage(&pool), before);
    let admitted = pool
        .admit_workspace_copy(
            copy_plan(&pool, 1, SOURCE_BYTES, 1),
            WorkspaceCopyLimits::new(capacity),
        )
        .unwrap();
    assert_eq!(admitted.bytes(), COPY_BYTES);
    assert_eq!(usage(&pool), (capacity, capacity, capacity));
    let (custody, scope) = admitted.into_parts();
    assert!(custody.pool().same_domain(&pool));
    assert_eq!(custody.bytes(), COPY_BYTES);
    scope.certify().unwrap();
    drop(custody);
    assert_eq!(pool.used_bytes().unwrap(), 7 + SOURCE_BYTES);
    assert_eq!(pool.effective_capacity().unwrap(), 4096);
    drop(source);
    assert_eq!(pool.used_bytes().unwrap(), 7);
}

#[test]
fn application_safety_foreign_and_unknown_rejections_leave_accounting_unchanged() {
    let pool = WorkingMemoryPool::new(4096, 0).unwrap();
    let _source = pool.register_storage([(1u32, SOURCE_BYTES)]).unwrap();
    let before = usage(&pool);
    let mut limits = WorkspaceCopyLimits::new(4096);
    limits.safety_reserve_bytes = 9;
    limits.application_memory_budget_bytes = Some(COPY_BYTES + 8);
    assert!(matches!(
        pool.admit_workspace_copy(copy_plan(&pool, 1, SOURCE_BYTES, 1), limits),
        Err(WorkspaceCopyAdmissionError::ApplicationBudgetExceeded {
            required_bytes,
            budget_bytes
        }) if required_bytes == COPY_BYTES + 9 && budget_bytes == COPY_BYTES + 8
    ));
    limits.safety_reserve_bytes = u64::MAX;
    assert!(matches!(
        pool.admit_workspace_copy(copy_plan(&pool, 1, SOURCE_BYTES, 1), limits),
        Err(WorkspaceCopyAdmissionError::Memory(
            WorkingMemoryError::Overflow
        ))
    ));
    let other = WorkingMemoryPool::new(4096, 0).unwrap();
    assert!(matches!(
        other.admit_workspace_copy(
            copy_plan(&pool, 1, SOURCE_BYTES, 1),
            WorkspaceCopyLimits::new(4096)
        ),
        Err(WorkspaceCopyAdmissionError::Memory(
            WorkingMemoryError::IdentityMismatch
        ))
    ));
    assert_eq!(other.used_bytes().unwrap(), 0);
    for facts in [
        Facts {
            missing_tensor: true,
            missing_host: false,
        },
        Facts {
            missing_tensor: false,
            missing_host: true,
        },
    ] {
        let (plan, source) = plan_parts(&pool, 1, SOURCE_BYTES, 1, facts);
        assert!(matches!(
            RegisteredWorkspaceCopy::bind(plan, source),
            Err(WorkspaceCopyAdmissionError::Memory(
                WorkingMemoryError::UnknownBound
            ))
        ));
    }
    assert_eq!(usage(&pool), before);
    limits.application_memory_budget_bytes = Some(COPY_BYTES + 9);
    limits.safety_reserve_bytes = 9;
    limits.capacity_bytes = SOURCE_BYTES + COPY_BYTES + 9;
    let admitted = pool
        .admit_workspace_copy(copy_plan(&pool, 1, SOURCE_BYTES, 1), limits)
        .unwrap();
    assert_eq!(admitted.bytes(), COPY_BYTES + 9);
    settle_unused(admitted);
}

#[test]
fn exact_selection_identity_cannot_be_replaced_by_an_equal_inventory() {
    let pool = WorkingMemoryPool::new(4096, 0).unwrap();
    let _source = pool.register_storage([(1u32, SOURCE_BYTES)]).unwrap();
    let (plan, original) = plan_parts(&pool, 1, SOURCE_BYTES, 1, Facts::default());
    let (_, different) = plan_parts(&pool, 1, SOURCE_BYTES, 1, Facts::default());
    let before = usage(&pool);
    assert!(matches!(
        RegisteredWorkspaceCopy::bind(plan, different),
        Err(WorkspaceCopyAdmissionError::Memory(
            WorkingMemoryError::IdentityMismatch
        ))
    ));
    assert_eq!(usage(&pool), before);
    drop(original);
    for (key, bytes) in [(2u32, SOURCE_BYTES), (1, SOURCE_BYTES - 1)] {
        let context = WorkspaceContext::new(Facts::default());
        let root = WorkspaceExistingStorage::new(Some(bytes), &context);
        assert!(matches!(
            RegisteredWorkspaceStorage::bind(&pool, &context, [(key, root)]),
            Err(WorkingMemoryError::IdentityMismatch)
        ));
    }
    assert_eq!(usage(&pool), before);
}

#[test]
fn source_pin_retires_at_certification_while_destination_aliases_keep_their_charge() {
    let pool = WorkingMemoryPool::new(4096, 0).unwrap();
    let source = pool.register_storage([(1u32, SOURCE_BYTES)]).unwrap();
    let admitted = pool
        .admit_workspace_copy(
            copy_plan(&pool, 1, SOURCE_BYTES, 2),
            WorkspaceCopyLimits::new(SOURCE_BYTES + 2 * COPY_BYTES),
        )
        .unwrap();
    let (custody, scope) = admitted.into_parts();
    drop(source);
    assert_eq!(pool.used_bytes().unwrap(), SOURCE_BYTES + 2 * COPY_BYTES);
    let mut outputs = scope
        .adopt_storage_individually([(2u32, 16), (3, 16)])
        .unwrap();
    let first = outputs.remove(&2).unwrap();
    let second = outputs.remove(&3).unwrap();
    let alias = first.clone();
    // Alias publication costs zero but preserves its original account origin.
    let extra = scope.adopt_storage_individually([(2u32, 16)]).unwrap();
    assert_eq!(pool.used_bytes().unwrap(), SOURCE_BYTES + 2 * COPY_BYTES);
    scope.certify().unwrap();
    assert_eq!(pool.used_bytes().unwrap(), 2 * COPY_BYTES);
    drop((custody, second, first, extra));
    assert_eq!(pool.used_bytes().unwrap(), 16);
    assert_eq!(
        pool.effective_capacity().unwrap(),
        SOURCE_BYTES + 2 * COPY_BYTES
    );
    drop(alias);
    assert_eq!(pool.used_bytes().unwrap(), 0);
    assert_eq!(pool.effective_capacity().unwrap(), 4096);
}

#[test]
fn partial_publication_and_abandoned_scope_keep_credit_and_source_pins_quarantined() {
    let pool = WorkingMemoryPool::new(4096, 0).unwrap();
    let source = pool.register_storage([(1u32, SOURCE_BYTES)]).unwrap();
    let admitted = pool
        .admit_workspace_copy(
            copy_plan(&pool, 1, SOURCE_BYTES, 2),
            WorkspaceCopyLimits::new(4096),
        )
        .unwrap();
    let (custody, scope) = admitted.into_parts();
    let mut outputs = scope
        .adopt_storage_individually([(2u32, 16), (3, 16)])
        .unwrap();
    let attached = outputs.remove(&2).unwrap();
    // Simulate an attachment error: unprocessed registrations retire, returning
    // their credit to the still-live envelope instead of refunding the pool.
    drop(outputs);
    drop((source, custody));
    assert_eq!(pool.used_bytes().unwrap(), SOURCE_BYTES + 2 * COPY_BYTES);
    drop(scope);
    drop(attached);
    assert_eq!(pool.used_bytes().unwrap(), SOURCE_BYTES + 2 * COPY_BYTES);
    assert!(matches!(
        pool.acquire_unquoted(),
        Err(WorkingMemoryError::ReservedWorkActive)
    ));
    let source_still_pinned = pool.pin_registered_storage([(1u32, SOURCE_BYTES)]).unwrap();
    drop(source_still_pinned);
    assert_eq!(pool.used_bytes().unwrap(), SOURCE_BYTES + 2 * COPY_BYTES);
}

#[test]
fn abandoned_unsplit_operation_quarantines_without_an_implicit_completion_claim() {
    let pool = WorkingMemoryPool::new(4096, 0).unwrap();
    let source = pool.register_storage([(1u32, SOURCE_BYTES)]).unwrap();
    let admitted = pool
        .admit_workspace_copy(
            copy_plan(&pool, 1, SOURCE_BYTES, 1),
            WorkspaceCopyLimits::new(4096),
        )
        .unwrap();
    drop((source, admitted));
    assert_eq!(pool.used_bytes().unwrap(), SOURCE_BYTES + COPY_BYTES);
    assert!(pool.pin_registered_storage([(1u32, SOURCE_BYTES)]).is_ok());
}

#[test]
fn source_origin_quarantine_after_cold_preparation_is_rechecked_before_commit() {
    let pool = WorkingMemoryPool::new(4096, 0).unwrap();
    let _source = pool.register_storage([(1u32, SOURCE_BYTES)]).unwrap();
    let first = pool
        .admit_workspace_copy(
            copy_plan(&pool, 1, SOURCE_BYTES, 1),
            WorkspaceCopyLimits::new(4096),
        )
        .unwrap();
    let (custody, scope) = first.into_parts();
    let _output = scope.adopt_storage_individually([(2u32, 16)]).unwrap();
    // Source accounting need not close: native source settlement is separately
    // established by the provider. Healthy live origins still pay their bytes.
    let healthy = pool
        .admit_workspace_copy(copy_plan(&pool, 2, 16, 1), WorkspaceCopyLimits::new(4096))
        .unwrap();
    settle_unused(healthy);
    let later = copy_plan(&pool, 2, 16, 1);
    drop((scope, custody));
    let before = usage(&pool);
    assert!(matches!(
        pool.admit_workspace_copy(later, WorkspaceCopyLimits::new(4096)),
        Err(WorkspaceCopyAdmissionError::Memory(
            WorkingMemoryError::ExecutionFenced
        ))
    ));
    assert_eq!(usage(&pool), before);
}

#[test]
fn registered_descendant_remains_source_after_original_custody_retires_and_keeps_ceiling() {
    let pool = WorkingMemoryPool::new(4096, 0).unwrap();
    let source = pool.register_storage([(1u32, SOURCE_BYTES)]).unwrap();
    let ceiling = SOURCE_BYTES + COPY_BYTES;
    let first = pool
        .admit_workspace_copy(
            copy_plan(&pool, 1, SOURCE_BYTES, 1),
            WorkspaceCopyLimits::new(ceiling),
        )
        .unwrap();
    let (custody, scope) = first.into_parts();
    let output = scope.adopt_storage_individually([(2u32, 16)]).unwrap();
    scope.certify().unwrap();
    drop((custody, source));
    assert_eq!(pool.used_bytes().unwrap(), 16);
    let before = usage(&pool);
    assert!(matches!(
        pool.admit_workspace_copy(copy_plan(&pool, 2, 16, 3), WorkspaceCopyLimits::new(4096)),
        Err(WorkspaceCopyAdmissionError::Memory(
            WorkingMemoryError::BudgetExceeded { .. }
        ))
    ));
    assert_eq!(usage(&pool), before);
    let next = pool
        .admit_workspace_copy(
            copy_plan(&pool, 2, 16, 2),
            WorkspaceCopyLimits::new(ceiling),
        )
        .unwrap();
    drop(output);
    assert_eq!(pool.used_bytes().unwrap(), 16 + 2 * COPY_BYTES);
    settle_unused(next);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn counter_overflow_poison_and_unquoted_exclusion_precede_destination_commit() {
    let pool = WorkingMemoryPool::new(4096, 0).unwrap();
    let _source = pool.register_storage([(1u32, SOURCE_BYTES)]).unwrap();
    let lease = pool.acquire_unquoted().unwrap();
    let before = usage(&pool);
    assert!(matches!(
        pool.admit_workspace_copy(
            copy_plan(&pool, 1, SOURCE_BYTES, 1),
            WorkspaceCopyLimits::new(4096)
        ),
        Err(WorkspaceCopyAdmissionError::Memory(
            WorkingMemoryError::UnknownBound
        ))
    ));
    assert_eq!(usage(&pool), before);
    drop(lease);
    let old_next = {
        let mut usage = pool.0.usage.lock().unwrap();
        std::mem::replace(&mut usage.next_funding, u64::MAX)
    };
    assert!(matches!(
        pool.admit_workspace_copy(
            copy_plan(&pool, 1, SOURCE_BYTES, 1),
            WorkspaceCopyLimits::new(4096)
        ),
        Err(WorkspaceCopyAdmissionError::Memory(
            WorkingMemoryError::Overflow
        ))
    ));
    assert_eq!(usage(&pool), before);
    pool.0.usage.lock().unwrap().next_funding = old_next;
    let plan = copy_plan(&pool, 1, SOURCE_BYTES, 1);
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _lock = pool.0.usage.lock().unwrap();
        panic!("poison copy-account fixture");
    }))
    .is_err());
    assert!(matches!(
        pool.admit_workspace_copy(plan, WorkspaceCopyLimits::new(4096)),
        Err(WorkspaceCopyAdmissionError::Memory(
            WorkingMemoryError::Poisoned
        ))
    ));
    let usage = pool.0.usage.lock().unwrap_err().into_inner();
    assert_eq!(usage.reserved, 0);
    assert_eq!(usage.registered, SOURCE_BYTES);
    assert_eq!(usage.peak, before.1);
    assert!(usage.funding.is_empty());
}

#[test]
fn empty_copy_is_explicit_and_zero_byte_sources_still_keep_identity() {
    let pool = WorkingMemoryPool::new(4096, 0).unwrap();
    let context = WorkspaceContext::new(Facts::default());
    let source = RegisteredWorkspaceStorage::bind(
        &pool,
        &context,
        std::iter::empty::<(u32, WorkspaceExistingStorage)>(),
    )
    .unwrap();
    let plan =
        WorkspaceIsolatedCopyPlan::prepare(&context, source.borrowed_storage(), &[]).unwrap();
    let copy = pool
        .admit_workspace_copy(
            RegisteredWorkspaceCopy::bind(plan, source).unwrap(),
            WorkspaceCopyLimits::new(4096),
        )
        .unwrap();
    assert_eq!(copy.bytes(), 0);
    assert!(matches!(
        pool.acquire_unquoted(),
        Err(WorkingMemoryError::ReservedWorkActive)
    ));
    settle_unused(copy);
    assert_eq!(pool.used_bytes().unwrap(), 0);
    let source = pool.register_storage([(1u32, 0)]).unwrap();
    let context = WorkspaceContext::new(Facts::default());
    let root = WorkspaceExistingStorage::new(Some(0), &context);
    let pin = RegisteredWorkspaceStorage::bind(&pool, &context, [(1u32, root.clone())]).unwrap();
    let empty = WorkspaceTensor::existing_with_storage(
        WorkspaceLayout::new(&[0], WorkspaceDtype::Uint32).unwrap(),
        &root,
        &context,
    )
    .unwrap();
    let plan =
        WorkspaceIsolatedCopyPlan::prepare(&context, pin.borrowed_storage(), &[empty]).unwrap();
    // Empty copies have no result payload; this fixture still declares its
    // explicit per-operation scratch/staging bounds.
    assert_eq!(plan.incremental_bytes(), Some(16));
    let copy = pool
        .admit_workspace_copy(
            RegisteredWorkspaceCopy::bind(plan, pin).unwrap(),
            WorkspaceCopyLimits::new(4096),
        )
        .unwrap();
    drop(source);
    settle_unused(copy);
    assert!(matches!(
        pool.pin_registered_storage([(1u32, 0)]),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
}
