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
    physical: Option<(
        std::sync::Arc<eredu_core::MemoryTopology>,
        eredu_core::MemoryPlacement,
    )>,
}

impl Facts {
    pub(super) fn with_pool(mut self, pool: &MemoryLedger) -> Self {
        self.physical = Some((
            pool.topology_handle(),
            (*pool.host_placement_handle()).clone(),
        ));
        self
    }
}

impl WorkspaceMechanisms for Facts {
    fn memory_topology(&self) -> Option<&eredu_core::MemoryTopology> {
        self.physical.as_ref().map(|value| value.0.as_ref())
    }
    fn output_placement(
        &self,
        _: eredu_nn::workspace::WorkspaceOperationView<'_>,
        _: usize,
    ) -> Option<&eredu_core::MemoryPlacement> {
        self.physical.as_ref().map(|value| &value.1)
    }
    fn scratch_placement(
        &self,
        _: eredu_nn::workspace::WorkspaceOperationView<'_>,
    ) -> Option<&eredu_core::MemoryPlacement> {
        self.physical.as_ref().map(|value| &value.1)
    }

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
    pool: &MemoryLedger,
    key: u32,
    source_bytes: u64,
    copies: usize,
    facts: Facts,
) -> (WorkspaceIsolatedCopyPlan, RegisteredWorkspaceStorage<u32>) {
    let context = WorkspaceContext::new(facts.with_pool(pool));
    let root = WorkspaceExistingStorage::try_new_placed(
        Some(source_bytes),
        &pool.host_placement_handle(),
        &context,
    )
    .unwrap();
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
    pool: &MemoryLedger,
    key: u32,
    source_bytes: u64,
    copies: usize,
) -> RegisteredWorkspaceCopy<u32> {
    let (plan, source) = plan_parts(pool, key, source_bytes, copies, Facts::default());
    assert_eq!(plan.incremental_bytes(), Some(COPY_BYTES * copies as u64));
    RegisteredWorkspaceCopy::bind(plan, source).unwrap()
}

fn charge(
    pool: &MemoryLedger,
    plan: &RegisteredWorkspaceCopy<u32>,
    limits: &WorkspaceCopyLimits,
) -> u64 {
    pool.workspace_copy_requirements(plan, limits)
        .unwrap()
        .get(pool.topology().host_domain())
        .unwrap()
        .total()
        .unwrap()
}
fn live(pool: &MemoryLedger) -> u64 {
    pool.snapshot().unwrap().domains[0].current_charge_bytes
}
fn physical_limits(bytes: u64) -> eredu_core::MemoryLimitDeclarations {
    eredu_core::MemoryLimitDeclarations::new([(
        "host".into(),
        eredu_core::MemoryLimit::Finite(bytes),
    )])
}
fn publish(
    scope: &WorkingMemoryFundingScope,
    entries: &[(u32, u64)],
) -> crate::working_memory::StorageRegistrations<u32> {
    let pool = scope.pool();
    crate::working_memory::StoragePublicationLayout::new(entries.len())
        .unwrap()
        .fund(pool)
        .unwrap()
        .adopt_storage_individually(
            scope,
            entries.iter().map(|&(key, bytes)| {
                (
                    key,
                    crate::working_memory::StorageAllocation::new(
                        bytes,
                        pool.host_placement_handle(),
                    ),
                )
            }),
        )
        .unwrap()
}

fn usage(pool: &MemoryLedger) -> (u64, u64, u64) {
    (
        pool.payload_used_bytes().unwrap(),
        pool.payload_peak_bytes().unwrap(),
        pool.payload_effective_capacity().unwrap(),
    )
}

fn settle_unused(copy: AdmittedWorkspaceCopy) {
    let (custody, scope) = copy.into_parts();
    scope.certify().unwrap();
    drop(custody);
}

#[test]
fn exact_capacity_counts_source_plus_closed_program_and_preserves_diagnostics() {
    let pool = crate::working_memory::memory_fixture::host_ledger(1_000_000, 7).unwrap();
    let source = pool.register_host_storage([(1u32, SOURCE_BYTES)]).unwrap();
    let plan = copy_plan(&pool, 1, SOURCE_BYTES, 1);
    let mut edited = plan.report().clone();
    edited.total_bytes = Some(0);
    edited.residual.as_mut().unwrap().total_bytes = Some(0);
    assert_eq!(plan.report().total_bytes, Some(COPY_BYTES));
    let required = charge(&pool, &plan, &WorkspaceCopyLimits::default());
    let capacity = live(&pool) + required;
    let before = usage(&pool);
    assert!(matches!(
        pool.admit_workspace_copy(plan, WorkspaceCopyLimits::new(physical_limits(capacity - 1))),
        Err(WorkspaceCopyAdmissionError::Memory(WorkingMemoryError::Domain(eredu_core::MemoryDomainError::BudgetExceeded {
            requested_bytes, limit_bytes, existing_bytes, ..
        }))) if requested_bytes == required && limit_bytes - existing_bytes == required - 1
    ));
    assert_eq!(usage(&pool), before);
    let admitted = pool
        .admit_workspace_copy(
            copy_plan(&pool, 1, SOURCE_BYTES, 1),
            WorkspaceCopyLimits::new(physical_limits(capacity)),
        )
        .unwrap();
    assert_eq!(
        admitted
            .requirements()
            .get(pool.topology().host_domain())
            .unwrap()
            .total()
            .unwrap(),
        required
    );
    assert_eq!(
        pool.snapshot().unwrap().domains[0].current_charge_bytes,
        capacity
    );
    assert_eq!(
        pool.payload_used_bytes().unwrap(),
        7 + SOURCE_BYTES + COPY_BYTES
    );
    let (custody, scope) = admitted.into_parts();
    assert!(custody.pool().same_ledger(&pool));
    assert_eq!(
        custody
            .requirements()
            .get(pool.topology().host_domain())
            .unwrap()
            .total()
            .unwrap(),
        required
    );
    scope.certify().unwrap();
    drop(custody);
    assert_eq!(pool.payload_used_bytes().unwrap(), 7 + SOURCE_BYTES);
    assert_eq!(pool.payload_effective_capacity().unwrap(), 1_000_000);
    drop(source);
    assert_eq!(pool.payload_used_bytes().unwrap(), 7);
}

#[test]
fn headroom_foreign_and_unknown_rejections_leave_accounting_unchanged() {
    let pool = crate::working_memory::memory_fixture::host_ledger(1_000_000, 0).unwrap();
    let _source = pool.register_host_storage([(1u32, SOURCE_BYTES)]).unwrap();
    let before = usage(&pool);
    let mut limits = WorkspaceCopyLimits::new(crate::working_memory::memory_fixture::host_limits(
        1_000_000,
    ));
    limits.additional_headroom = eredu_core::MemoryHeadroomDeclarations::new([("host".into(), 9)]);
    let prepared = copy_plan(&pool, 1, SOURCE_BYTES, 1);
    let required = charge(&pool, &prepared, &limits);
    let capacity = live(&pool) + required;
    limits.memory_limits = physical_limits(capacity - 1);
    assert!(matches!(
        pool.admit_workspace_copy(prepared, limits.clone()),
        Err(WorkspaceCopyAdmissionError::Memory(WorkingMemoryError::Domain(eredu_core::MemoryDomainError::BudgetExceeded { requested_bytes: required_bytes, limit_bytes, existing_bytes, .. }))) if required_bytes == required && limit_bytes - existing_bytes == required - 1
    ));
    limits.additional_headroom =
        eredu_core::MemoryHeadroomDeclarations::new([("host".into(), u64::MAX)]);
    assert!(matches!(
        pool.admit_workspace_copy(copy_plan(&pool, 1, SOURCE_BYTES, 1), limits.clone()),
        Err(WorkspaceCopyAdmissionError::Memory(
            WorkingMemoryError::Domain(eredu_core::MemoryDomainError::Overflow)
        ))
    ));
    let other = crate::working_memory::memory_fixture::host_ledger(1_000_000, 0).unwrap();
    assert!(matches!(
        other.admit_workspace_copy(
            copy_plan(&pool, 1, SOURCE_BYTES, 1),
            WorkspaceCopyLimits::new(crate::working_memory::memory_fixture::host_limits(
                1_000_000
            ))
        ),
        Err(WorkspaceCopyAdmissionError::Memory(
            WorkingMemoryError::IdentityMismatch
        ))
    ));
    assert_eq!(other.payload_used_bytes().unwrap(), 0);
    for facts in [
        Facts {
            missing_tensor: true,
            missing_host: false,
            physical: None,
        },
        Facts {
            missing_tensor: false,
            missing_host: true,
            physical: None,
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
    limits.additional_headroom = eredu_core::MemoryHeadroomDeclarations::new([("host".into(), 9)]);
    limits.memory_limits = physical_limits(capacity);
    let admitted = pool
        .admit_workspace_copy(copy_plan(&pool, 1, SOURCE_BYTES, 1), limits.clone())
        .unwrap();
    assert_eq!(
        admitted
            .requirements()
            .get(pool.topology().host_domain())
            .unwrap()
            .total()
            .unwrap(),
        required
    );
    settle_unused(admitted);
}

#[test]
fn exact_selection_identity_cannot_be_replaced_by_an_equal_inventory() {
    let pool = crate::working_memory::memory_fixture::host_ledger(1_000_000, 0).unwrap();
    let _source = pool.register_host_storage([(1u32, SOURCE_BYTES)]).unwrap();
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
        let context = WorkspaceContext::new(Facts::default().with_pool(&pool));
        let root = WorkspaceExistingStorage::try_new_placed(
            Some(bytes),
            &pool.host_placement_handle(),
            &context,
        )
        .unwrap();
        assert!(matches!(
            RegisteredWorkspaceStorage::bind(&pool, &context, [(key, root)]),
            Err(WorkingMemoryError::IdentityMismatch)
        ));
    }
    assert_eq!(usage(&pool), before);
}

#[test]
fn source_pin_retires_at_certification_while_destination_aliases_keep_their_charge() {
    let pool = crate::working_memory::memory_fixture::host_ledger(1_000_000, 0).unwrap();
    let source = pool.register_host_storage([(1u32, SOURCE_BYTES)]).unwrap();
    let admitted = pool
        .admit_workspace_copy(
            copy_plan(&pool, 1, SOURCE_BYTES, 2),
            WorkspaceCopyLimits::new(crate::working_memory::memory_fixture::host_limits(100_000)),
        )
        .unwrap();
    let (custody, scope) = admitted.into_parts();
    drop(source);
    assert_eq!(
        pool.payload_used_bytes().unwrap(),
        SOURCE_BYTES + 2 * COPY_BYTES
    );
    let mut outputs = publish(&scope, &[(2u32, 16), (3, 16)]);
    let first = outputs.remove(&2).unwrap();
    let second = outputs.remove(&3).unwrap();
    let alias = first.clone();
    // Alias publication costs zero but preserves its original account origin.
    let extra = publish(&scope, &[(2u32, 16)]);
    assert_eq!(
        pool.payload_used_bytes().unwrap(),
        SOURCE_BYTES + 2 * COPY_BYTES
    );
    scope.certify().unwrap();
    assert_eq!(pool.payload_used_bytes().unwrap(), 2 * COPY_BYTES);
    drop((custody, second, first, extra));
    assert_eq!(pool.payload_used_bytes().unwrap(), 16);
    assert_eq!(pool.payload_effective_capacity().unwrap(), 100_000);
    drop(alias);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    assert_eq!(pool.payload_effective_capacity().unwrap(), 1_000_000);
}

#[test]
fn partial_publication_and_abandoned_scope_keep_credit_and_source_pins_quarantined() {
    let pool = crate::working_memory::memory_fixture::host_ledger(1_000_000, 0).unwrap();
    let source = pool.register_host_storage([(1u32, SOURCE_BYTES)]).unwrap();
    let admitted = pool
        .admit_workspace_copy(
            copy_plan(&pool, 1, SOURCE_BYTES, 2),
            WorkspaceCopyLimits::new(crate::working_memory::memory_fixture::host_limits(
                1_000_000,
            )),
        )
        .unwrap();
    let (custody, scope) = admitted.into_parts();
    let mut outputs = publish(&scope, &[(2u32, 16), (3, 16)]);
    let attached = outputs.remove(&2).unwrap();
    // Simulate an attachment error: unprocessed registrations retire, returning
    // their credit to the still-live envelope instead of refunding the pool.
    drop(outputs);
    drop((source, custody));
    assert_eq!(
        pool.payload_used_bytes().unwrap(),
        SOURCE_BYTES + 2 * COPY_BYTES
    );
    drop(scope);
    drop(attached);
    assert_eq!(
        pool.payload_used_bytes().unwrap(),
        SOURCE_BYTES + 2 * COPY_BYTES
    );
    assert!(matches!(
        pool.acquire_unquoted(),
        Err(WorkingMemoryError::ReservedWorkActive)
    ));
    assert_eq!(
        pool.registered_capacity_for_test(&1u32).unwrap(),
        Some(SOURCE_BYTES)
    );
    assert_eq!(
        pool.payload_used_bytes().unwrap(),
        SOURCE_BYTES + 2 * COPY_BYTES
    );
}

#[test]
fn abandoned_unsplit_operation_quarantines_without_an_implicit_completion_claim() {
    let pool = crate::working_memory::memory_fixture::host_ledger(1_000_000, 0).unwrap();
    let source = pool.register_host_storage([(1u32, SOURCE_BYTES)]).unwrap();
    let admitted = pool
        .admit_workspace_copy(
            copy_plan(&pool, 1, SOURCE_BYTES, 1),
            WorkspaceCopyLimits::new(crate::working_memory::memory_fixture::host_limits(
                1_000_000,
            )),
        )
        .unwrap();
    drop((source, admitted));
    assert_eq!(
        pool.payload_used_bytes().unwrap(),
        SOURCE_BYTES + COPY_BYTES
    );
    assert_eq!(
        pool.registered_capacity_for_test(&1u32).unwrap(),
        Some(SOURCE_BYTES)
    );
}

#[test]
fn source_origin_quarantine_after_cold_preparation_is_rechecked_before_commit() {
    let pool = crate::working_memory::memory_fixture::host_ledger(1_000_000, 0).unwrap();
    let _source = pool.register_host_storage([(1u32, SOURCE_BYTES)]).unwrap();
    let first = pool
        .admit_workspace_copy(
            copy_plan(&pool, 1, SOURCE_BYTES, 1),
            WorkspaceCopyLimits::new(crate::working_memory::memory_fixture::host_limits(
                1_000_000,
            )),
        )
        .unwrap();
    let (custody, scope) = first.into_parts();
    let _output = publish(&scope, &[(2u32, 16)]);
    // Source accounting need not close: native source settlement is separately
    // established by the provider. Healthy live origins still pay their bytes.
    let healthy = pool
        .admit_workspace_copy(
            copy_plan(&pool, 2, 16, 1),
            WorkspaceCopyLimits::new(crate::working_memory::memory_fixture::host_limits(
                1_000_000,
            )),
        )
        .unwrap();
    settle_unused(healthy);
    let later = copy_plan(&pool, 2, 16, 1);
    drop((scope, custody));
    let before = usage(&pool);
    assert!(matches!(
        pool.admit_workspace_copy(
            later,
            WorkspaceCopyLimits::new(crate::working_memory::memory_fixture::host_limits(
                1_000_000
            ))
        ),
        Err(WorkspaceCopyAdmissionError::Memory(
            WorkingMemoryError::ExecutionFenced
        ))
    ));
    assert_eq!(usage(&pool), before);
}

#[test]
fn registered_descendant_remains_source_after_original_custody_retires_and_keeps_ceiling() {
    let pool = crate::working_memory::memory_fixture::host_ledger(1_000_000, 0).unwrap();
    let source = pool.register_host_storage([(1u32, SOURCE_BYTES)]).unwrap();
    let first_plan = copy_plan(&pool, 1, SOURCE_BYTES, 1);
    let constructor = charge(&pool, &first_plan, &WorkspaceCopyLimits::default());
    drop(first_plan);
    let publication = crate::working_memory::MemoryLedger::storage_metadata_control_bytes()
        .unwrap()
        + crate::working_memory::StoragePublicationLayout::<u32>::new(2)
            .unwrap()
            .requested_bytes();
    let ceiling = SOURCE_BYTES + 2 * constructor + 8 * publication;
    let first = pool
        .admit_workspace_copy(
            copy_plan(&pool, 1, SOURCE_BYTES, 1),
            WorkspaceCopyLimits::new(crate::working_memory::memory_fixture::host_limits(ceiling)),
        )
        .unwrap();
    let (custody, scope) = first.into_parts();
    let output = publish(&scope, &[(2u32, 16)]);
    scope.certify().unwrap();
    drop((custody, source));
    assert_eq!(pool.payload_used_bytes().unwrap(), 16);
    let before = usage(&pool);
    assert!(matches!(
        pool.admit_workspace_copy(
            copy_plan(&pool, 2, 16, 10_000),
            WorkspaceCopyLimits::new(crate::working_memory::memory_fixture::host_limits(
                1_000_000
            ))
        ),
        Err(WorkspaceCopyAdmissionError::Memory(
            WorkingMemoryError::Domain(eredu_core::MemoryDomainError::BudgetExceeded { .. })
        ))
    ));
    assert_eq!(usage(&pool), before);
    let next = pool
        .admit_workspace_copy(
            copy_plan(&pool, 2, 16, 2),
            WorkspaceCopyLimits::new(crate::working_memory::memory_fixture::host_limits(ceiling)),
        )
        .unwrap();
    drop(output);
    assert_eq!(pool.payload_used_bytes().unwrap(), 16 + 2 * COPY_BYTES);
    settle_unused(next);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[test]
fn counter_overflow_poison_and_unquoted_exclusion_precede_destination_commit() {
    let pool = crate::working_memory::memory_fixture::host_ledger(1_000_000, 0).unwrap();
    let _source = pool.register_host_storage([(1u32, SOURCE_BYTES)]).unwrap();
    let lease = pool.acquire_unquoted().unwrap();
    let before = usage(&pool);
    assert!(matches!(
        pool.admit_workspace_copy(
            copy_plan(&pool, 1, SOURCE_BYTES, 1),
            WorkspaceCopyLimits::new(crate::working_memory::memory_fixture::host_limits(
                1_000_000
            ))
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
            WorkspaceCopyLimits::new(crate::working_memory::memory_fixture::host_limits(
                1_000_000
            ))
        ),
        Err(WorkspaceCopyAdmissionError::Memory(
            WorkingMemoryError::Overflow
        ))
    ));
    assert_eq!(usage(&pool), before);
    pool.0.usage.lock().unwrap().next_funding = old_next;
    let plan = copy_plan(&pool, 1, SOURCE_BYTES, 1);
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _lock = pool.0.usage.lock().unwrap();
            panic!("poison copy-account fixture");
        }))
        .is_err()
    );
    assert!(matches!(
        pool.admit_workspace_copy(
            plan,
            WorkspaceCopyLimits::new(crate::working_memory::memory_fixture::host_limits(
                1_000_000
            ))
        ),
        Err(WorkspaceCopyAdmissionError::Memory(
            WorkingMemoryError::Poisoned
        ))
    ));
    let usage = pool.0.usage.lock().unwrap_err().into_inner();
    assert_eq!(usage.reserved, 0);
    assert_eq!(usage.registered - usage.registry_metadata, SOURCE_BYTES);
    assert_eq!(usage.peak - pool.0.existing, before.1);
    assert!(usage.funding.is_empty());
}

#[test]
fn empty_copy_is_explicit_and_zero_byte_sources_still_keep_identity() {
    let pool = crate::working_memory::memory_fixture::host_ledger(1_000_000, 0).unwrap();
    let context = WorkspaceContext::new(Facts::default().with_pool(&pool));
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
            WorkspaceCopyLimits::new(crate::working_memory::memory_fixture::host_limits(
                1_000_000,
            )),
        )
        .unwrap();
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    assert!(
        copy.requirements()
            .get(pool.topology().host_domain())
            .unwrap()
            .total()
            .unwrap()
            > 0
    );
    assert!(matches!(
        pool.acquire_unquoted(),
        Err(WorkingMemoryError::ReservedWorkActive)
    ));
    settle_unused(copy);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    let source = pool.register_host_storage([(1u32, 0)]).unwrap();
    let context = WorkspaceContext::new(Facts::default().with_pool(&pool));
    let root =
        WorkspaceExistingStorage::try_new_placed(Some(0), &pool.host_placement_handle(), &context)
            .unwrap();
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
            WorkspaceCopyLimits::new(crate::working_memory::memory_fixture::host_limits(
                1_000_000,
            )),
        )
        .unwrap();
    drop(source);
    settle_unused(copy);
    assert!(matches!(
        pool.pin_registered_storage([(1u32, 0)]),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
}
