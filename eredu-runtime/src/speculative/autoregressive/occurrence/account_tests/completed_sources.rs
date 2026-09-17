//! Neutral original-account fixture; no native tensor or completion is certified.
use super::*;
use crate::speculative::numerical::{SpeculativeNumericalKind, SpeculativeNumericalProgram};
use eredu_core::HostPreparationAuthority;
use std::mem::size_of;

#[test]
fn completed_mixed_sources_preserve_distinct_accounts_capacity_and_escaped_custody() {
    let selected = selected();
    let config = SpeculativeConfig {
        max_tokens: 3,
        max_draft_tokens: 1,
        ..Default::default()
    };
    let schedule = AutoregressiveSchedulePlan::new(
        &selected,
        NonZeroUsize::new(1).unwrap(),
        NonZeroU64::new(2).unwrap(),
        NonZeroU64::new(16).unwrap(),
        &config,
        SpeculativeSchedulerOptions::default(),
    )
    .unwrap();
    let invocation = AutoregressiveInvocation::decode(AutoregressivePass::DraftCommit, 1).unwrap();
    let report = report(
        schedule
            .workspace_geometry(2, invocation, NonZeroU64::new(1).unwrap())
            .unwrap(),
    );
    let bytes = report.span_workspace_plan().records()[0]
        .new_tensor_allocation_bytes()
        .unwrap();
    let source_layout = CompletedWorkspaceSourceLayout::new_accounts(2).unwrap();
    // The actual original role pays the source constructor's host destination;
    // this fixture does not substitute an unmanaged lifetime marker for a debit.
    let host_bytes = source_layout.requested_bytes()
        + 2 * size_of::<(WorkspaceExistingStorage, CompletedWorkspaceSourceAccount)>()
        + HostPreparationAuthority::retention_bytes::<OriginalSpeculativeBudgetCustody>().unwrap();
    let pool = WorkingMemoryPool::new(1 << 24, 0).unwrap();
    let request = OriginalSpeculativeRequest::prepare(
        &pool,
        &InferenceExecutionIdentity::default(),
        &schedule,
        1 << 24,
    )
    .unwrap();
    let mut cursor = schedule.into_cursor();
    let role = request
        .reserve_role(
            cursor.claim(2, invocation).unwrap(),
            SpeculativeInvocationRequirements::new(
                report.span_workspace_plan(),
                Some(bytes),
                Some(0),
                Some(0),
                Some(host_bytes as u64),
            )
            .unwrap(),
        )
        .unwrap();
    let model = role.budget_custody();
    let program =
        SpeculativeNumericalProgram::new(SpeculativeNumericalKind::Normalize, &[1, 1]).unwrap();
    let numerical = request
        .reserve_numerical(
            SpeculativeNumericalRequirements::new(program, Some(bytes), Some(0), Some(0), Some(0))
                .unwrap(),
            &[SpeculativeNumericalSource::Model(&model)],
        )
        .unwrap()
        .begin();
    let model_tag = CompletedWorkspaceSourceAccount::Model(model.clone());
    let numerical_tag = CompletedWorkspaceSourceAccount::Numerical(numerical.clone());
    assert!(!model_tag.same_account(&numerical_tag));
    assert!(model_tag.source().belongs_to_request(&request));
    assert!(numerical_tag.source().belongs_to_request(&request));
    let host = HostPreparationAuthority::retain(model.clone());
    let context = WorkspaceContext::new(ScalarSquare);
    let first = WorkspaceExistingStorage::new(Some(bytes), &context);
    let second = WorkspaceExistingStorage::new(Some(bytes), &context);
    let before = pool.used_bytes().unwrap();
    let duplicate = CompletedWorkspaceSourceLayout::new_accounts(2)
        .unwrap()
        .construct_accounts(
            &context,
            vec![
                (first.clone(), model_tag.clone()),
                (first.clone(), numerical_tag.clone()),
            ],
            &host,
        );
    assert!(matches!(
        duplicate,
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    let oversized = WorkspaceExistingStorage::new(Some(bytes + 1), &context);
    let excess = CompletedWorkspaceSourceLayout::new_accounts(2)
        .unwrap()
        .construct_accounts(
            &context,
            vec![
                (oversized, model_tag.clone()),
                (second.clone(), numerical_tag.clone()),
            ],
            &host,
        );
    assert!(matches!(excess, Err(WorkingMemoryError::IdentityMismatch)));
    assert_eq!(
        pool.used_bytes().unwrap(),
        before,
        "validation issues no replacement account"
    );
    let escaped = source_layout
        .construct_accounts(
            &context,
            vec![(first, model_tag), (second, numerical_tag)],
            &host,
        )
        .unwrap();
    assert_eq!(escaped.borrowed_storage().total_bytes(), 2 * bytes);
    request.close().unwrap();
    drop((host, model, numerical, role, request, context));
    assert!(
        pool.used_bytes().unwrap() > 0,
        "escaped source retains both actual accounts"
    );
    drop(escaped);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn unselected_bindings_retain_all_sources_for_one_union_and_keep_selection_guard() {
    use eredu_nn::workspace::{WorkspaceBorrowedStorage, WorkspaceBorrowedStorageError};
    #[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
    struct Key(u32);
    impl HostSlotStorageKey for Key {
        fn host_slot_identity(&self) -> Option<&crate::HostMetadataKey> { None }
    }
    let selected = selected();
    let config = SpeculativeConfig { max_tokens: 3, max_draft_tokens: 1, ..Default::default() };
    let schedule = AutoregressiveSchedulePlan::new(
        &selected, NonZeroUsize::new(1).unwrap(), NonZeroU64::new(2).unwrap(),
        NonZeroU64::new(16).unwrap(), &config, SpeculativeSchedulerOptions::default(),
    ).unwrap();
    let invocation = AutoregressiveInvocation::decode(AutoregressivePass::DraftCommit, 1).unwrap();
    let report = report(schedule.workspace_geometry(2, invocation, NonZeroU64::new(1).unwrap()).unwrap());
    let bytes = report.span_workspace_plan().records()[0].new_tensor_allocation_bytes().unwrap();
    let model_layout = CompletedWorkspaceSourceLayout::new_accounts(1).unwrap();
    let numerical_layout = CompletedWorkspaceSourceLayout::new_accounts(1).unwrap();
    let registered_layout = RegisteredWorkspaceStorageLayout::<Key>::new(1).unwrap();
    let mixed_layout = CompletedWorkspaceStorageLayout::<Key>::new(1, 1).unwrap();
    let completed_layout = CompletedWorkspaceStorageLayout::<Key>::new(0, 1).unwrap();
    let carrier_layout = OriginalStorageSourcesLayout::new(0).unwrap();
    let host_bytes = model_layout.requested_bytes() + numerical_layout.requested_bytes()
        + registered_layout.requested_bytes() + mixed_layout.requested_bytes().unwrap()
        + completed_layout.requested_bytes().unwrap() + 2 * carrier_layout.requested_bytes()
        + WorkspaceBorrowedStorage::construction_bytes(4).unwrap()
        + 2 * size_of::<(WorkspaceExistingStorage, CompletedWorkspaceSourceAccount)>()
        + HostPreparationAuthority::retention_bytes::<OriginalSpeculativeBudgetCustody>().unwrap();
    let pool = WorkingMemoryPool::new(1 << 24, 0).unwrap();
    let registration = pool.register_storage([(Key(1), 16), (Key(2), 24)]).unwrap();
    let request = OriginalSpeculativeRequest::prepare(&pool, &InferenceExecutionIdentity::default(), &schedule, 1 << 24).unwrap();
    let mut cursor = schedule.into_cursor();
    let role = request.reserve_role(cursor.claim(2, invocation).unwrap(),
        SpeculativeInvocationRequirements::new(report.span_workspace_plan(), Some(bytes), Some(0), Some(0), Some(host_bytes as u64)).unwrap()).unwrap();
    let model = role.budget_custody();
    let numerical = request.reserve_numerical(
        SpeculativeNumericalRequirements::new(SpeculativeNumericalProgram::new(SpeculativeNumericalKind::Normalize, &[1, 1]).unwrap(), Some(bytes), Some(0), Some(0), Some(0)).unwrap(),
        &[SpeculativeNumericalSource::Model(&model)],
    ).unwrap().begin();
    let host = HostPreparationAuthority::retain(model.clone());
    let context = WorkspaceContext::new(ScalarSquare);
    let a = WorkspaceExistingStorage::new(Some(16), &context);
    let b = WorkspaceExistingStorage::new(Some(24), &context);
    let c = WorkspaceExistingStorage::new(Some(bytes), &context);
    let d = WorkspaceExistingStorage::new(Some(bytes), &context);
    let model_source = model_layout.construct_accounts(&context, vec![(c.clone(), model.clone().into())], &host).unwrap();
    let numerical_source = numerical_layout.construct_accounts(&context, vec![(d, CompletedWorkspaceSourceAccount::Numerical(numerical.clone()))], &host).unwrap();
    let mut mixed_carrier = carrier_layout.construct(&pool, &host).unwrap();
    let mut numerical_carrier = carrier_layout.construct(&pool, &host).unwrap();
    let plain = registered_layout.construct_unselected(&pool, &context, [(Key(1), a.clone())]).unwrap();
    let mixed = mixed_layout.construct_unselected(&pool, &context, [(Key(2), b)], model_source, &mut mixed_carrier).unwrap();
    let completed = completed_layout.construct_unselected(&pool, &context, [], numerical_source, &mut numerical_carrier).unwrap();
    let before = pool.used_bytes().unwrap();
    assert!(matches!(RegisteredWorkspaceStorageLayout::<Key>::new(1).unwrap()
        .construct_unselected(&pool, &context, [(Key(99), a)]), Err(WorkingMemoryError::IdentityMismatch)));
    assert_eq!(pool.used_bytes().unwrap(), before, "unselected binding never registers missing sources");
    let union = WorkspaceBorrowedStorage::new_finite(&context,
        plain.borrowed_storage().roots().iter().chain(mixed.borrowed_storage().roots()).chain(completed.borrowed_storage().roots()), 4).unwrap();
    assert_eq!(union.roots().len(), 4);
    assert_eq!(union.total_bytes(), 40 + 2 * bytes);
    context.set_borrowed_storage_checked(union.clone()).unwrap();
    assert!(matches!(context.set_borrowed_storage_checked(union.clone()), Err(WorkspaceBorrowedStorageError::SelectionStarted)));
    let input = WorkspaceTensor::existing_with_storage(WorkspaceLayout::new(&[1], WorkspaceDtype::Float32).unwrap(), &c, &context).unwrap();
    context.begin_state_span([&input]).unwrap();
    let output = input.square(&context).unwrap();
    assert!(context.report(&[output]).unwrap().residual.is_some());
    let late = WorkspaceContext::new(ScalarSquare);
    let late_root = WorkspaceExistingStorage::new(Some(16), &late);
    late.begin_span();
    let late_binding = RegisteredWorkspaceStorageLayout::<Key>::new(1).unwrap()
        .construct_unselected(&pool, &late, [(Key(1), late_root.clone())]).unwrap();
    assert!(matches!(late.set_borrowed_storage_checked(late_binding.borrowed_storage().clone()), Err(WorkspaceBorrowedStorageError::SelectionStarted)));
    assert!(matches!(RegisteredWorkspaceStorageLayout::<Key>::new(1).unwrap()
        .construct(&pool, &late, [(Key(1), late_root)]), Err(WorkingMemoryError::IdentityMismatch)));
    request.close().unwrap();
    drop((registration, host, model, numerical, role, request, context, late, late_binding, union, input, c));
    assert!(pool.used_bytes().unwrap() > 40, "the bindings retain actual model/numerical and registered charges");
    drop((plain, mixed, completed, mixed_carrier, numerical_carrier));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
