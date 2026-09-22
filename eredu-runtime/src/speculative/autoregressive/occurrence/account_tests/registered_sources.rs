//! Copy-account provenance is distinct and never certifies a native array.
use super::*;
use eredu_nn::workspace::WorkspaceIsolatedCopyPlan;

#[test]
fn registered_copy_source_preserves_request_separation_and_escaped_account() {
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
    let pool = crate::working_memory::memory_fixture::host_ledger(1 << 24, 0).unwrap();
    let other_pool = crate::working_memory::memory_fixture::host_ledger(1 << 24, 0).unwrap();
    let execution = InferenceExecutionIdentity::default();
    let request = OriginalSpeculativeRequest::prepare(
        &pool,
        &execution,
        &schedule,
        crate::working_memory::memory_fixture::resolved_host_limits(&pool, 1 << 24),
    )
    .unwrap();
    let other_request = OriginalSpeculativeRequest::prepare(
        &pool,
        &execution,
        &schedule,
        crate::working_memory::memory_fixture::resolved_host_limits(&pool, 1 << 24),
    )
    .unwrap();
    let foreign = OriginalSpeculativeRequest::prepare(
        &other_pool,
        &execution,
        &schedule,
        crate::working_memory::memory_fixture::resolved_host_limits(&other_pool, 1 << 24),
    )
    .unwrap();
    let registered = pool.register_host_storage([(7u32, 16)]).unwrap();
    let baseline = pool.payload_used_bytes().unwrap();
    let context = WorkspaceContext::new(ScalarSquare);
    let root = WorkspaceExistingStorage::try_new_placed(
        Some(16),
        crate::working_memory::memory_fixture::host_placement(),
        &context,
    )
    .unwrap();
    let input = WorkspaceTensor::existing_with_storage(
        WorkspaceLayout::new(&[1, 4], WorkspaceDtype::Uint32).unwrap(),
        &root,
        &context,
    )
    .unwrap();
    let binding = RegisteredWorkspaceStorage::bind(&pool, &context, [(7u32, root)]).unwrap();
    let plan =
        WorkspaceIsolatedCopyPlan::prepare(&context, binding.borrowed_storage(), &[input]).unwrap();
    let copy = pool
        .admit_workspace_copy(
            RegisteredWorkspaceCopy::bind(plan, binding).unwrap(),
            WorkspaceCopyLimits::new(crate::working_memory::memory_fixture::host_limits(1 << 24)),
        )
        .unwrap();
    let (custody, scope) = copy.into_parts();
    let retention = custody.retention();
    let source = request.bind_registered_copy_source(&retention).unwrap();
    let provenance = SpeculativeNumericalSource::Registered(&source);
    assert!(provenance.belongs_to_request(&request));
    assert!(provenance.belongs_to_identity(&request.source_identity()));
    assert!(!provenance.belongs_to_request(&other_request));
    assert!(!provenance.belongs_to_identity(&other_request.source_identity()));
    assert!(matches!(
        foreign.bind_registered_copy_source(&retention),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    // This exercises neutral admission/retirement only; no native completion or
    // array ownership is manufactured by the provenance constructor.
    scope.certify().unwrap();
    drop((custody, retention));
    assert!(pool.payload_used_bytes().unwrap() > baseline);
    request.close().unwrap();
    drop(source);
    assert_eq!(pool.payload_used_bytes().unwrap(), baseline);
    drop((registered, request, other_request, context));
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    drop(foreign);
    assert_eq!(other_pool.payload_used_bytes().unwrap(), 0);
}
