use super::*;

#[test]
fn early_capture_lineage_is_exact_nonrefunding_and_reused_by_model_roles() {
    let selected = selected(SpeculativeStrategyClass::EmbeddedSequential);
    let schedule = plan(&selected, 3, true);
    let (invocation, _) = schedule.prefill_invocations(0).unwrap();
    let workspace = EmbeddedInvocationWorkspace::target(invocation).unwrap();
    let report = report(workspace.geometry());
    let capacity = 1 << 26;
    let pool = WorkingMemoryPool::new(capacity, 0).unwrap();
    let source = capture_source(&pool);
    let source_bytes = pool.used_bytes().unwrap();
    let request = OriginalSpeculativeRequest::prepare_embedded(
        &pool,
        &InferenceExecutionIdentity::default(),
        &schedule,
        capacity,
    )
    .unwrap();
    let other = OriginalSpeculativeRequest::prepare_embedded(
        &pool,
        &InferenceExecutionIdentity::default(),
        &schedule,
        capacity,
    )
    .unwrap();
    let mut cursor = schedule.into_cursor();
    let lineage = request.prepare_embedded_capture_lineage(&source).unwrap();
    let used = pool.used_bytes().unwrap();
    let alias = request.prepare_embedded_capture_lineage(&source).unwrap();
    assert!(lineage.ledger().same_storage(alias.ledger()));
    assert_eq!(pool.used_bytes().unwrap(), used);
    assert_eq!(
        request
            .inspect_embedded_capture_lineage_usage(&source, &alias)
            .unwrap(),
        CaptureUsage::default()
    );
    let foreign = other.prepare_embedded_capture_lineage(&source).unwrap();
    assert!(matches!(
        request.inspect_embedded_capture_lineage_usage(&source, &foreign),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    let seed = CaptureUsage {
        host_bytes: 4096,
        encoded_bytes: 8192,
        ..Default::default()
    };
    {
        let mut lease = lineage.ledger().borrow().unwrap();
        let mut logical =
            CaptureLedger::with_inherited_usage(source.plan().admission(), lease.usage()).unwrap();
        logical.reserve_quota(seed).unwrap();
        lease.record(logical.total());
    }
    assert_eq!(
        request
            .inspect_embedded_capture_lineage_usage(&source, &alias)
            .unwrap(),
        seed
    );
    let rows = [true];
    let origin = eredu_core::speculative::SpeculativeActivationOrigin {
        request: SpeculativeRequestId::new(96),
        committed_tokens: 0,
        prediction: 0,
        prefix_digest: [0; 32],
        optimistic: false,
    };
    let make = || {
        EmbeddedCaptureHostPlan::prepare(
            &source,
            CaptureRunHostPlan::prepare_invocation(
                source.plan(),
                CapturePhase::Prefill,
                0,
                CaptureInvocationShape {
                    batch: 1,
                    sequence: 3,
                    context: None,
                },
                &rows,
            )
            .unwrap(),
            workspace,
            origin,
        )
        .unwrap()
    };
    let legacy_peak = make().initialization_peak_bytes();
    let host = make()
        .with_lineage(&lineage)
        .unwrap()
        .with_quoted_usage(seed);
    assert_eq!(
        legacy_peak - host.initialization_peak_bytes(),
        crate::working_memory::CaptureRunLedger::control_bytes().unwrap()
    );
    let before = pool.used_bytes().unwrap();
    let rejected = request
        .reserve_embedded_role_with_capture(
            cursor.claim(invocation).unwrap(),
            workspace,
            requirements(report.span_workspace_plan()),
            make().with_lineage(&foreign).unwrap(),
        )
        .unwrap_err();
    assert!(matches!(
        rejected.cause(),
        WorkingMemoryError::IdentityMismatch
    ));
    assert_eq!(pool.used_bytes().unwrap(), before);
    let (role, pending) = request
        .reserve_embedded_role_with_capture(
            cursor.claim(invocation).unwrap(),
            workspace,
            requirements(report.span_workspace_plan()),
            host,
        )
        .unwrap();
    let mut owner = pending.begin().unwrap();
    let mut backend = Backend {
        custody: role.budget_custody(),
        calls: 0,
    };
    forward(&mut owner, &mut backend, 1).unwrap();
    let frame = owner.take_shared_step().unwrap().unwrap();
    let used = request
        .inspect_embedded_capture_lineage_usage(&source, &alias)
        .unwrap();
    assert_eq!(used.captures, 1);
    assert!(used.host_bytes >= seed.host_bytes + 24);
    assert_eq!(owner.usage(), used);
    assert_eq!(frame.cumulative_usage(), used);
    request.close().unwrap();
    assert!(request.prepare_embedded_capture_lineage(&source).is_err());
    assert!(
        request
            .inspect_embedded_capture_lineage_usage(&source, &lineage)
            .is_err()
    );
    drop((request, other, foreign, role, backend, owner));
    assert_eq!(alias.ledger().inspect_usage().unwrap(), used);
    drop((alias, lineage));
    assert!(pool.used_bytes().unwrap() > source_bytes);
    drop(frame);
    assert_eq!(pool.used_bytes().unwrap(), source_bytes);
    drop(source);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
