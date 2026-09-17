use super::*;

#[test]
fn model_capture_quote_rejects_stale_cumulative_usage_without_role_spend() {
    let selected = selected(SpeculativeStrategyClass::EmbeddedSequential);
    let schedule = plan(&selected, 3, true);
    let (invocation, _) = schedule.prefill_invocations(0).unwrap();
    let workspace = EmbeddedInvocationWorkspace::target(invocation).unwrap();
    let report = report(workspace.geometry());
    let capacity = 1 << 26;
    let pool = WorkingMemoryPool::new(capacity, 0).unwrap();
    let source = capture_source(&pool);
    let request = OriginalSpeculativeRequest::prepare_embedded(
        &pool,
        &InferenceExecutionIdentity::default(),
        &schedule,
        capacity,
    )
    .unwrap();
    let mut cursor = schedule.into_cursor();
    let selected_rows = [true];
    let origin = eredu_core::speculative::SpeculativeActivationOrigin {
        request: SpeculativeRequestId::new(95),
        committed_tokens: 0,
        prediction: 0,
        prefix_digest: [0; 32],
        optimistic: false,
    };
    let make = |usage| {
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
                &selected_rows,
            )
            .unwrap(),
            workspace,
            origin,
        )
        .unwrap()
        .with_quoted_usage(usage)
    };
    let initial = request.inspect_embedded_capture_usage(&source).unwrap();
    assert_eq!(initial, CaptureUsage::default());
    let stale = make(initial);
    let (role, pending) = request
        .reserve_embedded_role_with_capture(
            cursor.claim(invocation).unwrap(),
            workspace,
            requirements(report.span_workspace_plan()),
            make(initial),
        )
        .unwrap();
    let mut owner = pending.begin().unwrap();
    let mut backend = Backend {
        custody: role.budget_custody(),
        calls: 0,
    };
    forward(&mut owner, &mut backend, 1).unwrap();
    let frame = owner.take_shared_step().unwrap().unwrap();
    let current = request.inspect_embedded_capture_usage(&source).unwrap();
    assert_eq!(current, owner.usage());
    assert_eq!(current.captures, 1);
    let before = pool.used_bytes().unwrap();
    let refused = request
        .reserve_embedded_role_with_capture(
            cursor.claim(invocation).unwrap(),
            workspace,
            requirements(report.span_workspace_plan()),
            stale,
        )
        .unwrap_err();
    assert!(matches!(
        refused.cause(),
        WorkingMemoryError::IdentityMismatch
    ));
    assert_eq!(pool.used_bytes().unwrap(), before);
    assert_eq!(
        request.inspect_embedded_capture_usage(&source).unwrap(),
        current
    );
    let (_next, pending) = request
        .reserve_embedded_role_with_capture(
            cursor.claim(invocation).unwrap(),
            workspace,
            requirements(report.span_workspace_plan()),
            make(current),
        )
        .unwrap();
    drop(pending);
    assert_eq!(
        request.inspect_embedded_capture_usage(&source).unwrap(),
        current
    );
    let foreign = capture_source(&pool);
    assert!(matches!(
        request.inspect_embedded_capture_usage(&foreign),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    assert_eq!(
        request.inspect_embedded_capture_usage(&source).unwrap(),
        current
    );
    drop(frame);
}
