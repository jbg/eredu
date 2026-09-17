use super::*;

fn fixture() -> (ModelCapabilities, AdmissionRequest, RuntimeStateEstimate) {
    let capabilities = ModelCapabilities {
        effective_model_type: "portable incremental admission fixture".into(),
        native_max_context: Observed::exact(64, "fixture"),
        effective_max_context: Observed::exact(64, "fixture"),
        state_strategy: CacheStateStrategy::FullKv,
        modalities: InputModalities::TEXT,
        estimation: EstimationCompleteness::Complete,
    };
    let request = AdmissionRequest {
        input: InputTokenCount::text(5),
        max_output_tokens: 2,
        batch_size: 1,
        safety_reserve_bytes: 7,
        application_memory_budget_bytes: None,
        require_complete_estimate: false,
    };
    let geometry = crate::InferenceGeometry {
        batch_size: 1,
        cached_positions: 2,
        input_positions: 3,
        max_output_tokens: 2,
        prefill_chunk_positions: 2,
        output: crate::OutputDemand::LastPosition,
    };
    let layout = StateMemoryLayout::new(
        LayerSchedule::new(
            1,
            vec![LayerCachePolicy::key_only(AttentionPolicy::Full, 1, 2).unwrap()],
        )
        .unwrap(),
        vec![0],
        2,
        1,
        EstimationCompleteness::Complete,
    )
    .unwrap();
    let state = estimate_runtime_state(&layout, request.input, 2, 1, NonZeroU8::new(4).unwrap())
        .unwrap()
        .with_selected_state_backing(geometry, bound(128))
        .unwrap()
        .with_execution_workspace(ExecutionWorkspaceEstimate {
            geometry,
            activations: bound(64),
            attention: bound(0),
            vocabulary: bound(32),
            state_update: bound(0),
            materialization: bound(0),
            retained: bound(0),
        })
        .unwrap();
    (capabilities, request, state)
}

fn bound(bytes: u64) -> WorkspaceBound {
    WorkspaceBound::bounded(
        bytes,
        "provider fixture bound; this test supplies no reservation proof",
    )
}

fn admitted(result: AdmissionResult) -> Admission {
    match result {
        AdmissionResult::Admitted(admission) => admission,
        result => panic!("expected admission, got {result:?}"),
    }
}

fn available(bytes: u64) -> AvailableMemory {
    AvailableMemory {
        physical_memory_bytes: Observed::exact(1000, "fixture"),
        available_memory_bytes: Observed::exact(bytes, "fixture"),
        physical_semantics: PhysicalMemorySemantics::Unified,
    }
}

#[test]
fn incremental_budget_boundary_includes_safety_and_preserves_full_diagnostics() {
    let (capabilities, mut request, state) = fixture();
    assert!(state.context_state_bytes > 0);
    assert_eq!(state.requested_state_bytes, 128);
    request.application_memory_budget_bytes = Some(47);
    let result = admitted(
        crate::apply_admission_policy_with_incremental(
            &capabilities,
            request,
            state.clone(),
            &bound(40),
            None,
        )
        .unwrap(),
    );
    assert_eq!(result.incremental_required_bytes, 47);
    assert_eq!(result.requested_positions, 7);
    assert_eq!(result.state, state);
    assert_eq!(result.available_memory_bytes, None);

    request.application_memory_budget_bytes = Some(46);
    assert_eq!(
        apply_admission_policy_with_incremental(
            &capabilities,
            request,
            state.clone(),
            &bound(40),
            None,
        )
        .unwrap(),
        AdmissionResult::Rejected(AdmissionRejection::MemoryBudgetExceeded {
            required_bytes: 47,
            budget_bytes: 46,
        })
    );

    request.application_memory_budget_bytes = Some(7);
    let zero = admitted(
        apply_admission_policy_with_incremental(
            &capabilities,
            request,
            state.clone(),
            &bound(0),
            None,
        )
        .unwrap(),
    );
    assert_eq!(zero.incremental_required_bytes, 7);
    assert_eq!(zero.state, state);
    request.safety_reserve_bytes = 0;
    request.application_memory_budget_bytes = Some(0);
    assert_eq!(admitted(apply_admission_policy_with_incremental(
        &capabilities, request, state, &bound(0), None,
    ).unwrap()).incremental_required_bytes, 0);
}

#[test]
fn availability_uses_incremental_bytes_and_preserves_unavailable_signal_policy() {
    let (capabilities, request, state) = fixture();
    let exact = available(47);
    let result = admitted(
        apply_admission_policy_with_incremental(
            &capabilities,
            request,
            state.clone(),
            &bound(40),
            Some(&exact),
        )
        .unwrap(),
    );
    assert_eq!(result.available_memory_bytes, Some(47));
    let short = available(46);
    assert_eq!(
        apply_admission_policy_with_incremental(
            &capabilities,
            request,
            state.clone(),
            &bound(40),
            Some(&short),
        )
        .unwrap(),
        AdmissionResult::Rejected(AdmissionRejection::InsufficientAvailableMemory {
            required_bytes: 47,
            available_bytes: 46,
        })
    );
    for observation in [
        Observed::unavailable("provider unavailable"),
        Observed::unsupported("provider unsupported"),
    ] {
        let report = AvailableMemory {
            available_memory_bytes: observation,
            ..available(47)
        };
        assert!(matches!(
            apply_admission_policy_with_incremental(
                &capabilities,
                request,
                state.clone(),
                &bound(40),
                Some(&report),
            )
            .unwrap(),
            AdmissionResult::Rejected(AdmissionRejection::AvailableMemoryUnavailable { .. })
        ));
    }
}

#[test]
fn unknown_incremental_or_incomplete_full_coverage_rejects_even_permissive_requests() {
    let (capabilities, request, state) = fixture();
    let unknown = WorkspaceBound::Unknown {
        reason: "unregistered opening identity".into(),
    };
    let result = apply_admission_policy_with_incremental(
        &capabilities,
        request,
        state.clone(),
        &unknown,
        None,
    )
    .unwrap();
    assert!(
        matches!(result, AdmissionResult::Rejected(AdmissionRejection::EstimationUnsupported { reason })
        if reason.contains("unregistered opening identity"))
    );

    let mut missing_workspace = state.clone();
    missing_workspace.execution_workspace = None;
    let mut unknown_workspace = state.clone();
    unknown_workspace
        .execution_workspace
        .as_mut()
        .unwrap()
        .activations = unknown.clone();
    let mut unknown_backing = state.clone();
    unknown_backing
        .selected_state_backing
        .as_mut()
        .unwrap()
        .bound = unknown;
    let mut missing_persistent = state.clone();
    missing_persistent.persistent_state_completeness = EstimationCompleteness::PersistentStateOnly;
    let mut missing_total = state;
    missing_total.completeness = EstimationCompleteness::PersistentStateOnly;
    for incomplete in [
        missing_workspace,
        unknown_workspace,
        unknown_backing,
        missing_persistent,
        missing_total,
    ] {
        assert!(matches!(
            apply_admission_policy_with_incremental(
                &capabilities,
                request,
                incomplete.clone(),
                &bound(0),
                None,
            )
            .unwrap(),
            AdmissionResult::Rejected(AdmissionRejection::EstimationUnsupported { .. })
        ));
        // The original permissive reporting API retains its prior behavior.
        assert!(matches!(
            apply_admission_policy(&capabilities, request, incomplete, None).unwrap(),
            AdmissionResult::Admitted(_)
        ));
    }
}

#[test]
fn incremental_reporting_keeps_context_and_exact_geometry_validation() {
    let (capabilities, request, state) = fixture();
    let mut batch = state.clone();
    batch.assumptions.batch_size = 2;
    let mut positions = state.clone();
    positions.assumptions.requested_positions = 8;
    let mut workspace = state.clone();
    workspace
        .execution_workspace
        .as_mut()
        .unwrap()
        .geometry
        .prefill_chunk_positions = 1;
    for invalid in [batch, positions, workspace] {
        assert!(matches!(
            apply_admission_policy_with_incremental(
                &capabilities,
                request,
                invalid,
                &bound(0),
                None,
            ),
            Err(CapabilityError::InvalidConfiguration { .. })
        ));
    }
    let mut changed_request = request;
    changed_request.input = InputTokenCount::text(6);
    changed_request.max_output_tokens = 1;
    assert!(matches!(
        apply_admission_policy_with_incremental(
            &capabilities,
            changed_request,
            state.clone(),
            &bound(0),
            None,
        ),
        Err(CapabilityError::InvalidConfiguration {
            field: "admission_workspace",
            ..
        })
    ));

    for limit in [
        Observed::exact(4, "prompt boundary"),
        Observed::exact(6, "output boundary"),
        Observed::unavailable("context unavailable"),
    ] {
        let capabilities = ModelCapabilities {
            effective_max_context: limit,
            ..capabilities.clone()
        };
        let legacy = apply_admission_policy(&capabilities, request, state.clone(), None).unwrap();
        assert!(matches!(legacy, AdmissionResult::Rejected(_)));
        assert_eq!(
            apply_admission_policy_with_incremental(
                &capabilities,
                request,
                state.clone(),
                &bound(0),
                None,
            )
            .unwrap(),
            legacy
        );
    }
}

#[test]
fn incremental_and_full_diagnostic_arithmetic_remain_checked() {
    let (capabilities, mut request, state) = fixture();
    assert!(matches!(
        apply_admission_policy_with_incremental(
            &capabilities,
            request,
            state.clone(),
            &bound(u64::MAX),
            None,
        ),
        Err(CapabilityError::ArithmeticOverflow { .. })
    ));
    request.safety_reserve_bytes = 0;
    assert_eq!(
        admitted(
            apply_admission_policy_with_incremental(
                &capabilities,
                request,
                state.clone(),
                &bound(u64::MAX),
                None,
            )
            .unwrap()
        )
        .incremental_required_bytes,
        u64::MAX
    );
    let mut overflowing = state;
    overflowing.requested_state_bytes = u64::MAX;
    assert!(matches!(
        apply_admission_policy_with_incremental(
            &capabilities,
            request,
            overflowing,
            &bound(0),
            None,
        ),
        Err(CapabilityError::ArithmeticOverflow { .. })
    ));
}

#[test]
fn legacy_full_charge_and_policy_results_are_unchanged() {
    let (capabilities, request, state) = fixture();
    let full = state.requested_state_bytes
        + state
            .execution_workspace
            .as_ref()
            .unwrap()
            .peak_bytes()
            .unwrap()
            .unwrap();
    let result =
        admitted(apply_admission_policy(&capabilities, request, state.clone(), None).unwrap());
    assert_eq!(
        result.incremental_required_bytes,
        full + request.safety_reserve_bytes
    );
    assert_eq!(result.state, state);
    for budget in [None, Some(full + 6), Some(full + 7)] {
        let request = AdmissionRequest {
            application_memory_budget_bytes: budget,
            ..request
        };
        for report in [None, Some(available(full + 6)), Some(available(full + 7))] {
            assert_eq!(
                apply_admission_policy_with_incremental(
                    &capabilities,
                    request,
                    state.clone(),
                    &bound(full),
                    report.as_ref(),
                )
                .unwrap(),
                apply_admission_policy(&capabilities, request, state.clone(), report.as_ref())
                    .unwrap()
            );
        }
    }
    let mut conservative = state;
    conservative.completeness = EstimationCompleteness::Conservative;
    conservative.persistent_state_completeness = EstimationCompleteness::Conservative;
    assert!(matches!(
        apply_admission_policy_with_incremental(
            &capabilities,
            request,
            conservative,
            &bound(40),
            None,
        )
        .unwrap(),
        AdmissionResult::Admitted(_)
    ));
}
