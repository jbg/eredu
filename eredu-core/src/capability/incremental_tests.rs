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
        additional_headroom: crate::MemoryHeadroomDeclarations::new([("host".into(), 7)]),
        memory_limits: Default::default(),
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
            physical_domains: None,
            geometry,
            activations: bound(64),
            attention: bound(0),
            vocabulary: bound(32),
            state_update: bound(0),
            materialization: bound(0),
            retained: bound(0),
        })
        .unwrap();
    (capabilities, request.clone(), state)
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

#[test]
fn incremental_report_preserves_domain_policy_for_atomic_runtime_admission() {
    let (capabilities, mut request, state) = fixture();
    for limit in [
        crate::MemoryLimit::Finite(0),
        crate::MemoryLimit::Finite(47),
        crate::MemoryLimit::Unlimited,
    ] {
        request.memory_limits = crate::MemoryLimitDeclarations::new([("host".into(), limit)]);
        let result = admitted(
            apply_admission_policy_with_incremental(
                &capabilities,
                request.clone(),
                state.clone(),
                &bound(40),
            )
            .unwrap(),
        );
        assert_eq!(result.incremental_required_bytes, Some(40));
        assert_eq!(result.state, state);
        assert_eq!(result.memory_limits, request.memory_limits);
        assert_eq!(result.additional_headroom, request.additional_headroom);
    }
}

#[test]
fn unknown_incremental_or_incomplete_full_coverage_rejects_all_requests() {
    let (capabilities, request, state) = fixture();
    let unknown = WorkspaceBound::Unknown {
        reason: "unregistered opening identity".into(),
    };
    let result = apply_admission_policy_with_incremental(
        &capabilities,
        request.clone(),
        state.clone(),
        &unknown,
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
                request.clone(),
                incomplete.clone(),
                &bound(0)
            )
            .unwrap(),
            AdmissionResult::Rejected(AdmissionRejection::EstimationUnsupported { .. })
        ));
        // Descriptive whole-request and incremental comparisons share completeness.
        assert!(matches!(
            apply_admission_policy(&capabilities, request.clone(), incomplete).unwrap(),
            AdmissionResult::Rejected(AdmissionRejection::EstimationUnsupported { .. })
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
                request.clone(),
                invalid,
                &bound(0)
            ),
            Err(CapabilityError::InvalidConfiguration { .. })
        ));
    }
    let mut changed_request = request.clone();
    changed_request.input = InputTokenCount::text(6);
    changed_request.max_output_tokens = 1;
    assert!(matches!(
        apply_admission_policy_with_incremental(
            &capabilities,
            changed_request,
            state.clone(),
            &bound(0)
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
        let legacy = apply_admission_policy(&capabilities, request.clone(), state.clone()).unwrap();
        assert!(matches!(legacy, AdmissionResult::Rejected(_)));
        assert_eq!(
            apply_admission_policy_with_incremental(
                &capabilities,
                request.clone(),
                state.clone(),
                &bound(0)
            )
            .unwrap(),
            legacy
        );
    }
}

#[test]
fn incremental_and_full_diagnostic_arithmetic_remain_checked() {
    let (capabilities, mut request, state) = fixture();
    assert_eq!(
        admitted(
            apply_admission_policy_with_incremental(
                &capabilities,
                request.clone(),
                state.clone(),
                &bound(u64::MAX)
            )
            .unwrap()
        )
        .incremental_required_bytes,
        Some(u64::MAX)
    );
    let mut overflowing = state;
    overflowing.requested_state_bytes = u64::MAX;
    assert!(matches!(
        apply_admission_policy_with_incremental(
            &capabilities,
            request.clone(),
            overflowing,
            &bound(0)
        ),
        Err(CapabilityError::ArithmeticOverflow { .. })
    ));
}

#[test]
fn full_and_incremental_reports_preserve_the_same_domain_policy() {
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
        admitted(apply_admission_policy(&capabilities, request.clone(), state.clone()).unwrap());
    assert_eq!(result.incremental_required_bytes, Some(full));
    assert_eq!(result.state, state);
    for budget in [None, Some(full + 6), Some(full + 7)] {
        let request = AdmissionRequest {
            memory_limits: budget
                .map(|bytes| {
                    crate::MemoryLimitDeclarations::new([(
                        "host".into(),
                        crate::MemoryLimit::Finite(bytes),
                    )])
                })
                .unwrap_or_default(),
            ..request.clone()
        };

        assert_eq!(
            apply_admission_policy_with_incremental(
                &capabilities,
                request.clone(),
                state.clone(),
                &bound(full)
            )
            .unwrap(),
            apply_admission_policy(&capabilities, request.clone(), state.clone()).unwrap()
        );
    }
    let mut conservative = state;
    conservative.completeness = EstimationCompleteness::Conservative;
    conservative.persistent_state_completeness = EstimationCompleteness::Conservative;
    assert!(matches!(
        apply_admission_policy_with_incremental(
            &capabilities,
            request.clone(),
            conservative,
            &bound(40)
        )
        .unwrap(),
        AdmissionResult::Admitted(_)
    ));
}
