use super::*;
use crate::InferenceGeometry;

fn fixture() -> (ModelCapabilities, AdmissionRequest, RuntimeStateEstimate) {
    let capabilities = ModelCapabilities {
        effective_model_type: "neutral media policy fixture".into(),
        native_max_context: Observed::exact(32, "actual fixture maximum"),
        effective_max_context: Observed::exact(32, "actual fixture maximum"),
        state_strategy: CacheStateStrategy::FullKv,
        modalities: InputModalities::TEXT,
        estimation: EstimationCompleteness::Complete,
    };
    let input = InputTokenCount::prepared(2, 3, 5, 12, ObservationKind::Exact);
    let request = AdmissionRequest {
        input,
        max_output_tokens: 3,
        batch_size: 1,
        additional_headroom: crate::MemoryHeadroomDeclarations::new([("host".into(), 7)]),
        memory_limits: Default::default(),
    };
    let geometry = InferenceGeometry {
        batch_size: 1,
        cached_positions: 0,
        input_positions: 5,
        max_output_tokens: 3,
        prefill_chunk_positions: 2,
        output: crate::OutputDemand::LastPosition,
    };
    let b = |n| WorkspaceBound::bounded(n, "independent finite fixture term");
    let state = RuntimeStateEstimate {
        physical_domains: None,
        fixed_state_bytes: 8,
        bytes_per_position_per_batch: 2,
        context_state_bytes: 16,
        selected_state_backing: Some(SelectedStateBacking {
            geometry,
            bound: b(24),
        }),
        multimodal_embedding_bytes: 6,
        media_execution_workspace_bytes: 12,
        requested_state_bytes: 42,
        execution_workspace: Some(ExecutionWorkspaceEstimate {
            physical_domains: None,
            geometry,
            activations: b(11),
            attention: b(13),
            vocabulary: b(17),
            state_update: b(19),
            materialization: b(23),
            retained: b(29),
        }),
        persistent_state_completeness: EstimationCompleteness::Complete,
        assumptions: StateMemoryAssumptions {
            floating_state_dtype_bytes: NonZeroU8::new(4).unwrap(),
            batch_size: 1,
            requested_positions: 8,
            sliding_window_bounds: vec![4, 8],
            allocation_granularity: 2,
        },
        completeness: EstimationCompleteness::Complete,
    };
    (capabilities, request.clone(), state)
}
fn assert_same(
    c: &ModelCapabilities,
    r: AdmissionRequest,
    s: RuntimeStateEstimate,
    incremental: Option<&WorkspaceBound>,
) {
    let expected = apply_admission_policy_impl(c, r.clone(), s.clone(), incremental);
    let fixed = apply_admission_requirements(
        r.clone(),
        AdmissionRequirements::from_reports(c, &s, incremental),
    );
    match (expected, fixed) {
        (Ok(AdmissionResult::Admitted(a)), Ok(BorrowedAdmissionResult::Admitted(b))) => {
            assert_eq!(a.requested_positions, b.requested_positions);
            assert_eq!(a.incremental_required_bytes, b.incremental_required_bytes);
        }
        (Ok(AdmissionResult::Rejected(a)), Ok(BorrowedAdmissionResult::Rejected(b))) => {
            assert_eq!(a, b.into_owned())
        }
        (Err(a), Err(b)) => assert_eq!(format!("{a:?}"), format!("{:?}", CapabilityError::from(b))),
        (a, b) => panic!("reference {a:?}, borrowed {b:?}"),
    }
}

#[test]
fn borrowed_and_owned_policy_reports_preserve_precedence_and_boundaries() {
    let (c, r, s) = fixture();
    // 42 state + six distinct nonzero terms (112) + 7 reserve.
    for budget in [None, Some(160), Some(161), Some(162)] {
        for unknown in 0..5 {
            let mut c = c.clone();
            let mut s = s.clone();
            match unknown {
                1 => c.effective_max_context = Observed::unsupported("missing context"),
                2 => {
                    s.execution_workspace.as_mut().unwrap().attention = WorkspaceBound::Unknown {
                        reason: "missing attention".into(),
                    }
                }
                3 => {
                    s.selected_state_backing.as_mut().unwrap().bound = WorkspaceBound::Unknown {
                        reason: "missing backing".into(),
                    }
                }
                4 => s.persistent_state_completeness = EstimationCompleteness::PersistentStateOnly,
                _ => {}
            }
            let request = AdmissionRequest {
                memory_limits: budget
                    .map(|bytes| {
                        crate::MemoryLimitDeclarations::new([(
                            "host".into(),
                            crate::MemoryLimit::Finite(bytes),
                        )])
                    })
                    .unwrap_or_default(),
                ..r.clone()
            };
            for incremental in [
                None,
                Some(WorkspaceBound::bounded(154, "associated disjoint fixture")),
                Some(WorkspaceBound::Unknown {
                    reason: "missing incremental".into(),
                }),
            ] {
                assert_same(&c, request.clone(), s.clone(), incremental.as_ref());
            }
        }
    }
    for maximum in [4, 5, 7, 8, 32] {
        let mut c = c.clone();
        c.effective_max_context = Observed::exact(maximum, "actual maximum");
        // Context rejection precedes contradictory selected geometry.
        let mut s = s.clone();
        s.selected_state_backing
            .as_mut()
            .unwrap()
            .geometry
            .batch_size = 0;
        assert_same(&c, r.clone(), s, None);
    }
    let admitted =
        apply_admission_requirements(r.clone(), AdmissionRequirements::from_reports(&c, &s, None))
            .unwrap();
    assert!(matches!(
        admitted,
        BorrowedAdmissionResult::Admitted(AdmissionPolicyDecision {
            incremental_required_bytes: Some(154),
            requested_positions: 8,
            ..
        })
    ));
    assert_ne!(r.input.text_tokens, r.input.model_positions);
}

#[test]
fn borrowed_policy_preserves_checked_overflow_and_unknown_term_order() {
    let (c, r, s) = fixture();
    for index in 0..6 {
        for missing in 0..6 {
            let mut s = s.clone();
            let w = s.execution_workspace.as_mut().unwrap();
            let terms = [
                &mut w.activations,
                &mut w.attention,
                &mut w.vocabulary,
                &mut w.state_update,
                &mut w.materialization,
                &mut w.retained,
            ];
            for (i, term) in terms.into_iter().enumerate() {
                if i == index {
                    *term = WorkspaceBound::bounded(u64::MAX, "actual arithmetic limit");
                }
                if i == missing {
                    *term = WorkspaceBound::Unknown {
                        reason: "ordered missing term".into(),
                    };
                }
            }
            assert_same(
                &c,
                r.clone(),
                s,
                Some(&WorkspaceBound::Unknown {
                    reason: "later missing incremental".into(),
                }),
            );
        }
    }
    for field in 0..5 {
        let mut s = s.clone();
        let mut r = r.clone();
        let mut c = c.clone();
        match field {
            0 => s.requested_state_bytes = u64::MAX,
            1 => {
                r.additional_headroom =
                    crate::MemoryHeadroomDeclarations::new([("host".into(), u64::MAX)])
            }
            2 => {
                c.effective_max_context = Observed::exact(u64::MAX, "limit");
                r.input = InputTokenCount::text(u64::MAX);
            }
            3 => {
                s.selected_state_backing
                    .as_mut()
                    .unwrap()
                    .geometry
                    .cached_positions = u64::MAX
            }
            _ => {
                s.execution_workspace
                    .as_mut()
                    .unwrap()
                    .geometry
                    .prefill_chunk_positions = 0
            }
        }
        assert_same(&c, r.clone(), s, None);
    }
}

#[test]
fn borrowed_policy_retains_original_unknown_explanation_by_reference() {
    let (mut c, r, s) = fixture();
    c.effective_max_context = Observed::unsupported("source-owned unavailable context explanation");
    let Observed::Unsupported { reason } = &c.effective_max_context else {
        unreachable!()
    };
    let result =
        apply_admission_requirements(r.clone(), AdmissionRequirements::from_reports(&c, &s, None))
            .unwrap();
    let BorrowedAdmissionResult::Rejected(BorrowedAdmissionRejection::EstimationUnsupported(
        borrowed,
    )) = result
    else {
        panic!("wrong precedence")
    };
    assert_eq!(borrowed.as_ptr(), reason.as_ptr());
    assert_eq!(borrowed.len(), reason.len());
}
