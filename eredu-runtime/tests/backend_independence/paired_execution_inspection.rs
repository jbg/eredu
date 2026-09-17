//! Actual sessions; inspection grants no execution or mutation authority.
use super::*;
use eredu_runtime::ReplicatedTextSessionError;

type Session = ReplicatedTextSession<OrdinaryTextFixture, FakeBackend, ReferenceTextMechanisms>;
fn session(residency: LayerWeightResidency) -> (Session, ReplicatedSessionCounters) {
    let counters = ReplicatedSessionCounters::default();
    let architecture = OrdinaryTextFixture {
        static_modules: FakeOperator,
        trace: Vec::new(),
        counters: counters.clone(),
        inconsistent_transport: false,
        inconsistent_identity: false,
    };
    let selected = selected_reference_text(&architecture, residency);
    let identity = selected.requirements().architecture_identity().to_owned();
    let prepared = prepare_replicated_text_contract::<
        _,
        FakeBackend,
        DeviceState<FakeBackend, FakeLayerState>,
    >(&architecture, None, selected, &identity, &())
    .unwrap();
    let mechanisms = ReferenceTextMechanisms {
        tasks: Default::default(),
        completions: Default::default(),
        counters: counters.clone(),
        fail_completion: Default::default(),
        fail_checkpoint: false,
        fail_construction_report: false,
        prepared_partition: None,
        prompt_cache: None,
    };
    (
        construct_replicated_text_session::<_, FakeBackend, _>(
            architecture,
            None,
            prepared,
            mechanisms,
            &(),
        )
        .unwrap(),
        counters,
    )
}

#[test]
fn paired_inspection_borrows_actual_owners_across_both_residencies_and_cached_steps() {
    for residency in [
        LayerWeightResidency::FullyResident,
        LayerWeightResidency::LayerwiseHost(Default::default()),
    ] {
        let (mut session, counters) = session(residency);
        let mut checkpoint = session.checkpoint(&()).unwrap();
        checkpoint.as_mut()[0].1 = Some(FakeTensor(vec![13, -7, 29]));
        session.rollback(checkpoint, &()).unwrap();
        let before = counters.snapshot();
        let (mechanisms, state) = session.inspect_runtime(|m, s| Ok((m, s))).unwrap();
        let (_, paired_state, execution) = session
            .inspect_runtime_execution(|m, s, e| {
                assert!(std::ptr::eq(m, mechanisms));
                assert!(std::ptr::eq(s, state));
                assert_eq!(
                    s.as_ref()[0].1.as_ref().unwrap(),
                    &FakeTensor(vec![13, -7, 29])
                );
                e.validate_observation_binding(session.prepared_observation_paths().unwrap())
                    .unwrap();
                // The fixture has no declared immutable slot bound. Empty current
                // visits cannot turn its unknown future topology into a zero bound.
                assert_eq!(e.retained_value_slot_bound(), None);
                Ok((m, s, e))
            })
            .unwrap();
        assert!(std::ptr::eq(paired_state, state));
        let mut visited = Vec::new();
        let complete =
            execution.visit_retained_values(&mut |v| visited.push(v as *const FakeTensor));
        let mut original = Vec::new();
        assert_eq!(
            session
                .visit_retained_values(&mut |v| original.push(v as *const FakeTensor))
                .unwrap(),
            complete
        );
        assert_eq!(visited, original);
        assert_eq!(counters.snapshot(), before);
        // Previous shared borrows expire before a real cached step. No runtime
        // replacement or clone occurs in the inspector.
        let path_source = session.shared_observation_paths().unwrap().clone();
        session.decode(&FakeTensor(vec![3]), &()).unwrap();
        session.decode(&FakeTensor(vec![4]), &()).unwrap();
        let before = counters.snapshot();
        session
            .inspect_runtime_execution(|_, s, e| {
                assert_eq!(s.as_ref()[0].0, 2);
                assert_eq!(
                    s.as_ref()[0].1.as_ref().unwrap(),
                    &FakeTensor(vec![15, -7, 29])
                );
                // Ordinary custom traversal invalidates its earlier prepared
                // token. Paired inspection must preserve that stale rejection.
                assert!(matches!(
                    e.validate_observation_binding(session.prepared_observation_paths().unwrap()),
                    Err(ReplicatedTextSessionError::PreparedObservation(
                        eredu_runtime::PreparedSessionObservationError::BindingMismatch
                    ))
                ));
                Ok(())
            })
            .unwrap();
        assert_eq!(counters.snapshot(), before);
        // Only the existing explicit cold preparation boundary rebinds it;
        // the read-only inspector neither repairs nor replaces the token.
        session.rebind_observation_paths().unwrap();
        assert!(path_source.same_storage(session.shared_observation_paths().unwrap()));
        let rebound = counters.snapshot();
        session
            .inspect_runtime_execution(|_, s, e| {
                assert_eq!(s.as_ref()[0].0, 2);
                assert_eq!(
                    s.as_ref()[0].1.as_ref().unwrap(),
                    &FakeTensor(vec![15, -7, 29])
                );
                e.validate_observation_binding(session.prepared_observation_paths().unwrap())
                    .unwrap();
                Ok(())
            })
            .unwrap();
        assert_eq!(counters.snapshot(), rebound);
    }
}

#[test]
fn paired_inspection_preserves_callback_error_and_rejects_foreign_runtime_token() {
    let (a, counts) = session(LayerWeightResidency::FullyResident);
    let (b, _) = session(LayerWeightResidency::FullyResident);
    assert!(!a
        .shared_observation_paths()
        .unwrap()
        .same_storage(b.shared_observation_paths().unwrap()));
    let before = counts.snapshot();
    a.inspect_runtime_execution(|_, _, execution| {
        assert!(execution
            .validate_observation_binding(b.prepared_observation_paths().unwrap())
            .is_err());
        execution
            .validate_observation_binding(a.prepared_observation_paths().unwrap())
            .unwrap();
        Ok(())
    })
    .unwrap();
    static ORIGINAL: &str = "unchanged inspection callback cause";
    let error = a
        .inspect_runtime_execution::<()>(|_, _, _| Err(ORIGINAL))
        .unwrap_err();
    let eredu_runtime::ReplicatedTextSessionError::Mechanism(cause) = error else {
        panic!("typed mechanism error")
    };
    assert!(std::ptr::eq(cause.as_ptr(), ORIGINAL.as_ptr()));
    assert_eq!(
        a.inspect_runtime::<()>(|_, _| Err(ORIGINAL))
            .unwrap_err()
            .to_string(),
        eredu_runtime::ReplicatedTextSessionError::<Error, &'static str, &'static str>::Mechanism(
            ORIGINAL
        )
        .to_string()
    );
    assert_eq!(counts.snapshot(), before);
}

#[test]
fn paired_partition_inspection_keeps_local_pair_and_rejects_unresolved_or_fenced_state() {
    let commits = Rc::new(Cell::new(0));
    let phase = DistributedCommitPhase::DecisionCompletion;
    let (mut session, counters, _) = partitioned_agreement_session(
        0,
        LocalPartitionFailure::None,
        TestPhaseAgreement::Final {
            result: FinalDecisionResult::Indeterminate(phase),
            commits: commits.clone(),
        },
    );
    assert!(
        session.prepared_observation_paths().is_none(),
        "local inspection is not ordinary prepared traversal support"
    );
    let (m, s) = session.inspect_runtime(|m, s| Ok((m, s))).unwrap();
    session
        .inspect_runtime_execution(|paired_m, paired_s, _| {
            assert!(std::ptr::eq(m, paired_m));
            assert!(std::ptr::eq(s, paired_s));
            assert_eq!(paired_s.as_ref().len(), 1);
            Ok(())
        })
        .unwrap();
    assert!(session.decode(&FakeTensor(vec![3]), &()).is_err());
    let before = counters.snapshot();
    let error = session
        .inspect_runtime_execution::<()>(|_, _, _| panic!("unresolved inspector callback"))
        .unwrap_err();
    assert!(
        matches!(error, eredu_runtime::ReplicatedTextSessionError::CommitIndeterminate { epoch: DistributedCommitEpoch::FIRST, phase: actual } if actual == phase)
    );
    assert!(matches!(
        session.inspect_runtime::<()>(|_, _| panic!("old unresolved callback")),
        Err(eredu_runtime::ReplicatedTextSessionError::CommitIndeterminate { .. })
    ));
    assert_eq!(counters.snapshot(), before);
    assert_eq!(commits.get(), 1);
    let fixed = session
        .inspect_runtime_execution_fixed::<(), ()>(|_, _, _| panic!("fixed unresolved callback"))
        .unwrap_err();
    assert!(
        matches!(fixed, eredu_runtime::replicated_session::RuntimeInspectionBoundary::Indeterminate { epoch: DistributedCommitEpoch::FIRST, phase: actual } if actual == phase)
    );
    assert_eq!(counters.snapshot(), before);

    let (mut fenced, _, _, _, counters) = partitioned_cache_control_session(
        0,
        DistributedExecutionPhase::ControlCapturePreparation,
        None,
        true,
    );
    assert!(fenced.capture_control_state(&()).is_err());
    let before = counters.snapshot();
    let old = fenced
        .inspect_runtime::<()>(|_, _| panic!("old fenced callback"))
        .unwrap_err();
    let new = fenced
        .inspect_runtime_execution::<()>(|_, _, _| panic!("paired fenced callback"))
        .unwrap_err();
    assert_eq!(new.to_string(), old.to_string());
    let fixed = fenced
        .inspect_runtime_execution_fixed::<(), ()>(|_, _, _| panic!("fixed fenced callback"))
        .unwrap_err();
    assert_eq!(
        fixed,
        eredu_runtime::replicated_session::RuntimeInspectionBoundary::Fenced {
            phase: DistributedExecutionPhase::ControlCapturePreparation
        }
    );
    assert_eq!(counters.snapshot(), before);
}

#[test]
fn fixed_paired_inspection_keeps_actual_borrows_and_nonstatic_callback_failure() {
    for residency in [
        LayerWeightResidency::FullyResident,
        LayerWeightResidency::LayerwiseHost(Default::default()),
    ] {
        let (session, counts) = session(residency);
        let before = counts.snapshot();
        let original = session
            .inspect_runtime_execution(|m, s, e| Ok((m, s, e)))
            .unwrap();
        let exact = session
            .inspect_runtime_execution_fixed(|m, s, e| Ok::<_, ()>((m, s, e)))
            .unwrap()
            .unwrap();
        assert!(std::ptr::eq(original.0, exact.0));
        assert!(std::ptr::eq(original.1, exact.1));
        assert!(std::ptr::eq(original.2, exact.2));
        // No Display/Error/Clone/static requirement or wrapper is imposed on E.
        struct BorrowedFailure<'a>(&'a [u32]);
        let payload = [7u32, 19, 23];
        let error = session
            .inspect_runtime_execution_fixed(|_, _, _| Err::<(), _>(BorrowedFailure(&payload)))
            .unwrap()
            .err()
            .unwrap();
        assert!(std::ptr::eq(error.0, payload.as_slice()));
        assert_eq!(counts.snapshot(), before);
    }
}
