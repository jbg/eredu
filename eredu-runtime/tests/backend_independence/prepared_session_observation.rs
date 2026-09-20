use super::*;
use eredu_runtime::ReplicatedTextSessionError;
use eredu_runtime::{
    ActivationObserver, PreparedSessionObservationError, SharedLayeredObservationPaths,
};

type Session = ReplicatedTextSession<OrdinaryTextFixture, FakeBackend, ReferenceTextMechanisms>;
pub(super) fn session(residency: LayerWeightResidency) -> (Session, ReplicatedSessionCounters) {
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
    let contract = prepare_replicated_text_contract::<
        _,
        FakeBackend,
        DeviceState<FakeBackend, FakeLayerState>,
    >(&architecture, None, selected, &identity, &())
    .unwrap();
    let mechanisms = ReferenceTextMechanisms {
        tasks: Rc::new(RefCell::new(Vec::new())),
        completions: Rc::new(RefCell::new(Vec::new())),
        counters: counters.clone(),
        fail_completion: Rc::new(Cell::new(false)),
        fail_checkpoint: false,
        fail_construction_report: false,
        prepared_partition: None,
        prompt_cache: None,
    };
    (
        construct_replicated_text_session::<_, FakeBackend, _>(
            architecture,
            None,
            contract,
            mechanisms,
            &(),
        )
        .unwrap(),
        counters,
    )
}
struct Observer {
    paths: SharedLayeredObservationPaths,
    boundaries: usize,
    logits: usize,
    failure: bool,
    events: Vec<&'static str>,
}
impl Observer {
    fn new(session: &Session) -> Self {
        Self {
            paths: session.shared_observation_paths().unwrap().clone(),
            boundaries: 0,
            logits: 0,
            failure: false,
            events: Vec::new(),
        }
    }
}
impl ActivationObserver<FakeTensor, Error> for Observer {
    fn requires_prepared_traversal(&self) -> bool {
        true
    }
    fn transactional(&self) -> bool {
        true
    }
    fn prepare_transaction(
        &mut self,
        _: eredu_core::DistributedCommitEpoch,
        _: eredu_runtime::ExpertPass,
    ) -> Result<(), Error> {
        self.events.push("prepare");
        Ok(())
    }
    fn complete_transaction(&mut self, _: eredu_core::DistributedCommitEpoch) -> Result<(), Error> {
        self.events.push("complete");
        Ok(())
    }
    fn finish_transaction(&mut self, _: eredu_core::DistributedCommitEpoch, committed: bool) {
        self.events.push(if committed { "commit" } else { "abort" });
    }
    fn observe(&mut self, path: &str, value: &FakeTensor) -> Result<(), Error> {
        assert!(!value.0.is_empty());
        let (input, output) = self.paths.unit_paths(0, 0).unwrap();
        for expected in [input, output] {
            if path == expected {
                assert_eq!(path.as_ptr(), expected.as_ptr());
                self.boundaries += 1;
            }
        }
        if path == eredu_core::MODEL_LOGITS_OBSERVATION_PATH {
            self.logits += 1;
            if self.failure {
                return Err(Error::backend("prepared observer sentinel"));
            }
        }
        Ok(())
    }
}
pub(super) struct Slots;
impl eredu_nn::ParameterSlotVisitor<FakeTensor> for Slots {
    fn visit_slot(&mut self, _: eredu_nn::ParameterMetadataView<'_>, _: &mut FakeTensor) {}
}

#[test]
fn ordinary_generation_and_reset_preserve_prepared_observation_binding() {
    for residency in [
        LayerWeightResidency::FullyResident,
        LayerWeightResidency::LayerwiseHost(Default::default()),
    ] {
        let (mut session, _) = session(residency);
        let mut observer = Observer::new(&session);
        let source = observer.paths.clone();
        let tokens = FakeTensor(vec![3, 7, 2]);
        let expected = session.forward(&tokens, None, &()).unwrap();
        session.validate_prepared_observation_paths(&source).unwrap();
        session.forward(&FakeTensor(vec![11]), None, &()).unwrap();
        session.validate_prepared_observation_paths(&source).unwrap();
        session.reset(&()).unwrap();
        session.validate_prepared_observation_paths(&source).unwrap();
        let actual = session
            .forward_with_observer(&tokens, None, &(), &mut observer)
            .unwrap();
        assert_eq!(actual, expected);
        assert_eq!(observer.boundaries, 2);
        assert_eq!(observer.logits, 1);
        assert_eq!(observer.events, ["prepare", "complete", "commit"]);
        assert!(source.same_storage(session.shared_observation_paths().unwrap()));
    }
}

#[test]
fn prepared_session_keeps_actual_boundary_pointers_and_existing_transaction_equations() {
    for residency in [
        LayerWeightResidency::FullyResident,
        LayerWeightResidency::LayerwiseHost(Default::default()),
    ] {
        let (mut prepared, counters) = session(residency);
        let (mut baseline, _) = session(residency);
        let mut observer = Observer::new(&prepared);
        for prompt in [vec![3, 7, 2], vec![11], vec![13]] {
            let tokens = FakeTensor(prompt);
            let expected = baseline.forward(&tokens, None, &()).unwrap();
            let mut borrowed = eredu_runtime::BorrowedActivationObserver(&mut observer);
            let mut bridge = eredu_runtime::inspection::ObserverErrorBridge::new(
                &mut borrowed,
                |error: Error| error,
                |error: &Error| Error::backend(error.to_string()),
            );
            let actual = prepared
                .forward_with_observer(&tokens, None, &(), &mut bridge)
                .unwrap();
            assert_eq!(actual, expected);
            assert_eq!(
                prepared.report().unwrap().state_report(),
                baseline.report().unwrap().state_report()
            );
        }
        assert_eq!(observer.boundaries, 6);
        assert_eq!(observer.logits, 3);
        assert_eq!(
            observer.events,
            [
                "prepare", "complete", "commit", "prepare", "complete", "commit", "prepare",
                "complete", "commit"
            ]
        );
        assert_eq!(counters.snapshot().publications, 3);
        assert!(observer
            .paths
            .same_storage(prepared.shared_observation_paths().unwrap()));
    }
}

#[test]
fn stale_binding_rejects_before_state_and_cold_rebind_reuses_original_source() {
    for residency in [
        LayerWeightResidency::FullyResident,
        LayerWeightResidency::LayerwiseHost(Default::default()),
    ] {
        let (mut session, counters) = session(residency);
        let mut observer = Observer::new(&session);
        // Mutable parameter-owner exposure invalidates even if no slot is changed.
        let _ = session.visit_loaded_parameters(&mut Slots);
        let before = counters.snapshot();
        let error = session
            .forward_with_observer(&FakeTensor(vec![4, 9]), None, &(), &mut observer)
            .unwrap_err();
        let ReplicatedTextSessionError::BeforeStateMutation(error) = error else {
            panic!("pre-state rejection")
        };
        assert!(matches!(
            *error,
            ReplicatedTextSessionError::PreparedObservation(
                PreparedSessionObservationError::BindingMismatch
            )
        ));
        assert_eq!(counters.snapshot(), before);
        assert_eq!(session.report().unwrap().state_report(), &[0]);
        assert_eq!(observer.boundaries, 0);
        assert_eq!(observer.logits, 0);
        assert!(!observer.events.contains(&"prepare"));
        let source = session.shared_observation_paths().unwrap().clone();
        session.rebind_observation_paths().unwrap();
        assert!(source.same_storage(session.shared_observation_paths().unwrap()));
        session
            .forward_with_observer(&FakeTensor(vec![4, 9]), None, &(), &mut observer)
            .unwrap();
        assert_eq!(observer.boundaries, 2);
        assert_eq!(observer.logits, 1);
    }
}

#[test]
fn prepared_callback_failure_uses_original_rollback_and_preserves_reusable_path_binding() {
    let (mut session, counters) = session(LayerWeightResidency::FullyResident);
    let mut observer = Observer::new(&session);
    observer.failure = true;
    let error = session
        .forward_with_observer(&FakeTensor(vec![5, 8]), None, &(), &mut observer)
        .unwrap_err();
    assert!(error.to_string().contains("prepared observer sentinel"));
    assert_eq!(session.report().unwrap().state_report(), &[0]);
    assert_eq!(counters.snapshot().publications, 0);
    assert_eq!(observer.events, ["prepare", "abort"]);
    observer.failure = false;
    session
        .forward_with_observer(&FakeTensor(vec![5, 8]), None, &(), &mut observer)
        .unwrap();
    assert_eq!(session.report().unwrap().state_report(), &[1]);
    assert_eq!(counters.snapshot().publications, 1);
    assert_eq!(observer.boundaries, 4);
    assert_eq!(observer.logits, 2);
    assert!(observer
        .paths
        .same_storage(session.shared_observation_paths().unwrap()));
}

#[test]
fn cold_path_validation_requires_exact_owner_without_forward_or_rebinding() {
    for residency in [
        LayerWeightResidency::FullyResident,
        LayerWeightResidency::LayerwiseHost(Default::default()),
    ] {
        let (session, counters) = session(residency);
        let (other, other_counters) = self::session(residency);
        let source = session.shared_observation_paths().unwrap().clone();
        let foreign = other.shared_observation_paths().unwrap();
        assert_eq!(source.unit_paths(0, 0), foreign.unit_paths(0, 0));
        assert!(!source.same_storage(foreign));
        let before = counters.snapshot();
        let other_before = other_counters.snapshot();
        let token = session.prepared_observation_paths().unwrap() as *const _;
        for _ in 0..3 {
            session
                .validate_prepared_observation_paths(&source)
                .unwrap();
            assert!(matches!(
                session.validate_prepared_observation_paths(foreign),
                Err(ReplicatedTextSessionError::PreparedObservation(
                    PreparedSessionObservationError::BindingMismatch
                ))
            ));
        }
        assert_eq!(
            session.prepared_observation_paths().unwrap() as *const _,
            token
        );
        assert_eq!(counters.snapshot(), before);
        assert_eq!(other_counters.snapshot(), other_before);
        assert_eq!(session.report().unwrap().state_report(), &[0]);
    }
}

#[test]
fn cold_path_validation_rejects_stale_token_and_accepts_valid_same_source_rebind() {
    for residency in [
        LayerWeightResidency::FullyResident,
        LayerWeightResidency::LayerwiseHost(Default::default()),
    ] {
        let (mut session, counters) = session(residency);
        let source = session.shared_observation_paths().unwrap().clone();
        let original = source.unit_paths(0, 0).unwrap().0.as_ptr();
        session
            .validate_prepared_observation_paths(&source)
            .unwrap();
        let _ = session.visit_loaded_parameters(&mut Slots);
        let before = counters.snapshot();
        assert!(matches!(
            session.validate_prepared_observation_paths(&source),
            Err(ReplicatedTextSessionError::PreparedObservation(
                PreparedSessionObservationError::BindingMismatch
            ))
        ));
        assert_eq!(counters.snapshot(), before);
        session.rebind_observation_paths().unwrap();
        let rebound = counters.snapshot();
        session
            .validate_prepared_observation_paths(&source)
            .unwrap();
        assert_eq!(counters.snapshot(), rebound);
        assert!(source.same_storage(session.shared_observation_paths().unwrap()));
        assert_eq!(
            session
                .shared_observation_paths()
                .unwrap()
                .unit_paths(0, 0)
                .unwrap()
                .0
                .as_ptr(),
            original
        );
        assert_eq!(session.report().unwrap().state_report(), &[0]);
    }
}

#[test]
fn cold_path_validation_preserves_unavailable_and_session_fence_errors() {
    let (ordinary, _) = session(LayerWeightResidency::FullyResident);
    let source = ordinary.shared_observation_paths().unwrap();
    let (mut partition, _, _, calls, _) = partitioned_cache_control_session(
        0,
        DistributedExecutionPhase::ControlCapturePreparation,
        None,
        true,
    );
    assert!(matches!(
        partition.validate_prepared_observation_paths(source),
        Err(ReplicatedTextSessionError::PreparedObservation(
            PreparedSessionObservationError::Unavailable
        ))
    ));
    assert!(partition.capture_control_state(&()).is_err());
    let before_calls = calls.borrow().len();
    let expected = partition.inspect_runtime(|_, _| Ok(())).unwrap_err();
    let error = partition
        .validate_prepared_observation_paths(source)
        .unwrap_err();
    assert!(matches!(error, ReplicatedTextSessionError::Contract(_)));
    assert_eq!(error.to_string(), expected.to_string());
    assert_eq!(calls.borrow().len(), before_calls);
}

#[path = "prepared_session_observation/parameter_owner_sources.rs"]
mod parameter_owner_sources;
