use super::*;
use eredu_core::speculative::SpeculativeActivationPhase as Phase;
use eredu_runtime::ActivationObserver;

#[derive(Default)]
struct Trace {
    origin: Option<SpeculativeActivationOrigin>,
    origins: Vec<Option<SpeculativeActivationOrigin>>,
    capture_error: Option<eredu_core::capture::CaptureError>,
    phases: Vec<(Phase, usize)>,
    observations: Vec<(Phase, String, Tensor)>,
    finishes: Vec<bool>,
    active: Option<Phase>,
    remaining: usize,
    fail: Option<Phase>,
    change_proposal: bool,
}
struct Observer(Arc<Mutex<Trace>>);
impl ActivationObserver<Tensor, TestError> for Observer {
    fn observe(&mut self, path: &str, value: &Tensor) -> Result<(), TestError> {
        let mut trace = self.0.lock().unwrap();
        let phase = trace.active.expect("observation must have a phase");
        trace.observations.push((phase, path.into(), value.clone()));
        Ok(())
    }
    fn intervene(&mut self, path: &str, value: &Tensor) -> Result<Option<Tensor>, TestError> {
        Ok(
            (self.0.lock().unwrap().change_proposal && path == "fixture.prediction.logits")
                .then(|| Tensor(value.0.iter().map(|v| v + 70).collect())),
        )
    }
}
impl SpeculativeActivationObserver<Tensor, TestError> for Observer {
    fn set_activation_origin(&mut self, origin: Option<SpeculativeActivationOrigin>) {
        self.0.lock().unwrap().origin = origin;
    }
    fn take_activation_error(
        &mut self,
    ) -> Option<eredu_core::speculative::SpeculativeControlError> {
        self.0
            .lock()
            .unwrap()
            .capture_error
            .take()
            .map(eredu_core::speculative::SpeculativeControlError::Capture)
    }
    fn begin_activation_invocation(
        &mut self,
        phase: Phase,
        sequence: usize,
    ) -> Result<(), TestError> {
        let mut trace = self.0.lock().unwrap();
        let origin = trace.origin;
        trace.origins.push(origin);
        assert!(trace.active.replace(phase).is_none());
        trace.phases.push((phase, sequence));
        trace.remaining = trace
            .remaining
            .checked_sub(sequence)
            .ok_or_else(|| TestError("phase budget exhausted".into()))?;
        Ok(())
    }
    fn complete_activation_invocation(&mut self) -> Result<(), TestError> {
        let trace = self.0.lock().unwrap();
        if trace.active == trace.fail {
            return Err(TestError("injected phase delivery failure".into()));
        }
        Ok(())
    }
    fn finish_activation_invocation(&mut self, success: bool) {
        let mut trace = self.0.lock().unwrap();
        assert!(trace.active.take().is_some());
        trace.finishes.push(success);
    }
}
fn strategy() -> Strategy {
    Strategy {
        fused_rows: None,
        corrupt_capture: false,
        failure: Failure::None,
    }
}
fn trace(remaining: usize) -> Arc<Mutex<Trace>> {
    Arc::new(Mutex::new(Trace {
        remaining,
        ..Default::default()
    }))
}
fn observers(trace: &Arc<Mutex<Trace>>) -> EmbeddedPredictionObservers<Tensor, i32, TestError> {
    EmbeddedPredictionObservers::default().with_internal(Observer(Arc::clone(trace)))
}

#[test]
fn shared_driver_attributes_actual_phases_and_preserves_causal_values() {
    let trace = trace(20);
    trace.lock().unwrap().change_proposal = true;
    let mut strategy = strategy();
    let mut executor = EmbeddedPredictionExecutor::<_, Mechanisms>::with_observers(
        &mut strategy,
        observers(&trace),
    );
    let mut cache = cache();
    let (logits, state, _) = executor
        .prefill(vec![1, 2, 3], &mut cache, ())
        .unwrap()
        .into_parts();
    assert_eq!(logits, 103);
    let mut draft = executor.begin_proposal(&state, 3, 2, ()).unwrap();
    assert_eq!(executor.proposal_logits(&mut draft, 3, ()).unwrap(), 73);
    assert_eq!(executor.proposal_logits(&mut draft, 4, ()).unwrap(), 75);
    let checkpoint = executor.checkpoint(&cache).unwrap();
    let submission = executor
        .submit_verification(&[3, 4, 5], &mut cache, ())
        .unwrap();
    submission.completion.wait().unwrap();
    assert_eq!(
        executor
            .verification_logits(&submission.output, 1, ())
            .unwrap(),
        104
    );
    let (_, replayed) = executor
        .commit_verification(submission.output, draft, &mut cache, &checkpoint, 2, ())
        .unwrap()
        .into_parts();
    assert_eq!(replayed, 2);
    assert_eq!(cache.target, [9, 1, 2, 3, 3, 4]);
    assert_eq!(cache.prediction, [1, 2, 3, 3, 4, 4]);
    let trace = trace.lock().unwrap();
    assert_eq!(
        trace.phases,
        [
            (Phase::TargetPrefill, 3),
            (Phase::PredictionPrefill, 2),
            (Phase::Proposal { depth: 0 }, 1),
            (Phase::Proposal { depth: 1 }, 1),
            (Phase::Verification, 3),
            (Phase::PredictionReplay, 1),
            (Phase::TargetReplay, 2),
        ]
    );
    assert_eq!(trace.finishes, [true; 7]);
    assert_eq!(trace.remaining, 7);
    assert_eq!(
        trace.observations[2].2,
        Tensor(vec![3]),
        "observation precedes intervention"
    );
    assert_eq!(
        trace.observations[4].2,
        Tensor(vec![103, 104, 105]),
        "verification retains its physical rows, including rejected suffix"
    );
}

#[test]
fn prefill_delivery_failures_restore_state_without_restoring_authority() {
    for phase in [Phase::TargetPrefill, Phase::PredictionPrefill] {
        let trace = trace(10);
        trace.lock().unwrap().fail = Some(phase);
        let mut strategy = strategy();
        let mut executor = EmbeddedPredictionExecutor::<_, Mechanisms>::with_observers(
            &mut strategy,
            observers(&trace),
        );
        let mut cache = cache();
        let checkpoint = cache.clone();
        let error = executor
            .prefill(vec![1, 2, 3], &mut cache, ())
            .err()
            .unwrap();
        assert_eq!(error.0, "injected phase delivery failure");
        assert_eq!(cache, checkpoint);
        let consumed = if phase == Phase::TargetPrefill { 3 } else { 5 };
        assert_eq!(trace.lock().unwrap().remaining, 10 - consumed);
        trace.lock().unwrap().fail = None;
        let (logits, _, _) = executor
            .prefill(vec![1, 2, 3], &mut cache, ())
            .unwrap()
            .into_parts();
        assert_eq!(logits, 103);
        assert_eq!(cache.target, [9, 1, 2, 3]);
        assert_eq!(trace.lock().unwrap().remaining, 5 - consumed);
    }
}

#[test]
fn replay_delivery_failures_restore_preverification_state_and_keep_tentative_evidence() {
    for phase in [Phase::PredictionReplay, Phase::TargetReplay] {
        let trace = trace(20);
        let mut strategy = strategy();
        let mut executor = EmbeddedPredictionExecutor::<_, Mechanisms>::with_observers(
            &mut strategy,
            observers(&trace),
        );
        let mut cache = cache();
        let (_, state, _) = executor
            .prefill(vec![1, 2], &mut cache, ())
            .unwrap()
            .into_parts();
        let draft = executor.begin_proposal(&state, 2, 2, ()).unwrap();
        let checkpoint = executor.checkpoint(&cache).unwrap();
        let submission = executor
            .submit_verification(&[2, 3, 4], &mut cache, ())
            .unwrap();
        submission.completion.wait().unwrap();
        trace.lock().unwrap().fail = Some(phase);
        let error = executor
            .commit_verification(submission.output, draft, &mut cache, &checkpoint, 2, ())
            .err()
            .unwrap();
        assert_eq!(error.0, "injected phase delivery failure");
        assert_eq!(cache, checkpoint.cache);
        let trace = trace.lock().unwrap();
        assert_eq!(trace.phases.last().unwrap().0, phase);
        assert_eq!(trace.finishes.last(), Some(&false));
        assert!(trace
            .observations
            .iter()
            .any(|(phase, _, value)| *phase == Phase::Verification && value.0 == [102, 103, 104]));
        assert_eq!(
            trace.remaining,
            if phase == Phase::PredictionReplay {
                13
            } else {
                11
            }
        );
    }
}

#[test]
fn phase_admission_precedes_execution_and_unsupported_selection_precedes_admission() {
    for fused in [false, true] {
        let trace = trace(0);
        let mut strategy = strategy();
        strategy.fused_rows = fused.then_some(2);
        let mut executor = EmbeddedPredictionExecutor::<_, Mechanisms>::with_observers(
            &mut strategy,
            observers(&trace),
        );
        let mut cache = cache();
        let checkpoint = cache.clone();
        let error = executor.prefill(vec![1, 2], &mut cache, ()).err().unwrap();
        assert_eq!(cache, checkpoint);
        let trace = trace.lock().unwrap();
        assert!(trace.observations.is_empty());
        assert_eq!(trace.phases.len(), usize::from(!fused));
        assert!(error.0.contains(if fused {
            "no complete internal activation path"
        } else {
            "budget exhausted"
        }));
    }
}

#[test]
fn typed_executor_preserves_scheduler_origin_and_portable_capture_errors() {
    let trace = trace(10);
    let mut strategy = strategy();
    let mut erased = EmbeddedPredictionExecutor::<_, Mechanisms>::with_observers(
        &mut strategy,
        observers(&trace),
    );
    assert!(erased.requires_activation_origin());
    let origin = SpeculativeActivationOrigin {
        request: eredu_core::SpeculativeRequestId::new(2),
        committed_tokens: 0,
        prediction: 0,
        prefix_digest: [17; 32],
        optimistic: false,
    };
    erased.set_activation_origin(Some(origin));
    let mut cache = cache();
    erased.prefill(vec![1, 2, 3], &mut cache, ()).unwrap();
    erased.set_activation_origin(None);
    assert_eq!(trace.lock().unwrap().origins, [Some(origin); 2]);
    assert!(trace.lock().unwrap().origin.is_none());
    let error = eredu_core::capture::CaptureError::Limit {
        budget: eredu_core::capture::CaptureBudget::Host,
        cumulative: true,
    };
    trace.lock().unwrap().capture_error = Some(error.clone());
    assert!(
        matches!(erased.take_activation_error(), Some(eredu_core::speculative::SpeculativeControlError::Capture(actual)) if actual == error)
    );
    assert!(erased.take_activation_error().is_none());
    assert!(erased.take_activation_capture().is_none());
}

fn empty_authority() -> AdmittedSpeculativeActivations {
    use eredu_core::{capture::*, intervention::*, speculative::*};
    let discovery = SpeculativeActivationDiscovery {
        schema_version: SPECULATIVE_ACTIVATION_SCHEMA_VERSION,
        execution_identity: "selected".into(),
        captures: CaptureDiscovery {
            artifact_identity: "source".into(),
            catalog: eredu_core::ObservationCatalog {
                schema_version: eredu_core::DISCOVERY_SCHEMA_VERSION,
                points: vec![],
                completeness: eredu_core::DescriptionCompleteness::Complete,
            },
            support: eredu_core::ObservationSupportReport {
                schema_version: eredu_core::DISCOVERY_SCHEMA_VERSION,
                points: vec![],
                capture: Default::default(),
            },
        },
        interventions: InterventionDiscovery {
            schema_version: INTERVENTION_SCHEMA_VERSION,
            artifact_identity: "source".into(),
            session_identity: Some("session".into()),
            points: vec![],
        },
        bindings: vec![],
    };
    SpeculativeActivationPlan {
        schema_version: SPECULATIVE_ACTIVATION_SCHEMA_VERSION,
        captures: CapturePlan::none(),
        interventions: InterventionPlan::none(),
        bounds: CaptureInvocationBounds {
            batch: 1,
            max_sequence: 4,
            max_context: None,
            max_predictions: 8,
        },
    }
    .admit(&discovery)
    .unwrap()
}

#[test]
fn scoped_authority_restores_prior_observer_and_cannot_reset_an_active_allowance() {
    let trace = trace(20);
    let mut strategy = strategy();
    let mut executor = EmbeddedPredictionExecutor::<_, Mechanisms>::with_observers(
        &mut strategy,
        observers(&trace),
    );
    let request = eredu_core::SpeculativeRequestId::new(0);
    executor
        .configure_activation_capture(empty_authority(), request, ())
        .unwrap();
    assert!(!executor.requires_activation_origin());
    assert!(executor
        .configure_activation_capture(empty_authority(), request, ())
        .is_err());
    executor.prefill(vec![1, 2, 3], &mut cache(), ()).unwrap();
    assert!(trace.lock().unwrap().phases.is_empty());
    let restored = executor.into_observers();
    let mut executor =
        EmbeddedPredictionExecutor::<_, Mechanisms>::with_observers(&mut strategy, restored);
    assert!(executor.requires_activation_origin());
    executor.prefill(vec![1, 2, 3], &mut cache(), ()).unwrap();
    assert_eq!(trace.lock().unwrap().phases.len(), 2);
    assert!(executor
        .configure_activation_capture(empty_authority(), request, ())
        .is_err());
    assert!(executor.requires_activation_origin());
}
