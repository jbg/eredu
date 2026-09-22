//! Actual SessionPrefill issues the ticket; scalar mechanisms supply no native fact.
use super::*;
use eredu_core::{capture::*, *};
use eredu_runtime::{
    capture::{FundedCaptureError, ScheduledCaptureBackend},
    inspection::*,
    working_memory::*,
};

#[derive(Debug, thiserror::Error)]
#[error("original retained preparation sentinel")]
struct Original(Arc<()>);
#[derive(Clone, Copy, PartialEq, Eq)]
enum Failure {
    None,
    Bootstrap,
    Input,
    Retirement,
    RetirementAndSettlement,
    FinalCancellation,
    CompletedSources,
    PanicBootstrap,
    PanicInput,
    PanicRetirement,
}
struct CaptureBackend {
    scope: Option<WorkingMemoryFundingScope>,
    segment: Option<CaptureSourceSegment>,
    failure: Failure,
    identity: Arc<()>,
    prepared: usize,
    retired: usize,
    cancellation: GenerationCancellationToken,
}
impl ScheduledCaptureBackend for CaptureBackend {
    type Tensor = FakeTensor;
    type Error = Error;
    fn prepare_prefill_chunk_retention(
        &mut self,
        bootstrap: CapturePrefillSourceBootstrap<'_>,
        context: &PrefillChunkRetentionContext<'_>,
    ) -> Result<Option<PreparedPrefillChunkRetention>, FundedCaptureError<Error>> {
        check_inference_scope();
        self.prepared += 1;
        let (segment, registration) = bootstrap
            .begin_segment(self.scope.as_mut().unwrap(), context)
            .map_err(CaptureRunHostError::from)?;
        self.segment = Some(segment);
        if self.failure == Failure::PanicBootstrap {
            std::panic::panic_any(self.identity.clone());
        }
        if self.failure == Failure::Bootstrap {
            return Err(FundedCaptureError::Backend(Error::backend_retained_source(
                Original(self.identity.clone()),
            )));
        }
        Ok(Some(registration))
    }
    fn retire_prefill_chunk_retention(
        &mut self,
        ticket: SettledPrefillChunkRetention,
    ) -> Result<(), FundedCaptureError<Error>> {
        check_inference_scope();
        INFERENCE_SCOPE_TRACE.with(|trace| {
            let trace = trace.borrow();
            assert_eq!(trace.active, 1, "only the post-chunk guard remains active");
            assert_eq!(
                trace.opened,
                trace.finished + 1,
                "all prior preparation/model/completion guards settled before the ticket"
            );
        });
        self.segment
            .as_ref()
            .unwrap()
            .validate_settled_ticket(self.scope.as_ref().unwrap(), &ticket)
            .map_err(CaptureRunHostError::from)?;
        row_sources::retiring(&ticket);
        assert_eq!(ticket.chunk().input, 0..1);
        self.retired += 1;
        if self.failure == Failure::PanicRetirement {
            std::panic::panic_any(self.identity.clone());
        }
        if self.failure == Failure::FinalCancellation {
            self.cancellation.cancel();
        }
        if self.failure == Failure::RetirementAndSettlement {
            INFERENCE_SCOPE_TRACE.with(|s| s.borrow_mut().fail_finish = true);
        }
        if matches!(
            self.failure,
            Failure::Retirement | Failure::RetirementAndSettlement
        ) {
            return Err(FundedCaptureError::Backend(Error::backend_retained_source(
                Original(self.identity.clone()),
            )));
        }
        // A validates only the exact ticket. It intentionally leaves the channel
        // intact; no native carrier/physical retirement is implemented here.
        Ok(())
    }
    fn validate_source(
        &self,
        _: &FakeTensor,
        _: &CaptureTensorGeometry<'_>,
    ) -> Result<checkpoint::TensorDtype, Error> {
        unreachable!("declared fixture hook is deliberately not emitted")
    }
    fn estimate(
        &self,
        _: &FakeTensor,
        _: &CaptureTensorGeometry<'_>,
    ) -> Result<CaptureUsage, CaptureError> {
        unreachable!()
    }
    fn transform(
        &mut self,
        _: &FakeTensor,
        _: CaptureTensorClaim<'_, '_>,
    ) -> Result<ClaimedCaptureTensor, Error> {
        unreachable!()
    }
}
struct Input {
    geometry: InferenceGeometry,
    prepared: Rc<Cell<usize>>,
    failure: Failure,
    identity: Arc<()>,
}
impl
    PreparedPrefillSource<
        OrdinaryTextFixture,
        FakeBackend,
        DeviceState<FakeBackend, FakeLayerState>,
    > for Input
{
    type Chunk = FakeTensor;
    fn geometry(&self) -> InferenceGeometry {
        self.geometry
    }
    fn prepare_chunk(&self, _: &PrefillChunk, _: &()) -> Result<FakeTensor, Error> {
        check_inference_scope();
        self.prepared.set(self.prepared.get() + 1);
        if self.failure == Failure::PanicInput {
            std::panic::panic_any(self.identity.clone());
        }
        if self.failure == Failure::Input {
            return Err(Error::backend_retained_source(Original(
                self.identity.clone(),
            )));
        }
        Ok(FakeTensor(vec![3, 7]))
    }
    fn input<'a>(&'a self, value: &'a FakeTensor) -> &'a FakeTensor {
        value
    }
}
fn capture_source() -> SharedCapturePlan {
    let limits = CaptureUsage {
        captures: u64::MAX,
        retained_bytes: u64::MAX,
        host_bytes: u64::MAX,
        encoded_bytes: u64::MAX,
    };
    let point = ObservationPoint {
        path: "retained.unemitted".into(),
        node_id: "neutral".into(),
        meaning: "retention protocol fixture".into(),
        value_type: ObservationValueType::Tensor,
        dtype: ObservationDtype::Floating,
        axes: Some(vec![TensorAxis {
            name: "sequence".into(),
            dimension: SymbolicDimension::Sequence,
        }]),
        prefill: true,
        decode: true,
        requirements: vec![],
        position: ObservationPosition::BeforeIntervention,
        retained_bytes: None,
        host_bytes: None,
    };
    let catalog = ObservationCatalog {
        schema_version: 1,
        points: vec![point],
        completeness: DescriptionCompleteness::Complete,
    };
    let support = ObservationSupportReport {
        schema_version: 1,
        capture: Default::default(),
        points: vec![ObservationSupport {
            path: "retained.unemitted".into(),
            prefill: ObservationSupportStatus::Supported,
            decode: ObservationSupportStatus::Supported,
            floating_to_f32: true,
        }],
    };
    let capabilities = CaptureCapabilities {
        transformations: vec![CaptureTransformKind::FullTensor],
        max_histogram_bins: 0,
        conditions: vec![],
    };
    SharedCapturePlan::new(
        CapturePlan {
            schema_version: 1,
            selections: vec![CaptureSelection {
                id: "one".into(),
                path: "retained.unemitted".into(),
                schedule: CaptureSchedule::default(),
                slices: vec![],
                transform: CaptureTransform::FullTensor,
            }],
            limits: CaptureLimits {
                per_step: limits,
                cumulative: limits,
                on_limit: CaptureLimitPolicy::Fail,
            },
        }
        .admit(
            &catalog,
            &support,
            &capabilities,
            CaptureRequestShape {
                batch: 1,
                prompt_tokens: 1,
                max_predictions: 1,
            },
        )
        .unwrap(),
    )
}
fn cause<'a, T: std::error::Error + 'static>(
    mut error: &'a (dyn std::error::Error + 'static),
) -> Option<&'a T> {
    loop {
        if let Some(value) = error.downcast_ref() {
            return Some(value);
        }
        error = error.source()?;
    }
}
fn exercise(failure: Failure, initial_cancel: bool) {
    INFERENCE_SCOPE_TRACE.with(|s| {
        *s.borrow_mut() = InferenceScopeTrace {
            required: true,
            ..Default::default()
        }
    });
    let (mut session, counters) = session();
    let source = capture_source();
    let h = CaptureRunHostPlan::prepare(&source)
        .unwrap()
        .initialization_peak_bytes();
    let geometry = InferenceGeometry {
        batch_size: 1,
        cached_positions: 0,
        input_positions: 1,
        max_output_tokens: 1,
        prefill_chunk_positions: 1,
        output: OutputDemand::Sequence,
    };
    let mut admission = mock_inference_admission(geometry);
    let bound = |n| WorkspaceBound::bounded(n, "actual neutral scalar protocol account");
    admission.state = admission
        .state
        .with_execution_workspace(crate::memory::workspace(ExecutionWorkspaceEstimate {
            physical_domains: None,
            geometry,
            activations: bound(h + 384),
            attention: bound(0),
            vocabulary: bound(0),
            state_update: bound(0),
            materialization: bound(0),
            retained: bound(0),
        }))
        .unwrap();
    admission.incremental_required_bytes = Some(h + 384);
    let pool = crate::memory::host_ledger(crate::memory::reservation_bytes(&admission), 0).unwrap();
    let (reservation, run) = pool
        .reserve(session.inference_execution_identity(), &admission)
        .unwrap()
        .into_funding()
        .unwrap();
    let request = InferenceRequest::from(&reservation);
    let mut funded = run
        .prepare_capture_run(&reservation, CaptureRunHostPlan::prepare(&source).unwrap())
        .unwrap()
        .into_capture_session()
        .unwrap();
    let identity = Arc::new(());
    let prepared = Rc::new(Cell::new(0));
    let cancellation = GenerationCancellationToken::new();
    if initial_cancel {
        cancellation.cancel();
    }
    let mut backend = CaptureBackend {
        scope: Some(run.scope().unwrap()),
        segment: None,
        failure,
        identity: identity.clone(),
        prepared: 0,
        retired: 0,
        cancellation: cancellation.clone(),
    };
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        funded
            .with_observer(
                &mut backend,
                0,
                &|e| Error::backend_retained_source(e),
                |observer| {
                    let mut driver = eredu_runtime::prefill::PrefillDriver::new(
                        session.inference_execution_identity(),
                        request.clone(),
                        geometry,
                        cancellation,
                    )
                    .unwrap();
                    let mut borrowed = eredu_runtime::BorrowedActivationObserver(observer);
                    let mut executor = eredu_runtime::replicated_session::SessionPrefill::new(
                        &mut session,
                        Input {
                            geometry,
                            prepared: prepared.clone(),
                            failure,
                            identity: identity.clone(),
                        },
                        request.clone(),
                        &(),
                        &mut borrowed,
                    )
                    .unwrap();
                    let result = driver.run(&mut executor, |_, _| {});
                    drop(executor);
                    borrowed.finish_prefill(matches!(
                        result,
                        Ok(eredu_runtime::prefill::PrefillOutcome::Complete)
                    ));
                    result
                },
            )
            .unwrap()
    }));
    let result = match result {
        Ok(result) => Some(result),
        Err(payload) => {
            assert!(matches!(
                failure,
                Failure::PanicBootstrap | Failure::PanicInput | Failure::PanicRetirement
            ));
            let payload = payload.downcast::<Arc<()>>().expect("same panic type");
            assert!(Arc::ptr_eq(&payload, &identity));
            assert_eq!(backend.prepared, 1);
            assert_eq!(
                backend.retired,
                usize::from(failure == Failure::PanicRetirement)
            );
            assert_eq!(
                funded.spent_steps(),
                usize::from(failure == Failure::PanicRetirement)
            );
            let frame = funded.take_shared_step().unwrap();
            if failure == Failure::PanicRetirement {
                assert_eq!(
                    frame.expect("aborted sealed frame").outcome(),
                    CaptureStepOutcome::Aborted
                );
            } else {
                assert!(frame.is_none());
            }
            let before = counters.snapshot().forward_calls;
            assert!(session.decode(&FakeTensor(vec![9, 11]), &()).is_err());
            assert_eq!(counters.snapshot().forward_calls, before);
            INFERENCE_SCOPE_TRACE.with(|s| {
                let s = s.borrow();
                assert_eq!(s.abandoned, 1);
                assert_eq!(s.active, 0);
            });
            None
        }
    };
    if let Some(result) = result {
        if initial_cancel {
            assert!(matches!(
                result,
                Ok(eredu_runtime::prefill::PrefillOutcome::Cancelled)
            ));
            assert_eq!(
                (backend.prepared, backend.retired, prepared.get()),
                (0, 0, 0)
            );
            assert_eq!(funded.spent_steps(), 0);
        } else if failure == Failure::FinalCancellation {
            assert!(backend.cancellation.is_cancelled());
            assert_eq!(
                (backend.prepared, backend.retired, prepared.get()),
                (1, 1, 1),
                "cancellation occurs after the final model work and sole ticket delivery"
            );
            assert!(
                matches!(
                    result,
                    Ok(eredu_runtime::prefill::PrefillOutcome::Cancelled)
                ),
                "final callback cancellation result: {result:?}"
            );
            assert_eq!(funded.spent_steps(), 1);
            let frame = funded.take_shared_step().unwrap().unwrap();
            assert_eq!(frame.outcome(), CaptureStepOutcome::Aborted);
            drop(frame);
        } else if failure == Failure::CompletedSources {
            let eredu_runtime::prefill::PrefillError::Submission(error) =
                result.err().expect("completed-source failure")
            else {
                panic!("typed submission error")
            };
            assert!(matches!(
                error,
                ReplicatedTextSessionError::Mechanism("original completed source error")
            ));
            assert_eq!(
                (backend.prepared, backend.retired, prepared.get()),
                (1, 0, 1)
            );
            assert_eq!(funded.spent_steps(), 1);
            assert_eq!(
                funded.take_shared_step().unwrap().unwrap().outcome(),
                CaptureStepOutcome::Aborted
            );
            // Assert before this shared fixture's final trace reset/teardown.
            INFERENCE_SCOPE_TRACE.with(|t| {
                let t = t.borrow();
                assert_eq!(t.active, 0);
                assert_eq!(t.abandoned, 1);
            });
        } else if failure == Failure::None {
            assert!(
                matches!(result, Ok(eredu_runtime::prefill::PrefillOutcome::Complete)),
                "ordinary final chunk result: {result:?}"
            );
            assert_eq!(
                (backend.prepared, backend.retired, prepared.get()),
                (1, 1, 1)
            );
            assert_eq!(funded.spent_steps(), 1);
            let frame = funded.take_shared_step().unwrap().unwrap();
            assert_eq!(frame.outcome(), CaptureStepOutcome::Committed);
            drop(frame);
        } else {
            let error = result.err().expect("injected original failure");
            let eredu_runtime::prefill::PrefillError::Submission(error) = error else {
                panic!("typed submission error")
            };
            let mut error = &error;
            while let ReplicatedTextSessionError::BeforeStateMutation(inner) = error {
                error = inner;
            }
            let ReplicatedTextSessionError::Architecture(error) = error else {
                panic!("original architecture/observer domain, got {error}")
            };
            let found = cause::<Original>(error).expect("original typed error chain");
            assert!(Arc::ptr_eq(&identity, &found.0));
            assert_eq!(
                backend.retired,
                usize::from(matches!(
                    failure,
                    Failure::Retirement | Failure::RetirementAndSettlement
                ))
            );
            assert_eq!(prepared.get(), usize::from(failure != Failure::Bootstrap));
            assert_eq!(
                funded.spent_steps(),
                usize::from(matches!(
                    failure,
                    Failure::Retirement | Failure::RetirementAndSettlement
                ))
            );
            let frame = funded.take_shared_step().unwrap();
            match failure {
                Failure::Retirement | Failure::RetirementAndSettlement => {
                    let frame = frame.expect("retirement failure aborts the sealed final frame");
                    assert_eq!(frame.outcome(), CaptureStepOutcome::Aborted);
                    drop(frame);
                }
                Failure::Bootstrap | Failure::Input => {
                    assert!(
                        frame.is_none(),
                        "preparation failed before the first frame claim"
                    );
                }
                Failure::None
                | Failure::CompletedSources
                | Failure::FinalCancellation
                | Failure::PanicBootstrap
                | Failure::PanicInput
                | Failure::PanicRetirement => unreachable!(),
            }
        }
    }
    assert_eq!(
        counters.snapshot().forward_calls,
        usize::from(
            !initial_cancel
                && !matches!(
                    failure,
                    Failure::Bootstrap
                        | Failure::Input
                        | Failure::PanicBootstrap
                        | Failure::PanicInput
                )
        )
    );
    INFERENCE_SCOPE_TRACE.with(|s| {
        let s = s.borrow();
        assert_eq!(s.active, 0);
        assert_eq!(s.opened, s.finished + s.abandoned);
    });
    // Test mechanisms have no native work; their positive teardown releases any
    // failed mock guards after their custody and original cause were inspected.
    INFERENCE_SCOPE_TRACE.with(|s| *s.borrow_mut() = InferenceScopeTrace::default());
    drop((funded, session, request, reservation));
    backend.scope.take().unwrap().certify().unwrap();
    drop((backend, run));
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}
#[test]
fn actual_session_issues_final_ticket_before_outer_frame_commit() {
    exercise(Failure::None, false);
}
#[test]
fn initial_cancel_never_bootstraps_or_claims() {
    exercise(Failure::None, true);
}
#[test]
fn bootstrap_and_input_failures_keep_original_cause_without_frame_or_ticket() {
    for f in [Failure::Bootstrap, Failure::Input] {
        exercise(f, false);
    }
}
#[test]
fn retirement_error_survives_guard_settlement_failure_and_aborts_sealed_frame() {
    for f in [Failure::Retirement, Failure::RetirementAndSettlement] {
        exercise(f, false);
    }
}

#[test]
fn final_callback_cancellation_aborts_frame_after_one_ticket() {
    exercise(Failure::FinalCancellation, false);
}

#[test]
fn direct_session_bootstrap_input_and_retirement_unwind_fences_and_preserves_original_bank() {
    for failure in [
        Failure::PanicBootstrap,
        Failure::PanicInput,
        Failure::PanicRetirement,
    ] {
        exercise(failure, false);
    }
}

#[path = "retention/bounded_pins.rs"]
mod bounded_pins;

#[path = "retention/row_sources.rs"]
pub(crate) mod row_sources;
