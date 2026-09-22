use super::*;
use eredu_core::capture::*;
use eredu_core::intervention::*;
#[path = "observed_mock/funded.rs"]
mod funded;
pub(super) use funded::plan as funded_plan;

#[derive(Default)]
pub(super) struct State {
    pub sampling: Sampling,
    pub original: Option<admitted_text::PreparationOwner>,
    pub capture: Option<eredu_runtime::capture::CaptureSession>,
    pub funded: Option<eredu_runtime::capture::FundedCaptureSession>,
    pub(super) host_mechanism: Option<funded::HostMechanism>,
}

#[derive(Clone, Default)]
pub(super) struct Sampling {
    pub temperature: f32,
    pub seed: Option<u64>,
    pub prediction: u64,
}

pub(super) fn artifact_identity() -> eredu_core::artifact::ArtifactIdentity {
    use sha2::Digest;
    let equations = b"prefill: token_count; decode: token+1; logits intervention: token*scale";
    eredu_core::artifact::fingerprint_artifact(
        "neutral-arithmetic-fixture",
        [eredu_core::artifact::ArtifactMemberIdentity::new(
            "equations",
            equations.len() as u64,
            sha2::Sha256::digest(equations).into(),
        )],
    )
    .unwrap()
}

pub(super) fn discovery() -> CaptureDiscovery {
    let path = eredu_core::MODEL_LOGITS_OBSERVATION_PATH.to_owned();
    let capabilities = CaptureCapabilities {
        transformations: vec![CaptureTransformKind::Summary],
        max_histogram_bins: 0,
        conditions: vec![],
    };
    CaptureDiscovery {
        artifact_identity: artifact_identity().to_string(),
        catalog: eredu_core::ObservationCatalog {
            schema_version: 1,
            completeness: eredu_core::DescriptionCompleteness::Complete,
            points: vec![eredu_core::ObservationPoint {
                path: path.clone(),
                node_id: "output".into(),
                meaning: "mock logits".into(),
                value_type: eredu_core::ObservationValueType::Tensor,
                dtype: eredu_core::ObservationDtype::Floating,
                axes: Some(vec![
                    eredu_core::TensorAxis {
                        name: "batch".into(),
                        dimension: eredu_core::SymbolicDimension::Batch,
                    },
                    eredu_core::TensorAxis {
                        name: "sequence".into(),
                        dimension: eredu_core::SymbolicDimension::Sequence,
                    },
                    eredu_core::TensorAxis {
                        name: "vocabulary".into(),
                        dimension: eredu_core::SymbolicDimension::Known(1),
                    },
                ]),
                prefill: true,
                decode: true,
                requirements: vec![eredu_core::ObservationRequirement::ActivationHooks],
                position: eredu_core::ObservationPosition::BeforeIntervention,
                retained_bytes: None,
                host_bytes: None,
            }],
        },
        support: eredu_core::ObservationSupportReport {
            schema_version: 1,
            capture: capabilities,
            points: vec![eredu_core::ObservationSupport {
                path,
                prefill: eredu_core::ObservationSupportStatus::Supported,
                decode: eredu_core::ObservationSupportStatus::Supported,
                floating_to_f32: true,
            }],
        },
    }
}

pub(super) fn cost(shape: &[u64]) -> Result<CaptureUsage, CaptureError> {
    Ok(CaptureUsage {
        captures: 1,
        retained_bytes: mul(elements(shape)?, 4)?,
        host_bytes: 128,
        encoded_bytes: 512,
    })
}

pub(super) fn validate(plan: &AdmittedCapturePlan) -> Result<(), CaptureError> {
    eredu_runtime::capture::validate_session(plan, &discovery(), |shape, _, _| cost(shape))
}

struct Mechanism;

struct SpeculativeProvider;
impl eredu_runtime::capture::CaptureBackendProvider for SpeculativeProvider {
    type Tensor = Value;
    type Error = MockError;
    type Backend<'a> = Mechanism;
    fn backend(&mut self) -> Mechanism {
        Mechanism
    }
}

type InternalObserver = eredu_runtime::capture::SpeculativeCaptureObserver<
    SpeculativeProvider,
    fn(&eredu_runtime::capture::CaptureExecutionError<MockError>) -> MockError,
>;

pub(super) struct InternalCapture(InternalObserver);
impl InternalCapture {
    pub fn new(
        plan: &eredu_core::speculative::AdmittedSpeculativeActivations,
        request: eredu_core::SpeculativeRequestId,
    ) -> Result<Option<Self>, CaptureError> {
        fn error(error: &eredu_runtime::capture::CaptureExecutionError<MockError>) -> MockError {
            MockError::Capture(error.to_string())
        }
        Ok(
            eredu_runtime::capture::SpeculativeCaptureObserver::from_admitted(
                plan,
                SpeculativeProvider,
                error as fn(&eredu_runtime::capture::CaptureExecutionError<MockError>) -> MockError,
                request,
                std::sync::Arc::new(Estimates),
            )?
            .map(Self),
        )
    }
    pub fn origin(&mut self, origin: Option<eredu_core::speculative::SpeculativeActivationOrigin>) {
        use eredu_runtime::inspection::SpeculativeActivationObserver;
        self.0.set_activation_origin(origin);
    }
    pub fn take(&mut self) -> Option<eredu_core::speculative::SpeculativeActivationCapture> {
        use eredu_runtime::inspection::SpeculativeActivationObserver;
        self.0.take_activation_capture()
    }
    pub fn error(&mut self) -> Option<eredu_core::speculative::SpeculativeControlError> {
        use eredu_runtime::inspection::SpeculativeActivationObserver;
        self.0.take_activation_error()
    }
    pub fn run(
        &mut self,
        phase: eredu_core::speculative::SpeculativeActivationPhase,
        sequence: usize,
    ) -> Result<f32, MockError> {
        eredu_runtime::inspection::with_speculative_activation(
            Some(&mut self.0),
            phase,
            sequence,
            |observer| {
                let value = Value {
                    shape: [1, sequence as u64, 1],
                    scale: 1.0,
                };
                eredu_runtime::observe_and_intervene(
                    observer.unwrap(),
                    eredu_core::MODEL_LOGITS_OBSERVATION_PATH,
                    &value,
                )
                .map(|value| value.scale)
            },
        )
    }
}
#[derive(Clone)]
pub(crate) struct Value {
    shape: [u64; 3],
    scale: f32,
}
impl CaptureBackend for Mechanism {
    type Tensor = Value;
    type Error = MockError;
    fn shape(&self, tensor: &Self::Tensor) -> Result<Vec<u64>, Self::Error> {
        Ok(tensor.shape.to_vec())
    }
    fn estimate(
        &self,
        tensor: &Self::Tensor,
        _: &CaptureSelection,
        _: &ResolvedCaptureSlice,
    ) -> Result<CaptureUsage, CaptureError> {
        cost(&tensor.shape)
    }
    fn transform(
        &mut self,
        tensor: &Self::Tensor,
        selection: &CaptureSelection,
        _: &ResolvedCaptureSlice,
    ) -> Result<CapturePayload, Self::Error> {
        if selection.id == "injected-native-failure" {
            return Err(MockError::Capture("injected capture fault".into()));
        }
        let count = elements(&tensor.shape).unwrap();
        Ok(CapturePayload::Summary(CaptureSummary {
            elements: count,
            finite: count,
            non_finite: 0,
            nan: 0,
            positive_infinity: 0,
            negative_infinity: 0,
            min: Some(f64::from(tensor.scale)),
            max: Some(f64::from(tensor.scale)),
            mean: Some(f64::from(tensor.scale)),
            rms: Some(f64::from(tensor.scale.abs())),
        }))
    }
}

impl State {
    pub(super) fn take_capture(&mut self) -> Result<Option<SharedCapturedStep>, MockError> {
        match &mut self.funded {
            Some(capture) => Ok(capture.take_shared_step()?),
            None => Ok(self
                .capture
                .as_mut()
                .and_then(|capture| capture.take_shared_step())),
        }
    }
    pub fn observe(
        &mut self,
        phase: CapturePhase,
        sequence: usize,
        token: u32,
    ) -> Result<u32, MockError> {
        let mut value = Value {
            shape: [1, sequence as u64, 1],
            scale: 1.0,
        };
        if self.funded.is_some() {
            value = self.observe_funded(value, phase)?;
        } else if let Some(capture) = &mut self.capture {
            capture
                .begin_step(phase, self.sampling.prediction)
                .map_err(|e| MockError::Capture(e.to_string()))?;
            use eredu_runtime::ActivationObserver;
            let mut observer = eredu_runtime::intervention::CaptureObserver::new(
                capture,
                Mechanism,
                |error: eredu_runtime::capture::CaptureExecutionError<MockError>| {
                    MockError::Capture(error.to_string())
                },
            );
            observer.observe(eredu_core::MODEL_LOGITS_OBSERVATION_PATH, &value)?;
            if let Some(effective) =
                observer.intervene(eredu_core::MODEL_LOGITS_OBSERVATION_PATH, &value)?
            {
                value = effective;
            }
            observer.finish()?;
        }
        self.sampling.prediction += 1;
        Ok((token as f32 * value.scale) as u32)
    }
}

impl InterventionBackend for Mechanism {
    fn matches_intervention_shape(&self, value: &Value, shape: &[u64]) -> Result<bool, MockError> {
        Ok(value.shape == shape)
    }
    fn mask_components(&mut self, _: &Value, _: &[u32], _: bool) -> Result<Value, MockError> {
        Err(MockError::Capture(
            "mock does not advertise component masks".into(),
        ))
    }

    fn intervention_dtype(&self, _: &Value) -> Result<InterventionDtype, MockError> {
        Ok(InterventionDtype::Float32)
    }
    fn validate_intervention_geometry(
        &self,
        source: &[u64],
        slice: &ResolvedCaptureSlice,
    ) -> Result<(), CaptureError> {
        Estimates.validate_geometry(source, slice)
    }
    fn select_region(
        &mut self,
        value: &Value,
        slice: &ResolvedCaptureSlice,
    ) -> Result<Value, MockError> {
        Ok(Value {
            shape: slice
                .shape
                .as_slice()
                .try_into()
                .map_err(|_| MockError::CaptureGeometry)?,
            scale: value.scale,
        })
    }
    fn update_region(
        &mut self,
        value: &Value,
        _: &ResolvedCaptureSlice,
        replacement: &Value,
    ) -> Result<Value, MockError> {
        Ok(Value {
            shape: value.shape.clone(),
            scale: replacement.scale,
        })
    }
    fn zeros(&mut self, shape: &[u64], _: InterventionDtype) -> Result<Value, MockError> {
        Ok(Value {
            shape: shape.try_into().map_err(|_| MockError::CaptureGeometry)?,
            scale: 0.0,
        })
    }
    fn scale(&mut self, value: &Value, factor: f32) -> Result<Value, MockError> {
        if factor == 13.0 {
            return Err(MockError::InjectedCapture);
        }
        Ok(Value {
            shape: value.shape.clone(),
            scale: value.scale * factor,
        })
    }
    fn fill_masked(&mut self, _: &Value, _: &[bool], _: f32) -> Result<Value, MockError> {
        Err(MockError::Capture("unsupported mask".into()))
    }
    fn realize_tensor(&mut self, _: &InterventionTensor) -> Result<Value, MockError> {
        Err(MockError::Capture("unsupported payload".into()))
    }
    fn add(&mut self, _: &Value, _: &Value) -> Result<Value, MockError> {
        Err(MockError::Capture("unsupported addition".into()))
    }
    fn fill_columns(&mut self, _: &Value, _: &[u32], _: f32) -> Result<Value, MockError> {
        Err(MockError::Capture("unsupported column fill".into()))
    }
}

pub(super) struct Estimates;
impl InterventionEstimator for Estimates {
    fn activation_usage(
        &self,
        source: &[u64],
        slice: &ResolvedCaptureSlice,
        _: &InterventionAction,
    ) -> Result<CaptureUsage, CaptureError> {
        let elements = |shape: &[u64]| {
            shape
                .iter()
                .try_fold(1u64, |n, d| n.checked_mul(*d).ok_or(CaptureError::Overflow))
        };
        let bytes = elements(source)?
            .checked_add(
                elements(&slice.shape)?
                    .checked_mul(3)
                    .ok_or(CaptureError::Overflow)?,
            )
            .and_then(|n| n.checked_mul(8))
            .ok_or(CaptureError::Overflow)?;
        Ok(CaptureUsage {
            captures: 0,
            retained_bytes: bytes,
            host_bytes: bytes,
            encoded_bytes: 0,
        })
    }

    fn validate_geometry(&self, _: &[u64], _: &ResolvedCaptureSlice) -> Result<(), CaptureError> {
        Ok(())
    }
    fn capture_usage(
        &self,
        source: &[u64],
        _: &CaptureSelection,
        _: &ResolvedCaptureSlice,
    ) -> Result<CaptureUsage, CaptureError> {
        cost(source)
    }
    fn original_route_usage(
        &self,
        _: &InterventionRoutingPolicy,
        _: u64,
    ) -> Result<CaptureUsage, CaptureError> {
        Err(CaptureError::Unsupported(
            "mock has no routing mechanism".into(),
        ))
    }
}

pub(super) fn intervention_discovery(session: &str) -> InterventionDiscovery {
    let capture = discovery();
    InterventionDiscovery {
        schema_version: 1,
        artifact_identity: capture.artifact_identity,
        session_identity: Some(session.into()),
        points: vec![InterventionPoint {
            path: eredu_core::MODEL_LOGITS_OBSERVATION_PATH.into(),
            node_id: "output".into(),
            stage: InterventionStage::LogitsBeforeSampling,
            axes: capture.catalog.points[0].axes.clone().unwrap(),
            dtypes: vec![InterventionDtype::Float32],
            operations: vec![InterventionKind::Zero, InterventionKind::Scale],
            score_stages: vec![],
            prefill: eredu_core::ObservationSupportStatus::Supported,
            decode: eredu_core::ObservationSupportStatus::Supported,
            conditions: vec![],
            routed_units: None,
            routing: None,
        }],
    }
}

pub(super) fn plan() -> CapturePlan {
    let usage = CaptureUsage {
        captures: 100,
        retained_bytes: 1_000_000,
        host_bytes: 1_000_000,
        encoded_bytes: 1_000_000,
    };
    CapturePlan {
        schema_version: 1,
        selections: vec![CaptureSelection {
            id: "summary".into(),
            path: eredu_core::MODEL_LOGITS_OBSERVATION_PATH.into(),
            schedule: CaptureSchedule::default(),
            slices: vec![],
            transform: CaptureTransform::Summary,
        }],
        limits: CaptureLimits {
            per_step: usage,
            cumulative: usage,
            on_limit: CaptureLimitPolicy::Fail,
        },
    }
}

pub(super) fn intervention_plan(factor: f32) -> InterventionPlan {
    InterventionPlan {
        schema_version: 1,
        operations: vec![InterventionOperation {
            id: "scale-logits".into(),
            target: eredu_core::MODEL_LOGITS_OBSERVATION_PATH.into(),
            schedule: CaptureSchedule::default(),
            slices: vec![],
            action: InterventionAction::Scale {
                dtype: InterventionDtype::Float32,
                factor,
            },
            evidence: InterventionEvidence::Summary,
        }],
    }
}

#[test]
fn intervened_facade_keeps_outcomes_cancellation_failure_and_consumer_lifetimes() {
    use eredu::api::{ObservedGenerationEvent as Event, TraceLimits};
    use std::ops::ControlFlow;
    let words = WordLevel::builder()
        .vocab(
            std::iter::once(("[UNK]".into(), 0))
                .chain((1..64).map(|id| (format!("token{id}"), id)))
                .collect(),
        )
        .unk_token("[UNK]".into())
        .build()
        .unwrap();
    let mut tokenizer = Tokenizer::new(words);
    tokenizer.with_pre_tokenizer(Some(Whitespace::default()));
    tokenizer.with_decoder(Some(ByteLevel::default()));
    tokenizer
        .add_special_tokens([AddedToken::from("<|im_end|>", true).normalized(false)])
        .unwrap();
    let eos = tokenizer.token_to_id("<|im_end|>").unwrap();
    let mut model = original_sources::Fixture::from_runtime(
        ModelRuntime::prepare(MockBackend, ()).unwrap(),
        ChatTokenizer::from_tokenizer(tokenizer),
        LoadedTextModelConfig {
            model_family: ModelKind::Qwen2,
            effective_model_type: "qwen2".into(),
            model_id: "intervention-mock".into(),
            chat_template: Some(ModelChatTemplate::Single(QWEN_TEMPLATE.into())),
            eos_token_ids: vec![eos],
            checkpoint_generation_config: None,
        },
    )
    .unwrap();
    let chat = {
        let request = ChatTemplateRequest {
            messages: vec![serde_json::json!({"role":"user","content":"token1"})],
            add_generation_prompt: true,
            ..Default::default()
        };
        let cancellation = eredu_core::GenerationCancellationToken::new();
        let source = model
            .chat_source(!request.tools.is_empty(), &cancellation)
            .unwrap()
            .unwrap();
        model
            .prepare_chat(
                &source,
                &request,
                &crate::memory::limits(original_sources::CAPACITY),
                &cancellation,
            )
            .unwrap()
            .unwrap()
    };
    let settings = PreparedChatGenerationSettings {
        overrides: GenerationConfigOverrides {
            max_new_tokens: Some(3),
            ..Default::default()
        },
        seed: 19,
        ..Default::default()
    };
    let limits = TraceLimits {
        per_record_bytes: 65536,
        total_bytes: 1024 * 1024,
    };
    let baseline = {
        let cancellation = eredu_core::GenerationCancellationToken::new();
        let mut on_event = |_| {};
        let mut request = eredu::api::PreparedChatRequest::new(
            &chat,
            original_sources::settings(settings.clone()),
        );
        request.stop_sequences = &[];
        model
            .start_prepared_chat(request, &cancellation)
            .and_then(|session| {
                session
                    .expect("uncancelled original request")
                    .run(&cancellation, &mut on_event)
            })
    }
    .unwrap();
    let capture = plan();
    let unchanged = intervention_plan(1.0);
    let make_request = |intervention| {
        let mut request = eredu::api::PreparedChatRequest::new(
            &chat,
            original_sources::settings(settings.clone()),
        );
        request.capture = Some(&capture);
        request.intervention = Some(intervention);
        request
    };
    let mut records = Vec::new();
    let mut session = model
        .start_controlled_chat(make_request(&unchanged), limits, Default::default(), |r| {
            records.push(r);
            std::ops::ControlFlow::Continue(())
        })
        .unwrap()
        .unwrap();
    session
        .run(|r| {
            records.push(r);
            std::ops::ControlFlow::Continue(())
        })
        .unwrap();
    assert_eq!(session.token_ids(), baseline.token_ids.as_ref());
    assert_eq!(session.finish_reason(), Some(baseline.finish_reason));
    assert!(session.timing().time_to_first_token().is_some());
    let identity = match &records[0].instrumentation {
        eredu::api::PreparedInstrumentationRecord::Intervened {
            intervention_plan_id,
            ..
        } => intervention_plan_id.clone(),
        _ => panic!("intervention source"),
    };
    let mut predictions = Vec::new();
    for record in &records {
        assert!(
            matches!(&record.instrumentation,eredu::api::PreparedInstrumentationRecord::Intervened{intervention_plan_id,..} if *intervention_plan_id==identity)
        );
        if let Some(Event::Token {
            captures: Some(step),
            ..
        }) = record.event.progress()
        {
            assert_eq!(step.interventions[0].outcome, InterventionOutcome::Applied);
            assert_eq!(step.interventions[0].evidence.len(), 2);
            assert_eq!(step.step_usage.captures, 3);
            assert_eq!(
                step.records[0].payload,
                step.interventions[0].evidence[0].payload
            );
            assert_eq!(
                step.interventions[0].evidence[0].payload,
                step.interventions[0].evidence[1].payload
            );
            predictions.push(step.prediction_index);
        }
    }
    assert_eq!(predictions, [0, 1, 2]);
    drop(session);
    drop(records);
    let empty = InterventionPlan::none();
    let mut session = model
        .start_controlled_chat(make_request(&empty), limits, Default::default(), |_| {
            ControlFlow::Continue(())
        })
        .unwrap()
        .unwrap();
    session
        .run(|r| {
            assert!(!matches!(
                r.instrumentation,
                eredu::api::PreparedInstrumentationRecord::Intervened { .. }
            ));
            if let Some(Event::Token {
                captures: Some(step),
                ..
            }) = r.event.progress()
            {
                assert!(step.interventions.is_empty());
                assert_eq!(step.records.len(), 1);
            }
            ControlFlow::Continue(())
        })
        .unwrap();
    assert_eq!(session.token_ids(), baseline.token_ids.as_ref());
    assert_eq!(session.finish_reason(), Some(baseline.finish_reason));
    drop(session);
    let mut invalid = intervention_plan(1.0);
    invalid.operations[0].target = "nonexistent".into();
    assert!(model
        .start_controlled_chat(
            make_request(&invalid),
            limits,
            Default::default(),
            |_| panic!("invalid admission delivery")
        )
        .is_err());
    let failed = intervention_plan(13.0);
    let mut records = Vec::new();
    let mut session = model
        .start_controlled_chat(make_request(&failed), limits, Default::default(), |r| {
            records.push(r);
            ControlFlow::Continue(())
        })
        .unwrap()
        .unwrap();
    assert!(session
        .run(|r| {
            records.push(r);
            ControlFlow::Continue(())
        })
        .is_err());
    assert!(records.iter().any(|r|matches!(r.event.progress(),Some(Event::CaptureFailure{captures,..}) if matches!(captures.interventions[0].outcome,InterventionOutcome::Failed{..}))));
    assert!(matches!(
        records.last().unwrap().event.progress(),
        Some(Event::Failed { .. })
    ));
    drop(session);
    drop(records);
    for pre_cancel in [false, true] {
        let control = eredu_core::execution_control::GenerationControlHandle::default();
        if pre_cancel {
            control.cancel();
        }
        let mut closed = false;
        let session = model
            .start_controlled_chat(make_request(&unchanged), limits, control, |_| {
                ControlFlow::Continue(())
            })
            .unwrap();
        if pre_cancel {
            assert!(session.is_none());
            continue;
        }
        let mut session = session.unwrap();
        session
            .run(|r| {
                assert!(!closed, "delivery after consumer Break");
                if matches!(r.event.progress(), Some(Event::Token { .. })) {
                    closed = true;
                    ControlFlow::Break(())
                } else {
                    ControlFlow::Continue(())
                }
            })
            .unwrap();
        assert_eq!(session.finish_reason(), Some(FinishReason::Cancelled));
        assert_eq!(session.token_ids().len(), 1);
        assert!(session.timing().time_to_first_token().is_some());
    }
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut session = model
            .start_controlled_chat(make_request(&unchanged), limits, Default::default(), |_| {
                ControlFlow::Continue(())
            })
            .unwrap()
            .unwrap();
        let _ = session.run(|r| {
            if matches!(r.event.progress(), Some(Event::Token { .. })) {
                panic!("consumer unwinds");
            }
            ControlFlow::Continue(())
        });
    }))
    .is_err());
    let mut session = model
        .start_controlled_chat(make_request(&unchanged), limits, Default::default(), |_| {
            ControlFlow::Continue(())
        })
        .unwrap()
        .unwrap();
    session.run(|_| ControlFlow::Continue(())).unwrap();
    assert_eq!(session.token_ids(), baseline.token_ids.as_ref());
    assert_eq!(session.finish_reason(), Some(baseline.finish_reason));
    drop(session);
    let zero = intervention_plan(0.0);
    let mut session = model
        .start_controlled_chat(make_request(&zero), limits, Default::default(), |_| {
            ControlFlow::Continue(())
        })
        .unwrap()
        .unwrap();
    session.run(|_| ControlFlow::Continue(())).unwrap();
    assert_eq!(session.token_ids(), [0, 0, 0]);
}

#[test]
fn intervention_admission_cannot_transfer_between_identical_artifact_sessions() {
    let runtime = ModelRuntime::prepare(MockBackend, ()).unwrap();
    let other = ModelRuntime::prepare(MockBackend, ()).unwrap();
    let request = CaptureRequestShape {
        batch: 1,
        prompt_tokens: 2,
        max_predictions: 3,
    };
    let discovery = MockBackend::capture_discovery(&runtime).unwrap();
    let capture = plan()
        .admit(
            &discovery.catalog,
            &discovery.support,
            &discovery.support.capture,
            request,
        )
        .unwrap();
    let admitted = intervention_plan(1.0)
        .admit(
            &MockBackend::intervention_discovery(&runtime).unwrap(),
            request,
            "facade",
        )
        .unwrap();
    MockBackend::validate_text_interventions(&runtime, &capture, &admitted).unwrap();
    assert!(MockBackend::validate_text_interventions(&other, &capture, &admitted).is_err());
}
