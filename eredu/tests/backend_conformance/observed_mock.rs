use super::*;
use eredu_core::capture::*;
use eredu_core::intervention::*;

#[derive(Default)]
pub(super) struct State {
    pub sampling: Sampling,
    pub capture: Option<eredu_runtime::capture::CaptureSession>,
}

#[derive(Clone, Default)]
pub(super) struct Sampling {
    pub temperature: f32,
    pub seed: Option<u64>,
    pub prediction: u64,
}

pub(super) fn discovery() -> CaptureDiscovery {
    let path = eredu_core::MODEL_LOGITS_OBSERVATION_PATH.to_owned();
    let capabilities = CaptureCapabilities {
        transformations: vec![CaptureTransformKind::Summary],
        max_histogram_bins: 0,
        physical_native_limit: false,
        conditions: vec![],
    };
    CaptureDiscovery {
        artifact_identity: "conformance-mock-artifact".into(),
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
#[derive(Clone)]
struct Value {
    shape: Vec<u64>,
    scale: f32,
}
impl CaptureBackend for Mechanism {
    type Tensor = Value;
    type Error = MockError;
    fn shape(&self, tensor: &Self::Tensor) -> Result<Vec<u64>, Self::Error> {
        Ok(tensor.shape.clone())
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
    pub fn observe(
        &mut self,
        phase: CapturePhase,
        sequence: usize,
        token: u32,
    ) -> Result<u32, MockError> {
        let mut value = Value {
            shape: vec![1, sequence as u64, 1],
            scale: 1.0,
        };
        if let Some(capture) = &mut self.capture {
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
            shape: slice.shape.clone(),
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
            shape: shape.to_vec(),
            scale: 0.0,
        })
    }
    fn scale(&mut self, value: &Value, factor: f32) -> Result<Value, MockError> {
        if factor == 13.0 {
            return Err(MockError::Capture("injected intervention fault".into()));
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
            physical_native_bytes: None,
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
    tokenizer.with_pre_tokenizer(Some(Whitespace));
    tokenizer.with_decoder(Some(ByteLevel::default()));
    tokenizer
        .add_special_tokens([AddedToken::from("<|im_end|>", true).normalized(false)])
        .unwrap();
    let eos = tokenizer.token_to_id("<|im_end|>").unwrap();
    let mut model = LoadedModel::from_runtime(
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
    );
    let chat = model
        .prepare_chat(ChatTemplateRequest {
            messages: vec![serde_json::json!({"role":"user","content":"token1"})],
            add_generation_prompt: true,
            ..Default::default()
        })
        .unwrap();
    let settings = PreparedChatGenerationSettings {
        overrides: GenerationConfigOverrides {
            max_new_tokens: Some(3),
            ..Default::default()
        },
        seed: 19,
    };
    let limits = TraceLimits {
        per_record_bytes: 65536,
        total_bytes: 1024 * 1024,
    };
    let baseline = model
        .generate_prepared_chat(PreparedChatGenerationRequest {
            input: PreparedChatInput::rendered_prompt(&chat),
            settings,
            caller_stop_sequences: &[],
            cancellation: Default::default(),
            on_event: |_| {},
        })
        .unwrap();
    let prepared = model
        .prepare_intervened_chat(&chat, settings, plan(), intervention_plan(1.0), limits)
        .unwrap();
    let identity = prepared.intervention_plan().unwrap().identity().to_owned();
    let mut predictions = Vec::new();
    let output = model
        .generate_observed_chat(prepared, &[], Default::default(), |record| {
            assert_eq!(
                record.intervention_plan_id.as_deref(),
                Some(identity.as_str())
            );
            if let Event::Token {
                captures: Some(step),
                ..
            } = record.event
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
            ControlFlow::Continue(())
        })
        .unwrap();
    assert_eq!(output, baseline);
    assert_eq!(predictions, [0, 1, 2]);
    let empty = model
        .prepare_intervened_chat(&chat, settings, plan(), InterventionPlan::none(), limits)
        .unwrap();
    assert!(empty.intervention_plan().is_none());
    let unchanged = model
        .generate_observed_chat(empty, &[], Default::default(), |record| {
            assert!(record.intervention_plan_id.is_none());
            if let Event::Token {
                captures: Some(step),
                ..
            } = record.event
            {
                assert!(step.interventions.is_empty());
                assert_eq!(step.records.len(), 1);
            }
            ControlFlow::Continue(())
        })
        .unwrap();
    assert_eq!(unchanged, baseline);
    let mut invalid = intervention_plan(1.0);
    invalid.operations[0].target = "nonexistent".into();
    assert!(model
        .prepare_intervened_chat(&chat, settings, plan(), invalid, limits)
        .is_err());
    let prepared = model
        .prepare_intervened_chat(&chat, settings, plan(), intervention_plan(13.0), limits)
        .unwrap();
    let mut records = Vec::new();
    assert!(model
        .generate_observed_chat(prepared, &[], Default::default(), |r| {
            records.push(r);
            ControlFlow::Continue(())
        })
        .is_err());
    assert!(records.iter().any(
        |r| matches!(&r.event, Event::CaptureFailure { captures, .. }
        if matches!(captures.interventions[0].outcome, InterventionOutcome::Failed { .. }))
    ));
    assert!(matches!(
        records.last().unwrap().event,
        Event::Failed { .. }
    ));
    for pre_cancel in [false, true] {
        let prepared = model
            .prepare_intervened_chat(&chat, settings, plan(), intervention_plan(1.0), limits)
            .unwrap();
        let cancellation = eredu_core::GenerationCancellationToken::new();
        if pre_cancel {
            cancellation.cancel();
        }
        let mut closed = false;
        let output = model
            .generate_observed_chat(prepared, &[], cancellation, |record| {
                assert!(!closed, "delivery after consumer Break");
                if matches!(record.event, Event::Token { .. }) {
                    closed = true;
                    ControlFlow::Break(())
                } else {
                    ControlFlow::Continue(())
                }
            })
            .unwrap();
        assert_eq!(output.finish_reason, FinishReason::Cancelled);
        assert_eq!(output.token_ids.len(), usize::from(!pre_cancel));
    }
    let prepared = model
        .prepare_intervened_chat(&chat, settings, plan(), intervention_plan(1.0), limits)
        .unwrap();
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = model.generate_observed_chat(prepared, &[], Default::default(), |record| {
            if matches!(record.event, Event::Token { .. }) {
                panic!("consumer unwinds");
            }
            ControlFlow::Continue(())
        });
    }))
    .is_err());
    // Successful reuse proves no move-only submission lease escaped cancellation,
    // failure, or consumer unwinding in this synchronous conformance backend.
    let prepared = model
        .prepare_intervened_chat(&chat, settings, plan(), intervention_plan(1.0), limits)
        .unwrap();
    assert_eq!(
        model
            .generate_observed_chat(
                prepared,
                &[],
                Default::default(),
                |_| ControlFlow::Continue(())
            )
            .unwrap(),
        baseline
    );
    let prepared = model
        .prepare_intervened_chat(&chat, settings, plan(), intervention_plan(0.0), limits)
        .unwrap();
    assert_eq!(
        model
            .generate_observed_chat(
                prepared,
                &[],
                Default::default(),
                |_| ControlFlow::Continue(())
            )
            .unwrap()
            .token_ids,
        [0, 0, 0]
    );
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
