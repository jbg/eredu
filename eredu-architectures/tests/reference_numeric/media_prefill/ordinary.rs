use super::*;
use eredu_architectures::prepared_execution::*;

pub(super) struct Run {
    pub(super) outputs: Vec<NumericTensor>,
    pub(super) state: State,
    pub(super) media: Option<Report>,
}
struct Assembler;
impl PreparedExecutableAssembler<()> for Assembler {
    type Executable = Run;
    type Output = Run;
    type Error = String;
    fn floating_state_dtype(
        &mut self,
        _: &eredu_architectures::preparation::FloatingStateDtypeSource,
    ) -> Result<eredu_runtime::StateStorageDtype, String> {
        Ok(eredu_runtime::StateStorageDtype::F32)
    }
    fn validate_communication(&mut self, _: &CommunicationManifest, _: &()) -> Result<(), String> {
        Err("local media fixture has no communication".into())
    }
    fn finish(self, mut parts: PreparedExecutableParts<Run, ()>) -> Result<Run, String> {
        assert!(parts.take_communication().is_none());
        assert!(parts.take_processor().is_none());
        assert_eq!(parts.floating_state_bytes().get(), 4);
        Ok(parts.into_executable())
    }
}
struct Visitor<'a> {
    demand: OutputDemand,
    context: &'a NumericContext,
    input: &'a Input,
    schedule: Option<Schedule>,
    negative: bool,
    started: bool,
    external_source: bool,
    prediction_source: bool,
}
impl CompositeTextArchitectureVisitor<NumericBackend, State> for Visitor<'_> {
    type Output = Run;
    type Error = String;
    fn construction_started(&mut self) {
        self.started = true;
    }
    fn visit<A>(
        self,
        _: PreparedCompositeTextArchitecture<A, A::AdmissionConfig>,
        _: RetainedCheckpointSource,
    ) -> Result<Run, String>
    where
        A: CompositeArchitecture<NumericBackend, State, Error = Error>
            + eredu_runtime::RoutedLayeredArchitecture<NumericBackend, State>
            + 'static,
        A::InputPartPlan: 'static,
        A::StaticModules: Clone,
    {
        Err("selected media fixture lost its semantic capability".into())
    }
    fn visit_routed<A>(
        self,
        _: eredu_architectures::replicated_text::PreparedRoutedCompositeTextArchitecture<
            A,
            A::AdmissionConfig,
        >,
        _: RetainedCheckpointSource,
    ) -> Result<Run, String>
    where
        A: CompositeArchitecture<NumericBackend, State, Error = Error>
            + eredu_runtime::RoutedLayeredArchitecture<NumericBackend, State>
            + 'static,
        A::InputPartPlan: 'static,
        A::StaticModules: Clone,
    {
        Err("dense fixture unexpectedly selected routed execution".into())
    }
    fn visit_media<A>(
        self,
        prepared: PreparedCompositeTextArchitecture<A, A::AdmissionConfig>,
        checkpoint: RetainedCheckpointSource,
    ) -> Result<Run, String>
    where
        A: CompositeMediaIngressArchitecture<NumericBackend, State, Error = Error>
            + eredu_runtime::RoutedLayeredArchitecture<NumericBackend, State>
            + 'static,
        A::InputPartPlan: 'static,
        A::StaticModules: Clone,
    {
        assert!(self.started);
        let capability = prepared.capability_estimate().clone();
        let mechanisms = if matches!(
            prepared.selected().residency(),
            LayerWeightResidency::FullyResident
        ) {
            NumericReplicatedMechanisms::with_bound_checkpoint(checkpoint)
        } else {
            NumericReplicatedMechanisms::with_bounded_checkpoint(checkpoint)
        };
        let (mut session, facts) = construct_selected_composite_session::<NumericBackend, _, _, _>(
            prepared,
            mechanisms,
            self.context,
        )?;
        let (_, _, admission) = facts.into_parts();
        if self.negative {
            negatives::<A, _>(
                &mut session,
                &admission,
                &capability,
                self.input,
                self.context,
            )
            .map_err(|e| e.to_string())?;
            return Ok(Run {
                outputs: Vec::new(),
                state: snapshot::<A, _>(&session).map_err(|e| e.to_string())?,
                media: None,
            });
        }
        if let Some(schedule) = self.schedule {
            let report =
                scheduled::<A, _>(&mut session, &admission, self.input, self.context, schedule)
                    .map_err(|e| e.to_string())?;
            let mut outputs = report.output.clone().into_iter().collect::<Vec<_>>();
            outputs.extend(report.cached.iter().cloned());
            let state = report.final_state.clone();
            return Ok(Run {
                outputs,
                state,
                media: Some(report),
            });
        }
        let admitted = A::admit_prepared_input(&admission, self.input, &NumericInputInspector)
            .map_err(|e| e.to_string())?;
        let mut observer = Observer(Rc::new(RefCell::new(Trace::default())));
        let first =
            if self.prediction_source {
                use eredu_architectures::speculative_execution::{
                    AdmittedPredictionPrefill, PredictionPrefillPlan, PredictionPrefillSource,
                };
                use eredu_runtime::replicated_session::PreparedPrefillSource;
                let [batch_size, input_positions] = admitted.decoder_shape();
                let geometry = eredu_core::InferenceGeometry {
                    batch_size,
                    input_positions,
                    cached_positions: 0,
                    max_output_tokens: 0,
                    prefill_chunk_positions: input_positions,
                    output: self.demand,
                };
                let plan = AdmittedPredictionPrefill::new(
                    self.input.clone(),
                    admitted.clone(),
                    admission.clone(),
                    NumericInputInspector,
                    None,
                    None,
                );
                assert!(<_ as PredictionPrefillPlan<
                    PreparedCompositeArchitecture<A>,
                    NumericBackend,
                    State,
                >>::requires_whole_input(&plan));
                let source = <_ as PredictionPrefillPlan<
                    PreparedCompositeArchitecture<A>,
                    NumericBackend,
                    State,
                >>::into_source(plan, geometry)
                .map_err(|e| e.to_string())?
                .unwrap();
                let chunk = <_ as PreparedPrefillSource<
                    PreparedCompositeArchitecture<A>,
                    NumericBackend,
                    State,
                >>::prepare_chunk(
                    &source,
                    &PrefillChunk {
                        input: 0..input_positions,
                        position: 0,
                        output: self.demand,
                    },
                    self.context,
                )
                .map_err(|e| e.to_string())?;
                let tokens = <_ as PredictionPrefillSource<
                    PreparedCompositeArchitecture<A>,
                    NumericBackend,
                    State,
                >>::tokens(&source, &chunk);
                let paired = PreparedCompositeInput::new(self.input, &admitted)?;
                let expected_tokens = A::prepared_prediction_token_ids(paired, self.context)
                    .map_err(|e| e.to_string())?;
                assert_eq!(
                    tokens.shape,
                    vec![batch_size as i32, input_positions as i32]
                );
                assert_tensor_close(
                    tokens,
                    &expected_tokens,
                    "actual admitted semantic media placeholders",
                );
                drop(chunk);
                eredu_runtime::replicated_session::PrefillSourceOutcome::Complete(
                    run_whole_source::<A, _, _>(
                        &mut session,
                        source,
                        geometry,
                        self.context,
                        &mut observer,
                    )?,
                )
            } else if self.external_source {
                use eredu_architectures::prefill::PreparedExternalPrefill;
                use eredu_runtime::replicated_session::PreparedPrefillSource;
                let [batch_size, input_positions] = admitted.decoder_shape();
                let geometry = eredu_core::InferenceGeometry {
                    batch_size,
                    input_positions,
                    cached_positions: 0,
                    max_output_tokens: 0,
                    prefill_chunk_positions: input_positions,
                    output: self.demand,
                };
                let source = PreparedExternalPrefill::from_prepared(
                    self.input.clone(),
                    admitted.clone(),
                    geometry,
                    admission.clone(),
                    NumericInputInspector,
                    true,
                )
                .map_err(|e| e.to_string())?
                .unwrap();
                let before = self.context.projections.lock().unwrap().clone();
                for bad in [
                    PrefillChunk {
                        input: 1..input_positions,
                        position: 0,
                        output: self.demand,
                    },
                    PrefillChunk {
                        input: 0..input_positions,
                        position: 1,
                        output: self.demand,
                    },
                    PrefillChunk {
                        input: 0..input_positions,
                        position: 0,
                        output: if self.demand == OutputDemand::Sequence {
                            OutputDemand::LastPosition
                        } else {
                            OutputDemand::Sequence
                        },
                    },
                ] {
                    assert!(<_ as PreparedPrefillSource<
                        PreparedCompositeArchitecture<A>,
                        NumericBackend,
                        State,
                    >>::prepare_chunk(&source, &bad, self.context,)
                    .is_err());
                }
                assert!(PreparedExternalPrefill::from_prepared(
                    self.input.clone(),
                    admitted.clone(),
                    eredu_core::InferenceGeometry {
                        prefill_chunk_positions: 1,
                        ..geometry
                    },
                    admission.clone(),
                    NumericInputInspector,
                    true,
                )
                .is_err());
                assert_eq!(*self.context.projections.lock().unwrap(), before);
                eredu_runtime::replicated_session::PrefillSourceOutcome::Complete(
                    run_whole_source::<A, _, _>(
                        &mut session,
                        source,
                        geometry,
                        self.context,
                        &mut observer,
                    )?,
                )
            } else {
                let paired = PreparedCompositeInput::new(self.input, &admitted)?;
                eredu_runtime::replicated_session::PrefillSourceOutcome::Complete(
                    session
                        .prefill_input_with_readout(
                            paired,
                            None,
                            self.demand,
                            self.context,
                            &mut observer,
                        )
                        .map_err(|e| e.to_string())?,
                )
            };
        let eredu_runtime::replicated_session::PrefillSourceOutcome::Complete(Some(first)) = first
        else {
            return Err("reference requests completed scores".into());
        };
        let mut outputs = vec![first];
        for token in [2, 6, 1] {
            outputs.push(
                decode::<A, _>(&mut session, &admission, token, self.context)
                    .map_err(|e| e.to_string())?,
            );
        }
        Ok(Run {
            outputs,
            state: snapshot::<A, _>(&session).map_err(|e| e.to_string())?,
            media: None,
        })
    }
}

struct RequiredObserver;
impl eredu_runtime::ActivationObserver<NumericTensor, Error> for RequiredObserver {
    fn requires_sequence_readout(&self) -> bool {
        true
    }
    fn observe(&mut self, _: &str, _: &NumericTensor) -> Result<(), Error> {
        panic!("rejected observation must not run")
    }
}
fn negatives<A, D>(
    session: &mut Session<A, D>,
    admission: &A::AdmissionConfig,
    capability: &eredu_architectures::capability::CapabilityEstimate,
    input: &Input,
    context: &NumericContext,
) -> Result<(), Error>
where
    A: CompositeMediaIngressArchitecture<NumericBackend, State, Error = Error> + 'static,
    A::InputPartPlan: 'static,
    D: MediaTextExecutionStrategy<
        PreparedCompositeArchitecture<A>,
        NumericBackend,
        State,
        NumericReplicatedPolicy<A::Unit>,
        NumericReplicatedPolicy<A::Unit>,
    >,
{
    let geometry = eredu_core::InferenceGeometry {
        batch_size: 1,
        cached_positions: 0,
        input_positions: 7,
        max_output_tokens: 3,
        prefill_chunk_positions: 2,
        output: OutputDemand::LastPosition,
    };
    let make_plan =
        || A::prepare_ingress_plan(admission, input.clone(), &NumericInputInspector, geometry);
    let mut source = session
        .prepare_media_prefill_unbudgeted(make_plan()?)
        .map_err(|e| Error::backend(e.to_string()))?;
    let before = snapshot::<A, D>(session)?;
    let projections = context.projections.lock().unwrap().clone();
    let mechanisms = context.mechanism_trace();
    let roots = context.media_completions.lock().unwrap().clone();
    // This is the genuine existing scalar text admission, deliberately not a
    // media quote. Both forms must reject before preparing or executing media.
    for converted in [false, true] {
        use eredu_runtime::working_memory::{
            InferenceRequest, WorkingMemoryError, WorkingMemoryPool,
        };
        let pool = WorkingMemoryPool::new(1 << 30, 0).unwrap();
        let reservation = pool
            .reserve(
                session.inference_execution_identity(),
                &super::super::bounded_readout::scalar_session_admission(capability, geometry),
            )
            .unwrap();
        let (reservation, run) = if converted {
            let (reservation, run) = reservation.into_funding().unwrap();
            (reservation, Some(run))
        } else {
            (reservation, None)
        };
        let request: InferenceRequest = reservation.into();
        assert!(request.memory_reservation().is_some());
        assert_eq!(request.requires_funding_scope(), converted);
        let mut called = false;
        let mut observer = Observer(Rc::new(RefCell::new(Trace::default())));
        let rejected = session.try_prefill_media_source_cancellable(
            Some(&request),
            Some([1, 7]),
            None,
            std::num::NonZeroU64::new(2),
            |_| {
                called = true;
                make_plan()
            },
            &eredu_core::GenerationCancellationToken::new(),
            context,
            &mut observer,
        );
        match rejected {
            Err(eredu_runtime::ReplicatedTextSessionError::BeforeStateMutation(error)) => {
                assert!(matches!(
                    *error,
                    eredu_runtime::ReplicatedTextSessionError::WorkingMemory(
                        WorkingMemoryError::UnknownBound
                    )
                ));
            }
            Err(other) => panic!("wrong reserved media rejection: {other}"),
            Ok(_) => panic!("reserved media must reject before its factory"),
        }
        assert!(!called);
        same_state(&snapshot::<A, D>(session)?, &before);
        assert_eq!(*context.projections.lock().unwrap(), projections);
        assert_eq!(context.mechanism_trace(), mechanisms);
        assert_eq!(*context.media_completions.lock().unwrap(), roots);
        drop((request, run));
    }
    {
        let request = source.request().clone();
        let execution = session.inference_execution_identity().clone();
        let cancellation = eredu_core::GenerationCancellationToken::new();
        cancellation.cancel();
        let mut observer = Observer(Rc::new(RefCell::new(Trace::default())));
        let mut executor = SessionPrefill::new_media(session, &mut source, context, &mut observer)
            .map_err(|e| Error::backend(e.to_string()))?;
        let mut driver = PrefillDriver::new(&execution, request, geometry, cancellation)
            .map_err(|e| Error::backend(e.to_string()))?;
        assert!(matches!(
            driver
                .step(&mut executor)
                .map_err(|e| Error::backend(e.to_string()))?,
            PrefillProgress::Cancelled
        ));
        assert_eq!(*context.projections.lock().unwrap(), projections);
        assert_eq!(context.mechanism_trace(), mechanisms);
        assert_eq!(*context.media_completions.lock().unwrap(), roots);
    }
    assert!(
        source
            .request()
            .validate(
                &eredu_runtime::working_memory::InferenceExecutionIdentity::default(),
                geometry
            )
            .is_err(),
        "exact source request rejects a foreign execution identity"
    );
    assert!(
        SessionPrefill::new_media(session, &mut source, context, &mut RequiredObserver).is_err()
    );
    let checkpoint = session
        .checkpoint_complete(context)
        .map_err(|e| Error::backend(e.to_string()))?;
    session
        .rollback_complete(checkpoint, context)
        .map_err(|e| Error::backend(e.to_string()))?;
    let mut observer = Observer(Rc::new(RefCell::new(Trace::default())));
    assert!(
        SessionPrefill::new_media(session, &mut source, context, &mut observer).is_err(),
        "restored revision cannot reuse the old source"
    );
    same_state(&snapshot::<A, D>(session)?, &before);
    assert_eq!(*context.projections.lock().unwrap(), projections);
    assert_eq!(context.mechanism_trace(), mechanisms);
    assert_eq!(*context.media_completions.lock().unwrap(), roots);
    drop(source);
    let mut source = session
        .prepare_media_prefill_unbudgeted(make_plan()?)
        .map_err(|e| Error::backend(e.to_string()))?;
    let request = source.request().clone();
    let execution = session.inference_execution_identity().clone();
    let mut observer = Observer(Rc::new(RefCell::new(Trace::default())));
    let mut executor = SessionPrefill::new_media(session, &mut source, context, &mut observer)
        .map_err(|e| Error::backend(e.to_string()))?;
    let mut driver = PrefillDriver::new(
        &execution,
        request,
        geometry,
        eredu_core::GenerationCancellationToken::new(),
    )
    .map_err(|e| Error::backend(e.to_string()))?;
    context
        .fail_media_completion
        .store(true, std::sync::atomic::Ordering::SeqCst);
    assert!(
        driver.step(&mut executor).is_err(),
        "future media completion failure rejects the first span"
    );
    let failed_projections = context.projections.lock().unwrap().clone();
    let failed_roots = context.media_completions.lock().unwrap().clone();
    assert!(driver.step(&mut executor).is_err());
    assert_eq!(*context.projections.lock().unwrap(), failed_projections);
    assert_eq!(*context.media_completions.lock().unwrap(), failed_roots);
    drop(executor);
    same_state(&snapshot::<A, D>(session)?, &before);
    let mut retained = Vec::new();
    source.visit_retained_roots(&mut |v| retained.push((v.shape.clone(), v.data.clone())));
    assert!(
        retained.iter().any(|(shape, values)| shape.len() == 3
            && shape[1] == 4
            && values.iter().any(|v| v.abs() > 1e-9)),
        "failed source still owns actual future-span outputs"
    );
    Ok(())
}

fn run(
    config: &serde_json::Value,
    artifact: &tempfile::TempDir,
    residency: eredu_core::ResidencyPlan,
    schedule: Option<Schedule>,
    negative: bool,
) -> Run {
    run_with_demand(
        config,
        artifact,
        residency,
        schedule,
        negative,
        OutputDemand::LastPosition,
    )
}

fn run_with_demand(
    config: &serde_json::Value,
    artifact: &tempfile::TempDir,
    residency: eredu_core::ResidencyPlan,
    schedule: Option<Schedule>,
    negative: bool,
    demand: OutputDemand,
) -> Run {
    run_qwen_with_input(
        config,
        artifact,
        residency,
        schedule,
        negative,
        demand,
        selected::input(),
    )
}

fn run_qwen_with_input(
    config: &serde_json::Value,
    artifact: &tempfile::TempDir,
    residency: eredu_core::ResidencyPlan,
    schedule: Option<Schedule>,
    negative: bool,
    demand: OutputDemand,
    input: Input,
) -> Run {
    let result = run_with_input(
        config, artifact, residency, schedule, negative, demand, input,
    );
    if let Some(report) = &result.media {
        selected::assert_trace(
            report,
            schedule.unwrap(),
            true,
            true,
            config["text_config"]["hidden_size"].as_i64().unwrap() as i32,
        );
    }
    result
}

pub(super) fn run_with_input(
    config: &serde_json::Value,
    artifact: &tempfile::TempDir,
    residency: eredu_core::ResidencyPlan,
    schedule: Option<Schedule>,
    negative: bool,
    demand: OutputDemand,
    input: Input,
) -> Run {
    run_input(
        config, artifact, residency, schedule, negative, demand, input, false, false,
    )
}

pub(super) fn run_external_with_input(
    config: &serde_json::Value,
    artifact: &tempfile::TempDir,
    residency: eredu_core::ResidencyPlan,
    demand: OutputDemand,
    input: Input,
) -> Run {
    run_input(
        config, artifact, residency, None, false, demand, input, true, false,
    )
}

fn run_input(
    config: &serde_json::Value,
    artifact: &tempfile::TempDir,
    residency: eredu_core::ResidencyPlan,
    schedule: Option<Schedule>,
    negative: bool,
    demand: OutputDemand,
    input: Input,
    external_source: bool,
    prediction_source: bool,
) -> Run {
    let inspection = eredu_architectures::configuration::inspect_artifact(artifact.path()).unwrap();
    let plan = prepared_adapter::plan(None).with_residency(residency);
    let sources = prepared_adapter::prepare(
        &inspection,
        &plan,
        &prepared_adapter::NumericPreparationProvider { addressable: false },
    )
    .unwrap();
    let context = NumericContext {
        bind_checkpoint_values: true,
        ..Default::default()
    };
    let routes =
        PreparedExecutionRoutes::new().with_composite(
            CompositeRoute::<NumericBackend, State, _>::new(
                &context,
                &context,
                Visitor {
                    demand,
                    context: &context,
                    input: &input,
                    schedule,
                    negative,
                    started: false,
                    external_source,
                    prediction_source,
                },
            ),
        );
    construct_prepared_execution(sources, None, routes, Assembler).unwrap()
}
// Independent local execution has no decoder-to-decoder transport. Compare
// the one-span report with direct ordinary forward before lending its state
// checkpoints to the pipeline regression.
pub(super) fn independent_full_report(
    config: &serde_json::Value,
    artifact: &tempfile::TempDir,
) -> Report {
    let ordinary = run(
        config,
        artifact,
        eredu_core::ResidencyPlan::FullyResident,
        None,
        false,
    );
    let full = run(
        config,
        artifact,
        eredu_core::ResidencyPlan::FullyResident,
        Some(Schedule {
            output: OutputDemand::LastPosition,
            chunk: 7,
            stepped: false,
            cancel_after: None,
            locally_cancel: false,
            follow_decode: true,
        }),
        false,
    );
    assert_eq!(ordinary.outputs.len(), 4);
    assert_eq!(full.outputs.len(), ordinary.outputs.len());
    for (actual, expected) in full.outputs.iter().zip(&ordinary.outputs) {
        assert_tensor_close(
            actual,
            expected,
            "independent ordinary full media reference",
        );
    }
    same_state(&full.state, &ordinary.state);
    full.media.unwrap()
}

#[test]
fn selected_local_media_matches_full_prefill_and_cached_restore_in_all_residencies() {
    for hybrid in [false, true] {
        let config = selected::configuration(hybrid);
        let artifact = selected::fixture(&config);
        let reference = run(
            &config,
            &artifact,
            eredu_core::ResidencyPlan::FullyResident,
            None,
            false,
        );
        for residency in selected::residencies() {
            for (chunk, stepped) in [1, 2, 3, 7]
                .into_iter()
                .flat_map(|c| [false, true].map(|s| (c, s)))
            {
                let actual = run(
                    &config,
                    &artifact,
                    residency.clone(),
                    Some(Schedule {
                        output: OutputDemand::LastPosition,
                        chunk,
                        stepped,
                        cancel_after: None,
                        locally_cancel: false,
                        follow_decode: true,
                    }),
                    false,
                );
                for (actual, expected) in actual.outputs.iter().zip(&reference.outputs) {
                    nonzero(actual);
                    assert_tensor_close(actual, expected, "local selected media full/span parity");
                }
                assert_eq!(actual.outputs.len(), 4);
                same_state(&actual.state, &reference.state);
            }
        }
    }
    selected_local_media_full_sequence_spans_preserve_every_vocab_row_and_cached_state();
    selected_media_five_position_cut_preserves_ordered_image_video_and_projected_semantics();
}
#[test]
fn selected_media_rejects_required_observation_and_stale_source_and_retains_failed_future_roots() {
    for hybrid in [false, true] {
        let config = selected::configuration(hybrid);
        let artifact = selected::fixture(&config);
        for residency in selected::residencies() {
            run(&config, &artifact, residency, None, true);
        }
    }
}

fn selected_local_media_full_sequence_spans_preserve_every_vocab_row_and_cached_state() {
    for hybrid in [false, true] {
        let config = selected::configuration(hybrid);
        let artifact = selected::fixture(&config);
        let reference = run_with_demand(
            &config,
            &artifact,
            eredu_core::ResidencyPlan::FullyResident,
            None,
            false,
            OutputDemand::Sequence,
        );
        assert_eq!(reference.outputs[0].shape, [1, 7, 7]);
        for residency in selected::residencies() {
            for stepped in [false, true] {
                let actual = run_with_demand(
                    &config,
                    &artifact,
                    residency.clone(),
                    Some(Schedule {
                        output: OutputDemand::Sequence,
                        chunk: 2,
                        stepped,
                        cancel_after: None,
                        locally_cancel: false,
                        follow_decode: true,
                    }),
                    false,
                    OutputDemand::Sequence,
                );
                assert_eq!(actual.outputs[0].shape, [1, 7, 7]);
                for (a, b) in actual.outputs.iter().zip(&reference.outputs) {
                    nonzero(a);
                    assert_tensor_close(a, b, "all media vocabulary rows and cached outputs");
                }
                same_state(&actual.state, &reference.state);
            }
        }
    }
}

fn mixed_five_input(hidden: i32) -> Input {
    use eredu_core::{InputExtent, InputMetadataKey, InputModality};
    use eredu_runtime::{PreparedInputPart, PreparedInputPayload};
    let raw = |modality, sign: f32| {
        PreparedInputPart::new_with_extents(
            modality,
            PreparedInputPayload::Tensor(NumericTensor::new(
                [4, 24],
                (0..96)
                    .map(|i| sign * ((i % 19) as f32 - 9.) / 70.)
                    .collect(),
            )),
            [(
                InputMetadataKey::PatchGrid,
                NumericTensor::new([1, 3], vec![1., 2., 2.]),
            )],
            [InputExtent::PatchGrid {
                time: 1,
                height: 2,
                width: 2,
            }],
        )
        .unwrap()
    };
    Input::new(
        vec![
            PreparedInputPart::new(
                InputModality::Text,
                PreparedInputPayload::TokenIds(NumericTensor::token_ids(&[1])),
                [],
            )
            .unwrap(),
            PreparedInputPart::new(
                InputModality::Text,
                PreparedInputPayload::Embeddings(NumericTensor::new(
                    [1, 1, hidden],
                    (0..hidden).map(|i| (i as f32 + 1.) / 50.).collect(),
                )),
                [],
            )
            .unwrap(),
            raw(InputModality::Image, 1.),
            raw(InputModality::Video, -1.),
            PreparedInputPart::new(
                InputModality::Text,
                PreparedInputPayload::TokenIds(NumericTensor::token_ids(&[4])),
                [],
            )
            .unwrap(),
        ],
        |tensor| eredu_runtime::PreparedInputInspector::identity(&NumericInputInspector, tensor),
    )
    .unwrap()
}
fn selected_media_five_position_cut_preserves_ordered_image_video_and_projected_semantics() {
    for hybrid in [false, true] {
        let config = selected::configuration(hybrid);
        let artifact = selected::fixture(&config);
        let input = mixed_five_input(config["text_config"]["hidden_size"].as_i64().unwrap() as i32);
        for output in [OutputDemand::LastPosition, OutputDemand::Sequence] {
            let reference = run_qwen_with_input(
                &config,
                &artifact,
                eredu_core::ResidencyPlan::FullyResident,
                None,
                false,
                output,
                input.clone(),
            );
            for residency in selected::residencies() {
                for (chunk, stepped) in [1, 2, 3, 5]
                    .into_iter()
                    .flat_map(|c| [false, true].map(|s| (c, s)))
                {
                    let actual = run_qwen_with_input(
                        &config,
                        &artifact,
                        residency.clone(),
                        Some(Schedule {
                            output,
                            chunk,
                            stepped,
                            cancel_after: None,
                            locally_cancel: false,
                            follow_decode: true,
                        }),
                        false,
                        output,
                        input.clone(),
                    );
                    let report = actual.media.as_ref().unwrap();
                    if chunk == 2 {
                        assert_eq!(
                            report
                                .trace
                                .delivered
                                .iter()
                                .map(|s| s.input.clone())
                                .collect::<Vec<_>>(),
                            [0..2, 2..4, 4..5]
                        );
                    }
                    for (a, b) in actual.outputs.iter().zip(&reference.outputs) {
                        nonzero(a);
                        assert_tensor_close(a, b, "ordered image/video/projected mixed input");
                    }
                    same_state(&actual.state, &reference.state);
                }
            }
        }
    }
}

#[test]
fn pending_encoder_assembly_preserves_multirow_image_video_and_projected_order() {
    use eredu_core::{InputMetadataKey, InputModality};
    use eredu_runtime::{PreparedInputPart, PreparedInputPayload};
    for hybrid in [false, true] {
        let config = selected::configuration(hybrid);
        let artifact = selected::fixture(&config);
        let hidden = config["text_config"]["hidden_size"].as_i64().unwrap() as i32;
        let mut parts = mixed_five_input(hidden).into_parts();
        let raw = |modality, grid: Vec<f32>, sign: f32| {
            // The PatchGrid tensor retains every ordered row. The optional
            // single-grid extent cannot describe a multirow image part.
            PreparedInputPart::new(
                modality,
                PreparedInputPayload::Tensor(NumericTensor::new(
                    [8, 24],
                    (0..192)
                        .map(|i| sign * ((i % 23) as f32 - 11.) / 75.)
                        .collect(),
                )),
                [(
                    InputMetadataKey::PatchGrid,
                    NumericTensor::new([grid.len() as i32 / 3, 3], grid),
                )],
            )
            .unwrap()
        };
        parts[2] = raw(InputModality::Image, vec![1., 2., 2., 1., 2., 2.], 1.);
        parts[3] = raw(InputModality::Video, vec![2., 2., 2.], -1.);
        // Raw-image first exercises the actual admitted decoder batch, and the
        // two image rows must remain one ordered semantic part in pending input.
        parts.swap(0, 2);
        let input = Input::new(parts, |tensor| {
            eredu_runtime::PreparedInputInspector::identity(&NumericInputInspector, tensor)
        })
        .unwrap();
        let reference = run_qwen_with_input(
            &config,
            &artifact,
            eredu_core::ResidencyPlan::FullyResident,
            None,
            false,
            OutputDemand::Sequence,
            input.clone(),
        );
        assert_eq!(reference.outputs[0].shape, [1, 7, 7]);
        for residency in selected::residencies() {
            for stepped in [false, true] {
                let actual = run_qwen_with_input(
                    &config,
                    &artifact,
                    residency.clone(),
                    Some(Schedule {
                        output: OutputDemand::Sequence,
                        chunk: 2,
                        stepped,
                        cancel_after: None,
                        locally_cancel: false,
                        follow_decode: true,
                    }),
                    false,
                    OutputDemand::Sequence,
                    input.clone(),
                );
                assert_eq!(actual.outputs.len(), reference.outputs.len());
                assert_eq!(
                    actual
                        .media
                        .as_ref()
                        .unwrap()
                        .trace
                        .delivered
                        .iter()
                        .map(|s| s.input.clone())
                        .collect::<Vec<_>>(),
                    [0..2, 2..4, 4..6, 6..7]
                );
                for (a, b) in actual.outputs.iter().zip(&reference.outputs) {
                    nonzero(a);
                    assert_tensor_close(
                        a,
                        b,
                        "multirow pending assembly versus direct valid PreparedInput",
                    );
                }
                same_state(&actual.state, &reference.state);
            }
        }
    }
}

// Same shared driver for both genuine whole-input adapters; ordinary reference
// forward remains above and never passes through this helper.
fn run_whole_source<A, D, Source>(
    session: &mut Session<A, D>,
    source: Source,
    geometry: eredu_core::InferenceGeometry,
    context: &NumericContext,
    observer: &mut Observer,
) -> Result<Option<NumericTensor>, String>
where
    A: CompositeArchitecture<NumericBackend, State, Error = Error> + 'static,
    A::InputPartPlan: 'static,
    D: eredu_runtime::ReplicatedTextExecutionStrategy<
        PreparedCompositeArchitecture<A>,
        NumericBackend,
        State,
        NumericReplicatedPolicy<A::Unit>,
        NumericReplicatedPolicy<A::Unit>,
    >,
    Source: eredu_runtime::replicated_session::PreparedPrefillSource<
        PreparedCompositeArchitecture<A>,
        NumericBackend,
        State,
    >,
{
    let execution = session.inference_execution_identity().clone();
    let request = eredu_runtime::working_memory::InferenceRequest::without_memory_budget(
        &execution, geometry,
    )
    .map_err(|e| e.to_string())?;
    let mut executor = SessionPrefill::new(session, source, &request, context, observer)
        .map_err(|e| e.to_string())?;
    let mut driver = PrefillDriver::new(
        &execution,
        &request,
        geometry,
        eredu_core::GenerationCancellationToken::new(),
    )
    .map_err(|e| e.to_string())?;
    let (outcome, output) = driver.run_final(&mut executor).map_err(|e| e.to_string())?;
    assert_eq!(outcome, PrefillOutcome::Complete);
    assert_eq!(driver.completed_positions(), geometry.input_positions);
    Ok(output)
}

#[test]
fn admitted_prediction_media_source_preserves_semantic_tokens_and_nonzero_state() {
    let config = selected::configuration(true);
    let artifact = selected::fixture(&config);
    for residency in selected::residencies() {
        for demand in [OutputDemand::LastPosition, OutputDemand::Sequence] {
            let expected = run_with_input(
                &config,
                &artifact,
                residency.clone(),
                None,
                false,
                demand,
                selected::input(),
            );
            let actual = run_input(
                &config,
                &artifact,
                residency.clone(),
                None,
                false,
                demand,
                selected::input(),
                false,
                true,
            );
            assert_eq!(actual.outputs.len(), 4);
            for (actual, expected) in actual.outputs.iter().zip(&expected.outputs) {
                assert!(actual.data.iter().any(|x| x.abs() > 1e-8));
                assert_tensor_close(actual, expected, "admitted prediction whole media source");
            }
            same_state(&actual.state, &expected.state);
        }
    }
}
