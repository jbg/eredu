#[path = "bounded_readout/parallel_boundary.rs"]
pub(super) mod parallel_boundary;

use super::*;
use eredu_architectures::readout::execute_readout;
use eredu_core::OutputDemand;

#[test]
fn readout_selects_nonzero_hidden_values_before_projection_and_preserves_axes() {
    let context = NumericContext::default();
    // Two batches with visibly distinct positions; a post-projection slice
    // cannot satisfy the observed projection input shape/count assertions.
    let hidden = NumericTensor::new(
        vec![2, 3, 2],
        vec![
            0.2, 0.7, -0.4, 1.3, 0.9, -0.2, 1.1, 0.3, 0.5, -0.6, 0.8, 1.7,
        ],
    );
    let mut calls = vec![];
    let mut project = |input: &NumericTensor| {
        calls.push(input.shape.clone());
        let mut shape = input.shape.clone();
        *shape.last_mut().unwrap() = 3;
        let mut values = Vec::new();
        for row in input.data.chunks_exact(2) {
            values.extend([
                row[0] + 2.0 * row[1],
                -row[0] + row[1],
                row[0] * 0.5 - row[1],
            ]);
        }
        Ok(NumericTensor::new(shape, values))
    };
    assert!(
        execute_readout(&hidden, OutputDemand::StateOnly, 1, &context, &mut project)
            .unwrap()
            .is_none()
    );
    let all = execute_readout(&hidden, OutputDemand::Sequence, 1, &context, &mut project)
        .unwrap()
        .unwrap();
    let last = execute_readout(
        &hidden,
        OutputDemand::LastPosition,
        1,
        &context,
        &mut project,
    )
    .unwrap()
    .unwrap();
    assert_eq!(last.shape, [2, 1, 3]);
    assert_eq!(last.data, all.axis_slice(1, 2, 3).data);
    assert_eq!(calls, [vec![2, 3, 2], vec![2, 1, 2]]);
}

#[test]
fn readout_uses_declared_sequence_axis_including_multi_stream_hidden() {
    let context = NumericContext::default();
    for (shape, axis) in [(vec![4, 2], 0), (vec![2, 4, 3, 2], 1)] {
        let size: i32 = shape.iter().product();
        let hidden = NumericTensor::new(
            shape.clone(),
            (0..size).map(|x| x as f32 * 0.125 + 0.1).collect(),
        );
        let selected = execute_readout(
            &hidden,
            OutputDemand::LastPosition,
            axis,
            &context,
            |value| Ok(value.clone()),
        )
        .unwrap()
        .unwrap();
        let expected = hidden.axis_slice(axis, shape[axis] as usize - 1, shape[axis] as usize);
        assert_eq!(selected.shape, expected.shape);
        assert_eq!(selected.data, expected.data);
        assert!(execute_readout(
            &hidden,
            OutputDemand::Sequence,
            shape.len(),
            &context,
            |_| panic!("invalid axis must reject before projection")
        )
        .is_err());
    }
}

struct ReadoutTraversal;
impl<C> LayeredTraversalHook<NumericBackend, C, Error> for ReadoutTraversal {}

type ReadoutState = DeviceState<NumericBackend, NumericHybridLayerState>;

// One test driver exercises actual family equations and cache state, including
// recurrent lanes, while recording the inputs to the real projection operators.
fn check_family_readout<A>(name: &str, create: impl Fn(&NumericContext) -> A)
where
    A: eredu_runtime::ReplicatedTextArchitecture<NumericBackend, ReadoutState, Error = Error>
        + eredu_runtime::ArchitectureParameters<NumericBackend, DefinitionError = Error>,
{
    let context = NumericContext::default();
    let architecture = create(&context);
    let layout = architecture.state_layout(None).unwrap();
    let state = || {
        DeviceState::create(layout.clone(), |_, policy| {
            Ok::<_, Error>(NumericHybridLayerState::new(policy))
        })
        .unwrap()
    };
    let tokens = NumericTensor::token_ids(&[1, 3, 2, 4, 5]);
    let mut reference = ResidentRuntime::new(architecture, &context).unwrap();
    let mut reference_state = state();
    let expected = reference
        .forward(A::text_input(&tokens, None), &mut reference_state, &context)
        .unwrap();
    assert!(
        expected.data.iter().any(|value| value.abs() > 1e-6),
        "{name}: nonzero fixture"
    );
    let readout_name = context
        .projections
        .lock()
        .unwrap()
        .last()
        .unwrap()
        .0
        .clone();
    assert_eq!(context.projections.lock().unwrap().last().unwrap().1[1], 5);
    let mut expected_decodes = Vec::new();
    for token in [6, 2, 1] {
        expected_decodes.push(
            reference
                .forward(
                    A::text_input(&NumericTensor::token_ids(&[token]), None),
                    &mut reference_state,
                    &context,
                )
                .unwrap(),
        );
    }
    for streamed in [false, true] {
        for chunk in [1, 2, 3, 5] {
            for demand in [
                OutputDemand::StateOnly,
                OutputDemand::LastPosition,
                OutputDemand::Sequence,
            ] {
                let architecture = create(&context);
                let mut resident = ResidentRuntime::new(architecture, &context).unwrap();
                let mut bounded =
                    LayerwiseRuntime::new(create(&context), RebuildingUnitPolicy::default());
                let mut actual_state = state();
                let mut outputs = Vec::new();
                let mut expected_projection_positions = Vec::new();
                context.projections.lock().unwrap().clear();
                for start in (0..5).step_by(chunk) {
                    let end = (start + chunk).min(5);
                    let input = tokens.axis_slice(1, start, end);
                    let chunk_demand = demand.for_chunk(end == 5);
                    let output = if streamed {
                        bounded
                            .forward_with_traversal_hook_with_readout(
                                A::text_input(&input, None),
                                &mut actual_state,
                                &context,
                                &mut ReadoutTraversal,
                                chunk_demand,
                            )
                            .unwrap()
                            .0
                    } else {
                        resident
                            .forward_with_traversal_hook_with_readout(
                                A::text_input(&input, None),
                                &mut actual_state,
                                &context,
                                &mut ReadoutTraversal,
                                chunk_demand,
                            )
                            .unwrap()
                            .0
                    };
                    if let Some(output) = output {
                        expected_projection_positions
                            .push(chunk_demand.positions((end - start) as u64) as i32);
                        outputs.push(output);
                    } else {
                        assert_eq!(chunk_demand, OutputDemand::StateOnly);
                    }
                }
                let actual_projection_positions = context
                    .projections
                    .lock()
                    .unwrap()
                    .iter()
                    .filter(|(name, _)| name == &readout_name)
                    .map(|(_, shape)| shape[1])
                    .collect::<Vec<_>>();
                assert_eq!(
                    actual_projection_positions, expected_projection_positions,
                    "{name}: projection before selection, streamed={streamed}, chunk={chunk}, demand={demand:?}"
                );
                match demand {
                    OutputDemand::StateOnly => assert!(outputs.is_empty()),
                    OutputDemand::LastPosition => {
                        assert_tensor_close(&outputs[0], &expected.axis_slice(1, 4, 5), name)
                    }
                    OutputDemand::Sequence => assert_tensor_close(
                        &NumericTensor::concatenate(&outputs, 1, &context).unwrap(),
                        &expected,
                        name,
                    ),
                }
                for (token, expected_decode) in [6, 2, 1].into_iter().zip(&expected_decodes) {
                    let input = NumericTensor::token_ids(&[token]);
                    let actual = if streamed {
                        bounded
                            .forward(A::text_input(&input, None), &mut actual_state, &context)
                            .unwrap()
                    } else {
                        resident
                            .forward(A::text_input(&input, None), &mut actual_state, &context)
                            .unwrap()
                    };
                    assert_tensor_close(
                        &actual,
                        expected_decode,
                        &format!("{name}: decode after {demand:?} chunk {chunk}"),
                    );
                }
                assert_state_exact(&actual_state, &reference_state, layout.len(), name);
            }
        }
    }
}

#[test]
fn family_readout_matrix_preserves_nonzero_numerics_and_cached_decode() {
    for tied in [false, true] {
        for model_type in ["llama", "qwen2", "qwen3"] {
            let args = qwen::model_args_from_config_value(&config(model_type, tied));
            if model_type == "llama" {
                let args = llama::model_args_from_config_value(&config(model_type, tied)).unwrap();
                check_family_readout(model_type, |context| {
                    llama::LayeredModel::<NumericBackend>::new(args.clone(), context).unwrap()
                });
            } else {
                let args = args.unwrap();
                check_family_readout(model_type, |context| {
                    qwen::LayeredModel::<NumericBackend>::new(args.clone(), context).unwrap()
                });
            }
        }
    }
    for config in heterogeneous_replicated_configs() {
        let name = config["model_type"].as_str().unwrap();
        match name {
            "lfm2" => {
                let args = lfm2::model_args_from_config_value(&config).unwrap();
                check_family_readout(name, |context| {
                    lfm2::LayeredModel::<NumericBackend>::new(args.clone(), context).unwrap()
                });
            }
            "kimi_linear" => {
                let args = kimi_linear::model_args_from_config_value(&config).unwrap();
                check_family_readout(name, |context| {
                    kimi_linear::LayeredModel::<NumericBackend>::new(args.clone(), context).unwrap()
                });
            }
            "nemotron_h" => {
                let args = nemotron_h::model_args_from_config_value(&config).unwrap();
                check_family_readout(name, |context| {
                    nemotron_h::LayeredModel::<NumericBackend>::new(args.clone(), context).unwrap()
                });
            }
            _ => {
                let args = qwen::hybrid::model_args_from_config_value(&config)
                    .unwrap()
                    .text;
                check_family_readout(name, |context| {
                    qwen::hybrid::LayeredModel::<NumericBackend>::new(args.clone(), context)
                        .unwrap()
                });
            }
        }
    }
}

#[test]
fn ordinary_prepared_sessions_project_one_position() {
    let mut cases = heterogeneous_replicated_configs();
    cases.extend([
        config("llama", true),
        config("qwen3", false),
        nanbeige::tiny_config(false),
        gemma2::tiny_config(),
    ]);
    for config in cases {
        let context = NumericContext::default();
        let tokens = NumericTensor::token_ids(&[1, 3, 2, 4, 5]);
        let run = execute_numeric_replicated_visitor(
            &config,
            required_safetensors_parameters(&config),
            &context,
            &tokens,
        );
        assert_eq!(run.outputs.len(), 3);
        let projections = context.projections.lock().unwrap();
        let readout = &projections.last().unwrap().0;
        let positions = projections
            .iter()
            .filter(|(name, _)| name == readout)
            .map(|(_, shape)| shape[1])
            .collect::<Vec<_>>();
        assert_eq!(
            positions,
            [1, 1, 1],
            "{}: actual projection input",
            config["model_type"]
        );
    }
}

#[test]
fn quantized_prepared_readout_selects_before_tied_and_untied_projection() {
    for tied in [false, true] {
        let mut config = config("llama", tied);
        config["hidden_size"] = 16.into();
        config["intermediate_size"] = 32.into();
        config["head_dim"] = 8.into();
        config["vocab_size"] = 32.into();
        let context = NumericContext::default();
        let run = execute_numeric_replicated_visitor_with_quantization(
            &config,
            required_safetensors_parameters(&config),
            &context,
            &NumericTensor::token_ids(&[1, 3, 2, 4, 5]),
            Some(eredu_core::QuantizationRequest::Affine {
                group_size: 16,
                bits: 4,
            }),
        );
        assert!(run
            .outputs
            .iter()
            .all(|output| output.data.iter().any(|value| value.abs() > 1e-6)));
        let projections = context.projections.lock().unwrap();
        let readout = &projections.last().unwrap().0;
        assert_eq!(
            projections
                .iter()
                .filter(|(name, _)| name == readout)
                .map(|(_, shape)| shape[1])
                .collect::<Vec<_>>(),
            [1, 1, 1]
        );
    }
}

#[test]
fn v4_partition_readout_preserves_full_prediction_capture_policy() {
    use eredu_runtime::PartitionedLayeredArchitecture;
    #[derive(Default)]
    struct HiddenCapture(Option<NumericTensor>);
    impl LayeredTraversalHook<NumericBackend, deepseek::v4::ForwardContext<NumericTensor>, Error>
        for HiddenCapture
    {
        fn after_group(
            &mut self,
            group: usize,
            hidden: &mut NumericTensor,
            _: &mut deepseek::v4::ForwardContext<NumericTensor>,
            _: &NumericContext,
        ) -> Result<(), Error> {
            if group == 0 {
                self.0 = Some(hidden.clone());
            }
            Ok(())
        }
    }
    for capture_policy in [false, true] {
        let context = NumericContext::default();
        let mut args = tiny_v4_args();
        if capture_policy {
            args.target_capture_policy =
                Some(deepseek::config::V4TargetCapturePolicy::new(vec![1, 0], 3).unwrap());
        }
        let mut state = DeviceState::<NumericBackend, _>::create(
            deepseek::v4::state_layout(&args).unwrap(),
            |layer, _| {
                let ratios = match args.attention_policy(layer).unwrap() {
                    deepseek::V4AttentionPolicy::Local => vec![],
                    deepseek::V4AttentionPolicy::Compressed { ratio: 4 } => vec![4, 4],
                    deepseek::V4AttentionPolicy::Compressed { ratio } => vec![ratio],
                };
                Ok::<_, Error>(NumericPoolingCache::new(args.sliding_window, &ratios))
            },
        )
        .unwrap();
        let mut runtime = ResidentRuntime::new(
            deepseek::v4::Model::<NumericBackend>::new(args, &context).unwrap(),
            &context,
        )
        .unwrap();
        let mut capture = HiddenCapture::default();
        let (expected, forward) = runtime
            .forward_with_traversal_hook(
                deepseek::mtp::EmbeddedInput::target(&NumericTensor::token_ids(&[1, 2, 3]), None),
                &mut state,
                &context,
                &mut capture,
            )
            .unwrap();
        let expected_capture = forward.target_capture().unwrap();
        let hidden = capture.0.unwrap();
        for demand in [
            OutputDemand::Sequence,
            OutputDemand::LastPosition,
            OutputDemand::StateOnly,
        ] {
            context.projections.lock().unwrap().clear();
            let output = runtime
                .architecture_mut()
                .finish_partition_with_readout(
                    &hidden,
                    &mut state,
                    &forward,
                    true,
                    None,
                    &context,
                    &mut eredu_runtime::NoopObserver,
                    demand,
                )
                .unwrap();
            match output {
                eredu_runtime::LayeredPartitionOutput::Final { output, retained } => {
                    let wanted = if demand == OutputDemand::LastPosition {
                        expected.axis_slice(1, 2, 3)
                    } else {
                        expected.clone()
                    };
                    assert_tensor_close(&output, &wanted, "V4 selected scores");
                    assert_tensor_close(
                        retained.as_ref().unwrap(),
                        expected_capture,
                        "V4 full retained captures",
                    );
                    assert_eq!(
                        context.projections.lock().unwrap().last().unwrap().1[1],
                        demand.positions(3) as i32
                    );
                }
                eredu_runtime::LayeredPartitionOutput::StateOnly { retained } => {
                    assert_eq!(demand, OutputDemand::StateOnly);
                    assert_tensor_close(&retained, &hidden, "V4 completion dependency");
                    assert!(context.projections.lock().unwrap().is_empty());
                }
                _ => panic!("output owner emitted a boundary"),
            }
        }
    }
}

// Fixture cost contract for scheduler/session conformance, deliberately separate
// from native mechanism bounds. The tiny eager scalar operations run in this
// test process; these bounds are never used for MLX reporting or admission.
pub(super) fn scalar_session_admission(
    capability: &eredu_architectures::capability::CapabilityEstimate,
    geometry: eredu_core::InferenceGeometry,
) -> eredu_core::Admission {
    use eredu_core::*;
    let request = AdmissionRequest {
        input: InputTokenCount::text(geometry.cached_positions + geometry.input_positions),
        max_output_tokens: geometry.max_output_tokens,
        batch_size: geometry.batch_size,
        safety_reserve_bytes: 0,
        application_memory_budget_bytes: None,
        require_complete_estimate: true,
    };
    let bound = || WorkspaceBound::bounded(16 << 20, "tiny eager scalar fixture contract");
    let state = estimate_runtime_state(
        capability.state_layout(),
        request.input,
        request.max_output_tokens,
        request.batch_size,
        std::num::NonZeroU8::new(4).unwrap(),
    )
    .unwrap()
    .with_execution_workspace(ExecutionWorkspaceEstimate {
        geometry,
        activations: bound(),
        attention: bound(),
        vocabulary: bound(),
        state_update: bound(),
        materialization: bound(),
        retained: bound(),
    })
    .unwrap();
    match apply_admission_policy(capability.capabilities(), request, state, None).unwrap() {
        AdmissionResult::Admitted(admission) => admission,
        other => panic!("fixture admission: {other:?}"),
    }
}

struct FailableTextSource {
    source: eredu_architectures::prefill::PreparedTextPrefill<NumericTensor>,
    geometry: eredu_core::InferenceGeometry,
    fail_second: bool,
    preparations: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}
impl<A> eredu_runtime::replicated_session::PreparedPrefillSource<A, NumericBackend, ReadoutState>
    for FailableTextSource
where
    A: eredu_runtime::ReplicatedTextArchitecture<NumericBackend, ReadoutState, Error = Error>,
{
    type Chunk = eredu_architectures::prefill::TextPrefillChunk<NumericTensor>;
    fn geometry(&self) -> eredu_core::InferenceGeometry {
        self.geometry
    }
    fn prepare_chunk(
        &self,
        chunk: &eredu_runtime::prefill::PrefillChunk,
        context: &NumericContext,
    ) -> Result<Self::Chunk, Error> {
        self.preparations
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        if self.fail_second && chunk.input.start >= 2 {
            return Err(Error::backend("injected second span preparation failure"));
        }
        self.source.prepare_span(chunk, context)
    }
    fn input<'a>(&'a self, chunk: &'a Self::Chunk) -> A::Input<'a> {
        A::text_input(chunk.tokens(), chunk.mask())
    }
}

struct CompletedChunkVisitor<'a>(&'a NumericContext);

fn assert_span_rejection<T, A: std::fmt::Display, P: std::fmt::Display, M: std::fmt::Display>(
    result: Result<T, eredu_runtime::ReplicatedTextSessionError<A, P, M>>,
    expected: impl FnOnce(&eredu_runtime::working_memory::WorkingMemoryError) -> bool,
) {
    match result {
        Err(eredu_runtime::ReplicatedTextSessionError::BeforeStateMutation(error)) => {
            match *error {
                eredu_runtime::ReplicatedTextSessionError::WorkingMemory(error) => {
                    assert!(expected(&error), "unexpected admission failure: {error}");
                }
                error => panic!("unexpected rejection: {error}"),
            }
        }
        Err(error) => panic!("failure must precede state mutation: {error}"),
        Ok(_) => panic!("unquoted invocation was executed"),
    }
}

struct CancelCompletedSpan(eredu_core::GenerationCancellationToken, usize);

#[derive(Default)]
struct RejectCompletedSpan(usize);
impl eredu_runtime::ActivationObserver<NumericTensor, Error> for RejectCompletedSpan {
    fn requires_sequence_readout(&self) -> bool {
        false
    }
    fn transactional(&self) -> bool {
        true
    }
    fn observe(&mut self, _: &str, _: &NumericTensor) -> Result<(), Error> {
        Ok(())
    }
    fn complete_transaction(&mut self, _: eredu_core::DistributedCommitEpoch) -> Result<(), Error> {
        self.0 += 1;
        Err(Error::backend("reject completed decoder output"))
    }
}

impl eredu_runtime::ActivationObserver<NumericTensor, Error> for CancelCompletedSpan {
    fn requires_sequence_readout(&self) -> bool {
        false
    }
    fn transactional(&self) -> bool {
        true
    }
    fn observe(&mut self, _: &str, _: &NumericTensor) -> Result<(), Error> {
        Ok(())
    }
    fn finish_transaction(&mut self, _: eredu_core::DistributedCommitEpoch, committed: bool) {
        if committed {
            self.1 += 1;
            self.0.cancel();
        }
    }
}
impl ReplicatedTextArchitectureVisitor<NumericBackend, ReadoutState> for CompletedChunkVisitor<'_> {
    type Output = NumericReplicatedRun;
    type Error = String;
    fn construction_started(&mut self) {}
    fn visit<A>(
        self,
        prepared: PreparedReplicatedTextArchitecture<A>,
        checkpoint: RetainedCheckpointSource,
    ) -> Result<Self::Output, Self::Error>
    where
        A: eredu_runtime::ReplicatedTextArchitecture<NumericBackend, ReadoutState, Error = Error>
            + 'static,
        A::StaticModules: Clone,
        A::Error: std::fmt::Display,
    {
        let context = self.0;
        let capability = prepared.capability_estimate().clone();
        let mechanisms = if prepared.selected().residency().is_fully_resident() {
            NumericReplicatedMechanisms::with_bound_checkpoint(checkpoint)
        } else {
            NumericReplicatedMechanisms::with_bounded_checkpoint(checkpoint)
        };
        let (mut session, _) =
            eredu_architectures::prepared_execution::construct_selected_text_session::<
                NumericBackend,
                _,
                _,
            >(prepared, mechanisms, context)?;
        let tokens = NumericTensor::token_ids(&[1, 3, 2, 4, 5]);
        let expected_prefill = session
            .prefill(&tokens, None, context)
            .map_err(|e| e.to_string())?;
        let readout = context
            .projections
            .lock()
            .unwrap()
            .last()
            .unwrap()
            .0
            .clone();
        let mut outputs = vec![expected_prefill.clone()];
        for token in [6, 2, 1] {
            outputs.push(
                session
                    .decode(&NumericTensor::token_ids(&[token]), context)
                    .map_err(|e| e.to_string())?,
            );
        }
        let expected_state = session
            .report()
            .map_err(|e| e.to_string())?
            .state_report()
            .clone();
        for (chunk, stepped, cached) in [1, 2, 3, 5]
            .into_iter()
            .flat_map(|n| [(n, false, 0), (n, true, 0), (n, false, 2), (n, true, 2)])
        {
            session.reset(context).map_err(|e| e.to_string())?;
            if cached > 0 {
                session
                    .prefill(&tokens.axis_slice(1, 0, cached as usize), None, context)
                    .map_err(|e| e.to_string())?;
            }
            context.projections.lock().unwrap().clear();
            let geometry = eredu_core::InferenceGeometry {
                batch_size: 1,
                cached_positions: cached,
                input_positions: 5 - cached,
                max_output_tokens: 3,
                prefill_chunk_positions: chunk.min(5 - cached),
                output: OutputDemand::LastPosition,
            };
            let pool = eredu_runtime::working_memory::WorkingMemoryPool::new(1 << 30, 0).unwrap();
            let execution = session.inference_execution_identity().clone();
            let reservation = pool
                .reserve(&execution, &scalar_session_admission(&capability, geometry))
                .unwrap();
            let source =
                eredu_architectures::prefill::PreparedTextPrefill::<NumericTensor>::from_token_ids(
                    std::sync::Arc::from(&[1, 3, 2, 4, 5][cached as usize..]),
                    geometry,
                )
                .unwrap();
            let mut driver = eredu_runtime::prefill::PrefillDriver::new(
                &execution,
                reservation.clone(),
                geometry,
                eredu_core::GenerationCancellationToken::new(),
            )
            .unwrap();
            let mut spans = 0;
            let mut consume = |span: eredu_runtime::prefill::PrefillChunk,
                               result: Option<NumericTensor>| {
                spans += 1;
                if span.input.end == geometry.input_positions {
                    assert_tensor_close(
                        result.as_ref().unwrap(),
                        &expected_prefill,
                        "scheduled chunk prefill",
                    );
                } else {
                    assert!(result.is_none());
                    assert!(!context
                        .projections
                        .lock()
                        .unwrap()
                        .iter()
                        .any(|(name, _)| name == &readout));
                }
                assert_eq!(pool.used_bytes().unwrap(), reservation.bytes());
            };
            if stepped {
                loop {
                    // The persistent driver outlives the short session loan.
                    let mut observer = eredu_runtime::NoopObserver;
                    let mut executor = eredu_runtime::replicated_session::SessionPrefill::new(
                        &mut session,
                        source.clone(),
                        &reservation,
                        context,
                        &mut observer,
                    )
                    .unwrap();
                    match driver.step(&mut executor).map_err(|e| e.to_string())? {
                        eredu_runtime::prefill::PrefillProgress::Chunk { chunk, output } => {
                            consume(chunk, output)
                        }
                        eredu_runtime::prefill::PrefillProgress::Complete => break,
                        other => panic!("eager fixture failed to settle: {other:?}"),
                    }
                }
            } else {
                let mut observer = eredu_runtime::NoopObserver;
                let mut executor = eredu_runtime::replicated_session::SessionPrefill::new(
                    &mut session,
                    source,
                    &reservation,
                    context,
                    &mut observer,
                )
                .unwrap();
                assert_eq!(
                    driver
                        .run(&mut executor, &mut consume)
                        .map_err(|e| e.to_string())?,
                    eredu_runtime::prefill::PrefillOutcome::Complete
                );
            }
            assert_eq!(
                spans,
                geometry
                    .input_positions
                    .div_ceil(geometry.prefill_chunk_positions)
            );
            drop(driver);
            let charged = reservation.bytes();
            drop(reservation);
            assert_eq!(pool.used_bytes().unwrap(), charged);
            assert_eq!(
                context
                    .projections
                    .lock()
                    .unwrap()
                    .iter()
                    .filter(|(name, _)| name == &readout)
                    .map(|(_, shape)| shape[1])
                    .collect::<Vec<_>>(),
                [1]
            );
            let decode_checkpoint = session
                .checkpoint_complete(context)
                .map_err(|e| e.to_string())?;
            let before_rejection = session
                .report()
                .map_err(|e| e.to_string())?
                .state_report()
                .clone();
            let projection_count = context.projections.lock().unwrap().len();
            for invalid in [
                NumericTensor::token_ids(&[6, 2]),
                NumericTensor::new(vec![2, 1], vec![6.0, 2.0]),
            ] {
                assert_span_rejection(
                    session.decode_input_result_with_observer(
                        Ok(A::text_input(&invalid, None)),
                        context,
                        &mut eredu_runtime::NoopObserver,
                    ),
                    |error| {
                        matches!(
                            error,
                            eredu_runtime::working_memory::WorkingMemoryError::SpanShapeMismatch {
                                expected: [1, 1],
                                ..
                            }
                        )
                    },
                );
            }
            assert_span_rejection(
                session.prefill(&NumericTensor::token_ids(&[6]), None, context),
                |error| {
                    matches!(
                        error,
                        eredu_runtime::working_memory::WorkingMemoryError::InvocationPhaseMismatch
                    )
                },
            );
            assert_eq!(context.projections.lock().unwrap().len(), projection_count);
            assert_state_exact(
                session.report().map_err(|e| e.to_string())?.state_report(),
                &before_rejection,
                before_rejection.layout().len(),
                "invalid decode geometry performs no state work",
            );
            let mut rejected_output = RejectCompletedSpan::default();
            assert!(session
                .decode_with_observer(
                    &NumericTensor::token_ids(&[6]),
                    context,
                    &mut rejected_output,
                )
                .is_err());
            assert_eq!(rejected_output.0, 1);
            assert_state_exact(
                session.report().map_err(|e| e.to_string())?.state_report(),
                &before_rejection,
                before_rejection.layout().len(),
                "output failure restores state and the admitted decoder frontier",
            );
            assert_eq!(pool.used_bytes().unwrap(), charged);
            drop(before_rejection);
            for (token, expected) in [6, 2, 1].into_iter().zip(&outputs[1..]) {
                let actual = session
                    .decode(&NumericTensor::token_ids(&[token]), context)
                    .map_err(|e| e.to_string())?;
                assert_tensor_close(
                    &actual,
                    expected,
                    "decode after completed state-only chunks",
                );
                assert_eq!(pool.used_bytes().unwrap(), charged);
            }
            let projection_count = context.projections.lock().unwrap().len();
            assert_span_rejection(
                session.decode_input(
                    A::text_input(&NumericTensor::token_ids(&[3]), None),
                    context,
                ),
                |error| {
                    matches!(error, eredu_runtime::working_memory::WorkingMemoryError::OutputAllowanceExceeded { position: 8, limit: 8 })
                },
            );
            assert_eq!(context.projections.lock().unwrap().len(), projection_count);
            assert_state_exact(
                session.report().map_err(|e| e.to_string())?.state_report(),
                &expected_state,
                expected_state.layout().len(),
                "exhausted output allowance performs no state work",
            );
            session
                .rollback_complete(decode_checkpoint, context)
                .map_err(|e| e.to_string())?;
            assert_eq!(pool.used_bytes().unwrap(), charged);
            for (token, expected) in [6, 2, 1].into_iter().zip(&outputs[1..]) {
                let actual = session
                    .sequence_logits(
                        A::text_input(&NumericTensor::token_ids(&[token]), None),
                        eredu_runtime::ExpertPass::Decode,
                        context,
                    )
                    .map_err(|e| e.to_string())?;
                // A one-position sequence has the same priced readout as decode.
                assert_tensor_close(
                    &actual,
                    expected,
                    "restored admission replays the same bounded decoder spans",
                );
            }
            assert_span_rejection(
                session.decode(&NumericTensor::token_ids(&[3]), context),
                |error| {
                    matches!(error, eredu_runtime::working_memory::WorkingMemoryError::OutputAllowanceExceeded { position: 8, limit: 8 })
                },
            );
            assert_state_exact(
                session.report().map_err(|e| e.to_string())?.state_report(),
                &expected_state,
                expected_state.layout().len(),
                "completed state-only chunks preserve session state",
            );
            let checkpoint = session
                .checkpoint_complete(context)
                .map_err(|e| e.to_string())?;
            session.reset(context).map_err(|e| e.to_string())?;
            assert_eq!(pool.used_bytes().unwrap(), charged);
            session
                .rollback_complete(checkpoint, context)
                .map_err(|e| e.to_string())?;
            assert_eq!(pool.used_bytes().unwrap(), charged);
            assert_state_exact(
                session.report().map_err(|e| e.to_string())?.state_report(),
                &expected_state,
                expected_state.layout().len(),
                "restored checkpoint retains state and its charge",
            );
            let survivor = session
                .checkpoint_complete(context)
                .map_err(|e| e.to_string())?;
            session.reset(context).map_err(|e| e.to_string())?;
            assert_eq!(pool.used_bytes().unwrap(), charged);
            drop(survivor);
            assert_eq!(pool.used_bytes().unwrap(), 0);
        }
        // The ordinary native adapter uses this same automatic source driver.
        for cached in [0, 2] {
            for chunk in [2, 3] {
                session.reset(context).map_err(|e| e.to_string())?;
                if cached > 0 {
                    session
                        .prefill(&tokens.axis_slice(1, 0, cached), None, context)
                        .map_err(|e| e.to_string())?;
                }
                context.projections.lock().unwrap().clear();
                let actual = session
                    .try_prefill_source_cancellable(
                        None,
                        Some([1, (5 - cached) as u64]),
                        std::num::NonZeroU64::new(chunk),
                        |geometry| {
                            assert_eq!(geometry.cached_positions, cached as u64);
                            eredu_architectures::prefill::PreparedTextPrefill::from_tensor(
                                tokens.axis_slice(1, cached, 5),
                                None,
                                geometry,
                            )
                            .map(Some)
                        },
                        &eredu_core::GenerationCancellationToken::new(),
                        context,
                        &mut eredu_runtime::NoopObserver,
                    )
                    .map_err(|e| e.to_string())?;
                let eredu_runtime::replicated_session::PrefillSourceOutcome::Complete(actual) = actual else {
                    return Err("ordinary text source did not complete".into());
                };
                assert_tensor_close(
                    &actual,
                    &expected_prefill,
                    "automatic native-adapter prefill",
                );
                assert_eq!(
                    context
                        .projections
                        .lock()
                        .unwrap()
                        .iter()
                        .filter(|(name, _)| name == &readout)
                        .map(|(_, shape)| shape[1])
                        .collect::<Vec<_>>(),
                    [1]
                );
                for (token, expected) in [6, 2, 1].into_iter().zip(&outputs[1..]) {
                    let actual = session
                        .decode(&NumericTensor::token_ids(&[token]), context)
                        .map_err(|e| e.to_string())?;
                    assert_tensor_close(&actual, expected, "decode after automatic scheduling");
                }
                assert_state_exact(
                    session.report().map_err(|e| e.to_string())?.state_report(),
                    &expected_state,
                    expected_state.layout().len(),
                    "automatic scheduling retains state",
                );
            }
        }
        // The same production gateway must preserve supplied admission through
        // ingress, all spans and cached decode, and cannot choose its fallback.
        session.reset(context).map_err(|e| e.to_string())?;
        session
            .prefill(&tokens.axis_slice(1, 0, 2), None, context)
            .map_err(|e| e.to_string())?;
        let geometry = eredu_core::InferenceGeometry {
            batch_size: 1,
            cached_positions: 2,
            input_positions: 3,
            max_output_tokens: 3,
            prefill_chunk_positions: 2,
            output: OutputDemand::LastPosition,
        };
        let pool = eredu_runtime::working_memory::WorkingMemoryPool::new(1 << 30, 0).unwrap();
        let execution = session.inference_execution_identity().clone();
        let admission = scalar_session_admission(&capability, geometry);
        let request: eredu_runtime::working_memory::InferenceRequest =
            pool.reserve(&execution, &admission).unwrap().into();
        let not_prepared = |_| -> Result<
            Option<eredu_architectures::prefill::PreparedTextPrefill<NumericTensor>>,
            Error,
        > { panic!("rejected request must not construct ingress") };
        assert_span_rejection(
            session.try_prefill_source_cancellable(
                Some(&request),
                Some([1, 4]),
                std::num::NonZeroU64::new(2),
                not_prepared,
                &eredu_core::GenerationCancellationToken::new(),
                context,
                &mut eredu_runtime::NoopObserver,
            ),
            |error| {
                matches!(
                    error,
                    eredu_runtime::working_memory::WorkingMemoryError::SpanShapeMismatch { .. }
                )
            },
        );
        assert_span_rejection(
            session.try_prefill_source_cancellable(
                Some(&request),
                Some([1, 3]),
                std::num::NonZeroU64::new(1),
                not_prepared,
                &eredu_core::GenerationCancellationToken::new(),
                context,
                &mut eredu_runtime::NoopObserver,
            ),
            |error| {
                matches!(
                    error,
                    eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch
                )
            },
        );
        let foreign: eredu_runtime::working_memory::InferenceRequest = pool
            .reserve(
                &eredu_runtime::working_memory::InferenceExecutionIdentity::default(),
                &admission,
            )
            .unwrap()
            .into();
        assert_span_rejection(
            session.try_prefill_source_cancellable(
                Some(&foreign),
                Some([1, 3]),
                std::num::NonZeroU64::new(2),
                not_prepared,
                &eredu_core::GenerationCancellationToken::new(),
                context,
                &mut eredu_runtime::NoopObserver,
            ),
            |error| {
                matches!(
                    error,
                    eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch
                )
            },
        );
        drop(foreign);
        for alternate in [
            eredu_core::InferenceGeometry {
                cached_positions: 0,
                ..geometry
            },
            eredu_core::InferenceGeometry {
                output: OutputDemand::Sequence,
                ..geometry
            },
            eredu_core::InferenceGeometry {
                output: OutputDemand::StateOnly,
                ..geometry
            },
        ] {
            let mismatched: eredu_runtime::working_memory::InferenceRequest = pool
                .reserve(
                    &execution,
                    &scalar_session_admission(&capability, alternate),
                )
                .unwrap()
                .into();
            assert_span_rejection(
                session.try_prefill_source_cancellable(
                    Some(&mismatched),
                    Some([1, 3]),
                    std::num::NonZeroU64::new(2),
                    not_prepared,
                    &eredu_core::GenerationCancellationToken::new(),
                    context,
                    &mut eredu_runtime::NoopObserver,
                ),
                |error| {
                    if alternate.cached_positions == 0 {
                        matches!(error, eredu_runtime::working_memory::WorkingMemoryError::StateFrontierMismatch { .. })
                    } else {
                        matches!(error, eredu_runtime::working_memory::WorkingMemoryError::OutputDemandMismatch { .. })
                    }
                },
            );
        }
        let missing = session.try_prefill_source_cancellable(
            Some(&request),
            Some([1, 3]),
            std::num::NonZeroU64::new(2),
            |_| Ok(None::<eredu_architectures::prefill::PreparedTextPrefill<NumericTensor>>),
            &eredu_core::GenerationCancellationToken::new(),
            context,
            &mut eredu_runtime::NoopObserver,
        );
        assert_span_rejection(missing, |error| {
            matches!(
                error,
                eredu_runtime::working_memory::WorkingMemoryError::PreparedSourceUnavailable
            )
        });
        assert_span_rejection(
            session.try_prefill_source_cancellable(
                Some(&request),
                Some([1, 3]),
                std::num::NonZeroU64::new(2),
                not_prepared,
                &eredu_core::GenerationCancellationToken::new(),
                context,
                &mut eredu_runtime::NoopObserver,
            ),
            |error| {
                matches!(
                    error,
                    eredu_runtime::working_memory::WorkingMemoryError::AlreadyStarted
                )
            },
        );
        assert_eq!(
            pool.used_bytes().unwrap(),
            request.memory_reservation().unwrap().bytes()
        );
        drop(request);
        assert_eq!(pool.used_bytes().unwrap(), 0);
        let request: eredu_runtime::working_memory::InferenceRequest =
            pool.reserve(&execution, &admission).unwrap().into();
        let charged = request.memory_reservation().unwrap().bytes();
        let actual = session
            .try_prefill_source_cancellable(
                Some(&request),
                Some([1, 3]),
                std::num::NonZeroU64::new(2),
                |selected| {
                    assert_eq!(selected, geometry);
                    assert_eq!(pool.used_bytes().unwrap(), charged);
                    eredu_architectures::prefill::PreparedTextPrefill::from_tensor(
                        tokens.axis_slice(1, 2, 5),
                        None,
                        selected,
                    )
                    .map(Some)
                },
                &eredu_core::GenerationCancellationToken::new(),
                context,
                &mut eredu_runtime::NoopObserver,
            )
            .map_err(|e| e.to_string())?;
        let eredu_runtime::replicated_session::PrefillSourceOutcome::Complete(actual) = actual
        else {
            panic!("admitted source must complete without an unbudgeted fallback")
        };
        assert_tensor_close(
            &actual,
            &expected_prefill,
            "admitted source gateway prefill",
        );
        drop(request);
        assert_eq!(pool.used_bytes().unwrap(), charged);
        for (token, expected) in [6, 2, 1].into_iter().zip(&outputs[1..]) {
            let actual = session
                .decode(&NumericTensor::token_ids(&[token]), context)
                .map_err(|e| e.to_string())?;
            assert_tensor_close(&actual, expected, "decode retains gateway admission");
            assert_eq!(pool.used_bytes().unwrap(), charged);
        }
        assert_span_rejection(
            session.decode(&NumericTensor::token_ids(&[3]), context),
            |error| {
                matches!(
                    error,
                    eredu_runtime::working_memory::WorkingMemoryError::OutputAllowanceExceeded {
                        position: 8,
                        limit: 8
                    }
                )
            },
        );
        session.reset(context).map_err(|e| e.to_string())?;
        assert_eq!(pool.used_bytes().unwrap(), 0);

        // The production gateway cancels at the first completed span, preserving
        // exactly that prefix and a healthy session without vocabulary output.
        session.reset(context).map_err(|e| e.to_string())?;
        session
            .prefill(&tokens.axis_slice(1, 0, 2), None, context)
            .map_err(|e| e.to_string())?;
        let prefix = session
            .report()
            .map_err(|e| e.to_string())?
            .state_report()
            .clone();
        session.reset(context).map_err(|e| e.to_string())?;
        context.projections.lock().unwrap().clear();
        let cancellation = eredu_core::GenerationCancellationToken::new();
        let mut observer = CancelCompletedSpan(cancellation.clone(), 0);
        let cancel_geometry = eredu_core::InferenceGeometry {
            cached_positions: 0,
            input_positions: 5,
            ..geometry
        };
        let cancel_request: eredu_runtime::working_memory::InferenceRequest = pool
            .reserve(
                &execution,
                &scalar_session_admission(&capability, cancel_geometry),
            )
            .unwrap()
            .into();
        let cancellation_charge = cancel_request.memory_reservation().unwrap().bytes();
        let outcome = session
            .try_prefill_source_cancellable(
                Some(&cancel_request),
                Some([1, 5]),
                std::num::NonZeroU64::new(2),
                |geometry| {
                    eredu_architectures::prefill::PreparedTextPrefill::from_tensor(
                        tokens.clone(),
                        None,
                        geometry,
                    )
                    .map(Some)
                },
                &cancellation,
                context,
                &mut observer,
            )
            .map_err(|e| e.to_string())?;
        assert!(matches!(
            outcome,
            eredu_runtime::replicated_session::PrefillSourceOutcome::Cancelled
        ));
        assert_eq!(observer.1, 1);
        assert!(!context
            .projections
            .lock()
            .unwrap()
            .iter()
            .any(|(name, _)| name == &readout));
        assert_state_exact(
            session.report().map_err(|e| e.to_string())?.state_report(),
            &prefix,
            prefix.layout().len(),
            "gateway cancellation commits only one span",
        );
        drop(cancel_request);
        assert_eq!(pool.used_bytes().unwrap(), cancellation_charge);
        // An admission for an older cached prefix cannot prepare the first span.
        session.reset(context).map_err(|e| e.to_string())?;
        assert_eq!(pool.used_bytes().unwrap(), 0);
        session
            .prefill(&tokens.axis_slice(1, 0, 2), None, context)
            .map_err(|e| e.to_string())?;
        let retained = session
            .report()
            .map_err(|e| e.to_string())?
            .state_report()
            .clone();
        let geometry = eredu_core::InferenceGeometry {
            batch_size: 1,
            cached_positions: 0,
            input_positions: 5,
            max_output_tokens: 3,
            prefill_chunk_positions: 2,
            output: OutputDemand::LastPosition,
        };
        let pool = eredu_runtime::working_memory::WorkingMemoryPool::new(1 << 30, 0).unwrap();
        let execution = session.inference_execution_identity().clone();
        let reservation = pool
            .reserve(&execution, &scalar_session_admission(&capability, geometry))
            .unwrap();
        let preparations = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let source = FailableTextSource {
            source: eredu_architectures::prefill::PreparedTextPrefill::from_token_ids(
                std::sync::Arc::from([1, 3, 2, 4, 5]),
                geometry,
            )
            .unwrap(),
            geometry,
            fail_second: false,
            preparations: preparations.clone(),
        };
        let mut observer = eredu_runtime::NoopObserver;
        let mut executor = eredu_runtime::replicated_session::SessionPrefill::new(
            &mut session,
            source,
            &reservation,
            context,
            &mut observer,
        )
        .unwrap();
        let mut driver = eredu_runtime::prefill::PrefillDriver::new(
            &execution,
            &reservation,
            geometry,
            eredu_core::GenerationCancellationToken::new(),
        )
        .unwrap();
        assert!(matches!(
            driver.step(&mut executor),
            Err(eredu_runtime::prefill::PrefillError::Submission(
                eredu_runtime::ReplicatedTextSessionError::WorkingMemory(
                    eredu_runtime::working_memory::WorkingMemoryError::StateFrontierMismatch {
                        expected: 0,
                        actual: 2,
                    }
                )
            ))
        ));
        assert!(matches!(
            driver.step(&mut executor),
            Err(eredu_runtime::prefill::PrefillError::Failed)
        ));
        assert_eq!(preparations.load(std::sync::atomic::Ordering::SeqCst), 0);
        drop(driver);
        drop(executor);
        assert_state_exact(
            session.report().map_err(|e| e.to_string())?.state_report(),
            &retained,
            retained.layout().len(),
            "stale admission leaves retained prefix unchanged",
        );
        session.reset(context).map_err(|e| e.to_string())?;
        drop(reservation);
        assert_eq!(pool.used_bytes().unwrap(), 0);

        // Cancellation and typed preparation failure stop the same scheduler
        // after one committed state-only chunk, without losing its mutable state.
        for fail_second in [false, true] {
            session.reset(context).map_err(|e| e.to_string())?;
            session
                .prefill(&tokens.axis_slice(1, 0, 2), None, context)
                .map_err(|e| e.to_string())?;
            let partial = session
                .report()
                .map_err(|e| e.to_string())?
                .state_report()
                .clone();
            session.reset(context).map_err(|e| e.to_string())?;
            context.projections.lock().unwrap().clear();
            let geometry = eredu_core::InferenceGeometry {
                batch_size: 1,
                cached_positions: 0,
                input_positions: 5,
                max_output_tokens: 3,
                prefill_chunk_positions: 2,
                output: OutputDemand::LastPosition,
            };
            let pool = eredu_runtime::working_memory::WorkingMemoryPool::new(1 << 30, 0).unwrap();
            let execution = session.inference_execution_identity().clone();
            let reservation = pool
                .reserve(&execution, &scalar_session_admission(&capability, geometry))
                .unwrap();
            let preparations = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let source = FailableTextSource {
                source: eredu_architectures::prefill::PreparedTextPrefill::from_token_ids(
                    std::sync::Arc::from([1, 3, 2, 4, 5]),
                    geometry,
                )
                .unwrap(),
                geometry,
                fail_second,
                preparations: preparations.clone(),
            };
            let cancellation = eredu_core::GenerationCancellationToken::new();
            let mut observer = eredu_runtime::NoopObserver;
            let mut executor = eredu_runtime::replicated_session::SessionPrefill::new(
                &mut session,
                source,
                &reservation,
                context,
                &mut observer,
            )
            .unwrap();
            let substituted =
                eredu_runtime::working_memory::InferenceRequest::without_memory_budget(
                    &execution, geometry,
                )
                .unwrap();
            assert!(matches!(
                eredu_runtime::prefill::PrefillExecutor::submit_chunk(
                    &mut executor,
                    &eredu_runtime::prefill::PrefillChunk {
                        input: 0..2,
                        position: 0,
                        output: OutputDemand::StateOnly
                    },
                    substituted
                ),
                Err(eredu_runtime::ReplicatedTextSessionError::WorkingMemory(
                    eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch
                ))
            ));
            assert_eq!(preparations.load(std::sync::atomic::Ordering::SeqCst), 0);
            let mut driver = eredu_runtime::prefill::PrefillDriver::new(
                &execution,
                reservation.clone(),
                geometry,
                cancellation.clone(),
            )
            .unwrap();
            assert!(matches!(
                driver.step(&mut executor).map_err(|e| e.to_string())?,
                eredu_runtime::prefill::PrefillProgress::Chunk { output: None, .. }
            ));
            if fail_second {
                assert!(matches!(
                    driver.step(&mut executor),
                    Err(eredu_runtime::prefill::PrefillError::Submission(
                        eredu_runtime::ReplicatedTextSessionError::BeforeStateMutation(_)
                    ))
                ));
                assert!(matches!(
                    driver.step(&mut executor),
                    Err(eredu_runtime::prefill::PrefillError::Failed)
                ));
            } else {
                cancellation.cancel();
                for _ in 0..2 {
                    assert!(matches!(
                        driver.step(&mut executor).unwrap(),
                        eredu_runtime::prefill::PrefillProgress::Cancelled
                    ));
                }
            }
            assert_eq!(
                preparations.load(std::sync::atomic::Ordering::SeqCst),
                if fail_second { 2 } else { 1 }
            );
            assert!(!context
                .projections
                .lock()
                .unwrap()
                .iter()
                .any(|(name, _)| name == &readout));
            drop(driver);
            drop(executor);
            let projection_count = context.projections.lock().unwrap().len();
            assert_span_rejection(
                session.decode_input(
                    A::text_input(&NumericTensor::token_ids(&[3]), None),
                    context,
                ),
                |error| {
                    matches!(
                        error,
                        eredu_runtime::working_memory::WorkingMemoryError::InvocationPhaseMismatch
                    )
                },
            );
            assert_eq!(context.projections.lock().unwrap().len(), projection_count);
            assert_state_exact(
                session.report().map_err(|e| e.to_string())?.state_report(),
                &partial,
                partial.layout().len(),
                "stopped scheduler preserves committed prefix",
            );
            assert_eq!(pool.used_bytes().unwrap(), reservation.bytes());
            let charged = reservation.bytes();
            drop(reservation);
            assert_eq!(pool.used_bytes().unwrap(), charged);
            session.reset(context).map_err(|e| e.to_string())?;
            assert_eq!(pool.used_bytes().unwrap(), 0);
        }
        Ok(NumericReplicatedRun {
            outputs,
            state: expected_state,
            bank_reports: BTreeMap::new(),
        })
    }
}

#[test]
fn prepared_sessions_complete_state_only_chunks_in_all_weight_residencies() {
    let mut cases = heterogeneous_replicated_configs();
    cases.extend([
        config("llama", true),
        config("qwen3", false),
        nanbeige::tiny_config(false),
        gemma2::tiny_config(),
    ]);
    check_prepared_chunk_cases(cases);
}

#[test]
fn prepared_mistral_and_qwen2_sliding_chunks_preserve_state_and_cancellation() {
    let mut cases = Vec::new();
    for tied in [false, true] {
        let mut mistral = config("mistral", tied);
        mistral["num_hidden_layers"] = 2.into();
        mistral["sliding_window"] = 2.into();
        cases.push(mistral);

        let mut qwen2 = config("qwen2", tied);
        qwen2["num_hidden_layers"] = 2.into();
        // The first layer retains full attention; the second has a two-row window.
        qwen2["max_window_layers"] = 1.into();
        cases.push(qwen2);
    }
    // Reuse the selected-source run/step, complete-state, three-decode,
    // cancellation and failure assertions with the actual residency adapters.
    check_prepared_chunk_cases(cases);
}

fn check_prepared_chunk_cases(cases: impl IntoIterator<Item = serde_json::Value>) {
    for config in cases {
        let (artifact, _) =
            prepared_adapter::payload_fixture_config_with(&config, 1.0, |name, shape| {
                (config["model_type"] == "nemotron_h" && name.ends_with(".A_log")).then(|| {
                    NumericTensor::new(
                        shape.to_vec(),
                        vec![-0.7; shape.iter().product::<i32>() as usize],
                    )
                })
            });
        let inspection =
            eredu_architectures::configuration::inspect_artifact(artifact.path()).unwrap();
        for residency in [
            eredu_core::ResidencyPlan::FullyResident,
            eredu_core::ResidencyPlan::LayerwiseHost {
                device_layer_window: 1,
                device_budget_bytes: None,
                host_budget_bytes: None,
            },
            eredu_core::ResidencyPlan::DenseDiskStream {
                device_budget_bytes: 16 << 20,
                host_budget_bytes: 32 << 20,
                host_lookahead: 1,
                background_queue: 1,
            },
        ] {
            let mut context = NumericContext::default();
            context.bind_checkpoint_values = true;
            let plan = prepared_adapter::plan(None).with_residency(residency.clone());
            let sources = prepared_adapter::prepare(
                &inspection,
                &plan,
                &prepared_adapter::NumericPreparationProvider { addressable: false },
            )
            .unwrap();
            prepared_adapter::replicated_with_visitor(
                sources,
                &context,
                CompletedChunkVisitor(&context),
            )
            .unwrap_or_else(|error| panic!("{}: {residency:?}: {error}", config["model_type"]));
        }
    }
}

#[test]
fn text_prefill_source_preserves_batches_and_cached_mask_coordinates() {
    use eredu_architectures::prefill::PreparedTextPrefill;
    use eredu_runtime::prefill::PrefillChunk;
    let context = NumericContext::default();
    let geometry = eredu_core::InferenceGeometry {
        batch_size: 2,
        cached_positions: 2,
        input_positions: 5,
        max_output_tokens: 3,
        prefill_chunk_positions: 3,
        output: OutputDemand::LastPosition,
    };
    let tokens = NumericTensor::new(vec![2, 5], vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]);
    let mask = NumericTensor::new(
        vec![2, 1, 5, 7],
        (0..70).map(|n| n as f32 * 0.01 - 0.3).collect(),
    );
    let host = PreparedTextPrefill::<NumericTensor>::from_token_ids(
        std::sync::Arc::from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        geometry,
    )
    .unwrap();
    let native =
        PreparedTextPrefill::from_tensor(tokens.clone(), Some(mask.clone()), geometry).unwrap();
    let single_segment = PreparedTextPrefill::from_tensors(vec![tokens.clone()], geometry).unwrap();
    let single_row = PreparedTextPrefill::<NumericTensor>::from_token_ids(
        std::sync::Arc::from([1, 2, 3, 4, 5]),
        eredu_core::InferenceGeometry {
            batch_size: 1,
            ..geometry
        },
    )
    .unwrap();
    let segments = PreparedTextPrefill::from_tensors(
        vec![
            tokens.axis_slice(1, 0, 2),
            tokens.axis_slice(1, 2, 4),
            tokens.axis_slice(1, 4, 5),
        ],
        geometry,
    )
    .unwrap();
    for (start, end) in [(0, 3), (3, 5)] {
        let span = PrefillChunk {
            input: start..end,
            position: 2 + start,
            output: OutputDemand::LastPosition.for_chunk(end == 5),
        };
        let from_host = host.prepare_span(&span, &context).unwrap();
        let from_native = native.prepare_span(&span, &context).unwrap();
        let from_segments = segments.prepare_span(&span, &context).unwrap();
        let from_single_segment = single_segment.prepare_span(&span, &context).unwrap();
        let from_single_row = single_row.prepare_span(&span, &context).unwrap();
        let expected = tokens.axis_slice(1, start as usize, end as usize);
        assert_tensor_close(from_host.tokens(), &expected, "batched host span");
        assert_tensor_close(from_native.tokens(), &expected, "batched native span");
        assert_tensor_close(
            from_single_segment.tokens(),
            &expected,
            "one native segment uses the same span coordinates",
        );
        assert_tensor_close(
            from_single_row.tokens(),
            &expected.axis_slice(0, 0, 1),
            "contiguous host row preserves uneven spans",
        );
        assert_tensor_close(
            from_segments.tokens(),
            &expected,
            "span crossing native text segments",
        );
        assert_tensor_close(
            from_native.mask().unwrap(),
            &mask
                .axis_slice(2, start as usize, end as usize)
                .axis_slice(3, 0, (2 + end) as usize),
            "prompt-relative query, prefix-relative keys",
        );
    }
    let full_host = PreparedTextPrefill::<NumericTensor>::from_token_ids(
        std::sync::Arc::from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        eredu_core::InferenceGeometry {
            prefill_chunk_positions: 5,
            ..geometry
        },
    )
    .unwrap();
    let full = full_host
        .prepare_span(
            &PrefillChunk {
                input: 0..5,
                position: 2,
                output: OutputDemand::LastPosition,
            },
            &context,
        )
        .unwrap();
    assert_tensor_close(full.tokens(), &tokens, "contiguous complete batch span");
    let broadcast =
        NumericTensor::new(vec![2, 1, 1, 7], (0..14).map(|n| n as f32 * 0.02).collect());
    let source =
        PreparedTextPrefill::from_tensor(tokens, Some(broadcast.clone()), geometry).unwrap();
    let span = source
        .prepare_span(
            &PrefillChunk {
                input: 0..3,
                position: 2,
                output: OutputDemand::StateOnly,
            },
            &context,
        )
        .unwrap();
    assert_tensor_close(
        span.mask().unwrap(),
        &broadcast.axis_slice(3, 0, 5),
        "broadcast query preserves cached keys",
    );
}

#[test]
fn text_prefill_source_rejects_inconsistent_geometry_and_spans() {
    use eredu_architectures::prefill::PreparedTextPrefill;
    use eredu_runtime::prefill::PrefillChunk;
    let context = NumericContext::default();
    let geometry = eredu_core::InferenceGeometry {
        batch_size: 1,
        cached_positions: 2,
        input_positions: 5,
        max_output_tokens: 3,
        prefill_chunk_positions: 3,
        output: OutputDemand::LastPosition,
    };
    let tokens = NumericTensor::token_ids(&[1, 2, 3, 4, 5]);
    assert!(PreparedTextPrefill::<NumericTensor>::from_token_ids(
        std::sync::Arc::from([1, 2]),
        geometry
    )
    .is_err());
    for shape in [vec![5], vec![5, 5], vec![4, 7]] {
        let mask = NumericTensor::new(
            shape.clone(),
            vec![0.; shape.iter().product::<i32>() as usize],
        );
        assert!(PreparedTextPrefill::from_tensor(tokens.clone(), Some(mask), geometry).is_err());
    }
    let source = PreparedTextPrefill::from_tensor(tokens, None, geometry).unwrap();
    for span in [
        PrefillChunk {
            input: 0..3,
            position: 0,
            output: OutputDemand::StateOnly,
        },
        PrefillChunk {
            input: 0..4,
            position: 2,
            output: OutputDemand::StateOnly,
        },
        PrefillChunk {
            input: 3..6,
            position: 5,
            output: OutputDemand::LastPosition,
        },
        PrefillChunk {
            input: 3..3,
            position: 5,
            output: OutputDemand::StateOnly,
        },
        PrefillChunk {
            input: 3..5,
            position: 5,
            output: OutputDemand::StateOnly,
        },
        PrefillChunk {
            input: 0..3,
            position: 2,
            output: OutputDemand::Sequence,
        },
    ] {
        assert!(source.prepare_span(&span, &context).is_err(), "{span:?}");
    }
}

pub(super) fn scheduled_composite_text<A, D>(
    session: &mut eredu_runtime::ReplicatedTextSession<
        eredu_architectures::composite_execution::PreparedCompositeArchitecture<A>,
        NumericBackend,
        NumericReplicatedMechanisms,
        D,
    >,
    admission: &A::AdmissionConfig,
    input: &eredu_runtime::PreparedModelInput<NumericTensor>,
    context: &NumericContext,
    chunk: u64,
    stepped: bool,
) -> Result<NumericTensor, String>
where
    A: eredu_architectures::composite_execution::CompositeArchitecture<
            NumericBackend,
            ReadoutState,
            Error = Error,
        > + 'static,
    A::InputPartPlan: 'static,
    D: eredu_runtime::ReplicatedTextExecutionStrategy<
        eredu_architectures::composite_execution::PreparedCompositeArchitecture<A>,
        NumericBackend,
        ReadoutState,
        NumericReplicatedPolicy<A::Unit>,
        NumericReplicatedPolicy<A::Unit>,
    >,
{
    if !stepped {
        let admitted = A::admit_prepared_input(admission, input, &NumericInputInspector)
            .map_err(|error| error.to_string())?;
        let outcome = session
            .try_prefill_source_cancellable(
                None,
                Some(admitted.decoder_shape()),
                std::num::NonZeroU64::new(chunk),
                |geometry| {
                    eredu_architectures::prefill::PreparedCompositeTextPrefill::from_prepared_text(
                        input,
                        geometry,
                        admission.clone(),
                        NumericInputInspector,
                    )
                },
                &eredu_core::GenerationCancellationToken::new(),
                context,
                &mut eredu_runtime::NoopObserver,
            )
            .map_err(|error| error.to_string())?;
        return match outcome {
            eredu_runtime::replicated_session::PrefillSourceOutcome::Complete(output) => Ok(output),
            _ => Err("ordinary composite text source did not complete".into()),
        };
    }
    parallel_boundary::scheduled::<A, D>(
        session,
        admission,
        input,
        context,
        parallel_boundary::BoundarySchedule {
            chunk,
            stepped,
            cancel_after_first: false,
            max_output_tokens: 2,
            transactional: false,
        },
    )?
    .output
    .ok_or_else(|| "composite prefill did not return final scores".into())
}

#[test]
fn composite_text_spans_preserve_decoder_state_and_cached_decodes() {
    for config in [
        dense_muse_partition_fixture(),
        dense_inkling_partition_fixture(),
        conditional_qwen_partition_config(false),
        qwen_vl_partition_config(false),
        routed_muse_partition_fixture(),
        routed_inkling_partition_fixture(),
        conditional_qwen_partition_config(true),
        qwen_vl_partition_config(true),
    ] {
        let artifact = numeric_composite_artifact(&config);
        let inspection =
            eredu_architectures::configuration::inspect_artifact(artifact.path()).unwrap();
        let input = numeric_text_prepared_input(&[1, 3, 2, 4, 1]);
        let execute = |chunk| {
            let context = NumericContext::default();
            let sources = prepared_adapter::prepare(
                &inspection,
                &prepared_adapter::plan(None),
                &prepared_adapter::NumericPreparationProvider { addressable: false },
            )
            .unwrap();
            let run =
                prepared_adapter::composite_with_chunks(sources, &context, &input, None, chunk)
                    .unwrap();
            let projections = context.projections.lock().unwrap();
            let readout = &projections.last().unwrap().0;
            assert_eq!(
                projections
                    .iter()
                    .filter(|(name, _)| name == readout)
                    .map(|(_, shape)| shape[1])
                    .collect::<Vec<_>>(),
                [1, 1, 1],
                "composite text prefill and each decode project one row",
            );
            run
        };
        let reference = execute(None);
        assert!(reference.outputs[0]
            .data
            .iter()
            .any(|value| value.abs() > 1e-7));
        for chunk in [1, 2, 3, 5] {
            for stepped in [false, true] {
                let actual = execute(Some((chunk, stepped)));
                for (actual, expected) in actual.outputs.iter().zip(&reference.outputs) {
                    assert_tensor_close(actual, expected, "composite scheduled text and decode");
                }
                assert_state_exact(
                    &actual.state,
                    &reference.state,
                    reference.state.layout().len(),
                    "composite scheduled text retains all state",
                );
            }
        }
    }
}

#[test]
fn composite_text_spans_cross_tensor_and_pipeline_partitions() {
    for config in [
        dense_muse_partition_fixture(),
        dense_inkling_partition_fixture(),
        conditional_qwen_partition_config(false),
        qwen_vl_partition_config(false),
    ] {
        let inputs = [
            numeric_text_prepared_input(&[1, 3, 2, 4, 1]),
            numeric_text_prepared_input(&[4]),
            numeric_text_prepared_input(&[5]),
        ];
        for topology in [
            ParallelTopology::new(2, 1, 1, 1).unwrap(),
            ParallelTopology::new(1, 2, 1, 1).unwrap(),
            ParallelTopology::new(2, 2, 1, 1).unwrap(),
        ] {
            let (reference, _) = run_numeric_composite_partitions(&config, &inputs, topology);
            let reference: Vec<_> = reference.into_iter().map(Result::unwrap).collect();
            assert!(reference[0][0].data.iter().any(|value| value.abs() > 1e-7));
            for chunks in [(2, false), (3, true)] {
                let (actual, world) = run_numeric_composite_partitions_with_chunks(
                    &config,
                    &inputs,
                    topology,
                    Some(chunks),
                );
                for (rank, (actual, reference)) in actual.into_iter().zip(&reference).enumerate() {
                    let actual = actual.unwrap_or_else(|error| {
                        panic!(
                        "composite {:?}, topology {topology:?}, rank {rank}: {error}; trace={:?}",
                        config["model_type"], world.trace(),
                    )
                    });
                    assert_eq!(actual.len(), reference.len());
                    for (actual, reference) in actual.iter().zip(reference) {
                        assert_tensor_close(
                            actual,
                            reference,
                            "partitioned composite chunk/decode parity",
                        );
                    }
                }
            }
        }
    }
}
