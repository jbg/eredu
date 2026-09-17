use super::*;
use eredu_architectures::{
    llama,
    prepared_execution::{BorrowedTextSamplingWorkspace, PreparedExecutionError},
    prepared_sources::PreparedModelSources,
};
use eredu_runtime::{ActivationObserver, ResidentRuntime, SharedLayeredObservationPaths};

type State = DeviceState<WorkspaceBackend, WorkspaceResidentLayerState>;
fn model_config() -> serde_json::Value {
    let mut value = config("llama", false);
    value["num_hidden_layers"] = 2.into();
    value
}
fn state_and_paths(
    sources: &PreparedModelSources,
    context: &WorkspaceContext,
    cached: u64,
) -> (State, SharedLayeredObservationPaths) {
    let mut state = WorkspaceResidentStateFactory::new(
        NonZeroU32::new(1).unwrap(),
        NonZeroU32::new(256).unwrap(),
        context,
    )
    .unwrap()
    .realize(sources.selected().text_realization().state().layout())
    .unwrap();
    let architecture = llama::LayeredModel::<WorkspaceBackend>::new(
        llama::model_args_from_config_value(&model_config()).unwrap(),
        context,
    )
    .unwrap();
    let mut runtime =
        ResidentRuntime::<_, WorkspaceBackend, State>::new(architecture, context).unwrap();
    let paths = runtime.prepare_observation_paths().unwrap();
    if cached > 0 {
        let tokens = WorkspaceTensor::full_i32(7, &[1, cached as i32], context).unwrap();
        runtime
            .forward(
                llama::LayeredInput {
                    tokens: &tokens,
                    mask: None,
                },
                &mut state,
                context,
            )
            .unwrap();
    }
    (state, paths.source().clone())
}
fn request(outputs: u64) -> InferenceGeometry {
    InferenceGeometry {
        batch_size: 1,
        cached_positions: 4,
        input_positions: 5,
        max_output_tokens: outputs,
        prefill_chunk_positions: 5,
        output: OutputDemand::Sequence,
    }
}
fn config_sampling(adaptive: bool) -> eredu_core::TextGenerationConfig {
    let config = eredu_core::TextGenerationConfig::new(
        eredu_core::resolve_generation_config(
            None,
            eredu_core::GenerationConfigOverrides {
                temperature: Some(0.7),
                top_k: Some(7),
                repetition_penalty: Some(1.2),
                repeat_last_n: Some(-1),
                ..Default::default()
            },
        )
        .unwrap(),
    );
    if adaptive {
        config.with_mirostat_v2(3.7, 0.23).unwrap()
    } else {
        config
    }
}
struct Observer {
    context: WorkspaceContext,
    active: Vec<u64>,
    spans: Vec<(u64, u64)>,
    current: u64,
    logits: Vec<(u64, Vec<i32>)>,
    boundaries: Vec<(String, usize)>,
    retained: Vec<WorkspaceTensor>,
    capture: bool,
    release: bool,
    full: bool,
    ended: usize,
    failed: bool,
}
impl Observer {
    fn new(context: &WorkspaceContext, active: Vec<u64>) -> Self {
        Self {
            context: context.clone(),
            active,
            spans: Vec::new(),
            current: 0,
            logits: Vec::new(),
            boundaries: Vec::new(),
            retained: Vec::new(),
            capture: false,
            release: false,
            full: true,
            ended: 0,
            failed: false,
        }
    }
}
impl ActivationObserver<WorkspaceTensor, Error> for Observer {
    fn requires_sequence_readout(&self) -> bool {
        self.full
    }
    fn observe(&mut self, path: &str, value: &WorkspaceTensor) -> Result<(), Error> {
        self.context.validate_values([value])?;
        self.boundaries
            .push((path.to_owned(), path.as_ptr() as usize));
        if path == eredu_core::MODEL_LOGITS_OBSERVATION_PATH {
            self.logits.push((self.current, value.shape().to_vec()));
            if self.failed {
                return Err(Error::backend_source(std::io::Error::new(
                    std::io::ErrorKind::PermissionDenied,
                    "observed equation sentinel",
                )));
            }
            if self.capture {
                self.retained
                    .push(value.broadcast_to(&[1, 1024, 17], &self.context)?);
            }
        }
        Ok(())
    }
    fn observe_generated(
        &mut self,
        _: &str,
        _: &WorkspaceTensor,
        _: &eredu_core::capture::GeneratedCaptureSource,
        _: &mut dyn FnMut() -> Result<WorkspaceTensor, Error>,
    ) -> Result<(), Error> {
        Ok(())
    }
}
impl InferenceWorkspaceObserver for Observer {
    fn begin_span(
        &mut self,
        geometry: InferenceGeometry,
        span: &InferenceWorkspaceSpan,
        prediction: u64,
        context: &WorkspaceContext,
    ) -> Result<bool, Error> {
        assert!(prediction < geometry.max_output_tokens);
        let position = match span {
            InferenceWorkspaceSpan::Prefill(chunk) => chunk.position,
            InferenceWorkspaceSpan::Decode { position, .. } => *position,
        };
        context.validate_values(self.retained.iter())?;
        self.spans.push((prediction, position));
        self.current = prediction;
        Ok(self.active.contains(&prediction))
    }
    fn visit_retained(&self, visitor: &mut dyn FnMut(&WorkspaceTensor)) {
        for value in &self.retained {
            visitor(value)
        }
    }
    fn end_span(&mut self, _: &InferenceWorkspaceSpan, _: &WorkspaceContext) -> Result<(), Error> {
        self.ended += 1;
        if self.release {
            self.retained.clear();
        }
        Ok(())
    }
}
fn inner_error(error: &PreparedExecutionError<Error>) -> &Error {
    match error {
        PreparedExecutionError::Backend(error) => error,
        other => panic!("typed backend cause expected: {other}"),
    }
}
fn cause<'a, T: std::error::Error + 'static>(
    mut error: &'a (dyn std::error::Error + 'static),
) -> Option<&'a T> {
    loop {
        if let Some(value) = error.downcast_ref::<T>() {
            return Some(value);
        }
        error = error.source()?;
    }
}

#[test]
fn actual_resident_and_layerwise_equations_bind_shared_paths_map_origin_and_keep_full_logits() {
    let (artifact, values) = prepared_adapter::payload_fixture_config(&model_config(), 1.0);
    assert!(values
        .values()
        .any(|(_, bits)| bits.iter().any(|bits| f32::from_bits(*bits) != 0.0)));
    let inspection = eredu_architectures::configuration::inspect_artifact(artifact.path()).unwrap();
    let residencies = [
        None,
        Some(eredu_core::ResidencyPlan::LayerwiseHost {
            device_layer_window: 1,
            device_budget_bytes: None,
            host_budget_bytes: None,
        }),
        Some(eredu_core::ResidencyPlan::DenseDiskStream {
            device_budget_bytes: 16 << 20,
            host_budget_bytes: 32 << 20,
            host_lookahead: 1,
            background_queue: 1,
        }),
    ];
    for residency in residencies {
        let mut plan = prepared_adapter::plan(None);
        if let Some(residency) = residency {
            plan = plan.with_residency(residency);
        }
        let sources = prepared_adapter::prepare(
            &inspection,
            &plan,
            &prepared_adapter::NumericPreparationProvider { addressable: false },
        )
        .unwrap();
        let layerwise = !matches!(
            sources.selected().text_realization().residency(),
            eredu_runtime::LayerWeightResidency::FullyResident
        );
        let provider = layerwise::SelectedParameters::new(&sources);
        let before = sources.target().source_diagnostics().unwrap();
        for outputs in [0, 1, 3] {
            let facts = Facts::default();
            let context = WorkspaceContext::new(facts.clone());
            let (state, paths) = state_and_paths(&sources, &context, 4);
            let state_before = state
                .as_ref()
                .iter()
                .map(|lane| lane.position())
                .collect::<Vec<_>>();
            let mut observer = Observer::new(&context, vec![0, 2]);
            let geometry = request(outputs);
            let observed = if layerwise {
                sources
                    .inference_blueprint()
                    .quote_replicated_layerwise_text_observed(
                        geometry,
                        &state,
                        &context,
                        &provider,
                        &paths,
                        &mut observer,
                    )
            } else {
                sources
                    .inference_blueprint()
                    .quote_replicated_resident_text_observed(
                        geometry,
                        &state,
                        &context,
                        &paths,
                        &mut observer,
                    )
            }
            .unwrap();
            let plain = if layerwise {
                sources
                    .inference_blueprint()
                    .quote_replicated_layerwise_text(geometry, &state, &context, &provider)
            } else {
                sources
                    .inference_blueprint()
                    .quote_replicated_resident_text(geometry, &state, &context)
            }
            .unwrap();
            assert_eq!(observed.geometry(), geometry);
            assert_eq!(observed.completed_spans(), 1 + outputs);
            assert_eq!(
                observed.tensor_transient_peak_bytes(),
                plain.tensor_transient_peak_bytes()
            );
            assert_eq!(observed.retained_peak_bytes(), plain.retained_peak_bytes());
            assert_eq!(
                observer.spans,
                match outputs {
                    0 => vec![],
                    1 => vec![(0, 4)],
                    _ => vec![(0, 4), (1, 9), (2, 10)],
                }
            );
            assert_eq!(
                observer.logits,
                match outputs {
                    0 => vec![],
                    1 => vec![(0, vec![1, 5, 17])],
                    _ => vec![(0, vec![1, 5, 17]), (2, vec![1, 1, 17])],
                }
            );
            assert_eq!(observer.ended, observer.logits.len());
            assert_eq!(
                state
                    .as_ref()
                    .iter()
                    .map(|lane| lane.position())
                    .collect::<Vec<_>>(),
                state_before
            );
            if outputs > 0 {
                assert!(observed.host_peak_bytes().unwrap() > plain.host_peak_bytes().unwrap());
                for index in 0..paths.unit_count(0).unwrap() {
                    for path in [
                        paths.unit_paths(0, index).unwrap().0,
                        paths.unit_paths(0, index).unwrap().1,
                    ] {
                        let points = observer
                            .boundaries
                            .iter()
                            .filter(|(name, _)| name == path)
                            .collect::<Vec<_>>();
                        assert_eq!(points.len(), observer.logits.len());
                        assert!(points
                            .iter()
                            .all(|(_, pointer)| *pointer == path.as_ptr() as usize));
                    }
                }
            }
        }
        assert_eq!(before, sources.target().source_diagnostics().unwrap());
    }
}

#[test]
fn observed_configured_and_populated_sampling_index_after_full_sequence_without_mutating_source() {
    let (artifact, _) = prepared_adapter::payload_fixture_config(&model_config(), 1.0);
    let inspection = eredu_architectures::configuration::inspect_artifact(artifact.path()).unwrap();
    let sources = prepared_adapter::prepare(
        &inspection,
        &prepared_adapter::plan(None),
        &prepared_adapter::NumericPreparationProvider { addressable: false },
    )
    .unwrap();
    for adaptive in [false, true] {
        let facts = Facts::default();
        let context = WorkspaceContext::new(facts.clone());
        let (state, paths) = state_and_paths(&sources, &context, 4);
        let config = config_sampling(adaptive);
        let mut sampler = eredu_runtime::ConfiguredTextSampler::from_config(config).unwrap();
        for token in [3, 5, 7, 11, 13] {
            match &mut sampler {
                eredu_runtime::ConfiguredTextSampler::Standard(s) => s.accept_token(token),
                eredu_runtime::ConfiguredTextSampler::MirostatV2(s) => {
                    s.accept_token(token, 0.2).unwrap()
                }
            }
        }
        let before = format!("{sampler:?}");
        let key = WorkspaceSamplingRandomState::from_key(
            WorkspaceTensor::existing_with_storage(
                WorkspaceLayout::new(&[2], WorkspaceDtype::Uint32).unwrap(),
                &WorkspaceExistingStorage::new(Some(1024), &context),
                &context,
            )
            .unwrap(),
        )
        .unwrap();
        for borrowed in [false, true] {
            let mut observer = Observer::new(&context, vec![0, 1, 2]);
            let report = if borrowed {
                sources
                    .inference_blueprint()
                    .quote_replicated_resident_text_with_existing_sampling_observed(
                        request(3),
                        &state,
                        &context,
                        BorrowedTextSamplingWorkspace::new(
                            &sampler,
                            0.7,
                            Some(&key),
                            &TokenFilter::All,
                        ),
                        &paths,
                        &mut observer,
                    )
            } else {
                sources
                    .inference_blueprint()
                    .quote_replicated_resident_text_with_sampling_observed(
                        request(3),
                        &state,
                        &context,
                        config,
                        &TokenFilter::All,
                        &paths,
                        &mut observer,
                    )
            }
            .unwrap();
            assert_eq!(report.sampling.steps, 3);
            assert_eq!(report.sampling.output_width, 17);
            assert!(report.sampling.peak.bytes().is_some());
            assert_eq!(
                observer.logits,
                [
                    (0, vec![1, 5, 17]),
                    (1, vec![1, 1, 17]),
                    (2, vec![1, 1, 17])
                ]
            );
        }
        assert_eq!(format!("{sampler:?}"), before);
        assert_eq!(sampler.history_len(), 5);
        assert_eq!(sampler.history_capacity(), 8);
        assert!(facts.operations.lock().unwrap().iter().any(|op| matches!(
            &op.kind,
            WorkspaceOperationKind::Index { selected_axes: 1 }
        ) && op
            .inputs
            .first()
            .is_some_and(|value| value.shape() == [1, 5, 17])
            && op
                .outputs
                .first()
                .is_some_and(|value| value.shape() == [1, 17])));
    }
}

#[test]
fn prior_native_capture_roots_retire_only_at_the_declared_completed_span_boundary() {
    let (artifact, _) = prepared_adapter::payload_fixture_config(&model_config(), 1.0);
    let inspection = eredu_architectures::configuration::inspect_artifact(artifact.path()).unwrap();
    let sources = prepared_adapter::prepare(
        &inspection,
        &prepared_adapter::plan(None),
        &prepared_adapter::NumericPreparationProvider { addressable: false },
    )
    .unwrap();
    let mut peaks = Vec::new();
    for release in [true, false] {
        let context = WorkspaceContext::new(Facts::default());
        let (state, paths) = state_and_paths(&sources, &context, 4);
        let mut observer = Observer::new(&context, vec![1, 2, 3]);
        observer.capture = true;
        observer.release = release;
        let report = sources
            .inference_blueprint()
            .quote_replicated_resident_text_observed(
                request(4),
                &state,
                &context,
                &paths,
                &mut observer,
            )
            .unwrap();
        assert_eq!(observer.ended, 3);
        assert_eq!(observer.retained.len(), if release { 0 } else { 3 });
        assert!(
            report.retained_peak_bytes().unwrap() < 1024 * 17 * 4,
            "capture output cannot become decoder state"
        );
        peaks.push(report.transient().bytes().unwrap());
    }
    assert!(
        peaks[1] > peaks[0] + 1024 * 17 * 4,
        "two earlier live outputs coexist with the next span"
    );
}

#[test]
fn incompatible_geometry_foreign_retained_sources_and_typed_callback_failure_remain_explicit() {
    let (artifact, _) = prepared_adapter::payload_fixture_config(&model_config(), 1.0);
    let inspection = eredu_architectures::configuration::inspect_artifact(artifact.path()).unwrap();
    let sources = prepared_adapter::prepare(
        &inspection,
        &prepared_adapter::plan(None),
        &prepared_adapter::NumericPreparationProvider { addressable: false },
    )
    .unwrap();
    let facts = Facts::default();
    let context = WorkspaceContext::new(facts.clone());
    let (state, paths) = state_and_paths(&sources, &context, 4);
    let mut observer = Observer::new(&context, vec![0]);
    let error = sources
        .inference_blueprint()
        .quote_replicated_resident_text_observed(
            InferenceGeometry {
                output: OutputDemand::LastPosition,
                ..request(3)
            },
            &state,
            &context,
            &paths,
            &mut observer,
        )
        .unwrap_err();
    assert!(matches!(
        cause::<InferenceObservationError>(inner_error(&error)),
        Some(InferenceObservationError::SequenceReadoutRequired)
    ));
    assert!(observer.spans.is_empty());
    let error = sources
        .inference_blueprint()
        .quote_replicated_resident_text_observed(
            InferenceGeometry {
                prefill_chunk_positions: 2,
                ..request(3)
            },
            &state,
            &context,
            &paths,
            &mut observer,
        )
        .unwrap_err();
    assert!(matches!(
        cause::<InferenceObservationError>(inner_error(&error)),
        Some(InferenceObservationError::PartialPrefill)
    ));
    assert!(observer.logits.is_empty());
    observer = Observer::new(&context, vec![0]);
    observer.failed = true;
    let error = sources
        .inference_blueprint()
        .quote_replicated_resident_text_observed(
            request(3),
            &state,
            &context,
            &paths,
            &mut observer,
        )
        .unwrap_err();
    assert!(cause::<std::io::Error>(inner_error(&error)).is_some());
    assert_eq!(observer.ended, 0);
    let foreign = WorkspaceContext::new(Facts::default());
    observer = Observer::new(&context, vec![0]);
    observer.retained.push(
        WorkspaceTensor::existing(
            WorkspaceLayout::new(&[1], WorkspaceDtype::Float32).unwrap(),
            &foreign,
        )
        .unwrap(),
    );
    assert!(sources
        .inference_blueprint()
        .quote_replicated_resident_text_observed(
            request(3),
            &state,
            &context,
            &paths,
            &mut observer
        )
        .is_err());
    assert!(observer.logits.is_empty());
    assert!(state.as_ref().iter().all(|lane| lane.position() == 4));
    #[derive(Debug)]
    struct MissingFacts {
        base: Facts,
        omit_projection: bool,
    }
    impl WorkspaceMechanisms for MissingFacts {
        fn operation_bound(
            &self,
            operation: &WorkspaceOperation,
        ) -> Result<Option<WorkspaceOperationBound>, Error> {
            if self.omit_projection
                && matches!(operation.kind, WorkspaceOperationKind::Projection(_))
            {
                return Ok(None);
            }
            self.base.operation_bound(operation)
        }
        fn host_workspace_bound(
            &self,
            operation: &WorkspaceOperation,
        ) -> Result<Option<WorkspaceHostBound>, Error> {
            self.base.host_workspace_bound(operation)
        }
    }
    for (omit_host, omit_projection) in [(false, false), (true, false), (false, true)] {
        let context = WorkspaceContext::new(MissingFacts {
            base: Facts {
                omit_host,
                ..Default::default()
            },
            omit_projection,
        });
        let (state, paths) = state_and_paths(&sources, &context, 4);
        let mut observer = Observer::new(&context, vec![0, 1, 2]);
        let report = sources
            .inference_blueprint()
            .quote_replicated_resident_text_observed(
                request(3),
                &state,
                &context,
                &paths,
                &mut observer,
            )
            .unwrap();
        assert_eq!(
            report.transient().bytes().is_none(),
            omit_host || omit_projection
        );
    }
}

#[path = "observed/causal.rs"]
mod causal;
