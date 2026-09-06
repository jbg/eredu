#[test]
fn complete_qwen3_vl_variants_accept_paged_cache() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    for (expected_effective_model_type, moe) in
        [("qwen3_vl_text", false), ("qwen3_vl_moe_text", true)]
    {
        let checkpoint = tempfile::tempdir().unwrap();
        write_qwen3_vl_fixture(checkpoint.path(), moe);
        let backend = crate::native::backend(&stream, &stream);
        let paged = PagedCacheOptions::new(1, 32768, 32768, 1)
            .unwrap()
            .with_full_attention(true);
        let request = MlxLoadRequest::from_normalized(
            eredu_runtime::NormalizedLoadRequest::default()
                .with_state_residency(CacheResidencyPolicy::Paged(paged)),
        );
        let model = load_model(&backend, checkpoint.path(), request).unwrap();
        assert_eq!(model.effective_model_type(), expected_effective_model_type);
        let runtime = ModelRuntime::from_prepared(backend, model).unwrap();
        assert_eq!(
            runtime.session().effective_model_type(),
            expected_effective_model_type
        );
        assert_eq!(
            <MlxBackend<'_> as eredu_core::ModelCapabilityBackend>::model_capabilities(&runtime)
                .unwrap()
                .effective_model_type,
            expected_effective_model_type
        );

        assert!(runtime
            .session()
            .cache_residency_report()
            .unwrap()
            .is_some());
    }
}

#[test]
fn complete_gemma4_preserves_nested_effective_model_type() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let checkpoint = tempfile::tempdir().unwrap();
    write_gemma_fixture(checkpoint.path());
    let backend = crate::native::backend(&stream, &stream);
    let model = load_model(&backend, checkpoint.path(), MlxLoadRequest::default()).unwrap();
    assert_eq!(model.effective_model_type(), "gemma4_text");
    let runtime = ModelRuntime::from_prepared(backend, model).unwrap();
    assert_eq!(runtime.session().effective_model_type(), "gemma4_text");
    assert_eq!(
        <MlxBackend<'_> as eredu_core::ModelCapabilityBackend>::model_capabilities(&runtime)
            .unwrap()
            .effective_model_type,
        "gemma4_text"
    );
}

#[test]
fn public_replicated_prediction_variants_install_only_the_neutral_extension() {
    fn assert_extension(checkpoint: &std::path::Path, expected_depth: usize) {
        crate::tests::support::path_instrumentation::reset();
        let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let backend = crate::native::backend(&stream, &stream);
        let model = load_model(&backend, checkpoint, MlxLoadRequest::default())
            .unwrap()
            .into_inner();
        assert_eq!(
            crate::tests::support::path_instrumentation::snapshot().constructors,
            1
        );
        let mut session = MlxModelSession::from_model(
            model,
            eredu_core::SessionCapabilities::new(true, true, true),
        )
        .unwrap();
        assert_eq!(
            session.speculative_capability(),
            SpeculativeCapability::Ready {
                draft_source: eredu_core::SpeculativeDraftSource::Embedded,
            }
        );
        assert!(has_selected_embedded_prediction(&mut session));
        let target = session.neutral_prediction_target_mut().unwrap();
        assert!(target.has_embedded_prediction());
        let _ = expected_depth;
    }

    let deepseek = tempfile::tempdir().unwrap();
    write_deepseek_fixture_with_prediction(deepseek.path(), 2, 1);
    assert_extension(deepseek.path(), 1);

    let inkling = tempfile::tempdir().unwrap();
    write_inkling_mtp_fixture(inkling.path());
    assert_extension(inkling.path(), 2);

    let qwen = tempfile::tempdir().unwrap();
    write_qwen35_multimodal_fixture(qwen.path(), false);
    assert_extension(qwen.path(), 1);

    let nemotron = tempfile::tempdir().unwrap();
    write_nemotron_mtp_fixture(nemotron.path());
    assert_extension(nemotron.path(), 1);
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "requires a local MLX Metal device"]
fn public_deepseek_v3_embedded_scheduler_executes_on_metal() {
    let checkpoint = tempfile::tempdir().unwrap();
    write_deepseek_fixture_with_prediction(checkpoint.path(), 2, 1);
    let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
    let weights_stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let backend = crate::native::backend(&stream, &weights_stream);
    let model = load_model(&backend, checkpoint.path(), MlxLoadRequest::default()).unwrap();
    let mut runtime = ModelRuntime::from_prepared(backend, model).unwrap();
    assert_eq!(
        runtime.session().speculative_capability(),
        SpeculativeCapability::Ready {
            draft_source: eredu_core::SpeculativeDraftSource::Embedded,
        }
    );

    let prompt = Array::from_slice(&[1u32, 2], &[1, 2]);
    let parts = [text_input_part(&prompt)];
    let output = run_neutral_embedded_mtp(
        &mut runtime,
        synthetic_prediction_input(&parts, &[1, 2]),
        SpeculativeConfig {
            max_tokens: 3,
            max_draft_tokens: 1,
            temperature: 0.0,
            eos_token_ids: Vec::new(),
        },
    )
    .unwrap();
    assert_eq!(output.token_ids().len(), 3);
    assert!(output.stats().draft_tokens() > 0);
}

#[test]
fn synthetic_prediction_input_binds_prepared_shape_and_token_content() {
    let first_tokens = [1_u32, 2];
    let first = Array::from_slice(&first_tokens, &[1, 2]);
    let first_parts = [text_input_part(&first)];
    let first = synthetic_prediction_input(&first_parts, &first_tokens);
    let same = synthetic_prediction_input(&first_parts, &first_tokens);
    let changed = synthetic_prediction_input(&first_parts, &[2, 1]);
    let reshaped_tokens = Array::from_slice(&first_tokens, &[2, 1]);
    let reshaped_parts = [text_input_part(&reshaped_tokens)];
    let reshaped = synthetic_prediction_input(&reshaped_parts, &first_tokens);

    assert_eq!(first.cache_identity(), same.cache_identity());
    assert_ne!(first.cache_identity(), changed.cache_identity());
    assert_ne!(first.cache_identity(), reshaped.cache_identity());
    assert_eq!(
        first
            .cache_identity()
            .expect("synthetic input identity")
            .semantic_content_fingerprint(),
        eredu_core::cache::prompt_cache_token_fingerprint(&first_tokens)
    );
}

#[derive(Default)]
struct ExternalObservationTrace {
    counts: BTreeMap<String, usize>,
    proposal_logits: Vec<Vec<f32>>,
}

impl ExternalObservationTrace {
    fn record(&mut self, path: &str) {
        *self.counts.entry(path.to_owned()).or_default() += 1;
    }

    fn count(&self, path: &str) -> usize {
        self.counts.get(path).copied().unwrap_or_default()
    }
}

#[derive(Clone, Copy)]
enum ExternalTensorIntervention {
    None,
    Zero,
    ForceToken(usize),
    Fail,
}

fn forced_token_logits(value: &Array, token: usize) -> Result<Array, safemlx::error::Exception> {
    let width = value
        .shape()
        .last()
        .copied()
        .and_then(|width| usize::try_from(width).ok())
        .ok_or_else(|| safemlx::error::Exception::custom("external logits have no vocabulary"))?;
    if token >= width {
        return Err(safemlx::error::Exception::custom(
            "forced external token is outside the vocabulary",
        ));
    }
    let mut values = vec![-100.0_f32; value.size()];
    for row in values.chunks_exact_mut(width) {
        row[token] = 100.0;
    }
    Ok(Array::from_slice(&values, value.shape()))
}

struct ExternalTensorObserver {
    trace: Arc<Mutex<ExternalObservationTrace>>,
    path: Option<&'static str>,
    intervention: ExternalTensorIntervention,
    stream: Stream,
}

impl eredu_runtime::ActivationObserver<MlxTensor, safemlx::error::Exception>
    for ExternalTensorObserver
{
    fn observe(&mut self, path: &str, _value: &MlxTensor) -> Result<(), safemlx::error::Exception> {
        self.trace.lock().unwrap().record(path);
        if self.path == Some(path) && matches!(self.intervention, ExternalTensorIntervention::Fail)
        {
            return Err(safemlx::error::Exception::custom(
                "injected external observation failure",
            ));
        }
        Ok(())
    }

    fn intervene(
        &mut self,
        path: &str,
        value: &MlxTensor,
    ) -> Result<Option<MlxTensor>, safemlx::error::Exception> {
        if self.path != Some(path) {
            return Ok(None);
        }
        match self.intervention {
            ExternalTensorIntervention::None | ExternalTensorIntervention::Fail => Ok(None),
            ExternalTensorIntervention::Zero => Ok(Some(MlxTensor::from_array(
                safemlx::ops::zeros_like(value.as_array(), &self.stream)?,
            ))),
            ExternalTensorIntervention::ForceToken(token) => Ok(Some(MlxTensor::from_array(
                forced_token_logits(value.as_array(), token)?,
            ))),
        }
    }
}

struct ExternalLogitsObserver {
    trace: Arc<Mutex<ExternalObservationTrace>>,
    intervention: ExternalTensorIntervention,
}

impl eredu_runtime::ActivationObserver<Array, safemlx::error::Exception>
    for ExternalLogitsObserver
{
    fn observe(&mut self, path: &str, value: &Array) -> Result<(), safemlx::error::Exception> {
        let mut trace = self.trace.lock().unwrap();
        trace.record(path);
        if path
            == eredu_architectures::external_assistant::EXTERNAL_ASSISTANT_PROPOSAL_LOGITS_OBSERVATION_PATH
        {
            let evaluated = value.evaluated()?;
            trace.proposal_logits.push(
                evaluated
                    .try_to_vec::<f32>()
                    .map_err(|error| safemlx::error::Exception::custom(error.to_string()))?,
            );
            if matches!(self.intervention, ExternalTensorIntervention::Fail) {
                return Err(safemlx::error::Exception::custom(
                    "injected external proposal observation failure",
                ));
            }
        }
        Ok(())
    }

    fn intervene(
        &mut self,
        path: &str,
        value: &Array,
    ) -> Result<Option<Array>, safemlx::error::Exception> {
        if path
            != eredu_architectures::external_assistant::EXTERNAL_ASSISTANT_PROPOSAL_LOGITS_OBSERVATION_PATH
        {
            return Ok(None);
        }
        match self.intervention {
            ExternalTensorIntervention::ForceToken(token) => {
                Ok(Some(forced_token_logits(value, token)?))
            }
            ExternalTensorIntervention::Zero => Ok(Some(Array::from_slice(
                &vec![0.0_f32; value.size()],
                value.shape(),
            ))),
            ExternalTensorIntervention::None | ExternalTensorIntervention::Fail => Ok(None),
        }
    }
}

struct EmbeddedLogitsObserver {
    trace: Arc<Mutex<ExternalObservationTrace>>,
    intervention: ExternalTensorIntervention,
}

impl eredu_runtime::ActivationObserver<Array, safemlx::error::Exception>
    for EmbeddedLogitsObserver
{
    fn observe(&mut self, path: &str, value: &Array) -> Result<(), safemlx::error::Exception> {
        let mut trace = self.trace.lock().unwrap();
        trace.record(path);
        if path == eredu_architectures::speculative_execution::EMBEDDED_PROPOSAL_LOGITS_PATH {
            let evaluated = value.evaluated()?;
            trace.proposal_logits.push(
                evaluated
                    .try_to_vec::<f32>()
                    .map_err(|error| safemlx::error::Exception::custom(error.to_string()))?,
            );
            if matches!(self.intervention, ExternalTensorIntervention::Fail) {
                return Err(safemlx::error::Exception::custom(
                    "injected embedded proposal observation failure",
                ));
            }
        }
        Ok(())
    }

    fn intervene(
        &mut self,
        path: &str,
        value: &Array,
    ) -> Result<Option<Array>, safemlx::error::Exception> {
        if path != eredu_architectures::speculative_execution::EMBEDDED_PROPOSAL_LOGITS_PATH {
            return Ok(None);
        }
        match self.intervention {
            ExternalTensorIntervention::ForceToken(token) => {
                Ok(Some(forced_token_logits(value, token)?))
            }
            ExternalTensorIntervention::Zero => Ok(Some(Array::from_slice(
                &vec![0.0_f32; value.size()],
                value.shape(),
            ))),
            ExternalTensorIntervention::None | ExternalTensorIntervention::Fail => Ok(None),
        }
    }
}

#[test]
fn public_embedded_observers_are_installed_causally_and_transactionally() {
    const TARGET_CAPTURE: &str =
        eredu_architectures::speculative_execution::EMBEDDED_TARGET_CAPTURE_PATH;
    const PREDICTION_OUTPUT: &str =
        eredu_architectures::speculative_execution::EMBEDDED_PREDICTION_OUTPUT_PATH;
    const PROPOSAL_LOGITS: &str =
        eredu_architectures::speculative_execution::EMBEDDED_PROPOSAL_LOGITS_PATH;
    const VERIFICATION_LOGITS: &str =
        eredu_architectures::speculative_execution::EMBEDDED_VERIFICATION_LOGITS_PATH;

    fn run(
        checkpoint: &Path,
        tensor_path: Option<&'static str>,
        tensor_intervention: ExternalTensorIntervention,
        logits_intervention: ExternalTensorIntervention,
    ) -> (
        Result<eredu_core::SpeculativeGenerationOutput, String>,
        usize,
        Arc<Mutex<ExternalObservationTrace>>,
    ) {
        let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let backend = crate::native::backend(&stream, &stream);
        let model = load_model(&backend, checkpoint, MlxLoadRequest::default()).unwrap();
        let mut runtime = ModelRuntime::from_prepared(backend, model).unwrap();
        let trace = Arc::new(Mutex::new(ExternalObservationTrace::default()));
        runtime
            .session_mut()
            .install_embedded_prediction_observers(
                ExternalTensorObserver {
                    trace: Arc::clone(&trace),
                    path: tensor_path,
                    intervention: tensor_intervention,
                    stream: stream.clone(),
                },
                EmbeddedLogitsObserver {
                    trace: Arc::clone(&trace),
                    intervention: logits_intervention,
                },
            )
            .unwrap();
        let tokens = [1_u32, 2];
        let prompt = Array::from_slice(&tokens, &[1, 2]);
        let parts = [text_input_part(&prompt)];
        let (result, publications) = execute_neutral_embedded_mtp(
            &mut runtime,
            synthetic_prediction_input(&parts, &tokens),
            SpeculativeConfig {
                max_tokens: 3,
                max_draft_tokens: 2,
                temperature: 0.0,
                eos_token_ids: Vec::new(),
            },
        );
        (
            result.map_err(|error| error.to_string()),
            publications,
            trace,
        )
    }

    let checkpoint = tempfile::tempdir().unwrap();
    write_inkling_mtp_fixture(checkpoint.path());
    let (baseline, publications, trace) = run(
        checkpoint.path(),
        None,
        ExternalTensorIntervention::None,
        ExternalTensorIntervention::None,
    );
    let baseline = baseline.unwrap();
    assert!(publications > 0);
    let rounds = baseline.stats().rounds();
    let drafts = baseline.stats().draft_tokens();
    let trace = trace.lock().unwrap();
    assert_eq!(trace.count(TARGET_CAPTURE), rounds + 1);
    assert_eq!(trace.count(PREDICTION_OUTPUT), drafts);
    assert_eq!(trace.count(PROPOSAL_LOGITS), drafts);
    assert_eq!(trace.count(VERIFICATION_LOGITS), rounds);
    drop(trace);

    let (proposal, _, _) = run(
        checkpoint.path(),
        None,
        ExternalTensorIntervention::None,
        ExternalTensorIntervention::ForceToken(7),
    );
    let proposal = proposal.unwrap();
    assert!(
        proposal.token_ids() != baseline.token_ids()
            || proposal.stats().accept_lens() != baseline.stats().accept_lens(),
        "proposal intervention did not reach verification and commit",
    );

    let (verification, _, _) = run(
        checkpoint.path(),
        Some(VERIFICATION_LOGITS),
        ExternalTensorIntervention::ForceToken(6),
        ExternalTensorIntervention::None,
    );
    assert_ne!(verification.unwrap().token_ids(), baseline.token_ids());

    let (failed, failed_publications, failed_trace) = run(
        checkpoint.path(),
        Some(TARGET_CAPTURE),
        ExternalTensorIntervention::Fail,
        ExternalTensorIntervention::None,
    );
    let error = match failed {
        Ok(_) => panic!("injected embedded observation failure unexpectedly succeeded"),
        Err(error) => error,
    };
    assert!(error.contains("injected external observation failure"));
    assert_eq!(failed_publications, 0);
    let failed_trace = failed_trace.lock().unwrap();
    assert_eq!(failed_trace.count(TARGET_CAPTURE), 1);
    assert_eq!(failed_trace.count(PREDICTION_OUTPUT), 0);
    assert_eq!(failed_trace.count(PROPOSAL_LOGITS), 0);
    assert_eq!(failed_trace.count(VERIFICATION_LOGITS), 0);
}

fn execute_public_gemma_external_scheduler(
    target: &Path,
    assistant: &Path,
    target_device: DevicePlan,
    placement: DraftPlacementPlan,
    configure: impl FnOnce(&mut crate::composition::mlx::speculative::MlxDrafter),
) -> (
    Result<eredu_core::SpeculativeGenerationBatchOutput, String>,
    usize,
) {
    let plan = ExecutionPlan::fully_resident(target_device).with_drafting(DraftingPlan::External {
        model: assistant.display().to_string(),
        placement,
        max_draft_tokens: 2,
        lookahead: false,
        adaptive_lookahead: false,
    });
    let factory = crate::composition::mlx::automatic::MlxBackendFactory::default();
    let tokenizer_compatibility = TokenizerCompatibilityProof::prove([7; 32], [7; 32]).unwrap();
    let preparation = eredu_architectures::prepare_external_assistant(assistant).unwrap();
    let inspection = eredu_architectures::configuration::inspect_artifact(target).unwrap();
    let selected = eredu_core::select_execution_plan_target(&factory, &plan, inspection).unwrap();
    let external_artifact = eredu_core::select_execution_plan_drafting(
        &factory,
        &plan,
        &selected,
        Some(ExternalDraftArtifact {
            preparation,
            tokenizer_compatibility,
        }),
    )
    .unwrap();
    let realization = eredu_core::realize_execution_plan_target(&factory, &plan, selected).unwrap();
    let mut runtime = realization.into_runtime().unwrap();

    let mut drafting =
        eredu_core::realize_execution_plan_drafting(&factory, &plan, &runtime, external_artifact)
            .unwrap();
    let mut draft = drafting
        .as_speculative_draft()
        .expect("the external plan must realize a reusable assistant");
    match &mut draft {
        SpeculativeDraft::External(drafter) => configure(drafter),
        _ => panic!("the external plan must expose the materialized assistant"),
    }

    let sampling = eredu_core::resolve_generation_config(
        None,
        eredu_core::GenerationConfigOverrides {
            max_new_tokens: Some(3),
            temperature: Some(0.0),
            ..Default::default()
        },
    )
    .unwrap();
    let prompts = [[1u32, 2], [2u32, 1]]
        .into_iter()
        .map(|tokens| {
            let semantic_content_fingerprint =
                eredu_core::cache::prompt_cache_token_fingerprint(&tokens);
            let tokens = Array::from_slice(&tokens, &[1, 2]);
            let parts = [text_input_part(&tokens)];
            MlxModelInput::from(crate::backend::runtime::media::input::ModelInput::new(
                &parts,
            ))
            .with_semantic_content_fingerprint(semantic_content_fingerprint)
            .unwrap()
        })
        .collect::<Vec<_>>();
    let publications = Arc::new(AtomicUsize::new(0));
    let lanes = prompts
        .into_iter()
        .map(|prompt| {
            let publications = publications.clone();
            SpeculativeGenerationLane::new(
                prompt,
                TextGenerationConfig::new(sampling),
                SpeculativeConfig {
                    max_tokens: 3,
                    max_draft_tokens: 2,
                    temperature: 0.0,
                    eos_token_ids: Vec::new(),
                },
                AllowAllTokens,
                Box::<TokenOnlySemanticState>::default(),
                GenerationCancellationToken::new(),
                Box::new(move |_| {
                    publications.fetch_add(1, Ordering::Relaxed);
                }),
            )
        })
        .collect();
    let output = <MlxBackend<'static> as SpeculativeGenerationBackend>::with_speculative_execution(
        &mut runtime,
        SpeculativeGenerationBatchRequest::new(draft, lanes, [0; 32]),
        eredu_runtime::RunSpeculativeGeneration::default(),
    )
    .map_err(|error| error.to_string());
    (output, publications.load(Ordering::Relaxed))
}

fn run_public_gemma_external_scheduler(
    target: &Path,
    assistant: &Path,
    target_device: DevicePlan,
    placement: DraftPlacementPlan,
    expected_topology: SpeculativeExecutionTopology,
) {
    let (output, _) = execute_public_gemma_external_scheduler(
        target,
        assistant,
        target_device,
        placement,
        |_| {},
    );
    let output = output.unwrap();

    assert_eq!(output.scheduler().execution_topology(), expected_topology);
    assert!(output.scheduler().turns() > 0);
    let requests = output.into_requests();
    assert_eq!(requests.len(), 2);
    for request in requests {
        assert_eq!(request.token_ids().len(), 3);
        assert_eq!(request.stats().emitted_tokens(), 3);
        assert_eq!(request.stats().execution_topology(), expected_topology);
        assert!(
            request.stats().draft_tokens() > 0,
            "assistant did no proposal work"
        );
        assert!(
            request.stats().rounds() > 0,
            "target performed no verification"
        );
        assert_eq!(
            request.stats().accept_lens().len(),
            request.stats().rounds(),
            "each verification round must reach its transaction outcome"
        );
        assert!(
            request.stats().target_tokens() > 2,
            "verification must evaluate more than the two-token prefill"
        );
    }
}

#[test]
fn public_gemma_external_factory_scheduler_supports_target_and_cpu_split_placement() {
    let target = tempfile::tempdir().unwrap();
    let assistant = tempfile::tempdir().unwrap();
    write_gemma_fixture(target.path());
    write_gemma_assistant_fixture(assistant.path());

    run_public_gemma_external_scheduler(
        target.path(),
        assistant.path(),
        DevicePlan::new("mlx", "cpu:0").unwrap(),
        DraftPlacementPlan::Target,
        SpeculativeExecutionTopology::Single,
    );
    run_public_gemma_external_scheduler(
        target.path(),
        assistant.path(),
        DevicePlan::new("mlx", "cpu:0").unwrap(),
        DraftPlacementPlan::Device {
            device: DevicePlan::new("mlx", "cpu:0").unwrap(),
        },
        SpeculativeExecutionTopology::SameDeviceSplit,
    );
}

#[test]
fn public_gemma_external_observers_are_causal_exact_and_transactional() {
    const CAPTURE_PATH: &str = "model.language_model.layers.3.output";
    const PROPOSAL_PATH: &str = eredu_architectures::external_assistant::EXTERNAL_ASSISTANT_PROPOSAL_LOGITS_OBSERVATION_PATH;
    const VERIFICATION_PATH: &str = eredu_architectures::external_assistant::EXTERNAL_ASSISTANT_VERIFICATION_LOGITS_OBSERVATION_PATH;

    fn run(
        target: &Path,
        assistant: &Path,
        tensor_path: Option<&'static str>,
        tensor_intervention: ExternalTensorIntervention,
        proposal_intervention: ExternalTensorIntervention,
    ) -> (
        Result<eredu_core::SpeculativeGenerationBatchOutput, String>,
        usize,
        Arc<Mutex<ExternalObservationTrace>>,
    ) {
        let trace = Arc::new(Mutex::new(ExternalObservationTrace::default()));
        let installed = trace.clone();
        let (result, publications) = execute_public_gemma_external_scheduler(
            target,
            assistant,
            DevicePlan::new("mlx", "cpu:0").unwrap(),
            DraftPlacementPlan::Target,
            move |drafter| {
                drafter.install_external_observers(
                    ExternalTensorObserver {
                        trace: installed.clone(),
                        path: tensor_path,
                        intervention: tensor_intervention,
                        stream: Stream::new_with_device(&Device::new(DeviceType::Cpu, 0)),
                    },
                    ExternalLogitsObserver {
                        trace: installed,
                        intervention: proposal_intervention,
                    },
                ).unwrap();
            },
        );
        (result, publications, trace)
    }

    fn tokens(output: &eredu_core::SpeculativeGenerationBatchOutput) -> Vec<Vec<u32>> {
        output
            .requests()
            .iter()
            .map(|request| request.token_ids().to_vec())
            .collect()
    }

    fn accept_lens(output: &eredu_core::SpeculativeGenerationBatchOutput) -> Vec<Vec<usize>> {
        output
            .requests()
            .iter()
            .map(|request| request.stats().accept_lens().to_vec())
            .collect()
    }

    let target = tempfile::tempdir().unwrap();
    let assistant = tempfile::tempdir().unwrap();
    write_gemma_fixture(target.path());
    write_gemma_assistant_fixture(assistant.path());

    let (baseline, baseline_publications, baseline_trace) = run(
        target.path(),
        assistant.path(),
        None,
        ExternalTensorIntervention::None,
        ExternalTensorIntervention::None,
    );
    let baseline = baseline.unwrap();
    assert!(baseline_publications > 0);
    let baseline_tokens = tokens(&baseline);
    let total_drafts = baseline
        .requests()
        .iter()
        .map(|request| request.stats().draft_tokens())
        .sum::<usize>();
    let total_rounds = baseline
        .requests()
        .iter()
        .map(|request| request.stats().rounds())
        .sum::<usize>();
    let baseline_trace = baseline_trace.lock().unwrap();
    assert_eq!(
        baseline_trace.count(eredu_core::MODEL_LOGITS_OBSERVATION_PATH),
        2
    );
    assert_eq!(baseline_trace.count(PROPOSAL_PATH), total_drafts);
    assert!(baseline_trace.count(VERIFICATION_PATH) >= total_rounds);
    assert_eq!(
        baseline_trace.count(CAPTURE_PATH),
        baseline_trace.count(eredu_core::MODEL_LOGITS_OBSERVATION_PATH)
            + baseline_trace.count(VERIFICATION_PATH),
        "each target capture must be observed exactly once for each target forward",
    );
    let baseline_proposal = baseline_trace
        .proposal_logits
        .first()
        .expect("baseline assistant must produce proposal logits")
        .clone();
    drop(baseline_trace);

    let (capture_perturbed, _, capture_trace) = run(
        target.path(),
        assistant.path(),
        Some(CAPTURE_PATH),
        ExternalTensorIntervention::Zero,
        ExternalTensorIntervention::None,
    );
    capture_perturbed.unwrap();
    let capture_trace = capture_trace.lock().unwrap();
    assert_ne!(
        capture_trace
            .proposal_logits
            .first()
            .expect("capture-intervened assistant must produce proposal logits"),
        &baseline_proposal,
        "intervened target capture did not reach assistant proposal computation",
    );
    assert_eq!(
        capture_trace.count(CAPTURE_PATH),
        capture_trace.count(eredu_core::MODEL_LOGITS_OBSERVATION_PATH)
            + capture_trace.count(VERIFICATION_PATH),
    );
    drop(capture_trace);

    let (proposal_perturbed, _, proposal_trace) = run(
        target.path(),
        assistant.path(),
        None,
        ExternalTensorIntervention::None,
        ExternalTensorIntervention::ForceToken(31),
    );
    let proposal_perturbed = proposal_perturbed.unwrap();
    assert!(
        tokens(&proposal_perturbed) != baseline_tokens
            || accept_lens(&proposal_perturbed) != accept_lens(&baseline),
        "forced proposal logits did not affect verification or commit behavior",
    );
    assert_eq!(
        proposal_trace.lock().unwrap().count(PROPOSAL_PATH),
        proposal_perturbed
            .requests()
            .iter()
            .map(|request| request.stats().draft_tokens())
            .sum::<usize>(),
    );

    let (verification_perturbed, _, verification_trace) = run(
        target.path(),
        assistant.path(),
        Some(VERIFICATION_PATH),
        ExternalTensorIntervention::ForceToken(30),
        ExternalTensorIntervention::None,
    );
    let verification_perturbed = verification_perturbed.unwrap();
    assert_ne!(tokens(&verification_perturbed), baseline_tokens);
    assert!(verification_trace.lock().unwrap().count(VERIFICATION_PATH) > 0);

    let (final_perturbed, _, final_trace) = run(
        target.path(),
        assistant.path(),
        Some(eredu_core::MODEL_LOGITS_OBSERVATION_PATH),
        ExternalTensorIntervention::ForceToken(29),
        ExternalTensorIntervention::None,
    );
    let final_perturbed = final_perturbed.unwrap();
    assert!(final_perturbed
        .requests()
        .iter()
        .all(|request| request.token_ids().first() == Some(&29)),);
    assert_eq!(
        final_trace
            .lock()
            .unwrap()
            .count(eredu_core::MODEL_LOGITS_OBSERVATION_PATH),
        2,
    );

    let (failed, failed_publications, failed_trace) = run(
        target.path(),
        assistant.path(),
        Some(CAPTURE_PATH),
        ExternalTensorIntervention::Fail,
        ExternalTensorIntervention::None,
    );
    let error = match failed {
        Ok(_) => panic!("injected external observation failure unexpectedly succeeded"),
        Err(error) => error,
    };
    assert!(error.contains("injected external observation failure"));
    assert_eq!(
        failed_publications, 0,
        "failed observation published output"
    );
    let failed_trace = failed_trace.lock().unwrap();
    assert_eq!(failed_trace.count(CAPTURE_PATH), 1);
    assert_eq!(failed_trace.count(PROPOSAL_PATH), 0);
    assert_eq!(failed_trace.count(VERIFICATION_PATH), 0);
    assert_eq!(
        failed_trace.count(eredu_core::MODEL_LOGITS_OBSERVATION_PATH),
        0
    );

    let (verification_failed, verification_failed_publications, verification_failed_trace) = run(
        target.path(),
        assistant.path(),
        Some(VERIFICATION_PATH),
        ExternalTensorIntervention::Fail,
        ExternalTensorIntervention::None,
    );
    let error = match verification_failed {
        Ok(_) => panic!("injected verification observation failure unexpectedly succeeded"),
        Err(error) => error,
    };
    assert!(error.contains("injected external observation failure"));
    let verification_failed_trace = verification_failed_trace.lock().unwrap();
    assert_eq!(verification_failed_trace.count(VERIFICATION_PATH), 1);
    assert!(verification_failed_trace.count(PROPOSAL_PATH) > 0);
    assert_eq!(
        verification_failed_publications,
        verification_failed_trace.count(eredu_core::MODEL_LOGITS_OBSERVATION_PATH),
        "failed verification published beyond each lane's committed first token",
    );
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "requires local MLX Metal execution"]
fn public_gemma_external_factory_scheduler_supports_metal_target_cpu_assistant() {
    if !safemlx::metal::is_available().unwrap_or(false) {
        eprintln!("skipping Gemma external cross-device proof: MLX Metal is unavailable");
        return;
    }
    let target = tempfile::tempdir().unwrap();
    let assistant = tempfile::tempdir().unwrap();
    write_gemma_fixture(target.path());
    write_gemma_assistant_fixture(assistant.path());

    run_public_gemma_external_scheduler(
        target.path(),
        assistant.path(),
        DevicePlan::new("mlx", "metal:0").unwrap(),
        DraftPlacementPlan::Device {
            device: DevicePlan::new("mlx", "cpu:0").unwrap(),
        },
        SpeculativeExecutionTopology::CrossDeviceSplit,
    );
}

#[test]
fn gemma_external_assistant_capture_uses_neutral_target_and_rolls_back_failure() {
    use eredu_architectures::composite_execution::{
        ExternalPredictionCaptureRequest, ExternalPredictionTargetCapture,
        ExternalPredictionTargetOperation,
    };

    crate::tests::support::path_instrumentation::reset();
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let checkpoint = tempfile::tempdir().unwrap();
    write_gemma_fixture(checkpoint.path());
    let backend = crate::native::backend(&stream, &stream);
    let model = load_model(&backend, checkpoint.path(), MlxLoadRequest::default())
        .unwrap()
        .into_inner();
    assert_eq!(
        crate::tests::support::path_instrumentation::snapshot().constructors,
        1
    );
    let mut session = MlxModelSession::from_model(
        model,
        eredu_core::SessionCapabilities::new(true, true, true),
    )
    .unwrap();
    let target = session
        .neutral_prediction_target_mut()
        .unwrap()
        .external_prediction_mut()
        .expect("Gemma target must expose the external-assistant capability");
    let mut cache = target.prepare_external_prediction_target_cache().unwrap();
    let initial_offset = cache.generation().unwrap();
    let tokens = Array::from_slice(&[1u32, 2], &[1, 2]);
    let parts = [text_input_part(&tokens)];
    let invalid = ExternalPredictionCaptureRequest::Gemma4SharedAttention {
        final_hidden_path: "model.language_model.layers.99.output".into(),
    };
    let error = target
        .prefill_external_prediction_target(
            crate::backend::runtime::media::input::ModelInput::new(&parts),
            &invalid,
            &mut cache,
        )
        .unwrap_err();
    assert!(error.to_string().contains("did not reach capture path"));
    assert_eq!(cache.generation().unwrap(), initial_offset);

    let request = ExternalPredictionCaptureRequest::Gemma4SharedAttention {
        final_hidden_path: "model.language_model.layers.3.output".into(),
    };
    let (logits, capture) = target
        .prefill_external_prediction_target(
            crate::backend::runtime::media::input::ModelInput::new(&parts),
            &request,
            &mut cache,
        )
        .unwrap();
    assert_eq!(logits.as_array().shape(), [1, 2, 32]);
    assert_eq!(cache.generation().unwrap(), 2);
    let ExternalPredictionTargetCapture::Gemma4 { hidden, shared_kv } = capture else {
        panic!("Gemma target returned the wrong external-assistant capture")
    };
    assert_eq!(hidden.as_array().shape(), [1, 2, 8]);
    assert!(!shared_kv.is_empty());
    assert!(shared_kv.iter().all(|(_, keys, values)| {
        keys.as_array().dim(-2) == 2 && values.as_array().dim(-2) == 2
    }));
    let proposal = MlxTensor::from_array(Array::from_slice(&[3u32], &[1, 1]));
    let embedding = target
        .apply_external_prediction_target_operation(
            ExternalPredictionTargetOperation::TokenEmbeddings(&proposal),
        )
        .unwrap();
    assert_eq!(embedding.as_array().shape(), [1, 1, 8]);
}

#[test]
fn muse_external_assistant_capture_uses_neutral_target_in_exact_layer_order() {
    use eredu_architectures::composite_execution::{
        ExternalPredictionCaptureRequest, ExternalPredictionTargetCapture,
        ExternalPredictionTargetOperation,
    };

    crate::tests::support::path_instrumentation::reset();
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let checkpoint = tempfile::tempdir().unwrap();
    write_muse_glimmer_tensor_parallel_fixture(checkpoint.path());
    let backend = crate::native::backend(&stream, &stream);
    let model = load_model(&backend, checkpoint.path(), MlxLoadRequest::default())
        .unwrap()
        .into_inner();
    assert_eq!(
        crate::tests::support::path_instrumentation::snapshot().constructors,
        1
    );
    let mut session = MlxModelSession::from_model(
        model,
        eredu_core::SessionCapabilities::new(true, true, true),
    )
    .unwrap();
    let target = session
        .neutral_prediction_target_mut()
        .unwrap()
        .external_prediction_mut()
        .expect("Muse-Glimmer target must expose the external-assistant capability");
    let mut cache = target.prepare_external_prediction_target_cache().unwrap();
    let initial_offset = cache.generation().unwrap();
    let tokens = Array::from_slice(&[1u32, 2], &[1, 2]);
    let parts = [text_input_part(&tokens)];
    let invalid = ExternalPredictionCaptureRequest::MuseGlimmerDFlash {
        target_layers: vec![0, 1].into_boxed_slice(),
        target_paths: vec!["missing.0".into(), "missing.1".into()].into_boxed_slice(),
    };
    let error = target
        .prefill_external_prediction_target(
            crate::backend::runtime::media::input::ModelInput::new(&parts),
            &invalid,
            &mut cache,
        )
        .unwrap_err();
    assert!(error.to_string().contains("did not reach capture path"));
    assert_eq!(cache.generation().unwrap(), initial_offset);

    let request = ExternalPredictionCaptureRequest::MuseGlimmerDFlash {
        target_layers: vec![0, 1].into_boxed_slice(),
        target_paths: vec![
            "model.layers.0.output".into(),
            "model.layers.1.output".into(),
        ]
        .into_boxed_slice(),
    };
    let (logits, capture) = target
        .prefill_external_prediction_target(
            crate::backend::runtime::media::input::ModelInput::new(&parts),
            &request,
            &mut cache,
        )
        .unwrap();
    assert_eq!(logits.as_array().shape(), [1, 2, 32]);
    assert_eq!(cache.generation().unwrap(), 2);
    let ExternalPredictionTargetCapture::MuseGlimmerDFlash { target_states } = capture else {
        panic!("Muse-Glimmer target returned the wrong external-assistant capture")
    };
    assert_eq!(target_states.len(), 2);
    assert!(target_states
        .iter()
        .all(|state| state.as_array().shape() == [1, 2, 16]));
    let proposal = MlxTensor::from_array(Array::from_slice(&[3u32], &[1, 1]));
    let embedding = target
        .apply_external_prediction_target_operation(
            ExternalPredictionTargetOperation::TokenEmbeddings(&proposal),
        )
        .unwrap();
    assert_eq!(embedding.as_array().shape(), [1, 1, 16]);
    let projected = target
        .apply_external_prediction_target_operation(
            ExternalPredictionTargetOperation::ProjectLogits(&target_states[1]),
        )
        .unwrap();
    assert_eq!(projected.as_array().shape(), [1, 2, 32]);
}

#[test]
fn complete_family_adapters_return_final_output_interventions() {
    fn write_qwen(directory: &Path) {
        write_qwen_fixture(directory, "qwen3");
    }

    let fixtures: [(&str, fn(&Path)); 7] = [
        ("Llama", write_fixture),
        ("Qwen", write_qwen),
        ("DeepSeek", |directory| write_deepseek_fixture(directory, 2)),
        ("Gemma 4", write_gemma4_tensor_parallel_fixture),
        ("Inkling", write_inkling_fixture),
        ("Muse-Glimmer", write_muse_glimmer_tensor_parallel_fixture),
        ("Kimi Linear", write_kimi_linear_fixture),
    ];
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));

    for (family, write_fixture) in fixtures {
        struct ReplacingLogits {
            observed: bool,
        }

        impl eredu_runtime::ActivationObserver<MlxTensor, safemlx::error::Exception> for ReplacingLogits {
            fn observe(
                &mut self,
                path: &str,
                _value: &MlxTensor,
            ) -> Result<(), safemlx::error::Exception> {
                self.observed |= path == eredu_core::MODEL_LOGITS_OBSERVATION_PATH;
                Ok(())
            }

            fn intervene(
                &mut self,
                path: &str,
                value: &MlxTensor,
            ) -> Result<Option<MlxTensor>, safemlx::error::Exception> {
                Ok(
                    (path == eredu_core::MODEL_LOGITS_OBSERVATION_PATH).then(|| {
                        let value = value.as_array();
                        MlxTensor::from_array(Array::from_iter(
                            std::iter::repeat_n(41.0f32, value.size()),
                            value.shape(),
                        ))
                    }),
                )
            }
        }

        let checkpoint = tempfile::tempdir().unwrap();
        write_fixture(checkpoint.path());
        let backend = crate::native::backend(&stream, &stream);
        let model = load_model(&backend, checkpoint.path(), MlxLoadRequest::default())
            .unwrap_or_else(|error| panic!("{family} load failed: {error}"))
            .into_inner();
        let mut session = MlxModelSession::from_model(
            model,
            eredu_core::SessionCapabilities::new(true, true, true),
        )
        .unwrap();
        let tokens = Array::from_slice(&[1u32, 2], &[1, 2]);
        let parts = [text_input_part(&tokens)];
        let mut observer = ReplacingLogits { observed: false };
        let output = session
            .submit_prefill_with_observer(
                &backend,
                crate::backend::runtime::media::input::ModelInput::new(&parts).into(),
                &mut observer,
            )
            .unwrap_or_else(|error| panic!("{family} observed forward failed: {error}"));
        let output = output.wait().unwrap();
        assert!(observer.observed, "{family} did not report final logits");
        assert!(
            output
                .evaluated()
                .unwrap()
                .as_slice::<f32>()
                .iter()
                .all(|value| *value == 41.0),
            "{family} ignored the intervention"
        );
    }
}

fn resident_reference_for_prepared(
    checkpoint: &Path,
    prepared: &PreparedModelInput,
) -> (Vec<f32>, Vec<f32>) {
    let checkpoint = checkpoint.to_path_buf();
    let prepared = prepared.clone();
    std::thread::Builder::new()
        .name("resident-reference".into())
        .spawn(move || {
            let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
            resident_reference_for_prepared_inner(&checkpoint, &prepared, &stream)
        })
        .expect("resident-reference fixture thread")
        .join()
        .expect("resident-reference fixture thread panicked")
}

fn resident_reference_for_prepared_inner(
    checkpoint: &Path,
    prepared: &PreparedModelInput,
    stream: &Stream,
) -> (Vec<f32>, Vec<f32>) {
    let backend = crate::native::backend(stream, stream);
    let model = eredu_core::load_model(&backend, checkpoint, MlxLoadRequest::default())
        .unwrap()
        .into_inner();
    let mut session = MlxModelSession::from_model(
        model,
        eredu_core::SessionCapabilities::new(true, true, true),
    )
    .unwrap();
    let parts = prepared.input_parts();
    let prefill = session
        .prefill(
            &backend,
            crate::backend::runtime::media::input::ModelInput::new(parts).into(),
        )
        .unwrap()
        .wait()
        .unwrap()
        .into_logits()
        .unwrap()
        .into_array()
        .evaluated()
        .unwrap()
        .as_slice::<f32>()
        .to_vec();
    let token = Array::from_slice(&[0u32], &[1, 1]);
    let decode = session
        .decode(&backend, token)
        .unwrap()
        .wait()
        .unwrap()
        .into_logits()
        .unwrap()
        .into_array()
        .evaluated()
        .unwrap()
        .as_slice::<f32>()
        .to_vec();
    (prefill, decode)
}

fn assert_final_logits_close(actual: &Array, expected: &[f32], tolerance: f32) {
    let actual = actual.evaluated().unwrap();
    let values = actual.as_slice::<f32>();
    assert!(values.len() >= expected.len());
    let actual = &values[values.len() - expected.len()..];
    assert_eq!(actual.len(), expected.len());
    assert!(actual
        .iter()
        .zip(expected)
        .all(|(actual, expected)| (actual - expected).abs() <= tolerance),
        "pipeline logits diverged from the resident reference: actual={actual:?}, expected={expected:?}"
    );
}

struct ChildGuard {
    children: Vec<Child>,
}

impl ChildGuard {
    fn finish(mut self) -> Vec<Output> {
        self.children
            .drain(..)
            .map(|child| child.wait_with_output().unwrap())
            .collect()
    }
}

impl Drop for ChildGuard {
    fn drop(&mut self) {
        for child in &mut self.children {
            let _ = child.kill();
        }
        for child in &mut self.children {
            let _ = child.wait();
        }
    }
}
