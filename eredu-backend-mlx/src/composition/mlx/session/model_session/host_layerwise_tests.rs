use super::*;
use crate::backend::nn::shared::MlxNeuralBackend;
use crate::backend::runtime::residency::storage::RetainedStorage;
use crate::tests::support::path_instrumentation as paths;
use eredu_core::{
    ControlledTextGeneration, InferenceGeometry, OutputDemand, TextGeneration, TextGenerationInput,
    TextPreparationInput, TokenFilterController,
};
use eredu_runtime::working_memory::{WorkingMemoryError, WorkingMemoryPool};

#[derive(Clone, Default)]
struct Controller(Rc<Cell<(usize, usize)>>);

impl TokenFilterController for Controller {
    type Error = std::convert::Infallible;

    fn inference_workspace_is_run_owned(&self) -> bool {
        true // Shared counters are instrumentation; no shared numerical payload.
    }

    fn inference_workspace(&self, _: u64) -> Option<eredu_core::TextControllerWorkspace<'_>> {
        Some(eredu_core::TextControllerWorkspace {
            filter: (&TokenFilter::All).into(),
            additional_host_bytes: 0,
        })
    }

    fn current_filter(&mut self) -> Result<TokenFilter, Self::Error> {
        let (decisions, commits) = self.0.get();
        self.0.set((decisions + 1, commits));
        Ok(TokenFilter::All)
    }

    fn commit_token(&mut self, _: u32) -> Result<(), Self::Error> {
        let (decisions, commits) = self.0.get();
        self.0.set((decisions, commits + 1));
        Ok(())
    }

    fn is_complete(&mut self) -> Result<bool, Self::Error> {
        Ok(false)
    }
}

fn config(temperature: f32, capacity: Option<u64>) -> TextGenerationConfig {
    TextGenerationConfig::new(
        eredu_core::resolve_generation_config(
            None,
            eredu_core::GenerationConfigOverrides {
                temperature: Some(temperature),
                max_new_tokens: Some(4),
                top_k: Some(0),
                top_p: Some(1.0),
                min_p: Some(0.0),
                repetition_penalty: Some(1.0),
                ..Default::default()
            },
        )
        .unwrap(),
    )
    .with_seed(19)
    .with_inference_policy(eredu_core::TextInferencePolicy {
        prefill_chunk_positions: std::num::NonZeroU64::new(1),
        managed_memory_capacity_bytes: capacity,
        submission_tracking_capacity_bytes: None,
        graph_metadata_capacity_bytes: None,
    })
}

fn tokens() -> Vec<u32> {
    vec![1, 2, 3, 4, 5]
}

fn evidence() -> TextPreparationInput<'static, MlxModelInput> {
    let ids = tokens();
    TextPreparationInput::TokenIds {
        positions: ids.len() as u64,
        capacity_bytes: (ids.capacity() * std::mem::size_of::<u32>()) as u64,
    }
}

fn geometry() -> InferenceGeometry {
    InferenceGeometry {
        batch_size: 1,
        cached_positions: 0,
        input_positions: 5,
        max_output_tokens: 4,
        prefill_chunk_positions: 1,
        output: OutputDemand::LastPosition,
    }
}

pub(super) fn artifact(model_type: &str) -> tempfile::TempDir {
    use safetensors::tensor::{serialize_to_file, TensorView};

    let root = crate::composition::mlx::replicated_text::tests::tiny_artifact(model_type, true);
    let config_path = root.path().join("config.json");
    let mut config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&config_path).unwrap()).unwrap();
    config["num_hidden_layers"] = 3.into();
    std::fs::write(config_path, serde_json::to_vec(&config).unwrap()).unwrap();

    // Preserve the existing nonzero released-schema fixture, repeating its
    // numerical layer into three independently named residency units.
    let weights_path = root.path().join("model.safetensors");
    let bytes = std::fs::read(&weights_path).unwrap();
    let tensors = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    let mut views = Vec::new();
    let mut repeated = 0;
    for (name, value) in tensors.tensors() {
        if let Some(suffix) = name.strip_prefix("model.layers.0.") {
            repeated += 1;
            for ordinal in 0..3 {
                views.push((
                    format!("model.layers.{ordinal}.{suffix}"),
                    TensorView::new(value.dtype(), value.shape().to_vec(), value.data()).unwrap(),
                ));
            }
        } else {
            views.push((name, value));
        }
    }
    assert!(repeated > 0);
    serialize_to_file(views, None, &weights_path).unwrap();
    root
}

pub(super) fn runtime(
    stream: &Stream,
    pool: &WorkingMemoryPool,
    depth: Option<usize>,
) -> (ModelRuntime<MlxBackend<'static>>, tempfile::TempDir) {
    runtime_for(stream, pool, depth, "llama")
}

pub(super) fn runtime_with_context(
    stream: &Stream,
    pool: &WorkingMemoryPool,
    depth: Option<usize>,
    maximum_positions: usize,
) -> (ModelRuntime<MlxBackend<'static>>, tempfile::TempDir) {
    let artifact = artifact("llama");
    let path = artifact.path().join("config.json");
    let mut config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    config["max_position_embeddings"] = maximum_positions.into();
    std::fs::write(path, serde_json::to_vec(&config).unwrap()).unwrap();
    runtime_from_artifact(stream, pool, depth, artifact)
}

fn runtime_for(
    stream: &Stream,
    pool: &WorkingMemoryPool,
    depth: Option<usize>,
    model_type: &str,
) -> (ModelRuntime<MlxBackend<'static>>, tempfile::TempDir) {
    let (runtime, artifact) = runtime_from_artifact(stream, pool, depth, artifact(model_type));
    assert_eq!(
        runtime
            .session()
            .payload
            .model
            .erased()
            .state_snapshot()
            .len(),
        3
    );
    (runtime, artifact)
}

pub(super) fn runtime_from_artifact(
    stream: &Stream,
    pool: &WorkingMemoryPool,
    depth: Option<usize>,
    artifact: tempfile::TempDir,
) -> (ModelRuntime<MlxBackend<'static>>, tempfile::TempDir) {
    let source = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let backend = MlxBackend::new(stream, &source).with_memory_pool(pool.clone());
    let residency = depth.map_or_else(eredu_runtime::WeightResidency::fully_resident, |depth| {
        eredu_runtime::WeightResidency::layerwise_host(eredu_runtime::LayerwiseLoadOptions::new(
            eredu_core::residency::OffloadConfig::new(Some(u64::MAX), Some(u64::MAX), depth)
                .unwrap(),
        ))
    });
    let request = crate::MlxLoadRequest::from_normalized(
        eredu_runtime::NormalizedLoadRequest::default().with_weight_residency(residency),
    );
    let model = eredu_core::load_model(&backend, artifact.path(), request).unwrap();
    let runtime = ModelRuntime::from_prepared(backend, model).unwrap();
    crate::backend::submission_recovery::wait_for_retirement(|| {
        reclaim();
        pool.unquoted_owner_count().unwrap() == 0
    });
    assert!(runtime.session().payload.model.has_published_idle_storage());
    if depth.is_some() {
        let workspace = runtime
            .session()
            .payload
            .model
            .layerwise_workspace()
            .unwrap()
            .unwrap();
        let expected_units = workspace.layout().len();
        let report = runtime.session().residency_report().unwrap().unwrap();
        let layers = execution_units(&report);
        assert_eq!(layers.len(), expected_units);
        assert_eq!(
            layers
                .iter()
                .map(|unit| unit.id())
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            expected_units,
            "each execution layer has its own residency identity"
        );
        let pinned = report
            .units()
            .iter()
            .filter(|unit| unit.policy() == eredu_core::residency::ResidencyPolicy::Pinned)
            .collect::<Vec<_>>();
        assert!(
            !pinned.is_empty(),
            "static aggregates are pinned separately"
        );
        assert!(pinned.iter().all(|unit| {
            unit.planned_tier() == eredu_core::residency::MemoryTier::Device
                && unit.device_resident()
                && unit.expected_bytes() > 0
        }));
        assert_eq!(report.units().len(), layers.len() + pinned.len());
    }
    (runtime, artifact)
}

fn execution_units(
    report: &eredu_runtime::ResidencyReport,
) -> Vec<&eredu_core::residency::UnitResidencyReport> {
    let layers = report
        .units()
        .iter()
        .filter(|unit| unit.policy() == eredu_core::residency::ResidencyPolicy::Windowed)
        .collect::<Vec<_>>();
    assert!(layers
        .iter()
        .all(|unit| unit.planned_tier() == eredu_core::residency::MemoryTier::Host));
    layers
}

fn reclaim() {
    MlxNeuralBackend::reclaim_retired_resources();
    safemlx::reclaim_allocation_owners();
}

fn settle(pool: &WorkingMemoryPool, bytes: u64) {
    crate::backend::submission_recovery::wait_for_retirement(|| {
        reclaim();
        pool.used_bytes().unwrap() == bytes && pool.unquoted_owner_count().unwrap() == 0
    });
}

fn exact_capacity(
    runtime: &ModelRuntime<MlxBackend<'_>>,
    pool: &WorkingMemoryPool,
    temperature: f32,
) -> u64 {
    let baseline = pool.used_bytes().unwrap();
    let preparation = MlxBackend::admit_text_preparation(
        runtime,
        &evidence(),
        config(temperature, Some(u64::MAX)),
        &Controller::default(),
    )
    .unwrap();
    let reservation = preparation
        .request
        .as_ref()
        .unwrap()
        .request()
        .memory_reservation()
        .unwrap();
    assert!(reservation.bytes() > 0);
    let capacity = pool.used_bytes().unwrap();
    assert_eq!(capacity, baseline + reservation.bytes());
    drop(preparation);
    settle(pool, baseline);
    capacity
}

fn outputs(
    runtime: &mut ModelRuntime<MlxBackend<'_>>,
    temperature: f32,
    capacity: u64,
    controlled: bool,
) -> Vec<MlxTextToken> {
    let controller = Controller::default();
    let outputs = if controlled {
        ControlledTextGeneration::from_input(
            runtime,
            TextGenerationInput::TokenIds(tokens()),
            config(temperature, Some(capacity)),
            controller.clone(),
        )
        .unwrap()
        .map(|token| token.unwrap().into_output())
        .collect::<Vec<_>>()
    } else {
        TextGeneration::new(runtime, tokens(), config(temperature, Some(capacity)))
            .unwrap()
            .map(Result::unwrap)
            .collect::<Vec<_>>()
    };
    assert_eq!(outputs.len(), 4);
    for (index, token) in outputs.iter().enumerate() {
        assert_eq!(token.step_receipt().unwrap().attempt(), index as u64);
    }
    if controlled {
        assert_eq!(controller.0.get(), (4, 4));
    }
    assert!(runtime
        .session()
        .payload
        .model
        .erased()
        .state_snapshot()
        .iter()
        .all(|(position, _)| *position == 8));
    outputs
}

fn live_bytes(runtime: Option<&ModelRuntime<MlxBackend<'_>>>, arrays: &[&Array]) -> u64 {
    let mut storage = match runtime {
        Some(runtime) => {
            runtime.session().ensure_no_submission_in_flight().unwrap();
            let (mut nonstate, decoder) = runtime
                .session()
                .payload
                .retained_idle_storage()
                .unwrap()
                .into_parts();
            nonstate.merge(decoder).unwrap();
            nonstate
        }
        None => RetainedStorage::default(),
    };
    for value in arrays {
        storage.include_array(value).unwrap();
    }
    storage
        .byte_bound()
        .unwrap()
        .expect("complete completed physical inventory")
}

fn cause<'a, T: std::error::Error + 'static>(
    mut error: &'a (dyn std::error::Error + 'static),
) -> Option<&'a T> {
    loop {
        if let Some(found) = error.downcast_ref::<T>() {
            return Some(found);
        }
        error = error.source()?;
    }
}

#[test]
fn host_layerwise_quote_is_cold_and_one_byte_short_rejects_before_work() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    for depth in [1, 2] {
        let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
        let (mut runtime, _artifact) = runtime(&stream, &pool, Some(depth));
        let controller = Controller::default();
        let before_paths = paths::snapshot();
        let before_inputs = paths::session_input_creation_attempts();
        let before_resets = paths::session_reset_attempts();
        let before_sources = runtime.session().residency_report().unwrap();
        let before_state = runtime.session().payload.model.erased().state_snapshot();
        let baseline = pool.used_bytes().unwrap();
        let peak = pool.peak_bytes().unwrap();
        let mut quotes = Vec::new();
        for _ in 0..3 {
            let (quote, width) = text_quote::quote(
                runtime.session(),
                geometry(),
                20,
                config(0.7, Some(u64::MAX)),
                controller.inference_workspace(4).unwrap(),
            )
            .unwrap();
            assert_eq!(width, 64);
            assert!(
                quote
                    .execution_workspace
                    .as_ref()
                    .unwrap()
                    .materialization
                    .bytes()
                    .unwrap()
                    > 0
            );
            quotes.push(quote);
        }
        assert!(quotes.windows(2).all(|pair| pair[0] == pair[1]));
        assert_eq!(pool.used_bytes().unwrap(), baseline);
        assert_eq!(pool.peak_bytes().unwrap(), peak);

        let capacity = exact_capacity(&runtime, &pool, 0.7);
        let failure = MlxBackend::fail_next_sampling_for_test(Error::ArchitectureModel(
            "insufficient host-layerwise capacity must reject before sampling".into(),
        ));
        let error = ControlledTextGeneration::from_input(
            &mut runtime,
            TextGenerationInput::TokenIds(tokens()),
            config(0.7, Some(capacity - 1)),
            controller.clone(),
        )
        .err()
        .expect("a one-position chunk cannot be reduced to fit one byte less");
        let mut source: &(dyn std::error::Error + 'static) = &error;
        let memory_error = loop {
            if let Some(error) = source.downcast_ref::<WorkingMemoryError>() {
                break error;
            }
            source = source
                .source()
                .expect("typed budget rejection remains in error chain");
        };
        assert!(matches!(
            memory_error,
            WorkingMemoryError::BudgetExceeded { .. }
        ));
        assert!(TEST_SAMPLING_FAILURE.with(|slot| slot.borrow().is_some()));
        assert_eq!(controller.0.get(), (0, 0));
        assert_eq!(paths::snapshot(), before_paths);
        assert_eq!(paths::session_input_creation_attempts(), before_inputs);
        assert_eq!(paths::session_reset_attempts(), before_resets);
        assert_eq!(
            runtime.session().residency_report().unwrap(),
            before_sources
        );
        assert_eq!(
            runtime.session().payload.model.erased().state_snapshot(),
            before_state
        );
        assert_eq!(pool.used_bytes().unwrap(), baseline);
        drop(failure);
        let output = outputs(&mut runtime, 0.7, capacity, true);
        assert!(pool.peak_bytes().unwrap() <= capacity);
        drop((output, runtime));
        settle(&pool, 0);
    }
}

#[test]
fn host_layerwise_chunked_generation_and_three_cached_decodes_match_resident() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    for temperature in [0.0, 0.7] {
        let mut reference = None;
        for depth in [None, Some(1), Some(2)] {
            for controlled in [false, true] {
                let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
                let (mut runtime, _artifact) = runtime(&stream, &pool, depth);
                let capacity = exact_capacity(&runtime, &pool, temperature);
                let before = paths::bounded_unit_acquisitions();
                let output = outputs(&mut runtime, temperature, capacity, controlled);
                let ids = output
                    .iter()
                    .map(|token| token.token_id().unwrap())
                    .collect::<Vec<_>>();
                assert!(ids.iter().all(|id| *id < 64));
                if let Some(reference) = &reference {
                    assert_eq!(&ids, reference);
                } else {
                    reference = Some(ids);
                }
                if let Some(depth) = depth {
                    assert!(paths::bounded_unit_acquisitions() - before >= 3 * (5 + 3));
                    let report = runtime.session().residency_report().unwrap().unwrap();
                    let layers = execution_units(&report);
                    assert_eq!(layers.len(), 3);
                    assert_eq!(layers.iter().filter(|unit| unit.host_resident()).count(), 3);
                    assert!(layers.iter().filter(|unit| unit.device_resident()).count() <= depth);
                }
                assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
                assert!(pool.peak_bytes().unwrap() <= capacity);
                drop((output, runtime));
                settle(&pool, 0);
            }
        }
    }
}

#[test]
fn host_layerwise_source_window_and_escaped_token_retire_independently() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    for depth in [1, 2] {
        let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
        let (mut runtime, artifact) = runtime(&stream, &pool, Some(depth));
        let initial = live_bytes(Some(&runtime), &[]);
        let capacity = exact_capacity(&runtime, &pool, 0.7);
        let output = outputs(&mut runtime, 0.7, capacity, true);
        let value = output[0].token_id().unwrap();
        let escaped = output[0].value.clone();
        let alias = escaped.as_strided(&[1][..], &[1][..], 0, &stream).unwrap();
        alias.evaluated().unwrap();
        assert_eq!(
            alias.allocation_info().unwrap(),
            escaped.allocation_info().unwrap()
        );
        let token_bytes = live_bytes(None, &[&alias]);
        assert!(token_bytes > 0);
        let all_arrays = output.iter().map(|token| &token.value).collect::<Vec<_>>();
        let all = live_bytes(Some(&runtime), &all_arrays);
        settle(&pool, all);
        drop(all_arrays);
        drop((output, escaped));
        let session_and_alias = live_bytes(Some(&runtime), &[&alias]);
        settle(&pool, session_and_alias);
        assert!(live_bytes(Some(&runtime), &[]) > initial);
        assert!(
            session_and_alias < capacity,
            "completed work releases its transient envelope"
        );
        drop((runtime, artifact));
        settle(&pool, token_bytes);
        assert_eq!(alias.evaluated().unwrap().item::<u32>(), value);
        drop(alias);
        settle(&pool, 0);
    }
}

#[test]
fn host_layerwise_abandoned_controlled_run_releases_unused_workspace_after_completion() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    for depth in [1, 2] {
        let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
        let (mut runtime, artifact) = runtime(&stream, &pool, Some(depth));
        let capacity = exact_capacity(&runtime, &pool, 0.7);
        let controller = Controller::default();
        let mut run = ControlledTextGeneration::from_input(
            &mut runtime,
            TextGenerationInput::TokenIds(tokens()),
            config(0.7, Some(capacity)),
            controller.clone(),
        )
        .unwrap();
        let output = run.next().unwrap().unwrap().into_output();
        assert_eq!(output.step_receipt().unwrap().attempt(), 0);
        assert_eq!(controller.0.get(), (1, 1));
        // Three outputs remain authorized. Native completion alone must not
        // release workspace while the unique run owner can still submit them.
        reclaim();
        assert_eq!(pool.used_bytes().unwrap(), capacity);
        drop(run);

        let arrays = [&output.value];
        let retained = live_bytes(Some(&runtime), &arrays);
        settle(&pool, retained);
        assert!(retained < capacity);
        assert!(runtime
            .session()
            .payload
            .model
            .erased()
            .state_snapshot()
            .iter()
            .all(|(position, _)| *position == 5));
        let escaped = output.value.clone();
        let id = output.token_id().unwrap();
        let token_bytes = live_bytes(None, &[&escaped]);
        drop((output, runtime, artifact));
        settle(&pool, token_bytes);
        assert_eq!(escaped.evaluated().unwrap().item::<u32>(), id);
        drop(escaped);
        settle(&pool, 0);
    }
}

#[test]
fn host_layerwise_native_provider_failure_keeps_funding_and_original_cause() {
    use crate::tests::support::provider_failure::{self, Operator};
    use eredu_runtime::working_memory::PrefillPlanningError;

    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    for depth in [1, 2] {
        let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
        let (mut runtime, artifact) = runtime_for(&stream, &pool, Some(depth), "qwen3_moe");
        let baseline = pool.used_bytes().unwrap();
        let before = paths::snapshot();
        let controller = Controller::default();
        let preparation = MlxBackend::admit_text_preparation(
            &runtime,
            &evidence(),
            config(0.7, Some(u64::MAX)),
            &controller,
        );
        let preparation = match preparation {
            Ok(preparation) => preparation,
            Err(error) => {
                // Independent-bank or missing mechanism support must remain a
                // typed cold rejection; a reservation is never a bypass.
                assert!(
                    matches!(
                        cause::<WorkingMemoryError>(&error),
                        Some(WorkingMemoryError::UnknownBound)
                    ) || matches!(
                        cause::<PrefillPlanningError>(&error),
                        Some(PrefillPlanningError::Admission(
                            eredu_core::AdmissionRejection::EstimationUnsupported { .. }
                        ))
                    ),
                    "unexpected routed admission failure: {error}"
                );
                assert_eq!(paths::snapshot(), before);
                assert_eq!(controller.0.get(), (0, 0));
                assert_eq!(pool.used_bytes().unwrap(), baseline);
                eprintln!("host-layerwise provider-failure coverage unavailable at depth {depth}: {error}");
                drop((runtime, artifact));
                settle(&pool, 0);
                continue;
            }
        };
        let charge = preparation
            .request
            .as_ref()
            .unwrap()
            .request()
            .memory_reservation()
            .unwrap()
            .bytes();
        let capacity = pool.used_bytes().unwrap();
        drop(preparation);
        settle(&pool, baseline);
        let mut run = ControlledTextGeneration::from_input(
            &mut runtime,
            TextGenerationInput::TokenIds(tokens()),
            config(0.7, Some(capacity)),
            controller.clone(),
        )
        .unwrap();
        let acquisitions = paths::bounded_unit_acquisitions();
        let fault = provider_failure::arm(Operator::Gated, 1);
        let error = run
            .next()
            .expect("failed first prediction")
            .err()
            .expect("native provider failure");
        assert_eq!(provider_failure::hits(), 1);
        assert!(
            paths::bounded_unit_acquisitions() - acquisitions >= 2,
            "the fault follows a completed earlier unit and another real acquisition"
        );
        let native =
            cause::<safemlx::error::Exception>(&error).expect("original MLX source preserved");
        assert_eq!(
            (
                native.what().to_owned(),
                native.location().file().to_owned(),
                native.location().line()
            ),
            provider_failure::cause()
        );
        assert_eq!(controller.0.get(), (1, 0));
        assert!(run.next().is_none());
        assert!(pool.used_bytes().unwrap() >= charge);
        drop((fault, error, run));
        // Observe active-owner retirement without retaining the payload.
        // Injecting an Any semantic owner would make the inventory incomplete.
        let retired = runtime.session().test_payload_retirement_probe();
        drop((runtime, artifact));
        crate::backend::submission_recovery::wait_for_retirement(|| {
            reclaim();
            retired()
        });
        reclaim();
        // Session handle retirement and an ordinary recovery pass do not
        // certify this failed operation's complete physical inventory. Its
        // isolated funding account remains quarantined; dropping request
        // metadata and the model must not refund the unproved remainder.
        assert!(pool.used_bytes().unwrap() >= charge);
    }
}

pub(super) fn family_artifact(family: &str) -> tempfile::TempDir {
    use crate::composition::mlx::replicated_text::tests::{
        qwen_hybrid_config, tiny_heterogeneous_artifact,
    };
    match family {
        "qwen3_moe" => artifact(family),
        "qwen3_5_text" => {
            let mut config = qwen_hybrid_config();
            config["num_hidden_layers"] = 3.into();
            config["layer_types"] =
                serde_json::json!(["linear_attention", "full_attention", "linear_attention"]);
            tiny_heterogeneous_artifact(config)
        }
        "gemma4" => tiny_heterogeneous_artifact(serde_json::json!({
            "model_type": "gemma4",
            "tie_word_embeddings": false,
            "text_config": {
                "model_type": "gemma4_text", "hidden_size": 16,
                "num_hidden_layers": 3, "intermediate_size": 16,
                "num_attention_heads": 4, "num_key_value_heads": 2,
                "head_dim": 4, "rms_norm_eps": 0.00001,
                "vocab_size": 32, "max_position_embeddings": 128,
                "tie_word_embeddings": false, "attention_k_eq_v": false,
                "layer_types": ["full_attention", "sliding_attention", "full_attention"],
                "sliding_window": 8
            }
        })),
        _ => unreachable!("bounded fixture family"),
    }
}

fn family_parity(family: &str) {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    let configuration = |capacity| {
        config(0.7, capacity).with_inference_policy(eredu_core::TextInferencePolicy {
            // Five positions exercise two full chunks and a shorter final chunk.
            prefill_chunk_positions: std::num::NonZeroU64::new(2),
            managed_memory_capacity_bytes: capacity,
            submission_tracking_capacity_bytes: None,
            graph_metadata_capacity_bytes: None,
        })
    };
    let mut reference = None;
    for depth in [None, Some(1), Some(2)] {
        for controlled in [false, true] {
            eprintln!("host family parity: {family}, depth={depth:?}, controlled={controlled}");
            let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
            let (mut runtime, artifact) =
                runtime_from_artifact(&stream, &pool, depth, family_artifact(family));
            let controller = Controller::default();
            let state_layout = runtime
                .session()
                .payload
                .model
                .inference_blueprint()
                .unwrap()
                .selected()
                .text_realization()
                .state()
                .layout()
                .clone();
            let unit_layout = runtime
                .session()
                .payload
                .model
                .layerwise_workspace()
                .unwrap()
                .map(|workspace| workspace.layout().clone());
            let before_paths = paths::snapshot();
            let before_sources = runtime.session().residency_report().unwrap();
            let mut geometry = geometry();
            geometry.prefill_chunk_positions = 2;
            let (full, output_width) = text_quote::quote(
                runtime.session(),
                geometry,
                20,
                configuration(Some(u64::MAX)),
                controller.inference_workspace(4).unwrap(),
            )
            .unwrap_or_else(|error| panic!("{family} cold quote depth={depth:?}: {error}"));
            assert!(
                full.execution_workspace
                    .as_ref()
                    .unwrap()
                    .peak_bytes()
                    .unwrap()
                    .is_some(),
                "{family} must provide complete native workspace facts"
            );
            assert_eq!(paths::snapshot(), before_paths);
            assert_eq!(
                runtime.session().residency_report().unwrap(),
                before_sources,
                "cold quoting cannot acquire source custody"
            );
            let baseline = pool.used_bytes().unwrap();
            let preparation = MlxBackend::admit_text_preparation(
                &runtime,
                &evidence(),
                configuration(Some(u64::MAX)),
                &controller,
            )
            .unwrap_or_else(|error| panic!("{family} admission depth={depth:?}: {error}"));
            let reservation = preparation
                .request
                .as_ref()
                .unwrap()
                .request()
                .memory_reservation()
                .unwrap();
            let capacity = pool.used_bytes().unwrap();
            assert_eq!(capacity, baseline + reservation.bytes());
            assert!(reservation.bytes() > 0);
            assert_eq!(
                reservation
                    .admission()
                    .state
                    .execution_workspace
                    .as_ref()
                    .unwrap()
                    .geometry,
                geometry
            );
            assert_eq!(paths::snapshot(), before_paths);
            let admitted_sources = runtime.session().residency_report().unwrap();
            if let (Some(before), Some(admitted)) = (&before_sources, &admitted_sources) {
                assert_eq!(admitted.weight_store(), before.weight_store());
                assert_eq!(admitted.unit_sources(), before.unit_sources());
                assert_eq!(admitted.offload(), before.offload());
                assert_eq!(admitted.units().len(), before.units().len());
                for (before, admitted) in before.units().iter().zip(admitted.units()) {
                    assert_eq!(admitted.id(), before.id());
                    let added_pin = u64::from(
                        depth.is_some()
                            && before.policy() == eredu_core::residency::ResidencyPolicy::Windowed,
                    );
                    assert_eq!(admitted.host_pins(), before.host_pins() + added_pin);
                }
            } else {
                assert_eq!(admitted_sources, before_sources);
            }
            drop(preparation);
            settle(&pool, baseline);
            assert_eq!(paths::snapshot(), before_paths);
            assert_eq!(
                runtime.session().residency_report().unwrap(),
                before_sources,
                "dropping unused preparation releases only its source pins"
            );

            let before_acquisitions = paths::bounded_unit_acquisitions();
            let outputs = if controlled {
                ControlledTextGeneration::from_input(
                    &mut runtime,
                    TextGenerationInput::TokenIds(tokens()),
                    configuration(Some(capacity)),
                    controller.clone(),
                )
                .unwrap_or_else(|error| panic!("{family} controlled preparation: {error}"))
                .map(|result| {
                    result
                        .unwrap_or_else(|error| panic!("{family} controlled execution: {error}"))
                        .into_output()
                })
                .collect::<Vec<_>>()
            } else {
                TextGeneration::new(&mut runtime, tokens(), configuration(Some(capacity)))
                    .unwrap_or_else(|error| panic!("{family} ordinary preparation: {error}"))
                    .map(|result| {
                        result
                            .unwrap_or_else(|error| panic!("{family} ordinary execution: {error}"))
                    })
                    .collect::<Vec<_>>()
            };
            assert_eq!(outputs.len(), 4);
            let ids = outputs
                .iter()
                .enumerate()
                .map(|(index, output)| {
                    assert_eq!(output.step_receipt().unwrap().attempt(), index as u64);
                    let id = output.token_id().unwrap();
                    assert!((id as usize) < output_width);
                    id
                })
                .collect::<Vec<_>>();
            if let Some(reference) = &reference {
                assert_eq!(
                    &ids, reference,
                    "{family}, depth={depth:?}, controlled={controlled}"
                );
            } else {
                reference = Some(ids);
            }
            if controlled {
                assert_eq!(controller.0.get(), (4, 4));
            }
            let frontier = runtime.session().payload.model.erased().state_snapshot();
            assert_eq!(frontier.len(), state_layout.len());
            for (index, (position, _)) in frontier.iter().enumerate() {
                let expected = if matches!(
                    state_layout.layer(index),
                    Some(eredu_core::cache::LayerCachePolicy::NoState)
                ) {
                    0
                } else {
                    8
                };
                assert_eq!(*position, expected, "{family} state layer {index}");
            }
            if let (Some(depth), Some(layout)) = (depth, unit_layout) {
                // These fixtures have only text execution units. Derive their
                // count and per-group window bound from the actual selected graph.
                assert!(
                    paths::bounded_unit_acquisitions() - before_acquisitions
                        >= layout.len() * (3 + 3)
                );
                let group_window_bound = (0..layout.group_count())
                    .map(|group| layout.group_range(group).unwrap().len().min(depth))
                    .sum::<usize>();
                let report = runtime.session().residency_report().unwrap().unwrap();
                let layers = execution_units(&report);
                assert_eq!(layers.len(), layout.len());
                assert!(layers.iter().all(|unit| unit.host_resident()));
                assert!(
                    layers.iter().filter(|unit| unit.device_resident()).count()
                        <= group_window_bound
                );
                assert!(report
                    .active_window()
                    .iter()
                    .all(|id| layers.iter().any(|unit| unit.id() == id)));
            }
            assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
            assert!(pool.peak_bytes().unwrap() <= capacity);
            let arrays = outputs
                .iter()
                .map(|output| &output.value)
                .collect::<Vec<_>>();
            settle(&pool, live_bytes(Some(&runtime), &arrays));
            drop(arrays);
            drop((outputs, runtime, artifact));
            settle(&pool, 0);
        }
    }
}

#[test]
fn host_layerwise_routed_qwen3_moe_success_matches_resident() {
    family_parity("qwen3_moe");
}

#[test]
fn host_layerwise_hybrid_qwen35_success_matches_resident() {
    family_parity("qwen3_5_text");
}

#[test]
fn host_layerwise_text_only_gemma4_composite_success_matches_resident() {
    family_parity("gemma4");
}

mod rotary;
