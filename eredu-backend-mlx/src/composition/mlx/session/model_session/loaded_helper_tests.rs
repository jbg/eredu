use super::*;
use crate::backend::runtime::residency::storage::RetainedStorage;
use crate::composition::mlx::replicated_text::tests::{
    routed_deepseek_v4_config, tiny_artifact, tiny_heterogeneous_artifact,
};
use crate::tests::support::path_instrumentation as paths;
use eredu_runtime::working_memory::{WorkingMemoryError, WorkingMemoryPool};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

fn artifact(v4: bool) -> tempfile::TempDir {
    if v4 {
        return tiny_heterogeneous_artifact(routed_deepseek_v4_config());
    }
    let root = tiny_artifact("llama", false);
    let path = root.path().join("config.json");
    let mut config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    config["rope_scaling"] = serde_json::json!({
        "rope_type": "llama3", "factor": 8.0,
        "low_freq_factor": 1.0, "high_freq_factor": 4.0,
        "original_max_position_embeddings": 32
    });
    std::fs::write(path, serde_json::to_vec(&config).unwrap()).unwrap();
    root
}

fn reclaim() {
    crate::backend::nn::shared::MlxNeuralBackend::reclaim_retired_resources();
    safemlx::reclaim_allocation_owners();
}
fn settle(pool: &WorkingMemoryPool, bytes: u64) {
    crate::backend::submission_recovery::wait_for_retirement(|| {
        reclaim();
        pool.used_bytes().unwrap() == bytes && pool.unquoted_owner_count().unwrap() == 0
    });
}
fn usage(pool: &WorkingMemoryPool) -> (u64, u64) {
    (pool.used_bytes().unwrap(), pool.peak_bytes().unwrap())
}
fn cause<'a, T: std::error::Error + 'static>(
    mut error: &'a (dyn std::error::Error + 'static),
) -> Option<&'a T> {
    loop {
        if let Some(typed) = error.downcast_ref::<T>() {
            return Some(typed);
        }
        error = error.source()?;
    }
}

#[derive(Default)]
struct Parameters(BTreeMap<String, safemlx::AllocationIdentity>);
impl eredu_nn::ParameterSlotVisitor<crate::MlxTensor> for Parameters {
    fn visit_slot(&mut self, metadata: eredu_nn::ParameterMetadataView<'_>, value: &mut crate::MlxTensor) {
        let identity = value
            .as_array()
            .try_metadata_snapshot()
            .unwrap()
            .allocation()
            .unwrap()
            .identity();
        assert!(self
            .0
            .insert(metadata.id().as_str().to_owned(), identity)
            .is_none());
    }
}
fn parameters(model: &mut MlxModel) -> Parameters {
    let mut result = Parameters::default();
    assert!(model
        .executable_mut()
        .erased_mut()
        .visit_loaded_parameters(&mut result));
    result
}
fn module_storage(model: &mut MlxModel) -> RetainedStorage {
    let target = model.executable_mut().erased();
    let mut storage = target.retained_target_module_storage().unwrap();
    storage
        .merge(target.retained_prediction_storage().unwrap())
        .unwrap();
    storage
}

fn loaded_aliases(v4: bool) {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let backend = MlxBackend::new(&stream, &stream).with_memory_pool(pool.clone());
    let root = artifact(v4);
    let mut model =
        eredu_core::load_model(&backend, root.path(), crate::MlxLoadRequest::default()).unwrap();
    // No forward pass or test-side evaluation has been performed. A mutable
    // parameter visitor merely observes slots while original loading custody
    // is still retained by this prepared model; it never replaces a value.
    assert!(model.memory_owner().is_some());
    let before_parameters = parameters(&mut model);
    let parameter_ids: BTreeSet<_> = before_parameters.0.values().copied().collect();
    assert!(!parameter_ids.is_empty());
    let storage = module_storage(&mut model);
    assert!(storage.unknown_arrays().is_empty());
    let native_facts = storage.array_allocation_facts();
    let native_bytes = storage
        .byte_bound()
        .unwrap()
        .expect("loaded module helpers must be settled");
    let helper_facts: BTreeMap<_, _> = native_facts
        .iter()
        .filter(|(id, _)| !parameter_ids.contains(id))
        .map(|(id, bytes)| (*id, *bytes))
        .collect();
    assert_eq!(helper_facts.len(), if v4 { 5 } else { 1 });
    assert!(helper_facts.values().all(|bytes| *bytes > 0));
    let helpers = storage
        .into_retained_arrays()
        .unwrap()
        .filter(|array| {
            helper_facts.contains_key(
                &array
                    .try_metadata_snapshot()
                    .unwrap()
                    .allocation()
                    .unwrap()
                    .identity(),
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(helpers.len(), helper_facts.len());
    let paths_before = paths::snapshot();
    let reads_before = model.residency_report().unwrap();
    let frontier = model.executable_mut().erased().state_snapshot();
    let before_usage = usage(&pool);
    for _ in 0..3 {
        let cold = module_storage(&mut model);
        assert_eq!(cold.byte_bound().unwrap(), Some(native_bytes));
        assert_eq!(cold.array_allocation_facts(), native_facts);
        assert!(cold.unknown_arrays().is_empty());
        for helper in &helpers {
            let allocation = helper
                .try_metadata_snapshot()
                .unwrap()
                .allocation()
                .unwrap();
            assert_eq!(
                helper_facts.get(&allocation.identity()),
                Some(&(allocation.bytes() as u64))
            );
        }
    }
    assert_eq!(
        parameters(&mut model).0,
        before_parameters.0,
        "helpers never become editable checkpoint parameters"
    );
    assert_eq!(model.executable_mut().erased().state_snapshot(), frontier);
    assert_eq!(model.residency_report().unwrap(), reads_before);
    assert_eq!(paths::snapshot(), paths_before);
    assert_eq!(usage(&pool), before_usage);

    // Numerical observations occur only after the cold completion assertions.
    if v4 {
        let mut values = helpers
            .iter()
            .map(|helper| {
                assert_eq!(helper.shape(), &[1]);
                helper.evaluated().unwrap().try_to_vec::<f32>().unwrap()[0]
            })
            .collect::<Vec<_>>();
        values.sort_by(f32::total_cmp);
        // Main attention uses frequency_scale=1. The pooling compressor and
        // its indexer compressor each use the actual ratio=4.
        for (actual, expected) in values.into_iter().zip([0.25_f32, 0.25, 1.0, 1.0, 1.0]) {
            assert!((actual - expected).abs() <= 1e-6);
        }
    } else {
        assert_eq!(helpers[0].shape(), &[4]);
        let values = helpers[0].evaluated().unwrap().try_to_vec::<f32>().unwrap();
        for (actual, expected) in values.into_iter().zip([1.0_f32, 80.0, 800.0, 8000.0]) {
            assert!((actual - expected).abs() <= expected * 1e-5);
        }
    }
    let runtime = ModelRuntime::from_prepared(backend, model).unwrap();
    crate::backend::submission_recovery::wait_for_retirement(|| {
        reclaim();
        pool.unquoted_owner_count().unwrap() == 0
    });
    assert!(runtime.session().payload._memory_owner.is_none());
    assert!(runtime.session().payload.model.has_published_idle_storage());
    let inventory = runtime.session().payload.retained_idle_storage().unwrap();
    assert_eq!(inventory.decoder_state_bytes().unwrap(), Some(0));
    // Loading also retains input-derived source-metadata reservations. The
    // physical inventory is a lower bound on that combined charge; the exact
    // helper-only charge and complete source retirement are checked below.
    assert!(pool.used_bytes().unwrap() >= inventory.nonstate_bytes().unwrap().unwrap());
    drop(inventory);
    let aliases = helpers.clone();
    let helper_bytes: u64 = helper_facts.values().sum();
    drop(runtime);
    stream.synchronize().unwrap();
    settle(&pool, helper_bytes);
    drop(helpers);
    settle(&pool, helper_bytes);
    for alias in &aliases {
        assert!(alias
            .evaluated()
            .unwrap()
            .try_to_vec::<f32>()
            .unwrap()
            .iter()
            .all(|x| x.is_finite() && *x > 0.0));
    }
    drop(aliases);
    settle(&pool, 0);
}

#[test]
fn public_scaled_llama_load_finalizes_helper_without_parameter_promotion() {
    loaded_aliases(false);
}
#[test]
fn public_v4_load_finalizes_all_five_helpers_and_aliases_retain_exact_charge() {
    loaded_aliases(true);
}

#[derive(Debug)]
struct FinalizationFailure(Arc<AtomicBool>);
impl std::fmt::Display for FinalizationFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("injected after loaded helper submission")
    }
}
impl std::error::Error for FinalizationFailure {}
impl Drop for FinalizationFailure {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

#[test]
fn public_loading_failure_after_helper_submission_preserves_typed_cause_and_recovers() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let backend = MlxBackend::new(&stream, &stream).with_memory_pool(pool.clone());
    let root = artifact(true);
    let dropped = Arc::new(AtomicBool::new(false));
    Executable::reject_next_loaded_helper_finalization_for_test(Error::Other(Box::new(
        FinalizationFailure(Arc::clone(&dropped)),
    )));
    let before = paths::snapshot();
    let error = eredu_core::load_model(&backend, root.path(), crate::MlxLoadRequest::default())
        .err()
        .expect("injected post-submission rejection");
    let original =
        cause::<FinalizationFailure>(&error).expect("original typed native provider cause");
    assert!(Arc::ptr_eq(&original.0, &dropped));
    assert!(!dropped.load(Ordering::SeqCst));
    assert_ne!(
        paths::snapshot(),
        before,
        "loading reached actual parameter materialization"
    );
    assert_eq!(
        pool.used_bytes().unwrap(),
        0,
        "failure precedes physical publication"
    );
    drop(error);
    assert!(dropped.load(Ordering::SeqCst));
    stream.synchronize().unwrap();
    settle(&pool, 0);
}

#[cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
mod generation {
    use super::*;
    use eredu_core::{
        ControlledTextGeneration, InferenceGeometry, OutputDemand, TextGeneration,
        TextGenerationInput, TextPreparationInput, TokenFilterController,
    };
    use eredu_runtime::working_memory::PrefillPlanningError;

    #[derive(Clone, Default)]
    struct Controller(Rc<Cell<(usize, usize)>>);
    impl TokenFilterController for Controller {
        type Error = std::convert::Infallible;
        fn inference_workspace_is_run_owned(&self) -> bool {
            true
        }
        fn inference_workspace(&self, _: u64) -> Option<eredu_core::TextControllerWorkspace<'_>> {
            Some(eredu_core::TextControllerWorkspace {
                filter: (&TokenFilter::All).into(),
                // Shared counters are test instrumentation with no model payload.
                additional_host_bytes: std::mem::size_of::<Self>() as u64,
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
    fn tokens() -> Vec<u32> {
        vec![1, 2, 3, 4, 5]
    }
    fn config(capacity: Option<u64>) -> TextGenerationConfig {
        TextGenerationConfig::new(
            eredu_core::resolve_generation_config(
                None,
                eredu_core::GenerationConfigOverrides {
                    temperature: Some(0.0),
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
            prefill_chunk_positions: std::num::NonZeroU64::new(2),
            managed_memory_capacity_bytes: capacity,
            submission_tracking_capacity_bytes: None,
            graph_metadata_capacity_bytes: None,
        })
    }
    fn evidence() -> TextPreparationInput<'static, MlxModelInput> {
        let ids = tokens();
        TextPreparationInput::TokenIds {
            positions: ids.len() as u64,
            capacity_bytes: (ids.capacity() * std::mem::size_of::<u32>()) as u64,
        }
    }
    fn assert_incomplete(error: &(dyn std::error::Error + 'static)) {
        assert!(
            matches!(
                cause::<WorkingMemoryError>(error),
                Some(WorkingMemoryError::UnknownBound)
            ) || matches!(
                cause::<PrefillPlanningError>(error),
                Some(
                    PrefillPlanningError::IncompleteWorkspace(_)
                        | PrefillPlanningError::Admission(
                            eredu_core::AdmissionRejection::EstimationUnsupported { .. }
                        )
                )
            ),
            "unexpected full V4 quote rejection: {error}"
        );
    }

    #[test]
    fn v4_full_quote_reports_actual_completeness_and_three_decodes_match_controlled() {
        let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
        let source_stream =
            Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        let mut reference = None;
        for bounded in [false, true] {
            for controlled in [false, true] {
                let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
                let backend =
                    MlxBackend::new(&stream, &source_stream).with_memory_pool(pool.clone());
                let root = artifact(true);
                let model =
                    eredu_core::load_model(&backend, root.path(), crate::MlxLoadRequest::default())
                        .unwrap();
                let mut runtime = ModelRuntime::from_prepared(backend, model).unwrap();
                crate::backend::submission_recovery::wait_for_retirement(|| {
                    reclaim();
                    pool.unquoted_owner_count().unwrap() == 0
                });
                assert!(runtime.session().payload.model.has_published_idle_storage());
                let controller = Controller::default();
                let mut capacity = None;
                if bounded {
                    let geometry = InferenceGeometry {
                        batch_size: 1,
                        cached_positions: 0,
                        input_positions: 5,
                        max_output_tokens: 4,
                        prefill_chunk_positions: 2,
                        output: OutputDemand::LastPosition,
                    };
                    let before = (
                        paths::snapshot(),
                        paths::session_input_creation_attempts(),
                        paths::session_reset_attempts(),
                        usage(&pool),
                    );
                    let frontier = runtime.session().payload.model.erased().state_snapshot();
                    let residency = runtime.session().residency_report().unwrap();
                    let full_generation = runtime
                        .session()
                        .payload
                        .model
                        .quote_replicated_resident_text_with_sampling(
                            geometry,
                            config(Some(u64::MAX)),
                            &TokenFilter::All,
                        )
                        .unwrap();
                    let prompt = runtime
                        .session()
                        .payload
                        .model
                        .quote_text_prompt_workspace(geometry, Some(20))
                        .unwrap();
                    let (full, width) = text_quote::quote(
                        runtime.session(),
                        geometry,
                        20,
                        config(Some(u64::MAX)),
                        controller.inference_workspace(4).unwrap(),
                    )
                    .unwrap();
                    assert_eq!(width, 64);
                    let complete = full.completeness
                        != eredu_core::EstimationCompleteness::PersistentStateOnly
                        && full
                            .execution_workspace
                            .as_ref()
                            .unwrap()
                            .peak_bytes()
                            .unwrap()
                            .is_some();
                    eprintln!("V4 actual full quote complete={complete}: equations={:?}, retained={:?}, equation_gap={:?}, sampling={:?}, sampling_gap={:?}, prompt={:?}, state_coverage={:?}, selected_state={:?}, workspace={:?}",
                        full_generation.equations.transient(), full_generation.equations.retained_peak_bytes(), full_generation.equations.first_gap(),
                        full_generation.sampling.peak, full_generation.sampling.first_gap, prompt.peak(), full.persistent_state_completeness, full.selected_state_backing, full.execution_workspace);
                    assert_eq!(
                        (
                            paths::snapshot(),
                            paths::session_input_creation_attempts(),
                            paths::session_reset_attempts(),
                            usage(&pool)
                        ),
                        before
                    );
                    assert_eq!(
                        runtime.session().payload.model.erased().state_snapshot(),
                        frontier
                    );
                    assert_eq!(runtime.session().residency_report().unwrap(), residency);
                    let admitted = MlxBackend::admit_text_preparation(
                        &runtime,
                        &evidence(),
                        config(Some(u64::MAX)),
                        &controller,
                    );
                    if complete {
                        let preparation = admitted.unwrap();
                        let request = preparation.request.as_ref().unwrap().request();
                        let reservation = request.memory_reservation().unwrap();
                        assert!(reservation.bytes() > 0);
                        let ceiling = before.3 .0.checked_add(reservation.bytes()).unwrap();
                        assert_eq!(pool.used_bytes().unwrap(), ceiling);
                        // This probe includes our instrumented controller; its
                        // bound also covers the ordinary driver's default one.
                        capacity = Some(ceiling);
                        drop(preparation);
                        settle(&pool, before.3 .0);
                    } else {
                        let error = admitted
                            .err()
                            .expect("incomplete full workspace cannot admit");
                        assert_incomplete(&error);
                        eprintln!("V4 finite request rejected before work: {error}; remaining quote gap is not finite support");
                        if controlled {
                            let error = ControlledTextGeneration::from_input(
                                &mut runtime,
                                TextGenerationInput::TokenIds(tokens()),
                                config(Some(u64::MAX)),
                                controller.clone(),
                            )
                            .err()
                            .expect("controlled must reject incomplete quote");
                            assert_incomplete(&error);
                        } else {
                            let error =
                                TextGeneration::new(&mut runtime, tokens(), config(Some(u64::MAX)))
                                    .err()
                                    .expect("ordinary must reject incomplete quote");
                            assert_incomplete(&error);
                        }
                        assert_eq!(controller.0.get(), (0, 0));
                        assert_eq!(
                            (
                                paths::snapshot(),
                                paths::session_input_creation_attempts(),
                                paths::session_reset_attempts(),
                                usage(&pool)
                            ),
                            before
                        );
                        assert_eq!(
                            runtime.session().payload.model.erased().state_snapshot(),
                            frontier
                        );
                    }
                }
                let outputs = if controlled {
                    ControlledTextGeneration::from_input(
                        &mut runtime,
                        TextGenerationInput::TokenIds(tokens()),
                        config(capacity),
                        controller.clone(),
                    )
                    .unwrap()
                    .map(|token| token.unwrap().into_output())
                    .collect::<Vec<_>>()
                } else {
                    TextGeneration::new(&mut runtime, tokens(), config(capacity))
                        .unwrap()
                        .map(Result::unwrap)
                        .collect::<Vec<_>>()
                };
                assert_eq!(outputs.len(), 4);
                let ids = outputs
                    .iter()
                    .enumerate()
                    .map(|(index, output)| {
                        if capacity.is_some() {
                            assert_eq!(output.step_receipt().unwrap().attempt(), index as u64);
                        }
                        let id = output.token_id().unwrap();
                        assert!(id < 64);
                        id
                    })
                    .collect::<Vec<_>>();
                if let Some(expected) = &reference {
                    assert_eq!(
                        &ids, expected,
                        "bounded={bounded}, controlled={controlled}, admitted={capacity:?}"
                    );
                } else {
                    reference = Some(ids);
                }
                if controlled {
                    assert_eq!(controller.0.get(), (4, 4));
                }
                let frontier = runtime.session().payload.model.erased().state_snapshot();
                assert_eq!(frontier.len(), 3);
                assert!(
                    frontier.iter().all(|(position, _)| *position == 8),
                    "five prompt positions plus three cached decodes: {frontier:?}"
                );
                drop((outputs, runtime));
                stream.synchronize().unwrap();
                settle(&pool, 0);
            }
        }
    }
}
