use super::*;
use crate::tests::support::path_instrumentation as paths;
use eredu_core::{
    ControlledTextGeneration, TextGenerationInput, TextPreparationInput, TokenFilterController,
};
#[cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
use eredu_core::{InferenceGeometry, OutputDemand, TextGeneration};
use eredu_runtime::working_memory::{MemoryLedger, PrefillPlanningError, WorkingMemoryError};

#[derive(Clone, Default)]
struct Controller(Rc<Cell<(usize, usize, usize)>>);

impl TokenFilterController for Controller {
    type Error = std::convert::Infallible;

    fn inference_workspace_is_run_owned(&self) -> bool {
        // Shared counters are instrumentation, not numerical masks or history.
        true
    }

    fn inference_workspace(&self, _: u64) -> Option<eredu_core::TextControllerWorkspace<'_>> {
        Some(eredu_core::TextControllerWorkspace {
            filter: (&TokenFilter::All).into(),
            additional_host_bytes: (std::mem::size_of::<Self>()
                + std::mem::size_of::<Cell<(usize, usize, usize)>>()
                + 2 * std::mem::size_of::<usize>()) as u64,
        })
    }

    fn current_filter(&mut self) -> Result<TokenFilter, Self::Error> {
        let (decisions, commits, complete) = self.0.get();
        self.0.set((decisions + 1, commits, complete));
        Ok(TokenFilter::All)
    }

    fn commit_token(&mut self, _: u32) -> Result<(), Self::Error> {
        let (decisions, commits, complete) = self.0.get();
        self.0.set((decisions, commits + 1, complete));
        Ok(())
    }

    fn is_complete(&mut self) -> Result<bool, Self::Error> {
        let (decisions, commits, complete) = self.0.get();
        self.0.set((decisions, commits, complete + 1));
        Ok(false)
    }
}

fn config(temperature: f32, capacity: Option<u64>) -> TextGenerationConfig {
    TextGenerationConfig::new(
        eredu_core::resolve_generation_config(
            None,
            eredu_core::GenerationConfigOverrides {
                temperature: Some(temperature),
                max_new_tokens: Some(3),
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
        memory_limits: (capacity).map_or_else(
            eredu_core::MemoryLimitDeclarations::unlimited,
            |bytes| {
                eredu_core::MemoryLimitDeclarations::new([(
                    "host".into(),
                    eredu_core::MemoryLimit::Finite(bytes),
                )])
            },
        ),
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

fn runtime(
    stream: &Stream,
    pool: &MemoryLedger,
) -> (ModelRuntime<MlxBackend<'static>>, tempfile::TempDir) {
    let source = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let backend = MlxBackend::new(stream, &source).with_memory_ledger(pool.clone());
    let artifact = crate::composition::mlx::replicated_text::tests::tiny_artifact("llama", true);
    let model = eredu_core::load_model(&backend, artifact.path(), crate::MlxLoadRequest::default())
        .unwrap();
    let runtime = ModelRuntime::from_prepared(backend, model).unwrap();
    crate::backend::submission_recovery::wait_for_retirement(|| {
        reclaim();
        pool.unquoted_owner_count().unwrap() == 0
    });
    assert!(runtime.session().payload.model.has_published_idle_storage());
    assert!(pool.fixture_host_charge().unwrap() > 0);
    (runtime, artifact)
}

fn reclaim() {
    crate::backend::nn::shared::MlxNeuralBackend::reclaim_retired_resources();
    safemlx::reclaim_allocation_owners();
}

fn settle(pool: &MemoryLedger, bytes: u64) {
    crate::backend::submission_recovery::wait_for_retirement(|| {
        reclaim();
        pool.fixture_host_charge().unwrap() == bytes && pool.unquoted_owner_count().unwrap() == 0
    });
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

fn assert_missing_bound(error: &(dyn std::error::Error + 'static)) {
    assert!(
        matches!(
            cause::<WorkingMemoryError>(error),
            Some(WorkingMemoryError::UnknownBound)
        ) || matches!(
            cause::<PrefillPlanningError>(error),
            Some(PrefillPlanningError::Admission(
                eredu_core::AdmissionRejection::EstimationUnsupported { .. }
            ))
        ),
        "unexpected admission failure: {error}"
    );
}

struct ColdState {
    paths: paths::Counts,
    inputs: usize,
    resets: usize,
    frontier: Vec<(i32, Vec<(eredu_core::cache::StateTensorRole, bool)>)>,
    bytes: u64,
    peak: u64,
    nonstate: u64,
}

impl ColdState {
    fn capture(runtime: &ModelRuntime<MlxBackend<'_>>, pool: &MemoryLedger) -> Self {
        let inventory = runtime.session().payload.retained_idle_storage().unwrap();
        assert!(inventory.has_empty_decoder_storage().unwrap());
        Self {
            paths: paths::snapshot(),
            inputs: paths::session_input_creation_attempts(),
            resets: paths::session_reset_attempts(),
            frontier: runtime.session().payload.model.erased().state_snapshot(),
            bytes: pool.fixture_host_charge().unwrap(),
            peak: pool.fixture_host_peak().unwrap(),
            nonstate: inventory.nonstate_bytes().unwrap().unwrap(),
        }
    }

    fn assert_no_work(&self, runtime: &ModelRuntime<MlxBackend<'_>>, controller: &Controller) {
        assert_eq!(controller.0.get(), (0, 0, 0));
        assert_eq!(paths::snapshot(), self.paths);
        assert_eq!(paths::session_input_creation_attempts(), self.inputs);
        assert_eq!(paths::session_reset_attempts(), self.resets);
        assert_eq!(
            runtime.session().payload.model.erased().state_snapshot(),
            self.frontier
        );
        let inventory = runtime.session().payload.retained_idle_storage().unwrap();
        assert_eq!(inventory.nonstate_bytes().unwrap(), Some(self.nonstate));
        assert!(inventory.has_empty_decoder_storage().unwrap());
    }

    fn assert_uncharged(&self, pool: &MemoryLedger) {
        assert_eq!(pool.fixture_host_charge().unwrap(), self.bytes);
        assert_eq!(pool.fixture_host_peak().unwrap(), self.peak);
        assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
    }
}

#[test]
fn cpu_public_quote_retains_cpu_selection_and_rejects_missing_native_facts_before_work() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
    let (mut runtime, _artifact) = runtime(&stream, &pool);
    assert!(matches!(
        runtime
            .session()
            .payload
            .model
            .resident_workspace_mechanisms(),
        Some(crate::backend::nn::workspace::ResidentExecutionMechanisms::Cpu { .. })
    ));
    let controller = Controller::default();
    let before = ColdState::capture(&runtime, &pool);
    let _failure = MlxBackend::fail_next_sampling_for_test(Error::ArchitectureModel(
        "missing native facts must reject before sampler construction".into(),
    ));
    let error = text_quote::admit(
        &runtime,
        &evidence(),
        config(0.7, Some(u64::MAX)),
        &controller,
    )
    .err()
    .unwrap();
    assert_missing_bound(&error);
    let error = ControlledTextGeneration::from_input(
        &mut runtime,
        TextGenerationInput::TokenIds(tokens()),
        config(0.7, Some(u64::MAX)),
        controller.clone(),
    )
    .err()
    .unwrap();
    assert_missing_bound(&error);
    assert!(TEST_SAMPLING_FAILURE.with(|slot| slot.borrow().is_some()));
    before.assert_no_work(&runtime, &controller);
    before.assert_uncharged(&pool);
    drop(runtime);
    settle(&pool, 0);
}

#[cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
#[test]
fn sized_external_controller_rejects_before_sampling_or_callbacks() {
    #[derive(Clone)]
    struct SharedController {
        mask: Rc<TokenFilter>,
        descriptions: Rc<Cell<usize>>,
        probe: Controller,
    }

    impl TokenFilterController for SharedController {
        type Error = std::convert::Infallible;

        fn inference_workspace(&self, _: u64) -> Option<eredu_core::TextControllerWorkspace<'_>> {
            self.descriptions.set(self.descriptions.get() + 1);
            let TokenFilter::Allowed(mask) = self.mask.as_ref() else {
                unreachable!("the fixture owns a nonempty numerical mask")
            };
            Some(eredu_core::TextControllerWorkspace {
                filter: self.mask.as_ref().into(),
                additional_host_bytes: mask.capacity() as u64,
            })
        }

        // The default lifetime declaration must remain false: an external
        // clone below retains the same numerical backing beyond this run.
        fn current_filter(&mut self) -> Result<TokenFilter, Self::Error> {
            self.probe.current_filter()?;
            Ok(self.mask.as_ref().clone())
        }

        fn commit_token(&mut self, token: u32) -> Result<(), Self::Error> {
            self.probe.commit_token(token)
        }

        fn is_complete(&mut self) -> Result<bool, Self::Error> {
            self.probe.is_complete()
        }
    }

    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    let pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
    let (mut runtime, _artifact) = runtime(&stream, &pool);
    assert!(runtime.session().payload.model.has_workspace_mechanisms());
    let mask = Rc::new(TokenFilter::allowed(vec![true; 64]).unwrap());
    let descriptions = Rc::new(Cell::new(0));
    let probe = Controller::default();
    let controller = SharedController {
        mask: Rc::clone(&mask),
        descriptions: Rc::clone(&descriptions),
        probe: probe.clone(),
    };
    assert!(!controller.inference_workspace_is_run_owned());
    let before = ColdState::capture(&runtime, &pool);
    let _failure = MlxBackend::fail_next_sampling_for_test(Error::ArchitectureModel(
        "shared controller payload must reject before sampler construction".into(),
    ));
    let error = ControlledTextGeneration::from_input(
        &mut runtime,
        TextGenerationInput::TokenIds(tokens()),
        config(0.7, Some(u64::MAX)),
        controller,
    )
    .err()
    .expect("a complete size witness cannot certify external payload retirement");
    assert_eq!(
        cause::<WorkingMemoryError>(&error),
        Some(&WorkingMemoryError::UnknownBound)
    );
    assert_eq!(
        descriptions.get(),
        1,
        "the actual size witness was inspected"
    );
    assert!(TEST_SAMPLING_FAILURE.with(|slot| slot.borrow().is_some()));
    before.assert_no_work(&runtime, &probe);
    before.assert_uncharged(&pool);
    assert_eq!(Rc::strong_count(&mask), 1);
    assert!(
        matches!(mask.as_ref(), TokenFilter::Allowed(values) if values.iter().all(|valid| *valid))
    );
    drop(runtime);
    settle(&pool, 0);
}

#[cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
fn checked_prompt_provenance(
    runtime: &mut ModelRuntime<MlxBackend<'_>>,
    pool: &MemoryLedger,
    controller: &Controller,
    preparation: MlxTextPreparation,
) {
    let before = ColdState::capture(runtime, pool);
    let mut oversized = Vec::with_capacity(tokens().capacity() + 16);
    oversized.extend(tokens());
    let error =
        MlxBackend::prepare_text_prompt_admitted(runtime.backend(), oversized, &preparation)
            .err()
            .expect("spare source capacity beyond the real quote must reject");
    assert_eq!(
        cause::<WorkingMemoryError>(&error),
        Some(&WorkingMemoryError::IdentityMismatch)
    );
    before.assert_no_work(runtime, controller);
    before.assert_uncharged(pool);

    // The failed size check must not consume the single prompt-construction
    // stage. This succeeds under that same real, complete native preparation.
    let prompt =
        MlxBackend::prepare_text_prompt_admitted(runtime.backend(), tokens(), &preparation)
            .expect("rejected source capacity must leave prompt construction available");
    let before_binding = ColdState::capture(runtime, pool);
    let rebuilt = prompt.with_borrowed(|input| MlxModelInput::from(input));
    assert!(prompt
        .shared_cache_identity()
        .unwrap()
        .same_storage(rebuilt.shared_cache_identity().unwrap()));
    assert!(rebuilt.quote.is_none());
    let error =
        MlxBackend::bind_text_prompt_preparation(runtime.backend(), rebuilt.clone(), &preparation)
            .err()
            .expect("rebuilding a borrowed input cannot manufacture quote provenance");
    assert_eq!(
        cause::<WorkingMemoryError>(&error),
        Some(&WorkingMemoryError::IdentityMismatch)
    );
    assert!(
        rebuilt.quote.is_none(),
        "rejection must not relabel another alias"
    );

    // Even returning a changed chunk selection to its original value does not
    // restore provenance. The numerical arrays and final request geometry match.
    let original_chunk = prompt.prefill_chunk_positions.unwrap();
    let changed = prompt
        .clone()
        .with_prefill_chunk_positions(std::num::NonZeroU64::new(original_chunk.get() + 1).unwrap())
        .with_prefill_chunk_positions(original_chunk);
    let error = MlxBackend::bind_text_prompt_preparation(runtime.backend(), changed, &preparation)
        .err()
        .expect("mutable chunk policy exposure must invalidate quote provenance");
    assert_eq!(
        cause::<WorkingMemoryError>(&error),
        Some(&WorkingMemoryError::IdentityMismatch)
    );
    let relabelled = prompt
        .clone()
        .with_inference_request(preparation.request.as_ref().unwrap().request().clone());
    let error =
        MlxBackend::bind_text_prompt_preparation(runtime.backend(), relabelled, &preparation)
            .err()
            .expect("retaining the same request cannot restore quote provenance");
    assert_eq!(
        cause::<WorkingMemoryError>(&error),
        Some(&WorkingMemoryError::IdentityMismatch)
    );
    let error = prompt
        .clone()
        .with_semantic_content_fingerprint("changed-content")
        .unwrap_err();
    assert_eq!(
        cause::<WorkingMemoryError>(&error),
        Some(&WorkingMemoryError::UnknownBound)
    );
    before_binding.assert_no_work(runtime, controller);
    before_binding.assert_uncharged(pool);

    let alias = prompt.clone();
    drop(prompt);
    let bound = MlxBackend::bind_text_prompt_preparation(runtime.backend(), alias, &preparation)
        .expect("genuine alias must bind after all rejected substitutions");
    assert!(bound
        .quote
        .as_ref()
        .is_some_and(|quote| quote.same_owner(preparation.quote.as_ref().unwrap())));
    before_binding.assert_no_work(runtime, controller);
    before_binding.assert_uncharged(pool);
    let mut sampling = MlxBackend::start_text_generation_admitted(
        runtime.backend(),
        preparation.quote.as_ref().unwrap().config(),
        &preparation,
    )
    .unwrap();
    let mut installed_epoch = None;
    runtime
        .session()
        .validate_parameter_epoch(&mut installed_epoch)
        .unwrap();
    assert!(installed_epoch.is_some());
    assert_eq!(
        sampling.sampling.parameter_epoch, installed_epoch,
        "pending admitted sampling must retain its quoted parameter version before prefill"
    );
    let before_override = ColdState::capture(runtime, pool);
    let original_temperature = sampling.sampling.temperature;
    let had_rng = sampling.sampling.prng.is_some();
    use eredu_runtime::execution_control::{SamplingOverride, TextSamplingControlBackend};
    for change in [
        SamplingOverride {
            temperature: Some(0.0),
            reseed: None,
        },
        SamplingOverride {
            temperature: Some(0.4),
            reseed: Some(99),
        },
    ] {
        let error =
            MlxBackend::apply_sampling_override(runtime, &mut sampling, None, change).unwrap_err();
        assert_eq!(
            cause::<WorkingMemoryError>(&error),
            Some(&WorkingMemoryError::IdentityMismatch)
        );
        assert_eq!(sampling.sampling.temperature, original_temperature);
        assert_eq!(sampling.sampling.prng.is_some(), had_rng);
        assert_eq!(sampling.sampling.next_prediction, 0);
        before_override.assert_no_work(runtime, controller);
        before_override.assert_uncharged(pool);
    }
    let escaped_identity = bound.shared_cache_identity().unwrap().clone();
    let identity_bytes = escaped_identity.capacity_bytes().unwrap();
    drop((sampling, bound, rebuilt, preparation));
    settle(pool, before.nonstate + identity_bytes);
    assert_eq!(
        escaped_identity.prepared().parts()[0].payload().shape(),
        [1, 5]
    );
    drop(escaped_identity);
}

#[cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
fn live_storage_bytes(
    runtime: Option<&ModelRuntime<MlxBackend<'_>>>,
    outputs: &[MlxTextToken],
) -> u64 {
    use crate::backend::runtime::residency::storage::RetainedStorage;

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
    for token in outputs {
        storage.include_array(&token.value).unwrap();
    }
    // The temporary inventory retires here, so it cannot mask the following
    // session-first or output-first physical lifetime assertions.
    storage
        .byte_bound()
        .unwrap()
        .expect("completed model and output storage must have a complete physical inventory")
}

#[cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
fn accepted_run(
    mut runtime: ModelRuntime<MlxBackend<'static>>,
    pool: &MemoryLedger,
    temperature: f32,
    capacity: u64,
    controlled: bool,
    controller: Controller,
) -> Vec<u32> {
    let initial_storage = live_storage_bytes(Some(&runtime), &[]);
    settle(pool, initial_storage);
    let outputs = if controlled {
        ControlledTextGeneration::from_input(
            &mut runtime,
            TextGenerationInput::TokenIds(tokens()),
            config(temperature, Some(capacity)),
            controller.clone(),
        )
        .unwrap()
        .map(|token| token.unwrap().into_output())
        .collect::<Vec<_>>()
    } else {
        TextGeneration::new(&mut runtime, tokens(), config(temperature, Some(capacity)))
            .unwrap()
            .map(Result::unwrap)
            .collect::<Vec<_>>()
    };
    assert_eq!(outputs.len(), 3);
    let ids = outputs
        .iter()
        .map(|token| token.token_id().unwrap())
        .collect::<Vec<_>>();
    assert!(ids.iter().all(|id| *id < 64));
    if controlled {
        assert_eq!(controller.0.get().0, 3);
        assert_eq!(controller.0.get().1, 3);
    }
    let retained = runtime
        .session()
        .payload
        .model
        .erased()
        .retained_inference_authority()
        .unwrap();
    let request = retained
        .requests()
        .next()
        .expect("populated decoder retains its request");
    let charge = request
        .memory_reservation()
        .requirements()
        .get(crate::memory_fixture::topology().host_domain())
        .unwrap()
        .total()
        .unwrap();
    assert!(charge > 0);
    for (index, token) in outputs.iter().enumerate() {
        assert_eq!(token.step_receipt().unwrap().attempt(), index as u64);
        assert!(token
            .owner
            .inference_retention()
            .requests()
            .any(|held| held.validate_same_request(request).is_ok()));
    }
    assert!(runtime
        .session()
        .payload
        .model
        .erased()
        .state_snapshot()
        .iter()
        .all(|(position, _)| *position == 7));
    assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
    let live_storage = live_storage_bytes(Some(&runtime), &outputs);
    let session_storage = live_storage_bytes(Some(&runtime), &[]);
    let token_storage = live_storage_bytes(None, &outputs);
    assert!(session_storage > initial_storage);
    assert!(token_storage > 0);
    settle(pool, live_storage);
    let retained_increment = live_storage.checked_sub(initial_storage).unwrap();
    assert!(
        retained_increment < charge / 2,
        "completed sampling must release workspace: original={charge}, retained={retained_increment}"
    );
    assert!(live_storage <= capacity);
    assert!(pool.fixture_host_peak().unwrap() <= capacity);
    eprintln!(
        "funded Metal quote: temperature={temperature}, controlled={controlled}, \
         original={charge}, retained_increment={retained_increment}, \
         live={live_storage}, session={session_storage}, tokens={token_storage}, capacity={capacity}"
    );
    drop(retained);
    if controlled {
        // Escaped outputs retain exactly their shared physical backing after
        // model, decoder, sampler and exact completion wrappers retire.
        drop(runtime);
        settle(pool, token_storage);
        assert_eq!(
            outputs
                .iter()
                .map(|token| token.token_id().unwrap())
                .collect::<Vec<_>>(),
            ids
        );
        drop(outputs);
    } else {
        // Populated decoder and model storage remain independently charged
        // after all emitted outputs retire, without a full workspace hold.
        drop(outputs);
        settle(pool, session_storage);
        let retained = runtime
            .session()
            .payload
            .model
            .erased()
            .retained_inference_authority()
            .unwrap();
        assert_eq!(
            retained
                .requests()
                .next()
                .unwrap()
                .memory_reservation()
                .requirements()
                .get(crate::memory_fixture::topology().host_domain())
                .unwrap()
                .total()
                .unwrap(),
            charge
        );
        drop(retained);
        drop(runtime);
    }
    settle(pool, 0);
    ids
}

#[test]
#[cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
fn metal_real_quote_is_cold_and_either_explains_a_gap_or_runs_multiple_receipted_tokens() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    for temperature in [0.0, 0.7] {
        let mut sequences = Vec::new();
        for controlled in [false, true] {
            let pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
            let (mut runtime, _artifact) = runtime(&stream, &pool);
            let controller = Controller::default();
            let before = ColdState::capture(&runtime, &pool);
            let model = &runtime.session().payload.model;
            assert!(model.has_workspace_mechanisms());
            let geometry = InferenceGeometry {
                batch_size: 1,
                cached_positions: 0,
                input_positions: 5,
                max_output_tokens: 3,
                prefill_chunk_positions: 1,
                output: OutputDemand::LastPosition,
            };
            let generation = model
                .quote_replicated_resident_text_with_sampling(
                    geometry,
                    config(temperature, Some(u64::MAX)),
                    &TokenFilter::All,
                )
                .unwrap();
            let prompt = model
                .quote_text_prompt_workspace(geometry, Some(20))
                .unwrap();
            let complete = generation.equations.transient().bytes().is_some()
                && generation.equations.retained_peak_bytes().is_some()
                && generation.sampling.peak.bytes().is_some()
                && prompt.peak().bytes().is_some();
            eprintln!("Metal text quote temperature={temperature}: equations={:?}, state={:?}, equation_gap={:?}, sampling={:?}, sampling_gap={:?}, prompt={:?}",
                generation.equations.transient(), generation.equations.retained_peak_bytes(),
                generation.equations.first_gap(), generation.sampling.peak,
                generation.sampling.first_gap, prompt.peak());
            before.assert_no_work(&runtime, &controller);
            before.assert_uncharged(&pool);
            let admitted = text_quote::admit(
                &runtime,
                &evidence(),
                config(temperature, Some(u64::MAX)),
                &controller,
            );
            if !complete {
                assert_missing_bound(
                    &admitted
                        .err()
                        .expect("missing native component cannot admit"),
                );
                let _failure = MlxBackend::fail_next_sampling_for_test(Error::ArchitectureModel(
                    "missing Metal quote component must reject before sampler construction".into(),
                ));
                let error = ControlledTextGeneration::from_input(
                    &mut runtime,
                    TextGenerationInput::TokenIds(tokens()),
                    config(temperature, Some(u64::MAX)),
                    controller.clone(),
                )
                .err()
                .expect("public admission cannot fill a native quote gap");
                assert_missing_bound(&error);
                assert!(TEST_SAMPLING_FAILURE.with(|slot| slot.borrow().is_some()));
                before.assert_no_work(&runtime, &controller);
                before.assert_uncharged(&pool);
                drop(runtime);
                settle(&pool, 0);
                continue;
            }
            let (preparation, proof) = admitted.unwrap_or_else(|error| {
                panic!("complete real Metal components failed admission: {error}")
            });
            let charge = preparation
                .request()
                .memory_reservation()
                .requirements()
                .get(crate::memory_fixture::topology().host_domain())
                .unwrap()
                .total()
                .unwrap();
            let capacity = pool.fixture_host_charge().unwrap();
            assert_eq!(capacity, before.bytes + charge);
            assert!(charge > 0);
            assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
            before.assert_no_work(&runtime, &controller);
            eprintln!("Metal admitted temperature={temperature}, controlled={controlled}, existing={}, request={charge}, capacity={capacity}", before.bytes);
            checked_prompt_provenance(
                &mut runtime,
                &pool,
                &controller,
                MlxTextPreparation {
                    request: Some(preparation),
                    chunk: config(temperature, Some(u64::MAX))
                        .inference_policy()
                        .prefill_chunk_positions,
                    quote: Some(proof),
                },
            );
            settle(&pool, before.bytes);
            sequences.push(accepted_run(
                runtime,
                &pool,
                temperature,
                capacity,
                controlled,
                controller,
            ));
        }
        assert!(sequences.is_empty() || sequences.len() == 2);
        assert!(sequences.windows(2).all(|pair| pair[0] == pair[1]));
    }
}

#[cfg(test)]
#[allow(unused_imports)]
use crate::memory_fixture::LedgerFixture;
