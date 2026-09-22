use super::*;
use eredu_core::{
    AdmissionResult, ControlledTextGeneration, OutputDemand, PendingTextInput, TextGeneration,
    TextGenerationInput, TextPreparationInput,
};
use eredu_runtime::{execution_control::TextSnapshotBackend, working_memory::*};

struct AllTokens;
impl eredu_core::TokenFilterController for AllTokens {
    type Error = std::convert::Infallible;
    fn current_filter(&mut self) -> Result<TokenFilter, Self::Error> {
        Ok(TokenFilter::All)
    }
    fn commit_token(&mut self, _: u32) -> Result<(), Self::Error> {
        Ok(())
    }
    fn is_complete(&mut self) -> Result<bool, Self::Error> {
        Ok(false)
    }
}

fn config() -> TextGenerationConfig {
    TextGenerationConfig::new(
        eredu_core::resolve_generation_config(
            None,
            eredu_core::GenerationConfigOverrides {
                temperature: Some(0.7),
                max_new_tokens: Some(2),
                ..Default::default()
            },
        )
        .unwrap(),
    )
    .with_seed(19)
}
fn geometry() -> eredu_core::InferenceGeometry {
    eredu_core::InferenceGeometry {
        batch_size: 1,
        cached_positions: 0,
        input_positions: 5,
        max_output_tokens: 2,
        prefill_chunk_positions: 2,
        output: OutputDemand::LastPosition,
    }
}
fn runtime<'a>(
    stream: &'a Stream,
    source_stream: &'a Stream,
) -> (ModelRuntime<MlxBackend<'a>>, tempfile::TempDir) {
    let backend = MlxBackend::new(stream, source_stream);
    let root = crate::composition::mlx::replicated_text::tests::tiny_artifact("llama", true);
    let prepared =
        eredu_core::load_model(&backend, root.path(), crate::MlxLoadRequest::default()).unwrap();
    (
        ModelRuntime::from_prepared(backend, prepared).unwrap(),
        root,
    )
}

// Synthetic admission tests charge ownership, not complete native peak quoting.
fn reserve(runtime: &ModelRuntime<MlxBackend<'_>>, pool: &MemoryLedger) -> InferenceRequest {
    use eredu_core::{EstimationCompleteness as Complete, WorkspaceBound};
    let g = geometry();
    let request = eredu_core::AdmissionRequest {
        input: eredu_core::InputTokenCount::text(g.input_positions),
        max_output_tokens: g.max_output_tokens,
        batch_size: 1,
        additional_headroom: crate::memory_fixture::headroom(0),
        memory_limits: Default::default(),
    };
    let layout = eredu_core::StateMemoryLayout::new(
        eredu_core::LayerSchedule::new(
            1,
            vec![eredu_core::cache::LayerCachePolicy::key_value(
                eredu_core::AttentionPolicy::Full,
                1,
                8,
            )
            .unwrap()],
        )
        .unwrap(),
        vec![0],
        32,
        1,
        Complete::Complete,
    )
    .unwrap();
    let bound = |bytes| {
        WorkspaceBound::bounded(
            bytes,
            "synthetic authority-lifetime fixture, not a native peak quote",
        )
    };
    let state = eredu_core::estimate_runtime_state(
        &layout,
        request.input,
        g.max_output_tokens,
        1,
        std::num::NonZeroU8::new(4).unwrap(),
    )
    .unwrap()
    .with_execution_workspace(crate::memory_fixture::workspace(
        eredu_core::ExecutionWorkspaceEstimate {
            physical_domains: None,
            geometry: g,
            activations: bound(1 << 24),
            attention: bound(0),
            vocabulary: bound(0),
            state_update: bound(0),
            materialization: bound(0),
            retained: bound(4096),
        },
    ))
    .unwrap();
    let capabilities = eredu_core::ModelCapabilities {
        effective_model_type: "native lifetime fixture".into(),
        native_max_context: eredu_core::Observed::exact(32, "fixture"),
        effective_max_context: eredu_core::Observed::exact(32, "fixture"),
        state_strategy: eredu_core::CacheStateStrategy::FullKv,
        modalities: eredu_core::InputModalities::TEXT,
        estimation: Complete::Complete,
    };
    let AdmissionResult::Admitted(admitted) =
        eredu_core::apply_admission_policy(&capabilities, request, state).unwrap()
    else {
        panic!("fixture admission")
    };
    pool.reserve(
        runtime
            .session()
            .payload
            .model
            .erased()
            .inference_execution_identity(),
        &admitted,
    )
    .unwrap()
    .into()
}
fn memory_error<'a>(
    error: &'a (dyn std::error::Error + 'static),
) -> Option<&'a WorkingMemoryError> {
    let mut error = error;
    loop {
        if let Some(found) = error.downcast_ref::<WorkingMemoryError>() {
            return Some(found);
        }
        error = error.source()?;
    }
}

#[test]
fn native_preparation_public_maximum_preserves_a_smaller_admitted_chunk() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let (runtime, _root) = runtime(&stream, &stream);
    for maximum in [1, 2, 4, 16] {
        let pool = crate::memory_fixture::ledger(1 << 26, 0).unwrap();
        let request = reserve(&runtime, &pool);
        let prompt = MlxBackend::prepare_text_prompt(runtime.backend(), vec![1, 2, 3, 4, 5])
            .unwrap()
            .with_inference_request(request);
        let config = config().with_inference_policy(eredu_core::TextInferencePolicy {
            prefill_chunk_positions: std::num::NonZeroU64::new(maximum),
            memory_limits: eredu_core::MemoryLimitDeclarations::unlimited(),
            submission_tracking_capacity_bytes: None,
            graph_metadata_capacity_bytes: None,
        });
        let admitted = MlxBackend::admit_text_preparation(
            &runtime,
            &TextPreparationInput::Prepared(&prompt),
            config.clone(),
            &AllTokens,
        );
        if maximum < geometry().prefill_chunk_positions {
            assert_eq!(
                memory_error(&admitted.unwrap_err()),
                Some(&WorkingMemoryError::IdentityMismatch)
            );
        } else {
            let admitted = admitted.unwrap();
            assert_eq!(
                admitted.request.as_ref().unwrap().request().geometry(),
                geometry()
            );
            let prompt =
                MlxBackend::bind_text_prompt_preparation(runtime.backend(), prompt, &admitted)
                    .unwrap();
            assert_eq!(
                prompt.prefill_chunk_positions.unwrap().get(),
                geometry().prefill_chunk_positions
            );
        }
    }
}

#[test]
fn native_preparation_nonstate_inventory_keeps_weights_separate_from_quoted_cache() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let (mut runtime, _root) = runtime(&stream, &stream);
    let model = &runtime.session().payload.model;
    assert_eq!(
        model
            .erased()
            .retained_target_storage()
            .unwrap()
            .byte_bound()
            .unwrap(),
        model
            .erased()
            .retained_target_nonstate_storage()
            .unwrap()
            .byte_bound()
            .unwrap()
    );
    let tokens = TextGeneration::new(&mut runtime, vec![1, 2, 3, 4, 5], config())
        .unwrap()
        .map(Result::unwrap)
        .collect::<Vec<_>>();
    assert_eq!(tokens.len(), 2);
    let model = &runtime.session().payload.model;
    let mut full = model.erased().retained_target_storage().unwrap();
    let static_storage = model.erased().retained_target_nonstate_storage().unwrap();
    let full_bytes = full.byte_bound().unwrap().unwrap();
    let static_bytes = static_storage.byte_bound().unwrap().unwrap();
    assert!(static_bytes > 0);
    assert!(
        full_bytes > static_bytes,
        "decoder state must remain outside static registration when its backing is quoted"
    );
    full.merge(static_storage).unwrap();
    assert_eq!(
        full.byte_bound().unwrap(),
        Some(full_bytes),
        "static source and native allocation identities deduplicate"
    );
}

#[test]
fn native_preparation_public_policy_chunks_ordinary_and_controlled_generation() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let mut sequences = Vec::new();
    for chunk in [None, Some(1), Some(2), Some(9)] {
        for prepared in [false, true] {
            for controlled in [false, true] {
                let (mut runtime, _root) = runtime(&stream, &stream);
                let config = config().with_inference_policy(eredu_core::TextInferencePolicy {
                    prefill_chunk_positions: chunk.and_then(std::num::NonZeroU64::new),
                    memory_limits: eredu_core::MemoryLimitDeclarations::unlimited(),
                    submission_tracking_capacity_bytes: None,
                    graph_metadata_capacity_bytes: None,
                });
                let ids = vec![1, 2, 3, 4, 5];
                let input = if prepared {
                    TextGenerationInput::Prepared(
                        MlxBackend::prepare_text_prompt(runtime.backend(), ids).unwrap(),
                    )
                } else {
                    TextGenerationInput::TokenIds(ids)
                };
                let tokens = if controlled {
                    ControlledTextGeneration::from_input(&mut runtime, input, config, AllTokens)
                        .unwrap()
                        .map(|token| token.unwrap().into_output())
                        .collect::<Vec<_>>()
                } else {
                    match input {
                        TextGenerationInput::TokenIds(ids) => {
                            TextGeneration::new(&mut runtime, ids, config)
                        }
                        TextGenerationInput::OriginalTokenIds => {
                            unreachable!("legacy prepared-input fixture")
                        }
                        TextGenerationInput::OriginalPrepared(_) => {
                            unreachable!("ordinary fixture input")
                        }
                        TextGenerationInput::Prepared(prompt) => {
                            TextGeneration::from_prompt(&mut runtime, prompt, config)
                        }
                    }
                    .unwrap()
                    .map(Result::unwrap)
                    .collect::<Vec<_>>()
                };
                assert_eq!(tokens.len(), 2);
                sequences.push(
                    tokens
                        .iter()
                        .map(|token| token.token_id().unwrap())
                        .collect::<Vec<_>>(),
                );
            }
        }
    }
    assert!(sequences.windows(2).all(|pair| pair[0] == pair[1]));
}

#[test]
fn native_preparation_unknown_public_domain_rejects_before_sampling_or_controller_work() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    for capacity in [0, u64::MAX] {
        let (mut runtime, _root) = runtime(&stream, &stream);
        let _failure = MlxBackend::fail_next_sampling_for_test(Error::ArchitectureModel(
            "sampling must not start".into(),
        ));
        let config = config().with_inference_policy(eredu_core::TextInferencePolicy {
            prefill_chunk_positions: std::num::NonZeroU64::new(2),
            memory_limits: eredu_core::MemoryLimitDeclarations::new([(
                "host".into(),
                eredu_core::MemoryLimit::Finite(capacity),
            )]),
            submission_tracking_capacity_bytes: None,
            graph_metadata_capacity_bytes: None,
        });
        let before = runtime.session().payload.model.erased().state_snapshot();
        let error = TextGeneration::new(&mut runtime, vec![1, 2, 3, 4, 5], config.clone())
            .err()
            .unwrap();
        assert_eq!(
            memory_error(&error),
            Some(&WorkingMemoryError::UnknownBound)
        );
        assert!(TEST_SAMPLING_FAILURE.with(|slot| slot.borrow().is_some()));
        assert_eq!(
            runtime.session().payload.model.erased().state_snapshot(),
            before
        );
        // A supplied private reservation is not evidence for the public domain.
        let pool = crate::memory_fixture::ledger(1 << 26, 0).unwrap();
        let request = reserve(&runtime, &pool);
        let prompt = MlxBackend::prepare_text_prompt(runtime.backend(), vec![1, 2, 3, 4, 5])
            .unwrap()
            .with_inference_request(request);
        let error = MlxBackend::admit_text_preparation(
            &runtime,
            &TextPreparationInput::Prepared(&prompt),
            config.clone(),
            &AllTokens,
        )
        .unwrap_err();
        assert_eq!(
            memory_error(&error),
            Some(&WorkingMemoryError::UnknownBound)
        );
        assert!(TEST_SAMPLING_FAILURE.with(|slot| slot.borrow().is_some()));
    }
}

#[test]
fn native_preparation_authority_rejects_duplicates_and_changed_sampling_before_factory_work() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let (runtime, _root) = runtime(&stream, &stream);
    let pool = crate::memory_fixture::ledger(1 << 26, 0).unwrap();
    let request = reserve(&runtime, &pool);
    let prompt = MlxBackend::prepare_text_prompt(runtime.backend(), vec![1, 2, 3, 4, 5])
        .unwrap()
        .with_inference_request(request.clone());
    let preparation = MlxBackend::admit_text_preparation(
        &runtime,
        &TextPreparationInput::Prepared(&prompt),
        config(),
        &AllTokens,
    )
    .unwrap();
    let duplicate = MlxBackend::admit_text_preparation(
        &runtime,
        &TextPreparationInput::Prepared(&prompt),
        config(),
        &AllTokens,
    )
    .unwrap_err();
    assert_eq!(
        memory_error(&duplicate),
        Some(&WorkingMemoryError::PreparationAlreadyStarted)
    );
    let _failure = MlxBackend::fail_next_sampling_for_test(Error::ArchitectureModel(
        "sampling factory entered".into(),
    ));
    let changed = MlxBackend::start_text_generation_admitted(
        runtime.backend(),
        config().with_seed(20),
        &preparation,
    )
    .err()
    .unwrap();
    assert_eq!(
        memory_error(&changed),
        Some(&WorkingMemoryError::PreparationConfigurationMismatch)
    );
    assert!(TEST_SAMPLING_FAILURE.with(|slot| slot.borrow().is_some()));
    let failed =
        MlxBackend::start_text_generation_admitted(runtime.backend(), config(), &preparation)
            .err()
            .unwrap();
    assert!(failed.to_string().contains("sampling factory entered"));
    assert!(TEST_SAMPLING_FAILURE.with(|slot| slot.borrow().is_none()));
    let duplicate = MlxBackend::start_text_generation_admitted(
        runtime.backend(),
        config(),
        &preparation.clone(),
    )
    .err()
    .unwrap();
    assert_eq!(
        memory_error(&duplicate),
        Some(&WorkingMemoryError::PreparationAlreadyStarted)
    );
    drop((preparation, prompt, request));
    assert_eq!(pool.fixture_host_charge().unwrap(), 0);
}

#[test]
fn incomplete_native_preparation_rejects_prompt_and_sampler_without_mutation() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let (mut runtime, _root) = runtime(&stream, &stream);
    let pool = crate::memory_fixture::ledger(1 << 26, 0).unwrap();
    let request = reserve(&runtime, &pool);
    let preparation = MlxTextPreparation {
        request: Some(
            request
                .prepare_text(
                    runtime
                        .session()
                        .payload
                        .model
                        .erased()
                        .inference_execution_identity(),
                    geometry(),
                    config(),
                )
                .unwrap(),
        ),
        chunk: None,
        quote: None,
    };
    let before = (
        pool.fixture_host_charge().unwrap(),
        runtime.session().test_state_presence(),
    );
    let prompt_error = MlxBackend::prepare_text_prompt_admitted(
        runtime.backend(),
        vec![1, 2, 3, 4, 5],
        &preparation,
    )
    .err()
    .unwrap();
    let sampler_error =
        MlxBackend::start_text_generation_admitted(runtime.backend(), config(), &preparation)
            .err()
            .unwrap();
    for error in [prompt_error, sampler_error] {
        assert_eq!(
            memory_error(&error),
            Some(&WorkingMemoryError::UnknownBound)
        );
    }
    assert_eq!(
        (
            pool.fixture_host_charge().unwrap(),
            runtime.session().test_state_presence()
        ),
        before
    );
}

#[test]
fn native_preparation_authority_stays_with_escaped_tokens_after_both_generation_drivers_drop() {
    let source_stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    for device in [safemlx::DeviceType::Cpu, safemlx::DeviceType::Gpu] {
        if device == safemlx::DeviceType::Gpu && !cfg!(feature = "metal") {
            continue;
        }
        let stream = Stream::new_with_device(&safemlx::Device::new(device, 0));
        let mut sequences = Vec::new();
        for controlled in [false, true] {
            let (mut runtime, _root) = runtime(&stream, &source_stream);
            let pool = crate::memory_fixture::ledger(1 << 26, 0).unwrap();
            let request = reserve(&runtime, &pool);
            let charged = pool.fixture_host_charge().unwrap();
            let prompt = MlxBackend::prepare_text_prompt(runtime.backend(), vec![1, 2, 3, 4, 5])
                .unwrap()
                .with_inference_request(request.clone());
            let tokens = if controlled {
                ControlledTextGeneration::from_input(
                    &mut runtime,
                    TextGenerationInput::Prepared(prompt),
                    config(),
                    AllTokens,
                )
                .unwrap()
                .map(|token| token.unwrap().into_output())
                .collect::<Vec<_>>()
            } else {
                TextGeneration::from_prompt(&mut runtime, prompt, config())
                    .unwrap()
                    .map(Result::unwrap)
                    .collect::<Vec<_>>()
            };
            assert_eq!(tokens.len(), 2);
            let ids = tokens
                .iter()
                .map(|token| token.token_id().unwrap())
                .collect::<Vec<_>>();
            sequences.push(ids.clone());
            let duplicate = tokens[0].clone();
            drop((request, runtime));
            // Session payloads retire outside native locks. Drain that cleanup
            // while token owners are still live so they prove independent
            // retention after the sampler and executable have retired.
            crate::backend::ordinary_retirement::reclaim_all();
            assert_eq!(
                pool.fixture_host_charge().unwrap(),
                charged,
                "returned token owns its charge after session retirement"
            );
            assert_eq!(duplicate.token_id().unwrap(), ids[0]);
            drop(tokens);
            assert_eq!(pool.fixture_host_charge().unwrap(), charged);
            drop(duplicate);
            crate::backend::submission_recovery::wait_for_retirement(|| {
                pool.fixture_host_charge().unwrap() == 0
            });
            assert_eq!(pool.fixture_host_charge().unwrap(), 0);
            assert_eq!(pool.fixture_host_peak().unwrap(), charged);
        }
        assert_eq!(sequences[0], sequences[1]);
    }
}

#[test]
fn legacy_prepared_identity_alias_retains_reservation_after_prompt_and_request_retire() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let (runtime, artifact) = runtime(&stream, &stream);
    // Existing synthetic reservation tests prove legacy custody only. Native
    // execution belongs to the separate fixture domain, not this byte quote.
    let pool = crate::memory_fixture::ledger(1 << 26, 0).unwrap();
    let request = reserve(&runtime, &pool);
    let charged = pool.fixture_host_charge().unwrap();
    let preparation = MlxTextPreparation {
        request: Some(
            request
                .prepare_text(
                    runtime
                        .session()
                        .payload
                        .model
                        .erased()
                        .inference_execution_identity(),
                    geometry(),
                    config(),
                )
                .unwrap(),
        ),
        chunk: None,
        quote: None,
    };
    let prompt = MlxBackend::prepare_text_prompt_admitted(
        runtime.backend(),
        vec![17, 3, 29, 7, 11],
        &preparation,
    )
    .unwrap();
    prompt.with_borrowed(|input| {
        let _ = input.parts[0].payload().value().evaluated().unwrap();
    });
    let identity = prompt.shared_cache_identity().unwrap().clone();
    let second = identity.clone();
    let expected = eredu_core::cache::prompt_cache_token_fingerprint(&[17, 3, 29, 7, 11]);
    assert!(charged > identity.capacity_bytes().unwrap());
    drop((prompt, preparation, request, runtime, artifact));
    crate::backend::nn::shared::MlxNeuralBackend::reclaim_retired_resources();
    safemlx::reclaim_allocation_owners();
    assert_eq!(identity.semantic_content_fingerprint(), expected);
    assert_eq!(pool.fixture_host_charge().unwrap(), charged);
    assert!(matches!(
        pool.acquire_unquoted(),
        Err(WorkingMemoryError::ReservedWorkActive)
    ));
    drop(identity);
    assert_eq!(pool.fixture_host_charge().unwrap(), charged);
    drop(second);
    crate::backend::submission_recovery::wait_for_retirement(|| {
        crate::backend::nn::shared::MlxNeuralBackend::reclaim_retired_resources();
        safemlx::reclaim_allocation_owners();
        pool.fixture_host_charge().unwrap() == 0
    });
    assert!(pool.acquire_unquoted().is_ok());
}

#[test]
fn unquoted_prompt_identity_alias_keeps_original_exclusion_without_native_roots() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    for (rebuilt, replacement) in [(false, false), (false, true), (true, true)] {
        let pool = crate::memory_fixture::ledger(0, 0).unwrap();
        let backend = MlxBackend::new(&stream, &stream).with_memory_ledger(pool.clone());
        let prompt = MlxBackend::prepare_text_prompt(&backend, vec![17, 3, 29, 7, 11]).unwrap();
        let prompt = if rebuilt {
            let alias = prompt.with_borrowed(|input| MlxModelInput::from(input));
            drop(prompt);
            alias
        } else {
            prompt
        };
        let prompt = if replacement {
            prompt
                .with_semantic_content_fingerprint("replacement-content-17")
                .unwrap()
        } else {
            prompt
        };
        prompt.with_borrowed(|input| {
            let _ = input.parts[0].payload().value().evaluated().unwrap();
        });
        let identity = prompt.shared_cache_identity().unwrap().clone();
        let second = identity.clone();
        drop((prompt, backend));
        crate::backend::nn::shared::MlxNeuralBackend::reclaim_retired_resources();
        safemlx::reclaim_allocation_owners();
        assert_eq!(pool.unquoted_owner_count().unwrap(), 1);
        assert_eq!(identity.prepared().parts()[0].payload().shape(), [1, 5]);
        if replacement {
            assert_eq!(
                identity.semantic_content_fingerprint(),
                "replacement-content-17"
            );
        }
        drop(identity);
        assert_eq!(pool.unquoted_owner_count().unwrap(), 1);
        drop(second);
        crate::backend::submission_recovery::wait_for_retirement(|| {
            crate::backend::nn::shared::MlxNeuralBackend::reclaim_retired_resources();
            safemlx::reclaim_allocation_owners();
            pool.unquoted_owner_count().unwrap() == 0
        });
    }
}

#[cfg(test)]
#[allow(unused_imports)]
use crate::memory_fixture::LedgerFixture;
