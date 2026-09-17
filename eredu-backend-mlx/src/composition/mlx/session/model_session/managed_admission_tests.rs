use super::*;
use crate::backend::nn::shared::MlxNeuralBackend;
use eredu_core::{
    ControlledTextGeneration, TextGeneration, TextGenerationInput, TextPreparationInput,
};
use eredu_runtime::working_memory::{WorkingMemoryError, WorkingMemoryPool};

struct AllTokens;
impl eredu_core::TokenFilterController for AllTokens {
    type Error = std::convert::Infallible;

    fn inference_workspace_is_run_owned(&self) -> bool {
        true
    }

    fn inference_workspace(&self, _: u64) -> Option<eredu_core::TextControllerWorkspace<'_>> {
        Some(eredu_core::TextControllerWorkspace {
            filter: (&TokenFilter::All).into(),
            additional_host_bytes: 0,
        })
    }
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

struct UnquotedController;
impl eredu_core::TokenFilterController for UnquotedController {
    type Error = std::convert::Infallible;
    fn current_filter(&mut self) -> Result<TokenFilter, Self::Error> {
        panic!("unquoted controller must reject before decisions")
    }
    fn commit_token(&mut self, _: u32) -> Result<(), Self::Error> {
        panic!("unquoted controller must reject before commitment")
    }
    fn is_complete(&mut self) -> Result<bool, Self::Error> {
        panic!("unquoted controller must reject before completion checks")
    }
}

fn config(capacity: Option<u64>) -> TextGenerationConfig {
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
    .with_inference_policy(eredu_core::TextInferencePolicy {
        // Exact-capacity rejection uses the smallest admissible chunk; a
        // larger initial chunk could legitimately shrink to fit capacity - 1.
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
    TextPreparationInput::TokenIds {
        positions: 5,
        capacity_bytes: 20,
    }
}

fn runtime<'a>(
    stream: &'a Stream,
    pool: &WorkingMemoryPool,
) -> (ModelRuntime<MlxBackend<'a>>, tempfile::TempDir) {
    let weights = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let backend = MlxBackend::new(stream, &weights).with_memory_pool(pool.clone());
    let artifact = crate::composition::mlx::replicated_text::tests::tiny_artifact("llama", true);
    let model = eredu_core::load_model(&backend, artifact.path(), crate::MlxLoadRequest::default())
        .unwrap();
    (
        ModelRuntime::from_prepared(backend, model).unwrap(),
        artifact,
    )
}

fn memory_error<'a>(
    error: &'a (dyn std::error::Error + 'static),
) -> Option<&'a WorkingMemoryError> {
    let mut current = error;
    loop {
        if let Some(memory) = current.downcast_ref::<WorkingMemoryError>() {
            return Some(memory);
        }
        current = current.source()?;
    }
}

fn settle(pool: &WorkingMemoryPool, bytes: u64) {
    crate::backend::submission_recovery::wait_for_retirement(|| {
        MlxNeuralBackend::reclaim_retired_resources();
        safemlx::reclaim_allocation_owners();
        pool.used_bytes().unwrap() == bytes
    });
}

// Measure a real successful public admission; no synthetic state or enclosing
// workspace estimate enters these fixtures. The source Vec's actual capacity is
// the same twenty bytes used by ordinary/controlled preparation below.
#[cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
fn public_charge(runtime: &ModelRuntime<MlxBackend<'_>>, pool: &WorkingMemoryPool) -> (u64, u64) {
    let preparation = MlxBackend::admit_text_preparation(
        runtime,
        &evidence(),
        config(Some(u64::MAX)),
        &AllTokens,
    )
    .unwrap();
    let request = preparation.request.as_ref().unwrap().request();
    let bytes = request
        .memory_reservation()
        .expect("public managed admission must reserve")
        .bytes();
    assert!(bytes > 0);
    let total = pool.used_bytes().unwrap();
    assert!(
        total > bytes,
        "resident source and parameter storage is charged separately"
    );
    assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
    drop(preparation);
    settle(pool, total - bytes);
    (total, bytes)
}

#[test]
#[cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
fn public_resident_preparation_succeeds_at_its_exact_complete_domain_capacity() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (runtime, _artifact) = runtime(&stream, &pool);
    let before = runtime.session().payload.model.erased().state_snapshot();
    let (capacity, request_bytes) = public_charge(&runtime, &pool);
    let preparation = MlxBackend::admit_text_preparation(
        &runtime,
        &evidence(),
        config(Some(capacity)),
        &AllTokens,
    )
    .unwrap();
    assert_eq!(pool.used_bytes().unwrap(), capacity);
    assert_eq!(
        preparation
            .request
            .as_ref()
            .unwrap()
            .request()
            .memory_reservation()
            .unwrap()
            .bytes(),
        request_bytes
    );
    let prompt =
        MlxBackend::prepare_text_prompt_admitted(runtime.backend(), tokens(), &preparation)
            .unwrap();
    let prompt =
        MlxBackend::bind_text_prompt_preparation(runtime.backend(), prompt, &preparation).unwrap();
    let sampler = MlxBackend::start_text_generation_admitted(
        runtime.backend(),
        config(Some(capacity)),
        &preparation,
    )
    .unwrap();
    assert_eq!(pool.used_bytes().unwrap(), capacity);
    assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
    assert_eq!(
        runtime.session().payload.model.erased().state_snapshot(),
        before
    );
    drop((sampler, prompt, preparation));
    settle(&pool, capacity - request_bytes);
    drop(runtime);
    settle(&pool, 0);
}

#[test]
#[cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
fn public_resident_one_byte_short_rejects_before_prompt_sampling_or_state_mutation() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (mut runtime, _artifact) = runtime(&stream, &pool);
    let (capacity, _) = public_charge(&runtime, &pool);
    let before = runtime.session().payload.model.erased().state_snapshot();
    let usage = pool.used_bytes().unwrap();
    let peak = pool.peak_bytes().unwrap();
    let _failure = MlxBackend::fail_next_sampling_for_test(Error::ArchitectureModel(
        "sampling must follow admission".into(),
    ));
    let error = TextGeneration::new(&mut runtime, tokens(), config(Some(capacity - 1)))
        .err()
        .unwrap();
    assert!(
        matches!(
            memory_error(&error),
            Some(WorkingMemoryError::BudgetExceeded { .. })
        ),
        "{error}"
    );
    assert!(TEST_SAMPLING_FAILURE.with(|slot| slot.borrow().is_some()));
    assert_eq!(
        runtime.session().payload.model.erased().state_snapshot(),
        before
    );
    assert_eq!(pool.used_bytes().unwrap(), usage);
    assert_eq!(pool.peak_bytes().unwrap(), peak);
    assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
    drop(runtime);
    settle(&pool, 0);
}

#[test]
#[cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
fn public_preparations_share_one_domain_capacity_and_release_only_their_own_reservations() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (runtime, _artifact) = runtime(&stream, &pool);
    let (single, request_bytes) = public_charge(&runtime, &pool);
    let baseline = single - request_bytes;
    let capacity = single.checked_add(request_bytes).unwrap();
    let first = MlxBackend::admit_text_preparation(
        &runtime,
        &evidence(),
        config(Some(capacity)),
        &AllTokens,
    )
    .unwrap();
    let second = MlxBackend::admit_text_preparation(
        &runtime,
        &evidence(),
        config(Some(capacity)),
        &AllTokens,
    )
    .unwrap();
    assert_eq!(pool.used_bytes().unwrap(), capacity);
    let rejected = MlxBackend::admit_text_preparation(
        &runtime,
        &evidence(),
        config(Some(capacity)),
        &AllTokens,
    )
    .unwrap_err();
    assert!(
        matches!(
            memory_error(&rejected),
            Some(WorkingMemoryError::BudgetExceeded { .. })
        ),
        "{rejected}"
    );
    assert_eq!(pool.used_bytes().unwrap(), capacity);
    drop(first);
    settle(&pool, single);
    let replacement = MlxBackend::admit_text_preparation(
        &runtime,
        &evidence(),
        config(Some(capacity)),
        &AllTokens,
    )
    .unwrap();
    assert_eq!(pool.used_bytes().unwrap(), capacity);
    drop(second);
    settle(&pool, single);
    drop(replacement);
    settle(&pool, baseline);
    drop(runtime);
    settle(&pool, 0);
}

#[test]
#[cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
fn public_managed_generation_matches_ordinary_and_controlled_unbudgeted_tokens() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    let mut sequences = Vec::new();
    for managed in [false, true] {
        for controlled in [false, true] {
            let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
            let (mut runtime, _artifact) = runtime(&stream, &pool);
            let capacity = managed.then(|| public_charge(&runtime, &pool).0);
            let config = config(capacity);
            let tokens = if controlled {
                ControlledTextGeneration::from_input(
                    &mut runtime,
                    TextGenerationInput::TokenIds(tokens()),
                    config,
                    AllTokens,
                )
                .unwrap()
                .map(|token| token.unwrap().into_output())
                .collect::<Vec<_>>()
            } else {
                TextGeneration::new(&mut runtime, tokens(), config)
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
            if let Some(capacity) = capacity {
                assert!(pool.used_bytes().unwrap() <= capacity);
                assert!(pool.peak_bytes().unwrap() <= capacity);
                assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
            }
            drop((tokens, runtime));
            settle(&pool, 0);
        }
    }
    assert!(sequences.windows(2).all(|pair| pair[0] == pair[1]));
}

#[test]
fn public_managed_admission_preserves_typed_rejection_for_unproven_controller_and_native_facts() {
    for device in [safemlx::DeviceType::Cpu, safemlx::DeviceType::Gpu] {
        if device == safemlx::DeviceType::Gpu
            && !cfg!(all(
                target_vendor = "apple",
                feature = "metal",
                not(feature = "cuda")
            ))
        {
            continue;
        }
        let stream = Stream::new_with_device(&safemlx::Device::new(device, 0));
        let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
        let (runtime, _artifact) = runtime(&stream, &pool);
        let before = runtime.session().payload.model.erased().state_snapshot();
        let usage = pool.used_bytes().unwrap();
        let owners = pool.unquoted_owner_count().unwrap();
        let _failure = MlxBackend::fail_next_sampling_for_test(Error::ArchitectureModel(
            "unproven preparation must not reach sampling".into(),
        ));
        let controller = MlxBackend::admit_text_preparation(
            &runtime,
            &evidence(),
            config(Some(u64::MAX)),
            &UnquotedController,
        )
        .unwrap_err();
        assert_eq!(
            memory_error(&controller),
            Some(&WorkingMemoryError::UnknownBound)
        );
        // This CPU executable has no retained Metal workspace mechanism facts even
        // when the controller itself supplies a complete cold bound.
        if device == safemlx::DeviceType::Cpu {
            let native = MlxBackend::admit_text_preparation(
                &runtime,
                &evidence(),
                config(Some(u64::MAX)),
                &AllTokens,
            )
            .unwrap_err();
            assert_eq!(
                memory_error(&native),
                Some(&WorkingMemoryError::UnknownBound)
            );
        }
        assert!(TEST_SAMPLING_FAILURE.with(|slot| slot.borrow().is_some()));
        assert_eq!(
            runtime.session().payload.model.erased().state_snapshot(),
            before
        );
        assert_eq!(pool.used_bytes().unwrap(), usage);
        assert_eq!(pool.unquoted_owner_count().unwrap(), owners);
        drop(runtime);
        settle(&pool, 0);
    }
}
