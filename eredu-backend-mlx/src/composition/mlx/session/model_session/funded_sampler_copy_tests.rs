#![cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]

use super::*;
use crate::tests::support::path_instrumentation as paths;
use eredu_core::{
    TextGenerationContinuation, TextGenerationDriver, TextGenerationInput, TextPreparationInput,
    TokenFilterController,
};
use eredu_runtime::working_memory::{SamplerCopyLimits, WorkingMemoryError, WorkingMemoryPool};

struct AllTokens;
impl TokenFilterController for AllTokens {
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

fn config(adaptive: bool, managed: bool) -> TextGenerationConfig {
    let config = TextGenerationConfig::new(
        eredu_core::resolve_generation_config(
            None,
            eredu_core::GenerationConfigOverrides {
                temperature: Some(0.7),
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
        managed_memory_capacity_bytes: managed.then_some(u64::MAX),
        submission_tracking_capacity_bytes: None,
        graph_metadata_capacity_bytes: None,
    });
    if adaptive {
        config.with_mirostat_v2(5.0, 0.3).unwrap()
    } else {
        config
    }
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

fn runtime(pool: &WorkingMemoryPool) -> (ModelRuntime<MlxBackend<'static>>, tempfile::TempDir) {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    let source = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let backend = MlxBackend::new(&stream, &source).with_memory_pool(pool.clone());
    let artifact = crate::composition::mlx::replicated_text::tests::tiny_artifact("llama", true);
    let model = eredu_core::load_model(&backend, artifact.path(), crate::MlxLoadRequest::default())
        .unwrap();
    let runtime = ModelRuntime::from_prepared(backend, model).unwrap();
    crate::backend::submission_recovery::wait_for_retirement(|| {
        reclaim();
        pool.unquoted_owner_count().unwrap() == 0
    });
    assert!(runtime.session().payload.model.has_published_idle_storage());
    assert!(pool.used_bytes().unwrap() > 0);
    (runtime, artifact)
}

fn reclaim() {
    crate::backend::nn::shared::MlxNeuralBackend::reclaim_retired_resources();
    safemlx::reclaim_allocation_owners();
}

fn settle(pool: &WorkingMemoryPool, expected: u64) {
    crate::backend::submission_recovery::wait_for_retirement(|| {
        reclaim();
        pool.used_bytes().unwrap() == expected && pool.unquoted_owner_count().unwrap() == 0
    });
}

fn history(sampler: &eredu_runtime::ConfiguredTextSampler) -> &[u32] {
    match sampler {
        MlxTextSampler::Standard(sampler) => sampler.generated_tokens(),
        MlxTextSampler::MirostatV2(sampler) => sampler.generated_tokens(),
    }
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

fn accounting(pool: &WorkingMemoryPool) -> (u64, u64, u64) {
    (
        pool.used_bytes().unwrap(),
        pool.peak_bytes().unwrap(),
        pool.effective_capacity().unwrap(),
    )
}

fn advance_twice(
    driver: &mut TextGenerationDriver<'_, MlxBackend<'static>>,
    state: &mut TextGenerationContinuation<MlxBackend<'static>, AllTokens>,
) -> Vec<MlxTextToken> {
    (0..2)
        .map(|_| {
            let token = driver.advance(state).unwrap().unwrap().into_output();
            driver.take_completed_step(state).unwrap();
            token
        })
        .collect()
}

#[test]
fn admitted_standard_and_mirostat_sampler_copy_accept_exact_capacity_and_reject_one_short() {
    for adaptive in [false, true] {
        let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
        let (mut runtime, artifact) = runtime(&pool);
        let mut driver = TextGenerationDriver::new(&mut runtime);
        let mut state = driver
            .start_input(
                TextGenerationInput::TokenIds(tokens()),
                config(adaptive, true),
                AllTokens,
            )
            .unwrap();
        let outputs = advance_twice(&mut driver, &mut state);
        let ids = outputs
            .iter()
            .map(|token| token.token_id().unwrap())
            .collect::<Vec<_>>();
        let copy = {
            let boundary = driver.quiescent(&mut state).unwrap();
            let (runtime, generation, _) = boundary.parts();
            let source = &generation.sampling.sampler;
            assert!(generation.sampling.quote.is_some());
            assert_eq!(history(source.as_sampler()), ids);
            assert_eq!(source.as_sampler().history_len(), 2);
            let required = source.as_sampler().prepare_copy().unwrap().retained_bytes();
            assert!(required > 0);
            let key = generation
                .sampling
                .prng
                .as_ref()
                .unwrap()
                .as_array()
                .evaluated()
                .unwrap()
                .try_to_vec::<u32>()
                .unwrap();
            let before = accounting(&pool);
            let before_paths = paths::snapshot();
            let frontier = runtime.session().payload.model.erased().state_snapshot();
            let error = pool
                .copy_sampler(
                    source.borrow_funded().unwrap(),
                    SamplerCopyLimits::new(before.0 + required - 1),
                )
                .err()
                .expect("one byte below the exact destination component must reject");
            assert_eq!(
                cause::<WorkingMemoryError>(&error),
                Some(&WorkingMemoryError::BudgetExceeded {
                    required_bytes: required,
                    available_bytes: required - 1,
                })
            );
            assert_eq!(accounting(&pool), before);
            assert_eq!(paths::snapshot(), before_paths);
            assert_eq!(
                runtime.session().payload.model.erased().state_snapshot(),
                frontier
            );
            assert_eq!(history(source.as_sampler()), ids);
            let copy = pool
                .copy_sampler(
                    source.borrow_funded().unwrap(),
                    SamplerCopyLimits::new(before.0 + required),
                )
                .unwrap();
            assert_eq!(pool.used_bytes().unwrap(), before.0 + required);
            assert_eq!(copy.bytes(), required);
            assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
            assert_eq!(
                copy.as_sampler().history_capacity(),
                source.as_sampler().history_capacity()
            );
            assert_eq!(history(copy.as_sampler()), ids);
            assert!(!std::ptr::eq(
                history(copy.as_sampler()).as_ptr(),
                history(source.as_sampler()).as_ptr()
            ));
            // Configured Debug contains the actual standard controls, complete
            // boxed history and Mirostat scalars, without account/owner pointers.
            assert_eq!(
                format!("{:?}", copy.as_sampler()),
                format!("{:?}", source.as_sampler())
            );
            if let MlxTextSampler::MirostatV2(sampler) = copy.as_sampler() {
                assert_eq!((sampler.tau(), sampler.eta()), (5.0, 0.3));
                assert_ne!(sampler.mu(), 10.0);
            }
            assert_eq!(
                generation
                    .sampling
                    .prng
                    .as_ref()
                    .unwrap()
                    .as_array()
                    .evaluated()
                    .unwrap()
                    .try_to_vec::<u32>()
                    .unwrap(),
                key
            );
            assert_eq!(paths::snapshot(), before_paths);
            copy
        };
        let third = driver.advance(&mut state).unwrap().unwrap().into_output();
        driver.take_completed_step(&mut state).unwrap();
        assert_eq!(history(copy.as_sampler()), ids);
        drop(third);
        drop(copy);
        drop(outputs);
        drop(state);
        drop(driver);
        drop(runtime);
        drop(artifact);
        settle(&pool, 0);
    }
}

#[test]
fn copied_sampler_and_its_descendant_retain_only_their_own_component_accounts() {
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (mut runtime, artifact) = runtime(&pool);
    let mut driver = TextGenerationDriver::new(&mut runtime);
    let mut state = driver
        .start_input(
            TextGenerationInput::TokenIds(tokens()),
            config(false, true),
            AllTokens,
        )
        .unwrap();
    let outputs = advance_twice(&mut driver, &mut state);
    let (copy, required, ids) = {
        let boundary = driver.quiescent(&mut state).unwrap();
        let source = &boundary.parts().1.sampling.sampler;
        let required = source.as_sampler().prepare_copy().unwrap().retained_bytes();
        let ids = history(source.as_sampler()).to_vec();
        let copy = pool
            .copy_sampler(
                source.borrow_funded().unwrap(),
                SamplerCopyLimits::new(u64::MAX),
            )
            .unwrap();
        (copy, required, ids)
    };
    drop(outputs);
    drop(state);
    drop(driver);
    drop(runtime);
    drop(artifact);
    settle(&pool, required);
    assert_eq!(history(copy.as_sampler()), ids);
    let descendant = pool
        .copy_sampler(copy.borrow_funded(), SamplerCopyLimits::new(required * 2))
        .unwrap();
    assert_eq!(pool.used_bytes().unwrap(), required * 2);
    assert_eq!(history(descendant.as_sampler()), ids);
    assert!(!std::ptr::eq(
        history(copy.as_sampler()).as_ptr(),
        history(descendant.as_sampler()).as_ptr()
    ));
    drop(copy);
    settle(&pool, required);
    assert_eq!(history(descendant.as_sampler()), ids);
    drop(descendant);
    settle(&pool, 0);
}

#[test]
fn native_unquoted_sources_and_foreign_accounts_cannot_authorize_sampler_copy() {
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (mut runtime, artifact) = runtime(&pool);
    let unquoted =
        MlxBackend::start_text_generation(runtime.backend(), config(false, false)).unwrap();
    let before = accounting(&pool);
    assert!(matches!(
        unquoted.sampling.sampler.borrow_funded(),
        Err(WorkingMemoryError::UnknownBound)
    ));
    assert_eq!(accounting(&pool), before);
    drop(unquoted);
    crate::backend::submission_recovery::wait_for_retirement(|| {
        reclaim();
        pool.unquoted_owner_count().unwrap() == 0
    });
    let mut driver = TextGenerationDriver::new(&mut runtime);
    let mut state = driver
        .start_input(
            TextGenerationInput::TokenIds(tokens()),
            config(false, true),
            AllTokens,
        )
        .unwrap();
    let outputs = advance_twice(&mut driver, &mut state);
    let other = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    {
        let boundary = driver.quiescent(&mut state).unwrap();
        let source = &boundary.parts().1.sampling.sampler;
        let before = (accounting(&pool), accounting(&other), paths::snapshot());
        let error = other
            .copy_sampler(
                source.borrow_funded().unwrap(),
                SamplerCopyLimits::new(u64::MAX),
            )
            .err()
            .unwrap();
        assert_eq!(
            cause::<WorkingMemoryError>(&error),
            Some(&WorkingMemoryError::IdentityMismatch)
        );
        assert_eq!(
            (accounting(&pool), accounting(&other), paths::snapshot()),
            before
        );
        assert_eq!(source.as_sampler().history_len(), 2);
    }
    drop(outputs);
    drop(state);
    drop(driver);
    drop(runtime);
    drop(artifact);
    settle(&pool, 0);

    // A rejected foreign sampler scope has covered no work. Its closed
    // host-only custody retires without quarantining the foreign account.
    let source_pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let foreign_pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (source, source_artifact) = self::runtime(&source_pool);
    let (foreign, foreign_artifact) = self::runtime(&foreign_pool);
    let source_preparation =
        MlxBackend::admit_text_preparation(&source, &evidence(), config(false, true), &AllTokens)
            .unwrap();
    let foreign_preparation =
        MlxBackend::admit_text_preparation(&foreign, &evidence(), config(false, true), &AllTokens)
            .unwrap();
    let before = (
        accounting(&source_pool),
        accounting(&foreign_pool),
        paths::snapshot(),
    );
    let stage = source_preparation
        .request
        .as_ref()
        .unwrap()
        .claim_sampling(config(false, true))
        .unwrap();
    let scope = foreign_preparation
        .quote
        .as_ref()
        .unwrap()
        .sampler_scope()
        .unwrap();
    let error = stage.construct_sampler(scope).err().unwrap();
    assert_eq!(error, WorkingMemoryError::IdentityMismatch);
    assert_eq!(
        (
            accounting(&source_pool),
            accounting(&foreign_pool),
            paths::snapshot()
        ),
        before
    );
    let fresh_scope = foreign_preparation
        .quote
        .as_ref()
        .unwrap()
        .sampler_scope()
        .unwrap();
    drop(fresh_scope);
    assert_eq!(
        (
            accounting(&source_pool),
            accounting(&foreign_pool),
            paths::snapshot()
        ),
        before
    );
    drop(source_preparation);
    drop(foreign_preparation);
    drop(source);
    drop(foreign);
    drop(source_artifact);
    drop(foreign_artifact);
    settle(&source_pool, 0);
    settle(&foreign_pool, 0);
}

#[test]
fn funded_sampler_component_does_not_enable_whole_native_sampling_snapshot() {
    use eredu_runtime::execution_control::TextSnapshotBackend;

    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (mut runtime, artifact) = runtime(&pool);
    let preparation =
        MlxBackend::admit_text_preparation(&runtime, &evidence(), config(false, true), &AllTokens)
            .unwrap();
    let prompt =
        MlxBackend::prepare_text_prompt_admitted(runtime.backend(), tokens(), &preparation)
            .unwrap();
    let prompt =
        MlxBackend::bind_text_prompt_preparation(runtime.backend(), prompt, &preparation).unwrap();
    let state = MlxBackend::start_text_generation_admitted(
        runtime.backend(),
        config(false, true),
        &preparation,
    )
    .unwrap();
    let copy = pool
        .copy_sampler(
            state.sampling.sampler.borrow_funded().unwrap(),
            SamplerCopyLimits::new(u64::MAX),
        )
        .unwrap();
    let before = (
        accounting(&pool),
        paths::snapshot(),
        runtime.session().payload.model.erased().state_snapshot(),
    );
    let error = MlxBackend::copy_sampling_state(&mut runtime, &state.sampling)
        .err()
        .unwrap();
    assert_eq!(
        cause::<WorkingMemoryError>(&error),
        Some(&WorkingMemoryError::UnknownBound)
    );
    assert_eq!(
        (
            accounting(&pool),
            paths::snapshot(),
            runtime.session().payload.model.erased().state_snapshot()
        ),
        before
    );
    drop(copy);
    drop(state);
    drop(prompt);
    drop(preparation);
    drop(runtime);
    drop(artifact);
    settle(&pool, 0);
}
