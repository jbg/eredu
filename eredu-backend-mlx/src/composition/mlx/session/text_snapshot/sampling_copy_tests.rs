use super::*;
use eredu_runtime::working_memory::{InferenceRetention, WorkingMemoryPool};

fn with_runtime(run: impl FnOnce(&mut ModelRuntime<MlxBackend<'_>>, &Stream)) {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let backend = MlxBackend::new(&stream, &stream).with_memory_pool(pool);
    let artifact = crate::composition::mlx::replicated_text::tests::tiny_artifact("llama", true);
    let model = eredu_core::load_model(&backend, artifact.path(), crate::MlxLoadRequest::default())
        .unwrap();
    let mut runtime = ModelRuntime::from_prepared(backend, model).unwrap();
    run(&mut runtime, &stream);
}

fn sampling(runtime: &ModelRuntime<MlxBackend<'_>>, spare: bool) -> MlxTextSamplingState {
    let mut sampler = GenerationSampler::new()
        .top_k(0)
        .top_p(1.0)
        .min_p(0.0)
        .penalties(1.1, 4, 0.03, 0.07);
    let history = [1, 7, 3, 2, 5];
    if spare {
        for token in history {
            sampler.accept_token(token);
        }
    } else {
        sampler.set_generated_tokens(history);
    }
    let owner = NativeMemoryOwner::acquire(runtime.backend().memory_pool()).unwrap();
    let random = RandomState::with_seed(12345).unwrap();
    random.as_array().evaluated().unwrap();
    owner.retain_array(random.as_array()).unwrap();
    MlxTextSamplingState {
        temperature: 0.7,
        prng: Some(random),
        sampler: MlxOrdinarySampler::Unquoted(MlxTextSampler::Standard(sampler)),
        next_prediction: 5,
        parameter_epoch: Some(7),
        inference_retention: InferenceRetention::new(),
        memory_retention: NativeMemoryRetention::from_owner(&owner),
        quote: None,
    }
}

fn history(sampling: &MlxTextSamplingState) -> &[u32] {
    match sampling.sampler.as_sampler() {
        MlxTextSampler::Standard(sampler) => sampler.generated_tokens(),
        MlxTextSampler::MirostatV2(sampler) => sampler.generated_tokens(),
    }
}

fn key(sampling: &MlxTextSamplingState) -> Vec<u32> {
    sampling
        .prng
        .as_ref()
        .unwrap()
        .as_array()
        .evaluated()
        .unwrap()
        .try_to_vec::<u32>()
        .unwrap()
}

#[test]
fn sampling_snapshot_cost_counts_spare_and_cleared_boxed_history() {
    with_runtime(|runtime, _| {
        let mut spare = sampling(runtime, true);
        let tight = sampling(runtime, false);
        assert_eq!(history(&spare), history(&tight));
        assert_eq!(spare.sampler.history_capacity(), 8);
        assert_eq!(tight.sampler.history_capacity(), 5);
        let cost = |state: &MlxTextSamplingState| {
            MlxBackend::estimate_sampling_state(runtime, state)
                .unwrap()
                .unwrap()
        };
        let full_cost = cost(&spare);
        let tight_cost = cost(&tight);
        assert_eq!(full_cost.retained_bytes - tight_cost.retained_bytes, 12);
        assert_eq!(full_cost.copy_bytes - tight_cost.copy_bytes, 12);
        let MlxOrdinarySampler::Unquoted(MlxTextSampler::Standard(sampler)) = &mut spare.sampler
        else {
            unreachable!()
        };
        sampler.clear_generated_tokens();
        assert!(history(&spare).is_empty());
        assert_eq!(spare.sampler.history_capacity(), 8);
        let cleared_cost = cost(&spare);
        assert_eq!(cleared_cost.retained_bytes, full_cost.retained_bytes);
        assert_eq!(cleared_cost.copy_bytes, full_cost.copy_bytes);

        // The same backend worker preserves the cleared destination capacity.
        let copy = MlxBackend::copy_sampling_state(runtime, &spare).unwrap();
        assert!(history(&copy).is_empty());
        assert_eq!(copy.sampler.history_capacity(), 8);
        assert_eq!(key(&copy), key(&spare));
    });
}

#[test]
fn copied_sampler_preserves_rng_continuation_without_drawing_or_sharing_history() {
    with_runtime(|runtime, stream| {
        let mut source = sampling(runtime, true);
        let before = key(&source);
        let mut copy = MlxBackend::copy_sampling_state(runtime, &source).unwrap();
        assert_eq!(key(&source), before);
        assert_eq!(key(&copy), before);
        assert_eq!(history(&copy), &[1, 7, 3, 2, 5]);
        assert_eq!(copy.sampler.history_capacity(), 8);
        assert_eq!(copy.next_prediction, 5);
        assert_eq!(copy.parameter_epoch, Some(7));
        assert_eq!(copy.temperature, source.temperature);
        assert!(!std::ptr::eq(
            history(&source).as_ptr(),
            history(&copy).as_ptr()
        ));
        let backing = |state: &MlxTextSamplingState| {
            state
                .prng
                .as_ref()
                .unwrap()
                .as_array()
                .allocation_info()
                .unwrap()
                .unwrap()
                .identity()
        };
        assert_ne!(backing(&source), backing(&copy));

        let logits = MlxTensor::from_array(Array::from_slice(
            &[0.3_f32, 1.7, -0.2, 0.8, 2.1, 1.3, -0.7, 0.6],
            &[1, 8],
        ));
        let sample = |state: &mut MlxTextSamplingState| {
            let token = state
                .sampler
                .prepare_sample()
                .unwrap()
                .sample(&logits, state.temperature, state.prng.as_mut(), stream)
                .unwrap();
            MlxSamplingBackend::token_id(&token, stream).unwrap()
        };
        for _ in 0..6 {
            assert_eq!(sample(&mut source), sample(&mut copy));
            assert_eq!(history(&source), history(&copy));
            assert_eq!(key(&source), key(&copy));
        }
        assert_ne!(key(&source), before);
        let MlxOrdinarySampler::Unquoted(MlxTextSampler::Standard(sampler)) = &mut source.sampler
        else {
            unreachable!()
        };
        sampler.clear_generated_tokens();
        assert_eq!(copy.sampler.history_len(), 11);
        let saved_key = key(&copy);
        drop(source);
        assert_eq!(key(&copy), saved_key);
        assert!(sample(&mut copy) < 8);
        assert_eq!(copy.sampler.history_len(), 12);
    });
}

#[test]
fn paired_saved_components_restore_decoder_sampler_and_pending_input_together() {
    use eredu_core::execution_control::NativeTextStateBackend;
    use eredu_runtime::execution_control::SamplingCopyPolicy;
    with_runtime(|runtime, _| {
        let prompt = MlxBackend::prepare_text_prompt(runtime.backend(), vec![1, 2, 3]).unwrap();
        drop(runtime.prefill(prompt).unwrap().wait().unwrap());
        let source = sampling(runtime, true);
        let source_key = key(&source);
        let pending = MlxBackend::prepare_text_prompt(runtime.backend(), vec![4, 5]).unwrap();
        let positions = runtime.session().test_state_presence();
        let saved = MlxBackend::capture_saved_components(
            runtime,
            &source,
            Some(PendingTextInput::Prefill(&pending)),
            SamplingCopyPolicy::Unquoted,
        )
        .unwrap();
        assert_eq!(runtime.session().test_state_presence(), positions);
        assert_eq!(key(&source), source_key);
        let duplicate =
            MlxBackend::copy_saved_components(runtime, &saved, SamplingCopyPolicy::Unquoted)
                .unwrap();
        let estimate = MlxBackend::estimate_saved_components(runtime, &saved)
            .unwrap()
            .unwrap();
        assert!(estimate.retained_bytes > 0 && estimate.copy_bytes > 0);
        drop((source, pending, saved));
        drop(
            runtime
                .decode(Array::from_slice(&[6_u32], &[1, 1]))
                .unwrap()
                .wait()
                .unwrap(),
        );
        let advanced = runtime.session().test_state_presence();
        assert_ne!(advanced, positions);

        let (mut slot, resumed, pending) =
            MlxBackend::prepare_saved_components_resume(runtime, &duplicate).unwrap();
        // Staging is independent and must not install any part of the saved pair.
        assert_eq!(runtime.session().test_state_presence(), advanced);
        assert_eq!(key(&resumed), source_key);
        assert_eq!(history(&resumed), &[1, 7, 3, 2, 5]);
        assert_eq!(resumed.next_prediction, 5);
        let Some(PendingTextInput::Prefill(pending)) = pending else {
            panic!("saved pending prompt")
        };
        assert_eq!(
            with_text_prompt_array(&pending, |a| a
                .evaluated()
                .unwrap()
                .try_to_vec::<u32>()
                .unwrap())
            .unwrap(),
            [4, 5]
        );
        MlxBackend::exchange_native_text_state(runtime, &mut slot).unwrap();
        assert_eq!(runtime.session().test_state_presence(), positions);
        let output = runtime
            .decode(Array::from_slice(&[4_u32], &[1, 1]))
            .unwrap()
            .wait()
            .unwrap();
        let expected = output
            .logits()
            .unwrap()
            .as_array()
            .evaluated()
            .unwrap()
            .try_to_vec::<f32>()
            .unwrap();
        assert!(expected.iter().any(|v| v.abs() > 1e-6));
        drop(output);
        let (mut second_slot, second_sampler, second_pending) =
            MlxBackend::prepare_saved_components_resume(runtime, &duplicate).unwrap();
        MlxBackend::exchange_native_text_state(runtime, &mut second_slot).unwrap();
        assert_eq!(key(&second_sampler), source_key);
        assert!(second_pending.is_some());
        let output = runtime
            .decode(Array::from_slice(&[4_u32], &[1, 1]))
            .unwrap()
            .wait()
            .unwrap();
        let actual = output
            .logits()
            .unwrap()
            .as_array()
            .evaluated()
            .unwrap()
            .try_to_vec::<f32>()
            .unwrap();
        assert_eq!(actual, expected);
    });
}

#[test]
fn bounded_pair_rejects_before_component_copy_and_preserves_the_source() {
    use eredu_runtime::{
        execution_control::SamplingCopyPolicy, working_memory::WorkspaceCopyLimits,
    };
    with_runtime(|runtime, _| {
        let source = sampling(runtime, true);
        let saved = MlxBackend::capture_saved_components(
            runtime,
            &source,
            None,
            SamplingCopyPolicy::Unquoted,
        )
        .unwrap();
        let pool = runtime.backend().memory_pool();
        let before = (
            pool.used_bytes().unwrap(),
            pool.unquoted_owner_count().unwrap(),
        );
        let before_key = key(&source);
        let before_state = runtime.session().test_state_presence();
        let policy = SamplingCopyPolicy::Bounded(WorkspaceCopyLimits {
            capacity_bytes: u64::MAX,
            application_memory_budget_bytes: None,
            safety_reserve_bytes: 0,
        });
        let capture = MlxBackend::capture_saved_components(runtime, &source, None, policy)
            .err()
            .unwrap();
        let copy = MlxBackend::copy_saved_components(runtime, &saved, policy)
            .err()
            .unwrap();
        for error in [capture, copy] {
            let Error::Other(error) = error else {
                panic!("typed memory rejection")
            };
            assert!(matches!(
                error.downcast_ref::<eredu_runtime::working_memory::WorkingMemoryError>(),
                Some(eredu_runtime::working_memory::WorkingMemoryError::UnknownBound)
            ));
        }
        let pool = runtime.backend().memory_pool();
        assert_eq!(
            (
                pool.used_bytes().unwrap(),
                pool.unquoted_owner_count().unwrap()
            ),
            before
        );
        assert_eq!(key(&source), before_key);
        assert_eq!(runtime.session().test_state_presence(), before_state);
        assert_eq!(
            MlxBackend::saved_sampling_prediction(MlxBackend::saved_sampling(&saved)),
            5
        );
    });
}

#[test]
fn paired_resume_rejects_a_different_loaded_executable_without_installing_state() {
    use eredu_runtime::execution_control::SamplingCopyPolicy;
    with_runtime(|first, _| {
        let source = sampling(first, true);
        let saved = MlxBackend::capture_saved_components(
            first,
            &source,
            None,
            SamplingCopyPolicy::Unquoted,
        )
        .unwrap();
        with_runtime(|other, _| {
            let before = other.session().test_state_presence();
            assert!(MlxBackend::validate_saved_components(other, &saved).is_err());
            assert!(MlxBackend::prepare_saved_components_resume(other, &saved).is_err());
            assert_eq!(other.session().test_state_presence(), before);
        });
        assert!(MlxBackend::validate_saved_components(first, &saved).is_ok());
    });
}
