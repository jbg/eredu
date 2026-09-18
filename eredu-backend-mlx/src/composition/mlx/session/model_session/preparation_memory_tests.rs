use super::*;
use eredu_core::{
    execution_control::NativeTextStateBackend, Admission, EstimationCompleteness,
    ExecutionWorkspaceEstimate, InferenceGeometry, InputTokenCount, LayerSchedule, OutputDemand,
    PendingTextInput, StateMemoryLayout, WorkspaceBound,
};
use eredu_runtime::{
    execution_control::{
        SamplingOverride, TextSamplingControlBackend,
        TextSnapshotBackend,
    },
    working_memory::{InferenceExecutionIdentity, WorkingMemoryError, WorkingMemoryPool},
};

fn stream() -> Stream {
    Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0))
}

fn backend(stream: &Stream, pool: &WorkingMemoryPool) -> MlxBackend<'static> {
    MlxBackend::new(stream, stream).with_memory_pool(pool.clone())
}

fn config(temperature: f32) -> TextGenerationConfig {
    TextGenerationConfig::new(
        eredu_core::resolve_generation_config(
            None,
            eredu_core::GenerationConfigOverrides {
                temperature: Some(temperature),
                max_new_tokens: Some(2),
                ..Default::default()
            },
        )
        .unwrap(),
    )
    .with_seed(19)
}

fn geometry() -> InferenceGeometry {
    InferenceGeometry {
        batch_size: 1,
        cached_positions: 0,
        input_positions: 5,
        max_output_tokens: 2,
        prefill_chunk_positions: 2,
        output: OutputDemand::LastPosition,
    }
}

// Synthetic ownership evidence only; these tests do not certify native peaks.
fn admission(bytes: u64) -> Admission {
    let g = geometry();
    let layout = StateMemoryLayout::new(
        LayerSchedule::new(1, vec![eredu_core::cache::LayerCachePolicy::NoState]).unwrap(),
        vec![0],
        1,
        1,
        EstimationCompleteness::Complete,
    )
    .unwrap();
    let bound = |bytes| WorkspaceBound::bounded(bytes, "synthetic preparation ownership fixture");
    let state = eredu_core::estimate_runtime_state(
        &layout,
        InputTokenCount::text(g.input_positions),
        g.max_output_tokens,
        g.batch_size,
        std::num::NonZeroU8::new(4).unwrap(),
    )
    .unwrap()
    .with_execution_workspace(ExecutionWorkspaceEstimate {
        geometry: g,
        activations: bound(bytes),
        attention: bound(0),
        vocabulary: bound(0),
        state_update: bound(0),
        materialization: bound(0),
        retained: bound(0),
    })
    .unwrap();
    Admission {
        requested_positions: g.input_positions + g.max_output_tokens,
        state,
        incremental_required_bytes: bytes,
        available_memory_bytes: None,
    }
}

fn runtime(
    stream: &Stream,
    preparation_pool: &WorkingMemoryPool,
) -> (ModelRuntime<MlxBackend<'static>>, tempfile::TempDir) {
    // The already loaded model belongs to a separate domain. This deliberately
    // isolates admission of the independent operation under test.
    let model_pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let loader = backend(stream, &model_pool);
    let root = crate::composition::mlx::replicated_text::tests::tiny_artifact("llama", true);
    let model =
        eredu_core::load_model(&loader, root.path(), crate::MlxLoadRequest::default()).unwrap();
    (
        ModelRuntime::from_prepared(backend(stream, preparation_pool), model).unwrap(),
        root,
    )
}

fn settle(pool: &WorkingMemoryPool, owners: usize) {
    crate::backend::submission_recovery::wait_for_retirement(|| {
        crate::backend::nn::shared::MlxNeuralBackend::reclaim_retired_resources();
        safemlx::reclaim_allocation_owners();
        pool.unquoted_owner_count().unwrap() == owners
    });
}

fn assert_unquoted(pool: &WorkingMemoryPool, owners: usize) {
    settle(pool, owners);
    assert!(matches!(
        pool.reserve(&InferenceExecutionIdentity::default(), &admission(0)),
        Err(WorkingMemoryError::UnknownBound)
    ));
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

fn prompt_array(prompt: &MlxModelInput) -> Array {
    prompt.with_borrowed(|input| {
        let input::InputPayload::TokenIds(tokens) = input.parts[0].payload() else {
            panic!("fixture token prompt");
        };
        tokens.clone()
    })
}

#[test]
fn independent_prompt_clones_and_escaped_native_views_retain_memory_authority() {
    let stream = stream();
    let pool = WorkingMemoryPool::new(0, 0).unwrap();
    let backend = backend(&stream, &pool);
    let prompt = MlxBackend::prepare_text_prompt(&backend, vec![1, 2, 3, 4, 5]).unwrap();
    let clone = prompt.clone();
    let escaped = prompt_array(&prompt);
    drop((prompt, backend));
    assert_unquoted(&pool, 1);
    drop(clone);
    assert_unquoted(&pool, 1);
    // The raw lazy view is now the only surviving native owner.
    escaped.evaluated().unwrap();
    let allocation = escaped.allocation_info().unwrap().unwrap();
    let view = escaped.try_index_device((.., 1..), &stream).unwrap();
    view.evaluated().unwrap();
    assert_eq!(view.allocation_info().unwrap(), Some(allocation));
    drop(escaped);
    assert_unquoted(&pool, 1);
    assert_eq!(view.evaluated().unwrap().as_slice::<u32>(), &[2, 3, 4, 5]);
    assert_eq!(view.allocation_info().unwrap(), Some(allocation));
    drop(view);
    settle(&pool, 0);
    assert!(pool
        .reserve(&InferenceExecutionIdentity::default(), &admission(0))
        .is_ok());
}

#[test]
fn independent_greedy_and_stochastic_samplers_outlive_their_backend() {
    for temperature in [0.0, 0.7] {
        let stream = stream();
        let pool = WorkingMemoryPool::new(0, 0).unwrap();
        let backend = backend(&stream, &pool);
        let sampler = MlxBackend::start_text_generation(&backend, config(temperature)).unwrap();
        assert_eq!(sampler.sampling.prng.is_some(), temperature > 0.0);
        drop(backend);
        assert_unquoted(&pool, 1);
        drop(sampler);
        settle(&pool, 0);
        assert!(pool
            .reserve(&InferenceExecutionIdentity::default(), &admission(0))
            .is_ok());
    }
}

#[test]
fn live_zero_byte_reservation_rejects_independent_preparation_before_sampling_factory() {
    let stream = stream();
    let pool = WorkingMemoryPool::new(0, 0).unwrap();
    let backend = backend(&stream, &pool);
    let reserved = pool
        .reserve(&InferenceExecutionIdentity::default(), &admission(0))
        .unwrap();
    let failure = MlxBackend::fail_next_sampling_for_test(Error::ArchitectureModel(
        "sampling factory sentinel".into(),
    ));
    let prompt_error = MlxBackend::prepare_text_prompt(&backend, vec![1, 2, 3])
        .err()
        .unwrap();
    assert_eq!(
        memory_error(&prompt_error),
        Some(&WorkingMemoryError::ReservedWorkActive)
    );
    let sampler_error = MlxBackend::start_text_generation(&backend, config(0.7))
        .err()
        .unwrap();
    assert_eq!(
        memory_error(&sampler_error),
        Some(&WorkingMemoryError::ReservedWorkActive)
    );
    assert!(TEST_SAMPLING_FAILURE.with(|slot| slot.borrow().is_some()));
    assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
    assert_eq!(pool.used_bytes().unwrap(), 0);
    assert_eq!(pool.peak_bytes().unwrap(), 0);
    drop(reserved);
    let entered = MlxBackend::start_text_generation(&backend, config(0.7))
        .err()
        .unwrap();
    assert!(entered.to_string().contains("sampling factory sentinel"));
    assert!(TEST_SAMPLING_FAILURE.with(|slot| slot.borrow().is_none()));
    drop(failure);
    settle(&pool, 0);
}

#[test]
fn admitted_preparation_uses_its_same_domain_reservation_without_unquoted_acquisition() {
    let stream = stream();
    let pool = WorkingMemoryPool::new(1 << 25, 0).unwrap();
    let (runtime, _root) = runtime(&stream, &pool);
    let identity = runtime
        .session()
        .payload
        .model
        .erased()
        .inference_execution_identity();
    let request: eredu_runtime::working_memory::InferenceRequest =
        pool.reserve(identity, &admission(1 << 24)).unwrap().into();
    let preparation = MlxTextPreparation {
        request: Some(
            request
                .prepare_text(identity, geometry(), config(0.7))
                .unwrap(),
        ),
        chunk: None,
        quote: None,
    };
    let prompt = MlxBackend::prepare_text_prompt_admitted(
        runtime.backend(),
        vec![1, 2, 3, 4, 5],
        &preparation,
    )
    .unwrap();
    let prompt =
        MlxBackend::bind_text_prompt_preparation(runtime.backend(), prompt, &preparation).unwrap();
    let sampler =
        MlxBackend::start_text_generation_admitted(runtime.backend(), config(0.7), &preparation)
            .unwrap();
    assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
    assert_eq!(pool.used_bytes().unwrap(), 1 << 24);
    assert_eq!(
        prompt_array(&prompt).evaluated().unwrap().as_slice::<u32>(),
        &[1, 2, 3, 4, 5]
    );
    drop((request, preparation, runtime, prompt));
    assert_eq!(pool.used_bytes().unwrap(), 1 << 24);
    drop(sampler);
    crate::backend::submission_recovery::wait_for_retirement(|| {
        safemlx::reclaim_allocation_owners();
        pool.used_bytes().unwrap() == 0
    });
}

#[test]
fn admitted_prompt_backing_retains_reservation_when_it_escapes_before_binding() {
    let stream = stream();
    let pool = WorkingMemoryPool::new(1 << 25, 0).unwrap();
    let (runtime, _root) = runtime(&stream, &pool);
    let identity = runtime
        .session()
        .payload
        .model
        .erased()
        .inference_execution_identity();
    let request: eredu_runtime::working_memory::InferenceRequest =
        pool.reserve(identity, &admission(1 << 24)).unwrap().into();
    let preparation = MlxTextPreparation {
        request: Some(
            request
                .prepare_text(identity, geometry(), config(0.7))
                .unwrap(),
        ),
        chunk: None,
        quote: None,
    };
    let prompt = MlxBackend::prepare_text_prompt_admitted(
        runtime.backend(),
        vec![1, 2, 3, 4, 5],
        &preparation,
    )
    .unwrap();
    prompt.with_borrowed(|input| {
        assert_eq!(
            input
                .inference_request()
                .unwrap()
                .validate_same_request(&request),
            Ok(())
        );
    });
    let escaped = prompt_array(&prompt);
    // No bind call occurs: the native backing already owns its reservation.
    drop((prompt, preparation, request, runtime));
    settle(&pool, 0);
    assert_eq!(pool.used_bytes().unwrap(), 1 << 24);
    assert!(matches!(
        pool.acquire_unquoted(),
        Err(WorkingMemoryError::ReservedWorkActive)
    ));
    escaped.evaluated().unwrap();
    let allocation = escaped.allocation_info().unwrap().unwrap();
    let alias = escaped.try_index_device((.., 1..), &stream).unwrap();
    alias.evaluated().unwrap();
    assert_eq!(alias.allocation_info().unwrap(), Some(allocation));
    drop(escaped);
    safemlx::reclaim_allocation_owners();
    assert_eq!(pool.used_bytes().unwrap(), 1 << 24);
    assert_eq!(alias.evaluated().unwrap().as_slice::<u32>(), &[2, 3, 4, 5]);
    assert_eq!(alias.allocation_info().unwrap(), Some(allocation));
    drop(alias);
    crate::backend::submission_recovery::wait_for_retirement(|| {
        safemlx::reclaim_allocation_owners();
        pool.used_bytes().unwrap() == 0
    });
    let unquoted = pool.acquire_unquoted().unwrap();
    drop(unquoted);
}

#[test]
fn explicitly_unbudgeted_preparation_still_acquires_independent_memory_authority() {
    for blocked in [false, true] {
        let stream = stream();
        let pool = WorkingMemoryPool::new(0, 0).unwrap();
        let (runtime, _root) = runtime(&stream, &pool);
        let identity = runtime
            .session()
            .payload
            .model
            .erased()
            .inference_execution_identity();
        let request = eredu_runtime::working_memory::InferenceRequest::without_memory_budget(
            identity,
            geometry(),
        )
        .unwrap();
        assert!(request.memory_reservation().is_none());
        let preparation = MlxTextPreparation {
            request: Some(
                request
                    .prepare_text(identity, geometry(), config(0.7))
                    .unwrap(),
            ),
            chunk: None,
            quote: None,
        };
        let reserved = blocked.then(|| pool.reserve(identity, &admission(0)).unwrap());
        if blocked {
            let failure = MlxBackend::fail_next_sampling_for_test(Error::ArchitectureModel(
                "unbudgeted sampling factory sentinel".into(),
            ));
            let prompt_error = MlxBackend::prepare_text_prompt_admitted(
                runtime.backend(),
                vec![1, 2, 3, 4, 5],
                &preparation,
            )
            .err()
            .unwrap();
            let sampler_error = MlxBackend::start_text_generation_admitted(
                runtime.backend(),
                config(0.7),
                &preparation,
            )
            .err()
            .unwrap();
            for error in [prompt_error, sampler_error] {
                assert_eq!(
                    memory_error(&error),
                    Some(&WorkingMemoryError::ReservedWorkActive)
                );
            }
            assert!(TEST_SAMPLING_FAILURE.with(|slot| slot.borrow().is_some()));
            assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
            assert_eq!(pool.used_bytes().unwrap(), 0);
            assert_eq!(pool.peak_bytes().unwrap(), 0);
            drop(failure);
            drop((reserved, preparation, request, runtime));
        } else {
            let prompt = MlxBackend::prepare_text_prompt_admitted(
                runtime.backend(),
                vec![1, 2, 3, 4, 5],
                &preparation,
            )
            .unwrap();
            let prompt =
                MlxBackend::bind_text_prompt_preparation(runtime.backend(), prompt, &preparation)
                    .unwrap();
            let sampler = MlxBackend::start_text_generation_admitted(
                runtime.backend(),
                config(0.7),
                &preparation,
            )
            .unwrap();
            assert_unquoted(&pool, 2);
            assert_eq!(pool.used_bytes().unwrap(), 0);
            drop((preparation, request, runtime));
            assert_unquoted(&pool, 2);
            assert_eq!(
                prompt_array(&prompt).evaluated().unwrap().as_slice::<u32>(),
                &[1, 2, 3, 4, 5]
            );
            drop(prompt);
            assert_unquoted(&pool, 1);
            drop(sampler);
        }
        settle(&pool, 0);
    }
}

#[test]
fn independent_prompt_and_sampler_copies_outlive_their_sources_and_runtime() {
    let stream = stream();
    let pool = WorkingMemoryPool::new(0, 0).unwrap();
    let source_pool = WorkingMemoryPool::new(0, 0).unwrap();
    let source_backend = backend(&stream, &source_pool);
    let (mut runtime, _root) = runtime(&stream, &pool);
    let prompt = MlxBackend::prepare_text_prompt(&source_backend, vec![1, 2, 3, 4, 5]).unwrap();
    let sampler = MlxBackend::start_text_generation(&source_backend, config(0.7)).unwrap();
    let sampling_copy = MlxBackend::copy_sampling_state(&mut runtime, &sampler.sampling).unwrap();
    let Some(PendingTextInput::Prefill(prompt_copy)) =
        MlxBackend::copy_pending_input(&mut runtime, Some(PendingTextInput::Prefill(&prompt)))
            .unwrap()
    else {
        panic!("copied prompt");
    };
    assert_unquoted(&pool, 2);
    let escaped = prompt_array(&prompt_copy);
    drop((prompt, sampler, source_backend, runtime, prompt_copy));
    assert_unquoted(&pool, 2);
    drop(sampling_copy);
    assert_unquoted(&pool, 1);
    assert_eq!(
        escaped.evaluated().unwrap().as_slice::<u32>(),
        &[1, 2, 3, 4, 5]
    );
    drop(escaped);
    settle(&pool, 0);
}

#[test]
fn independent_native_snapshots_and_copies_keep_distinct_memory_authorities() {
    let stream = stream();
    let pool = WorkingMemoryPool::new(0, 0).unwrap();
    let (mut runtime, _root) = runtime(&stream, &pool);
    let submission = runtime
        .decode(Array::from_slice(&[1_u32], &[1, 1]))
        .unwrap();
    submission.completion.wait().unwrap();
    drop(submission);
    let snapshot = MlxBackend::capture_native_text_state(&mut runtime).unwrap();
    let copy = MlxBackend::copy_native_text_state(&mut runtime, &snapshot).unwrap();
    // Installed ordinary state retains its operation owner independently of
    // the two copied snapshots.
    assert_unquoted(&pool, 3);
    drop((snapshot, runtime));
    assert_unquoted(&pool, 1);
    drop(copy);
    settle(&pool, 0);
}

#[test]
fn native_state_exchange_moves_memory_authority_with_installed_and_displaced_state() {
    for swap_back in [false, true] {
        for runtime_first in [false, true] {
            let stream = stream();
            let model_pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
            let snapshot_pool = WorkingMemoryPool::new(0, 0).unwrap();
            let loader = backend(&stream, &model_pool);
            let root =
                crate::composition::mlx::replicated_text::tests::tiny_artifact("llama", true);
            let prepared =
                eredu_core::load_model(&loader, root.path(), crate::MlxLoadRequest::default())
                    .unwrap();
            let mut runtime =
                ModelRuntime::from_prepared(backend(&stream, &snapshot_pool), prepared).unwrap();
            drop(loader);
            let submission = runtime
                .decode(Array::from_slice(&[1_u32], &[1, 1]))
                .unwrap();
            submission.completion.wait().unwrap();
            drop(submission);
            let mut slot = MlxBackend::capture_native_text_state(&mut runtime).unwrap();
            let submission = runtime
                .decode(Array::from_slice(&[2_u32], &[1, 1]))
                .unwrap();
            submission.completion.wait().unwrap();
            drop(submission);
            MlxBackend::exchange_native_text_state(&mut runtime, &mut slot).unwrap();
            assert_unquoted(&model_pool, 1);
            assert_unquoted(&snapshot_pool, 2);
            if swap_back {
                MlxBackend::exchange_native_text_state(&mut runtime, &mut slot).unwrap();
                assert_unquoted(&model_pool, 1);
                assert_unquoted(&snapshot_pool, 2);
            }
            if runtime_first {
                drop(runtime);
                // The displaced state always retains the model's authority.
                assert_unquoted(&model_pool, 1);
                if swap_back {
                    // A second exchange moves independently copied state back
                    // into the detached slot, including fresh copy authority
                    // and the ordinary operation authority inherited at exchange.
                    assert_unquoted(&snapshot_pool, 2);
                } else {
                    // Displaced ordinary state keeps its context-domain owner.
                    assert_unquoted(&snapshot_pool, 1);
                }
                drop(slot);
            } else {
                drop(slot);
                assert_unquoted(&model_pool, 1);
                if swap_back {
                    assert_unquoted(&snapshot_pool, 1);
                } else {
                    assert_unquoted(&snapshot_pool, 2);
                }
                drop(runtime);
            }
            settle(&model_pool, 0);
            settle(&snapshot_pool, 0);
        }
    }
}

#[test]
fn reserved_domain_rejects_copy_capture_and_reseed_without_mutating_sources() {
    let stream = stream();
    let pool = WorkingMemoryPool::new(0, 0).unwrap();
    let source_pool = WorkingMemoryPool::new(0, 0).unwrap();
    let source_backend = backend(&stream, &source_pool);
    let (mut runtime, _root) = runtime(&stream, &pool);
    // Populate the independent model domain without first admitting an ordinary
    // operation into the destination domain whose copy rejection is under test.
    runtime
        .session_mut()
        .neutral_prediction_target_mut()
        .unwrap()
        .decode(&Array::from_slice(&[1_u32], &[1, 1]), &stream)
        .unwrap()
        .evaluated()
        .unwrap();
    let prompt = MlxBackend::prepare_text_prompt(&source_backend, vec![1, 2, 3, 4, 5]).unwrap();
    let mut sampler = MlxBackend::start_text_generation(&source_backend, config(0.7)).unwrap();
    let before_rng = sampler
        .sampling
        .prng
        .as_ref()
        .unwrap()
        .as_array()
        .evaluated()
        .unwrap()
        .as_slice::<u32>()
        .to_vec();
    let before_model = runtime
        .session_mut()
        .neutral_prediction_target_mut()
        .unwrap()
        .fixed_numeric_state_snapshot()
        .unwrap();
    let reserved = pool
        .reserve(&InferenceExecutionIdentity::default(), &admission(0))
        .unwrap();
    let capture = MlxBackend::capture_native_text_state(&mut runtime)
        .err()
        .unwrap();
    let sampling_copy = MlxBackend::copy_sampling_state(&mut runtime, &sampler.sampling)
        .err()
        .unwrap();
    let input_copy =
        MlxBackend::copy_pending_input(&mut runtime, Some(PendingTextInput::Prefill(&prompt)))
            .err()
            .unwrap();
    let action = SamplingOverride {
            temperature: Some(0.3),
            reseed: Some(77),
        };
    let eredu_core::SamplingOverrideError::Backend(reseed) =
        MlxBackend::apply_sampling_override(&mut runtime, &mut sampler, None, action).unwrap_err()
        else { panic!("expected native reservation refusal") };
    for error in [capture, sampling_copy, input_copy, reseed] {
        assert_eq!(
            memory_error(&error),
            Some(&WorkingMemoryError::ReservedWorkActive)
        );
    }
    assert_eq!(sampler.sampling.temperature, 0.7);
    assert_eq!(
        sampler
            .sampling
            .prng
            .as_ref()
            .unwrap()
            .as_array()
            .evaluated()
            .unwrap()
            .as_slice::<u32>(),
        before_rng
    );
    assert_eq!(
        runtime
            .session_mut()
            .neutral_prediction_target_mut()
            .unwrap()
            .fixed_numeric_state_snapshot()
            .unwrap(),
        before_model
    );
    assert_eq!(
        prompt_array(&prompt).evaluated().unwrap().as_slice::<u32>(),
        &[1, 2, 3, 4, 5]
    );
    assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
    assert_eq!(pool.used_bytes().unwrap(), 0);
    assert_eq!(pool.peak_bytes().unwrap(), 0);
    drop(reserved);
    settle(&pool, 0);
}
