use super::*;
use eredu_core::{
    Admission, EstimationCompleteness, ExecutionWorkspaceEstimate, InferenceGeometry,
    InputTokenCount, LayerSchedule, OutputDemand, StateMemoryLayout, WorkspaceBound,
};
use eredu_runtime::{
    execution_control::{SamplingOverride, TextSamplingControlBackend},
    working_memory::{InferenceExecutionIdentity, MemoryLedger, WorkingMemoryError},
};

fn stream() -> Stream {
    Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0))
}

fn backend(stream: &Stream, pool: &MemoryLedger) -> MlxBackend<'static> {
    MlxBackend::new(stream, stream).with_memory_ledger(pool.clone())
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
    .with_execution_workspace(crate::memory_fixture::workspace(
        ExecutionWorkspaceEstimate {
            physical_domains: None,
            geometry: g,
            activations: bound(bytes),
            attention: bound(0),
            vocabulary: bound(0),
            state_update: bound(0),
            materialization: bound(0),
            retained: bound(0),
        },
    ))
    .unwrap();
    crate::memory_fixture::admission(Admission {
        additional_headroom: Default::default(),
        memory_limits: Default::default(),
        requested_positions: g.input_positions + g.max_output_tokens,
        state,
        incremental_required_bytes: Some(bytes),
    })
}

fn runtime(
    stream: &Stream,
    preparation_pool: &MemoryLedger,
) -> (ModelRuntime<MlxBackend<'static>>, tempfile::TempDir) {
    // The already loaded model belongs to a separate domain. This deliberately
    // isolates admission of the independent operation under test.
    let model_pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
    let loader = backend(stream, &model_pool);
    let root = crate::composition::mlx::replicated_text::tests::tiny_artifact("llama", true);
    let model =
        eredu_core::load_model(&loader, root.path(), crate::MlxLoadRequest::default()).unwrap();
    (
        ModelRuntime::from_prepared(backend(stream, preparation_pool), model).unwrap(),
        root,
    )
}

fn settle(pool: &MemoryLedger, owners: usize) {
    crate::backend::submission_recovery::wait_for_retirement(|| {
        safemlx::memory::clear_cache();
        crate::backend::nn::shared::MlxNeuralBackend::reclaim_retired_resources();
        safemlx::reclaim_allocation_owners();
        pool.unquoted_owner_count().unwrap() == owners
    });
}

fn assert_unquoted(pool: &MemoryLedger, owners: usize) {
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
    let pool = crate::memory_fixture::ledger(1 << 24, 0).unwrap();
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
        let pool = crate::memory_fixture::ledger(1 << 24, 0).unwrap();
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
    let pool = crate::memory_fixture::ledger(1 << 24, 0).unwrap();
    let backend = backend(&stream, &pool);
    let reserved = pool
        .reserve(&InferenceExecutionIdentity::default(), &admission(0))
        .unwrap();
    let before = pool.snapshot().unwrap();
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
    assert_eq!(pool.snapshot().unwrap(), before);
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
fn reservation_without_original_source_refuses_prompt_and_sampler_before_allocation() {
    let stream = stream();
    let pool = crate::memory_fixture::ledger(1 << 25, 0).unwrap();
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
    let before = pool.snapshot().unwrap();
    let prompt = MlxBackend::prepare_text_prompt_admitted(
        runtime.backend(),
        vec![1, 2, 3, 4, 5],
        &preparation,
    )
    .err()
    .expect("a reservation is not an original input producer");
    assert_eq!(
        memory_error(&prompt),
        Some(&WorkingMemoryError::UnknownBound)
    );
    let factory = MlxBackend::fail_next_sampling_for_test(Error::ArchitectureModel(
        "sampling factory sentinel".into(),
    ));
    let sampler =
        MlxBackend::start_text_generation_admitted(runtime.backend(), config(0.7), &preparation)
            .err()
            .expect("a reservation is not an original sampler producer");
    assert_eq!(
        memory_error(&sampler),
        Some(&WorkingMemoryError::UnknownBound)
    );
    assert!(TEST_SAMPLING_FAILURE.with(|slot| slot.borrow().is_some()));
    assert_eq!(pool.snapshot().unwrap(), before);
    drop(factory);
    drop((preparation, request, runtime));
    settle(&pool, 0);
    assert_eq!(pool.fixture_host_charge().unwrap(), 0);
}

#[test]
fn reserved_ledger_rejects_unadmitted_reseed_without_mutating_sources() {
    let stream = stream();
    let pool = crate::memory_fixture::ledger(1 << 24, 0).unwrap();
    let source_pool = crate::memory_fixture::ledger(1 << 24, 0).unwrap();
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
    let before = pool.snapshot().unwrap();
    let action = SamplingOverride {
        temperature: Some(0.3),
        reseed: Some(77),
    };
    let eredu_core::SamplingOverrideError::Backend(reseed) =
        MlxBackend::apply_sampling_override(&mut runtime, &mut sampler, None, action).unwrap_err()
    else {
        panic!("expected native reservation refusal")
    };
    for error in [reseed] {
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
    assert_eq!(pool.snapshot().unwrap(), before);
    drop(reserved);
    settle(&pool, 0);
}

#[cfg(test)]
#[allow(unused_imports)]
use crate::memory_fixture::LedgerFixture;
