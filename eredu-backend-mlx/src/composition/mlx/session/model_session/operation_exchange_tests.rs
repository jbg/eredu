use super::*;
use eredu_core::execution_control::NativeTextStateBackend;
use eredu_runtime::working_memory::WorkingMemoryPool;

fn settle(pool: &WorkingMemoryPool, owners: usize) {
    crate::backend::submission_recovery::wait_for_retirement(|| {
        crate::backend::nn::shared::MlxNeuralBackend::reclaim_retired_resources();
        safemlx::reclaim_allocation_owners();
        pool.unquoted_owner_count().unwrap() == owners
    });
}

#[test]
fn exchanged_decoder_state_retains_its_independent_operation_domain() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let model_pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let operation_pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let loader = MlxBackend::new(&stream, &stream).with_memory_pool(model_pool.clone());
    let root = crate::composition::mlx::replicated_text::tests::tiny_artifact("llama", true);
    let model =
        eredu_core::load_model(&loader, root.path(), crate::MlxLoadRequest::default()).unwrap();
    let backend = MlxBackend::new(&stream, &stream).with_memory_pool(operation_pool.clone());
    let mut runtime = ModelRuntime::from_prepared(backend, model).unwrap();
    let empty_state = runtime.session().payload.model.erased().state_snapshot();
    assert!(empty_state.iter().all(|(position, _)| *position == 0));

    // This guard belongs to the empty copy. Exchange moves it into the session,
    // so it cannot hide a missing operation guard on the displaced state.
    let mut slot = MlxBackend::capture_native_text_state(&mut runtime).unwrap();
    settle(&operation_pool, 1);
    let prompt = MlxBackend::prepare_text_prompt(&loader, vec![1, 2, 3]).unwrap();
    let prompt_identity = prompt.shared_cache_identity().unwrap().clone();
    let output = runtime.prefill(prompt).unwrap().wait().unwrap();
    let values = output
        .logits()
        .unwrap()
        .as_array()
        .evaluated()
        .unwrap()
        .as_slice::<f32>()
        .to_vec();
    assert!(values.iter().any(|value| value.abs() > 1e-6));
    let populated_state = runtime.session().payload.model.erased().state_snapshot();
    assert!(populated_state.iter().any(|(position, _)| *position == 3));
    settle(&operation_pool, 2);

    MlxBackend::exchange_native_text_state(&mut runtime, &mut slot).unwrap();
    assert_eq!(
        runtime.session().payload.model.erased().state_snapshot(),
        empty_state
    );
    let payload_retired = runtime.session().test_payload_retirement_probe();
    drop((output, runtime, loader));
    stream.synchronize().unwrap();
    crate::backend::submission_recovery::wait_for_retirement(|| {
        crate::backend::nn::shared::MlxNeuralBackend::reclaim_retired_resources();
        safemlx::reclaim_allocation_owners();
        payload_retired() && operation_pool.unquoted_owner_count().unwrap() == 1
    });
    // Displaced state retains its original model authority, and the shared
    // committed prompt identity retains its independent preparation authority.
    assert_eq!(model_pool.unquoted_owner_count().unwrap(), 2);
    drop(slot);
    settle(&operation_pool, 0);
    settle(&model_pool, 1);
    assert_eq!(
        prompt_identity.prepared().parts()[0].payload().shape(),
        [1, 3]
    );
    drop(prompt_identity);
    settle(&model_pool, 0);
}
