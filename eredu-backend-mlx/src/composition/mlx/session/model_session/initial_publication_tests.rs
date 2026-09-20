use super::*;
use crate::tests::support::path_instrumentation as paths;
use eredu_core::{
    Admission, EstimationCompleteness, ExecutionWorkspaceEstimate, InferenceGeometry,
    InputTokenCount, LayerSchedule, OutputDemand, StateMemoryLayout, WorkspaceBound,
};
use eredu_runtime::working_memory::{
    InferenceExecutionIdentity, WorkingMemoryError, WorkingMemoryPool,
};

fn fixture(
    scaled_rotary: bool,
) -> (
    Stream,
    WorkingMemoryPool,
    MlxBackend<'static>,
    eredu_core::PreparedModel<MlxModel>,
    tempfile::TempDir,
) {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let backend = MlxBackend::new(&stream, &stream).with_memory_pool(pool.clone());
    let root = crate::composition::mlx::replicated_text::tests::tiny_artifact("llama", true);
    if scaled_rotary {
        let path = root.path().join("config.json");
        let mut config: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        config["rope_scaling"] = serde_json::json!({
            "rope_type": "llama3",
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 32
        });
        std::fs::write(path, serde_json::to_vec(&config).unwrap()).unwrap();
    }
    let model =
        eredu_core::load_model(&backend, root.path(), crate::MlxLoadRequest::default()).unwrap();
    (stream, pool, backend, model, root)
}

// This zero-byte synthetic admission tests the domain exclusion protocol only.
// It makes no claim about a bound for future native inference operations.
fn zero_admission() -> Admission {
    let geometry = InferenceGeometry {
        batch_size: 1,
        cached_positions: 0,
        input_positions: 1,
        max_output_tokens: 0,
        prefill_chunk_positions: 1,
        output: OutputDemand::StateOnly,
    };
    let layout = StateMemoryLayout::new(
        LayerSchedule::new(1, vec![eredu_core::cache::LayerCachePolicy::NoState]).unwrap(),
        vec![0],
        1,
        1,
        EstimationCompleteness::Complete,
    )
    .unwrap();
    let zero = || WorkspaceBound::bounded(0, "initial storage publication ownership fixture");
    let state = eredu_core::estimate_runtime_state(
        &layout,
        InputTokenCount::text(1),
        0,
        1,
        std::num::NonZeroU8::new(4).unwrap(),
    )
    .unwrap()
    .with_execution_workspace(ExecutionWorkspaceEstimate {
        geometry,
        activations: zero(),
        attention: zero(),
        vocabulary: zero(),
        state_update: zero(),
        materialization: zero(),
        retained: zero(),
    })
    .unwrap();
    Admission {
        requested_positions: 1,
        state,
        incremental_required_bytes: 0,
        available_memory_bytes: None,
    }
}

fn reclaim() {
    crate::backend::nn::shared::MlxNeuralBackend::reclaim_retired_resources();
    safemlx::reclaim_allocation_owners();
}

fn settle(stream: &Stream, pool: &WorkingMemoryPool, owners: usize) {
    stream.synchronize().unwrap();
    crate::backend::submission_recovery::wait_for_retirement(|| {
        reclaim();
        pool.unquoted_owner_count().unwrap() == owners
    });
}

fn fully_retired(stream: &Stream, pool: &WorkingMemoryPool) {
    stream.synchronize().unwrap();
    crate::backend::submission_recovery::wait_for_retirement(|| {
        reclaim();
        pool.unquoted_owner_count().unwrap() == 0 && pool.used_bytes().unwrap() == 0
    });
}

fn assert_excluded(pool: &WorkingMemoryPool) {
    assert!(matches!(
        pool.reserve(&InferenceExecutionIdentity::default(), &zero_admission()),
        Err(WorkingMemoryError::UnknownBound)
    ));
}

fn assert_reserved(error: &(dyn std::error::Error + 'static)) {
    let mut current = error;
    loop {
        if let Some(memory) = current.downcast_ref::<WorkingMemoryError>() {
            assert_eq!(*memory, WorkingMemoryError::ReservedWorkActive);
            return;
        }
        current = current.source().expect("preserve typed domain rejection");
    }
}

#[test]
fn complete_fresh_session_publishes_exact_storage_and_releases_loading_authority() {
    let (stream, pool, backend, model, _root) = fixture(false);
    assert!(model.memory_owner().is_some());
    assert_eq!(pool.unquoted_owner_count().unwrap(), 1);
    let mut runtime = ModelRuntime::from_prepared(backend, model).unwrap();
    settle(&stream, &pool, 0);
    assert!(runtime.session().payload._memory_owner.is_none());
    assert!(!runtime
        .session()
        .payload
        .operation_memory
        .borrow()
        .covers_pool(&pool));
    let idle = runtime.session().payload.retained_idle_storage().unwrap();
    let bytes = idle.nonstate_bytes().unwrap().unwrap();
    assert!(bytes > 0);
    assert!(idle.has_empty_decoder_storage().unwrap());
    // The pool also retains source-metadata estimates beyond physical storage.
    let charged = pool.used_bytes().unwrap();
    assert!(charged >= bytes);
    let frontier = runtime.session().payload.model.erased().state_snapshot();
    assert!(frontier.iter().all(|(position, _)| *position == 0));
    let before = paths::snapshot();
    assert!(!runtime
        .session_mut()
        .publish_initial_idle_storage()
        .unwrap());
    assert_eq!(paths::snapshot(), before);
    assert_eq!(pool.used_bytes().unwrap(), charged);
    assert_eq!(
        runtime.session().payload.model.erased().state_snapshot(),
        frontier
    );
    assert!(matches!(
        pool.reserve_with_capacity(
            &InferenceExecutionIdentity::default(),
            &zero_admission(),
            charged - 1,
        ),
        Err(WorkingMemoryError::CapacityBelowUsage { capacity_bytes, used_bytes })
            if capacity_bytes == charged - 1 && used_bytes == charged
    ));
    assert_eq!(pool.used_bytes().unwrap(), charged);
    let reservation = pool
        .reserve_with_capacity(
            &InferenceExecutionIdentity::default(),
            &zero_admission(),
            charged,
        )
        .unwrap();
    assert_eq!(pool.used_bytes().unwrap(), charged);
    drop((reservation, idle, runtime));
    fully_retired(&stream, &pool);
}

#[test]
fn preexisting_model_owner_clone_remains_excluding_after_local_publication() {
    let (stream, pool, backend, model, _root) = fixture(false);
    let escaped_loading_owner = model.memory_owner().unwrap().clone();
    let runtime = ModelRuntime::from_prepared(backend, model).unwrap();
    settle(&stream, &pool, 1);
    assert!(runtime.session().payload._memory_owner.is_none());
    let idle = runtime.session().payload.retained_idle_storage().unwrap();
    let bytes = idle.nonstate_bytes().unwrap().unwrap();
    assert!(bytes > 0);
    let charged = pool.used_bytes().unwrap();
    assert!(charged >= bytes);
    assert_excluded(&pool);
    drop(escaped_loading_owner);
    settle(&stream, &pool, 0);
    let reservation = pool
        .reserve(&InferenceExecutionIdentity::default(), &zero_admission())
        .unwrap();
    assert_eq!(pool.used_bytes().unwrap(), charged);
    drop((reservation, idle, runtime));
    fully_retired(&stream, &pool);
}

#[test]
fn ordinary_decode_reacquires_authority_retained_by_escaped_output_backing() {
    let (stream, pool, backend, model, _root) = fixture(false);
    let operation_backend = MlxBackend::new(&stream, &stream).with_memory_pool(pool.clone());
    let mut runtime = ModelRuntime::from_prepared(backend, model).unwrap();
    settle(&stream, &pool, 0);
    let output = runtime
        .session_mut()
        .submit_token_decode(&operation_backend, 2)
        .unwrap()
        .wait()
        .unwrap();
    let escaped = output.into_logits().unwrap().into_array();
    let values = escaped.evaluated().unwrap().try_to_vec::<f32>().unwrap();
    assert!(values.iter().all(|value| value.is_finite()));
    assert!(values.iter().any(|value| value.abs() > 1e-6));
    let allocation = escaped.allocation_info().unwrap().unwrap();
    let alias = escaped.clone();
    let view = escaped.as_strided(&[4][..], &[2][..], 1, &stream).unwrap();
    view.evaluated().unwrap();
    assert_eq!(view.allocation_info().unwrap(), Some(allocation));
    assert_eq!(pool.unquoted_owner_count().unwrap(), 1);
    assert_excluded(&pool);
    let retired = runtime.session().test_payload_retirement_probe();
    drop((runtime, operation_backend, escaped));
    crate::backend::submission_recovery::wait_for_retirement(|| {
        reclaim();
        retired() && pool.unquoted_owner_count().unwrap() == 1
    });
    drop(alias);
    settle(&stream, &pool, 1);
    assert_eq!(
        view.evaluated().unwrap().try_to_vec::<f32>().unwrap(),
        values
            .into_iter()
            .skip(1)
            .step_by(2)
            .take(4)
            .collect::<Vec<_>>()
    );
    assert_excluded(&pool);
    drop(view);
    fully_retired(&stream, &pool);
}

#[test]
fn live_reservation_blocks_reset_and_decode_input_before_native_work() {
    let (stream, pool, backend, model, _root) = fixture(false);
    let operation_backend = MlxBackend::new(&stream, &stream).with_memory_pool(pool.clone());
    let mut runtime = ModelRuntime::from_prepared(backend, model).unwrap();
    settle(&stream, &pool, 0);
    let reservation = pool
        .reserve(&InferenceExecutionIdentity::default(), &zero_admission())
        .unwrap();
    let before = paths::snapshot();
    let resets = paths::session_reset_attempts();
    let inputs = paths::session_input_creation_attempts();
    let bytes = pool.used_bytes().unwrap();
    let peak = pool.peak_bytes().unwrap();
    let frontier = runtime.session().payload.model.erased().state_snapshot();
    assert_reserved(&runtime.session_mut().reset().unwrap_err());
    assert_reserved(
        &runtime
            .session_mut()
            .submit_token_decode(&operation_backend, 3)
            .err()
            .unwrap(),
    );
    let called = Cell::new(false);
    assert_reserved(
        &runtime
            .session_mut()
            .submit_decode_input(&operation_backend, || {
                called.set(true);
                Ok(Array::from_slice(&[3_u32], &[1, 1]))
            })
            .err()
            .unwrap(),
    );
    assert!(!called.get());
    assert_eq!(paths::snapshot(), before);
    assert_eq!(paths::session_reset_attempts(), resets);
    assert_eq!(paths::session_input_creation_attempts(), inputs);
    assert_eq!(
        runtime.session().payload.model.erased().state_snapshot(),
        frontier
    );
    assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
    assert_eq!(pool.used_bytes().unwrap(), bytes);
    assert_eq!(pool.peak_bytes().unwrap(), peak);
    assert!(runtime.session().ensure_no_submission_in_flight().is_ok());
    drop((reservation, runtime));
    fully_retired(&stream, &pool);
}

#[test]
fn loaded_scaled_rotary_has_complete_initial_publication_without_inventory_work() {
    let (stream, pool, backend, model, _root) = fixture(true);
    let mut runtime = ModelRuntime::from_prepared(backend, model).unwrap();
    settle(&stream, &pool, 0);
    assert!(runtime.session().payload._memory_owner.is_none());
    let idle = runtime.session().payload.retained_idle_storage().unwrap();
    assert!(idle.nonstate_bytes().unwrap().is_some());
    assert!(idle.has_empty_decoder_storage().unwrap());
    let (nonstate, decoder) = idle.into_parts();
    assert!(nonstate.unknown_arrays().is_empty());
    let facts = nonstate.array_allocation_facts();
    let before = paths::snapshot();
    let frontier = runtime.session().payload.model.erased().state_snapshot();
    let bytes = pool.used_bytes().unwrap();
    assert!(!runtime
        .session_mut()
        .publish_initial_idle_storage()
        .unwrap());
    let repeated = runtime.session().payload.retained_idle_storage().unwrap();
    let (repeated_nonstate, repeated_decoder) = repeated.into_parts();
    assert_eq!(repeated_nonstate.array_allocation_facts(), facts);
    assert!(repeated_nonstate.unknown_arrays().is_empty());
    assert_eq!(paths::snapshot(), before);
    assert_eq!(
        runtime.session().payload.model.erased().state_snapshot(),
        frontier
    );
    assert_eq!(pool.used_bytes().unwrap(), bytes);
    assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
    let reservation = pool
        .reserve(&InferenceExecutionIdentity::default(), &zero_admission())
        .unwrap();
    drop((
        reservation,
        repeated_nonstate,
        repeated_decoder,
        nonstate,
        decoder,
        runtime,
    ));
    fully_retired(&stream, &pool);
}

#[test]
fn nonempty_decoder_prevents_publication_even_with_certified_nonstate_storage() {
    let (stream, pool, backend, mut model, _root) = fixture(true);
    // Exercise the actual publication gate before session construction, whose
    // mandatory reset intentionally discards preexisting decoder state. This
    // plain fixture has no processor, communication or parameter-overlay payload.
    let memory = model.memory_owner().unwrap().clone();
    let output = crate::backend::submission_recovery::detached_retained(memory.clone(), || {
        let token = Array::from_slice(&[2_u32], &[1, 1]);
        let executable = model.executable_mut();
        let output = executable.erased_mut().decode(&token, &stream)?;
        let roots = executable
            .erased()
            .retained_decoder_state_storage()?
            .into_retained_arrays()
            .map_err(|(cause, _storage)| cause)?;
        crate::backend::MlxCompletion::submission_retaining(output, roots)?.wait()
    })
    .unwrap();
    assert!(output
        .evaluated()
        .unwrap()
        .as_slice::<f32>()
        .iter()
        .any(|value| value.abs() > 1e-6));
    drop(output);
    settle(&stream, &pool, 1);
    let executable = model.executable_mut();
    let idle = executable
        .retained_idle_storage(Some(Default::default()))
        .unwrap();
    assert!(idle.nonstate_bytes().unwrap().unwrap() > 0);
    assert!(!idle.has_empty_decoder_storage().unwrap());
    let frontier = executable.erased().state_snapshot();
    assert!(frontier.iter().any(|(position, _)| *position > 0));
    let before = paths::snapshot();
    let owners = pool.unquoted_owner_count().unwrap();
    let bytes = pool.used_bytes().unwrap();
    assert!(!executable
        .publish_initial_idle_storage(Default::default(), &memory)
        .unwrap());
    assert!(!executable.has_published_idle_storage());
    assert_eq!(paths::snapshot(), before);
    assert_eq!(executable.erased().state_snapshot(), frontier);
    assert_eq!(pool.unquoted_owner_count().unwrap(), owners);
    assert_eq!(pool.used_bytes().unwrap(), bytes);
    assert!(model.memory_owner().is_some());
    assert_excluded(&pool);
    drop((idle, model, memory, backend));
    fully_retired(&stream, &pool);
}
