use super::*;
use crate::tests::support::path_instrumentation as paths;
use eredu_core::{
    Admission, EstimationCompleteness, ExecutionWorkspaceEstimate, InferenceGeometry,
    InputTokenCount, LayerSchedule, OutputDemand, StateMemoryLayout, WorkspaceBound,
};
use eredu_runtime::working_memory::{InferenceExecutionIdentity, MemoryLedger, WorkingMemoryError};

fn fixture(
    scaled_rotary: bool,
) -> (
    Stream,
    MemoryLedger,
    MlxBackend<'static>,
    eredu_core::PreparedModel<MlxModel>,
    tempfile::TempDir,
) {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
    let backend = MlxBackend::new(&stream, &stream).with_memory_ledger(pool.clone());
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
    .with_execution_workspace(crate::memory_fixture::workspace(
        ExecutionWorkspaceEstimate {
            physical_domains: None,
            geometry,
            activations: zero(),
            attention: zero(),
            vocabulary: zero(),
            state_update: zero(),
            materialization: zero(),
            retained: zero(),
        },
    ))
    .unwrap();
    crate::memory_fixture::admission(Admission {
        additional_headroom: Default::default(),
        memory_limits: Default::default(),
        requested_positions: 1,
        state,
        incremental_required_bytes: Some(0),
    })
}

fn reclaim() {
    crate::backend::nn::shared::MlxNeuralBackend::reclaim_retired_resources();
    safemlx::reclaim_allocation_owners();
}

fn settle(stream: &Stream, pool: &MemoryLedger, owners: usize) {
    stream.synchronize().unwrap();
    crate::backend::submission_recovery::wait_for_retirement(|| {
        reclaim();
        pool.unquoted_owner_count().unwrap() == owners
    });
}

fn fully_retired(stream: &Stream, pool: &MemoryLedger) {
    stream.synchronize().unwrap();
    crate::backend::submission_recovery::wait_for_retirement(|| {
        reclaim();
        safemlx::memory::clear_cache().unwrap();
        reclaim();
        pool.unquoted_owner_count().unwrap() == 0 && pool.fixture_host_charge().unwrap() == 0
    });
}

fn assert_excluded(pool: &MemoryLedger) {
    assert!(matches!(
        pool.reserve(&InferenceExecutionIdentity::default(), &zero_admission()),
        Err(WorkingMemoryError::UnknownBound)
    ));
}

fn assert_memory(error: &(dyn std::error::Error + 'static), expected: WorkingMemoryError) {
    let mut current = error;
    loop {
        if let Some(memory) = current.downcast_ref::<WorkingMemoryError>() {
            assert_eq!(*memory, expected);
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
    let charged = pool.fixture_host_charge().unwrap();
    assert!(charged >= bytes);
    let frontier = runtime.session().payload.model.erased().state_snapshot();
    assert!(frontier.iter().all(|(position, _)| *position == 0));
    let before = paths::snapshot();
    assert!(!runtime
        .session_mut()
        .publish_initial_idle_storage()
        .unwrap());
    assert_eq!(paths::snapshot(), before);
    assert_eq!(pool.fixture_host_charge().unwrap(), charged);
    assert_eq!(
        runtime.session().payload.model.erased().state_snapshot(),
        frontier
    );
    let admission = zero_admission();
    let existing = pool.fixture_host_current().unwrap();
    let requested = pool
        .reservation_requirements(&admission, None)
        .unwrap()
        .get(pool.topology().host_domain())
        .unwrap()
        .total()
        .unwrap();
    let exact = existing.checked_add(requested).unwrap();
    let before_rejection = pool.snapshot().unwrap();
    assert!(matches!(
        pool.reserve_with_capacity(
            &InferenceExecutionIdentity::default(), &admission,
            crate::memory_fixture::physical_host_limits(&pool, exact - 1),
        ),
        Err(WorkingMemoryError::Domain(eredu_core::MemoryDomainError::BudgetExceeded {
            domain, limit_bytes, existing_bytes, requested_bytes,
        })) if domain == pool.topology().host_domain() && limit_bytes == exact - 1
            && existing_bytes == existing && requested_bytes == requested
    ));
    assert_eq!(pool.snapshot().unwrap(), before_rejection);
    let reservation = pool
        .reserve_with_capacity(
            &InferenceExecutionIdentity::default(),
            &admission,
            crate::memory_fixture::physical_host_limits(&pool, exact),
        )
        .unwrap();
    assert_eq!(pool.fixture_host_charge().unwrap(), charged);
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
    let charged = pool.fixture_host_charge().unwrap();
    assert!(charged >= bytes);
    assert_excluded(&pool);
    drop(escaped_loading_owner);
    settle(&stream, &pool, 0);
    let reservation = pool
        .reserve(&InferenceExecutionIdentity::default(), &zero_admission())
        .unwrap();
    assert_eq!(pool.fixture_host_charge().unwrap(), charged);
    drop((reservation, idle, runtime));
    fully_retired(&stream, &pool);
}

#[test]
fn raw_decode_without_request_bounds_rejects_before_native_work() {
    let (stream, pool, backend, model, _root) = fixture(false);
    let operation_backend = MlxBackend::new(&stream, &stream).with_memory_ledger(pool.clone());
    let mut runtime = ModelRuntime::from_prepared(backend, model).unwrap();
    settle(&stream, &pool, 0);
    let before = pool.snapshot().unwrap();
    let frontier = runtime.session().payload.model.erased().state_snapshot();
    let inputs = paths::session_input_creation_attempts();
    assert_memory(
        &runtime
            .session_mut()
            .submit_token_decode(&operation_backend, 2)
            .err()
            .expect("a raw token grants no finite output or workspace evidence"),
        WorkingMemoryError::UnknownBound,
    );
    assert_eq!(pool.snapshot().unwrap(), before);
    assert_eq!(
        runtime.session().payload.model.erased().state_snapshot(),
        frontier
    );
    assert_eq!(paths::session_input_creation_attempts(), inputs);
    drop((runtime, operation_backend));
    fully_retired(&stream, &pool);
}

#[test]
fn live_reservation_blocks_reset_and_decode_input_before_native_work() {
    let (stream, pool, backend, model, _root) = fixture(false);
    let operation_backend = MlxBackend::new(&stream, &stream).with_memory_ledger(pool.clone());
    let mut runtime = ModelRuntime::from_prepared(backend, model).unwrap();
    settle(&stream, &pool, 0);
    let reservation = pool
        .reserve(&InferenceExecutionIdentity::default(), &zero_admission())
        .unwrap();
    let before = paths::snapshot();
    let resets = paths::session_reset_attempts();
    let inputs = paths::session_input_creation_attempts();
    let bytes = pool.fixture_host_charge().unwrap();
    let peak = pool.fixture_host_peak().unwrap();
    let frontier = runtime.session().payload.model.erased().state_snapshot();
    assert_memory(
        &runtime.session_mut().reset().unwrap_err(),
        WorkingMemoryError::ReservedWorkActive,
    );
    assert_memory(
        &runtime
            .session_mut()
            .submit_token_decode(&operation_backend, 3)
            .err()
            .unwrap(),
        WorkingMemoryError::UnknownBound,
    );
    let called = Cell::new(false);
    assert_memory(
        &runtime
            .session_mut()
            .submit_decode_input(&operation_backend, || {
                called.set(true);
                Ok(Array::from_slice(&[3_u32], &[1, 1]))
            })
            .err()
            .unwrap(),
        WorkingMemoryError::UnknownBound,
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
    assert_eq!(pool.fixture_host_charge().unwrap(), bytes);
    assert_eq!(pool.fixture_host_peak().unwrap(), peak);
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
    let bytes = pool.fixture_host_charge().unwrap();
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
    assert_eq!(pool.fixture_host_charge().unwrap(), bytes);
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
    let bytes = pool.fixture_host_charge().unwrap();
    assert!(!executable
        .publish_initial_idle_storage(Default::default(), &memory)
        .unwrap());
    assert!(!executable.has_published_idle_storage());
    assert_eq!(paths::snapshot(), before);
    assert_eq!(executable.erased().state_snapshot(), frontier);
    assert_eq!(pool.unquoted_owner_count().unwrap(), owners);
    assert_eq!(pool.fixture_host_charge().unwrap(), bytes);
    assert!(model.memory_owner().is_some());
    assert_excluded(&pool);
    drop((idle, model, memory, backend));
    fully_retired(&stream, &pool);
}

#[cfg(test)]
#[allow(unused_imports)]
use crate::memory_fixture::LedgerFixture;
